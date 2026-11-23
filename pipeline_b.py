import json
import time
from pathlib import Path
from typing import Callable, List, Literal, Optional

from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field

from schema import (
    FarmerProfile,
    RegulatoryCorpus,
    RegulatoryRule,
    RoiHandoffImpact,
)
from dotenv import load_dotenv

load_dotenv()  # Automatically loads variables from .env into os.environ

import threading

class AssessmentCancelled(Exception):
    """Raised when the user cancels while Gemini is being retried."""

RETRYABLE_CODES = {500, 503, 504}   # overloaded / unavailable: retry until cancel or time limit
RATE_LIMIT_CODE = 429               # may be a daily quota, so only a few rounds
MAX_RATE_LIMIT_ROUNDS = 2
BACKOFF_START_S = 5
BACKOFF_MAX_S = 30
MAX_RETRY_SECONDS = 600             # stop after 10 minutes even if nobody cancels

GRADE_TO_SCORE = {
    "A": 95,
    "B": 80,
    "C": 65,
    "D": 45,
    "F": 15,
}

# Ordered from newest Flash model to fast fallback Flash models
CANDIDATE_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

FEDERAL_OSHA_STATES = {"IL", "TX"}

# --- 1. Gemini Structured Output Schema ---
class GeminiRegulatoryAssessment(BaseModel):
    feasibility: Literal["PERMITTED", "CONDITIONAL", "PROHIBITED"] = Field(
        description="Overall legal feasibility based strictly on the provided verified rules."
    )
    letter_grade: Literal["A", "B", "C", "D", "F"] = Field(
        description=(
            "A = Fully permitted in-field and on-road (if applicable) with minimal friction; "
            "B = Permitted in-field with minor operational conditions (e.g., perimeter exclusion or manual road transport); "
            "C = Permitted only with significant supervision/transport workarounds; "
            "D = Heavily restricted for the selected mode; "
            "F = Prohibited under current enacted law."
        )
    )
    headline_summary: str = Field(
        description="One-sentence plain-language verdict summarizing both in-field and road status."
    )
    in_field_assessment: str = Field(
        description=(
            "Dedicated 2-3 sentence assessment of autonomous operation INSIDE field boundaries, "
            "citing applicable workplace/OSHA/Cal-OSHA rules and employee exposure rules."
        )
    )
    public_road_assessment: str = Field(
        description=(
            "Dedicated 2-3 sentence assessment of autonomous operation ON PUBLIC ROADS between fields. "
            "If `public_road_transit` is False, explicitly state that no public road transit was requested "
            "and note that public road autonomy would require separate DMV/Vehicle Code compliance."
        )
    )
    applicable_citations: List[str] = Field(
        description="Exact statutory/regulatory citations from the provided rules that govern this operation."
    )
    plain_language_caveats: List[str] = Field(
        description="Clear, farmer-friendly bullet points prefixed with '[In-Field]' or '[Public Road]'."
    )
    recommended_workaround: str = Field(
        description="Concrete operational adjustment the farmer can make to comply or improve their grade."
    )


# --- 2. Final Output Contract Returned to Tkinter & Composite Scorer ---
class PipelineBResult(BaseModel):
    feasibility: Literal["PERMITTED", "CONDITIONAL", "PROHIBITED"]
    letter_grade: Literal["A", "B", "C", "D", "F"]
    numeric_score: int
    weighted_contribution: float
    osha_small_farm_exempt: bool
    model_used: str
    elapsed_seconds: float
    headline_summary: str
    in_field_assessment: str
    public_road_assessment: str
    applicable_citations: List[str]
    plain_language_caveats: List[str]
    recommended_workaround: str
    roi_handoff: RoiHandoffImpact


# --- 3. Deterministic Corpus Loader & Pre-Filter ---
def load_corpus(corpus_path: str | Path = "regulatory_corpus.json") -> RegulatoryCorpus:
    raw_data = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    return RegulatoryCorpus.model_validate(raw_data)


def is_osha_small_farm_exempt(farmer: FarmerProfile) -> bool:
    return (
        farmer.state in FEDERAL_OSHA_STATES
        and farmer.non_family_employees <= 10
        and not farmer.temporary_labor_camp
    )


def filter_applicable_rules(
    farmer: FarmerProfile,
    corpus: RegulatoryCorpus,
) -> tuple[List[RegulatoryRule], bool, RoiHandoffImpact]:
    exempt = is_osha_small_farm_exempt(farmer)
    allowed_jurisdictions = {farmer.state}
    if farmer.state in FEDERAL_OSHA_STATES:
        allowed_jurisdictions.add("FEDERAL")

    matched_rules: List[RegulatoryRule] = []
    max_supervision_override: Optional[int] = None
    requires_road_driver = False

    for rule in corpus.rules:
        # Gate 1: Only human-verified enacted laws or official agency guidance govern runtime scoring
        if not rule.verified or rule.legal_status not in ("ENACTED", "AGENCY_GUIDANCE"):
            continue
        if rule.jurisdiction not in allowed_jurisdictions:
            continue

        cond = rule.applicability

        # Gate 3: Federal OSHA small-farm enforcement exemption (IL only)
        if cond.requires_osha_enforcement and exempt:
            continue

        # Gate 4: Activity & equipment match
        if "ALL" not in cond.activities and farmer.activity not in cond.activities:
            continue
        if "ALL" not in cond.equipment_types and farmer.equipment_type not in cond.equipment_types:
            continue

        # Gate 5: Public road transit trigger
        if cond.requires_public_road_transit is True and not farmer.public_road_transit:
            continue

        # Gate 6: Employee zone exposure trigger (Cal/OSHA § 2.2 OSHAB doctrine)
        if cond.requires_employee_exposure is not None:
            if cond.requires_employee_exposure != farmer.employee_zone_exposure:
                continue

        matched_rules.append(rule)

        if farmer.supervision_mode in cond.blocked_supervision_modes:
            sup_pct = rule.roi_impact.min_supervision_pct_override
            if sup_pct is not None:
                max_supervision_override = max(max_supervision_override or 0, sup_pct)
            if rule.roi_impact.requires_human_road_transport:
                requires_road_driver = True

    roi_handoff = RoiHandoffImpact(
        min_supervision_pct_override=max_supervision_override,
        requires_human_road_transport=requires_road_driver,
    )
    return matched_rules, exempt, roi_handoff


# --- 4. Gemini API Evaluation ---
SYSTEM_INSTRUCTION = """
You are the regulatory compliance evaluator for RateOnomy, an agricultural autonomy decision-support tool.
Evaluate the farmer's proposed operation strictly using ONLY the verified, enacted legal rules or agency guidance provided in the prompt.

Rules for your evaluation:
1. Do NOT cite, infer, or rely on any external laws, pending petitions, or assumptions outside the provided rules.
2. Always separate your analysis into two distinct domains:
   - `in_field_assessment`: Legal status of autonomous operation strictly within private field boundaries (governed by OSHA / Cal/OSHA rules and employee exposure).
   - `public_road_assessment`: Legal status of autonomous transit on public highways/roads between fields (governed by Vehicle Code / DMV Autonomous Vehicle rules).
3. Check whether the farmer's `supervision_mode` appears in a rule's `blocked_supervision_modes`. If any triggered rule blocks their supervision mode for either in-field or public road transit, the overall operation cannot be rated 'PERMITTED' or given an 'A'.
4. Prefix each item in `plain_language_caveats` with either `[In-Field]` or `[Public Road]` so the farmer sees which rule applies where.
5. Keep explanations concise, practical, and free of legalese.
"""


def evaluate_regulatory_compliance(
    farmer: FarmerProfile,
    corpus_path: str | Path = "regulatory_corpus.json",
    legal_weight: float = 0.5,
    status_callback: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> PipelineBResult:
    start_time = time.perf_counter()

    def report(msg: str) -> None:
        if status_callback:
            status_callback(msg)

    report("Loading and validating regulatory corpus...")
    corpus = load_corpus(corpus_path)

    report("Filtering enacted rules & checking OSHA small-farm exemption...")
    applicable_rules, exempt, roi_handoff = filter_applicable_rules(farmer, corpus)
    report(
        f"Matched {len(applicable_rules)} verified rule(s) "
        f"[OSHA Exempt: {'Yes' if exempt else 'No'}]."
    )

    if not applicable_rules:
        score = GRADE_TO_SCORE["A"]
        elapsed = round(time.perf_counter() - start_time, 2)

        in_field_default = (
            f"Permitted within private field boundaries in {farmer.state}. "
            "This operation qualifies for the federal OSHA small-farm enforcement exemption "
            "(10 or fewer non-family employees and no temporary labor camp in the past 12 months), "
            "and no state workplace restrictions apply."
            if exempt
            else f"Permitted within private field boundaries in {farmer.state}. "
                 "No enacted state or federal restrictions in the verified corpus prohibit autonomous "
                 f"{farmer.activity} under '{farmer.supervision_mode}' supervision."
        )

        public_road_default = (
            "No public road transit was selected for this operation; autonomous equipment is assumed "
            "to remain inside field boundaries or be hauled via conventional transport. Note that "
            "operating autonomously on public highways would trigger separate state Vehicle Code rules."
            if not farmer.public_road_transit
            else f"No specific autonomous public-road prohibitions were triggered in the verified corpus "
                 f"for {farmer.state}, though standard slow-moving vehicle signage, width, and lighting laws still apply."
        )

        return PipelineBResult(
            feasibility="PERMITTED",
            letter_grade="A",
            numeric_score=score,
            weighted_contribution=round(score * legal_weight, 2),
            osha_small_farm_exempt=exempt,
            model_used="deterministic-only (0 rules triggered)",
            elapsed_seconds=elapsed,
            headline_summary="No enacted state or federal restrictions block this autonomous operation.",
            in_field_assessment=in_field_default,
            public_road_assessment=public_road_default,
            applicable_citations=[],
            plain_language_caveats=[
                "[In-Field] Operation qualifies for federal OSHA small-farm enforcement exemption."
                if exempt
                else "[In-Field] No autonomous field-operation prohibitions triggered in the verified corpus.",
                "[Public Road] Public road transit was not requested; equipment stays off public roadways."
                if not farmer.public_road_transit
                else "[Public Road] No specific autonomous transit prohibitions matched in the verified corpus.",
            ],
            recommended_workaround="Maintain standard manufacturer safety geofencing and owner's manual protocols.",
            roi_handoff=roi_handoff,
        )

    rules_payload = [
        {
            "rule_id": r.rule_id,
            "jurisdiction": r.jurisdiction,
            "category": r.category,
            "citation": r.citation,
            "title": r.title,
            "blocked_supervision_modes": r.applicability.blocked_supervision_modes,
            "constraint_summary": r.constraint_summary,
            "operational_workaround": r.operational_workaround,
        }
        for r in applicable_rules
    ]

    prompt = (
        f"FARMER OPERATION PROFILE:\n"
        f"{farmer.model_dump_json(indent=2)}\n\n"
        f"FEDERAL OSHA SMALL-FARM EXEMPT STATUS: {exempt}\n\n"
        f"APPLICABLE VERIFIED ENACTED RULES:\n"
        f"{json.dumps(rules_payload, indent=2)}"
    )

    # Cap each HTTP call at 20s and disable hidden SDK retries so failover is immediate
    client = genai.Client(
        http_options=types.HttpOptions(
            timeout=20_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        )
    )

    def check_cancel() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise AssessmentCancelled("Assessment cancelled.")

    response = None
    used_model = ""
    last_error: Optional[Exception] = None
    round_num = 0
    rate_limit_rounds = 0
    delay = BACKOFF_START_S
    deadline = time.monotonic() + MAX_RETRY_SECONDS

    while True:
        round_num += 1
        all_rate_limited = True

        for model_name in CANDIDATE_MODELS:
            check_cancel()
            try:
                report(f"Querying {model_name} (attempt {round_num})...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        response_mime_type="application/json",
                        response_schema=GeminiRegulatoryAssessment,
                        temperature=0.0,
                        thinking_config=types.ThinkingConfig(thinking_level="LOW"),
                    ),
                )
                used_model = model_name
                break
            except APIError as err:
                last_error = err
                if err.code == RATE_LIMIT_CODE:
                    report(f"{model_name} unavailable, trying next model...")
                    continue
                if err.code in RETRYABLE_CODES:
                    all_rate_limited = False
                    report(f"{model_name} busy (HTTP {err.code}), trying next model...")
                    continue
                raise  # 400/401/403 etc. won't fix themselves
            except Exception as err:  # timeouts / connection errors from whatever HTTP library the SDK uses
                last_error = err
                all_rate_limited = False
                report(f"{model_name} timed out, trying next model...")
                continue

        if response is not None:
            break

        # Every model failed this round
        if all_rate_limited:
            rate_limit_rounds += 1
            if rate_limit_rounds >= MAX_RATE_LIMIT_ROUNDS:
                raise RuntimeError(
                    "Gemini rate limit or quota reached. Try again later. "
                    f"Last error: {last_error}"
                )
        if time.monotonic() + delay > deadline:
            raise RuntimeError(
                f"Gemini still unavailable after {MAX_RETRY_SECONDS // 60} minutes. "
                f"Last error: {last_error}"
            )

        # Wait before the next round, counting down and checking for cancel every second
        for remaining in range(delay, 0, -1):
            check_cancel()
            report(f"Gemini busy, retrying in {remaining}s (attempt {round_num + 1})...")
            if cancel_event is not None:
                if cancel_event.wait(1):
                    raise AssessmentCancelled("Assessment cancelled.")
            else:
                time.sleep(1)
        delay = min(delay * 2, BACKOFF_MAX_S)

    report("Parsing structured regulatory assessment...")
    assessment: GeminiRegulatoryAssessment = response.parsed
    numeric_score = GRADE_TO_SCORE[assessment.letter_grade]
    elapsed = round(time.perf_counter() - start_time, 2)

    return PipelineBResult(
        feasibility=assessment.feasibility,
        letter_grade=assessment.letter_grade,
        numeric_score=numeric_score,
        weighted_contribution=round(numeric_score * legal_weight, 2),
        osha_small_farm_exempt=exempt,
        model_used=used_model,
        elapsed_seconds=elapsed,
        headline_summary=assessment.headline_summary,
        applicable_citations=assessment.applicable_citations,
        plain_language_caveats=assessment.plain_language_caveats,
        recommended_workaround=assessment.recommended_workaround,
        roi_handoff=roi_handoff,
        in_field_assessment=assessment.in_field_assessment,
        public_road_assessment=assessment.public_road_assessment,
    )