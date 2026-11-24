import json
import time
import os
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

GRADE_TO_SCORE = {
    "A": 95,
    "B": 80,
    "C": 70,
    "D": 60,
    "F": 15,
}

# Ordered from newest Flash model to fast fallback Flash models
CANDIDATE_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
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

def compute_deterministic_grade(
    farmer: FarmerProfile,
    matched_rules: List[RegulatoryRule],
) -> tuple[Literal["PERMITTED", "CONDITIONAL", "PROHIBITED"], Literal["A", "B", "C", "D", "F"]]:
    """Deterministically compute feasibility and letter grade from triggered rules."""
    in_field_blocked = any(
        r.category == "in_field_supervision"
        and farmer.supervision_mode in r.applicability.blocked_supervision_modes
        for r in matched_rules
    )
    road_blocked = farmer.public_road_transit and any(
        r.category == "public_road_transit"
        and farmer.supervision_mode in r.applicability.blocked_supervision_modes
        for r in matched_rules
    )
    has_conditional_guidance = any(
        r.legal_status == "AGENCY_GUIDANCE" for r in matched_rules
    )

    if in_field_blocked and road_blocked:
        return "PROHIBITED", "F"   # 15: Blocked both in-field and on public roads
    if in_field_blocked:
        return "PROHIBITED", "D"   # 45: Blocked in-field (core farm activity prohibited)
    if road_blocked and has_conditional_guidance:
        return "CONDITIONAL", "C"  # 65: In-field requires strict exclusion + road transit blocked
    if road_blocked or has_conditional_guidance:
        return "CONDITIONAL", "B"  # 80: Permitted with single operational workaround (e.g., manual road drive)
    return "PERMITTED", "A"        # 95: No enacted restrictions block the requested operation

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
) -> PipelineBResult:
    start_time = time.perf_counter()

    def report(msg: str) -> None:
        if status_callback:
            status_callback(msg)

    report("Loading and validating regulatory corpus...")
    corpus = load_corpus(corpus_path)

    report("Filtering enacted rules & checking OSHA small-farm exemption...")
    applicable_rules, exempt, roi_handoff = filter_applicable_rules(farmer, corpus)
    feasibility, letter_grade = compute_deterministic_grade(farmer, applicable_rules)
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
        f"LOCKED GRADE: {letter_grade} ({feasibility}) — explain why this grade applies.\n"
        f"{json.dumps(rules_payload, indent=2)}"
    )

    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip().strip("'\"")
    if not api_key or api_key == "NONE":
        raise RuntimeError(
            "GEMINI_API_KEY is missing or still set to 'NONE' in your .env file. "
            "Update .env next to app.py with a valid Google AI Studio key."
        )

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=30_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )

    response = None
    used_model = ""
    last_error: Optional[Exception] = None

    for model_name in CANDIDATE_MODELS:
        try:
            report(f"Querying {model_name} ({len(applicable_rules)} rules)...")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=GeminiRegulatoryAssessment,
                    temperature=0.0,
                ),
            )
            used_model = model_name
            break
        except APIError as err:
            last_error = err
            if err.code in (400,404,429, 500, 503, 504):
                report(f"{model_name} busy (HTTP {err.code}), failing over to next model...")
                continue
            raise
        except Exception as err:  # noqa: BLE001
            # Catch network/read timeouts and try the next model immediately
            last_error = err
            report(f"{model_name} timed out, failing over to next model...")
            continue

    numeric_score = GRADE_TO_SCORE[letter_grade]
    elapsed = round(time.perf_counter() - start_time, 2)

    def _build_local_fallback(reason: str) -> PipelineBResult:
        in_field_rules = [r for r in applicable_rules if r.category != "public_road_transit"]
        road_rules = [r for r in applicable_rules if r.category == "public_road_transit"]
        workarounds = [r.operational_workaround for r in applicable_rules if r.operational_workaround]

        return PipelineBResult(
            feasibility=feasibility,
            letter_grade=letter_grade,
            numeric_score=numeric_score,
            weighted_contribution=round(numeric_score * legal_weight, 2),
            osha_small_farm_exempt=exempt,
            model_used=f"deterministic-fallback ({reason})",
            elapsed_seconds=elapsed,
            headline_summary=f"Rated {letter_grade} ({feasibility}) under {farmer.state} and federal rules (local corpus fallback).",
            in_field_assessment=(
                " ".join(r.constraint_summary for r in in_field_rules)
                if in_field_rules
                else f"No enacted state or federal restrictions prohibit autonomous {farmer.activity} inside private fields in {farmer.state}."
            ),
            public_road_assessment=(
                "No public road transit was requested for this operation."
                if not farmer.public_road_transit
                else (
                    " ".join(r.constraint_summary for r in road_rules)
                    if road_rules
                    else f"No autonomous public-road prohibitions were triggered for {farmer.state}."
                )
            ),
            applicable_citations=[r.citation for r in applicable_rules],
            plain_language_caveats=[
                f"[{'Public Road' if r.category == 'public_road_transit' else 'In-Field'}] {r.constraint_summary}"
                for r in applicable_rules
            ],
            recommended_workaround=" ".join(
                workarounds) if workarounds else "Follow standard manufacturer safety protocols.",
            roi_handoff=roi_handoff,
        )

    # 1. Fallback if all cloud LLM endpoints timed out or were overloaded
    if response is None:
        return _build_local_fallback("API timeout")

    # 2. Parse Gemini's structured output (with backup raw-text JSON parse)
    report("Parsing structured regulatory assessment...")
    assessment: Optional[GeminiRegulatoryAssessment] = getattr(response, "parsed", None)

    if assessment is None and getattr(response, "text", None):
        raw_text = (
            response.text.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            assessment = GeminiRegulatoryAssessment.model_validate_json(raw_text)
        except Exception:  # noqa: BLE001
            assessment = None

    # 3. Fallback if the model returned empty or unparseable output
    if assessment is None:
        return _build_local_fallback(f"{used_model} empty response")

    # 4. Normal return when Gemini succeeds
    return PipelineBResult(
        feasibility=feasibility,
        letter_grade=letter_grade,
        numeric_score=numeric_score,
        weighted_contribution=round(numeric_score * legal_weight, 2),
        osha_small_farm_exempt=exempt,
        model_used=used_model,
        elapsed_seconds=elapsed,
        headline_summary=assessment.headline_summary,
        in_field_assessment=assessment.in_field_assessment,
        public_road_assessment=assessment.public_road_assessment,
        applicable_citations=assessment.applicable_citations,
        plain_language_caveats=assessment.plain_language_caveats,
        recommended_workaround=assessment.recommended_workaround,
        roi_handoff=roi_handoff,
    )