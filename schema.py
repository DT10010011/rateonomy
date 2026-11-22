from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# --- 1. Farmer Input Contract (8 Inputs Collected from Tkinter UI) ---
class FarmerProfile(BaseModel):
    state: Literal["IL", "CA", "IA", "TX", "AZ"]
    activity: Literal["tillage", "planting", "spraying", "harvesting"]
    equipment_type: Literal["autonomous_tractor", "retrofit_kit"]
    public_road_transit: bool
    supervision_mode: Literal[
        "on_site_line_of_sight",
        "remote_monitoring",
        "fully_unattended",
    ]
    non_family_employees: int = Field(ge=0)
    temporary_labor_camp: bool
    employee_zone_exposure: bool = Field(
        description="True if employees are present at the worksite or have predictable access to the operational zone of danger."
    )


# --- 2. Regulatory Corpus Record Contract ---
class ApplicabilityConditions(BaseModel):
    """Deterministic filter criteria evaluated by Python before calling Gemini."""
    activities: List[Literal["tillage", "planting", "spraying", "harvesting", "ALL"]] = ["ALL"]
    equipment_types: List[Literal["autonomous_tractor", "retrofit_kit", "ALL"]] = ["ALL"]
    requires_public_road_transit: Optional[bool] = None
    requires_employee_exposure: Optional[bool] = None  # True = workers exposed; False = zero worker access; None = both
    blocked_supervision_modes: List[
        Literal["on_site_line_of_sight", "remote_monitoring", "fully_unattended"]
    ] = []
    requires_osha_enforcement: bool = False


class RoiHandoffImpact(BaseModel):
    """Structured flags passed to the ROI calculator if this rule restricts the operation."""
    min_supervision_pct_override: Optional[int] = Field(default=None, ge=0, le=100)
    requires_human_road_transport: bool = False


class RegulatoryRule(BaseModel):
    rule_id: str
    jurisdiction: Literal["CA", "IL", "IA", "TX", "AZ", "FEDERAL"]
    category: Literal["in_field_supervision", "public_road_transit", "osha_exemption", "permitting"]
    citation: str
    title: str
    legal_status: Literal["ENACTED", "AGENCY_GUIDANCE", "PROPOSED_OR_PETITION", "EXPIRED"]
    effective_date: Optional[str] = None
    source_url: str

    verified: bool = False
    verified_by: Optional[str] = None
    verified_date: Optional[str] = None

    applicability: ApplicabilityConditions
    constraint_summary: str
    operational_workaround: Optional[str] = None
    roi_impact: RoiHandoffImpact = RoiHandoffImpact()


class RegulatoryCorpus(BaseModel):
    schema_version: str = "1.1"
    last_updated: str
    rules: List[RegulatoryRule]