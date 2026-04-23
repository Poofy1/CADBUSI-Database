"""Pydantic schema for structured extraction of full synoptic pathology reports.

Designed for Mayo's formal synoptic pathology reports stored in
`Pathology.synoptic_report`. These average ~3K chars and contain full structured
fields (Nottingham, tumor size, margins, LN breakdown, pTNM, ER/PR/HER2).

One SynopticExtraction per Pathology.path_id. Some Pathology rows describe
multi-part specimens (A, B, C — different biopsies/excisions from the same case);
we extract the PRIMARY finding and note when multiple specimens are described.
"""
from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================

class SpecimenType(str, Enum):
    CORE_BIOPSY = "CORE_BIOPSY"
    FINE_NEEDLE_ASPIRATION = "FINE_NEEDLE_ASPIRATION"
    EXCISIONAL_BIOPSY = "EXCISIONAL_BIOPSY"
    LUMPECTOMY = "LUMPECTOMY"
    SEGMENTAL_MASTECTOMY = "SEGMENTAL_MASTECTOMY"
    SIMPLE_MASTECTOMY = "SIMPLE_MASTECTOMY"
    TOTAL_MASTECTOMY = "TOTAL_MASTECTOMY"
    RADICAL_MASTECTOMY = "RADICAL_MASTECTOMY"
    SENTINEL_LYMPH_NODE_BIOPSY = "SENTINEL_LYMPH_NODE_BIOPSY"
    AXILLARY_LYMPH_NODE_DISSECTION = "AXILLARY_LYMPH_NODE_DISSECTION"
    OTHER = "OTHER"
    NOT_SPECIFIED = "NOT_SPECIFIED"


class SpecimenSite(str, Enum):
    BREAST = "BREAST"
    AXILLARY_LYMPH_NODE = "AXILLARY_LYMPH_NODE"
    INTERNAL_MAMMARY_LYMPH_NODE = "INTERNAL_MAMMARY_LYMPH_NODE"
    SUPRACLAVICULAR_LYMPH_NODE = "SUPRACLAVICULAR_LYMPH_NODE"
    SKIN = "SKIN"
    CHEST_WALL = "CHEST_WALL"
    OTHER = "OTHER"
    NOT_SPECIFIED = "NOT_SPECIFIED"


class Laterality(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    BILATERAL = "BILATERAL"
    NOT_SPECIFIED = "NOT_SPECIFIED"


class PrimaryDiagnosis(str, Enum):
    INVASIVE_CARCINOMA = "INVASIVE_CARCINOMA"
    IN_SITU_ONLY = "IN_SITU_ONLY"                     # DCIS/LCIS without invasion
    ATYPICAL = "ATYPICAL"                              # ADH/ALH only
    BENIGN = "BENIGN"
    METASTATIC_LYMPH_NODE = "METASTATIC_LYMPH_NODE"   # LN has mets; primary elsewhere
    NEGATIVE_LYMPH_NODE = "NEGATIVE_LYMPH_NODE"
    INSUFFICIENT = "INSUFFICIENT"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class HistologicType(str, Enum):
    IDC = "INVASIVE_DUCTAL_CARCINOMA"
    IDC_NST = "INVASIVE_CARCINOMA_NO_SPECIAL_TYPE"    # NST = NOS with ductal phenotype
    ILC = "INVASIVE_LOBULAR_CARCINOMA"
    IDC_WITH_LOBULAR = "INVASIVE_DUCTAL_WITH_LOBULAR_FEATURES"
    MIXED_DUCTAL_LOBULAR = "MIXED_DUCTAL_AND_LOBULAR"
    MUCINOUS = "MUCINOUS_CARCINOMA"
    TUBULAR = "TUBULAR_CARCINOMA"
    PAPILLARY = "PAPILLARY_CARCINOMA"
    MEDULLARY = "MEDULLARY_CARCINOMA"
    METAPLASTIC = "METAPLASTIC_CARCINOMA"
    MICROPAPILLARY = "MICROPAPILLARY_CARCINOMA"
    APOCRINE = "APOCRINE_CARCINOMA"
    ADENOID_CYSTIC = "ADENOID_CYSTIC_CARCINOMA"
    INFLAMMATORY = "INFLAMMATORY_CARCINOMA"
    DCIS = "DUCTAL_CARCINOMA_IN_SITU"
    LCIS = "LOBULAR_CARCINOMA_IN_SITU"
    PHYLLODES = "PHYLLODES"
    OTHER = "OTHER"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Grade(str, Enum):
    GRADE_1 = "GRADE_1"
    GRADE_2 = "GRADE_2"
    GRADE_3 = "GRADE_3"
    NOT_APPLICABLE = "NOT_APPLICABLE"      # e.g., for DCIS (uses nuclear grade instead)
    NOT_REPORTED = "NOT_REPORTED"


class NuclearGrade(str, Enum):
    LOW = "LOW"
    INTERMEDIATE = "INTERMEDIATE"
    HIGH = "HIGH"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NOT_REPORTED = "NOT_REPORTED"


class Focality(str, Enum):
    SINGLE = "SINGLE"
    MULTIFOCAL = "MULTIFOCAL"              # multiple foci in one quadrant
    MULTICENTRIC = "MULTICENTRIC"          # foci in different quadrants
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NOT_REPORTED = "NOT_REPORTED"


class MarginStatus(str, Enum):
    NEGATIVE = "NEGATIVE"                  # R0, clear margins
    POSITIVE = "POSITIVE"                  # R1, tumor at inked margin
    CLOSE = "CLOSE"                        # within 1-2 mm but not at ink
    NOT_APPLICABLE = "NOT_APPLICABLE"      # biopsy specimens
    NOT_REPORTED = "NOT_REPORTED"


class ReceptorStatus(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    LOW_POSITIVE = "LOW_POSITIVE"          # 1-10% for ER/PR
    EQUIVOCAL = "EQUIVOCAL"                # for HER2 IHC 2+ pending FISH
    PENDING = "PENDING"
    NOT_PERFORMED = "NOT_PERFORMED"
    NOT_REPORTED = "NOT_REPORTED"


class HER2Method(str, Enum):
    IHC = "IHC"
    FISH = "FISH"
    DUAL = "DUAL"                          # both reported
    NOT_REPORTED = "NOT_REPORTED"


class PTCategory(str, Enum):
    pTis = "pTis"
    pT1mi = "pT1mi"
    pT1a = "pT1a"
    pT1b = "pT1b"
    pT1c = "pT1c"
    pT1 = "pT1"
    pT2 = "pT2"
    pT3 = "pT3"
    pT4 = "pT4"
    pT4a = "pT4a"
    pT4b = "pT4b"
    pT4c = "pT4c"
    pT4d = "pT4d"
    pTx = "pTx"
    NOT_REPORTED = "NOT_REPORTED"


class PNCategory(str, Enum):
    pN0 = "pN0"
    pN0_i_plus = "pN0_i_plus"              # isolated tumor cells
    pN0_mol_plus = "pN0_mol_plus"
    pN1mi = "pN1mi"                        # micrometastases only
    pN1 = "pN1"
    pN1a = "pN1a"
    pN1b = "pN1b"
    pN1c = "pN1c"
    pN2 = "pN2"
    pN2a = "pN2a"
    pN2b = "pN2b"
    pN3 = "pN3"
    pN3a = "pN3a"
    pN3b = "pN3b"
    pN3c = "pN3c"
    pNx = "pNx"
    NOT_REPORTED = "NOT_REPORTED"


class PMCategory(str, Enum):
    pM0 = "pM0"
    pM1 = "pM1"
    cM0 = "cM0"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NOT_REPORTED = "NOT_REPORTED"


class TreatmentEffect(str, Enum):
    COMPLETE_RESPONSE = "COMPLETE_RESPONSE"
    PARTIAL_RESPONSE = "PARTIAL_RESPONSE"
    MINIMAL_RESPONSE = "MINIMAL_RESPONSE"
    NO_RESPONSE = "NO_RESPONSE"
    NOT_APPLICABLE = "NOT_APPLICABLE"      # no neoadjuvant
    NOT_REPORTED = "NOT_REPORTED"


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# ============================================================================
# Sub-models
# ============================================================================

class InvasiveTumor(BaseModel):
    """Details of the invasive component (if present)."""
    histologic_type: HistologicType = Field(HistologicType.NOT_APPLICABLE)
    grade_overall: Grade = Field(Grade.NOT_REPORTED, description="Overall Nottingham grade (1/2/3)")
    nottingham_glandular: Optional[int] = Field(None, ge=1, le=3, description="Glandular/tubular differentiation score 1-3")
    nottingham_nuclear: Optional[int] = Field(None, ge=1, le=3, description="Nuclear pleomorphism score 1-3")
    nottingham_mitotic: Optional[int] = Field(None, ge=1, le=3, description="Mitotic rate score 1-3")
    tumor_size_mm: Optional[float] = Field(None, description="Largest dimension in mm (convert cm → mm if reported in cm)")
    tumor_size_additional_mm: List[float] = Field(default_factory=list, description="Other dimensions if reported (e.g., 30, 22 from '30 x 22 mm')")
    focality: Focality = Field(Focality.NOT_REPORTED)
    lymphovascular_invasion: Optional[bool] = Field(None)
    dermal_lymphovascular_invasion: Optional[bool] = Field(None)
    perineural_invasion: Optional[bool] = Field(None)


class InSituComponent(BaseModel):
    """Details of in-situ components (DCIS or LCIS)."""
    dcis_present: Optional[bool] = Field(None)
    dcis_nuclear_grade: NuclearGrade = Field(NuclearGrade.NOT_APPLICABLE)
    dcis_pattern: List[str] = Field(default_factory=list, description="Free-text patterns: solid, cribriform, papillary, comedo, micropapillary")
    dcis_necrosis: Optional[bool] = Field(None)
    dcis_size_mm: Optional[float] = Field(None)
    lcis_present: Optional[bool] = Field(None)


class Margins(BaseModel):
    """Margin status (for excision/mastectomy specimens)."""
    invasive_margin_status: MarginStatus = Field(MarginStatus.NOT_REPORTED)
    closest_invasive_margin_name: Optional[str] = Field(None, description="e.g., posterior, superior, anterior, medial, lateral")
    closest_invasive_margin_distance_mm: Optional[float] = Field(None, description="Distance from tumor to closest margin in mm")
    dcis_margin_status: MarginStatus = Field(MarginStatus.NOT_APPLICABLE)
    closest_dcis_margin_distance_mm: Optional[float] = Field(None)


class LymphNodes(BaseModel):
    """Lymph node findings (from sentinel or axillary dissection)."""
    total_examined: Optional[int] = Field(None)
    sentinel_examined: Optional[int] = Field(None)
    with_macrometastases: Optional[int] = Field(None, description="Metastases > 2.0 mm")
    with_micrometastases: Optional[int] = Field(None, description="Metastases 0.2 - 2.0 mm")
    with_isolated_tumor_cells: Optional[int] = Field(None, description="ITCs ≤ 0.2 mm or ≤ 200 cells")
    largest_deposit_mm: Optional[float] = Field(None, description="Size of largest metastatic deposit in mm")
    extranodal_extension: Optional[bool] = Field(None)


class StagingTNM(BaseModel):
    """AJCC 8th edition pathologic TNM."""
    pT: PTCategory = Field(PTCategory.NOT_REPORTED)
    pN: PNCategory = Field(PNCategory.NOT_REPORTED)
    pM: PMCategory = Field(PMCategory.NOT_REPORTED)
    sentinel_modifier: bool = Field(False, description="True if pN has 'sn' modifier")
    overall_stage: Optional[str] = Field(None, description="e.g., IA, IIA, IIIB; free text since this is derived not always stated")


class ReceptorPanel(BaseModel):
    """Receptor / biomarker testing. Fields may be empty if testing not performed on this specimen."""
    er_status: ReceptorStatus = Field(ReceptorStatus.NOT_REPORTED)
    er_percent: Optional[float] = Field(None, description="Percent of tumor cells staining for ER (0-100)")
    er_intensity: Optional[str] = Field(None, description="Free text: weak / moderate / strong")
    pr_status: ReceptorStatus = Field(ReceptorStatus.NOT_REPORTED)
    pr_percent: Optional[float] = Field(None)
    pr_intensity: Optional[str] = Field(None)
    her2_status: ReceptorStatus = Field(ReceptorStatus.NOT_REPORTED)
    her2_method: HER2Method = Field(HER2Method.NOT_REPORTED)
    her2_ihc_score: Optional[int] = Field(None, ge=0, le=3, description="0, 1+, 2+, or 3+ if IHC")
    her2_fish_ratio: Optional[float] = Field(None, description="HER2/CEP17 ratio if FISH")
    ki67_percent: Optional[float] = Field(None)


class TreatmentContext(BaseModel):
    """Neoadjuvant / post-treatment context."""
    post_neoadjuvant: Optional[bool] = Field(None, description="True if specimen was post neoadjuvant chemotherapy/endocrine")
    treatment_effect: TreatmentEffect = Field(TreatmentEffect.NOT_REPORTED)


class BenignFinding(BaseModel):
    """When primary_diagnosis is BENIGN, capture what was found."""
    primary_benign_dx: Optional[str] = Field(None, description="Primary benign finding from diagnosis line (e.g., 'fibroadenoma', 'fibrocystic change', 'intraductal papilloma')")
    atypia_present: Optional[bool] = Field(None)
    atypia_type: Optional[str] = Field(None, description="e.g., 'ADH', 'ALH', 'flat epithelial atypia'")


# ============================================================================
# Root extraction
# ============================================================================

class SynopticExtraction(BaseModel):
    """Structured extraction of one Pathology.synoptic_report entry."""

    # Specimen metadata
    specimen_type: SpecimenType = Field(SpecimenType.NOT_SPECIFIED)
    specimen_site: SpecimenSite = Field(SpecimenSite.NOT_SPECIFIED)
    laterality: Laterality = Field(Laterality.NOT_SPECIFIED)
    multi_part_specimen: bool = Field(False, description="True if the report describes multiple lettered parts (A, B, C) — extraction focuses on the primary breast finding")

    # Top-level diagnosis
    primary_diagnosis: PrimaryDiagnosis = Field(PrimaryDiagnosis.UNKNOWN)

    # Detailed sub-components (only populate fields that are actually reported)
    invasive: Optional[InvasiveTumor] = Field(None, description="Populate if invasive carcinoma present")
    in_situ: Optional[InSituComponent] = Field(None, description="Populate if DCIS/LCIS present")
    margins: Optional[Margins] = Field(None, description="Populate for excision/mastectomy specimens where margins are reported")
    lymph_nodes: Optional[LymphNodes] = Field(None, description="Populate when LN examination is reported")
    staging: Optional[StagingTNM] = Field(None, description="Populate when pTNM is stated")
    receptors: Optional[ReceptorPanel] = Field(None, description="Populate when ER/PR/HER2 results are in the report")
    treatment_context: Optional[TreatmentContext] = Field(None)
    benign: Optional[BenignFinding] = Field(None, description="Populate when primary_diagnosis is BENIGN")

    # Meta
    confidence: Confidence = Field(Confidence.MEDIUM)
    extraction_notes: Optional[str] = Field(None, description="Short note if anything important wasn't captured (≤100 chars)")
