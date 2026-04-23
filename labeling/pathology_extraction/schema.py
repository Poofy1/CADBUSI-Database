"""Pydantic schema for structured pathology extraction from rad_pathology_txt.

Designed for Mayo CADBUSI breast-ultrasound report addenda, which are typically
short declarative phrases (1-5 sentences) stating biopsy results.
"""
from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class PrimaryDiagnosis(str, Enum):
    """Overall diagnostic category."""
    MALIGNANT = "MALIGNANT"            # any carcinoma (invasive or in-situ)
    BENIGN = "BENIGN"                   # purely benign findings
    ATYPICAL = "ATYPICAL"               # ADH, ALH, atypia without carcinoma
    SUSPICIOUS = "SUSPICIOUS"           # "suspicious for" but not definitive
    INSUFFICIENT = "INSUFFICIENT"       # non-diagnostic sample
    UNKNOWN = "UNKNOWN"                 # text is present but not extractable


class CancerSubtype(str, Enum):
    """Specific carcinoma types. Can co-occur (e.g., IDC + DCIS)."""
    IDC = "INVASIVE_DUCTAL_CARCINOMA"
    DCIS = "DUCTAL_CARCINOMA_IN_SITU"
    ILC = "INVASIVE_LOBULAR_CARCINOMA"
    LCIS = "LOBULAR_CARCINOMA_IN_SITU"
    INVASIVE_CARCINOMA_NOS = "INVASIVE_CARCINOMA_UNSPECIFIED"
    INVASIVE_MAMMARY_CARCINOMA = "INVASIVE_MAMMARY_CARCINOMA"
    MUCINOUS = "MUCINOUS_CARCINOMA"
    TUBULAR = "TUBULAR_CARCINOMA"
    PAPILLARY = "PAPILLARY_CARCINOMA"
    MEDULLARY = "MEDULLARY_CARCINOMA"
    METASTATIC = "METASTATIC_CARCINOMA"
    ADENOID_CYSTIC = "ADENOID_CYSTIC_CARCINOMA"
    INFLAMMATORY = "INFLAMMATORY_CARCINOMA"
    CARCINOMA_NOS = "CARCINOMA_UNSPECIFIED"
    OTHER_CARCINOMA = "OTHER_CARCINOMA"


class BenignSubtype(str, Enum):
    """Benign findings. Can co-occur."""
    FIBROADENOMA = "FIBROADENOMA"
    CYST = "CYST"
    FIBROCYSTIC_CHANGE = "FIBROCYSTIC_CHANGE"
    PAPILLOMA = "PAPILLOMA"
    ADH = "ATYPICAL_DUCTAL_HYPERPLASIA"
    ALH = "ATYPICAL_LOBULAR_HYPERPLASIA"
    FLAT_EPITHELIAL_ATYPIA = "FLAT_EPITHELIAL_ATYPIA"
    FIBROSIS_SCAR = "FIBROSIS_OR_SCAR"
    FAT_NECROSIS = "FAT_NECROSIS"
    PHYLLODES_BENIGN = "PHYLLODES_BENIGN"
    HAMARTOMA = "HAMARTOMA"
    LYMPH_NODE_BENIGN = "LYMPH_NODE_BENIGN"
    PASH = "PSEUDOANGIOMATOUS_STROMAL_HYPERPLASIA"
    RADIAL_SCAR = "RADIAL_SCAR"
    SCLEROSING_ADENOSIS = "SCLEROSING_ADENOSIS"
    USUAL_DUCTAL_HYPERPLASIA = "USUAL_DUCTAL_HYPERPLASIA"
    COLUMNAR_CELL_CHANGE = "COLUMNAR_CELL_CHANGE"
    BENIGN_BREAST_TISSUE = "BENIGN_BREAST_TISSUE"
    OTHER_BENIGN = "OTHER_BENIGN"


class Laterality(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    BILATERAL = "BILATERAL"
    NOT_SPECIFIED = "NOT_SPECIFIED"


class Grade(str, Enum):
    """Histologic grade. Nottingham for invasive; nuclear grade for DCIS/LCIS."""
    NOTTINGHAM_1 = "NOTTINGHAM_1"
    NOTTINGHAM_2 = "NOTTINGHAM_2"
    NOTTINGHAM_3 = "NOTTINGHAM_3"
    NUCLEAR_LOW = "NUCLEAR_LOW"
    NUCLEAR_INTERMEDIATE = "NUCLEAR_INTERMEDIATE"
    NUCLEAR_HIGH = "NUCLEAR_HIGH"
    NOT_SPECIFIED = "NOT_SPECIFIED"


class LymphNodeStatus(str, Enum):
    NEGATIVE = "NEGATIVE"           # "negative for metastatic carcinoma"
    POSITIVE = "POSITIVE"           # metastasis found
    ISOLATED_TUMOR_CELLS = "ISOLATED_TUMOR_CELLS"
    MICROMETASTASIS = "MICROMETASTASIS"
    NOT_REPORTED = "NOT_REPORTED"


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class PathologyExtraction(BaseModel):
    """Structured extraction from one rad_pathology_txt entry."""

    primary_diagnosis: PrimaryDiagnosis = Field(
        ..., description="Overall diagnostic category of the pathology finding"
    )
    cancer_subtypes: List[CancerSubtype] = Field(
        default_factory=list,
        description="Specific carcinoma types explicitly mentioned. Empty if no carcinoma present.",
    )
    benign_subtypes: List[BenignSubtype] = Field(
        default_factory=list,
        description="Specific benign findings explicitly mentioned. Empty if none.",
    )
    laterality: Laterality = Field(
        Laterality.NOT_SPECIFIED,
        description="Side mentioned in the text (LEFT/RIGHT/BILATERAL/NOT_SPECIFIED)",
    )
    size_mm: Optional[float] = Field(
        None, description="Largest dimension in mm if stated; null otherwise"
    )
    grade: Grade = Field(
        Grade.NOT_SPECIFIED,
        description="Histologic grade if reported (Nottingham for invasive, nuclear for in-situ)",
    )
    lymph_node_status: LymphNodeStatus = Field(
        LymphNodeStatus.NOT_REPORTED,
        description="Lymph node involvement status if the text mentions a sentinel or axillary biopsy",
    )
    is_lymph_node_biopsy: bool = Field(
        False,
        description="True if the pathology text describes a lymph node biopsy (not a primary breast biopsy)",
    )
    confidence: Confidence = Field(
        Confidence.MEDIUM,
        description="Extraction confidence. LOW if text is ambiguous, abbreviated, or truncated.",
    )
    notes: Optional[str] = Field(
        None,
        description="Short free-text note if anything important couldn't be captured in the enums (≤50 chars)",
    )
