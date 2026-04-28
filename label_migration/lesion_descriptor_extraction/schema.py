"""Pydantic schema for BI-RADS lexicon descriptor extraction.

Source: short free-text from `cadbusi.Lesions.description`
(e.g. "oval hypoechoic circumscribed parallel nonvascular non shadowing mass").

Enum vocabularies follow the ACR BI-RADS US lexicon, with extra "OTHER"
buckets for the few non-lexicon things radiologists write.
"""
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Shape(str, Enum):
    OVAL = "OVAL"
    ROUND = "ROUND"
    IRREGULAR = "IRREGULAR"
    LOBULATED = "LOBULATED"
    NOT_SPECIFIED = "NOT_SPECIFIED"


class Orientation(str, Enum):
    PARALLEL = "PARALLEL"
    NOT_PARALLEL = "NOT_PARALLEL"            # "taller-than-wide"
    NOT_SPECIFIED = "NOT_SPECIFIED"


class MarginClarity(str, Enum):
    """Top-level margin classifier. Detail (e.g. spiculated vs angular) goes in MarginDetail."""
    CIRCUMSCRIBED = "CIRCUMSCRIBED"
    NOT_CIRCUMSCRIBED = "NOT_CIRCUMSCRIBED"
    INDISTINCT = "INDISTINCT"
    OBSCURED = "OBSCURED"
    NOT_SPECIFIED = "NOT_SPECIFIED"


class MarginDetail(str, Enum):
    """Sub-classifier when margin is not_circumscribed."""
    ANGULAR = "ANGULAR"
    MICROLOBULATED = "MICROLOBULATED"
    SPICULATED = "SPICULATED"
    IRREGULAR = "IRREGULAR"
    NOT_SPECIFIED = "NOT_SPECIFIED"


class EchoPattern(str, Enum):
    ANECHOIC = "ANECHOIC"
    HYPOECHOIC = "HYPOECHOIC"
    HYPERECHOIC = "HYPERECHOIC"
    ISOECHOIC = "ISOECHOIC"
    COMPLEX_CYSTIC_AND_SOLID = "COMPLEX_CYSTIC_AND_SOLID"
    HETEROGENEOUS = "HETEROGENEOUS"
    NOT_SPECIFIED = "NOT_SPECIFIED"


class PosteriorFeatures(str, Enum):
    ENHANCEMENT = "ENHANCEMENT"
    SHADOWING = "SHADOWING"
    COMBINED = "COMBINED"
    NONE = "NONE"
    NOT_SPECIFIED = "NOT_SPECIFIED"


class Calcifications(str, Enum):
    PRESENT_IN_MASS = "PRESENT_IN_MASS"
    PRESENT_OUTSIDE_MASS = "PRESENT_OUTSIDE_MASS"
    INTRADUCTAL = "INTRADUCTAL"
    ABSENT = "ABSENT"
    NOT_SPECIFIED = "NOT_SPECIFIED"


class Vascularity(str, Enum):
    ABSENT = "ABSENT"
    INTERNAL = "INTERNAL"
    RIM = "RIM"
    NOT_SPECIFIED = "NOT_SPECIFIED"


class LesionKind(str, Enum):
    """Coarse category — useful when description doesn't yield BI-RADS descriptors."""
    MASS = "MASS"
    SIMPLE_CYST = "SIMPLE_CYST"
    COMPLICATED_CYST = "COMPLICATED_CYST"
    LYMPH_NODE = "LYMPH_NODE"
    DUCTAL = "DUCTAL"
    OTHER = "OTHER"
    NOT_SPECIFIED = "NOT_SPECIFIED"


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class LesionDescriptors(BaseModel):
    """Structured BI-RADS-style descriptors extracted from a lesion description phrase."""

    shape: Shape = Field(Shape.NOT_SPECIFIED)
    orientation: Orientation = Field(Orientation.NOT_SPECIFIED)
    margin_clarity: MarginClarity = Field(MarginClarity.NOT_SPECIFIED)
    margin_detail: MarginDetail = Field(MarginDetail.NOT_SPECIFIED)
    echo_pattern: EchoPattern = Field(EchoPattern.NOT_SPECIFIED)
    posterior_features: PosteriorFeatures = Field(PosteriorFeatures.NOT_SPECIFIED)
    calcifications: Calcifications = Field(Calcifications.NOT_SPECIFIED)
    vascularity: Vascularity = Field(Vascularity.NOT_SPECIFIED)
    lesion_kind: LesionKind = Field(LesionKind.NOT_SPECIFIED)
    confidence: Confidence = Field(Confidence.MEDIUM)
    notes: Optional[str] = Field(
        None, description="Short note (≤50 chars) for anything the enums can't capture"
    )
