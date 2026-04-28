"""LLM-based BI-RADS descriptor extraction from cadbusi.Lesions.description."""
from .schema import (
    LesionDescriptors,
    Shape,
    Orientation,
    MarginClarity,
    MarginDetail,
    EchoPattern,
    PosteriorFeatures,
    Calcifications,
    Vascularity,
    LesionKind,
    Confidence,
)
from .extractor import LesionDescriptorExtractor, ExtractionResult

__all__ = [
    "LesionDescriptors",
    "Shape",
    "Orientation",
    "MarginClarity",
    "MarginDetail",
    "EchoPattern",
    "PosteriorFeatures",
    "Calcifications",
    "Vascularity",
    "LesionKind",
    "Confidence",
    "LesionDescriptorExtractor",
    "ExtractionResult",
]
