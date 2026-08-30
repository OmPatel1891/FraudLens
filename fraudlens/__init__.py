"""FraudLens - end-to-end transaction fraud detection."""

from .features import FeatureEngineer
from .modeling import (
    CalibratedModel,
    ScaledModel,
    evaluate,
    pick_threshold,
    sweep_thresholds,
)
from .preprocessing import Preprocessor

__version__ = "1.0.0"

__all__ = [
    "CalibratedModel",
    "FeatureEngineer",
    "Preprocessor",
    "ScaledModel",
    "evaluate",
    "pick_threshold",
    "sweep_thresholds",
]
