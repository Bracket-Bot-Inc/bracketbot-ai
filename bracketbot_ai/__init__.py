"""BracketBot AI - Object Detection and Speech Recognition for RK3588"""

from .detector import Detector
from .transcriber import Transcriber
from .model_manager import list_available_models, ensure_model

__version__ = "0.0.1"
__all__ = ["Detector", "Transcriber", "list_available_models", "ensure_model"]
