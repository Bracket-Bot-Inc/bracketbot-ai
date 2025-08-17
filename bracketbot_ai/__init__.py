"""BracketBot AI - Object Detection and Speech Recognition for RK3588"""

"""BracketBot AI - Object Detection and Speech Recognition for RK3588"""

# Lazy imports to make dependencies independent
def __getattr__(name):
    if name == "Detector":
        from .detector import Detector
        return Detector
    elif name == "Transcriber":
        from .transcriber import Transcriber
        return Transcriber
    elif name == "list_available_models":
        from .model_manager import list_available_models
        return list_available_models
    elif name == "ensure_model":
        from .model_manager import ensure_model
        return ensure_model
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__version__ = "0.0.1"
__all__ = ["Detector", "Transcriber", "list_available_models", "ensure_model"]
