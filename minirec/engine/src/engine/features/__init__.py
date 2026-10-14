from engine.features.config import AVAILABLE_FEATURES
from engine.features.manager import SimpleManager
from engine.interfaces import FeatureManager

FEATURE_MANAGERS: dict[str, type[FeatureManager]] = {
    "SimpleManager": SimpleManager,
}

__all__ = [
    "SimpleManager",
    "FEATURE_MANAGERS",
    "AVAILABLE_FEATURES"
]
