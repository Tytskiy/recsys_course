from engine.interfaces import RuntimeProfile
from engine.runtime_profile.runtime_ials import RuntimeIALS

RUNTIME_PROFILES: dict[str, type[RuntimeProfile]] = {
    "RuntimeIALS": RuntimeIALS,
}

__all__ = ["RUNTIME_PROFILES", "RuntimeIALS"]
