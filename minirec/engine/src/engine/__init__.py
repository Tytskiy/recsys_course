from typing import TYPE_CHECKING

from engine.interfaces import Recommender

if TYPE_CHECKING:
    from engine.recommender import RecommenderEngine

__all__ = [
    "RecommenderEngine",
    "Recommender",
]


def __getattr__(name: str):
    if name == "RecommenderEngine":
        from engine.recommender import RecommenderEngine

        return RecommenderEngine
    raise AttributeError(f"module 'engine' has no attribute {name!r}")
