from engine.interfaces import Ranker
from engine.ranking.feature_ranker import FeatureReranker
from engine.ranking.model_ranker import ModelRanker

RANKERS: dict[str, type[Ranker]] = {
    "FeatureReranker": FeatureReranker,
    "ModelRanker": ModelRanker,
}

__all__ = ["FeatureReranker", "ModelRanker", "RANKERS"]
