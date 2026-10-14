from __future__ import annotations

from engine.interfaces import (
    Candidate,
    ComponentConfig,
    ComponentDeps,
    Factor,
    Ranker,
    RecContext,
)


class FeatureRerankerConfig(ComponentConfig):
    feature_key: str
    default: int | float | str
    descending: bool = True


class FeatureReranker(Ranker):
    config_model = FeatureRerankerConfig

    def __init__(self, feature_key: str, default: Factor, descending: bool = True):
        self.feature_key = feature_key
        self.descending = descending
        self.default = default

    @property
    def name(self):
        return f"{self.feature_key}-Reranker"

    @classmethod
    def start_component(cls, config: FeatureRerankerConfig, deps: ComponentDeps) -> FeatureReranker:
        return cls(
            feature_key=config.feature_key,
            default=config.default,
            descending=config.descending,
        )

    def rerank(
        self,
        candidates: list[Candidate],
        context: RecContext,
    ) -> list[Candidate]:
        return sorted(
            candidates,
            key=lambda c: c.factors.get(self.feature_key, self.default),
            reverse=self.descending,
        )
