from __future__ import annotations

from collections import Counter

import numpy as np

from engine.interfaces import (
    ID,
    ActionType,
    ComponentConfig,
    ComponentDeps,
    Embedding,
    EmbeddingStorage,
    RecContext,
    RuntimeProfile,
)


class RuntimeIALSConfig(ComponentConfig):
    action_type: ActionType
    item_gramian_path: str
    embedding_type: str = "iALS"
    alpha: float = 1.0
    l2_alpha: float = 0.1
    max_history: int | None = None


class RuntimeIALS(RuntimeProfile):
    config_model = RuntimeIALSConfig

    def __init__(
        self,
        embedding_storage: EmbeddingStorage,
        action_type: ActionType,
        item_gramian: np.ndarray,
        embedding_type: str = "iALS",
        alpha: float = 1.0,
        l2_alpha: float = 0.1,
        max_history: int | None = None,
    ):
        self.embedding_storage = embedding_storage
        self.action_type = action_type
        self.item_gramian = item_gramian
        self.embedding_type = embedding_type
        self.alpha = alpha
        self.l2_alpha = l2_alpha
        self.max_history = max_history

    @property
    def name(self) -> str:
        return "RuntimeIALS"

    @classmethod
    def start_component(cls, config: RuntimeIALSConfig, deps: ComponentDeps) -> RuntimeIALS:
        return cls(
            embedding_storage=deps.embedding_storage,
            action_type=config.action_type,
            item_gramian=np.load(deps.engine_data.resolve_path(config.item_gramian_path)),
            embedding_type=config.embedding_type,
            alpha=config.alpha,
            l2_alpha=config.l2_alpha,
            max_history=config.max_history,
        )

    def solve(self, factors: np.ndarray, confidence: np.ndarray) -> Embedding:
        """Solve `(item_gramian + factors^T diag(confidence - 1) factors + l2_alpha * I) u
        = factors^T confidence` for the user embedding `u` (the iALS user step)."""
        user_embedding = None
        # TODO: Your solution
        return user_embedding

    def collect_from_history(self, history_items: list[ID]) -> tuple[list[Embedding], list[float]]:
        """Collect item embeddings and confidences `alpha * count`, sorted by item id;
        items missing from the storage are skipped."""
        factors, confidence = [], []
        # TODO: Your solution
        return factors, confidence

    def hydrate_context(self, context: RecContext) -> RecContext:
        """Put `solve` over the first `max_history` items of `user_history[action_type]` into
        `user_profile[f"user_{embedding_type}"]`; write nothing if none of them has an embedding."""
        # TODO: Your solution
        return context
