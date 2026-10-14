from __future__ import annotations

import numpy as np

from engine.interfaces import (
    Candidate,
    CandidateGenerator,
    ComponentConfig,
    ComponentDeps,
    EmbeddingStorage,
    RecContext,
)


class ItemToItemKNNConfig(ComponentConfig):
    embedding_type: str


class ItemToItemKNN(CandidateGenerator):
    config_model = ItemToItemKNNConfig

    def __init__(
        self,
        embedding_type: str,
        embedding_storage: EmbeddingStorage,
    ):
        self.embedding_type = embedding_type
        self.embedding_storage = embedding_storage

    @classmethod
    def start_component(cls, config: ItemToItemKNNConfig, deps: ComponentDeps) -> ItemToItemKNN:
        return cls(
            embedding_type=config.embedding_type,
            embedding_storage=deps.embedding_storage,
        )

    @property
    def name(self) -> str:
        return f"{self.embedding_type}_context_ANN"

    def extract_candidates(self, num_candidates: int, context: RecContext):
        """ANN-search `item_<embedding_type>` with the mean embedding of
        `context.context_item_ids`. Yield nothing when none of them has an embedding.
        """
        context_item_embeds = self.embedding_storage.get(
            f"item_{self.embedding_type}", context.context_item_ids
        )
        if not context_item_embeds:
            return
        query = np.mean([embed for _, embed in context_item_embeds], axis=0)
        ann_response = self.embedding_storage.search(
            f"item_{self.embedding_type}", query, num_candidates
        )
        for iid, _ in ann_response:
            yield Candidate(item_id=iid, factors={})
