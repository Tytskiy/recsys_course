from __future__ import annotations

from engine.interfaces import (
    Candidate,
    CandidateGenerator,
    ComponentConfig,
    ComponentDeps,
    Embedding,
    EmbeddingStorage,
    RecContext,
)


class UserToItemKNNConfig(ComponentConfig):
    embedding_type: str


class UserToItemKNN(CandidateGenerator):
    config_model = UserToItemKNNConfig

    def __init__(
        self,
        embedding_type: str,
        embedding_storage: EmbeddingStorage,
    ):
        self.embedding_type = embedding_type
        self.embedding_storage = embedding_storage

    @classmethod
    def start_component(cls, config: UserToItemKNNConfig, deps: ComponentDeps) -> UserToItemKNN:
        return cls(
            embedding_type=config.embedding_type,
            embedding_storage=deps.embedding_storage,
        )

    @property
    def name(self) -> str:
        return f"{self.embedding_type}_userKNN"

    def extract_candidates(self, num_candidates: int, context: RecContext):
        """ANN-search `item_<embedding_type>` with the user embedding and yield the hits.

        The embedding is `context.user_profile[f"user_{embedding_type}"]` (written by
        `RuntimeIALS` or the feature manager). Yield nothing when it is missing.
        """
        # TODO: Your solution
