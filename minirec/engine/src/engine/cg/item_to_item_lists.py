from __future__ import annotations

from typing import Generator

from engine.interfaces import (
    ID,
    ActionType,
    Candidate,
    CandidateGenerator,
    ComponentConfig,
    ComponentDeps,
    FeatureStorage,
    RecContext,
)


class ItemToItemListsConfig(ComponentConfig):
    action_type: ActionType
    storage_feature_name: str
    num_items_from_history: int


class ItemToItemLists(CandidateGenerator):
    config_model = ItemToItemListsConfig

    def __init__(
        self,
        feature_storage: FeatureStorage,
        storage_feature_name: str,
        num_items_from_history: int,
        action_type: ActionType,
    ):
        self.feature_storage = feature_storage
        self.action_type = action_type
        self.storage_feature_name = storage_feature_name
        self.num_items_from_history = num_items_from_history

    @property
    def name(self) -> str:
        return "ItemToItemLists"

    @classmethod
    def start_component(cls, config: ItemToItemListsConfig, deps: ComponentDeps) -> ItemToItemLists:
        return cls(
            feature_storage=deps.feature_storage,
            action_type=config.action_type,
            storage_feature_name=config.storage_feature_name,
            num_items_from_history=config.num_items_from_history,
        )

    def extract_candidates(
        self,
        num_candidates: int,
        context: RecContext,
    ) -> Generator[Candidate, None, None]:
        """Yield the stored item lists of the first `num_items_from_history` unique
        history items (history order, then list order), skipping candidates already
        yielded, at most `num_candidates` in total (none for 0).
        """
        # TODO: Your solution
