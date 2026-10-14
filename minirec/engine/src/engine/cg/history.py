from __future__ import annotations

from typing import Generator

from engine.interfaces import (
    ActionType,
    Candidate,
    CandidateGenerator,
    ComponentConfig,
    ComponentDeps,
    RecContext,
)


class HistoryConfig(ComponentConfig):
    action_type: ActionType


class History(CandidateGenerator):
    config_model = HistoryConfig

    def __init__(self, action_type: ActionType):
        self.action_type = action_type

    @property
    def name(self) -> str:
        return "History"

    @classmethod
    def start_component(cls, config: HistoryConfig, deps: ComponentDeps) -> History:
        return cls(config.action_type)

    def extract_candidates(
        self,
        num_candidates: int,
        context: RecContext,
    ) -> Generator[Candidate, None, None]:
        history_items = context.user_history[self.action_type]
        unique_ids = list(dict.fromkeys(history_items).keys())
        for item_id in unique_ids[:num_candidates]:
            yield Candidate(item_id=item_id, factors={})
