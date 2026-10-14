from __future__ import annotations

from typing import Iterable, Self

from engine.interfaces import (
    Candidate,
    ComponentConfig,
    ComponentDeps,
    Filterer,
    RecContext,
)


class SeenItemsFilterer(Filterer):
    config_model = ComponentConfig

    @property
    def name(self):
        return "SeenItemsFilter"

    @classmethod
    def start_component(cls, config: ComponentConfig, deps: ComponentDeps) -> Self:
        return cls()

    def filter(self, candidates: Iterable[Candidate], context: RecContext):
        yield from filter(lambda x: x.item_id not in context.item_id_filter, candidates)


class HistoryFilterer(Filterer):
    config_model = ComponentConfig

    @property
    def name(self):
        return "HistoryFilter"

    @classmethod
    def start_component(cls, config: ComponentConfig, deps: ComponentDeps) -> Self:
        return cls()

    def filter(self, candidates: Iterable[Candidate], context: RecContext):
        history_items = set(context.user_history["AT_Purchase"])
        yield from filter(lambda x: x.item_id not in history_items, candidates)
