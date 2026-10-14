"""Test doubles shared by `tests/` and the conftests: `from tests.fakes import ...`.

Import only what `tests/conftest.py` already imports (no CG or ranker module), so
one task's unfinished code never breaks another task's collection.
"""

from __future__ import annotations

from typing import Literal, get_args

import numpy as np
from engine.bloom import BloomToken
from engine.candidate_logger import CandidateLogger
from engine.interfaces import (
    ID,
    ActionType,
    Candidate,
    CandidateGenerator,
    EmbeddingStorage,
    Factor,
    FeatureStorage,
    RecContext,
    SourceType,
)

ACTION_TYPES: tuple[ActionType, ...] = get_args(ActionType.__value__)


class UnusedFeatureStorage(FeatureStorage):
    """Stands in for a storage the component under test never reads."""


class UnusedEmbeddingStorage(EmbeddingStorage):
    """Stands in for a storage the component under test never reads."""


class RecordingCandidateLogger(CandidateLogger):
    """Keeps records in memory, so component tests never depend on the
    `candidate_logger` task or write files."""

    def __init__(self):
        self.records: list[dict] = []
        self.cg_records: list[tuple[RecContext, dict[str, list[ID]]]] = []

    def log_candidates(
        self, request_id: str, context: RecContext, candidates: list[Candidate], num_served: int
    ) -> None:
        self.records.append({"request_id": request_id, "candidates": candidates, "num_served": num_served})

    def log_cg_candidates(self, context: RecContext, candidate_lists: dict[str, list[ID]]) -> None:
        self.cg_records.append((context, dict(candidate_lists)))

    def close(self) -> None:
        pass


class DictFeatureStorage(FeatureStorage):
    """`bulk_extract` over `{item_id: {feature: value}}`; absent items and features are left out."""

    def __init__(self, item_features: dict[ID, dict[str, Factor]]):
        self.item_features = item_features

    def bulk_extract(
        self, features: list[str], item_ids: list[ID], user_id: ID | None = None
    ) -> dict[ID, dict[str, Factor]]:
        return {
            item_id: {f: v for f, v in self.item_features.get(item_id, {}).items() if f in features}
            for item_id in item_ids
        }


class DictEmbeddingStorage(EmbeddingStorage):
    """`get` and exact dot-product `search` over `{embedding_type: {id: vector}}`.
    Records `(embedding_type, ids)` per `get` in `requests`."""

    def __init__(self, tables: dict[str, dict[ID, np.ndarray]]):
        self.tables = tables
        self.requests: list[tuple[str, list[ID]]] = []

    def get(self, embedding_type: str, ids: list[ID]) -> list[tuple[ID, np.ndarray]]:
        self.requests.append((embedding_type, list(ids)))
        table = self.tables.get(embedding_type, {})
        return [(i, table[i]) for i in ids if i in table]

    def search(self, embedding_type: str, query: np.ndarray, k: int) -> list[tuple[ID, float]]:
        scores = {i: float(np.dot(vector, query)) for i, vector in self.tables[embedding_type].items()}
        return sorted(scores.items(), key=lambda hit: -hit[1])[:k]


class ListCG(CandidateGenerator):
    """Yields the first `num_candidates` of `ids`; records each requested count in `requested`."""

    def __init__(self, name: str, ids: list[ID]):
        self._name = name
        self.ids = ids
        self.requested: list[int] = []

    @classmethod
    def start_component(cls, config, deps):
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._name

    def extract_candidates(self, num_candidates: int, context: RecContext):
        self.requested.append(num_candidates)
        for item_id in self.ids[:num_candidates]:
            yield Candidate(item_id=item_id, factors={})


class SlotTableModel:
    """Mixigen prescorer: scores a slot row `[*context, cg_index, position]` as
    `table[cg_index][position]`; keeps the rows of each `predict` call in `calls`."""

    def __init__(self, table: list[list[float]] | np.ndarray):
        self.table = np.asarray(table, dtype=np.float64)
        self.calls: list[list[list[Factor]]] = []

    def predict(self, factors: list[list[Factor]], prediction_type: Literal["RawFormulaVal", "Probability"]) -> np.ndarray:
        self.calls.append([list(row) for row in factors])
        slots = np.asarray(factors, dtype=np.float64)[:, -2:].astype(np.int64)
        return self.table[slots[:, 0], slots[:, 1]]


def make_context(
    history: dict[ActionType, list[ID]] | None = None,
    *,
    user_id: ID = 1,
    context_item_ids: list[ID] | None = None,
    source_type: SourceType = "ST_Feed",
) -> RecContext:
    """A request as the engine builds it; `history` fills `user_history`, other action
    types get an empty list."""
    context = RecContext.from_request(
        user_id=user_id,
        current_time=None,
        context_item_ids=context_item_ids,
        item_id_filter=BloomToken(),
        source_type=source_type,
    )
    context.user_history = {action_type: list((history or {}).get(action_type, [])) for action_type in ACTION_TYPES}
    return context
