import math
from typing import Protocol

import polars as pl
from engine.interfaces import ID


class Metric(Protocol):
    name: str

    def compute(
        self, candidates: list[ID], positives: list[ID], uid: int | None = None
    ) -> float:
        """Compute the metric"""


class Recall(Metric):
    def __init__(self, k: int | None):
        self.k = k
        self.name = f"Recall@{k}" if k else "Recall"

    def compute(
        self, candidates: list[ID], positives: list[ID], uid: int | None = None
    ) -> float:
        if self.k is not None:
            candidates = candidates[: self.k]
        num_positives = len(positives)
        num_hits = len(set(candidates) & set(positives))
        return num_hits / num_positives if num_positives else 0.0


class Precision(Metric):
    def __init__(self, k: int | None):
        self.k = k
        self.name = f"Precision@{k}" if k else "Precision"

    def compute(
        self, candidates: list[ID], positives: list[ID], uid: int | None = None
    ) -> float:
        if self.k is not None:
            candidates = candidates[: self.k]
        num_retrieved = len(candidates)
        num_hits = len(set(candidates) & set(positives))
        return num_hits / num_retrieved if num_retrieved else 0.0


class MAP(Metric):
    def __init__(self, k: int | None):
        self.k = k
        self.name = f"MAP@{k}" if k else "MAP"

    # TODO: Your solution

    def compute(
        self, candidates: list[ID], positives: list[ID], uid: int | None = None
    ) -> float:
        """APK@k of `candidates` against `positives` (formula in the task README)."""
        # TODO: Your solution


class DCG(Metric):
    def __init__(self, k: int | None):
        self.k = k
        self.name = f"DCG@{k}" if k else "DCG"

    @staticmethod
    def _dcg(relevance: list[int]) -> float:
        return sum(r / math.log(i + 2) for i, r in enumerate(relevance))

    def compute(
        self, candidates: list[ID], positives: list[ID], uid: int | None = None
    ) -> float:
        """DCG@k: sum of 1 / log(i + 1) over the positions i of hits among the top k."""
        # TODO: Your solution


class NDCG(DCG):
    def __init__(self, k: int | None):
        self.k = k
        self.name = f"NDCG@{k}" if k else "NDCG"

    def compute(
        self, candidates: list[ID], positives: list[ID], uid: int | None = None
    ) -> float:
        """NDCG@k = DCG@k / IDCG@k, the ideal over min(|positives|, k) positives; 0 when IDCG is 0."""
        # TODO: Your solution


class Serendipity(Metric):
    def __init__(self, actions: pl.DataFrame, k: int | None):
        """Precompute each user's history (set of item_ids) from `actions`."""
        self.k = k
        self.name = f"Serendipity@{k}" if k else "Serendipity"
        assert set(actions.columns).issuperset(["item_id", "user_id"])
        # TODO: Your solution

    def compute(
        self, candidates: list[ID], positives: list[ID], uid: int | None = None
    ) -> float:
        """Share of hits in the top k that are not in user `uid`'s history; 0 without hits."""
        # TODO: Your solution


class CategoryDiversity(Metric):
    def __init__(
        self, dm_product: pl.DataFrame, k: int | None = None
    ):
        """Every candidate passed to `compute` must be present in `dm_product`."""
        self.k = k
        self.name = f"CategoryDiversity@{k}" if k else "CategoryDiversity"
        assert set(dm_product.columns).issuperset(["item_id", "item_category"])
        # TODO: Your solution

    def compute(
        self, candidates: list[ID], positives: list[ID], uid: int | None = None
    ) -> float:
        """Distinct categories among the top k divided by their count; 0 for no candidates."""
        # TODO: Your solution
