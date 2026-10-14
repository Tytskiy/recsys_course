"""Ranking and beyond-accuracy metrics (task `metrics`); formulas in the task README, natural log.
Pinned: `k` cuts the candidate list (`None` = whole list); MAP divides by min(|R|, k), counts a repeated
hit once and is 1 for no positives; NDCG's ideal ranks min(|R|, k) positives, retrieved or not;
Serendipity counts hits outside the user's history (a user without actions has none); CategoryDiversity
is distinct categories over list length; empty candidate lists give 0.
"""

import math

import polars as pl
import pytest
from tasks.evaluation.metric import DCG, MAP, NDCG, CategoryDiversity, Precision, Recall, Serendipity


def inv_log(rank: int) -> float:
    """Gain of a hit at 1-based `rank`."""
    return 1 / math.log(rank + 1)


@pytest.mark.parametrize(
    ("metric", "positives", "candidates", "expected"),
    [
        pytest.param(Recall(2), [2, 4, 5], [1, 2, 3, 4], 1 / 3, id="Recall@2"),
        pytest.param(Recall(None), [2, 4, 5], [1, 2, 3, 4], 2 / 3, id="Recall"),
        pytest.param(Precision(2), [2, 4, 5], [1, 2, 3, 4], 1 / 2, id="Precision@2"),
        pytest.param(Precision(None), [2, 4, 5], [1, 2, 3, 4], 2 / 4, id="Precision"),
        pytest.param(MAP(3), [2, 3], [1, 2, 3, 4], (1 / 2 + 2 / 3) / 2, id="MAP@3"),
        # (1/1 + 2/2) / min(3, 2) = 1, not / |R| = 2/3
        pytest.param(MAP(2), [1, 2, 3], [1, 2, 4], 1.0, id="MAP@2-capped-at-k"),
        pytest.param(MAP(None), [1, 2, 3], [1], 1.0, id="MAP-capped-at-list-length"),
        pytest.param(MAP(None), [1, 2], [1, 1, 2], (1 + 2 / 3) / 2, id="MAP-duplicate-hit-counted-once"),
        pytest.param(MAP(3), [], [1, 2, 3, 4], 1.0, id="MAP-no-positives-is-1"),
        pytest.param(MAP(None), [1], [], 0.0, id="MAP-empty-candidates"),
        pytest.param(DCG(None), [2, 4], [1, 2, 3, 4], inv_log(2) + inv_log(4), id="DCG"),
        pytest.param(DCG(2), [1, 3], [1, 2, 3], inv_log(1), id="DCG@2"),
        pytest.param(
            NDCG(None), [2, 4], [1, 2, 3, 4], (inv_log(2) + inv_log(4)) / (inv_log(1) + inv_log(2)), id="NDCG"
        ),
        pytest.param(
            NDCG(None),
            [1, 5, 6],
            [1, 2, 3],
            inv_log(1) / (inv_log(1) + inv_log(2) + inv_log(3)),
            id="NDCG-unretrieved-positives-lower-score",
        ),
        pytest.param(NDCG(2), [1, 5, 6], [1, 2, 3], inv_log(1) / (inv_log(1) + inv_log(2)), id="NDCG@2-ideal-capped-at-k"),
        pytest.param(NDCG(None), [], [1, 2, 3], 0.0, id="NDCG-no-positives-is-0"),
    ],
)
def test_ranking_metric(metric, positives, candidates, expected):
    assert metric.compute(candidates, positives) == pytest.approx(expected)


ACTIONS = pl.DataFrame({"user_id": [1, 1, 1, 2], "item_id": [10, 20, 10, 30]})


@pytest.mark.parametrize(
    ("k", "uid", "positives", "candidates", "expected"),
    [
        pytest.param(None, 1, [10, 40, 50], [10, 40, 60, 50], 2 / 3, id="Serendipity"),
        pytest.param(2, 1, [10, 40, 50], [10, 40, 60, 50], 1 / 2, id="Serendipity@2"),
        pytest.param(None, 1, [10], [60, 10], 0.0, id="Serendipity-only-seen-hits"),
        pytest.param(None, 2, [10], [60, 10], 1.0, id="Serendipity-history-is-per-user"),
        pytest.param(None, 3, [10, 20], [10, 60, 20], 1.0, id="Serendipity-cold-user"),
        pytest.param(None, 1, [40], [60, 70], 0.0, id="Serendipity-no-hits"),
        pytest.param(None, 1, [10], [], 0.0, id="Serendipity-empty-candidates"),
    ],
)
def test_serendipity(k, uid, positives, candidates, expected):
    assert Serendipity(ACTIONS, k).compute(candidates, positives, uid=uid) == pytest.approx(expected)


PRODUCTS = pl.DataFrame({"item_id": [1, 2, 3, 4, 1], "item_category": ["a", "b", "a", "c", "a"]})


@pytest.mark.parametrize(
    ("k", "candidates", "expected"),
    [
        pytest.param(None, [1, 2, 4], 1.0, id="CategoryDiversity-all-distinct"),
        pytest.param(None, [1, 2, 3], 2 / 3, id="CategoryDiversity-repeated-category"),
        pytest.param(2, [1, 3, 2], 1 / 2, id="CategoryDiversity@2"),
        pytest.param(None, [], 0.0, id="CategoryDiversity-empty-candidates"),
    ],
)
def test_category_diversity(k, candidates, expected):
    assert CategoryDiversity(PRODUCTS, k).compute(candidates, []) == pytest.approx(expected)
