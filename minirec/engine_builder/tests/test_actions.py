"""User-item matrix (task `user_item_matrix`). Pinned: per-(user, item) action counts turned into the
`target_type` confidence; pairs with confidence 0 dropped; users and items indexed by ascending id (u64 ids
included) in the returned mapping frames; BM25 reweighting per the README formula, with `k1` and `b`, and
non-positive weights dropped before indexing.
"""

import math

import polars as pl
import pytest
import scipy.sparse as sps
from tasks.actions import bm25_reweight, build_user_item_matrix

BIG = 2**63 + 5

# (user, item, action, times)
LOG = [
    (BIG, 100, "AT_CartUpdate", 2),
    (BIG, 200, "AT_Purchase", 1),
    (BIG, 200, "AT_CartUpdate", 1),
    (BIG, 300, "AT_Click", 3),
    (20, 100, "AT_CartUpdate", 1),
    (20, 300, "AT_Purchase", 1),
    (30, 200, "AT_Click", 1),
]


def actions(log: list[tuple[int, int, str, int]]) -> pl.DataFrame:
    rows = [(user, item, action) for user, item, action, times in log for _ in range(times)]
    return pl.DataFrame(
        rows, schema={"user_id": pl.UInt64, "item_id": pl.UInt64, "action_type": pl.String}, orient="row"
    )


def as_dict(matrix: sps.csr_array, user_to_idx: pl.DataFrame, item_to_idx: pl.DataFrame) -> dict:
    users = dict(zip(user_to_idx["user_idx"], user_to_idx["user_id"]))
    items = dict(zip(item_to_idx["item_idx"], item_to_idx["item_id"]))
    coo = matrix.tocoo()
    return {(users[u], items[i]): float(v) for u, i, v in zip(coo.row, coo.col, coo.data)}


@pytest.mark.parametrize(
    ("target_type", "expected"),
    [
        pytest.param("CartUpdate", {(BIG, 100): 2, (BIG, 200): 1, (20, 100): 1}, id="CartUpdate"),
        pytest.param(
            "MixedCartUpdatePurchase",
            {(BIG, 100): 2, (BIG, 200): 2, (20, 100): 1, (20, 300): 1},
            id="MixedCartUpdatePurchase",
        ),
        pytest.param("Click", {(BIG, 300): 3, (30, 200): 1}, id="Click"),
    ],
)
def test_confidence_per_target_type(target_type, expected):
    matrix, user_to_idx, item_to_idx = build_user_item_matrix(actions(LOG), target_type=target_type)
    assert isinstance(matrix, sps.csr_array)
    assert matrix.nnz == len(expected)
    assert as_dict(matrix, user_to_idx, item_to_idx) == expected


def test_ids_are_indexed_in_ascending_order():
    pairs = [(BIG, 300), (30, 200), (20, 100), (10, 400), (5, 250)]
    log = [(user, item, "AT_CartUpdate", 1) for user, item in pairs]
    matrix, user_to_idx, item_to_idx = build_user_item_matrix(actions(log), target_type="CartUpdate")
    assert user_to_idx.columns == ["user_id", "user_idx"]
    assert item_to_idx.columns == ["item_id", "item_idx"]
    assert user_to_idx.sort("user_idx").rows() == [(5, 0), (10, 1), (20, 2), (30, 3), (BIG, 4)]
    assert item_to_idx.sort("item_idx").rows() == [(100, 0), (200, 1), (250, 2), (300, 3), (400, 4)]
    assert matrix.shape == (5, 5)
    assert as_dict(matrix, user_to_idx, item_to_idx) == {pair: 1.0 for pair in pairs}


# BM25 log, CartUpdate: N = 4 users; df(A=1) = 1, df(B=2) = 2, df(C=3) = 2;
# L_1 = 2 + 1 = 3, L_2 = 3, L_3 = 1, L_4 = 1 -> L_bar = 2; user 5 only clicked: confidence 0, not in N, df or L
BM25_LOG = [(1, 1, "AT_CartUpdate", 2), (1, 2, "AT_CartUpdate", 1), (2, 2, "AT_CartUpdate", 3),
            (3, 3, "AT_CartUpdate", 1), (4, 3, "AT_CartUpdate", 1), (5, 1, "AT_Click", 1)]  # fmt: skip


def bm25(conf: float, df: int, length: float, k1: float = 100.0, b: float = 0.8) -> float:
    """The README formula for N = 4 users and L_bar = 2."""
    idf = math.log(4) - math.log(1 + df)
    return idf * conf * (k1 + 1) / (conf + k1 * (1 - b + b * length / 2))


def test_bm25_reweighting_follows_the_formula():
    matrix, user_to_idx, item_to_idx = build_user_item_matrix(
        actions(BM25_LOG), target_type="CartUpdate", use_bm25_reweighting=True
    )
    assert as_dict(matrix, user_to_idx, item_to_idx) == {
        (1, 1): pytest.approx(bm25(2, df=1, length=3)),
        (1, 2): pytest.approx(bm25(1, df=2, length=3)),
        (2, 2): pytest.approx(bm25(3, df=2, length=3)),
        (3, 3): pytest.approx(bm25(1, df=2, length=1)),
        (4, 3): pytest.approx(bm25(1, df=2, length=1)),
    }


@pytest.mark.parametrize(("k1", "b"), [pytest.param(1.0, 0.0, id="k1_1_b_0"), pytest.param(10.0, 1.0, id="k1_10_b_1")])
def test_bm25_reweight_uses_k1_and_b(k1, b):
    frame = pl.DataFrame({"user_id": [1, 1, 2, 3, 4], "item_id": [1, 2, 2, 3, 3], "confidence": [2.0, 1.0, 3.0, 1.0, 1.0]})
    result = bm25_reweight(frame, k1=k1, b=b).sort("user_id", "item_id")
    lengths, dfs = {1: 3, 2: 3, 3: 1, 4: 1}, {1: 1, 2: 2, 3: 2}
    expected = [bm25(c, dfs[i], lengths[u], k1, b) for u, i, c in frame.rows()]
    assert result.columns == ["user_id", "item_id", "confidence"]
    assert result["confidence"].to_list() == pytest.approx(expected)


def test_bm25_reweighting_drops_non_positive_weights():
    """An item every user touched gets idf < 0, which `implicit` would read as negative feedback."""
    # N = 3: idf(10) = log 3 - log 4 < 0 (dropped, with users 2 and 3), idf(20) = log 3 - log 2 > 0
    log = [(1, 10, "AT_CartUpdate", 1), (2, 10, "AT_CartUpdate", 1), (3, 10, "AT_CartUpdate", 1),
           (1, 20, "AT_CartUpdate", 1)]  # fmt: skip
    matrix, user_to_idx, item_to_idx = build_user_item_matrix(
        actions(log), target_type="CartUpdate", use_bm25_reweighting=True
    )
    assert matrix.nnz == 1
    assert item_to_idx["item_id"].to_list() == [20]
    assert user_to_idx["user_id"].to_list() == [1]


def test_bm25_is_off_by_default():
    matrix, *_ = build_user_item_matrix(actions(BM25_LOG), target_type="CartUpdate")
    assert sorted(matrix.data.tolist()) == [1.0, 1.0, 1.0, 2.0, 3.0]
