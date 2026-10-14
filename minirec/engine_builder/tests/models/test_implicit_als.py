"""iALS pipeline, builder half (task `implicit_als`). Pinned: `split_by_activity` splits rows by their sum
(warm at `>= warm_user_threshold`) and keeps row order; `train_model` returns one user row per matrix row:
warm rows are `implicit`'s fit on the warm rows, a cold row is one exact iALS user step on the returned item
factors; `export_embeddings` writes `item_<type>` and `user_<type>` with the Dot metric, keyed by the
original ids; `train_implicit_als` builds the matrix from `params.target_type` and
`params.use_bm25_reweighting`, and the whole pipeline beats TopPop on held-out data. The KNN CG half is engine/tests/unit/cgs/test_user_to_item_knn.py.
"""

import dataclasses

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sps
import tasks.models.implicit_als as implicit_als
from engine.interfaces import EmbeddingStorage
from engine.storages import QdrantEmbeddingStorage
from implicit.als import AlternatingLeastSquares
from tests.synthetic import SMALL, make_actions, recall_ndcg_at_k, split_ts, top_pop_factors, train_test

# row sums: 4, 3, 0, 3.5, 5 (BM25 weights are not integers)
MATRIX = sps.csr_array(
    np.array(
        [
            [1.0, 3.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.5, 1.0, 1.0],
            [0.0, 2.0, 3.0],
        ]
    )
)


@pytest.mark.parametrize(
    ("threshold", "expected_cold"),
    [
        pytest.param(4, [1, 2, 3], id="threshold_4_boundary_warm"),
        pytest.param(3, [2], id="threshold_3"),
        pytest.param(6, [0, 1, 2, 3, 4], id="all_cold"),
    ],
)
def test_split_by_activity(threshold, expected_cold):
    warm, cold, cold_ids = implicit_als.split_by_activity(MATRIX, warm_user_threshold=threshold)
    expected_warm = [u for u in range(5) if u not in expected_cold]
    assert cold_ids.tolist() == expected_cold
    np.testing.assert_array_equal(sps.csr_array(warm).toarray(), MATRIX.toarray()[expected_warm])
    np.testing.assert_array_equal(sps.csr_array(cold).toarray(), MATRIX.toarray()[expected_cold])


PARAMS = implicit_als.IALSParams(dim=4, max_iter=3, l2_alpha=0.1, conf_alpha=2.0, warm_user_threshold=3, random_state=0)


def interactions(seed: int = 0) -> sps.csr_array:
    rng = np.random.default_rng(seed)
    counts = (rng.random((30, 20)) < 0.3) * rng.integers(1, 4, size=(30, 20))
    counts[[3, 7]] = 0
    counts[3, 5] = 1
    counts[7, [2, 9]] = 1
    return sps.csr_array(counts.astype(np.float32))


def test_train_model_shapes():
    users, items = implicit_als.train_model(interactions(), PARAMS)
    assert users.shape == (30, 4)
    assert items.shape == (20, 4)


def test_warm_rows_are_the_fit_on_warm_users():
    """Warm rows keep their matrix positions: they equal `implicit`'s fit of the same params on the warm rows."""
    matrix = interactions()
    users, items = implicit_als.train_model(matrix, PARAMS)
    warm, _, cold_ids = implicit_als.split_by_activity(matrix, PARAMS.warm_user_threshold)
    model = AlternatingLeastSquares(factors=4, regularization=0.1, alpha=2.0, iterations=3, random_state=0)
    model.fit(warm)
    np.testing.assert_allclose(users[np.setdiff1d(np.arange(30), cold_ids)], model.user_factors, rtol=1e-5)
    np.testing.assert_allclose(items, model.item_factors, rtol=1e-5)


def test_cold_user_is_one_exact_user_step():
    """A cold row is the iALS user step on the trained items:
    (Y^T Y + l2 I + sum_j (c_j - 1) y_j y_j^T) x = sum_j c_j y_j, with c_j = conf_alpha * r_j."""
    matrix = interactions()
    users, items = implicit_als.train_model(matrix, PARAMS)
    Y = items.astype(np.float64)
    for u in (3, 7):
        row = matrix[[u]]
        F, c = Y[row.indices], PARAMS.conf_alpha * row.data
        A = Y.T @ Y + PARAMS.l2_alpha * np.eye(4) + (F * (c - 1)[:, None]).T @ F
        np.testing.assert_allclose(users[u], np.linalg.solve(A, F.T @ c), rtol=1e-3, atol=1e-4)


class RecordingStorage(EmbeddingStorage):
    def __init__(self):
        self.indexes = {}

    def build_embedding_index(self, embedding_type, embeddings, metric="Dot"):
        self.indexes[embedding_type] = (metric, {int(i): np.asarray(e) for i, e in embeddings})


def test_export_embeddings_keys_by_original_ids():
    storage = RecordingStorage()
    item_factors = np.array([[1.0, 0.0], [0.0, 1.0]])
    user_factors = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    item_to_idx = pl.DataFrame({"item_id": [200, 100], "item_idx": [1, 0]})
    user_to_idx = pl.DataFrame({"user_id": [7, 5, 6], "user_idx": [2, 0, 1]})
    implicit_als.export_embeddings(item_factors, user_factors, item_to_idx, user_to_idx, storage, "test")

    assert storage.indexes.keys() == {"item_test", "user_test"}
    item_metric, item_vectors = storage.indexes["item_test"]
    user_metric, user_vectors = storage.indexes["user_test"]
    assert (item_metric, user_metric) == ("Dot", "Dot")
    assert {k: v.tolist() for k, v in item_vectors.items()} == {100: [1.0, 0.0], 200: [0.0, 1.0]}
    assert {k: v.tolist() for k, v in user_vectors.items()} == {5: [1.0, 1.0], 6: [2.0, 2.0], 7: [3.0, 3.0]}


# N = 3 clicking users: item 10 has df = 3 (idf = log 3 - log 4 < 0, dropped with BM25), items 11-13 df = 1;
# user 4 only adds to cart
CLICK_LOG = pl.DataFrame(
    {
        "user_id": [1, 1, 2, 2, 3, 3, 4],
        "item_id": [10, 11, 10, 12, 10, 13, 10],
        "action_type": ["AT_Click"] * 6 + ["AT_CartUpdate"],
    },
    schema={"user_id": pl.UInt64, "item_id": pl.UInt64, "action_type": pl.String},
)


@pytest.mark.parametrize(
    ("use_bm25_reweighting", "expected_items"),
    [pytest.param(False, {10, 11, 12, 13}, id="counts"), pytest.param(True, {11, 12, 13}, id="bm25_drops_item_10")],
)
def test_train_implicit_als_builds_matrix_from_params(use_bm25_reweighting, expected_items):
    storage = RecordingStorage()
    params = dataclasses.replace(PARAMS, target_type="Click", use_bm25_reweighting=use_bm25_reweighting)
    implicit_als.train_implicit_als(CLICK_LOG, storage, params, embedding_type="test")
    assert set(storage.indexes["user_test"][1]) == {1, 2, 3}
    assert set(storage.indexes["item_test"][1]) == expected_items


def test_train_implicit_als_beats_top_pop_on_held_out_log(qdrant_env):
    """The exported embeddings generalise: trained on the first 50 days of a synthetic log with planted
    taste clusters (tests/synthetic.py), their top-20 finds the users' new cart items of the last 10 days
    at least twice as often as popularity does."""
    actions = make_actions(SMALL)
    ts = split_ts(SMALL)
    split = train_test(actions, ts)
    params = implicit_als.IALSParams(
        dim=32,
        max_iter=20,
        l2_alpha=0.1,
        conf_alpha=1,
        warm_user_threshold=3,
        use_bm25_reweighting=True,
        target_type="CartUpdate",
        random_state=0,
    )
    storage = QdrantEmbeddingStorage.start({})
    implicit_als.train_implicit_als(actions.filter(pl.col("timestamp") < ts), storage, params, embedding_type="iALS_test")
    user_ids, item_ids = split.user_ids.tolist(), split.item_ids.tolist()
    user_embeddings = dict(storage.get("user_iALS_test", user_ids))
    item_embeddings = dict(storage.get("item_iALS_test", item_ids))
    recall, _ = recall_ndcg_at_k(
        np.stack([user_embeddings[i] for i in user_ids]),
        np.stack([item_embeddings[i] for i in item_ids]),
        split,
    )
    top_pop_recall, _ = recall_ndcg_at_k(*top_pop_factors(split.train), split)
    # observed 3.6x (seeds 1-3: 3.7-4.4x); 2x is below half the observed margin
    assert recall >= 2 * top_pop_recall
