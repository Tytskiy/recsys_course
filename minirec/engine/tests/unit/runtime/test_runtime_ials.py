"""Runtime iALS user step (task `runtime_als`). Pinned: `collect_from_history` returns the stored
`item_<type>` embeddings of the distinct history items sorted by id, with confidence `alpha * count`, and
skips items without an embedding; `solve` is the exact iALS user step with the full item Gramian;
`hydrate_context` solves over the first `max_history` items of `user_history[action_type]` and writes
`user_profile["user_<type>"]`, or nothing when no item has an embedding.
"""

from __future__ import annotations

import numpy as np
import pytest
from engine.interfaces import EmbeddingStorage, RecContext
from engine.runtime_profile.runtime_ials import RuntimeIALS
from tests.fakes import DictEmbeddingStorage

E1, E2, E4 = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.8, 0.0, 0.0]
ITEM_VECTORS = {1: E1, 2: E2, 3: [0.0, 0.0, 1.0], 4: E4}


@pytest.fixture
def embedding_storage() -> DictEmbeddingStorage:
    return DictEmbeddingStorage({"item_iALS": {i: np.array(v) for i, v in ITEM_VECTORS.items()}})


def make_runtime(
    storage: EmbeddingStorage, alpha: float = 1.0, max_history: int | None = None, embedding_type: str = "iALS"
) -> RuntimeIALS:
    return RuntimeIALS(
        embedding_storage=storage,
        action_type="AT_View",
        item_gramian=2.0 * np.eye(3, dtype=np.float32),
        embedding_type=embedding_type,
        alpha=alpha,
        l2_alpha=0.5,
        max_history=max_history,
    )


@pytest.mark.parametrize(
    ("history", "alpha", "expected_factors", "expected_confidence"),
    [
        pytest.param([4, 1, 2, 4, 1, 4], 40.0, [E1, E2, E4], [80.0, 40.0, 120.0], id="counts_scaled_by_alpha"),
        pytest.param([2, 1], 1.0, [E1, E2], [1.0, 1.0], id="sorted_by_item_id"),
        pytest.param([1, 2, 999], 5.0, [E1, E2], [5.0, 5.0], id="item_without_embedding_skipped"),
        pytest.param([100, 200], 40.0, [], [], id="no_known_items"),
    ],
)
def test_collect_from_history(embedding_storage, history, alpha, expected_factors, expected_confidence):
    """Confidence follows the offline `implicit` convention `alpha * count`."""
    factors, confidence = make_runtime(embedding_storage, alpha).collect_from_history(history)
    np.testing.assert_allclose(np.reshape(factors, (-1, 3)), np.reshape(expected_factors, (-1, 3)), atol=1e-6)
    np.testing.assert_allclose(confidence, expected_confidence)


@pytest.mark.parametrize(
    ("factors", "confidence", "expected"),
    [
        # A = 2I + diag(3-1, 1-1, 0) + 0.5I = diag(4.5, 2.5, 2.5); b = 3*e1 + 1*e2 -> u = [2/3, 0.4, 0]
        pytest.param([E1, E2], [3.0, 1.0], [2 / 3, 0.4, 0.0], id="two_items"),
        pytest.param([E1], [1.0], [0.4, 0.0, 0.0], id="confidence_one_uses_gramian_only"),
        # A = 2.5I + 9 * 0.64 e1e1^T = diag(8.26, 2.5, 2.5); b = 10 * 0.8 e1 = 8 e1 -> u = [8/8.26, 0, 0]
        pytest.param([E4], [10.0], [8 / 8.26, 0.0, 0.0], id="scaled_item"),
    ],
)
def test_solve(embedding_storage, factors, confidence, expected):
    """The iALS user step: (Y^T Y + F^T (C - I) F + l2 I) u = F^T C with the full item Gramian Y^T Y."""
    runtime = make_runtime(embedding_storage)
    user = runtime.solve(np.array(factors, dtype=np.float32), np.array(confidence, dtype=np.float32))
    np.testing.assert_allclose(user, expected, atol=1e-5)


@pytest.mark.parametrize(
    ("history", "max_history", "expected"),
    [
        pytest.param([1, 2], None, [0.4, 0.4, 0.0], id="full_history"),
        pytest.param([1, 2], 1, [0.4, 0.0, 0.0], id="truncated_to_max_history"),
    ],
)
def test_hydrate_context_writes_user_embedding(embedding_storage, feed_context: RecContext, history, max_history, expected):
    """The embedding goes to `user_profile["user_<embedding_type>"]`, where `UserToItemKNN` reads it."""
    feed_context.user_history = {"AT_View": history, "AT_Click": [4]}
    context = make_runtime(embedding_storage, max_history=max_history).hydrate_context(feed_context)
    np.testing.assert_allclose(np.asarray(context.user_profile["user_iALS"]), expected, atol=1e-5)


def test_embedding_type_names_collection_and_profile_key(feed_context: RecContext):
    storage = DictEmbeddingStorage({"item_iALS": {1: np.array(E1)}, "item_BPR": {1: np.array(E2)}})
    feed_context.user_history = {"AT_View": [1]}
    context = make_runtime(storage, embedding_type="BPR").hydrate_context(feed_context)
    # item_BPR[1] = e2 with confidence 1 -> u = e2 / (2 + 0.5)
    np.testing.assert_allclose(np.asarray(context.user_profile["user_BPR"]), [0.0, 0.4, 0.0], atol=1e-5)
    assert "user_iALS" not in context.user_profile


def test_collect_fetches_each_item_once(embedding_storage):
    make_runtime(embedding_storage).collect_from_history([4, 1, 4, 4])
    assert [(table, sorted(ids)) for table, ids in embedding_storage.requests] == [("item_iALS", [1, 4])]


def test_hydrate_context_without_known_items_writes_nothing(embedding_storage, feed_context: RecContext):
    feed_context.user_history = {"AT_View": [100, 200]}
    context = make_runtime(embedding_storage).hydrate_context(feed_context)
    assert "user_iALS" not in context.user_profile
