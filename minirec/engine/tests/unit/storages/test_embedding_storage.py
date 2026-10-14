"""Qdrant embedding storage (task `embedding_storage`), on an in-memory Qdrant client. Pinned:
`build_embedding_index` creates a collection with the given metric and replaces an existing one, and
rejects an empty list; `delete_embedding_index` drops it (a missing one is fine); `upsert_embeddings`
inserts and overwrites by id; `search` returns the top `k` (id, score) by the metric; `score` scores
exactly the requested ids that exist; `get` returns (id, vector) for the existing ids in request order.
"""

from __future__ import annotations

import numpy as np
import pytest
from engine.interfaces import EmbeddingMetric
from engine.storages import QdrantEmbeddingStorage

ITEMS = "item_test"
VECTORS = {
    1: [1.0, 0.0, 0.0],
    2: [0.0, 1.0, 0.0],
    3: [0.0, 0.0, 1.0],
    4: [0.5, 0.0, 0.0],
    5: [3.0, 3.0, 0.0],
}
QUERY = np.array([1.0, 0.5, 0.0])


def build(storage: QdrantEmbeddingStorage, metric: EmbeddingMetric = "Dot", vectors: dict = VECTORS) -> None:
    storage.build_embedding_index(ITEMS, [(i, np.array(v)) for i, v in vectors.items()], metric=metric)


@pytest.fixture
def storage(qdrant_env):
    instance = QdrantEmbeddingStorage.start({})
    build(instance)
    yield instance
    instance.close()


def ids_and_scores(hits: list[tuple[int, float]]) -> tuple[list[int], list[float]]:
    return [int(i) for i, _ in hits], [float(s) for _, s in hits]


@pytest.mark.parametrize(
    ("metric", "k", "expected_ids", "expected_scores"),
    [
        pytest.param("Dot", 2, [5, 1], [4.5, 1.0], id="dot_top_k"),
        pytest.param("Cosine", 1, [5], [4.5 / np.sqrt(18 * 1.25)], id="cosine"),
        pytest.param("Euclid", 2, [1, 4], [0.5, np.sqrt(0.5)], id="euclid_nearest_first"),
    ],
)
def test_search(qdrant_env, metric, k, expected_ids, expected_scores):
    storage = QdrantEmbeddingStorage.start({})
    build(storage, metric)
    ids, scores = ids_and_scores(storage.search(ITEMS, QUERY, k=k))
    assert ids == expected_ids
    assert scores == pytest.approx(expected_scores)


def test_search_k_past_collection_returns_all(storage):
    ids, _ = ids_and_scores(storage.search(ITEMS, QUERY, k=10))
    assert sorted(ids) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    ("request_ids", "expected"),
    [
        pytest.param([], {}, id="empty"),
        pytest.param([999, 1000], {}, id="missing_ids"),
        pytest.param([3, 5, 999, 1], {3: 0.0, 5: 4.5, 1: 1.0}, id="existing_ids_only"),
    ],
)
def test_score(storage, request_ids, expected):
    hits = storage.score(ITEMS, QUERY, request_ids)
    assert len(hits) == len(expected)
    assert dict(hits) == pytest.approx(expected)


def test_score_more_ids_than_a_default_page(storage):
    storage.upsert_embeddings(ITEMS, [(100 + i, np.array([float(i), 0.0, 0.0])) for i in range(12)])
    # QUERY = [1, 0.5, 0] -> score(100 + i) = i
    assert dict(storage.score(ITEMS, QUERY, [100 + i for i in range(12)])) == pytest.approx(
        {100 + i: float(i) for i in range(12)}
    )


@pytest.mark.parametrize(
    ("request_ids", "expected_ids"),
    [
        pytest.param([], [], id="empty"),
        pytest.param([999, 1000], [], id="missing_ids"),
        pytest.param([3, 999, 1], [3, 1], id="request_order_missing_dropped"),
    ],
)
def test_get(storage, request_ids, expected_ids):
    hits = storage.get(ITEMS, request_ids)
    assert [int(i) for i, _ in hits] == expected_ids
    for item_id, vector in hits:
        np.testing.assert_allclose(vector, VECTORS[item_id])


def test_upsert_inserts_and_overwrites(storage):
    storage.upsert_embeddings(ITEMS, [(9, np.array([1.0, 1.0, 1.0])), (1, np.array([0.0, 0.0, 2.0]))])
    fetched = dict(storage.get(ITEMS, [1, 2, 9]))
    np.testing.assert_allclose(fetched[9], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(fetched[1], [0.0, 0.0, 2.0])
    np.testing.assert_allclose(fetched[2], VECTORS[2])


def test_upsert_more_than_one_batch(storage):
    """More points than one upload batch (64) all arrive."""
    storage.upsert_embeddings(ITEMS, [(100 + i, np.array([float(i), 0.0, 1.0])) for i in range(150)])
    assert len(storage.get(ITEMS, list(range(100, 250)))) == 150


def test_build_replaces_existing_collection(storage):
    build(storage, vectors={42: [0.0, 1.0, 0.0]})
    assert [i for i, _ in storage.get(ITEMS, [1, 42])] == [42]


def test_build_with_new_dimension(storage):
    """A rebuilt collection takes the dimension of the new vectors."""
    build(storage, vectors={7: [1.0, 2.0]})
    ids, scores = ids_and_scores(storage.search(ITEMS, np.array([1.0, 1.0]), k=1))
    assert (ids, scores) == ([7], [pytest.approx(3.0)])


def test_build_rejects_empty_embeddings(storage):
    with pytest.raises(ValueError):
        storage.build_embedding_index("empty", [], metric="Dot")


def test_delete_embedding_index(storage):
    storage.delete_embedding_index(ITEMS)
    assert not storage.client.collection_exists(ITEMS)
    storage.delete_embedding_index(ITEMS)
