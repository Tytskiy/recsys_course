"""User-to-item KNN CG (task `implicit_als`, engine half). Pinned: the CG searches `item_<type>` with
`user_profile["user_<type>"]` for `num_candidates` hits and yields them in search order (none for 0); no user embedding
means no candidates.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from engine.cg import UserToItemKNN
from engine.cg.user_to_item_knn import UserToItemKNNConfig
from engine.interfaces import RecContext
from tests.fakes import DictEmbeddingStorage

ITEM_VECTORS = {
    1: np.array([1.0, 0.0, 0.0]),
    2: np.array([0.0, 1.0, 0.0]),
    3: np.array([0.0, 0.0, 1.0]),
    4: np.array([0.8, 0.0, 0.0]),
}
OTHER_VECTORS = {item_id: -vector for item_id, vector in ITEM_VECTORS.items()}


def storage() -> DictEmbeddingStorage:
    """Exact dot-product search over `item_iALS` and `item_BPR`."""
    return DictEmbeddingStorage({"item_iALS": ITEM_VECTORS, "item_BPR": OTHER_VECTORS})


def test_start_component(deps):
    candgen = UserToItemKNN.start_component(
        UserToItemKNNConfig(embedding_type="iALS"),
        dataclasses.replace(deps, embedding_storage=storage()),
    )
    assert isinstance(candgen, UserToItemKNN)
    assert candgen.name == "iALS_userKNN"


@pytest.mark.parametrize(
    ("num_candidates", "expected"),
    [
        pytest.param(4, [1, 4, 2, 3], id="all_items_by_dot_product"),
        pytest.param(2, [1, 4], id="top_num_candidates_only"),
        pytest.param(0, [], id="zero_candidates"),
    ],
)
def test_candidates_follow_search_order(
    num_candidates, expected, feed_context: RecContext
):
    candgen = UserToItemKNN(
        embedding_type="iALS", embedding_storage=storage()
    )
    feed_context.user_profile["user_iALS"] = np.array([0.6, 0.2, 0.0])
    candidates = list(
        candgen.extract_candidates(num_candidates=num_candidates, context=feed_context)
    )
    assert [candidate.item_id for candidate in candidates] == expected


def test_embedding_type_selects_profile_key_and_collection(feed_context: RecContext):
    candgen = UserToItemKNN(embedding_type="BPR", embedding_storage=storage())
    feed_context.user_profile["user_iALS"] = np.array([0.6, 0.2, 0.0])
    feed_context.user_profile["user_BPR"] = np.array([-0.6, -0.2, 0.0])
    candidates = list(candgen.extract_candidates(num_candidates=2, context=feed_context))
    assert [candidate.item_id for candidate in candidates] == [1, 4]
    assert candgen.name == "BPR_userKNN"


def test_embedding_missing(feed_context: RecContext):
    candgen = UserToItemKNN(
        embedding_type="iALS", embedding_storage=storage()
    )
    candidates = list(
        candgen.extract_candidates(num_candidates=3, context=feed_context)
    )
    assert candidates == []
