from __future__ import annotations

import numpy as np
import pytest
from engine.features import SimpleManager
from engine.interfaces import Candidate


@pytest.mark.parametrize(
    ("schema", "error"),
    [
        pytest.param(["item:num_view"], "Unknown features", id="typo_in_name"),
        pytest.param(["item:cart_update_turn_rate_7d"], "Unknown features", id="uncatalogued_transform"),
        pytest.param(
            ["item:num_views", "item:num_cart_updates", "item:cart_update_turn_rate_30d"],
            "needs",
            id="transform_without_its_statistics",
        ),
    ],
)
def test_rejects_invalid_schema(schema, error, feature_storage, empty_embedding_storage):
    with pytest.raises(ValueError, match=error):
        SimpleManager(schema, feature_storage, empty_embedding_storage)


def test_runtime_profile_embedding_is_not_overwritten(feature_storage, embedding_storage, feed_context):
    """A runtime profile runs before the feature manager; its fresh `user_iALS`
    wins over the offline one stored in Qdrant."""
    manager = SimpleManager(["iALS_score"], feature_storage, embedding_storage)
    fresh = np.array([0.0, 0.0, 1.0])
    feed_context.user_profile["user_iALS"] = fresh

    context = manager.hydrate_context(feed_context)

    np.testing.assert_array_equal(context.user_profile["user_iALS"], fresh)


def test_item_to_history_is_mean_dot_product(feature_storage, embedding_storage, feed_context):
    manager = SimpleManager(["iALS_to_history"], feature_storage, embedding_storage)
    feed_context.user_history = {"AT_Purchase": [1, 2]}
    candidates = [Candidate(item_id=3), Candidate(item_id=4)]

    hydrated = manager.hydrate_candidates(candidates, feed_context)

    scores = {c.item_id: c.factors["iALS_to_history"] for c in hydrated}
    assert scores == pytest.approx({3: 0.0, 4: 0.4})
