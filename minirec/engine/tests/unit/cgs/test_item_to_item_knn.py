from __future__ import annotations

import pytest
from engine.cg import ItemToItemKNN


@pytest.mark.parametrize(
    "context_item_ids",
    [
        pytest.param([], id="no_context_items"),
        pytest.param([999], id="context_item_without_embedding"),
    ],
)
def test_yields_nothing_without_context_embeddings(embedding_storage, feed_context, context_item_ids):
    candgen = ItemToItemKNN(embedding_type="iALS", embedding_storage=embedding_storage)
    feed_context.context_item_ids = context_item_ids

    assert list(candgen.extract_candidates(num_candidates=3, context=feed_context)) == []


def test_searches_with_mean_context_embedding(embedding_storage, feed_context):
    candgen = ItemToItemKNN(embedding_type="iALS", embedding_storage=embedding_storage)
    feed_context.context_item_ids = [1, 2]

    candidates = list(candgen.extract_candidates(num_candidates=3, context=feed_context))

    assert {c.item_id for c in candidates[:2]} == {1, 2}
    assert candidates[2].item_id == 4
