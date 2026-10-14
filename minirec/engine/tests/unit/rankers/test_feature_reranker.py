from __future__ import annotations

from engine.interfaces import Candidate
from engine.ranking.feature_ranker import FeatureReranker


def test_by_feature_reranker_orders_by_score_and_index(feed_context):
    reranker = FeatureReranker(feature_key="score", default=-1, descending=True)
    candidates = [
        Candidate(item_id=1, factors={"score": 0.5}),
        Candidate(item_id=2, factors={}),
        Candidate(item_id=3, factors={"score": 0.5}),
        Candidate(item_id=4, factors={"score": 1.0}),
    ]

    reranked = reranker.rerank(candidates, feed_context)
    assert [c.item_id for c in reranked] == [4, 1, 3, 2]
