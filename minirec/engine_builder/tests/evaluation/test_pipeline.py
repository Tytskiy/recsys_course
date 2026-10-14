import math
import time
from datetime import UTC, datetime

import polars as pl
import pytest
from engine.storages import QdrantEmbeddingStorage
from tasks.evaluation import evaluate_knn_cg
from tasks.evaluation.metric import Recall
from tasks.evaluation.pipeline import EvaluationPipeline, EvaluationResponse, Request, RequestItem
from tasks.models.implicit_als import IALSParams, train_implicit_als


def test_build_requests_request_time_is_utc_on_non_utc_host(monkeypatch):
    """Request time must not depend on the host time zone: CandidateLogger writes
    it with its offset and the pool builder labels actions in UTC epoch seconds."""
    monkeypatch.setenv("TZ", "Europe/Moscow")
    time.tzset()
    try:
        dm_action = pl.DataFrame({
            "dt": ["2024-01-01"],
            "request_id": [1],
            "user_id": [10],
            "item_id": [100],
            "source_type": ["ST_Feed"],
            "timestamp": [1_704_110_400],  # 2024-01-01 12:00:00 UTC
            "action_type": ["AT_View"],
        })
        [request] = EvaluationPipeline.build_requests(dm_action)
    finally:
        monkeypatch.undo()
        time.tzset()

    assert request.request_time == datetime(2024, 1, 1, 11, 59, 55, tzinfo=UTC)


def test_compute_batch_metric_without_positives_is_nan():
    """A slice with no positive of the requested action type has no defined mean."""
    request = Request(
        dt="2024-01-01",
        user_id=10,
        request_id=1,
        request_time=datetime(2024, 1, 1, tzinfo=UTC),
        source_type="ST_Feed",
        actions=[RequestItem(item_id=100, timestamp=0, action_type="AT_View")],
    )
    batch = [EvaluationResponse(request=request, candidates=[100])]

    value = EvaluationPipeline.compute_batch_metric(batch, Recall(10), "AT_Purchase")

    assert math.isnan(value)


@pytest.mark.usefixtures("qdrant_env")
def test_evaluate_knn_cg_finds_hits(dm_action: pl.DataFrame):
    """The tuning path of train_implicit_als.py: train iALS, then score the kNN CG
    on a held-out day. Hydrating the user embedding is what makes any hit possible."""
    train = dm_action.filter(pl.col("dt") < "2024-02-02")
    val = dm_action.filter(pl.col("dt") == "2024-02-02")
    storage = QdrantEmbeddingStorage.start({})
    train_implicit_als(train, storage, IALSParams(dim=16, max_iter=5, random_state=0), embedding_type="iALS")

    metrics = evaluate_knn_cg(val, storage, "iALS")

    assert metrics["Recall@50"] > 0
