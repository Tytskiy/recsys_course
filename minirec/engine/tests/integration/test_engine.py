"""`RecommenderEngine.startup` -> `recommend` -> `shutdown` over a real snapshot,
fakeredis Valkey and in-memory Qdrant."""

from __future__ import annotations

import copy
import io
import tarfile
from typing import Any
from unittest.mock import Mock

import numpy as np
import orjson
import pytest
from engine.cg.mixing import CGSchema
from engine.parameters import ExperimentParams
from engine.recommender import RecommenderEngine

CONFIG: dict[str, Any] = {
    "experiment_manager_enabled": False,
    "embedding_storage": {"name": "QdrantEmbeddingStorage"},
    "feature_storage": {"name": "ValkeyFeatureStorage", "readonly": True},
    "runtime_profile": [
        {
            "type": "RuntimeIALS",
            "parameters": {
                "action_type": "AT_Purchase",
                "embedding_type": "iALS",
                "item_gramian_path": "item_iALS_gram.npy",
                "alpha": 40.0,
                "l2_alpha": 0.1,
                "max_history": 10,
            },
        }
    ],
    "candidate_generators": [
        {"type": "UserToItemKNN", "parameters": {"embedding_type": "iALS"}},
        {"type": "History", "parameters": {"action_type": "AT_Purchase"}},
        {"type": "TopPop", "parameters": {"path": "top_pop.json", "seed": 42}},
    ],
    "candidate_mixer": {
        "type": "SchemaCandidateMixer",
        "parameters": {
            "default_schema": {
                "ordered_cgs": [
                    {"cg_name": "iALS_userKNN", "num_candidates": 1},
                    {"cg_name": "History", "num_candidates": 5},
                    {"cg_name": "TopPop", "num_candidates": 1},
                ],
                "total_candidates": 10,
            },
        },
    },
    "ranking": [
        {
            "type": "FeatureReranker",
            "parameters": {"feature_key": "item:cart_update_turn_rate", "default": -1, "descending": True},
        }
    ],
    "feature_manager": {
        "type": "SimpleManager",
        "parameters": {
            "feature_schema": ["item:num_views", "item:num_cart_updates", "item:cart_update_turn_rate"],
        },
    },
}


@pytest.fixture
def engine(
    monkeypatch, tmp_path, rec_engine_files_dir, candidate_log_paths, feature_storage, embedding_storage, item_vectors
):
    # item embeddings (conftest): 1=[1,0,0], 2=[0,1,0], 3=[0,0,1], 4=[0.8,0,0]
    feature_storage.update({"user_history_AT_Purchase": [2]}, user_id=7)
    feature_storage.update({"num_views": 10, "num_cart_updates": 1}, item_id=2)
    feature_storage.update({"num_views": 10, "num_cart_updates": 5}, item_id=3)
    monkeypatch.setattr(
        RecommenderEngine,
        "setup_storages",
        staticmethod(lambda config: (feature_storage, embedding_storage)),
    )
    gramian = io.BytesIO()
    np.save(gramian, item_vectors.T @ item_vectors)
    archive = tmp_path / "archive.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, data in [("top_pop.json", b'{"3": 1}'), ("item_iALS_gram.npy", gramian.getvalue())]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    engine = RecommenderEngine.startup({**CONFIG, "engine_data_path": str(archive)})
    yield engine
    engine.candidate_logger.close()


def test_recommend_runs_full_pipeline(engine, feed_context):
    """RuntimeIALS turns purchase history [2] (confidence 40) into
    user_iALS = 40 / (1 + 39 + 0.1) * e2 ~ [0, 1, 0], so
    UserToItemKNN's top-1 is item 2 (History's 2 is then a duplicate) and TopPop
    adds its only item 3. Turn rates: item 2 = 1/10, item 3 = 5/10 -> 3 ranks first."""
    candidates = engine.recommend(10, feed_context)

    assert [(c.item_id, c.factors["source_cg"]) for c in candidates] == [
        (3, "TopPop"),
        (2, "iALS_userKNN"),
    ]


def test_experiment_schema_overrides_configured_schema(engine, feed_context, monkeypatch):
    schema = CGSchema.model_validate(
        {"ordered_cgs": [{"cg_name": "TopPop", "num_candidates": 1}], "total_candidates": 1}
    )
    params = ExperimentParams(filter_history=False, cg_schema=schema)
    monkeypatch.setattr(engine.exp_manager, "resolve", lambda user_id, source_type: params)

    candidates = engine.recommend(10, feed_context)

    assert [c.item_id for c in candidates] == [3]


@pytest.mark.parametrize(
    "filter_history, expected",
    [
        pytest.param(False, [3, 2], id="off_keeps_purchased"),
        pytest.param(True, [3], id="on_drops_purchased_item_2"),
    ],
)
def test_filter_history_is_a_per_request_parameter(engine, feed_context, monkeypatch, filter_history, expected):
    params = ExperimentParams(filter_history=filter_history, cg_schema=None)
    monkeypatch.setattr(engine.exp_manager, "resolve", lambda user_id, source_type: params)

    candidates = engine.recommend(10, feed_context)

    assert [c.item_id for c in candidates] == expected


def test_startup_rejects_schema_with_unknown_cg(monkeypatch, rec_engine_files_dir, candidate_log_paths):
    config = copy.deepcopy(CONFIG)
    config["candidate_generators"] = config["candidate_generators"][:2]
    config["runtime_profile"] = []
    config["engine_data_path"] = None
    # the failed startup closes its storages, so they must not be the shared fixtures
    monkeypatch.setattr(RecommenderEngine, "setup_storages", staticmethod(lambda config: (Mock(), Mock())))

    with pytest.raises(ValueError, match="TopPop"):
        RecommenderEngine.startup(config)


def test_recommend_logs_ranked_candidates(engine, feed_context, candidate_log_paths):
    engine.recommend(10, feed_context)

    records = [orjson.loads(line) for line in candidate_log_paths[0].read_bytes().splitlines()]
    assert [[c["item_id"] for c in r["candidates"]] for r in records] == [[3, 2]]


def test_shutdown_closes_storages(engine, monkeypatch):
    closed = []
    monkeypatch.setattr(engine.feature_storage, "close", lambda: closed.append("feature"))
    monkeypatch.setattr(engine.embedding_storage, "close", lambda: closed.append("embedding"))

    engine.shutdown()

    assert sorted(closed) == ["embedding", "feature"]
