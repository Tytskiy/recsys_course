from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import engine.candidate_logger as cl
import fakeredis
import numpy as np
import pytest
from engine.bloom import BloomToken
from engine.interfaces import ComponentDeps, RecContext
from engine.storages import QdrantEmbeddingStorage, ValkeyFeatureStorage
from engine.utils import EngineData
from qdrant_client import QdrantClient
from tests.fakes import RecordingCandidateLogger, UnusedEmbeddingStorage, UnusedFeatureStorage

TEST_DATA_PATH = Path(__file__).parent / "test_data"


_FACTOR_SCHEMA_PATH = TEST_DATA_PATH / "factor_schema.json"
_SAMPLE_ROW_PATH = TEST_DATA_PATH / "sample_row.csv"

_CANONICAL_FACTOR_SCHEMA: list[tuple[str, bool]] = [
    (name, is_ctx) for name, is_ctx in json.loads(_FACTOR_SCHEMA_PATH.read_text())
]
_CANONICAL_FEATURE_NAMES = [name for name, _ in _CANONICAL_FACTOR_SCHEMA]

with _SAMPLE_ROW_PATH.open() as f:
    _SAMPLE_FACTORS: dict[str, float] = {
        k: float(v) for k, v in next(csv.DictReader(f)).items()
    }


@pytest.fixture
def deps(engine_data) -> ComponentDeps:
    """`ComponentDeps` with storages that must stay unused; swap one in with
    `dataclasses.replace(deps, feature_storage=...)`."""
    return ComponentDeps(
        feature_storage=UnusedFeatureStorage(),
        embedding_storage=UnusedEmbeddingStorage(),
        engine_data=engine_data,
        candidate_logger=RecordingCandidateLogger(),
    )


@pytest.fixture
def candidate_log_paths(tmp_path, monkeypatch) -> tuple[Path, Path]:
    """Point the engine's candidate logs (read by `RecommenderEngine.startup`) into tmp_path."""
    paths = tmp_path / "candidates.jsonl", tmp_path / "cg_candidates.jsonl"
    monkeypatch.setattr(cl, "CANDIDATES_LOG_PATH", str(paths[0]))
    monkeypatch.setattr(cl, "CG_CANDIDATES_LOG_PATH", str(paths[1]))
    return paths


@pytest.fixture
def stats_ranker_model(rec_engine_files_dir) -> str:
    """Place the real stats-only CatBoost ranker (tests/test_data/ranker.cbm)
    into rec_engine_files_dir and return its relative model key for engine_data
    lookup."""
    rec_engine_files_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TEST_DATA_PATH / "ranker.cbm", rec_engine_files_dir / "ranker.cbm")
    return "ranker.cbm"


@pytest.fixture
def canonical_factor_schema() -> list[tuple[str, bool]]:
    """`(factor_name, is_context_factor)` list mirroring `ranker.cbm`'s
    `feature_names_`."""
    return list(_CANONICAL_FACTOR_SCHEMA)


@pytest.fixture
def sample_factors() -> dict[str, float]:
    """A real `(item_id, factors)` row from candidates.jsonl."""
    return dict(_SAMPLE_FACTORS)


@pytest.fixture
def sample_factor_row() -> list[float]:
    """`sample_factors` flattened into the positional row CatBoost expects
    (column order = `feature_names_`)."""
    return [_SAMPLE_FACTORS[name] for name in _CANONICAL_FEATURE_NAMES]


@pytest.fixture
def expected_raw_score() -> float:
    """`m.predict([sample_factor_row], "RawFormulaVal")[0]`."""
    return 0.142681


@pytest.fixture
def expected_prob_positive() -> float:
    """`m.predict([sample_factor_row], "Probability")[0, 1]`."""
    return 0.535610


@pytest.fixture(scope="function")
def feed_context() -> RecContext:
    context = RecContext.from_request(
        user_id=7,
        current_time=None,
        context_item_ids=None,
        item_id_filter=BloomToken(),
        source_type="ST_Feed",
    )
    context.user_profile["user_bias"] = 1.0
    return context


@pytest.fixture(scope="function")
def valkey_env(monkeypatch):
    fake_client = fakeredis.FakeValkey()
    monkeypatch.setattr(
        ValkeyFeatureStorage,
        "get_client",
        staticmethod(lambda: fake_client),
    )
    yield fake_client
    fake_client.close()


@pytest.fixture(scope="function")
def feature_storage(valkey_env):
    instance = ValkeyFeatureStorage.start({"readonly": False})
    yield instance


@pytest.fixture(scope="function")
def readonly_storage(valkey_env):
    instance = ValkeyFeatureStorage.start({"readonly": True})
    yield instance


@pytest.fixture(scope="function")
def qdrant_env(monkeypatch):
    client = QdrantClient(location=":memory:")
    monkeypatch.setattr(
        QdrantEmbeddingStorage,
        "get_client",
        staticmethod(lambda host=None, port=None: client),
    )
    yield client
    client.close()


@pytest.fixture(scope="function")
def rec_engine_files_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "engine-files"
    monkeypatch.setenv("REC_ENGINE_FILES_DIR", data_dir.as_posix())
    return data_dir


@pytest.fixture(scope="function")
def engine_data(rec_engine_files_dir):
    return EngineData(dir_path=rec_engine_files_dir)


@pytest.fixture(scope="function")
def item_vectors():
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.8, 0.0, 0.0],
    ]).astype(np.float32)


@pytest.fixture(scope="function")
def user_vectors():
    return np.array([[0.6, 0.2, 0.0]]).astype(np.float32)


@pytest.fixture(scope="function")
def empty_embedding_storage(qdrant_env):
    embedding_storage = QdrantEmbeddingStorage.start({})
    yield embedding_storage
    embedding_storage.close()


@pytest.fixture(scope="function")
def embedding_storage(qdrant_env, item_vectors, user_vectors):
    item_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    user_ids = np.array([7], dtype=np.int64)

    embedding_storage = QdrantEmbeddingStorage.start({})
    embedding_storage.build_embedding_index(
        "item_iALS",
        embeddings=[
            (int(item_id), vector) for item_id, vector in zip(item_ids, item_vectors)
        ],
        metric="Dot",
    )
    embedding_storage.build_embedding_index(
        "user_iALS",
        embeddings=[
            (int(item_id), vector) for item_id, vector in zip(user_ids, user_vectors)
        ],
        metric="Dot",
    )

    yield embedding_storage

    embedding_storage.delete_embedding_index("item_iALS")
    embedding_storage.delete_embedding_index("user_iALS")
    embedding_storage.close()
