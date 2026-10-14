from pathlib import Path

import polars as pl
import pytest
from engine.storages import QdrantEmbeddingStorage
from qdrant_client import QdrantClient

TEST_DATA_PATH = Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def dm_product():
    dataframe = pl.read_parquet(TEST_DATA_PATH / "test_dm_product.parquet")
    return dataframe.sort("item_id")


@pytest.fixture(scope="session")
def dm_action():
    dataframe = pl.read_parquet(TEST_DATA_PATH / "test_dm_action.parquet")
    return dataframe.sort("user_id", "item_id")


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
