from __future__ import annotations

import os
from itertools import batched
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
    VectorParams,
)

from engine.interfaces import (
    ID,
    Embedding,
    EmbeddingMetric,
    EmbeddingStorage,
)

METRICS = {
    "Dot": Distance.DOT,
    "Cosine": Distance.COSINE,
    "Euclid": Distance.EUCLID,
}
HNSW_M = 32
HNSW_EF_CONSTRUCT = 100
UPLOAD_BATCH_SIZE = 64


class QdrantEmbeddingStorage(EmbeddingStorage):
    def __init__(self, client: QdrantClient | None = None):
        self.client = client or self.get_client()

    @staticmethod
    def get_client() -> QdrantClient:
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_HTTP_PORT", "6333"))
        return QdrantClient(host=host, port=port)

    @classmethod
    def start(cls, config: dict[str, Any]) -> QdrantEmbeddingStorage:
        return cls()

    def close(self) -> None:
        self.client.close()

    # TODO: Your solution
