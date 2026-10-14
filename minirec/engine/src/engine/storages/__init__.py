from __future__ import annotations

from engine.interfaces import EmbeddingStorage, FeatureStorage
from engine.storages.embedding_storage import QdrantEmbeddingStorage
from engine.storages.feature_storage import ValkeyFeatureStorage

EMBEDDING_STORAGES: dict[str, type[EmbeddingStorage]] = {
    "QdrantEmbeddingStorage": QdrantEmbeddingStorage,
}
FEATURE_STORAGES: dict[str, type[FeatureStorage]] = {
    "ValkeyFeatureStorage": ValkeyFeatureStorage,
}


__all__ = [
    "QdrantEmbeddingStorage",
    "ValkeyFeatureStorage",
    "EMBEDDING_STORAGES",
    "FEATURE_STORAGES",
]
