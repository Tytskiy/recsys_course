from __future__ import annotations

import os
from typing import Any

import orjson
import valkey

from engine.interfaces import ID, Factor, FeatureStorage
from engine.storages.hash_key import build_hash_key


class ValkeyFeatureStorage(FeatureStorage):
    def __init__(
        self,
        client: valkey.Valkey | None = None,
        readonly: bool = False,
    ):
        self.readonly = readonly
        self.client = client or self.get_client()

    @staticmethod
    def get_client():
        host = os.getenv("VALKEY_HOST", "localhost")
        port = int(os.getenv("VALKEY_PORT", "6379"))
        return valkey.Valkey(host=host, port=port)

    @classmethod
    def start(cls, config: dict[str, Any]) -> ValkeyFeatureStorage:
        """Build a storage from `get_client()`; `config["readonly"]` is required."""
        # TODO: Your solution

    # TODO: Your solution
