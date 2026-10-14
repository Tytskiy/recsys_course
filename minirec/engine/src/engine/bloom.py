from __future__ import annotations

import base64
import hashlib

import numpy as np
from rbloom import Bloom

EXPECTED_ITEMS = 2000
FALSE_POSITIVE_RATE = 1e-3


class BloomToken:
    def __init__(self, token: str | None = None) -> None:
        self.bloom: Bloom
        if token is None:
            self.bloom = Bloom(EXPECTED_ITEMS, FALSE_POSITIVE_RATE, BloomToken.hash)
        else:
            self.bloom = Bloom.load_bytes(base64.b64decode(token), BloomToken.hash)

    @staticmethod
    def hash(obj: int | np.integer) -> int:
        # int() makes np.int64(5) and 5 hash alike; decimal text fits any uint64 ID.
        digest = hashlib.sha256(str(int(obj)).encode()).digest()
        return int.from_bytes(digest[:8], "big")

    def add(self, iid: int | np.integer) -> None:
        self.bloom.add(iid)

    def __contains__(self, iid: int | np.integer) -> bool:
        return iid in self.bloom

    def encode(self) -> str:
        return base64.b64encode(self.bloom.save_bytes()).decode("ascii")
