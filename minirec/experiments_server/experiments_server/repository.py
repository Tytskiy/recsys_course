from __future__ import annotations

import hashlib


class ExperimentRepository:
    def __init__(self, db_path: str):
        self.sqlite_db = f"sqlite:///{db_path}"
    
    @staticmethod
    def salted_hash(user_id: int, salt: str, mod: int) -> int:
        payload = f"{user_id}:{salt}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        hash_int = int.from_bytes(digest[:8], "big")
        return (hash_int % mod) + 1
