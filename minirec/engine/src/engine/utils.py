import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sps
from catboost import CatBoost
from orjson import loads


class EngineData:
    def __init__(self, archive_path: str | None = None, dir_path: Path | None = None):
        if dir_path is None:
            root = Path(os.getenv("REC_ENGINE_FILES_DIR", "rec-engine-files"))
            root.mkdir(parents=True, exist_ok=True)
            dir_path = Path(tempfile.mkdtemp(prefix="engine-", dir=root))
        dir_path.mkdir(parents=True, exist_ok=True)
        self.dir_path = dir_path

        if archive_path is not None:
            with tarfile.open(archive_path, "r:*") as tar:
                tar.extractall(self.dir_path, filter="data")

    def clear(self) -> None:
        shutil.rmtree(self.dir_path, ignore_errors=True)

    def resolve_path(self, subpath: str | Path) -> Path:
        return self.dir_path / subpath

    def read_json(self, subpath: str | Path) -> Any:
        path = self.resolve_path(subpath)
        with open(path, "rb") as f:
            return loads(f.read())

    def read_cbm(self, subpath: str | Path):
        path = self.resolve_path(subpath)
        return CatBoost().load_model(path)

    def read_np(self, subpath: str | Path):
        path = self.resolve_path(subpath)
        return np.load(path)

    def read_sparse_npz(self, subpath: str | Path):
        path = self.resolve_path(subpath)
        return sps.load_npz(path)
