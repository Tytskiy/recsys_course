from __future__ import annotations

from pathlib import Path

import experiments_server.app as app_module
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """The app on a fresh SQLite file in `tmp_path`."""
    monkeypatch.setenv(
        "EXPERIMENT_SQLITE_PATH", (tmp_path / "experiments.db").as_posix()
    )
    with TestClient(app_module.app) as client:
        yield client
