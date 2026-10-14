from __future__ import annotations

import io
import tarfile

import pytest
from engine.utils import EngineData


def _write_archive(path, members: dict[str, bytes]) -> str:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return str(path)


def test_clear_keeps_other_instance_files(rec_engine_files_dir, tmp_path):
    """Hot reload: the old engine's `clear` runs after the new engine extracted
    its snapshot, and must not delete the new engine's files."""
    archive = _write_archive(tmp_path / "archive.tar.gz", {"top_pop.json": b"{}"})
    old, new = EngineData(archive), EngineData(archive)

    old.clear()

    assert new.read_json("top_pop.json") == {}


def test_rejects_archive_member_outside_dir(rec_engine_files_dir, tmp_path):
    archive = _write_archive(tmp_path / "evil.tar.gz", {"../escape.json": b"{}"})

    with pytest.raises(tarfile.OutsideDestinationError):
        EngineData(archive)
