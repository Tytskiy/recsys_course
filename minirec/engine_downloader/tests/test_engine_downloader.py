from __future__ import annotations

import boto3
import engine_downloader as downloader_module
import pytest
from botocore.exceptions import EndpointConnectionError
from moto import mock_aws

BUCKET = "rec-engine"
KEY = "service/v1/archive.tar.gz"


class StopLoop(Exception):
    pass


@pytest.fixture
def downloader(monkeypatch, tmp_path):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://s3.amazonaws.com")
    monkeypatch.setenv(downloader_module.ARCHIVE_S3_PATH_ENV, f"s3://{BUCKET}/{KEY}")
    monkeypatch.setenv(
        downloader_module.LOCAL_ARCHIVE_PATH_ENV, str(tmp_path / "archive.tar.gz")
    )
    monkeypatch.setenv(
        downloader_module.UPDATE_MARKER_PATH_ENV, str(tmp_path / "archive.version.json")
    )
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        yield downloader_module.EngineDownloader()


def upload(downloader, body: bytes) -> None:
    downloader.client.put_object(Bucket=BUCKET, Key=KEY, Body=body)


def test_sync_downloads_new_archive_once(downloader) -> None:
    upload(downloader, b"v1")

    assert downloader.sync_once() is True
    assert downloader.archive_path.read_bytes() == b"v1"
    assert sorted(p.name for p in downloader.archive_path.parent.iterdir()) == [
        "archive.tar.gz",
        "archive.version.json",
    ]
    assert downloader.sync_once() is False


def test_periodic_run_survives_sync_errors(downloader, monkeypatch) -> None:
    """A missing snapshot or an unreachable S3 is logged and retried, not fatal."""

    def sync_once():
        raise EndpointConnectionError(endpoint_url="http://s3:5000")

    def sleep(seconds):
        raise StopLoop

    monkeypatch.setattr(downloader, "sync_once", sync_once)
    monkeypatch.setattr(downloader_module.time, "sleep", sleep)
    with pytest.raises(StopLoop):
        downloader.run(periodic=True)
