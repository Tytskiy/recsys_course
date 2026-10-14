from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import boto3
import click
from botocore.exceptions import BotoCoreError, ClientError

ARCHIVE_S3_PATH_ENV = "REC_ENGINE_ARCHIVE_S3_PATH"
LOCAL_ARCHIVE_PATH_ENV = "REC_ENGINE_LOCAL_ARCHIVE_PATH"
UPDATE_MARKER_PATH_ENV = "REC_ENGINE_UPDATE_MARKER_PATH"

DEFAULT_POLL_INTERVAL_SEC = 30.0

logger = logging.getLogger(__name__)


class EngineDownloader:
    def __init__(self):
        self.client = self.get_client()
        s3_path = os.getenv(
            ARCHIVE_S3_PATH_ENV, "s3://rec-engine/service/v1/archive.tar.gz"
        )
        self.s3_bucket, self.s3_key = self.parse_s3_path(s3_path)

        self.archive_path = Path(
            os.getenv(LOCAL_ARCHIVE_PATH_ENV, "/shared/archive.tar.gz")
        )
        self.marker_path = Path(
            os.getenv(UPDATE_MARKER_PATH_ENV, "/shared/archive.version.json")
        )

    @staticmethod
    def parse_s3_path(url: str) -> tuple[str, str]:
        parsed = urlparse(url)
        return parsed.netloc, parsed.path.lstrip("/")

    @staticmethod
    def get_client():
        endpoint_url = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:9002")
        access_key = os.environ.get("AWS_ACCESS_KEY_ID", "admin")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "password")
        return boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    @staticmethod
    def read_marker_last_modified(marker_path: Path) -> datetime | None:
        if not marker_path.exists():
            return None
        payload = json.loads(marker_path.read_text())
        return datetime.fromisoformat(payload["last_modified"])

    def sync_once(self) -> bool:
        head = self.client.head_object(Bucket=self.s3_bucket, Key=self.s3_key)
        remote_last_modified: datetime = head["LastModified"]
        local_last_modified = self.read_marker_last_modified(self.marker_path)

        if (
            self.archive_path.exists()
            and local_last_modified is not None
            and local_last_modified >= remote_last_modified
        ):
            return False

        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        self.marker_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_archive = self.archive_path.with_name(self.archive_path.name + ".part")
        self.client.download_file(self.s3_bucket, self.s3_key, tmp_archive.as_posix())
        os.replace(tmp_archive, self.archive_path)

        marker = {
            "bucket": self.s3_bucket,
            "key": self.s3_key,
            "etag": head["ETag"],
            "last_modified": remote_last_modified.isoformat(),
        }
        tmp_marker = self.marker_path.with_name(self.marker_path.name + ".part")
        tmp_marker.write_text(json.dumps(marker, sort_keys=True))
        os.replace(tmp_marker, self.marker_path)
        logger.info(
            "downloaded s3://%s/%s (%s)", self.s3_bucket, self.s3_key, head["ETag"]
        )
        return True

    def run(self, periodic: bool = True) -> None:
        if not periodic:
            self.sync_once()
            return
        while True:
            try:
                self.sync_once()
            except (BotoCoreError, ClientError, OSError) as exc:
                logger.warning("archive sync failed, retrying: %s", exc)
            time.sleep(DEFAULT_POLL_INTERVAL_SEC)


@click.command()
@click.option(
    "--periodic/--no-periodic",
    default=True,
    show_default=True,
)
def main(periodic: bool) -> None:
    logging.basicConfig(level=logging.INFO)
    downloader = EngineDownloader()
    downloader.run(periodic=periodic)


if __name__ == "__main__":
    main()
