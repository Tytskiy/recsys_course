import os
import shutil
import tarfile
import tempfile
from collections import defaultdict
from itertools import batched
from pathlib import Path
from typing import Iterable, Literal
from urllib.parse import urlparse

import boto3
from engine.interfaces import ID, Embedding, EmbeddingStorage, Factor
from engine.storages import ValkeyFeatureStorage
from loguru import logger


def create_workdir() -> Path:
    workdir = Path(tempfile.mkdtemp(prefix="rec-engine-files-"))
    return workdir


def upload_files(
    target_archive_path: str,
    files_to_save: list[str],
    workdir: Path,
) -> None:
    try:
        archive_path = workdir / "archive.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            for filename in files_to_save:
                tar.add(workdir / filename, arcname=filename)
        upload_file_to_s3(str(archive_path), target_archive_path)
        logger.info(f"Uploaded rec-engine archive to {target_archive_path}, files: {files_to_save}")
    finally:
        shutil.rmtree(workdir)


def add_files_to_archive(target_archive_path: str, files: list[Path]) -> None:
    workdir = create_workdir()
    local_archive = workdir / "archive.tar.gz"
    download_file_from_s3(target_archive_path, str(local_archive))
    with tarfile.open(local_archive, "r:gz") as tar:
        tar.extractall(workdir, filter="data")
    local_archive.unlink()
    for path in files:
        shutil.copy(path, workdir / path.name)
    upload_files(target_archive_path, sorted(p.name for p in workdir.iterdir()), workdir)


def upload_embeddings(
    embeddings: list[tuple[ID, Embedding]],
    embedding_storage: EmbeddingStorage,
    embedding_type: str,
    metric: Literal["Dot", "Cosine", "Euclid"] = "Dot",
    overwrite_collection: bool = True,
):
    if overwrite_collection:
        embedding_storage.build_embedding_index(
            embedding_type=embedding_type,
            embeddings=embeddings,
            metric=metric,
        )
    else:
        embedding_storage.upsert_embeddings(embedding_type=embedding_type, embeddings=embeddings)
    logger.info(
        "Exported embeddings to Qdrant: collection={}, mode={}",
        embedding_type,
        "overwrite" if overwrite_collection else "update",
    )


def merge_profiles(
    profile_subsets: Iterable[dict[ID, dict[str, Factor]]],
) -> dict[ID, dict[str, Factor]]:
    profiles = defaultdict(dict)
    for subset in profile_subsets:
        for iid, profile in subset.items():
            profiles[iid] |= profile
    return profiles


def upload_profiles(
    profiles: dict[ID, dict[str, Factor]],
    user: bool = True,
    batch_size: int = 10_000,
) -> None:
    client = ValkeyFeatureStorage.get_client()
    try:
        for batch in batched(profiles.items(), batch_size):
            pipe = client.pipeline(transaction=False)
            storage = ValkeyFeatureStorage(client=pipe, readonly=False)
            for profile_id, features in batch:
                if user:
                    storage.update(user_id=profile_id, features=features)
                else:
                    storage.update(item_id=profile_id, features=features)
            pipe.execute()
    finally:
        client.close()
    logger.info("Exported {} profiles to Valkey", len(profiles))


def get_storage_options() -> dict[str, str]:
    return {
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID", "admin"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY", "password"),
        "aws_endpoint_url": os.getenv("AWS_ENDPOINT_URL", "http://localhost:9002"),
        "aws_allow_http": "true",
    }


def get_s3_client():
    storage_options = get_storage_options()
    return boto3.client(
        "s3",
        endpoint_url=storage_options["aws_endpoint_url"],
        aws_access_key_id=storage_options["aws_access_key_id"],
        aws_secret_access_key=storage_options["aws_secret_access_key"],
    )


def parse_s3_path(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Invalid s3 uri: {url}")
    return parsed.netloc, parsed.path.lstrip("/")


def download_file_from_s3(s3_path: str, local_path: str) -> None:
    bucket, key = parse_s3_path(s3_path)
    boto_client = get_s3_client()
    boto_client.download_file(bucket, key, local_path)


def upload_file_to_s3(local_path: str, s3_path: str) -> None:
    bucket, key = parse_s3_path(s3_path)
    boto_client = get_s3_client()
    boto_client.upload_file(local_path, bucket, key)
