from __future__ import annotations

import click
from constants import DATASET_PATH, ENGINE_ARCHIVE_PATH
from tasks.actions import (
    load_actions,
)
from tasks.features import compute_action_statistics, compute_user_history
from tasks.models import (
    train_implicit_als,
    train_top_pop,
)
from tasks.utils import (
    create_workdir,
    merge_profiles,
    upload_files,
    upload_profiles,
)


def build_recommender(
    dataset_path: str,
    latest_dt: str,
    archive_path: str,
    skip_heavy: bool = False,
) -> None:
    workdir = create_workdir()
    dm_action = load_actions(f"{dataset_path}/dm_action.parquet", last_dt=latest_dt)

    if not skip_heavy:
        train_implicit_als(dm_action, embedding_type="iALS")

    artifacts = [
        train_top_pop(dm_action, workdir),
    ]
    files_to_upload = [name for a in artifacts for name in a.files]
    item_profiles = merge_profiles(
        [
            *(a.item_profiles for a in artifacts),
            compute_action_statistics(latest_dt, dm_action, user=False),
        ]
    )
    user_profiles = merge_profiles(
        [
            compute_action_statistics(latest_dt, dm_action, user=True),
            compute_user_history(dm_action),
        ]
    )

    upload_profiles(item_profiles, user=False)
    upload_profiles(user_profiles, user=True)
    upload_files(archive_path, files_to_upload, workdir)


@click.command
@click.option("--latest-dt", type=str, required=True)
@click.option("--dataset-path", type=click.Path(), default=DATASET_PATH)
@click.option("--archive-path", type=click.Path(), default=ENGINE_ARCHIVE_PATH)
@click.option("--skip-heavy", is_flag=True, default=False, help="Keep the iALS embeddings already in Qdrant.")
def main(dataset_path: str, latest_dt: str, archive_path: str, skip_heavy: bool):
    build_recommender(dataset_path, latest_dt, archive_path, skip_heavy)


if __name__ == "__main__":
    main()
