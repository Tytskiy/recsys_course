from __future__ import annotations

import click
from constants import DATASET_PATH
from engine.storages import QdrantEmbeddingStorage
from tasks.actions import (
    date_add,
    load_actions,
)
from tasks.evaluation import evaluate_knn_cg
from tasks.models import train_implicit_als
from tasks.models.implicit_als import IALSParams


def train_and_evaluate(dataset_path: str, latest_dt: str, num_val_days: int) -> None:
    dm_action_path = f"{dataset_path}/dm_action.parquet"
    training_last_dt = date_add(latest_dt, -num_val_days - 1)
    val_first_dt = date_add(latest_dt, -num_val_days)
    train_actions = load_actions(dm_action_path, last_dt=training_last_dt)
    val_actions = load_actions(dm_action_path, first_dt=val_first_dt, last_dt=latest_dt)
    embed_storage = QdrantEmbeddingStorage()
    train_implicit_als(train_actions, embed_storage, IALSParams(), embedding_type="iALS")
    evaluate_knn_cg(val_actions, embed_storage, "iALS")


@click.command
@click.option("--latest-dt", type=str, required=True)
@click.option("--num-val-days", type=int, default=14)
@click.option("--dataset-path", type=click.Path(), default=DATASET_PATH)
def main(latest_dt: str, num_val_days: int, dataset_path: str):
    train_and_evaluate(dataset_path, latest_dt, num_val_days)


if __name__ == "__main__":
    main()
