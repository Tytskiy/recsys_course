from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import click
import numpy as np
import orjson
import polars as pl
import yaml
from build_recommender import build_recommender
from constants import (
    DATASET_PATH,
    ENGINE_ARCHIVE_PATH,
    ENGINE_CONFIG_PATH,
    RANKER_CONFIG_PATH,
    REPLAY_ARCHIVE_PATH,
)
from engine.candidate_logger import (
    CANDIDATES_LOG_PATH,
    CG_CANDIDATES_LOG_PATH,
    ROTATING_LOGFILE_BACKUP_COUNT,
)
from engine.interfaces import Candidate, RecContext
from engine.recommender import RecommenderEngine
from loguru import logger
from tasks.actions import date_add, load_actions
from tasks.evaluation.pipeline import EvaluationPipeline
from tasks.ranking.click_model import estimate_pbm, ipw_weights, pbm_counts
from tasks.ranking.pool_builder import PoolBuilder
from tasks.ranking.training import CatBoostTrainParams, train_catboost
from tasks.utils import add_files_to_archive, download_file_from_s3


class EngineHandle:
    def __init__(self, config: dict, dataset_path: str, archive_s3_path: str):
        self.base_config = config
        self.dataset_path = dataset_path
        self.archive_s3_path = archive_s3_path
        self.engine: RecommenderEngine | None = None
        self.tmpdir: Path | None = None

    def recommend(self, num_candidates: int, context: RecContext) -> list[Candidate]:
        assert self.engine is not None, "reload(dt) must run before recommend"
        return self.engine.recommend(num_candidates, context)

    def reload(self, dt: str) -> None:
        self.shutdown()

        # a snapshot of days < dt: building with dt itself leaks the day's labels
        build_recommender(
            dataset_path=self.dataset_path,
            latest_dt=date_add(dt, -1),
            archive_path=self.archive_s3_path,
            skip_heavy=True,
        )

        self.tmpdir = Path(tempfile.mkdtemp(prefix="rec-engine-data-"))
        config = dict(self.base_config)

        local_archive = self.tmpdir / "archive.tar.gz"
        download_file_from_s3(self.archive_s3_path, str(local_archive))
        config["engine_data_path"] = str(local_archive)

        self.engine = RecommenderEngine.startup(config)

    def shutdown(self) -> None:
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None
        if self.tmpdir is not None:
            shutil.rmtree(self.tmpdir, ignore_errors=True)
            self.tmpdir = None


def merge_rotated_log(path: Path) -> None:
    backups = [path.with_name(f"{path.name}.{i}") for i in range(ROTATING_LOGFILE_BACKUP_COUNT, 0, -1)]
    present = [b for b in backups if b.exists()]
    if not present:
        return
    if backups[0].exists():
        logger.warning("{} used all {} backups; records older than {} may be lost", path, len(backups), backups[0])
    merged = path.with_name(f"{path.name}.merged")
    with open(merged, "wb") as out:
        for src in [*present, path]:
            with open(src, "rb") as f:
                shutil.copyfileobj(f, out)
    merged.replace(path)
    for backup in present:
        backup.unlink()


def collect_logged_features(
    dm_action: pl.DataFrame,
    dataset_path: str = DATASET_PATH,
    config_path: str = ENGINE_CONFIG_PATH,
    archive_s3_path: str = REPLAY_ARCHIVE_PATH,
) -> None:
    """Replay `dm_action` through the engine, whose `CandidateLogger` writes both candidate logs.

    The replay rewrites the live Valkey profiles and keeps the iALS embeddings in Qdrant,
    so factors of early days can see later data.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    for path in (Path(CANDIDATES_LOG_PATH), Path(CG_CANDIDATES_LOG_PATH)):
        if path.exists():
            path.write_bytes(b"")
        for i in range(1, ROTATING_LOGFILE_BACKUP_COUNT + 1):
            path.with_name(f"{path.name}.{i}").unlink(missing_ok=True)
    handle = EngineHandle(config, dataset_path, archive_s3_path)
    pipeline = EvaluationPipeline(
        dm_action=dm_action,
        recommend_handle=handle.recommend,
        reload_handle=handle.reload,
        use_candidates_from_actions=True,
    )
    logger.info("Running pipeline over {} requests", len(pipeline.requests))
    try:
        pipeline.run()
    finally:
        handle.shutdown()
    merge_rotated_log(Path(CANDIDATES_LOG_PATH))
    merge_rotated_log(Path(CG_CANDIDATES_LOG_PATH))
    logger.info("Step 1 done — candidates written to {}", CANDIDATES_LOG_PATH)


def fit_surface_thetas(actions: pl.DataFrame, max_pos: int, n_iter: int, min_impressions: int) -> dict[str | None, np.ndarray]:
    """PBM theta per surface (`source_type`) with at least `min_impressions` impressions, plus the
    theta of all surfaces pooled under the key `None`. Logs each curve next to naive CTR."""
    thetas: dict[str | None, np.ndarray] = {}
    surfaces = actions.filter(pl.col("action_type") == "AT_View")["source_type"].drop_nulls().unique().sort()
    for surface in [None, *surfaces]:
        counts = pbm_counts(actions, max_pos, source_type=surface)
        if surface is not None and counts["impressions"].sum() < min_impressions:
            continue
        thetas[surface] = estimate_pbm(counts, max_pos, n_iter).theta
        naive = counts.group_by("position").agg(ctr=pl.col("clicks").sum() / pl.col("impressions").sum()).sort("position")
        logger.info(
            "PBM {}: theta {} | naive CTR {}",
            surface or "all surfaces",
            np.round(thetas[surface][:10], 3).tolist(),
            np.round(naive["ctr"].to_numpy()[:10] / naive["ctr"][0], 3).tolist(),
        )
    return thetas


def pbm_weights(dataset: pl.DataFrame, thetas: dict[str | None, np.ndarray], clip: float) -> pl.DataFrame:
    """Add `weight` from `ipw_weights`, each row with the theta of its label's surface (pooled if unfitted)."""
    parts = [
        part.with_columns(ipw_weights(part, thetas.get(surface, thetas[None]), clip))
        for (surface,), part in dataset.partition_by("label_source_type", as_dict=True).items()
    ]
    return pl.concat(parts)


def build_ranker(
    dataset_path: str,
    last_dt: str,
    first_dt: str | None = None,
    config_path: str = ENGINE_CONFIG_PATH,
    ranker_config_path: str = RANKER_CONFIG_PATH,
    archive_s3_path: str = ENGINE_ARCHIVE_PATH,
    replay_archive_s3_path: str = REPLAY_ARCHIVE_PATH,
) -> None:
    """End-to-end ranker training pipeline (provided; needs the compose stack).

    `load_actions` → `collect_logged_features` (replay, writes `CANDIDATES_LOG_PATH`) →
    `PoolBuilder` (labels, time split) → PBM theta per surface on the training window and IPW
    weights → `train_catboost` → `ranker.cbm` + `factor_schema.json` added to the snapshot at
    `archive_s3_path`, where `ModelRanker` reads them.
    Run it after `build_recommender`, which replaces the snapshot.
    """
    logger.info("Loading actions from {} ({} → {})", dataset_path, first_dt, last_dt)
    dm_action = load_actions(
        f"{dataset_path}/dm_action.parquet",
        last_dt=last_dt,
        first_dt=first_dt,
    )
    logger.info("Loaded {} action rows", len(dm_action))

    collect_logged_features(
        dm_action=dm_action,
        dataset_path=dataset_path,
        config_path=config_path,
        archive_s3_path=replay_archive_s3_path,
    )

    with open(ranker_config_path) as f:
        ranker_config = yaml.safe_load(f)
    pbm = ranker_config["pbm"]

    builder = PoolBuilder(
        num_features=ranker_config["num_features"],
        positive_actions=ranker_config["positive_actions"],
        negative_actions=ranker_config["negative_actions"],
        cat_features=ranker_config.get("cat_features"),
        window_size=ranker_config["window_size"],
        default_feature_value=ranker_config["default_feature_value"],
    )
    feature_log = builder.read_logs(CANDIDATES_LOG_PATH)
    joined = builder.join_asof_actions_to_features(feature_log, dm_action, window_size=builder.window_size)
    train_df, test_df = builder.train_test_split_by_time(builder.build_dataset(joined))

    split_ts = test_df.select(pl.col("request_time").min().dt.epoch("s")).item()
    thetas = fit_surface_thetas(
        dm_action.filter(pl.col("timestamp") < split_ts), pbm["max_pos"], pbm["n_iter"], pbm["min_impressions"]
    )
    train_df = pbm_weights(train_df, thetas, pbm["clip"])
    model = train_catboost(builder.build_cbm_pool(train_df), builder.build_cbm_pool(test_df), CatBoostTrainParams())

    context_features = set(ranker_config.get("context_features", []))
    factor_schema = [(name, name in context_features) for name in model.feature_names_]
    with tempfile.TemporaryDirectory(prefix="rec-ranker-") as tmpdir:
        model_path = Path(tmpdir) / "ranker.cbm"
        model.save_model(str(model_path))
        factor_schema_path = Path(tmpdir) / "factor_schema.json"
        factor_schema_path.write_bytes(orjson.dumps(factor_schema))
        add_files_to_archive(archive_s3_path, [model_path, factor_schema_path])
    logger.info("Ranker added to snapshot {}", archive_s3_path)


@click.command()
@click.option("--last-dt", type=str, required=True)
@click.option("--first-dt", type=str, default=None)
@click.option("--dataset-path", type=click.Path(), default=DATASET_PATH)
@click.option("--config-path", type=click.Path(), default=ENGINE_CONFIG_PATH)
@click.option("--ranker-config-path", type=click.Path(), default=RANKER_CONFIG_PATH)
@click.option("--archive-s3-path", type=str, default=ENGINE_ARCHIVE_PATH, help="Snapshot that receives the ranker files.")
@click.option("--replay-archive-s3-path", type=str, default=REPLAY_ARCHIVE_PATH, help="Scratch snapshot for the replay.")
def main(last_dt, first_dt, dataset_path, config_path, ranker_config_path, archive_s3_path, replay_archive_s3_path):
    build_ranker(
        dataset_path=dataset_path,
        last_dt=last_dt,
        first_dt=first_dt,
        config_path=config_path,
        ranker_config_path=ranker_config_path,
        archive_s3_path=archive_s3_path,
        replay_archive_s3_path=replay_archive_s3_path,
    )


if __name__ == "__main__":
    main()
