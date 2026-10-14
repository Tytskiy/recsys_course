import dataclasses
from pathlib import Path

import fastslim
import orjson
import polars as pl
import scipy.sparse as sps
from loguru import logger

from tasks.actions import TargetType, build_user_item_matrix
from tasks.models.artifacts import TrainingArtifacts


@dataclasses.dataclass(frozen=True)
class SLIMParams:
    max_iter: int = 50
    alpha: float = 1
    beta: float = 1
    n_threads: int = 4
    target_type: TargetType = "CartUpdate"
    weight_path: str = "slim_weights.npz"
    mapping_path: str = "slim_mapping.json"


def train_model(matrix: sps.sparray, params: SLIMParams) -> sps.csr_matrix:
    logger.info("Training SLIM...")
    weights = fastslim.fit(
        matrix,
        params.alpha,
        params.beta,
        params.max_iter,
        params.n_threads,
    )
    return weights


def export_files(
    workdir: Path,
    params: SLIMParams,
    weights: sps.csr_matrix,
    item_mapping: pl.DataFrame,
) -> None:
    logger.info("Exporting model files...")
    rows = [(int(iid), int(idx)) for iid, idx in item_mapping.iter_rows()]
    mappings = {
        "item_to_index": {iid: idx for iid, idx in rows},
        "index_to_item": {idx: iid for iid, idx in rows},
    }

    with open(workdir / params.mapping_path, "wb") as f:
        f.write(orjson.dumps(mappings, option=orjson.OPT_NON_STR_KEYS))
    sps.save_npz(workdir / params.weight_path, weights) 


def train_slim(
    dm_action: pl.DataFrame,
    workdir: Path,
    params: SLIMParams = SLIMParams(),
) -> TrainingArtifacts:
    matrix, _, item_to_idx = build_user_item_matrix(
        dm_action=dm_action,
        target_type=params.target_type,
    )
    weights = train_model(matrix, params)
    export_files(
        workdir,
        params,
        weights,
        item_to_idx,
    )
    return TrainingArtifacts(files=[params.mapping_path, params.weight_path])
