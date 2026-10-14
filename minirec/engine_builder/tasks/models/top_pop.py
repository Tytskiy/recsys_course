from pathlib import Path

import orjson
import polars as pl
from loguru import logger

from tasks.models.artifacts import TrainingArtifacts


def train_top_pop(
    actions: pl.DataFrame,
    workdir: Path,
    filename: str = "top_pop.json",
    action_type: str = "AT_CartUpdate",
) -> TrainingArtifacts:
    """Calculates popularity statistics for TopPop and writes them into a JSON in workdir."""
    workdir_subpath = workdir / filename
    # TODO: Your solution
    return TrainingArtifacts(files=[filename])
