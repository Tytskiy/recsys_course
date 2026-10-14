import dataclasses

import numpy as np
import polars as pl
import scipy.sparse as sps
from engine.interfaces import EmbeddingStorage
from engine.storages import QdrantEmbeddingStorage
from implicit.als import AlternatingLeastSquares

from tasks.actions import TargetType, build_user_item_matrix
from tasks.utils import upload_embeddings


@dataclasses.dataclass(frozen=True)
class IALSParams:
    dim: int = 128
    max_iter: int = 100
    l2_alpha: float = 0.01
    conf_alpha: float = 1
    warm_user_threshold: int = 4
    use_bm25_reweighting: bool = True
    target_type: TargetType = "CartUpdate"
    random_state: int = 42


def split_by_activity(
    user_item_matrix: sps.sparray,
    warm_user_threshold: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split rows by their sum: warm if `>= warm_user_threshold`.

    Return the warm rows, the cold rows and the cold row indices.
    """
    # TODO: Your solution
    return warm_users, cold_users, cold_user_ids


def train_model(
    matrix: sps.sparray,
    params: IALSParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit `implicit`'s ALS on the warm users, fold the cold users in with one
    `recalculate_user` step, and return `(user_factors, item_factors)` with one
    user row per matrix row.
    """
    # TODO: Your solution
    return user_factors, item_factors


def export_embeddings(
    item_factors: np.ndarray,
    user_factors: np.ndarray,
    item_to_idx: pl.DataFrame,
    user_to_idx: pl.DataFrame,
    embedding_storage: EmbeddingStorage,
    embedding_type: str = "iALS",
) -> None:
    """Upload item and user factors as `item_<embedding_type>` and
    `user_<embedding_type>` (dot-product metric), keyed by the original ids.
    """
    # TODO: Your solution


def train_implicit_als(
    dm_action: pl.DataFrame,
    embedding_storage: EmbeddingStorage | None = None,
    params: IALSParams = IALSParams(),
    embedding_type: str = "iALS",
) -> None:
    """Build the `params.target_type` matrix, train the model and export the embeddings."""
    embedding_storage = embedding_storage or QdrantEmbeddingStorage()
    # TODO: Your solution
