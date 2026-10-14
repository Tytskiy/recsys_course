import datetime
from typing import Literal, assert_never

import polars as pl
import scipy.sparse as sps
from loguru import logger

from tasks.utils import get_storage_options


def date_add(dt_str: str, days: int = 0) -> str:
    parsed = datetime.datetime.fromisoformat(dt_str)
    dt = parsed + datetime.timedelta(days=days)
    return dt.isoformat()[:10]

def read_from_s3(path: str) -> pl.DataFrame:
    return pl.read_parquet(path, storage_options=get_storage_options())


def load_actions(path: str, last_dt: str | None = None, first_dt: str | None = None) -> pl.DataFrame:
    actions = read_from_s3(path)
    if first_dt is not None:
        actions = actions.filter(pl.col("dt") >= first_dt)
    if last_dt is not None:
        actions = actions.filter(pl.col("dt") <= last_dt)
    return actions


type TargetType = Literal["MixedCartUpdatePurchase", "CartUpdate", "Click"]


def target_action_agg(
    num_purchases: pl.Expr,
    num_cart_updates: pl.Expr,
    num_clicks: pl.Expr,
    target_type: TargetType,
) -> pl.Expr:
    match target_type:
        case "MixedCartUpdatePurchase":
            return num_purchases + num_cart_updates
        case "CartUpdate":
            return num_cart_updates
        case "Click":
            return num_clicks
        case _:
            assert_never(target_type)


def bm25_reweight(
    user_item_conf: pl.DataFrame,
    k1: float = 100.0,
    b: float = 0.8,
) -> pl.DataFrame:
    """Apply BM25 re-weighting to the `confidence` column of a
    `(user_id, item_id, confidence)` frame; see the task README for the formula."""
    # TODO: Your solution


def build_user_item_matrix(
    dm_action: pl.DataFrame,
    target_type: TargetType,
    use_bm25_reweighting: bool = False,
) -> tuple[sps.csr_array, pl.DataFrame, pl.DataFrame]:
    """Count actions per (user, item), turn them into `target_type` confidences with
    `target_action_agg`, keep positive ones (optionally BM25-reweighted, positive
    only), and return the CSR matrix with the `user_id -> user_idx` and
    `item_id -> item_idx` frames (ids sorted ascending).
    """
    # TODO: Your solution
    return matrix, user_to_index, item_to_index
