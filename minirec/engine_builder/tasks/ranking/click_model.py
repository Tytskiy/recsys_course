from __future__ import annotations

import dataclasses

import numpy as np
import polars as pl


def pbm_counts(
    actions: pl.DataFrame,
    max_pos: int,
    click_action: str = "AT_CartUpdate",
    source_type: str | None = None,
) -> pl.DataFrame:
    """Aggregate a `dm_action` log into PBM counts: `item_id`, `position`, `impressions`, `clicks`.

    An impression is a distinct `(request_id, item_id)` with an `AT_View` at `position_in_request`
    in `1..max_pos` (the smallest position if the item was viewed twice). It is clicked when the same
    `(request_id, item_id)` has a `click_action`. `source_type` keeps one surface only.
    """
    if source_type is not None:
        actions = actions.filter(pl.col("source_type") == source_type)
    keys = ["request_id", "item_id"]
    views = (
        actions.filter(
            (pl.col("action_type") == "AT_View")
            & pl.col("request_id").is_not_null()
            & pl.col("position_in_request").is_between(1, max_pos)
        )
        .group_by(keys)
        .agg(pl.col("position_in_request").min().alias("position"))
    )
    clicked = actions.filter(pl.col("action_type") == click_action).select(keys).unique().with_columns(clicked=pl.lit(1))
    return (
        views.join(clicked, on=keys, how="left")
        .group_by("item_id", "position")
        .agg(impressions=pl.len().cast(pl.Int64), clicks=pl.col("clicked").fill_null(0).sum().cast(pl.Int64))
        .sort("item_id", "position")
    )


@dataclasses.dataclass(frozen=True)
class PBMResult:
    """`theta`: shape `(max_pos,)`, examination probability of position k + 1, `theta[0] == 1.0`.
    `log_likelihood`: one value per iteration, computed before that iteration's update."""

    theta: np.ndarray
    log_likelihood: list[float]


def estimate_pbm(counts: pl.DataFrame, max_pos: int, n_iter: int) -> PBMResult:
    """Fit the position-based click model P(click | item i at position k) = theta[k] * gamma[i] by EM.

    `counts` has `item_id`, `position` (1-based, `1..max_pos`), `impressions` (n) and `clicks` (c),
    one row per `(item_id, position)`, in any order. Start from theta = gamma = 0.5 everywhere and
    run exactly `n_iter` iterations. Each iteration:

    1. Append the log-likelihood of the current parameters,
       sum over rows of c * log(theta * gamma) + (n - c) * log(1 - theta * gamma).
    2. E-step (for the n - c non-clicks of a row, with the current theta and gamma):
       P(examined | no click) = theta * (1 - gamma) / (1 - theta * gamma),
       P(relevant | no click) = (1 - theta) * gamma / (1 - theta * gamma).
       A row without non-clicks (n == c) adds 0 to both non-click terms (0 * log 0 = 0), even when
       theta * gamma == 1 (a position or item that was always clicked).
    3. M-step, both updates from the same E-step:
       theta[k] = sum over rows at k of (c + (n - c) * P(examined | no click)) / sum of n at k,
       gamma[i] = sum over rows of i of (c + (n - c) * P(relevant | no click)) / sum of n of i.

    Only the product theta * gamma enters the likelihood, so the scale is not identified: return
    `theta / theta[0]` (normalize once, after the last iteration). A position without impressions
    keeps theta = 0.5 before the normalization.
    """
    # TODO: Your solution


def ipw_weights(df: pl.DataFrame, theta: np.ndarray, clip: float) -> pl.Series:
    """Inverse-propensity weights for a labeled pool, as a Float64 Series named `weight`.

    `df` has `target` (1.0 positive, 0.0 negative) and `label_position` (1-based position of the
    label action in the historical request, nullable). A positive at position p gets
    `min(1 / theta[min(p, len(theta)) - 1], clip)`: positions past the fitted range share the last
    theta (the tail bucket). Negatives and positives without a position get 1.0.
    """
    # TODO: Your solution
