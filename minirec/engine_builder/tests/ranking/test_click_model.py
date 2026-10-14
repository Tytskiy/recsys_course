"""Spec for the position-based click model (task build_pool, `tasks/ranking/click_model.py`).

Behaviours, one named test or case each:
- `estimate_pbm`: the exact EM updates from theta = gamma = 0.5 (hand case, one and two steps:
  both M-steps from one E-step, normalized once at the end), `theta[0] == 1`, one
  log-likelihood value per iteration, computed before the update and never decreasing, finite
  results when a row has no non-clicks (p = 1), row order irrelevant, and recovery of the true theta on the seeded simulator where naive CTR fails.
- `ipw_weights`: `min(1 / theta[p - 1], clip)` for positives (clip boundary pair), the last theta
  past the fitted range, 1.0 for negatives and for positives without a position, and better
  per-item calibration than naive CTR on the simulator.
`pbm_counts` is provided; the simulator is `tests/ranking/sim.py`.
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from tasks.ranking.click_model import estimate_pbm, ipw_weights, pbm_counts

from tests.ranking.sim import WorldConfig, actions_frame, make_world, naive_ctr_by_position, sample_log

# Two items x two positions. One EM step from theta = gamma = 0.5 (p = 0.25 everywhere):
# P(examined | no click) = P(relevant | no click) = 0.5 * 0.5 / 0.75 = 1/3.
# theta1 = (4 + 6/3 + 2 + 8/3) / 20 = 8/15; theta2 = (2 + 8/3 + 1 + 9/3) / 20 = 13/30
# normalized theta2 = 13/30 / 8/15 = 0.8125. The table is symmetric, so gamma = (8/15, 13/30).
# Second step, from the unnormalized theta and that gamma (P(examined | no click) per row:
# 8/23, 91/346, 68/173, 13/43): theta1 = (4 + 6 * 8/23 + 2 + 8 * 68/173) / 20 = 0.561573,
# theta2 = (2 + 8 * 91/346 + 1 + 9 * 13/43) / 20 = 0.391249, normalized theta2 = 0.696701.
# Normalizing after every step gives 0.737422; updating gamma with the new theta gives 0.695313.
HAND_COUNTS = pl.DataFrame({
    "item_id": [1, 1, 2, 2],
    "position": [1, 2, 1, 2],
    "impressions": [10, 10, 10, 10],
    "clicks": [4, 2, 2, 1],
})
HAND_LOG_LIKELIHOOD_0 = 9 * math.log(0.25) + 31 * math.log(0.75)


@pytest.fixture(scope="module")
def world_log():
    """The test world: 20 000 requests, theta = 1/k, a relevance-sorted logger (noise 0.3)
    that shuffles 10 % of the lists."""
    world = make_world(WorldConfig(seed=0))
    return world, sample_log(world, eps=0.1, seed=0)


@pytest.fixture(scope="module")
def world_counts(world_log) -> pl.DataFrame:
    return pbm_counts(actions_frame(world_log[1]), max_pos=10)


@pytest.mark.parametrize(
    ("n_iter", "theta2"),
    [
        pytest.param(1, 0.8125, id="one_step"),
        pytest.param(2, 0.6967013066759641, id="two_steps_one_e_step_normalize_once"),
    ],
)
def test_em_steps_hand_case(n_iter: int, theta2: float):
    result = estimate_pbm(HAND_COUNTS, max_pos=2, n_iter=n_iter)

    np.testing.assert_allclose(result.theta, [1.0, theta2], rtol=1e-9)


def test_log_likelihood_is_recorded_before_each_update():
    result = estimate_pbm(HAND_COUNTS, max_pos=2, n_iter=3)

    assert len(result.log_likelihood) == 3
    assert result.log_likelihood[0] == pytest.approx(HAND_LOG_LIKELIHOOD_0, rel=1e-12)


def test_rows_without_non_clicks_stay_finite():
    """Item 3 is shown once, at position 3, and clicked: one step sets theta3 = gamma3 = 1, so
    p = 1 on that row. Its non-click terms are 0 (0 * log 0 = 0), not nan."""
    counts = pl.concat([
        HAND_COUNTS,
        pl.DataFrame({"item_id": [3], "position": [3], "impressions": [1], "clicks": [1]}),
    ])

    result = estimate_pbm(counts, max_pos=3, n_iter=5)

    assert np.isfinite(result.theta).all()
    assert np.isfinite(result.log_likelihood).all()


def test_theta_normalized_first_position_is_one(world_counts):
    result = estimate_pbm(world_counts, max_pos=10, n_iter=50)

    assert result.theta.shape == (10,)
    assert result.theta[0] == pytest.approx(1.0)


def test_log_likelihood_non_decreasing(world_counts):
    """EM never lowers the likelihood; a wrong E-step does."""
    ll = np.array(estimate_pbm(world_counts, max_pos=10, n_iter=100).log_likelihood)

    assert (np.diff(ll) >= -1e-9 * np.abs(ll[1:])).all()


def test_row_order_and_extra_columns_irrelevant(world_counts):
    shuffled = world_counts.sample(fraction=1.0, shuffle=True, seed=1).with_columns(extra=pl.lit("x"))

    np.testing.assert_allclose(
        estimate_pbm(shuffled, 10, 30).theta, estimate_pbm(world_counts, 10, 30).theta, rtol=1e-9
    )


def test_recovers_simulator_theta(world_log, world_counts):
    """Recovered within 0.07 of the true 1/k (observed worst over 20 seeds: 0.034), while the
    naive CTR-by-position curve is off by more than 0.07 (observed 0.105 here): the sorted logger
    puts relevant items on top, so naive CTR mixes position with relevance."""
    world, log = world_log
    theta = estimate_pbm(world_counts, max_pos=10, n_iter=200).theta

    assert np.abs(theta - world.theta).max() < 0.07
    assert np.abs(naive_ctr_by_position(log) - world.theta).max() > 0.07


THETA = np.array([1.0, 0.5, 0.25])


@pytest.mark.parametrize(
    ("target", "position", "clip", "expected"),
    [
        pytest.param(1.0, 2, 10.0, 2.0, id="positive_one_over_theta"),
        pytest.param(1.0, 3, 4.0, 4.0, id="one_over_theta_equals_clip"),
        pytest.param(1.0, 3, 3.5, 3.5, id="one_over_theta_above_clip"),
        pytest.param(1.0, 7, 10.0, 4.0, id="position_beyond_max_uses_tail_bucket"),
        pytest.param(1.0, None, 10.0, 1.0, id="null_position_weight_one"),
        pytest.param(0.0, 3, 10.0, 1.0, id="negative_weight_one"),
    ],
)
def test_ipw_weights(target: float, position: int | None, clip: float, expected: float):
    df = pl.DataFrame(
        {"target": [target], "label_position": [position]},
        schema={"target": pl.Float32, "label_position": pl.Int64},
    )

    weights = ipw_weights(df, THETA, clip)

    assert weights.name == "weight"
    assert weights.dtype == pl.Float64
    assert weights.to_list() == [pytest.approx(expected)]


def test_ipw_improves_calibration(world_log, world_counts):
    """Per-item click rate reweighted by 1 / theta estimates relevance gamma; naive CTR
    underestimates it for items shown low. IPW mean abs error must be < 0.6 x naive
    (observed 0.33-0.39x over 5 seeds). Items with >= 20 impressions."""
    world, log = world_log
    theta = estimate_pbm(world_counts, max_pos=10, n_iter=200).theta
    rows = pl.DataFrame({
        "item_id": log.items.ravel(),
        "target": log.clicks.ravel().astype(np.float64),
        "label_position": np.tile(np.arange(1, 11), log.items.shape[0]),
    })
    per_item = (
        rows.with_columns(ipw_weights(rows, theta, clip=100.0))
        .group_by("item_id")
        .agg(
            naive=pl.col("target").mean(),
            ipw=(pl.col("target") * pl.col("weight")).mean(),
            n=pl.len(),
        )
        .filter(pl.col("n") >= 20)
    )
    gamma = world.gamma[per_item["item_id"].to_numpy()]

    naive_mae = np.abs(per_item["naive"].to_numpy() - gamma).mean()
    ipw_mae = np.abs(per_item["ipw"].to_numpy() - gamma).mean()
    assert ipw_mae < 0.6 * naive_mae
