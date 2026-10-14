"""Seeded position-based click world with known truth: the shared fixture for the click model
(build_pool) tests.

Test scaffolding, not student code: tests import it as `tests.ranking.sim`.

The world: `n_requests` requests, each with `list_len` distinct candidate items drawn by
popularity, true relevance `gamma ~ Beta(1, 6)` per item and examination `theta[k] = (k + 1) ** -eta`.
The logger shows a fixed base order per request (relevance-sorted with Gumbel noise, or
anti-sorted), and with probability `eps` shuffles the list uniformly instead. So the propensity of
an (item, position) pair is known exactly: `eps / list_len + (1 - eps) * [base position == position]`.
A click is a Bernoulli draw with probability `theta[position] * gamma[item]`.
"""

import dataclasses
from typing import Literal

import numpy as np
import polars as pl


@dataclasses.dataclass(frozen=True)
class WorldConfig:
    n_requests: int = 20_000
    n_items: int = 300
    list_len: int = 10
    eta: float = 1.0
    noise: float = 0.3
    logger: Literal["sorted", "anti"] = "sorted"
    seed: int = 0


@dataclasses.dataclass(frozen=True)
class World:
    theta: np.ndarray  # (list_len,), theta[0] == 1
    gamma: np.ndarray  # (n_items,)
    base: np.ndarray  # (n_requests, list_len) item ids in the logger's base order


@dataclasses.dataclass(frozen=True)
class Log:
    items: np.ndarray  # (n_requests, list_len) item shown at position k + 1
    propensity: np.ndarray  # (n_requests, list_len) P(logger puts items[r, k] at position k + 1)
    clicks: np.ndarray  # (n_requests, list_len) 0/1


def make_world(cfg: WorldConfig = WorldConfig()) -> World:
    rng = np.random.default_rng(cfg.seed)
    theta = np.arange(1, cfg.list_len + 1, dtype=np.float64) ** -cfg.eta
    gamma = rng.beta(1.0, 6.0, cfg.n_items)
    log_pop = np.log(rng.dirichlet(np.ones(cfg.n_items)))
    # Gumbel top-k = sampling list_len distinct items proportional to popularity
    keys = log_pop + rng.gumbel(size=(cfg.n_requests, cfg.n_items))
    candidates = np.argpartition(-keys, cfg.list_len, axis=1)[:, : cfg.list_len]
    score = np.log(gamma[candidates]) + cfg.noise * rng.gumbel(size=candidates.shape)
    order = np.argsort(-score if cfg.logger == "sorted" else score, axis=1, kind="stable")
    return World(theta=theta, gamma=gamma, base=np.take_along_axis(candidates, order, axis=1))


def sample_log(world: World, eps: float, seed: int) -> Log:
    """One log of the world: fresh eps-shuffles and fresh clicks, same candidate sets and base order."""
    rng = np.random.default_rng(seed)
    n_requests, list_len = world.base.shape
    shuffled = rng.random(n_requests) < eps
    perm = np.argsort(rng.random((n_requests, list_len)), axis=1)
    items = np.where(shuffled[:, None], np.take_along_axis(world.base, perm, axis=1), world.base)
    propensity = eps / list_len + (1 - eps) * (items == world.base)
    clicks = (rng.random(items.shape) < world.theta[None, :] * world.gamma[items]).astype(np.int64)
    return Log(items=items, propensity=propensity, clicks=clicks)


def log_frame(log: Log) -> pl.DataFrame:
    """OPE log: `request_id`, `item_id`, `position` (1-based), `click`, `propensity`."""
    n_requests, list_len = log.items.shape
    return pl.DataFrame({
        "request_id": np.repeat(np.arange(n_requests), list_len),
        "item_id": log.items.ravel(),
        "position": np.tile(np.arange(1, list_len + 1), n_requests),
        "click": log.clicks.ravel().astype(np.float64),
        "propensity": log.propensity.ravel(),
    })


def actions_frame(log: Log, source_type: str = "ST_Search") -> pl.DataFrame:
    """The log as `dm_action` rows: an `AT_View` per shown item and an `AT_CartUpdate` per click."""
    shown = log_frame(log).with_columns(source_type=pl.lit(source_type))
    views = shown.with_columns(action_type=pl.lit("AT_View"))
    carts = shown.filter(pl.col("click") == 1).with_columns(action_type=pl.lit("AT_CartUpdate"))
    return (
        pl.concat([views, carts])
        .rename({"position": "position_in_request"})
        .select(
            pl.col("request_id").cast(pl.UInt64),
            pl.col("item_id").cast(pl.UInt64),
            "position_in_request",
            "source_type",
            "action_type",
        )
    )


def naive_ctr_by_position(log: Log) -> np.ndarray:
    """Clicks / impressions per position, normalized to position 1: the estimate PBM corrects."""
    ctr = log.clicks.mean(axis=0)
    return ctr / ctr[0]
