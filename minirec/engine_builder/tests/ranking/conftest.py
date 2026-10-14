"""Pool-builder fixtures layered on the parent conftest's `dm_action`; constants live in pool_data.py."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

import polars as pl
import pytest
from tasks.ranking.pool_builder import PoolBuilder

from tests.ranking.pool_data import ITEM_ID, NEGATIVE_ACTIONS, POSITIVE_ACTIONS, USER_ID


@pytest.fixture(scope="session")
def pool_builder() -> PoolBuilder:
    return PoolBuilder(
        num_features=[],
        positive_actions=POSITIVE_ACTIONS,
        negative_actions=NEGATIVE_ACTIONS,
    )


@pytest.fixture(scope="session")
def make_feature_log() -> Callable[..., pl.DataFrame]:
    """Factory: one-row feature log at `request_ts` for USER_ID + `item_id`."""
    def _make(request_ts: int, item_id: int = ITEM_ID, request_id: int = 1) -> pl.DataFrame:
        dt = datetime.fromtimestamp(request_ts, tz=timezone.utc).replace(tzinfo=None)
        return pl.DataFrame(
            {
                "request_id": pl.Series([request_id], dtype=pl.UInt64),
                "user_id": pl.Series([USER_ID], dtype=pl.UInt64),
                "request_time": pl.Series([dt]).cast(pl.Datetime("us")),
                "item_id": pl.Series([item_id], dtype=pl.UInt64),
            }
        )
    return _make


@pytest.fixture(scope="session")
def make_joined_candidates() -> Callable[[list[dict]], pl.DataFrame]:
    """Factory: minimal pre-joined candidate DataFrame as produced by
    `PoolBuilder.join_asof_actions_to_features`."""
    def _make(rows: list[dict]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "request_id": pl.Series([r["request_id"] for r in rows], dtype=pl.UInt64),
                "user_id": pl.Series([USER_ID] * len(rows), dtype=pl.UInt64),
                "item_id": pl.Series([r["item_id"] for r in rows], dtype=pl.UInt64),
                "request_time": pl.Series(
                    [datetime(2023, 10, 1)] * len(rows)
                ).cast(pl.Datetime("us")),
                "action_type": pl.Series([r["action_type"] for r in rows], dtype=pl.String),
                "score": pl.Series([r.get("score", 0.5) for r in rows], dtype=pl.Float32),
            }
        )
    return _make
