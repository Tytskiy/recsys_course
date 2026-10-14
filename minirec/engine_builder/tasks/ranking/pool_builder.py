from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import orjson
import polars as pl
from catboost import Pool

ACTION_PRIORITY: dict[str, int] = {
    "AT_View": 1,
    "AT_Click": 2,
    "AT_CartUpdate": 3,
    "AT_Purchase": 4,
}


class PoolBuilder:
    def __init__(
        self,
        num_features: list[str],
        positive_actions: list[str],
        negative_actions: list[str],
        cat_features: list[str] | None = None,
        window_size: int = 1800,
        default_feature_value: float = 0.0,
    ):
        self.num_features = num_features
        self.cat_features = cat_features or []
        self.all_features = num_features + self.cat_features
        self.window_size = window_size
        self.default_feature_value = default_feature_value
        self.positive_actions = positive_actions
        self.negative_actions = negative_actions

    @staticmethod
    def parse_log_json(log_entry: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Flatten one `candidates.jsonl` record into per-candidate rows: `request_id`, `user_id`,
        `request_time`, `item_id`, `position` (1-based index in the logged ranking), `served`
        (`position <= num_served`), every `user_profile` key and every `factors` key."""
        user_profile = log_entry["user_profile"]
        for position, candidate in enumerate(log_entry["candidates"], start=1):
            yield {
                "request_id": log_entry["request_id"],
                "user_id": log_entry["user_id"],
                "request_time": log_entry["request_time"],
                "item_id": candidate["item_id"],
                "position": position,
                "served": position <= log_entry["num_served"],
                **user_profile,
                **candidate["factors"],
            }

    @staticmethod
    def read_logs(log_path: str) -> pl.DataFrame:
        """Read a JSONL candidates log into a flat Polars DataFrame.

        One output row per candidate per request, flattened with `parse_log_json` (so the
        rows carry `position` and `served`). Cast `user_id`/`item_id` to
        `UInt64` and `request_time` to a naive UTC `Datetime("us")` so the
        downstream point-in-time join can match on a stable schema. The logger
        writes ISO strings with an offset (`...+00:00`); convert them to UTC
        before dropping the time zone.
        """
        # TODO: Your solution

    @staticmethod
    def join_asof_actions_to_features(
        feature_log: pl.DataFrame,
        action_log: pl.DataFrame,
        window_size: int = 1800,
    ) -> pl.DataFrame:
        """Point-in-time join: label each logged candidate with the user's best
        future action on that item within `window_size` seconds.

        "Forward-only" — actions strictly in the past relative to `request_time`
        are excluded to avoid label leakage. If multiple actions match, the one
        with the highest `ACTION_PRIORITY` wins. Returns one row per
        `(request_id, item_id)` with the winning action's `action_type`,
        `label_position` (its `position_in_request`) and `label_source_type` (its
        `source_type`) joined in: the click model needs where and on which surface
        the label was observed. The feature log may carry its own `source_type`
        (from `user_profile`); keep it unchanged.
        """
        # TODO: Your solution

    def build_pools_from_log(
        self,
        log_path: str,
        action_log: pl.DataFrame,
        test_size: float = 0.2,
        random_seed: int = 42,
    ) -> tuple[Pool, Pool]:
        feature_log = self.read_logs(log_path)
        joined = self.join_asof_actions_to_features(feature_log, action_log, window_size=self.window_size)
        dataset = self.build_dataset(joined, random_seed=random_seed)
        train_df, test_df = self.train_test_split_by_time(dataset, test_size=test_size)
        return self.build_cbm_pool(train_df), self.build_cbm_pool(test_df)

    @staticmethod
    def train_test_split_by_time(
        dataset: pl.DataFrame,
        test_size: float = 0.2,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Split by time, never at random: a random split puts a user's later requests
        into train and their earlier ones into test, so the model learns from the future.

        Order requests by their earliest `request_time`; the first
        `int(n_requests * (1 - test_size))` requests go to train, the rest to test.
        All rows of one request stay on the same side.
        """
        # TODO: Your solution

    @staticmethod
    def sample_per_partition(
        rows: pl.DataFrame,
        partition_col: str,
        n: int,
        random_seed: int,
    ) -> pl.DataFrame:
        return (
            rows
            .sort([partition_col, "user_id", "item_id"])
            .with_columns(
                pl.int_range(pl.len()).shuffle(seed=random_seed).over(partition_col).alias("sample_order")
            )
            .filter(pl.col("sample_order") < n)
            .drop("sample_order")
        )

    def build_dataset(
        self,
        candidates: pl.DataFrame,
        random_seed: int = 42,
    ) -> pl.DataFrame:
        """Turn labeled candidates into a class-balanced training dataset.

        Drop requests without any positive action (nothing to learn from), keep
        at most 9 negatives per request via `sample_per_partition` (deterministic
        under `random_seed`), set `target` (1.0 for positive actions / 0.0 for
        negatives) and `group_id = str(request_id)` for CatBoost group metrics.
        """
        # TODO: Your solution

    def build_cbm_pool(self, dataset: pl.DataFrame) -> Pool:
        """`catboost.Pool` over `self.all_features`, sorted by `group_id` (CatBoost needs contiguous
        groups). A missing numeric factor becomes `default_feature_value`, the value `ModelRanker`
        uses online, so the same candidate gives the same row offline and online. A `weight` column,
        when present, becomes the per-row weight."""
        dataset = dataset.sort("group_id", maintain_order=True).with_columns(
            pl.col(self.num_features).fill_null(self.default_feature_value)
        )
        return Pool(
            data=dataset.select(self.all_features).to_numpy(),
            label=dataset["target"].to_numpy(),
            feature_names=self.all_features,
            cat_features=self.cat_features,
            group_id=dataset["group_id"].to_numpy(),
            weight=dataset["weight"].to_numpy() if "weight" in dataset.columns else None,
        )
