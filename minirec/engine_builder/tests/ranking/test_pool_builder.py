"""Spec for `PoolBuilder` (task build_pool, together with test_click_model.py).

Behaviours, one named test or case each:
- `join_asof_actions_to_features`: forward-only, window inclusive, highest priority wins, one row per
  `(request_id, item_id)`, unmatched candidates dropped, helper columns dropped, the winning action's
  `label_position` / `label_source_type` carried, the feature log's own columns kept.
- `build_dataset`: positive / negative labels, requests without a positive dropped, every positive
  kept and at most 9 negatives per request, unknown actions and null request ids dropped,
  `group_id` = str(request_id), one row per `(request_id, item_id)`, sorted by `group_id`.
- `read_logs`: one row per candidate with `position` / `served`, blank lines skipped, ids as UInt64,
  `request_time` as naive UTC.
- `train_test_split_by_time`: by earliest request time (not by id), whole requests, `int()` rounding.
- the end-to-end `build_pools_from_log` run.
`parse_log_json`, `build_cbm_pool` and `sample_per_partition` are provided.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import orjson
import polars as pl
import pytest
from catboost import Pool
from tasks.ranking.pool_builder import PoolBuilder

from tests.ranking.pool_data import (
    CART_UPDATE_TS,
    ITEM_ID,
    NEGATIVE_ACTIONS,
    POSITIVE_ACTIONS,
    USER_ID,
)


# dm_action holds, for USER_ID + ITEM_ID: CartUpdate at CART_UPDATE_TS, View at +21 s, Purchase at +400 s
@pytest.mark.parametrize(
    ("request_ts", "window_size", "item_id", "expected"),
    [
        # View at +14 s comes first, Purchase at +393 s has the higher priority
        pytest.param(CART_UPDATE_TS + 7, 500, ITEM_ID, "AT_Purchase", id="highest_priority_action_in_window_wins"),
        pytest.param(CART_UPDATE_TS + 7, 100, ITEM_ID, "AT_View", id="past_action_excluded_next_in_window_matched"),
        pytest.param(CART_UPDATE_TS - 2000, 1800, ITEM_ID, None, id="action_outside_window_excluded"),
        pytest.param(CART_UPDATE_TS, 1800, 0, None, id="unknown_item_returns_empty"),
        pytest.param(CART_UPDATE_TS - 100, 100, ITEM_ID, "AT_CartUpdate", id="action_at_exact_window_boundary_included"),
        pytest.param(CART_UPDATE_TS - 100, 99, ITEM_ID, None, id="action_one_second_past_window_excluded"),
        pytest.param(CART_UPDATE_TS, 0, ITEM_ID, "AT_CartUpdate", id="action_at_exact_request_time_matched"),
    ],
)
def test_join_asof_actions_to_features(request_ts, window_size, item_id, expected, dm_action, make_feature_log):
    feature_log = make_feature_log(request_ts, item_id=item_id)
    result = PoolBuilder.join_asof_actions_to_features(feature_log, dm_action, window_size=window_size)

    assert result["action_type"].to_list() == ([expected] if expected else [])


def test_join_output_drops_intermediate_columns(dm_action: pl.DataFrame, make_feature_log):
    feature_log = make_feature_log(CART_UPDATE_TS - 7)
    result = PoolBuilder.join_asof_actions_to_features(feature_log, dm_action, window_size=100)
    assert "request_timestamp" not in result.columns
    assert "action_priority" not in result.columns
    assert "timestamp" not in result.columns
    assert "action_type" in result.columns


def test_join_carries_label_position_and_source_type(dm_action: pl.DataFrame, make_feature_log):
    """The winning action's position in its historical request and its surface are the
    inputs of the click model: CartUpdate at +7 s was shown at position 3 of an ST_Search request."""
    feature_log = make_feature_log(CART_UPDATE_TS - 7)
    result = PoolBuilder.join_asof_actions_to_features(feature_log, dm_action, window_size=100)

    assert result.select("action_type", "label_position", "label_source_type").row(0) == (
        "AT_CartUpdate", 3, "ST_Search",
    )


def test_join_purchase_label_has_no_position(dm_action: pl.DataFrame, make_feature_log):
    """Purchases happen outside any request, so a purchase label has a null position
    (and later gets IPW weight 1). Window 500 s reaches the Purchase at +393 s."""
    feature_log = make_feature_log(CART_UPDATE_TS + 7)
    result = PoolBuilder.join_asof_actions_to_features(feature_log, dm_action, window_size=500)

    assert result.select("action_type", "label_position", "label_source_type").row(0) == ("AT_Purchase", None, None)


def test_join_keeps_feature_log_source_type(dm_action: pl.DataFrame, make_feature_log):
    """`source_type` from `user_profile` is a request feature; the label's surface goes to
    `label_source_type` and must not overwrite or rename it."""
    feature_log = make_feature_log(CART_UPDATE_TS - 7).with_columns(source_type=pl.lit("ST_Feed"))
    result = PoolBuilder.join_asof_actions_to_features(feature_log, dm_action, window_size=100)

    assert result.select("source_type", "label_source_type").row(0) == ("ST_Feed", "ST_Search")


def test_join_multiple_log_rows_each_get_best_action(dm_action: pl.DataFrame, make_feature_log):
    row1 = make_feature_log(CART_UPDATE_TS - 7, request_id=1)
    row2 = make_feature_log(CART_UPDATE_TS - 30, request_id=2)
    feature_log = pl.concat([row1, row2])
    result = PoolBuilder.join_asof_actions_to_features(feature_log, dm_action, window_size=100)

    assert len(result) == 2
    assert set(result["action_type"].to_list()) == {"AT_CartUpdate"}


def test_build_dataset_cart_update_is_positive_view_is_negative(pool_builder: PoolBuilder, make_joined_candidates):
    candidates = make_joined_candidates([
        {"request_id": 1, "item_id": 10, "action_type": "AT_CartUpdate"},
        {"request_id": 1, "item_id": 20, "action_type": "AT_View"},
    ])
    result = pool_builder.build_dataset(candidates)

    assert result.filter(pl.col("item_id") == 10)["target"][0] == pytest.approx(1.0)
    assert result.filter(pl.col("item_id") == 20)["target"][0] == pytest.approx(0.0)


def test_build_dataset_purchase_is_positive(pool_builder: PoolBuilder, make_joined_candidates):
    candidates = make_joined_candidates([
        {"request_id": 1, "item_id": 10, "action_type": "AT_Purchase"},
        {"request_id": 1, "item_id": 20, "action_type": "AT_View"},
    ])
    result = pool_builder.build_dataset(candidates)

    assert result.filter(pl.col("item_id") == 10)["target"][0] == pytest.approx(1.0)


def test_build_dataset_request_without_positive_is_dropped(pool_builder: PoolBuilder, make_joined_candidates):
    candidates = make_joined_candidates([
        {"request_id": 1, "item_id": 10, "action_type": "AT_View"},
        {"request_id": 1, "item_id": 20, "action_type": "AT_View"},
        {"request_id": 2, "item_id": 30, "action_type": "AT_CartUpdate"},
        {"request_id": 2, "item_id": 40, "action_type": "AT_View"},
    ])
    result = pool_builder.build_dataset(candidates)

    assert set(result["request_id"].cast(pl.UInt64).to_list()) == {2}


def test_build_dataset_negatives_capped_at_9_per_request(pool_builder: PoolBuilder, make_joined_candidates):
    candidates = make_joined_candidates(
        [{"request_id": 1, "item_id": 1, "action_type": "AT_CartUpdate"}]
        + [{"request_id": 1, "item_id": i, "action_type": "AT_View"} for i in range(2, 22)]
    )
    result = pool_builder.build_dataset(candidates)

    assert len(result.filter(pl.col("target") == 1.0)) == 1
    assert len(result.filter(pl.col("target") == 0.0)) == 9


def test_build_dataset_negatives_sampled_independently_per_request(pool_builder: PoolBuilder, make_joined_candidates):
    candidates = make_joined_candidates(
        [{"request_id": 1, "item_id": 1, "action_type": "AT_CartUpdate"}]
        + [{"request_id": 1, "item_id": i, "action_type": "AT_View"} for i in range(100, 115)]
        + [{"request_id": 2, "item_id": 2, "action_type": "AT_CartUpdate"}]
        + [{"request_id": 2, "item_id": i, "action_type": "AT_View"} for i in range(200, 204)]
    )
    result = pool_builder.build_dataset(candidates)

    negatives_req1 = result.filter((pl.col("request_id") == 1) & (pl.col("target") == 0.0))
    negatives_req2 = result.filter((pl.col("request_id") == 2) & (pl.col("target") == 0.0))
    assert len(negatives_req1) <= 9
    assert len(negatives_req2) == 4


def test_build_dataset_all_positives_no_negatives(pool_builder: PoolBuilder, make_joined_candidates):
    candidates = make_joined_candidates([
        {"request_id": 1, "item_id": 10, "action_type": "AT_CartUpdate"},
        {"request_id": 1, "item_id": 20, "action_type": "AT_Purchase"},
    ])
    result = pool_builder.build_dataset(candidates)

    assert set(result["request_id"].cast(pl.UInt64).to_list()) == {1}
    assert (result["target"] == 1.0).all()


def test_build_dataset_unknown_action_type_excluded(pool_builder: PoolBuilder, make_joined_candidates):
    candidates = make_joined_candidates([
        {"request_id": 1, "item_id": 10, "action_type": "AT_CartUpdate"},
        {"request_id": 1, "item_id": 20, "action_type": "AT_Click"},
    ])
    result = pool_builder.build_dataset(candidates)

    assert set(result["item_id"].to_list()) == {10}


def test_build_dataset_null_request_id_excluded(pool_builder: PoolBuilder, make_joined_candidates):
    candidates = make_joined_candidates([
        {"request_id": 1, "item_id": 10, "action_type": "AT_CartUpdate"},
        {"request_id": 1, "item_id": 20, "action_type": "AT_View"},
    ])
    with_null = candidates.with_columns(
        pl.when(pl.col("item_id") == 20)
        .then(None)
        .otherwise(pl.col("request_id"))
        .alias("request_id")
    )
    result = pool_builder.build_dataset(with_null)

    assert set(result["item_id"].to_list()) == {10}


def test_build_dataset_group_id_is_string_request_id(pool_builder: PoolBuilder, make_joined_candidates):
    candidates = make_joined_candidates([
        {"request_id": 7, "item_id": 10, "action_type": "AT_CartUpdate"},
        {"request_id": 7, "item_id": 20, "action_type": "AT_View"},
    ])
    result = pool_builder.build_dataset(candidates)

    assert result["group_id"].dtype == pl.String
    assert result["group_id"][0] == "7"


def test_build_dataset_one_row_per_request_item_sorted_by_group_id(pool_builder: PoolBuilder, make_joined_candidates):
    candidates = make_joined_candidates([
        {"request_id": 2, "item_id": 20, "action_type": "AT_CartUpdate"},
        {"request_id": 1, "item_id": 10, "action_type": "AT_CartUpdate"},
        {"request_id": 2, "item_id": 20, "action_type": "AT_CartUpdate"},
        {"request_id": 1, "item_id": 11, "action_type": "AT_View"},
        {"request_id": 2, "item_id": 21, "action_type": "AT_View"},
    ])
    result = pool_builder.build_dataset(candidates)

    assert result["group_id"].to_list() == ["1", "1", "2", "2"]
    assert sorted(result.select("request_id", "item_id").rows()) == [(1, 10), (1, 11), (2, 20), (2, 21)]


def _write_jsonl(path, entries: list[dict]) -> None:
    with open(path, "wb") as f:
        for entry in entries:
            f.write(orjson.dumps(entry))
            f.write(b"\n")


def test_read_logs_happy_path(tmp_path):
    log_path = tmp_path / "candidates.jsonl"
    _write_jsonl(log_path, [
        {
            "request_id": "req-A",
            "user_id": 100,
            "request_time": "2024-01-01T12:00:00",
            "num_served": 10,
            "user_profile": {},
            "candidates": [
                {"item_id": 1, "factors": {"score_a": 0.9, "score_b": 0.1}},
                {"item_id": 2, "factors": {"score_a": 0.5, "score_b": 0.4}},
                {"item_id": 3, "factors": {"score_a": 0.2, "score_b": 0.7}},
            ],
        },
        {
            "request_id": "req-B",
            "user_id": 200,
            "request_time": "2024-01-01T12:05:00",
            "num_served": 10,
            "user_profile": {},
            "candidates": [
                {"item_id": 4, "factors": {"score_a": 0.8, "score_b": 0.2}},
                {"item_id": 5, "factors": {"score_a": 0.3, "score_b": 0.6}},
                {"item_id": 6, "factors": {"score_a": 0.1, "score_b": 0.9}},
            ],
        },
    ])

    df = PoolBuilder.read_logs(str(log_path))

    assert len(df) == 6
    assert {"request_id", "user_id", "item_id", "request_time", "score_a", "score_b"} <= set(df.columns)
    assert df.schema["user_id"] == pl.UInt64
    assert df.schema["item_id"] == pl.UInt64
    assert df.schema["request_time"] == pl.Datetime("us")


def test_read_logs_blank_lines_skipped(tmp_path):
    """Blank lines in the JSONL must not produce empty rows; only real entries
    show up in the parsed DataFrame."""
    log_path = tmp_path / "with_blanks.jsonl"
    valid_entry = {
        "request_id": "req-A",
        "user_id": 100,
        "request_time": "2024-01-01T12:00:00",
        "num_served": 10,
        "user_profile": {},
        "candidates": [{"item_id": 1, "factors": {"score_a": 0.9}}],
    }
    with open(log_path, "wb") as f:
        f.write(b"\n")
        f.write(orjson.dumps(valid_entry))
        f.write(b"\n\n")

    df = PoolBuilder.read_logs(str(log_path))

    assert len(df) == 1
    assert df["item_id"][0] == 1


def test_read_logs_carries_position_and_served(tmp_path):
    """Rows keep the logged rank (1-based) and whether the client saw the row:
    with `num_served = 2`, positions 1 and 2 were served, position 3 was not."""
    log_path = tmp_path / "candidates.jsonl"
    _write_jsonl(log_path, [{
        "request_id": "req-A",
        "user_id": 100,
        "request_time": "2024-01-01T12:00:00",
        "num_served": 2,
        "user_profile": {},
        "candidates": [{"item_id": i, "factors": {}} for i in (30, 10, 20)],
    }])

    df = PoolBuilder.read_logs(str(log_path))

    assert df.select("item_id", "position", "served").rows() == [(30, 1, True), (10, 2, True), (20, 3, False)]


@pytest.mark.parametrize(
    ("logged_time", "expected"),
    [
        pytest.param("2024-01-01T12:00:00+00:00", datetime(2024, 1, 1, 12), id="utc_offset_kept"),
        pytest.param("2024-01-01T15:00:00+03:00", datetime(2024, 1, 1, 12), id="non_utc_offset_converted_to_utc"),
        pytest.param("2024-01-01T12:00:00", datetime(2024, 1, 1, 12), id="naive_string_read_as_utc"),
    ],
)
def test_read_logs_request_time_is_naive_utc(tmp_path, logged_time: str, expected: datetime):
    """`request_time` is compared against epoch-second action timestamps, so
    every logged time must land on the same UTC clock whatever its offset."""
    log_path = tmp_path / "candidates.jsonl"
    _write_jsonl(log_path, [{
        "request_id": "req-A",
        "user_id": 100,
        "request_time": logged_time,
        "num_served": 10,
        "user_profile": {},
        "candidates": [{"item_id": 1, "factors": {}}],
    }])

    df = PoolBuilder.read_logs(str(log_path))

    assert df.schema["request_time"] == pl.Datetime("us")
    assert df["request_time"][0] == expected


def test_build_pools_from_log_end_to_end(tmp_path, dm_action: pl.DataFrame):
    """read_logs -> join_asof_actions_to_features -> build_dataset -> build_cbm_pool, as build_ranker drives it."""
    request_dt_1 = datetime.fromtimestamp(CART_UPDATE_TS - 100, tz=timezone.utc).replace(tzinfo=None)
    request_dt_2 = datetime.fromtimestamp(CART_UPDATE_TS - 60, tz=timezone.utc).replace(tzinfo=None)
    log_path = tmp_path / "candidates.jsonl"
    _write_jsonl(log_path, [
        {
            "request_id": "req-1",
            "user_id": USER_ID,
            "request_time": request_dt_1.isoformat(),
            "num_served": 10,
            "user_profile": {},
            "candidates": [{"item_id": ITEM_ID, "factors": {"score": 0.9}}],
        },
        {
            "request_id": "req-2",
            "user_id": USER_ID,
            "request_time": request_dt_2.isoformat(),
            "num_served": 10,
            "user_profile": {},
            "candidates": [{"item_id": ITEM_ID, "factors": {"score": 0.7}}],
        },
    ])
    builder = PoolBuilder(
        num_features=["score"],
        positive_actions=POSITIVE_ACTIONS,
        negative_actions=NEGATIVE_ACTIONS,
    )

    train_pool, test_pool = builder.build_pools_from_log(str(log_path), dm_action)

    assert isinstance(train_pool, Pool)
    assert isinstance(test_pool, Pool)
    assert train_pool.get_feature_names() == ["score"]
    assert test_pool.get_feature_names() == ["score"]
    assert train_pool.num_row() + test_pool.num_row() == 2
    assert train_pool.num_row() >= 1 and test_pool.num_row() >= 1


SPLIT_BASE = datetime(2024, 1, 1)


def _request_ids(df: pl.DataFrame) -> set[int]:
    return set(df["request_id"].to_list())


def _linear_time_dataset(n_requests: int, rows_per_request: int = 1) -> pl.DataFrame:
    """Build a `(request_id, request_time, item_id)` frame with `n_requests`
    requests at `SPLIT_BASE + Nh`, each repeated `rows_per_request` times."""
    rids = [rid for rid in range(1, n_requests + 1) for _ in range(rows_per_request)]
    times = [SPLIT_BASE + timedelta(hours=rid) for rid in rids]
    item_ids = list(range(len(rids)))
    return pl.DataFrame({
        "request_id": pl.Series(rids, dtype=pl.UInt64),
        "request_time": pl.Series(times).cast(pl.Datetime("us")),
        "item_id": pl.Series(item_ids, dtype=pl.UInt64),
    })


def test_train_test_split_by_time_earliest_in_train_latest_in_test():
    """Default `test_size` 0.2: 8 of 10 requests go to train."""
    train, test = PoolBuilder.train_test_split_by_time(_linear_time_dataset(10))

    assert _request_ids(train) == {1, 2, 3, 4, 5, 6, 7, 8}
    assert _request_ids(test) == {9, 10}


def test_train_test_split_orders_by_time_not_by_request_id():
    """Request ids are not chronological: id 3 is the earliest request, id 1 the latest.
    Sorting by id would put the latest request into train (a leak from the future)."""
    rids, hours = [1, 2, 3], [3, 2, 1]
    dataset = pl.DataFrame({
        "request_id": pl.Series(rids, dtype=pl.UInt64),
        "request_time": pl.Series([SPLIT_BASE + timedelta(hours=h) for h in hours]).cast(pl.Datetime("us")),
        "item_id": pl.Series(range(3), dtype=pl.UInt64),
    })

    train, test = PoolBuilder.train_test_split_by_time(dataset, test_size=0.34)

    assert _request_ids(train) == {3}
    assert _request_ids(test) == {1, 2}


def test_train_test_split_keeps_all_rows_of_a_request_together():
    """Rows of one request never split across train and test."""
    train, test = PoolBuilder.train_test_split_by_time(
        _linear_time_dataset(n_requests=5, rows_per_request=4), test_size=0.2,
    )

    assert _request_ids(train).isdisjoint(_request_ids(test))
    assert len(train) == 16
    assert len(test) == 4


def test_train_test_split_uses_min_request_time_per_request():
    """Request 1 has rows at 5 h and 1 h: its earliest row orders it first."""
    rids, hours = [1, 1, 2, 3], [5, 1, 2, 3]
    dataset = pl.DataFrame({
        "request_id": pl.Series(rids, dtype=pl.UInt64),
        "request_time": pl.Series([SPLIT_BASE + timedelta(hours=h) for h in hours]).cast(pl.Datetime("us")),
        "item_id": pl.Series(range(len(rids)), dtype=pl.UInt64),
    })

    train, test = PoolBuilder.train_test_split_by_time(dataset, test_size=0.34)

    assert _request_ids(train) == {1}
    assert _request_ids(test) == {2, 3}


def test_train_test_split_rounds_down_via_int_cast():
    """`int(n * (1 - test_size))` rounds toward zero, not nearest. With 7 requests
    and test_size=0.3 → split_idx = int(7 * 0.7) = int(4.9) = 4 → 4 train / 3 test."""
    train, test = PoolBuilder.train_test_split_by_time(_linear_time_dataset(7), test_size=0.3)

    assert len(_request_ids(train)) == 4
    assert len(_request_ids(test)) == 3

