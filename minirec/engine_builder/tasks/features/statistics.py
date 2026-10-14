import polars as pl
from engine.features.config import STATISTIC_COUNTERS
from engine.interfaces import ID, ActionType, Factor

from tasks.actions import date_add

COUNTER_NAMES: dict[ActionType, str] = {
    "AT_View": "num_views",
    "AT_Click": "num_clicks",
    "AT_CartUpdate": "num_cart_updates",
}
HISTORY_ACTION_TYPES: tuple[ActionType, ...] = ("AT_Purchase", "AT_CartUpdate")


def aggregate_counts(frame: pl.DataFrame, key: str, suffix: str = "") -> pl.DataFrame:
    suffix = f"_{suffix}" if suffix else suffix
    return frame.group_by(key).agg(
        (pl.col("action_type") == action_type).sum().alias(f"{name}{suffix}")
        for action_type, name in COUNTER_NAMES.items()
    )


def compute_action_statistics(
    current_dt: str,
    actions: pl.DataFrame,
    user: bool = True,
) -> dict[ID, dict[str, Factor]]:
    key = "user_id" if user else "item_id"
    actions = actions.filter(pl.col("action_type").is_in(list(COUNTER_NAMES)))

    dt_30d_before = date_add(current_dt, -30)
    stats_30d = aggregate_counts(actions.filter(pl.col("dt") >= dt_30d_before), key, suffix="30d")

    dt_7d_before = date_add(current_dt, -7)
    stats_7d = aggregate_counts(actions.filter(pl.col("dt") >= dt_7d_before), key, suffix="7d")

    stats_full = aggregate_counts(actions, key)

    # write explicit zeros: Valkey HSET merges, so an omitted counter keeps its previous value
    statistics = (
        stats_full
        .join(stats_30d, on=key, how="left")
        .join(stats_7d, on=key, how="left")
        .fill_null(0)
    )
    return {
        row[key]: {feature: row[feature] for feature in STATISTIC_COUNTERS}
        for row in statistics.iter_rows(named=True)
    }


def compute_user_history(
    actions: pl.DataFrame,
    max_num_items: int = 30,
) -> dict[ID, dict[str, Factor]]:
    """`user_history_<action_type>` profiles read by `RecommenderEngine.get_user_history`:
    distinct item ids, most recent first, at most `max_num_items` per action type."""
    profiles: dict[ID, dict[str, Factor]] = {}
    for action_type in HISTORY_ACTION_TYPES:
        histories = (
            actions
            .filter(pl.col("action_type") == action_type)
            .sort("timestamp", descending=True)
            .unique(subset=["user_id", "item_id"], keep="first", maintain_order=True)
            .group_by("user_id", maintain_order=True)
            .agg(pl.col("item_id").head(max_num_items))
        )
        for user_id, item_ids in histories.iter_rows():
            profiles.setdefault(user_id, {})[f"user_history_{action_type}"] = item_ids
    return profiles
