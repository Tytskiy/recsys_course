import polars as pl
from tasks.features.statistics import compute_action_statistics, compute_user_history


def test_windowed_counters_are_zero_not_missing():
    """Valkey HSET merges fields, so a counter left out of the profile keeps its
    previous build's value. An entity idle for the last 30 days must get explicit 0s."""
    actions = pl.DataFrame({
        "dt": ["2024-01-01"],
        "item_id": [100],
        "action_type": ["AT_View"],
    })

    profiles = compute_action_statistics("2024-03-01", actions, user=False)

    assert profiles[100]["num_views"] == 1
    assert profiles[100]["num_views_30d"] == 0
    assert profiles[100]["num_views_7d"] == 0


def test_user_history_is_distinct_most_recent_first_and_capped():
    """`RecommenderEngine.get_user_history` truncates the stored list to the first
    N ids, so the builder must store distinct ids, newest first."""
    actions = pl.DataFrame({
        "user_id": [7, 7, 7, 7],
        "item_id": [1, 2, 3, 1],
        "timestamp": [10, 20, 30, 40],
        "action_type": ["AT_Purchase"] * 4,
    })

    profiles = compute_user_history(actions, max_num_items=2)

    assert profiles == {7: {"user_history_AT_Purchase": [1, 3]}}
