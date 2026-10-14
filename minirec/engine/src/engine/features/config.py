STATISTIC_COUNTERS = [
    "num_views",
    "num_clicks",
    "num_cart_updates",
    "num_views_30d",
    "num_clicks_30d",
    "num_cart_updates_30d",
    "num_views_7d",
    "num_clicks_7d",
    "num_cart_updates_7d",
]
ITEM_STATISTICS = [f"item:{fn}" for fn in STATISTIC_COUNTERS]
USER_STATISTICS = [f"user:{fn}" for fn in STATISTIC_COUNTERS]


ITEM_LEVEL_FEATURES = [
    "iALS_score",
    "iALS_to_history",
    "iALS_context_score",
    *ITEM_STATISTICS,
]
REQUEST_LEVEL_FEATURES = ["source_type", *USER_STATISTICS]


ITEM_LEVEL_TRANSFORMS = [
    "item:cart_update_turn_rate",
    "item:cart_update_turn_rate_30d",
]
REQUEST_LEVEL_TRANSFORMS = [
    "user:cart_update_turn_rate",
    "user:cart_update_turn_rate_30d",
]


FEATURE_TO_STORAGE_KEYS = {f"item:{fn}": fn for fn in STATISTIC_COUNTERS} | {
    f"user:{fn}": fn for fn in STATISTIC_COUNTERS
}
FEATURE_TO_USER_EMBEDDINGS = {
    "iALS_score": "user_iALS",
}


AVAILABLE_FEATURES = (
    ITEM_LEVEL_FEATURES
    + REQUEST_LEVEL_FEATURES
    + ITEM_LEVEL_TRANSFORMS
    + REQUEST_LEVEL_TRANSFORMS
)
