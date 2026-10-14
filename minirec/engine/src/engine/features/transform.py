from engine.interfaces import (
    ID,
    Candidate,
    Factor,
    RecContext,
)


def item_level_cart_update_turn_rate(
    candidate_map: dict[ID, Candidate],
    period: str | None = None,
) -> dict[ID, Factor]:
    view_key = f"item:num_views{period or ''}"
    cart_update_key = f"item:num_cart_updates{period or ''}"
    features = {}
    for iid, candidate in candidate_map.items():
        views: int = candidate.factors.get(view_key, 0) # ty: ignore[invalid-assignment]
        cart_updates: int = candidate.factors.get(cart_update_key, 0) # ty: ignore[invalid-assignment]
        features[iid] = float(cart_updates) / max(views, 1)
    return features


def request_level_cart_update_turn_rate(
    context: RecContext,
    period: str | None = None,
) -> Factor:
    views: int = context.user_profile.get(f"user:num_views{period or ''}", 0) # ty: ignore[invalid-assignment]
    cart_updates: int = context.user_profile.get(f"user:num_cart_updates{period or ''}", 0) # ty: ignore[invalid-assignment]
    return float(cart_updates) / max(views, 1)
