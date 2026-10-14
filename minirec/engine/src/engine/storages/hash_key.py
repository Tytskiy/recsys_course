from engine.interfaces import ID


def build_hash_key(
    user_id: ID | None = None,
    item_id: ID | None = None,
) -> str:
    if user_id is None and item_id is None:
        raise ValueError("At least one of user_id or item_id must be provided")
    parts: list[str] = []
    if user_id is not None:
        parts.append(f"user:{user_id}")
    if item_id is not None:
        parts.append(f"item:{item_id}")
    return ":".join(parts)
