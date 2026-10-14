import numpy as np

from engine.interfaces import (
    ID,
    Candidate,
    Embedding,
    EmbeddingStorage,
    Factor,
    RecContext,
)


def user_item_dot_product(
    candidate_map: dict[ID, Candidate],
    embedding_type: str,
    item_storage: EmbeddingStorage,
    context: RecContext,
) -> dict[ID, Factor]:
    query: Embedding = context.user_profile.get(f"user_{embedding_type}") # ty: ignore[invalid-assignment]
    if query is None:
        return {}
    ids = list(candidate_map.keys())
    collection = f"item_{embedding_type}"
    return dict(item_storage.score(collection, query, ids))


def item_item_dot_product(
    candidate_map: dict[ID, Candidate],
    embedding_type: str,
    item_storage: EmbeddingStorage,
    context: RecContext,
) -> dict[ID, Factor]:
    collection = f"item_{embedding_type}"
    context_embeds = item_storage.get(collection, context.context_item_ids)
    if not context_embeds:
        return {}
    ids = list(candidate_map.keys())
    return dict(item_storage.score(collection, context_embeds[0][1], ids))


def item_to_history_mean_dot_product(
    candidate_map: dict[ID, Candidate],
    embedding_type: str,
    item_storage: EmbeddingStorage,
    context: RecContext,
    history_len: int = 10,
) -> dict[ID, Factor]:
    history_actions = context.user_history.get("AT_Purchase", [])[:history_len]
    collection = f"item_{embedding_type}"
    history_embeds = item_storage.get(collection, history_actions)
    if not history_embeds:
        return {}
    mean_embed = np.mean([embed for _, embed in history_embeds], axis=0)
    return dict(item_storage.score(collection, mean_embed, list(candidate_map.keys())))
