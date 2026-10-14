# ItemToItemLists Candidate Generator

## About the task
ItemToItemLists is a simple candidate generator that takes the user's history, takes similar items for each item in it, merges them and returns them as candidates. The key step is defining similarity: for example, it can be based on content embeddings; in our case, the text name of the product.

## Statement
Implement the `extract_candidates` method of the `ItemToItemLists` candidate generator in [`engine/cg/item_to_item_lists.py`](../../../engine/src/engine/cg/item_to_item_lists.py).
- The lists are stored in the FeatureStorage. In the tests the storage is replaced by a dictionary-based fake, so this task's tests do not need the feature_storage task, but running the whole service does.
- The anchors are the first `num_items_from_history` unique items from `user_history[action_type]` (in history order; repeats do not take up a slot).
- The candidate order must follow the history first, and then the sort order within each item list.
- Do not forget to deduplicate candidates: an item that already appeared in the list of an earlier anchor is skipped; at most `num_candidates` in total.
- In the tests the storage is replaced by a dictionary that implements only `bulk_extract`, so for the engine this task does not depend on `feature_storage`.

Implement the training of this candidate generator in [`engine_builder/tasks/models/item_to_item.py`](../../../engine_builder/tasks/models/item_to_item.py):
- The content model is already given: `TextEmbedder` computes normalized embeddings of the names with a pretrained BERT (`all-MiniLM-L6-v2`). In the tests it is replaced by a TF-IDF model, so your job is the lists, their order and deduplication.
- The `ItemToItemTrainer` class has one method, `train`, which takes the item dataframe and returns lists `{ID: [ID, ...]}`. The ready-made function `train_item_to_item` packs them into item feature profiles `{ID: {feature_name: list}}` and returns them as `TrainingArtifacts.item_profiles`.
- You need to compute a feature that takes the `num_items_per_list` items closest by vector distance; use cosine similarity.
- The lists must be ordered `score desc, item_id asc` and must not contain the anchor item itself.

## How to check yourself
```bash
(cd engine && uv run pytest tests/unit/cgs/test_item_to_item.py)
(cd engine_builder && uv run pytest tests/models/test_item_to_item.py)
```
