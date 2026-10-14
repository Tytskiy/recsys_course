# Implicit ALS Candidate Generator

## About the task
[iALS, Alternating Least Squares for Implicit Feedback](https://education.yandex.ru/handbook/ml/article/rekomendacii-na-osnove-matrichnyh-razlozhenij#ials-implicit-als) is a variant of ALS that uses implicit feedback and builds `confidence` from it: the confidence in the importance of the feedback for a specific `user-item` pair.

Writing a fast Python solver for this variant is already hard, because it requires parallelization over `item/user` and solving a linear system for each object separately; the overhead becomes too large even with numpy. Fortunately, there is the `implicit` library, which provides such a solver in `Cython` with a convenient API.

In this task you will write a small module for training this model and exporting it to the embedding storage.

## Statement
Implement the `UserToItemKNN` candidate generator in [`engine/cg/user_to_item_knn.py`](../../../engine/src/engine/cg/user_to_item_knn.py).
- Use the EmbeddingStorage and its methods; that is practically all the logic.
- The needed user embedding is in `context.user_profile[f"user_{embedding_type}"]` (for example, `user_iALS`); it is put there by `RuntimeIALS` or the feature manager. If there is no embedding, there are no candidates.

Implement the training of this candidate generator in [`engine_builder/tasks/models/implicit_als.py`](../../../engine_builder/tasks/models/implicit_als.py):
- Implement splitting the `user-item` matrix into two by users, warm and cold, by the sum of the matrix row values (with BM25 this is the sum of weights): a warm user has a sum `>= warm_user_threshold`.
- For warm users you can use the ordinary training procedure, and for cold users do only one step (the implicit framework API allows this out of the box).
- Warm users are trained by `implicit.als.AlternatingLeastSquares(factors=dim, regularization=l2_alpha, alpha=conf_alpha, iterations=max_iter, random_state=random_state)` on the warm users' rows; the tests compare their embeddings with this model, so the rows of the returned matrix must keep the order of the input rows.
- Implement the training process itself with the given parameters and the export to Qdrant; study the ready functions in `engine_builder`.

The quality test trains the model on the first 50 days of a synthetic log with built-in taste clusters ([`engine_builder/tests/synthetic.py`](../../../engine_builder/tests/synthetic.py)) and requires Recall@20 on new items in baskets over the last 10 days (items already seen by the user are excluded) to be at least 2 times higher than TopPop's.

## How to check yourself
```bash
(cd engine && uv run pytest tests/unit/cgs/test_user_to_item_knn.py)
(cd engine_builder && uv run pytest tests/models/test_implicit_als.py)
```

Both test sets must pass: the candidate generator and the training.
