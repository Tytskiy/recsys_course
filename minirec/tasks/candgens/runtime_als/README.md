# Runtime iALS

## About the task
iALS is a very good candidate generator, but in the usual setup it has several problems:
- We know nothing about new users
- For existing users it gradually loses relevance unless we retrain regularly
- There is no easy way to take the latest information about the user into account at runtime
- We need to store many user embeddings; sometimes we want to trade memory for time

One way to beat this is to store the item embeddings after training and get the user representation at runtime, doing one optimization step over all embeddings from the user's recent history to compute an up-to-date embedding. This way we can solve all the problems above at the cost of extra computation during the request; however, this approach requires a fairly large user history available at runtime and item embeddings of good enough quality and stability (it is still better to retrain them, but you can do it less often).

## Statement
Implement the `RuntimeIALS` component in [`engine/src/engine/runtime_profile/runtime_ials.py`](../../../engine/src/engine/runtime_profile/runtime_ials.py).
- Collect the user's history (the first `max_history` elements of `context.user_history[action_type]`), count each item and get the embeddings of the unique items (`item_{embedding_type}`) from the `EmbeddingStorage` with a single `get` call. Items without an embedding are skipped; rows are ordered by item id.
- Build the vector `C` as in the `implicit` library: $C = \alpha_{conf} \cdot cnt(action)$, where $\alpha_{conf}$ is the same `alpha` as in offline iALS training. The vector must be consistent in order with the embedding matrix.
- Do an iALS step for one user by solving the system of normal equations, where $\text{factors}$ is the matrix of embeddings of the history items, $Y^TY$ is the Gram matrix of **all** items (`self.item_gramian`, ready and loaded from the snapshot), and $u$ is the user embedding being sought:

$$(Y^TY + \text{factors}^T \cdot diag(C - 1) \cdot \text{factors} + \alpha_{reg} \cdot I) \cdot u = \text{factors}^T \cdot C$$

  The $Y^TY$ term is the contribution of the items the user did not interact with: in iALS they have confidence 1 and target value 0. Without it, the solution does not match `recalculate_user` from `implicit`, and the online embedding diverges from the offline one.
- Put this embedding into `context.user_profile[f"user_{embedding_type}"]` (for example, `user_iALS`): `UserToItemKNN` reads it from there. If no history item has an embedding, write nothing.

The tests use a stub embedding storage, so the task does not depend on your `embedding_storage`.

## How to check yourself
```bash
(cd engine && uv run pytest tests/unit/runtime/test_runtime_ials.py)
```
