# Explicit ALS solver

## About the task
[ALS, Alternating Least Squares](https://education.yandex.ru/handbook/ml/article/rekomendacii-na-osnove-matrichnyh-razlozhenij#alternating-least-squares-als) is a very popular matrix factorization algorithm.
In this task you will write your own simple Numpy solver for the ordinary (not Implicit) decomposition.

We predict the score with the following formula: $\hat{r}_{ui} = \langle p_u, q_i \rangle$, where $p_u \in \mathbb{R}^d$ and $q_i \in \mathbb{R}^d$ are the latent vectors (embeddings) of user $u$ and item $i$.

We optimize MSE with regularization:
$$
L = \sum_{(u, i) \in R} (\hat{r}_{ui} - r_{ui})^2 + \lambda_{emb} \left(\sum_{u \in U} \|p_u\|^2 + \sum_{i \in I} \|q_i\|^2\right)
$$

Here $R$ is the set of **observed** pairs $(u, i)$: the non-zero elements of the `actions` matrix. A zero in the matrix means "no rating", not "rating 0", so it does not enter the sum.

ALS fixes one of the matrices in turn and finds the exact minimum of $L$ over the other. Deriving the step formulas and coding them is up to you! Hint: with fixed $q_i$, the problem splits into independent problems per user, and the system for $p_u$ includes only the items that $u$ rated. For a user (item) without a single rating, the solution is zero.

## Statement
Implement the `ExplicitALS` matrix factorization algorithm in [`engine_builder/tasks/models/factorization.py`](../../../engine_builder/tasks/models/factorization.py).
- The `user_update` method replaces all $p_u$ with the exact minimum of $L$ for fixed $q_i$.
- The `item_update` method replaces all $q_i$ with the exact minimum of $L$ for fixed $p_u$.
- The `fit` method calls `initialize_model`, then `user_update` and `item_update` in a row `max_iter` times. $\lambda_{emb}$ is `l2_alpha` (the ready matrix `self.lambda_eye`).
- You may change other functions, but change signatures carefully: follow the tests!


The quality test trains the model on the basket matrix of a synthetic log ([`engine_builder/tests/synthetic.py`](../../../engine_builder/tests/synthetic.py)) and requires Recall@10 on the users' own interactions to be at least 4 times higher than TopPop's. Quality on held-out interactions is not evaluated: on implicit feedback, a loss over observed pairs only does not generalize (on the synthetic log it is no better than random ranking).

## How to check yourself
```bash
(cd engine_builder && uv run pytest tests/models/test_explicit_als.py)
```
