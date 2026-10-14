# User-item matrix

## About the task
Most matrix factorization or Item2Item decomposition algorithms (SLIM / EASE) take as input a sparse user-item matrix of target values, where in the simple case the target is a binary value reflecting whether an interaction exists.

However, all these algorithms work considerably better if the target is built in a more sophisticated way.
- The number of interactions instead of a binary marker
- Mixing different kinds of interactions
- Reweighting or transforming the matrix: for example, using BM25 for reweighting.

## BM25 reweighting
Inspired by Okapi BM25 from classical IR:
$$
\text{reweighted conf}(u,i) = \text{idf}(i) \cdot \frac{\text{conf}(u,i) * (k_1 + 1)}{\text{conf}(u,i) + k_1\left(1 - b + b \cdot \frac{L_u}{\overline{L}}\right)}
$$

$$
\text{idf}(i) = \log(N) - \log(1 + \text{df}(i))
$$

$$
L_u = \sum_{j} \text{conf}(u,j), \quad \overline{L} = \frac{1}{N}\sum_{u} L_u
$$

Where $N$ is the total number of users in the matrix and $\text{df}(i)$ is the number of users with $\text{conf}_{u,i} > 0$.

For items that almost all users interacted with, $\text{idf}(i) \le 0$. After reweighting, keep only the pairs with a positive weight: `implicit` treats negative values as negative feedback.


## Statement
Implement, with `polars`, building the target matrix in [`engine_builder/tasks/actions.py`](../../../engine_builder/tasks/actions.py), as well as the BM25 transform for it.
- For each user and item, collect the counts of the different actions.
- Converting the action counts, as polars columns, into the different kinds of targets is done for you.
- Then convert the dataframe into a sparse matrix.
- Do not forget the mappings from real IDs to matrix indices.

User and item indices follow the ascending order of their ids (ids are `u64`).

## How to check yourself
```bash
(cd engine_builder && uv run pytest tests/test_actions.py)
```
