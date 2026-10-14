# Bayesian Personalized Ranking

## About the task
[BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/pdf/1205.2618) is an algorithm for training recommendation models with a matrix factorization backbone (neural networks can be used too). Unlike the more classical methods you covered in the seminar, BPR does not try to predict the probability or the absolute value of positive feedback for a user-item pair; instead it aims to rank different pairs correctly, assuming that the items the user interacted with have higher priority than those without interactions.

The model is trained with SGD (or another optimizer), and its pairwise loss function maximizes the probability that $\hat{x}_{u,pos}$ is higher than $\hat{x}_{u,neg}$:

$$L(u, i, j) = -\ln \sigma(\hat{x}_{ui} - \hat{x}_{uj}) + \frac{\lambda}{2} || \Theta||^2$$

Here $\hat{x}_{ui}$ is the model score, in this case of a matrix factorization without biases:
$$\hat{x}_{ui} = \langle p_u, q_i \rangle$$

Positives are the user's history, negatives are items sampled at random from the catalog (uniformly, by unigram frequency or in a more complex way); the second term is the L2 regularization of the model parameters.

## Statement
Implement the `BPR` matrix factorization algorithm in [`engine_builder/tasks/models/factorization.py`](../../../engine_builder/tasks/models/factorization.py).
- The `optimizer_step` function, for a triple of sampled IDs, must compute the loss and do one optimization step for all model parameters with the given parameters.
- The `sample_unigram` function, for the sequence of all interactions in the dataset represented as two ID arrays of equal length, must sample a random user, their positive and a random negative; we do not guarantee that the user did not interact with the negative.
- In the model class, implement the training loop in the `fit` method: you need to run several epochs, in each of which you sample a triple from the dataset `num_actions` times (`num_actions` is the dataset length), check that it is valid and do an optimization step for it if its negative is not a positive for the user. `num_actions` is the full number of triples per epoch, including invalid ones.
- You may change other functions, but change signatures carefully: follow the tests!

*Hint:* passing some of the task's tests may depend on the chosen way of sampling triples. The reference solution assumes that unigram negative sampling is done without explicitly computing per-item probabilities/counts.


The quality test trains the model on the first 50 days of a synthetic log with built-in taste clusters ([`engine_builder/tests/synthetic.py`](../../../engine_builder/tests/synthetic.py)) and requires Recall@20 on new items in baskets over the last 10 days (items already seen by the user are excluded) to be at least 2 times higher than TopPop's.

## How to check yourself
```bash
(cd engine_builder && uv run pytest tests/models/test_bpr.py)
```
