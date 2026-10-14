"""Explicit ALS (task `explicit_als`). Pinned: `user_update` / `item_update` solve the README normal
equations over the observed (non-zero) entries only, a row without ratings gets 0, the README-loss gradient
is zero after each half-step, `fit` = `initialize_model` + `max_iter` rounds of user then item update, and
the fitted model ranks the users' own items far above popularity.
"""

import numpy as np
import pytest
import scipy.sparse as sps
from tasks.evaluation.metric import Recall
from tasks.models.factorization import ExplicitALS
from tests.synthetic import SMALL, make_actions, split_ts, top_pop_factors, train_test


def make_model(actions: list[list[float]], dim: int, l2_alpha: float, seed: int = 0) -> ExplicitALS:
    model = ExplicitALS(dim=dim, l2_alpha=l2_alpha, embed_scale=0.1, max_iter=1, random_seed=seed)
    model.initialize_model(sps.csr_array(actions))
    return model


@pytest.mark.parametrize(
    ("actions", "item_factors", "expected"),
    [
        # observed items 0, 2: p = (1*1 + 2*3) / (1^2 + 3^2 + 1) = 7/11 (a dense solve over item 1 gives 7/15)
        pytest.param([[1.0, 0.0, 2.0]], [[1.0], [2.0], [3.0]], [[7 / 11]], id="unobserved_item_excluded"),
        # observed items 0, 1: (diag(1, 4) + I) p = (3*[1, 0] + 1*[0, 2]) -> p = [3/2, 2/5]; item 2 plays no part
        pytest.param([[3.0, 1.0, 0.0]], [[1.0, 0.0], [0.0, 2.0], [5.0, 5.0]], [[1.5, 0.4]], id="two_dims"),
        pytest.param([[0.0, 0.0, 0.0]], [[1.0], [2.0], [3.0]], [[0.0]], id="user_without_ratings_is_zero"),
    ],
)
def test_user_update_solves_over_observed_items(actions, item_factors, expected):
    """`user_update` minimises the README loss, whose sum runs over rated (u, i) only."""
    model = make_model(actions, dim=len(item_factors[0]), l2_alpha=1.0)
    model.item_factors = np.array(item_factors)
    model.user_update()
    np.testing.assert_allclose(model.user_factors, expected, atol=1e-6)


@pytest.mark.parametrize(
    ("actions", "user_factors", "expected"),
    [
        # item rated by users 0, 2: q = (1*1 + 2*3) / (1^2 + 3^2 + 1) = 7/11
        pytest.param([[1.0], [0.0], [2.0]], [[1.0], [2.0], [3.0]], [[7 / 11]], id="unobserved_user_excluded"),
        pytest.param([[0.0], [0.0], [0.0]], [[1.0], [2.0], [3.0]], [[0.0]], id="item_without_ratings_is_zero"),
    ],
)
def test_item_update_solves_over_observed_users(actions, user_factors, expected):
    """`item_update` is the same solve with the roles of users and items swapped."""
    model = make_model(actions, dim=1, l2_alpha=1.0)
    model.user_factors = np.array(user_factors)
    model.item_update()
    np.testing.assert_allclose(model.item_factors, expected, atol=1e-6)


def readme_loss_gradients(model: ExplicitALS) -> tuple[np.ndarray, np.ndarray]:
    ratings = model.actions.toarray()
    residual = (ratings != 0) * (model.user_factors @ model.item_factors.T - ratings)
    grad_p = 2 * residual @ model.item_factors + 2 * model.l2_alpha * model.user_factors
    grad_q = 2 * residual.T @ model.user_factors + 2 * model.l2_alpha * model.item_factors
    return grad_p, grad_q


@pytest.mark.parametrize("update", ["user_update", "item_update"])
def test_update_zeroes_its_loss_gradient(update):
    """An exact ALS half-step leaves the README-loss gradient w.r.t. the updated factors at zero."""
    rng = np.random.default_rng(0)
    actions = (rng.random((6, 5)) < 0.5) * rng.integers(1, 6, size=(6, 5)).astype(float)
    model = make_model(actions.tolist(), dim=3, l2_alpha=0.1)
    getattr(model, update)()
    grad_p, grad_q = readme_loss_gradients(model)
    np.testing.assert_allclose(grad_p if update == "user_update" else grad_q, 0.0, atol=1e-8)


@pytest.mark.parametrize("max_iter", [pytest.param(1, id="one_round"), pytest.param(3, id="three_rounds")])
def test_fit_alternates_user_then_item_updates(max_iter):
    """`fit` = `initialize_model`, then `max_iter` rounds of `user_update` followed by `item_update`."""
    actions = [[1.0, 0.0, 2.0], [0.0, 3.0, 1.0]]
    fitted = ExplicitALS(dim=2, l2_alpha=0.05, embed_scale=0.1, max_iter=max_iter, random_seed=0)
    fitted.fit(sps.csr_array(actions))

    manual = make_model(actions, dim=2, l2_alpha=0.05)
    for _ in range(max_iter):
        manual.user_update()
        manual.item_update()

    np.testing.assert_allclose(fitted.user_factors, manual.user_factors, atol=1e-10)
    np.testing.assert_allclose(fitted.item_factors, manual.item_factors, atol=1e-10)


def mean_recall_at_k(scores: np.ndarray, positives: sps.csr_array, k: int) -> float:
    """Mean Recall@k (the provided `Recall`) of each row's top-k against its non-zero columns."""
    recall = Recall(k=k)
    values = []
    for u in range(positives.shape[0]):
        items = positives.indices[positives.indptr[u] : positives.indptr[u + 1]]
        if len(items):
            values.append(recall.compute(np.argsort(-scores[u], kind="stable")[:k].tolist(), items.tolist()))
    return float(np.mean(values))


def test_fit_ranks_own_items_above_top_pop():
    """The fitted model recovers the users' own interactions (the train counts of the synthetic log,
    tests/synthetic.py) far better than popularity does.
    Non-goal: held-out quality. On implicit counts the observed-only objective does not generalise: held-out
    Recall@20 is 0.017 vs TopPop 0.087, while the wrong dense solve (zeros as ratings) scores 0.285, so a
    held-out bound would reward the wrong solution."""
    train = train_test(make_actions(SMALL), split_ts(SMALL)).train
    model = ExplicitALS(dim=32, l2_alpha=0.05, embed_scale=0.1, max_iter=10, random_seed=0)
    model.fit(train)
    recall = mean_recall_at_k(model.user_factors @ model.item_factors.T, train, k=10)
    user_ones, popularity = top_pop_factors(train)
    top_pop_recall = mean_recall_at_k(user_ones @ popularity.T, train, k=10)
    # observed 8.1x (seeds 1-2: 8.6x, 9.2x); 4x is half the observed margin
    assert recall >= 4 * top_pop_recall
