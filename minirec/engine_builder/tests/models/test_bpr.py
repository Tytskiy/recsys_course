"""BPR (task `bpr`). Pinned: `optimizer_step` is one SGD step on the README loss for the triplet, in place,
touching only the user and the two items; `sample_unigram` draws the user and the positive as one
interaction and the negative by popularity; `fit` initialises the factors, then runs `max_iter` epochs of
`nnz` sampled triplets, skipping a triplet whose negative is a positive of the user; the fitted model
beats TopPop on held-out data.
"""

import math
from collections import Counter

import numpy as np
import pytest
import scipy.sparse as sps
from tasks.models.factorization import BPR, optimizer_step, sample_unigram
from tests.synthetic import SMALL, make_actions, recall_ndcg_at_k, split_ts, top_pop_factors, train_test


SIG2 = 1 / (1 + math.exp(2))


@pytest.mark.parametrize(
    ("user_factors", "item_factors", "user", "pos", "neg", "lr", "alpha", "expected_users", "expected_items"),
    [
        pytest.param(
            [[9.0], [1.0]], [[2.0], [7.0], [0.0]], 1, 0, 2, 0.1, 0.0,
            [[9.0], [1 + 0.2 * SIG2]], [[2 + 0.1 * SIG2], [7.0], [-0.1 * SIG2]],
            id="dim1_no_l2",
        ),
        pytest.param(
            [[0.1, -0.2, 0.3], [0.0, 0.05, -0.1]], [[0.2, -0.1, 0.0], [0.05, 0.1, -0.2], [-0.2, 0.0, 0.15]],
            0, 1, 2, 0.05, 0.01,
            [[0.1065122398, -0.1972751041, 0.2906628642], [0.0, 0.05, -0.1]],
            [[0.2, -0.1, 0.0], [0.0525998959, 0.0947002081, -0.1920253122], [-0.2025248959, 0.0052497919, 0.1420503122]],
            id="dim3_with_l2",
        ),
        pytest.param(
            [[0.3, -0.1, 0.25], [0.2, 0.4, -0.05]], [[-0.1, 0.2, 0.05], [0.05, -0.3, 0.1], [0.2, 0.1, -0.15]],
            1, 0, 2, 0.1, 0.02,
            [[0.3, -0.1, 0.25], [0.18437502, 0.404275, -0.039750014]],
            [[-0.08965001, 0.21989998, 0.047362503], [0.05, -0.3, 0.1], [0.18945001, 0.07950003, -0.14716251]],
            id="dim3_other_user",
        ),
    ],
)
def test_optimizer_step(user_factors, item_factors, user, pos, neg, lr, alpha, expected_users, expected_items):
    """In place on float32 factors; rows other than the user, the positive and the negative stay unchanged."""
    users = np.array(user_factors, dtype=np.float32)
    items = np.array(item_factors, dtype=np.float32)
    optimizer_step(users, items, user, pos, neg, lr, alpha)
    np.testing.assert_allclose(users, expected_users, atol=1e-5)
    np.testing.assert_allclose(items, expected_items, atol=1e-5)


ITEM_IDS = np.array([4, 1, 2, 4, 3, 3, 4, 4, 2, 3])
USER_IDS = np.array([1, 1, 2, 2, 3, 3, 4, 4, 3, 3])


@pytest.fixture(scope="module")
def unigram_draws() -> list[tuple[int, ...]]:
    rng = np.random.default_rng(0)
    return [tuple(int(v) for v in sample_unigram(ITEM_IDS, USER_IDS, rng)) for _ in range(100_000)]


def test_sample_unigram_user_and_positive_form_an_interaction(unigram_draws):
    pairs = set(zip(USER_IDS.tolist(), ITEM_IDS.tolist()))
    assert {(user, pos) for user, pos, _ in unigram_draws} <= pairs


@pytest.mark.parametrize(
    ("field", "ids"),
    [
        pytest.param(0, USER_IDS, id="user_by_activity"),
        pytest.param(1, ITEM_IDS, id="positive_by_popularity"),
        pytest.param(2, ITEM_IDS, id="negative_by_popularity"),
    ],
)
def test_sample_unigram_frequencies(unigram_draws, field, ids):
    sampled = Counter(draw[field] for draw in unigram_draws)
    expected = Counter(ids.tolist())
    for value, count in expected.items():
        assert sampled[value] / len(unigram_draws) == pytest.approx(count / len(ids), abs=0.01)


ACTIONS = sps.csr_array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])


def fit_with_fixed_triplet(triplet: tuple[int, int, int], max_iter: int) -> tuple[BPR, int]:
    model = BPR(dim=2, l2_alpha=0.01, embed_scale=0.1, max_iter=max_iter, learning_rate=0.05, random_seed=0)
    calls = 0

    def fixed_triplet():
        nonlocal calls
        calls += 1
        return triplet

    setattr(model, "sample_triplet", fixed_triplet)
    model.fit(ACTIONS)
    return model, calls


def initial_factors() -> BPR:
    model = BPR(dim=2, l2_alpha=0.01, embed_scale=0.1, learning_rate=0.05, random_seed=0)
    model.initialize_parameters(2, 3)
    return model


@pytest.mark.parametrize("max_iter", [pytest.param(1, id="one_epoch"), pytest.param(3, id="three_epochs")])
def test_fit_samples_nnz_triplets_per_epoch(max_iter):
    _, calls = fit_with_fixed_triplet((0, 0, 1), max_iter)
    assert calls == 3 * max_iter


@pytest.mark.parametrize("max_iter", [pytest.param(1, id="one_epoch"), pytest.param(3, id="three_epochs")])
def test_fit_steps_once_per_valid_triplet(max_iter):
    """(0, 0, 1) is valid: item 1 is not a positive of user 0, so every draw is one step from the initial factors."""
    model, _ = fit_with_fixed_triplet((0, 0, 1), max_iter)
    expected = initial_factors()
    for _ in range(3 * max_iter):
        optimizer_step(expected.user_factors, expected.item_factors, 0, 0, 1, 0.05, 0.01)
    np.testing.assert_allclose(model.user_factors, expected.user_factors, atol=1e-12)
    np.testing.assert_allclose(model.item_factors, expected.item_factors, atol=1e-12)


def test_fit_skips_triplet_with_positive_negative():
    """(0, 0, 2): item 2 is a positive of user 0, so no step is taken and the factors stay at their initial values."""
    model, calls = fit_with_fixed_triplet((0, 0, 2), max_iter=2)
    expected = initial_factors()
    assert calls == 6
    np.testing.assert_array_equal(model.user_factors, expected.user_factors)
    np.testing.assert_array_equal(model.item_factors, expected.item_factors)


def test_fit_beats_top_pop_on_held_out_log():
    """BPR generalises: trained on the first 50 days of a synthetic log with planted taste clusters
    (tests/synthetic.py), its top-20 finds the users' new cart items of the last 10 days at least twice
    as often as popularity does."""
    split = train_test(make_actions(SMALL), split_ts(SMALL))
    model = BPR(dim=32, l2_alpha=0.01, embed_scale=0.1, max_iter=10, learning_rate=0.05, random_seed=0)
    model.fit(split.train)
    recall, _ = recall_ndcg_at_k(model.user_factors, model.item_factors, split)
    top_pop_recall, _ = recall_ndcg_at_k(*top_pop_factors(split.train), split)
    # observed 3.1x (seeds 1-3: 3.1-3.7x); 2x is half the observed margin
    assert recall >= 2 * top_pop_recall
