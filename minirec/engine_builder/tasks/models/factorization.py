from collections import defaultdict

import numpy as np
import scipy.sparse as sps
from loguru import logger
from numba import njit
from tqdm.auto import trange


class MatrixFactorization:
    def __init__(
        self,
        dim: int = 128,
        l2_alpha: float = 0.01,
        embed_scale: float = 1.0,
        max_iter: int = 100,
        random_seed: int = 0,
    ):
        self.dim = dim
        self.l2_alpha = l2_alpha
        self.max_iter = max_iter
        self.embed_scale = embed_scale
        self.rng = np.random.default_rng(random_seed)
        self.user_factors: np.ndarray
        self.item_factors: np.ndarray
    
    def initialize_parameters(self, num_users, num_items):
        self.user_factors = self.rng.normal(size=(num_users, self.dim), scale=self.embed_scale)
        self.item_factors = self.rng.normal(size=(num_items, self.dim), scale=self.embed_scale)

    def export_factors(self):
        return {"user": self.user_factors, "item": self.item_factors}


class ExplicitALS(MatrixFactorization):
    def user_update(self):
        """Replace every `p_u` with the minimiser of the README loss for fixed `Q`,
        summing over the items `u` rated (non-zero entries of `self.actions` row `u`)."""
        # TODO: Your solution

    def item_update(self):
        """Replace every `q_i` with the minimiser of the README loss for fixed `P`,
        summing over the users who rated `i` (non-zero entries of column `i`)."""
        # TODO: Your solution

    def initialize_model(self, actions):
        self.actions = actions
        num_users, num_items = actions.shape
        self.initialize_parameters(num_users, num_items)
        self.lambda_eye = self.l2_alpha * np.eye(self.dim)

    def fit(self, actions: sps.sparray):
        """Initialize the model on `actions`, then run `max_iter` rounds of
        `user_update` followed by `item_update`."""
        # TODO: Your solution


@njit
def optimizer_step(user_factors, item_factors, user_idx, pos_item_idx, neg_item_idx, lr, alpha):
    """
    Optimization step for BPR model:
    1. Calculate the dot-product scores for the positive and negative items.
    2. Apply sigmoid function to the difference of the scores to get P(u, i > j).
    3. Calculate the gradients for the user and item vectors.
    4. Update inplace the user and item vectors.
    """
    # TODO: Your solution


@njit
def sample_unigram(item_ids: np.ndarray, user_ids: np.ndarray, rng: np.random.Generator):
    """Sample by popularity from two sequences (user_id, item_id) for each action"""
    # TODO: Your solution
    return user_id, positive_id, negative_id


class BPR(MatrixFactorization):
    def __init__(
        self,
        dim: int = 128,
        l2_alpha: float = 0.01,
        embed_scale: float = 0.2,
        max_iter: int = 100,
        learning_rate: float = 0.01,
        random_seed: int = 0,
    ):
        super().__init__(dim, l2_alpha, embed_scale, max_iter, random_seed)
        self.learning_rate = learning_rate
        self.item_ids: np.ndarray
        self.user_ids: np.ndarray

    def sample_triplet(self):
        return sample_unigram(self.item_ids, self.user_ids, self.rng)
    
    # TODO: Your solution

    def fit(self, actions: sps.sparray):
        """Run `max_iter` epochs of `nnz` sampled triplets each; skip a triplet whose
        negative is a positive of the user, otherwise take one `optimizer_step`.
        """
        skipped = 0
        # TODO: Your solution
        logger.info(f"Skipped triplets: {100 * skipped / len(self.item_ids) / self.max_iter:.1f}%")
