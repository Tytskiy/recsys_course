"""Seeded synthetic action log with planted cluster structure, a temporal train/test split and a
chunked top-k evaluator: the shared fixture for held-out quality checks.

Test scaffolding, not student code: tests import it as `tests.synthetic`.

Determinism rule for `make_actions`: only PCG64 integer and uniform draws, `+ - * /`, `cumsum` and
`searchsorted`. No `exp`/`log`/`pow`, `rng.geometric` or `arr.sum()` on floats (pairwise, SIMD-dependent),
so the log is bit-identical across platforms.
"""

import dataclasses

import numpy as np
import polars as pl
import scipy.sparse as sps

DAY = 86_400
T0 = 1_700_006_400  # 2023-11-15 00:00:00 UTC
MAX_EVENTS = 400


@dataclasses.dataclass(frozen=True)
class SynthConfig:
    n_users: int = 20_000
    n_items: int = 5_000
    n_clusters: int = 25
    days: int = 60
    mean_events: int = 14  # per user, geometric length
    in_cluster: float = 0.75  # P(an event comes from the user's own cluster(s))
    second_cluster: float = 0.3  # P(a user has a second cluster)
    late_items: float = 0.08  # share of items launched in the last 25% of the period
    late_users: float = 0.08  # share of users whose first event is in the last 20%
    click_ratio: float = 0.5  # uniform AT_Click noise rows per cart update
    seed: int = 0


LARGE = SynthConfig()
SMALL = SynthConfig(n_users=5_000, n_items=1_500, n_clusters=15)


def make_actions(cfg: SynthConfig = SMALL) -> pl.DataFrame:
    """`dm_action`-shaped log (`user_id`, `item_id` u64 ids, `timestamp`, `action_type`, `dt`).

    Item i belongs to cluster `perm[i] % K` with Zipf popularity `1 / (rank_in_cluster + 3)`. A user
    has one cluster (two with p=`second_cluster`); each `AT_CartUpdate` comes from the user's
    cluster(s) with p=`in_cluster`, otherwise from global popularity. Repeats become counts > 1.
    """
    rng = np.random.default_rng(cfg.seed)
    n_users, n_items, n_clusters = cfg.n_users, cfg.n_items, cfg.n_clusters

    perm = rng.permutation(n_items)
    item_cluster = perm % n_clusters
    pop = 1.0 / (perm // n_clusters + 3.0)
    launch = np.zeros(n_items, dtype=np.int64)
    late = rng.random(n_items) < cfg.late_items
    launch[late] = rng.integers(int(0.75 * cfg.days) * DAY, cfg.days * DAY, late.sum())

    cluster_items = [np.flatnonzero(item_cluster == k) for k in range(n_clusters)]
    cluster_cdf = [_cdf(pop[ix]) for ix in cluster_items]
    global_cdf = _cdf(pop)

    first_cluster = rng.integers(0, n_clusters, n_users)
    second_cluster = np.where(
        rng.random(n_users) < cfg.second_cluster,
        rng.integers(0, n_clusters, n_users),
        first_cluster,
    )
    # 1 + geometric length; the pmf is built by repeated multiplication (no libm)
    q = 1.0 - 1.0 / cfg.mean_events
    pmf = np.empty(MAX_EVENTS)
    pmf[0] = 1.0
    for j in range(1, MAX_EVENTS):
        pmf[j] = pmf[j - 1] * q
    lengths = np.minimum(
        1 + np.searchsorted(_cdf(pmf), rng.random(n_users), side="right"), MAX_EVENTS
    )
    start = np.zeros(n_users, dtype=np.int64)
    late_u = rng.random(n_users) < cfg.late_users
    start[late_u] = rng.integers(
        int(0.8 * cfg.days) * DAY, cfg.days * DAY - DAY, late_u.sum()
    )

    users = np.repeat(np.arange(n_users), lengths)
    n = users.size
    ts = start[users] + (rng.random(n) * (cfg.days * DAY - start[users])).astype(
        np.int64
    )
    use_cluster = rng.random(n) < cfg.in_cluster
    pick_second = rng.random(n) < 0.5
    cluster = np.where(pick_second, second_cluster[users], first_cluster[users])
    u01 = rng.random(n)
    items = np.empty(n, dtype=np.int64)
    items[~use_cluster] = np.searchsorted(global_cdf, u01[~use_cluster], side="right")
    for k in range(n_clusters):
        m = use_cluster & (cluster == k)
        items[m] = cluster_items[k][
            np.searchsorted(cluster_cdf[k], u01[m], side="right")
        ]
    items = np.minimum(items, n_items - 1)
    before = ts < launch[items]
    ts[before] = launch[items[before]] + (
        rng.random(before.sum()) * (cfg.days * DAY - launch[items[before]])
    ).astype(np.int64)

    user_ids = rng.integers(1, 2**63 - 1, n_users, dtype=np.int64).astype(np.uint64)
    item_ids = rng.integers(1, 2**63 - 1, n_items, dtype=np.int64).astype(np.uint64)
    carts = pl.DataFrame(
        {
            "user_id": user_ids[users],
            "item_id": item_ids[items],
            "timestamp": T0 + ts,
            "action_type": "AT_CartUpdate",
        }
    )
    n_clicks = int(cfg.click_ratio * n)
    clicks = pl.DataFrame(
        {
            "user_id": user_ids[rng.integers(0, n_users, n_clicks)],
            "item_id": item_ids[rng.integers(0, n_items, n_clicks)],
            "timestamp": T0 + rng.integers(0, cfg.days * DAY, n_clicks),
            "action_type": "AT_Click",
        }
    )
    return (
        pl.concat([carts, clicks])
        .with_columns(
            pl.from_epoch("timestamp", time_unit="s")
            .dt.strftime("%Y-%m-%d")
            .alias("dt")
        )
        .sort("timestamp", "user_id", "item_id")
    )


def _cdf(weights: np.ndarray) -> np.ndarray:
    c = np.cumsum(weights)
    return c / c[-1]


def split_ts(cfg: SynthConfig = SMALL, test_days: int = 10) -> int:
    """First timestamp of the test period: the last `test_days` days."""
    return T0 + (cfg.days - test_days) * DAY


@dataclasses.dataclass(frozen=True)
class TrainTest:
    train: sps.csr_array  # action counts, train users x train items
    test: sps.csr_array  # binary, same index space, pairs not already in train
    user_ids: np.ndarray  # row index -> user_id, ascending
    item_ids: np.ndarray  # column index -> item_id, ascending


def train_test(
    actions: pl.DataFrame, ts: int, action_type: str = "AT_CartUpdate"
) -> TrainTest:
    """Rows with `timestamp < ts` form train, the rest test. The index space comes from train only;
    test pairs of cold users or cold items are dropped, and so are pairs already seen in train."""
    rows = actions.filter(pl.col("action_type") == action_type)
    counts = (
        rows.filter(pl.col("timestamp") < ts)
        .group_by("user_id", "item_id")
        .agg(pl.len().cast(pl.Float64).alias("count"))
    )
    users = counts.select("user_id").unique().sort("user_id").with_row_index("u")
    items = counts.select("item_id").unique().sort("item_id").with_row_index("i")
    shape = (users.height, items.height)
    c = counts.join(users, on="user_id").join(items, on="item_id")
    train = sps.csr_array(
        (c["count"].to_numpy(), (c["u"].to_numpy(), c["i"].to_numpy())), shape=shape
    )
    t = (
        rows.filter(pl.col("timestamp") >= ts)
        .select("user_id", "item_id")
        .unique()
        .join(counts, on=["user_id", "item_id"], how="anti")
        .join(users, on="user_id")
        .join(items, on="item_id")
    )
    test = sps.csr_array(
        (np.ones(t.height), (t["u"].to_numpy(), t["i"].to_numpy())), shape=shape
    )
    return TrainTest(
        train, test, users["user_id"].to_numpy(), items["item_id"].to_numpy()
    )


def top_pop_factors(train: sps.csr_array) -> tuple[np.ndarray, np.ndarray]:
    """TopPop as a rank-1 factor model: every user scores an item by its train user count."""
    return np.ones((train.shape[0], 1)), np.asarray(
        (train > 0).sum(axis=0), dtype=np.float64
    ).reshape(-1, 1)


def recall_ndcg_at_k(
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    split: TrainTest,
    k: int = 20,
    chunk: int = 1024,
) -> tuple[float, float]:
    """Mean Recall@k and NDCG@k (log2 discount, IDCG over min(|test|, k)) over users with test pairs.

    Scores are `user_factors @ item_factors.T`, train items are excluded, ties go to the lower item
    index; at most `chunk` x n_items scores exist at a time.
    """
    users = np.flatnonzero(np.diff(split.test.indptr) > 0)
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    ideal = np.cumsum(discount)
    recall, ndcg = [], []
    for start in range(0, len(users), chunk):
        u = users[start : start + chunk]
        scores = np.asarray(user_factors[u] @ item_factors.T, dtype=np.float64)
        seen = split.train[u]
        scores[
            np.repeat(np.arange(len(u)), np.diff(seen.indptr)), seen.indices
        ] = -np.inf
        # exact tie contract, independent of argpartition's platform-dependent choice among ties:
        # everything above the k-th score, then the lowest indices among items equal to it
        kth = -np.partition(-scores, k - 1, axis=1)[:, k - 1 : k]
        above = scores > kth
        at_kth = scores == kth
        room = k - above.sum(axis=1, keepdims=True)
        top = np.nonzero(above | (at_kth & (np.cumsum(at_kth, axis=1) <= room)))[1]
        top = top.reshape(len(u), k)
        order = np.lexsort((top, -np.take_along_axis(scores, top, axis=1)), axis=1)
        top = np.take_along_axis(top, order, axis=1)
        test = split.test[u]
        hits = (
            np.asarray(test[np.repeat(np.arange(len(u)), k), top.ravel()]).reshape(
                len(u), k
            )
            > 0
        )
        n_pos = np.diff(test.indptr)
        recall.append(hits.sum(axis=1) / n_pos)
        ndcg.append((hits * discount).sum(axis=1) / ideal[np.minimum(n_pos, k) - 1])
    return float(np.concatenate(recall).mean()), float(np.concatenate(ndcg).mean())
