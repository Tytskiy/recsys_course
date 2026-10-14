"""TopPop: `probabilities` is counts / sum(counts) in the key order of `counts`;
`extract_candidates` yields min(num_candidates, len(counts)) distinct known items (none for 0);
draws follow `probabilities` (checked statistically, not by exact draw);
the same seed gives the same draws, a different seed different ones."""

from __future__ import annotations

from collections import Counter

import numpy as np
import orjson
import pytest
from engine.cg import TopPop
from engine.cg.top_pop import TopPopConfig
from engine.interfaces import RecContext
from tests.fakes import make_context

COUNTS = {1: 10, 2: 20, 3: 30, 4: 40}


@pytest.fixture
def context() -> RecContext:
    return make_context()


def draw(cg: TopPop, num_candidates: int, context: RecContext) -> list[int]:
    return [c.item_id for c in cg.extract_candidates(num_candidates, context)]


def test_start_component_reads_counts(rec_engine_files_dir, deps):
    rec_engine_files_dir.mkdir(parents=True, exist_ok=True)
    (rec_engine_files_dir / "counts.json").write_bytes(orjson.dumps(COUNTS, option=orjson.OPT_NON_STR_KEYS))

    cg = TopPop.start_component(TopPopConfig(path="counts.json", seed=123), deps)

    assert isinstance(cg, TopPop) and cg.name == "TopPop"
    assert np.allclose(cg.probabilities, [0.1, 0.2, 0.3, 0.4])


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        pytest.param({7: 1, 8: 3}, [0.25, 0.75], id="two_items"),
        pytest.param({9: 3, 5: 1}, [0.75, 0.25], id="key_order_kept"),
        pytest.param({42: 5}, [1.0], id="single_item"),
    ],
)
def test_probabilities_are_normalised_counts(counts, expected):
    assert np.allclose(TopPop(counts=counts, seed=0).probabilities, expected)


@pytest.mark.parametrize(
    ("num_candidates", "expected_len"),
    [
        pytest.param(0, 0, id="zero"),
        pytest.param(1, 1, id="one"),
        pytest.param(3, 3, id="fewer_than_items"),
        pytest.param(4, 4, id="exactly_all_items"),
        pytest.param(5, 4, id="one_past_all_items_capped"),
        pytest.param(100, 4, id="far_past_all_items_capped"),
    ],
)
def test_extract_returns_distinct_known_items(context, num_candidates, expected_len):
    cg = TopPop(counts=COUNTS, seed=123)
    for _ in range(20):
        items = draw(cg, num_candidates, context)
        assert len(items) == expected_len
        assert len(set(items)) == expected_len
        assert set(items) <= set(COUNTS)


def test_draws_follow_probabilities(context):
    """Over many single draws, each item's share is close to its probability.
    Tolerance 0.04 is about 5 standard deviations at 4000 draws."""
    cg = TopPop(counts=COUNTS, seed=7)
    num_draws = 4000
    first = Counter(draw(cg, 1, context)[0] for _ in range(num_draws))
    for item_id, p in zip(COUNTS, [0.1, 0.2, 0.3, 0.4]):
        assert abs(first[item_id] / num_draws - p) < 0.04, item_id


def test_same_seed_same_draws(context):
    left, right = TopPop(counts=COUNTS, seed=11), TopPop(counts=COUNTS, seed=11)
    assert [draw(left, 2, context) for _ in range(30)] == [draw(right, 2, context) for _ in range(30)]


def test_different_seeds_different_draws(context):
    left, right = TopPop(counts=COUNTS, seed=11), TopPop(counts=COUNTS, seed=12)
    assert [draw(left, 2, context) for _ in range(30)] != [draw(right, 2, context) for _ in range(30)]
