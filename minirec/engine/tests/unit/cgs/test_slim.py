"""SLIM CG (task `slim`). Pinned: `start_component` reads the mappings JSON and the weight matrix through
`deps.engine_data`; `extract_candidates` scores items by summing the weight rows of the distinct known
items in `user_history[action_type]`, yields the top `num_candidates` by score (at most the catalogue, none for 0),
and does not exclude history items or zero scores. Non-goal: the order among equal scores.
"""

from __future__ import annotations

import numpy as np
import orjson
import pytest
import scipy.sparse as sps
from engine.cg import SLIMToHistory
from engine.cg.slim import SLIMToHistoryConfig
from engine.interfaces import RecContext


def test_start_component(rec_engine_files_dir, deps):
    mappings = {"item_to_index": {10: 0, 20: 1}, "index_to_item": {0: 10, 1: 20}}
    rec_engine_files_dir.joinpath("slim_mapping.json").write_bytes(
        orjson.dumps(mappings, option=orjson.OPT_NON_STR_KEYS)
    )
    weights = np.array([[0.0, 0.2], [0.1, 0.0]], dtype=np.float32)
    sps.save_npz(rec_engine_files_dir / "slim_weights.npz", sps.csr_array(weights))

    config = SLIMToHistoryConfig(
        action_type="AT_View", mappings_path="slim_mapping.json", weights_path="slim_weights.npz"
    )
    candgen = SLIMToHistory.start_component(config, deps)

    assert isinstance(candgen, SLIMToHistory)
    assert candgen.name == "SLIM_AT_View_history"
    assert candgen.num_items == 2
    assert candgen.mappings == mappings
    np.testing.assert_array_equal(sps.csr_array(candgen.item_weights).toarray(), weights)


WEIGHTS = np.array(
    [
        [0.0, 0.1, 0.2, 0.3],
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.2, 0.0, 0.1],
        [0.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)


@pytest.fixture
def candgen() -> SLIMToHistory:
    return SLIMToHistory(
        mappings={
            "item_to_index": {10: 0, 20: 1, 30: 2, 40: 3},
            "index_to_item": {0: 10, 1: 20, 2: 30, 3: 40},
        },
        item_weights=sps.csr_array(WEIGHTS),
        action_type="AT_View",
    )


@pytest.mark.parametrize(
    ("history", "num_candidates", "expected"),
    [
        pytest.param([10, 30], 3, [10, 40, 20], id="rows_summed_history_not_excluded"),
        pytest.param([10, 30, 10], 3, [10, 40, 20], id="repeated_item_counted_once"),
        pytest.param([10, 999, 30], 3, [10, 40, 20], id="unknown_item_ignored"),
        pytest.param([30], 3, [10, 20, 40], id="single_item_other_action_type_ignored"),
        pytest.param([10, 30], 0, [], id="zero_candidates"),
        pytest.param([10, 30], 1, [10], id="one_candidate"),
        pytest.param([10, 30], 4, [10, 40, 20, 30], id="num_candidates_equals_catalogue"),
        pytest.param([10, 30], 5, [10, 40, 20, 30], id="num_candidates_past_catalogue"),
        # row 30 = [0.5, 0.2, 0.0, 0.1]: item 30 scores 0 and still fills the list
        pytest.param([30], 4, [10, 20, 40, 30], id="zero_score_item_still_returned"),
    ],
)
def test_extract_candidates(candgen, feed_context: RecContext, history, num_candidates, expected):
    feed_context.user_history = {"AT_View": history, "AT_Click": [10]}
    candidates = list(candgen.extract_candidates(num_candidates=num_candidates, context=feed_context))
    assert [candidate.item_id for candidate in candidates] == expected
