"""ItemToItemLists: anchors are the first `num_items_from_history` unique items of
`user_history[action_type]` (history order; other action types ignored); candidates are
each anchor's stored list in list order, anchor by anchor, duplicates removed, at most
`num_candidates` (none for 0); an anchor without a stored list contributes nothing.
The storage is a dict-backed fake, so this test does not depend on the feature_storage task."""

from __future__ import annotations

import pytest
from engine.cg import ItemToItemLists
from engine.interfaces import ID, ActionType
from tests.fakes import DictFeatureStorage, make_context

FEATURE = "i2i"

LISTS = {
    10: [101, 102],
    20: [201],
    30: [301, 302],
    40: [101, 401],
}


def extract(history: dict[ActionType, list[ID]], num_candidates: int = 10, num_items_from_history: int = 10) -> list[ID]:
    context = make_context(history)
    cg = ItemToItemLists(
        feature_storage=DictFeatureStorage({item_id: {FEATURE: items} for item_id, items in LISTS.items()}),
        storage_feature_name=FEATURE,
        num_items_from_history=num_items_from_history,
        action_type="AT_View",
    )
    return [c.item_id for c in cg.extract_candidates(num_candidates=num_candidates, context=context)]


@pytest.mark.parametrize(
    ("history", "expected"),
    [
        pytest.param({"AT_View": [10, 20]}, [101, 102, 201], id="history_order_then_list_order"),
        pytest.param({"AT_View": [20, 10]}, [201, 101, 102], id="reversed_history"),
        pytest.param({"AT_View": [10, 10, 20]}, [101, 102, 201], id="repeated_anchor_once"),
        pytest.param({"AT_View": [10, 40]}, [101, 102, 401], id="duplicate_candidate_removed"),
        pytest.param({"AT_View": [99, 20]}, [201], id="anchor_without_list_skipped"),
        pytest.param({"AT_View": [20], "AT_Purchase": [30]}, [201], id="other_action_types_ignored"),
        pytest.param({"AT_View": [], "AT_Purchase": [30]}, [], id="empty_history_of_type_empty"),
    ],
)
def test_candidates_follow_history_then_list_order(history, expected):
    assert extract(history) == expected


@pytest.mark.parametrize(
    ("num_items_from_history", "expected"),
    [
        pytest.param(1, [101, 102], id="one_anchor"),
        pytest.param(2, [101, 102, 201], id="two_anchors_after_dedup"),
        pytest.param(3, [101, 102, 201, 301, 302], id="exactly_all_anchors"),
        pytest.param(4, [101, 102, 201, 301, 302], id="more_slots_than_anchors"),
    ],
)
def test_anchors_limited_by_num_items_from_history(num_items_from_history, expected):
    history: dict[ActionType, list[ID]] = {"AT_View": [10, 10, 20, 30]}
    assert extract(history, num_items_from_history=num_items_from_history) == expected


@pytest.mark.parametrize(
    ("num_candidates", "expected"),
    [
        pytest.param(0, [], id="zero"),
        pytest.param(1, [101], id="one"),
        pytest.param(2, [101, 102], id="cut_after_first_anchor"),
        pytest.param(3, [101, 102, 401], id="skipped_duplicate_not_counted"),
        pytest.param(4, [101, 102, 401, 301], id="cut_inside_a_list"),
        pytest.param(5, [101, 102, 401, 301, 302], id="exactly_all"),
        pytest.param(6, [101, 102, 401, 301, 302], id="one_past_all"),
    ],
)
def test_stops_at_num_candidates(num_candidates, expected):
    # 10 -> 101, 102; 40 -> 101 (already yielded), 401; 30 -> 301, 302
    assert extract({"AT_View": [10, 40, 30]}, num_candidates=num_candidates) == expected
