"""Mixigen mixer (task `mixigen`). Behaviours:
- `build_prescore_row`: `[*context_features from user_profile (default_feature_value when
  absent), cg_index as int, float(position)]`;
- `compute_quotas` scores every `(cg, position)` slot, `position < num_candidates_per_cg`,
  in one `predict` call, takes the top `total_candidates` slots by score and returns
  `max selected position + 1` per CG; a CG without a selected slot is left out;
- `merge` asks each CG for exactly its quota (CGs without one are not asked), skips CGs
  missing from `candgens`, logs the raw per-CG lists once, then yields in `cg_names`
  order with duplicates dropped (the first CG in `cg_names` wins) and `source_cg` set;
  the experiment `schema` argument is ignored.
Non-goal: the order among equal slot scores. The prescorer is `SlotTableModel`
(`tests/fakes.py`): a slot's score is `table[cg_index][position]`. `start_component` is
provided.
"""

from __future__ import annotations

import pytest
from engine.cg.mixing import CGParams, CGSchema, MixigenCandidateMixer
from tests.fakes import ListCG, RecordingCandidateLogger, SlotTableModel, make_context

CG_NAMES = ["cg_a", "cg_b"]
# slot scores, all distinct: a0=9 > b0=8 > b2=6 > a1=5 > a2=1 > b1=0
TABLE = [[9, 5, 1], [8, 0, 6]]
A_FIRST = [[9, 8, 7], [3, 2, 1]]
B_FIRST = [[2, 1], [4, 3]]


def make_mixer(table, total_candidates, candidate_logger=None, context_features=(), default=0.0):
    return MixigenCandidateMixer(
        model_wrapper=SlotTableModel(table),
        cg_names=CG_NAMES,
        context_features=list(context_features),
        total_candidates=total_candidates,
        num_candidates_per_cg=len(table[0]),
        default_feature_value=default,
        candidate_logger=candidate_logger,
    )


def make_candgens() -> dict[str, ListCG]:
    return {"cg_a": ListCG("cg_a", [1, 2]), "cg_b": ListCG("cg_b", [3, 1])}


def test_prescore_row_layout():
    mixer = make_mixer(TABLE, 3, context_features=["user_bias", "absent"], default=-3.0)
    context = make_context()
    context.user_profile["user_bias"] = 1.5

    row = mixer.build_prescore_row("cg_b", 2, context)

    assert row == [1.5, -3.0, 1, 2.0]
    assert [type(v) for v in row] == [float, float, int, float]


def test_all_slots_scored_in_one_predict_call():
    mixer = make_mixer(TABLE, 3, context_features=["user_bias"])
    context = make_context()
    context.user_profile["user_bias"] = 1.5

    mixer.compute_quotas(context)

    (rows,) = mixer.model_wrapper.calls
    expected = [[1.5, cg, float(pos)] for cg in range(2) for pos in range(3)]
    assert sorted(rows) == sorted(expected)


@pytest.mark.parametrize(
    ("table", "total_candidates", "expected"),
    [
        pytest.param(TABLE, 1, {"cg_a": 1}, id="one_slot"),
        pytest.param(TABLE, 2, {"cg_a": 1, "cg_b": 1}, id="first_slot_of_each"),
        # b1 is not selected, yet the quota covers it (max position + 1, not a count)
        pytest.param(TABLE, 3, {"cg_a": 1, "cg_b": 3}, id="quota_is_max_position_plus_one"),
        pytest.param(TABLE, 4, {"cg_a": 2, "cg_b": 3}, id="four_slots"),
        pytest.param(TABLE, 6, {"cg_a": 3, "cg_b": 3}, id="exactly_all_slots"),
        pytest.param(TABLE, 7, {"cg_a": 3, "cg_b": 3}, id="one_past_all_slots"),
        pytest.param(A_FIRST, 3, {"cg_a": 3}, id="cg_without_selected_slot_left_out"),
    ],
)
def test_compute_quotas(table, total_candidates, expected):
    assert make_mixer(table, total_candidates).compute_quotas(make_context()) == expected


@pytest.mark.parametrize(
    ("table", "total_candidates", "requested"),
    [
        pytest.param(TABLE, 3, {"cg_a": [1], "cg_b": [3]}, id="each_cg_its_quota"),
        pytest.param(A_FIRST, 3, {"cg_a": [3], "cg_b": []}, id="cg_without_quota_not_asked"),
    ],
)
def test_merge_fetches_quotas(table, total_candidates, requested):
    candgens = {"cg_a": ListCG("cg_a", [10, 11, 12]), "cg_b": ListCG("cg_b", [20, 21, 22])}
    list(make_mixer(table, total_candidates).merge(candgens, make_context()))
    assert {name: cg.requested for name, cg in candgens.items()} == requested


def test_merge_order_and_dedup_follow_cg_names():
    """All four B_FIRST slots are selected (quotas 2 and 2), cg_b's slots scoring higher.
    The output still goes cg_a first, so the shared item 1 keeps `source_cg = cg_a`."""
    merged = list(make_mixer(B_FIRST, 4).merge(make_candgens(), make_context()))

    assert [(c.item_id, c.factors["source_cg"]) for c in merged] == [(1, "cg_a"), (2, "cg_a"), (3, "cg_b")]


def test_merge_skips_missing_cg():
    merged = list(make_mixer(B_FIRST, 4).merge({"cg_a": ListCG("cg_a", [1, 2])}, make_context()))
    assert [(c.item_id, c.factors["source_cg"]) for c in merged] == [(1, "cg_a"), (2, "cg_a")]


def test_merge_ignores_experiment_schema():
    """Mixigen computes its own quotas: a schema asking cg_b for 1 candidate changes nothing."""
    schema = CGSchema(ordered_cgs=[CGParams(cg_name="cg_b", num_candidates=1)], total_candidates=1)
    candgens = make_candgens()

    merged = list(make_mixer(B_FIRST, 4).merge(candgens, make_context(), schema))

    assert {name: cg.requested for name, cg in candgens.items()} == {"cg_a": [2], "cg_b": [2]}
    assert [c.item_id for c in merged] == [1, 2, 3]


def test_merge_logs_raw_lists_once():
    """The log keeps each CG's list as fetched, before dedup: item 1 appears under both."""
    logger = RecordingCandidateLogger()
    context = make_context()

    list(make_mixer(B_FIRST, 4, candidate_logger=logger).merge(make_candgens(), context))

    assert logger.cg_records == [(context, {"cg_a": [1, 2], "cg_b": [3, 1]})]
