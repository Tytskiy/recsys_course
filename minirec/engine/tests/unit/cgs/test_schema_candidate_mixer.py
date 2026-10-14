from __future__ import annotations

import pytest
from engine.cg.mixing import CGParams, CGSchema, SchemaCandidateMixer, SchemaMixerConfig
from engine.interfaces import ID, CandidateGenerator
from tests.fakes import ListCG


def test_start_component(deps):
    config = SchemaMixerConfig.model_validate(
        {
            "default_schema": {
                "ordered_cgs": [
                    {"cg_name": "cg1", "num_candidates": 2},
                    {"cg_name": "cg2", "num_candidates": 3},
                ],
                "total_candidates": 4,
            }
        }
    )
    mixer = SchemaCandidateMixer.start_component(config, deps)

    assert isinstance(mixer, SchemaCandidateMixer)
    assert mixer.default_schema == CGSchema(
        ordered_cgs=[
            CGParams(cg_name="cg1", num_candidates=2),
            CGParams(cg_name="cg2", num_candidates=3),
        ],
        total_candidates=4,
    )


def test_override_schema(feed_context):
    mixer = SchemaCandidateMixer(
        default_schema=CGSchema(
            ordered_cgs=[CGParams(cg_name="cg1", num_candidates=3)],
            total_candidates=3,
        )
    )
    candgens: dict[str, CandidateGenerator] = {
        "cg1": ListCG("cg1", [1, 2, 3]),
        "cg2": ListCG("cg2", [10, 11, 12]),
    }
    override_schema = CGSchema(
        ordered_cgs=[CGParams(cg_name="cg2", num_candidates=2)],
        total_candidates=2,
    )

    candidates = list(
        mixer.merge(candgens=candgens, context=feed_context, schema=override_schema)
    )

    assert [candidate.item_id for candidate in candidates] == [10, 11]
    assert [candidate.factors["source_cg"] for candidate in candidates] == [
        "cg2",
        "cg2",
    ]


@pytest.mark.parametrize(
    (
        "default_schema",
        "cg_items",
        "expected_item_ids",
        "expected_sources",
    ),
    [
        pytest.param(
            {
                "ordered_cgs": [
                    {"cg_name": "cg1", "num_candidates": 3},
                    {"cg_name": "cg2", "num_candidates": 3},
                    {"cg_name": "cg3", "num_candidates": 3},
                ],
                "total_candidates": 5,
            },
            {
                "cg1": [1, 2, 3, 4],
                "cg2": [3, 4, 5, 6],
                "cg3": [1, 2, 5, 6],
            },
            [1, 2, 3, 4, 5],
            ["cg1", "cg1", "cg1", "cg2", "cg2"],
            id="deduplicates-and-tracks-source",
        ),
        pytest.param(
            {
                "ordered_cgs": [
                    {"cg_name": "cg1", "num_candidates": 2},
                    {"cg_name": "cg2", "num_candidates": 2},
                ],
                "total_candidates": 10,
            },
            {
                "cg1": [1, 2],
                "cg2": [2, 3],
            },
            [1, 2, 3],
            ["cg1", "cg1", "cg2"],
            id="total-greater-than-all-cg-limits",
        ),
        pytest.param(
            {
                "ordered_cgs": [
                    {"cg_name": "cg1", "num_candidates": 3},
                    {"cg_name": "cg2", "num_candidates": 3},
                ],
                "total_candidates": 2,
            },
            {
                "cg1": [1, 2, 3],
                "cg2": [3, 4, 5],
            },
            [1, 2],
            ["cg1", "cg1"],
            id="total-less-than-all-cg-limits",
        ),
    ],
)
def test_merge_candidates(
    feed_context,
    default_schema,
    cg_items: dict[str, list[ID]],
    expected_item_ids,
    expected_sources,
):
    mixer = SchemaCandidateMixer(default_schema=CGSchema(**default_schema))
    candgens: dict[str, CandidateGenerator] = {
        cg_name: ListCG(cg_name, ids)
        for cg_name, ids in cg_items.items()
    }

    candidates = list(mixer.merge(candgens=candgens, context=feed_context))

    assert [candidate.item_id for candidate in candidates] == expected_item_ids
    assert [candidate.factors["source_cg"] for candidate in candidates] == (
        expected_sources
    )
