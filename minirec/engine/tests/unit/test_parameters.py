from __future__ import annotations

import httpx
import pytest
from engine.parameters import DEFAULT_EXPERIMENT_PARAMS, ExperimentManager

SCHEMA = {"ordered_cgs": [{"cg_name": "TopPop", "num_candidates": 5}], "total_candidates": 5}


def _manager(handler) -> ExperimentManager:
    manager = ExperimentManager(cg_names={"TopPop"}, enabled=False)
    manager.client = httpx.Client(base_url="http://experiments", transport=httpx.MockTransport(handler))
    return manager


def _respond(params: dict):
    return lambda request: httpx.Response(200, json={"group_name": "test", "params": params})


def _timeout(request):
    raise httpx.ReadTimeout("timed out", request=request)


@pytest.mark.parametrize(
    "handler",
    [
        pytest.param(lambda request: httpx.Response(404), id="no_experiment_404"),
        pytest.param(_timeout, id="timeout"),
        pytest.param(
            _respond({"filter_history": True, "cg_schema": {**SCHEMA, "ordered_cgs": [{"cg_name": "Missing", "num_candidates": 5}]}}),
            id="unknown_cg_in_schema",
        ),
    ],
)
def test_resolve_falls_back_to_defaults(handler):
    assert _manager(handler).resolve(7, "ST_Feed") == DEFAULT_EXPERIMENT_PARAMS


@pytest.mark.parametrize(
    ("params", "expected_schema_cgs"),
    [
        pytest.param({"filter_history": True, "cg_schema": SCHEMA}, ["TopPop"], id="schema_given"),
        pytest.param({"filter_history": True}, None, id="schema_omitted"),
    ],
)
def test_resolve_parses_experiment_params(params, expected_schema_cgs):
    resolved = _manager(_respond(params)).resolve(7, "ST_Feed")

    assert resolved.filter_history is True
    schema_cgs = resolved.cg_schema.cg_names if resolved.cg_schema else None
    assert schema_cgs == expected_schema_cgs
