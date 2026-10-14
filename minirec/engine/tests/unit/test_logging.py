"""Spec for `CandidateLogger` (task candidate_logger).

Behaviours, one named test or case each:
- `JSONFormatter` writes `record.msg` as one JSON line, NumPy scalars included.
- `__init__` builds two per-instance loggers, each with one `RotatingFileHandler` sized by the
  module constants, and creates missing directories.
- `filter_factors` keeps `str`/`int`/`float` (so `bool` and `np.float64`) and drops the rest.
- `log_candidates` writes the request record: ids, ISO time, `num_candidates`, `num_served`
  (capped at the number of candidates), a top-level scalar `user_profile` (present even when
  empty), and `candidates` as `{item_id, factors}` in the ranked order.
- `log_cg_candidates` writes the per-CG record with scalar `features` and `candidate_lists`.
"""
from __future__ import annotations

import contextlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import engine.candidate_logger as cl
import numpy as np
import orjson
import pytest
from engine.candidate_logger import CandidateLogger, JSONFormatter
from engine.interfaces import Candidate

REQUEST_TIME = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _read_log_file(path: Path) -> list[dict]:
    with open(path, "rb") as f:
        return [orjson.loads(line) for line in f if line.strip()]


@pytest.mark.parametrize(
    ("msg", "expected"),
    [
        pytest.param({"a": 1, "b": "x"}, {"a": 1, "b": "x"}, id="plain_dict"),
        pytest.param({"score": np.float64(0.5)}, {"score": 0.5}, id="numpy_float64"),
        pytest.param({"count": np.int64(3)}, {"count": 3}, id="numpy_int64"),
    ],
)
def test_json_formatter_serializes_msg(msg, expected):
    record = logging.LogRecord(name="test", level=logging.INFO, pathname="", lineno=0, msg=msg, args=None, exc_info=None)
    assert orjson.loads(JSONFormatter().format(record)) == expected


@pytest.fixture
def make_logger():
    """`CandidateLogger(...)` closed at teardown, so no log file stays open."""
    loggers: list[CandidateLogger] = []

    def make(*paths: Path) -> CandidateLogger:
        loggers.append(CandidateLogger(*paths))
        return loggers[-1]

    yield make
    for logger in loggers:
        # an unfinished `__init__` builds no loggers; its test has already failed
        with contextlib.suppress(AttributeError):
            logger.close()


@pytest.fixture
def paths(tmp_path) -> tuple[Path, Path]:
    return tmp_path / "logs" / "candidates.jsonl", tmp_path / "logs" / "cg_candidates.jsonl"


def test_init_creates_two_loggers_and_the_log_directory(make_logger, paths):
    logger = make_logger(*paths)

    assert logger.logger.name == "engine.candidates"
    assert logger.cg_logger.name == "engine.cg_candidates"
    assert len(logger.logger.handlers) == len(logger.cg_logger.handlers) == 1
    assert paths[0].parent.is_dir()


def test_two_loggers_write_only_their_own_files(make_logger, tmp_path, feed_context):
    """Handlers belong to the instance: a second logger with other paths neither steals nor
    duplicates the first one's records."""
    first = make_logger(tmp_path / "a.jsonl", tmp_path / "a_cg.jsonl")
    second = make_logger(tmp_path / "b.jsonl", tmp_path / "b_cg.jsonl")

    first.log_candidates("req-a", feed_context, [], num_served=10)
    second.log_candidates("req-b", feed_context, [], num_served=10)
    first.close()
    second.close()

    assert [r["request_id"] for r in _read_log_file(tmp_path / "a.jsonl")] == ["req-a"]
    assert [r["request_id"] for r in _read_log_file(tmp_path / "b.jsonl")] == ["req-b"]


def test_log_rotates_at_the_configured_size(make_logger, paths, feed_context, monkeypatch):
    """`RotatingFileHandler` sized by the module constants: at 300 bytes with 2 backups, 20
    records (about 140 bytes each) leave `candidates.jsonl`, `.1` and `.2`, nothing older."""
    monkeypatch.setattr(cl, "ROTATING_LOGFILE_MAX_BYTES", 300)
    monkeypatch.setattr(cl, "ROTATING_LOGFILE_BACKUP_COUNT", 2)
    logger = make_logger(*paths)
    for i in range(20):
        logger.log_candidates(f"req-{i}", feed_context, [], num_served=0)
    logger.close()

    names = sorted(p.name for p in paths[0].parent.iterdir())
    assert names == ["candidates.jsonl", "candidates.jsonl.1", "candidates.jsonl.2", "cg_candidates.jsonl"]


@pytest.mark.parametrize(
    ("factors", "expected"),
    [
        pytest.param({"a": 1, "b": 2.5, "c": "s"}, {"a": 1, "b": 2.5, "c": "s"}, id="all_scalars_kept"),
        pytest.param({"vec": [1.0, 2.0], "scalar": 3}, {"scalar": 3}, id="list_dropped"),
        pytest.param({"nested": {"x": 1}, "ok": "y"}, {"ok": "y"}, id="dict_dropped"),
        pytest.param({}, {}, id="empty"),
        pytest.param({"flag": True, "other": False}, {"flag": True, "other": False}, id="bool_kept"),
        pytest.param({"missing": None, "score": 0.9}, {"score": 0.9}, id="none_dropped"),
        pytest.param({"score": np.float64(0.9)}, {"score": 0.9}, id="numpy_float64_kept"),
    ],
)
def test_filter_factors_keeps_only_scalars(factors, expected):
    assert CandidateLogger.filter_factors(factors) == expected


def test_log_candidates_writes_expected_record(make_logger, paths, feed_context):
    feed_context.current_time = REQUEST_TIME
    feed_context.user_profile["user:num_views"] = 12
    feed_context.user_profile["embedding"] = [0.1, 0.2]
    candidates = [
        Candidate(item_id=10, factors={"score": 0.9, "vec": [1.0]}),
        Candidate(item_id=20, factors={"score": 0.5}),
    ]
    logger = make_logger(*paths)
    logger.log_candidates("req-1", feed_context, candidates, num_served=1)
    logger.close()

    assert _read_log_file(paths[0]) == [{
        "request_id": "req-1",
        "user_id": 7,
        "request_time": "2024-01-01T12:00:00+00:00",
        "num_candidates": 2,
        "num_served": 1,
        "user_profile": {"user_bias": 1.0, "user:num_views": 12},
        "candidates": [
            {"item_id": 10, "factors": {"score": 0.9}},
            {"item_id": 20, "factors": {"score": 0.5}},
        ],
    }]


@pytest.mark.parametrize(
    ("num_served", "expected"),
    [
        pytest.param(2, 2, id="fewer_requested_than_ranked"),
        pytest.param(3, 3, id="requested_equals_ranked"),
        pytest.param(4, 3, id="requested_above_ranked_capped"),
    ],
)
def test_log_candidates_num_served_capped_at_candidates(make_logger, paths, feed_context, num_served: int, expected: int):
    """`num_served` says which logged rows the user could see (the top `num_served`), so it
    never exceeds the number of logged candidates."""
    logger = make_logger(*paths)
    logger.log_candidates("req-1", feed_context, [Candidate(item_id=i, factors={}) for i in (1, 2, 3)], num_served=num_served)
    logger.close()

    (rec,) = _read_log_file(paths[0])
    assert (rec["num_candidates"], rec["num_served"]) == (3, expected)


def test_log_candidates_keeps_ranked_order(make_logger, paths, feed_context):
    """The pool builder derives each candidate's position from its index in the record."""
    logger = make_logger(*paths)
    logger.log_candidates("req-1", feed_context, [Candidate(item_id=i, factors={}) for i in (30, 10, 20)], num_served=3)
    logger.close()

    (rec,) = _read_log_file(paths[0])
    assert [c["item_id"] for c in rec["candidates"]] == [30, 10, 20]


def test_log_candidates_emits_empty_user_profile(make_logger, paths, feed_context):
    feed_context.user_profile.clear()
    logger = make_logger(*paths)
    logger.log_candidates("req-empty", feed_context, [Candidate(item_id=1, factors={"score": 0.1})], num_served=1)
    logger.close()

    (rec,) = _read_log_file(paths[0])
    assert rec["user_profile"] == {}


def test_log_cg_candidates_writes_expected_record(make_logger, paths, feed_context):
    feed_context.current_time = REQUEST_TIME
    feed_context.user_profile["embedding"] = [0.1, 0.2]
    logger = make_logger(*paths)
    logger.log_cg_candidates(feed_context, {"TopPop": [1, 2], "UserToItemKNN": [3]})
    logger.close()

    assert _read_log_file(paths[1]) == [{
        "request_id": feed_context.request_id,
        "user_id": 7,
        "source_type": "ST_Feed",
        "request_time": "2024-01-01T12:00:00+00:00",
        "features": {"user_bias": 1.0},
        "candidate_lists": {"TopPop": [1, 2], "UserToItemKNN": [3]},
    }]
