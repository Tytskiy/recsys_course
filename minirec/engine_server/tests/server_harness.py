"""A fake EngineRuntime for the engine_server tests, so the app is tested without the
engine_server_reload task. The tests replace the name `EngineRuntime` in both
`engine_server.app` and `engine_server.runtime`, so build the runtime through that name."""

from __future__ import annotations

from pathlib import Path

import engine_server.app as app_module
import engine_server.runtime as runtime_module
import yaml
from engine.interfaces import Candidate, RecContext

CONFIG = {"engine_data_path": "unused", "name": "test-config"}


class FakeRuntime:
    """Records its lifecycle and requests. `recommend` returns `candidate_ids` in order,
    skipping ids in `context.item_id_filter`, as the real engine does."""

    instances: list[FakeRuntime] = []
    candidate_ids: list[int] = list(range(10))

    def __init__(self, config):
        self.config = config
        self.events: list[str] = []
        self.calls: list[tuple[int, RecContext]] = []
        # ids in context.item_id_filter at call time (the handler adds to the filter afterwards)
        self.filtered: list[set[int]] = []
        FakeRuntime.instances.append(self)

    async def start(self) -> None:
        self.events.append("start")

    async def stop(self) -> None:
        self.events.append("stop")

    def recommend(self, num_candidates: int, context: RecContext) -> list[Candidate]:
        self.calls.append((num_candidates, context))
        self.filtered.append(
            {i for i in self.candidate_ids if i in context.item_id_filter}
        )
        unseen = [i for i in self.candidate_ids if i not in self.filtered[-1]]
        return [Candidate(item_id=i) for i in unseen[:num_candidates]]


def patch_app(
    monkeypatch, tmp_path: Path, candidate_ids: list[int] | None = None
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG))
    monkeypatch.setenv(app_module.CONFIG_PATH_ENV, config_path.as_posix())
    monkeypatch.setattr(FakeRuntime, "instances", [])
    monkeypatch.setattr(FakeRuntime, "candidate_ids", candidate_ids or list(range(10)))
    monkeypatch.setattr(app_module, "EngineRuntime", FakeRuntime)
    monkeypatch.setattr(runtime_module, "EngineRuntime", FakeRuntime)
