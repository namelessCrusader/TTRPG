"""Phase 3 — persistence checkpoints, presentation frames, LM metrics."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.sim.game_loop import GameLoop, make_test_world
from src.sim.lm_adapter import MockLMAdapter
from src.sim.lm_budget import format_metrics_summary, metrics_snapshot, wrap_adapter_metrics
from src.sim.persistence import (
    SAVEGAME_VERSION,
    list_checkpoints,
    load_latest_checkpoint,
    load_world,
    migrate_save_dict,
    save_checkpoint,
    save_world,
)
from src.sim.presentation_emitter import build_tick_frame
from src.sim.schemas import EntityKind


def _player_id(world):
    for eid, ent in world.spatial.entities.items():
        if ent.kind == EntityKind.PLAYER:
            return eid
    return None


def test_migrate_legacy_save_dict():
    raw = {
        "version": "1",
        "state": {
            "tick": 3,
            "name": "legacy",
            "regions": {},
        },
    }
    migrated = migrate_save_dict(raw)
    assert migrated["version"] == SAVEGAME_VERSION
    assert "world_facts" in migrated["state"]
    assert "player_chronicle" in migrated["state"]


def test_checkpoint_rotation():
    world = make_test_world()
    world.meta["pack_id"] = "default"
    player_id = _player_id(world)
    assert player_id is not None

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for tick in (1, 2, 3):
            world.tick = tick
            save_checkpoint(world, d, pack_id="default", max_slots=3)

        files = list_checkpoints(d)
        assert len(files) == 3
        latest = load_latest_checkpoint(d)
        assert latest.tick == 3
        assert (d / "latest.json").exists()


def test_save_includes_lm_metrics():
    world = make_test_world()
    adapter = MockLMAdapter()
    wrap_adapter_metrics(adapter, world)
    adapter.infer(
        __import__("src.sim.projection", fromlist=["project"]).project(
            world, _player_id(world)
        ),
        "wait",
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.json"
        save_world(world, path, lm_metrics=metrics_snapshot(world))
        raw = json.loads(path.read_text())
        assert raw["version"] == SAVEGAME_VERSION
        assert raw["lm_metrics"]["total_calls"] >= 1


def test_game_loop_emits_tick_frame():
    world = make_test_world()
    loop = GameLoop(world, adapter=MockLMAdapter())
    player_id = _player_id(world)
    result = loop.step("wait")
    assert result.tick_frame is not None
    assert loop.presentation is not None
    assert len(loop.presentation.stream.frames) == 1
    frame = build_tick_frame(result, world, player_id=player_id)
    assert frame.tick == result.tick


def test_metrics_summary_string():
    world = make_test_world()
    wrap_adapter_metrics(MockLMAdapter(), world)
    from src.sim.lm_budget import record_lm_call

    record_lm_call(world, "infer", latency_ms=12.5, tokens_estimated=100)
    summary = format_metrics_summary(world)
    assert "LM calls" in summary
    assert "infer=1" in summary
