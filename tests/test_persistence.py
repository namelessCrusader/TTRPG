"""Savegame round-trip and replay verification."""

import copy
import json
import tempfile
from pathlib import Path

from src.sim.game_loop import GameLoop
from src.sim.lm_adapter import MockLMAdapter
from src.sim.persistence import load_world, save_world, verify_save_replay
from src.sim.world_loader import load_world_pack


def test_save_load_round_trip():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")
    world.meta["pack_id"] = "tavern"
    world.tick = 5

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "save.json"
        save_world(world, path, pack_id="tavern", max_events=100)
        loaded = load_world(path)

    assert loaded.tick == 5
    assert loaded.meta.get("pack_id") == "tavern"
    assert len(loaded.event_log) == len(world.event_log)


def test_save_truncates_event_log():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "default")
    world.rng_seed = 1
    loop = GameLoop(world, adapter=MockLMAdapter())
    for _ in range(5):
        loop.autonomous_tick()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trim.json"
        n = save_world(world, path, max_events=2)
        raw = json.loads(path.read_text())
        assert n == 2
        assert raw["event_count"] == 2
        loaded = load_world(path)
        assert len(loaded.event_log) == 2


def test_verify_save_replay_after_autonomous_ticks():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")
    world.rng_seed = 99
    initial = copy.deepcopy(world)
    loop = GameLoop(world, adapter=MockLMAdapter())
    for _ in range(8):
        loop.autonomous_tick()

    ok, detail = verify_save_replay(loop.world, initial_world=initial)
    assert ok, detail
