"""Replay hash must match live run for deterministic mock adapter."""

import copy
from pathlib import Path

from src.sim.game_loop import GameLoop
from src.sim.lm_adapter import MockLMAdapter
from src.sim.state_hash import world_state_fingerprint
from src.sim.world_loader import load_world_pack


def test_replay_matches_live_fingerprint():
    pack = Path(__file__).resolve().parents[1] / "worlds" / "tavern"
    world = load_world_pack(pack)
    world.rng_seed = 42
    initial = copy.deepcopy(world)
    loop = GameLoop(world, adapter=MockLMAdapter())
    for _ in range(10):
        loop.autonomous_tick()
    live_hash = world_state_fingerprint(world)
    reconstructed = loop.replay(initial)
    assert world_state_fingerprint(reconstructed) == live_hash
