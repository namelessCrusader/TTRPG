"""Simulation debug snapshots."""

from pathlib import Path

from src.sim.sim_debug import capture_snapshot, format_snapshot_diff
from src.sim.world_loader import load_world_pack


def test_snapshot_detects_meta_change():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "spicy")
    before = capture_snapshot(world)
    world.meta["scenario_phase"] = 2
    after = capture_snapshot(world)
    diff = format_snapshot_diff(before, after)
    assert "scenario_phase" in diff
