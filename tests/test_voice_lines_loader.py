"""Entity voice_lines from world packs."""

from pathlib import Path

from src.sim.world_loader import load_world_pack


def test_tavern_loads_voice_lines():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")
    mira = next(e for e in world.spatial.entities.values() if e.name == "Mira")
    assert len(mira.voice_lines) >= 2
