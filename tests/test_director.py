from pathlib import Path

from src.sim.director import NarrativeDirector
from src.sim.world_loader import load_world_pack


def test_scenario_beats_loaded():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")
    scenario = world.config.extra.get("scenario") or {}
    assert scenario.get("id") == "tavern_night1"
    assert len(scenario.get("beats") or []) >= 1
    d = NarrativeDirector(world)
    assert len(d._beats) >= 1


def test_director_emits_concrete_dm_objective():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")
    mira = next(e for e in world.spatial.entities.values() if e.name == "Mira")
    d = NarrativeDirector(world)
    brief = d.brief_for_npc(mira)
    block = d.format_hint_block(brief)

    assert brief.objective
    assert "DM objective this turn:" in block
    assert any(
        word in brief.objective.lower()
        for word in ("greet", "warn", "drink", "secret", "choice", "drive")
    )
