"""Tests for deterministic scenario beat entry effects."""

from pathlib import Path

from src.sim.director import NarrativeDirector
from src.sim.game_loop import GameLoop
from src.sim.lm_adapter import MockLMAdapter
from src.sim.scenario_executor import apply_scenario_entry_effects
from src.sim.world_loader import load_world_pack


def test_scenario_beat_injects_goals_on_tick_zero():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")
    world.tick = 0
    world.meta.pop("scenario_last_beat_key", None)

    lines = apply_scenario_entry_effects(world)
    assert any("[scenario] beat entered" in ln for ln in lines)

    mira = next(e for e in world.spatial.entities.values() if e.name == "Mira")
    linna = next(e for e in world.spatial.entities.values() if e.name == "Linna")
    assert any("warm and orderly" in g for g in mira.goals)
    assert any("gossip worth a song" in g for g in linna.goals)


def test_scenario_beat_effects_fire_once_per_beat():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")
    world.tick = 10
    world.meta.pop("scenario_last_beat_key", None)

    first = apply_scenario_entry_effects(world)
    second = apply_scenario_entry_effects(world)

    assert first
    assert not second
    assert any("door_opens" in ln or "door" in ln.lower() for ln in first)


def test_autonomous_tick_surfaces_scenario_lines():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")
    world.tick = 0
    world.meta.pop("scenario_last_beat_key", None)
    loop = GameLoop(world, MockLMAdapter())

    result = loop.autonomous_tick()
    assert result.scenario_lines
    assert any("[scenario]" in ln for ln in result.scenario_lines)


def test_director_tick_start_returns_scenario_lines():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")
    world.tick = 0
    world.meta.pop("scenario_last_beat_key", None)

    director = NarrativeDirector(world)
    lines = director.tick_start(world)
    assert lines
