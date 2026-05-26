"""Latent pressure evaluation."""

from pathlib import Path

from src.sim.compiler import apply_transitions
from src.sim.pressure_eval import evaluate_pressures
from src.sim.world_loader import load_world_pack


def test_spicy_pressure_fires_early_beat():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "spicy")
    world.tick = 5
    lines, transitions = evaluate_pressures(world)
    apply_transitions(world, transitions)
    assert any("early_warmth" in ln for ln in lines)
    assert world.meta.get("active_pressures")


def test_pressure_ambient_multiplier():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "spicy")
    world.tick = 20
    # Simulate memory of Madame
    elara = next(e for e in world.spatial.entities.values() if e.name == "Elara")
    elara.meta["observation_memory"] = [
        {"summary": "You mentioned Madame Voss", "tick": 19, "priority": 20}
    ]
    _, transitions = evaluate_pressures(world)
    apply_transitions(world, transitions)
    mults = world.meta.get("ambient_probability_mult") or {}
    assert mults.get("footsteps_corridor", 1.0) >= 2.0


def test_goal_text_not_dialogue():
    from src.sim.speech_utils import is_real_dialogue, matches_entity_goal_text

    goals = ["make tonight feel genuine rather than routine"]
    assert matches_entity_goal_text(
        "make tonight feel genuine rather than routine", goals
    )
    assert not is_real_dialogue(
        "make tonight feel genuine rather than routine", goals=goals
    )
    assert is_real_dialogue(
        "Come closer — the night is young.", goals=goals
    )
