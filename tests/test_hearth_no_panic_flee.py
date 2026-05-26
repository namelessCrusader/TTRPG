"""Hearth fire must not trigger mass NPC flee."""

from pathlib import Path

from src.sim.game_loop import GameLoop
from src.sim.lm_adapter import MockLMAdapter
from src.sim.npc_reflection import run_npc_reflection
from src.sim.npc_policy import ReactivePolicy
from src.sim.schemas import EmotionalState
from src.sim.world_loader import load_world_pack


def test_hearth_does_not_mark_threat_or_flee():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")
    mira = next(e for e in world.spatial.entities.values() if e.name == "Mira")
    run_npc_reflection(mira, world, MockLMAdapter())
    assert mira.meta.get("focus_topic") != "threat"
    assert mira.meta.get("near_hearth") is True
    assert mira.emotional_state != EmotionalState.FEARFUL

    policy = ReactivePolicy()
    cands = policy.generate_candidates(mira, world)
    verbs = [str(c[0].verb).lower().split(".")[-1] for c in cands]
    assert "flee" not in verbs


def test_autonomous_tick_no_mass_flee():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")
    loop = GameLoop(world, adapter=MockLMAdapter())
    flee_count = 0
    for _ in range(8):
        result = loop.autonomous_tick()
        for er in result.entity_results:
            if er.action and str(er.action.verb).lower().endswith("flee"):
                flee_count += 1
    assert flee_count < 3, f"too many flee actions: {flee_count}"
