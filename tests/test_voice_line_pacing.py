"""Voice-line rotation and speak pacing via npc_mind."""

from pathlib import Path

from src.sim.npc_mind import interpret_mind, load_mind, score_action
from src.sim.npc_policy import ReactivePolicy
from src.sim.world_loader import load_world_pack


def test_min_speak_gap_suppresses_speak_scoring():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")
    world.meta["npc_min_speak_gap"] = 3
    ent = next(e for e in world.spatial.entities.values() if "Mira" in e.name)
    policy = ReactivePolicy()
    cands = policy.generate_candidates(ent, world, max_candidates=28)
    speak = next((a, w) for a, _, w in cands if str(a.verb).lower() == "speak")

    ent.meta["last_speak_tick"] = world.tick - 1
    interpret_mind(ent, world)
    assert not load_mind(ent).speak_ready
    blocked = score_action(ent, world, speak[0], speak[1])

    ent.meta["last_speak_tick"] = world.tick - 10
    interpret_mind(ent, world)
    assert load_mind(ent).speak_ready
    ready = score_action(ent, world, speak[0], speak[1])
    assert ready > blocked
