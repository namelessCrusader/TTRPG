"""NPCs should not spam observe after speaking."""

from pathlib import Path

from src.sim.game_loop import GameLoop
from src.sim.lm_adapter import MockLMAdapter
from src.sim.npc_policy import ReactivePolicy
from src.sim.world_loader import load_world_pack


def test_tavern_mock_mostly_social_not_observe():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")
    loop = GameLoop(world, adapter=MockLMAdapter(), npc_policy=ReactivePolicy(), player_id=None)
    for _ in range(3):
        loop.autonomous_tick()
    verbs = [str(e.action.verb).split(".")[-1] for e in world.event_log]
    observe = verbs.count("observe")
    speak = verbs.count("speak")
    assert speak >= observe
    assert speak >= 4
