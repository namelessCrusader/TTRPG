"""Per-NPC observational memory (Markov window)."""

from pathlib import Path

from src.sim.npc_policy import ReactivePolicy
from src.sim.observation_memory import (
    has_fresh_stimulus,
    memory_tick_window,
    observations_in_window,
)
from src.sim.social_stimulus import broadcast_stimulus_from_event
from src.sim.schemas import (
    ActionType,
    EntityKind,
    EntityId,
    EntityState,
    Event,
    IntentBlock,
    SemanticAction,
    Transition,
    TransitionKind,
    new_entity_id,
    new_event_id,
)
from src.sim.world_loader import load_world_pack


def _tavern_with_player():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")
    player_id = new_entity_id()
    mira = next(e for e in world.spatial.entities.values() if e.name == "Mira")
    world.spatial.entities[player_id] = EntityState(
        entity_id=player_id,
        name="You",
        kind=EntityKind.PLAYER,
        position=mira.position,
        sight_range=15,
    )
    return world, player_id


def _speech_event(world, player_id, mira_id, tick: int) -> Event:
    return Event(
        event_id=new_event_id(),
        tick=tick,
        action=SemanticAction(
            verb=ActionType.SPEAK,
            actor=player_id,
            target=mira_id,
            intent=IntentBlock(manner="Hello?"),
            raw_input="say to Mira: Hello?",
        ),
        transitions=[
            Transition(
                kind=TransitionKind.DIALOGUE_SPOKEN,
                payload={
                    "actor": str(player_id),
                    "text": "Hello?",
                    "target": str(mira_id),
                },
            )
        ],
        witnesses=[],
    )


def test_memory_stores_observation_and_syncs_stimulus():
    world, player_id = _tavern_with_player()
    mira_id = next(eid for eid, e in world.spatial.entities.items() if e.name == "Mira")
    ev = _speech_event(world, player_id, mira_id, world.tick)
    broadcast_stimulus_from_event(world, ev)
    mira = world.spatial.entities[mira_id]
    assert len(observations_in_window(mira, world)) >= 1
    assert mira.meta.get("social_stimulus", {}).get("addressed") is True


def test_memory_decays_after_markov_window():
    world, player_id = _tavern_with_player()
    mira_id = next(eid for eid, e in world.spatial.entities.items() if e.name == "Mira")
    window = memory_tick_window(world)
    ev = _speech_event(world, player_id, mira_id, tick=0)
    broadcast_stimulus_from_event(world, ev)
    mira = world.spatial.entities[mira_id]
    world.tick = window + 1
    assert not has_fresh_stimulus(mira, world)
    assert observations_in_window(mira, world) == []


def test_npc_does_not_reply_to_expired_memory():
    world, player_id = _tavern_with_player()
    mira_id = next(eid for eid, e in world.spatial.entities.items() if e.name == "Mira")
    ev = _speech_event(world, player_id, mira_id, tick=0)
    broadcast_stimulus_from_event(world, ev)
    mira = world.spatial.entities[mira_id]
    world.tick = memory_tick_window(world) + 2
    policy = ReactivePolicy()
    action = policy.decide(mira, world)
    assert "stimulus" not in (action.raw_input or "")
