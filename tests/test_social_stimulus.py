"""Hear-and-react: social stimulus broadcast and forced NPC replies."""

from pathlib import Path

from src.sim.game_loop import GameLoop
from src.sim.lm_adapter import MockLMAdapter
from src.sim.npc_policy import ReactivePolicy
from src.sim.schemas import (
    ActionType,
    EntityKind,
    EntityId,
    Event,
    IntentBlock,
    SemanticAction,
    Transition,
    TransitionKind,
    new_event_id,
)
from src.sim.social_stimulus import (
    broadcast_stimulus_from_event,
    has_fresh_stimulus,
)
from src.sim.world_loader import load_world_pack


def _tavern_with_player():
    from src.sim.schemas import Coord, EntityState, new_entity_id

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


def test_broadcast_marks_addressee():
    world, player_id = _tavern_with_player()
    mira_id = next(
        eid for eid, e in world.spatial.entities.items() if e.name == "Mira"
    )
    ev = Event(
        event_id=new_event_id(),
        tick=world.tick,
        action=SemanticAction(
            verb=ActionType.SPEAK,
            actor=player_id,
            target=mira_id,
            intent=IntentBlock(manner="What was that noise outside?"),
            raw_input='say to Mira: What was that noise outside?',
        ),
        transitions=[
            Transition(
                kind=TransitionKind.DIALOGUE_SPOKEN,
                payload={
                    "actor": str(player_id),
                    "text": "What was that noise outside?",
                    "target": str(mira_id),
                },
            )
        ],
        witnesses=[],
    )
    lines = broadcast_stimulus_from_event(world, ev)
    mira = world.spatial.entities[mira_id]
    assert mira.meta.get("social_stimulus", {}).get("addressed") is True
    assert any("Mira" in ln for ln in lines)


def test_mira_replies_when_addressed():
    world, player_id = _tavern_with_player()
    mira_id = next(
        eid for eid, e in world.spatial.entities.items() if e.name == "Mira"
    )
    ev = Event(
        event_id=new_event_id(),
        tick=world.tick,
        action=SemanticAction(
            verb=ActionType.SPEAK,
            actor=player_id,
            target=mira_id,
            intent=IntentBlock(manner="Stand down."),
            raw_input="say to Mira: Stand down.",
        ),
        transitions=[
            Transition(
                kind=TransitionKind.DIALOGUE_SPOKEN,
                payload={
                    "actor": str(player_id),
                    "text": "Stand down.",
                },
            )
        ],
        witnesses=[],
    )
    broadcast_stimulus_from_event(world, ev)
    mira = world.spatial.entities[mira_id]
    assert has_fresh_stimulus(mira, world)

    policy = ReactivePolicy()
    action = policy.decide(mira, world)
    assert str(action.verb).lower() in ("speak", "say")
    assert str(action.target) == str(player_id)


def test_player_step_includes_npc_reaction():
    world, player_id = _tavern_with_player()
    loop = GameLoop(
        world,
        MockLMAdapter(),
        player_id=player_id,
        npc_policy=ReactivePolicy(),
    )
    result = loop.step('say to Mira: Who was shouting?')
    assert result.stimulus_feedback
    assert result.npc_results
    spoke_to_player = any(
        str(nr.action.target) == str(player_id)
        and str(nr.action.verb).lower() in ("speak", "say")
        for nr in result.npc_results
        if nr.action
    )
    assert spoke_to_player
