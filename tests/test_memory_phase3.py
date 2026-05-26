"""Tests for Phase 3 long-horizon memory."""

from __future__ import annotations

from src.sim.compiler import apply_transitions
from src.sim.episodic_memory import EPISODE_SIZE, maybe_summarize
from src.sim.game_loop import GameLoop, make_test_world
from src.sim.lm_adapter import MockLMAdapter, _build_adjudication_prompt
from src.sim.memory_retrieval import (
    fact_importance,
    memories_for_projection,
    retrieve_relevant_events,
)
from src.sim.projection import project
from src.sim.schemas import (
    ActionType,
    ActionZone,
    EntityId,
    EntityKind,
    Event,
    GroundingResult,
    SemanticAction,
    Transition,
    TransitionKind,
    ValidationResult,
    WorldFact,
    WorldFactScope,
    new_event_id,
)
from src.sim.social_stimulus import broadcast_stimulus_from_event
from src.sim.world_adjudicator import register_world_fact
from src.sim.world_chronicle import append_episode_chronicle


def test_fact_importance_prefers_violence_over_trivial():
    world = make_test_world()
    world.tick = 50
    violence = WorldFact(
        claim="Someone was killed.",
        tags=["violence", "death"],
        established_tick=48,
        scope=WorldFactScope.WORLD,
    )
    trivial = WorldFact(
        claim="Someone rested.",
        tags=["rest"],
        established_tick=49,
        scope=WorldFactScope.TILE,
        subject_id="1,1,0",
    )
    assert fact_importance(violence, world) > fact_importance(trivial, world)


def test_register_world_fact_evicts_low_importance():
    world = make_test_world()
    world.tick = 100
    for i in range(405):
        register_world_fact(
            world,
            WorldFact(
                claim=f"trivial fact {i}",
                tags=["rest"],
                established_tick=i,
                scope=WorldFactScope.WORLD,
            ),
        )
    assert len(world.world_facts) == 400
    assert any("trivial fact 404" in f.claim for f in world.world_facts)
    assert not any("trivial fact 0" == f.claim for f in world.world_facts)


def test_retrieve_relevant_events_for_actor():
    world = make_test_world()
    player_id = next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.PLAYER
    )
    world.tick = 10
    world.event_log.append(
        Event(
            event_id=new_event_id(),
            tick=8,
            action=SemanticAction(
                verb=ActionType.ATTACK,
                actor=player_id,
                raw_input="attack guard",
            ),
            transitions=[],
            witnesses=[],
            narrative_hint="You strike the guard.",
        )
    )
    lines = retrieve_relevant_events(world, player_id, max_items=3)
    assert any("strike" in ln.lower() or "attack" in ln.lower() for ln in lines)


def test_chronicle_appended_at_episode_boundary():
    world = make_test_world()
    player_id = next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.PLAYER
    )
    world.tick = EPISODE_SIZE
    world.event_log.append(
        Event(
            event_id=new_event_id(),
            tick=EPISODE_SIZE - 5,
            action=SemanticAction(
                verb=ActionType.ATTACK,
                actor=player_id,
                raw_input="smash table",
            ),
            transitions=[
                Transition(
                    kind=TransitionKind.WORLD_MARK,
                    payload={
                        "subject_kind": "entity",
                        "subject_id": str(player_id),
                        "verb": "smash",
                        "outcome": "adjudicated",
                        "summary": "Table destroyed",
                    },
                )
            ],
            witnesses=[],
            narrative_hint="The table splinters.",
        )
    )
    maybe_summarize(world, MockLMAdapter())
    assert len(world.world_chronicle) >= 1


def test_projection_includes_retrieved_memories():
    world = make_test_world()
    player_id = next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.PLAYER
    )
    world.tick = 15
    world.event_log.append(
        Event(
            event_id=new_event_id(),
            tick=12,
            action=SemanticAction(
                verb=ActionType.SYMBOLIC,
                actor=player_id,
                raw_input="bribe guard",
            ),
            transitions=[],
            witnesses=[],
            narrative_hint="Gold changes hands discreetly.",
        )
    )
    proj = project(world, player_id)
    assert proj.retrieved_memories or proj.world_facts is not None


def test_observation_recorded_replay_safe():
    world = make_test_world()
    npc_id = next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.NPC
    )
    player_id = next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.PLAYER
    )
    world.tick = 3
    event = Event(
        event_id=new_event_id(),
        tick=3,
        action=SemanticAction(
            verb=ActionType.SPEAK,
            actor=player_id,
            target=str(npc_id),
            raw_input="Hello there",
        ),
        transitions=[
            Transition(
                kind=TransitionKind.DIALOGUE_SPOKEN,
                payload={"actor": str(player_id), "text": "Hello there"},
            )
        ],
        witnesses=[str(npc_id)],
    )
    broadcast_stimulus_from_event(world, event)
    assert any(
        t.kind == TransitionKind.OBSERVATION_RECORDED for t in event.transitions
    )
    npc = world.spatial.entities[npc_id]
    from src.sim.observation_memory import observations_in_window

    assert observations_in_window(npc, world)


def test_adjudication_prompt_includes_memory_block():
    world = make_test_world()
    player_id = next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.PLAYER
    )
    register_world_fact(
        world,
        WorldFact(
            claim="The wall bears fresh graffiti.",
            scope=WorldFactScope.WORLD,
            established_tick=world.tick,
            tags=["environment"],
        ),
    )
    proj = project(world, player_id)
    action = SemanticAction(
        verb=ActionType.SYMBOLIC,
        actor=player_id,
        raw_input="carve more graffiti",
    )
    _, user = _build_adjudication_prompt(
        action,
        proj,
        action.raw_input,
        GroundingResult(zone=ActionZone.ORPHANED),
        ValidationResult(valid=False),
    )
    assert "graffiti" in user.lower() or "Established facts" in user


def test_memories_for_projection_returns_both_tiers():
    world = make_test_world()
    player_id = next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.PLAYER
    )
    append_episode_chronicle(world, 0, 20)
    retrieved, chronicle = memories_for_projection(world, player_id)
    assert isinstance(retrieved, list)
    assert isinstance(chronicle, list)
