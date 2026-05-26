"""Tests for Phase 2 — delayed consequences, quest journal, fact cap."""

from __future__ import annotations

from src.sim.compiler import apply_transitions
from src.sim.delayed_consequences import schedule_delays_from_facts
from src.sim.game_loop import GameLoop, make_test_world
from src.sim.lm_adapter import MockLMAdapter
from src.sim.quest_journal import build_quest_journal
from src.sim.schemas import (
    EntityKind,
    ScheduledEffectKind,
    SemanticAction,
    Transition,
    TransitionKind,
    WorldFact,
    WorldFactScope,
)
from src.sim.world_adjudicator import register_world_fact
from src.sim.world_clock import world_tick


def _player_id(world):
    return next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.PLAYER
    )


def test_schedule_delays_from_theft_fact():
    world = make_test_world()
    pid = _player_id(world)
    register_world_fact(
        world,
        WorldFact(
            claim="Someone lost a gold pouch.",
            scope=WorldFactScope.WORLD,
            established_tick=world.tick,
            established_by=str(pid),
            tags=["theft", "investigation"],
        ),
    )
    before = len(world.scheduled_effects)
    lines, transitions = schedule_delays_from_facts(world)
    apply_transitions(world, transitions)
    assert len(world.scheduled_effects) > before
    assert any("theft" in ln for ln in lines)
    assert world.world_facts[-1].meta.get("delay_scheduled")


def test_scheduled_narration_surfaces_in_world_tick():
    world = make_test_world()
    from src.sim.world_adjudicator import schedule_effect
    from src.sim.schemas import ScheduledEffect

    schedule_effect(
        world,
        ScheduledEffect(
            fire_tick=world.tick,
            created_tick=world.tick,
            kind=ScheduledEffectKind.NARRATION,
            narration="The guard arrives, looking grim.",
        ),
    )
    events = world_tick(world)
    assert any(
        "guard arrives" in (ev.narrative_hint or "").lower()
        for ev in events
    )


def test_quest_journal_includes_emergent_fact():
    world = make_test_world()
    pid = _player_id(world)
    player = world.spatial.entities[pid]
    register_world_fact(
        world,
        WorldFact(
            claim="Blood on the floor near the bar.",
            scope=WorldFactScope.TILE,
            subject_id=f"{player.position.x},{player.position.y},{player.position.z}",
            established_tick=world.tick,
            tags=["violence", "investigation"],
        ),
    )
    journal = build_quest_journal(world, pid)
    assert any("blood" in e.detail.lower() for e in journal)


def test_quest_journal_includes_pack_goals():
    world = make_test_world()
    pid = _player_id(world)
    if not world.config.goals:
        return
    journal = build_quest_journal(world, pid)
    assert any(e.source == "pack" for e in journal)


def test_register_world_fact_respects_400_cap():
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


def test_step_result_includes_quest_journal():
    world = make_test_world()
    pid = _player_id(world)
    loop = GameLoop(world, adapter=MockLMAdapter(), player_id=pid)
    result = loop.step("look around")
    assert hasattr(result, "quest_journal")
    assert isinstance(result.quest_journal, list)


def test_scheduled_goal_transition_fires():
    world = make_test_world()
    nid = next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.NPC
    )
    npc = world.spatial.entities[nid]
    before_goals = len(npc.goals)

    from src.sim.world_adjudicator import schedule_effect
    from src.sim.schemas import ScheduledEffect, TransitionProposal

    schedule_effect(
        world,
        ScheduledEffect(
            fire_tick=world.tick,
            created_tick=world.tick,
            kind=ScheduledEffectKind.TRANSITIONS,
            transitions=[
                TransitionProposal(
                    kind=TransitionKind.ENTITY_GOAL_ADDED.value,
                    payload={
                        "entity_id": str(nid),
                        "goal": "Investigate the delayed theft.",
                        "cause": "test_delayed",
                        "priority": 0.7,
                    },
                ),
            ],
        ),
    )
    world_tick(world)
    assert len(npc.goals) > before_goals
