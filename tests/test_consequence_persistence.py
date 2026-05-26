"""Replay-safe scheduled effects and extended state fingerprints."""

from __future__ import annotations

import copy

from src.sim.compiler import apply_transitions
from src.sim.game_loop import GameLoop, make_test_world
from src.sim.lm_adapter import MockLMAdapter
from src.sim.projection import project
from src.sim.schemas import (
    EntityKind,
    ScheduledEffect,
    ScheduledEffectKind,
    SemanticAction,
    TransitionKind,
    WorldFact,
    WorldFactScope,
)
from src.sim.state_hash import durable_state_fingerprint, world_state_fingerprint
from src.sim.world_adjudicator import (
    materialize_adjudication,
    scheduled_effect_transition,
    world_fact_transition,
)


def _player_id(world):
    return next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.PLAYER
    )


def test_scheduled_effect_transition_replay_safe():
    world = make_test_world()
    effect = ScheduledEffect(
        fire_tick=world.tick + 3,
        created_tick=world.tick,
        kind=ScheduledEffectKind.NARRATION,
        narration="A rumor spreads.",
    )
    apply_transitions(world, [scheduled_effect_transition(effect)])
    assert len(world.scheduled_effects) == 1
    assert world.scheduled_effects[0].narration == "A rumor spreads."


def test_adjudication_queues_effects_via_transition():
    from src.sim.grounding import classify
    from src.sim.schemas import ActionType, SemanticAction, ValidationResult
    from src.sim.world_adjudicator import rule_based_adjudicate

    world = make_test_world()
    player_id = _player_id(world)
    action = SemanticAction(
        verb=ActionType.SYMBOLIC,
        actor=player_id,
        raw_input="leave a note that says meet me at midnight",
    )
    adj = rule_based_adjudicate(
        world,
        action,
        action.raw_input,
        classify(action, world, action.raw_input),
        ValidationResult(valid=False),
    )
    assert adj.scheduled_effects

    proj = project(world, player_id)
    transitions = materialize_adjudication(world, action, adj, proj)
    assert any(
        t.kind == TransitionKind.SCHEDULED_EFFECT_QUEUED for t in transitions
    )

    before = len(world.scheduled_effects)
    apply_transitions(world, transitions)
    assert len(world.scheduled_effects) > before


def test_fingerprint_includes_facts_and_scheduled_queue():
    world = make_test_world()
    base = world_state_fingerprint(world)

    apply_transitions(
        world,
        [
            world_fact_transition(
                WorldFact(
                    claim="The door was forced open.",
                    scope=WorldFactScope.WORLD,
                    established_tick=world.tick,
                ),
            ),
            scheduled_effect_transition(
                ScheduledEffect(
                    fire_tick=world.tick + 2,
                    created_tick=world.tick,
                    kind=ScheduledEffectKind.NARRATION,
                    narration="Footsteps echo.",
                ),
            ),
        ],
    )
    assert durable_state_fingerprint(world) != base


def test_open_verb_affordances_surface_in_projection():
    world = make_test_world()
    player_id = _player_id(world)
    proj = project(world, player_id)
    verbs = {a.verb for a in proj.available_affordances}
    assert "bribe" in verbs or "intimidate" in verbs


def test_scheduled_queued_replays_from_event_log():
    from src.sim.schemas import Event, new_event_id

    world = make_test_world()
    initial = copy.deepcopy(world)
    effect = ScheduledEffect(
        fire_tick=world.tick + 5,
        created_tick=world.tick,
        kind=ScheduledEffectKind.NARRATION,
        narration="Someone will remember this.",
    )
    transition = scheduled_effect_transition(effect)
    world.event_log.append(
        Event(
            event_id=new_event_id(),
            tick=world.tick,
            action=SemanticAction(
                verb="symbolic",
                actor=_player_id(world),
                raw_input="[test]",
            ),
            transitions=[transition],
            witnesses=[],
        )
    )
    apply_transitions(world, [transition])

    loop = GameLoop(world, adapter=MockLMAdapter(), player_id=_player_id(world))
    reconstructed = loop.replay(initial)
    assert len(reconstructed.scheduled_effects) == 1
    assert reconstructed.scheduled_effects[0].narration == effect.narration
