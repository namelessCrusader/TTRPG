"""
Tests for the world adjudication layer (Phase 1).
"""

from __future__ import annotations

from src.sim.game_loop import GameLoop, make_test_world
from src.sim.grounding import classify
from src.sim.lm_adapter import MockLMAdapter
from src.sim.projection import project
from src.sim.schemas import (
    ActionType,
    ActionZone,
    EntityKind,
    ScheduledEffect,
    ScheduledEffectKind,
    SemanticAction,
    TransitionKind,
    ValidationResult,
    WorldFact,
    WorldFactScope,
)
from src.sim.world_adjudicator import (
    materialize_adjudication,
    needs_adjudication,
    register_world_fact,
    resolve_with_adjudication,
    rule_based_adjudicate,
    schedule_effect,
)
from src.sim.world_clock import world_tick


def _player_id(world):
    for eid, ent in world.spatial.entities.items():
        if ent.kind == EntityKind.PLAYER:
            return eid
    return None


def test_needs_adjudication_zone3_and_rejected():
    from src.sim.schemas import GroundingResult, RejectionReason

    g_orphan = GroundingResult(zone=ActionZone.ORPHANED)
    g_ok = GroundingResult(zone=ActionZone.GROUNDED)
    ok = ValidationResult(valid=True, concrete_transitions=[])
    bad = ValidationResult(
        valid=False,
        rejection_reason=RejectionReason.PHYSICALLY_IMPOSSIBLE,
        rejection_detail="blocked",
    )

    assert needs_adjudication(g_orphan, ok)
    assert needs_adjudication(g_ok, bad)
    assert not needs_adjudication(g_ok, ok)


def test_rule_based_registers_world_fact_and_tile_mark():
    world = make_test_world()
    world.config.consequence_policy.record_traces = False
    player_id = _player_id(world)
    assert player_id is not None

    action = SemanticAction(
        verb=ActionType.SYMBOLIC,
        actor=player_id,
        raw_input="carve my initials into the bar",
    )
    grounding = classify(action, world, "carve my initials into the bar")
    result = ValidationResult(valid=False, rejection_detail="orphaned")

    adj = rule_based_adjudicate(
        world, action, "carve my initials into the bar", grounding, result,
    )
    assert adj.facts
    assert any(
        p.kind == TransitionKind.TILE_MARKED.value
        for p in adj.transition_proposals
    )

    proj = project(world, player_id)
    transitions = materialize_adjudication(world, action, adj, proj)
    assert transitions
    from src.sim.compiler import apply_transitions

    apply_transitions(world, transitions)
    assert len(world.world_facts) == 1
    assert world.world_facts[0].scope == WorldFactScope.TILE


def test_scheduled_effect_fires_on_tick():
    world = make_test_world()
    player_id = _player_id(world)
    assert player_id is not None

    schedule_effect(
        world,
        ScheduledEffect(
            fire_tick=world.tick,
            created_tick=world.tick,
            source_actor=str(player_id),
            kind=ScheduledEffectKind.NARRATION,
            narration="A guard remembers what you did.",
        ),
    )
    assert len(world.scheduled_effects) == 1

    world_tick(world)
    assert len(world.scheduled_effects) == 0
    assert any(
        ev.narrative_hint and "guard remembers" in ev.narrative_hint
        for ev in world.event_log
    )


def test_facts_surface_in_projection():
    world = make_test_world()
    player_id = _player_id(world)
    assert player_id is not None

    register_world_fact(
        world,
        WorldFact(
            claim="The bar counter has fresh carvings.",
            scope=WorldFactScope.WORLD,
            established_tick=world.tick,
        ),
    )
    proj = project(world, player_id)
    assert "carvings" in " ".join(proj.world_facts).lower()


def test_game_loop_adjudicates_zone3_action():
    world = make_test_world()
    player_id = _player_id(world)
    assert player_id is not None

    loop = GameLoop(world, adapter=MockLMAdapter(), player_id=player_id)
    result = loop.step("scribble graffiti on the tavern wall")

    assert result.event is not None
    assert len(world.world_facts) >= 1
    assert result.narration


def test_semantic_lm_enabled_for_non_mock_adapter():
    from src.sim.lm_adapter import OllamaLMAdapter

    world = make_test_world()
    GameLoop(world, adapter=OllamaLMAdapter(model="test"))
    assert world.config.semantic.use_mock is False

    world_mock = make_test_world()
    GameLoop(world_mock, adapter=MockLMAdapter())
    assert world_mock.config.semantic.use_mock is True


def test_adjudication_verb_rescue_with_target():
    world = make_test_world()
    player_id = _player_id(world)
    assert player_id is not None

    target_id = next(
        eid for eid, e in world.spatial.entities.items()
        if eid != player_id and e.alive
    )
    intent = "shake hands with " + world.spatial.entities[target_id].name.lower()
    action = SemanticAction(
        verb=ActionType.WAIT,
        actor=player_id,
        target=target_id,
        raw_input=intent,
    )
    grounding = classify(action, world, intent)
    compile_fail = ValidationResult(valid=False)

    proj = project(world, player_id)
    _, new_result, new_grounding, extra, ruling, rescued = resolve_with_adjudication(
        world,
        MockLMAdapter(),
        action,
        intent,
        proj,
        grounding,
        compile_fail,
    )

    if rescued:
        assert new_result.valid
        assert new_grounding.zone == ActionZone.GROUNDED
    else:
        assert extra or ruling
