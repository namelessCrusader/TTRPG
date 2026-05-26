"""Tests for deterministic rule-based post-action ripples (D5 fallback)."""

from __future__ import annotations

from src.sim.compiler import apply_transitions, compile_action
from src.sim.game_loop import GameLoop, make_test_world
from src.sim.lm_adapter import MockLMAdapter
from src.sim.rule_ripples import propose_rule_ripples
from src.sim.schemas import (
    ActionType,
    EntityId,
    EntityKind,
    SemanticAction,
    Transition,
    TransitionKind,
    WorldFactScope,
)


def _player_id(world):
    return next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.PLAYER
    )


def _npc_id(world, *, exclude=None):
    for eid, e in world.spatial.entities.items():
        if e.kind == EntityKind.NPC and e.alive and eid != exclude:
            return eid
    raise RuntimeError("no npc")


def test_rule_ripples_theft_registers_fact_and_witness_suspicion():
    world = make_test_world()
    pid = _player_id(world)
    nid = _npc_id(world, exclude=pid)

    primary = [
        Transition(
            kind=TransitionKind.ITEM_TRANSFERRED,
            payload={
                "object_id": "obj_test",
                "from_entity": str(nid),
                "to_entity": str(pid),
            },
        ),
    ]
    ripples = propose_rule_ripples(
        world,
        SemanticAction(verb="steal", actor=pid, target=str(nid)),
        primary,
    )
    assert any(t.kind == TransitionKind.WORLD_FACT_REGISTERED for t in ripples)
    assert any(
        t.kind == TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED
        for t in ripples
    )


def test_mock_adapter_npc_action_gets_rule_ripples():
    """NPC actions get deterministic ripples even without LM D5."""
    world = make_test_world()
    pid = _player_id(world)
    nid = _npc_id(world, exclude=pid)
    npc = world.spatial.entities[nid]
    npc.stats["gold"] = 50.0
    loop = GameLoop(world, adapter=MockLMAdapter(), player_id=pid)

    action = SemanticAction(
        verb="bribe",
        actor=nid,
        target=str(pid),
        raw_input="bribe them with 15 gold",
    )
    result = compile_action(action, world)
    assert result.valid

    from src.sim.projection import project
    from src.sim.schemas import Event

    event = Event(
        tick=world.tick,
        action=action,
        transitions=list(result.concrete_transitions),
    )
    apply_transitions(world, event.transitions)

    before_transitions = len(event.transitions)
    loop._apply_post_action_lm_layer(action, event, project(world, nid))

    assert len(world.world_facts) >= 1
    assert len(event.transitions) >= before_transitions


def test_hire_generic_verb_leaves_gold_and_claim_fact():
    world = make_test_world()
    pid = _player_id(world)
    nid = _npc_id(world, exclude=pid)
    player = world.spatial.entities[pid]
    player.stats["gold"] = 100.0

    action = SemanticAction(
        verb="hire",
        actor=pid,
        target=str(nid),
        raw_input="hire the guard for 25 gold",
    )
    result = compile_action(action, world)
    assert result.valid
    apply_transitions(world, result.concrete_transitions)

    assert any(
        f.scope == WorldFactScope.ENTITY and "gold" in f.claim.lower()
        for f in world.world_facts
    )
    assert any(
        t.kind == TransitionKind.CLAIM_MADE
        for t in result.concrete_transitions
    )


def test_destroy_object_emits_noise_and_destruction():
    world = make_test_world()
    pid = _player_id(world)
    obj = next(iter(world.spatial.objects.values()), None)
    if obj is None:
        return

    action = SemanticAction(
        verb="smash",
        actor=pid,
        target=str(obj.object_id),
        raw_input="smash the barrel",
    )
    result = compile_action(action, world)
    assert result.valid
    kinds = {t.kind for t in result.concrete_transitions}
    assert TransitionKind.ITEM_DESTROYED in kinds or TransitionKind.TILE_MARKED in kinds
    assert TransitionKind.NOISE_EVENT in kinds


def test_filter_duplicate_ripples_skips_existing_theft_fact():
    world = make_test_world()
    pid = _player_id(world)
    nid = _npc_id(world, exclude=pid)

    primary = [
        Transition(
            kind=TransitionKind.ITEM_TRANSFERRED,
            payload={
                "object_id": "obj_test",
                "from_entity": str(nid),
                "to_entity": str(pid),
            },
        ),
    ]
    from src.sim.fact_registry import promote_compile_to_fact_transitions
    from src.sim.compiler import apply_transitions

    fact_tr = promote_compile_to_fact_transitions(
        world,
        SemanticAction(verb="steal", actor=pid, target=str(nid)),
        primary,
    )
    apply_transitions(world, primary + fact_tr)
    existing = primary + fact_tr

    from src.sim.rule_ripples import filter_duplicate_ripples, propose_rule_ripples

    ripples = filter_duplicate_ripples(
        existing,
        propose_rule_ripples(
            world,
            SemanticAction(verb="steal", actor=pid, target=str(nid)),
            existing,
        ),
    )
    assert not any(t.kind == TransitionKind.WORLD_FACT_REGISTERED for t in ripples)
    assert any(
        t.kind == TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED for t in ripples
    )
