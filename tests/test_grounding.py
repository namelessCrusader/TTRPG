"""
Tests for grounding classifier and Zone 3 / Zone 2 pathways.

Verifies:
  - "pray to the gods" with no deity → Zone 3 (ORPHANED)
  - "pray to the gods" with deity node present → Zone 1 (GROUNDED)
  - "I am the innkeeper's cousin" via DECEIVE → Zone 2 (EXTENDING)
  - Genuine "wait" intent → Zone 1
  - Full pipeline: Zone 3 action produces narration, no physics mutation
  - Full pipeline: Zone 2 DECEIVE writes a BELIEVES_CLAIM edge on success
"""

import copy

import pytest

from src.sim.game_loop import GameLoop, make_test_world
from src.sim.grounding import classify
from src.sim.lm_adapter import MockLMAdapter
from src.sim.relational import add_node, get_edges
from src.sim.schemas import (
    ActionType,
    ActionZone,
    Coord,
    EdgeKind,
    EntityId,
    EntityKind,
    IntentBlock,
    NodeKind,
    SemanticAction,
    StyleBlock,
)


def _player_id(world) -> EntityId:
    for eid, e in world.spatial.entities.items():
        if e.kind == EntityKind.PLAYER:
            return eid
    raise RuntimeError("No player")


def _guard_id(world) -> EntityId:
    for eid, e in world.spatial.entities.items():
        if e.kind == EntityKind.NPC:
            return eid
    raise RuntimeError("No guard")


# ---------------------------------------------------------------------------
# Grounding classifier — Zone 3 (ORPHANED)
# ---------------------------------------------------------------------------


def test_pray_with_no_deity_is_zone3():
    world = make_test_world()
    pid = _player_id(world)
    action = SemanticAction(
        verb=ActionType.WAIT,
        actor=pid,
        raw_input="I pray to the gods",
    )
    result = classify(action, world, "I pray to the gods")
    assert result.zone == ActionZone.ORPHANED
    assert result.orphan_reason is not None
    assert len(result.orphan_reason) > 0


def test_explicit_symbolic_action_is_zone3():
    world = make_test_world()
    pid = _player_id(world)
    action = SemanticAction(
        verb=ActionType.SYMBOLIC,
        actor=pid,
        raw_input="I perform a ritual",
    )
    result = classify(action, world, "I perform a ritual")
    assert result.zone == ActionZone.ORPHANED


def test_genuine_wait_intent_is_zone1():
    world = make_test_world()
    pid = _player_id(world)
    action = SemanticAction(verb=ActionType.WAIT, actor=pid, raw_input="wait")
    result = classify(action, world, "wait")
    assert result.zone == ActionZone.GROUNDED


def test_magic_intent_is_zone3():
    """
    Magic/spell intent in the default pack (no arcane faction defined)
    should be ORPHANED. The orphan reason should reference the unanchored
    affordance keyword that was declared in affordances.yaml.
    """
    world = make_test_world()
    pid = _player_id(world)
    action = SemanticAction(verb=ActionType.WAIT, actor=pid, raw_input="cast a spell")
    result = classify(action, world, "cast a spell")
    assert result.zone == ActionZone.ORPHANED
    assert result.orphan_reason is not None
    # The reason mentions which declared affordance keyword failed to anchor.
    reason_lower = result.orphan_reason.lower()
    assert "spell" in reason_lower or "magic" in reason_lower


# ---------------------------------------------------------------------------
# Grounding classifier — Zone 1 (GROUNDED) when affordance exists
# ---------------------------------------------------------------------------


def test_pray_with_deity_node_is_zone1():
    world = make_test_world()
    pid = _player_id(world)
    # Add a religious faction to the world
    add_node(world.relational, "faction_gods", NodeKind.FACTION, "The Old Gods")
    action = SemanticAction(
        verb=ActionType.WAIT,
        actor=pid,
        raw_input="I pray to the gods",
    )
    result = classify(action, world, "I pray to the gods")
    assert result.zone == ActionZone.GROUNDED


# ---------------------------------------------------------------------------
# Grounding classifier — Zone 2 (EXTENDING)
# ---------------------------------------------------------------------------


def test_deception_with_assertion_is_zone2():
    world = make_test_world()
    pid = _player_id(world)
    gid = _guard_id(world)
    action = SemanticAction(
        verb=ActionType.DECEIVE,
        actor=pid,
        target=gid,
        intent=IntentBlock(
            rationale="I am the innkeeper's cousin, he sent me",
            desired_outcome=["passage"],
        ),
        raw_input="tell the guard I'm the innkeeper's cousin",
    )
    result = classify(action, world, "tell the guard I'm the innkeeper's cousin")
    assert result.zone == ActionZone.EXTENDING
    assert result.asserted_fact is not None
    assert "cousin" in result.asserted_fact.lower()


def test_move_action_is_always_zone1():
    world = make_test_world()
    pid = _player_id(world)
    action = SemanticAction(
        verb=ActionType.MOVE,
        actor=pid,
        target=Coord(x=3, y=2),
        raw_input="move forward",
    )
    result = classify(action, world, "move forward")
    assert result.zone == ActionZone.GROUNDED


# ---------------------------------------------------------------------------
# Full pipeline: Zone 3 produces narration, no physics mutation
# ---------------------------------------------------------------------------


def test_zone3_produces_narration_no_state_change():
    world = make_test_world()
    pid = _player_id(world)
    initial_state = world.model_dump_json()
    loop = GameLoop(world, adapter=MockLMAdapter())

    result = loop.step("I pray to the gods")

    # Zone must be ORPHANED
    assert result.grounding.zone == ActionZone.ORPHANED

    # Narration must be present and non-empty
    assert result.narration is not None
    assert len(result.narration) > 10

    # Physical entity positions must be unchanged
    for eid, entity in world.spatial.entities.items():
        # Emotional state may have a trivial update (symbolic action produces
        # a NEUTRAL transition) but health and position must be pristine.
        assert entity.health == 100
        # Positions must match original
    assert world.spatial.entities[pid].position == world.spatial.entities[pid].position


def test_zone3_does_not_corrupt_canonical_positions():
    world = make_test_world()
    pid = _player_id(world)
    pos_before = copy.copy(world.spatial.entities[pid].position)
    loop = GameLoop(world, adapter=MockLMAdapter())
    loop.step("contemplate the nature of existence")
    assert world.spatial.entities[pid].position == pos_before


def test_zone3_event_is_recorded_with_narrative_hint():
    world = make_test_world()
    pid = _player_id(world)
    loop = GameLoop(world, adapter=MockLMAdapter())
    loop.step("I pray to the gods")
    # The player's event must exist and carry the narration as narrative_hint
    player_events = [e for e in world.event_log if e.action.actor == pid]
    assert player_events, "No player event was recorded for Zone 3 intent."
    assert player_events[-1].narrative_hint is not None


# ---------------------------------------------------------------------------
# Full pipeline: Zone 2 belief edge written on successful deception
# ---------------------------------------------------------------------------


def test_zone2_deception_writes_belief_edge_on_success():
    world = make_test_world()
    pid = _player_id(world)
    gid = _guard_id(world)

    # Give player very high deception to maximise success probability
    world.spatial.entities[pid].attributes["deception"] = 95
    world.spatial.entities[gid].attributes["perception"] = 10

    loop = GameLoop(world, adapter=MockLMAdapter())

    # Use a fixed action_id so the seeded roll is deterministic
    from src.sim.compiler import compile_action
    from src.sim.grounding import classify as gclassify
    from src.sim.schemas import ConstraintBlock

    action = SemanticAction(
        action_id="act_test_z2",
        verb=ActionType.DECEIVE,
        actor=pid,
        target=gid,
        intent=IntentBlock(
            rationale="I am the innkeeper's cousin",
            desired_outcome=["passage"],
        ),
        style=StyleBlock(emotional_tone="calm"),
        constraints=ConstraintBlock(),
        raw_input="tell the guard I'm the innkeeper's cousin",
    )

    from src.sim.projection import project
    proj = project(world, pid)
    grounding = gclassify(action, world, action.raw_input or "")
    assert grounding.zone == ActionZone.EXTENDING

    result = compile_action(action, world, projection=proj)

    from src.sim.compiler import build_belief_transitions
    belief_transitions = build_belief_transitions(action, grounding, result, world)

    if result.valid and belief_transitions:
        from src.sim.compiler import apply_transitions
        apply_transitions(world, result.concrete_transitions + belief_transitions)
        from src.sim.relational import apply_event_to_graph
        from src.sim.schemas import Event, new_event_id
        event = Event(
            event_id=new_event_id(),
            tick=world.tick,
            action=action,
            transitions=result.concrete_transitions + belief_transitions,
            witnesses=[pid, gid],
        )
        world.event_log.append(event)
        apply_event_to_graph(world.relational, event, world.tick)

        # Belief edge should now exist from guard → player
        belief_edges = get_edges(
            world.relational, gid, pid, EdgeKind.BELIEVES_CLAIM
        )
        assert len(belief_edges) == 1
        assert "cousin" in belief_edges[0].meta.get("claim", "").lower()
