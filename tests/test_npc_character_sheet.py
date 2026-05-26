"""
Tests for `build_npc_character_sheet`.

The character sheet is the NPC's self-view fed to the LM-driven policy.
The kiss-blindness regression test guards the architectural fix: an NPC
must perceive open-verb interactions targeting them even when the
freeform compiler records them as DIALOGUE_SPOKEN rather than
CONTACT_INITIATED. Before this work, the guard could be `kiss`ed (or
`jerk`ed off, etc.) and the policy would not see anything had
happened.
"""

from __future__ import annotations

from src.sim.compiler import apply_transitions, compile_action
from src.sim.game_loop import make_test_world
from src.sim.npc_policy import build_npc_character_sheet
from src.sim.relational import add_edge, add_node, apply_event_to_graph
from src.sim.schemas import (
    ActionType,
    AlertnessLevel,
    EdgeKind,
    EmotionalState,
    EntityId,
    EntityKind,
    Event,
    IntentBlock,
    NodeKind,
    SemanticAction,
    new_event_id,
)


def _player_id(world):
    for eid, e in world.spatial.entities.items():
        if e.kind == EntityKind.PLAYER:
            return eid
    return None


def _guard_id(world):
    for eid, e in world.spatial.entities.items():
        if e.kind == EntityKind.NPC:
            return eid
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Plain fields are carried verbatim from canonical state
# ─────────────────────────────────────────────────────────────────────────────


def test_character_sheet_carries_role_personality_drive():
    world = make_test_world()
    gid = _guard_id(world)
    guard = world.spatial.entities[gid]
    guard.role = "city guard"
    guard.personality = "stoic, professional"
    guard.drive = "watch the gate"
    guard.emotional_state = EmotionalState.SUSPICIOUS
    guard.alertness = AlertnessLevel.HIGH

    sheet = build_npc_character_sheet(guard, world)

    assert sheet.npc_id == gid
    assert sheet.role == "city guard"
    assert sheet.personality == "stoic, professional"
    assert sheet.drive == "watch the gate"
    assert sheet.emotional_state == EmotionalState.SUSPICIOUS
    assert sheet.alertness == AlertnessLevel.HIGH
    assert sheet.health_fraction == 1.0


def test_character_sheet_summarizes_outbound_relations():
    world = make_test_world()
    gid = _guard_id(world)
    # Make sure both ends of every edge exist as nodes.
    add_node(world.relational, "captain", NodeKind.ENTITY, "Captain Vex")
    add_edge(
        world.relational, gid, "captain", EdgeKind.RESPECTS,
        weight=0.6, tick=0,
    )

    sheet = build_npc_character_sheet(world.spatial.entities[gid], world)
    joined = "\n".join(sheet.relations)
    assert "respects Captain Vex" in joined
    assert "0.60" in joined


# ─────────────────────────────────────────────────────────────────────────────
# Kiss-blindness regression — the original bug
# ─────────────────────────────────────────────────────────────────────────────


def _commit_action(world, action: SemanticAction) -> Event:
    """Helper: compile an action, append the resulting event to the log,
    apply transitions. Mirrors what GameLoop.step does internally so
    each test can put exactly the events it wants into the log."""
    result = compile_action(action, world)
    assert result.valid, f"Setup action failed to compile: {result}"
    apply_transitions(world, result.concrete_transitions)
    witnesses = [
        e.entity_id for e in world.spatial.entities.values()
        # The actor and the target are always witnesses; the test world
        # is small enough that everyone has line-of-sight to everyone.
    ]
    event = Event(
        event_id=new_event_id(),
        tick=world.tick,
        action=action,
        transitions=result.concrete_transitions,
        witnesses=witnesses,
    )
    world.event_log.append(event)
    apply_event_to_graph(world.relational, event, world.tick)
    return event


def test_npc_perceives_freeform_verb_targeting_them():
    """Regression: an open-verb action (e.g. 'kiss') that goes through
    the freeform compiler (emits DIALOGUE_SPOKEN, NOT CONTACT_INITIATED)
    must still appear in the NPC's character sheet under recent
    perceptions. Before the fix, `_most_recent_contact_against` was the
    only perception path and it filtered exclusively on
    CONTACT_INITIATED — so the guard had no way to know they'd been
    kissed."""
    world = make_test_world()
    pid = _player_id(world)
    gid = _guard_id(world)

    # Move player adjacent to guard so any action is plausible.
    player = world.spatial.entities[pid]
    guard = world.spatial.entities[gid]
    player.position = guard.position.neighbors()[0]

    # The freeform path: verb 'kiss' is not a kernel verb, not in
    # built-in defaults, and (by default in this pack) not in templates
    # either, so it flows through _compile_freeform → DIALOGUE_SPOKEN.
    kiss = SemanticAction(
        verb="kiss",
        actor=pid,
        target=gid,
        intent=IntentBlock(manner="lightly"),
    )
    _commit_action(world, kiss)
    world.tick += 1

    sheet = build_npc_character_sheet(guard, world)
    joined = "\n".join(sheet.recent_perceptions)
    assert "kiss" in joined, (
        f"Expected the guard to perceive the kiss; got perceptions: "
        f"{sheet.recent_perceptions!r}"
    )
    assert "you" in joined.lower(), (
        "Perception line should be framed as something happening to 'you' "
        "(the focal NPC), not just as a third-party event."
    )


def test_npc_perceives_their_own_past_actions():
    """The sheet also surfaces the NPC's own recent actions so the LM
    can stay consistent ('I just spoke to the player; I shouldn't speak
    again immediately')."""
    world = make_test_world()
    gid = _guard_id(world)
    pid = _player_id(world)

    speak = SemanticAction(
        verb=ActionType.SPEAK,
        actor=gid,
        target=pid,
        intent=IntentBlock(rationale="State your business."),
    )
    _commit_action(world, speak)
    world.tick += 1

    sheet = build_npc_character_sheet(world.spatial.entities[gid], world)
    joined = "\n".join(sheet.recent_perceptions)
    assert "you" in joined.lower() and "speak" in joined


def test_npc_with_no_events_has_empty_perceptions():
    """A clean world: the sheet's perceptions list is empty, not
    populated with stale or hallucinated entries."""
    world = make_test_world()
    gid = _guard_id(world)
    sheet = build_npc_character_sheet(world.spatial.entities[gid], world)
    assert sheet.recent_perceptions == []
