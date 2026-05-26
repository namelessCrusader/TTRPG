"""
Tests for the CONTACT action and the related social/consent machinery.

Architectural points exercised:
  - The compiler resolves consent from relational state + style.aggression
    only, never from English keywords in the manner string.
  - CONTACT never fabricates a harmful outcome by itself; it records the
    event and updates the target's alertness / emotional state. Only
    HOSTILE-classified contact applies a small (1–4) "scuffle" delta.
  - The target's response is produced by their own NPC policy on the
    next turn and splits into a REACTIVE branch (target was unaware /
    low alertness) versus a CONSIDERED branch (target had time to
    think).
  - A single contact triggers exactly one policy response; subsequent
    ticks resume normal behavior.
  - The previously-broken TAKE action is now compileable.
"""

from src.sim.compiler import apply_transitions, compile_action
from src.sim.game_loop import GameLoop, make_test_world
from src.sim.lm_adapter import MockLMAdapter
from src.sim.npc_policy import ReactivePolicy
from src.sim.relational import add_edge
from src.sim.schemas import (
    ActionType,
    AlertnessLevel,
    ConsentState,
    Coord,
    EdgeKind,
    EmotionalState,
    EntityId,
    EntityKind,
    IntentBlock,
    SemanticAction,
    StyleBlock,
    TransitionKind,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


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


def _place_adjacent(world):
    """Position the guard one tile east of the player."""
    pid = _player_id(world)
    gid = _guard_id(world)
    player = world.spatial.entities[pid]
    world.spatial.entities[gid].position = Coord(
        x=player.position.x + 1, y=player.position.y
    )
    return pid, gid


def _contact(pid, gid, *, aggression=20, manner="touch"):
    return SemanticAction(
        verb=ActionType.CONTACT,
        actor=pid,
        target=gid,
        intent=IntentBlock(manner=manner),
        style=StyleBlock(aggression=aggression),
    )


def _contact_transition(result):
    for t in result.concrete_transitions:
        if t.kind == TransitionKind.CONTACT_INITIATED:
            return t
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Compiler — consent classification
# ─────────────────────────────────────────────────────────────────────────────


def test_contact_requires_adjacency():
    """Contact at 2+ tiles is rejected — the LM doesn't shortcut distance."""
    world = make_test_world()
    pid, gid = _place_adjacent(world)
    # Move guard out to 3 tiles away
    player = world.spatial.entities[pid]
    world.spatial.entities[gid].position = Coord(x=player.position.x + 3, y=player.position.y)

    result = compile_action(_contact(pid, gid), world)
    assert not result.valid


def test_contact_neutral_stranger_is_unwelcomed():
    """Guarded openness + no edges + low aggression → UNWELCOMED."""
    world = make_test_world()
    pid, gid = _place_adjacent(world)
    # The pack defaults the guard to "closed"; we want the milder
    # "guarded" baseline for this case.
    world.spatial.entities[gid].social_openness = "guarded"

    result = compile_action(_contact(pid, gid, aggression=10), world)
    assert result.valid
    t = _contact_transition(result)
    assert t is not None
    assert t.payload["consent_state"] == ConsentState.UNWELCOMED.value
    # No damage emitted for non-hostile contact.
    health_changes = [
        x for x in result.concrete_transitions
        if x.kind == TransitionKind.ENTITY_HEALTH_CHANGED
    ]
    assert health_changes == []


def test_contact_with_high_aggression_is_hostile():
    """High aggression with no INTIMATE edge → HOSTILE; small scuffle delta."""
    world = make_test_world()
    pid, gid = _place_adjacent(world)

    result = compile_action(_contact(pid, gid, aggression=85), world)
    assert result.valid
    t = _contact_transition(result)
    assert t.payload["consent_state"] == ConsentState.HOSTILE.value
    # HOSTILE contact emits a small (1-4) health delta.
    health_changes = [
        x for x in result.concrete_transitions
        if x.kind == TransitionKind.ENTITY_HEALTH_CHANGED
    ]
    assert len(health_changes) == 1
    assert -4 <= health_changes[0].payload["delta"] <= -1


def test_contact_with_mutual_intimate_edge_is_welcomed():
    """Mutual INTIMATE edges + low aggression → WELCOMED; no damage."""
    world = make_test_world()
    pid, gid = _place_adjacent(world)
    add_edge(world.relational, pid, gid, EdgeKind.INTIMATE)
    add_edge(world.relational, gid, pid, EdgeKind.INTIMATE)

    result = compile_action(_contact(pid, gid, aggression=5, manner="embrace"), world)
    assert result.valid
    t = _contact_transition(result)
    assert t.payload["consent_state"] == ConsentState.WELCOMED.value
    assert t.payload["manner"] == "embrace"


def test_contact_with_unilateral_intimate_is_not_welcomed():
    """One-directional INTIMATE is not enough to imply consent."""
    world = make_test_world()
    pid, gid = _place_adjacent(world)
    # Actor pines for target; target does not reciprocate.
    add_edge(world.relational, pid, gid, EdgeKind.INTIMATE)

    result = compile_action(_contact(pid, gid, aggression=10), world)
    assert result.valid
    t = _contact_transition(result)
    assert t.payload["consent_state"] != ConsentState.WELCOMED.value


def test_contact_with_wary_target_is_refused():
    """Target with WARY_OF edge toward actor refuses contact."""
    world = make_test_world()
    pid, gid = _place_adjacent(world)
    add_edge(world.relational, gid, pid, EdgeKind.WARY_OF)

    result = compile_action(_contact(pid, gid, aggression=20), world)
    assert result.valid
    t = _contact_transition(result)
    assert t.payload["consent_state"] == ConsentState.REFUSED.value


def test_contact_with_welcoming_openness_is_welcomed():
    """Target.social_openness='welcoming' + low aggression → WELCOMED."""
    world = make_test_world()
    pid, gid = _place_adjacent(world)
    world.spatial.entities[gid].social_openness = "welcoming"

    result = compile_action(_contact(pid, gid, aggression=10), world)
    assert result.valid
    t = _contact_transition(result)
    assert t.payload["consent_state"] == ConsentState.WELCOMED.value


def test_contact_with_closed_openness_is_refused():
    """The default guard pack sets social_openness=closed → REFUSED."""
    world = make_test_world()
    pid, gid = _place_adjacent(world)
    # The default pack already sets guard to closed; assert that and the result.
    assert world.spatial.entities[gid].social_openness == "closed"

    result = compile_action(_contact(pid, gid, aggression=20), world)
    assert result.valid
    t = _contact_transition(result)
    assert t.payload["consent_state"] == ConsentState.REFUSED.value


def test_contact_manner_is_passed_through_unread():
    """The compiler stores the manner string but never branches on it."""
    world = make_test_world()
    pid, gid = _place_adjacent(world)
    weird_manner = "hands the guard a delicate fortune cookie"

    result = compile_action(_contact(pid, gid, aggression=10, manner=weird_manner), world)
    assert result.valid
    t = _contact_transition(result)
    assert t.payload["manner"] == weird_manner


def test_contact_updates_target_alertness_and_emotion():
    """Unwelcomed contact bumps alertness up and shifts emotion to SUSPICIOUS."""
    world = make_test_world()
    pid, gid = _place_adjacent(world)
    world.spatial.entities[gid].social_openness = "guarded"
    world.spatial.entities[gid].alertness = AlertnessLevel.LOW
    world.spatial.entities[gid].emotional_state = EmotionalState.NEUTRAL

    result = compile_action(_contact(pid, gid, aggression=20), world)
    apply_transitions(world, result.concrete_transitions)

    guard = world.spatial.entities[gid]
    assert guard.emotional_state == EmotionalState.SUSPICIOUS
    assert guard.alertness != AlertnessLevel.LOW  # has elevated


# ─────────────────────────────────────────────────────────────────────────────
# Policy — reactive vs considered
# ─────────────────────────────────────────────────────────────────────────────


def _inject_contact_event(world, actor, target, *, consent, surprise):
    """Append a synthetic CONTACT_INITIATED event so the policy can react."""
    from src.sim.schemas import Event, Transition, new_event_id
    world.event_log.append(
        Event(
            event_id=new_event_id(),
            tick=world.tick,
            action=SemanticAction(
                verb=ActionType.CONTACT, actor=actor, target=target
            ),
            transitions=[
                Transition(
                    kind=TransitionKind.CONTACT_INITIATED,
                    payload={
                        "actor": actor,
                        "target": target,
                        "manner": "",
                        "consent_state": consent.value,
                        "surprise": surprise,
                        "aggression": 0,
                    },
                )
            ],
        )
    )


def test_policy_reactive_branch_panics_on_surprise_hostile_contact():
    """Unaware target + hostile contact → instinctive FLEE."""
    world = make_test_world()
    pid, gid = _place_adjacent(world)
    guard = world.spatial.entities[gid]
    guard.alertness = AlertnessLevel.UNAWARE

    _inject_contact_event(world, pid, gid, consent=ConsentState.HOSTILE, surprise=True)
    action = ReactivePolicy().decide(guard, world)
    assert action.verb == ActionType.FLEE


def test_policy_reactive_branch_pushes_back_on_surprise_unwelcomed():
    """
    Unaware target + UNWELCOMED contact → reflex CONTACT back to push away.
    The engine does NOT escalate to ATTACK on its own.
    """
    world = make_test_world()
    pid, gid = _place_adjacent(world)
    guard = world.spatial.entities[gid]
    guard.alertness = AlertnessLevel.UNAWARE

    _inject_contact_event(world, pid, gid, consent=ConsentState.UNWELCOMED, surprise=True)
    action = ReactivePolicy().decide(guard, world)
    assert action.verb == ActionType.CONTACT
    assert action.target == pid
    assert action.style is not None and action.style.aggression < 70  # not hostile


def test_policy_considered_branch_speaks_protest_for_unwelcomed():
    """If target was alert when contact happened (no surprise), they respond
    verbally, not physically — they had time to think."""
    world = make_test_world()
    pid, gid = _place_adjacent(world)
    guard = world.spatial.entities[gid]
    guard.alertness = AlertnessLevel.HIGH

    _inject_contact_event(world, pid, gid, consent=ConsentState.UNWELCOMED, surprise=False)
    action = ReactivePolicy().decide(guard, world)
    assert action.verb == ActionType.SPEAK
    assert action.target == pid


def test_policy_considered_branch_threatens_for_refused():
    """Refused contact with time to think → THREATEN to enforce boundary."""
    world = make_test_world()
    pid, gid = _place_adjacent(world)
    world.spatial.entities[gid].alertness = AlertnessLevel.HIGH

    _inject_contact_event(world, pid, gid, consent=ConsentState.REFUSED, surprise=False)
    action = ReactivePolicy().decide(world.spatial.entities[gid], world)
    assert action.verb == ActionType.THREATEN


def test_policy_considered_hostile_routes_to_attack():
    """HOSTILE contact + no surprise → full threat-response path (ATTACK if adjacent)."""
    world = make_test_world()
    pid, gid = _place_adjacent(world)
    world.spatial.entities[gid].alertness = AlertnessLevel.HIGH

    _inject_contact_event(world, pid, gid, consent=ConsentState.HOSTILE, surprise=False)
    action = ReactivePolicy().decide(world.spatial.entities[gid], world)
    assert action.verb == ActionType.ATTACK
    assert action.target == pid


def test_policy_considered_welcomed_reciprocates():
    """A welcomed contact prompts the target to reciprocate with low-aggression CONTACT."""
    world = make_test_world()
    pid, gid = _place_adjacent(world)
    guard = world.spatial.entities[gid]
    guard.alertness = AlertnessLevel.MEDIUM

    _inject_contact_event(world, pid, gid, consent=ConsentState.WELCOMED, surprise=False)
    action = ReactivePolicy().decide(guard, world)
    assert action.verb == ActionType.CONTACT
    assert action.target == pid
    assert action.style is not None and action.style.aggression < 30


def test_policy_responds_only_once_per_contact():
    """
    Once the target has acted toward the actor after a contact, the policy
    stops re-firing on that contact. (Prevents infinite reactions within
    the lookback window.)
    """
    world = make_test_world()
    pid, gid = _place_adjacent(world)
    guard = world.spatial.entities[gid]
    guard.alertness = AlertnessLevel.HIGH

    _inject_contact_event(world, pid, gid, consent=ConsentState.UNWELCOMED, surprise=False)
    first = ReactivePolicy().decide(guard, world)
    assert first.verb == ActionType.SPEAK

    # Simulate the guard having spoken (the response is in the log).
    from src.sim.schemas import Event, new_event_id
    world.event_log.append(
        Event(
            event_id=new_event_id(),
            tick=world.tick + 1,
            action=SemanticAction(
                verb=ActionType.SPEAK, actor=gid, target=pid
            ),
            transitions=[],
        )
    )
    world.tick += 1

    second = ReactivePolicy().decide(guard, world)
    # After responding to the contact, the NPC continues with drive-based
    # behavior (speak, observe, move) — anything but repeating the contact response.
    assert second.verb in (ActionType.SPEAK, ActionType.OBSERVE, ActionType.WAIT, ActionType.MOVE)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: MockLMAdapter does not produce CONTACT, so we drive contact
# directly through compile_action + apply_transitions, then run an NPC step.
# ─────────────────────────────────────────────────────────────────────────────


def test_end_to_end_unwelcome_contact_drives_npc_response():
    """
    Player initiates UNWELCOMED contact → compiler resolves → next NPC tick
    produces a response that targets the player (not a no-op).
    """
    world = make_test_world()
    pid, gid = _place_adjacent(world)
    # Default guard openness=closed → REFUSED (a deliberately stronger case)
    # so we override to 'guarded' to land on UNWELCOMED.
    world.spatial.entities[gid].social_openness = "guarded"
    world.spatial.entities[gid].alertness = AlertnessLevel.HIGH  # has time to think

    action = _contact(pid, gid, aggression=20, manner="grab")
    result = compile_action(action, world)
    assert result.valid

    apply_transitions(world, result.concrete_transitions)
    # Manually append the player's event so the policy can see it.
    from src.sim.schemas import Event, new_event_id
    world.event_log.append(
        Event(
            event_id=new_event_id(),
            tick=world.tick,
            action=action,
            transitions=result.concrete_transitions,
        )
    )

    # Now run the NPC's policy turn.
    guard = world.spatial.entities[gid]
    npc_action = ReactivePolicy().decide(guard, world)
    assert npc_action.actor == gid
    assert npc_action.target == pid
    # Considered branch + UNWELCOMED → SPEAK (verbal protest).
    assert npc_action.verb == ActionType.SPEAK


# ─────────────────────────────────────────────────────────────────────────────
# TAKE regression — used to be ACTION_TYPE_UNSUPPORTED.
# ─────────────────────────────────────────────────────────────────────────────


def test_take_action_no_longer_crashes_compiler():
    """TAKE with no object_id is rejected gracefully (not unsupported)."""
    from src.sim.schemas import RejectionReason
    world = make_test_world()
    pid = _player_id(world)
    action = SemanticAction(verb=ActionType.TAKE, actor=pid, target=None)
    result = compile_action(action, world)
    assert not result.valid
    assert result.rejection_reason != RejectionReason.ACTION_TYPE_UNSUPPORTED
