"""
Tests for the NPC behavior layer.

Verifies that:
  - ReactivePolicy reads canonical state only (alertness, health, event log,
    relational graph) and never invents English keywords.
  - A guard recently attacked by the player retaliates with ATTACK if
    adjacent, MOVE if not in melee range.
  - The guard's action flows through the same Compiler and produces an
    event in the canonical event log.
  - NPC events are visible to the player on the next projection.
  - Default policy returns WAIT for an idle NPC with nothing to react to.
"""

from src.sim.game_loop import GameLoop, make_test_world
from src.sim.lm_adapter import MockLMAdapter
from src.sim.npc_policy import ReactivePolicy
from src.sim.schemas import (
    ActionType,
    AlertnessLevel,
    Coord,
    EmotionalState,
    EntityKind,
    SemanticAction,
    TransitionKind,
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
# Policy in isolation
# ─────────────────────────────────────────────────────────────────────────────


def test_idle_npc_waits():
    """Calm NPC isolated from all others → WAIT (no waypoints, no one in range)."""
    world = make_test_world()
    gid = _guard_id(world)
    guard = world.spatial.entities[gid]
    guard.alertness = AlertnessLevel.UNAWARE
    guard.emotional_state = EmotionalState.NEUTRAL
    guard.waypoints = []
    # Move every OTHER entity far out of this NPC's sight + seek range so the
    # NPC truly has nobody to interact with (tests the pure-idle fallback).
    for eid, ent in world.spatial.entities.items():
        if eid != gid:
            ent.position = Coord(x=0, y=0)
    guard.position = Coord(x=40, y=40)

    policy = ReactivePolicy()
    action = policy.decide(guard, world)
    assert action.verb == ActionType.WAIT
    assert action.actor == gid


def test_idle_npc_with_waypoints_patrols():
    """Idle NPC with waypoints → MOVE toward next waypoint."""
    from src.sim.schemas import Coord
    world = make_test_world()
    gid = _guard_id(world)
    guard = world.spatial.entities[gid]
    guard.alertness = AlertnessLevel.UNAWARE
    guard.emotional_state = EmotionalState.NEUTRAL

    policy = ReactivePolicy()
    action = policy.decide(guard, world)
    assert action.verb == ActionType.MOVE
    assert action.actor == gid


def test_alert_npc_observes_visible_entity():
    """Alertness MEDIUM with nearby entity (no waypoints) → actively engages."""
    world = make_test_world()
    gid = _guard_id(world)
    pid = _player_id(world)
    guard = world.spatial.entities[gid]
    # Stand them next to each other so visibility is guaranteed.
    # Clear waypoints so the test focuses on social reaction, not patrol.
    guard.position = Coord(x=3, y=2)
    guard.alertness = AlertnessLevel.MEDIUM
    guard.waypoints = []

    action = ReactivePolicy().decide(guard, world)
    # Alert NPCs challenge/speak or observe — both are valid active responses.
    assert action.verb in (ActionType.SPEAK, ActionType.OBSERVE)
    assert action.target == pid


def test_npc_retaliates_when_attacker_adjacent():
    """
    After the player damages the NPC, the policy should respond with
    an ATTACK targeting the player if they're adjacent.
    """
    world = make_test_world()
    gid = _guard_id(world)
    pid = _player_id(world)
    guard = world.spatial.entities[gid]
    player = world.spatial.entities[pid]
    guard.position = Coord(x=3, y=2)  # adjacent to player at (2,2)

    # Simulate that the player just hurt the guard last tick.
    from src.sim.schemas import Event, Transition, new_event_id
    world.event_log.append(
        Event(
            event_id=new_event_id(),
            tick=world.tick,
            action=SemanticAction(
                verb=ActionType.ATTACK, actor=pid, target=gid
            ),
            transitions=[
                Transition(
                    kind=TransitionKind.ENTITY_HEALTH_CHANGED,
                    payload={"entity_id": gid, "delta": -10,
                             "cause": "attack", "actor": pid},
                )
            ],
        )
    )

    action = ReactivePolicy().decide(guard, world)
    assert action.verb == ActionType.ATTACK
    assert action.target == pid


def test_wounded_npc_flees_attacker():
    """If health is critically low, prefer flee over retaliation."""
    world = make_test_world()
    gid = _guard_id(world)
    pid = _player_id(world)
    guard = world.spatial.entities[gid]
    guard.position = Coord(x=3, y=2)
    guard.health = 10  # < 25% of max_health (100)

    from src.sim.schemas import Event, Transition, new_event_id
    world.event_log.append(
        Event(
            event_id=new_event_id(),
            tick=world.tick,
            action=SemanticAction(
                verb=ActionType.ATTACK, actor=pid, target=gid
            ),
            transitions=[
                Transition(
                    kind=TransitionKind.ENTITY_HEALTH_CHANGED,
                    payload={"entity_id": gid, "delta": -90,
                             "cause": "attack", "actor": pid},
                )
            ],
        )
    )

    action = ReactivePolicy().decide(guard, world)
    assert action.verb == ActionType.FLEE


def test_npc_closes_distance_to_distant_attacker():
    """If the attacker is visible but not adjacent, NPC MOVEs toward them."""
    world = make_test_world()
    gid = _guard_id(world)
    pid = _player_id(world)
    guard = world.spatial.entities[gid]
    player = world.spatial.entities[pid]
    # Default positions: player (2,2), guard (5,5) — line of sight OK,
    # not adjacent.
    from src.sim.schemas import Event, Transition, new_event_id
    world.event_log.append(
        Event(
            event_id=new_event_id(),
            tick=world.tick,
            action=SemanticAction(
                verb=ActionType.ATTACK, actor=pid, target=gid
            ),
            transitions=[
                Transition(
                    kind=TransitionKind.ENTITY_HEALTH_CHANGED,
                    payload={"entity_id": gid, "delta": -10,
                             "cause": "attack", "actor": pid},
                )
            ],
        )
    )

    action = ReactivePolicy().decide(guard, world)
    assert action.verb == ActionType.MOVE
    assert action.target == pid  # entity-targeted move, compiler resolves


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: NPC reacts through the GameLoop
# ─────────────────────────────────────────────────────────────────────────────


def test_attacked_guard_records_npc_event_in_log():
    """
    After the player attacks an adjacent guard, the next NPC tick should
    add an NPC-authored event (an ATTACK or MOVE) to the canonical log.
    """
    from src.sim.schemas import FacingDirection
    world = make_test_world()
    gid = _guard_id(world)
    pid = _player_id(world)
    # Put them adjacent so the mock 'attack' command resolves
    world.spatial.entities[gid].position = Coord(x=3, y=2)
    # Turn player east so the guard (east of player) is within the FOV cone.
    world.spatial.entities[pid].facing = FacingDirection.EAST

    loop = GameLoop(world, adapter=MockLMAdapter())
    result = loop.step("attack")

    # Player's attack must have happened
    assert result.validation.valid
    assert result.event is not None
    assert result.event.action.actor == pid

    # And the guard reacted in the same tick
    assert len(result.npc_results) >= 1
    npc_result = result.npc_results[0]
    assert npc_result.event is not None
    assert npc_result.event.action.actor == gid
    assert npc_result.action.verb in (
        ActionType.ATTACK, ActionType.MOVE, ActionType.FLEE
    )


def test_npc_events_appear_in_player_event_log():
    """
    NPC events are appended to the canonical event log, so they replay
    correctly and become visible context for the next player projection.
    """
    from src.sim.schemas import FacingDirection
    world = make_test_world()
    gid = _guard_id(world)
    pid = _player_id(world)
    world.spatial.entities[gid].position = Coord(x=3, y=2)
    # Turn player east so the guard is within the FOV cone.
    world.spatial.entities[pid].facing = FacingDirection.EAST

    loop = GameLoop(world, adapter=MockLMAdapter())
    loop.step("attack")

    # At least one player event AND one NPC event in the log
    actors_in_log = {e.action.actor for e in world.event_log}
    assert pid in actors_in_log
    assert gid in actors_in_log


def test_npc_disabled_when_flag_off():
    """npcs_act_each_turn=False suppresses NPC reactions entirely."""
    world = make_test_world()
    gid = _guard_id(world)
    world.spatial.entities[gid].position = Coord(x=3, y=2)

    loop = GameLoop(world, adapter=MockLMAdapter(), npcs_act_each_turn=False)
    result = loop.step("attack")
    assert result.npc_results == []
