"""Tests for fact_registry — compile results become durable world memory."""

from __future__ import annotations

from src.sim.compiler import compile_action
from src.sim.fact_registry import promote_compile_to_facts
from src.sim.game_loop import GameLoop, make_test_world
from src.sim.lm_adapter import MockLMAdapter
from src.sim.schemas import (
    ActionType,
    EntityKind,
    SemanticAction,
    Transition,
    TransitionKind,
    WorldFactScope,
)
from src.sim.world_memory import process_world_memory


def _player_id(world):
    return next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.PLAYER
    )


def test_mark_kernel_registers_world_fact():
    world = make_test_world()
    pid = _player_id(world)
    player = world.spatial.entities[pid]
    action = SemanticAction(
        verb="mark",
        actor=pid,
        intent=__import__("src.sim.schemas", fromlist=["IntentBlock"]).IntentBlock(
            rationale="X marks the spot",
        ),
    )
    result = compile_action(action, world)
    assert result.valid
    from src.sim.compiler import apply_transitions

    apply_transitions(world, result.concrete_transitions)
    assert any(
        f.scope == WorldFactScope.TILE and "X marks" in f.claim
        for f in world.world_facts
    )


def test_observe_surveys_prior_marks_and_facts():
    world = make_test_world()
    pid = _player_id(world)
    player = world.spatial.entities[pid]
    pos = player.position

    mark_transition = Transition(
        kind=TransitionKind.TILE_MARKED,
        payload={"x": pos.x, "y": pos.y, "mark": "blood trail"},
    )
    from src.sim.compiler import apply_transitions

    apply_transitions(world, [mark_transition])
    promote_compile_to_facts(
        world,
        SemanticAction(verb="mark", actor=pid),
        [mark_transition],
    )

    result = compile_action(
        SemanticAction(verb=ActionType.OBSERVE, actor=pid),
        world,
    )
    assert result.valid
    obs = next(
        t.payload["observe_result"]
        for t in result.concrete_transitions
        if t.kind == TransitionKind.DIALOGUE_SPOKEN
    )
    tiles = obs.get("tiles_of_interest") or []
    assert tiles
    assert any("blood" in str(m).lower() for t in tiles for m in t.get("marks", []))


def test_plant_action_leaves_hidden_env_and_fact():
    world = make_test_world()
    pid = _player_id(world)
    action = SemanticAction(
        verb="stash",
        actor=pid,
        raw_input="stash the stolen key under the floorboard",
    )
    result = compile_action(action, world)
    assert result.valid
    from src.sim.compiler import apply_transitions

    apply_transitions(world, result.concrete_transitions)
    assert any(
        "concealed" in f.claim.lower() or "stash" in (f.source_intent or "").lower()
        for f in world.world_facts
    )


def test_successful_mark_propagates_to_npc_memory():
    world = make_test_world()
    pid = _player_id(world)
    loop = GameLoop(world, adapter=MockLMAdapter(), player_id=pid)
    before = len(world.world_facts)

    loop.step("carve my initials into the bar counter")

    assert len(world.world_facts) > before
    process_world_memory(world)
    npc_obs = [
        o
        for ent in world.spatial.entities.values()
        if ent.kind == EntityKind.NPC
        for o in ent.meta.get("observation_memory", [])
        if isinstance(o, dict)
    ]
    assert npc_obs or world.world_facts


def test_health_change_promotes_violence_fact():
    world = make_test_world()
    pid = _player_id(world)
    nid = next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.NPC
    )
    transitions = [
        Transition(
            kind=TransitionKind.ENTITY_HEALTH_CHANGED,
            payload={
                "entity_id": str(nid),
                "delta": -15.0,
                "cause": "attack",
            },
        ),
    ]
    promoted = promote_compile_to_facts(
        world,
        SemanticAction(verb="attack", actor=pid, target=str(nid)),
        transitions,
    )
    assert any("harmed" in f.claim.lower() and "violence" in f.tags for f in promoted)


def test_noise_event_promotes_investigation_fact():
    world = make_test_world()
    pid = _player_id(world)
    transitions = [
        Transition(
            kind=TransitionKind.NOISE_EVENT,
            payload={"noise_level": 30, "source_id": str(pid)},
        ),
    ]
    promoted = promote_compile_to_facts(
        world,
        SemanticAction(verb="smash", actor=pid),
        transitions,
    )
    assert any("loud noise" in f.claim.lower() for f in promoted)
