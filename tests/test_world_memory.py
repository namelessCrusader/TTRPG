"""Tests for world_memory — durable facts become NPC knowledge."""

from __future__ import annotations

from src.sim.game_loop import GameLoop, make_test_world
from src.sim.lm_adapter import MockLMAdapter
from src.sim.observation_memory import observations_in_window
from src.sim.schemas import (
    EntityKind,
    TransitionKind,
    WorldFact,
    WorldFactScope,
)
from src.sim.world_adjudicator import register_world_fact
from src.sim.world_memory import process_world_memory, seed_beliefs_from_world_facts


def _npc_ids(world):
    return [
        str(eid)
        for eid, ent in world.spatial.entities.items()
        if ent.kind == EntityKind.NPC and ent.alive
    ]


def test_world_fact_seeds_beliefs_for_nearby_npcs():
    world = make_test_world()
    npc_id = _npc_ids(world)[0]
    npc = world.spatial.entities[npc_id]

    register_world_fact(
        world,
        WorldFact(
            claim="Fresh graffiti covers the tavern wall.",
            scope=WorldFactScope.TILE,
            subject_id=f"{npc.position.x},{npc.position.y},{npc.position.z}",
            established_tick=world.tick,
            established_by=str(npc_id),
            tags=["environment"],
        ),
    )

    transitions = seed_beliefs_from_world_facts(world)
    assert transitions
    assert any(t.kind == TransitionKind.BELIEF_PROPAGATED for t in transitions)

    from src.sim.compiler import apply_transitions

    apply_transitions(world, transitions)
    from src.sim.schemas import EdgeKind

    has_belief = any(
        edge.kind == EdgeKind.BELIEVES_CLAIM
        and "graffiti" in edge.meta.get("claim", "").lower()
        for edges in world.relational.edges.values()
        for edge_list in edges.values()
        for edge in edge_list
    )
    assert has_belief


def test_world_fact_recorded_in_observation_memory():
    world = make_test_world()
    npc_id = _npc_ids(world)[0]
    npc = world.spatial.entities[npc_id]

    register_world_fact(
        world,
        WorldFact(
            claim="A strange smell lingers in the corner.",
            scope=WorldFactScope.TILE,
            subject_id=f"{npc.position.x},{npc.position.y},{npc.position.z}",
            established_tick=world.tick,
            tags=["environment"],
        ),
    )

    process_world_memory(world)
    obs = observations_in_window(npc, world)
    assert any("smell" in o.get("summary", "").lower() for o in obs)


def test_player_adjudication_triggers_world_memory_propagation():
    world = make_test_world()
    player_id = next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.PLAYER
    )
    loop = GameLoop(world, adapter=MockLMAdapter(), player_id=player_id)
    before_facts = len(world.world_facts)

    result = loop.step("scribble graffiti on the tavern wall")

    assert len(world.world_facts) > before_facts
    assert result.event is not None

    from src.sim.schemas import EdgeKind

    belief_edges = [
        edge
        for holder, targets in world.relational.edges.items()
        for _tgt, edges in targets.items()
        for edge in edges
        if edge.kind == EdgeKind.BELIEVES_CLAIM
    ]
    assert belief_edges or any(
        o.get("kind") == "world_fact"
        for ent in world.spatial.entities.values()
        if ent.kind == EntityKind.NPC
        for o in observations_in_window(ent, world)
    )


def test_npc_belief_synthesis_and_gossip():
    from src.sim.observation_memory import record_observation
    from src.sim.compiler import apply_transitions, compile_action
    from src.sim.schemas import SemanticAction, ActionType, EdgeKind, IntentBlock, StyleBlock

    world = make_test_world()
    npcs = [e for e in world.spatial.entities.values() if e.kind == EntityKind.NPC and e.alive]
    assert len(npcs) >= 2
    npc_a = npcs[0]
    npc_b = npcs[1]

    # 1. Directly record a high priority observation in NPC A's short-term memory
    record_observation(
        npc_a,
        world,
        {
            "tick": world.tick,
            "event_id": "test_blood_trail_obs",
            "kind": "action",
            "summary": "Saw a mysterious blood trail near the hearth",
            "priority": 65,
            "actor_id": "unknown",
        }
    )

    # 2. Run process_world_memory to synthesize the observation into a BELIEVES_CLAIM edge
    transitions = process_world_memory(world)
    assert any(t.kind == TransitionKind.BELIEF_PROPAGATED for t in transitions)
    apply_transitions(world, transitions)

    # Verify belief is held
    has_belief = False
    for tgt_id, edges in world.relational.edges.get(str(npc_a.entity_id), {}).items():
        for edge in edges:
            if edge.kind == EdgeKind.BELIEVES_CLAIM and "blood trail" in edge.meta.get("claim", ""):
                has_belief = True
    assert has_belief

    # 3. NPC A speaks to NPC B, which should propagate the belief (gossip!)
    action = SemanticAction(
        action_id="test_gossip_action",
        actor=npc_a.entity_id,
        verb=ActionType.SPEAK,
        target=npc_b.entity_id,
        intent=IntentBlock(
            manner="Have you heard about the mysterious blood trail? It is quite scary!",
            rationale="gossip",
        ),
        style=StyleBlock(),
    )
    result = compile_action(action, world)
    assert result.valid

    # Verify the compiled action produces a BELIEF_PROPAGATED transition with NPC A's belief
    propagated_belief = None
    for t in result.concrete_transitions:
        if t.kind == TransitionKind.BELIEF_PROPAGATED:
            propagated_belief = t.payload.get("belief")
    assert propagated_belief == "Saw a mysterious blood trail near the hearth"

