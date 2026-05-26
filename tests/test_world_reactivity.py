"""Tests for durable world changes driving NPC and projection reactivity."""

from __future__ import annotations

from src.sim.game_loop import make_test_world
from src.sim.memory_retrieval import ranked_facts_for_projection
from src.sim.npc_policy import ReactivePolicy
from src.sim.observation_memory import get_actionable_stimulus, observations_in_window
from src.sim.projection import project
from src.sim.schemas import (
    EntityId,
    EntityKind,
    Transition,
    TransitionKind,
    WorldFact,
    WorldFactScope,
)
from src.sim.world_adjudicator import register_world_fact
from src.sim.world_memory import process_world_memory
from src.sim.world_propagation import causal_consequence_pass


def _npc(world):
    return next(
        ent
        for ent in world.spatial.entities.values()
        if ent.kind == EntityKind.NPC and ent.alive
    )


def test_nearby_tile_fact_appears_in_projection():
    world = make_test_world()
    npc = _npc(world)
    player_pos = next(
        e.position
        for e in world.spatial.entities.values()
        if e.kind == EntityKind.PLAYER
    )
    # Fact on an adjacent tile — should be perceptually relevant.
    adj = player_pos.neighbors()[0]
    register_world_fact(
        world,
        WorldFact(
            claim="Fresh graffiti covers the wall.",
            scope=WorldFactScope.TILE,
            subject_id=f"{adj.x},{adj.y},{adj.z}",
            established_tick=world.tick,
            tags=["environment"],
        ),
    )
    player_id = next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.PLAYER
    )
    facts = ranked_facts_for_projection(world, player_id)
    assert any("graffiti" in f.lower() for f in facts)

    proj = project(world, player_id)
    notes = " ".join(proj.environment.physical_notes).lower()
    assert "graffiti" in notes or any(
        "graffiti" in f.lower() for f in proj.world_facts
    )


def test_npc_generates_investigate_candidates_for_world_fact():
    world = make_test_world()
    npc = _npc(world)
    npc_id = str(npc.entity_id)

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

    policy = ReactivePolicy()
    labels = [
        label.lower()
        for _, label, _ in policy.generate_candidates(npc, world)
    ]
    assert any("investigate" in lbl or "examine" in lbl for lbl in labels)


def test_tile_mark_ripple_records_witness_observations():
    world = make_test_world()
    npc = _npc(world)
    mark = Transition(
        kind=TransitionKind.TILE_MARKED,
        payload={
            "x": npc.position.x,
            "y": npc.position.y,
            "z": npc.position.z,
            "mark": "scratches on the floor",
        },
    )
    ripples = causal_consequence_pass(world, [mark])
    obs_transitions = [
        t for t in ripples if t.kind == TransitionKind.OBSERVATION_RECORDED
    ]
    assert obs_transitions
    from src.sim.compiler import apply_transitions

    apply_transitions(world, obs_transitions)
    other_npc = next(
        ent
        for ent in world.spatial.entities.values()
        if ent.kind == EntityKind.NPC and ent.entity_id != npc.entity_id
    )
    obs = observations_in_window(other_npc, world)
    assert any("mark" in o.get("summary", "").lower() for o in obs)


def test_world_fact_becomes_actionable_stimulus():
    world = make_test_world()
    npc = _npc(world)

    register_world_fact(
        world,
        WorldFact(
            claim="Blood stains the floorboards.",
            scope=WorldFactScope.TILE,
            subject_id=f"{npc.position.x},{npc.position.y},{npc.position.z}",
            established_tick=world.tick,
            tags=["environment", "violence"],
        ),
    )
    process_world_memory(world)
    stim = get_actionable_stimulus(npc, world)
    assert stim is not None
    assert stim.get("kind") == "world_fact"
    assert "blood" in stim.get("summary", "").lower()


def test_nearby_entity_fact_surfaces_to_witness_projection():
    world = make_test_world()
    npc = _npc(world)
    other_npc = next(
        ent
        for ent in world.spatial.entities.values()
        if ent.kind == EntityKind.NPC and ent.entity_id != npc.entity_id
    )
    register_world_fact(
        world,
        WorldFact(
            claim=f"{npc.name} was harmed in a brawl.",
            scope=WorldFactScope.ENTITY,
            subject_id=str(npc.entity_id),
            established_tick=world.tick,
            tags=["violence", "investigation"],
        ),
    )
    if other_npc.position.manhattan(npc.position) > other_npc.sight_range:
        return
    facts = ranked_facts_for_projection(world, EntityId(str(other_npc.entity_id)))
    assert any("harmed" in f.lower() for f in facts)
