"""
World Memory — durable facts become NPC knowledge and reactions.

When adjudication (or other systems) registers ``WorldFact`` records, this
module ensures the simulation *reacts*:

  1. ``seed_beliefs_from_world_facts`` — perceiving NPCs gain BELIEVES_CLAIM
     edges so rumours propagate through the existing belief system.

  2. ``record_facts_in_observation_memory`` — nearby NPCs store the fact in
     their short Markov observation window, driving immediate policy reactions.

All outputs are ``Transition`` objects applied via ``apply_transitions``.
No LM calls — deterministic and replay-safe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .schemas import (
    EdgeKind,
    EntityId,
    EntityKind,
    Transition,
    TransitionKind,
    WorldFact,
    WorldFactScope,
    WorldState,
)

if TYPE_CHECKING:
    pass

# Max NPCs that learn a world-scoped fact in one tick (avoid O(n²) blowups).
_MAX_WORLD_FACT_OBSERVERS = 8
# Proximity for tile/entity-scoped facts.
_TILE_OBSERVER_RANGE = 10
_ENTITY_OBSERVER_RANGE = 8


def _memory_window(world: WorldState) -> int:
    return int(world.config.consequence_policy.world_memory_window)


def _observers_for_fact(world: WorldState, fact: WorldFact) -> list[str]:
    """Return entity ids that could plausibly learn ``fact`` this tick."""
    from .region_utils import (
        entities_in_region,
        entity_or_none,
        iter_entities,
        parse_scoped_tile_key,
    )

    observers: list[str] = []

    if fact.scope == WorldFactScope.TILE and fact.subject_id:
        region_id, focal = parse_scoped_tile_key(fact.subject_id)
        candidates = (
            entities_in_region(world, region_id).values()
            if region_id
            else (ent for _, ent, _ in iter_entities(world))
        )
        for ent in candidates:
            if not ent.alive or ent.kind == EntityKind.PLAYER:
                continue
            if region_id and ent.region_id != region_id:
                continue
            if ent.position.manhattan(focal) <= _TILE_OBSERVER_RANGE:
                if ent.position.manhattan(focal) <= ent.sight_range:
                    observers.append(str(ent.entity_id))

    elif fact.scope == WorldFactScope.ENTITY and fact.subject_id:
        subject = entity_or_none(world, EntityId(fact.subject_id))
        if subject:
            for eid, ent in entities_in_region(world, subject.region_id).items():
                if not ent.alive or ent.kind == EntityKind.PLAYER:
                    continue
                if str(eid) == fact.subject_id:
                    observers.append(str(eid))
                elif ent.position.manhattan(subject.position) <= _ENTITY_OBSERVER_RANGE:
                    if ent.position.manhattan(subject.position) <= ent.sight_range:
                        observers.append(str(eid))

    elif fact.scope == WorldFactScope.REGION:
        region_id = fact.subject_id or world.active_region_id
        for eid, ent in entities_in_region(world, region_id).items():
            if not ent.alive or ent.kind == EntityKind.PLAYER:
                continue
            observers.append(str(eid))

    else:
        establisher = fact.established_by
        actor_ent = (
            entity_or_none(world, EntityId(establisher)) if establisher else None
        )
        if actor_ent is not None:
            for eid, ent in entities_in_region(world, actor_ent.region_id).items():
                if not ent.alive or ent.kind == EntityKind.PLAYER:
                    continue
                if str(eid) == establisher:
                    continue
                if ent.position.manhattan(actor_ent.position) <= ent.sight_range:
                    observers.append(str(eid))
        else:
            for eid, ent, _ in iter_entities(world):
                if ent.alive and ent.kind != EntityKind.PLAYER:
                    observers.append(str(eid))

    seen: set[str] = set()
    unique: list[str] = []
    for oid in observers:
        if oid in seen:
            continue
        seen.add(oid)
        unique.append(oid)
    return unique[:_MAX_WORLD_FACT_OBSERVERS]


def _already_believes(world: WorldState, entity_id: str, claim: str) -> bool:
    graph = world.relational
    for _tgt, edges in graph.edges.get(entity_id, {}).items():
        for edge in edges:
            if (
                edge.kind == EdgeKind.BELIEVES_CLAIM
                and edge.meta.get("claim") == claim
            ):
                return True
    return False


def _observation_already_recorded(entity, fact_id: str) -> bool:
    from .observation_memory import _memory_list

    return any(r.get("event_id") == fact_id for r in _memory_list(entity))


def seed_beliefs_from_world_facts(
    world: WorldState,
    *,
    recent_tick_window: Optional[int] = None,
) -> list[Transition]:
    """
    Turn newly established world facts into BELIEVES_CLAIM edges for observers.

    Idempotent: skips facts older than ``recent_tick_window`` or claims an
    observer already holds.
    """
    if recent_tick_window is None:
        recent_tick_window = _memory_window(world)

    transitions: list[Transition] = []
    tick = world.tick

    for fact in world.world_facts:
        if tick - fact.established_tick > recent_tick_window:
            continue

        source = fact.established_by or "world"
        claim = fact.claim.strip()
        if not claim:
            continue

        for observer_id in _observers_for_fact(world, fact):
            if _already_believes(world, observer_id, claim):
                continue
            transitions.append(
                Transition(
                    kind=TransitionKind.BELIEF_PROPAGATED,
                    payload={
                        "from_entity": source,
                        "to_entity": observer_id,
                        "belief": claim[:200],
                        "original_source": source,
                        "fidelity": round(float(fact.confidence), 3),
                        "fact_id": fact.fact_id,
                    },
                )
            )

    return transitions


def record_facts_in_observation_memory(
    world: WorldState,
    *,
    recent_tick_window: Optional[int] = None,
) -> None:
    """
    Write recent world facts into nearby NPC observation memory.

    Mutates entity meta in place (observational layer — not event-logged).
    Idempotent via fact_id deduplication.
    """
    from .observation_memory import record_observation

    if recent_tick_window is None:
        recent_tick_window = _memory_window(world)

    tick = world.tick
    for fact in world.world_facts:
        if tick - fact.established_tick > recent_tick_window:
            continue

        priority = 45
        if "environment" in fact.tags:
            priority = 55
        if fact.scope == WorldFactScope.ENTITY:
            priority = 60

        for observer_id in _observers_for_fact(world, fact):
            from .region_utils import entity_or_none

            ent = entity_or_none(world, EntityId(observer_id))
            if ent is None or not ent.alive:
                continue
            if _observation_already_recorded(ent, fact.fact_id):
                continue
            record_observation(
                ent,
                world,
                {
                    "tick": fact.established_tick,
                    "event_id": fact.fact_id,
                    "kind": "world_fact",
                    "summary": fact.claim[:120],
                    "priority": priority,
                    "actor_id": fact.established_by or "world",
                },
            )


def synthesize_beliefs_from_observations(
    world: WorldState,
) -> list[Transition]:
    """
    NPCs synthesize important direct observations into long-term beliefs.

    Look at the short-term observation memory of each alive NPC. Any high-priority
    observation (priority >= 40) is synthesized into a BELIEVES_CLAIM edge
    toward the actor of that observation (or 'world').
    """
    from .region_utils import iter_entities
    from .observation_memory import _memory_list

    transitions: list[Transition] = []

    for eid, ent, _ in iter_entities(world):
        if not ent.alive or ent.kind == EntityKind.PLAYER:
            continue

        eid_str = str(eid)
        obs_list = _memory_list(ent)
        for obs in obs_list:
            priority = int(obs.get("priority", 0))
            if priority < 40:
                continue

            claim = str(obs.get("summary", "")).strip()
            if not claim:
                continue

            # Check if the NPC already believes this claim
            if _already_believes(world, eid_str, claim):
                continue

            source = obs.get("actor_id") or "world"
            # Limit the source string length and make sure it is valid
            if source == eid_str:
                source = "world"

            fidelity = 0.9 if priority >= 60 else 0.7

            transitions.append(
                Transition(
                    kind=TransitionKind.BELIEF_PROPAGATED,
                    payload={
                        "from_entity": source,
                        "to_entity": eid_str,
                        "belief": claim[:200],
                        "original_source": source,
                        "fidelity": fidelity,
                        "fact_id": obs.get("event_id"),
                    },
                )
            )
    return transitions


def process_world_memory(world: WorldState) -> list[Transition]:
    """
    Run the full world-memory pass for the current tick.

    Returns transitions to apply (belief seeding and synthesis). Observation memory is
    updated in place.
    """
    window = _memory_window(world)
    record_facts_in_observation_memory(world, recent_tick_window=window)
    
    # 1. Seed beliefs from world facts
    transitions = seed_beliefs_from_world_facts(world, recent_tick_window=window)
    
    # 2. Synthesize beliefs from direct observations
    transitions.extend(synthesize_beliefs_from_observations(world))
    
    return transitions


__all__ = [
    "process_world_memory",
    "record_facts_in_observation_memory",
    "seed_beliefs_from_world_facts",
    "synthesize_beliefs_from_observations",
]
