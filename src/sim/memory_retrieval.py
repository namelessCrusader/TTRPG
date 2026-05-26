"""
Long-horizon memory retrieval — query the event log and chronicle by relevance.

Hot tier: observation_memory (3 ticks) — unchanged, in observation_memory.py.
Warm tier: episode_memory (LM/rule summaries every 20 ticks) — episodic_memory.py.
Cold tier: world_chronicle (compressed milestones) — world_chronicle.py.

This module scores and retrieves past events/facts for projection and adjudication
without exposing raw WorldState to the LM.
"""

from __future__ import annotations

from typing import Optional

from .schemas import (
    EntityId,
    TransitionKind,
    WorldFact,
    WorldFactScope,
    WorldState,
    coord_key,
)

# Tag weights for durable fact importance (higher = keep longer).
_FACT_TAG_WEIGHTS: dict[str, float] = {
    "violence": 10.0,
    "death": 12.0,
    "bribe": 8.0,
    "intimidation": 7.0,
    "faction": 6.0,
    "investigation": 5.0,
    "message": 5.0,
    "environment": 4.0,
    "trade": 3.0,
    "craft": 3.0,
    "rest": 1.0,
    "adjudicated": 2.0,
}

_SCOPE_WEIGHTS: dict[WorldFactScope, float] = {
    WorldFactScope.WORLD: 3.0,
    WorldFactScope.REGION: 2.5,
    WorldFactScope.ENTITY: 4.0,
    WorldFactScope.TILE: 2.0,
}

_VIOLENT_TRANSITIONS = frozenset({
    TransitionKind.ENTITY_HEALTH_CHANGED,
    TransitionKind.ENTITY_DIED,
    TransitionKind.NOISE_EVENT,
})


def fact_importance(fact: WorldFact, world: WorldState) -> float:
    """Score a durable fact for retention and projection priority."""
    score = float(fact.confidence)
    score += _SCOPE_WEIGHTS.get(fact.scope, 1.0)
    for tag in fact.tags:
        score += _FACT_TAG_WEIGHTS.get(tag, 0.5)
    age = max(0, world.tick - fact.established_tick)
    score += max(0.0, 8.0 - age * 0.15)
    if fact.established_by:
        from .schemas import EntityKind

        ent = world.spatial.entities.get(EntityId(fact.established_by))
        if ent is not None and ent.kind == EntityKind.PLAYER:
            score += 3.0
    if len(fact.claim) > 80:
        score += 1.0
    return score


def _fact_relevant_to_focal(fact: WorldFact, world: WorldState, focal_id: str) -> bool:
    from .region_utils import entity_or_none, parse_scoped_tile_key, scoped_tile_key

    focal = entity_or_none(world, EntityId(focal_id))
    if focal is None:
        return fact.scope == WorldFactScope.WORLD

    focal_tile_key = scoped_tile_key(focal.region_id, focal.position)
    legacy_tile_key = coord_key(focal.position)
    if fact.scope == WorldFactScope.WORLD:
        return True
    if fact.scope == WorldFactScope.REGION:
        return fact.subject_id in (None, world.active_region_id, focal.region_id)
    if fact.scope == WorldFactScope.TILE:
        if fact.subject_id in (focal_tile_key, legacy_tile_key):
            return True
        if fact.subject_id:
            fact_region, fact_pos = parse_scoped_tile_key(fact.subject_id)
            if fact_region and fact_region != focal.region_id:
                return False
            if focal.position.manhattan(fact_pos) <= focal.sight_range:
                return True
        return False
    if fact.scope == WorldFactScope.ENTITY:
        if fact.subject_id in (focal_id, None):
            return True
        if fact.subject_id:
            subject = entity_or_none(world, EntityId(fact.subject_id))
            if (
                subject is not None
                and subject.region_id == focal.region_id
                and focal.position.manhattan(subject.position) <= focal.sight_range
            ):
                return True
        return False
    if fact.established_by == focal_id:
        return True
    return False


def ranked_facts_for_projection(
    world: WorldState,
    focal_entity_id: EntityId,
    *,
    max_items: int = 6,
) -> list[str]:
    """Return fact claims sorted by importance × relevance."""
    focal_id = str(focal_entity_id)
    scored: list[tuple[float, str]] = []
    for fact in world.world_facts:
        if not _fact_relevant_to_focal(fact, world, focal_id):
            continue
        score = fact_importance(fact, world)
        if fact.established_by == focal_id:
            score += 2.0
        if fact.scope == WorldFactScope.ENTITY and fact.subject_id == focal_id:
            score += 3.0
        scored.append((score, fact.claim))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [claim for _, claim in scored[:max_items]]


def _event_line(world: WorldState, event) -> str:
    hint = (event.narrative_hint or "").strip()
    if hint:
        return f"tick {event.tick}: {hint[:140]}"
    verb = str(event.action.verb).split(".")[-1]
    actor = world.spatial.entities.get(EntityId(str(event.action.actor)))
    aname = actor.name if actor else str(event.action.actor)
    return f"tick {event.tick}: {aname} {verb}"


def _event_relevance_score(
    world: WorldState,
    event,
    focal_id: str,
    *,
    keyword_boost: set[str] | None = None,
) -> float:
    score = 0.0
    action = event.action
    actor_id = str(action.actor)
    target_id = str(action.target) if action.target else ""

    if actor_id == focal_id:
        score += 6.0
    if target_id == focal_id:
        score += 9.0
    if focal_id in (str(w) for w in event.witnesses):
        score += 5.0

    for t in event.transitions:
        p = t.payload
        if str(p.get("entity_id", "")) == focal_id:
            score += 4.0
        if t.kind in _VIOLENT_TRANSITIONS:
            score += 3.0
        if t.kind == TransitionKind.WORLD_MARK:
            score += 2.5
        if t.kind == TransitionKind.TILE_MARKED:
            score += 2.0

    age = max(0, world.tick - event.tick)
    score += max(0.0, 10.0 - age * 0.12)

    if event.narrative_hint and len(event.narrative_hint) > 20:
        score += 1.5

    # Keyword relevance boost: if the event mentions entities currently visible
    # to the focal entity, it's more likely to be pertinent right now.
    if keyword_boost:
        hint_lower = (event.narrative_hint or "").lower()
        raw_input_lower = (action.raw_input or "").lower()
        for kw in keyword_boost:
            if kw in hint_lower or kw in raw_input_lower:
                score += 2.0
                break

    return score


def retrieve_relevant_events(
    world: WorldState,
    focal_entity_id: EntityId,
    *,
    max_items: int = 5,
    lookback_ticks: int = 100,
) -> list[str]:
    """Retrieve scored event-log snippets for the focal entity.

    Keywords from currently visible entities and recent world facts are used
    to boost relevance of older events, so past interactions with nearby NPCs
    resurface even after many ticks.
    """
    focal_id = str(focal_entity_id)
    cutoff = world.tick - lookback_ticks
    scored: list[tuple[float, str]] = []

    # Build keyword boost set from visible entities and recent facts
    keyword_boost: set[str] = set()
    focal = world.spatial.entities.get(focal_entity_id)
    if focal is not None:
        for eid, ent in world.spatial.entities.items():
            if not ent.alive or eid == focal_entity_id:
                continue
            dist = focal.position.manhattan(ent.position)
            if dist <= focal.sight_range:
                keyword_boost.add(ent.name.lower())
    for fact in world.world_facts[-10:]:
        for word in fact.claim.lower().split():
            if len(word) >= 5:
                keyword_boost.add(word)

    for event in reversed(world.event_log):
        if event.tick < cutoff:
            break
        if str(event.action.actor) == "system" and not event.narrative_hint:
            continue
        score = _event_relevance_score(
            world, event, focal_id, keyword_boost=keyword_boost
        )
        if score < 2.0:
            continue
        scored.append((score, _event_line(world, event)))

    scored.sort(key=lambda x: x[0], reverse=True)
    seen: set[str] = set()
    lines: list[str] = []
    for _, line in scored:
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
        if len(lines) >= max_items:
            break
    lines.reverse()
    return lines


def retrieve_chronicle_snippets(
    world: WorldState,
    focal_entity_id: EntityId,
    *,
    max_items: int = 4,
) -> list[str]:
    """Cold-tier chronicle lines relevant to the focal entity."""
    focal_id = str(focal_entity_id)
    focal = world.spatial.entities.get(focal_entity_id)
    region_id = focal.region_id if focal else world.active_region_id

    scored: list[tuple[float, str]] = []
    for entry in world.world_chronicle:
        score = float(entry.importance)
        if entry.actor_id == focal_id:
            score += 4.0
        if entry.subject_id == focal_id:
            score += 5.0
        if entry.region_id and entry.region_id == region_id:
            score += 2.0
        if entry.scope == "world":
            score += 1.0
        age = max(0, world.tick - entry.tick)
        score += max(0.0, 6.0 - age * 0.05)
        scored.append((score, entry.summary))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:max_items]]


def memories_for_projection(
    world: WorldState,
    focal_entity_id: EntityId,
    *,
    max_retrieved: int = 5,
    max_chronicle: int = 4,
) -> tuple[list[str], list[str]]:
    """Return (retrieved_event_memories, chronicle_snippets) for LM projection."""
    retrieved = retrieve_relevant_events(
        world, focal_entity_id, max_items=max_retrieved,
    )
    chronicle = retrieve_chronicle_snippets(
        world, focal_entity_id, max_items=max_chronicle,
    )
    return retrieved, chronicle


__all__ = [
    "fact_importance",
    "ranked_facts_for_projection",
    "retrieve_relevant_events",
    "retrieve_chronicle_snippets",
    "memories_for_projection",
]
