"""
Cold-tier world chronicle — compressed long-horizon memory.

Rule-based episode compression (no LM required) appends milestone entries
to ``WorldState.world_chronicle`` at episode boundaries.  Entries survive
across sessions and feed projection / NPC character sheets via
``memory_retrieval.retrieve_chronicle_snippets``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .memory_retrieval import _event_line
from .schemas import (
    EntityKind,
    TransitionKind,
    WorldChronicleEntry,
    WorldState,
)

if TYPE_CHECKING:
    from .schemas import Event

MAX_WORLD_CHRONICLE = 80
_MIN_EVENT_SCORE = 4.0


def _entry_importance_from_event(
    world: WorldState,
    event: "Event",
) -> float:
    score = 3.0
    for t in event.transitions:
        if t.kind in (
            TransitionKind.ENTITY_DIED,
            TransitionKind.ENTITY_HEALTH_CHANGED,
        ):
            score += 5.0
        if t.kind == TransitionKind.WORLD_MARK:
            score += 3.0
        if t.kind == TransitionKind.TILE_MARKED:
            score += 2.0
    actor = world.spatial.entities.get(event.action.actor)
    if actor is not None and actor.kind == EntityKind.PLAYER:
        score += 2.0
    if event.narrative_hint and len(event.narrative_hint) > 30:
        score += 1.5
    return score


def append_episode_chronicle(
    world: WorldState,
    episode_start: int,
    episode_end: int,
) -> int:
    """
    Compress significant events from ``[episode_start, episode_end)`` into
    chronicle entries.  Returns the number of entries added.
    """
    episode_events = [
        e for e in world.event_log
        if episode_start <= e.tick < episode_end
    ]
    if not episode_events:
        return 0

    added = 0
    seen_summaries: set[str] = set()

    for event in episode_events:
        importance = _entry_importance_from_event(world, event)
        if importance < _MIN_EVENT_SCORE:
            continue

        summary = _event_line(world, event)
        if summary in seen_summaries:
            continue
        seen_summaries.add(summary)

        actor_id = str(event.action.actor)
        subject_id = str(event.action.target) if event.action.target else None
        actor = world.spatial.entities.get(event.action.actor)
        region_id = actor.region_id if actor else world.active_region_id

        tags: list[str] = []
        for t in event.transitions:
            if t.kind == TransitionKind.ENTITY_DIED:
                tags.append("death")
            if t.kind in (TransitionKind.ENTITY_HEALTH_CHANGED,):
                if float(t.payload.get("delta", 0)) < -10:
                    tags.append("violence")
            if t.kind == TransitionKind.WORLD_MARK:
                tags.append("milestone")

        scope = "region" if region_id else "world"
        world.world_chronicle.append(
            WorldChronicleEntry(
                tick=event.tick,
                summary=summary[:200],
                importance=importance,
                actor_id=actor_id if actor_id != "system" else None,
                subject_id=subject_id,
                region_id=region_id,
                scope=scope,
                tags=tags,
            )
        )
        added += 1

    _trim_chronicle(world)
    return added


def _trim_chronicle(world: WorldState) -> None:
    if len(world.world_chronicle) <= MAX_WORLD_CHRONICLE:
        return
    ranked = sorted(
        enumerate(world.world_chronicle),
        key=lambda pair: (pair[1].importance, pair[1].tick),
    )
    drop_count = len(world.world_chronicle) - MAX_WORLD_CHRONICLE
    drop_indices = {idx for idx, _ in ranked[:drop_count]}
    world.world_chronicle = [
        e for i, e in enumerate(world.world_chronicle) if i not in drop_indices
    ]


__all__ = ["append_episode_chronicle", "MAX_WORLD_CHRONICLE"]
