"""
Per-NPC observational memory (short Markov window).

Each entity only retains what they could plausibly have seen or heard. Reactions
and dialogue use the last ``MEMORY_TICK_WINDOW`` simulation ticks — not the full
log — so behavior is locally realistic (recent events dominate, old ones decay).

Stored on ``entity.meta["observation_memory"]`` as a list of records (newest last).
Long-form compression still lives in ``world.episode_memory`` (episodic LM summaries).
"""

from __future__ import annotations

from typing import Any, Optional

from .schemas import EntityId, EntityKind, EntityState, Event, WorldState
from .social_stimulus import (
    build_stimulus_from_event,
    clear_stimulus,
    format_player_feedback,
)

# Markov window: only these ticks inform immediate reactions (configurable per world).
DEFAULT_MEMORY_TICK_WINDOW = 3
# Hard cap so one crowded tick cannot blow meta size.
MAX_OBSERVATIONS_PER_ENTITY = 12

_META_KEY = "observation_memory"


def memory_tick_window(world: WorldState) -> int:
    sem = getattr(world.config, "semantic", None)
    if sem is not None and hasattr(sem, "reaction_memory_ticks"):
        return max(1, int(sem.reaction_memory_ticks))
    return DEFAULT_MEMORY_TICK_WINDOW


def _memory_list(entity: EntityState) -> list[dict[str, Any]]:
    raw = entity.meta.get(_META_KEY)
    if not isinstance(raw, list):
        return []
    return [x for x in raw if isinstance(x, dict)]


def _set_memory_list(entity: EntityState, records: list[dict[str, Any]]) -> None:
    entity.meta[_META_KEY] = records[-MAX_OBSERVATIONS_PER_ENTITY:]


def prune_memory(entity: EntityState, world: WorldState) -> None:
    """Drop observations older than the Markov window."""
    window = memory_tick_window(world)
    cutoff = world.tick - window
    kept = [r for r in _memory_list(entity) if int(r.get("tick", -1)) >= cutoff]
    _set_memory_list(entity, kept)


def record_observation(
    entity: EntityState,
    world: WorldState,
    observation: dict[str, Any],
) -> None:
    """Append one witnessed event; prune by tick window."""
    records = _memory_list(entity)
    eid = observation.get("event_id")
    if eid and any(r.get("event_id") == eid for r in records):
        return
    records.append(observation)
    _set_memory_list(entity, records)
    prune_memory(entity, world)
    _sync_social_stimulus(entity, world)


def apply_observation_payload(
    entity: EntityState,
    world: WorldState,
    payload: dict[str, Any],
) -> None:
    """Apply an OBSERVATION_RECORDED transition payload (replay-safe)."""
    observation = {
        "tick": int(payload.get("tick", world.tick)),
        "event_id": payload.get("event_id"),
        "kind": payload.get("kind", "action"),
        "summary": str(payload.get("summary", ""))[:120],
        "priority": int(payload.get("priority", 0)),
        "actor_id": payload.get("actor_id"),
        "target_id": payload.get("target_id"),
        "addressed": bool(payload.get("addressed", False)),
    }
    if observation.get("event_id"):
        record_observation(entity, world, observation)


def observation_transition(
    entity_id: str,
    observation: dict[str, Any],
) -> "Transition":
    """Build a replay-safe OBSERVATION_RECORDED transition."""
    from .schemas import Transition, TransitionKind

    return Transition(
        kind=TransitionKind.OBSERVATION_RECORDED,
        payload={
            "entity_id": entity_id,
            "tick": observation.get("tick"),
            "event_id": observation.get("event_id"),
            "kind": observation.get("kind", "action"),
            "summary": observation.get("summary", ""),
            "priority": observation.get("priority", 0),
            "actor_id": observation.get("actor_id"),
            "target_id": observation.get("target_id"),
            "addressed": observation.get("addressed", False),
        },
    )


def _sync_social_stimulus(entity: EntityState, world: WorldState) -> None:
    """Denormalize strongest in-window observation for fast policy reads."""
    stim = get_actionable_stimulus(entity, world)
    if stim is not None:
        entity.meta["social_stimulus"] = stim
    else:
        clear_stimulus(entity)


def record_event_for_observers(
    world: WorldState, event: Event,
) -> tuple[list[str], list]:
    """
    Every entity that could perceive ``event`` stores it via OBSERVATION_RECORDED
    transitions (replay-safe). Returns UI feedback lines and transitions.
    """
    from .schemas import Transition

    from .social_stimulus import _can_perceive

    stimulus = build_stimulus_from_event(world, event)
    if stimulus is None:
        return [], []

    actor_id = stimulus["actor_id"]
    target_id = stimulus.get("target_id")
    lines: list[str] = []
    transitions: list[Transition] = []

    from .region_utils import entities_in_region, entity_or_none

    actor = entity_or_none(world, EntityId(actor_id))
    if actor is None:
        return [], []

    for eid, entity in entities_in_region(world, actor.region_id).items():
        if not entity.alive or entity.health <= 0:
            continue
        if str(eid) == actor_id:
            continue
        if entity.kind not in (EntityKind.NPC, EntityKind.PLAYER):
            continue
        if not _can_perceive(world, entity, actor_id):
            continue

        addressed = target_id is not None and str(eid) == target_id
        priority = 0
        if addressed:
            priority = 100
        elif stimulus["kind"] == "speech" and target_id is None:
            priority = 40
        elif stimulus["kind"] == "violence":
            priority = 70
        elif stimulus["kind"] in ("contact", "noise"):
            priority = 50
        else:
            priority = 20

        obs = {
            **stimulus,
            "addressed": addressed,
            "priority": priority,
        }
        transitions.append(observation_transition(str(eid), obs))

        kind = stimulus.get("kind", "action")
        if addressed and kind == "speech":
            lines.append(f"  → {entity.name} (addressed) heard you")
        elif kind == "speech" and priority >= 40:
            lines.append(f"  → {entity.name} heard: {stimulus['summary'][:50]}")
        elif kind in ("contact", "violence", "noise") and priority >= 50:
            lines.append(f"  → {entity.name} reacted to: {stimulus['summary'][:50]}")
        elif priority >= 50:
            lines.append(f"  → {entity.name} noticed: {stimulus['summary'][:50]}")

    return lines, transitions


def observations_in_window(
    entity: EntityState,
    world: WorldState,
) -> list[dict[str, Any]]:
    prune_memory(entity, world)
    return list(_memory_list(entity))


def get_actionable_stimulus(
    entity: EntityState,
    world: WorldState,
) -> Optional[dict[str, Any]]:
    """
    Best observation to react to now: prefer addressed speech, then highest priority,
    then most recent (Markov — within tick window only).
    """
    obs = observations_in_window(entity, world)
    if not obs:
        return None

    addressed = [o for o in obs if o.get("addressed")]
    if addressed:
        return addressed[-1]

    # Durable world changes outrank generic ambient noise.
    world_facts = [
        o for o in obs
        if o.get("kind") == "world_fact" and int(o.get("priority", 0)) >= 40
    ]
    if world_facts:
        return max(
            world_facts,
            key=lambda o: (int(o.get("priority", 0)), int(o.get("tick", 0))),
        )

    ranked = sorted(
        obs,
        key=lambda o: (int(o.get("priority", 0)), int(o.get("tick", 0))),
    )
    return ranked[-1] if ranked else obs[-1]


def has_actionable_memory(
    entity: EntityState,
    world: WorldState,
) -> bool:
    return get_actionable_stimulus(entity, world) is not None


def stimulus_age(world: WorldState, entity: EntityState) -> int:
    stim = get_actionable_stimulus(entity, world)
    if stim is None:
        return 999
    return world.tick - int(stim.get("tick", -999))


def has_fresh_stimulus(
    entity: EntityState,
    world: WorldState,
    max_age: Optional[int] = None,
) -> bool:
    """True if there is something in the Markov window worth reacting to."""
    if max_age is None:
        max_age = memory_tick_window(world)
    return stimulus_age(world, entity) <= max_age


def stimulus_priority(entity: EntityState) -> int:
    stim = entity.meta.get("social_stimulus")
    if not isinstance(stim, dict):
        return 0
    return int(stim.get("priority", 0))


def clear_actionable_stimulus(entity: EntityState, world: WorldState) -> None:
    """Forget the current actionable observation (e.g. after replying)."""
    stim = entity.meta.get("social_stimulus")
    if isinstance(stim, dict):
        eid = stim.get("event_id")
        if eid:
            kept = [r for r in _memory_list(entity) if r.get("event_id") != eid]
            _set_memory_list(entity, kept)
    _sync_social_stimulus(entity, world)


def format_observations_for_sheet(
    entity: EntityState,
    world: WorldState,
) -> list[str]:
    """Short lines for LM / character sheet (observed facts only)."""
    lines: list[str] = []
    for obs in observations_in_window(entity, world):
        tick = obs.get("tick", "?")
        summary = obs.get("summary", "")
        if obs.get("addressed"):
            lines.append(f"tick {tick} (to you): {summary}")
        else:
            lines.append(f"tick {tick}: {summary}")
    return lines[-6:]
