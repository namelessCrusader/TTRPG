"""
Episodic NPC memory.

Every EPISODE_SIZE ticks, ``maybe_summarize()`` builds a short prose summary
of recent events for each entity using the LM, then stores them in
``WorldState.episode_memory``.  Long-running autonomous simulations (100+
ticks) can call this so NPCs don't lose context older than the raw
perception window.

Integration points
------------------
- Called from ``GameLoop.step()`` and ``GameLoop.autonomous_tick()`` after
  every tick when ``tick % EPISODE_SIZE == 0``.
- ``npc_policy.build_npc_character_sheet()`` reads
  ``world.episode_memory[npc_id]`` and injects them as
  ``NpcCharacterSheet.episode_summaries``.
- Player episodes also append to ``WorldState.player_chronicle`` for
  session-long story continuity.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .lm_adapter import LMAdapter
    from .schemas import EntityKind, EntityState, Event, WorldState

logger = logging.getLogger(__name__)

# How many ticks make up one episode.
EPISODE_SIZE: int = 20
# How many episode summaries to keep per entity (oldest drop off).
MAX_EPISODES_PER_ENTITY: int = 5
# Max player chronicle entries persisted across a session.
MAX_PLAYER_CHRONICLE: int = 12
# Max events to include in a single summarization prompt.
_MAX_EVENTS_IN_PROMPT: int = 15


def maybe_summarize(world: "WorldState", adapter: "LMAdapter") -> None:
    """
    Create a new episode summary when the world tick crosses an episode
    boundary (``tick % EPISODE_SIZE == 0``).

    No-ops if:
      - ``tick == 0``
      - we are not at an episode boundary
      - the event log has no events in the just-completed episode window

    Safe to call every tick — it returns immediately in the common case.
    """
    if world.tick == 0 or world.tick % EPISODE_SIZE != 0:
        return

    episode_start = world.tick - EPISODE_SIZE
    episode_end = world.tick
    episode_events = [
        e for e in world.event_log
        if episode_start <= e.tick < episode_end
    ]
    if not episode_events:
        return

    from .world_chronicle import append_episode_chronicle

    append_episode_chronicle(world, episode_start, episode_end)

    logger.debug(
        "Summarizing episode ticks %d–%d for %d entities "
        "(%d events in window)",
        episode_start, episode_end,
        len(world.spatial.entities),
        len(episode_events),
    )

    from .schemas import EntityKind

    player_id: str | None = None
    for eid, ent in world.spatial.entities.items():
        if ent.kind == EntityKind.PLAYER:
            player_id = str(eid)
            break

    for eid, entity in list(world.spatial.entities.items()):
        summary = _summarize_for_entity(
            entity, episode_events, world, adapter,
            episode_start, episode_end,
        )
        if summary:
            bucket = world.episode_memory.setdefault(eid, [])
            bucket.append(summary)
            if len(bucket) > MAX_EPISODES_PER_ENTITY:
                world.episode_memory[eid] = bucket[-MAX_EPISODES_PER_ENTITY:]

            if player_id and str(eid) == player_id:
                _append_player_chronicle(world, summary)


def _append_player_chronicle(world: "WorldState", entry: str) -> None:
    """Persist a player-facing chronicle line for REPL / recap."""
    text = entry.strip()
    if not text:
        return
    world.player_chronicle.append(text)
    if len(world.player_chronicle) > MAX_PLAYER_CHRONICLE:
        world.player_chronicle = world.player_chronicle[-MAX_PLAYER_CHRONICLE:]


def chronicle_lines(world: "WorldState", *, max_entries: int = 5) -> list[str]:
    """Recent player chronicle entries (oldest first within window)."""
    return list(world.player_chronicle[-max_entries:])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_relevant(event: "Event", entity_id: str) -> bool:
    """Return True if the entity was actor, target, or witness of the event."""
    action = event.action
    if action.actor == entity_id:
        return True
    if isinstance(action.target, str) and action.target == entity_id:
        return True
    if entity_id in event.witnesses:
        return True
    for t in event.transitions:
        payload = t.payload
        if payload.get("entity_id") == entity_id:
            return True
        if payload.get("source") == entity_id or payload.get("target") == entity_id:
            return True
    return False


def _summarize_rule_based(
    entity: "EntityState",
    events: list["Event"],
    world: "WorldState",
) -> str:
    """Deterministic fallback when LM narrate is unavailable."""
    from .narrator import render_event

    lines: list[str] = []
    for ev in events[:_MAX_EVENTS_IN_PROMPT]:
        rendered = render_event(ev, world)
        if rendered:
            lines.append(rendered[:100])

    if not lines:
        return (
            f"{entity.name} passed through ticks quietly, "
            "watching the room without decisive action."
        )

    joined = "; ".join(lines[:4])
    if len(joined) > 380:
        joined = joined[:377] + "..."
    return f"{entity.name} remembers: {joined}"


def _summarize_for_entity(
    entity: "EntityState",
    events: list["Event"],
    world: "WorldState",
    adapter: "LMAdapter",
    episode_start: int,
    episode_end: int,
) -> str | None:
    """Build a prose summary of episode events for one entity."""
    from .narrator import render_event

    relevant = [e for e in events if _is_relevant(e, entity.entity_id)]
    if not relevant:
        return None

    lines = [render_event(e, world) for e in relevant[:_MAX_EVENTS_IN_PROMPT]]
    events_text = "\n".join(f"  - {ln}" for ln in lines if ln)

    prompt = (
        f"Write a 2-3 sentence past-tense memory summary for {entity.name} "
        f"({entity.role or 'a person in a story'}), covering ticks "
        f"{episode_start}–{episode_end}. "
        f"Focus on what {entity.name} directly experienced or witnessed. "
        f"Be specific and terse; omit mechanical details.\n\n"
        f"Events:\n{events_text}\n\n"
        f"Memory summary (2-3 sentences, past tense):"
    )

    summary: str | None = None
    try:
        raw = adapter.narrate(prompt)
        if raw:
            summary = raw.strip()
            if len(summary) > 400:
                summary = summary[:397] + "..."
    except Exception as exc:
        logger.warning(
            "Episode summary failed for %s: %s", entity.name, exc
        )

    if not summary:
        summary = _summarize_rule_based(entity, relevant, world)

    logger.debug(
        "Episode summary for %s (ticks %d–%d): %s",
        entity.name, episode_start, episode_end, summary[:80],
    )
    return summary
