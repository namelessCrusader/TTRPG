"""
Multi-turn dialogue threading.

When two entities engage in a speaking verb (speak, ask, whisper, tell,
flirt, taunt, …) the engine opens a ``ConversationThread`` between them.
Subsequent exchanges between the same pair extend the thread.  A thread
closes automatically when neither party speaks for ``THREAD_IDLE_TICKS``
ticks.

The active thread is injected into each participant's ``NpcCharacterSheet``
as ``active_dialogue``, a compact prose summary, so the NPC's LM call sees
the conversation history without relying on the raw perception window.

Integration points
------------------
- ``update_threads(world, event)`` — called in ``GameLoop._step_one_entity()``
  and ``GameLoop.step()`` after each event is recorded.
- ``get_active_thread_for(world, entity_id)`` — read by
  ``npc_policy.build_npc_character_sheet()``.
- Threads are stored on ``WorldState.active_conversations`` so they
  persist in savegames.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .schemas import (
        ConversationThread,
        DialogueExchange,
        EntityId,
        Event,
        WorldState,
    )

# A thread is considered "idle" and closed once this many ticks have passed
# since the last exchange.
THREAD_IDLE_TICKS: int = 5

# Verbs that constitute "speaking" for the purpose of threading.
_SPEAKING_VERBS: frozenset[str] = frozenset(
    {
        "speak",
        "ask",
        "tell",
        "say",
        "whisper",
        "shout",
        "taunt",
        "flirt",
        "seduce",
        "threaten",
        "plead",
        "beg",
        "argue",
        "gossip",
        "confess",
        "apologize",
        "boast",
        "negotiate",
        "haggle",
        "complain",
        "joke",
        "tease",
        "comfort",
        "persuade",
        "intimidate",
        "challenge",
        "accuse",
        "praise",
        "insult",
    }
)

# How many exchanges to keep per thread (oldest drop off).
_MAX_EXCHANGES: int = 10


def is_speaking_verb(verb: str) -> bool:
    """Return True for any verb that counts as dialogue."""
    return verb.lower() in _SPEAKING_VERBS


def update_threads(world: "WorldState", event: "Event") -> None:
    """
    Update conversation threads based on a newly recorded event.

    Called after every event.  Does two things:
      1. If the event is a speaking verb between two entities, opens or
         extends their ConversationThread.
      2. Closes threads that have been idle for THREAD_IDLE_TICKS ticks.
    """
    from .schemas import ConversationThread, DialogueExchange

    verb = str(event.action.verb).lower()
    actor_id = event.action.actor
    target = event.action.target
    tick = event.tick

    if is_speaking_verb(verb) and isinstance(target, str) and target:
        _open_or_extend(world, event, actor_id, target, verb, tick)

    # Close idle threads
    for thread in world.active_conversations:
        if not thread.closed and (tick - thread.last_exchange_at) >= THREAD_IDLE_TICKS:
            thread.closed = True


def get_active_thread_for(
    world: "WorldState",
    entity_id: "EntityId",
) -> "Optional[ConversationThread]":
    """Return the most recent non-closed thread this entity is in, or None."""
    candidates = [
        t for t in world.active_conversations
        if not t.closed and entity_id in t.participant_ids
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda t: t.last_exchange_at)


def format_thread_for_npc(
    thread: "ConversationThread",
    pov_entity_id: "EntityId",
) -> str:
    """
    Format the thread as a compact prose summary for an NPC character sheet.

    Example output:
        [Active conversation with Tomas (started tick 12)]
        tick 12 | You: "Have you news from the east?"
        tick 13 | Tomas: "What's it worth to you?"
        tick 14 | You: "A round of drinks."
    """
    other_names = [
        name
        for pid, name in zip(thread.participant_ids, thread.participant_names)
        if pid != pov_entity_id
    ]
    other = ", ".join(other_names) if other_names else "someone"
    header = (
        f"[Active conversation with {other} "
        f"(started tick {thread.opened_at})]"
    )
    lines: list[str] = [header]
    for ex in thread.exchanges[-_MAX_EXCHANGES:]:
        speaker = (
            "You" if ex.speaker_id == pov_entity_id else ex.speaker_name
        )
        text_part = f': "{ex.text}"' if ex.text else f" [{ex.verb}]"
        lines.append(f"  tick {ex.tick} | {speaker}{text_part}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_thread_id() -> str:
    return f"thr_{uuid.uuid4().hex[:8]}"


def _canonical_pair(a: str, b: str) -> list[str]:
    """Return [a, b] sorted so the same pair always maps to the same key."""
    return sorted([a, b])


def _open_or_extend(
    world: "WorldState",
    event: "Event",
    actor_id: str,
    target_id: str,
    verb: str,
    tick: int,
) -> None:
    """Open a new thread or append an exchange to an existing open thread."""
    from .schemas import ConversationThread, DialogueExchange

    pair = _canonical_pair(actor_id, target_id)

    # Find an existing open thread for this exact pair
    existing = next(
        (
            t
            for t in world.active_conversations
            if not t.closed and sorted(t.participant_ids) == pair
        ),
        None,
    )

    # Resolve display names
    def _name(eid: str) -> str:
        ent = world.spatial.entities.get(eid)  # type: ignore[arg-type]
        return ent.name if ent else eid

    text = None
    if event.action.intent:
        text = event.action.intent.rationale or None

    exchange = DialogueExchange(
        tick=tick,
        speaker_id=actor_id,
        speaker_name=_name(actor_id),
        listener_id=target_id,
        listener_name=_name(target_id),
        verb=verb,
        text=text,
    )

    if existing is not None:
        existing.exchanges.append(exchange)
        existing.last_exchange_at = tick
        # Trim to max
        if len(existing.exchanges) > _MAX_EXCHANGES:
            existing.exchanges = existing.exchanges[-_MAX_EXCHANGES:]
    else:
        names = [_name(pid) for pid in pair]
        thread = ConversationThread(
            thread_id=_make_thread_id(),
            participant_ids=pair,
            participant_names=names,
            exchanges=[exchange],
            opened_at=tick,
            last_exchange_at=tick,
        )
        world.active_conversations.append(thread)
