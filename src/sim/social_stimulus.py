"""
Broadcast actionable social stimuli so NPCs react to what others just did.

After any entity's action is committed to the event log, nearby entities receive
a structured ``social_stimulus`` entry in ``entity.meta``.  ReactivePolicy treats
fresh stimuli as highest-priority behavior (reply, alarm, step back, etc.).
"""

from __future__ import annotations

from typing import Any, Optional

from .schemas import (
    ActionType,
    EntityId,
    EntityKind,
    EntityState,
    Event,
    IntentBlock,
    SemanticAction,
    TransitionKind,
    WorldState,
)
from .spatial import visible_from

_SPEECH_VERBS = frozenset({
    "speak", "say", "tell", "ask", "shout", "whisper", "yell", "call",
    "announce", "mutter", "greet", "challenge", "warn", "threaten", "sing",
})
_VIOLENCE_VERBS = frozenset({
    "attack", "accuse", "threaten", "intimidate", "fight", "strike",
})
_CONTACT_VERBS = frozenset({
    "contact", "kiss", "hug", "punch", "slap", "grab", "push", "shove",
})


def _verb_str(verb: Any) -> str:
    if isinstance(verb, str):
        return verb.lower().split(".")[-1]
    return str(verb).lower().split(".")[-1]


def _entity_name(world: WorldState, eid: str) -> str:
    from .region_utils import entity_name as _lookup_name

    return _lookup_name(world, eid)


def _extract_dialogue_text(event: Event) -> str:
    for t in event.transitions:
        if t.kind == TransitionKind.DIALOGUE_SPOKEN:
            text = (t.payload.get("text") or "").strip()
            if text and text != "...":
                return text
    intent = event.action.intent
    if intent and intent.manner:
        m = intent.manner.strip()
        if m and not m.startswith("["):
            return m
    return ""


def _stimulus_kind_for_event(event: Event) -> str:
    verb = _verb_str(event.action.verb)
    if verb in _SPEECH_VERBS:
        return "speech"
    if verb in _VIOLENCE_VERBS:
        return "violence"
    if verb in _CONTACT_VERBS:
        return "contact"
    for t in event.transitions:
        k = t.kind.value if hasattr(t.kind, "value") else str(t.kind)
        if k == TransitionKind.ENTITY_DIED.value:
            return "violence"
        if k == TransitionKind.ENTITY_HEALTH_CHANGED.value:
            if float(t.payload.get("delta", 0)) < -10:
                return "violence"
        if k == TransitionKind.NOISE_EVENT.value:
            return "noise"
    return "action"


def build_stimulus_from_event(world: WorldState, event: Event) -> Optional[dict[str, Any]]:
    """Summarise one event as a stimulus dict (or None if not worth broadcasting)."""
    actor_id = str(event.action.actor)
    if actor_id == "system":
        return None

    verb = _verb_str(event.action.verb)
    kind = _stimulus_kind_for_event(event)
    text = ""
    if kind == "speech":
        from .speech_utils import sanitize_dialogue_text

        raw = _extract_dialogue_text(event)
        text = sanitize_dialogue_text(raw)

    target_id: Optional[str] = None
    raw_target = event.action.target
    if isinstance(raw_target, str):
        target_id = raw_target

    from .region_utils import entity_or_none

    actor_ent = entity_or_none(world, EntityId(actor_id))
    actor_kind = actor_ent.kind.value if actor_ent else "npc"

    summary_parts = [f"{_entity_name(world, actor_id)} {verb}"]
    if target_id:
        summary_parts.append(f"→ {_entity_name(world, target_id)}")
    if text:
        summary_parts.append(f'"{text[:80]}"')

    return {
        "event_id": str(event.event_id),
        "tick": event.tick,
        "kind": kind,
        "verb": verb,
        "actor_id": actor_id,
        "actor_name": _entity_name(world, actor_id),
        "actor_kind": actor_kind,
        "target_id": target_id,
        "text": text,
        "summary": " ".join(summary_parts),
        "loud": any(
            t.kind == TransitionKind.DIALOGUE_SPOKEN and t.payload.get("loud")
            for t in event.transitions
        ),
    }


def _can_perceive(world: WorldState, observer: EntityState, actor_id: str) -> bool:
    from .region_utils import entity_or_none, grid_for

    if str(observer.entity_id) == actor_id:
        return False
    actor = entity_or_none(world, EntityId(actor_id))
    if actor is None or not actor.alive:
        return False
    if observer.region_id != actor.region_id:
        return False
    try:
        grid = grid_for(world, observer.entity_id)
    except KeyError:
        return False
    visible = visible_from(grid, observer.position, observer.sight_range)
    return actor.position in visible


def broadcast_stimulus_from_event(world: WorldState, event: Event) -> list[str]:
    """
    Record what each nearby entity could observe and return UI feedback lines.

    Applies OBSERVATION_RECORDED transitions so witness memory is replay-safe.
    """
    from .compiler import apply_transitions
    from .observation_memory import record_event_for_observers

    lines, transitions = record_event_for_observers(world, event)
    if transitions:
        apply_transitions(world, transitions)
        event.transitions.extend(transitions)
    return lines


def stimulus_age(world: WorldState, entity: EntityState) -> int:
    from .observation_memory import stimulus_age as _age

    return _age(world, entity)


def has_fresh_stimulus(
    entity: EntityState,
    world: WorldState,
    max_age: Optional[int] = None,
) -> bool:
    from .observation_memory import has_fresh_stimulus as _fresh

    return _fresh(entity, world, max_age)


def stimulus_priority(entity: EntityState) -> int:
    from .observation_memory import stimulus_priority as _prio

    return _prio(entity)


def clear_stimulus(entity: EntityState) -> None:
    entity.meta.pop("social_stimulus", None)


def format_player_feedback(lines: list[str]) -> str:
    if not lines:
        return ""
    return "[Others noticed]\n" + "\n".join(lines)
