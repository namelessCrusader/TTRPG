"""
Per-NPC internal turn — process recent events before choosing an action.

Updates ``entity.meta`` only (no canonical world mutation). Reflection
feeds character sheets, enrichment intent hints, and reactive policy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from .needs import NEED_LABELS, critical_needs
from .schemas import EntityId, TransitionKind
from .speech_utils import is_real_dialogue, normalize_speech_key

if TYPE_CHECKING:
    from .lm_adapter import LMAdapter
    from .schemas import EntityState, Event, WorldState

logger = logging.getLogger(__name__)

REFLECTION_LOOKBACK_EVENTS = 12
MAX_HEARD_LINES = 5


def run_npc_reflection(
    entity: "EntityState",
    world: "WorldState",
    adapter: Optional["LMAdapter"] = None,
) -> None:
    """
    Internal processing step for one NPC at the start of their turn.

    Writes to ``entity.meta``:
      - heard_recently, should_reply_to, focus_topic
      - inner_thought (short situational summary)
      - reflection_tick
    """
    eid = str(entity.entity_id)
    grid = world.spatial
    heard: list[str] = []
    heard_keys: set[str] = set()
    addressed_me = False
    last_speaker_id = ""
    last_speaker_name = ""
    hostile_nearby = False
    near_hearth = False

    cfg = world.config.physics_config
    if cfg is not None:
        for ent in grid.entities.values():
            if ent.entity_id == entity.entity_id or not ent.alive:
                continue
            if ent.position.manhattan(entity.position) > entity.sight_range:
                continue
            if "on_fire" in ent.tags:
                hostile_nearby = True
                break
        for obj in grid.objects.values():
            if obj.position is None:
                continue
            if entity.position.manhattan(obj.position) > entity.sight_range:
                continue
            if "on_fire" not in obj.tags:
                continue
            if "heat_source" in obj.tags or "fixture" in obj.tags:
                near_hearth = True
                continue
            if entity.position.manhattan(obj.position) <= 3:
                hostile_nearby = True
                break
    entity.meta["near_hearth"] = near_hearth
    entity.meta["hostile_nearby"] = hostile_nearby

    for ev in reversed(world.event_log[-REFLECTION_LOOKBACK_EVENTS:]):
        for t in ev.transitions:
            if t.kind != TransitionKind.DIALOGUE_SPOKEN:
                continue
            txt = (t.payload.get("text") or "").strip()
            if not txt:
                intent_data = t.payload.get("intent")
                if isinstance(intent_data, dict):
                    txt = (
                        (intent_data.get("manner") or intent_data.get("rationale") or "")
                        .strip()
                    )
            if not txt and ev.action.intent:
                txt = (
                    (ev.action.intent.manner or ev.action.intent.rationale or "")
                    .strip()
                )
            if not is_real_dialogue(txt):
                continue
            actor_id = str(t.payload.get("actor", ev.action.actor))
            speaker = grid.entities.get(EntityId(actor_id))
            if speaker is None:
                continue
            dist = entity.position.manhattan(speaker.position)
            if dist > entity.sight_range:
                continue
            key = normalize_speech_key(txt)
            if key in heard_keys:
                continue
            heard_keys.add(key)
            heard.append(f'{speaker.name}: "{txt[:90]}"')
            if len(heard) >= MAX_HEARD_LINES:
                break
            if ev.action.target == eid or t.payload.get("target") == eid:
                addressed_me = True
                last_speaker_id = actor_id
                last_speaker_name = speaker.name
        if len(heard) >= MAX_HEARD_LINES:
            break

        # Threat / combat cues in recent log
        verb = str(ev.action.verb).lower().split(".")[-1]
        if verb in ("attack", "threaten", "draw_weapon") and (
            ev.action.target == eid or eid in (ev.witnesses or [])
        ):
            hostile_nearby = True

    entity.meta["heard_recently"] = heard
    entity.meta["should_reply_to"] = last_speaker_name if addressed_me else ""
    entity.meta["should_reply_to_id"] = last_speaker_id if addressed_me else ""
    entity.meta["reflection_tick"] = world.tick

    focus = _derive_focus_topic(entity, world, addressed_me, hostile_nearby)
    entity.meta["focus_topic"] = focus

    inner = _build_inner_thought(entity, world, heard, focus, addressed_me, last_speaker_name)
    entity.meta["inner_thought"] = inner

    _maybe_nudge_emotional_state(entity, hostile_nearby, addressed_me, world)

    from .npc_mind import interpret_mind

    interpret_mind(entity, world)

    if adapter is not None and heard and addressed_me:
        _maybe_lm_reflection(entity, world, adapter, heard, inner)


def speech_intent_for_entity(entity: "EntityState") -> str:
    """In-character goal phrase for speech enrichment (never planning stubs)."""
    focus = str(entity.meta.get("focus_topic") or "")
    mapping = {
        "need_food": "You are hungry — ask for food or ale in your own voice.",
        "need_rest": "You are exhausted — suggest resting or finding a seat.",
        "need_company": "You feel lonely — start a friendly conversation.",
        "reply": f"Reply naturally to {entity.meta.get('should_reply_to', 'them')}.",
        "pursue_drive": "Say something that advances your drive, in character.",
    }
    if focus in mapping:
        return mapping[focus]
    if entity.meta.get("should_reply_to"):
        return mapping["reply"]
    drive = (entity.drive or "").split(";")[0].strip()
    if drive:
        return f"Advance your drive: {drive[:60]}"
    return ""


def _derive_focus_topic(
    entity: "EntityState",
    world: "WorldState",
    addressed_me: bool,
    hostile_nearby: bool,
) -> str:
    if hostile_nearby:
        return "threat"
    urgent = critical_needs(entity)
    if urgent:
        need_name = urgent[0][0]
        if need_name == "hunger":
            return "need_food"
        if need_name == "fatigue":
            return "need_rest"
        if need_name == "social_need":
            return "need_company"
    if addressed_me:
        return "reply"
    if entity.meta.get("near_hearth"):
        return "ambient_warmth"
    if entity.goals:
        return "pursue_goal"
    return "pursue_drive"


def _build_inner_thought(
    entity: "EntityState",
    world: "WorldState",
    heard: list[str],
    focus: str,
    addressed_me: bool,
    last_speaker_name: str,
) -> str:
    parts: list[str] = []
    if heard:
        parts.append(f"Just heard: {heard[0]}")
        if len(heard) > 1:
            parts.append(f"(and {len(heard) - 1} more lines nearby)")
    if addressed_me and last_speaker_name:
        parts.append(f"{last_speaker_name} spoke to you — consider answering.")
    urgent = critical_needs(entity)
    if urgent:
        need_name, val = urgent[0]
        label = NEED_LABELS.get(need_name, need_name)
        parts.append(f"Urgent: {label} ({val:.0f}).")
    elif focus == "pursue_drive" and entity.drive:
        parts.append(f"On your mind: {(entity.drive.split(';')[0].strip())[:50]}.")
    if focus == "threat":
        parts.append("Something hostile happened nearby — stay alert.")
    if not parts:
        parts.append(f"Watching the room at tick {world.tick}.")
    return " ".join(parts)[:240]


def _maybe_nudge_emotional_state(
    entity: "EntityState",
    hostile_nearby: bool,
    addressed_me: bool,
    world: "WorldState | None" = None,
) -> None:
    from .schemas import AlertnessLevel, EmotionalState

    if world is not None and world.meta.get("magic_duel"):
        entity.alertness = AlertnessLevel.COMBAT
        return

    if hostile_nearby:
        entity.alertness = AlertnessLevel.HIGH
        if entity.emotional_state == EmotionalState.NEUTRAL:
            entity.emotional_state = EmotionalState.FEARFUL
        return
    # Hearth warmth — do not panic; slight comfort at most
    if entity.meta.get("near_hearth") and entity.emotional_state == EmotionalState.FEARFUL:
        entity.emotional_state = EmotionalState.NEUTRAL
    if addressed_me and entity.emotional_state == EmotionalState.NEUTRAL:
        entity.emotional_state = EmotionalState.FRIENDLY


def _maybe_lm_reflection(
    entity: "EntityState",
    world: "WorldState",
    adapter: "LMAdapter",
    heard: list[str],
    rule_thought: str,
) -> None:
    """Optional one-line LM reflection; falls back to rule_thought on failure."""
    from .lm_adapter import MockLMAdapter

    if isinstance(adapter, MockLMAdapter):
        return
    mode = world.config.npc_policy.cognition.mode
    if mode == "reactive_only":
        return
    # At most one LM reflection call per world tick (shared budget).
    budget_key = "_lm_reflection_tick"
    if world.config.extra.get(budget_key) == world.tick:
        return
    try:
        prompt = (
            f"You are {entity.name}. In one short sentence, what are you thinking "
            f"right now? No JSON. Facts only from:\n"
            f"{rule_thought}\n"
            f"Recent speech:\n" + "\n".join(f"- {h}" for h in heard[:3])
        )
        line = (adapter.narrate(prompt, world, entity.entity_id) or "").strip()
        if line and len(line) < 200 and not line.startswith("{"):
            entity.meta["inner_thought"] = line[:200]
            world.config.extra[budget_key] = world.tick
    except Exception as exc:
        logger.debug("LM reflection skipped for %s: %s", entity.name, exc)
