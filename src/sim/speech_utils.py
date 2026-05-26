"""
Shared helpers for judging whether text is real spoken dialogue.

Used by the narrator, NPC policy enrichment, perception lines, and LM prompts
so small models cannot pass prompt fragments through as dialogue.
"""

from __future__ import annotations

# Bare tone words that should never be rendered as dialogue
TONE_WORDS: frozenset[str] = frozenset({
    "authoritative", "stern", "neutral", "calm", "friendly", "hostile",
    "hushed", "alarmed", "quiet", "loud", "confident", "nervous", "angry",
    "sad", "happy", "fearful", "suspicious", "curious", "bored", "tired",
    "firm", "firmly", "gentle", "cold", "warm", "stiff", "formal", "informal",
    "gruff", "polite", "serious", "grave", "urgent", "casual", "harsh",
    "soft", "softly", "sharp", "flat", "deadpan", "earnest", "weary",
    "purposeful", "reactive", "conversational",
})

# Verbs that must carry utterance-quality text in intent (compiler-enforced).
SPEECH_LIKE_VERBS: frozenset[str] = frozenset({
    "speak", "say", "tell", "ask", "shout", "whisper", "yell",
    "call", "announce", "mutter", "greet", "challenge", "warn",
    "threaten", "sing", "gossip", "accuse", "apologize", "joke",
    "console", "tease", "lie", "promise", "flirt", "compliment",
    "haggle", "mock", "scoff",
})

# Phrases copied from verb-guide examples; never treat as finished dialogue.
_PROMPT_FRAGMENT_SUBSTRINGS: tuple[str, ...] = (
    "bawdy ballad",
    "softly across the cheek",
    "a firm handshake",
    "by the arm",
    "pulling back",
    "in peace",
    "the eastern gate",
)

_INTENT_STARTERS: frozenset[str] = frozenset({
    "demand", "request", "threaten", "offer", "suggest", "propose",
    "address", "rebuke", "warn", "challenge", "respond", "react",
    "comment", "whisper", "ask",
})

# Substrings that indicate reactive / goal stubs, not spoken lines.
_PLANNING_SUBSTRINGS: tuple[str, ...] = (
    " or know where to",
    "know where to get",
    "before i can act",
    "who else knows about",
    "need to know before",
    "desperately ask",
    "must eat",
    "must rest",
    "stomach is growling",
)

# Placeholder manners from candidate generation — always enrich.
_ENRICHMENT_PLACEHOLDERS: frozenset[str] = frozenset({
    "in character", "reactive", "conversational",
})

# Candidate / policy rationale templates — never spoken or re-broadcast.
_REPLY_STUB_PREFIXES: tuple[str, ...] = (
    "reply to ",
    "respond to ",
    "react to ",
    "speak to ",
)


def is_reply_rationale_stub(text: str) -> bool:
    """Candidate / policy placeholder — never broadcast or store as dialogue."""
    if not text:
        return False
    lower = text.lower().strip()
    if any(lower.startswith(p) for p in _REPLY_STUB_PREFIXES):
        return True
    # Nested reply-loop garbage from broken enrichment cycles
    if lower.count("reply to ") >= 2:
        return True
    return False


def is_planning_text(text: str) -> bool:
    """
    Return True if the text looks like an internal planning string rather
    than real spoken dialogue.
    """
    if not text:
        return True
    if is_reply_rationale_stub(text):
        return True
    lower = text.lower().strip()
    words = lower.split()
    if len(words) == 1 and lower.strip(",.!?") in TONE_WORDS:
        return True
    stripped_words = [w.strip(",.!?") for w in words]
    if stripped_words and all(w in TONE_WORDS for w in stripped_words):
        return True
    if words and words[0] in _INTENT_STARTERS and len(words) >= 3:
        return True
    # Meta-goals copied from drive / infer_npc (not in-character dialogue)
    _PLANNING_PREFIXES = (
        "i need to ", "to gather ", "to ensure ", "perform a ", "seeking to ",
        "i will ", "i must ", "gather insights", "eastern trade route",
    )
    if any(lower.startswith(p) for p in _PLANNING_PREFIXES):
        return True
    if lower.startswith("ask if ") or lower.startswith("ask whether "):
        return True
    if any(sub in lower for sub in _PLANNING_SUBSTRINGS):
        return True
    # Quest objective stubs from goals.yaml (not spoken lines)
    _META_GOAL_MARKERS = (
        "without getting involved",
        "decide whether to",
        "reduce ",
        "prevent ",
        "defeat ",
        "find out what",
        "get through tonight",
    )
    if any(m in lower for m in _META_GOAL_MARKERS):
        return True
    return False


def matches_entity_goal_text(text: str, goals: list[str]) -> bool:
    """True if text is essentially a quest objective line (not dialogue)."""
    if not text or not goals:
        return False
    key = normalize_speech_key(text)
    for g in goals:
        gkey = normalize_speech_key(g)
        if not gkey:
            continue
        if key == gkey or key in gkey or gkey in key:
            return True
    return False


def _normalize_verb(verb: object) -> str:
    return str(verb).lower().split(".")[-1]


def action_needs_spoken_line(action) -> bool:
    return _normalize_verb(getattr(action, "verb", "")) in SPEECH_LIKE_VERBS


def is_descriptor_not_utterance(text: str) -> bool:
    """
    True when text looks like a mood/personality label list, not speech.

    Generic structural check (comma-heavy, no sentence end, no dialogue openers).
    """
    t = text.strip()
    if not t or t[-1] in ".!?…\"":
        return False
    if "," not in t:
        return False
    words = t.split()
    if len(words) > 16:
        return False
    lower = t.lower()
    if any(
        lower.startswith(p)
        for p in ("i ", "i'", "you ", "we ", "they ", "what ", "how ", "why ", "the ")
    ):
        return False
    return True


def echoes_character_blurb(
    text: str,
    *,
    personality: str = "",
    drive: str = "",
    role: str = "",
    emotional_state: str = "",
) -> bool:
    """True when spoken text is mostly copied from the character sheet blurb."""
    key = normalize_speech_key(text)
    if len(key) < 10:
        return False
    blob = normalize_speech_key(
        " ".join(filter(None, [personality, drive, role, emotional_state]))
    )
    if len(blob) < 8:
        return False
    words = {w for w in key.replace(",", " ").split() if len(w) > 2}
    blurb = {w for w in blob.replace(",", " ").split() if len(w) > 2}
    if not words:
        return False
    overlap = len(words & blurb) / len(words)
    return overlap >= 0.6


def validate_spoken_line(
    text: str,
    *,
    goals: list[str] | None = None,
    personality: str = "",
    drive: str = "",
    role: str = "",
    emotional_state: str = "",
) -> tuple[bool, str]:
    """
    Compiler gate for speak-like verbs. Returns (ok, rejection_detail).
    """
    if not is_real_dialogue(text, goals=goals):
        return False, "spoken line must be a complete in-character sentence"
    if is_descriptor_not_utterance(text):
        return False, "spoken line must be dialogue, not a mood or personality label"
    if echoes_character_blurb(
        text,
        personality=personality,
        drive=drive,
        role=role,
        emotional_state=emotional_state,
    ):
        return False, "spoken line must not repeat your personality or drive text verbatim"
    return True, ""


def is_real_dialogue(text: str, *, goals: list[str] | None = None) -> bool:
    """
    Return True if ``text`` is suitable to show as spoken dialogue.

    Rejects planning strings, prompt fragments, and short non-sentences.
    """
    if not text or not text.strip():
        return False
    stripped = text.strip()
    if goals and matches_entity_goal_text(stripped, goals):
        return False
    if is_planning_text(stripped):
        return False
    if is_descriptor_not_utterance(stripped):
        return False
    lower = stripped.lower()
    if lower in _ENRICHMENT_PLACEHOLDERS:
        return False
    for frag in _PROMPT_FRAGMENT_SUBSTRINGS:
        if frag in lower:
            return False
    # Short fragment without sentence ending
    if len(stripped) < 40 and stripped[-1] not in ".!?…\"":
        words = lower.split()
        if len(words) <= 5 and words and words[0] in ("a", "an", "the", "in", "on", "at"):
            return False
        if len(words) <= 3:
            return False
    return True


def best_speech_line(action, *, goals: list[str] | None = None) -> str:
    """Pick the best candidate spoken line from a SemanticAction intent block."""
    if not action or not action.intent:
        return ""
    rationale = (action.intent.rationale or "").strip()
    manner = (action.intent.manner or "").strip()
    # Enriched dialogue lands in manner; rationale is often internal planning.
    if is_real_dialogue(manner, goals=goals):
        return manner
    if is_real_dialogue(rationale, goals=goals):
        return rationale
    return ""


def pick_fallback_voice_line(entity, world) -> str:
    """Author voice line when LM enrichment failed (autonomous / small models)."""
    import random as _rnd

    lines = list(getattr(entity, "voice_lines", None) or [])
    if not lines:
        return ""
    _rnd.seed(int(abs(hash(f"{entity.entity_id}:{world.tick}:voice"))) % 99991)
    for _ in range(len(lines)):
        pick = _rnd.choice(lines)
        if is_real_dialogue(pick):
            return pick.strip()
    return lines[0].strip() if lines else ""


def sanitize_dialogue_text(text: str, *, goals: list[str] | None = None) -> str:
    """Strip planning stubs before storing in event log or observation memory."""
    t = (text or "").strip()
    if not t or not is_real_dialogue(t, goals=goals):
        return ""
    return t


def normalize_speech_key(text: str) -> str:
    """Normalize spoken text for duplicate detection."""
    import re as _re
    return _re.sub(r"\s+", " ", text.lower().strip().strip("\"'"))


def collect_recent_room_dialogue(
    world,
    *,
    max_lines: int = 5,
    lookback_events: int = 12,
) -> list[str]:
    """
    Unique recent spoken lines from the event log (any actor), newest first.

    Used in NPC LM prompts as a do-not-repeat list.
    """
    from .schemas import EntityId, TransitionKind

    seen: set[str] = set()
    lines: list[str] = []
    for ev in reversed(world.event_log[-lookback_events:]):
        for t in ev.transitions:
            if t.kind != TransitionKind.DIALOGUE_SPOKEN:
                continue
            txt = (t.payload.get("text") or "").strip()
            if not txt or txt == "...":
                intent = ev.action.intent
                if intent:
                    txt = (intent.manner or "").strip()
            txt = sanitize_dialogue_text(txt)
            if not txt:
                continue
            key = normalize_speech_key(txt)
            if key in seen:
                continue
            seen.add(key)
            actor_id = t.payload.get("actor", "")
            ent = world.spatial.entities.get(EntityId(str(actor_id)))
            name = ent.name if ent else str(actor_id)[:8]
            lines.append(f'{name}: "{txt[:100]}"')
            if len(lines) >= max_lines:
                return lines
    return lines
