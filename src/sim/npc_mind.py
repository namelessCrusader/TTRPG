"""
NPC mind layer — interpret context, score actions, record expression.

Replaces scattered meta hacks (verb cooldowns, stimulus rolls, stim_boost,
reaction-verb injection) with one pipeline:

  interpret_mind(entity, world)  → MindState in entity.meta["mind"]
  score_action(...)              → contextual utility for a candidate
  pick_best_action(...)          → highest-scoring feasible action
  record_expression(...)         → update mind after an action is taken
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from .needs import critical_needs
from .observation_memory import get_actionable_stimulus, has_fresh_stimulus
from .schemas import ActionType, AlertnessLevel, EdgeKind, SemanticAction

if TYPE_CHECKING:
    from .schemas import EntityState, WorldState

# ── Verb classes (used for intent matching, not hardcoded policy branches) ──

SPEECH_VERBS: frozenset[str] = frozenset({
    "speak", "say", "tell", "ask", "shout", "whisper", "yell", "call",
    "announce", "mutter", "greet", "challenge", "warn", "threaten",
    "gossip", "accuse", "lie", "promise", "sing", "hum",
})

GESTURE_VERBS: frozenset[str] = frozenset({
    "nod", "wave", "toast", "laugh", "listen", "pat", "shrug", "wink",
    "smile", "frown", "glare", "scoff", "bow", "bow_to", "salute",
})

PASSIVE_VERBS: frozenset[str] = frozenset({"observe", "wait", "rest"})
STATE_CHANGE_VERBS: frozenset[str] = frozenset({
    "move", "flee", "attack", "take", "pickup", "drop", "give", "throw",
    "equip", "unequip", "wear", "use", "cast", "drink", "eat", "pour_drink",
    "open", "close", "lock", "unlock", "break", "repair",
})
SOCIAL_RISK_VERBS: frozenset[str] = frozenset({
    "accuse", "challenge", "warn", "threaten", "intimidate", "deceive",
    "persuade", "haggle", "gossip", "whisper", "lie", "promise",
})

# Intent kinds stored on MindState
INTENT_REPLY = "reply"
INTENT_ADVANCE_GOAL = "advance_goal"
INTENT_PURSUE_DRIVE = "pursue_drive"
INTENT_ACKNOWLEDGE = "acknowledge"
INTENT_SOCIALIZE = "socialize"
INTENT_THREAT = "threat"
INTENT_NEED = "need"
INTENT_OBSERVE = "observe"
INTENT_PATROL = "patrol"


@dataclass
class MindState:
    """Per-entity situational mind — serialized in entity.meta['mind']."""

    intent: str = INTENT_SOCIALIZE
    attention_target: str = ""
    attention_reason: str = ""
    tension_topics: list[str] = field(default_factory=list)
    # entity_id → tick when we last used a gesture toward them
    recent_gestures: dict[str, int] = field(default_factory=dict)
    last_verb: str = ""
    speak_ready: bool = True
    ticks_since_speak: int = 999

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "attention_target": self.attention_target,
            "attention_reason": self.attention_reason,
            "tension_topics": list(self.tension_topics),
            "recent_gestures": dict(self.recent_gestures),
            "last_verb": self.last_verb,
            "speak_ready": self.speak_ready,
            "ticks_since_speak": self.ticks_since_speak,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MindState":
        if not isinstance(data, dict):
            return cls()
        return cls(
            intent=str(data.get("intent") or INTENT_SOCIALIZE),
            attention_target=str(data.get("attention_target") or ""),
            attention_reason=str(data.get("attention_reason") or ""),
            tension_topics=list(data.get("tension_topics") or []),
            recent_gestures={
                str(k): int(v)
                for k, v in (data.get("recent_gestures") or {}).items()
            },
            last_verb=str(data.get("last_verb") or ""),
            speak_ready=bool(data.get("speak_ready", True)),
            ticks_since_speak=int(data.get("ticks_since_speak", 999)),
        )


def load_mind(entity: "EntityState") -> MindState:
    return MindState.from_dict(entity.meta.get("mind"))


def save_mind(entity: "EntityState", mind: MindState) -> None:
    entity.meta["mind"] = mind.to_dict()


def _scoring_config(world: "WorldState") -> dict[str, float]:
    cfg = world.config.npc_policy
    ms = getattr(cfg, "mind_scoring", None)
    if ms is None:
        return _DEFAULT_SCORING
    return {
        "intent_reply_speak": float(getattr(ms, "intent_reply_speak", 0.45)),
        "intent_goal_speak": float(getattr(ms, "intent_goal_speak", 0.38)),
        "intent_drive_speak": float(getattr(ms, "intent_drive_speak", 0.32)),
        "gesture_when_reply": float(getattr(ms, "gesture_when_reply", 0.40)),
        "gesture_repeat": float(getattr(ms, "gesture_repeat", 0.55)),
        "same_verb": float(getattr(ms, "same_verb", 0.30)),
        "speak_not_ready": float(getattr(ms, "speak_not_ready", 0.70)),
        "novelty": float(getattr(ms, "novelty", 0.08)),
        "tension_match": float(getattr(ms, "tension_match", 0.18)),
        "acknowledge_gesture": float(getattr(ms, "acknowledge_gesture", 0.28)),
    }


_DEFAULT_SCORING = {
    "intent_reply_speak": 0.45,
    "intent_goal_speak": 0.38,
    "intent_drive_speak": 0.32,
    "gesture_when_reply": 0.40,
    "gesture_repeat": 0.55,
    "same_verb": 0.30,
    "speak_not_ready": 0.70,
    "novelty": 0.08,
    "tension_match": 0.18,
    "acknowledge_gesture": 0.28,
}


def _recent_actor_verbs(
    entity: "EntityState",
    world: "WorldState",
    *,
    limit: int = 8,
) -> list[str]:
    verbs: list[str] = []
    eid = str(entity.entity_id)
    for ev in reversed(world.event_log):
        action = getattr(ev, "action", None)
        if action is None or str(getattr(action, "actor", "")) != eid:
            continue
        verbs.append(_verb_of(action))
        if len(verbs) >= limit:
            break
    return verbs


def _action_text(action: SemanticAction) -> str:
    chunks: list[str] = []
    chunks.append(str(action.verb))
    if action.raw_input:
        chunks.append(str(action.raw_input))
    intent = getattr(action, "intent", None)
    if intent is not None:
        for attr in ("rationale", "manner"):
            val = getattr(intent, attr, None)
            if val:
                chunks.append(str(val))
        for attr in ("desired_outcome",):
            val = getattr(intent, attr, None)
            if val:
                chunks.extend(str(v) for v in val)
    if action.target is not None:
        chunks.append(str(action.target))
    return " ".join(chunks).lower()


def _has_rich_intent(action: SemanticAction) -> bool:
    intent = getattr(action, "intent", None)
    if intent is None:
        return False
    for attr in ("rationale", "manner"):
        if getattr(intent, attr, None):
            return True
    outcomes = getattr(intent, "desired_outcome", None)
    return bool(outcomes)


def _keyword_overlap(text: str, sources: list[str]) -> int:
    if not text or not sources:
        return 0
    tokens = {
        tok.strip(".,:;!?()[]{}\"'").lower()
        for tok in text.split()
        if len(tok.strip(".,:;!?()[]{}\"'")) >= 4
    }
    if not tokens:
        return 0
    hits = 0
    for src in sources:
        src_l = str(src).lower()
        if any(tok in src_l for tok in tokens):
            hits += 1
    return hits


def _speak_gap_ticks(entity: "EntityState", world: "WorldState") -> int:
    cfg = world.config.npc_policy
    base = int(getattr(getattr(cfg, "mind_scoring", None), "min_speak_gap_ticks", 2))
    world_gap = int(world.meta.get("npc_min_speak_gap", 0))
    if world_gap > 0:
        return world_gap
    openness = entity.social_openness.lower()
    openness_map = {
        "welcoming": 1,
        "open": 2,
        "guarded": 3,
        "reserved": 4,
        "closed": 5,
    }
    return max(base, openness_map.get(openness, 3))


def _gesture_memory_ticks(world: "WorldState") -> int:
    cfg = world.config.npc_policy
    return int(getattr(getattr(cfg, "mind_scoring", None), "gesture_memory_ticks", 4))


def _director_tension_topics(world: "WorldState") -> list[str]:
    focus = str(world.meta.get("director_scene_focus") or "").lower()
    topics: list[str] = []
    if any(w in focus for w in ("secret", "gossip", "accuse", "eavesdrop")):
        topics.extend(["secrets", "gossip"])
    if any(w in focus for w in ("greet", "ale", "settle", "mood")):
        topics.append("opening")
    active = world.meta.get("active_pressures") or []
    if isinstance(active, list):
        for p in active:
            if isinstance(p, dict) and p.get("id") == "secrets_rising":
                topics.append("secrets")
            if isinstance(p, dict) and p.get("id") == "opening_mood":
                topics.append("opening")
    return list(dict.fromkeys(topics))


def interpret_mind(entity: "EntityState", world: "WorldState") -> MindState:
    """
    Derive intent and attention from reflection output, stimulus, needs, goals.
    """
    mind = load_mind(entity)
    eid = str(entity.entity_id)
    tick = world.tick

    last_speak = int(entity.meta.get("last_speak_tick", -999))
    mind.ticks_since_speak = tick - last_speak
    gap = _speak_gap_ticks(entity, world)
    mind.speak_ready = mind.ticks_since_speak >= gap
    mind.last_verb = str(entity.meta.get("last_action_verb", "")).lower()
    mind.tension_topics = _director_tension_topics(world)

    # Prune old gesture memory
    mem_ticks = _gesture_memory_ticks(world)
    mind.recent_gestures = {
        tid: t
        for tid, t in mind.recent_gestures.items()
        if tick - t <= mem_ticks
    }

    # Priority: threat / urgent need (from reflection focus_topic)
    focus = str(entity.meta.get("focus_topic") or "")
    if focus == "threat" or entity.meta.get("hostile_nearby"):
        mind.intent = INTENT_THREAT
        mind.attention_reason = "threat"
        save_mind(entity, mind)
        return mind

    urgent = critical_needs(entity)
    if urgent or focus.startswith("need_"):
        mind.intent = INTENT_NEED
        mind.attention_reason = urgent[0][0] if urgent else focus
        save_mind(entity, mind)
        return mind

    stim = get_actionable_stimulus(entity, world) if has_fresh_stimulus(entity, world) else None
    if stim is not None:
        kind = str(stim.get("kind", "action"))
        actor_id = str(stim.get("actor_id", ""))
        addressed = bool(stim.get("addressed"))
        stim_verb = str(stim.get("verb", "")).lower()

        if actor_id:
            mind.attention_target = actor_id

        if kind == "speech":
            text = (stim.get("text") or "").strip()
            if addressed and text:
                mind.intent = INTENT_REPLY
                mind.attention_reason = "addressed_speech"
            elif addressed:
                mind.intent = INTENT_REPLY
                mind.attention_reason = "addressed"
            else:
                mind.intent = INTENT_SOCIALIZE
                mind.attention_reason = "overheard_speech"
        elif kind == "violence":
            mind.intent = INTENT_THREAT
            mind.attention_reason = "violence"
        elif stim_verb in GESTURE_VERBS and actor_id:
            # Already acknowledged this person recently → pursue something richer
            last_g = mind.recent_gestures.get(actor_id, -999)
            if tick - last_g <= mem_ticks:
                if entity.goals:
                    mind.intent = INTENT_ADVANCE_GOAL
                elif entity.drive:
                    mind.intent = INTENT_PURSUE_DRIVE
                else:
                    mind.intent = INTENT_SOCIALIZE
                mind.attention_reason = "gesture_already_done"
            elif "opening" in mind.tension_topics and mind.ticks_since_speak >= gap:
                mind.intent = INTENT_ACKNOWLEDGE
                mind.attention_reason = "ritual_greeting"
            else:
                mind.intent = INTENT_ACKNOWLEDGE
                mind.attention_reason = "gesture_stimulus"
        elif addressed and actor_id:
            mind.intent = INTENT_REPLY
            mind.attention_reason = "addressed_action"
        else:
            mind.intent = INTENT_SOCIALIZE
            mind.attention_reason = "ambient_stimulus"
    elif focus == "reply" or entity.meta.get("should_reply_to_id"):
        mind.intent = INTENT_REPLY
        mind.attention_target = str(entity.meta.get("should_reply_to_id") or "")
        mind.attention_reason = "pending_reply"
    elif entity.waypoints and entity.alertness in (
        AlertnessLevel.UNAWARE,
        AlertnessLevel.LOW,
    ):
        mind.intent = INTENT_PATROL
        mind.attention_reason = "patrol_route"
    elif focus == "pursue_goal" and entity.goals:
        mind.intent = INTENT_ADVANCE_GOAL
        mind.attention_reason = "goal"
    elif focus == "pursue_drive" and entity.drive:
        mind.intent = INTENT_PURSUE_DRIVE
        mind.attention_reason = "drive"
    elif entity.goals:
        mind.intent = INTENT_ADVANCE_GOAL
        mind.attention_reason = "goal_idle"
    elif entity.drive:
        mind.intent = INTENT_PURSUE_DRIVE
        mind.attention_reason = "drive_idle"
    else:
        mind.intent = INTENT_SOCIALIZE
        mind.attention_reason = "room"

    save_mind(entity, mind)
    return mind


def _action_target_id(action: SemanticAction) -> str:
    t = action.target
    return str(t) if isinstance(t, str) else ""


def _verb_of(action: SemanticAction) -> str:
    return str(action.verb).lower().split(".")[-1]


def score_action(
    entity: "EntityState",
    world: "WorldState",
    action: SemanticAction,
    base_weight: float,
) -> float:
    """Contextual utility for one candidate action."""
    mind = load_mind(entity)
    cfg = _scoring_config(world)
    v = _verb_of(action)
    score = base_weight
    target = _action_target_id(action)
    tick = world.tick
    mem_ticks = _gesture_memory_ticks(world)
    text = _action_text(action)
    recent_verbs = _recent_actor_verbs(entity, world)

    if not mind.speak_ready and v in SPEECH_VERBS:
        score -= cfg["speak_not_ready"]

    # ── Intent alignment ─────────────────────────────────────────────
    if mind.intent == INTENT_REPLY:
        if v in SPEECH_VERBS and target == mind.attention_target:
            score += cfg["intent_reply_speak"]
        elif v in GESTURE_VERBS:
            score -= cfg["gesture_when_reply"]

    elif mind.intent == INTENT_ADVANCE_GOAL:
        if v in SPEECH_VERBS:
            score += cfg["intent_goal_speak"]
            if target and entity.goals:
                score += 0.06
        elif v in GESTURE_VERBS:
            score -= cfg["gesture_when_reply"] * 0.5

    elif mind.intent == INTENT_PURSUE_DRIVE:
        if v in SPEECH_VERBS:
            score += cfg["intent_drive_speak"]
        elif v in ("gossip", "eavesdrop", "accuse", "haggle"):
            score += cfg["tension_match"]

    elif mind.intent == INTENT_ACKNOWLEDGE:
        if v in GESTURE_VERBS and target == mind.attention_target:
            last_g = mind.recent_gestures.get(target, -999)
            if tick - last_g <= mem_ticks:
                score -= cfg["gesture_repeat"]
            else:
                score += cfg["acknowledge_gesture"]
        elif v in SPEECH_VERBS and mind.speak_ready:
            score += cfg["intent_drive_speak"] * 0.5

    elif mind.intent == INTENT_SOCIALIZE:
        if v in SPEECH_VERBS and mind.speak_ready:
            score += cfg["intent_drive_speak"] * 0.4
        if v in ("gossip", "eavesdrop", "perform", "joke", "tease"):
            score += cfg["tension_match"] * 0.5

    elif mind.intent == INTENT_NEED:
        if v in ("eat", "drink", "rest", "speak") or ActionType.REST == action.verb:
            score += 0.35

    elif mind.intent == INTENT_THREAT:
        if v in ("flee", "attack", "observe", "speak", "threaten"):
            score += 0.25

    elif mind.intent == INTENT_PATROL:
        if v == "move":
            score += 0.55
        elif v in SPEECH_VERBS or v in GESTURE_VERBS:
            score -= 0.35

    # ── Environmental hazards ────────────────────────────────────────
    from .hazard_avoidance import assess_hazards

    _haz = assess_hazards(entity, world)
    if _haz.should_flee and v == "flee":
        score += 0.9
    elif _haz.should_move_away and v in ("flee", "move"):
        score += 0.5

    # ── Director tension ─────────────────────────────────────────────
    topics = mind.tension_topics
    if "secrets" in topics and v in ("gossip", "eavesdrop", "accuse", "whisper", "speak"):
        score += cfg["tension_match"]
    if "opening" in topics and v in ("greet", "toast", "pour_drink") and mind.intent == INTENT_ACKNOWLEDGE:
        score += cfg["acknowledge_gesture"] * 0.5

    # ── Relationship toward target ───────────────────────────────────
    if target and target != str(entity.entity_id):
        edges = world.relational.edges.get(str(entity.entity_id), {}).get(target, [])
        for edge in edges:
            if edge.kind in (EdgeKind.ENEMY_OF, EdgeKind.WARY_OF):
                if v in ("glare", "accuse", "threaten", "attack"):
                    score += 0.12
            if edge.kind in (EdgeKind.ALLY_OF, EdgeKind.INTIMATE):
                if v in ("toast", "pat", "speak", "gossip"):
                    score += 0.08

    # ── Recency / novelty ────────────────────────────────────────────
    if v == mind.last_verb:
        score *= cfg["same_verb"]
    elif v not in PASSIVE_VERBS:
        score += cfg["novelty"]

    # ── Dramatic utility / interestingness ───────────────────────────
    # A tabletop-style scene should prefer actions that advance goals,
    # move state, take a risk, or reveal information over repeated no-ops.
    if v in STATE_CHANGE_VERBS:
        score += 0.22
    rich_intent = _has_rich_intent(action)
    if v in SPEECH_VERBS | SOCIAL_RISK_VERBS:
        score += 0.12
    if v in SOCIAL_RISK_VERBS:
        score += 0.05 if rich_intent else -0.30
    if target and target != str(entity.entity_id):
        score += 0.08

    # Goal/drive fit: overlap action language with authored goals/drives.
    goal_sources = [str(g) for g in (entity.goals or [])]
    if entity.drive:
        goal_sources.append(str(entity.drive))
    if entity.secrets and v in ("whisper", "deceive", "gossip", "accuse", "speak"):
        goal_sources.extend(str(s) for s in entity.secrets)
    overlap = _keyword_overlap(text, goal_sources)
    if overlap:
        score += min(0.30, 0.10 * overlap)

    # Repetition penalties from recent event history, not only last tick.
    if recent_verbs:
        repeat_count = sum(1 for rv in recent_verbs[:5] if rv == v)
        if repeat_count:
            score -= min(0.45, 0.14 * repeat_count)
        distinct_recent = len(set(recent_verbs[:4]))
        if v not in recent_verbs[:4] and v not in PASSIVE_VERBS:
            score += 0.08 + 0.02 * distinct_recent

    if v in PASSIVE_VERBS:
        has_reason = bool(target or "because" in text or "watch" in text or "listen" in text)
        if not has_reason:
            score -= 0.35
        if int(entity.meta.get("observe_streak", 0)) >= 1 or mind.last_verb in PASSIVE_VERBS:
            score -= 0.55

    # Discourage hammering the same social affordance repeatedly.
    for overused in ("eavesdrop", "perform", "wait", "joke"):
        if v == overused and mind.last_verb == overused:
            score -= cfg["gesture_repeat"] * 0.6

    if v in GESTURE_VERBS and target:
        last_g = mind.recent_gestures.get(target, -999)
        if tick - last_g <= mem_ticks:
            score -= cfg["gesture_repeat"]

    return max(0.02, score)


def _stable_roll(entity: "EntityState", world: "WorldState", bucket: str, n: int) -> int:
    from .compiler import _seeded_float

    seed = f"mind:{world.rng_seed}:{entity.entity_id}:{world.tick}:{bucket}"
    return int(_seeded_float(seed) * max(n, 1)) % max(n, 1)


def pick_best_action(
    entity: "EntityState",
    world: "WorldState",
    candidates: list[tuple[SemanticAction, str, float]],
    *,
    filter_passive_loops: bool = True,
) -> SemanticAction:
    """Pick the highest mind-scored candidate."""
    if not candidates:
        return SemanticAction(
            verb=ActionType.WAIT,
            actor=entity.entity_id,
            raw_input="[npc_auto:idle_fallback]",
        )

    pool = list(candidates)
    if filter_passive_loops:
        mind = load_mind(entity)
        passive = PASSIVE_VERBS
        if mind.last_verb in passive or int(entity.meta.get("observe_streak", 0)) >= 1:
            active = [c for c in pool if _verb_of(c[0]) not in passive]
            if active:
                pool = active

    scored = [
        (action, label, score_action(entity, world, action, weight))
        for action, label, weight in pool
    ]
    scored.sort(key=lambda c: c[2], reverse=True)
    if not scored:
        return SemanticAction(
            verb=ActionType.WAIT,
            actor=entity.entity_id,
            raw_input="[npc_auto:idle_fallback]",
        )

    top_score = scored[0][2]
    near = [c for c in scored if c[2] >= top_score - 0.05]
    if len(near) == 1:
        choice = near[0]
    else:
        idx = _stable_roll(entity, world, "pick", len(near))
        choice = near[idx]

    return choice[0]


def record_expression(
    entity: "EntityState",
    world: "WorldState",
    action: SemanticAction,
) -> None:
    """Update mind after an action is committed."""
    mind = load_mind(entity)
    v = _verb_of(action)
    target = _action_target_id(action)
    tick = world.tick

    mind.last_verb = v
    if v in SPEECH_VERBS:
        mind.ticks_since_speak = 0
        mind.speak_ready = False
    else:
        mind.ticks_since_speak = tick - int(entity.meta.get("last_speak_tick", tick))

    if v in GESTURE_VERBS and target:
        mind.recent_gestures[target] = tick

    save_mind(entity, mind)


def interestingness_breakdown(
    entity: "EntityState",
    world: "WorldState",
    action: SemanticAction,
    *,
    base_weight: float = 0.5,
) -> dict[str, float | str]:
    """Small debug/trace helper for why an action beat a passive alternative."""
    v = _verb_of(action)
    text = _action_text(action)
    goal_sources = [str(g) for g in (entity.goals or [])]
    if entity.drive:
        goal_sources.append(str(entity.drive))
    repeat_count = sum(1 for rv in _recent_actor_verbs(entity, world, limit=5) if rv == v)
    return {
        "verb": v,
        "score": round(score_action(entity, world, action, base_weight), 3),
        "state_change": 1.0 if v in STATE_CHANGE_VERBS else 0.0,
        "social_risk": 1.0 if v in SOCIAL_RISK_VERBS else 0.0,
        "goal_overlap": float(_keyword_overlap(text, goal_sources)),
        "recent_repeat_count": float(repeat_count),
        "passive": 1.0 if v in PASSIVE_VERBS else 0.0,
    }
