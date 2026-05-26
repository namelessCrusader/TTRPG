"""
Adjudication index resolver (Overhaul D).

Extracts coarse slots from player intent and matches pack-authored
``adjudication_index.yaml`` entries before LM adjudication.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import Optional

from .schemas import (
    AdjudicationIndexEntry,
    AdjudicationResult,
    GroundingResult,
    SemanticAction,
    TransitionProposal,
    ValidationResult,
    WorldFact,
    WorldFactScope,
    WorldState,
)

_ACTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("barricade", re.compile(r"\b(wedge|jam|barricade|block|prop|brace)\b", re.I)),
    ("unlock", re.compile(r"\b(unlock|pick|force open|pry open|break open)\b", re.I)),
    ("ignite", re.compile(r"\b(ignite|light|set fire|torch|burn|kindle|set alight)\b", re.I)),
    ("extinguish", re.compile(r"\b(extinguish|douse|put out|smother)\b", re.I)),
    ("spill", re.compile(r"\b(spill|pour|tip|empty|slosh)\b", re.I)),
    ("hide", re.compile(r"\b(hide|conceal|duck behind|take cover|crouch behind)\b", re.I)),
    ("distract", re.compile(r"\b(distract|divert attention|create a diversion|decoy|lure|bait)\b", re.I)),
    ("trap", re.compile(r"\b(trap|snare|tripwire|rig|sabotage|tamper|disable)\b", re.I)),
    ("climb", re.compile(r"\b(climb|scale|scramble up|clamber)\b", re.I)),
    ("throw", re.compile(r"\b(throw|hurl|fling|toss)\b", re.I)),
    ("eavesdrop", re.compile(r"\b(eavesdrop|listen in|overhear|spy on)\b", re.I)),
    ("disarm", re.compile(r"\b(disarm|knock weapon|strip weapon|grab weapon)\b", re.I)),
    ("craft", re.compile(r"\b(improvise|fashion|make|cobble)\b.{0,20}\b(weapon|tool|bomb|molotov)\b", re.I)),
    ("mark", re.compile(r"\b(carve|scratch|write|mark|graffiti|draw|paint)\b", re.I)),
]


@dataclass
class AdjudicationSlots:
    text: str
    lower: str
    action_types: list[str]
    substances: list[str]
    confidence: float


def extract_adjudication_slots(text: str) -> AdjudicationSlots:
    lower = text.lower().strip()
    action_types: list[str] = []
    for name, pat in _ACTION_PATTERNS:
        if pat.search(lower):
            action_types.append(name)
    substances = [
        w for w in ("oil", "ale", "water", "fire", "door", "chair", "weapon", "liquid")
        if w in lower
    ]
    confidence = min(1.0, 0.35 + 0.15 * len(action_types) + 0.05 * len(substances))
    return AdjudicationSlots(
        text=text,
        lower=lower,
        action_types=action_types,
        substances=substances,
        confidence=confidence,
    )


def _pattern_matches(pattern: str, text: str) -> bool:
    p = pattern.lower().strip()
    if "*" in p or "?" in p:
        return fnmatch.fnmatch(text.lower(), p)
    return p in text.lower()


def _score_entry(entry: AdjudicationIndexEntry, slots: AdjudicationSlots) -> float:
    if slots.confidence < entry.min_confidence:
        return -1.0
    score = float(entry.priority)
    if entry.patterns:
        if not any(_pattern_matches(p, slots.text) for p in entry.patterns):
            return -1.0
        score += 2.0
    if entry.action_types:
        if not any(at in slots.action_types for at in entry.action_types):
            return -1.0
        score += 1.5
    if entry.substance_keywords:
        if not any(kw in slots.lower for kw in entry.substance_keywords):
            return -1.0
        score += 0.5
    if not entry.patterns and not entry.action_types:
        return -1.0
    return score + slots.confidence


def match_adjudication_index(
    world: WorldState,
    intent: str,
) -> Optional[AdjudicationIndexEntry]:
    if not world.config.adjudication_index:
        return None
    slots = extract_adjudication_slots(intent)
    best: Optional[tuple[float, AdjudicationIndexEntry]] = None
    for entry in world.config.adjudication_index:
        score = _score_entry(entry, slots)
        if score < 0:
            continue
        if best is None or score > best[0]:
            best = (score, entry)
    return best[1] if best else None


def _substitute_payloads(
    effects: list[TransitionProposal],
    *,
    actor_id: str,
    target_id: Optional[str],
    x: int,
    y: int,
) -> list[TransitionProposal]:
    out: list[TransitionProposal] = []
    for eff in effects:
        payload = dict(eff.payload or {})
        for key, val in list(payload.items()):
            if val == "$actor":
                payload[key] = actor_id
            elif val == "$target" and target_id:
                payload[key] = target_id
            elif val == "$x":
                payload[key] = x
            elif val == "$y":
                payload[key] = y
        out.append(TransitionProposal(kind=eff.kind, payload=payload))
    return out


def resolve_from_index(
    world: WorldState,
    action: SemanticAction,
    intent: str,
    grounding: GroundingResult,
    result: ValidationResult,
) -> Optional[AdjudicationResult]:
    """Build AdjudicationResult from a matched index entry, or None."""
    entry = match_adjudication_index(world, intent)
    if entry is None:
        return None

    from .region_utils import entity_or_none

    actor = entity_or_none(world, action.actor)
    if actor is None:
        return None

    actor_id = str(action.actor)
    target_id = str(action.target) if isinstance(action.target, str) else None
    x, y = actor.position.x, actor.position.y

    adj = AdjudicationResult(
        ruling_text=entry.ruling or f"The world accepts: {entry.id.replace('_', ' ')}.",
        synthesized_verb=entry.synthesized_verb,
    )
    adj.transition_proposals = _substitute_payloads(
        list(entry.effects),
        actor_id=actor_id,
        target_id=target_id,
        x=x,
        y=y,
    )
    adj.facts.append(
        WorldFact(
            claim=intent[:160],
            scope=WorldFactScope.TILE,
            subject_id=f"{x},{y}",
            established_tick=world.tick,
            established_by=actor_id,
            source_intent=intent[:200],
            tags=["adjudicated", "index", *entry.fact_tags],
        )
    )
    return adj
