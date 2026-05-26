"""
Skill Learning System
=====================

Skills improve through use: every time an entity successfully performs an
action, the skill associated with that action increments slightly.  Over
many sessions an NPC bard becomes noticeably better at persuasion while
a guard who never fights stays mediocre in combat.

Standard skills (all 0–100):
  combat      — melee and ranged violence
  arcane      — casting, reading magical artifacts, spell recognition
  social      — persuasion, deception, performance, intimidation
  stealth     — sneaking, hiding, pickpocketing
  crafting    — combining materials, repairing items, cooking
  perception  — noticing hidden things, reading environments

Skills live in ``entity.skills`` (dict[str, float] added to EntityState).

This module provides:
  • ``SKILL_DEFAULTS``        — occupation-aware starting skill maps
  • ``verb_to_skill(verb)``   — which skill a verb trains
  • ``skill_bonus(entity, skill)`` — 0.0–0.3 contest bonus from skill level
  • ``maybe_gain_skill(...)`` — returns SKILL_INCREASED Transition or None
"""

from __future__ import annotations

import random as _random
from typing import Optional, TYPE_CHECKING

from .schemas import Transition, TransitionKind

if TYPE_CHECKING:
    from .schemas import EntityState

# ── Starting values by occupation ────────────────────────────────────────────

_OCC_SKILL_PRESETS: dict[str, dict[str, float]] = {
    "bartender":  {"social": 40.0, "perception": 30.0, "crafting": 15.0},
    "innkeeper":  {"social": 35.0, "perception": 25.0, "crafting": 20.0},
    "bard":       {"social": 55.0, "perception": 35.0, "stealth": 20.0},
    "merchant":   {"social": 45.0, "perception": 30.0, "crafting": 10.0},
    "veteran":    {"combat": 65.0, "perception": 40.0, "stealth": 25.0},
    "guard":      {"combat": 45.0, "perception": 35.0, "social": 15.0},
    "soldier":    {"combat": 50.0, "perception": 30.0},
    "thief":      {"stealth": 60.0, "perception": 45.0, "social": 25.0},
    "scholar":    {"arcane": 50.0, "perception": 40.0, "social": 20.0},
    "mage":       {"arcane": 65.0, "perception": 35.0},
    "herbalist":  {"crafting": 50.0, "perception": 40.0},
    "blacksmith": {"crafting": 60.0, "combat": 20.0},
    "assassin":   {"stealth": 65.0, "combat": 45.0, "perception": 35.0},
    "spy":        {"stealth": 55.0, "social": 50.0, "perception": 45.0},
}

_BASE_DEFAULTS: dict[str, float] = {
    "combat":     5.0,
    "arcane":     2.0,
    "social":     10.0,
    "stealth":    5.0,
    "crafting":   5.0,
    "perception": 10.0,
}

# ── Verb → skill mapping ──────────────────────────────────────────────────────

_VERB_SKILL: dict[str, str] = {
    # Combat
    "attack": "combat", "fight": "combat", "stab": "combat",
    "shoot": "combat", "parry": "combat", "block": "combat",
    "disarm": "combat",
    # Arcane
    "cast": "arcane", "channel": "arcane", "dispel": "arcane",
    "enchant": "arcane", "inscribe": "arcane",
    # Social
    "speak": "social", "persuade": "social", "intimidate": "social",
    "flirt": "social", "beg": "social", "negotiate": "social",
    "perform": "social", "sing": "social", "bribe": "social",
    "threaten": "social", "ask": "social", "greet": "social",
    "trade": "social",
    # Stealth
    "sneak": "stealth", "hide": "stealth", "steal": "stealth",
    "pickpocket": "stealth", "shadow": "stealth",
    # Crafting
    "craft": "crafting", "brew": "crafting", "cook": "crafting",
    "repair": "crafting", "combine": "crafting", "mix": "crafting",
    # Perception
    "observe": "perception", "examine": "perception", "inspect": "perception",
    "listen": "perception", "search": "perception", "read": "perception",
}

# ── Tuning ────────────────────────────────────────────────────────────────────

_GAIN_PER_SUCCESS: float = 0.4          # flat gain on any successful use
_GAIN_VARIANCE: float = 0.2             # ±random variance
_SKILL_MAX: float = 100.0
_SKILL_GAIN_DECAY: float = 0.92         # diminishing returns near cap


def skill_defaults(occupation: str = "") -> dict[str, float]:
    """Return a full skill dict for the given occupation."""
    defaults = dict(_BASE_DEFAULTS)
    presets = _OCC_SKILL_PRESETS.get(occupation.lower(), {})
    defaults.update(presets)
    return defaults


def verb_to_skill(verb: str) -> Optional[str]:
    """Return the skill trained by this verb, or None if untrained."""
    return _VERB_SKILL.get(verb.lower())


def skill_bonus(entity: "EntityState", skill: str) -> float:
    """
    Contest bonus from skill level: 0.0 at skill=0, up to +0.30 at skill=100.
    Applied as a flat probability shift in ContestSpec resolution.
    """
    val = entity.skills.get(skill, _BASE_DEFAULTS.get(skill, 5.0))
    return round(val / 100.0 * 0.30, 4)


def maybe_gain_skill(
    entity: "EntityState",
    verb: str,
    success: bool,
    *,
    rng_seed: int = 0,
) -> Optional[Transition]:
    """
    If the entity successfully performed a skill-relevant action, return a
    SKILL_INCREASED Transition.  Returns None if the verb trains no skill,
    or a random roll suppresses gain (prevents grinding to 100 trivially).
    """
    if not success:
        return None
    skill = verb_to_skill(verb)
    if skill is None:
        return None
    current = entity.skills.get(skill, _BASE_DEFAULTS.get(skill, 5.0))
    if current >= _SKILL_MAX:
        return None
    # Diminishing returns — near cap, most rolls produce nothing
    room = (_SKILL_MAX - current) / _SKILL_MAX
    rng = _random.Random(rng_seed)
    if rng.random() > room * _SKILL_GAIN_DECAY + 0.1:
        return None
    raw_gain = _GAIN_PER_SUCCESS + rng.uniform(-_GAIN_VARIANCE, _GAIN_VARIANCE)
    gain = round(min(raw_gain, _SKILL_MAX - current), 3)
    if gain <= 0:
        return None
    new_val = round(current + gain, 2)
    return Transition(
        kind=TransitionKind.SKILL_INCREASED,
        payload={
            "entity_id": str(entity.entity_id),
            "skill": skill,
            "delta": gain,
            "new_value": new_val,
            "cause": f"successful_{verb}",
        },
    )


__all__ = [
    "skill_defaults",
    "verb_to_skill",
    "skill_bonus",
    "maybe_gain_skill",
    "SKILL_INCREASED",
]

# Re-export for convenience
SKILL_INCREASED = TransitionKind.SKILL_INCREASED
