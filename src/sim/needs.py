"""
Needs & Drives System
=====================

Every entity has four numeric needs that tick down each turn.  When a need
drops below a critical threshold it overrides normal occupation-driven
behaviour and the entity actively works to satisfy it.

  hunger       — need for food (bartenders/innkeepers decay slower)
  thirst       — need for drinkable liquids (see fluids.py / drink verb)
  bladder      — pressure from consumed liquids (bodily.py)
  fatigue      — need for rest (running and fighting drain this faster)
  social_need  — need for meaningful interaction (isolation makes NPCs restless)
  purpose      — need for meaningful activity (idle NPCs become erratic)

Needs live in ``entity.stats`` (already an arbitrary float dict) so no schema
change is needed for storage.  This module provides:

  • ``needs_defaults``        — occupation-aware starting values
  • ``needs_tick(world)``     — called once per world_tick(); returns Transitions
  • ``satisfy_need(...)``     — returns a single NEED_CHANGED Transition
  • ``critical_needs(entity)`` — returns sorted list of (need, value) below threshold
  • ``need_urgency(entity)``  — single float 0-1 measuring how desperate the NPC is

The ``ReactivePolicy`` imports ``critical_needs`` to inject urgent need-driven
candidates before occupation-specific ones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .schemas import Transition, TransitionKind

if TYPE_CHECKING:
    from .schemas import EntityState, WorldState

# ── Decay rates (per tick) ────────────────────────────────────────────────────
# Occupations that satisfy a need passively lower its decay rate.

_BASE_DECAY: dict[str, float] = {
    "hunger":      2.5,
    "thirst":      1.8,
    "fatigue":     1.0,
    "social_need": 1.5,
    "purpose":     0.8,
    "bladder":     0.0,  # rises from drinking, not passive decay
}

# Occupations that passively reduce decay for specific needs
_OCCUPATION_MODIFIERS: dict[str, dict[str, float]] = {
    "bartender":  {"hunger": -1.5, "social_need": -0.8},  # always near food/people
    "innkeeper":  {"hunger": -1.5, "social_need": -0.5},
    "bard":       {"social_need": -1.0, "purpose": -0.4},
    "merchant":   {"purpose": -0.3},
    "veteran":    {"fatigue": -0.3},
    "guard":      {"purpose": -0.5, "fatigue": -0.2},
    "soldier":    {"purpose": -0.5, "fatigue": -0.2},
    "scholar":    {"social_need": 0.3, "purpose": -0.6},  # scholars are lonely
}

# ── Critical thresholds — below these the need dominates behaviour ────────────
CRITICAL: dict[str, float] = {
    "hunger":      30.0,
    "fatigue":     20.0,
    "social_need": 25.0,
    "purpose":     20.0,
}

# ── Starting values ───────────────────────────────────────────────────────────
_DEFAULTS: dict[str, float] = {
    "hunger":      75.0,
    "thirst":      70.0,
    "fatigue":     85.0,
    "social_need": 60.0,
    "purpose":     70.0,
    "bladder":     10.0,
}

# ── Satisfaction amounts per action verb ──────────────────────────────────────
VERB_SATISFY: dict[str, dict[str, float]] = {
    "eat":         {"hunger": +35.0},
    "drink":       {},  # thirst/bladder handled in fluid_compiler
    "rest":        {"fatigue": +40.0, "purpose": +5.0},
    "sleep":       {"fatigue": +70.0},
    "speak":       {"social_need": +8.0},
    "ask":         {"social_need": +5.0},
    "greet":       {"social_need": +4.0},
    "perform":     {"social_need": +12.0, "purpose": +15.0},
    "craft":       {"purpose": +20.0},
    "examine":     {"purpose": +5.0},
    "patrol":      {"purpose": +8.0},
    "trade":       {"purpose": +12.0, "hunger": +5.0},
}

# ── Need → narrative label for goal text ─────────────────────────────────────
NEED_LABELS: dict[str, str] = {
    "hunger":      "find something to eat",
    "thirst":      "find something to drink",
    "bladder":     "find a place to relieve yourself",
    "fatigue":     "find a place to rest",
    "social_need": "talk to someone — anyone",
    "purpose":     "find something meaningful to do",
}


def needs_defaults(occupation: str = "") -> dict[str, float]:
    """Return sensible starting need values for the given occupation."""
    from .bodily import bodily_defaults

    vals = {**bodily_defaults(), **_DEFAULTS}
    mod = _OCCUPATION_MODIFIERS.get(occupation.lower(), {})
    for need, adj in mod.items():
        # Better-adapted occupations start with higher values for relevant needs
        vals[need] = min(100.0, vals.get(need, 75.0) + adj * 10.0)
    return vals


def _decay_rate(need: str, occ: str) -> float:
    base = _BASE_DECAY.get(need, 1.0)
    adj = _OCCUPATION_MODIFIERS.get(occ.lower(), {}).get(need, 0.0)
    return max(0.0, base + adj)


def needs_tick(world: "WorldState") -> list[Transition]:
    """
    Decrement all entity needs by one tick's worth of decay.
    Returns NEED_CHANGED Transitions (applied by the caller via apply_transitions).
    """
    transitions: list[Transition] = []
    for entity in world.spatial.entities.values():
        if not entity.alive:
            continue
        occ = (entity.occupation or "").lower()
        for need in ("hunger", "thirst", "fatigue", "social_need", "purpose"):
            current = entity.stats.get(need)
            if current is None:
                # Initialise from defaults if not yet set
                entity.stats[need] = needs_defaults(occ).get(need, _DEFAULTS[need])
                current = entity.stats[need]
            rate = _decay_rate(need, occ)
            if rate > 0:
                transitions.append(Transition(
                    kind=TransitionKind.NEED_CHANGED,
                    payload={
                        "entity_id": str(entity.entity_id),
                        "need": need,
                        "delta": -rate,
                        "cause": "time_passing",
                    },
                ))
    return transitions


def satisfy_need(entity_id: str, need: str, amount: float, cause: str = "action") -> Transition:
    """Return a single NEED_CHANGED transition that restores a need."""
    return Transition(
        kind=TransitionKind.NEED_CHANGED,
        payload={
            "entity_id": entity_id,
            "need": need,
            "delta": +amount,
            "cause": cause,
        },
    )


def critical_needs(entity: "EntityState") -> list[tuple[str, float]]:
    """
    Return (need, current_value) pairs below the critical threshold,
    sorted most-urgent first.
    """
    result = []
    for need, threshold in CRITICAL.items():
        val = entity.stats.get(need, _DEFAULTS.get(need, 50.0))
        if val < threshold:
            result.append((need, val))
    result.sort(key=lambda x: x[1])  # lowest value = most urgent
    return result


def need_urgency(entity: "EntityState") -> float:
    """
    Single float 0–1 indicating how desperate the entity is overall.
    0 = all needs met; 1 = multiple needs in crisis.
    """
    total_deficit = 0.0
    for need, threshold in CRITICAL.items():
        val = entity.stats.get(need, _DEFAULTS.get(need, 50.0))
        if val < threshold:
            total_deficit += (threshold - val) / threshold
    return min(1.0, total_deficit / len(CRITICAL))


__all__ = [
    "needs_defaults",
    "needs_tick",
    "satisfy_need",
    "critical_needs",
    "need_urgency",
    "VERB_SATISFY",
    "NEED_LABELS",
    "CRITICAL",
]
