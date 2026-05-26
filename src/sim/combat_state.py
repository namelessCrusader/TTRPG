"""
Combat alertness and pursuit state — deterministic post-violence bookkeeping.

Runs once per world tick (via ``world_clock``). Does not replace combat
resolution in the compiler; it keeps alertness / pursuit meta coherent
across ticks so reactive and LM policies see a stable combat picture.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .schemas import AlertnessLevel, EntityId, Transition, TransitionKind

if TYPE_CHECKING:
    from .schemas import WorldState

_HIDDEN_DECAY_TICKS = 8


def stealth_tick(world: "WorldState") -> list[Transition]:
    """Decay hidden status when the hider stays still too long or is exposed."""
    transitions: list[Transition] = []
    for eid, entity in world.all_entities().items():
        if not entity.alive or not entity.meta.get("hidden"):
            entity.meta.pop("hidden_ticks", None)
            continue
        streak = int(entity.meta.get("hidden_ticks", 0)) + 1
        entity.meta["hidden_ticks"] = streak
        if streak >= _HIDDEN_DECAY_TICKS:
            eid_s = str(eid)
            entity.meta.pop("hidden", None)
            entity.meta.pop("hidden_ticks", None)
            transitions.append(
                Transition(
                    kind=TransitionKind.ENTITY_PROPERTY_CHANGED,
                    payload={
                        "entity_id": eid_s,
                        "prop": "hidden",
                        "new_value": None,
                    },
                )
            )
    return transitions

_VIOLENCE_VERBS = frozenset({"attack", "fight", "threaten", "intimidate"})
_COMBAT_DECAY_TICKS = 6
_LOOKBACK = 4


def _violence_near_entity(world: "WorldState", eid: str, tick: int) -> bool:
    cutoff = tick - _LOOKBACK
    for ev in reversed(world.event_log):
        if ev.tick < cutoff:
            break
        verb = str(ev.action.verb).lower().split(".")[-1]
        if verb not in _VIOLENCE_VERBS:
            if not any(
                t.kind == TransitionKind.ENTITY_HEALTH_CHANGED
                and t.payload.get("cause") in ("combat", "attack")
                for t in ev.transitions
            ):
                continue
        actor = str(ev.action.actor)
        target = str(ev.action.target or "")
        if eid in (actor, target):
            return True
        if eid in (str(w) for w in (ev.witnesses or [])):
            return True
    return False


def combat_tick(world: "WorldState") -> list[Transition]:
    """
    Decay stale COMBAT alertness and refresh ``combat_target`` from recent log.
    """
    transitions: list[Transition] = []
    tick = world.tick

    for eid, entity in world.all_entities().items():
        if not entity.alive:
            entity.meta.pop("combat_target", None)
            entity.meta.pop("combat_ticks", None)
            continue

        eid_s = str(eid)
        violent = _violence_near_entity(world, eid_s, tick)

        if violent:
            entity.meta["combat_ticks"] = 0
            # Track who they were fighting
            for ev in reversed(world.event_log[-_LOOKBACK * 3:]):
                if ev.tick < tick - _LOOKBACK:
                    break
                verb = str(ev.action.verb).lower().split(".")[-1]
                if verb != "attack" and not any(
                    t.kind == TransitionKind.ENTITY_HEALTH_CHANGED
                    for t in ev.transitions
                ):
                    continue
                actor = str(ev.action.actor)
                target = str(ev.action.target or "")
                if eid_s == actor and target:
                    entity.meta["combat_target"] = target
                    if entity.alertness != AlertnessLevel.COMBAT:
                        entity.alertness = AlertnessLevel.COMBAT
                    break
                if eid_s == target and actor:
                    entity.meta["combat_target"] = actor
                    if entity.alertness != AlertnessLevel.COMBAT:
                        entity.alertness = AlertnessLevel.COMBAT
                    break
            continue

        if entity.alertness == AlertnessLevel.COMBAT:
            streak = int(entity.meta.get("combat_ticks", 0)) + 1
            entity.meta["combat_ticks"] = streak
            if streak >= _COMBAT_DECAY_TICKS:
                entity.alertness = AlertnessLevel.HIGH
                entity.meta.pop("combat_ticks", None)
                entity.meta.pop("combat_target", None)
                transitions.append(
                    Transition(
                        kind=TransitionKind.ENTITY_ALERTNESS_CHANGED,
                        payload={
                            "entity_id": eid_s,
                            "from": AlertnessLevel.COMBAT.value,
                            "to": AlertnessLevel.HIGH.value,
                            "cause": "combat_decay",
                        },
                    )
                )
        else:
            entity.meta.pop("combat_ticks", None)

    from .combat_tactics import refresh_combat_initiative

    refresh_combat_initiative(world)
    return transitions
