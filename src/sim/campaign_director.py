"""
Campaign Director — condition-triggered reactive world beats.

Unlike tick-scheduled scenario beats (``scenario_executor``), reactive
triggers fire when pack-authored **conditions** become true: a violence
fact appears, an NPC dies, a bribe is registered, etc.

Triggers are declared in ``worlds/<pack>/reactive_triggers.yaml`` and
loaded into ``world.config.reactive_triggers``.  Each trigger fires at
most once by default (``once: true``) and records its id in
``world.meta["campaign_director_fired"]`` for replay-safe idempotency.

All mutations reuse the same deterministic effect bundle as scenario
beats — goals, ambient events, edge nudges, scheduled ripples — so
emergent story stays auditable in the event log.
"""

from __future__ import annotations

import logging
from typing import Any

from .pressure_eval import conditions_met
from .scenario_executor import apply_effect_bundle, log_scripted_event
from .schemas import WorldState

logger = logging.getLogger(__name__)

_FIRED_META_KEY = "campaign_director_fired"


def _fired_ids(world: WorldState) -> set[str]:
    raw = world.meta.get(_FIRED_META_KEY) or []
    if isinstance(raw, list):
        return {str(x) for x in raw}
    return set()


def _mark_fired(world: WorldState, trigger_id: str) -> None:
    fired = _fired_ids(world)
    fired.add(trigger_id)
    world.meta[_FIRED_META_KEY] = sorted(fired)


def evaluate_reactive_triggers(world: WorldState) -> list[str]:
    """
    Evaluate condition-fired triggers for the current tick.

    Call after propagation so newly registered ``world_facts`` and
    transition effects are visible to condition checks.

    Returns log lines for REPL / autonomous output.
    """
    triggers = world.config.reactive_triggers or []
    if not triggers:
        return []

    log_lines: list[str] = []
    fired = _fired_ids(world)

    for trigger in triggers:
        tid = str(trigger.get("id", "")).strip()
        if not tid:
            continue

        once = trigger.get("once", True)
        if once and tid in fired:
            continue

        when = trigger.get("when") or trigger.get("conditions") or []
        if isinstance(when, dict):
            when = [when]
        if not when:
            continue
        if not conditions_met(when, world):
            continue

        effects = trigger.get("then") or trigger.get("effects") or {}
        if not isinstance(effects, dict):
            continue

        bundle_lines, transitions = apply_effect_bundle(
            world,
            effects,
            cause="campaign_trigger",
            log_prefix=f"[campaign] {tid}",
        )
        log_lines.append(f"[campaign] trigger fired: {tid}")
        log_lines.extend(bundle_lines)

        if transitions:
            log_scripted_event(
                world,
                transitions,
                source=f"campaign:{tid}",
                narrative_hint=f"[campaign trigger {tid}]",
            )

        if once:
            _mark_fired(world, tid)

        title = trigger.get("title") or trigger.get("note")
        if title:
            log_lines.append(f"  → {title}")

        logger.info("Campaign trigger fired: %s (tick=%d)", tid, world.tick)

    return log_lines


def run_campaign_director(world: WorldState) -> list[str]:
    """
    Evaluate reactive triggers and conditional scenario beats.

    Single entry point for ``game_loop`` after propagation passes.
    """
    from .scenario_executor import apply_conditional_scenario_beats

    lines = evaluate_reactive_triggers(world)
    lines.extend(apply_conditional_scenario_beats(world))
    return lines


__all__ = ["evaluate_reactive_triggers", "run_campaign_director", "_FIRED_META_KEY"]
