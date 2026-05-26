"""
Deterministic scenario beat executor.

Scenario YAML beats can declare one-shot ``effects`` that fire when the
simulation enters a new beat (tick threshold crossed).  This gives the
NarrativeDirector teeth beyond LM hints: forced ambient events, injected
goals, edge nudges, and world meta — all replay-safe and pack-authored.

Persistent beat guidance (focus, tension, phase) is still handled by
``NarrativeDirector.brief_for_npc``; this module handles *actions*.
"""

from __future__ import annotations

from typing import Any, Optional

from .compiler import apply_transitions
from .pressure_eval import _apply_edge_nudge, _resolve_entity_ref
from .schemas import (
    ActionType,
    Coord,
    Event,
    ObjectId,
    ScheduledEffect,
    ScheduledEffectKind,
    SemanticAction,
    Transition,
    TransitionKind,
    TransitionProposal,
    WorldFact,
    WorldFactScope,
    WorldState,
    new_event_id,
)
from .world_clock import SYSTEM_ACTOR, force_ambient_by_id

_BEAT_KEY_META = "scenario_last_beat_key"


def _current_beat(beats: list[dict], tick: int) -> dict | None:
    active: dict | None = None
    for beat in beats:
        if int(beat.get("tick", 0)) <= tick:
            active = beat
        else:
            break
    return active


def _beat_key(beat: dict) -> str:
    return f"{int(beat.get('tick', 0))}:{int(beat.get('phase', 0))}"


def apply_scenario_entry_effects(world: WorldState) -> list[str]:
    """
    Apply one-shot effects when the active scenario beat changes.

    Returns log lines suitable for autonomous / debug output.
    """
    scenario = (world.config.extra or {}).get("scenario") or {}
    beats: list[dict] = list(scenario.get("beats") or [])
    if not beats:
        return []

    beat = _current_beat(beats, world.tick)
    if beat is None:
        return []

    key = _beat_key(beat)
    if world.meta.get(_BEAT_KEY_META) == key:
        return []

    world.meta[_BEAT_KEY_META] = key
    world.meta["scenario_phase"] = int(
        beat.get("phase", world.meta.get("scenario_phase", 0))
    )

    effects = beat.get("effects") or {}
    if not isinstance(effects, dict):
        return []

    log_lines: list[str] = [
        f"[scenario] beat entered: tick>={beat.get('tick')} phase={beat.get('phase')}"
    ]

    title = beat.get("title") or scenario.get("title")
    if title:
        log_lines[0] += f" — {title}"

    bundle_lines, transitions = apply_effect_bundle(
        world,
        effects,
        cause="scenario_beat",
        log_prefix="[scenario]",
    )
    log_lines.extend(bundle_lines)

    if transitions:
        log_scripted_event(
            world,
            transitions,
            source=f"scenario:{key}",
            narrative_hint=f"[scenario beat {key}]",
        )

    return log_lines


def apply_effect_bundle(
    world: WorldState,
    effects: dict[str, Any],
    *,
    cause: str,
    log_prefix: str = "[script]",
) -> tuple[list[str], list[Transition]]:
    """
    Apply a shared effects dict (scenario beat or campaign trigger).

    Supported keys mirror scenario YAML ``effects`` blocks plus
    ``schedule`` for delayed narration ripples.
    """
    from .world_adjudicator import scheduled_effect_transition

    log_lines: list[str] = []
    transitions: list[Transition] = []

    for meta_key, meta_val in (effects.get("set_world_meta") or {}).items():
        world.meta[str(meta_key)] = meta_val
        log_lines.append(f"{log_prefix} meta[{meta_key}] = {meta_val!r}")

    if effects.get("director_focus"):
        world.meta["director_scene_focus"] = str(effects["director_focus"])
        log_lines.append(f"{log_prefix} focus: {effects['director_focus']}")
    if effects.get("director_tension") is not None:
        world.meta["director_tension_override"] = float(effects["director_tension"])
        log_lines.append(f"{log_prefix} tension → {effects['director_tension']}")

    ambient_ids = effects.get("force_ambient")
    if isinstance(ambient_ids, str):
        ambient_ids = [ambient_ids]
    for amb_id in ambient_ids or []:
        ev = force_ambient_by_id(world, str(amb_id))
        if ev is not None:
            narrative = ""
            for t in ev.transitions:
                if t.kind == TransitionKind.AMBIENT_EVENT:
                    narrative = str(t.payload.get("narrative", ""))
                    break
            log_lines.append(f"{log_prefix} ambient forced: {amb_id}")
            if narrative:
                log_lines.append(f"  [WORLD] {narrative}")

    for spec in effects.get("inject_goals") or []:
        t = _goal_transition(world, spec, cause=cause)
        if t is not None:
            transitions.append(t)
            log_lines.append(
                f"{log_prefix} goal → {_resolve_entity_ref(world, spec.get('entity'))}: "
                f"{spec.get('goal', '')[:60]}"
            )

    for nudge in effects.get("edge_nudge") or []:
        _apply_edge_nudge(world, nudge)
        log_lines.append(
            f"{log_prefix} edge nudge: {nudge.get('source')}→{nudge.get('target')} "
            f"({nudge.get('edge_kind', 'interacted')})"
        )

    transitions.extend(_effect_field_deposits(world, effects, cause=cause, log_prefix=log_prefix, log_lines=log_lines))
    transitions.extend(_effect_tile_modifies(world, effects, cause=cause, log_prefix=log_prefix, log_lines=log_lines))
    transitions.extend(_effect_object_modifies(world, effects, cause=cause, log_prefix=log_prefix, log_lines=log_lines))
    transitions.extend(_effect_register_facts(world, effects, cause=cause, log_prefix=log_prefix, log_lines=log_lines))

    for sched in effects.get("schedule") or []:
        if not isinstance(sched, dict):
            continue
        offset = int(sched.get("tick_offset", sched.get("delay", 2)))
        narration = str(
            sched.get("text") or sched.get("narration") or ""
        ).strip()
        if narration:
            transitions.append(
                scheduled_effect_transition(
                    ScheduledEffect(
                        fire_tick=world.tick + offset,
                        created_tick=world.tick,
                        source_actor="system",
                        kind=ScheduledEffectKind.NARRATION,
                        narration=narration[:240],
                        rationale=f"{cause} scheduled ripple",
                    ),
                ),
            )
            log_lines.append(
                f"{log_prefix} scheduled narration @ tick+{offset}: {narration[:50]}"
            )

        sched_goals = sched.get("inject_goals") or []
        if sched_goals:
            proposals = []
            for spec in sched_goals:
                t = _goal_transition(world, spec, cause=f"{cause}_delayed")
                if t is None:
                    continue
                proposals.append(
                    TransitionProposal(
                        kind=t.kind.value,
                        payload=dict(t.payload),
                    )
                )
            if proposals:
                transitions.append(
                    scheduled_effect_transition(
                        ScheduledEffect(
                            fire_tick=world.tick + offset,
                            created_tick=world.tick,
                            source_actor="system",
                            kind=ScheduledEffectKind.TRANSITIONS,
                            transitions=proposals,
                            rationale=f"{cause} scheduled goals",
                        ),
                    ),
                )
                log_lines.append(
                    f"{log_prefix} scheduled {len(proposals)} goal(s) @ tick+{offset}"
                )

    if transitions:
        apply_transitions(world, transitions)

    return log_lines, transitions


def log_scripted_event(
    world: WorldState,
    transitions: list[Transition],
    *,
    source: str,
    narrative_hint: str,
) -> None:
    """Record scenario/campaign-driven mutations in the canonical event log."""
    if not transitions:
        return
    world.event_log.append(
        Event(
            event_id=new_event_id(),
            tick=world.tick,
            action=SemanticAction(
                verb=ActionType.AMBIENT,
                actor=SYSTEM_ACTOR,
                raw_input=f"[{source}]",
            ),
            transitions=transitions,
            witnesses=[],
            narrative_hint=narrative_hint,
        )
    )


def apply_conditional_scenario_beats(world: WorldState) -> list[str]:
    """
    Fire scenario beats whose ``when`` conditions are met (not tick-only).

    Beats may declare optional ``when:`` alongside ``tick:`` minimum threshold.
    Each conditional beat fires once per unique beat id/key.
    """
    from .pressure_eval import conditions_met

    scenario = (world.config.extra or {}).get("scenario") or {}
    beats: list[dict] = list(scenario.get("beats") or [])
    if not beats:
        return []

    fired_key = "scenario_conditional_fired"
    fired: set[str] = set(world.meta.get(fired_key) or [])
    log_lines: list[str] = []

    for beat in beats:
        when = beat.get("when")
        if not when:
            continue
        min_tick = int(beat.get("tick", 0))
        if world.tick < min_tick:
            continue
        if isinstance(when, dict):
            when = [when]
        if not conditions_met(when, world):
            continue

        bid = str(beat.get("id") or _beat_key(beat))
        if bid in fired:
            continue

        effects = beat.get("effects") or {}
        if not isinstance(effects, dict):
            continue

        fired.add(bid)
        world.meta[fired_key] = sorted(fired)

        title = beat.get("title") or bid
        log_lines.append(f"[scenario] conditional beat: {title}")
        bundle_lines, transitions = apply_effect_bundle(
            world,
            effects,
            cause="scenario_conditional",
            log_prefix="[scenario]",
        )
        log_lines.extend(bundle_lines)
        if transitions:
            log_scripted_event(
                world,
                transitions,
                source=f"scenario:conditional:{bid}",
                narrative_hint=f"[scenario conditional {title}]",
            )

    return log_lines


def _goal_transition(
    world: WorldState,
    spec: dict[str, Any],
    *,
    cause: str,
) -> Transition | None:
    eid = _resolve_entity_ref(world, spec.get("entity"))
    goal = str(spec.get("goal", "")).strip()[:160]
    if not eid or not goal:
        return None
    ent = world.spatial.entities.get(eid)  # type: ignore[arg-type]
    if ent is None or goal in ent.goals:
        return None
    return Transition(
        kind=TransitionKind.ENTITY_GOAL_ADDED,
        payload={
            "entity_id": eid,
            "goal": goal,
            "cause": cause,
            "priority": float(spec.get("priority", 0.7)),
        },
    )


def _log_scenario_event(
    world: WorldState,
    transitions: list[Transition],
    *,
    beat_key: str,
    kind: str,
) -> None:
    """Backward-compatible alias for scripted event logging."""
    log_scripted_event(
        world,
        transitions,
        source=f"scenario:{kind}:{beat_key}",
        narrative_hint=f"[scenario beat {beat_key}]",
    )


def _resolve_object_ref(world: WorldState, ref: Any) -> Optional[ObjectId]:
    if ref is None:
        return None
    s = str(ref).lower()
    pack_map = world.meta.get("object_pack_ids") or {}
    for oid_str, pack_id in pack_map.items():
        if s == str(pack_id).lower():
            return ObjectId(oid_str)
    for oid, obj in world.spatial.objects.items():
        if s in str(oid).lower() or s in obj.name.lower():
            return oid
    return None


def _resolve_scenario_coord(world: WorldState, spec: dict[str, Any]) -> Optional[Coord]:
    if spec.get("x") is not None and spec.get("y") is not None:
        return Coord(
            x=int(spec["x"]),
            y=int(spec["y"]),
            z=int(spec.get("z", 0)),
        )
    oid = _resolve_object_ref(world, spec.get("object"))
    if oid is not None:
        obj = world.spatial.objects.get(oid)
        if obj is not None and obj.position is not None:
            return obj.position
    eid = _resolve_entity_ref(world, spec.get("entity"))
    if eid is not None:
        ent = world.spatial.entities.get(eid)  # type: ignore[arg-type]
        if ent is not None:
            return ent.position
    return None


def _effect_field_deposits(
    world: WorldState,
    effects: dict[str, Any],
    *,
    cause: str,
    log_prefix: str,
    log_lines: list[str],
) -> list[Transition]:
    from .fields import MEDIA_TILE_GROUND, deposit_at

    out: list[Transition] = []
    for dep in effects.get("field_deposits") or []:
        if not isinstance(dep, dict):
            continue
        coord = _resolve_scenario_coord(world, dep)
        if coord is None:
            continue
        substance = str(dep.get("substance", "")).lower()
        if not substance:
            continue
        medium = str(dep.get("medium", MEDIA_TILE_GROUND))
        amount = float(dep.get("amount", 30.0))
        out.append(
            deposit_at(
                coord,
                medium,
                substance,
                amount,
                cause=str(dep.get("cause") or cause),
                tick=world.tick,
            )
        )
        log_lines.append(
            f"{log_prefix} field deposit: {substance} ({amount:.0f}) @ {coord}"
        )
    return out


def _effect_tile_modifies(
    world: WorldState,
    effects: dict[str, Any],
    *,
    cause: str,
    log_prefix: str,
    log_lines: list[str],
) -> list[Transition]:
    out: list[Transition] = []
    for spec in effects.get("tile_modify") or []:
        if not isinstance(spec, dict):
            continue
        coord = _resolve_scenario_coord(world, spec)
        if coord is None:
            continue
        payload: dict[str, Any] = {
            "x": coord.x,
            "y": coord.y,
            "z": coord.z,
            "cause": str(spec.get("cause") or cause),
        }
        if spec.get("add_tags"):
            payload["add_tags"] = list(spec["add_tags"])
        if spec.get("remove_tags"):
            payload["remove_tags"] = list(spec["remove_tags"])
        if "add_tags" not in payload and "remove_tags" not in payload:
            continue
        out.append(Transition(kind=TransitionKind.STRUCTURE_MODIFIED, payload=payload))
        log_lines.append(f"{log_prefix} tile modify @ {coord}: {payload.get('add_tags')}")
    return out


def _effect_object_modifies(
    world: WorldState,
    effects: dict[str, Any],
    *,
    cause: str,
    log_prefix: str,
    log_lines: list[str],
) -> list[Transition]:
    out: list[Transition] = []
    for spec in effects.get("object_modify") or []:
        if not isinstance(spec, dict):
            continue
        oid = _resolve_object_ref(world, spec.get("object"))
        if oid is None:
            continue
        obj = world.spatial.objects.get(oid)
        if obj is None:
            continue
        for tag in spec.get("add_tags") or []:
            if tag not in obj.tags:
                obj.tags.append(str(tag))
        for tag in spec.get("remove_tags") or []:
            tag_s = str(tag)
            if tag_s in obj.tags:
                obj.tags.remove(tag_s)
        meta_set = spec.get("meta_set") or {}
        if isinstance(meta_set, dict):
            obj.meta.update(meta_set)
        fluid_vol = spec.get("fluid_volume_ml")
        if fluid_vol is not None:
            fluid = dict(obj.meta.get("fluid") or {})
            fluid["volume_ml"] = float(fluid_vol)
            obj.meta["fluid"] = fluid
        out.append(
            Transition(
                kind=TransitionKind.FLUID_CHANGED,
                payload={
                    "target_kind": "object",
                    "object_id": str(oid),
                    "material": str((obj.meta.get("fluid") or {}).get("material", "")),
                    "volume_ml": float((obj.meta.get("fluid") or {}).get("volume_ml", 0)),
                    "cause": str(spec.get("cause") or cause),
                    "_scripted_object_modify": True,
                },
            )
        )
        log_lines.append(f"{log_prefix} object modify: {spec.get('object')}")
    return out


def _effect_register_facts(
    world: WorldState,
    effects: dict[str, Any],
    *,
    cause: str,
    log_prefix: str,
    log_lines: list[str],
) -> list[Transition]:
    from .world_adjudicator import world_fact_transition

    out: list[Transition] = []
    for spec in effects.get("register_facts") or []:
        if not isinstance(spec, dict):
            continue
        claim = str(spec.get("claim", "")).strip()
        if not claim:
            continue
        tags = [str(t) for t in (spec.get("tags") or [])]
        out.append(
            world_fact_transition(
                WorldFact(
                    claim=claim[:240],
                    scope=WorldFactScope.WORLD,
                    established_tick=world.tick,
                    established_by="system",
                    tags=tags,
                    source_intent=str(spec.get("cause") or cause),
                )
            )
        )
        log_lines.append(f"{log_prefix} fact: {claim[:60]}")
    return out
