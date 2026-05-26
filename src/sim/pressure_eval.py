"""
Evaluate pack-declared latent pressures each tick.

Pressures are AND-combined conditions on world state; when active they set
``world.meta["active_pressures"]`` and per-entity ``pressure_boosts`` for
the reactive policy and NarrativeDirector — implicit triggers without
hand-authored dialogue.

State is applied via ``PRESSURE_STATE_UPDATED`` transitions for replay safety.
"""

from __future__ import annotations

from typing import Any, Optional

from .schemas import EntityId, Transition, TransitionKind, WorldState

_PRESSURE_META = "active_pressures"
_BOOST_META = "pressure_boosts"
_AMBIENT_MULT_META = "ambient_probability_mult"


def evaluate_pressures(world: WorldState) -> tuple[list[str], list[Transition]]:
    """
    Evaluate all rules in ``world.config.pressures``.

    Returns log lines for UI and replay-safe transitions to apply.
    """
    rules = world.config.pressures or []
    if not rules:
        return [], [
            Transition(
                kind=TransitionKind.PRESSURE_STATE_UPDATED,
                payload={
                    "active_pressures": [],
                    "ambient_probability_mult": {},
                    "entity_boosts": {},
                    "meta_patches": {},
                    "clear_entity_boosts": True,
                },
            )
        ]

    active: list[dict[str, Any]] = []
    ambient_mult: dict[str, float] = {}
    boosts_by_entity: dict[str, list[dict[str, Any]]] = {}
    meta_patches: dict[str, Any] = {}
    log_lines: list[str] = []
    edge_transitions: list[Transition] = []

    for rule in rules:
        rid = str(rule.get("id", ""))
        when = rule.get("when") or rule.get("conditions") or []
        if isinstance(when, dict):
            when = [when]
        if not _conditions_met(when, world):
            continue
        effects = rule.get("effects") or {}
        entry = {"id": rid, **{k: v for k, v in effects.items() if k != "candidate_boosts"}}
        active.append(entry)
        log_lines.append(f"[pressure] {rid}")

        if effects.get("director_focus"):
            meta_patches["director_scene_focus"] = str(effects["director_focus"])
        if effects.get("director_tension") is not None:
            meta_patches["director_tension_override"] = float(effects["director_tension"])

        for k, v in (effects.get("set_world_meta") or {}).items():
            meta_patches[str(k)] = v

        for amb_id, mult in (effects.get("ambient_probability_mult") or {}).items():
            ambient_mult[str(amb_id)] = max(
                ambient_mult.get(str(amb_id), 1.0), float(mult)
            )

        for boost in effects.get("candidate_boosts") or []:
            eid = _resolve_entity_ref(world, boost.get("entity"))
            if not eid:
                continue
            boosts_by_entity.setdefault(str(eid), []).append(
                {
                    "raw_input": boost.get("raw_input"),
                    "verb": boost.get("verb"),
                    "weight_delta": float(boost.get("weight_delta", 0.1)),
                }
            )

        for nudge in effects.get("edge_nudge") or []:
            edge_transitions.extend(_edge_nudge_transitions(world, nudge))

    pressure_transition = Transition(
        kind=TransitionKind.PRESSURE_STATE_UPDATED,
        payload={
            "active_pressures": active,
            "ambient_probability_mult": ambient_mult,
            "entity_boosts": boosts_by_entity,
            "meta_patches": meta_patches,
            "clear_entity_boosts": True,
        },
    )
    return log_lines, edge_transitions + [pressure_transition]


def ambient_probability_multiplier(world: WorldState, event_id: str) -> float:
    mults = world.meta.get(_AMBIENT_MULT_META) or {}
    return float(mults.get(event_id, 1.0))


def pressure_weight_delta(
    entity_id: EntityId, raw_input: str, verb: str, world: WorldState
) -> float:
    ent = world.spatial.entities.get(entity_id)
    if ent is None:
        return 0.0
    delta = 0.0
    verb_l = verb.lower().split(".")[-1]
    for boost in ent.meta.get(_BOOST_META) or []:
        if boost.get("raw_input") and boost["raw_input"] != raw_input:
            continue
        if boost.get("verb") and str(boost["verb"]).lower() != verb_l:
            continue
        delta += float(boost.get("weight_delta", 0.0))
    return delta


def _conditions_met(conditions: list, world: WorldState) -> bool:
    if not conditions:
        return False
    for cond in conditions:
        if not _one_condition(cond, world):
            return False
    return True


def _one_condition(cond: dict, world: WorldState) -> bool:
    ctype = cond.get("type", "")
    if ctype == "tick_gte":
        return world.tick >= int(cond.get("tick", 0))
    if ctype == "tick_lt":
        return world.tick < int(cond.get("tick", 999999))
    if ctype == "meta_flag":
        return bool(world.meta.get(str(cond.get("key", ""))))
    if ctype == "meta_equals":
        return world.meta.get(str(cond.get("key"))) == cond.get("value")
    if ctype == "goal_incomplete":
        gid = str(cond.get("goal_id", ""))
        for g in world.config.goals:
            if g.id == gid:
                return not g.completed
        return False
    if ctype == "goal_complete":
        gid = str(cond.get("goal_id", ""))
        for g in world.config.goals:
            if g.id == gid:
                return g.completed
        return False
    if ctype == "edge_weight_gte":
        w = _edge_weight(
            world,
            _resolve_entity_ref(world, cond.get("source")),
            _resolve_entity_ref(world, cond.get("target")),
            str(cond.get("edge_kind", "")),
        )
        return w is not None and w >= float(cond.get("weight", 0.5))
    if ctype == "event_occurred":
        return _event_in_log(world, cond)
    if ctype == "any_memory_contains":
        needle = str(cond.get("text", "")).lower()
        if not needle:
            return False
        for ent in world.spatial.entities.values():
            for rec in ent.meta.get("observation_memory") or []:
                if needle in str(rec.get("summary", "")).lower():
                    return True
        return False
    if ctype == "scenario_phase_gte":
        phase = int(world.meta.get("scenario_phase", 0))
        return phase >= int(cond.get("phase", 0))
    if ctype == "world_fact_tag":
        tag = str(cond.get("tag", ""))
        if not tag:
            return False
        within = int(cond.get("within_ticks", 99999))
        cutoff = world.tick - within
        for fact in reversed(world.world_facts):
            if fact.established_tick < cutoff:
                break
            if tag in fact.tags:
                return True
        return False
    if ctype == "world_fact_claim_contains":
        needle = str(cond.get("text", "")).lower()
        if not needle:
            return False
        within = int(cond.get("within_ticks", 99999))
        cutoff = world.tick - within
        for fact in reversed(world.world_facts):
            if fact.established_tick < cutoff:
                break
            if needle in fact.claim.lower():
                return True
        return False
    if ctype == "entity_dead":
        eid = _resolve_entity_ref(world, cond.get("entity"))
        if not eid:
            return False
        ent = world.spatial.entities.get(EntityId(eid))
        if ent is None:
            return True
        return not ent.alive or ent.health <= 0
    if ctype == "entity_alive":
        eid = _resolve_entity_ref(world, cond.get("entity"))
        if not eid:
            return False
        ent = world.spatial.entities.get(EntityId(eid))
        return ent is not None and ent.alive and ent.health > 0
    if ctype == "transition_in_log":
        kind = str(cond.get("kind", ""))
        cause = cond.get("cause")
        lookback = int(cond.get("lookback_ticks", 20))
        cutoff = world.tick - lookback
        for ev in reversed(world.event_log):
            if ev.tick < cutoff:
                break
            for t in ev.transitions:
                tk = t.kind.value if hasattr(t.kind, "value") else str(t.kind)
                if kind and tk != kind:
                    continue
                if cause is not None and t.payload.get("cause") != cause:
                    continue
                return True
        return False
    if ctype == "trigger_not_fired":
        tid = str(cond.get("trigger_id", ""))
        fired = world.meta.get("campaign_director_fired") or []
        return tid not in fired
    if ctype == "atmosphere_substance_gte":
        from .fields import region_atmosphere_amount

        substance = str(cond.get("substance", "")).lower()
        threshold = float(cond.get("amount", cond.get("threshold", 10.0)))
        if not substance:
            return False
        return region_atmosphere_amount(world, substance) >= threshold
    if ctype == "tile_substance_gte":
        from .fields import MEDIA_TILE_GROUND, get_layer
        from .schemas import Coord

        substance = str(cond.get("substance", "")).lower()
        threshold = float(cond.get("amount", cond.get("threshold", 10.0)))
        if not substance:
            return False
        coord: Optional[Coord] = None
        if cond.get("x") is not None and cond.get("y") is not None:
            coord = Coord(
                x=int(cond["x"]),
                y=int(cond["y"]),
                z=int(cond.get("z", 0)),
            )
        else:
            obj_ref = cond.get("object")
            if obj_ref:
                pack_map = world.meta.get("object_pack_ids") or {}
                oid_str = None
                s = str(obj_ref).lower()
                for oid, pack_id in pack_map.items():
                    if s == str(pack_id).lower():
                        oid_str = oid
                        break
                if oid_str is None:
                    for oid, obj in world.spatial.objects.items():
                        if s in str(oid).lower() or s in obj.name.lower():
                            oid_str = str(oid)
                            break
                if oid_str:
                    obj = world.spatial.objects.get(oid_str)  # type: ignore[arg-type]
                    if obj is not None and obj.position is not None:
                        coord = obj.position
        if coord is None:
            return False
        tile = world.spatial.tile_at(coord)
        medium = str(cond.get("medium", MEDIA_TILE_GROUND))
        layer = get_layer(tile.env, medium, legacy_tile=tile)
        amt = float((layer.get(substance) or {}).get("amount", 0))
        return amt >= threshold
    return False


def conditions_met(conditions: list, world: WorldState) -> bool:
    """Public wrapper — evaluate AND-combined trigger conditions."""
    return _conditions_met(conditions, world)


def _event_in_log(world: WorldState, cond: dict) -> bool:
    verb = (cond.get("verb") or "").lower()
    actor = _resolve_entity_ref(world, cond.get("actor"))
    target = _resolve_entity_ref(world, cond.get("target"))
    lookback = int(cond.get("lookback_ticks", 9999))
    cutoff = world.tick - lookback
    for ev in reversed(world.event_log):
        if ev.tick < cutoff:
            break
        if verb and str(ev.action.verb).lower().split(".")[-1] != verb:
            continue
        if actor and str(ev.action.actor) != actor:
            continue
        if target:
            tgt = str(ev.action.target or "")
            if target not in tgt and tgt != target:
                continue
        return True
    return False


def _edge_weight(
    world: WorldState, source: Optional[str], target: Optional[str], edge_kind: str
) -> Optional[float]:
    if not source or not target or not edge_kind:
        return None
    for e in world.relational.edges.get(source, {}).get(target, []):
        ek = e.kind.value if hasattr(e.kind, "value") else str(e.kind)
        if ek == edge_kind:
            return float(e.weight)
    return None


def _resolve_entity_ref(world: WorldState, ref: Any) -> Optional[str]:
    if ref is None:
        return None
    s = str(ref)
    for eid, ent in world.all_entities().items():
        if s == str(eid) or s.lower() in str(eid).lower():
            return str(eid)
        if s.lower() in ent.name.lower():
            return str(eid)
    return s


def _edge_nudge_transitions(world: WorldState, nudge: dict) -> list[Transition]:
    from .contact_effects import _edge_nudge_transitions

    src = _resolve_entity_ref(world, nudge.get("source"))
    tgt = _resolve_entity_ref(world, nudge.get("target"))
    if not src or not tgt:
        return []
    return _edge_nudge_transitions(
        src,
        tgt,
        str(nudge.get("edge_kind", "interacted")),
        float(nudge.get("weight", 0.5)),
        world.tick,
    )


def _apply_edge_nudge(world: WorldState, nudge: dict) -> None:
    """Legacy direct apply — prefer ``evaluate_pressures`` transitions."""
    from .compiler import apply_transitions

    transitions = _edge_nudge_transitions(world, nudge)
    if transitions:
        apply_transitions(world, transitions)
