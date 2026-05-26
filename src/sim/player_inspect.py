"""
Player-facing inspect / examine — surfaces tag-derived actions (Qud-style).

Shows what the engine can compile from visible entities and objects,
without calling the LM.

The richer this report is, the less the player has to "guess what the DM
accepts" — and the more the simulation feels like a real system being
played, not an LM being prompted.  Specifically this module surfaces:

  - Entity drive + active plan (long-horizon intent visible to the player)
  - Conditions and equipped slots (mechanical state on the entity)
  - Material / fluid contents on objects
  - Field state on the player's tile (substances, fluid spills, marks)
  - Property-interaction rules that *could* fire on visible targets given
    the player's currently equipped/inventory tags ("if you used X here,
    Y would happen")
  - Recent world facts whose subject matches the target
"""

from __future__ import annotations

from typing import Optional

from .capability_menu import pack_capability_candidates
from .interaction_resolver import interaction_hint_strings
from .projection import visible_from
from .schemas import (
    Coord,
    EntityId,
    EntityState,
    ObjectState,
    PropertyInteractionRule,
    WorldState,
)
from .social_inspect import format_social_inspect


def _resolve_entity(world: WorldState, ref: str) -> tuple[Optional[EntityState], Optional[EntityId]]:
    ref_l = ref.lower().strip()
    for eid, ent in world.all_entities().items():
        if ref_l in ent.name.lower() or ref_l in str(eid).lower():
            return ent, eid
    return None, None


def _resolve_object(world: WorldState, ref: str) -> Optional[ObjectState]:
    ref_l = ref.lower().strip()
    for obj in world.spatial.objects.values():
        if ref_l in obj.name.lower() or ref_l in str(obj.object_id).lower():
            return obj
    return None


def _combination_hints(world: WorldState, tags: set[str]) -> list[str]:
    """Rules from combinations.yaml whose inputs overlap object tags."""
    if not tags:
        return []
    tag_l = {t.lower() for t in tags}
    hints: list[str] = []
    for rule in world.config.combination_rules or []:
        a_tags = {t.lower() for t in (getattr(rule, "input_a_tags", None) or [])}
        b_tags = {t.lower() for t in (getattr(rule, "input_b_tags", None) or [])}
        if tag_l.intersection(a_tags) or tag_l.intersection(b_tags):
            name = getattr(rule, "name", None) or getattr(rule, "id", "combine")
            hints.append(str(name))
    return hints[:6]


# ─── new helpers ────────────────────────────────────────────────────────────


def _format_entity_intent_block(ent: EntityState) -> list[str]:
    """Reveal long-horizon intent: drive, active plan progress, goals."""
    lines: list[str] = []
    if ent.drive:
        lines.append(f"    drive: {ent.drive[:100]}")
    plan = getattr(ent, "active_plan", None)
    if plan is not None and getattr(plan, "steps", None):
        try:
            from .schemas import PlanStatus

            cur = int(plan.current_step or 0)
            total = len(plan.steps)
            status_val = (
                plan.status.value
                if hasattr(plan.status, "value")
                else str(plan.status)
            )
            step_label = ""
            if 0 <= cur < total:
                step = plan.steps[cur]
                step_label = step.label or step.kind.value
            lines.append(
                f"    plan: {plan.name or plan.drive_source[:40]} "
                f"[{status_val} {cur + 1}/{total}"
                + (f": {step_label}" if step_label else "")
                + "]"
            )
            if getattr(plan, "failure_reason", ""):
                lines.append(f"      failure_reason: {plan.failure_reason}")
            paused_reason = (plan.meta or {}).get("paused_reason")
            if paused_reason:
                lines.append(f"      paused: {paused_reason}")
        except Exception:
            pass
    goals = list(getattr(ent, "goals", None) or [])
    if goals:
        lines.append(f"    goals: {goals[0][:100]}"
                     + (f"  (+{len(goals) - 1} more)" if len(goals) > 1 else ""))
    conditions = getattr(ent, "conditions", None) or {}
    if conditions:
        cond_strs = [f"{k}({v})" for k, v in list(conditions.items())[:5]]
        lines.append(f"    conditions: {', '.join(cond_strs)}")
    return lines


def _format_entity_equipment_block(ent: EntityState, world: WorldState) -> list[str]:
    """List equipped slots + the tags those items carry — directly drives property rules."""
    slots: dict[str, str] = dict(getattr(ent, "equipped_slots", None) or {})
    if getattr(ent, "equipped_weapon", None) and "main_hand" not in slots:
        slots["main_hand"] = str(ent.equipped_weapon)
    if getattr(ent, "equipped_armor", None) and "chest" not in slots:
        slots["chest"] = str(ent.equipped_armor)
    if not slots:
        return []
    out = ["    equipped:"]
    for slot, oid in list(slots.items())[:8]:
        obj = world.spatial.objects.get(oid)
        name = obj.name if obj else str(oid)[:8]
        tag_hint = ""
        if obj and obj.tags:
            tag_hint = f"  [{', '.join(obj.tags[:4])}]"
        out.append(f"      • {slot}: {name}{tag_hint}")
    return out


def _format_object_material_block(obj: ObjectState) -> list[str]:
    """Surface material, fluid contents, and condition for an object."""
    out: list[str] = []
    meta = obj.meta or {}
    material = meta.get("material")
    if material:
        out.append(f"    material: {material}")
    fluid = meta.get("fluid") or meta.get("contents")
    if isinstance(fluid, dict):
        kind = fluid.get("material") or fluid.get("kind") or "liquid"
        vol = fluid.get("volume_ml") or fluid.get("amount")
        if vol is not None:
            out.append(f"    contains: {kind} ({vol})")
        else:
            out.append(f"    contains: {kind}")
    elif isinstance(fluid, str):
        out.append(f"    contains: {fluid}")
    cond = meta.get("condition") or meta.get("integrity")
    if cond is not None:
        out.append(f"    condition: {cond}")
    return out


def _recent_facts_mentioning(
    world: WorldState,
    subject_id: str,
    limit: int = 4,
) -> list[str]:
    """Last N world_facts whose subject_id matches or whose claim mentions the id."""
    facts = list(getattr(world, "world_facts", None) or [])
    if not facts:
        return []
    sid = str(subject_id).lower()
    out: list[str] = []
    for fact in reversed(facts):
        try:
            subj = str(getattr(fact, "subject_id", "") or "").lower()
            claim = str(getattr(fact, "claim", "") or "")
        except Exception:
            continue
        if subj == sid or sid in claim.lower():
            tick = getattr(fact, "established_tick", None)
            prefix = f"t={tick} " if tick is not None else ""
            out.append(f"      • {prefix}{claim[:100]}")
            if len(out) >= limit:
                break
    return out


def _format_tile_fields_block(world: WorldState, pos: Coord) -> list[str]:
    """Show substances, fluid spills, and marks on the given tile."""
    grid = world.spatial
    try:
        tile = grid.tile_at(pos)
    except Exception:
        return []
    if tile is None:
        return []
    out: list[str] = []
    env = getattr(tile, "env", None) or {}
    fields_bucket = env.get("fields") if isinstance(env, dict) else None
    canonical_ground_materials: set[str] = set()
    if isinstance(fields_bucket, dict):
        for medium, layer in fields_bucket.items():
            if not isinstance(layer, dict):
                continue
            substances = [
                f"{name}({int(data.get('amount', 0)) if isinstance(data, dict) else 0})"
                for name, data in list(layer.items())[:5]
            ]
            if substances:
                out.append(f"    {medium}: {', '.join(substances)}")
            if medium == "ground":
                canonical_ground_materials = {str(n).lower() for n in layer.keys()}
    # Show the legacy spill mirror only when the canonical layer is
    # empty for this material — otherwise it's a duplicate of the
    # ground line above.
    spill = env.get("fluid_spill") if isinstance(env, dict) else None
    if isinstance(spill, dict) and spill.get("material"):
        mat = str(spill["material"]).lower()
        if mat not in canonical_ground_materials:
            vol = spill.get("volume_ml") or spill.get("amount")
            suffix = f" ({vol})" if vol is not None else ""
            out.append(f"    spill: {spill['material']}{suffix}")
    marks = getattr(tile, "marks", None) or []
    if marks:
        head = marks[-3:]
        out.append("    marks: " + "; ".join(str(m)[:60] for m in head))
    return out


def _possible_property_rule_hits(
    world: WorldState,
    actor: EntityState,
) -> list[str]:
    """
    Preview which ``property_interactions`` rules could fire right now,
    given the actor's current source tags and any nearby targets.

    This is the "what could happen here" surface that converts Mark_1
    from a guess-the-verb shell into a system the player can read and
    react to.  Returns at most 6 one-line hints.
    """
    rules: list[PropertyInteractionRule] = list(
        world.config.property_interactions or []
    )
    if not rules:
        return []

    from .property_interaction_resolver import (
        _actor_source_tags,
        _entity_tags,
        _has_any,
        _object_tags,
    )

    src_tags = _actor_source_tags(actor, world)
    if not src_tags:
        return []

    grid = world.spatial
    if actor.position is None:
        return []
    ax, ay = actor.position.x, actor.position.y
    SCAN_RADIUS = 8

    candidates: list[tuple[str, frozenset[str], float]] = []
    for eid, ent in grid.entities.items():
        if eid == actor.entity_id or ent.position is None or not ent.alive:
            continue
        dx = ent.position.x - ax
        dy = ent.position.y - ay
        dist = (dx * dx + dy * dy) ** 0.5
        if dist <= SCAN_RADIUS:
            tags = _entity_tags(ent)
            if tags:
                candidates.append((ent.name, tags, dist))
    for oid, obj in grid.objects.items():
        if obj.position is None:
            continue
        dx = obj.position.x - ax
        dy = obj.position.y - ay
        dist = (dx * dx + dy * dy) ** 0.5
        if dist <= SCAN_RADIUS:
            tags = _object_tags(obj)
            if tags:
                candidates.append((obj.name, tags, dist))

    seen: set[str] = set()
    hints: list[str] = []
    for rule in sorted(rules, key=lambda r: -int(getattr(r, "priority", 0))):
        need_src = frozenset(t.lower() for t in rule.actor_or_tool_tags)
        if not _has_any(need_src, src_tags):
            continue
        need_tgt = frozenset(t.lower() for t in rule.target_tags)
        immunity = frozenset(t.lower() for t in rule.target_immunity_tags)
        for name, ttags, tdist in candidates:
            if tdist > rule.max_distance:
                continue
            if not _has_any(need_tgt, ttags):
                continue
            if immunity and (immunity & ttags):
                continue
            label = (
                f"{rule.name or rule.id} → {name}"
                if getattr(rule, "name", None) or getattr(rule, "id", None)
                else f"({', '.join(sorted(need_src))[:30]}) on {name}"
            )
            if label in seen:
                continue
            seen.add(label)
            hints.append(f"    → {label}")
            break  # one target per rule keeps the list readable
        if len(hints) >= 6:
            break
    return hints


def _format_entity_block(ent: EntityState, world: WorldState) -> list[str]:
    lines = [
        f"  {ent.name} ({ent.entity_id})",
        f"    mood={ent.emotional_state.value}  alert={ent.alertness.value}"
        f"  hp={ent.health}/{ent.max_health}",
    ]
    if ent.tags:
        lines.append(f"    tags: {', '.join(ent.tags)}")
    if ent.inventory:
        inv_names = []
        for oid in ent.inventory[:8]:
            obj = world.spatial.objects.get(oid)
            inv_names.append(obj.name if obj else str(oid)[:8])
        lines.append(f"    inventory: {', '.join(inv_names)}")
    lines.extend(_format_entity_equipment_block(ent, world))
    lines.extend(_format_entity_intent_block(ent))
    return lines


def _format_object_block(obj: ObjectState, world: WorldState) -> list[str]:
    lines = [f"  {obj.name} ({obj.object_id})"]
    if obj.tags:
        lines.append(f"    tags: {', '.join(obj.tags)}")
    if obj.position:
        lines.append(f"    at ({obj.position.x},{obj.position.y},{obj.position.z})")
    lines.extend(_format_object_material_block(obj))
    combo = _combination_hints(world, set(obj.tags or []))
    if combo:
        lines.append(f"    chemistry: {'; '.join(combo)}")
    return lines


def format_inspect_report(
    world: WorldState,
    player_id: Optional[EntityId],
    target_ref: Optional[str] = None,
) -> str:
    """
    Build an inspect report for the player.

    With no target: room survey + suggested actions.
    With target: entity or object detail + compilable verb hints.
    """
    player = world.spatial.entities.get(player_id) if player_id else None
    if player is None:
        return "(no player entity)"

    lines: list[str] = ["── Inspect ──"]

    if target_ref:
        ent, eid = _resolve_entity(world, target_ref)
        if ent is not None:
            lines.extend(_format_entity_block(ent, world))
            target_label = ent.name
            # Recent world facts naming this entity.
            facts = _recent_facts_mentioning(world, str(eid), limit=4)
            if facts:
                lines.append("")
                lines.append("    recent facts:")
                lines.extend(facts)
            social = format_social_inspect(world, player_id, target_ref)
            if social and "Social" in social:
                lines.append("")
                lines.extend(social.splitlines()[1:])
        else:
            obj = _resolve_object(world, target_ref)
            if obj is None:
                return f"(nothing matching {target_ref!r} in the world)"
            lines.extend(_format_object_block(obj, world))
            target_label = obj.name
            facts = _recent_facts_mentioning(world, str(obj.object_id), limit=3)
            if facts:
                lines.append("")
                lines.append("    recent facts:")
                lines.extend(facts)
    else:
        grid = world.spatial
        visible = visible_from(grid, player.position, player.sight_range)
        lines.append(f"  You are at ({player.position.x},{player.position.y}).")
        # Tile-level field state on the player's current tile — substances,
        # fluid spills, marks.  Lets the player *see* the chemistry the
        # simulator is tracking instead of having to guess.
        tile_lines = _format_tile_fields_block(world, player.position)
        if tile_lines:
            lines.append("  Current tile:")
            lines.extend(tile_lines)
        lines.append("  Nearby:")
        shown = 0
        for ent in grid.entities.values():
            if not ent.alive or ent.entity_id == player_id:
                continue
            if ent.position in visible and player.position.manhattan(ent.position) <= 10:
                # Surface drive/plan one-liner so the player can read
                # what each NPC is *trying* to do, not just their mood.
                drive_hint = ""
                if ent.drive:
                    drive_hint = f"  — drive: {ent.drive[:60]}"
                plan = getattr(ent, "active_plan", None)
                if plan is not None and plan.steps:
                    try:
                        from .schemas import PlanStatus

                        if plan.status == PlanStatus.ACTIVE:
                            drive_hint = (
                                f"  — plan: {plan.name or plan.drive_source[:40]}"
                                f" ({plan.current_step + 1}/{len(plan.steps)})"
                            )
                    except Exception:
                        pass
                lines.append(
                    f"    • {ent.name} — {ent.emotional_state.value}, "
                    f"{ent.alertness.value} alert{drive_hint}"
                )
                shown += 1
        for obj in grid.objects.values():
            if obj.position and obj.position in visible:
                if player.position.manhattan(obj.position) <= 8:
                    tag_str = ", ".join(obj.tags[:4]) if obj.tags else "no tags"
                    lines.append(f"    • {obj.name} — {tag_str}")
                    shown += 1
        if shown == 0:
            lines.append("    (nothing notable in sight)")
        target_label = "room"

    # Property-interaction preview — what *could* happen given the player's
    # current tags + nearby targets.  This is the single biggest "show, don't
    # guess" lever: it tells the player which mechanical combinations the
    # engine recognises right here, so they can act on the system instead of
    # poking at it.
    prop_hints = _possible_property_rule_hits(world, player)
    if prop_hints:
        lines.append("")
        lines.append("  Possible reactions here:")
        lines.extend(prop_hints)

    grammar = interaction_hint_strings(world, player_id, max_hints=8)
    if grammar:
        lines.append("")
        lines.append("  Tag-grammar interactions:")
        for hint in grammar:
            lines.append(f"    → {hint}")

    candidates = pack_capability_candidates(player, world, max_add=16)
    if candidates:
        lines.append("")
        lines.append(f"  Actions you could try ({target_label}):")
        seen: set[str] = set()
        for _action, label, _weight in candidates:
            if label in seen:
                continue
            seen.add(label)
            lines.append(f"    → {label}")
            if len(seen) >= 12:
                break
    else:
        lines.append("")
        if not grammar:
            lines.append("  (no pack-authored actions visible — try kernel verbs:"
                         " move, take, attack, speak, or describe freely)")

    if not target_ref:
        social = format_social_inspect(world, player_id)
        if social:
            lines.append("")
            lines.extend(social.splitlines())

    return "\n".join(lines)
