"""
Player-facing world diff — shows what actually changed after a turn.

Lighter than sim_debug snapshots; always safe to show in normal play.
"""

from __future__ import annotations

from typing import Any, Optional

from .schemas import EntityId, Transition, TransitionKind, WorldState


def capture_player_snapshot(world: WorldState, player_id: Optional[EntityId]) -> dict[str, Any]:
    """Compact snapshot of player-visible durable state."""
    snap: dict[str, Any] = {
        "tick": world.tick,
        "entities": {},
        "objects": {},
        "tile_marks": {},
        "facts_n": len(world.world_facts),
    }
    player = world.spatial.entities.get(player_id) if player_id else None
    origin = player.position if player else None
    grid = world.spatial

    for eid, ent in grid.entities.items():
        if not ent.alive:
            continue
        if origin and origin.manhattan(ent.position) > 14:
            continue
        snap["entities"][str(eid)] = {
            "name": ent.name,
            "mood": ent.emotional_state.value,
            "alert": ent.alertness.value,
            "hp": ent.health,
            "pos": f"{ent.position.x},{ent.position.y}",
        }

    for oid, obj in grid.objects.items():
        if obj.position and origin and origin.manhattan(obj.position) > 14:
            continue
        snap["objects"][str(oid)] = {
            "name": obj.name,
            "tags": sorted(obj.tags or [])[:8],
        }

    if origin:
        for dx in range(-12, 13):
            for dy in range(-12, 13):
                from .schemas import Coord
                c = Coord(x=origin.x + dx, y=origin.y + dy, z=origin.z)
                tile = grid.tile_at(c)
                if tile.marks:
                    snap["tile_marks"][f"{c.x},{c.y}"] = "|".join(tile.marks[:3])

    return snap


def _describe_transition(t: Transition, world: WorldState) -> Optional[str]:
    kind = t.kind
    p = t.payload or {}

    def _ent_name(eid: str) -> str:
        ent = world.spatial.entities.get(EntityId(eid)) if eid else None
        return ent.name if ent else eid[:10]

    if kind == TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED:
        return f"{_ent_name(p.get('entity_id', ''))}: mood → {p.get('to')}"
    if kind == TransitionKind.ENTITY_ALERTNESS_CHANGED:
        return f"{_ent_name(p.get('entity_id', ''))}: alertness → {p.get('to')}"
    if kind == TransitionKind.ENTITY_HEALTH_CHANGED:
        delta = p.get("delta", 0)
        sign = "+" if delta > 0 else ""
        return f"{_ent_name(p.get('entity_id', ''))}: hp {sign}{delta}"
    if kind == TransitionKind.ENTITY_MOVED:
        return f"{_ent_name(p.get('entity_id', ''))}: moved"
    if kind == TransitionKind.ITEM_TRANSFERRED:
        return f"item transferred ({p.get('object_id', '?')[:8]})"
    if kind == TransitionKind.TILE_MARKED:
        mark = p.get("mark", "?")
        if p.get("remove"):
            return f"tile ({p.get('x')},{p.get('y')}): mark cleared"
        return f"tile ({p.get('x')},{p.get('y')}): «{mark}»"
    if kind == TransitionKind.STRUCTURE_CREATED:
        return f"structure placed: {p.get('name', 'object')} at ({p.get('x')},{p.get('y')})"
    if kind == TransitionKind.STRUCTURE_MODIFIED:
        parts = []
        if "set_passable" in p:
            parts.append("passable" if p["set_passable"] else "blocked")
        if p.get("add_tags"):
            parts.append(f"+tags {p['add_tags']}")
        return f"structure ({p.get('x')},{p.get('y')}): {', '.join(parts) or 'modified'}"
    if kind == TransitionKind.FLUID_CHANGED:
        return f"fluid changed at ({p.get('x', '?')},{p.get('y', '?')})"
    if kind == TransitionKind.EDGE_CREATED:
        ek = p.get("edge_kind") or p.get("kind", "edge")
        return f"relationship: {_ent_name(p.get('source', ''))} —{ek}→ {_ent_name(p.get('target', ''))}"
    if kind == TransitionKind.EDGE_UPDATED:
        ek = p.get("edge_kind") or "interacted"
        return f"relationship updated ({ek})"
    if kind.value == "belief_propagated":
        return f"rumour spread: {(p.get('claim') or '')[:50]}"
    if kind == TransitionKind.WORLD_FACT_REGISTERED:
        fact = p.get("fact") or {}
        claim = fact.get("claim", "")[:60]
        return f"world fact: {claim}"
    if kind == TransitionKind.ENTITY_TAG_CHANGED:
        tag = p.get("tag", "?")
        add = p.get("add", True)
        return f"{_ent_name(p.get('entity_id', ''))}: {'+' if add else '-'}{tag}"
    if kind == TransitionKind.ENVIRONMENT_STATE_CHANGED:
        return f"environment: {p.get('key')}={p.get('value')}"
    if kind == TransitionKind.DIALOGUE_SPOKEN:
        return None  # narration covers speech
    if kind == TransitionKind.WORLD_MARK:
        return None
    return f"{kind.value}"


def format_world_diff(
    before: dict[str, Any],
    after: dict[str, Any],
    transitions: list[Transition],
    world: WorldState,
) -> str:
    """
    Human-readable summary of durable changes this turn.
    Returns empty string if nothing meaningful changed.
    """
    lines: list[str] = []

    # Transition-derived lines (most authoritative)
    seen: set[str] = set()
    for t in transitions:
        desc = _describe_transition(t, world)
        if desc and desc not in seen:
            seen.add(desc)
            lines.append(f"  • {desc}")

    # Entity field diffs (catches propagation not in player event)
    be, ae = before.get("entities") or {}, after.get("entities") or {}
    for eid in sorted(set(be) | set(ae)):
        b, a = be.get(eid) or {}, ae.get(eid) or {}
        name = a.get("name") or b.get("name") or eid[:8]
        for field, label in (("mood", "mood"), ("alert", "alertness"), ("hp", "hp"), ("pos", "position")):
            if b.get(field) != a.get(field) and a.get(field) is not None:
                key = f"{name}:{field}"
                if key not in seen:
                    lines.append(f"  • {name}: {label} {b.get(field, '—')} → {a.get(field)}")

    # New tile marks
    bm, am = before.get("tile_marks") or {}, after.get("tile_marks") or {}
    for k in sorted(set(am) - set(bm)):
        lines.append(f"  • tile {k}: «{am[k]}»")

    # Facts
    if after.get("facts_n", 0) > before.get("facts_n", 0):
        delta = after["facts_n"] - before["facts_n"]
        lines.append(f"  • +{delta} world fact(s) recorded")

    if not lines:
        return ""

    return "── World changed ──\n" + "\n".join(lines[:14])
