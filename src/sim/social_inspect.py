"""
Social inspect and belief presentation (Overhaul E).
"""

from __future__ import annotations

from typing import Optional

from .relational import get_relationship_summary
from .schemas import EdgeKind, EntityId, WorldState


def _resolve_entity(world: WorldState, ref: str):
    ref_l = ref.lower().strip()
    for eid, ent in world.all_entities().items():
        if ref_l in ent.name.lower() or ref_l in str(eid).lower():
            return ent, eid
    return None, None


def _beliefs_about(world: WorldState, holder_id: str, subject_id: str) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for edge in world.relational.edges.get(holder_id, {}).get(subject_id, []):
        if edge.kind == EdgeKind.BELIEVES_CLAIM:
            claim = (edge.meta or {}).get("claim", "")
            if claim:
                out.append((claim[:120], round(edge.weight, 2)))
    return out


def _outbound_edges_summary(world: WorldState, eid: str, *, max_lines: int = 8) -> list[str]:
    lines: list[str] = []
    for tgt, edges in world.relational.edges.get(eid, {}).items():
        tgt_ent = world.all_entities().get(EntityId(str(tgt))) or world.spatial.entities.get(EntityId(str(tgt)))
        tname = tgt_ent.name if tgt_ent else str(tgt)[:8]
        for edge in edges:
            if edge.kind == EdgeKind.BELIEVES_CLAIM:
                continue
            ek = edge.kind.value if hasattr(edge.kind, "value") else str(edge.kind)
            lines.append(f"  → {tname}: {ek} ({edge.weight:.2f})")
            if len(lines) >= max_lines:
                return lines
    return lines


def _witnessed_events(world: WorldState, entity_id: str, *, max_events: int = 5) -> list[str]:
    lines: list[str] = []
    eid = EntityId(entity_id)
    for evt in reversed(world.event_log[-40:]):
        if eid in evt.witnesses or evt.action.actor == eid:
            desc = evt.narrative_hint or f"{evt.action.verb} (tick {evt.tick})"
            lines.append(f"  tick {evt.tick}: {desc[:100]}")
        elif isinstance(evt.action.target, str) and evt.action.target == entity_id:
            desc = evt.narrative_hint or f"{evt.action.verb} targeted them"
            lines.append(f"  tick {evt.tick}: {desc[:100]}")
        if len(lines) >= max_events:
            break
    return lines


def _rumours_about_player(world: WorldState, player_id: EntityId) -> list[str]:
    """Beliefs other entities hold that mention or target the player."""
    pid = str(player_id)
    player = world.spatial.entities.get(player_id)
    pname = player.name.lower() if player else ""
    rumours: list[str] = []
    for holder_id, targets in world.relational.edges.items():
        if holder_id == pid:
            continue
        holder = world.all_entities().get(EntityId(holder_id)) or world.spatial.entities.get(EntityId(holder_id))
        hname = holder.name if holder else holder_id[:8]
        for subject_id, edges in targets.items():
            for edge in edges:
                if edge.kind != EdgeKind.BELIEVES_CLAIM:
                    continue
                claim = (edge.meta or {}).get("claim", "")
                if not claim:
                    continue
                cl = claim.lower()
                if subject_id == pid or (pname and pname.split()[0] in cl):
                    fidelity = edge.weight
                    tag = "rumour" if fidelity < 0.85 else "belief"
                    rumours.append(f"  {hname} ({tag}, {fidelity:.0%}): {claim[:100]}")
    return rumours[:8]


def format_social_inspect(
    world: WorldState,
    player_id: Optional[EntityId],
    target_ref: Optional[str] = None,
) -> str:
    lines: list[str] = ["── Social ──"]
    player = world.spatial.entities.get(player_id) if player_id else None
    if player is None:
        return "(no player entity)"

    if target_ref:
        ent, eid = _resolve_entity(world, target_ref)
        if ent is None:
            return f"(no entity matching {target_ref!r})"
        lines.append(f"  {ent.name} — mood={ent.emotional_state.value}, alert={ent.alertness.value}")
        rel = get_relationship_summary(world.relational, str(player_id), str(eid))
        if rel:
            lines.append(f"  Your relationship: {'; '.join(rel[:3])}")
        about_player = _beliefs_about(world, str(eid), str(player_id))
        if about_player:
            lines.append("  They believe about you:")
            for claim, conf in about_player:
                lines.append(f"    • ({conf:.0%}) {claim}")
        you_believe = _beliefs_about(world, str(player_id), str(eid))
        if you_believe:
            lines.append("  You believe about them:")
            for claim, conf in you_believe:
                lines.append(f"    • ({conf:.0%}) {claim}")
        edges = _outbound_edges_summary(world, str(eid))
        if edges:
            lines.append("  Their connections:")
            lines.extend(edges)
        witnessed = _witnessed_events(world, str(eid))
        if witnessed:
            lines.append("  Recent involvement:")
            lines.extend(witnessed)
        return "\n".join(lines)

    rumours = _rumours_about_player(world, player_id)
    if rumours:
        lines.append("  Rumours & beliefs about you:")
        lines.extend(rumours)

    tensions = world.meta.get("active_pressures") or []
    if tensions:
        ids = [str(p.get("id", "?")) for p in tensions if isinstance(p, dict)]
        if ids:
            lines.append(f"  Active pressures: {', '.join(ids[:6])}")

    sc = world.relational
    if sc.edges.get(str(player_id)):
        lines.append("  Your standing:")
        lines.extend(_outbound_edges_summary(world, str(player_id), max_lines=6))

    if len(lines) == 1:
        lines.append("  (no notable social dynamics yet)")
    return "\n".join(lines)
