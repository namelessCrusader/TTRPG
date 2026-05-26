"""
Simulation debug snapshots and diffs.

Used by the REPL and autonomous runner with ``--debug`` to show how
NPC internal state and world meta change each tick.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from .schemas import EntityId, EntityKind, WorldState

# Max chars per field in diffs (0 = unlimited)
DEFAULT_MAX_FIELD_CHARS = 0

# World.meta keys surfaced in debug output
_WORLD_META_KEYS = (
    "active_pressures",
    "ambient_probability_mult",
    "scenario_phase",
    "director_scene_focus",
    "director_tension_override",
    "spicy_intimate_phase",
    "castle_dungeon_whispers",
    "debug",
)

# entity.meta keys (omit huge blobs; summarise lists)
_ENTITY_META_KEYS = (
    "social_stimulus",
    "pressure_boosts",
    "last_policy_branch",
    "director_hint",
    "last_spoken_text",
    "last_reply_to",
    "last_reply_tick",
    "observe_streak",
    "beat_first_contact",
    "director_infer_budget",
)


def set_debug_mode(world: WorldState, enabled: bool) -> None:
    world.meta["debug"] = bool(enabled)


def is_debug_mode(world: WorldState) -> bool:
    return bool(world.meta.get("debug"))


def capture_snapshot(world: WorldState) -> dict[str, Any]:
    """Lightweight canonical snapshot for diffing."""
    snap: dict[str, Any] = {
        "tick": world.tick,
        "meta": _pick_meta(world.meta),
        "entities": {},
        "edges": _edge_snapshot(world),
    }
    for eid, ent in world.all_entities().items():
        if not ent.alive and ent.health <= 0:
            continue
        snap["entities"][str(eid)] = _entity_snapshot(ent)
    return snap


def format_snapshot_diff(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    title: str = "state",
) -> str:
    """Human-readable diff between two snapshots."""
    lines: list[str] = [f"  ── debug: {title} (tick {before.get('tick')} → {after.get('tick')}) ──"]

    # World meta
    bm, am = before.get("meta") or {}, after.get("meta") or {}
    for key in sorted(set(bm) | set(am)):
        if bm.get(key) != am.get(key):
            lines.append(
                f"    world.meta[{key!r}]:\n"
                f"        - {_format_value(bm.get(key), DEFAULT_MAX_FIELD_CHARS)}\n"
                f"        + {_format_value(am.get(key), DEFAULT_MAX_FIELD_CHARS)}"
            )

    # Pressures detail when active list changes
    if (bm.get("active_pressures") or []) != (am.get("active_pressures") or []):
        ids = [p.get("id") for p in (am.get("active_pressures") or []) if isinstance(p, dict)]
        if ids:
            lines.append(f"    active pressures: {', '.join(str(i) for i in ids)}")

    # Edges
    be, ae = before.get("edges") or {}, after.get("edges") or {}
    for ek in sorted(set(be) | set(ae)):
        if be.get(ek) != ae.get(ek):
            lines.append(f"    edge {ek}: {be.get(ek, '—')} → {ae.get(ek, '—')}")

    # Entities
    be_ent, ae_ent = before.get("entities") or {}, after.get("entities") or {}
    for eid in sorted(set(be_ent) | set(ae_ent)):
        b, a = be_ent.get(eid), ae_ent.get(eid)
        if b == a:
            continue
        name = (a or b or {}).get("name", eid)
        ent_lines = _entity_diff_lines(b or {}, a or {}, DEFAULT_MAX_FIELD_CHARS)
        if ent_lines:
            lines.append(f"    [{name}]")
            lines.extend(ent_lines)

    if len(lines) == 1:
        lines.append("    (no meta/entity/edge changes)")
    return "\n".join(lines)


def format_entity_state(world: WorldState, entity_ref: str) -> str:
    """Full debug dump for one entity (name or id substring)."""
    ent, eid = _resolve_entity(world, entity_ref)
    if ent is None:
        return f"  (no entity matching {entity_ref!r})"
    lines = [f"  ── {ent.name} ({eid}) ──"]
    lines.append(
        f"    mood={ent.emotional_state.value} alert={ent.alertness.value} "
        f"hp={ent.health}/{ent.max_health} pos={ent.position}"
    )
    if ent.goals:
        lines.append(f"    goals (planner): {ent.goals}")
    snap = _entity_snapshot(ent)
    for k, v in sorted((snap.get("meta") or {}).items()):
        lines.append(f"    meta.{k}: {_short(v)}")
    obs = ent.meta.get("observation_memory") or []
    if obs:
        lines.append(f"    observation_memory ({len(obs)}):")
        for rec in obs[-4:]:
            if isinstance(rec, dict):
                lines.append(
                    f"      tick {rec.get('tick')}: {rec.get('summary', '')}"
                )
    # Outbound edges
    for tgt, edges in world.relational.edges.get(eid, {}).items():
        tgt_ent = world.all_entities().get(tgt) or world.spatial.entities.get(tgt)
        tname = tgt_ent.name if tgt_ent else str(tgt)[:8]
        for e in edges:
            ek = e.kind.value if hasattr(e.kind, "value") else e.kind
            lines.append(f"    → {tname} {ek} w={e.weight:.2f}")
    return "\n".join(lines)


def format_world_debug(world: WorldState) -> str:
    """One-shot world-level debug summary."""
    snap = capture_snapshot(world)
    lines = [
        f"  tick={snap['tick']}  phase={snap['meta'].get('scenario_phase', '—')}",
    ]
    pressures = snap["meta"].get("active_pressures") or []
    if pressures:
        lines.append(
            "  pressures: "
            + ", ".join(
                str(p.get("id", "?")) for p in pressures if isinstance(p, dict)
            )
        )
    sem = world.meta.get("semantic") or {}
    if isinstance(sem, dict) and sem.get("pressure"):
        lines.append(f"  semantic.pressure: {_short(sem['pressure'])}")
    goals_pending = [g.title for g in world.config.goals if not g.completed]
    if goals_pending:
        lines.append(f"  pending goals: {', '.join(goals_pending[:5])}")
    return "\n".join(lines)


def _entity_snapshot(ent) -> dict[str, Any]:
    meta_out: dict[str, Any] = {}
    for k in _ENTITY_META_KEYS:
        if k not in ent.meta:
            continue
        v = ent.meta[k]
        if k == "social_stimulus" and isinstance(v, dict):
            meta_out[k] = {
                "kind": v.get("kind"),
                "summary": (v.get("summary") or "")[:60],
                "addressed": v.get("addressed"),
                "tick": v.get("tick"),
            }
        elif k == "pressure_boosts":
            meta_out[k] = list(v) if v else []
        else:
            meta_out[k] = v
    obs = ent.meta.get("observation_memory")
    if isinstance(obs, list):
        meta_out["observation_memory_n"] = len(obs)
    return {
        "name": ent.name,
        "kind": ent.kind.value if hasattr(ent.kind, "value") else str(ent.kind),
        "mood": ent.emotional_state.value,
        "alert": ent.alertness.value,
        "health": ent.health,
        "pos": f"({ent.position.x},{ent.position.y})",
        "meta": meta_out,
    }


def _entity_diff_lines(
    b: dict, a: dict, max_field_chars: int = DEFAULT_MAX_FIELD_CHARS
) -> list[str]:
    out: list[str] = []
    for field in ("mood", "alert", "health", "pos"):
        if b.get(field) != a.get(field):
            out.append(
                f"      {field}:\n"
                f"        - {b.get(field, '—')}\n"
                f"        + {a.get(field, '—')}"
            )
    bm, am = b.get("meta") or {}, a.get("meta") or {}
    for k in sorted(set(bm) | set(am)):
        if bm.get(k) != am.get(k):
            out.append(
                f"      meta.{k}:\n"
                f"        - {_format_value(bm.get(k), max_field_chars)}\n"
                f"        + {_format_value(am.get(k), max_field_chars)}"
            )
    return out


def _format_value(v: Any, max_chars: int) -> str:
    if isinstance(v, (dict, list)):
        try:
            s = json.dumps(v, ensure_ascii=False, indent=2)
        except TypeError:
            s = repr(v)
    else:
        s = repr(v) if not isinstance(v, str) else v
    if max_chars > 0 and len(s) > max_chars:
        return s[: max_chars - 1] + "…"
    return s


def _edge_snapshot(world: WorldState) -> dict[str, str]:
    out: dict[str, str] = {}
    for src, tgts in world.relational.edges.items():
        src_ent = world.all_entities().get(src) or world.spatial.entities.get(src)
        sname = (src_ent.name if src_ent else str(src)[:6]).replace(" ", "_")
        for tgt, edges in tgts.items():
            tgt_ent = world.all_entities().get(tgt) or world.spatial.entities.get(tgt)
            tname = (tgt_ent.name if tgt_ent else str(tgt)[:6]).replace(" ", "_")
            for e in edges:
                ek = e.kind.value if hasattr(e.kind, "value") else str(e.kind)
                key = f"{sname}→{tname}:{ek}"
                out[key] = f"{e.weight:.2f}"
    return out


def _pick_meta(meta: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in _WORLD_META_KEYS:
        if k in meta:
            out[k] = meta[k]
    sem = meta.get("semantic")
    if isinstance(sem, dict) and sem.get("pressure"):
        out["semantic_pressure"] = sem["pressure"]
    return out


def _short(v: Any, max_len: int = 72) -> str:
    return _format_value(v, max_len if max_len > 0 else 0)


def _resolve_entity(world: WorldState, ref: str):
    ref_l = ref.lower()
    for eid, ent in world.all_entities().items():
        if ref_l in ent.name.lower() or ref_l in str(eid).lower():
            return ent, eid
    return None, None
