"""
Canonical state fingerprint for replay verification and experiment traces.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .schemas import WorldState


def world_state_fingerprint(world: WorldState) -> str:
    """
    Deterministic hash of replay-relevant canonical fields.

    Excludes projection/ephemeral meta keys used only for LM steering.
    """
    payload = _canonical_snapshot(world)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _canonical_snapshot(world: WorldState) -> dict[str, Any]:
    entities: dict[str, Any] = {}
    objects: dict[str, Any] = {}
    for region_id in sorted(world.regions.keys()):
        grid = world.regions[region_id].grid
        for eid, ent in sorted(grid.entities.items(), key=lambda x: str(x[0])):
            key = f"{region_id}:{eid}"
            entities[key] = {
                "pos": [ent.position.x, ent.position.y, ent.position.z],
                "health": ent.health,
                "alive": ent.alive,
                "goals": sorted(ent.goals),
                "inv": sorted(str(x) for x in ent.inventory),
            }
        for oid, obj in sorted(grid.objects.items(), key=lambda x: str(x[0])):
            pos = obj.position
            objects[f"{region_id}:{oid}"] = {
                "owner": str(obj.owner) if obj.owner else None,
                "pos": [pos.x, pos.y, pos.z] if pos else None,
            }
    edges: list[list[Any]] = []
    for src, targets in sorted(world.relational.edges.items()):
        for tgt, elist in sorted(targets.items()):
            for e in elist:
                # Belief edges are delayed semantic/cognitive interpretation,
                # not direct replay state for deterministic action playback.
                if e.kind.value == "believes_claim":
                    continue
                edges.append([src, tgt, e.kind.value, round(e.weight, 3)])
    edges.sort()
    return {
        "tick": world.tick,
        "active_region": world.active_region_id,
        "entities": entities,
        "objects": objects,
        "edges": edges,
    }


def durable_state_fingerprint(world: WorldState) -> str:
    """
    Extended hash including world facts and pending scheduled effects.

    Use for save verification — not for event-log replay checks, because
    some subsystems (economy tick, world clock firing) mutate state outside
    the log.  Core replay invariant uses ``world_state_fingerprint``.
    """
    core = _canonical_snapshot(world)
    facts: list[list[Any]] = []
    for fact in sorted(world.world_facts, key=lambda f: (f.established_tick, f.claim)):
        facts.append([
            fact.claim[:120],
            fact.scope.value if hasattr(fact.scope, "value") else str(fact.scope),
            fact.established_tick,
            sorted(fact.tags),
        ])
    scheduled: list[list[Any]] = []
    for effect in sorted(
        world.scheduled_effects,
        key=lambda e: (e.fire_tick, e.rationale or "", e.narration or ""),
    ):
        scheduled.append([
            effect.fire_tick,
            effect.kind.value,
            effect.rationale or "",
            (effect.narration or "")[:80],
        ])
    payload = {**core, "world_facts": facts, "scheduled_effects": scheduled}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
