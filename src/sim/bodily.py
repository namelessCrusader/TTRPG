"""
Bodily functions: thirst, bladder pressure, wetness decay, hold-injury.

Runs each world tick after needs_tick (see world_clock.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .schemas import EntityId, Transition, TransitionKind

if TYPE_CHECKING:
    from .schemas import WorldState

BLADDER_CRITICAL = 85.0
BLADDER_ACCIDENT = 95.0
HOLD_INJURY_TICKS = 8
HOLD_DAMAGE_PER_TICK = 3


def bodily_defaults() -> dict[str, float]:
    return {
        "thirst": 70.0,
        "bladder": 10.0,
    }


def bodily_tick(world: "WorldState") -> list[Transition]:
    transitions: list[Transition] = []
    grid = world.spatial

    for entity in grid.entities.values():
        if not entity.alive:
            continue
        eid = str(entity.entity_id)

        bladder = float(entity.stats.get("bladder", 10.0))
        hold = int(entity.stats.get("bladder_hold_ticks", 0))

        # Bladder drains slowly when not desperate (restroom not modeled yet).
        if bladder < 50:
            transitions.append(Transition(
                kind=TransitionKind.NEED_CHANGED,
                payload={
                    "entity_id": eid,
                    "need": "bladder",
                    "delta": -0.5,
                    "cause": "natural_relief",
                },
            ))

        bladder = float(entity.stats.get("bladder", bladder))
        if bladder >= BLADDER_CRITICAL:
            hold += 1
            entity.stats["bladder_hold_ticks"] = float(hold)
            if hold >= HOLD_INJURY_TICKS:
                transitions.append(Transition(
                    kind=TransitionKind.ENTITY_HEALTH_CHANGED,
                    payload={
                        "entity_id": eid,
                        "delta": -HOLD_DAMAGE_PER_TICK,
                        "cause": "bladder_strain",
                    },
                ))
        else:
            entity.stats["bladder_hold_ticks"] = 0.0

        if bladder >= BLADDER_ACCIDENT:
            transitions.append(Transition(
                kind=TransitionKind.FLUID_CHANGED,
                payload={
                    "target_kind": "entity",
                    "entity_id": eid,
                    "material": "water",
                    "volume_ml": 200.0,
                    "cause": "bladder_accident",
                    "wet": True,
                },
            ))
            transitions.append(Transition(
                kind=TransitionKind.NEED_CHANGED,
                payload={
                    "entity_id": eid,
                    "need": "bladder",
                    "delta": 30.0 - bladder,
                    "cause": "bladder_accident",
                },
            ))

        # Wet entities dry slowly.
        if "wet" in entity.tags and world.tick % 5 == 0:
            entity.tags = [t for t in entity.tags if t != "wet"]
            entity.meta.pop("wet_material", None)

    # Wet tiles dry — applies to BOTH the canonical ground field layer
    # and the legacy fluid_spill mirror so the two stay in lockstep.
    from .field_coords import iter_tiles

    for coord, tile in iter_tiles(grid):
        if not tile.env.get("wet") and "wet" not in tile.tags:
            continue
        if world.tick % 6 != 0:
            continue
        spill = tile.env.get("fluid_spill") or {}
        vol = float(spill.get("volume_ml", 0.0)) - 30.0

        # Mirror the evaporation onto the canonical field layer too.
        fields_bucket = tile.env.get("fields")
        ground_layer = (
            fields_bucket.get("ground")
            if isinstance(fields_bucket, dict)
            else None
        )
        if isinstance(ground_layer, dict):
            for sid, entry in list(ground_layer.items()):
                if not isinstance(entry, dict):
                    continue
                meta = dict(entry.get("meta") or {})
                meta_vol = float(meta.get("volume_ml", 0.0))
                if meta_vol <= 0.0:
                    continue
                new_meta_vol = meta_vol - 30.0
                if new_meta_vol <= 0.0:
                    ground_layer.pop(sid, None)
                else:
                    meta["volume_ml"] = new_meta_vol
                    entry["meta"] = meta
                    entry["amount"] = max(
                        0.0,
                        min(100.0, float(entry.get("amount", 0.0)) - 5.0),
                    )
                    ground_layer[sid] = entry
            if ground_layer:
                fields_bucket["ground"] = ground_layer
            else:
                if isinstance(fields_bucket, dict):
                    fields_bucket.pop("ground", None)
                if not fields_bucket:
                    tile.env.pop("fields", None)

        if vol <= 0:
            tile.env.pop("fluid_spill", None)
            tile.env.pop("wet", None)
            tile.tags = [t for t in tile.tags if t != "wet"]
        else:
            spill["volume_ml"] = vol
            tile.env["fluid_spill"] = spill
        grid.set_tile(coord, tile)

    return transitions
