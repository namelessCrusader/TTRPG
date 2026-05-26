"""
Unified substance / physical-state helpers.

Single ontology for materials, temperature, tags, and hazards — used by
physics, spells, projection, and NPC cognition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

from .schemas import EntityId, EntityState, ObjectId, ObjectState

if TYPE_CHECKING:
    from .physics import PhysicsConfig
    from .schemas import SpatialGrid, WorldState

EntityLike = Union[EntityState, ObjectState]


def resolve_material(
    subject: EntityLike,
    config: Optional["PhysicsConfig"],
) -> Optional[Any]:
    """Return MaterialDef for an entity or object, or None."""
    if config is None:
        return None
    from .physics import _entity_material, _object_material

    if isinstance(subject, EntityState):
        return _entity_material(subject, config)
    return _object_material(subject, config)


# Sane bounds for living targets — prevents runaway spell stacking.
_TEMPERATURE_MIN = -40.0
_TEMPERATURE_MAX = 320.0


def clamp_temperature(value: float) -> float:
    return max(_TEMPERATURE_MIN, min(_TEMPERATURE_MAX, float(value)))


def get_temperature(
    subject: EntityLike,
    config: Optional["PhysicsConfig"],
) -> float:
    ambient = config.ambient_temp if config else 20.0
    raw = float(subject.meta.get("temperature", ambient))
    clamped = clamp_temperature(raw)
    if clamped != raw:
        subject.meta["temperature"] = clamped
    return clamped


def apply_temperature_delta(
    subject: EntityLike,
    delta: float,
    config: Optional["PhysicsConfig"],
) -> float:
    """Apply a temperature change with clamping; return the delta actually applied."""
    before = get_temperature(subject, config)
    after = clamp_temperature(before + float(delta))
    subject.meta["temperature"] = after
    return after - before


def can_ignite(subject: EntityLike, config: Optional["PhysicsConfig"]) -> bool:
    """True if target can catch fire (flammable tag or material ignition threshold)."""
    if "on_fire" in subject.tags:
        return False
    if "flammable" in subject.tags or "wooden" in subject.tags:
        return True
    mat = resolve_material(subject, config)
    if mat is None:
        return False
    if mat.ignition_point is not None:
        temp = get_temperature(subject, config)
        if temp >= mat.ignition_point * 0.85:
            return True
        if "flammable" in (mat.tags or []):
            return True
    return False


def describe_physical_state(
    subject: EntityLike,
    config: Optional["PhysicsConfig"],
    *,
    ambient: Optional[float] = None,
) -> str:
    """Short phrase for projection (token-capped)."""
    parts: list[str] = []
    if "on_fire" in subject.tags:
        parts.append("burning")
    if "frozen" in subject.tags:
        parts.append("frozen")
    if "wet" in subject.tags:
        parts.append("soaked")
    from .fluids import describe_fluid, fluid_capacity_ml

    if isinstance(subject, ObjectState) and fluid_capacity_ml(subject) > 0:
        catalog = None
        fill = describe_fluid(subject, catalog)
        if fill:
            parts.append(fill)
    if "boiling" in subject.tags:
        parts.append("boiling")

    amb = ambient if ambient is not None else (
        config.ambient_temp if config else 20.0
    )
    temp = get_temperature(subject, config)
    delta = temp - amb
    if delta >= 80:
        parts.append("scorching hot")
    elif delta >= 35:
        parts.append("very warm")
    elif delta >= 12:
        parts.append("warm")
    elif delta <= -25:
        parts.append("freezing cold")
    elif delta <= -8:
        parts.append("chilled")

    mat = resolve_material(subject, config)
    if mat and mat.display and not parts:
        parts.append(mat.display.lower())

    for cond in ("poisoned", "stunned", "hypothermic"):
        if cond in getattr(subject, "conditions", {}):
            parts.append(cond)

    if isinstance(subject, EntityState):
        from .contamination import describe_entity_contamination

        catalog = None
        sick_note = describe_entity_contamination(subject, catalog)
        if sick_note:
            parts.append(sick_note)
        stats = getattr(subject, "stats", {}) or {}
        bladder = float(stats.get("bladder", 0))
        if bladder >= 85:
            parts.append("desperate to pee")
        elif bladder >= 60:
            parts.append("restless")

    return ", ".join(parts[:5]) if parts else ""


def queue_impulse(
    world: "WorldState",
    target_id: str,
    prop: str,
    delta: float,
    *,
    source: str = "unknown",
    target_kind: str = "entity",
) -> None:
    """Queue a physics impulse applied at the start of the next physics_tick."""
    bucket = world.meta.setdefault("physics_impulses", [])
    bucket.append({
        "target_id": target_id,
        "target_kind": target_kind,
        "prop": prop,
        "delta": float(delta),
        "source": source,
        "tick": world.tick,
    })


def drain_impulse_queue(
    world: "WorldState",
    config: "PhysicsConfig",
) -> list:
    """Apply queued impulses; return transitions."""
    from .schemas import Transition, TransitionKind

    raw = world.meta.pop("physics_impulses", [])
    if not raw:
        return []

    grid = world.spatial
    transitions: list[Transition] = []
    for item in raw:
        tid = str(item.get("target_id", ""))
        prop = str(item.get("prop", "temperature"))
        delta = float(item.get("delta", 0))
        kind = str(item.get("target_kind", "entity"))
        source = str(item.get("source", "impulse"))

        if kind == "object":
            obj = grid.objects.get(ObjectId(tid))
            if obj is None:
                continue
            current = float(obj.meta.get(prop, config.ambient_temp if prop == "temperature" else 0))
            obj.meta[prop] = current + delta
            transitions.append(Transition(
                kind=TransitionKind.ENTITY_PROPERTY_CHANGED,
                payload={
                    "entity_id": tid,
                    "object_id": tid,
                    "prop": prop,
                    "delta": delta,
                    "cause": source,
                    "_object": True,
                },
            ))
        else:
            ent = grid.entities.get(EntityId(tid))
            if ent is None:
                continue
            if prop == "temperature":
                applied = apply_temperature_delta(ent, delta, config)
                delta = applied
            else:
                current = float(ent.meta.get(prop, 0))
                ent.meta[prop] = current + delta
            transitions.append(Transition(
                kind=TransitionKind.ENTITY_HEALTH_CHANGED,
                payload={
                    "entity_id": tid,
                    "delta": 0,
                    "prop_delta": {prop: delta},
                    "cause": source,
                },
            ))
    return transitions


def build_room_physical_notes(
    world: "WorldState",
    focal: EntityState,
    visible_coords: set,
    config: Optional["PhysicsConfig"],
) -> list[str]:
    """Ambient physical cues for EnvironmentSummary (hearth, smoke, cold draft)."""
    notes: list[str] = []
    grid = world.spatial
    amb = config.ambient_temp if config else 20.0

    warm_sources = 0
    fires = 0
    for ent in grid.entities.values():
        if not ent.alive or ent.position not in visible_coords:
            continue
        if ent.entity_id == focal.entity_id:
            continue
        if "on_fire" in ent.tags:
            fires += 1
        t = get_temperature(ent, config)
        if t > amb + 25:
            warm_sources += 1

    for obj in grid.objects.values():
        if obj.position is None or obj.position not in visible_coords:
            continue
        if "on_fire" in obj.tags:
            fires += 1
        t = get_temperature(obj, config)
        if t > amb + 40:
            warm_sources += 1

    hearth_fixtures = sum(
        1
        for obj in grid.objects.values()
        if obj.position
        and obj.position in visible_coords
        and "on_fire" in obj.tags
        and ("heat_source" in obj.tags or "fixture" in obj.tags)
    )
    entity_on_fire = any(
        ent.alive
        and ent.position in visible_coords
        and "on_fire" in ent.tags
        for ent in grid.entities.values()
    )
    wild_object_fire = any(
        obj.position
        and obj.position in visible_coords
        and "on_fire" in obj.tags
        and "heat_source" not in obj.tags
        and "fixture" not in obj.tags
        for obj in grid.objects.values()
    )
    if entity_on_fire or wild_object_fire:
        notes.append("smoke and dangerous heat in the air")
    elif hearth_fixtures >= 1 or warm_sources >= 1:
        notes.append("warm hearth glow and comfortable heat")
    if world.meta.get("cold_draft"):
        notes.append("cold draft from the door")

    from .field_coords import coord_from_tile_key, iter_tiles

    from .fields import MEDIA_TILE_GROUND, get_layer as _get_layer

    for c, tile in iter_tiles(grid):
        if c not in visible_coords:
            continue
        # Prefer the canonical ground field layer, fall back to legacy mirror.
        ground_layer = _get_layer(tile.env, MEDIA_TILE_GROUND, legacy_tile=tile)
        wet_mat = None
        wet_volume = 0.0
        for sid, entry in ground_layer.items():
            if not isinstance(entry, dict):
                continue
            meta = entry.get("meta") or {}
            vol = float(meta.get("volume_ml", 0.0))
            if vol > wet_volume:
                wet_volume = vol
                wet_mat = sid
        if wet_mat is None and (tile.env.get("wet") or tile.env.get("fluid_spill")):
            wet_mat = (tile.env.get("fluid_spill") or {}).get("material", "liquid")
        if wet_mat:
            notes.append(f"the floor is wet with {wet_mat}")
            break

    # Region atmosphere fields (smoke, odor, alcohol fumes …)
    region = world.regions.get(world.active_region_id) if world.regions else None
    atm = (region.grid.region_env.get("fields") or {}).get("atmosphere", {}) if region else {}
    sub_catalog = world.config.substance_catalog if world.config else None
    for sub_id, data in atm.items():
        amt = float(data.get("amount", 0))
        if amt < 8.0:
            continue
        if sub_catalog:
            sdef = sub_catalog.get(sub_id)
            name = sdef.display if sdef else sub_id
        else:
            name = sub_id
        if len(notes) < 5:
            notes.append(f"{name} fills the air")

    from .contamination import describe_tile_contamination

    catalog = world.config.agent_catalog if world.config else None
    for c, tile in iter_tiles(grid):
        if c not in visible_coords:
            continue
        for line in describe_tile_contamination(tile, catalog):
            notes.append(line)
            if len(notes) >= 5:
                return notes[:5]

    return notes[:5]


def visible_object_physical_lines(
    world: "WorldState",
    focal: EntityState,
    visible_coords: set,
    config: Optional["PhysicsConfig"],
    *,
    max_lines: int = 6,
) -> list[str]:
    """Physical state of notable ground objects in view."""
    lines: list[str] = []
    grid = world.spatial
    for obj in sorted(grid.objects.values(), key=lambda o: str(o.object_id)):
        if obj.position is None or obj.position not in visible_coords:
            continue
        state = describe_physical_state(obj, config)
        from .consequences import traces_for_projection

        traces = traces_for_projection(obj.meta, max_items=1)
        if traces:
            state = (state + "; " if state else "") + f"recently: {traces[0]}"
        if not state and "on_fire" not in obj.tags:
            continue
        dist = focal.position.manhattan(obj.position)
        rel = "nearby" if dist <= 4 else "across the room"
        label = state or "unusual"
        lines.append(f"{obj.name} ({rel}): {label}")
        if len(lines) >= max_lines:
            break
    return lines
