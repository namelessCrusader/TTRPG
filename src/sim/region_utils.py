"""
Multi-region helpers — entity-scoped grid lookup and region-qualified tile keys.

Prefer these over ``world.spatial`` when logic must work across Castle-style
multi-region packs, not only the player's active region.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Optional

from .schemas import Coord, EntityId, coord_key, parse_coord_key

if TYPE_CHECKING:
    from .schemas import EntityState, SpatialGrid, WorldState


def entity_or_none(world: "WorldState", entity_id: EntityId) -> Optional["EntityState"]:
    try:
        return world.grid_for_entity(entity_id).entities.get(entity_id)
    except KeyError:
        return None


def grid_for(world: "WorldState", entity_id: EntityId) -> "SpatialGrid":
    return world.grid_for_entity(entity_id)


def region_id_for(world: "WorldState", entity_id: EntityId) -> str:
    return world.region_of_entity(entity_id).region_id


def scoped_tile_key(region_id: str, coord: Coord) -> str:
    """Region-qualified tile key for durable facts (avoids cross-region collisions)."""
    return f"{region_id}:{coord_key(coord)}"


def parse_scoped_tile_key(key: str) -> tuple[Optional[str], Coord]:
    """Parse ``region:x,y,z`` or legacy ``x,y,z``."""
    if ":" in key:
        prefix, _, suffix = key.partition(":")
        try:
            return prefix, parse_coord_key(suffix)
        except ValueError:
            pass
    return None, parse_coord_key(key)


def tile_key_for_entity(world: "WorldState", entity_id: EntityId, coord: Coord) -> str:
    rid = region_id_for(world, entity_id)
    return scoped_tile_key(rid, coord)


def iter_entities(
    world: "WorldState",
) -> Iterator[tuple[EntityId, "EntityState", "SpatialGrid"]]:
    for region in world.regions.values():
        grid = region.grid
        for eid, ent in grid.entities.items():
            yield eid, ent, grid


def entities_in_region(world: "WorldState", region_id: str) -> dict[EntityId, "EntityState"]:
    region = world.regions.get(region_id)
    if region is None:
        return {}
    return dict(region.grid.entities)


def witness_entity_ids(world: "WorldState", actor_id: EntityId) -> list[EntityId]:
    """Entities in the actor's region that can see the actor (includes actor)."""
    from .spatial import visible_from

    try:
        grid = world.grid_for_entity(actor_id)
    except KeyError:
        return []

    actor = grid.entities.get(actor_id)
    if actor is None:
        return []

    witnesses: list[EntityId] = [actor_id]
    for eid, entity in grid.entities.items():
        if eid == actor_id:
            continue
        visible = visible_from(grid, entity.position, entity.sight_range)
        if actor.position in visible:
            witnesses.append(eid)
    return witnesses


def entity_name(world: "WorldState", entity_id: str) -> str:
    ent = entity_or_none(world, EntityId(entity_id))
    return ent.name if ent is not None else entity_id


__all__ = [
    "entity_name",
    "entity_or_none",
    "entities_in_region",
    "grid_for",
    "iter_entities",
    "parse_scoped_tile_key",
    "region_id_for",
    "scoped_tile_key",
    "tile_key_for_entity",
    "witness_entity_ids",
]
