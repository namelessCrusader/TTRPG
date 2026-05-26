"""
Canonical coordinate keys for field / contamination / fluid tile payloads.

All tile-scoped field transitions use ``x``, ``y``, and ``z`` in payloads.
Pending spread maps and dedup keys use ``coord_key`` → ``"x,y,z"``.

Legacy ``"x,y"`` keys and payloads without ``z`` are still accepted on read.
"""

from __future__ import annotations

from typing import Iterator, Optional

from .schemas import Coord, SpatialGrid, Tile, coord_key, parse_coord_key


def coord_from_field_payload(payload: dict) -> Optional[Coord]:
    """Parse ``x`` / ``y`` / optional ``z`` from a transition or spread payload."""
    x, y = payload.get("x"), payload.get("y")
    if x is None or y is None:
        return None
    return Coord(x=int(x), y=int(y), z=int(payload.get("z", 0)))


def attach_field_coords(payload: dict, coord: Coord) -> dict:
    """Write canonical coordinates into a payload dict (mutates and returns)."""
    payload["x"] = coord.x
    payload["y"] = coord.y
    payload["z"] = coord.z
    return payload


def coord_from_tile_key(key: str) -> Coord:
    """Parse a tile map key or pending spread key (``x,y`` or ``x,y,z``)."""
    return parse_coord_key(key)


def field_tile_coord(coord: Coord) -> Coord:
    """
    Horizontal index for ``SpatialGrid.tiles`` / ``tile_at``.

    Substance layers in ``tile.env`` follow the footprint column; ``z`` is
    preserved in transition payloads for 3D worlds and replay hashing.
    """
    return coord.as_2d()


def iter_tiles(grid: SpatialGrid) -> Iterator[tuple[Coord, Tile]]:
    """Yield ``(coord, tile)`` for every entry in the 2D tile map."""
    for key, tile in grid.tiles.items():
        yield coord_from_tile_key(key), tile


def field_spread_neighbors(
    grid: SpatialGrid,
    coord: Coord,
    *,
    airborne: bool,
) -> list[Coord]:
    """Neighbors for substance spread (6-connected air in 3D, 8-horizontal ground)."""
    if airborne and grid.depth > 1:
        return [n for n in coord.neighbors_6() if grid.is_in_bounds(n)]
    out = [n for n in coord.neighbors() if grid.is_in_bounds(n)]
    if grid.depth > 1 and not airborne:
        below = coord.offset(dz=-1)
        if grid.is_in_bounds(below):
            out.append(below)
    return out


def fluid_tile_payload(coord: Coord, **fields) -> dict:
    """Build a FLUID_CHANGED / wet-tile payload with canonical coords."""
    base = {"x": coord.x, "y": coord.y, "z": coord.z}
    base.update(fields)
    return base
