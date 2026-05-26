"""
M1: Canonical spatial simulation.

Responsibilities:
  - Grid construction and tile management
  - Entity placement, movement, collision enforcement
  - Line-of-sight (Bresenham's ray casting)
  - Inventory management (item transfer between entities and ground)
  - Pathfinding (A* for reachability queries)

This module never calls the LM and never reads SemanticProjection.
It is the authoritative spatial layer.
"""

from __future__ import annotations

import heapq
import math
from typing import Callable, Optional

from .schemas import (
    Coord,
    EntityId,
    EntityState,
    FacingDirection,
    FACING_VECTORS,
    ObjectId,
    ObjectState,
    PhysicsViolationError,
    SpatialGrid,
    TerrainType,
    Tile,
)


# ---------------------------------------------------------------------------
# Grid factory helpers
# ---------------------------------------------------------------------------


def make_empty_grid(width: int, height: int, depth: int = 1) -> SpatialGrid:
    """Create a grid filled with floor tiles."""
    return SpatialGrid(width=width, height=height, depth=depth)


def make_walled_room(width: int, height: int) -> SpatialGrid:
    """Create a grid with floor interior and wall border."""
    grid = SpatialGrid(width=width, height=height)
    for x in range(width):
        for y in range(height):
            if x == 0 or x == width - 1 or y == 0 or y == height - 1:
                grid.set_tile(Coord(x=x, y=y), Tile(terrain=TerrainType.WALL))
    return grid


def add_wall(grid: SpatialGrid, coord: Coord) -> None:
    grid.set_tile(coord, Tile(terrain=TerrainType.WALL))


def add_door(grid: SpatialGrid, coord: Coord, *, open: bool = False) -> None:
    terrain = TerrainType.DOOR_OPEN if open else TerrainType.DOOR_CLOSED
    grid.set_tile(coord, Tile(terrain=terrain))


# ---------------------------------------------------------------------------
# Entity management
# ---------------------------------------------------------------------------


def place_entity(grid: SpatialGrid, entity: EntityState) -> None:
    """Place an entity on the grid, enforcing collision."""
    if not grid.is_in_bounds(entity.position):
        raise PhysicsViolationError(
            f"Position {entity.position} is out of bounds for "
            f"{grid.width}x{grid.height}x{grid.depth} grid."
        )
    if not grid.is_passable(entity.position):
        raise PhysicsViolationError(
            f"Cannot place entity '{entity.name}' on impassable tile "
            f"{entity.position}."
        )
    for existing in grid.entities.values():
        if existing.position == entity.position:
            raise PhysicsViolationError(
                f"Position {entity.position} already occupied by "
                f"'{existing.name}'."
            )
    grid.entities[entity.entity_id] = entity


def remove_entity(grid: SpatialGrid, entity_id: EntityId) -> EntityState:
    if entity_id not in grid.entities:
        raise KeyError(f"Entity '{entity_id}' not found in grid.")
    return grid.entities.pop(entity_id)


def move_entity(
    grid: SpatialGrid,
    entity_id: EntityId,
    destination: Coord,
) -> None:
    """
    Move an entity to destination, enforcing passability and collision.

    Raises PhysicsViolationError if the move is illegal.
    """
    entity = grid.entities.get(entity_id)
    if entity is None:
        raise PhysicsViolationError(f"Entity '{entity_id}' not found.")
    if not grid.is_in_bounds(destination):
        raise PhysicsViolationError(
            f"Destination {destination} out of bounds."
        )
    if not grid.tile_at(destination).passable:
        raise PhysicsViolationError(
            f"Tile {destination} is not passable "
            f"(terrain: {grid.tile_at(destination).terrain})."
        )
    for other in grid.entities.values():
        if other.entity_id != entity_id and other.position == destination:
            raise PhysicsViolationError(
                f"Tile {destination} is occupied by '{other.name}'."
            )
    entity.position = destination


# ---------------------------------------------------------------------------
# Line-of-sight (Bresenham's algorithm)
# ---------------------------------------------------------------------------


def _bresenham(start: Coord, end: Coord) -> list[Coord]:
    """Return all integer grid cells on the ray from start to end (inclusive)."""
    x0, y0 = start.x, start.y
    x1, y1 = end.x, end.y
    cells: list[Coord] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy
    while True:
        cells.append(Coord(x=x0, y=y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return cells


def has_line_of_sight(
    grid: SpatialGrid,
    origin: Coord,
    target: Coord,
) -> bool:
    """
    Returns True if origin can see target without an opaque tile blocking the
    ray.  The origin and target tiles themselves are not checked for opacity.
    """
    if grid.depth > 1 or grid.use_voxels:
        from .voxel.spatial3d import has_line_of_sight_3d

        return has_line_of_sight_3d(grid, origin, target)
    ray = _bresenham(origin, target)
    # Exclude first and last cell from opacity checks
    for cell in ray[1:-1]:
        if not grid.is_transparent(cell):
            return False
    return True


def _cone_cos(fov_degrees: float) -> float:
    """Return cos(half_fov) for the given total FOV in degrees.
    A tile is in-cone when its dot-product-over-magnitude ≥ this value."""
    return math.cos(math.radians(fov_degrees / 2.0))


def _in_facing_cone(
    dx: int, dy: int,
    facing: FacingDirection,
    half_cos: float,
) -> bool:
    """Return True if offset (dx, dy) falls within the facing cone.

    The entity tile itself (dx=dy=0) is always visible.
    """
    if dx == 0 and dy == 0:
        return True
    fx, fy = FACING_VECTORS[facing.value]
    dot = dx * fx + dy * fy
    mag = math.sqrt(dx * dx + dy * dy)
    return (dot / mag) >= half_cos


def visible_from(
    grid: SpatialGrid,
    origin: Coord,
    sight_range: int,
    *,
    facing: Optional[FacingDirection] = None,
    fov_degrees: float = 360.0,
) -> set[Coord]:
    """
    Return the set of grid coordinates visible from origin within sight_range.

    When ``facing`` is provided and ``fov_degrees`` < 360, the result is
    clipped to a vision cone centred on the facing direction.  Tiles outside
    the cone are omitted entirely (no partial-transparency effect is needed
    at this level of simulation).

    A ``fov_degrees`` of 360 (the default) restores omnidirectional vision,
    preserving backward compatibility for callers that don't set facing.
    """
    if grid.depth > 1 or grid.use_voxels:
        from .voxel.spatial3d import visible_from_3d

        return visible_from_3d(
            grid, origin, sight_range,
            facing=facing, fov_degrees=fov_degrees,
        )

    omnidirectional = facing is None or fov_degrees >= 360.0
    half_cos = _cone_cos(fov_degrees) if not omnidirectional else 0.0

    visible: set[Coord] = {origin}
    x0, y0 = origin.x, origin.y
    for x in range(max(0, x0 - sight_range), min(grid.width, x0 + sight_range + 1)):
        for y in range(
            max(0, y0 - sight_range), min(grid.height, y0 + sight_range + 1)
        ):
            target = Coord(x=x, y=y)
            dist = origin.manhattan(target)
            if dist > sight_range:
                continue
            if not omnidirectional and not _in_facing_cone(
                x - x0, y - y0, facing, half_cos
            ):
                continue
            if has_line_of_sight(grid, origin, target):
                visible.add(target)
    return visible


def entities_visible_from(
    grid: SpatialGrid,
    origin_entity_id: EntityId,
) -> list[EntityState]:
    """Return all entities visible to the given entity (excludes self),
    respecting the entity's facing direction and FOV cone."""
    origin = grid.entities.get(origin_entity_id)
    if origin is None:
        return []
    visible_coords = visible_from(
        grid, origin.position, origin.sight_range,
        facing=origin.facing, fov_degrees=origin.fov_degrees,
    )
    return [
        e
        for e in grid.entities.values()
        if e.entity_id != origin_entity_id
        and e.position in visible_coords
        and _entity_visible_to_observer(grid, origin, e)
    ]


def _entity_visible_to_observer(
    grid: SpatialGrid,
    observer: EntityState,
    target: EntityState,
) -> bool:
    """Hidden entities require a perception vs stealth contest to spot."""
    if not target.meta.get("hidden"):
        return True
    import hashlib

    stealth = int(target.attributes.get("stealth", 50))
    perception = int(observer.attributes.get("perception", 50))
    alertness_bonus = {
        "unaware": -15, "low": -5, "medium": 0, "high": 15, "combat": 25,
    }.get(observer.alertness.value, 0)
    dist = observer.position.manhattan(target.position)
    target_score = perception + alertness_bonus + dist * 4
    seed = f"spot_{observer.entity_id}_{target.entity_id}_{target.meta.get('hidden_ticks', 0)}"
    digest = hashlib.sha256(seed.encode()).digest()
    noise = (int.from_bytes(digest[:4], "big") / (2**32)) * 40 - 20
    hider_wins = (stealth + 8 + noise) > target_score
    return not hider_wins


# ---------------------------------------------------------------------------
# Pathfinding (A*)
# ---------------------------------------------------------------------------


def find_path(
    grid: SpatialGrid,
    start: Coord,
    goal: Coord,
    *,
    max_steps: int = 256,
    step_cost_fn: Optional[Callable[[Coord], float]] = None,
) -> Optional[list[Coord]]:
    """
    Return the shortest passable path from start to goal, or None if
    unreachable.  The start tile is excluded from the returned path.

    ``step_cost_fn`` adds per-tile traversal penalty (smoke, fire, …) but
    never blocks a passable tile — use it for hazard-aware routing.
    """
    if not grid.is_in_bounds(goal):
        return None

    if grid.depth > 1 or grid.use_voxels:
        from .voxel.spatial3d import find_path_3d

        path = find_path_3d(
            grid, start, goal,
            max_steps=max_steps,
            step_cost_fn=step_cost_fn,
        )
        return path if path else None

    # For pathfinding, temporarily treat the goal tile as passable even if an
    # entity occupies it (we care about reachability, not exact landing).
    def is_walkable(c: Coord) -> bool:
        if c == goal:
            return grid.is_in_bounds(c) and grid.tile_at(c).passable
        return grid.is_passable(c)

    open_heap: list[tuple[float, int, Coord]] = []
    counter = 0
    heapq.heappush(open_heap, (0.0, counter, start))
    from .schemas import coord_key

    came_from: dict[str, Optional[Coord]] = {coord_key(start): None}
    g_score: dict[str, float] = {coord_key(start): 0.0}
    steps = 0

    while open_heap and steps < max_steps:
        steps += 1
        _, _, current = heapq.heappop(open_heap)

        if current == goal:
            path: list[Coord] = []
            node: Optional[Coord] = current
            while node is not None and node != start:
                path.append(node)
                node = came_from[coord_key(node)]
            path.reverse()
            return path

        for neighbor in current.neighbors():
            if not is_walkable(neighbor):
                continue
            key = coord_key(neighbor)
            extra = float(step_cost_fn(neighbor)) if step_cost_fn else 0.0
            tentative = g_score[coord_key(current)] + 1.0 + extra
            if tentative < g_score.get(key, float("inf")):
                came_from[key] = current
                g_score[key] = tentative
                f = tentative + neighbor.manhattan(goal)
                counter += 1
                heapq.heappush(open_heap, (f, counter, neighbor))

    return None


def is_reachable(
    grid: SpatialGrid,
    start: Coord,
    goal: Coord,
    max_steps: int = 256,
) -> bool:
    return find_path(grid, start, goal, max_steps=max_steps) is not None


# ---------------------------------------------------------------------------
# Inventory management
# ---------------------------------------------------------------------------


def pick_up(
    grid: SpatialGrid,
    entity_id: EntityId,
    object_id: ObjectId,
) -> None:
    """
    Transfer an object from the ground to an entity's inventory.
    The object must be on the same tile as the entity.
    """
    entity = grid.entities.get(entity_id)
    if entity is None:
        raise PhysicsViolationError(f"Entity '{entity_id}' not found.")
    obj = grid.objects.get(object_id)
    if obj is None:
        raise PhysicsViolationError(f"Object '{object_id}' not found.")
    if obj.position is None:
        raise PhysicsViolationError(
            f"Object '{object_id}' is already in an inventory."
        )
    if obj.position != entity.position:
        raise PhysicsViolationError(
            f"Object '{object_id}' at {obj.position} is not at "
            f"entity '{entity_id}' position {entity.position}."
        )
    obj.position = None
    obj.owner = entity_id
    if object_id not in entity.inventory:
        entity.inventory.append(object_id)


def drop(
    grid: SpatialGrid,
    entity_id: EntityId,
    object_id: ObjectId,
) -> None:
    """Drop an item from inventory onto the entity's current tile."""
    entity = grid.entities.get(entity_id)
    if entity is None:
        raise PhysicsViolationError(f"Entity '{entity_id}' not found.")
    obj = grid.objects.get(object_id)
    if obj is None:
        raise PhysicsViolationError(f"Object '{object_id}' not found.")
    if object_id not in entity.inventory:
        raise PhysicsViolationError(
            f"Object '{object_id}' is not in entity '{entity_id}' inventory."
        )
    entity.inventory.remove(object_id)
    obj.owner = None
    obj.position = entity.position


def transfer_item(
    grid: SpatialGrid,
    from_entity_id: EntityId,
    to_entity_id: EntityId,
    object_id: ObjectId,
) -> None:
    """
    Transfer an item directly between two entities.
    Both must be present in the grid.
    """
    from_entity = grid.entities.get(from_entity_id)
    to_entity = grid.entities.get(to_entity_id)
    if from_entity is None:
        raise PhysicsViolationError(f"Source entity '{from_entity_id}' not found.")
    if to_entity is None:
        raise PhysicsViolationError(f"Target entity '{to_entity_id}' not found.")
    obj = grid.objects.get(object_id)
    if obj is None:
        raise PhysicsViolationError(f"Object '{object_id}' not found.")
    if object_id not in from_entity.inventory:
        raise PhysicsViolationError(
            f"Object '{object_id}' is not in entity '{from_entity_id}' inventory."
        )
    from_entity.inventory.remove(object_id)
    obj.owner = to_entity_id
    if object_id not in to_entity.inventory:
        to_entity.inventory.append(object_id)


# ---------------------------------------------------------------------------
# Noise / awareness propagation (simple range check)
# ---------------------------------------------------------------------------


def entities_within_hearing(
    grid: SpatialGrid,
    origin: Coord,
    noise_radius: int,
    *,
    exclude: Optional[EntityId] = None,
) -> list[EntityState]:
    """Return entities within Manhattan noise radius of origin."""
    return [
        e
        for e in grid.entities.values()
        if e.entity_id != exclude
        and e.position.manhattan(origin) <= noise_radius
    ]
