"""3D line-of-sight, visibility, and pathfinding on SpatialGrid volumes."""

from __future__ import annotations

import heapq
import math
from typing import Callable, Optional

from ..schemas import Coord, FacingDirection, FACING_VECTORS, SpatialGrid


def bresenham_3d(start: Coord, end: Coord) -> list[Coord]:
    """Integer 3D ray from start to end (inclusive)."""
    x0, y0, z0 = start.x, start.y, start.z
    x1, y1, z1 = end.x, end.y, end.z
    cells: list[Coord] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)
    sx = 1 if x1 >= x0 else -1
    sy = 1 if y1 >= y0 else -1
    sz = 1 if z1 >= z0 else -1
    dm = max(dx, dy, dz)
    if dm == 0:
        return [start]
    err_x = err_y = err_z = dm // 2
    for _ in range(dm + 1):
        cells.append(Coord(x=x0, y=y0, z=z0))
        if x0 == x1 and y0 == y1 and z0 == z1:
            break
        err_x -= dx
        if err_x < 0:
            err_x += dm
            x0 += sx
        err_y -= dy
        if err_y < 0:
            err_y += dm
            y0 += sy
        err_z -= dz
        if err_z < 0:
            err_z += dm
            z0 += sz
    return cells


def has_line_of_sight_3d(
    grid: SpatialGrid,
    origin: Coord,
    target: Coord,
) -> bool:
    ray = bresenham_3d(origin, target)
    for cell in ray[1:-1]:
        if not grid.is_transparent(cell):
            return False
    return True


def _cone_cos(fov_degrees: float) -> float:
    return math.cos(math.radians(fov_degrees / 2.0))


def _in_facing_cone_3d(
    dx: int,
    dy: int,
    dz: int,
    facing: FacingDirection,
    half_cos: float,
) -> bool:
    if dx == dy == dz == 0:
        return True
    fx, fy = FACING_VECTORS[facing.value]
    # Horizontal cone; vertical offset always visible within range
    dot = dx * fx + dy * fy
    mag = math.sqrt(dx * dx + dy * dy)
    if mag < 1e-6:
        return True
    return (dot / mag) >= half_cos


def visible_from_3d(
    grid: SpatialGrid,
    origin: Coord,
    sight_range: int,
    *,
    facing: Optional[FacingDirection] = None,
    fov_degrees: float = 360.0,
) -> set[Coord]:
    omnidirectional = facing is None or fov_degrees >= 360.0
    half_cos = _cone_cos(fov_degrees) if not omnidirectional else 0.0
    visible: set[Coord] = {origin}
    r = sight_range
    for x in range(max(0, origin.x - r), min(grid.width, origin.x + r + 1)):
        for y in range(max(0, origin.y - r), min(grid.height, origin.y + r + 1)):
            for z in range(max(0, origin.z - r), min(grid.depth, origin.z + r + 1)):
                target = Coord(x=x, y=y, z=z)
                if origin.chebyshev(target) > r:
                    continue
                if not omnidirectional and not _in_facing_cone_3d(
                    x - origin.x, y - origin.y, z - origin.z, facing, half_cos
                ):
                    continue
                if has_line_of_sight_3d(grid, origin, target):
                    visible.add(target)
    return visible


def support_coord_below(grid: SpatialGrid, coord: Coord) -> Optional[Coord]:
    """Lowest passable voxel at (x,y) with z <= coord.z that has solid support below."""
    for z in range(coord.z, -1, -1):
        c = Coord(x=coord.x, y=coord.y, z=z)
        if not grid.is_in_bounds(c):
            continue
        if not grid.is_passable(c):
            continue
        below = c.offset(dz=-1)
        if z == 0 or (grid.is_in_bounds(below) and grid.is_solid(below)):
            return c
    return None


def find_path_3d(
    grid: SpatialGrid,
    start: Coord,
    goal: Coord,
    *,
    max_steps: int = 512,
    allow_vertical: bool = True,
    step_cost_fn: Optional[Callable[[Coord], float]] = None,
) -> list[Coord]:
    """A* over 6-connected (or 4-connected) passable voxels.

    ``step_cost_fn`` adds extra cost per tile (smoke, fire, etc.) but never
    makes a passable tile unreachable — hazards are unpleasant, not walls.
    """
    if start == goal:
        return []
    if not grid.is_passable(goal):
        return []

    def neighbors(c: Coord) -> list[Coord]:
        if allow_vertical and grid.depth > 1:
            return c.neighbors_6()
        return c.neighbors()

    def heuristic(a: Coord, b: Coord) -> float:
        return float(a.manhattan(b))

    open_heap: list[tuple[float, int, Coord]] = []
    counter = 0
    heapq.heappush(open_heap, (heuristic(start, goal), counter, start))
    came_from: dict[Coord, Coord] = {}
    g_score: dict[Coord, float] = {start: 0.0}
    closed: set[Coord] = set()

    while open_heap and counter < max_steps * 20:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path: list[Coord] = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
        closed.add(current)
        for nb in neighbors(current):
            if not grid.is_in_bounds(nb) or not grid.is_passable(nb):
                continue
            # Prefer standing on solid ground when moving horizontally
            if allow_vertical and nb.z < current.z:
                sup = support_coord_below(grid, nb)
                if sup is None or sup != nb:
                    continue
            extra = float(step_cost_fn(nb)) if step_cost_fn else 0.0
            tentative = g_score[current] + 1.0 + extra
            if tentative < g_score.get(nb, float("inf")):
                came_from[nb] = current
                g_score[nb] = tentative
                counter += 1
                heapq.heappush(
                    open_heap,
                    (tentative + heuristic(nb, goal), counter, nb),
                )
    return []
