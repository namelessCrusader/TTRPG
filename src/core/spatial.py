"""Spatial queries — the one home for grid geometry. Now voxel (x,y,z).

Interface: lift, l1, adjacent, line_of_sight, visible, supported, walk_neighbors,
footprint. 2D positions auto-lift to z=0 everywhere they enter, so 2D-authored
scenes and tests stay valid. Footprint takes size (n×m + orientation in the
signature); v1 implements squares — rotation arrives with the first shaped creature.
"""

from __future__ import annotations

from .state import World, is_solid


def lift(p):
    """(x,y) → (x,y,0); 3-tuples pass through. The 2D-compat shim, in one place."""
    return (p[0], p[1], 0) if len(p) == 2 else tuple(p)


def l1(a, b) -> int:
    a, b = lift(a), lift(b)
    return sum(abs(x - y) for x, y in zip(a, b))


def adjacent(a, b) -> bool:
    return l1(a, b) == 1


def footprint(e) -> set:
    """Cells an entity occupies: n×n square anchored at pos (D&D size categories).
    Signature reserves n×m + orientation; squares until a shaped creature exists."""
    n = int(e.props.get("size", 1))
    x, y, z = lift(e.pos)
    return {(x + i, y + j, z) for i in range(n) for j in range(n)}


def supported(world: World, p) -> bool:
    """Can something stand here? Own floor slab, or solid ground below, or bedrock z=0."""
    x, y, z = lift(p)
    if z == 0:
        return True
    c = world.cell((x, y, z))
    if getattr(c, "floor", None):
        return True
    below = (x, y, z - 1)
    return world.in_bounds(below) and is_solid(world.cell(below).material)


def walk_neighbors(world: World, p) -> list:
    """Where can a walker step from p? In-plane 4-way onto passable supported cells,
    plus up/down through stairs. (Stepping into unsupported air is a CHOICE —
    falling handles it — so unsupported in-plane cells are still returned.)"""
    x, y, z = lift(p)
    out = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        q = (x + dx, y + dy, z)
        if world.in_bounds(q) and not is_solid(world.cell(q).material):
            out.append(q)
    c = world.cell((x, y, z))
    up = (x, y, z + 1)
    if "stairs" in c.tags and world.in_bounds(up) and not is_solid(world.cell(up).material):
        out.append(up)
    down = (x, y, z - 1)
    if world.in_bounds(down) and "stairs" in world.cell(down).tags and not getattr(c, "floor", None):
        out.append(down)
    return out


def bfs_path(world: World, start, goal, adjacent_ok=True) -> list | None:
    """Shortest walkable path start→goal (or adjacent to goal). None if unreachable."""
    from collections import deque
    start, goal = lift(start), lift(goal)
    q, seen = deque([(start, [])]), {start}
    while q:
        pos, path = q.popleft()
        if pos == goal or (adjacent_ok and l1(pos, goal) == 1):
            return path
        for n in walk_neighbors(world, pos):
            if n not in seen:
                seen.add(n)
                q.append((n, path + [n]))
    return None


def line_of_sight(world: World, a, b) -> bool:
    """No solid cell strictly between a and b; floor slabs block sight across
    z-levels (you see downstairs only through openings — burned holes, stairwells)."""
    a, b = lift(a), lift(b)
    if a == b:
        return True
    n = max(abs(b[i] - a[i]) for i in range(3))
    prev = a
    for i in range(1, n + 1):
        t = i / n
        p = tuple(round(a[k] + (b[k] - a[k]) * t) for k in range(3))
        if p not in (a, b):
            if world.in_bounds(p) and is_solid(world.cell(p).material):
                return False
        if p[2] != prev[2]:                          # crossing a level: the higher cell's floor blocks
            hi = p if p[2] > prev[2] else prev
            if world.in_bounds(hi) and getattr(world.cell(hi), "floor", None):
                return False
        prev = p
    return True


def visible(world: World, viewer, target, radius: int) -> bool:
    return l1(viewer, target) <= radius and line_of_sight(world, viewer, target)
