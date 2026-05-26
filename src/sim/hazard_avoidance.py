"""
Environmental hazard assessment for NPC movement.

Distinguishes controlled hearth warmth from runaway fires (oil spills,
room-filling smoke) so tavern hearths do not cause mass panic while
kitchen-fire scenarios do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .fields import MEDIA_TILE_AIR, MEDIA_TILE_GROUND, get_layer, region_atmosphere_amount
from .schemas import Coord, EntityId, EntityState, TerrainType, Transition, TransitionKind, WorldState

if TYPE_CHECKING:
    from .schemas import SpatialGrid

_OIL_FIRE_MIN = 15.0
_SMOKE_FLEE_ATM = 35.0
_SMOKE_MOVE_ATM = 12.0


def _hazard_distance(entity_pos: Coord, threat: Coord) -> int:
    """
    Horizontal distance for floor-level fire/smoke; full 3D otherwise.

    Voxel tavern stores spill/fire on z=0 while patrons walk at z=1.
    """
    if threat.z <= entity_pos.z:
        return max(abs(entity_pos.x - threat.x), abs(entity_pos.y - threat.y))
    return entity_pos.manhattan(threat)


def entity_walk_coord(entity: EntityState, grid: "SpatialGrid") -> Coord:
    """Z-level for pathfinding when the entity sits on a solid floor slab.

    Voxel packs sometimes store NPC positions at z=0 (footprint) even though
    they walk at z=1.  Movement and flee routing must use the passable layer.
    """
    pos = entity.position
    if grid.is_passable(pos):
        return pos
    for z in range(pos.z + 1, grid.depth):
        c = Coord(x=pos.x, y=pos.y, z=z)
        if grid.is_in_bounds(c) and grid.is_passable(c):
            return c
    return pos


@dataclass
class HazardAssessment:
    severity: float = 0.0
    should_flee: bool = False
    should_move_away: bool = False
    threat_coords: list[Coord] = field(default_factory=list)
    exit_coords: list[Coord] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def _oil_amount_at(grid: "SpatialGrid", coord: Coord) -> float:
    tile = grid.tile_at(coord)
    layer = get_layer(tile.env, MEDIA_TILE_GROUND, legacy_tile=tile)
    return float((layer.get("oil") or {}).get("amount", 0))


def _is_controlled_hearth_at(
    grid: "SpatialGrid",
    coord: Coord,
) -> bool:
    """True when *coord* hosts a fixture hearth / heat source, not spilled fuel."""
    if _oil_amount_at(grid, coord) >= _OIL_FIRE_MIN:
        return False
    for obj in grid.objects.values():
        if obj.position != coord:
            continue
        tags = {t.lower() for t in obj.tags}
        if "heat_source" in tags or "hearth" in tags:
            return True
        if "fixture" in tags and ("on_fire" in tags or "wooden" in tags or "wood" in tags):
            return True
    return False


def _dangerous_fire_coords(
    entity: EntityState,
    grid: "SpatialGrid",
    *,
    max_range: int,
) -> list[Coord]:
    out: list[Coord] = []
    z = entity.position.z
    for x in range(max(0, entity.position.x - max_range), min(grid.width, entity.position.x + max_range + 1)):
        for y in range(max(0, entity.position.y - max_range), min(grid.height, entity.position.y + max_range + 1)):
            c = Coord(x=x, y=y, z=z)
            if not grid.is_in_bounds(c):
                continue
            tile = grid.tile_at(c)
            if "on_fire" not in {t.lower() for t in tile.tags}:
                continue
            if _is_controlled_hearth_at(grid, c):
                continue
            out.append(c)
    for obj in grid.objects.values():
        if obj.position is None or "on_fire" not in obj.tags:
            continue
        if entity.position.chebyshev(obj.position) > max_range:
            continue
        if _is_controlled_hearth_at(grid, obj.position):
            continue
        if obj.position not in out:
            out.append(obj.position)
    if "on_fire" in entity.tags:
        out.append(entity.position)
    return out


def _exit_coords(grid: "SpatialGrid", origin: Coord, *, max_range: int = 32) -> list[Coord]:
    exits: list[Coord] = []
    z = origin.z
    for x in range(grid.width):
        for y in range(grid.height):
            c = Coord(x=x, y=y, z=z)
            if origin.manhattan(c) > max_range:
                continue
            tile = grid.tile_at(c)
            if tile.terrain in (
                TerrainType.DOOR_OPEN,
                TerrainType.DOOR_CLOSED,
                TerrainType.STAIRS_UP,
                TerrainType.STAIRS_DOWN,
            ) or "exit" in {t.lower() for t in tile.tags}:
                exits.append(c)

    # Voxel packs carve doorways as passable air on the perimeter rather than
    # DOOR_* terrain tiles — treat those boundary gaps as exits too.
    if not exits and (grid.depth > 1 or grid.use_voxels):
        for x in range(grid.width):
            for y in range(grid.height):
                on_edge = x in (0, grid.width - 1) or y in (0, grid.height - 1)
                if not on_edge:
                    continue
                c = Coord(x=x, y=y, z=z)
                if origin.manhattan(c) > max_range:
                    continue
                if grid.is_passable(c):
                    exits.append(c)
    return exits


def assess_hazards(entity: EntityState, world: WorldState) -> HazardAssessment:
    grid = world.spatial
    smoke = region_atmosphere_amount(world, "smoke")
    sight = max(4, int(entity.sight_range))
    fires = _dangerous_fire_coords(entity, grid, max_range=sight)
    nearest_fire = (
        min(_hazard_distance(entity.position, c) for c in fires) if fires else 99
    )

    result = HazardAssessment(
        threat_coords=list(fires),
        exit_coords=_exit_coords(grid, entity_walk_coord(entity, grid)),
    )

    if "on_fire" in entity.tags:
        result.severity = 1.0
        result.should_flee = True
        result.reasons.append("self on fire")
        return result

    if nearest_fire <= 1:
        result.severity = 0.95
        result.should_flee = True
        result.reasons.append("adjacent fire")
        return result

    if smoke >= _SMOKE_FLEE_ATM:
        result.severity = max(result.severity, 0.9)
        result.should_flee = True
        result.reasons.append("thick smoke")
        return result

    if smoke >= _SMOKE_MOVE_ATM and fires and nearest_fire <= sight:
        result.severity = 0.75
        result.should_flee = True
        result.reasons.append("smoke and fire in room")
        return result

    if fires and nearest_fire <= 4:
        result.severity = 0.5
        result.should_move_away = True
        result.reasons.append("fire nearby")
        return result

    return result


def flee_coord_score(
    coord: Coord,
    *,
    threat_coords: list[Coord],
    exit_coords: list[Coord],
    hostile_coords: list[Coord] | None = None,
) -> float:
    """Higher is better — farther from threats, closer to exits."""
    threats = list(threat_coords)
    if hostile_coords:
        threats.extend(hostile_coords)
    threat_dist = min((_hazard_distance(coord, t) for t in threats), default=12.0)
    exit_dist = min((coord.manhattan(e) for e in exit_coords), default=12.0)
    return threat_dist * 2.5 - exit_dist * 0.4


def tile_traversal_penalty(coord: Coord, world: WorldState) -> float:
    """Extra A* cost for tiles with smoke or fire.

    Hazards make a tile *unpleasant*, not *impassable*.  NPCs can cough
    their way through a smoke-filled common room or dash across burning
    oil to reach the door — they just prefer clearer routes when one exists.
    """
    grid = world.spatial
    if not grid.is_in_bounds(coord):
        return 999.0
    tile = grid.tile_at(coord)
    penalty = 0.0

    air = get_layer(tile.env, MEDIA_TILE_AIR, legacy_tile=tile)
    local_smoke = float((air.get("smoke") or {}).get("amount", 0.0))
    penalty += local_smoke * 0.06

    atm_smoke = region_atmosphere_amount(world, "smoke")
    penalty += atm_smoke * 0.03

    if any(t.lower() == "on_fire" for t in (tile.tags or [])):
        penalty += 6.0

    for obj in grid.objects.values():
        if obj.position == coord and "on_fire" in (obj.tags or []):
            penalty += 4.0

    return penalty


def find_hazard_aware_path(
    grid: "SpatialGrid",
    start: Coord,
    goal: Coord,
    world: WorldState,
    *,
    max_steps: int = 512,
) -> Optional[list[Coord]]:
    """Shortest passable path that *prefers* clear air but can cross smoke/fire."""
    from .spatial import find_path

    return find_path(
        grid,
        start,
        goal,
        max_steps=max_steps,
        step_cost_fn=lambda c: tile_traversal_penalty(c, world),
    )


def flee_plan_destination(entity: EntityState, world: WorldState) -> Optional[Coord]:
    """Pick a flee waypoint: several steps toward the nearest exit through
    smoke if needed, else one step away from threats."""
    grid = world.spatial
    if entity.position is None:
        return None

    walk = entity_walk_coord(entity, grid)
    haz = assess_hazards(entity, world)
    speed = max(1, int(entity.attributes.get("speed", 4)))

    if haz.exit_coords:
        nearest_exit = min(
            haz.exit_coords,
            key=lambda e: walk.manhattan(e),
        )
        path = find_hazard_aware_path(
            grid, walk, nearest_exit, world, max_steps=512,
        )
        if path:
            idx = min(len(path) - 1, speed - 1)
            return path[idx]

    return pick_flee_coord(entity, world)


def movement_hazard_transitions(
    world: WorldState,
    entity_id: EntityId,
    dest: Coord,
) -> list[Transition]:
    """Consequences of stepping into smoke or fire — movement still succeeds."""
    grid = world.spatial
    if not grid.is_in_bounds(dest):
        return []

    tile = grid.tile_at(dest)
    eid = str(entity_id)
    transitions: list[Transition] = []

    on_fire = any(t.lower() == "on_fire" for t in (tile.tags or []))
    for obj in grid.objects.values():
        if obj.position == dest and "on_fire" in (obj.tags or []):
            on_fire = True

    if on_fire:
        transitions.append(Transition(
            kind=TransitionKind.ENTITY_HEALTH_CHANGED,
            payload={
                "entity_id": eid,
                "delta": -2,
                "cause": "walked_through_flames",
            },
        ))

    air = get_layer(tile.env, MEDIA_TILE_AIR, legacy_tile=tile)
    local_smoke = float((air.get("smoke") or {}).get("amount", 0.0))
    atm_smoke = region_atmosphere_amount(world, "smoke")
    effective_smoke = max(local_smoke, atm_smoke * 0.6)

    if effective_smoke >= 15.0:
        transitions.append(Transition(
            kind=TransitionKind.ENTITY_CONDITION_CHANGED,
            payload={
                "entity_id": eid,
                "condition": "coughing",
                "ticks": 2,
            },
        ))
    if effective_smoke >= 30.0:
        transitions.append(Transition(
            kind=TransitionKind.ENTITY_HEALTH_CHANGED,
            payload={
                "entity_id": eid,
                "delta": -1,
                "cause": "smoke_inhalation",
            },
        ))

    return transitions


def pick_flee_coord(entity: EntityState, world: WorldState) -> Coord | None:
    """Best passable step away from hazards — can cross smoke/fire en route."""
    from .schemas import EmotionalState

    grid = world.spatial
    walk = entity_walk_coord(entity, grid)
    haz = assess_hazards(entity, world)
    hostiles = [
        e.position
        for e in grid.entities.values()
        if e.entity_id != entity.entity_id
        and e.emotional_state in (EmotionalState.HOSTILE, EmotionalState.ANGRY)
        and walk.chebyshev(entity_walk_coord(e, grid)) <= entity.sight_range
    ]

    # When panicking, try to step along a hazard-aware path toward an exit
    # instead of only scoring immediate neighbors (which stalls in uniform smoke).
    if haz.should_flee and haz.exit_coords:
        nearest_exit = min(
            haz.exit_coords,
            key=lambda e: walk.manhattan(e),
        )
        path = find_hazard_aware_path(
            grid, walk, nearest_exit, world, max_steps=512,
        )
        if path:
            return path[0]

    candidates = [n for n in walk.neighbors() if grid.is_passable(n)]
    if not candidates:
        return None
    if not haz.threat_coords and not hostiles and not haz.exit_coords:
        return candidates[0]
    return max(
        candidates,
        key=lambda c: flee_coord_score(
            c,
            threat_coords=haz.threat_coords,
            exit_coords=haz.exit_coords,
            hostile_coords=hostiles,
        ),
    )
