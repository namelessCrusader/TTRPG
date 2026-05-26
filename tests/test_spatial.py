"""
Tests for M1: Spatial simulation.

Acceptance criteria from the spec:
  - Entity cannot move to a tile occupied by an impassable object
  - Entity A's projection does not include Entity B if B is outside A's LOS
  - Removing an item from inventory of A and adding it to B leaves no copy in A
  - LOS is symmetric (A sees B iff B sees A, given equal sight ranges)
"""

import pytest

from src.sim.schemas import (
    Coord,
    EntityKind,
    EntityState,
    ObjectState,
    PhysicsViolationError,
    TerrainType,
    Tile,
    new_entity_id,
    new_object_id,
)
from src.sim.spatial import (
    drop,
    entities_visible_from,
    has_line_of_sight,
    is_reachable,
    make_walled_room,
    move_entity,
    pick_up,
    place_entity,
    transfer_item,
    visible_from,
)


def _make_entity(pos: Coord, sight: int = 8, kind=EntityKind.PLAYER) -> EntityState:
    return EntityState(
        entity_id=new_entity_id(),
        name="TestEntity",
        kind=kind,
        position=pos,
        sight_range=sight,
    )


# ---------------------------------------------------------------------------
# Placement and collision
# ---------------------------------------------------------------------------


def test_place_entity_on_floor():
    grid = make_walled_room(10, 10)
    e = _make_entity(Coord(x=2, y=2))
    place_entity(grid, e)
    assert e.entity_id in grid.entities


def test_place_entity_on_wall_raises():
    grid = make_walled_room(10, 10)
    e = _make_entity(Coord(x=0, y=0))  # wall
    with pytest.raises(PhysicsViolationError):
        place_entity(grid, e)


def test_place_entity_occupied_raises():
    grid = make_walled_room(10, 10)
    e1 = _make_entity(Coord(x=2, y=2))
    e2 = _make_entity(Coord(x=2, y=2))
    place_entity(grid, e1)
    with pytest.raises(PhysicsViolationError):
        place_entity(grid, e2)


def test_move_entity_to_impassable_raises():
    grid = make_walled_room(10, 10)
    e = _make_entity(Coord(x=2, y=2))
    place_entity(grid, e)
    wall = Coord(x=0, y=0)
    with pytest.raises(PhysicsViolationError):
        move_entity(grid, e.entity_id, wall)


def test_move_entity_success():
    grid = make_walled_room(10, 10)
    e = _make_entity(Coord(x=2, y=2))
    place_entity(grid, e)
    move_entity(grid, e.entity_id, Coord(x=3, y=2))
    assert grid.entities[e.entity_id].position == Coord(x=3, y=2)


def test_move_to_occupied_tile_raises():
    grid = make_walled_room(10, 10)
    e1 = _make_entity(Coord(x=2, y=2))
    e2 = _make_entity(Coord(x=3, y=2), kind=EntityKind.NPC)
    place_entity(grid, e1)
    place_entity(grid, e2)
    with pytest.raises(PhysicsViolationError):
        move_entity(grid, e1.entity_id, Coord(x=3, y=2))


# ---------------------------------------------------------------------------
# Line of sight
# ---------------------------------------------------------------------------


def test_los_open_room():
    grid = make_walled_room(10, 10)
    # No obstacles; LOS should be True across the room
    assert has_line_of_sight(grid, Coord(x=1, y=1), Coord(x=8, y=8))


def test_los_blocked_by_wall():
    grid = make_walled_room(10, 10)
    # Add a solid wall column at x=5
    for y in range(10):
        grid.set_tile(Coord(x=5, y=y), Tile(terrain=TerrainType.WALL))
    assert not has_line_of_sight(grid, Coord(x=2, y=5), Coord(x=8, y=5))


def test_los_symmetry():
    """A sees B iff B sees A (equal sight ranges)."""
    grid = make_walled_room(15, 15)
    e1 = _make_entity(Coord(x=2, y=2), sight=10)
    e2 = _make_entity(Coord(x=8, y=8), sight=10)
    place_entity(grid, e1)
    place_entity(grid, e2)
    visible_from_e1 = visible_from(grid, e1.position, e1.sight_range)
    visible_from_e2 = visible_from(grid, e2.position, e2.sight_range)
    e1_sees_e2 = e2.position in visible_from_e1
    e2_sees_e1 = e1.position in visible_from_e2
    assert e1_sees_e2 == e2_sees_e1


def test_entity_not_in_projection_if_behind_wall():
    """Entity B is behind a wall — A should not see B."""
    grid = make_walled_room(15, 15)
    for y in range(1, 14):
        grid.set_tile(Coord(x=7, y=y), Tile(terrain=TerrainType.WALL))
    e1 = _make_entity(Coord(x=2, y=7), sight=12)
    e2 = _make_entity(Coord(x=12, y=7), sight=12, kind=EntityKind.NPC)
    place_entity(grid, e1)
    place_entity(grid, e2)
    visible = entities_visible_from(grid, e1.entity_id)
    assert e2 not in visible


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


def test_pick_up_item():
    grid = make_walled_room(10, 10)
    e = _make_entity(Coord(x=3, y=3))
    place_entity(grid, e)
    oid = new_object_id()
    obj = ObjectState(object_id=oid, name="Dagger", position=Coord(x=3, y=3))
    grid.objects[oid] = obj
    pick_up(grid, e.entity_id, oid)
    assert oid in grid.entities[e.entity_id].inventory
    assert grid.objects[oid].position is None
    assert grid.objects[oid].owner == e.entity_id


def test_item_not_duplicated_on_transfer():
    """After transfer A→B, A's inventory must not contain the item."""
    grid = make_walled_room(10, 10)
    e1 = _make_entity(Coord(x=3, y=3))
    e2 = _make_entity(Coord(x=4, y=3), kind=EntityKind.NPC)
    place_entity(grid, e1)
    place_entity(grid, e2)
    oid = new_object_id()
    obj = ObjectState(object_id=oid, name="Coin", position=Coord(x=3, y=3))
    grid.objects[oid] = obj
    pick_up(grid, e1.entity_id, oid)
    transfer_item(grid, e1.entity_id, e2.entity_id, oid)
    assert oid not in grid.entities[e1.entity_id].inventory
    assert oid in grid.entities[e2.entity_id].inventory


def test_drop_item():
    grid = make_walled_room(10, 10)
    e = _make_entity(Coord(x=3, y=3))
    place_entity(grid, e)
    oid = new_object_id()
    obj = ObjectState(object_id=oid, name="Key", position=Coord(x=3, y=3))
    grid.objects[oid] = obj
    pick_up(grid, e.entity_id, oid)
    drop(grid, e.entity_id, oid)
    assert oid not in grid.entities[e.entity_id].inventory
    assert grid.objects[oid].position == Coord(x=3, y=3)


# ---------------------------------------------------------------------------
# Pathfinding
# ---------------------------------------------------------------------------


def test_path_through_open_room():
    grid = make_walled_room(10, 10)
    path = is_reachable(grid, Coord(x=1, y=1), Coord(x=8, y=8))
    assert path is True


def test_path_blocked_by_wall():
    grid = make_walled_room(10, 10)
    for y in range(10):
        grid.set_tile(Coord(x=5, y=y), Tile(terrain=TerrainType.WALL))
    assert not is_reachable(grid, Coord(x=1, y=5), Coord(x=8, y=5))
