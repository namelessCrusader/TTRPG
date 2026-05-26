"""3D voxel spatial subsystem tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.sim.schemas import Coord, SpatialGrid, VoxelCell, VoxelMaterial, coord_key, parse_coord_key
from src.sim.spatial import find_path, has_line_of_sight, visible_from
from src.sim.voxel.build import build_tavern_volume
from src.sim.voxel.loader import load_voxel_map
from src.sim.voxel.spatial3d import bresenham_3d, find_path_3d, support_coord_below
from src.sim.world_loader import load_world_pack


def test_coord_z_and_keys():
    c = Coord(x=3, y=4, z=5)
    assert coord_key(c) == "3,4,5"
    assert parse_coord_key("3,4,5") == c
    assert parse_coord_key("3,4") == Coord(x=3, y=4, z=0)
    assert len(c.neighbors_6()) == 6


def test_voxel_grid_passability():
    g = SpatialGrid(width=4, height=4, depth=3, use_voxels=True)
    g.set_voxel(Coord(x=1, y=1, z=0), VoxelCell.from_material(VoxelMaterial.STONE))
    g.set_voxel(Coord(x=1, y=1, z=1), VoxelCell.air())
    assert g.is_solid(Coord(x=1, y=1, z=0))
    assert g.is_passable(Coord(x=1, y=1, z=1))
    assert not g.is_passable(Coord(x=1, y=1, z=0))


def test_build_tavern_volume():
    g = SpatialGrid(width=12, height=10, depth=6, use_voxels=True)
    build_tavern_volume(g)
    assert len(g.voxels) > 100
    assert g.is_solid(Coord(x=0, y=0, z=1))
    assert g.voxel_at(Coord(x=2, y=5, z=1)).material == VoxelMaterial.BAR


def test_3d_pathfinding():
    g = SpatialGrid(width=5, height=5, depth=3, use_voxels=True)
    for x in range(5):
        for y in range(5):
            g.set_voxel(Coord(x=x, y=y, z=0), VoxelCell.from_material(VoxelMaterial.STONE))
    for x in range(1, 4):
        for y in range(1, 4):
            g.set_voxel(Coord(x=x, y=y, z=1), VoxelCell.air())
    start = Coord(x=1, y=1, z=1)
    goal = Coord(x=3, y=3, z=1)
    path = find_path_3d(g, start, goal)
    assert path
    assert path[-1] == goal


def test_visible_from_3d_delegation():
    g = SpatialGrid(width=5, height=5, depth=3, use_voxels=True)
    build_tavern_volume(g) if g.width >= 12 else None
    origin = Coord(x=2, y=2, z=1)
    vis = visible_from(g, origin, sight_range=4)
    assert origin in vis


def test_load_voxel_tavern_pack():
    pack = Path(__file__).resolve().parent.parent / "worlds" / "voxel_tavern"
    if not pack.is_dir():
        pytest.skip("voxel_tavern pack missing")
    world = load_world_pack(pack)
    grid = world.spatial
    assert grid.use_voxels
    assert grid.depth >= 6
    assert len(grid.voxels) > 50
    for ent in grid.entities.values():
        assert 0 <= ent.position.z < grid.depth


def test_bresenham_3d_includes_end():
    ray = bresenham_3d(Coord(x=0, y=0, z=0), Coord(x=3, y=1, z=2))
    assert ray[0] == Coord(x=0, y=0, z=0)
    assert ray[-1] == Coord(x=3, y=1, z=2)
