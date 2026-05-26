"""Isometric voxel drawing helpers."""

from src.sim.schemas import Coord, SpatialGrid
from src.sim.voxel.build import build_tavern_volume
from src.sim.voxel_draw import (
    material_faces,
    project_voxel_offset,
    rotate_grid_xy,
    rotate_xy_yaw,
    snap_camera_to,
    voxel_to_iso,
)
from src.sim.voxel_layers import LayerNavigator


def test_rotate_xy_yaw_pivot():
    import math
    x, y = rotate_xy_yaw(1.0, 0.0, math.pi / 2, pivot_x=0.0, pivot_y=0.0)
    assert abs(x) < 1e-6 and abs(y - 1.0) < 1e-6


def test_project_voxel_offset_yaw_changes_basis():
    import math
    x0, y0 = project_voxel_offset(1, 0, 0, view_yaw=0)
    x1, y1 = project_voxel_offset(1, 0, 0, view_yaw=math.pi / 2)
    assert (round(x0), round(y0)) == (20, 10)
    assert (round(x1), round(y1)) == (-10, 20)


def test_rotate_grid_xy_quadrants():
    assert rotate_grid_xy(1, 2, 0, width=8, height=8) == (1, 2)
    assert rotate_grid_xy(1, 2, 1, width=8, height=8) == (2, 6)
    assert rotate_grid_xy(1, 2, 2, width=8, height=8) == (6, 5)


def test_voxel_to_iso_stacks_z():
    x0, y0 = voxel_to_iso(2, 3, 0)
    x1, y1 = voxel_to_iso(2, 3, 2)
    assert y1 < y0


def test_snap_camera_center_fallback():
    g = SpatialGrid(width=8, height=8, depth=4, use_voxels=True)
    build_tavern_volume(g)
    cx, cy = snap_camera_to(g, None, z=1)
    mx, my = voxel_to_iso(4, 4, 1)
    assert cx == mx and cy == my


def test_material_faces_differ_by_shade():
    from src.sim.schemas import VoxelMaterial
    top, left, right, _ = material_faces(VoxelMaterial.BRICK)
    assert top != left != right


def test_draw_slice_includes_floor_markers():
    import pygame
    pygame.init()
    g = SpatialGrid(width=6, height=4, depth=3, use_voxels=True)
    build_tavern_volume(g)
    surf = pygame.Surface((320, 240))
    nav = LayerNavigator(depth=3, slice_z=1)
    from src.sim.voxel_draw import draw_voxel_isometric
    draw_voxel_isometric(surf, g, nav, cam_x=0, cam_y=0)
    # Should have painted something non-background
    assert surf.get_at((160, 120)) != (10, 14, 24)
    pygame.quit()
