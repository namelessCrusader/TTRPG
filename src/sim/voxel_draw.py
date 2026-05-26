"""
Isometric voxel drawing — readable 2.5D blocks (same projection as iso_renderer).

Each solid cell is drawn as a unit cube: top diamond + left/right faces with
material tinting and neighbor culling. Used by voxel_client and iso_renderer.
"""

from __future__ import annotations

import colorsys
import math
from typing import Optional

import pygame

from .schemas import Coord, EntityKind, SpatialGrid, VoxelMaterial
from .voxel.mesher import MATERIAL_COLORS
from .voxel_layers import LayerNavigator

# Slightly smaller than full iso tiles so tavern-scale volumes fit on screen.
VOX_TW = 40
VOX_TH = 20
VOX_BH = 22

C_BG_VOID = (10, 14, 24)
C_FLOOR_DIM = (22, 42, 58)
C_FLOOR_EDGE = (14, 32, 48)
C_EDGE = (8, 10, 16)
C_ENTITY = {
    EntityKind.PLAYER: (255, 220, 60),
    EntityKind.NPC: (255, 110, 90),
    EntityKind.CREATURE: (180, 100, 255),
    EntityKind.AMBIENT: (160, 200, 180),
}


def entity_color(ent) -> tuple[int, int, int]:
    """Stable per-entity tint so NPCs are visually distinct."""
    base = C_ENTITY.get(ent.kind, (200, 200, 220))
    if ent.kind == EntityKind.PLAYER:
        return base
    key = getattr(ent, "entity_id", None) or getattr(ent, "name", "") or ""
    h = sum(ord(c) for c in str(key)) % 360
    # Rotate hue slightly from the NPC base color
    r, g, b = base
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    hh, ss, vv = colorsys.rgb_to_hsv(rf, gf, bf)
    hh = (hh + h / 360.0) % 1.0
    ss = min(1.0, ss + 0.15)
    nr, ng, nb = colorsys.hsv_to_rgb(hh, ss, vv)
    return (int(nr * 255), int(ng * 255), int(nb * 255))


def rotate_grid_xy(
    gx: int,
    gy: int,
    rot: int,
    *,
    width: int,
    height: int,
) -> tuple[int, int]:
    """Rotate grid coords in 90° steps (0–3) for isometric view yaw."""
    r = rot % 4
    if r == 0:
        return gx, gy
    if r == 1:
        return gy, width - 1 - gx
    if r == 2:
        return width - 1 - gx, height - 1 - gy
    return height - 1 - gy, gx


def voxel_to_iso(gx: int, gy: int, gz: int = 0) -> tuple[int, int]:
    """Grid (x,y,z) → isometric offset before camera (top vertex of block)."""
    sx, sy = voxel_to_iso_f(float(gx), float(gy), float(gz))
    return int(sx), int(sy)


def voxel_to_iso_f(gx: float, gy: float, gz: float = 0.0) -> tuple[float, float]:
    """Float grid coords → isometric offset (before camera / zoom)."""
    sx = (gx - gy) * VOX_TW / 2.0
    sy = (gx + gy) * VOX_TH / 2.0 - gz * VOX_BH
    return sx, sy


def rotate_xy_yaw(
    gx: float,
    gy: float,
    yaw: float,
    *,
    pivot_x: float,
    pivot_y: float,
) -> tuple[float, float]:
    """Rotate grid position around a pivot (radians, counter-clockwise)."""
    if abs(yaw) < 1e-6:
        return gx, gy
    dx, dy = gx - pivot_x, gy - pivot_y
    c, s = math.cos(yaw), math.sin(yaw)
    return pivot_x + dx * c - dy * s, pivot_y + dx * s + dy * c


def _rotate_screen_vec(x: float, y: float, yaw: float) -> tuple[float, float]:
    if abs(yaw) < 1e-6:
        return x, y
    c, s = math.cos(yaw), math.sin(yaw)
    return x * c - y * s, x * s + y * c


def projection_basis(
    *,
    view_yaw: float = 0.0,
    view_zoom: float = 1.0,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Screen-space basis vectors for +x, +y, and downward vertical."""
    z = max(0.25, view_zoom)
    ux, uy = VOX_TW * z / 2.0, VOX_TH * z / 2.0
    vx, vy = -VOX_TW * z / 2.0, VOX_TH * z / 2.0
    u = _rotate_screen_vec(ux, uy, view_yaw)
    v = _rotate_screen_vec(vx, vy, view_yaw)
    d = (0.0, VOX_BH * z)
    return u, v, d


def project_voxel_offset(
    gx: float,
    gy: float,
    gz: float = 0.0,
    *,
    view_yaw: float = 0.0,
    view_zoom: float = 1.0,
) -> tuple[float, float]:
    """Project voxel coords into screen offset using the current camera yaw."""
    u, v, d = projection_basis(view_yaw=view_yaw, view_zoom=view_zoom)
    sx = gx * u[0] + gy * v[0] - gz * d[0]
    sy = gx * u[1] + gy * v[1] - gz * d[1]
    return sx, sy


def _shade(base: tuple[int, int, int], bright: float, dim: float) -> tuple[int, int, int]:
    r, g, b = base
    if bright > 0:
        return (
            min(255, int(r + (255 - r) * bright)),
            min(255, int(g + (255 - g) * bright)),
            min(255, int(b + (255 - b) * bright)),
        )
    f = 1.0 + dim  # dim negative
    return (max(0, int(r * f)), max(0, int(g * f)), max(0, int(b * f)))


def material_faces(mat: VoxelMaterial) -> tuple[tuple, tuple, tuple, tuple]:
    base = MATERIAL_COLORS.get(mat, (100, 100, 100))
    top = _shade(base, 0.18, 0)
    right = _shade(base, 0.0, -0.12)
    left = _shade(base, 0.0, -0.28)
    edge = _shade(base, 0.0, -0.45)
    return top, left, right, edge


def _neighbor_solid(grid: SpatialGrid, x: int, y: int, z: int) -> bool:
    return grid.is_solid(Coord(x=x, y=y, z=z))


def _top_vertex(
    gx: int,
    gy: int,
    gz: int,
    cam_x: int,
    cam_y: int,
    view_w: int,
    view_h: int,
    *,
    view_rot: int = 0,
    view_yaw: float = 0.0,
    view_zoom: float = 1.0,
    grid_w: int = 0,
    grid_h: int = 0,
) -> tuple[int, int]:
    gxf, gyf = float(gx), float(gy)
    if view_rot and grid_w and grid_h:
        ix, iy = rotate_grid_xy(gx, gy, view_rot, width=grid_w, height=grid_h)
        gxf, gyf = float(ix), float(iy)
    sx, sy = project_voxel_offset(
        gxf, gyf, float(gz), view_yaw=view_yaw, view_zoom=view_zoom
    )
    tx = int(view_w // 2 + sx - cam_x)
    ty = int(view_h // 3 + sy - cam_y)
    return tx, ty


def _draw_cube(
    surf: pygame.Surface,
    tx: int, ty: int,
    top: tuple, left: tuple, right: tuple, edge: tuple,
    *,
    show_top: bool,
    show_left: bool,
    show_right: bool,
    zoom: float = 1.0,
    view_yaw: float = 0.0,
) -> None:
    """Draw one voxel cube anchored at screen top vertex (tx, ty)."""
    u, v, d = projection_basis(view_yaw=view_yaw, view_zoom=zoom)
    p = (float(tx), float(ty))
    pu = (p[0] + u[0], p[1] + u[1])
    pv = (p[0] + v[0], p[1] + v[1])
    puv = (p[0] + u[0] + v[0], p[1] + u[1] + v[1])
    pud = (pu[0] + d[0], pu[1] + d[1])
    pvd = (pv[0] + d[0], pv[1] + d[1])
    puvd = (puv[0] + d[0], puv[1] + d[1])

    if show_right:
        rf = [pu, puv, puvd, pud]
        pygame.draw.polygon(surf, right, rf)
        pygame.draw.polygon(surf, edge, rf, 1)

    if show_left:
        lf = [pv, puv, puvd, pvd]
        pygame.draw.polygon(surf, left, lf)
        pygame.draw.polygon(surf, edge, lf, 1)

    if show_top:
        top_pts = [p, pu, puv, pv]
        pygame.draw.polygon(surf, top, top_pts)
        pygame.draw.polygon(surf, edge, top_pts, 1)


def _draw_floor_marker(
    surf: pygame.Surface,
    tx: int,
    ty: int,
    *,
    zoom: float = 1.0,
    view_yaw: float = 0.0,
) -> None:
    u, v, _d = projection_basis(view_yaw=view_yaw, view_zoom=zoom)
    p = (float(tx), float(ty))
    pts = [
        p,
        (p[0] + u[0], p[1] + u[1]),
        (p[0] + u[0] + v[0], p[1] + u[1] + v[1]),
        (p[0] + v[0], p[1] + v[1]),
    ]
    pygame.draw.polygon(surf, C_FLOOR_DIM, pts)
    pygame.draw.polygon(surf, C_FLOOR_EDGE, pts, 1)


def snap_camera_to(
    grid: SpatialGrid,
    player_id: Optional[str],
    *,
    z: Optional[int] = None,
) -> tuple[int, int]:
    """Return (cam_x, cam_y) iso offsets to centre on player or map middle."""
    if player_id and player_id in grid.entities:
        p = grid.entities[player_id]
        gz = z if z is not None else p.position.z
        return voxel_to_iso(p.position.x, p.position.y, gz)
    return voxel_to_iso(grid.width // 2, grid.height // 2, z or 0)


def _draw_entity_marker(
    surf: pygame.Surface,
    tx: int,
    ty: int,
    ent,
    *,
    font: Optional[pygame.font.Font] = None,
    zoom: float = 1.0,
    view_yaw: float = 0.0,
) -> None:
    """Draw a small 3D-ish figure plus optional name tag."""
    col = entity_color(ent)
    z = max(0.25, zoom)
    body_h = int((28 if ent.kind == EntityKind.PLAYER else 22) * z)
    _tw, th, bh = int(VOX_TW * z) // 2, int(VOX_TH * z) // 2, body_h
    top = _shade(col, 0.22, 0)
    left = _shade(col, 0.0, -0.2)
    right = _shade(col, 0.0, -0.12)
    edge = _shade(col, 0.0, -0.4)
    _draw_cube(
        surf, tx, ty - bh // 2, top, left, right, edge,
        show_top=True, show_left=True, show_right=True, zoom=z, view_yaw=view_yaw,
    )
    # Facing tick on the ground
    fx, fy = ent.facing.vector if hasattr(ent, "facing") else (0, 1)
    u, v, _d = projection_basis(view_yaw=view_yaw, view_zoom=z)
    vec_x = fx * u[0] + fy * v[0]
    vec_y = fx * u[1] + fy * v[1]
    n = max(1.0, math.hypot(vec_x, vec_y))
    tip_x = tx + int(vec_x / n * 14 * z)
    tip_y = ty + th + int(vec_y / n * 14 * z)
    pygame.draw.line(surf, (240, 240, 255), (tx, ty + th), (tip_x, tip_y), 2)
    if font and getattr(ent, "name", None):
        label = str(ent.name)[:14]
        tag = font.render(label, True, (230, 235, 245))
        bg = pygame.Rect(tx - tag.get_width() // 2 - 2, ty - bh - tag.get_height() - 6,
                         tag.get_width() + 4, tag.get_height() + 2)
        pygame.draw.rect(surf, (8, 12, 22), bg, border_radius=2)
        surf.blit(tag, (bg.x + 2, bg.y + 1))


def draw_voxel_isometric(
    surf: pygame.Surface,
    grid: SpatialGrid,
    layers: LayerNavigator,
    *,
    cam_x: int,
    cam_y: int,
    player_id: Optional[str] = None,
    view_radius: int = 14,
    show_slice_floor: bool = True,
    view_rot: int = 0,
    view_yaw: float = 0.0,
    view_zoom: float = 1.0,
    font: Optional[pygame.font.Font] = None,
) -> None:
    """
    Paint voxels in painter's order.

    slice_z set: only that horizontal layer (+ dim floor markers for air cells).
    slice_z None: full volume (all solid voxels stacked).
    """
    surf.fill(C_BG_VOID)
    vw, vh = surf.get_size()

    player = grid.entities.get(player_id) if player_id else None
    px = player.position.x if player else grid.width // 2
    py = player.position.y if player else grid.height // 2
    z_filter = layers.slice_z
    all_layers = z_filter is None

    z_min = 0 if all_layers else z_filter
    z_max = grid.depth if all_layers else z_filter + 1

    cells: list[tuple[int, int, int]] = []
    for gz in range(z_min, z_max):
        for gx in range(max(0, px - view_radius), min(grid.width, px + view_radius + 1)):
            for gy in range(max(0, py - view_radius), min(grid.height, py + view_radius + 1)):
                c = Coord(x=gx, y=gy, z=gz)
                if grid.is_solid(c):
                    cells.append((gx, gy, gz))
                elif (
                    show_slice_floor
                    and not all_layers
                    and grid.is_in_bounds(c)
                    and (grid.use_voxels or grid.has_voxel(c))
                ):
                    cells.append((gx, gy, gz))  # floor marker pass

    rot = view_rot % 4
    gw, gh = grid.width, grid.height
    zscale = max(0.25, view_zoom)
    focus_z = (
        layers.slice_z
        if layers.slice_z is not None
        else (player.position.z if player else 0)
    )
    base_cam_x, base_cam_y = voxel_to_iso(px, py, focus_z)
    pan_x, pan_y = cam_x - base_cam_x, cam_y - base_cam_y
    focus_x, focus_y = project_voxel_offset(
        float(px), float(py), float(focus_z),
        view_yaw=view_yaw,
        view_zoom=zscale,
    )
    render_cam_x = focus_x + pan_x
    render_cam_y = focus_y + pan_y

    def _sort_key(cell: tuple[int, int, int]) -> tuple[float, int]:
        sx, sy = project_voxel_offset(
            float(cell[0]), float(cell[1]), float(cell[2]),
            view_yaw=view_yaw,
            view_zoom=zscale,
        )
        return sy, cell[2]

    cells.sort(key=_sort_key)

    for gx, gy, gz in cells:
        c = Coord(x=gx, y=gy, z=gz)
        tx, ty = _top_vertex(
            gx, gy, gz, render_cam_x, render_cam_y, vw, vh,
            view_rot=rot,
            view_yaw=view_yaw,
            view_zoom=zscale,
            grid_w=gw,
            grid_h=gh,
        )

        if not grid.is_solid(c):
            _draw_floor_marker(surf, tx, ty, zoom=zscale, view_yaw=view_yaw)
            continue

        mat = grid.voxel_at(c).material
        top, left, right, edge = material_faces(mat)
        show_top = not _neighbor_solid(grid, gx, gy, gz + 1)
        show_right = not _neighbor_solid(grid, gx + 1, gy, gz)
        show_left = not _neighbor_solid(grid, gx, gy + 1, gz)
        _draw_cube(
            surf, tx, ty, top, left, right, edge,
            show_top=show_top,
            show_left=show_left,
            show_right=show_right,
            zoom=zscale,
            view_yaw=view_yaw,
        )

    # Entities (slice: same z only; all-layers: draw at their z)
    ents = [
        e for e in grid.entities.values()
        if e.alive
        and (z_filter is None or e.position.z == z_filter)
        and abs(e.position.x - px) <= view_radius + 2
        and abs(e.position.y - py) <= view_radius + 2
    ]
    ents.sort(key=lambda e: e.position.x + e.position.y + e.position.z)
    for ent in ents:
        tx, ty = _top_vertex(
            ent.position.x, ent.position.y, ent.position.z,
            render_cam_x, render_cam_y, vw, vh,
            view_rot=rot,
            view_yaw=view_yaw,
            view_zoom=zscale,
            grid_w=gw,
            grid_h=gh,
        )
        _draw_entity_marker(
            surf, tx, ty, ent, font=font, zoom=zscale, view_yaw=view_yaw
        )


def draw_layer_banner(
    surf: pygame.Surface,
    layers: LayerNavigator,
    font: Optional[pygame.font.Font],
    *,
    world_name: str = "",
) -> None:
    if font is None:
        return
    label = layers.label()
    text = f"{world_name}  ·  {label}" if world_name else label
    t = font.render(text, True, (190, 210, 220))
    pygame.draw.rect(surf, (12, 18, 30), (8, 8, t.get_width() + 12, t.get_height() + 8))
    surf.blit(t, (14, 12))
