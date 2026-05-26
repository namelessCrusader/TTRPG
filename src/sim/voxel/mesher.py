"""Greedy mesh extraction for voxel renderer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from ..schemas import Coord, SpatialGrid, VoxelMaterial

# RGBA per material (pleasant tavern palette)
MATERIAL_COLORS: dict[VoxelMaterial, tuple[int, int, int]] = {
    VoxelMaterial.AIR: (0, 0, 0),
    VoxelMaterial.STONE: (90, 88, 82),
    VoxelMaterial.BRICK: (140, 72, 58),
    VoxelMaterial.WOOD: (120, 82, 48),
    VoxelMaterial.PLANKS: (160, 110, 62),
    VoxelMaterial.GLASS: (140, 200, 220),
    VoxelMaterial.WATER: (40, 90, 160),
    VoxelMaterial.GRASS: (60, 140, 55),
    VoxelMaterial.DIRT: (100, 75, 45),
    VoxelMaterial.LEAVES: (50, 120, 45),
    VoxelMaterial.METAL: (160, 165, 175),
    VoxelMaterial.EMBER: (255, 140, 40),
    VoxelMaterial.BAR: (90, 55, 30),
    VoxelMaterial.TABLE: (130, 90, 50),
    VoxelMaterial.THATCH: (180, 150, 70),
    VoxelMaterial.ROOF: (70, 45, 35),
}

FACE_NORMALS = {
    "px": (1, 0, 0),
    "nx": (-1, 0, 0),
    "py": (0, 1, 0),
    "ny": (0, -1, 0),
    "pz": (0, 0, 1),
    "nz": (0, 0, -1),
}


@dataclass
class MeshFace:
    x: int
    y: int
    z: int
    axis: str  # 'x', 'y', or 'z'
    direction: int  # +1 or -1
    width: int
    height: int
    material: VoxelMaterial
    color: tuple[int, int, int]


class GreedyMesh:
    """Extract exposed voxel faces from a SpatialGrid for rendering."""

    def __init__(self, grid: SpatialGrid) -> None:
        self.grid = grid

    def _solid(self, c: Coord) -> bool:
        if not self.grid.is_in_bounds(c):
            return False
        return self.grid.is_solid(c)

    def _material(self, c: Coord) -> VoxelMaterial:
        return self.grid.voxel_at(c).material

    def iter_faces(self) -> Iterator[MeshFace]:
        g = self.grid
        for axis, dims, u_dim, v_dim in (
            ("x", (g.width, g.depth, g.height), "y", "z"),
            ("y", (g.height, g.width, g.depth), "x", "z"),
            ("z", (g.depth, g.width, g.height), "x", "y"),
        ):
            d0, d1, d2 = dims
            for q in range(d0):
                for v in range(d2):
                    mask: list[tuple[VoxelMaterial | None, int]] = []
                    for u in range(d1):
                        if axis == "x":
                            c = Coord(x=q, y=u, z=v)
                            n = Coord(x=q + 1, y=u, z=v)
                        elif axis == "y":
                            c = Coord(x=u, y=q, z=v)
                            n = Coord(x=u, y=q + 1, z=v)
                        else:
                            c = Coord(x=u, y=v, z=q)
                            n = Coord(x=u, y=v, z=q + 1)
                        if self._solid(c) and not self._solid(n):
                            mask.append((self._material(c), 1))
                        elif not self._solid(c) and self._solid(n):
                            mask.append((self._material(n), -1))
                        else:
                            mask.append((None, 0))
                    # Greedy merge rows in mask
                    u = 0
                    while u < d1:
                        mat, direction = mask[u]
                        if mat is None:
                            u += 1
                            continue
                        w = 1
                        while u + w < d1 and mask[u + w] == (mat, direction):
                            w += 1
                        h = 1
                        done = False
                        while v + h < d2 and not done:
                            for k in range(w):
                                if mask[u + k] != (mat, direction):
                                    done = True
                                    break
                            if not done:
                                # check next row in v
                                for k in range(w):
                                    uu = u + k
                                    if axis == "x":
                                        cc = Coord(x=q, y=uu, z=v + h)
                                        nn = Coord(x=q + 1, y=uu, z=v + h)
                                    elif axis == "y":
                                        cc = Coord(x=uu, y=q, z=v + h)
                                        nn = Coord(x=uu, y=q + 1, z=v + h)
                                    else:
                                        cc = Coord(x=uu, y=v + h, z=q)
                                        nn = Coord(x=uu, y=v + h, z=q + 1)
                                    if direction > 0:
                                        ok = self._solid(cc) and not self._solid(nn)
                                        m = self._material(cc) if ok else None
                                    else:
                                        ok = not self._solid(cc) and self._solid(nn)
                                        m = self._material(nn) if ok else None
                                    if m != mat:
                                        done = True
                                        break
                            if not done:
                                h += 1
                        if axis == "x":
                            fx, fy, fz = q + (1 if direction > 0 else 0), u, v
                        elif axis == "y":
                            fx, fy, fz = u, q + (1 if direction > 0 else 0), v
                        else:
                            fx, fy, fz = u, v, q + (1 if direction > 0 else 0)
                        color = MATERIAL_COLORS.get(mat, (128, 128, 128))
                        yield MeshFace(
                            x=fx,
                            y=fy,
                            z=fz,
                            axis=axis,
                            direction=direction,
                            width=w,
                            height=h,
                            material=mat,
                            color=color,
                        )
                        for k in range(w):
                            mask[u + k] = (None, 0)
                        u += w
