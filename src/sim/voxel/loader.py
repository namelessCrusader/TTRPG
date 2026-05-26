"""Load voxel volumes from world pack YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..schemas import Coord, SpatialGrid, Tile, TerrainType, VoxelCell, VoxelMaterial
from .build import build_tavern_volume, fill_floor_slab, stamp_ascii_layer

VOXEL_CHAR_MAP: dict[str, VoxelMaterial] = {
    " ": VoxelMaterial.AIR,
    ".": VoxelMaterial.AIR,
    "#": VoxelMaterial.STONE,
    "+": VoxelMaterial.BRICK,
    "B": VoxelMaterial.BAR,
    "T": VoxelMaterial.TABLE,
    "W": VoxelMaterial.WOOD,
    "P": VoxelMaterial.PLANKS,
    "G": VoxelMaterial.GLASS,
    "~": VoxelMaterial.WATER,
    "=": VoxelMaterial.GLASS,
    "R": VoxelMaterial.ROOF,
    "H": VoxelMaterial.THATCH,
    "M": VoxelMaterial.METAL,
    "L": VoxelMaterial.LEAVES,
    "D": VoxelMaterial.DIRT,
    "F": VoxelMaterial.DIRT,
    "*": VoxelMaterial.EMBER,
}


def load_voxel_map(pack: Path, world_meta: dict) -> SpatialGrid:
    """
    Build a 3D SpatialGrid from pack metadata.

    Supports:
      - ``voxel_map.yaml`` with ``layers`` (list of z + ascii rows)
      - ``spatial.voxel_preset: tavern_3d`` procedural generator
      - Fallback: extrude ``map.txt`` into a shallow 3D shell
    """
    spatial_meta = world_meta.get("spatial") or world_meta
    width = int(spatial_meta.get("width", 24))
    height = int(spatial_meta.get("height", 16))
    depth = int(spatial_meta.get("depth", 12))

    grid = SpatialGrid(
        width=width,
        height=height,
        depth=depth,
        use_voxels=True,
    )

    voxel_file = spatial_meta.get("voxel_file", "voxel_map.yaml")
    voxel_path = pack / voxel_file
    preset = spatial_meta.get("voxel_preset")

    if voxel_path.exists():
        doc = yaml.safe_load(voxel_path.read_text(encoding="utf-8")) or {}
        load_voxel_layers(grid, doc)
    elif preset == "tavern_3d":
        build_tavern_volume(grid)
    else:
        map_path = pack / spatial_meta.get("map_file", "map.txt")
        if map_path.exists():
            rows = _read_ascii_rows(map_path, width, height)
            fill_floor_slab(grid, z=0)
            stamp_ascii_layer(grid, rows, z=1, char_map=VOXEL_CHAR_MAP)
            _extrude_walls(grid, z_min=0, z_max=depth - 1)
        else:
            build_tavern_volume(grid)

    _sync_legacy_tiles(grid)
    return grid


def load_voxel_layers(grid: SpatialGrid, doc: dict) -> None:
    """Apply ``layers: [{z: N, rows: [...]}]`` from voxel_map.yaml."""
    for layer in doc.get("layers") or []:
        z = int(layer.get("z", 0))
        rows = layer.get("rows") or []
        stamp_ascii_layer(grid, rows, z=z, char_map=VOXEL_CHAR_MAP)
    fills = doc.get("fills") or []
    for fill in fills:
        mat = VoxelMaterial(str(fill.get("material", "stone")))
        cell = VoxelCell.from_material(mat)
        x0, x1 = int(fill.get("x0", 0)), int(fill.get("x1", grid.width))
        y0, y1 = int(fill.get("y0", 0)), int(fill.get("y1", grid.height))
        z0, z1 = int(fill.get("z0", 0)), int(fill.get("z1", grid.depth))
        for x in range(x0, min(x1, grid.width)):
            for y in range(y0, min(y1, grid.height)):
                for z in range(z0, min(z1, grid.depth)):
                    grid.set_voxel(Coord(x=x, y=y, z=z), cell)


def _extrude_walls(grid: SpatialGrid, z_min: int, z_max: int) -> None:
    """Copy ground-level solids upward as stone shell."""
    for z in range(z_min + 1, z_max + 1):
        for x in range(grid.width):
            for y in range(grid.height):
                below = Coord(x=x, y=y, z=z - 1)
                if grid.is_solid(below):
                    c = Coord(x=x, y=y, z=z)
                    if not grid.has_voxel(c):
                        grid.set_voxel(c, VoxelCell.from_material(VoxelMaterial.STONE))


def _sync_legacy_tiles(grid: SpatialGrid) -> None:
    """Mirror ground voxels into 2D Tile map for hybrid compiler paths."""
    for x in range(grid.width):
        for y in range(grid.height):
            c = Coord(x=x, y=y, z=0)
            if grid.is_solid(c):
                grid.set_tile(c, Tile(terrain=TerrainType.WALL))
            else:
                grid.set_tile(c, Tile(terrain=TerrainType.FLOOR))


def _read_ascii_rows(path: Path, width: int, height: int) -> list[str]:
    raw = path.read_text(encoding="utf-8").splitlines()
    grid_lines: list[str] = []
    in_grid = False
    for line in raw:
        stripped = line.strip()
        if not in_grid:
            if stripped and len(stripped) >= width - 2:
                in_grid = True
                grid_lines.append(stripped[:width].ljust(width)[:width])
            continue
        if stripped:
            grid_lines.append(stripped[:width].ljust(width)[:width])
    return grid_lines[:height]
