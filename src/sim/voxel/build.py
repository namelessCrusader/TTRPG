"""Procedural 3D world builders."""

from __future__ import annotations

from ..schemas import Coord, SpatialGrid, VoxelCell, VoxelMaterial


def fill_floor_slab(grid: SpatialGrid, z: int = 0, material: VoxelMaterial = VoxelMaterial.STONE) -> None:
    cell = VoxelCell.from_material(material)
    for x in range(grid.width):
        for y in range(grid.height):
            grid.set_voxel(Coord(x=x, y=y, z=z), cell)


def stamp_ascii_layer(
    grid: SpatialGrid,
    rows: list[str],
    *,
    z: int,
    char_map: dict[str, VoxelMaterial],
) -> None:
    for y, row in enumerate(rows):
        if y >= grid.height:
            break
        for x, ch in enumerate(row[: grid.width]):
            mat = char_map.get(ch, VoxelMaterial.AIR)
            if mat == VoxelMaterial.AIR:
                continue
            grid.set_voxel(Coord(x=x, y=y, z=z), VoxelCell.from_material(mat))


def build_tavern_volume(grid: SpatialGrid) -> None:
    """
    Procedural multi-storey tavern: stone foundation, timber hall, thatch roof,
    bar along west wall, tables, stage loft, open door at south.
    """
    w, h, d = grid.width, grid.height, grid.depth
    fill_floor_slab(grid, z=0)

    # Foundation walls (perimeter shell z=1..4)
    for z in range(1, 5):
        for x in range(w):
            for y in range(h):
                edge = x in (0, w - 1) or y in (0, h - 1)
                if edge:
                    grid.set_voxel(
                        Coord(x=x, y=y, z=z),
                        VoxelCell.from_material(VoxelMaterial.BRICK),
                    )

    # Main hall floor z=1 — walkable planks (not solid; support comes from z=0 stone)
    walk_floor = VoxelCell.from_material(VoxelMaterial.PLANKS)
    walk_floor.solid = False
    walk_floor.transparent = True
    for x in range(1, w - 1):
        for y in range(1, h - 1):
            grid.set_voxel(Coord(x=x, y=y, z=1), walk_floor)

    # Bar counter west side (solid furniture — NPCs stand in front at x=4+)
    for y in range(3, h - 4):
        for z in range(1, 3):
            grid.set_voxel(Coord(x=2, y=y, z=z), VoxelCell.from_material(VoxelMaterial.BAR))
            grid.set_voxel(Coord(x=3, y=y, z=z), VoxelCell.from_material(VoxelMaterial.BAR))
    # Clear walkable strip along bar front
    for y in range(3, h - 4):
        grid.set_voxel(Coord(x=4, y=y, z=1), VoxelCell.air())

    # Tables scattered
    table_spots = [(6, 4), (10, 5), (8, 9), (14, 7), (12, 11)]
    for tx, ty in table_spots:
        if 1 <= tx < w - 1 and 1 <= ty < h - 1:
            grid.set_voxel(Coord(x=tx, y=ty, z=1), VoxelCell.from_material(VoxelMaterial.TABLE))
            grid.set_voxel(Coord(x=tx, y=ty, z=2), VoxelCell.from_material(VoxelMaterial.TABLE))

    # Stage / loft east (raised platform z=2)
    for x in range(w - 7, w - 2):
        for y in range(2, 6):
            grid.set_voxel(Coord(x=x, y=y, z=2), VoxelCell.from_material(VoxelMaterial.PLANKS))
            grid.set_voxel(Coord(x=x, y=y, z=3), VoxelCell.from_material(VoxelMaterial.WOOD))

    # South door opening
    door_x = w // 2
    for z in range(1, 4):
        grid.set_voxel(Coord(x=door_x, y=h - 1, z=z), VoxelCell.air())
        grid.set_voxel(Coord(x=door_x - 1, y=h - 1, z=z), VoxelCell.air())

    # South windows
    for wx in (4, w - 5):
        grid.set_voxel(Coord(x=wx, y=h - 1, z=2), VoxelCell.from_material(VoxelMaterial.GLASS))

    # Roof layers
    roof_z = min(d - 2, 6)
    for x in range(w):
        for y in range(h):
            grid.set_voxel(
                Coord(x=x, y=y, z=roof_z),
                VoxelCell.from_material(VoxelMaterial.ROOF),
            )
    # Thatch peak
    peak_z = min(d - 1, roof_z + 1)
    for x in range(2, w - 2):
        for y in range(2, h - 2):
            grid.set_voxel(
                Coord(x=x, y=y, z=peak_z),
                VoxelCell.from_material(VoxelMaterial.THATCH),
            )

    # Hearth ember center
    hx, hy = w // 2, h // 2
    grid.set_voxel(Coord(x=hx, y=hy, z=1), VoxelCell.from_material(VoxelMaterial.EMBER))
