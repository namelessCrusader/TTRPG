"""
Dwarf Fortress–style Z-layer navigation and ASCII slice views.

Shared by the terminal REPL (``layer`` command) and the pygame voxel client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .schemas import Coord, EntityId, EntityKind, SpatialGrid, VoxelMaterial

# Top-down glyphs for a horizontal slice at fixed z
_SLICE_SOLID = {
    VoxelMaterial.STONE: "#",
    VoxelMaterial.BRICK: "+",
    VoxelMaterial.WOOD: "W",
    VoxelMaterial.PLANKS: "=",
    VoxelMaterial.GLASS: "G",
    VoxelMaterial.WATER: "~",
    VoxelMaterial.BAR: "B",
    VoxelMaterial.TABLE: "T",
    VoxelMaterial.ROOF: "R",
    VoxelMaterial.THATCH: "H",
    VoxelMaterial.METAL: "M",
    VoxelMaterial.EMBER: "*",
    VoxelMaterial.DIRT: ",",
    VoxelMaterial.GRASS: "\"",
    VoxelMaterial.LEAVES: "l",
}
_DEFAULT_SOLID = "#"


@dataclass
class LayerNavigator:
    """
    Track which Z layer is visible (single slice vs full volume).

    Keys (documented for pygame + REPL help):
      PgUp / ``layer up``   — higher Z
      PgDn / ``layer down`` — lower Z
      Home / ``layer 0``    — ground slice
      End                   — top slice
      0 / ``layer all``     — show all layers (3D view only)
    """

    depth: int
    slice_z: Optional[int] = None  # None = all layers (3D orbit mode)

    def __post_init__(self) -> None:
        if self.depth < 1:
            self.depth = 1
        if self.slice_z is not None:
            self.slice_z = max(0, min(self.depth - 1, self.slice_z))

    @property
    def is_slice_mode(self) -> bool:
        return self.slice_z is not None

    def show_all(self) -> None:
        self.slice_z = None

    def set_z(self, z: int) -> None:
        self.slice_z = max(0, min(self.depth - 1, z))

    def up(self) -> None:
        if self.slice_z is None:
            self.slice_z = 0
        else:
            self.slice_z = min(self.depth - 1, self.slice_z + 1)

    def down(self) -> None:
        if self.slice_z is None:
            self.slice_z = self.depth - 1
        else:
            self.slice_z = max(0, self.slice_z - 1)

    def ground(self) -> None:
        self.slice_z = 0

    def top(self) -> None:
        self.slice_z = self.depth - 1

    def label(self) -> str:
        if self.slice_z is None:
            return f"all layers (0–{self.depth - 1})"
        return f"layer z={self.slice_z} / {self.depth - 1}"

    def handle_repl_command(self, raw: str) -> Optional[str]:
        """
        Parse ``layer``, ``layer N``, ``layer up|down|all|top|ground``.

        Returns a status line if handled, else None.
        """
        parts = raw.lower().strip().split()
        if not parts or parts[0] != "layer":
            return None
        if len(parts) == 1:
            return self.label()
        sub = parts[1]
        if sub == "all":
            self.show_all()
        elif sub == "up":
            self.up()
        elif sub == "down":
            self.down()
        elif sub == "top":
            self.top()
        elif sub in ("ground", "floor", "0"):
            self.ground()
        elif sub.isdigit():
            self.set_z(int(sub))
        else:
            return f"Unknown layer command: {sub!r}"
        return self.label()


def render_layer_ascii(
    grid: SpatialGrid,
    z: int,
    *,
    player_id: Optional[EntityId] = None,
    show_legend: bool = True,
) -> str:
    """Top-down ASCII map of one horizontal slice (Dwarf Fortress style)."""
    if z < 0 or z >= grid.depth:
        return f"(layer z={z} out of range 0..{grid.depth - 1})"

    lines: list[str] = []
    header = f"── Layer z={z}  ({grid.width}×{grid.height}) ──"
    lines.append(header)

    ent_at: dict[tuple[int, int], str] = {}
    for ent in grid.entities.values():
        if not ent.alive or ent.position.z != z:
            continue
        ch = "@"
        if ent.kind == EntityKind.NPC:
            ch = ent.name[:1].upper() if ent.name else "N"
        elif ent.kind == EntityKind.PLAYER:
            ch = "@"
        ent_at[(ent.position.x, ent.position.y)] = ch

    for y in range(grid.height):
        row: list[str] = []
        for x in range(grid.width):
            c = Coord(x=x, y=y, z=z)
            if (x, y) in ent_at:
                row.append(ent_at[(x, y)])
                continue
            if grid.use_voxels or grid.has_voxel(c):
                if grid.is_solid(c):
                    mat = grid.voxel_at(c).material
                    row.append(_SLICE_SOLID.get(mat, _DEFAULT_SOLID))
                else:
                    row.append(".")
            else:
                tile = grid.tile_at(c)
                if not tile.passable:
                    row.append("#")
                else:
                    row.append(".")
        lines.append("".join(row))

    if show_legend:
        lines.append("  @ you  N npc  #+W=… solid  . air/floor")
    return "\n".join(lines)


def layer_help_lines() -> list[str]:
    return [
        "── Z layers (voxel worlds) ──",
        "  layer              — current layer",
        "  layer N            — show slice at height N",
        "  layer up / down    — change slice",
        "  layer ground|top — floor / roof slice",
        "  layer all          — all layers (voxel view)",
    ]
