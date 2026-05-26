"""Fixed-size voxel chunks for streaming large worlds."""

from __future__ import annotations

from typing import Iterator

from ..schemas import Coord, VoxelCell, VoxelMaterial, coord_key

CHUNK_SIZE = 16


def chunk_key(cx: int, cy: int, cz: int) -> str:
    return f"{cx},{cy},{cz}"


def world_to_chunk_coord(x: int, y: int, z: int) -> tuple[int, int, int, int, int, int]:
    """Return (chunk_x, chunk_y, chunk_z, local_x, local_y, local_z)."""
    cx = x // CHUNK_SIZE
    cy = y // CHUNK_SIZE
    cz = z // CHUNK_SIZE
    lx = x % CHUNK_SIZE
    ly = y % CHUNK_SIZE
    lz = z % CHUNK_SIZE
    return cx, cy, cz, lx, ly, lz


class VoxelChunk:
    """Dense 16×16×16 voxel storage."""

    __slots__ = ("cx", "cy", "cz", "_cells")

    def __init__(self, cx: int = 0, cy: int = 0, cz: int = 0) -> None:
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self._cells: list[VoxelCell | None] = [None] * (CHUNK_SIZE ** 3)

    def _index(self, lx: int, ly: int, lz: int) -> int:
        return lx + CHUNK_SIZE * (ly + CHUNK_SIZE * lz)

    def get_local(self, lx: int, ly: int, lz: int) -> VoxelCell:
        if not (0 <= lx < CHUNK_SIZE and 0 <= ly < CHUNK_SIZE and 0 <= lz < CHUNK_SIZE):
            return VoxelCell.air()
        cell = self._cells[self._index(lx, ly, lz)]
        return cell if cell is not None else VoxelCell.air()

    def set_local(self, lx: int, ly: int, lz: int, cell: VoxelCell) -> None:
        idx = self._index(lx, ly, lz)
        if cell.material == VoxelMaterial.AIR and not cell.solid:
            self._cells[idx] = None
        else:
            self._cells[idx] = cell

    def world_coord(self, lx: int, ly: int, lz: int) -> Coord:
        return Coord(
            x=self.cx * CHUNK_SIZE + lx,
            y=self.cy * CHUNK_SIZE + ly,
            z=self.cz * CHUNK_SIZE + lz,
        )

    def iter_filled(self) -> Iterator[tuple[Coord, VoxelCell]]:
        for lz in range(CHUNK_SIZE):
            for ly in range(CHUNK_SIZE):
                for lx in range(CHUNK_SIZE):
                    cell = self._cells[self._index(lx, ly, lz)]
                    if cell is not None:
                        yield self.world_coord(lx, ly, lz), cell

    def to_flat_dict(self) -> dict[str, VoxelCell]:
        return {coord_key(c): cell for c, cell in self.iter_filled()}


class ChunkStore:
    """Sparse chunk map backing a SpatialGrid-sized volume."""

    def __init__(self) -> None:
        self.chunks: dict[str, VoxelChunk] = {}

    def chunk_at(self, cx: int, cy: int, cz: int) -> VoxelChunk:
        key = chunk_key(cx, cy, cz)
        if key not in self.chunks:
            self.chunks[key] = VoxelChunk(cx, cy, cz)
        return self.chunks[key]

    def get(self, coord: Coord) -> VoxelCell:
        cx, cy, cz, lx, ly, lz = world_to_chunk_coord(coord.x, coord.y, coord.z)
        return self.chunk_at(cx, cy, cz).get_local(lx, ly, lz)

    def set(self, coord: Coord, cell: VoxelCell) -> None:
        cx, cy, cz, lx, ly, lz = world_to_chunk_coord(coord.x, coord.y, coord.z)
        self.chunk_at(cx, cy, cz).set_local(lx, ly, lz, cell)

    def flush_to_grid(self, grid) -> None:
        """Copy all chunk voxels into a SpatialGrid.voxels dict."""
        for chunk in self.chunks.values():
            for coord, cell in chunk.iter_filled():
                grid.set_voxel(coord, cell)
