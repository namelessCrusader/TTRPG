"""
3D voxel spatial subsystem.

Provides chunk storage, volumetric world building, 3D LOS/pathfinding,
greedy meshing for the renderer, and pack loading for ``voxel_map.yaml``.
"""

from .chunk import VoxelChunk, chunk_key
from .loader import load_voxel_layers, load_voxel_map
from .mesher import GreedyMesh, MeshFace
from .spatial3d import (
    bresenham_3d,
    find_path_3d,
    has_line_of_sight_3d,
    support_coord_below,
    visible_from_3d,
)

__all__ = [
    "VoxelChunk",
    "chunk_key",
    "load_voxel_layers",
    "load_voxel_map",
    "GreedyMesh",
    "MeshFace",
    "bresenham_3d",
    "find_path_3d",
    "has_line_of_sight_3d",
    "support_coord_below",
    "visible_from_3d",
]
