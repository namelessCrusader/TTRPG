"""Unit cube and optional glTF mesh buffers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Unit cube: 36 vertices (6 faces × 2 tris × 3 verts), position + normal interleaved
def _cube_data() -> np.ndarray:
    faces = [
        # +Y top
        ((0, 1, 0), (0, 1, 0)),
        ((1, 1, 0), (0, 1, 0)),
        ((1, 1, 1), (0, 1, 0)),
        ((0, 1, 0), (0, 1, 0)),
        ((1, 1, 1), (0, 1, 0)),
        ((0, 1, 1), (0, 1, 0)),
        # -Y bottom
        ((0, 0, 0), (0, -1, 0)),
        ((1, 0, 1), (0, -1, 0)),
        ((1, 0, 0), (0, -1, 0)),
        ((0, 0, 0), (0, -1, 0)),
        ((0, 0, 1), (0, -1, 0)),
        ((1, 0, 1), (0, -1, 0)),
        # +X
        ((1, 0, 0), (1, 0, 0)),
        ((1, 1, 1), (1, 0, 0)),
        ((1, 1, 0), (1, 0, 0)),
        ((1, 0, 0), (1, 0, 0)),
        ((1, 0, 1), (1, 0, 0)),
        ((1, 1, 1), (1, 0, 0)),
        # -X
        ((0, 0, 0), (-1, 0, 0)),
        ((0, 1, 0), (-1, 0, 0)),
        ((0, 1, 1), (-1, 0, 0)),
        ((0, 0, 0), (-1, 0, 0)),
        ((0, 1, 1), (-1, 0, 0)),
        ((0, 0, 1), (-1, 0, 0)),
        # +Z
        ((0, 0, 1), (0, 0, 1)),
        ((1, 1, 1), (0, 0, 1)),
        ((1, 0, 1), (0, 0, 1)),
        ((0, 0, 1), (0, 0, 1)),
        ((0, 1, 1), (0, 0, 1)),
        ((1, 1, 1), (0, 0, 1)),
        # -Z
        ((0, 0, 0), (0, 0, -1)),
        ((1, 0, 0), (0, 0, -1)),
        ((0, 1, 0), (0, 0, -1)),
        ((0, 0, 0), (0, 0, -1)),
        ((1, 1, 0), (0, 0, -1)),
        ((1, 0, 0), (0, 0, -1)),
    ]
    verts = []
    for (p, n) in faces:
        verts.extend([*p, *n])
    return np.array(verts, dtype=np.float32)


_CUBE_VERTS = _cube_data()


def unit_cube_vertices() -> np.ndarray:
    """Interleaved [px,py,pz,nx,ny,nz] × 36."""
    return _CUBE_VERTS.copy()


def load_gltf_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load first mesh from glTF/glb (requires pygltflib)."""
    from pygltflib import GLTF2

    gltf = GLTF2().load(str(path))
    if not gltf.meshes:
        raise ValueError(f"No meshes in {path}")

    blob = gltf.binary_blob()
    if blob is None and hasattr(gltf, "glb_data"):
        blob = gltf.glb_data

    prim = gltf.meshes[0].primitives[0]
    acc = gltf.accessors[prim.attributes.POSITION]
    bv = gltf.bufferViews[acc.bufferView]
    start = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    count = acc.count
    verts = np.frombuffer(blob, dtype=np.float32, count=count * 3, offset=start)
    verts = verts.reshape(-1, 3).copy()

    if prim.indices is not None:
        iacc = gltf.accessors[prim.indices]
        ibv = gltf.bufferViews[iacc.bufferView]
        istart = (ibv.byteOffset or 0) + (iacc.byteOffset or 0)
        idtype = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32}[iacc.componentType]
        idx = np.frombuffer(blob, dtype=idtype, count=iacc.count, offset=istart)
        if iacc.componentType == 5121:
            idx = idx.astype(np.uint32)
        elif iacc.componentType == 5123:
            idx = idx.astype(np.uint32)
        tris = idx.reshape(-1, 3)
    else:
        tris = np.arange(len(verts), dtype=np.uint32).reshape(-1, 3)

    return verts, tris


def build_instance_buffer(
    positions: list[tuple[int, int, int]],
    mat_ids: list[int],
) -> np.ndarray:
    """Packed instance data: x,y,z, mat_id per solid voxel."""
    n = len(positions)
    data = np.zeros((n, 4), dtype=np.float32)
    for i, ((x, y, z), mid) in enumerate(zip(positions, mat_ids)):
        data[i] = (float(x), float(y), float(z), float(mid))
    return data
