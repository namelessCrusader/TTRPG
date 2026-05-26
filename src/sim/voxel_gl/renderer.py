"""OpenGL instanced voxel renderer with GLSL lighting."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..schemas import Coord, SpatialGrid, VoxelMaterial
from ..voxel_layers import LayerNavigator
from .assets import MaterialAsset, load_material_assets
from .camera import OrbitCamera
from .mesh import build_instance_buffer, unit_cube_vertices
from .shaders import INSTANCED_FRAG, INSTANCED_VERT

_GL_OK: Optional[bool] = None


def gl_available() -> bool:
    global _GL_OK
    if _GL_OK is not None:
        return _GL_OK
    try:
        import moderngl  # noqa: F401
        _GL_OK = True
    except ImportError:
        _GL_OK = False
    return _GL_OK


def _material_index_map(assets: dict[VoxelMaterial, MaterialAsset]) -> dict[VoxelMaterial, int]:
    order = sorted(assets.keys(), key=lambda m: m.value)
    return {m: i for i, m in enumerate(order)}


class VoxelGLRenderer:
    """GPU scene: instanced unit cubes with per-material shader colors."""

    MAX_MATS = 32

    def __init__(self, ctx) -> None:
        self.ctx = ctx
        self.assets = load_material_assets()
        self.mat_index = _material_index_map(self.assets)
        self.camera = OrbitCamera()

        self._prog = ctx.program(
            vertex_shader=INSTANCED_VERT,
            fragment_shader=INSTANCED_FRAG,
        )

        cube = unit_cube_vertices()
        self._vbo_cube = ctx.buffer(cube.tobytes())
        self._instance_vbo = ctx.buffer(reserve=4)
        self._vao_inst = self._make_vao(self._instance_vbo)
        self._instance_count = 0

    def _make_vao(self, instance_vbo):
        return self.ctx.vertex_array(
            self._prog,
            [
                (self._vbo_cube, "3f 3f", "in_vertex", "in_normal"),
                (instance_vbo, "3f 1f/i", "i_pos", "i_mat_id"),
            ],
        )

    def rebuild_instances(
        self,
        grid: SpatialGrid,
        layers: Optional[LayerNavigator],
    ) -> None:
        """Upload solid voxel instances from the spatial grid."""
        positions: list[tuple[int, int, int]] = []
        mat_ids: list[int] = []

        z_lo = 0
        z_hi = grid.depth
        if layers is not None:
            z_lo = 0 if layers.slice_z is None else layers.slice_z
            z_hi = grid.depth if layers.slice_z is None else layers.slice_z + 1
        for z in range(z_lo, z_hi):
            for x in range(grid.width):
                for y in range(grid.height):
                    c = Coord(x=x, y=y, z=z)
                    if not grid.is_solid(c):
                        continue
                    mat = grid.voxel_at(c).material
                    if mat == VoxelMaterial.AIR:
                        continue
                    mid = self.mat_index.get(mat, 0)
                    positions.append((x, z, y))
                    mat_ids.append(mid)

        self._instance_count = len(positions)
        if self._instance_count == 0:
            return

        data = build_instance_buffer(positions, mat_ids)
        if self._instance_vbo.size < data.nbytes:
            self._instance_vbo.release()
            self._instance_vbo = self.ctx.buffer(data.tobytes())
            self._vao_inst = self._make_vao(self._instance_vbo)
        else:
            self._instance_vbo.write(data.tobytes())

    def _upload_material_uniforms(self) -> None:
        colors: list[tuple[float, float, float]] = []
        for mat in sorted(self.assets.keys(), key=lambda m: m.value):
            colors.append(self.assets[mat].color)
        n = len(self.assets)
        while len(colors) < self.MAX_MATS:
            colors.append((0.5, 0.5, 0.5))

        self._prog["u_colors"].value = tuple(colors[: self.MAX_MATS])
        self._prog["u_color_count"].value = n

    def draw(
        self,
        *,
        viewport_size: tuple[int, int],
    ) -> None:
        w, h = viewport_size
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.enable(self.ctx.DEPTH_TEST)
        self.ctx.disable(self.ctx.CULL_FACE)
        self.ctx.clear(0.06, 0.08, 0.14, 1.0)

        mvp = self.camera.mvp(w, h)
        # OpenGL expects column-major mat4
        self._upload_material_uniforms()
        self._prog["mvp"].write(np.ascontiguousarray(mvp.T, dtype=np.float32).tobytes())
        light = np.array([0.35, 0.85, 0.45], dtype=np.float32)
        light = light / np.linalg.norm(light)
        self._prog["u_light_dir"].value = tuple(light.tolist())

        if self._instance_count > 0:
            self._vao_inst.render(instances=self._instance_count)

    def focus_grid(self, grid: SpatialGrid) -> None:
        self.camera.focus_volume(grid.width, grid.height, grid.depth)
