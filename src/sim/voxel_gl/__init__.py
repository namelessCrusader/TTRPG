"""GPU voxel renderer (OpenGL + GLSL). Optional: pip install -r requirements-voxel-gl.txt"""

from .renderer import VoxelGLRenderer, gl_available

__all__ = ["VoxelGLRenderer", "gl_available"]
