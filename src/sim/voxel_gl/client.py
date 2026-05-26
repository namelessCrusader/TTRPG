"""Deprecated: GL view is ``play_client.PlayClient`` with view='gl'."""

from __future__ import annotations

import warnings

from ..play_client import PlayClient as VoxelGLClient


def require_gl() -> None:
    from .renderer import gl_available
    if not gl_available():
        raise ImportError(
            "GPU voxel view needs moderngl. Install:\n"
            "  pip install -r requirements-voxel-gl.txt"
        )


def _warn():
    warnings.warn(
        "voxel_gl.client merged into play_client — use: "
        "python3 -m src.sim.play --pygame --gl",
        DeprecationWarning,
        stacklevel=3,
    )
