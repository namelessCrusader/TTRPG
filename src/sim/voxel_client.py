"""Deprecated: use ``src.sim.play_client.PlayClient`` (view=voxel)."""

from __future__ import annotations

import warnings

from .play_client import PlayClient as VoxelInteractiveClient

__all__ = ["VoxelInteractiveClient"]


def _warn():
    warnings.warn(
        "voxel_client merged into play_client — use: "
        "python3 -m src.sim.play --pygame",
        DeprecationWarning,
        stacklevel=3,
    )
