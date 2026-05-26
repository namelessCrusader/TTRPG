"""Load material manifest and optional glTF meshes from assets/voxels/."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from ..schemas import VoxelMaterial
from ..voxel.mesher import MATERIAL_COLORS

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MANIFEST = _REPO_ROOT / "assets" / "voxels" / "manifest.yaml"


@dataclass
class MaterialAsset:
    material: VoxelMaterial
    color: tuple[float, float, float]  # 0–1 RGB
    roughness: float = 0.75
    metallic: float = 0.0
    scale: float = 1.0
    model_path: Optional[Path] = None
    # Loaded mesh: Nx3 vertices, Mx3 uint32 indices (object-local space)
    vertices: Optional[np.ndarray] = field(default=None, repr=False)
    indices: Optional[np.ndarray] = field(default=None, repr=False)


def _rgb255_to_unit(rgb: list) -> tuple[float, float, float]:
    return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)


def _fallback_color(mat: VoxelMaterial) -> tuple[float, float, float]:
    rgb = MATERIAL_COLORS.get(mat, (128, 128, 128))
    return _rgb255_to_unit(list(rgb))


def load_material_assets(
    manifest_path: Optional[Path] = None,
    *,
    load_models: bool = True,
) -> dict[VoxelMaterial, MaterialAsset]:
    """Parse manifest.yaml and optionally load glTF models."""
    path = manifest_path or _DEFAULT_MANIFEST
    base_dir = path.parent

    defaults = {"scale": 1.0, "roughness": 0.75, "metallic": 0.0}
    entries: dict = {}

    if path.is_file():
        data = yaml.safe_load(path.read_text()) or {}
        defaults.update(data.get("defaults") or {})
        entries = data.get("materials") or {}

    assets: dict[VoxelMaterial, MaterialAsset] = {}

    for mat in VoxelMaterial:
        if mat == VoxelMaterial.AIR:
            continue
        key = mat.value
        cfg = entries.get(key, {})
        color = _rgb255_to_unit(cfg["color"]) if "color" in cfg else _fallback_color(mat)
        asset = MaterialAsset(
            material=mat,
            color=color,
            roughness=float(cfg.get("roughness", defaults["roughness"])),
            metallic=float(cfg.get("metallic", defaults["metallic"])),
            scale=float(cfg.get("scale", defaults["scale"])),
        )
        if load_models and cfg.get("model"):
            model_rel = cfg["model"]
            model_path = (base_dir / model_rel).resolve()
            asset.model_path = model_path
            if model_path.is_file():
                try:
                    from .mesh import load_gltf_mesh
                    verts, indices = load_gltf_mesh(model_path)
                    asset.vertices = verts * asset.scale
                    asset.indices = indices
                except Exception:
                    pass  # fall back to instanced cube + color
        assets[mat] = asset

    return assets
