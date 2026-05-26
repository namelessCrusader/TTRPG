"""Orbit camera for voxel GL view."""

from __future__ import annotations

import math

import numpy as np


def perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Right-handed view matrix (column vectors)."""
    forward = eye - target
    forward = forward / np.linalg.norm(forward)
    right = np.cross(up, forward)
    n = np.linalg.norm(right)
    if n < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        right = right / n
    up_v = np.cross(forward, right)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = right
    m[1, :3] = up_v
    m[2, :3] = forward
    m[:3, 3] = -m[:3, :3] @ eye
    return m


class OrbitCamera:
    def __init__(self) -> None:
        self.yaw = 0.8
        self.pitch = 0.55
        self.distance = 22.0
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def focus_volume(self, width: int, height: int, depth: int) -> None:
        self.target = np.array(
            [width / 2.0, depth / 2.0, height / 2.0], dtype=np.float32
        )
        self.distance = max(width, height, depth) * 1.55

    def eye(self) -> np.ndarray:
        cp = math.cos(self.pitch)
        x = self.distance * cp * math.cos(self.yaw)
        y = self.distance * math.sin(self.pitch)
        z = self.distance * cp * math.sin(self.yaw)
        return self.target + np.array([x, y, z], dtype=np.float32)

    def view_matrix(self) -> np.ndarray:
        return look_at(self.eye(), self.target, np.array([0.0, 1.0, 0.0]))

    def mvp(self, width: int, height: int) -> np.ndarray:
        aspect = max(1, width) / max(1, height)
        p = perspective(50.0, aspect, 0.1, 500.0)
        v = self.view_matrix()
        return (p @ v).astype(np.float32)
