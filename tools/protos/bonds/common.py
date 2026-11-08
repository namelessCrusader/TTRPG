"""What both models are measured with: one drawing, one angle, one clock."""

import time

import numpy as np


def ascii_side(layers, walls, xlim, zlim, cols=72, rows=24, y=None):
    """A side view (x across, z up) as text. `layers` is a list of
    (points (n, 3), char); later layers draw over earlier ones. `walls` is a
    list of (lo, hi) boxes, drawn as '='. `y`, when given as (lo, hi), keeps
    only points in that slab."""
    g = [[" "] * cols for _ in range(rows)]
    sx = cols / (xlim[1] - xlim[0])
    sz = rows / (zlim[1] - zlim[0])

    def put(x, z, ch):
        c = int((x - xlim[0]) * sx)
        r = rows - 1 - int((z - zlim[0]) * sz)
        if 0 <= c < cols and 0 <= r < rows:
            g[r][c] = ch
    for lo, hi in walls:
        for c in range(cols):
            x = xlim[0] + (c + 0.5) / sx
            if not lo[0] <= x <= hi[0]:
                continue
            for r in range(rows):
                z = zlim[0] + (rows - r - 0.5) / sz
                if lo[2] <= z <= hi[2]:
                    g[r][c] = "="
    for pts, ch in layers:
        pts = np.asarray(pts).reshape(-1, 3)
        if y is not None:
            pts = pts[(pts[:, 1] >= y[0]) & (pts[:, 1] <= y[1])]
        for p in pts:
            put(p[0], p[2], ch)
    return "\n".join("|" + "".join(r) + "|" for r in g)


def long_axis(x, m=None):
    """Angle of a body's long axis above horizontal, in degrees (src/mpm
    scenes.long_axis, mass-weighted when `m` is given)."""
    m = np.ones(len(x)) if m is None else m
    c = x - (m[:, None] * x).sum(0) / m.sum()
    vec = np.linalg.eigh((m[:, None] * c).T @ c)[1][:, -1]
    return float(np.degrees(np.arcsin(min(abs(vec[2]), 1.0))))


RAMP = 0.25


def envelope(t, on, off):
    """bonds._envelope, for the continuum's pins: (on-ness, its rate)."""
    if t <= on or t >= off:
        return 0.0, 0.0
    if t < on + RAMP:
        u = (t - on) / RAMP
        return u * u * (3 - 2 * u), 6 * u * (1 - u) / RAMP
    if t > off - RAMP:
        u = (off - t) / RAMP
        return u * u * (3 - 2 * u), -6 * u * (1 - u) / RAMP
    return 1.0, 0.0


class Clock:
    """Wall-clock seconds spent inside `with clock:` blocks."""

    def __init__(self):
        self.s = 0.0

    def __enter__(self):
        self._t = time.perf_counter()

    def __exit__(self, *a):
        self.s += time.perf_counter() - self._t


def mpm_momentum(w):
    """(P, L about the origin) of an MPM world's particles. L counts each
    particle's own spin, which the affine velocity C carries: for the
    quadratic B-spline a particle is a blob with second moment dx^2/4 per
    axis, so it spins with inertia m*dx^2/2 at the rate axial(C)."""
    P = (w.m[:, None] * w.v).sum(0)
    L = np.cross(w.x, w.m[:, None] * w.v).sum(0)
    C = w.C
    ax = 0.5 * np.stack([C[:, 2, 1] - C[:, 1, 2], C[:, 0, 2] - C[:, 2, 0],
                         C[:, 1, 0] - C[:, 0, 1]], 1)
    spin = (w.m[:, None] * ax).sum(0) * w.dx ** 2 / 2.0
    return P, L, L + spin
