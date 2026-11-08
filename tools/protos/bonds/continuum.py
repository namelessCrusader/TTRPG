"""The four scenes in the continuum (src/mpm), built exactly as the solver
builds them today, for bonds.py to be measured against.

Nothing here changes src/: two rows are added to `MAT` in this process only
(a rope and a rigid weight, which the table does not have), and gravity is
set to zero in this process only for the free-space checks."""

import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import src.mpm.core as core                                     # noqa: E402
from common import envelope                                     # noqa: E402
from src.mpm import MPM                                         # noqa: E402
from src.mpm import scenes as ms                                # noqa: E402

TICK = 0.025

# 12 mm manila (bonds.ROPE) drawn in one 5 cm cell: the same breaking load
# (10.6 kN) and stretch (E*A 1.06e5 N) and the same 0.1 kg a metre, spread
# over 25 cm^2 instead of 1.13 cm^2. No rigidity: a rope keeps no shape.
core.MAT.setdefault("rope", (40.0, 1.06e5 / 2.5e-3, 0.40, None, 1.00,
                             1.06e4 / 2.5e-3, 0.00))
# the 5 kg weight: a 10 cm box at 5000 kg/m^3, held rigid by its cluster
core.MAT.setdefault("bob", (5000.0, 1.0e9, 0.30, None, 1.00, None, 1.00))


def tick(w, dt=TICK, after=None):
    """One tick in as many substeps as the solver's own CFL asks for, with
    `after(h)` called after every substep (pins and loads write there,
    which `MPM.step(each=)` does not allow)."""
    want = max(1, int(np.ceil(dt / w.cfl_dt())))
    h = dt / want
    for _ in range(want):
        w.substep(h)
        if after is not None:
            after(h)
    return want


class Pin:
    """Particles held where they are told to be: x0 + amp*sin(2 pi f t),
    the shaking eased in and out over [on, off] as bonds.py eases it."""

    def __init__(self, w, mask, amp=(0, 0, 0), freq=0.0, on=-1e30, off=1e30,
                 heavy=1e6):
        self.w, self.idx = w, np.flatnonzero(mask)
        self.x0 = w.x[self.idx].copy()
        # A PIN MUST HOLD THE GRID, NOT ONLY ITS OWN PARTICLES. Setting the
        # pinned particles' velocity each substep moves only them: the nodes
        # they share with the rest get their pull diluted by mass, and a rope
        # of light fibre slid out from under a pin of its own particles under
        # a 5 kg weight. Made a million times heavier, the pinned particles
        # set their nodes' velocity, which is what a clamp is.
        w.m[self.idx] *= heavy
        self.amp, self.f = np.asarray(amp, float), float(freq)
        self.on, self.off = on, off

    def __call__(self, h):
        w2 = 2 * np.pi * self.f
        t = self.w.t
        e, de = envelope(t, self.on, self.off)
        sn, cs = np.sin(w2 * t), np.cos(w2 * t)
        self.w.x[self.idx] = self.x0 + self.amp * sn * e
        self.w.v[self.idx] = self.amp * (w2 * cs * e + sn * de)
        self.w.C[self.idx] = 0.0


def load(w, idx, force):
    """A force (N, a 3-vector) shared over particles `idx`, as an impulse
    after each substep."""
    def push(h):
        w.v[idx] += np.asarray(force(), float) / w.m[idx].sum() * h
    return push


# ── 1. the tree ──────────────────────────────────────────────────────────────

def tree_world(nx=40, ny=40, nz=70, cx=20, cy=20):
    """`tree_fell`'s world and tree (voxel World, 5 cm voxels)."""
    from src.voxel.sim import STONE, World
    from src.voxel.scenes import _grow_tree
    w = World(nx, ny, nz, voxel_cm=5)
    w.fill(0, nx, 0, ny, 0, 1, STONE)
    _grow_tree(w, cx, cy)
    return w


def tree_promoted():
    """The whole tree handed to the particle solver in ONE promote, as one
    bound cluster: the continuum's best case for a standing, shaking tree
    (on the lattice it cannot move at all). Returns (world, mpm)."""
    from src.voxel.sim import AIR, LEAF, WOOD
    w = tree_world()
    cells = np.argwhere((w.mat == WOOD) | (w.mat == LEAF))
    w.matter.promote(cells)
    m = w.matter.mpm
    m.wall[:] = w.mat != AIR
    return w, m


# ── 2. the rope ──────────────────────────────────────────────────────────────

def rope(stiff=1e-4, angle=0.0):
    """1 m of rope from a pinned top at (0.6, 0.3, 2.0), a 5 kg weight below,
    the whole thing turned `angle` about the top."""
    w = MPM((24, 12, 46), dx=0.05, stiff=stiff)
    w.add_wall((0.0, 0.0, 0.0), (1.2, 0.6, 0.1))        # a floor, far below
    n0 = w.add_box((0.575, 0.275, 1.0), (0.625, 0.325, 2.0), "rope")
    w.add_box((0.55, 0.25, 0.9), (0.65, 0.35, 1.0), "bob")
    rope_ = np.arange(n0)
    bob = np.arange(n0, len(w.x))
    w.bind(n0)
    top = np.array((0.6, 0.3, 2.0))
    c, s = np.cos(angle), np.sin(angle)
    r = w.x - top
    w.x = top + np.stack([c * r[:, 0] - s * r[:, 2], r[:, 1], s * r[:, 0] + c * r[:, 2]], 1)
    # the top layer of rope particles is tied to the beam (`Pin` them)
    tied = np.isin(np.arange(len(w.x)), rope_) & (r[:, 2] > -0.026)
    return w, dict(rope=rope_, bob=bob, tied=tied, top=top)


# ── 3. the glass pane ────────────────────────────────────────────────────────

def glass(bound=False):
    """src/mpm scenes.glass_edge, started 5 cm above the table at its 6.3 m/s
    arrival speed (tests/test_mpm.py `_glass`); `bound` makes it one cluster,
    as `promote` would hand it over (`_bound_glass`)."""
    w = ms.glass_edge()
    w.x[:, 2] -= 3.2 - 1.25
    w.v[:, 2] = -6.3
    if bound:
        w.bind(0)
    return w


# ── 4. the plank ─────────────────────────────────────────────────────────────

def plank_tip(bound=False):
    w = ms.plank()
    if bound:
        w.bind(0)
    return w


def plank_bend(stiff=1e-4, bound=False):
    w = MPM((30, 6, 14), dx=0.1, stiff=stiff)
    w.add_wall((0.0, 0.0, 0.0), (3.0, 0.6, 0.1))        # floor
    w.add_wall((0.5, 0.0, 0.0), (0.7, 0.6, 1.0))        # supports
    w.add_wall((2.3, 0.0, 0.0), (2.5, 0.6, 1.0))
    w.add_box((0.5, 0.2, 1.0), (2.5, 0.4, 1.1), "wood")
    if bound:
        w.bind(0)
    mid = (np.abs(w.x[:, 0] - 1.5) < 0.05) & (w.x[:, 2] > 1.05)
    return w, dict(load=np.flatnonzero(mid))
