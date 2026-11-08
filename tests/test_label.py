"""ONE LABELLER (docs/sim-mind-design.md, 3.3 and step E1).

The torque law and the reach both asked "which cells are one piece?" with
their own numpy flood, and the hand asked "which cells are joined to this
one?" with a Python one. `src/voxel/label.py` answers all three, compiled,
and it is only allowed to if it gives THE SAME ANSWER: not the same pieces
under other names, but the same label in every cell, because the callers
read the label values (the torque law ranks them, the reach compares them),
and the same cells in the same ORDER from a flood, because the first cell is
the one a take row names.

So the floods are copied here as they were in the laws before E1, word for
word, and frozen: they are the definition, and each test holds the compiled
path and the plain one to them on random lattices. The lattices have slabs
on the faces of the box on purpose. The reach box wraps round its faces (it
was written with `np.roll`), and a labeller that got every interior right
would still differ there.
"""
import numpy as np
import pytest

from src.voxel.label import flood, label
from src.voxel.sim import AIR, FLESH, HAVE_NUMBA

if not HAVE_NUMBA:
    pytest.skip("numba is optional; without it there is only the plain path",
                allow_module_level=True)


# ── the floods as the laws had them (base of wave 3), frozen ─────────────────
def torque_flood(mat):
    """`_law_torque`, support.py: same material, largest index, no wrap."""
    solid = mat != AIR
    lab = np.where(solid, np.arange(solid.size, dtype=np.int32).reshape(mat.shape), -1)
    while True:                                      # same-material flood labeling
        new = lab.copy()
        for axis in range(3):
            a = [slice(None)] * 3; b = [slice(None)] * 3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            same = solid[a] & solid[b] & (mat[a] == mat[b])
            new[a] = np.where(same, np.maximum(new[a], lab[b]), new[a])
            new[b] = np.where(same, np.maximum(new[b], lab[a]), new[b])
        if (new == lab).all():
            break
        lab = new
    return lab


def reach_flood(box, still):
    """`_objects_within_reach`, hands.py: same material, not flesh, not drawn
    particle matter, smallest index, WRAPPING round the box (`np.roll`)."""
    solid = (box != AIR) & (box != FLESH) & still
    lab = np.where(solid, np.arange(solid.size).reshape(solid.shape), -1)
    while True:
        prev = lab
        m = lab.copy()
        for ax in (0, 1, 2):
            for sh in (1, -1):
                n = np.roll(lab, sh, axis=ax)
                same = solid & (np.roll(box, sh, axis=ax) == box)
                take = same & (n >= 0) & ((m < 0) | (n < m))
                m = np.where(take, n, m)
        lab = np.where(solid, m, -1)
        f = lab.ravel()
        for _ in range(3):
            f = np.where(f >= 0, f[np.maximum(f, 0)], -1)
        lab = f.reshape(lab.shape)
        if np.array_equal(lab, prev):
            break
    return lab


def object_at(mat, moving, x, y, z, cap=4000):
    """`_object_at`, hands.py: the depth-first flood from one cell."""
    m0 = int(mat[x, y, z])
    if m0 == AIR or moving[x, y, z]:
        return np.zeros((0, 3), np.int32)
    nx, ny, nz = mat.shape
    seen = {(x, y, z)}
    stack = [(x, y, z)]
    out = []
    while stack and len(out) < cap:
        cx, cy, cz = stack.pop()
        out.append((cx, cy, cz))
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                           (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            t = (cx + dx, cy + dy, cz + dz)
            if t in seen or not (0 <= t[0] < nx and 0 <= t[1] < ny
                                 and 0 <= t[2] < nz):
                continue
            if int(mat[t]) == m0 and not moving[t]:
                seen.add(t)
                stack.append(t)
    return np.array(out, np.int32)


# ── random lattices ─────────────────────────────────────────────────────────
def lattice(rng, shape):
    """Blocks of four materials, slabs ON THE FACES of the box (so a piece can
    meet itself round the wrap), and a sprinkle of flesh."""
    mat = np.zeros(shape, np.uint8)
    for _ in range(3 * max(shape)):
        lo = rng.integers(0, np.array(shape))
        sz = rng.integers(1, 6, 3)
        sl = tuple(slice(int(l), int(min(l + s, n))) for l, s, n in zip(lo, sz, shape))
        mat[sl] = rng.integers(1, 5)
    mat[0, :, : max(shape[2] // 2, 1)] = 2
    mat[-1, :, : max(shape[2] // 2, 1)] = 2
    mat[:, 0, -2:] = 3
    mat[:, -1, -2:] = 3
    mat[:, : max(shape[1] // 3, 1), 0] = 4
    mat[:, : max(shape[1] // 3, 1), -1] = 4
    mat[rng.random(shape) < 0.03] = FLESH
    return mat


SHAPES = [(1, 1, 1), (1, 5, 7), (2, 2, 9), (3, 1, 4), (9, 2, 2)]


def shapes(n, seed):
    rng = np.random.default_rng(seed)
    return SHAPES + [tuple(int(v) for v in rng.integers(3, 19, 3)) for _ in range(n)]


@pytest.mark.parametrize("fused", [True, False])
def test_the_TORQUE_labels_are_the_floods_array_for_array(fused):
    """Same material, joined across faces, each piece labelled by its largest
    flat index, nothing joined round the edge of the world."""
    rng = np.random.default_rng(1)
    for shape in shapes(30, 2):
        mat = lattice(rng, shape)
        got = label(mat != AIR, mat, canon="max", fused=fused)
        want = torque_flood(mat)
        assert np.array_equal(got, want), f"{shape}: the labels differ"


@pytest.mark.parametrize("fused", [True, False])
def test_the_REACH_labels_are_the_floods_array_for_array_WRAP_and_all(fused):
    """Same material, not flesh, not the solver's, each piece labelled by its
    smallest flat index, and a piece on one face of the box joined to the
    same material on the opposite face. The wrap is checked to matter: on
    these lattices it changes the labels, so a labeller without it fails."""
    rng = np.random.default_rng(3)
    wrapped = 0
    for shape in shapes(30, 4):
        box = lattice(rng, shape)
        still = rng.random(shape) > 0.05           # a few cells are drawn
        solid = (box != AIR) & (box != FLESH) & still
        got = label(solid, box, canon="min", wrap=True, fused=fused)
        want = reach_flood(box, still)
        assert np.array_equal(got, want), f"{shape}: the labels differ"
        wrapped += not np.array_equal(
            want, label(solid, box, canon="min", wrap=False, fused=fused))
    assert wrapped >= 20, f"the wrap changed only {wrapped} lattices"


def test_the_TWO_PATHS_agree_on_the_ways_no_law_labels_yet():
    """The smallest label without the wrap (what the identity pass will ask,
    3.3) and the largest with it, and the labels with no key (any material,
    the way free fall joins things): no flood of the old laws answers these,
    so the compiled path and the plain one are held to each other, and both
    to the definition of a canonical label, the largest or smallest flat
    index of the cells that carry it."""
    rng = np.random.default_rng(5)
    for shape in shapes(20, 6):
        mat = lattice(rng, shape)
        mask = (mat != AIR) & (rng.random(shape) > 0.1)
        for canon in ("min", "max"):
            for wrap in (False, True):
                for key in (mat, None):
                    a = label(mask, key, canon=canon, wrap=wrap, fused=True)
                    b = label(mask, key, canon=canon, wrap=wrap, fused=False)
                    assert np.array_equal(a, b), (shape, canon, wrap, key is None)
                    assert np.array_equal(a < 0, ~mask)
                    for v in np.unique(a[mask]):
                        at = np.flatnonzero(a.ravel() == v)
                        assert v == (at.max() if canon == "max" else at.min())


@pytest.mark.parametrize("fused", [True, False])
def test_the_FLOOD_returns_the_same_cells_in_the_same_ORDER(fused):
    """From random cells of random lattices, with drawn cells in the way, and
    with a small cap as well as the hand's 4000, so a flood cut short is cut
    at the same cell."""
    rng = np.random.default_rng(7)
    n = 0
    for shape in shapes(25, 8):
        mat = lattice(rng, shape)
        moving = rng.random(shape) < 0.05
        seeds = np.argwhere((mat != AIR) & ~moving)
        for s in seeds[rng.integers(0, len(seeds), 6)]:
            for cap in (4000, 7):
                want = object_at(mat, moving, *map(int, s), cap=cap)
                got = flood(mat, *map(int, s), cap=cap, block=moving, fused=fused)
                assert got.dtype == np.int32 and got.shape == want.shape
                assert np.array_equal(got, want), f"{shape} from {tuple(s)}, cap {cap}"
                n += len(want)
    assert n > 1000, "the floods went somewhere"
