"""Two small kernels for the lazy answers (wave 15, S4-3). Plain numpy is the
definition; the compiled loops give the same answers (the tests compare them).

`nz_box(a)` is the box round the non-zero cells of an array, in one pass over
it: what `_window(a != 0, 0)` finds with a whole-lattice temporary and three
reductions. `all_settled(...)` is the quiet test of a cold world, with no box
to keep: is every cell within the tolerance of its own rest?
"""
import numpy as np

from .kernels import njit


@njit(cache=True)
def nz_box(a):
    """(lo, hi): per axis the first and last index holding `a != 0`, or hi = -1
    on every axis when the array is all zero (NaN counts as non-zero, as in
    numpy). A row with nothing in it is read once and costs one OR a cell."""
    nx, ny, nz = a.shape
    lo = np.array([nx, ny, nz])
    hi = np.array([-1, -1, -1])
    for i in range(nx):
        for j in range(ny):
            hit = False
            for k in range(nz):
                hit |= a[i, j, k] != 0
            if hit:
                if i < lo[0]:
                    lo[0] = i
                if i > hi[0]:
                    hi[0] = i
                if j < lo[1]:
                    lo[1] = j
                if j > hi[1]:
                    hi[1] = j
                k0 = 0
                while not (a[i, j, k0] != 0):
                    k0 += 1
                k1 = nz - 1
                while not (a[i, j, k1] != 0):
                    k1 -= 1
                if k0 < lo[2]:
                    lo[2] = k0
                if k1 > hi[2]:
                    hi[2] = k1
    return lo, hi


@njit(cache=True)
def all_settled(E, C, cls, rest, eout, tol):
    """True when every cell has |E - target * C| <= tol * C, the target being
    `eout` on outdoor cells (bit 0 of `cls`) and `rest` elsewhere. The same
    arithmetic as `outside._box_nb`, in the same types (the scalars are
    float32), so the same cells count as off rest; it stops at the first x-slab
    that holds one, because the caller then asks the full kernel for the box."""
    nx, ny, nz = E.shape
    for i in range(nx):
        bad = 0
        for j in range(ny):
            for k in range(nz):
                c = C[i, j, k]
                tgt = eout if (cls[i, j, k] & 1) else rest
                d = E[i, j, k] - tgt * c
                bad += 0 if (abs(d) <= tol * c) else 1
        if bad:
            return False
    return True


@njit(cache=True)
def same4(a0, b0, a1, b1, a2, b2, a3, b3):
    """True when a0 == b0, a1 == b1, a2 == b2 and a3 == b3 on every cell (flat
    arrays of one length each pair): `np.array_equal` four times, in ONE pass
    with no boolean temporaries, stopping at the first block that differs. A NaN
    is not equal to itself here, as in numpy."""
    n = a0.shape[0]
    for s in range(0, n, 8192):
        e = min(s + 8192, n)
        d = False
        for i in range(s, e):
            d |= (a0[i] != b0[i]) | (a1[i] != b1[i]) | (a2[i] != b2[i]) | (a3[i] != b3[i])
        if d:
            return False
    return True


@njit(cache=True)
def any_nz(a):
    """`a.any()` for a flat array: True at the first non-zero (a NaN is one),
    reading blocks of 4096 so a warm cell near the start ends the pass at once."""
    n = a.shape[0]
    for s in range(0, n, 4096):
        e = min(s + 4096, n)
        d = False
        for i in range(s, e):
            d |= a[i] != 0
        if d:
            return True
    return False
