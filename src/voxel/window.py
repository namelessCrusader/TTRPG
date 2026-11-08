"""Work only where the matter is (wave 12, lane SCALE).

A yard with one person in it is 1.6 million voxels, and about thirty-three
thousand of them are not air. The torque law, the support law and the sight
rays used to write their answers into whole-lattice arrays, and the time they
took grew with the lattice (77 ms a tick at 1.6 M voxels, 250-330 at 5.1 M)
while the work they had to do grew with the matter. Here the matter is listed
once (`np.flatnonzero`, in the flat order a boolean mask would give) and the
laws work on that list.

Every function here gives the SAME ANSWER as the whole-lattice code it stands
in for, to the bit; none of it is an approximation. The whole-lattice code
stays where it was as the plain path (`w.windows = False`), and
tests/test_scale.py runs the same worlds both ways and asks for equal bits.
"""
import numpy as np

from .kernels import njit
from .label import _root


@njit(cache=True)
def _compact_labels(idx, key, ny, nz):
    """The pieces of a list of cells: `idx` holds the flat indices of the
    cells (ascending), `key[i]` the join key of cell i. Two cells join when
    they share a face and have equal keys. Each piece is numbered by the rank
    of its largest flat index among all the pieces', exactly as
    `np.unique(label(mask, key, "max")[mask], return_inverse=True)` numbers
    them. Also, for each cell, the number of the piece of the cell directly
    below it (-1 when that is not in the list or there is no cell below).

    The neighbour of cell i one step back along y or x is found by walking a
    second index forward through the same sorted list (the cell we want is
    always `f - nz` or `f - ny * nz`, and both grow with f), so nothing here
    is the size of the lattice."""
    n = idx.shape[0]
    par = np.arange(n)
    sx = ny * nz
    jy = 0
    jx = 0
    for i in range(n):
        f = idx[i]
        z = f % nz
        y = (f // nz) % ny
        x = f // sx
        for d in range(3):
            j = -1
            if d == 0:
                if z > 0 and i > 0 and idx[i - 1] == f - 1:
                    j = i - 1
            elif d == 1:
                if y > 0:
                    t = f - nz
                    while jy < i and idx[jy] < t:
                        jy += 1
                    if jy < i and idx[jy] == t:
                        j = jy
            else:
                if x > 0:
                    t = f - sx
                    while jx < i and idx[jx] < t:
                        jx += 1
                    if jx < i and idx[jx] == t:
                        j = jx
            if j < 0 or key[j] != key[i]:
                continue
            a = _root(par, i)
            b = _root(par, j)
            if a != b:
                # the root is the largest member: the canonical cell
                if a > b:
                    par[b] = a
                else:
                    par[a] = b
    comp = np.empty(n, np.int64)
    rank = np.empty(n, np.int64)
    m = 0
    for i in range(n):
        if par[i] == i:                  # roots come up in order of their labels
            rank[i] = m
            m += 1
    for i in range(n):
        comp[i] = rank[_root(par, i)]
    below = np.full(n, -1, np.int64)
    for i in range(1, n):
        if idx[i] % nz > 0 and idx[i - 1] == idx[i] - 1:
            below[i] = comp[i - 1]
    return comp, below, m


def compact_labels(idx, key, shape):
    """`_compact_labels` for a lattice of this `shape`: (the piece number of
    each cell in the list, the piece number of the cell below each, how many
    pieces)."""
    comp, below, m = _compact_labels(idx, np.ascontiguousarray(key, np.int64),
                                     shape[1], shape[2])
    return comp, below, int(m)


@njit(cache=True)
def _piece_bounds(comp, cx, cy, cz, below, n, nz):
    """What `_law_torque` wants to know about each piece, in one pass over
    the cells (it was nine `ufunc.at` calls, 1.2 ms for 33 thousand cells):

    * the extent in x and y of the piece at each height z (bx0, bx1, by0, by1,
      n x nz);
    * the footprint it stands on: the extent (fx0, fx1, fy0, fy1) and the
      lowest height (fz) of the cells that touch the ground or another piece;
    * the lowest and highest piece it stands on above the ground (sup_lo,
      sup_hi), and whether it touches the ground (has_ground).

    Integers all, so the order of the pass cannot change an answer."""
    big = 10 ** 6
    bx0 = np.full((n, nz), big)
    bx1 = np.full((n, nz), -big)
    by0 = np.full((n, nz), big)
    by1 = np.full((n, nz), -big)
    fx0 = np.full(n, big)
    fx1 = np.full(n, -big)
    fy0 = np.full(n, big)
    fy1 = np.full(n, -big)
    fz = np.full(n, big)
    sup_lo = np.full(n, 10 ** 9)
    sup_hi = np.full(n, -1)
    has_ground = np.zeros(n, np.bool_)
    for i in range(comp.shape[0]):
        c = comp[i]
        x = cx[i]
        y = cy[i]
        z = cz[i]
        if x < bx0[c, z]:
            bx0[c, z] = x
        if x > bx1[c, z]:
            bx1[c, z] = x
        if y < by0[c, z]:
            by0[c, z] = y
        if y > by1[c, z]:
            by1[c, z] = y
        if z == 0 or (below[i] >= 0 and below[i] != c):
            if x < fx0[c]:
                fx0[c] = x
            if x > fx1[c]:
                fx1[c] = x
            if y < fy0[c]:
                fy0[c] = y
            if y > fy1[c]:
                fy1[c] = y
            if z < fz[c]:
                fz[c] = z
            if z == 0:
                has_ground[c] = True
            else:
                if below[i] < sup_lo[c]:
                    sup_lo[c] = below[i]
                if below[i] > sup_hi[c]:
                    sup_hi[c] = below[i]
    return bx0, bx1, by0, by1, fx0, fx1, fy0, fy1, fz, sup_lo, sup_hi, has_ground


@njit(cache=True)
def _relax_list(idx, solid, span, rubble, slack):
    """`kernels._relax_nb`, starting from a list of the solid cells in place
    of a scan of the whole lattice for them: the same queue, the same rule,
    and so the same field (that kernel says why order cannot matter). The
    scan was most of its cost in a world that is mostly air."""
    nx, ny, nz = solid.shape
    n = nx * ny * nz
    cap = idx.shape[0] + 1               # a cell is in the queue once at most
    q = np.empty(cap, np.int64)
    inq = np.zeros(n, np.bool_)
    head = 0
    tail = 0
    for k in range(idx.shape[0]):
        q[tail] = idx[k]
        tail += 1
        inq[idx[k]] = True
    while head != tail:
        i = q[head]
        head += 1
        if head == cap:
            head = 0
        inq[i] = False
        z = i % nz
        y = (i // nz) % ny
        x = i // (ny * nz)
        cand = -1
        if z > 0 and slack[x, y, z - 1] >= 0:
            cand = np.int64(slack[x, y, z - 1])
        if rubble[x, y, z]:
            reach = cand
            capv = 0
        else:
            nb = -1
            if x > 0 and slack[x - 1, y, z] - 1 > nb:
                nb = slack[x - 1, y, z] - 1
            if x < nx - 1 and slack[x + 1, y, z] - 1 > nb:
                nb = slack[x + 1, y, z] - 1
            if y > 0 and slack[x, y - 1, z] - 1 > nb:
                nb = slack[x, y - 1, z] - 1
            if y < ny - 1 and slack[x, y + 1, z] - 1 > nb:
                nb = slack[x, y + 1, z] - 1
            if z > 0 and slack[x, y, z - 1] - 1 > nb:
                nb = slack[x, y, z - 1] - 1
            reach = cand if cand > nb else nb
            capv = np.int64(span[x, y, z])
        v = reach if reach < capv else capv
        if v > slack[x, y, z]:
            slack[x, y, z] = v
            for d in range(5):
                xx, yy, zz = x, y, z
                if d == 0:
                    xx = x + 1
                elif d == 1:
                    xx = x - 1
                elif d == 2:
                    yy = y + 1
                elif d == 3:
                    yy = y - 1
                else:
                    zz = z + 1
                if xx < 0 or yy < 0 or xx >= nx or yy >= ny or zz >= nz:
                    continue
                if not solid[xx, yy, zz]:
                    continue
                j = (xx * ny + yy) * nz + zz
                if not inq[j]:
                    q[tail] = j
                    tail += 1
                    if tail == cap:
                        tail = 0
                    inq[j] = True
    return slack


def box_volume(box):
    """The number of cells in a box of slices (as `_window` gives it)."""
    v = 1
    for s in box:
        v *= s.stop - s.start
    return v
