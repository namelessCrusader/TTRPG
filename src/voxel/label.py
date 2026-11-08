"""One labeller: which cells are joined into one piece.

Three laws ask that question, and each answered it with its own numpy flood:
torque (same material, the whole world), reach (same material, in a box round
a hand, wrapping round the box's faces) and free fall (any material). A flood
moves a label ONE VOXEL A ROUND until nothing changes, which is fine for a
crate and ruinous for a house: the torque law labelled the whole lattice in
1,895 ms, and it did that every tick that any flesh moved.

A union-find answers in one pass (2.1 ms for the same house), and the answer
is the same ARRAY, not merely the same pieces. A flood converges on a
canonical label for each piece, the largest flat index in it (torque) or the
smallest (reach), and the union-find always makes that cell the root. So no
caller has to change what it does with the labels (docs/sim-mind-design.md,
3.3).

The numpy floods stay, word for word, as the plain path (`fused=False`).
Like the kernels in kernels.py, the compiled path must agree with them cell
for cell, and tests/test_label.py says so.

Two cells are joined when both are in `mask`, they share a face, and, when a
`key` is given, their keys are equal. The key is `mat` today: a table with
iron legs is two things. The entity steps make it the key of 3.3, under
which a named axe is one thing whatever it is made of.
"""
import numpy as np

from .kernels import HAVE_NUMBA, njit


# ── the compiled path ───────────────────────────────────────────────────────
@njit(cache=True)
def _root(par, a):
    while par[a] != a:
        par[a] = par[par[a]]          # halve the path on the way up
        a = par[a]
    return a


@njit(cache=True)
def _union_find(mask, key, use_key, canon_max, wrap):
    """The labels, with each piece rooted at its largest (`canon_max`) or
    smallest flat index: -1 outside `mask`. Each cell joins the neighbour
    BEFORE it on each axis, so every shared face is looked at once; `wrap`
    also joins the first slab of an axis to the last, as `np.roll` does."""
    nx, ny, nz = mask.shape
    par = np.arange(nx * ny * nz)
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if not mask[x, y, z]:
                    continue
                i = (x * ny + y) * nz + z
                for d in range(3):
                    xx, yy, zz = x, y, z
                    if d == 0:
                        if x > 0:
                            xx = x - 1
                        elif wrap and nx > 1:
                            xx = nx - 1
                        else:
                            continue
                    elif d == 1:
                        if y > 0:
                            yy = y - 1
                        elif wrap and ny > 1:
                            yy = ny - 1
                        else:
                            continue
                    else:
                        if z > 0:
                            zz = z - 1
                        elif wrap and nz > 1:
                            zz = nz - 1
                        else:
                            continue
                    if not mask[xx, yy, zz]:
                        continue
                    if use_key and key[xx, yy, zz] != key[x, y, z]:
                        continue
                    a = _root(par, i)
                    b = _root(par, (xx * ny + yy) * nz + zz)
                    if a != b:
                        # THE ROOT IS THE CANONICAL MEMBER, so there is no
                        # second pass to find each piece's max or min
                        if (a > b) == canon_max:
                            par[b] = a
                        else:
                            par[a] = b
    out = np.full(mask.shape, -1, np.int64)
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if mask[x, y, z]:
                    out[x, y, z] = _root(par, (x * ny + y) * nz + z)
    return out


# THE SAME LABELS, A RUN AT A TIME. Most of a lattice is long runs of one
# thing up a column: the air of a room, a wall, a floor. `_union_find` makes
# every cell a node; this makes every RUN one (cells in mask, side by side up
# z, with one key), and joins two runs of neighbouring columns when they
# overlap in z (and, with a key, share it). A house's air labels in about
# 1 ms here against 16 ms a cell at a time.
#
# The array is the same, not merely the pieces. A run's own canonical cell is
# its top (`canon_max`: the largest flat index in it) or its bottom, so the
# piece's largest or smallest cell is the canonical cell of one of its runs,
# and the union keeps that run as the root, as `_union_find` keeps the cell.
@njit(cache=True)
def _run_forest(mask, key, use_key, canon_max):
    """The runs (per column `cstart`, from `rs` to `re` up z), joined: `par`
    is the union-find over runs, and `rep[r]` is run r's canonical flat index
    (for a root, its piece's). Runs are numbered in flat order."""
    nx, ny, nz = mask.shape
    ncol = nx * ny
    cstart = np.zeros(ncol + 1, np.int64)
    rs = np.empty(ncol * nz + 1, np.int64)
    re = np.empty(ncol * nz + 1, np.int64)
    nr = 0
    for x in range(nx):
        for y in range(ny):
            c = x * ny + y
            cstart[c] = nr
            z = 0
            while z < nz:
                if mask[x, y, z]:
                    z0 = z
                    z += 1
                    while z < nz and mask[x, y, z] and \
                            (not use_key or key[x, y, z] == key[x, y, z0]):
                        z += 1
                    rs[nr] = z0
                    re[nr] = z
                    nr += 1
                else:
                    z += 1
    cstart[ncol] = nr
    par = np.arange(nr)
    rep = np.empty(nr, np.int64)
    for c in range(ncol):
        for r in range(cstart[c], cstart[c + 1]):
            rep[r] = c * nz + (re[r] - 1 if canon_max else rs[r])
    for x in range(nx):
        for y in range(ny):
            c = x * ny + y
            for d in range(2):
                if d == 0:
                    if x == 0:
                        continue
                    c2 = c - ny
                    x2, y2 = x - 1, y
                else:
                    if y == 0:
                        continue
                    c2 = c - 1
                    x2, y2 = x, y - 1
                i = cstart[c]
                j = cstart[c2]
                ie = cstart[c + 1]
                je = cstart[c2 + 1]
                while i < ie and j < je:
                    if rs[i] < re[j] and rs[j] < re[i] and \
                            (not use_key or key[x, y, rs[i]] == key[x2, y2, rs[j]]):
                        a = _root(par, i)
                        b = _root(par, j)
                        if a != b:
                            if (rep[a] > rep[b]) == canon_max:
                                par[b] = a
                            else:
                                par[a] = b
                    if re[i] < re[j]:
                        i += 1
                    else:
                        j += 1
    return cstart, rs[:nr], re[:nr], par, rep


@njit(cache=True)
def _label_runs(mask, key, use_key, canon_max):
    """`_union_find` without `wrap`, a run at a time: the same array."""
    nx, ny, nz = mask.shape
    cstart, rs, re, par, rep = _run_forest(mask, key, use_key, canon_max)
    out = np.full(mask.shape, -1, np.int64)
    for x in range(nx):
        for y in range(ny):
            c = x * ny + y
            for r in range(cstart[c], cstart[c + 1]):
                v = rep[_root(par, r)]
                for z in range(rs[r], re[r]):
                    out[x, y, z] = v
    return out


@njit(cache=True)
def _regions_runs(mask):
    """`regions`, compiled: each piece numbered by the rank of its largest
    flat index among all the pieces', as np.unique's inverse numbers them.
    Runs are in flat order and a root run holds its piece's largest cell, so
    the roots come up in the order of their labels and are counted as met."""
    nx, ny, nz = mask.shape
    cstart, rs, re, par, rep = _run_forest(mask, mask, False, True)
    nr = rs.shape[0]
    rank = np.empty(nr, np.int32)
    n = 0
    for r in range(nr):
        if _root(par, r) == r:
            rank[r] = n
            n += 1
    top = np.empty(n, np.int64)          # each piece's largest flat index
    for r in range(nr):
        if _root(par, r) == r:
            top[rank[r]] = rep[r]
    out = np.full(mask.shape, -1, np.int32)
    for x in range(nx):
        for y in range(ny):
            c = x * ny + y
            for r in range(cstart[c], cstart[c + 1]):
                v = rank[_root(par, r)]
                for z in range(rs[r], re[r]):
                    out[x, y, z] = v
    return out, n, top


@njit(cache=True)
def _patch_regions(new, kept, out, top, seen, mark, queue):
    """Bring `kept` (the mask the labels `out` were made for) up to `new`,
    patching `out` as it goes, when that can be done without any piece
    appearing, vanishing, joining another or changing its largest cell: 0 when
    it was done, 1 when it cannot be (the caller then labels everything).

    Cells change one at a time, each against the mask as it stands, so each
    step is a step between two masks whose labels we hold:
      * a cell that becomes air takes the label of the air beside it, which
        must be one piece's (none: a new piece; two: a join) and must not be
        larger than that piece's largest cell;
      * a cell that stops being air must not be its piece's largest cell, and
        the air beside it must still be one piece without it: the search goes
        out from one neighbour through the air until it meets the others, and
        gives up at the length of `queue`.
    The pieces are numbered by their largest cell, so while none of that
    changes, the numbers are exactly those of a labelling from scratch."""
    nx, ny, nz = new.shape
    n = nx * ny * nz
    nb = np.empty(6, np.int64)
    # which cells differ, in flat order
    ch = np.empty(4096, np.int64)
    nch = 0
    fn = new.reshape(n)
    fk = kept.reshape(n)
    B = 4096
    for s in range(0, n, B):                     # blocks that are the same are
        e = min(s + B, n)                        # passed over without a branch
        acc = False
        for i in range(s, e):
            acc |= fn[i] != fk[i]
        if acc:
            for i in range(s, e):
                if fn[i] != fk[i]:
                    if nch == ch.shape[0]:
                        return 1
                    ch[nch] = i
                    nch += 1
    fo = out.reshape(n)
    for k in range(nch):
        c = ch[k]
        z = c % nz
        y = (c // nz) % ny
        x = c // (ny * nz)
        m = 0                                    # the neighbours that are air now
        for d in range(6):
            if d == 0:
                if x == 0:
                    continue
                t = c - ny * nz
            elif d == 1:
                if x == nx - 1:
                    continue
                t = c + ny * nz
            elif d == 2:
                if y == 0:
                    continue
                t = c - nz
            elif d == 3:
                if y == ny - 1:
                    continue
                t = c + nz
            elif d == 4:
                if z == 0:
                    continue
                t = c - 1
            else:
                if z == nz - 1:
                    continue
                t = c + 1
            if fk[t]:
                nb[m] = t
                m += 1
        if fn[c]:                                # becomes air
            if m == 0:
                return 1
            lab = fo[nb[0]]
            for j in range(1, m):
                if fo[nb[j]] != lab:
                    return 1
            if c > top[lab]:
                return 1
            fk[c] = True
            fo[c] = lab
        else:                                    # stops being air
            lab = fo[c]
            if top[lab] == c:
                return 1
            fk[c] = False
            fo[c] = -1
            if m == 0:
                return 1                         # it was a piece by itself
            if m == 1:
                continue                         # it hung on by one face
            mark[0] += 1
            stamp = mark[0]
            seen[nb[0]] = stamp
            queue[0] = nb[0]
            head = 0
            tail = 1
            found = 1
            while head < tail and found < m:
                f = queue[head]
                head += 1
                fz = f % nz
                fy = (f // nz) % ny
                fx = f // (ny * nz)
                for d in range(6):
                    if d == 0:
                        if fx == 0:
                            continue
                        t = f - ny * nz
                    elif d == 1:
                        if fx == nx - 1:
                            continue
                        t = f + ny * nz
                    elif d == 2:
                        if fy == 0:
                            continue
                        t = f - nz
                    elif d == 3:
                        if fy == ny - 1:
                            continue
                        t = f + nz
                    elif d == 4:
                        if fz == 0:
                            continue
                        t = f - 1
                    else:
                        if fz == nz - 1:
                            continue
                        t = f + 1
                    if not fk[t] or seen[t] == stamp:
                        continue
                    seen[t] = stamp
                    for j in range(1, m):
                        if nb[j] == t:
                            found += 1
                    if tail == queue.shape[0]:
                        return 1
                    queue[tail] = t
                    tail += 1
            if found < m:
                return 1
    return 0


@njit(cache=True)
def _flood_nb(key, block, x, y, z, cap):
    """`_flood_py`, compiled: the same stack, the same order of neighbours,
    a cell marked seen when it is pushed and written out when it is popped,
    so the cells come back in the same ORDER."""
    nx, ny, nz = key.shape
    k0 = key[x, y, z]
    seen = np.zeros(key.shape, np.bool_)
    stack = np.empty((nx * ny * nz, 3), np.int64)
    out = np.empty((max(min(cap, nx * ny * nz), 0), 3), np.int32)
    stack[0, 0], stack[0, 1], stack[0, 2] = x, y, z
    seen[x, y, z] = True
    sp, n = 1, 0
    while sp > 0 and n < cap:
        sp -= 1
        cx, cy, cz = stack[sp, 0], stack[sp, 1], stack[sp, 2]
        out[n, 0], out[n, 1], out[n, 2] = cx, cy, cz
        n += 1
        for d in range(6):
            tx, ty, tz = cx, cy, cz
            if d == 0:
                tx = cx + 1
            elif d == 1:
                tx = cx - 1
            elif d == 2:
                ty = cy + 1
            elif d == 3:
                ty = cy - 1
            elif d == 4:
                tz = cz + 1
            else:
                tz = cz - 1
            if tx < 0 or ty < 0 or tz < 0 or tx >= nx or ty >= ny or tz >= nz:
                continue
            if seen[tx, ty, tz]:
                continue
            if key[tx, ty, tz] == k0 and not block[tx, ty, tz]:
                seen[tx, ty, tz] = True
                stack[sp, 0], stack[sp, 1], stack[sp, 2] = tx, ty, tz
                sp += 1
    return out[:n].copy()


# ── the plain path: the floods as the laws had them ─────────────────────────
def _spread_max(mask, key):
    """The torque law's flood (`_law_torque` before E1): every label grows to
    the largest flat index it touches, one voxel a round, and stops at the
    edges of the box."""
    lab = np.where(mask, np.arange(mask.size, dtype=np.int32).reshape(mask.shape), -1)
    while True:                                      # same-material flood labeling
        new = lab.copy()
        for axis in range(3):
            a = [slice(None)] * 3; b = [slice(None)] * 3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            same = mask[a] & mask[b]
            if key is not None:
                same &= key[a] == key[b]
            new[a] = np.where(same, np.maximum(new[a], lab[b]), new[a])
            new[b] = np.where(same, np.maximum(new[b], lab[a]), new[b])
        if (new == lab).all():
            break
        lab = new
    return lab


def _spread_jump(mask, key, canon_max, wrap):
    """The reach flood (`_objects_within_reach` before E1): the smallest label
    spreads, then every label JUMPS through the table it is part of, which
    halves the distance to the root each time. It wraps round the faces of the
    box, as `np.roll` does; `wrap=False` closes them, and `canon_max` grows
    the largest label instead (no caller had either, so both are new)."""
    lab = np.where(mask, np.arange(mask.size).reshape(mask.shape), -1)
    while True:
        prev = lab
        m = lab.copy()
        for ax in (0, 1, 2):
            for sh in (1, -1):
                n = np.roll(lab, sh, axis=ax)
                if not wrap:                        # nothing comes round
                    edge = [slice(None)] * 3
                    edge[ax] = 0 if sh == 1 else -1
                    n[tuple(edge)] = -1
                same = mask if key is None else mask & (np.roll(key, sh, axis=ax) == key)
                better = (n > m) if canon_max else ((m < 0) | (n < m))
                take = same & (n >= 0) & better
                m = np.where(take, n, m)
        lab = np.where(mask, m, -1)
        f = lab.ravel()
        for _ in range(3):
            f = np.where(f >= 0, f[np.maximum(f, 0)], -1)
        lab = f.reshape(lab.shape)
        if np.array_equal(lab, prev):
            break
    return lab


def _flood_py(key, block, x, y, z, cap):
    """`_object_at`'s depth-first flood (hands.py before E1), for any key."""
    m0 = key[x, y, z]
    nx, ny, nz = key.shape
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
            if key[t] == m0 and not block[t]:
                seen.add(t)
                stack.append(t)
    return np.array(out, np.int32).reshape(-1, 3)


# ── the two questions ───────────────────────────────────────────────────────
def label(mask, key=None, canon="max", wrap=False, fused=True):
    """Which piece each cell of `mask` is in, as the flat index of that
    piece's largest (`canon="max"`) or smallest (`"min"`) cell; -1 outside
    `mask`. Cells join across a shared face, and only where `key` is equal
    when a key is given. `wrap` also joins opposite faces of the box.

    `fused` is the world's own switch (`w.fused`): compiled when it is on and
    numba is there, the numpy flood otherwise. The two give equal arrays."""
    canon_max = canon == "max"
    if fused and HAVE_NUMBA:
        k = mask if key is None else key
        if not wrap:
            return _label_runs(mask, k, key is not None, canon_max)
        return _union_find(mask, k, key is not None, canon_max, wrap)
    if canon_max and not wrap:
        return _spread_max(mask, key)
    return _spread_jump(mask, key, canon_max, wrap)


# THE AIR LABELS ARE KEPT, because they hardly ever change. Labelling the air
# of a yard takes 3-4 ms at 1.6 M voxels, and it was done on every tick that
# anyone moved, to find out that the one big airspace was still one big
# airspace with a person-shaped hole that had moved over. So the last
# labelling of each size of lattice is kept with the mask it was made for, and
# a new mask that differs from it in a few cells is patched into it (see
# `_patch_regions`) when that provably leaves every piece alone. Anything else
# is labelled from scratch, as before, and kept in its turn. What is kept is
# only ever a labelling of the mask it sits with, so a different world of the
# same size can use it as well as the one that made it. Callers are always
# handed a copy.
_INCR_MIN = 1 << 18             # smaller lattices are labelled in no time
_INCR_QUEUE = 4096              # how far a search for a split may go
_KEPT_REGIONS = {}              # lattice shape -> [mask, labels, tops, seen, mark]


def _regions_kept(mask):
    """`regions` for a big lattice with the last labelling of its size kept."""
    mask = np.ascontiguousarray(mask)
    st = _KEPT_REGIONS.get(mask.shape)
    if st is not None:
        if _patch_regions(mask, st[0], st[1], st[2], st[3], st[4],
                          np.empty(_INCR_QUEUE, np.int64)) == 0:
            return st[1].copy(), int(st[2].shape[0])
    out, n, top = _regions_runs(mask)
    if st is None:
        if len(_KEPT_REGIONS) >= 4:                    # a few sizes at most
            _KEPT_REGIONS.pop(next(iter(_KEPT_REGIONS)))
        st = _KEPT_REGIONS[mask.shape] = [None, None, None,
                                          np.zeros(mask.size, np.int32),
                                          np.zeros(1, np.int32)]
    st[0], st[1], st[2] = mask.copy(), out.copy(), top
    return out, int(n)


def regions(mask, fused=True):
    """The pieces of `mask` (no key, no wrap), numbered 0, 1, 2... in the
    order of their largest flat index, -1 outside `mask`, as an int32 array;
    and how many there are. What the air law keeps as `air_region`."""
    if fused and HAVE_NUMBA:
        if mask.size >= _INCR_MIN and mask.dtype == np.bool_:
            return _regions_kept(mask)
        out, n, _top = _regions_runs(mask)
        return out, int(n)
    lab = label(mask, canon="max", fused=fused)
    _, compact = np.unique(lab[mask], return_inverse=True)
    out = np.full(mask.shape, -1, np.int32)
    out[mask] = compact.astype(np.int32)
    return out, int(compact.max()) + 1 if compact.size else 0


def flood(key, x, y, z, cap=4000, block=None, fused=True):
    """The cells joined to (x, y, z), depth first: same `key`, not `block`ed,
    at most `cap` of them, as an (n, 3) int32 array. The ORDER is part of the
    answer: the first cell is the one a take row names, and `_launch` keeps
    the order it is given."""
    if block is None:
        block = np.zeros(key.shape, bool)
    if fused and HAVE_NUMBA:
        return _flood_nb(key, block, int(x), int(y), int(z), int(cap))
    return _flood_py(key, block, int(x), int(y), int(z), cap)
