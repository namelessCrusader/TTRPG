"""Optional numba kernels: fused per-cell loops that must agree with the numpy laws."""
import numpy as np


# OPTIONAL ACCELERATOR. numpy cannot express the shape a cellular-automata
# engine wants — one sweep of the lattice applying many rules per cell — so it
# makes a full pass per arithmetic step instead, and we are memory-bandwidth
# bound: conduction alone builds about seventeen whole-lattice temporaries per
# axis. numba compiles one loop that does all of it per cell.
#
# It is OPTIONAL on purpose. The engine's only hard dependency is PyYAML, and
# the numpy path each caller keeps alongside these kernels stays the
# definition of what the law MEANS; the compiled one has to agree with it
# cell for cell, which is what `test_the_FAST_PATH_and_the_PLAIN_ONE_agree`
# exists to check. Without numba installed, nothing changes except the speed.
try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:                        # pragma: no cover - depends on install
    HAVE_NUMBA = False

    def njit(*args, **kwargs):           # a decorator that does nothing
        def wrap(fn):
            return fn
        return wrap if not args or not callable(args[0]) else args[0]


@njit(cache=True)
def _o2_diffuse(o2, T, por, di, dj, dl):
    """One axis of oxygen mixing, fused. Same shape as conduction and the same
    discipline: numpy's order of operations, kept step for step, and the loss
    swept over every cell before the gain is."""
    nx, ny, nz = o2.shape
    q = np.empty((nx - di, ny - dj, nz - dl), np.float32)
    base = np.float32(0.1)
    div = np.float32(1600.0)
    top = np.float32(0.45)
    for i in range(nx - di):
        for j in range(ny - dj):
            for l in range(nz - dl):
                i2 = i + di
                j2 = j + dj
                l2 = l + dl
                k = base + (T[i, j, l] + T[i2, j2, l2]) / div
                if k < base:
                    k = base
                if k > top:
                    k = top
                pa = por[i, j, l]
                pb = por[i2, j2, l2]
                k = k * (pa if pa < pb else pb)
                q[i, j, l] = k * (o2[i, j, l] - o2[i2, j2, l2])
    for i in range(nx - di):
        for j in range(ny - dj):
            for l in range(nz - dl):
                o2[i, j, l] -= q[i, j, l]
    for i in range(nx - di):
        for j in range(ny - dj):
            for l in range(nz - dl):
                o2[i + di, j + dj, l + dl] += q[i, j, l]


@njit(cache=True)
def _gas_overturn(o2, smoke, T, air):
    """Warm gas sitting UNDER cool gas trades places with it, carrying whatever
    it is made of. One parcel, so its oxygen and its soot move together and by
    exactly the same fraction — which is why both gases share one kernel and one
    buffer, and why oxygen goes first here as it does in the plain law."""
    nx, ny, nz = o2.shape
    f = np.empty((nx, ny, nz - 1), np.float32)
    rate = np.float32(0.02)
    cap = np.float32(0.45)
    zero = np.float32(0.0)
    for i in range(nx):
        for j in range(ny):
            for l in range(nz - 1):
                if air[i, j, l] and air[i, j, l + 1]:
                    v = (T[i, j, l] - T[i, j, l + 1]) * rate
                    if v < zero:
                        v = zero
                    if v > cap:
                        v = cap
                    f[i, j, l] = v
                else:
                    f[i, j, l] = zero
    for gas in (o2, smoke):
        q = np.empty((nx, ny, nz - 1), np.float32)
        for i in range(nx):
            for j in range(ny):
                for l in range(nz - 1):
                    q[i, j, l] = f[i, j, l] * (gas[i, j, l] - gas[i, j, l + 1])
        for i in range(nx):
            for j in range(ny):
                for l in range(nz - 1):
                    gas[i, j, l] -= q[i, j, l]
        for i in range(nx):
            for j in range(ny):
                for l in range(nz - 1):
                    gas[i, j, l + 1] += q[i, j, l]


@njit(cache=True)
def _look_rays(ex, ey, cs, sn, R, nlog, nlog_band, tau_max, stop, reach, lit):
    """`_look_around`'s rays, marched one by one: every cell a ray REACHED at
    eye height goes into `reach`, and every cell the light band let it see
    into `lit`.

    The optical depth of each cell, -log of what gets through it, comes in
    already worked out by numpy, per column (`nlog` at eye height, `nlog_band`
    for the band from the feet to the eyes). numpy's log and numba's need not
    agree in the last bit, so the log stays numpy's and this loop only adds
    it up. The sum is in float32 and in the same order as numpy's cumsum
    along a ray, so the depth before each cell is the same number.

    A ray stops once both depths have passed `tau_max`, as no later cell can
    be reached then. That holds only while every depth is >= 0, so `stop` is
    False when either grid has a negative one (a cell a hair over 1 clear),
    and then every ray runs its full length, as numpy's does. Leaving the
    world never stops a ray: its cells are clipped to the edge, as numpy's
    are, and the edge cell is still read and marked."""
    nx, ny = nlog.shape
    hx = np.float64(nx - 1)
    hy = np.float64(ny - 1)
    for r in range(cs.shape[0]):
        c = cs[r]
        s = sn[r]
        te = np.float32(0.0)
        tb = np.float32(0.0)
        for k in range(1, R + 1):
            if stop and te >= tau_max and tb >= tau_max:
                break
            st = np.float64(np.float32(k))
            fx = np.rint(ex + c * st)
            fy = np.rint(ey + s * st)
            if fx < 0.0:
                fx = 0.0
            if fx > hx:
                fx = hx
            if fy < 0.0:
                fy = 0.0
            if fy > hy:
                fy = hy
            xi = np.int64(fx)
            yi = np.int64(fy)
            if te < tau_max:
                reach[xi, yi] = True
            if tb < tau_max:
                lit[xi, yi] = True
            te = te + nlog[xi, yi]
            tb = tb + nlog_band[xi, yi]


# ── the coarse air grid (air.py: `_cells`, `_blur_cells`, `_upcell`) ─────────
# numpy takes a block mean as `a.reshape(X, 4, Y, 4, Z, 4).mean(axis=(1, 3,
# 5))`: for each block it starts from 0, adds each run of 4 cells up z (from
# 0 too), and adds that run's sum to the block's, the runs in x-then-y order.
# A float32 sum taken in another order is another number, so these loops take
# it in that order. A cell past the edge of the world (numpy pads with zeros)
# is skipped: adding +0 to a sum leaves it as it was. That holds while a
# block's runs are summed one at a time, which numpy does whenever a block is
# more than one run deep in z; when the world is 4 cells high or less, numpy
# sums each block's 16 or 64 cells as one run, in another order, so the
# caller keeps numpy for those (`_cells_fast`).
@njit(cache=True)
def _cells_sum(a, mask, use_mask, ones, cs, mean):
    """The block sum (or mean, `mean`) of `a` over cs^3 blocks, in numpy's
    order. With `use_mask`, a cell outside `mask` counts as 0 (np.where(mask,
    a, 0)); with `ones` too, a cell inside it counts as 1 (mask.astype), and
    `a` only gives the dtype."""
    nx, ny, nz = mask.shape if use_mask else a.shape
    X = (nx + cs - 1) // cs
    Y = (ny + cs - 1) // cs
    Z = (nz + cs - 1) // cs
    out = np.empty((X, Y, Z), a.dtype)
    zero = a.dtype.type(0)
    one = a.dtype.type(1)
    div = a.dtype.type(cs * cs * cs)
    for bx in range(X):
        for by in range(Y):
            for bz in range(Z):
                acc = zero
                for i in range(bx * cs, min(bx * cs + cs, nx)):
                    for j in range(by * cs, min(by * cs + cs, ny)):
                        r = zero
                        for k in range(bz * cs, min(bz * cs + cs, nz)):
                            if use_mask:
                                if mask[i, j, k]:
                                    r += one if ones else a[i, j, k]
                                else:
                                    r += zero
                            else:
                                r += a[i, j, k]
                        acc = acc + r
                out[bx, by, bz] = acc / div if mean else acc
    return out


@njit(cache=True)
def _blur_nb(carr, passes):
    """`_blur_cells`, one pass a sweep: each cell's own value, then its
    neighbours' added in numpy's order (x+1, x-1, y+1, y-1, z+1, z-1), over
    the count of them."""
    out = carr
    X, Y, Z = carr.shape
    one = carr.dtype.type(1)
    for _ in range(passes):
        m2 = np.empty_like(out)
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    s = out[x, y, z]
                    c = one
                    if x < X - 1:
                        s += out[x + 1, y, z]
                        c += one
                    if x > 0:
                        s += out[x - 1, y, z]
                        c += one
                    if y < Y - 1:
                        s += out[x, y + 1, z]
                        c += one
                    if y > 0:
                        s += out[x, y - 1, z]
                        c += one
                    if z < Z - 1:
                        s += out[x, y, z + 1]
                        c += one
                    if z > 0:
                        s += out[x, y, z - 1]
                        c += one
                    m2[x, y, z] = s / c
        out = m2
    return out


@njit(cache=True)
def _upcell_nb(carr, nx, ny, nz, cs):
    """`_upcell`: every voxel takes its block's value."""
    out = np.empty((nx, ny, nz), carr.dtype)
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                out[x, y, z] = carr[x // cs, y // cs, z // cs]
    return out


@njit(cache=True, error_model="numpy")
def _upcell_at(carr, mask, cs):
    """`_upcell(carr)[mask]`, without the whole lattice: the block value of
    each cell in `mask`, in the order boolean indexing gives them. (Each
    layer's block is worked out once, not once a cell.)"""
    nx, ny, nz = mask.shape
    bz = np.empty(nz, np.int64)
    for z in range(nz):
        bz[z] = z // cs
    n = 0
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                n += mask[x, y, z]
    out = np.empty(n, carr.dtype)
    n = 0
    for x in range(nx):
        bx = x // cs
        for y in range(ny):
            by = y // cs
            for z in range(nz):
                if mask[x, y, z]:
                    out[n] = carr[bx, by, bz[z]]
                    n += 1
    return out


@njit(cache=True)
def _relax_nb(solid, span, rubble, slack):
    """`_relax_slack`'s fixpoint (support.py), a cell at a time.

    The numpy loop updates every cell at once until nothing changes; this one
    keeps a queue of the cells whose inputs changed and updates them one by
    one. Both reach the same field, to the cell: an update only ever raises a
    cell's slack, and it raises it by a rule that gives more when its inputs
    are more, so whatever the order, both stop at the smallest field that the
    rule cannot raise any further and that is at least the seeds. A cell reads
    the one below it (carried) and its four sideways neighbours and the one
    below (span - 1), never the one above, so a change wakes the cells that
    read it: the four beside it and the one above."""
    nx, ny, nz = solid.shape
    n = nx * ny * nz
    cap = n + 1                          # a cell is in the queue once at most
    q = np.empty(cap, np.int64)
    inq = np.zeros(n, np.bool_)
    head = 0
    tail = 0
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if solid[x, y, z]:
                    i = (x * ny + y) * nz + z
                    q[tail] = i
                    tail += 1
                    inq[i] = True
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


@njit(cache=True, error_model="numpy")
def _diff4(a, b, c, d, ka, kb, kc, kd, out):
    """Where do four flat fields differ from their kept copies? Writes the
    flat index of each such cell into `out`, in order, and returns how many
    there are: 0 when nothing differs, -1 as soon as there are more than
    `out` can hold (then the caller rebuilds everything, as if it had not
    asked). "Differs" is numpy's `!=`, so a NaN always differs and -0.0
    equals 0.0, exactly as `np.array_equal` judges them.

    Blocks of cells are first asked whether anything in them differs, in a
    loop with no branch in it (which the compiler can run several cells at
    a time), and only a block that says yes is looked through cell by cell."""
    n = 0
    cap = out.shape[0]
    N = a.shape[0]
    B = 2048
    for s in range(0, N, B):
        e = min(s + B, N)
        acc = False
        for i in range(s, e):
            acc |= (a[i] != ka[i]) | (b[i] != kb[i]) | (c[i] != kc[i]) | (d[i] != kd[i])
        if acc:
            for i in range(s, e):
                if a[i] != ka[i] or b[i] != kb[i] or c[i] != kc[i] or d[i] != kd[i]:
                    if n == cap:
                        return -1
                    out[n] = i
                    n += 1
    return n


@njit(cache=True, error_model="numpy")
def _carry_gas(field, por, vc, di, dj, dl, cs):
    """One axis of the wind carrying a gas (`_law_air`), fused. The wind on a
    face is its lower cell's coarse velocity, times 0.25, clipped to +-0.4; a
    face passes what the smaller of its two cells' pores allows. Every face's
    two flows are worked out first from the field as it was, then the field
    takes them in numpy's order: every lower cell first, then every upper
    one, so a cell in the middle sees its own loss before its neighbour's."""
    nx, ny, nz = field.shape
    fwd = np.empty((nx - di, ny - dj, nz - dl), np.float32)
    back = np.empty((nx - di, ny - dj, nz - dl), np.float32)
    quarter = np.float32(0.25)
    lo = np.float32(-0.4)
    hi = np.float32(0.4)
    zero = np.float32(0.0)
    bz = np.empty(nz, np.int64)                  # each layer's coarse cell
    for l in range(nz):
        bz[l] = l // cs
    for i in range(nx - di):
        bi = i // cs
        for j in range(ny - dj):
            bj = j // cs
            for l in range(nz - dl):
                w = vc[bi, bj, bz[l]] * quarter      # np.clip(v * 0.25, -0.4, 0.4)
                w = lo if w < lo else w
                w = hi if w > hi else w
                pa = por[i, j, l]
                pb = por[i + di, j + dj, l + dl]
                pk = pa if pa < pb else pb
                pk = pa if pa != pa else pk          # np.minimum keeps a NaN
                fa = w * field[i, j, l]
                fb = -w * field[i + di, j + dj, l + dl]
                f = fa if w > zero else zero
                b = fb if w < zero else zero
                fwd[i, j, l] = f * pk
                back[i, j, l] = b * pk
    for i in range(nx - di):
        for j in range(ny - dj):
            for l in range(nz - dl):
                field[i, j, l] += back[i, j, l] - fwd[i, j, l]
    for i in range(nx - di):
        for j in range(ny - dj):
            for l in range(nz - dl):
                field[i + di, j + dj, l + dl] += fwd[i, j, l] - back[i, j, l]


@njit(cache=True)
def _any_in(a, table):
    """`table[a].any()` for a flat array of small codes, stopping at the
    first hit and making no array of its own. A code past the end of the
    table (which numpy would refuse) answers yes, so the law runs and its
    own reads meet it."""
    n = table.shape[0]
    for i in range(a.shape[0]):
        if a[i] >= n or table[a[i]]:
            return True
    return False


# ── hearing, a batch at a time (senses.py: `_level`, `_levels`) ──────────────
# `_level` is the definition: one sound, one ear, in numpy. A house of people
# asks it for every sound at every ear, every tick, and most of what it costs is
# Python and numpy overhead on a ray a hundred cells long. This does the same
# arithmetic for a whole list of (sound, ear) pairs in one compiled call.
#
# It must give the SAME NUMBER, to the last bit, so it is written step for step
# after the numpy: float32 where numpy has float32, a float32 sum taken in
# numpy's pairwise order (`_pw32`), and the same order of the same float64
# operations after that. The logs and the power are the one thing a different
# library could round differently, so `exact_math()` asks the compiled ones, at
# the first use, whether they agree with numpy's over the values the law uses,
# and the caller keeps the numpy path when they do not.
@njit(cache=True)
def _pw_block(a, lo, n):
    """The sum of a block of at most 128 float32s, in numpy's order: runs
    under 8 added one by one; else eight running sums over strides of eight,
    joined as a tree, and the odd few at the end added last."""
    if n < 8:
        res = np.float32(0.0)
        for i in range(n):
            res += a[lo + i]
        return res
    r0 = a[lo]
    r1 = a[lo + 1]
    r2 = a[lo + 2]
    r3 = a[lo + 3]
    r4 = a[lo + 4]
    r5 = a[lo + 5]
    r6 = a[lo + 6]
    r7 = a[lo + 7]
    i = 8
    lim = n - (n % 8)
    while i < lim:
        r0 += a[lo + i]
        r1 += a[lo + i + 1]
        r2 += a[lo + i + 2]
        r3 += a[lo + i + 3]
        r4 += a[lo + i + 4]
        r5 += a[lo + i + 5]
        r6 += a[lo + i + 6]
        r7 += a[lo + i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[lo + i]
        i += 1
    return res


@njit(cache=True)
def _pw32(a, lo, n):
    """`a[lo:lo+n].sum()` for a float32 array, in numpy's order. Up to 128
    cells it is one block (`_pw_block`); longer, numpy splits in two at a
    multiple of eight and adds the halves. (A stack, not a recursion: numba
    cannot load a recursive function back from its cache.)"""
    if n <= 128:
        return _pw_block(a, lo, n)
    slo = np.empty(64, np.int64)
    sn = np.empty(64, np.int64)
    sv = np.zeros(64, np.bool_)
    vals = np.empty(64, np.float32)
    nv = 0
    slo[0] = lo
    sn[0] = n
    sv[0] = False
    sp = 1
    while sp > 0:
        sp -= 1
        l = slo[sp]
        m = sn[sp]
        if sv[sp]:                           # both halves are done: add them
            nv -= 1
            vals[nv - 1] = vals[nv - 1] + vals[nv]
        elif m <= 128:
            vals[nv] = _pw_block(a, l, m)
            nv += 1
        else:
            n2 = m // 2
            n2 -= n2 % 8
            slo[sp] = l                      # this node again, once done
            sn[sp] = m
            sv[sp] = True
            slo[sp + 1] = l + n2             # then the right half
            sn[sp + 1] = m - n2
            sv[sp + 1] = False
            slo[sp + 2] = l                  # and the left half first
            sn[sp + 2] = n2
            sv[sp + 2] = False
            sp += 3
    return vals[0]


@njit(cache=True)
def _owned(keys, lo, hi, key):
    """Is `key` among the sorted `keys[lo:hi]`? (A body's own cells.)"""
    end = hi
    while lo < hi:
        mid = (lo + hi) // 2
        if keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    return lo < end and keys[lo] == key


@njit(cache=True)
def _math_probe(x, mode):
    """The compiled log10 and power, in the shapes `_hear_levels` uses."""
    if mode == 0:
        return np.log10(x)
    return 10.0 ** x


_EXACT = {}                                  # the answer, once asked


def exact_math():
    """Do the compiled log10 and `10.0 ** x` agree, bit for bit, with numpy's
    over the values `_level` feeds them? Asked once, at the first use."""
    if "ok" not in _EXACT:
        ok = HAVE_NUMBA
        if ok:
            rng = np.random.RandomState(7)
            logs = np.exp(rng.uniform(np.log(1e-12), np.log(1e6), 3000))
            logs = np.concatenate([logs, np.linspace(1.0, 300.0, 700),
                                   [3.0, 1.0, 1e-12, 0.5]])
            pows = np.concatenate([rng.uniform(-12.0, 0.0, 3000), [0.0, -0.0]])
            for x in logs:
                if _math_probe(float(x), 0) != np.log10(float(x)):
                    ok = False
                    break
            for x in pows:
                if ok and _math_probe(float(x), 1) != 10.0 ** float(x):
                    ok = False
        _EXACT["ok"] = bool(ok)
    return _EXACT["ok"]


@njit(cache=True)
def _hear_levels(mat, smass, dvl, A, B, DB, DV, PI, QI, okeys, ostart, vox_m,
                 shadow_db, bend_cap, leaf_db_m, floor, hasfloor, least,
                 m_air, m_flesh, m_leaf, level, spread, need):
    """`_level` up to (not including) the hole the straight line may pass
    through and the way round, for every pair k: a sound `DB[k]` at a metre
    from `A[k]`, heard at `B[k]` (both float32), `DV[k]` the distance between
    them in cells as numpy's norm gives it. The bodies of person `PI[k]` and
    person `QI[k]` (-1: nobody) are not in the way; person i's own cells are
    the sorted flat indices `okeys[ostart[i]:ostart[i+1]]`.

    Writes `level[k]` (the level so far), `spread[k]`, and `need[k]`: 1 where
    the way round is worth looking for, 2 where the straight line was all
    there is to ask (the caller then looks for a hole in the line), 0 where
    `_level` would have returned already."""
    nx, ny, nz = mat.shape
    K = A.shape[0]
    one = np.float32(1.0)
    zero = np.float32(0.0)
    thr = np.float32(1e-4)
    nmax = 3
    for k in range(K):
        d = max(abs(B[k, 0] - A[k, 0]), abs(B[k, 1] - A[k, 1]),
                abs(B[k, 2] - A[k, 2]))
        nk = int(np.ceil(d)) + 1
        if nk > nmax:
            nmax = nk
    wmass = np.empty(nmax, np.float32)
    lfill = np.empty(nmax, np.float32)
    for k in range(K):
        spr = DB[k] - 20.0 * np.log10(max(DV[k] * vox_m, 1.0))
        spread[k] = spr
        level[k] = spr
        need[k] = 0
        dx = B[k, 0] - A[k, 0]
        dy = B[k, 1] - A[k, 1]
        dz = B[k, 2] - A[k, 2]
        n = int(np.ceil(max(abs(dx), abs(dy), abs(dz)))) + 1
        if n <= 2:
            continue
        step = 1.0 / (n - 1)
        pi = PI[k]
        qi = QI[k]
        kept = 0
        nw = 0
        nl = 0
        bodies = 0
        prev_wall = False
        prev_body = False
        leak = 1.0
        runmin = one
        for i in range(1, n - 1):
            t = i * step
            x = int(np.rint(np.float64(A[k, 0]) + t * np.float64(dx)))
            y = int(np.rint(np.float64(A[k, 1]) + t * np.float64(dy)))
            z = int(np.rint(np.float64(A[k, 2]) + t * np.float64(dz)))
            x = min(max(x, 0), nx - 1)
            y = min(max(y, 0), ny - 1)
            z = min(max(z, 0), nz - 1)
            key = (x * ny + y) * nz + z
            if pi >= 0 and _owned(okeys, ostart[pi], ostart[pi + 1], key):
                continue
            if qi >= 0 and _owned(okeys, ostart[qi], ostart[qi + 1], key):
                continue
            kept += 1
            m = mat[x, y, z]
            ms = smass[x, y, z]
            f = ms / dvl[m]
            if f < zero:
                f = zero
            if f > one:
                f = one
            solid = f > zero
            is_wall = solid and m != m_air and m != m_flesh and m != m_leaf
            is_body = solid and m == m_flesh
            if solid and m == m_leaf:
                lfill[nl] = f
                nl += 1
            if is_wall:
                wmass[nw] = ms
                nw += 1
                g = one - f
                if g < thr:
                    g = zero
                if prev_wall:
                    if g < runmin:
                        runmin = g
                else:
                    runmin = g
            elif prev_wall:
                leak *= np.float64(runmin)
            prev_wall = is_wall
            if is_body and not prev_body:
                bodies += 1
            prev_body = is_body
        if prev_wall:
            leak *= np.float64(runmin)
        if kept == 0:
            continue
        loss = 0.0
        if nw > 0:
            sigma = np.float64(_pw32(wmass, 0, nw)) / 1000.0 \
                / max(vox_m * vox_m, 1e-9)
            tl = 20.0 * np.log10(max(sigma, 1e-12) * 500.0) - 47.0
            if not tl > 0.0:
                tl = 0.0
            v = leak + 10.0 ** (-tl / 10.0)
            if 1.0 < v:
                v = 1.0
            loss = -10.0 * np.log10(v)
        if bodies > 0:
            v = shadow_db * bodies
            if bend_cap < v:
                v = bend_cap
            loss += v
        if nl > 0:
            leaf_m = np.float64(_pw32(lfill, 0, nl)) * vox_m
            if leaf_m >= 20.0:
                v = leaf_db_m * min(leaf_m, 200.0)
                loss += v
            elif leaf_m >= 10.0:
                loss += 1.0
        level[k] = spr - loss
        need[k] = 2
        if loss - least > 1e-6 and ((not hasfloor) or spr - least >= floor):
            need[k] = 3
