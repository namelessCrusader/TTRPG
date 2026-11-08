"""Exact route sweeps, standing surfaces (levels) and the 2.5-D planner.

WHAT THIS IS (T11, wave 13). `motion._routes` runs a Dijkstra over a dict, a
heapq and a list copy of the plane in pure python: 31 / 133 / 222 ms on an
open plane of 16k / 64k / 102k columns. This module is the same sweep as a
compiled kernel, plus the two things a body that can step needs:

  * `sweep` / `sweep_py`      the flat sweep, python reference and numba kernel
  * `routes_flat`             `_routes` itself (aim at the nearest fit cell,
                              read each path back out), early stop included
  * `surfaces`                the standing surfaces ("levels") of a window of
                              columns: a floor, a loft, up to K of them
  * `sweep25` / `sweep25_py`  Dijkstra on nodes (level, x, y) with a step range
  * `routes25`                the paths of 2-D targets out of that sweep

EXACTNESS (F9). The python sweep pops (cost, node) from a heapq and relaxes a
node only on a STRICT improvement. Any heap that pops by the total order
(cost, node index) therefore settles every node in the same order and leaves
the same `prev[]`. The kernel packs the pair into one int64, `cost * N +
node`, so the two orders are the same order by construction, and a test
checks `cost[]` and `prev[]` element for element. EARLY STOP: the sweep stops
the moment every wanted target has been SETTLED (its path is final); a target
that cannot be reached floods its whole component, which is never more than
the full sweep.

The plain python path is the definition; numba is optional (the repo's
accelerator pattern, `kernels.py`) and a test makes the two agree bit for bit.

WHAT IS NOT HERE, on purpose (spec v2, 4.12): no component test and no fit
cache. The label pass costs about as much as the flood it would save, and a
cache of the fit grid cannot be exact (the grid depends on where the body
stands now and `free`/`known` are edited in place with no version counter).
"""
import heapq
import numpy as np

from .materials import AIR, PUSH_THROUGH, _DENS_ARR

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:                        # pragma: no cover - depends on install
    HAVE_NUMBA = False

    def njit(*args, **kwargs):           # a decorator that does nothing
        def wrap(fn):
            return fn
        return wrap if not args or not callable(args[0]) else args[0]

# The eight steps, grid steps first, in the order motion._STEPS8 has them (the
# first route found wins a tie, so the ORDER is part of the law).
STEPS8 = ((1, 0, 70), (-1, 0, 70), (0, 1, 70), (0, -1, 70),
          (1, 1, 99), (1, -1, 99), (-1, 1, 99), (-1, -1, 99))
_DX = np.array([s[0] for s in STEPS8], np.int64)
_DY = np.array([s[1] for s in STEPS8], np.int64)
_C = np.array([s[2] for s in STEPS8], np.int64)
INF = np.int64(1) << 40                  # "not reached" in cost[]
FLAT_F = np.array([8], np.int64)         # step cost table of a flat world: F[0] = 8


# ── the heap: int64 keys, key = cost * N + node ─────────────────────────────
@njit(cache=True)
def _push(h, n, k):
    i = n
    h[i] = k
    while i > 0:
        p = (i - 1) >> 1
        if h[p] <= h[i]:
            break
        t = h[p]
        h[p] = h[i]
        h[i] = t
        i = p
    return n + 1


@njit(cache=True)
def _pop(h, n):
    top = h[0]
    n -= 1
    h[0] = h[n]
    i = 0
    while True:
        lc = 2 * i + 1
        if lc >= n:
            break
        rc = lc + 1
        m = lc if (rc >= n or h[lc] <= h[rc]) else rc
        if h[i] <= h[m]:
            break
        t = h[m]
        h[m] = h[i]
        h[i] = t
        i = m
    return top, n


# ── the flat sweep ──────────────────────────────────────────────────────────
def sweep_py(ok, nx, ny, s0, want=None, ngroups=0, stats=None):
    """The definition: motion._routes' sweep, as arrays out.

    `ok` is the flat fit plane (node = x * ny + y); `s0` the start node, which
    is never tested. `want` is an int32 array over nodes (0 = not wanted, g > 0
    = a member of target group g); the sweep stops when each of the `ngroups`
    groups has had one node settled. Returns (cost, prev) as int64 arrays of
    the plane's size: INF / -1 where nothing was reached, prev -1 at the start.
    `stats`, if a dict, gets "maxheap" and "pops"."""
    N = nx * ny
    okl = np.asarray(ok).ravel().tolist()
    wl = None if want is None else np.asarray(want).ravel().tolist()
    cost = {s0: 0}
    prev = {s0: -1}
    heap = [(0, s0)]
    left = ngroups
    seen = set()
    maxheap = 1
    pops = 0
    while heap:
        d, cur = heapq.heappop(heap)
        pops += 1
        if d > cost[cur]:
            continue
        if wl is not None:
            g = wl[cur]
            if g and g not in seen:
                seen.add(g)
                left -= 1
                if left == 0:
                    break
        cx, cy = divmod(cur, ny)
        for dx, dy, c in STEPS8:
            x, y = cx + dx, cy + dy
            if not (0 <= x < nx and 0 <= y < ny):
                continue
            nxt = x * ny + y
            if not okl[nxt]:
                continue
            if dx and dy and not (okl[x * ny + cy] and okl[cx * ny + y]):
                continue
            nd = d + c
            if nd < cost.get(nxt, nd + 1):
                cost[nxt] = nd
                prev[nxt] = cur
                heapq.heappush(heap, (nd, nxt))
                if len(heap) > maxheap:
                    maxheap = len(heap)
    if stats is not None:
        stats["maxheap"], stats["pops"] = maxheap, pops
    ca = np.full(N, INF, np.int64)
    pa = np.full(N, -1, np.int64)
    if cost:
        ks = np.fromiter(cost.keys(), np.int64, len(cost))
        ca[ks] = np.fromiter(cost.values(), np.int64, len(cost))
        pa[ks] = np.fromiter((prev[k] for k in cost), np.int64, len(cost))
    return ca, pa


@njit(cache=True)
def _sweep_nb(ok, nx, ny, s0, want, ngroups, dxs, dys, cs):
    N = nx * ny
    big = np.int64(1) << 40
    cost = np.full(N, big, np.int64)
    prev = np.full(N, -1, np.int64)
    heap = np.empty(8 * N + 8, np.int64)
    seen = np.zeros(ngroups + 1, np.bool_)
    cost[s0] = 0
    n = _push(heap, 0, np.int64(s0))
    left = ngroups
    while n > 0:
        k, n = _pop(heap, n)
        d = k // N
        cur = k - d * N
        if d > cost[cur]:
            continue
        g = want[cur]
        if g > 0 and not seen[g]:
            seen[g] = True
            left -= 1
            if left == 0:
                break
        cx = cur // ny
        cy = cur - cx * ny
        for j in range(8):
            x = cx + dxs[j]
            y = cy + dys[j]
            if x < 0 or x >= nx or y < 0 or y >= ny:
                continue
            nxt = x * ny + y
            if not ok[nxt]:
                continue
            if dxs[j] != 0 and dys[j] != 0:
                if not (ok[x * ny + cy] and ok[cx * ny + y]):
                    continue
            nd = d + cs[j]
            if nd < cost[nxt]:
                cost[nxt] = nd
                prev[nxt] = cur
                n = _push(heap, n, nd * N + nxt)
    return cost, prev


def sweep(ok, nx, ny, s0, want=None, ngroups=0, numba=None):
    """The flat sweep: the kernel when numba is there (and `numba` is not
    False), the python reference otherwise. Same arrays either way."""
    if numba is None:
        numba = HAVE_NUMBA
    if not numba:
        return sweep_py(ok, nx, ny, s0, want, ngroups)
    okf = np.ascontiguousarray(ok, dtype=np.bool_).ravel()
    w = np.zeros(nx * ny, np.int32) if want is None else \
        np.ascontiguousarray(want, dtype=np.int32).ravel()
    return _sweep_nb(okf, nx, ny, int(s0), w, int(ngroups), _DX, _DY, _C)


# ── _routes, from the sweep ─────────────────────────────────────────────────
def _aim(fit, targets, cand_cache):
    """Each target's goal: itself if a body fits there, else the nearest fit
    column by Manhattan distance, the first in row-major order of argwhere on a
    tie (the same expression motion._routes uses, so the same cell)."""
    goals = []
    for t in targets:
        goal = (int(t[0]), int(t[1]))
        if not fit[goal]:
            if cand_cache[0] is None:
                cand_cache[0] = np.argwhere(fit)
            cand = cand_cache[0]
            if not len(cand):
                goals.append(None)
                continue
            d = np.abs(cand[:, 0] - goal[0]) + np.abs(cand[:, 1] - goal[1])
            near = cand[int(d.argmin())]
            goal = (int(near[0]), int(near[1]))
        goals.append(goal)
    return goals


def _path_flat(prev, goal, ny):
    cur = goal[0] * ny + goal[1]
    path = []
    pl = prev
    while cur >= 0:
        path.append(divmod(cur, ny))
        cur = int(pl[cur])
    return path[::-1][1:]


def routes_flat(fit, start, targets, numba=None, early_stop=True):
    """`motion._routes` with the exact kernel. `fit` is the fit plane, `start`
    the (x, y) the body stands in (never tested). Returns {(tx, ty): path or
    None} with every path an element-for-element copy of the python answer.

    The goal of a target nobody fits at is the nearest fit column (the
    "aim" step), and the sweep stops once every such goal is settled."""
    fit = np.asarray(fit)
    nx, ny = fit.shape
    s0 = int(start[0]) * ny + int(start[1])
    cache = [None]
    goals = _aim(fit, targets, cache)
    want = np.zeros(nx * ny, np.int32)
    gid = {}
    for g in goals:
        if g is not None and g not in gid:
            gid[g] = len(gid) + 1
            want[g[0] * ny + g[1]] = gid[g]
    cost, prev = sweep(fit, nx, ny, s0, want if early_stop else None,
                       len(gid) if early_stop else 0, numba=numba)
    out = {}
    for t, goal in zip(targets, goals):
        key = (int(t[0]), int(t[1]))
        if goal is None or cost[goal[0] * ny + goal[1]] >= INF:
            out[key] = None
            continue
        out[key] = _path_flat(prev, goal, ny)
    return out


# ── standing surfaces (levels) ──────────────────────────────────────────────
class Surfaces:
    """The standing surfaces of a window of columns.

    `top[k, i, j]` is the standing z (the z of the body's soles; the floor is
    the cell under it) of level k in column (x0 + i, y0 + j), -1 where there is
    none. Level 0 is the lowest. `overflow` counts the columns that had more
    surfaces than K (the extra ones are dropped, not hidden)."""
    __slots__ = ("top", "box", "overflow", "h", "K")

    def __init__(self, top, box, overflow, h, K):
        self.top, self.box, self.overflow, self.h, self.K = \
            top, box, overflow, h, K

    def levels_at(self, x, y):
        x0, _x1, y0, _y1 = self.box
        col = self.top[:, x - x0, y - y0]
        return [int(v) for v in col if v >= 0]


def _denom(vox_l):
    """Per-material divisor of smass, exactly as motion._walkable forms it."""
    return np.maximum(_DENS_ARR * vox_l, 1e-9)


def _own_mask(shape3, box, own):
    """A bool mask of the window with the body's own cells set, or None."""
    if own is None or not len(own):
        return None
    x0, x1, y0, y1 = box
    c = np.asarray(own, np.int64).reshape(-1, 3)
    c = c[(c[:, 0] >= x0) & (c[:, 0] < x1) & (c[:, 1] >= y0) & (c[:, 1] < y1)
          & (c[:, 2] >= 0) & (c[:, 2] < shape3[2])]
    if not len(c):
        return None
    m = np.zeros((x1 - x0, y1 - y0, shape3[2]), np.bool_)
    m[c[:, 0] - x0, c[:, 1] - y0, c[:, 2]] = True
    return m


def surfaces_py(mat, smass, denom, own3, h, K, push):
    """The definition, in numpy. A column has a surface at z when the cell
    below is something (mat != AIR) that is not the body's own flesh and the
    h cells z .. z+h-1 are all sparse enough to occupy (packed < PUSH_THROUGH,
    the `_walkable` rule; the body's own cells count as empty). After a
    surface at z the next one is looked for from z + h up, so a sparse cell
    (a hedge) is not a second floor."""
    bx, by, nz = mat.shape
    clear = (smass / denom[mat]) < push
    sup = mat != AIR
    if own3 is not None:
        clear = clear | own3
        sup = sup & ~own3
    rl = np.zeros((bx, by, nz + 1), np.int32)
    for z in range(nz - 1, -1, -1):
        rl[:, :, z] = np.where(clear[:, :, z], rl[:, :, z + 1] + 1, 0)
    top = np.full((K, bx, by), -1, np.int16)
    cnt = np.zeros((bx, by), np.int64)
    nxt = np.ones((bx, by), np.int64)
    over = np.zeros((bx, by), np.bool_)
    for z in range(1, nz - h + 1):
        cand = sup[:, :, z - 1] & (rl[:, :, z] >= h) & (nxt <= z)
        over |= cand & (cnt >= K)
        take = cand & (cnt < K)
        xs, ys = np.nonzero(take)
        top[cnt[xs, ys], xs, ys] = z
        cnt[xs, ys] += 1
        nxt[xs, ys] = z + h
    return top, int(over.sum())


@njit(cache=True)
def _surf_nb(mat, smass, denom, own3, has_own, h, K, push, top):
    bx, by, nz = mat.shape
    rl = np.zeros(nz + 1, np.int32)
    over = 0
    for x in range(bx):
        for y in range(by):
            rl[nz] = 0
            for z in range(nz - 1, -1, -1):
                c = (smass[x, y, z] / denom[mat[x, y, z]]) < push
                if has_own and own3[x, y, z]:
                    c = True
                rl[z] = rl[z + 1] + 1 if c else 0
            k = 0
            z = 1
            while z <= nz - h:
                s = mat[x, y, z - 1] != 0
                if has_own and own3[x, y, z - 1]:
                    s = False
                if s and rl[z] >= h:
                    if k < K:
                        top[k, x, y] = z
                        k += 1
                        z += h
                        continue
                    over += 1
                    break
                z += 1
    return over


def surfaces(mat, smass, vox_l, h, K=2, box=None, own=None, numba=None):
    """The standing surfaces of the columns in `box` = (x0, x1, y0, y1)
    (default: the whole plane) for a body h cells tall, up to K per column.

    `own` is the body's own flesh as an (n, 3) array of cells: it never counts
    as support and never as an obstacle (the failure of 2026-09). No global
    cache and no change tracking: ask for the window you need, when you look.
    Measured on a synthetic plane (K = 2): 0.7 / 2.7 / 6.3 ms for the whole
    plane at 16k / 64k / 102k columns."""
    nx, ny, nz = mat.shape
    if box is None:
        box = (0, nx, 0, ny)
    x0, x1, y0, y1 = (int(v) for v in box)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, nx), min(y1, ny)
    box = (x0, x1, y0, y1)
    if numba is None:
        numba = HAVE_NUMBA
    m = mat[x0:x1, y0:y1]
    s = smass[x0:x1, y0:y1]
    denom = _denom(vox_l)
    push = denom.dtype.type(PUSH_THROUGH)
    own3 = _own_mask(mat.shape, box, own)
    h = int(h)
    K = int(K)
    if not numba:
        top, over = surfaces_py(m, s, denom, own3, h, K, push)
        return Surfaces(top, box, over, h, K)
    top = np.full((K, x1 - x0, y1 - y0), -1, np.int16)
    has = own3 is not None
    if own3 is None:
        own3 = np.zeros((1, 1, 1), np.bool_)
    over = _surf_nb(m, s, denom, own3, has, h, K, push, top)
    return Surfaces(top, box, int(over), h, K)


# ── footprint, step costs ───────────────────────────────────────────────────
def footprint(cells):
    """The body's footprint as offsets from its centre column, (0, 0) left out
    (a node is its own centre). `cells` is the body's flesh, (n, 3)."""
    c = np.asarray(cells)
    cx = int(round(float(c[:, 0].mean())))
    cy = int(round(float(c[:, 1].mean())))
    offs = sorted({(int(a) - cx, int(b) - cy) for a, b in c[:, :2]} - {(0, 0)})
    fo = np.array([o[0] for o in offs], np.int64)
    fy = np.array([o[1] for o in offs], np.int64)
    return fo, fy


def edge_cost(base, dz, ftab):
    """A step's cost: integer, base * F[|dz|] // 8, F[0] = 8, so a flat edge
    costs exactly today's 70 or 99 and equal routes tie exactly."""
    return int(base) * int(ftab[abs(int(dz))]) // 8


# ── the 2.5-D sweep ─────────────────────────────────────────────────────────
def _arrive_py(T, K, x, y, z, rup, rdn):
    """The level of column (x, y) a body standing at z can step onto: the one
    with the smallest |dz| in [-rdn, +rup] (the lowest level on a tie), or -1."""
    best, bd = -1, 1 << 30
    for k in range(K):
        t = T[k][x][y]
        if t < 0:
            continue
        dz = t - z
        if -rdn <= dz <= rup and abs(dz) < bd:
            best, bd = k, abs(dz)
    return best


def sweep25_py(top, r_up, r_dn, fox, foy, ftab, s0, want=None, ngroups=0,
               stats=None):
    """The definition of the 2.5-D sweep, in plain python.

    Nodes are (level k, x, y), index (k * nx + x) * ny + y. An edge from
    (k, x, y) to a neighbour column exists when that column has a level within
    [-r_dn, +r_up] of the current height AND the body's whole footprint fits
    there: every footprint column needs a level within range of the ARRIVAL
    height. That test is lazy, once per node, and memoised. A diagonal may not
    cut a corner: both side columns must be steppable from the current node.
    The start node is never tested. With K = 1 and r_up = r_dn = 0 this is
    `sweep` over the eroded fit grid, edge for edge."""
    K, nx, ny = top.shape
    N = K * nx * ny
    T = top.tolist()
    fo = list(zip(np.asarray(fox).tolist(), np.asarray(foy).tolist()))
    F = np.asarray(ftab).tolist()
    wl = None if want is None else np.asarray(want).ravel().tolist()
    memo = {s0: True}

    def fits(k, x, y):
        node = (k * nx + x) * ny + y
        r = memo.get(node)
        if r is None:
            z = T[k][x][y]
            r = True
            for ox, oy in fo:
                xx, yy = x + ox, y + oy
                if not (0 <= xx < nx and 0 <= yy < ny) \
                        or _arrive_py(T, K, xx, yy, z, r_up, r_dn) < 0:
                    r = False
                    break
            memo[node] = r
        return r

    cost = {s0: 0}
    prev = {s0: -1}
    heap = [(0, s0)]
    left = ngroups
    seen = set()
    maxheap, pops = 1, 0
    while heap:
        d, cur = heapq.heappop(heap)
        pops += 1
        if d > cost[cur]:
            continue
        if wl is not None:
            g = wl[cur]
            if g and g not in seen:
                seen.add(g)
                left -= 1
                if left == 0:
                    break
        k, rem = divmod(cur, nx * ny)
        cx, cy = divmod(rem, ny)
        z = T[k][cx][cy]
        for dx, dy, c in STEPS8:
            x, y = cx + dx, cy + dy
            if not (0 <= x < nx and 0 <= y < ny):
                continue
            k2 = _arrive_py(T, K, x, y, z, r_up, r_dn)
            if k2 < 0 or not fits(k2, x, y):
                continue
            if dx and dy:
                ka = _arrive_py(T, K, x, cy, z, r_up, r_dn)
                kb = _arrive_py(T, K, cx, y, z, r_up, r_dn)
                if ka < 0 or kb < 0 or not fits(ka, x, cy) \
                        or not fits(kb, cx, y):
                    continue
            nxt = (k2 * nx + x) * ny + y
            nd = d + c * F[abs(T[k2][x][y] - z)] // 8
            if nd < cost.get(nxt, nd + 1):
                cost[nxt] = nd
                prev[nxt] = cur
                heapq.heappush(heap, (nd, nxt))
                if len(heap) > maxheap:
                    maxheap = len(heap)
    if stats is not None:
        stats["maxheap"], stats["pops"] = maxheap, pops
    ca = np.full(N, INF, np.int64)
    pa = np.full(N, -1, np.int64)
    ks = np.fromiter(cost.keys(), np.int64, len(cost))
    ca[ks] = np.fromiter(cost.values(), np.int64, len(cost))
    pa[ks] = np.fromiter((prev[k] for k in cost), np.int64, len(cost))
    return ca, pa


@njit(cache=True)
def _arrive_nb(top, K, x, y, z, rup, rdn):
    best = -1
    bd = 1 << 30
    for k in range(K):
        t = np.int64(top[k, x, y])
        if t < 0:
            continue
        dz = t - z
        if dz >= -rdn and dz <= rup:
            a = dz if dz >= 0 else -dz
            if a < bd:
                best = k
                bd = a
    return best


@njit(cache=True)
def _fits_nb(top, K, nx, ny, k, x, y, rup, rdn, fox, foy, memo):
    node = (k * nx + x) * ny + y
    r = memo[node]
    if r == 0:
        z = np.int64(top[k, x, y])
        r = 1
        for i in range(fox.shape[0]):
            xx = x + fox[i]
            yy = y + foy[i]
            if xx < 0 or xx >= nx or yy < 0 or yy >= ny \
                    or _arrive_nb(top, K, xx, yy, z, rup, rdn) < 0:
                r = 2
                break
        memo[node] = r
    return r == 1


@njit(cache=True)
def _sweep25_nb(top, rup, rdn, fox, foy, ftab, s0, want, ngroups,
                dxs, dys, cs):
    K, nx, ny = top.shape
    N = K * nx * ny
    big = np.int64(1) << 40
    cost = np.full(N, big, np.int64)
    prev = np.full(N, -1, np.int64)
    memo = np.zeros(N, np.int8)
    memo[s0] = 1
    heap = np.empty(8 * N + 8, np.int64)
    seen = np.zeros(ngroups + 1, np.bool_)
    cost[s0] = 0
    n = _push(heap, 0, np.int64(s0))
    left = ngroups
    while n > 0:
        key, n = _pop(heap, n)
        d = key // N
        cur = key - d * N
        if d > cost[cur]:
            continue
        g = want[cur]
        if g > 0 and not seen[g]:
            seen[g] = True
            left -= 1
            if left == 0:
                break
        k = cur // (nx * ny)
        rem = cur - k * nx * ny
        cx = rem // ny
        cy = rem - cx * ny
        z = np.int64(top[k, cx, cy])
        for j in range(8):
            x = cx + dxs[j]
            y = cy + dys[j]
            if x < 0 or x >= nx or y < 0 or y >= ny:
                continue
            k2 = _arrive_nb(top, K, x, y, z, rup, rdn)
            if k2 < 0 or not _fits_nb(top, K, nx, ny, k2, x, y, rup, rdn,
                                      fox, foy, memo):
                continue
            if dxs[j] != 0 and dys[j] != 0:
                ka = _arrive_nb(top, K, x, cy, z, rup, rdn)
                kb = _arrive_nb(top, K, cx, y, z, rup, rdn)
                if ka < 0 or kb < 0:
                    continue
                if not _fits_nb(top, K, nx, ny, ka, x, cy, rup, rdn,
                                fox, foy, memo):
                    continue
                if not _fits_nb(top, K, nx, ny, kb, cx, y, rup, rdn,
                                fox, foy, memo):
                    continue
            nxt = (k2 * nx + x) * ny + y
            dz = np.int64(top[k2, x, y]) - z
            if dz < 0:
                dz = -dz
            nd = d + cs[j] * ftab[dz] // 8
            if nd < cost[nxt]:
                cost[nxt] = nd
                prev[nxt] = cur
                n = _push(heap, n, nd * N + nxt)
    return cost, prev


def sweep25(top, r_up, r_dn, fox, foy, ftab, s0, want=None, ngroups=0,
            numba=None):
    """The 2.5-D sweep: the kernel when numba is there, the python reference
    otherwise. Same arrays either way (see `sweep25_py` for the rules)."""
    if numba is None:
        numba = HAVE_NUMBA
    ftab = np.ascontiguousarray(ftab, np.int64)
    need = max(int(r_up), int(r_dn)) + 1
    if len(ftab) < need:
        raise ValueError("step cost table shorter than the step range")
    if not numba:
        return sweep25_py(top, r_up, r_dn, fox, foy, ftab, s0, want, ngroups)
    top = np.ascontiguousarray(top, np.int16)
    K, nx, ny = top.shape
    w = np.zeros(K * nx * ny, np.int32) if want is None else \
        np.ascontiguousarray(want, np.int32).ravel()
    return _sweep25_nb(top, int(r_up), int(r_dn),
                       np.ascontiguousarray(fox, np.int64),
                       np.ascontiguousarray(foy, np.int64), ftab, int(s0), w,
                       int(ngroups), _DX, _DY, _C)


def start_node(top, x, y, z):
    """The node a body standing at (x, y, z) starts from. If its column has no
    level at z in the belief, the nearest level slot is overwritten (a COPY of
    `top` is returned in that case): the body is standing there, whatever it
    last saw. Returns (node, top)."""
    K, nx, ny = top.shape
    col = top[:, x, y]
    hit = np.flatnonzero(col == z)
    if len(hit):
        k = int(hit[0])
    else:
        top = top.copy()
        free = np.flatnonzero(top[:, x, y] < 0)
        if len(free):
            k = int(free[0])
        else:
            k = int(np.abs(top[:, x, y].astype(np.int64) - z).argmin())
        top[k, x, y] = z
    return (k * nx + x) * ny + y, top


@njit(cache=True)
def _fit_levels_kernel(z, r_up, r_dn, fox, foy):
    K, nx, ny = z.shape
    out = np.zeros((K, nx, ny), np.bool_)
    for k in range(K):
        for x in range(nx):
            for y in range(ny):
                if z[k, x, y] < 0:
                    continue
                good = True
                for i in range(len(fox)):
                    x2 = x + fox[i]
                    y2 = y + foy[i]
                    if x2 < 0 or x2 >= nx or y2 < 0 or y2 >= ny:
                        good = False       # outside the grid: no neighbour
                        break
                    has = False
                    for k2 in range(K):
                        if z[k2, x2, y2] >= 0:
                            dz = z[k2, x2, y2] - z[k, x, y]
                            if dz >= -r_dn and dz <= r_up:
                                has = True
                                break
                    if not has:
                        good = False
                        break
                out[k, x, y] = good
    return out


def fit_levels_py(top, r_up, r_dn, fox, foy, numba=None):
    """For every node, does the footprint fit? bool [K, nx, ny]. Eager, for
    readers that want a plane (`_places`, the menu): the sweep itself is lazy."""
    K, nx, ny = top.shape
    z = top.astype(np.int32)
    out = z >= 0
    for ox, oy in zip(np.asarray(fox).tolist(), np.asarray(foy).tolist()):
        xs = (slice(max(0, -ox), nx - max(0, ox)),
              slice(max(0, -oy), ny - max(0, oy)))
        xt = (slice(max(0, ox), nx - max(0, -ox)),
              slice(max(0, oy), ny - max(0, -oy)))
        # neighbour column at offset (ox, oy) as seen from (x, y)
        has = np.zeros((K, nx, ny), np.bool_)
        for k in range(K):
            for k2 in range(K):
                dz = z[k2][xt] - z[k][xs]
                ok = (z[k2][xt] >= 0) & (dz >= -r_dn) & (dz <= r_up)
                has[k][xs] |= ok
        out &= has
    return out


def fit_levels(top, r_up, r_dn, fox, foy, numba=None):
    """For every node, does the footprint fit? bool [K, nx, ny] (the kernel when
    numba is there; `fit_levels_py` is the definition)."""
    if numba is None:
        numba = HAVE_NUMBA
    if not numba:
        return fit_levels_py(top, r_up, r_dn, fox, foy)
    return _fit_levels_kernel(np.ascontiguousarray(top, np.int32), int(r_up),
                              int(r_dn), np.asarray(fox, np.int64),
                              np.asarray(foy, np.int64))


def routes25(top, start, r_up, r_dn, fox, foy, ftab, targets, fit2d=None,
             numba=None, early_stop=True):
    """The paths of 2-D targets over the levels. A target is a column; its goal
    is itself if some level there fits the body, else the nearest column that
    has a fitting level (the aim step, by Manhattan distance, first in
    row-major order). The sweep stops when the first node of each goal column
    is settled, which Dijkstra guarantees is its cheapest level.

    `start` = (x, y, z). Returns ({target: [(x, y), ...] or None}, cost) where
    cost is {target: integer cost or None}."""
    top = np.asarray(top, np.int16)
    K, nx, ny = top.shape
    s0, top = start_node(top, int(start[0]), int(start[1]), int(start[2]))
    if fit2d is None:
        fit2d = fit_levels(top, r_up, r_dn, fox, foy).any(axis=0)
    cache = [None]
    goals = _aim(fit2d, targets, cache)
    want = np.zeros(K * nx * ny, np.int32)
    gid = {}
    for g in goals:
        if g is None or g in gid:
            continue
        nodes = [(k * nx + g[0]) * ny + g[1] for k in range(K)
                 if top[k, g[0], g[1]] >= 0]
        if nodes:
            gid[g] = len(gid) + 1
            for nd in nodes:
                want[nd] = gid[g]
    cost, prev = sweep25(top, r_up, r_dn, fox, foy, ftab, s0,
                         want if early_stop else None,
                         len(gid) if early_stop else 0, numba=numba)
    paths, costs = {}, {}
    for t, goal in zip(targets, goals):
        key = (int(t[0]), int(t[1]))
        if goal is None:
            paths[key], costs[key] = None, None
            continue
        best, bc = -1, INF
        for k in range(K):
            nd = (k * nx + goal[0]) * ny + goal[1]
            if cost[nd] < bc:
                best, bc = nd, cost[nd]
        if best < 0:
            paths[key], costs[key] = None, None
            continue
        cur, path = best, []
        while cur >= 0:
            rem = cur % (nx * ny)
            path.append(divmod(rem, ny))
            cur = int(prev[cur])
        paths[key] = path[::-1][1:]
        costs[key] = int(bc)
    return paths, costs
