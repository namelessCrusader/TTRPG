"""routing.py: the exact route sweeps, levels (standing surfaces) and the 2.5-D
planner (T11 spec v2, tests T-R1 .. T-R8).

Acceptance tags: [I] independent (a closed form or a copy of the old code),
[C] code against its own table. Nothing here is physical data.
"""
import heapq
import time

import numpy as np
import pytest

from src.voxel import routing as R
from src.voxel import scenes
from src.voxel.materials import FLESH, LEAF, STONE, _DENS_ARR
from src.voxel.sim import World

F_TAB = np.array([8, 30, 50, 70, 90, 110, 130, 150, 170], np.int64)
CROSS = (np.array([1, -1, 0, 0], np.int64), np.array([0, 0, 1, -1], np.int64))


def _old_sweep(fit, start):
    """motion._routes' sweep, copied: the reference this module must equal."""
    nx, ny = fit.shape
    ok = fit.ravel().tolist()
    s0 = start[0] * ny + start[1]
    cost, prev, heap = {s0: 0}, {s0: None}, [(0, s0)]
    while heap:
        d, cur = heapq.heappop(heap)
        if d > cost[cur]:
            continue
        cx, cy = divmod(cur, ny)
        for dx, dy, c in R.STEPS8:
            x, y = cx + dx, cy + dy
            if not (0 <= x < nx and 0 <= y < ny):
                continue
            nxt = x * ny + y
            if not ok[nxt]:
                continue
            if dx and dy and not (ok[x * ny + cy] and ok[cx * ny + y]):
                continue
            nd = d + c
            if nd < cost.get(nxt, nd + 1):
                cost[nxt], prev[nxt] = nd, cur
                heapq.heappush(heap, (nd, nxt))
    return cost, prev


def _old_routes(fit, start, targets):
    """motion._routes' tail, copied."""
    nx, ny = fit.shape
    cost, prev = _old_sweep(fit, start)
    cand, out = None, {}
    for t in targets:
        goal = (int(t[0]), int(t[1]))
        if not fit[goal]:
            if cand is None:
                cand = np.argwhere(fit)
            if not len(cand):
                out[(int(t[0]), int(t[1]))] = None
                continue
            d = np.abs(cand[:, 0] - goal[0]) + np.abs(cand[:, 1] - goal[1])
            near = cand[int(d.argmin())]
            goal = (int(near[0]), int(near[1]))
        cur = goal[0] * ny + goal[1]
        if cur not in prev:
            out[(int(t[0]), int(t[1]))] = None
            continue
        path = []
        while cur is not None:
            path.append(divmod(cur, ny))
            cur = prev[cur]
        out[(int(t[0]), int(t[1]))] = path[::-1][1:]
    return out


def _random_fit(seed, nx=40, ny=30, dens=0.25):
    r = np.random.default_rng(seed)
    g = r.random((nx, ny)) > dens
    return g


# ── T-R1: numba and python give the same arrays ─────────────────────────────
@pytest.mark.skipif(not R.HAVE_NUMBA, reason="numba not installed")
def test_NUMBA_and_PYTHON_flat_sweeps_agree_element_for_element():
    for seed, dens in zip(range(4), (0.1, 0.25, 0.35, 0.0)):
        g = _random_fit(seed, dens=dens)
        ny = g.shape[1]
        s0 = 5 * ny + 5
        a = R.sweep_py(g, *g.shape, s0)
        b = R.sweep(g, *g.shape, s0, numba=True)
        assert (a[0] == b[0]).all() and (a[1] == b[1]).all()


def _levels(seed, K, nx=24, ny=20):
    r = np.random.default_rng(seed)
    top = np.full((K, nx, ny), -1, np.int16)
    top[0] = np.where(r.random((nx, ny)) > 0.1, r.integers(1, 5, (nx, ny)), -1)
    if K > 1:
        top[1] = np.where(r.random((nx, ny)) > 0.6, r.integers(8, 11, (nx, ny)), -1)
    top[0, 3, 3] = 2
    return top


@pytest.mark.skipif(not R.HAVE_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("K", [1, 2])
def test_NUMBA_and_PYTHON_25D_sweeps_agree_for_K1_and_K2(K):
    for seed in range(4):
        top = _levels(seed, K)
        ny = top.shape[2]
        a = R.sweep25_py(top, 3, 3, *CROSS, F_TAB, 3 * ny + 3)
        b = R.sweep25(top, 3, 3, *CROSS, F_TAB, 3 * ny + 3, numba=True)
        assert (a[0] == b[0]).all() and (a[1] == b[1]).all()
        assert (a[0] < R.INF).sum() > 5
        # and with an early stop on a group of nodes
        want = np.zeros(top.size, np.int32)
        reach = np.flatnonzero(a[0] < R.INF)
        want[reach[len(reach) // 2]] = 1
        a = R.sweep25_py(top, 3, 3, *CROSS, F_TAB, 3 * ny + 3, want, 1)
        b = R.sweep25(top, 3, 3, *CROSS, F_TAB, 3 * ny + 3, want, 1, numba=True)
        assert (a[0] == b[0]).all() and (a[1] == b[1]).all()


# ── T-R2: equal to the base18 _routes, on a real World ──────────────────────
def _world(seed):
    """A small room with random baffles and an enclosed pocket, one person."""
    r = np.random.default_rng(seed)
    w = World(48, 36, 34, voxel_cm=5)
    w.fill(0, 48, 0, 36, 0, 1, STONE)
    for _ in range(7):
        x0, y0 = int(r.integers(18, 32)), int(r.integers(2, 32))
        if r.random() < 0.5:
            w.fill(x0, x0 + int(r.integers(1, 10)), y0, y0 + 1, 1, 33, STONE)
        else:
            w.fill(x0, x0 + 1, y0, y0 + int(r.integers(1, 10)), 1, 33, STONE)
    # a sealed pocket: a ring the body cannot enter, a different component
    w.fill(33, 48, 19, 20, 1, 33, STONE)
    w.fill(33, 48, 35, 36, 1, 33, STONE)
    w.fill(33, 34, 19, 36, 1, 33, STONE)
    scenes._person(w, 10, 10)
    w.step()
    p = w.persons[0]
    comp, sl = w._person_cells(p)
    return w, p, w._claim_cells(p, comp, sl)


def _targets(seed, shape):
    r = np.random.default_rng(1000 + seed)
    t = [(int(r.integers(0, shape[0])), int(r.integers(0, shape[1])))
         for _ in range(16)]
    return t + [(41, 28), (11, 11), (20, 20), (0, 0)]       # pocket, start, ...


def test_the_new_sweep_equals_motion_routes_for_20_targets_on_real_worlds():
    n_none = n_path = 0
    for seed in range(3):
        w, p, cells = _world(seed)
        fit, start = w._fit_grid(cells, p)
        targets = _targets(seed, fit.shape)
        old = w._routes(cells, targets, p, fit=fit)
        assert old == _old_routes(fit, start, targets)       # the copy is the code
        new = R.routes_flat(fit, start, targets)
        assert new == old
        n_none += sum(v is None for v in old.values())
        n_path += sum(bool(v) for v in old.values())
        assert old[(11, 11)] == []                           # goal == start
        assert old[(41, 28)] is None                         # another component
        # the START NOT IN FIT: the sweep starts anyway
        fit2 = fit.copy()
        fit2[start] = False
        assert R.routes_flat(fit2, start, targets) == \
            w._routes(cells, targets, p, fit=fit2)
        # a BODY THAT HAS JUST MOVED
        moved = cells + np.array([1, 0, 0], cells.dtype)
        fit3, start3 = w._fit_grid(moved, p)
        assert R.routes_flat(fit3, start3, targets) == \
            w._routes(moved, targets, p, fit=fit3)
        # without the early stop, and on the python reference
        assert R.routes_flat(fit, start, targets, early_stop=False) == old
        assert R.routes_flat(fit, start, targets, numba=False) == old
    assert n_none > 0 and n_path > 20                          # the cases were hit


def test_a_goal_inside_a_wall_aims_at_the_first_nearest_cell_like_the_old_code():
    # a wall slab with a goal in its middle: equal Manhattan ties everywhere
    fit = np.ones((30, 30), bool)
    fit[10:20, 10:20] = False
    targets = [(15, 15), (10, 10), (19, 19), (12, 15)]
    assert R.routes_flat(fit, (2, 2), targets) == _old_routes(fit, (2, 2), targets)


def test_a_FLOODED_sweep_has_the_same_prev_as_the_old_one_everywhere():
    for seed in range(3):
        fit = _random_fit(seed + 10, 36, 28, 0.2)
        start = (4, 4)
        fit[start] = True
        cost, prev = _old_sweep(fit, start)
        ny = fit.shape[1]
        c2, p2 = R.sweep(fit, *fit.shape, start[0] * ny + start[1])
        assert (c2 < R.INF).sum() == len(cost)
        for node, d in cost.items():
            assert c2[node] == d
            assert p2[node] == (-1 if prev[node] is None else prev[node])


# ── T-R3: early stop ────────────────────────────────────────────────────────
def test_EARLY_STOP_keeps_the_paths_it_settled_and_the_heap_stays_under_8N():
    fit = _random_fit(77, 50, 40, 0.15)
    ny = fit.shape[1]
    s0 = 3 * ny + 3
    fit[3, 3] = True
    full_c, full_p = R.sweep(fit, *fit.shape, s0)
    reach = np.flatnonzero(full_c < R.INF)
    tg = [int(reach[len(reach) // 3]), int(reach[len(reach) // 2])]
    want = np.zeros(fit.size, np.int32)
    want[tg[0]], want[tg[1]] = 1, 2
    c, p = R.sweep(fit, *fit.shape, s0, want, 2)
    assert (c < R.INF).sum() < (full_c < R.INF).sum()          # it did stop early
    for t in tg:                                               # settled: final
        cur = t
        while cur != s0:
            assert p[cur] == full_p[cur] and c[cur] == full_c[cur]
            cur = int(p[cur])
    stats = {}
    R.sweep_py(fit, *fit.shape, s0, stats=stats)
    assert stats["maxheap"] <= 8 * fit.size
    # a target nobody can reach floods its component and is None
    fit[30:33, :] = False
    assert R.routes_flat(fit, (3, 3), [(40, 20)]) == _old_routes(fit, (3, 3), [(40, 20)])


# ── surfaces (levels) ───────────────────────────────────────────────────────
VOX_L = 0.125                                                  # 5 cm voxels


def _blank(nx, ny, nz):
    return np.zeros((nx, ny, nz), np.uint8), np.zeros((nx, ny, nz), np.float32)


def _solid(mat, smass, x0, x1, y0, y1, z0, z1, m=STONE, frac=1.0):
    mat[x0:x1, y0:y1, z0:z1] = m
    smass[x0:x1, y0:y1, z0:z1] = _DENS_ARR[m] * VOX_L * frac


def _foot3():
    return (np.array([-1, -1, -1, 0, 0, 1, 1, 1], np.int64),
            np.array([-1, 0, 1, -1, 1, -1, 0, 1], np.int64))


def _fit_flat(top0, fox, foy, nx, ny):
    """The old _fit_grid erosion of a flat walkable plane."""
    walk = top0 >= 0
    fit = walk.copy()
    for dx, dy in zip(fox.tolist(), foy.tolist()):
        sh = np.zeros_like(walk)
        xl, xh = max(0, -dx), min(nx, nx - dx)
        yl, yh = max(0, -dy), min(ny, ny - dy)
        sh[xl:xh, yl:yh] = walk[xl + dx:xh + dx, yl + dy:yh + dy]
        fit &= sh
    return fit


def test_K1_R0_sweep25_equals_the_flat_sweep_over_the_eroded_fit():
    for seed in range(3):
        r = np.random.default_rng(seed)
        nx, ny = 32, 24
        top = np.where(r.random((nx, ny)) > 0.2, 1, -1).astype(np.int16)[None]
        fox, foy = _foot3()
        s = (6, 6)
        top[0, s[0], s[1]] = 1
        fit = _fit_flat(top[0], fox, foy, nx, ny)
        fit[s] = True                                    # (the start is never tested)
        for numba in ([False, True] if R.HAVE_NUMBA else [False]):
            c1, p1 = R.sweep25(top, 0, 0, fox, foy, R.FLAT_F, s[0] * ny + s[1],
                               numba=numba)
            c2, p2 = R.sweep(fit, nx, ny, s[0] * ny + s[1], numba=numba)
            assert (c1 == c2).all() and (p1 == p2).all()


def _kerb_world(height, nx=40, ny=20, nz=20):
    mat, sm = _blank(nx, ny, nz)
    _solid(mat, sm, 0, nx, 0, ny, 0, 1)                         # floor, stand at 1
    _solid(mat, sm, 20, nx, 0, ny, 1, 1 + height)               # a kerb right across
    return mat, sm


def test_a_4_cell_kerb_crosses_with_R_up_6_and_not_with_3():
    mat, sm = _kerb_world(4)
    s = R.surfaces(mat, sm, VOX_L, 6, K=2)
    assert s.overflow == 0 and (s.top[1] < 0).all()
    fox, foy = _foot3()
    tg = [(30, 10)]
    for numba in ([False, True] if R.HAVE_NUMBA else [False]):
        ok, cost = R.routes25(s.top, (5, 10, 1), 6, 6, fox, foy, F_TAB, tg,
                              numba=numba)
        assert ok[(30, 10)], "the kerb should cross with R_up = 6"
        # up only costs more than the flat distance: the step table is in it
        assert cost[(30, 10)] > 25 * 70
        no, _ = R.routes25(s.top, (5, 10, 1), 3, 3, fox, foy, F_TAB, tg,
                           numba=numba)
        assert no[(30, 10)] is None
    # and back down needs R_dn
    back, _ = R.routes25(s.top, (30, 10, 5), 6, 6, fox, foy, F_TAB, [(5, 10)])
    assert back[(5, 10)]
    back, _ = R.routes25(s.top, (30, 10, 5), 6, 3, fox, foy, F_TAB, [(5, 10)])
    assert back[(5, 10)] is None


def _stairs_world(riser, n=12, tread=5, nx=None, ny=20, nz=80):
    nx = nx or (n * tread + 24)
    mat, sm = _blank(nx, ny, nz)
    _solid(mat, sm, 0, nx, 0, ny, 0, 1)
    for i in range(n):
        x0 = 10 + tread * i
        _solid(mat, sm, x0, nx, 0, ny, 1, 1 + riser * (i + 1))
    return mat, sm, 10 + tread * n


def test_a_12_step_flight_routes_with_R_6_and_not_with_a_7_cell_riser():
    fox, foy = _foot3()
    mat, sm, xtop = _stairs_world(4)
    s = R.surfaces(mat, sm, VOX_L, 6, K=2)
    ztop = 1 + 4 * 12
    tgt = [(xtop + 8, 10)]
    up, _ = R.routes25(s.top, (3, 10, 1), 6, 6, fox, foy, F_TAB, tgt)
    assert up[tgt[0]], "bottom to top"
    assert len(up[tgt[0]]) > 60
    dn, _ = R.routes25(s.top, (xtop + 8, 10, ztop), 6, 6, fox, foy, F_TAB, [(3, 10)])
    assert dn[(3, 10)], "and the reverse"
    mat7, sm7, xtop7 = _stairs_world(7)
    s7 = R.surfaces(mat7, sm7, VOX_L, 6, K=2)
    no, _ = R.routes25(s7.top, (3, 10, 1), 6, 6, fox, foy, F_TAB, [(xtop7 + 8, 10)])
    assert no[(xtop7 + 8, 10)] is None
    # a flight of risers 6 with only R_up 5 is shut; the path costs follow F
    no, _ = R.routes25(s.top, (3, 10, 1), 3, 6, fox, foy, F_TAB, tgt)
    assert no[tgt[0]] is None


def _loft_world():
    nx, ny, nz = 50, 24, 40
    mat, sm = _blank(nx, ny, nz)
    _solid(mat, sm, 0, nx, 0, ny, 0, 1)
    for i in range(5):                                          # stairs, risers 2, treads 3
        x0 = 5 + 3 * i
        _solid(mat, sm, x0, 20, 10, 16, 1, 1 + 2 * (i + 1))
    _solid(mat, sm, 20, 45, 0, ny, 10, 11)                      # the loft slab over the floor
    return mat, sm


def test_a_LOFT_over_floor_space_gives_separate_level_nodes_and_a_route_up_and_back():
    mat, sm = _loft_world()
    s = R.surfaces(mat, sm, VOX_L, 6, K=2)
    assert s.overflow == 0
    assert s.top[0, 30, 5] == 1 and s.top[1, 30, 5] == 11         # floor under, loft over
    assert s.top[0, 18, 12] == 11 and s.top[1, 18, 12] == -1       # the stair top: one level
    fox, foy = _foot3()
    under, over = (30, 5), (30, 5)
    tgt = [over]
    # from the floor under the loft to the loft above the same columns
    paths, cost = R.routes25(s.top, (30, 5, 1), 3, 3, fox, foy, F_TAB, tgt)
    # the target column's cheapest level is the floor itself (we are standing on it)
    assert paths[over] == []
    # route to a column over the loft's far end: must go out, up the stairs, over
    up, _ = R.routes25(s.top, (30, 5, 1), 3, 3, fox, foy, F_TAB,
                       [(40, 5)], numba=False)
    # the floor under the loft is also at (40, 5): the cheaper (floor) level wins.
    assert up[(40, 5)]
    # force the loft level: a start already on the loft goes along it, and the
    # path never touches the stair columns when the loft is the shorter way
    lp, _ = R.routes25(s.top, (22, 5, 11), 3, 3, fox, foy, F_TAB, [(40, 5)])
    assert lp[(40, 5)] and all(x >= 21 for x, _y in lp[(40, 5)])
    # from the floor under the loft to a spot only the loft has (over the stair
    # head): the route must use the stairs
    top = s.top.copy()
    top[0, 30:45, :] = -1                                         # remove the floor under the loft
    top[0, 30:45, :] = top[1, 30:45, :]
    top[1, 30:45, :] = -1
    # (the loft is the only level there now): from the floor at x=3 to x=40
    up, _ = R.routes25(top, (3, 5, 1), 3, 3, fox, foy, F_TAB, [(40, 12)])
    assert up[(40, 12)], "under, up the stairs, over"
    assert any(5 <= x < 20 and 10 <= y < 16 for x, y in up[(40, 12)])
    back, _ = R.routes25(top, (40, 12, 11), 3, 3, fox, foy, F_TAB, [(3, 5)])
    assert back[(3, 5)], "and the reverse"


def test_own_flesh_never_counts_as_support_or_as_an_obstacle():
    mat, sm = _blank(20, 20, 40)
    _solid(mat, sm, 0, 20, 0, 20, 0, 1)
    # a body 6 cells tall standing at (10, 10), 3 x 3
    cells = np.array([(x, y, z) for x in range(9, 12) for y in range(9, 12)
                      for z in range(1, 7)])
    mat[tuple(cells.T)] = FLESH
    sm[tuple(cells.T)] = _DENS_ARR[FLESH] * VOX_L
    without = R.surfaces(mat, sm, VOX_L, 6, K=2)
    assert without.top[0, 10, 10] == 7                  # standing on its own head
    withown = R.surfaces(mat, sm, VOX_L, 6, K=2, own=cells)
    assert withown.top[0, 10, 10] == 1 and withown.top[1, 10, 10] == -1
    assert (withown.top[0, 9:12, 9:12] == 1).all()
    # the neighbours do not change
    assert (withown.top[0, :9] == 1).all()


def test_a_sparse_hedge_does_not_block_and_is_not_a_second_floor():
    mat, sm = _blank(10, 10, 30)
    _solid(mat, sm, 0, 10, 0, 10, 0, 1)
    _solid(mat, sm, 4, 6, 4, 6, 1, 5, m=LEAF, frac=0.2)          # packed 0.2 < 0.4
    s = R.surfaces(mat, sm, VOX_L, 6, K=2)
    assert s.top[0, 4, 4] == 1 and s.top[1, 4, 4] == -1
    _solid(mat, sm, 4, 6, 4, 6, 1, 5, m=LEAF, frac=0.9)          # a thicket
    s = R.surfaces(mat, sm, VOX_L, 6, K=2)
    assert s.top[0, 4, 4] != 1


def test_overflow_is_counted_when_a_column_has_more_than_K_surfaces():
    mat, sm = _blank(6, 6, 60)
    _solid(mat, sm, 0, 6, 0, 6, 0, 1)
    _solid(mat, sm, 0, 3, 0, 6, 12, 13)
    _solid(mat, sm, 0, 3, 0, 6, 24, 25)
    s = R.surfaces(mat, sm, VOX_L, 6, K=2)
    assert s.overflow == 18 and (s.top[:, 0, 0] == [1, 13]).all()
    s3 = R.surfaces(mat, sm, VOX_L, 6, K=3)
    assert s3.overflow == 0 and (s3.top[:, 0, 0] == [1, 13, 25]).all()


def test_a_WINDOW_scan_equals_the_same_columns_of_the_whole_plane_after_edits():
    r = np.random.default_rng(5)
    mat, sm = _blank(30, 26, 36)
    _solid(mat, sm, 0, 30, 0, 26, 0, 1)
    for _ in range(40):                                          # random edits
        x, y, z = int(r.integers(0, 30)), int(r.integers(0, 26)), int(r.integers(1, 30))
        m = [STONE, LEAF][int(r.integers(0, 2))]
        _solid(mat, sm, x, x + int(r.integers(1, 5)), y, y + int(r.integers(1, 5)),
               z, z + int(r.integers(1, 3)), m=m, frac=float(r.random()))
    for numba in ([False, True] if R.HAVE_NUMBA else [False]):
        whole = R.surfaces(mat, sm, VOX_L, 6, K=2, numba=numba)
        for box in ((0, 30, 0, 26), (5, 17, 3, 20), (28, 40, 20, 40), (0, 1, 0, 1)):
            win = R.surfaces(mat, sm, VOX_L, 6, K=2, box=box, numba=numba)
            x0, x1, y0, y1 = win.box
            assert (win.top == whole.top[:, x0:x1, y0:y1]).all()
    if R.HAVE_NUMBA:
        a = R.surfaces(mat, sm, VOX_L, 6, K=2, numba=False)
        b = R.surfaces(mat, sm, VOX_L, 6, K=2, numba=True)
        assert (a.top == b.top).all() and a.overflow == b.overflow


def test_surfaces_with_one_level_equal_motion_walkable_on_a_real_world():
    w, p, cells = _world(1)
    flat = w._walkable(cells)
    zlo, zhi = int(cells[:, 2].min()), int(cells[:, 2].max())
    s = R.surfaces(w.mat, w.smass, w.vox_l, zhi - zlo + 1, K=1, own=cells)
    mine = s.top[0] == zlo
    # the old rule includes the body's own flesh as blocking, so compare off it
    own_cols = {(int(a), int(b)) for a, b, _ in cells}
    for x in range(flat.shape[0]):
        for y in range(flat.shape[1]):
            if (x, y) in own_cols:
                continue
            assert bool(flat[x, y]) == bool(mine[x, y]), (x, y)


def test_footprint_and_edge_cost():
    cells = np.array([(x, y, 1) for x in range(9, 12) for y in range(9, 12)])
    fox, foy = R.footprint(cells)
    assert len(fox) == 8 and (0, 0) not in set(zip(fox.tolist(), foy.tolist()))
    assert R.edge_cost(70, 0, R.FLAT_F) == 70 and R.edge_cost(99, 0, F_TAB) == 99
    assert R.edge_cost(70, 2, F_TAB) == 70 * 50 // 8


# ── T-R8: the speed ratio ───────────────────────────────────────────────────
@pytest.mark.skipif(not R.HAVE_NUMBA, reason="numba not installed")
def test_the_NUMBA_sweep_is_at_least_10x_faster_than_python_in_the_same_process():
    nx, ny = 320, 320                                           # 102k columns, open
    g = np.ones((nx, ny), bool)
    s0 = (nx // 2) * ny + ny // 2
    R.sweep(g, nx, ny, s0, numba=True)                          # compile
    tp = tn = 1e9
    for _ in range(2):
        t = time.perf_counter()
        R.sweep_py(g, nx, ny, s0)
        tp = min(tp, time.perf_counter() - t)
    for _ in range(3):
        t = time.perf_counter()
        R.sweep(g, nx, ny, s0, numba=True)
        tn = min(tn, time.perf_counter() - t)
    assert tp / tn >= 10.0, (tp, tn)


def test_the_COMPILED_fit_levels_is_the_PLAIN_one_bit_for_bit():
    """`fit_levels_py` is the definition; the numba kernel must give the same
    plane for any levels, step range and footprint, edges and dz == r_up included."""
    import numpy as np
    from src.voxel import routing
    if not routing.HAVE_NUMBA:
        return
    rng = np.random.default_rng(13)
    for case in range(300):
        K = int(rng.integers(1, 4))
        nx, ny = int(rng.integers(1, 9)), int(rng.integers(1, 9))
        top = rng.integers(-1, 7, size=(K, nx, ny)).astype(np.int32)
        r_up, r_dn = int(rng.integers(0, 4)), int(rng.integers(0, 4))
        n = int(rng.integers(1, 6))
        fox = rng.integers(-2, 3, size=n)
        foy = rng.integers(-2, 3, size=n)
        a = routing.fit_levels(top, r_up, r_dn, fox, foy, numba=True)
        b = routing.fit_levels_py(top, r_up, r_dn, fox, foy)
        assert a.dtype == b.dtype and np.array_equal(a, b), case
