"""rasterize's per-cell sums over the WORLD (matter.py:238-267) vs over only
the occupied cells (np.unique + inverse). bincount adds in input order per
bin either way, so the sums should be bit-identical; the majority rules
(`win = g > best_g`, first material wins ties) likewise. And the cost."""
import time
import numpy as np
rng = np.random.default_rng(0)
shape = (100, 80, 50)                          # fire_alarm / house_fire
n = int(np.prod(shape))
N = 3662                                       # tree_fell's tree
c = rng.integers((40, 30, 1), (60, 50, 40), (N, 3))
flat = np.ravel_multi_index(tuple(c.T), shape)
m = rng.random(N); E = rng.random(N) * 50; vm = rng.integers(1, 4, N).astype(np.int16)
eid = rng.integers(1, 40, N).astype(np.int32)

def world_way():
    grams = np.bincount(flat, m * 1000.0, n)
    best, bg = np.zeros(n, np.int16), np.zeros(n)
    for v in np.unique(vm):
        g = np.bincount(flat, np.where(vm == v, m, 0.0), n); win = g > bg
        best[win], bg[win] = v, g[win]
    be, beg = np.zeros(n, np.int32), np.zeros(n)
    for e in np.unique(eid):
        g = np.bincount(flat, np.where(eid == e, m, 0.0), n); win = g > beg
        be[win], beg[win] = e, g[win]
    return grams, best, be, np.bincount(flat, E, n)

def sparse_way():
    u, inv = np.unique(flat, return_inverse=True); k = len(u)
    grams = np.bincount(inv, m * 1000.0, k)
    best, bg = np.zeros(k, np.int16), np.zeros(k)
    for v in np.unique(vm):
        g = np.bincount(inv, np.where(vm == v, m, 0.0), k); win = g > bg
        best[win], bg[win] = v, g[win]
    # majority eid without one pass per eid: sort by (cell, eid), sum runs
    o = np.lexsort((eid, inv)); ci, ei, mi = inv[o], eid[o], m[o]
    run = np.flatnonzero(np.r_[True, (ci[1:] != ci[:-1]) | (ei[1:] != ei[:-1])])
    rs = np.add.reduceat(mi, run)              # NOTE: summation order differs from bincount
    rc, re_ = ci[run], ei[run]
    be, beg = np.zeros(k, np.int32), np.full(k, -1.0)
    for cc, ee, ss in zip(rc, re_, rs):       # tiny: one row per (cell, eid) pair
        if ss > beg[cc]: be[cc], beg[cc] = ee, ss
    return u, grams, best, be, np.bincount(inv, E, k)

t0 = time.perf_counter(); a = world_way(); t1 = time.perf_counter(); b = sparse_way(); t2 = time.perf_counter()
u = b[0]
print("grams bit-identical:", np.array_equal(a[0][u], b[1]), "| untouched cells all zero:", not a[0][np.setdiff1d(np.arange(n), u)].any())
print("majority material identical:", np.array_equal(a[1][u], b[2]))
print("majority eid identical:", np.array_equal(a[2][u], b[3]))
print("E bit-identical:", np.array_equal(a[3][u], b[4]))
print(f"world-sized: {1e3*(t1-t0):.1f} ms  ({len(np.unique(eid))} eids, 3 materials) | sparse: {1e3*(t2-t1):.1f} ms")
t0 = time.perf_counter(); np.bincount(flat, m, n); per = (time.perf_counter() - t0) * 1e3
print(f"one world-sized bincount at {n} cells: {per:.2f} ms, {n*8/1e6:.1f} MB")
