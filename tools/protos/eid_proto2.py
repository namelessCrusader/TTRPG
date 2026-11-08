"""Scratch prototype #2 (design phase; synthetic arrays, no World built).

1. A numba port of `_object_at` (sim.py:8318-8342) keyed on (mat, eid) that
   returns the SAME CELLS IN THE SAME ORDER as the Python DFS when every eid
   is 0 -- order matters downstream (obj[0] is the event word at
   sim.py:7092-7094; cell order feeds `_launch`).
2. The rasterize eid rule: per drawn cell, the eid (0 included) with the most
   particle mass AMONG PARTICLES OF THE CELL'S WINNING MATERIAL, ties to the
   lowest eid -- checked against brute force; cost at 20k particles.
3. Hygiene cost: clear edge/rubble/eid where a cell became AIR since the last
   pass, at house size (100x80x50).
"""
import time
import numpy as np
from numba import njit

AIR = 0


def object_at_py(mat, x, y, z, cap=4000):            # sim.py:8318-8342 verbatim
    m0 = int(mat[x, y, z])
    if m0 == AIR:
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
            if int(mat[t]) == m0:
                seen.add(t)
                stack.append(t)
    return np.array(out, np.int32)


@njit(cache=False)
def object_at_nb(key, x, y, z, cap):
    """Same traversal: LIFO stack, neighbours pushed in the same fixed order,
    a cell is emitted when popped, `seen` marked when pushed."""
    nx, ny, nz = key.shape
    k0 = key[x, y, z]
    seen = np.zeros(key.shape, np.bool_)
    stack = np.empty((nx * ny * nz, 3), np.int32)
    out = np.empty((min(cap, nx * ny * nz), 3), np.int32)
    sp, n = 0, 0
    stack[0, 0], stack[0, 1], stack[0, 2] = x, y, z
    sp = 1
    seen[x, y, z] = True
    D = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    while sp > 0 and n < cap:
        sp -= 1
        cx, cy, cz = stack[sp, 0], stack[sp, 1], stack[sp, 2]
        out[n, 0], out[n, 1], out[n, 2] = cx, cy, cz
        n += 1
        for d in D:
            tx, ty, tz = cx + d[0], cy + d[1], cz + d[2]
            if tx < 0 or ty < 0 or tz < 0 or tx >= nx or ty >= ny or tz >= nz:
                continue
            if seen[tx, ty, tz]:
                continue
            if key[tx, ty, tz] == k0:
                seen[tx, ty, tz] = True
                stack[sp, 0], stack[sp, 1], stack[sp, 2] = tx, ty, tz
                sp += 1
    return out[:n]


def raster_eid(flat, vmat, m, peid, best_mat, n):
    """Vectorised rule. Returns eid per flat cell (0 where none)."""
    ok = vmat == best_mat[flat]                   # only the winning material
    f, e, w = flat[ok], peid[ok].astype(np.int64), m[ok]
    if not len(f):
        return np.zeros(n, np.int32)
    E = int(e.max()) + 1
    key = f * E + e
    u, inv = np.unique(key, return_inverse=True)
    g = np.bincount(inv, w)
    uf, ue = u // E, u % E
    order = np.lexsort((ue, -g, uf))              # by cell, heaviest, lowest eid
    uf, ue = uf[order], ue[order]
    first = np.ones(len(uf), bool)
    first[1:] = uf[1:] != uf[:-1]
    out = np.zeros(n, np.int32)
    out[uf[first]] = ue[first]
    return out


def raster_eid_brute(flat, vmat, m, peid, best_mat, n):
    out = np.zeros(n, np.int32)
    for c in np.unique(flat):
        sel = (flat == c) & (vmat == best_mat[c])
        tot = {}
        for e, w in zip(peid[sel], m[sel]):
            tot[int(e)] = tot.get(int(e), 0.0) + float(w)
        if tot:
            best = max(tot.values())
            out[c] = min(e for e, w in tot.items() if w == best)
    return out


def main():
    rng = np.random.default_rng(1)
    # 1. flood order golden
    bad, t_py, t_nb, cells = 0, 0.0, 0.0, 0
    for t in range(40):
        shape = tuple(int(v) for v in rng.integers(5, 30, 3))
        mat = np.zeros(shape, np.uint8)
        for _ in range(30):
            lo = rng.integers(0, np.array(shape))
            sz = rng.integers(1, 9, 3)
            mat[tuple(slice(int(l), int(min(l + s, q))) for l, s, q in zip(lo, sz, shape))] = rng.integers(1, 4)
        seeds = np.argwhere(mat != AIR)
        if not len(seeds):
            continue
        for s in seeds[rng.integers(0, len(seeds), 5)]:
            cap = int(rng.choice([4000, 50]))
            t0 = time.perf_counter(); a = object_at_py(mat, *map(int, s), cap=cap)
            t1 = time.perf_counter(); b = object_at_nb(mat.astype(np.int64), *map(int, s), cap)
            t2 = time.perf_counter()
            if t > 0:
                t_py += t1 - t0; t_nb += t2 - t1; cells += len(a)
            bad += not np.array_equal(a, b)
    print(f"flood golden: order mismatches {bad}; python {1e6 * t_py / max(cells, 1):.2f} us/cell, "
          f"numba {1e6 * t_nb / max(cells, 1):.3f} us/cell ({cells} cells)")
    # 2. rasterize eid rule
    bad = 0
    for t in range(30):
        n = 200
        k = 2000
        flat = rng.integers(0, n, k)
        vmat = rng.integers(1, 4, k).astype(np.int16)
        m = rng.choice([0.5, 1.0, 1.5], k)
        peid = np.where(rng.random(k) < 0.5, rng.integers(1, 5, k), 0)
        best = np.zeros(n, np.int16)
        bg = np.zeros(n)
        for vm in np.unique(vmat):
            g = np.bincount(flat, np.where(vmat == vm, m, 0.0), n)
            win = g > bg
            best[win], bg[win] = vm, g[win]
        bad += not np.array_equal(raster_eid(flat, vmat, m, peid, best, n),
                                  raster_eid_brute(flat, vmat, m, peid, best, n))
    print(f"raster eid rule vs brute force: mismatches {bad}")
    N = 100 * 80 * 50
    k = 20000
    flat = rng.integers(0, N, k)
    vmat = rng.integers(1, 3, k).astype(np.int16)
    m = rng.random(k)
    peid = np.where(rng.random(k) < 0.7, 3, 0)
    best = np.ones(N, np.int16)
    t0 = time.perf_counter()
    for _ in range(10):
        raster_eid(flat, vmat, m, peid, best, N)
    print(f"raster eid at 20k particles, 400k-cell world: "
          f"{1e3 * (time.perf_counter() - t0) / 10:.2f} ms/tick")
    # 3. hygiene
    shape = (100, 80, 50)
    mat = np.zeros(shape, np.uint8); mat[:, :, 0] = 2; mat[10:90, 10:70, :45] = 2
    mat[11:89, 11:69, 1:44] = 0
    prev = mat.copy()
    edge = np.zeros(shape, np.float32); rub = np.zeros(shape, bool); eid = np.zeros(shape, np.int32)

    def hygiene():
        ch = np.flatnonzero(mat.ravel() != prev.ravel())
        if len(ch):
            gone = ch[mat.ravel()[ch] == AIR]
            edge.ravel()[gone] = 0; rub.ravel()[gone] = False; eid.ravel()[gone] = 0
        return len(ch)
    t0 = time.perf_counter()
    for _ in range(50):
        hygiene()
    q = (time.perf_counter() - t0) / 50
    mat.ravel()[rng.integers(0, N, 300)] = AIR
    t0 = time.perf_counter()
    for _ in range(50):
        hygiene()
    b = (time.perf_counter() - t0) / 50
    t0 = time.perf_counter()
    for _ in range(50):
        np.copyto(prev, mat)
    c = (time.perf_counter() - t0) / 50
    print(f"hygiene at 400k cells: quiet tick {1e6 * q:.0f} us, 300 changed {1e6 * b:.0f} us, "
          f"prev copy {1e6 * c:.0f} us")


if __name__ == "__main__":
    main()
