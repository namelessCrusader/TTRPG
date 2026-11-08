"""Scratch prototype (design phase, nothing in src/ touched).

Question: can ONE numba union-find reproduce each of the three legacy
labellers in sim.py BIT FOR BIT -- same partition AND same label values --
including the reach labeller's np.roll wrap-around and min-vs-max canon?
And does a (mat, eid) KEY give the corrected join rule for free (eid>0 joins
on eid across materials, eid==0 joins on mat), identical to today when every
eid is 0?

Legacy routines are copied verbatim (modulo `self.`) from:
  free fall  sim.py:1521-1532  (any material, max propagation, no wrap)
  torque     sim.py:4057-4069  (same material, max propagation, no wrap)
  reach      sim.py:6054-6083  (same material, min propagation, np.roll WRAP,
                                pointer jumps)
"""
import time
import numpy as np
from numba import njit

AIR, FLESH = 0, 9


# ---------------------------------------------------------------- legacy
def legacy_freefall(falling):
    lab = np.where(falling, np.arange(falling.size, dtype=np.int32).reshape(falling.shape), -1)
    while True:
        new = lab.copy()
        for axis in range(3):
            a = [slice(None)] * 3; b = [slice(None)] * 3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            both = falling[a] & falling[b]
            new[a] = np.where(both, np.maximum(new[a], lab[b]), new[a])
            new[b] = np.where(both, np.maximum(new[b], lab[a]), new[b])
        if (new == lab).all():
            break
        lab = new
    return lab


def legacy_torque(mat):
    solid = mat != AIR
    lab = np.where(solid, np.arange(solid.size, dtype=np.int32).reshape(mat.shape), -1)
    while True:
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


def legacy_reach(box):
    solid = (box != AIR) & (box != FLESH)
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


# ---------------------------------------------------------------- one kernel
@njit(cache=False)
def _find(par, a):
    while par[a] != a:
        par[a] = par[par[a]]
        a = par[a]
    return a


@njit(cache=False)
def label_uf(mask, key, use_key, canon_max, wrap):
    """mask: bool 3-D. key: int64 3-D (joined only where equal) if use_key.
    Returns int64 labels: -1 outside mask; else the MAX (canon_max) or MIN
    flat index of the component -- the canonical value each legacy caller
    converges to. wrap: also join across opposite faces (np.roll semantics)."""
    nx, ny, nz = mask.shape
    N = nx * ny * nz
    par = np.arange(N)
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if not mask[x, y, z]:
                    continue
                i = (x * ny + y) * nz + z
                for d in range(3):
                    if d == 0:
                        if x > 0:
                            xx, yy, zz = x - 1, y, z
                        elif wrap and nx > 1:
                            xx, yy, zz = nx - 1, y, z
                        else:
                            continue
                    elif d == 1:
                        if y > 0:
                            xx, yy, zz = x, y - 1, z
                        elif wrap and ny > 1:
                            xx, yy, zz = x, ny - 1, z
                        else:
                            continue
                    else:
                        if z > 0:
                            xx, yy, zz = x, y, z - 1
                        elif wrap and nz > 1:
                            xx, yy, zz = x, y, nz - 1
                        else:
                            continue
                    if not mask[xx, yy, zz]:
                        continue
                    if use_key and key[xx, yy, zz] != key[x, y, z]:
                        continue
                    j = (xx * ny + yy) * nz + zz
                    a, b = _find(par, i), _find(par, j)
                    if a != b:
                        # root = the canonical member, so no second pass is
                        # needed to find each component's max / min index
                        if canon_max:
                            if a > b:
                                par[b] = a
                            else:
                                par[a] = b
                        else:
                            if a < b:
                                par[b] = a
                            else:
                                par[a] = b
    out = np.full(N, -1, np.int64)
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if mask[x, y, z]:
                    i = (x * ny + y) * nz + z
                    out[i] = _find(par, i)
    return out.reshape(mask.shape)


def key_of(mat, eid):
    """THE JOIN RULE as one key: eid>0 joins on eid (any material), eid==0
    joins on material. Disjoint value ranges (mat < 256 <= 256 + eid)."""
    return np.where(eid > 0, 256 + eid.astype(np.int64), mat.astype(np.int64))


def rand_world(rng, shape, nmat=4, blobs=40):
    mat = np.zeros(shape, np.uint8)
    for _ in range(blobs):
        lo = rng.integers(0, np.array(shape))
        sz = rng.integers(1, 6, 3)
        sl = tuple(slice(int(l), int(min(l + s, n))) for l, s, n in zip(lo, sz, shape))
        mat[sl] = rng.integers(1, nmat + 1)
    # some flesh and face-touching slabs on purpose (the wrap cases)
    mat[0, :, : shape[2] // 2] = 2
    mat[-1, :, : shape[2] // 2] = 2
    mat[:, 0, -2:] = 3
    mat[:, -1, -2:] = 3
    mat[rng.random(shape) < 0.03] = FLESH
    return mat


def main():
    rng = np.random.default_rng(0)
    nbad = {"freefall": 0, "torque": 0, "reach": 0, "reach_nowrap_differs": 0,
            "eid0_key_equals_mat": 0}
    trials = 60
    for t in range(trials):
        shape = tuple(int(v) for v in rng.integers(4, 18, 3))
        mat = rand_world(rng, shape)
        falling = (mat != AIR) & (rng.random(shape) < 0.8)
        L = label_uf(falling, mat.astype(np.int64), False, True, False)
        nbad["freefall"] += not np.array_equal(L, legacy_freefall(falling))
        L = label_uf(mat != AIR, mat.astype(np.int64), True, True, False)
        nbad["torque"] += not np.array_equal(L, legacy_torque(mat))
        solid = (mat != AIR) & (mat != FLESH)
        Lw = label_uf(solid, mat.astype(np.int64), True, False, True)
        nbad["reach"] += not np.array_equal(Lw, legacy_reach(mat))
        Ln = label_uf(solid, mat.astype(np.int64), True, False, False)
        nbad["reach_nowrap_differs"] += not np.array_equal(Ln, Lw)
        eid0 = np.zeros(shape, np.int32)
        nbad["eid0_key_equals_mat"] += not np.array_equal(
            label_uf(solid, key_of(mat, eid0), True, False, True), Lw)
    print(f"{trials} random worlds (4..17 per side, face-touching slabs, flesh):")
    for k, v in nbad.items():
        print(f"  {k:24s} mismatches: {v}")

    # the corrected join rule on a named axe: iron head + wood haft, eid 7,
    # touching an UNNAMED iron post (eid 0)
    IRON, WOOD = 3, 1
    mat = np.zeros((12, 6, 10), np.uint8)
    eid = np.zeros_like(mat, np.int32)
    mat[2:4, 2, 5] = IRON; eid[2:4, 2, 5] = 7          # head
    mat[4:9, 2, 5] = WOOD; eid[4:9, 2, 5] = 7          # haft
    mat[1, 2, 1:9] = IRON                              # unnamed post, touching head
    solid = (mat != AIR) & (mat != FLESH)
    old = label_uf(solid, mat.astype(np.int64), True, False, False)
    new = label_uf(solid, key_of(mat, eid), True, False, False)
    def comps(L, m):
        return sorted(len(np.flatnonzero(L == v)) for v in np.unique(L[m]))
    print("axe+post by material:", comps(old, solid), "(head joined to post, haft apart)")
    print("axe+post by key     :", comps(new, solid), "(axe 7 cells one thing, post 8)")

    # speed, house-sized, whole-world torque labelling
    shape = (100, 80, 50)
    mat = np.zeros(shape, np.uint8)
    mat[:, :, 0] = 2
    mat[10:90, 10:70, 0:45] = 2
    mat[11:89, 11:69, 1:44] = 0
    for _ in range(40):
        lo = rng.integers((12, 12, 1), (80, 60, 30))
        sz = rng.integers(1, 12, 3)
        mat[lo[0]:lo[0] + sz[0], lo[1]:lo[1] + sz[1], lo[2]:lo[2] + sz[2]] = rng.integers(1, 5)
    t0 = time.perf_counter(); a = legacy_torque(mat); t1 = time.perf_counter()
    k = mat.astype(np.int64)
    label_uf(mat != AIR, k, True, True, False)
    t2 = time.perf_counter()
    for _ in range(5):
        b = label_uf(mat != AIR, k, True, True, False)
    t3 = time.perf_counter()
    print(f"house {shape}: legacy torque labelling {1e3 * (t1 - t0):.0f} ms, "
          f"union-find {1e3 * (t3 - t2) / 5:.1f} ms, identical={np.array_equal(a, b)}")
    box = mat[30:60, 20:50, 0:30].copy()
    t0 = time.perf_counter()
    for _ in range(5):
        a = legacy_reach(box)
    t1 = time.perf_counter()
    kb = box.astype(np.int64)
    sb = (box != AIR) & (box != FLESH)
    for _ in range(5):
        b = label_uf(sb, kb, True, False, True)
    t2 = time.perf_counter()
    print(f"reach box {box.shape}: legacy {1e3 * (t1 - t0) / 5:.2f} ms, "
          f"union-find {1e3 * (t2 - t1) / 5:.2f} ms, identical={np.array_equal(a, b)}")


if __name__ == "__main__":
    main()
