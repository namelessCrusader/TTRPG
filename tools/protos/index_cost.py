"""cells_of index rebuild cost at house size (synthetic)."""
import time, numpy as np
rng = np.random.default_rng(0)
eid = np.zeros((100, 80, 50), np.int32)
for k in range(1, 41):
    x, y = rng.integers(10, 80), rng.integers(10, 60)
    eid[x:x+3, y:y+3, 1:6] = k
for k in range(41, 45):                       # four bodies, 396 cells each
    x, y = rng.integers(10, 80), rng.integers(10, 60)
    eid[x:x+3, y:y+5, 1:27] = k
def rebuild():
    nz = np.flatnonzero(eid)
    ids = eid.ravel()[nz]
    o = np.argsort(ids, kind="stable")
    ids, nz = ids[o], nz[o]
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    return ids[starts], starts, nz
rebuild()
t0 = time.perf_counter()
for _ in range(100): r = rebuild()
print(f"index rebuild at 400k cells, {int((eid>0).sum())} named cells: {1e6*(time.perf_counter()-t0)/100:.0f} us")
