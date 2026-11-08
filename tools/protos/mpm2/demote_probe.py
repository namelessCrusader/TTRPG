"""Demotion done so it cannot overfill: the hinged pole (test_voxel.py:304)
at rest, its 62 particles assigned ONE CELL EACH (own cell first, else the
nearest free neighbour), dry-run through the support law on copies, written
to the lattice, then run 200 more ticks. Compared with 'just keep what
rasterize drew' (the overfull version)."""
import sys, time
sys.path.insert(0, "/home/nisargparikh/Desktop/Fun Stuff/Mark_1")
import numpy as np
from itertools import product
from src.voxel.sim import World, STONE, WOOD, AIR, _DENS_ARR
from src.mpm.matter import Matter

def pole():
    w = World(24, 8, 16)
    w.fill(0, 24, 0, 8, 0, 1, STONE)
    w.fill(4, 7, 3, 5, 1, 12, WOOD)
    w.mat[5:7, 3:5, 1] = AIR
    w.smass[5:7, 3:5, 1] = 0.0
    for _ in range(60):
        w.step()
    return w

def assign(w):
    m, mt = w.matter.mpm, w.matter
    dx = m.dx
    blocked = (w.mat != AIR) & ~mt.drawn               # lattice-solid, not ours
    own = np.floor(m.x / dx).astype(np.int64)
    centre = lambda c: (np.asarray(c) + 0.5) * dx
    order = np.argsort(np.linalg.norm(m.x - centre(own), axis=1), kind="stable")
    taken, out = set(), np.zeros_like(own)
    nbr = [np.array(o) for o in product((-1, 0, 1), repeat=3)]
    for p in order:
        cands = [own[p] + o for o in nbr]
        cands.sort(key=lambda c: (float(np.linalg.norm(m.x[p] - centre(c))), tuple(c)))
        for c in cands:
            t = tuple(int(v) for v in c)
            if all(0 <= t[a] < w.shape[a] for a in range(3)) and not blocked[t] and t not in taken:
                taken.add(t); out[p] = c; break
        else:
            return None
    return out

def dry_run(w, cells, rubble_mask):
    """support law on COPIES: would any of these cells be unsupported?"""
    m = w.matter.mpm
    mat = w.mat.copy(); smass = w.smass.copy(); rub = w.rubble.copy()
    d = w.matter.drawn
    mat[d] = AIR; smass[d] = 0.0; rub[d] = False       # what unrasterize leaves
    idx = tuple(cells.T)
    mat[idx] = m.vmat; smass[idx] = (m.m * 1000.0).astype(np.float32); rub[idx] = rubble_mask
    keep = (w.mat, w.smass, w.rubble)
    w.mat, w.smass, w.rubble = mat, smass, rub
    try:
        t0 = time.perf_counter()
        slack = w._relax_slack(mat != AIR)
        ms = (time.perf_counter() - t0) * 1e3
    finally:
        w.mat, w.smass, w.rubble = keep
    return int((slack[idx] < 0).sum()), ms

w = pole()
m = w.matter.mpm
cells = assign(w)
fill = (m.m * 1000.0) / (_DENS_ARR[m.vmat] * w.vox_l)
print(f"at rest: {len(m.x)} particles; one-cell-each assignment: "
      f"{'FAILED' if cells is None else f'{len(np.unique(cells, axis=0))} distinct cells'}; "
      f"particles moved off their own cell {int((cells != np.floor(m.x/m.dx)).any(axis=1).sum())}; "
      f"max fill {fill.max():.2f}")
for label, rmask in (("rubble = dmg", m.dmg), ("rubble = dmg & loose", m.dmg & (m.cid < 0))):
    bad, ms = dry_run(w, cells, rmask)
    print(f"dry run ({label}): unsupported cells {bad}, _relax_slack {ms:.1f} ms on {w.mat.size} cells")

# demote for real: unrasterize, write the cells, drop the particles
grams0 = w.total_mass(WOOD)
w.matter.unrasterize()
idx = tuple(cells.T)
w.mat[idx] = m.vmat; w.smass[idx] = (m.m * 1000.0).astype(np.float32)
w.E[idx] = m.E.astype(np.float32); w.edge[idx] = m.edge.astype(np.float32)
w.rubble[idx] = m.dmg & (m.cid < 0)
w._rub_gen += 1
w.matter = Matter(w)
print(f"wood ledger across demotion: {grams0:.3f} -> {w.total_mass(WOOD):.3f} g")
t0 = time.perf_counter()
for _ in range(200):
    w.step()
ms = (time.perf_counter() - t0) / 200 * 1e3
zs = np.argwhere(w.mat == WOOD)[:, 2]; xs = np.argwhere(w.mat == WOOD)[:, 0]
print(f"200 ticks after: re-promoted {w.matter.promoted} clusters, particles {len(w.matter)}, "
      f"step {ms:.1f} ms/tick (was ~127 ms with the pole as particles), "
      f"test-304 criteria: top z {zs.max()} (<=5), max x {xs.max()} (>=9), wood {w.total_mass(WOOD):.3f} g")
