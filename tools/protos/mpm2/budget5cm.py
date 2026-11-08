"""Baseline at voxel_cm=5 THROUGH World.step: a person standing beside a
wooden plank (2x2x14 cells = 56 particles) that is promoted and falls over.
How many ms of each tick are the solver's?"""
import sys, time
sys.path.insert(0, "/home/nisargparikh/Desktop/Fun Stuff/Mark_1")
import numpy as np
from src.voxel.sim import World, STONE, WOOD
from src.voxel.scenes import _person

w = World(40, 20, 30, voxel_cm=5)
w.fill(0, 40, 0, 20, 0, 1, STONE)
_person(w, 8, 10, z0=1)
w.fill(20, 22, 9, 11, 1, 15, WOOD)
for _ in range(3):
    w.step()
n = w.matter.promote(np.argwhere(w.mat == WOOD), vel=(4.0, 0.0, 0.0))  # a shove, cells/s
tm = []
orig = w.matter.step
def timed(dt):
    t0 = time.perf_counter(); r = orig(dt); tm.append((time.perf_counter() - t0) * 1e3); return r
w.matter.step = timed
rows = []
for t in range(30):
    t0 = time.perf_counter(); w.step(); st = (time.perf_counter() - t0) * 1e3
    rows.append((t, st, tm[-1]))
m = w.matter.mpm
sub = int(np.ceil(0.025 / m.cfl_dt()))
print(f"{n} particles, {sub} substeps/tick at 5 cm, grid {w.shape} ({w.mat.size} cells)")
for t, st, mt in rows[::5] + [rows[-1]]:
    print(f"t{t:2d}: step {st:7.1f} ms, matter.step {mt:7.1f} ms ({100*mt/st:4.1f}%)")
print(f"mean over 30 ticks: step {np.mean([r[1] for r in rows]):.1f} ms, matter {np.mean([r[2] for r in rows]):.1f} ms; "
      f"mean|v| at end {float(np.linalg.norm(m.v,axis=1).mean()):.3f} m/s")
