"""The box room at 5 cm voxels — the first scene meant to be WATCHED.

4 m x 4 m x 3 m: stone floor and walls. A wooden table. A stick on the table with
oil pooled around its east end. A person standing clear. A torch lights the stick;
fire creeps; the oil catches; the tabletop burns; then water hits the WEST half
(wood fire: quenched) while the east half (oil fire) burns on.

Run:  python -m src.voxel.room <frames_dir>     writes one .npz per 10 ticks.
"""
import os
import sys

import numpy as np

from .sim import AIR, FLESH, OIL, STONE, WATER, WOOD, World

NX, NY, NZ = 80, 80, 60


def build():
    w = World(NX, NY, NZ, voxel_cm=5)
    w.fill(0, NX, 0, NY, 0, 1, STONE)                    # floor
    for (x0, x1, y0, y1) in ((0, 1, 0, NY), (NX - 1, NX, 0, NY),
                             (0, NX, 0, 1), (0, NX, NY - 1, NY)):
        w.fill(x0, x1, y0, y1, 0, NZ, STONE)             # walls
    w.fill(28, 53, 28, 53, 15, 16, WOOD, frac=0.8)       # tabletop, 1.25 m square
    for (lx, ly) in ((29, 29), (29, 50), (50, 29), (50, 50)):
        w.fill(lx, lx + 2, ly, ly + 2, 1, 15, WOOD)      # legs
    w.fill(32, 49, 40, 41, 16, 17, WOOD, frac=0.5)       # the stick, lying on top
    w.fill(8, 13, 58, 63, 1, 30, FLESH, frac=0.9)        # a person, out of the shot
    w.fill(8, 13, 58, 63, 30, 35, FLESH, frac=0.7)       # (head)
    return w


def oil_spout(w, t):
    if t < 30:                                           # a jug POURED, not teleported:
        w.pour(46, 41, 18, OIL, 340.0)                   # it falls, pools, finds its shape


def torch(w, t):
    if 60 <= t < 100:                                    # lit after the oil settles
        w.E[32, 40, 16] += 2500.0                        # flame held to the stick's west end


def water(w):
    for x in range(26, 56):                              # a WIDE dump from near the
        for y in range(26, 56):                          # ceiling: it rains down over
            w.pour(x, y, 55, WATER, w.cap)               # table, oil, floor


def run(frames_dir, ticks=900, every=10, douse_at=400):
    os.makedirs(frames_dir, exist_ok=True)
    w = build()
    for t in range(ticks):
        oil_spout(w, t)
        torch(w, t)
        if t == douse_at:
            water(w)
        w.step()
        if t % every == 0 or t == ticks - 1:
            np.savez_compressed(
                os.path.join(frames_dir, f"f{t:04d}.npz"),
                mat=w.mat, fl=w.fl, fvol=w.fvol.astype(np.float16),
                T=w.T().astype(np.float16), burn=w.burning(),
                smoke=w.smoke.astype(np.float16))
            if t % 100 == 0:
                print(f"t{t}: burning {int(w.burning().sum())}, "
                      f"wood {w.total_wood():.0f} g, oil {w.total_fluid(OIL):.0f} ml, "
                      f"peak {w.T().max():.0f} °C", flush=True)
    print("frames written to", frames_dir)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "frames")
