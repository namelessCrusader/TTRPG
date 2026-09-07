"""The first look at the voxel core: a wooden stick, an oil puddle, a torch.

Three scenes, printed as side-view slices so a person can watch with their eyes:
  A  torch lights one end of a stick; fire creeps along it and reaches the oil
  B  water dumped on a burning STICK  -> quenched (wood needs 300 °C)
  C  water dumped on a burning OIL pool -> keeps burning (oil needs only 250 °C,
     and the water boils away before it can cool that far)
Nothing in the code knows these outcomes. They fall out of the four laws.

Run:  python -m src.voxel.demo [A|B|C]     (no arg = all three)
"""
import sys

import numpy as np

from .sim import AIR, OIL, STONE, WATER, WOOD, World

GLYPH_HOT = 300.0


def render_side(w: World, y: int) -> str:
    """One x-z slice, z up. #=stone W=wood w=burning-wood o=oil f=burning-oil
    ~=water *=hot air .=air"""
    T, burn = w.T(), w.burning()
    rows = []
    for z in range(w.shape[2] - 1, -1, -1):
        row = []
        for x in range(w.shape[0]):
            p = (x, y, z)
            if w.mat[p] == STONE:
                row.append("#")
            elif w.mat[p] == WOOD:
                row.append("w" if burn[p] else "W")
            elif w.fl[p] == OIL and w.fvol[p] > 5:
                row.append("f" if burn[p] else "o")
            elif w.fl[p] == WATER and w.fvol[p] > 5:
                row.append("~")
            elif T[p] > GLYPH_HOT:
                row.append("*")
            else:
                row.append(".")
        rows.append("".join(row))
    return "\n".join(rows)


def build(oil_ml=800.0):
    """Stone floor, a 14-voxel wooden stick lying on it, oil pooled at its east end."""
    w = World(36, 7, 10)
    w.fill(0, 36, 0, 7, 0, 1, STONE)
    w.fill(8, 22, 3, 4, 1, 2, WOOD, frac=0.25)          # the stick: thin wood, mostly air
    for x in range(20, 27):
        for y in range(2, 5):
            if w.mat[x, y, 1] == AIR:
                w.pour(x, y, 1, OIL, oil_ml)            # the puddle laps the stick's end
    return w


def torch(w, x, y, z, joules=9000.0):
    """A held flame is nothing but energy arriving at a spot."""
    w.E[x, y, z] += joules


def dump_water(w, x0, x1, ml=None):
    for x in range(x0, x1):
        for y in range(2, 5):
            w.pour(x, y, 4, WATER, ml or w.cap)         # a barrel emptied from above


def run(scene="A", ticks=440, show_every=60):
    w = build()
    print(f"\n=== scene {scene} ===  (side view at y=3; z up. "
          f"#stone Wwood wBURNING oOIL fOIL-FIRE ~water *hot-air)")
    for t in range(ticks):
        if t < 25:
            torch(w, 8, 3, 1)                           # the flame licks the stick itself
        if scene == "B" and t == 200:
            print(f"\n-- t{t}: WATER dumped on the burning stick --")
            dump_water(w, 8, 20)
        if scene == "C" and t == 400:
            print(f"\n-- t{t}: WATER dumped on the burning oil pool --")
            dump_water(w, 20, 27)
        w.step()
        if t % show_every == 0 or t == ticks - 1:
            nb = int(w.burning().sum())
            print(f"\n-- t{t}  burning voxels: {nb}  wood left: {w.total_wood():.0f} g"
                  f"  oil left: {w.total_fluid(OIL):.0f} ml"
                  f"  water left: {w.total_fluid(WATER):.0f} ml  peak T: {w.T().max():.0f} °C")
            print(render_side(w, 3))
    return w


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "ABC"
    for s in which:
        run(s, ticks=500 if s == "C" else 440)
