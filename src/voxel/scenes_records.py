"""Proof-scene builders for the records (T2, wave 13c lane C5). No scene code: a thing, a
record the scene DECLARES on it, and the laws.

    quench_blade   P1: a hot iron blade of a given thickness, in water or in air
    glued_pair     P4 (part): two wooden parts joined by a glue line that must cure
"""
import numpy as np

from .materials import AIR, AMBIENT, IRON, STONE, WATER, WOOD
from .scenes import thing
from .sim import World


def quench_blade(thick_mm=2.0, T0=850.0, water=True, length_cells=8, record=True):
    """A blade `thick_mm` thick, 4 cm wide and `length_cells` x 5 cm long, at T0 C, lying
    on a stone slab. With `water` the slab holds a pool 3 cells deep around it. The mass
    is that of the real blade (iron, 7800 kg/m3): a cell holds a fraction of a cell of
    iron, the way the sim stores a thin plate. `record` declares a SECTION of the real
    thickness on it."""
    n = length_cells
    w = World(n + 4, 9, 8, voxel_cm=5)
    w.fill(0, n + 4, 0, 9, 0, 1, STONE)
    if water:                                      # a stone tank round the blade, water in it
        w.fill(0, n + 4, 0, 1, 1, 5, STONE)
        w.fill(0, n + 4, 8, 9, 1, 5, STONE)
        w.fill(0, 1, 0, 9, 1, 5, STONE)
        w.fill(n + 3, n + 4, 0, 9, 1, 5, STONE)
    with thing(w, "blade", kind="tool", parts={"blade": IRON}):
        w.fill(2, 2 + n, 4, 5, 1, 2, IRON)
    if water:
        for x in range(1, n + 3):
            for y in range(1, 8):
                for z in range(1, 5):
                    if (x, y, z) != (0, 0, 0) and w.mat[x, y, z] == AIR:
                        w.pour(x, y, z, WATER, w.cap)
    cell_g = 7.8 * (thick_mm / 10.0) * 4.0 * 5.0              # g: rho x thick x width x 5 cm
    c = (slice(2, 2 + n), slice(4, 5), slice(1, 2))
    w.smass[c] = cell_g
    w.E[c] = (T0 - AMBIENT) * w.heat_capacity()[c]
    rid = None
    if record:
        eid = int(w.eid[2, 4, 1])
        rid = w.declare_section(eid, thick_mm / 1000.0, width=0.04, length=0.05 * n)
    return w, rid


def glued_pair(kind="hide_glue", grams=2.0, area=0.002):
    """Two wooden blocks, one on the other, joined by a glue line the act brings."""
    w = World(10, 8, 8, voxel_cm=5)
    w.fill(0, 10, 0, 8, 0, 1, STONE)
    with thing(w, "head", kind="tool", parts={"head": WOOD}):
        w.fill(2, 4, 3, 5, 1, 3, WOOD)
    with thing(w, "haft", kind="tool", parts={"haft": WOOD}):
        w.fill(4, 8, 3, 4, 2, 3, WOOD)
    a, b = int(w.eid[2, 3, 1]), int(w.eid[5, 3, 2])
    w.join_things(a, b, kind, area, mass_g=grams, name="hafted head")
    return w, a, b
