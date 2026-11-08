"""Craft proofs from general laws (wave 13c, lane C12; T1 spec section 9, T9 spec 6).

There is NO craft code here and none anywhere in the engine: no function knows a bronze bar,
a brine cup or a file. Each builder lays out matter (a mix, a lot, a thing) and lets the laws
that already exist run. None is registered in SCENARIOS (no golden case moves).

    bar(rows)            P1: a 16-cell bar of any mix on stone, one end held at 400 C
    brine_cup(low)       P2: a 5 cm cup column: 20 percent brine in one cell, fresh water in the other
    solder_sweat()       P6: a voxel of 80/20 lead-tin held at 195 C, with open floor around it
    seed_lot(n)          P8: a thousand seeds with a spread of densities (a LOT, hand-made, seeded)
    brine_density(w, m)  the density (kg/m3) of a mix row, read from the world's own mix table
    file_removal(...)    the file test (hand-set temper, see the note on it)

Missing general laws are listed in tests/test_c12_crafts.py (xfail tests that say what is missing).
"""
import numpy as np

from . import lots
from . import matedge as ME
from . import matrec as MATREC
from .materials import AIR, STONE, WATER
from .scenes_mixtures import BRINE20
from .sim import World

COPPER = [("copper", "bound", 0, 1.0)]
BRONZE = [("copper", "bound", 0, 0.88), ("tin", "bound", 0, 0.12)]
HOT_C = 400.0


def bar(rows, n=16):
    """P1. A bar `n` cells long (1 cm cells, 2 x 2 cells across) of the mix `rows`, lying on a
    stone floor. The four cells at the left end are a dev thermostat at 400 C."""
    w = World(n + 6, 8, 6, voxel_cm=1)
    w.fill(0, n + 6, 0, 8, 0, 1, STONE)
    m = w.ensure_mixes().parse(rows)
    w.fill(3, 3 + n, 3, 5, 1, 3, None, mix=m)
    w.thermostats = [(3, y, z, HOT_C) for y in (3, 4) for z in (1, 2)]
    return w


def brine_cup(low=True, rows=BRINE20):
    """P2. A stone cup one cell wide. A full cell of the brine `rows` and a full cell of fresh
    water, brine UNDER the water when `low`, OVER it when not. Returns (world, mix id)."""
    w = World(5, 5, 8, voxel_cm=5)
    w.fill(0, 5, 0, 5, 0, 1, STONE)
    w.fill(0, 5, 0, 5, 1, 6, STONE)
    w.fill(2, 3, 2, 3, 1, 6, AIR)
    m = w.ensure_mixes().parse(rows)
    for z, brine in ((1, low), (2, not low)):
        if brine:
            w.pour(2, 2, z, None, w.cap, mix=m)
        else:
            w.pour(2, 2, z, WATER, w.cap)
    return w, m


SOLDER_80_20 = [("lead", "bound", 0, 0.8), ("tin", "bound", 0, 0.2)]
SWEAT_CELL = (3, 2, 1)


def solder_sweat(t_c=195.0):
    """P6. One 2 cm voxel of 80/20 lead-tin on a stone floor, held at `t_c` by a dev thermostat.
    The phase rule (mixphase lever rule) decides how much of it is liquid and what that liquid
    holds; nothing says 'solder'."""
    w = World(8, 6, 8, voxel_cm=2)
    w.fill(0, 8, 0, 6, 0, 1, STONE)
    w.fill(3, 4, 2, 3, 1, 2, None, mix=w.ensure_mixes().parse(SOLDER_80_20))
    w.thermostats = [(*SWEAT_CELL, float(t_c))]
    return w


def mix_density(w, m):
    """kg/m3 of a mix row, from the world's own table (the number the layering law uses)."""
    return float(w.mixes.prop("rho", np.array([m]))[0])


def seed_lot(n=1000, seed=7):
    """P8. `n` seeds of 3 mm: sound ones about 1250 kg/m3, hollow or eaten ones lighter, with
    a spread. A deterministic generator; a LOT, so the count is an integer and exact."""
    rng = np.random.default_rng(seed)
    sound = rng.random(n) < 0.8
    rho = np.where(sound, rng.normal(1250.0, 40.0, n), rng.normal(1000.0, 120.0, n))
    mass = rho * (np.pi / 6.0) * 3.0e-3 ** 3
    return lots.make_lot(mass, rho, np.full(n, 3.0e-3)), sound


# THE FILE TEST (the smithing slice). The piece's hardness is hand-set: `temper` multiplies H
# (T9 section 6 hook; `things[eid]["temper"]`). NO heat-history law exists, so nothing here
# turns a quench into a temper: a number is typed in. This proves only that, given a temper,
# the edge law's file removal follows hardness. The law that makes the temper is listed as
# missing (D1).
def file_removal(temper, file_temper=7.5, strokes=20, force_N=30.0, stroke_m=0.1):
    """Volume (m3) a file takes off a piece of iron of the given temper in `strokes` strokes.
    It is the Archard law the edge uses (matedge.hone, before its tip-radius clamp):
    V = K(H_file / H_piece) F s n / H_piece."""
    iron = MATREC.REC["iron"]
    H_piece = float(iron.at(20.0, 0.0, temper).H)
    H_file = float(iron.at(20.0, 0.0, file_temper).H)
    K = float(ME.k_archard(H_file / H_piece))
    return K * force_N * stroke_m * strokes / H_piece
