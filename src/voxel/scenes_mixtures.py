"""Proof-scene builders for the mixtures switch (wave 13b, lane C2; T4 spec section 7).
None is registered in SCENARIOS: the golden cases stay as they were, and no golden world
holds a mix. Each returns a World; the lane's tests and the cost measurements build these.

    brine_pot(rows)     scenes 1 and 2: one walled cell of 50 ml (pure water when `rows` is
                        None, else the mix) on a stone floor, to be heated by the caller
    solder_cell(rows)   scene 7 (a piece): one voxel (2 cm) of a lead-tin mix on stone
    wet_pot_world(...)  the speed scene: a wide pond of pure water, and a walled pot of
                        brine in a corner. The compiled twin of the settling sweep does the
                        pond; the pot takes the Python path
"""
from .materials import AIR, STONE, WATER
from .sim import World

BRINE10 = [("water", "dissolved", 0, 0.9), ("sodium_chloride", "dissolved", 0, 0.1)]
BRINE20 = [("water", "dissolved", 0, 0.8), ("sodium_chloride", "dissolved", 0, 0.2)]
SOLDER_60_40 = [("lead", "bound", 0, 0.4), ("tin", "bound", 0, 0.6)]
POT_CELL = (1, 2, 1)


def brine_pot(rows=None, ml=50.0):
    w = World(8, 6, 6, voxel_cm=10)
    w.fill(0, 8, 0, 6, 0, 1, STONE)
    w.fill(0, 3, 1, 4, 1, 3, STONE)
    w.fill(1, 2, 2, 3, 1, 3, AIR)
    if rows is None:
        w.pour(*POT_CELL, WATER, ml)
    else:
        w.pour(*POT_CELL, None, ml, mix=w.ensure_mixes().parse(rows))
    return w


def solder_cell(rows=None):
    w = World(8, 6, 6, voxel_cm=2)
    w.fill(0, 8, 0, 6, 0, 1, STONE)
    w.fill(3, 4, 2, 3, 1, 2, None, mix=w.ensure_mixes().parse(rows or SOLDER_60_40))
    return w


def wet_pot_world(brine=True):
    w = World(60, 60, 20, voxel_cm=5)
    w.fill(0, 60, 0, 60, 0, 1, STONE)
    for x0, x1, y0, y1 in ((0, 1, 0, 60), (59, 60, 0, 60), (0, 60, 0, 1), (0, 60, 59, 60)):
        w.fill(x0, x1, y0, y1, 0, 6, STONE)
    w.fill(40, 47, 40, 47, 1, 5, STONE)
    w.fill(41, 46, 41, 46, 1, 5, AIR)
    for x in range(5, 38, 2):
        for y in range(5, 55, 2):
            w.pour(x, y, 1, WATER, 40.0)
    if brine:
        row = w.ensure_mixes().parse(BRINE10)
        for x in range(41, 46):
            for y in range(41, 46):
                w.pour(x, y, 1, None, 60.0, mix=row)
    return w
