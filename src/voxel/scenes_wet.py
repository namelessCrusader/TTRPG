"""Proof-scene builders for the matter processes (T6, wave 13c lane C3). Each returns a
WetWorld with `enable_wet` already called. No scene-specific code: every result comes
from the surface law, the boil, the saturation adjustment and the ice law.

    kettle       S2: 5x5 cells of water in a stone pot, a 150 C heater block under it
    puddle       a thin film of water on a stone floor in air of a given humidity
    pond         S9: a stone-lined pond of water, every cell cooled to t_c
    wet_log      S11: a wood block with moisture in it, in air of a given humidity
    fog_room     a closed stone room, a hot pot of water in it (mist and dew)
"""
from .materials import AIR, AMBIENT, IRON, STONE, WATER, WOOD
from .wetworld import WetWorld


def _pot(w, x0=8, y0=8, n=5, h=4):
    w.fill(0, w.shape[0], 0, w.shape[1], 0, 1, STONE)
    w.fill(x0 - 1, x0 + n + 1, y0 - 1, y0 + n + 1, 1, 1 + h, STONE)
    w.fill(x0, x0 + n, y0, y0 + n, 1, 1 + h, AIR)
    return x0, y0, n


def kettle(rh=0.5, gas_flow=False, heater_c=150.0, fill=2):
    w = WetWorld(24, 24, 30, voxel_cm=5)
    w.gas_flow = gas_flow
    x0, y0, n = _pot(w)
    for x in range(x0, x0 + n):
        for y in range(y0, y0 + n):
            for z in range(1, 1 + fill):
                w.pour(x, y, z, WATER, w.cap)
    w.enable_wet(rh=rh)
    w.thermostats = [(x0 + n // 2, y0 + n // 2, 0, heater_c)]
    w.scene_info = {"pot": (x0, y0, n), "fill": fill}
    return w


def puddle(rh=0.5, ml=30.0, size=6):
    w = WetWorld(16, 16, 12, voxel_cm=5)
    w.fill(0, 16, 0, 16, 0, 1, STONE)
    for x in range(5, 5 + size):
        for y in range(5, 5 + size):
            w.pour(x, y, 1, WATER, ml)
    w.enable_wet(rh=rh)
    return w


def pond(t_c=-5.0, depth=2, size=6):
    w = WetWorld(12, 12, 12, voxel_cm=10)
    w.fill(0, 12, 0, 12, 0, 1, STONE)
    w.fill(2, 10, 2, 10, 1, 1 + depth + 2, STONE)
    w.fill(3, 9, 3, 9, 1, 1 + depth + 2, AIR)
    for x in range(3, 3 + size):
        for y in range(3, 3 + size):
            for z in range(1, 1 + depth):
                w.pour(x, y, z, WATER, w.cap)
    w.enable_wet(rh=0.5)
    C = w.heat_capacity()
    w.E[3:3 + size, 3:3 + size, 1:1 + depth] = (t_c - AMBIENT) * C[3:3 + size, 3:3 + size, 1:1 + depth]
    return w


def wet_log(rh=0.5, moisture=0.30):
    """A 4x4x4 wood block holding `moisture` g of water per g of dry wood."""
    w = WetWorld(14, 14, 14, voxel_cm=5)
    w.fill(0, 14, 0, 14, 0, 1, STONE)
    w.fill(5, 9, 5, 9, 1, 5, WOOD)
    sm = w.smass[5:9, 5:9, 1:5]
    w.fl[5:9, 5:9, 1:5] = WATER
    w.fvol[5:9, 5:9, 1:5] = sm * moisture
    w.enable_wet(rh=rh)
    return w


def fog_room(rh=0.5):
    """A closed stone room, a pot of hot water on the floor: steam fills it, the cold wall
    takes dew, the plume makes mist."""
    w = WetWorld(20, 20, 16, voxel_cm=5)
    w.fill(0, 20, 0, 20, 0, 1, STONE)
    w.fill(0, 20, 0, 20, 1, 14, STONE)
    w.fill(1, 19, 1, 19, 1, 13, AIR)
    w.fill(0, 20, 0, 20, 13, 14, STONE)
    for x in range(8, 12):
        for y in range(8, 12):
            w.pour(x, y, 1, WATER, w.cap * 0.6)
    w.enable_wet(rh=rh)
    C = w.heat_capacity()
    w.E[8:12, 8:12, 1] = (99.0 - AMBIENT) * C[8:12, 8:12, 1]
    return w
