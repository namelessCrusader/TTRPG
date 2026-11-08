"""Proof-scene builders for contact events and the slip law (T3 P3, P8, P9; lane C13).

    walkway   a stone floor, a person at the west end, and a way to lay layers on
              the floor faces by column ranges (P3 Holmes, P9 grease)
"""
from .layer_data import AMBIENT, CHALK, CLAY, DISSOLVED, GRAIN, LAMP_OIL, WATER
from .layers import cell_key
from .materials import STONE
from .scenes import _person
from .sim import World


def walkway(nx=60, ny=14, nz=40, person_x=6):
    """A 5 cm stone floor (top at z = 1) and one person at x = person_x. Returns (w, p)."""
    w = World(nx, ny, nz, voxel_cm=5)
    w.fill(0, nx, 0, ny, 0, 1, STONE)
    _person(w, person_x, ny // 2)
    w.step()
    w.ensure_layers()
    return w, w.persons[0]


def lay(w, x0, x1, comp, y0=0, y1=None, cover=1.0):
    """Put `comp` ({(species, form, size): grams}) on every floor face in columns x0..x1."""
    y1 = w.shape[1] if y1 is None else y1
    for x in range(x0, x1):
        for y in range(y0, y1):
            w.layers.add(cell_key(x, y, 0, 4, w.shape), dict(comp), cover=cover, Ts=AMBIENT)


def mud(g=1.0):
    return {(CLAY, GRAIN, 8): g, (WATER, DISSOLVED, 0): 0.6 * g}


def chalk(g=0.3):
    return {(CHALK, GRAIN, 8): g}


def oil(g=0.5e-3 * 0.05 * 0.05 * 1e6 * 0.8):
    return {(LAMP_OIL, DISSOLVED, 0): g}
