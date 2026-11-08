"""Proof-scene builders for the legs' step range (T11, wave 13c lane C9). Geometry and a
person; the planner, the legs and the books are the laws.

    stair_flight   P7: a flight (riser, tread, steps) up to a landing, a person at the foot
    kerb_yard      a floor with a kerb of a given height across it, a person before it
"""
from .materials import STONE
from .scenes import _person
from .sim import World


def stair_flight(steps=4, riser=3, tread=6, landing=14, width=14, person_x=6):
    """A flight of `steps` stone treads, each `riser` cells higher than the last and `tread`
    cells deep (the default at 5 cm: four 15 cm risers, 30 cm treads), ending in a
    landing `landing` cells deep. A person stands at the foot. Returns (world, person).

    The default flight is the one the legs climb standing up. A longer or steeper one
    (12 steps of 20 cm, the spec's P7) makes a rigid body that is one piece: it
    balance-steps back and forth on the treads and a body there can go over (lane C9
    report, open problem 1)."""
    nx = person_x + 8 + steps * tread + landing
    nz = 1 + steps * riser + 40
    w = World(nx, width, nz, voxel_cm=5)
    w.fill(0, nx, 0, width, 0, 1, STONE)
    x0 = person_x + 8
    for i in range(steps):
        w.fill(x0 + i * tread, nx, 0, width, 1, 1 + (i + 1) * riser, STONE)
    _person(w, person_x, width // 2)
    w.step()
    return w, w.persons[0]


def kerb_yard(height_cells=3, x_kerb=20, nx=48):
    """A floor with a kerb `height_cells` high across it from `x_kerb` on, a person at x = 8."""
    w = World(nx, 24, 40, voxel_cm=5)
    w.fill(0, nx, 0, 24, 0, 1, STONE)
    w.fill(x_kerb, nx, 0, 24, 1, 1 + height_cells, STONE)
    _person(w, 8, 12)
    w.step()
    return w, w.persons[0]
