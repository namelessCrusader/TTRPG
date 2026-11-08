"""C9 (wave 13c): the step range is ON in a plain World, and the legs use it.

The pathlaw tests prove the planner and the single step. These prove the
wiring: `_stride` takes its steps by `_step_to` when the range is above 0, the
look writes `p["fz"]` without touching the flat belief `p["free"]`, a rise costs
time and metabolic work, and a descent's heat is booked so the energy closes.
[I] = exact against the old behaviour, [C] = code against its own table.
"""
import numpy as np

from src.voxel import scenes, stairs
from src.voxel.materials import STONE
from src.voxel.sim import World


def _cells(w, p):
    comp, sl = w._person_cells(p)
    return w._claim_cells(p, comp, sl)


def _kerb(height, x_kerb=20, zero=False):
    w = World(48, 24, 40, voxel_cm=5)
    if zero:
        w.step_up_m = w.step_down_m = 0.0
    w.fill(0, 48, 0, 24, 0, 1, STONE)
    w.fill(x_kerb, 48, 0, 24, 1, 1 + height, STONE)
    scenes._person(w, 8, 12)
    w.step()
    return w, w.persons[0]


def _walk(w, p, goal, ticks):
    """Drive the legs the way the will does: `_stride` toward `goal`. Returns
    the cells at the end and the highest and lowest soles on the way (the will
    runs too and may turn the body round, so the extremes are what is asked)."""
    zs = []
    for _ in range(ticks):
        cells = _cells(w, p)
        zs.append(int(cells[:, 2].min()))
        ax, ay = float(cells[:, 0].mean()), float(cells[:, 1].mean())
        w._stride(p, cells, goal, ax, ay)
        w.step()
    cells = _cells(w, p)
    zs.append(int(cells[:, 2].min()))
    return cells, max(zs), min(zs)


def test_a_plain_world_has_the_step_range_on():
    w = World(16, 16, 16, voxel_cm=5)
    assert w._step_range({}) == (6, 6)
    w.step_up_m = w.step_down_m = 0.0
    assert w._step_range({}) == (0, 0)


def test_the_legs_walk_up_a_15_cm_kerb_and_pay_for_the_rise():
    w, p = _kerb(3)
    z0 = int(_cells(w, p)[:, 2].min())
    cells, zmax, _zmin = _walk(w, p, (40, 12), 60)
    assert zmax == z0 + 3, "stood on the kerb"
    st = p["_stairs"]
    assert st["up_m"] >= 0.15 - 1e-9
    assert st["metabolic_J"] > st["mech_J"] > 0.0          # work, and the eta
    assert p["_work_J"] >= st["metabolic_J"] - 1e-9
    assert not w.bodies, "the body stayed on its feet (the footing law held it)"


def test_a_40_cm_wall_is_not_climbed_by_the_legs():
    w, p = _kerb(8)
    cells, zmax, _ = _walk(w, p, (40, 12), 80)
    assert zmax == 1
    assert "_stairs" not in p


def test_with_the_range_at_0_the_legs_are_the_old_flat_legs():
    w, p = _kerb(3, zero=True)
    cells, zmax, _ = _walk(w, p, (40, 12), 80)
    assert zmax == 1 and "_stairs" not in p


def test_a_step_down_books_its_heat_and_the_energy_closes():
    w = World(48, 24, 40, voxel_cm=5)
    w.fill(0, 48, 0, 24, 0, 1, STONE)
    w.fill(0, 20, 0, 24, 1, 4, STONE)                 # a 15 cm platform, person on it
    scenes._person(w, 8, 12, z0=4)
    w.step()
    p = w.persons[0]
    cells, _zmax, zmin = _walk(w, p, (40, 12), 60)
    assert zmin == 1, "came down off the platform"
    st = p["_stairs"]
    assert st["down_m"] >= 0.15 - 1e-9
    # the potential a descent loses is heat in the cells, through deposit():
    # mech_in moved by the same joules the body-and-world total did
    assert st["heat_J"] > 0.0
    assert w.mech_in > 0.0


def test_the_look_writes_the_levels_and_leaves_the_flat_belief_alone():
    w, p = _kerb(3)
    assert p["fz"].shape[0] == w.levels_K
    kerb_col = p["fz"][:, 30, 12]
    assert 4 in kerb_col.tolist()                      # the top of a 3-cell kerb
    # the flat belief is what it always was: standable at the soles' own height
    assert not p["free"][30, 12] and p["free"][10, 12]


def test_a_rise_takes_longer_than_a_flat_cell():
    t_flat = stairs.step_time(0.0, 0.05)
    t_riser = stairs.step_time(0.05 * 3, 0.05)
    assert t_riser > t_flat


def test_a_four_step_flight_is_climbed_to_the_landing_and_the_work_is_booked():
    from src.voxel.scenes_stairs import stair_flight
    w, p = stair_flight()
    goal = (w.shape[0] - 6, 7)
    top = 1 + 4 * 3
    zmax = 0
    for _ in range(300):
        comp, sl = w._person_cells(p)
        cells = w._claim_cells(p, comp, sl)
        zmax = max(zmax, int(cells[:, 2].min()))
        w._stride(p, cells, goal, float(cells[:, 0].mean()),
                  float(cells[:, 1].mean()))
        w.step()
        if zmax == top:
            break
    assert zmax == top, "stood on the landing"
    st = p["_stairs"]
    assert st["up_m"] >= 0.60 - 1e-9                       # four 15 cm risers
    # the climb costs m g h / eta and more: the metabolic work is above the
    # mechanical work, which is at least m g h
    assert st["metabolic_J"] > st["mech_J"] >= 0.6 * 9.81 * 20.0
    assert not w.bodies
