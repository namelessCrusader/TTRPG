"""Wave 14a lane B: wood soak per species and per grain (wooddata, FluidsMixin.set_wood)."""
import numpy as np
import pytest

from src.voxel import wooddata as wd
from src.voxel.fluids import SORPTIVITY
from src.voxel.materials import AIR, STONE, TICK_S, WEAK_ACID, WATER, WOOD
from src.voxel.sim import World


def _tub(species=None, grain=None):
    """C-T3's tub: one-voxel wood walls and floor, a 3x3x3 cavity of vinegar."""
    w = World(9, 9, 7, voxel_cm=5)
    w.fill(0, 9, 0, 9, 0, 1, STONE)
    w.fill(2, 7, 2, 7, 1, 5, WOOD)
    w.mat[3:6, 3:6, 2:5] = AIR
    w.smass[3:6, 3:6, 2:5] = 0.0
    if species is not None:
        w.set_wood((slice(2, 7), slice(2, 7), slice(1, 5)), species, grain)
    for x in range(3, 6):
        for y in range(3, 6):
            for z in range(2, 5):
                w.pour(x, y, z, WEAK_ACID, w.cap)
    return w


def _drunk(w, ticks):
    ml0 = w.total_fluid(WEAK_ACID)
    for _ in range(ticks):
        w.step()
    inside = float(w.fvol[3:6, 3:6, 2:5][w.fl[3:6, 3:6, 2:5] == WEAK_ACID].sum())
    return ml0 - inside, ml0


def test_generic_row_is_untouched():
    assert SORPTIVITY[WOOD] == (1.0, 0.6)
    assert WOOD in SORPTIVITY


def test_naming_no_species_is_the_old_wood_bit_for_bit():
    a, b = _tub(), _tub()
    b.set_wood((slice(2, 7), slice(2, 7), slice(1, 5)), None)
    for _ in range(300):
        a.step()
        b.step()
    assert np.array_equal(a.fvol, b.fvol) and np.array_equal(a.smass, b.smass)


def test_max_moisture_reproduces_the_handbook_ends():
    assert wd.mmax(0.30) == pytest.approx(2.67, abs=0.01)      # 267 %
    assert wd.mmax(0.90) == pytest.approx(0.444, abs=0.005)    # 44 %


def test_every_species_is_faster_along_the_grain_and_sourced_where_it_claims():
    for name in wd.NAMES[1:]:
        al, ac, wm = wd.row(name)
        assert al > ac > 0 and wm > 0
    assert wd.row("pine")[0] == pytest.approx(13.0)             # the softwood 10-16 band
    assert 10 <= wd.row("pine")[0] <= 16 and 1 <= wd.row("pine")[1] <= 7


def test_C_T3_a_white_oak_cask_holds_its_vinegar():
    """The cask test on real data: tyloses make white oak tight (S across 0.3)."""
    lost, ml0 = _drunk(_tub("oak"), 2000)
    assert lost < 1.0, f"the oak cask drank {lost:.2f} ml of {ml0:.0f}"


def test_a_pine_tub_drinks_more_than_an_oak_cask_and_balsa_most():
    d = {s: _drunk(_tub(s), 600)[0] for s in ("oak", "pine", "balsa")}
    assert d["oak"] < d["pine"] < d["balsa"], d
    # side grain throughout: uptake goes as S across (water is the wetting liquid, wick 1)
    assert d["pine"] / d["oak"] == pytest.approx(wd.row("pine")[1] / wd.row("oak")[1], rel=0.35)


def _slab(species, grain):
    """A 4x4 slab under a pool of water; only its top face is wet."""
    w = World(8, 8, 6, voxel_cm=5)
    w.fill(0, 8, 0, 8, 0, 1, STONE)
    w.fill(1, 7, 1, 7, 1, 3, STONE)
    w.fill(2, 6, 2, 6, 1, 2, WOOD)
    w.mat[2:6, 2:6, 2] = AIR
    w.smass[2:6, 2:6, 2] = 0.0
    w.set_wood((slice(2, 6), slice(2, 6), slice(1, 2)), species, grain)
    for x in range(2, 6):
        for y in range(2, 6):
            w.pour(x, y, 2, WATER, w.cap)
    return w


def test_grain_direction_is_a_property_of_the_thing():
    """Water on the top face: with the fibres standing (z) it is an end face, lying (x) a side face."""
    up, side = _slab("pine", "z"), _slab("pine", "x")
    ml0 = up.total_fluid(WATER)
    for _ in range(400):
        up.step()
        side.step()
    u, s = ml0 - up.total_fluid(WATER) , ml0 - side.total_fluid(WATER)
    held_u = float(up.fvol[2:6, 2:6, 1][up.mat[2:6, 2:6, 1] == WOOD].sum())
    held_s = float(side.fvol[2:6, 2:6, 1][side.mat[2:6, 2:6, 1] == WOOD].sum())
    assert held_u > 2.0 * held_s > 0, (held_u, held_s)
    t = 400 * TICK_S
    assert held_s == pytest.approx(wd.row("pine")[1] * np.sqrt(t) * 16 * 0.0025, rel=0.4)


def test_soaking_a_species_keeps_the_fluid_ledger():
    w = _slab("beech", "z")
    ml0 = w.total_fluid(WATER)
    for _ in range(300):
        w.step()
    assert w.total_fluid(WATER) == pytest.approx(ml0, rel=1e-5)
