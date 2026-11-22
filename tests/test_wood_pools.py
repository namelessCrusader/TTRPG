"""Wave 14a refinement R3: wood water in two pools (woodwater.py, FluidsMixin.set_wood(pools=True)).
Free water is fvol, bound water is `wood_bound`, vapour is the wet store; every conversion is booked."""
import numpy as np
import pytest

from src.voxel import psychro, woodwater as ww
from src.voxel import wooddata as wd
from src.voxel.materials import AIR, CHAR, NOFLUID, WATER, WOOD
from src.voxel.sim import World

DAY = 86400.0


class W(World):          # R4: World already carries WetMixin (humid by default)
    pass


def _block(rh=0.5, grain="z", species="pine", n=3, free=0.0, moisture=0.12):
    w = W(9, 9, 9, voxel_cm=5)
    w.enable_wet(rh=rh)
    s = slice(3, 3 + n)
    w.fill(3, 3 + n, 3, 3 + n, 3, 3 + n, WOOD)
    w.set_wood((s, s, s), species, grain, pools=True, moisture=moisture)
    if free:
        w.fl[s, s, s] = WATER
        w.fvol[s, s, s] = free * w.smass[s, s, s]
    return w


def _vent(w):
    """Open air: whatever vapour the wood gave out is carried away (booked as gone)."""
    ex = w.wet.store.excess("H2O")
    w.gone["vapour"] = w.gone.get("vapour", 0.0) + float(ex.sum())
    ex[:] = 0.0


def _water(w):
    return w.total_water() + w.bound_water_g()


def _days(w, d, dt=600.0, vent=True):
    """vent=True is open air AND a warm room: the heat law is not run here, so the cell
    temperatures are held at their start values (else drying would cool the wood for good)."""
    E0 = w.E.copy() if vent else None
    for _ in range(int(d * DAY / dt)):
        w._law_wood_pools(dt)
        if vent:
            _vent(w)
            w.E[...] = E0


def test_gate_off_is_the_old_wood():
    w = World(9, 9, 9, voxel_cm=5)
    w.fill(3, 6, 3, 6, 3, 6, WOOD)
    w.set_wood((slice(3, 6),) * 3, "pine", "z")
    assert w.wood_bound is None and w.bound_water_g() == 0.0 and w.bound_enthalpy_J() == 0.0
    assert w.skin_water().sum() == 0


def test_every_conversion_is_booked_and_water_is_conserved():
    w = _block(free=0.5)
    t0 = _water(w)
    _days(w, 20)
    assert _water(w) == pytest.approx(t0, rel=1e-5)
    assert w.wood_book["free_to_bound"] > 50.0


def test_bound_water_stops_at_the_fibre_saturation_point():
    w = _block(free=2.0)
    _days(w, 120)
    u = w.moisture()[3:6, 3:6, 3:6]
    assert u.max() <= ww.FSP + 1e-4 and u.max() > 0.95 * ww.FSP
    assert w.fvol[3:6, 3:6, 3:6].sum() > 0           # the cavities still hold liquid


def test_bound_exchange_takes_weeks_and_is_faster_along_the_grain():
    # a one-cell board: z is the only axis with open faces both ways in a 1-cell slab
    def t63(grain):
        w = _block(free=2.0, grain=grain, n=1)
        u0 = w.moisture()[3, 3, 3]
        for h in range(24 * 200 * 6):
            w._law_wood_pools(600.0)
            if w.moisture()[3, 3, 3] >= u0 + 0.63 * (ww.FSP - u0):
                return h / 144.0
        return 1e9
    a, c = t63("z"), t63(None)
    assert 3.0 < c < 120.0                           # weeks, not seconds
    assert c / a == pytest.approx(ww.GRAIN_RATIO, rel=0.15)


def test_air_dry_wood_in_dry_air_sheds_bound_water_slowly():
    w = _block(rh=0.2, n=1, moisture=0.12)
    u_eq = float(psychro.gab_w(0.2))
    _days(w, 1)
    assert w.moisture()[3, 3, 3] > 0.12 - 0.15 * (0.12 - u_eq)    # one day: under 15 percent of the way (a 5 cm board)
    _days(w, 80)
    u = w.moisture()[3, 3, 3]
    assert u_eq < u < 0.12 - 0.5 * (0.12 - u_eq)     # weeks later: well on its way, not there yet
    assert w.wood_book["bound_to_vapour"] > 0


def test_wood_takes_vapour_back_slowly():
    w = _block(rh=0.9, n=1, moisture=0.06)
    _days(w, 30)
    u = w.moisture()[3, 3, 3]
    assert 0.06 < u < float(psychro.gab_w(0.9))
    assert w.wood_book["vapour_to_bound"] > 0


def test_step_size_does_not_change_the_curve():
    r = []
    for dt in (60.0, 600.0):
        w = _block(rh=0.2, n=1)
        _days(w, 30, dt)
        r.append(float(w.moisture()[3, 3, 3]))
    assert max(r) - min(r) < 0.0015                  # a ten-times-bigger step keeps the curve (the gas pot, not dt, limits a step: dt over about 1000 s needs venting between steps)


def test_heat_of_sorption_is_booked():
    w = _block(free=0.5)
    E0 = w.total_energy() + w.bound_enthalpy_J()
    _days(w, 20, vent=False)
    assert w.total_energy() + w.bound_enthalpy_J() == pytest.approx(E0, rel=1e-4, abs=2.0)
    # wetting warmed the wood: the sorption heat is real (about 10 percent of the latent heat)
    assert w.bound_enthalpy_J() < -1000.0


def test_hot_wood_dries():
    w = _block(n=1)
    w.E[3, 3, 3] += 400.0 * float(w.heat_capacity()[3, 3, 3])
    _days(w, 0.05, dt=10.0)
    assert w.moisture()[3, 3, 3] < 0.01


def test_a_burnt_cell_gives_its_bound_water_to_the_gas():
    w = _block(n=1)
    t0 = _water(w)
    w.mat[3, 3, 3] = CHAR
    w._law_wood_pools(1.0)
    assert w.bound_water_g() == 0.0
    assert _water(w) == pytest.approx(t0, rel=1e-6)
    assert w.wood_book["released"] > 0


def test_the_walls_take_room_from_the_cavity():
    from src.voxel.fluids import _WOOD
    w = _block(n=1, moisture=ww.FSP)
    room_free = float(_WOOD[2][wd.species_id("pine")] * w.smass[3, 3, 3] - w.wood_bound[3, 3, 3])
    assert room_free < _WOOD[2][wd.species_id("pine")] * w.smass[3, 3, 3]


def test_skin_water_counts_bound_water():
    w = _block(n=3, moisture=0.2)
    assert w.skin_water()[3, 3, 3] > 0


def test_pine_means_sapwood_and_heartwood_is_slower():
    assert wd.row("pine_heart")[0] < wd.row("pine")[0] and wd.row("pine_heart")[1] < wd.row("pine")[1]
