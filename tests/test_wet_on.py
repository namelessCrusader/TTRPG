"""Matter processes on (wave 13c, lane C3, T6): the gate, steam kept, the surface law,
mist, ice, moisture in wood, the flow hooks, and one world with every capability."""
import numpy as np
import pytest

from src.voxel import icephase as IP
from src.voxel import psychro
from src.voxel.coldworld import ColdWorld
from src.voxel.materials import AIR, AMBIENT, ICE, IRON, STONE, WATER, WOOD
from src.voxel.matterdata import CI, CV, CW, L0, L_CONV20
from src.voxel.scenes_wet import fog_room, kettle, pond, puddle, wet_log
from src.voxel.wetworld import EVERY, WetMixin, WetWorld


def _closes(w, e0, m0, rel=2e-6):
    scale = float(np.abs(w.E).sum()) + abs(e0) + 1.0
    assert abs(w.total_energy() - e0) < rel * scale, (w.total_energy() - e0, scale)
    assert abs(w.total_water() - m0) < rel * max(m0, 1.0) + 1e-3, (w.total_water() - m0, m0)


def _hot_cup(cls):
    w = cls(14, 14, 14, voxel_cm=5)
    w.fill(0, 14, 0, 14, 0, 1, STONE)
    w.fill(4, 9, 4, 9, 1, 4, STONE)
    w.fill(5, 8, 5, 8, 1, 4, AIR)
    for x in range(5, 8):
        for y in range(5, 8):
            w.pour(x, y, 1, WATER, w.cap)
    C = w.heat_capacity()
    w.E[5:8, 5:8, 1] = (108.0 - AMBIENT) * C[5:8, 5:8, 1]
    return w


# ---- the gate ------------------------------------------------------------------
def test_wet_off_is_the_old_world_bit_for_bit():
    # R4 (wave 14a refinement): every World is humid by default, so "wet off" is now the
    # explicit gate `w.wet = None` (every method falls back to super() on it), set on both.
    a, b = _hot_cup(ColdWorld), _hot_cup(WetWorld)
    a.wet = b.wet = None
    for _ in range(300):
        a.step()
        b.step()
    for f in ("E", "fvol", "fl", "mat", "smass"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f
    assert b.total_energy() == a.total_energy()


# ---- steam kept ---------------------------------------------------------------
def test_boil_keeps_steam_and_the_books_close():
    w = _hot_cup(WetWorld)
    w.enable_wet(rh=0.5)
    w.edges_off = None
    w.wet.edges = False                     # a closed test: nothing leaves the box
    e0, m0 = w.total_energy(), w.total_water()
    liquid0 = w.total_fluid(WATER)
    for _ in range(2 * EVERY):
        w.step()
    assert w.wet.stats["boil_g"] > 5.0                        # it really boiled
    assert w.total_fluid(WATER) < liquid0 - 5.0
    assert w.wet.store.total("H2O") + w.wet.store.total("MIST") > 3.0   # and the steam is HERE
    _closes(w, e0, m0)


def test_boil_ends_each_cell_at_the_boiling_point():
    from src.voxel.materials import BOIL
    w = _hot_cup(WetWorld)
    w.enable_wet(rh=0.5)
    w._law_boil()
    T = w.T()[5:8, 5:8, 1]
    assert np.all(np.abs(T - BOIL) < 0.05), T


# ---- the surface law ------------------------------------------------------------
def test_puddle_evaporates_at_the_surface_law_rate_and_cools():
    w = puddle(rh=0.5, ml=30.0)
    w.wet.edges = False
    e0, m0 = w.total_energy(), w.total_water()
    # ONE slow step, worked out by hand: tau = V res / (A F), m = (rho_eq - rho_air) V (1 - e^-dt/tau)
    A, V = 0.05 ** 2, (0.125 - 0.030) / 1000.0
    rho_eq = float(psychro.rho_sat(20.0))
    res = 1.0 / float(psychro.mass_coeff(10.0))
    F = float(psychro.stefan_factor(float(psychro.p_sat_water(20.0)) / 101325.0))
    tau = V * res / (A * F)
    per_cell = 0.5 * rho_eq * V * (1.0 - np.exp(-1.0 / tau))
    liquid0 = w.total_fluid(WATER)
    w._wet_surface(1.0)
    got = liquid0 - w.total_fluid(WATER)
    assert abs(got / (36 * per_cell) - 1.0) < 0.02, (got, 36 * per_cell)
    for _ in range(5 * EVERY):
        w.step()
    assert float(w.T()[7, 7, 1]) < AMBIENT                                # it cooled
    _closes(w, e0, m0)


def test_humid_air_stops_the_evaporation():
    dry, wet = puddle(rh=0.2), puddle(rh=0.95)
    for w in (dry, wet):
        w.wet.edges = False
        for _ in range(3 * EVERY):
            w.step()
    assert wet.wet.stats["evap_g"] < 0.1 * dry.wet.stats["evap_g"]


def test_dew_forms_on_a_cold_wall_in_wet_air():
    w = fog_room(rh=0.5)
    w.wet.edges = False
    e0, m0 = w.total_energy(), w.total_water()
    for _ in range(10 * EVERY):
        w.step()
    assert w.wet.stats["cond_g"] > 0.5
    _closes(w, e0, m0)


# ---- mist ------------------------------------------------------------------
def test_mist_by_saturation_adjustment_in_one_small_voxel():
    """T6 spec 24c: a 5 cm voxel at 150 percent saturation at 20 C ends saturated at about
    25.2 C with about 0.33 mg condensed; E and water close."""
    w = WetWorld(5, 5, 5, voxel_cm=5)
    w.enable_wet(rh=0.0)
    V = 0.05 ** 3
    st = w.wet.store
    st.add("H2O", (2, 2, 2), 1.5 * float(psychro.rho_sat(20.0)) * V)
    e0 = w.total_energy()
    w.wet.edges = False
    w._wet_mist(1.0)
    T = float(w.T()[2, 2, 2])
    mist = float(st.excess("MIST")[2, 2, 2])
    assert 24.2 < T < 26.2, T
    assert 0.2e-3 < mist < 0.5e-3, mist
    assert abs(w.total_energy() - e0) < 1e-6
    rh = float(w.rh_at((2, 2, 2)))
    assert 0.97 < rh < 1.03, rh


def test_mist_goes_back_to_vapour_in_dry_air():
    w = WetWorld(5, 5, 5, voxel_cm=5)
    w.enable_wet(rh=0.2)
    w.wet.store.add("MIST", (2, 2, 2), 2e-4)
    w.wet.edges = False
    m0 = w.total_water()
    for _ in range(40):
        w._wet_mist(1.0)
    assert float(w.wet.store.excess("MIST")[2, 2, 2]) < 1e-5
    assert abs(w.total_water() - m0) < 1e-9


# ---- moisture in wood ------------------------------------------------------------
def test_wet_log_dries_in_dry_air_and_the_books_close():
    w = wet_log(rh=0.3, moisture=0.30)
    w.wet.edges = False
    e0, m0 = w.total_energy(), w.total_water()
    f0 = w.total_fluid(WATER)
    for _ in range(20 * EVERY):
        w.step()
    assert w.total_fluid(WATER) < f0 - 1e-3
    _closes(w, e0, m0)


def test_wood_below_equilibrium_takes_water_from_wet_air():
    w = wet_log(rh=0.9, moisture=0.02)
    w.wet.edges = False
    f0 = w.total_fluid(WATER)
    for _ in range(20 * EVERY):
        w.step()
    assert w.total_fluid(WATER) > f0


# ---- ice ------------------------------------------------------------------------
def test_water_at_minus_five_freezes_to_the_pin_and_the_enthalpy_closes():
    w = pond(t_c=-5.0)
    w.wet.edges = False
    C0 = float(w.heat_capacity()[5, 5, 1])
    e0, m0 = w.total_energy(), w.total_water()
    w._wet_ice(1.0)
    n = 6 * 6 * 2
    expect = n * C0 * 5.0 / 333.55
    assert abs(w.ice_grams() - expect) / expect < 0.02
    T = w.T()[5, 5, 1]
    assert abs(T) < 0.01
    _closes(w, e0, m0, rel=1e-6)


def test_ice_cell_promotes_and_expansion_is_expelled_up():
    w = pond(t_c=-1.0, depth=1, size=2)
    w.wet.edges = False
    # a very cold pond: everything freezes, and the surplus volume goes up
    C = w.heat_capacity()
    w.E[3:5, 3:5, 1] = (-200.0 - AMBIENT) * C[3:5, 3:5, 1]
    e0, m0 = w.total_energy(), w.total_water()
    for _ in range(3):
        w._wet_ice(1.0)
    assert (w.mat[3:5, 3:5, 1] == ICE).all()
    cell = float(w.smass[3, 3, 1])
    assert abs(cell - 0.917 * w.cap) < 1.0                     # 917 g in a litre cell
    assert w.fvol[3, 3, 2] > 0                                 # the 83 g went up
    _closes(w, e0, m0, rel=1e-6)


def test_ice_melts_back_and_demotes():
    w = pond(t_c=-200.0, depth=1, size=2)
    w.wet.edges = False
    for _ in range(3):
        w._wet_ice(1.0)
    assert (w.mat[3:5, 3:5, 1] == ICE).all()
    C = w.heat_capacity()
    w.E[3:5, 3:5, 1:3] = (250.0 - AMBIENT) * C[3:5, 3:5, 1:3]     # enough heat to melt it all
    e0, m0 = w.total_energy(), w.total_water()
    for _ in range(4):
        w._wet_ice(1.0e6)
    assert w.ice_grams() < 1.0
    assert (w.mat[3:5, 3:5, 1] != ICE).all()
    _closes(w, e0, m0, rel=1e-6)


def test_salt_free_water_does_not_freeze_above_zero():
    w = pond(t_c=3.0)
    w._wet_ice(1.0)
    assert w.ice_grams() == 0.0


# ---- gas flow carries the vapour -------------------------------------------
def test_gas_flow_carries_steam_and_the_books_close():
    w = kettle(gas_flow=True, heater_c=130.0)
    w.thermostats = []
    C = w.heat_capacity()
    x0, y0, n = w.scene_info["pot"]
    w.E[x0:x0 + n, y0:y0 + n, 1:3] = (104.0 - AMBIENT) * C[x0:x0 + n, y0:y0 + n, 1:3]
    e0, m0 = w.total_energy(), w.total_water()
    for _ in range(3 * EVERY):
        w.step()
    assert w.wet.stats["boil_g"] > 1.0
    assert w.wet.store.total("H2O") > 0.2
    _closes(w, e0, m0, rel=5e-6)


# ---- one world -------------------------------------------------------------
def test_wet_composes_with_cold_body_heat_mixtures_and_material_laws():
    from src.voxel import climate as K
    from src.voxel.bodyworld import BodyWorld
    from src.voxel.scenes_mixtures import BRINE10

    class W(BodyWorld):        # R4: World already carries WetMixin
        pass

    n = 30
    w = W(n, n, 30, voxel_cm=5)
    w.fill(0, n, 0, n, 0, 1, STONE)
    w.fill(10, 15, 10, 15, 1, 4, STONE)
    w.fill(11, 14, 11, 14, 1, 4, AIR)
    w.pour(12, 12, 1, None, 50.0, mix=w.ensure_mixes().parse(BRINE10))
    for x in range(11, 14):
        w.pour(x, 11, 1, WATER, 60.0)
    w.set_climate(K.Climate.reference(t_c=-4.0), soak=True)
    w.enable_wet(rh=0.6, t_c=-4.0)
    assert w.climate is not None and w.mixes_live and w.wet is not None
    w.wet.edges = False
    e0, m0 = w.total_energy(), w.total_water()
    for _ in range(2 * EVERY):
        w.step()
    # the energy closes tightly. The water carries a SMALL drift that is not T6's: the same
    # world with wet OFF loses 0.059 g of this 230 g in 3 ticks (the legacy flow law's
    # slivers, C3 report open problems), so the water is held to 0.1 g here.
    scale = float(np.abs(w.E).sum()) + abs(e0) + 1.0
    assert abs(w.total_energy() - e0) < 5e-6 * scale
    assert abs(w.total_water() - m0) < 0.1


def test_gas_flow_lifts_the_vapour_with_the_warm_air_it_came_with():
    """Steam rises ONLY with gas_flow on (T6 spec 7): the same warm moist block, flow on,
    carries its vapour up; flow off (and the diffusion off) leaves it where it was."""
    z = {}
    for gf in (True, False):
        w = WetWorld(20, 20, 30, voxel_cm=5)
        w.gas_flow = gf
        w.fill(0, 20, 0, 20, 0, 1, STONE)
        w.enable_wet(rh=0.5)
        w.wet.edges = False
        w.wet.seep = False
        C = w.heat_capacity()
        w.E[8:12, 8:12, 2:4] = 80.0 * C[8:12, 8:12, 2:4]
        w.wet.store.add("H2O", (np.array([9, 10]), np.array([9, 10]), np.array([2, 3])),
                        np.array([1.0, 1.0]))
        e0, m0 = w.total_energy(), w.total_water()
        for _ in range(45):
            w.step()
        p = w.wet.store.excess("H2O").sum(axis=(0, 1))
        z[gf] = float((p * np.arange(30)).sum() / p.sum())
        if gf:
            _closes(w, e0, m0, rel=5e-6)
    assert z[True] > z[False] + 1.5, z
