"""T6 gas store and solutions (wave 13, lane A-T6). Tags as in test_t6_psychro.py."""
import numpy as np
import pytest

from src.voxel import _standin as S
from src.voxel import matterdata as D
from src.voxel import psychro as P
from src.voxel import solution as SOL
from src.voxel.gasstore import GasStore, gas_volume

VOX_L = 0.125                      # a 5 cm voxel


# ---- the gas store ---------------------------------------------------------------
def test_store_is_lazy_and_fresh_air_is_zero():
    gs = GasStore((4, 4, 4), VOX_L)
    assert gs.names() == [] and gs.total("H2O") == 0.0
    gs.set_background("H2O", 0.5 * float(P.rho_sat(20.0)))
    assert gs.names() == []                                     # nothing allocated yet
    assert gs.bg("CO2") == D.FRESH_CO2_G_M3 and gs.bg("O2") == 0.0
    assert np.allclose(gs.actual("H2O"), 0.5 * float(P.rho_sat(20.0)))
    assert gs.names() == ["H2O"]
    assert gs.total("H2O") == 0.0                                # excess, not actual


def test_add_take_conserve_float64():
    gs = GasStore((5, 5, 5), VOX_L, dtype=np.float64)
    rng = np.random.default_rng(2)
    put = 0.0
    for _ in range(50):
        idx = tuple(rng.integers(0, 5, 3))
        g = rng.uniform(0, 1e-3)
        put += float(gs.add("H2O", idx, g))
    assert gs.total("H2O") == pytest.approx(put, rel=1e-12)
    taken = 0.0
    for _ in range(50):
        idx = tuple(rng.integers(0, 5, 3))
        taken += float(gs.take("H2O", idx, 5e-4))
    assert gs.total("H2O") == pytest.approx(put - taken, abs=1e-15)
    assert taken <= put + 1e-15           # H2O has no background here: cannot go below 0


def test_add_take_float32_conserves_to_1e6():
    gs = GasStore((6, 6, 6), VOX_L)
    mask = np.zeros((6, 6, 6), bool)
    mask[1:5, 1:5, 1:5] = True
    gs.add("H2O", mask, np.full(64, 3.3e-3))
    t0 = gs.total("H2O")
    got = gs.take("H2O", mask, np.full(64, 1.1e-3))
    assert gs.total("H2O") == pytest.approx(t0 - float(got.sum()), rel=1e-6)


def test_excess_floor_is_minus_the_background():
    gs = GasStore((2, 2, 2), VOX_L, dtype=np.float64)
    V = VOX_L / 1000.0
    floor = -D.FRESH_CO2_G_M3 * V
    got = gs.take("CO2", (0, 0, 0), 1.0)                  # try to eat far more than exists
    assert float(got) == pytest.approx(-floor)            # the whole background, no more
    assert gs.excess("CO2")[0, 0, 0] == pytest.approx(floor)
    assert gs.actual("CO2")[0, 0, 0] == pytest.approx(0.0, abs=1e-12)
    assert float(gs.take("CO2", (0, 0, 0), 1.0)) == 0.0   # nothing left to take
    gs.add("CO2", (0, 0, 0), 1e-4)
    assert gs.actual("CO2")[0, 0, 0] == pytest.approx(1e-4 / V)


def test_rh_is_over_the_free_gas_volume():
    gs = GasStore((3, 3, 3), VOX_L, dtype=np.float64)
    gs.set_background("H2O", 0.5 * float(P.rho_sat(20.0)))
    V = VOX_L / 1000.0
    # add exactly the grams that saturate a full voxel
    gs.add("H2O", (1, 1, 1), 0.5 * float(P.rho_sat(20.0)) * V)
    assert gs.rh("H2O", 20.0)[1, 1, 1] == pytest.approx(1.0, rel=1e-12)
    assert gs.rh("H2O", 20.0)[0, 0, 0] == pytest.approx(0.5, rel=1e-12)
    # a half-full-of-water voxel has half the gas: the same grams read higher
    half = np.full((3, 3, 3), VOX_L / 2000.0)
    assert gs.rh("H2O", 20.0, half)[1, 1, 1] > 1.4
    # and the rh agrees with p_sat: rho_actual / rho_sat = p_actual / p_sat at one T
    rho = gs.actual("H2O")[1, 1, 1]
    assert rho / float(P.rho_sat(20.0)) * float(P.p_sat_water(20.0)) == pytest.approx(
        float(P.p_sat_water(20.0)), rel=1e-12)


def test_gas_volume_leaves_what_liquid_and_solid_leave():
    v = gas_volume(0.125, np.array([0.0, 62.5, 125.0]), 0.0)
    assert v[0] == pytest.approx(0.125e-3) and v[1] == pytest.approx(0.0625e-3)
    assert v[2] > 0                                   # a full cell never divides by zero


def test_partial_pressure_of_fresh_co2_is_400_ppm():
    gs = GasStore((2, 2, 2), VOX_L)
    p = float(gs.partial_pressure("CO2", D.M["CO2"])[0, 0, 0])
    assert abs(p - 40.5) < 1.5                         # 400 ppm of 101325 Pa


def test_legacy_seep_conserves_and_spreads():
    gs = GasStore((8, 8, 8), VOX_L, dtype=np.float64)
    air = np.ones((8, 8, 8), bool)
    air[4, :, :] = False                                # a wall
    por = np.ones((8, 8, 8))
    gs.add("H2O", (2, 4, 4), 1.0)
    for _ in range(40):
        gs.legacy_seep("H2O", por, air, 1.0 / 6.0)
    e = gs.excess("H2O")
    assert gs.total("H2O") == pytest.approx(1.0, rel=1e-12)
    assert e[4, :, :].sum() == 0.0                     # the wall holds no gas
    assert e[3, 4, 4] > e[0, 0, 0] > 0
    assert e[5:, :, :].sum() == 0.0                    # nothing crossed the wall
    with pytest.raises(ValueError):
        gs.legacy_seep("H2O", por, air, 0.5)


def test_shift_outdoor_books_the_exact_grams_and_a_sealed_room_keeps_its_own():
    shape = (6, 6, 6)
    gs = GasStore(shape, VOX_L, dtype=np.float64)
    rh = 0.5
    gs.set_background("H2O", rh * float(P.rho_sat(20.0)))
    out = np.ones(shape, bool)
    room = np.zeros(shape, bool)
    room[2:4, 2:4, 2:4] = True
    out &= ~room
    gs.add("H2O", room, np.full(8, 2e-4))               # humid sealed room
    room_total = gs.total("H2O")
    booked = 0.0
    T_prev = 20.0
    for T in np.linspace(20.0, 0.0, 21)[1:]:           # t_out ramps 20 -> 0 C
        booked += gs.shift_outdoor("H2O", out, rh * float(P.rho_sat(T_prev)),
                                   rh * float(P.rho_sat(T)))
        T_prev = T
    V = VOX_L / 1000.0
    want = out.sum() * (rh * float(P.rho_sat(0.0)) - rh * float(P.rho_sat(20.0))) * V
    assert booked == pytest.approx(want, rel=1e-9)
    e = gs.excess("H2O")
    assert e[room].sum() == pytest.approx(room_total, rel=1e-12)       # sealed room untouched
    assert gs.total("H2O") == pytest.approx(room_total + booked, rel=1e-9)
    # the outdoor cells now hold fresh air at the new climate: actual = rh rho_sat(0)
    assert gs.actual("H2O")[0, 0, 0] == pytest.approx(rh * float(P.rho_sat(0.0)), rel=1e-9)


def test_background_is_fixed_once_the_store_holds_it():
    gs = GasStore((2, 2, 2), VOX_L)
    gs.set_background("H2O", 5.0)
    gs.add("H2O", (0, 0, 0), 1e-4)
    gs.set_background("H2O", 5.0)                       # same value: fine
    with pytest.raises(ValueError):
        gs.set_background("H2O", 6.0)


def test_generation_changes_when_the_store_changes():
    gs = GasStore((2, 2, 2), VOX_L)
    g0 = gs.generation
    gs.add("H2O", (0, 0, 0), 1e-5)
    assert gs.generation > g0


# ---- solutions ---------------------------------------------------------------------
def _dens(species, wt, **kw):
    v = SOL.v_of(species)
    ms = np.array([[wt]], np.float64)
    return float(SOL.density(np.array([100.0 - wt]), ms, v, **kw)[0])


def test_u_volume_rule_densities():
    assert abs(_dens("NaCl", 20.0) - 1.148) < 0.005
    assert abs(_dens("sucrose", 20.0) - 1.081) < 0.005
    mw, ms = 100.0, np.array([[203.9]])
    v = SOL.v_of("sucrose")
    assert abs(float(SOL.density(np.array([mw]), ms, v)[0]) - 1.33) < 0.01
    V = float(SOL.solution_volume(np.array([mw]), ms, v)[0])
    assert abs(V / mw - 2.27) < 0.01


def test_water_grams_inverts_the_volume():
    rng = np.random.default_rng(5)
    v = SOL.v_of("NaCl", "sucrose")
    mw = rng.uniform(10, 200, 20)
    ms = rng.uniform(0, 50, (20, 2))
    V = SOL.solution_volume(mw, ms, v)
    assert np.allclose(SOL.water_grams(V, ms, v), mw, rtol=0, atol=1e-12)
    assert np.allclose(SOL.conc(ms[:, 0], V), ms[:, 0] / V)


def test_the_volume_rule_is_injected():
    v = SOL.v_of("NaCl")
    rule = SOL.VolumeRule(lambda mw, ms, vv: mw + 0.5 * np.sum(ms * vv, axis=-1),
                          lambda V, ms, vv: V - 0.5 * np.sum(ms * vv, axis=-1))
    ms = np.array([[20.0]])
    V = float(SOL.solution_volume(np.array([80.0]), ms, v, rule)[0])
    assert V == pytest.approx(80.0 + 0.5 * 0.355 * 20.0)
    assert float(SOL.water_grams(np.array([V]), ms, v, rule)[0]) == pytest.approx(80.0)
    assert _dens("NaCl", 20.0, rule=rule) != pytest.approx(_dens("NaCl", 20.0))


def test_dissolve_never_passes_c_sat_for_any_dt():
    b_sat = float(S.solubility_load("NaCl", 20.0))
    mw = 100.0
    for dt in (1e-3, 1.0, 100.0, 1e6, 1e12):
        for kappa in (1e-4, 0.05, 50.0):
            xs, xd, d = SOL.dissolve_step(40.0, 0.0, mw, b_sat, kappa, dt)
            assert xd <= b_sat * mw * (1 + 1e-12)
            assert xs + xd == pytest.approx(40.0, rel=1e-12) and xs >= 0
    # already saturated: nothing dissolves
    xs, xd, d = SOL.dissolve_step(5.0, b_sat * mw, mw, b_sat, 1.0, 100.0)
    assert d == 0.0
    # a small pile dissolves completely
    xs, xd, d = SOL.dissolve_step(1.0, 0.0, mw, b_sat, 1.0, 1e6)
    assert xs == 0.0 and xd == pytest.approx(1.0)


def test_dissolve_follows_the_solubility_points_with_T():
    for sp in ("NaCl", "sucrose"):
        for T in (0.0, 20.0, 40.0, 60.0, 80.0, 100.0):
            b = float(S.solubility_load(sp, T))
            xs, xd, _ = SOL.dissolve_step(1e4, 0.0, 100.0, b, 1.0, 1e9)
            assert xd == pytest.approx(100.0 * b, rel=1e-9)
    assert float(S.solubility("sucrose", 20.0)) == pytest.approx(203.9)


def test_dissolve_approach_is_exponential():
    b = float(S.solubility_load("NaCl", 20.0))
    _, d1, _ = SOL.dissolve_step(40.0, 0.0, 100.0, b, 0.1, 5.0)
    _, d2, _ = SOL.dissolve_step(40.0, d1, 100.0, b, 0.1, 5.0)
    _, d10, _ = SOL.dissolve_step(40.0, 0.0, 100.0, b, 0.1, 10.0)
    assert d2 == pytest.approx(d10, rel=1e-12)             # two 5-s steps = one 10-s step


def test_sugar_stays_dissolved_to_s_crit_then_crystallises_and_a_seed_removes_the_barrier():
    b = float(S.solubility_load("sucrose", 20.0))
    sc = SOL.s_crit("sucrose")
    mw = 100.0
    # 5 % over saturation: supersaturated but inside the barrier
    assert float(SOL.precipitate_step(1.05 * b * mw, mw, b, sc, 60.0, 600.0)) == 0.0
    # 20 % over: crystallises, toward saturation, never below
    p = float(SOL.precipitate_step(1.20 * b * mw, mw, b, sc, 60.0, 1e9))
    assert p == pytest.approx(0.20 * b * mw, rel=1e-9)
    p1 = float(SOL.precipitate_step(1.20 * b * mw, mw, b, sc, 60.0, 30.0))
    assert 0 < p1 < p
    # a seed removes the barrier: the 5 % over crystallises
    ps = float(SOL.precipitate_step(1.05 * b * mw, mw, b, sc, 60.0, 1e9, seeded=True))
    assert ps == pytest.approx(0.05 * b * mw, rel=1e-9)


def test_evaporating_dilute_salt_water_concentrates_then_crystallises_near_10_percent_left():
    b = float(S.solubility_load("NaCl", 20.0))
    mw0, s0 = 96.5, 3.5                                  # 3.5 wt % salt water
    first = None
    mw, d, x = mw0, s0, 0.0
    for i in range(1, 2000):
        mw, d, x = SOL.evaporate_concentrate(mw, d, x, 0.05, b)
        if x > 0 and first is None:
            first = mw / mw0
        # mass exact at every step
        assert d + x == pytest.approx(s0, abs=1e-12)
        assert mw >= 0.0
    assert first is not None and 0.07 < first < 0.12
    assert mw == pytest.approx(0.0, abs=1e-9) and x == pytest.approx(s0)   # a crust
    # dissolved never above saturation on the way
    mw2, d2, x2 = SOL.evaporate_concentrate(50.0, 3.5, 0.0, 49.0, b)
    assert d2 <= b * mw2 + 1e-12


def test_heat_of_solution_sign_and_size():
    assert SOL.heat_of_solution("NaCl", 10.0) == pytest.approx(-660.0)      # takes 660 J
    assert SOL.heat_of_solution("NaCl", -10.0) == pytest.approx(660.0)      # crystallising gives it
    assert SOL.heat_of_solution("NaOH", 1.0) == pytest.approx(1113.0)       # exothermic
    assert SOL.heat_of_solution("sucrose", 100.0) == pytest.approx(-1700.0)


def test_u_crust_takes_water_at_80_percent_rh_but_not_at_70():
    b_sat = float(S.solubility_load("NaCl", 20.0))
    a_crust = float(S.aw("NaCl", b_sat))
    assert abs(a_crust - 0.753) < 0.01
    rho = float(P.rho_sat(20.0))
    J80 = float(P.surface_flux(10.0, 20.0, a_crust, 0.80 * rho))
    J70 = float(P.surface_flux(10.0, 20.0, a_crust, 0.70 * rho))
    assert J80 < 0 < J70


def test_gasstore_duplicate_indices_respect_the_floor():
    gs = GasStore((4, 4, 4), 0.125)
    idx = (np.array([1, 1, 2]), np.array([1, 1, 2]), np.array([1, 1, 2]))
    floor = -float(gs.bg("CO2")) * 0.125 / 1000.0
    app = gs.add("CO2", idx, np.array([-1.0, -1.0, 0.5]))
    ex = gs.excess("CO2")
    assert float(ex[1, 1, 1]) >= floor * (1 + 1e-5)
    assert float(ex[1, 1, 1]) == pytest.approx(floor, rel=1e-5)
    assert float(ex[2, 2, 2]) == pytest.approx(0.5, rel=1e-6)
    assert float(app[:2].sum()) == pytest.approx(floor, rel=1e-5)
