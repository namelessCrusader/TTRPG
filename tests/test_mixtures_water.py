"""T4 mixtures, water (wave 13, law L14): vapour, ice, the activity of water, boiling
rise, freezing drop, osmosis, density, closed forms. No World.

Tags: test_exact_* (conservation, exactness, identities), test_strict_* (independent
number, a row that is not (u)), test_u_* (physical number, source unverified: reads
tests/data/sourced_numbers.json when it exists), test_echo_* (a fitted constant
reproducing its fit), test_codeonly_* (the code against its own table).
"""
import json
import os
import time

import numpy as np
import pytest

from src.voxel import mixdata as D
from src.voxel import mixwater as W
from src.voxel import mixtures as M

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCED = os.path.join(HERE, "data", "sourced_numbers.json")
TIMING = not os.environ.get("W13_NO_TIMING")


def sourced(key):
    try:
        with open(SOURCED) as f:
            e = json.load(f)["entries"]
    except OSError:
        return None, "no-file"
    if key not in e:
        return None, "no-key"
    return e[key]["value"], e[key]["status"]


def best(fn, n=30):
    b = 1e9
    for _ in range(n):
        t = time.perf_counter()
        fn()
        b = min(b, time.perf_counter() - t)
    return b


mk = M.mix


def brine(wt):
    return mk(h2o=100.0 - wt, nacl=wt)


def molal(name, m, kg=1.0):
    return mk(h2o=1000.0 * kg, **{name: m * D.SPECIES[name].M * kg})


# -- vapour ------------------------------------------------------------------------------
def test_strict_if97_region4_matches_the_iapws_verification_values():
    for k in ("n", "p_min_pa", "p_max_pa"):
        D.refuse_u("if97", k)
    # IAPWS-IF97 Table 35 (p_sat at T) and Table 36 (T_sat at p)
    for Tk, p in ((300.0, 0.353658941e-2), (500.0, 0.263889776e1), (600.0, 0.123443146e2)):
        assert W._psat_k(Tk) / 1e6 == pytest.approx(p, rel=1e-8)
    for p, Tk in ((0.1, 0.372755919e3), (1.0, 0.453035632e3), (10.0, 0.584149488e3)):
        assert W.tsat_water(p * 1e6) + 273.15 == pytest.approx(Tk, abs=1e-5)


def test_strict_tsat_of_one_atmosphere_and_psat_at_20_c():
    assert W.tsat_water(101325.0) == pytest.approx(99.974, abs=0.002)
    assert W.psat_water(20.0) == pytest.approx(2339.0, abs=2.0)
    assert W.tsat_water(611.657) == pytest.approx(0.01, abs=0.005)       # the triple point


def test_exact_tsat_and_psat_return_nan_outside_the_domain():
    assert np.isnan(W.tsat_water(300.0)) and np.isnan(W.tsat_water(30e6)) and np.isnan(W.tsat_water(50.0))
    assert np.isnan(W.psat_water(-1.0)) and np.isnan(W.psat_water(400.0))
    assert np.isnan(W.psat_ice(5.0))
    got = W.tsat_water(np.array([300.0, 101325.0, 30e6]))
    assert np.isnan(got[0]) and np.isnan(got[2]) and np.isfinite(got[1])


def test_exact_round_trip_psat_tsat_to_1e_6():
    T = np.linspace(1.0, 370.0, 200)
    assert np.max(np.abs(W.tsat_water(W.psat_water(T)) - T)) < 1e-6


def test_codeonly_antoine_row_of_t1_within_1_percent_and_magnus_within_half():
    A, B, C = D.row("antoine", "water_t1")
    T = np.linspace(1.0, 100.0, 60)
    ant = 10.0 ** (A - B / (C + T)) * D.MMHG
    assert np.max(np.abs(ant / W.psat_water(T) - 1.0)) < 0.01
    T = np.linspace(0.0, 50.0, 40)
    mag = 6.1094 * np.exp(17.625 * T / (T + 243.04)) * 100.0
    assert np.max(np.abs(mag / W.psat_water(T) - 1.0)) < 0.005


def test_u_murphy_koop_ice_and_the_shifted_ratio():
    assert float(W.ln_ice_ratio(0.0)) == 0.0                       # exactly zero at 0 C
    assert W.psat_ice(-20.0) == pytest.approx(103.3, rel=0.01)
    assert float(W.psat_ice(-20.0)) < float(W.psat_water(0.0)) * np.exp(-1.0)
    assert np.all(np.diff(W.ln_ice_ratio(np.linspace(-60.0, 0.0, 50))) > 0)
    for k in ("ice", "liquid"):
        D.refuse_u("mk05", k)


def test_codeonly_fusion_heat_follows_kirchhoff():
    assert float(W.Lfus(0.0)) == 333.55
    assert float(W.Lfus(-21.0)) == pytest.approx(289.66, abs=0.3)
    assert float(W.Lfus(-21.0)) == pytest.approx(333.55 + 2.09 * -21.0)


# -- pure water: offsets are exactly zero ------------------------------------------------------
def test_exact_pure_water_rise_and_drop_are_exactly_zero():
    c = mk(h2o=100.0)
    assert float(W.boiling_rise(c)) == 0.0
    assert float(W.freezing_drop(c)) == 0.0
    assert float(W.boiling_rise(c, 70.1e3)) == 0.0
    cc = np.stack([c, brine(10.0), c])
    r = W.boiling_rise(cc)
    d = W.freezing_drop(cc)
    assert r[0] == 0.0 and r[2] == 0.0 and r[1] > 1.0
    assert d[0] == 0.0 and d[2] == 0.0 and d[1] > 5.0
    assert float(W.ln_aw(c)) == 0.0 and float(W.osmolality(c)) == 0.0


# -- the activity of water ---------------------------------------------------------------------
def test_u_nacl_and_kcl_aw_at_the_reference_molalities():
    assert float(W.aw(molal("nacl", 6.15), 25.0)) == pytest.approx(0.752, abs=0.004)
    assert float(W.aw(molal("kcl", 4.8), 25.0)) == pytest.approx(0.843, abs=0.005)
    phi = float(W.phi_pitzer(0.1554, 25.0, "nacl"))
    assert phi == pytest.approx(0.926, abs=0.005)


def test_echo_saturated_nacl_aw_at_25_c_and_at_the_crc_boiling_point():
    sat = mk(h2o=100.0, nacl=36.0)
    assert float(W.aw(sat, 25.0)) == pytest.approx(0.753, abs=0.003)
    # the CRC boiling point 108.7 C implies aw = P / psat(108.7 C) = 0.738
    sat2 = mk(h2o=100.0, nacl=39.7)
    assert D.P_ATM / float(W.psat_water(108.7)) == pytest.approx(0.738, abs=0.006)
    assert float(W.aw(sat2, 108.7)) == pytest.approx(0.738, abs=0.006)


def test_exact_zsr_split_invariance_one_species_in_two_halves():
    with D.scratch_registry():
        D.register_species("nacl_b", 58.443, "salt", nu=2, model="nacl", src="pack:test")
        D.register_species("sucrose_b", 342.30, "sugar", model="sucrose", src="pack:test")
        D.register_species("glucose_b", 180.156, "sugar", model="glucose", src="pack:test")
        whole = mk(h2o=1000.0, nacl=6.15 * 58.443)
        halves = mk(h2o=1000.0, nacl=3.075 * 58.443, nacl_b=3.075 * 58.443)
        for T in (25.0, 60.0, 100.0):
            a, b = float(W.aw(whole, T)), float(W.aw(halves, T))
            assert abs(a - b) < 1e-9, (T, a, b)
        assert float(W.aw(whole, 25.0)) == pytest.approx(0.7521, abs=2e-4)
        s_whole = mk(h2o=100.0, sucrose=200.0)
        s_half = mk(h2o=100.0, sucrose=100.0, sucrose_b=100.0)
        assert abs(float(W.aw(s_whole)) - float(W.aw(s_half))) < 1e-9
        g1 = mk(h2o=100.0, glucose=40.0, fructose=40.0)
        g2 = mk(h2o=100.0, glucose=20.0, glucose_b=20.0, fructose=40.0)
        assert abs(float(W.aw(g1)) - float(W.aw(g2))) < 1e-9          # ZSR path, two model groups
        # and the old Ross rule would have moved it: the sucrose halves differ by 0.02 under Ross
        halves_ross = np.exp(2 * W.lnaw_binary("sucrose", 1000 * 100.0 / 342.30 / 100.0, 25.0))
        assert abs(halves_ross - float(W.aw(s_whole))) > 0.01


def test_exact_aw_never_above_one_and_each_added_solute_lowers_it():
    a_nacl = float(W.aw(mk(h2o=1000.0, nacl=2 * 58.443)))
    a_suc = float(W.aw(mk(h2o=1000.0, sucrose=3 * 342.3)))
    a_both = float(W.aw(mk(h2o=1000.0, nacl=2 * 58.443, sucrose=3 * 342.3)))
    assert a_both < min(a_nacl, a_suc) < max(a_nacl, a_suc) < 1.0
    ms = np.linspace(0.01, 6.0, 40)
    aws = np.array([float(W.aw(molal("nacl", m))) for m in ms])
    assert np.all(np.diff(aws) < 0) and aws.max() < 1.0


def test_u_zsr_mixed_solutes_and_honey():
    mixed = float(W.aw(mk(h2o=1000.0, nacl=2 * 58.443, sucrose=3 * 342.3)))
    assert mixed == pytest.approx(0.8500, abs=0.003)                # Ross would give 0.8688
    assert float(W.aw(mk(h2o=18.0, glucose=40.0, fructose=40.0))) == pytest.approx(0.58, abs=0.03)
    for water, a in ((17.0, 0.55), (22.0, 0.65)):                  # the 0.60 gate sits between
        rest = (100.0 - water) / 2.0
        assert float(W.aw(mk(h2o=water, glucose=rest, fructose=rest))) == pytest.approx(a, abs=0.03)


def test_u_sucrose_aw_by_norrish():
    assert float(W.aw(mk(h2o=35.0, sucrose=65.0))) == pytest.approx(0.865, abs=0.02)   # CRC about 0.85
    D.refuse_u("norrish", "sucrose")


# -- boiling -------------------------------------------------------------------------------------
def test_echo_saturated_nacl_boils_at_108_7_with_the_fitted_kappa():
    assert float(W.boiling_point(mk(h2o=100.0, nacl=39.7))) == pytest.approx(108.7, abs=0.3)
    assert D.src_of("pitzer", "nacl_kappa") == "fit:crc-nacl-bp"


def test_u_brine_and_syrup_boiling_points():
    assert float(W.boiling_point(brine(10.0))) == pytest.approx(101.8, abs=0.3)
    assert float(W.boiling_point(brine(20.0))) == pytest.approx(104.6, abs=0.4)
    assert float(W.boiling_rise(mk(h2o=33.5, sucrose=66.5))) == pytest.approx(4.0, abs=0.7)
    r85 = float(W.boiling_rise(mk(h2o=15.0, sucrose=85.0)))
    assert 12.0 <= r85 <= 20.0


def test_strict_three_thousand_metres_boils_at_90_c():
    assert float(W.tsat_water(70.1e3)) == pytest.approx(90.0, abs=0.3)
    assert float(W.boiling_point(mk(h2o=100.0), 70.1e3)) == pytest.approx(90.0, abs=0.3)


def test_u_dilute_limit_of_the_nacl_rise_and_drop():
    r = float(W.boiling_rise(molal("nacl", 1e-4))) / 1e-4
    assert r == pytest.approx(1.02, rel=0.03)
    assert float(W.boiling_rise(molal("nacl", 0.1))) / 0.1 == pytest.approx(0.92, abs=0.03)
    assert float(W.freezing_drop(molal("nacl", 1e-3))) / 1e-3 == pytest.approx(3.71, rel=0.03)


def test_exact_boiling_rise_and_point_are_monotone():
    ws = np.linspace(0.0, 26.0, 30)
    rise = np.array([float(W.boiling_rise(brine(w))) for w in ws])
    assert np.all(np.diff(rise) > 0)
    P = np.linspace(2e3, 2e6, 80)
    assert np.all(np.diff(W.tsat_water(P)) > 0)


def test_exact_boiling_rise_is_nan_past_the_critical_pressure():
    assert np.isnan(float(W.boiling_rise(mk(h2o=100.0, nacl=39.7), 22e6)))


def test_exact_a_host_boiling_point_sets_the_pure_water_value():
    c = brine(10.0)
    assert float(W.boiling_point(mk(h2o=100.0), tb_water=100.0)) == 100.0
    assert float(W.boiling_point(c, tb_water=100.0)) == pytest.approx(100.0 + float(W.boiling_rise(c)), abs=1e-12)


# -- freezing ---------------------------------------------------------------------------------------
def test_u_nacl_freezing_drop_from_the_crc_table():
    for wt, t in ((1, -0.59), (5, -3.05), (10, -6.56), (20, -16.46)):
        assert float(W.freeze_point(brine(float(wt)))) == pytest.approx(t, abs=0.3), wt


def test_u_nacl_freezing_curve_against_the_sourced_table(record_property):
    v, status = sourced("nacl.freezing_curve_C")
    record_property("sourced_status", status)
    if v is None:
        pytest.skip(f"sourced_numbers: {status}")
    for wt, t in v["wt_pct_to_freezing_C"].items():
        assert float(W.freeze_point(brine(float(wt)))) == pytest.approx(t, abs=0.3), wt


def test_u_sucrose_freezing_drops():
    assert float(W.freeze_point(mk(h2o=80.0, sucrose=20.0))) == pytest.approx(-1.5, abs=0.3)
    assert float(W.freeze_point(mk(h2o=60.0, sucrose=40.0))) == pytest.approx(-4.3, abs=0.6)


def test_echo_ethanol_water_freezing_from_the_digitised_ice_line():
    d20 = float(W.freeze_point(mk(h2o=83.6, ethanol=16.4)))
    d40 = float(W.freeze_point(mk(h2o=66.6, ethanol=33.4)))
    assert d20 == pytest.approx(-9.0, abs=2.0) and d40 == pytest.approx(-27.0, abs=3.0)
    assert np.isnan(float(W.freeze_point(mk(h2o=50.0, ethanol=50.0))))       # no data above 40 wt%
    assert D.src_of("ethanol_ice", "points") == "fit:ethanol-ice"


def test_u_ethanol_freezing_against_the_sourced_crc_curve(record_property):
    """The sourced secondary CRC curve reads 16.4 wt% = -7.5 C and 33.8 wt% = -23.6 C;
    the digitised rule-of-thumb line the spec calls for reads -9 and -27. Within
    the spec band at 20 percent ABV; 3.4 K colder at 40 percent: the source-check
    pass should replace the table (reported to the user)."""
    v, status = sourced("ethanol.freezing_curve_C")
    record_property("sourced_status", status)
    if v is None:
        pytest.skip(f"sourced_numbers: {status}")
    w, t = np.array(v["wt_pct_ethanol"]), np.array(v["freezing_C"])
    d16 = float(W.freeze_point(mk(h2o=100 - 16.4, ethanol=16.4)))
    assert d16 == pytest.approx(np.interp(16.4, w, t), abs=2.0)
    d33 = float(W.freeze_point(mk(h2o=100 - 33.4, ethanol=33.4)))
    record_property("ethanol_33wt_model_minus_sourced_K", d33 - float(np.interp(33.4, w, t)))
    assert abs(d33 - float(np.interp(33.4, w, t))) < 4.0


# -- osmosis -----------------------------------------------------------------------------------------
def test_u_saline_osmolality_freezing_and_osmotic_pressure():
    c = mk(h2o=99.1, nacl=0.9)
    assert 0.285 <= float(W.osmolality(c)) <= 0.295
    assert float(W.freeze_point(c)) == pytest.approx(-0.52, abs=0.03)
    assert float(W.osmotic_pressure(c, 37.0)) / D.P_ATM == pytest.approx(7.3, abs=0.3)


def test_exact_tonicity_ethanol_is_zero_and_nacl_equals_osmolality():
    e = mk(h2o=90.0, ethanol=10.0)
    n = brine(10.0)
    assert float(W.tonicity(e)) == pytest.approx(0.0, abs=1e-12)
    assert float(W.tonicity(n)) == pytest.approx(float(W.osmolality(n)), rel=1e-12)
    assert float(W.osmolality(e)) > 0.5                                  # ethanol still counts in the total
    both = mk(h2o=90.0, ethanol=5.0, nacl=5.0)
    assert 0.0 < float(W.tonicity(both)) < float(W.osmolality(both))


# -- vectors -----------------------------------------------------------------------------------------
def _big(N=100_000, seed=3):
    rng = np.random.default_rng(seed)
    c = np.zeros((N, D.n_species()))
    c[:, 0] = rng.uniform(50, 500, N)
    c[:, D.INDEX["nacl"]] = c[:, 0] * rng.uniform(0.01, 0.3, N)
    c[::3, D.INDEX["sucrose"]] = c[::3, 0] * 0.2
    return c, rng


def test_exact_vector_call_equals_the_permuted_input_call_exactly():
    c, rng = _big()
    b = W.boiling_point(c)
    perm = rng.permutation(len(c))
    assert np.array_equal(W.boiling_point(c[perm]), b[perm])
    d = W.ln_aw(c, 100.0)
    assert np.array_equal(W.ln_aw(c[perm], 100.0), d[perm])


def test_exact_vector_call_equals_a_one_element_loop_to_1e_12():
    c, rng = _big(20_000)
    b = W.boiling_point(c)
    idx = rng.choice(len(c), 120, replace=False)
    one = np.array([float(W.boiling_point(c[i:i + 1])[0]) for i in idx])
    assert np.max(np.abs(one - b[idx]) / np.abs(b[idx])) < 1e-12
    f = W.freezing_drop(c[idx])
    f1 = np.array([float(W.freezing_drop(c[i:i + 1])[0]) for i in idx])
    assert np.max(np.abs(f - f1) / np.abs(f)) < 1e-12


def test_codeonly_a_hundred_thousand_cells_stay_inside_memory_and_time():
    c, _ = _big()
    t = time.perf_counter()
    b = W.boiling_point(c)
    dt = time.perf_counter() - t
    assert np.isfinite(b).all() and b.min() > 100.0
    if TIMING:
        assert dt < 5.0, dt                    # the spec's per-element figure is far below this


def test_exact_the_prefilter_on_a_hundred_thousand_brine_cells_costs_under_2_ms():
    rng = np.random.default_rng(1)
    ids = rng.integers(0, 50, 100_000)
    rows = np.stack([brine(float(w)) for w in np.linspace(1.0, 25.0, 50)])
    tb_lo = M.t_boil_1atm(rows)
    T = np.full(100_000, 25.0)
    assert not M.boil_prefilter(ids, T, tb_lo).any()                    # a cold brine is never evaluated
    assert M.boil_prefilter(ids, T + 90.0, tb_lo).all()
    if TIMING:
        assert best(lambda: M.boil_prefilter(ids, T, tb_lo), 20) < 2e-3


def test_exact_by_row_evaluates_once_per_distinct_row_and_gathers():
    calls = []
    rows = np.stack([brine(float(w)) for w in (5.0, 10.0, 15.0)])

    def fn(u, T):
        calls.append(len(u))
        return W.boiling_point(rows[u], 101325.0), u * 2

    ids = np.array([[0, 1, 2, 1], [2, 2, 0, 0]])
    Tq = np.full(ids.shape, 90.0)
    tb, dbl = M.by_row(fn, ids, Tq)
    assert calls == [3] and tb.shape == ids.shape and np.array_equal(dbl, ids * 2)
    assert np.array_equal(tb, W.boiling_point(rows[ids]))


# -- density and heat capacity --------------------------------------------------------------------------
def test_u_density_of_brine_and_spirit():
    assert float(W.density(brine(14.0), 20.0)) == pytest.approx(1.0995, abs=0.004)
    assert float(W.density(brine(20.0), 20.0)) == pytest.approx(1.147, abs=0.006)
    assert float(W.density(mk(h2o=100.0), 20.0)) == pytest.approx(0.9982, abs=0.0005)
    # specific gravity 1.13 is about 18 wt% salt (the seed-sorting brine)
    assert float(W.density(brine(18.0), 20.0)) == pytest.approx(1.131, abs=0.006)
    assert float(W.density(mk(h2o=60.0, ethanol=40.0), 20.0)) == pytest.approx(0.93518, abs=1e-4)
    # ethanol-water is not volume additive: 50/50 by mass is denser than the volume-additive guess
    add = 100.0 / (50.0 / 0.998 + 50.0 / 0.789)
    assert float(W.density(mk(h2o=50.0, ethanol=50.0), 20.0)) > add + 0.01
    assert float(W.density(mk(h2o=50.0, ethanol=50.0), 20.0)) < float(W.density(mk(h2o=100.0), 20.0))
    assert 1.03 < float(W.density(brine(5.0))) < 1.10 < float(W.density(brine(14.0)))     # an egg floats in 14 percent


def test_exact_heat_capacity_is_mass_weighted_and_ice_counts_2_09():
    c = mk(h2o=100.0, nacl=10.0)
    assert float(W.heat_capacity(c)) == pytest.approx(100 * 4.18 + 10 * 0.86)
    assert float(W.heat_capacity(c, ice_g=100.0)) == pytest.approx(100 * 2.09 + 10 * 0.86)
    assert float(W.heat_capacity(c, ice_g=40.0)) == pytest.approx(60 * 4.18 + 40 * 2.09 + 8.6)


def test_u_abv_to_grams_ten_percent_is_eight_grams_in_a_hundred():
    assert float(W.abv_to_grams(10.0, 100.0)) == pytest.approx(8.0, abs=0.2)
    c = mk(h2o=92.0, ethanol=8.0)
    assert float(W.abv_of(c)) == pytest.approx(10.0, abs=0.3)
    assert float(W.abv_to_grams(0.0, 100.0)) == 0.0


def test_exact_water_for_aw_and_for_boiling_rise_close():
    c = mk(h2o=100.0, nacl=10.0)
    rem = float(W.water_for_aw(c, 0.80, 25.0))
    after = c.copy()
    after[0] -= rem
    assert float(W.aw(after, 25.0)) == pytest.approx(0.80, abs=1e-9)
    rem = float(W.water_for_boiling_rise(c, 5.0))
    after = c.copy()
    after[0] -= rem
    assert float(W.boiling_rise(after)) == pytest.approx(5.0, abs=1e-6)
    assert rem > 0.0 and float(W.water_for_aw(c, 0.99, 25.0)) < 0.0     # a dilute target means adding water


def test_exact_coverage_names_the_model_and_flags_gaps():
    cov = W.coverage(mk(h2o=100.0, nacl=5.0, urea=3.0, ethanol=2.0))
    assert cov["models"] == {"nacl": "pitzer", "urea": "norrish", "ethanol": "vanlaar"}
    assert any("Van Laar" in f for f in cov["flags"]) and any("several solutes" in f for f in cov["flags"])
    assert any("ideal" in f for f in W.coverage(mk(h2o=100.0, glycerol=5.0, h2so4=1.0))["flags"])
    assert any("extrapolated" in f for f in W.coverage(mk(h2o=100.0, ethanol=5.0), T=5.0)["flags"])
