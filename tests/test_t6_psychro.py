"""T6 psychrometry and the stand-in (wave 13, lane A-T6).

Tags (RULES / plan 9): a line about physical data is test_u_* ([U]: independent but
source unverified); test_echo_* = the constant was fitted to that number; test_codeonly_*
= code against its own table. Conservation, exactness, closed-form and arithmetic
identity tests carry no prefix and gate scenes. A [U] test says which sourced-number
status it saw (tests/data/sourced_numbers.json, lane A-S).
"""
import json
import os

import numpy as np
import pytest

from src.voxel import _standin as S
from src.voxel import matterdata as D
from src.voxel import psychro as P

HERE = os.path.dirname(os.path.abspath(__file__))


def _sourced_status():
    p = os.path.join(HERE, "data", "sourced_numbers.json")
    return "sourced_numbers.json present (format not read here)" if os.path.exists(p) \
        else "sourced_numbers.json absent: numbers stay (u)"


# ---- [U] physical data ---------------------------------------------------------
IAPWS = {0.01: 611.657, 20.0: 2339.2, 50.0: 12352.0, 80.0: 47416.0, 100.0: 101418.0,
         120.0: 198670.0, 150.0: 476160.0}          # Pa, IF97 region 4 (u: from memory)


def test_u_psat_water_within_015_percent_of_iapws():
    for T, ref in IAPWS.items():
        got = float(P.p_sat_water(T))
        assert abs(got / ref - 1) < 0.0015, (T, got, ref, _sourced_status())


def test_u_rho_sat_20_and_100():
    assert abs(float(P.rho_sat(20.0)) - 17.3) < 0.1
    assert abs(float(P.rho_sat(100.0)) - 589.0) < 3.0
    assert abs(float(P.rho_sat(25.0)) - 23.04) < 0.15


STEAM = {0: 2501.0, 20: 2454.0, 50: 2382.0, 100: 2257.0, 120: 2202.0, 150: 2113.0}   # J/g (u)


def test_u_h_fg_against_steam_table():
    for T, ref in STEAM.items():
        tol = 0.008 if T <= 100 else 0.02
        assert abs(float(P.h_fg(T)) / ref - 1) < tol, (T, _sourced_status())


def test_u_wet_bulb_chart():
    assert abs(float(P.wet_bulb(25.0, 0.5)) - 17.9) < 0.5
    assert abs(float(P.wet_bulb(20.0, 0.5)) - 13.8) < 0.5


def test_wet_bulb_is_independent_of_h_and_of_the_stefan_factor():
    a = float(P.wet_bulb(25.0, 0.5, h_conv=5.0))
    b = float(P.wet_bulb(25.0, 0.5, h_conv=40.0))
    assert abs(a - b) < 1e-9
    assert abs(a - float(P.wet_bulb(25.0, 0.5, stefan=False))) < 0.1
    assert float(P.wet_bulb(30.0, 1.0)) == pytest.approx(30.0, abs=1e-6)


def test_u_gab_wood_anchors():
    anchors = {0.30: 6.3, 0.50: 9.2, 0.65: 12.0, 0.80: 16.0, 0.90: 21.0, 0.95: 24.0}
    for a, w in anchors.items():
        assert abs(float(P.gab_w(a)) * 100 - w) < 0.8, (a, _sourced_status())
    aw = np.linspace(0.01, 0.99, 99)
    assert np.all(np.diff(P.gab_w(aw)) > 0)
    assert float(P.gab_w(0.999)) < 0.30


def test_u_standin_water_activity_and_boiling_freezing():
    assert abs(float(S.aw("NaCl", 0.36)) - 0.753) < 0.01
    assert abs(float(S.boil_point(101325.0, "sucrose", 65 / 35)) - 104.1) < 0.5
    for wt, ref in ((5, -3.05), (10, -6.56), (20, -16.46)):
        assert abs(float(S.freeze_point("NaCl", wt / (100 - wt))) - ref) < 0.6, wt
    assert abs(float(S.freeze_point("NaCl", 3.5 / 96.5)) + 2.1) < 0.2
    assert abs(float(S.boil_point(101325.0, "NaCl", 0.36)) - 108.1) < 1.0
    assert abs(float(S.boil_point(70e3)) - 90.0) < 0.5
    assert abs(float(S.boil_point(200e3)) - 120.2) < 0.5


def test_u_standin_vle_ethanol_water():
    x_az, T_az = S.azeotrope()
    assert abs(x_az - 0.894) < 0.025 and abs(T_az - 78.15) < 0.5
    for x, y, T in D.VLE_TABLE:
        Tm, ym = S.vle_ethanol(x)
        assert abs(float(Tm) - T) < 1.0 and abs(float(ym) - y) < 0.02, (x, Tm, ym)
    x = S.mole_fraction_ethanol(8.0, 92.0)             # 10 % ABV wash
    T, y = S.vle_ethanol(x)
    assert 92.0 < float(T) < 95.0
    assert 0.45 < float(S.abv_vol(y)) < 0.53           # volume figure: ~52 %


def test_u_eutectic_falls_out_of_two_curves():
    T_e, b_e = S.EUTECTIC_NACL
    assert abs(T_e + 21.6) < 0.6
    assert abs(100 * b_e / (1 + b_e) - 23.3) < 0.6


def test_u_ice_vapour_pressure():
    assert abs(float(P.p_sat_ice(0.01)) / float(P.p_sat_water(0.01)) - 1) < 0.005
    for T in (-1.0, -10.0, -30.0):
        assert float(P.p_sat_ice(T)) < float(P.p_sat_water(T))


# ---- exactness: the books ---------------------------------------------------------
def test_evaporation_books_close_to_1e9():
    rng = np.random.default_rng(1)
    Ec, Eg, St = 5000.0, 30.0, 0.0
    tot0 = Ec + Eg + St
    for _ in range(500):
        g = rng.uniform(-0.5, 0.5)
        Tv = rng.uniform(0.0, 100.0)
        Ec, Eg, St = P.phase_transfer(Ec, Eg, St, g, Tv)
    assert abs((Ec + Eg + St) - tot0) < 1e-9


def test_evaporate_then_condense_returns_every_number():
    a = (123.4, 5.6, 7.8)
    b = P.phase_transfer(*a, 0.37, 62.0)
    c = P.phase_transfer(*b, -0.37, 62.0)
    assert np.allclose(c, a, atol=1e-12, rtol=0)


def test_cell_and_gas_split_equals_kirchhoff_latent_plus_sensible():
    cell, gas, st = P.evap_energy(55.0)
    assert cell - gas == pytest.approx(D.L0)
    assert cell == pytest.approx(float(P.h_fg(55.0)) + D.CW * 35.0)


def test_boil_at_plus_100_K_ends_at_Tb_to_float_precision():
    for Tb in (100.0, 108.1, 92.8):
        C = 1000.0 + 4.2 * 300.0
        E = C * (Tb + 100.0 - 20.0)
        g, E2, C2, Eg, st = P.boil_step(E, C, 300.0, Tb)
        T2 = 20.0 + E2 / C2
        assert abs(T2 - Tb) < 1e-9, (Tb, T2)
        # books: cell + gas + stored = before
        assert abs((E2 + Eg + st) - E) < 1e-9
        assert 0 < g < 300


def test_boil_is_water_limited():
    g, E2, C2, *_ = P.boil_step(1e6, 10.0, 0.5, 100.0)
    assert g == pytest.approx(0.5)


# ---- the surface law ---------------------------------------------------------------
def test_stefan_factor_values():
    for x, ref in ((0.05, 1.03), (0.3, 1.19), (0.7, 1.72), (0.9, 2.56)):
        assert abs(float(P.stefan_factor(x)) - ref) < 0.01
    assert float(P.stefan_factor(0.0)) == 1.0
    assert float(P.stefan_factor(0.99)) == pytest.approx(float(P.stefan_factor(0.9)))


def test_surface_flux_zero_at_equilibrium_and_flips_at_dew_point():
    Ts = 20.0
    rho_eq = float(P.rho_sat(Ts))
    assert abs(float(P.surface_flux(10.0, Ts, 1.0, rho_eq))) < 1e-12
    assert float(P.surface_flux(10.0, Ts, 1.0, 0.5 * rho_eq)) > 0
    assert float(P.surface_flux(10.0, Ts, 1.0, 1.5 * rho_eq)) < 0       # condensation
    f1 = float(P.surface_flux(10.0, Ts, 1.0, 0.5 * rho_eq))
    f2 = float(P.surface_flux(20.0, Ts, 1.0, 0.5 * rho_eq))
    assert f2 / f1 == pytest.approx(2.0, rel=1e-9)                       # scales with h
    rho_air = 9.0
    Td = float(P.dew_point(rho_air))
    assert float(P.surface_flux(10.0, Td + 0.5, 1.0, rho_air)) > 0
    assert float(P.surface_flux(10.0, Td - 0.5, 1.0, rho_air)) < 0


def test_dry_skin_resistance_slows_the_flux():
    base = float(P.surface_flux(10.0, 20.0, 1.0, 5.0))
    dry = float(P.surface_flux(10.0, 20.0, 1.0, 5.0, R_int=23000.0))
    assert dry < 0.01 * base


def test_dew_point_inverts_rho_sat():
    for T in (-10.0, 0.0, 12.3, 55.0, 99.0):
        assert float(P.dew_point(float(P.rho_sat(T)))) == pytest.approx(T, abs=1e-5)


def test_exponential_surface_step_never_overshoots():
    V, A = 0.125e-3, 0.0025
    rho_eq, rho_air = 17.29, 6.0
    full = (rho_eq - rho_air) * V
    for h in (0.1, 1.0, 10.0, 100.0):
        for R_int in (0.0, 120.0, 23000.0):
            for dt in (1.0, 10.0, 600.0):
                m = float(P.surface_step(rho_eq, rho_air, V, A, h, 1.0, R_int, dt))
                assert 0 <= m <= full * (1 + 1e-12)
                m2 = float(P.surface_step(rho_air, rho_eq, V, A, h, 1.0, R_int, dt))
                assert -full * (1 + 1e-12) <= m2 <= 0
    # tau from 0.5 to 600 s at dt = 1: reachable by choosing h; never over
    for tau in (0.5, 6.0, 60.0, 600.0):
        h = V * 1200.0 / (A * tau)                  # 1/h_m = tau A / V
        m = float(P.surface_step(rho_eq, rho_air, V, A, h, 1.0, 0.0, 1.0))
        assert abs(m / full - (1 - np.exp(-1.0 / tau))) < 1e-9


def test_five_cm_voxel_time_constant_is_about_six_seconds():
    V, A = 0.125e-3, 0.0025
    tau = V / (A / (1200.0 / 10.0))
    assert 5.0 < tau < 7.0


# ---- mist ---------------------------------------------------------------------------
def test_mist_saturation_adjustment_150_percent():
    V = 0.125e-3
    Cair = 1.2 * 0.125
    m0 = float(P.rho_sat(20.0)) * V * 1.5
    mc, T1 = P.saturation_adjust(m0, 20.0, V, Cair)
    assert abs(float(T1) - 25.2) < 0.3
    assert abs(float(mc) * 1e3 - 0.33) < 0.03                    # mg
    # ends saturated, books close
    left = m0 - float(mc)
    assert left == pytest.approx(float(P.rho_sat(float(T1))) * V, rel=1e-4)
    E_in = D.L0 * float(mc)                                       # gas E gained
    heat = (Cair + D.CV * left + D.CW * float(mc)) * (float(T1) - 20.0)
    assert heat == pytest.approx(E_in, rel=1e-6)


def test_mist_adjustment_leaves_unsaturated_voxels_alone():
    V = 0.125e-3
    mc, T1 = P.saturation_adjust(0.5 * float(P.rho_sat(20.0)) * V, 20.0, V, 0.15)
    assert float(mc) == 0.0 and float(T1) == pytest.approx(20.0)


def test_steam_into_a_small_voxel_stays_in_the_mixing_bounds():
    T = 20.0
    for _ in range(50):
        T2 = float(P.mix_steam(T, 0.15, 0.0, 0.022, 100.0))
        assert T <= T2 <= 100.0
        T = T2
    assert T < 100.0


def test_mist_extinction_and_evaporation():
    K = float(P.mist_extinction(0.1))
    assert K == pytest.approx(0.03)
    assert 100 < 3.9 / K < 160
    m = float(P.mist_evaporate(1e-3, 0.0, 20.0, 0.125e-3, 1.0))
    assert 0 < m <= 1e-3
    assert float(P.mist_evaporate(1e-9, 0.0, 20.0, 0.125e-3, 1e6)) == pytest.approx(1e-9)


# ---- closed forms -------------------------------------------------------------------
def test_gab_inverse_roundtrip():
    a = np.linspace(0.02, 0.97, 40)
    assert np.allclose(P.gab_aw(P.gab_w(a)), a, atol=1e-12)
    assert float(P.gab_aw(0.0)) == 0.0


def test_gab_derivative_matches_difference():
    a, h = 0.55, 1e-6
    num = (float(P.gab_w(a + h)) - float(P.gab_w(a - h))) / (2 * h)
    assert float(P.gab_dwda(a)) == pytest.approx(num, rel=1e-6)


def test_crank_slab_starts_at_c0_and_ends_at_c1():
    x = np.linspace(-0.1, 0.1, 9)
    assert np.allclose(P.crank_slab(x[1:-1], 10.0, 1e-6, 0.1, 1.0, 0.0), 1.0, atol=1e-3)
    assert np.allclose(P.crank_slab(x, 1e6, 1e-6, 0.1, 1.0, 0.0), 0.0, atol=1e-9)
    # the mean is the integral of the profile
    xx = np.linspace(-0.1, 0.1, 4001)
    prof = P.crank_slab(xx, 400.0, 1e-6, 0.1, 1.0, 0.0)
    assert float(P.crank_slab_mean(400.0, 1e-6, 0.1, 1.0, 0.0)) == pytest.approx(
        float(np.trapezoid(prof, xx)) / 0.2, abs=1e-4)


def test_fick_erf_limits():
    assert float(P.fick_erf(0.0, 100.0, 1e-6, 1.0, 0.0)) == pytest.approx(0.0)
    assert float(P.fick_erf(10.0, 100.0, 1e-6, 1.0, 0.0)) == pytest.approx(1.0)
    assert float(P.fick_erf(0.01, 100.0, 1e-6, 1.0, 0.0)) == pytest.approx(0.5205, abs=1e-3)


# ---- provider ------------------------------------------------------------------------
def test_codeonly_provider_is_named():
    assert P.PROVIDER in ("standin", "mixwater")


def test_standin_vs_mixwater_cross():
    mw = pytest.importorskip("src.voxel.mixwater")
    if not hasattr(mw, "psat_water"):
        pytest.skip("mixwater has no psat_water yet")
    for T in (0.01, 20.0, 60.0, 100.0, 150.0):
        a, b = float(S.psat_water(T)), float(mw.psat_water(T))
        assert abs(a / b - 1) < 1e-3, (T, a, b)
    if hasattr(mw, "psat_ice"):
        for T in (-30.0, -5.0):
            a, b = float(S.psat_ice(T)), float(mw.psat_ice(T))
            assert abs(a / b - 1) < 2e-3
    if hasattr(mw, "Lfus"):
        assert float(S.lfus(0.0)) == pytest.approx(float(mw.Lfus(0.0)), rel=1e-3)


def test_saturation_adjust_never_makes_nan_and_converges_for_any_surplus():
    V = 0.125e-3
    for T0 in (-30.0, -10.0, 0.0, 20.0, 30.0, 60.0, 80.0, 95.0, 120.0):
        for k in (1.0001, 1.1, 2.0, 5.0, 10.0, 20.0, 34.0, 100.0, 1e3, 1e5):
            for C in (0.01, 0.15, 5.0):
                m0 = k * float(P.rho_sat(T0)) * V
                mc, T1 = P.saturation_adjust(m0, T0, V, C)
                mc, T1 = float(mc), float(T1)
                assert np.isfinite(mc) and np.isfinite(T1), (T0, k, C)
                assert 0.0 <= mc <= m0
                if T1 < 370.0:                              # inside the vapour table
                    left = m0 - mc
                    assert abs(left - float(P.rho_sat(T1)) * V) < 1e-6 * m0, (T0, k, C)


def test_steam_grams_into_a_small_voxel_make_finite_mist():
    V, C = 0.125e-3, 0.15
    for g in (0.5, 1.0, 2.0, 5.0):
        T = float(P.mix_steam(20.0, C, 0.0, g, 100.0))
        mc, T1 = P.saturation_adjust(g, T, V, C)
        assert np.isfinite(float(mc)) and np.isfinite(float(T1))
        assert 0.0 <= float(mc) <= g


def test_saturation_adjust_arrays_and_unsaturated_cells_mix():
    V = 0.125e-3
    m = np.array([0.0, 1e-3, 2.0])
    T = np.array([20.0, 20.0, 95.0])
    mc, T1 = P.saturation_adjust(m, T, V, 0.15)
    assert mc[0] == 0.0 and mc[1] == 0.0 and mc[2] > 0.0
    assert np.all(np.isfinite(T1)) and T1[1] == pytest.approx(20.0)
