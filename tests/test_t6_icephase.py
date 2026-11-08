"""T6 ice as a phase of water (wave 13, lane A-T6). Tags as in test_t6_psychro.py."""
import numpy as np
import pytest

from src.voxel import _standin as S
from src.voxel import icephase as IP
from src.voxel import matterdata as D
from src.voxel import psychro as P

CW, CI, L = D.CW, D.CI, D.L_CONV20


def _settle(E, Co, mw, mi, s, x, nuc=0.0, dt=np.inf, liq=IP.NACL):
    return IP.settle(E, Co, mw, mi, s, x, nuc, dt, liq)


def _E(T, Co, mw, mi):
    return (Co + CW * mw + CI * mi) * (T - 20.0)


def test_ice_row_and_the_conversion_latent():
    assert IP.ICE_ROW["rho"] == 917.0 and IP.ICE_ROW["Lf"] == 333.55
    assert IP.ICE_ROW["solid"][0] == 917.0
    assert D.L_CONV20 == pytest.approx(333.55 + 20 * (4.2 - 2.09))
    assert abs(D.L_CONV20 - 375.8) < 0.1
    assert IP.ICE_ROW["expansion"] == pytest.approx(1.0905, abs=1e-3)


def test_enthalpy_is_conserved_through_random_freezes_and_melts():
    rng = np.random.default_rng(11)
    n = 400
    mw = rng.uniform(0, 200, n)
    mi = rng.uniform(0, 200, n)
    s = rng.uniform(0, 30, n) * (rng.random(n) < 0.6)
    x = rng.uniform(0, 10, n) * (rng.random(n) < 0.3)
    Co = rng.uniform(1, 40, n)
    T = rng.uniform(-30, 8, n)
    E = _E(T, Co, mw, mi)
    for liq in (IP.NACL, IP.PURE, IP.SUCROSE):
        r = _settle(E, Co, mw, mi, s, x, 0.0, np.inf, liq)
        H0 = IP.enthalpy(E, mi)
        H1 = IP.enthalpy(r.E, r.m_i)
        assert np.abs(H1 - H0).max() < 1e-9
        assert np.abs((r.m_w + r.m_i) - (mw + mi)).max() < 1e-12
        assert np.all(r.m_w >= -1e-12) and np.all(r.m_i >= -1e-12)
        assert np.abs((r.m_i - mi) - r.frozen).max() < 1e-12
        # crystals plus dissolved stay equal to the solute there is (nothing lost)
        assert np.all(r.x >= -1e-9) and np.all(r.x <= s + x + 1e-9)
        # settled cells are fixed points
        r2 = _settle(r.E, Co, r.m_w, r.m_i, s + x - r.x, r.x, 0.0, np.inf, liq)
        assert np.abs(r2.m_i - r.m_i).max() < 1e-7


def test_pure_water_freezes_to_the_pin_with_the_right_amount():
    Co, mw = 10.0, 100.0
    for T0 in (-0.5, -5.0, -30.0):
        C = Co + CW * mw
        r = _settle(C * (T0 - 20), Co, mw, 0.0, 0.0, 0.0, 0.0, np.inf, IP.PURE)
        assert float(r.m_i) == pytest.approx(-C * T0 / 333.55, rel=1e-12)
        assert abs(float(r.T)) < 1e-9
        assert float(IP.melt_extent_pure(C, T0)) == pytest.approx(float(r.m_i), rel=1e-12)
    # supercooled so deep the whole cell freezes: T falls below the pin
    r = _settle(_E(-200.0, 1.0, 5.0, 0.0), 1.0, 5.0, 0.0, 0.0, 0.0, 0.0, np.inf, IP.PURE)
    assert float(r.m_i) == pytest.approx(5.0, abs=1e-9) and float(r.T) < 0


def test_pure_water_cooling_curve_has_a_plateau():
    Co, mw = 5.0, 50.0
    E = _E(5.0, Co, mw, 0.0)
    mi, mw_, temps, ices = 0.0, mw, [], []
    dE = -100.0
    for _ in range(500):
        E += dE
        r = _settle(E, Co, mw_, mi, 0.0, 0.0, 0.0, np.inf, IP.PURE)
        E, mw_, mi = float(r.E), float(r.m_w), float(r.m_i)
        temps.append(float(r.T))
        ices.append(mi)
    t = np.array(temps)
    plateau = np.abs(t) < 1e-9
    assert plateau.sum() > 20                      # a long stretch exactly at 0 C
    assert ices[-1] == pytest.approx(mw, abs=1e-9)  # then all ice and still cooling
    assert t[-1] < -10
    assert np.all(np.diff(t[~plateau & (t < 0)]) <= 1e-9)


def test_u_brine_3p5_starts_freezing_at_minus_2p1_and_stops_at_the_eutectic():
    S_tot, Mw = 3.5, 96.5
    Co = 0.85 * S_tot
    lo, hi = -2.0, -2.3
    # first temperature (cooling) at which ice forms with nucleation 0
    first = None
    for T in np.arange(-1.5, -3.0, -0.01):
        r = _settle(_E(T, Co, Mw, 0.0), Co, Mw, 0.0, S_tot, 0.0)
        if float(r.m_i) > 0:
            first = T
            break
    assert first is not None and abs(first + 2.1) < 0.2
    # cool it to the end by removing heat
    E = _E(-1.0, Co, Mw, 0.0)
    mw, mi, x = Mw, 0.0, 0.0
    T_hist, wt_hist, states = [], [], []
    for _ in range(3000):
        E -= 30.0
        r = _settle(E, Co, mw, mi, S_tot - x, x)
        E, mw, mi, x = float(r.E), float(r.m_w), float(r.m_i), float(r.x)
        T_hist.append(float(r.T))
        d = S_tot - x
        wt_hist.append(100 * d / (d + mw) if mw > 1e-9 else np.nan)
        states.append((mw, d, x))
    T_hist = np.array(T_hist)
    # brine concentration rises as ice forms (until the eutectic)
    w = np.array(wt_hist)
    w_ok = w[~np.isnan(w)]
    assert np.all(np.diff(w_ok) >= -1e-9)
    # T stops at the eutectic, crystals appear, the liquid keeps 23.3 wt %
    plateau = np.where(np.abs(T_hist - S.EUTECTIC_NACL[0]) < 1e-9)[0]
    assert len(plateau) > 10
    assert abs(T_hist[plateau[0]] + 21.6) < 0.6
    assert abs(w[plateau[0]] - 23.3) < 0.6
    assert x > 0 and mi > 0
    # at the plateau the liquid is exactly at b_e and dissolved + crystals = all the salt
    mw_p, d_p, x_p = states[plateau[0]]
    assert d_p / mw_p == pytest.approx(S.EUTECTIC_NACL[1], rel=1e-6)
    assert d_p + x_p == pytest.approx(S_tot, abs=1e-12)


def test_u_salt_on_ice_100g_plus_10g_ends_near_minus_21_with_about_84g_ice():
    """T6's settle() leaves out the heat of solution of the salt (T4's dissolved_J book has it) and
    puts the eutectic on the hydrohalite line (-21.58 C): 83.7 g of ice here. T4's phase_step books
    the heat of solution and leaves 85.1 g (N3, 2026-10-30: the plan accepts 85 +- 1 g; the old
    plan number was 82.7). The window below holds both."""
    Co = 0.85 * 10.0
    E = _E(0.0, Co, 0.0, 100.0)
    r = _settle(E, Co, 0.0, 100.0, 0.0, 10.0)
    assert abs(float(r.T) + 21.0) < 1.0
    assert 81.0 < float(r.m_i) < 86.0
    assert float(r.x) > 0                                   # crystals left over
    # at -5 C the ice melts to the liquidus and the cell cools
    r5 = _settle(_E(-5.0, Co, 0.0, 100.0), Co, 0.0, 100.0, 0.0, 10.0)
    assert float(r5.m_w) > 5 and float(r5.T) < -5
    # at -25 C (below the eutectic) nothing melts
    r25 = _settle(_E(-25.0, Co, 0.0, 100.0), Co, 0.0, 100.0, 0.0, 10.0)
    assert float(r25.m_i) == pytest.approx(100.0, abs=1e-9)
    assert float(r25.T) == pytest.approx(-25.0, abs=1e-9)


def test_hydrohalite_cool_to_minus_25_and_back_gives_the_same_totals():
    Co, Mw, Stot = 0.85 * 10.0, 90.0, 10.0
    E0 = _E(0.0, Co, Mw, 0.0)
    s0 = _settle(E0, Co, Mw, 0.0, Stot, 0.0)
    assert float(s0.m_i) == 0.0                              # liquid at 0 C, 10 wt %
    E, mw, mi, x = E0, Mw, 0.0, 0.0
    for _ in range(400):
        r = _settle(E - 300.0, Co, mw, mi, Stot - x, x)
        E, mw, mi, x = float(r.E), float(r.m_w), float(r.m_i), float(r.x)
        if float(r.T) <= -25.0:
            break
    assert float(r.T) <= -25.0 and x > 0
    H_cold = E - L * mi
    # now give the same heat back
    H0 = E0
    gain = H0 - H_cold
    E = E + gain
    r2 = _settle(E, Co, mw, mi, Stot - x, x)
    assert float(r2.m_i) == pytest.approx(0.0, abs=1e-8)
    assert float(r2.x) == pytest.approx(0.0, abs=1e-8)
    assert float(r2.T) == pytest.approx(0.0, abs=1e-8)
    assert float(r2.m_w) == pytest.approx(Mw, abs=1e-8)
    assert float(r2.E) == pytest.approx(E0, abs=1e-8)


def test_u_liquidus_and_solubility_cross_at_the_eutectic():
    T_e, b_e = S.EUTECTIC_NACL
    assert abs(T_e + 21.1) < 0.6
    assert float(S.freeze_point("NaCl", b_e)) == pytest.approx(T_e, abs=1e-6)
    assert float(S.solubility_load("NaCl", T_e)) == pytest.approx(b_e, abs=1e-6)


def test_brine_rejection_the_freezing_point_falls_as_ice_forms():
    Co, Mw, Stot = 0.85 * 10.0, 90.0, 10.0
    last = None
    for T0 in (-8.0, -10.0, -14.0, -18.0):
        r = _settle(_E(T0, Co, Mw, 0.0), Co, Mw, 0.0, Stot, 0.0)
        assert float(r.T) == pytest.approx(float(S.freeze_point("NaCl", Stot / float(r.m_w))), abs=1e-7)
        if last is not None:
            assert float(r.T) < last
        last = float(r.T)


def test_u_unseeded_clean_water_supercools_2K_seeded_freezes_at_once():
    Co, mw = 2.0, 40.0
    E = _E(-1.5, Co, mw, 0.0)
    clean = IP.nucleation_k()
    assert float(clean) == 2.0
    r = _settle(E, Co, mw, 0.0, 0.0, 0.0, clean, np.inf, IP.PURE)
    assert float(r.m_i) == 0.0                                # still liquid at -1.5
    E2 = _E(-2.5, Co, mw, 0.0)
    r = _settle(E2, Co, mw, 0.0, 0.0, 0.0, clean, np.inf, IP.PURE)
    assert float(r.m_i) > 0 and abs(float(r.T)) < 1e-9        # recalescence to the pin
    for kw in (dict(ice_in_cell=True), dict(ice_in_neighbour=True), dict(cold_wall=True),
               dict(free_surface=True)):
        k = IP.nucleation_k(**kw)
        assert float(k) == 0.0
        rr = _settle(E, Co, mw, 0.0, 0.0, 0.0, k, np.inf, IP.PURE)
        assert float(rr.m_i) > 0
    # with ice already in the cell it grows at once even with a 2 K clean-water number
    Ei = _E(-0.5, Co, mw - 1.0, 1.0)
    ri = _settle(Ei, Co, mw - 1.0, 1.0, 0.0, 0.0, clean, np.inf, IP.PURE)
    assert float(ri.m_i) > 1.0


def test_melting_relaxes_with_tau_and_never_overshoots():
    Co, mw, mi = 2.0, 0.0, 30.0
    E = _E(5.0, Co, mw, mi)
    full = _settle(E, Co, mw, mi, 0.0, 0.0, 0.0, np.inf, IP.PURE)
    half = _settle(E, Co, mw, mi, 0.0, 0.0, 0.0, 600.0 * np.log(2), IP.PURE)
    melted_full = mi - float(full.m_i)
    assert (mi - float(half.m_i)) == pytest.approx(0.5 * melted_full, rel=1e-9)
    assert abs(IP.enthalpy(half.E, half.m_i) - IP.enthalpy(E, mi)) < 1e-9


def test_expansion_1000g_needs_1p09_times_the_volume():
    assert 1000.0 / 917.0 == pytest.approx(1.0905, abs=1e-3)
    f, e, pend = IP.expand_on_freeze(1000.0, 1000.0, 1000.0)
    assert float(f) == pytest.approx(917.0, abs=1e-9)
    assert float(e) == pytest.approx(83.0, abs=1e-9)
    assert float(f) + float(e) == pytest.approx(1000.0, abs=1e-12)     # mass exact
    assert float(pend) == 0.0
    # a half-empty cell takes the expansion without expelling anything
    f, e, pend = IP.expand_on_freeze(500.0, 500.0, 1000.0)
    assert float(e) == 0.0 and float(f) == 500.0
    # a mostly full cell freezing part of its water expels the overflow only
    f, e, pend = IP.expand_on_freeze(990.0, 400.0, 1000.0)
    assert float(f) == 400.0
    assert float(e) == pytest.approx(400.0 / 0.917 + 590.0 - 1000.0, abs=1e-9)
    assert float(f) + float(e) < 990.0 + 1e-9
    # nowhere to go: a sealed jar stays partly liquid and the shortfall is pending
    f, e, pend = IP.expand_on_freeze(1000.0, 1000.0, 1000.0, room_ml=0.0)
    assert float(e) == 0.0 and float(f) < 1000.0
    assert float(pend) == pytest.approx(83.0, abs=1e-9)
    # level rise of a pond: 9 % of the ice depth
    assert float(IP.ice_rise_ratio()) == pytest.approx(0.0905, abs=1e-3)


def test_30_ml_cell_becomes_mat_ice_only_past_half_with_hysteresis():
    vox_l = 0.030                                            # a 30 ml cell
    half_g = 0.5 * 30.0 * 0.917
    was = False
    states = []
    for g in np.linspace(0.0, 30.0 * 0.917, 200):
        was = bool(IP.promote(g, vox_l, was))
        states.append(was)
    first = int(np.argmax(states))
    g_first = np.linspace(0.0, 30.0 * 0.917, 200)[first]
    assert g_first >= half_g - 1e-9 and states[-1]
    # no flicker for +/-1e-3 noise around the promote threshold
    rng = np.random.default_rng(0)
    was = False
    flips = 0
    for _ in range(500):
        g = half_g * (1 + 1e-3 * rng.standard_normal())
        new = bool(IP.promote(g, vox_l, was))
        flips += new != was
        was = new
    assert flips <= 1
    # once ICE it stays ICE above 0.3 and reverts below
    assert bool(IP.promote(0.35 * 30 * 0.917, vox_l, True))
    assert not bool(IP.promote(0.29 * 30 * 0.917, vox_l, True))
    assert not bool(IP.promote(0.35 * 30 * 0.917, vox_l, False))
    # a 0.1 mm window film is nowhere near
    assert not bool(IP.promote(0.0001 * 100 * 0.917, 0.125, False))
    # nothing between the thresholds flips when noise is added to a settled ICE cell
    g = 0.4 * 30.0 * 0.917
    assert all(bool(IP.promote(g * (1 + 1e-3 * e), vox_l, True)) for e in (-1, 1))


def test_stefan_against_the_formula_and_the_day_week_figures():
    dT, t = 10.0, 86400.0
    h = float(IP.stefan(dT, t))
    assert h == pytest.approx(np.sqrt(2 * 2.2 * dT * t / (917.0 * 333.55e3)), rel=1e-12)
    assert 0.10 < h < 0.12                                   # 11 cm in a day
    assert 0.28 < float(IP.stefan(dT, 7 * t)) < 0.32         # 30 cm in a week
    # thickness goes as sqrt(t)
    assert float(IP.stefan(dT, 4 * t)) == pytest.approx(2 * h, rel=1e-12)
    # an air film only slows it, and the film form solves its own quadratic
    ha = 20.0
    hf = float(IP.stefan_film(dT, t, ha))
    assert hf < h
    assert hf * hf / (2 * 2.2) + hf / ha == pytest.approx(dT * t / (917.0 * 333.55e3), rel=1e-9)
    assert float(IP.stefan_film(dT, t, 1e9)) == pytest.approx(h, rel=1e-6)


def test_water_ice_vapour_sum_is_constant_over_a_random_cycle():
    rng = np.random.default_rng(4)
    mw, mi, mv = 80.0, 10.0, 0.0
    total = mw + mi + mv
    Co = 3.0
    E = _E(-1.0, Co, mw, mi)
    for _ in range(200):
        # freeze / melt
        r = _settle(E + rng.uniform(-200, 200), Co, mw, mi, 0.0, 0.0, 0.0, np.inf, IP.PURE)
        E, mw, mi = float(r.E), float(r.m_w), float(r.m_i)
        # evaporate / condense with the books
        g = rng.uniform(-0.3, 0.3)
        g = min(g, mw) if g > 0 else max(g, -mv)
        cell, gas, st = P.evap_energy(float(IP_T(E, Co, mw, mi)))
        E = E - g * cell + g * gas
        mw, mv = mw - g, mv + g
        assert mw + mi + mv == pytest.approx(total, abs=1e-9)


def IP_T(E, Co, mw, mi):
    return 20.0 + E / (Co + CW * mw + CI * mi)


def test_standin_vs_t4_phase_step_agree():
    mp = pytest.importorskip("src.voxel.mixphase")
    if not hasattr(mp, "phase_step"):
        pytest.skip("mixphase.phase_step not there yet")
    pytest.skip("T4 phase_step takes T4 comp arrays; the adapter is wired in phase B (B3)")
