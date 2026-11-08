"""The closed-form library of the slow-time scheduler (wave 13, T5, spec 12 T1-T14c).

Tags (RULES.md, acceptance tags): plain test names are conservation, exactness,
closed-form and arithmetic-identity lines and may gate a scene. test_u_* are
physical data that is independent but whose source is unverified until
tests/data/sourced_numbers.json marks it sourced (the test says which status it
saw). test_echo_* and test_codeonly_* never gate a scene.
"""
import json
import math
import os

import numpy as np
import pytest

from src.voxel import slowforms as F
from src.voxel import slowtwins as TW
from src.voxel.physiology import BREATH

T = F.TICK_S
HOUR = 144000
DAY = 24 * HOUR
FORMS = F.FORMS
_HERE = os.path.dirname(os.path.abspath(__file__))


def _sourced(name):
    """'sourced', 'unsourced' or 'no file': what tests/data/sourced_numbers.json
    says about a (u) number. The test passes either way and reports it."""
    try:
        with open(os.path.join(_HERE, "data", "sourced_numbers.json")) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return "no file"
    row = d.get(name) if isinstance(d, dict) else None
    return "sourced" if row else "unsourced"


def _euler_cohb(c0, ticks):
    c = c0
    dt = T
    for _ in range(ticks):
        out = c * math.log(2.0) / (BREATH["co_half_min"] * 60.0) * dt
        c = min(max(c + 0.0 - out, 0.0), 1.0)
    return c


# ───────────────────────────────────────────────────── T1 decay ──
@pytest.mark.parametrize("c0,want", [(0.35, 0.2369947), (0.30, 0.2031383)])
def test_cohb_continuous_closed_form_agrees_with_432000_euler_ticks(c0, want):
    ticks = 3 * HOUR
    assert ticks == 432000
    euler = _euler_cohb(c0, ticks)
    (cont,), _ = FORMS["decay"].advance((c0,), {"half_s": BREATH["co_half_min"] * 60.0},
                                        {}, 0, ticks)
    assert abs(cont - want) < 2e-7 and abs(euler - want) < 2e-7
    assert abs(cont / euler - 1.0) < 2e-7


@pytest.mark.parametrize("c0", [0.35, 0.30])
def test_cohb_discrete_closed_form_is_the_tick_laws_own_factor(c0):
    """The twin's factor (1 - k)^n against the same Euler loop: 1e-11 (1.8e-12 measured)."""
    ticks = 3 * HOUR
    euler = _euler_cohb(c0, ticks)
    k = math.log(2.0) * T / (BREATH["co_half_min"] * 60.0)
    disc = TW.geo_decay(c0, k, ticks)
    assert abs(disc / euler - 1.0) < 1e-11
    # and the exp() form is NOT that exact: that is why the twin does not use it
    (cont,), _ = FORMS["decay"].advance((c0,), {"half_s": BREATH["co_half_min"] * 60.0},
                                        {}, 0, ticks)
    assert abs(cont / euler - 1.0) > 1e-8


def test_decay_piecewise_schedule_sums_rates_and_cross_hits_the_level():
    p = {"A": 1e10, "Ea": 80e3, "drive": "T"}
    tab = F.TableDrive([(2 * HOUR, 20.0), (5 * HOUR, 60.0), (9 * HOUR, 5.0)])
    k = F.arrhenius(p)
    kint = (k(20.0) * 2 * HOUR + k(60.0) * 3 * HOUR + k(5.0) * 4 * HOUR) * T
    (x,), fl = FORMS["decay"].advance((1.0,), p, {"T": tab}, 0, 9 * HOUR)
    assert abs(x - math.exp(-kint)) < 1e-15
    # fine quadrature of the same integral (the table drive's pieces are exact)
    fine = sum(k(tab.value(t)) for t in range(0, 9 * HOUR, 3600)) * 3600 * T
    assert abs(fine - kint) / kint < 1e-9
    level = 0.5 * x + 0.5 * math.exp(-kint / 2)
    tc = FORMS["decay"].cross((1.0,), p, {"T": tab}, 0, "x", level, 9 * HOUR)
    a, _ = FORMS["decay"].advance((1.0,), p, {"T": tab}, 0, tc)
    b, _ = FORMS["decay"].advance((1.0,), p, {"T": tab}, 0, tc - 1)
    assert a[0] <= level < b[0]


def test_decay_on_a_sine_drive_is_a_semigroup_to_1e9():
    p = {"A": 1e10, "Ea": 80e3, "drive": "T"}
    d = F.SineDrive(15.0, 10.0, 86400.0)
    one, _ = FORMS["decay"].advance((1.0,), p, {"T": d}, 0, 3 * DAY)
    mid, _ = FORMS["decay"].advance((1.0,), p, {"T": d}, 0, 1 * DAY + 12345)
    two, _ = FORMS["decay"].advance(mid, p, {"T": d}, 1 * DAY + 12345, 3 * DAY)
    assert abs(one[0] / two[0] - 1.0) < 1e-9


# ───────────────────────────────────────────── T3 relax and drain ──
def _rk4_relax(x0, tau, target_fn, t1_s, n=20000):
    h = t1_s / n
    x, t = x0, 0.0
    f = lambda t, x: (target_fn(t) - x) / tau         # noqa: E731
    for _ in range(n):
        k1 = f(t, x)
        k2 = f(t + h / 2, x + h / 2 * k1)
        k3 = f(t + h / 2, x + h / 2 * k2)
        k4 = f(t + h, x + h * k3)
        x += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return x


def test_relax_matches_rk4_for_const_table_and_sine_targets():
    tau = 3600.0
    (c,), _ = FORMS["relax"].advance((10.0,), {"tau_s": tau, "target": 30.0}, {}, 0, 2 * HOUR)
    assert abs(c - _rk4_relax(10.0, tau, lambda t: 30.0, 7200.0)) < 1e-9
    tab = F.TableDrive([(HOUR, 30.0), (2 * HOUR, -5.0)])
    (t_,), _ = FORMS["relax"].advance((10.0,), {"tau_s": tau}, {"target": tab}, 0, 3 * HOUR)
    ref = _rk4_relax(10.0, tau, lambda t: 30.0 if t < 3600 else -5.0, 3 * 3600.0, 60000)
    assert abs(t_ - ref) < 2e-3                         # RK4 straddles the step: 1/60000 s of it
    sine = F.SineDrive(5.0, 15.0, 86400.0, 0.0)
    (s_,), _ = FORMS["relax"].advance((10.0,), {"tau_s": 4 * 3600.0}, {"target": sine}, 0, DAY)
    ref = _rk4_relax(10.0, 4 * 3600.0, lambda t: sine.value(t / T), 86400.0, 60000)
    assert abs(s_ - ref) < 1e-6


def test_relax_is_a_semigroup():
    p = {"tau_s": 5000.0}
    sine = F.SineDrive(5.0, 15.0, 86400.0, 0.0)
    a, _ = FORMS["relax"].advance((1.0,), p, {"target": sine}, 0, 5 * HOUR)
    b, _ = FORMS["relax"].advance(a, p, {"target": sine}, 5 * HOUR, 11 * HOUR + 7)
    c, _ = FORMS["relax"].advance((1.0,), p, {"target": sine}, 0, 11 * HOUR + 7)
    assert abs(b[0] - c[0]) < 1e-9 * max(abs(c[0]), 1.0)


def test_drain_floors_and_its_crossing_is_closed_form():
    p = {"floor": 2.0}
    d = F.TableDrive([(HOUR, 0.001), (2 * HOUR, 0.01)])
    (x,), fl = FORMS["drain"].advance((100.0,), p, {"rate": d}, 0, 2 * HOUR)
    assert abs(x - (100.0 - 0.001 * 3600 - 0.01 * 3600)) < 1e-9
    assert abs(fl[("store", "x", "drained")] - (100.0 - x)) < 1e-12
    (y,), fl = FORMS["drain"].advance((100.0,), p, {"rate": d}, 0, 40 * HOUR)
    assert y == 2.0 and abs(fl[("store", "x", "drained")] - 98.0) < 1e-12
    tc = FORMS["drain"].cross((100.0,), p, {"rate": d}, 0, "x", 90.0, 40 * HOUR)
    a, _ = FORMS["drain"].advance((100.0,), p, {"rate": d}, 0, tc)
    b, _ = FORMS["drain"].advance((100.0,), p, {"rate": d}, 0, tc - 1)
    assert a[0] <= 90.0 < b[0]


# ───────────────────────────────────────────────────── T4 fick ──
D_AUST = 2.3e-5 * math.exp(-148e3 / (F.R_GAS * (925.0 + F.K0)))


def test_halfspace_is_the_erf_and_its_uptake_is_the_profile_integral():
    deff = 1e-7
    xs = np.linspace(0, 20 * math.sqrt(deff), 40001)
    c = np.array([F.halfspace_c(x, deff, 0.2, 1.0) for x in xs])
    assert abs(c[0] - 1.0) < 1e-12 and abs(c[-1] - 0.2) < 1e-12
    up = float(np.trapezoid(c - 0.2, xs))
    assert abs(up / F.halfspace_uptake(deff, 0.2, 1.0) - 1.0) < 1e-6
    # the erf solves the diffusion equation: c_t = D c_xx by finite differences in Deff
    x0, h = 0.5 * math.sqrt(deff), 1e-9
    ct = (F.halfspace_c(x0, deff * (1 + 1e-6), 0.2, 1.0) - F.halfspace_c(x0, deff, 0.2, 1.0)) / (1e-6 * deff)
    cxx = (F.halfspace_c(x0 + h, deff, 0.2, 1.0) - 2 * F.halfspace_c(x0, deff, 0.2, 1.0)
           + F.halfspace_c(x0 - h, deff, 0.2, 1.0)) / h ** 2
    assert abs(ct - cxx) / abs(cxx) < 2e-3                      # c_t (Deff-time) = c_xx


def test_u_constant_D_carburising_depth_is_0_79_mm_and_flagged_low():
    """Dilute-D number (Callister Table 5.2, as recalled (u)): 0.79 mm at 925 C,
    8 h, Cs 1.0, C0 0.2, to 0.4 %C. It undershoots the industrial case depth."""
    print("callister D0, Q:", _sourced("callister_carbon_gamma_D"))
    assert abs(D_AUST - 8.12e-12) / 8.12e-12 < 5e-3
    p = {"geometry": "half", "D0": 2.3e-5, "Q": 148e3, "C0": 0.2, "Cs": 1.0, "A1_C": 727.0}
    st, fl = FORMS["fick"].advance((0.0,), p, {"T": F.ConstDrive(925.0)}, 0, 8 * HOUR)
    depth = FORMS["fick"].observe(st, p, ("depth", 0.4))
    assert abs(depth * 1e3 - 0.787) < 0.003
    assert abs(depth - F.halfspace_depth(0.4, D_AUST * 8 * 3600, 0.2, 1.0)) < 1e-12
    p2 = dict(p, Dratio=1.59)                        # D varies by 59 % with carbon
    ok, until, why = FORMS["fick"].valid(st, p2, {"T": F.ConstDrive(925.0)}, 0, 8 * HOUR)
    assert not ok and "20%" in why


def _wells_fd(hours=8, N=800, L=0.01):
    R = 1.987
    T_ = 925 + 273.15
    Dw = lambda C: (0.07 + 0.06 * C) * np.exp(-32000 / (R * T_)) * 1e-4          # noqa: E731
    x = np.linspace(0, L, N)
    dx = x[1] - x[0]
    C = np.full(N, 0.2)
    C[0] = 1.0
    t, tend = 0.0, hours * 3600
    while t < tend:
        Dm = Dw(0.5 * (C[1:] + C[:-1]))
        dt = min(0.4 * dx * dx / Dm.max(), tend - t)
        fl = Dm * (C[1:] - C[:-1]) / dx
        C[1:-1] += dt * (fl[1:] - fl[:-1]) / dx
        t += dt
    i = np.argmax(C < 0.4)
    return float(np.interp(0.4, C[i - 1:i + 1][::-1], x[i - 1:i + 1][::-1]))


def test_u_wells_similarity_depth_is_1_11_mm_within_3_percent_of_finite_difference():
    print("wells-mehl constants:", _sourced("wells_mehl_carbon_D"))
    assert abs(F.wells_D(0.2, 925) - 1.19e-11) < 3e-13
    assert abs(F.wells_D(1.0, 925) - 1.89e-11) < 4e-13
    p = {"geometry": "wells", "wa": 0.07, "wb": 0.06, "Cs": 1.0, "C0": 0.2,
         "Q_cal": 32000.0, "A1_C": 727.0}
    st, fl = FORMS["fick"].advance((0.0,), p, {"T": F.ConstDrive(925.0)}, 0, 8 * HOUR)
    depth = FORMS["fick"].observe(st, p, ("depth", 0.4))
    fd = _wells_fd()
    assert abs(depth * 1e3 - 1.11) / 1.11 < 0.03
    assert abs(depth / fd - 1.0) < 0.03                  # the similarity table vs the FD run
    const = F.halfspace_depth(0.4, D_AUST * 8 * 3600, 0.2, 1.0)
    assert depth > 1.3 * const                           # NOT the constant-D 0.79 mm
    assert fl[("mass", "c", "face0")] < 0.0              # the host gained carbon


def test_u_industrial_case_depth_table_row_is_a_sanity_band():
    """Harris (as recalled (u)): total case depth in inches = 31.6 sqrt(t_h)/10^(6700/T_R)
    at 925 C, 8 h is 1.8 mm; the 0.4 %C effective depth must sit below it, above half of it."""
    TR = (925 * 9 / 5 + 32) + 459.67
    total = 31.6 * math.sqrt(8) / 10 ** (6700 / TR) * 25.4
    p = {"geometry": "wells", "wa": 0.07, "wb": 0.06, "Cs": 1.0, "C0": 0.2, "Q_cal": 32000.0}
    st, _ = FORMS["fick"].advance((0.0,), p, {"T": F.ConstDrive(925.0)}, 0, 8 * HOUR)
    eff = FORMS["fick"].observe(st, p, ("depth", 0.4)) * 1e3
    assert 0.4 * total < eff < total
    print("industrial table row:", _sourced("harris_case_depth"))


def test_austenite_row_cuts_the_drive_and_refuses_below_it():
    p = {"geometry": "wells", "wa": 0.07, "wb": 0.06, "Cs": 1.0, "C0": 0.2,
         "Q_cal": 32000.0, "A1_C": 727.0}
    cool = F.TableDrive([(HOUR, 925.0), (2 * HOUR, 600.0)])
    ok, until, why = FORMS["fick"].valid((0.0,), p, {"T": cool}, 0, 2 * HOUR)
    assert not ok and until == HOUR and "austenite" in why
    with pytest.raises(F.OutOfRange):
        FORMS["fick"].advance((0.0,), p, {"T": F.ConstDrive(600.0)}, 0, HOUR)
    sine = F.SineDrive(800.0, 100.0, 7200.0)             # dips under 727 C
    ok, until, why = FORMS["fick"].valid((0.0,), p, {"T": sine}, 0, 7200 * 40)
    assert not ok and 0 < until < 7200 * 40
    assert sine.value(until) < 727.0 <= sine.value(until - 1) or sine.value(until - 1) >= 727.0


# --- slab -----------------------------------------------------------------
def _fd_slab(L, D, hours, c0, cs, N=401):
    x = np.linspace(0, L, N)
    dx = x[1] - x[0]
    c = np.full(N, c0)
    c[0] = c[-1] = cs
    dt = 0.2 * dx * dx / D
    t, tend = 0.0, hours * 3600.0
    while t < tend:
        s = min(dt, tend - t)
        c[1:-1] += D * s / dx ** 2 * (c[2:] - 2 * c[1:-1] + c[:-2])
        t += s
    return x, c


def test_slab_40_modes_agree_with_a_finite_difference_run_to_1e5():
    L, D = 0.02, 1e-9
    p = {"L": L, "D": D, "Cs": 0.0, "bc": "dd"}
    b0 = F.slab_uniform(1.0, 0.0, L, 40)
    b1, fl = FORMS["fick"].advance(tuple(b0), p, {}, 0, 3 * HOUR)
    x, fd = _fd_slab(L, D, 3, 1.0, 0.0)
    modes = F.slab_profile(b1, x, L, "dd", 0.0)
    assert float(np.abs(modes - fd).max()) < 1e-5


def test_slab_face_flux_equals_the_change_of_the_profile_integral():
    L, D = 0.02, 1e-9
    p = {"L": L, "D": D, "Cs": 0.3, "bc": "dd"}
    b0 = tuple(F.slab_uniform(1.0, 0.3, L, 40))
    b1, fl = FORMS["fick"].advance(b0, p, {}, 0, 3 * HOUR)
    m0, m1 = F.slab_mass(b0, L, "dd", 0.3), F.slab_mass(b1, L, "dd", 0.3)
    assert abs((m0 - m1) - (fl[("mass", "c", "face0")] + fl[("mass", "c", "faceL")])) < 1e-12
    assert abs(fl[("mass", "c", "face0")] - fl[("mass", "c", "faceL")]) < 1e-12   # symmetric


def test_slab_sealed_variants_are_the_mirror_of_the_open_slab():
    L, D = 0.02, 1e-9
    open_ = {"L": 2 * L, "D": D, "Cs": 0.0, "bc": "dd"}
    seal = {"L": L, "D": D, "Cs": 0.0, "bc": "ds"}
    b_open, _ = FORMS["fick"].advance(tuple(F.slab_uniform(1.0, 0.0, 2 * L, 80)), open_, {}, 0, 2 * HOUR)
    b_seal, fl = FORMS["fick"].advance(tuple(F.slab_uniform(1.0, 0.0, L, 40, "ds")), seal, {}, 0, 2 * HOUR)
    x = np.linspace(0, L, 21)
    a = F.slab_profile(b_open, x, 2 * L, "dd", 0.0)
    b = F.slab_profile(b_seal, x, L, "ds", 0.0)
    assert float(np.abs(a - b).max()) < 3e-3          # 40 vs 80 modes of the step start
    # both faces sealed: the mean is all there is, and nothing leaves
    ss = {"L": L, "D": D, "Cs": 0.0, "bc": "ss"}
    b0 = (0.7, 0.2, -0.1, 0.05)
    b1, fl = FORMS["fick"].advance(b0, ss, {}, 0, 5 * HOUR)
    assert b1[0] == 0.7 and fl[("mass", "c", "face0")] == 0.0 and fl[("mass", "c", "faceL")] == 0.0
    assert abs(F.slab_mass(b1, L, "ss", 0.0) - F.slab_mass(b0, L, "ss", 0.0)) < 1e-15


def _bump(x, L, c_in=1.0, cs=0.1):
    return cs + (c_in - cs) * np.exp(-((x - 0.5 * L) / (0.15 * L)) ** 2) * \
        np.sin(np.pi * x / L) ** 2


def test_slab_fold_fits_a_hot_spot_conserves_mass_and_does_not_ring():
    L = 0.02
    p = {"L": L, "D": 1e-9, "Cs": 0.1, "bc": "dd"}
    x = np.linspace(0, L, 201)
    c = _bump(x, L)
    c[0] = c[-1] = 0.1
    w = 0.5 * (np.r_[0.0, np.diff(x)] + np.r_[np.diff(x), 0.0])
    m_ref = float(np.dot(w, c))
    b, info = F.slab_project(x, c, p)
    assert 1 < info["K"] <= 32
    rec = F.slab_inject(b, x, p)
    assert abs(float(np.dot(w, rec)) - m_ref) / m_ref < 1e-9
    rng = c.max() - c.min()
    assert rec.min() >= c.min() - 1e-3 * rng and rec.max() <= c.max() + 1e-3 * rng
    # idempotent: fold -> inject -> fold is the same state (no drift)
    b2, _ = F.slab_project(x, rec, p, mass_ref=m_ref)
    assert max(abs(u - v) for u, v in zip(b, b2)) < 1e-9 * max(abs(v) for v in b) + 1e-12


def test_slab_fold_refuses_a_mismatched_face_and_a_profile_it_would_ring_on():
    L = 0.02
    p = {"L": L, "D": 1e-9, "Cs": 0.1, "bc": "dd"}
    x = np.linspace(0, L, 201)
    c = _bump(x, L)
    c[0] = c[-1] = 0.1
    bad = c.copy()
    bad[0] = 0.6
    with pytest.raises(F.Refused, match="face 0"):
        F.slab_project(x, bad, p)
    step = np.full_like(x, 1.0)
    step[0] = step[-1] = 0.1
    with pytest.raises(F.Refused):
        F.slab_project(x, step, p)                      # a step cannot be 32 modes


# ───────────────────────────────────────────── T7 population ──
def _rk4_pop(y, q, mu, t, ymax, n=20000):
    h = t / n
    f = lambda y, q: (mu * q / (1 + q) * (1 - math.exp(y - ymax)), mu * q)    # noqa: E731
    for _ in range(n):
        k1 = f(y, q)
        k2 = f(y + h / 2 * k1[0], q + h / 2 * k1[1])
        k3 = f(y + h / 2 * k2[0], q + h / 2 * k2[1])
        k4 = f(y + h * k3[0], q + h * k3[1])
        y += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        q += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return y, q


POP = {"b": 0.0437, "Tmin": -3.0, "Tmax": 48.0, "c": 0.3, "ymax": math.log(1e9),
       "T_kill": 60.0, "D_ref_s": 180.0, "T_ref_C": 121.1, "z": 10.0}


def test_population_one_piece_matches_rk4_to_1e12():
    y0, q0 = math.log(1e3), 1.0 / (math.exp(2.0) - 1.0)
    T_c = 20.0
    mu_h = F.ratkowsky_mu(T_c, 0.0437, -3.0, 48.0, 0.3)
    (y, lq), fl = FORMS["population"].advance((y0, math.log(q0)), POP, {"T": F.ConstDrive(T_c)},
                                              0, 10 * HOUR)
    yr, qr = _rk4_pop(y0, q0, mu_h / 3600.0, 36000.0, POP["ymax"])
    assert abs(y - yr) < 1e-11 and abs(math.exp(lq) / qr - 1.0) < 1e-11


def test_population_three_piece_schedule_matches_rk4_piece_by_piece():
    y0, q0 = math.log(1e3), 1.0 / (math.exp(2.0) - 1.0)
    tab = F.TableDrive([(4 * HOUR, 8.0), (9 * HOUR, 30.0), (12 * HOUR, 15.0)])
    (y, lq), _ = FORMS["population"].advance((y0, math.log(q0)), POP, {"T": tab}, 0, 12 * HOUR)
    yr, qr = y0, q0
    for T_c, hrs in ((8.0, 4), (30.0, 5), (15.0, 3)):
        mu = F.ratkowsky_mu(T_c, 0.0437, -3.0, 48.0, 0.3) / 3600.0
        yr, qr = _rk4_pop(yr, qr, mu, hrs * 3600.0, POP["ymax"])
    assert abs(y - yr) < 1e-10 and abs(math.exp(lq) / qr - 1.0) < 1e-10


def test_population_year_long_piece_does_not_overflow_and_reaches_the_cap():
    y0, lq0 = math.log(1e3), 0.0
    (y, lq), _ = FORMS["population"].advance((y0, lq0), POP, {"T": F.ConstDrive(37.0)},
                                             0, 365 * DAY)
    assert math.isfinite(y) and math.isfinite(lq)
    assert abs(y - POP["ymax"]) < 1e-9                 # the logistic cap, never above it


def test_population_heated_then_cooled_stew_keeps_one_state():
    y0, lq0 = math.log(1e5), math.log(0.5)
    sched = F.TableDrive([(2 * HOUR, 20.0), (2 * HOUR + 600 * 40, 120.0), (8 * HOUR, 20.0)])
    (y, lq), fl = FORMS["population"].advance((y0, lq0), POP, {"T": sched}, 0, 8 * HOUR)
    # the same thing in three calls, carrying (y, lq) by hand
    s1, _ = FORMS["population"].advance((y0, lq0), POP, {"T": F.ConstDrive(20.0)}, 0, 2 * HOUR)
    s2, f2 = FORMS["population"].advance(s1, POP, {"T": F.ConstDrive(120.0)}, 0, 600 * 40)
    s3, _ = FORMS["population"].advance(s2, POP, {"T": F.ConstDrive(20.0)}, 0, 8 * HOUR - 2 * HOUR - 600 * 40)
    assert abs(y - s3[0]) < 1e-12 and abs(lq - s3[1]) < 1e-12
    assert s2[0] < s1[0] - 1.0                          # the heat killed most of them
    assert f2[("mass", "biomass", "dead")] >= 0.0


def test_u_ratkowsky_full_form_has_a_maximum_and_these_rows():
    print("ratkowsky b, Tmin, Tmax, c:", _sourced("ratkowsky_enteric"))
    mu = lambda T_: F.ratkowsky_mu(T_, 0.0437, -3.0, 48.0, 0.3)           # noqa: E731
    assert abs(mu(4.0) - 0.094) < 0.001 and abs(mu(10.0) - 0.32) < 0.01
    assert abs(mu(20.0) - 1.0) < 0.05 and abs(mu(37.0) - 2.8) < 0.1 and abs(mu(45.0) - 1.5) < 0.1
    assert mu(50.0) == 0.0 and mu(48.0) == 0.0 and mu(-3.0) == 0.0 and mu(-5.0) == 0.0
    assert mu(60.0) == 0.0                              # v1's form was 6 per hour here
    assert abs(math.log(2.0) / mu(37.0) - 0.2477) < 0.01


def test_population_death_piece_follows_the_z_rule():
    """D(T) = D_ref 10^((T_ref - T)/z): 10 C hotter kills ten times as fast."""
    y0 = math.log(1e9)
    one = FORMS["population"].advance((y0, 0.0), POP, {"T": F.ConstDrive(111.1)}, 0, 40 * 600)[0][0]
    ten = FORMS["population"].advance((y0, 0.0), POP, {"T": F.ConstDrive(121.1)}, 0, 40 * 60)[0][0]
    d1, d2 = y0 - one, y0 - ten
    assert abs(d1 - d2) < 1e-9                          # 600 s at 111.1 = 60 s at 121.1
    assert abs(d2 - math.log(10.0) * 60.0 / 180.0) < 1e-9


# ───────────────────────────────────────────────────── T8 front ──
def test_stefan_ice_0_235_m_in_10_days_matches_the_60_s_ode_to_1e4():
    p = {"mode": "ice", "K": 2.2, "R0": 1 / 15.0, "rho": 917.0, "L_f": 334e3, "Tf": 0.0}
    (x,), fl = FORMS["front"].advance((0.0,), p, {"T": F.ConstDrive(-10.0)}, 0, 10 * DAY)
    h, ki, rho, Lf = 1e-6, 2.2, 917.0, 334e3
    for _ in range(10 * 86400 // 60):
        h += 10.0 / (rho * Lf * (h / ki + 1 / 15.0)) * 60
    assert abs(x - 0.23495) < 2e-5 and abs(x / h - 1.0) < 1e-4
    p0 = dict(p, R0=0.0)
    (x0,), _ = FORMS["front"].advance((0.0,), p0, {"T": F.ConstDrive(-10.0)}, 0, 10 * DAY)
    assert abs(x0 - 0.352) < 0.001                      # no air film
    S = 10.0 * 10 * 86400 / (917.0 * 334e3)
    assert abs(x0 - math.sqrt(2 * 2.2 * S)) < 1e-12     # R0 = 0: sqrt(2 K S)
    pb = dict(p, R0=1e3)                                # K R0 large: x = S / R0
    (xb,), _ = FORMS["front"].advance((0.0,), pb, {"T": F.ConstDrive(-10.0)}, 0, 10 * DAY)
    assert abs(xb / (S / 1e3) - 1.0) < 1e-3


def test_a_summed_drive_cuts_at_its_level_crossings_like_a_sine_alone():
    """SumDrive.level_ticks: a constant plus a sine crosses 0 C at the ticks the
    sine alone does (analytic); a diurnal-plus-annual sum finds every crossing a
    minute-by-minute scan finds."""
    sine = F.SineDrive(-2.0, 10.0, 86400.0)
    summed = F.SumDrive(F.ConstDrive(-2.0), F.SineDrive(0.0, 10.0, 86400.0))
    assert summed.level_ticks(0, 10 * DAY, 0.0) == sine.level_ticks(0, 10 * DAY, 0.0)
    assert len(summed.level_ticks(0, 10 * DAY, 0.0)) == 20
    clim = F.SumDrive(F.ConstDrive(8.0), F.SineDrive(0.0, 10.0, 86400.0, 21600.0),
                      F.SineDrive(0.0, 10.0, 365 * 86400.0, 7776000.0))
    cuts = clim.level_ticks(0, 60 * DAY, 0.0)
    scan = [t for t in range(0, 60 * DAY, 2400) if (clim.value(t) >= 0.0) != (clim.value(t + 2400) >= 0.0)]
    assert len(cuts) == len(scan) and len(cuts) > 10
    for t in cuts:                                      # first tick on the far side
        assert (clim.value(t - 1) >= 0.0) != (clim.value(t) >= 0.0)


def test_ice_under_a_summed_climate_drive_matches_the_60_s_ode_to_1e4():
    """The verifier's case: -2 C plus a 10 K diurnal swing for 10 days. A piece
    that straddled 0 C used its mean and was 2.5 % off; now the sum is cut at
    its crossings and the form is as exact as for a plain sine."""
    p = {"mode": "ice", "K": 2.2, "R0": 1 / 15.0, "rho": 917.0, "L_f": 334e3, "Tf": 0.0}
    d = F.SumDrive(F.ConstDrive(-2.0), F.SineDrive(0.0, 10.0, 86400.0))
    (x,), _ = FORMS["front"].advance((0.0,), p, {"T": d}, 0, 10 * DAY)
    h, ki, rho, Lf = 1e-6, 2.2, 917.0, 334e3
    for i in range(10 * 86400 // 60):
        tm = (i + 0.5) * 60.0 / T
        h += max(0.0 - d.value(tm), 0.0) / (rho * Lf * (h / ki + 1 / 15.0)) * 60
    assert abs(x / h - 1.0) < 1e-4, (x, h)


def test_population_death_under_a_sine_drive_is_gl4_exact():
    """Above T_kill the log-count falls by the integral of ln10/D(T(t)): the
    same quadrature as growth. A sine across T_kill, against a 5 s integration."""
    y0 = math.log(1e9)
    d = F.SineDrive(60.0, 12.0, 6 * 3600.0)
    (y, _), _ = FORMS["population"].advance((y0, 0.0), dict(POP, y_floor=-1e9), {"T": d}, 0, 6 * HOUR)
    yr, dt = y0, 5.0
    for i in range(int(6 * 3600 / dt)):
        Tm = d.value((i + 0.5) * dt / T)
        if Tm >= POP["T_kill"]:
            yr -= math.log(10.0) / (POP["D_ref_s"] * 10.0 ** ((POP["T_ref_C"] - Tm) / POP["z"])) * dt
        else:
            mu = F.ratkowsky_mu(Tm, POP["b"], POP["Tmin"], POP["Tmax"], POP.get("c", 0.3))
            assert mu == 0.0                             # above Tmax (48 C) or below T_kill: no growth here
    assert abs(y - yr) < 1e-9, (y, yr)


def test_ice_flux_dict_books_latent_heat_up_and_nine_percent_volume():
    p = {"mode": "ice", "K": 2.2, "R0": 1 / 15.0, "rho": 917.0, "L_f": 334e3, "Tf": 0.0,
         "area": 2.0}
    (x,), fl = FORMS["front"].advance((0.0,), p, {"T": F.ConstDrive(-10.0)}, 0, 10 * DAY)
    assert abs(fl[("E", "latent", "air_top")] - x * 917.0 * 334e3 * 2.0) < 1e-3 * x
    assert fl[("E", "latent", "water")] == 0.0
    water_m3 = x * 2.0 * 917.0 / 1000.0
    assert abs(fl[("vol", "ice", "host")] / water_m3 - (1000.0 / 917.0 - 1.0)) < 1e-12
    assert abs((1000.0 / 917.0 - 1.0) - 0.0905) < 1e-3                        # +9 %


def test_ice_with_a_water_side_flux_matches_a_fine_ode_and_stops_growing():
    p = {"mode": "ice", "K": 2.2, "R0": 1 / 15.0, "rho": 917.0, "L_f": 334e3, "Tf": 0.0,
         "q_w": 20.0}
    (x,), fl = FORMS["front"].advance((0.01,), p, {"T": F.ConstDrive(-10.0)}, 0, 5 * DAY)
    h, ki, rho, Lf = 0.01, 2.2, 917.0, 334e3
    for _ in range(5 * 86400 // 30):
        h += (10.0 / (h / ki + 1 / 15.0) - 20.0) / (rho * Lf) * 30
    assert abs(x / h - 1.0) < 2e-4
    assert abs(fl[("E", "latent", "water")] + 20.0 * 5 * 86400) < 1e-6     # water gave q_w t
    # latent + water-side heat all went up through the ice
    assert abs(fl[("E", "latent", "air_top")] + fl[("E", "latent", "water")]
               - (x - 0.01) * 917.0 * 334e3) < 1e-3
    (xs,), _ = FORMS["front"].advance((x,), p, {"T": F.ConstDrive(-10.0)}, 0, 200 * DAY)
    umax = 10.0 / 20.0
    assert xs < (umax - 1 / 15.0) * 2.2 + 1e-9          # it asymptotes at the balance thickness


def test_ice_front_valid_flags_the_stefan_number_and_melting():
    p = {"mode": "ice", "K": 2.2, "R0": 0.0, "rho": 917.0, "L_f": 334e3, "Tf": 0.0, "c_ice": 2100.0}
    ok, _, _ = FORMS["front"].valid((0.1,), p, {"T": F.ConstDrive(-10.0)}, 0, DAY)
    assert ok                                           # 2100 * 10 / 334e3 = 0.063
    ok, _, why = FORMS["front"].valid((0.1,), p, {"T": F.ConstDrive(-30.0)}, 0, DAY)
    assert not ok and "Stefan" in why


def test_carbonation_depth_goes_as_the_square_root_of_time():
    p = {"mode": "carb", "K": 5.8e-12, "R0": 0.0, "c_rate": 1.0, "drive": "RH", "cap_mol_m3": 5400.0}
    (a,), _ = FORMS["front"].advance((0.0,), dict(p, co2_avail=1e9), {"RH": F.ConstDrive(1.0)}, 0, 100 * DAY)
    (b,), _ = FORMS["front"].advance((0.0,), dict(p, co2_avail=1e9), {"RH": F.ConstDrive(1.0)}, 0, 400 * DAY)
    assert abs(b / a - 2.0) < 1e-12
    assert abs(a - math.sqrt(2 * 5.8e-12 * 100 * 86400)) < 1e-15


def test_carbonation_is_supply_limited_when_the_room_cannot_supply_the_co2():
    """A 20 m2 x 1 cm wall needs 48 kg of CO2; a 100 m3 room at 400 ppm holds 0.07 kg."""
    area = 20.0
    cap = 5400.0
    need_kg = area * 0.01 * cap * 0.044
    assert abs(need_kg - 47.5) < 0.1
    room_kg = 100.0 * 400e-6 * 1.8
    assert room_kg < 0.08
    p = {"mode": "carb", "K": 5.8e-12, "R0": 0.0, "c_rate": 1.0, "drive": "RH",
         "cap_mol_m3": cap, "area": area, "co2_avail": room_kg / 0.044 / area}
    (x,), fl = FORMS["front"].advance((0.0,), p, {"RH": F.ConstDrive(1.0)}, 0, 5000 * DAY)
    free = math.sqrt(2 * 5.8e-12 * 5000 * 86400)
    assert x < 0.01 * free and fl[("flag", "supply_limited", "-")] == 1.0
    assert abs(-fl[("mass", "co2", "room")] / 1000.0 - room_kg) < 1e-9       # it took what there was
    ok, _, why = FORMS["front"].valid((0.0,), dict(p, co2_avail=None), {}, 0, DAY)
    assert not ok and "supply" in why


# ─────────────────────────────────────────────── T9 power law ──
def test_powerlaw_b_half_gives_sqrt_t_and_126_um_in_ten_years():
    p = {"b": 0.5, "r": 40.0, "floor": 1e-9}
    wet = {"T": F.ConstDrive(15.0), "RH": F.ConstDrive(95.0)}
    (d10,), fl = FORMS["powerlaw"].advance((1e-9,), p, wet, 0, int(10 * F.YEAR_S / T))
    (d40,), _ = FORMS["powerlaw"].advance((1e-9,), p, wet, 0, int(40 * F.YEAR_S / T))
    assert abs(d10 - 40.0 * math.sqrt(10.0)) < 1e-3 and abs(d10 - 126.5) < 0.1
    assert abs(d40 / d10 - 2.0) < 1e-6


def test_powerlaw_changing_r_midway_keeps_the_damage_continuous():
    """Equivalent time: moving to a dry cellar (w = 0) changes the rate and not D."""
    p = {"b": 0.5, "r": 40.0}
    wet = {"T": F.ConstDrive(15.0), "RH": F.ConstDrive(95.0)}
    dry = {"T": F.ConstDrive(15.0), "RH": F.ConstDrive(50.0)}
    yr = int(F.YEAR_S / T)
    d1, _ = FORMS["powerlaw"].advance((1e-9,), p, wet, 0, 3 * yr)
    d2, _ = FORMS["powerlaw"].advance(d1, p, dry, 0, 5 * yr)
    assert d2[0] == d1[0]                               # dry air: no new damage, no jump either
    d3, _ = FORMS["powerlaw"].advance(d2, dict(p, r=10.0), wet, 0, 2 * yr)
    d_ref, _ = FORMS["powerlaw"].advance((1e-9,), p, wet, 0, 3 * yr)
    # with r changed, D continues from where it was: Z = D^2 grows linearly at r'^2
    assert abs(d3[0] ** 2 - (d_ref[0] ** 2 + 10.0 ** 2 * 2)) < 1e-6
    # a semigroup for any r
    a, _ = FORMS["powerlaw"].advance((1e-9,), p, wet, 0, 7 * yr)
    b1, _ = FORMS["powerlaw"].advance((1e-9,), p, wet, 0, 2 * yr + 5)
    b2, _ = FORMS["powerlaw"].advance(b1, p, wet, 2 * yr + 5, 7 * yr)
    assert abs(a[0] / b2[0] - 1.0) < 1e-12


def test_powerlaw_wetness_row_needs_rh_above_80_and_t_above_0():
    p = {"b": 0.5, "r": 40.0}
    yr = int(F.YEAR_S / T)
    for T_c, rh, grows in ((15, 95, True), (15, 79, False), (-5, 95, False), (15, 80, False)):
        (d,), _ = FORMS["powerlaw"].advance((1e-9,), p, {"T": F.ConstDrive(T_c), "RH": F.ConstDrive(rh)}, 0, yr)
        assert (d > 1.0) == grows
    assert F.wetness(10, 81) == 1.0 and F.wetness(0.0, 90) == 0.0
    # half the year below 0 C: half the growth in Z
    tab = F.TableDrive([(yr // 2, 10.0), (yr, -10.0)])
    (d,), fl = FORMS["powerlaw"].advance((1e-9,), p, {"T": tab, "RH": F.ConstDrive(95.0)}, 0, yr)
    assert abs(d - 40.0 * math.sqrt(0.5)) < 1e-6
    assert fl[("vol", "rust", "host")] > 0 and fl[("mass", "fe", "rust")] > 0


# ──────────────────────────────────────────────────── T10 drying ──
def test_drying_period_one_is_linear_and_continuous_at_the_critical_moisture():
    p = {"X_c": 0.3, "X_eq": 0.12, "rate_c": 1e-6, "D": 5e-11, "l": 0.0127}
    s0 = (0.6, 0.0)
    (a, wa), _ = FORMS["drying"].advance(s0, p, {}, 0, 10 * HOUR)
    (b, wb), _ = FORMS["drying"].advance(s0, p, {}, 0, 20 * HOUR)
    assert abs((0.6 - a) - 1e-6 * 36000) < 1e-12 and abs((0.6 - b) - 1e-6 * 72000) < 1e-12
    t_c = int(0.3 / 1e-6 / T)                           # 3e5 s: X reaches X_c
    (x_c, w_c), _ = FORMS["drying"].advance(s0, p, {}, 0, t_c)
    assert abs(x_c - 0.3) < 1e-6 and w_c == 0.0
    (after, w2), _ = FORMS["drying"].advance(s0, p, {}, 0, t_c + HOUR)
    assert 0.12 < after < x_c and w2 > 0                # falling rate has begun, continuously
    mr = F.drying_series(w2, 0.0127)                    # and it is that series, from MR = 1 at X_c
    assert abs(after - (0.12 + 0.18 * mr)) < 1e-12 and 0.0 < 1.0 - mr < 0.05


def test_drying_series_is_the_mean_of_the_fick_slab_it_stands_for():
    l, deff = 0.0127, 5e-12
    half = {"L": 2 * l, "D": 1.0, "Cs": 0.0, "bc": "dd"}      # two-sided: faces at both ends
    b0 = tuple(F.slab_uniform(1.0, 0.0, 2 * l, 399))
    for t_s in (3600.0, 86400.0 * 30, 86400.0 * 120):
        b1, _ = FORMS["fick"].advance(b0, dict(half, D=deff), {}, 0, int(t_s / T))
        x = np.linspace(0, 2 * l, 4001)
        mean = float(np.trapezoid(F.slab_profile(b1, x, 2 * l, "dd", 0.0), x) / (2 * l))
        assert abs(mean - F.drying_series(deff * t_s, l)) < 5e-4


def test_drying_series_hands_over_to_the_early_time_form_without_a_step():
    l = 0.0127
    x = 0.05
    early = 1 - 2 * math.sqrt(x / math.pi)
    assert abs(F.drying_modes(x, 2000) - early) < 1e-8
    assert F.drying_series(0.0, l) == 1.0
    assert abs(F.drying_series(x * l * l, l) - F.drying_modes(x)) < 1e-8


def test_drying_valid_fails_when_d_varies_by_more_than_20_percent():
    p = {"X_c": 0.3, "X_eq": 0.12, "rate_c": 1e-6, "D": 5e-11, "l": 0.0127}
    assert FORMS["drying"].valid((0.2, 1.0), p, {}, 0)[0]
    ok, _, why = FORMS["drying"].valid((0.2, 1.0), dict(p, Dratio=10.0), {}, 0)
    assert not ok and "20%" in why
    st = F.drying_state(0.2, p)
    assert abs(FORMS["drying"].observe(st, p, "X") - 0.2) < 1e-12
    assert abs((0.2 - 0.12) / (0.3 - 0.12) - F.drying_series(st[1], p["l"])) < 1e-9


def test_u_one_year_per_inch_is_a_calibration_check_not_a_law():
    print("wood bound-water D:", _sourced("wood_bound_water_D"))
    l = 0.0127
    year = F.YEAR_S
    assert abs(F.drying_series(5e-12 * year, l) - 0.07) < 0.005
    assert F.drying_series(5e-11 * year, l) < 1e-6


# ─────────────────────────────────────────────── T11 degree-days ──
def test_gdd_clipped_sine_matches_a_fine_numeric_integral_and_stages_cross():
    d = F.SineDrive(12.0, 14.0, 86400.0, 0.0)
    p = {"T_base": 5.0, "T_cap": 22.0}
    (g,), _ = FORMS["rate_integral"].advance((0.0,), p, {"T": d}, 0, 30 * DAY)
    ts = np.arange(0, 30 * 86400, 20.0) + 10.0
    Tv = 12.0 + 14.0 * np.sin(2 * np.pi * ts / 86400.0)
    num = float(np.sum(np.clip(np.minimum(Tv, 22.0) - 5.0, 0.0, None)) * 20.0 / 86400.0)
    assert abs(g / num - 1.0) < 1e-4
    # a table drive, exactly: 10 C above base for 3 days
    tab = F.TableDrive([(3 * DAY, 15.0), (9 * DAY, 2.0)])
    (g2,), _ = FORMS["rate_integral"].advance((0.0,), p, {"T": tab}, 0, 9 * DAY)
    assert abs(g2 - 30.0) < 1e-9
    tc = FORMS["rate_integral"].cross((0.0,), p, {"T": d}, 0, "gdd", 100.0, 100 * DAY)
    a, _ = FORMS["rate_integral"].advance((0.0,), p, {"T": d}, 0, tc)
    b, _ = FORMS["rate_integral"].advance((0.0,), p, {"T": d}, 0, tc - 1)
    assert a[0] >= 100.0 > b[0]


# ──────────────────────────────────────────────── T12 two process ──
def test_two_process_s_curves_and_the_pure_exponential_durations():
    tp = {"H": 0.60, "L": 0.17, "a": 0.0, "phi_h": 0.0}
    assert abs(F.two_process_S(0.2, True, 16.0, 18.2, 4.2) - (1 - 0.8 * math.exp(-16 / 18.2))) < 1e-15
    assert abs(F.two_process_S(0.6, False, 8.0, 18.2, 4.2) - 0.6 * math.exp(-8 / 4.2)) < 1e-15
    # awake 0.2 -> 0.6 takes 18.2 ln 2 = 12.6 h; asleep 0.6 -> 0.2 takes 4.2 ln 3 = 4.6 h
    t_up = FORMS["two_process"].cross((0.2, 1.0), tp, {}, 0, "switch", 0.0, 40 * HOUR)
    assert abs(t_up / HOUR - 18.2 * math.log(2.0)) < 1e-3
    t_dn = FORMS["two_process"].cross((0.6, 0.0), dict(tp, L=0.2), {}, 0, "switch", 0.0, 40 * HOUR)
    assert abs(t_dn / HOUR - 4.2 * math.log(3.0)) < 1e-3


def test_two_process_crossing_of_the_c_modulated_line_lands_on_one_tick():
    tp = {"H": 0.60, "L": 0.17, "a": 0.10, "phi_h": 0.0}
    tc = FORMS["two_process"].cross((0.3, 1.0), tp, {}, 0, "switch", 0.0, 60 * HOUR)
    S = lambda t: F.two_process_S(0.3, True, t * T / 3600.0, 18.2, 4.2)       # noqa: E731
    line = lambda t: 0.60 + 0.10 * F.two_process_C(t, 0.0)                    # noqa: E731
    assert S(tc) >= line(tc) and S(tc - 1) < line(tc - 1)
    (S1, aw), _ = FORMS["two_process"].advance((0.3, 1.0), tp, {}, 0, tc + 1000)
    assert aw == 0.0 and S1 < line(tc)                   # it switched to sleep and began to fall


def test_codeonly_two_process_calibration_cycle_with_these_rows():
    """CALIBRATION (u, BLOCKING for any scene): with H 0.60, L 0.17, a 0.10 and phi 0
    the cycle of the pure exponentials is 17.2 h; C stretches it. It is run for
    ten days and the sleep onsets are required to repeat within a stated bound.
    This pins the code to its rows; it says nothing about people."""
    tp = {"H": 0.60, "L": 0.17, "a": 0.10, "phi_h": 0.0}
    state, t, onsets = (0.3, 1.0), 0, []
    for _ in range(40):
        ts = FORMS["two_process"].cross(state, tp, {}, t, "switch", 0.0, t + 60 * HOUR)
        state, _ = FORMS["two_process"].advance(state, tp, {}, t, ts)
        t = ts
        if state[1] == 0.0:
            onsets.append(t / HOUR)
    gaps = np.diff(onsets)
    assert 8 < len(gaps)
    assert np.all((gaps > 15.0) & (gaps < 26.0))
    print("sleep-onset gaps (h):", np.round(gaps[-5:], 3), _sourced("borbely_rows"))


# ───────────────────────────────────── T13 semigroup, T14 drives ──
_SEMIGROUP = [
    ("decay", {"A": 1e10, "Ea": 80e3}, (1.0,), "T"),
    ("relax", {"tau_s": 7200.0}, (1.0,), "target"),
    ("population", POP, (math.log(1e3), 0.0), "T"),
    ("powerlaw", {"b": 0.5, "r": 40.0}, (1.0,), "T"),
    ("rate_integral", {"T_base": 5.0, "T_cap": 22.0}, (0.0,), "T"),
]


def _drives_for(form, name, kind):
    if kind == "const":
        d = F.ConstDrive(30.0 if name != "target" else 40.0)
    elif kind == "table":
        d = F.TableDrive([(5 * HOUR, 12.0), (14 * HOUR, 33.0), (40 * HOUR, 9.0)])
    else:
        d = F.SineDrive(14.0, 9.0, 86400.0, 1000.0)
    out = {name: d}
    if form == "powerlaw":
        out["RH"] = F.ConstDrive(95.0)
    return out


@pytest.mark.parametrize("kind,tol", [("const", 1e-12), ("table", 1e-12), ("sine", 1e-9)])
@pytest.mark.parametrize("form,p,s0,name", _SEMIGROUP)
def test_semigroup_advance_a_then_b_equals_advance_a_plus_b(form, p, s0, name, kind, tol):
    f = FORMS[form]
    dr = _drives_for(form, name, kind)
    a, b, c = 3 * HOUR + 17, 11 * HOUR + 3, 30 * HOUR + 1
    mid, _ = f.advance(s0, p, dr, 0, a)
    two, _ = f.advance(mid, p, dr, a, b)
    two, _ = f.advance(two, p, dr, b, c)
    one, _ = f.advance(s0, p, dr, 0, c)
    for u, v in zip(one, two):
        assert abs(u - v) <= tol * max(abs(u), abs(v), 1e-30)


def test_semigroup_of_fick_slab_front_and_rc_net():
    L, D = 0.02, 1e-9
    p = {"L": L, "D": D, "Cs": 0.0, "bc": "dd"}
    b0 = tuple(F.slab_uniform(1.0, 0.0, L, 40))
    a, _ = FORMS["fick"].advance(b0, p, {}, 0, 3 * HOUR + 5)
    b, _ = FORMS["fick"].advance(a, p, {}, 3 * HOUR + 5, 8 * HOUR)
    c, _ = FORMS["fick"].advance(b0, p, {}, 0, 8 * HOUR)
    assert max(abs(u - v) for u, v in zip(b, c)) < 1e-14
    pf = {"mode": "ice", "K": 2.2, "R0": 0.1, "rho": 917.0, "L_f": 334e3, "Tf": 0.0}
    d = F.TableDrive([(2 * DAY, -4.0), (5 * DAY, -12.0), (9 * DAY, 3.0)])
    x1, _ = FORMS["front"].advance((0.0,), pf, {"T": d}, 0, 4 * DAY + 9)
    x2, _ = FORMS["front"].advance(x1, pf, {"T": d}, 4 * DAY + 9, 9 * DAY)
    x3, _ = FORMS["front"].advance((0.0,), pf, {"T": d}, 0, 9 * DAY)
    assert abs(x2[0] - x3[0]) < 1e-12 * x3[0]


def test_drive_pieces_reuse_the_absolute_grid_across_splits():
    d = F.SineDrive(10.0, 5.0, 86400.0)
    g = 12000
    whole = F.pieces({"T": d}, 7, 100000, g)
    left = F.pieces({"T": d}, 7, 43210, g)
    right = F.pieces({"T": d}, 43210, 100000, g)
    assert sorted(set(left) | set(right)) == sorted(set(whole) | {43210})
    assert all(x % g == 0 for x in whole[1:-1])
    tab = F.TableDrive([(500, 1.0), (900, 2.0)])
    assert F.pieces({"T": tab}, 0, 1000, 12000) == [0, 500, 900, 1000]     # kinks only
    assert tab.kinks(0, 1000) == [500, 900] and tab.value(499) == 1.0 and tab.value(500) == 2.0


def test_gl4_error_at_a_large_rate_change_across_a_cell():
    """ln k changing by 0.8 across one cell: the midpoint rule is off by 2.6e-2, GL4 by 9e-11."""
    def err(change):
        a = change / 2
        f = lambda t: math.exp(2 * a * (t - 0.5))                              # noqa: E731
        exact = (math.exp(a) - math.exp(-a)) / (2 * a)
        return abs(F.gl4(f, 0.0, 1.0) / exact - 1.0), abs(f(0.5) / exact - 1.0)
    gl8, mid8 = err(0.8)
    gl4_, _ = err(0.4)
    assert gl8 < 1e-10 and abs(mid8 - 2.6e-2) < 1e-3
    assert gl4_ < 1e-12
    # and the grid chooser keeps ln k to 0.5 per cell: an Arrhenius swing of 20 K
    d = F.SineDrive(15.0, 10.0, 86400.0)
    g = F.choose_grid({"T": d}, dlnk_dT=0.112)
    assert 0.112 * d.max_slope() * g <= 0.5 + 1e-9


def test_drive_mean_is_analytic_and_equals_the_numeric_mean():
    for d in (F.SineDrive(10.0, 5.0, 3600.0, 100.0), F.ConstDrive(3.0),
              F.TableDrive([(1000, 1.0), (5000, 4.0)]),
              F.SumDrive(F.SineDrive(1.0, 2.0, 7200.0), F.TableDrive([(2000, 3.0), (9000, 5.0)]))):
        a, b = 321, 7777
        num = np.mean([d.value(t) for t in np.arange(a, b, 0.25)])
        assert abs(d.mean(a, b) - num) < 2e-3 * max(1.0, abs(num))
    s = F.SumDrive(F.TableDrive([(100, 1.0)]), F.TableDrive([(250, 2.0)]))
    assert s.kinks(0, 1000) == [100, 250]
    assert F.pieces({"T": s}, 0, 1000) == [0, 100, 250, 1000]
    sine = F.SineDrive(10.0, 5.0, 86400.0)
    ticks = sine.level_ticks(0, 2 * DAY, 10.0)
    assert len(ticks) == 4 and all(abs(sine.value(t) - 10.0) < 0.01 for t in ticks)


def test_drive_plain_data_registry_builds_by_name():
    d = F.make_drive(("sum", ("const", 2.0), ("table", [(100, 1.0)])))
    assert d.value(50) == 3.0 and d.value(500) == 2.0 + 1.0


# ────────────────────────────────────────────── T14b flux closure ──
_CLOSURE = {
    "decay": ({"half_s": 3600.0, "product": "y", "heat_J_per_unit": 5.0}, (2.0,), {}),
    "relax": ({"tau_s": 1000.0, "target": 3.0, "heat_J_per_unit": 1.0}, (1.0,), {}),
    "drain": ({"floor": 0.0, "rate": 1e-3}, (10.0,), {}),
    "fick": ({"L": 0.02, "D": 1e-9, "Cs": 0.2, "bc": "dd"}, tuple(F.slab_uniform(1.0, 0.2, 0.02, 30)), {}),
    "population": (POP, (math.log(1e3), 0.0), {"T": F.ConstDrive(25.0)}),
    "front": ({"mode": "ice", "K": 2.2, "R0": 0.05, "rho": 917.0, "L_f": 334e3, "Tf": 0.0, "q_w": 5.0},
              (0.01,), {"T": F.ConstDrive(-8.0)}),
    "powerlaw": ({"b": 0.5, "r": 40.0, "area": 2.0}, (1.0,), {"T": F.ConstDrive(15.0), "RH": F.ConstDrive(90.0)}),
    "drying": ({"X_c": 0.3, "X_eq": 0.1, "rate_c": 1e-6, "D": 5e-11, "l": 0.0127, "dry_g": 500.0}, (0.5, 0.0), {}),
    "rate_integral": ({"T_base": 5.0}, (0.0,), {"T": F.ConstDrive(15.0)}),
    "two_process": ({}, (0.3, 1.0), {}),
}


@pytest.mark.parametrize("kind", sorted(_CLOSURE))
def test_every_form_returns_every_flux_key_it_declares(kind):
    p, s0, dr = _CLOSURE[kind]
    f = FORMS[kind]
    s1, fl = f.advance(s0, p, dr, 0, 6 * HOUR)
    assert set(fl) == set(f.flux_keys(p)), f"{kind}: {set(f.flux_keys(p)) ^ set(fl)}"
    assert all(isinstance(v, float) for v in fl.values())


def test_flux_closure_per_form_the_state_change_is_paid_by_a_named_sink():
    t1 = 6 * HOUR
    s, fl = FORMS["decay"].advance((2.0,), *(_CLOSURE["decay"][0], {}, 0, t1))
    assert abs((2.0 - s[0]) - fl[("mass", "y", "product")]) < 1e-15
    assert abs(fl[("E", "heat", "air")] - 5.0 * (2.0 - s[0])) < 1e-12
    s, fl = FORMS["relax"].advance((1.0,), _CLOSURE["relax"][0], {}, 0, t1)
    assert abs((s[0] - 1.0) + fl[("store", "x", "sink")]) < 1e-15
    s, fl = FORMS["drain"].advance((10.0,), _CLOSURE["drain"][0], {}, 0, t1)
    assert abs((10.0 - s[0]) - fl[("store", "x", "drained")]) < 1e-15
    p, s0, _ = _CLOSURE["fick"]
    s, fl = FORMS["fick"].advance(s0, p, {}, 0, t1)
    d = F.slab_mass(s0, 0.02, "dd", 0.2) - F.slab_mass(s, 0.02, "dd", 0.2)
    assert abs(d - fl[("mass", "c", "face0")] - fl[("mass", "c", "faceL")]) < 1e-14
    p, s0, dr = _CLOSURE["front"]
    s, fl = FORMS["front"].advance(s0, p, dr, 0, 3 * DAY)
    latent = (s[0] - 0.01) * 917.0 * 334e3
    assert abs(fl[("E", "latent", "air_top")] + fl[("E", "latent", "water")] - latent) < 1e-3
    p, s0, dr = _CLOSURE["powerlaw"]
    s, fl = FORMS["powerlaw"].advance(s0, p, dr, 0, 365 * DAY)
    assert abs(fl[("mass", "fe", "rust")] - (s[0] - 1.0) * 1e-6 * 7870.0 * 1e3 * 2.0) < 1e-9
    p, s0, dr = _CLOSURE["drying"]
    s, fl = FORMS["drying"].advance(s0, p, dr, 0, 100 * HOUR)
    assert abs(fl[("mass", "h2o", "vapor")] - (0.5 - s[0]) * 500.0) < 1e-9
    p, s0, dr = _CLOSURE["population"]
    s, fl = FORMS["population"].advance(s0, p, dr, 0, 10 * HOUR)
    assert fl[("mass", "nutrient", "biomass")] > 0 and fl[("mass", "biomass", "dead")] == 0.0
    s2, fl2 = FORMS["population"].advance(s0, p, {"T": F.ConstDrive(90.0)}, 0, 10 * HOUR)
    assert fl2[("mass", "biomass", "dead")] > 0 and fl2[("mass", "nutrient", "biomass")] == 0.0
    for kind in ("rate_integral", "two_process"):                  # a clock and a view book nothing
        p, s0, dr = _CLOSURE[kind]
        assert FORMS[kind].advance(s0, p, dr, 0, t1)[1] == {}


def test_a_form_with_a_missing_flux_would_fail_the_closure_check():
    f = FORMS["decay"]
    p = _CLOSURE["decay"][0]
    _, fl = f.advance((2.0,), p, {}, 0, HOUR)
    broken = {k: v for k, v in fl.items() if k[0] != "E"}
    assert set(broken) != set(f.flux_keys(p))


def test_crossing_is_inverse_of_advance_for_every_monotone_form():
    cases = [("decay", {"half_s": 3600.0}, (2.0,), {}, "x", 0.5),
             ("relax", {"tau_s": 1000.0, "target": 3.0}, (1.0,), {}, "x", 2.5),
             ("powerlaw", {"b": 0.5, "r": 40.0}, (1.0,), {"T": F.ConstDrive(15.0), "RH": F.ConstDrive(90.0)}, "D", 30.0),
             ("population", POP, (math.log(1e3), 0.0), {"T": F.ConstDrive(25.0)}, "log10", 6.0)]
    for kind, p, s0, dr, key, level in cases:
        f = FORMS[kind]
        tc = f.cross(s0, p, dr, 0, key, level, 400 * DAY)
        assert tc is not None, kind
        a, _ = f.advance(s0, p, dr, 0, tc)
        b, _ = f.advance(s0, p, dr, 0, tc - 1)
        va, vb = f.observe(a, p, key), f.observe(b, p, key)
        assert (va - level) * (vb - level) <= 0 and va != vb, kind
    assert FORMS["decay"].cross((2.0,), {"half_s": 3600.0}, {}, 0, "x", 3.0, DAY) is None


# ─────────────────────────────────────────────── T14c rc_net ──
def _three_node():
    C = [2000.0, 3000.0, 1500.0]
    spec = F.NetSpec(C, [0, 1], [1, 2], [0.8, 0.5], [0.0, 0.0, 0.3], q=[0.0, 4.0, 0.0])
    return spec


def _rk4_net(spec, T0, tb, t_s, n=40000):
    A = spec.laplacian_dense() + np.diag(spec.Gb)
    f = spec.Gb * tb + spec.q
    T_ = np.array(T0, float)
    h = t_s / n
    rhs = lambda T: (f - A @ T) / spec.C                                      # noqa: E731
    for _ in range(n):
        k1 = rhs(T_)
        k2 = rhs(T_ + h / 2 * k1)
        k3 = rhs(T_ + h / 2 * k2)
        k4 = rhs(T_ + h * k3)
        T_ += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return T_


def test_rc_net_three_nodes_match_rk4_to_1e9_with_constant_coefficients():
    spec = _three_node()
    T0 = np.array([40.0, 25.0, 60.0])
    T1, fl = F.rc_advance_arrays(spec, T0, {"Tb": F.ConstDrive(5.0)}, 0, 5 * HOUR)
    ref = _rk4_net(spec, T0, 5.0, 5 * 3600.0)
    assert float(np.abs(T1 - ref).max()) < 1e-9
    E0, E1 = spec.stored(T0), spec.stored(T1)
    # the energy balance: source in minus stored change is what left through the boundary
    assert abs(fl[("E", "boundary", "sink")] - (4.0 * 5 * 3600.0 - (E1 - E0))) < 1e-6
    assert fl[("E", "boundary", "sink")] > 0


def test_rc_net_steady_state_is_the_large_dt_limit():
    spec = _three_node()
    Tss = F.rc_steady(spec, 5.0)
    T1, _ = F.rc_advance_arrays(spec, np.array([40.0, 25.0, 60.0]), {"Tb": F.ConstDrive(5.0)}, 0, 400 * HOUR)
    assert float(np.abs(T1 - Tss).max()) < 1e-6
    assert float(np.abs(spec.laplacian_dense() @ Tss + spec.Gb * (Tss - 5.0) - spec.q).max()) < 1e-9


def test_rc_net_follows_a_sine_boundary_and_is_a_semigroup():
    spec = _three_node()
    sine = F.SineDrive(5.0, 8.0, 86400.0)
    T0 = np.array([20.0, 20.0, 20.0])
    one, _ = F.rc_advance_arrays(spec, T0, {"Tb": sine}, 0, 2 * DAY)
    mid, _ = F.rc_advance_arrays(spec, T0, {"Tb": sine}, 0, 17 * HOUR + 99)
    two, _ = F.rc_advance_arrays(spec, mid, {"Tb": sine}, 17 * HOUR + 99, 2 * DAY)
    assert float(np.abs(one - two).max()) < 5e-4          # piecewise-linear forcing: 1e-3 of the swing
    fine = _rk4_sine(spec, T0, sine, 2 * 86400.0)
    assert float(np.abs(one - fine).max()) < 5e-4 * 16.0


def _rk4_sine(spec, T0, sine, t_s, n=60000):
    A = spec.laplacian_dense() + np.diag(spec.Gb)
    T_ = np.array(T0, float)
    h = t_s / n
    rhs = lambda t, T: (spec.Gb * sine.value(t / F.TICK_S) + spec.q - A @ T) / spec.C     # noqa: E731
    t = 0.0
    for _ in range(n):
        k1 = rhs(t, T_)
        k2 = rhs(t + h / 2, T_ + h / 2 * k1)
        k3 = rhs(t + h / 2, T_ + h / 2 * k2)
        k4 = rhs(t + h, T_ + h * k3)
        T_ += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return T_


def _grid_net(nx, ny, nz, gb_face=0.5):
    n = nx * ny * nz
    idx = np.arange(n).reshape(nx, ny, nz)
    rows, cols = [], []
    for sl_a, sl_b in (((slice(None, -1), slice(None), slice(None)), (slice(1, None), slice(None), slice(None))),
                       ((slice(None), slice(None, -1), slice(None)), (slice(None), slice(1, None), slice(None))),
                       ((slice(None), slice(None), slice(None, -1)), (slice(None), slice(None), slice(1, None)))):
        rows.append(idx[sl_a].ravel())
        cols.append(idx[sl_b].ravel())
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    rng = np.random.default_rng(3)
    g = 0.5 + rng.random(len(rows))
    C = 800.0 + 600.0 * rng.random(n)
    Gb = np.zeros(n)
    Gb[idx[0].ravel()] = gb_face
    return F.NetSpec(C, rows, cols, g, Gb)


def test_rc_net_cg_path_agrees_with_the_dense_eigen_path_on_a_1500_node_net():
    spec = _grid_net(10, 10, 15)
    assert spec.n == 1500
    T0 = 20.0 + 10.0 * np.random.default_rng(5).random(spec.n)
    d = {"Tb": F.ConstDrive(-4.0)}
    eig, fe = F.rc_advance_arrays(spec, T0, d, 0, 2 * HOUR, solver="eigen")
    cg, fc = F.rc_advance_arrays(spec, T0, d, 0, 2 * HOUR, solver="cg", tol=1e-8)
    swing = float(np.ptp(T0)) + 24.0
    assert float(np.abs(eig - cg).max()) < 1e-4 * swing
    assert abs(fe[("E", "boundary", "sink")] / fc[("E", "boundary", "sink")] - 1.0) < 1e-4


def _dyn(kind, link, na, nb, c, side, **over):
    k = dict(dx=0.05, k_air=0.026, eddy=1000.0 * 1.2 / 7500.0, t0=20.0, h_conv=10.0, eps_gas=0.2,
             sigma=5.670374e-8, grav=9.81, tick_s=T, rise_max=0.5)
    k.update(over)
    return F.NetDyn(kind, link, na, nb, c, side, k)


def test_a_net_whose_dynamic_links_do_not_depend_on_temperature_is_the_linear_net():
    """Surface | gas faces with the gas cooler than the surface have G = H_CONV c,
    whatever T is. A chain 0 > 1 > 2 > boundary (heated at the head) keeps every
    gas colder than its surface, so the stepped nonlinear advance equals the
    closed form: this checks the stepping, the energy books and the steady state."""
    C = [800.0, 5.0, 900.0]
    lin = F.NetSpec(C, [0, 1], [1, 2], [0.4, 0.3], [0.0, 0.0, 0.2], q=[3.0, 0.0, 0.0])
    # dynamic G = H_CONV c: link 0 -> 10 * 0.04 = 0.4, link 1 -> 10 * 0.03, boundary of node 2 -> 10 * 0.02
    dyn = _dyn([1, 1, 1], [0, 1, -1], [0, 1, 2], [1, 2, -1], [0.04, 0.03, 0.02], [False, False, False])
    nl = F.NetSpec(C, [0, 1], [1, 2], [0.0, 0.0], [0.0, 0.0, 0.0], q=[3.0, 0.0, 0.0], dyn=dyn)
    T0 = np.array([60.0, 40.0, 25.0])
    d = {"Tb": F.ConstDrive(5.0)}
    a, fa = F.rc_advance_arrays(lin, T0, d, 0, 3 * HOUR)
    b, fb = F.rc_advance_arrays(nl, T0, d, 0, 3 * HOUR)
    assert np.abs(a - b).max() < 1e-6
    assert abs(fa[("E", "boundary", "sink")] - fb[("E", "boundary", "sink")]) < 1e-3
    ss = F.rc_steady(nl, 5.0)
    assert np.abs(ss - F.rc_steady(lin, 5.0)).max() < 1e-6


def test_the_buoyant_rise_link_moves_heat_up_only_and_follows_the_tick_rule():
    """Two air nodes, a hot one below a cold one, closed: the heat goes up by
    the lattice's own rule (share = min(sqrt(g dx dT / T) dt/dx, 1/2) of the
    excess a tick) until they are level, energy exact. The same pair upside down
    does not move. Against the tick rule itself, run explicitly."""
    c = 0.15
    dyn = _dyn([2], [0], [0], [1], [c], [True])                 # node 0 is the lower
    for lower, upper, moves in ((50.0, 20.0, True), (20.0, 50.0, False)):
        spec = F.NetSpec([c, c], [0], [1], [0.0], [0.0, 0.0], dyn=dyn)
        T1, fl = F.rc_advance_arrays(spec, np.array([lower, upper]), {"Tb": F.ConstDrive(20.0)}, 0, int(20 / T))
        assert abs(spec.stored(T1) - spec.stored(np.array([lower, upper]))) < 1e-9
        assert (abs(T1[0] - lower) > 1.0) == moves
        if moves:
            tl, tu = lower, upper                                  # the tick rule, one tick at a time
            for _ in range(int(20 / T)):
                dT = tl - tu
                if dT > 0:
                    v = math.sqrt(9.81 * 0.05 * dT / (tu + 273.15))
                    q = min(v * T / 0.05, 0.5) * c * dT
                    tl, tu = tl - q / c, tu + q / c
            assert abs(T1[0] - tl) < 0.2 and abs(T1[1] - tu) < 0.2, (T1, tl, tu)
            assert T1[0] >= T1[1] - 1e-6                            # never past level


def test_a_gas_gas_link_grows_with_the_heat_of_the_air_it_joins():
    """The eddy term: k = k_air + 1000 AIR_C/EDDY (T - 20)+ on each side, harmonic."""
    dyn = _dyn([0], [0], [0], [1], [0.05], [False])
    spec = F.NetSpec([0.15, 0.15], [0], [1], [0.0], [0.0, 0.0], dyn=dyn)
    cold = spec.at(np.array([20.0, 20.0]), 20.0).g[0]
    hot = spec.at(np.array([60.0, 60.0]), 20.0).g[0]
    assert abs(cold - 0.026 * 0.05) < 1e-12
    assert abs(hot - (0.026 + 0.16 * 40.0) * 0.05) < 1e-9


def test_rc_net_cg_path_is_exact_in_energy_balance():
    spec = _grid_net(8, 8, 8)
    T0 = np.full(spec.n, 30.0)
    T1, fl = F.rc_advance_arrays(spec, T0, {"Tb": F.ConstDrive(0.0)}, 0, HOUR, solver="cg", tol=1e-8)
    assert abs(fl[("E", "boundary", "sink")] - (spec.stored(T0) - spec.stored(T1))) < 1e-6


def test_radiative_relinearisation_matches_the_exact_lumped_solution():
    C, a = 5000.0, 0.9 * F.SIGMA * 0.2           # J/K, W/K^4
    spec = F.NetSpec([C], [], [], [], [0.0], rad=[a])
    T0, Tinf = 527.0, 20.0                       # 800 K stone block in a 20 C yard
    for t_s in (30.0, 600.0, 3600.0):
        T1, fl = F.rc_advance_arrays(spec, np.array([T0]), {"Tb": F.ConstDrive(Tinf)}, 0, int(t_s / T))
        exact = F.lumped_rad_cool(T0, Tinf, a / C, t_s)
        assert abs(T1[0] - exact) < 2e-3 * (T0 - exact) + 1e-6, (t_s, T1[0], exact)
        assert abs(fl[("E", "boundary", "sink")] - C * (T0 - T1[0])) < 1e-6
    # the exact solution itself is the ODE (dT/dt from a small step)
    e1, e2 = F.lumped_rad_cool(T0, Tinf, a / C, 100.0), F.lumped_rad_cool(T0, Tinf, a / C, 100.001)
    Tm = 0.5 * (e1 + e2) + F.K0
    assert abs((e2 - e1) / 0.001 + a / C * (Tm ** 4 - (Tinf + F.K0) ** 4)) < 1e-3 * abs((e2 - e1) / 0.001)


def test_wall_lag_under_a_t_out_step_matches_a_finite_difference_run():
    """V1: a 0.5 m stone wall (alpha 1e-6, L^2/alpha = 70 h) is a chain of nodes;
    the outside face goes to -12 C at t = 0 and the inside face is sealed."""
    k, rho_c, Lw, n = 2.0, 2.0e6, 0.5, 25
    dx = Lw / n
    C = np.full(n, rho_c * dx)                   # J/(m2 K) per slab
    g = np.full(n - 1, k / dx)
    Gb = np.zeros(n)
    Gb[0] = 2.0 * k / dx                         # half a cell to the face, then the face at T_out
    spec = F.NetSpec(C, np.arange(n - 1), np.arange(1, n), g, Gb)
    alpha = k / rho_c
    assert abs(Lw * Lw / alpha / 3600.0 - 69.4) < 0.1
    T0 = np.full(n, 20.0)
    ts = [6 * HOUR, 24 * HOUR, 72 * HOUR]
    # FD reference: fine explicit grid, 4x finer
    m = 4 * n
    dxf = Lw / m
    Tf = np.full(m, 20.0)
    done = 0
    for t1 in ts:
        T1, _ = F.rc_advance_arrays(spec, T0, {"Tb": F.ConstDrive(-12.0)}, 0, t1)
        steps_t = (t1 - done) * T
        dt = 0.2 * dxf * dxf / alpha
        t = 0.0
        while t < steps_t:
            s = min(dt, steps_t - t)
            lap = np.empty(m)
            lap[1:-1] = Tf[2:] - 2 * Tf[1:-1] + Tf[:-2]
            lap[0] = Tf[1] - 3 * Tf[0] + 2 * (-12.0)
            lap[-1] = Tf[-2] - Tf[-1]
            Tf += alpha * s / dxf ** 2 * lap
            t += s
        done = t1
        ref = Tf.reshape(n, 4).mean(axis=1)
        assert float(np.abs(T1 - ref).max()) < 0.01 * 32.0, (t1 / HOUR, np.abs(T1 - ref).max())
    # the wall still lags: after 6 h the inner face has barely moved
    T6, _ = F.rc_advance_arrays(spec, T0, {"Tb": F.ConstDrive(-12.0)}, 0, 6 * HOUR)
    assert T6[-1] > 18.5 and T6[0] < 5.0         # erfc(1.7) of the step, doubled by the sealed face


def test_energy_conservation_with_sources_in_a_closed_net():
    spec = F.NetSpec([1000.0, 2000.0], [0], [1], [3.0], [0.0, 0.0], q=[10.0, 5.0])
    T0 = np.array([20.0, 20.0])
    T1, fl = F.rc_advance_arrays(spec, T0, {"Tb": F.ConstDrive(20.0)}, 0, 2 * HOUR)
    gained = spec.stored(T1) - spec.stored(T0)
    assert abs(gained - 15.0 * 7200.0) < 1e-6 and abs(fl[("E", "boundary", "sink")]) < 1e-6
    assert abs(fl[("E", "source", "twin")] + 15.0 * 7200.0) < 1e-9


def test_no_random_module_is_imported_by_the_scheduler_files():
    import ast
    for name in ("slowforms", "slowtime", "slowtwins", "slownet", "quiet"):
        src = open(os.path.join(_HERE, "..", "src", "voxel", name + ".py")).read()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                assert all(a.name.split(".")[0] != "random" for a in node.names), name
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] != "random", name
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                assert not (node.value.id == "np" and node.attr == "random"), name
