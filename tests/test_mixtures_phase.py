"""T4 mixtures, phase (wave 13, law L14): the boil step, volatile mixtures (Van Laar,
Rayleigh), eutectics, alloys and the one energy-coupled melt/freeze/dissolve solver.
No World.

Tags: test_exact_* (conservation, exactness), test_strict_* (independent, not (u)),
test_u_* (source unverified), test_echo_* (fitted constant reproducing its fit),
test_codeonly_* (code against its own table). Where the plan's number could not be
met by an energy-conserving calculation the test says so and is marked xfail.
"""
import json
import os
import time

import numpy as np
import pytest

from src.voxel import mixdata as D
from src.voxel import mixphase as P
from src.voxel import mixsol as S
from src.voxel import mixtures as M
from src.voxel import mixvle as V
from src.voxel import mixwater as W

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCED = os.path.join(HERE, "data", "sourced_numbers.json")
TIMING = not os.environ.get("W13_NO_TIMING")
mk = M.mix
IE, IM, IW = D.INDEX["ethanol"], D.INDEX["methanol"], D.INDEX["h2o"]


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


def brines(N, seed=1, lo=0.02, hi=0.30):
    rng = np.random.default_rng(seed)
    Wt = rng.uniform(50, 2000, N)
    Sx = Wt * rng.uniform(lo, hi, N)
    c = np.zeros((N, D.n_species()))
    c[:, IW], c[:, D.INDEX["nacl"]] = Wt, Sx
    return c, W.heat_capacity(c), rng


# ===================================== the boil step ==========================================
def test_exact_boil_step_lands_on_the_boiling_point_of_what_is_left():
    for N in (16, 1600):
        c, C, rng = brines(N)
        T = W.boiling_point(c) + rng.uniform(0.1, 1.0, N)
        o = V.boil_step(c, T, C)
        steam, cn, Tn, dE, bj = o
        assert np.max(np.abs(Tn - W.boiling_point(cn))) < 1e-4, N
        assert np.all(bj > 0.0)


def test_exact_a_big_superheat_takes_the_newton_pass_and_still_lands():
    c, C, rng = brines(200)
    T = W.boiling_point(c) + rng.uniform(2.5, 30.0, 200)
    o = V.boil_step(c, T, C)
    assert np.max(np.abs(o[2] - W.boiling_point(o[1]))) < 1e-4


def test_exact_a_second_call_boils_under_a_milligram():
    c, C, rng = brines(300)
    T = W.boiling_point(c) + rng.uniform(0.1, 20.0, 300)
    o = V.boil_step(c, T, C)
    o2 = V.boil_step(o[1], o[2], C)
    assert np.max(o2[0].sum(axis=1)) < 1e-3


def test_exact_boil_step_matches_a_sixty_step_bisection_to_1e_3_K():
    c, C, rng = brines(24)
    T = W.boiling_point(c) + rng.uniform(0.5, 8.0, 24)
    o = V.boil_step(c, T, C)
    L = D.LATENT_SIM
    lo = np.zeros(24)
    hi = 0.999 * c[:, IW]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cm = c.copy()
        cm[:, IW] -= mid
        f = (T - W.boiling_point(cm)) * C - L * mid
        lo = np.where(f > 0.0, mid, lo)
        hi = np.where(f > 0.0, hi, mid)
    dm = 0.5 * (lo + hi)
    assert np.max(np.abs(T - L * dm / C - o[2])) < 1e-3


def test_exact_boil_step_conserves_mass_and_energy():
    c, C, rng = brines(500)
    cry0 = np.zeros_like(c)
    T = W.boiling_point(c) + rng.uniform(0.1, 15.0, 500)
    o = V.boil_step(c, T, C, crystal=cry0)
    steam, cn, Tn, dE, bj = o
    assert np.allclose(steam + cn + (o.crystal_new - cry0), c, rtol=1e-12, atol=1e-12)
    assert np.allclose(dE, C * (Tn - T), rtol=1e-12)
    assert np.allclose(dE, -bj + o.crystal_J, rtol=1e-12, atol=1e-9)         # joules: the steam took boiled_J
    assert np.allclose(bj, steam[:, IW] * D.LATENT_SIM, rtol=1e-12)
    assert np.all(steam[:, IW] <= 0.999 * c[:, IW])


def test_exact_pure_water_boils_by_the_existing_law():
    c = np.zeros((3, D.n_species()))
    c[:, IW] = (100.0, 1000.0, 5.0)
    C = c[:, IW] * 4.18
    T = np.array([104.0, 101.0, 130.0])
    o = V.boil_step(c, T, C, tb_water=D.BOIL_SIM)
    want = np.minimum(0.999 * c[:, IW], (T - 100.0) * C / D.LATENT_SIM)
    assert np.allclose(o[0][:, IW], want, rtol=1e-12)
    assert np.allclose(o[2], T - want * D.LATENT_SIM / C, rtol=1e-12)


def test_exact_a_boiled_dry_brine_keeps_its_salt_as_crystal_exactly():
    c = mk(h2o=100.0, nacl=30.0)
    cry = np.zeros_like(c)
    T = 104.0
    for _ in range(40):
        C = W.heat_capacity(c) + 1000.0
        o = V.boil_step(c, T + 25.0, C, crystal=cry)
        c, cry, T = o[1], o.crystal_new, o[2]
        assert np.isfinite(T) and np.all(np.isfinite(c))
    salt = c[D.INDEX["nacl"]] + cry[D.INDEX["nacl"]]
    assert salt == pytest.approx(30.0, rel=1e-12)
    assert cry[D.INDEX["nacl"]] == pytest.approx(30.0, rel=1e-9) and c[IW] < 1e-9


def test_exact_a_syrup_at_saturation_is_not_crystallised():
    c = mk(h2o=100.0, sucrose=700.0)             # past saturation at its boiling point but inside ratio 1.3
    tb = float(W.boiling_point(c))
    C = float(W.heat_capacity(c))
    o = V.boil_step(c, tb + 5.0, C)
    ratio = o[1][D.INDEX["sucrose"]] / (float(S.solubility_water("sucrose", o[2])) * o[1][IW] / 100.0)
    assert o.crystal_new[D.INDEX["sucrose"]] == 0.0
    assert 1.0 < ratio < 1.3, ratio              # supersaturated but, below the ratio, still dissolved
    o2 = V.boil_step(mk(h2o=100.0, sucrose=350.0), 110.0, 600.0)
    assert o2.crystal_new[D.INDEX["sucrose"]] == 0.0


def test_exact_nacl_past_its_nucleation_ratio_crystallises_in_the_call():
    c = mk(h2o=100.0, nacl=60.0)                 # 60 g in 100 g water: far past 36
    o = V.boil_step(c, 104.0, 500.0)
    assert o.crystal_new[D.INDEX["nacl"]] > 20.0
    assert o[1][D.INDEX["nacl"]] + o.crystal_new[D.INDEX["nacl"]] == pytest.approx(60.0, rel=1e-12)
    assert o.crystal_J > 0.0                     # the crystal gives its heat of solution back


@pytest.mark.skipif(not TIMING, reason="W13_NO_TIMING set")
def test_exact_boil_step_time_16_cells_and_1600_cells():
    c16, C16, r16 = brines(16, seed=5, hi=0.25)
    T16 = W.boiling_point(c16) + r16.uniform(0.1, 1.0, 16)
    c1600, C1600, r = brines(1600, seed=5, hi=0.25)
    T1600 = W.boiling_point(c1600) + r.uniform(0.1, 1.0, 1600)
    # the box is shared with wave 12: scale the budget by how slow a fixed numpy
    # workload is right now (quiet box: 6.3 ms for 1000 exp calls on 1600 floats)
    a = np.random.default_rng(0).random(1600)
    slow = max(1.0, best(lambda: [np.exp(a) for _ in range(1000)], 5) / 6.3e-3)
    t16 = best(lambda: V.boil_step(c16, T16, C16), 60)
    t1600 = best(lambda: V.boil_step(c1600, T1600, C1600), 60)
    assert t16 < 0.6e-3 * slow, (t16, slow)
    assert t1600 < 1.5e-3 * slow, (t1600, slow)


# ===================================== volatile mixtures ======================================
def _vle_y_minus_x(x):
    c = mk(h2o=(1 - x) * D.SPECIES["h2o"].M, ethanol=x * D.SPECIES["ethanol"].M)
    T, y, L = V.bubble_point(c)
    me, mw = y[IE] / D.SPECIES["ethanol"].M, y[IW] / D.SPECIES["h2o"].M
    return me / (me + mw) - x, float(T)


def test_exact_van_laar_y_minus_x_changes_sign_exactly_once_near_the_azeotrope():
    xs = np.linspace(0.8, 0.99, 191)
    d = np.array([_vle_y_minus_x(x)[0] for x in xs])
    flips = np.nonzero((d[:-1] > 0) & (d[1:] <= 0))[0]
    assert len(flips) == 1 and not np.any((d[:-1] <= 0) & (d[1:] > 0))


def test_echo_the_van_laar_azeotrope_is_at_the_real_one():
    lo, hi = 0.85, 0.95
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if _vle_y_minus_x(mid)[0] > 0.0:
            lo = mid
        else:
            hi = mid
    xaz = 0.5 * (lo + hi)
    assert xaz == pytest.approx(0.894, abs=0.01)
    assert _vle_y_minus_x(xaz)[1] == pytest.approx(78.15, abs=0.2)
    assert D.src_of("vanlaar", "ethanol") == "fit:crc-azeotrope"


def test_u_thirteen_point_textbook_vle_table():
    tab = [(0.0190, 0.1700, 95.5), (0.0721, 0.3891, 89.0), (0.0966, 0.4375, 86.7), (0.1238, 0.4704, 85.3),
           (0.1661, 0.5089, 84.1), (0.2337, 0.5445, 82.7), (0.3273, 0.5826, 81.5), (0.3965, 0.6122, 80.7),
           (0.5079, 0.6564, 79.8), (0.5732, 0.6841, 79.3), (0.6763, 0.7385, 78.74), (0.7472, 0.7815, 78.41),
           (0.8943, 0.8943, 78.15)]
    for x, y, T in tab:
        d, Tm = _vle_y_minus_x(x)
        assert abs(Tm - T) < 0.8, (x, Tm, T)
        assert abs((d + x) - y) < 0.02, (x, d + x, y)


def test_u_pure_ethanol_and_the_ten_percent_wash():
    assert float(V.bubble_point(mk(ethanol=100.0))[0]) == pytest.approx(78.3, abs=0.3)
    D.refuse_u("antoine", "ethanol")
    c = mk(h2o=92.0, ethanol=8.0)
    T, y, L = V.bubble_point(c)
    assert float(T) == pytest.approx(92.8, abs=0.8)
    assert 0.44 <= y[IE] <= 0.48                                       # vapour 44-48 wt% ethanol
    assert float(L) == pytest.approx(1600.0, rel=0.03)                 # L_eff of the vapour
    assert y.sum() == pytest.approx(1.0, abs=1e-12)
    assert float(W.abv_to_grams(10.0, 100.0)) == pytest.approx(8.0, abs=0.2)


def test_u_rayleigh_distillation_of_the_wash():
    c = mk(h2o=92.0, ethanol=8.0)
    d, r = V.rayleigh(c, 0.97)
    assert float(W.abv_of(d)) == pytest.approx(29.0, abs=3.0)
    n0 = c / np.array([D.SPECIES[x].M for x in D.ORDER])
    n1 = (d + r) / np.array([D.SPECIES[x].M for x in D.ORDER])
    assert np.allclose(n0, n1, rtol=1e-9, atol=1e-12)                 # moles conserved
    assert d[IE] == pytest.approx(0.97 * 8.0, rel=0.02)
    assert np.all(d >= 0) and np.all(r >= -1e-12)
    xd = (d[IE] / 46.068) / (d[IE] / 46.068 + d[IW] / 18.01528)
    assert xd < 0.894                                                   # never past the azeotrope
    # the first drops are richer than the average distillate, and the pot warms
    d1, r1 = V.rayleigh(c, 0.05)
    assert float(W.abv_of(d1)) > float(W.abv_of(d)) + 10.0
    assert float(V.bubble_point(r)[0]) > float(V.bubble_point(c)[0]) + 5.0
    # more steps change little (the midpoint stepping has converged)
    assert float(W.abv_of(V.rayleigh(c, 0.97, steps=160)[0])) == pytest.approx(float(W.abv_of(d)), abs=0.5)


def test_u_methanol_is_richer_in_the_head_of_a_spirit():
    """Methanol boils lower, so in a strong spirit its share in the first vapour is
    above its share in the pot. (In a dilute aqueous wash the (u) Van Laar pair for
    methanol, 0.58 / 0.56, makes ethanol the more volatile: reported to the user.)"""
    c = mk(h2o=10.0, ethanol=80.0, methanol=10.0)
    T, y, L = V.bubble_point(c)
    assert y[IM] / y[IE] > c[IM] / c[IE]
    d, r = V.rayleigh(c, 0.3)
    assert d[IM] / d[IE] > c[IM] / c[IE] > r[IM] / r[IE]


def test_exact_condensing_the_steam_returns_the_booked_joules():
    c = mk(h2o=92.0, ethanol=8.0)
    o = V.boil_step(c, 96.0, 100 * 3.8)
    steam, cn, Tn, dE, bj = o
    assert steam[IE] > 0.0 and V.condense_J(steam) == pytest.approx(bj, rel=1e-9)
    assert dE == pytest.approx(-bj, rel=1e-9)
    assert abs(Tn - float(V.bubble_point(cn)[0])) < 1e-3
    # the energy per gram of first vapour is about 29 percent below water alone
    assert bj / steam.sum() == pytest.approx(1600.0, rel=0.04)
    assert 1.0 - bj / steam.sum() / D.LATENT_SIM == pytest.approx(0.29, abs=0.03)
    assert np.allclose(steam + cn, c, rtol=1e-12)


# ===================================== alloys and eutectics =====================================
def solder(sn):
    return mk(pb=100.0 - sn, sn=sn)


def test_strict_the_nacl_eutectic_is_computed_not_typed():
    TE, wE = P.eutectic("nacl")
    assert TE == pytest.approx(-21.1, abs=0.4) and wE == pytest.approx(23.3, abs=0.5)
    D.refuse_u("eutectic_ref", "nacl")                           # the reference row is sourced (crc)
    for solute in ("kcl", "sucrose"):                            # computed too, compared loosely (u)
        T, w = P.eutectic(solute)
        refT, refw = D.row("eutectic_ref", solute)
        assert abs(T - refT) < 2.0 and abs(w - refw) < 1.5, (solute, T, w)


def test_u_the_nacl_eutectic_against_the_sourced_value(record_property):
    v, status = sourced("nacl.eutectic")
    record_property("sourced_status", status)
    if v is None:
        pytest.skip(f"sourced_numbers: {status}")
    TE, wE = P.eutectic("nacl")
    assert TE == pytest.approx(v["T_C"], abs=0.3) and wE == pytest.approx(v["wt_pct"], abs=0.5)


def test_exact_freeze_split_conserves_water_and_solute_and_ice_grows_as_it_cools():
    c = mk(h2o=100.0, nacl=10.0)
    ices = []
    for T in (-7.0, -10.0, -15.0, -20.0):
        ice, liq, lc = P.freeze_split(c, T)
        assert ice + lc[IW] == pytest.approx(100.0, rel=1e-12)
        assert lc[D.INDEX["nacl"]] == pytest.approx(10.0, rel=1e-12)
        assert liq == pytest.approx(lc.sum(), rel=1e-12)
        ices.append(float(ice))
    assert all(b > a for a, b in zip(ices, ices[1:]))
    ice_cold, liq_cold, lc_cold = P.freeze_split(c, -21.5)
    assert liq_cold == 0.0 and ice_cold == 100.0                  # below T_E nothing is liquid
    assert P.freeze_split(c, -3.0)[0] == 0.0                      # no ice above the freezing point of the mix (-6.7 C)


def test_exact_freeze_split_of_two_solutes_keeps_the_ledger():
    c = mk(h2o=100.0, nacl=5.0, sucrose=20.0)
    ice, liq, lc = P.freeze_split(c, -8.0)
    assert 0.0 < ice < 100.0
    assert ice + lc[IW] == pytest.approx(100.0, rel=1e-12)
    assert lc[D.INDEX["nacl"]] == pytest.approx(5.0, rel=1e-12) and lc[D.INDEX["sucrose"]] == pytest.approx(20.0, rel=1e-12)
    # the liquid left sits on the ice line: its aw equals the ice's
    assert float(W.ln_aw(lc, -8.0)) == pytest.approx(float(W.ln_ice_ratio(-8.0)), abs=1e-6)


# -- lead-tin solder
def test_u_solder_60_40():
    c = solder(60.0)
    Ts, Tl = P.solidus_liquidus(c)
    assert float(Ts) == 183.0 and float(Tl) == pytest.approx(190.0, abs=3.0)
    assert float(P.liquid_fraction(c, 182.9)) == 0.0                    # solid below 183
    assert float(P.liquid_fraction(c, 183.1)) == pytest.approx(0.956, abs=0.02)
    f = [float(P.liquid_fraction(c, T)) for T in (183.0, 184.0, 186.0, 188.0, 190.0, 195.0)]
    assert all(b >= a for a, b in zip(f, f[1:])) and f[-1] == 1.0


def test_u_solder_63_37_is_nearly_eutectic():
    Ts, Tl = P.solidus_liquidus(solder(63.0))
    assert float(Tl) - float(Ts) < 2.0
    assert float(P.liquid_fraction(solder(63.0), 184.4)) == 1.0
    assert float(P.solidus_liquidus(solder(61.9))[1]) == pytest.approx(183.0, abs=0.01)


def test_u_solder_50_50_and_95_percent_tin():
    assert float(P.solidus_liquidus(solder(50.0))[1]) == pytest.approx(216.0, abs=3.0)
    c = solder(95.0)
    assert float(P.solidus_liquidus(c)[0]) == pytest.approx(183.0, abs=0.01)    # first liquid at 183 C
    assert float(P.liquid_fraction(c, 182.9)) == 0.0 and float(P.liquid_fraction(c, 183.1)) > 0.0
    assert float(P.liquid_fraction(c, 224.0)) == 1.0


def test_u_copper_alloys_and_slag():
    for name, comp, ts, tl in (("bronze", mk(cu=90.0, sn=10.0), 854.0, 1000.0),
                               ("brass", mk(cu=70.0, zn=30.0), 915.0, 955.0)):
        Ts, Tl = P.solidus_liquidus(comp)
        assert float(Ts) == pytest.approx(ts, abs=5.0) and float(Tl) == pytest.approx(tl, abs=5.0), name
    D.refuse_u("sys_cusn", "c90500")                                     # the CDA data-sheet row is sourced
    fay = mk(feo=70.5, sio2=29.5)
    assert float(P.solidus_liquidus(fay)[1]) == pytest.approx(1205.0, abs=5.0)
    lows = [float(P.solidus_liquidus(mk(feo=100 - w, sio2=w))[0]) for w in np.linspace(0, 60, 121)]
    assert min(lows) == pytest.approx(1177.0, abs=5.0)
    assert float(P.solidus_liquidus(mk(feo=100.0))[1]) == pytest.approx(1377.0, abs=1.0)  # wuestite alone


def test_u_iron_carbon():
    TE, wE = P.eutectic("fe-c")
    assert TE == pytest.approx(1147.0, abs=1.0) and wE == pytest.approx(4.3, abs=0.1)
    pts = P.fe_c_points(np.array([0.4, 0.0, 1.0, 3.0, 4.3]))
    assert np.all(pts["Ae1"][:5] == 727.0)
    assert float(pts["Ae3"][0]) == pytest.approx(790.0, abs=15.0)
    assert np.isnan(pts["Ae3"][2]) and float(pts["Acm"][2]) > 727.0       # Ae3 below 0.77, Acm above
    assert float(pts["liquidus"][1]) == 1538.0 and float(pts["solidus"][4]) == 1147.0
    assert float(pts["liquidus"][3]) < float(pts["liquidus"][2]) < float(pts["liquidus"][1])
    assert np.isnan(P.fe_c_points(5.0)["liquidus"])


def test_exact_phase_assembly_parts_sum_to_the_bulk():
    for sn, T in ((20.0, 200.0), (20.0, 250.0), (60.0, 183.5), (80.0, 200.0), (95.0, 190.0)):
        c = solder(sn)
        f, liq, sol, cry = P.phase_assembly(c, T)
        assert np.max(np.abs(liq + sol + cry - c)) <= 1e-12 * c.sum()
        assert np.all(liq >= -1e-15) and np.all(sol >= -1e-12)
    for c, T in ((mk(h2o=100.0, nacl=10.0), -10.0), (mk(h2o=100.0, nacl=40.0), 20.0), (mk(h2o=100.0, nacl=10.0), -25.0)):
        f, liq, ice, cry = P.phase_assembly(c, T)
        assert np.max(np.abs(liq + ice + cry - c)) <= 1e-12 * c.sum()


def test_u_a_half_melted_alloy_gives_a_liquid_of_another_composition():
    c = solder(20.0)                                                  # 20 wt% Sn held at 200 C
    f, liq, sol, cry = P.phase_assembly(c, 200.0)
    wl = liq[D.INDEX["sn"]] / liq.sum()
    ws = sol[D.INDEX["sn"]] / sol.sum()
    assert wl * 100.0 == pytest.approx(56.0, abs=3.0)                 # richer in tin than the bulk
    assert ws < 0.20 and 0.0 < f < 0.2                                 # the solid is lead rich
    assert liq.sum() == pytest.approx(f * 100.0, rel=1e-12)
    assert np.all(cry == 0.0)


# ===================================== the coupled step ======================================
def test_exact_a_pure_metal_equals_the_old_pin_to_1e_12():
    tm, lat = D.pure_metal_points()["sn"]
    for m, T, C in ((100.0, 240.0, 22.8), (50.0, 233.0, 100.0), (10.0, 400.0, 30.0)):
        c = mk(sn=m)
        o = P.phase_step(c, 0.0, T, C)
        dm_old = min(m, (T - tm) * C / lat)                             # the sim's old pin
        assert o[0] * m == pytest.approx(dm_old, rel=1e-12)
        assert o[1] == pytest.approx(T - dm_old * lat / C, rel=1e-12)
        assert o[2] == pytest.approx(-dm_old * lat, rel=1e-9)
    # freezing back is the same law in the other direction
    c = mk(sn=100.0)
    o = P.phase_step(c, 1.0, 200.0, 22.8)
    assert o[0] == 0.0 or o[1] <= tm + 1e-9


def test_exact_phase_step_energy_before_and_after_are_equal():
    cases = [(solder(60.0), 0.0, 150.0, 20.0), (solder(60.0), 0.0, 190.0, 20.0), (solder(20.0), 0.0, 250.0, 22.0),
             (solder(30.0), 1.0, 170.0, 22.0), (mk(cu=90.0, sn=10.0), 0.0, 900.0, 40.0),
             (mk(feo=70.0, sio2=30.0), 0.0, 1190.0, 60.0)]
    for c, f0, T, C in cases:
        o = P.phase_step(c, f0, T, C)
        e0 = P.energy_content(c, f0, T, C)
        e1 = P.energy_content(c, o[0], o[1], C)
        assert e1 == pytest.approx(e0, rel=1e-9, abs=1e-6), (c.sum(), T)
        assert o[2] == pytest.approx(C * (o[1] - T), rel=1e-12)


def test_exact_phase_step_f_is_monotone_in_heat_and_idempotent():
    c = solder(60.0)
    C = 20.0
    T, f = 150.0, 0.0
    fs, Ts = [], []
    for _ in range(300):
        T = T + 40.0 / C                                              # 40 J in
        o = P.phase_step(c, f, T, C)
        f, T = float(o[0]), float(o[1])
        fs.append(f)
        Ts.append(T)
    assert all(b >= a - 1e-15 for a, b in zip(fs, fs[1:])) and fs[-1] == 1.0
    assert all(b >= a - 1e-12 for a, b in zip(Ts, Ts[1:]))
    # the mush pins at the eutectic first
    assert any(abs(t - 183.0) < 1e-9 and 0.0 < f_ < 1.0 for t, f_ in zip(Ts, fs))
    o2 = P.phase_step(c, f, T, C)
    assert abs(o2[0] - f) < 1e-9 and abs(o2[1] - T) < 1e-9
    # a mid-mush state is unchanged by a second call
    o = P.phase_step(c, 0.0, 186.0, C)
    o2 = P.phase_step(c, o[0], o[1], C)
    assert abs(o2[0] - o[0]) < 1e-9 and abs(o2[1] - o[1]) < 1e-9


def test_u_the_eutectic_composition_pins_and_one_side_has_a_mush():
    C, m = 20.0, 100.0
    eut = solder(61.9)
    o = P.phase_step(eut, 0.0, 190.0, C)
    assert o[1] == pytest.approx(183.0, abs=1e-9) and 0.0 < o[0] < 1.0 and o.pinned
    side = solder(40.0)
    assert float(P.solidus_liquidus(side)[1]) - float(P.solidus_liquidus(side)[0]) > 40.0
    ts = [float(P.phase_step(side, 0.0, T, C)[1]) for T in (200.0, 230.0)]
    assert ts[0] < 200.0 and ts[1] < 230.0                            # heat is taken by the melt


def test_exact_a_bronze_mush_stalls_in_the_melting_range():
    c = mk(cu=88.0, sn=12.0)
    C, T, f = 38.0, 800.0, 0.0
    states = []
    for _ in range(120):
        T += 500.0 / C
        o = P.phase_step(c, f, T, C)
        f, T = float(o[0]), float(o[1])
        states.append((T, f))
    assert states[0][1] == 0.0 and states[-1][1] == 1.0
    mush = [t for t, f_ in states if 0.0 < f_ < 1.0]
    assert min(mush) >= 830.0 - 1e-6 and max(mush) <= 985.0 + 1e-6


def test_u_salt_on_ice_settles_at_the_eutectic_in_one_call():
    c = mk(h2o=100.0, nacl=10.0)
    C = float(W.heat_capacity(c, ice_g=100.0))
    o = P.phase_step(c, 0.0, 0.0, C, solid_old=(100.0, 10.0))
    TE, wE = P.eutectic("nacl")
    assert o[1] == pytest.approx(-21.14, abs=0.2) and o[1] >= TE - 1e-12     # never below T_E
    assert bool(o.pinned)
    ice, crystal = o.solid_new
    assert 0.0 < crystal < 10.0 and ice > 70.0
    # the liquid left is the eutectic liquid
    f, liq, ice_v, cry = P.phase_assembly(c, o[1], f=o[0])
    assert liq[D.INDEX["nacl"]] / liq.sum() == pytest.approx(wE / 100.0, abs=1e-6)
    # idempotent
    C2 = float(W.heat_capacity(c, ice_g=ice))
    o2 = P.phase_step(c, o[0], o[1], C2, solid_old=o.solid_new)
    assert abs(o2[0] - o[0]) < 1e-9 and abs(o2[1] - o[1]) < 1e-9
    # energy before and after
    e0 = P.energy_content(c, 0.0, 0.0, C, (100.0, 10.0))
    e1 = P.energy_content(c, o[0], o[1], C, o.solid_new)
    assert e1 == pytest.approx(e0, rel=1e-9, abs=1e-6)
    # the control without salt melts at 0 C
    o0 = P.phase_step(mk(h2o=100.0), 0.0, 3.0, 209.0, solid_old=(100.0, 0.0))
    assert o0[1] == pytest.approx(0.0, abs=1e-12) and 0.0 < o0[0] < 1.0


def test_u_salt_on_ice_leaves_85_g_of_ice():
    """N3, accepted by the user on 2026-10-30: 100 g ice + 10 g NaCl at 0 C, no heat from outside,
    ends at the eutectic with 85 +- 1 g of ice. (The plan said 82.7 +- 2. Its prototype used
    Lfus(T), which is already Kirchhoff, together with the cell's liquid cp, which counts the
    4.18 against 2.09 J/g/K difference twice: an arithmetic slip, not physics.)

    THE ENERGY BALANCE, typed here from the plan's constants and the SOURCED eutectic (-21.12 C at
    23.31 wt percent, sourced_numbers.json nacl.eutectic), not from the module:
      the cell is C = 100 g x 2.09 + 10 g x 0.86 = 217.6 J/K;
      from 0 C down to T_E it gives 217.6 x 21.12 = 4596 J;
      a gram of the eutectic liquid (w_E = 0.2331 NaCl) costs
        (1 - 0.2331) Lfus(T_E) + 0.2331 x 66 = 0.7669 x 289.41 + 15.38 = 237.33 J,
        with Lfus(T_E) = 333.55 - 2.09 x 21.12 = 289.41 J/g (Kirchhoff) and the heat of
        solution of NaCl 66 J/g (u);
      so 4596 / 237.33 = 19.36 g of liquid forms: 14.85 g of ice melts and 4.51 g of salt
      dissolves, leaving 85.15 g of ice and 5.49 g of salt crystals.
    The module computes its own eutectic (-21.159 C, 23.30 wt percent, from the ice and
    solubility lines) and leaves 85.12 g: 0.03 g from the balance above."""
    TE, wE = -21.12, 0.2331
    C = 100.0 * 2.09 + 10.0 * 0.86
    Lfus = 333.55 - 2.09 * -TE
    cost = (1.0 - wE) * Lfus + wE * 66.0
    liquid = C * -TE / cost
    ice_by_hand = 100.0 - (1.0 - wE) * liquid
    assert (C, round(liquid, 2), round(ice_by_hand, 2)) == (pytest.approx(217.6), 19.36, 85.15)
    c = mk(h2o=100.0, nacl=10.0)
    o = P.phase_step(c, 0.0, 0.0, float(W.heat_capacity(c, ice_g=100.0)), solid_old=(100.0, 10.0))
    assert o.solid_new[0] == pytest.approx(85.0, abs=1.0)
    assert o.solid_new[0] == pytest.approx(ice_by_hand, abs=0.1)          # and it IS the balance above


def test_codeonly_salt_on_ice_energy_balance_value():
    c = mk(h2o=100.0, nacl=10.0)
    o = P.phase_step(c, 0.0, 0.0, float(W.heat_capacity(c, ice_g=100.0)), solid_old=(100.0, 10.0))
    ice, crystal = o.solid_new
    assert ice == pytest.approx(85.1, abs=0.4) and crystal == pytest.approx(5.5, abs=0.4)
    # by hand: C_ice (0 - T_E) = Lambda, Lambda = f m [(1 - w_E) Lfus(T_E) + w_E Hsol]
    TE, wE = P.eutectic("nacl")
    L = (1 - wE / 100) * float(W.Lfus(TE)) + wE / 100 * 66.0
    assert o[0] * 110.0 * L == pytest.approx(217.6 * -TE, rel=1e-6)


def test_exact_salt_on_ice_never_goes_below_the_eutectic_whatever_the_salt():
    TE, _ = P.eutectic("nacl")
    for salt in (1.0, 5.0, 10.0, 25.0, 40.0):
        c = mk(h2o=100.0, nacl=salt)
        o = P.phase_step(c, 0.0, 0.0, float(W.heat_capacity(c, ice_g=100.0)), solid_old=(100.0, salt))
        assert o[1] >= TE - 1e-9
        ice, crystal = o.solid_new
        assert ice >= -1e-12 and crystal >= -1e-12 and ice <= 100.0 + 1e-12
    # a little salt melts only a little ice and stops warm
    c = mk(h2o=100.0, nacl=1.0)
    o = P.phase_step(c, 0.0, 0.0, float(W.heat_capacity(c, ice_g=100.0)), solid_old=(100.0, 1.0))
    assert o[1] > TE + 3.0 and o.solid_new[1] == 0.0                    # all the salt dissolves


def test_exact_ice_grows_monotonically_as_heat_is_removed():
    c = mk(h2o=100.0, nacl=10.0)
    T, f, solids = 5.0, 1.0, (0.0, 0.0)
    ices, Ts = [], []
    for _ in range(300):
        C = float(W.heat_capacity(c, ice_g=solids[0]))
        o = P.phase_step(c, f, T - 90.0 / C, C, solid_old=solids)         # 90 J out
        f, T, solids = float(o[0]), float(o[1]), tuple(o.solid_new)
        ices.append(solids[0])
        Ts.append(T)
    assert all(b >= a - 1e-12 for a, b in zip(ices, ices[1:])) and ices[-1] > 50.0
    assert min(Ts) >= P.eutectic("nacl")[0] - 1e-9 or Ts[-1] < P.eutectic("nacl")[0]
    assert all(b <= a + 1e-12 for a, b in zip(Ts, Ts[1:]))


def test_exact_cooling_a_brine_to_minus_25_and_warming_back_returns_the_same_totals():
    c = mk(h2o=100.0, nacl=10.0)
    Cref = 100 * 2.09 + 10 * 0.86
    T0 = 5.0
    T, f, solids = T0, 1.0, (0.0, 0.0)
    q_out = 0.0
    for _ in range(900):
        C = float(W.heat_capacity(c, ice_g=solids[0]))
        step = 200.0                      # the host cools by the cell's CURRENT heat capacity
        o = P.phase_step(c, f, T - step / C, C, solid_old=solids)
        q_out += step
        f, T, solids = float(o[0]), float(o[1]), tuple(o.solid_new)
        if T <= -25.0 + 1e-9 and f == 0.0:
            break
    assert f == 0.0 and solids[0] == pytest.approx(100.0, abs=1e-9) and solids[1] == pytest.approx(10.0, abs=1e-9)
    for _ in range(1200):                                                  # warm back with the same heat
        C = float(W.heat_capacity(c, ice_g=solids[0]))
        step = min(200.0, q_out)
        if step <= 0.0:
            break
        o = P.phase_step(c, f, T + step / C, C, solid_old=solids)
        q_out -= step
        f, T, solids = float(o[0]), float(o[1]), tuple(o.solid_new)
    assert q_out == pytest.approx(0.0, abs=1e-9)
    assert T == pytest.approx(T0, abs=1e-6) and f == pytest.approx(1.0, abs=1e-9)
    assert solids[0] == pytest.approx(0.0, abs=1e-9) and solids[1] == pytest.approx(0.0, abs=1e-9)


def test_u_a_potassium_nitrate_cold_pack_stops_when_the_solubility_does():
    c = mk(h2o=100.0, kno3=31.6)
    C = float(W.heat_capacity(c))
    o = P.phase_step(c, 100.0 / 131.6, 20.0, C, solid_old=(0.0, 31.6))
    assert o[1] == pytest.approx(5.7, abs=3.0)
    left = o.solid_new[1]
    assert 0.0 < left < 31.6                                              # part of the salt dissolves, part does not
    # at that temperature the dissolved share is the saturation: solubility, not heat, is the limit
    sat = float(S.solubility_water("kno3", o[1]))
    assert (31.6 - left) == pytest.approx(sat, rel=0.02)
    e0 = P.energy_content(c, 100.0 / 131.6, 20.0, float(P.reference_heat_capacity(c, C, (0.0, 31.6))), (0.0, 31.6))
    e1 = P.energy_content(c, o[0], o[1], float(P.reference_heat_capacity(c, C, (0.0, 31.6))), o.solid_new)
    assert e1 == pytest.approx(e0, rel=1e-9, abs=1e-6)


def test_exact_phase_step_runs_on_many_cells_at_once():
    cs = np.stack([solder(60.0), solder(20.0), solder(61.9)])
    sysm = P.get_system("pb-sn")
    o = P.phase_step(cs, np.zeros(3), np.array([190.0, 250.0, 190.0]), np.full(3, 20.0), system=sysm)
    one = [P.phase_step(cs[i], 0.0, t, 20.0, system=sysm) for i, t in enumerate((190.0, 250.0, 190.0))]
    assert np.allclose(o[0], [x[0] for x in one], rtol=1e-12) and np.allclose(o[1], [x[1] for x in one], rtol=1e-12)


def test_exact_system_for_picks_the_pair_and_refuses_three_species():
    assert P.system_for(mk(pb=1.0, sn=1.0)).name == "pb-sn"
    assert P.system_for(mk(h2o=1.0, nacl=1.0)).name == "h2o-nacl"
    assert P.system_for(mk(sn=1.0)).name == "sn"
    assert P.system_for(mk(h2o=1.0, nacl=1.0, sucrose=1.0)) is None
    with pytest.raises(ValueError):
        P.phase_step(mk(h2o=1.0, nacl=1.0, sucrose=1.0), 0.0, 0.0, 1.0)


def test_exact_the_aqueous_systems_are_built_from_the_water_activity_model():
    s = P.get_system("h2o-nacl")
    assert s.pure_solids and s.rule == "eutectic" and s.TE == pytest.approx(-21.16, abs=0.1)
    ice_w = np.interp(-9.94, s.tl[::-1], s.wl[::-1])
    assert ice_w * 100.0 == pytest.approx(14.0, abs=0.5)                # CRC: 14 wt% freezes at -9.94 C
    assert P.get_system("h2o-ethanol").rule == "ice_line"


def test_u_latent_heat_of_a_point_is_mass_weighted_from_the_pure_latents():
    c = solder(40.0)
    assert float(P.latent(c, 200.0)) == pytest.approx(0.6 * 23.0 + 0.4 * 59.0)
    assert float(P.latent(mk(h2o=100.0, nacl=10.0), -10.0)) < float(W.Lfus(-10.0))   # the salt branch is cheaper
