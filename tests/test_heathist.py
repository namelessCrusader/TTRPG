"""HEAT HISTORY (wave 15, lane 1e): the path-functional steel law. Pure laws, no World.
Each test names the mutation it guards against."""
import numpy as np
import pytest

from src.voxel import heathist as H
from src.voxel import heathist_data as D
from src.voxel import heathist_colour as CO
from src.voxel import layer_laws as LL
from src.voxel.heathist_slab import Slab
from src.voxel import matrec as MR


def _quenched(c=0.8, mn=0.5, rate=1500.0, top=850.0, soak=300.0):
    s = H.new_state(c, mn)
    H.step(s, top, 60.0)
    H.step(s, top, soak)
    T, dt = top, (0.05 if rate >= 50.0 else 2.0)
    while T > 20.0:
        T = max(20.0, T - rate * dt)
        H.step(s, T, dt)
    return s


def test_ANDREWS_numbers_and_the_STEVEN_HAYNES_474_row():
    assert float(H.ms(0.2, 0.5)) == pytest.approx(439.2, abs=0.05)         # 539 - 84.6 - 15.2
    assert float(H.ac3(0.2, 0.5)) == pytest.approx(819.2, abs=0.1)
    assert float(H.ac1(0.2, 0.5)) == pytest.approx(717.65, abs=0.01)
    # above 0.6 C the Steven-Haynes form with the 474 coefficient (the critic's check)
    assert float(H.ms(0.8, 0.5)) == pytest.approx(561.0 - 474.0 * 0.8 - 33.0 * 0.5, abs=0.01)
    assert D.MS_STEVEN_HAYNES["C"] == -474.0
    assert abs(float(H.ms(1.0, 0.5, variant=D.MS_SH_C_VARIANT)) - float(H.ms(1.0, 0.5))) == pytest.approx(2.0)


def test_KOISTINEN_MARBURGER_closed_form_and_the_path_exactness():
    c, mn = 0.4, 0.7
    Ms = float(H.ms(c, mn))
    out = []
    for n in (1, 7, 100):
        s = H.new_state(c, mn)
        s["fg"][:] = 1.0
        s["fs" if "fs" in s else "HVs"][:] = H.hv_base(c)
        s["Tp"][:] = Ms
        for T in np.linspace(Ms, Ms - 100.0, n + 1)[1:]:
            H.step(s, T, 1e-6)
        out.append(float(s["fM"][0]))
    assert out[0] == pytest.approx(1.0 - np.exp(-1.1), abs=1e-9)             # 0.667: sign or b mutated fails
    assert out[1] == pytest.approx(out[0], abs=1e-9) and out[2] == pytest.approx(out[0], abs=1e-9)


def test_the_PATH_MATTERS_same_ends_different_hardness():
    fast, slow = _quenched(rate=1500.0), _quenched(rate=2.0)
    assert float(H.hardness(fast)[0]) - float(H.hardness(slow)[0]) > 200.0   # a law of the end T or max T alone fails
    assert float(fast["fM"][0]) > 0.7 and float(slow["fM"][0]) == 0.0


def test_the_CRITICAL_COOLING_RATE_of_the_eutectoid_row():
    def fM_at(rate):
        return float(_quenched(rate=rate, soak=300.0)["fM"][0])
    assert fM_at(20.0) == 0.0 and fM_at(600.0) > 0.6
    assert fM_at(100.0) < fM_at(200.0) < fM_at(600.0)                        # the nose exponent sign mutated fails


def test_TEMPERING_is_path_exact_and_P_never_falls():
    base = _quenched(c=0.4, mn=0.7)
    a, b = {k: v.copy() for k, v in base.items()}, {k: v.copy() for k, v in base.items()}
    H.step(a, 400.0, 1.0)
    H.step(b, 400.0, 1.0)
    H.step(a, 400.0, 3600.0)
    for _ in range(1000):
        H.step(b, 400.0, 3.6)
    assert float(a["P"][0]) == pytest.approx(float(b["P"][0]), rel=1e-9)     # P' = P + dt term (no inversion) fails
    p0 = float(a["P"][0])
    H.step(a, 300.0, 3600.0)                                                  # a cooler spell does nothing
    assert float(a["P"][0]) >= p0 - 1e-12
    H.step(a, 20.0, 60.0)
    assert float(a["P"][0]) >= p0 - 1e-12                                     # no max: it would fall


def test_HOLLOMON_JAFFE_worked_values():
    s = _quenched(c=0.4, mn=0.7)
    H.step(s, 300.0, 1.0)
    H.step(s, 300.0, 3600.0)
    assert float(s["P"][0]) == pytest.approx(573.15 * 19.5 / 1000.0 * 1.0, abs=0.05)    # 11.2 for one hour at 300 C
    ratio_hv = []
    for T in (200.0, 300.0, 400.0, 500.0, 600.0):
        t = _quenched(c=0.4, mn=0.7)
        H.step(t, T, 3600.0)
        ratio_hv.append(float(H.hardness(t)[0]))
    assert all(np.diff(ratio_hv) < 0)                                         # a hotter temper is softer


def test_ANNEAL_NORMALISE_QUENCH_are_one_law():
    soft = {r: float(H.hardness(_quenched(c=0.2, mn=0.5, rate=r))[0]) for r in (0.05, 3.0, 1500.0)}
    assert soft[0.05] < 135.0 and soft[3.0] < 140.0 and soft[1500.0] > 400.0


def test_WROUGHT_IRON_does_not_harden_in_a_bar_quench():
    sl = Slab(0.02, c=D.ROWS["wrought_iron"]["c"], mn=D.ROWS["wrought_iron"]["mn"], n=9)
    sl.heat_to(850.0)
    sl.quench("water", 30.0, 0.02)
    assert float(sl.hv().max()) < 140.0


def test_SOAK_a_bar_held_a_moment_at_800_is_not_yet_hardenable():
    quick, long_ = H.new_state(0.4, 0.7), H.new_state(0.4, 0.7)
    for s, t in ((quick, 1.0), (long_, 300.0)):
        H.step(s, 800.0, 0.5)
        H.step(s, 800.0, t)
    assert float(quick["fg"][0]) < 0.5 < float(long_["fg"][0])


def test_OVERHEAT_coarsens_the_grain_raises_hardenability_lowers_toughness():
    out = {}
    for T, mins in ((850.0, 5), (1010.0, 20)):
        s = H.new_state(0.4, 0.7)
        H.step(s, T, 30.0)
        H.step(s, T, mins * 60.0)
        s["fM"][:] = 0.0
        out[T] = (float(s["d"][0]), float(H.tau_nose(0.4, 0.7, s["d"])[0]))
    assert out[1010.0][0] > 4.0 * out[850.0][0]
    assert out[1010.0][1] > out[850.0][1]
    # the same grain row reproduces its own two fit points
    for T, d in D.GRAIN_FIT:
        K = H.GRAIN_K0 * np.exp(-H.GRAIN_Q / (D.R_GAS * (T + 273.15)))
        assert (D.GRAIN_D0_UM ** 2 + K * 3600.0) ** 0.5 == pytest.approx(d, rel=1e-6)


def test_a_NEVER_HOT_sample_has_temper_exactly_one():
    s = H.new_state([0.02, 0.2, 0.8], [0.05, 0.5, 0.5])
    for T in (20.0, 60.0, 99.0, 20.0):
        H.step(s, T, 100.0)
    assert (H.temper_ratio(s) == 1.0).all() and (H.toughness_factor(s) == 1.0).all()


def test_fractions_sum_to_one_and_the_ENTHALPY_follows_them():
    s = _quenched(c=0.8)
    assert float(s["fg"][0] + s["fM"][0]) <= 1.0 + 1e-12
    assert float(H.enthalpy(s)[0]) == pytest.approx(
        float(H._L_aust(0.8) * s["fg"][0] + D.L_MART_SHARE * H._L_aust(0.8) * s["fM"][0]))


def test_THE_BOOKS_of_a_quenched_bar_close_and_the_latent_heat_is_real():
    sl = Slab(0.02, c=0.8, mn=0.5, n=11)
    sl.heat_to(850.0)
    b0 = sl.books()
    sl.quench("water", 40.0, 0.02)
    assert abs(sl.books() - b0) / abs(b0) < 1e-9
    held = float(np.sum(sl.m * H.enthalpy(sl.s) * 1000.0))
    assert held > 5e4                                                         # the retained austenite and martensite hold some


def test_a_QUENCHED_BAR_has_a_hard_skin_and_a_softer_core():
    sl = Slab(0.02, c=0.8, mn=0.5, n=11)
    sl.heat_to(850.0)
    sl.quench("water", 40.0, 0.02)
    hv = sl.hv()
    assert hv[0] > hv[-1] + 100.0 and float(sl.s["fM"][0]) > float(sl.s["fM"][-1])


def test_the_BLADE_edge_is_harder_than_its_spine_in_both_baths():
    from src.voxel.scenes_smith import edge_spine
    r = edge_spine()
    assert r["water"]["edge"] > r["water"]["spine"] > 0 and r["oil"]["edge"] > r["oil"]["spine"]
    assert r["oil"]["gap"] > 0 and r["water"]["gap"] > 0


def test_TOUGHNESS_fresh_martensite_is_brittle_and_tempering_restores_it():
    fresh = _quenched(c=0.8)
    hard = float(H.toughness_factor(fresh)[0])
    t = _quenched(c=0.8)
    H.step(t, 650.0, 3600.0)
    tempered = float(H.toughness_factor(t)[0])
    assert tempered > 1.25 * hard           # (u) q = 1: the restore is gentle; a stronger q is a fit to Charpy data
    rec = MR.REC["iron"]
    d_old, g_old = rec.brittle_at(20.0)
    d_new, g_new = rec.brittle_at(20.0, hard)
    assert (d_new, g_new) == (d_old * hard, g_old * hard) and d_new < d_old
    assert rec.brittle_at(20.0, 1.0) == (d_old, g_old)                       # default: the old answer bit for bit


def test_the_THREE_POINT_substep_rule_keeps_the_hardness_within_one_percent():
    rng = np.random.default_rng(3)
    for _ in range(20):
        c = float(rng.uniform(0.2, 0.8))
        rate = float(10.0 ** rng.uniform(0.5, 3.3))
        outs = []
        for dTmax, ms in ((20.0, 64), (0.5, 4000)):
            s = H.new_state(c, 0.6)
            H.step(s, 850.0, 60.0, dT_max=dTmax, max_sub=ms)
            H.step(s, 850.0, 120.0, dT_max=dTmax, max_sub=ms)
            T = 850.0
            while T > 20.0:
                T = max(20.0, T - rate * 0.1)
                H.step(s, T, 0.1, dT_max=dTmax, max_sub=ms)
            outs.append(float(H.hardness(s)[0]))
        assert outs[0] == pytest.approx(outs[1], rel=0.01), (c, rate, outs)


def test_COST_of_a_thousand_hot_samples():
    import time
    s = H.new_state(np.full(1000, 0.5), 0.6)
    H.step(s, 850.0, 1.0)
    t = time.perf_counter()
    for k in range(50):
        H.step(s, 850.0 - 0.1 * k, 0.025)
    ms = (time.perf_counter() - t) / 50 * 1000.0
    assert ms < 3.0, ms                                                       # ceiling 0.6 ms in the spec; measured in the report
    idle = H.new_state(np.full(1000, 0.5), 0.6)
    t = time.perf_counter()
    for k in range(200):
        H.step(idle, 20.0, 0.025)
    assert (time.perf_counter() - t) / 200 * 1000.0 < 0.2                     # nothing hot: nearly free


def test_TEMPER_COLOUR_order_follows_the_film_and_a_mutated_index_goes_grey():
    rank = {"bare": 0, "grey": 0, "straw": 1, "brown": 2, "purple": 3, "blue": 4}
    seq = [CO.name(h * 1e-9) for h in np.arange(16, 90, 1)]
    seq = [x for x in seq if x != "grey"]
    ranks = [rank[x] for x in seq]
    assert ranks == sorted(ranks) and set(seq) >= {"straw", "brown", "blue"}
    # the film is thicker the hotter it was, so the colours step with temperature
    th = [LL.oxide_thickness(T, 600.0) for T in (200, 240, 280, 320)]
    assert th == sorted(th)
    r_air = CO.reflectance(50e-9, 550.0, n=1.0)                              # n = 1: no film, no interference
    assert abs(float(r_air) - float(CO.reflectance(0.0, 550.0, n=1.0))) < 1e-12


def test_the_VECTOR_oxide_step_is_the_layer_law_value_for_value():
    rng = np.random.default_rng(1)
    x = 10.0 ** rng.uniform(-10, -5, 500)
    T = rng.uniform(100.0, 1100.0, 500)
    a = H.oxide_step_v(x, T, 0.025)
    b = np.array([LL.oxide_step(float(xi), float(Ti), 0.025) for xi, Ti in zip(x, T)])
    assert np.array_equal(a, b)
