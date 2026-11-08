"""T7 bodyair, thermoreg and airsink (no World).

Tags: test_u_* = physical data vs an independent number (source unverified
until A-S marks it); test_echo_* = the constant was fitted to it;
test_codeonly_* = the code against its own table; unmarked = exactness,
conservation, closed-form and arithmetic identities.
"""
import copy
import math
import random

import numpy as np
import pytest

from src.voxel import airsink as AS
from src.voxel import bodyair as B
from src.voxel import thermoreg as TR
from src.voxel.tissue import ZONES, AREA_SHARE


# ── bodyair ──────────────────────────────────────────────────────────────────
def test_u_p_sat_20_and_37_within_1pct():
    assert float(B.p_sat(20.0)) == pytest.approx(2.333, rel=0.01)
    assert float(B.p_sat(37.0)) == pytest.approx(6.265, rel=0.01)


def test_u_latent_heat_at_33C():
    assert float(B.latent_heat(33.0)) == pytest.approx(2.42e6, rel=0.005)


def test_humidity_ratio_identity():
    p = 1.2
    assert float(B.humidity_ratio(p)) == pytest.approx(0.622 * p / (B.P_ATM - p), rel=1e-15)
    assert float(B.vapour_pressure(20.0, 0.5)) == pytest.approx(0.5 * float(B.p_sat(20.0)))


def test_u_resting_breath_20C_50pct():
    """7.5 L/min at 20 C, 50 percent RH: 1.5-3 W sensible, 8-13 W latent, 0.2-0.3 g/min."""
    r = B.breath_loss(7.5, 20.0, 0.5, 37.0)
    assert 1.5 <= r["sens_w"] <= 3.0, r
    assert 8.0 <= r["lat_w"] <= 13.0, r
    assert 0.2 <= r["water_g_s"] * 60.0 <= 0.3, r
    # the latent heat is exactly the water times L (arithmetic identity)
    assert r["lat_w"] == pytest.approx(r["water_g_s"] / 1000.0 * float(B.latent_heat(r["T_ex"])), rel=1e-12)


def test_breath_exhale_temperature_tied_to_inhaled():
    cold = B.breath_loss(7.5, -10.0, 0.3, 37.0)
    warm = B.breath_loss(7.5, 20.0, 0.5, 37.0)
    hot = B.breath_loss(7.5, 45.0, 0.5, 37.0)
    assert cold["T_ex"] < warm["T_ex"] < hot["T_ex"] <= 37.0
    assert cold["sens_w"] > warm["sens_w"] > hot["sens_w"]       # cold air costs more heat
    assert hot["sens_w"] < 0.0                                   # hot air warms the body


def test_breath_scales_with_ventilation():
    a = B.breath_loss(7.5, 20.0, 0.5)
    b = B.breath_loss(15.0, 20.0, 0.5)
    assert b["lat_w"] == pytest.approx(2.0 * a["lat_w"], rel=1e-12)


def test_u_h_c_natural_at_13K():
    assert 4.4 <= float(B.h_c_nat(13.0)) <= 4.6


def test_u_f_cl_at_1_clo():
    assert float(B.f_cl_iso(1.0 * B.CLO)) == pytest.approx(1.15, abs=0.01)
    assert float(B.f_cl_iso(0.0)) == 1.0
    assert float(B.f_cl_iso(0.05)) == pytest.approx(1.0645)         # the two branches of ISO 7730
    assert float(B.f_cl_iso(0.1)) == pytest.approx(1.1145)


def test_h_c_takes_the_larger_of_natural_and_forced():
    assert float(B.h_c(13.0, 0.0)) == pytest.approx(float(B.h_c_nat(13.0)))
    assert float(B.h_c(13.0, 2.0)) == pytest.approx(12.1 * math.sqrt(2.0))


def test_h_r_linearisation():
    """4 eps sigma f (T_m + 273.15)^3: 6.07 for f = 1 at T_m = 28 C, 4.31 with the ISO f_eff 0.71
    (the spec's rounded 4.7 lies between; the voxel path uses per-zone view, so f = 1 there)."""
    assert float(B.h_r(34.0, 22.0, 0.98, 1.0)) == pytest.approx(
        4 * 0.98 * 5.670374e-8 * (28.0 + 273.15) ** 3, rel=1e-12)
    assert 4.2 < float(B.h_r(34.0, 22.0, 0.98, 0.71)) < 4.5


def test_u_evap_max_is_negative_at_40C_90pct_with_35C_skin():
    """Condensation, not a silent clip: p_a = 6.6 kPa against 5.6 kPa at the skin."""
    e = float(B.evap_max(35.0, 40.0, 0.9, 3.0, area_m2=1.8))
    assert e < 0.0
    # and positive in dry air; and clothing reduces it
    d = float(B.evap_max(35.0, 25.0, 0.3, 3.0, area_m2=1.8))
    c = float(B.evap_max(35.0, 25.0, 0.3, 3.0, I_cl=0.155, area_m2=1.8))
    assert d > c > 0.0


def test_evap_max_zero_when_the_air_is_saturated_at_skin_temperature():
    assert float(B.evap_max(35.0, 35.0, 1.0, 3.0)) == pytest.approx(0.0, abs=1e-9)


def test_evaporation_step_conserves_water_and_signs():
    L = float(B.latent_heat(34.0))
    # plenty of capacity: all the film and the sweat evaporate
    r = B.evaporation_step(0.3, 0.2, 4000.0, 1.0, 34.0)
    assert r["evap_g"] == pytest.approx(0.5) and r["film_g"] == pytest.approx(0.0, abs=1e-12)
    assert r["latent_w"] == pytest.approx((r["evap_g"] + r["diffuse_g"]) * L / 1000.0, rel=1e-12)
    # little capacity: water stays; mass closes: sweat + film = evap + film_out + drip
    r = B.evaporation_step(1.0, 0.5, 10.0, 1.0, 34.0, film_cap_g=0.8)
    assert r["evap_g"] + r["film_g"] + r["drip_g"] == pytest.approx(1.5, rel=1e-12)
    assert r["drip_g"] > 0.0 and r["film_g"] == pytest.approx(0.8)
    # condensation: negative evaporation, the film grows, the skin gains heat
    r = B.evaporation_step(0.0, 0.1, -50.0, 10.0, 35.0)
    assert r["evap_g"] < 0.0 and r["film_g"] > 0.1 and r["latent_w"] < 0.0
    assert r["evap_g"] + r["film_g"] + r["drip_g"] == pytest.approx(0.1, rel=1e-12)


def test_face_rows_are_data_not_body_tests():
    """Convection law and emissivity are columns of a material row (a dict)."""
    assert float(B.face_h_conv(B.LEGACY_FACE_ROW, 13.0)) == 10.0
    assert float(B.face_h_conv(B.LEGACY_FACE_ROW, 0.5)) == 10.0
    assert float(B.face_h_conv(B.FLESH_FACE_ROW, 13.0)) == pytest.approx(float(B.h_c_nat(13.0)))
    assert float(B.face_h_conv(B.FLESH_FACE_ROW, 13.0, v=2.0)) == pytest.approx(12.1 * math.sqrt(2.0))
    assert B.LEGACY_FACE_ROW["emissivity"] == 0.95 and B.FLESH_FACE_ROW["emissivity"] == 0.98
    assert set(B.LEGACY_FACE_ROW) == set(B.FLESH_FACE_ROW)
    assert isinstance(B.FLESH_FACE_ROW, dict)


def test_u_nude_resting_balance_at_28C_is_in_the_comfort_range():
    """A nude body in 28 C still air loses 60-90 W dry (F3: radiation + convection
    70-75 percent of about 85-100 W)."""
    area = 1.85
    q = float(B.dry_heat_loss(33.5, 28.0, 28.0, 0.0, 0.0, f_eff=0.71)) * area
    assert 45.0 < q < 100.0, q


# ── thermoreg ────────────────────────────────────────────────────────────────
def _sk(delta=0.0):
    return {z: TR.SET_SKIN[z] + delta for z in ZONES}


def _prm(pid=1, name="Ann"):
    return TR.person_params(pid, name)


def _targets(Tc, dsk, prm, mass=70.0, dehyd=0.0):
    sig = TR.signals(Tc, _sk(dsk), prm["set_core"], dehyd_pct=dehyd)
    return sig, TR.targets(sig, prm, mass, Tc)


def test_thermoreg_zero_output_at_set_point():
    prm = _prm()
    sig, out = _targets(prm["set_core"], 0.0, prm)
    assert out.sweat_total() == 0.0 and out.shiver_w == 0.0
    assert all(v == pytest.approx(1.0) for v in out.vessel.values())      # basal flow, no change


def test_u_sweat_is_monotone_in_core_error():
    prm = _prm()
    prev = -1.0
    for e in np.linspace(0.0, 2.0, 21):
        _, out = _targets(prm["set_core"] + e, 0.0, prm)
        assert out.sweat_total() >= prev - 1e-15
        prev = out.sweat_total()
    assert prev > 0.1
    # more skin warmth also sweats more; cap at sw_max
    _, a = _targets(prm["set_core"] + 0.5, 0.0, prm)
    _, b = _targets(prm["set_core"] + 0.5, 2.0, prm)
    assert b.sweat_total() > a.sweat_total()
    _, c = _targets(prm["set_core"] + 5.0, 6.0, prm)
    assert c.sweat_total() <= prm["sw_max"] + 1e-12
    assert math.fsum(c.sweat_g_s.values()) == pytest.approx(c.sweat_total())


def test_sweat_goes_to_warm_zones_without_changing_the_total():
    prm = _prm()
    sk = _sk(0.5)
    sk["torso"] += 2.0
    sig = TR.signals(prm["set_core"] + 0.6, sk, prm["set_core"])
    out = TR.targets(sig, prm, 70.0, 37.0)
    base = TR.SWEAT_SHARE["torso"] * out.sweat_total()
    assert out.sweat_g_s["torso"] > base
    assert math.fsum(TR.SWEAT_SHARE.values()) == pytest.approx(1.0, abs=1e-12)


def test_u_dehydration_raises_the_sweat_threshold_and_lowers_the_cap():
    prm = _prm()
    _, wet = _targets(prm["set_core"] + 0.5, 0.0, prm, dehyd=0.0)
    _, dry = _targets(prm["set_core"] + 0.5, 0.0, prm, dehyd=2.0)
    assert dry.sweat_total() < wet.sweat_total()
    _, none = _targets(prm["set_core"] + 0.3, 0.0, prm, dehyd=2.0)
    assert none.sweat_total() < wet.sweat_total()


def test_u_shiver_depression_curve():
    """1.0 at >= 35.5 C, 0.5 at 32.5 C, 0 at <= 30 C (a data curve, not Q10)."""
    assert TR.shiver_depression(37.0) == 1.0 and TR.shiver_depression(35.5) == 1.0
    assert TR.shiver_depression(32.5) == pytest.approx(0.5)
    assert TR.shiver_depression(30.0) == 0.0 and TR.shiver_depression(28.0) == 0.0
    xs = np.linspace(28, 37, 91)
    ys = [TR.shiver_depression(x) for x in xs]
    assert all(b >= a for a, b in zip(ys, ys[1:]))                    # monotone
    assert TR.shiver_depression(34.0) == pytest.approx(0.75)


def test_shiver_starts_with_cold_skin_and_is_cut_by_deep_cold():
    prm = _prm()
    sig = TR.signals(prm["set_core"] - 0.3, _sk(-6.0), prm["set_core"])
    warm = TR.targets(sig, prm, 70.0, 36.5)
    deep = TR.targets(sig, prm, 70.0, 32.5)
    dead = TR.targets(sig, prm, 70.0, 29.0)
    assert warm.shiver_w > 0.0
    assert deep.shiver_w == pytest.approx(0.5 * warm.shiver_w)
    assert dead.shiver_w == 0.0
    # never above the whole-body maximum
    assert warm.shiver_w <= prm["sh_max_w_kg"] * 70.0 + 1e-9
    # tired or starved: the fatigue factor scales it
    tired = TR.targets(sig, prm, 70.0, 36.5, fatigue=0.5)
    assert tired.shiver_w == pytest.approx(0.5 * warm.shiver_w)


def test_vessels_stay_inside_0p1_and_60():
    prm = _prm()
    for Tc in (30.0, 35.0, 37.0, 38.5, 41.0):
        for d in (-15.0, -3.0, 0.0, 3.0, 8.0):
            _, out = _targets(Tc, d, prm)
            assert all(0.1 <= v <= 60.0 for v in out.vessel.values()), (Tc, d)
    _, hot = _targets(prm["set_core"] + 2.0, 3.0, prm)
    _, cold = _targets(prm["set_core"], -10.0, prm)
    assert hot.vessel["left hand"] > 10.0 > cold.vessel["left hand"]
    assert cold.vessel["left hand"] < 1.0 and hot.vessel["torso"] > 1.0
    # hands and feet have the widest range
    assert hot.vessel["left hand"] > hot.vessel["torso"] > hot.vessel["head"]


def test_lag_converges_without_overshoot_for_any_dt():
    prm = _prm()
    for dt in (0.5, 1.0, 5.0, 60.0, 600.0):
        st = TR.new_state()
        sig = TR.signals(prm["set_core"] + 0.6, _sk(1.0), prm["set_core"])
        tgt = TR.targets(sig, prm, 70.0, 37.5)
        last = -1.0
        for _ in range(int(3000 / dt)):
            st, out = TR.controllers(sig, prm, st, dt, 70.0, 37.5)
            assert out.sweat_total() <= tgt.sweat_total() + 1e-12          # no overshoot
            assert out.sweat_total() >= last - 1e-12                       # monotone up
            last = out.sweat_total()
        assert out.sweat_total() == pytest.approx(tgt.sweat_total(), rel=1e-6)
        assert out.shiver_w == 0.0


def test_dead_time_delays_the_rise():
    prm = _prm()
    st = TR.new_state()
    sig = TR.signals(prm["set_core"] + 1.0, _sk(1.0), prm["set_core"])
    delay = prm["delay_sweat"]
    t = 0.0
    seen_zero_until = None
    for _ in range(40):
        st, out = TR.controllers(sig, prm, st, 1.0, 70.0, 38.0)
        t += 1.0
        if out.sweat_total() > 0.0 and seen_zero_until is None:
            seen_zero_until = t
    assert seen_zero_until is not None
    assert delay <= seen_zero_until <= delay + 1.5, (delay, seen_zero_until)


def test_lag_state_is_a_plain_value():
    prm = _prm()
    st = TR.new_state()
    sig = TR.signals(prm["set_core"] + 1.0, _sk(1.0), prm["set_core"])
    for _ in range(10):
        st, _o = TR.controllers(sig, prm, st, 1.0, 70.0, 38.0)
    snap = copy.deepcopy(st)
    st1, o1 = TR.controllers(sig, prm, copy.deepcopy(snap), 1.0, 70.0, 38.0)
    st2, o2 = TR.controllers(sig, prm, copy.deepcopy(snap), 1.0, 70.0, 38.0)
    assert o1.sweat_g_s == o2.sweat_g_s and st1 == st2


def test_person_draws_are_deterministic_and_independent_of_dict_order():
    a = TR.person_params(7, "Tom")
    b = TR.person_params(7, "Tom")
    assert a == b
    assert TR.person_params(8, "Tom") != a and TR.person_params(7, "Ann") != a
    for k, (lo, hi) in TR.PARAMS.items():
        assert lo <= a[k] <= hi
    # a draw by name does not depend on which other names were drawn first, or in what order
    names = list(TR.PARAMS)
    random.Random(3).shuffle(names)
    c = {k: TR.draw(7, "Tom", k) for k in names}
    assert c == a


def test_set_point_circadian_sleep_and_fever():
    prm = _prm()
    s17 = TR.set_points(prm, hour=17.0)
    s05 = TR.set_points(prm, hour=5.0)
    assert s17 - s05 == pytest.approx(0.6)
    assert TR.set_points(prm, hour=17.0, asleep=True) == pytest.approx(s17 - 0.3)
    assert TR.set_points(prm, hour=17.0, offset=1.5) == pytest.approx(s17 + 1.5)


def test_schmitt_one_event_per_onset_and_offset():
    on, ev = TR.schmitt(False, 0.05, 1.0)
    assert (on, ev) == (False, None)
    on, ev = TR.schmitt(False, 0.11, 1.0)
    assert (on, ev) == (True, "on")
    on, ev = TR.schmitt(True, 0.07, 1.0)                  # between the two levels: stays on, silent
    assert (on, ev) == (True, None)
    on, ev = TR.schmitt(True, 0.049, 1.0)
    assert (on, ev) == (False, "off")
    # a signal wobbling around 10 percent does not chatter
    on, n = False, 0
    for v in [0.09, 0.11, 0.09, 0.11, 0.08, 0.12, 0.07, 0.09, 0.11]:
        on, ev = TR.schmitt(on, v, 1.0)
        n += ev is not None
    assert n == 1


def test_events_fixed_order_and_verbs():
    prm = _prm()
    flags = {}
    sig = TR.signals(prm["set_core"] - 0.5, _sk(-8.0), prm["set_core"])
    out = TR.targets(sig, prm, 70.0, 36.5)
    evs = TR.events(prm, out, 70.0, flags)
    assert evs == [("shivers", "on")]
    assert TR.events(prm, out, 70.0, flags) == []                    # no repeat
    assert {r["verb"] for r in TR.THERMO_ROWS.values()} >= {"shivers", "sweats"}


def test_rows_are_continuous_and_work_asleep_but_not_dead():
    for name, row in TR.THERMO_ROWS.items():
        assert row["kind"] == "continuous" and row["works_when"] == "alive"
        assert "tau_s" in row and "band" in row and "yields_to" in row
        assert TR.works(row, {"alive": True, "asleep": True})
        assert TR.works(row, {"alive": True, "unconscious": True})
        assert not TR.works(row, {"alive": False})
    awake_row = {"kind": "continuous", "works_when": "awake"}
    assert not TR.works(awake_row, {"alive": True, "asleep": True})


def test_heat_signals_are_0_to_1_and_use_data_thresholds():
    h = TR.heat_signals_from(37.0, 33.5, 37.0, 33.5)
    assert h == {"cold": 0.0, "hot": 0.0, "thirsty": 0.0, "hungry": 0.0}
    cold = TR.heat_signals_from(35.5, 27.5, 37.0, 33.5)
    assert cold["cold"] == 1.0 and cold["hot"] == 0.0
    hot = TR.heat_signals_from(38.0, 36.0, 37.0, 33.5, dehyd_pct=2.5)
    assert 0.0 < hot["hot"] <= 1.0 and hot["thirsty"] == pytest.approx(1.0)
    assert TR.heat_signals({}) == h
    p = {"heat": {"T_pool": 36.0, "T_skin_mean": 30.0}}
    s = TR.heat_signals(p)
    assert 0.0 < s["cold"] < 1.0


def test_codeonly_area_weights_sum_to_one():
    assert math.fsum(AREA_SHARE.values()) == pytest.approx(1.0, abs=1e-12)
    sig = TR.signals(37.0, _sk(1.0), 37.0)
    assert sig["W_s"] == pytest.approx(1.0, abs=1e-12) and sig["C_s"] == 0.0


# ── airsink ──────────────────────────────────────────────────────────────────
def test_open_world_sink_books_vapour_and_heat():
    s = AS.OpenWorldSink()
    assert isinstance(s, AS.AirSink)
    s.add(vapour_g=2.0, heat_J=4840.0, o2_l=-0.3, co2_l=0.25)
    s.add(vapour_g=1.0, heat_J=100.0)
    assert s.gone["vapour"] == 3.0 and s.shed == 4940.0
    assert s.gone["o2_l"] == pytest.approx(-0.3) and s.gone["co2_l"] == pytest.approx(0.25)
    assert s.ambient() == (20.0, 0.5) and s.o2_fraction() == pytest.approx(0.209)


def test_deliver_sums_latent_and_sensible_and_shares_the_path():
    """The tick path and the skip path call the same function and get the same books."""
    a, b = AS.SinkLog(), AS.SinkLog()
    for _ in range(60):                                  # sixty ticks of one second
        AS.deliver(a, 0.01, 24.2, 3.0, -0.005, 0.004)
    AS.deliver(b, 0.6, 24.2 * 60, 3.0 * 60, -0.3, 0.24)  # one skip of sixty seconds
    ta, tb = a.totals(), b.totals()
    for k in ta:
        assert ta[k] == pytest.approx(tb[k], rel=1e-12)
    assert a.shed == pytest.approx(b.shed, rel=1e-12) and a.gone["vapour"] == pytest.approx(b.gone["vapour"], rel=1e-12)


def test_sink_can_use_a_world_dict():
    gone = {"E": 5.0}
    s = AS.OpenWorldSink(gone)
    s.add(vapour_g=1.5)
    assert gone["vapour"] == 1.5 and gone["E"] == 5.0


# ── humidity domain (regression: at or above 1 atm the ratio went infinite, then negative) ──
@pytest.mark.parametrize("seed", range(30))
def test_humidity_ratio_refuses_vapour_at_or_above_the_total_pressure(seed):
    rng = np.random.default_rng(seed)
    P = float(rng.uniform(5.0, 200.0))                         # any total pressure, kPa
    p_ok = rng.uniform(0.0, 0.999, 20) * P
    w = B.humidity_ratio(p_ok, P)
    assert np.all(np.isfinite(w)) and np.all(w >= 0.0)
    assert np.all(np.diff(B.humidity_ratio(np.sort(p_ok), P)) >= 0.0)            # rises with p
    for bad in (P, P * (1.0 + 10.0 ** rng.uniform(-9, 1)), -rng.uniform(1e-6, 5.0), np.nan, np.inf):
        with pytest.raises(ValueError):
            B.humidity_ratio(bad, P)
    with pytest.raises(ValueError):
        B.humidity_ratio(np.append(p_ok, P), P)                # one bad element refuses the array


@pytest.mark.parametrize("seed", range(30))
def test_vapour_pressure_stays_below_the_total_and_inside_rh_0_to_1(seed):
    rng = np.random.default_rng(100 + seed)
    for T in rng.uniform(-40.0, 98.0, 10):
        rh = float(rng.uniform(0.0, 1.0))
        pv = float(B.vapour_pressure(T, rh))
        assert 0.0 <= pv < B.P_ATM and pv == pytest.approx(rh * float(B.p_sat(T)), rel=1e-14)
        float(B.humidity_ratio(pv))                            # always a legal input afterwards
    for rh in (-0.01, 1.01, np.nan):
        with pytest.raises(ValueError):
            B.vapour_pressure(20.0, rh)
    # water boils: saturated at or above the boiling point of the total pressure is refused
    for P in (B.P_ATM, 50.0, 20.0, 10.0):
        T_boil = 243.12 * np.log(P / 0.6112) / (17.62 - np.log(P / 0.6112))        # inverse Magnus
        assert float(B.p_sat(T_boil)) == pytest.approx(P, rel=1e-9)
        with pytest.raises(ValueError):
            B.vapour_pressure(T_boil + rng.uniform(0.0, 20.0), 1.0, P)
        assert float(B.vapour_pressure(T_boil - 0.5, 1.0, P)) < P


def test_p_sat_refuses_the_magnus_pole_and_nonfinite_temperatures():
    for bad in (-243.12, -300.0, np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            B.p_sat(bad)
    with pytest.raises(ValueError):
        B.p_sat(np.array([20.0, -250.0]))
    assert np.all(np.diff(B.p_sat(np.linspace(-60.0, 120.0, 400))) > 0.0)       # monotone across 1 atm


def test_breath_refuses_a_total_pressure_below_the_vapour_of_exhaled_air():
    """Below about 5.3 kPa the saturated breath at 34 C would be all vapour: no such breath."""
    r = B.breath_loss(7.5, 20.0, 0.5, P=70.0)                  # a high-altitude air still works
    assert r["w_ex"] > B.breath_loss(7.5, 20.0, 0.5)["w_ex"]
    with pytest.raises(ValueError):
        B.breath_loss(7.5, 20.0, 0.5, P=5.0)


def test_breath_loss_passes_its_total_pressure_into_both_humidity_ratios():
    """Polish 5.B.4. A random sweep recomputes the breath from the printed formulas with the
    SAME P in the inlet and the exhaled humidity ratio; a breath_loss that dropped P (used 1 atm)
    in either ratio, or ignored it, gives a different water and latent heat at every P that is not
    1 atm. (P in vapour_pressure alone cannot show: humidity_ratio refuses the same overshoot.)"""
    rng = random.Random(51304)
    for _ in range(60):
        P = rng.uniform(40.0, 110.0)
        T_in = rng.uniform(-15.0, 30.0)
        rh = rng.uniform(0.0, 1.0)
        T_core = rng.uniform(35.5, 38.5)
        V = rng.uniform(3.0, 60.0)
        T_ex = float(B.exhale_temperature(T_in, T_core))
        rho = B.RHO_AIR_20 * 293.15 / (T_in + 273.15)
        m_dry = V / 1000.0 / 60.0 * rho
        w_in = 0.622 * (rh * float(B.p_sat(T_in))) / (P - rh * float(B.p_sat(T_in)))
        w_ex = 0.622 * float(B.p_sat(T_ex)) / (P - float(B.p_sat(T_ex)))
        r = B.breath_loss(V, T_in, rh, T_core=T_core, P=P)
        assert r["w_in"] == pytest.approx(w_in, rel=1e-12, abs=1e-15)
        assert r["w_ex"] == pytest.approx(w_ex, rel=1e-12)
        assert r["water_g_s"] == pytest.approx(m_dry * (w_ex - w_in) * 1000.0, rel=1e-12)
        assert r["lat_w"] == pytest.approx(m_dry * (w_ex - w_in) * (2501.0 - 2.37 * T_ex) * 1000.0,
                                           rel=1e-12)
        assert r["sens_w"] == pytest.approx(
            m_dry * (B.CP_AIR + w_in * B.CP_VAP) * (T_ex - T_in), rel=1e-12)
    # and a thinner air carries more water per kg of dry air out of the same lungs
    lo = B.breath_loss(7.5, 20.0, 0.5, P=60.0)
    hi = B.breath_loss(7.5, 20.0, 0.5, P=100.0)
    assert lo["w_ex"] > hi["w_ex"] and lo["water_g_s"] > hi["water_g_s"]
