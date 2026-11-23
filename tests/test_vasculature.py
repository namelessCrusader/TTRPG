"""Vasculature: pressure, cut vessels, clot, press/bind/tourniquet, pooling, ischaemia (src/voxel/vasculature.py, wave 13 T10).

(The file is test_vasculature because tests/test_vessels.py is about containers.)

Tags: test_echo_* checks a number the constants were FITTED to; test_codeonly_*
checks the code against its own table; test_u_* leans on an unverified number;
untagged tests are exact (identities, closed forms, conservation).
"""
import math

import pytest

from src.voxel import anatomy as A
from src.voxel import vasculature as V
from src.voxel.materials import BLEED

V0 = 4900.0                       # 70 kg x 70 ml/kg


def adult():
    return A.make_sheet(mass_kg=70.0, height_m=1.75, draws=False)


def cut_of(vessel, f_open=1.0, size=1.0, along=0.5):
    """A cut of a named vessel at its adult size (the unit tests run at 70 kg)."""
    v = A.VESSEL_OF[vessel]
    return dict(vessel=vessel, kind=v["kind"], region=v["region"], f_open=f_open,
                d_m=v["d_mm"] * 1e-3, Q_full=v["Q_ml_s"], along=along,
                seal_s=0.0, sealed=False, t=0.0)


def bleed(cut, press=0.0, z=0.0, tmax=900.0, dt=0.5, met=1.0, T=37.0, stop_when_sealed=True):
    """Run one cut on a 4.9 L man; returns a summary and the ledger check."""
    V_ml, lost, t = V0, 0.0, 0.0
    faint = death = None
    worst = 0.0
    while t < tmax:
        ml = V.step_wound([cut], V_ml, V0, [press], met, dt, T, z)
        V_ml -= ml
        lost += ml
        t += dt
        worst = max(worst, abs(V_ml + lost - V0))
        p = V.perfusion(V_ml / V0)
        if faint is None and p < 0.55:
            faint = t
        if death is None and p < 0.25:
            death = t
        if stop_when_sealed and cut["sealed"]:
            break
    return dict(t=t, lost=lost, V=V_ml, faint=faint, death=death, sealed=cut["sealed"],
                ledger_err=worst)


# ── pressure and perfusion ──────────────────────────────────────────────────

def test_map_rel_endpoints_and_dead():
    assert V.map_rel(1.0) == 1.0 and V.map_rel(0.8) == 1.0
    assert V.map_rel(0.25) == 0.0 and V.map_rel(0.1) == 0.0
    assert V.map_rel(0.525) == pytest.approx(0.5)
    assert V.map_rel(1.0, alive=False) == 0.0 and V.map_rel(0.6, alive=False) == 0.0


def test_perfusion_equals_the_old_BLEED_numbers_at_20_values():
    for i in range(20):
        f = i / 19.0 * 1.2                                      # 0 .. 1.2, past both ends
        x = (f - BLEED["f0"]) / (1.0 - BLEED["f0"])
        old = float(min(max(x, 0.0), 1.0) ** BLEED["k"])
        assert V.perfusion(f) == pytest.approx(old, rel=1e-12, abs=1e-15)
    assert V.PERF_F0 == BLEED["f0"] and V.PERF_K == BLEED["k"]
    assert V.ML_PER_KG == BLEED["ml_per_kg"]


def test_the_faint_and_death_lines_are_unchanged():
    assert V.faint_loss_frac() == pytest.approx(0.361, abs=0.002)
    assert V.death_loss_frac() == pytest.approx(0.504, abs=0.002)
    assert V.perfusion(1.0 - V.faint_loss_frac()) == pytest.approx(0.55, abs=1e-9)
    assert V.perfusion(1.0 - V.death_loss_frac()) == pytest.approx(0.25, abs=1e-9)


# ── a cut vessel: the worked bleeds ─────────────────────────────────────────

def test_u_full_femoral_artery_cut_faints_in_90_to_180_s_and_kills_in_under_5_min():
    r = bleed(cut_of("femoral artery"))
    assert 90.0 <= r["faint"] <= 180.0, r
    assert r["death"] < 300.0, r
    assert r["lost"] / V0 > 0.5                                  # it does not stop before it kills
    assert r["faint"] == pytest.approx(128.0, abs=8.0)           # the prototype: 128 s and 210 s
    assert r["death"] == pytest.approx(210.0, abs=10.0)


def test_u_a_30_percent_cut_of_the_femoral_artery_faints_later():
    r = bleed(cut_of("femoral artery", 0.3), tmax=900.0)
    full = bleed(cut_of("femoral artery"))
    assert r["faint"] > full["faint"] and r["faint"] == pytest.approx(277.0, abs=15.0)


def test_u_a_press_of_15_kPa_or_more_stops_the_flow():
    for p in (15e3, 20e3, 40e3 * 0.37):
        c = cut_of("femoral artery")
        r = bleed(c, press=p, tmax=60.0, stop_when_sealed=False)
        assert r["lost"] == 0.0 and r["V"] == V0, p
    c = cut_of("femoral artery")
    r = bleed(c, press=8e3, tmax=60.0, stop_when_sealed=False)
    assert 0.0 < r["lost"]                                       # a light press only slows it
    c = cut_of("femoral artery")
    slow = bleed(c, press=8e3, tmax=60.0, stop_when_sealed=False)["lost"]
    fast = bleed(cut_of("femoral artery"), tmax=60.0, stop_when_sealed=False)["lost"]
    assert slow < fast


def test_u_a_press_does_not_leave_the_cut_sealed_for_ever():
    c = cut_of("femoral artery")
    bleed(c, press=20e3, tmax=300.0, stop_when_sealed=False)
    assert c["sealed"] is True
    r = bleed(c, press=0.0, tmax=30.0, stop_when_sealed=False)   # the hand lifts: 12 kPa > the hold
    assert r["lost"] > 0 and c["sealed"] is False                # it REOPENS, and the clot restarts
    assert c["seal_s"] <= 30.0 + 1e-9


def test_u_scalp_artery_and_scratches_self_seal_in_under_6_minutes():
    c = cut_of("scalp arteries")
    r = bleed(c, tmax=600.0)
    assert r["sealed"] and r["t"] < 360.0, r
    assert r["lost"] / V0 < 0.1
    for name in ("radial artery", "ulnar artery"):
        small = cut_of(name, f_open=0.1)                         # a scratch across the artery
        rr = bleed(small, tmax=600.0)
        assert rr["sealed"] and rr["t"] < 360.0, (name, rr)


def test_u_a_vein_of_3_mm_or_less_seals_at_4_kPa_and_alone_in_a_minute_or_two():
    for name in ("cephalic vein", "saphenous vein", "great saphenous"):
        c = cut_of(name)
        r = bleed(c, press=4e3, tmax=20.0, stop_when_sealed=False)
        assert r["lost"] == 0.0 and c["sealed"], name
    c = cut_of("cephalic vein")
    r = bleed(c, tmax=600.0)
    assert r["sealed"] and r["t"] < 120.0 and r["lost"] < 100.0


def test_u_the_femoral_vein_seals_in_under_10_min_and_the_hold_is_1_4_kPa():
    c = cut_of("femoral vein")
    assert V.hold_pa(dict(c, t=1e6), 1e6) == pytest.approx(1.4e3, rel=1e-3)
    r = bleed(c, tmax=900.0)
    assert r["sealed"] and r["t"] < 600.0, r


def test_the_clot_hold_is_14_kPa_per_mm_grows_with_time_and_falls_with_width():
    c = cut_of("scalp arteries")
    c["d_m"] = 1e-3                                              # a 1 mm vessel, open right through
    c["f_open"] = 1.0
    late = dict(c, t=1e6)                                        # past the spasm: use the cut's own d for the closed form
    big = dict(c, kind="v")                                      # a vein does not narrow
    assert V.hold_pa(big, 1e6) == pytest.approx(14e3, rel=1e-6)
    assert V.hold_pa(big, 0.0) == 0.0
    assert V.hold_pa(big, 180.0) == pytest.approx(14e3 * (1 - math.exp(-1.0)), rel=1e-9)
    wide = dict(big, f_open=0.5)
    assert V.hold_pa(wide, 180.0) == pytest.approx(2 * V.hold_pa(big, 180.0), rel=1e-9)
    assert V.hold_pa(big, 1e6, T_body=27.0) == pytest.approx(7e3, rel=1e-6)          # cold: half
    assert V.hold_pa(big, 1e6, T_body=0.0) == pytest.approx(1.4e3, rel=1e-6)         # floor 0.1
    assert late["t"] == 1e6


def test_spasm_shrinks_d_eff_and_feeds_the_hold_for_small_arteries_only():
    radial = cut_of("radial artery")
    assert V.d_eff_ratio(radial, 0.0) == 1.0
    assert V.d_eff_ratio(radial, 30.0) == pytest.approx(math.sqrt(1 - 0.8 * (1 - math.exp(-1))))
    assert V.d_eff_ratio(radial, 1e6) == pytest.approx(math.sqrt(0.2))
    assert V.d_eff_ratio(cut_of("femoral artery"), 1e6) == 1.0       # a big artery does not narrow
    assert V.d_eff_ratio(cut_of("cephalic vein"), 1e6) == 1.0        # nor does a vein
    c0, c1 = dict(radial, t=0.0), dict(radial, t=120.0)
    assert V.hold_pa(c1, 120.0) > V.hold_pa(c0, 120.0)               # a narrower width holds better
    # and the flow falls by the same factor squared
    f0 = V.flow_ml_s(c0, 1.0)
    f1 = V.flow_ml_s(c1, 1.0)
    assert f1 / f0 == pytest.approx(V.d_eff_ratio(c1, 120.0) ** 2)


def test_u_a_radial_artery_seals_as_pressure_falls():
    c = cut_of("radial artery")
    r = bleed(c, tmax=900.0)
    assert r["sealed"] and r["lost"] / V0 < 0.15
    # with more of the vessel open the same artery bleeds longer
    c2 = cut_of("radial artery", f_open=0.5)
    r2 = bleed(c2, tmax=900.0)
    assert r2["lost"] < r["lost"] or r2["t"] < r["t"]


def test_the_flow_formula_is_the_spec_one():
    c = cut_of("femoral artery", f_open=0.5)
    g = 0.5 / (0.5 + 0.5 * 0.5)
    assert V.g_open(0.5) == pytest.approx(g) and V.g_open(1.0) == 1.0 and V.g_open(0.0) == 0.0
    assert V.flow_ml_s(c, 1.0) == pytest.approx(15.0 * g)
    assert V.flow_ml_s(c, 1.0, p_ext=6e3) == pytest.approx(15.0 * g * 0.5)
    assert V.flow_ml_s(c, 1.0, p_ext=12e3) == 0.0
    assert V.flow_ml_s(c, 1.0, alive=False) == 0.0
    # exercise raises the arterial pressure 8 % a MET
    assert V.flow_ml_s(c, 1.0, met=4.0) == pytest.approx(15.0 * g * 1.24)
    vein = cut_of("femoral vein")
    assert V.flow_ml_s(vein, 1.0) == pytest.approx(6.0)


def test_flow_falls_with_blood_loss_as_MAP_falls():
    c = cut_of("femoral artery")
    q = [V.flow_ml_s(c, f) for f in (1.0, 0.8, 0.7, 0.5, 0.3, 0.25)]
    assert q == sorted(q, reverse=True) and q[0] == 15.0 and q[-1] == 0.0


# ── height above the heart ──────────────────────────────────────────────────

def test_hydrostatic_term_is_10_4_kPa_a_metre_and_vanishes_with_the_circulation():
    assert V.hydro_pa(1.0, 1.0) == pytest.approx(10.4e3)
    assert V.hydro_pa(-0.5, 1.0) == pytest.approx(-5.2e3)
    assert V.hydro_pa(1.0, 0.15) == pytest.approx(5.2e3)           # min(1, MAP_rel / 0.3)
    assert V.hydro_pa(1.0, 0.0) == 0.0


def test_u_a_limb_raised_half_a_metre_bleeds_less():
    q_lie = V.flow_ml_s(cut_of("brachial artery"), 1.0, z_rel_m=0.0)
    q_up = V.flow_ml_s(cut_of("brachial artery"), 1.0, z_rel_m=-0.5)
    assert q_up < 0.7 * q_lie
    a = bleed(cut_of("brachial artery"), z=0.0, tmax=60.0, stop_when_sealed=False)["lost"]
    b = bleed(cut_of("brachial artery"), z=-0.5, tmax=60.0, stop_when_sealed=False)["lost"]
    assert b < a


def test_u_a_standing_ankle_vein_bleeds_more_than_a_lying_one():
    lie = V.flow_ml_s(cut_of("saphenous vein"), 1.0, z_rel_m=0.0)
    stand = V.flow_ml_s(cut_of("saphenous vein"), 1.0, z_rel_m=0.9)
    assert stand > 5 * lie
    a = bleed(cut_of("saphenous vein"), z=0.0, tmax=60.0)
    b = bleed(cut_of("saphenous vein"), z=0.9, tmax=60.0)
    assert b["lost"] > a["lost"]
    # lying down stops a standing leg wound sooner: the same cut, with the column removed
    c = cut_of("saphenous vein")
    bleed(c, z=0.9, tmax=30.0, stop_when_sealed=False)
    assert not c["sealed"]
    bleed(c, z=0.0, tmax=30.0, stop_when_sealed=False)
    assert c["sealed"]


# ── what a wound crosses ────────────────────────────────────────────────────

def test_a_wound_that_reaches_the_vessels_depth_across_its_azimuth_cuts_it():
    s = adult()
    w = dict(depth_m=0.045, length_m=0.10, az=60.0, along=0.5)
    cuts = V.crosses(w, "left thigh", s)
    names = {c["vessel"] for c in cuts}
    assert "femoral artery" in names and "femoral vein" in names
    for c in cuts:
        assert 0.0 < c["f_open"] <= 1.0 and c["sealed"] is False and c["seal_s"] == 0.0
    art = [c for c in cuts if c["vessel"] == "femoral artery"][0]
    assert art["Q_full"] == pytest.approx(15.0)
    assert art["d_m"] == pytest.approx(8e-3)


def test_a_shallow_or_blunt_wound_or_one_off_the_azimuth_cuts_nothing():
    s = adult()
    assert V.crosses(dict(depth_m=0.002, length_m=0.10, az=60.0), "left thigh", s) == []
    assert V.crosses(dict(depth_m=0.0, length_m=0.10, az=60.0), "left thigh", s) == []
    # the test_wounds forearm blow lands at az 270, off the radial and ulnar
    assert V.crosses(dict(depth_m=0.03, length_m=0.05, az=270.0), "left forearm", s) == []
    deep = V.crosses(dict(depth_m=0.03, length_m=0.05, az=40.0), "left forearm", s)
    assert {c["vessel"] for c in deep} >= {"radial artery", "ulnar artery"}


def test_f_open_is_depth_fraction_times_arc_fraction():
    s = adult()
    D = V.limb_thickness_m("left thigh", s)
    d = 8e-3
    v_depth = 30.0 / 150.0 * D
    w = dict(depth_m=v_depth, length_m=math.pi * D, az=60.0)       # 180 degrees: covers the vessel arc
    art = [c for c in V.crosses(w, "left thigh", s) if c["vessel"] == "femoral artery"][0]
    assert art["f_open"] == pytest.approx(0.5, abs=1e-9)           # half the vessel's depth, all its arc
    w2 = dict(depth_m=v_depth + d, length_m=math.pi * D, az=60.0)
    assert [c for c in V.crosses(w2, "left thigh", s) if c["vessel"] == "femoral artery"][0]["f_open"] == 1.0
    # an arc that overlaps half the vessel's width
    thin = dict(depth_m=v_depth + d, length_m=0.001, az=60.0 + 20.0 + 0.0)
    cut = [c for c in V.crosses(thin, "left thigh", s) if c["vessel"] == "femoral artery"]
    assert cut and 0.0 < cut[0]["f_open"] < 1.0


def test_the_stock_figure_limb_is_5_cm_thick_when_the_wound_says_so():
    s = A.make_sheet(draws=False)
    w = dict(depth_m=0.03, length_m=0.15, az=60.0, d_limb_m=0.05)
    names = {c["vessel"] for c in V.crosses(w, "left thigh", s)}
    assert "femoral artery" in names
    assert V.crosses(dict(w, depth_m=0.0), "left thigh", s) == []


def test_flexor_tendons_are_a_structure_not_a_vessel():
    s = adult()
    D = V.limb_thickness_m("left forearm", s)
    assert V.tendon_cut(dict(depth_m=0.03, length_m=0.2, az=45.0), "left forearm", s) > 0.5
    assert V.tendon_cut(dict(depth_m=0.003, length_m=0.2, az=45.0), "left forearm", s) == 0.0
    assert V.tendon_cut(dict(depth_m=0.03, length_m=0.2, az=45.0), "left thigh", s) == 0.0
    assert V.tendon_cut(dict(depth_m=0.03, length_m=0.02, az=270.0), "left forearm", s) == 0.0
    assert D > 0


def test_cut_q_full_scales_as_mass_to_the_two_thirds():
    big = A.make_sheet(mass_kg=140.0, height_m=2.0, draws=False)
    w = dict(depth_m=0.07, length_m=0.2, az=60.0, d_limb_m=0.2)
    art = [c for c in V.crosses(w, "left thigh", big) if c["vessel"] == "femoral artery"][0]
    assert art["Q_full"] == pytest.approx(15.0 * 2.0 ** (2.0 / 3.0), rel=1e-12)


# ── press, bind, tourniquet ─────────────────────────────────────────────────

def test_press_hoop_and_aop_formulas():
    assert V.press_pa(100.0, 4e-4, 0.0) == pytest.approx(250e3)          # 100 N on 4 cm2
    assert V.press_pa(100.0, 4e-4, 0.04) == pytest.approx(250e3 * math.exp(-1.0))
    assert V.press_pa(40.0, 1e-3, 0.03) > 15e3                           # a firm hand stops the femoral
    assert V.hoop_pa(30.0, 0.05, 0.04) == pytest.approx(30.0 / (0.05 * 0.04))
    sbp = 16e3
    # WAVE 15: the width term SATURATES (the old fit 1 + 0.1 circ / width kept falling, which
    # the thigh and arm cuff studies do not: little change between 13 and 18 cm). The rows
    # the fit is held to (SBP 120 mmHg): thigh (58 cm) narrow 235-259 mmHg, wide (13.5 cm)
    # 144 mmHg; upper arm (30 cm) 145 mmHg at 5 cm and 120 mmHg at 12 cm.
    mmhg = 133.322
    assert 225 <= V.aop_pa(sbp, 0.58, 0.045) / mmhg <= 275        # narrow strap, thigh
    assert 135 <= V.aop_pa(sbp, 0.58, 0.135) / mmhg <= 155        # wide cuff, thigh
    assert 135 <= V.aop_pa(sbp, 0.30, 0.05) / mmhg <= 155         # 5 cm on the arm
    assert 115 <= V.aop_pa(sbp, 0.30, 0.12) / mmhg <= 130         # 12 cm on the arm
    assert V.aop_pa(sbp, 0.58, 0.13) - V.aop_pa(sbp, 0.58, 0.18) < 0.15 * sbp   # saturates
    assert V.aop_pa(sbp, 0.58, 0.18) > sbp                         # never below the systolic
    assert V.aop_pa(sbp, 0.60, 0.05) > V.aop_pa(sbp, 0.40, 0.05)   # rises with circumference
    assert V.aop_pa(sbp, 0.50, 0.10) < V.aop_pa(sbp, 0.50, 0.04)   # falls with width


def test_a_tourniquet_acts_only_distal_to_the_cuff():
    sbp = 16e3
    tq = dict(pressure_Pa=40e3, width_m=0.05, along=0.3)
    distal = cut_of("femoral artery", along=0.7)
    proximal = cut_of("femoral artery", along=0.1)
    assert V.tourniquet_pext(distal, tq, sbp=sbp, circ_m=0.5) == 40e3
    assert V.tourniquet_pext(proximal, tq, sbp=sbp, circ_m=0.5) == 0.0
    # too loose for an artery (below AOP): no arterial occlusion; a vein is still closed
    loose = dict(tq, pressure_Pa=20e3)
    assert V.tourniquet_pext(distal, loose, sbp=sbp, circ_m=0.5) == 0.0
    vein = cut_of("femoral vein", along=0.7)
    assert V.tourniquet_pext(vein, loose, sbp=sbp, circ_m=0.5) == 20e3
    # and the flow follows: distal cut stops, proximal cut bleeds on
    f = 1.0
    q_d = V.flow_ml_s(distal, f, p_ext=V.tourniquet_pext(distal, tq, sbp=sbp, circ_m=0.5))
    q_p = V.flow_ml_s(proximal, f, p_ext=V.tourniquet_pext(proximal, tq, sbp=sbp, circ_m=0.5))
    assert q_d == 0.0 and q_p == 15.0


def test_p_ext_may_be_a_list_or_a_dict_by_vessel():
    c1, c2 = cut_of("femoral artery"), cut_of("femoral vein")
    c3, c4 = cut_of("femoral artery"), cut_of("femoral vein")
    a = V.step_wound([c1, c2], V0, V0, [20e3, 0.0], 1.0, 1.0, 37.0, 0.0)
    b = V.step_wound([c3, c4], V0, V0, {"femoral artery": 20e3}, 1.0, 1.0, 37.0, 0.0)
    assert a == b and a == pytest.approx(6.0)                         # the vein only


# ── ischaemia ───────────────────────────────────────────────────────────────

def test_ischaemia_2_hours_is_under_10_percent_dead_tissue_closed_form():
    d = V.isch_dead_frac(2 * 3600.0)
    assert d == pytest.approx(1.0 - math.exp(-(2.0 / 8.0) ** 2), rel=1e-12)
    assert d == pytest.approx(0.0606, abs=5e-4) and d < 0.10
    assert V.isch_dead_frac(0.0) == 0.0
    assert V.isch_dead_frac(8 * 3600.0) == pytest.approx(1 - math.exp(-1.0))
    ts = [i * 1800.0 for i in range(0, 30)]
    ds = [V.isch_dead_frac(t) for t in ts]
    assert ds == sorted(ds) and ds[-1] < 1.0
    assert V.isch_dead_combine(0.0, 7200.0) == pytest.approx(d)


# ── closed bleeding ─────────────────────────────────────────────────────────

def test_pooling_saturates_at_the_cavity_and_never_exceeds_it():
    pooled, V_ml, f = 0.0, 4900.0, 1.0
    last = 0.0
    for _ in range(2000):
        ml = V.pool_step("left thigh", pooled, f, 60.0, V_ml=V_ml)
        pooled += ml
        V_ml -= ml
        f = V_ml / V0
        assert pooled <= 1500.0 + 1e-9
        assert pooled >= last
        last = pooled
    assert pooled < 1500.0 and pooled > 1000.0           # MAP falls as it pools: it slows before the cavity is full


def test_pooling_is_pressure_driven_and_has_the_1800_s_time_constant_at_full_map():
    ml = V.pool_step("left thigh", 0.0, 1.0, 1800.0)
    assert ml == pytest.approx(1500.0 * (1 - math.exp(-1.0)), rel=1e-12)
    assert V.pool_step("left thigh", 0.0, 0.525, 1800.0) < ml           # half pressure: slower
    assert V.pool_step("left thigh", 0.0, 0.25, 1800.0) == 0.0          # MAP_rel 0
    assert V.pool_step("left thigh", 0.0, 1.0, 1800.0, alive=False) == 0.0
    assert V.pool_step("left thigh", 1500.0, 1.0, 1800.0) == 0.0         # the cavity is full
    # a splint's tamponade slows it
    assert V.pool_step("left thigh", 0.0, 1.0, 600.0, splint=1.0) < V.pool_step("left thigh", 0.0, 1.0, 600.0)
    # exact for any step: two half steps equal one whole (MAP_rel held)
    a = V.pool_step("left thigh", 0.0, 1.0, 600.0)
    b = V.pool_step("left thigh", a, 1.0, 600.0)
    assert a + b == pytest.approx(V.pool_step("left thigh", 0.0, 1.0, 1200.0), rel=1e-12)


def test_cavity_sizes_by_region_and_the_clamp_to_the_blood_there_is():
    assert V.cavity_ml("abdomen") == 2500.0 and V.cavity_ml("left shin") == 750.0
    assert V.cavity_ml("left thigh", adult()) == 1500.0
    assert V.cavity_ml("left thigh", A.make_sheet(mass_kg=35.0, height_m=1.4, draws=False)) == 750.0
    # a bilateral femur plus pelvis in a small body cannot pool more than the blood there is
    V_ml = 300.0
    out = V.pool_step("abdomen", 0.0, 1.0, 1e6, V_ml=V_ml)
    assert out == V_ml


def test_blood_ledger_closes_every_step_with_poured_pooled_and_circulating():
    V_ml, poured, pooled = V0, 0.0, 0.0
    cut_a = cut_of("femoral artery", 0.6)
    cut_v = cut_of("femoral vein", 0.8)
    for step in range(1200):
        ml = V.step_wound([cut_a, cut_v], V_ml, V0, [0.0, 0.0], 1.0, 0.5, 37.0, 0.0)
        V_ml -= ml
        poured += ml
        p = V.pool_step("left thigh", pooled, V_ml / V0, 0.5, V_ml=V_ml)
        pooled += p
        V_ml -= p
        assert V_ml >= 0.0
        assert V_ml + poured + pooled == pytest.approx(V0, abs=1e-9, rel=0)
    assert poured > 0 and pooled > 0


def test_a_dead_body_bleeds_nothing_and_pools_nothing():
    c = cut_of("femoral artery")
    assert V.step_wound([c], V0, V0, [0.0], 1.0, 1.0, 37.0, 0.0, alive=False) == 0.0
    assert V.pool_step("left thigh", 0.0, 1.0, 600.0, alive=False) == 0.0
    assert V.flow_ml_s(c, 1.0, alive=False) == 0.0


def test_the_step_is_clamped_to_the_blood_there_is():
    c = cut_of("femoral artery")
    assert V.step_wound([c], 2940.0, V0, [0.0], 1.0, 1000.0, 37.0, 0.0) == pytest.approx(2940.0)
    c = cut_of("femoral artery")
    assert V.step_wound([c], 0.0, V0, [0.0], 1.0, 1.0, 37.0, 0.0) == 0.0


def test_cold_blood_clots_slower():
    warm, cold = cut_of("scalp arteries"), cut_of("scalp arteries")
    a = bleed(warm, tmax=900.0, T=37.0)
    b = bleed(cold, tmax=900.0, T=27.0)
    assert b["t"] > a["t"]


def test_internal_organ_cuts_flow_like_any_cut_and_pool():
    s = adult()
    c = V.internal_cut("liver", 1.0, s)
    assert c["internal"] and c["Q_full"] == pytest.approx(8.0) and c["region"] == "abdomen"
    assert V.internal_cut("aorta", 1.0, s)["Q_full"] == pytest.approx(30.0)
    lung = V.internal_cut("lung", 1.0, s)
    assert lung["kind"] == "v" and V.flow_ml_s(lung, 1.0) < 2.0           # low pressure


def test_codeonly_the_damage_combine_is_the_one_rule():
    # imported from matdamage when it exists, a local copy otherwise: same answer
    assert V.isch_dead_combine(0.5, 0.0) == pytest.approx(0.5 + 0.0)
    assert V.isch_dead_combine(0.5, 8 * 3600.0) == pytest.approx(1 - 0.5 * math.exp(-1.0))
