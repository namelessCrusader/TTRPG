"""Bone records: capacity, the elliptic break rule, load bearing, healing (src/voxel/bonestate.py, wave 13 T10).

Tags: test_echo_* checks a number the constants were FITTED to; test_codeonly_*
checks the code against its own table; test_u_* leans on an unverified number;
untagged tests are exact (arithmetic identities, closed forms).
"""
import math

import pytest

from src.voxel import anatomy as A
from src.voxel import bonestate as B

G = 9.81
WEEK = 7 * 86400.0


def adult():
    return A.make_sheet(mass_kg=70.0, height_m=1.75, draws=False)


def stock():
    return A.make_sheet(draws=False)


def broken_shin(sheet=None, crush=1.0):
    r = B.new_bone()
    B.break_(r, "left shin", sheet or adult(), 9000.0, crush)
    return r


# ── capacity and the break rule ─────────────────────────────────────────────

def test_capacity_is_the_member_table_times_bone_scale_for_an_intact_bone():
    s = adult()
    r = B.new_bone()
    assert B.capacity(r, "left shin", s) == (8500.0, pytest.approx(A.m_fail_adult("tibia")))
    assert B.capacity(r, "left thigh", s)[0] == 12000.0
    s2 = stock()
    assert B.capacity(r, "left thigh", s2)[0] == pytest.approx(12000.0 * s2["bone_scale"])
    assert B.capacity(r, "left thigh", s2)[1] == pytest.approx(A.m_fail_adult("femur") * s2["bone_scale"])
    # a name that is itself a member
    assert B.capacity(r, "calcaneus", s)[0] == 6500.0


def test_no_break_below_F_fail_with_no_lateral_moment():
    s = adult()
    r = B.new_bone()
    Ff = B.capacity(r, "left thigh", s)[0]
    for x in (0.0, 0.3, 0.9, 0.999):
        assert not B.check(r, "left thigh", s, x * Ff)
    assert B.check(r, "left thigh", s, Ff)                   # at the failure load it fails
    assert B.check(r, "left thigh", s, 1.5 * Ff)


def test_a_lateral_moment_above_M_fail_breaks_with_no_axial_load():
    s = adult()
    r = B.new_bone()
    Mf = B.capacity(r, "left thigh", s)[1]
    assert not B.check(r, "left thigh", s, 0.0, lateral_Nm=0.99 * Mf)
    assert B.check(r, "left thigh", s, 0.0, lateral_Nm=Mf)
    assert B.check(r, "left thigh", s, 0.0, lateral_Nm=1.2 * Mf)


def test_the_elliptic_rule_axial_and_lateral_trade_off():
    s = adult()
    r = B.new_bone()
    Ff, Mf = B.capacity(r, "left thigh", s)
    # half the lateral capacity leaves sqrt(1 - 1/4) of the axial one
    ax = Ff * math.sqrt(0.75)
    assert B.check(r, "left thigh", s, ax * 1.0001, lateral_Nm=0.5 * Mf)
    assert not B.check(r, "left thigh", s, ax * 0.9999, lateral_Nm=0.5 * Mf)


def test_there_is_no_hidden_eccentricity_axial_below_F_fail_never_breaks():
    s = adult()
    r = B.new_bone()
    for name in ("left shin", "left thigh", "left upper arm", "left forearm", "head", "torso"):
        Ff = B.capacity(r, name, s)[0]
        assert not B.check(r, name, s, 0.9999 * Ff, lateral_Nm=0.0)


def test_break_force_is_exactly_where_check_flips():
    s = adult()
    r = B.new_bone()
    for M in (0.0, 80.0, 200.0, 300.0):
        F = B.break_force(r, "left thigh", s, lateral_Nm=M)
        assert B.check(r, "left thigh", s, F * (1.0 + 1e-9), lateral_Nm=M)
        assert not B.check(r, "left thigh", s, F * (1.0 - 1e-9), lateral_Nm=M)
    Mf = B.capacity(r, "left thigh", s)[1]
    assert B.break_force(r, "left thigh", s, lateral_Nm=Mf) == 0.0


def test_break_force_with_dynamic_factor_and_referral():
    s = adult()
    r = B.new_bone()
    F = B.break_force(r, "left shin", s, dyn=1.25, m_above=0.96)
    assert F == pytest.approx(1.25 * 8500.0 / 0.96)
    assert B.check(r, "left shin", s, 1.25 * 8500.0 * 1.0001, dyn=1.25)
    assert not B.check(r, "left shin", s, 1.25 * 8500.0 * 0.9999, dyn=1.25)


# ── the break record and load bearing ───────────────────────────────────────

def test_break_sets_integrity_gap_angle_and_peak():
    s = adult()
    r = B.new_bone()
    B.break_(r, "left shin", s, 9100.0, 0.4)
    assert r["integrity"] == pytest.approx(0.6)
    assert r["gap_m"] == pytest.approx(0.5e-3 + 2.5e-3 * 0.4)
    assert r["angle"] == pytest.approx(0.2 * 0.4)
    assert r["peak_N"] == 9100.0 and r["member"] == "tibia"
    B.break_(r, "left shin", s, 5000.0, 1.0, side=-1.0)
    assert r["integrity"] == 0.0 and r["gap_m"] == pytest.approx(3.0e-3)
    assert r["angle"] == pytest.approx(-0.2) and r["peak_N"] == 9100.0
    r2 = B.new_bone()
    B.break_(r2, "left shin", s, 100.0, 0.0)               # no crush: not a break
    assert r2["integrity"] == 1.0 and r2["gap_m"] == 0.0


def test_a_second_break_keeps_the_more_limiting_member():
    s = adult()
    r = B.new_bone()
    B.break_(r, "left shin", s, 9000.0, 0.2, member="calcaneus")
    assert r["member"] == "calcaneus"
    B.break_(r, "left shin", s, 9000.0, 1.0, member="tibia")
    assert r["integrity"] == 0.0 and r["member"] == "tibia"


def test_residual_formula_and_cap():
    assert B.residual({"callus": 0.0, "splint": 0.0}) == pytest.approx(0.02)
    assert B.residual({"callus": 0.0, "splint": 0.6}) == pytest.approx(0.02 + 0.3)
    assert B.residual({"callus": 0.5, "splint": 0.0}) == pytest.approx(0.02 + 0.98 * 0.125)
    assert B.residual({"callus": 1.0, "splint": 1.0}) == 1.0


def test_u_a_broken_shin_carries_under_3_percent_and_under_half_a_body_weight():
    s = adult()
    r = broken_shin(s)
    cap = B.cap_N(r, "left shin", s)
    assert cap / 8500.0 < 0.03
    assert cap < 0.5 * 70.0 * G
    assert cap == pytest.approx(0.02 * 8500.0)
    assert not B.bears(r, "left shin", s, 70.0 * G)
    assert not B.bears(r, "left shin", s, 0.5 * 70.0 * G)
    assert B.bears(r, "left shin", s, 100.0)
    # the stock figure too: its shin is weaker, its body lighter
    st = stock()
    r2 = broken_shin(st)
    assert B.cap_N(r2, "left shin", st) < 0.5 * 43.9 * G


def test_a_splint_raises_capacity_and_the_break_check_uses_the_same_function():
    s = adult()
    r = broken_shin(s)
    c0 = B.cap_N(r, "left shin", s)
    r["splint"] = B.splint_fraction(200.0, 200.0)               # an equal plank: 0.5
    c1 = B.cap_N(r, "left shin", s)
    assert c1 > c0 * 10
    assert B.bears(r, "left shin", s, 70.0 * G)                  # 0.02 + 0.25 = 0.27 of 8.5 kN
    # check() reads the same function: a broken shin fails at the capacity bears() says
    assert B.check(r, "left shin", s, c1 * 1.0001) and not B.check(r, "left shin", s, c1 * 0.9999)


def test_splint_fraction_and_bone_EI():
    assert B.splint_fraction(1.0, 3.0) == pytest.approx(0.25)
    assert B.splint_fraction(0.0, 5.0) == 0.0
    EI = B.bone_EI("left shin", adult())
    assert EI == pytest.approx(17e9 * math.pi / 64 * (0.027 ** 4 - 0.017 ** 4), rel=1e-9)


def test_reduce_zeros_the_angle_and_shrinks_the_gap():
    s = adult()
    r = broken_shin(s, 0.7)
    r["angle"] = 0.14
    B.reduce(r)
    assert r["angle"] == 0.0 and r["gap_m"] == pytest.approx(0.5e-3)
    r0 = B.new_bone()
    B.reduce(r0)
    assert r0["gap_m"] == 0.0


# ── healing ─────────────────────────────────────────────────────────────────

def _weeks_until_bearing(s, rec, peak, load_BW=1.2, dt=3600.0):
    W = s["mass_kg"] * G
    t = 0.0
    while not B.bears(rec, "left shin", s, load_BW * W):
        B.heal_step(rec, "left shin", s, dt, peak_N=peak)
        t += dt
        if t > 60 * WEEK:
            return None
    return t / WEEK


def test_u_1_2_BW_load_bearing_returns_at_11_weeks_for_a_70kg_tibia():
    s = adult()
    r = broken_shin(s)
    weeks = _weeks_until_bearing(s, r, B.peak_for("lying", mass_kg=70.0))
    assert weeks is not None and 9.0 <= weeks <= 14.0, weeks
    assert weeks == pytest.approx(11.2, abs=0.3)          # tau 20 weeks, c^3 = 0.0785
    # closed form: residual = 0.02 + 0.98 c^3 = 1.2 W / F_fail, c = 1 - exp(-t/tau)
    c = ((1.2 * 70 * G / 8500.0 - 0.02) / 0.98) ** (1.0 / 3.0)
    assert weeks == pytest.approx(-20.0 * math.log(1.0 - c), abs=0.05)


def test_u_the_stock_figure_returns_earlier_than_the_adult():
    s = stock()
    r = broken_shin(s)
    weeks = _weeks_until_bearing(s, r, B.peak_for("lying", mass_kg=43.9))
    assert weeks is not None and 9.0 <= weeks <= 11.5, weeks


def test_one_big_healing_step_equals_many_hourly_steps():
    s = adult()
    a, b = broken_shin(s), broken_shin(s)
    B.heal_step(a, "left shin", s, 8 * WEEK, peak_N=34.0)
    for _ in range(8 * 7 * 24):
        B.heal_step(b, "left shin", s, 3600.0, peak_N=34.0)
    assert a["callus"] == pytest.approx(b["callus"], rel=1e-9)
    assert a["callus"] == pytest.approx(1.0 - math.exp(-8.0 / 20.0), rel=1e-9)


def test_callus_rises_monotonically_inside_the_strain_window():
    s = adult()
    r = broken_shin(s)
    last = 0.0
    for _ in range(300):
        B.heal_step(r, "left shin", s, 3600.0, peak_N=34.0)
        assert r["callus"] >= last
        last = r["callus"]
    assert last > 0.05


def test_u_strain_above_0_10_never_heals_non_union():
    s = adult()
    r = broken_shin(s)
    walk = B.peak_for(None, "walk", mass_kg=70.0)
    assert B.strain(r, "left shin", s, walk) > 0.10
    for _ in range(100):
        B.heal_step(r, "left shin", s, 7 * 86400.0, peak_N=walk)
    assert r["callus"] == 0.0 and r["nonunion"] is True
    assert not B.bears(r, "left shin", s, 70.0 * G)


def test_u_a_splint_lets_a_walking_shin_heal():
    s = adult()
    r = broken_shin(s)
    r["splint"] = 0.5
    walk = B.peak_for(None, "walk", mass_kg=70.0)
    assert B.strain(r, "left shin", s, walk) < 0.10
    B.heal_step(r, "left shin", s, 4 * WEEK, peak_N=walk)
    assert r["callus"] > 0.1 and not r["nonunion"]


def test_healing_unites_the_bone_and_an_intact_bone_does_not_change():
    s = adult()
    r = broken_shin(s)
    B.heal_step(r, "left shin", s, 400 * WEEK, peak_N=34.0)
    assert r["integrity"] == 1.0 and r["gap_m"] == 0.0 and not B.is_broken(r)
    intact = B.new_bone()
    before = dict(intact)
    B.heal_step(intact, "left shin", s, WEEK, peak_N=1e6)
    assert intact == before


def test_codeonly_tau_by_bone_in_weeks():
    r = B.new_bone()
    assert B.tau_s(r, "left shin") == 20 * WEEK and B.tau_s(r, "left thigh") == 18 * WEEK
    assert B.tau_s(r, "left upper arm") == 12 * WEEK and B.tau_s(r, "left forearm") == 10 * WEEK
    assert B.tau_s(r, "head") == 16 * WEEK


# ── loads, gait ─────────────────────────────────────────────────────────────

def test_peak_for_orders_lying_below_standing_below_walking():
    m = 70.0
    lie, sit = B.peak_for("lying", mass_kg=m), B.peak_for("sitting", mass_kg=m)
    stand = B.peak_for("standing", mass_kg=m)
    one = B.peak_for("standing", mass_kg=m, legs=1)
    walk, hop = B.peak_for(None, "walk", mass_kg=m), B.peak_for(None, "hop", mass_kg=m)
    assert lie < sit < stand < one < walk < hop
    W = m * G
    assert lie == pytest.approx(0.05 * W) and sit == pytest.approx(0.1 * W)
    assert stand == pytest.approx(0.5 * W) and one == pytest.approx(W)
    assert walk == pytest.approx(1.2 * W) and hop == pytest.approx(2.0 * W)
    assert B.peak_for(None, "walk", mass_kg=m, ratio=0.5) == pytest.approx(0.6 * W)


def test_leg_ratio_and_gait_classes_at_the_boundaries():
    s = adult()
    W = 70.0 * G
    bones = {}
    assert B.leg_ratio(bones, "left", s, W) == 1.0
    # capacity exactly 1.2 W walks at ratio 1
    bones = {"left shin": {"integrity": 1.0, "callus": 0.0, "splint": 0.0}}
    assert B.leg_ratio(bones, "left", s, W) == 1.0           # 8.5 kN >> 1.2 W
    br = broken_shin(s)
    bones["left shin"] = br
    ratio = B.leg_ratio(bones, "left", s, W)
    assert ratio == pytest.approx(170.0 / (1.2 * W)) and ratio < 0.3
    assert B.gait_class(1.0, 1.0) == "walk"
    assert B.gait_class(0.3, 1.0) == "limp" and B.gait_class(1.0, 0.3) == "limp"
    assert B.gait_class(0.29999, 1.0) == "hop" and B.gait_class(0.5, 0.1) == "hop"
    assert B.gait_class(0.1, 0.2) == "crawl" and B.gait_class(0.0, 0.0) == "crawl"
    assert B.gait_class(0.99, 1.0) == "limp"


def test_pace_factor_is_0_4_plus_0_6_ratio():
    assert B.pace_factor(1.0) == 1.0
    assert B.pace_factor(0.5) == pytest.approx(1.0 / 0.7)
    assert B.pace_factor(0.3) == pytest.approx(1.0 / 0.58)
    assert B.pace_factor(0.0) == pytest.approx(1.0 / 0.4)


def test_the_weaker_of_shin_and_thigh_sets_the_leg_ratio():
    s = adult()
    W = 70.0 * G
    thigh = B.new_bone()
    B.break_(thigh, "right thigh", s, 14000.0, 1.0)
    bones = {"right shin": B.new_bone(), "right thigh": thigh}
    assert B.leg_ratio(bones, "right", s, W) == pytest.approx(0.02 * 12000.0 / (1.2 * W))
    assert B.leg_ratio(bones, "left", s, W) == 1.0


# ── every member heals at bed rest; union hands the record back clean ───────

@pytest.mark.parametrize("member,name", [
    ("calcaneus", "left shin"), ("tibia", "left shin"), ("femur", "left thigh"),
    ("femoral_neck", "left thigh"), ("pelvis", "torso"), ("lumbar", "torso"),
    ("ribs", "torso"), ("humerus", "left upper arm"), ("clavicle", "left upper arm"),
    ("forearm", "left forearm"), ("radius_distal", "left forearm"), ("skull", "head"),
])
def test_u_every_member_is_inside_the_strain_window_at_bed_rest(member, name):
    # a fully crushed member (widest gap) at the lying load (0.05 BW) must heal
    s = adult()
    r = B.new_bone()
    B.break_(r, name, s, 1000.0, 1.0, member=member)
    eps = B.strain(r, name, s, B.peak_for("lying", mass_kg=70.0), member=member)
    assert eps <= B.EPS_WINDOW, (member, eps)
    for _ in range(200):
        B.heal_step(r, name, s, WEEK, B.peak_for("lying", mass_kg=70.0), member=member)
    assert r["integrity"] == 1.0 and r["gap_m"] == 0.0, member


def test_a_united_bone_is_a_plain_bone_again():
    s = adult()
    r = B.new_bone()
    B.break_(r, "left shin", s, 1000.0, 0.5, member="calcaneus")
    for _ in range(200):
        B.heal_step(r, "left shin", s, WEEK, 34.0)
    assert r["integrity"] == 1.0
    assert "member" not in r and r["angle"] == 0.0
    assert B.cap_N(r, "left shin", s) == B.cap_N(B.new_bone(), "left shin", s) == 8500.0


# ── regression (P-T10T11): a partial fracture heals by the SAME callus law ──

def test_a_partial_shaft_fracture_heals_at_bed_rest_like_a_full_break_random_sweep():
    # Cause: k_fx ignored the part of the section that is still whole, and a
    # partial break has a smaller gap, so its strain came out HIGHER than a full
    # break's and it never left the non-union band. Every member, random people,
    # random crush: the callus after n weeks is the closed form 1 - exp(-n/tau),
    # the same for a crush of 0.05 and of 1.0, and the bone unites.
    import random
    rng = random.Random(1310)
    members = [(m, n) for m, n in [
        ("tibia", "left shin"), ("femur", "left thigh"), ("humerus", "left upper arm"),
        ("forearm", "left forearm"), ("calcaneus", "left shin"), ("femoral_neck", "left thigh"),
        ("ribs", "torso"), ("lumbar", "torso"), ("pelvis", "torso"), ("clavicle", "left upper arm"),
        ("radius_distal", "left forearm"), ("skull", "head")]]
    n_in = n_out = 0
    for _ in range(120):
        mass = rng.uniform(40.0, 110.0)
        # a real build: BMI 18.5 to 32
        bmi = rng.uniform(18.5, 32.0)
        height = min(1.95, max(1.5, math.sqrt(mass / bmi)))
        mass = bmi * height * height
        s = A.make_sheet(mass_kg=mass, height_m=height, pid=rng.randrange(10**6),
                         name="x%d" % rng.randrange(99), draws=True)
        member, name = rng.choice(members)
        crush = rng.uniform(0.02, 1.0)
        pk = B.peak_for("lying", mass_kg=mass, bone=name)        # each bone at its own limb's lying load (N7)
        full, part = B.new_bone(), B.new_bone()
        B.break_(full, name, s, 9000.0, 1.0, member=member)
        B.break_(part, name, s, 9000.0, crush, member=member)
        sp, sf = (B.strain(r, name, s, pk, member=member) for r in (part, full))
        assert sp <= sf + 1e-15            # the whole part of the section only stiffens it
        if sf > B.EPS_WINDOW:              # the full break itself does not heal for this build
            n_out += 1                     # (with the leg's lying load on every bone, 5 of these 120 did: N7)
            continue
        n_in += 1
        assert sp <= B.EPS_WINDOW          # so a partial break heals wherever a full one does
        weeks = rng.randrange(1, 9)
        for r in (full, part):
            for _ in range(weeks):
                B.heal_step(r, name, s, WEEK, pk, member=member)
        tau = B.tau_s(full, name, member) / WEEK
        want = 1.0 - math.exp(-weeks / tau)
        assert full["callus"] == pytest.approx(want, rel=1e-9)
        assert part["callus"] == pytest.approx(full["callus"], rel=1e-12)
        assert not part["nonunion"]
        for r in (full, part):
            for _ in range(400):
                B.heal_step(r, name, s, WEEK, pk, member=member)
            assert r["integrity"] == 1.0 and r["gap_m"] == 0.0
    assert n_out == 0 and n_in > 40, (n_in, n_out)


def test_u_a_tall_heavy_mans_full_forearm_break_heals_at_bed_rest():
    """Decision N7 (was a strict xfail). The leg's lying load, 0.05 BW, was applied to an arm bone:
    this 91 kg, 1.91 m man's FULL forearm break sat at strain 0.107 against the 0.10 window and
    never healed. An arm bone has its own load class now (0.022 BW lying, see ARM_LOAD_BW); the
    window is the same 0.10 and the strain is 0.047."""
    s = A.make_sheet(mass_kg=91.26, height_m=1.911, pid=1, name="x", draws=False)
    r = B.new_bone()
    B.break_(r, "left forearm", s, 9000.0, 1.0, member="forearm")
    assert B.EPS_WINDOW == 0.10
    assert B.strain(r, "left forearm", s, B.peak_for("lying", mass_kg=91.26), member="forearm") > B.EPS_WINDOW
    eps = B.strain(r, "left forearm", s, B.peak_for("lying", mass_kg=91.26, bone="left forearm"), member="forearm")
    assert eps == pytest.approx(0.047, abs=0.002) and eps <= B.EPS_WINDOW
    for _ in range(200):
        B.heal_step(r, "left forearm", s, WEEK, B.peak_for("lying", mass_kg=91.26, bone="left forearm"), member="forearm")
    assert r["integrity"] == 1.0 and r["gap_m"] == 0.0


def test_each_limb_has_its_own_load_class_and_the_leg_numbers_are_unchanged():
    m = 70.0
    W = m * G
    for b in (None, "left shin", "right thigh", "torso", "head"):      # legs, no bone, torso, head: as before
        assert B.peak_for("lying", mass_kg=m, bone=b) == pytest.approx(0.05 * W)
        assert B.peak_for("sitting", mass_kg=m, bone=b) == pytest.approx(0.1 * W)
        assert B.peak_for("standing", mass_kg=m, bone=b) == pytest.approx(0.5 * W)
        assert B.peak_for("standing", mass_kg=m, legs=1, bone=b) == pytest.approx(1.0 * W)
        assert B.peak_for(None, "walk", mass_kg=m, ratio=0.5, bone=b) == pytest.approx(0.6 * W)
    w = A.WINTER                                                       # Winter's segment fractions
    rest = w["forearm"] + w["hand"]                                    # 0.022
    whole = w["upper arm"] + w["forearm"] + w["hand"]                  # 0.050
    for b in ("left forearm", "right forearm", "left upper arm", "right upper arm"):
        assert B.limb_of(b) == "arm"
        assert B.peak_for("lying", mass_kg=m, bone=b) == pytest.approx(rest * W)
        assert B.peak_for("sitting", mass_kg=m, bone=b) == pytest.approx(rest * W)
        assert B.peak_for("standing", mass_kg=m, bone=b) == pytest.approx(whole * W)
        assert B.peak_for("lying", mass_kg=m, bone=b) < B.peak_for("lying", mass_kg=m, bone="left shin")
        # the legs walk, hop or run: the arm hangs; on all fours every limb carries a quarter
        for gait in ("walk", "limp", "hop", "run"):
            assert B.peak_for(None, gait, mass_kg=m, bone=b) == pytest.approx(whole * W)
        assert B.peak_for(None, "crawl", mass_kg=m, bone=b) == pytest.approx(0.25 * W)
    assert B.limb_of("left shin") == B.limb_of("right thigh") == "leg"
    assert B.limb_of("torso") is None and B.limb_of("head") is None and B.limb_of(None) is None


def test_u_a_full_arm_break_heals_at_bed_rest_for_every_build_with_headroom():
    """1000 random people (35 to 150 kg, 1.45 to 2.05 m, no BMI limit), a full break of every arm
    member, each at the ARM's lying load: none leaves the 10 percent window, and the worst has
    headroom (it is under 0.8 of the window), so a (u) arm number that is a little off does not
    flip the answer. The NEGATIVE CONTROL: the leg's 0.05 BW on the same bones leaves the window
    for a good share of them (473 of 3000 in a wider run), so this sweep can tell the two apart."""
    import random
    rng = random.Random(1311)
    arms = [("humerus", "left upper arm"), ("forearm", "left forearm"),
            ("clavicle", "left upper arm"), ("radius_distal", "left forearm")]
    worst, leg_out = 0.0, 0
    for _ in range(1000):
        mass, height = rng.uniform(35.0, 150.0), rng.uniform(1.45, 2.05)
        s = A.make_sheet(mass_kg=mass, height_m=height, pid=rng.randrange(10**6),
                         name="x%d" % rng.randrange(99), draws=True)
        member, name = rng.choice(arms)
        r = B.new_bone()
        B.break_(r, name, s, 9000.0, 1.0, member=member)
        eps = B.strain(r, name, s, B.peak_for("lying", mass_kg=mass, bone=name), member=member)
        worst = max(worst, eps)
        assert eps <= B.EPS_WINDOW, (mass, height, member, eps)
        leg_out += B.strain(r, name, s, B.peak_for("lying", mass_kg=mass), member=member) > B.EPS_WINDOW
    assert worst < 0.8 * B.EPS_WINDOW, worst
    assert leg_out > 100, leg_out


def test_a_partial_shin_fracture_unites_at_bed_rest_in_the_weeks_of_the_callus_law():
    s = adult()
    for crush in (0.37, 0.63, 0.9):
        r = broken_shin(s, crush)
        pk = B.peak_for("lying", mass_kg=70.0)
        n = 0
        while r["integrity"] < 1.0 and n < 400:
            B.heal_step(r, "left shin", s, WEEK, pk)
            n += 1
        # united when callus reaches UNITED: n = -tau ln(1 - UNITED), tau 20 weeks
        assert n == math.ceil(-20.0 * math.log(1.0 - B.UNITED)), (crush, n)
