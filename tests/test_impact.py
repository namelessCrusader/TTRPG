"""Impact: the fuse chain, landings, blows, the head (src/voxel/impact.py, wave 13 T10).

Tags: test_echo_* checks a number the constants were FITTED to (the landing
shares and the side/back organ tolerances were tuned to the survivors' mean and
LD50 targets); test_codeonly_* checks the code against its own table; test_u_*
leans on an unverified number; untagged tests are exact (balance, conservation,
monotonicity, identities).
"""
import math

import pytest

from src.voxel import anatomy as A
from src.voxel import bonestate as B
from src.voxel import impact as I

G = 9.81


def adult():
    return A.make_sheet(mass_kg=70.0, height_m=1.75, draws=False)


def stock():
    return A.make_sheet(draws=False)


ADULT_KW = dict(legs_N=2060.0, crouch_m=0.286)
STOCK_KW = dict(legs_N=1400.0, crouch_m=0.252)


def names(res):
    return [f["name"] for f in res["failed"]]


def fall(sheet, h, kw, **more):
    return I.landing(sheet, "feet", h, prep=more.pop("prep", 1.0), **kw, **more)


# ── the chain: balance and conservation ─────────────────────────────────────

def _balance(res):
    W, k, F, w = res["W_N"], res["k_el"], res["F_peak"], res["work"]
    h = res["h_eff"]
    lhs = (F - W) ** 2 / (2 * k) + w["plateau_J"]
    return lhs, W * h + W * W / (2 * k)


def test_the_balance_holds_with_gravity_work_and_partial_crush_to_1e_9():
    s = adult()
    seen_partial = seen_plateau = 0
    for h in (0.5, 1, 2, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 12, 15, 20):
        for kw in ({"share_feet": (1, 1)}, {"share_feet": (1, 0)}):
            for prep in (0.0, 0.4, 1.0):
                r = I.landing(s, "feet", h, prep=prep, **ADULT_KW, **kw)
                lhs, rhs = _balance(r)
                assert lhs == pytest.approx(rhs, rel=1e-9, abs=1e-9), (h, kw, prep)
                if any(0.0 < f["frac"] < 1.0 for f in r["failed"]):
                    seen_partial += 1
                if r["F_peak"] in [f["C"] for f in r["failed"]]:
                    seen_plateau += 1
    assert seen_partial > 3 and seen_plateau > 3, "the sweep must visit plateau points"


def test_energy_sinks_sum_to_W_times_h_plus_stroke():
    for s, kw in ((adult(), ADULT_KW), (stock(), STOCK_KW)):
        for h in (1, 2, 4, 6, 10, 15):
            for share in ((1, 1), (1, 0)):
                r = I.landing(s, "feet", h, prep=1.0, share_feet=share, **kw)
                w = r["work"]
                assert w["total_J"] == pytest.approx(w["W_h_X_J"], rel=1e-12)
                assert w["total_J"] == pytest.approx(r["W_N"] * (h + w["X_m"]), rel=1e-9)
                assert w["elastic_J"] >= 0 and w["bone_J"] >= 0 and w["control_J"] >= 0


def test_the_sinks_are_booked_where_they_belong():
    r = fall(adult(), 5.0, ADULT_KW)
    w = r["work"]
    assert w["control_J"] > 0                                   # the legs flexed: heat in the leg muscle
    assert w["bone_J"] > 0                                      # bones crushed at 5 m
    assert w["bone_J"] == pytest.approx(sum(f["C"] * f["crush_used"] for f in r["failed"]
                                            if f["kind"] == "bone"), rel=1e-12)
    easy = fall(adult(), 1.0, ADULT_KW)
    assert easy["work"]["bone_J"] == 0.0 and easy["work"]["control_J"] > 0
    assert easy["contusion_J"] == easy["work"]["elastic_J"]


def test_F_peak_is_monotone_in_height():
    s = adult()
    last = 0.0
    for h in [0.25 * i for i in range(1, 100)]:
        F = fall(s, h, ADULT_KW)["F_peak"]
        assert F >= last
        last = F


def test_adding_a_cheaper_element_never_raises_the_peak():
    W, k = 700.0, 70e3
    base = [dict(name="a", C=9000.0, crush=0.02), dict(name="b", C=15000.0, crush=0.03)]
    for h in (1.0, 3.0, 6.0, 12.0):
        F0 = I.chain(W, h, k, base)[0]
        for C in (2000.0, 5000.0, 8000.0, 12000.0):
            F1 = I.chain(W, h, k, base + [dict(name="cheap", C=C, crush=0.01)])[0]
            assert F1 <= F0 + 1e-9, (h, C)


def test_failed_list_is_ordered_by_referred_capacity():
    r = fall(adult(), 12.0, ADULT_KW)
    caps = [f["C"] for f in r["failed"]]
    assert caps == sorted(caps) and len(caps) >= 3
    assert names(r) == ["lumbar", "calcaneus", "pelvis", "tibia"]


def test_the_last_failure_is_partial_when_the_energy_ends_inside_a_step():
    r = fall(adult(), 4.0, ADULT_KW)
    assert names(r) == ["lumbar"]
    assert 0.0 < r["failed"][0]["frac"] < 1.0
    assert r["F_peak"] == pytest.approx(r["failed"][0]["C"])      # the plateau: F = C_i exactly
    # a partial failure: crush used = frac x stroke
    f = r["failed"][0]
    assert f["crush_used"] == pytest.approx(f["frac"] * f["crush"])


def test_an_element_below_body_weight_has_already_failed_by_standing():
    W = 700.0
    F, failed, work = I.chain(W, 0.0, 70e3, [dict(name="weak", C=500.0, crush=0.02, kind="bone"),
                                             dict(name="ok", C=9000.0, crush=0.02, kind="bone")])
    assert failed[0]["name"] == "weak" and failed[0]["pre_failed"] and failed[0]["crush_used"] == 0.0
    s = adult()
    s2 = dict(s, bone_scale=0.01)                               # a bone far too weak for the body
    r = I.landing(s2, "feet", 0.0, prep=1.0, **ADULT_KW)
    assert any(f.get("pre_failed") for f in r["failed"]), names(r)


def test_stiff_landing_peak_is_higher_than_prepared():
    s = adult()
    for h in (1, 2, 3):
        assert I.landing(s, "feet", h, prep=0.0, **ADULT_KW)["F_peak"] \
            > I.landing(s, "feet", h, prep=1.0, **ADULT_KW)["F_peak"]


def test_prep_rises_with_airborne_time_and_is_zero_unconscious():
    assert I.prep_of(0.05) == 0.0 and I.prep_of(0.10) == 0.0
    assert I.prep_of(0.20) == pytest.approx(0.5) and I.prep_of(0.30) == pytest.approx(1.0)
    assert I.prep_of(0.9) == 1.0
    ts = [0.05 * i for i in range(0, 12)]
    ps = [I.prep_of(t) for t in ts]
    assert ps == sorted(ps)
    assert I.prep_of(2.0, aware=False) == 0.0
    s = adult()
    # a body that was airborne a long time but is unconscious lands stiff
    a = I.landing(s, "feet", 2.0, t_air_s=0.64, aware=True, **ADULT_KW)
    b = I.landing(s, "feet", 2.0, t_air_s=0.64, aware=False, **ADULT_KW)
    assert a["prep"] == 1.0 and b["prep"] == 0.0 and b["F_peak"] > a["F_peak"]
    # a short hop (airborne 0.12 s) leaves the legs mostly unready
    c = I.landing(s, "feet", 0.07, t_air_s=0.12, **ADULT_KW)
    assert 0.0 < c["prep"] < 0.2


def test_mass_doubles_energy_doubles_and_force_rises():
    a = A.make_sheet(mass_kg=40.0, height_m=1.6, draws=False)
    b = A.make_sheet(mass_kg=80.0, height_m=1.6, draws=False)
    ra = I.landing(a, "feet", 3.0, prep=0.0, legs_N=1500.0, crouch_m=0.25)
    rb = I.landing(b, "feet", 3.0, prep=0.0, legs_N=1500.0, crouch_m=0.25)
    assert rb["E_contact_J"] == pytest.approx(2 * ra["E_contact_J"], rel=1e-12)
    assert rb["F_peak"] > ra["F_peak"]


def test_rolling_shares_lower_the_peak_of_each_landing():
    s = adult()
    for h in (6.0, 10.0):
        one = I.landing(s, "feet", h, **ADULT_KW)
        roll = I.landing(s, "feet", h, shares=3, **ADULT_KW)
        assert roll["F_peak"] < one["F_peak"]
        assert max(st["F_peak"] for st in roll["stages"]) == roll["F_peak"]
        assert [st["mode"] for st in roll["stages"]] == ["feet", "side", "back"]
        # every joule has a sink in a roll too
        w = roll["work"]
        assert w["total_J"] == pytest.approx(w["W_h_X_J"], rel=1e-9)


def test_chain_capacity_equals_bonestate_break_load_exactly():
    s = adult()
    r = I.landing(s, "feet", 12.0, prep=1.0, **ADULT_KW)
    by = {f["name"]: f for f in r["failed"]}
    rec = B.new_bone()
    # calcaneus: two bones in parallel, referred by 0.99, dynamic factor 1.25
    expect = 2 * B.break_force(rec, "left shin", s, dyn=A.DYN, member="calcaneus") / 0.99
    assert by["calcaneus"]["C"] == expect
    expect = 2 * B.break_force(rec, "left shin", s, dyn=A.DYN, member="tibia") / 0.96
    assert by["tibia"]["C"] == expect
    expect = 2 * B.break_force(rec, "torso", s, dyn=A.DYN, member="pelvis") / 0.68
    assert by["pelvis"]["C"] == expect
    expect = B.break_force(rec, "torso", s, dyn=A.DYN, member="lumbar") / 0.62
    assert by["lumbar"]["C"] == expect
    # and bonestate breaks the bone exactly at that load (the load on the bone is C x m_above / count)
    load = by["lumbar"]["C"] * 0.62
    assert B.check(rec, "torso", s, load * (1 + 1e-12), dyn=A.DYN, member="lumbar")
    assert not B.check(rec, "torso", s, load * (1 - 1e-12), dyn=A.DYN, member="lumbar")
    # a bone already weakened lowers the chain's capacity through the same function
    weak = B.new_bone()
    B.break_(weak, "left shin", s, 9000.0, 0.5, member="tibia")
    r2 = I.landing(s, "feet", 12.0, prep=1.0, recs={"left shin": weak}, **ADULT_KW)
    tib = {f["name"]: f for f in r2["failed"]}["tibia"]
    both = (B.break_force(weak, "left shin", s, dyn=A.DYN, member="tibia")
            + B.break_force(B.new_bone(), "right shin", s, dyn=A.DYN, member="tibia"))
    assert tib["C"] == both / 0.96
    assert tib["C"] < by["tibia"]["C"]


# ── the worked tables (u): prepared 70 kg adult and the stock figure ────────

def test_u_prepared_adult_both_feet_table():
    s = adult()
    assert names(fall(s, 1.0, ADULT_KW)) == [] and names(fall(s, 2.0, ADULT_KW)) == []
    assert names(fall(s, 3.0, ADULT_KW)) == []                  # 1-3 m: no fracture
    r4 = fall(s, 4.0, ADULT_KW)
    assert names(r4) == ["lumbar"] and 0.2 < r4["failed"][0]["frac"] < 0.7
    assert r4["failed"][0]["frac"] == pytest.approx(0.37, abs=0.03)
    assert names(fall(s, 5.0, ADULT_KW)) == ["lumbar", "calcaneus"]
    assert "pelvis" in names(fall(s, 8.0, ADULT_KW)) and "tibia" not in names(fall(s, 8.0, ADULT_KW))
    assert "tibia" in names(fall(s, 10.0, ADULT_KW))
    # the peak forces of the proto in body weights
    assert fall(s, 2.0, ADULT_KW)["a_body_g"] == pytest.approx(12.1, abs=0.2)
    assert fall(s, 1.0, ADULT_KW)["a_body_g"] == pytest.approx(4.5, abs=0.2)


def test_u_prepared_adult_one_foot_breaks_a_calcaneus_at_3m():
    s = adult()
    kw = dict(share_feet=(1, 0), **ADULT_KW)
    assert names(I.landing(s, "feet", 2.0, prep=1.0, **kw)) == []
    r3 = I.landing(s, "feet", 3.0, prep=1.0, **kw)
    assert names(r3) == ["calcaneus"] and r3["failed"][0]["bones"] == ("left shin",)
    r4 = I.landing(s, "feet", 4.0, prep=1.0, **kw)
    assert names(r4) == ["calcaneus", "tibia"] and 0.0 < r4["failed"][1]["frac"] < 1.0
    # the chain used half the stiffness
    both = I.landing(s, "feet", 3.0, prep=1.0, **ADULT_KW)
    assert r3["k_el"] == pytest.approx(0.5 * both["k_el"])


def test_u_unprepared_adult_table():
    s = adult()
    r2 = I.landing(s, "feet", 2.0, prep=0.0, **ADULT_KW)
    assert names(r2) == [] and r2["a_body_g"] == pytest.approx(20.6, abs=0.4)
    assert names(I.landing(s, "feet", 3.0, prep=0.0, **ADULT_KW)) == ["lumbar"]
    assert names(I.landing(s, "feet", 4.0, prep=0.0, **ADULT_KW)) == ["lumbar", "calcaneus"]
    assert "tibia" in names(I.landing(s, "feet", 8.0, prep=0.0, **ADULT_KW))


def test_u_stock_figure_table():
    s = stock()
    for h in (1, 2, 3, 4):
        assert names(fall(s, float(h), STOCK_KW)) == [], h
    r5 = fall(s, 5.0, STOCK_KW)
    assert names(r5) == ["lumbar"] and r5["failed"][0]["frac"] == pytest.approx(0.53, abs=0.03)
    assert names(fall(s, 6.0, STOCK_KW)) == ["lumbar", "calcaneus"]
    assert "pelvis" in names(fall(s, 10.0, STOCK_KW))
    assert "tibia" in names(fall(s, 12.0, STOCK_KW))
    one = dict(share_feet=(1, 0), **STOCK_KW)
    assert names(I.landing(s, "feet", 4.0, prep=1.0, **one)) == ["calcaneus"]
    assert names(I.landing(s, "feet", 6.0, prep=1.0, **one)) == ["calcaneus", "tibia"]
    assert names(I.landing(s, "feet", 3.0, prep=0.0, **STOCK_KW)) == []
    assert names(I.landing(s, "feet", 4.0, prep=0.0, **STOCK_KW)) == ["lumbar"]
    assert names(I.landing(s, "feet", 5.0, prep=0.0, **STOCK_KW)) == ["lumbar", "calcaneus"]


# ── the random-orientation ensemble ─────────────────────────────────────────

def test_echo_random_orientation_ensemble_matches_the_realism_map_targets():
    s = adult()
    fit = I.fit_numbers(s)
    print(I.fit_report(s))                                       # the lane prints the fitted shares
    assert fit["mean_survivable_m"] == pytest.approx(5.2, abs=0.3)
    assert fit["ld50_m"] is not None and 12.0 <= fit["ld50_m"] <= 15.0
    assert math.fsum(e[0] for e in I.ENSEMBLE) == pytest.approx(1.0, abs=1e-12)


def test_the_ensemble_survival_curve_falls_with_height():
    hs, S = I.survival_curve(adult())
    assert S[0] == pytest.approx(1.0, abs=1e-12) and S[-1] < 0.5
    assert all(S[i + 1] <= S[i] + 1e-12 for i in range(len(S) - 1))


def test_codeonly_feet_first_is_the_best_orientation_and_head_first_kills_first():
    s = adult()
    def dies_at(mode, aware, one):
        for h in I.FALL_GRID:
            if I.outcome(I._case(s, mode, aware, one, h), s)["dead"]:
                return h
        return 99.0
    feet = dies_at("feet", True, False)
    side_limp, head = dies_at("side", False, False), dies_at("head", False, False)
    assert feet > side_limp >= head
    assert feet > dies_at("back", True, False) > head


def test_a_head_landing_reads_acceleration_and_hic():
    s = adult()
    last = 0.0
    for h in (0.5, 1, 2, 3, 4, 6, 8):
        r = I.landing(s, "head", float(h), aware=False)
        hd = r["head"]
        assert hd["hic"] >= last
        last = hd["hic"]
        assert hd["a_peak_g"] == pytest.approx(r["F_peak"] / (hd["m_eff"] * G))
    assert I.landing(s, "head", 2.0, aware=False)["head"]["hic"] < I.HIC_SKULL
    assert I.landing(s, "head", 8.0, aware=False)["head"]["dead"]
    assert I.hic(10.0, 1.0) == pytest.approx(0.015 * 10 ** 2.5)             # T is capped at 15 ms
    assert I.hic(10.0, 0.005) == pytest.approx(0.005 * 10 ** 2.5)


def test_knock_out_threshold_scales_with_ko_scale():
    base = adult()
    tough = dict(base, ko_scale=1.15)
    weak = dict(base, ko_scale=0.85)
    # find a height where the weak one is out and the tough one is not
    got = False
    for h in [0.1 * i for i in range(2, 40)]:
        a = I.landing(weak, "head", h, aware=False)["head"]
        b = I.landing(tough, "head", h, aware=False)["head"]
        if a["ko"] and not b["ko"]:
            got = True
            break
    assert got
    h1 = I.landing(weak, "head", 3.0, aware=False)["head"]
    h2 = I.landing(tough, "head", 3.0, aware=False)["head"]
    assert h1["t_ko_s"] > h2["t_ko_s"] > 0
    assert h1["t_ko_s"] <= I.T_KO_MAX_S
    assert I.landing(adult(), "head", 0.3, aware=False)["head"]["ko"] is False


def test_contact_energy_uses_effective_mass():
    assert I.contact_energy(70.0, 4.0) == pytest.approx(0.5 * 70 * 16)
    e = I.contact_energy(70.0, 4.0, r_cross_n=0.5, I_kgm2=10.0)
    m_eff = 1.0 / (1.0 / 70.0 + 0.25 / 10.0)
    assert e == pytest.approx(0.5 * m_eff * 16.0) and e < 0.5 * 70 * 16
    s = adult()
    r = I.landing(s, "side", 0.0, E_contact_J=500.0, energy=False)
    assert r["E_contact_J"] == pytest.approx(500.0, rel=1e-12)
    assert r["residual_J"] == 0.0 or r["residual_J"] >= 0.0


def test_a_side_landing_takes_a_share_and_leaves_the_rest_as_motion():
    s = adult()
    aware = I.landing(s, "side", 6.0, aware=True)
    limp = I.landing(s, "side", 6.0, aware=False)
    W = 70.0 * G
    assert aware["E_contact_J"] == pytest.approx(0.15 * W * 6.0, rel=1e-12)
    assert limp["E_contact_J"] == pytest.approx(0.35 * W * 6.0, rel=1e-12)
    assert aware["residual_J"] == pytest.approx(0.85 * W * 6.0, rel=1e-12)
    assert limp["F_peak"] > aware["F_peak"]
    # side elements are on the hit side only
    r = I.landing(s, "side", 12.0, side="right")
    for f in r["failed"]:
        assert all(b in ("right thigh", "right upper arm", "torso") for b in f["bones"])


def test_u_a_side_fall_breaks_ribs_and_hip_where_a_prepared_feet_first_fall_breaks_nothing():
    s = adult()
    assert names(fall(s, 2.0, ADULT_KW)) == []
    # (wave 14a H2: the side landing is softer, k 31 kN/m; a healthy adult keeps
    # his ribs from 1 m and breaks them from 2 m, the hip from 3 m)
    assert names(I.landing(s, "side", 1.0)) == []
    assert "ribs lateral" in names(I.landing(s, "side", 2.0))
    side = names(I.landing(s, "side", 3.0))
    assert "ribs lateral" in side and "femoral neck" in side
    assert names(I.landing(s, "side", 0.3)) == []                 # a stumble leaves no fracture
    assert "femoral neck" in names(I.landing(s, "side", 4.0))
    assert "pelvis lateral" in names(I.landing(s, "side", 8.0, aware=False))


def test_the_fall_grab_mode_arms_out_has_its_own_control_element():
    s = adult()
    soft = I.landing(s, "arms", 1.0, prep=0.0)
    ready = I.landing(s, "arms", 1.0, prep=1.0)
    assert ready["F_peak"] <= soft["F_peak"] + 1e-9
    r = I.landing(s, "arms", 4.0)
    assert "distal radius" in names(r) or "clavicle" in names(r)
    lhs, rhs = _balance(r)
    assert lhs == pytest.approx(rhs, rel=1e-9)


def test_outcome_kinds():
    s = adult()
    ok = I.outcome(fall(s, 2.0, ADULT_KW), s)
    assert not ok["dead"] and not ok["ko"] and ok["hemorrhage_ml"] == 0.0
    dead = I.outcome(I.landing(s, "head", 8.0, aware=False), s)
    assert dead["dead"] and dead["why"] == "brain" and dead["ko"]
    hurt = I.outcome(fall(s, 10.0, ADULT_KW), s)
    assert hurt["hemorrhage_ml"] > 0 and hurt["bleed_frac"] > 0


# ── blows ───────────────────────────────────────────────────────────────────

CLUB_K = I.BLOW_K[("club", "thigh")]


def test_u_club_on_thigh_gives_2_9_kN_at_60_J_and_does_not_break():
    s = adult()
    r = I.blow_on_bone(s, "left thigh", 60.0, 0.05 ** 2, 0.0, CLUB_K, 3.0)
    assert r["E_c"] == pytest.approx(60.0 * 7.0 / 10.0, rel=1e-12)
    assert r["F_c"] == pytest.approx(2.9e3, rel=0.01)
    assert r["F_c"] == pytest.approx(math.sqrt(2 * CLUB_K * r["E_c"]), rel=1e-12)
    assert not r["broke"] and r["integrity"] == 1.0
    assert r["F_bend"] == pytest.approx(3.2e3, rel=0.02)           # 4 x 364 / 0.45


def test_u_the_thigh_breaks_near_75_J_and_never_at_40_J():
    s = adult()
    for J in (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0):
        assert not I.blow_on_bone(s, "left thigh", J, 0.0025, 0.0, CLUB_K, 3.0)["broke"], J
    first = None
    for J in [60.0 + 0.5 * i for i in range(0, 80)]:
        if I.blow_on_bone(s, "left thigh", J, 0.0025, 0.0, CLUB_K, 3.0)["broke"]:
            first = J
            break
    assert first is not None and 70.0 <= first <= 80.0, first
    assert first == pytest.approx(75.0, abs=2.5)


def test_a_harder_blow_crushes_the_broken_bone_more_and_the_record_follows():
    s = adult()
    rec = B.new_bone()
    r1 = I.blow_on_bone(s, "left thigh", 90.0, 0.0025, 0.0, CLUB_K, 3.0)
    r2 = I.blow_on_bone(s, "left thigh", 150.0, 0.0025, 0.0, CLUB_K, 3.0)
    assert r1["broke"] and r2["broke"]
    assert 0.0 < r1["frac"] < r2["frac"] <= 1.0
    assert r2["rec"]["integrity"] < r1["rec"]["integrity"] < 1.0
    assert r2["rec"]["gap_m"] > 0.5e-3
    # every joule has a sink: E_c = elastic + bone crush
    assert r2["absorbed_J"] == pytest.approx(r2["E_c"], rel=1e-9)
    assert rec["integrity"] == 1.0


def test_an_edge_at_bone_depth_lowers_integrity_by_the_cut_fraction_and_F_bend():
    s = adult()
    blunt = I.blow_on_bone(s, "left thigh", 20.0, 0.0025, 0.0, CLUB_K, 3.0)
    edged = I.blow_on_bone(s, "left thigh", 20.0, 2e-4, 0.4, CLUB_K, 3.0)
    assert edged["integrity"] == pytest.approx(0.6)
    assert edged["F_bend"] == pytest.approx(0.6 * blunt["F_bend"], rel=1e-12)
    assert edged["rec"]["integrity"] == pytest.approx(0.6)
    assert edged["pressure_Pa"] > 10 * blunt["pressure_Pa"]      # small area: a cut; a face: a bruise
    # a weakened bone breaks at a blow the intact one shrugs off
    assert not blunt["broke"] and I.blow_on_bone(s, "left thigh", 40.0, 2e-4, 0.4, CLUB_K, 3.0)["broke"]


def test_blow_on_other_bones_uses_the_same_rule():
    s = adult()
    for bone in ("left shin", "left upper arm", "left forearm", "head"):
        lo = I.blow_on_bone(s, bone, 1.0, 0.0025, 0.0, 50e3, 1.0)
        hi = I.blow_on_bone(s, bone, 1e4, 0.0025, 0.0, 50e3, 1.0)
        assert not lo["broke"] and hi["broke"], bone


def test_a_landing_with_no_foot_on_the_ground_lands_on_the_side_not_a_crash():
    s = adult()
    r = I.landing(s, "feet", 2.0, share_feet=(0, 0))
    assert r["mode"] == "side" and r["F_peak"] > 0.0


def test_a_roll_reports_the_region_first_touched_not_the_one_asked_for():
    # Cause: the roll result copied the REQUESTED mode; a feet landing with no
    # foot down starts on the side, so "feet" was wrong. Random people, shares,
    # heights, and every first mode: the top-level mode is stage 0's mode.
    import random
    rng = random.Random(1311)
    for _ in range(80):
        mass = rng.uniform(40.0, 110.0)
        s = A.make_sheet(mass_kg=mass, height_m=rng.uniform(1.5, 1.95), pid=rng.randrange(10**6),
                         draws=True)
        mode = rng.choice(["feet", "side", "back", "shoulder", "arms"])
        feet = rng.choice([(1, 1), (1, 0), (0, 1), (0, 0)])
        shares = rng.choice([1, 2, 3])
        r = I.landing(s, mode, rng.uniform(1.0, 8.0), share_feet=feet, shares=shares)
        assert r["mode"] == r["stages"][0]["mode"]
        if mode in ("feet", "arms") and feet == (0, 0):
            assert r["mode"] == "side"
        elif feet != (0, 0) or mode not in ("feet", "arms"):
            assert r["mode"] == mode
    r = I.landing(adult(), "feet", 6.0, share_feet=(0, 0), shares=3)
    assert r["mode"] == "side"
    r = I.landing(adult(), "side", 6.0, shares=3)
    assert r["mode"] == "side"
