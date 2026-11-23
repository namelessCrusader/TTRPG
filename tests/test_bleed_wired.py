"""BLEEDING BY VESSEL, WIRED (wave 15, S2 A; owner W15-Q1 on by default).

vasculature.py is the pure law (tests/test_vasculature.py holds it). These tests hold
the WIRING: the wound's place on the bone, the cuts it makes, the one outside pressure
the acts write (press, bind, tourniquet), the ledger, the gate, and the limb law that
reads the ischaemia. FAST tier: most call `_law_bleed` directly on a standing body; a
few step the world.

Each test says the mutation it must catch.
"""
import copy
import math

import numpy as np
import pytest

from src.voxel import anatomy as AN, bleeding as BL, firstaid as FA, vasculature as VA
from src.voxel.materials import BLOOD, TICK_S
from src.voxel.scenes import _person
from src.voxel.sim import STONE, World

AXE = 2e-4


def _body(x=12):
    w = World(30, 20, 40, voxel_cm=5)
    w.fill(0, 30, 0, 20, 0, 1, STONE)
    p = _person(w, x, 10)
    return w, p


def _cell(w, p, bone, pick="anterior"):
    """A cell on the outside of a bone: the front (anterior), the back, or the side nearest the middle."""
    fr = w._bone_frame(p, bone)
    c = w._bone_parts(p, [bone])[0]
    d = {"anterior": fr["f"], "posterior": -fr["f"], "medial": fr["e2"]}[pick]
    return tuple(int(v) for v in c[np.argmax((c - fr["o"]) @ d)])


def _wound(w, p, bone, az, along, depth, length, only=None):
    """A wound as `_blow` makes one, with its place given: the cuts it crosses. `only`
    keeps the named vessels (to isolate one)."""
    w._blood0(p)
    wd = {"bone": bone, "tick": w.tick, "joules": 100.0, "area_m2": AXE, "depth_m": depth,
          "length_m": length, "armour": None, "open": True, "through": False,
          "az": az, "along": along}
    sheet = {"mass_kg": BL.mass_kg(p)}
    reg = VA.region_of_bone(bone)
    D = VA.limb_thickness_m(reg, sheet)
    cuts = VA.crosses(wd, reg, sheet)
    for c in cuts:
        v = AN.VESSEL_OF[c["vessel"]]
        c["bone"], c["limb"] = bone, VA.limb_of_bone(bone)
        c["depth_m"] = v["depth_mm"] / AN.D_ADULT_MM[bone.split(" ", 1)[1]] * D
        c["along"] = VA.limb_along(bone, c["along"])
    wd["cuts"] = [c for c in cuts if only is None or c["vessel"] in only]
    p.setdefault("wounds", []).append(wd)
    return wd


def _run(w, p, seconds, spill=True):
    """Run only the bleeding law for `seconds`; blood leaves whole (the spill stubbed) so the
    flow law is read alone."""
    if spill:
        w._spill_from = lambda p_, bone, fluid, ml: ml
    t = 0.0
    while t < seconds and p["alive"]:
        w.tick += 1
        w._law_bleed()
        t += TICK_S
    return t


def _lost_frac(p):
    return 1.0 - p["blood_ml"] / p["blood0_ml"]


# -- the place of a wound, in the bone's own frame ------------------------------
def test_every_bone_of_every_layout_resolves_to_a_region_or_none():
    from src.voxel import bodysheet as BS
    w, p = _body()
    names = set(p["segs"]) | {"head", "torso"}
    for n in names:
        r = VA.region_of_bone(n)
        assert r is None or (r.split(" ", 1)[-1] if r[:4] in ("left", "righ") else r) in AN.D_ADULT_MM, n
    # the sheet's own segment names resolve too (the hand and the foot ride in other bones)
    for part in list(BS.WINTER) + ["shank", "trunk"]:
        for pre in ("", "left ", "right "):
            VA.region_of_bone(pre + part)          # must not raise
    assert VA.region_of_bone("left hand") is None and VA.region_of_bone("right foot") is None


def test_azimuth_is_read_in_the_bones_own_frame_and_survives_a_swing():
    """The same point of a limb keeps its azimuth when the limb is swung or the body turned.
    MUTATION: reading the azimuth against the WORLD's axes (or the chest, not carried through
    the swing) fails the forearm raised forward and the quarter turn."""
    w, p = _body()
    fr = w._bone_frame(p, "right forearm")
    assert np.allclose(fr["f"], (0, 1, 0)) and np.allclose(fr["e2"], (-1, 0, 0))   # front +y, medial -x
    front = _cell(w, p, "right forearm", "anterior")
    back = _cell(w, p, "right forearm", "posterior")
    az_f, _ = w._frame_az_along(fr, front)
    az_b, _ = w._frame_az_along(fr, back)
    assert min(az_f, 360 - az_f) < 30 and abs(az_b - 180) < 30, (az_f, az_b)
    # raise the arm forward: its axis turns from down to forward, and its FRONT now faces up
    assert w._repose(p, "right arm", [1.4, 0.0], toward=(0, 1)) == "moved"
    fr2 = w._bone_frame(p, "right forearm")
    assert np.allclose(fr2["axis"], (0, 1, 0), atol=0.15), fr2["axis"]
    assert np.allclose(fr2["f"], (0, 0, 1), atol=0.15), "the anterior is carried through the swing"
    assert np.allclose(fr2["e2"], (-1, 0, 0), atol=0.15), "and medial is still toward the middle"
    # and the whole body turned a quarter: every cell keeps its azimuth
    w2, p2 = _body()
    cells = lambda: np.asarray(p2["segs"]["left thigh"]) + np.asarray(
        w2._claim_cells(p2, *w2._person_cells(p2))).min(axis=0)
    a0 = [w2._frame_az_along(w2._bone_frame(p2, "left thigh"), c)[0] for c in cells()]
    assert w2._turn(p2, 1)
    a1 = [w2._frame_az_along(w2._bone_frame(p2, "left thigh"), c)[0] for c in cells()]
    assert np.allclose(a0, a1)


def test_medial_is_toward_the_middle_on_both_sides():
    """MUTATION: one sign for both sides puts the right limb's vessels on its outside."""
    w, p = _body()
    for bone in ("left thigh", "right thigh", "left shin", "right shin"):
        fr = w._bone_frame(p, bone)
        toward = fr["trunk"] - fr["o"]
        assert float(toward @ fr["e2"]) > 0.0, bone


def test_a_wound_on_a_new_blow_carries_its_place_and_its_cuts():
    w, p = _body()
    c = _cell(w, p, "left thigh", "medial")
    wd = w._blow(p, c, 5000.0, AXE)
    assert wd["open"] and 0.0 <= wd["along"] <= 1.0 and 0.0 <= wd["az"] < 360.0
    assert "cuts" in wd or wd["depth_m"] > 0.0


def test_a_wound_on_the_hand_or_the_belly_makes_no_cuts():
    w, p = _body()
    wd = w._blow(p, _cell(w, p, "right forearm", "anterior"), 5000.0, AXE)
    assert "az" in wd                              # a forearm wound has its place
    w2, p2 = _body()
    belly = tuple(int(v) for v in w2._bone_parts(p2, ["torso"])[0][0])      # the lowest cell
    wb = w2._blow(p2, belly, 5000.0, AXE)
    assert not wb.get("cuts"), "a belly wound reaches no named vessel (organs: impact only)"


# -- the loop -------------------------------------------------------------------
def test_T1_an_unattended_femoral_artery_takes_the_faint_line_in_one_to_five_minutes():
    """MUTATION: dropping the vessel term (only the capillary ooze) takes far longer than 5 minutes."""
    w, p = _body()
    wd = _wound(w, p, "left thigh", 60.0, 0.5, 0.06, 0.12, only=("femoral artery",))
    assert wd["cuts"] and wd["cuts"][0]["f_open"] > 0.9
    t = 0.0
    while _lost_frac(p) < VA.faint_loss_frac() and t < 600.0:
        t += _run(w, p, 1.0)
    assert 60.0 <= t <= 300.0, t


def test_a_cut_with_no_vessel_bleeds_bit_for_bit_as_the_old_law():
    """T7. MUTATION: adding the vessel flow to every wound changes the blood at once."""
    a, pa = _body()
    wa = {"bone": "left thigh", "tick": 0, "joules": 9.0, "area_m2": AXE, "depth_m": 0.004,
          "length_m": 0.08, "armour": None, "open": True, "through": False}
    wa2 = dict(wa)
    a._blood0(pa)
    pa.setdefault("wounds", []).append(wa)
    b, pb = _body()
    b._blood0(pb)
    pb.setdefault("wounds", []).append(wa2)
    b.vessel_bleed = False                         # the old law only
    for w_, p_ in ((a, pa), (b, pb)):
        w_._spill_from = lambda p, bone, fluid, ml: ml
        for _ in range(400):
            w_.tick += 1
            w_._law_bleed()
    assert pa["blood_ml"] == pb["blood_ml"] and wa["open"] == wa2["open"]


def test_the_gate_off_makes_no_cuts_and_keeps_no_geometry():
    """T9. MUTATION: building the cuts whatever the gate says."""
    w, p = _body()
    w.vessel_bleed = False
    wd = w._blow(p, _cell(w, p, "left thigh", "medial"), 5000.0, AXE)
    assert "cuts" not in wd and "az" not in wd and "along" not in wd


def test_T3_a_press_stops_the_bleed_and_lifting_the_hand_reopens_it_and_the_clot_restarts():
    """MUTATION: never clearing seal_s on a reopen fails the restart."""
    w, p = _body()
    wd = _wound(w, p, "left forearm", 45.0, 0.5, 0.05, 0.1, only=("radial artery", "ulnar artery"))
    assert wd["cuts"]
    FA.press(wd, 100.0, 4e-4, 400.0)
    _run(w, p, 20.0)
    assert all(c["sealed"] for c in wd["cuts"]), "100 N on 4 cm2 stops a radial bleed"
    lost = p["blood0_ml"] - p["blood_ml"]
    FA.release_press(wd)
    _run(w, p, 0.05)
    assert any(not c["sealed"] for c in wd["cuts"]), "the hand is off and the young plug cannot hold"
    assert all(c["seal_s"] < 1.0 for c in wd["cuts"] if not c["sealed"]), "the clot restarts from nothing"
    _run(w, p, 5.0)
    assert p["blood0_ml"] - p["blood_ml"] > lost


def test_a_pressing_hand_tires_and_lets_go_by_Rohmert():
    """At f = 0.5 of the hand's maximum a hold lasts about 86 s. MUTATION: no fatigue step."""
    w, p = _body()
    wd = _wound(w, p, "left forearm", 45.0, 0.5, 0.05, 0.1, only=("radial artery",))
    FA.press(wd, 200.0, 4e-4, 400.0)               # f = 0.5
    t = 0.0
    while wd["press"]["F_eff"] > 0.0 and t < 400.0:
        FA.press_step(wd, 1.0)
        t += 1.0
    assert 60.0 <= t <= 120.0, t


def test_T4_a_tourniquet_below_AOP_does_not_stop_a_thigh_artery_and_one_above_does():
    """MUTATION: removing the AOP test lets a slack strap stop an artery."""
    sbp = FA.SBP_PA
    r, width = 0.09, 0.045                          # a 58 cm thigh, a 4.5 cm strap
    need = VA.aop_pa(sbp, 2.0 * math.pi * r, width)
    for T, stops in ((0.6 * need * r * width, False), (1.5 * need * r * width, True)):
        w, p = _body()
        wd = _wound(w, p, "left thigh", 60.0, 0.5, 0.06, 0.12, only=("femoral artery",))
        wd["cuts"][0]["along"] = VA.limb_along("left thigh", 0.7)
        FA.tourniquet_on(p, "left thigh", 0.3, T, r, width, 1e9)
        _run(w, p, 30.0)
        if stops:
            assert wd["cuts"][0]["sealed"], "above AOP the cuff shuts the artery below it"
        else:
            assert not wd["cuts"][0]["sealed"] and p["blood_ml"] < 0.9 * p["blood0_ml"], \
                "below AOP the artery stays open and bleeds on"


def test_a_wider_strap_needs_less_pressure_and_saturates():
    sbp = FA.SBP_PA
    circ = 0.58
    a = [VA.aop_pa(sbp, circ, wd) for wd in (0.03, 0.05, 0.08, 0.13, 0.18)]
    assert a == sorted(a, reverse=True)
    assert a[3] - a[4] < 0.15 * sbp < a[0] - a[1] + 1.0e5


def test_a_cut_above_the_cuff_keeps_bleeding_and_a_cut_on_another_limb_is_not_helped():
    """Distal-only. MUTATION: ignoring the limb makes a thigh cuff close an arm."""
    w, p = _body()
    near = _wound(w, p, "left thigh", 60.0, 0.1, 0.06, 0.12, only=("femoral artery",))
    far = _wound(w, p, "left shin", 160.0, 0.5, 0.05, 0.1, only=("posterior tibial",))
    arm = _wound(w, p, "left upper arm", 100.0, 0.5, 0.05, 0.1, only=("brachial artery",))
    r, width = 0.09, 0.045
    need = VA.aop_pa(FA.SBP_PA, 2.0 * math.pi * r, width)
    FA.tourniquet_on(p, "left thigh", 0.5, 2.0 * need * r * width, r, width, 1e9)
    _run(w, p, 40.0)
    assert not near["cuts"][0]["sealed"], "a cut above (proximal to) the cuff still bleeds"
    assert far["cuts"][0]["sealed"], "a cut below it is shut"
    assert not arm["cuts"][0]["sealed"], "another limb is not helped"


def test_a_strap_pulled_past_its_limit_tears():
    w, p = _body()
    r, width = 0.09, 0.045
    tq = FA.tourniquet_on(p, "left thigh", 0.3, 400.0, r, width, 300.0)
    assert not FA.strap_check(tq)
    wd = _wound(w, p, "left thigh", 60.0, 0.5, 0.06, 0.12, only=("femoral artery",))
    _run(w, p, 1.0)
    assert not p["tourniquets"], "the strap tore and the limb is perfused again"


def test_T5_a_forearm_raised_above_the_heart_bleeds_less_by_the_hydrostatic_term():
    """MUTATION: dropping hydro makes the two equal. (The wound's height is a field of the wound.)"""
    out = []
    for z in (0.0, -0.5):                           # level with the heart; 0.5 m ABOVE it
        w, p = _body()
        wd = _wound(w, p, "left forearm", 0.0, 0.5, 0.04, 0.1, only=("cephalic vein",))
        wd["z_rel_m"] = z
        w.tick = 1                                   # (the height is re-read every Z_EVERY ticks: not now)
        w._spill_from = lambda p_, bone, fluid, ml: ml
        for _ in range(8):
            BL_ = BL.step_wound_cuts
            ml = BL_(w, p, wd, p["blood_ml"], TICK_S)
            wd["z_rel_m"] = z
        out.append(wd["cuts"][0]["flow"])
    assert out[1] < out[0]


def test_T6_blood_in_the_body_and_on_the_lattice_is_one_quantity_with_several_wounds():
    """MUTATION: subtracting the poured blood twice breaks the sum."""
    w, p = _body()
    w._blood0(p)
    for bone, az, vs in (("left thigh", 70.0, ("femoral artery",)), ("left forearm", 45.0, ("radial artery",))):
        _wound(w, p, bone, az, 0.5, 0.05, 0.1, only=vs)
    start = w.total_fluid(BLOOD)               # the ledger counts the body and the lattice together
    for _ in range(600):                           # 15 s of the real world: the lattice moves the blood
        w.step()
    owed = sum(x.get("_owe", 0.0) for x in p["wounds"])
    now = w.total_fluid(BLOOD)
    assert now == pytest.approx(start, abs=0.003 * start), (start, now)
    shed = sum(x.get("shed_ml", 0.0) for x in p["wounds"])
    assert shed > 100.0 and p["blood_ml"] == pytest.approx(p["blood0_ml"] - shed, abs=1e-6)


def test_T10_the_same_world_gives_the_same_blood_and_a_person_list_order_changes_nothing():
    runs = []
    for _ in range(2):
        w, p = _body()
        _wound(w, p, "left thigh", 70.0, 0.5, 0.05, 0.1, only=("femoral artery",))
        _run(w, p, 5.0)
        runs.append(p["blood_ml"])
    assert runs[0] == runs[1]


# -- tourniquet ischaemia reaches the limb law (W15-Q2) ---------------------------
def test_ischaemia_under_a_closed_cuff_weakens_the_limb_through_the_limb_law():
    """MUTATION: ignoring p["isch"] in the limb law leaves the dead forearm as strong."""
    w, p = _body()
    r, width = 0.04, 0.045
    need = VA.aop_pa(FA.SBP_PA, 2.0 * math.pi * r, width)
    FA.tourniquet_on(p, "right upper arm", 0.9, 2.0 * need * r * width, r, width, 1e9)
    before = w._hand_strength(p, "right arm")
    # three hours under the cuff: one step of 3 h (the law is a closed form of time)
    from src.voxel import firstaid
    firstaid.isch_step(p, p["tourniquets"], 3 * 3600.0, lambda limb: [b for b in p["segs"] if VA.limb_of_bone(b) == limb])
    dead = p["isch"]["right forearm"]["dead"]
    assert dead == pytest.approx(VA.isch_dead_frac(3 * 3600.0), rel=1e-9) and 0.1 < dead < 0.6
    after = w._hand_strength(p, "right arm")
    assert after == pytest.approx(before * (1.0 - dead)) and after < before
    assert "right upper arm" not in p["isch"] or p["isch"]["right upper arm"]["dead"] <= dead * 0.2, \
        "only what is below the cuff dies"
    assert p["alive"], "no amputation: nothing is lost"


def test_a_dead_leg_bears_less_weight_and_a_body_with_no_cuff_is_untouched():
    w, p = _body()
    base = w._legs_force(p) if hasattr(w, "_legs_force") else None
    bones = ["left shin", "left thigh", "right shin", "right thigh"]
    assert w._isch_dead(p, bones) == 0.0
    p["isch"] = {"left shin": {"t_s": 0, "share": 1.0, "dead": 0.5}}
    assert w._isch_dead(p, bones) == pytest.approx(0.125)


def test_a_slack_tourniquet_does_not_kill_the_limb():
    """A cuff below AOP leaves the artery open: no ischaemia. MUTATION: accruing under any cuff."""
    w, p = _body()
    r, width = 0.04, 0.045
    FA.tourniquet_on(p, "right upper arm", 0.9, 5.0, r, width, 1e9)
    FA.isch_step(p, p["tourniquets"], 3600.0, lambda limb: [b for b in p["segs"] if VA.limb_of_bone(b) == limb])
    assert not p.get("isch")


# -- the lattice's fallen bodies ---------------------------------------------------
def test_a_body_that_has_fallen_still_pours_its_blood():
    """_spill_from finds no bones on a body that lay down (the segments are the standing ones);
    it falls back to the flesh. MUTATION: no fallback leaves the blood owed for ever."""
    w, p = _body()
    w._blood0(p)
    p["segs"] = {k: v + np.array([50, 50, 50]) for k, v in p["segs"].items()}   # stale: finds no flesh
    got = w._spill_from(p, "left thigh", BLOOD, 20.0)
    assert got > 0.0


def test_a_forearm_cut_through_the_tendons_writes_the_grips_tendon_cut():
    """The grip law (lane 2c) reads p["body"]["tendon_cut"][arm]. MUTATION: not writing it leaves the hand whole."""
    w, p = _body()
    fr = w._bone_frame(p, "right forearm")
    # a cell on the medial-anterior side of the forearm, where the flexors are (az about 45)
    c = w._bone_parts(p, ["right forearm"])[0]
    cell = tuple(int(v) for v in c[np.argmax((c - fr["o"]) @ (fr["f"] + fr["e2"]))])
    got = w._blow(p, cell, 5000.0, AXE)
    tcut = (p.get("body") or {}).get("tendon_cut", {}).get("right arm", 0.0)
    assert got["depth_m"] > 0.05 and tcut > 0.5, (got["depth_m"], tcut)
    assert tcut == pytest.approx(VA.tendon_cut(got, "right forearm", {"mass_kg": BL.mass_kg(p)}), abs=1e-9)
    assert p["body"]["tendon_cut"]["right arm"] == tcut       # where grip.tendon_cut reads it
    w2, p2 = _body()
    w2.vessel_bleed = False
    w2._blow(p2, cell, 5000.0, AXE)
    assert "tendon_cut" not in (p2.get("body") or {}), "the gate off keeps no new record"


def _press_then_lift(vessel, bone, az, press_s):
    w, p = _body()
    wd = _wound(w, p, bone, az, 0.5, 0.06, 0.12, only=(vessel,))
    FA.press(wd, 150.0, 4e-4, 1e9)                 # a firm hand that does not tire (the clot is the question)
    _run(w, p, press_s)
    FA.release_press(wd)
    _run(w, p, 120.0)
    return wd["cuts"][0]["sealed"]


def test_the_clot_constants_reproduce_the_first_aid_rule_press_for_ten_minutes():
    """The clot constants (14 kPa mm, 180 s) are (u); this holds them to first aid, not to a duel:
    a bleed that is let go after a minute or two starts again, one pressed for ten minutes stays shut
    (Red Cross and Stop the Bleed teach firm pressure for at least five to ten minutes, (k)), and a
    femoral artery's plug (8 mm) cannot hold its own pressure whatever the time (that is why it
    needs a tourniquet). MUTATION: a clot that matures in seconds lets the one-minute press hold."""
    assert not _press_then_lift("radial artery", "left forearm", 30.0, 60.0), "lifted at a minute: reopens"
    assert _press_then_lift("radial artery", "left forearm", 30.0, 600.0), "pressed for ten minutes: holds"
    assert not _press_then_lift("femoral artery", "left thigh", 60.0, 600.0), "the femoral plug never holds alone"
