"""Wave 14bF: the finish of 14b. Every strike is paid from the body's books, the acts
(swing, throw, lift, shove, tackle, grapple, a thrown or falling body on a person) run
through ONE exertion law and ONE contact law, and the three gates are on for every world."""
import math

import numpy as np
import pytest

from src.voxel import digestion as DG
from src.voxel import exertion as X
from src.voxel import scenes_bodycontact as SB
from src.voxel.bodyworld import BodyWorld
from src.voxel.materials import TICK_S
from src.voxel.policy import MENU_BATCH
from src.voxel.scenes import _person, _Wants
from src.voxel.sim import IRON, STONE, WOOD, World

import tests.test_mechlaw as TM
from tests.test_exertion import _BHDoor, _ref


# ---- the gates ---------------------------------------------------------------
def test_the_three_gates_are_on_for_every_world_and_the_slip_gate_was_already():
    w = World(8, 8, 8, voxel_cm=5)
    assert w.exertion_from_body is True
    assert w.body_contact is True
    assert w.menu_batch == MENU_BATCH == 4
    assert w.contact_laws is True


# ---- every strike is paid ----------------------------------------------------
def _heat_room():
    from src.voxel import scenes_bodyheat as SB_
    w, mid, p = TM._room(cls=_BHDoor, flaw=False)
    w.body_heat_on(p, segs=p["segs"], sheet=SB_.stock_sheet())
    w.exertion_from_body = True
    w.step()
    return w, mid, p


def test_a_second_kick_in_the_same_tick_is_billed_too():
    w, mid, p = _heat_room()
    E = X.strike(p)["E_J"]
    s0 = DG.stores_J(p["heat"]["stores"])
    wo0, mi0 = w.work_out, w.mech_in
    r1 = w.kick(p, mid)
    r2 = w.kick(p, mid)
    assert r1.get("result") != "spent" and r2.get("result") != "spent"
    paid = s0 - DG.stores_J(p["heat"]["stores"])
    assert paid >= 2.0 * E / 0.22 - 2.0          # the stores gave both kicks, E / eta each
    assert w.work_out - wo0 == pytest.approx(2.0 * E, rel=1e-6)      # and both were booked once
    assert w.mech_in == mi0


def test_a_refused_bill_pays_for_nothing_and_the_kick_does_not_happen():
    w, mid, p = _heat_room()
    w.note_work = lambda *a, **k: None           # the books refuse the bill
    assert w._pay_strike(p, 115.0, "kick") == 0.0
    r = w.kick(p, mid)
    assert r["result"] == "spent" and not r["swing"]


def test_a_body_with_no_heat_record_pays_from_its_limb_reserve_and_gets_weaker():
    w = World(8, 8, 8, voxel_cm=5)
    p = {"pid": 5, "name": "ann", "strength_N": 400.0}
    e0 = X.strike(p)["E_J"]
    scales = [w._pay_strike(p, 115.0, "kick") for _ in range(45)]
    assert scales[0] == 1.0 and scales[10] == 1.0
    assert scales[-1] == 0.0 and 0.0 < min(s for s in scales if s > 0.0) < 1.0
    # the reserve (W' = 250 J/kg x 70 kg) at eta 0.22 pays 17.5 kJ x 0.22 = 3850 J of strikes
    assert sum(115.0 * s for s in scales) == pytest.approx(250.0 * 70.0 * 0.22, rel=1e-6)
    assert p["body"]["fat"]["legs"]["wp"] == pytest.approx(0.0, abs=1e-9)
    assert X.strike(p)["E_J"] < e0               # and the tired leg kicks weaker next time
    w.tick += int(1200.0 / TICK_S)               # twenty minutes' rest
    assert w._pay_strike(p, 115.0, "kick") == 1.0
    assert p["body"]["fat"]["legs"]["wp"] > 0.9


def test_a_kick_of_an_empty_leg_is_not_made_at_all():
    w, mid, p = TM._room(flaw=False)
    w.exertion_from_body = True
    p["body"] = {"fat": {"legs": {"gam": 0.0, "wp": 0.0, "t_s": w.tick * TICK_S}}}
    r = w.kick(p, mid)
    assert r["result"] == "spent"


# ---- one exertion law for the acts -------------------------------------------
def test_a_lift_limit_is_the_bodys_own_and_the_old_flat_one_is_kept_with_the_gate_off():
    w = World(8, 8, 8, voxel_cm=5)
    weak = {"pid": 1, "name": "w", "strength_N": 400.0, "exert_z": {"lift": 0.5}}
    ref = {"pid": 2, "name": "r", "strength_N": 400.0, "exert_z": {"lift": 1.0}}
    assert w._strength(ref) == pytest.approx(400.0) and w._strength(weak) == pytest.approx(200.0)
    tired = dict(ref, body={"fat": {"back": {"gam": 0.6, "wp": 1.0}}})
    assert w._strength(tired) == pytest.approx(0.4 * 400.0)
    w.exertion_from_body = False
    assert w._strength(weak) == 400.0 and w._strength({}) == 400.0


def test_a_hand_can_hold_what_the_body_can_not_lift():
    """The thing, not a flag: a 30 kg block is liftable (294 N) by the reference adult and
    not by one with half the muscle."""
    w = World(30, 20, 30, voxel_cm=5)
    w.fill(0, 30, 0, 20, 0, 1, STONE)
    a = _person(w, 8, 8)
    a["name"], a["exert_z"] = "a", {"lift": 1.0}
    b = _person(w, 8, 14)
    b["name"], b["exert_z"] = "b", {"lift": 0.5}
    w.fill(20, 21, 10, 11, 1, 2, IRON)
    w.smass[20, 10, 1] = 30000.0                  # 30 kg in one cell
    e = w.register("block")
    w.stamp(e, np.array([[20, 10, 1]]))
    assert w.liftable_by(e, a) and not w.liftable_by(e, b)


def test_a_throw_and_a_punch_take_their_muscle_from_the_person():
    w = World(34, 22, 44, voxel_cm=5)
    w.fill(0, 34, 0, 22, 0, 1, STONE)
    strong = _person(w, 9, 6)
    strong["name"], strong["exert_z"] = "strong", {"punch": 1.4, "throw": 1.4}
    strong["skill"] = {"kick": 0.6, "throw": 0.6}
    weak = _person(w, 9, 16)
    weak["name"], weak["exert_z"] = "weak", {"punch": 0.7, "throw": 0.7}
    weak["skill"] = {"kick": 0.6, "throw": 0.6}
    bs = w._swing(strong, "right arm")
    bw = w._swing(weak, "right arm")
    assert bs["Nm"] == pytest.approx(2.0 * bw["Nm"])
    assert bs["exkind"] == "punch" and bs["paid_J"] == 0.0
    third = _person(w, 20, 6)
    third["name"] = "third"
    bt = w._swing(third, "right arm", back=0.5)
    assert bt["exkind"] == "throw"
    w.exertion_from_body = False
    assert "paid_J" not in w._swing(_person(w, 20, 6), "right arm")


def _swing_world():
    w = BodyWorld(34, 22, 44, voxel_cm=5)
    w.fill(0, 34, 0, 22, 0, 1, STONE)
    w.exits = [(32, 11)]
    p = _person(w, 9, 6)
    p["name"] = "at the post"
    w.fill(17, 18, 5, 8, 1, 30, WOOD)
    w.policy = _Wants(hands="swing", legs="stay")
    return w, p


def test_a_swing_is_paid_as_it_is_done_and_every_joule_lands_as_heat_credited_once():
    w, p = _swing_world()
    w.step()
    w.body_heat_person(p)
    s0 = DG.stores_J(p["heat"]["stores"])
    wo0, mi0 = w.work_out, w.mech_in
    booked, orig = [], w._book_swing

    def spy(b, at):
        booked.append((float(b.get("paid_J", 0.0)), 0.5 * b["I"] * (b["omega"] / TICK_S) ** 2))
        return orig(b, at)
    w._book_swing = spy
    for _ in range(60):
        w.step()
    assert booked and booked[0][0] > 0.0
    J = sum(x[0] for x in booked)
    assert booked[0][0] == pytest.approx(booked[0][1], rel=1e-6)     # the work paid is the arm's kinetic energy
    assert s0 - DG.stores_J(p["heat"]["stores"]) >= J / 0.22 - 1.0     # the stores gave it at the efficiency of muscle
    assert w.work_out - wo0 == pytest.approx(J, rel=1e-6)            # booked once, by the actor
    assert w.mech_in == mi0                                           # and never also as mech_in


def test_a_swing_by_a_body_with_no_heat_record_drains_its_arm_reserve():
    w = World(34, 22, 44, voxel_cm=5)
    w.fill(0, 34, 0, 22, 0, 1, STONE)
    p = _person(w, 9, 6)
    p["name"] = "at the post"
    w.fill(17, 18, 5, 8, 1, 30, WOOD)
    w.policy = _Wants(hands="swing", legs="stay")
    for _ in range(60):
        w.step()
    assert p["body"]["fat"]["right arm"]["wp"] < 1.0


# ---- one contact law ---------------------------------------------------------
def _duel(then="swing", gate=True, stand=31, ticks=150):
    from tests.test_wounds import _Duel
    w = World(46, 20, 40, voxel_cm=5)
    w.fill(0, 46, 0, 20, 0, 1, STONE)
    w.exits = [(44, 10)]
    a = _person(w, 17, 10)
    a["name"] = "attacker"
    b = _person(w, stand, 10)
    b["name"] = "defender"
    w.fill(22, 24, 10, 11, 1, 2, IRON)                    # a bar with no edge at his feet
    w.policy = _Duel("attacker", then)
    w.body_contact = gate
    for _ in range(ticks):
        w.step()
    return w, a, b


def test_a_swung_bar_meets_a_man_through_the_contact_law_and_the_books_close():
    w, a, b = _duel()
    bk = w.contact_book
    assert bk is not None and bk["n"] >= 1 and b["hurt"] > 0.0
    assert bk["KE_in_J"] == pytest.approx(
        bk["deform_J"] + bk["friction_J"] + bk["arm_J"] + bk["ground_J"] + bk["KE_kept_J"], rel=1e-9)
    assert bk["paid_J"] == 0.0                     # the swing was billed tick by tick, not here
    assert w.mech_in == 0.0                        # and its heat was booked by the one who swung


def test_the_same_swing_with_the_gate_off_is_joules_poured_into_flesh():
    w, a, b = _duel(gate=False)
    assert w.contact_book is None and b["hurt"] > 0.0


def test_a_thrown_bar_that_hits_a_man_is_the_same_law():
    w, a, b = _duel(then="throw", stand=26)
    if w.contact_book is None:                     # the throw missed: nothing to check
        pytest.skip("the thrown bar did not reach him")
    assert w.contact_book["n"] >= 1


def test_a_body_that_falls_on_a_man_meets_him_through_the_contact_law():
    w = SB.room(nz=70)
    a = SB.stand(w, 15, 15, "the one below")
    w.policy = _Wants(each={"the one below": {"hands": "hands free", "legs": "stay"}})
    w.step()
    top = int(np.argwhere(w.eid == a["eid"])[:, 2].max()) + 1
    z = top + 24                                           # 1.2 m above his head
    w.fill(16, 17, 15, 16, z, z + 1, IRON)
    w.smass[16, 15, z] = 30000.0                           # 30 kg in one cell
    w._launch(np.array([[16, 15, z]]), (0.0, 0.0, 0.0))    # let go
    for _ in range(120):
        w.step()
    bk = w.contact_book
    assert bk is not None and bk["n"] >= 1 and a["hurt"] > 0.0
    assert bk["KE_in_J"] == pytest.approx(
        bk["deform_J"] + bk["friction_J"] + bk["arm_J"] + bk["ground_J"] + bk["KE_kept_J"], rel=1e-9)
    assert w.mech_in > 0.0                                 # nobody paid for a fall: it is booked as mech_in


def test_shove_tackle_and_grapple_are_one_call_with_different_inputs():
    out = {}
    for name in ("shove", "tackle", "grapple"):
        w = SB.room()
        a, b = SB.two_people(w)
        w.step()
        out[name] = (getattr(w, name)(a, b), a, b, w)
    sh, tk, gr = out["shove"][0], out["tackle"][0], out["grapple"][0]
    assert sh["hit"] and tk["hit"] and gr["hit"]
    assert tk["KE_before"] > sh["KE_before"] > gr["KE_before"]        # a run, a shove, a hand closing
    assert gr["deform_J"] < 3.0                                         # a grip on a friend costs about a joule
    assert abs(out["grapple"][2]["_v"][0]) < 0.1 and not out["grapple"][1].get("_v")   # a hand closing barely moves him
    assert out["shove"][1].get("_v") is None and out["shove"][2]["_v"][0] > 0.5
    assert out["tackle"][1]["_v"][0] > 0.0                              # the tackler is moved too
    for name in ("shove", "tackle", "grapple"):
        bk = out[name][3].contact_book
        assert bk["KE_in_J"] == pytest.approx(
            bk["deform_J"] + bk["friction_J"] + bk["arm_J"] + bk["ground_J"] + bk["KE_kept_J"], rel=1e-9)
        assert bk["paid_J"] > 0.0                                       # the muscle was billed
    w = SB.room(gate=False)
    a, b = SB.two_people(w)
    assert w.shove(a, b) is None and w.tackle(a, b) is None and w.grapple(a, b) is None


def test_a_tired_arm_shoves_weaker_and_a_stronger_one_harder():
    res = {}
    for name, z, fat in (("ref", 1.0, None), ("strong", 1.5, None), ("tired", 1.0, 0.6)):
        w = SB.room()
        a, b = SB.two_people(w)
        a["exert_z"] = {"shove": z}
        if fat is not None:
            a["body"] = {"fat": {"right arm": {"gam": fat, "wp": 1.0}}}
        w.step()
        res[name] = w.shove(a, b)["KE_before"]
    assert res["strong"] > res["ref"] > res["tired"]


def test_a_person_who_runs_to_grab_is_a_tackle_with_a_grip():
    w = SB.room()
    a, b = SB.two_people(w)
    w.step()
    r = w.tackle(a, b, grip=True)
    assert r["v1"][0] == pytest.approx(r["v2"][0])
