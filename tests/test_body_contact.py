"""Wave 14b, lane E: body-on-body contact (one law) and the Hof balance
mechanisms (ankle limit, counter-rotation, a hand on something that holds)."""
import math

import numpy as np
import pytest

from src.voxel import bodycontact as BC
from src.voxel import exertion as EX
from src.voxel import scenes_bodycontact as SB
from src.voxel.scenes import _Wants
from src.voxel.sim import FLESH, STONE, WOOD

N = (1.0, 0.0, 0.0)


# ── the pure law ─────────────────────────────────────────────────────────────
def test_momentum_is_conserved_and_every_joule_is_booked():
    for e in (0.0, 0.2, 0.6, 1.0):
        for v1, v2, n in (((4, 0, 0), (0, 0, 0), N), ((3, 1, 0), (-1, 0.5, 0), N),
                          ((2, 2, 0), (0, 0, 0), (0.6, 0.8, 0))):
            r = BC.collide(70, v1, 55, v2, n, e=e)
            p0 = [70 * v1[i] + 55 * v2[i] for i in range(3)]
            p1 = [70 * r["v1"][i] + 55 * r["v2"][i] for i in range(3)]
            assert np.allclose(p0, p1)
            assert r["KE_before"] == pytest.approx(
                r["KE_after"] + r["deform_J"] + r["friction_J"], rel=1e-9, abs=1e-9)
            assert r["deform_J"] >= 0.0 and r["friction_J"] >= -1e-9


def test_flesh_restitution_sets_the_speed_they_part_at():
    r = BC.collide(70, (4, 0, 0), 70, (0, 0, 0), N, e=BC.E_FLESH)
    assert r["v2"][0] - r["v1"][0] == pytest.approx(BC.E_FLESH * 4.0)
    r = BC.collide(70, (4, 0, 0), 70, (0, 0, 0), N, e=BC.E_FLESH, grip=True)
    assert r["v1"][0] == pytest.approx(r["v2"][0]) == pytest.approx(2.0)     # held: they go together
    assert r["deform_J"] == pytest.approx(0.5 * 35 * 16)


def test_not_closing_nothing_happens():
    r = BC.collide(70, (0, 0, 0), 70, (2, 0, 0), N)
    assert not r["hit"] and r["v2"] == (2.0, 0.0, 0.0) and r["deform_J"] == 0.0


def test_friction_is_coulomb_and_a_grip_never_slips():
    # sliding past with a hard press: friction takes mu x J_n; a light press: it sticks
    r = BC.collide(70, (1, 3, 0), 70, (0, 0, 0), N)
    assert r["J_t"] == pytest.approx(BC.MU_BODY * r["J_n"])
    assert r["friction_J"] > 0.0
    r2 = BC.collide(70, (5, 0.1, 0), 70, (0, 0, 0), N)
    assert abs(r2["v1"][1] - r2["v2"][1]) < 1e-9            # sticks (the slide is small)
    r3 = BC.collide(70, (1, 3, 0), 70, (0, 0, 0), N, grip=True)
    assert r3["v1"][1] == pytest.approx(r3["v2"][1])


def test_the_lighter_body_is_thrown_further_and_the_law_is_order_independent():
    r = BC.collide(90, (3, 0, 0), 45, (0, 0, 0), N)
    eq = BC.collide(70, (3, 0, 0), 70, (0, 0, 0), N)
    assert r["v2"][0] > eq["v2"][0] > eq["v1"][0]          # the lighter one goes further
    assert r["v1"][0] > eq["v1"][0]                       # the heavier one keeps more
    s = BC.collide(45, (0, 0, 0), 90, (3, 0, 0), (-1, 0, 0))   # same meeting, the other way round
    assert s["v1"][0] == pytest.approx(r["v2"][0]) and s["v2"][0] == pytest.approx(r["v1"][0])
    assert s["deform_J"] == pytest.approx(r["deform_J"])


def test_springs_in_series_share_the_energy_by_compliance():
    s1, s2 = BC.partition(60e3, 120e3)
    assert s1 + s2 == pytest.approx(1.0) and s1 == pytest.approx(2.0 / 3.0)   # the softer takes more
    f = BC.peak_force(35.0, 4.0, 60e3, 60e3)
    assert f == pytest.approx(4.0 * math.sqrt(30e3 * 35.0))


# ── the Hof mechanisms, pure ─────────────────────────────────────────────────
def test_counter_rotation_buys_centimetres_and_state_and_delay_and_busy_hands_cost_it():
    w0 = BC.w0_of(0.9)
    full = BC.flywheel_b_max(70, 1.75, w0, 0.10)
    assert 0.03 < full < 0.12                                  # (u) a few centimetres
    assert BC.flywheel_b_max(70, 1.75, w0, 0.10, state=0.5) == pytest.approx(full * 0.5)
    assert BC.flywheel_b_max(70, 1.75, w0, 0.20) < full         # a slower reflex recovers less
    assert BC.flywheel_b_max(70, 1.75, w0, 0.10, arms_free=False) < 0.6 * full


def test_ankle_uses_the_whole_foot_unless_the_leg_is_weak():
    assert BC.ankle_cop_m(70, 1.0) > 0.13           # more than half a foot
    assert BC.ankle_cop_m(70, 0.2) < 0.05


def test_a_wall_holds_a_person_gives_way():
    wall = BC.push_force_N(70, 1.0, math.inf)
    person = BC.push_force_N(70, 1.0, BC.holds_N(45.0))
    assert person < wall and wall == pytest.approx(BC.ARM_PUSH_BW * 70 * 9.81)
    assert BC.external_b_max(70, 1.1, wall) == pytest.approx(0.4 * 1.1)


# ── the world ────────────────────────────────────────────────────────────────
def _cells(w, p):
    return np.argwhere((w.mat == FLESH) & (w.eid == p["eid"]))


def _lean(v, gate=True, setup=None, ticks=40):
    """A person pushed to speed v along x; what happened."""
    w = SB.room(gate=gate)
    p = SB.stand(w, 15, 15, "the person")
    w.policy = _Wants(each={"the person": {"hands": "hands free", "legs": "stay"}})
    if setup:
        setup(w, p)
    w.step()
    x0 = int(_cells(w, p)[:, 0].min())
    w.push_person(p, v, 0.0)
    for _ in range(ticks):
        w.step()
    c = _cells(w, p)
    ev = [e.text for e in w.events.since(-1) if e.why == "reflex:balance"]
    if w.bodies or len(c) != 396:
        return "fell", ev
    dx = int(c[:, 0].min()) - x0
    return ("stood" if dx == 0 else "stepped %d" % dx), ev


def _tired(w, p):
    p["body"] = {"fat": {"legs": {"gam": 0.3, "wp": 1.0}, "right arm": {"gam": 0.3, "wp": 1.0}}}


def _wall(w, p):
    w.mat[24:27, 1:29, 4:39] = STONE
    w.smass[24:27, 1:29, 4:39] = 2600 * w.vox_l
    w._torque_solid = None


def test_gate_off_is_the_old_reflex():
    assert SB.World.body_contact                 # on for every world since 14bF
    how, ev = _lean(0.3, gate=False)
    assert how.startswith("stepped") and ev == ["the person catches their balance with a step"]


def test_a_small_push_is_caught_by_the_arms_and_trunk_with_no_step():
    how, ev = _lean(0.3)
    assert how == "stood" and "flings out their arms" in ev[0]


def test_a_bigger_push_still_needs_a_step_and_a_tired_body_steps_sooner():
    assert _lean(0.8)[0].startswith("stepped")
    assert _lean(0.3, setup=_tired)[0] == "stepped 1"


def test_a_hand_on_a_wall_keeps_a_body_on_its_feet_that_would_have_stepped():
    assert _lean(0.6, gate=False, setup=_wall)[0].startswith("stepped")
    how, ev = _lean(0.6, setup=_wall)
    assert how == "stood" and "puts a hand on something solid" in ev[0]


def test_a_person_with_full_hands_has_no_arms_to_fling():
    """The same small push: free hands fling the arms and keep their feet; a hand
    that has taken up a (nearly weightless) piece of wood has lost the arm share of the
    counter-rotation, so the same body no longer stands still and never flings its arms."""
    def full(w, p):
        c = _cells(w, p)
        x0 = int(c[:, 0].min()) - 1
        w.fill(x0, x0 + 1, 15, 16, 12, 13, WOOD)           # 0.06 kg of wood within reach
        w.policy = _Wants(each={"the person": {"hands": "take", "legs": "stay"}})
    how, ev = _lean(0.3)
    assert how == "stood" and "flings out their arms" in ev[0]
    how_full, ev_full = _lean(0.3, setup=full)
    assert how_full != "stood"
    assert not any("flings out their arms" in t for t in ev_full)


def _ke_in_cells(w):
    return float(w.E.astype(np.float64).sum())


def test_a_shove_pays_from_the_pushers_energy_and_every_joule_lands_as_heat():
    w = SB.room()
    a, b = SB.two_people(w)
    w.step()
    s = EX.strike(a, "shove")                 # the pusher's own shove (about 120 J)
    e0, m0 = _ke_in_cells(w), float(w.mech_in)
    r = w.body_collide(a, b, v_a=(s["v_ms"], 0.0), m_a_eff=s["m_kg"])
    assert r is not None and r["hit"]
    bk = w._cbook()
    assert bk["KE_in_J"] == pytest.approx(
        bk["deform_J"] + bk["friction_J"] + bk["arm_J"] + bk["KE_kept_J"], rel=1e-9)
    assert b["_v"][0] > 0.5 and not a.get("_v")             # the pushed moves; the pusher's feet took it
    for _ in range(80):
        w.step()
    assert not b.get("_v")
    heat = bk["deform_J"] + bk["friction_J"] + bk["arm_J"] + bk["foot_J"] + bk["to_fall_J"] * 0
    assert float(w.mech_in) - m0 == pytest.approx(heat, rel=1e-3)
    assert bk["foot_J"] + bk["to_fall_J"] == pytest.approx(bk["KE_kept_J"], rel=1e-6)


def test_one_law_tackle_grapple_and_shove_differ_only_in_inputs():
    out = {}
    for name, kw in (("tackle", dict(v_a=(4.0, 0.0))), ("grapple", dict(v_a=(4.0, 0.0), grip=True))):
        w = SB.room()
        a, b = SB.two_people(w)
        w.step()
        out[name] = w.body_collide(a, b, **kw)
    assert out["grapple"]["deform_J"] > out["tackle"]["deform_J"]      # held: nothing comes back
    assert out["grapple"]["v1"][0] == pytest.approx(out["grapple"]["v2"][0])


def test_a_hard_tackle_hurts_both_and_a_touch_hurts_nobody_and_gate_off_does_nothing():
    w = SB.room()
    a, b = SB.two_people(w)
    w.step()
    r = w.body_collide(a, b, v_a=(0.5, 0.0))
    assert b["hurt"] == 0.0 and a["hurt"] == 0.0 and not r["injury"]["the pushed"]["broke"]
    w = SB.room()
    a, b = SB.two_people(w)
    w.step()
    r = w.body_collide(a, b, v_a=(9.0, 0.0))        # a sprint into a standing man (44 kg stock figures)
    assert a["hurt"] > 0.0 and b["hurt"] > 0.0
    assert r["F_peak_N"] > 3000.0
    w = SB.room(gate=False)
    a, b = SB.two_people(w)
    w.step()
    assert w.body_collide(a, b, v_a=(6.0, 0.0)) is None and b["hurt"] == 0.0


def test_a_fast_tackle_puts_the_other_person_on_the_floor():
    w = SB.room()
    a, b = SB.two_people(w)
    w.step()
    w.body_collide(a, b, v_a=(8.0, 0.0))
    for _ in range(60):
        w.step()
    assert w.bodies or len(_cells(w, b)) != 396 or int(_cells(w, b)[:, 0].min()) > 25


def test_the_mixin_shadows_no_method_of_any_other_world_mixin():
    """(a helper once named _nearest_cell silently replaced will.py's)"""
    mine = {k for k, v in vars(BC.BodyContactMixin).items() if callable(v) or k in ("body_contact", "contact_book")}
    for cls in SB.World.__mro__:
        if cls is BC.BodyContactMixin or cls is object:
            continue
        assert not (mine & set(vars(cls))), (cls.__name__, mine & set(vars(cls)))
