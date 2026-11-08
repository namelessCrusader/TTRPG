"""Locks (src/voxel/locks.py, wave 13 T11): a pin-tumbler lock as geometry.

Tags: test_echo_* checks a number the constants were FITTED to (the click
energy fraction eta_click was chosen to hit 30-40 dB); test_codeonly_* checks
the code against its own table; test_u_* leans on an unverified number.
Untagged tests are geometry, closed forms and arithmetic identities.
"""
import math

import numpy as np
import pytest

from src.voxel import locks as L
from src.voxel import mechanisms as M

SKILLED = 0.9


def locks(n=200, **kw):
    return [L.make_lock(s, **kw) for s in range(n)]


# ── key and lock ─────────────────────────────────────────────────────────────

def test_the_right_key_turns_one_depth_step_off_does_not_003_mm_off_does_worn_012_does_not():
    for s in range(30):
        lock = L.make_lock(s)
        key = L.key_for(lock)
        assert L.key_matches(lock, key)
        assert L.try_key(lock, key, 1.0)["result"] == "turns"
        for i in range(lock["n"]):
            for dstep in (+1, -1):
                bad = L.make_key([c + dstep for c in key["cuts"][:i]] + [key["cuts"][i] + dstep]
                                 + key["cuts"][i + 1:], blade=lock["keyway"])
                assert not L.key_matches(lock, bad)                          # one depth step off
            off = L.make_key(key["cuts"], wear_mm=[0.03 if j == i else 0.0 for j in range(lock["n"])],
                             blade=lock["keyway"])
            assert L.key_matches(lock, off)                                  # 0.03 mm off still turns
            worn = L.make_key(key["cuts"], wear_mm=[0.12 if j == i else 0.0 for j in range(lock["n"])],
                              blade=lock["keyway"])
            assert not L.key_matches(lock, worn)                             # a worn cut, 0.12 mm
            assert L.try_key(lock, worn, 1.0)["result"] == "wrong_cuts"


def test_one_depth_step_is_far_outside_every_window():
    assert L.STEP_MM > 4 * L.W_HI
    for lock in locks(40):
        assert all(p["w_mm"] <= L.W_HI and p["w_mm"] >= L.W_LO for p in lock["pins"])


def test_the_blade_must_fit_the_keyway_whatever_the_bitting():
    lock = L.make_lock(1)
    right = L.key_for(lock)
    fat = L.make_key(right["cuts"], blade=lock["keyway"] | 0b1000_0001)
    assert L.key_matches(lock, fat)                                          # the bitting is right...
    assert not L.blade_fits(lock, fat)
    assert L.try_key(lock, fat, 1.0)["result"] == "wrong_blade"             # ...and it never enters
    thin = L.make_key(right["cuts"], blade=0b0000_0110)
    assert L.blade_fits(lock, thin)                                          # a thinner blade is fine


def test_rises_are_quantised_to_depth_steps():
    for lock in locks(30):
        for i, p in enumerate(lock["pins"]):
            k = (L.rise_mm(lock, i) - lock["base_mm"]) / lock["step_mm"]
            assert k == pytest.approx(round(k), abs=1e-9) and 0 <= round(k) <= 9
            assert 0.005 <= p["e_mm"] <= 0.05


def test_two_random_locks_accept_each_others_keys_0_of_1000():
    hits = 0
    for j in range(1000):
        a, b = L.make_lock(10_000 + 2 * j), L.make_lock(10_001 + 2 * j)
        hits += L.key_matches(b, L.key_for(a))
    assert hits == 0


def test_locks_replay_from_the_seed():
    assert L.make_lock(77) == L.make_lock(77)
    assert L.make_lock(77) != L.make_lock(78)


def test_a_copy_cut_by_a_machine_keeps_the_steps_and_changes_the_millimetres():
    lock = L.make_lock(5)
    key = L.key_for(lock)
    good = L.copy_key(lock, key, 0.01, seed=1)
    sloppy = L.copy_key(lock, key, 0.2, seed=1)
    assert good["cuts"] == key["cuts"] == sloppy["cuts"]
    assert L.key_matches(lock, good) and not L.key_matches(lock, sloppy)
    assert L.copy_key(lock, key, 0.2, seed=1) == sloppy                       # replays


def test_key_snaps_against_a_high_seize_torque_and_the_bow_is_gone():
    lock = L.make_lock(3, seize_Nm=6.0)                  # a rusted lock
    key = L.key_for(lock)
    My = L.key_yield_Nm(key)
    assert My == pytest.approx(3.0, rel=0.01)            # brass bow, 200 MPa, 15 mm3
    soft = L.try_key(lock, key, 2.0)                     # a gentle hand: stuck, not broken
    assert soft["result"] == "stuck" and not key.get("broken")
    hard = L.try_key(lock, key, 5.0)
    assert hard["result"] == "snaps" and key["bow_Z_mm3"] == 0.0 and key["broken"]
    assert lock["obstructed"]                            # a piece stays in the keyway
    assert L.try_key(lock, key, 1.0)["result"] == "no_bow"
    # a fine lock never snaps it, a wrong key snaps before the pins give
    ok = L.make_lock(3)
    assert L.try_key(ok, L.key_for(ok), 5.0)["result"] == "turns"
    wrong = L.make_lock(4)
    assert L.try_key(wrong, L.make_key([0] * 5, blade=wrong["keyway"]), 10.0)["result"] in ("snaps", "wrong_cuts")


# ── the tension window ───────────────────────────────────────────────────────

def opened(tau, skill=SKILLED, n=200, **kw):
    return sum(L.pick_session(lock, tau, skill, pid="t", **kw)["opened"] for lock in locks(n))


def test_the_window_constants_the_regression_depends_on():
    assert L.TAU_BIND == 0.01 and L.TAU_HOLD == 0.005 and L.TIP_FORCE_MAX == 4.0
    assert L.spring_force_N() == pytest.approx(0.64, abs=0.005)               # 0.64 N/mm x 1 mm
    lo, hi = L.window_Nm()
    assert hi == pytest.approx(0.071, abs=0.002)                              # (4 - 0.64) r / mu
    assert lo < 0.02 < 0.05 < hi < 0.12


def test_tension_window_regression():
    """0.005 opens none; 0.02 and 0.05 open at least 95 percent of 200 locks
    at skill 0.9; 0.12 and 0.3 at most 5 percent. Depends on TAU_BIND 0.01,
    tau_max 0.071, MU_PIN 0.3, TIP_FORCE_MAX 4 N, a 60-probe budget."""
    assert opened(0.005) == 0
    assert opened(0.02) >= 190
    assert opened(0.05) >= 190
    assert opened(0.12) <= 10
    assert opened(0.30) <= 10


def test_the_pick_tool_state_is_kept_per_session_and_a_heavy_wrench_bends_it():
    tool = L.make_pick()
    lock = L.make_lock(1)
    L.pick_session(lock, 0.30, SKILLED, tool=tool)
    assert tool["D"] > 0.0                                  # 0.3 N m is past the 0.175 N m wrench yield
    assert L.tip_force_max(tool) < L.TIP_FORCE_MAX
    gentle = L.make_pick()
    L.pick_session(L.make_lock(1), 0.05, SKILLED, tool=gentle)
    assert gentle["D"] == 0.0


# ── binding order, setting, overset, set loss ────────────────────────────────

def test_the_pin_with_the_largest_bore_error_binds_first_and_setting_it_makes_the_next_bind():
    lock = L.make_lock(8)
    order = sorted(range(lock["n"]), key=lambda i: -lock["pins"][i]["e_mm"])
    tau = 0.03
    seen = []
    for step in range(lock["n"]):
        b = L.binding_pin(lock, tau)
        assert b == order[step]
        assert L.feel_ratio(lock, b, tau) > 0.0
        assert all(L.feel_ratio(lock, i, tau) == 0.0 for i in range(lock["n"]) if i != b)
        assert L.probe(lock, b, tau, L.rise_mm(lock, b)) == "click"
        seen.append(b)
    assert seen == order
    assert L.binding_pin(lock, tau) is None and L.is_open(lock, tau)
    assert L.binding_pin(L.make_lock(8), 0.005) is None                       # under tau_bind nothing binds


def test_a_free_pin_only_springs_and_a_short_lift_falls_back():
    lock = L.make_lock(8)
    b = L.binding_pin(lock, 0.03)
    free = (b + 1) % lock["n"]
    assert L.probe(lock, free, 0.03, L.rise_mm(lock, free)) == "springy"
    assert not lock["pins"][free]["set"]
    assert L.probe(lock, b, 0.03, L.rise_mm(lock, b) - 0.2) == "short"
    assert not lock["pins"][b]["set"]


def test_overset_blocks_the_plug_until_the_tension_is_released():
    lock = L.make_lock(9)
    tau = 0.03
    for _ in range(lock["n"] - 1):
        b = L.binding_pin(lock, tau)
        L.probe(lock, b, tau, L.rise_mm(lock, b))
    last = L.binding_pin(lock, tau)
    w = lock["pins"][last]["w_mm"]
    assert L.probe(lock, last, tau, L.rise_mm(lock, last) + w + 0.1) == "overset"
    assert not L.is_open(lock, tau)                                           # pinned above the shear line
    assert lock["pins"][last]["overset"]
    L.apply_tension(lock, 0.001)                                              # let go
    assert not any(p["overset"] or p["set"] for p in lock["pins"])


def test_set_pins_fall_when_tension_drops_under_tau_hold():
    lock = L.make_lock(11)
    b = L.binding_pin(lock, 0.03)
    L.probe(lock, b, 0.03, L.rise_mm(lock, b))
    assert lock["pins"][b]["set"]
    assert L.apply_tension(lock, 0.01) is False and lock["pins"][b]["set"]    # 0.01 holds
    assert L.apply_tension(lock, 0.004) is True
    assert not lock["pins"][b]["set"]


def test_heavy_tension_jams_the_binding_pin():
    lock = L.make_lock(12)
    b = L.binding_pin(lock, 0.12)
    assert L.probe(lock, b, 0.12, L.rise_mm(lock, b)) == "jammed"
    assert L.lift_force(lock, b, 0.12) > L.TIP_FORCE_MAX
    assert L.lift_force(lock, b, 0.05) < L.TIP_FORCE_MAX


def test_spool_false_set_needs_the_tension_eased():
    lock = L.make_lock(13, spool=1.0)                       # every pin a spool
    tau = 0.05
    b = L.binding_pin(lock, tau)
    assert L.probe(lock, b, tau, L.rise_mm(lock, b)) == "click"
    assert lock["false_set"]
    nxt = L.binding_pin(lock, tau)
    assert L.probe(lock, nxt, tau, L.rise_mm(lock, nxt)) == "false_set"       # tension too high to go on
    assert not lock["pins"][nxt]["set"]
    assert L.probe(lock, nxt, 0.012, L.rise_mm(lock, nxt)) == "springy"        # eased: the waist clears
    assert not lock["false_set"]
    assert L.probe(lock, nxt, tau, L.rise_mm(lock, nxt)) == "click"
    # and a picker who eases opens the all-spool lock
    got = sum(L.pick_session(L.make_lock(s, spool=1.0), 0.05, SKILLED, pid="s")["opened"] for s in range(40))
    assert got >= 35
    plain = sum(L.pick_session(L.make_lock(s), 0.05, SKILLED, pid="s")["opened"] for s in range(40))
    spool_probes = np.mean([L.pick_session(L.make_lock(s, spool=1.0), 0.05, SKILLED, pid="s")["probes"] for s in range(40)])
    plain_probes = np.mean([L.pick_session(L.make_lock(s), 0.05, SKILLED, pid="s")["probes"] for s in range(40)])
    assert spool_probes > plain_probes and plain >= 38


# ── skill ────────────────────────────────────────────────────────────────────

def test_a_novice_opens_fewer_and_takes_more_probes_than_an_expert():
    nov = [L.pick_session(L.make_lock(s), 0.05, 0.1, pid="n") for s in range(100)]
    exp = [L.pick_session(L.make_lock(s), 0.05, 0.9, pid="e") for s in range(100)]
    assert sum(r["opened"] for r in nov) < sum(r["opened"] for r in exp)
    assert np.mean([r["probes"] for r in nov]) > np.mean([r["probes"] for r in exp])
    assert np.mean([r["time_s"] for r in nov]) > np.mean([r["time_s"] for r in exp])
    assert sum(r["oversets"] for r in nov) > sum(r["oversets"] for r in exp)
    # neither opens at 0.3
    assert sum(L.pick_session(L.make_lock(s), 0.3, 0.1, pid="n")["opened"] for s in range(100)) == 0
    assert sum(L.pick_session(L.make_lock(s), 0.3, 0.9, pid="e")["opened"] for s in range(100)) <= 5


def test_the_same_ids_replay_identically():
    a = L.pick_session(L.make_lock(21), 0.04, 0.5, pid="bob")
    b = L.pick_session(L.make_lock(21), 0.04, 0.5, pid="bob")
    c = L.pick_session(L.make_lock(21), 0.04, 0.5, pid="eve")
    assert a == b
    assert isinstance(c["probes"], int)                       # another picker draws other numbers, same law


def test_weber_and_probe_time_follow_skill():
    assert L.weber(0.0) == pytest.approx(0.4) and L.weber(1.0) == pytest.approx(0.1)
    assert L.probe_time_s(0.0) == 3.0 and L.probe_time_s(1.0) == pytest.approx(0.6)


def test_skill_reads_the_person_sheet_when_it_exists():
    assert L.skill_pick(None, default=0.33) == 0.33
    assert L.skill_pick({"skill": {"pick": 0.77}, "pid": 1, "name": "x"}) in (0.77, 0.1)
    try:
        from src.voxel import bodysheet
    except ImportError:
        pytest.skip("bodysheet not built")
    assert L.skill_pick({"skill": {"pick": 0.77}, "pid": 1, "name": "x"}) == 0.77
    d = L.skill_pick({"pid": 2, "name": "y"})
    assert d == bodysheet.skill_default({"pid": 2, "name": "y"}, "pick") and 0.0 <= d <= 0.15


# ── forcing ──────────────────────────────────────────────────────────────────

def test_u_forcing_needs_about_8_5_N_m_at_the_plug():
    lock = L.make_lock(30)
    assert L.forcing_torque_Nm(lock) == pytest.approx(8.5, rel=0.02)
    below = L.force_open(lock, 7.0)
    assert not below["opened"] and not any(p["sheared"] for p in lock["pins"])
    above = L.force_open(lock, 9.0)
    assert above["opened"] and len(above["sheared"]) == lock["n"]
    assert all(p["sheared"] for p in lock["pins"])                           # the pins are gone
    assert L.is_open(lock, 0.0 + L.TAU_HOLD)


def test_forcing_with_the_pick_wrench_bends_it_first():
    tool = L.make_pick()
    lock = L.make_lock(31)
    r = L.force_open(lock, 9.0, tool=tool)
    assert r["tool_yields"] and not r["opened"]
    assert tool["D"] > 0.9                                                  # 1 - 0.175/9
    assert L.tip_force_max(tool) < 0.1 * L.TIP_FORCE_MAX


def test_a_wrench_bends_at_its_yield_and_the_tip_force_falls():
    tool = L.make_pick()
    L.load_wrench(tool, 0.175)
    assert tool["D"] == 0.0
    L.load_wrench(tool, 0.35)
    assert tool["D"] == pytest.approx(0.5)
    assert L.tip_force_max(tool) == pytest.approx(2.0)


# ── the click, through the real _noise law ───────────────────────────────────
@pytest.fixture(scope="module", params=[5, 10])
def world(request):
    from src.voxel.sim import World
    return World(60, 30, 20, voxel_cm=request.param)


def test_echo_click_through_the_real_noise_law_is_30_to_40_dB_at_one_metre(world):
    w = world
    snd = L.emit_click(w, (10.0, 15.0, 10.0))
    assert snd is not None
    db1 = snd["db"]
    assert 30.0 <= db1 <= 40.0                                              # eta_click was chosen to hit this
    # J = eta F dx; the law: 10 log10(ring J / 0.05 / 1e-12) - 8, iron ring 1e-3
    J = L.click_energy_J()
    assert J == pytest.approx(L.ETA_CLICK * 0.6366 * 0.3e-3, rel=0.01)
    assert db1 == pytest.approx(10 * math.log10(1e-3 * J / 0.05 / 1e-12) - 8.0, abs=1e-6)
    cell = 0.1 * w.scale                                                    # metres per cell
    ear = lambda m: (10.0 + m / cell, 15.0, 10.0)             # cell = 0.05 or 0.1 m
    at = snd["at"]
    at10 = w._level(at, ear(10.0), db1)
    at05 = w._level(at, ear(0.5), db1)
    assert at10 < 35.0                                                      # silent 10 m away
    assert at05 > 35.0                                                      # a person at the lock hears it
    assert at10 < db1 - 19.0


def test_a_louder_eta_is_audible_further_and_the_law_is_monotone(world):
    lo = L.emit_click(world, (5, 5, 5), eta=1e-3)["db"]
    hi = L.emit_click(world, (5, 5, 5), eta=1e-2)["db"]
    assert hi - lo == pytest.approx(10.0, abs=1e-6)


def test_a_pick_is_not_a_sounding_verb_the_click_carries_its_own_event(world):
    from src.voxel import events
    assert "picks" not in getattr(events, "_SOUNDING", ())
