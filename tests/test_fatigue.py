"""T10a: fatigue.py (wave 13, spec T10 sections 3.5 and 7 "fatigue").

Tags (plan section 9): closed forms and arithmetic identities have no prefix
and gate. test_u_* = physical data not yet sourced. test_echo_* = a constant
fitted to that number. test_codeonly_* = the code against its own table.
"""
import copy
import json
import os

import pytest

from src.voxel import fatigue as F

HERE = os.path.dirname(os.path.abspath(__file__))


def _status(key):
    """What tests/data/sourced_numbers.json (lane A-S) says about a number, or
    "absent". A test_u_* test puts it in its failure message."""
    path = os.path.join(HERE, "data", "sourced_numbers.json")
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return "absent"
    row = data.get(key) if isinstance(data, dict) else None
    if isinstance(row, dict):
        return str(row.get("status", "listed"))
    return "absent" if row is None else str(row)


def _hold_until_fail(frac, dt=0.1, tmax=20000.0, mvc=400.0):
    s = F.new_state()
    t = 0.0
    while t < tmax:
        s = F.step_static(s, frac * mvc, mvc, dt)
        t += dt
        if s["gam"] >= 1.0:
            return t
    return None


# ------------------------------------------------------------------ static --
def test_u_et_curve_at_half_mvc_is_86_s():
    assert abs(F.et_s(0.5) - 86.0) <= 1.0, _status("et_curve")
    assert abs(F.et_s(0.2) / 60.0 - 9.0) < 0.5            # about 9 minutes
    assert abs(F.et_s(1.0) - 21.9) < 1e-9


@pytest.mark.parametrize("frac", [0.2, 0.5, 0.8])
def test_constant_hold_ends_within_5_percent_of_ET(frac):
    t = _hold_until_fail(frac)
    assert t is not None
    assert 0.95 * F.et_s(frac) <= t <= 1.05 * F.et_s(frac)


@pytest.mark.parametrize("frac", [0.2, 0.5, 0.8])
def test_miners_rule_is_exact_at_constant_load(frac):
    # one big step and many small steps give the same gam (a closed form)
    one = F.step_static(F.new_state(), frac * 400, 400, 30.0)["gam"]
    s = F.new_state()
    for _ in range(300):
        s = F.step_static(s, frac * 400, 400, 0.1)
    assert abs(s["gam"] - one) < 1e-12
    assert abs(one - 30.0 / F.et_s(frac)) < 1e-12


def test_mixed_loads_add_their_fractions_of_life():
    s = F.new_state()
    s = F.step_static(s, 0.8 * 400, 400, 0.25 * F.et_s(0.8))
    s = F.step_static(s, 0.4 * 400, 400, 0.5 * F.et_s(0.4))
    assert abs(s["gam"] - 0.75) < 1e-12                   # Miner: 0.25 + 0.5


def test_gam_saturates_at_one_and_the_hold_has_failed():
    s = F.step_static(F.new_state(), 360.0, 400.0, 1e6)
    assert s["gam"] == 1.0
    assert not F.holds(s, 360.0, 400.0)
    assert not F.holds(s, 40.0, 400.0)                    # worn out: even less


def test_hold_over_the_rested_maximum_fails_at_once():
    assert not F.holds(F.new_state(), 401.0, 400.0)
    assert F.holds(F.new_state(), 399.0, 400.0)
    assert F.holds(F.new_state(), 0.0, 0.0)
    assert not F.holds(F.new_state(), 5.0, 0.0)           # no arm, no hold


def test_recovery_returns_to_10_percent_in_about_10_min():
    s = {"gam": 1.0, "wp": 1.0}
    s = F.step_static(s, 0.0, 400.0, 600.0)
    assert 0.05 < s["gam"] < 0.12
    s2 = F.step_static({"gam": 1.0, "wp": 1.0}, 0.0, 400.0, 240.0 * 2.302585)
    assert abs(s2["gam"] - 0.1) < 1e-3                    # exactly e-fold


def test_u_five_minutes_of_rest_lets_him_hold_again():
    s = F.new_state()
    s = F.step_static(s, 0.5 * 400, 400, 80.0)            # nearly done
    assert s["gam"] > 0.9
    s = F.step_static(s, 0.0, 400.0, 300.0)               # rest
    assert s["gam"] < 0.3
    # he can hold 50 percent again for most of the time he could before
    t = 0.0
    while s["gam"] < 1.0:
        s = F.step_static(s, 200.0, 400.0, 0.5)
        t += 0.5
    assert t > 0.65 * F.et_s(0.5), _status("recovery_tau")


def test_light_load_counts_as_rest_and_recovers():
    s = {"gam": 0.5, "wp": 1.0}
    s2 = F.step_static(s, 0.04 * 400, 400.0, 120.0)       # f 0.04 < 0.05
    assert s2["gam"] < s["gam"]


def test_light_hold_rate_falls_off_fast_below_10_percent():
    assert F.et_s(0.1) > 2000.0
    assert F.et_s(0.05) > 50 * F.et_s(0.1)
    # continuous at the join, and monotone: lighter is longer
    assert abs(F.et_s(0.1 + 1e-9) / F.et_s(0.1 - 1e-9) - 1.0) < 1e-6
    fs = [0.01, 0.03, 0.06, 0.1, 0.2, 0.5, 0.9, 1.0]
    ets = [F.et_s(f) for f in fs]
    assert ets == sorted(ets, reverse=True)


def test_capacity_floor_and_value_at_failure():
    assert F.capacity({"gam": 1.0, "wp": 1.0}, 0.0) == 0.1
    assert F.capacity({"gam": 0.0, "wp": 1.0}, 0.5) == 1.0
    s = {"gam": 1.0, "wp": 1.0}
    assert abs(F.capacity(s, 0.5) - 0.5) < 1e-12          # = the load
    for g in (0.0, 0.3, 0.7, 1.0):
        for f in (0.0, 0.2, 0.9, 1.0, 5.0):
            assert F.capacity({"gam": g, "wp": 1.0}, f) >= 0.1


def test_step_static_does_not_change_its_input():
    s = {"gam": 0.3, "wp": 0.8}
    before = copy.deepcopy(s)
    out = F.step_static(s, 200.0, 400.0, 5.0)
    assert s == before and out is not s and out["wp"] == 0.8


# ----------------------------------------------------------------- dynamic --
def test_below_critical_power_the_reserve_stays_full():
    s = F.new_state()
    for _ in range(100):
        s = F.step_dynamic(s, F.CP_MET - 0.01, 1.0, 70.0)
    assert s["wp"] == 1.0
    assert F.push_factor(s) == 1.0


def test_above_critical_power_the_reserve_falls_linearly():
    met = F.CP_MET + 2.0
    rate = 2.0 * F.MET_W_PER_KG / F.W_PRIME_J_KG          # per second
    s = F.new_state()
    s1 = F.step_dynamic(s, met, 10.0, 70.0)
    s2 = F.step_dynamic(s1, met, 10.0, 70.0)
    assert abs((1.0 - s1["wp"]) - 10.0 * rate) < 1e-12
    assert abs((s1["wp"] - s2["wp"]) - 10.0 * rate) < 1e-12
    empty = F.step_dynamic(s, met, 1e5, 70.0)
    assert empty["wp"] == 0.0


def test_reserve_recovers_with_tau_300_s_when_below_cp():
    s = {"gam": 0.0, "wp": 0.0}
    s = F.step_dynamic(s, 1.5, 300.0, 70.0)
    assert abs(s["wp"] - (1.0 - 1.0 / 2.718281828459045)) < 1e-9
    again = F.step_dynamic({"gam": 0.0, "wp": 0.0}, 1.5, 300.0, 55.0)
    assert again["wp"] == s["wp"]                         # mass does not enter


def test_push_factor_full_above_the_knee_then_falls():
    assert F.push_factor({"gam": 0.0, "wp": 1.0}) == 1.0
    assert F.push_factor({"gam": 0.0, "wp": 0.2}) == 1.0
    mid = F.push_factor({"gam": 0.0, "wp": 0.1})
    assert F.PUSH_AT_EMPTY < mid < 1.0
    assert F.push_factor({"gam": 0.0, "wp": 0.0}) == F.PUSH_AT_EMPTY


# ------------------------------------------------------------------ energy --
def test_metabolic_eta_is_022():
    assert F.ETA == 0.22
    assert abs(F.metabolic_J(100.0, 1.0, 1.0) - 100.0 / 0.22) < 1e-9
    assert abs(F.metabolic_J(100.0, 0.5, 4.0) - 200.0 / 0.22) < 1e-9


def test_u_static_hold_at_30_percent_mvc_costs_about_10_W():
    w = F.metabolic_J(0.0, 0.0, 1.0, static_f=0.3)
    assert 9.0 <= w <= 11.0, _status("static_hold_w")
    assert 5.0 <= F.static_power_w(0.2) <= 15.0           # 5-15 W for 10-30%
    assert 5.0 <= F.static_power_w(0.3) <= 15.0
    assert F.static_power_w(0.0) == 0.0


def test_lowering_costs_less_than_lifting_by_the_eccentric_divisor():
    up = F.metabolic_J(100.0, 1.0, 1.0)
    down = F.metabolic_J(100.0, -1.0, 1.0)
    assert abs(down - 100.0 / F.ECC_DIV) < 1e-9
    assert down < up


def test_holding_moving_load_adds_both_terms():
    both = F.metabolic_J(50.0, 0.2, 2.0, static_f=0.25)
    assert abs(both - (50.0 * 0.2 / 0.22 + 60.0 * 0.25 ** 1.5) * 2.0) < 1e-9


# ------------------------------------------------------------- the reflex view --
def test_fatigue_of_is_the_larger_of_sleep_and_limb():
    p = {"fatigue": 0.2, "body": {"fat": {"left arm": {"gam": 0.7, "wp": 1.0},
                                          "legs": {"gam": 0.1, "wp": 1.0}}}}
    assert F.fatigue_of(p, "left arm") == 0.7             # a tired hand
    assert F.fatigue_of(p, "legs") == 0.2                 # sleep wins here
    assert F.fatigue_of(p, "right arm") == 0.2            # no record: sleep
    assert F.fatigue_of({}, "legs") == 0.0
    assert F.fatigue_of({"fatigue": 3.0}, "back") == 1.0


def test_codeonly_constants_match_the_spec():
    assert F.LIMBS == ("legs", "left arm", "right arm", "back")
    assert (F.ET_A, F.ET_N) == (21.9, 1.98)
    assert F.REC_TAU_S == 240.0 and F.WP_TAU_S == 300.0
    assert F.CP_MET == 6.5 and F.W_PRIME_J_KG == 250.0
    assert F.STATIC_W == 60.0 and F.STATIC_EXP == 1.5


def test_standin_constants_equal_energetics_when_it_is_there():
    EN = pytest.importorskip("src.voxel.energetics")
    assert F.STAND_IN["ETA"] == EN.ETA
    assert F.STAND_IN["MET_W_PER_KG"] == EN.MET_W_PER_KG
    assert F.STAND_IN["ECC_DIV"] == EN.ECC_DIV
