"""stairs.py: IRC limits as data, climb and descent work, the time table and the
tumble() stand-in (T11 spec v2, tests T-S1 .. T-S4).

Tags: [I] closed form / independent sum, [U] a physical number (u) not yet
sourced (sourced_numbers.json is read when present), [C] code against its own
table. Only [I] lines gate a scene.
"""
import json
import math
import os

import pytest

from src.voxel import stairs as S

SOURCED = os.path.join(os.path.dirname(__file__), "data", "sourced_numbers.json")


def _sourced():
    try:
        with open(SOURCED) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def test_codeonly_IRC_limits_are_data_and_the_headroom_is_2032_not_2036():
    assert S.IRC["riser_max_mm"] == pytest.approx(7.75 * 25.4)
    assert S.IRC["tread_min_mm"] == pytest.approx(10 * 25.4)
    assert S.IRC["headroom_min_mm"] == pytest.approx(80 * 25.4) == 2032.0


@pytest.mark.parametrize("cm", [5, 10])
def test_stair_cells_at_5_and_10_cm_voxels(cm):
    d = cm * 10.0
    st = S.stair_cells(cm, riser_cells=2 if cm == 10 else 4,
                       tread_cells=3 if cm == 10 else 5, n_steps=12)
    assert st["d_mm"] == d
    assert st["riser_mm"] == (200.0 if cm else 0) and st["angle_deg"] > 30
    assert st["rise_cells"] == 12 * (2 if cm == 10 else 4)
    # a flight needs 2032 mm overhead, rounded UP to whole cells
    assert st["headroom_cells"] * d >= 2032.0 > (st["headroom_cells"] - 1) * d
    assert len(st["steps"]) == 12 and st["steps"][-1][1] == st["rise_cells"]
    assert st["issues"] == []                  # half a cell of rounding allowed


def test_the_5cm_flight_4_by_5_is_38_7_degrees_and_passes_with_rounding():
    st = S.stair_cells(5)
    assert st["riser_mm"] == 200.0 and st["tread_mm"] == 250.0
    assert st["angle_deg"] == pytest.approx(math.degrees(math.atan2(200, 250)))
    assert round(st["angle_deg"], 1) == 38.7
    assert 600.0 <= st["blondel_mm"] <= 650.0  # 2R + T = 650 (u)
    assert S.stair_cells(5, riser_cells=3)["issues"] == []


def test_the_IRC_check_flags_a_22_cm_riser_a_20_cm_tread_and_1_9_m_headroom():
    assert S.irc_check(220.0, 254.0, 2032.0) == ["riser too high"]
    assert S.irc_check(180.0, 200.0, 2032.0) == ["tread too short"]
    assert S.irc_check(180.0, 254.0, 1900.0) == ["headroom too low"]
    assert S.irc_check(196.85, 254.0, 2032.0) == []        # the limits themselves pass
    assert S.irc_check(180.0, 254.0, 2031.0) == ["headroom too low"]
    assert len(S.irc_check(250, 150, 1500)) == 3


# ── T-S2: work ──────────────────────────────────────────────────────────────
def test_climb_70kg_3m_run_3_75_gives_10_0_kJ_metabolic_within_1_percent():
    m, dz, run = 70.0, 3.0, 3.75
    w = S.climb_work(m, dz, run)
    by_hand = m * 9.81 * dz / 0.22 + 2.5 * m * run          # the stated formula
    assert w["metabolic_J"] == pytest.approx(by_hand, rel=1e-12)
    assert w["metabolic_J"] == pytest.approx(10000.0, rel=0.01)
    assert w["mech_J"] == pytest.approx(2060.1, rel=1e-3)
    assert w["level_J"] == pytest.approx(656.25)


def test_the_climb_identity_actor_metabolic_equals_body_heat_plus_m_g_dz():
    for m, dz, run in ((70.0, 3.0, 3.75), (43.9, 0.9, 1.0), (90.0, 5.0, 6.0)):
        w = S.climb_work(m, dz, run)
        assert w["metabolic_J"] == pytest.approx(w["heat_J"] + m * 9.81 * dz, rel=1e-12)


def test_descent_is_cheaper_than_climbing_dearer_than_the_level_and_heats_the_body():
    m, dz, run = 70.0, 3.0, 3.75
    up, dn = S.climb_work(m, dz, run), S.descent_work(m, dz, run)
    assert dn["level_J"] < dn["metabolic_J"] < up["metabolic_J"]
    # heat in the body on the way down = m g dz + the metabolic part
    assert dn["heat_J"] == pytest.approx(m * 9.81 * dz + dn["metabolic_J"], rel=1e-12)
    assert dn["heat_vertical_J"] == pytest.approx(3776.85, rel=1e-4)    # spec: 3.78 kJ
    assert dn["fall_heat_J"] == pytest.approx(m * 9.81 * dz)
    # sign picks the regime; flat is only the level part
    assert S.step_work(m, -dz, run) == dn and S.step_work(m, dz, run) == up
    flat = S.step_work(m, 0.0, run)
    assert flat["metabolic_J"] == pytest.approx(flat["level_J"])


def test_the_efficiency_is_the_body_heat_topics_number():
    assert S.ETA_UP == 0.22 and S.ECC_DIV == 1.2
    try:
        from src.voxel import energetics as E
    except ImportError:                              # pragma: no cover
        pytest.skip("energetics.py (A-T7b) not present")
    assert (S.ETA_UP, S.ECC_DIV, S.GRAV) == (E.ETA, E.ECC_DIV, E.GRAV)
    # and the descent heat is the same bookkeeping as energetics.act_power
    a = E.act_power(2060.1, kind="lower")
    d = S.descent_work(70.0, 3.0, 0.0)
    assert d["heat_vertical_J"] == pytest.approx(a["heat_J"], rel=1e-3)
    assert d["vertical_J"] == pytest.approx(a["met_J"], rel=1e-3)


# ── T-S3: time ──────────────────────────────────────────────────────────────
def test_F0_is_exactly_8_and_the_table_never_falls_below_it():
    for cm in (5, 10):
        F = S.step_table(cm, 8)
        assert F[0] == 8 and len(F) == 9
        assert all(a >= 8 for a in F) and F == sorted(F)


def test_a_20_cm_riser_at_250_W_costs_0_55_s_and_a_flat_cell_0_036_s():
    t = S.step_time(0.20, 0.05, mass_kg=70.0)
    assert t == pytest.approx(0.20 * 70.0 * 9.81 / 250.0, rel=1e-12)
    assert t == pytest.approx(0.55, abs=0.005)
    assert S.step_time(0.0, 0.05) == pytest.approx(0.05 / 1.4)
    assert 0.0357 < S.step_time(0.0, 0.05) < 0.0358
    # F for a 4-cell rise at 5 cm is the time ratio, in eighths
    F = S.step_table(5, 6)
    assert F[4] == round(8 * t / (0.05 / 1.4)) == 123


def test_the_unhurried_stair_pace_agrees_with_the_power_cap_to_2_percent():
    p = S.unhurried_power(70.0, 0.20, 1.8)
    assert p == pytest.approx(1.8 * 0.20 * 70.0 * 9.81)
    assert p <= S.P_STAIR
    assert abs(p - S.P_STAIR) / S.P_STAIR < 0.02
    assert S.STAIR_V_X == pytest.approx(0.457, abs=0.001)
    # strength scales the cap
    assert S.climb_speed(70.0, 800.0) == pytest.approx(2 * S.climb_speed(70.0, 400.0))


def test_u_the_stair_power_is_marked_unsourced_until_a_source_says_otherwise():
    s = _sourced()
    seen = "no sourced_numbers.json" if s is None else "file present"
    assert S.P_STAIR == 250.0, seen          # a (u) number, status: %s


# ── T-S4: tumble ────────────────────────────────────────────────────────────
CASES = [(1.2, 0.02, 12), (1.2, 0.02, 8), (2.0, 0.02, 12), (3.0, 0.05, 12),
         (0.5, 0.4, 6), (2.5, 0.0, 12), (1.0, 0.1, 3)]


@pytest.mark.parametrize("vx0,mu,n", CASES)
def test_tumble_the_two_independent_energy_sums_agree_to_1e_9(vx0, mu, n):
    r = S.tumble(30.0, vx0, 0.2, 0.25, n, mu=mu)
    assert r["tumbled"]
    assert r["speed_loss_J"] == pytest.approx(r["formula_loss_J"], rel=1e-9, abs=1e-9)
    for im in r["impacts"]:
        assert im["speed_loss_J"] == pytest.approx(im["formula_loss_J"], rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("vx0,mu,n", CASES)
def test_tumble_m_g_H_plus_KE0_equals_the_losses_plus_KE_end(vx0, mu, n):
    m = 30.0
    r = S.tumble(m, vx0, 0.2, 0.25, n, mu=mu)
    lhs = m * 9.81 * r["H_m"] + r["ke0"]
    assert lhs == pytest.approx(r["speed_loss_J"] + r["ke_end"], rel=1e-9, abs=1e-9)
    assert lhs == pytest.approx(r["formula_loss_J"] + r["ke_end"], rel=1e-9, abs=1e-9)


def test_tumble_with_vx0_zero_from_rest_m_g_H_is_losses_plus_KE_end():
    m = 30.0
    r = S.tumble(m, 0.0, 0.2, 0.25, 12)
    assert not r["tumbled"] and r["n_impacts"] == 0 and r["ke_end"] == 0.0


def test_no_tumble_on_a_flat_landing_or_on_a_static_slope():
    flat = S.tumble(70.0, 3.0, 0.2, 0.25, 0)
    assert not flat["tumbled"] and flat["n_impacts"] == 0
    assert flat["ke_end"] == flat["ke0"]
    still = S.tumble(70.0, 0.0, 0.2, 0.25, 12)
    assert not still["tumbled"] and still["ke_end"] == 0.0


def test_tumble_impacts_and_losses_rise_with_the_flight_and_exit_is_slower_than_free_fall():
    runs = [S.tumble(30.0, 1.2, 0.2, 0.25, n, mu=0.02) for n in (2, 4, 8, 12)]
    imp = [r["n_impacts"] for r in runs]
    assert imp == sorted(imp) and imp[-1] > imp[0]
    loss = [r["speed_loss_J"] for r in runs]
    assert loss == sorted(loss)
    for n, r in zip((2, 4, 8, 12), runs):
        assert r["exit"] is not None
        v = math.hypot(*r["exit"])
        assert v < math.sqrt(2 * 9.81 * n * 0.2)           # below the free-fall speed
        assert v > 0.0
    # and it ends at the landing, not on a tread
    assert all(r["stopped_on"] is None for r in runs)


def test_a_slow_enough_body_comes_to_rest_on_a_tread_with_its_energy_booked():
    r = S.tumble(30.0, 0.5, 0.2, 0.25, 12, mu=0.4)
    assert r["stopped_on"] is not None and r["exit"] is None
    assert r["ke_end"] == pytest.approx(0.0, abs=1e-12)
    assert r["H_m"] == pytest.approx(r["stopped_on"] * 0.2)
