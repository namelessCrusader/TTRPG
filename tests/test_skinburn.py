"""T10a: skinburn.py (wave 13, spec T10 sections 3.1 and 7 "skinburn").

Tags (plan section 9): arithmetic identities, closed forms and bit-for-bit
checks have no prefix and gate. test_u_* = physical data whose source is not
verified (independent of the code, not yet sourced). test_echo_* = a constant
fitted to that number. test_codeonly_* = the code against its own table.
No World is built here.
"""
import copy
import json
import os

import numpy as np
import pytest

from src.voxel import skinburn as S

HERE = os.path.dirname(os.path.abspath(__file__))


def _status(key):
    """What tests/data/sourced_numbers.json (lane A-S) says about a number:
    its status string, or "absent" when the file or the entry is missing. A
    test_u_* test puts this in its failure message, so a failure says which
    status the number had."""
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


def _hold(T_surf):
    """A surface held at T_surf: a contact with an enormous effusivity."""
    return {"kind": "contact", "T_o": T_surf, "e_o": 1e12}


def _time_to(exposure, node, thr, tmax, dt=0.025, face=None, key="omega"):
    f = face if face is not None else S.new_face(region="torso")
    t = 0.0
    while t < tmax:
        S.step([f], dt, [exposure])
        t += dt
        if f[key][node] >= thr:
            return t, f
    return None, f


def _flame(q, **kw):
    e = {"kind": "gas", "q_rad": q, "T_gas": 20.0, "h_c": 10.0, "T_env": 20.0,
         "eps": 0.0}
    e.update(kw)
    return e


# ------------------------------------------------------------ the integral --
def test_henriques_rate_at_60C_is_about_one_per_second():
    r = float(S.henriques_rate(60.0))
    assert 0.5 <= r <= 2.0


def test_omega_rate_is_zero_below_44C_and_positive_above_it():
    assert S.henriques_rate(43.99) == 0.0
    assert S.henriques_rate(25.0) == 0.0
    assert S.henriques_rate(44.5) > 0.0


def test_omega_rate_has_no_step_at_44C():
    # Continuous: zero at 44 C, and a hair above it is still a hair.
    assert S.henriques_rate(44.0) == 0.0
    assert float(S.henriques_rate(44.0 + 1e-6)) < 1e-9
    assert float(S.henriques_rate(44.0 + 1e-6)) >= 0.0


def test_omega_rate_is_monotone_in_temperature():
    T = np.linspace(44.0, 120.0, 400)
    r = S.henriques_rate(T)
    assert np.all(np.diff(r) > 0.0)


def test_u_second_degree_times_within_2x_of_moritz_henriques():
    # Moritz and Henriques 1947 (u): about 1 s at 70, 2 s at 65, 5 s at 60,
    # 30 s at 55 and 10 min at 50 C.
    rows = ((70.0, 1.0, 0.025), (65.0, 2.0, 0.025), (60.0, 5.0, 0.05),
            (55.0, 30.0, 0.5), (50.0, 600.0, 2.0))
    for T, ref, dt in rows:
        t, _ = _time_to(_hold(T), S.BASAL, 1.0, 5000.0, dt=dt)
        assert t is not None
        assert ref / 2.0 <= t <= ref * 2.0, (T, t, ref, _status("henriques_times"))


def test_u_constant_surface_times_match_the_worked_numbers():
    # the prototype's table: 70 C 0.98 s, 65 C 2.3 s, 60 C 8.3 s.
    for T, lo, hi in ((70.0, 0.5, 2.0), (60.0, 3.0, 14.0)):
        t, _ = _time_to(_hold(T), S.BASAL, 1.0, 100.0)
        assert lo <= t <= hi


def test_u_50C_does_nothing_in_two_minutes():
    f = S.new_face()
    for _ in range(int(120 / 0.5)):
        S.step([f], 0.5, [_hold(50.0)])
    assert f["omega"][S.BASAL] < 0.5
    assert S.degree(f) == 0


def test_u_constant_flux_times():
    # absorbed 0.94 q, h 10: 10 kW/m2 in 8-25 s, 40 kW/m2 under 3 s.
    t10, _ = _time_to(_flame(10e3), S.BASAL, 1.0, 100.0)
    t40, _ = _time_to(_flame(40e3), S.BASAL, 1.0, 100.0)
    assert 8.0 <= t10 <= 25.0
    assert t40 < 3.0


def test_no_omega_below_44C_for_a_warm_face_held_an_hour():
    f = S.new_face()
    for _ in range(360):
        S.step([f], 10.0, [_hold(43.0)])
    assert not f["omega"].any()


def test_omega_never_decreases_through_heat_then_cool():
    f = S.new_face()
    last = f["omega"].copy()
    plan = [_flame(30e3)] * 40 + [{"kind": "liquid", "T_liq": 15.0}] * 80 \
        + [_flame(5e3)] * 40
    for e in plan:
        S.step([f], 0.025, [e])
        assert np.all(f["omega"] >= last)
        last = f["omega"].copy()


def test_u_liquid_at_15C_stops_the_gain_within_3_s():
    f = S.new_face()
    for _ in range(40):                                  # 1 s of 40 kW/m2
        S.step([f], 0.025, [_flame(40e3)])
    assert f["omega"][S.BASAL] > 0.05 and f["omega"][0] > 1.0   # cooking
    cool = {"kind": "liquid", "T_liq": 15.0}
    for _ in range(int(3.0 / 0.025)):
        S.step([f], 0.025, [cool])
    assert f["T"].max() < S.T_DAMAGE
    om3 = f["omega"].copy()
    for _ in range(int(17.0 / 0.025)):
        S.step([f], 0.025, [cool])
    assert np.array_equal(f["omega"], om3)               # no gain after 3 s


def test_hot_tissue_keeps_cooking_after_the_flame_goes():
    f = S.new_face()
    for _ in range(40):
        S.step([f], 0.025, [_flame(40e3)])
    om = f["omega"].copy()
    S.step([f], 0.025, [None])
    assert f["omega"].sum() > om.sum()


def test_u_thick_node_undercounts_an_80_kW_dash_at_least_5x():
    # one 3.7 mm node (SKIN_T of the old heat law, rho c d = 13 kJ/m2K)
    thick = S.make_mesh([3.7e-3], [0.5], [3.5e6], [0.0])
    assert abs(3.5e6 * 3.7e-3 - 13e3) < 0.1e3
    bc = S.flux_bc(q_abs=0.94 * 80e3, h=10.0, T_ref=20.0)
    T = np.full((1, 1), 33.0)
    om_thick = 0.0
    for _ in range(40):
        T, _ = S.advance(thick, T, 0.025, bc)
        om_thick += float(S.henriques_rate(T[0, 0])) * 0.025
    f = S.new_face()
    for _ in range(40):
        S.step([f], 0.025, [_flame(80e3)])
    rise_thin = f["T"][0] - 33.0
    rise_thick = T[0, 0] - 33.0
    assert rise_thin >= 5.0 * rise_thick
    assert f["omega"][S.BASAL] >= 1.0                    # a second-degree burn
    assert om_thick * 5.0 < f["omega"][S.BASAL]          # thick node: none


# ------------------------------------------------------------- convergence --
def _ref_time(bc_kind, val, n=120, dt=0.002, tmax=40.0):
    """Times to omega 1 at the basal depth and at the dermis centre with a
    120-node graded mesh and dt 2 ms (a dense solve; the reference)."""
    b = S.graded_bounds(n)
    mesh = S.mesh_from_bounds(b)
    cen = b - mesh["th"] / 2.0
    idx = [int(np.argmin(abs(cen - d))) for d in (0.075e-3, 1.75e-3)]
    nn, G, C, wbth = mesh["n"], mesh["G"], mesh["C"], mesh["wbth"]
    T = S.steady_profile(mesh, 33.0)
    om = np.zeros(nn)
    res = [None, None]
    cache = {}
    rows = np.arange(nn)
    t = 0.0
    while t < tmax and None in res:
        on = om < 1.0
        key = on.tobytes()
        if key not in cache:
            A = np.zeros((nn, nn))
            A[rows, rows] = C / dt + wbth * on
            A[rows[:-1], rows[1:]] = -G
            A[rows[1:], rows[:-1]] = -G
            A[rows[:-1], rows[:-1]] += G
            A[rows[1:], rows[1:]] += G
            A[-1, -1] += mesh["Gb"]
            A[0, 0] += mesh["Gs"] if bc_kind == "T" else 10.0
            cache[key] = np.linalg.inv(A)
        rhs = C / dt * T + wbth * on * 37.0
        rhs[-1] += mesh["Gb"] * 37.0
        if bc_kind == "T":
            rhs[0] += mesh["Gs"] * val
        else:
            rhs[0] += 10.0 * 20.0 + 0.94 * val
        T = cache[key] @ rhs
        om += S.henriques_rate(T) * dt
        t += dt
        for k in range(2):
            if res[k] is None and om[idx[k]] >= 1.0:
                res[k] = t
    return res


def _coarse_time(exposure, tmax=60.0):
    f = S.new_face()
    res = [None, None]
    t = 0.0
    while t < tmax and None in res:
        S.step([f], 0.025, [exposure])
        t += 0.025
        for k, node in enumerate((S.BASAL, S.DERMIS_BASE)):
            if res[k] is None and f["omega"][node] >= 1.0:
                res[k] = t
    return res


@pytest.mark.parametrize("kind,val", [("T", 80.0), ("q", 40e3), ("q", 20e3)])
def test_u_8_node_solution_within_15_percent_of_120_node_reference(kind, val):
    ref = _ref_time(kind, val)
    exposure = _hold(val) if kind == "T" else _flame(val)
    got = _coarse_time(exposure)
    assert None not in ref and None not in got
    for g, r in zip(got, ref):                     # second then third degree
        assert abs(g / r - 1.0) <= 0.15, (kind, val, got, ref)


# ----------------------------------------------------------------- contact --
def test_contact_T_is_the_effusivity_closed_form():
    assert S.contact_T(1100.0, 33.0, 14000.0, 150.0) == \
        (1100.0 * 33.0 + 14000.0 * 150.0) / 15100.0
    assert S.contact_T(1.0, 0.0, 1.0, 100.0) == 50.0
    # equal bodies meet in the middle, a huge body wins
    assert abs(S.contact_T(1100.0, 33.0, 1e12, 150.0) - 150.0) < 1e-6


def test_contact_iron_vs_wood_follow_the_effusivity_formula():
    iron = {"kind": "contact", "T_o": 150.0, "e_o": 14000.0}
    wood = {"kind": "contact", "T_o": 150.0, "e_o": 400.0}
    fi, fw = S.new_face(), S.new_face()
    s0 = fi["T"][0]
    for _ in range(20):                                  # 0.5 s
        S.step([fi, fw], 0.025, [iron, wood])
    Ti = S.contact_T(S.SKIN_E, s0, 14000.0, 150.0)
    Tw = S.contact_T(S.SKIN_E, s0, 400.0, 150.0)
    # the top node sits just under the interface it is held at (the gap is
    # the conduction drop across half of a 0.05 mm node)
    assert Ti - 15.0 < fi["T"][0] <= Ti
    assert Tw - 15.0 < fw["T"][0] <= Tw
    assert fi["T"][0] - fw["T"][0] > 0.8 * (Ti - Tw)
    # and the ranking decides the burn: iron cooks in a blink, wood slower
    ti, _ = _time_to(iron, S.BASAL, 1.0, 60.0)
    tw, _ = _time_to(wood, S.BASAL, 1.0, 60.0)
    assert ti < 0.1 and tw > 10.0 * ti


def test_a_150C_contact_face_separates_from_a_cool_neighbour():
    hot, cool = S.new_face(), S.new_face()
    iron = {"kind": "contact", "T_o": 150.0, "e_o": 14000.0}
    for _ in range(40):
        S.step([hot, cool], 0.025, [iron, None])
    assert hot["T"][0] > 100.0
    assert hot["omega"][S.BASAL] > 1.0
    assert cool["T"][0] < 34.0
    assert not cool["omega"].any()


def test_contact_bookkeeping_is_kept_only_while_in_contact():
    f = S.new_face()
    S.step([f], 0.025, [{"kind": "contact", "T_o": 100.0, "e_o": 14000.0}])
    assert f["semi_infinite"] is True and "contact_T0" in f
    S.step([f], 0.025, [None])
    assert "contact_T0" not in f and "contact_Q" not in f


def test_a_thin_hot_object_gives_up_its_heat():
    thin = {"kind": "contact", "T_o": 150.0, "e_o": 14000.0,
            "heat_cap_area": 2000.0}
    semi = {"kind": "contact", "T_o": 150.0, "e_o": 14000.0}
    a, b = S.new_face(), S.new_face()
    for _ in range(200):
        S.step([a, b], 0.025, [thin, semi])
    assert a["semi_infinite"] is False
    assert a["T"][0] < b["T"][0] - 20.0
    assert a["omega"][S.DERMIS_BASE] < b["omega"][S.DERMIS_BASE]


# --------------------------------------------------------- liquid and steam --
def test_steam_burns_faster_than_dry_air_at_the_same_temperature():
    dry = {"kind": "gas", "T_gas": 100.0, "h_c": 25.0, "T_env": 20.0}
    steam = {"kind": "steam", "T_gas": 100.0}
    ts, _ = _time_to(steam, S.BASAL, 1.0, 600.0)
    td, _ = _time_to(dry, S.BASAL, 1.0, 600.0, dt=0.5)
    assert ts is not None and ts < 2.0
    assert td is None or td > 20.0 * ts


def test_a_scald_burns_and_oil_less_than_water():
    water = {"kind": "liquid", "T_liq": 65.0, "liquid": "water"}
    oil = {"kind": "liquid", "T_liq": 65.0, "liquid": "oil"}
    tw, _ = _time_to(water, S.BASAL, 1.0, 600.0)
    to, _ = _time_to(oil, S.BASAL, 1.0, 600.0)
    assert tw is not None and to is not None and tw < to


def test_unknown_exposure_kind_is_refused():
    with pytest.raises(ValueError):
        S.step([S.new_face()], 0.025, [{"kind": "plasma"}])


# ----------------------------------------------------------------- clothing --
def test_a_clothing_layer_delays_the_burn_and_leaves_with_the_garment():
    cloth = {"thickness": 2e-3, "k": 0.05, "rho_c": 1.2e5, "T0": 20.0}
    bare, dressed = S.new_face(), S.new_face()
    for _ in range(int(3.0 / 0.025)):
        S.step([bare, dressed], 0.025,
               [_flame(20e3), _flame(20e3, clothing=cloth)])
    assert dressed["T"][S.BASAL] < bare["T"][S.BASAL]
    assert dressed["omega"][S.BASAL] < bare["omega"][S.BASAL]
    assert "T_cloth" in dressed and dressed["T_cloth"] > 100.0
    S.step([dressed], 0.025, [None])
    assert "T_cloth" not in dressed


def test_wet_clothing_holds_more_heat_than_dry():
    dry = {"thickness": 2e-3, "k": 0.05, "rho_c": 1.2e5, "T0": 20.0}
    wet = dict(dry, wet=0.5)
    a, b = S.new_face(), S.new_face()
    for _ in range(80):
        S.step([a, b], 0.025, [_flame(10e3, clothing=dry),
                               _flame(10e3, clothing=wet)])
    assert b["T_cloth"] < a["T_cloth"]


# ------------------------------------------------------------------ the rest --
def test_perfusion_cools_and_a_cooked_node_loses_it():
    a, b = S.new_face(), S.new_face()
    for _ in range(int(20.0 / 0.5)):
        S.step([a, b], 0.5, [_hold(45.0), _hold(45.0)], f=[1.0, 0.0])
    assert a["T"][S.DERMIS_BASE] < b["T"][S.DERMIS_BASE]
    c, d = S.new_face(), S.new_face()
    d["omega"][2:6] = 1.0                                # stasis in the dermis
    for _ in range(int(20.0 / 0.5)):
        S.step([c, d], 0.5, [_hold(45.0), _hold(45.0)])
    assert d["T"][S.DERMIS_BASE] > c["T"][S.DERMIS_BASE]


def test_new_face_is_the_steady_profile_between_skin_and_core():
    f = S.new_face(T_start=33.0)
    assert f["T"].shape == (8,) and f["omega"].shape == (8,)
    assert np.all(np.diff(f["T"]) > 0.0)
    assert 32.9 < f["T"][0] < 33.1 and f["T"][-1] < 37.0
    again = S.new_face(T_start=33.0)
    assert np.array_equal(f["T"], again["T"])
    # at rest in its own steady state it stays put (the profile IS steady)
    h = copy.deepcopy(f)
    S.step([h], 0.025, [_hold(33.0)])
    assert np.allclose(h["T"], f["T"], atol=1e-6)


def test_thicker_dermis_burns_through_more_slowly():
    sheet_t = S.new_face(region="back")
    sheet_n = S.new_face(region="face")
    assert sheet_t["dermis_mm"] == 3.0 and sheet_n["dermis_mm"] == 1.2
    assert S.new_face(region="left forearm")["dermis_mm"] == 1.5
    assert S.new_face(region="nowhere")["dermis_mm"] == 2.0
    assert S.new_face(sheet={"skin_scale": 1.5})["dermis_mm"] == 3.0
    b = _hold(70.0)
    t3_back, _ = _time_to(b, S.DERMIS_BASE, 1.0, 400.0, face=sheet_t, dt=0.1)
    t3_face, _ = _time_to(b, S.DERMIS_BASE, 1.0, 400.0, face=sheet_n, dt=0.1)
    assert t3_face < t3_back


def test_degree_depth_and_labels_come_from_the_table():
    f = S.new_face()
    assert S.degree(f) == 0 and S.burn_depth_m(f) == 0.0
    f["omega"][S.BASAL] = 0.54
    assert S.degree(f) == 1
    f["omega"][S.BASAL] = 1.0
    assert S.degree(f) == 2
    f["omega"][:2] = 1.0
    assert abs(S.burn_depth_m(f) - 0.10e-3) < 1e-9
    f["omega"][S.DERMIS_BASE] = 1.0
    assert S.degree(f) == 3
    assert abs(S.burn_depth_m(f) - 2.0e-3) < 1e-9
    # a different table is honoured
    assert S.degree(f, table=((9, S.BASAL, 0.5),)) == 9


def test_pain_heat_starts_at_44_and_is_half_at_49_5():
    assert S.pain_heat(40.0) == 0.0
    assert S.pain_heat(44.0) == 0.0
    assert abs(S.pain_heat(49.5) - 0.5) < 1e-12
    assert S.pain_heat(60.0) == 1.0


def test_energy_open_readout_matches_the_profile_heat_for_a_flash():
    # no perfusion, a 1 s flash: the heat applied is the heat the profile holds
    f = S.new_face()
    C = S.mesh_for(f["dermis_mm"])["C"]
    H0 = float(np.sum(C * f["T"]))
    for _ in range(40):
        S.step([f], 0.025, [_flame(20e3)], f=0.0)
    dH = float(np.sum(C * f["T"])) - H0
    assert abs(dH / f["q_in"] - 1.0) < 0.03


def test_overlay_excess_enthalpy_is_small_against_the_host():
    # the overlay is a bounded one-off of the profile: about 0.5 kJ per face
    # after a 5 s touch at 80 C (the spec's "at most 0.7 kJ"), and it saturates
    # (about 2.3 kJ after ten minutes), against a 5 cm flesh cell of about
    # 440 kJ/K that has taken the energy of reaching that temperature.
    f = S.new_face()
    for _ in range(5):
        S.step([f], 1.0, [_hold(80.0)])
    assert 0.0 < S.excess_enthalpy_J(f, T_host=33.0) < 0.7e3
    for _ in range(595):
        S.step([f], 1.0, [_hold(80.0)])
    host_J = 440e3 * (f["T"][0] - 33.0)
    ex = S.excess_enthalpy_J(f, T_host=33.0)
    assert 0.0 < ex < 3.0e3
    assert ex < 0.01 * host_J


# --------------------------------------------------------------- cold rows --
def test_freeze_rate_table_rows():
    for T, t in S.T_FREEZE:
        assert abs(1.0 / float(S.freeze_rate(T - 1e-9)) - t) < 1e-4 * t
    assert S.freeze_rate(-0.5) == 0.0 and S.freeze_rate(5.0) == 0.0
    assert float(S.freeze_rate(-60.0)) == float(S.freeze_rate(-30.0))
    T = np.linspace(-30.0, -0.6, 100)
    assert np.all(np.diff(S.freeze_rate(T)) < 0.0)


def test_u_cold_rows_frostnip_near_300_s_at_minus_15_C():
    t, f = _time_to(_hold(-15.0), S.BASAL, 1.0, 1200.0, dt=1.0, key="omega_c")
    assert t is not None and 200.0 <= t <= 450.0, _status("frostnip_300s")
    assert S.cold_degree(f) >= 1
    assert not f["omega"].any()                       # cold is not a burn
    assert S.degree(f) == 0


def test_u_minus_half_a_C_does_nothing_in_an_hour():
    f = S.new_face()
    for _ in range(360):
        S.step([f], 10.0, [_hold(-0.5)])
    assert S.cold_degree(f) == 0 and not f["omega_c"].any()


def test_deep_frostbite_needs_the_dermis_base():
    f = S.new_face()
    f["omega_c"][S.BASAL] = 1.0
    assert S.cold_degree(f) == 1
    f["omega_c"][S.DERMIS_BASE] = 1.0
    assert S.cold_degree(f) == 2


# ---------------------------------------------------------- area and shock --
def test_du_bois_70kg_175cm_is_1_85():
    assert abs(S.bsa_dubois(70.0, 175.0) - 1.85) < 0.02


def test_tbsa_arithmetic_is_exact():
    faces = []
    for deg, n in ((1, 7), (2, 10), (3, 4)):
        for _ in range(n):
            f = S.new_face()
            if deg >= 1:
                f["omega"][S.BASAL] = (0.6, 1.0, 1.0)[deg - 1]
            if deg == 3:
                f["omega"][S.DERMIS_BASE] = 1.0
            faces.append(f)
    bsa = 1.85
    part, full, wtd = S.tbsa(faces, bsa)
    a = S.FACE_AREA_M2 / bsa * 100.0
    assert part == 10 * a and full == 4 * a
    assert wtd == (10 * 0.6 + 4 * 1.0) * a             # first degree: 0
    assert S.tbsa([], bsa) == (0.0, 0.0, 0.0)
    assert S.tbsa(faces, bsa, weights={1: 0.0, 2: 1.0, 3: 1.0})[2] == 14 * a


def test_burn_shock_survival_table():
    # 20 percent TBSA_w: loss 24 percent, perfusion about 0.73, conscious.
    L20 = S.shock_events([[0.0, 20.0]], 1e9)
    assert abs(L20 - 0.24) < 1e-9
    assert abs(S.perfusion_after_loss(L20) - 0.73) < 0.01
    faint, death = 0.55, 0.25                          # BODY faint_o2, death_o2
    assert S.perfusion_after_loss(L20) > faint
    L35 = S.shock_events([[0.0, 35.0]], 1e9)
    p35 = S.perfusion_after_loss(L35)
    assert death < p35 < faint                         # faints and survives
    L45 = S.shock_events([[0.0, 45.0]], 1e9)
    assert S.perfusion_after_loss(L45) < death         # dies hours later
    # the thresholds the spec states: faint above ~30 percent, death above ~42
    assert S.perfusion_after_loss(S.shock_events([[0.0, 29.0]], 1e9)) > faint
    assert S.perfusion_after_loss(S.shock_events([[0.0, 31.0]], 1e9)) < faint
    assert S.perfusion_after_loss(S.shock_events([[0.0, 41.0]], 1e9)) > death
    assert S.perfusion_after_loss(S.shock_events([[0.0, 43.0]], 1e9)) < death


def test_shock_accumulates_per_event_and_not_as_one():
    h = 3600.0
    ev = [[0.0, 10.0], [4 * h, 10.0]]
    t = 6 * h
    want = 0.012 * 10.0 * ((1 - np.exp(-t / (6 * h)))
                           + (1 - np.exp(-(t - 4 * h) / (6 * h))))
    assert abs(S.shock_events(ev, t) - want) < 1e-12
    as_one = S.shock_events([[0.0, 20.0]], t)
    assert S.shock_events(ev, t) < as_one - 0.01
    # a burn that has not happened yet adds nothing; at t_i it is zero
    assert S.shock_events(ev, 2 * h) == S.shock_events(ev[:1], 2 * h)
    assert S.shock_events([[5.0, 20.0]], 5.0) == 0.0
    assert S.shock_events([], 1e6) == 0.0


def test_shock_grows_with_time_and_settles():
    ev = [[0.0, 30.0]]
    vals = [S.shock_events(ev, t * 3600.0) for t in (0, 1, 6, 12, 24, 48)]
    assert vals == sorted(vals) and vals[0] == 0.0
    assert abs(vals[-1] - 0.36) < 1e-3


def test_old_events_fold_into_one_constant():
    h = 3600.0
    ev = [[0.0, 20.0], [1 * h, 10.0], [30 * h, 10.0]]
    t = 50 * h
    full = S.shock_events(ev, t)
    folded = S.fold_events(ev, t)
    assert len(folded) == 2 and folded[0][0] == S.SHOCK_CONST_T
    assert abs(S.shock_events(folded, t) - full) < 1e-12
    # the fold keeps the value at fold time; later growth is the small miss
    later = S.shock_events(folded, t + 20 * h)
    assert abs(later - S.shock_events(ev, t + 20 * h)) < 0.01 * 0.012 * 40.0
    assert S.fold_events([], t) == []
    assert S.fold_events(folded, t) == [[S.SHOCK_CONST_T, folded[0][1]]] \
        + folded[1:]


# ------------------------------------------------------------------ batching --
def _varied(n):
    faces, exps = [], []
    kinds = [
        lambda i: _flame(5e3 * (1 + i % 7), h_c=10.0 + i),
        lambda i: {"kind": "contact", "T_o": 40.0 + 2 * i, "e_o": 400.0 * (1 + i)},
        lambda i: {"kind": "liquid", "T_liq": 20.0 + i, "liquid": "water"},
        lambda i: {"kind": "steam"},
        lambda i: {"kind": "gas", "T_gas": 60.0 + i, "h_c": 30.0},
        lambda i: None,
    ]
    for i in range(n):
        f = S.new_face(region=("torso", "back", "face")[i % 3],
                       T_start=30.0 + (i % 5))
        faces.append(f)
        exps.append(kinds[i % len(kinds)](i))
    return faces, exps


def _same(a, b):
    for k in ("T", "omega", "omega_c"):
        if a[k].tobytes() != b[k].tobytes():
            return False
    return a["q_in"] == b["q_in"]


def test_a_batch_of_64_faces_is_bitwise_equal_to_singles():
    batch, ex = _varied(64)
    singles = copy.deepcopy(batch)
    fv = [1.0, 0.5, 0.8, 1.0]
    for _ in range(60):
        S.step(batch, 0.025, ex, f=[fv[i % 4] for i in range(64)])
        for i, f in enumerate(singles):
            S.step([f], 0.025, [ex[i]], f=fv[i % 4])
    assert all(_same(a, b) for a, b in zip(batch, singles))
    assert any(f["omega"].max() > 1.0 for f in batch)        # it did cook
    assert any(f["omega_c"].max() == 0.0 for f in batch)


def test_batch_with_clothing_and_a_cold_face_is_bitwise_equal_to_singles():
    cloth = {"thickness": 1e-3, "k": 0.05, "rho_c": 1e5, "T0": 20.0}
    batch, ex = _varied(12)
    ex[0] = _flame(20e3, clothing=cloth)
    ex[7] = _flame(20e3, clothing=cloth)
    ex[3] = _hold(-20.0)
    singles = copy.deepcopy(batch)
    for _ in range(40):
        S.step(batch, 0.025, ex)
        for i, f in enumerate(singles):
            S.step([f], 0.025, [ex[i]])
    assert all(_same(a, b) for a, b in zip(batch, singles))
    assert batch[0]["T_cloth"] == singles[0]["T_cloth"]
    assert batch[3]["omega_c"].max() > 0.0


def test_order_of_faces_in_a_batch_does_not_change_any_face():
    batch, ex = _varied(20)
    rev = copy.deepcopy(batch)[::-1]
    for _ in range(30):
        S.step(batch, 0.025, ex)
        S.step(rev, 0.025, ex[::-1])
    assert all(_same(a, b) for a, b in zip(batch, rev[::-1]))


def test_step_does_not_change_its_exposures_and_handles_nothing():
    batch, ex = _varied(10)
    ex[0] = _flame(20e3, clothing={"thickness": 1e-3, "k": 0.05,
                                   "rho_c": 1e5})
    before = copy.deepcopy(ex)
    S.step(batch, 0.025, ex)
    assert ex == before
    S.step([], 0.025, [])                              # no faces: no error
    S.step(batch, 0.025)                               # no exposures: still air


def test_prunable_and_face_key():
    f = S.new_face()
    assert S.prunable(f)
    f["omega"][0] = 1e-9
    assert not S.prunable(f)
    g = S.new_face()
    S.step([g], 0.025, [_flame(40e3)])
    assert not S.prunable(g)
    assert S.face_key("left forearm", 12, "+x") == "left forearm|12|+x"


def test_codeonly_constants_match_the_spec_table():
    assert S.BOUNDS_MM == (0.05, 0.10, 0.40, 1.00, 1.50, 2.00, 4.0, 10.0)
    assert np.allclose(S.bounds_for(2.0) * 1e3, S.BOUNDS_MM)
    assert S.HEN_P == 3.1e98 and S.HEN_DE == 6.28e5
    assert abs(S.HEN_DE_R - 75540.0) < 5.0
    assert S.SHOCK_PER_PCT == 0.012 and S.SHOCK_TAU_S == 6 * 3600.0
    assert S.TBSA_WEIGHTS == {1: 0.0, 2: 0.6, 3: 1.0}
    assert S.LIQUID_H == {"water": 1500.0, "oil": 400.0}
    assert S.STEAM_H == 5000.0


# ------------------------------------------------- review fixes (wave 13 A) --
def test_a_cold_thin_object_cools_the_skin_and_never_passes_it():
    # A thin cold blade (heat_cap_area given) must cool the skin like the
    # semi-infinite case does at first, and relax UP toward the skin's
    # starting temperature, never past it.
    thin = {"kind": "contact", "T_o": -20.0, "e_o": 14000.0,
            "heat_cap_area": 5000.0}
    thick = {"kind": "contact", "T_o": -20.0, "e_o": 14000.0}
    ft, fs = S.new_face(), S.new_face()
    for _ in range(40):
        S.step([ft, fs], 0.025, [thin, thick])
    assert ft["T"][0] < 10.0          # clearly cooled, not "ignored"
    assert ft["T"][0] > fs["T"][0]    # the thin one warms up sooner
    for _ in range(4000):
        S.step([ft], 0.025, [thin])
        assert ft["T"][0] <= 33.5     # never above the start (relaxed up to)
    T_o_end = min(-20.0 - ft["contact_Q"] / 5000.0, ft["contact_T0"])
    assert T_o_end <= ft["contact_T0"]


def test_a_thin_object_at_skin_temperature_does_nothing():
    f = S.new_face()
    T0 = f["T"].copy()
    ex = {"kind": "contact", "T_o": f["T"][0], "e_o": 14000.0,
          "heat_cap_area": 5000.0}
    for _ in range(80):
        S.step([f], 0.025, [ex])
    assert abs(f["T"] - T0).max() < 0.05


def test_a_resting_face_stays_at_its_steady_profile_in_default_air():
    for region in ("torso", "palm", "face"):
        f = S.new_face(region=region)
        T0 = f["T"].copy()
        for _ in range(2000):
            S.step([f], 0.025, None)
        assert abs(f["T"] - T0).max() < 1e-6, region
        assert S.prunable(f)


def test_a_hot_face_still_cools_in_default_air():
    f = S.new_face()
    f["T"] = f["T"] + 8.0
    T_hot = f["T"][0]
    for _ in range(4000):
        S.step([f], 0.025, None)
    assert f["T"][0] < T_hot - 1.0
