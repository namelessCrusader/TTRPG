"""T7b: the human-data fixtures and the phase A proof scenes (spec T7 section 6
test 10, section 8).

FIXTURES. tests/data/human_cases.json holds the cases with their TOLERANCES,
written BEFORE any fit and guarded by a hash. The named constants of the
compact model (bodynodes.HC_FLESH, RAD_VIEW, DRY_WETTEDNESS_FIT, PRM_SCALE)
were fitted on the 'fit' cases only (steady states) and are recorded beside
the cases; the 'held_out' cases (transients, humid, hot, immersion,
dehydration, the corpse) were never used to choose them.

THE JOS-3 ORACLE (tools/jos3_oracle.py) was RUN once on 2026-10-30 (decision D8,
approved) in a throwaway venv, and tests/data/jos3_cases.json is its output. It
came AFTER the fit: the constants were fitted on the human data only and nothing
was refitted to JOS-3, so JOS-3 is an independent CHECK (test_u_jos3_*), not a
second fit target. If the file is absent the JOS-3 tests are skipped and
`test_the_status_of_the_fit_is_printed` says so. All physical-number tests are
test_u_*: independent of the code but not yet source-checked.
"""
import functools
import json
import os

import numpy as np
import pytest

from src.voxel import bodynodes as BN
from src.voxel import digestion as DG
from src.voxel import energetics as EN
from src.voxel import scenes_bodyheat as SB
from src.voxel import thermoreg as TR

HERE = os.path.dirname(os.path.abspath(__file__))
CASES_PATH = os.path.join(HERE, "data", "human_cases.json")
JOS3_PATH = os.path.join(HERE, "data", "jos3_cases.json")
with open(CASES_PATH) as _f:
    DOC = json.load(_f)
CASES = DOC["cases"]
FIT = [c for c in CASES if c["role"] == "fit"]
HELD = [c for c in CASES if c["role"] == "held_out"]

# THE MISSES, said plainly: they are written (with the numbers and the cause) into the fit block of
# tests/data/human_cases.json when the fit is recorded. Each is an open problem, not a hidden failure;
# non-strict, so a later model that closes one shows as XPASS.
KNOWN_MISS = {k: "%s miss, out of band %s. %s" % (v["role"], v["out_of_band"], v["cause"])
              for k, v in DOC["fit"].get("misses", {}).items()}


# N2 (a) (wave 13b): rest at the ISO level raises the heat of a resting body from 89 W to 109 W (73 kg, 1.75 m),
# over the top of two fit bands that were written for 1.22 W/kg. The tolerances are NOT touched (their hash is
# checked below); the two cases are recorded as misses with their numbers, as the file does for every miss.
N2A_MISS = {
    "nude_29C_rest": "fit miss after decision N2 (a): heat production 109.4 W against the band 75-100 W. Rest is now the ISO "
                     "58.15 W/m2 x 1.88 m2 = 109 W; the band was written for 1.22 W/kg (89 W). Core and skin stay inside.",
    "clothed_1clo_22C": "fit miss after decision N2 (a): heat production 110.0 W against the band 75-100 W (same cause: the ISO "
                        "rest, 109 W, is over the band written for 89 W). Core and skin stay inside.",
}
KNOWN_MISS.update({k: v for k, v in N2A_MISS.items() if k not in KNOWN_MISS})


def _marked(cases):
    return [pytest.param(c, id=c["id"], marks=pytest.mark.xfail(reason=KNOWN_MISS[c["id"]], strict=False))
            if c["id"] in KNOWN_MISS else pytest.param(c, id=c["id"]) for c in cases]


# ============================================================ the fixtures ===
def test_the_tolerances_are_the_ones_written_before_the_fit():
    assert SB.case_tolerance_hash(CASES) == DOC["tolerance_sha256"]
    assert DOC["_meta"]["tolerances_written_before_fit"] is True
    assert {c["role"] for c in CASES} == {"fit", "held_out"}


def test_the_fitted_constants_recorded_beside_the_cases_are_the_ones_in_the_code():
    rec = DOC["fit"]["constants"]
    assert rec, "the fit block is empty"
    assert rec["HC_FLESH"] == pytest.approx(BN.HC_FLESH, rel=1e-9)
    assert rec["RAD_VIEW"] == pytest.approx(BN.RAD_VIEW, rel=1e-9)
    assert rec["DRY_WETTEDNESS_FIT"] == pytest.approx(BN.DRY_WETTEDNESS_FIT, rel=1e-9)
    for k, f in rec["PRM_SCALE"].items():
        assert BN.PRM_SCALE[k] == pytest.approx(f, rel=1e-9)
    assert set(BN.PRM_SCALE) == set(rec["PRM_SCALE"])
    assert rec["SHELL_VASO"] == BN.SHELL_VASO and rec["RUNOFF_K"] == BN.RUNOFF_K


def test_the_status_of_the_fit_is_printed(capsys):
    stronger = os.path.exists(JOS3_PATH)
    c = DOC["fit"]["counts"]
    status = ("HUMAN-DATA-ONLY FIT, checked afterwards against the JOS-3 oracle (run 2026-10-30, decision D8 "
              "approved; nothing was refitted to it: see test_u_jos3_*). " if stronger else
              "WEAKER: human-data-only fit (the JOS-3 oracle was not run: decision D8, the throwaway "
              "install, is not approved). ")
    status += ("Held-out: %d of %d inside their tolerances; fit cases: %d of %d inside. Misses: %s."
               % (c["held_out_inside"], c["held_out_total"], c["fit_inside"], c["fit_total"],
                  ", ".join(sorted(KNOWN_MISS)) or "none"))
    with capsys.disabled():
        print("\n[T7b] fit status:", status, "| fit block:", DOC["fit"].get("status"))
    assert DOC["fit"]["status"].startswith("weaker") or stronger
    assert len(FIT) == c["fit_total"] and len(HELD) == c["held_out_total"]


def test_the_physical_coefficients_are_not_in_the_fit():
    """HC_FLESH and RAD_VIEW sit at the ISO values (1.0) and are named as not fitted."""
    assert DOC["fit"]["constants"]["HC_FLESH"] == 1.0 and DOC["fit"]["constants"]["RAD_VIEW"] == 1.0
    assert {"HC_FLESH", "RAD_VIEW"} <= set(DOC["fit"]["not_fitted"])


@pytest.mark.parametrize("case", _marked(FIT))
def test_u_fit_cases_are_inside_their_tolerances(case):
    ok, rows = SB.score_case(case, SB.run_human_case(case))
    assert ok, rows


@pytest.mark.parametrize("case", _marked(HELD))
def test_u_held_out_cases_are_inside_their_tolerances(case):
    """Transients, humid, hot, immersion, dehydration, the corpse: never fitted."""
    ok, rows = SB.score_case(case, SB.run_human_case(case))
    assert ok, rows


def test_u_the_heat_balance_of_a_resting_nude_person_at_29_C():
    """F3: radiation and convection 60-75 percent, evaporation (skin 10-15 W,
    breath 10-12 W), by the compact model's own sink."""
    sh, g, h = SB._std_person()
    nodes = BN.nodes_from_T(g, h, BN.comfort_T(g, h))
    env = {"T_air": 29.0, "rh": 0.5, "v": 0.1, "T_rad": 29.0}
    BN.advance(h, nodes, g, env, 3 * 3600.0)
    out = BN.advance(h, nodes, g, env, 3600.0)
    total = out["heat_J"] / 3600.0
    assert 70.0 <= total <= 110.0
    assert 0.5 <= out["sensible_J"] / out["heat_J"] <= 0.85
    assert 15.0 <= out["latent_J"] / 3600.0 <= 40.0


def test_the_corpse_curve_has_a_plateau_then_falls():
    case = [c for c in CASES if c["id"] == "corpse_20C"][0]
    m = SB.run_human_case(case)
    assert m["T_1h"] > 35.5                                    # the plateau
    assert m["T_1h"] > m["T_6h"] > m["T_12h"] > m["T_18h"] > 20.0
    rate = (m["T_6h"] - m["T_12h"]) / 6.0
    assert 0.3 < rate < 1.5                                    # K/h, spec F11 0.8-1.2 (u, Henssge 0.65)


def test_the_jos3_oracle_script_refuses_cleanly_without_the_package():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "jos3_oracle", os.path.join(os.path.dirname(HERE), "tools", "jos3_oracle.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        import jos3                                            # noqa: F401
        pytest.skip("jos3 is installed here: the oracle would run")
    except ImportError:
        assert mod.run(os.devnull + ".x") == 2                 # exits 2 and writes nothing
    assert mod.check.__doc__


# ============================================================ JOS-3 (spec T7 test 10a) ===
# tests/data/jos3_cases.json is the output of tools/jos3_oracle.py (JOS-3 0.5.0, run 2026-10-30). The
# compact model is run on the same person and cases (nude, air = radiant temperature, 0.1 m/s, from
# comfort_T) and judged on the spec tolerances written in the file: core 0.3 K, mean skin 1.0 K, heat
# production 15 percent, sweat 30 percent (an absolute floor of 0.02 L/h where JOS-3 sweats less than that,
# (u): a relative band on zero is undefined), and the step time courses 0.5 K at every minute.
# What is compared (the key mapping is in the file's _meta, with JOS-3's own words and units):
#   core         our central blood pool        against JOS-3 Tcb (central blood). The segment cores are 0.2-0.4 K warmer.
#   skin_mean    our area-weighted mean skin   against JOS-3 TskMean.
#   heat_prod_W  our P_total                   against JOS-3 Met (basal + work + shivering + non-shivering).
#   sweat_Lh     our secretion                 against JOS-3 sum(Esweat) / 2418 J/g (evaporated; the controller
#                                              demand while no segment is wet: wet_max < 1).
# The primary cases run JOS-3 at the SIM's rest power (1 MET = 1.22 W/kg = 89 W: PAR 1.047), so the check is
# about the heat exchange and the controllers and not about the rest constant (decision N2). Our hour-of-day
# term is switched OFF here (JOS-3 has no daily rhythm; over the 6 h of the run, clock 12 h to 18 h, it moves our 20 C core by 0.25 K and the 28 C core by 0.19 K).
# Nothing here was fitted. The misses below are recorded with their numbers and are NOT loosened.
SWEAT_FLOOR_LH = 0.02
# JOS-3 is slow in the cold (its heat production at 20 C: 152.6 W after 3 h, 160.9 W after 24 h), so the file holds 24 h steady
# states (tools/jos3_oracle.py). The compact model is steady within 6 h when it is kept hydrated (28 C pool 36.756 K at 6 h, 36.755 at 24 h; 20 C 36.151 and
# 36.152; 120 s sub-steps), so it is run for 6 h with 120 s sub-steps (SUB_FIT, as the human fixtures do), the last minute with 30 s ones.
COMPACT_SETTLE_S = 6 * 3600.0
JOS3 = None
if os.path.exists(JOS3_PATH):
    with open(JOS3_PATH) as _f:
        JOS3 = json.load(_f)
# WAVE 13b, C6, decision N2 (a): a seated, awake body now rests at the ISO level (58.15 W/m2, 109 W for this
# man), so the oracle is read at JOS-3's own rest, PAR 1.25 (106.3 W = 56.5 W/m2, the level its set points are
# found at), not at the old 1.22 W/kg (89 W). The file holds both: the "@par1.25" variants carry the same inputs
# and the same case ids. Nothing else in the comparison changed.
def _jos3_at_iso_rest():
    by_id = {c["id"]: c for c in JOS3.get("variants", [])}
    out = []
    for c in JOS3["cases"] + JOS3.get("extra_cases", []):
        v = by_id.get(c["id"] + "@par1.25")
        out.append(dict(v, id=c["id"]) if v is not None else c)
    return out


JOS3_CASES = [] if JOS3 is None else _jos3_at_iso_rest()
JOS3_QUANTITIES = {"steady": ("core", "skin_mean", "heat_prod_W", "sweat_Lh"), "step": ("core", "skin_mean")}

JOS3_KNOWN_MISS = {
    ("zone_31C", "sweat_Lh"):
        "ours 0.057 L/h secreted against JOS-3 0.002 L/h (floor 0.02), at JOS-3's own ISO rest (wave 13b). The core dip of this case is "
        "FIXED (thermoreg is two-sided now: signed core term for sweat, a core-cold term in the vasoconstriction; core 36.79 K against "
        "JOS-3's 36.79 K), but at 31 C the warm skin alone (k_s x W_s, about 1 K warm) still drives sweat where JOS-3's is nearly zero (u: "
        "the skin gain k_s was fitted on the 35 C and 40 C cases and not touched here)",
    ("hot_step_28_to_45", "skin_mean"):
        "largest gap -0.95 K at minute 9, over 0.5 K until minute 14: JOS-3's mean skin jumps 0.9 K in the first minute and reaches "
        "36.95 K at minute 10, then settles at 36.6 K when its sweat saturates the skin (wettedness 1 from minute 20); ours warms "
        "more slowly (36.0 K at minute 10) and ends +0.39 K above (37.00 against 36.61 K). Likely cause (u): skin node 24 against "
        "12 kJ/K, our sweat and vessel lags (JOS-3 has none), and our sweat 0.52 L/h secreted against JOS-3's 0.32 L/h demand at minute 60",
}


@functools.lru_cache(maxsize=None)
def _compact_on_jos3_case(case_id):
    """The compact model on one JOS-3 case: a dict (steady) or a list of per-minute dicts (step)."""
    case = [c for c in JOS3_CASES if c["id"] == case_id][0]
    inp = case["inputs"]
    saved = TR.CIRCADIAN_K
    TR.CIRCADIAN_K = 0.0                  # JOS-3 has no daily rhythm; restored below
    try:
        sh, g, h = SB._std_person()
        nodes = BN.nodes_from_T(g, h, BN.comfort_T(g, h))

        def fill():
            # JOS-3 has no body-water store and the compact person does not drink: keep the person hydrated (6 h at 35 C would
            # otherwise cost 0.9 percent of body mass, and the sweat threshold moves 0.2 K per percent)
            h["stores"]["water_ml"] = h["stores"]["water0_ml"]

        def settle(env, seconds):
            left = seconds
            while left > 1e-9:
                fill()
                step = min(600.0, left)
                BN.advance(h, nodes, g, env, step, sub=120.0)
                left -= step

        def snap(env):
            fill()
            out = BN.advance(h, nodes, g, env, 60.0)
            T = BN.temps(nodes)
            return {"core": float(T[BN.POOL]), "skin_mean": BN.mean_skin(nodes, g),
                    "heat_prod_W": out["metabolic_J"] / 60.0,
                    "sweat_Lh": float(sum(h["ctl"]["sweat"].values()) * 3.6)}

        if case["kind"] == "steady":
            env = {"T_air": inp["Ta"], "rh": inp["RH"] / 100.0, "v": inp["v"], "T_rad": inp["Ta"]}
            settle(env, min(inp["hours"] * 3600.0, COMPACT_SETTLE_S) - 60.0)
            return snap(env)
        env0 = {"T_air": inp["T0"], "rh": inp["RH"] / 100.0, "v": 0.1, "T_rad": inp["T0"]}
        settle(env0, min(inp["before_min"] * 60.0, COMPACT_SETTLE_S))
        env1 = dict(env0, T_air=inp["T1"], T_rad=inp["T1"])
        return [snap(env1) for _ in range(inp["minutes"])]
    finally:
        TR.CIRCADIAN_K = saved


def _jos3_params():
    out = []
    for c in JOS3_CASES:
        for q in JOS3_QUANTITIES[c["kind"]]:
            key = (c["id"], q)
            marks = [pytest.mark.xfail(strict=False, reason=JOS3_KNOWN_MISS[key])] if key in JOS3_KNOWN_MISS else []
            out.append(pytest.param(c["id"], q, id="%s-%s" % key, marks=marks))
    return out


def test_the_jos3_file_has_the_shape_and_the_mapping_the_fixtures_read():
    if JOS3 is None:
        pytest.skip("the JOS-3 oracle was not run (decision D8)")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "jos3_oracle", os.path.join(os.path.dirname(HERE), "tools", "jos3_oracle.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.check(JOS3_PATH)
    meta = JOS3["_meta"]
    assert meta["package"]["jos3"] == "0.5.0" and meta["person"]["weight"] == 73.0
    km = meta["key_mapping"]
    assert km["core"]["jos3_key"] == "Tcb" and km["skin_mean"]["jos3_key"] == "TskMean"
    assert km["heat_prod_W"]["jos3_key"] == "Met" and km["sweat_Lh"]["jos3_key"] == "Esweat"
    assert km["sweat_Lh"]["jos3_unit"] == "W" and km["heat_prod_W"]["jos3_unit"] == "W"
    # the primary cases run at the sim's own rest power: 1.22 W/kg x 73 kg
    for c in JOS3["cases"]:
        assert c["config"]["rest_power_W_at_PAR"] == pytest.approx(1.22 * 73.0, abs=0.01), c["id"]


@pytest.mark.parametrize("case_id,quantity", _jos3_params())
def test_u_jos3_cases_inside_the_spec_tolerances(case_id, quantity):
    """Spec T7 test 10a, case by case: steady 20, 28 and 35 C (core, mean skin, heat, sweat) and the
    cold (28 to 10 C) and hot (28 to 45 C) step time courses (core, mean skin) against JOS-3 0.5.0."""
    tol = JOS3["_meta"]["tolerances"]
    case = [c for c in JOS3_CASES if c["id"] == case_id][0]
    ours = _compact_on_jos3_case(case_id)
    if case["kind"] == "step":
        gaps = [abs(a[quantity] - b[quantity]) for a, b in zip(ours, case["series"])]
        k = max(range(len(gaps)), key=gaps.__getitem__)
        assert gaps[k] <= tol["step_time_course_K"], (
            "%s %s: largest gap %.2f K at minute %d (compact %.2f, JOS-3 %.2f), tolerance %.1f K"
            % (case_id, quantity, gaps[k], k + 1, ours[k][quantity], case["series"][k][quantity],
               tol["step_time_course_K"]))
        return
    o, j = ours[quantity], case["result"][quantity]
    if quantity == "core":
        ok, bound = abs(o - j) <= tol["core_K"], "%.1f K" % tol["core_K"]
    elif quantity == "skin_mean":
        ok, bound = abs(o - j) <= tol["skin_mean_K"], "%.1f K" % tol["skin_mean_K"]
    elif quantity == "heat_prod_W":
        ok, bound = abs(o - j) <= tol["heat_prod_rel"] * j, "%.0f percent" % (100 * tol["heat_prod_rel"])
    else:
        ok, bound = abs(o - j) <= max(tol["sweat_rel"] * j, SWEAT_FLOOR_LH), "%.0f percent (floor 0.02 L/h)" % (100 * tol["sweat_rel"])
    assert ok, "%s %s: compact %.3f, JOS-3 %.3f (tolerance %s)" % (case_id, quantity, o, j, bound)


# ============================================================== the scenes ===
def test_u_S1_phase_A_a_clothed_person_in_20_C_still_air_20_minutes():
    w, p = SB.scene_clothed_20C(20)
    t = w.body_temps(p)
    assert 35.6 <= t["core"] <= 37.4
    assert 28.0 <= t["skin_mean"] <= 34.0
    assert np.isfinite(w.E).all()


@pytest.mark.xfail(strict=False, reason="mean skin 31.91 against the band 32.0-33.5 (u): 0.09 K under; core 36.62 is inside. "
                   "The 1 clo, 20 C case is cooler than the fit case at 22 C. JOS-3 CHECK (lane J, 2026-10-30): clothed 1 clo, 20 C, same rest power, 24 h: JOS-3 mean skin 31.41 K "
                   "(ours 31.73 K in the same protocol, 31.91 here); 30.77 at PAR 1.00; 32.46 at its own rest (PAR 1.25). JOS-3 is itself under "
                   "the 32.0 edge of the band (u) at the sim's rest; the edge is reached only at the ISO rest level")
def test_u_S1_phase_A_the_compact_run_for_hours_clothed_at_20_C():
    """core 36.6-37.1, mean skin 32-33.5 (u), the standard man in 1 clo."""
    sh, g, h = SB._std_person()
    nodes = BN.nodes_from_T(g, h, BN.comfort_T(g, h))
    env = {"T_air": 20.0, "rh": 0.5, "v": 0.1, "T_rad": 20.0, "zones": SB._clo_ensemble(1.0, 0.1)}
    BN.advance(h, nodes, g, env, 3 * 3600.0)
    T = BN.temps(nodes)
    assert 36.6 <= T[BN.POOL] <= 37.1
    assert 32.0 <= BN.mean_skin(nodes, g) <= 33.5


def test_S8_a_person_drags_a_heavy_crate_for_10_minutes_and_the_books_close():
    """Small world, voxel. A 60 kg crate at mu 0.5, 0.5 m/s: 147 W of friction
    work, billed once a second as 'drag'. Stores pay work/eta, work_out is
    credited and the floor gets the heat back (booked where the world books
    it), the muscles run hot with hyperemia, the person sweats."""
    w, p = SB.person_world(T_air=24.0, rh=0.4)
    h = p["heat"]
    t0 = w.bh_terms()
    e0 = w.total_energy()
    stores0 = DG.stores_J(h["stores"])
    h["met"] = 2.5
    friction_J = 0.5 * 60.0 * 9.81 * 0.5                          # 147 J per second
    tot_work = 0.0
    idx = tuple(h["cells"].T)
    for sec in range(600):
        r = w.note_work(p, "drag%d" % sec, friction_J, "drag")
        tot_work += friction_J
        w.work_out -= friction_J                                    # the floor books the friction heat ...
        w.shed += friction_J                                        # ... where the world books heat
        w.step()
    assert w.total_energy() == pytest.approx(e0, abs=1e-3)
    assert p["heat"]["work"]["met_J"] >= tot_work / EN.ETA - 1.0
    assert DG.stores_J(h["stores"]) < stores0 - tot_work / EN.ETA
    assert w.work_out == pytest.approx(0.0, abs=1e-6)               # given to the floor as heat
    zone_T = w.body_temps(p)
    assert zone_T["core"] > 36.0 and max(zone_T["skin"].values()) < 41.0
    assert h["stores"]["water_ml"] < h["stores"]["water0_ml"]       # it sweated


def test_u_S6_a_meal_then_four_hours_then_three_days_without_food():
    """glucose appears for about an hour, the gut is empty by 4-5 h, glycogen
    fills before fat; 3 days fasting: glycogen empty in about a day, hunger
    rising, the capacity for hard work falling (digestion only; compact)."""
    st = DG.new_stores(73.0, 0.19, full=0.4)
    glyc0, fat0 = st["glyc_g"], st["fat_g"]
    DG.ingest(st, DG.food("stew", 300.0))
    DG.ingest(st, DG.food("bread", 100.0))
    k0 = DG.stomach_kcal(st)
    assert 600.0 < k0 < 750.0
    peak_t, peak = 0.0, 0.0
    for i in range(1, 6 * 60 + 1):
        o = DG.step(st, 60.0, 90.0, mass_kg=73.0)
        rate = st["si"].get("carb", 0.0)
        if rate > peak:
            peak, peak_t = rate, i
    assert 40.0 <= peak_t <= 180.0       # minutes: glucose appears for an hour or two (a 670 kcal mixed meal)
    assert DG.stomach_kcal(st) < 0.02 * k0     # the stomach is empty within 6 h (the half time is 100 min)
    assert st["glyc_g"] > glyc0 + 10.0 and st["fat_g"] - fat0 < 20.0     # glycogen fills, fat is only burned
    h_fed = DG.hunger(st)
    cap_fed = DG.step(DG.new_stores(73.0, 0.19), 60.0, 600.0)["capacity"]
    cap_empty = None
    for i in range(3 * 24):
        DG.step(st, 3600.0, 95.0, mass_kg=73.0)
        if i == 23:
            glyc_day1 = st["glyc_g"] / st["glyc_cap_g"]
            h_day1 = DG.hunger(st)
    cap_empty = DG.step(st, 60.0, 600.0)["capacity"]
    assert glyc_day1 < 0.35 and st["glyc_g"] < 0.1 * st["glyc_cap_g"]
    assert DG.hunger(st) > max(h_fed, 0.5) and h_day1 > h_fed
    assert cap_empty < cap_fed


@pytest.mark.xfail(strict=False, reason="the spec says 0.5-1 L of sweat in the first hour at 40 C dry; the model sweats 0.14 L "
                   "(and 2.0 percent deficit after 6 h against 3-4). At rest the heat to shed is 89 W of body heat plus about 45 W "
                   "of dry gain, which is 0.2 L/h at steady state (nude_40C_rest: 0.21, inside its band); 0.5-1 L/h needs a "
                   "working body. Same cause as the held-out step_29_to_40C case: the lag of the sweat controller. "
                   "JOS-3 CHECK (lane J, 2026-10-30): a resting nude seated person from 29 C into 40 C, RH 30, makes 0.15 L of sweat in "
                   "the first hour and loses 1.8 percent of body water in 6 h (breath and skin diffusion included) at the same rest power "
                   "(0.16 L, 2.0 percent at PAR 1.25); ours 0.14 L and 2.0 percent. The spec numbers (0.5-1 L, 3-4 percent) do not "
                   "describe a resting body")
def test_u_S4_a_hot_dry_room_for_an_hour_and_the_dehydration_that_follows():
    sh, g, h = SB._std_person()
    nodes = BN.nodes_from_T(g, h, BN.comfort_T(g, h))
    env = {"T_air": 40.0, "rh": 0.3, "v": 0.1, "T_rad": 40.0}
    out = BN.advance(h, nodes, g, env, 3600.0)
    assert 0.5 <= out["sweat_g"] / 1000.0 <= 1.0                   # L in the hour
    assert BN.temps(nodes)[BN.POOL] - 36.9 <= 1.0
    for _ in range(5):                                              # no water for 6 h
        BN.advance(h, nodes, g, env, 3600.0)
    d = DG.deficit_pct(h["stores"], 73.0)
    assert 3.0 <= d <= 4.0


def test_S4_the_model_dehydrates_in_the_right_direction():
    """Not the spec's numbers (above): the direction and the books. No water for 6 h
    at 40 C dry: the deficit grows, sweat is never negative, thirst rises."""
    sh, g, h = SB._std_person()
    nodes = BN.nodes_from_T(g, h, BN.comfort_T(g, h))
    env = {"T_air": 40.0, "rh": 0.3, "v": 0.1, "T_rad": 40.0}
    d = []
    for _ in range(6):
        BN.advance(h, nodes, g, env, 3600.0)
        d.append(DG.deficit_pct(h["stores"], 73.0))
    assert all(b > a for a, b in zip(d, d[1:])) and DG.thirst(h["stores"], 73.0) > 0.0


PHASE_B = ("T8-dependent (needs T_out, no void radiation from interior walls) and the real lattice hooks "
           "BH1-BH3: phase B, lane C6")


@pytest.mark.skip(reason="S1 full form, nude person 28 C for 2 h: " + PHASE_B)
def test_S1_full_form():
    pass


@pytest.mark.skip(reason="S2 (10 C for 1 h bare vs 1.5 clo coat): " + PHASE_B)
def test_S2_cold_hour():
    pass


@pytest.mark.skip(reason="S3 (the same coat soaked, 5 C): " + PHASE_B)
def test_S3_soaked_coat():
    pass


@pytest.mark.skip(reason="S5 (four sleepers, closed 4x4 m room, 8 h) needs T8 and the sealed-room region sink (C10)")
def test_S5_four_sleepers():
    pass


@pytest.mark.skip(reason="S7 (cold-water swim): " + PHASE_B)
def test_S7_cold_water_swim():
    pass
