"""JOS-3 ORACLE: write tests/data/jos3_cases.json from the published model.

JOS-3 (Takahashi et al. 2021, "Thermoregulation Model JOS-3 with New Open Source
Code", Energy and Buildings 231) is the 85-node open model this body law's
compact form is checked against: steady state core +-0.3 K, mean skin +-1.0 K,
total heat production +-15 percent, sweat +-30 percent, and the cold step
(28 -> 10 C) and hot step (28 -> 45 C) time courses +-0.5 K (spec T7 test 10a).

THIS SCRIPT NEVER INSTALLS ANYTHING. Decision D8 (approved 2026-10-30): JOS-3 is
installed ONCE in a throwaway venv, and only the JSON it makes comes into the repo:

    python3 -m venv /tmp/jos3venv && /tmp/jos3venv/bin/pip install jos3==0.5.0 numpy
    /tmp/jos3venv/bin/python -I tools/jos3_oracle.py

(jos3 0.5.0, MIT, on PyPI; tests/data/sourced_numbers.json "jos3.package". The
package declares no dependencies; it needs numpy.) Without the package the script
prints that and exits 2.

    python3 tools/jos3_oracle.py [--out PATH] [--check PATH]

THE KEY MAPPING (read from the installed 0.5.0 source, lane J, 2026-10-30; the
JSON repeats it with the package's own words and units):
    core         <- Tcb     central blood [C]. Our core is the central blood POOL.
                    (extra output; needs ex_output). The segment cores are 0.2-0.4 K
                    warmer than Tcb in JOS-3 and 0.2 K warmer than our pool, so the
                    alternatives are kept: core_head (TcrHead, the hypothalamus
                    stand-in that drives its sweat), core_pelvis (TcrPelvis).
    skin_mean    <- TskMean mean skin temperature [C], weights = JOS-3's segment areas.
    heat_prod_W  <- Met     total heat production of the whole body [W]: basal + work
                    (PAR) + shivering + non-shivering. NOT THLsk (that is the loss at
                    the skin, per segment).
    sweat_Lh     <- sum over the 17 segments of Esweat [W] / 2418 J/g * 3.6 -> L/h.
                    Esweat is the evaporative heat loss "by only sweating" AFTER the
                    wettedness cap (extra output). The package keys Wet* are skin
                    WETTEDNESS (0.06-1, no unit), Esk* include the 6 percent diffusion,
                    Wle is a mass rate [g/s] that also holds breath: none of them is
                    the sweat rate. JOS-3 has no run-off, so while max(Wet*) < 1 this
                    equals what the controller asks for (our sweat_Lh is the secretion).
The first draft of this tool read "Esk", "SkinWet", "Wet" for sweat (none of them
exists as a plain key, so sweat came out empty) and the pelvis core for core.

STEADY STATE: JOS-3 is slow in the cold (its heat production at 20 C is 152.6 W after 3 h, 160.8 W after 12 h, 160.9 W after
24 h), so the steady cases and the time before a step are 24 h long, settled in 10-minute implicit steps (same steady state as
1-minute steps: checked). The compact model is steady within 6 h.

THE PERSON: the reference man of the T7 spec (73 kg, 1.75 m, 30 y, male). Fat 19.3 percent
is our Sheet's Deurenberg value (JOS-3 only uses it to pick a conductance table: 17.5-22.5
percent; its default 15 picks the 12.5-17.5 one). BSA (DuBois) is 1.881 m2 in both.

THE REST LEVEL (PAR, JOS-3's physical activity ratio = metabolic rate / basal rate). The
sim's rest is 1 MET = 1.22 W/kg (spec F1) = 89.06 W for 73 kg = 47.3 W/m2. JOS-3's basal rate
for this man is 85.03 W (Harris-Benedict), so
  met1     PAR = 89.06 / 85.03 = 1.047   THE PRIMARY CASES: the same rest power as the sim,
                                          so the comparison is about the heat exchange and the
                                          controllers, not about the rest constant;
  par1.00  PAR = 1.0                      basal only (the first draft's choice);
  par1.25  PAR = 1.25                     JOS-3's own default and the level its set points are
                                          found at (56.5 W/m2, close to the ISO 58.15): the
                                          evidence for decision N2.
The two other levels are written as "variants" so decision N2 can be judged without a second
run. Postures: the primary cases are standing, fully exposed (what the existing test runs on the
compact side); the human-chamber conditions are SEATED (JOS-3 posture "sitting").
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "tests", "data", "jos3_cases.json")

# ---- the person (the reference man of the T7 spec) and the cases ---------------------
PERSON = {"height": 1.75, "weight": 73.0, "age": 30, "sex": "male", "fat": 19.3}
MET_W_PER_KG = 1.22            # the sim's MET (ACSM), energetics.MET_W_PER_KG; this tool imports no src/
LATENT_J_PER_G = 2418.0        # JOS-3's own latent heat (jos3.py: wlesk = (e_sweat + ...) / 2418)
SEGMENTS = ("Head", "Neck", "Chest", "Back", "Pelvis", "LShoulder", "LArm", "LHand", "RShoulder",
            "RArm", "RHand", "LThigh", "LLeg", "LFoot", "RThigh", "RLeg", "RFoot")
# 1 clo on the trunk, upper arms, forearms, thighs and shins; none on head, neck, hands, feet
# (the zones scenes_bodyheat._clo_ensemble covers; neck is part of our head zone)
ICL_1CLO = (0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0)

# JOS-3 is slow in the cold: at 20 C nude its heat production is 152.6 W after 3 h, 159.1 W after 6 h, 160.8 W after 12 h and
# 160.9 W after 24 h (lane J, 2026-10-30; w13/jos3/evidence_converge_jos3.txt). A "steady state" is therefore 24 h here.
# The compact model is steady within 3 h (185.3 W at 3 h, 185.2 W at 6 h), so the sandbox test runs it for 6 h.
STEADY_H = 24.0
# The settle is run in 10-minute steps: JOS-3 is an implicit (backward Euler) scheme, so its steady state does not depend on the step.
# Checked (w13/jos3/evidence_dt600.txt): six steady states and a following cold step are equal to 0.0001 K and 0.001 W against 1-minute
# steps. Everything that is reported (the last minute of a steady state; the minutes after a step) is in 1-minute steps.
SETTLE_DT = 600.0
STEADY = [  # (name, Ta C, RH percent, wind m/s, hours): the T7 spec cases
    ("steady_20C", 20.0, 40.0, 0.1, STEADY_H),
    ("steady_28C", 28.0, 40.0, 0.1, STEADY_H),
    ("steady_35C", 35.0, 40.0, 0.1, STEADY_H),
]
STEPS = [  # (name, Ta before, Ta after, RH, minutes after); 120 min at Ta before
    ("cold_step_28_to_10", 28.0, 10.0, 40.0, 60),
    ("hot_step_28_to_45", 28.0, 45.0, 40.0, 60),
]
BEFORE_MIN = int(STEADY_H * 60)
EXTRA_STEADY = [  # beyond the spec's three: the middle of the neutral zone, where the compact core dips (lane J sweep)
    ("zone_31C", 31.0, 40.0, 0.1, STEADY_H),
]
# the conditions of the human-chamber fixtures (tests/data/human_cases.json), SEATED, nude unless
# a garment is named; written as variants so the human-data cases can be compared with JOS-3 too
HUMAN_STEADY = [  # (name, Ta, RH, v, posture, clo, hours)
    ("H_nude_22C_seated", 22.0, 50.0, 0.1, "sitting", 0.0, STEADY_H),
    ("H_nude_29C_seated", 29.0, 50.0, 0.1, "sitting", 0.0, STEADY_H),
    ("H_nude_35C_seated", 35.0, 50.0, 0.1, "sitting", 0.0, STEADY_H),
    ("H_nude_40C_dry_seated", 40.0, 30.0, 0.1, "sitting", 0.0, STEADY_H),
    ("H_clothed_1clo_22C_seated", 22.0, 50.0, 0.1, "sitting", 1.0, STEADY_H),
    ("S1_clothed_1clo_20C_standing", 20.0, 50.0, 0.1, "standing", 1.0, STEADY_H),
]
HUMAN_STEPS = [  # (name, T0, T1, RH, minutes, posture)
    ("H_step_29_to_40C_seated", 29.0, 40.0, 40.0, 60, "sitting"),
    ("H_step_29_to_15C_seated", 29.0, 15.0, 50.0, 30, "sitting"),
]
LEVELS = (("met1", None), ("par1.00", 1.0), ("par1.25", 1.25))      # None = the sim's 1 MET
TOLERANCES = {"core_K": 0.3, "skin_mean_K": 1.0, "heat_prod_rel": 0.15, "sweat_rel": 0.30,
              "step_time_course_K": 0.5}

# our name -> (JOS-3 key, per segment?, how it is turned into our unit)
KEYS = {
    "core": ("Tcb", False, "value, C"),
    "skin_mean": ("TskMean", False, "value, C"),
    "heat_prod_W": ("Met", False, "value, W"),
    "sweat_Lh": ("Esweat", True, "sum over 17 segments, W / 2418 J/g * 3600 s/h / 1000 g/L"),
}
# kept beside the four (not our names; for the reader who prefers another core)
EXTRAS = {
    "core_head": ("TcrHead", False), "core_pelvis": ("TcrPelvis", False), "core_chest": ("TcrChest", False),
    "res_W": ("RES", False), "wet_mean": ("WetMean", False),
}
EX_OUTPUT = ["Tcb", "Esweat", "Mshiv", "Mnst"]      # the extra outputs this tool needs

INSTALL_HELP = (
    "THE JOS-3 ORACLE NEEDS THE PACKAGE (decision D8, a throwaway venv; only the JSON comes into the repo):\n"
    "    python3 -m venv /tmp/jos3venv && /tmp/jos3venv/bin/pip install jos3==0.5.0 numpy\n"
    "    /tmp/jos3venv/bin/python -I tools/jos3_oracle.py\n")


# ---- reading the results --------------------------------------------------------------
def _need(res, key):
    if key not in res:
        raise KeyError("JOS-3 result key %r is missing; the keys are %s" % (key, sorted(res)[:40]))
    return res[key]


def _row(res, i):
    """One minute of history, in our names and units."""
    out = {}
    for ours, (key, per_seg, _how) in KEYS.items():
        if per_seg:
            s = sum(float(_need(res, key + sg)[i]) for sg in SEGMENTS)
            out[ours] = s / LATENT_J_PER_G * 3600.0 / 1000.0
        else:
            out[ours] = float(_need(res, key)[i])
    out["sweat"] = out["sweat_Lh"]                 # the first draft's name, same number (L/h)
    for ours, (key, per_seg) in EXTRAS.items():
        out[ours] = float(_need(res, key)[i])
    out["shiver_W"] = sum(float(_need(res, "Mshiv" + sg)[i]) for sg in SEGMENTS)
    out["nonshiver_W"] = sum(float(_need(res, "Mnst" + sg)[i]) for sg in SEGMENTS)
    out["wet_max"] = max(float(_need(res, "Wet" + sg)[i]) for sg in SEGMENTS)
    return {k: round(v, 5) for k, v in out.items()}


def _model(jos3, posture, level, RH, va, clo):
    m = jos3.JOS3(ex_output=EX_OUTPUT, **PERSON)
    m.posture = posture
    if level is None:                               # the sim's 1 MET
        basal_w = float(m.BMR * m.BSA.sum())
        m.PAR = MET_W_PER_KG * PERSON["weight"] / basal_w
    else:
        m.PAR = level
    m.RH = RH
    m.Va = va
    m.Icl = list(ICL_1CLO) if clo > 0 else 0.0     # setter takes a number or 17 values
    return m


def _rows(m):
    res = m.dict_results()
    n = len(res["CycleTime"])
    return [_row(res, i) for i in range(n)]       # row 0 is the constructor's own set-point run


def _config(posture, lname, level, clo, m):
    return {"posture": posture, "level": lname, "PAR": round(float(m.PAR), 5),
            "rest_power_W_at_PAR": round(float(m.PAR * m.BMR * m.BSA.sum()), 3),
            "Icl_clo": clo, "Va": 0.1}


def steady_case(jos3, name, Ta, RH, va, hours, posture, lname, level, clo=0.0):
    m = _model(jos3, posture, level, RH, va, clo)
    m.Ta = m.Tr = Ta
    n = int(round(hours * 3600.0 / SETTLE_DT))
    m.simulate(times=n, dtime=SETTLE_DT)
    m.simulate(times=1, dtime=60)                  # one more minute in the usual step: the minute that is reported
    rows = _rows(m)                                # row 0 = the constructor, row i = i x SETTLE_DT, last row = n x SETTLE_DT + 60 s
    last = rows[-1]
    per_hour = int(round(3600.0 / SETTLE_DT))
    qs = ("core", "skin_mean", "heat_prod_W", "sweat_Lh")
    trace = [dict(hour=h, **{q: rows[h * per_hour][q] for q in qs}) for h in (1, 3, 6, 12) if h < hours]
    trace.append(dict(hour=hours, **{q: rows[n][q] for q in qs}))
    drift = {q: round(rows[n][q] - rows[n - per_hour][q], 5) for q in qs}      # over the last hour of the settle
    return {"id": name, "kind": "steady", "inputs": {"Ta": Ta, "RH": RH, "v": va, "hours": hours},
            "config": _config(posture, lname, level, clo, m), "result": last,
            "trace": trace, "drift_last_hour": drift}


def step_case(jos3, name, T0, T1, RH, minutes, posture, lname, level):
    m = _model(jos3, posture, level, RH, 0.1, 0.0)
    m.Ta = m.Tr = T0
    n = int(round(BEFORE_MIN * 60.0 / SETTLE_DT))
    m.simulate(times=n, dtime=SETTLE_DT)
    m.Ta = m.Tr = T1
    m.simulate(times=minutes, dtime=60)
    rows = _rows(m)                                # rows 1..n = the settle, rows n+1.. = minutes after the step
    pre = rows[n]
    series = []
    for k in range(1, minutes + 1):
        r = dict(rows[n + k])
        r["minute"] = k
        series.append(r)
    return {"id": name, "kind": "step", "inputs": {"T0": T0, "T1": T1, "RH": RH, "minutes": minutes,
                                                    "before_min": BEFORE_MIN},
            "config": _config(posture, lname, level, 0.0, m), "pre": pre, "series": series}


def _mapping_doc(jos3):
    """Our name -> the JOS-3 key, its meaning and unit in the package's own words."""
    import importlib
    docs = importlib.import_module("jos3.params").ALL_OUT_PARAMS
    out = {}
    for ours, (key, per_seg, how) in KEYS.items():
        d = docs.get(key, {})
        out[ours] = {"jos3_key": key, "per_segment_17": per_seg, "jos3_meaning": d.get("meaning"),
                     "jos3_unit": d.get("unit"), "how": how}
    for ours, (key, per_seg) in EXTRAS.items():
        d = docs.get(key, {})
        out[ours] = {"jos3_key": key, "per_segment_17": per_seg, "jos3_meaning": d.get("meaning"),
                     "jos3_unit": d.get("unit"), "how": "value (alternative, not our core)"}
    return out


def _reference_state(jos3):
    """JOS-3's own neutral state (found by its constructor at PAR 1.25, nude, 50 percent RH, 0.1 m/s)."""
    m = jos3.JOS3(ex_output=EX_OUTPUT, **PERSON)
    r = _rows(m)[0]
    return {"Ta_C": round(float(m.Ta[0]), 4), "PAR": float(m.PAR), "posture": m.posture,
            "BMR_W": round(float(m.BMR * m.BSA.sum()), 3), "BMR_W_per_m2": round(float(m.BMR), 3),
            "BSA_m2": round(float(m.BSA.sum()), 4), "row": r}


def run(out_path=OUT):
    try:
        import jos3
    except ImportError:
        sys.stderr.write(INSTALL_HELP + "jos3 is not installed: nothing written.\n")
        return 2
    import numpy
    cases, variants, extra_cases = [], [], []
    for lname, level in LEVELS:
        target = cases if lname == "met1" else variants
        for name, Ta, RH, v, hours in STEADY:
            c = steady_case(jos3, name if lname == "met1" else name + "@" + lname, Ta, RH, v, hours,
                            "standing", lname, level)
            target.append(c)
        for name, Ta, RH, v, hours in EXTRA_STEADY:
            c = steady_case(jos3, name if lname == "met1" else name + "@" + lname, Ta, RH, v, hours,
                            "standing", lname, level)
            (extra_cases if lname == "met1" else variants).append(c)
        for name, T0, T1, RH, minutes in STEPS:
            c = step_case(jos3, name if lname == "met1" else name + "@" + lname, T0, T1, RH, minutes,
                          "standing", lname, level)
            target.append(c)
        for name, Ta, RH, v, posture, clo, hours in HUMAN_STEADY:
            variants.append(steady_case(jos3, name + "@" + lname, Ta, RH, v, hours, posture, lname, level, clo))
        for name, T0, T1, RH, minutes, posture in HUMAN_STEPS:
            variants.append(step_case(jos3, name + "@" + lname, T0, T1, RH, minutes, posture, lname, level))
    try:
        with open(os.path.realpath(__file__), "rb") as f:
            me = hashlib.sha256(f.read()).hexdigest()
    except OSError:
        me = None
    doc = {"_meta": {
        "source": "JOS-3 " + str(getattr(jos3, "__version__", "?")),
        "package": {"jos3": getattr(jos3, "__version__", "?"), "numpy": numpy.__version__,
                    "python": sys.version.split()[0],
                    "installed_from": "PyPI wheel jos3-0.5.0-py3-none-any.whl in a throwaway venv (decision D8)"},
        "oracle_script_sha256": me,
        "person": PERSON,
        "protocol": ("steady: 24 h (JOS-3 needs 12-24 h in the cold) at the given air = radiant temperature, the settle in 10 minute steps and "
                     "the reported minute in a 1 minute step; step: 24 h at the first temperature, then the second for the minutes given in "
                     "1 minute steps; nude unless "
                     "Icl_clo > 0; wind 0.1 m/s; the sim's rest power unless the level says otherwise (see the "
                     "docstring of tools/jos3_oracle.py)"),
        "key_mapping": _mapping_doc(jos3),
        "jos3_reference_state": _reference_state(jos3),
        "latent_heat_J_per_g": LATENT_J_PER_G,
        "segments_17": list(SEGMENTS),
        "levels": {n: ("the sim's 1 MET = %.2f W/kg" % MET_W_PER_KG if lv is None else "PAR %.2f" % lv)
                   for n, lv in LEVELS},
        "tolerances": dict(TOLERANCES),
        "notes": [
            "cases = the T7 spec cases at the sim's own rest power (what the existing test reads); "
            "extra_cases = more steady states at the same rest power (31 C, the middle of the neutral zone); "
            "variants = the same cases at PAR 1.00 and PAR 1.25, and the human-chamber conditions "
            "(seated, clothed) at all three levels",
            "core is the central blood pool Tcb; the segment cores (core_head, core_pelvis, core_chest) are "
            "kept beside it",
            "sweat_Lh is the evaporated sweat equivalent (Esweat); JOS-3 has no run-off; it equals the "
            "controller's demand while wet_max < 1",
        ]},
        "cases": cases, "extra_cases": extra_cases, "variants": variants}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=1)
    print("wrote", out_path, "with", len(cases), "cases,", len(extra_cases), "extra cases and", len(variants), "variants")
    return 0


def check(path=OUT):
    """True if a written file has the shape the fixtures read."""
    with open(path) as f:
        d = json.load(f)
    ids = {c["id"] for c in d["cases"]}
    km = d["_meta"].get("key_mapping", {})
    return (all(n in ids for n, *_ in STEADY + STEPS) and "tolerances" in d["_meta"]
            and all(k in km for k in KEYS) and all(
                d_["result"].get(q) is not None for d_ in d["cases"] if d_["kind"] == "steady"
                for q in KEYS))


def main(argv):
    out = OUT
    if "--out" in argv:
        out = argv[argv.index("--out") + 1]
    if "--check" in argv:
        p = argv[argv.index("--check") + 1]
        ok = check(p)
        print("ok" if ok else "bad shape")
        return 0 if ok else 1
    return run(out)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
