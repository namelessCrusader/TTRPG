"""Step 2 of SV5 (2026-11-21): L1 and L2 with the effects arm, the labels control arm and the rules null, world all6s, Fenn = P-need(0.9),
Oren = P-hard(0.9). Reported, not gated.

    lab/.venv/bin/python -m lab.mindL.step2_report RUNS_DIR OUT.json

Lifetimes: L12_<A|B|deaf>_<mind2e|mind2|rules>_seed<s>_n<N>_all6s_need.json (mind2e = effects arm, mind2 = the labels control arm: the mind
with the 2026-10-29 fixes as today, rules = the null). Tests L1 and L2 as in lab/mindL/report_L.py (unchanged), paired by seed, 95 percent t
intervals over seeds 1-10. Also the mean drive error of the focal over the last 100 nights, per arm and paired.
"""
import json
import sys
from pathlib import Path

import numpy as np

from lab.mindL import report_L as R1
from lab.mindL.report_L import ci, fmt

SEEDS = tuple(range(1, 11))
ARMS = {"effects": "mind2e", "labels": "mind2"}


def load(runs, arm):
    out = {}
    for fp in Path(runs).glob("L12_*_all6s_need.json"):
        r = json.loads(fp.read_text())
        c = r["chooser"]
        if c == ARMS[arm]:
            c = "mind"
        elif c != "rules":
            continue
        out[(r["variant"], c, r["seed"])] = r
    return out


def diff(a, b):
    return ci([a[s] - b[s] for s in SEEDS if s in a and s in b and a[s] is not None and b[s] is not None])


def main(runs, out):
    cells, per = {}, {}
    for arm in ARMS:
        R = load(runs, arm)
        if not any(k[1] == "mind" for k in R):
            continue
        n = next(iter(R.values()))["nights"]
        r1, r2 = R1.l12(R)
        cells[arm] = {"nights": n, "cut": (2 * n) // 3, "lives": len(R),
                      "L1": {b: {c: v["value"] for c, v in r1["rows"][b].items()} for b in r1["rows"]},
                      "L2": {k: {c: v["value"] for c, v in r2[k].items()} for k in ("effect_kind", "effect_hard", "contrast")},
                      "L1_verdict": r1["verdict"], "L2_verdict": r2["verdict"]}
        err = {}
        for (v, c, s), r in R.items():
            e = r["drive_error"]
            err.setdefault(c, {}).setdefault(v, {})[s] = float(np.mean(e[-100:])) if len(e) >= 100 else float(np.mean(e))
        cells[arm]["error_last100"] = {c: {v: ci(list(d.values())) for v, d in vs.items()} for c, vs in err.items()}
        per[arm] = {"L1": {b: r1["per_seed"][b].get("mind", {}) for b in r1["per_seed"]},
                    "L1_rules": {b: r1["per_seed"][b].get("rules", {}) for b in r1["per_seed"]},
                    "L2": {k: r2["per_seed"][k].get("mind", {}) for k in ("effect_kind", "effect_hard", "contrast")},
                    "L2_rules": {k: r2["per_seed"][k].get("rules", {}) for k in ("effect_kind", "effect_hard", "contrast")},
                    "err": err.get("mind", {})}
        # what the effects arm believes: worth of acts at each partner, last night before the partners turn deaf
        if arm == "effects":
            fx = {}
            for (v, c, s), r in R.items():
                if c == "mind" and v in ("A", "B"):
                    row = next((x for x in r["nights_rows"] if x["n"] == (2 * n) // 3 - 1 and "fx" in x), None)
                    if row:
                        kp, hp = ("Fenn", "Oren") if v == "A" else ("Oren", "Fenn")
                        for who, role in ((kp, "P-need partner"), (hp, "P-hard partner")):
                            for a, val in row["fx"][who].items():
                                fx.setdefault(f"{role}: worth of {a}", {}).setdefault(s, []).append(val)
            cells[arm]["fx_worth"] = {k: ci([float(np.mean(v)) for v in d.values()]) for k, d in sorted(fx.items())}
    arms = {}
    if "effects" in per and "labels" in per:
        A, B = per["effects"], per["labels"]
        arms = {"L1 late1": diff(A["L1"]["late1"], B["L1"]["late1"]), "L1 late2": diff(A["L1"]["late2"], B["L1"]["late2"]),
                "L1 shared_early": diff(A["L1"]["shared_early"], B["L1"]["shared_early"]),
                "L2 effect_kind": diff(A["L2"]["effect_kind"], B["L2"]["effect_kind"]),
                "L2 effect_hard": diff(A["L2"]["effect_hard"], B["L2"]["effect_hard"]),
                "L2 contrast": diff(A["L2"]["contrast"], B["L2"]["contrast"])}
        for v in ("A", "B", "deaf"):
            arms[f"error last 100, life {v}"] = diff(A["err"].get(v, {}), B["err"].get(v, {}))
    Path(out).write_text(json.dumps({"cells": cells, "effects_minus_labels": arms, "per_seed": per}, indent=1))
    for arm, c in cells.items():
        print(f"\n== {arm}  ({c['lives']} lives incl. rules, {c['nights']} nights, deaf at {c['cut']})")
        for b in ("late1", "late2", "shared_early", "shared_late"):
            print(f"  L1 {b:13s}", "  ".join(f"{k}: {fmt(v)}" for k, v in c["L1"][b].items() if k in ("mind", "rules")))
        for kk in ("effect_kind", "effect_hard", "contrast"):
            print(f"  L2 {kk:13s}", "  ".join(f"{k}: {fmt(v)}" for k, v in c["L2"][kk].items() if k in ("mind", "rules")))
        print("  verdicts:", c["L1_verdict"], "|", c["L2_verdict"])
        for who, d in c["error_last100"].items():
            print(f"  drive error, last 100 nights, {who}:", {v: fmt(x) for v, x in d.items()})
        for k, v in c.get("fx_worth", {}).items():
            print(f"  effects arm, {k}: {fmt(v, 4)}")
    print("\neffects minus labels (paired by seed):")
    for k, v in arms.items():
        print(f"  {k:28s} {fmt(v)}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
