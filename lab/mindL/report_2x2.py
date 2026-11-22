"""The 2x2 of the delayed-credit test: {all5, all6} x {traces off, traces on}, mind2 as the focal character,
tests L1 and L2 of lab/mindL/report_L.py (unchanged), plus the learned worth of got_kind and got_harm.

    python -m lab.mindL.report_2x2 <runs dir> <out json>

Lifetimes: lab/results/mindL3/mind2x2/runs/L12_<A|B|deaf>_<mind2|mind2t|rules|bandit>_seed<s>_n400_<all5|all6>.json
(mind2 = traces off, mind2t = traces on). The nulls (rules, bandit) were run in all6 only and are shown in both all6 cells.
Windows scale with the life as in round 2 (partners turn deaf at 2/3 of it): n=400 gives cut 266, window 226-266.
"""
import json
import sys
from pathlib import Path

import numpy as np

from lab.mindL import report_L as R1
from lab.mindL.report_L import ci, fmt

SEEDS = tuple(range(1, 11))


def load(runs, world, mind):
    out = {}
    for fp in Path(runs).glob(f"L12_*_n400_{world}.json"):
        r = json.loads(fp.read_text())
        c = r["chooser"]
        if c == mind:
            c = "mind"
        elif c not in ("rules", "bandit") or world != "all6":
            continue
        out[(r["variant"], c, r["seed"])] = r
    return out


def vkh(R, cut):
    """vk, vh of the focal mind at the night before the partners turn deaf, and at the last night; mean over lives A, B."""
    res = {"pre": {}, "end": {}}
    for s in SEEDS:
        for key, night in (("pre", cut - 1), ("end", None)):
            v = []
            for var in ("A", "B"):
                r = R.get((var, "mind", s))
                if r is None:
                    continue
                rows = [x for x in r["nights_rows"] if "vkh" in x]
                row = rows[-1] if night is None else next((x for x in rows if x["n"] == night), None)
                if row:
                    v.append(row["vkh"])
            if v:
                res[key][s] = np.mean(v, axis=0).tolist()
    return {k: {"vk": ci([x[0] for x in d.values()]), "vh": ci([x[1] for x in d.values()]),
                "abs_vk": ci([abs(x[0]) for x in d.values()]), "abs_vh": ci([abs(x[1]) for x in d.values()]),
                "gap_vk_minus_vh": ci([x[0] - x[1] for x in d.values()]), "per_seed": d} for k, d in res.items()}


def diff(a, b):
    """a minus b, paired by seed (dicts seed -> value)."""
    return ci([a[s] - b[s] for s in SEEDS if s in a and s in b and a[s] is not None and b[s] is not None])


def main(runs, out):
    cells = {}
    for world in ("all5", "all6"):
        for name, mind in (("off", "mind2"), ("on", "mind2t")):
            R = load(runs, world, mind)
            if not any(k[1] == "mind" for k in R):
                continue
            n = next(iter(R.values()))["nights"]
            cut = (2 * n) // 3
            r1, r2 = R1.l12(R)
            cells[f"{world}/{name}"] = {
                "n_lives": len(R), "nights": n, "cut": cut,
                "L1": {b: r1["rows"][b].get("mind", {}).get("value") for b in r1["rows"]},
                "L1_nulls": {b: {c: v["value"] for c, v in r1["rows"][b].items() if c != "mind"} for b in r1["rows"]},
                "L1_verdict": r1["verdict"],
                "L2": {k: r2[k].get("mind", {}).get("value") for k in ("effect_kind", "effect_hard", "contrast")},
                "L2_nulls": {k: {c: v["value"] for c, v in r2[k].items() if c != "mind"} for k in ("effect_kind", "effect_hard", "contrast")},
                "L2_verdict": r2["verdict"],
                "L2_internal": r2["mind_internal"]["ci"],
                "vkh": vkh(R, cut),
                "per_seed": {"L1": {b: r1["per_seed"][b].get("mind", {}) for b in r1["per_seed"]},
                             "L2_contrast": r2["per_seed"]["contrast"].get("mind", {}),
                             "L2_kind": r2["per_seed"]["effect_kind"].get("mind", {}),
                             "L2_hard": r2["per_seed"]["effect_hard"].get("mind", {})}}
    # which arm moves it: paired differences between cells (same seed, same focal, same worlds draws within a world)
    arms = {}
    for label, (a, b) in {"traces in all5 (on-off)": ("all5/on", "all5/off"), "traces in all6 (on-off)": ("all6/on", "all6/off"),
                          "world, traces off (all6-all5)": ("all6/off", "all5/off"),
                          "world, traces on (all6-all5)": ("all6/on", "all5/on"),
                          "both (all6/on - all5/off)": ("all6/on", "all5/off")}.items():
        if a not in cells or b not in cells:
            continue
        A, B = cells[a]["per_seed"], cells[b]["per_seed"]
        arms[label] = {"L1 late1": diff(A["L1"]["late1"], B["L1"]["late1"]), "L1 late2": diff(A["L1"]["late2"], B["L1"]["late2"]),
                       "L2 contrast": diff(A["L2_contrast"], B["L2_contrast"]),
                       "gap pre-deaf": diff({s: v[0] - v[1] for s, v in cells[a]["vkh"]["pre"]["per_seed"].items()},
                                            {s: v[0] - v[1] for s, v in cells[b]["vkh"]["pre"]["per_seed"].items()})}
    Path(out).write_text(json.dumps({"cells": cells, "arms": arms}, indent=1))
    for k, c in cells.items():
        print(f"\n== {k}  ({c['n_lives']} lives, {c['nights']} nights, deaf at {c['cut']})")
        for b in ("late1", "late2", "shared_early", "shared_late"):
            print(f"  L1 {b:13s} mind {fmt(c['L1'][b])}")
        for kk in ("effect_kind", "effect_hard", "contrast"):
            print(f"  L2 {kk:13s} mind {fmt(c['L2'][kk])}")
        for when in ("pre", "end"):
            v = c["vkh"][when]
            print(f"  night {'pre-deaf' if when == 'pre' else 'last'}: vk {fmt(v['vk'], 4)}  vh {fmt(v['vh'], 4)}  gap {fmt(v['gap_vk_minus_vh'], 4)}  |vk| {fmt(v['abs_vk'], 4)} |vh| {fmt(v['abs_vh'], 4)}")
        print("  internal [p_harm gap, p_kind gap, valence kind-hard, p_harm vs deaf]:", [fmt(x) for x in c["L2_internal"]])
        print("  nulls L2 contrast:", {n: fmt(v) for n, v in c["L2_nulls"]["contrast"].items()})
        print("  nulls L1 late1:", {n: fmt(v) for n, v in c["L1_nulls"]["late1"].items()})
    for k, a in arms.items():
        print("\n", k, {m: fmt(v) for m, v in a.items()})


if __name__ == "__main__":
    main(*sys.argv[1:3])
