"""Step 3 of SV5 (2026-11-21): L1 and L2 for the arms of the concerns step, and the diagnosis table. Reported, not gated.

    lab/.venv/bin/python -m lab.mindL.step3_report OUT.json ARM=DIR:CHOOSER [ARM=DIR:CHOOSER ...]

DIR holds lifetimes L12_<A|B|deaf>_<CHOOSER>_seed<s>_n<N>_all6s_need.json (the stored step 2 runs have n 300 and chooser mind2e (effects),
mind2 (labels), rules; the step 3 runs have n 220 and chooser = the arm name). The partners go deaf at night 200 in all of them, so the
blocks are fixed: late1 160-180, late2 180-200, shared_early 200-220; L2 compares nights 160-200 of the A and B lives with the deaf life
(the same definitions as lab/mindL/report_L.py l12, but the cut is 200 whatever the length of the life). 95 percent t intervals over seeds
1-10, paired by seed.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

from lab.mindL.report_L import ci, fmt, treat

SEEDS = tuple(range(1, 11))
CUT = 200
WORLD = os.environ.get("STEP3_WORLD", "all6s")      # all6s or all6h
BLOCKS = {"late1": (CUT - 40, CUT - 20), "late2": (CUT - 20, CUT), "shared_early": (CUT, CUT + 20)}
LATE = (CUT - 40, CUT)


def load(d, chooser):
    out = {}
    for s in SEEDS:
        for v in ("A", "B", "deaf"):
            for n in (220, 300):
                fp = Path(d) / f"L12_{v}_{chooser}_seed{s}_n{n}_{WORLD}_need.json"
                if fp.exists():
                    out[(v, s)] = json.loads(fp.read_text())
                    break
    return out


def l12(R):
    L1 = {b: {} for b in BLOCKS}
    k, h, c = {}, {}, {}
    for s in SEEDS:
        A, B, D = R.get(("A", s)), R.get(("B", s)), R.get(("deaf", s))
        if A is None or B is None:
            continue
        for b, (a0, a1) in BLOCKS.items():
            L1[b][s] = 0.5 * ((treat(A, "Fenn", a0, a1) - treat(B, "Fenn", a0, a1)) + (treat(B, "Oren", a0, a1) - treat(A, "Oren", a0, a1)))
        if D is None:
            continue
        ek, eh = [], []
        for v, run in (("A", A), ("B", B)):
            kp, hp = ("Fenn", "Oren") if v == "A" else ("Oren", "Fenn")
            ek.append(treat(run, kp, *LATE) - treat(D, kp, *LATE))
            eh.append(treat(run, hp, *LATE) - treat(D, hp, *LATE))
        k[s], h[s], c[s] = float(np.mean(ek)), float(np.mean(eh)), float(np.mean(ek) - np.mean(eh))
    return {"L1": L1, "L2": {"effect_kind": k, "effect_hard": h, "contrast": c}}


def paired(a, b):
    return ci([a[s] - b[s] for s in SEEDS if s in a and s in b])


def main(out, specs):
    res, per = {}, {}
    for spec in specs:
        arm, rest = spec.split("=", 1)
        d, ch = rest.rsplit(":", 1)
        R = load(d, ch)
        r = l12(R)
        per[arm] = r
        res[arm] = {"lives": len(R), "L1": {b: ci(list(v.values())) for b, v in r["L1"].items()},
                    "L2": {k: ci(list(v.values())) for k, v in r["L2"].items()}}
    Path(out).write_text(json.dumps({"cells": res, "per_seed": per}, indent=1, default=float))
    for arm, c in res.items():
        print(f"== {arm} ({c['lives']} lives)")
        for b, v in c["L1"].items():
            print(f"  L1 {b:13s} {fmt(v)}")
        for k, v in c["L2"].items():
            print(f"  L2 {k:13s} {fmt(v)}")
    names = list(per)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            print(f"-- {a} minus {b} (paired by seed)")
            for blk in BLOCKS:
                print(f"  L1 {blk:13s} {fmt(paired(per[a]['L1'][blk], per[b]['L1'][blk]))}")
            print(f"  L2 contrast   {fmt(paired(per[a]['L2']['contrast'], per[b]['L2']['contrast']))}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
