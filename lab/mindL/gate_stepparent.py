"""mindL6 stage B1, item 5: THE STEPPARENT TEST (PASS_RULE_B1.txt section C, written before this ran; library answer D-f, L9).
A controlled coarse past, gate_kin's method, no world run: a child C (18 at the check), a biological parent P1 (together since birth) and a stepparent S who joins
y years before the check (y = 0.25 for 'about 0', 5, 17). The SAME exposure for both parents: only the duration cues differ. 30 draws.
Measure: w(C to S) / w(C to P1), and the same for the kin estimate khat.   lab/.venv/bin/python -m lab.mindL.gate_stepparent"""
import json
import math
from pathlib import Path

import numpy as np
import scipy.stats as st

from lab.mindL import household as H
from lab.mindL.gate_kin import SCR, ci, uniform

RES = Path(__file__).resolve().parents[1] / "results" / "mindL6" / "B1"
N = 30
TARGETS = {0.25: (0.43, 0.43), 5.0: (0.51, 0.47), 17.0: (0.72, 0.59)}


def run(E, seed, y, params=None):
    people = ["P1", "S", "C"]
    age = dict(P1=44, S=44, C=18)
    birth = {n: -float(a) for n, a in age.items()}
    spans = dict(P1=[(-18.0, 0.0)], S=[(-y, 0.0)], C=[(-18.0, 0.0)])
    U = uniform(E)

    def fn(i, j, t):
        e = {x: U[x] for x in H.EXPO_KEYS}
        e["feat"] = dict(U["feat"])
        e.update(res=1.0, ej=0.0, obs_by={}, obs=0.0)
        return e
    states, rows = H.prehistory(seed, None, spans=spans, birth=birth, t0=-18.0, t1=0.0, who=people, expo_fn=fn, noise=True, params=params)
    return states


def main(params=None, tag="b1"):
    E = H.merge_expo([json.loads((SCR / f"sample_{s}.json").read_text())["tele"] for s in (101, 102, 103)])
    out = {}
    for y, (tp, te) in TARGETS.items():
        rs, ks = [], []
        for s in range(1, N + 1):
            r = run(E, s, y, params)
            wp, ws = r["C"]["P1"]["w"], r["C"]["S"]["w"] if "S" in r["C"] else 0.0
            rs.append(ws / wp if wp > 0 else float("nan"))
            ks.append(r["C"]["S"]["khat"] / r["C"]["P1"]["khat"] if "S" in r["C"] and r["C"]["P1"]["khat"] > 0 else float("nan"))
        out[str(y)] = dict(ratio_w=ci(rs), ratio_khat=ci(ks), target_practical=tp, target_emotional=te)
    r17 = out["17.0"]["ratio_w"][0]
    out["PASS_practical"] = bool(0.62 <= r17 <= 0.82)
    out["PASS_emotional"] = bool(0.49 <= r17 <= 0.69)
    out["FAIL_large"] = bool(r17 > 0.90)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"stepparent_{tag}.json").write_text(json.dumps(out, indent=1, default=float))
    for k, v in out.items():
        print(k, v)
    return out


if __name__ == "__main__":
    main()
