"""The hard inn world check (2026-11-21, lane A): the step 1 check re-run in all6h, next to the stored all6s numbers.

    lab/.venv/bin/python -m lab.mindL.hardinn_report RUNS_NEED_S RUNS_KIND_S RUNS_NEED_H RUNS_KIND_H OUT.json

S = the stored step 1 runs (all6s with P-need; all6 with P-kind), H = the same in all6h (all6 + signs + hard). Every number is the
step 1 report's (lab/mindL/step1_report.py), plus the gossip mix: the share of tellings whose deed had a negative sign (the inn's
DEED table), per seed, 95% t interval, against the field's 35.9 percent negative (Dores Cruz et al., Library request S2). Read only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from lab.mindL import step1_report as S1


def gossip(R):
    """Per chooser: the negative share of tellings per seed (CI), and tellings a night."""
    out = {}
    for c in S1.CH:
        sh, rate, good, bad = [], [], 0, 0
        for s in range(1, 11):
            r = R.get((c, s))
            if not r:
                continue
            g, b = r["gossip_effect"]["good"][0], r["gossip_effect"]["bad"][0]
            n = len(r["err"]) - r["win0"]
            good, bad = good + g, bad + b
            if g + b:
                sh.append(b / (g + b))
            rate.append((g + b) / n)
        out[c] = dict(share_negative=S1.ci(sh), tellings_a_night=S1.ci(rate), good=good, bad=bad, pooled=bad / max(1, good + bad))
    return out


def main():
    d = {}
    for path, label in zip(sys.argv[1:5], ("S P-need, all6s", "S P-kind, all6", "H P-need, all6h", "H P-kind, all6h")):
        R = S1.load(path)
        d[label] = S1.setting(R, label)
        d[label]["gossip"] = gossip(R)
        print(S1.show(d[label]))
        for c, g in d[label]["gossip"].items():
            print(f"  gossip, {c}: negative share {S1.fmt(g['share_negative'])}, pooled {g['pooled']:.3f} ({g['bad']} of {g['good'] + g['bad']}); tellings a night {S1.fmt(g['tellings_a_night'], 2)}")
        print()
    Path(sys.argv[5]).write_text(json.dumps(d, indent=1))


if __name__ == "__main__":
    main()
