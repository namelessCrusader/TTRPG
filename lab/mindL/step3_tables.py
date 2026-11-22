"""Step 3: what the concerns formed in the real lives (fxc arm, all6s). Weights at the end of a night, by role, and the clone distances.

    lab/.venv/bin/python -m lab.mindL.step3_tables RUNS_DIR OUT.json
Roles: kind = the P-need partner (Fenn in life A, Oren in B), hard = the P-hard partner, deaf = Fenn/Oren in the deaf life (they never
answer: the placebo partners), other = the rest of the cast. D_life = L1 distance of the two tables over the cast between the clones of
life A and life B (same seed, nature and draws); D_rng = the same life A with another seed for the mind's own draws (the variant Ar).
"""
import json
import sys
from pathlib import Path

import numpy as np

from lab.mindL.report_L import ci, fmt

SEEDS = tuple(range(1, 11))
NIGHT = 199


def rows(runs, v, s, n=220, arm="fxc"):
    for nn in (n, 200, 300, 120):
        fp = Path(runs) / f"L12_{v}_{arm}_seed{s}_n{nn}_all6s_need.json"
        if fp.exists():
            return json.loads(fp.read_text())
    return None


def at(run, night):
    r = next((r for r in run["nights_rows"] if r["n"] == night), None)
    return r["cn"] if r and "cn" in r else None


def main(runs, out):
    res = {}
    kind, hard, deaf, other, dk, dh = [], [], [], [], [], []
    for s in SEEDS:
        A, B, D = rows(runs, "A", s), rows(runs, "B", s), rows(runs, "deaf", s)
        if A is None or B is None or D is None:
            continue
        a, b, d = at(A, NIGHT), at(B, NIGHT), at(D, NIGHT)
        focal = A["focal"]
        kind.append(np.mean([a.get("Fenn", 0.0), b.get("Oren", 0.0)]))
        hard.append(np.mean([a.get("Oren", 0.0), b.get("Fenn", 0.0)]))
        deaf.append(np.mean([d.get("Fenn", 0.0), d.get("Oren", 0.0)]))
        oth = [v for t in (a, b, d) for k, v in t.items() if k not in ("Fenn", "Oren")]
        other.append(float(np.mean(oth)) if oth else 0.0)
    res["weights at night 199"] = {"kind partner": ci(kind), "hard partner": ci(hard), "deaf partners (life deaf)": ci(deaf),
                                   "others (mean of the rest of the cast)": ci(other), "kind minus hard (paired)": ci(np.array(kind) - np.array(hard))}
    names = ("Fenn", "Oren", "Tom", "Sir Ada", "Marta", "the bandit")
    dl, dr = [], []
    for s in SEEDS:
        A, B, Ar = rows(runs, "A", s), rows(runs, "B", s), rows(runs, "Ar", s)
        if A is None or B is None or Ar is None:
            continue
        a, b, ar = at(A, NIGHT), at(B, NIGHT), at(Ar, NIGHT)
        dist = lambda x, y: float(sum(abs(x.get(n, 0.0) - y.get(n, 0.0)) for n in names))
        dl.append(dist(a, b))
        dr.append(dist(a, ar))
    if dl:
        L, R = ci(dl), ci(dr)
        res["clones"] = {"D_life (A vs B)": L, "D_rng (A vs A, other mind seed)": R, "ratio of means": float(L["mean"] / R["mean"]) if R["mean"] else None,
                         "paired D_life - D_rng": ci(np.array(dl) - np.array(dr)), "seeds": len(dl)}
    # the weight over the life, kind minus hard, by 20-night block (formation)
    blocks = {}
    for lo in range(0, 200, 40):
        v = []
        for s in SEEDS:
            A, B = rows(runs, "A", s), rows(runs, "B", s)
            if A is None or B is None:
                continue
            a, b = at(A, lo + 39), at(B, lo + 39)
            if a and b:
                v.append(np.mean([a.get("Fenn", 0.0) - a.get("Oren", 0.0), b.get("Oren", 0.0) - b.get("Fenn", 0.0)]))
        if v:
            blocks[f"night {lo + 39}"] = ci(v)
    res["kind minus hard weight over the life"] = blocks
    Path(out).write_text(json.dumps(res, indent=1, default=float))
    for k, v in res.items():
        print(k)
        for kk, vv in v.items():
            print("  ", kk, fmt(vv) if isinstance(vv, dict) else vv)


if __name__ == "__main__":
    main(*sys.argv[1:3])
