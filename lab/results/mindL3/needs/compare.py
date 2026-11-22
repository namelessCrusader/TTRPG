"""World X against all4 (the luck world, v=0.3), paired by seed and focal.
usage: compare.py DIR WORLD"""
import json, sys
from pathlib import Path
import numpy as np
from lab.mindL.tournament_report import ci, fmt, SEEDS, outcome
OLD = Path("/home/nisargparikh/Desktop/Fun Stuff/Mark_1/lab/results/mindL3/luck/runs_v03")
NEW, W = Path(sys.argv[1]), sys.argv[2]
def load(d):
    out = {}
    for fp in sorted(d.glob("*.json")):
        r = json.loads(fp.read_text()); out[(r["kind"], r["world"], r["chooser"], r["seed"])] = r
    return out
O, N = load(OLD), load(NEW)
def err(R, kind, world, c):
    return {s: outcome(R[(kind, world, c, s)])["err"] for s in SEEDS if (kind, world, c, s) in R}
def dead(R, kind, world, c):
    return np.mean([sum(R[(kind, world, c, s)]["dead"][20:]) for s in SEEDS if (kind, world, c, s) in R])
print(f"{W.upper()} AGAINST all4, paired by seed (same focal, same night seeds); error = focal drive error nights 20-99; diff = {W} minus all4 [95% CI]\n")
print(f"{'chooser':14s} {'all4':>7s} {W:>7s}  diff                          deaths/80n all4 -> {W}")
for c in sorted({k[2] for k in N if k[0] == "A"}):
    a, b = err(O, "A", "luck", c), err(N, "A", W, c)
    d = [b[s] - a[s] for s in SEEDS if s in a and s in b]
    print(f"{c:14s} {np.mean(list(a.values())):7.3f} {np.mean(list(b.values())):7.3f}  {fmt(ci(d)):28s} {dead(O,'A','luck',c):.2f} -> {dead(N,'A',W,c):.2f}")
print("\nORACLE FAMILY (L2 setting)")
for c in ("rules", "oracle_avoid", "oracle_kind", "oracle_both"):
    a, b = err(O, "O", "luck", c), err(N, "O", W, c)
    if a and b:
        d = [b[s] - a[s] for s in SEEDS if s in a and s in b]
        print(f"  {c:14s} all4 {np.mean(list(a.values())):.3f}  {W} {np.mean(list(b.values())):.3f}  diff {fmt(ci(d))}")
