"""Round 2 extra: split the L1/L2 treatment index into its two shares (kind share, provocation share) and the
raw counts, kind partner minus hard partner, window = the 40 nights before the partners turn deaf.
    MINDL_RUNS=lab/results/mindL2/runs lab/.venv/bin/python -m lab.mindL.decomp_L2"""
import json
import os
os.environ.setdefault("MINDL_RUNS", "lab/results/mindL2/runs")
from lab.mindL import report_L2 as R2
from lab.mindL.report_L import ci, fmt, shares, SEEDS

R = R2.load("L12")
cut = 400
out = {}
for c in ("mind", "bandit", "ppo", "rules", "random"):
    acc = {k: [] for k in ("kind_share", "prov_share", "kind_per_night", "prov_per_night", "near", "harsh_received_hard")}
    for s in SEEDS:
        v = []
        for var, kp in (("A", "Fenn"), ("B", "Oren")):
            run = R.get((var, c, s))
            if run is None:
                continue
            hp = "Oren" if kp == "Fenn" else "Fenn"
            a, b = shares(run, kp, cut - 40, cut), shares(run, hp, cut - 40, cut)
            v.append({k: a[k] - b[k] for k in ("kind_share", "prov_share", "kind_per_night", "prov_per_night", "near")})
        if len(v) == 2:
            for k in v[0]:
                acc[k].append(sum(x[k] for x in v) / 2)
    out[c] = {k: ci(x) for k, x in acc.items() if x}
json.dump(out, open("lab/results/mindL2/r2_L2_decomposition.json", "w"), indent=1)
for c, d in out.items():
    print(c, " | ".join(f"{k}: {fmt(v)}" for k, v in d.items() if k != "harsh_received_hard"))
