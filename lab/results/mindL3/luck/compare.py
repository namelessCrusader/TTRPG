"""Fixed world ('fix') against the old all2 world, paired by seed and focal. Reads the two run dirs."""
import json, sys
from pathlib import Path
import numpy as np
from lab.mindL.tournament_report import ci, fmt, outcome, SEEDS
OLD = Path("/home/nisargparikh/Desktop/Fun Stuff/Mark_1/lab/results/mindL3/inn/runs")
NEW = Path(sys.argv[1])
def load(d):
    out = {}
    for fp in sorted(d.glob("*.json")):
        r = json.loads(fp.read_text()); out[(r["kind"], r["world"], r["chooser"], r["seed"], r.get("rate"))] = r
    return out
O, N = load(OLD), load(NEW)
def err(R, kind, world, c):
    return {s: outcome(R[(kind, world, c, s, None)])["err"] for s in SEEDS if (kind, world, c, s, None) in R}
def per_night(r, key, who="made"):
    n = len(r["err"]) - r["win0"]
    return sum(v for k, v in r[who].items() if k in key) / n
HARSH = {"strike","shove","threaten","insult","steal","accuse","search","refuse"}
KIND = {"praise","give","help up","bandage","defend","accept","serve"}
L = ["LUCK WORLD AGAINST all3 ('fix'), paired by seed (same focal, same night seeds)",
     "error = focal drive error nights 20-99 (lower is better); diff = luck minus all3 [95% CI over seeds]\n",
     f"{"chooser":14s} {"all3":>7s} {"luck":>7s}  diff"]
cs = sorted({k[2] for k in N if k[0] == "A"})
for c in cs:
    a, b = err(O, "A", "fix", c), err(N, "A", "luck", c)
    if not a or not b: continue
    d = [b[s] - a[s] for s in SEEDS if s in a and s in b]
    L.append(f"{c:14s} {np.mean(list(a.values())):7.3f} {np.mean(list(b.values())):7.3f}  {fmt(ci(d))}")
L.append("\nWHAT THE FOCAL DOES AND RECEIVES, per night, mean over seeds (harsh = world's harmful acts and refusing; kind = praise, give, help up, bandage, defend, accept, serve)")
L.append(f"{'chooser':14s} {'world':5s} {'made kind':>9s} {'made harsh':>10s} {'got kind':>9s} {'got harsh':>10s} {'served':>7s} {'ate':>5s} {'strikes got':>11s}")
for c in cs:
    for (R, w, tag) in ((O, "fix", "all3"), (N, "luck", "luck")):
        rs = [R[("A", w, c, s, None)] for s in SEEDS if ("A", w, c, s, None) in R]
        if not rs: continue
        m = lambda key, who: np.mean([per_night(r, key, who) for r in rs])
        L.append(f"{c:14s} {tag:5s} {m(KIND,'made'):9.1f} {m(HARSH,'made'):10.1f} {m(KIND,'recv'):9.1f} {m(HARSH,'recv'):10.1f} "
                 f"{m({'serve'},'recv'):7.2f} {m({'eat'},'made'):5.2f} {m({'strike'},'recv'):11.2f}")
for kind in ("O",):
    L.append("\nORACLE FAMILY (L2 setting): rules and oracle_both error, luck vs all3")
    for c in ("rules", "oracle_avoid", "oracle_kind", "oracle_both"):
        a, b = err(O, "O", "fix", c), err(N, "O", "luck", c)
        if a and b:
            d = [b[s] - a[s] for s in SEEDS if s in a and s in b]
            L.append(f"  {c:14s} all3 {np.mean(list(a.values())):.3f}  luck {np.mean(list(b.values())):.3f}  diff {fmt(ci(d))}")
print("\n".join(L))
