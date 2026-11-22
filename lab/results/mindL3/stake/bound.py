"""The ORACLE-GAP UPPER BOUND (library S3.3): the number of partner-specific acts in a night x (b - c), computed from the logs of the
partner-blind chooser (rules) in the oracle setting (Fenn P-kind, Oren P-hard). usage: bound.py RUNS_DIR WORLD
A partner-specific act is an act the rules focal aims at Fenn or Oren (the two whose rule the oracle knows); only there can knowing the
partner change what is chosen. The headline counts the KIND acts among them (the acts whose worth depends on the partner's rule); the all-acts
form counts every person-directed act at the two (praise, accuse, ask ...), which overstates. Per act the most it can be worth to the focal is the benefit b the kind act gives minus its cost c (the
partner answers a kind act with a kind act, so the benefit is at most b). Three forms of (b - c), per act, from all the focal's done
kind acts (agg 'verb|focal') in the same lifetimes:
  generous : (b_imm + bpot) - c_imm            what the act does at once, not time-weighted
  timed    : b_rem - c_rem                     weighted by the share of the night left (what a night-mean error can see)
Plus the harm the oracle could avoid: the harsh acts of Oren (the P-hard partner) on the focal, per night, times their mean damage."""
import sys, json
from pathlib import Path
import numpy as np
from lab.mindL.tournament import KIND_ACTS
D, W = Path(sys.argv[1]), sys.argv[2]
KIND = set(KIND_ACTS) - {"gossip"}
HARSH = {"strike", "shove", "threaten", "insult", "steal", "accuse", "search"}
rows = []
for fp in sorted(D.glob(f"O_{W}_rules_seed*_n100.json")):
    r = json.loads(fp.read_text())
    n = len(r["err"]) - r["win0"]
    to_ps = sum(v for k, v in r["made_to"].items() if k.split(">")[1] in ("Fenn", "Oren"))
    kind_to_ps = sum(v for k, v in r["made_to"].items() if k.split(">")[1] in ("Fenn", "Oren") and k.split(">")[0] in KIND)
    tot = dict(n=0, b=0, bpot=0, c=0, b_rem=0, c_rem=0)
    for key, a in r["agg"].items():
        v, role = key.split("|")
        if v in KIND and role == "focal":
            for k in tot:
                tot[k] += a["n_done"] if k == "n" else a[k]
    n_act = max(1, tot["n"])
    gen = (tot["b"] + tot["bpot"] - tot["c"]) / n_act
    timed = (tot["b_rem"] - tot["c_rem"]) / n_act
    harm = 0.0
    hn = 0
    for key, a in r["agg"].items():
        v, role = key.split("|")
        if v in HARSH and role == "toF":
            hn += a["n_done"]
            harm += -(a["b"]) 
    harm_per_act = harm / max(1, hn)
    from_oren = sum(v for k, v in r["recv_from"].items() if k.endswith("<Oren") and k.split("<")[0] in HARSH)
    rows.append(dict(seed=r["seed"], acts_per_night=to_ps / n, kind_acts_per_night=kind_to_ps / n, b_minus_c_generous=gen, b_minus_c_timed=timed,
                     harm_from_oren_per_night=from_oren / n, harm_per_act=harm_per_act,
                     bound_generous=(kind_to_ps / n) * max(0.0, gen) + (from_oren / n) * max(0.0, harm_per_act),
                     bound_timed=(kind_to_ps / n) * max(0.0, timed) + (from_oren / n) * max(0.0, harm_per_act),
                     bound_all_acts_generous=(to_ps / n) * max(0.0, gen)))
m = {k: float(np.mean([x[k] for x in rows])) for k in rows[0] if k != "seed"}
out = dict(world=W, seeds=[x["seed"] for x in rows], mean=m, rows=rows)
print(json.dumps(out["mean"], indent=1))
(Path(__file__).resolve().parent / f"bound_{W}.json").write_text(json.dumps(out, indent=1))
