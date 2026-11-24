"""mindL6 stage B1, item 4: gate (i) with BOTH yardsticks (PASS_RULE_B1.txt section B, written before this ran). Applied to the stage A full runs
(scratch mindL6/A/full_rel_{A,B}_{1,2,3}.json): no new full run.   lab/.venv/bin/python -m lab.mindL.gate_i_b1"""
import json
from pathlib import Path

import numpy as np

from lab.mindL import household as H
from lab.mindL.gate_i_rel import OLD, SCR, SEEDS, coarse

RES = Path(__file__).resolve().parents[1] / "results" / "mindL6" / "B1"


def main():
    E = H.merge_expo([json.loads((OLD / f"sample_{s}.json").read_text())["tele"] for s in (101, 102, 103)])
    out = {}
    for setup in ("A", "B"):
        full = {s: json.loads((SCR / f"full_rel_{setup}_{s}.json").read_text())["snaps"]["365"] for s in SEEDS}
        cf = {s: H.compare(full[s], coarse(s, setup, E, H.SEASON)) for s in SEEDS}
        ff = {(a, b): H.compare(full[a], full[b]) for a in SEEDS for b in SEEDS if a < b}
        m = lambda rs, k: float(np.mean([r[k] for r in rs]))
        cfl, ffl = list(cf.values()), list(ff.values())
        r = dict(cf_dw=m(cfl, "mean_dw"), cf_dw_seeds=[cf[s]["mean_dw"] for s in SEEDS], cf_tau=m(cfl, "tau"), cf_dk=m(cfl, "mean_dk"),
                 ff_dw=m(ffl, "mean_dw"), ff_dw_pairs=[x["mean_dw"] for x in ffl], ff_tau=m(ffl, "tau"), ff_tau_pairs=[x["tau"] for x in ffl],
                 ff_dk=m(ffl, "mean_dk"), ff_dk_pairs=[x["mean_dk"] for x in ffl])
        # Y1 absolute
        r["Y1_T1"] = bool(r["cf_dw"] <= 0.10 and max(r["cf_dw_seeds"]) <= 0.15)
        r["Y1_T2"] = bool(r["cf_tau"] >= 0.60)
        r["Y1_T3"] = bool(r["cf_dk"] <= 0.10)
        # Y2 noise relative
        r["Y2_R1"] = bool(r["cf_dw"] <= 1.10 * r["ff_dw"] and max(r["cf_dw_seeds"]) <= 1.30 * r["ff_dw"])
        r["Y2_R2"] = bool(r["cf_tau"] >= r["ff_tau"] - 0.10)
        r["Y2_R3"] = bool(r["cf_dk"] <= 1.10 * r["ff_dk"])
        # not gated: the paired per-seed version: coarse vs full_s against full_s vs the other two full runs
        pw = []
        for s in SEEDS:
            o = [ff[tuple(sorted((s, t)))] for t in SEEDS if t != s]
            pw.append(dict(seed=s, dw_cf=cf[s]["mean_dw"], dw_ff=float(np.mean([x["mean_dw"] for x in o])), tau_cf=cf[s]["tau"], tau_ff=float(np.mean([x["tau"] for x in o])),
                           dk_cf=cf[s]["mean_dk"], dk_ff=float(np.mean([x["mean_dk"] for x in o]))))
        r["paired_per_seed"] = pw
        out[setup] = r
    out["PASS_Y1"] = bool(all(out[s][k] for s in ("A", "B") for k in ("Y1_T1", "Y1_T2", "Y1_T3")))
    out["PASS_Y2"] = bool(all(out[s][k] for s in ("A", "B") for k in ("Y2_R1", "Y2_R2", "Y2_R3")))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "gate_i_b1.json").write_text(json.dumps(out, indent=1, default=float))
    for s in ("A", "B"):
        print(s, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[s].items() if k != "paired_per_seed"})
        for x in out[s]["paired_per_seed"]:
            print("   ", {k: round(v, 3) for k, v in x.items()})
    print("PASS_Y1", out["PASS_Y1"], "PASS_Y2", out["PASS_Y2"])


if __name__ == "__main__":
    main()
