"""Stage A analysis: gate (i) twins, noise floor, step-size drift, care flows.  lab/.venv/bin/python -m lab.mindL.analyze_a"""
import json
from pathlib import Path
import numpy as np
from lab.mindL import household as H

SCR = Path("/home/nisargparikh/.cache/mark1-build/mindL5/A")
RES = Path(__file__).resolve().parents[1] / "results" / "mindL5" / "A"
SEEDS = (1, 2, 3)


def load(p):
    return json.loads((SCR / p).read_text())


def coarse(seed, setup, E, years, step):
    allp = {n: [(0.0, years + 1.0)] for n in H.NAMES}
    if setup == "A":
        x0, _ = H.prehistory(seed, E)
        st, _ = H.prehistory(seed, E, spans=allp, t0=0.0, t1=years, step=step, rows0=x0)
    else:
        st, _ = H.prehistory(seed, E, spans=allp, t0=0.0, t1=years, step=step)
    return st


def main():
    E = H.merge_expo([load(f"sample_{s}.json")["tele"] for s in (101, 102, 103)])
    out = {}
    for setup in ("A", "B"):
        full = {s: load(f"full_{setup}_cn2_{s}.json") for s in SEEDS}
        for yrs, night in ((1, "365"), (2, "730")):
            rows = []
            for s in SEEDS:
                rows.append(H.compare(full[s]["snaps"][night], coarse(s, setup, E, yrs, H.SEASON)))
            floor = [H.compare(full[a]["snaps"][night], full[b]["snaps"][night]) for a, b in ((1, 2), (1, 3), (2, 3))]
            m = lambda rs, k: float(np.mean([r[k] for r in rs]))
            res = dict(per_seed=rows, mean_dw=m(rows, "mean_dw"), max_dw_any=max(r["max_dw"] for r in rows), max_seed_mean_dw=max(r["mean_dw"] for r in rows),
                       tau=m(rows, "tau"), mean_dk=m(rows, "mean_dk"), mean_w_full=m(rows, "mean_w_full"), mean_w_coarse=m(rows, "mean_w_coarse"),
                       floor_dw=m(floor, "mean_dw"), floor_tau=m(floor, "tau"), floor_dk=m(floor, "mean_dk"))
            res["T1"] = bool(res["mean_dw"] <= 0.10 and res["max_seed_mean_dw"] <= 0.15)
            res["T2"] = bool(res["tau"] >= 0.60)
            res["T3"] = bool(res["mean_dk"] <= 0.10)
            steps = {}
            for stp in (30.0, H.SEASON, 182.5):
                steps[str(stp)] = float(np.mean([H.compare(full[s]["snaps"][night], coarse(s, setup, E, yrs, stp))["mean_dw"] for s in SEEDS]))
            res["drift_by_step_nights"] = steps
            out[f"{setup}_{yrs}y"] = res
    out["PASS_i"] = all(out[k][t] for k in out if k.endswith("y") for t in ("T1", "T2", "T3"))
    # flows
    fl = {}
    fam = ("Jon", "Mara", "Pip", "Gran")
    lod = ("Tom", "Sir Ada", "Oren")
    for arm in ("rules", "off", "past"):
        R = []
        for s in (11, 12, 13, 14, 15, 16):
            p = SCR / f"flow_{arm}_{s}.json"
            if not p.exists():
                continue
            f = json.loads(p.read_text())["flows"]
            tf = sum(v for k, v in f.items() if k.split(">")[0] in fam and k.split(">")[1] in fam)
            tl = sum(v for k, v in f.items() if k.split(">")[0] in fam and k.split(">")[1] in lod)
            # family actors have 3 family and 3 lodger targets: equal opportunity
            R.append((tf, tl))
        fl[arm] = dict(family_care=[r[0] for r in R], lodger_care=[r[1] for r in R], pooled_ratio=(sum(r[0] for r in R) + 0.5) / (sum(r[1] for r in R) + 0.5) if R else None)
    out["flows"] = fl
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "gate_i.json").write_text(json.dumps(out, indent=1, default=float))
    for k, v in out.items():
        if k.endswith("y"):
            print(k, {x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items() if x != "per_seed"})
    print("PASS_i", out["PASS_i"]); print(json.dumps(fl))


if __name__ == "__main__":
    main()
