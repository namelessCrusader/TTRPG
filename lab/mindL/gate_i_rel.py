"""mindL6 stage A, piece 8: gate (i) of mindL5 (coarse pre-history against a full run, seeds 1-3, ONE year) re-run with the relative grading
(concerns2 params rel=True). Limits are mindL5's, PASS_RULE_A.txt, unchanged:  T1 mean |dw| <= 0.10 (no seed above 0.15), T2 mean Kendall tau >= 0.60,
T3 mean |d khat| <= 0.10, at night 365, both setups (A: from the 36-year coarse past, B: from strangers).

    lab/.venv/bin/python -m lab.mindL.gate_i_rel runs      (6 full runs of 365 nights, 2 workers)
    lab/.venv/bin/python -m lab.mindL.gate_i_rel analyze
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
OLD = Path("/home/nisargparikh/.cache/mark1-build/mindL5/A")
SCR = Path("/home/nisargparikh/.cache/mark1-build/mindL6/A")
RES = Path(__file__).resolve().parents[1] / "results" / "mindL6" / "A"
SEEDS = (1, 2, 3)
PAR = {"rel": True}
NIGHTS = 365


def _run(a):
    seed, setup = a
    from lab.mindL import household as H
    out = SCR / f"full_rel_{setup}_{seed}.json"
    if out.exists():
        return a, "kept"
    init = ages = None
    if setup == "A":
        E = H.merge_expo([json.loads((OLD / f"sample_{s}.json").read_text())["tele"] for s in (101, 102, 103)])
        init, _ = H.prehistory(seed, E, params=PAR)
        ages = dict(H.FAMILY_AGE)
    r = H.run_full(seed, NIGHTS, arm="cn2", init=init, ages=ages, snaps=(30, 91, 182, 365), extra={"cn": {"params": PAR}})
    H.dump(dict(snaps=r["snaps"], flows=r["flows"], seconds=r["seconds"]), out)
    return a, r["seconds"]


def coarse(seed, setup, E, step):
    from lab.mindL import household as H
    allp = {n: [(0.0, 2.0)] for n in H.NAMES}
    if setup == "A":
        x0, _ = H.prehistory(seed, E, params=PAR)
        st, _ = H.prehistory(seed, E, spans=allp, t0=0.0, t1=1.0, step=step, rows0=x0, params=PAR)
    else:
        st, _ = H.prehistory(seed, E, spans=allp, t0=0.0, t1=1.0, step=step, params=PAR)
    return st


def analyze():
    from lab.mindL import household as H
    E = H.merge_expo([json.loads((OLD / f"sample_{s}.json").read_text())["tele"] for s in (101, 102, 103)])
    out = {}
    for setup in ("A", "B"):
        full = {s: json.loads((SCR / f"full_rel_{setup}_{s}.json").read_text())["snaps"]["365"] for s in SEEDS}
        old = {s: json.loads((OLD / f"full_{setup}_cn2_{s}.json").read_text())["snaps"]["365"] for s in SEEDS}
        rows = [H.compare(full[s], coarse(s, setup, E, H.SEASON)) for s in SEEDS]
        floor = [H.compare(full[a], full[b]) for a, b in ((1, 2), (1, 3), (2, 3))]
        m = lambda rs, k: float(np.mean([r[k] for r in rs]))
        res = dict(mean_dw=m(rows, "mean_dw"), max_seed_mean_dw=max(r["mean_dw"] for r in rows), max_dw_any=max(r["max_dw"] for r in rows), tau=m(rows, "tau"),
                   mean_dk=m(rows, "mean_dk"), mean_w_full=m(rows, "mean_w_full"), mean_w_coarse=m(rows, "mean_w_coarse"),
                   floor_dw=m(floor, "mean_dw"), floor_tau=m(floor, "tau"), floor_dk=m(floor, "mean_dk"))
        res["T1"] = bool(res["mean_dw"] <= 0.10 and res["max_seed_mean_dw"] <= 0.15)
        res["T2"] = bool(res["tau"] >= 0.60)
        res["T3"] = bool(res["mean_dk"] <= 0.10)
        # saturation: the mean of each input in the full run, new against old
        def sat(snaps):
            ps = [s for sn in snaps.values() for i, st in sn.items() for j, s in st.items() if j in H.NAMES and j != i]
            return {k: float(np.mean([p[k] for p in ps])) for k in ("w", "F", "EJ", "khat")}
        res["full_inputs_rel"], res["full_inputs_old"] = sat(full), sat(old)
        # order of ties: Kendall tau between w and (the true closeness: kin 2, then co-resident years) is not available; report the spread of w per person
        res["w_spread_rel"] = float(np.mean([np.std([st[j]["w"] for j in st if j in H.NAMES and j != i]) for sn in full.values() for i, st in sn.items()]))
        res["w_spread_old"] = float(np.mean([np.std([st[j]["w"] for j in st if j in H.NAMES and j != i]) for sn in old.values() for i, st in sn.items()]))
        kin = [("Jon", "Mara"), ("Mara", "Jon"), ("Mara", "Pip"), ("Jon", "Pip"), ("Pip", "Mara"), ("Pip", "Jon"), ("Mara", "Gran"), ("Gran", "Mara"), ("Gran", "Pip"), ("Pip", "Gran")]
        lod = [(i, j) for i in H.NAMES[:4] for j in H.NAMES[4:]] + [(i, j) for i in H.NAMES[4:] for j in H.NAMES[:4]]
        for tag, snaps in (("rel", full), ("old", old)):
            kw = [sn[i][j]["w"] for sn in snaps.values() for i, j in kin if j in sn.get(i, {})]
            lw = [sn[i][j]["w"] for sn in snaps.values() for i, j in lod if j in sn.get(i, {})]
            res[f"w_family_{tag}"], res[f"w_lodger_{tag}"] = float(np.mean(kw)), float(np.mean(lw))
        out[f"{setup}_1y"] = res
    out["PASS_i"] = all(out[k][t] for k in out if k.endswith("y") for t in ("T1", "T2", "T3"))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "gate_i_rel.json").write_text(json.dumps(out, indent=1, default=float))
    for k, v in out.items():
        if k.endswith("y"):
            print(k, {x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items()})
    print("PASS_i", out["PASS_i"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["runs", "analyze"])
    a = ap.parse_args()
    if a.stage == "analyze":
        analyze()
    else:
        SCR.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        items = [(s, st) for s in SEEDS for st in ("A", "B")]
        with mp.get_context("spawn").Pool(2, maxtasksperchild=1) as p:
            for k, r in enumerate(p.imap_unordered(_run, items)):
                print(f"[{k + 1}/{len(items)} {time.perf_counter() - t0:.0f} s] {r}", flush=True)
