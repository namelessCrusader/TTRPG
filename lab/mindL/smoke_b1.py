"""mindL6 stage B1, item 6: SMOKE (PASS_RULE_B1.txt section E): 3 seeds, the lethal family inn, 30 nights, 7 people each, every person loaded from its childhood
(length 30 and 90). Arms: naive (drives2 SCBDW + body2, no childhood), p30, p90, and the rules for reference. Not a verdict.
    lab/.venv/bin/python -m lab.mindL.smoke_b1 run        (2 workers; resumable)
    lab/.venv/bin/python -m lab.mindL.smoke_b1 analyze
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import pickle
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
SCR = Path("/home/nisargparikh/.cache/mark1-build/mindL6/B1")
RES = Path(__file__).resolve().parents[1] / "results" / "mindL6" / "B1"
SEEDS = (91, 92, 93)
LENGTHS = (30, 90)
NIGHTS = 30
PEOPLE = ["Jon", "Mara", "Pip", "Gran", "Tom", "Sir Ada", "Oren"]
ARMS = ("naive", "p30", "p90", "rules")


def _child(a):
    seed, person, nights = a
    from lab.mindL import childhood as C
    r = C.run_childhood(seed, person, nights)
    return a, r["seconds"]


def _adult(a):
    arm, seed = a
    out = SCR / f"smoke_{arm}_{seed}.json"
    if out.exists():
        return a, "kept"
    from lab.mindL import childhood as C
    from lab.mindL import household as H
    from lab.mindL import reader
    rec = {}
    pre = None
    if arm in ("p30", "p90"):
        n = int(arm[1:])
        pre = {p: pickle.loads((C.SCR / f"child_{seed}_{p.replace(' ', '_')}_{n}.pkl").read_bytes())["blob"] for p in PEOPLE}

    def hook(pol):
        orig = pol.agent

        def agent(p):
            new = p.name not in pol.agents
            ag = orig(p)
            if new:
                v = reader.VI["drink"]
                rec[p.name] = dict(cnt_drink=float(ag.wm.cnt[v]), n0_max=float(ag.wm.n0.max()), decisions=int(ag.decisions), n=float(ag.D.n), m=float(ag.D.m))
            return ag
        pol.agent = agent
    if arm == "rules":
        r = H.run_full(seed, NIGHTS, arm="rules", lethal=True, res=True, drinks=True, flows=False, wflags=("body2",))
    else:
        r = H.run_full(seed, NIGHTS, arm="off", lethal=True, res=True, drinks=True, extra=C.ADULT, flows=False, wflags=("body2",), preload=pre, pol_hook=hook)
    H.dump(dict(deaths=r["deaths"], res=r["res"], drinks=r["drinks"], seconds=r["seconds"], start=rec), out)
    return a, r["seconds"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def analyze():
    out = {"arms": {}, "childhoods": {}}
    for arm in ARMS:
        D = {s: json.loads((SCR / f"smoke_{arm}_{s}.json").read_text()) for s in SEEDS}
        n = 7 * len(SEEDS)
        causes: dict = {}
        for s in SEEDS:
            for v in D[s]["deaths"].values():
                causes[str(v["cause"])] = causes.get(str(v["cause"]), 0) + 1
        dead = sum(causes.values())
        th = causes.get("thirst", 0)
        dr = [sum(D[s]["drinks"][k][p] for p in PEOPLE) for s in SEEDS for k in range(len(D[s]["drinks"]))]
        alive_nights = sum(1 for s in SEEDS for k in range(len(D[s]["drinks"])) for p in PEOPLE if D[s]["deaths"].get(p, {"night": 999})["night"] > k)
        start = [v for s in SEEDS for v in D[s]["start"].values()]
        out["arms"][arm] = dict(lives=n, dead=dead, causes=causes, dead_share=round(dead / n, 3), dead_wilson=wilson(dead, n), thirst_share=round(th / n, 3),
                                thirst_wilson=wilson(th, n), drinks_per_living_night=round(sum(dr) / max(1, alive_nights), 2),
                                first_death_night=[min([v["night"] for v in D[s]["deaths"].values()] or [NIGHTS + 1]) for s in SEEDS],
                                loaded=dict(persons=len(start), with_drink_experience=sum(1 for x in start if x["cnt_drink"] > 0), with_told_priors=sum(1 for x in start if x["n0_max"] > 0),
                                            mean_drink_cnt=round(float(np.mean([x["cnt_drink"] for x in start])), 1) if start else None),
                                exps_n_m=[(round(x["n"], 2), round(x["m"], 2)) for x in start[:7]], seconds=round(sum(D[s]["seconds"] for s in SEEDS), 1))
    for L in LENGTHS:
        st = []
        for s in SEEDS:
            for p in PEOPLE:
                r = pickle.loads((SCR / "childhoods" / f"child_{s}_{p.replace(' ', '_')}_{L}.pkl").read_bytes())
                st.append(r["stats"] | {"sec": r["seconds"]})
        q = lambda f: round(float(np.mean([f(x) for x in st])), 2)
        k = max(1, L // 4)
        out["childhoods"][str(L)] = dict(n=len(st), episodes_per_hour=q(lambda x: x["episodes"] / (L * 24.0)), share_via_child=q(lambda x: x["via_child"] / max(1, x["episodes"])),
                                         taken_per_night=q(lambda x: x["taken"] / L), errands_per_night=q(lambda x: x["errands"] / L), given_serves_total=q(lambda x: x["given"]),
                                         drinks_first_quarter=q(lambda x: np.mean(x["own_drinks"][:k])), drinks_last_quarter=q(lambda x: np.mean(x["own_drinks"][-k:])),
                                         drink_cnt=q(lambda x: x["drink_cnt"]), watched=q(lambda x: x["watched"]), decisions=q(lambda x: x["decisions"]), seconds_each=q(lambda x: x["sec"]),
                                         never_drank_in_childhood=sum(1 for x in st if sum(x["own_drinks"]) == 0))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "smoke_b1.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "analyze"])
    a = ap.parse_args()
    if a.stage == "analyze":
        analyze()
    else:
        t0 = time.perf_counter()
        ctx = mp.get_context("spawn")
        items = [(s, p, L) for L in LENGTHS for s in SEEDS for p in PEOPLE]
        with ctx.Pool(2, maxtasksperchild=4) as pool:
            for k, r in enumerate(pool.imap_unordered(_child, items)):
                print(f"[child {k + 1}/{len(items)} {time.perf_counter() - t0:.0f} s] {r}", flush=True)
        items = [(arm, s) for s in SEEDS for arm in ARMS]
        with ctx.Pool(2, maxtasksperchild=1) as pool:
            for k, r in enumerate(pool.imap_unordered(_adult, items)):
                print(f"[adult {k + 1}/{len(items)} {time.perf_counter() - t0:.0f} s] {r}", flush=True)
