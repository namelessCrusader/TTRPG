"""Stage B runner (mindL5): the population. lab/.venv/bin/python -m lab.mindL.family_b STAGE
Long stages run under: flock .../lab.lock .../cap.sh nice -n 19 lab/.venv/bin/python -m lab.mindL.family_b STAGE
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
A = Path("/home/nisargparikh/.cache/mark1-build/mindL5/A")
SCR = Path("/home/nisargparikh/.cache/mark1-build/mindL5/B")
SEEDS = tuple(range(11, 41))
SAMPLE_SEEDS = (101, 102, 103)
FLOW_NIGHTS = 45
LETHAL_NIGHTS = 90
ARM = {"rules": "rules", "off": "off", "past": "cn2"}


def _run(a):
    kind, seed, arm = a
    from lab.mindL import household as H
    out = SCR / f"{kind}_{arm}_{seed}.json"
    if out.exists():
        return a, "kept"
    init = ages = None
    if arm == "past":
        E = H.merge_expo([json.loads((A / f"sample_{s}.json").read_text())["tele"] for s in SAMPLE_SEEDS])
        init, _ = H.prehistory(seed, E)
        ages = dict(H.FAMILY_AGE)
    if kind == "flow":
        r = H.run_full(seed, FLOW_NIGHTS, arm=ARM[arm], init=init, ages=ages, snaps=(FLOW_NIGHTS,))
    else:
        r = H.run_full(seed, LETHAL_NIGHTS, arm=ARM[arm], init=init, ages=ages, lethal=True, res=True)
    H.dump(dict(flows=r["flows"], deaths=r["deaths"], res=r["res"], snaps=r["snaps"], seconds=r["seconds"]), out)
    return a, r["seconds"]


def pool_run(items, workers=2):
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(min(2, workers), maxtasksperchild=1) as p:
        for k, r in enumerate(p.imap_unordered(_run, items)):
            print(f"[{k + 1}/{len(items)} {time.perf_counter() - t0:.0f} s] {r}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["flow", "lethal"])
    ap.add_argument("--arms", default="rules,past,off")
    a = ap.parse_args()
    pool_run([(a.stage, s, arm) for arm in a.arms.split(",") for s in SEEDS])
