"""Stage A runner (mindL5): exposure samples, twins, controlled pasts.   lab/.venv/bin/python -m lab.mindL.family_a STAGE [--workers 2]

Long stages run under: flock .../lab.lock .../cap.sh nice -n 19 lab/.venv/bin/python -m lab.mindL.family_a STAGE
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
SCR = Path("/home/nisargparikh/.cache/mark1-build/mindL5/A")
RES = Path(__file__).resolve().parents[1] / "results" / "mindL5" / "A"
SAMPLE_SEEDS = (101, 102, 103)
TWIN_SEEDS = (1, 2, 3)
YEARS_N = 730


def _sample(seed):
    from lab.mindL import household as H
    out = SCR / f"sample_{seed}.json"
    if out.exists():
        return seed, "kept"
    r = H.run_full(seed, 90, arm="passive", tele=(30, 90))
    H.dump(dict(tele=r["tele"], seconds=r["seconds"]), out)
    return seed, r["seconds"]


def _twin(a):
    """a = (seed, setup, arm): a full run of 730 nights. setup A starts from the coarse past, B from strangers; arm 'cn2' (steering) or 'passive'."""
    seed, setup, arm = a
    from lab.mindL import household as H
    out = SCR / f"full_{setup}_{arm}_{seed}.json"
    if out.exists():
        return a, "kept"
    init = ages = None
    if setup == "A":
        E = H.merge_expo([json.loads((SCR / f"sample_{s}.json").read_text())["tele"] for s in SAMPLE_SEEDS])
        init, _ = H.prehistory(seed, E)
        ages = dict(H.FAMILY_AGE)
    r = H.run_full(seed, YEARS_N, arm=arm, init=init, ages=ages, snaps=(30, 91, 182, 365, 730))
    H.dump(dict(snaps=r["snaps"], flows=r["flows"], deaths=r["deaths"], seconds=r["seconds"]), out)
    return a, r["seconds"]


FLOW_SEEDS = (11, 12, 13, 14, 15, 16)
FLOW_NIGHTS = 45


def _flow(a):
    """a = (seed, arm): 45 nights of the family inn; arm rules, off (minds without concerns), past (cn2 minds that start from the coarse pre-history)."""
    seed, arm = a
    from lab.mindL import household as H
    out = SCR / f"flow_{arm}_{seed}.json"
    if out.exists():
        return a, "kept"
    init = ages = None
    if arm == "past":
        E = H.merge_expo([json.loads((SCR / f"sample_{s}.json").read_text())["tele"] for s in SAMPLE_SEEDS])
        init, _ = H.prehistory(seed, E)
        ages = dict(H.FAMILY_AGE)
    r = H.run_full(seed, FLOW_NIGHTS, arm={"rules": "rules", "off": "off", "past": "cn2"}[arm], init=init, ages=ages, snaps=(FLOW_NIGHTS,))
    H.dump(dict(flows=r["flows"], deaths=r["deaths"], snaps=r["snaps"], seconds=r["seconds"]), out)
    return a, r["seconds"]


def pool_run(fn, items, workers):
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(min(2, workers), maxtasksperchild=1) as p:
        for k, r in enumerate(p.imap_unordered(fn, items)):
            print(f"[{k + 1}/{len(items)} {time.perf_counter() - t0:.0f} s] {r}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sample", "twins", "flows"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--arm", default="cn2")
    a = ap.parse_args()
    SCR.mkdir(parents=True, exist_ok=True)
    if a.stage == "sample":
        pool_run(_sample, list(SAMPLE_SEEDS), a.workers)
    elif a.stage == "twins":
        items = [(s, st, a.arm) for s in TWIN_SEEDS for st in ("A", "B")]
        pool_run(_twin, items, a.workers)
    elif a.stage == "flows":
        pool_run(_flow, [(s, arm) for s in FLOW_SEEDS for arm in ("rules", "off", "past")], a.workers)


if __name__ == "__main__":
    main()
