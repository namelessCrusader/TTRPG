"""mindL6 stage A, piece 9: the diagnostic (rule: PASS_RULE_A6.txt, written before the run).
    lab/.venv/bin/python -m lab.mindL.diag_a6 run        (45 runs of 90 nights, 2 workers)
    lab/.venv/bin/python -m lab.mindL.diag_a6 analyze
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
SCR = Path("/home/nisargparikh/.cache/mark1-build/mindL6/A")
OLD = Path("/home/nisargparikh/.cache/mark1-build/mindL5/B")
RES = Path(__file__).resolve().parents[1] / "results" / "mindL6" / "A"
SEEDS = tuple(range(11, 26))
NIGHTS = 90
ARMS = {"old": ("off", {}), "drives2": ("off", {"drives2": True}), "rules": ("rules", {})}


def _run(a):
    arm, seed = a
    from lab.mindL import household as H
    out = SCR / f"diag_{arm}_{seed}.json"
    if out.exists():
        return a, "kept"
    kind, extra = ARMS[arm]
    r = H.run_full(seed, NIGHTS, arm=kind, lethal=True, res=True, drinks=True, extra=extra, flows=False)
    H.dump(dict(deaths=r["deaths"], res=r["res"], drinks=r["drinks"], seconds=r["seconds"]), out)
    return a, r["seconds"]


def boot(f, seeds, n=2000, seed=1):
    rng = np.random.default_rng(seed)
    v = [f(list(rng.choice(seeds, len(seeds)))) for _ in range(n)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def analyze():
    D = {arm: {s: json.loads((SCR / f"diag_{arm}_{s}.json").read_text()) for s in SEEDS} for arm in ARMS}
    names = ["Jon", "Mara", "Pip", "Gran", "Tom", "Sir Ada", "Oren"]
    out = {}

    def share(arm, seeds, cause=None):
        tot = dead = 0
        for s in seeds:
            d = D[arm][s]["deaths"]
            tot += 7
            dead += sum(1 for v in d.values() if cause is None or v["cause"] == cause)
        return dead / tot
    for arm in ARMS:
        causes = {}
        for s in SEEDS:
            for v in D[arm][s]["deaths"].values():
                causes[str(v["cause"])] = causes.get(str(v["cause"]), 0) + 1
        first = [min([v["night"] for v in D[arm][s]["deaths"].values()] or [91]) for s in SEEDS]
        # drinks per living person-night by window; stable drinkers at night 30
        win = {}
        for lo, hi in ((0, 10), (10, 30), (30, 90)):
            num = den = 0
            for s in SEEDS:
                dead_by = {n: v["night"] for n, v in D[arm][s]["deaths"].items()}
                for night in range(lo, min(hi, len(D[arm][s]["drinks"]))):
                    for n in names:
                        if dead_by.get(n, 999) > night:
                            num += D[arm][s]["drinks"][night][n]
                            den += 1
            win[f"{lo}-{hi - 1}"] = num / den if den else None
        stable = tot = 0
        for s in SEEDS:
            dead_by = {n: v["night"] for n, v in D[arm][s]["deaths"].items()}
            for n in names:
                if dead_by.get(n, 999) > 30 and len(D[arm][s]["drinks"]) >= 30:
                    tot += 1
                    stable += all(D[arm][s]["drinks"][k][n] > 0 for k in range(20, 30))
        out[arm] = dict(dead=share(arm, SEEDS), thirst=share(arm, SEEDS, "thirst"), thirst_ci=boot(lambda ss: share(arm, ss, "thirst"), SEEDS),
                        hunger=share(arm, SEEDS, "hunger"), other=share(arm, SEEDS) - share(arm, SEEDS, "thirst") - share(arm, SEEDS, "hunger"),
                        causes=causes, median_first_death=float(np.median(first)), first_deaths=first, drinks=win,
                        stable_drinkers=(stable / tot if tot else None), n_stable=tot,
                        burn=float(np.mean([r[n]["burn"] for s in SEEDS for r in D[arm][s]["res"] for n in names])))
    diff = lambda ss: share("old", ss, "thirst") - share("drives2", ss, "thirst")
    out["old_minus_drives2_thirst"] = dict(mean=diff(list(SEEDS)), ci=boot(diff, SEEDS))
    rd = lambda ss: share("old", ss) - share("drives2", ss)
    out["old_minus_drives2_dead"] = dict(mean=rd(list(SEEDS)), ci=boot(rd, SEEDS))
    m1 = out["drives2"]["thirst"]
    sd = out["drives2"]["stable_drinkers"]
    if m1 <= 0.05 and sd is not None and sd >= 0.90:
        v = "YES"
    elif out["old_minus_drives2_thirst"]["ci"][0] > 0:
        v = "PARTIAL"
    else:
        v = "NO"
    out["VERDICT"] = v
    # reproducibility of the old arms against mindL5 stage B (same seeds)
    rep = {}
    for arm, f in (("old", "lethal_off"), ("rules", "lethal_rules")):
        same = 0
        for s in SEEDS:
            o = json.loads((OLD / f"{f}_{s}.json").read_text())["deaths"]
            same += int(o == D[arm][s]["deaths"])
        rep[arm] = f"{same}/{len(SEEDS)} seeds with identical deaths"
    out["reproduced_vs_mindL5"] = rep
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "diag_a6.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "analyze"])
    a = ap.parse_args()
    if a.stage == "analyze":
        analyze()
    else:
        t0 = time.perf_counter()
        items = [(arm, s) for s in SEEDS for arm in ARMS]
        with mp.get_context("spawn").Pool(2, maxtasksperchild=1) as p:
            for k, r in enumerate(p.imap_unordered(_run, items)):
                print(f"[{k + 1}/{len(items)} {time.perf_counter() - t0:.0f} s] {r}", flush=True)
