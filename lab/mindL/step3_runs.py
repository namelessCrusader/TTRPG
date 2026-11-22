"""Step 3 of SV5 (2026-11-21): lifetimes for the L1/L2 tables and the diagnosis, world all6s, Fenn = P-need(0.9), Oren = P-hard(0.9).

    MINDL_RUNS=DIR lab/.venv/bin/python -m lab.mindL.step3_runs ARM[,ARM] --workers 3 [--seeds 1,2] [--variants A,B,deaf] [--nights 220]

The partners go deaf at night CUT = 200 whatever the length of the life (so a 220-night life is the first 220 nights of the stored
300-night lives: same seed, same world, same decisions; L1 and L2 read blocks up to night 220 only). Arms are the mind's settings.
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

CUT = 200
PAID = 0.02          # the paid world (tests_L.PaidError): each answer from the kind partner is worth this much as a drop of error
ARMS = {
    "fx": dict(effects=True),                                              # step 2's effects arm (reproduces the stored one)
    "fxnoP": dict(effects=True, modules="AKVWEC"),                         # the same without the partner term (diagnosis)
    "fxfix": dict(effects=True, fx_fix=True),                              # the step 3 diagnosis fix
    "fxc": dict(effects=True, fx_fix=True, concerns=True),                 # effects + acquired concerns
    "fxcnoP": dict(effects=True, fx_fix=True, concerns=True, modules="AKVWEC"),
    "fxcP": dict(effects=True, fx_fix=True, concerns=True),                # the same, in the PAID world (see PAID)
    "fxP": dict(effects=True, fx_fix=True),                                # effects only, in the PAID world
    "labels": dict(b_trust=1.0),                                           # the labels control arm (= tests_L mind2: mind.FIXES)
    "rules": None,                                                         # the rules null
}


def specs(arms, seeds, variants, nights, world="all6s"):
    from lab.mindL.tests_L import FOCALS3, FADE
    out = []
    for arm in arms:
        for s in seeds:
            for v in variants:
                f, o = {"A": ("need", "hard"), "Ar": ("need", "hard"), "B": ("hard", "need"), "deaf": ("deaf", "deaf")}[v]      # Ar: life A, the mind's own draws from another seed (D_rng)
                ph = {"Fenn": [[0, f]], "Oren": [[0, o]]}
                if v not in ("deaf",):
                    ph = {k: [*x, [CUT, "deaf"]] for k, x in ph.items()}
                paid = arm in ("fxcP", "fxP")
                out.append(dict(family="L12", variant=v, chooser="rules" if arm == "rules" else "mindx", cfg=ARMS[arm], arm=arm, seed=s, nights=nights, mind_seed=s + 1000 if v == "Ar" else s,
                                **(dict(payer="Fenn" if v in ("A", "Ar") else "Oren", pay=PAID) if paid else {}),
                                focal=FOCALS3[s % 3], reward="paid" if paid else "drives", partners=ph, fade=FADE, flags=world, fenn="need",
                                id=f"L12_{v}_{arm}_seed{s}_n{nights}_{world}_need"))
    return out


def work(spec):
    from lab.mindL.tests_L import _work
    return _work(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--variants", default="A,B")
    ap.add_argument("--nights", type=int, default=220)
    ap.add_argument("--world", default="all6s", choices=["all6s", "all6h"])
    a = ap.parse_args()
    from lab.mindL import tests_L as T
    todo = [s for s in specs(a.arms.split(","), [int(x) for x in a.seeds.split(",")], a.variants.split(","), a.nights, a.world)
            if not (T.RUNS / f"{s['id']}.json").exists()]
    todo.sort(key=lambda s: s["seed"])
    print(f"{len(todo)} lifetimes, runs in {T.RUNS}", flush=True)
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(min(3, a.workers), maxtasksperchild=1) as pool:
        for k, r in enumerate(pool.imap_unordered(work, todo)):
            print(f"[{k + 1}/{len(todo)} {time.perf_counter() - t0:.0f} s] {r[0]}: {r[1]}", flush=True)


if __name__ == "__main__":
    main()
