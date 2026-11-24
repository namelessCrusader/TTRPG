"""mindL6 stage B2: THE POPULATION (rule: PASS_RULE_B2.txt, written before any run).
    lab/.venv/bin/python -m lab.mindL.b2 run <arm|childhoods>   (2 workers, resumable; arms: rules naive child full oldchild)
    lab/.venv/bin/python -m lab.mindL.b2 analyze
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
SCR = Path("/home/nisargparikh/.cache/mark1-build/mindL6/B2")
CH = SCR / "childhoods"
RES = Path(__file__).resolve().parents[1] / "results" / "mindL6" / "B2"
SEEDS = tuple(range(11, 26))
NIGHTS = 90
CHILD_NIGHTS = 30
PEOPLE = ["Jon", "Mara", "Pip", "Gran", "Tom", "Sir Ada", "Oren"]
FAM = ["Jon", "Mara", "Pip", "Gran"]
LOD = ["Tom", "Sir Ada", "Oren"]
ARMS = ("rules", "naive", "child", "full", "oldchild")
SCBD = dict(drives2="SCBD", d2={"dial": True}, taste=True)
OLD = dict(drives2=False, d2=None, watch_age=False, taste=False)
# the childhood of each variant: (tag, learner_cfg for the childhood, teach, world flags of the childhood)
VAR = {"child": ("C", dict(drives2="SCBD", watch_age=False), False, ("drives2", "body2")),
       "full": ("F", {}, True, ("drives2", "body2")),
       "oldchild": ("O", OLD, False, ("body2",))}


def _child(a):
    arm, seed, person = a
    from lab.mindL import childhood as C
    tag, cfg, teach, wf = VAR[arm]
    r = C.run_childhood(seed, person, CHILD_NIGHTS, learner_cfg=cfg, tag=tag, teach=teach, wflags=wf, scr=CH)
    return a, r["seconds"]


def _adult(a):
    arm, seed = a
    out = SCR / f"b2_{arm}_{seed}.json"
    if out.exists():
        return a, "kept"
    from lab.mindL import childhood as C
    from lab.mindL import household as H
    pre = None
    if arm in VAR:
        tag = VAR[arm][0]
        pre = {p: pickle.loads((CH / f"child{tag}_{seed}_{p.replace(' ', '_')}_{CHILD_NIGHTS}.pkl").read_bytes())["blob"] for p in PEOPLE}
    if arm == "rules":
        r = H.run_full(seed, NIGHTS, arm="rules", lethal=True, res=True, drinks=True, flows=True, wflags=("body2",))
    else:
        extra = {"naive": SCBD, "child": SCBD, "full": C.ADULT, "oldchild": {}}[arm]
        r = H.run_full(seed, NIGHTS, arm="off", lethal=True, res=True, drinks=True, flows=True, extra=extra, wflags=("body2",), preload=pre)
    H.dump(dict(deaths=r["deaths"], res=r["res"], drinks=r["drinks"], flows=r["flows"], seconds=r["seconds"]), out)
    return a, r["seconds"]


def run(what):
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    if what == "childhoods":
        items = [(arm, s, p) for arm in VAR for s in SEEDS for p in PEOPLE]
        fn = _child
    else:
        items, fn = [(what, s) for s in SEEDS], _adult
    with ctx.Pool(2, maxtasksperchild=4) as pool:
        for k, r in enumerate(pool.imap_unordered(fn, items)):
            print(f"[{what} {k + 1}/{len(items)} {time.perf_counter() - t0:.0f} s] {r}", flush=True)


# ---------------------------------------------------------------------------------------------- analysis
RNG = np.random.default_rng(2026)
BOOT = [RNG.integers(0, len(SEEDS), len(SEEDS)) for _ in range(2000)]     # one fixed set of seed resamples for every interval (paired)


def ci(v):
    """Mean of per-seed values and the seed-bootstrap 95% interval."""
    v = np.asarray(v, float)
    b = [np.nanmean(v[ix]) for ix in BOOT]
    return float(np.nanmean(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def fm(t, d=3):
    return f"{t[0]:+.{d}f} [{t[1]:+.{d}f}, {t[2]:+.{d}f}]"


def fp(t, d=3):
    return f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"


def load(arm):
    return {s: json.loads((SCR / f"b2_{arm}_{s}.json").read_text()) for s in SEEDS}


CARE_N = 500


def seed_stats(r):
    d = r["deaths"]
    res = r["res"]
    N = len(res)
    dead_th = sum(1 for v in d.values() if v["cause"] == "thirst")
    dead_hu = sum(1 for v in d.values() if v["cause"] == "hunger")
    dep = []
    burn = cap = alive = low = 0.0
    for n in range(N):
        for m in PEOPLE:
            x = res[n][m]
            dep.append(1.0 if x["dead"] else 1.0 - x["min"])
            if not x["dead"]:
                alive += 1
                burn += x["burn"]
                cap += x["caph"]
                low += int(x["min"] < 0.5)
    fl = r["flows"]
    fa = sum(fl.get(f"{a}>{t}", 0) for a in FAM for t in FAM if a != t)
    lo = sum(fl.get(f"{a}>{t}", 0) for a in FAM for t in LOD)
    return dict(dead=len(d) / 7.0, thirst=dead_th / 7.0, hunger=dead_hu / 7.0, other=(len(d) - dead_th - dead_hu) / 7.0, dep=float(np.mean(dep)), burn=burn / max(1, alive),
                x1=cap / max(1, alive), x2=low / max(1, alive), fam=fa, lod=lo, share=fa / (fa + lo) if fa + lo else float("nan"),
                first=min([v["night"] for v in d.values()] or [NIGHTS + 1]), allDead=int(len(d) == 7), sec=r["seconds"],
                drinks=sum(sum(day.values()) for day in r["drinks"]) / max(1, alive))


def analyze():
    A = {a: load(a) for a in ARMS}
    S = {a: {s: seed_stats(A[a][s]) for s in SEEDS} for a in ARMS}
    col = lambda a, k: np.array([S[a][s][k] for s in SEEDS], float)
    out, J = [], {}
    P = out.append
    P("MINDL6 B2 TABLES. 15 seeds (11-25), 105 lives an arm, lethal family inn, 90 nights, body2. Intervals: seed bootstrap 95 percent.")
    P("")
    P("M1 DEATHS (share of 105 lives; thirst share per seed mean with interval; median first death night of 15 households, none = 91)")
    P(f"{'arm':9s} {'dead':>7s} {'thirst':>7s} {'hunger':>7s} {'other':>7s}  thirst per seed (CI)          dead per seed (CI)            median first  households all dead")
    for a in ARMS:
        causes = {}
        for s in SEEDS:
            for v in A[a][s]["deaths"].values():
                causes[v["cause"]] = causes.get(v["cause"], 0) + 1
        th, de = ci(col(a, "thirst")), ci(col(a, "dead"))
        P(f"{a:9s} {col(a, 'dead').mean():7.3f} {col(a, 'thirst').mean():7.3f} {col(a, 'hunger').mean():7.3f} {col(a, 'other').mean():7.3f}  {fp(th):28s}  {fp(de):28s}  {np.median(col(a, 'first')):5.0f}   {int(col(a, 'allDead').sum())}   causes {causes}")
        J[a] = dict(dead=de, thirst=th, causes=causes, median_first=float(np.median(col(a, "first"))))
    P("")
    P("M2 THE FOOD STEP (per household, mean over seeds; dead night counts depletion 1)")
    P(f"{'arm':9s} {'depletion':28s} {'burn/living night':28s} {'X1 hunger-cap turns/night':28s} {'X2 share nights min<0.5':28s} drinks/living night")
    for a in ARMS:
        t = {k: ci(col(a, k)) for k in ("dep", "burn", "x1", "x2")}
        P(f"{a:9s} {fp(t['dep'], 4):28s} {fp(t['burn'], 4):28s} {fp(t['x1'], 3):28s} {fp(t['x2'], 4):28s} {col(a, 'drinks').mean():.2f}")
        J[a].update({k: v for k, v in t.items()})
    P("")
    P("PAIRED DIFFERENCES (first minus second, by seed). A difference EXCLUDES 0 when its interval does.")
    pairs = [("naive", "rules"), ("child", "rules"), ("full", "rules"), ("oldchild", "rules"), ("child", "naive"), ("full", "child"), ("full", "naive"), ("child", "oldchild"), ("naive", "oldchild")]
    for k, nm in (("thirst", "thirst share"), ("dead", "dead share"), ("dep", "depletion"), ("burn", "burn"), ("x1", "X1"), ("x2", "X2")):
        P(f"  {nm}")
        for x, y in pairs:
            t = ci(col(x, k) - col(y, k))
            mark = "excludes 0" if (t[1] > 0 or t[2] < 0) else "includes 0"
            P(f"    {x:9s} minus {y:9s} {fm(t, 4):34s} {mark}")
            J[f"d_{k}_{x}_{y}"] = t
    P("")
    P(f"M3 KIN CARE (care acts of family actors; family targets 3, lodger targets 3: chance 0.5; testable if at least {CARE_N} family-actor acts pooled)")
    for a in ARMS:
        F_ = sum(S[a][s]["fam"] for s in SEEDS)
        L_ = sum(S[a][s]["lod"] for s in SEEDS)
        sh = ci(np.array([S[a][s]["share"] for s in SEEDS]))
        fam = np.array([S[a][s]["fam"] for s in SEEDS], float)
        lod = np.array([S[a][s]["lod"] for s in SEEDS], float)
        rb = [fam[ix].sum() / max(1.0, lod[ix].sum()) for ix in BOOT]
        ratio = F_ / L_ if L_ else float("nan")
        rc = (ratio, float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5)))
        ok = F_ + L_ >= CARE_N
        P(f"  {a:9s} family acts {F_:6d}, lodger acts {L_:6d}; share {fp(sh)}; ratio {fp(rc, 2)}; {'testable' if ok else 'NOT TESTABLE'}")
        J[a].update(share=sh, ratio=rc, fam_acts=F_, lod_acts=L_, testable=ok)
    for a in ARMS[1:]:
        t = ci(np.array([S[a][s]["share"] - S["rules"][s]["share"] for s in SEEDS]))
        P(f"  share {a} minus rules (paired) {fm(t)}")
        J[f"d_share_{a}_rules"] = t
    P("")
    P("THE OWNER'S BAR (AQ4) for the FULL MIND")
    dep, dd = J["d_dep_full_rules"], J["d_dead_full_rules"]
    f = J["full"]
    bdep = dep[2] < 0
    bdead = dd[1] <= 0
    bkin = f["testable"] and f["ratio"][1] > 1 and f["share"][1] > 0.5
    P(f"  B-DEP  depletion full minus rules {fm(dep, 4)}: interval entirely below 0? {bdep}")
    P(f"  B-DEAD dead share full minus rules {fm(dd, 4)}: lower bound at or below 0? {bdead}")
    P(f"  B-KIN  ratio {fp(f['ratio'], 2)} (lower > 1) and share {fp(f['share'])} (lower > 0.5), testable {f['testable']}: {bkin}")
    verdict = "YES" if (bdep and bdead and bkin) else "NO"
    P(f"  VERDICT: the full mind beats the rules: {verdict}")
    J["bar"] = dict(bdep=bool(bdep), bdead=bool(bdead), bkin=bool(bkin), verdict=verdict)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "tables.txt").write_text("\n".join(out))
    (RES / "tables.json").write_text(json.dumps(J, default=float))
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "analyze"])
    ap.add_argument("what", nargs="?", default="")
    a = ap.parse_args()
    analyze() if a.stage == "analyze" else run(a.what)
