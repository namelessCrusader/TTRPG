"""Stage B tables (mindL5). lab/.venv/bin/python -m lab.mindL.analyze_b  -> prints, writes tables.txt and tables.json in results/mindL5/B"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

SCR = Path("/home/nisargparikh/.cache/mark1-build/mindL5/B")
RES = Path(__file__).resolve().parents[1] / "results" / "mindL5" / "B"
SEEDS = list(range(11, 41))
FAM = ["Jon", "Mara", "Pip", "Gran"]
LOD = ["Tom", "Sir Ada", "Oren"]
ARMS = ["rules", "off", "past"]
RNG = np.random.default_rng(5)


def load(kind, arm, s):
    p = SCR / f"{kind}_{arm}_{s}.json"
    return json.loads(p.read_text()) if p.exists() else None


def tci(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return m, m - h, m + h


def boot(x, f, B=2000):
    x = np.asarray(x, float)
    v = [f(x[RNG.integers(0, len(x), len(x))]) for _ in range(B)]
    return np.percentile(v, [2.5, 97.5])


def fm(t):
    return f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"


def acts(fl, A, T):
    return sum(fl.get(f"{a}>{t}", 0) for a in A for t in T if a != t)


def care(arm):
    rows = []
    for s in SEEDS:
        r = load("flow", arm, s)
        if r is None:
            continue
        fl = r["flows"]
        fam = acts(fl, FAM, FAM)
        lod = acts(fl, FAM, LOD)
        lf = acts(fl, LOD, FAM)          # lodgers at family (4 targets), lodgers at lodgers (2 targets)
        ll = acts(fl, LOD, LOD)
        rows.append(dict(seed=s, fam=fam, lod=lod, share=fam / (fam + lod) if fam + lod else np.nan, lf=lf, ll=ll,
                         lshare=(lf / 4) / (lf / 4 + ll / 2) if lf + ll else np.nan, fl=fl))
    return rows


def recip(rows):
    """Slope of help i>j on help j>i over ordered pairs: kin (both family) and other pairs, pooled over seeds."""
    names = FAM + LOD
    out = {}
    for lab, ok in (("kin", lambda i, j: i in FAM and j in FAM), ("other", lambda i, j: not (i in FAM and j in FAM))):
        xs, ys = [], []
        for r in rows:
            for i in names:
                for j in names:
                    if i != j and ok(i, j):
                        xs.append(r["fl"].get(f"{j}>{i}", 0))
                        ys.append(r["fl"].get(f"{i}>{j}", 0))
        xs, ys = np.array(xs, float), np.array(ys, float)
        if xs.std() > 0:
            sl = stats.linregress(xs, ys)
            out[lab] = (sl.slope, sl.stderr, len(xs))
        else:
            out[lab] = (float("nan"), float("nan"), len(xs))
    return out


def life(r, name):
    res = r["res"]
    N = len(res)
    dep = burn = cap = 0.0
    for n in range(N):
        x = res[n][name]
        dep += 1.0 if x["dead"] else 1.0 - x["min"]
        burn += x["burn"]
        cap += x["caph"]
    return dict(dep=dep / N, burn=burn / N, cap=cap / N, dead=int(res[-1][name]["dead"]))


def surv(arm):
    rows = []
    for s in SEEDS:
        r = load("lethal", arm, s)
        if r is None:
            continue
        d = {k: {m: life(r, m)[k] for m in FAM + LOD} for k in ("dep", "burn", "cap", "dead")}
        rows.append(dict(seed=s, d=d, deaths=r["deaths"], sec=r["seconds"]))
    return rows


def grp(row, k, G):
    return float(np.mean([row["d"][k][m] for m in G]))


def main():
    out, J = [], {}
    P = out.append
    C = {a: care(a) for a in ARMS}
    P("MINDL5 STAGE B TABLES (CIs 95%)")
    P("(a) KIN CARE (non-lethal family inn, 45 nights)")
    for a in ARMS:
        R = C[a]
        if not R:
            continue
        sh = [r["share"] for r in R if not np.isnan(r["share"])]
        F_, L_ = sum(r["fam"] for r in R), sum(r["lod"] for r in R)
        ratio = F_ / L_ if L_ else float("nan")
        f = lambda idx, R=R: sum(R[i]["fam"] for i in idx) / max(1, sum(R[i]["lod"] for i in idx))
        idx = np.arange(len(R))
        bs = [f(RNG.integers(0, len(R), len(R))) for _ in range(2000)]
        ls = [r["lshare"] for r in R if not np.isnan(r["lshare"])]
        rc = recip(R)
        P(f"  {a:6s} n={len(R)} seeds  family acts {F_}, lodger acts {L_}; family actors: mean share {fm(tci(sh))}; pooled ratio {ratio:.2f} [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]"
          f"; lodger actors' per-target share to family {fm(tci(ls))}")
        P(f"         reciprocity slope kin {rc['kin'][0]:+.3f} (se {rc['kin'][1]:.3f}, n {rc['kin'][2]}), other {rc['other'][0]:+.3f} (se {rc['other'][1]:.3f}, n {rc['other'][2]})")
        J[f"care_{a}"] = dict(share=tci(sh), ratio=ratio, ratio_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], lshare=tci(ls), recip=rc)
    for b in ("rules", "off"):
        d = {r["seed"]: r["share"] for r in C[b]}
        x = [r["share"] - d[r["seed"]] for r in C["past"] if r["seed"] in d and not np.isnan(r["share"]) and not np.isnan(d[r["seed"]])]
        P(f"  share past minus {b}, paired: {fm(tci(x))} (n {len(x)})")
        J[f"care_diff_{b}"] = tci(x)
    P("(b)/(c) SURVIVAL (lethal + perma family inn, 90 nights)")
    S = {a: surv(a) for a in ARMS}
    for a in ARMS:
        R = S[a]
        if not R:
            continue
        for k, nm in (("dep", "depletion"), ("burn", "burn (reserve days a night)"), ("cap", "extreme-hunger turns a night"), ("dead", "dead at 90 nights")):
            fa = [grp(r, k, FAM) for r in R]
            lo = [grp(r, k, LOD) for r in R]
            al = [grp(r, k, FAM + LOD) for r in R]
            dd = np.array(fa) - np.array(lo)
            bsd = boot(dd, np.mean)
            P(f"  {a:6s} {nm:30s} family {np.mean(fa):.4f}  lodgers {np.mean(lo):.4f}  all {np.mean(al):.4f}  family-lodger {fm(tci(dd))} boot [{bsd[0]:+.4f}, {bsd[1]:+.4f}] (n {len(R)})")
            J[f"surv_{a}_{k}"] = dict(fam=float(np.mean(fa)), lod=float(np.mean(lo)), all=float(np.mean(al)), gap=tci(dd), gap_boot=list(map(float, bsd)))
        dn = sum(len(r["deaths"]) for r in R)
        P(f"         deaths {dn} in {len(R)} runs of 7; first-death night median {np.median([min([v['night'] for v in r['deaths'].values()], default=90) for r in R]):.0f}; mean run seconds {np.mean([r['sec'] for r in R]):.0f}")
    for b in ("rules", "off"):
        if not S["past"] or not S[b]:
            continue
        d = {r["seed"]: r for r in S[b]}
        for k, nm in (("dep", "depletion"), ("burn", "burn"), ("cap", "extreme-hunger"), ("dead", "dead share")):
            x = [grp(r, k, FAM + LOD) - grp(d[r["seed"]], k, FAM + LOD) for r in S["past"] if r["seed"] in d]
            bs = boot(x, np.mean)
            P(f"  household {nm:15s} past minus {b:5s} paired {fm(tci(x))} boot [{bs[0]:+.4f}, {bs[1]:+.4f}] (n {len(x)})")
            J[f"c_{b}_{k}"] = dict(t=tci(x), boot=list(map(float, bs)))
    (RES / "tables.txt").write_text("\n".join(out))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "tables.json").write_text(json.dumps(J, default=float))
    (RES / "tables.txt").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    RES.mkdir(parents=True, exist_ok=True)
    main()
