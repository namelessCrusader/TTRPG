"""The lethal2 survival check (2026-11-22): a HAZARD MODEL over every focal death, paired by seed. Reads the runs of world all6hl2.

    lab/.venv/bin/python -m lab.mindL.lethal2_report RUNS_NEED RUNS_KIND OUT.json

Replaces the per-seed 1 / years-at-risk measure of lab/mindL/lethal_report.py (which gave a death at night 10 a rate of 36 and was dominated by early
deaths). Here every life contributes its time at risk and whether it died.

 PRIMARY  Cox proportional hazards, STRATIFIED BY SEED (the pair of lives of one seed share a stratum, so the seed's cast, weather and luck cancel),
          one binary covariate (oracle = 1, rules = 0), time = nights lived, a life that does not die is censored at the end. Partial likelihood with
          the Breslow treatment of ties. The hazard ratio is exp(beta); its 95 percent interval is the PROFILE-LIKELIHOOD interval (chi-square 3.84);
          the Wald interval and a bootstrap over seeds are shown beside it.
 CHECK    the Mantel-Haenszel rate ratio with person-time, strata = seed (Greenland-Robins variance).
 CAST     every person of every run: deaths, person-years, a Poisson (Garwood) interval, by cause and by who delivered the blow.

PASS RULE (PASS_RULE.txt, written before the runs): hazard ratio oracle / rules below 1 with the 95 percent interval excluding 1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.optimize as opt
import scipy.stats as st

NPY = 365.0
NAMES = ["Tom", "Sir Ada", "Oren", "Fenn", "Marta"]
RAIDERS = {"a raider", "the wolf"}


def load(path):
    R = {}
    for f in Path(path).glob("*.json"):
        r = json.load(open(f))
        R[(r["chooser"], r["seed"])] = r
    return R


def focal_life(r):
    """(nights at risk, died?, cause, by)."""
    d = [x for x in r["deaths"] if x["name"] == r["focal"]]
    if d:
        return d[0]["night"] + 1, True, d[0]["cause"], d[0].get("by")
    return len(r["err"]), False, None, None


# ------------------------------------------------------------------ the Cox fit
def cox_rows(R, seeds):
    rows = []
    for s in seeds:
        for c, x in (("rules", 0), ("oracle_both", 1)):
            if (c, s) in R:
                n, died, *_ = focal_life(R[(c, s)])
                rows.append((s, n, int(died), x))
    return rows


def loglik(beta, rows):
    """Stratified Cox partial log likelihood (Breslow ties)."""
    ll = 0.0
    for s, t, e, x in rows:
        if not e:
            continue
        risk = [xj for (sj, tj, ej, xj) in rows if sj == s and tj >= t]
        ll += beta * x - np.log(sum(np.exp(beta * xj) for xj in risk))
    return ll


def info(beta, rows):
    """Observed information: the variance of the covariate in each risk set at each death, summed."""
    v = 0.0
    for s, t, e, x in rows:
        if not e:
            continue
        w = np.array([np.exp(beta * xj) for (sj, tj, ej, xj) in rows if sj == s and tj >= t])
        xs = np.array([xj for (sj, tj, ej, xj) in rows if sj == s and tj >= t], float)
        p = (w * xs).sum() / w.sum()
        v += (w * (xs - p) ** 2).sum() / w.sum()
    return v


def cox(rows):
    """-> dict(beta, hr, profile CI, Wald CI, events by arm). Needs a death in each arm to be finite."""
    d1 = sum(e for s, t, e, x in rows if x == 1)
    d0 = sum(e for s, t, e, x in rows if x == 0)
    out = {"events_oracle": d1, "events_rules": d0, "n_strata": len({s for s, *_ in rows})}
    if d1 == 0 or d0 == 0:
        out.update(hr=None, hr_profile=None, hr_wald=None)
        return out
    res = opt.minimize_scalar(lambda b: -loglik(b, rows), bounds=(-8, 8), method="bounded")
    b = float(res.x)
    l0 = loglik(b, rows)
    crit = st.chi2.ppf(0.95, 1) / 2.0

    def edge(sign):
        f = lambda z: l0 - loglik(z, rows) - crit
        lo, hi = (b, b + 8) if sign > 0 else (b - 8, b)
        try:
            return float(opt.brentq(f, lo, hi))
        except ValueError:
            return float("nan")
    lo, hi = edge(-1), edge(+1)
    se = 1.0 / np.sqrt(info(b, rows))
    out.update(beta=b, se=float(se), hr=float(np.exp(b)), hr_profile=[float(np.exp(lo)), float(np.exp(hi))],
               hr_wald=[float(np.exp(b - 1.96 * se)), float(np.exp(b + 1.96 * se))],
               p_lr=float(st.chi2.sf(2 * (l0 - loglik(0.0, rows)), 1)))
    return out


def boot(rows, n=2000, seed=1):
    """The hazard ratio over resamples of the seeds (whole strata)."""
    rng = np.random.default_rng(seed)
    by = {}
    for r in rows:
        by.setdefault(r[0], []).append(r)
    keys = list(by)
    hrs = []
    for _ in range(n):
        pick, k = [], 0
        for s in rng.choice(keys, len(keys)):
            k += 1
            pick += [(k, t, e, x) for (_, t, e, x) in by[s]]
        c = cox(pick)
        if c["hr"] is not None:
            hrs.append(c["hr"])
    if len(hrs) < n * 0.8:
        return None
    return [float(np.percentile(hrs, 2.5)), float(np.percentile(hrs, 97.5)), len(hrs)]


# ------------------------------------------------------------------ Mantel-Haenszel rate ratio
def mh(rows):
    by = {}
    for s, t, e, x in rows:
        a = by.setdefault(s, {0: [0, 0.0], 1: [0, 0.0]})
        a[x][0] += e
        a[x][1] += t / NPY
    num = den = vn = 0.0
    P = Q = R_ = S = 0.0
    for s, a in by.items():
        d1, t1, d0, t0 = a[1][0], a[1][1], a[0][0], a[0][1]
        T = t1 + t0
        if T == 0:
            continue
        R1, R0 = d1 * t0 / T, d0 * t1 / T
        num += R1
        den += R0
        D = d1 + d0
        P += D * t1 * t0 / T ** 2
    if num == 0 or den == 0:
        return {"rr": None}
    rr = num / den
    se = np.sqrt(P / (num * den))
    return {"rr": float(rr), "ci": [float(rr * np.exp(-1.96 * se)), float(rr * np.exp(1.96 * se))]}


# ------------------------------------------------------------------ survival and the cast
def km(lives, grid):
    out = []
    for g in grid:
        s = 1.0
        for t in sorted({n for n, d in lives if d and n <= g}):
            at = sum(1 for n, _ in lives if n >= t)
            dd = sum(1 for n, d in lives if d and n == t)
            s *= 1 - dd / at
        out.append(s)
    return out


def garwood(k, py):
    """A Poisson count k over py person-years: rate per 1000 and the exact 95 percent interval."""
    lo = 0.0 if k == 0 else st.chi2.ppf(0.025, 2 * k) / 2
    hi = st.chi2.ppf(0.975, 2 * (k + 1)) / 2
    return [1000 * k / py, 1000 * lo / py, 1000 * hi / py]


def cast_table(R):
    py, deaths = 0.0, []
    for r in R.values():
        for nm in NAMES:
            d = [x for x in r["deaths"] if x["name"] == nm]
            py += (d[0]["night"] + 1 if d else len(r["err"])) / NPY
            if d:
                deaths.append(d[0])
    by_cause = {}
    for d in deaths:
        c = d["cause"]
        if c == "violence":
            b = d.get("by")
            c = "violence: a raid" if b in RAIDERS else ("violence: the bandit" if b == "the bandit" else "violence: unknown killer" if b is None else "violence: a fight in the inn")
        by_cause[c] = by_cause.get(c, 0) + 1
    return {"person_years": py, "deaths": len(deaths), "by_cause": by_cause,
            "laid_at_death": sum(1 for d in deaths if d["laid"]),
            "rate": garwood(len(deaths), py), "rates": {k: garwood(v, py) for k, v in by_cause.items()}}


def setting(R, label):
    seeds = sorted({s for _, s in R})
    rows = cox_rows(R, seeds)
    o = {"label": label, "seeds": len(seeds), "cox": cox(rows), "boot": boot(rows), "mh": mh(rows)}
    for c in ("rules", "oracle_both"):
        L = [focal_life(R[(c, s)]) for s in seeds if (c, s) in R]
        o[c] = {"deaths": sum(1 for v in L if v[1]), "person_years": sum(v[0] for v in L) / NPY,
                "km": dict(zip((100, 200, 365, 500, 730), km([(v[0], v[1]) for v in L], (100, 200, 365, 500, 730)))),
                "causes": {k: sum(1 for v in L if v[1] and v[2] == k) for k in {v[2] for v in L if v[1]}}}
    o["cast_all"] = cast_table(R)
    o["cast_rules"] = cast_table({k: v for k, v in R.items() if k[0] == "rules"})
    o["cast_oracle"] = cast_table({k: v for k, v in R.items() if k[0] == "oracle_both"})
    return o


def f2(x, d=2):
    return "n/a" if x is None else f"{x:.{d}f}"


def show(o):
    c, L = o["cox"], [f"== {o['label']} ({o['seeds']} seeds)"]
    for k in ("rules", "oracle_both"):
        x = o[k]
        L.append(f"{k:12s} focal deaths {x['deaths']:2d} in {x['person_years']:6.1f} person-years; survival at 100/200/365/500/730 nights "
                 + " ".join(f"{v:.2f}" for v in x["km"].values()) + f"; causes {x['causes']}")
    if c["hr"] is None:
        L.append(f"Cox (stratified by seed): no deaths in one arm (oracle {c['events_oracle']}, rules {c['events_rules']}): the hazard ratio is not finite")
    else:
        L.append(f"Cox stratified by seed: hazard ratio oracle/rules {c['hr']:.2f}, 95% profile CI [{c['hr_profile'][0]:.2f}, {c['hr_profile'][1]:.2f}]; Wald "
                 f"[{c['hr_wald'][0]:.2f}, {c['hr_wald'][1]:.2f}]; seed bootstrap " + ("n/a" if o["boot"] is None else f"[{o['boot'][0]:.2f}, {o['boot'][1]:.2f}]")
                 + f"; LR p = {c['p_lr']:.2f}; events oracle {c['events_oracle']}, rules {c['events_rules']}")
    m = o["mh"]
    L.append("Mantel-Haenszel rate ratio with person-time (strata = seed): " + ("n/a" if m["rr"] is None else f"{m['rr']:.2f} [{m['ci'][0]:.2f}, {m['ci'][1]:.2f}]"))
    if c["hr"] is not None:
        met = c["hr"] < 1 and c["hr_profile"][1] < 1
        L.append("PASS RULE: " + ("MET" if met else "NOT MET") + (": the interval is wide (upper end %.2f), underpowered" % c["hr_profile"][1] if not met and c["hr_profile"][1] > 1.5 else ""))
    else:
        L.append("PASS RULE: NOT MET (no finite estimate)")
    for key, name in (("cast_all", "whole cast, both choosers"), ("cast_rules", "whole cast, rules arm")):
        t = o[key]
        L.append(f"{name}: {t['deaths']} deaths in {t['person_years']:.1f} person-years = {t['rate'][0]:.1f} per 1000 py [{t['rate'][1]:.1f}, {t['rate'][2]:.1f}]; laid up at death {t['laid_at_death']}")
        for k, v in sorted(t["rates"].items()):
            L.append(f"     {k:30s} {t['by_cause'][k]:3d} deaths = {v[0]:7.1f} per 1000 py [{v[1]:.1f}, {v[2]:.1f}]")
    return "\n".join(L)


def main():
    d = {}
    for path, label in zip(sys.argv[1:3], ("P-need Fenn, all6hl2", "P-kind Fenn, all6hl2")):
        if not Path(path).exists() or not list(Path(path).glob("*.json")):
            continue
        o = setting(load(path), label)
        d[label] = o
        print(show(o) + "\n")
    Path(sys.argv[3]).write_text(json.dumps(d, indent=1, default=str))


if __name__ == "__main__":
    main()
