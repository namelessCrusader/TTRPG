"""The lethal world check (2026-11-21, owner D1 lever (c)): judge by SURVIVAL. Reads the runs of world all6hl.

    lab/.venv/bin/python -m lab.mindL.lethal_report RUNS_NEED RUNS_KIND OUT.json

Per setting (P-need Fenn, then the old P-kind Fenn): the focal death rate per person-year (a night is a day), the survival curve
(Kaplan-Meier over the seeds), the hazard ratio oracle / rules (pooled rate ratio, 95% interval from resampling the seeds) and the paired
per-seed difference of the rates (t interval over the 10 seeds), the night-averaged drive error with death counted (a night the focal is
dead at its start counts the WORST error its nature can score, the weights times (1 - set point); the night it dies counts as measured),
the same over the nights both alive (the continuity gap), and the deaths of the whole cast (every person of every run).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.stats as st

SEEDS = range(1, 11)
NPY = 365.0


def ci(x):
    x = np.asarray([v for v in x if v == v], float)
    if len(x) < 2:
        return [float(x.mean()) if len(x) else float("nan"), None, None]
    h = st.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h)]


def fm(c, d=3):
    return "n/a" if c[1] is None else f"{c[0]:+.{d}f} [{c[1]:+.{d}f}, {c[2]:+.{d}f}]"


def load(path):
    R = {}
    for f in Path(path).glob("*.json"):
        r = json.load(open(f))
        R[(r["chooser"], r["seed"])] = r
    return R


def focal_life(r):
    """(nights at risk, died?, night of death, cause)."""
    d = [x for x in r["deaths"] if x["name"] == r["focal"]]
    if d:
        return d[0]["night"] + 1, True, d[0]["night"], d[0]["cause"]
    return len(r["err"]), False, None, None


def km(lives, grid):
    """Kaplan-Meier survival at the nights in grid; lives: [(nights at risk, died)]."""
    out = []
    for g in grid:
        s = 1.0
        for t in sorted({n for n, d in lives if d and n <= g}):
            at = sum(1 for n, _ in lives if n >= t)
            dd = sum(1 for n, d in lives if d and n == t)
            s *= 1 - dd / at
        out.append(s)
    return out


def setting(R, label):
    out = {"label": label}
    ch = sorted({c for c, _ in R})
    for c in ch:
        L = {s: focal_life(R[(c, s)]) for s in SEEDS if (c, s) in R}
        deaths = sum(1 for n, d, *_ in L.values() if d)
        py = sum(n for n, *_ in L.values()) / NPY
        per_seed = {s: (1.0 if v[1] else 0.0) / (v[0] / NPY) for s, v in L.items()}
        out[c] = {"deaths": deaths, "person_years": py, "rate_pooled": deaths / py, "lives": L,
                  "km": dict(zip((100, 200, 365, 500, 730), km([(v[0], v[1]) for v in L.values()], (100, 200, 365, 500, 730)))),
                  "causes": sorted([(v[3], v[2]) for v in L.values() if v[1]], key=str)}
    if "rules" in out and "oracle_both" in out:
        ss = [s for s in SEEDS if ("rules", s) in R and ("oracle_both", s) in R]
        a, b = out["rules"]["lives"], out["oracle_both"]["lives"]

        def hr(idx):
            do = sum(b[s][1] for s in idx); po = sum(b[s][0] for s in idx)
            dr = sum(a[s][1] for s in idx); pr = sum(a[s][0] for s in idx)
            return (do / po) / (dr / pr) if dr and do else (float("nan") if not dr else 0.0)
        out["hr"] = hr(ss)
        rng = np.random.default_rng(1)
        boots = [hr(list(rng.choice(ss, len(ss)))) for _ in range(4000)]
        boots = [x for x in boots if x == x]
        out["hr_ci"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), len(boots)]
        diff = [(out["rules"]["rate_pooled"] and (1.0 if a[s][1] else 0.0) / (a[s][0] / NPY)) - (1.0 if b[s][1] else 0.0) / (b[s][0] / NPY) for s in ss]
        out["rate_diff"] = ci(diff)                                    # rules minus oracle, per person-year, paired by seed
        out["rate_diff_death_free"] = sum(1 for d in diff if d != 0)
        # error with death counted
        def err_counted(r, alive_only=False):
            n_at, died, nd, _ = focal_life(r)
            e = r["err"][r["win0"]:]
            mx = r["max_err"][r["focal"]]
            tot = []
            for i, x in enumerate(e):
                n = r["win0"] + i
                if died and n > nd:
                    tot.append(mx)
                else:
                    tot.append(x)
            return float(np.mean(tot))
        out["err_counted"] = {c: ci([err_counted(R[(c, s)]) for s in ss]) for c in ("rules", "oracle_both")}
        out["gap_counted"] = ci([err_counted(R[("rules", s)]) - err_counted(R[("oracle_both", s)]) for s in ss])
        gaps = []
        for s in ss:
            r0, r1 = R[("rules", s)], R[("oracle_both", s)]
            n0, n1 = focal_life(r0), focal_life(r1)
            last = min(n0[0] if n0[1] else 10 ** 9, n1[0] if n1[1] else 10 ** 9, len(r0["err"]))
            idx = range(r0["win0"], min(last, len(r0["err"])))
            if len(idx):
                gaps.append(np.mean([r0["err"][i] - r1["err"][i] for i in idx]))
        out["gap_alive"] = ci(gaps)
    # the whole cast, every run
    cast = {}
    for (c, s), r in R.items():
        names = ["Tom", "Sir Ada", "Oren", "Fenn", "Marta"]
        for nm in names:
            d = [x for x in r["deaths"] if x["name"] == nm]
            n_at = d[0]["night"] + 1 if d else len(r["err"])
            a = cast.setdefault(nm, {"deaths": 0, "py": 0.0, "causes": {}, "laid_at_death": 0})
            a["py"] += n_at / NPY
            if d:
                a["deaths"] += 1
                a["causes"][d[0]["cause"]] = a["causes"].get(d[0]["cause"], 0) + 1
                a["laid_at_death"] += bool(d[0]["laid"])
    tot_d = sum(a["deaths"] for a in cast.values()); tot_py = sum(a["py"] for a in cast.values())
    out["cast"] = {"per_person": cast, "deaths": tot_d, "person_years": tot_py, "rate_per_1000": 1000 * tot_d / tot_py}
    cause = {}
    for a in cast.values():
        for k, v in a["causes"].items():
            cause[k] = cause.get(k, 0) + v
    out["cast"]["causes"] = cause
    out["cast"]["laid_at_death"] = sum(a["laid_at_death"] for a in cast.values())
    laid = sum(sum(r["laid_nights"].values()) for r in R.values()); nn = sum(5 * len(r["err"]) for r in R.values())
    out["laid_share_of_person_nights"] = laid / nn
    return out


def show(o):
    L = [f"== {o['label']}"]
    for c in ("rules", "oracle_both"):
        if c in o:
            x = o[c]
            L.append(f"{c:12s} focal deaths {x['deaths']} in {x['person_years']:.1f} person-years: {x['rate_pooled']:.3f} per person-year; survival at 100/200/365/500/730 nights: " +
                     " ".join(f"{v:.2f}" for v in x["km"].values()) + f"; causes {x['causes']}")
    if "hr" in o:
        L.append(f"hazard ratio oracle/rules (pooled, seed bootstrap) {o['hr']:.2f} [{o['hr_ci'][0]:.2f}, {o['hr_ci'][1]:.2f}] ({o['hr_ci'][2]} usable resamples)")
        L.append(f"paired rate difference rules minus oracle (deaths per person-year, t over seeds) {fm(o['rate_diff'])}")
        L.append("drive error with death counted: " + ", ".join(f"{c} {fm(v)}" for c, v in o["err_counted"].items()) + f"; gap rules - oracle {fm(o['gap_counted'])}")
        L.append(f"old oracle gap (nights both alive, error only) {fm(o['gap_alive'])}")
    c = o["cast"]
    L.append(f"whole cast: {c['deaths']} deaths in {c['person_years']:.1f} person-years = {c['rate_per_1000']:.1f} per 1000 person-years; causes {c['causes']}; laid up at death {c['laid_at_death']}")
    for n, a in c["per_person"].items():
        L.append(f"   {n:8s} deaths {a['deaths']:3d} in {a['py']:6.1f} py = {a['deaths'] / a['py']:.3f} /py  causes {a['causes']}")
    L.append(f"share of person-nights laid up {o['laid_share_of_person_nights']:.3f}")
    return "\n".join(L)


def main():
    d = {}
    for path, label in zip(sys.argv[1:3], ("P-need Fenn, all6hl", "P-kind Fenn, all6hl")):
        o = setting(load(path), label)
        d[label] = o
        print(show(o) + "\n")
    Path(sys.argv[3]).write_text(json.dumps(d, indent=1, default=str))


if __name__ == "__main__":
    main()
