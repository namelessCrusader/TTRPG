"""The food check (2026-11-22): the inn's FOOD BALANCE, and the continuous SURVIVAL-RISK measure with its pass rule. Reads the runs of a tournament sweep.

    lab/.venv/bin/python -m lab.mindL.food_report balance RUNS_DIR              # where the food goes (works on the old lethal2 runs too)
    lab/.venv/bin/python -m lab.mindL.food_report risk RUNS_NEED [RUNS_KIND] OUT.json

BALANCE per night, from each lifetime's logs: the catches (rations), the counter's refill (the cook's rations; the start of a night minus the end of the one before),
the work yield (end + eaten - start of the counter), the rations eaten (own catch / another's catch / counter), what spoiled (catch - eaten from catches), the need
(NEED_RATIONS a living person a day), who ate how much of the need, and how often a person-day or an inn-day falls short.

RISK (PASS_RULE_FOOD.txt, written before the sweep): the primary measure is DEPLETION, the mean over the nights of 1 - f where f is the lowest reserve fraction of the night
(a dead night counts as 1). The pass rule is paired by seed with a t interval AND a seed bootstrap, both excluding 0; Cox on the focal deaths is secondary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.stats as st

from lab.mindL import lethal2_report as L
from lab.world2.world import FIELD_KCAL_DAY, NEED_RATIONS

NAMES = L.NAMES
NPY = 365.0
RATION_KCAL = FIELD_KCAL_DAY / NEED_RATIONS


def life_nights(r):
    """name -> the nights lived (the death night included); everybody lived len(r['catch']) nights unless they died."""
    N = len(r["catch"])
    dn = {d["name"]: d["night"] for d in r["deaths"]}
    return {n: (min(N, dn[n] + 1) if n in dn else N) for n in NAMES}, N


# ------------------------------------------------------------------ the food balance
def balance(R):
    R = list(R.values()) if isinstance(R, dict) else R
    tot = {k: 0.0 for k in ("nights", "alive", "catch", "refill", "work", "own", "oth", "cnt", "demand", "short_inn", "inn_nights")}
    per = {n: {"days": 0, "intake": 0.0, "own": 0.0, "oth": 0.0, "cnt": 0.0, "lt50": 0, "lt75": 0, "lt100": 0, "laid": 0, "cap": 0, "cap_t": 0, "relief": 0.0, "lt75_laid": 0, "lt75_well": 0}
           for n in NAMES}
    catches, well_catch = [], []
    have_res = all("res" in r and r["res"] for r in R)
    for r in R:
        lived, N = life_nights(r)
        for n in range(N):
            alive = [p for p in NAMES if n < lived[p]]
            tot["nights"] += 1
            tot["alive"] += len(alive)
            tot["demand"] += len(alive) * NEED_RATIONS
            c = r["catch"][n]
            catch = sum(c.values())
            tot["catch"] += catch
            h = r["house"][n]
            start = sum(h[0:2])
            end = sum(h[2:4])
            eaten_cnt = sum(r["fed"][n][p][2] for p in NAMES)
            refill = start - (sum(r["house"][n - 1][2:4]) if n > 0 else 0)
            tot["refill"] += refill if n > 0 else 0
            tot["work"] += max(0, end + eaten_cnt - start)
            tot["cnt"] += eaten_cnt
            tot["own"] += sum(r["fed"][n][p][0] for p in NAMES)
            tot["oth"] += sum(r["fed"][n][p][1] for p in NAMES)
            tot["inn_nights"] += 1
            if n > 0 and catch + refill < len(alive) * NEED_RATIONS:
                tot["short_inn"] += 1
            res = r["res"][n] if have_res else None
            for p in alive:
                f = r["fed"][n][p]
                x = sum(f)
                a = per[p]
                a["days"] += 1
                a["intake"] += x
                a["own"] += f[0]
                a["oth"] += f[1]
                a["cnt"] += f[2]
                s = x / NEED_RATIONS
                a["lt50"] += s < 0.5
                a["lt75"] += s < 0.75
                a["lt100"] += s < 1.0
                if res:
                    laid = res[p]["laid"]
                    a["laid"] += laid
                    a["cap"] += res[p]["cap"] > 0
                    a["cap_t"] += res[p].get("cap_t", 0) > 0
                    a["relief"] += res[p].get("relief", 0.0)
                    if s < 0.75:
                        a["lt75_laid" if laid else "lt75_well"] += 1
                if p != "Marta":
                    catches.append(c.get(p, 0))
                    if res and not res[p]["laid"]:
                        well_catch.append(c.get(p, 0))
    nights, alive = tot["nights"], tot["alive"]
    out = {"nights": int(nights), "person_nights": int(alive), "rations_per_night": {k: tot[k] / nights for k in ("catch", "refill", "work", "own", "oth", "cnt")},
           "eaten_per_night": (tot["own"] + tot["oth"] + tot["cnt"]) / nights, "demand_per_night": tot["demand"] / nights,
           "supply_per_night": (tot["catch"] + tot["refill"] + tot["work"]) / nights,
           "supply_no_work_per_night": (tot["catch"] + tot["refill"]) / nights,
           "eaten_over_need": (tot["own"] + tot["oth"] + tot["cnt"]) / tot["demand"],
           "spoiled_per_night": (tot["catch"] - tot["own"] - tot["oth"]) / nights,
           "inn_days_short_share": tot["short_inn"] / max(1, tot["inn_nights"] - len(R)),
           "per_person": {p: {k: (a[k] / a["days"] if k in ("intake", "own", "oth", "cnt") else a[k] / a["days"]) for k in a} | {"days": a["days"]} for p, a in per.items()}}
    cs = np.array(catches, float)
    out["catch_producers"] = {"mean": float(cs.mean()), "empty": float(np.mean(cs == 0)), "cv": float(cs.std() / cs.mean()) if cs.mean() else None}
    if well_catch:
        w = np.array(well_catch, float)
        out["catch_well_producers"] = {"mean": float(w.mean()), "empty": float(np.mean(w == 0)), "cv": float(w.std() / w.mean())}
    out["kcal"] = {"ration": RATION_KCAL, "need_person_day": FIELD_KCAL_DAY, "eaten_person_day": out["eaten_over_need"] * FIELD_KCAL_DAY}
    return out


def show_balance(b, label):
    r = b["rations_per_night"]
    L_ = [f"== FOOD BALANCE {label}: {b['nights']} inn-nights, {b['person_nights']} person-nights alive (a ration = {b['kcal']['ration']:.0f} kcal; a person's day = {b['kcal']['need_person_day']:.0f} kcal = {NEED_RATIONS:.1f} rations)",
          f"production per night (rations): catches {r['catch']:.2f} + counter refill {r['refill']:.2f} = {b['supply_no_work_per_night']:.2f}; plus work at the counter {r['work']:.2f} (labour, not in the mean supply)",
          f"demand per night {b['demand_per_night']:.2f}; production/demand {b['supply_no_work_per_night'] / b['demand_per_night']:.2f} (with work {b['supply_per_night'] / b['demand_per_night']:.2f})",
          f"eaten per night {b['eaten_per_night']:.2f} = {b['eaten_over_need']:.0%} of the need ({b['kcal']['eaten_person_day']:.0f} kcal a person-day); spoiled or unused catch {b['spoiled_per_night']:.2f} a night",
          f"eaten from: own catch {r['own']:.2f}, another's catch {r['oth']:.2f}, counter {r['cnt']:.2f} (rations a night)",
          f"inn-days whose own production (catches + refill) was below the cast's need: {b['inn_days_short_share']:.0%}"]
    c = b["catch_producers"]
    L_.append(f"producers' catch: mean {c['mean']:.2f}, empty {c['empty']:.2f}, person-day CV {c['cv']:.2f} (all producer-days, laid-up included)")
    if "catch_well_producers" in b:
        c = b["catch_well_producers"]
        L_.append(f"well producers' catch: mean {c['mean']:.2f}, empty {c['empty']:.2f}, person-day CV {c['cv']:.2f}")
    L_.append("who eats (rations a day; share of the need; own/other's/counter; days under 50/75/100 percent of the need; laid-up days; days at an extreme)")
    for p, a in b["per_person"].items():
        L_.append(f"  {p:8s} {a['intake']:.2f} = {a['intake'] / NEED_RATIONS:4.0%}  ({a['own']:.2f}/{a['oth']:.2f}/{a['cnt']:.2f})  <50% {a['lt50']:.2f}  <75% {a['lt75']:.2f}  <100% {a['lt100']:.2f}"
                  f"  laid {a['laid']:.2f}  hunger-extreme {a['cap']:.3f}  thirst-extreme {a['cap_t']:.3f}  relief {a['relief']:.2f} of the 0.90 drift ({a['relief'] / 0.9:.0%}; a ration is worth {a['relief'] / a['intake'] / 0.25 if a['intake'] else 0:.2f} of its nominal 0.25)" + (f"  of the <75% days: laid {a['lt75_laid']:.3f}, well {a['lt75_well']:.3f}" if a["laid"] or a["lt75_laid"] or a["lt75_well"] else ""))
    return "\n".join(L_)


# ------------------------------------------------------------------ the risk measure
def life_risk(r, name):
    """-> dict for one person's life: depletion, near-miss nights, burn, extreme nights, over a horizon of len(r['res']) nights (dead nights count as depleted)."""
    res = r["res"]
    N = len(res)
    dep = nm = ex = 0
    burn = 0.0
    nights_alive = 0
    for n in range(N):
        x = res[n][name]
        if x["dead"]:
            dep += 1.0
            nm += 1
            ex += 1
        else:
            nights_alive += 1
            dep += 1.0 - x["min"]
            nm += x["min"] < 0.20
            ex += x["cap"] > 0
            burn += x["burn"]
    return {"depletion": dep / N, "near_miss_per_1000": 1000.0 * nm / N, "burn_per_year": burn / (N / NPY), "extreme_share": ex / N, "nights": N}


def bins_table(R):
    """The arm-blind calibrator: P(die within 30 nights | f bin), from every person-night of every run."""
    edges = [0.0, 0.05, 0.10, 0.20, 0.40, 1.01]
    cnt = np.zeros(5)
    die = np.zeros(5)
    deaths = 0
    for r in R.values():
        res = r["res"]
        N = len(res)
        dn = {d["name"]: d["night"] for d in r["deaths"]}
        deaths += len(dn)
        for p in NAMES:
            for n in range(N):
                x = res[n][p]
                if x["dead"]:
                    break
                k = int(np.searchsorted(edges, x["min"], side="right")) - 1
                k = min(max(k, 0), 4)
                cnt[k] += 1
                if p in dn and dn[p] - n <= 30 and dn[p] >= n:
                    die[k] += 1
    return {"edges": edges, "n": cnt.tolist(), "die": die.tolist(), "deaths": deaths, "p": (die / np.maximum(cnt, 1)).tolist()}


def est_risk(r, name, tab):
    res, p = r["res"], tab["p"]
    edges = tab["edges"]
    tot = 0.0
    N = len(res)
    for n in range(N):
        x = res[n][name]
        if x["dead"]:
            break
        k = min(max(int(np.searchsorted(edges, x["min"], side="right")) - 1, 0), 4)
        tot += p[k]
    return tot / N * NPY / 30.0


def paired(R, key, focal_only=True):
    """seed -> (rules value, oracle value) of a per-life measure of the focal."""
    out = {}
    for s in sorted({s for _, s in R}):
        if ("rules", s) in R and ("oracle_both", s) in R:
            a = R[("rules", s)]
            b = R[("oracle_both", s)]
            out[s] = (key(a, a["focal"]), key(b, b["focal"]))
    return out


def ci_paired(d, seed=1, n=2000):
    d = np.asarray(d, float)
    m = float(d.mean())
    t = st.t.interval(0.95, len(d) - 1, loc=m, scale=st.sem(d)) if d.std() > 0 else (m, m)
    rng = np.random.default_rng(seed)
    bs = np.percentile([rng.choice(d, len(d)).mean() for _ in range(n)], [2.5, 97.5])
    try:
        w = float(st.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        w = 1.0
    return {"mean": m, "t_ci": [float(t[0]), float(t[1])], "boot_ci": [float(bs[0]), float(bs[1])], "wilcoxon_p": w, "n": len(d)}


def risk_setting(R, label):
    tab = bins_table(R)
    o = {"label": label, "seeds": len({s for _, s in R}), "bins": tab, "measures": {}}
    keys = {"depletion": lambda r, f: life_risk(r, f)["depletion"],
            "near_miss_per_1000": lambda r, f: life_risk(r, f)["near_miss_per_1000"],
            "burn_per_year": lambda r, f: life_risk(r, f)["burn_per_year"],
            "extreme_share": lambda r, f: life_risk(r, f)["extreme_share"]}
    if tab["deaths"] >= 15:
        keys["est_death_risk_per_year"] = lambda r, f: est_risk(r, f, tab)
    for k, f in keys.items():
        pr = paired(R, f)
        if not pr:
            continue
        a = np.array([v[0] for v in pr.values()])
        b = np.array([v[1] for v in pr.values()])
        o["measures"][k] = {"rules_mean": float(a.mean()), "oracle_mean": float(b.mean()), "diff": ci_paired(b - a), "zero_both_share": float(np.mean((a == 0) & (b == 0)))}
    # whole cast depletion (all five, by arm)
    cast = {}
    for arm in ("rules", "oracle_both"):
        v = [life_risk(r, p)["depletion"] for (c, s), r in R.items() if c == arm for p in NAMES]
        cast[arm] = float(np.mean(v))
    pc = {}
    for arm in ("rules", "oracle_both"):
        for p in NAMES:
            pc[f"{arm}:{p}"] = float(np.mean([life_risk(r, p)["depletion"] for (c, s), r in R.items() if c == arm]))
    o["cast_depletion"], o["cast_depletion_by_person"] = cast, pc
    seeds = sorted({s for _, s in R})
    rows = L.cox_rows(R, seeds)
    o["cox"], o["mh"] = L.cox(rows), L.mh(rows)
    o["cast_all"] = L.cast_table(R)
    o["balance"] = balance(R)
    o["balance_rules"] = balance({k: v for k, v in R.items() if k[0] == "rules"})
    return o


def show_risk(o):
    out = [f"== RISK, {o['label']} ({o['seeds']} seeds, paired by seed)"]
    for k, m in o["measures"].items():
        d = m["diff"]
        out.append(f"{k:24s} rules {m['rules_mean']:.4g}  oracle {m['oracle_mean']:.4g}  oracle-rules {d['mean']:+.4g}  t CI [{d['t_ci'][0]:+.4g}, {d['t_ci'][1]:+.4g}]  "
                   f"bootstrap [{d['boot_ci'][0]:+.4g}, {d['boot_ci'][1]:+.4g}]  Wilcoxon p {d['wilcoxon_p']:.3f}  (both zero in {m['zero_both_share']:.0%} of seeds)")
    m = o["measures"].get("depletion")
    if m:
        d = m["diff"]
        if m["zero_both_share"] > 0.8:
            v = "NO EXPOSURE (both arms have zero depletion in more than 80 percent of seeds)"
        elif d["mean"] < 0 and d["t_ci"][1] < 0 and d["boot_ci"][1] < 0:
            v = "MET"
        else:
            v = "NOT MET" + (" (interval wide / includes 0)" if d["t_ci"][1] >= 0 else "")
        out.append(f"PASS RULE (depletion, oracle below rules, t and bootstrap CIs both exclude 0): {v}")
    t = o["bins"]
    out.append(f"calibrator (30 nights, pooled both arms, {t['deaths']} deaths): bins {t['edges']}, person-nights {[int(x) for x in t['n']]}, deaths {[int(x) for x in t['die']]}")
    c = o["cox"]
    if c["hr"] is None:
        out.append(f"Cox on focal deaths: no deaths in an arm (oracle {c['events_oracle']}, rules {c['events_rules']}): not finite")
    else:
        out.append(f"Cox on focal deaths: HR oracle/rules {c['hr']:.2f}, profile CI [{c['hr_profile'][0]:.2f}, {c['hr_profile'][1]:.2f}]; events oracle {c['events_oracle']}, rules {c['events_rules']}")
    mh = o["mh"]
    out.append("MH rate ratio: " + ("n/a" if mh["rr"] is None else f"{mh['rr']:.2f} [{mh['ci'][0]:.2f}, {mh['ci'][1]:.2f}]"))
    out.append("whole-cast depletion (mean of 5 people): rules {rules:.4f}, oracle {oracle_both:.4f}".format(**o["cast_depletion"]))
    out.append("   by person: " + ", ".join(f"{k} {v:.4f}" for k, v in o["cast_depletion_by_person"].items()))
    t = o["cast_all"]
    out.append(f"whole cast, both choosers: {t['deaths']} deaths in {t['person_years']:.1f} person-years = {t['rate'][0]:.1f} per 1000 py [{t['rate'][1]:.1f}, {t['rate'][2]:.1f}]")
    for k, v in sorted(t["rates"].items()):
        out.append(f"     {k:30s} {t['by_cause'][k]:3d} deaths = {v[0]:7.2f} per 1000 py [{v[1]:.2f}, {v[2]:.2f}]")
    out.append(show_balance(o["balance"], "all lives"))
    return "\n".join(out)


def main():
    cmd = sys.argv[1]
    if cmd == "balance":
        R = L.load(sys.argv[2])
        print(show_balance(balance(R), sys.argv[2]))
        return
    d = {}
    for path, label in zip(sys.argv[2:4], ("P-need Fenn, all6hl2f", "P-kind Fenn, all6hl2f")):
        if path.endswith(".json") or not Path(path).exists() or not list(Path(path).glob("*.json")):
            continue
        o = risk_setting(L.load(path), label)
        d[label] = o
        print(show_risk(o) + "\n")
    Path(sys.argv[-1]).write_text(json.dumps(d, indent=1, default=str))


if __name__ == "__main__":
    main()
