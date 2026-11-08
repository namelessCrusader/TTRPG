"""The numbers of the tests L0-L5 from the lifetimes in lab/results/mindL/runs/
(made by lab/mindL/tests_L.py), and the report.

    lab/.venv/bin/python -m lab.mindL.tests_L report

THE MEASURE OF TREATMENT. For the focal character F and another person X,
over a block of nights: kind share = F's kind acts aimed at X / all of F's
kind acts (KIND_ACTS); provocation share = F's insults, blows, shoves and
threats aimed at X / all of F's such acts. treat(X) = kind share -
provocation share, from -1 (every provocation and no kindness goes to X) to
+1. Shares, not counts, so a chooser that shoves everyone 250 times a night is
compared by WHOM it shoves, not by how much. A P-hard partner answers exactly
the acts in the provocation share; a P-kind partner pays back gifts, the
largest part of the kind share.

Every interval is a 95% t interval over the 10 seeds (the seed is the unit:
the nights of one life are not independent). "Paired" means the same seed,
the same focal character, the same world draws.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
from scipy import stats as st

from lab.mindL.tests_L import CATS, CHOOSERS, FOCALS3, LEARNERS, N04, N12, N35, OUT, RUNS, SEEDS

NULLS = ("bandit", "ppo", "rules", "random")
OUTD = OUT if RUNS == OUT / "runs" else RUNS.parent
TAG = os.environ.get("MINDL_TAG", "")       # a prefix for the output names (e.g. "fade1.0_")


def load(family):
    out = {}
    for fp in RUNS.glob(f"{family}_*.json"):
        r = json.loads(fp.read_text())
        out[(r["variant"], r["chooser"], r["seed"])] = r
    return out


def ci(xs):
    x = np.array([v for v in xs if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    n = len(x)
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    m = float(x.mean())
    if n < 2:
        return {"mean": round(m, 4), "lo": None, "hi": None, "n": n}
    h = float(st.t.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n))
    return {"mean": round(m, 4), "lo": round(m - h, 4), "hi": round(m + h, 4), "n": n}


def above(c):
    return c["lo"] is not None and c["lo"] > 0


def below(c):
    return c["hi"] is not None and c["hi"] < 0


def fmt(c, d=3):
    if c["mean"] is None:
        return "—"
    if c["lo"] is None:
        return f"{c['mean']:+.{d}f}"
    return f"{c['mean']:+.{d}f} [{c['lo']:+.{d}f}, {c['hi']:+.{d}f}]"


def treat(run, x, a, b):
    """treat(X) over nights [a, b): kind share - provocation share."""
    rows = [r for r in run["nights_rows"] if a <= r["n"] < b]
    k = sum(r["to"][x][0] for r in rows)
    pv = sum(r["to"][x][1] for r in rows)
    K = sum(sum(v[0] for v in r["to"].values()) for r in rows)
    PV = sum(sum(v[1] for v in r["to"].values()) for r in rows)
    return (k / K if K else 0.0) - (pv / PV if PV else 0.0)


def shares(run, x, a, b):
    rows = [r for r in run["nights_rows"] if a <= r["n"] < b]
    k = sum(r["to"][x][0] for r in rows)
    pv = sum(r["to"][x][1] for r in rows)
    K = sum(sum(v[0] for v in r["to"].values()) for r in rows)
    PV = sum(sum(v[1] for v in r["to"].values()) for r in rows)
    n = max(1, len(rows))
    return {"kind_share": k / K if K else 0.0, "prov_share": pv / PV if PV else 0.0,
            "kind_per_night": k / n, "prov_per_night": pv / n,
            "from_kind_per_night": sum(r["from"][x][0] for r in rows) / n,
            "from_harsh_per_night": sum(r["from"][x][1] for r in rows) / n,
            "near": float(np.mean([r["near"][x] for r in rows])) if rows else 0.0}


def mind_at(run, x, night):
    r = next((r for r in run["nights_rows"] if r["n"] == night), None)
    return r["mind"][x] if r and "mind" in r else None


def paired(table, c, o):
    """c minus o, paired by seed: table[chooser] = {seed: value}."""
    return ci([table[c][s] - table[o][s] for s in SEEDS if s in table.get(c, {}) and s in table.get(o, {})
               and table[c][s] is not None and table[o][s] is not None])


def null_rows(table, sign):
    """Each chooser's interval, and the mind minus each null, paired.
    sign +1: the effect should be > 0; -1: < 0."""
    rows = {}
    for c in CHOOSERS:
        if c not in table:
            continue
        v = ci(list(table[c].values()))
        rows[c] = {"value": v, "shows": above(v) if sign > 0 else below(v)}
        if c != "mind" and "mind" in table:
            d = paired(table, "mind", c)
            rows[c]["mind_minus"] = d
            rows[c]["mind_beats"] = above(d) if sign > 0 else below(d)
    return rows


def verdict(rows, mind_pass):
    """The plan's rule: a capability counts only if the mind shows it and the
    nulls do not (a null that shows it too must be beaten, paired)."""
    if not mind_pass:
        return "FAIL: the mind does not show it"
    match = [c for c in NULLS if c in rows and rows[c]["shows"] and not rows[c].get("mind_beats")]
    if match:
        return "FAIL as a capability of the mind: the mind shows it, but so does " + ", ".join(match) + \
               " (and the mind does not beat it, paired)"
    return "PASS"


# ------------------------------------------------------------------ L0, L4
def l0(R):
    n = next(iter(R.values()))["nights"]
    q = n // 5
    table, deaths, first = {}, {}, {}
    for c in CHOOSERS:
        for s in SEEDS:
            r = R.get(("drives", c, s))
            if r is None:
                continue
            e = r["drive_error"]
            table.setdefault(c, {})[s] = float(np.mean(e[n - q:n]))
            first.setdefault(c, {})[s] = float(np.mean(e[:q]))
            deaths.setdefault(c, {})[s] = sum(x["dead"] for x in r["nights_rows"])
    rows = {c: {"last": ci(list(table[c].values())), "first": ci(list(first[c].values())),
                "deaths_total": int(sum(deaths[c].values())), "seeds": sorted(table[c])} for c in table}
    for c in table:
        for o in ("rules", "random", "ppo", "bandit"):
            if o != c and o in table:
                rows[c][f"minus_{o}"] = paired(table, c, o)

    def passes(c):
        r = rows.get(c, {})
        return ("minus_rules" in r and not above(r["minus_rules"]) and "minus_random" in r
                and below(r["minus_random"]))
    res = {"test": "L0", "measure": f"focal drive error, mean of nights {n - q}-{n - 1}", "nights": n,
           "rows": rows, "pass": {c: passes(c) for c in rows if c not in ("rules", "random")},
           "per_seed": table}
    return res


def l4(R):
    table = {}
    detail = {}
    for c in LEARNERS:
        for rw in ("drives", "designer"):
            for s in SEEDS:
                r = R.get((rw, c, s))
                if r is None:
                    continue
                rh = [v["rho"] for v in r["context"].values() if v.get("rho") is not None]
                table.setdefault(f"{c}/{rw}", {})[s] = float(np.mean(rh)) if rh else None
                for d, v in r["context"].items():
                    detail.setdefault(f"{c}/{rw}", {}).setdefault(d, []).append(v.get("rho"))
    for c in ("rules", "random"):
        for s in SEEDS:
            r = R.get(("drives", c, s))
            if r is None:
                continue
            rh = [v["rho"] for v in r["context"].values() if v.get("rho") is not None]
            table.setdefault(f"{c}/drives", {})[s] = float(np.mean(rh)) if rh else None
            for d, v in r["context"].items():
                detail.setdefault(f"{c}/drives", {}).setdefault(d, []).append(v.get("rho"))
    rows = {k: {"index": ci(list(v.values())), "by_drive": {d: ci(x) for d, x in detail[k].items()}}
            for k, v in table.items()}
    diff = {c: paired(table, f"{c}/drives", f"{c}/designer") for c in LEARNERS
            if f"{c}/drives" in table and f"{c}/designer" in table}
    ok = {c: above(rows.get(f"{c}/drives", {"index": ci([])})["index"]) and c in diff and above(diff[c])
          for c in LEARNERS}
    return {"test": "L4", "measure": "context index: mean over drives (energy, rest, curiosity, safety) of the "
                                     "Spearman correlation between the drive's need and the chance put on the acts "
                                     "that serve it, over choices with such an act on the menu, last half of life",
            "rows": rows, "drives_minus_designer": diff, "pass": ok, "per_seed": table}


# ------------------------------------------------------------------ L1, L2 (+ controls)
def l12(R):
    n = next(iter(R.values()))["nights"]
    cut = (2 * n) // 3
    late = (cut - 40, cut)
    blocks = {"late1": (cut - 40, cut - 20), "late2": (cut - 20, cut), "shared_early": (cut, cut + 20),
              "shared_late": (n - 40, n)}
    kindp = {"A": "Fenn", "B": "Oren"}
    L1 = {b: {} for b in blocks}
    L2k, L2h, L2c, S2k, S2h, S2c, deafdrift, internal = {}, {}, {}, {}, {}, {}, {}, {}
    plain = {}
    for c in CHOOSERS:
        for s in SEEDS:
            A, B, D = R.get(("A", c, s)), R.get(("B", c, s)), R.get(("deaf", c, s))
            if A is None or B is None:
                continue
            for b, (a0, a1) in blocks.items():
                L1[b].setdefault(c, {})[s] = 0.5 * ((treat(A, "Fenn", a0, a1) - treat(B, "Fenn", a0, a1)) +
                                                    (treat(B, "Oren", a0, a1) - treat(A, "Oren", a0, a1)))
            if D is None:
                continue
            for real, out_k, out_h, out_c in ((("A", "B"), L2k, L2h, L2c), (("shufA", "shufB"), S2k, S2h, S2c)):
                ek, eh = [], []
                for v in real:
                    run = R.get((v, c, s))
                    if run is None:
                        continue
                    kp = kindp[v[-1]]
                    hp = "Oren" if kp == "Fenn" else "Fenn"
                    ek.append(treat(run, kp, *late) - treat(D, kp, *late))
                    eh.append(treat(run, hp, *late) - treat(D, hp, *late))
                if len(ek) == 2:
                    out_k.setdefault(c, {})[s] = float(np.mean(ek))
                    out_h.setdefault(c, {})[s] = float(np.mean(eh))
                    out_c.setdefault(c, {})[s] = float(np.mean(ek) - np.mean(eh))
            deafdrift.setdefault(c, {})[s] = float(np.mean([treat(D, x, cut - 40, cut) - treat(D, x, 0, 40)
                                                            for x in ("Fenn", "Oren")]))
            if c == "mind":
                vals = []
                for v in ("A", "B"):
                    kp = kindp[v]
                    hp = "Oren" if kp == "Fenn" else "Fenn"
                    run = R[(v, c, s)]
                    mk, mh, md = mind_at(run, kp, cut - 1), mind_at(run, hp, cut - 1), mind_at(D, hp, cut - 1)
                    if mk and mh and md:
                        vals.append([mh[1] - mk[1], mk[0] - mh[0], mk[2] - mh[2], mh[1] - md[1]])
                if vals:
                    internal[s] = np.mean(vals, axis=0).tolist()
            for v in ("A", "B"):
                kp = kindp[v]
                hp = "Oren" if kp == "Fenn" else "Fenn"
                run = R[(v, c, s)]
                plain.setdefault(c, []).append({"kind_partner": shares(run, kp, *late),
                                                "hard_partner": shares(run, hp, *late),
                                                "deaf_partner": shares(D, hp, *late)})
    res1 = {"test": "L1", "measure": "clone index: 1/2[(treat(Fenn | Fenn kind) - treat(Fenn | Fenn hard)) + "
                                     "(treat(Oren | Oren kind) - treat(Oren | Oren hard))], clones = same seed",
            "blocks": {b: list(v) for b, v in blocks.items()},
            "rows": {b: null_rows(L1[b], +1) for b in blocks}, "per_seed": L1}
    mind_ok = all(res1["rows"][b].get("mind", {}).get("shows") for b in ("late1", "late2", "shared_early"))
    res1["mind_pass"] = mind_ok
    res1["nulls_pass"] = {c: all(res1["rows"][b].get(c, {}).get("shows") for b in ("late1", "late2", "shared_early"))
                          for c in NULLS}
    beats = {c: all(res1["rows"][b].get(c, {}).get("mind_beats") for b in ("late1", "late2", "shared_early"))
             for c in NULLS}
    match = [c for c in NULLS if res1["nulls_pass"][c] and not beats[c]]
    res1["verdict"] = ("FAIL: the mind does not show it" if not mind_ok else
                       ("FAIL as a capability of the mind: " + ", ".join(match) + " shows it too" if match else "PASS"))
    res1["contingency_broken_decay"] = {c: paired({"x": L1["late2"].get(c, {}), "y": L1["shared_late"].get(c, {})},
                                                  "x", "y") for c in CHOOSERS}

    r_k, r_h, r_c = null_rows(L2k, +1), null_rows(L2h, -1), null_rows(L2c, +1)
    shows = {c: r_k.get(c, {}).get("shows") and r_h.get(c, {}).get("shows") for c in CHOOSERS}
    res2 = {"test": "L2", "window": list(late),
            "measure": "within one life, against the same person when P-deaf (the deaf run, same seed): "
                       "effect_kind = treat(kind partner) - treat(it when deaf), effect_hard likewise; "
                       "averaged over the two lives (each partner is kind in one and hard in the other)",
            "effect_kind": r_k, "effect_hard": r_h, "contrast": r_c, "shows": shows,
            "shuffled": {"effect_kind": null_rows(S2k, +1), "effect_hard": null_rows(S2h, -1),
                         "contrast": null_rows(S2c, +1),
                         "real_minus_shuffled": {c: paired({"r": L2c.get(c, {}), "s": S2c.get(c, {})}, "r", "s")
                                                 for c in CHOOSERS}},
            "deaf_drift": {c: ci(list(deafdrift[c].values())) for c in deafdrift},
            "mind_internal": {"what": "at the last night before the partners turn deaf, mean over the two lives: "
                                      "[p_harm(hard) - p_harm(kind) for provocations, p_kind(kind) - p_kind(hard) for a "
                                      "gift, valence(kind) - valence(hard), p_harm(hard partner) - p_harm(same person "
                                      "when deaf)]",
                              "ci": [ci([v[i] for v in internal.values()]) for i in range(4)]},
            "plain": {c: {who: {k: round(float(np.mean([x[who][k] for x in v])), 3) for k in v[0][who]}
                          for who in ("kind_partner", "hard_partner", "deaf_partner")} for c, v in plain.items()},
            "per_seed": {"effect_kind": L2k, "effect_hard": L2h, "contrast": L2c, "shuffled_contrast": S2c}}
    nulls_both = [c for c in NULLS if shows.get(c)]
    beat = {c: r_c.get(c, {}).get("mind_beats") for c in NULLS}
    res2["verdict"] = ("FAIL: the mind does not show it" if not shows.get("mind") else
                       ("FAIL as a capability of the mind: " + ", ".join(c for c in nulls_both if not beat[c]) +
                        " shows it too" if any(not beat[c] for c in nulls_both) else "PASS"))
    return res1, res2


# ------------------------------------------------------------------ L3, L5 (+ control)
def l35(R):
    n = next(iter(R.values()))["nights"]
    b = [round(n * x) for x in (0.2, 0.4, 0.5, 0.7)]
    W = 10
    d1, d2a, d2b, sec_vs_first, sec_vs_same_age = {}, {}, {}, {}, {}
    order, order_late, internal3, internal5, plain = {}, {}, {}, {}, {}
    for c in CHOOSERS:
        for s in SEEDS:
            A, B = R.get(("A", c, s)), R.get(("B", c, s))
            if A is None or B is None:
                continue

            def resp(run, S, w=W):
                return treat(run, "Fenn", S, S + w) - treat(run, "Fenn", S - 20, S)
            d1.setdefault(c, {})[s] = resp(A, b[0])
            d2a.setdefault(c, {})[s] = resp(A, b[3])
            d2b.setdefault(c, {})[s] = resp(B, b[3])
            sec_vs_first.setdefault(c, {})[s] = resp(A, b[3]) - resp(A, b[0])
            sec_vs_same_age.setdefault(c, {})[s] = resp(A, b[3]) - resp(B, b[3])
            order.setdefault(c, {})[s] = treat(B, "Fenn", b[1], b[1] + 20) - treat(A, "Fenn", b[1], b[1] + 20)
            order_late.setdefault(c, {})[s] = treat(B, "Fenn", b[2] - 20, b[2]) - treat(A, "Fenn", b[2] - 20, b[2])
            plain.setdefault(c, []).append({
                "A_before_b1": shares(A, "Fenn", b[0] - 20, b[0]), "A_after_b1": shares(A, "Fenn", b[0], b[0] + W),
                "A_before_b4": shares(A, "Fenn", b[3] - 20, b[3]), "A_after_b4": shares(A, "Fenn", b[3], b[3] + W),
                "B_after_b4": shares(B, "Fenn", b[3], b[3] + W),
                "A_test": shares(A, "Fenn", b[1], b[1] + 20), "B_test": shares(B, "Fenn", b[1], b[1] + 20)})
            if c == "mind":
                mA = [mind_at(A, "Fenn", k) for k in (b[0] - 1, b[0] + 4, b[3] - 1, b[3] + 4)]
                mB = [mind_at(B, "Fenn", k) for k in (b[3] - 1, b[3] + 4)]
                if all(mA) and all(mB):
                    internal3[s] = [mA[1][1] - mA[0][1], mA[3][1] - mA[2][1], mB[1][1] - mB[0][1]]
                a5, b5 = mind_at(A, "Fenn", b[1] - 1), mind_at(B, "Fenn", b[1] - 1)
                if a5 and b5:
                    internal5[s] = [a5[1] - b5[1], b5[0] - a5[0], b5[2] - a5[2]]
    r1, r2a, r2b = null_rows(d1, -1), null_rows(d2a, -1), null_rows(d2b, -1)
    rs, ra = null_rows(sec_vs_first, -1), null_rows(sec_vs_same_age, -1)
    follows = r1.get("mind", {}).get("shows")
    faster = ra.get("mind", {}).get("shows")
    nulls3 = [c for c in NULLS if r1.get(c, {}).get("shows") and ra.get(c, {}).get("shows")
              and not ra[c].get("mind_beats")]
    res3 = {"test": "L3", "switch_nights": b, "window": W,
            "measure": f"response to a betrayal (P-kind turns P-hard at night S): treat(Fenn) over nights "
                       f"[S, S+{W}) minus over [S-20, S); below 0 = it trusts Fenn less within {W} nights",
            "first_betrayal_A": r1, "second_betrayal_A": r2a, "first_betrayal_B_same_age": r2b,
            "second_minus_first_A": rs, "second_A_minus_first_B_same_age": ra,
            "mind_internal": {"what": "change of p_harm(Fenn, provocations) over the first 5 nights after the switch: "
                                      "[A first betrayal, A second, B first at the same age]",
                              "ci": [ci([v[i] for v in internal3.values()]) for i in range(3)]},
            "plain": {c: {k: {m: round(float(np.mean([x[k][m] for x in v])), 3) for m in v[0][k]} for k in v[0]}
                      for c, v in plain.items()},
            "per_seed": {"first_A": d1, "second_A": d2a, "first_B": d2b, "second_minus_first_same_age": sec_vs_same_age}}
    res3["verdict"] = ("FAIL: the mind's trust does not follow the first betrayal within the window" if not follows else
                       "FAIL: the second betrayal is not learned faster" if not faster else
                       ("FAIL as a capability of the mind: " + ", ".join(nulls3) + " shows it too" if nulls3
                        else "PASS"))
    r5, r5l = null_rows(order, +1), null_rows(order_late, +1)
    res5 = {"test": "L5", "test_window": [b[1], b[1] + 20],
            "measure": "order index: treat(Fenn | twin B: hard then kind) - treat(Fenn | twin A: kind then hard), "
                       "in the first 20 nights after both twins meet the same P-deaf Fenn; above 0 = the twin whose "
                       "last lived phase was kind treats Fenn better",
            "order": r5, "order_late_deaf_phase": r5l,
            "mind_internal": {"what": "at the night before the test window, A minus B in p_harm(Fenn), B minus A in "
                                      "p_kind(Fenn, gift), B minus A in valence(Fenn)",
                              "ci": [ci([v[i] for v in internal5.values()]) for i in range(3)]},
            "per_seed": order}
    res5["verdict"] = verdict(r5, r5.get("mind", {}).get("shows"))
    return res3, res5


# ------------------------------------------------------------------ the report
def table_md(rows, label="value"):
    lines = [f"| chooser | {label} (95% CI) | shows it | mind minus it (paired, 95% CI) |", "| --- | --- | --- | --- |"]
    for c in CHOOSERS:
        if c not in rows:
            continue
        r = rows[c]
        lines.append(f"| {c} | {fmt(r['value'])} | {'yes' if r['shows'] else 'no'} | "
                     f"{fmt(r['mind_minus']) if 'mind_minus' in r else '—'} |")
    return "\n".join(lines)


def report():
    L04, L12, L35 = load("L04"), load("L12"), load("L35")
    out = {}
    if L04:
        out["L0"], out["L4"] = l0(L04), l4(L04)
    if L12:
        out["L1"], out["L2"] = l12(L12)
    if L35:
        out["L3"], out["L5"] = l35(L35)
    secs = {f: round(sum(r["seconds"] for r in R.values()) / 3600, 2) for f, R in (("L04", L04), ("L12", L12),
                                                                                    ("L35", L35)) if R}
    counts = {f: len(R) for f, R in (("L04", L04), ("L12", L12), ("L35", L35))}
    for k, v in out.items():
        (OUTD / f"{TAG}{k}_seeds1-10.json").write_text(json.dumps({**v, "seeds": list(SEEDS), "lifetimes": counts,
                                                             "cpu_hours": secs}, indent=1))
    (OUTD / f"{TAG}Ltests_summary.json").write_text(json.dumps({"lifetimes": counts, "cpu_hours": secs,
                                                         "verdicts": {k: v.get("verdict", v.get("pass"))
                                                                      for k, v in out.items()}}, indent=1))
    print(json.dumps({"lifetimes": counts, "cpu_hours": secs}, indent=1))
    md = tables(out)
    (Path.home() / ".cache" / "mark1-build" / "mindL" / "tests" / f"{TAG}tables_L.md").write_text(md)
    print(md)
    return out


def tables(out):
    """All the tables, as markdown (the report quotes them)."""
    T = []
    if "L0" in out:
        r = out["L0"]
        T += [f"## L0 ({r['measure']})", "", "| chooser | first 20% | last 20% | minus rules | minus random | minus PPO | "
              "minus bandit | deaths |", "| --- " * 8 + "|"]
        for c in CHOOSERS:
            if c in r["rows"]:
                x = r["rows"][c]
                T.append(f"| {c} | {fmt(x['first'])} | {fmt(x['last'])} | " + " | ".join(
                    fmt(x[f"minus_{o}"]) if f"minus_{o}" in x else "—" for o in ("rules", "random", "ppo", "bandit"))
                         + f" | {x['deaths_total']} |")
        T += ["", f"pass (as low as rules, far below random): {r['pass']}", ""]
    if "L4" in out:
        r = out["L4"]
        T += [f"## L4 ({r['measure']})", "", "| chooser / reward | index | " + " | ".join(CATS) + " |",
              "| --- " * (2 + len(CATS)) + "|"]
        for k, x in r["rows"].items():
            T.append(f"| {k} | {fmt(x['index'])} | " + " | ".join(fmt(x["by_drive"].get(d, ci([]))) for d in CATS) + " |")
        T += ["", "drives minus designer (paired): " + "; ".join(f"{c}: {fmt(v)}" for c, v in
                                                               r["drives_minus_designer"].items()),
              f"pass: {r['pass']}", ""]
    if "L1" in out:
        r = out["L1"]
        T += [f"## L1 ({r['measure']})", ""]
        for b, rows in r["rows"].items():
            T += [f"### block {b} nights {r['blocks'][b]}", "", table_md(rows, "clone index"), ""]
        T += ["contingency broken: index late phase 1 minus late shared world (decay; paired): " +
              "; ".join(f"{c}: {fmt(v)}" for c, v in r["contingency_broken_decay"].items()),
              f"verdict: {r['verdict']}", ""]
    if "L2" in out:
        r = out["L2"]
        T += [f"## L2 ({r['measure']}), nights {r['window']}", "", "### effect_kind (should be > 0)", "",
              table_md(r["effect_kind"]), "", "### effect_hard (should be < 0)", "", table_md(r["effect_hard"]), "",
              "### contrast kind minus hard", "", table_md(r["contrast"]), "",
              "### SHUFFLED answers: contrast", "", table_md(r["shuffled"]["contrast"]), "",
              "real minus shuffled contrast (paired): " + "; ".join(
                  f"{c}: {fmt(v)}" for c, v in r["shuffled"]["real_minus_shuffled"].items()), "",
              "P-deaf drift (treat of the deaf partners, late phase 1 minus first 40 nights): " + "; ".join(
                  f"{c}: {fmt(v)}" for c, v in r["deaf_drift"].items()), "",
              f"mind internals ({r['mind_internal']['what']}): " + "; ".join(fmt(v) for v in r["mind_internal"]["ci"]),
              "", "plain (means): " + json.dumps(r["plain"]), "", f"verdict: {r['verdict']}", ""]
    if "L3" in out:
        r = out["L3"]
        T += [f"## L3 ({r['measure']}), switches {r['switch_nights']}", ""]
        for k in ("first_betrayal_A", "second_betrayal_A", "first_betrayal_B_same_age", "second_minus_first_A",
                  "second_A_minus_first_B_same_age"):
            T += [f"### {k}", "", table_md(r[k]), ""]
        T += [f"mind internals ({r['mind_internal']['what']}): " + "; ".join(fmt(v) for v in r["mind_internal"]["ci"]),
              "", "plain (means): " + json.dumps(r["plain"]), "", f"verdict: {r['verdict']}", ""]
    if "L5" in out:
        r = out["L5"]
        T += [f"## L5 ({r['measure']}), window {r['test_window']}", "", table_md(r["order"], "order index"), "",
              "### late in the deaf phase", "", table_md(r["order_late_deaf_phase"], "order index"), "",
              f"mind internals ({r['mind_internal']['what']}): " + "; ".join(fmt(v) for v in r["mind_internal"]["ci"]),
              "", f"verdict: {r['verdict']}", ""]
    return "\n".join(T)


if __name__ == "__main__":
    report()
