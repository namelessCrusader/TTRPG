"""Step 1 of SV5 (2026-11-21): the world check for the P-need partner. Reads the O-family runs of two settings and prints the numbers.

    lab/.venv/bin/python -m lab.mindL.step1_report RUNS_NEED RUNS_KIND OUT.json

RUNS_NEED: all6s, Fenn = P-need(0.9), Oren = P-hard(0.9).  RUNS_KIND: all6, Fenn = P-kind(0.9), Oren = P-hard(0.9) (the old test).
Per setting: error of rules and the three oracles (mean of the scored nights, 95% t interval over seeds 1-10), the gap (rules minus the
oracle, paired by seed), the gap bound (the recipe of lab/results/mindL3/stake/bound.py: partner-specific kind acts a night times
(b - c), generous and night-averaged), each partner's answers (per night, and the drop in the focal's drive error per answer, b plus
bpot), and the oracle's contrast: how differently it treats Fenn and Oren (kind minus harsh acts a night at Fenn, minus the same at
Oren), against rules'. All numbers per seed are in the json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.stats as st

from lab.world2.world import HARSH_ACTS, KIND_ACTS

KIND = set(KIND_ACTS) - {"gossip"}
HARSH = {"strike", "shove", "threaten", "insult", "steal", "accuse", "search"} | set(HARSH_ACTS)
CH = ("rules", "oracle_avoid", "oracle_kind", "oracle_both")


def ci(xs):
    x = np.array([v for v in xs if v is not None and np.isfinite(v)], dtype=float)
    if len(x) < 2:
        return {"mean": float(x.mean()) if len(x) else None, "lo": None, "hi": None, "n": len(x)}
    h = float(st.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
    return {"mean": float(x.mean()), "lo": float(x.mean() - h), "hi": float(x.mean() + h), "n": len(x)}


def fmt(c, d=3):
    if c["mean"] is None:
        return "n/a"
    if c["lo"] is None:
        return f"{c['mean']:+.{d}f}"
    return f"{c['mean']:+.{d}f} [{c['lo']:+.{d}f}, {c['hi']:+.{d}f}]"


def load(d):
    R = {}
    for fp in sorted(Path(d).glob("O_*.json")):
        r = json.loads(fp.read_text())
        R[(r["chooser"], r["seed"])] = r
    return R


def night_err(r):
    return float(np.mean(r["err"][r["win0"]:]))


def bound(R):
    rows = []
    for (c, s), r in R.items():
        if c != "rules":
            continue
        n = len(r["err"]) - r["win0"]
        ps = ("Fenn", "Oren")
        kind_to_ps = sum(v for k, v in r["made_to"].items() if k.split(">")[1] in ps and k.split(">")[0] in KIND)
        tot = dict(n=0, b=0, bpot=0, c=0, b_rem=0, c_rem=0)
        for key, a in r["agg"].items():
            v, role = key.split("|")
            if v in KIND and role == "focal":
                for k in tot:
                    tot[k] += a["n_done"] if k == "n" else a[k]
        na = max(1, tot["n"])
        gen = (tot["b"] + tot["bpot"] - tot["c"]) / na
        timed = (tot["b_rem"] - tot["c_rem"]) / na
        hn = hs = 0.0
        for key, a in r["agg"].items():
            v, role = key.split("|")
            if v in {"strike", "shove", "threaten", "insult", "steal", "accuse", "search"} and role == "toF":
                hn += a["n_done"]
                hs += -a["b"]
        hpa = hs / max(1, hn)
        frm = sum(v for k, v in r["recv_from"].items() if k.endswith("<Oren") and k.split("<")[0] in HARSH)
        rows.append(dict(seed=s, kind_acts=kind_to_ps / n, gen=gen, timed=timed,
                         generous=(kind_to_ps / n) * max(0, gen) + (frm / n) * max(0, hpa),
                         nightavg=(kind_to_ps / n) * max(0, timed) + (frm / n) * max(0, hpa)))
    return {k: float(np.mean([x[k] for x in rows])) for k in ("kind_acts", "gen", "timed", "generous", "nightavg")}


def answers(R, chooser, partner):
    """per seed: answers a night at the focal, the drop per answer (b + bpot), by verb counts."""
    per_night, drop, verbs = [], [], {}
    for s in range(1, 11):
        r = R.get((chooser, s))
        if not r:
            continue
        n = len(r["err"]) - r["win0"]
        bp = r.get("by_partner", {}).get(partner, {})
        k = sum(v[0] for v in bp.values())
        per_night.append(k / n)
        drop.append(sum(v[1] + v[2] for v in bp.values()) / k if k else float("nan"))
        for vb, v in bp.items():
            verbs[vb] = verbs.get(vb, 0) + v[0] / n / 10
    return ci(per_night), ci(drop), {k: round(v, 3) for k, v in sorted(verbs.items(), key=lambda kv: -kv[1])}


def contrast(R, chooser):
    """per seed: (kind - harsh acts a night at Fenn) minus the same at Oren."""
    out, kf, ko, hf, ho = [], [], [], [], []
    for s in range(1, 11):
        r = R.get((chooser, s))
        if not r:
            continue
        n = len(r["err"]) - r["win0"]

        def cnt(who, acts):
            return sum(v for k, v in r["made_to"].items() if k.split(">")[1] == who and k.split(">")[0] in acts) / n
        a, b, c, d = cnt("Fenn", KIND), cnt("Oren", KIND), cnt("Fenn", HARSH), cnt("Oren", HARSH)
        kf.append(a), ko.append(b), hf.append(c), ho.append(d)
        out.append((a - c) - (b - d))
    return out, dict(kind_Fenn=ci(kf), kind_Oren=ci(ko), harsh_Fenn=ci(hf), harsh_Oren=ci(ho))


def setting(R, label):
    errs = {c: [night_err(R[(c, s)]) if (c, s) in R else None for s in range(1, 11)] for c in CH}
    res = {"label": label, "error": {c: ci(v) for c, v in errs.items()}, "per_seed_error": errs, "gap": {}, "answers": {}, "contrast": {}}
    for c in CH[1:]:
        g = [a - b if a is not None and b is not None else None for a, b in zip(errs["rules"], errs[c])]
        res["gap"][c] = {**ci(g), "per_seed": g}
    res["bound"] = bound(R)
    for partner in ("Fenn", "Oren"):
        res["answers"][partner] = {c: dict(zip(("per_night", "drop_per_answer", "verbs"), answers(R, c, partner))) for c in ("rules", "oracle_both")}
    cr, cm = {}, {}
    for c in ("rules", "oracle_both", "oracle_kind", "oracle_avoid"):
        cr[c], cm[c] = contrast(R, c)
    res["contrast"] = {c: {"per_seed": cr[c], "ci": ci(cr[c]), "parts": cm[c]} for c in cr}
    diff = [a - b for a, b in zip(cr["oracle_both"], cr["rules"])]
    res["contrast_oracle_minus_rules"] = {**ci(diff), "per_seed": diff}
    return res


def show(res):
    L = [f"== {res['label']}"]
    L.append("error (mean of scored nights; 95% CI over seeds 1-10)")
    for c in CH:
        L.append(f"  {c:13s} {fmt(res['error'][c])}")
    L.append("oracle gap = rules minus oracle, paired by seed")
    for c in CH[1:]:
        L.append(f"  {c:13s} {fmt(res['gap'][c])}")
    b = res["bound"]
    L.append(f"gap bound: kind acts at the partners {b['kind_acts']:.2f} a night; (b-c) per act {b['gen']:.4f} generous, {b['timed']:.4f} night-averaged; "
             f"bound {b['generous']:.3f} generous, {b['nightavg']:.3f} night-averaged")
    for p in ("Fenn", "Oren"):
        for c in ("rules", "oracle_both"):
            a = res["answers"][p][c]
            L.append(f"answers of {p} at the focal, chooser {c}: {fmt(a['per_night'], 2)} a night; drop in the focal's drive error per answer "
                     f"{fmt(a['drop_per_answer'], 4)}; by act {a['verbs']}")
    L.append("contrast = (kind - harsh acts a night at Fenn) minus the same at Oren")
    for c in ("rules", "oracle_both", "oracle_kind", "oracle_avoid"):
        x = res["contrast"][c]
        pt = x["parts"]
        L.append(f"  {c:13s} {fmt(x['ci'], 2)}   (kind F {pt['kind_Fenn']['mean']:.2f} O {pt['kind_Oren']['mean']:.2f}; harsh F {pt['harsh_Fenn']['mean']:.2f} O {pt['harsh_Oren']['mean']:.2f})")
    L.append(f"  oracle_both minus rules {fmt(res['contrast_oracle_minus_rules'], 2)}")
    return "\n".join(L)


def main():
    out = {}
    for d, label in ((sys.argv[1], "P-need in all6 with signs"), (sys.argv[2], "P-kind in all6 (old)")):
        R = load(d)
        out[label] = setting(R, label)
        print(show(out[label]), "\n")
    Path(sys.argv[3]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
