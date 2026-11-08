"""Round 2 analysis: the same numbers as report_L.py, from the all2 lifetimes of lab/results/mindL2/runs/
(chooser mind2 = the fixed mind, read as "mind" by the unchanged analysis), plus a test-by-test comparison
with round 1 (lab/results/mindL/*_seeds1-10.json).

    MINDL_RUNS=lab/results/mindL2/runs MINDL_TAG=r2_ lab/.venv/bin/python -m lab.mindL.report_L2
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("MINDL_RUNS", "lab/results/mindL2/runs")
os.environ.setdefault("MINDL_TAG", "r2_")

from pathlib import Path

from lab.mindL import report_L as R1
from lab.mindL.report_L import ci, fmt

R1.RUNS = Path(os.environ["MINDL_RUNS"]).resolve()


def load(family):
    out = {}
    for fp in R1.RUNS.glob(f"{family}_*_all2.json"):
        r = json.loads(fp.read_text())
        c = "mind" if r["chooser"] == "mind2" else r["chooser"]
        out[(r["variant"], c, r["seed"])] = r
    return out


R1.load = load
OLD = Path(__file__).resolve().parents[1] / "results" / "mindL"
NEW = R1.OUTD


def val(rows, key="value"):
    return {c: r[key] for c, r in rows.items() if key in r}


def compare():
    L = {k: (json.loads((OLD / f"{k}_seeds1-10.json").read_text()), json.loads((NEW / f"r2_{k}_seeds1-10.json").read_text()))
         for k in ("L0", "L1", "L2", "L3", "L4", "L5") if (NEW / f"r2_{k}_seeds1-10.json").exists()}
    out, md = {}, []

    def row(name, a, b):
        md.append(f"| {name} | {fmt(a) if a else '—'} | {fmt(b) if b else '—'} |")
        out.setdefault(cur[0], {})[name] = {"round1": a, "round2": b}

    def table(title, pairs):
        md.extend([f"### {title}", "", "| item | round 1 | round 2 |", "| --- | --- | --- |"])
        for n, a, b in pairs:
            row(n, a, b)
        md.append("")
    cur = [""]
    for k, (o, n) in L.items():
        cur[0] = k
        md.append(f"## {k}: round 1 verdict `{o.get('verdict', o.get('pass'))}`; round 2 verdict `{n.get('verdict', n.get('pass'))}`\n")
        P = []
        if k == "L0":
            for c in ("mind", "bandit", "ppo", "rules", "random"):
                if c in o["rows"] and c in n["rows"]:
                    P.append((f"{c} last-20% error", o["rows"][c]["last"], n["rows"][c]["last"]))
            for c in ("mind", "bandit", "ppo"):
                if c in o["rows"] and c in n["rows"]:
                    P.append((f"{c} minus rules", o["rows"][c].get("minus_rules"), n["rows"][c].get("minus_rules")))
            P.append(("mind minus ppo", o["rows"]["mind"].get("minus_ppo"), n["rows"]["mind"].get("minus_ppo")))
        elif k == "L1":
            for b in o["rows"]:
                for c in o["rows"][b]:
                    if c in n["rows"].get(b, {}):
                        P.append((f"{b} {c}", o["rows"][b][c]["value"], n["rows"][b][c]["value"]))
        elif k == "L2":
            for c in o["contrast"]:
                if c in n["contrast"]:
                    P.append((f"contrast {c}", o["contrast"][c]["value"], n["contrast"][c]["value"]))
            P.append(("mind real minus shuffled", o["shuffled"]["real_minus_shuffled"].get("mind"),
                      n["shuffled"]["real_minus_shuffled"].get("mind")))
        elif k == "L3":
            for key in ("first_betrayal_A", "second_A_minus_first_B_same_age"):
                for c in o[key]:
                    if c in n[key]:
                        P.append((f"{key} {c}", o[key][c]["value"], n[key][c]["value"]))
        elif k == "L4":
            for key in o["rows"]:
                if key in n["rows"]:
                    P.append((f"index {key}", o["rows"][key]["index"], n["rows"][key]["index"]))
            for c in o["drives_minus_designer"]:
                if c in n["drives_minus_designer"]:
                    P.append((f"drives minus designer {c}", o["drives_minus_designer"][c], n["drives_minus_designer"][c]))
        elif k == "L5":
            for c in o["order"]:
                if c in n["order"]:
                    P.append((f"order {c}", o["order"][c]["value"], n["order"][c]["value"]))
        table("numbers", P)
    (NEW / "r2_comparison.json").write_text(json.dumps(out, indent=1))
    (NEW / "r2_comparison.md").write_text("\n".join(md))
    return "\n".join(md)


if __name__ == "__main__":
    R1.report()
    print(compare())
