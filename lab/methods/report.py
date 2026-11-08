"""Tables for Workstream A: method x model x factor.

    python3 -m lab.methods.report            # writes lab/results/methods/report.md

For each factor, one table: a row per scorer, a column per value, and in
each cell the share of cases whose pick is in the acceptable set, with its
95% range (Wilson). The first row is chance: the acceptable share of the list.
Cells where the scorer could not run (for example a list too long for its
input) are counted and shown, not hidden.
"""
from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
RES = HERE.parent / "results" / "methods"
ORDER = ["size", "optlen", "position", "similar", "situation", "good"]
VALUES = {"size": [4, 8, 16, 32, 64, 128], "optlen": ["short", "long"],
          "position": ["first", "middle", "last"], "similar": ["unrelated", "same", "near"],
          "situation": ["short", "long"], "good": [1, 2, 4]}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0, c - h), min(1, c + h))


def load():
    runs = {}
    for f in sorted(glob.glob(str(RES / "*.jsonl"))):
        name = os.path.basename(f)[:-6]
        if "." in name:            # smoke and other tagged runs
            continue
        runs[name.replace("__", ":")] = [json.loads(l) for l in open(f)]
    return runs


def main():
    cases = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "cases_things.jsonl")}
    runs = load()
    out = ["# Workstream A: which method, and why does Laya get worse?", "",
           "Each cell: the share of cases whose pick is in the acceptable set, "
           "with the 95% range. n per cell is shown in the chance row. "
           "`x` = could not run (the list did not fit the model's input).", ""]
    # timing summary
    out += ["## Scorers", "", "| scorer | cases | could not run | ms per case (median) |", "| --- | --- | --- | --- |"]
    for name, rows in runs.items():
        ms = sorted(r["ms"] for r in rows)
        out.append(f"| {name} | {len(rows)} | {sum('error' in r for r in rows)} | "
                   f"{ms[len(ms) // 2]:.0f} |")
    out.append("")
    for factor in ORDER:
        vals = VALUES[factor]
        out += [f"## Factor: {factor}", "",
                "| scorer | " + " | ".join(str(v) for v in vals) + " |",
                "| --- |" + " --- |" * len(vals)]
        # chance
        ch = []
        for v in vals:
            cs = [c for c in cases.values() if c["factor"] == factor and c["value"] == v]
            ch.append(f"{sum(len(c['good']) / len(c['options']) for c in cs) / max(len(cs), 1):.0%} (n={len(cs)})")
        out.append("| chance | " + " | ".join(ch) + " |")
        for name, rows in runs.items():
            by = defaultdict(list)
            for r in rows:
                if r["factor"] == factor:
                    by[r["value"]].append(r)
            cells = []
            for v in vals:
                rs = by.get(v, [])
                good = [r for r in rs if "error" not in r]
                bad = len(rs) - len(good)
                if not rs:
                    cells.append("")
                    continue
                p, lo, hi = wilson(sum(r["ok"] for r in good), len(good))
                cell = f"{p:.0%} ({lo:.0%}-{hi:.0%})" if good else ""
                if bad:
                    cell += f" x{bad}"
                cells.append(cell)
            out.append(f"| {name} | " + " | ".join(cells) + " |")
        out.append("")
    # probability mass where available
    out += ["## Probability on the acceptable set (methods that give probabilities)", "",
            "| scorer | " + " | ".join(f"size {v}" for v in VALUES["size"]) + " |",
            "| --- |" + " --- |" * len(VALUES["size"])]
    for name, rows in runs.items():
        if not any("mass" in r for r in rows):
            continue
        cells = []
        for v in VALUES["size"]:
            ms = [r["mass"] for r in rows if r["factor"] == "size" and r["value"] == v and "mass" in r]
            cells.append(f"{sum(ms) / len(ms):.2f}" if ms else "")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    out.append("")
    inv = sorted(glob.glob(str(RES / "*.inv.jsonl")))
    if inv:
        out += ["## Invariance: the same case asked in 4 orders", "",
                "`same pick` = the pick did not change with the order. `sometimes right` = right in "
                "some orders and wrong in others. `spread` = how far the probability on the "
                "acceptable set moved between orders.", "",
                "| scorer | size | cases | same pick | sometimes right | spread |",
                "| --- | --- | --- | --- | --- | --- |"]
        for f in inv:
            name = os.path.basename(f)[:-10].replace("__", ":")
            rows = [json.loads(l) for l in open(f)]
            for size in sorted({r["value"] for r in rows if r["factor"] == "size"}):
                rs = [r for r in rows if r["factor"] == "size" and r["value"] == size and "picks" in r]
                if not rs:
                    continue
                k = len(rs[0]["picks"])
                some = sum(0 < r["ok_orders"] < k for r in rs)
                sp = [r["mass_spread"] for r in rs if "mass_spread" in r]
                out.append(f"| {name} | {size} | {len(rs)} | {sum(r['same'] for r in rs) / len(rs):.0%} | "
                           f"{some / len(rs):.0%} | {sum(sp) / len(sp):.2f} |" if sp else
                           f"| {name} | {size} | {len(rs)} | {sum(r['same'] for r in rs) / len(rs):.0%} | "
                           f"{some / len(rs):.0%} | |")
        out.append("")
    (RES / "report.md").write_text("\n".join(out) + "\n")
    print(RES / "report.md")


if __name__ == "__main__":
    main()
