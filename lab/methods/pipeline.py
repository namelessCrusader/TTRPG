"""Short list, then choose: a fast order-free model keeps the best k options,
and a stronger, slower model chooses among those k.

The two models do not fit on the GPU together, so it runs in two stages:

    1. run_all.sh --only size=32,size=64,size=128 --keep-scores --tag scores -- bev:all ...
    2. python3 -m lab.methods.pipeline build     # cases_pipe.jsonl: each case cut to its short list
    3. run_all.sh --cases lab/methods/cases_pipe.jsonl --tag pipe -- qwen17:two onli2b:one
    4. python3 -m lab.methods.pipeline report    # lab/results/methods/pipeline.md

The short list keeps the options in their ORIGINAL order, so the second model
does not see the first model's ranking. A case whose short list lost every
acceptable option stays in, with no acceptable option: the pipeline fails it.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

from lab.methods.report import wilson

HERE = Path(__file__).parent
RES = HERE.parent / "results" / "methods"
KS = (8, 16)
SIZES = (32, 64, 128)


def build():
    cases = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "cases_things.jsonl")}
    out = []
    for f in sorted(glob.glob(str(RES / "*.scores.jsonl"))):
        s1 = os.path.basename(f)[: -len(".scores.jsonl")]
        for l in open(f):
            r = json.loads(l)
            if "scores" not in r or r["value"] not in SIZES:
                continue
            c = cases[r["id"]]
            best = sorted(range(len(r["scores"])), key=lambda i: -r["scores"][i])
            for k in KS:
                keep = sorted(best[:k])                      # original order
                good = set(c["good"])
                out.append({**{x: c[x] for x in ("kind", "factor", "value", "need", "situation", "question")},
                            "id": f"pipe/{s1}/k{k}/{c['id']}", "stage1": s1.replace("__", ":"), "k": k,
                            "options": [c["options"][i] for i in keep],
                            "names": [c["names"][i] for i in keep],
                            "good": [j for j, i in enumerate(keep) if i in good]})
    with open(HERE / "cases_pipe.jsonl", "w") as f:
        for c in out:
            f.write(json.dumps(c) + "\n")
    print(f"{len(out)} short-list cases -> {HERE / 'cases_pipe.jsonl'}")


def _cell(rs):
    if not rs:
        return ""
    p, lo, hi = wilson(sum(bool(r.get("ok")) for r in rs), len(rs))
    return f"{p:.0%} ({lo:.0%}-{hi:.0%})"


def report():
    pcases = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "cases_pipe.jsonl")}
    L = ["# Short list, then choose", "",
         "Share of picks in the acceptable set, 60 cases per cell. `kept` = the short list still "
         "holds an acceptable option (the best the second model can do).", "",
         "| first model (keeps k) | second model | " + " | ".join(f"size {s}" for s in SIZES) + " |",
         "| --- | --- |" + " --- |" * len(SIZES)]
    s1s = sorted({c["stage1"] for c in pcases.values()})
    for s1 in s1s:
        for k in KS:
            ids = {i for i, c in pcases.items() if c["stage1"] == s1 and c["k"] == k}
            kept = [c for i, c in pcases.items() if i in ids]
            L.append(f"| {s1} (k={k}) | kept | " + " | ".join(
                f"{sum(bool(c['good']) for c in kept if c['value'] == s) / max(1, sum(c['value'] == s for c in kept)):.0%}"
                for s in SIZES) + " |")
            for f in sorted(glob.glob(str(RES / "*.pipe.jsonl"))):
                s2 = os.path.basename(f)[: -len(".pipe.jsonl")].replace("__", ":")
                rs = [json.loads(l) for l in open(f)]
                rs = [r for r in rs if r["id"] in ids]
                L.append(f"| {s1} (k={k}) | {s2} | " + " | ".join(
                    _cell([r for r in rs if r["value"] == s]) for s in SIZES) + " |")
    L += ["", "Each model alone, on the full list:", "",
          "| model | " + " | ".join(f"size {s}" for s in SIZES) + " |", "| --- |" + " --- |" * len(SIZES)]
    for n in s1s + sorted({os.path.basename(f)[: -len(".pipe.jsonl")].replace("__", ":")
                           for f in glob.glob(str(RES / "*.pipe.jsonl"))}):
        f = RES / f"{n.replace(':', '__')}.jsonl"
        if f.exists():
            rs = [json.loads(l) for l in open(f)]
            L.append(f"| {n} | " + " | ".join(
                _cell([r for r in rs if r["factor"] == "size" and r["value"] == s and "ok" in r]) for s in SIZES) + " |")
    (RES / "pipeline.md").write_text("\n".join(L) + "\n")
    print(RES / "pipeline.md")


if __name__ == "__main__":
    {"build": build, "report": report}[sys.argv[1]]()
