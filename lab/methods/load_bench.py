"""How heavy and how fast is GLiNER2.5-Decide's load and its scoring (wave O,
lane MODEL). Each run is ONE process: one load, then N recorded requests.

    lab/.venv/bin/python -m lab.methods.load_bench run OUT_DIR --req A.jsonl --req B.jsonl --n 500 [--batch 8]
    python3 -m lab.methods.load_bench compare A_DECISIONS B_DECISIONS
    python3 -m lab.methods.load_bench table METRICS.json ...
    python3 -m lab.methods.load_bench gate BASE_NUMBERS.json NEW_NUMBERS.json

The load is chosen as score.py chooses it: LAB_GLINER_LOAD (direct or old)
and LAB_GLINER_DTYPE (fp16 by default on the GPU, fp32, bf16). A run writes OUT_DIR/decisions.jsonl
(the rows lab.methods.score_requests would write) and OUT_DIR/metrics.json:
    import_s, load_s        the time to import torch and the package, and to load
    load                    during the load: the peak RSS and its anonymous and
                            file parts (score_requests' Sampler), and the peak
                            that torch allocated and reserved on the GPU
    after_load              RSS now (all, anonymous, file) and what nvidia-smi
                            says the process holds
    score                   the same peaks while scoring, the median and 90th
                            percentile ms per request, and requests per second
    steady                  RSS and the GPU after the last request
`--n N` takes N requests split evenly over the --req files, the first ones of
each file. `--batch B` scores B requests in one forward pass (rank_many).

COMPARE matches two decision files by request and digest: how many picks
agree (with a Wilson 95% interval), and how far the scores are apart (the
largest difference, and how many requests have bit-identical scores).

GATE compares two lab.methods.m9_gate reports (numbers.json) of the same
requests scored two ways: every rate (a dict with n, k and rate) of the new
one against the base's 95% interval (its own ci95, or a Wilson interval),
and the verdict. `moved` lists the rates that left the interval.

The locks are the caller's (the GPU lock, the test lock, the heat guard).
Like the scorer, this imports nothing from src/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def _status():
    """RSS now, in MiB: all, anonymous, file, shared (the CUDA driver's)."""
    st = dict(x.split(":", 1) for x in Path("/proc/self/status").read_text().splitlines() if ":" in x)
    return {k: round(int(st[s].split()[0]) / 1024, 1)
            for k, s in (("rss", "VmRSS"), ("anon", "RssAnon"), ("file", "RssFile"),
                         ("shmem", "RssShmem"))}


def _gpu_process_mib():
    try:
        got = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=20).stdout
        mine = [int(x.split(",")[1]) for x in got.splitlines()
                if x.strip() and int(x.split(",")[0]) == os.getpid()]
        return mine[0] if mine else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _rows(files, n):
    from lab.methods.score_requests import requests
    each = math.ceil(n / len(files))
    out = []
    for f in files:
        out += requests(f)[:each]
    return out[:n]


def _peaks(sampler, torch, on_gpu):
    """The sampler's peaks since its last reset, and torch's GPU peaks."""
    p = sampler.peak
    out = {k: round(p.get(s, 0), 1) for k, s in (("rss_peak_mib", "VmRSS"), ("anon_peak_mib", "RssAnon"),
                                                 ("file_peak_mib", "RssFile"),
                                                 ("shmem_peak_mib", "RssShmem"))}
    if "cg_anon" in p:
        out["cgroup_anon_peak_mib"] = round(p["cg_anon"], 1)
        out["cgroup_file_peak_mib"] = round(p["cg_file"], 1)
    if on_gpu:
        out["cuda_alloc_peak_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        out["cuda_reserved_peak_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    return out


def run(a):
    from lab.methods.score_requests import Sampler, decide, decide_many
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = _rows(a.req, a.n)
    sampler = Sampler(dt=0.01)
    sampler.start()
    t = time.perf_counter()
    import torch
    from lab.methods.score import make
    import gliner2  # noqa: F401
    import_s = time.perf_counter() - t
    on_gpu = os.environ.get("LAB_DEVICE", "cuda") == "cuda"
    s = make("gdecide:all")
    sampler.peak = {}
    t = time.perf_counter()
    s.load()
    if on_gpu:
        torch.cuda.synchronize()
    load_s = time.perf_counter() - t
    time.sleep(0.05)                      # one more sample of the end of the load
    m = {"how": s.how, "dtype": s.dtype, "batch": a.batch, "n": len(rows),
         "import_s": round(import_s, 2), "load_s": round(load_s, 2),
         "load": _peaks(sampler, torch, on_gpu),
         "after_load": _status() | ({"gpu_process_mib": _gpu_process_mib(),
                                     "cuda_alloc_mib": round(torch.cuda.memory_allocated() / 2**20, 1)}
                                    if on_gpu else {})}
    sampler.peak = {}
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    duty = float(os.environ.get("LAB_DUTY", "0.5"))
    ds, t = [], time.perf_counter()
    busy = 0.0
    for i in range(0, len(rows), a.batch):
        part = rows[i:i + a.batch]
        got = decide_many(s, "gdecide:all", part) if a.batch > 1 else [decide(s, "gdecide:all", part[0])]
        ds += got
        spent = sum(d["ms"] for d in got) / 1000
        busy += spent
        if duty < 1:                      # the rest between requests (score_requests)
            time.sleep(spent * (1 / duty - 1))
    ms = [d["ms"] for d in ds]
    m["score"] = _peaks(sampler, torch, on_gpu) | {
        "ms_median": round(statistics.median(ms), 2),
        "ms_p90": round(sorted(ms)[int(0.9 * (len(ms) - 1))], 2),
        "per_s": round(len(ds) / busy, 1)}
    m["steady"] = _status() | ({"gpu_process_mib": _gpu_process_mib()} if on_gpu else {})
    (out / "decisions.jsonl").write_text("".join(json.dumps(d) + "\n" for d in ds))
    (out / "metrics.json").write_text(json.dumps(m, indent=1) + "\n")
    print(json.dumps(m), flush=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def compare(a_path, b_path):
    """Picks and scores of two decision files, matched by request and digest."""
    def load(p):
        return {(d["request"], d["digest"]): d
                for d in (json.loads(x) for x in Path(p).read_text().splitlines() if x.strip())}
    a, b = load(a_path), load(b_path)
    keys = sorted(set(a) & set(b))
    same = sum(a[k]["pick"] == b[k]["pick"] for k in keys)
    diffs = [max((abs(x - y) for x, y in zip(a[k]["scores"], b[k]["scores"])), default=0.0)
             for k in keys]
    return {"n": len(keys), "only_a": len(set(a) - set(b)), "only_b": len(set(b) - set(a)),
            "same_pick": same, "same_pick_rate": round(same / len(keys), 5) if keys else None,
            "same_pick_ci95": wilson(same, len(keys)),
            "bit_identical": sum(a[k]["scores"] == b[k]["scores"] for k in keys),
            "max_score_diff": max(diffs, default=0.0),
            "median_score_diff": statistics.median(diffs) if diffs else 0.0}


def gate(base_path, new_path):
    """Each rate of the M9 report NEW against the 95% interval of BASE."""
    base, new = json.loads(Path(base_path).read_text()), json.loads(Path(new_path).read_text())
    rates, moved = {}, []

    def walk(b, n, pre=""):
        for k, v in b.items():
            if not isinstance(v, dict) or not isinstance(n.get(k), dict):
                continue
            if {"n", "k", "rate"} <= set(v):
                ci = v.get("ci95") or wilson(v["k"], v["n"])
                r = n[k]["rate"]
                rates[pre + k] = {"base": round(v["rate"], 5), "new": round(r, 5), "base_ci95": ci,
                                  "n": [v["n"], n[k]["n"]], "inside": ci[0] <= r <= ci[1]}
                if not rates[pre + k]["inside"]:
                    moved.append(pre + k)
            else:
                walk(v, n[k], pre + k + ".")
    walk(base, new)
    return {"rates": rates, "moved": moved, "gate_base": base.get("gate"), "gate_new": new.get("gate")}


def table(paths):
    """The median of each number over several runs' metrics."""
    ms = [json.loads(Path(p).read_text()) for p in paths]

    def flat(d, pre=""):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from flat(v, f"{pre}{k}.")
            elif isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None:
                yield f"{pre}{k}", v
    cols = {}
    for m in ms:
        for k, v in flat(m):
            cols.setdefault(k, []).append(v)
    return {k: {"median": statistics.median(v), "runs": v} for k, v in cols.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m lab.methods.load_bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("out")
    r.add_argument("--req", action="append", required=True)
    r.add_argument("--n", type=int, default=500)
    r.add_argument("--batch", type=int, default=1)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    t = sub.add_parser("table")
    t.add_argument("metrics", nargs="+")
    g = sub.add_parser("gate")
    g.add_argument("base")
    g.add_argument("new")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a)
    elif a.cmd == "compare":
        print(json.dumps(compare(a.a, a.b), indent=1))
    elif a.cmd == "gate":
        print(json.dumps(gate(a.base, a.new), indent=1))
    else:
        print(json.dumps(table(a.metrics), indent=1))


if __name__ == "__main__":
    sys.exit(main())
