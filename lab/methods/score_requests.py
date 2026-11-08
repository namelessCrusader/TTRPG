"""Score the requests that a sim run recorded, with a decision model, offline
(docs/sim-mind-design.md 3.5: record, score, replay; steps M6 and M9).

    lab/.venv/bin/python -m lab.methods.score_requests gdecide:all REQUESTS.jsonl DECISIONS.jsonl
    python3 -m lab.methods.score_requests fake REQUESTS.jsonl DECISIONS.jsonl    # no model: for tests

REQUESTS are the case rows that src/mind/record.py writes: one for each
decision of a run. For each row, the scorer gets the situation, the question
and the options, and nothing more (not the rule scores, the tick or who
decides). One decision row for each request goes to DECISIONS:
    request   the id of the request
    digest    copied from the request. The replay uses it to see that the
              decision answers the same question.
    pick      the index of the model's best option
    scores    the model's score of each option (probabilities for
              gdecide), or null when the scorer gives none
    chooser   the name of the scorer
    ms        the time of the forward pass
src/mind/record.py (Replay) reads the file back.

THE ISOLATION RULE (lab/README.md). This file imports nothing from src/,
and src/ imports nothing from the lab: the two files are the only crossing.

ONE MODEL AT A TIME, as lab/methods/run.py runs it. Before a model loads,
the free GPU memory is checked (LAB_DEVICE=cpu runs on the CPU and skips
the check). Rows go out one at a time, and a row that is already in
DECISIONS (the same request and digest) is not scored again: a stopped run
continues, and the requests of a later round go to the same file. The rest
between rows (LAB_DUTY) keeps the laptop cool.

BATCHES (wave O). `--batch N` puts N requests in one forward pass (gdecide
only). Faster for offline scoring, but the texts are padded to the longest,
so the scores can differ from one-at-a-time in the last digits; the default
is 1. The load itself is lab/methods/score.py's (LAB_GLINER_LOAD,
LAB_GLINER_DTYPE); lab/methods/load_bench.py measures both.

SEVERAL FILES, ONE LOAD, IN CHUNKS (step M9). `--also REQUESTS DECISIONS`
adds another pair of files: the model loads once for all of them.
`--budget SECONDS` stops between two rows once that much time has gone, so a
long scoring runs in chunks that let the locks go between them; the next
chunk continues where this one stopped. `--limit N` scores at most N rows.

MEMORY. At the end the scorer prints one line, `memory: {...}`: the peak RSS
of the process, and on the GPU the peak memory torch allocated and reserved,
and what nvidia-smi says the process holds (the CUDA context too). `--mem
FILE` also writes it to FILE. That is what a server of this model would need
(lab/serve.py, step M11).
A thread also reads /proc/self/status and the cgroup's memory.stat every
20 ms, so the line says WHAT the peak was made of: the peak of anonymous
memory (what the process really holds) apart from file pages (the weights'
file and the libraries, which the kernel can drop and read again), and the
same for the cgroup's charge.

THE FAKE SCORER has no model, so it needs no torch and no GPU. The score of
an option comes from a hash of its words, and the scores of one request sum
to 1. The tests of the sim run it in a separate process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

NEEDS = ("id", "situation", "question", "options", "digest")   # what a request row must have


class Fake:
    """No model: each option's score comes from a hash of its words. The same
    options get the same scores, in every process."""
    params = 0

    def load(self):
        pass

    def rank(self, sit, question, options):
        s = [(int(hashlib.sha256(o.encode()).hexdigest()[:8], 16) % 1000 + 1) / 1001 for o in options]
        total = sum(s)
        p = [x / total for x in s]
        order = sorted(range(len(p)), key=lambda i: -p[i])
        return {"scores": p, "pick": order[0], "order": order, "prob": True}


def make(name):
    """The scorer `name`. Only a real model imports torch (lab.methods.score)."""
    if name == "fake":
        return Fake()
    from lab.methods.score import make as make_model
    return make_model(name)


def requests(path):
    """The request rows of `path`. A row that lacks a field the scorer needs
    is an error."""
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    for n, r in enumerate(rows, 1):
        lack = [k for k in NEEDS if k not in r]
        if lack:
            raise SystemExit(f"{path}:{n}: a request row has {', '.join(NEEDS)};"
                             f" this one lacks {', '.join(lack)}")
    return rows


def _row(name, row, r, ms):
    scores = None if r.get("scores") is None else [float(x) for x in r["scores"]]
    return {"request": row["id"], "digest": row["digest"], "pick": int(r["pick"]), "scores": scores,
            "chooser": name, "ms": round(ms, 3)}


def decide(scorer, name, row):
    """The decision row of one request."""
    t = time.perf_counter()
    r = scorer.rank(row["situation"], row["question"], list(row["options"]))
    return _row(name, row, r, (time.perf_counter() - t) * 1000)


def decide_many(scorer, name, rows):
    """The decision rows of several requests, in one forward pass when the
    scorer has rank_many (gdecide). `ms` is the pass's time shared out."""
    if not hasattr(scorer, "rank_many"):
        return [decide(scorer, name, r) for r in rows]
    t = time.perf_counter()
    rs = scorer.rank_many([(r["situation"], r["question"], list(r["options"])) for r in rows])
    ms = (time.perf_counter() - t) * 1000 / len(rows)
    return [_row(name, row, r, ms) for row, r in zip(rows, rs)]


def _cgroup():
    """The directory of this process's cgroup (cgroup v2), or None."""
    try:
        return Path("/sys/fs/cgroup" + Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1])
    except (OSError, IndexError):
        return None


class Sampler(threading.Thread):
    """Every `dt` seconds: the process's RssAnon and RssFile, and its cgroup's
    anon and file charge (memory.stat). Keeps the peak of each, and the parts
    of the RSS at its own peak (VmHWM says how big the peak was, not what it
    was made of). A daemon: it never keeps the process alive."""
    def __init__(self, dt=0.02):
        super().__init__(daemon=True)
        self.dt, self.peak, self.at_rss_peak, self.phase = dt, {}, {}, "load"
        self.cg = _cgroup()

    def read(self):
        got = {}
        try:
            for x in Path("/proc/self/status").read_text().splitlines():
                k, _, v = x.partition(":")
                if k in ("RssAnon", "RssFile", "RssShmem", "VmRSS"):
                    got[k] = int(v.split()[0]) / 1024
        except (OSError, ValueError):
            pass
        if self.cg is not None:
            try:
                st = dict(x.split() for x in (self.cg / "memory.stat").read_text().splitlines())
                got["cg_anon"] = int(st["anon"]) / 2**20
                got["cg_file"] = int(st["file"]) / 2**20
            except (OSError, KeyError, ValueError):
                pass
        return got

    def run(self):
        while True:
            got = self.read()
            for k, v in got.items():
                self.peak[k] = max(v, self.peak.get(k, 0))
            if got.get("VmRSS", 0) >= self.at_rss_peak.get("VmRSS", 0):
                self.at_rss_peak = dict(got, phase=self.phase)
            time.sleep(self.dt)

    def summary(self):
        p, at = self.peak, self.at_rss_peak
        out = {"rss_anon_peak_mib": p.get("RssAnon"), "rss_file_peak_mib": p.get("RssFile"),
               "at_rss_peak": {"anon_mib": at.get("RssAnon"), "file_mib": at.get("RssFile"),
                               "phase": at.get("phase")}}
        if "cg_anon" in p:
            out["cgroup_anon_peak_mib"] = p["cg_anon"]
            out["cgroup_file_peak_mib"] = p["cg_file"]
        return {k: ({kk: round(vv, 1) if isinstance(vv, float) else vv for kk, vv in v.items()}
                    if isinstance(v, dict) else round(v, 1) if isinstance(v, float) else v)
                for k, v in out.items()}


def memory(on_gpu):
    """The memory of this process, in MiB: the peak RSS and the RSS now
    (anonymous and file-backed: the weights' file pages can be given back),
    the peak of its cgroup (what the kernel charged it: a page that another
    process brought into the page cache is not charged), and on the GPU the peak torch allocated and reserved, and what nvidia-smi
    says the process holds now (with the CUDA context)."""
    out = {"rss_peak_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
    try:                                # now, at the end: what a server would keep holding
        st = dict(x.split(":", 1) for x in Path("/proc/self/status").read_text().splitlines() if ":" in x)
        for k, name in (("VmRSS", "rss_now_mib"), ("RssAnon", "rss_anon_now_mib"),
                        ("RssFile", "rss_file_now_mib")):
            out[name] = round(int(st[k].split()[0]) / 1024, 1)
    except (OSError, KeyError, ValueError):
        pass
    try:                                # the peak of the process's cgroup (lab runs it in a capped scope)
        out["cgroup_peak_mib"] = round(int((_cgroup() / "memory.peak").read_text()) / 2**20, 1)
        # the cgroup's hard cap (cap.sh: no swap), or null with no cap. A run
        # that ends under a cap of N MiB did all its work in N MiB: under the
        # cap the kernel drops file pages before it kills
        cap = (_cgroup() / "memory.max").read_text().strip()
        out["cgroup_max_mib"] = None if cap == "max" else round(int(cap) / 2**20, 1)
    except (OSError, TypeError, ValueError):
        pass
    torch = sys.modules.get("torch")
    if on_gpu and torch is not None and torch.cuda.is_available():
        out["cuda_alloc_peak_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        out["cuda_reserved_peak_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
        try:
            got = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                 timeout=20).stdout
            mine = [int(x.split(",")[1]) for x in got.splitlines()
                    if x.strip() and int(x.split(",")[0]) == os.getpid()]
            out["gpu_process_mib"] = mine[0] if mine else None
        except (OSError, ValueError, subprocess.SubprocessError):
            out["gpu_process_mib"] = None
    return out


def _todo(path, out):
    """The rows of `path` that `out` has no decision for, each question once,
    and the number of decisions `out` has."""
    rows = requests(path)
    done = set()
    if out.exists():
        done = {(d["request"], d["digest"])
                for d in (json.loads(x) for x in out.read_text().splitlines() if x.strip())}
    todo = {}                           # each question once
    for r in rows:
        k = (r["id"], r["digest"])
        if k not in done:
            todo.setdefault(k, r)
    return list(todo.values()), len(done)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m lab.methods.score_requests",
                                 description="Score recorded sim requests with a decision model.")
    ap.add_argument("scorer", help="a scorer of lab.methods.score (gdecide:all), or fake")
    ap.add_argument("requests", help="the case rows that src/mind/record.py wrote")
    ap.add_argument("out", help="the decisions file: new rows are added to it")
    ap.add_argument("--also", nargs=2, action="append", default=[], metavar=("REQUESTS", "OUT"),
                    help="another pair of files, scored after the first by the same model")
    ap.add_argument("--budget", type=float, default=0, help="stop after this many seconds (0: no limit)")
    ap.add_argument("--limit", type=int, default=0, help="score at most this many rows (0: all)")
    ap.add_argument("--mem", help="write the memory line to this file too")
    ap.add_argument("--batch", type=int, default=1,
                    help="requests in one forward pass (gdecide; scores can differ in the last"
                         " digits from one at a time)")
    a = ap.parse_args(argv)

    jobs = []
    for req, out in [(a.requests, a.out)] + [tuple(x) for x in a.also]:
        todo, done = _todo(req, Path(out))
        print(f"{a.scorer}: {len(todo)} to do; {out} has {done} decisions", flush=True)
        if todo:
            jobs.append((Path(out), todo))
    if a.limit:
        left, cut = a.limit, []
        for out, todo in jobs:
            if left > 0:
                cut.append((out, todo[:left]))
                left -= len(cut[-1][1])
        jobs = cut
    if not jobs:
        return
    s = make(a.scorer)
    on_gpu = a.scorer != "fake" and os.environ.get("LAB_DEVICE", "cuda") == "cuda"
    if on_gpu:
        from lab.methods.run import gpu_check
        gpu_check(s.params)
    sampler = Sampler()
    sampler.start()
    t0 = time.time()
    s.load()
    sampler.phase = "score"
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    duty = float(os.environ.get("LAB_DUTY", "0.5"))
    t0, stop = time.time(), False
    for out, todo in jobs:
        n = 0
        with out.open("a") as f:
            for n in range(0, len(todo), a.batch):
                if a.budget and time.time() - t0 > a.budget:
                    stop = True
                    break
                part = todo[n:n + a.batch]
                ds = decide_many(s, a.scorer, part) if a.batch > 1 else [decide(s, a.scorer, part[0])]
                for d in ds:
                    f.write(json.dumps(d) + "\n")
                f.flush()
                if duty < 1:            # the rest between requests (lab.methods.run)
                    time.sleep(sum(d["ms"] for d in ds) / 1000 * (1 / duty - 1))
                if n // a.batch % max(1, 100 // a.batch) == 0:
                    print(f"  {n}/{len(todo)}", flush=True)
            else:
                n = len(todo)
        print(f"{'stopped at the budget' if stop else 'done'}: {n} of {len(todo)} decisions in {out}",
              flush=True)
        if stop:
            break
    mem = memory(on_gpu) | sampler.summary()
    print("memory: " + json.dumps(mem), flush=True)
    if a.mem:
        Path(a.mem).write_text(json.dumps(mem) + "\n")


if __name__ == "__main__":
    main()
