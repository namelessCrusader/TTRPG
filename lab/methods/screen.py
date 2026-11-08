"""A standing screen of small decision models, on one fixed test set (wave 10, LAB10).

    python3 -m lab.methods.screen build                 # the fixed test set, lab/results/screen/q
    lab/.venv/bin/python -m lab.methods.screen run MODEL [--budget S] [--limit N]
    python3 -m lab.methods.screen report MODEL          # one row of lab/results/screen/leaderboard.jsonl
    python3 -m lab.methods.screen table                 # the leaderboard as a table

New sub-1B decision models keep appearing (the user, 2026-10-10). A new model
costs ONE ADAPTER below (in the pattern of lab/methods/score.py: load() and
rank(situation, question, options)) and one entry in MODELS, and then about
20 GPU minutes: `run` in chunks under the locks (lab/results/describe_ab/job.sh
runs one chunk), then `report`.

THE FIXED TEST SET (q/, made once by `build`; a model is only compared on it):
    labeled      the labeled cases of the describe A/B (lab/methods/describe_ab.py),
                 in the src describe's words (what the engine sends a model), with
                 their acceptable sets: accuracy@1
    labeled_rev  every 4th of them with the options reversed: position bias
    m9           500 of M9's recorded sim requests (lab/results/m9/q: the first 250
                 of the inn and of the voxel scenes, as lab/methods/load_bench.py
                 takes them). No acceptable set: the time per decision and the
                 memory on real menus
    m9_rev       the same requests reversed (M9's "#p1" rows): position bias on
                 real menus
A menu that a model refuses (Laya: more option text than its head takes) is
written with pick -1 and counts as a wrong pick; the row says how many.
The rows are copied with their ids and digests, so a decision already made for
the same question elsewhere (the describe A/B) is the same decision.

THE ROW of a model: its size and license; accuracy@1 with a Wilson 95%
interval; how often the pick moves when the options are reversed (labeled
and M9); the median ms per decision (M9 and labeled); the peak RAM (RSS, and
its anonymous part) and VRAM over its chunks. A model that gives a set of
answers and an abstain flag (VegaML) also gets: how often its set holds an
acceptable answer, the mean set size, how often it abstains, and its accuracy
when it does not.

Like the scorer, this file imports nothing from src/ (lab/README.md).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE.parent / "results" / "screen"
AB = HERE.parent / "results" / "describe_ab"
M9 = HERE.parent / "results" / "m9" / "q"
FILES = ("labeled", "labeled_rev", "m9", "m9_rev")
N_M9 = 500

# THE BOOKS SET (wave 11b, the book pilot). Labeled decision cases from books
# the user owns (tools/export_book_cases.py), each asked in six ways (the
# formulations), with the same options in the same order. Everything with
# book-derived text in it (the requests, and the decisions next to them)
# stays in the git-ignored books/derived/; only numbers come out, into
# leaderboard_books.jsonl here. The default set above is untouched by it.
#   python3 -m lab.methods.screen build --set books
#   lab/.venv/bin/python -m lab.methods.screen run MODEL --set books [--only a_lab b_src ...]
#   python3 -m lab.methods.screen report MODEL --set books [--only ...]
#   python3 -m lab.methods.screen table --set books
BOOKS = HERE.parent.parent / "books" / "derived"
BOOK_Q = BOOKS / "screen" / "q"
BOOK_D = BOOKS / "screen" / "d"
BOOK_FORMS = ("a_lab", "b_src", "c_narrative", "d_traits", "e_bio", "f_past_acts")
BOOK_REV_EVERY = 4          # every 4th case asked again with the options reversed
# Cases whose persona or scene gives the outcome away (the pilot's quality
# check, 2026-10-16: past acts, beliefs, holds or narrative taken from after
# the decision). Scored, but left out of every number.
BOOK_LEAKS = ("locke/008/10", "locke/006/7", "lattes/001/9", "locke/006/0", "lattes/004/39", "locke/006/11")
SETS = {"default": FILES, "books": BOOK_FORMS + tuple(f"{f}_rev" for f in BOOK_FORMS),
        "books_v2": (), "books_v3": ()}   # their files are whatever build_books_v2 wrote (see below)

# THE BOOKS_V2 SET (wave 11b, recipe v2). The v2 case set (books/derived/v2/
# cases_v2.jsonl) posed in every way tools/export_book_cases.py --v2 writes
# (books/derived/v2/formulations_v2/, its manifest says each file in a line),
# plus the minimal pairs' variants (formulations_v2/pairs/) when they exist.
# Each option has a GRADE (the mean of three blind raters, 0-3), so a pick is
# scored two ways: acc@1 (the pick is in the acceptable set, mean grade >= 2)
# and its grade. A file whose texts equal another's costs nothing: a
# decision is reused by its digest (the same situation, question and options
# give the same decision), within the model. A row with `menu2` is decided in
# two steps (the kind of act, then the act among that kind's options).
#   python3 -m lab.methods.screen build --set books_v2 [--root DIR]
#   lab/.venv/bin/python -m lab.methods.screen run MODEL --set books_v2 [--only a_lab 'no_*' ...] [--limit N]
#   python3 -m lab.methods.screen report MODEL --set books_v2 [--only ...]
#   python3 -m lab.methods.screen table --set books_v2
# --root points the set at another tree with the same layout (the dry run's).
BOOK2_ROOT = BOOKS / "v2"
BOOK2_REV = ("a_lab", "ctx_full")   # asked again with the options reversed (every BOOK_REV_EVERY-th case)
BOOK2_BOARD = OUT / "leaderboard_books_v2.jsonl"

# THE BOOKS_V3 SET (books wave 3, recipe v2.1). The same machinery as books_v2, pointed at
# books/derived/v3/ (cases_v3.jsonl: 17 new books plus the fixed pilot): its request files
# come from formulations_v3/ (python3 tools/export_book_cases.py --v3), its decisions go to
# v3/screen/d/<model>/, its board is leaderboard_books_v3.jsonl. The pilot cases the fix pass
# left unchanged take the v2 decision of the same model by digest (`reused`, no GPU time).
# No reversed repeats. books_v2 and the other sets score exactly as before.
#   python3 -m lab.methods.screen build --set books_v3
#   lab/.venv/bin/python -m lab.methods.screen run MODEL --set books_v3 [--only ctx_full ...] [--budget S]
#   python3 -m lab.methods.screen report MODEL --set books_v3
BOOK3_ROOT = BOOKS / "v3"
BOOK3_BOARD = OUT / "leaderboard_books_v3.jsonl"


def _is_v3(root):
    return root is not None and Path(root).resolve() == BOOK3_ROOT.resolve()


# ---------------------------------------------------------------- adapters
class VegaML:
    """NandhaKishorM/vegaml 0.6.1 (Apache-2.0): a frozen Qwen3.5-0.8B read to
    layer 19 and a small trained "physics" decision engine (57 MB, plus two
    adapters it gates itself), zero-shot. Every answer has calibrated
    probabilities, a split-conformal answer set at alpha 0.1 and an abstain
    flag. Nothing is trained here (training its head on our data is the
    user's decision). The package is not in lab/.venv: put it on PYTHONPATH
    (`pip download vegaml==0.6.1 --no-deps` and unzip the wheel)."""
    params = 0.87e9

    def load(self):
        import vegaml
        self.v = vegaml.load()

    def rank(self, sit, question, options):
        q = {"q": {"type": "choice", "instructions": question, "criteria": {o: None for o in options}}}
        a = self.v.decide(sit, q)["answers"]["q"]
        p = [a["probs"][o] for o in options]
        inset = set(a["prediction_set"] or [])
        return {"scores": p, "pick": max(range(len(p)), key=p.__getitem__), "prob": True,
                "set": [i for i, o in enumerate(options) if o in inset], "abstain": a["abstain"],
                "unbound": a["unbound"]}


class GLiClass:
    """knowledgator/gliclass-modern-base-v3.0 (Apache-2.0): the options ride in
    the input as labels, one pass. It has no question slot, so the question is
    written above the situation (as lab/models.py does)."""
    params = 0.15e9
    repo = "knowledgator/gliclass-modern-base-v3.0"

    def load(self):
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline
        from transformers import AutoTokenizer
        m = GLiClassModel.from_pretrained(self.repo)
        tok = AutoTokenizer.from_pretrained(self.repo, add_prefix_space=True)
        self.pipe = ZeroShotClassificationPipeline(m, tok, classification_type="single-label",
                                                   device="cuda:0" if os.environ.get("LAB_DEVICE", "cuda") == "cuda"
                                                   else "cpu")

    def rank(self, sit, question, options):
        got = self.pipe(f"{question}\n{sit}", list(options), threshold=0.0)[0]
        s = {g["label"]: g["score"] for g in got}
        p = [float(s.get(o, 0.0)) for o in options]
        return {"scores": p, "pick": max(range(len(p)), key=p.__getitem__), "prob": False}


class Scored:
    """A model of lab/methods/score.py by its name there ("gdecide:all")."""
    def __init__(self, name):
        from lab.methods.score import make
        self.inner = make(name)
        self.params = self.inner.params

    def load(self):
        self.inner.load()

    def rank(self, sit, question, options):
        return self.inner.rank(sit, question, options)


# name: (how to make it, its size, its license). The baselines are the models
# already in the cache.
MODELS = {
    "vegaml": (VegaML, "0.8B frozen Qwen3.5 + 14M engine", "Apache-2.0 (vegaml and Qwen3.5-0.8B)"),
    "gdecide": (lambda: Scored("gdecide:all"), "0.34B", "Apache-2.0"),
    "gdecide1b": (lambda: Scored("gdecide1b:all"), "1B", "Apache-2.0"),
    "gliclass": (GLiClass, "0.15B", "Apache-2.0"),
    "bev": (lambda: Scored("bev:all"), "0.4B", "CC-BY-NC-4.0 (testing only)"),
    "laya": (lambda: Scored("laya_en:all"), "0.42B", "see the model card"),
    "von": (lambda: Scored("von:all"), "0.4B", "see the model card"),
    "qwen06": (lambda: Scored("qwen06:one"), "0.6B", "Apache-2.0"),
    # no model: the option sharing most content words with the situation (the
    # word-overlap leak check, and the books_v2 dry run's chooser)
    "overlap": (lambda: Scored("overlap:all"), "0", "none (no model)"),
}


# ---------------------------------------------------------------- the test set
def rows(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def write(path, rs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rs))


def build(out=OUT):
    man = rows(AB / "q" / "manifest.jsonl")
    if not man:
        raise SystemExit("no describe A/B manifest: run lab.methods.describe_ab build first")
    good = {m["id"]: m["good"] for m in man[1:]}
    lab = [dict(r, good=good[r["id"].rsplit("#", 1)[0]]) for r in rows(AB / "q" / "new.jsonl")]
    rev = [dict(r, good=good[r["id"].split("#")[0]]) for r in rows(AB / "q" / "new_perm.jsonl")
           if r["id"].endswith("#p1")]
    each = math.ceil(N_M9 / 2)
    m9 = (rows(M9 / "rooms.jsonl")[:each] + rows(M9 / "voxel.jsonl")[:each])[:N_M9]
    ids = {r["id"] for r in m9}
    m9_rev = [r for f in ("rooms_perm.jsonl", "voxel_perm.jsonl") for r in rows(M9 / f)
              if r["id"].endswith("#p1") and r["id"][:-3] in ids]
    out = Path(out)
    for name, rs in zip(FILES, (lab, rev, m9, m9_rev)):
        write(out / "q" / f"{name}.jsonl", rs)
        print(f"{name}: {len(rs)} requests")
    (out / "q" / "source.json").write_text(json.dumps({
        "labeled": "describe_ab q/new.jsonl (the src describe)", "src": man[0]["src"],
        "m9": f"the first {each} of m9/q/rooms.jsonl and of m9/q/voxel.jsonl"}, indent=1) + "\n")


# ---------------------------------------------------------------- scoring
def run(model, budget=0.0, limit=0, out=OUT, set_="default", only=()):
    """`set_` "books" scores the book pilot's files instead (see BOOKS below);
    `only` keeps those of the set's files."""
    from lab.methods.score_requests import Sampler, _todo, memory
    out = Path(out)
    qd, dd = (out / "q", out / "d") if set_ == "default" else (BOOK_Q, BOOK_D)
    d = dd / model
    d.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name in SETS[set_]:
        if only and name not in only:
            continue
        todo, done = _todo(qd / f"{name}.jsonl", d / f"{name}.jsonl")
        print(f"{model} {name}: {len(todo)} to do, {done} done", flush=True)
        if todo:
            jobs.append((d / f"{name}.jsonl", todo))
    if limit:
        jobs = [(p, t[:limit]) for p, t in jobs]
    if not jobs:
        return
    s = MODELS[model][0]()
    on_gpu = os.environ.get("LAB_DEVICE", "cuda") == "cuda"
    if on_gpu:
        from lab.methods.run import gpu_check
        gpu_check(s.params)
    sampler = Sampler()
    sampler.start()
    t0 = time.time()
    s.load()
    load_s = time.time() - t0
    sampler.phase = "score"
    print(f"loaded in {load_s:.1f}s", flush=True)
    duty = float(os.environ.get("LAB_DUTY", "0.5"))
    t0, stop, n_done = time.time(), False, 0
    for path, todo in jobs:
        with path.open("a") as f:
            for r in todo:
                if budget and time.time() - t0 > budget:
                    stop = True
                    break
                t = time.perf_counter()
                try:
                    got = s.rank(r["situation"], r["question"], list(r["options"]))
                except ValueError as e:         # the model refuses this menu (too long for it)
                    got = {"pick": -1, "error": str(e)[:200]}
                ms = (time.perf_counter() - t) * 1000
                row = {"request": r["id"], "digest": r["digest"], "pick": int(got["pick"]),
                       "scores": None if got.get("scores") is None else [float(x) for x in got["scores"]],
                       "chooser": model, "ms": round(ms, 3)}
                for k in ("set", "abstain", "unbound", "error"):
                    if k in got:
                        row[k] = got[k]
                f.write(json.dumps(row) + "\n")
                f.flush()
                n_done += 1
                if duty < 1:
                    time.sleep(ms / 1000 * (1 / duty - 1))
                if n_done % 200 == 0:
                    print(f"  {n_done} ({path.name})", flush=True)
        if stop:
            break
    mem = memory(on_gpu) | sampler.summary() | {"load_s": round(load_s, 2), "decisions": n_done}
    print(("stopped at the budget" if stop else "done") + f": {n_done} decisions", flush=True)
    print("memory: " + json.dumps(mem), flush=True)
    k = len(list(d.glob("mem_*.json")))
    (d / f"mem_{k + 1}.json").write_text(json.dumps(mem) + "\n")


# ---------------------------------------------------------------- the row
def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def _moves(q_rev, d_base, d_rev):
    """How often the pick names another option when the options are reversed."""
    base = {r["request"]: r for r in d_base}
    rev = {r["request"]: r for r in d_rev}
    n = moved = 0
    for r in q_rev:
        d, b = rev.get(r["id"]), base.get(r["id"].rsplit("#p", 1)[0])
        if d is None or b is None or d["pick"] < 0 or b["pick"] < 0:
            continue
        n += 1
        moved += r["perm"][d["pick"]] != b["pick"]
    return {"n": n, "moved": round(moved / n, 4) if n else None, "ci": wilson(moved, n)}


def report(model, out=OUT):
    out = Path(out)
    d = out / "d" / model
    q = {n: rows(out / "q" / f"{n}.jsonl") for n in FILES}
    dec = {n: rows(d / f"{n}.jsonl") for n in FILES}
    good = {r["id"]: set(r["good"]) for r in q["labeled"]}
    lab = [x for x in dec["labeled"] if x["request"] in good]
    hits = [x["pick"] in good[x["request"]] for x in lab]
    mems = [json.loads(p.read_text()) for p in sorted(d.glob("mem_*.json"))]

    def peak(k):
        xs = [x.get(k) for x in mems if x.get(k) is not None]
        return max(xs) if xs else None

    row = {"model": model, "size": MODELS[model][1], "license": MODELS[model][2],
           "when": time.strftime("%Y-%m-%d"),
           "complete": {n: f"{len(dec[n])}/{len(q[n])}" for n in FILES},
           "acc1": round(sum(hits) / len(hits), 4) if hits else None, "acc1_ci": wilson(sum(hits), len(hits)),
           "n_labeled": len(hits),
           # a menu the model refuses (too long for it) counts as a wrong pick
           "refused": {n: sum(x["pick"] < 0 for x in dec[n]) for n in FILES},
           "chance": round(sum(len(good[r["id"]]) / len(r["options"]) for r in q["labeled"]) / len(q["labeled"]), 4),
           "moves_labeled": _moves(q["labeled_rev"], dec["labeled"], dec["labeled_rev"]),
           "moves_m9": _moves(q["m9_rev"], dec["m9"], dec["m9_rev"]),
           "ms_m9_median": round(statistics.median(x["ms"] for x in dec["m9"]), 2) if dec["m9"] else None,
           "ms_labeled_median": round(statistics.median(x["ms"] for x in lab), 2) if lab else None,
           "ram_rss_peak_mib": peak("rss_peak_mib"), "ram_anon_peak_mib": peak("rss_anon_peak_mib"),
           "vram_alloc_peak_mib": peak("cuda_alloc_peak_mib"),
           "vram_process_mib": peak("gpu_process_mib"), "load_s": peak("load_s")}
    if lab and "set" in lab[0]:
        cover = [bool(set(x["set"]) & good[x["request"]]) for x in lab]
        sizes = [len(x["set"]) for x in lab]
        keep = [h for h, x in zip(hits, lab) if not x["abstain"]]
        row["conformal"] = {"alpha": 0.1, "covers": round(sum(cover) / len(cover), 4),
                            "covers_ci": wilson(sum(cover), len(cover)),
                            "mean_set_size": round(sum(sizes) / len(sizes), 3),
                            "mean_options": round(sum(len(r["options"]) for r in q["labeled"]) / len(q["labeled"]), 2),
                            "abstain_rate": round(1 - len(keep) / len(lab), 4),
                            "acc1_when_answering": round(sum(keep) / len(keep), 4) if keep else None,
                            "acc1_when_answering_ci": wilson(sum(keep), len(keep)),
                            "unbound_rate": round(sum(bool(x.get("unbound")) for x in lab) / len(lab), 4)}
    board = [r for r in rows(out / "leaderboard.jsonl") if r["model"] != model] + [row]
    write(out / "leaderboard.jsonl", board)
    print(json.dumps(row, indent=1))
    return row


def table(out=OUT):
    """The leaderboard. The last column is the lab's own calibration of each
    model's scores (lab/methods/calibrate.py, wave 11: a temperature and a
    split-conformal threshold at alpha 0.1, judged on the other half of the
    labeled cases), when it has run."""
    def pct(x):
        return "-" if x is None else f"{100 * x:.1f}"
    calib = Path(out).parent / "calib" / "numbers.json"
    calib = json.loads(calib.read_text()) if calib.exists() else {}
    print("| model | size | license | acc@1 % [95% CI] | moves rev labeled % | moves rev M9 % |"
          " ms M9 | ms labeled | RAM peak MiB (anon) | VRAM peak MiB (torch / nvidia-smi) | conformal covers % / mean set |"
          " calibrated: covers % / mean set / abstain % |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows(Path(out) / "leaderboard.jsonl"):
        if r.get("status"):                 # tried, and not screened: the row says why
            print(f"| {r['model']} | {r['size']} | {r['license']} | {r['status']} |" + " |" * 8)
            continue
        k = next((k for k in calib if k.split(" ")[0] == r["model"] and "wave 10 text" in k), None)
        t = calib[k]["test"] if k else None
        cal = f"{pct(t['coverage'])} / {t['mean_set']} / {pct(t['abstain'])}" if t else "-"
        c = r.get("conformal")
        print(f"| {r['model']} | {r['size']} | {r['license']} | {pct(r['acc1'])} "
              f"[{pct(r['acc1_ci'][0])}-{pct(r['acc1_ci'][1])}] | {pct(r['moves_labeled']['moved'])} | "
              f"{pct(r['moves_m9']['moved'])} | {r['ms_m9_median']} | {r['ms_labeled_median']} | "
              f"{r['ram_rss_peak_mib']} ({r['ram_anon_peak_mib']}) | {r['vram_alloc_peak_mib']} / {r['vram_process_mib']} | "
              + (f"{pct(c['covers'])} / {c['mean_set_size']}" if c else "-") + f" | {cal} |")


# ---------------------------------------------------------------- the books set
def _digest(situation, question, options):
    """src/mind/record.py `digest`, as lab/methods/describe_ab.py writes it out."""
    import hashlib
    data = json.dumps([situation, question, list(options)], ensure_ascii=False)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def build_books():
    """One request file per formulation (every case, ids "<case>#<form>") and its
    reversed sample ("<case>#<form>#p1", every BOOK_REV_EVERY-th case)."""
    for f in BOOK_FORMS:
        cs = rows(BOOKS / "formulations" / f"{f}.jsonl")
        if not cs:
            raise SystemExit(f"no {BOOKS}/formulations/{f}.jsonl: run tools/export_book_cases.py first")
        q, rev = [], []
        for i, c in enumerate(cs):
            opts = list(c["options"])
            q.append({"id": f"{c['id']}#{f}", "situation": c["situation"], "question": c["question"],
                      "options": opts, "digest": _digest(c["situation"], c["question"], opts), "good": c["good"]})
            if i % BOOK_REV_EVERY == 0 and len(opts) > 1:
                order = list(range(len(opts)))[::-1]
                ro = [opts[k] for k in order]
                rev.append({"id": f"{c['id']}#{f}#p1", "situation": c["situation"], "question": c["question"],
                            "options": ro, "digest": _digest(c["situation"], c["question"], ro), "perm": order,
                            "good": [order.index(g) for g in c["good"]]})
        write(BOOK_Q / f"{f}.jsonl", q)
        write(BOOK_Q / f"{f}_rev.jsonl", rev)
        print(f"books {f}: {len(q)} requests, {len(rev)} reversed")


def book_cases():
    """The pilot's cases by id, each with the facts the books numbers need:
    its acceptable set, the book's choice, the obvious act, its option lengths,
    and whether it is length-fair (the book's option is shorter than the
    longest one, so picking the longest never lands on it)."""
    got = {}
    for c in rows(BOOKS / "cases_books.jsonl"):
        n = [len(o) for o in c["options"]]
        top = max(n)
        got[c["id"]] = {"good": set(c["good"]), "book": c["book"], "kind": c["factor"], "choice": c["book_choice"],
                        "obvious": c["obvious"], "nvo": bool(c["nature_vs_obvious"]), "n": len(n),
                        "longest": [i for i, x in enumerate(n) if x == top], "first_longest": n.index(top),
                        "fair": n[c["book_choice"]] < top, "leak": c["id"] in BOOK_LEAKS}
    return got


def _book_block(ds, meta):
    """accuracy@1 and its controls on decisions `ds` ({case id: decision})."""
    ids = list(ds)
    if not ids:
        return {"n": 0}
    hits = [ds[i]["pick"] in meta[i]["good"] for i in ids]
    lb = [meta[i]["first_longest"] in meta[i]["good"] for i in ids]
    longp = [ds[i]["pick"] in meta[i]["longest"] for i in ids]
    book = [ds[i]["pick"] == meta[i]["choice"] for i in ids]
    return {"n": len(ids), "acc1": round(sum(hits) / len(ids), 4), "acc1_ci": wilson(sum(hits), len(ids)),
            "chance": round(sum(len(meta[i]["good"]) / meta[i]["n"] for i in ids) / len(ids), 4),
            "longest_baseline": round(sum(lb) / len(ids), 4),
            "book_choice": round(sum(book) / len(ids), 4),
            "picks_longest": round(sum(longp) / len(ids), 4),
            "picks_longest_chance": round(sum(len(meta[i]["longest"]) / meta[i]["n"] for i in ids) / len(ids), 4)}


def book_decisions(model, form):
    """{case id: decision} of `model` on formulation `form` (canonical order)."""
    return {r["request"].rsplit("#", 1)[0]: r for r in rows(BOOK_D / model / f"{form}.jsonl")}


def report_books(model, forms=BOOK_FORMS):
    meta = book_cases()
    d = BOOK_D / model
    mems = [json.loads(p.read_text()) for p in sorted(d.glob("mem_*.json"))]

    def peak(k):
        xs = [x.get(k) for x in mems if x.get(k) is not None]
        return max(xs) if xs else None

    got = []
    for f in forms:
        dec = book_decisions(model, f)
        if not dec:
            continue
        use = {i: x for i, x in dec.items() if i in meta and not meta[i]["leak"]}
        fair = {i: x for i, x in use.items() if meta[i]["fair"]}
        nvo = [x for i, x in use.items() if meta[i]["nvo"] and meta[i]["obvious"] is not None]
        q_rev = rows(BOOK_Q / f"{f}_rev.jsonl")
        d_rev = rows(d / f"{f}_rev.jsonl")
        row = {"set": "books", "model": model, "formulation": f, "size": MODELS[model][1],
               "license": MODELS[model][2], "when": time.strftime("%Y-%m-%d"),
               "complete": f"{len(dec)}/{len(meta)}", "leaks_left_out": len(dec) - len(use),
               "refused": sum(x["pick"] < 0 for x in dec.values()),
               "all": _book_block(use, meta), "fair": _book_block(fair, meta),
               "nature_vs_obvious": {"n": len(nvo),
                                     "picks_book": round(sum(x["pick"] == meta[x["request"].rsplit("#", 1)[0]]["choice"]
                                                             for x in nvo) / len(nvo), 4) if nvo else None,
                                     "picks_obvious": round(sum(x["pick"] == meta[x["request"].rsplit("#", 1)[0]]["obvious"]
                                                                for x in nvo) / len(nvo), 4) if nvo else None},
               "moves_rev": _moves(q_rev, list(dec.values()), d_rev) if d_rev else None,
               "ms_median": round(statistics.median(x["ms"] for x in dec.values()), 2),
               "ram_rss_peak_mib": peak("rss_peak_mib"), "vram_alloc_peak_mib": peak("cuda_alloc_peak_mib")}
        got.append(row)
    keys = {(r["model"], r["formulation"]) for r in got}
    path = OUT / "leaderboard_books.jsonl"
    write(path, [r for r in rows(path) if (r["model"], r["formulation"]) not in keys] + got)
    for r in got:
        print(json.dumps(r))
    return got


def table_books(out=OUT):
    def pct(x):
        return "-" if x is None else f"{100 * x:.1f}"
    print("| model | formulation | acc@1 fair % [95% CI] | n fair | chance / longest pick, fair % |"
          " acc@1 all % [95% CI] | longest pick, all % | picks longest % (chance) | book over obvious % |"
          " moves rev % | ms |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows(Path(out) / "leaderboard_books.jsonl"):
        a, f, v = r["all"], r["fair"], r["nature_vs_obvious"]
        mv = r["moves_rev"]["moved"] if r.get("moves_rev") else None
        print(f"| {r['model']} | {r['formulation']} | {pct(f['acc1'])} [{pct(f['acc1_ci'][0])}-{pct(f['acc1_ci'][1])}]"
              f" | {f['n']} | {pct(f['chance'])} / {pct(f['longest_baseline'])}"
              f" | {pct(a['acc1'])} [{pct(a['acc1_ci'][0])}-{pct(a['acc1_ci'][1])}] | {pct(a['longest_baseline'])}"
              f" | {pct(a['picks_longest'])} ({pct(a['picks_longest_chance'])}) | {pct(v['picks_book'])}"
              f" | {pct(mv)} | {r['ms_median']} |")


# ---------------------------------------------------------------- the books_v2 set
def _v2paths(root=None):
    root = Path(root) if root else BOOK2_ROOT
    if _is_v3(root):
        return root, root / "formulations_v3", root / "screen" / "q", root / "screen" / "d", BOOK3_BOARD
    board = BOOK2_BOARD if root.resolve() == BOOK2_ROOT.resolve() else root / "screen" / "leaderboard.jsonl"
    return root, root / "formulations_v2", root / "screen" / "q", root / "screen" / "d", board


def _digest2(sit, question, options, menu=None):
    """_digest, and a two-level menu's own (its two steps are another decision)."""
    if not menu:
        return _digest(sit, question, options)
    import hashlib
    data = json.dumps([sit, question, list(options), menu["question"], menu["scopes"], menu["groups"]],
                      ensure_ascii=False)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _req2(c, form, rid):
    opts = list(c["options"])
    r = {"id": rid, "situation": c["situation"], "question": c["question"], "options": opts,
         "digest": _digest2(c["situation"], c["question"], opts, c.get("menu2")),
         "good": c.get("good") or [], "grades": c.get("grades"), "book": c.get("book"),
         "decision_kind": c.get("decision_kind"), "book_choice": c.get("book_choice"),
         "obvious": c.get("obvious"), "decoys": c.get("decoys") or []}
    if list(c.get("names") or opts) != opts:
        r["names"] = list(c["names"])
    for k in ("menu2", "after", "pair_id", "base_id", "variant"):
        if c.get(k) is not None:
            r[k] = c[k]
    return r


def _perm2(r, order, rid):
    """Request `r` with its options in `order` (new position j holds old order[j])."""
    inv = {old: new for new, old in enumerate(order)}
    out = dict(r, id=rid, options=[r["options"][k] for k in order], perm=order,
               good=[inv[g] for g in r["good"]],
               grades=None if r.get("grades") is None else [r["grades"][k] for k in order],
               book_choice=None if r.get("book_choice") is None else inv[r["book_choice"]],
               obvious=None if r.get("obvious") is None else inv[r["obvious"]],
               decoys=[inv[d] for d in r.get("decoys") or []])
    if r.get("names"):
        out["names"] = [r["names"][k] for k in order]
    if r.get("menu2"):
        m = r["menu2"]
        out["menu2"] = dict(m, groups=[sorted(inv[i] for i in g) for g in m["groups"]])
    out["digest"] = _digest2(out["situation"], out["question"], out["options"], out.get("menu2"))
    return out


def build_books_v2(root=None):
    """One request file per formulation file ("<case>#<form>"), its reversed
    sample for BOOK2_REV ("<case>#<form>#p1"), and one per pair file
    ("pairs__<form>": "<pair_id>#<form>")."""
    root, forms, q, _, _ = _v2paths(root)
    files = sorted(forms.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no {forms}/*.jsonl: run python3 tools/export_book_cases.py --v2 first")
    for old in q.glob("*.jsonl"):
        old.unlink()
    n = {}
    for p in files:
        f = p.stem
        cs = rows(p)
        reqs = [_req2(c, f, f"{c['id']}#{f}") for c in cs]
        write(q / f"{f}.jsonl", reqs)
        n[f] = len(reqs)
        if f in BOOK2_REV and not _is_v3(root):
            rev = [_perm2(r, list(range(len(r["options"])))[::-1], r["id"] + "#p1")
                   for i, r in enumerate(reqs) if i % BOOK_REV_EVERY == 0 and len(r["options"]) > 1]
            write(q / f"{f}_rev.jsonl", rev)
            n[f"{f}_rev"] = len(rev)
    for p in sorted((forms / "pairs").glob("*.jsonl")):
        reqs = [_req2(c, p.stem, f"{c['pair_id']}#{p.stem}") for c in rows(p)]
        write(q / f"pairs__{p.stem}.jsonl", reqs)
        n[f"pairs__{p.stem}"] = len(reqs)
    (q / "source.json").write_text(json.dumps({"root": str(root), "files": n}, indent=1) + "\n")
    print(f"books_v2: {len(n)} request files in {q} ({sum(n.values())} requests)")


def _v2names(q, only=()):
    import fnmatch
    names = sorted(p.stem for p in q.glob("*.jsonl"))
    if only:
        names = [x for x in names if any(fnmatch.fnmatch(x, o) for o in only)]
    return names


def _rank_menu2(s, r):
    """The two-step decision of a row with a two-level menu."""
    m = r["menu2"]
    if len(m["groups"]) == 1:
        g, first = m["groups"][0], {"pick": 0}
    else:
        first = s.rank(r["situation"], m["question"], list(m["scopes"]))
        g = m["groups"][int(first["pick"])]
    if len(g) == 1:
        return {"pick": g[0], "scores": None, "scope_pick": int(first["pick"])}
    second = s.rank(r["situation"], r["question"], [r["options"][i] for i in g])
    return {"pick": g[int(second["pick"])], "scores": None, "scope_pick": int(first["pick"])}


def run_books_v2(model, budget=0.0, limit=0, only=(), root=None):
    """Score the books_v2 request files `only` (fnmatch patterns; all when
    empty), each case once per model: a request whose digest this model has
    already decided (in any books_v2 file) takes that decision (`reused`)."""
    from lab.methods.score_requests import Sampler, _todo, memory
    _, _, q, dd, _ = _v2paths(root)
    d = dd / model
    d.mkdir(parents=True, exist_ok=True)
    seen = {}
    for p in d.glob("*.jsonl"):
        for x in rows(p):
            if x["pick"] >= 0 and not x.get("reused"):
                seen.setdefault(x["digest"], x)
    if _is_v3(root):                        # v2's decisions of the same model, by digest
        for p in sorted((BOOK2_ROOT / "screen" / "d" / model).glob("*.jsonl")):
            for x in rows(p):
                if x["pick"] >= 0 and not x.get("reused"):
                    seen.setdefault(x["digest"], dict(x, request="v2:" + x["request"]))
    jobs = []
    for name in _v2names(q, only):
        todo, done = _todo(q / f"{name}.jsonl", d / f"{name}.jsonl")
        if limit:
            todo = todo[:limit]
        print(f"{model} {name}: {len(todo)} to do, {done} done", flush=True)
        if todo:
            jobs.append((d / f"{name}.jsonl", todo))
    if not jobs:
        return
    s, on_gpu, sampler, load_s = None, os.environ.get("LAB_DEVICE", "cuda") == "cuda", None, 0.0
    duty = float(os.environ.get("LAB_DUTY", "0.5"))
    t0, stop, n_done, n_reused = time.time(), False, 0, 0
    for path, todo in jobs:
        with path.open("a") as f:
            for r in todo:
                if budget and time.time() - t0 > budget:
                    stop = True
                    break
                old = seen.get(r["digest"])
                if old is not None:
                    row = dict(old, request=r["id"], reused=old["request"])
                    f.write(json.dumps(row) + "\n")
                    n_reused += 1
                    continue
                if s is None:                       # the model loads only when a case needs it
                    s = MODELS[model][0]()
                    if on_gpu:
                        from lab.methods.run import gpu_check
                        gpu_check(s.params)
                    sampler = Sampler()
                    sampler.start()
                    tl = time.time()
                    s.load()
                    load_s = time.time() - tl
                    sampler.phase = "score"
                    print(f"loaded in {load_s:.1f}s", flush=True)
                t = time.perf_counter()
                try:
                    got = _rank_menu2(s, r) if r.get("menu2") else s.rank(r["situation"], r["question"],
                                                                           list(r["options"]))
                except ValueError as e:         # the model refuses this menu (too long for it)
                    got = {"pick": -1, "error": str(e)[:200]}
                ms = (time.perf_counter() - t) * 1000
                row = {"request": r["id"], "digest": r["digest"], "pick": int(got["pick"]),
                       "scores": None if got.get("scores") is None else [float(x) for x in got["scores"]],
                       "chooser": model, "ms": round(ms, 3)}
                for k in ("set", "abstain", "unbound", "error", "scope_pick"):
                    if k in got:
                        row[k] = got[k]
                f.write(json.dumps(row) + "\n")
                f.flush()
                if row["pick"] >= 0:
                    seen.setdefault(r["digest"], row)
                n_done += 1
                if duty < 1:
                    time.sleep(ms / 1000 * (1 / duty - 1))
                if n_done % 200 == 0:
                    print(f"  {n_done} ({path.name})", flush=True)
        if stop:
            break
    print(("stopped at the budget" if stop else "done") + f": {n_done} decisions, {n_reused} reused", flush=True)
    if s is not None:
        mem = memory(on_gpu) | sampler.summary() | {"load_s": round(load_s, 2), "decisions": n_done}
        print("memory: " + json.dumps(mem), flush=True)
        k = len(list(d.glob("mem_*.json")))
        (d / f"mem_{k + 1}.json").write_text(json.dumps(mem) + "\n")


def _mean_ci(xs):
    """The mean and a normal 95% interval."""
    n = len(xs)
    if not n:
        return None, [None, None]
    m = sum(xs) / n
    sd = statistics.pstdev(xs) if n > 1 else 0.0
    h = 1.96 * sd / math.sqrt(n)
    return round(m, 4), [round(m - h, 4), round(m + h, 4)]


def _v2block(reqs, dec):
    """acc@1, the graded score and their controls on the requests `reqs` that
    `dec` ({request id: decision}) has decided."""
    rs = [r for r in reqs if r["id"] in dec]
    if not rs:
        return {"n": 0}
    out = {"n": len(rs)}
    picks = [dec[r["id"]]["pick"] for r in rs]
    hits = [p in r["good"] for p, r in zip(picks, rs)]
    out["acc1"], out["acc1_ci"] = round(sum(hits) / len(rs), 4), wilson(sum(hits), len(rs))
    out["chance"] = round(sum(len(r["good"]) / len(r["options"]) for r in rs) / len(rs), 4)
    gr = [r for r, p in zip(rs, picks) if r.get("grades") is not None]
    if gr:
        gp = [(r["grades"][dec[r["id"]]["pick"]] if dec[r["id"]]["pick"] >= 0 else 0.0) for r in gr]
        out["grade"], out["grade_ci"] = _mean_ci(gp)
        out["grade_chance"] = round(sum(sum(r["grades"]) / len(r["grades"]) for r in gr) / len(gr), 4)
        out["grade_ceiling"] = round(sum(max(r["grades"]) for r in gr) / len(gr), 4)
    lens = [[len(x) for x in (r.get("names") or r["options"])] for r in rs]
    longest = [[i for i, x in enumerate(n) if x == max(n)] for n in lens]
    first_long = [n.index(max(n)) for n in lens]
    out["longest_baseline"] = round(sum(fl in r["good"] for fl, r in zip(first_long, rs)) / len(rs), 4)
    if gr:
        out["longest_grade"] = round(sum(r["grades"][fl] for fl, r in zip(first_long, rs)
                                         if r.get("grades") is not None) / len(gr), 4)
    out["picks_longest"] = round(sum(p in lg for p, lg in zip(picks, longest)) / len(rs), 4)
    out["picks_longest_chance"] = round(sum(len(lg) / len(r["options"]) for lg, r in zip(longest, rs)) / len(rs), 4)
    bc = [r for r in rs if r.get("book_choice") is not None]
    if bc:
        out["book_choice"] = round(sum(dec[r["id"]]["pick"] == r["book_choice"] for r in bc) / len(bc), 4)
    ob = [r for r in rs if r.get("obvious") is not None]
    if ob:
        out["picks_obvious"] = round(sum(dec[r["id"]]["pick"] == r["obvious"] for r in ob) / len(ob), 4)
    out["picks_decoy"] = round(sum(dec[r["id"]]["pick"] in (r.get("decoys") or []) for r in rs) / len(rs), 4)
    out["refused"] = sum(p < 0 for p in picks)
    mn = [r for r in rs if r.get("menu2") and "scope_pick" in dec[r["id"]]]
    if mn:
        ok = [bool(set(r["menu2"]["groups"][dec[r["id"]]["scope_pick"]]) & set(r["good"])) for r in mn]
        out["scope_holds_good"] = round(sum(ok) / len(mn), 4)
        out["scope_holds_good_chance"] = round(sum(
            sum(bool(set(g) & set(r["good"])) for g in r["menu2"]["groups"]) / len(r["menu2"]["groups"])
            for r in mn) / len(mn), 4)
    return out


def report_books_v2(model, only=(), root=None):
    """One leaderboard row per books_v2 file `model` has decided (pairs files
    are pairs.py's), numbers only."""
    _, _, q, dd, board = _v2paths(root)
    d = dd / model
    mems = [json.loads(p.read_text()) for p in sorted(d.glob("mem_*.json"))]

    def peak(k):
        xs = [x.get(k) for x in mems if x.get(k) is not None]
        return max(xs) if xs else None

    got = []
    for name in _v2names(q, only):
        if name.endswith("_rev") or name.startswith("pairs__") or not (d / f"{name}.jsonl").exists():
            continue
        reqs = rows(q / f"{name}.jsonl")
        dec = {x["request"]: x for x in rows(d / f"{name}.jsonl")}
        own = [x for x in dec.values() if not x.get("reused")]
        row = {"set": "books_v2", "model": model, "formulation": name, "size": MODELS[model][1],
               "license": MODELS[model][2], "when": time.strftime("%Y-%m-%d"),
               "complete": f"{sum(r['id'] in dec for r in reqs)}/{len(reqs)}",
               "reused": len(dec) - len(own), **_v2block(reqs, dec),
               "ms_median": round(statistics.median(x["ms"] for x in own), 2) if own else None,
               "ram_rss_peak_mib": peak("rss_peak_mib"), "vram_alloc_peak_mib": peak("cuda_alloc_peak_mib")}
        if (q / f"{name}_rev.jsonl").exists() and (d / f"{name}_rev.jsonl").exists():
            row["moves_rev"] = _moves(rows(q / f"{name}_rev.jsonl"), list(dec.values()), rows(d / f"{name}_rev.jsonl"))
        got.append(row)
    keys = {(r["model"], r["formulation"]) for r in got}
    write(board, [r for r in rows(board) if (r["model"], r["formulation"]) not in keys] + got)
    for r in got:
        print(json.dumps(r))
    return got


def table_books_v2(root=None):
    _, _, _, _, board = _v2paths(root)

    def pct(x):
        return "-" if x is None else f"{100 * x:.1f}"

    def g(x):
        return "-" if x is None else f"{x:.2f}"
    print("| model | formulation | n | acc@1 % [95% CI] | chance % | grade of pick [95% CI] | grade chance / ceiling |"
          " longest: acc % / grade | picks longest % (chance) | book's choice % | obvious % | decoy % | moves rev % |"
          " reused | ms |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows(board):
        mv = r["moves_rev"]["moved"] if r.get("moves_rev") else None
        if not r.get("n"):
            continue
        print(f"| {r['model']} | {r['formulation']} | {r['n']} | {pct(r['acc1'])} [{pct(r['acc1_ci'][0])}-"
              f"{pct(r['acc1_ci'][1])}] | {pct(r['chance'])} | {g(r.get('grade'))} [{g(r.get('grade_ci', [None])[0])}-"
              f"{g(r.get('grade_ci', [None, None])[1])}] | {g(r.get('grade_chance'))} / {g(r.get('grade_ceiling'))} |"
              f" {pct(r['longest_baseline'])} / {g(r.get('longest_grade'))} | {pct(r['picks_longest'])}"
              f" ({pct(r['picks_longest_chance'])}) | {pct(r.get('book_choice'))} | {pct(r.get('picks_obvious'))} |"
              f" {pct(r['picks_decoy'])} | {pct(mv)} | {r['reused']} | {r['ms_median']} |")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m lab.methods.screen")
    ap.add_argument("what", choices=("build", "run", "report", "table"))
    ap.add_argument("model", nargs="?")
    ap.add_argument("--budget", type=float, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--set", default="default", choices=sorted(SETS))
    ap.add_argument("--only", nargs="*", default=[], help="books: these formulations (and their _rev files);"
                    " books_v2: fnmatch patterns of its request files")
    ap.add_argument("--root", default=None, help="books_v2: another tree with the v2 layout (the dry run's)")
    a = ap.parse_args(argv)
    if a.what in ("run", "report") and a.model not in MODELS:
        raise SystemExit(f"a model of {sorted(MODELS)}")
    if a.set in ("books_v2", "books_v3"):
        if a.set == "books_v3":
            a.root = str(BOOK3_ROOT)
        if a.what == "build":
            build_books_v2(a.root)
        elif a.what == "run":
            run_books_v2(a.model, a.budget, a.limit, tuple(a.only), a.root)
        elif a.what == "report":
            report_books_v2(a.model, tuple(a.only), a.root)
        else:
            table_books_v2(a.root)
        return
    books = a.set == "books"
    only = tuple(a.only) + tuple(f"{f}_rev" for f in a.only)
    if a.what == "build":
        build_books() if books else build()
    elif a.what == "run":
        run(a.model, a.budget, a.limit, set_=a.set, only=only)
    elif a.what == "report":
        report_books(a.model, tuple(a.only) or BOOK_FORMS) if books else report(a.model)
    else:
        table_books() if books else table()


if __name__ == "__main__":
    sys.exit(main())
