"""Favourite items: correct them without flattening character (plan v4, section 3).

    python3 -m lab.methods.prior build       # cases_die.jsonl (the fair-die probe)
    python3 -m lab.methods.prior report      # lab/results/methods/prior.md
    python3 -m lab.methods.prior selftest    # the corrections on synthetic data
    python3 -m lab.methods.prior table       # prior_offsets.json, for the chooser (LAB_PRIOR)

Each model has favourite items it picks 3-7 times more than chance, whatever
the need. Subtracting the score with the need removed did not help
GLiNER-Decide. A correction must remove the favourites and KEEP real
preferences: a fair die gives 1-6 evenly, but an angry man who is scolded
should still lean towards the stone.

THE FAIR-DIE PROBE. "Tom rolls a six-sided die. Which face comes up?" (and a
twenty-sided die, and "Tom picks one thing at random"). Every answer is
right, so how uneven the model is shows its pure prior. Each die is asked in
many orders, so a preference for a PLACE in the list and a preference for a
LABEL can be told apart.

THE CORRECTIONS, fitted on half of the needs and tested on the other half
(from scores kept with --keep-scores, and vectors kept with --capture):
    mean      subtract each item's average log-probability over other
              contexts, times alpha
    even      fit one offset per item so that, on the fitting half, each item
              is picked about as often as it appears (the most even spread);
              iterative scaling
    vector    subtract each item's average OPTION VECTOR (the part that says
              which item it is, whatever the need) before GLiNER-Decide's own
              classifier, times beta

MEASURES on the test half: the share right; the favourite lift (favourites
found on the fitting half); evenness (the effective number of items picked,
over the number the presence of items allows; 1 = as even as possible).
THE CHARACTER CHECK: the same corrections applied to the personality runs of
stages.py; the confront shift between a hot-tempered and a timid Tom must not
shrink.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RES = HERE.parent / "results" / "methods"

WORDS6 = ["one", "two", "three", "four", "five", "six"]


# ------------------------------------------------------------------ the die probe
def build():
    rng = random.Random("die")
    out = []

    def add(kind, sit, q, opts, k):
        out.append({"id": f"die/{kind}/{k}", "kind": "die", "factor": "die", "value": kind,
                    "need": kind, "situation": sit, "question": q, "options": opts, "names": opts,
                    "good": list(range(len(opts))), "keep_scores": True})
    for k in range(20):
        faces = [str(i) for i in range(1, 7)]
        rng.shuffle(faces)
        add("d6", "Tom rolls a fair six-sided die.", "Which face comes up?", faces, k)
        words = WORDS6[:]
        rng.shuffle(words)
        add("d6words", "Tom rolls a fair six-sided die.", "Which face comes up?", words, k)
    for k in range(10):
        faces = [str(i) for i in range(1, 21)]
        rng.shuffle(faces)
        add("d20", "Tom rolls a fair twenty-sided die.", "Which face comes up?", faces, k)
    things = [json.loads(l) for l in open(HERE / "cases_things.jsonl")]
    things = [c for c in things if c["factor"] == "size" and c["value"] == 16]
    for k, c in enumerate(things):
        add("random_thing", "Tom picks one thing from the storeroom at random, without looking.",
            "Which thing does he pick?", c["names"], k)
    with open(HERE / "cases_die.jsonl", "w") as f:
        for c in out:
            f.write(json.dumps(c) + "\n")
    print(f"{len(out)} die cases -> {HERE / 'cases_die.jsonl'}")


def _p(scores):
    s = np.asarray(scores, dtype=float)
    if s.min() >= 0 and s.max() <= 1 and abs(s.sum() - 1) < 1e-3:
        return s
    e = np.exp(s - s.max())
    return e / e.sum()


def die_section():
    cases = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "cases_die.jsonl")} \
        if (HERE / "cases_die.jsonl").exists() else {}
    files = sorted(glob.glob(str(RES / "*.die.jsonl")))
    if not files:
        return []
    L = ["## The fair-die probe", "",
         "Every answer is right; the model should spread its probability evenly. `even` = the effective "
         "number of faces over the number of faces (1 = perfectly even). `by label` / `by place` = the most "
         "over the least probable face label / list place, averaged over the shuffled orders "
         "(1.0 = no preference).", "",
         "| model | die | even | by label | by place | favourite label |", "| --- | --- | --- | --- | --- | --- |"]
    for f in files:
        name = os.path.basename(f)[: -len(".die.jsonl")].replace("__", ":")
        rows = [json.loads(l) for l in open(f)]
        by = defaultdict(list)
        for r in rows:
            if "scores" in r and r["id"] in cases:
                by[cases[r["id"]]["value"]].append(r)
        for kind, rs in sorted(by.items()):
            lab, pos, ev = defaultdict(list), defaultdict(list), []
            for r in rs:
                c = cases[r["id"]]
                p = _p(r["scores"])
                ev.append(math.exp(-(p * np.log(np.clip(p, 1e-12, 1))).sum()) / len(p))
                for i, o in enumerate(c["options"]):
                    lab[o].append(p[i])
                    pos[i].append(p[i])
            lm = {k: np.mean(v) for k, v in lab.items()}
            pm = {k: np.mean(v) for k, v in pos.items()}
            fav = max(lm, key=lm.get)
            L.append(f"| {name} | {kind} | {np.mean(ev):.2f} | {max(lm.values()) / max(1e-9, min(lm.values())):.1f} | "
                     f"{max(pm.values()) / max(1e-9, min(pm.values())):.1f} | {fav} ({lm[fav]:.0%}) |")
    return L + [""]


# ------------------------------------------------------------------ data
def _half(need):
    return int(hashlib.md5(need.encode()).hexdigest(), 16) % 2


def load(scorer, tag, cases_file):
    """Rows with every option's score, joined with their cases."""
    cases = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / cases_file)}
    f = RES / f"{scorer.replace(':', '__')}{'.' + tag if tag else ''}.jsonl"
    if not f.exists():
        return []
    out = []
    for l in open(f):
        r = json.loads(l)
        c = cases.get(r["id"])
        if c and "scores" in r:
            p = _p(r["scores"])
            out.append({"id": r["id"], "need": c["need"], "names": c["names"], "good": set(c["good"]),
                        "logp": np.log(np.clip(p, 1e-12, 1)), "value": c["value"], "factor": c["factor"]})
    return out


def load_vectors(scorer, tag):
    d = RES / f"{scorer.replace(':', '__')}{'.' + tag if tag else ''}.cap"
    vec = {}
    for part in sorted(d.glob("part*.npz")) if d.exists() else []:
        z = np.load(part)
        off = 0
        for cid, n in zip(z["ids"], z["n"]):
            vec[str(cid)] = (z["opts"][off:off + n].astype(np.float32), z["logits"][off:off + n])
            off += n
    return vec


# ------------------------------------------------------------------ corrections
def fit_mean(rows):
    """Each item's average centred log-probability across contexts."""
    acc = defaultdict(list)
    for r in rows:
        z = r["logp"] - r["logp"].mean()
        for n, v in zip(r["names"], z):
            acc[n].append(v)
    return {n: float(np.mean(v)) for n, v in acc.items()}


def fit_even(rows, iters=300, lr=0.5):
    """One offset per item so that, over these rows, each item is picked about
    as often as it appears (as an even share of each list)."""
    items = sorted({n for r in rows for n in r["names"]})
    ix = {n: i for i, n in enumerate(items)}
    b = np.zeros(len(items))
    target = np.zeros(len(items))
    for r in rows:
        for n in r["names"]:
            target[ix[n]] += 1 / len(r["names"])
    idx = [np.array([ix[n] for n in r["names"]]) for r in rows]
    for _ in range(iters):
        got = np.zeros(len(items))
        for r, ii in zip(rows, idx):
            z = r["logp"] - b[ii]
            p = np.exp(z - z.max())
            got[ii] += p / p.sum()
        b += lr * (np.log(got + 1e-9) - np.log(target + 1e-9))
        b -= b.mean()
    return {n: float(b[ix[n]]) for n in items}


def apply_offsets(rows, off, alpha=1.0):
    return [int(np.argmax(r["logp"] - alpha * np.array([off.get(n, 0.0) for n in r["names"]]))) for r in rows]


def correct(p, names, offsets, alpha=1.0):
    """The chooser's correction (to-do A4): subtract alpha times each option's
    offset from its log-probability and renormalise. An option with no offset
    (a thing never seen when fitting) is left as it is."""
    lp = np.log(np.clip(np.asarray(p, dtype=float), 1e-12, 1)) - alpha * np.array(
        [offsets.get(n, 0.0) for n in names])
    q = np.exp(lp - lp.max())
    return q / q.sum()


TABLE = RES / "prior_offsets.json"
TABLE_SOURCES = [("gdecide:all", "cap_things", "cases_things.jsonl"),
                 ("gdecide:all", "cap_people", "cases_people.jsonl"),
                 ("gdecide:all", "cap_places", "cases_places.jsonl")]


def write_table(alpha=1.0):
    """Fit the item-mean offsets on every saved list and write the table the
    chooser reads (LAB_PRIOR=<this file>). Before that, check that the path
    through correct() gives the same numbers as the 'mean' rows of prior.md."""
    offsets, n_lists = {}, 0
    for scorer, tag, cf in TABLE_SOURCES:
        rows = load(scorer, tag, cf)
        if not rows:
            continue
        for fit_half in (0, 1):
            fit = [r for r in rows if _half(r["need"]) == fit_half]
            test = [r for r in rows if _half(r["need"]) != fit_half]
            m = fit_mean(fit)
            a = apply_offsets(test, m, alpha)
            b = [int(np.argmax(correct(np.exp(r["logp"]), r["names"], m, alpha))) for r in test]
            assert a == b, f"correct() does not match apply_offsets on {tag}"
        for n, v in fit_mean(rows).items():
            if n in offsets and abs(offsets[n] - v) > 1e-9:
                print(f"  {n!r} is in two spaces; keeping the first offset")
                continue
            offsets[n] = v
        n_lists += len(rows)
    TABLE.write_text(json.dumps({"alpha": alpha, "method": "item mean", "lists": n_lists,
                                 "offsets": dict(sorted(offsets.items()))}, indent=0))
    print(f"{len(offsets)} offsets from {n_lists} lists -> {TABLE} (correct() matches prior.md's path)")


def fit_vector_means(rows, vec):
    acc = defaultdict(list)
    allv = []
    for r in rows:
        if r["id"] not in vec:
            continue
        V = vec[r["id"]][0]
        for n, v in zip(r["names"], V):
            acc[n].append(v)
            allv.append(v)
    mu = np.mean(allv, axis=0) if allv else None
    return {n: np.mean(v, axis=0) - mu for n, v in acc.items()}


_CLF = None


def classifier():
    """GLiNER-Decide's own final layer, on the CPU."""
    global _CLF
    if _CLF is None:
        import torch
        from gliner2 import AutoExtractor
        m = AutoExtractor.from_pretrained("fastino/GLiNER2.5-Decide").eval()
        clf = m.classifier.float().cpu()

        def f(V):
            with torch.no_grad():
                return clf(torch.tensor(V, dtype=torch.float32)).reshape(-1).numpy()
        _CLF = f
    return _CLF


def apply_vector(rows, vec, dev, beta=1.0):
    clf = classifier()
    picks = []
    for r in rows:
        if r["id"] not in vec:
            picks.append(int(np.argmax(r["logp"])))
            continue
        V = vec[r["id"]][0] - beta * np.stack([dev.get(n, 0 * vec[r["id"]][0][0]) for n in r["names"]])
        picks.append(int(np.argmax(clf(V))))
    return picks


# ------------------------------------------------------------------ measures
def favourites(rows, picks, k=10):
    present, picked = Counter(), Counter()
    for r, p in zip(rows, picks):
        if p in r["good"]:
            continue
        for i, n in enumerate(r["names"]):
            if i not in r["good"]:
                present[n] += 1
        picked[r["names"][p]] += 1
    score = {n: picked[n] / present[n] for n in picked if present[n] >= 5}
    return set(sorted(score, key=lambda n: -score[n])[:k])


def lift(rows, picks, favs):
    hit = base = n = 0
    for r, p in zip(rows, picks):
        if p in r["good"]:
            continue
        bad = [i for i in range(len(r["names"])) if i not in r["good"]]
        n += 1
        hit += r["names"][p] in favs
        base += sum(r["names"][i] in favs for i in bad) / len(bad)
    return (hit / n) / (base / n) if n and base else float("nan")


def evenness(rows, picks):
    pc, pr = Counter(), Counter()
    for r, p in zip(rows, picks):
        pc[r["names"][p]] += 1
        for n in r["names"]:
            pr[n] += 1 / len(r["names"])

    def eff(c):
        t = sum(c.values())
        return math.exp(-sum(v / t * math.log(v / t) for v in c.values() if v))
    return eff(pc) / eff(pr)


def evaluate(rows, vec=None, alphas=(0.5, 1.0), betas=(0.5, 1.0)):
    """Fit on one half of the needs, test on the other, both ways; average."""
    out = defaultdict(list)
    for fit_half in (0, 1):
        fit = [r for r in rows if _half(r["need"]) == fit_half]
        test = [r for r in rows if _half(r["need"]) != fit_half]
        if not fit or not test:
            continue
        base = [int(np.argmax(r["logp"])) for r in fit]
        favs = favourites(fit, base)
        methods = {"none": [int(np.argmax(r["logp"])) for r in test]}
        m = fit_mean(fit)
        for a in alphas:
            methods[f"mean a={a}"] = apply_offsets(test, m, a)
        e = fit_even(fit)
        methods["even"] = apply_offsets(test, e, 1.0)
        if vec:
            dev = fit_vector_means(fit, vec)
            for bt in betas:
                methods[f"vector b={bt}"] = apply_vector(test, vec, dev, bt)
        for name, picks in methods.items():
            out[name].append((np.mean([p in r["good"] for r, p in zip(test, picks)]),
                              lift(test, picks, favs), evenness(test, picks), len(test)))
    return {k: tuple(np.nanmean([x[i] for x in v]) for i in range(3)) + (sum(x[3] for x in v),)
            for k, v in out.items()}


def character_check(scorer="gdecide:all"):
    """The corrections fitted on the people runs, applied to the personality
    runs: the confront shift between hot and timid must survive."""
    f = RES / "stages" / f"personality__{scorer.replace(':', '__')}.jsonl"
    rows = load(scorer, "cap_people", "cases_people.jsonl")
    if not f.exists() or not rows:
        return []
    cases = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "cases_personality.jsonl")}
    off = {"mean": fit_mean(rows), "even": fit_even(rows)}
    L = ["## The character check", "",
         "Confront share of the draw, hot-tempered minus timid, before and after each correction "
         "(fitted on the people runs). A correction that shrinks the shift flattens character.", "",
         "| arm | none | mean | even |", "| --- | --- | --- | --- |"]
    recs = [json.loads(l) for l in open(f)]
    for arm in ("text", "appraisal", "oracle"):
        cells = []
        for method in ("none", "mean", "even"):
            shift = []
            for r in recs:
                c = cases.get(r["id"])
                if not c or "traits" not in r:
                    continue
                vals = {}
                for t in ("hot", "timid"):
                    lp = np.log(np.clip(np.array(r["traits"][t][arm]["p"]), 1e-12, 1))
                    if method != "none":
                        lp = lp - np.array([off[method].get(n, 0.0) for n in c["names"]])
                    p = np.exp(lp - lp.max())
                    p /= p.sum()
                    vals[t] = p[c["confront"]].sum()
                shift.append(vals["hot"] - vals["timid"])
            cells.append(f"{np.mean(shift):+.2f}" if shift else "")
        L.append(f"| {arm} | " + " | ".join(cells) + " |")
    return L + [""]


def report():
    L = ["# Favourite items: corrections that keep character", "",
         "Fitted on one half of the needs, tested on the other, both ways. `right` = picks in the acceptable "
         "set; `fav lift` = how much more often than chance a wrong pick is one of the favourites found on the "
         "fitting half (1 = no favourites); `even` = the effective number of items picked over what their "
         "presence allows (1 = as even as possible). Built by `lab/methods/prior.py`.", ""]
    L += die_section()
    for scorer, tag, cf in [("gdecide:all", "cap_things", "cases_things.jsonl"),
                            ("gdecide:all", "cap_people", "cases_people.jsonl"),
                            ("gdecide:all", "cap_places", "cases_places.jsonl")]:
        rows = load(scorer, tag, cf)
        if not rows:
            continue
        vec = load_vectors(scorer, tag)
        res = evaluate(rows, vec or None)
        L += [f"## {scorer}, {cf.split('_')[1].split('.')[0]} ({len(rows)} lists"
              f"{', with vectors' if vec else ''})", "",
              "| correction | right | fav lift | even |", "| --- | --- | --- | --- |"]
        L += [f"| {k} | {v[0]:.0%} | {v[1]:.2f} | {v[2]:.2f} |" for k, v in res.items()] + [""]
    L += character_check()
    (RES / "prior.md").write_text("\n".join(L) + "\n")
    print(RES / "prior.md")


# ------------------------------------------------------------------ self test
def selftest(seed=0):
    """Synthetic lists: every item has a planted prior; the right item gets a
    context bonus. The corrections must remove the prior, keep the context,
    and a 'trait' bonus on some items must survive them."""
    rng = np.random.default_rng(seed)
    items = [f"item{i}" for i in range(60)]
    prior = {n: rng.normal(0, 1.2) for n in items}
    rows = []
    for k in range(600):
        need = f"need{k % 30}"
        names = list(rng.choice(items, 16, replace=False))
        good = {int(rng.integers(16))}
        z = np.array([prior[n] + (2.5 if i in good else 0) + rng.normal(0, 0.8) for i, n in enumerate(names)])
        p = np.exp(z - z.max())
        rows.append({"id": str(k), "need": need, "names": names, "good": good,
                     "logp": np.log(p / p.sum()), "value": 16, "factor": "size"})
    res = evaluate(rows)
    for k, v in res.items():
        print(f"  {k:12s} right {v[0]:.0%}  fav lift {v[1]:.2f}  even {v[2]:.2f}")
    assert res["even"][0] > res["none"][0] + 0.05, "the even correction should recover accuracy"
    assert res["even"][1] < res["none"][1], "the even correction should cut the favourite lift"
    # a trait bonus on "confront" items, only in some contexts, must survive
    conf = set(items[:10])
    hot = []
    for r in rows[:200]:
        z = r["logp"] + np.array([1.5 if n in conf else 0 for n in r["names"]])
        hot.append(dict(r, logp=z - np.log(np.exp(z).sum())))
    off = fit_even(rows)
    share = lambda rs, o: np.mean([sum(np.exp(r["logp"] - np.array([o.get(n, 0) for n in r["names"]]))
                                         [[i for i, n in enumerate(r["names"]) if n in conf]]) /
                                   np.exp(r["logp"] - np.array([o.get(n, 0) for n in r["names"]])).sum()
                                   for r in rs])
    before = share(hot, {}) - share(rows[:200], {})
    after = share(hot, off) - share(rows[:200], off)
    print(f"  trait shift before {before:+.2f}, after the even correction {after:+.2f}")
    assert after > 0.5 * before, "the correction must not flatten a context-driven shift"
    print("selftest passed")


if __name__ == "__main__":
    {"build": build, "report": report, "selftest": selftest, "table": write_table}[sys.argv[1]]()
