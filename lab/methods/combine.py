"""Combine answers we already saved, on the CPU (to-do A2, A3, A5).

    python3 -m lab.methods.combine report     # lab/results/methods/combine.md
    python3 -m lab.methods.combine selftest   # the combining code on made-up numbers

Plan v4 asked GLiNER-Decide small questions and saved every option's
probability in every arm (lab/methods/stages.py), and the vectors behind each
answer (--capture). These tests combine what was saved, so they need no GPU:

A2  THE PERSONALITY GUARD. The appraisal arm ("Tom feels angry. He wants to
    stand up to them.") moves the trait most (+0.28), but valid picks fall
    from 76% to about 50%. Combine it with the no-persona arm (the sense):
        score = log p(none) + g * (log p(appraisal) - log p(none))
    only for the options whose no-persona probability is at least d times the
    best one's (the guard). `soft`: the other options keep their no-persona
    score. `hard`: the other options are removed. g and d are chosen on half
    the scenes and tested on the other half, both ways.

A3  "CALLS FOR" AS A WEIGHT. Each people option has a purpose, and each
    purpose maps to one "calls for" label. Bonus = l * log p(the option's
    call), with p(call) from GLiNER's own answer, from a head trained on the
    answer vectors of other scenes, or from the right answer (oracle).

A5  THE RELEVANCE THRESHOLD. p(yes) for every sentence, from the saved
    vectors. Is there a threshold that keeps every sentence that matters and
    still drops the filler?
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RES = HERE.parent / "results" / "methods"
ST = RES / "stages"


def _half(key):
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 2


def _softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()


def _log(p):
    return np.log(np.clip(np.asarray(p, dtype=float), 1e-9, 1))


def _captured(exp, scorer="gdecide:all"):
    """{case id: [(question, labels, logits, vectors)]} in the order asked."""
    d = ST / f"{exp}__{scorer.replace(':', '__')}.cap"
    out = defaultdict(list)
    for part in sorted(d.glob("part*.npz"), key=lambda p: int(p.stem[4:])) if d.exists() else []:
        z = np.load(part)
        off = 0
        for cid, q, labs, n in zip(z["ids"], z["question"], z["labels"], z["n"]):
            out[str(cid)].append((str(q), str(labs).split("␟"), z["logits"][off:off + n].astype(float),
                                  z["opts"][off:off + n].astype(np.float32)))
            off += n
    return out


def _wilson(k, n):
    from lab.methods.report import wilson
    _, lo, hi = wilson(k, n)
    return f"{k / n:.0%} ({lo:.0%}-{hi:.0%})" if n else "-"


# ------------------------------------------------------------------ A2 the personality guard
def guard_scores(p0, pc, g, d, mode):
    """The combined log scores: personality moves only the options whose
    no-persona probability is at least d times the best one's."""
    lp0, lpc = _log(p0), _log(pc)
    keep = np.asarray(p0) >= d * max(p0)
    z = np.where(keep, lp0 + g * (lpc - lp0), lp0)
    if mode == "hard":
        z = np.where(keep, z, -1e9)
    return z


def guard_measure(recs, cases, carrier, g, d, mode):
    """Per trait: the confront share of the draw, pick valid, draw valid."""
    out = {}
    for trait in ("hot", "timid"):
        conf, pick_ok, draw_ok = [], [], []
        for r in recs:
            c = cases[r["id"]]
            a = r["traits"][trait]
            z = guard_scores(a["none"]["p"], a[carrier]["p"], g, d, mode)
            q = _softmax(z)
            good = c["good"]
            conf.append(q[c["confront"]].sum())
            pick_ok.append(int(np.argmax(z)) in set(good))
            draw_ok.append(q[good].sum())
        out[trait] = (float(np.mean(conf)), pick_ok, float(np.mean(draw_ok)))
    shift = out["hot"][0] - out["timid"][0]
    ok = min(np.mean(out["hot"][1]), np.mean(out["timid"][1]))
    return {"shift": shift, "ok_min": float(ok), "hot": out["hot"], "timid": out["timid"]}


GRID_G = (0.5, 1.0, 2.0, 3.0)
GRID_D = (0.0, 0.1, 0.3, 0.5)


def guard_choose(recs, cases, carrier, need_ok=0.70):
    """The biggest shift with valid picks at or above need_ok for both traits;
    if none reaches it, the best valid-pick rate."""
    best = None
    for mode in ("soft", "hard"):
        for g in GRID_G:
            for d in GRID_D:
                m = guard_measure(recs, cases, carrier, g, d, mode)
                key = (m["ok_min"] >= need_ok, m["shift"] if m["ok_min"] >= need_ok else m["ok_min"])
                if best is None or key > best[0]:
                    best = (key, (g, d, mode))
    return best[1]


def guard_section(scorer="gdecide:all"):
    f = ST / f"personality__{scorer.replace(':', '__')}.jsonl"
    if not f.exists():
        return ["## A2. The personality guard", "", "No personality results yet.", ""]
    cases = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "cases_personality.jsonl")}
    recs = [json.loads(l) for l in open(f)]
    recs = [r for r in recs if r["id"] in cases and "traits" in r]
    L = ["## A2. The personality guard", "",
         "`shift` = the confronting share of the draw, hot-tempered minus timid. `pick valid` = the top option is "
         "acceptable. `draw valid` = the probability on acceptable options. g = how far the trait moves the "
         "scores; d = the guard (only options with at least d times the best no-persona probability can move; "
         "0 = no guard); `soft` = the others keep their no-persona score, `hard` = the others are removed.", "",
         "### Fixed settings, all 45 lists", "",
         "| carrier | g | d | mode | shift | pick valid, hot | pick valid, timid | draw valid, hot | draw valid, timid |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    rows = [("none", 0.0, 0.0, "soft")]
    for carrier in ("text", "appraisal"):
        rows += [(carrier, 1.0, 0.0, "soft"), (carrier, 1.0, 0.3, "soft"), (carrier, 2.0, 0.3, "soft"),
                 (carrier, 1.0, 0.3, "hard"), (carrier, 2.0, 0.3, "hard"), (carrier, 3.0, 0.3, "hard"),
                 (carrier, 2.0, 0.5, "hard")]
    for carrier, g, d, mode in rows:
        m = guard_measure(recs, cases, "appraisal" if carrier == "none" else carrier, g, d, mode)
        n = len(m["hot"][1])
        L.append(f"| {carrier} | {g} | {d} | {mode} | {m['shift']:+.2f} | {_wilson(sum(m['hot'][1]), n)} | "
                 f"{_wilson(sum(m['timid'][1]), n)} | {m['hot'][2]:.0%} | {m['timid'][2]:.0%} |")
    L += ["", "### Chosen on half the scenes, tested on the other half", "",
          "The setting with the biggest shift whose valid picks stay at 70% or more for both traits, chosen on "
          "one half of the scenes; the numbers are on the other half.", "",
          "| carrier | test half | chosen g, d, mode | shift | pick valid, hot | pick valid, timid | "
          "no persona, pick valid |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for carrier in ("text", "appraisal"):
        for h in (0, 1):
            fit = [r for r in recs if _half(cases[r["id"]]["need"]) != h]
            test = [r for r in recs if _half(cases[r["id"]]["need"]) == h]
            if not fit or not test:
                continue
            g, d, mode = guard_choose(fit, cases, carrier)
            m = guard_measure(test, cases, carrier, g, d, mode)
            base = guard_measure(test, cases, carrier, 0.0, 0.0, "soft")
            n = len(m["hot"][1])
            L.append(f"| {carrier} | {h} ({n} lists) | {g}, {d}, {mode} | {m['shift']:+.2f} | "
                     f"{_wilson(sum(m['hot'][1]), n)} | {_wilson(sum(m['timid'][1]), n)} | "
                     f"{_wilson(sum(base['hot'][1]), n)} |")
    return L + [""]


# ------------------------------------------------------------------ A3 "calls for" as a weight
def _call_of_options(case):
    from lab.methods.stages import PURPOSE_CALL, _people
    P = _people()[case["need"]]
    return [PURPOSE_CALL.get(P["opt"][n]["purpose"]) if n in P["opt"] else None for n in case["names"]]


def call_probs(cases, recs, scorer="gdecide:all", folds=5):
    """p(call) for each case, three ways: GLiNER's own answer, a head trained
    on the answer vectors of OTHER scenes, and the right answer."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from lab.methods.stages import CALLS
    cap = _captured("people", scorer)
    own, V, need, ids = {}, [], [], []
    for cid, qs in cap.items():
        for q, labels, lg, vec in qs:
            if q.startswith("What does this situation call for") and labels == CALLS and cid in recs:
                own[cid] = _softmax(lg)
                V.append(vec)
                need.append(cases[cid]["need"] if cid in cases else "")
                ids.append(cid)
    head = {}
    if ids:
        right = [np.array([lab in set(recs[c]["stage"]["right_calls"]) for lab in CALLS]) for c in ids]
        need = np.array(need)
        for tr, te in GroupKFold(folds).split(ids, groups=need):
            X = np.concatenate([V[k] for k in tr])
            y = np.concatenate([right[k] for k in tr]).astype(int)
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
            clf.fit(X, y)
            for k in te:
                head[ids[k]] = _softmax(clf.decision_function(V[k]))
    oracle = {}
    for cid, r in recs.items():
        rc = set(r["stage"]["right_calls"])
        if rc:
            oracle[cid] = np.array([1.0 / len(rc) if c in rc else 1e-3 for c in CALLS])
    return {"GLiNER's call": own, "head's call": head, "right call": oracle}


def call_weighted(p_none, calls, pc, lam):
    """Add lam * log p(the option's call); an option with no call gets the
    least likely call's value."""
    floor = float(np.min(pc))
    bonus = np.array([math.log(max(pc[k] if k is not None else floor, 1e-9)) for k in calls])
    return _log(p_none) + lam * bonus


def calls_section(scorer="gdecide:all"):
    f = ST / f"people__{scorer.replace(':', '__')}.jsonl"
    L = ["## A3. \"Calls for\" as a weight on people's options", ""]
    if not f.exists():
        return L + ["No people results yet.", ""]
    cases = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "cases_people.jsonl")}
    recs = {r["id"]: r for r in (json.loads(l) for l in open(f)) if "arms" in r and r["id"] in cases}
    probs = call_probs(cases, recs, scorer)
    calls = {cid: _call_of_options(cases[cid]) for cid in recs}
    right_top = {src: np.mean([CALLS_RIGHT(recs[c], p[c]) for c in p if c in recs]) for src, p in probs.items()
                 if p}
    L += ["Each option's purpose maps to one \"calls for\" label; the bonus is l * log p(that label). "
          "`sentence` = the model's call written into the scene (the plan v4 arm). How often each source's top "
          "call is right: " + ", ".join(f"{k} {v:.0%}" for k, v in right_top.items()) + ". l is chosen on half "
          "the scenes and tested on the other half (both ways); the table shows the test halves together.", ""]
    cells = sorted({(cases[c]["factor"], str(cases[c]["value"])) for c in recs})
    head = ["none", "sentence"] + [f"{src}" for src in probs]
    L += ["| cell | " + " | ".join(head) + " |", "| --- | " + " | ".join("---" for _ in head) + " |"]
    chosen = {}
    for src, p in probs.items():
        for h in (0, 1):
            fit = [c for c in recs if c in p and _half(cases[c]["need"]) != h]
            best = max((0.25, 0.5, 1.0, 2.0), key=lambda lam: np.mean(
                [int(np.argmax(call_weighted(recs[c]["arms"]["none"]["p"], calls[c], p[c], lam)))
                 in set(cases[c]["good"]) for c in fit]) if fit else 0)
            chosen[(src, h)] = best
    for cell in cells:
        ids = [c for c in recs if (cases[c]["factor"], str(cases[c]["value"])) == cell]
        row = [_wilson(sum(recs[c]["arms"]["none"]["ok"] for c in ids), len(ids)),
               _wilson(sum(recs[c]["arms"]["model_call_only"]["ok"] for c in ids), len(ids))]
        for src, p in probs.items():
            got = [int(np.argmax(call_weighted(recs[c]["arms"]["none"]["p"], calls[c], p[c],
                                               chosen[(src, _half(cases[c]["need"]))])))
                   in set(cases[c]["good"]) for c in ids if c in p]
            row.append(_wilson(sum(got), len(got)) if got else "-")
        L.append(f"| {cell[0]}={cell[1]} | " + " | ".join(row) + " |")
    L += ["", "l chosen: " + ", ".join(f"{src} half {h}: {v}" for (src, h), v in chosen.items()), ""]
    return L


def CALLS_RIGHT(rec, p):
    from lab.methods.stages import CALLS
    return CALLS[int(np.argmax(p))] in set(rec["stage"]["right_calls"])


# ------------------------------------------------------------------ A5 the relevance threshold
def relevance_rows(scorer="gdecide:all"):
    """One row per judged sentence: p(yes), whether it matters, whether it is
    the need sentence, and its case."""
    from lab.methods.cases import FILLER
    from lab.methods.stages import _cases, _sentences
    cap = _captured("relevance", scorer)
    frame = {"He looks at what is in the storeroom.", "Tom is there."}
    filler = set(FILLER)
    rows = []
    for c in _cases("relevance"):
        qs = [x for x in cap.get(c["id"], []) if not x[0].startswith("FINAL")]
        sents = [s for s in _sentences(c["situation"]) if s not in frame]
        if len(qs) != len(sents):
            continue
        first = _sentences(c["need"])[0] if c["kind"] == "people" else None
        for s, (q, labels, lg, _) in zip(sents, qs):
            rows.append({"case": c["id"], "kind": c["kind"], "p_yes": float(_softmax(lg)[0]),
                         "matters": s not in filler,
                         "need": s == first if first else s.startswith("Kael needs"), "sent": s})
    return rows


def relevance_section(scorer="gdecide:all"):
    from sklearn.metrics import roc_auc_score
    rows = relevance_rows(scorer)
    L = ["## A5. The relevance threshold", ""]
    if not rows:
        return L + ["No relevance vectors yet.", ""]
    y = np.array([r["matters"] for r in rows])
    p = np.array([r["p_yes"] for r in rows])
    by = defaultdict(list)
    for r in rows:
        by[r["case"]].append(r)
    L += [f"{len(rows)} sentences in {len(by)} long scenes. AUC of p(yes) for the sentences that matter against "
          f"the filler: {roc_auc_score(y, p):.2f}. `all kept` = the share of scenes that keep every sentence that "
          "matters; `need kept` = the share that keep the need sentence; `filler dropped` = the share of filler "
          "sentences dropped. 0.50 is what the model's own answer used (the top label).", "",
          "| keep when p(yes) is at least | all kept | need kept | filler dropped |", "| --- | --- | --- | --- |"]
    for t in (0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02):
        allk = np.mean([all(r["p_yes"] >= t for r in rs if r["matters"]) for rs in by.values()])
        needk = np.mean([all(r["p_yes"] >= t for r in rs if r["need"]) for rs in by.values()])
        fd = np.mean([r["p_yes"] < t for r in rows if not r["matters"]])
        L.append(f"| {t} | {allk:.0%} | {needk:.0%} | {fd:.0%} |")
    rule = np.mean([all(r["p_yes"] >= 0.5 or r["need"] for r in rs if r["matters"]) for rs in by.values()])
    L += ["", f"With the need sentence always kept and the rest by the model's answer: every sentence that matters "
          f"is kept in {rule:.0%} of scenes.", ""]
    low = sorted((r for r in rows if r["matters"]), key=lambda r: r["p_yes"])[:8]
    L += ["The sentences that matter with the lowest p(yes):", ""]
    L += [f"- {r['p_yes']:.3f} ({r['kind']}): \"{r['sent']}\"" for r in low] + [""]
    return L


# ------------------------------------------------------------------ report
def report():
    L = ["# Combining saved answers (CPU)", "",
         "Built by `lab/methods/combine.py` from the plan v4 results (`lab/results/methods/stages/`). No new model "
         "runs.", ""]
    for sec in (guard_section, calls_section, relevance_section):
        L += sec()
    (RES / "combine.md").write_text("\n".join(L) + "\n")
    print(RES / "combine.md")


def selftest():
    # the guard: g = 0 gives the no-persona scores; a big d leaves only the best option free
    p0, pc = [0.5, 0.3, 0.2], [0.1, 0.1, 0.8]
    assert np.allclose(guard_scores(p0, pc, 0.0, 0.0, "soft"), _log(p0))
    z = guard_scores(p0, pc, 1.0, 0.0, "soft")
    assert np.allclose(z, _log(pc))
    z = guard_scores(p0, pc, 1.0, 0.5, "soft")              # only options with p0 >= 0.25 move
    assert np.allclose(z[2], math.log(0.2)) and np.allclose(z[1], math.log(0.1))
    z = guard_scores(p0, pc, 1.0, 0.5, "hard")
    assert z[2] < -1e8 and z[0] > -1e8 and z[1] > -1e8
    # the call weight: an option whose call is likely rises
    s = call_weighted([0.5, 0.5], [0, 1], np.array([0.9, 0.1]), 1.0)
    assert s[0] > s[1]
    s = call_weighted([0.5, 0.5], [None, 1], np.array([0.9, 0.1]), 1.0)
    assert abs(s[0] - s[1]) < 1e-9                           # no call = the least likely call
    print("selftest passed")


if __name__ == "__main__":
    {"report": report, "selftest": selftest}[sys.argv[1]]()
