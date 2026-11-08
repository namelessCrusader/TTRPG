"""Calibrate a decision model's scores: a temperature and a conformal threshold
(KnowNo-style), so a pick comes with an ACCEPTABLE SET and a "not sure"
(wave 11, D11; no GPU: it reads scores already made).

    python3 -m lab.methods.calibrate              # every source below that has its files
    python3 -m lab.methods.calibrate --alpha 0.1 --splits 200

THE DATA. The labeled cases of the describe A/B (lab/methods/describe_ab.py)
and of the model screen (lab/methods/screen.py) were scored with a score for
each option, and each case has its acceptable set (`good`). The sources:
    gdecide v2     GLiNER2.5-Decide on the src text of this wave (mark1-mind-v2,
                   describe_ab q/new.jsonl, d/new.jsonl): what the engine sends now
    gdecide v1     the same model on wave 10's src text (the screen's set)
    gdecide lab    the same model on the lab's own text (describe_ab d/lab.jsonl)
    <model>        the other models of the screen, on wave 10's src text
A case's scores become probabilities: p = softmax(log(s) / T), with s
floored at 1e-9 (a scorer that gives 0, GLiClass, has no log of it).

THE METHOD (KnowNo, Ren et al. 2023; split conformal prediction). The cases
are cut in two halves by their id (a hash: the same halves every time):
    1. CALIBRATION half. The temperature T is the one that makes the
       acceptable set most likely: it minimises the mean of
       -log(sum of p over the acceptable options). Then each case gets a
       nonconformity score 1 - max(p over its acceptable options), and the
       threshold q is the ceil((n + 1)(1 - alpha))-th smallest of them.
    2. TEST half. The set of a case is every option with p >= 1 - q. It
       COVERS when it holds an acceptable option; conformal prediction
       promises coverage >= 1 - alpha on average over cases like these.
       The model is SURE when the set has exactly one option; else it
       abstains ("not sure": ask for help, or let the rules choose).
The numbers on the test half: coverage (target 90% at alpha 0.1), the mean
set size, the abstain rate, the accuracy when sure, the accuracy@1, and the
expected calibration error of the top probability with T = 1 and with the
fitted T. Then the same on SPLITS random halves (seeded), as a mean, a
standard deviation and a 5th percentile of the coverage, so one lucky cut
does not make the number. Coverage per case set is reported too: the
promise is for the whole, and a set (people, say) can fall short.

The guarantee holds only for cases exchangeable with the calibration
cases: the lab's labeled cases, not the sim's decisions, which have no
acceptable set. Like the scorer, this file imports nothing from src/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"
AB, SCREEN = RESULTS / "describe_ab", RESULTS / "screen"
OUT = RESULTS / "calib"
FLOOR = 1e-9
ALPHA = 0.1
SPLITS = 200
SIZES = (4, 8, 16, 32, 64, 10 ** 9)     # the menu sizes the numbers are cut by
SCREEN_MODELS = ("gdecide1b", "laya", "gliclass", "von", "vegaml")


def rows(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def sources():
    """{name: [(case id, set, scores, good)]} for each source with its files."""
    man = rows(AB / "q" / "manifest.jsonl")
    meta = {m["id"]: m for m in man[1:]}
    out = {}

    def take(name, dec, key):
        got = []
        for d in rows(dec):
            cid = key(d["request"])
            m = meta.get(cid)
            if m is None or not d.get("scores") or len(d["scores"]) != m["n"]:
                continue
            got.append((cid, m["set"], [float(x) for x in d["scores"]], list(m["good"])))
        if got:
            out[name] = got

    spelling = man[0]["src"]["spelling"] if man else "?"
    take(f"gdecide {spelling}", AB / "d" / "new.jsonl", lambda r: r[:-len("#new")])
    take("gdecide v1 (wave 10 text)", SCREEN / "d" / "gdecide" / "labeled.jsonl", lambda r: r[:-len("#new")])
    take("gdecide lab text", AB / "d" / "lab.jsonl", lambda r: r[:-len("#lab")])
    for m in SCREEN_MODELS:
        take(f"{m} (wave 10 text)", SCREEN / "d" / m / "labeled.jsonl", lambda r: r[:-len("#new")])
    return out


def probs(scores, t):
    z = [math.log(max(s, FLOOR)) / t for s in scores]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def nll(cases, t):
    return -sum(math.log(max(sum(p[i] for i in good), 1e-300))
                for _, _, sc, good in cases for p in [probs(sc, t)]) / len(cases)


def fit_t(cases):
    """The temperature with the least NLL of the acceptable sets: a grid in
    log T from 0.05 to 20, then a golden-section search around its best."""
    grid = [math.exp(math.log(0.05) + k * (math.log(400)) / 120) for k in range(121)]
    vals = [nll(cases, t) for t in grid]
    k = min(range(len(grid)), key=vals.__getitem__)
    a, b = math.log(grid[max(k - 1, 0)]), math.log(grid[min(k + 1, len(grid) - 1)])
    g = (math.sqrt(5) - 1) / 2
    for _ in range(40):
        c, d = b - g * (b - a), a + g * (b - a)
        if nll(cases, math.exp(c)) < nll(cases, math.exp(d)):
            b = d
        else:
            a = c
    return math.exp((a + b) / 2)


def threshold(cases, t, alpha):
    """q: the ceil((n + 1)(1 - alpha))-th smallest nonconformity score."""
    s = sorted(1 - max(probs(sc, t)[i] for i in good) for _, _, sc, good in cases)
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return 1.0 if k > len(s) else s[k - 1]


def ece(cases, t, bins=10):
    """The expected calibration error of the top probability (10 equal bins)."""
    b = defaultdict(lambda: [0, 0.0, 0])
    for _, _, sc, good in cases:
        p = probs(sc, t)
        j = max(range(len(p)), key=p.__getitem__)
        x = b[min(int(p[j] * bins), bins - 1)]
        x[0] += 1
        x[1] += p[j]
        x[2] += j in good
    return sum(abs(c / n - a / n) * n for n, c, a in b.values()) / len(cases)


def size_of(n):
    """The menu-size class of a case of `n` options: "<= 8", ..., "> 64"."""
    b = next(b for b in SIZES if n <= b)
    return f"<= {b}" if b != SIZES[-1] else f"> {SIZES[-2]}"


def thresholds(cases, t, alpha):
    """MONDRIAN: one threshold for each menu-size class, each fitted on the
    calibration cases of its class (a class with fewer than 1 / alpha of them
    gets q = 1, every option)."""
    by = defaultdict(list)
    for c in cases:
        by[size_of(len(c[2]))].append(c)
    return {k: threshold(v, t, alpha) for k, v in by.items()}


def judge(cases, t, q):
    """The test numbers of sets {p >= 1 - q}; `q` a number, or a dict of
    one for each menu-size class (Mondrian)."""
    cover = size = sure = sure_ok = top = empty = 0
    by = defaultdict(lambda: [0, 0])
    big = defaultdict(lambda: [0, 0, 0, 0])         # by menu size: n, covered, set sizes, sure
    for _, kind, sc, good in cases:
        p = probs(sc, t)
        qq = q.get(size_of(len(p)), 1.0) if isinstance(q, dict) else q
        got = [i for i, x in enumerate(p) if x >= 1 - qq]
        ok = any(i in good for i in got)
        cover += ok
        size += len(got)
        empty += not got
        by[kind][0] += 1
        by[kind][1] += ok
        x = big[size_of(len(p))]
        x[0] += 1
        x[1] += ok
        x[2] += len(got)
        x[3] += len(got) == 1
        if len(got) == 1:
            sure += 1
            sure_ok += got[0] in good
        top += max(range(len(p)), key=p.__getitem__) in good
    n = len(cases)
    return {"n": n, "coverage": round(cover / n, 4), "mean_set": round(size / n, 3),
            "abstain": round(1 - sure / n, 4), "empty": round(empty / n, 4),
            "acc_when_sure": round(sure_ok / sure, 4) if sure else None, "acc1": round(top / n, 4),
            "coverage_by_set": {k: [v[0], round(v[1] / v[0], 4)] for k, v in sorted(by.items())},
            "by_menu_size": {k: {"n": v[0], "coverage": round(v[1] / v[0], 4),
                                 "mean_set": round(v[2] / v[0], 3), "abstain": round(1 - v[3] / v[0], 4)}
                             for k, v in sorted(big.items(), key=lambda kv: (kv[0][0] == ">", int(kv[0][3:])))}}


def half(cid):
    return int(hashlib.sha256(cid.encode()).hexdigest()[:8], 16) % 2


def run(cases, alpha=ALPHA, splits=SPLITS):
    cal = [c for c in cases if half(c[0]) == 0]
    test = [c for c in cases if half(c[0]) == 1]
    t = fit_t(cal)
    q = threshold(cal, t, alpha)
    q1 = threshold(cal, 1.0, alpha)
    res = {"n_cal": len(cal), "n_test": len(test), "alpha": alpha, "T": round(t, 4), "q": round(q, 6),
           "p_min": round(1 - q, 6), "test": judge(test, t, q),
           "test_T1": dict(judge(test, 1.0, q1), q=round(q1, 6)),
           "nll_test": {"T1": round(nll(test, 1.0), 4), "T": round(nll(test, t), 4)},
           "ece_test": {"T1": round(ece(test, 1.0), 4), "T": round(ece(test, t), 4)}}
    # random halves: T is kept from the fixed split (fitting it 200 times is slow
    # and moves the sets little); q is fitted again on each calibration half
    qm = thresholds(cal, t, alpha)
    res["mondrian"] = {"q": {k: round(v, 6) for k, v in sorted(qm.items())}, "test": judge(test, t, qm)}
    res["splits"] = _splits(cases, t, alpha, splits, threshold)
    res["mondrian"]["splits"] = _splits(cases, t, alpha, splits, thresholds)
    return res


def _splits(cases, t, alpha, splits, fit):
    """Coverage over `splits` random halves (seeded): `fit` makes q on one
    half, the other half is judged."""
    rng = random.Random(0)
    covs, sizes, abst = [], [], []
    for _ in range(splits):
        ix = list(range(len(cases)))
        rng.shuffle(ix)
        c_, t_ = [cases[i] for i in ix[:len(ix) // 2]], [cases[i] for i in ix[len(ix) // 2:]]
        j = judge(t_, t, fit(c_, t, alpha))
        covs.append(j["coverage"])
        sizes.append(j["mean_set"])
        abst.append(j["abstain"])
    mean = sum(covs) / len(covs)
    return {"n": splits, "coverage_mean": round(mean, 4),
            "coverage_sd": round(math.sqrt(sum((x - mean) ** 2 for x in covs) / len(covs)), 4),
            "coverage_p5": round(sorted(covs)[len(covs) // 20], 4),
            "mean_set": round(sum(sizes) / len(sizes), 3), "abstain": round(sum(abst) / len(abst), 4)}


def markdown(res, alpha):
    out = ["# Calibration of the decision models (KnowNo-style split conformal)\n",
           f"alpha {alpha}: the target coverage is {100 * (1 - alpha):.0f}%. Fitted on one half of the"
           " labeled cases (by a hash of the id), judged on the other. A set is every option with"
           " p >= p_min; the model is SURE when the set has one option, else it abstains.\n",
           "| source | n cal / test | T | p_min | coverage % | mean set | abstain % | acc when sure % |"
           " acc@1 % | ECE T=1 -> T | coverage over random halves: mean (sd, p5) % | T=1: coverage % / set /"
           " abstain % |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x: "-" if x is None else f"{100 * x:.1f}"
    for name, r in res.items():
        t, s, t1 = r["test"], r["splits"], r["test_T1"]
        out.append(f"| {name} | {r['n_cal']} / {r['n_test']} | {r['T']} | {r['p_min']:.4f} | {f(t['coverage'])}"
                   f" | {t['mean_set']} | {f(t['abstain'])} | {f(t['acc_when_sure'])} | {f(t['acc1'])}"
                   f" | {r['ece_test']['T1']:.3f} -> {r['ece_test']['T']:.3f}"
                   f" | {f(s['coverage_mean'])} ({f(s['coverage_sd'])}, {f(s['coverage_p5'])})"
                   f" | {f(t1['coverage'])} / {t1['mean_set']} / {f(t1['abstain'])} |")
    out += ["", "## Mondrian: one threshold for each menu size", "",
            "Conformal coverage is promised on average over the cases; a big menu can fall short of it while"
            " a small one is over. A threshold for each menu-size class (fitted on that class's calibration"
            " cases) promises the coverage within each class.", "",
            "| source | coverage % | mean set | abstain % | acc when sure % | coverage over random halves: mean"
            " (sd, p5) % |", "|---|---|---|---|---|---|"]
    for name, r in res.items():
        t, s = r["mondrian"]["test"], r["mondrian"]["splits"]
        out.append(f"| {name} | {f(t['coverage'])} | {t['mean_set']} | {f(t['abstain'])} | {f(t['acc_when_sure'])}"
                   f" | {f(s['coverage_mean'])} ({f(s['coverage_sd'])}, {f(s['coverage_p5'])}) |")
    for name, r in res.items():
        m = r["mondrian"]["test"]["by_menu_size"]
        out += ["", f"## By menu size: {name}", "",
                "| options | n test | coverage % | mean set | abstain % | Mondrian p_min | Mondrian: coverage % /"
                " mean set / abstain % |", "|---|---|---|---|---|---|---|"]
        out += [f"| {k} | {v['n']} | {f(v['coverage'])} | {v['mean_set']} | {f(v['abstain'])}"
                f" | {1 - r['mondrian']['q'].get(k, 1.0):.4f} | {f(m[k]['coverage'])} / {m[k]['mean_set']} /"
                f" {f(m[k]['abstain'])} |" for k, v in r["test"]["by_menu_size"].items()]
        out += ["", f"## Coverage by case set: {name}", "", "| set | n test | coverage % |", "|---|---|---|"]
        out += [f"| {k} | {n} | {f(c)} |" for k, (n, c) in r["test"]["coverage_by_set"].items()]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m lab.methods.calibrate")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--splits", type=int, default=SPLITS)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    res = {}
    for name, cases in sources().items():
        res[name] = run(cases, a.alpha, a.splits)
        print(f"{name}: T {res[name]['T']}, test {res[name]['test']['coverage']}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "numbers.json").write_text(json.dumps(res, indent=1) + "\n")
    (out / "report.md").write_text(markdown(res, a.alpha))
    print(markdown(res, a.alpha))


if __name__ == "__main__":
    main()
