"""Score one decision model on a card set.

    lab/.venv/bin/python -m lab.run laya  [--cards lab/cards/social_v1.jsonl]

ONE MODEL PER PROCESS, on purpose. The machine has 14 GB of RAM and 6 GB of
VRAM, and the simulation's suites want most of the first; loading two 400M
encoders side by side to save a process start is how a laptop swaps. Each run
writes lab/results/<model>__<cards>.jsonl (every answer, with its probability
and time) and prints the scoreboard.

The scoreboard always carries the MAJORITY baseline beside the model: 12 of 30
MIND cards are `comply`, so 40% on reply is what guessing buys, and a number
without that line beside it means nothing.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from lab.models import MODELS, state_text

HERE = Path(__file__).parent


def load(path):
    return [json.loads(line) for line in open(path)]


def majority(cards):
    """The best any model that ignores the state can do: always the commonest
    answer, or a blind pick where the options differ card to card (a target
    list with one name on it is a free hit, and must not count as skill)."""
    best, chance = {}, {}
    for c in cards:
        for q, g in c["gold"].items():
            k = (c["track"], q)
            best.setdefault(k, Counter())[json.dumps(g)] += 1
            crit = c["questions"][q].get("criteria") or [0, 1]
            chance.setdefault(k, []).append(1 / len(crit))
    return {k: max(v.most_common(1)[0][1] / sum(v.values()),
                   sum(chance[k]) / len(chance[k]))
            for k, v in best.items()}


def auc(pairs):
    """Does a yes rank above a no? Zero-shot, a model's probabilities can carry
    the signal and sit on the wrong side of 0.5 — Laya's "asks" put every
    request above every non-request and still called most of them no. 0.5 is
    no signal at all; accuracy alone cannot tell the two apart."""
    pos = [p for p, g in pairs if g and p is not None]
    neg = [p for p, g in pairs if not g and p is not None]
    if not pos or not neg:
        return None
    won = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return won / (len(pos) * len(neg))


def grade(q, gold, ans):
    kind = q["type"]
    got = ans.get("answer")
    if kind == "score":
        return {"hit": got == gold, "near": got is not None and abs(got - gold) <= 1}
    if kind == "noul":
        p = ans.get("p_true")
        return {"hit": got == gold,
                "brier": None if p is None else (p - float(gold)) ** 2}
    return {"hit": got == gold}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("--cards", default=str(HERE / "cards" / "social_v1.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cards = load(args.cards)[: args.limit or None]
    model = MODELS[args.model]()
    t0 = time.perf_counter()
    model.load()
    load_s = time.perf_counter() - t0

    rows = []
    for c in cards:
        t = time.perf_counter()
        answers = model.decide(state_text(c["state"]), c["questions"])
        ms = (time.perf_counter() - t) * 1000
        marks = {q: grade(c["questions"][q], c["gold"][q], answers.get(q, {}))
                 for q in c["questions"]}
        rows.append({"id": c["id"], "track": c["track"], "ms": ms,
                     "gold": c["gold"], "answers": answers, "marks": marks})

    out = HERE / "results" / f"{args.model}__{Path(args.cards).stem}.jsonl"
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    base = majority(cards)
    print(f"{args.model}: loaded in {load_s:.1f}s, "
          f"{sum(r['ms'] for r in rows) / len(rows):.0f} ms a card "
          f"(first {rows[0]['ms']:.0f}), peak VRAM {model.vram_mb():.0f} MB")
    print(f"{'track':6} {'question':10} {'model':>6} {'guess':>6}  extra")
    for (track, q), guess in sorted(base.items()):
        m = [r["marks"][q] for r in rows if r["track"] == track]
        hit = sum(x["hit"] for x in m) / len(m)
        extra = ""
        if "near" in m[0]:
            extra = f"within one {sum(x['near'] for x in m) / len(m):.0%}"
        if m[0].get("brier") is not None:
            extra = f"brier {sum(x['brier'] for x in m) / len(m):.3f}"
            a = auc([(r["answers"][q].get("p_true"), r["gold"][q])
                     for r in rows if r["track"] == track])
            if a is not None:
                extra += f"  auc {a:.2f}"
        print(f"{track:6} {q:10} {hit:6.0%} {guess:6.0%}  {extra}")
    print(out)


if __name__ == "__main__":
    main()
