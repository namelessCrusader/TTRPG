"""Does a formulation make the task doable, and is it the MODEL doing it?

    lab/.venv/bin/python -m lab.suite                      # everything
    lab/.venv/bin/python -m lab.suite --rooms 10 --only base,both --pickers words

Four formulations (lab/rooms.py `Room.enrich`, `Room.view`) in a 2x2, so each
change is measured alone and together:

    base   as the rooms were
    opts   options say what they bring you to ("one step closer to the chest")
    view   detail only for what is near or known about; the rest by name
    both

Three pickers on every formulation:

    random  the floor
    words   picks the option sharing the most content words with the goal,
            what you know and what you hold. THE LEAKAGE CHECK: if a word
            matcher does as well as the model, the formulation gave the answer
            away and the model proved nothing.
    laya    zero-shot

Two measurements, both on rooms built from the TEST name pool:

    A  single decisions: agreement with the expert, per question
    B  whole rooms: how far each gets

A formulation is progress only where Laya beats `words` on both.
"""
from __future__ import annotations

import argparse
import os
import random
import re
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")      # the laptop runs hot

from lab.rooms import MILESTONES, Room, play       # noqa: E402

LIMBS = ("legs", "hands", "mouth")
VARIANTS = {"base": (False, "full"), "opts": (True, "full"),
            "view": (False, "focus"), "both": (True, "focus")}
STOP = set("the a an of to in on at is it you your and or for with what do now which "
           "one step take nothing say out get want are am be".split())


def words(t):
    return {w for w in re.findall(r"[a-z]+", t.lower()) if w not in STOP}


class Random:
    name = "random"

    def __init__(self):
        self.rng = random.Random(0)

    def pick(self, state, q):
        return self.rng.choice(list(q["criteria"]))


class Words:
    """Content-word overlap between an option (name and description) and the
    lines that carry intent: the goal, what you know, what you hold."""
    name = "words"

    def __init__(self):
        self.rng = random.Random(0)

    def pick(self, state, q):
        ctx = " ".join(l for l in state.split("\n")
                       if l.startswith(("You are locked", "You know", "You are holding")))
        cw = words(ctx)
        score = {k: len(words(k + " " + (v or "")) & cw) for k, v in q["criteria"].items()}
        top = max(score.values())
        return self.rng.choice([k for k, v in score.items() if v == top])


class Laya:
    name = "laya"

    def __init__(self):
        import torch
        torch.set_num_threads(4)
        import laya
        self.a = laya.load("convaiinnovations/laya")

    def pick(self, state, q):
        return self.a.predict(state, {"q": q})["answers"]["q"]["choice"]


def decisions(seeds, noise=0.25):
    """Expert play with noise; each visited state with every question and the
    expert's answer. Re-generated per formulation, since the text differs."""
    rng = random.Random(seeds[0])
    out = []
    for s in seeds:
        r = Room(s, "test")
        for _ in range(60):
            st, qs = r.state(), r.questions()
            best = r.expert()
            tq = r.target_question()
            if r.target in tq["criteria"]:
                out.append(("target", st, tq, r.target))
            for l in LIMBS:
                if len(qs[l]["criteria"]) > 1:
                    out.append((l, st, qs[l], best[l]))
            r.act(best if rng.random() > noise else
                  {l: rng.choice(list(q["criteria"])) for l, q in qs.items()})
            if r.out:
                break
    return out


def suite_a(picker, rows):
    hit, n = {}, {}
    for kind, st, q, gold in rows:
        n[kind] = n.get(kind, 0) + 1
        hit[kind] = hit.get(kind, 0) + (picker.pick(st, q) == gold)
    return {k: hit[k] / n[k] for k in n}


def suite_b(picker, seeds):
    counts = {m: 0 for m in MILESTONES}

    def choose(room, st, qs):
        return {l: (next(iter(q["criteria"])) if len(q["criteria"]) == 1 else picker.pick(st, q))
                for l, q in qs.items()}

    for s in seeds:
        r = play(Room(s, "test"), choose)
        for m in MILESTONES:
            counts[m] += m in r.done
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, default=30)
    ap.add_argument("--only", default=",".join(VARIANTS))
    ap.add_argument("--pickers", default="random,words,laya")
    args = ap.parse_args()
    pickers = []
    for name in args.pickers.split(","):
        pickers.append({"random": Random, "words": Words, "laya": Laya}[name]())
    a_seeds = list(range(50_000, 50_000 + args.rooms))
    b_seeds = list(range(70_000, 70_000 + args.rooms))
    lines = []

    def say(line):
        print(line, flush=True)
        lines.append(line)

    short = {"learned where the key is": "learned", "reached the key's container": "at box",
             "opened the key's container": "opened", "took the key": "key",
             "reached the door": "at door", "unlocked the door": "unlocked",
             "opened the door": "door open", "out": "OUT"}
    for v in args.only.split(","):
        Room.enrich, Room.view = VARIANTS[v]
        rows = decisions(a_seeds)
        for p in pickers:
            t0 = time.time()
            a = suite_a(p, rows)
            b = suite_b(p, b_seeds)
            say(f"{v:5} {p.name:6} A: " + "  ".join(f"{k} {a[k]:.0%}" for k in
                                                   ("target", "legs", "hands", "mouth") if k in a)
                + "  | B: " + "  ".join(f"{short[m]} {b[m]}" for m in MILESTONES)
                + f"  /{args.rooms}  ({time.time() - t0:.0f}s)")
    Room.enrich, Room.view = False, "full"
    out = os.path.join(os.path.dirname(__file__), "results", "suite.txt")
    with open(out, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M\n") + "\n".join(lines) + "\n\n")


if __name__ == "__main__":
    main()
