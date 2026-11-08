"""The inn goldens: three nights of the lab inn under the rules policy,
written to tests/data. The rooms port (src/rooms) must give the same bytes.

    python3 -m lab.world2.golden            # write tests/data/inn_rules_seed{0,1,2}.jsonl
    python3 -m lab.world2.golden --check    # run the nights again; exit 1 if a file differs

Line 1 of each file is a header: the seed, the turns, the number of events and
the sha1 of the two lab files that made the night. Each line after it is one
event, written as lab/world2/run.py writes it (json.dumps of the log row, in
log order), so the events equal lab/results/world2b/rules/seed<k>.events.jsonl.

A night runs as run.py runs it: World(seed, turns), then Rules() for each
turn, and it stops early only when every person is dead or down.

src and lab never import each other (docs/sim-mind-design.md, 2.1). The port
reads these files and does not import the lab.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from lab.world2.policy import Rules
from lab.world2.world import World

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "tests" / "data"
SEEDS, TURNS = (0, 1, 2), 300
SOURCES = ("lab/world2/world.py", "lab/world2/policy.py")


def night(seed, turns=TURNS):
    w = World(seed=seed, turns=turns)
    policy = Rules()
    for _ in range(turns):
        w.step(policy)
        if all(p.dead or p.down for p in w.people.values()):
            break
    return w


def golden(seed, turns=TURNS):
    """The text of one golden file: the header line, then the events."""
    w = night(seed, turns)
    head = {"golden": "inn", "policy": "rules", "seed": seed, "turns": turns, "events": len(w.log),
            "sha1": {f: hashlib.sha1((REPO / f).read_bytes()).hexdigest() for f in SOURCES}}
    return "\n".join(json.dumps(e) for e in [head] + w.log) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare with the files on disk; write nothing")
    a = ap.parse_args()
    bad = 0
    for seed in SEEDS:
        f = OUT / f"inn_rules_seed{seed}.jsonl"
        text = golden(seed)
        n = text.count("\n") - 1
        if not a.check:
            OUT.mkdir(parents=True, exist_ok=True)
            f.write_text(text)
            print(f"{f.relative_to(REPO)}: {n} events, sha256 {hashlib.sha256(text.encode()).hexdigest()}")
            continue
        old = f.read_text() if f.exists() else None
        if old == text:
            print(f"{f.name}: SAME ({n} events)")
            continue
        bad += 1
        if old is None:
            print(f"{f.name}: MISSING")
            continue
        # the header and the events are told apart: a changed lab file that
        # gives the same events only needs the header written again
        was, now = old.splitlines(), text.splitlines()
        head = "SAME" if was[:1] == now[:1] else "DIFFERS"
        i = next((i for i, (x, y) in enumerate(zip(was, now)) if i and x != y), min(len(was), len(now)))
        events = "SAME" if was[1:] == now[1:] else f"DIFFER from line {i + 1}"
        print(f"{f.name}: header {head}, events {events}; {len(was) - 1} events on disk, {n} now")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
