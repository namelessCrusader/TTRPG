"""Worlds by the thousand, for the decisions they contain.

WHY THIS IS NOT A SCENARIO. The twenty-three scenarios in `scenes.py` each ask
one question and are built by hand to ask it well. Harvested for decisions they
are eleven worlds, and eleven worlds is what you get: measured across six of
them, 809 real choices resolving to **67 distinct (percept, limb, options)
rules**. A model trained on that learns 67 rules and nothing else, however many
rows you feed it.

So this file builds worlds instead of scenes. Nothing here is aimed at a
question — the point is that the ground is different, the things to hand are
different, the drop is a different depth, and the body arrives in a different
state. What varies is chosen to change WHAT IS ON THE MENU, because the menu is
what a decision is made from.

WHAT IT IS TUNED FOR, and it is not realism. Decisions are dense where a body
keeps re-deciding — falling, jumping, catching, being dragged — and nearly
absent in a long fire, because `emergency` commits a body and it stops weighing
the ordinary. Measured: 20 rows a second from `jumper` against 0.02 from
`house_fire`, a thousandfold. Small worlds, short runs, something happening.

THE NUMBER TO WATCH IS DISTINCT RULES, NOT ROWS. A harvest that doubles its
rows and not its rules has done nothing, and that is the failure this whole
file exists to avoid — so `census` reports both and they are printed together.
"""
from __future__ import annotations

import os

import numpy as np

from .sim import (World, ACTS, AIR, WOOD, STONE, IRON, FLESH, LEAD,
                  GLASS, BODY)
from .scenes import _person, _room_for, _Wants
from . import seam


def _rng(seed):
    return np.random.default_rng(seed)


class Roam:
    """A body that tries things, because a body that stands still decides once.

    The scenes drive their people with `_Wants`, which pins one act per limb so
    a render stays about the physics. Harvested, that is a body choosing the
    null act every tick: 25 worlds gave 143 rows and 24 rules.

    This picks off the menu at random, leaning away from doing nothing —
    `_bias` is the chance of preferring a real act over the null one when both
    are offered. That is not a good policy and is not meant to be. It is an
    EXPLORING one, and what a harvest wants is the situations a body gets into,
    not the quality of how it got there. The label on each row is whatever the
    teacher says later; this only decides where the body ends up standing.

    Seeded per world, so a harvest replays exactly."""

    name = "roam"

    def __init__(self, seed, bias=0.8):
        self.r = np.random.default_rng(seed ^ 0x5EED)
        self.bias = bias

    def pick(self, sit, menu):
        if len(menu) < 2:
            return 0
        doing = [i for i, o in enumerate(menu)
                 if not ACTS.get(o.get("verb"), {}).get("null")]
        if doing and float(self.r.random()) < self.bias:
            return int(self.r.choice(doing))
        return int(self.r.integers(0, len(menu)))


def world(seed):
    """One small world with somebody in it, and something about to happen.

    Small on purpose — a 26x18x36 room steps in milliseconds where a 120x60
    house takes a second, and a decision costs the same in either. Everything
    drawn here is drawn to put something different on a menu."""
    r = _rng(seed)
    nx, ny, nz = 26, 18, 36
    open_sky = bool(r.integers(0, 2))

    # THE SHAPE OF THE PLACE. A ledge makes falling and catching possible; a
    # gap makes leaping possible; pillars make going ROUND something possible.
    # Each of those is a different menu.
    kind = int(r.integers(0, 4))
    lip = 0
    solid = []                                            # stone, as fill boxes
    if kind == 1:                                         # a ledge to stand on
        lip = int(r.integers(9, 15))
        h = int(r.integers(8, 20))
        solid.append((lip, nx, 0, ny, 1, h))
    elif kind == 2:                                       # a gap to leap
        g0 = int(r.integers(8, 13))
        g1 = g0 + int(r.integers(2, 6))
        h = int(r.integers(6, 14))
        solid.append((0, g0, 0, ny, 1, h))
        solid.append((min(g1, nx - 2), nx, 0, ny, 1, h))
        lip = g0
    elif kind == 3:                                       # posts to go round
        for _ in range(int(r.integers(1, 4))):
            px = int(r.integers(4, nx - 4))
            py = int(r.integers(3, ny - 4))
            solid.append((px, px + 2, py, py + 2, 1, int(r.integers(8, 30))))
    # AS HIGH ABOVE ITS RAISED FLOOR AS ABOVE ITS GROUND. A ledge or the two
    # sides of a gap are floors people are put on, and a room 36 cells (1.8 m)
    # high over the ground was 1.3 m or less over them: a 1.5 m man put there
    # was built without his head (`fill` clamps at the lattice), in 53 of 150
    # worlds before this (F9). So the room is 36 cells over the higher floor.
    # Drawn in the same order as before; only the lattice is taller. A post
    # is not a floor, and nobody is put on one (`_room_for`, below).
    if kind in (1, 2):
        nz += h - 1
    w = World(nx, ny, nz, voxel_cm=5)
    w.open_sky = open_sky
    w.fill(0, nx, 0, ny, 0, 1, STONE)                     # the ground
    for box in solid:
        w.fill(*box, STONE)

    def ground(x, y):
        """The first free cell above whatever is at this column.

        Worked out per COLUMN, not once for the room. A ledge means the floor
        is at two heights and a body placed at the average of them is either
        buried or standing in the air — and a body placed in stone takes the
        mean of no cells somewhere deep in the percept and hands back a NaN.
        Measured: seed 20 fell over exactly that way."""
        col = np.argwhere(w.mat[x, y, :] != AIR)
        return int(col[:, 0].max()) + 1 if len(col) else 1

    # SOMEBODY, and sometimes somebody else — two people is the only way a
    # menu ever says "take hold of" a person, or "catch".
    #
    # ONLY WHERE THE WHOLE BODY FITS (`_room_for`). The draw says where we
    # would like him; if the body does not fit there (a post under part of
    # him, a ledge through his leg, a post so tall his head is past the
    # ceiling), the nearest x that does is taken instead, nearest first and
    # the lower x on a tie. No random draw is spent on the search, so every
    # world whose drawn place fits is the same world it always was.
    # Measured before this: seed 31 built 'one' as two legs on a post, 84
    # cells of his 396, with no pelvis to join them.
    def place(x, lo):
        for c in sorted(range(lo, nx - 4), key=lambda c: (abs(c - x), c)):
            z = ground(c, ny // 2)
            if _room_for(w, c, ny // 2, z):
                return c, z
        return None

    people = []
    at = place(int(r.integers(3, 8)), 2)
    if at is None:
        raise RuntimeError(f"harvest seed {seed}: no room for anybody")
    px, floor = at
    p = _person(w, px, ny // 2, z0=floor)
    p["name"], p["facing"] = "one", (0.0, 1.0) if r.integers(0, 2) else (1.0, 0.0)
    people.append(p)
    # (from px + 7 on, so his box never meets the first man's)
    if r.integers(0, 3) == 0 and px + 9 < nx - 3 \
            and (at := place(px + 9, px + 7)) is not None:
        q = _person(w, at[0], ny // 2, z0=at[1])
        q["name"] = "two"
        people.append(q)

    # SOMETHING TO HAND. What is reachable decides half of what the hands are
    # offered, and a thing of a different material is a different option.
    for _ in range(int(r.integers(0, 3))):
        mat = (IRON, WOOD, LEAD, GLASS)[int(r.integers(0, 4))]
        ox = int(r.integers(2, nx - 4))
        oy = int(r.integers(2, ny - 4))
        oz = floor + int(r.integers(0, 6))
        if (w.mat[ox:ox + 2, oy:oy + 2, oz:oz + 3] == AIR).all():
            w.fill(ox, ox + 2, oy, oy + 2, oz, oz + 3, mat,
                   frac=float(r.uniform(0.5, 1.0)))

    # A FIRE, sometimes, and small — it is here to put smoke and heat in the
    # percept, not to burn the world down before anybody decides anything.
    if r.integers(0, 4) == 0:
        fx = int(r.integers(2, nx - 4))
        fy = int(r.integers(2, ny - 4))
        if (w.mat[fx:fx + 2, fy:fy + 2, floor:floor + 2] == AIR).all():
            w.fill(fx, fx + 2, fy, fy + 2, floor, floor + 2, WOOD, frac=0.7)
            w.E[fx, fy, floor] = 9.0e5

    # A WAY OUT, sometimes — `knows_a_way_out` and every `go(the door ...)`
    # option depend on there being one.
    w.exits = [(nx - 2, ny // 2)] if r.integers(0, 3) else []

    # AND THE BODY DOES NOT ALWAYS ARRIVE FRESH. A hurt, winded, half-blind
    # body is the same world and a different decision — or it should be, and
    # whether it is is exactly what a harvest is for finding out.
    #
    # ALIVE, every one. The hurt was drawn in [0, 0.5) with death at 0.45
    # (BODY death_burn), so about one body in ten was dead on its first
    # tick: a corpse that never decided anything (the user's decision,
    # 2026-10-10). Drawn below the death line now.
    #
    # AND AWAKE ON ITS BREATH. The blood was drawn in [0.35, 1.0), under the
    # faint line (0.55) about a third of the time. When the blood came back
    # in a second or so that was a body that fainted and CAME ROUND inside
    # the run. Since the breathing clock (tau 40 s, R5) it comes round only
    # after tens of seconds, far past a 90-tick (2.25 s) run, so it was one
    # more body that never decided: 32 worlds fell from 211 rows and 71
    # situations to 143 and 55 with no other change. A body that is out is
    # still drawn, on purpose, by `awake` below. Same draws in the same
    # order; only the ranges moved.
    #
    # AND ON ITS FEET. Hurt between the faint line (0.2, BODY faint_burn)
    # and the death line is a body that is out from its first tick, and
    # stays out: burns do not heal inside a run. Drawn below the faint line
    # now (the user's decision, 2026-10-16). 32 worlds give 684 rows and 168
    # situations this way (wave 11), against 217 rows before.
    for q in people:
        q["hurt"] = float(round(r.uniform(0.0, BODY["faint_burn"]), 3))
        q["blood_o2"] = float(round(r.uniform(BODY["faint_o2"], 1.0), 3))
        if r.integers(0, 4) == 0:
            q["awake"] = False
    return w, people, lip


def run(seed, ticks=90, build=None):
    """One world, stepped, handed back as spelled decision rows.

    `build(seed)` makes the world, as `world` does and by default. A named
    scene (`scenes.workshop_world`) passes its own, so that the things on
    its menus are called by their nouns."""
    w, people, _lip = (build or world)(seed)
    w.seed = seed                          # what every `hash01` draw is keyed on
    w.recall_check = bool(seed % 8 == 0)   # what the cap hid, on one run in 8
    w.policy = Roam(seed)
    for _ in range(ticks):
        w.step()
        for p in people:
            p["events"].clear()
    return rows_of(w, seed)


# WHAT A LINKED MIND'S BODY WROTE IS NOT A CHOICE OVER ITS MENU. A body that a
# mind drives (mindlink.py) is asked one limb at a time, and the row it picks
# is the one that matches the step of a job the mind chose: its trace rows
# say `by: "mind"`, and what they record is a matcher finding its row, not a
# policy weighing the options. Learnt as per-limb choices, they would teach
# that every other row on the menu was passed over for a reason. The mind's
# own decisions are its rows (`Mind.rows`), and the lab reads those.
LINKED = "mind"                     # MindPolicy.name


def rows_of(w, seed):
    """The spelled decision rows of a stepped world, each with how its body
    ended and the seed: the rows of a choice (more than one option), and
    not the rows of a linked mind's body."""
    out = []
    for t in w.trace_outcomes():
        if t.get("by") == LINKED:
            continue
        for row in t.get("spelled", ()):
            if len(row["options"]) > 1:    # ONE option is not a choice
                out.append(dict(row, outcome=t.get("outcome"), seed=seed))
    return out


def census(rows):
    """Rows, and the far smaller number of RULES they resolve to.

    A rule is (percept, limb, the options offered) -> which one. That is the
    function a student has to learn; everything else in the text is context it
    may or may not turn out to need. Reporting rows without this is how a
    harvest convinces itself it is working."""
    import collections
    rules = collections.defaultdict(set)
    for r in rows:
        head = dict(l.split(": ", 1) for l in r["text"].split("\n")
                    if ": " in l and not l.startswith("  "))
        rules[(head.get("percept"), r["limb"], tuple(r["options"]))].add(r["pick"])
    # `situations` IS THE ONE THAT MATTERS while the picks come from `Roam`.
    # A random explorer's answer is noise — it is where the body ENDS UP that
    # a harvest is buying, and the label on each row is whatever the teacher
    # says later. `split` counts how often roam answered one situation two
    # ways, which measures the explorer and not the world.
    return {"rows": len(rows),
            "situations": len(rules),
            "split": sum(1 for v in rules.values() if len(v) > 1),
            "texts": len({r["key"] for r in rows}),
            "limbs": dict(collections.Counter(r["limb"] for r in rows)),
            "outcomes": dict(collections.Counter(r["outcome"] for r in rows))}


def _one(seed):
    """One world, for a worker. Top level so it can be pickled."""
    try:
        return run(seed)
    except Exception as e:                       # a world that falls over is
        return [{"seed": seed, "broke": f"{type(e).__name__}: {e}"}]


class MixedSpelling(ValueError):
    """Rows written in two spellings, in one file."""


def _spelling_of(path):
    """The spelling of the first row of a harvest file, or None if it has
    none yet."""
    import json
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            if line.strip():
                return json.loads(line).get("spelling")
    return None


def to_file(path, seeds, workers=None, chunk=8, append=False):
    """Harvest to a JSONL file, across cores, streaming.

    WORLDS ARE INDEPENDENT AND SEEDED, so this is the easy kind of parallel:
    no shared state, no ordering, and a worker that dies takes one world with
    it. What it is NOT is free in memory — each worker carries its own numpy
    and its own interpreter, about 100 MB, against 0.6 MB for the world it is
    actually simulating. The cost of running a world is almost entirely the
    cost of having somewhere to run it.

    So the default is HALF the cores rather than all of them, because this
    machine has 16 and 2 GB free, and a harvest that swaps is slower than a
    harvest that waits. Pass `workers` to override.

    ONE FILE, ONE SPELLING. A row labelled under one spelling and a row
    labelled under another are not the same kind of row (`seam`), and a file
    that holds both is a dataset nobody can tell is broken. So `append` onto
    a file of another spelling is refused before anything runs, and so is
    any row that comes back spelled differently from the rows before it.

    Rows stream to disk as they arrive rather than piling up: a hundred
    thousand of them is not large, but a harvest that has to finish before it
    writes anything is a harvest you cannot stop.
    """
    import json
    import multiprocessing as mp

    seeds = list(seeds)
    workers = workers or max(1, min(8, (os.cpu_count() or 2) // 2))
    n = broke = 0
    spelling = _spelling_of(path) if append else None
    if spelling is not None and spelling != seam.SPELLING:
        raise MixedSpelling(f"{path} is spelled {spelling!r}; this harvest "
                            f"spells {seam.SPELLING!r}")
    with open(path, "a" if append else "w") as f:
        if workers == 1:
            it = map(_one, seeds)
        else:
            pool = mp.Pool(workers)
            it = pool.imap_unordered(_one, seeds, chunksize=chunk)
        try:
            for rows in it:
                for r in rows:
                    if r.get("broke"):
                        broke += 1
                        continue
                    spelling = spelling or r["spelling"]
                    if r["spelling"] != spelling:
                        raise MixedSpelling(f"a row spelled {r['spelling']!r} "
                                            f"among rows spelled {spelling!r}")
                    f.write(json.dumps(r) + "\n")
                    n += 1
        finally:
            if workers > 1:
                pool.close()
                pool.join()
    return {"rows": n, "worlds": len(seeds), "broke": broke, "path": path,
            "workers": workers}


def read(path):
    """The rows back off disk, for a census or a teacher. A file in more
    than one spelling is refused rather than read (`to_file`)."""
    import json
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    got = sorted({str(r.get("spelling")) for r in rows})
    if len(got) > 1:
        raise MixedSpelling(f"{path} mixes spellings: {', '.join(got)}")
    return rows


if __name__ == "__main__":
    import sys, time
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    n = int(args[0]) if args else 50
    out = args[1] if len(args) > 1 else "harvest.jsonl"
    workers = next((int(a.split("=")[1]) for a in sys.argv[1:]
                    if a.startswith("--workers=")), None)
    t0 = time.time()
    got = to_file(out, range(n), workers=workers)
    secs = time.time() - t0
    c = census(read(out))
    print(f"{got['worlds']} worlds on {got['workers']} workers in {secs:.1f}s "
          f"({got['worlds'] / secs:.1f} worlds/sec, {got['broke']} fell over)")
    print("\n".join(f"  {k}: {v}" for k, v in c.items()))
    print(f"  written to: {out}")
