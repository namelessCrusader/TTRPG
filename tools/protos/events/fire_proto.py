"""PROTOTYPE (scratch only): fire events from ONE predicate, `burning()`,
computed once at the perception slot and handed to `_law_will`.

Checks: (1) cost per tick of burning() + the box bookkeeping; (2) the event
list is the same with skip_quiet True and False; (3) every lit cell at the
perception slot lies in a box (so a sees_fire cell always has an ignites
cause); (4) flicker: how many burns_out with and without hysteresis.

    flock <lock> python3 fire_proto.py
"""
import sys
import time

import numpy as np

REPO = "/home/nisargparikh/Desktop/Fun Stuff/Mark_1"
sys.path.insert(0, REPO)
from src.voxel.sim import World, STONE, WOOD                 # noqa: E402
from src.voxel.scenes import _person                         # noqa: E402

GROW = 2          # a new lit cell within 2 voxels of a box joins it
SEED = 3          # new cells closer than this (Chebyshev) are one new fire
OUT_AFTER = 10    # ticks with nothing lit in a box before it "burns out"


class Fires:
    def __init__(self):
        self.boxes = {}          # eid -> [lo(3), hi(3), last_lit_tick]
        self.prev = None
        self.log = []            # (tick, verb, eid, cause)
        self.next = 0
        self.uncovered = 0

    def update(self, w, lit, hyst=True):
        tick = w.tick
        cells = np.argwhere(lit) if lit is not None else np.zeros((0, 3), np.int64)
        claimed = np.zeros(len(cells), bool)
        for eid, box in self.boxes.items():          # insertion = eid order
            if not len(cells):
                break
            lo, hi = box[0] - GROW, box[1] + GROW
            inb = ((cells >= lo) & (cells <= hi)).all(axis=1) & ~claimed
            if inb.any():
                box[0] = np.minimum(box[0], cells[inb].min(axis=0))
                box[1] = np.maximum(box[1], cells[inb].max(axis=0))
                box[2] = tick
                claimed |= inb
        rest = cells[~claimed]                         # argwhere: sorted, so
        while len(rest):                               # deterministic
            seed = rest[0]
            lo, hi = seed.copy(), seed.copy()
            while True:
                inb = ((rest >= lo - SEED) & (rest <= hi + SEED)).all(axis=1)
                nlo, nhi = rest[inb].min(axis=0), rest[inb].max(axis=0)
                if (nlo == lo).all() and (nhi == hi).all():
                    break
                lo, hi = nlo, nhi
            eid = self.next; self.next += 1
            self.boxes[eid] = [lo, hi, tick]
            self.log.append((tick, "ignites", eid, None))
            rest = rest[~inb]
        for eid in list(self.boxes):
            if tick - self.boxes[eid][2] >= (OUT_AFTER if hyst else 1):
                self.log.append((tick, "burns_out", self.next, eid))
                self.next += 1
                del self.boxes[eid]
        # invariant: every lit cell is inside some box (grown or new)
        for c in cells:
            if not any(((c >= b[0]) & (c <= b[1])).all() for b in self.boxes.values()):
                self.uncovered += 1


def build(skip):
    w = World(60, 60, 40, voxel_cm=5)
    w.skip_quiet = skip
    w.fill(0, 60, 0, 60, 0, 1, STONE)
    w.fill(26, 32, 26, 32, 1, 6, WOOD, frac=0.7)     # a crib
    w.E[28, 28, 2] = 9.0e5                           # lit
    w.fill(40, 44, 10, 14, 1, 4, WOOD, frac=0.7)     # a second crib, lit later
    a = _person(w, 10, 30); a["name"] = "a"
    b = _person(w, 48, 40); b["name"] = "b"
    return w


def run(skip, ticks=160, hyst=True):
    w = build(skip)
    F = Fires()
    cost = []
    legacy_hear = World._law_hear

    def perceive(self):
        t0 = time.perf_counter()
        lit = self.burning() if (self.persons and self._busy(self.E)) else None
        F.update(self, lit, hyst)
        cost.append(time.perf_counter() - t0)
        legacy_hear(self)
    w._law_hear = perceive.__get__(w)
    t = time.perf_counter()
    for k in range(ticks):
        if k == 60:
            w.E[42, 12, 2] = 9.0e5
        w.step()
        for p in w.persons:
            p["events"].clear()
    tick_ms = 1e3 * (time.perf_counter() - t) / ticks
    return F, 1e3 * float(np.mean(cost)), tick_ms


if __name__ == "__main__":
    Fa, ca, ta = run(True)
    Fb, cb, tb = run(False)
    Fc, _, _ = run(True, hyst=False)
    print(f"skip_quiet=True : {len(Fa.log)} fire events, bookkeeping "
          f"{ca:.2f} ms/tick of a {ta:.0f} ms tick")
    print(f"skip_quiet=False: {len(Fb.log)} fire events, bookkeeping "
          f"{cb:.2f} ms/tick of a {tb:.0f} ms tick")
    print("event lists equal across skip_quiet:", Fa.log == Fb.log)
    print("lit cells outside every box:", Fa.uncovered, Fb.uncovered)
    print("with hysteresis:", [(t, v) for t, v, _e, _c in Fa.log])
    print("without hysteresis: burns_out count",
          sum(1 for _t, v, _e, _c in Fc.log if v == "burns_out"),
          "ignites count", sum(1 for _t, v, _e, _c in Fc.log if v == "ignites"))
