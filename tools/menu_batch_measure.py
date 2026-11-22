"""Measure the batch size for menus batched then merged (wave 14b lane F).

    python3 tools/menu_batch_measure.py [rules|voxel|all]

rules: for a chooser that DRAWS (the rules, softmax at T_RULES), how far the
  tournament's win chances are from the one-pass draw, by batch size and
  menu length (exact: the tournament is run over many seeds). Also: the
  argmax chooser merges to the exact best at every size.
voxel: the reflex table on a crowded hall. Every decision of a world with
  `menu_batch` set is also asked of the table on the WHOLE menu, and the two
  picks are compared, by batch size.
Heavy part (voxel) must run under the shared lock and cap.
"""
import dataclasses
import random
import sys
import time

from src.mind.api import DecisionRequest
from src.mind.choosers import Rules
from src.mind.core import softmax, T_RULES
from src.mind.tourney import Tourney


def synth(n, rng):
    pr = tuple(round(rng.gauss(0, 0.6), 3) for _ in range(n))     # rule scores spread, about the inn's
    return DecisionRequest("r", "w", "idle", "s", "q", tuple(f"opt {i:02d}" for i in range(n)),
                           tuple("g%d" % (i // 4) for i in range(n)), pr, pr, (0.0,) * n)


def rules_table(samples=1500, menus=12):
    print("RULES (draws): total variation distance of the tournament from the one-pass draw")
    print("  n    " + "".join(f"B={b:<7}" for b in (2, 3, 4, 6, 8, 16)) + " | noise floor (one pass, same samples) | argmax exact")
    for n in (8, 16, 32, 64):
        rng = random.Random(n)
        row, exact = [], True
        tv = {b: 0.0 for b in (2, 3, 4, 6, 8, 16)}
        floor = 0.0
        for m in range(menus):
            r = synth(n, rng)
            full = softmax([a + h for a, h in zip(r.prior, r.habit)], T_RULES)
            one = [0] * n
            for s in range(samples):
                one[Rules(seed=s).decide(dataclasses.replace(r, id=f"r{s}")).pick] += 1
            floor += 0.5 * sum(abs(one[i] / samples - full[i]) for i in range(n)) / menus
            for b in tv:
                # the mean of the tournament's own reported joint over seeds is
                # NOT the win chance (it follows the realised path), so draw it
                win = [0] * n
                for s in range(samples):
                    d = Tourney(Rules(seed=s), b)
                    d.submit(dataclasses.replace(r, id=f"r{s}"))
                    win[d.poll(f"r{s}").pick] += 1
                tot = samples
                tv[b] += 0.5 * sum(abs(win[i] / tot - full[i]) for i in range(n)) / menus
            for b in (2, 4, 8):
                t = Tourney(Rules(seed=0, argmax=True), b)
                t.submit(r)
                exact &= t.poll("r").pick == max(range(n), key=r.prior.__getitem__)
        print(f"  {n:<4} " + "".join(f"{tv[b]:<9.3f}" for b in tv) + f" | {floor:<8.3f} | {exact}")


def voxel_table():
    import numpy as np
    from src.voxel.sim import World
    from src.voxel.materials import AIR, STONE, WOOD
    from src.voxel.scenes import _person
    from src.voxel.will import WillMixin
    print("VOXEL reflex table, crowded hall with a fire, 200 ticks: tournament pick == whole-menu pick")
    print("  B    decisions  merged  same   time ms/tick")
    for B in (3, 4, 5, 7, 10):
        w = World(90, 60, 40, voxel_cm=5)
        w.open_sky = False
        w.fill(0, 90, 0, 60, 0, 40, STONE)
        w.mat[1:89, 1:59, 1:39] = AIR
        w.smass[1:89, 1:59, 1:39] = 0.0
        w.exits = [(88, 12), (88, 30), (88, 48), (1, 12), (1, 48), (45, 58)]
        w.fill(10, 18, 20, 30, 1, 3, WOOD, frac=0.8)
        w.E[10, 20, 1] = 9.0e5
        for n, (x, y) in enumerate(((30, 10), (60, 40), (20, 50), (70, 12))):
            _person(w, x, y)["name"] = f"P{n}"
        w.menu_batch = B
        count = [0, 0, 0]
        real = WillMixin._merged_pick

        def spy(self, sit, limb, chosen, menu, batch, _real=real, _c=count):
            i = _real(self, sit, limb, chosen, menu, batch)
            j = self.policy.pick(dict(sit, limb=limb, chosen=dict(chosen)), menu)
            _c[1] += 1
            _c[2] += menu[i]["key"] == menu[j]["key"] or \
                (menu[i]["tag"], menu[i].get("danger", 0), menu[i].get("away", 0)) == \
                (menu[j]["tag"], menu[j].get("danger", 0), menu[j].get("away", 0))
            return i
        WillMixin._merged_pick = spy
        try:
            t0 = time.perf_counter()
            for _ in range(200):
                w.step()
            ms = (time.perf_counter() - t0) / 200 * 1000
        finally:
            WillMixin._merged_pick = real
        decisions = len(w.traces)
        print(f"  {B:<4} {decisions:<10} {count[1]:<7} {count[2]:<6} {ms:.2f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("rules", "all"):
        rules_table()
    if which in ("voxel", "all"):
        voxel_table()
