"""W13 T8: does the 0 C floor ever fire in a golden case?

`World.step` ends with `E = np.maximum(E, -AMBIENT * C)`: nothing below 0 C.
T8 (cold) removes it. If a golden case ever pushes a cell under it, removing it
is a declared golden diff; if none does, it is golden SAME. This reads the 35
golden cases and says which, without changing the sim.

Two readings, both read-only:
  pre-clamp   `np.maximum` is wrapped for the one call whose first argument is a
              world's `E`: it counts the cells that were below the floor BEFORE
              the clamp, and the coldest temperature they would have had. This is
              the exact answer ("did the clamp change anything?").
  post-step   after every step: the coldest cell, and the cells sitting exactly
              on the floor (E <= -AMBIENT * C * (1 - 1e-6)), which only the clamp
              (or a cell at exactly 0 C) leaves.

    python3 tools/w13_floor_probe.py [--tiny] [--only a,b] [--out results.json] [--note TEXT]
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.realpath(os.environ.get("MARK1_REPO", os.path.dirname(HERE)))
os.environ["PYTHONHASHSEED"] = "0"                # tools/golden.py would re-exec without it
os.environ["MARK1_REPO"] = REPO
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))
RESULTS = os.environ.get("W13_RESULTS", "/home/nisargparikh/.cache/mark1-build/w13/M/results.json")
KEY = "floor_probe"

import numpy as np                                                # noqa: E402

from src.voxel import sim                                          # noqa: E402
from src.voxel.materials import AMBIENT, STONE                     # noqa: E402


# ── results file (same few lines in every w13 probe) ──────────────────────────
def record(path, key, row):
    """Merge `row` into the json at `path` under `key` (read, update, write)."""
    try:
        import fcntl
        lk = open(path + ".lk", "a")
        fcntl.flock(lk, fcntl.LOCK_EX)
    except (ImportError, OSError):
        lk = None
    try:
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {}
        data[key] = row
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path + ".tmp", "w") as f:
            json.dump(data, f, indent=1, sort_keys=True)
        os.replace(path + ".tmp", path)
    finally:
        if lk is not None:
            lk.close()


_T_START = time.time()                # when this process began (after the lock, if one was used)


def load_note(extra=""):
    """Machine load note for a result: cpus, load average, how long the shared
    test lock was waited for (the driver sets W13_WAIT_START, epoch seconds)."""
    l1, l5, l15 = os.getloadavg()
    t = os.environ.get("W13_WAIT_START")
    wait = f"waited {int(_T_START - float(t))} s for the shared test lock; " if t else ""
    return (f"{wait}{os.cpu_count()} cpus, load average {l1:.1f}/{l5:.1f}/{l15:.1f} at finish; "
            f"shared with wave 12 and other jobs, so timings are noisy"
            + (f"; {extra}" if extra else ""))



class _NP:
    """`numpy` as sim.py sees it, with `maximum` swapped. Only sim.py's own name
    `np` is replaced: numba walks numpy's ufuncs, so numpy itself is left alone."""

    def __init__(self, real, maximum):
        self._real, self.maximum = real, maximum

    def __getattr__(self, name):
        return getattr(self._real, name)


class Probe:
    """Wraps World.step and sim.py's np.maximum for the length of a `with` block."""

    def __init__(self):
        self.case = None
        self.cur = [None]
        self.pre = {}                    # case -> [cells below the floor, lowest would-be T C]
        self.post_min = {}               # case -> coldest T after any step
        self.post_floor = {}             # case -> cells on the floor after a step (max over steps)

    def __enter__(self):
        self.o_step, self.o_np = sim.World.step, sim.np
        o_max = np.maximum
        me = self

        def step(w, *a, **k):
            me.cur[0] = w
            r = me.o_step(w, *a, **k)
            C = w.heat_capacity()
            T = AMBIENT + w.E / C
            me.post_min[me.case] = min(me.post_min.get(me.case, 99.0), float(T.min()))
            at = int((w.E <= -AMBIENT * C * (1 - 1e-6)).sum())
            if at:
                me.post_floor[me.case] = max(me.post_floor.get(me.case, 0), at)
            return r

        def maxi(a, b=None, *rest, **kw):
            w = me.cur[0]
            if w is not None and a is w.E and isinstance(b, np.ndarray) and b.shape == a.shape:
                below = a < b
                n = int(below.sum())
                if n:
                    aa, bb = a[below].astype(np.float64), b[below].astype(np.float64)
                    t = float((AMBIENT - aa * AMBIENT / bb).min())     # b = -AMBIENT * C
                    row = me.pre.setdefault(me.case, [0, 99.0])
                    row[0] += n
                    row[1] = min(row[1], t)
            return o_max(a, b, *rest, **kw)
        sim.World.step = step
        sim.np = _NP(np, maxi)
        return self

    def __exit__(self, *exc):
        sim.World.step, sim.np = self.o_step, self.o_np


def selftest():
    """The probe must SEE a clamp: one cell is put at -25 C by hand."""
    from src.voxel.sim import World
    w = World(16, 16, 16, voxel_cm=5)
    w.fill(0, 16, 0, 16, 0, 1, STONE)
    C = w.heat_capacity()
    w.E[8, 8, 1] = (np.float32(-25.0) - AMBIENT) * C[8, 8, 1]
    with Probe() as pr:
        pr.case = "selftest"
        for _ in range(3):
            w.step()
    return {"clamped_cells": pr.pre.get("selftest", [0, 0])[0],
            "would_be_T_C": round(pr.pre.get("selftest", [0, 0])[1], 2)}


def run(tiny=False, note="", only=None):
    t0 = time.perf_counter()
    st = selftest()
    seen = st["clamped_cells"] > 0
    if tiny:
        return {"status": "ok", "value": 0 if seen else None, "units": "golden cases below 0 C (tiny: self-test only)",
                "cases_below_0C": [], "selftest": st, "probe_sees_a_clamp": seen,
                "seconds": round(time.perf_counter() - t0, 1), "load_note": load_note(note)}
    import golden as g
    names = only or list(g.ALL)
    errors = {}
    with Probe() as pr:
        for n in names:
            pr.case = n
            try:
                g.run_case(n)
            except Exception as e:
                errors[n] = repr(e)[:160]
    below = sorted(n for n in names if n in pr.pre)
    return {"status": "ok" if not errors else "partial", "value": len(below),
            "units": "golden cases in which a cell was below 0 C before the floor clamp",
            "cases_below_0C": below,
            "pre_clamp": {k: {"cells": v[0], "coldest_would_be_C": round(v[1], 2)} for k, v in pr.pre.items()},
            "cases_with_cells_on_the_floor_after_a_step": sorted(pr.post_floor),
            "min_T_per_case_C": {k: round(v, 2) for k, v in pr.post_min.items()},
            "cases_run": len(names), "errors": errors, "selftest": st,
            "probe_sees_a_clamp": seen,
            "seconds": round(time.perf_counter() - t0, 1), "load_note": load_note(note)}


def main(argv):
    tiny = "--tiny" in argv
    out = argv[argv.index("--out") + 1] if "--out" in argv else RESULTS
    note = argv[argv.index("--note") + 1] if "--note" in argv else ""
    only = argv[argv.index("--only") + 1].split(",") if "--only" in argv else None
    row = run(tiny, note, only)
    record(out, KEY + ("_tiny" if tiny else ""), row)
    print(json.dumps(row, indent=1)[:4000])
    return row


if __name__ == "__main__":
    main(sys.argv[1:])
