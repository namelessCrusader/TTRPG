"""W13 T11: what does route planning cost on a real World?

The cost of a route is three things, and the plan (C9) needs them apart:
  fit grid     `_fit_grid`: which columns the whole body fits in (a floor test,
               then one shifted AND per distinct offset of the body's cells);
  sweep        `_routes` on a ready `fit`: the eight-way priced search from the
               body's column, read out for each target (1 near target, 3 targets);
  per-look     `_walkable`: the floor test alone, which `senses.py` runs on every
               look of every person (`walk_now = self._walkable(cells)`).

Measured at 0.4, 1.6 and 5.1 M voxels (5 cm), one person standing on a stone
plane, everything known, targets 40 and 30 columns away and a far corner. Median
of a few repeats. The per-look number is the one SCALE owns; it is reported, not
claimed.

    python3 tools/w13_route_bench.py [--tiny] [--out results.json] [--note TEXT]
"""
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.realpath(os.environ.get("MARK1_REPO", os.path.dirname(HERE)))
sys.path.insert(0, REPO)
RESULTS = os.environ.get("W13_RESULTS", "/home/nisargparikh/.cache/mark1-build/w13/M/results.json")
KEY = "route_bench"

import numpy as np                                                # noqa: E402

from src.voxel import scenes                                       # noqa: E402
from src.voxel.sim import World, STONE, FLESH                      # noqa: E402

SHAPES = {"0.4M": (160, 100, 26), "1.6M": (320, 200, 25), "5.1M": (400, 256, 50)}


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



def med_ms(fn, reps):
    ts = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), out


def bench(shape, tiny):
    nx, ny, nz = shape
    w = World(nx, ny, nz, voxel_cm=5)
    w.fill(0, nx, 0, ny, 0, 1, STONE)
    if tiny:                                      # a 16^3 world cannot hold a person: a small block
        w.fill(6, 8, 6, 8, 1, 5, FLESH)
        cells = np.argwhere(w.mat == FLESH)
        pr = None
    else:
        scenes._person(w, nx // 2, ny // 2)
        for _ in range(2):
            w.step()
        pr = w.persons[0]
        cells = np.array(sorted(pr["_own"]))
        pr["known"][:] = True
    reps = 3 if tiny else 5
    t_walk, _ = med_ms(lambda: w._walkable(cells), reps)
    t_fit, fs = med_ms(lambda: w._fit_grid(cells, pr), reps)
    fit, start = fs
    cx, cy = start
    tg = [(min(cx + 40, nx - 2), cy), (max(cx - 30, 1), min(cy + 20, ny - 2)), (3, 3)]
    t_r3, r3 = med_ms(lambda: w._routes(cells, tg, pr, fit=fit), 3 if not tiny else 2)
    t_r1, _ = med_ms(lambda: w._routes(cells, tg[:1], pr, fit=fit), 3 if not tiny else 2)
    t_all, _ = med_ms(lambda: w._routes(cells, tg, pr), 3 if not tiny else 2)
    return {"voxels": nx * ny * nz, "plane_columns": nx * ny, "body_cells": int(len(cells)),
            "fit_columns": int(fit.sum()),
            "per_look_walkable_ms": round(t_walk, 2),
            "fit_grid_ms": round(t_fit, 2),
            "fit_grid_beyond_floor_test_ms": round(max(t_fit - t_walk, 0.0), 2),
            "sweep_3_targets_ms": round(t_r3, 1), "sweep_1_near_target_ms": round(t_r1, 1),
            "route_total_ms_fit_plus_sweep_3_targets": round(t_all, 1),
            "route_lengths": [len(v) if v else None for v in r3.values()]}


def run(tiny=False, note=""):
    t0 = time.perf_counter()
    shapes = {"tiny": (16, 16, 16)} if tiny else SHAPES
    table = {name: bench(sh, tiny) for name, sh in shapes.items()}
    big = table.get("5.1M") or next(iter(table.values()))
    return {"status": "ok", "value": big["route_total_ms_fit_plus_sweep_3_targets"],
            "units": "ms for one full route decision (fit grid + sweep, 3 targets) at the largest size",
            "plan_budget_ms": "1-8 (plan, C9)",
            "table": table, "seconds": round(time.perf_counter() - t0, 1),
            "load_note": load_note(note)}


def main(argv):
    tiny = "--tiny" in argv
    out = argv[argv.index("--out") + 1] if "--out" in argv else RESULTS
    note = argv[argv.index("--note") + 1] if "--note" in argv else ""
    row = run(tiny, note)
    record(out, KEY + ("_tiny" if tiny else ""), row)
    print(json.dumps(row, indent=1)[:5000])
    return row


if __name__ == "__main__":
    main(sys.argv[1:])
