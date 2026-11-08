"""W13 M2: what does float64 E cost in `_law_conduct`?

T7 (body heat) needs the energy array in float64: the float32 add in
`E += qs + qc` rounds a warm flesh cell (about 6000 J, one step 5e-4 J) and
that is the walking-body drift w13_walk_leak.py measures. Option P1 is a world
setting `E_dtype`; the spec accepts it if `_law_conduct` grows by 15% or less.
This times `_law_conduct` itself, and one whole tick, with E float32 and with E
float64, in the two worlds the spec names: the 100x80x50 room with warm people
and the house (the house_fire scene, a few dozen ticks in, so a fire is out).

    python3 tools/w13_f64_conduct.py [--tiny] [--out results.json] [--note TEXT]

The numba kernel compiles per dtype, so both dtypes are called twice before any
timing. The call is made on the hot window the sim itself would use. Medians,
because the machine is shared.
"""
import contextlib
import io
import json
import os
import statistics
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.realpath(os.environ.get("MARK1_REPO", os.path.dirname(HERE)))
sys.path.insert(0, REPO)
RESULTS = os.environ.get("W13_RESULTS", "/home/nisargparikh/.cache/mark1-build/w13/M/results.json")
KEY = "f64_conduct"

import numpy as np                                                # noqa: E402

from src.voxel import scenes, sim                                  # noqa: E402
from src.voxel.sim import World, FLESH, STONE, AIR, AMBIENT        # noqa: E402

LIMIT_PCT = 15.0                      # the T7 spec's acceptance line for option P1


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



# ── worlds ────────────────────────────────────────────────────────────────────
def room_world(npeople=3, tiny=False):
    """The T7 room: 100x80x50 voxels at 5 cm, a stone box, warm (34 C) people."""
    if tiny:
        w = World(16, 16, 16, voxel_cm=5)
        w.fill(0, 16, 0, 16, 0, 1, STONE)
        w.fill(4, 8, 4, 8, 1, 6, FLESH)
    else:
        w = World(100, 80, 50, voxel_cm=5)
        w.fill(0, 100, 0, 80, 0, 1, STONE)
        w.fill(10, 90, 10, 70, 0, 45, STONE)
        w.mat[11:89, 11:69, 1:44] = AIR
        w.smass[11:89, 11:69, 1:44] = 0.0
        for i in range(npeople):
            scenes._person(w, 20 + 18 * i, 20 + 10 * (i % 2) * 2)
    C = w.heat_capacity()
    f = (w.mat == FLESH)
    w.E[f] = (np.float32(34.0) - AMBIENT) * C[f]
    return w


class _Stop(Exception):
    pass


def house_world(ticks_in=30, tiny=False):
    """The house_fire scene, `ticks_in` ticks after it starts (a fire is going)."""
    if tiny:
        w = room_world(tiny=True)
        w.E[8:10, 8:10, 1:3] = np.float32(500.0)
        return w
    real = sim.World.step
    seen = {"n": 0}

    def step(self, *a, **k):
        if seen["n"] >= ticks_in:
            raise _Stop(self)
        seen["n"] += 1
        return real(self, *a, **k)
    sim.World.step = step
    try:
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            scenes.house_fire(d, ticks=ticks_in + 5, every=10 ** 9)
    except _Stop as e:
        return e.args[0]
    finally:
        sim.World.step = real
    raise RuntimeError("house_fire never reached its first tick")


# ── timing ────────────────────────────────────────────────────────────────────
# The machine is shared and its load moves while a probe runs, so the two dtypes
# are timed in ALTERNATION (one call on the float32 world, one on the float64
# world, repeat): a slow stretch hits both.
def prep(w, dtype, fused=True):
    w.E = w.E.astype(dtype)
    w.fused = fused
    sl = w._hot_window()
    sl = sl if sl is not None else (slice(None),) * 3
    for _ in range(2):                    # compile / warm caches for this dtype
        w._law_conduct(sl)
    return sl


def alt_conduct(w32, w64, reps, fused=True):
    s32, s64 = prep(w32, np.float32, fused), prep(w64, np.float64, fused)
    t32, t64 = [], []
    for _ in range(reps):
        for w, sl, ts in ((w32, s32, t32), (w64, s64, t64)):
            t0 = time.perf_counter()
            w._law_conduct(sl)
            ts.append((time.perf_counter() - t0) * 1e3)
    box = int(np.prod([x.stop - x.start for x in s32 if x.stop is not None] or [0]))
    return t32, t64, box


def alt_tick(w32, w64, reps):
    w32.E, w64.E = w32.E.astype(np.float32), w64.E.astype(np.float64)
    t32, t64 = [], []
    for _ in range(reps):
        for w, ts in ((w32, t32), (w64, t64)):
            t0 = time.perf_counter()
            w.step()
            ts.append((time.perf_counter() - t0) * 1e3)
    return t32, t64


def pct(a, b):
    return round(100.0 * (statistics.median(b) / statistics.median(a) - 1.0), 1)


def measure(make, tiny, reps):
    out = {"f32": {}, "f64": {}}
    t32, t64, box = alt_conduct(make(), make(), reps)
    out["hot_window_voxels"] = box
    out["f32"].update(conduct_ms_median=round(statistics.median(t32), 3), conduct_ms_min=round(min(t32), 3))
    out["f64"].update(conduct_ms_median=round(statistics.median(t64), 3), conduct_ms_min=round(min(t64), 3))
    out["conduct_pct_vs_f32"] = pct(t32, t64)
    out["conduct_pct_vs_f32_by_min"] = round(100.0 * (min(t64) / min(t32) - 1.0), 1)
    w = make()
    out["voxels"] = int(np.prod(w.shape))
    k32, k64 = alt_tick(make(), make(), 4 if tiny else 14)
    out["f32"]["tick_ms_median"], out["f64"]["tick_ms_median"] = round(statistics.median(k32), 3), round(statistics.median(k64), 3)
    out["tick_pct_vs_f32"] = pct(k32, k64)
    if not tiny:
        n32, n64, _ = alt_conduct(make(), make(), 8, fused=False)
        out["f32"]["conduct_ms_median_numpy_path"] = round(statistics.median(n32), 3)
        out["f64"]["conduct_ms_median_numpy_path"] = round(statistics.median(n64), 3)
        out["conduct_pct_vs_f32_numpy_path"] = pct(n32, n64)
    out["extra_memory_MB_at_5p1M_voxels"] = round(5.1e6 * 4 / 1e6, 1)
    return out


def run(tiny=False, note=""):
    t0 = time.perf_counter()
    reps = 5 if tiny else 60
    room = measure(lambda: room_world(3, tiny), tiny, reps)
    house = measure(lambda: house_world(30, tiny), tiny, reps)
    worst = max(room["conduct_pct_vs_f32"], house["conduct_pct_vs_f32"])
    return {"status": "ok", "value": worst, "units": "percent more time in _law_conduct, float64 E vs float32 E (worst of room and house)",
            "limit_pct": LIMIT_PCT, "within_limit": worst <= LIMIT_PCT,
            "room": room, "house": house,
            "seconds": round(time.perf_counter() - t0, 1), "load_note": load_note(note)}


def main(argv):
    tiny = "--tiny" in argv
    out = argv[argv.index("--out") + 1] if "--out" in argv else RESULTS
    note = argv[argv.index("--note") + 1] if "--note" in argv else ""
    row = run(tiny, note)
    record(out, KEY + ("_tiny" if tiny else ""), row)
    print(json.dumps(row, indent=1)[:4000])
    return row


if __name__ == "__main__":
    main(sys.argv[1:])
