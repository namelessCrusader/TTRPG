"""Costs of the slow-time scheduler (wave 13, T5, spec section 13).

    python3 tools/bench_slow.py            # the table
    python3 tools/bench_slow.py --json     # the same, as one JSON object

Measures on THIS machine, nothing assumed: the heap, pre() with nothing due,
a deepcopy of 10,000 and 20,000 processes, the cheap quiet gates and the digest
on a room, a probe period, a tier-0 jump against the same ticks, one form
advance each. A world benchmark steps real ticks, so run it under
`flock .../test.lock .../cap.sh` like any heavy job. Wall-clock numbers go to
this table and nowhere else (SkipReport carries none).
"""
import copy
import heapq
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np                                    # noqa: E402

from src.voxel import quiet as Q                      # noqa: E402
from src.voxel import slowforms as F                  # noqa: E402
from src.voxel import slowtime as ST                  # noqa: E402
from src.voxel.materials import STONE                 # noqa: E402
from src.voxel.scenes import _person                  # noqa: E402
from src.voxel.sim import World                       # noqa: E402


class SlowWorld(World):
    slow = None

    def _tick_start(self):
        super()._tick_start()
        if self.slow is not None:
            self.slow.pre(self)


def _room(nx=24, ny=20, nz=40):
    w = SlowWorld(nx, ny, nz, voxel_cm=5)
    w.fill(0, nx, 0, ny, 0, 1, STONE)
    w.fill(0, nx, 0, 1, 0, nz, STONE)
    w.fill(0, nx, ny - 1, ny, 0, nz, STONE)
    w.fill(0, 1, 0, ny, 0, nz, STONE)
    w.fill(nx - 1, nx, 0, ny, 0, nz, STONE)
    w.fill(0, nx, 0, ny, nz - 1, nz, STONE)
    w.exits = []
    return w


def timeit(fn, n=1):
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n


def many(n):
    s = ST.Scheduler()
    for i in range(n):
        s.add("decay", "owner", ("world",), {"half_s": 3600.0 + i % 7}, (1.0,), 0)
    return s


def main():
    out = {}
    h = []
    out["heap push (us)"] = timeit(lambda: heapq.heappush(h, (np.random.randint(1 << 30), 0, 1, 0, 0)), 20000) * 1e6
    out["heap pop at 20000 (us)"] = timeit(lambda: heapq.heappop(h), 10000) * 1e6
    w = _room()
    p = _person(w, 8, 8, z0=1)
    p["awake"], p["asleep"] = True, True
    w.slow = ST.Scheduler()
    for _ in range(60):
        w.step()
    out["pre() nothing due (us)"] = timeit(lambda: w.slow.pre(w), 200000) * 1e6
    for n in (1000, 10000, 20000):
        s = many(n)
        out[f"deepcopy {n} processes (ms)"] = timeit(lambda: copy.deepcopy(s)) * 1e3
    out["add 20000 owners (ms)"] = timeit(lambda: many(20000)) * 1e3
    out["quiet cheap gates, 24x20x40 (ms)"] = timeit(lambda: Q.cheap_blockers(w), 50) * 1e3
    out["digest, 24x20x40 + a person (ms)"] = timeit(lambda: Q.digest(w), 5) * 1e3
    out["ms per tick, that room"] = timeit(w.step, 300) * 1e3
    a, b = _room(), _room()
    for k in (a, b):
        q = _person(k, 8, 8, z0=1)
        q["awake"], q["asleep"] = True, True
    b.slow = ST.Scheduler()
    for _ in range(60):
        a.step()
        b.step()
    N = 20000
    out[f"{N} real ticks (s)"] = timeit(lambda: [a.step() for _ in range(N)])
    t = time.perf_counter()
    rep = ST.advance(b, N, sched=b.slow)
    out[f"advance({N}) with a jump (s)"] = time.perf_counter() - t
    out["  of which jumped / stepped"] = f"{rep.jumped} / {rep.stepped}"
    Cc = F.FORMS
    d = F.SineDrive(15.0, 10.0, 86400.0)
    day = 144000
    out["decay advance 30 days, sine (us)"] = timeit(
        lambda: Cc["decay"].advance((1.0,), {"A": 1e10, "Ea": 80e3}, {"T": d}, 0, 30 * day), 20) * 1e6
    p0 = {"b": 0.0437, "Tmin": -3.0, "Tmax": 48.0, "c": 0.3, "T_kill": 60.0,
          "D_ref_s": 180.0, "T_ref_C": 121.1, "z": 10.0}
    out["population advance 10 days, sine (us)"] = timeit(
        lambda: Cc["population"].advance((6.9, 0.0), p0, {"T": d}, 0, 10 * day), 20) * 1e6
    out["wells similarity table, first build (ms)"] = timeit(
        lambda: (F._SIM.clear(), F.similarity_table(0.07, 0.06, 1.0, 0.2))) * 1e3
    spec = F.NetSpec([1000.0] * 200, np.arange(199), np.arange(1, 200), [1.0] * 199,
                     [0.1] + [0.0] * 199)
    out["rc_net eigen 200 nodes, 8 h (ms)"] = timeit(
        lambda: F.rc_advance_arrays(spec, np.full(200, 20.0), {"Tb": F.ConstDrive(5.0)}, 0, 8 * 144000), 3) * 1e3
    # tier 1: the 300 W stand-in for four sleepers in a closed 22 x 22 x 18 room (5 cm cells)
    from src.voxel import slownet as SN              # noqa: E402
    from src.voxel.materials import IRON             # noqa: E402
    for secs, mx in ((300.0, 10000), (300.0, 2000), (3600.0, 2000)):
        r = SlowWorld(22, 22, 18, voxel_cm=5)
        r.fill(0, 22, 0, 22, 0, 1, STONE)
        r.fill(0, 22, 0, 1, 0, 18, STONE)
        r.fill(0, 22, 21, 22, 0, 18, STONE)
        r.fill(0, 1, 0, 22, 0, 18, STONE)
        r.fill(21, 22, 0, 22, 0, 18, STONE)
        r.fill(0, 22, 0, 22, 17, 18, STONE)
        r.fill(9, 13, 9, 11, 1, 6, IRON)
        for _ in range(3):
            r.step()
        cells = np.argwhere(r.mat[9:13, 9:11, 1:6] == IRON) + np.array([9, 9, 1])
        t = time.perf_counter()
        ok, info = SN.advance_region(r, int(secs / 0.025), sources=[(cells, 300.0)], max_nodes=mx)
        out[f"tier-1 jump, 300 W room, {secs:g} s, max_nodes {mx} (s)"] = time.perf_counter() - t
        out["  of which nodes / B / ok"] = f"{info.get('nodes')} / {info.get('B')} / {ok}"
    if "--json" in sys.argv:
        print(json.dumps(out, indent=1, default=str))
        return
    width = max(len(k) for k in out)
    for k, v in out.items():
        print(f"{k:<{width}}  {v if isinstance(v, str) else format(v, '.4g')}")


if __name__ == "__main__":
    main()
