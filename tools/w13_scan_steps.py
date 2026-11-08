"""W13 T11: which golden cases have a ledge or rubble a stepping body could use?

T11 turns on a step range (up 0.30 m, down 0.30 m; the planner's walk is flat
today: a kerb is a wall). A case whose people stand near a ledge of 5-30 cm, or
whose floor is rubble of that height, MOVES when it is turned on. This finds
them, read-only, over the 35 golden cases (and the 8 harvest seeds are among
them).

A STANDABLE COLUMN has a solid cell under its floor level and `head` clear cells
above it (1.7 m). Its floor is the highest such level. A LEDGE EDGE is a pair of
side-by-side columns (4-neighbours) whose floors differ by 5 cm to 30 cm. A
RUBBLE STEP is a ledge edge whose higher floor cell is rubble. NEAR A PERSON means
within 2 m (centre of the person's flesh, in columns).

The world is scanned after its first step, every 25th tick, and at the end of
the case (the rubble made by the case). Counts are the largest seen.

    python3 tools/w13_scan_steps.py [--tiny] [--only a,b] [--out results.json] [--note TEXT]
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.realpath(os.environ.get("MARK1_REPO", os.path.dirname(HERE)))
os.environ["PYTHONHASHSEED"] = "0"
os.environ["MARK1_REPO"] = REPO
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))
RESULTS = os.environ.get("W13_RESULTS", "/home/nisargparikh/.cache/mark1-build/w13/M/results.json")
KEY = "scan_steps"

import numpy as np                                                # noqa: E402

from src.voxel import sim                                          # noqa: E402
from src.voxel.materials import AIR, STONE, FLESH                         # noqa: E402

LO_M, HI_M, HEAD_M, NEAR_M = 0.05, 0.30, 1.70, 2.0


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



def floors(w, head):
    """Top-of-support z per column: the highest z with solid below and `head`
    clear cells above it, -1 for none."""
    solid = (w.mat != AIR) & (w.mat != FLESH)    # a body is not terrain
    nx, ny, nz = w.shape
    out = np.full((nx, ny), -1, np.int32)
    for z in range(1, nz):                    # (the top of the world counts as open)
        ok = solid[:, :, z - 1] & ~solid[:, :, z:min(z + head, nz)].any(axis=2)
        out[ok] = z
    return out


def people_columns(w):
    """(x, y) of the centre of each person's flesh, from `_own`."""
    pts = []
    for p in getattr(w, "persons", []):
        own = p.get("_own") or ()
        if own:
            a = np.array(list(own), np.float64)
            pts.append((float(a[:, 0].mean()), float(a[:, 1].mean())))
    return pts


def scan(w):
    dx = 0.1 * w.scale                                    # metres per voxel
    lo, hi = max(1, int(np.ceil(LO_M / dx - 1e-9))), int(np.floor(HI_M / dx + 1e-9))
    head = max(2, int(round(HEAD_M / dx)))
    h = floors(w, head)
    people = people_columns(w)
    near = int(round(NEAR_M / dx))
    edges = rub = n_near = n_rub_near = 0
    hist = {}
    for ex, ey in ((1, 0), (0, 1)):
        a = h[:h.shape[0] - ex, :h.shape[1] - ey]
        b = h[ex:, ey:]
        d = np.abs(a - b)
        both = (a >= 0) & (b >= 0) & (d >= lo) & (d <= hi)
        xs, ys = np.nonzero(both)
        edges += len(xs)
        for v in np.unique(d[both]):
            k = int(round(v * dx * 100))
            hist[k] = hist.get(k, 0) + int((d[both] == v).sum())
        if not len(xs):
            continue
        hi_is_a = a[xs, ys] > b[xs, ys]
        hx = np.where(hi_is_a, xs, xs + ex)
        hy = np.where(hi_is_a, ys, ys + ey)
        hz = np.maximum(a[xs, ys], b[xs, ys]) - 1         # the higher floor cell
        r = w.rubble[hx, hy, hz]
        rub += int(r.sum())
        m = np.zeros(len(xs), bool)                       # near ANY person
        for px, py in people:
            m |= (np.abs(xs - px) <= near) & (np.abs(ys - py) <= near)
        n_near += int(m.sum())
        n_rub_near += int((m & r).sum())
    return {"ledge_edges": edges, "ledge_edges_near_a_person": n_near, "rubble_steps": rub,
            "rubble_steps_near_a_person": n_rub_near, "persons": len(people),
            "rubble_cells": int(w.rubble.sum()), "heights_cm": dict(sorted(hist.items())),
            "shape": list(w.shape), "voxel_m": dx}


class Scanner:
    def __init__(self):
        self.case, self.out, self.last, self.first_seen = None, {}, None, set()

    def keep(self, w):
        row = scan(w)
        best = self.out.setdefault(self.case, {})
        for k, v in row.items():
            if isinstance(v, int) and not isinstance(v, bool):
                best[k] = max(best.get(k, 0), v)
            elif k == "heights_cm":
                for hk, hv in v.items():
                    cur = best.setdefault(k, {})
                    cur[hk] = max(cur.get(hk, 0), hv)
            else:
                best[k] = v

    def __enter__(self):
        self.o_step = sim.World.step
        me = self

        def step(w, *a, **k):
            r = me.o_step(w, *a, **k)
            me.last = w
            if id(w) not in me.first_seen or w.tick % 25 == 0:
                me.first_seen.add(id(w))
                me.keep(w)
            return r
        sim.World.step = step
        return self

    def __exit__(self, *exc):
        sim.World.step = self.o_step


def run(tiny=False, note="", only=None):
    t0 = time.perf_counter()
    if tiny:
        w = sim.World(16, 16, 16, voxel_cm=5)
        w.fill(0, 16, 0, 16, 0, 1, STONE)
        w.fill(8, 16, 0, 16, 1, 3, STONE)                  # a 10 cm ledge
        w.rubble[8:10, 4, 2] = True
        sc = Scanner()
        sc.case = "tiny"
        sc.keep(w)
        r = sc.out["tiny"]
        return {"status": "ok", "value": r["ledge_edges"], "units": "ledge edges in the tiny test world",
                "cases": {"tiny": r}, "cases_with_ledge_near_a_person": [], "cases_with_rubble_step": [],
                "seconds": round(time.perf_counter() - t0, 1), "load_note": load_note(note)}
    import golden as g
    names = only or list(g.ALL)
    errors = {}
    sc = Scanner()
    with sc:
        for n in names:
            sc.case, sc.first_seen, sc.last = n, set(), None
            try:
                g.run_case(n)
            except Exception as e:
                errors[n] = repr(e)[:160]
            if sc.last is not None:
                sc.keep(sc.last)
    cases = sc.out
    near = sorted(n for n, r in cases.items() if r.get("ledge_edges_near_a_person"))
    rub = sorted(n for n, r in cases.items() if r.get("rubble_steps"))
    anyl = sorted(n for n, r in cases.items() if r.get("ledge_edges"))
    return {"status": "ok" if not errors else "partial", "value": len(near),
            "units": "golden cases with a 5-30 cm ledge within 2 m of a person (candidates to MOVE when the step range is on)",
            "cases_with_ledge_near_a_person": near, "cases_with_rubble_step": rub,
            "cases_with_any_ledge": anyl, "ledge_range_m": [LO_M, HI_M], "head_m": HEAD_M, "near_m": NEAR_M,
            "cases": cases, "cases_run": len(names), "errors": errors,
            "seconds": round(time.perf_counter() - t0, 1), "load_note": load_note(note)}


def main(argv):
    tiny = "--tiny" in argv
    out = argv[argv.index("--out") + 1] if "--out" in argv else RESULTS
    note = argv[argv.index("--note") + 1] if "--note" in argv else ""
    only = argv[argv.index("--only") + 1].split(",") if "--only" in argv else None
    row = run(tiny, note, only)
    record(out, KEY + ("_tiny" if tiny else ""), row)
    print(json.dumps({k: v for k, v in row.items() if k != "cases"}, indent=1)[:4000])
    return row


if __name__ == "__main__":
    main(sys.argv[1:])
