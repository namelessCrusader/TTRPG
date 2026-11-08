"""W13 M3: where does a warm WALKING body lose or gain energy?

A warm (34 C) body that walks drifts the world's energy by a few joules over a
few hundred ticks; a warm body that stands still drifts by almost nothing.
This bisects that with a float64 ledger. Nothing here edits the sim: it only
wraps calls and reads.

  1. Controls: the same floor, the same warm body, stand / walk / cold walk.
     Drift of the float64 ledger every 50 ticks, in joules and watts.
  2. Function level: `sys.setprofile` measures the ledger at every call and
     return of a repo function (to a depth limit) and charges the change to the
     function that was running (EXCLUSIVE of its callees). The top movers are
     the suspects; the named candidates (`_walk`, `_lift`, `_put`, the pose cell
     moves, support.py's shard split) are listed with their numbers.
  3. Line level: `sys.settrace` on the top movers charges the change to the
     source line that was running (a line that calls something is charged for
     what the call did).

    python3 tools/w13_walk_leak.py [--tiny] [--out results.json] [--note TEXT]

The ledger is `ledger()`: the same terms as `World.total_energy`, summed in
float64 (the sim's own total_energy sums a float32 array, whose rounding is the
size of the drift being hunted).
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.realpath(os.environ.get("MARK1_REPO", os.path.dirname(HERE)))
sys.path.insert(0, REPO)
RESULTS = os.environ.get("W13_RESULTS", "/home/nisargparikh/.cache/mark1-build/w13/M/results.json")
KEY = "walk_leak"

import numpy as np                                                # noqa: E402

from src.voxel.sim import World, FLESH, STONE, TICK_S, AMBIENT     # noqa: E402
from src.voxel.scenes import _person, _Wants                       # noqa: E402

SRC = os.path.join(REPO, "src", "voxel") + os.sep


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


# ── the float64 ledger ────────────────────────────────────────────────────────
def ledger(w):
    """Every joule, in float64: cells at rest, bodies in flight, particles, what
    went past the edge, what was shed (the terms of World.total_energy), and the
    matter a `_lift` holds in its hand until the `_put` that sets it down (so a
    lift and its put do not swing the ledger; what a put OVERWRITES shows)."""
    return (float(w.E[~w.moving].sum(dtype=np.float64))
            + sum(float(b["Es"].sum(dtype=np.float64)) for b in w.bodies)
            + float(w.matter.energy()) + float(w.gone.get("E", 0.0)) + float(w.shed)
            + float(getattr(w, "_w13_hand", 0.0)))


class in_hand:
    """Context manager: count the joules a `_lift` holds until its `_put`. Also
    records, per call site, the joules a `_put` overwrote (the energy of the
    cells it set matter down on, which a put destroys without a word)."""

    def __init__(self, w):
        self.w, self.cls = w, type(w)
        self.over = {}                        # "file:line in fn" -> [J, calls]

    def __enter__(self):
        from src.voxel.bodies import CARRIED
        iE = CARRIED.index("E")
        cls, w = self.cls, self.w
        self.o_lift, self.o_put = cls._lift, cls._put
        o_lift, o_put, over = self.o_lift, self.o_put, self.over
        w._w13_hand = 0.0

        def lift(self_, idx):
            vals = o_lift(self_, idx)
            self_._w13_hand += float(np.sum(vals[iE], dtype=np.float64))
            return vals

        def put(self_, idx, vals):
            held = float(np.sum(vals[iE], dtype=np.float64))
            was = float(np.sum(self_.E[idx], dtype=np.float64))
            self_._w13_hand -= held
            f = sys._getframe(1)
            key = f"{f.f_code.co_filename.replace(REPO + os.sep, '')}:{f.f_lineno} in {f.f_code.co_name}"
            row = over.setdefault(key, [0.0, 0])
            row[0] -= was                      # energy that was there and is gone
            row[1] += 1
            return o_put(self_, idx, vals)
        cls._lift, cls._put = lift, put
        return self

    def __exit__(self, *exc):
        self.cls._lift, self.cls._put = self.o_lift, self.o_put

    def held(self):
        return float(getattr(self.w, "_w13_hand", 0.0))


# ── worlds ────────────────────────────────────────────────────────────────────
def warm_flesh(w, t_c=34.0):
    C = w.heat_capacity()
    f = (w.mat == FLESH)
    w.E[f] = (np.float32(t_c) - AMBIENT) * C[f]


def floor_world(walk, warm, metres=6.0, tiny=False):
    """A stone floor with a door at the far end and one person on it."""
    if tiny:
        w = World(16, 16, 16, voxel_cm=5)
        w.fill(0, 16, 0, 16, 0, 1, STONE)
        w.fill(3, 5, 6, 8, 1, 5, FLESH)               # a small block that can be walked
        for x, y, z in np.argwhere(w.mat == FLESH):
            w.smass[x, y, z] = 1000.0
        if warm:
            warm_flesh(w)
        return w, None
    nx, ny = int(metres / 0.05) + 24, 20
    w = World(nx, ny, 40, voxel_cm=5)
    w.fill(0, nx, 0, ny, 0, 1, STONE)
    w.exits = [(nx - 12, ny // 2)]
    p = _person(w, 12, ny // 2)
    p["name"] = "walker"
    p["known"][:] = True
    w.policy = _Wants(each={"walker": {"legs": "go(the door"}} if walk else {})
    if warm:
        warm_flesh(w)
    return w, p


def advance(w, p, tiny, walk):
    """One tick; in the tiny world the 'walker' is pushed by `_walk` itself."""
    if tiny:
        if walk:
            cells = np.argwhere(w.mat == FLESH)
            w._walk(cells, 1 if w.tick % 4 < 2 else -1, 0)
    w.step()
    if p is not None:
        p["events"].clear()


# ── control runs ──────────────────────────────────────────────────────────────
class no_turn:
    """Context manager: a body that never squares up (so `_turn` never runs)."""

    def __init__(self, on, w):
        self.on, self.cls = on, type(w)

    def __enter__(self):
        self.o = getattr(self.cls, "_square_up", None)
        if self.on and self.o is not None:
            self.cls._square_up = lambda self_, p: False

    def __exit__(self, *exc):
        if self.on and self.o is not None:
            self.cls._square_up = self.o


def drift_run(walk, warm, ticks, every, tiny, e64=False, noturn=False):
    """Float64 ledger drift of the world every `every` ticks. `e64`: hold E in
    float64. `noturn`: the body never squares up (the `_turn` overwrite is out)."""
    w, p = floor_world(walk, warm, tiny=tiny)
    if e64:
        w.E = w.E.astype(np.float64)
    e0, x0 = ledger(w), np.argwhere(w.mat == FLESH)[:, 0].mean()
    pts = []
    with no_turn(noturn, w):
        for t in range(ticks):
            advance(w, p, tiny, walk)
            if (t + 1) % every == 0:
                pts.append(round(ledger(w) - e0, 4))
    own = np.argwhere(w.mat == FLESH)
    return {"drift_J_every": pts, "every_ticks": every,
            "metres_moved": round(float((own[:, 0].mean() - x0) * 0.05), 2) if len(own) else None,
            "watts": round(pts[-1] / (ticks * TICK_S), 4) if pts else None}


# ── function-level attribution ────────────────────────────────────────────────
class Attribution:
    """Charge each change of the ledger to the repo function running at the time
    (exclusive of the functions it called), down to `depth` calls below step."""

    def __init__(self, w, depth=4):
        self.w, self.depth = w, depth
        self.stack = []                       # code objects of tracked frames
        self.last = 0.0
        self.excl = {}                        # code -> [delta, calls]
        self.deep = 0                         # python repo frames below step

    def _tracked(self, code):
        return code.co_filename.startswith(SRC)

    def _charge(self):
        now = ledger(self.w)
        if self.stack:
            row = self.excl.setdefault(self.stack[-1], [0.0, 0])
            row[0] += now - self.last
        self.last = now

    def __call__(self, frame, event, arg):
        if event not in ("call", "return"):
            return
        code = frame.f_code
        if not self._tracked(code):
            return
        if event == "call":
            self.deep += 1
            if self.deep <= self.depth:
                self._charge()
                self.stack.append(code)
                self.excl.setdefault(code, [0.0, 0])[1] += 1
        else:
            if self.deep <= self.depth:
                self._charge()
                self.stack.pop()
            self.deep -= 1

    def run(self, w_step):
        self.last = ledger(self.w)
        self.stack, self.deep = [], 0
        sys.setprofile(self)
        try:
            w_step()
        finally:
            sys.setprofile(None)


def short(code):
    return f"{code.co_filename.replace(REPO + os.sep, '')}:{code.co_firstlineno} {code.co_name}"


def attribute(walk, ticks, depth, tiny):
    w, p = floor_world(walk, True, tiny=tiny)
    att = Attribution(w, depth)
    t0 = time.perf_counter()
    with in_hand(w) as hand:
        for _ in range(ticks):
            att.run(lambda: advance(w, p, tiny, walk))
        held, over = hand.held(), hand.over
    rows = [{"where": short(c), "delta_J": d, "calls": n} for c, (d, n) in att.excl.items()]
    rows.sort(key=lambda r: -abs(r["delta_J"]))
    return rows, att, w, p, time.perf_counter() - t0, held, over


# ── line-level trace of the suspects ──────────────────────────────────────────
def cancel_pairs(steps, tol=1e-6):
    """Drop pairs of ledger steps that undo each other (a loss written down one
    line and booked on the next: `E -= loss` then `shed += ...`). What is left
    is what a function really changed. steps = [(line, delta)]."""
    left = [list(x) for x in steps if x[1] != 0.0]
    i = 0
    while i < len(left):
        for j in range(i + 1, len(left)):
            if left[i][1] * left[j][1] < 0 and abs(left[i][1] + left[j][1]) <= tol + 1e-9 * abs(left[i][1]):
                del left[j]
                del left[i]
                i -= 1
                break
        i += 1
    return left


def settle_lines(walk, ticks, suspects, tiny):
    """Run `ticks` ticks with a line tracer on the `suspects` (code objects).

    A line event fires BEFORE its line runs, so the ledger read at an event is
    the result of the line before it: every event gives (line just run, change
    it made). Steps that undo each other inside one call are cancelled
    (`cancel_pairs`) and what is left is charged to its line. A line that calls
    a repo function is charged for what that function did; `calls[(code, line)]`
    names such callees so the caller can look inside them next.
    Returns (rows, calls, held): rows = {(code, line): [delta J, hits]}, held =
    joules lifted and never put down."""
    w, p = floor_world(walk, True, tiny=tiny)
    codes = set(suspects)
    rows, calls, live = {}, {}, {}

    def local(frame, event, arg):
        rec = live[id(frame)]
        if event == "call":
            return None
        now = ledger(w)
        rec["steps"].append((rec["prev"], now - rec["last"]))
        rec["last"], rec["prev"] = now, frame.f_lineno
        if event == "return":
            for line, d in cancel_pairs(rec["steps"]):
                row = rows.setdefault((frame.f_code, line), [0.0, 0])
                row[0] += d
                row[1] += 1
            live.pop(id(frame), None)
        return local

    def glob(frame, event, arg):
        back = frame.f_back
        if back is not None and back.f_code in codes and frame.f_code.co_filename.startswith(SRC):
            ck = calls.setdefault((back.f_code, back.f_lineno), {})
            ck[frame.f_code] = ck.get(frame.f_code, 0) + 1
        if frame.f_code in codes:
            now = ledger(w)
            live[id(frame)] = {"steps": [], "last": now, "prev": frame.f_code.co_firstlineno}
            return local
        return None

    with in_hand(w) as hand:
        for _ in range(ticks):
            sys.settrace(glob)
            try:
                advance(w, p, tiny, walk)
            finally:
                sys.settrace(None)
        held = hand.held()
    return rows, calls, held


def line_trace(walk, ticks, suspects, tiny, rounds=4):
    """`settle_lines`, then look inside the callees of the worst line, again,
    up to `rounds` times. Returns the settled lines of the last round that found
    anything, worst first, with the chain of functions that led to them."""
    codes = list(suspects)
    best, chain = [], []
    for _ in range(rounds):
        rows, calls, _held = settle_lines(walk, ticks, codes, tiny)
        lines = [{"where": f"{c.co_filename.replace(REPO + os.sep, '')}:{ln} in {c.co_name}",
                  "delta_J": d, "hits": n, "_key": (c, ln)} for (c, ln), (d, n) in rows.items()]
        lines.sort(key=lambda r: -abs(r["delta_J"]))
        lines = [r for r in lines if abs(r['delta_J']) > 1e-6]
        if not lines:
            break
        best = lines
        callees = calls.get(lines[0]["_key"], {})
        callees = [c for c in callees if c not in codes]
        if not callees:
            break
        chain.append(lines[0]["where"])
        codes += callees
    for r in best:
        r.pop("_key", None)
    return best, chain


def run(tiny=False, note=""):
    t_all = time.perf_counter()
    ticks, every = (6, 3) if tiny else (300, 50)
    controls = {}
    for name, walk, warm in (("stand_warm", False, True), ("walk_warm", True, True),
                             ("walk_cold", True, False)):
        controls[name] = drift_run(walk, warm, ticks, every, tiny)
    controls["stand_warm_E64"] = drift_run(False, True, ticks, every, tiny, e64=True)
    controls["walk_warm_E64"] = drift_run(True, True, ticks, every, tiny, e64=True)
    controls["walk_warm_noturn"] = drift_run(True, True, ticks, every, tiny, noturn=True)
    controls["walk_warm_E64_noturn"] = drift_run(True, True, ticks, every, tiny, e64=True, noturn=True)
    k = 4 if tiny else 100
    depth = 3 if tiny else 4
    rows, att, w, p, secs, held, over = attribute(True, k, depth, tiny)
    stand_rows, *_ = attribute(False, k, depth, tiny)
    stand = {r["where"]: r["delta_J"] for r in stand_rows}
    for r in rows:
        r["stand_delta_J"] = stand.get(r["where"], 0.0)
        r["extra_over_stand_J"] = r["delta_J"] - r["stand_delta_J"]
    top = sorted(rows, key=lambda r: -abs(r["extra_over_stand_J"]))[:8]
    by_code = {short(c): c for c in att.excl}
    suspects = [by_code[r["where"]] for r in top[:4]
                if r["where"] in by_code and abs(r["extra_over_stand_J"]) > 1e-6]
    lines, chain = line_trace(True, k, suspects, tiny) if suspects else ([], [])
    cand = {}                       # the candidates named in the T7 spec, with what was measured
    for r in rows:
        nm = r["where"].split()[-1]
        if nm in ("_walk", "_lift", "_put", "_turn", "_square_up", "_repose") \
                or "pose.py" in r["where"] or "support.py" in r["where"]:
            cand[r["where"]] = {"delta_J": r["delta_J"], "calls": r["calls"]}
    puts = sorted(({"where": k_, "overwritten_J": v[0], "calls": v[1]} for k_, v in over.items()),
                  key=lambda r: -abs(r["overwritten_J"]))
    walk_w = controls["walk_warm"]["watts"]
    stand_w = controls["stand_warm"]["watts"]
    excess = (walk_w - stand_w) if (walk_w is not None and stand_w is not None) else None
    big = [r for r in lines if abs(r["delta_J"]) > (1e-3 if tiny else 0.05)]
    row = {"status": "ok", "units": "J (float64 ledger) and W",
           "value": excess, "value_meaning": "walking minus standing drift, W (negative = loses energy)",
           "controls": controls,
           "found": bool(big) or bool(puts and abs(puts[0]["overwritten_J"]) > (1e-3 if tiny else 0.05)),
           "causes": [{"where": r["where"], "delta_J": r["delta_J"], "hits": r["hits"],
                       "per_tick_J": r["delta_J"] / k} for r in big[:4]],
           "put_overwrites": puts[:6], "lifted_never_put_J": held,
           "top_functions": top, "top_lines": lines[:10], "descent": chain,
           "candidates": cand, "attribution_ticks": k,
           "seconds": round(time.perf_counter() - t_all, 1), "load_note": load_note(note)}
    row["fix_text"] = fix_text(row)
    return row


# What each known culprit means, and what base19 should do about it. Keyed by
# the function name of a settled line; text for a line not listed is generic.
KNOWN_FIXES = {
    "_turn": ("`_turn` (pose.py) sets the turned body down with `self._put(want, self._lift(own))`. "
              "The cells it turns INTO that were not its own held air, and `_put` writes E over them: "
              "the warm air that was there is destroyed (a one-off per turn, about 1.2 J for a "
              "34 C body at 5 cm). FIX: do what `_walk` does. Lift the air of the destination cells "
              "that are not the body's own, lift the body, put the body on `want`, then put the air "
              "into the cells the body vacated (the two sets are the same size for a permutation). "
              "TEST 14b: wrap `_put` and assert that the E of every cell it sets matter on "
              "is zero or was lifted in the same call."),
    "_law_conduct": ("`_law_conduct` (heat.py, the line `E += qs + qc`) adds in float32. A warm flesh "
                     "cell holds about 6000 J, so one float32 step is about 5e-4 J, and heat that "
                     "leaves it as a few 1e-3 J per tick is rounded, with a bias that is stronger "
                     "while the body moves (about 8e-3 J per tick, 0.33 W, against 5e-4 J standing). "
                     "FIX: hold E in float64 for a world that has warm bodies (T7 option P1, the "
                     "`E_dtype` setting); measured here: with E in float64 AND the "
                     "body never turning, the walking drift is exactly 0 (control walk_warm_E64_noturn)."),
    "_put": ("`_put` (bodies.py) overwrites the E of the cells it sets matter on. See the caller "
             "listed with it: any site that puts matter onto cells it did not lift must lift "
             "that cells' contents first and put them somewhere (as `_walk` does)."),
}


def fix_text(row):
    if not row["found"]:
        return ("not found: nothing changed the float64 ledger by more than the noise floor over the "
                "attribution ticks while the body walked more than it did standing. If the control "
                "drift is still non-zero, re-run on base19 with more ticks.")
    seen, parts = set(), []
    names = [c["where"].split(" in ")[-1] for c in row["causes"]] \
        + [p["where"].split(" in ")[-1] for p in row["put_overwrites"][:1]]
    for nm in names:
        if nm in KNOWN_FIXES and nm not in seen:
            seen.add(nm)
            parts.append(KNOWN_FIXES[nm])
    for c in row["causes"][:4]:
        parts.insert(0, f"{c['where']}: {c['delta_J']:+.4f} J over {row['attribution_ticks']} ticks "
                        f"({c['per_tick_J']:+.5f} J per tick).")
    return " ".join(parts) or "see causes"


def main(argv):
    tiny = "--tiny" in argv
    out = argv[argv.index("--out") + 1] if "--out" in argv else RESULTS
    note = argv[argv.index("--note") + 1] if "--note" in argv else ""
    row = run(tiny, note)
    record(out, KEY + ("_tiny" if tiny else ""), row)
    print(json.dumps(row, indent=1)[:6000])
    return row


if __name__ == "__main__":
    main(sys.argv[1:])
