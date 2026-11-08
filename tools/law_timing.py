"""PER-LAW TIMING: how many milliseconds a tick spends in each law (wave 13, lane B1).

Every C lane of wave 13 has a cost ceiling in ms per tick (docs/wave13-plan.md,
section 7) and reports the MEASURED number against it. This is the one tool that
measures it: it wraps every `_law_*` method of the World, and a few helpers and
seams, with a clock, steps a scene, and prints for each

    calls/tick   how often the law ran (a gate that skips it brings this under 1)
    mean ms      the time it took per tick, averaged over ALL timed ticks, the
                 ones where it was skipped counting as 0 (so mean x ticks is the
                 law's total cost, and the means of the top-level laws add up
                 to the tick)
    p99 ms       the 99th percentile of that per-tick time (the tick that hurts)

and, at the top, the same two numbers for the whole of `step()`.

    python3 tools/law_timing.py                       # the three default scenes
    python3 tools/law_timing.py --scenes forge,pond --ticks 80
    python3 tools/law_timing.py --repeat 3            # the median of three runs of each scene
    python3 tools/law_timing.py --world mymod:build   # any callable returning a World
    python3 tools/law_timing.py --json out.json       # the same numbers, for a lane's report
    python3 tools/law_timing.py --tiny                # a 16-cube world, for the smoke test

THE TIMES ARE INCLUSIVE. A law that calls another (`_law_burn` calls
`_burn_cells`) counts the callee in its own row too, so rows do not sum to 100%.
The rows marked `*` are helpers and seams (the derived fields, `_hot_window`,
`_heat_busy`, `_floor`...): inside the laws above them, shown so a lane can see
where a law's time goes. The first `--skip` ticks are not counted (numba
compiles and caches fill then).

TREE. The repo it measures is $MARK1_REPO (default: the repo this file is in),
as for tools/golden.py, so `stepcheck.sh`'s sandbox can be measured with
`MARK1_REPO=$WORK/<id>/mine`. The wrappers are put on the CLASSES for the length
of a run and taken off again in a `finally`; they only read the clock, so the
world they time is the world it would have been.

Wall-clock numbers on a shared machine move by 15 percent or more between runs:
compare two runs made one after the other, never a number from last week.
"""
import argparse
import contextlib
import importlib
import io
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.realpath(os.environ.get("MARK1_REPO", os.path.dirname(HERE)))
sys.path.insert(0, REPO)

import numpy as np                                                # noqa: E402

from src.voxel import scenes, sim                                  # noqa: E402

DEFAULT_SCENES = ("torch_pillar", "forge", "pond")
# helpers and seams: not `_law_*`, but where the time of the laws goes
HELPERS = ("_tick_start", "_tick_end", "_hot_window", "_heat_busy", "_unsettled", "_floor",
           "_env_radiation", "_top_sink", "_face_env", "_surfaces", "_surface_geometry",
           "heat_capacity", "porosity", "burning", "T", "_air_regions", "_mixing", "_pushed",
           "_burn_cells", "_law_skin_water", "_law_radiate")
SHOW_MIN_MS = 0.005                     # rows under this mean are left out of the table


class Clock:
    """The wrappers and what they have seen. `acc` is the current tick's
    {name: [seconds, calls]}; `rows` has one such dict per timed tick."""

    def __init__(self):
        self.acc = {}
        self.rows = []
        self.steps = []                 # seconds of each timed step()
        self.undo = []                  # (class, name, original) to put back
        self.ticks = 0
        self.skip = 0

    def wrap(self, cls, name):
        fn = cls.__dict__[name]
        me = self

        def timed(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                cell = me.acc.setdefault(name, [0.0, 0])
                cell[0] += time.perf_counter() - t0
                cell[1] += 1
        timed.__name__, timed.__doc__ = name, fn.__doc__
        setattr(cls, name, timed)
        self.undo.append((cls, name, fn))

    def wrap_step(self, cls):
        fn = cls.__dict__["step"]
        me = self

        def step(world, *a, **k):
            me.acc = {}
            t0 = time.perf_counter()
            try:
                return fn(world, *a, **k)
            finally:
                dt = time.perf_counter() - t0
                me.ticks += 1
                if me.ticks > me.skip:
                    me.steps.append(dt)
                    me.rows.append(me.acc)
        step.__name__, step.__doc__ = "step", fn.__doc__
        cls.step = step
        self.undo.append((cls, "step", fn))

    def restore(self):
        for cls, name, fn in reversed(self.undo):
            setattr(cls, name, fn)
        self.undo = []


def _definers(cls, names):
    """{name: the class in cls's MRO whose own dict defines it} for the plain
    methods among `names`."""
    out = {}
    for name in names:
        for k in cls.__mro__:
            v = k.__dict__.get(name)
            if v is not None:
                if callable(v) and not isinstance(v, (staticmethod, classmethod, property)):
                    out[name] = k
                break
    return out


@contextlib.contextmanager
def timed(world_cls=None, skip=10):
    """Put the clocks on `world_cls` (default sim.World) for a with-block."""
    cls = world_cls or sim.World
    clock = Clock()
    clock.skip = skip
    names = sorted({n for n in dir(cls) if n.startswith("_law_")} | set(HELPERS))
    try:
        for name, owner in _definers(cls, names).items():
            clock.wrap(owner, name)
        clock.wrap_step(_definers(cls, ["step"])["step"])
        yield clock
    finally:
        clock.restore()


def summarize(clock):
    """{"ticks", "step": {mean_ms, p99_ms}, "laws": {name: {calls_per_tick,
    mean_ms, p99_ms}}} over the timed ticks."""
    n = len(clock.steps)
    if n == 0:
        raise RuntimeError("no tick was timed (is --skip as large as --ticks?)")
    ms = np.array(clock.steps) * 1e3
    out = {"ticks": n, "step": {"mean_ms": float(ms.mean()), "p99_ms": float(np.percentile(ms, 99))},
           "laws": {}}
    names = sorted({k for row in clock.rows for k in row})
    for name in names:
        t = np.array([row.get(name, (0.0, 0))[0] for row in clock.rows]) * 1e3
        c = np.array([row.get(name, (0.0, 0))[1] for row in clock.rows])
        out["laws"][name] = {"calls_per_tick": float(c.mean()), "mean_ms": float(t.mean()),
                             "p99_ms": float(np.percentile(t, 99))}
    return out


def median_of(runs):
    """The result of several runs of the same scene as one: every number is
    the median over the runs (the rule is the median of 3, under the lock,
    because the machine is shared). `runs` is the count."""
    if len(runs) == 1:
        return runs[0]
    med = lambda xs: float(np.median(xs))
    out = {"ticks": runs[0]["ticks"], "runs": len(runs),
           "step": {k: med([r["step"][k] for r in runs]) for k in ("mean_ms", "p99_ms")}, "laws": {}}
    for name in sorted({n for r in runs for n in r["laws"]}):
        rows = [r["laws"].get(name, {"calls_per_tick": 0.0, "mean_ms": 0.0, "p99_ms": 0.0}) for r in runs]
        out["laws"][name] = {k: med([x[k] for x in rows]) for k in ("calls_per_tick", "mean_ms", "p99_ms")}
    return out


def table(title, res, show_all=False):
    """The result as text, the biggest means first."""
    runs = f", median of {res['runs']} runs" if res.get("runs", 1) > 1 else ""
    lines = [f"{title}  ({res['ticks']} ticks timed{runs})  step mean {res['step']['mean_ms']:.3f} ms, "
             f"p99 {res['step']['p99_ms']:.3f} ms",
             f"  {'law':<24}{'calls/tick':>11}{'mean ms':>10}{'p99 ms':>10}{'of step':>9}"]
    step = res["step"]["mean_ms"] or 1.0
    rows = sorted(res["laws"].items(), key=lambda kv: -kv[1]["mean_ms"])
    for name, r in rows:
        if r["mean_ms"] < SHOW_MIN_MS and not show_all:
            continue
        mark = " " if name.startswith("_law_") else "*"
        lines.append(f"{mark} {name:<24}{r['calls_per_tick']:>10.2f}{r['mean_ms']:>10.3f}"
                     f"{r['p99_ms']:>10.3f}{100.0 * r['mean_ms'] / step:>8.1f}%")
    return "\n".join(lines)


# ── what to time ─────────────────────────────────────────────────────────────
def run_scene(name, ticks, skip):
    """A named scene from scenes.SCENARIOS, as tools/golden.py plays it (frames
    are not written), timed over its steps."""
    saved_save = scenes._save
    scenes._save = lambda w, frames_dir, t: None
    try:
        with timed(sim.World, skip) as clock, tempfile.TemporaryDirectory() as d, \
                contextlib.redirect_stdout(io.StringIO()):
            scenes.SCENARIOS[name](d, ticks=ticks, every=10 ** 9)
        return summarize(clock)
    finally:
        scenes._save = saved_save


def run_world(spec, ticks, skip):
    """`module:function` returns a built World; step it `ticks` times."""
    mod, _, fn = spec.partition(":")
    w = getattr(importlib.import_module(mod), fn)()
    with timed(type(w), skip) as clock:
        for _ in range(ticks):
            w.step()
    return summarize(clock)


def tiny_world():
    """A 16x16x16 room with a hot block: for the smoke test, not for numbers."""
    w = sim.World(16, 16, 16, voxel_cm=10)
    w.fill(0, 16, 0, 16, 0, 1, sim.STONE)
    w.fill(6, 9, 6, 9, 1, 3, sim.WOOD)
    w.E[7, 7, 2] = 4000.0
    return w


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scenes", default=",".join(DEFAULT_SCENES),
                    help="comma list of names in src.voxel.scenes.SCENARIOS")
    ap.add_argument("--world", help="module:function that returns a built World")
    ap.add_argument("--ticks", type=int, default=60)
    ap.add_argument("--skip", type=int, default=10, help="ticks not timed (warm-up)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run each scene this many times and report the median (use 3)")
    ap.add_argument("--json", help="write the numbers to this file")
    ap.add_argument("--all", action="store_true", help="show rows under 0.005 ms too")
    ap.add_argument("--tiny", action="store_true", help="a 16-cube world, 12 ticks (smoke test)")
    a = ap.parse_args(argv)
    results = {}
    if a.tiny:
        sim_ticks, skip = 12, 4
        with timed(sim.World, skip) as clock:
            w = tiny_world()
            for _ in range(sim_ticks):
                w.step()
        results["tiny"] = summarize(clock)
    elif a.world:
        results[a.world] = median_of([run_world(a.world, a.ticks, a.skip) for _ in range(a.repeat)])
    else:
        for name in [s for s in a.scenes.split(",") if s]:
            if name not in scenes.SCENARIOS:
                ap.error(f"no scene {name!r}; known: {', '.join(sorted(scenes.SCENARIOS))}")
            results[name] = median_of([run_scene(name, a.ticks, a.skip) for _ in range(a.repeat)])
    for name, res in results.items():
        print(table(name, res, a.all))
        print()
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"repo": REPO, "results": results}, f, indent=1, sort_keys=True)
    return results


if __name__ == "__main__":
    main()
