"""W4a SURVEY (read only): what would the members law do to the golden scenes?

Runs `matbeam.extract_members` + `accumulate_loads` + `analyse` on the FINAL
arrays of every golden case (tools/golden.py) and prints every member whose
utilisation u is over 0.5: cell, material, kind, mode, u. Nothing is wired and
no World file is touched; the case is run as golden.py runs it, the world is
read at the end. The table is the evidence for or against "expected SAME" when
the veto (W4b) is wired: it is wired only if no golden member is above 0.8.

    tools/members_survey.py                      # every case, members over 0.5
    tools/members_survey.py --only duel,forge    # some cases
    tools/members_survey.py --quick              # skip the SLOW cases
    tools/members_survey.py --over 0.3           # another threshold
    tools/members_survey.py --no-tension         # treat dry masonry as no-tension

Wrap a real run in the shared lock and the memory cap:

    flock -w 600 $LOCK $CAP python3 tools/members_survey.py

What it measures: STATIC loads only (dead weight; grip-carried cells are left
off the lattice path, a person's flesh and anything not structural is weight
that passes down and is never checked). No walking extra, no landing. A scene
that ends mid-collapse shows what is left standing. The record is matrec's when
it exists, else the stand-in of the spec's 12 rows (the line says which).
"""
import contextlib
import importlib.util
import io
import os
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.realpath(os.environ.get("MARK1_REPO", os.path.dirname(HERE)))
sys.path.insert(0, REPO)

MATNAME = ("air", "wood", "stone", "iron", "flesh", "ash", "glass", "leaf", "char", "tin",
           "lead", "rope")

# The spec's 12 rows (T9-material 4.1), used when matrec is not importable.
# (rho, E, nu, sy, st, sc, sf, tau, ductility, derate)
_IRON_D = ((20, 1, 1), (100, 1, 1), (200, 1, .9), (300, 1, .8), (400, 1, .7), (500, .78, .6),
           (600, .47, .31), (700, .23, .13), (800, .11, .09), (900, .06, .0675),
           (1000, .04, .045), (1100, .02, .0225), (1200, 0, 0))
_STONE_D = ((20, 1, 1), (100, 1, 1), (200, .95, .95), (300, .85, .85), (400, .75, .75),
            (500, .6, .6), (600, .45, .45), (700, .3, .3), (800, .15, .15), (900, .08, .08),
            (1000, .04, .04))
_STANDIN = [
    (1.2, 0, 0, None, 0, 0, None, 0, 0, None),
    (600, 1.1e10, .35, None, 6e7, 4e7, 7e7, 8e6, .01, ((20, 1, 1), (100, .75, .75), (200, .5, .5),
                                                      (300, .2, .2), (350, 0, 0))),
    (2700, 5e10, .25, None, 1e7, 1.5e8, 1.5e7, 2e7, .001, _STONE_D),
    (7800, 2e11, .29, 2.5e8, 3.5e8, 2.5e8, None, 1.45e8, .2, _IRON_D),
    (1000, 1e6, .45, None, 1e6, 1e6, 1e6, 3e5, .5, None),
    (150, 1e8, .3, None, 0, 1e4, 0, 0, 0, None),
    (2500, 7e10, .22, None, 5e7, 1e9, 5e7, 3e7, 0, ((20, 1, 1), (400, 1, 1), (500, .6, .6),
                                                     (550, .3, .3), (720, .02, .02))),
    (200, 2e9, .3, None, 2e7, 0, 0, 0, .1, None),
    (300, 1e9, .3, None, 1e6, 2e6, 1e6, 1e6, .005, None),
    (7300, 5e10, .33, 1.1e7, 2e7, 1.1e7, None, 1.2e7, .4, ((20, 1, 1), (232, 0, 0))),
    (11340, 1.6e10, .44, 5.5e6, 1.2e7, 5.5e6, None, 6e6, .5, ((20, 1, 1), (327, 0, 0))),
    (885, 9.4e8, .4, None, 9.4e7, 0, 0, 0, .15, None),
]


def _load_golden():
    spec = importlib.util.spec_from_file_location("golden_for_survey",
                                                  os.path.join(REPO, "tools", "golden.py"))
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)          # re-executes this script once to pin PYTHONHASHSEED
    return g


class _Null:
    """The recorder golden.py calls from every step; the survey records nothing."""

    def note(self, *a, **k):
        return None

    render = tap = note


def _record(B):
    """matrec's record when it can be built, else the stand-in rows."""
    try:
        return B.rec_from_matrec(), "matrec"
    except Exception as e:                                   # not there, or a row it cannot read
        rows = [dict(rho=r[0], E=r[1], nu=r[2], sy=r[3], st=r[4], sc=r[5], sf=r[6], tau=r[7],
                     ductility=r[8], derate=r[9]) for r in _STANDIN]
        return B.rec_from_rows(rows), f"stand-in ({type(e).__name__})"


def survey_world(w, rec, B, no_tension=False):
    """Members of one finished World: (Members, Loads, Analysis)."""
    import numpy as np
    from src.voxel import sim
    solid = w.mat != sim.AIR
    dens = np.array(sim._DENS_ARR, float)                            # as long as the pad: a cell's id indexes it
    dens[:sim.NMAT] = [sim.SOLID[k][0] for k in range(sim.NMAT)]      # the old twelve, the dict's own floats
    fill = np.clip(w.smass / np.maximum(dens[w.mat] * w.vox_l, 1e-9), 0.0, 1.0)
    h = w.scale * 0.1
    try:
        grip = set(w._grip_cells()) | set(w._borne_cells())
    except Exception:
        grip = set()
    owned = w.mat == sim.FLESH
    m = B.extract_members(solid, fill, w.mat, w.eid, None, w.T(), w.rubble, owned, grip,
                          w._is_thing(w.eid), None, w.moving, h, rec=rec)
    gm = B._mask(grip, solid.shape)
    W = np.where(solid, B.weights_N(w.smass, gm), 0.0)
    loads = B.accumulate_loads(m, W)
    return m, loads, B.analyse(m, rec, loads, no_tension=no_tension)


def main(argv):
    only = None
    over = 0.5
    quick = "--quick" in argv
    no_tension = "--no-tension" in argv
    if "--only" in argv:
        only = argv[argv.index("--only") + 1].split(",")
    if "--over" in argv:
        over = float(argv[argv.index("--over") + 1])
    g = _load_golden()
    from src.voxel import matbeam as B
    rec, src = _record(B)
    names = only or (list(g.CASES) if quick else list(g.ALL))
    worlds = []
    orig = g.Configured.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        worlds.append(self)

    g.Configured.__init__ = init
    g.REC = _Null()
    print(f"members survey: record from {src}; members over u = {over}; "
          f"{len(names)} cases; repo {REPO}", flush=True)
    table = []
    for name in names:
        worlds.clear()
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                g.ALL[name]()
        except Exception as e:
            print(f"  {name:14s} RAISED {type(e).__name__}: {e}", flush=True)
            continue
        t1 = time.perf_counter()
        seen = []
        for w in worlds:
            if any(w is v for v in seen):
                continue
            seen.append(w)
        for wi, w in enumerate(seen):
            t2 = time.perf_counter()
            try:
                m, loads, res = survey_world(w, rec, B, no_tension)
            except Exception as e:
                print(f"  {name:14s} world {wi}: survey RAISED {type(e).__name__}: {e}", flush=True)
                continue
            ms = (time.perf_counter() - t2) * 1e3
            top = max((r["u"] for r in res.rows), default=0.0)
            hot = res.over(over)
            bal = abs(float(loads.ground.sum() + loads.edge.sum() + loads.lost.sum()
                            - loads.total.sum()))
            print(f"  {name:14s} world {wi} tick {w.tick:4d}  {len(m.runs):3d} runs "
                  f"{len(m.columns):3d} columns  max u {top:7.3f}  over {over}: {len(hot):3d}  "
                  f"lost {loads.lost.sum():8.1f} N  balance {bal:.1e}  "
                  f"({t1 - t0:5.1f} s run, {ms:6.1f} ms survey)", flush=True)
            for r in hot:
                table.append((name, wi, r))
    print()
    print(f"MEMBERS WITH u > {over}  ({len(table)} rows)")
    print(f"  {'case':14s} {'w':>2s} {'cell':>14s} {'material':8s} {'kind':16s} {'mode':10s} "
          f"{'u':>8s} {'u_sus':>8s} {'cells':>5s}")
    for name, wi, r in table:
        cell = "(%d,%d,%d)" % r["cell"]
        print(f"  {name:14s} {wi:2d} {cell:>14s} {MATNAME[r['mat']]:8s} {r['kind']:16s} "
              f"{B.MODE_NAME[r['mode']]:10s} {r['u']:8.3f} {r['u_sus']:8.3f} {r['n']:5d}")
    if not table:
        print("  (none)")
    worst = max((r["u"] for _, _, r in table), default=0.0)
    print(f"\nlargest utilisation over the table: {worst:.3f}; "
          f"{'ABOVE' if worst > 0.8 else 'not above'} the W4b gate of 0.8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
