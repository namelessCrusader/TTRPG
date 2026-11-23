"""STAIRS TRACE HARNESS (wave 15, S2 item 4a, step 0). A script, not a test.

    python3 tools/stairs_trace.py [--steps 12] [--riser 4] [--tread 6]
                                  [--ticks 900] [--out trace.jsonl] [--quiet]

Walks the stock person up a flight (scenes_stairs.stair_flight) the way
tests/test_stairs_on.py does (`_stride` toward a goal on the landing, then
`step()`), and writes ONE JSON LINE PER TICK with:

    tick, zlo, com, xcom, hull (the firm contacts' convex hull), n_firm,
    footprint (full body shadow: cell count, width x, width y) against the
    STANCE footprint (sole columns only), the planned path head, _stride debt,
    the number of free bodies, and the events of that tick:
      step_to_k   (sx, sy, kmax) -> (dz, k) or None, and when None, WHICH
                  condition refused each candidate (world / blocked / support /
                  balance), re-derived here from the world, independently of
                  routing.step_clear;
      overbalanced  calls on the real body and on ghosts, with the answer;
      reflex      _balance_reflex over, plan, return;  step_target answers;
      plan        a new path (head), and whether the SAME route exists with
                  the stance footprint in place of the full one (H1 test).

The SUMMARY at the end scores the five candidate mechanisms named in the spec
(S2_bodies, 4.1) from counts, and prints the first tick where the climb
stopped going up. It changes nothing in the sim (it wraps methods only); it
costs nothing when it is not run. No fix is made here: the report names the
cause first.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.voxel import routing, sim                                  # noqa: E402
from src.voxel.materials import AIR, PUSH_THROUGH, _DENS_ARR       # noqa: E402
from src.voxel.pathlaw import WADE_FRAC, _candidates               # noqa: E402
from src.voxel.scenes_stairs import stair_flight                    # noqa: E402

EV = []                      # the events of the tick being traced
CALLER = ["will"]            # who is walking: "harness" (our own _stride) or "will" (the person's mind in step())
COUNT = collections.Counter()


def _cells(w, p):
    comp, sl = w._person_cells(p)
    return w._claim_cells(p, comp, sl)


def stance_footprint(cells):
    """The columns of the soles only (the lowest layer): what the body stands
    on, without the arms, head or trunk overhang. Offsets from the centre
    column, as routing.footprint gives."""
    c = np.asarray(cells)
    cx = int(round(float(c[:, 0].mean())))
    cy = int(round(float(c[:, 1].mean())))
    sole = c[c[:, 2] == c[:, 2].min()]
    offs = sorted({(int(a) - cx, int(b) - cy) for a, b in sole[:, :2]} - {(0, 0)})
    return (np.array([o[0] for o in offs], np.int64),
            np.array([o[1] for o in offs], np.int64))


def explain_step(w, p, cells, sx, sy, kmax):
    """For every candidate (dz, k), the FIRST condition that refuses it
    ('world', 'blocked', 'support', 'balance') or 'ok'. Re-derived from the
    world cell by cell with plain python and numpy; shares nothing with
    routing.step_clear."""
    up, dn = w._step_range(p)
    cells = np.asarray(cells, np.int64)
    nx, ny, nz = w.shape
    own = set(map(tuple, cells.tolist()))
    out = []
    ghost = None
    for dz, k in _candidates(up, dn, kmax):
        tgt = cells + np.array([sx * k, sy * k, dz])
        if ((tgt[:, 0] < 0) | (tgt[:, 0] >= nx) | (tgt[:, 1] < 0)
                | (tgt[:, 1] >= ny) | (tgt[:, 2] < 0) | (tgt[:, 2] >= nz)).any():
            out.append(((dz, k), "world"))
            continue
        bad = False
        for c in map(tuple, tgt.tolist()):
            if c in own:
                continue
            packed = w.smass[c] / max(_DENS_ARR[w.mat[c]] * w.vox_l, 1e-9)
            if w.moving[c] or packed >= PUSH_THROUGH \
                    or w.fvol[c] > WADE_FRAC * w.cap:
                bad = True
                break
        if bad:
            out.append(((dz, k), "blocked"))
            continue
        zmin = int(tgt[:, 2].min())
        if zmin > 0:
            low = tgt[tgt[:, 2] == zmin]
            if not any(w.mat[x, y, zmin - 1] != AIR and (x, y, zmin - 1) not in own
                       for x, y, _z in low.tolist()):
                out.append(((dz, k), "support"))
                continue
        if dz and kmax > 1:
            if ghost is None:
                ghost = dict(p, held=None, dragging=None, _v=None,
                             _ghost_from=cells,
                             _ghost_kg=w.smass[tuple(cells.T)].copy())
            over = w._overbalanced(ghost, tgt)
            if over is not None:
                out.append(((dz, k), "balance"))
                continue
        out.append(((dz, k), "ok"))
    return out


def sole_support(w, cells, sx, sy, dz, k):
    """The share of the sole columns that stand over something (not air, not
    the body's own cell) after a step of (sx*k, sy*k, dz). A step is taken when
    ANY is supported (`_step_to_k`), so a share below 1 is a foot over a lip."""
    cells = np.asarray(cells, np.int64)
    own = set(map(tuple, cells.tolist()))
    tgt = cells + np.array([sx * k, sy * k, dz])
    zmin = int(tgt[:, 2].min())
    low = tgt[tgt[:, 2] == zmin]
    if zmin == 0:
        return 1.0
    ok = [w.mat[x, y, zmin - 1] != AIR and (x, y, zmin - 1) not in own
          for x, y, _z in low.tolist()]
    return round(sum(ok) / max(len(ok), 1), 2)


def install():
    """Wrap the World methods the trace reads. Returns the originals."""
    W = sim.World
    orig = {n: getattr(W, n) for n in (
        "_step_to_k", "_overbalanced", "_balance_reflex", "_step_target",
        "_plan_path", "_balance_step")}

    def step_to_k(self, cells, sx, sy, p=None, kmax=1):
        share = None
        cpre = np.asarray(cells, np.int64).copy()
        r = orig["_step_to_k"](self, cells, sx, sy, p, kmax)
        if r is not None:
            # the support the step had BEFORE it moved the body (the cells it
            # left are the world's again now, so judge from a copy of the body
            # with the moved cells removed from `own`: done by the pre-move
            # copy, the floor is unchanged by a move)
            share = sole_support(self, cpre, sx, sy, r[0], r[1])
        # explain AFTER the move would be wrong (the body has moved); explain
        # only a refusal, where nothing moved
        ev = {"ev": "step_to_k", "dir": (sx, sy), "kmax": kmax, "got": r,
              "who": CALLER[0], "sole_support": share}
        if r is not None and share is not None and share < 1.0:
            COUNT["step_partial_support"] += 1
            COUNT["partial_" + CALLER[0]] += 1
        COUNT["steps_" + CALLER[0]] += 1
        if r is not None and (sx < 0) and r[0] == 0:
            COUNT["flat_step_back_" + CALLER[0]] += 1
        if r is None and p is not None:
            why = explain_step(self, p, cells, sx, sy, kmax)
            ev["why"] = [(list(c), s) for c, s in why]
            ev["why_count"] = dict(collections.Counter(s for _c, s in why))
            COUNT["step_refused"] += 1
            for _c, s in why:
                COUNT["refuse_" + s] += 1
        else:
            COUNT["step_ok"] += 1
        EV.append(ev)
        return r

    def overbalanced(self, p, own, raw=False):
        r = orig["_overbalanced"](self, p, own, raw)
        ghost = p.get("_ghost_from") is not None
        EV.append({"ev": "overbalanced", "ghost": ghost,
                   "over": None if r is None else [round(float(v), 2) for v in r]})
        COUNT["over_ghost" if ghost else "over_real"] += 1
        if r is not None:
            COUNT["over_ghost_yes" if ghost else "over_real_yes"] += 1
        return r

    def balance_reflex(self, p, own, over):
        r = orig["_balance_reflex"](self, p, own, over)
        EV.append({"ev": "reflex", "over": [round(float(v), 2) for v in over],
                   "held_up": r, "zlo": int(np.asarray(own)[:, 2].min())})
        COUNT["reflex"] += 1
        COUNT["reflex_held_up" if r else "reflex_fell"] += 1
        return r

    def step_target(self, p, own, over):
        r = orig["_step_target"](self, p, own, over)
        EV.append({"ev": "step_target", "over": [round(float(v), 2) for v in over],
                   "answer": None if r is None else list(r)})
        COUNT["target_none" if r is None else "target_step"] += 1
        if r is not None:
            COUNT["target_back"] += (r[0] < 0)
            COUNT["target_fwd"] += (r[0] > 0)
        return r

    def balance_step(self, p, own, shift):
        r = orig["_balance_step"](self, p, own, shift)
        EV.append({"ev": "balance_step", "shift": list(shift), "ok": r})
        return r

    def plan_path(self, cells, ex, p=None):
        path = orig["_plan_path"](self, cells, ex, p)
        ev = {"ev": "plan", "goal": list(map(float, ex)) if hasattr(ex, "__len__") else ex,
              "len": len(path or []), "head": [list(map(int, q)) for q in (path or [])[:4]]}
        # H1 TEST: the same query with the stance footprint in the full one's place
        real = routing.footprint
        try:
            routing.footprint = lambda c: stance_footprint(c)
            alt = orig["_plan_path"](self, cells, ex, p)
        finally:
            routing.footprint = real
        ev["stance_len"] = len(alt or [])
        ev["full_fp_cells"] = int(len(real(cells)[0]))
        ev["stance_fp_cells"] = int(len(stance_footprint(cells)[0]))
        if (not path) != (not alt):
            COUNT["plan_differs_by_footprint"] += 1
        COUNT["plan_empty_full"] += (not path)
        COUNT["plan_empty_stance"] += (not alt)
        COUNT["plan"] += 1
        EV.append(ev)
        return path

    W._step_to_k = step_to_k
    W._overbalanced = overbalanced
    W._balance_reflex = balance_reflex
    W._step_target = step_target
    W._balance_step = balance_step
    W._plan_path = plan_path
    return orig


def tick_record(w, p, cells, t):
    zlo = int(cells[:, 2].min())
    com, _kg = w._com_of(cells)
    foot, _grips = w._footing(p, cells)
    hull = w._hull_of([(int(a), int(b)) for a, b in np.asarray(foot)[:, :2]]) \
        if len(foot) else []
    xc = w._xcom(p, com, zlo)
    fo, fy = routing.footprint(cells)
    so, sy = stance_footprint(cells)
    return {"tick": t, "zlo": zlo,
            "com": [round(float(v), 2) for v in com],
            "xcom": [round(float(xc[0]), 2), round(float(xc[1]), 2)],
            "hull": hull, "n_firm": int(len(foot)),
            "fp": {"cells": int(len(fo)) + 1,
                   "w_x": int(fo.max() - fo.min() + 1) if len(fo) else 1,
                   "w_y": int(fy.max() - fy.min() + 1) if len(fy) else 1},
            "stance_fp": {"cells": int(len(so)) + 1,
                          "w_x": int(so.max() - so.min() + 1) if len(so) else 1,
                          "w_y": int(sy.max() - sy.min() + 1) if len(sy) else 1},
            "stride_debt": round(float(p.get("_stride", 0.0)), 3),
            "bodies": len(w.bodies),
            "belief_levels": int((p["fz"] >= 0).sum()) if p.get("fz") is not None else None}


def run(steps=12, riser=4, tread=6, ticks=900, out=None, quiet=False, solo=True):
    """Trace one flight. The wrapped methods (and the will, when solo) are put
    back when it returns, so a test that calls it leaves the World as it was."""
    saved = {n: sim.World.__dict__[n] for n in (
        "_step_to_k", "_overbalanced", "_balance_reflex", "_step_target",
        "_plan_path", "_balance_step", "_law_will") if n in sim.World.__dict__}
    inherited = [n for n in ("_step_to_k", "_overbalanced", "_balance_reflex",
                             "_step_target", "_plan_path", "_balance_step",
                             "_law_will") if n not in saved]
    try:
        return _run(steps, riser, tread, ticks, out, quiet, solo)
    finally:
        for n, f in saved.items():
            setattr(sim.World, n, f)
        for n in inherited:
            if n in sim.World.__dict__:
                delattr(sim.World, n)


def _run(steps, riser, tread, ticks, out, quiet, solo):
    install()
    if solo:
        # the person's own will is OFF: it plans its own path into p['_path'],
        # the same slot ours uses, and the two walk the body against each other
        # (measured: the 6 x 15 cm flight failed that way). `--with-will` keeps
        # it, as tests/test_stairs_on.py does.
        sim.World._law_will = lambda self: None
    w, p = stair_flight(steps=steps, riser=riser, tread=tread)
    goal = (w.shape[0] - 6, 7)
    top = 1 + steps * riser
    rows = []
    zmax = 0
    zprev = None
    stuck_since = None
    first_down = None
    first_free = None
    fh = open(out, "w") if out else None
    for t in range(ticks):
        EV.clear()
        cells = _cells(w, p)
        zlo = int(cells[:, 2].min())
        zmax = max(zmax, zlo)
        rec = tick_record(w, p, cells, t)
        CALLER[0] = "harness"
        w._stride(p, cells, goal, float(cells[:, 0].mean()), float(cells[:, 1].mean()))
        CALLER[0] = "will"
        w.step()
        rec["events"] = list(EV)
        rec["bodies_after"] = len(w.bodies)
        if zprev is not None and zlo < zprev and first_down is None:
            first_down = t
        if rec["bodies_after"] and first_free is None:
            first_free = t
        zprev = zlo
        rows.append(rec)
        if fh:
            fh.write(json.dumps(rec, default=int) + "\n")
        if zmax >= top or first_free is not None:
            break
    if fh:
        fh.close()
    return w, p, rows, dict(top=top, zmax=zmax, ticks=t + 1,
                            first_down=first_down, first_free=first_free)


def verdict(rows, info):
    """Score the five candidate mechanisms from the counts. Prints evidence,
    not a guess: each line says what was counted."""
    lines = []
    c = COUNT
    lines.append(f"climbed to z {info['zmax']} of {info['top']} in {info['ticks']} ticks; "
                 f"first step down at tick {info['first_down']}; free body at tick {info['first_free']}")
    lines.append(f"steps: ok {c['step_ok']}, refused {c['step_refused']}; refusals by candidate: "
                 f"world {c['refuse_world']}, blocked {c['refuse_blocked']}, "
                 f"support {c['refuse_support']}, balance {c['refuse_balance']}")
    fp = [r["fp"]["w_y"] for r in rows]
    sp = [r["stance_fp"]["w_y"] for r in rows]
    lines.append(f"H1 FOOTPRINT: full body shadow width (y) max {max(fp)} cells vs stance "
                 f"{max(sp)}; plans {c['plan']}, empty with full footprint {c['plan_empty_full']}, "
                 f"empty with stance footprint {c['plan_empty_stance']}, differing "
                 f"{c['plan_differs_by_footprint']}")
    lines.append(f"H2 LUNGE HULL: ghost balance asked {c['over_ghost']}, refused {c['over_ghost_yes']}; "
                 f"real body overbalanced {c['over_real_yes']} of {c['over_real']}")
    lines.append(f"H3 REFLEX TARGET: reflex calls {c['reflex']} (held up {c['reflex_held_up']}, "
                 f"fell {c['reflex_fell']}); targets: step {c['target_step']} "
                 f"(back {c['target_back']}, forward {c['target_fwd']}), none {c['target_none']}")
    bl = [r["belief_levels"] for r in rows if r["belief_levels"] is not None]
    lines.append(f"WHO WALKS: steps by the harness {c['steps_harness']}, by the person's own will "
                 f"{c['steps_will']}; flat steps BACK (x-) by harness {c['flat_step_back_harness']}, "
                 f"by will {c['flat_step_back_will']}; steps taken with the soles only partly "
                 f"supported: {c['step_partial_support']} (harness {c['partial_harness']}, will {c['partial_will']})")
    lines.append(f"H4 BELIEF: levels believed min {min(bl) if bl else None} max {max(bl) if bl else None}")
    debt = [r["stride_debt"] for r in rows]
    lines.append(f"H5 STRIDE DEBT: min {min(debt)} max {max(debt)} (a debt below 0 or above "
                 f"a few paces is drift)")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--riser", type=int, default=4)
    ap.add_argument("--tread", type=int, default=6)
    ap.add_argument("--ticks", type=int, default=900)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--with-will", action="store_true",
                    help="leave the person's own will on (the test's way)")
    a = ap.parse_args()
    w, p, rows, info = run(a.steps, a.riser, a.tread, a.ticks, a.out, a.quiet,
                           solo=not a.with_will)
    for ln in verdict(rows, info):
        print(ln)


if __name__ == "__main__":
    main()
