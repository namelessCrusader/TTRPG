"""THE BODY FUZZER: run many small worlds and check what must hold in ANY of
them, instead of hand-picking the handful a scene author thought to try.

Built for step EF (docs/sim-mind-design.md), after roadmap item 85's fuzzer
and aimed at the one bug it never finished running down (item 87, "two men who
fall together end up sharing a body"). Each world is `harvest.world(seed)`
(src/voxel/harvest.py:73), with a second `scenes._person` body added if the
harvest world did not already drop one in — the fuzzer always wants two
bodies in play, because the bug it is hunting needs two.

Checked every tick, for 300 ticks:
  - no two people own the same voxel
  - no living person owns a dead one's flesh: the corpse is what the dead
    man owned when he was last alive (what he was built as, if he died on
    the first step), less every cell it has left since (step F9; the
    check above skips the dead, so it missed a living man claiming a whole
    corpse lying against him)
  - no living person owns more flesh than he was built with. A body only
    loses flesh (burns, wounds); a deed that grows past it has taken
    somebody else's: a corpse that fell and landed against him, which the
    check above loses sight of once it has been through the particle
    solver, or a neighbour's swung arm (roadmap 87)
  - a person's owned cells are ONE connected lump
  - no NaN or infinity in smass, E, fvol, fpot
  - nobody stays "aloft" (a free body, or claimed by nobody) for the whole run
And once, at the end, only if nothing burned in that world:
  - total_mass per material is unchanged from the start

This does NOT fix anything it finds. It reports.

A body is the flesh that carries its eid (step E7), so the first three
checks hold by construction now; they stay, to say so. "Aloft" is a man the
claim gives no body to move (`_claim_mask` None): in a free body, or half
landed. A body in PIECES is still one person (identity is the eid), so the
lump check now reports what it finds, with the sizes of the pieces and the
first tick, in `lumps` (a list, which fuzz_run.py does not add up).

With --audit (steps E6 and E7) every world is built with the body audit on
(`audit_bodies`), and the run ends with how much flesh nobody had stamped the
claim adopted, and by whom (`persons._adopt_flesh`): after E7 the one place a
claim can still differ from the eid it found.

    python3 tools/fuzz_bodies.py --worlds 5
    python3 tools/fuzz_bodies.py --worlds 150 --ticks 300
    python3 tools/fuzz_bodies.py --worlds 150 --audit --others others.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.realpath(os.environ.get("MARK1_REPO", os.path.dirname(HERE)))
sys.path.insert(0, REPO)

import numpy as np                                  # noqa: E402

from src.voxel import harvest                        # noqa: E402
from src.voxel.scenes import _person, _room_for       # noqa: E402
from src.voxel.materials import NMAT_ROWS             # noqa: E402  (every id with a row: ICE too)
from src.voxel.sim import FLESH                        # noqa: E402
from src.voxel.things import ThingsMixin               # noqa: E402

TICKS = 300


def audit_on():
    """Make every World built from now on audit its bodies. `harvest.world`
    builds its own World and adds people in it at once, so the switch must
    be on before the first `_person`: set it where every World sets it off,
    at the end of `_init_things`."""
    if getattr(ThingsMixin._init_things, "audits", False):
        return
    plain = ThingsMixin._init_things

    def init(self):
        plain(self)
        self.audit_bodies = True
    init.audits = True
    ThingsMixin._init_things = init


def audit_tally(w, seed, hist, others):
    """Add one world's adoptions to `hist` ({"adopted": [cells, person-ticks,
    worlds]}) and each one to `others`. Returns the world's own count,
    {"adopted": [cells, person-ticks, first tick]}."""
    mine = {}
    for r in w.id_audit:
        if "adopted" not in r:
            continue                          # an author's affords finding
        m = mine.setdefault("adopted", [0, 0, r["tick"]])
        m[0] += r["adopted"]
        m[1] += 1
        h = hist["adopted"]
        h[0] += r["adopted"]
        h[1] += 1
        others.append(dict(r, seed=seed))
    if mine:
        hist["adopted"][2] += 1
    return mine


def _ensure_two(w, people, lip):
    """harvest.world gives one person always and a second one in roughly a
    third of worlds (harvest.py:118-123). The fuzzer wants two every time, so
    it adds one itself when the world did not."""
    if len(people) >= 2:
        return people
    p0 = people[0]
    px0, py0 = p0["px"], p0["py"]
    nx, ny, _nz = w.shape
    # ANYWHERE CLEAR, near enough that the two can interact (fall together,
    # catch each other) but not inside the first body's own voxels.
    # CLEAR FOR THE WHOLE BODY (`_room_for`, F9). This asked only x px-2 to
    # px+2, and asked z up to floor+30 with a slice the lattice cut off at nz
    # without a word: on a tall post (seed 21) the answer was yes, and 'two'
    # was built without his head. Asked properly, the first six places are
    # sometimes all taken (a thing to hand by his right arm), so it goes on
    # out to 15 either side; the first six are tried first, as before.
    for dx in (9, -9, 7, -7, 11, -11, 8, -8, 10, -10, 12, -12, 13, -13,
               14, -14, 15, -15):
        px = px0 + dx
        if not (3 <= px <= nx - 4):
            continue
        col = np.argwhere(w.mat[px, py0, :] != 0)
        floor = int(col[:, 0].max()) + 1 if len(col) else 1
        if _room_for(w, px, py0, floor):
            q = _person(w, px, py0, z0=floor)
            q["name"] = "two"
            people.append(q)
            return people
    return people          # no clear spot found; the world runs with one


def _flood_lump(cells):
    """Is this set of (x, y, z) cells one 6-connected lump? Built local to the
    cells' own bounding box — a body is a few hundred voxels, never the whole
    lattice, so a plain BFS in that small box is cheap."""
    if not cells:
        return True
    pts = np.array(list(cells), dtype=np.int64)
    lo = pts.min(axis=0)
    shape = tuple(pts.max(axis=0) - lo + 1)
    box = np.zeros(shape, bool)
    local = pts - lo
    box[tuple(local.T)] = True
    seen = np.zeros(shape, bool)
    stack = [tuple(local[0])]
    seen[stack[0]] = True
    count = 1
    while stack:
        x, y, z = stack.pop()
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                           (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            nxp, nyp, nzp = x + dx, y + dy, z + dz
            if not (0 <= nxp < shape[0] and 0 <= nyp < shape[1]
                    and 0 <= nzp < shape[2]):
                continue
            if box[nxp, nyp, nzp] and not seen[nxp, nyp, nzp]:
                seen[nxp, nyp, nzp] = True
                count += 1
                stack.append((nxp, nyp, nzp))
    return count == len(cells)


def _sizes(cells):
    """The sizes of the 6-connected lumps these cells make, largest first."""
    left, out = set(cells), []
    while left:
        stack, n = [left.pop()], 1
        while stack:
            x, y, z = stack.pop()
            for c in ((x + 1, y, z), (x - 1, y, z), (x, y + 1, z),
                      (x, y - 1, z), (x, y, z + 1), (x, y, z - 1)):
                if c in left:
                    left.remove(c)
                    stack.append(c)
                    n += 1
        out.append(n)
    return sorted(out, reverse=True)


def run_one(seed, ticks=TICKS, audit=None):
    """Step one world and return a dict of what the invariants found.
    `audit`, if given, is (hist, others) for `audit_tally`."""
    w, people, lip = harvest.world(seed)
    people = _ensure_two(w, people, lip)
    w.policy = harvest.Roam(seed)

    bugs = {"shared_voxel": 0, "not_one_lump": 0, "nan": 0, "claims_dead": 0,
            "oversized": 0}
    aloft_run = {id(q): 0 for q in people}
    # WHAT EACH MAN OWNED while he was last alive, and to start with what he
    # was built as: the flesh in his own building box (`_room_for`; the
    # anchor is px+1). Without that, a man dead on the first step had no
    # deed on record and his corpse was invisible to the check: six of the
    # nine worlds the E6 audit found are exactly that.
    last_own = {}
    for q in people:
        ax, ay = q["anchor"]
        box = w.mat[ax - 3:ax + 4, ay - 2:ay + 3, :] == FLESH
        last_own[id(q)] = frozenset(
            (int(x) + ax - 3, int(y) + ay - 2, int(z))
            for x, y, z in np.argwhere(box))
    built = {k: len(v) for k, v in last_own.items()}
    max_aloft = 0
    lumps = []                 # the first tick each man was in pieces
    burned = False
    mass0 = {m: w.total_mass(m) for m in range(NMAT_ROWS)}

    for t in range(ticks):
        w.step()
        if w.burning().any():
            burned = True

        for arr in (w.smass, w.E, w.fvol, w.fpot):
            if not np.isfinite(arr).all():
                bugs["nan"] += 1

        owns = []
        corpses = []
        any_aloft = False
        for q in people:
            if not q["alive"] and not q.get("safe") and last_own[id(q)]:
                # ONLY EVER LESS: a cell the corpse has left is not the
                # corpse again when somebody living steps into it later
                last_own[id(q)] = frozenset(
                    c for c in last_own[id(q)]
                    if w.mat[c] == FLESH and not w.moving[c])
                corpses.append(last_own[id(q)])
            if not q["alive"] or q.get("safe"):
                aloft_run[id(q)] = 0
                continue
            w._person_cells(q)                 # refreshes q["_own"] this tick
            own = q.get("_own") or frozenset()
            if q.get("_claim_mask") is not None:
                last_own[id(q)] = own
                owns.append((q["name"], own))
                if len(own) > built[id(q)]:
                    bugs["oversized"] += 1
                aloft_run[id(q)] = 0
                if not _flood_lump(own):
                    bugs["not_one_lump"] += 1
                    if not any(r[1] == q["name"] for r in lumps):
                        lumps.append((t, q["name"], _sizes(own), len(own),
                                      built[id(q)]))
            else:
                aloft_run[id(q)] += 1           # off the lattice: aloft
                any_aloft = True
        if any_aloft:
            max_aloft = max(max_aloft, max(aloft_run.values()))

        for i in range(len(owns)):
            for j in range(i + 1, len(owns)):
                if owns[i][1] & owns[j][1]:
                    bugs["shared_voxel"] += 1
        for _name, own in owns:
            for dead in corpses:
                if own & dead:
                    bugs["claims_dead"] += 1

    mass_drift = 0.0
    if not burned:
        mass1 = {m: w.total_mass(m) for m in range(NMAT_ROWS)}
        mass_drift = max(abs(mass1[m] - mass0[m]) for m in mass0)

    bugs["stuck_aloft"] = int(max_aloft >= ticks)
    bugs["lumps"] = lumps
    bugs["mass_drift"] = mass_drift if not burned else None
    if audit is not None:
        bugs["audit"] = audit_tally(w, seed, *audit)
    return bugs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=5)
    ap.add_argument("--ticks", type=int, default=TICKS)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--audit", action="store_true",
                    help="print what the claim adopted (E6, E7)")
    ap.add_argument("--others", default=None,
                    help="with --audit: write every 'other' case here, as "
                         "JSON lines")
    args = ap.parse_args()
    audit = None
    if args.audit:
        audit_on()
        audit = ({"adopted": [0, 0, 0]}, [])

    t0 = time.time()
    totals = {"shared_voxel": 0, "not_one_lump": 0, "nan": 0, "stuck_aloft": 0,
              "claims_dead": 0, "oversized": 0}
    worst_drift = 0.0
    worlds_with = dict.fromkeys(totals, 0)
    for i in range(args.worlds):
        seed = args.seed0 + i
        bugs = run_one(seed, args.ticks, audit)
        if bugs.get("audit"):
            print(f"  seed {seed}: " + ", ".join(
                f"{k} {c} cells/{pt} ticks from t{first}"
                for k, (c, pt, first) in sorted(bugs["audit"].items())),
                flush=True)
        for t, name, sizes, n, built in bugs["lumps"]:
            print(f"  seed {seed}: {name} in pieces from t{t}: {sizes} "
                  f"({n} of the {built} he was built as)", flush=True)
        for k in totals:
            totals[k] += bugs[k]
            if bugs[k]:
                worlds_with[k] += 1
        if bugs["mass_drift"] is not None:
            worst_drift = max(worst_drift, bugs["mass_drift"])
    dt = time.time() - t0

    print(f"{args.worlds} worlds, {args.ticks} ticks each, in {dt:.1f} s "
          f"({dt / args.worlds:.2f} s/world)")
    for k in totals:
        print(f"  {k}: {totals[k]} ticks over {worlds_with[k]} of "
              f"{args.worlds} worlds")
    print(f"  worst mass drift (no-burn worlds): {worst_drift:.3f} g")
    if audit is None:
        return
    hist, others = audit
    c, pt, nw = hist["adopted"]
    print(f"body audit: the claim adopted {c} unstamped cells, "
          f"{pt} person-ticks, {nw} worlds")
    for o in others[:40]:
        print("   ", json.dumps(o))
    if args.others:
        with open(args.others, "w") as f:
            for o in others:
                f.write(json.dumps(o) + "\n")


if __name__ == "__main__":
    main()
