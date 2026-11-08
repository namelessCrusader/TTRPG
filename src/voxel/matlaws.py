"""MATLAWS: the T9 material laws, wired into a World (wave 13b, lane C8a).

The stand-alone modules (matfric, matdamage, matbeam, matstress) hold the
closed forms and know nothing about a World. This file is the glue: it reads a
world's arrays, calls them, and writes back through the laws the world already
has. Four switches, each its own name in `world.mat_laws`:

  friction  the force to START a haul is mu_s m g and the force while sliding
            (and the heat of dragging) is mu_k m g, with mu read from the PAIR
            of surfaces and the film between them (matfric). Before: 0.40.
  members   the load-aware support check (matbeam) as a VETO on the old SPAN
            law: a beam or column that is over its strength becomes a PIECE
            (`rubble`, `_rub_gen` bumped), which the existing support law
            already knows how to drop. It never lets stand what SPAN drops.
  damage    a blow that does not reach the shatter threshold leaves a crack
            (`dmg`, one writer: `add_damage`, one combine rule); the next blow
            on that cell breaks it at thr x (1 - D) (bodies.py reads it).
  spall     a hot stone or glass face that is held in compression by the cooler
            body behind it throws a flake off (matstress), on its own gate.

THE GATE. `mat_laws` is a CLASS attribute (a frozenset of those names), so a
world saved before this existed still loads, and `world.mat_laws = frozenset()`
gives any world the old behaviour back (the old scenes' golden is one set away).
`FLAT` in the set makes friction read the flat 0.40 table (the W2 step).

Every number here is matfric's, matdamage's, matbeam's or matstress's; the
glue adds only these (all (u), each in the switch ledger of the C8a report):
  VETO_EVERY    ticks between two checks of an unchanged layout
  SPALL_EVERY   ticks between two spall passes
  ETA           the fraction of a blow's energy that opens crack surface
"""
import functools
import weakref

import numpy as np

from .materials import AIR, FLESH, MATREC, SKIN_T

ALL_LAWS = frozenset(("friction", "members", "damage", "spall", "edge", "wound"))

VETO_EVERY = 10          # ticks between two member checks
VETO_EVERY_HOT = 40      # ... while something structural is hot (1 s of sim time)
VETO_SOON = 2            # ... after a member broke or took damage: its neighbours are asked at once
HOT_STRUCT_C = 100.0     # a structural cell this hot has lost strength (wood 0.75 at 100 C); cooler is flat
CHAR_KS, CHAR_KE = 0.05, 0.09   # wood above 350 C keeps char's share of strength and stiffness (CHAR row / WOOD row)
MIN_FILL = 0.15          # a cell with less of its volume filled than this (a frozen splat, a charred sliver) is weight, not a member
CROP_MARGIN = 24         # cells of room round the changed cells in the crop (1.2 m at 5 cm)
TOUCH = 4                # members within this many cells of a changed cell are asked
SPALL_EVERY = 20         # ticks between two spall passes (1 s of sim time at 25 ms)
HOT_C = 150.0            # a cell cooler than this is not a spall candidate (stone is flat in strength below it)
FIRE_PATH_C = 20.0       # ambient: the body mean is measured against this at the least


# WHAT THE VETO REMEMBERS between two looks (the layout it saw, what it still owes
# a look at) lives HERE, keyed by the world, and not on the world: the quiet
# gate and the golden hash both read a world's own attributes, and a tick stamp
# or a copy of `mat` on the world would make a resting world look busy (the
# scheduler could never jump a night) and would show up as new state.
_STATE = weakref.WeakKeyDictionary()


@functools.lru_cache(maxsize=1)
def _record():
    """matbeam's record dict over ALL 64 material ids: the twelve rows, and
    nothing (not structural, no strength) for an id with no row yet (ice, the
    mixes' reserved ids), so an id the lattice holds can always be indexed."""
    from . import matbeam as B
    rec = B.rec_from_rows(list(MATREC.by_id_list) + [None] * (64 - len(MATREC.by_id_list)))
    # WOOD ABOVE ITS CHAR TEMPERATURE IS CHAR, NOT NOTHING. The record's derate
    # row for wood goes to zero at 350 C (a 5 cm cell at that mean temperature
    # has no wood left in it), and zero strength would break a charred post
    # under its own weight. What is left is the CHAR row's strength: compression
    # 2 MPa against wood's 40 (0.05), stiffness 1 GPa against 11 (0.09). This
    # floor is the veto's alone; the table stays as the spec wrote it.
    wood = [r.key for r in MATREC.by_id_list].index("wood")
    hot = (rec["T0"] + rec["dT"] * np.arange(rec["nT"])) >= 350.0
    rec["ks"][wood, hot] = np.maximum(rec["ks"][wood, hot], CHAR_KS)
    rec["kE"][wood, hot] = np.maximum(rec["kE"][wood, hot], CHAR_KE)
    return rec


def has(w, law):
    return law in w.mat_laws


# ─────────────────────────────── friction ───────────────────────────────────
def contact_mu(w, cells, kinetic):
    """The friction coefficient for a thing resting on the floor, or None when
    nothing is under it (the caller then keeps the old 0.40).

    The pair is the thing's own material at the lowest cell of each column of
    it, against the material directly under that cell. The film is what the
    fluid in the thing's lowest cells is (matfric.film_of: oil, water, blood).
    The answer is the mean over the columns, so a plank half on a rug and half
    on stone reads between the two."""
    from . import matfric
    c = np.asarray(cells, np.int64).reshape(-1, 3)
    if not len(c) or w.mat_laws is None:
        return None
    order = np.lexsort((c[:, 2], c[:, 1], c[:, 0]))
    c = c[order]
    first = np.ones(len(c), bool)
    first[1:] = (c[1:, 0] != c[:-1, 0]) | (c[1:, 1] != c[:-1, 1])
    b = c[first]
    b = b[b[:, 2] > 0]
    if not len(b):
        return None
    under = (b[:, 0], b[:, 1], b[:, 2] - 1)
    floor = w.mat[under]
    on = floor != AIR
    if not on.any():
        return None
    b, floor = b[on], floor[on]
    at = tuple(b.T)
    thing = w.mat[at]
    wet = np.asarray(w.fvol[at]) > 0.05 * w.cap
    film = matfric.film_of(np.where(wet, w.fl[at], 0))
    table = (matfric.FLAT_S, matfric.FLAT_K) if "flat" in w.mat_laws else None
    mu = matfric.mu_pair(thing, floor, film, kinetic=kinetic, table=table)
    mu = np.atleast_1d(mu)
    return float(mu[0]) if (mu == mu[0]).all() else float(np.mean(mu))   # (the mean of equals is not always equal)


# ─────────────────────────────── damage ─────────────────────────────────────
def add_damage(w, cells, dD):
    """THE ONE WRITER of `dmg`: the damage of each cell becomes combine(D, dD)
    (independent causes, matdamage.combine). Makes the lazy array on first use
    and bumps `_dmg_gen`. Returns the number of cells that changed."""
    from . import matdamage
    cells = np.asarray(cells, np.int64).reshape(-1, 3)
    if not len(cells):
        return 0
    arr = w.ensure_damage()
    at = tuple(cells.T)
    new = matdamage.combine(arr[at].astype(np.float64), dD).astype(arr.dtype)
    changed = int((new != arr[at]).sum())
    arr[at] = new
    if changed:
        w._dmg_gen = getattr(w, "_dmg_gen", 0) + 1
        st = _STATE.get(w)
        if st is not None:                       # a cracked cell is asked about at the next look
            st["dirty"].append(cells.copy())
    return changed


def _gimp_table():
    g = np.full(64, 1e12)                       # an id without a row never breaks
    for i, r in enumerate(MATREC.by_id_list):
        g[i] = r.gimp
    return g


_GIMP = _gimp_table()


def _melt_table():
    t = np.full(64, np.inf)                      # no melting point: never "molten"
    for i, r in enumerate(MATREC.by_id_list):
        if r.melt is not None:
            t[i] = r.melt.T_C
    return t


_MELT_C = _melt_table()


ETA = 0.3                # (u) matdamage.ETA_IMPACT, the share of a blow's energy that opens crack surface


def impact_damage(w, cell, e_J, face_area_m2):
    """A blow of e_J joules on a face of face_area_m2 that did NOT break the
    cell leaves dD = eta e / (gimp A) of crack (matdamage.impact_dD). A cell
    that is flesh, air or has no toughness takes none."""
    from . import matdamage
    m = int(w.mat[cell])
    if m == AIR or m == FLESH:
        return 0
    dD = matdamage.impact_dD(e_J, _GIMP[m], face_area_m2, ETA)
    if dD <= 0.0:
        return 0
    return add_damage(w, [cell], dD)


# ─────────────────────────────── members ────────────────────────────────────
def temperature(w):
    """The temperature field, C, made WITHOUT touching any cache of the world.
    `w.T()` asks for the heat capacity, which is kept (`_cap`, `_cap_key`) and is
    part of the state a world is compared by; a law that only wants to LOOK must
    not move it. So the capacity is worked out here for the box round the heat
    only, and everything outside the box is ambient exactly."""
    from .materials import AMBIENT
    T = np.full(w.shape, AMBIENT, np.float32)
    box = w._window(w.E != 0, 0)
    if box is None:
        return T
    sc, fc = w._mix_args()
    cap = w._cap_of(w.mat[box], w.smass[box], w.fl[box], w.fvol[box],
                    None if sc is None else sc[box], None if fc is None else fc[box])
    T[box] = AMBIENT + w.E[box] / cap
    return T


def _fill(w):
    from . import sim
    dens = np.array(sim._DENS_ARR, float)
    dens[:sim.NMAT] = [sim.SOLID[k][0] for k in range(sim.NMAT)]
    return np.clip(w.smass / np.maximum(dens[w.mat] * w.vox_l, 1e-9), 0.0, 1.0)


def _structural_hot(w, T, rec):
    """Cells of a structural material at or above HOT_STRUCT_C: where fire
    can have taken the strength out of a member."""
    return (T >= HOT_STRUCT_C) & rec["structural"][w.mat]


def analyse_world(w, want, rec=None, T=None):
    """(Members, Loads, Analysis, offset) of the members NEAR the cells `want`
    (a list of (n, 3) arrays), static loads only (dead weight; a person's flesh
    and grip-carried cells are weight that is passed down and never a member).

    Cropped to the box of `want` grown by CROP_MARGIN cells sideways, from the
    floor to the top of the solids; the crop edge counts as a support, which
    can only make a member look safer, never break one wrongly. Only members
    within TOUCH cells of `want` are analysed (the loads are worked out for all
    of them: they pass down through members that are not asked about)."""
    from . import matbeam as B
    rec = rec if rec is not None else _record()
    solid = w.mat != AIR
    if not solid.any() or not len(want):
        return None
    pts = np.concatenate([np.asarray(a, np.int64).reshape(-1, 3) for a in want])
    lo = np.maximum(pts.min(0) - CROP_MARGIN, 0)
    hi = np.minimum(pts.max(0) + CROP_MARGIN + 1, w.shape)
    zt = np.flatnonzero(solid.any(axis=(0, 1)))
    lo[2], hi[2] = 0, int(zt[-1]) + 2
    hi[2] = min(hi[2], w.shape[2])
    sl = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
    try:
        grip = set(w._grip_cells()) | set(w._borne_cells()) if w.persons else set()
    except Exception:
        grip = set()
    off = lo.astype(np.int64)
    grip_l = {(x - off[0], y - off[1], z - off[2]) for (x, y, z) in grip
              if all(l <= v < h for v, l, h in zip((x, y, z), lo, hi))}
    sub = lambda a: a[sl]
    mat, eid = sub(w.mat), sub(w.eid)
    T = temperature(w) if T is None else T
    T = T[sl]
    fill = sub(_fill(w))
    m = B.extract_members(
        sub(solid), fill, mat, eid, None if w.dmg is None else sub(w.dmg), T,
        sub(w.rubble), mat == FLESH, grip_l, sub(w._is_thing(w.eid)), None,
        sub(w.moving) | (T >= _MELT_C[mat]) | (fill < MIN_FILL),   # a solid at its melting point is a liquid to be, and a film is not a post: neither is a member
        w.scale * 0.1, rec=rec, edge_supports=True)
    gm = B._mask(grip_l, m.shape)
    W = np.where(sub(solid), B.weights_N(sub(w.smass), gm), 0.0)
    loads = B.accumulate_loads(m, W)
    mask = np.zeros(m.shape, bool)
    for a in want:
        c = np.asarray(a, np.int64).reshape(-1, 3) - off
        c = c[((c >= 0) & (c < np.array(m.shape))).all(axis=1)]
        mask[tuple(c.T)] = True
    for _ in range(TOUCH):                       # grow by TOUCH cells (6-neighbour)
        g = mask.copy()
        g[1:] |= mask[:-1]; g[:-1] |= mask[1:]
        g[:, 1:] |= mask[:, :-1]; g[:, :-1] |= mask[:, 1:]
        g[:, :, 1:] |= mask[:, :, :-1]; g[:, :, :-1] |= mask[:, :, 1:]
        mask = g
    runs = {i for i, r in enumerate(m.runs) if mask[tuple(r.cells.T)].any()}
    cols = {i for i, c in enumerate(m.columns)
            if mask[c.xy[:, 0], c.xy[:, 1], c.z0:c.z1 + 1].any()}
    res = B.analyse(m, rec, loads, only_runs=runs, only_cols=cols)
    return m, loads, res, off


def _break(w, rows, off):
    """Make the failed members pieces. A beam: the critical cell becomes
    rubble (the support law then drops it if air is below it). A column: it
    goes through `_shatter`; boxed in, it becomes rubble in place. Returns the
    cells that broke (the next pass looks at their neighbours)."""
    broke = []
    for r in sorted(rows, key=lambda r: r["cell"]):
        x, y, z = (int(v) + int(o) for v, o in zip(r["cell"], off))
        if w.mat[x, y, z] == AIR or w.rubble[x, y, z]:
            continue
        before = int(w.mat[x, y, z])
        if r["kind"].endswith("column"):
            w._shatter(x, y, z, 1.0)
        if int(w.mat[x, y, z]) == before and not w.rubble[x, y, z]:
            w.rubble[x, y, z] = True
            w._rub_gen += 1
        broke.append((x, y, z))
    return broke


def members_veto(w):
    """THE VETO, asked about what CHANGED and what is HOT, never about the whole
    world. Every VETO_EVERY ticks (VETO_EVERY_HOT while anything structural is
    hot) it looks for
      * cells whose matter changed since the last look, a person's own flesh
        walking about excepted (a world with nobody moving costs one compare);
      * cells that took damage or were just broken (the stubs of a snapped beam);
      * structural cells at or above HOT_STRUCT_C (a fire has the strength out
        of them: wood is nothing at 350 C).
    and checks only the members near those cells. The first look only records
    the layout: a world that was BUILT standing is taken to stand (SPAN holds it
    and nothing has changed to ask the question)."""
    st = _STATE.get(w)
    if st is None or st["mat"].shape != w.mat.shape:
        _STATE[w] = {"tick": w.tick, "mat": w.mat.copy(), "dirty": [], "hot": False}
        return 0
    gap = VETO_EVERY_HOT if st["hot"] else VETO_EVERY
    if st["dirty"]:
        gap = min(gap, VETO_SOON)
    if w.tick - st["tick"] < gap:
        return 0
    st["tick"] = w.tick
    want = list(st["dirty"])
    st["dirty"] = []
    cur, old = w.mat.ravel(), st["mat"].ravel()
    if not np.array_equal(cur, old):
        idx = np.flatnonzero(cur != old)
        o, n = old[idx], cur[idx]
        quiet = ((o == AIR) | (o == FLESH)) & ((n == AIR) | (n == FLESH))   # a body walked, no matter moved
        old[idx] = n
        idx = idx[~quiet]
        if len(idx):
            want.append(np.stack(np.unravel_index(idx, w.shape), axis=1))
    st["hot"] = False
    T = None
    if w._heat_busy():
        rec = _record()
        T = temperature(w)
        hot = np.argwhere(_structural_hot(w, T, rec))
        if len(hot):
            st["hot"] = True
            want.append(hot)
    if not want:
        return 0
    out = analyse_world(w, want, T=T if st['hot'] else None)
    n = 0
    if out is not None:
        m, loads, res, off = out
        broke = _break(w, res.failing(), off)
        n = len(broke)
        if broke:
            st["dirty"].append(np.array(broke, np.int64))
    return n


# ─────────────────────────────── spall ──────────────────────────────────────
def spall_pass(w):
    """A hot, exposed stone or glass face held in compression by the cooler
    body behind it throws off a flake (matstress.spall). Runs on its own gate:
    every SPALL_EVERY ticks, in the hot window only."""
    from . import matstress, matbeam as B
    if w.tick % SPALL_EVERY:
        return 0
    win = w._hot_window(pad=0)
    if win is None:
        return 0
    rec = _record()
    rows = MATREC.by_id_list
    sl = tuple(slice(*s.indices(n)[:2]) for s, n in zip(win, w.shape))
    T = temperature(w)
    sub_mat = w.mat[sl]
    cand = (sub_mat != AIR) & (T[sl] > HOT_C) & (rec["sc"][sub_mat] > 0) & (rec["st"][sub_mat] < 0.5 * rec["sc"][sub_mat])
    if not cand.any():
        return 0
    n = 0
    skin = getattr(w, "skin", None)
    h = w.scale * 0.1
    off = np.array([s.start for s in sl])
    for c in np.argwhere(cand):
        x, y, z = (int(v) + int(o) for v, o in zip(c, off))
        m = int(w.mat[x, y, z])
        if w.rubble[x, y, z]:
            continue
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            px, py, pz = x + d[0], y + d[1], z + d[2]
            if not (0 <= px < w.shape[0] and 0 <= py < w.shape[1] and 0 <= pz < w.shape[2]):
                continue
            if w.mat[px, py, pz] != AIR:
                continue
            # the flux line goes INTO the body, away from the exposed face
            line_T, line_f, q = [], [], (x, y, z)
            for _ in range(max(1, int(round(matstress.BODY_MAX_M / h)))):
                if not (0 <= q[0] < w.shape[0] and 0 <= q[1] < w.shape[1] and 0 <= q[2] < w.shape[2]):
                    break
                if w.mat[q] == AIR:
                    break
                line_T.append(float(T[q]))
                line_f.append(1.0)
                q = (q[0] - d[0], q[1] - d[1], q[2] - d[2])
            Tb = matstress.body_mean(line_T, line_f, h)
            Ts = line_T[0] + (float(skin[x, y, z]) if skin is not None else 0.0)
            eff = B.effective(rec, np.array([m]), np.array([Ts]), np.array([0.0 if w.dmg is None else float(w.dmg[x, y, z])]))
            sig = matstress.surface_stress(float(eff["E"][0]), float(rec["nu"][m]), float(rows[m].alpha), Ts, Tb)
            row = rows[m]
            rho = float(row.rho)
            depth = matstress.skin_depth(row.k, rho, row.cp, SKIN_T[0])
            fails, dep = matstress.spall({"sc": eff["sc"]}, sig, depth, True)
            if bool(fails[0]):
                flake_g, e_flake, _, _ = matstress.spall_flake(
                    float(w.smass[x, y, z]), float(w.E[x, y, z]), rho, float(dep[0]), h * h)
                if flake_g > 0.0:
                    w.smass[x, y, z] -= np.float32(flake_g)
                    w.E[x, y, z] -= np.float32(e_flake)
                    w.mat[px, py, pz] = m
                    w.smass[px, py, pz] += np.float32(flake_g)
                    w.E[px, py, pz] += np.float32(e_flake)
                    w.rubble[px, py, pz] = True
                    w._rub_gen += 1
                    n += 1
            break
    return n
