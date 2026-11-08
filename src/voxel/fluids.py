"""Fluid laws: fall, spread, head pressure, drops, stirring, splashes, soaking, and acid."""
import functools
import weakref

import numpy as np
from .kernels import HAVE_NUMBA, njit
from .materials import (ACID, AIR, AMBIENT, BLOOD, BOIL, CHAR, FLUID, GRAVITY, MIRON, MLEAD, MTIN, NMAT, NMAT_PAD,
                        NOFLUID, OIL, REACTIONS, ROPE, SOLID, TICK_S, WATER, WEAK_ACID, WOOD, _CSOLID_ARR, _FDENS_ARR,
                        _FLIDS, _MIXHEAT, _RATE_PTS, _REACTIVE, _SKIN_DEPTH_ARR, MISCIBLE)


# ── THE COMPILED SWEEPS ─────────────────────────────────────────────────────
#
# `_law_flow` and `_law_head` walk every wet cell in Python, a hundred
# microseconds a cell: a still 2 m pond cost 110 ms a tick to change nothing.
# The compiled path below does the SAME sweep in the SAME order and must give
# the same bits (tests/test_fluids.py runs worlds both ways, tick by tick); the
# Python path stays beside it and is the definition. Turn `w.fused` off, or
# `skip_quiet` or `sleep_regions`, to get it.
#
# Two things make a quiet world cost almost nothing:
#   - the list of wet cells, and the order they are swept in, is kept while the
#     set of wet cells is the same (a still pond's is, tick after tick);
#   - a scan that asks, of the state as it stands, whether ANY cell would move
#     any fluid. If none would, the sweep is skipped: the first cell to act in
#     a sweep acts on the state the sweep began with, because nothing has
#     changed before it, so a world where none can act at the start is a world
#     where none acts at all.
#
# EXACT, NOT NEAR. The numbers must come out as numpy's scalar arithmetic gives
# them, which is not always what a compiled loop would give by default: a
# float32 cell times a Python float stays float32 (NumPy 2), a float32 and a
# float64 mix as float64, and a `min` of two float32 is a float32. The
# arithmetic below is written to follow the Python one operation by operation,
# with the type of each result in a comment where it is not plain.
# A table that is read here is passed IN, never frozen into the compiled code:
# a cached kernel would otherwise keep the old row after a table was edited.
_NFL = len(_FDENS_ARR)                     # fluid ids are 0 (none) .. _NFL - 1: NFLUID_PAD (64) since
                                           # wave 13, whatever is listed. A table below is a row for each
                                           # listed liquid; an id past them reads the "none" row (0.0, or
                                           # -1 = "do not mix"), never a made-up liquid.


# ── WHAT A LIQUID LEAVES BEHIND (docs/realism-map.md, 3.2; docs/realism-references.md G16) ──
#
# THE FILM. Spilled on a floor, a liquid does not thin out for ever: surface
# tension pulls the edge back, and the puddle stops at the thickness
#     h = 2 * lc * sin(theta / 2),     lc = sqrt(sigma / (rho * g)),
# lc the capillary length and theta the contact angle (Mugele & Heikenfeld,
# source 95). Water on a floor is 1-4 mm, a wetting oil under 1 mm, molten lead
# about 3-4 mm (it does not wet stone). Per liquid: surface tension sigma, N/m,
# and the contact angle on a stone floor, degrees. The angles are a game choice
# inside the measured ranges: water 40-90 on dirty stone, oil 10-40, a metal
# that has not wetted its floor 90-150. UNVERIFIED for acid and blood.
WETTING = {WATER: (0.0728, 60.0), OIL: (0.030, 30.0), ACID: (0.055, 40.0),
           WEAK_ACID: (0.045, 60.0), MTIN: (0.54, 120.0), MLEAD: (0.448, 120.0),
           MIRON: (1.87, 120.0), BLOOD: (0.058, 40.0)}
# THE RETAINED FILM. A liquid falling past a wall leaves a thin coat on it,
# which drains down slowly: the thickness that stays on each face it touches,
# metres, and the time (s) in which what is above that coat runs off. A
# viscous liquid keeps a thicker coat. UNVERIFIED (a game choice from Landau-
# Levich drainage films, 0.1-0.5 mm); a molten metal freezes before it drains.
RETAINED = {WATER: 2.5e-4, OIL: 5e-4, ACID: 3e-4, WEAK_ACID: 2.5e-4, MTIN: 5e-5,
            MLEAD: 5e-5, MIRON: 5e-5, BLOOD: 5e-4}
DRAIN_S = 60.0
# THE LEVEL TOLERANCE. Liquid runs from a cell to its neighbour while their
# depths differ by more than this, metres: a step smaller than it is the
# roughness of a floor, which a liquid does not feel. It was one millilitre
# whatever the voxel, so a pile left a cone whose height grew with the number
# of cells it spread over (a 10 L spill came out 5 mm deep at 5 cm). 0.02 mm
# keeps even a 25-cell cone under a sixth of a 3 mm film.
LEVEL_TOL_M = 2e-5
# SORPTION. A porous solid drinks from a wet face: the water taken in per m² of
# face is  i = S * sqrt(t)  (Wood Handbook, source 109: pine 10-16 along the
# grain, 1-7 across). A cell has no grain, and the faces of a plank, a tub or
# a post are nearly all SIDE grain, so S is the bottom of the across-grain
# range: 1. (At 7 a fifty-second-old vinegar tub drank 5 ml of its 3.4 L, and
# the cask test, which allows 1 ml, called it a leak.) The second number is the
# most water a gram of the solid holds, g/g. Rope and char are UNVERIFIED (a
# fibre wick takes faster than wood; char is the dark, hydrophobic end).
SORPTIVITY = {WOOD: (1.0, 0.6), ROPE: (15.0, 0.8), CHAR: (3.0, 0.5)}
# How fast each liquid wicks, against water (S goes as sqrt(sigma cos(theta)/
# viscosity), so a thick oil is slow). A metal soaks into nothing.
WICKING = {WATER: 1.0, WEAK_ACID: 1.0, BLOOD: 0.4, OIL: 0.15, ACID: 0.2}
# THE SPLASH. How fast a splash throws its droplets, m/s, up and out. UNVERIFIED:
# this is what the lattice threw at 5 cm (0.55 and 0.8 cells a tick), written in
# SI so that it no longer changes with the voxel; gravity on them is 9.81 m/s².
SPLASH_UP, SPLASH_SIDE = 1.1, 1.6


@functools.lru_cache(maxsize=8)
def _wet_tab(scale):
    """What a liquid leaves on a face, in ml per voxel face of a world of this
    scale: the film it comes to rest at, and the coat it keeps falling past."""
    face_ml = (0.1 * scale) ** 2 * 1e6                  # ml in a layer one metre thick on a voxel face
    film, ret = np.zeros(_NFL), np.zeros(_NFL)
    for f, (sigma, theta) in WETTING.items():
        lc = np.sqrt(sigma / (FLUID[f][0] * 1000.0 * GRAVITY))
        film[f] = 2.0 * lc * np.sin(np.radians(theta) / 2.0) * face_ml
        ret[f] = RETAINED[f] * face_ml
    return film, ret, LEVEL_TOL_M * face_ml


def _tables():
    """The material and fluid rows the kernels read, as arrays."""
    mixt = np.full((_NFL, _NFL), -1, np.int16)         # species after f_in joins f_at
    for a in range(_NFL):
        for b in range(_NFL):
            if b in (NOFLUID, a):                       # `_mix`, read off the table
                mixt[a, b] = a
            else:
                mixt[a, b] = int(MISCIBLE.get(frozenset({a, b}), -1))
    fd, fc, fv = (np.zeros(_NFL) for _ in range(3))    # density, heat capacity, spread
    for f in _FLIDS:
        fd[f], fc[f], fv[f] = FLUID[f][0], FLUID[f][1], FLUID[f][2]
    sch = np.zeros(NMAT_PAD)                           # c of each solid, by material id: the old
    sch[:NMAT] = [float(SOLID[m][1]) for m in range(NMAT)]   # twelve exactly as the dict gives them,
    sch[NMAT:] = _CSOLID_ARR[NMAT:]                    # the rest from the padded array (ICE, then AIR)
    n = max([len(v[0]) for v in _MIXHEAT.values()] + [1])
    has = np.zeros(_NFL, np.bool_)
    mhp, mhq = np.zeros((_NFL, n)), np.zeros((_NFL, n))
    mhn = np.zeros(_NFL, np.int64)
    for f, (p_at, q_at) in _MIXHEAT.items():
        has[f], mhn[f] = True, len(p_at)
        mhp[f, :len(p_at)], mhq[f, :len(q_at)] = p_at, q_at
    return (mixt, fd, fc, fv, sch, has, mhp, mhq, mhn,
            np.asarray(_FDENS_ARR, np.float32))


@njit(cache=True)
def _interp(x, xp, fp, n):
    """`np.interp` of one number, step for step as numpy does it."""
    if x != x:
        return x
    if x > xp[n - 1]:
        return fp[n - 1]
    if x < xp[0]:
        return fp[0]
    lo, hi = 0, n - 1                       # the last j with xp[j] <= x
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xp[mid] <= x:
            lo = mid
        else:
            hi = mid
    j = hi if xp[hi] <= x else lo
    if xp[j] == x:
        return fp[j]
    slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])
    r = slope * (x - xp[j]) + fp[j]
    if r != r:
        r = slope * (x - xp[j + 1]) + fp[j + 1]
        if r != r and fp[j] == fp[j + 1]:
            r = fp[j]
    return r


@njit(cache=True)
def _move(src, dst, ml, ml32, fl, fvol, fpot, E, smass, mat, cap32, cab32, tb):
    """`FluidsMixin._move_fluid` on flat arrays. `ml32` says the amount was a
    numpy float32 where the Python call had one (the flow law's), and a plain
    float otherwise (the head law's): the two round differently."""
    mixt, fd, fc, _fv, sch, has, mhp, mhq, mhn, _fdens = tb
    f = fl[src]
    fo = mixt[f, fl[dst]]
    if fo < 0:
        return
    # A source left with only dust (0.01 ml or less) sends that too, when the
    # destination has the room: dust is not thrown away (the ledger is exact).
    rest = fvol[src] - np.float32(ml)
    if rest > 0 and rest <= np.float32(0.01) and cap32 - fvol[dst] - np.float32(ml) >= rest:
        ml = np.float64(fvol[src])
    dens, c_f = fd[f], fc[f]
    m32 = np.float32(ml)
    # c_src: float32 all the way (a float32 times a Python float is float32)
    c_src = (np.float32(smass[src]) * np.float32(sch[mat[src]])
             + fvol[src] * np.float32(dens) * np.float32(c_f) + cab32)
    mx = np.float64(c_src)
    if 1e-6 > mx:
        mx = 1e-6
    if ml32:
        sh = np.float64((m32 * np.float32(dens) * np.float32(c_f)) / np.float32(mx))
    else:
        sh = (ml * dens * c_f) / mx
    if 1.0 < sh:
        sh = 1.0
    dE = np.float64(E[src]) * sh
    old = np.float64(fvol[dst])
    p_dst, p_src = np.float64(fpot[dst]), np.float64(fpot[src])
    t1 = fpot[dst] * np.float32(old)
    if ml32:
        fpot[dst] = (t1 + np.float32(p_src) * m32) / (np.float32(old) + m32)
    else:
        fpot[dst] = (t1 + np.float32(p_src * ml)) / np.float32(old + ml)
    if has[fo] and old > 0.0 and p_src != p_dst:
        # `_dilution_heat(fo, old, p_dst, ml, p_src, p)`; its (v1 + v2) is a
        # float32 when ml was one
        s12 = np.float64(np.float32(old) + m32) if ml32 else old + ml
        p = np.float64(fpot[dst])
        n = mhn[fo]
        h = (s12 * _interp(p, mhp[fo], mhq[fo], n)
             - old * _interp(p_dst, mhp[fo], mhq[fo], n)
             - ml * _interp(p_src, mhp[fo], mhq[fo], n))
        E[dst] = E[dst] + np.float32(h)
    fvol[src] = fvol[src] - np.float32(ml)
    E[src] = E[src] - np.float32(dE)
    if fvol[src] <= 0:
        fvol[src] = 0.0
        fl[src] = NOFLUID
        fpot[src] = 0.0
    fl[dst] = fo
    fvol[dst] = fvol[dst] + np.float32(ml)
    E[dst] = E[dst] + np.float32(dE)


@njit(cache=True, inline="always")
def _fall_ml(i, x, y, z, nx, ny, nz, mat, fl, fvol, ret, film, kdr, room):
    """How much of the fluid in this cell falls into the open cell below it,
    which has `room`: everything above the coat it keeps on the solid faces it
    touches (`RETAINED`), and a share of the coat itself, which runs off
    slowly (`DRAIN_S`) whatever runs past it. A liquid that falls into a puddle
    (the cell below holds a film or more) joins it whole: the coat is what is
    left on a dry wall."""
    n = 0
    if x + 1 < nx and mat[i + ny * nz] != AIR:
        n += 1
    if x > 0 and mat[i - ny * nz] != AIR:
        n += 1
    if y + 1 < ny and mat[i + nz] != AIR:
        n += 1
    if y > 0 and mat[i - nz] != AIR:
        n += 1
    if z + 1 < nz and mat[i + 1] != AIR:
        n += 1
    fv = fvol[i]
    if fvol[i - 1] >= np.float32(film[fl[i]]):                 # it falls into a puddle:
        n = 0                                                  # no coat is left on a wall
    r = np.float32(ret[fl[i]]) * np.float32(n)
    over = fv - r if fv > r else np.float32(0.0)               # what is more than the coat
    allowed = over + (fv if fv < r else r) * np.float32(kdr)   # and the coat's own share
    return room if room < allowed else allowed


@njit(cache=True, inline="always")
def _acts(i, x, y, z, nx, ny, nz, mat, fl, fvol, cap32, rest, film, ret, kdr, tol, tb):
    """Would the sweep move any fluid at this cell, in the state as it stands?"""
    mixt = tb[0]
    if fvol[i] <= 0 or mat[i] != AIR:
        return False
    if z > 0 and mat[i - 1] == AIR:
        if mixt[fl[i], fl[i - 1]] >= 0:
            room = cap32 - fvol[i - 1]
            if room > 0 and _fall_ml(i, x, y, z, nx, ny, nz, mat, fl, fvol, ret, film, kdr, room) > 0:
                return True
    if not (z == 0 or mat[i - 1] != AIR or fvol[i - 1] >= rest):
        return False
    fm = np.float32(film[fl[i]]) if (z == 0 or mat[i - 1] != AIR) else np.float32(0.0)
    for k in range(4):
        dx = 1 if k == 0 else (-1 if k == 1 else 0)
        dy = 1 if k == 2 else (-1 if k == 3 else 0)
        if not (0 <= x + dx < nx and 0 <= y + dy < ny):
            continue
        p = i + dx * ny * nz + dy * nz
        if mat[p] != AIR or mixt[fl[i], fl[p]] < 0:
            continue
        if fvol[i] - fvol[p] > np.float32(tol) and fvol[i] - fm > 0:
            return True
    return False


@njit(cache=True)
def _flow_quiet(cells, nx, ny, nz, mat, fl, fvol, cap, film, ret, kdr, tol, tb):
    """True when the layer swap has nothing to swap and no wet cell would move
    any fluid: the whole of `_law_flow` would change nothing."""
    cap32 = np.float32(cap)
    rest = np.float32(cap - 0.01)
    fdens = tb[9]
    for k in range(len(cells)):
        x, y, z = cells[k, 0], cells[k, 1], cells[k, 2]
        i = (x * ny + y) * nz + z
        if _acts(i, x, y, z, nx, ny, nz, mat, fl, fvol, cap32, rest, film, ret, kdr, tol, tb):
            return False
        if (z > 0 and fvol[i] > 1.0 and fvol[i - 1] > 1.0 and mat[i] == AIR
                and mat[i - 1] == AIR and fdens[fl[i]] > fdens[fl[i - 1]]):
            return False
    return True


@njit(cache=True)
def _swap_layers(nx, ny, nz, mat, fl, fvol, fpot, E, tb):
    """A heavier fluid directly above a lighter one: they swap. The pairs are
    found first, all of them, and written as numpy writes them: every upper
    cell, then every lower one (a cell that is both takes the lower's)."""
    fdens = tb[9]
    n = 0
    for x in range(nx):
        for y in range(ny):
            for z in range(nz - 1):
                lo = (x * ny + y) * nz + z
                up = lo + 1
                if (mat[up] == AIR and mat[lo] == AIR and fvol[up] > 1.0
                        and fvol[lo] > 1.0 and fdens[fl[up]] > fdens[fl[lo]]):
                    n += 1
    if n == 0:
        return
    los = np.empty(n, np.int64)
    n = 0
    for x in range(nx):
        for y in range(ny):
            for z in range(nz - 1):
                lo = (x * ny + y) * nz + z
                up = lo + 1
                if (mat[up] == AIR and mat[lo] == AIR and fvol[up] > 1.0
                        and fvol[lo] > 1.0 and fdens[fl[up]] > fdens[fl[lo]]):
                    los[n] = lo
                    n += 1
    ta = np.empty(n, np.float32)
    tb_ = np.empty(n, np.float32)
    for j in range(n):                                  # fl
        ta[j], tb_[j] = fl[los[j]], fl[los[j] + 1]
    for j in range(n):
        fl[los[j] + 1] = np.uint8(ta[j])
    for j in range(n):
        fl[los[j]] = np.uint8(tb_[j])
    for arr in (fvol, E, fpot):
        for j in range(n):
            ta[j], tb_[j] = arr[los[j]], arr[los[j] + 1]
        for j in range(n):
            arr[los[j] + 1] = ta[j]
        for j in range(n):
            arr[los[j]] = tb_[j]


@njit(cache=True)
def _sweep(cells, nx, ny, nz, mat, fl, fvol, fpot, E, smass, cap, cab32, film, ret, kdr, tol, tb):
    """The settling sweep of `_law_flow`, lowest cells first: fall, then
    spread by level difference, never below the film a liquid rests at on a
    solid floor."""
    mixt, _fd, _fc, fv = tb[0], tb[1], tb[2], tb[3]
    cap32 = np.float32(cap)
    rest = np.float32(cap - 0.01)
    four = np.float32(4.0)
    tol32 = np.float32(tol)
    for k in range(len(cells)):
        x, y, z = cells[k, 0], cells[k, 1], cells[k, 2]
        i = (x * ny + y) * nz + z
        if fvol[i] <= 0:
            continue
        if mat[i] != AIR:
            continue
        if z > 0 and mat[i - 1] == AIR:                      # 1a: fall
            if mixt[fl[i], fl[i - 1]] >= 0:
                room = cap32 - fvol[i - 1]
                if room > 0:
                    ml = _fall_ml(i, x, y, z, nx, ny, nz, mat, fl, fvol, ret, film, kdr, room)
                    if ml > 0:
                        _move(i, i - 1, ml, True, fl, fvol, fpot, E, smass, mat, cap32, cab32, tb)
        if fvol[i] <= 0:
            continue
        floor = z == 0 or mat[i - 1] != AIR
        if not (floor or fvol[i - 1] >= rest):
            continue
        visc = np.float32(fv[fl[i]])                         # 1b: spread by level
        fm = np.float32(film[fl[i]]) if floor else np.float32(0.0)
        for k2 in range(4):
            dx = 1 if k2 == 0 else (-1 if k2 == 1 else 0)
            dy = 1 if k2 == 2 else (-1 if k2 == 3 else 0)
            if not (0 <= x + dx < nx and 0 <= y + dy < ny):
                continue
            p = i + dx * ny * nz + dy * nz
            if mat[p] != AIR:
                continue
            if mixt[fl[i], fl[p]] < 0:
                continue
            diff = fvol[i] - fvol[p]
            if diff > tol32:
                give = fvol[i] - fm
                if give > 0:
                    a = diff * visc / four
                    r = cap32 - fvol[p]
                    ml = r if r < a else a
                    ml = give if give < ml else ml
                    _move(i, p, ml, True, fl, fvol, fpot, E, smass, mat, cap32, cab32, tb)


@njit(cache=True, inline="always")
def _mixed_at(j, fcomp, fl, fvol):
    """Does cell j hold a MIXED liquid, as `FieldsMixin._frow` reads it?"""
    return fcomp[j] > 0 and fl[j] != NOFLUID and fvol[j] > 0


@njit(cache=True)
def _sweep_stop(cells, start, nx, ny, nz, mat, fl, fvol, fpot, E, smass, fcomp, cap, cab32,
                film, ret, kdr, tol, tb):
    """`_sweep` for a world that has mixes (wave 13b, C2). The same arithmetic on the same
    cells in the same order, from index `start`, with one difference: it STOPS, before
    touching anything, at the first cell that holds a mixed liquid or has one for a
    neighbour it could hand fluid to (below it, or the four beside it). It returns that
    cell's index; the caller does that one cell in Python (`_flow_cell`, which carries the
    rows and the mass-exact blend) and calls again from the next. A cell with only pure
    cells around it moves exactly what `_sweep` moves, so the twin is the plain path bit
    for bit. Returns len(cells) when the sweep is done."""
    mixt, _fd, _fc, fv = tb[0], tb[1], tb[2], tb[3]
    cap32 = np.float32(cap)
    rest = np.float32(cap - 0.01)
    four = np.float32(4.0)
    tol32 = np.float32(tol)
    for k in range(start, len(cells)):
        x, y, z = cells[k, 0], cells[k, 1], cells[k, 2]
        i = (x * ny + y) * nz + z
        if fvol[i] <= 0:
            continue
        if mat[i] != AIR:
            continue
        if (_mixed_at(i, fcomp, fl, fvol) or (z > 0 and _mixed_at(i - 1, fcomp, fl, fvol))
                or (x + 1 < nx and _mixed_at(i + ny * nz, fcomp, fl, fvol))
                or (x > 0 and _mixed_at(i - ny * nz, fcomp, fl, fvol))
                or (y + 1 < ny and _mixed_at(i + nz, fcomp, fl, fvol))
                or (y > 0 and _mixed_at(i - nz, fcomp, fl, fvol))):
            return k
        if z > 0 and mat[i - 1] == AIR:                      # 1a: fall
            if mixt[fl[i], fl[i - 1]] >= 0:
                room = cap32 - fvol[i - 1]
                if room > 0:
                    ml = _fall_ml(i, x, y, z, nx, ny, nz, mat, fl, fvol, ret, film, kdr, room)
                    if ml > 0:
                        _move(i, i - 1, ml, True, fl, fvol, fpot, E, smass, mat, cap32, cab32, tb)
        if fvol[i] <= 0:
            continue
        floor = z == 0 or mat[i - 1] != AIR
        if not (floor or fvol[i - 1] >= rest):
            continue
        visc = np.float32(fv[fl[i]])                         # 1b: spread by level
        fm = np.float32(film[fl[i]]) if floor else np.float32(0.0)
        for k2 in range(4):
            dx = 1 if k2 == 0 else (-1 if k2 == 1 else 0)
            dy = 1 if k2 == 2 else (-1 if k2 == 3 else 0)
            if not (0 <= x + dx < nx and 0 <= y + dy < ny):
                continue
            p = i + dx * ny * nz + dy * nz
            if mat[p] != AIR:
                continue
            if mixt[fl[i], fl[p]] < 0:
                continue
            diff = fvol[i] - fvol[p]
            if diff > tol32:
                give = fvol[i] - fm
                if give > 0:
                    a = diff * visc / four
                    r = cap32 - fvol[p]
                    ml = r if r < a else a
                    ml = give if give < ml else ml
                    _move(i, p, ml, True, fl, fvol, fpot, E, smass, mat, cap32, cab32, tb)
    return len(cells)


@njit(cache=True)
def _find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


@njit(cache=True)
def _head(nx, ny, nz, mat, fl, fvol, fpot, E, smass, cap, cab32, ids, idxmap, tb):
    """`_law_head`, all the liquids in one pass. `idxmap` is scratch, -1
    everywhere on the way in and on the way out."""
    mixt = tb[0]
    sx, sy = ny * nz, nz
    cap32_half = np.float32(cap - 0.5)
    N = mat.size
    nw = 0
    for i in range(N):
        if fvol[i] > 1.0 and mat[i] == AIR:
            nw += 1
    wet = np.empty(nw, np.int64)
    flv = np.empty(nw, np.uint8)
    nw = 0
    for i in range(N):
        if fvol[i] > 1.0 and mat[i] == AIR:
            wet[nw] = i
            flv[nw] = fl[i]
            nw += 1
    colslot = np.full(nx * ny, -1, np.int64)
    for f in ids:
        m = 0
        for j in range(nw):
            if flv[j] == f:
                m += 1
        if m < 4:
            continue
        cells = np.empty(m, np.int64)
        m = 0
        for j in range(nw):
            if flv[j] == f:
                cells[m] = wet[j]
                m += 1
        for j in range(m):
            idxmap[cells[j]] = j
        parent = np.arange(m)
        for j in range(m):
            i = cells[j]
            x, y, z = i // sx, (i // sy) % ny, i % nz
            for k in range(3):
                if k == 0:
                    ok, nb = x + 1 < nx, i + sx
                elif k == 1:
                    ok, nb = y + 1 < ny, i + sy
                else:
                    ok, nb = z + 1 < nz, i + 1
                if ok and idxmap[nb] >= 0:
                    ri, rj = _find(parent, j), _find(parent, idxmap[nb])
                    if ri != rj:
                        parent[ri] = rj
        body = np.empty(m, np.int64)                  # bodies, in order of
        rootbody = np.full(m, -1, np.int64)           # their first cell
        nbody = 0
        for j in range(m):
            r = _find(parent, j)
            if rootbody[r] < 0:
                rootbody[r] = nbody
                nbody += 1
            body[j] = rootbody[r]
        start = np.zeros(nbody + 1, np.int64)         # cells of a body, in order
        for j in range(m):
            start[body[j] + 1] += 1
        for b in range(nbody):
            start[b + 1] += start[b]
        fill = start[:-1].copy()
        members = np.empty(m, np.int64)
        for j in range(m):
            members[fill[body[j]]] = j
            fill[body[j]] += 1
        for b in range(nbody):
            lo_, hi_ = start[b], start[b + 1]
            cx = np.empty(hi_ - lo_, np.int64)        # one slot per (x, y), in
            cy = np.empty(hi_ - lo_, np.int64)        # order of first sight
            cz = np.empty(hi_ - lo_, np.int64)
            nc = 0
            for q in range(lo_, hi_):
                i = cells[members[q]]
                x, y, z = i // sx, (i // sy) % ny, i % nz
                s = colslot[x * ny + y]
                if s < 0:
                    s = nc
                    colslot[x * ny + y] = s
                    cx[s], cy[s], cz[s] = x, y, z
                    nc += 1
                elif z > cz[s]:
                    cz[s] = z
            if nc >= 2:
                h = np.empty(nc, np.float64)
                tot = 0.0
                for s in range(nc):
                    h[s] = cz[s] + np.float64(fvol[(cx[s] * ny + cy[s]) * nz + cz[s]]) / cap
                    tot = h[s] if s == 0 else tot + h[s]
                mean = tot / nc
                ng = 0
                for s in range(nc):
                    if h[s] > mean + 0.1:
                        ng += 1
                give = np.empty(ng, np.int64)
                ng = 0
                for s in range(nc):
                    if h[s] > mean + 0.1:
                        give[ng] = s
                        ng += 1
                # descending by (h, x, y): ascending by y, x, h with stable
                # sorts, then read backwards
                o = np.argsort(cy[give], kind="mergesort")
                give = give[o]
                o = np.argsort(cx[give], kind="mergesort")
                give = give[o]
                o = np.argsort(h[give], kind="mergesort")
                give = give[o][::-1]
                nt = 0
                take = np.empty(nc, np.int64)
                tzs = np.empty(nc, np.int64)
                for s in range(nc):
                    if h[s] >= mean - 0.1:
                        continue
                    tz = cz[s]
                    j0 = (cx[s] * ny + cy[s]) * nz
                    if fvol[j0 + tz] >= cap32_half:
                        tz += 1
                        if (tz >= nz or mat[j0 + tz] != AIR
                                or not (fl[j0 + tz] == NOFLUID or fl[j0 + tz] == f)):
                            continue
                    take[nt] = s
                    tzs[nt] = tz
                    nt += 1
                take = take[:nt]
                tzs = tzs[:nt]
                o = np.argsort(h[take], kind="mergesort")
                take = take[o]
                tzs = tzs[o]
                for k in range(min(ng, nt)):
                    g, t = give[k], take[k]
                    gi = (cx[g] * ny + cy[g]) * nz + cz[g]
                    ti = (cx[t] * ny + cy[t]) * nz + tzs[k]
                    amount = 0.25 * cap * (h[g] - h[t])
                    a2 = np.float64(fvol[gi]) - 1.0
                    if a2 < amount:
                        amount = a2
                    a3 = cap - np.float64(fvol[ti])
                    if a3 < amount:
                        amount = a3
                    if amount > 0.5:
                        _move(gi, ti, amount, False, fl, fvol, fpot, E, smass, mat,
                              np.float32(cap), cab32, tb)
            for s in range(nc):
                colslot[cx[s] * ny + cy[s]] = -1
        for j in range(m):
            idxmap[cells[j]] = -1


_KEEP = weakref.WeakKeyDictionary()          # per world: the wet list and its order
_TB = []
_IDS = np.array(_FLIDS, np.int64)
_SCRATCH = {}


def _scratch(shape):
    """A -1-filled int32 map of the lattice for `_head`, which puts it back as
    it found it."""
    a = _SCRATCH.get(shape)
    if a is None:
        a = _SCRATCH[shape] = np.full(int(np.prod(shape)), -1, np.int32)
    return a


def _tb():
    if not _TB:
        _TB.append(_tables())
    return _TB[0]


def _sorb_tab():
    """The sorption rows as arrays: S by solid (0 where it soaks nothing), the
    most water it holds per gram, and how fast each liquid wicks."""
    sw = np.zeros((2, NMAT_PAD))                        # indexed by a cell's material id
    for m, (S, wmax) in SORPTIVITY.items():
        sw[0, m], sw[1, m] = S, wmax
    wick = np.zeros(_NFL)
    for f, v in WICKING.items():
        wick[f] = v
    return sw, wick


_SORB = _sorb_tab()


@njit(cache=True)
def _soak_pairs(cells, nx, ny, nz, mat, fl, fvol, sorbs, wick):
    """Every (wet open cell, porous solid beside it) pair, grouped by the
    direction from the wet cell to the solid; within one direction the cells
    are all different, so a whole group can be worked at once. Returns the
    two cell lists and where each direction's group starts."""
    sx, sy = ny * nz, nz
    n = 0
    for d in range(6):
        for k in range(len(cells)):
            x, y, z = cells[k, 0], cells[k, 1], cells[k, 2]
            i = (x * ny + y) * nz + z
            if mat[i] != AIR or fvol[i] <= 0 or wick[fl[i]] <= 0:
                continue
            if d == 0:
                ok, j = x + 1 < nx, i + sx
            elif d == 1:
                ok, j = x > 0, i - sx
            elif d == 2:
                ok, j = y + 1 < ny, i + sy
            elif d == 3:
                ok, j = y > 0, i - sy
            elif d == 4:
                ok, j = z + 1 < nz, i + 1
            else:
                ok, j = z > 0, i - 1
            if ok and sorbs[mat[j]] > 0:
                n += 1
    a = np.empty(n, np.int64)
    b = np.empty(n, np.int64)
    start = np.zeros(7, np.int64)
    n = 0
    for d in range(6):
        for k in range(len(cells)):
            x, y, z = cells[k, 0], cells[k, 1], cells[k, 2]
            i = (x * ny + y) * nz + z
            if mat[i] != AIR or fvol[i] <= 0 or wick[fl[i]] <= 0:
                continue
            if d == 0:
                ok, j = x + 1 < nx, i + sx
            elif d == 1:
                ok, j = x > 0, i - sx
            elif d == 2:
                ok, j = y + 1 < ny, i + sy
            elif d == 3:
                ok, j = y > 0, i - sy
            elif d == 4:
                ok, j = z + 1 < nz, i + 1
            else:
                ok, j = z > 0, i - 1
            if ok and sorbs[mat[j]] > 0:
                a[n], b[n] = i, j
                n += 1
        start[d + 1] = n
    return a, b, start


class FluidsMixin:

    _KINDS = ("uint8", "uint8", "float32", "float32", "float32", "float32")

    def _compiled_fluids(self):
        """Run the compiled sweeps (`_sweep`, `_head`)? Only with numba, with
        quiet things skipped (`sleep_regions`), and on the arrays the kernels
        were written for."""
        if not (self.fused and HAVE_NUMBA and self._sleeps()):
            return False
        if self.mixes_live:
            # THE COMPILED SWEEPS CARRY NO MIX (T1, wave 13a). A world with a
            # mix in it takes the Python path, which is the definition the
            # compiled one is tested against, and which carries `fcomp` and
            # compares the density of the row. Both paths therefore give the
            # same answer by construction. (A compiled twin that carries the
            # mix, with the table's stop-and-resume blend, is the open
            # speedup: see the lane report.)
            return False
        a = (self.mat, self.fl, self.fvol, self.fpot, self.E, self.smass)
        return all(x.flags.c_contiguous and x.dtype.name == k
                   for x, k in zip(a, self._KINDS))

    def _compiled_mixed(self):
        """Run the settling sweep of a world that HAS mixes with the compiled twin
        (`_sweep_stop`)? The cells with no mixed liquid near them are swept compiled; the
        others, one by one, by `_flow_cell`. The layer swap and the head law stay in
        Python (they compare the density of each cell's own row)."""
        if not (self.fused and HAVE_NUMBA and self._sleeps() and self.mixes_live):
            return False
        if getattr(self, "mix_compiled", True) is False:         # (a switch for the test that
            return False                                         # compares the two paths)
        a = (self.mat, self.fl, self.fvol, self.fpot, self.E, self.smass, self.fcomp)
        return all(x.flags.c_contiguous and x.dtype.name == k
                   for x, k in zip(a, self._KINDS + ("uint32",)))

    # ── the four laws ────────────────────────────────────────────────────────
    def _move_fluid(self, src, dst, ml):
        """Move ml of fluid src→dst, carrying its share of the source's heat and
        its potency: the destination blends by volume, so strong acid dripping
        into a water pool leaves a paler, weaker acid — never a hidden flag.
        And a blend of two strengths of vitriol gets HOT (`_dilution_heat`).

        A move that involves a MIX (T1: either cell has an `fcomp`) is the
        mass-exact blend instead (`_move_mixed`); a pure move into a pure cell
        is the code below, bit for bit."""
        if self.mixes_live and (self._frow(src) or self._frow(dst)):
            return self._move_mixed(src, dst, ml)
        f = self.fl[src]
        f_out = self._mix(f, self.fl[dst])
        if f_out is None:
            return
        # A source left with only dust (0.01 ml or less) sends that too, when the
        # destination has the room: dust is not thrown away (the ledger is exact).
        rest = self.fvol[src] - np.float32(ml)
        if 0 < rest <= np.float32(0.01) and np.float32(self.cap) - self.fvol[dst] - np.float32(ml) >= rest:
            ml = type(ml)(self.fvol[src])
        dens, c_f = FLUID[f][0], FLUID[f][1]
        c_src = (self.smass[src] * SOLID[int(self.mat[src])][1]        # scalar capacity of
                 + self.fvol[src] * dens * c_f + self.c_airbase)       # ONE voxel — never
        share = (ml * dens * c_f) / max(float(c_src), 1e-6)            # the whole lattice
        dE = float(self.E[src]) * min(share, 1.0)
        old = float(self.fvol[dst])
        p_dst, p_src = float(self.fpot[dst]), float(self.fpot[src])
        self.fpot[dst] = (self.fpot[dst] * old + float(self.fpot[src]) * ml) / (old + ml)
        if f_out in _MIXHEAT and old > 0.0 and p_src != p_dst:
            self.E[dst] += self._dilution_heat(f_out, old, p_dst, ml, p_src,
                                               float(self.fpot[dst]))
        self.fvol[src] -= ml
        self.E[src] -= dE
        if self.fvol[src] <= 0:
            self.fvol[src] = 0.0
            self.fl[src] = NOFLUID
            self.fpot[src] = 0.0
        self.fl[dst] = f_out
        self.fvol[dst] += ml
        self.E[dst] += dE

    def _move_mixed(self, src, dst, ml):
        """`_move_fluid` for a hand-over where either side is a mix (T1 4.7).
        The fluid carries its row. Mass is exact and volume follows
        (`_fluid_plan`): what leaves the source is what float32 really took
        from it, and the destination gets exactly that mass. Heat travels with
        the share of the source's heat capacity, as in the pure move. The
        source keeps its row; if it is emptied its row goes with it."""
        T = self.mixes
        f_out = self._mix_cells(src, dst)
        if f_out is None:
            return
        f = int(self.fl[src])
        old_s = float(self.fvol[src])
        if old_s <= 0:
            return
        rest = self.fvol[src] - np.float32(ml)
        if 0 < rest <= np.float32(0.01) and np.float32(self.cap) - self.fvol[dst] - np.float32(ml) >= rest:
            ml = type(ml)(self.fvol[src])         # dust goes with it (as the pure move)
        r_src = self._fluid_row(src)
        src_mixed = self._frow(src) > 0
        plan = None
        ml_try = float(ml)
        for _ in range(2):                         # the second pass only if the cell was too full
            new_s = np.float32(old_s) - np.float32(ml_try)
            got = old_s - float(new_s) if new_s > 0 else old_s
            plan = self._fluid_plan(dst, r_src, got)
            if plan is None:
                return
            if plan[0] >= got * (1.0 - 1e-6):
                break
            ml_try = plan[0]
        got = plan[0] if plan[0] < got else got
        rho_s = float(T.prop("rho", np.array([r_src]))[0]) / 1000.0
        c_f = float(T.prop("c", np.array([r_src]))[0]) if src_mixed else FLUID[f][1]
        dens = rho_s if src_mixed else FLUID[f][0]
        c_src = (self.smass[src] * _CSOLID_ARR[int(self.mat[src])]
                 + self.fvol[src] * dens * c_f + self.c_airbase)
        share = (got * dens * c_f) / max(float(c_src), 1e-6)
        dE = float(self.E[src]) * min(share, 1.0)
        self._fluid_commit(dst, (got,) + tuple(plan[1:]), f)
        if old_s - got <= 0:
            self.fvol[src] = 0.0
            self.fl[src] = NOFLUID
            self.fpot[src] = 0.0
            self.fcomp[src] = 0
        else:
            v = np.float32(old_s - got)
            self.fvol[src] = v
            if src_mixed:                          # what float32 could not keep of the source
                ent = T.entries(np.array([r_src]))
                T.book("f32", ent.species, ent.frac * ((old_s - got) - float(v)) * rho_s)
        self.E[src] -= dE
        self.E[dst] += dE

    @staticmethod
    def _dilution_heat(f, v1, p1, v2, p2, p):
        """Joules given off when v1 ml at potency p1 and v2 ml at p2 of acid f
        become one liquid at potency p: the heat of mixing the blend holds,
        less what the two parts already held (materials.DILUTION). Water is
        potency 0, so this is also water poured into vitriol, or vitriol into
        water: about 0.86 kJ per ml of stock at 1:1 (realism C11). It always
        comes out >= 0, because the table is read as a concave curve."""
        p_at, q_at = _MIXHEAT[f]
        return float((v1 + v2) * np.interp(p, p_at, q_at)
                     - v1 * np.interp(p1, p_at, q_at) - v2 * np.interp(p2, p_at, q_at))

    def _fly_drops(self):
        """Ballistic fluid parcels (the Noita trick): a splash throws real droplets
        that ARC through the air under gravity and rejoin the grid where they land.
        Deterministic — direction comes from the splash geometry, never RNG."""
        if not self.drops:
            return
        nx, ny, nz = self.shape
        kept = []
        g = GRAVITY * TICK_S ** 2 / (0.1 * self.scale)       # cells per tick², from 9.81 m/s²
        for d in self.drops:
            d[5] -= g                                    # gravity, in SI
            d[0] += d[3]; d[1] += d[4]; d[2] += d[5]
            x = min(max(int(round(d[0])), 0), nx - 1)
            y = min(max(int(round(d[1])), 0), ny - 1)
            z = min(max(int(round(d[2])), 0), nz - 1)
            landed = (self.mat[x, y, z] != AIR or self.fvol[x, y, z] > 1.0 or z == 0
                      or self.mat[x, y, z - 1] != AIR
                      or self.fvol[x, y, z - 1] > 0.7 * self.cap)
            if not landed:
                kept.append(d)
                continue
            zz = z                                       # deposit in the first open cell
            if self.mixes_live:
                self._deposit_drop(d, x, y, z, kept)
                continue
            while zz < nz - 1 and (self.mat[x, y, zz] != AIR
                                   or self._mix(d[7], self.fl[x, y, zz]) is None):
                zz += 1
            fo = self._mix(d[7], self.fl[x, y, zz])
            if self.mat[x, y, zz] == AIR and fo is not None:
                room = self.cap - self.fvol[x, y, zz]
                add = min(d[6], max(room, 0.0))
                if add > 0:
                    old = float(self.fvol[x, y, zz])
                    p_at = float(self.fpot[x, y, zz])
                    self.fpot[x, y, zz] = (self.fpot[x, y, zz] * old
                                           + d[8] * add) / (old + add)
                    if fo in _MIXHEAT and old > 0.0 and d[8] != p_at:
                        self.E[x, y, zz] += self._dilution_heat(
                            fo, old, p_at, add, d[8], float(self.fpot[x, y, zz]))
                    self.fl[x, y, zz] = fo
                    self.fvol[x, y, zz] += add
                    d[6] -= add
            if d[6] > 0.5:                               # remainder keeps falling
                d[0], d[1], d[2] = x, y, min(zz + 1, nz - 1)
                d[3] = d[4] = d[5] = 0.0
                kept.append(d)
        self.drops = kept

    def _deposit_drop(self, d, x, y, z, kept):
        """`_fly_drops` where it lands, in a world with mixes (T1). A parcel
        carries its fluid mix as a tenth entry (`d[9]`, a table row; a pure
        parcel has nine entries, as ever), and joins the cell by the exact blend
        (`_fluid_plan`). Same walk upward to the first cell that will take it,
        same remainder rule. Does the work of the pure branch, for a world where
        a cell or a parcel may be a mix."""
        nz = self.shape[2]
        row = int(d[9]) if len(d) > 9 and d[9] else 64 + int(d[7])
        mixed = row >= 128
        zz = z
        while zz < nz - 1 and (self.mat[x, y, zz] != AIR or self._mix_ids(
                d[7], mixed, self.fl[x, y, zz], self._frow((x, y, zz)) > 0) is None):
            zz += 1
        at = (x, y, zz)
        fo = self._mix_ids(d[7], mixed, self.fl[at], self._frow(at) > 0)
        if self.mat[at] == AIR and fo is not None and not mixed and not self._frow(at):
            # a pure parcel into a pure cell: the old arithmetic, unchanged
            room = self.cap - self.fvol[at]
            add = min(d[6], max(room, 0.0))
            if add > 0:
                old = float(self.fvol[at])
                p_at = float(self.fpot[at])
                self.fpot[at] = (self.fpot[at] * old + d[8] * add) / (old + add)
                if fo in _MIXHEAT and old > 0.0 and d[8] != p_at:
                    self.E[at] += self._dilution_heat(fo, old, p_at, add, d[8], float(self.fpot[at]))
                self.fl[at] = fo
                self.fvol[at] += add
                d[6] -= add
        elif self.mat[at] == AIR and fo is not None:
            old = float(self.fvol[at])
            plan = self._fluid_plan(at, row, float(d[6])) if self.cap - old > 0 else None
            if plan is not None and plan[0] > 0:
                p_at = float(self.fpot[at])
                add = plan[0]
                self.fpot[at] = (self.fpot[at] * old + d[8] * add) / (old + add)
                if fo in _MIXHEAT and old > 0.0 and d[8] != p_at:
                    self.E[at] += self._dilution_heat(fo, old, p_at, add, d[8], float(self.fpot[at]))
                self._fluid_commit(at, plan, d[7])
                d[6] -= add
        if d[6] > 0.5:                                   # remainder keeps falling
            d[0], d[1], d[2] = x, y, min(zz + 1, nz - 1)
            d[3] = d[4] = d[5] = 0.0
            kept.append(d)

    def _law_head(self):
        """PRESSURE, hydrostatic half: p = rho*g*h. Within ONE connected body of
        the same liquid, the free surface seeks a single level — columns above
        the body's mean surface bleed into columns below it. This is why water
        poured into one tank RISES in the tank connected to it by a pipe.
        v1 limit: the transfer rate is fixed — a narrow pipe does not yet
        throttle it (that needs the full momentum field).

        ONE PASS FOR ALL THE LIQUIDS (`sleep_regions`): the cells that hold
        any liquid are found once, and each liquid takes its own from that
        list, in the same order. Exact, because what one liquid moves here
        is only ever its own cells and empty ones, which no other liquid's
        list holds before or after."""
        if self._compiled_fluids():
            ids = _IDS
            idxmap = _scratch(self.shape)
            nx, ny, nz = self.shape
            mat, fl, fvol, fpot, E, smass = (a.reshape(-1) for a in (
                self.mat, self.fl, self.fvol, self.fpot, self.E, self.smass))
            _head(nx, ny, nz, mat, fl, fvol, fpot, E, smass, self.cap,
                  np.float32(self.c_airbase), ids, idxmap, _tb())
            return
        wet = flv = None
        if self._sleeps():
            wet = np.flatnonzero((self.fvol > 1.0) & (self.mat == AIR))
            flv = self.fl.reshape(-1)[wet]
        for f in _FLIDS:
            if wet is not None:
                cells = np.stack(np.unravel_index(wet[flv == f], self.shape), axis=1)
            else:
                cells = np.argwhere((self.fl == f) & (self.fvol > 1.0) & (self.mat == AIR))
            if len(cells) < 4:
                continue
            index = {(int(c[0]), int(c[1]), int(c[2])): i for i, c in enumerate(cells)}
            parent = list(range(len(cells)))

            def find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            for (x, y, z), i in index.items():
                for nb in ((x + 1, y, z), (x, y + 1, z), (x, y, z + 1)):
                    j = index.get(nb)
                    if j is not None:
                        ri, rj = find(i), find(j)
                        if ri != rj:
                            parent[ri] = rj
            bodies = {}
            for (x, y, z), i in index.items():
                bodies.setdefault(find(i), {}).setdefault((x, y), []).append(z)
            for cols in bodies.values():
                if len(cols) < 2:
                    continue
                surf = {}
                for (x, y), zlist in cols.items():
                    zt = max(zlist)
                    surf[(x, y)] = zt + float(self.fvol[x, y, zt]) / self.cap
                mean = sum(surf.values()) / len(surf)
                give = sorted(((h, c) for c, h in surf.items() if h > mean + 0.1),
                              reverse=True)
                take = []                                # only columns that CAN rise —
                for (x, y), h in surf.items():           # a roofed pipe is not a taker
                    if h >= mean - 0.1:
                        continue
                    tz = max(cols[(x, y)])
                    if self.fvol[x, y, tz] >= self.cap - 0.5:
                        tz += 1
                        if tz >= self.shape[2] or self.mat[x, y, tz] != AIR \
                                or self.fl[x, y, tz] not in (NOFLUID, f):
                            continue                     # sealed under a roof: skip
                    take.append((h, (x, y), tz))
                take.sort(key=lambda t: t[0])
                for (hg, (gx, gy)), (ht, (tx, ty), tz) in zip(give, take):
                    gz = max(cols[(gx, gy)])
                    amount = min(0.25 * self.cap * (hg - ht),
                                 float(self.fvol[gx, gy, gz]) - 1.0,
                                 self.cap - float(self.fvol[tx, ty, tz]))
                    if amount > 0.5:
                        self._move_fluid((gx, gy, gz), (tx, ty, tz), amount)

    def _flow_compiled(self):
        """`_law_flow` with the settling sweep compiled (see the top of this
        file). The wet cells and their order are kept while the SET of wet
        cells is the same; a world where no cell would move anything is not
        swept at all."""
        nx, ny, nz = self.shape
        wet = self.fvol > 0
        kept = _KEEP.get(self)
        if kept is None or kept[0].shape != wet.shape or not np.array_equal(kept[0], wet):
            active = np.argwhere(wet)
            cells = active[np.argsort(active[:, 2])]            # settle lowest first
            _KEEP[self] = kept = (wet, cells)
        cells = kept[1]
        tb = _tb()
        film, ret, tol = _wet_tab(self.scale)
        kdr = TICK_S / DRAIN_S
        mat, fl, fvol = (a.reshape(-1) for a in (self.mat, self.fl, self.fvol))
        if not _flow_quiet(cells, nx, ny, nz, mat, fl, fvol, self.cap, film, ret, kdr, tol, tb):
            fpot, E, smass = (a.reshape(-1) for a in (self.fpot, self.E, self.smass))
            _swap_layers(nx, ny, nz, mat, fl, fvol, fpot, E, tb)
            _sweep(cells, nx, ny, nz, mat, fl, fvol, fpot, E, smass, self.cap,
                   np.float32(self.c_airbase), film, ret, kdr, tol, tb)
        self._law_soak(cells)

    def _law_flow(self):
        self._fly_drops()
        if self._compiled_fluids():
            self._flow_compiled()
            return
        # density layering: a heavier fluid directly ABOVE a lighter one sinks — they
        # swap. This is WHY water fails against an oil fire: it slips underneath and
        # the burning oil rides up on top of it (no rule about fires anywhere).
        up, lo = (slice(None), slice(None), slice(1, None)), (slice(None), slice(None), slice(None, -1))
        # ONLY THE BOX THE LIQUID IS IN (`sleep_regions`): a swap needs liquid
        # above and below, and swapping two cells' liquid leaves the set of wet
        # cells as it was, so it can be found first and the box drawn round it
        box = (slice(None),) * 3
        active = None
        if self._sleeps():
            active = np.argwhere(self.fvol > 0)
            if len(active):
                a0, a1 = active.min(axis=0), active.max(axis=0) + 1
                box = tuple(slice(int(p), int(q)) for p, q in zip(a0, a1))
        fl_b, fvol_b, mat_b = self.fl[box], self.fvol[box], self.mat[box]
        # (a world with mixes compares the density of each cell's own row, so
        # brine goes under fresh water though both are WATER; any other world
        # gets `_FDENS_ARR[fl]`, the old expression)
        dens = self._fdens_grid()[box] if self.mixes_live else _FDENS_ARR[fl_b]
        m = ((mat_b[up] == AIR) & (mat_b[lo] == AIR)
             & (fvol_b[up] > 1.0) & (fvol_b[lo] > 1.0) & (dens[up] > dens[lo]))
        if m.any():
            for arr in ((self.fl, self.fvol, self.E, self.fpot) if not self.mixes_live
                        else (self.fl, self.fvol, self.E, self.fpot, self.fcomp)):
                a, b = arr[box][up], arr[box][lo]
                a[m], b[m] = b[m], a[m].copy()
        nx, ny, nz = self.shape
        if active is None:
            active = np.argwhere(self.fvol > 0)
        order = np.argsort(active[:, 2])                    # settle lowest first
        film, ret, tol = _wet_tab(self.scale)
        kdr = np.float32(TICK_S / DRAIN_S)
        tol32 = np.float32(tol)
        cells = active[order]
        if self._compiled_mixed():
            self._sweep_mixed(cells, nx, ny, film, ret, tol32, kdr, tol)
        else:
            for x, y, z in cells:
                self._flow_cell(x, y, z, nx, ny, film, ret, tol32, kdr)
        self._law_soak(active)

    def _flow_cell(self, x, y, z, nx, ny, film, ret, tol32, kdr):
        """ONE CELL of the settling sweep of `_law_flow`: fall, then spread by level
        difference. The loop body, kept as a method so a world with mixes can do the
        cells that touch a mix here and the rest compiled (`_sweep_stop`)."""
        nz = self.shape[2]
        if self.fvol[x, y, z] <= 0:
            return
        if self.mat[x, y, z] != AIR:                # moisture held INSIDE a solid
            return                                # (fresh leaves, soaked wood)
                                                    # stays put — it only boils away
        if z > 0 and self.mat[x, y, z - 1] == AIR:      # 1a: fall
            below = (x, y, z - 1)
            if self._mix_cells((x, y, z), below) is not None:
                room = self.cap - self.fvol[below]
                if room > 0:
                    # all of it, less the coat it keeps on the solid faces it
                    # touches, and the coat runs off slowly too
                    # (`RETAINED`, `DRAIN_S`)
                    n = sum(1 for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                                        (0, 0, 1))
                            if 0 <= x + d[0] < nx and 0 <= y + d[1] < ny
                            and z + d[2] < nz and self.mat[x + d[0], y + d[1], z + d[2]] != AIR)
                    fv = self.fvol[x, y, z]
                    if self.fvol[below] >= np.float32(film[self.fl[x, y, z]]):
                        n = 0                       # into a puddle: no coat on a wall
                    r = np.float32(ret[self.fl[x, y, z]]) * np.float32(n)
                    over = fv - r if fv > r else np.float32(0.0)
                    ml = min(over + (fv if fv < r else r) * kdr, room)
                    if ml > 0:
                        self._move_fluid((x, y, z), below, ml)
        if self.fvol[x, y, z] <= 0:
            return
        floor = z == 0 or self.mat[x, y, z - 1] != AIR
        if not (floor or self.fvol[x, y, z - 1] >= self.cap - 0.01):     # resting
            return
        visc = FLUID[self.fl[x, y, z]][2]               # 1b: spread by level difference
        # ...never below the film this liquid rests at on a solid floor
        fm = np.float32(film[self.fl[x, y, z]]) if floor else np.float32(0.0)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p = (x + dx, y + dy, z)
            if not (0 <= p[0] < nx and 0 <= p[1] < ny) or self.mat[p] != AIR:
                continue
            if self._mix_cells((x, y, z), p) is None:
                continue
            diff = self.fvol[x, y, z] - self.fvol[p]
            if diff > tol32:
                give = self.fvol[x, y, z] - fm
                if give > 0:
                    self._move_fluid((x, y, z), p, min(diff * visc / 4.0,
                                                       self.cap - self.fvol[p], give))

    def _sweep_mixed(self, cells, nx, ny, film, ret, tol32, kdr, tol):
        """The settling sweep for a world with mixes: compiled where no mixed liquid is near,
        Python (`_flow_cell`) for the cells that touch one. Same cells, same order, same
        arithmetic as the plain loop, so the same bits (tests/test_c2_mixtures_on.py)."""
        nz = self.shape[2]
        tb = _tb()
        mat, fl, fvol, fpot, E, smass, fcomp = (a.reshape(-1) for a in (
            self.mat, self.fl, self.fvol, self.fpot, self.E, self.smass, self.fcomp))
        cab32 = np.float32(self.c_airbase)
        n, k = len(cells), 0
        while k < n:
            k = _sweep_stop(cells, k, nx, ny, nz, mat, fl, fvol, fpot, E, smass, fcomp, self.cap,
                            cab32, film, ret, TICK_S / DRAIN_S, tol, tb)
            if k >= n:
                break
            x, y, z = cells[k]
            self._flow_cell(x, y, z, nx, ny, film, ret, tol32, kdr)
            k += 1

    def _law_soak(self, cells):
        """SORPTION: a porous solid drinks from a wet face. The water taken in
        per m² of face follows i = S * sqrt(t), which is the same as saying a
        face that has taken m has m_next = sqrt(m² + S² dt): the clock is the
        water the cell already holds (spread over the faces that are wet), so
        a face that dried takes water fast again. The water goes into the
        solid's own cell, as moisture held INSIDE it: it carries its heat, the
        boil law takes it out at 100 °C, and it is counted in the fluid
        ledger where it sits. A liquid that would mix with what the cell holds
        (an acid into acid-wet wood) is passed over: only the same liquid, or
        a dry cell, takes it, so no heat of mixing is owed. A solid hotter than
        the boiling point (`BOIL`) takes none: liquid water cannot stay in it,
        and a face that hot is dry (the water would only be boiled at once).

        `cells` is the list of cells holding any fluid when the law began. The
        work is one pass over it to find the pairs; a pond that touches no
        porous solid finds none and stops there."""
        if not len(cells):
            return
        sw, wick = _SORB
        if not (sw[0][self.mat] > 0).any():              # nothing here soaks anything
            return
        nx, ny, nz = self.shape
        a_all, b_all, start = _soak_pairs(
            np.ascontiguousarray(cells), nx, ny, nz, self.mat.reshape(-1),
            self.fl.reshape(-1), self.fvol.reshape(-1), sw[0], wick)
        if not len(a_all):
            return
        fl, fvol, fpot, E, smass, mat = (a.reshape(-1) for a in (
            self.fl, self.fvol, self.fpot, self.E, self.smass, self.mat))
        tb = _tb()
        hcap = self.heat_capacity().reshape(-1)
        dens_t, c_t, sch, has = tb[1], tb[2], tb[4], tb[5]
        A = (0.1 * self.scale) ** 2                              # m² of one voxel face
        sx, sy = ny * nz, nz
        offs = (sx, -sx, sy, -sy, 1, -1)
        for d in range(6):
            a, b = a_all[start[d]:start[d + 1]], b_all[start[d]:start[d + 1]]
            if not len(a):
                continue
            f = fl[a]                                   # (an earlier pass may have emptied it)
            ok = (f != NOFLUID) & (fvol[a] > 0) & ((fl[b] == NOFLUID) | (fl[b] == f)) \
                & ~(has[f] & (fvol[b] > 0))
            ok &= AMBIENT + E[b] / hcap[b] < BOIL       # a solid past boiling holds no liquid
            if self.mixes_live:                         # D6: soaking writes fl and fvol with no
                ok &= ((self.fcomp.reshape(-1)[a] == 0) & (self.fcomp.reshape(-1)[b] == 0)
                       & (self.scomp.reshape(-1)[b] == 0))  # row, so mixed cells are passed over
            a, b, f = a[ok], b[ok], f[ok]
            if not len(a):
                continue
            dens = dens_t[f]
            # the faces of the solid cell that are wet, with this liquid
            n_w = np.zeros(len(b))
            bx, by, bz = b // sx, (b // sy) % ny, b % nz
            for o, (ex, ey, ez) in zip(offs, ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                                              (0, 0, 1), (0, 0, -1))):
                inside = ((bx + ex >= 0) & (bx + ex < nx) & (by + ey >= 0) & (by + ey < ny)
                          & (bz + ez >= 0) & (bz + ez < nz))
                nb = np.where(inside, b + o, 0)
                n_w += inside & (mat[nb] == AIR) & (fvol[nb] > 0) & (fl[nb] == f)
            m_face = fvol[b] * dens / (A * np.maximum(n_w, 1.0))          # g/m² so far
            S = sw[0][mat[b]] * wick[f]
            m_next = np.sqrt(m_face ** 2 + (S * S) * TICK_S)
            ml = (m_next - m_face) * A / dens                             # one face's share
            room = sw[1][mat[b]] * smass[b] / dens - fvol[b]              # what it can hold
            room = np.minimum(room, self.cap - fvol[b])
            ml = np.minimum(np.minimum(ml, fvol[a]), room)
            rest = fvol[a] - ml                                            # dust goes with it
            ml = np.where((rest > 0) & (rest <= 0.01) & (room - ml >= rest), fvol[a], ml)
            go = ml > 0
            if not go.any():
                continue
            a, b, f, ml, dens = a[go], b[go], f[go], ml[go], dens[go]
            ml32 = ml.astype(np.float32)
            c_src = (smass[a] * sch[mat[a]].astype(np.float32)
                     + fvol[a] * dens.astype(np.float32) * c_t[f].astype(np.float32)
                     + np.float32(self.c_airbase))
            share = ml32 * dens.astype(np.float32) * c_t[f].astype(np.float32) \
                / np.maximum(c_src, np.float32(1e-6))
            dE = E[a] * np.minimum(share, np.float32(1.0))
            old = fvol[b].copy()
            fpot[b] = (fpot[b] * old + fpot[a] * ml32) / (old + ml32)
            fvol[a] -= ml32
            E[a] -= dE
            gone = fvol[a] <= 0                                           # nothing left
            fvol[a[gone]], fl[a[gone]], fpot[a[gone]] = 0.0, NOFLUID, 0.0
            fl[b] = f
            fvol[b] += ml32
            E[b] += dE

    def skin_water(self):
        """ml of liquid held in the SKIN of each porous solid cell: the outer
        layer a flame reaches (its depth is the material's thermal skin,
        `_SKIN_DEPTH_ARR`) fills first, and what the skin cannot hold is in
        the core. A fire reads it: a skin holding water cannot pass 100 °C
        while the water lasts. Zero where the cell soaks nothing."""
        sw, _wick = _SORB
        S = sw[0][self.mat] > 0
        out = np.zeros(self.shape, np.float32)
        if not S.any():
            return out
        air = self.mat == AIR
        n_out = np.zeros(self.shape, np.int8)                  # faces open to air
        for ax in range(3):
            lo = [slice(None)] * 3; hi = [slice(None)] * 3
            lo[ax], hi[ax] = slice(None, -1), slice(1, None)
            lo, hi = tuple(lo), tuple(hi)
            n_out[lo] += air[hi]
            n_out[hi] += air[lo]
        dx = 0.1 * self.scale
        share = np.clip(n_out * _SKIN_DEPTH_ARR[self.mat] / dx, 0.0, 1.0)
        dens = (self._fdens_grid() if self.mixes_live
                else np.asarray(_FDENS_ARR, np.float32)[self.fl])
        held = np.where(dens > 0, sw[1][self.mat] * self.smass * share / np.maximum(dens, 1e-9), 0.0)
        return np.where(S, np.minimum(self.fvol, held), 0.0).astype(np.float32)

    def _law_stir(self):
        """Potency evens out WITHIN a standing pool — reaction heat and density
        currents stir a real liquid, so the layer working the wood face is fed
        by the potent bulk above it instead of exhausting and playing dead.
        Evening out two strengths of vitriol is diluting one of them, so it
        gives off the heat of dilution, half into each cell."""
        for f in _REACTIVE:
            liq = (self.fl == f) & (self.fvol > 1.0) & (self.mat == AIR)
            if self.mixes_live:                      # D6: potency is fpot's, not a mix's
                liq &= self.fcomp == 0
            if not liq.any():
                continue
            for axis in range(3):
                a = [slice(None)] * 3; b = [slice(None)] * 3
                a[axis], b[axis] = slice(None, -1), slice(1, None)
                a, b = tuple(a), tuple(b)
                m = liq[a] & liq[b]
                if not m.any():
                    continue
                lo = np.minimum(self.fvol[a][m], self.fvol[b][m])
                d = 0.1 * (self.fpot[a][m] - self.fpot[b][m]) * lo
                if f in _MIXHEAT and d.any():
                    va, vb = self.fvol[a][m], self.fvol[b][m]
                    pa, pb = self.fpot[a][m], self.fpot[b][m]
                    p_at, q_at = _MIXHEAT[f]
                    give = 0.5 * (va * (np.interp(pa - d / va, p_at, q_at)
                                        - np.interp(pa, p_at, q_at))
                                  + vb * (np.interp(pb + d / vb, p_at, q_at)
                                          - np.interp(pb, p_at, q_at)))
                    self.E[a][m] += give
                    self.E[b][m] += give
                self.fpot[a][m] -= d / self.fvol[a][m]     # active ml conserved:
                self.fpot[b][m] += d / self.fvol[b][m]     # what one loses, one gains

    def _splash(self, x, y, z, tz=None):
        """Shove the struck fluid cell outward — into lateral neighbors at level
        tz (its own level for a sinking entry; the level ABOVE for a surface
        slap, where the water erupts up-and-out around what hit it) — and throw
        droplet parcels. Deep in a pool every neighbor is full, so nothing
        sprays — the quiet deep falls out for free."""
        if tz is None:
            tz = z
        vol, f = float(self.fvol[x, y, z]), self.fl[x, y, z]
        if vol <= 1.0 or f == NOFLUID:
            return
        nx, ny = self.shape[0], self.shape[1]
        room = [(dx, dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= x + dx < nx and 0 <= y + dy < ny
                and self.mat[x + dx, y + dy, tz] == AIR
                and self._mix_cells((x, y, z), (x + dx, y + dy, tz)) is not None
                and self.fvol[x + dx, y + dy, tz] < 0.7 * self.cap]
        for (dx, dy) in room:
            part = min(0.4 * vol / len(room),
                       self.cap - float(self.fvol[x + dx, y + dy, tz]))
            if part > 0.5:
                self._move_fluid((x, y, z), (x + dx, y + dy, tz), part)
            if len(self.drops) < 2000:                   # and real droplets FLY:
                dml = 0.2 * vol / len(room)              # an arcing crown parcel
                if dml > 1.0:
                    self.fvol[x, y, z] -= dml
                    side = SPLASH_SIDE * TICK_S / (0.1 * self.scale)   # m/s -> cells/tick
                    up = SPLASH_UP * TICK_S / (0.1 * self.scale)
                    drop = [float(x + dx), float(y + dy), float(tz + 1),
                            side * dx, side * dy, up, dml, int(f),
                            float(self.fpot[x, y, z])]
                    if self._frow((x, y, z)):            # a mix rides in the tenth entry
                        drop.append(self._frow((x, y, z)))
                    self.drops.append(drop)

    def _law_acid(self):
        """Dissolution — combustion's quieter sibling, driven ENTIRELY by the
        REACTIONS table. Where a reactive fluid touches a solid face, solid mass
        goes at the row's measured rate for the fluid's strength (mg per cm² of
        wetted face per hour); the fluid is SPENT doing it and the row's heat is
        released. The eaten mass leaves as fumes (the products, H2 and CO2, wait
        for gas species). What is left of the solid becomes the row's residue
        (vitriol leaves wood as char) or nothing: the hole an acid bores is
        just mat -> AIR, so the fluid FALLS into it and keeps eating (support
        and flow laws do the rest). A new acid is a data row, never code.

        A SLOW ROW KEEPS ITS OWN CLOCK. Real rates are tiny next to a voxel:
        steel in dilute acid loses about a microgram a tick from a 5 cm face,
        and a 975 g voxel in float32 cannot even hold that change (its smallest
        step there is 61 µg), so the change would round away to nothing. So a
        row runs once every N ticks with N ticks' worth of eating, N chosen so
        that one bite is at least 16 of those smallest steps: rounding then
        costs at most about 3%. A fast row has N = 1 and runs every tick."""
        solid = self.mat != AIR
        per_h = 100.0 * self.scale ** 2 * 1e-3 * TICK_S / 3600.0   # mg/(cm²·h) -> g per
        for f in _REACTIVE:                                        # face per tick
            has = (self.fl == f) & (self.fvol > 1.0) & (self.fpot > 0.01)
            if self.mixes_live:                      # D6: a mixed fluid is never acid here, and
                has &= self.fcomp == 0               # acid on brine is refused (_mix_ids)
            if not has.any():
                continue
            rows = {}                                # this tick's rows: mat -> (N, row)
            for (rf, rm), row in REACTIONS.items():
                if rf != f:
                    continue
                bite = float(_RATE_PTS[(rf, rm)][1].max()) * per_h
                step = float(np.spacing(np.float32(SOLID[rm][0] * self.vox_l)))
                n = max(1, int(np.ceil(16.0 * step / bite)))
                if (self.tick + 1) % n == 0:         # pays for the n ticks just gone
                    rows[rm] = (n, row)
            if not rows:
                continue
            on = np.zeros(NMAT_PAD, bool)            # indexed by the material id of every cell
            spend = np.zeros(NMAT_PAD, np.float32)
            heat = np.zeros(NMAT_PAD, np.float32)
            for rm, (_n, (_pts, sp, h, _res)) in rows.items():
                on[rm], spend[rm], heat[rm] = True, sp, h
            omat = on[self.mat]
            if self.mixes_live:                      # D6: a mixed solid is not dissolved by the
                omat = omat & (self.scomp == 0)      # legacy law (it would delete the rest of the mix)
            hit = np.zeros(self.shape, bool)         # solid cells eaten this tick
            for axis in range(3):
                a = [slice(None)] * 3; b = [slice(None)] * 3
                a[axis], b[axis] = slice(None, -1), slice(1, None)
                a, b = tuple(a), tuple(b)
                for s_, d_ in ((a, b), (b, a)):              # fluid at s_, solid at d_
                    m = has[s_] & solid[d_] & omat[d_]
                    if not m.any():
                        continue
                    mats = self.mat[d_][m]
                    pots = self.fpot[s_][m]
                    # the rate is READ off the row's points at this strength —
                    # thresholds and all — not scaled by potency
                    want = np.zeros(len(mats))
                    for rm, (n, _row) in rows.items():
                        sel = mats == rm
                        if sel.any():
                            want[sel] = np.interp(pots[sel], *_RATE_PTS[(f, rm)]) * per_h * n
                    dm = np.minimum(self.smass[d_][m],
                                    want * np.minimum(self.fvol[s_][m] / self.cap, 1.0))
                    active_ml = self.fpot[s_][m] * self.fvol[s_][m]
                    dm = np.minimum(dm, active_ml / np.maximum(spend[mats], 1e-6))
                    old = self.smass[d_][m]
                    new = (old - dm).astype(np.float32)
                    dm = old - new                           # what float32 really took
                    self.smass[d_][m] = new
                    # SPENT reagent stays in the pool as inert volume: potency
                    # falls, the liquid itself remains (and pales, on camera)
                    self.fpot[s_][m] = np.maximum(
                        (active_ml - dm * spend[mats]) / self.fvol[s_][m], 0.0)
                    self.smoke[d_][m] += dm                  # dissolved mass leaves as fumes
                    self.E[s_][m] += dm * heat[mats]
                    hit[d_] |= m
            for rm, (_n, (_pts, _sp, _h, res)) in rows.items():
                if res is None:
                    continue
                left = hit & (self.mat == rm) & (self.smass <= res[1] * SOLID[rm][0] * self.vox_l)
                if left.any():
                    self.set_mat(left, res[0])               # vitriol's black wood: char
        gone = solid & (self.smass <= 0.01)
        if gone.any():
            self._unmake(gone, "dissolves")                  # dissolved CLEAN — no ash
            # ONE EVENT PER TICK, aggregated across every material an acid
            # ate through this tick (3.4) -- not one per cell.
            self._defer("dissolves", data=(("cells", int(gone.sum())),),
                      text="")
