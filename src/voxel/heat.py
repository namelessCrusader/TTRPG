"""Heat laws: conduction, buoyant rise, combustion, boiling, melting and freezing.

Every law here is written in SI per second and multiplied by TICK_S
(docs/realism-references.md, G4 and G5)."""
import math
import weakref

import numpy as np
from .kernels import HAVE_NUMBA, _any_in, njit
from .faceenv import (ENV_BOIL, ENV_OVERLAY, NO_ENV, OV_AZ, OV_HNAT, OV_RADW, OV_RSER,
                      FaceEnv, exchange_np, film_np, ids_np, overlay_table)
from .materials import (AIR, AIR_C, AMBIENT, ASH, BOIL, BURN_AREA, CHAR, CHI_R, EDDY, EPS_FLAME, EPS_GAS, FACE_CLIP, FLESH, FLUID, FREEZE, GLOW, GRAVITY, H_CONV, LATENT, MELT, NOFLUID, O2_FACE, O2_PER_L, O2_RATIO, O2_RATIO_OIL, OIL, PLUME_REACH_M, PYRO, PYRO_OIL, SIGMA, SMOKE_EXT, SOLID, SOOT, SOOT_OIL, TICK_S, WATER, WOOD, GLOW_COLORS, GLOW_RED, ROPE, flame_temp, _TRANSMIT_ARR, _CSOLID_ARR, _DENS_ARR, _EMISS_ARR, _FC_ARR, _FDENS_ARR, _FEMISS_ARR, _FK_ARR, _FSKIN_DEPTH_ARR, _FSKIN_PATH_ARR, _FLK_ARR, _IGN_ARR, _KSOLID_ARR, _SKIN_DEPTH_ARR, _SKIN_PATH_ARR,
                        _EMISS_FILM_ARR, _HNAT_ARR, NMAT_PAD, SKIN_T)
from . import heatmix as _heatmix


def _count_nb(mask):
    """How many of each cell's six neighbours `mask` is true of. Past the edge
    of the array there is nothing."""
    n = np.zeros(mask.shape, np.int8)
    n[1:, :, :] += mask[:-1, :, :]
    n[:-1, :, :] += mask[1:, :, :]
    n[:, 1:, :] += mask[:, :-1, :]
    n[:, :-1, :] += mask[:, 1:, :]
    n[:, :, 1:] += mask[:, :, :-1]
    n[:, :, :-1] += mask[:, :, 1:]
    return n


def _count_open_at(mat, fvol, film, i):
    """`_count_nb(mat == AIR)` and `_count_nb((mat == AIR) & (fvol <= film))`
    at the flat cells `i` only: the same counts, without a pass over the
    whole lattice for the sake of a few cells."""
    at = np.unravel_index(i, mat.shape)
    n_out = np.zeros(len(i), np.int8)
    n_gas = np.zeros(len(i), np.int8)
    for axis in range(3):
        for d in (-1, 1):
            c = list(at)
            c[axis] = at[axis] + d
            ok = (c[axis] >= 0) & (c[axis] < mat.shape[axis])
            nb = tuple(a[ok] for a in c)
            gas = mat[nb] == AIR
            n_out[ok] += gas
            n_gas[ok] += gas & (fvol[nb] <= film)
    return n_out, n_gas


def _mix_pick(mixes, name, base, ids):
    """`base` (one number per cell, from the pure tables) with the cells whose
    mix id in `ids` is above 0 read from the mix table's column `name` instead
    (T1 mixes, wave 13). `ids` None, or no mixed cell in it, gives `base` itself:
    a pure world gathers exactly what it always did. A column that is not a
    finite number for a row (a rule the row has no data for) leaves that cell
    on the pure table, which fails safe."""
    if ids is None:
        return base
    sel = ids > 0
    if not sel.any():
        return base
    v = mixes.prop(name, ids[sel])
    out = base.copy()
    cur = out[sel]
    good = np.isfinite(v)
    cur[good] = v[good]
    out[sel] = cur
    return out


_K0 = np.float32(273.15)
FILM = 0.01                      # a cell 1% full of liquid is wet: a 0.5 mm film
                                 # at 5 cm, which takes heat from the floor it lies on
_TINF = np.float32(AMBIENT + 273.15)
_T_FREE = float(flame_temp(SOLID[WOOD][4], O2_RATIO[WOOD]))   # a free wood flame, °C
_TINF2 = np.float32((AMBIENT + 273.15) ** 2)
_MELTABLE_ARR = np.array([m in MELT for m in range(len(_DENS_ARR))])
_FREEZABLE_ARR = np.array([f in FREEZE for f in range(len(_FK_ARR))])
_EPSIG = np.float32(EPS_GAS * SIGMA)
# EFFUSIVITY sqrt(k rho c), W·s^½/(m²K): how hard a body holds its surface
# temperature when something cold touches it. Iron 14,000, stone 2,000,
# water 1,600, wood 400.
_EFF_ARR = np.sqrt(_KSOLID_ARR.astype(np.float64) * _DENS_ARR * _CSOLID_ARR * 1000.0).astype(np.float32)
_FEFF_ARR = np.sqrt(_FLK_ARR.astype(np.float64) * _FDENS_ARR * _FC_ARR * 1.0e6).astype(np.float32)
_FOURSIG = np.float32(4.0 * SIGMA)
# THE MATERIAL FILM COLUMNS AS AN OVERLAY (decision D4). When `World.film_cols` is
# on, the face law reads each material's natural-convection column through the face
# bundle: a material whose `_HNAT_ARR` entry is above 0 gets overlay row = its id; every
# other material keeps id 0, the default row, i.e. the law as it always was.
_FILM_ID = np.where(_HNAT_ARR > 0, np.arange(NMAT_PAD), 0).astype(np.uint16)
_FILM_OV = overlay_table([{"hnat": float(_HNAT_ARR[m])} if _HNAT_ARR[m] > 0 else {}
                          for m in range(1, NMAT_PAD)])      # row 0 is the default row
RISE_MAX = 0.5                   # the most of its gas a cell hands up in a tick:
                                 # half, where it and the cell above are even


# ── FLAMES AND RADIATION (docs/realism-map.md, lane 5) ──────────────────────
# A fire is a CLUSTER of burning cells. Its heat release Q and the width D of
# its foot give its flame a height (Heskestad, F25); the flame is a set of
# cells that heat the surfaces they touch out of the fire's own gas heat; and
# the radiant share of its heat, with the hot surfaces of the room, is thrown
# across the air (G7) and lands in the skins of whatever it reaches.
RAD_EVERY = 6                    # ticks between radiation passes (4-8)
RAD_MAX_SRC = 24                 # the most sources a pass throws (the strongest)
RAD_Q_MIN = 250.0                # W/m²: a flux below this is not followed
RAD_HOT = 500.0                  # °C: a surface hotter than this is a source
RAD_RANGE = 60                   # cells: the farthest a pass looks
FIRE_JOIN = 2                    # burning cells this near (in cells, each axis)
                                 # are one fire: a gap of one cell does not part
                                 # a crib's sticks
FLAME_PAY_MAX = 1.0              # the most of its gas heat a fire may hand to the
                                 # surfaces its flame touches: all of it, if they
                                 # ask for it (CALIBRATION: in the F-T2 crib of
                                 # 5 cm sticks, a share of 0.8 caught 23% of the
                                 # sticks in 2 min, 0.9 36%, 1.0 51%)
LUMP_M = 0.0125                  # m, a lump of charcoal or a pebble in a bed (u: 1-2 cm, Kunii and Smith 1960)
FLAME_PASS = 0.4                 # a cell at least this empty (void share) is a way
                                 # through for a flame (the gaps of a crib)
FLAME_WALL = 0.5                 # a side is a wall when this share of the cells
                                 # beside the fire's foot is solid
CHAR_K = 0.1                     # W/(m·K), the char layer on burning wood (0.05-0.15)
CHAR_YIELD = 0.10                # the share of its mass wood keeps as char in a fire
                                 # (0.1-0.2); the char depth is the mass lost over
                                 # (density × area × (1 - this))
DT_CHF = 30.0                    # boiling curve (water at one atmosphere,
DT_LEID = 130.0                  # Nukiyama): wall superheat in K where the
Q_CHF = 1.1e6                    # critical heat flux (W/m²) is reached, where the
H_FILM = 250.0                   # film boiling takes over, and the film's
H_NUC = 1500.0                   # convective h (W/m²K) and the plain wet h


_F_NUC, _F_FILM = np.float32(H_NUC), np.float32(H_FILM)
_F_QCHF, _F_CHF, _F_LEID = np.float32(Q_CHF), np.float32(DT_CHF), np.float32(DT_LEID)
_F_BOIL = np.float32(BOIL)
_F_QMIN = np.float32(H_FILM * DT_LEID)
_F_QPEAK = np.float32(H_NUC * DT_CHF + Q_CHF)


@njit(cache=True)
def _hboil_t(h, tw, tl, es, el, e, foursig, k0, tbo):
    """THE BOILING CURVE (Nukiyama), as the heat-transfer coefficient, W/(m²K),
    between a wall at `tw` °C and the water at `tl`; `h` is the plain wet
    coefficient it falls back to. What boils the water is the temperature of
    the INTERFACE, which the wall holds against the water by its effusivity
    `es` (the water's is `el`): iron holds its surface and keeps the water
    boiling hard, wood gives its surface up at once and never gets the water
    to boil at all (and cannot make a steam film: a burning log is quenched,
    a hot pan is not). With `dT` K of the interface above 100 °C: up to
    DT_CHF bubbles carry the heat away, the flux climbing as the cube of dT
    to the critical heat flux (about 1.1 MW/m²); past it a film of vapour
    begins to cover the wall and the flux FALLS to a minimum at DT_LEID K; beyond
    that the wall rides on a film of steam (the Leidenfrost state), and only
    the film's conduction and the wall's radiation carry heat. A red-hot bar
    goes in the water with a skin of steam and cools slowly until it is cool
    enough for the steam to break (H27).

    `tbo` is the boiling point of THIS liquid, °C: BOIL for pure water, and
    in the face-environment bundle (faceenv.py) a float32 per cell, so that a
    brine, which boils hotter, can use the same curve."""
    tc = (es * tw + el * tl) / (es + el)
    dT = tc - tbo
    if dT <= np.float32(0.0):
        return h
    if dT <= _F_CHF:
        x = dT / _F_CHF
        q = _F_NUC * dT + _F_QCHF * (x * x * x)
    elif dT < _F_LEID:
        q = _F_QPEAK + (_F_QMIN - _F_QPEAK) * ((dT - _F_CHF) / (_F_LEID - _F_CHF))
    else:
        tk = tc + k0
        sk = tbo + k0
        q = _F_FILM * dT + e * (foursig * np.float32(0.25)) * ((tk * tk + sk * sk) * (tk + sk)) * dT
        return q / (tw - tl)
    hb = q / (tw - tl)              # bubbles only ever add to the plain wet coefficient:
    return hb if hb > h else h      # a wall that can't heat its interface (wood) keeps `h`


@njit(cache=True)
def _hboil(h, tw, tl, es, el, e, foursig, k0):
    """`_hboil_t` for water at the plain boiling point (BOIL): the curve as it
    has always been called."""
    return _hboil_t(h, tw, tl, es, el, e, foursig, k0, _F_BOIL)


@njit(cache=True)
def _face_heat(Ts, Tb, Csur, Ccor, k, hl, kind, lit, void, eps, af, bc, eff, qs, qc, dx, dt,
               A, hconv, epsig, foursig, k0, clip, di, dj, dl, env, bz, ov, bp):
    """One axis of `_law_conduct`'s face loop, compiled: every face visited
    once. `kind` is 0 for a solid cell, 1 for gas, 2 for liquid. The arithmetic
    follows the numpy law step for step, in float32 and in the same order, and
    every cell takes its losses before its gains, as the numpy law does, so the
    two agree to the bit.

    The last four arguments are the FACE-ENVIRONMENT BUNDLE (faceenv.py):
    `env` is a bit set (ENV_OVERLAY: `bz` and the table `ov` are live;
    ENV_BOIL: `bp` is live) and with `env` = 0, which is what every world has
    today, none of the four is read and the law is the one it always was.
    Returns the flux across each face (J, from the lower cell to the higher),
    which the caller may keep."""
    nx, ny, nz = Ts.shape
    q = np.empty((nx - di, ny - dj, nz - dl), np.float32)
    two = np.float32(2.0)
    half = np.float32(0.5)
    quarter = np.float32(0.25)
    tiny = np.float32(1e-12)
    zero = np.float32(0.0)
    one = np.float32(1.0)
    nrows = ov.shape[0]
    for i in range(nx - di):
        for j in range(ny - dj):
            for l in range(nz - dl):
                i2 = i + di
                j2 = j + dj
                l2 = l + dl
                ka = kind[i, j, l]
                kb = kind[i2, j2, l2]
                sa = ka == 0
                sb = kb == 0
                ua = (sa and not sb) or (ka == 2 and kb == 1)
                uc = (sb and not sa) or (kb == 2 and ka == 1)
                if ua:
                    ta = Ts[i, j, l]
                    ca = Csur[i, j, l]
                else:
                    ta = Tb[i, j, l]
                    ca = Ccor[i, j, l]
                if uc:
                    tc = Ts[i2, j2, l2]
                    cc = Csur[i2, j2, l2]
                else:
                    tc = Tb[i2, j2, l2]
                    cc = Ccor[i2, j2, l2]
                x = k[i, j, l]
                y = k[i2, j2, l2]
                m = x + y
                if m < tiny:
                    m = tiny
                g = two * x * y / m * dx
                if sa and sb:
                    v = void[i, j, l]                # the cavity view of the more open cell
                    w2 = void[i2, j2, l2]
                    if w2 > v:
                        v = w2
                    e = eps[i, j, l]
                    e2 = eps[i2, j2, l2]
                    eg = e * e2 / (e + e2 - e * e2)  # two grey surfaces facing each other
                    tka = ta + k0
                    tkc = tc + k0
                    g = g + v * eg * (foursig * quarter * (tka * tka + tkc * tkc) * (tka + tkc)) * A
                if (sa and kb == 2) or (ka == 2 and sb):
                    if ka == 2:
                        h = hl[i, j, l]
                        tbo = _F_BOIL
                        if env & ENV_BOIL:
                            tbo = bp[i, j, l]
                        if bc[i, j, l] and tc > tbo and tc > ta:
                            h = _hboil_t(h, tc, ta, eff[i2, j2, l2], eff[i, j, l],
                                         eps[i2, j2, l2], foursig, k0, tbo)
                    else:
                        h = hl[i2, j2, l2]
                        tbo = _F_BOIL
                        if env & ENV_BOIL:
                            tbo = bp[i2, j2, l2]
                        if bc[i2, j2, l2] and ta > tbo and ta > tc:
                            h = _hboil_t(h, ta, tc, eff[i, j, l], eff[i2, j2, l2],
                                         eps[i, j, l], foursig, k0, tbo)
                    g = h * A * af[i, j, l] * af[i2, j2, l2]
                    if env & ENV_OVERLAY:           # the solid cell's overlay: immersion
                        oid = bz[i, j, l] if sa else bz[i2, j2, l2]
                        if oid != 0 and oid < nrows:
                            rs = ov[oid, OV_RSER]
                            if rs > zero:
                                ar = A * af[i, j, l] * af[i2, j2, l2]
                                if ar > zero:
                                    g = g / (one + g * rs / ar)
                            g = g * ov[oid, OV_AZ]
                if ((sa or ka == 2) and kb == 1) or (ka == 1 and (sb or kb == 2)):
                    tka = ta + k0
                    tkc = tc + k0
                    if ka == 1:
                        hot = ta > tc
                        burns = lit[i2, j2, l2]
                    else:
                        hot = tc > ta
                        burns = lit[i, j, l]
                    if hot and burns:
                        g = zero
                    else:
                        rad = zero
                        if hot:
                            rad = epsig * (tka * tka + tkc * tkc) * (tka + tkc)
                        oid = 0
                        if env & ENV_OVERLAY:       # the surface cell's overlay (the one that is not gas)
                            oid = bz[i2, j2, l2] if ka == 1 else bz[i, j, l]
                            if oid >= nrows:
                                oid = 0
                        if oid == 0:
                            g = (hconv + rad) * A * af[i, j, l] * af[i2, j2, l2]
                        else:
                            hn = ov[oid, OV_HNAT]
                            hc = hconv
                            if hn > zero:
                                dd = ta - tc
                                if dd < zero:
                                    dd = -dd
                                hc = hn * math.sqrt(math.sqrt(dd))
                            g = (hc + rad * ov[oid, OV_RADW]) * A * af[i, j, l] * af[i2, j2, l2]
                            rs = ov[oid, OV_RSER]
                            if rs > zero:
                                ar = A * af[i, j, l] * af[i2, j2, l2]
                                if ar > zero:
                                    g = g / (one + g * rs / ar)
                            g = g * ov[oid, OV_AZ]
                d = ta - tc
                v = g * dt * d
                cm = ca if ca < cc else cc
                ad = d if d >= zero else -d
                lim = clip * cm * ad
                if v < -lim:
                    v = -lim
                if v > lim:
                    v = lim
                q[i, j, l] = v
    for i in range(nx - di):
        for j in range(ny - dj):
            for l in range(nz - dl):
                v = q[i, j, l]
                ka = kind[i, j, l]
                kb = kind[i + di, j + dj, l + dl]
                if (ka == 0 and kb != 0) or (ka == 2 and kb == 1):
                    qs[i, j, l] -= v
                else:
                    qc[i, j, l] -= v
    for i in range(nx - di):
        for j in range(ny - dj):
            for l in range(nz - dl):
                v = q[i, j, l]
                ka = kind[i, j, l]
                kb = kind[i + di, j + dj, l + dl]
                if (kb == 0 and ka != 0) or (kb == 2 and ka == 1):
                    qs[i + di, j + dj, l + dl] += v
                else:
                    qc[i + di, j + dj, l + dl] += v
    return q


# the six faces, in the order `_hand_out` visits them: +x, -x, +y, -y, +z, -z
_DIRS = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
                 np.int64)


@njit(cache=True)
def _give(src, budget, sa, da, void, by_gap, got, dirs):
    """`_hand_out`'s gifts, compiled. Every sum is taken in the order the numpy
    law takes it (a giver's faces +x, -x, +y, -y, +z, -z; a taker's givers
    from -x, +x, -y, +y, -z, +z), so the two agree to the bit."""
    nx, ny, nz = src.shape
    take = np.zeros(src.shape)
    for i in range(nx):
        for j in range(ny):
            for l in range(nz):
                if not src[i, j, l]:
                    continue
                for k in range(6):
                    i2 = i + dirs[k, 0]
                    j2 = j + dirs[k, 1]
                    l2 = l + dirs[k, 2]
                    if i2 < 0 or j2 < 0 or l2 < 0 or i2 >= nx or j2 >= ny or l2 >= nz:
                        continue
                    g = sa[i, j, l] * da[i2, j2, l2]
                    if by_gap:
                        v = void[i, j, l]
                        if void[i2, j2, l2] > v:
                            v = void[i2, j2, l2]
                        g = g * v
                    take[i, j, l] += g
    scale = np.ones(src.shape)
    for i in range(nx):
        for j in range(ny):
            for l in range(nz):
                t = take[i, j, l]
                if t > budget[i, j, l]:
                    scale[i, j, l] = budget[i, j, l] / max(t, 1e-30)
    for i in range(nx):                          # what each giver pays
        for j in range(ny):
            for l in range(nz):
                if not src[i, j, l]:
                    continue
                for k in range(6):
                    i2 = i + dirs[k, 0]
                    j2 = j + dirs[k, 1]
                    l2 = l + dirs[k, 2]
                    if i2 < 0 or j2 < 0 or l2 < 0 or i2 >= nx or j2 >= ny or l2 >= nz:
                        continue
                    g = sa[i, j, l] * da[i2, j2, l2]
                    if by_gap:
                        v = void[i, j, l]
                        if void[i2, j2, l2] > v:
                            v = void[i2, j2, l2]
                        g = g * v
                    budget[i, j, l] -= g * scale[i, j, l]
    for i in range(nx):                          # what each taker gets, from
        for j in range(ny):                      # -x, +x, -y, +y, -z, +z
            for l in range(nz):
                for k in range(6):
                    i0 = i - dirs[k, 0]
                    j0 = j - dirs[k, 1]
                    l0 = l - dirs[k, 2]
                    if i0 < 0 or j0 < 0 or l0 < 0 or i0 >= nx or j0 >= ny or l0 >= nz:
                        continue
                    if not src[i0, j0, l0]:
                        continue
                    g = sa[i0, j0, l0] * da[i, j, l]
                    if by_gap:
                        v = void[i0, j0, l0]
                        if void[i, j, l] > v:
                            v = void[i, j, l]
                        g = g * v
                    got[i, j, l] += g * scale[i0, j0, l0]


@njit(cache=True)
def _nonzero_flat(a):
    """The flat indices where `a` is not zero, ascending (`np.flatnonzero`,
    without the whole-world temporaries: it is asked of every cell of the
    world every tick a fire burns)."""
    f = a.reshape(-1)
    n = 0
    for i in range(f.shape[0]):
        if f[i] != 0.0:
            n += 1
    out = np.empty(n, np.int64)
    k = 0
    for i in range(f.shape[0]):
        if f[i] != 0.0:
            out[k] = i
            k += 1
    return out


@njit(cache=True)
def _root(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


@njit(cache=True)
def _cluster(ix, iy, iz, join):
    """Which fire each burning cell is in: cells within `join` cells of each
    other (on every axis) are one fire. Labels run 0.. in the order the fires'
    first cells are met, which is ascending flat index, so the answer does not
    depend on how the cells were found."""
    n = ix.shape[0]
    lab = np.zeros(n, np.int32)
    if n == 0:
        return lab, 0
    x0, y0, z0 = ix.min(), iy.min(), iz.min()
    sx, sy, sz = ix.max() - x0 + 1, iy.max() - y0 + 1, iz.max() - z0 + 1
    ids = np.full((sx, sy, sz), -1, np.int32)
    for i in range(n):
        ids[ix[i] - x0, iy[i] - y0, iz[i] - z0] = i
    parent = np.arange(n)
    for i in range(n):
        a0, b0, c0 = ix[i] - x0, iy[i] - y0, iz[i] - z0
        for dx in range(0, join + 1):
            for dy in range(-join, join + 1):
                for dz in range(-join, join + 1):
                    if dx == 0 and (dy < 0 or (dy == 0 and dz <= 0)):
                        continue
                    a, b, c = a0 + dx, b0 + dy, c0 + dz
                    if a >= sx or b < 0 or b >= sy or c < 0 or c >= sz:
                        continue
                    j = ids[a, b, c]
                    if j >= 0:
                        ri = _root(parent, i)
                        rj = _root(parent, j)
                        if ri != rj:
                            parent[rj] = ri
    first = np.full(n, -1, np.int32)
    nf = 0
    for i in range(n):
        r = _root(parent, i)
        if first[r] < 0:
            first[r] = nf
            nf += 1
        lab[i] = first[r]
    return lab, nf


@njit(cache=True)
def _fire_stats(lab, nf, ix, iy, iz, bj, nx, ny):
    """Per fire: the heat (J) it released this tick, how many columns its foot
    covers, its lowest and highest burning cell, and the box of its foot."""
    n = lab.shape[0]
    big = 1 << 30
    q = np.zeros(nf)
    zlo = np.full(nf, big, np.int64)
    ztp = np.full(nf, -1, np.int64)
    x0 = np.full(nf, big, np.int64)
    x1 = np.full(nf, -1, np.int64)
    y0 = np.full(nf, big, np.int64)
    y1 = np.full(nf, -1, np.int64)
    keys = np.empty(n, np.int64)
    for i in range(n):
        f = lab[i]
        q[f] += bj[i]
        if iz[i] < zlo[f]:
            zlo[f] = iz[i]
        if iz[i] > ztp[f]:
            ztp[f] = iz[i]
        if ix[i] < x0[f]:
            x0[f] = ix[i]
        if ix[i] > x1[f]:
            x1[f] = ix[i]
        if iy[i] < y0[f]:
            y0[f] = iy[i]
        if iy[i] > y1[f]:
            y1[f] = iy[i]
        keys[i] = (f * nx + ix[i]) * ny + iy[i]
    ncol = np.zeros(nf, np.int64)
    ks = np.unique(keys)
    for k in ks:
        ncol[k // (nx * ny)] += 1
    return q, ncol, zlo, ztp, x0, x1, y0, y1


@njit(cache=True)
def _walls(st, x0, x1, y0, y1, zlo, ztp, share):
    """How many walls each fire stands against: 0 in the open, 1 against a
    wall, 2 in a corner. A side is a wall when `share` of the cells one or two
    cells beyond the edge of the fire's foot are solid (2 in `st`: a puddle
    or a gap is not a wall), from the fire's lowest cell up to `ztp`, which the
    caller sets to the lower half of the flame: a rim a cell high round a lamp
    is not a wall to a flame half a metre tall.
    Both sides of one axis count once (a fire in a passage is still one wall
    deep: the flame is not pushed taller by a second wall)."""
    nf = x0.shape[0]
    wx, wy, wz = st.shape
    out = np.zeros(nf, np.int64)
    for f in range(nf):
        zl, zt = zlo[f], min(ztp[f], wz - 1)
        nz_ = zt - zl + 1
        ax = False
        ay = False
        for side in range(4):
            best = 0.0
            for gap in (1, 2):
                n_solid = 0
                n_all = 0
                if side == 0:
                    a = x0[f] - gap
                    if a < 0:
                        continue
                    for b in range(max(y0[f], 0), min(y1[f] + 1, wy)):
                        for c in range(zl, zt + 1):
                            n_all += 1
                            if st[a, b, c] == 2:
                                n_solid += 1
                elif side == 1:
                    a = x1[f] + gap
                    if a >= wx:
                        continue
                    for b in range(max(y0[f], 0), min(y1[f] + 1, wy)):
                        for c in range(zl, zt + 1):
                            n_all += 1
                            if st[a, b, c] == 2:
                                n_solid += 1
                elif side == 2:
                    b = y0[f] - gap
                    if b < 0:
                        continue
                    for a in range(max(x0[f], 0), min(x1[f] + 1, wx)):
                        for c in range(zl, zt + 1):
                            n_all += 1
                            if st[a, b, c] == 2:
                                n_solid += 1
                else:
                    b = y1[f] + gap
                    if b >= wy:
                        continue
                    for a in range(max(x0[f], 0), min(x1[f] + 1, wx)):
                        for c in range(zl, zt + 1):
                            n_all += 1
                            if st[a, b, c] == 2:
                                n_solid += 1
                if n_all > 0 and n_solid / n_all > best:
                    best = n_solid / n_all
            if best >= share:
                if side < 2:
                    ax = True
                else:
                    ay = True
        out[f] = (1 if ax else 0) + (1 if ay else 0)
    return out


@njit(cache=True)
def _note(colf, zlo, ztp, Lc, cx, cy, z, f):
    """Mark column (cx, cy) as carrying fire f's flame, up from z."""
    g = colf[cx, cy]
    if g == f:
        if z < zlo[cx, cy]:
            zlo[cx, cy] = z
        if z > ztp[cx, cy]:
            ztp[cx, cy] = z
    elif g < 0 or z + Lc[f] > ztp[cx, cy] + Lc[g]:
        colf[cx, cy] = f
        zlo[cx, cy] = z
        ztp[cx, cy] = z


@njit(cache=True)
def _flame_grow(st, bx, by, bz, blab, Lc, own, wgt):
    """The cells each fire's flame fills. `st` is 0 for open gas, 1 for a
    cell with room in it for a flame to pass (the gaps of a crib), 2 for a
    solid and 3 for a liquid (both block it); `Lc` is the flame length of each fire in cells.

    A flame stands on its burning cell and rises, no wider than the cell,
    to its length above the fire's highest cell. Where the cell above a
    burning one is a solid (a burning wall, a burning ceiling) the flame is
    held against the face, in the open cells beside it, and climbs there. It
    ends at the first solid it meets: there it spreads sideways under it for
    the length it has left (F25, ceiling). The cells beside a burning face
    are the flame too, wherever they are: they are where it is attached, and
    over a liquid fuel beside a burning cell the cell above it is too, at half
    the flux (how a fire spreads across a pool).
    `own` is -1 where no flame is, else the fire's number; `wgt` is how much
    of the flame's flux a flame cell gives (1, or a half for a cell across an
    edge from the burning one, which is further from it)."""
    wx, wy, wz = st.shape
    n = bx.shape[0]
    colf = np.full((wx, wy), -1, np.int32)
    zlo = np.zeros((wx, wy), np.int32)
    ztp = np.zeros((wx, wy), np.int32)
    for i in range(n):
        f = blab[i]
        x, y, z = bx[i], by[i], bz[i]
        if z + 1 < wz and st[x, y, z + 1] < 2:
            _note(colf, zlo, ztp, Lc, x, y, z, f)
        else:
            for k in range(4):
                cx = x + (1 if k == 0 else (-1 if k == 1 else 0))
                cy = y + (1 if k == 2 else (-1 if k == 3 else 0))
                if cx < 0 or cy < 0 or cx >= wx or cy >= wy:
                    continue
                if st[cx, cy, z] < 2:
                    _note(colf, zlo, ztp, Lc, cx, cy, z, f)
    cap = wx * wy * 4
    qx = np.empty(cap, np.int32)
    qy = np.empty(cap, np.int32)
    qz = np.empty(cap, np.int32)
    rem = np.full((wx, wy, wz), -1, np.int16)
    tail = 0
    for cx in range(wx):
        for cy in range(wy):
            f = colf[cx, cy]
            if f < 0:
                continue
            s = zlo[cx, cy]
            if st[cx, cy, s] >= 2:
                s += 1
            if s >= wz or st[cx, cy, s] >= 2:
                continue
            top = ztp[cx, cy] + int(np.ceil(Lc[f]))
            z = s
            while z <= top and z < wz:
                if st[cx, cy, z] >= 2:
                    break
                own[cx, cy, z] = f
                wgt[cx, cy, z] = 1.0
                z += 1
            if z <= top and z < wz and z > s:      # stopped under a solid
                left = top - z + 1                 # with length to spare
                if left + 1 > rem[cx, cy, z - 1] and tail < cap:
                    rem[cx, cy, z - 1] = left + 1
                    qx[tail] = cx
                    qy[tail] = cy
                    qz[tail] = z - 1
                    tail += 1
    head = 0
    while head < tail:
        x, y, z = qx[head], qy[head], qz[head]
        head += 1
        r = rem[x, y, z] - 1
        if r <= 0:
            continue
        f = own[x, y, z]
        for k in range(4):
            a = x + (1 if k == 0 else (-1 if k == 1 else 0))
            b = y + (1 if k == 2 else (-1 if k == 3 else 0))
            if a < 0 or b < 0 or a >= wx or b >= wy:
                continue
            if st[a, b, z] >= 2 or rem[a, b, z] >= r:
                continue
            rem[a, b, z] = r
            if own[a, b, z] < 0:
                own[a, b, z] = f
                wgt[a, b, z] = 1.0
            if tail < cap:
                qx[tail] = a
                qy[tail] = b
                qz[tail] = z
                tail += 1
    for i in range(n):
        f = blab[i]
        for k in range(6):
            a = bx[i] + (1 if k == 0 else (-1 if k == 1 else 0))
            b = by[i] + (1 if k == 2 else (-1 if k == 3 else 0))
            c = bz[i] + (1 if k == 4 else (-1 if k == 5 else 0))
            if a < 0 or b < 0 or c < 0 or a >= wx or b >= wy or c >= wz:
                continue
            if st[a, b, c] == 0 and own[a, b, c] < 0:
                own[a, b, c] = f
                wgt[a, b, c] = 1.0
            elif st[a, b, c] == 3 and k < 4 and c + 1 < wz:
                # a liquid fuel beside it (a pool): the cell over that is where
                # the fire goes on across the surface, a little further off
                if st[a, b, c + 1] == 0 and own[a, b, c + 1] < 0:
                    own[a, b, c + 1] = f
                    wgt[a, b, c + 1] = 0.5


@njit(cache=True)
def _flame_prep(solid, wet, void, smass, mat, fl, Ts, ign, oil_id, oil_ign, passv,
                st, wallg, ok):
    """What each cell of the window is to a flame, in one pass. `st`: 0 for
    open gas, 1 for a cell with room in it for a flame to pass (a solid at
    least `passv` empty: the gaps of a crib), 2 for a solid that blocks it, 3
    for a liquid. `wallg`: 2
    for a solid that is a wall (not a puddle, not a gap). `ok`: a surface that
    can take the flame's heat: fuel that is not yet burning (a solid that can
    burn and is under its ignition point, or oil under its fire point); a
    stone floor or an iron pan is warmed by the hot gas instead."""
    wx, wy, wz = solid.shape
    for i in range(wx):
        for j in range(wy):
            for k in range(wz):
                if solid[i, j, k]:
                    if void[i, j, k] >= passv:
                        st[i, j, k] = 1
                    else:
                        st[i, j, k] = 2
                        wallg[i, j, k] = 2
                    ig = ign[mat[i, j, k]]
                    if smass[i, j, k] > 0.0 and ig < 1e30 and Ts[i, j, k] < ig:
                        ok[i, j, k] = True
                elif wet[i, j, k]:
                    st[i, j, k] = 3
                    if fl[i, j, k] == oil_id and Ts[i, j, k] < oil_ign:
                        ok[i, j, k] = True


@njit(cache=True)
def _flame_touch(own, st, wgt, ok, Ts, af, nout, tf4, eps, sig, A, dt, at, fire, amt):
    """What the flame hands the surfaces it touches. For every flame cell
    (`own` >= 0): each face it shares with a surface that is not itself
    burning (`ok`) takes eps*sig*(Tflame^4 - Ts^4) W/m² over the open part of
    the face, and a cell with room in it (st == 1) that the flame fills takes
    it over all its open faces. Each gift is written down as (the cell it
    goes to, the fire it is asked of, joules), in `at`, `fire`, `amt`; the
    count comes back."""
    wx, wy, wz = own.shape
    n = 0
    for i in range(wx):
        for j in range(wy):
            for k in range(wz):
                f = own[i, j, k]
                if f < 0:
                    continue
                w8 = wgt[i, j, k]
                if st[i, j, k] == 1 and ok[i, j, k]:
                    tk = Ts[i, j, k] + 273.15
                    t4 = tk * tk * tk * tk
                    if t4 < tf4:
                        at[n] = (i * wy + j) * wz + k
                        fire[n] = f
                        amt[n] = w8 * eps * sig * (tf4 - t4) * nout[i, j, k] * A * af[i, j, k] * dt
                        n += 1
                for d in range(6):
                    a = i + (1 if d == 0 else (-1 if d == 1 else 0))
                    b = j + (1 if d == 2 else (-1 if d == 3 else 0))
                    c = k + (1 if d == 4 else (-1 if d == 5 else 0))
                    if a < 0 or b < 0 or c < 0 or a >= wx or b >= wy or c >= wz:
                        continue
                    if own[a, b, c] >= 0 or st[a, b, c] == 0 or not ok[a, b, c]:
                        continue
                    tk = Ts[a, b, c] + 273.15
                    t4 = tk * tk * tk * tk
                    if t4 < tf4:
                        at[n] = (a * wy + b) * wz + c
                        fire[n] = f
                        amt[n] = w8 * eps * sig * (tf4 - t4) * A * af[a, b, c] * dt
                        n += 1
    return n


@njit(cache=True)
def _hot_scan(E, C, skin, thr):
    """The box (lo, hi per axis) round every cell whose surface (cell average
    plus the size of its skin) is at least `thr` °C above ambient 20; hi < lo
    when there is none. One pass, no temporaries: the question is asked of a
    whole world every few ticks."""
    nx, ny, nz = E.shape
    lo = np.array([nx, ny, nz])
    hi = np.array([-1, -1, -1])
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if E[i, j, k] / C[i, j, k] + abs(skin[i, j, k]) >= thr:
                    if i < lo[0]:
                        lo[0] = i
                    if i > hi[0]:
                        hi[0] = i
                    if j < lo[1]:
                        lo[1] = j
                    if j > hi[1]:
                        hi[1] = j
                    if k < lo[2]:
                        lo[2] = k
                    if k > hi[2]:
                        hi[2] = k
    return lo, hi


@njit(cache=True)
def _rad_prep(solid, wet, smass, void, smoke, mat, fl, trans, emis, femis, kext,
              recv, od, alpha):
    """For radiation, what each cell of the window is: a receiver (a solid
    with mass, or a liquid), the optical depth it puts in a line through it
    (a solid by the share of the cell it does not cover, a liquid 2, a gas by
    the smoke in it), and how much of what lands on it it takes."""
    nx, ny, nz = solid.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if solid[i, j, k]:
                    recv[i, j, k] = smass[i, j, k] > 0.0
                    v = void[i, j, k]
                    tau = v + (1.0 - v) * trans[mat[i, j, k]]
                    if tau < 1e-3:
                        tau = 1e-3
                    od[i, j, k] = -np.log(tau)
                    alpha[i, j, k] = emis[mat[i, j, k]]
                elif wet[i, j, k]:
                    recv[i, j, k] = True
                    od[i, j, k] = 2.0
                    alpha[i, j, k] = femis[fl[i, j, k]]
                else:
                    od[i, j, k] = kext * smoke[i, j, k]


@njit(cache=True)
def _throw(recv, gas, od, af, alpha, lit, sx, sy, sz, sP, sD, snx, sny, snz,
           dxm, kdt, qmin, qcap, got, landed):
    """RADIATION ACROSS THE AIR (G7). Every source (a flame point or a cluster
    of hot surface, in cell units; power in W; `sD` how one-sided its light is,
    along (snx, sny, snz)) lights every open face of every receiver it can
    see: flux = P g cos / (4 pi r²), cut by exp(-optical depth along the
    line), never more than `qcap` (the flame's own surface emissive power),
    and ignored below `qmin`. The line is walked in half-cell steps; a solid
    cell in the way carries its own optical depth (`od`), so walls shade and
    smoke dims. One line serves all the open faces of a cell that face a
    source (they differ by half a cell). What a receiver absorbs is its
    emissivity (`alpha`) of that, for `kdt` seconds, and lands in `got`;
    `landed` adds it up per source. A face that is burning (`lit`) is not
    lit by it: the heat balance of a burning face already counts the flame it
    sits in, radiation included (`PYRO`'s q_fl), and the fires round it."""
    wx, wy, wz = recv.shape
    ns = sx.shape[0]
    A = dxm * dxm
    reach2 = np.empty(ns)                        # how far (cells²) a source can
    for s in range(ns):                          # still give `qmin`
        reach2[s] = sP[s] * (1.0 + 3.0 * sD[s]) / (4.0 * np.pi * qmin) / A + 2.0
    fd = np.empty(6, np.int64)
    for i in range(wx):
        for j in range(wy):
            for k in range(wz):
                if not recv[i, j, k] or lit[i, j, k]:
                    continue
                nfc = 0
                for d in range(6):
                    a = i + (1 if d == 0 else (-1 if d == 1 else 0))
                    b = j + (1 if d == 2 else (-1 if d == 3 else 0))
                    c = k + (1 if d == 4 else (-1 if d == 5 else 0))
                    if a < 0 or b < 0 or c < 0 or a >= wx or b >= wy or c >= wz:
                        continue
                    if gas[a, b, c]:
                        fd[nfc] = d
                        nfc += 1
                if nfc == 0:
                    continue
                for s in range(ns):
                    ux = sx[s] - (i + 0.5)
                    uy = sy[s] - (j + 0.5)
                    uz = sz[s] - (k + 0.5)
                    if ux * ux + uy * uy + uz * uz > reach2[s]:
                        continue
                    tot = -1.0
                    for f in range(nfc):
                        d = fd[f]
                        nxd = 1 if d == 0 else (-1 if d == 1 else 0)
                        nyd = 1 if d == 2 else (-1 if d == 3 else 0)
                        nzd = 1 if d == 4 else (-1 if d == 5 else 0)
                        px = i + 0.5 + 0.5 * nxd
                        py = j + 0.5 + 0.5 * nyd
                        pz = k + 0.5 + 0.5 * nzd
                        vx = sx[s] - px
                        vy = sy[s] - py
                        vz = sz[s] - pz
                        r2 = vx * vx + vy * vy + vz * vz
                        if r2 < 0.04:
                            continue
                        rt = np.sqrt(r2)
                        ct = (vx * nxd + vy * nyd + vz * nzd) / rt
                        if ct <= 0.02:
                            continue
                        g = 1.0
                        if sD[s] > 0.0:
                            cs = -(vx * snx[s] + vy * sny[s] + vz * snz[s]) / rt
                            if cs < 0.0:
                                cs = 0.0
                            g = (1.0 - sD[s]) + sD[s] * 4.0 * cs
                        r = rt
                        if r < 0.5:              # nearer than a flame's own radius
                            r = 0.5              # (0.5 cell): the source is not a
                                                 # point there, and the flux is capped
                        rm = r * dxm
                        q0 = sP[s] * g * ct / (4.0 * np.pi * rm * rm)
                        if q0 < qmin:
                            continue
                        if tot < 0.0:            # the line, walked once
                            tot = 0.0
                            n = int(rt * 2.0) - 2        # (the last cell is the source's)
                            for m in range(1, n):
                                t = m / (n + 2)
                                ci = int(px + vx * t)
                                cj = int(py + vy * t)
                                ck = int(pz + vz * t)
                                if ci < 0 or cj < 0 or ck < 0 or ci >= wx or cj >= wy or ck >= wz:
                                    continue
                                tot += od[ci, cj, ck] * 0.5
                                if tot > 6.0:
                                    break
                        if tot > 6.0:
                            break
                        q = q0 * np.exp(-tot)
                        if q > qcap:
                            q = qcap
                        e = q * alpha[i, j, k] * A * af[i, j, k] * kdt
                        got[i, j, k] += e
                        landed[s] += e


def _o2_factor(health, lim):
    """How much of its flame a burning face keeps, from how much oxygen its
    air holds (1 = fresh). A flame burns near full strength down to a little
    above its limit, then fails: flaming stops near 15% O2 (F11), and does not
    fade from the first breath of foul air."""
    return np.clip((health - lim) / (0.25 * (1.0 - lim)), 0.0, 1.0)


class _Surf:
    """What `_surfaces` works out about a block of cells: which are solid,
    liquid or gas, how open each is, and its skin."""


_GEO = weakref.WeakKeyDictionary()      # world -> its `_surface_geometry`
_FIRES = weakref.WeakKeyDictionary()    # world -> (tick, its fires and flames), kept
                                        # out of the world so a snapshot is not fat
_GEO_FIELDS = ("mat", "smass", "fl", "fvol", "C", "solid", "wet", "gas", "n_out",
               "n_gas", "af", "void", "has", "Cs", "Gsb", "Cb", "Csur", "Ccor")


class HeatMixin:
    _flame_T = _T_FREE                   # the flame's temperature, °C (`_burn_cells`)
    film_cols = False                    # D4: read the surface-film DATA COLUMNS of the
                                         # material row (`_HNAT_ARR`, `_EMISS_FILM_ARR`)
                                         # instead of the constants H_CONV and `_EMISS_ARR`.
                                         # A class attribute: golden does not see it, and
                                         # nothing sets it before 13b.

    # ── the skin (materials.SKIN_T) ─────────────────────────────────────────
    def _skin_field(self):
        """Each cell's skin temperature above the cell's average, K. Zero where
        a cell has no skin. Made on first use, so a world built before it
        existed still runs."""
        s = getattr(self, "skin", None)
        if s is None or s.shape != self.shape:
            s = self.skin = np.zeros(self.shape, np.float32)
        return s

    def _grown(self, sl):
        """`sl` grown by one cell each way, and the slice that crops it back."""
        big, crop = [], []
        for s_, n in zip(sl, self.shape):
            a = 0 if s_.start is None else s_.start
            b = n if s_.stop is None else s_.stop
            a2, b2 = max(a - 1, 0), min(b + 1, n)
            big.append(slice(a2, b2))
            crop.append(slice(a - a2, a - a2 + (b - a)))
        return tuple(big), tuple(crop)

    def _surfaces(self, sl):
        """The cells of `sl` as surfaces: solid, liquid (a cell with more than
        a film of fluid in it, FILM) or gas; how many faces each has open to the
        outside (a solid's to gas or liquid, a liquid's to gas); and the SKIN
        of each one that has open faces.

        The skin (SKIN_T) is a layer sqrt(a*t) deep over the open faces. A
        material whose skin would be a quarter of the cell or deeper (iron,
        tin, lead, molten metal) is even all through and has none. A partly
        filled solid is a thinner thing in the cell (a stick, a bed of coals):
        its surface is sqrt(fill) of the faces, about a square rod's.

        All of that is made of MATTER alone, so it is worked out once for the
        whole world and kept (`_surface_geometry`) until matter changes; only
        the temperatures are read fresh."""
        f32 = np.float32
        geo = self._surface_geometry()
        u = _Surf()
        for name in _GEO_FIELDS:
            setattr(u, name, getattr(geo, name)[sl])
        has = u.has
        u.T = AMBIENT + self.E[sl] / u.C
        s = self._skin_field()[sl]
        u.Ts = np.where(has, u.T + s, u.T)
        u.Tb = np.where(has, u.T - s * u.Cs / np.where(has, u.Cb, f32(1.0)), u.T)
        return u

    def _surface_geometry(self):
        """The matter half of `_surfaces`, for the whole world. KEPT against
        the heat-capacity array: `heat_capacity` builds a new one exactly when
        matter changes (it compares the four fields it is made of), and these
        are made of the same four fields. So the check is right by
        construction, as that cache is; conduction and burning, which both ask
        every tick, share one build, and a world where only heat moves (a bar
        cooling) builds it once. Kept outside the world (`_GEO`), so it is
        never part of its state.

        And when matter HAS changed, only where it changed is built again: a
        fire eats a little wood every tick, so the whole world was rebuilt
        every tick for the sake of a few hundred cells (17 ms of a house-sized
        world's tick). Every number here is a cell's own, made of its own four
        fields and of how many of its six neighbours are open, so the cells
        that changed and their neighbours are worked out again by the same
        arithmetic (`_geo_cells`), and every other cell keeps what it had,
        which is what a full build would give it. The two agree to the bit
        (tests/test_voxel.py checks it, tick by tick, on a burning world)."""
        C = self.heat_capacity()
        geo = _GEO.get(self)
        if geo is not None and geo.C is C:
            return geo
        mat, smass, fl, fvol = self.mat, self.smass, self.fl, self.fvol
        sc, fc = self._mix_args()            # (None, None) until a cell holds a mix
        if geo is not None and (getattr(geo, "sc", None) is None) != (sc is None):
            geo = None                       # a mix appeared (or went): build afresh
        if geo is not None and geo.mat.shape == mat.shape:
            moved = ((geo.mat != mat) | (geo.smass != smass)
                     | (geo.fl != fl) | (geo.fvol != fvol))
            moved |= (geo.C != C)            # a capacity that moved with the same matter (a wet world's vapour, mist, ice)
            if sc is not None:               # a changed mix id is changed matter
                moved |= (geo.sc != sc) | (geo.fc != fc)
            near = moved.copy()                  # and their six neighbours,
            near[1:] |= moved[:-1]               # whose open faces may have
            near[:-1] |= moved[1:]               # changed
            near[:, 1:] |= moved[:, :-1]
            near[:, :-1] |= moved[:, 1:]
            near[:, :, 1:] |= moved[:, :, :-1]
            near[:, :, :-1] |= moved[:, :, 1:]
            i = np.flatnonzero(near)
            if 8 * i.size < near.size:           # a few cells: just those
                if self._sleeps():               # counted at those cells alone
                    n_out, n_gas = _count_open_at(mat, fvol, FILM * self.cap, i)
                else:
                    n_out = _count_nb(mat == AIR).reshape(-1)[i]
                    n_gas = _count_nb((mat == AIR)
                                      & (fvol <= FILM * self.cap)).reshape(-1)[i]
                part = self._geo_cells(*(a.reshape(-1)[i] for a in
                                         (mat, smass, fl, fvol, C)), n_out, n_gas,
                                       self._char_field().reshape(-1)[i],
                                       *(() if sc is None else
                                         (sc.reshape(-1)[i], fc.reshape(-1)[i])))
                for name in _GEO_FIELDS:
                    if name not in ("mat", "smass", "fl", "fvol", "C"):
                        getattr(geo, name).reshape(-1)[i] = part[name]
                j = np.flatnonzero(moved)
                for name, a in (("mat", mat), ("smass", smass), ("fl", fl), ("fvol", fvol)):
                    getattr(geo, name).reshape(-1)[j] = a.reshape(-1)[j]
                if sc is not None:
                    geo.sc.reshape(-1)[j] = sc.reshape(-1)[j]
                    geo.fc.reshape(-1)[j] = fc.reshape(-1)[j]
                geo.C = C
                return geo
        u = _Surf()
        # the matter fields are the world's own arrays, which change in place:
        # keep copies, so a slice of them taken later is of THIS state
        u.mat, u.smass, u.fl, u.fvol = mat.copy(), smass.copy(), fl.copy(), fvol.copy()
        u.C = C
        if sc is not None:
            u.sc, u.fc = sc.copy(), fc.copy()
        part = self._geo_cells(mat, smass, fl, fvol, C, _count_nb(mat == AIR),
                               _count_nb((mat == AIR) & (fvol <= FILM * self.cap)),
                               self._char_field(), sc, fc)
        for name, a in part.items():
            setattr(u, name, a)
        _GEO[self] = u
        return u

    def _geo_cells(self, mat, smass, fl, fvol, C, n_out, n_gas, chd=None,
                   scomp=None, fcomp=None):
        """`_surface_geometry`'s numbers for a set of cells, from each cell's
        own fields and its open-neighbour counts. Every step is cell by cell,
        so it gives the same bits for a whole world or for a list of cells.

        `scomp` and `fcomp` (T1) are the mix ids of those same cells, None for a
        world with no mix. A cell that holds a mix gathers its density, heat
        capacity, conductivity and so its skin from the mix table; every other
        cell reads the pure tables and gets the very bits it always got."""
        f32 = np.float32
        dx = f32(0.1 * self.scale)
        solid = mat != AIR
        wet = ~solid & (fvol > FILM * self.cap)
        gas = ~solid & ~wet
        mx = self.mixes
        dens_s, c_s, k_s = _DENS_ARR[mat], _CSOLID_ARR[mat], _KSOLID_ARR[mat]
        depth_s, path_s = _SKIN_DEPTH_ARR[mat], _SKIN_PATH_ARR[mat]
        dens_f, c_f, k_f = _FDENS_ARR[fl], _FC_ARR[fl], _FLK_ARR[fl]
        depth_f, path_f = _FSKIN_DEPTH_ARR[fl], _FSKIN_PATH_ARR[fl]
        if scomp is not None and (scomp > 0).any():
            dens_s, c_s = _mix_pick(mx, "rho", dens_s, scomp), _mix_pick(mx, "c", c_s, scomp)
            k_s = _mix_pick(mx, "k", k_s, scomp)
            depth_s, path_s = self._mix_skin(scomp, dens_s, c_s, k_s, 1000.0, depth_s, path_s)
        if fcomp is not None and (fcomp > 0).any():
            sel = fcomp > 0
            dens_f = dens_f.copy()
            dens_f[sel] = mx.prop("rho", fcomp[sel]) / 1000.0        # g/ml, as `_fdens_grid`
            c_f, k_f = _mix_pick(mx, "c", c_f, fcomp), _mix_pick(mx, "k", k_f, fcomp)
            depth_f, path_f = self._mix_skin(fcomp, dens_f, c_f, k_f, 1.0e6, depth_f, path_f)
        fill = np.clip(smass / np.maximum(dens_s * self.vox_l, 1e-9), 0.0, 1.0)
        af = np.where(solid, np.sqrt(fill), f32(1.0)).astype(np.float32)
        void = np.where(solid, f32(1.0) - fill, f32(0.0)).astype(np.float32)
        fmass = fvol * dens_f
        n_open = np.where(solid, n_out, np.where(wet, n_gas, 0))
        depth = np.where(solid, depth_s, depth_f)
        path = np.where(solid, path_s, path_f)
        rhoc = np.where(solid, dens_s * c_s * f32(1000.0),
                        dens_f * c_f * f32(1e6))
        own = np.where(solid, smass * c_s, fmass * c_f)
        k = np.where(solid, k_s, k_f)
        has = (solid | wet) & (n_open > 0) & (own > 0) & (depth < dx / 4)
        area = n_open.astype(np.float32) * (dx * dx) * af
        cs = np.minimum(rhoc * depth * area, f32(0.5) * own)
        if chd is not None:                      # a char layer is one more resistance
            path = path + np.where(solid, chd, f32(0.0)) * (k / f32(CHAR_K))   # in the way
        g = k / np.maximum(path, f32(1e-9)) * area
        z = f32(0.0)
        Cs = np.where(has, cs, z).astype(np.float32)
        Cb = C - Cs
        return {"solid": solid, "wet": wet, "gas": gas, "n_out": n_out, "n_gas": n_gas,
                "af": af, "void": void, "has": has, "Cs": Cs,
                "Gsb": np.where(has, g, z).astype(np.float32), "Cb": Cb,
                "Csur": np.where(has, Cs, C), "Ccor": np.where(has, Cb, C)}

    def _mix_skin(self, ids, rho, c, k, scale, depth, path):
        """The skin depth and path (SKIN_T) of the cells that hold a mix, from
        the diffusivity of THAT mix, a = k / (rho c) (`scale` turns the units of
        rho c into J/(m3 K): 1000 for a solid in g/L and J/(g K), 1e6 for a liquid
        in g/ml). Every other cell, and a cell whose numbers are not finite,
        keeps the pure table's."""
        sel = ids > 0
        a = k[sel].astype(np.float64) / (rho[sel].astype(np.float64) * c[sel] * scale)
        good = np.isfinite(a) & (a > 0)
        depth, path = depth.copy(), path.copy()
        at = np.flatnonzero(sel)[good]
        depth.reshape(-1)[at] = np.sqrt(a[good] * SKIN_T[0]).astype(np.float32)
        path.reshape(-1)[at] = np.sqrt(a[good] * SKIN_T[1]).astype(np.float32)
        return depth, path

    def flame(self, x, y, z, watts):
        """A held flame: `watts` of its heat taken up by the surface of the cell
        at (x, y, z), for one tick. It warms the skin first, as a real flame
        does, so a match can light kindling long before the wood's middle is
        warm. Real sizes (F24): a candle or a lamp wick is about 80 W of flame;
        a surface held in it takes some tens of watts. A torch held to wood
        bathes ~0.01-0.02 m² at 50-100 kW/m²: one to two kilowatts.

        Any of x, y, z may be a slice: then `watts` goes into EACH cell of the
        block, as one call a cell would (a radiant panel over a slab)."""
        q = np.float32(watts * TICK_S)
        sl = tuple(c if isinstance(c, slice) else slice(c, c + 1) for c in (x, y, z))
        u = self._surfaces(sl)
        E, s = self.E[sl], self._skin_field()[sl]
        E += q
        if u.has.any():
            s[u.has] += q * (1.0 / u.Cs[u.has] - 1.0 / u.C[u.has])

    @property
    def hrr(self):
        """HEAT RELEASE RATE, W: what burned last tick, times its heat of
        combustion, per second (G4). What a calorimeter hood would read; the
        number every fire test is checked against."""
        t, w = getattr(self, "_hrr", (None, 0.0))
        return w if t == self.tick - 1 else 0.0

    # ── char, glow, blast ──────────────────────────────────────────────────
    def _char_field(self):
        """How deep the char layer is on each burning solid, m (see
        `_burn_cells`). Made on first use."""
        c = getattr(self, "char_d", None)
        if c is None or c.shape != self.shape:
            c = self.char_d = np.zeros(self.shape, np.float32)
        return c

    def char_depth(self):
        """Char depth, m, of every cell: the layer burning has turned to
        charcoal on the outside of the fuel, worked out from the mass the
        cell has lost (depth = mass lost / (density × area burning ×
        (1 - char yield))). Wood under a fire chars at 0.6-0.8 mm a minute
        (F5), which is where the burning rate of a slab puts it."""
        return np.where(self.mat != AIR, self._char_field(), 0.0).astype(np.float32)

    def glow(self):
        """How brightly each cell shines by its own heat, 0 to 1: nothing below
        the Draper point (525 °C, where a hot thing first glows red), white at
        1300 °C. Read from the surface temperature, as a person sees it."""
        T = self.T() + np.where(self.mat != AIR, self._skin_field(), 0.0)
        top = GLOW_COLORS[-1][0]
        return np.clip((T - GLOW_RED) / (top - GLOW_RED), 0.0, 1.0).astype(np.float32)

    @staticmethod
    def glow_name(T):
        """The smith's name for the colour of a thing at T °C, or None when it
        does not glow."""
        name = None
        for t0, nm in GLOW_COLORS:
            if T >= t0:
                name = nm
        return name

    def ignite(self, x, y, z, watts, seconds):
        """Hold a flame of `watts` (see `materials.IGNITER_W`) to the cell at
        (x, y, z) for `seconds`: a lamp left burning where it fell, a torch
        held to kindling. It is `flame` every tick until the time is up."""
        if not hasattr(self, "igniters"):
            self.igniters = []
        self.igniters.append((int(x), int(y), int(z), float(watts),
                              self.tick + int(round(seconds / TICK_S))))

    def _law_ignite(self):
        """The igniters held to cells: each puts its flame in, each tick."""
        keep = []
        for g in self.igniters:
            if g[4] > self.tick:
                self.flame(g[0], g[1], g[2], g[3])
                keep.append(g)
        self.igniters = keep

    def blast(self, x, y, z, lps):
        """A blast of air, `lps` litres a second, into the cell at (x, y, z) for
        one tick: a bellows' nozzle, a bellows' stroke, a mouth blowing on
        coals. The oxygen in it is shared among the faces burning within two
        cells, and each burns the extra fuel it can pay for (`_burn_cells`):
        more air, more heat, a hotter bed. Held up every tick, as `flame` is."""
        if not hasattr(self, "_blasts"):
            self._blasts = []
        self._blasts.append((int(x), int(y), int(z), float(lps)))

    def _blast_boost(self, u, Tsur, n_open):
        """Extra oxygen flux, g/(m²·s), on each burning face that a blast of
        air reaches: the blast's oxygen over the area of all the faces burning
        within two cells of its nozzle."""
        boost = np.zeros(self.shape, np.float64)
        dx = 0.1 * self.scale
        for (x, y, z, lps) in self._blasts:
            sl = tuple(slice(max(c - 2, 0), c + 3) for c in (x, y, z))
            hot = ((Tsur[sl] >= _IGN_ARR[self.mat[sl]]) & (self.smass[sl] > 0)
                   & (n_open[sl] > 0))
            area = np.where(hot, n_open[sl] * dx * dx * u.af[sl], 0.0)
            tot = float(area.sum())
            if tot > 0.0:
                boost[sl] += np.where(hot, lps * O2_PER_L / tot, 0.0)
        return boost

    # ── fires and their flames (F25) ───────────────────────────────────────
    def fires(self):
        """THE FIRES burning now, labelled by the world's own law (not by
        anyone's perception): a list of dicts, one per cluster of burning
        cells. `q_w` is its heat release; `d_m` the width of its foot (the
        diameter of the circle of its area); `height_m` the flame's length
        (Heskestad: 0.235 Q^0.4 - 1.02 D, Q in kW, with Q doubled against a
        wall and quadrupled in a corner, `walls` 0-2); `cells` how many cells
        burn; `x_m`, `y_m` the middle of its foot and `z_m` the height of
        its lowest burning cell."""
        t, f = _FIRES.get(self, (None, None))
        if f is None or t not in (self.tick, self.tick - 1):
            return []
        dx = 0.1 * self.scale
        return [{"q_w": float(f["q"][i]), "d_m": float(f["d"][i]),
                 "height_m": float(f["L"][i]), "walls": int(f["walls"][i]),
                 "cells": int(f["cells"][i]),
                 "x_m": float(f["cx"][i] * dx), "y_m": float(f["cy"][i] * dx),
                 "z_m": float(f["zlo"][i] * dx)} for i in range(len(f["q"]))]

    def _fires_and_flames(self, burnt, gas_heat, u):
        """Label this tick's fires, give each its flame, and let the flame
        heat what it touches out of the fire's own gas heat.

        THE FIRE: burning cells within FIRE_JOIN of each other are one fire;
        Q is the heat they released this tick; the foot is the columns they
        stand in, D the diameter of a circle of that area. THE FLAME: Heskestad's
        L = 0.235 Q^0.4 - 1.02 D (Q in kW), the length of the flame above the
        fire's highest cell; against a wall the flame is the one of twice the
        heat, in a corner of four times (the wall's mirror image). The cells
        it fills run from the burning face up to that length, stop at a solid,
        and run out sideways under it for what length is left. Flame cells
        heat every surface they touch, EPS_FLAME·σ·(T_flame⁴ - T⁴) on the open
        part of its face, and that is paid for out of this fire's gas heat
        (never more than FLAME_PAY_MAX of it): nothing is made. This is how a
        spill's flame reaches the bed 20 cm over it, how a crib lights from
        the bottom to the top, and how a flame climbs a wall."""
        f32 = np.float32
        nx, ny, nz = self.shape
        dx, dt = 0.1 * self.scale, TICK_S
        cells = _nonzero_flat(burnt)
        ix, iy, iz = np.unravel_index(cells, self.shape)
        ix, iy, iz = ix.astype(np.int64), iy.astype(np.int64), iz.astype(np.int64)
        bj = burnt.reshape(-1)[cells].astype(np.float64)
        lab, nf = _cluster(ix, iy, iz, FIRE_JOIN)
        q, ncol, zlo, ztp, x0, x1, y0, y1 = _fire_stats(lab, nf, ix, iy, iz, bj, nx, ny)
        Qkw = q / dt / 1000.0
        D = np.sqrt(4.0 * ncol * dx * dx / np.pi)
        L4 = np.maximum(0.235 * (2.0 * Qkw) ** 0.4 - 1.02 * D, 0.0)   # (against a wall)
        mg = int(np.ceil(float(L4.max()) / dx)) + 3
        wl = (max(int(x0.min()) - mg, 0), max(int(y0.min()) - mg, 0), max(int(zlo.min()) - 1, 0))
        wh = (min(int(x1.max()) + mg + 1, nx), min(int(y1.max()) + mg + 1, ny),
              min(int(ztp.max()) + int(np.ceil(float(L4.max()) / dx)) + 3, nz))
        sl = tuple(slice(a, b) for a, b in zip(wl, wh))
        shp = tuple(b - a for a, b in zip(wl, wh))
        st = np.zeros(shp, np.uint8)
        wallg = np.zeros(shp, np.uint8)
        ok = np.zeros(shp, bool)
        Ts = u.Ts[sl]
        _flame_prep(u.solid[sl], u.wet[sl], u.void[sl], u.smass[sl], u.mat[sl], u.fl[sl], Ts,
                    _IGN_ARR.astype(np.float64), OIL, FLUID[OIL][3], FLAME_PASS, st, wallg, ok)
        half = ztp + 1 + np.ceil(0.5 * np.maximum(0.235 * Qkw ** 0.4 - 1.02 * D, 0.0) / dx)
        walls = _walls(wallg, x0 - wl[0], x1 - wl[0], y0 - wl[1], y1 - wl[1],
                       zlo - wl[2], (half - wl[2]).astype(np.int64), FLAME_WALL)
        fac = np.array([1.0, 2.0, 4.0])[walls]
        L = np.maximum(0.235 * (fac * Qkw) ** 0.4 - 1.02 * D, 0.0)
        own = np.full(shp, -1, np.int32)
        wgt = np.zeros(shp, np.float32)
        _flame_grow(st, (ix - wl[0]).astype(np.int32), (iy - wl[1]).astype(np.int32),
                    (iz - wl[2]).astype(np.int32), lab, L / dx, own, wgt)
        # what the flame heats: every gift, written down once, then paid in the
        # share the fire it is asked of can pay
        tf4 = (self._flame_T + 273.15) ** 4
        A = dx * dx
        cap = 7 * int(np.count_nonzero(own >= 0)) + 16
        at, fire, amt = np.empty(cap, np.int64), np.empty(cap, np.int32), np.empty(cap)
        n = _flame_touch(own, st, wgt, ok, Ts, u.af[sl], u.n_out[sl], tf4, EPS_FLAME,
                         SIGMA, A, dt, at, fire, amt)
        at, fire, amt = at[:n], fire[:n], amt[:n]
        dem = np.bincount(fire, weights=amt, minlength=nf)
        G = np.bincount(lab, weights=gas_heat.reshape(-1)[cells], minlength=nf)
        pay = np.where(dem > 0.0,
                       np.minimum(1.0, FLAME_PAY_MAX * G / np.maximum(dem, 1e-30)), 0.0)
        if (pay > 0.0).any():
            got = np.bincount(at, weights=amt * pay[fire], minlength=int(np.prod(shp)))
            self._land(got.reshape(shp), sl)
            cut = 1.0 - pay * dem / np.maximum(G, 1e-30)
            gas_heat.reshape(-1)[cells] *= cut[lab]
        cx = np.bincount(lab, weights=ix + 0.5, minlength=nf) / np.bincount(lab, minlength=nf)
        cy = np.bincount(lab, weights=iy + 0.5, minlength=nf) / np.bincount(lab, minlength=nf)
        _FIRES[self] = (self.tick, {
            "q": q / dt, "d": D, "L": L, "walls": walls, "cells": np.bincount(lab, minlength=nf),
            "cx": cx, "cy": cy, "zlo": zlo, "own": own, "org": wl, "lab": lab,
            "at": (ix, iy, iz)})

    def _land(self, got, sl):
        """Energy (J) that landed on the cells of the box `sl`: it goes into
        the skin first, as a flame's heat does (`flame`). Only the cells that
        got any are touched."""
        hit = np.nonzero(got > 0.0)
        if not len(hit[0]):
            return
        geo = self._surface_geometry()
        at = tuple(h + (s_.start or 0) for h, s_ in zip(hit, sl))
        g = got[hit]
        C, Cs, has = geo.C[at], geo.Cs[at], geo.has[at]
        Cn = np.where(has, Cs, C)
        self.E[at] += g.astype(np.float32)
        self._skin_field()[at] += np.where(has, g * (1.0 / Cn - 1.0 / C), 0.0).astype(np.float32)

    # ── radiation across the air (G7) ──────────────────────────────────────
    def _law_radiate(self):
        """RADIATION ACROSS THE AIR, every RAD_EVERY ticks (G7). Sources: the
        radiant share (CHI_R) of every fire, as points through its flame
        (one for each 20 cm of it), and every surface hotter than RAD_HOT,
        clustered in 20 cm cells (the strongest RAD_MAX_SRC in all). Each
        lights the faces it can see (`_throw`): 1/r², the face's slant,
        smoke along the line, walls in the way, never past a flame's own
        surface emissive power. What a face absorbs lands in its skin, so a
        wall, a bar of iron or a person's cheek warms by it. The energy is the
        energy the fire and the hot faces had already shed (`shed`), so
        every joule that lands is a joule taken back out of it: nothing is
        made, and anything that would land past what was shed is scaled down."""
        k = self.tick - getattr(self, "_rad_last", self.tick - RAD_EVERY)
        self._rad_last = self.tick
        bank = getattr(self, "_rad_bank", 0.0)       # what the flames shed since the
        self._rad_bank = 0.0                         # last pass, spent or not
        if k <= 0:
            return
        kdt = k * TICK_S
        dx = 0.1 * self.scale
        A = dx * dx
        cs = 4
        S = []                                           # (x, y, z, P, D, nx, ny, nz)
        t, f = _FIRES.get(self, (None, None))
        if f is not None and t == self.tick:             # the fires' flames
            own, wl = f["own"], f["org"]
            fx, fy, fz = np.nonzero(own >= 0)
            fo = own[fx, fy, fz].astype(np.int64)
            if len(fo):
                key = ((fo * 4096 + fx // cs) * 4096 + fy // cs) * 4096 + fz // cs
                uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
                per = np.bincount(fo, minlength=len(f["q"]))
                fire_of = np.bincount(inv, weights=fo, minlength=len(uk)) / cnt
                fire_of = fire_of.astype(np.int64)
                P = CHI_R * f["q"][fire_of] * cnt / per[fire_of]
                px = np.bincount(inv, weights=fx + wl[0] + 0.5, minlength=len(uk)) / cnt
                py = np.bincount(inv, weights=fy + wl[1] + 0.5, minlength=len(uk)) / cnt
                pz = np.bincount(inv, weights=fz + wl[2] + 0.5, minlength=len(uk)) / cnt
                S.append((px, py, pz, P, np.zeros_like(P), np.zeros_like(P),
                          np.zeros_like(P), np.zeros_like(P)))
        # the hot surfaces
        hot_P = 0.0
        lo_h, hi_h = _hot_scan(self.E, self.heat_capacity(), self._skin_field(),
                               RAD_HOT - AMBIENT)
        box = None
        if (hi_h >= lo_h).all():
            box = tuple(slice(int(a), int(b) + 1) for a, b in zip(lo_h, hi_h))
        if box is not None:
            u = self._surfaces(box)
            surf = (u.solid & (u.smass > 0)) | u.wet
            hm = surf & (u.Ts >= RAD_HOT) & (u.n_gas > 0)
            hx, hy, hz = np.nonzero(hm)
            if len(hx):
                sc, fc = self._mix_window(box)
                e = np.where(u.solid, _mix_pick(self.mixes, "eps", self._emiss_table()[u.mat], sc),
                             _mix_pick(self.mixes, "eps", _FEMISS_ARR[u.fl], fc))[hm]
                tk = u.Ts[hm].astype(np.float64) + 273.15
                Pi = (e * SIGMA * (tk ** 4 - self._hot_env_k4(box, hm)) * A * u.af[hm]
                      * u.n_gas[hm])
                gx, gy, gz = (hx + (box[0].start or 0), hy + (box[1].start or 0),
                              hz + (box[2].start or 0))
                nvec = np.zeros((len(hx), 3))
                for d, (a, b, c) in enumerate(((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                                              (0, -1, 0), (0, 0, 1), (0, 0, -1))):
                    xx, yy, zz = gx + a, gy + b, gz + c
                    ok = (xx >= 0) & (yy >= 0) & (zz >= 0) & (xx < self.shape[0]) \
                        & (yy < self.shape[1]) & (zz < self.shape[2])
                    open_ = np.zeros(len(hx), bool)
                    open_[ok] = (self.mat[xx[ok], yy[ok], zz[ok]] == AIR) \
                        & (self.fvol[xx[ok], yy[ok], zz[ok]] <= FILM * self.cap)
                    nvec += open_[:, None] * np.array([a, b, c], float)
                nlen = np.sqrt((nvec ** 2).sum(axis=1))
                nh = nvec / np.maximum(nlen, 1e-9)[:, None]
                key = ((gx // cs) * 4096 + gy // cs) * 4096 + gz // cs
                uk, inv = np.unique(key, return_inverse=True)
                Pc = np.bincount(inv, weights=Pi, minlength=len(uk))
                hot_P = float(Pc.sum())
                pos = [np.bincount(inv, weights=Pi * (g + 0.5 + 0.5 * nh[:, a]),
                                   minlength=len(uk)) / np.maximum(Pc, 1e-30)
                       for a, g in enumerate((gx, gy, gz))]
                nn = [np.bincount(inv, weights=Pi * nh[:, a], minlength=len(uk))
                      / np.maximum(Pc, 1e-30) for a in range(3)]
                Dd = np.sqrt(nn[0] ** 2 + nn[1] ** 2 + nn[2] ** 2)
                S.append((pos[0], pos[1], pos[2], Pc, Dd,
                          nn[0] / np.maximum(Dd, 1e-9), nn[1] / np.maximum(Dd, 1e-9),
                          nn[2] / np.maximum(Dd, 1e-9)))
        if not S:
            return
        src = [np.concatenate([s[i] for s in S]) for i in range(8)]
        order = np.argsort(-src[3], kind="stable")[:RAD_MAX_SRC]
        src = [a[order] for a in src]
        src = [a[src[3] > 0.0] for a in src]
        if not len(src[3]):
            return
        rmax = min(RAD_RANGE, int(np.ceil(np.sqrt(src[3].max() / (4.0 * np.pi * RAD_Q_MIN)) / dx)) + 1)
        lo = [max(int(src[a].min()) - rmax, 0) for a in range(3)]
        hi = [min(int(src[a].max()) + rmax + 1, self.shape[a]) for a in range(3)]
        rb = tuple(slice(a, b) for a, b in zip(lo, hi))
        geo = self._surface_geometry()
        shp = tuple(b - a for a, b in zip(lo, hi))
        recv = np.zeros(shp, bool)
        od = np.zeros(shp, np.float32)
        alpha = np.zeros(shp, np.float32)
        _rad_prep(np.ascontiguousarray(geo.solid[rb]), np.ascontiguousarray(geo.wet[rb]),
                  np.ascontiguousarray(geo.smass[rb]), np.ascontiguousarray(geo.void[rb]),
                  np.ascontiguousarray(self.smoke[rb]), np.ascontiguousarray(geo.mat[rb]),
                  np.ascontiguousarray(geo.fl[rb]), _TRANSMIT_ARR, _EMISS_ARR, _FEMISS_ARR,
                  SMOKE_EXT / self.vox_l * dx, recv, od, alpha)
        burning = np.zeros(shp, bool)                    # the faces already alight
        if f is not None and t == self.tick:
            bx, by, bz = f["at"]
            ins = ((bx >= lo[0]) & (bx < hi[0]) & (by >= lo[1]) & (by < hi[1])
                   & (bz >= lo[2]) & (bz < hi[2]))
            burning[bx[ins] - lo[0], by[ins] - lo[1], bz[ins] - lo[2]] = True
        got = np.zeros(recv.shape, np.float64)
        landed = np.zeros(len(src[3]))
        qcap = EPS_FLAME * SIGMA * (self._flame_T + 273.15) ** 4
        _throw(recv, np.ascontiguousarray(geo.gas[rb]), od,
               np.ascontiguousarray(geo.af[rb]), alpha, burning,
               src[0] - lo[0], src[1] - lo[1], src[2] - lo[2], src[3], src[4],
               src[5], src[6], src[7], dx, kdt, RAD_Q_MIN, qcap,
               got, landed)
        total = float(got.sum())
        if total <= 0.0:
            return
        budget = bank + hot_P * kdt
        if total > budget:                               # never more than was shed
            got *= budget / total
            total = budget
        self.shed -= total
        self._land(got, rb)
        self._rad_landed = getattr(self, "_rad_landed", 0.0) + total

    # ── conduction, convection and loss through the faces ──────────────────
    def _cavity_view(self, void, dx):
        """THE CAVITY VIEW of a porous cell (wave 14a, H23): the share of a face
        that looks onto glowing lumps instead of a wall. A bar buried in a bed of
        coals is not touched by the empty share of the cell (that is 0.2 for a
        packed bed); it sits in a cavity whose every wall is hot, and a ray from
        its face crosses `dx / LUMP_M` lumps' worth of gaps before it is stopped.
        So the view is 1 - fill^N: 0 for a dense cell, 0.59 for the packed bed at
        5 cm (fill 0.8, N 4), near 1 for a loose heap or a coarse cell. A tiny
        void (under a percent) stays small: 1 - (1 - v)^4 is about 4 v."""
        if not void.any():
            return void
        n = np.float32(float(dx) / LUMP_M)
        m = void > 0                                     # only the porous cells pay for the power
        out = np.zeros(void.shape, np.float32)
        out[m] = np.float32(1.0) - np.power(np.float32(1.0) - void[m], n)
        return out

    def _law_conduct(self, sl=None):
        """Heat across every face, in SI: W per kelvin of difference, times
        TICK_S. What a face is decides its law:
          solid | solid   conduction, k the harmonic mean of the two halves (H9),
                          plus radiation across the gaps of a partly filled cell
          gas | gas       still air, plus the eddies of hot air (EDDY)
          liquid | liquid the liquid's own k
          solid | gas     free convection H_CONV, plus the radiation of hot gas
          liquid | gas    onto a cooler surface (EPS_GAS)
          solid | liquid  the liquid's h (FLUID, last column)
        A surface meets the outside with its SKIN (`_surfaces`) and whatever
        is behind it with its core. Every face that looks onto gas also
        radiates, e·σ·A·(T⁴ − T∞⁴), and that heat leaves the world (`shed`): it
        is the only loss there is (H14, H15). Every face is clipped
        (FACE_CLIP), so no cell passes the temperature of its hottest
        neighbour (H18, H19)."""
        sl = sl if sl is not None else (slice(None),) * 3
        E = self.E[sl]                                   # a view: writing to it
        skin = self._skin_field()[sl]                    # writes the world
        f32 = np.float32
        dx, dt = f32(0.1 * self.scale), f32(TICK_S)
        A = dx * dx
        u = self._surfaces(sl)
        solid, wet, gas, T, Ts, Tb = u.solid, u.wet, u.gas, u.T, u.Ts, u.Tb
        mat, fl = u.mat, u.fl
        # conductivity of each cell's half of a face. A cell that holds a mix
        # (T1) reads the mix's k and emissivity; with no mix in the world the
        # pick returns the pure table's array itself (`_mix_pick`).
        sc, fc = self._mix_window(sl)
        k = np.where(solid, _mix_pick(self.mixes, "k", _KSOLID_ARR[mat], sc),
                     _mix_pick(self.mixes, "k", _FLK_ARR[fl], fc))
        ref = self._rest_ref(sl)       # cold (T8): warm against the cell's own rest
        eddy = (f32(0.0) if self.gas_momentum else f32(1000.0 * AIR_C / EDDY)) * np.maximum(
            T - AMBIENT if ref is None else T - (AMBIENT + ref), 0.0)
        k = np.where(gas, _KSOLID_ARR[AIR] + eddy, k).astype(np.float32)
        h_liq = _FK_ARR[fl]
        eps = _mix_pick(self.mixes, "eps", self._emiss_table()[mat], sc)
        eps = self._eps_over(eps, sl)                    # T3: what is on the faces
        # a surface already burning sits inside its own flame, whose heat it
        # gets as q_fl (`_law_burn`): the hot gas beside it does not ALSO heat
        # it, or the flame would be counted twice (it can still lose heat to
        # gas cooler than itself)
        lit = np.where(solid, (Ts >= _IGN_ARR[mat]) & (u.smass > 0),
                       (fl == OIL) & (Ts >= FLUID[OIL][3]))
        af = u.af
        void = self._cavity_view(u.void, dx)             # what a pore looks onto
        qs = np.zeros(E.shape, np.float32)               # into skins
        qc = np.zeros(E.shape, np.float32)               # into cores
        bc = wet & (fl == WATER)                         # water: the boiling curve
        eff = self._effusivity(solid, mat, fl, sc, fc) if bc.any() else None
        if eff is None:
            eff = np.zeros(E.shape, np.float32)
        # THE FACE-ENVIRONMENT BUNDLE (faceenv.py): inert unless a lane fills it
        # in, and then the same for the compiled law and the numpy one below.
        env = self._face_env(sl)
        if env.env and not env.fits(Ts.shape):           # (compiled code does not check)
            raise ValueError(f"the face-environment bundle is cut to "
                             f"{env.bz.shape if env.overlay else env.bp.shape}, "
                             f"the cells it is used on are {Ts.shape}")
        flux = []                                        # face by face, if asked (env.ff)
        if self.fused and HAVE_NUMBA:
            kind = np.where(solid, 0, np.where(gas, 1, 2)).astype(np.uint8)
            for di, dj, dl in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                fq = _face_heat(Ts, Tb, u.Csur, u.Ccor, k, h_liq, kind, lit, void, eps, af,
                                bc, eff, qs, qc, dx, dt, A, f32(H_CONV), _EPSIG, _FOURSIG, _K0,
                                f32(FACE_CLIP), di, dj, dl, env.env, env.bz, env.ov, env.bp)
                if env.ff:
                    flux.append(fq)
        for axis in (range(3) if not (self.fused and HAVE_NUMBA) else ()):
            a = [slice(None)] * 3; b = [slice(None)] * 3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            sa, sb = solid[a], solid[b]
            ga, gb = gas[a], gas[b]
            wa, wb = wet[a], wet[b]
            ua = (sa & ~sb) | (wa & gb)                  # a meets b with its skin
            uc = (sb & ~sa) | (wb & ga)
            Ta = np.where(ua, Ts[a], Tb[a])
            Tc = np.where(uc, Ts[b], Tb[b])
            Ca = np.where(ua, u.Csur[a], u.Ccor[a])
            Cc = np.where(uc, u.Csur[b], u.Ccor[b])
            ka, kb = k[a], k[b]
            G = 2.0 * ka * kb / np.maximum(ka + kb, 1e-12) * dx
            ss = sa & sb
            if ss.any():                                 # radiation across the
                tka, tkc = Ta + _K0, Tc + _K0            # gaps of a porous cell
                ea, ec = eps[a], eps[b]
                rad = (np.maximum(void[a], void[b]) * (ea * ec / np.maximum(ea + ec - ea * ec, f32(1e-12)))
                       * (_FOURSIG * f32(0.25) * (tka * tka + tkc * tkc) * (tka + tkc)) * A)
                G = np.where(ss, G + rad, G)
            sw = (sa & wb) | (wa & sb)
            if sw.any():
                hh = np.where(wa, h_liq[a], h_liq[b])
                tw, tl = np.where(wa, Tc, Ta), np.where(wa, Ta, Tc)
                # the boiling point of the liquid: BOIL, or its own (bundle `bp`)
                tbo = _F_BOIL if not env.boils else np.where(wa, env.bp[a], env.bp[b])
                boil = np.where(wa, bc[a], bc[b]) & (tw > tbo) & (tw > tl)
                if boil.any():                           # the boiling curve
                    ew = np.where(wa, eps[b], eps[a])
                    es = np.where(boil, np.where(wa, eff[b], eff[a]), f32(1.0))
                    el = np.where(boil, np.where(wa, eff[a], eff[b]), f32(1.0))
                    tc = (es * tw + el * tl) / (es + el)
                    dT = tc - tbo
                    x = dT / _F_CHF
                    tk, sk = tc + _K0, tbo + _K0
                    q = np.where(
                        dT <= _F_CHF, _F_NUC * dT + _F_QCHF * (x * x * x),
                        np.where(dT < _F_LEID,
                                 _F_QPEAK + (_F_QMIN - _F_QPEAK) * ((dT - _F_CHF) / (_F_LEID - _F_CHF)),
                                 _F_FILM * dT + ew * (_FOURSIG * f32(0.25)) * ((tk * tk + sk * sk) * (tk + sk)) * dT))
                    hb = q / np.where(boil, tw - tl, f32(1.0))
                    hb = np.where(dT < _F_LEID, np.maximum(hb, hh), hb)   # (see `_hboil`)
                    hh = np.where(boil & (dT > f32(0.0)), hb, hh)
                Gw = hh * A * af[a] * af[b]
                if env.overlay:                          # immersion: the solid cell's overlay
                    oid = ids_np(np.where(sa, env.bz[a], env.bz[b]), env.ov.shape[0])
                    Gw = np.where(oid != 0, exchange_np(
                        Gw, env.ov, oid, A * af[a] * af[b]), Gw)
                G = np.where(sw, Gw, G)
            sg = ((sa | wa) & gb) | (ga & (sb | wb))     # a surface and the gas
            if sg.any():
                tk_a, tk_c = Ta + _K0, Tc + _K0
                gas_hot = np.where(ga, Ta > Tc, Tc > Ta)
                burns = np.where(ga, lit[b], lit[a])
                rad = np.where(gas_hot & ~burns, _EPSIG * (tk_a * tk_a + tk_c * tk_c)
                               * (tk_a + tk_c), 0.0)
                if not env.overlay:
                    G = np.where(sg, np.where(gas_hot & burns, f32(0.0),
                                              (H_CONV + rad) * A * af[a] * af[b]), G)
                else:                                    # the surface cell's overlay (BH2, BH3)
                    oid = ids_np(np.where(ga, env.bz[b], env.bz[a]), env.ov.shape[0])
                    hc, radw = film_np(env.ov, oid, Ta - Tc, H_CONV, rad)
                    Ga = (hc + radw) * A * af[a] * af[b]
                    Ga = np.where(oid != 0, exchange_np(
                        Ga, env.ov, oid, A * af[a] * af[b]), Ga)
                    G = np.where(sg, np.where(gas_hot & burns, f32(0.0), Ga), G)
            d = Ta - Tc
            q = (G * dt * d).astype(np.float32)
            lim = f32(FACE_CLIP) * np.minimum(Ca, Cc) * np.abs(d)
            q = np.clip(q, -lim, lim)
            if env.ff:
                flux.append(q)
            qa = np.where(ua, q, 0.0)
            qs[a] -= qa
            qc[a] -= q - qa
            qb = np.where(uc, q, 0.0)
            qs[b] += qb
            qc[b] += q - qb
        # radiation out of the world, from every surface that looks onto gas
        # (the seam `_env_radiation`: what the surface radiates TO)
        rad_from = (solid | wet) & (u.n_gas > 0)
        radiates = rad_from.any()
        if radiates:
            loss = self._env_radiation(u, eps, sl)
        else:
            loss = np.zeros(E.shape, np.float32)
        E += qs + qc
        if radiates:
            was = float(E.sum(dtype=np.float64))
            E -= loss
            gone = was - float(E.sum(dtype=np.float64))
            self.shed += gone
            self._radiated(gone)
        # the skin: what reached it, what reached the core, then the slow
        # leak between them (exact, so it cannot overshoot)
        has = u.has
        if has.any():
            Cs_, Cb_ = np.where(has, u.Cs, f32(1.0)), np.where(has, u.Cb, f32(1.0))
            ts = Ts + (qs - loss) / Cs_
            tb = Tb + qc / Cb_
            gap = (ts - tb) * np.exp(-env.gsb(u, dx) * dt * (1.0 / Cs_ + 1.0 / Cb_))
            skin[...] = np.where(has, gap * Cb_ / u.C, 0.0)
        else:
            skin[...] = 0.0
        if flux:
            self._face_flux(sl, flux)

    def _face_env(self, sl):
        """THE SEAM for the face-environment bundle (faceenv.py): what the
        face law should know about the cells of `sl` besides their matter. The
        world's lazy `face_env` (a `FaceEnv`, None until a lane makes one)
        cut to the window, or the inert bundle, with which the law is the
        one it always was.

        Two things of the material row fill the bundle when they are on, and
        are nothing when they are off (every world today):
          * `film_cols` (D4): a surface whose material has a natural-convection
            column (`_HNAT_ARR`, FLESH) gets that overlay row; every other
            material keeps the default row, the constant H_CONV;
          * a liquid that holds a mix (T1, T4) gets its own boiling point
            (`_boil_points`), so the boiling curve is measured from it.
        A world that sets its own `face_env` with an overlay and also turns
        `film_cols` on is refused: the two id spaces are merged in 13b."""
        fe = getattr(self, "face_env", None)
        bp = self._boil_points(sl)
        cols = self.film_cols
        if cols:
            if fe is not None and fe.overlay:
                raise ValueError("film_cols and a face_env overlay cannot both be on yet "
                                 "(their row ids would collide; merged in 13b)")
            ids = _FILM_ID[self.mat[sl]]
            if not ids.any():
                cols = False                     # no such surface here: the old law
        win = None if fe is None else fe.window(sl)
        if win is None:
            if not cols and bp is None:
                return NO_ENV
            return FaceEnv(bz=ids if cols else None, ov=_FILM_OV if cols else None, bp=bp)
        if win.boils:
            bp = win.bp                          # the world's own boiling points win
        if not cols and (bp is None or bp is win.bp):
            return win
        if win.overlay:
            return FaceEnv(bz=win.bz, ov=win.ov, bp=bp, ff=win.ff)
        return FaceEnv(bz=ids if cols else None, ov=_FILM_OV if cols else None,
                       bp=bp, ff=win.ff)

    def _mix_window(self, sl):
        """(solid mix ids, fluid mix ids) of the cells of `sl`, or (None, None)
        while no cell holds a mix: the gathers then take the pure tables."""
        if not self.mixes_live:
            return None, None
        return self.scomp[sl], self.fcomp[sl]

    def _eps_over(self, eps, sl):
        """The emissivity array with the surface layers on it (T3). The plain world
        has none: the array comes back untouched. `LayersMixin` overrides this."""
        return eps

    def _emiss_table(self):
        """The per-material emissivity the laws read: the old `_EMISS_ARR`, or
        the film column of the row (FLESH 0.98) when `film_cols` is on."""
        return _EMISS_FILM_ARR if self.film_cols else _EMISS_ARR

    def _effusivity(self, solid, mat, fl, sc, fc):
        """sqrt(k rho c) of each cell, W s^1/2/(m2 K), for the boiling curve: the
        pure tables, and for a cell that holds a mix its own k, rho and c."""
        eff = np.where(solid, _EFF_ARR[mat], _FEFF_ARR[fl])
        if sc is not None and (sc > 0).any():
            at = (sc > 0) & solid
            if at.any():
                mx, ids = self.mixes, sc[at]
                e = np.sqrt(mx.prop("k", ids).astype(np.float64) * mx.prop("rho", ids)
                            * mx.prop("c", ids) * 1000.0)
                eff[at] = np.where(np.isfinite(e), e, eff[at])
        if fc is not None and (fc > 0).any():
            at = (fc > 0) & ~solid
            if at.any():
                mx, ids = self.mixes, fc[at]
                e = np.sqrt(mx.prop("k", ids).astype(np.float64) * (mx.prop("rho", ids) / 1000.0)
                            * mx.prop("c", ids) * 1.0e6)
                eff[at] = np.where(np.isfinite(e), e, eff[at])
        return eff

    def _boil_points(self, sl):
        """T4: the boiling point, C, of the liquid in each cell of `sl` as a
        float32 array for the face bundle, or None when no cell holds a mixed
        liquid (every world today): the boiling curve then uses BOIL.
        BOIL + the rise of the mix's water activity (`heatmix.boil_rise`)."""
        if not self.mixes_live:
            return None
        fc = self.fcomp[sl]
        at = (fc > 0) & (self.fvol[sl] > 0)
        if not at.any():
            return None
        bp = np.full(fc.shape, _F_BOIL, np.float32)
        bp[at] = np.float32(BOIL) + _heatmix.boil_rise(self.mixes, fc[at])
        return bp

    def _face_flux(self, sl, fluxes):
        """THE SEAM for the flux readout (T10): called once a tick, after the
        face law, only when the bundle asks (`ff`), with the joules each face
        carried, one array per axis, signed from the lower cell to the higher.
        Nothing to do by default; a reader must write nothing back."""

    def _rest_ref(self, sl):
        """THE SEAM for the eddy term's reference (conduct, gas | gas): the K
        above 20 C that a gas cell's warmth is measured against, per cell of
        `sl`, or None for the fixed ambient (the old law). Cold (T8) gives each
        cell its own rest: the outside air for outdoor cells."""
        return None

    def _radiated(self, gone):
        """THE SEAM after the radiation of a surface is taken: `gone` is the
        float64 joules that actually left the cells of the box (already in
        `shed`). Nothing to do by default; cold (T8) books it by cause, so the
        books equal `shed` to the last bit."""

    def _hot_env_k4(self, box, hm):
        """THE SEAM for what a hot surface radiates TO in the radiation law
        (`_law_radiate`): T^4 in K^4, a number or one per cell of the mask `hm`
        in `box`. The default is the fixed ambient. Cold (T8) gives the cell's
        own environment, so the books close in a climate world."""
        return (AMBIENT + 273.15) ** 4

    def _env_radiation(self, u, eps, sl):
        """THE SEAM for what a surface radiates TO. Returns the joules each
        cell of `sl` loses by radiating out of the world this tick (float32,
        zero for a cell that does not look onto gas) and books nothing: the
        caller takes the difference into `shed`. It is only asked when some
        cell does look onto gas. The default is the old law: every such surface
        radiates to the fixed ambient, e·σ·A·(T⁴ − T∞⁴), and cold (T8) replaces
        it with a per-cell environment temperature and a radiant node per room.
        `u` is the `_surfaces` block of the cells of `sl`, `eps` the emissivity
        of each cell's solid."""
        f32 = np.float32
        dx, dt = f32(0.1 * self.scale), f32(TICK_S)
        A = dx * dx
        Ts, af = u.Ts, u.af
        rad_from = (u.solid | u.wet) & (u.n_gas > 0)
        fc = self._mix_window(sl)[1]
        e = np.where(u.solid, eps, _mix_pick(self.mixes, "eps", _FEMISS_ARR[u.fl], fc))
        tk = Ts + _K0                                    # T⁴ − T∞⁴ written so it is
        fourth = (tk * tk + _TINF2) * (tk + _TINF) * (Ts - AMBIENT)   # EXACTLY 0 at
        loss = (e * f32(SIGMA) * A * af * u.n_gas * dt * fourth)      # ambient
        lim = f32(FACE_CLIP) * u.Csur * np.abs(Ts - AMBIENT)
        return np.where(rad_from, np.clip(loss, -lim, lim), 0.0).astype(np.float32)

    def _law_rise(self, sl=None):
        """Buoyancy: hot gas rises. A hot air voxel hands much of its excess heat to the
        air above it; at the open top of the world, the plume leaves entirely. THIS is why
        a doused fire can stay out — the reheating cloud escapes upward instead of sitting.

        Written in SI, times TICK_S. Warm gas rises at its BUOYANT speed,
        v = sqrt(g · dx · ΔT / T), the speed a parcel one cell tall reaches
        when it is ΔT warmer than the gas above it (T in kelvin): about
        1.3 m/s for flame gas (ΔT ~ 1000 K) at 5 cm, 0.4 m/s for air 100 K
        warm, 0.1 m/s for air 5 K warm. In one tick it carries v · TICK_S / dx
        of its cell's gas, and so that share of its excess heat, up through
        the face. A share is never more than half (RISE_MAX), where the two
        cells would be even: a hot plume at 5 cm would want more than that in
        a tick (2-5 m/s over a real flame), and is held at the limit. The top
        layer of an open world loses its excess to the open air above the
        same way, out of the world (`shed`).

        It was a fixed share, 14 a second (and 20 out of the top), whatever
        the difference: a degree of warmth rose as fast as a flame. This
        moves heat only; the gas itself (its smoke, its oxygen) is moved by
        `_law_smoke` and `_law_air`, and carrying heat with the gas parcel is
        G7's business."""
        sl = sl if sl is not None else (slice(None),) * 3
        E, mat = self.E[sl], self.mat[sl]
        T = AMBIENT + E / self.heat_capacity()[sl]
        lo = (slice(None), slice(None), slice(None, -1))
        up = (slice(None), slice(None), slice(1, None))
        both_air = (mat[lo] == AIR) & (mat[up] == AIR)
        dT = T[lo] - T[up]
        # only the GAS'S share of the cell's heat rises (c_airbase, the air's own
        # capacity) — a pool of liquid is not a plume, its heat stays put. Found the
        # hard way: rising 0.35*C_total blew a lamp's oil heat skyward and no taper
        # could ever light a pool. For pure air the two are identical.
        rises = both_air & (dT > 0)
        q = np.zeros(dT.shape, np.float32)
        if rises.any():
            q[rises] = (self._rise_share(dT[rises], T[up][rises]) * self.c_airbase
                        * dT[rises]).astype(np.float32)
        E[lo] -= q
        E[up] += q
        self._top_sink()

    def _top_sink(self):
        """THE SEAM for the open top of the world (the end of `_law_rise`): a
        plume that reaches open sky is gone, and the world says so (`shed`).
        The whole top layer, whatever window the rise ran in. The default is
        the old law, which measures the excess against the fixed ambient;
        cold (T8) measures it against the outside air."""
        top = (slice(None), slice(None), -1)
        sky = (self.mat[top] == AIR) & (self.fvol[top] < 0.1 * self.cap)
        if sky.any():                                    # open sky: the plume is
            was = float(self.E[top].sum(dtype=np.float64))   # gone, and the
            Et = self.E[top]                                  # world says so
            Tt = AMBIENT + Et / self.heat_capacity()[top]
            ex = np.maximum(Tt - AMBIENT, 0.0)
            Et[sky] *= (1.0 - self._rise_share(ex, AMBIENT))[sky].astype(np.float32)
            self.shed += was - float(self.E[top].sum(dtype=np.float64))

    def _rise_share(self, dT, T_above):
        """The share of a cell's gas that rises through its top face in one
        tick when it is dT warmer than the gas above it (`_law_rise`)."""
        dx = 0.1 * self.scale
        v = np.sqrt(GRAVITY * dx * np.maximum(dT, 0.0) / (T_above + 273.15))
        return np.minimum(v * (TICK_S / dx), RISE_MAX)

    def _gas_access(self):
        """Fire breathes: a voxel can only burn with gas nearby — itself or a neighbor
        that is open air not drowned in fluid. A water blanket or a full seal SMOTHERS
        combustion regardless of temperature (the oxygen field proper comes later;
        this is its contact condition)."""
        airish = (self.mat == AIR) & (self.fvol < 0.5 * self.cap)
        return airish | (_count_nb(airish) > 0)

    def _law_burn(self):
        """Burning, then the radiation the fire and the hot things throw across
        the air (`_law_radiate`, every RAD_EVERY ticks). A wet skin is held at the
        boil first (`_law_skin_water`)."""
        self._law_skin_water()
        self._burn_cells()
        if self.tick % RAD_EVERY == 0:
            self._law_radiate()

    def _law_skin_water(self):
        """A SKIN THAT HOLDS WATER CANNOT PASS 100 °C while the water lasts
        (F-T9, the wood half: a doused crib is put out, not just wetted). The
        outer layer of a porous solid holds what it has soaked up
        (`World.skin_water`, ml). Heat that would take that layer above the
        boiling point goes into boiling that water instead, 2260 J a gram, and
        the steam leaves as it does in `_law_boil`. Where the water runs out,
        the part of the excess it could not pay stays. It runs before
        `_law_burn`, so a wet face is not hot enough to burn, and what the
        flame gave it this tick is what the water takes.

        Only cells with water in them (a fluid cell of solid material) and a
        skin hotter than the boil are looked at, so a dry world pays one mask."""
        wet = (self.fl == WATER) & (self.fvol > 0) & (self.mat != AIR) & (self.smass > 0)
        ok = self.legacy_ok(solid=False)         # D6: a mixed liquid is not boiled here
        if ok is not None:                       # (heatmix.CRUTCHES, "_law_skin_water")
            wet &= ok
        if not wet.any():
            return
        geo = self._surface_geometry()
        skin = self._skin_field()
        at = np.nonzero(wet & geo.has & (skin > 0))
        if not len(at[0]):
            return
        C, Cs, Cb = geo.C[at], geo.Cs[at], geo.Cb[at]
        Ts = AMBIENT + self.E[at] / C + skin[at]
        hot = Ts > BOIL
        if not hot.any():
            return
        sk = self.skin_water()[at]
        d = np.where(hot, Ts - BOIL, 0.0)
        ml = np.minimum(sk, d * Cs / LATENT)
        go = ml > 0
        if not go.any():
            return
        at = tuple(a[go] for a in at)
        ml, Cs, Cb, C = ml[go], Cs[go], Cb[go], C[go]
        dE = ml * LATENT                              # taken from the skin node
        self.E[at] -= dE.astype(np.float32)
        skin[at] -= (dE / Cs * Cb / C).astype(np.float32)
        self.fvol[at] -= ml.astype(np.float32)
        self.p_add[at] += (ml * 0.1).astype(np.float32)
        dry = (self.fl == WATER) & (self.fvol <= 0.01)
        if ok is not None:
            dry &= ok
        if dry.any():
            self.fl[dry] = NOFLUID
            self.fvol[dry] = 0.0
            self.fpot[dry] = 0.0

    def _burn_cells(self):
        """COMBUSTION, in SI (G4). Burning happens at SURFACES: a face open to
        gas, at the skin's temperature (`_law_conduct`).

        A solid that flames (PYRO) burns like water boils: its skin sits at the
        ignition temperature, and the heat above it turns solid into gas at L_g
        a gram. The gas burns: of its heat, the flame gives the surface back up
        to q_fl W/m², CHI_R leaves as radiation, and the rest rises as hot gas
        into the open cells beside it — where it is the flame, and heats
        whatever else it touches. Charcoal (GLOW) burns AT its surface, with no
        flame, faster the hotter it is. Oil is a pool: it burns at its top.

        No face burns faster than its material's rate (g/(m²·s)), nor with less
        oxygen than its limit: a flame keeps its full rate until the last
        quarter of the way from fresh air to the limit, then fails in a straight
        line to nothing at it (`_o2_factor`; flaming stops near 15% O2, F11).
        The heat-release rate is `hrr`."""
        # nothing hot enough to burn, judged by the warmest part of each cell
        # (skin or core: neither is further from the cell's average than its
        # skin is, as a skin holds no more than its core): then nothing below
        # can change, and it is not worth asking
        T0 = self.T()
        hi = T0 + np.abs(self._skin_field())
        warm = (hi >= _IGN_ARR[self.mat]) & (self.smass > 0)
        if not (warm.any()
                or ((self.fl == OIL) & (self.fvol > 0) & (hi >= FLUID[OIL][3])).any()):
            self._hrr = (self.tick, 0.0)
            _FIRES[self] = (self.tick, None)
            self._blasts = []
            return
        full = (slice(None),) * 3
        u = self._surfaces(full)
        T, C, has, Tsur = u.T, u.C, u.has, u.Ts
        skin = self._skin_field()
        dx, dt = 0.1 * self.scale, TICK_S
        A = dx * dx
        air = self.mat == AIR
        airish = air & (self.fvol < 0.5 * self.cap)
        n_open = _count_nb(airish)
        access = airish | (n_open > 0)
        Cnode = np.where(has, u.Cs, C)
        # a fire ENTRAINS air — its plume drags in far more oxygen than diffusion
        # brings. Each burning voxel may draw an allowance from ITS OWN airspace,
        # scaled by how much oxygen that airspace still holds: a sealed room's fire
        # starves as the room empties, the room NEXT DOOR keeps its air unless a
        # doorway (or a burned-through wall) joins the two into one lung.
        fresh = O2_PER_L * self.vox_l
        # ONLY THE CELLS THAT CAN BURN (`sleep_regions`). Every cell a solid
        # burns in this tick is among the `warm` ones: it is that material
        # (or charcoal its wood or flesh became this tick, which was hotter
        # than that wood's point), it has mass, and its skin is past its
        # point, which no part of a cell is hotter than `hi`. So each
        # material's burning cells are picked from that list, and every
        # array is read and written at those cells by index, in the order a
        # boolean mask gives them, rather than through a mask swept over the
        # whole world twenty times a material. Which airspace each one
        # breathes is asked of those cells alone, too (`_region_at`).
        sleeps = self._sleeps()
        warm_at = np.flatnonzero(warm) if sleeps else None
        surf = None if sleeps else self._region_of_surface()

        def surf_of(at):
            return surf[at] if surf is not None else self._region_at(at)
        cshape = tuple((s + self._CS - 1) // self._CS for s in self.shape)
        drawn_c = np.zeros(cshape, np.float64)               # entrained O2, owed by the
                                                             # fire's OWN neighborhood
        drawn_r = np.zeros(self._region_count, np.float64)   # and only ever paid by the
                                                             # fire's OWN airspace
        # entrainment health is LOCAL (this cell's air), not the room average —
        # so a fire beside the doorway keeps breathing the draft while the far
        # corner starves first, and the blaze TAPERS instead of dying as one
        o2c = self._cells_where(air, self.o2)
        # WHERE THE AIR IS does not change when the air changes — only when the
        # WALLS do. This was a full downsample of the lattice plus a wide blur
        # of it, rebuilt every tick, for a field that depends on nothing but
        # which voxels are air. Thrown away by `_air_regions`, which already
        # compares the air mask against last tick's and so already knows: no
        # extra comparison, and right by construction rather than by anyone
        # remembering that a burning wall changes where the air is.
        #
        # Keyed on the MATTER fields first, which was the obvious thing and the
        # wrong one — a fire eats wood, `smass` changes every tick, and the
        # cache was thrown away every tick for a field that had not moved.
        airc = self._aircells
        if airc is None:
            airc = self._aircells = self._cells(air.astype(np.float32))
        # a plume's CATCHMENT: it entrains from meters around, not from its own
        # shoebox — health and the debt below must use the SAME reach, or the
        # fire eats a small pocket, reads the vacuum as suffocation, and
        # strangles itself while the room is still full of air (measured).
        # The blur is AIR-WEIGHTED: a wall is not a region of zero oxygen, it
        # is simply not air — averaging it in as 0 suffocated every fire that
        # burned near a floor (measured: health 0.21 under a fresh sky)
        reach = max(2, int(round(PLUME_REACH_M / (0.1 * self.scale * self._CS))))
        o2b = self._blur_cells(o2c, reach)               # o2c is already per-SLOT
        airb = self._airblur
        if airb is None:
            airb = self._airblur = np.maximum(
                self._blur_cells(airc, reach), 1e-6)
        hc = np.clip((o2b / airb) / fresh, 0.0, 1.0)     # so o2b/airb = per-AIR-voxel
        health_vox = None if self._sleeps() else self._upcell(hc).astype(np.float32)
        # the cell blur cannot see a voxel-thin wall — ceiling each fire's
        # health by ITS airspace's true average, so a sealed room's fire still
        # starves as the room empties, whatever the far side of the wall holds
        reg_air = self.air_region[air]
        reg_frac = np.ones(max(self._region_count, 1), np.float32)
        if air.any():
            reg_frac = (np.bincount(reg_air, weights=self.o2[air],
                                    minlength=self._region_count)
                        / np.maximum(np.bincount(reg_air,
                                                 minlength=self._region_count), 1)
                        / fresh).astype(np.float32)
            reg_frac = np.clip(reg_frac, 0.0, 1.0)

        def health_of(at):
            r = surf_of(at)
            if health_vox is not None:
                hv = health_vox[at]
            else:                                    # `_upcell(hc)[at]`, read off
                cs = self._CS                        # the coarse cells directly
                at = at if isinstance(at, tuple) else np.nonzero(at)
                hv = hc[at[0] // cs, at[1] // cs, at[2] // cs].astype(np.float32)
            return np.where(r >= 0, np.minimum(hv, reg_frac[np.maximum(r, 0)]), 0.0)

        def ration(mask, ratio, dm_want, area, health):
            """O2-limit dm for the masked voxels; consume local pores first, log the
            entrained remainder against the voxel's own coarse-cell neighborhood
            (the plume pulls from the meter around it, not from the far corner).
            What a face may pull in is O2_FACE per m² of it, scaled by how much
            oxygen its air still holds."""
            r = surf_of(mask)
            allow = O2_FACE * area * dt * health * (r >= 0)
            dm = np.minimum(dm_want, (self.o2[mask] + allow) / ratio)
            local = np.minimum(dm * ratio, self.o2[mask])
            self.o2[mask] -= local
            need = dm * ratio - local
            ok = (r >= 0) & (need > 0)
            if ok.any():
                cells = (np.stack(mask, axis=1) if isinstance(mask, tuple)
                         else np.argwhere(mask))[ok] // self._CS
                np.add.at(drawn_c, tuple(cells.T), need[ok])
                np.add.at(drawn_r, r[ok], need[ok])
            return dm

        gas_heat = np.zeros(self.shape, np.float64)          # flame heat for the
        released = 0.0                                       # open cells beside it
        radiated = 0.0
        burnt = np.zeros(self.shape, np.float32)             # J each cell released
        t_flame = None                                       # the flame's own °C
        boost = self._blast_boost(u, Tsur, n_open) if getattr(self, "_blasts", None) else None
        chd = self._char_field()
        lick = np.zeros(self.shape, np.float64)              # flame each burning cell
        licking = np.zeros(self.shape, bool)                 # can lick its neighbours with
        # D6: a cell that holds a mix is not burnt, charred or ashed by the law of
        # the material it views as (heatmix.CRUTCHES, "_burn_cells"). None: no mix.
        ok_s, ok_f = self.legacy_ok(fluid=False), self.legacy_ok(solid=False)
        ok_flat = None if ok_s is None else ok_s.reshape(-1)
        for m, (dens, c, _k, ign, heat, rate) in SOLID.items():
            if ign is None:
                continue
            if sleeps:                                   # (see `warm_at`)
                c_ = warm_at
                c_ = c_[(self.mat.reshape(-1)[c_] == m) & (self.smass.reshape(-1)[c_] > 0)
                        & (Tsur.reshape(-1)[c_] >= ign) & (n_open.reshape(-1)[c_] > 0)]
                if ok_flat is not None:
                    c_ = c_[ok_flat[c_]]
                if not c_.size:
                    continue
                w = np.unravel_index(c_, self.shape)
            else:
                w = ((self.mat == m) & (self.smass > 0) & (Tsur >= ign) & (n_open > 0))
                if ok_s is not None:
                    w &= ok_s
                if not w.any():
                    continue
            area = n_open[w] * A * u.af[w] * BURN_AREA.get(m, 1.0)
            health = health_of(w)
            # a blast of air (bellows) lets the face burn faster than the
            # natural draught does: the extra oxygen burns its share of fuel
            rate_w = rate if boost is None else rate + boost[w] / O2_RATIO.get(m, 1.3)
            if m in PYRO:
                L_g, q_fl, lim = PYRO[m]
                f = _o2_factor(health, lim)
                want = np.minimum((Tsur[w] - ign) * Cnode[w] / L_g, rate_w * area * dt * f)
            else:
                lim, keep, ramp = GLOW[m]
                f = _o2_factor(health, lim)
                want = rate_w * area * dt * f * np.clip((Tsur[w] - ign) / ramp, 0.0, 1.0)
            want = np.minimum(want, self.smass[w])
            dm = ration(w, O2_RATIO.get(m, 1.3), want, area, health)   # no oxygen, no fire
            sens = dm * c * (Tsur[w] - AMBIENT)              # the gas leaves warm
            if m in PYRO:
                rel = dm * (heat + L_g)                      # the volatiles burn
                fb = np.minimum(q_fl * area * dt * f, (1.0 - CHI_R) * rel)
                to_skin = fb - dm * L_g - sens
                to_gas = (1.0 - CHI_R) * rel - fb + sens
                radiated += float((CHI_R * rel).sum())
                lick[w] = q_fl * A * dt * f * (dm > 0)       # per face of gap it licks
                licking[w] = True
            else:
                to_skin = keep * dm * heat - sens
                to_gas = (1.0 - keep) * dm * heat + sens
            if dm.any():                                     # the flame's own temperature,
                tm = float(flame_temp(heat, O2_RATIO.get(m, 1.3),   # from this fuel
                                      float((health * dm).sum() / dm.sum())))  # and its air
                t_flame = tm if t_flame is None else max(t_flame, tm)
            burnt[w] = (dm * heat).astype(np.float32)
            if m in (WOOD, FLESH, ROPE):                     # the char it leaves grows
                chd[w] += (dm / (dens * 1000.0 * np.maximum(area, 1e-9))   # inward as the
                           / (1.0 - CHAR_YIELD)).astype(np.float32)        # fuel is lost
            self.smass[w] -= dm.astype(np.float32)
            self.E[w] += to_skin.astype(np.float32)
            skin[w] += np.where(has[w], to_skin * (1.0 / Cnode[w] - 1.0 / C[w]),
                                0.0).astype(np.float32)
            gas_heat[w] += to_gas
            released += float((dm * heat).sum())
            self.smoke[w] += dm * SOOT.get(m, 0.7)           # only the soot SHOWS
            self.p_add[w] += dm * 0.8                        # hot gas EXPANDS — gently:
                                                             # the outward push must not
                                                             # blow away the fire's own air
            if m in (WOOD, FLESH, ROPE):                     # the VOLATILES flame off
                if sleeps:                                   # (~80% of the mass);
                    c_ = warm_at                             # what's left is charcoal
                    sm = self.smass.reshape(-1)[c_]
                    charring = c_[(self.mat.reshape(-1)[c_] == m)
                                  & (Tsur.reshape(-1)[c_] >= ign)
                                  & (sm < 0.18 * dens * self.vox_l) & (sm > 0)]
                    if ok_flat is not None:
                        charring = charring[ok_flat[charring]]
                    self.mat[np.unravel_index(charring, self.shape)] = CHAR
                    chd[np.unravel_index(charring, self.shape)] = 0.0    # (it IS char now)
                else:
                    charring = ((self.mat == m) & (Tsur >= ign)
                                & (self.smass < 0.18 * dens * self.vox_l)
                                & (self.smass > 0))
                    if ok_s is not None:
                        charring &= ok_s
                    self.mat[charring] = CHAR                # black forever, embers
                    chd[charring] = 0.0
            gone = (self.mat == m) & (self.smass < 0.05 * dens * self.vox_l)
            if ok_s is not None:
                gone &= ok_s
            self.mat[gone] = ASH                             # ...becomes gas + a little ash
            chd[gone] = 0.0
            self.smoke[gone] += self.smass[gone] * 0.5
            self.smass[gone] *= 0.5
        if licking.any():
            self._flame_reach(lick, licking, gas_heat, u, Tsur, Cnode, C)
        ign, heat, rate = FLUID[OIL][3], FLUID[OIL][4], FLUID[OIL][5]
        dens, c_oil = FLUID[OIL][0], FLUID[OIL][1]
        # the BODY of a pool cannot pass its boiling point either: a liquid
        # stirs itself, so heat that reaches its middle past that point (a
        # flame held into the pan) rises to its top, where it boils off below
        pool = (self.fl == OIL) & (self.fvol > 0) & has
        if ok_f is not None:
            pool &= ok_f
        if pool.any():
            Cs_ = u.Cs[pool]
            q = np.maximum(u.Tb[pool] - ign, 0.0) * u.Cb[pool]
            if (q > 0).any():                    # core -> skin; the cell's heat
                skin[pool] += (q / Cs_).astype(np.float32)   # is unchanged
                Tsur[pool] += q / Cs_
        o = (self.fl == OIL) & (self.fvol > 0) & (Tsur >= ign) & access
        if ok_f is not None:
            o &= ok_f
        if o.any():
            L_g, q_fl, lim = PYRO_OIL
            if sleeps:                                   # by index, as the solids
                o = np.nonzero(o)
            health = health_of(o)
            f = _o2_factor(health, lim)
            area = np.full(len(health), A)                   # a pool burns at its top
            # a pool boils off as fast as its flame heats its skin; the
            # ceiling only holds back a smear too thin to have a skin
            boil = np.minimum((Tsur[o] - ign) * Cnode[o] / L_g, self.fvol[o] * dens)
            want = np.minimum(boil, rate * area * dt * f)
            dg = ration(o, O2_RATIO_OIL, want, area, health)
            # and a liquid cannot pass its boiling point: what the heat boils
            # off beyond what burns (past the rate, or with no oxygen for it)
            # leaves UNBURNED, a hot fuel-rich vapour that rises into the gas
            # above and sets there as oily smoke. Without it, a pool held over
            # a big flame went on heating, past 2000 °C
            vap = np.maximum(boil - dg, 0.0)
            rel = dg * (heat + L_g)
            fb = np.minimum(q_fl * area * dt * f, (1.0 - CHI_R) * rel)
            to_skin = fb - (dg + vap) * L_g
            # what leaves, burned or not, takes the warmth it had as liquid at
            # the point it boils at; it is the pool's liquid that grows less,
            # so that warmth comes out of the whole cell, not its skin
            sens = (dg + vap) * c_oil * (ign - AMBIENT)
            if dg.any():
                tm = float(flame_temp(heat, O2_RATIO_OIL, float((health * dg).sum() / dg.sum())))
                t_flame = tm if t_flame is None else max(t_flame, tm)
            burnt[o] = (dg * heat).astype(np.float32)
            self.fvol[o] -= ((dg + vap) / dens).astype(np.float32)
            self.E[o] += (to_skin - sens).astype(np.float32)
            skin[o] += np.where(has[o], to_skin * (1.0 / Cnode[o] - 1.0 / C[o])
                                + sens / C[o], 0.0).astype(np.float32)
            gas_heat[o] += (1.0 - CHI_R) * rel - fb + sens + vap * L_g
            self.smoke[o] += vap
            radiated += float((CHI_R * rel).sum())
            released += float((dg * heat).sum())
            self.p_add[o] += dg * 0.8
            self.smoke[o] += dg * SOOT_OIL                   # oil burns SOOTY — it all
            out = (self.fl == OIL) & (self.fvol <= 0.01)     # leaves as smoke mass
            if ok_f is not None:
                out &= ok_f
            self.fl[out] = NOFLUID
            self.fvol[out] = 0.0
        self._flame_T = t_flame if t_flame is not None else _T_FREE
        self._blasts = []
        _FIRES[self] = (self.tick, None)
        if burnt.any():
            self._fires_and_flames(burnt, gas_heat, u)
        if gas_heat.any():
            # the flame is the gas beside the burning face: its heat goes there,
            # shared among the open cells of DRY gas (a shallow pool's cell is
            # open to the air but is not the flame); a face with none heats
            # its own cell
            dry = air & (self.fvol <= FILM * self.cap)
            src = gas_heat != 0
            # all of it happens within two cells of a burning face: done in the
            # box around them, which changes nothing (cells outside get 0)
            box = self._window(src, 2)
            dry_b, src_b, gh = dry[box], src[box], gas_heat[box]
            n_dry = _count_nb(dry_b)
            n = np.maximum(n_dry, 1).astype(np.float64)
            share = np.where(src_b, gh / n, 0.0)
            add = np.where(src_b & (n_dry == 0), gh, 0.0)
            for axis in range(3):
                lo = [slice(None)] * 3; hi = [slice(None)] * 3
                lo[axis], hi[axis] = slice(None, -1), slice(1, None)
                lo, hi = tuple(lo), tuple(hi)
                add[hi] += np.where(dry_b[hi], share[lo], 0.0)
                add[lo] += np.where(dry_b[lo], share[hi], 0.0)
            self.E[box] += add.astype(np.float32)
            self._flame_zone(dry, u)
        if radiated:
            self.shed += radiated
            self._rad_bank = getattr(self, "_rad_bank", 0.0) + radiated
        self._hrr = (self.tick, released / dt)
        if (drawn_c > 0).any():
            # collect the debt from the same catchment the health read, in
            # proportion to where the air actually IS — but never across a
            # wall: the cell blur gives the catchment its SHAPE (fading with
            # distance), while the airspace REGION bounds who can pay (a
            # voxel-thin wall is invisible at cell scale, and chamber B's air
            # must stay chamber B's). Gradients now form across a room, and
            # wind + diffusion must genuinely resupply a far corner — but a
            # plume never starves while its own catchment still holds air.
            # the debt is paid by the gas that actually PASSED THROUGH the
            # flame — the hot plume and the ceiling layer it feeds — far more
            # than by the cool inflow skimming the floor. Weighting by warmth
            # is what makes the HOT LAYER the depleted one (and the classic
            # room-fire profile: foul air descending from above, crawl-low
            # air surviving longest at the floor).
            hot_w = 1.0 + np.clip(T - AMBIENT, 0.0, 300.0) / 50.0
            wvox = self._upcell(self._blur_cells(drawn_c, reach)
                                .astype(np.float32)) * self.o2 * hot_w
            for r in np.nonzero(drawn_r > 0)[0]:
                in_r = air & (self.air_region == r)
                w_r = np.where(in_r, wvox, 0.0)
                wsum = float(w_r.sum())
                if wsum <= 1e-9:
                    continue
                take = np.minimum(w_r * (drawn_r[r] / wsum), self.o2)
                self.o2 -= take.astype(np.float32)

    def _flame_reach(self, reach, reached, gas_heat, u, Tsur, Cnode, C):
        """A flame licks THROUGH THE GAPS of a partly filled cell. A stick is a
        rod in a mostly empty cell: where one cell of it burns, its flame
        bathes the next cell of the rod across the open part of the face
        between them, at its own q_fl, which is how fire creeps along a stick
        or through a crib. A packed block has no gap: its fire spreads over its
        open faces instead, through the gas. The heat is the flame's own
        (taken from what it gives the gas), so nothing is made."""
        box = self._window(reached, 1)
        mat = self.mat[box]
        cold = (u.solid[box] & (self.smass[box] > 0) & np.isfinite(_IGN_ARR[mat])
                & (Tsur[box] < _IGN_ARR[mat]))
        self._hand_out(reached[box], gas_heat[box], reach[box], cold.astype(np.float64),
                       u, True, box)

    def _hand_out(self, src, budget, src_amt, dst_amt, u, by_gap, box):
        """Each cell of `src` hands each of its six neighbours
        src_amt[src] * dst_amt[dst] joules (times the open share of the face
        between them, `by_gap`), never more in all than its `budget`, which is
        drawn down by what it gives. What a neighbour gets lands in its skin.
        Every array is of the cells of `box`, which holds `src` and its
        neighbours: nothing outside it can be given anything."""
        s_, b_ = src, budget                             # b_ writes the budget
        sa, da = src_amt, dst_amt
        void = u.void[box]
        if self.fused and HAVE_NUMBA:
            got = np.zeros(s_.shape, np.float64)
            sa_ = (np.ascontiguousarray(sa, np.float64) if np.ndim(sa)
                   else np.full(s_.shape, float(sa)))
            bb = np.ascontiguousarray(b_)
            _give(np.ascontiguousarray(s_), bb, sa_, np.ascontiguousarray(da, np.float64),
                  np.ascontiguousarray(void), bool(by_gap), got, _DIRS)
            b_[...] = bb
            self._hand_in(got, u, box)
            return
        gives, take = [], np.zeros(s_.shape, np.float64)
        for axis in range(3):
            lo = [slice(None)] * 3; hi = [slice(None)] * 3
            lo[axis], hi[axis] = slice(None, -1), slice(1, None)
            lo, hi = tuple(lo), tuple(hi)
            for a, b in ((lo, hi), (hi, lo)):
                g = np.where(s_[a], (sa[a] if np.ndim(sa) else sa) * da[b], 0.0)
                if by_gap:
                    g = g * np.maximum(void[a], void[b])
                take[a] += g
                gives.append((a, b, g))
        scale = np.where(take > b_, b_ / np.maximum(take, 1e-30), 1.0)
        got = np.zeros(s_.shape, np.float64)
        for a, b, g in gives:
            g = g * scale[a]
            got[b] += g
            b_[a] -= g
        self._hand_in(got, u, box)

    def _hand_in(self, got, u, box):
        """What `_hand_out` gave lands in each taker's skin."""
        hit = got > 0
        if hit.any():
            E, skin = self.E[box], self._skin_field()[box]
            has, Cs, C = u.has[box], u.Cs[box], u.C[box]
            Cn = np.where(has, Cs, C)
            E[hit] += got[hit].astype(np.float32)
            skin[hit] += np.where(has[hit], got[hit] * (1.0 / Cn[hit] - 1.0 / C[hit]),
                                  0.0).astype(np.float32)

    def _flame_zone(self, dry, u):
        """A FLAME GROWS TALLER, NOT HOTTER. The gas a fire heats is the flame,
        and a flame is no hotter than its fuel and air allow (`materials.
        flame_temp`: 800-1000 °C under a metre, 1100-1200 °C for big pools,
        H28): it is diluted by the air it pulls in, which the lattice does not
        move. So gas past that temperature (`_flame_T`) hands its excess
        to the gas above it, up the column, and the flame stands as tall as
        the heat it carries needs. Out of the top of an open world, it is
        gone (`shed`).

        Under a CEILING (anything above that is not dry gas: a stone slab, a
        board, a bed's platform, a burning one too) the flame is pressed flat
        against it, and the excess goes INTO that ceiling's skin, as a flame
        under a shelf heats the shelf. Without this, gas caught between two
        burning boards had no way out at all (a burning surface takes no heat
        from its own flame's gas, `_law_conduct`) and ran up to thousands of
        degrees. A burning ceiling turns the heat into more burning, up to its
        rate; past that its skin warms and radiates it away.

        But heat only flows from hot to cold. The flame is at `_flame_T`, so it
        can warm the ceiling's surface (its skin, or the whole cell if it has
        none) up to that and no further. Without that check a ceiling that
        could not burn the heat away (a roof starved of air on its inner face,
        or one at its rate ceiling) was pushed past the flame that heats it,
        to 1600-2000 °C. What the ceiling cannot take, the flame radiates: it
        leaves the world (`shed`), as the rest of a flame's radiation does
        until radiation across the air (G7) lands. A real flame would spread
        out sideways under the ceiling, and up past its edge (flame height,
        F25); neither is modelled."""
        C = self.heat_capacity()
        cap = (self._flame_T - AMBIENT) * C
        over = dry & (self.E > cap)
        if not over.any():
            return
        skin = self._skin_field()
        layers = np.nonzero(over.any(axis=(0, 1)))[0]
        lo = int(layers[0])
        # ONLY THE COLUMNS A FLAME IS IN (`sleep_regions`). Heat here moves
        # straight up and nowhere else, so a column with no cell over the cap
        # never gets any: the box of those columns holds every move there is.
        # And past the highest such cell, a layer that hands nothing up ends
        # it, as the layer above it is as it was, and under the cap.
        bx = by = slice(None)
        hi = self.shape[2]
        if self._sleeps():
            cols = over.any(axis=2)
            xs, ys = np.flatnonzero(cols.any(axis=1)), np.flatnonzero(cols.any(axis=0))
            bx, by = slice(int(xs[0]), int(xs[-1]) + 1), slice(int(ys[0]), int(ys[-1]) + 1)
            hi = int(layers[-1])
        for z in range(lo, self.shape[2]):
            e = self.E[bx, by, z]
            extra = np.where(dry[bx, by, z], np.maximum(e - cap[bx, by, z], 0.0), 0.0)
            if not extra.any():
                if z >= hi:
                    break
                continue
            if z + 1 < self.shape[2]:
                move = extra.astype(np.float32)
                e -= move
                up = self.E[bx, by, z + 1]
                ceil = ~dry[bx, by, z + 1] & (move > 0)
                if ceil.any():                           # into the ceiling, up
                    has = u.has[bx, by, z + 1][ceil]     # to the flame's own
                    Cc = C[bx, by, z + 1][ceil]          # temperature
                    Cs = np.where(has, u.Cs[bx, by, z + 1][ceil], Cc)
                    s = skin[bx, by, z + 1]
                    Tn = AMBIENT + up[ceil] / Cc + np.where(has, s[ceil], 0.0)
                    room = np.maximum((self._flame_T - Tn) * Cs, 0.0)
                    take = np.minimum(move[ceil], room).astype(np.float32)
                    rest = move[ceil] - take
                    s[ceil] += np.where(has, take * (1.0 / Cs - 1.0 / Cc),
                                        0.0).astype(np.float32)
                    move[ceil] = take
                    self.shed += float(rest.sum(dtype=np.float64))
                up += move
            else:                                        # (summed over the whole
                top = self.E[:, :, z]                    # top, as it always was)
                was = float(top.sum(dtype=np.float64))
                e -= extra.astype(np.float32)
                self.shed += was - float(top.sum(dtype=np.float64))

    def _law_boil(self):
        if self.mixes_live and getattr(self, "mix_laws", True):
            # C2 (T4): a mixed liquid (brine, a wash) boils at the point of its own water
            # activity and keeps its solute. The pure cells below are untouched.
            _heatmix.boil_mixed(self, self.T(), self.heat_capacity(), BOIL)
        wet = (self.fl == WATER) & (self.fvol > 0)
        # D6 / T4 6.2 item 1, THE GUARD: a cell that holds a mixed liquid (brine) is
        # not boiled here. This law deletes the steam and the cell's water view; the
        # solute would go with it. (heatmix.CRUTCHES, "_law_boil")
        ok = self.legacy_ok(solid=False)
        if ok is not None:
            wet &= ok
        if not wet.any():                        # no water anywhere: nothing to boil
            return
        T, C = self.T(), self.heat_capacity()
        # pressure raises the boiling point (the pressure-cooker effect, stolen as
        # DESIGN from the sandbox lineage): a sealed hot room boils water later
        boil_at = BOIL + (1.0 * np.clip(self._upcell(self.pcell), 0.0, 18.0)
                          if self.pcell is not None else 0.0)
        w = (self.fl == WATER) & (self.fvol > 0) & (T > boil_at)
        if ok is not None:
            w &= ok
        if w.any():
            boil_g = np.minimum(self.fvol[w], (T[w] - boil_at[w] if self.pcell is not None
                                               else T[w] - BOIL) * C[w] / LATENT)
            self.fvol[w] -= boil_g
            self.E[w] -= boil_g * LATENT
            self.p_add[w] += boil_g * 0.1                # steam pushes (gently — an
                                                         # open puddle is no boiler)
            dry = (self.fl == WATER) & (self.fvol <= 0.01)
            if ok is not None:
                dry &= ok
            self.fl[dry] = NOFLUID
            self.fvol[dry] = 0.0
            self.fpot[dry] = 0.0

    def _law_melt(self):
        """PHASE CHANGE for metals, both directions, energy-honest (the same
        shape as boiling): heat above the melt point converts solid grams to
        molten milliliters at the rate the excess energy can pay the latent
        heat — so a melting bar PINS near its melt point, exactly as boiling
        water pins near 100. The melt stays held in the bar's cell until the
        solid is gone, then the flow law takes the puddle. Cooling runs the
        table backwards: a molten pool below its point sets solid where it
        lies, giving the latent heat back — a splat the shape of wherever it
        finally froze."""
        if self.mixes_live and getattr(self, "mix_laws", True):
            # C2 (T4): a mix (solder, bronze) melts over its own range, by the lever rule.
            _heatmix.melt_mixed(self, self.T(), self.heat_capacity())
        # nothing that melts or sets is here at all: nothing below can change
        # (asked of the materials, not the temperatures, so it costs a pass)
        if self.fused and HAVE_NUMBA:                    # (the same question,
            here = (_any_in(np.ascontiguousarray(self.mat).reshape(-1), _MELTABLE_ARR)
                    or _any_in(np.ascontiguousarray(self.fl).reshape(-1), _FREEZABLE_ARR))
        else:                                            # compiled: `_any_in`)
            here = _MELTABLE_ARR[self.mat].any() or _FREEZABLE_ARR[self.fl].any()
        if not here:
            return
        T, C = self.T(), self.heat_capacity()
        ok = self.legacy_ok()        # D6: a mixed cell is not melted or set here
        for m, (mt, fl_id, lat) in MELT.items():  # (heatmix.CRUTCHES, "_law_melt")
            w = ((self.mat == m) & (self.smass > 0) & (T > mt)
                 & ((self.fl == NOFLUID) | (self.fl == fl_id)))
            if ok is not None:
                w &= ok
            if w.any():
                dm = np.minimum(self.smass[w], (T[w] - mt) * C[w] / lat)
                self.smass[w] -= dm
                self.E[w] -= dm * lat
                self.fl[w] = fl_id
                self.fvol[w] += dm / FLUID[fl_id][0]
                gone = (self.mat == m) & (self.smass <= 0.01)
                if ok is not None:
                    gone &= ok
                self._unmake(gone, "melts")      # nothing left but the melt —
                                                 # it flows from here
                # AGGREGATED, not one event per cell -- a whole bar going
                # at once is one `melts`, not a thousand (3.4).
                if gone.any():
                    self._defer("melts", target=("mat", int(m)),
                              data=(("cells", int(gone.sum())),), text="")
        for fl_id, (mt, m, lat) in FREEZE.items():
            w = ((self.fl == fl_id) & (self.fvol > 0) & (T < mt - 2.0)
                 & ((self.mat == AIR) | (self.mat == m)))
            if ok is not None:
                w &= ok
            if w.any():
                g = np.minimum(self.fvol[w] * FLUID[fl_id][0],
                               (mt - 2.0 - T[w]) * C[w] / lat)
                self.fvol[w] -= g / FLUID[fl_id][0]
                self.E[w] += g * lat
                self.mat[w] = m
                self.smass[w] += g
                dry = (self.fl == fl_id) & (self.fvol <= 0.01)
                if ok is not None:
                    dry &= ok
                self.fl[dry] = NOFLUID
                self.fvol[dry] = 0.0
                self._defer("sets", target=("mat", int(m)),
                          data=(("cells", int(w.sum())),), text="")
