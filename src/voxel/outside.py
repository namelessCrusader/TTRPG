"""OUTSIDE KERNELS: array laws for the cold topic (wave 13, T8). No World here.

Every kernel has two forms. The numpy form is the DEFINITION of what the law
means; the numba form (when numba is installed) is a fused loop that must give
EQUAL BITS for arrays. Booked sums are float64 and agree to 1e-12 relative (the
sums are taken in a different order). `fused=None` means "use numba if there";
tests pass `fused=False` for the definition.

The conventions, all the same as the sim's:

    E       float32 joules ABOVE 20 C (the fixed reference stays at 20 C)
    C       float32 heat capacity of the cell, J/K
    rest_k  the rest temperature MINUS 20, K (a scalar for solids and enclosed
            air); `th_out` is T_out - 20, K, for outdoor air
    cls     uint8 per cell: bit 0 = outdoor air (sees the sky), bit 1 = touches
            a surface (the film surrogate acts at k_film instead of k_far)
    Hs      int (nx, ny): the index of the highest cell of static solid
            structure in each column (-1 where there is none)

Heat is booked with the sign "heat that LEFT the world" (shed's sign): a
kernel that returns joules returns shed's increment, so a gain is negative.

Known limit (spec 3.7): the interior of a room is ONE radiant node
(`mrt_star`), with no view factors between its surfaces.
"""
from collections import namedtuple

import numpy as np

from .kernels import HAVE_NUMBA, njit

AMBIENT = 20.0               # the fixed reference of E (materials.AMBIENT)
SIGMA = 5.670374e-8          # materials.SIGMA
FACE_CLIP = 0.15             # materials.FACE_CLIP
H_CONV = 10.0                # materials.H_CONV
SETTLE_K = 0.005             # K, the climate world's tolerance for 'at rest' (u)
PSI_OUT = 0.3                # sky view above which a gas cell is outdoor (u)
NDIR = 16                    # compass directions for the sky view
RAD_HOT = 450.0              # C, surfaces at or above this take the old radiation route
_K0 = np.float32(273.15)
_F = np.float32


# ======================================================================
# 1. the quiet gate
# ======================================================================
@njit(cache=True)
def _box_nb(E, C, cls, rest, eout, tol, exposed, use_exp):
    nx, ny, nz = E.shape
    lo = np.array([nx, ny, nz])
    hi = np.array([-1, -1, -1])
    nun = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = C[i, j, k]
                tgt = eout if (cls[i, j, k] & 1) else rest
                d = E[i, j, k] - tgt * c
                off = not (abs(d) <= tol * c)
                if off:
                    nun += 1
                if off or (use_exp and exposed[i, j, k]):
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
    return lo, hi, nun


def _window_np(there, pad):
    """The box around everything `there` is true of, grown by pad (fields._window
    without the skip_quiet switch); None if it is nowhere."""
    out = []
    for ax in range(3):
        line = there.any(axis=tuple(a for a in range(3) if a != ax))
        nz = np.flatnonzero(line)
        if not len(nz):
            return None
        out.append(slice(max(int(nz[0]) - pad, 0), min(int(nz[-1]) + 1 + pad, there.shape[ax])))
    return tuple(out)


def driven_reasons(t_rest, t_out, t_sky_eff, settle_k=SETTLE_K):
    """Why an exposed face is driven: 'interface' when the rest temperature
    differs from T_out, 'sky_drive' when the sky the faces effectively see
    differs from the air. Empty in the reference climate."""
    why = []
    if abs(t_rest - t_out) > settle_k:
        why.append("interface")
    if abs(t_sky_eff - t_out) > settle_k:
        why.append("sky_drive")
    return why


def unsettled_box(E, C, cls, rest_k, th_out, settle_k, exposed=None, driven=(), pad=0, fused=None):
    """(box, reasons): the box of cells that are not at rest, joined with the
    exposed list when the world is driven. `box` is a tuple of three slices or
    None. Quiet means: every cell is at rest AND nothing drives an interface.

        settled(cell) = |E - f32(rest)*C| <= f32(settle_k)*C     (rest = th_out
                        on outdoor cells, rest_k elsewhere)

    With settle_k = 0, rest_k = 0 and no outdoor cells this is exactly the old
    `_window(E != 0, pad)`. `driven` is a bool or the reasons list from
    `driven_reasons`; `exposed` is a bool array (surface cells with an outdoor
    face, plus film air cells), used only when driven. reasons: 'off_rest'
    (some cell is off its rest) and the driven reasons; empty list = quiet."""
    use_exp = bool(driven) and exposed is not None
    reasons = []
    if fused is None:
        fused = HAVE_NUMBA
    rest, eo, tol = _F(rest_k), _F(th_out), _F(settle_k)
    if fused:
        ex = exposed if use_exp else np.zeros((1, 1, 1), bool)
        lo, hi, nun = _box_nb(np.ascontiguousarray(E), np.ascontiguousarray(C),
                              np.ascontiguousarray(cls), rest, eo, tol, ex, use_exp)
        if nun:
            reasons.append("off_rest")
        if use_exp:
            reasons += list(driven) if not isinstance(driven, bool) else ["driven"]
        if hi[0] < 0:
            return None, reasons
        box = tuple(slice(max(int(lo[a]) - pad, 0), min(int(hi[a]) + 1 + pad, E.shape[a])) for a in range(3))
        return box, reasons
    tgt = np.where((cls & 1) != 0, eo, rest).astype(np.float32)
    off = ~(np.abs(E - tgt * C) <= tol * C)
    if off.any():
        reasons.append("off_rest")
    there = off | exposed if use_exp else off
    if use_exp:
        reasons += list(driven) if not isinstance(driven, bool) else ["driven"]
    return _window_np(there, pad), reasons


# ======================================================================
# 2. outdoor air: the shift, and the two relaxation rates
# ======================================================================
def relax_factor(k_per_s, dt_s):
    """1 - exp(-k dt): the share of a deviation an exact exponential removes in
    one tick. Always in [0, 1): it can never overshoot, at any k."""
    return float(-np.expm1(-float(k_per_s) * float(dt_s)))


@njit(cache=True)
def _shift_nb(E, C, cls, th_old, th_new):
    nx, ny, nz = E.shape
    tot = 0.0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if cls[i, j, k] & 1:
                    c = C[i, j, k]
                    eo = th_old * c
                    en = th_new * c
                    e = E[i, j, k]
                    n = en if e == eo else e + (en - eo)
                    tot += np.float64(n) - np.float64(e)
                    E[i, j, k] = n
    return tot


def outdoor_shift(E, C, cls, e_old, e_new, fused=None):
    """Move every outdoor cell with T_out: e_old and e_new are the OLD and NEW
    T_out - 20 (K, scalars). A settled cell (E == f32(e_old)*C) lands EXACTLY
    on f32(e_new)*C, bit for bit; a cell holding a plume keeps its deviation
    (E += the difference). Works in place on E and returns the joules the
    world GAINED (float64): the caller books shed -= it and
    outside_booked['shift'] -= it."""
    if fused is None:
        fused = HAVE_NUMBA
    if _F(e_old) == _F(e_new):
        return 0.0
    if fused:
        return float(_shift_nb(E, C, cls, _F(e_old), _F(e_new)))
    m = (cls & 1) != 0
    c = C[m]
    eo, en = _F(e_old) * c, _F(e_new) * c
    e = E[m]
    n = np.where(e == eo, en, e + (en - eo)).astype(np.float32)
    E[m] = n
    return float(np.sum(n.astype(np.float64) - e.astype(np.float64)))


@njit(cache=True)
def _relax_nb(E, C, cls, th, ff, fa, snap, hot, smoke, still, use_smoke):
    nx, ny, nz = E.shape
    bf = 0.0
    bfar = 0.0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cl = cls[i, j, k]
                if cl & 1:
                    c = C[i, j, k]
                    et = th * c
                    e = E[i, j, k]
                    if e != et and e - et <= hot * c and not (use_smoke and smoke[i, j, k] > still):
                        d = e - et
                        n = e - d * (ff if (cl & 2) else fa)
                        if d > 0 and n < et:
                            n = et
                        elif d < 0 and n > et:
                            n = et
                        if abs(n - et) <= snap * c:
                            n = et
                        b = np.float64(e) - np.float64(n)
                        if cl & 2:
                            bf += b
                        else:
                            bfar += b
                        E[i, j, k] = n
    return bf, bfar


def outdoor_relax(E, C, cls, th_out, f_film, f_far, snap, fused=None, hot_k=float('inf'),
                  smoke=None, smoke_still=0.0):
    """Relax outdoor air toward T_out. th_out = T_out - 20 (K, scalar); f_film
    and f_far = relax_factor(k, dt) for cells that touch a surface and for free
    air (two rates, 1/s and 0.1/s: a 10x difference); `snap` = SETTLE_K, a cell
    within snap*C of its target lands on it. Exact exponential, clamped so it
    cannot cross the target. In place on E; returns (film, far): the joules
    that LEFT the world through each, float64 (a gain is negative).

    `hot_k`: a cell more than this many kelvin ABOVE T_out is a plume or a
    flame, not far-field air. It is left to the buoyancy law (which lifts it out
    of the top, booked). A plume's heat flux is conserved as it widens: relaxing
    it at 0.1/s would delete the heat a fire gives (measured: the forge scene
    lost 640 kJ in 7 s, 10 percent of its flame temperature). The relaxation is
    the return of MILD warm or cold air to the outside, so the world does not
    keep a heated bubble forever. `smoke` (grams a voxel) with `smoke_still`:
    a cell holding more smoke than that is part of a plume and is left alone
    too (spec 3.6.2)."""
    if fused is None:
        fused = HAVE_NUMBA
    th, ff, fa, sn = _F(th_out), _F(f_film), _F(f_far), _F(snap)
    hk = _F(hot_k) if np.isfinite(hot_k) else _F(3.0e38)
    use_smoke = smoke is not None
    sm = np.ascontiguousarray(smoke) if use_smoke else np.zeros((1, 1, 1), np.float32)
    if fused:
        a, b = _relax_nb(E, C, cls, th, ff, fa, sn, hk, sm, _F(smoke_still), use_smoke)
        return float(a), float(b)
    m = (cls & 1) != 0
    c = C[m]
    et = th * c
    e = E[m]
    cl = cls[m]
    live = np.ones(e.shape, bool) if not np.isfinite(hot_k) else (e - et) <= hk * c
    if use_smoke:
        live = live & ~(smoke[m] > _F(smoke_still))
    d = e - et
    f = np.where((cl & 2) != 0, ff, fa).astype(np.float32)
    n = (e - d * f).astype(np.float32)
    n = np.where((d > 0) & (n < et), et, n)
    n = np.where((d < 0) & (n > et), et, n)
    n = np.where(np.abs(n - et) <= sn * c, et, n)
    n = np.where((e != et) & live, n, e).astype(np.float32)
    b = e.astype(np.float64) - n.astype(np.float64)
    E[m] = n
    film = (cl & 2) != 0
    return float(b[film].sum()), float(b[~film].sum())


# ======================================================================
# 3. sky view and the outdoor class
# ======================================================================
def view_radius_cells(Hs, dx_m):
    """R = min(max(2 m, 6 x tallest), 12 m), in cells (u)."""
    tallest = (int(Hs.max()) + 1) if Hs.size else 0
    r = min(max(2.0 / dx_m, 6.0 * tallest), 12.0 / dx_m)
    return int(max(1, r))


def ray_steps(R):
    """Distances (cells) sampled along a ray: every cell out to 16, then +6%."""
    out, d = [], 1.0
    while d <= R:
        out.append(float(int(d)) if d < 16.0 else float(round(d, 3)))
        d = d + 1.0 if d < 16.0 else d * 1.06
    return np.asarray(out, np.float64)


def _dirs(ndir, phase):
    th = phase + (2.0 * np.pi / ndir) * np.arange(ndir)
    return np.cos(th), np.sin(th)


def _psi_plain(Hs, cells, steps, cx, cy):
    nx, ny = Hs.shape
    x, y, z = (cells[:, 0].astype(np.int64), cells[:, 1].astype(np.int64),
               cells[:, 2].astype(np.int64))
    under = Hs[x, y] >= z
    acc = np.zeros(len(x), np.float64)
    zc = z.astype(np.float64) + 0.5
    for k in range(len(cx)):
        t = np.zeros(len(x), np.float64)
        for d in steps:
            px = np.floor(x + d * cx[k] + 0.5).astype(np.int64)
            py = np.floor(y + d * cy[k] + 0.5).astype(np.int64)
            ok = (px >= 0) & (px < nx) & (py >= 0) & (py < ny)
            h = Hs[np.clip(px, 0, nx - 1), np.clip(py, 0, ny - 1)].astype(np.float64)
            s = np.where(ok, (h + 1.0 - zc) / d, 0.0)
            t = np.maximum(t, s)
        acc += 1.0 / (1.0 + t * t)
    psi = acc / len(cx)
    psi[under] = 0.0
    return psi


@njit(cache=True)
def _psi_one(Hs, x, y, z, steps, cx, cy):
    nx, ny = Hs.shape
    if Hs[x, y] >= z:
        return 0.0
    zc = z + 0.5
    acc = 0.0
    for k in range(cx.shape[0]):
        t = 0.0
        for q in range(steps.shape[0]):
            d = steps[q]
            px = int(np.floor(x + d * cx[k] + 0.5))
            py = int(np.floor(y + d * cy[k] + 0.5))
            if px >= 0 and px < nx and py >= 0 and py < ny:
                s = (np.float64(Hs[px, py]) + 1.0 - zc) / d
                if s > t:
                    t = s
        acc += 1.0 / (1.0 + t * t)
    return acc / cx.shape[0]


@njit(cache=True)
def _psi_nb(Hs, cells, steps, cx, cy):
    out = np.empty(cells.shape[0], np.float64)
    for i in range(cells.shape[0]):
        out[i] = _psi_one(Hs, cells[i, 0], cells[i, 1], cells[i, 2], steps, cx, cy)
    return out


def sky_view(Hs, cells, R, ndir=NDIR, phase=0.0, fused=None):
    """Sky view factor psi of air cells (N, 3 int), Johnson and Watson 1984:
    the mean over `ndir` compass directions of cos^2(beta), beta the elevation
    of the highest obstruction along the ray within R cells. cos^2(atan s) is
    written 1/(1 + s^2), pure arithmetic, so the numba and numpy forms agree to
    the bit. A flat open plane is 1.0; under a roof (a solid at or above the
    cell in its column) 0. An infinite canyon of width W and wall height H is
    1/sqrt(1 + (2H/W)^2) (0.243 for W 2 m, H 4 m; 16 directions give 0.233,
    tested within 0.03). Rays leaving the lattice see nothing.

    Known limit: 16 compass samples. A canyon with H/W above about 2 depends on the
    compass phase: W 1 m, H 4 m gives 0.166, 0.123 and 0.100 at three phases
    against 0.124 analytic. Use `ndir` 32 or more there; the two canyons in the
    tests (H/W 2 and 0.5) hold the 0.03 tolerance."""
    cells = np.ascontiguousarray(np.atleast_2d(cells))
    if len(cells) == 0:
        return np.zeros(0, np.float64)
    if fused is None:
        fused = HAVE_NUMBA
    steps = ray_steps(R)
    cx, cy = _dirs(ndir, phase)
    if fused:
        return _psi_nb(np.ascontiguousarray(Hs), cells.astype(np.int64), steps, cx, cy)
    return _psi_plain(Hs, cells, steps, cx, cy)


def _runmax(a, r, axis):
    """Sliding max of window 2r+1 along an axis, edges padded with -1 (nothing)."""
    if r <= 0:
        return a.copy()
    n = a.shape[axis]
    w = 2 * r + 1
    pad_shape = list(a.shape)
    pad_shape[axis] = r
    padv = np.full(pad_shape, -1, a.dtype)
    p = np.concatenate([padv, a, padv], axis=axis)
    m, size = p, 1
    while size * 2 <= w:                 # m[i] = max(p[i : i + size])
        sl_a = [slice(None)] * p.ndim
        sl_b = [slice(None)] * p.ndim
        sl_a[axis] = slice(0, m.shape[axis] - size)
        sl_b[axis] = slice(size, m.shape[axis])
        m = np.maximum(m[tuple(sl_a)], m[tuple(sl_b)])
        size *= 2
    sl_a = [slice(None)] * p.ndim
    sl_b = [slice(None)] * p.ndim
    sl_a[axis] = slice(0, n)
    sl_b[axis] = slice(w - size, w - size + n)
    return np.maximum(m[tuple(sl_a)], m[tuple(sl_b)])


def column_max(Hs, r):
    """Max of Hs over the square window of radius r cells around each column."""
    return _runmax(_runmax(Hs, r, 0), r, 1)


@njit(cache=True)
def _zstar_nb(Hs, hfar, steps, cx, cy, pmin, nz):
    nx, ny = Hs.shape
    out = np.empty((nx, ny), np.int64)
    for i in range(nx):
        for j in range(ny):
            lo = Hs[i, j] + 1
            hi = max(hfar[i, j], Hs[i, j]) + 1
            while lo < hi:
                mid = (lo + hi) // 2
                if _psi_one(Hs, i, j, mid, steps, cx, cy) >= pmin:
                    hi = mid
                else:
                    lo = mid + 1
            out[i, j] = lo
    return out


def _zstar_plain(Hs, hfar, steps, cx, cy, pmin):
    nx, ny = Hs.shape
    lo = (Hs + 1).astype(np.int64).ravel()
    hi = (np.maximum(hfar, Hs) + 1).astype(np.int64).ravel()
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    while True:
        act = lo < hi
        if not act.any():
            break
        idx = np.flatnonzero(act)
        mid = (lo[idx] + hi[idx]) // 2
        cells = np.stack([ii[idx], jj[idx], mid], axis=1)
        ok = _psi_plain(Hs, cells, steps, cx, cy) >= pmin
        hi[idx] = np.where(ok, mid, hi[idx])
        lo[idx] = np.where(ok, lo[idx], mid + 1)
    return lo.reshape(nx, ny)


def outdoor_class(Hs, air_mask, film_mask, R, ndir=NDIR, psi_min=PSI_OUT, open_sky=True,
                  phase=0.0, fused=None):
    """cls bits per cell (uint8): bit 0 = OUTDOOR air, bit 1 = touches a surface.

    Outdoor means SKY VIEW, not connectivity: a gas cell is outdoor only if its
    sky view is at least psi_min. A room under a roof is never outdoor, however
    open its door; a 1-cell hole in a roof does not make the room outdoor (the
    air in the hole itself may be). Sky view never falls as a cell goes up a
    column, so the lowest outdoor height of each column, z*, is found by a
    binary search, and every gas cell at or above it (and above the highest
    solid) is outdoor. Where nothing within R is taller than the ground, z* is
    just above it with no search at all. Spec 3.6.1 used a 20 cm column rule
    for free air and psi only on the film list; this is the exact form of the
    same idea (a courtyard floor 4 m across between 4 m walls is a canyon of
    0.24, not outdoor).

    Hs: (nx, ny) int; air_mask: bool (nx, ny, nz) of gas cells (mat AIR,
    little liquid, smoke at or below SMOKE_STILL); film_mask: bool of air cells
    that touch a surface face. open_sky False gives nothing outdoor."""
    nx, ny, nz = air_mask.shape
    cls = np.zeros(air_mask.shape, np.uint8)
    if not open_sky:
        return cls
    if fused is None:
        fused = HAVE_NUMBA
    steps = ray_steps(R)
    cx, cy = _dirs(ndir, phase)
    hfar = column_max(Hs, R)
    Hs64 = np.ascontiguousarray(Hs.astype(np.int64))
    if fused:
        zs = _zstar_nb(Hs64, hfar.astype(np.int64), steps, cx, cy, float(psi_min), nz)
    else:
        zs = _zstar_plain(Hs64, hfar.astype(np.int64), steps, cx, cy, float(psi_min))
    z = np.arange(nz)[None, None, :]
    out = air_mask & (z >= zs[:, :, None]) & (z > Hs[:, :, None])
    cls[out] = 1
    cls[out & film_mask] = 3
    return cls


# ======================================================================
# 4. faces: which environment does each surface look at
# ======================================================================
FaceEnv = namedtuple("FaceEnv", "n_out w_sky n_in room n_edge")

_DIRS6 = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


@njit(cache=True)
def _face_nb(surf, gas, outdoor, psi, room, open_sky, ground_k):
    nx, ny, nz = surf.shape
    n_out = np.zeros((nx, ny, nz), np.uint8)
    n_in = np.zeros((nx, ny, nz), np.uint8)
    n_edge = np.zeros((nx, ny, nz), np.uint8)
    w_sky = np.zeros((nx, ny, nz), np.float32)
    rid = np.zeros((nx, ny, nz), np.int32)
    dirs = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    cand = np.zeros(6, np.int32)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not surf[i, j, k]:
                    continue
                ncand = 0
                for q in range(6):
                    a, b, c = i + dirs[q, 0], j + dirs[q, 1], k + dirs[q, 2]
                    if a < 0 or a >= nx or b < 0 or b >= ny or c < 0 or c >= nz:
                        if open_sky and q != 5 and k >= ground_k:
                            n_out[i, j, k] += 1
                            n_edge[i, j, k] += 1
                            w_sky[i, j, k] += np.float32(1.0) if q == 4 else np.float32(0.5)
                        continue
                    if not gas[a, b, c]:
                        continue
                    if outdoor[a, b, c]:
                        n_out[i, j, k] += 1
                        if q == 4:
                            w_sky[i, j, k] += psi[a, b, c]
                        elif q != 5:
                            w_sky[i, j, k] += np.float32(0.5) * psi[a, b, c]
                    else:
                        n_in[i, j, k] += 1
                        cand[ncand] = room[a, b, c]
                        ncand += 1
                if ncand > 0:
                    best, bestn = cand[0], 0
                    for p in range(ncand):
                        cnt = 0
                        for r in range(ncand):
                            if cand[r] == cand[p]:
                                cnt += 1
                        if cnt > bestn:
                            best, bestn = cand[p], cnt
                    rid[i, j, k] = best
    return n_out, w_sky, n_in, rid, n_edge


def _shift_nb_free(a, ax, sgn, fill):
    """b[cell] = a[cell + sgn on axis], `fill` outside the lattice."""
    out = np.full(a.shape, fill, a.dtype)
    n = a.shape[ax]
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    if sgn > 0:
        src[ax], dst[ax] = slice(1, n), slice(0, n - 1)
    else:
        src[ax], dst[ax] = slice(0, n - 1), slice(1, n)
    out[tuple(dst)] = a[tuple(src)]
    return out


def face_env(surf, gas, cls, psi, room, open_sky=True, ground_k=0, fused=None):
    """Per surface cell: how many of its faces look at the OUTDOORS and how many
    at a ROOM, the sky weight, the room id, and the edge-face count.

        n_out   faces onto an outdoor air cell, or on a lattice edge (x, y or
                top) with open_sky True and at or above cell height ground_k
                (edge faces below the ground are buried soil: nothing)
        w_sky   the sky share summed over the outdoor faces: psi of the
                neighbour for an up face, 0.5 psi for a side face, 0 for a down
                face; an edge face is 1.0 (top) or 0.5 (side)
        n_in    faces onto any other gas cell (it has a roof over it)
        room    the room id (air region) of the neighbours with the most faces
        n_edge  the edge faces among n_out

    surf, gas: bool arrays; cls: the outdoor class (bit 0 = outdoor); psi:
    float32 sky view of each outdoor air cell (0 elsewhere); room: int32 air
    region ids. Returns FaceEnv(n_out, w_sky, n_in, room, n_edge); cells that
    are not surfaces are all zero."""
    if fused is None:
        fused = HAVE_NUMBA
    outdoor = (cls & 1) != 0
    psi32 = np.ascontiguousarray(psi, np.float32)
    room32 = np.ascontiguousarray(room, np.int32)
    if fused:
        return FaceEnv(*_face_nb(np.ascontiguousarray(surf), np.ascontiguousarray(gas),
                                 np.ascontiguousarray(outdoor), psi32, room32,
                                 bool(open_sky), int(ground_k)))
    shp = surf.shape
    n_out = np.zeros(shp, np.uint8)
    n_in = np.zeros(shp, np.uint8)
    n_edge = np.zeros(shp, np.uint8)
    w_sky = np.zeros(shp, np.float32)
    cands, valids = [], []
    zidx = np.arange(shp[2])[None, None, :]
    for q, (dx, dy, dz) in enumerate(_DIRS6):
        ax = (0, 0, 1, 1, 2, 2)[q]
        sgn = 1 if q % 2 == 0 else -1
        g = _shift_nb_free(gas, ax, sgn, False)
        inside = _shift_nb_free(np.ones(shp, bool), ax, sgn, False)
        o = _shift_nb_free(outdoor, ax, sgn, False)
        p = _shift_nb_free(psi32, ax, sgn, np.float32(0))
        r = _shift_nb_free(room32, ax, sgn, np.int32(0))
        edge = ~inside & bool(open_sky) & (q != 5) & (zidx >= ground_k) & surf
        hit_out = inside & g & o & surf
        hit_in = inside & g & ~o & surf
        n_out += hit_out.astype(np.uint8) + edge.astype(np.uint8)
        n_edge += edge.astype(np.uint8)
        n_in += hit_in.astype(np.uint8)
        if q == 4:
            w_sky += np.where(hit_out, p, np.float32(0))
        elif q != 5:
            w_sky += np.where(hit_out, np.float32(0.5) * p, np.float32(0))
        w_sky += np.where(edge, np.float32(1.0) if q == 4 else np.float32(0.5), np.float32(0))
        cands.append(r)
        valids.append(hit_in)
    rid = np.zeros(shp, np.int32)
    best = np.zeros(shp, np.int32)
    for p_ in range(6):
        cnt = np.zeros(shp, np.int32)
        for r_ in range(6):
            cnt += (valids[p_] & valids[r_] & (cands[p_] == cands[r_])).astype(np.int32)
        take = valids[p_] & (cnt > best)
        rid = np.where(take, cands[p_], rid)
        best = np.where(take, cnt, best)
    return FaceEnv(n_out, w_sky, n_in, rid, n_edge)


def env_temperature(n_out, w_sky, t_sky_c, t_out_c):
    """The effective environment temperature (C, float32) the outdoor faces of
    each cell radiate to. Each outdoor face sees the sky over a share psi_f and
    the outdoor surroundings (at T_out) over the rest, so
    T_env^4 = (w_sky T_sky^4 + (n_out - w_sky) T_out^4)/n_out in kelvin.
    Exact ends: no sky weight gives T_out, all sky gives T_sky, and a sky equal
    to the air gives T_out, with no power round trip."""
    n = np.asarray(n_out, np.float64)
    w = np.asarray(w_sky, np.float64)
    if float(t_sky_c) == float(t_out_c):
        return np.full(n.shape, t_out_c, np.float32)
    ks, ko = float(t_sky_c) + 273.15, float(t_out_c) + 273.15
    frac = np.where(n > 0, w / np.maximum(n, 1.0), 0.0)
    t4 = frac * ks ** 4 + (1.0 - frac) * ko ** 4
    t = np.sqrt(np.sqrt(t4)) - 273.15
    t = np.where(frac <= 0.0, t_out_c, np.where(frac >= 1.0, t_sky_c, t))
    return t.astype(np.float32)


def env_t4(n_out, w_sky, t_sky_c, t_out_c):
    """The same environment as T^4 (K^4, float32), for the body law's 1/s
    mean-radiant correction (T7's w.env_t4(cell))."""
    t = env_temperature(n_out, w_sky, t_sky_c, t_out_c).astype(np.float64) + 273.15
    return (t ** 4).astype(np.float32)


# ======================================================================
# 5. radiation, film, ground
# ======================================================================
@njit(cache=True)
def _rad_scalar(ts, eps, af, n, te, A, dt, sw, csur, cl, sig, k0):
    tk = ts + k0
    tv = te + k0
    fourth = (tk * tk + tv * tv) * (tk + tv) * (ts - te)
    loss = eps * sig * A * af * n * dt * fourth
    lim = cl * csur * abs(ts - te)
    if loss > lim:
        loss = lim
    if loss < -lim:
        loss = -lim
    return loss - sw * A * af * dt


@njit(cache=True)
def _rad_nb(Ts, eps, af, n, Te, A, dt, sw, Csur, cl, sig, k0):
    out = np.zeros(Ts.shape, np.float32)
    for i in range(Ts.shape[0]):
        if n[i] > 0:
            out[i] = _rad_scalar(Ts[i], eps[i], af[i], np.float32(n[i]), Te[i], A, dt, sw[i],
                                 Csur[i], cl, sig, k0)
    return out


def radiation_out(Ts, eps, af, n_out, Tenv, A, dt, sw, Csur, clip=FACE_CLIP, fused=None):
    """Heat each outdoor-facing surface cell LOSES to its environment in one
    tick, float32 joules (positive leaves; a cold surface under a warm
    environment gains). The old 'radiate to 20 C' block with the environment
    made a per-cell variable:

        loss = e sigma A af n_out dt (tk^2 + te^2)(tk + te)(Ts - Tenv)

    in the factored form, EXACTLY 0 when Ts == Tenv, the radiative part clipped
    by clip*Csur*|Ts - Tenv| as the old term is. `sw` is the absorbed shortwave
    per area (W/m2, zero in T8): it is subtracted AFTER the clip, so it reduces
    the loss one for one. Ts, Tenv in C. Arrays of one shape; cells with n_out
    = 0 give 0."""
    if fused is None:
        fused = HAVE_NUMBA
    f = np.float32
    shp = Ts.shape
    args = [np.ascontiguousarray(a, np.float32).ravel() for a in (Ts, eps, af)]
    n = np.ascontiguousarray(n_out).ravel()
    te = np.ascontiguousarray(Tenv, np.float32).ravel()
    swv = np.ascontiguousarray(np.broadcast_to(np.asarray(sw, np.float32), shp)).ravel()
    cs = np.ascontiguousarray(Csur, np.float32).ravel()
    if fused:
        out = _rad_nb(args[0], args[1], args[2], n, te, f(A), f(dt), swv, cs, f(clip), f(SIGMA), _K0)
        return out.reshape(shp)
    ts, e, a = args
    nf = n.astype(np.float32)
    tk = ts + _K0
    tv = te + _K0
    fourth = (tk * tk + tv * tv) * (tk + tv) * (ts - te)
    loss = e * f(SIGMA) * f(A) * a * nf * f(dt) * fourth
    lim = f(clip) * cs * np.abs(ts - te)
    loss = np.minimum(np.maximum(loss, -lim), lim)
    net = loss - swv * f(A) * a * f(dt)
    return np.where(n > 0, net, f(0)).astype(np.float32).reshape(shp)


def edge_film(Ts, n_edge, t_out_c, af, A, dt, Csur, h_conv=H_CONV, clip=FACE_CLIP):
    """Convective film on lattice-edge faces above the ground, to the outside
    air: q = h A af n_edge dt (Ts - T_out), clipped like every face. Float32
    joules that LEAVE the cell. (Booked to shed and outside_booked['edge'].)"""
    f = np.float32
    d = (np.asarray(Ts, np.float32) - f(t_out_c))
    q = f(h_conv) * f(A) * np.asarray(af, np.float32) * np.asarray(n_edge, np.float32) * f(dt) * d
    lim = f(clip) * np.asarray(Csur, np.float32) * np.abs(d)
    return np.where(np.asarray(n_edge) > 0, np.minimum(np.maximum(q, -lim), lim), f(0)).astype(np.float32)


def ground_flux(Ts, t_ground_c, k_cell, A, dt, dist_m, Csur, clip=FACE_CLIP):
    """Heat the bottom layer of solid cells gives the deep ground in one tick:
    q = k A/(ground_depth + dx/2) (T - T_g) dt (positive leaves the cell),
    clipped by clip*Csur*|T - T_g|. `dist_m` = ground_depth_m + dx/2. Booked to
    shed and outside_booked['ground']."""
    f = np.float32
    d = np.asarray(Ts, np.float32) - f(t_ground_c)
    q = np.asarray(k_cell, np.float32) * f(A) / f(dist_m) * f(dt) * d
    lim = f(clip) * np.asarray(Csur, np.float32) * np.abs(d)
    return np.minimum(np.maximum(q, -lim), lim).astype(np.float32)


# ======================================================================
# 6. the indoor radiant star
# ======================================================================
MRT = namedtuple("MRT", "q resid tmr_cell tmr_room")


@njit(cache=True)
def _mrt_q_nb(tk, tm, w, sig_dt, lim):
    out = np.zeros(tk.shape, np.float64)
    for i in range(tk.shape[0]):
        if w[i] > 0.0:
            q = sig_dt * w[i] * (tk[i] * tk[i] + tm[i] * tm[i]) * (tk[i] + tm[i]) * (tk[i] - tm[i])
            m = lim[i]
            if q > m:
                q = m
            if q < -m:
                q = -m
            out[i] = q
    return out


def mrt_star(Ts, eps, af, n_in, room, A, dt, Csur, rad_hot=RAD_HOT, clip=FACE_CLIP,
             dtype=np.float32, default_c=AMBIENT, fused=None):
    """The indoor radiation of every room as a STAR through a radiant node.

    For each room id, T_mr^4 = sum(w T^4)/sum(w) with w = A eps af n_in over the
    surface cells of that room at Ts < rad_hot, and each cell gives
    q_i = sigma dt w_i (T_i^4 - T_mr^4) (factored form, clipped by
    clip*Csur*|Ts - T_mr|). Before the clip the room sums to ZERO by the
    definition of T_mr: nothing leaves, a hot hearth wall warms the cold walls,
    a cold window cools what is near it. Rooms do not exchange. A uniform room
    gives exactly 0. T_mr starts from the exact mean of T^4 (float64, any
    spread, cold member first or last) and is polished by Newton on the
    factored form about the room's first member, so equal temperatures give
    equal bits and a nearly uniform room keeps full precision.

    Returns MRT(q, resid, tmr_cell, tmr_room): q = joules that LEAVE each cell
    (dtype), resid = sum(q) in float64 (the clip's residue and rounding, booked
    in outside_booked['mrt_resid'] and shed), tmr_cell = T_mr in C seen by every
    cell with a room (default_c where the room has no pool) so surfaces at or
    above rad_hot, which take the old route, can radiate to it, tmr_room =
    T_mr in K per room id. Pass dtype=np.float64 to see the 1e-12 zero-sum."""
    if fused is None:
        fused = HAVE_NUMBA
    shp = Ts.shape
    ts = np.asarray(Ts, np.float64).ravel()
    ep = np.asarray(eps, np.float64).ravel()
    a = np.asarray(af, np.float64).ravel()
    nin = np.asarray(n_in, np.float64).ravel()
    rid = np.asarray(room).ravel().astype(np.int64)
    cs = np.asarray(Csur, np.float64).ravel()
    pool = (nin > 0) & (rid > 0) & (ts < rad_hot)
    w = np.where(pool, float(A) * ep * a * nin, 0.0)
    tk = ts + 273.15
    r = np.where(pool, rid, 0)
    nroom = int(r.max()) + 1 if r.size else 1
    sw = np.bincount(r, weights=w, minlength=nroom)
    idx = np.flatnonzero(pool)
    tref = np.zeros(nroom)
    if idx.size:
        uniq, first = np.unique(r[idx], return_index=True)
        tref[uniq] = tk[idx][first]
    tr = tref[r]
    d4 = (tk * tk + tr * tr) * (tk + tr) * (tk - tr)
    safe = np.where(sw > 0, sw, 1.0)
    delta4 = np.bincount(r, weights=w * d4, minlength=nroom) / safe
    # start from the exact mean of T^4 (any spread), then polish on the factored
    # form, which keeps full precision when the room is nearly uniform
    T = np.maximum(tref ** 4 + delta4, 1.0) ** 0.25
    for _ in range(3):
        f = (T * T + tref * tref) * (T + tref) * (T - tref) - delta4
        T = T - f / (4.0 * np.maximum(T, 1.0) ** 3)
    T = np.where(delta4 == 0.0, tref, T)
    tm = T[r]
    lim = clip * cs * np.abs(ts - (tm - 273.15))
    sig_dt = SIGMA * float(dt)
    if fused:
        q64 = _mrt_q_nb(tk, tm, w, sig_dt, lim)
    else:
        q64 = sig_dt * w * (tk * tk + tm * tm) * (tk + tm) * (tk - tm)
        q64 = np.where(w > 0.0, np.minimum(np.maximum(q64, -lim), lim), 0.0)
    q = q64.astype(dtype)
    has = np.bincount(r, minlength=nroom) > 0
    has[0] = False
    tmr_room = np.where(has & (sw > 0), T, 0.0)
    rr = np.where(rid > 0, np.minimum(rid, nroom - 1), 0)
    tmr_c = np.where((rid > 0) & (rid < nroom) & (tmr_room[rr] > 0), tmr_room[rr] - 273.15, default_c)
    return MRT(q.reshape(shp), float(q.sum(dtype=np.float64)), tmr_c.astype(np.float32).reshape(shp),
               tmr_room)


# ======================================================================
# 7. soak, T fill, and the matter rule
# ======================================================================
def soak_fill(C, cls, rest_k, th_out):
    """E for a world soaked at rest: f32(th_out)*C on outdoor cells,
    f32(rest_k)*C elsewhere. Scenes call it LAST, after fill() and fixtures
    (never on the cells of a living person)."""
    return (np.where((cls & 1) != 0, _F(th_out), _F(rest_k)) * C).astype(np.float32)


def t_fill(cls, t_rest_c, t_out_c):
    """The temperature (C) of cells outside the unsettled box: T_rest on
    enclosed cells, T_out on outdoor cells, built from cls and two scalars."""
    return np.where((cls & 1) != 0, np.float32(t_out_c), np.float32(t_rest_c)).astype(np.float32)


def rest_shift(c_old, c_new, t_cell_rest_c):
    """THE MATTER RULE: joules to ADD to a cell's E when its heat capacity
    changes in place at fixed temperature, so the REST part of the energy
    travels with the capacity: (T_rest - 20)(C_new - C_old). Zero when the rest
    is 20 C or the capacity is unchanged. The difference is real (the reference
    offset of different matter): book it as outside_booked['ref_shift'] with
    shed's sign, which is its negative."""
    return (np.asarray(t_cell_rest_c, np.float64) - AMBIENT) * (
        np.asarray(c_new, np.float64) - np.asarray(c_old, np.float64))


def matter_change(E, c_old, c_new, t_rest_c):
    """A cell's capacity goes from c_old to c_new (matter converted or leaving).
    Returns (E_new, ref_booked, carried), all float64, with
    E_new + ref_booked + carried == E exactly:

      * the rest part moves with the capacity: E + rest_shift;
      * the deviation dev = E - rest_k c_old. If dev < 0 (cold matter) its share
        dev (c_old - c_new)/c_old LEAVES WITH THE MATTER (`carried`, to the
        receiving cell, the rubble or w.gone, booked there); a cold cell is
        never left holding its cold on far lighter air (the -43,000 C bug);
      * if dev >= 0 the old behaviour is kept: the heat stays.

    `ref_booked` is in shed's sign (heat that left: minus the rest_shift)."""
    E = np.asarray(E, np.float64)
    c_old = np.asarray(c_old, np.float64)
    c_new = np.asarray(c_new, np.float64)
    rk = np.asarray(t_rest_c, np.float64) - AMBIENT
    rs = rk * (c_new - c_old)
    dev = E - rk * c_old
    lost = np.maximum(c_old - c_new, 0.0)
    share = np.where(dev < 0.0, dev * lost / np.where(c_old > 0, c_old, 1.0), 0.0)
    E_new = E + rs - share
    return E_new, -rs, share
