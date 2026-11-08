"""GAS FLOW: hot air rises because it is lighter, and everything it carries
goes with it (docs/realism-map.md, section 6, lane 7).

The old gas laws move heat, smoke and oxygen with local rules: a hot voxel
hands some of its heat to the voxel above (`_law_rise`), smoke "rises hard"
and races along the ceiling by a constant (`_law_smoke`), and a stir pulls
each voxel toward the mean of its zone (`_law_o2`). Each one is a stand-in
for the same thing, a flow, and each has its own number. This file is the
flow itself, so those stand-ins can go.

IT IS BEHIND A SWITCH (`World.gas_flow`, read in air.py), OFF BY DEFAULT.

One tick of it:

  1. BUOYANCY. Every 4-voxel cell (20 cm at 5 cm voxels) has a mean gas
     excess temperature. A face between two cells is pushed up by
     g * dT / T (Boussinesq: the lighter gas weighs less than the gas it
     displaces). The speed on each face is kept from tick to tick and bled
     off by a drag, so a plume has a little inertia.
  2. PROJECTION. The speeds are made to leave no cell with more gas than it
     started with: a pressure is solved so that what flows out of a cell
     equals what the gas in it expands by. Walls block (a face passes what
     its pores allow, read lane by lane through the voxels, so a wall one
     voxel thick inside a cell still shuts the face). The open sky holds the
     pressure at zero. The solve is a preconditioned conjugate gradient, run
     to a tolerance, started from the last tick's answer.
     Pushing air up the plume pulls air in at its foot: that is ENTRAINMENT,
     and nothing here is told it. A hot room with a doorway fills its top
     with hot gas and draws cool air in at its bottom: the neutral plane,
     likewise.
  3. ADVECTION. The speeds are carried down to voxels (a straight line
     between a cell's two faces, which keeps the flow free of divergence
     voxel by voxel) and move the gas's heat, its smoke and its oxygen
     deficit in FLUX FORM with a slope limiter. Flux form means what leaves
     one voxel arrives in the next, so heat and smoke are exact; the limiter
     means a sharp layer stays sharp and no new extreme appears.
     The cells where gas moves faster than a voxel a tick take SUBSTEPS: the
     rest of the world does not pay for the flame.
     Gas goes through the pores of a solid that has pores (a bed's planks), and
     takes the heat of the air in them: the air leaves such a voxel at the
     voxel's own temperature, so a hot flow warms it and a hot plank warms the
     flow. A solid with no pores passes nothing.

Oxygen is carried as its DEFICIT below fresh air (zero in fresh air), so air
that is all fresh stays exactly as it is whatever the flow does to it, and
air arriving from outside is fresh.

WHAT LEAVES THE WORLD is booked: heat into `shed`, smoke into
`gone["smoke"]`.

WHAT IS NOT IN IT, said so no one thinks it is: no momentum advection (speeds
are carried by the pressure and the drag, not by themselves; the plume does not
turn the corner under a ceiling at speed, so a ceiling jet comes out about
twice as hot and a third as fast as Alpert's, and this is the first thing to
add), no turbulence model (the only mixing is the scheme's own, and
GAS_ENTRAIN), and a sealed room's pressure is not tracked (see
`_gf_geometry`). A plume thinner than a cell (a few kW below a metre) shows as
the same heat in twice the gas.
"""
import weakref

import numpy as np
from .kernels import HAVE_NUMBA, njit
from .label import regions
from .materials import AIR, AIR_DENS, AMBIENT, GRAVITY, O2_PER_L, TICK_S

# ── numbers (all of them are data a reader can change; none is a flag) ──────
GAS_DRAG = 0.75          # 1/s. Bleeds a face's speed off each tick. A parcel
                         # 100 K warm would reach g dT / T / drag = 3.3 m/s before it
                         # stops speeding up. With GAS_ENTRAIN, the one number the
                         # plume's speed and its entrainment depend on (chosen
                         # against Heskestad's plume, Zukoski's filling and the
                         # doorway together; see tests/test_gasflow.py)
GAS_DIFFUSIVITY = 1e-4   # m^2/s, what smoke seeps by besides the flow (air's own is
                         # 2e-5; the rest is the slow eddying the lattice cannot see)
GAS_QUIET_K = 0.1        # K. Gas warmer or cooler than ambient by less than this is
                         # not worth carrying (conduction spreads a trace of heat
                         # to the whole room in seconds; carrying all of it
                         # would sweep the whole room every tick). It stays
                         # where it is, and is still counted.
GAS_ENTRAIN = 0.03       # the turbulent mixing the cells cannot see: scalars diffuse
                         # with K = GAS_ENTRAIN * (cell width) * (local speed), the
                         # radial entrainment of a plume (alpha, about 0.1 in
                         # Morton, Taylor and Turner's top-hat plume). Zero in air at
                         # rest, so a still layer stays sharp
GAS_KMAX = 0.1           # most a voxel hands a neighbour of the difference in a
                         # tick by that mixing (stable below 1/6 a face)
GAS_QUIET_SMOKE = 1e-4   # g a voxel. Less smoke than this is not worth carrying (it is
                         # a trace: a room with a fire in it holds a gram a voxel)
GAS_QUIET_O2 = 1e-2      # share of fresh air's oxygen. A smaller deficit is not carried
                         # (a fire only feels the air below 15%)
GAS_CFL = 0.8            # most of a voxel's gas that may leave it in one pass
GAS_SUBSTEPS_MAX = 8     # most passes a fast zone takes in a tick
GAS_P_TOL = 1e-5         # the solve stops when the residual is this fraction
                         # of the right-hand side
GAS_P_ITERS = 400        # ...or after this many iterations (it says so)
GAS_DRAG_QUAD = 0.0      # m^0. The drag that grows with speed: du/dt = -GAS_DRAG_QUAD
                         # * |u| * u / (cell width). Zero until a check wants it
GAS_GEO_TICKS = 20       # ticks between looks at the pores (half a second)
GAS_STILL = 1e-3        # m/s. A flow slower than this is not going anywhere
GAS_UMAX = 60.0          # m/s, a sanity bound on a face (no real flow is near it)
GAS_FVOL = 0.01          # a voxel holding more liquid than this share of its
                         # capacity is not gas, and its heat does not ride
POR_WALL = 0.01          # a voxel with less pore space than this passes no gas
_BIG = 1e9               # the resistance of a wall voxel in a lane
_T0K = AMBIENT + 273.15  # ambient, kelvin


def _sl(axis, s):
    t = [slice(None)] * 3
    t[axis] = s
    return tuple(t)


# Arrays the carry makes every tick (a face array per axis, a flux per axis)
# are the same shape tick after tick, and fresh memory costs a page fault per
# 4 KB: they are kept beside the world (not in it, so a snapshot does not copy
# them) and handed out again. Whoever takes one overwrites all of it.
_SCRATCH = weakref.WeakKeyDictionary()


def _scratch(world, name, shape, dtype):
    kept = _SCRATCH.setdefault(world, {})
    a = kept.get(name)
    if a is None or a.shape != shape or a.dtype != dtype:
        a = kept[name] = np.empty(shape, dtype)
    return a


def face_open(por, cs, open_sky):
    """How open is every face between two cells, 0 to 1, as three arrays
    (one per axis, one longer than the cell count along that axis: the first
    and last are the world's edge, shut unless the sky is open there).

    A face's openness is read through the VOXELS: the flow from one cell's
    centre to the next's goes through `cs` voxels in a row (half of each
    cell), a lane; the lane's resistance is the sum of 1/porosity of those
    voxels, and the face is the mean of its `cs` x `cs` lanes' conductances
    (an open lane is 1). So a wall one voxel thick, deep inside a cell,
    still shuts every lane it crosses, which a read of the cell's mean
    porosity cannot see (the coarse cell mean calls a 5 cm wall in a 20 cm
    cell three-quarters air). A door with 3% pore space passes 3% of a lane.
    Beyond the lattice the pad is wall, or open air at an open sky."""
    nc = tuple(-(-n // cs) for n in por.shape)
    N = tuple(c * cs for c in nc)
    pad = 1.0 if open_sky else _BIG
    inv = np.full(N, pad, np.float64)
    ok = por >= POR_WALL
    inv[:por.shape[0], :por.shape[1], :por.shape[2]] = np.where(
        ok, 1.0 / np.maximum(por, POR_WALL), _BIG)
    half = cs // 2
    out = []
    for axis in range(3):
        v = np.moveaxis(inv, axis, 0)
        n = nc[axis]
        A, B = v.shape[1] // cs, v.shape[2] // cs
        F = np.zeros((n + 1, A, B))

        def lanes(R):
            lane = np.where(R < _BIG, cs / np.minimum(R, _BIG), 0.0)
            return lane.reshape(R.shape[0], A, cs, B, cs).mean(axis=(2, 4))
        if n > 1:
            R = sum(v[half + k: half + k + cs * (n - 1): cs] for k in range(cs))
            F[1:n] = lanes(R)
        if open_sky:
            if axis != 2:                                # (z=0 is the ground)
                R = sum(v[k] for k in range(half)) + half
                F[0] = lanes(R[None])[0]
            R = sum(v[N[axis] - 1 - k] for k in range(half)) + half
            F[n] = lanes(R[None])[0]
        out.append(np.moveaxis(F, 0, axis))
    return out


def _block_sum(a, cs):
    """Sum of every cs x cs x cs block of `a` (zero-padded to whole blocks)."""
    pad = [(0, (-s) % cs) for s in a.shape]
    if any(p[1] for p in pad):
        a = np.pad(a, pad)
    n = [s // cs for s in a.shape]
    return a.reshape(n[0], cs, n[1], cs, n[2], cs).sum(axis=(1, 3, 5))


def _face_gather(u, axis, sl, cs):
    """A cell-face array `u` (one longer than the cells along `axis`) seen at
    the VOXEL faces of the region `sl`: a straight line between a cell's two
    faces. At a cell face it is that face's value. Shape: the region, one
    longer along `axis`."""
    lo, hi = sl[axis].start, sl[axis].stop
    m = np.arange(lo, hi + 1)
    n = u.shape[axis] - 1
    c0 = m // cs
    c1 = np.minimum(c0 + 1, n)
    k = ((m % cs) / cs).astype(np.float32)
    idx0, idx1 = [], []
    for ax in range(3):
        sh = [1, 1, 1]
        sh[ax] = -1
        if ax == axis:
            idx0.append(c0.reshape(sh))
            idx1.append(c1.reshape(sh))
        else:
            t = (np.arange(sl[ax].start, sl[ax].stop) // cs).reshape(sh)
            idx0.append(t)
            idx1.append(t)
    sh = [1, 1, 1]
    sh[axis] = -1
    a, b = u[tuple(idx0)], u[tuple(idx1)]
    return a + (b - a) * k.reshape(sh)


def _flux(f, s, ok, axis):
    """The flux of `f` through every face along `axis`, in units of `f`
    (positive = toward higher index), for face Courant numbers `s` (zero where
    nothing may pass). Second order where the field is smooth and first order
    at a jump: the slope in a voxel is the van Leer mean of the two one-sided
    differences, and zero at an extreme; the face takes the upwind voxel's
    value plus half its slope, less the half the voxel moves in the step.
    `ok` says which faces may pass gas at all (differences across the others
    are not differences of gas). Gas entering from outside the box is zero
    in every field carried here."""
    f = np.moveaxis(f, axis, 0)
    s = np.moveaxis(s, axis, 0)
    ok = np.moveaxis(ok, axis, 0)
    n = f.shape[0]
    fl = np.zeros(s.shape, np.float32)
    if n > 1:
        d = (f[1:] - f[:-1]) * ok[1:-1]
        slope = np.zeros(f.shape, np.float32)
        if n > 2:
            a, b = d[:-1], d[1:]
            prod = a * b
            den = a + b
            lim = np.zeros(prod.shape, np.float32)
            np.divide(np.float32(2.0) * prod, den, out=lim, where=prod > 0)
            slope[1:-1] = lim
        si = s[1:-1]
        half = np.float32(0.5)
        one = np.float32(1.0)
        phi = np.where(si > 0, f[:-1] + half * (one - si) * slope[:-1],
                       f[1:] - half * (one + si) * slope[1:])
        fl[1:-1] = si * phi
    fl[0] = np.where(s[0] < 0, s[0] * f[0], 0.0)
    fl[-1] = np.where(s[-1] > 0, s[-1] * f[-1], 0.0)
    return np.moveaxis(fl, 0, axis)


def _diffuse(f, Kv, ok, coef, kmax):
    """Turbulent mixing of `f` between neighbouring voxels, axis by axis: a face
    with speed-dependent diffusivity passes `min(mean K * coef, kmax)` of the
    difference (`ok` says which faces pass this kind of field). Flux form:
    what one voxel gives its neighbour takes."""
    half = np.float32(0.5)
    for ax in range(3):
        a, b = _sl(ax, slice(None, -1)), _sl(ax, slice(1, None))
        kf = np.minimum((Kv[a] + Kv[b]) * half * coef, kmax)
        q = np.where(ok[ax][_sl(ax, slice(1, -1))], kf * (f[a] - f[b]), np.float32(0.0))
        f[a] -= q
        f[b] += q
    return f


def _pcg(x, d, cond, D, active, tol, maxit):
    """Solve L x = d for the Laplacian L whose face conductances are `cond`
    (interior faces, one list entry per axis) and whose diagonal is `D`
    (every face, the world's open edge included: a pressure of zero out
    there). Jacobi-preconditioned conjugate gradient, to `tol` of |d|, from
    `x`. Returns x, the iterations used and the relative residual."""
    def mv(v):
        y = D * v
        for ax in range(3):
            a, b = _sl(ax, slice(None, -1)), _sl(ax, slice(1, None))
            c = cond[ax]
            y[a] -= c * v[b]
            y[b] -= c * v[a]
        return y
    inv = np.where(active, 1.0 / np.where(active, D, 1.0), 0.0)
    x = np.where(active, x, 0.0)
    r = np.where(active, d - mv(x), 0.0)
    dn = float(np.sqrt((d * d).sum()))
    if dn == 0.0:
        return np.zeros_like(x), 0, 0.0
    z = inv * r
    p = z.copy()
    rz = float((r * z).sum())
    it = 0
    res = float(np.sqrt((r * r).sum())) / dn
    while res > tol and it < maxit:
        Ap = mv(p)
        pAp = float((p * Ap).sum())
        if pAp <= 0.0:
            break
        alpha = rz / pAp
        x += alpha * p
        r -= alpha * Ap
        it += 1
        res = float(np.sqrt((r * r).sum())) / dn
        z = inv * r
        rz2 = float((r * z).sum())
        p = z + (rz2 / rz) * p
        rz = rz2
    return x, it, res


# ── the compiled path ────────────────────────────────────────────────────────
# Each kernel below does what its numpy twin does, element by element, in the
# same float32 order (tests/test_gasflow.py holds them to the bit). The numpy
# versions (`_face_gather`, the mask block of `_gf_faces`, `_flux`, and the
# diff-and-subtract in `_gf_carry`) are the definition.
@njit(cache=True)
def _gf_faces_nb(u, axis, cx, cy, cz, fx, fy, fz, dtdx, por, pok, actv, fastv,
                 lo_edge, hi_edge, sP, okP, ff, fa):
    """The face block of `_gf_faces`, fused: for each voxel face of the region
    the Courant number (`_face_gather`, times `dtdx`) less what the pores hold
    back, whether a pore passes at all, whether it is a fast face, and whether
    it is an ACTIVE face (next to a voxel that is not at rest: the rest are not worked
    out at all). `cx`... are the cell each voxel face lies in (or just below),
    per axis, and `fx`... how far along the cell it is."""
    n = por.shape[axis]
    nu = u.shape[axis] - 1
    zero = np.float32(0.0)
    for i in range(sP.shape[0]):
        for j in range(sP.shape[1]):
            for l in range(sP.shape[2]):
                m = i if axis == 0 else (j if axis == 1 else l)
                ai = i - 1 if axis == 0 else i
                aj = j - 1 if axis == 1 else j
                al = l - 1 if axis == 2 else l
                # the voxels either side (a: below the face, b: above it)
                if 0 < m < n:
                    act = actv[ai, aj, al] or actv[i, j, l]
                elif m == 0:
                    act = actv[i, j, l]
                else:
                    act = actv[ai, aj, al]
                fa[i, j, l] = act
                if not act:
                    sP[i, j, l] = zero
                    okP[i, j, l] = False
                    ff[i, j, l] = False
                    continue
                if axis == 0:
                    c1 = min(cx[i] + 1, nu)
                    ua, ub = u[cx[i], cy[j], cz[l]], u[c1, cy[j], cz[l]]
                    k = fx[i]
                elif axis == 1:
                    c1 = min(cy[j] + 1, nu)
                    ua, ub = u[cx[i], cy[j], cz[l]], u[cx[i], c1, cz[l]]
                    k = fy[j]
                else:
                    c1 = min(cz[l] + 1, nu)
                    ua, ub = u[cx[i], cy[j], cz[l]], u[cx[i], cy[j], c1]
                    k = fz[l]
                sv = (ua + (ub - ua) * k) * dtdx
                pe = False
                pf = zero
                f = False
                if 0 < m < n:
                    pe = pok[ai, aj, al] and pok[i, j, l]
                    pa, pb = por[ai, aj, al], por[i, j, l]
                    pf = pa if pa < pb else pb
                    f = fastv[ai, aj, al] or fastv[i, j, l]
                elif m == 0:
                    if lo_edge:
                        pe = pok[i, j, l]
                        pf = por[i, j, l]
                    f = fastv[i, j, l]
                else:
                    if hi_edge:
                        pe = pok[ai, aj, al]
                        pf = por[ai, aj, al]
                    f = fastv[ai, aj, al]
                sP[i, j, l] = sv * pf if pe else zero
                okP[i, j, l] = pe
                ff[i, j, l] = f


@njit(cache=True)
def _gf_flux_nb(f, s, ok, ff, fa, inv, p, nsub, axis, fl):
    """`_flux` along `axis` for the Courant numbers a pass uses (the fast faces
    take 1/nsub of theirs; later passes move only the fast ones), fused. A face
    with nothing moving on either side (`fa` false) has no flux."""
    nx, ny, nz = f.shape
    n = f.shape[axis]
    zero = np.float32(0.0)
    half = np.float32(0.5)
    one = np.float32(1.0)
    two = np.float32(2.0)
    for i in range(fl.shape[0]):
        for j in range(fl.shape[1]):
            for l in range(fl.shape[2]):
                if not fa[i, j, l]:
                    fl[i, j, l] = zero               # at rest on both sides
                    continue
                m = i if axis == 0 else (j if axis == 1 else l)
                sv = s[i, j, l]
                if nsub > 1:
                    if ff[i, j, l]:
                        sv = sv * inv
                    elif p > 0:
                        sv = zero
                di = 1 if axis == 0 else 0
                dj = 1 if axis == 1 else 0
                dl = 1 if axis == 2 else 0
                if m == 0:
                    fl[i, j, l] = sv * f[i, j, l] if sv < zero else zero
                elif m == n:
                    fl[i, j, l] = sv * f[i - di, j - dj, l - dl] if sv > zero else zero
                else:
                    # the upwind voxel, and the slope in it
                    if sv > zero:
                        c = m - 1
                        ci, cj, cl = i - di, j - dj, l - dl
                        # faces c and c + 1 are m - 1 and m
                        fm = (f[ci - di, cj - dj, cl - dl] if c > 0 else zero)
                        f0 = f[ci, cj, cl]
                        fp = f[i, j, l]
                        okl = ok[ci, cj, cl] if c > 0 else False
                        okr = ok[i, j, l]
                        c_is_end = c == 0
                    else:
                        c = m
                        ci, cj, cl = i, j, l
                        fm = f[i - di, j - dj, l - dl]
                        f0 = f[i, j, l]
                        fp = (f[i + di, j + dj, l + dl] if c < n - 1 else zero)
                        okl = ok[i, j, l]
                        okr = ok[i + di, j + dj, l + dl] if c < n - 1 else False
                        c_is_end = c == n - 1
                    slope = zero
                    if not c_is_end:
                        dlo = (f0 - fm) * (one if okl else zero)
                        dhi = (fp - f0) * (one if okr else zero)
                        prod = dlo * dhi
                        if prod > zero:
                            slope = (two * prod) / (dlo + dhi)
                    if sv > zero:
                        phi = f0 + half * (one - sv) * slope
                    else:
                        phi = f0 - half * (one + sv) * slope
                    fl[i, j, l] = sv * phi


@njit(cache=True)
def _gf_apply_nb(f, fl0, fl1, fl2, new):
    """`new = f - diff(fl0) - diff(fl1) - diff(fl2)`, axis by axis in order."""
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            for l in range(f.shape[2]):
                v = f[i, j, l]
                v -= fl0[i + 1, j, l] - fl0[i, j, l]
                v -= fl1[i, j + 1, l] - fl1[i, j, l]
                v -= fl2[i, j, l + 1] - fl2[i, j, l]
                new[i, j, l] = v


@njit(cache=True)
def _gf_diffuse_nb(f, Kv, ok, coef, kmax, axis, q):
    """One axis of `_diffuse`, in place (`q` is a work array one shorter than
    `f` along `axis`)."""
    half = np.float32(0.5)
    zero = np.float32(0.0)
    di = 1 if axis == 0 else 0
    dj = 1 if axis == 1 else 0
    dl = 1 if axis == 2 else 0
    for i in range(q.shape[0]):
        for j in range(q.shape[1]):
            for l in range(q.shape[2]):
                if ok[i + di, j + dj, l + dl]:
                    x = (Kv[i, j, l] + Kv[i + di, j + dj, l + dl]) * half * coef
                    kf = x if x < kmax else kmax
                    q[i, j, l] = kf * (f[i, j, l] - f[i + di, j + dj, l + dl])
                else:
                    q[i, j, l] = zero
    for i in range(q.shape[0]):
        for j in range(q.shape[1]):
            for l in range(q.shape[2]):
                f[i, j, l] -= q[i, j, l]
    for i in range(q.shape[0]):
        for j in range(q.shape[1]):
            for l in range(q.shape[2]):
                f[i + di, j + dj, l + dl] += q[i, j, l]



class GasFlowMixin:
    """The gas-flow law. Everything here runs only when `gas_flow` is on."""

    # ── the pores of the world, as faces between cells ───────────────────────
    def _gf_geometry(self):
        """Face openness, the solver's matrix and the airspaces it splits
        into. Read off the pores again every GAS_GEO_TICKS ticks, not every
        tick: a burning room's wood thins every tick, and a person walking
        changes where the air is as often, but a wall's openness to a flow
        moves on the scale of seconds. (A wall that burns through opens to
        the flow within half a second.)

        An airspace with no way out (no open sky reachable through any pore)
        cannot take a net expansion: gas that heats there would raise the
        pressure, and a real room does, until its leaks or its windows give.
        This keeps no pressure for such a room: the mean of what expands is
        taken out, and the pressure is the one that moves the gas about
        inside. A leak makes it an airspace WITH a way out, and it takes the
        full push through the leak."""
        key = self.__dict__.get("_gf_geo")
        epoch = self.tick // GAS_GEO_TICKS
        if key is not None and key[0] == epoch and key[1] == self.open_sky:
            return key[2]
        por = self.porosity()
        cs = self._CS
        dx = 0.1 * self.scale
        H = cs * dx
        F = face_open(por, cs, self.open_sky)
        cond = [H * f for f in F]
        D = 0.0
        for ax in range(3):
            D = D + cond[ax][_sl(ax, slice(None, -1))] + cond[ax][_sl(ax, slice(1, None))]
        active = D > 0
        shape = D.shape
        # the airspaces: cells and open faces as nodes of a doubled grid
        big = tuple(2 * n - 1 for n in shape)
        M = np.zeros(big, bool)
        M[::2, ::2, ::2] = active
        for ax in range(3):
            t = [slice(None, None, 2)] * 3
            t[ax] = slice(1, None, 2)
            M[tuple(t)] = cond[ax][_sl(ax, slice(1, -1))] > 0
        lab, count = regions(M, fused=self.fused)
        lab = lab[::2, ::2, ::2]
        # cells that touch the world's open edge through a face that passes gas
        sky = np.zeros(shape, bool)
        for ax in range(3):
            sky[_sl(ax, 0)] |= cond[ax][_sl(ax, 0)] > 0
            sky[_sl(ax, -1)] |= cond[ax][_sl(ax, -1)] > 0
        has_way_out = np.zeros(max(count, 1), bool)
        if count:
            has_way_out[lab[sky & active]] = True
        closed = active & ~has_way_out[np.maximum(lab, 0)]
        lab_closed = lab[closed]
        count_closed = np.bincount(lab_closed, minlength=max(count, 1))
        geo = {"F": F, "cond": cond, "cond_int": [c[_sl(ax, slice(1, -1))]
                                                   for ax, c in enumerate(cond)],
               "D": D, "active": active, "closed": closed,
               "lab_closed": lab_closed, "n_closed": count_closed, "H": H}
        self._gf_geo = (epoch, self.open_sky, geo)
        return geo

    # ── the state kept from tick to tick ─────────────────────────────────────
    def _gf_state(self):
        """Speeds on the cell faces, the last pressure, and the gas's last
        temperatures (to see how much it has warmed since: warming is
        expansion). Started afresh when a tick was skipped, so a flow that
        stopped being asked about does not wake up with old speeds."""
        st = self.__dict__.get("_gf")
        if st is not None and st["tick"] >= self.tick - 1 \
                and st["shape"] == self.shape and st["open_sky"] == self.open_sky:
            return st
        geo = self._gf_geometry()
        st = {"u": [np.zeros(f.shape, np.float32) for f in geo["F"]],
              "phi": np.zeros(geo["D"].shape, np.float64),
              "theta": np.zeros(self.shape, np.float32),
              "gas": np.zeros(self.shape, bool), "reg": None, "box": None,
              "tick": self.tick, "shape": self.shape, "open_sky": self.open_sky,
              "stats": {}}
        self._gf = st
        return st

    @staticmethod
    def _gf_forget(st):
        """Wipe what was kept for the last region: it is rewritten for this
        tick's, and nothing old may be left outside it."""
        if st["reg"] is not None:
            st["theta"][st["reg"]] = 0.0
            st["gas"][st["reg"]] = False
        st["reg"], st["box"] = None, None

    def _gf_live(self):
        """Is there a flow still going, to be bled off or carried on?"""
        st = self.__dict__.get("_gf")
        return bool(st is not None and st["tick"] >= self.tick - 1
                    and any(float(np.abs(u).max()) > GAS_STILL for u in st["u"]))

    @property
    def gas_flow_stats(self):
        """What the last tick of the flow cost and how well it solved:
        iterations, relative residual, passes, the box worked over."""
        st = self.__dict__.get("_gf")
        return dict(st["stats"]) if st else {}

    # ── the region worked over ───────────────────────────────────────────────
    def _gf_region(self, there, pad, prev):
        """The box that holds everything `there` is true of and everything it
        was true of last tick (`prev`, a box or None), grown by `pad` voxels
        and rounded out to whole cells. Returns that region and the box of
        `there` alone; (None, None) when there is nothing at all."""
        cs = self._CS
        if not self.skip_quiet:
            box = [(0, n) for n in self.shape]
        else:
            box = []
            for ax in range(3):
                line = there.any(axis=tuple(a for a in range(3) if a != ax))
                nz = np.flatnonzero(line)
                if not len(nz):
                    box = None
                    break
                box.append((int(nz[0]), int(nz[-1]) + 1))
        if box is None and prev is None:
            return None, None
        both = box if prev is None else (prev if box is None else
                                         [(min(a, c), max(b, d)) for (a, b), (c, d) in zip(box, prev)])
        out = []
        for ax, (a, b) in enumerate(both):
            a = max(a - pad, 0) // cs * cs
            b = min(-(-(b + pad) // cs) * cs, self.shape[ax])
            out.append(slice(a, b))
        return tuple(out), box

    # ── the law ──────────────────────────────────────────────────────────────
    def _law_gasflow(self):
        """One tick of the flow: buoyancy, projection, then carrying the heat,
        smoke and oxygen (the module docstring says how). Runs in place of
        `_law_air` while `gas_flow` is on."""
        cs = self._CS
        dx = 0.1 * self.scale
        H = cs * dx
        dt = TICK_S
        geo = self._gf_geometry()
        st = self._gf_state()
        u = st["u"]
        fresh = np.float32(O2_PER_L * self.vox_l)
        # COLD (T8 3.8b): gas is compared with the OUTSIDE air, not with 20 C. th_out is the
        # outside air in K above 20 C (0.0 with no climate: the old lines, verbatim).
        th_out = 0.0 if self.climate is None or self.t_out is None else self.t_out - AMBIENT
        if th_out != 0.0:
            ref = np.where(self.mat == AIR, np.float32(th_out), self._rest_cell())
            warm = np.abs(self.E - ref * self.heat_capacity()) > GAS_QUIET_K * self.c_airbase
        else:
            warm = np.abs(self.E) > GAS_QUIET_K * self.c_airbase
        there = (warm
                 | (self.smoke > GAS_QUIET_SMOKE)
                 | (np.abs(self.o2 - fresh) > GAS_QUIET_O2 * fresh))
        extra = self._gf_extra_there()           # C3: vapour is carried too (None = none)
        if extra is not None:
            there = there | extra
        reg, box = self._gf_region(there, GAS_SUBSTEPS_MAX + 1, st["box"])
        st["tick"] = self.tick
        self.p_add[:] = 0.0                      # (the old law's push: not used)
        stats = st["stats"] = {"iters": 0, "res": 0.0, "passes": 0, "box": 0}
        if reg is None:                          # nothing to carry: bleed it off
            for a in u:
                a *= np.float32(1.0 - GAS_DRAG * dt)
                if float(np.abs(a).max()) <= GAS_STILL:
                    a[...] = 0.0                 # (and call it still)
            self._gf_forget(st)
            return
        stats["box"] = int(np.prod([s.stop - s.start for s in reg]))
        # ── the gas in the region, and the cells of it ──
        E = self.E[reg]
        C = self.heat_capacity()[reg]
        gas = (self.mat[reg] == AIR) & (self.fvol[reg] <= GAS_FVOL * self.cap)
        theta = np.where(gas, E / C, np.float32(0.0)).astype(np.float32)
        nc = geo["D"].shape
        ci = tuple(slice(s.start // cs, -(-s.stop // cs)) for s in reg)
        gsum = _block_sum(gas.astype(np.float32), cs)
        tsum = _block_sum(theta, cs)
        th_c = np.full(nc, th_out, np.float64) if th_out != 0.0 else np.zeros(nc, np.float64)   # (outside the region: the outside air)
        th_c[ci] = tsum / np.maximum(gsum, 1e-6)
        # expansion: gas that warmed since the last tick took more room
        # (dV = V dT / T); gas that cooled took less
        was_t = st["theta"][reg]
        was_g = st["gas"][reg]
        both = gas & was_g
        dexp = np.where(both, (theta - was_t) / (_T0K + theta), np.float32(0.0))
        Q = np.zeros(nc, np.float64)
        Q[ci] = _block_sum(dexp.astype(np.float32), cs) * (dx ** 3) / dt
        # ── the push: buoyancy on the vertical faces, drag on all ──
        bz = GRAVITY * (th_c - th_out) / (_T0K + th_c) if th_out != 0.0 else GRAVITY * th_c / (_T0K + th_c)
        # the drag: a steady part, and a part that grows with the speed itself
        # (the turbulent drag of a flow through a cell's width of mixing length)
        us = [a * np.maximum(1.0 - dt * (GAS_DRAG + GAS_DRAG_QUAD * np.abs(a) / H), 0.0
                             ).astype(np.float32) for a in u]
        F = geo["F"]
        bface = np.zeros(F[2].shape)
        bface[:, :, 1:-1] = 0.5 * (bz[:, :, :-1] + bz[:, :, 1:])
        bface[:, :, -1] = 0.5 * bz[:, :, -1]
        us[2] = us[2] + (dt * bface).astype(np.float32)
        for ax in range(3):
            us[ax] = us[ax] * (F[ax] > 0)
        # ── the projection ──
        A = H * H
        d = -Q
        for ax in range(3):
            fl = (A * F[ax]) * us[ax]
            d = d + fl[_sl(ax, slice(1, None))] - fl[_sl(ax, slice(None, -1))]
        closed = geo["closed"]
        if closed.any():                         # no way out: no net push
            sums = np.bincount(geo["lab_closed"], weights=d[closed],
                               minlength=len(geo["n_closed"]))
            d[closed] -= (sums / np.maximum(geo["n_closed"], 1))[geo["lab_closed"]]
        phi, it, res = _pcg(st["phi"], d, geo["cond_int"], geo["D"], geo["active"],
                            GAS_P_TOL, GAS_P_ITERS)
        st["phi"] = phi
        stats["iters"], stats["res"] = it, res
        # (flow from low to high is positive, and the pressure pushes from high
        # to low: u = u* - (phi_low - phi_high) / H, with a zero beyond the edge)
        for ax in range(3):
            un = us[ax].astype(np.float64)
            un[_sl(ax, slice(1, -1))] -= (phi[_sl(ax, slice(None, -1))]
                                          - phi[_sl(ax, slice(1, None))]) / H
            un[_sl(ax, 0)] += phi[_sl(ax, 0)] / H
            un[_sl(ax, -1)] -= phi[_sl(ax, -1)] / H
            un *= F[ax] > 0
            u[ax] = np.clip(un, -GAS_UMAX, GAS_UMAX).astype(np.float32)
        # ── advection: how fast, and where a pass is not enough ──
        cour = np.zeros(nc)
        for ax in range(3):
            lo, hi = u[ax][_sl(ax, slice(None, -1))], u[ax][_sl(ax, slice(1, None))]
            cour += (np.maximum(np.maximum(lo, hi), 0.0)
                     + np.maximum(np.maximum(-lo, -hi), 0.0))
        cour *= dt / dx
        ratio = float(cour.max()) / (GAS_CFL * GAS_SUBSTEPS_MAX)
        if ratio > 1.0:                          # faster than the passes can carry:
            for a in u:                          # the flow is slowed to what they can
                a /= np.float32(ratio)
            cour /= ratio
        nsub = int(max(np.ceil(cour.max() / GAS_CFL), 1))
        fast_c = cour > GAS_CFL
        stats["passes"] = nsub
        if self.pcell is None:
            self.pcell = np.zeros(nc, np.float32)
            self.vcell = [np.zeros(nc, np.float32) for _ in range(3)]
        self.pcell[...] = (phi * AIR_DENS / dt).astype(np.float32)
        for ax in range(3):
            self.vcell[ax][...] = 0.5 * (u[ax][_sl(ax, slice(None, -1))]
                                         + u[ax][_sl(ax, slice(1, None))])
        # the mixing the cells cannot see grows with how fast the gas moves
        speed = np.sqrt(sum(v.astype(np.float64) ** 2 for v in self.vcell))
        kc = (GAS_ENTRAIN * H * speed).astype(np.float32)
        if th_out != 0.0:
            # the edge inflow carries E = 0, which must mean OUTSIDE air: carry the heat above the
            # outside (E - th_out C) and put the outside back after, on the same gas cells
            Er = self.E[reg]
            Er[gas] -= (np.float32(th_out) * C[gas]).astype(np.float32)
        self._gf_carry(reg, ci, u, fast_c, nsub, fresh, dx, dt, kc, there[reg], gas, C)
        if th_out != 0.0:
            gas_a = (self.mat[reg] == AIR) & (self.fvol[reg] <= GAS_FVOL * self.cap)
            Er = self.E[reg]
            Er[gas_a] += (np.float32(th_out) * self.heat_capacity()[reg][gas_a]).astype(np.float32)
        # ── and what the next tick will want to know ──
        gas2 = (self.mat[reg] == AIR) & (self.fvol[reg] <= GAS_FVOL * self.cap)
        self._gf_forget(st)
        st["theta"][reg] = np.where(gas2, self.E[reg] / self.heat_capacity()[reg],
                                    np.float32(0.0))
        st["gas"][reg] = gas2
        st["reg"], st["box"] = reg, box

    # ── carrying heat, smoke and oxygen ──────────────────────────────────────
    def _gf_faces(self, reg, por, u, fastv, actv, nsub, dx, dt):
        """For every voxel face of the region, along each axis: the Courant
        number of the flow through it (what share of the voxel upstream moves
        across in a tick) less what the pores hold back (the narrower voxel's
        porosity), whether a pore passes at all (heat, smoke and oxygen all go
        through the same pores), whether it is a fast face (next to a cell that
        needs substeps), and whether it is ACTIVE (next to a voxel that is not
        at rest: only those faces are worked out, and the rest carry nothing).
        Compiled when the world is fused, with the same sums in the same
        order."""
        cs = self._CS
        shape = tuple(s.stop - s.start for s in reg)
        pass_ok = por >= POR_WALL
        fused = self.fused and HAVE_NUMBA
        dtdx = np.float32(dt / dx)
        out = []
        for ax in range(3):
            lo_edge = reg[ax].start == 0 and ax != 2 and self.open_sky
            hi_edge = reg[ax].stop == self.shape[ax] and self.open_sky
            fshape = tuple(n + 1 if a == ax else n for a, n in enumerate(shape))
            if fused:
                cc, ff_ = [], []
                for a_ in range(3):
                    m = np.arange(reg[a_].start, reg[a_].stop + 1)
                    cc.append(np.minimum(m // cs, u[ax].shape[a_] - 1))
                    ff_.append(((m % cs) / cs).astype(np.float32))
                sP = _scratch(self, "sP%d" % ax, fshape, np.float32)
                okP, ff, fa = (_scratch(self, "%s%d" % (nm, ax), fshape, np.bool_)
                               for nm in ("okP", "ff", "fa"))
                _gf_faces_nb(u[ax], ax, cc[0], cc[1], cc[2], ff_[0], ff_[1], ff_[2], dtdx,
                             por, pass_ok, actv, fastv, lo_edge, hi_edge,
                             sP, okP, ff, fa)
            else:
                s = _face_gather(u[ax], ax, reg, cs) * dtdx
                a, b = _sl(ax, slice(None, -1)), _sl(ax, slice(1, None))
                pe = np.zeros(fshape, bool)
                pe[_sl(ax, slice(1, -1))] = pass_ok[a] & pass_ok[b]
                pf = np.zeros(fshape, np.float32)
                pf[_sl(ax, slice(1, -1))] = np.minimum(por[a], por[b])
                fa = np.zeros(fshape, bool)
                fa[_sl(ax, slice(1, -1))] = actv[a] | actv[b]
                fa[_sl(ax, 0)] = actv[_sl(ax, 0)]
                fa[_sl(ax, -1)] = actv[_sl(ax, -1)]
                if lo_edge:
                    pe[_sl(ax, 0)] = pass_ok[_sl(ax, 0)]
                    pf[_sl(ax, 0)] = por[_sl(ax, 0)]
                if hi_edge:
                    pe[_sl(ax, -1)] = pass_ok[_sl(ax, -1)]
                    pf[_sl(ax, -1)] = por[_sl(ax, -1)]
                ff = np.zeros(fshape, bool)
                if nsub > 1:
                    ff[_sl(ax, slice(1, -1))] = fastv[a] | fastv[b]
                    ff[_sl(ax, 0)] = fastv[_sl(ax, 0)]
                    ff[_sl(ax, -1)] = fastv[_sl(ax, -1)]
                pe, ff = pe & fa, ff & fa
                sP = np.where(pe, s * pf, np.float32(0.0))
                okP = pe
            out.append((sP, okP, ff, fa))
        return out

    def _gf_carry(self, reg, ci, u, fast_c, nsub, fresh, dx, dt, kc, core, gas, C):
        cs = self._CS
        shape = tuple(s.stop - s.start for s in reg)
        fastv = np.zeros(shape, bool)
        if nsub > 1:
            fastv = self._gf_up(fast_c[ci], shape)
        # WHERE ANYTHING IS MOVING: the blocks of cells that hold something not
        # at rest (`core`), and the blocks a pass can carry it into. Faces
        # with nothing either side carry nothing, and are not worked out.
        if self.skip_quiet:
            held = _block_sum(core.astype(np.float32), cs) > 0
            for _ in range(-(-nsub // cs)):
                held = self._gf_grow(held)
            actv = self._gf_up(held, shape)
        else:
            actv = np.ones(shape, bool)
        por = np.ascontiguousarray(self.porosity()[reg])
        faces = self._gf_faces(reg, por, u, np.ascontiguousarray(fastv),
                               np.ascontiguousarray(actv), nsub, dx, dt)
        sP = [f[0] for f in faces]
        okP = [f[1] for f in faces]
        fastf = [f[2] for f in faces]
        actf = [f[3] for f in faces]
        # WHAT THE FLOW CARRIES AS HEAT. In a gas voxel, all of its heat. In a
        # voxel that is a solid with pores in it (a bed's planks, a thatch) the
        # air that passes through it is at the solid's temperature, and only
        # that air's heat rides: the pore share of an air voxel's capacity, at
        # the voxel's temperature. The solid keeps the rest, and is warmed or
        # cooled by what passes (the carried field's change is added to it).
        # Air that cannot pass the pores (a wall) carries nothing through.
        E0 = self.E[reg]
        carried = np.where(gas, E0, (por * np.float32(self.c_airbase))
                           * (E0 / C)).astype(np.float32)
        fields = [[carried.copy(), sP, okP]]
        smoke = self.smoke[reg]
        has_smoke = bool((smoke != 0).any())
        if has_smoke:
            fields.append([smoke.copy(), sP, okP])
        delta = fresh - self.o2[reg]
        has_o2 = bool((delta != 0).any())
        if has_o2:
            fields.append([delta.copy(), sP, okP])
        extras = self._gf_extra_fields(reg)      # C3: [(name, array)] carried like smoke
        for _nm, _arr in extras:
            fields.append([_arr.copy(), sP, okP])
        out = [0.0] * len(fields)
        box = None
        if nsub > 1 and fastv.any():
            box = []
            for ax in range(3):
                line = fastv.any(axis=tuple(a for a in range(3) if a != ax))
                nz = np.flatnonzero(line)
                box.append(slice(max(int(nz[0]) - 2, 0), min(int(nz[-1]) + 3, shape[ax])))
            box = tuple(box)
        inv = np.float32(1.0 / nsub)
        fused = self.fused and HAVE_NUMBA
        for p in range(nsub):
            if p > 0 and box is None:
                break
            win = box if p > 0 else tuple(slice(0, n) for n in shape)
            for i, (f, ss, oks) in enumerate(fields):
                fw = np.ascontiguousarray(f[win])
                fs = [tuple(slice(w.start, w.stop + 1) if a == ax else w
                            for a, w in enumerate(win)) for ax in range(3)]
                s3 = [np.ascontiguousarray(ss[ax][fs[ax]]) for ax in range(3)]
                ok3 = [np.ascontiguousarray(oks[ax][fs[ax]]) for ax in range(3)]
                ff3 = [np.ascontiguousarray(fastf[ax][fs[ax]]) for ax in range(3)]
                fa3 = [np.ascontiguousarray(actf[ax][fs[ax]]) for ax in range(3)]
                fl3 = [_scratch(self, "fl%d" % ax, x.shape, np.float32)
                       for ax, x in enumerate(s3)]
                new = _scratch(self, "new", fw.shape, np.float32)
                if fused:
                    for ax in range(3):
                        _gf_flux_nb(fw, s3[ax], ok3[ax], ff3[ax], fa3[ax], inv, p, nsub, ax, fl3[ax])
                    _gf_apply_nb(fw, fl3[0], fl3[1], fl3[2], new)
                else:
                    for ax in range(3):
                        s_ax = s3[ax]
                        if nsub > 1:
                            s_ax = np.where(ff3[ax], s_ax * inv,
                                            s_ax if p == 0 else np.float32(0.0))
                        fl3[ax] = np.where(fa3[ax], _flux(fw, s_ax, ok3[ax], ax),
                                           np.float32(0.0))
                    new = fw.copy()
                    for ax in range(3):
                        new -= np.diff(fl3[ax], axis=ax)
                for ax in range(3):
                    out[i] += self._gf_edges(fl3[ax], ax, reg, win)
                f[win] = new
        # the mixing of what the flow carries, where the gas is moving
        if float(kc[ci].max()) > 0.0:
            ix = [np.arange(r.start, r.stop) // cs for r in reg]
            Kv = np.ascontiguousarray(kc[np.ix_(*ix)])
            coef, kmax = np.float32(dt / (dx * dx)), np.float32(GAS_KMAX)
            for f, ss, oks in fields:
                if fused:
                    for ax in range(3):
                        qs = tuple(n - 1 if a == ax else n for a, n in enumerate(shape))
                        _gf_diffuse_nb(f, Kv, oks[ax], coef, kmax, ax,
                                       _scratch(self, "q%d" % ax, qs, np.float32))
                else:
                    _diffuse(f, Kv, oks, coef, kmax)
        self.E[reg] = np.where(gas, fields[0][0], E0 + (fields[0][0] - carried))
        k = 1
        if has_smoke:
            self.smoke[reg] = fields[k][0]
            k += 1
        if has_o2:
            self.o2[reg] = fresh - fields[k][0]
            k += 1
        for _nm, _arr in extras:
            self._gf_extra_done(reg, _nm, fields[k][0], out[k])
            k += 1
        self.shed += out[0]
        if has_smoke:
            self.gone["smoke"] = self.gone.get("smoke", 0.0) + out[1]

    # ── hooks for extra carried fields (C3: vapour). Defaults carry nothing. ──
    def _gf_extra_there(self):
        """A boolean per cell of extra things the flow must carry, or None."""
        return None

    def _gf_extra_fields(self, reg):
        """[(name, float32 array of the region)] carried like smoke."""
        return []

    def _gf_extra_done(self, reg, name, arr, left):
        """Take the carried field back; `left` is what went out through the world's edge."""

    def _gf_up(self, cells, shape):
        """A boolean per cell, as a boolean per voxel of a region of `shape`."""
        cs = self._CS
        up = np.repeat(np.repeat(np.repeat(cells, cs, 0), cs, 1), cs, 2)
        return up[:shape[0], :shape[1], :shape[2]]

    @staticmethod
    def _gf_grow(cells):
        """The cells and every cell that touches one of them by a face."""
        out = cells.copy()
        for ax in range(3):
            lo, hi = _sl(ax, slice(None, -1)), _sl(ax, slice(1, None))
            out[lo] |= cells[hi]
            out[hi] |= cells[lo]
        return out

    def _gf_edges(self, fl, ax, reg, win):
        """What a pass sent out through the world's own edge: the flux on the
        first and last faces of the box when they are the lattice's."""
        tot = 0.0
        if win[ax].start == 0 and reg[ax].start == 0:
            tot -= float(np.take(fl, 0, axis=ax).sum(dtype=np.float64))
        if win[ax].stop == reg[ax].stop - reg[ax].start and reg[ax].stop == self.shape[ax]:
            tot += float(np.take(fl, -1, axis=ax).sum(dtype=np.float64))
        return tot
