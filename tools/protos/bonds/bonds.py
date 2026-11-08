"""BONDED OBJECTS: rigid pieces joined by bonds that stretch, bend and break.

A prototype for roadmap decision 2 ("explicit bonds, replacing the
continuum"), built BESIDE src/mpm and changing nothing in it. The user chose
"measure first", so this file exists to be measured against the continuum on
the same scenes (scenes.py, continuum.py, run.py). It is not wired into the
voxel world.

THE MODEL is the bonded-particle model of rock mechanics (Potyondy and
Cundall, "A bonded-particle model for rock", 2004) with the bond written in
total form, as a beam between two rigid pieces:

- A PIECE is a rigid box: position, velocity, orientation (a quaternion),
  angular momentum, mass and principal inertia. It touches walls and other
  pieces through a sphere of radius `rc` at its centre.
- A BOND joins two pieces at a shared point (the midpoint at rest). When the
  two attachment points drift apart, a spring pulls them back: stiff along
  the bond (stretch), less stiff across it (shear). When the pieces turn
  relative to each other, a rotational spring turns them back: bend and
  twist. Every stiffness is a beam's: E*A/L, G*A/L, E*I/L, G*J/L. A BALL
  joint (a rope's link) pulls the two points together the same every way
  and hardly resists turning.
- A bond BREAKS when the stress at the edge of its section passes its
  strength: tension plus bending (F/A + M*r/I) against the tensile strength,
  shear plus twist against the shear strength. A broken bond is gone, and its
  load goes to its neighbours; that is how a crack runs.

WHAT A BOND KNOWS IS ITS MATERIAL. Wood is strong along the grain and weak
across it (Hankinson's formula between the two); glass is the same every way
and has no give; a rope is a chain of short links on ball joints; a leaf
hangs on a thin petiole that bends.

CONSERVATION BY CONSTRUCTION. Each bond pushes its two pieces with equal and
opposite forces applied at ONE point, and equal and opposite moments, so the
internal forces add no momentum and no angular momentum. The integrator
(symplectic Euler, angular momentum as the stored state) keeps both exactly
in arithmetic. `Bonded.ledger` adds up every outside push (gravity, walls,
drag, loads, pinned pieces) as it happens, so the totals can be checked in
any scene, not only in free flight.

UNITS are SI, z is up, as in src/mpm.
"""

import numpy as np
from numba import njit

GRAVITY = 9.81
AIR_DENS = 1.2           # kg/m^3, the voxel sim's AIR_DENS
DRAG_CD = 1.1            # the voxel sim's DRAG_CD
RAMP = 0.25              # s: how long a pinned piece's shaking eases in and out


# ── materials ────────────────────────────────────────────────────────────────
# E (Pa), Poisson's ratio, tensile strength (Pa), damping ratio of a bond,
# density (kg/m^3). For wood the second row is ACROSS the grain.
#
# Wood and glass use the numbers src/mpm/core.py MAT uses (E 1.1e10 and
# 7e10; strength 1e7 and 5e7), so the two models are fed the same matter.
# The handbook puts wood's strength along the grain at 45-121 MPa and its
# bending strength at 59-99 MPa (docs/realism-references.md S3), so 1e7 is
# low; it is kept so that the comparison is like with like. Across the grain
# wood is about 1/20 as stiff (S2) and about 1/20 as strong (S4: 2.1-5.5 MPa
# against 45-121).
#
# Rope is 12 mm manila. Rope tables give a breaking load near 10.6 kN and
# about 10% stretch at break, so E*A is about 1.06e5 N over a 1.13e-4 m^2
# section (both from rope makers' tables; not opened here, so unverified).
#
# A leaf petiole is a green stalk: E a few hundred MPa, strength some tens
# of MPa (unverified, order of magnitude only). It matters only through
# where leaves break off, which the scene reports.
WOOD_L = (1.1e10, 0.35, 1.0e7, 0.10, 600.0)
WOOD_T = (1.1e10 / 20.0, 0.35, 1.0e7 / 20.0, 0.10, 600.0)
# ...and wood at the handbook's strength: 60 MPa along the grain (the low end
# of bending strength, S3) and 3 MPa across it (S4)
WOOD_L_HB = (1.1e10, 0.35, 6.0e7, 0.10, 600.0)
WOOD_T_HB = (1.1e10 / 20.0, 0.35, 3.0e6, 0.10, 600.0)
GLASS = (7.0e10, 0.22, 5.0e7, 0.02, 2500.0)
ROPE = (9.4e8, 0.40, 9.4e7, 0.10, 885.0)
PETIOLE = (3.0e8, 0.40, 2.0e7, 0.10, 900.0)
# ...and in autumn, when an abscission layer has formed at its base and the
# leaf is ready to drop: a tenth as strong (an assumption, for contrast)
PETIOLE_AUTUMN = (3.0e8, 0.40, 2.0e6, 0.10, 900.0)


def hankinson(a, b, cos2):
    """Hankinson's formula: a property `a` along the grain and `b` across it,
    for a direction whose cosine to the grain squared is `cos2`."""
    return a * b / (a * (1.0 - cos2) + b * cos2)


# ── small vector and quaternion algebra, on tuples ───────────────────────────
# Tuples, not arrays: numba keeps a tuple in registers, while a small array
# made per bond per substep is a heap allocation, and there are dozens of
# them in a bond.

@njit(cache=True, inline="always")
def _cross(a0, a1, a2, b0, b1, b2):
    return a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0


@njit(cache=True, inline="always")
def _qmul(a0, a1, a2, a3, b0, b1, b2, b3):
    return (a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
            a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
            a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0)


@njit(cache=True, inline="always")
def _rot(R, i, a0, a1, a2):
    """R[i] @ a"""
    return (R[i, 0, 0] * a0 + R[i, 0, 1] * a1 + R[i, 0, 2] * a2,
            R[i, 1, 0] * a0 + R[i, 1, 1] * a1 + R[i, 1, 2] * a2,
            R[i, 2, 0] * a0 + R[i, 2, 1] * a1 + R[i, 2, 2] * a2)


@njit(cache=True, inline="always")
def _rotT(R, i, a0, a1, a2):
    """R[i].T @ a"""
    return (R[i, 0, 0] * a0 + R[i, 1, 0] * a1 + R[i, 2, 0] * a2,
            R[i, 0, 1] * a0 + R[i, 1, 1] * a1 + R[i, 2, 1] * a2,
            R[i, 0, 2] * a0 + R[i, 1, 2] * a1 + R[i, 2, 2] * a2)


@njit(cache=True)
def _q2R(q, R):
    """Rotation matrices of the unit quaternions `q` (w, x, y, z), into R."""
    for i in range(q.shape[0]):
        w, x, y, z = q[i, 0], q[i, 1], q[i, 2], q[i, 3]
        R[i, 0, 0] = 1 - 2 * (y * y + z * z); R[i, 0, 1] = 2 * (x * y - w * z)
        R[i, 0, 2] = 2 * (x * z + w * y); R[i, 1, 0] = 2 * (x * y + w * z)
        R[i, 1, 1] = 1 - 2 * (x * x + z * z); R[i, 1, 2] = 2 * (y * z - w * x)
        R[i, 2, 0] = 2 * (x * z - w * y); R[i, 2, 1] = 2 * (y * z + w * x)
        R[i, 2, 2] = 1 - 2 * (x * x + y * y)


@njit(cache=True, inline="always")
def _qlog2(w, x, y, z):
    """The rotation vector of a unit quaternion, the short way round."""
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    s = np.sqrt(x * x + y * y + z * z)
    if s < 1e-12:
        return 2.0 * x, 2.0 * y, 2.0 * z
    k = 2.0 * np.arctan2(s, w) / s
    return k * x, k * y, k * z


@njit(cache=True, inline="always")
def _qexp(r0, r1, r2):
    """The unit quaternion of a rotation vector."""
    a = np.sqrt(r0 * r0 + r1 * r1 + r2 * r2)
    if a < 1e-12:
        w, x, y, z = 1.0, 0.5 * r0, 0.5 * r1, 0.5 * r2
    else:
        s = np.sin(0.5 * a) / a
        w, x, y, z = np.cos(0.5 * a), s * r0, s * r1, s * r2
    nq = np.sqrt(w * w + x * x + y * y + z * z)
    return w / nq, x / nq, y / nq, z / nq


@njit(cache=True, inline="always")
def _envelope(t, on, off):
    """How much of a pinned piece's shaking is on at time `t`, and how fast
    that changes: 0 outside [on, off], easing in and out over RAMP seconds
    (smoothstep), so a shaker starts and stops without a jolt."""
    if t <= on or t >= off:
        return 0.0, 0.0
    if t < on + RAMP:
        u = (t - on) / RAMP
        return u * u * (3 - 2 * u), 6 * u * (1 - u) / RAMP
    if t > off - RAMP:
        u = (off - t) / RAMP
        return u * u * (3 - 2 * u), -6 * u * (1 - u) / RAMP
    return 1.0, 0.0


# ── the solver ───────────────────────────────────────────────────────────────

@njit(cache=True)
def _run(nsub, h, t0,
         x, v, q, ell, m, Ibinv, rc, cda, kin, x0, amp, freq, kwin, fext, kc, cc,
         bi, bj, ai, aj, Q0, kn, ks, kb, kt, cn, cs, cb, ct,
         A, rb, Ib, Jb, sig, tau, alive, tbreak, smax,
         adj_ptr, adj_b, wlo, whi, mu, zc, g, rho_air, pp, ledger):
    """`nsub` substeps of `h` seconds from time `t0`. `ledger` holds the
    outside impulse and the outside angular impulse about the origin,
    [J(3), H(3)], and is added to as each outside push happens."""
    n = x.shape[0]
    nb = bi.shape[0]
    nw = wlo.shape[0]
    F = np.zeros((n, 3))
    T = np.zeros((n, 3))
    R = np.zeros((n, 3, 3))
    om = np.zeros((n, 3))
    # the contact grid for piece against piece: a hash of cells two contact
    # radii wide, rebuilt every substep
    rmax = 1e-9
    for i in range(n):
        if not kin[i] and rc[i] > rmax:
            rmax = rc[i]
    cell = 2.0 * rmax
    tsize = 1
    while tsize < 2 * n:
        tsize *= 2
    head = np.zeros(tsize + 1, np.int64)
    fill = np.zeros(tsize, np.int64)
    order = np.empty(n, np.int64)
    key = np.empty(n, np.int64)
    ci = np.empty((n, 3), np.int64)
    for s in range(nsub):
        t = t0 + s * h
        # pinned pieces move as they are told
        for i in range(n):
            if kin[i]:
                w2 = 2.0 * np.pi * freq[i]
                e, de = _envelope(t, kwin[i, 0], kwin[i, 1])
                sn, cs_ = np.sin(w2 * t), np.cos(w2 * t)
                for a in range(3):
                    x[i, a] = x0[i, a] + amp[i, a] * sn * e
                    v[i, a] = amp[i, a] * (w2 * cs_ * e + sn * de)
        _q2R(q, R)
        for i in range(n):
            if kin[i]:
                om[i, 0] = om[i, 1] = om[i, 2] = 0.0
            else:
                l0, l1, l2 = _rotT(R, i, ell[i, 0], ell[i, 1], ell[i, 2])
                om[i, 0], om[i, 1], om[i, 2] = _rot(R, i, l0 * Ibinv[i, 0],
                                                    l1 * Ibinv[i, 1], l2 * Ibinv[i, 2])
            for a in range(3):
                F[i, a] = 0.0
                T[i, a] = 0.0
        # gravity, loads and air drag: outside pushes on the free pieces
        for i in range(n):
            if kin[i]:
                continue
            sp = np.sqrt(v[i, 0] ** 2 + v[i, 1] ** 2 + v[i, 2] ** 2)
            dr = 0.5 * rho_air * cda[i] * sp
            f0 = fext[i, 0] - dr * v[i, 0]
            f1 = fext[i, 1] - dr * v[i, 1]
            f2 = fext[i, 2] - dr * v[i, 2] - m[i] * g
            F[i, 0] += f0; F[i, 1] += f1; F[i, 2] += f2
            t0_, t1_, t2_ = _cross(x[i, 0], x[i, 1], x[i, 2], f0, f1, f2)
            ledger[0] += f0 * h; ledger[1] += f1 * h; ledger[2] += f2 * h
            ledger[3] += t0_ * h; ledger[4] += t1_ * h; ledger[5] += t2_ * h
        # bonds
        for b in range(nb):
            if not alive[b]:
                continue
            i = bi[b]; j = bj[b]
            ri0, ri1, ri2 = _rot(R, i, ai[b, 0], ai[b, 1], ai[b, 2])
            rj0, rj1, rj2 = _rot(R, j, aj[b, 0], aj[b, 1], aj[b, 2])
            d0 = (x[j, 0] + rj0) - (x[i, 0] + ri0)
            d1 = (x[j, 1] + rj1) - (x[i, 1] + ri1)
            d2 = (x[j, 2] + rj2) - (x[i, 2] + ri2)
            n0 = x[j, 0] - x[i, 0]; n1 = x[j, 1] - x[i, 1]; n2 = x[j, 2] - x[i, 2]
            ln = max(np.sqrt(n0 * n0 + n1 * n1 + n2 * n2), 1e-12)
            n0 /= ln; n1 /= ln; n2 /= ln
            dn = d0 * n0 + d1 * n1 + d2 * n2
            # how fast the two attachment points move apart
            wi0, wi1, wi2 = _cross(om[i, 0], om[i, 1], om[i, 2], ri0, ri1, ri2)
            wj0, wj1, wj2 = _cross(om[j, 0], om[j, 1], om[j, 2], rj0, rj1, rj2)
            u0 = (v[j, 0] + wj0) - (v[i, 0] + wi0)
            u1 = (v[j, 1] + wj1) - (v[i, 1] + wi1)
            u2 = (v[j, 2] + wj2) - (v[i, 2] + wi2)
            un = u0 * n0 + u1 * n1 + u2 * n2
            fn_el = kn[b] * dn
            fn = fn_el + cn[b] * un
            s0 = d0 - dn * n0; s1 = d1 - dn * n1; s2 = d2 - dn * n2
            fs_el = ks[b] * np.sqrt(s0 * s0 + s1 * s1 + s2 * s2)
            us0 = u0 - un * n0; us1 = u1 - un * n1; us2 = u2 - un * n2
            # the force on i (and minus it on j)
            g0 = fn * n0 + ks[b] * s0 + cs[b] * us0
            g1 = fn * n1 + ks[b] * s1 + cs[b] * us1
            g2 = fn * n2 + ks[b] * s2 + cs[b] * us2
            # how far j has turned from where it sat on i, as a rotation
            # vector in the world frame: bend about the axes across the bond,
            # twist about the bond itself
            c0, c1, c2, c3 = _qmul(q[i, 0], -q[i, 1], -q[i, 2], -q[i, 3],
                                   q[j, 0], q[j, 1], q[j, 2], q[j, 3])
            c0, c1, c2, c3 = _qmul(c0, c1, c2, c3,
                                   Q0[b, 0], -Q0[b, 1], -Q0[b, 2], -Q0[b, 3])
            e0, e1, e2 = _qlog2(c0, c1, c2, c3)
            th0, th1, th2 = _rot(R, i, e0, e1, e2)
            tt = th0 * n0 + th1 * n1 + th2 * n2
            hb0 = th0 - tt * n0; hb1 = th1 - tt * n1; hb2 = th2 - tt * n2
            dw0 = om[j, 0] - om[i, 0]; dw1 = om[j, 1] - om[i, 1]; dw2 = om[j, 2] - om[i, 2]
            dwt = dw0 * n0 + dw1 * n1 + dw2 * n2
            M0 = kb[b] * hb0 + kt[b] * tt * n0 + cb[b] * (dw0 - dwt * n0) + ct[b] * dwt * n0
            M1 = kb[b] * hb1 + kt[b] * tt * n1 + cb[b] * (dw1 - dwt * n1) + ct[b] * dwt * n1
            M2 = kb[b] * hb2 + kt[b] * tt * n2 + cb[b] * (dw2 - dwt * n2) + ct[b] * dwt * n2
            # the stress at the edge of the section, against its strength
            mb = kb[b] * np.sqrt(hb0 * hb0 + hb1 * hb1 + hb2 * hb2)
            mt = kt[b] * abs(tt)
            st = fn_el / A[b] + mb * rb[b] / Ib[b]
            ts = fs_el / A[b] + mt * rb[b] / Jb[b]
            r = max(st / sig[b], ts / tau[b])
            if r > smax[b]:
                smax[b] = r
            if r > 1.0:
                alive[b] = False
                tbreak[b] = t
                continue
            # both pushes act at ONE point, the middle of the two attachment
            # points, so the pair adds no turning to the whole
            p0 = x[i, 0] + ri0 + 0.5 * d0
            p1 = x[i, 1] + ri1 + 0.5 * d1
            p2 = x[i, 2] + ri2 + 0.5 * d2
            if not kin[i]:
                F[i, 0] += g0; F[i, 1] += g1; F[i, 2] += g2
                k0, k1, k2 = _cross(p0 - x[i, 0], p1 - x[i, 1], p2 - x[i, 2], g0, g1, g2)
                T[i, 0] += k0 + M0; T[i, 1] += k1 + M1; T[i, 2] += k2 + M2
            if not kin[j]:
                F[j, 0] -= g0; F[j, 1] -= g1; F[j, 2] -= g2
                k0, k1, k2 = _cross(p0 - x[j, 0], p1 - x[j, 1], p2 - x[j, 2], -g0, -g1, -g2)
                T[j, 0] += k0 - M0; T[j, 1] += k1 - M1; T[j, 2] += k2 - M2
            # a pinned end pushes from outside
            if kin[i] != kin[j]:
                sg = -1.0 if kin[i] else 1.0       # the sign on the free end
                k0, k1, k2 = _cross(p0, p1, p2, sg * g0, sg * g1, sg * g2)
                ledger[0] += sg * g0 * h; ledger[1] += sg * g1 * h; ledger[2] += sg * g2 * h
                ledger[3] += (k0 + sg * M0) * h
                ledger[4] += (k1 + sg * M1) * h
                ledger[5] += (k2 + sg * M2) * h
        # walls: axis-aligned boxes that never move
        for i in range(n):
            if kin[i]:
                continue
            for w in range(nw):
                inside = True
                cp0 = min(max(x[i, 0], wlo[w, 0]), whi[w, 0])
                cp1 = min(max(x[i, 1], wlo[w, 1]), whi[w, 1])
                cp2 = min(max(x[i, 2], wlo[w, 2]), whi[w, 2])
                for a in range(3):
                    if x[i, a] < wlo[w, a] or x[i, a] > whi[w, a]:
                        inside = False
                nr0 = 0.0; nr1 = 0.0; nr2 = 0.0
                if inside:
                    best = 1e30
                    for a in range(3):
                        lo_ = x[i, a] - wlo[w, a]
                        hi_ = whi[w, a] - x[i, a]
                        if lo_ < best or hi_ < best:
                            sgn = -1.0 if lo_ < hi_ else 1.0
                            best = min(lo_, hi_)
                            nr0 = sgn if a == 0 else 0.0
                            nr1 = sgn if a == 1 else 0.0
                            nr2 = sgn if a == 2 else 0.0
                    pen = rc[i] + best
                else:
                    e0 = x[i, 0] - cp0; e1 = x[i, 1] - cp1; e2 = x[i, 2] - cp2
                    dist = np.sqrt(e0 * e0 + e1 * e1 + e2 * e2)
                    if dist >= rc[i]:
                        continue
                    dist = max(dist, 1e-12)
                    nr0 = e0 / dist; nr1 = e1 / dist; nr2 = e2 / dist
                    pen = rc[i] - dist
                r0 = -rc[i] * nr0; r1 = -rc[i] * nr1; r2 = -rc[i] * nr2
                w0, w1, w2 = _cross(om[i, 0], om[i, 1], om[i, 2], r0, r1, r2)
                vc0 = v[i, 0] + w0; vc1 = v[i, 1] + w1; vc2 = v[i, 2] + w2
                vn = vc0 * nr0 + vc1 * nr1 + vc2 * nr2
                fnm = kc[i] * pen - cc[i] * vn
                if fnm <= 0.0:
                    continue
                f0 = fnm * nr0; f1 = fnm * nr1; f2 = fnm * nr2
                vt0 = vc0 - vn * nr0; vt1 = vc1 - vn * nr1; vt2 = vc2 - vn * nr2
                svt = np.sqrt(vt0 * vt0 + vt1 * vt1 + vt2 * vt2)
                if svt > 1e-12:
                    # Coulomb friction, capped at what stops the sliding in
                    # this substep, so a hard landing cannot reverse it
                    ib = max(Ibinv[i, 0], max(Ibinv[i, 1], Ibinv[i, 2]))
                    meff = 1.0 / (1.0 / m[i] + rc[i] * rc[i] * ib)
                    ft = min(mu * fnm, 0.5 * meff * svt / h) / svt
                    f0 -= ft * vt0; f1 -= ft * vt1; f2 -= ft * vt2
                F[i, 0] += f0; F[i, 1] += f1; F[i, 2] += f2
                k0, k1, k2 = _cross(r0, r1, r2, f0, f1, f2)
                T[i, 0] += k0; T[i, 1] += k1; T[i, 2] += k2
                k0, k1, k2 = _cross(x[i, 0] + r0, x[i, 1] + r1, x[i, 2] + r2, f0, f1, f2)
                ledger[0] += f0 * h; ledger[1] += f1 * h; ledger[2] += f2 * h
                ledger[3] += k0 * h; ledger[4] += k1 * h; ledger[5] += k2 * h
        # piece against piece, for pieces not bonded to each other
        if pp:
            for k in range(tsize + 1):
                head[k] = 0
            for i in range(n):
                for a in range(3):
                    ci[i, a] = int(np.floor(x[i, a] / cell))
                hk = ((ci[i, 0] * 73856093) ^ (ci[i, 1] * 19349663)
                      ^ (ci[i, 2] * 83492791)) & (tsize - 1)
                key[i] = hk
                head[hk + 1] += 1
            for k in range(tsize):
                head[k + 1] += head[k]
            for k in range(tsize):
                fill[k] = head[k]
            for i in range(n):
                order[fill[key[i]]] = i
                fill[key[i]] += 1
            for i in range(n):
                if kin[i]:
                    continue
                for ox in range(-1, 2):
                    for oy in range(-1, 2):
                        for oz in range(-1, 2):
                            hk = (((ci[i, 0] + ox) * 73856093) ^ ((ci[i, 1] + oy) * 19349663)
                                  ^ ((ci[i, 2] + oz) * 83492791)) & (tsize - 1)
                            for e in range(head[hk], head[hk + 1]):
                                j = order[e]
                                if j <= i or kin[j]:
                                    continue
                                if (ci[j, 0] != ci[i, 0] + ox or ci[j, 1] != ci[i, 1] + oy
                                        or ci[j, 2] != ci[i, 2] + oz):
                                    continue
                                e0 = x[j, 0] - x[i, 0]; e1 = x[j, 1] - x[i, 1]; e2 = x[j, 2] - x[i, 2]
                                dist = np.sqrt(e0 * e0 + e1 * e1 + e2 * e2)
                                if dist >= rc[i] + rc[j] or dist < 1e-12:
                                    continue
                                bonded = False
                                for k in range(adj_ptr[i], adj_ptr[i + 1]):
                                    bb = adj_b[k]
                                    if alive[bb] and bi[bb] + bj[bb] - i == j:
                                        bonded = True
                                        break
                                if bonded:
                                    continue
                                nr0 = e0 / dist; nr1 = e1 / dist; nr2 = e2 / dist
                                pen = rc[i] + rc[j] - dist
                                lv = rc[i] - 0.5 * pen
                                p0 = x[i, 0] + lv * nr0; p1 = x[i, 1] + lv * nr1; p2 = x[i, 2] + lv * nr2
                                a0, a1, a2 = _cross(om[j, 0], om[j, 1], om[j, 2],
                                                    p0 - x[j, 0], p1 - x[j, 1], p2 - x[j, 2])
                                b0, b1, b2 = _cross(om[i, 0], om[i, 1], om[i, 2],
                                                    p0 - x[i, 0], p1 - x[i, 1], p2 - x[i, 2])
                                u0 = (v[j, 0] + a0) - (v[i, 0] + b0)
                                u1 = (v[j, 1] + a1) - (v[i, 1] + b1)
                                u2 = (v[j, 2] + a2) - (v[i, 2] + b2)
                                un = u0 * nr0 + u1 * nr1 + u2 * nr2
                                mr = m[i] * m[j] / (m[i] + m[j])
                                kk = min(kc[i] / m[i], kc[j] / m[j]) * mr
                                fnm = kk * pen - 2.0 * zc * np.sqrt(kk * mr) * un
                                if fnm <= 0.0:
                                    continue
                                f0 = fnm * nr0; f1 = fnm * nr1; f2 = fnm * nr2   # on j
                                ut0 = u0 - un * nr0; ut1 = u1 - un * nr1; ut2 = u2 - un * nr2
                                sut = np.sqrt(ut0 * ut0 + ut1 * ut1 + ut2 * ut2)
                                if sut > 1e-12:
                                    ft = min(mu * fnm, 0.5 * mr * sut / h) / sut
                                    f0 -= ft * ut0; f1 -= ft * ut1; f2 -= ft * ut2
                                F[j, 0] += f0; F[j, 1] += f1; F[j, 2] += f2
                                F[i, 0] -= f0; F[i, 1] -= f1; F[i, 2] -= f2
                                k0, k1, k2 = _cross(p0 - x[j, 0], p1 - x[j, 1], p2 - x[j, 2], f0, f1, f2)
                                T[j, 0] += k0; T[j, 1] += k1; T[j, 2] += k2
                                k0, k1, k2 = _cross(p0 - x[i, 0], p1 - x[i, 1], p2 - x[i, 2], -f0, -f1, -f2)
                                T[i, 0] += k0; T[i, 1] += k1; T[i, 2] += k2
        # move: velocity first, then position (symplectic Euler); the
        # turning state is the angular momentum, so a torque-free piece
        # keeps it exactly
        for i in range(n):
            if kin[i]:
                continue
            for a in range(3):
                v[i, a] += F[i, a] / m[i] * h
                ell[i, a] += T[i, a] * h
                x[i, a] += v[i, a] * h
            l0, l1, l2 = _rotT(R, i, ell[i, 0], ell[i, 1], ell[i, 2])
            w0, w1, w2 = _rot(R, i, l0 * Ibinv[i, 0], l1 * Ibinv[i, 1], l2 * Ibinv[i, 2])
            e0, e1, e2, e3 = _qexp(w0 * h, w1 * h, w2 * h)
            q[i, 0], q[i, 1], q[i, 2], q[i, 3] = _qmul(e0, e1, e2, e3,
                                                       q[i, 0], q[i, 1], q[i, 2], q[i, 3])



def qmul(a, b):
    return np.array(_qmul(*a, *b))


def qconj(a):
    return np.array((a[0], -a[1], -a[2], -a[3]))


def q_from_to(a, b):
    """The shortest turn that takes direction `a` to direction `b`
    (identity when `b` is None)."""
    if b is None:
        return np.array((1.0, 0.0, 0.0, 0.0))
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    c = np.dot(a, b)
    if c < -1 + 1e-12:                       # opposite: half a turn about any normal
        ax = np.cross(a, (0, 0, 1) if abs(a[2]) < 0.9 else (1, 0, 0))
        return np.concatenate([[0.0], ax / np.linalg.norm(ax)])
    q = np.concatenate([[1.0 + c], np.cross(a, b)])
    return q / np.linalg.norm(q)


# ── building an object ───────────────────────────────────────────────────────

class Bonded:
    """A set of pieces, the bonds between them, and the walls they meet.

    Build with `piece`, `bond` and `wall`, then `finish(stiff)`, then
    `step(dt)` as often as wanted. Everything is plain numpy arrays, so a
    scene may read or change any of it between steps (a load on a piece, a
    bond cut by an axe)."""

    def __init__(self, mu=0.4, pp=True, zeta_c=0.5):
        self._p = []          # piece dicts while building
        self._b = []          # bond dicts while building
        self._w = []          # wall boxes
        self.mu = float(mu)
        # how much of a contact's spring is damped (of critical): what an
        # impact loses at the surface. 0.5 gives a 20 cm block of glass or
        # wood a bounce near the handbook's 0.4-0.66 (b_bounce measures it).
        self.zeta_c = float(zeta_c)
        self.pp = bool(pp)    # piece against piece contacts
        self.g = GRAVITY      # 0 for a check in free space
        self.t = 0.0
        self.substeps = 0

    # ---- pieces
    def piece(self, x, m, half, rc=None, area=0.0, kin=False, amp=(0, 0, 0),
              freq=0.0, tag=0, along=None):
        """A rigid box of mass `m` (kg) centred at `x` with half-extents
        `half` (m), its first axis turned to point `along` when given. `rc`
        is its contact sphere (default: its smallest half-extent); `area` is
        the area air pushes on (m^2). A `kin` piece is pinned: it moves as
        x + amp*sin(2 pi freq t) whatever pushes on it."""
        hx, hy, hz = (float(h) for h in half)
        I = np.array((hy * hy + hz * hz, hx * hx + hz * hz, hx * hx + hy * hy)) * m / 3.0
        self._p.append(dict(x=np.asarray(x, float), m=float(m), I=I,
                            rc=float(min(half) if rc is None else rc),
                            cda=DRAG_CD * float(area), kin=bool(kin),
                            amp=np.asarray(amp, float), freq=float(freq), tag=int(tag),
                            q=q_from_to((1.0, 0.0, 0.0), along)))
        return len(self._p) - 1

    # ---- bonds
    def bond(self, i, j, mat, area, inertia, radius, length=None, grain=None,
             flaw=1.0, ball=False, bend=1.0, tag=0):
        """A beam from piece `i` to piece `j`: section `area` (m^2), second
        moment `inertia` (m^4, about either axis across the bond), edge
        distance `radius` (m), made of `mat` (a row above, or (along, across)
        for a grained material with `grain` its direction). `flaw` scales
        the strength; `bend` scales the bending and twisting stiffness. A
        `ball` joint is as stiff across as along: its spring pulls the two
        attachment points together whatever the bond's direction, which is
        what keeps a joint that turns a long way (a rope's) from making
        energy out of the turning."""
        xi, xj = self._p[i]["x"], self._p[j]["x"]
        L = float(np.linalg.norm(xj - xi)) if length is None else float(length)
        if isinstance(mat[0], tuple):
            (EL, nu, SL, zeta, _), (ET, _, ST, _, _) = mat
            nb_ = (xj - xi) / max(np.linalg.norm(xj - xi), 1e-12)
            c2 = float(np.dot(nb_, np.asarray(grain, float))) ** 2
            E, S = hankinson(EL, ET, c2), hankinson(SL, ST, c2)
        else:
            E, nu, S, zeta, _ = mat
        G = E / (2.0 * (1.0 + nu))
        J = 2.0 * inertia
        self._b.append(dict(i=i, j=j, E=E, G=G, A=float(area), I=float(inertia),
                            J=J, rb=float(radius), L=L, S=float(S) * flaw,
                            zeta=zeta, ball=bool(ball), bend=float(bend),
                            tag=int(tag)))
        return len(self._b) - 1

    def wall(self, lo, hi):
        self._w.append((np.asarray(lo, float), np.asarray(hi, float)))

    # ---- arrays
    def finish(self, stiff=1.0, strength_scale=None, safety=0.8):
        """Freeze the pieces and bonds into arrays and choose the substep.

        `stiff` softens every modulus, exactly as src/mpm's `stiff` does, so
        that the two can be compared at the same softening. With it, every
        strength is scaled by sqrt(stiff) unless `strength_scale` says
        otherwise: that keeps the SPEED AT WHICH A THING BREAKS (an impact
        makes a stress of rho*c*v, and c goes as sqrt(E)), which is the
        choice src/mpm makes. For a static load the honest choice is
        `strength_scale=1.0` and real stiffness."""
        P, B = self._p, self._b
        n, nb = len(P), len(B)
        self.n, self.nb = n, nb
        self.x = np.array([p["x"] for p in P]).reshape(n, 3)
        self.v = np.zeros((n, 3))
        self.q = np.array([p["q"] for p in P]).reshape(n, 4)
        self.ell = np.zeros((n, 3))
        self.m = np.array([p["m"] for p in P])
        self.I = np.array([p["I"] for p in P]).reshape(n, 3)
        self.Ibinv = 1.0 / self.I
        self.rc = np.array([p["rc"] for p in P])
        self.cda = np.array([p["cda"] for p in P])
        self.kin = np.array([p["kin"] for p in P], bool)
        self.x0 = self.x.copy()
        self.amp = np.array([p["amp"] for p in P]).reshape(n, 3)
        self.freq = np.array([p["freq"] for p in P])
        self.kwin = np.tile([-1e30, 1e30], (n, 1))     # when the shaking is on
        self.fext = np.zeros((n, 3))
        self.tag = np.array([p["tag"] for p in P], np.int64)
        ss = np.sqrt(stiff) if strength_scale is None else float(strength_scale)
        self.stiff, self.strength_scale = float(stiff), float(ss)
        f = lambda k: np.array([b[k] for b in B], float)
        self.bi = np.array([b["i"] for b in B], np.int64)
        self.bj = np.array([b["j"] for b in B], np.int64)
        # each bond holds at the midpoint, written in each piece's own frame,
        # and remembers how the two pieces sat turned to each other
        c = 0.5 * (self.x[self.bj] + self.x[self.bi])
        R = np.zeros((n, 3, 3))
        _q2R(self.q, R)
        self.ai = np.einsum("nji,nj->ni", R[self.bi], c - self.x[self.bi])
        self.aj = np.einsum("nji,nj->ni", R[self.bj], c - self.x[self.bj])
        self.Q0 = np.array([qmul(qconj(self.q[i]), self.q[j])
                            for i, j in zip(self.bi, self.bj)]).reshape(nb, 4)
        E, G, L = f("E") * stiff, f("G") * stiff, f("L")
        self.A, self.Ib, self.Jb, self.rb = f("A"), f("I"), f("J"), f("rb")
        self.kn = E * self.A / L
        self.ks = np.where([b["ball"] for b in B], self.kn, G * self.A / L)
        self.kb = E * self.Ib / L * f("bend")
        self.kt = G * self.Jb / L * f("bend")
        self.sig = f("S") * ss
        self.tau = self.sig.copy()
        self.btag = np.array([b["tag"] for b in B], np.int64)
        # damping: a dashpot beside each spring, `zeta` of critical for the
        # two pieces' reduced mass (and reduced inertia)
        mr = self.m[self.bi] * self.m[self.bj] / (self.m[self.bi] + self.m[self.bj])
        Imin = self.I.min(axis=1)
        Ir = Imin[self.bi] * Imin[self.bj] / (Imin[self.bi] + Imin[self.bj])
        z = f("zeta")
        self.cn = 2 * z * np.sqrt(self.kn * mr)
        self.cs = 2 * z * np.sqrt(self.ks * mr)
        self.cb = 2 * z * np.sqrt(self.kb * Ir)
        self.ct = 2 * z * np.sqrt(self.kt * Ir)
        self.alive = np.ones(nb, bool)
        self.tbreak = np.full(nb, np.nan)
        self.smax = np.zeros(nb)
        # which bonds each piece has, for the "already bonded" contact test
        deg = np.bincount(np.concatenate([self.bi, self.bj]), minlength=n)
        self.adj_ptr = np.concatenate([[0], np.cumsum(deg)]).astype(np.int64)
        fillp = self.adj_ptr[:-1].copy()
        self.adj_b = np.zeros(2 * nb, np.int64)
        for b in range(nb):
            for e in (self.bi[b], self.bj[b]):
                self.adj_b[fillp[e]] = b
                fillp[e] += 1
        self.wlo = np.array([w[0] for w in self._w]).reshape(-1, 3)
        self.whi = np.array([w[1] for w in self._w]).reshape(-1, 3)
        # CONTACT is as stiff as the piece's own stiffest bond, so an impact
        # meets the matter's own stiffness, which is what sets the impact
        # stress; a piece with no bonds gets a 300 Hz contact.
        kk = np.maximum(self.kn, self.ks)
        kmax = np.zeros(n)
        np.maximum.at(kmax, self.bi, kk)
        np.maximum.at(kmax, self.bj, kk)
        self.kc = np.where(kmax > 0, 2.0 * kmax, self.m * (2 * np.pi * 300.0) ** 2)
        self.cc = 2 * self.zeta_c * np.sqrt(self.kc * self.m)
        # THE SUBSTEP: Gershgorin's bound on the fastest mode, from each
        # bond's exact 12 x 12 stiffness at rest (both pieces' push and turn),
        # weighted by mass and by inertia about each piece's own axes. It is a
        # true upper bound, so the step it gives is stable; using the
        # smallest inertia for every turn instead made slender pieces (a
        # twig, a rope link) look forty times stiffer than they are.
        self.safety = safety
        self.retime()
        self.ledger = np.zeros(6)
        self.M = float(self.m[~self.kin].sum())
        self.P0, self.L0 = self.momentum()
        return self

    def retime(self):
        """Choose the substep again (after a scene changed a stiffness)."""
        self.h = self.safety * 2.0 / np.sqrt(max(self._fastest(), 1e-12))

    def _fastest(self):
        """Gershgorin's bound on the largest squared frequency (rad/s)^2."""
        n, nb = self.n, self.nb
        R = np.zeros((n, 3, 3))
        _q2R(self.q, R)
        free = ~self.kin
        im = np.where(free, 1.0 / np.sqrt(self.m), 0.0)
        # W = R diag(I)^-1/2 R^T: inertia-weighting of a world-frame turn
        W = np.einsum("nij,nj,nkj->nik", R, 1.0 / np.sqrt(self.I), R) * free[:, None, None]
        rows = np.zeros((n, 6))
        rows[:, :3] += (self.kc / self.m)[:, None] * free[:, None]
        if nb:
            nv = self.x[self.bj] - self.x[self.bi]
            nv /= np.linalg.norm(nv, axis=1, keepdims=True)
            nn = nv[:, :, None] * nv[:, None, :]
            I3 = np.eye(3)[None]
            Kg = self.kn[:, None, None] * nn + self.ks[:, None, None] * (I3 - nn)
            Kt = self.kb[:, None, None] * (I3 - nn) + self.kt[:, None, None] * nn
            ri = np.einsum("nij,nj->ni", R[self.bi], self.ai)
            rj = np.einsum("nij,nj->ni", R[self.bj], self.aj)

            def skew(r):
                z = np.zeros(len(r))
                return np.stack([np.stack([z, -r[:, 2], r[:, 1]], 1),
                                 np.stack([r[:, 2], z, -r[:, 0]], 1),
                                 np.stack([-r[:, 1], r[:, 0], z], 1)], 1)
            Bm = np.concatenate([-np.repeat(I3, nb, 0), skew(ri),
                                 np.repeat(I3, nb, 0), -skew(rj)], 2)      # (nb,3,12)
            Cm = np.concatenate([np.zeros((nb, 3, 3)), -np.repeat(I3, nb, 0),
                                 np.zeros((nb, 3, 3)), np.repeat(I3, nb, 0)], 2)
            K = np.einsum("nai,nab,nbj->nij", Bm, Kg, Bm) + \
                np.einsum("nai,nab,nbj->nij", Cm, Kt, Cm)                  # (nb,12,12)
            D = np.zeros((nb, 12, 12))
            for blk, (p, w) in enumerate(((self.bi, None), (self.bi, 1), (self.bj, None), (self.bj, 1))):
                sl = slice(3 * blk, 3 * blk + 3)
                D[:, sl, sl] = (np.eye(3)[None] * im[p][:, None, None]) if w is None else W[p]
            Kn = np.abs(np.einsum("nij,njk,nkl->nil", D, K, D)).sum(2)       # row sums
            np.add.at(rows, self.bi, Kn[:, :6])
            np.add.at(rows, self.bj, Kn[:, 6:])
        return float(rows.max())

    # ---- running
    def step(self, dt):
        """Advance `dt` seconds in as many substeps as `h` allows."""
        nsub = max(1, int(np.ceil(dt / self.h)))
        h = dt / nsub
        _run(nsub, h, self.t, self.x, self.v, self.q, self.ell, self.m,
             self.Ibinv, self.rc, self.cda, self.kin, self.x0, self.amp,
             self.freq, self.kwin, self.fext, self.kc, self.cc, self.bi, self.bj, self.ai,
             self.aj, self.Q0, self.kn, self.ks, self.kb, self.kt, self.cn,
             self.cs, self.cb, self.ct, self.A, self.rb, self.Ib, self.Jb,
             self.sig, self.tau, self.alive, self.tbreak, self.smax,
             self.adj_ptr, self.adj_b, self.wlo, self.whi, self.mu, self.zeta_c, self.g,
             AIR_DENS, self.pp, self.ledger)
        self.t += dt
        self.substeps += nsub
        return nsub

    # ---- reading
    def pin(self, i, amp=(0, 0, 0), freq=0.0, on=-1e30, off=1e30):
        """Hold piece `i` from now on: it stays where it is, shaken by
        amp*sin(2 pi freq t) between `on` and `off`. The totals' baseline
        starts again, since the free set changed."""
        self.kin[i] = True
        self.x0[i] = self.x[i]
        self.amp[i] = amp
        self.freq[i] = freq
        self.kwin[i] = (on, off)
        self.ell[i] = 0.0
        self.rebase()

    def unpin(self, i):
        """Let piece `i` go, with the velocity it has."""
        self.kin[i] = False
        self.rebase()

    def rebase(self):
        """Start the conservation check again from here."""
        self.ledger[:] = 0.0
        self.P0, self.L0 = self.momentum()

    def momentum(self):
        """(P, L about the origin) of the free pieces."""
        d = ~self.kin
        P = (self.m[d, None] * self.v[d]).sum(0)
        L = (np.cross(self.x[d], self.m[d, None] * self.v[d]) + self.ell[d]).sum(0)
        return P, L

    def residual(self):
        """What the totals moved by that no outside push accounts for:
        (|dP - J|, |dL - H|). Both are zero in exact arithmetic."""
        P, L = self.momentum()
        return (float(np.linalg.norm(P - self.P0 - self.ledger[:3])),
                float(np.linalg.norm(L - self.L0 - self.ledger[3:])))

    def kinetic(self):
        d = ~self.kin
        R = np.zeros((self.n, 3, 3))
        _q2R(self.q, R)
        lb = np.einsum("nji,nj->ni", R[d], self.ell[d])
        return float(0.5 * (self.m[d] * (self.v[d] ** 2).sum(1)).sum()
                     + 0.5 * (lb * lb * self.Ibinv[d]).sum())

    def fragments(self, among=None):
        """Label every piece by the piece-group its unbroken bonds hold
        together. `among` (a mask) limits it to some pieces."""
        par = np.arange(self.n)

        def root(a):
            while par[a] != a:
                par[a] = par[par[a]]
                a = par[a]
            return a
        for b in np.flatnonzero(self.alive):
            i, j = self.bi[b], self.bj[b]
            if among is not None and not (among[i] and among[j]):
                continue
            ri, rj = root(i), root(j)
            if ri != rj:
                par[max(ri, rj)] = min(ri, rj)
        return np.array([root(a) for a in range(self.n)])


def flaw_field(shape, seed=0):
    """Where an object is weak: src/mpm's `_flaw_field` (a uniform field
    smoothed twice along each axis, scaled to [0, 1]), copied so that bonds
    and particles are given flaws with the same statistics."""
    rng = np.random.default_rng(seed)
    f = rng.random(shape)
    for _ in range(2):
        for a in range(3):
            pad = np.pad(f, [(1, 1) if b == a else (0, 0) for b in range(3)],
                         mode="edge")
            lo_ = [slice(None)] * 3; lo_[a] = slice(0, -2)
            hi_ = [slice(None)] * 3; hi_[a] = slice(2, None)
            f = (f + pad[tuple(lo_)] + pad[tuple(hi_)]) / 3.0
    f -= f.min()
    return f / max(float(f.max()), 1e-12)


def flaw_factor(f):
    """src/mpm's spread of strength over a flaw value: 0.45 to 2.0 times."""
    return 0.45 + 1.55 * f


def lattice(B, lo, hi, s, mat, grain=None, seed=0, tag=0, flaw_dx=0.1,
            area=None, field=None):
    """Fill the box `lo`-`hi` with cube pieces of side `s` and bond each to
    its six neighbours. Flaws are read at each bond's midpoint from a field
    in the object's own coordinates, on `flaw_dx` cells as src/mpm's
    `add_box` reads them; pass src/mpm's own `field` and the two models
    share the same weak regions. Returns the piece indices as an
    (nx, ny, nz) array."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    nn = np.maximum(np.round((hi - lo) / s).astype(int), 1)
    dens = mat[0][4] if isinstance(mat[0], tuple) else mat[4]
    m = dens * s ** 3
    idx = np.empty(tuple(nn), np.int64)
    for a in range(nn[0]):
        for b in range(nn[1]):
            for c in range(nn[2]):
                x = lo + (np.array((a, b, c)) + 0.5) * s
                idx[a, b, c] = B.piece(x, m, (s / 2,) * 3, tag=tag,
                                       area=0.0 if area is None else area)
    if field is None:
        field = flaw_field(tuple(int(np.ceil(nn[k] * s / flaw_dx)) + 2
                                 for k in range(3)), seed)
    fl = field
    shape = fl.shape
    A, I = s * s, s ** 4 / 12.0
    for a in range(nn[0]):
        for b in range(nn[1]):
            for c in range(nn[2]):
                for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    a2, b2, c2 = a + d[0], b + d[1], c + d[2]
                    if a2 >= nn[0] or b2 >= nn[1] or c2 >= nn[2]:
                        continue
                    mid = (np.array((a + a2, b + b2, c + c2)) * 0.5 + 0.5) * s
                    ij = np.minimum((mid / flaw_dx).astype(int), np.array(shape) - 1)
                    B.bond(idx[a, b, c], idx[a2, b2, c2], mat, A, I, s / 2,
                           grain=grain, flaw=flaw_factor(fl[tuple(ij)]), tag=tag)
    return idx
