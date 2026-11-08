"""The bond solver: rigid pieces joined by beams that stretch, bend and break.

Ported from the BD prototype (tools/protos/bonds/bonds.py, where it was
measured against the continuum), with two changes for the voxel world:

- THE LATTICE IS THE COLLIDER. A piece is a rigid lump of voxel cells, and
  each of its cells (a MEMBER) touches the world as a sphere half a cell
  across, against the lattice's solid cells (a bool grid) and against the
  members of other pieces. The prototype had one sphere per piece and boxes
  for walls; one sphere per piece let a felled trunk roll like a string of
  beads.
- No shaker. A pinned piece (an ANCHOR) is a lattice cell a bonded thing is
  tied to: it never moves, and what its bonds push is an outside push.

THE MODEL is the bonded-particle model of rock mechanics (Potyondy and
Cundall 2004), a bond written in total form as a beam between two rigid
pieces: a spring along it (stretch), across it (shear), and turning ones
(bend, twist), each a beam's (E*A/L, G*A/L, E*I/L, G*J/L), with a dashpot
beside each. A bond BREAKS when the stress at the edge of its section passes
its strength: tension plus bending against the tensile strength, shear plus
twist against the shear strength. Its load then goes to its neighbours,
which is how a crack runs.

CONSERVATION BY CONSTRUCTION. A bond pushes its two pieces with equal and
opposite forces at ONE point, and equal and opposite moments, so it adds no
momentum and no angular momentum. Contacts between pieces do the same. The
integrator (symplectic Euler, with angular momentum as the stored state)
keeps both exactly in arithmetic. `ledger` adds up every outside push
(gravity, the lattice, air, loads, anchors) as it happens, so the totals
can be checked in any scene, not only in free flight.

Units are SI, z is up, as in src/mpm. A cell (i, j, k) spans
[i*dx, (i+1)*dx) on each axis, as the seam draws it.
"""

import numpy as np
from numba import njit


# ── small vector and quaternion algebra, on tuples ───────────────────────────
# Tuples, not arrays: numba keeps a tuple in registers, while a small array
# made per bond per substep is a heap allocation.

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
def q2R(q, R):
    """Rotation matrices of the unit quaternions `q` (w, x, y, z), into R."""
    for i in range(q.shape[0]):
        w, x, y, z = q[i, 0], q[i, 1], q[i, 2], q[i, 3]
        R[i, 0, 0] = 1 - 2 * (y * y + z * z)
        R[i, 0, 1] = 2 * (x * y - w * z)
        R[i, 0, 2] = 2 * (x * z + w * y)
        R[i, 1, 0] = 2 * (x * y + w * z)
        R[i, 1, 1] = 1 - 2 * (x * x + z * z)
        R[i, 1, 2] = 2 * (y * z - w * x)
        R[i, 2, 0] = 2 * (x * z - w * y)
        R[i, 2, 1] = 2 * (y * z + w * x)
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
def _solid(wall, i, j, k):
    """Is lattice cell (i, j, k) solid for the pieces? Past the world's
    sides and floor it is (the world's edge holds things in, as src/mpm's
    edge clamp does); above its top it is open sky."""
    if i < 0 or j < 0 or k < 0 or i >= wall.shape[0] or j >= wall.shape[1]:
        return True
    if k >= wall.shape[2]:
        return False
    return wall[i, j, k]


@njit(cache=True)
def members_at(x, q, mp, off, out):
    """Where each member's centre is in the world: x[p] + R[p] @ off."""
    R = np.zeros((x.shape[0], 3, 3))
    q2R(q, R)
    for k in range(mp.shape[0]):
        p = mp[k]
        r0, r1, r2 = _rot(R, p, off[k, 0], off[k, 1], off[k, 2])
        out[k, 0] = x[p, 0] + r0
        out[k, 1] = x[p, 1] + r1
        out[k, 2] = x[p, 2] + r2


# ── the solver ───────────────────────────────────────────────────────────────

@njit(cache=True)
def run(nsub, h, x, v, q, ell, m, Ibinv, cda, kin, fext,
        bi, bj, ai, aj, Q0, kn, ks, kb, kt, cn, cs, cb, ct,
        A, rb, Ib, Jb, sig, tau, alive, tbreak, smax, t0,
        adj_ptr, adj_b, mp, off, mrad, kcm, ccm, mtie,
        wall, dx, mu, zc, g, rho_air, ledger, xw, xbody, bmass, bvel, ximp, xled, pface):
    """`nsub` substeps of `h` seconds. A member does not touch the lattice
    cell it is tied to (`mtie`): the knot or stalk holds it there, and the
    cell's face would fight the knot. `pface` is how many members of each
    piece one face of it can press on a surface at once (see
    `Bonds._kc`): friction is shared out over them, so that the pushes of
    all the members that touch together never stop a slide more than once.
    `ledger` holds the outside impulse
    and the outside angular impulse about the origin, [J(3), H(3)], and is
    added to as each outside push happens. Returns how many bonds broke.

    THE OTHER SOLVER'S MATTER IS A WALL TOO (cross-solver walls, the
    user's decision, 2026-10-16). `wall` holds the cells the continuum's
    particles were in at the start of the tick, and `xw` (the world's
    shape, or a 1x1x1 stand-in when there are none) says which: the index
    of that cell in `ximp`, or -1. A push from such a cell is an outside
    push like any other, on `ledger`, and it is also added to `ximp` (the
    impulse each of those cells gave the bonds, which the seam hands back
    to the particles there, reversed) and to `xled`, [J(3), H(3)].

    SUCH A WALL WEIGHS SOMETHING, AND IT GIVES. Each of those cells is
    part of a BODY (`xbody`: the bound cluster with the most mass in it,
    or the loose particles in it), of mass `bmass`, moving at `bvel` as
    the particles' own step left it. The cell does not move in this step
    (one way per tick), but its body's velocity takes every push the
    members give it, and a member meets the cell at their relative speed:
    never pushed harder than what stops that relative speed in one
    substep, for the two masses together (a meeting with no rebound), and
    not at all once the body is moving away. A wall that could not give
    would take any push: a 40 kg pane on two posts, nudged down by a 2.7 kg
    stone landing on it, was squeezed between the posts and the stone's
    cells for the whole tick and threw the stone four times higher than it
    fell."""
    n = x.shape[0]
    nb = bi.shape[0]
    nm = mp.shape[0]
    F = np.zeros((n, 3))
    T = np.zeros((n, 3))
    R = np.zeros((n, 3, 3))
    om = np.zeros((n, 3))
    xm = np.zeros((nm, 3))
    broke = 0
    # the contact grid for member against member: a hash of cells as wide as
    # the widest member, rebuilt every substep
    rmax = 1e-9
    for k in range(nm):
        if mrad[k] > rmax:
            rmax = mrad[k]
    cell = 2.0 * rmax
    tsize = 1
    while tsize < 2 * nm:
        tsize *= 2
    head = np.zeros(tsize + 1, np.int64)
    fill = np.zeros(tsize, np.int64)
    order = np.empty(nm, np.int64)
    key = np.empty(nm, np.int64)
    ci = np.empty((nm, 3), np.int64)
    # up to 27 lattice contacts per member, merged by normal
    cn_ = np.zeros((27, 3))
    cpen = np.zeros(27)
    cxc = np.full(27, -1, np.int64)      # the continuum cell behind each, or -1
    cross = xw.shape[0] == wall.shape[0] and xw.shape[1] == wall.shape[1] \
        and xw.shape[2] == wall.shape[2]
    # which pieces are one member (one cell): see "A CELL DOES NOT ROLL"
    nmem = np.zeros(n, np.int64)
    for k in range(nm):
        nmem[mp[k]] += 1
    for s in range(nsub):
        t = t0 + s * h
        q2R(q, R)
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
        for k in range(nm):
            p = mp[k]
            r0, r1, r2 = _rot(R, p, off[k, 0], off[k, 1], off[k, 2])
            xm[k, 0] = x[p, 0] + r0
            xm[k, 1] = x[p, 1] + r1
            xm[k, 2] = x[p, 2] + r2
        # gravity, loads and air drag: outside pushes on the free pieces
        for i in range(n):
            if kin[i]:
                continue
            sp = np.sqrt(v[i, 0] ** 2 + v[i, 1] ** 2 + v[i, 2] ** 2)
            dr = 0.5 * rho_air * cda[i] * sp
            f0 = fext[i, 0] - dr * v[i, 0]
            f1 = fext[i, 1] - dr * v[i, 1]
            f2 = fext[i, 2] - dr * v[i, 2] - m[i] * g
            F[i, 0] += f0
            F[i, 1] += f1
            F[i, 2] += f2
            t0_, t1_, t2_ = _cross(x[i, 0], x[i, 1], x[i, 2], f0, f1, f2)
            ledger[0] += f0 * h
            ledger[1] += f1 * h
            ledger[2] += f2 * h
            ledger[3] += t0_ * h
            ledger[4] += t1_ * h
            ledger[5] += t2_ * h
        # bonds
        for b in range(nb):
            if not alive[b]:
                continue
            i = bi[b]
            j = bj[b]
            ri0, ri1, ri2 = _rot(R, i, ai[b, 0], ai[b, 1], ai[b, 2])
            rj0, rj1, rj2 = _rot(R, j, aj[b, 0], aj[b, 1], aj[b, 2])
            d0 = (x[j, 0] + rj0) - (x[i, 0] + ri0)
            d1 = (x[j, 1] + rj1) - (x[i, 1] + ri1)
            d2 = (x[j, 2] + rj2) - (x[i, 2] + ri2)
            n0 = x[j, 0] - x[i, 0]
            n1 = x[j, 1] - x[i, 1]
            n2 = x[j, 2] - x[i, 2]
            ln = max(np.sqrt(n0 * n0 + n1 * n1 + n2 * n2), 1e-12)
            n0 /= ln
            n1 /= ln
            n2 /= ln
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
            s0 = d0 - dn * n0
            s1 = d1 - dn * n1
            s2 = d2 - dn * n2
            fs_el = ks[b] * np.sqrt(s0 * s0 + s1 * s1 + s2 * s2)
            us0 = u0 - un * n0
            us1 = u1 - un * n1
            us2 = u2 - un * n2
            # the force on i (and minus it on j)
            g0 = fn * n0 + ks[b] * s0 + cs[b] * us0
            g1 = fn * n1 + ks[b] * s1 + cs[b] * us1
            g2 = fn * n2 + ks[b] * s2 + cs[b] * us2
            # how far j has turned from where it sat on i, as a rotation
            # vector in the world frame: bend about the axes across the
            # bond, twist about the bond itself
            c0, c1, c2, c3 = _qmul(q[i, 0], -q[i, 1], -q[i, 2], -q[i, 3],
                                   q[j, 0], q[j, 1], q[j, 2], q[j, 3])
            c0, c1, c2, c3 = _qmul(c0, c1, c2, c3,
                                   Q0[b, 0], -Q0[b, 1], -Q0[b, 2], -Q0[b, 3])
            e0, e1, e2 = _qlog2(c0, c1, c2, c3)
            th0, th1, th2 = _rot(R, i, e0, e1, e2)
            tt = th0 * n0 + th1 * n1 + th2 * n2
            hb0 = th0 - tt * n0
            hb1 = th1 - tt * n1
            hb2 = th2 - tt * n2
            dw0 = om[j, 0] - om[i, 0]
            dw1 = om[j, 1] - om[i, 1]
            dw2 = om[j, 2] - om[i, 2]
            dwt = dw0 * n0 + dw1 * n1 + dw2 * n2
            M0 = kb[b] * hb0 + kt[b] * tt * n0 + cb[b] * (dw0 - dwt * n0) + ct[b] * dwt * n0
            M1 = kb[b] * hb1 + kt[b] * tt * n1 + cb[b] * (dw1 - dwt * n1) + ct[b] * dwt * n1
            M2 = kb[b] * hb2 + kt[b] * tt * n2 + cb[b] * (dw2 - dwt * n2) + ct[b] * dwt * n2
            # the stress at the edge of the section, against its strength
            mb_ = kb[b] * np.sqrt(hb0 * hb0 + hb1 * hb1 + hb2 * hb2)
            mt = kt[b] * abs(tt)
            st = fn_el / A[b] + mb_ * rb[b] / Ib[b]
            ts = fs_el / A[b] + mt * rb[b] / Jb[b]
            r = max(st / sig[b], ts / tau[b])
            if r > smax[b]:
                smax[b] = r
            if r > 1.0:
                alive[b] = False
                tbreak[b] = t
                broke += 1
                continue
            # both pushes act at ONE point, the middle of the two attachment
            # points, so the pair adds no turning to the whole
            p0 = x[i, 0] + ri0 + 0.5 * d0
            p1 = x[i, 1] + ri1 + 0.5 * d1
            p2 = x[i, 2] + ri2 + 0.5 * d2
            if not kin[i]:
                F[i, 0] += g0
                F[i, 1] += g1
                F[i, 2] += g2
                k0, k1, k2 = _cross(p0 - x[i, 0], p1 - x[i, 1], p2 - x[i, 2], g0, g1, g2)
                T[i, 0] += k0 + M0
                T[i, 1] += k1 + M1
                T[i, 2] += k2 + M2
            if not kin[j]:
                F[j, 0] -= g0
                F[j, 1] -= g1
                F[j, 2] -= g2
                k0, k1, k2 = _cross(p0 - x[j, 0], p1 - x[j, 1], p2 - x[j, 2], -g0, -g1, -g2)
                T[j, 0] += k0 - M0
                T[j, 1] += k1 - M1
                T[j, 2] += k2 - M2
            # an anchored end pushes from outside
            if kin[i] != kin[j]:
                sg = -1.0 if kin[i] else 1.0       # the sign on the free end
                k0, k1, k2 = _cross(p0, p1, p2, sg * g0, sg * g1, sg * g2)
                ledger[0] += sg * g0 * h
                ledger[1] += sg * g1 * h
                ledger[2] += sg * g2 * h
                ledger[3] += (k0 + sg * M0) * h
                ledger[4] += (k1 + sg * M1) * h
                ledger[5] += (k2 + sg * M2) * h
        # THE LATTICE: each member against the solid cells it reaches. Two
        # cells of one flat face give the same normal, and only the deeper
        # of them pushes, so a member resting across a seam between two
        # floor cells is not pushed twice.
        for k in range(nm):
            p = mp[k]
            if kin[p]:
                continue
            rad = mrad[k]
            c0 = xm[k, 0]
            c1 = xm[k, 1]
            c2 = xm[k, 2]
            nc = 0
            lo0 = int(np.floor((c0 - rad) / dx))
            lo1 = int(np.floor((c1 - rad) / dx))
            lo2 = int(np.floor((c2 - rad) / dx))
            hi0 = int(np.floor((c0 + rad) / dx))
            hi1 = int(np.floor((c1 + rad) / dx))
            hi2 = int(np.floor((c2 + rad) / dx))
            for ii in range(lo0, hi0 + 1):
                for jj in range(lo1, hi1 + 1):
                    for kk in range(lo2, hi2 + 1):
                        if not _solid(wall, ii, jj, kk):
                            continue
                        if ii == mtie[k, 0] and jj == mtie[k, 1] and kk == mtie[k, 2]:
                            continue        # the cell it is tied to holds it
                        b0 = ii * dx
                        b1 = jj * dx
                        b2 = kk * dx
                        inside = (b0 <= c0 <= b0 + dx and b1 <= c1 <= b1 + dx
                                  and b2 <= c2 <= b2 + dx)
                        if inside:
                            # out the nearest face that is not itself solid
                            best = 1e30
                            nr0 = 0.0
                            nr1 = 0.0
                            nr2 = 1.0
                            for a in range(3):
                                ca = c0 if a == 0 else (c1 if a == 1 else c2)
                                ba = b0 if a == 0 else (b1 if a == 1 else b2)
                                for sgn in (-1, 1):
                                    d_ = (ca - ba) if sgn < 0 else (ba + dx - ca)
                                    o0 = ii + (sgn if a == 0 else 0)
                                    o1 = jj + (sgn if a == 1 else 0)
                                    o2 = kk + (sgn if a == 2 else 0)
                                    if _solid(wall, o0, o1, o2):
                                        continue
                                    if d_ < best:
                                        best = d_
                                        nr0 = float(sgn) if a == 0 else 0.0
                                        nr1 = float(sgn) if a == 1 else 0.0
                                        nr2 = float(sgn) if a == 2 else 0.0
                            if best > 1e29:
                                continue        # buried: nowhere to go
                            pen = rad + best
                        else:
                            e0 = c0 - min(max(c0, b0), b0 + dx)
                            e1 = c1 - min(max(c1, b1), b1 + dx)
                            e2 = c2 - min(max(c2, b2), b2 + dx)
                            dist = np.sqrt(e0 * e0 + e1 * e1 + e2 * e2)
                            if dist >= rad:
                                continue
                            dist = max(dist, 1e-12)
                            nr0 = e0 / dist
                            nr1 = e1 / dist
                            nr2 = e2 / dist
                            pen = rad - dist
                        xc = -1
                        if cross and kk < wall.shape[2] and ii >= 0 and jj >= 0 \
                                and kk >= 0 and ii < wall.shape[0] and jj < wall.shape[1]:
                            xc = xw[ii, jj, kk]
                        # merge with a contact of the same normal
                        same = -1
                        for c in range(nc):
                            if (cn_[c, 0] * nr0 + cn_[c, 1] * nr1
                                    + cn_[c, 2] * nr2) > 0.999:
                                same = c
                                break
                        if same < 0:
                            if nc < 27:
                                cn_[nc, 0] = nr0
                                cn_[nc, 1] = nr1
                                cn_[nc, 2] = nr2
                                cpen[nc] = pen
                                cxc[nc] = xc
                                nc += 1
                        elif pen > cpen[same]:
                            cpen[same] = pen
                            cxc[same] = xc
            fsum = 0.0
            for c in range(nc):
                nr0 = cn_[c, 0]
                nr1 = cn_[c, 1]
                nr2 = cn_[c, 2]
                pen = cpen[c]
                # the contact point, from the piece's centre
                r0 = c0 - rad * nr0 - x[p, 0]
                r1 = c1 - rad * nr1 - x[p, 1]
                r2 = c2 - rad * nr2 - x[p, 2]
                w0, w1, w2 = _cross(om[p, 0], om[p, 1], om[p, 2], r0, r1, r2)
                vc0 = v[p, 0] + w0
                vc1 = v[p, 1] + w1
                vc2 = v[p, 2] + w2
                xc = cxc[c]
                xb = -1
                if xc >= 0:                     # the wall is the continuum's
                    xb = xbody[xc]
                    vc0 -= bvel[xb, 0]
                    vc1 -= bvel[xb, 1]
                    vc2 -= bvel[xb, 2]
                vn = vc0 * nr0 + vc1 * nr1 + vc2 * nr2
                fnm = kcm[k] * pen - ccm[k] * vn
                if xc >= 0 and fnm > 0.0:
                    ib = max(Ibinv[p, 0], max(Ibinv[p, 1], Ibinv[p, 2]))
                    meff = 1.0 / (1.0 / m[p] + (r0 * r0 + r1 * r1 + r2 * r2) * ib)
                    mr = meff * bmass[xb] / (meff + bmass[xb])
                    fnm = min(fnm, mr * max(-vn, 0.0) / h)
                if fnm <= 0.0:
                    continue
                fsum += fnm
                f0 = fnm * nr0
                f1 = fnm * nr1
                f2 = fnm * nr2
                vt0 = vc0 - vn * nr0
                vt1 = vc1 - vn * nr1
                vt2 = vc2 - vn * nr2
                svt = np.sqrt(vt0 * vt0 + vt1 * vt1 + vt2 * vt2)
                if svt > 1e-12:
                    # Coulomb friction, capped at what stops the sliding in
                    # this substep, so a hard landing cannot reverse it. The
                    # cap is the PIECE's, shared by the members of a face
                    # that press together: a plank landing flat on stone
                    # touches with sixteen members at once, and each one
                    # alone stopping the slide reversed it sixteen times over.
                    ib = max(Ibinv[p, 0], max(Ibinv[p, 1], Ibinv[p, 2]))
                    rr = r0 * r0 + r1 * r1 + r2 * r2
                    meff = 1.0 / (1.0 / m[p] + rr * ib)
                    ft = min(mu * fnm, 0.5 * meff * svt / (h * pface[p])) / svt
                    f0 -= ft * vt0
                    f1 -= ft * vt1
                    f2 -= ft * vt2
                F[p, 0] += f0
                F[p, 1] += f1
                F[p, 2] += f2
                k0, k1, k2 = _cross(r0, r1, r2, f0, f1, f2)
                T[p, 0] += k0
                T[p, 1] += k1
                T[p, 2] += k2
                k0, k1, k2 = _cross(x[p, 0] + r0, x[p, 1] + r1, x[p, 2] + r2, f0, f1, f2)
                ledger[0] += f0 * h
                ledger[1] += f1 * h
                ledger[2] += f2 * h
                ledger[3] += k0 * h
                ledger[4] += k1 * h
                ledger[5] += k2 * h
                if xc >= 0:
                    bvel[xb, 0] -= f0 * h / bmass[xb]
                    bvel[xb, 1] -= f1 * h / bmass[xb]
                    bvel[xb, 2] -= f2 * h / bmass[xb]
                    ximp[xc, 0] += f0 * h
                    ximp[xc, 1] += f1 * h
                    ximp[xc, 2] += f2 * h
                    xled[0] += f0 * h
                    xled[1] += f1 * h
                    xled[2] += f2 * h
                    xled[3] += k0 * h
                    xled[4] += k1 * h
                    xled[5] += k2 * h
            # A CELL DOES NOT ROLL. A piece of one member meets the lattice
            # as a sphere, and a sphere rolls: friction stops its contact
            # point and it rolls on for ever (sprays shed by a felled crown
            # rolled across the floor at 0.1 m/s for five seconds). It is a
            # cube. To roll, a cube resting on a face must lift its centre
            # over an edge, which takes a turning push of its weight times
            # half its width; less than that, and it stops turning. So a
            # one-member piece pressed on the lattice feels a couple against
            # its spin, up to that (its load, but no more than its weight: a
            # blow is the sphere's contact, as before), and at most what
            # stops the spin in half a substep. A couple from outside:
            # booked on the ledger.
            if nmem[p] == 1 and fsum > 0.0:
                so = np.sqrt(om[p, 0] ** 2 + om[p, 1] ** 2 + om[p, 2] ** 2)
                if so > 1e-12:
                    ib = max(Ibinv[p, 0], max(Ibinv[p, 1], Ibinv[p, 2]))
                    tr = min(min(fsum, m[p] * g) * rad, 0.5 * so / (h * ib)) / so
                    for a in range(3):
                        T[p, a] -= tr * om[p, a]
                        ledger[3 + a] -= tr * om[p, a] * h
        # member against member, for pieces not bonded to each other
        for kk in range(tsize + 1):
            head[kk] = 0
        for k in range(nm):
            for a in range(3):
                ci[k, a] = int(np.floor(xm[k, a] / cell))
            hk = ((ci[k, 0] * 73856093) ^ (ci[k, 1] * 19349663)
                  ^ (ci[k, 2] * 83492791)) & (tsize - 1)
            key[k] = hk
            head[hk + 1] += 1
        for kk in range(tsize):
            head[kk + 1] += head[kk]
        for kk in range(tsize):
            fill[kk] = head[kk]
        for k in range(nm):
            order[fill[key[k]]] = k
            fill[key[k]] += 1
        for k in range(nm):
            i = mp[k]
            if kin[i]:
                continue
            for ox in range(-1, 2):
                for oy in range(-1, 2):
                    for oz in range(-1, 2):
                        hk = (((ci[k, 0] + ox) * 73856093) ^ ((ci[k, 1] + oy) * 19349663)
                              ^ ((ci[k, 2] + oz) * 83492791)) & (tsize - 1)
                        for e in range(head[hk], head[hk + 1]):
                            l_ = order[e]
                            if l_ <= k:
                                continue
                            j = mp[l_]
                            if j == i or kin[j]:
                                continue
                            if (ci[l_, 0] != ci[k, 0] + ox or ci[l_, 1] != ci[k, 1] + oy
                                    or ci[l_, 2] != ci[k, 2] + oz):
                                continue
                            e0 = xm[l_, 0] - xm[k, 0]
                            e1 = xm[l_, 1] - xm[k, 1]
                            e2 = xm[l_, 2] - xm[k, 2]
                            dist = np.sqrt(e0 * e0 + e1 * e1 + e2 * e2)
                            if dist >= mrad[k] + mrad[l_] or dist < 1e-12:
                                continue
                            bonded = False
                            for kb_ in range(adj_ptr[i], adj_ptr[i + 1]):
                                bb = adj_b[kb_]
                                if alive[bb] and bi[bb] + bj[bb] - i == j:
                                    bonded = True
                                    break
                            if bonded:
                                continue
                            nr0 = e0 / dist
                            nr1 = e1 / dist
                            nr2 = e2 / dist
                            pen = mrad[k] + mrad[l_] - dist
                            lv = mrad[k] - 0.5 * pen
                            p0 = xm[k, 0] + lv * nr0
                            p1 = xm[k, 1] + lv * nr1
                            p2 = xm[k, 2] + lv * nr2
                            a0, a1, a2 = _cross(om[j, 0], om[j, 1], om[j, 2],
                                                p0 - x[j, 0], p1 - x[j, 1], p2 - x[j, 2])
                            b0, b1, b2 = _cross(om[i, 0], om[i, 1], om[i, 2],
                                                p0 - x[i, 0], p1 - x[i, 1], p2 - x[i, 2])
                            u0 = (v[j, 0] + a0) - (v[i, 0] + b0)
                            u1 = (v[j, 1] + a1) - (v[i, 1] + b1)
                            u2 = (v[j, 2] + a2) - (v[i, 2] + b2)
                            un = u0 * nr0 + u1 * nr1 + u2 * nr2
                            mr = m[i] * m[j] / (m[i] + m[j])
                            kk_ = min(kcm[k] / m[i], kcm[l_] / m[j]) * mr
                            fnm = kk_ * pen - 2.0 * zc * np.sqrt(kk_ * mr) * un
                            if fnm <= 0.0:
                                continue
                            f0 = fnm * nr0          # on j
                            f1 = fnm * nr1
                            f2 = fnm * nr2
                            ut0 = u0 - un * nr0
                            ut1 = u1 - un * nr1
                            ut2 = u2 - un * nr2
                            sut = np.sqrt(ut0 * ut0 + ut1 * ut1 + ut2 * ut2)
                            if sut > 1e-12:
                                ft = min(mu * fnm, 0.5 * mr * sut
                                         / (h * min(pface[i], pface[j]))) / sut
                                f0 -= ft * ut0
                                f1 -= ft * ut1
                                f2 -= ft * ut2
                            F[j, 0] += f0
                            F[j, 1] += f1
                            F[j, 2] += f2
                            F[i, 0] -= f0
                            F[i, 1] -= f1
                            F[i, 2] -= f2
                            k0, k1, k2 = _cross(p0 - x[j, 0], p1 - x[j, 1], p2 - x[j, 2], f0, f1, f2)
                            T[j, 0] += k0
                            T[j, 1] += k1
                            T[j, 2] += k2
                            k0, k1, k2 = _cross(p0 - x[i, 0], p1 - x[i, 1], p2 - x[i, 2], -f0, -f1, -f2)
                            T[i, 0] += k0
                            T[i, 1] += k1
                            T[i, 2] += k2
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
    return broke
