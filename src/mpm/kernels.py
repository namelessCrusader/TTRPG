"""Optional numba kernels for the particle solver: each one must agree with the numpy path in core.py bit for bit."""
import numpy as np

# OPTIONAL, exactly as the voxel sim's own kernels are (src/voxel/kernels.py),
# and copied rather than imported because src/mpm imports nothing from
# src/voxel. The numpy code in core.py stays the DEFINITION of the solver;
# these loops only do the same arithmetic in the same order without building
# 27 temporaries per quantity per substep. `MPM.compiled = False` runs the
# definition, and `MPM.fused = False` runs the compiled loops one stage at a
# time instead of the whole substep in one call (`substep`, at the end);
# tests/test_mpm.py holds all three equal after every substep and
# tests/test_mpm_seam.py after every tick of a World. Without numba installed
# nothing changes except the speed.
try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:                        # pragma: no cover - depends on install
    HAVE_NUMBA = False

    def njit(*args, **kwargs):           # a decorator that does nothing
        def wrap(fn):
            return fn
        return wrap if not args or not callable(args[0]) else args[0]


@njit(cache=True)
def _stencil(base, fx, lo, top, dx, ni, dd):
    """Per particle, axis and stencil offset i: the node it reaches in the
    window (clipped to the world as the numpy loop clips it, less `lo`)
    into `ni`, and (i - fx) dx into `dd`. Worked out once here instead of
    27 times in P2G and G2P; the same numbers either way."""
    for p in range(base.shape[0]):
        for a in range(3):
            for i in range(3):
                ni[p, a, i] = min(max(base[p, a] + i, 0), top[a]) - lo[a]
                dd[p, a, i] = (i - fx[p, a]) * dx


@njit(cache=True)
def p2g(base, fx, w, m, v, affine, blk, lo, top, wy, wz, dx, gm, gp, gb):
    """Particles to grid over the window, as core.py's numpy loop does it.

    THE ORDER OF THE ADDITIONS IS THE RESULT. For each of the 27 offsets numpy
    sums every node's contributions from zero in particle order (that is what
    `np.bincount` does), and only then adds that partial into the grid. So
    this does the same: a scratch sum per node (`tm`, `tp`, `tb`), then one
    add into the grid and the scratch reset to zero. The affine product is
    summed in numpy's order too, (a0 d0 + a2 d2) + a1 d1; see `MPM._p2g`."""
    N = base.shape[0]
    tm = np.zeros(gm.shape[0])
    tp = np.zeros((gm.shape[0], 3))
    tb = np.zeros((gm.shape[0], 3))
    node = np.empty(N, np.int64)
    ni = np.empty((N, 3, 3), np.int64)
    dd = np.empty((N, 3, 3))
    _stencil(base, fx, lo, top, dx, ni, dd)
    mv = np.empty((N, 3))
    for p in range(N):
        for a in range(3):
            mv[p, a] = m[p] * v[p, a]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for p in range(N):
                    f = (ni[p, 0, i] * wy + ni[p, 1, j]) * wz + ni[p, 2, k]
                    node[p] = f
                    wt = w[i, p, 0] * w[j, p, 1] * w[k, p, 2]
                    d0 = dd[p, 0, i]
                    d1 = dd[p, 1, j]
                    d2 = dd[p, 2, k]
                    wm = wt * m[p]
                    tm[f] += wm
                    for a in range(3):
                        af = wt * ((affine[p, a, 0] * d0 + affine[p, a, 2] * d2)
                                   + affine[p, a, 1] * d1)
                        tp[f, a] += wt * mv[p, a] + af
                        tb[f, a] += wm * blk[p, a]
                for p in range(N):
                    f = node[p]
                    gm[f] += tm[f]
                    tm[f] = 0.0
                    for a in range(3):
                        gp[f, a] += tp[f, a]
                        tp[f, a] = 0.0
                        gb[f, a] += tb[f, a]
                        tb[f, a] = 0.0


@njit(cache=True)
def g2p(base, fx, w, gv, lo, top, wy, wz, dx, cC, newv, newC):
    """Grid to particles: each particle gathers its 27 nodes in the numpy
    loop's offset order, so its sums are added in the same order."""
    N = base.shape[0]
    ni = np.empty((N, 3, 3), np.int64)
    dd = np.empty((N, 3, 3))
    _stencil(base, fx, lo, top, dx, ni, dd)
    for p in range(N):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    f = (ni[p, 0, i] * wy + ni[p, 1, j]) * wz + ni[p, 2, k]
                    wt = w[i, p, 0] * w[j, p, 1] * w[k, p, 2]
                    d0 = dd[p, 0, i]
                    d1 = dd[p, 1, j]
                    d2 = dd[p, 2, k]
                    s = cC * wt
                    for a in range(3):
                        g = gv[f, a]
                        newv[p, a] += wt * g
                        newC[p, a, 0] += s * (g * d0)
                        newC[p, a, 1] += s * (g * d1)
                        newC[p, a, 2] += s * (g * d2)


@njit(cache=True)
def contact_faces(x, wall, inv_dx, dx, top, blk):
    """`MPM._contact_faces`: for each particle not inside a wall, which side
    a wall lies on within a tenth of a cell, axis by axis, in the numpy
    loop's order (so where both sides touch, the +1 written last wins)."""
    N = x.shape[0]
    for p in range(N):
        c0 = min(max(int(x[p, 0] * inv_dx), 0), top[0])
        c1 = min(max(int(x[p, 1] * inv_dx), 0), top[1])
        c2 = min(max(int(x[p, 2] * inv_dx), 0), top[2])
        if wall[c0, c1, c2]:
            continue
        for a in range(3):
            for sgn in (-1.0, 1.0):
                q0 = x[p, 0]
                q1 = x[p, 1]
                q2 = x[p, 2]
                step = sgn * 0.1 * dx
                if a == 0:
                    q0 += step
                elif a == 1:
                    q1 += step
                else:
                    q2 += step
                d0 = min(max(int(q0 * inv_dx), 0), top[0])
                d1 = min(max(int(q1 * inv_dx), 0), top[1])
                d2 = min(max(int(q2 * inv_dx), 0), top[2])
                if wall[d0, d1, d2]:
                    blk[p, a] = sgn


@njit(cache=True)
def sample(field, pts, inv_dx, gtop, top, lo, wtop, out):
    """`MPM._sample` into `out` (one column per component of `field`, which
    is always 4-D here: a scalar field comes in as field[..., None]). The
    clip is numpy's, max then min, and each sum starts from zero and takes
    the eight corners in the numpy loop's order."""
    N = pts.shape[0]
    K = field.shape[3]
    i0 = np.empty(3, np.int64)
    i1 = np.empty(3, np.int64)
    fr = np.empty(3)
    for p in range(N):
        for a in range(3):
            g = pts[p, a] * inv_dx - 0.5
            g = g if g > 0.0 else 0.0
            g = g if g < gtop[a] else gtop[a]
            i = int(g)
            fr[a] = g - i
            j = min(i + 1, top[a])
            i0[a] = min(max(i - lo[a], 0), wtop[a])
            i1[a] = min(max(j - lo[a], 0), wtop[a])
        for k in range(K):
            out[p, k] = 0.0
        for a in range(2):
            wa = fr[0] if a else 1.0 - fr[0]
            ia = i1[0] if a else i0[0]
            for b in range(2):
                wb = fr[1] if b else 1.0 - fr[1]
                ib = i1[1] if b else i0[1]
                for c in range(2):
                    wc = fr[2] if c else 1.0 - fr[2]
                    ic = i1[2] if c else i0[2]
                    wt = wa * wb * wc
                    for k in range(K):
                        out[p, k] = out[p, k] + wt * field[ia, ib, ic, k]


@njit(cache=True)
def _dt1d(f, d, v, z):
    """`MPM._dt1d`, line for line, writing into `d` with scratch `v` and `z`."""
    n = f.shape[0]
    k = 0
    v[0] = 0
    z[0] = -np.inf
    z[1] = np.inf
    for q in range(1, n):
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        while s <= z[k]:
            k -= 1
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = (q - v[k]) ** 2 + f[v[k]]


@njit(cache=True)
def edt2(seed):
    """`MPM._edt2`: the squared distance in cells from every cell to the
    nearest `seed`, one axis after another. Every value is a sum of squared
    whole numbers, so it is exact, and equal to the numpy path's."""
    nx, ny, nz = seed.shape
    big = 1e12
    f = np.empty((nx, ny, nz))
    for a in range(nx):
        for b in range(ny):
            for c in range(nz):
                f[a, b, c] = 0.0 if seed[a, b, c] else big
    nmax = max(nx, max(ny, nz))
    buf = np.empty(nmax)
    out = np.empty(nmax)
    v = np.zeros(nmax, np.int64)
    z = np.empty(nmax + 1)
    for b in range(ny):                      # axis 0
        for c in range(nz):
            for a in range(nx):
                buf[a] = f[a, b, c]
            _dt1d(buf[:nx], out[:nx], v[:nx], z[:nx + 1])
            for a in range(nx):
                f[a, b, c] = out[a]
    for a in range(nx):                      # axis 1
        for c in range(nz):
            for b in range(ny):
                buf[b] = f[a, b, c]
            _dt1d(buf[:ny], out[:ny], v[:ny], z[:ny + 1])
            for b in range(ny):
                f[a, b, c] = out[b]
    for a in range(nx):                      # axis 2
        for b in range(ny):
            for c in range(nz):
                buf[c] = f[a, b, c]
            _dt1d(buf[:nz], out[:nz], v[:nz], z[:nz + 1])
            for c in range(nz):
                f[a, b, c] = out[c]
    return f


# ------------------------------------------------------ THE FUSED SUBSTEP (P10)
#
# Everything `MPM.substep` does for one island, in one call: the stencil
# weights, which wall faces each particle touches, the stress (with the
# fixed-sweep SVD and eigenvalue of core.py), P2G, the grid, G2P, the move,
# the hold on each cluster's shape, contact with the walls, and the clip at
# the world's edge. The numpy path in core.py stays the DEFINITION; every
# line below does the same operations in the same order on one particle or
# one node at a time, so the answer is the same bits (tests/test_mpm.py
# compares the two after every substep). Two things keep that true: numba
# does not fuse a multiply and an add into one rounding unless it is told
# to (fastmath is never on here), and every numpy max, min and clip is
# written with numpy's own rule for ties and NaN (`_mx`, `_clip`).
#
# `error_model="numpy"`: a division by zero gives inf or nan, as it does in
# numpy, instead of numba's default of raising ZeroDivisionError. That is
# the numpy path's own meaning, and it also takes the check for a zero out
# of every division, which is what lets the compiler run a loop over
# particles four at a time (measured: the Jacobi sweeps 17 us -> 6 us for
# 56 particles, the same bits). Each lane still rounds every operation
# exactly as one particle alone would.


@njit(cache=True, error_model="numpy")
def _mx(a, b):
    """np.maximum(a, b) for two floats, ties and NaN included."""
    return a if (a >= b or a != a) else b


@njit(cache=True, error_model="numpy")
def _clip(x, lo, hi):
    """np.clip(x, lo, hi) for floats: max with lo, then min with hi."""
    y = x if (x != x or x > lo) else lo
    return y if (y != y or y < hi) else hi


@njit(cache=True, error_model="numpy")
def weights(x, inv_dx, base, fx, w):
    """`MPM._weights` into `base`, `fx` and `w`."""
    for p in range(x.shape[0]):
        for a in range(3):
            px = x[p, a] * inv_dx
            b = np.floor(px - 0.5)
            base[p, a] = np.int64(b)
            f = px - np.float64(base[p, a])
            fx[p, a] = f
            t0 = 1.5 - f
            t1 = f - 1.0
            t2 = f - 0.5
            w[0, p, a] = 0.5 * (t0 * t0)
            w[1, p, a] = 0.75 - t1 * t1
            w[2, p, a] = 0.5 * (t2 * t2)


@njit(cache=True, error_model="numpy")
def window_holds(wall, cmin, cmax, pad_lo, pad_hi, lo, hi, sub):
    """`MPM._sdf_window`'s test of a kept window [lo, hi), cut with the
    walls `sub`: does it reach `pad_lo` cells below the lowest particle cell
    and `pad_hi` above the highest (both clipped to the world), and are its
    walls the world's walls still? Row by row, stopping at the first row
    that changed."""
    for a in range(3):
        if max(cmin[a] - pad_lo, 0) < lo[a] or min(cmax[a] + pad_hi, wall.shape[a]) > hi[a]:
            return False
    for i in range(hi[0] - lo[0]):
        for j in range(hi[1] - lo[1]):
            row = wall[lo[0] + i, lo[1] + j, lo[2]:hi[2]]
            diff = False
            for k in range(hi[2] - lo[2]):
                diff |= row[k] != sub[i, j, k]
            if diff:
                return False
    return True


@njit(cache=True, error_model="numpy")
def bounds(x, inv_dx, top):
    """The lowest and highest stencil base and cell over all particles:
    (base min, base max, cell min, cell max), each int64[3], and how many
    nodes the box of their stencils holds (as `MPM._islands` counts it)."""
    bmin = np.empty(3, np.int64)
    bmax = np.empty(3, np.int64)
    cmin = np.empty(3, np.int64)
    cmax = np.empty(3, np.int64)
    for a in range(3):
        bmin[a] = np.iinfo(np.int64).max
        bmax[a] = np.iinfo(np.int64).min
        cmin[a] = np.iinfo(np.int64).max
        cmax[a] = np.iinfo(np.int64).min
    for p in range(x.shape[0]):
        for a in range(3):
            px = x[p, a] * inv_dx
            b = np.int64(np.floor(px - 0.5))
            c = np.int64(np.floor(px))
            bmin[a] = min(bmin[a], b)
            bmax[a] = max(bmax[a], b)
            cmin[a] = min(cmin[a], c)
            cmax[a] = max(cmax[a], c)
    nodes = 1
    for a in range(3):
        nodes *= min(bmax[a] + 2, top[a]) - max(bmin[a], 0) + 1
    return bmin, bmax, cmin, cmax, nodes


@njit(cache=True, error_model="numpy")
def _sweeps(B, V, sweeps):
    """The Jacobi sweeps of `core._svd3` on every particle at once: B and V
    hold one particle per column, entry (i, j) of its matrix in row 3 i + j.
    One rotation for all particles before the next, as numpy does them, so
    each particle meets the same operations in the same order; and a loop
    over particles with no branch in it, which the compiler can vectorize
    (each lane still rounds every operation on its own)."""
    N = B.shape[1]
    for _ in range(sweeps):
        for pq in range(3):
            p = 0 if pq < 2 else 1
            q = 1 if pq == 0 else 2
            for n in range(N):
                b0p, b1p, b2p = B[p, n], B[3 + p, n], B[6 + p, n]
                b0q, b1q, b2q = B[q, n], B[3 + q, n], B[6 + q, n]
                al = (b0p * b0p + b1p * b1p) + b2p * b2p
                be = (b0q * b0q + b1q * b1q) + b2q * b2q
                ga = (b0p * b0q + b1p * b1q) + b2p * b2q
                nz = ga != 0.0
                th = (be - al) / (2.0 * ga if nz else 1.0)
                t = (1.0 if th >= 0.0 else -1.0) / (abs(th) + np.sqrt(1.0 + th * th))
                t = t if nz else 0.0
                c = 1.0 / np.sqrt(1.0 + t * t)
                s = c * t
                B[p, n] = c * b0p - s * b0q
                B[3 + p, n] = c * b1p - s * b1q
                B[6 + p, n] = c * b2p - s * b2q
                B[q, n] = s * b0p + c * b0q
                B[3 + q, n] = s * b1p + c * b1q
                B[6 + q, n] = s * b2p + c * b2q
                v0p, v1p, v2p = V[p, n], V[3 + p, n], V[6 + p, n]
                v0q, v1q, v2q = V[q, n], V[3 + q, n], V[6 + q, n]
                V[p, n] = c * v0p - s * v0q
                V[3 + p, n] = c * v1p - s * v1q
                V[6 + p, n] = c * v2p - s * v2q
                V[q, n] = s * v0p + c * v0q
                V[3 + q, n] = s * v1p + c * v1q
                V[6 + q, n] = s * v2p + c * v2q


@njit(cache=True, error_model="numpy")
def _eig_sweeps(D, sweeps):
    """The sweeps of `core._top_eig3` on every particle at once: D holds
    one particle per column, rows d0 d1 d2 o01 o02 o12. Returns the top
    eigenvalue of each."""
    N = D.shape[1]
    for _ in range(sweeps):
        for k in range(3):
            # (p, q, r): (0, 1, 2), (0, 2, 1), (1, 2, 0); the rows of
            # d_p, d_q, o_pq, o_rp, o_rq
            ip = 0 if k < 2 else 1
            iq = 1 if k == 0 else 2
            ipq = 3 if k == 0 else (4 if k == 1 else 5)
            irp = 4 if k == 0 else 3
            irq = 5 if k < 2 else 4
            for n in range(N):
                app, aqq, apq = D[ip, n], D[iq, n], D[ipq, n]
                nz = apq != 0.0
                th = (aqq - app) / (2.0 * apq if nz else 1.0)
                t = (1.0 if th >= 0.0 else -1.0) / (abs(th) + np.sqrt(1.0 + th * th))
                t = t if nz else 0.0
                c = 1.0 / np.sqrt(1.0 + t * t)
                s = c * t
                D[ip, n] = app - t * apq
                D[iq, n] = aqq + t * apq
                D[ipq, n] = 0.0
                arp, arq = D[irp, n], D[irq, n]
                D[irp, n] = c * arp - s * arq
                D[irq, n] = s * arp + c * arq
    out = np.empty(N)
    for n in range(N):
        out[n] = _mx(_mx(D[0, n], D[1, n]), D[2, n])
    return out


@njit(cache=True, error_model="numpy")
def _finish_svd(Bn, Vn, B, V, S, U):
    """The rest of `core._svd3` for one particle, after its sweeps: its
    columns Bn and Vn (rows 3 i + j) into B and V, then S largest first,
    and U."""
    for i in range(3):
        for j in range(3):
            B[i, j] = Bn[3 * i + j]
            V[i, j] = Vn[3 * i + j]
    for k in range(3):
        S[k] = np.sqrt((B[0, k] * B[0, k] + B[1, k] * B[1, k]) + B[2, k] * B[2, k])
    for n in range(3):                       # largest first, ties left in place
        p = 1 if n == 1 else 0
        q = p + 1
        if S[q] > S[p]:
            for i in range(3):
                tb = B[i, p]
                B[i, p] = B[i, q]
                B[i, q] = tb
                tv = V[i, p]
                V[i, p] = V[i, q]
                V[i, q] = tv
            ts = S[p]
            S[p] = S[q]
            S[q] = ts
    s0 = S[0]
    for i in range(3):
        if s0 > 0.0:
            U[i, 0] = B[i, 0] / s0
        else:
            U[i, 0] = 1.0 if i == 0 else 0.0
    d = (U[0, 0] * B[0, 1] + U[1, 0] * B[1, 1]) + U[2, 0] * B[2, 1]
    y0 = B[0, 1] - d * U[0, 0]
    y1 = B[1, 1] - d * U[1, 0]
    y2 = B[2, 1] - d * U[2, 0]
    n1 = np.sqrt((y0 * y0 + y1 * y1) + y2 * y2)
    if n1 > 0.0:
        U[0, 1] = y0 / n1
        U[1, 1] = y1 / n1
        U[2, 1] = y2 / n1
    else:
        k = 1 if abs(U[1, 0]) < abs(U[0, 0]) else 0
        if abs(U[2, 0]) < abs(U[k, 0]):
            k = 2
        uk = U[k, 0]
        f0 = (1.0 if k == 0 else 0.0) - uk * U[0, 0]
        f1 = (1.0 if k == 1 else 0.0) - uk * U[1, 0]
        f2 = (1.0 if k == 2 else 0.0) - uk * U[2, 0]
        nf = np.sqrt((f0 * f0 + f1 * f1) + f2 * f2)
        U[0, 1] = f0 / nf
        U[1, 1] = f1 / nf
        U[2, 1] = f2 / nf
    c0 = U[1, 0] * U[2, 1] - U[2, 0] * U[1, 1]
    c1 = U[2, 0] * U[0, 1] - U[0, 0] * U[2, 1]
    c2 = U[0, 0] * U[1, 1] - U[1, 0] * U[0, 1]
    if (c0 * B[0, 2] + c1 * B[1, 2]) + c2 * B[2, 2] < 0.0:
        c0, c1, c2 = -c0, -c1, -c2
    U[0, 2] = c0
    U[1, 2] = c1
    U[2, 2] = c2


@njit(cache=True, error_model="numpy")
def stress(F, C, m, mu, lam, yld, eta, strength, dmg, vol0, dt, c4,
           dmg_yield, svd_sweeps, eig_sweeps, affine):
    """`MPM._stress(dt) + m C` into `affine`. Like the numpy path it
    re-forms F in place where matter flowed and marks new cracks in `dmg`."""
    N = F.shape[0]
    Bs = np.empty((9, N))
    Vs = np.zeros((9, N))
    for n in range(N):
        for i in range(3):
            for j in range(3):
                Bs[3 * i + j, n] = F[n, i, j]
        Vs[0, n] = 1.0
        Vs[4, n] = 1.0
        Vs[8, n] = 1.0
    _sweeps(Bs, Vs, svd_sweeps)
    B = np.empty((3, 3))
    V = np.empty((3, 3))
    U = np.empty((3, 3))
    S = np.empty(3)
    R = np.empty((3, 3))
    A = np.empty((3, 3))
    PFs = np.empty((N, 3, 3))
    Js = np.empty(N)
    D = np.zeros((6, N))
    brittle = False
    for p in range(N):
        _finish_svd(Bs[:, p], Vs[:, p], B, V, S, U)
        yl = dmg_yield if dmg[p] else yld[p]
        plastic = yl > 0.0
        if plastic:
            lo = 1.0 - yl
            hi = 1.0 + yl
            for k in range(3):
                S[k] = _clip(S[k], lo, hi)
        for i in range(3):
            for j in range(3):
                R[i, j] = (U[i, 0] * V[j, 0] + U[i, 1] * V[j, 1]) + U[i, 2] * V[j, 2]
        J = (S[0] * S[1]) * S[2]
        Js[p] = J
        if plastic:
            for i in range(3):
                for j in range(3):
                    F[p, i, j] = ((U[i, 0] * S[0]) * V[j, 0] + (U[i, 1] * S[1]) * V[j, 1]) \
                        + (U[i, 2] * S[2]) * V[j, 2]
        mu2 = 2.0 * mu[p]
        for i in range(3):
            for j in range(3):
                A[i, j] = mu2 * (F[p, i, j] - R[i, j])
        PF = PFs[p]
        for i in range(3):
            for j in range(3):
                PF[i, j] = (A[i, 0] * F[p, j, 0] + A[i, 1] * F[p, j, 1]) + A[i, 2] * F[p, j, 2]
        vol = lam[p] * (J - 1.0) * J
        for i in range(3):
            PF[i, i] += vol
        for i in range(3):
            for j in range(3):
                PF[i, j] += eta[p] * (C[p, i, j] + C[p, j, i])
        if strength[p] > 0.0:
            brittle = True
            jj = _mx(J, 1e-9)
            D[0, p] = (0.5 * (PF[0, 0] + PF[0, 0])) / jj
            D[1, p] = (0.5 * (PF[1, 1] + PF[1, 1])) / jj
            D[2, p] = (0.5 * (PF[2, 2] + PF[2, 2])) / jj
            D[3, p] = (0.5 * (PF[0, 1] + PF[1, 0])) / jj
            D[4, p] = (0.5 * (PF[0, 2] + PF[2, 0])) / jj
            D[5, p] = (0.5 * (PF[1, 2] + PF[2, 1])) / jj
    if brittle:
        top = _eig_sweeps(D, eig_sweeps)
        for p in range(N):
            if strength[p] > 0.0 and top[p] > strength[p]:
                dmg[p] = True
    for p in range(N):
        PF = PFs[p]
        if dmg[p]:
            for i in range(3):
                for j in range(3):
                    PF[i, j] *= 0.05
            if Js[p] > 1.0:
                for i in range(3):
                    for j in range(3):
                        PF[i, j] = 0.0
        k = -(dt * vol0[p]) * c4
        for i in range(3):
            for j in range(3):
                affine[p, i, j] = k * PF[i, j] + m[p] * C[p, i, j]


@njit(cache=True, error_model="numpy")
def grid(gm, gp, gb, lo, hi, wy, wz, shape, dt, grav, grid_contact, friction,
         phi_node, lo_s, hi_s, thr):
    """The grid step of `MPM._solve`, node by node: momentum to velocity,
    gravity, the world's faces, the walls' faces with their friction, and
    the nodes deep inside a wall. Returns the velocities (nodes, 3)."""
    ncell = gm.shape[0]
    gv = np.zeros((ncell, 3))
    wx = ncell // (wy * wz)
    for f in range(ncell):
        ix = f // (wy * wz)
        iy = (f // wz) % wy
        iz = f % wz
        if gm[f] > 1e-12:
            for a in range(3):
                gv[f, a] = gp[f, a] / gm[f]
            for a in range(3):
                gv[f, a] = gv[f, a] + grav[a] * dt
        for a in range(3):
            ia = ix if a == 0 else (iy if a == 1 else iz)
            na = wx if a == 0 else (wy if a == 1 else wz)
            if lo[a] == 0 and ia == 0:
                if gv[f, a] < 0.0:
                    gv[f, a] = 0.0
            if hi[a] == shape[a] and ia == na - 1:
                if gv[f, a] > 0.0:
                    gv[f, a] = 0.0
        if grid_contact:
            mm = _mx(gm[f], 1e-12)
            removed = 0.0
            for a in range(3):
                fr = gb[f, a] / mm
                va = gv[f, a]
                if (fr > 0.05 and va > 0.0) or (fr < -0.05 and va < 0.0):
                    removed += abs(va)
                    gv[f, a] = 0.0
            if removed > 0.0:
                v0, v1, v2 = gv[f, 0], gv[f, 1], gv[f, 2]
                vtn = np.sqrt((v0 * v0 + v1 * v1) + v2 * v2)
                keep = _mx(0.0, 1.0 - friction * removed / _mx(vtn, 1e-12))
                gv[f, 0] = v0 * keep
                gv[f, 1] = v1 * keep
                gv[f, 2] = v2 * keep
        g0, g1, g2 = lo[0] + ix, lo[1] + iy, lo[2] + iz
        if lo_s[0] <= g0 < hi_s[0] and lo_s[1] <= g1 < hi_s[1] and lo_s[2] <= g2 < hi_s[2]:
            if phi_node[g0 - lo_s[0], g1 - lo_s[1], g2 - lo_s[2]] < thr:
                gv[f, 0] = 0.0
                gv[f, 1] = 0.0
                gv[f, 2] = 0.0
    return gv


@njit(cache=True, error_model="numpy")
def move(x, v, C, F, dt):
    """x + dt v and (I + dt C) F, as `MPM.substep` writes them."""
    N = x.shape[0]
    xn = np.empty_like(x)
    Fn = np.empty_like(F)
    A = np.empty((3, 3))
    for p in range(N):
        for a in range(3):
            xn[p, a] = x[p, a] + dt * v[p, a]
        for i in range(3):
            for j in range(3):
                A[i, j] = (1.0 if i == j else 0.0) + dt * C[p, i, j]
        for i in range(3):
            for j in range(3):
                Fn[p, i, j] = (A[i, 0] * F[p, 0, j] + A[i, 1] * F[p, 1, j]) + A[i, 2] * F[p, 2, j]
    return xn, Fn


@njit(cache=True, error_model="numpy")
def hold(x, v, m, rig, cid, ncl, vol0, lam, mu, strength, dmg):
    """`MPM._hold_shape`: each cluster's motion pulled toward a rigid one,
    and the bonds that carried too much parted. Writes v, cid and dmg."""
    N = x.shape[0]
    W = np.zeros(ncl)
    xc = np.zeros((ncl, 3))
    vc = np.zeros((ncl, 3))
    L = np.zeros((ncl, 3))
    I = np.zeros((ncl, 3, 3))
    for p in range(N):
        c = cid[p]
        if c >= 0:
            W[c] += m[p] * rig[p]
    for a in range(3):
        for p in range(N):
            c = cid[p]
            if c >= 0:
                xc[c, a] += m[p] * rig[p] * x[p, a]
        for p in range(N):
            c = cid[p]
            if c >= 0:
                vc[c, a] += m[p] * rig[p] * v[p, a]
    for c in range(ncl):
        if W[c] > 1e-12:
            for a in range(3):
                xc[c, a] = xc[c, a] / W[c]
                vc[c, a] = vc[c, a] / W[c]
    r = np.empty(3)
    u = np.empty(3)
    rv = np.empty(3)
    for p in range(N):
        c = cid[p]
        if c < 0:
            continue
        w = m[p] * rig[p]
        for a in range(3):
            r[a] = x[p, a] - xc[c, a]
            u[a] = v[p, a] - vc[c, a]
        rv[0] = r[1] * u[2] - r[2] * u[1]
        rv[1] = r[2] * u[0] - r[0] * u[2]
        rv[2] = r[0] * u[1] - r[1] * u[0]
        for a in range(3):
            L[c, a] += w * rv[a]
        rr = (r[0] * r[0] + r[1] * r[1]) + r[2] * r[2]
        for a in range(3):
            for b in range(3):
                term = (rr if a == b else 0.0) - r[a] * r[b]
                I[c, a, b] += w * term
    om = np.empty((ncl, 3))
    for c in range(ncl):
        reg = 1e-9 + 1e-6 * ((I[c, 0, 0] + I[c, 1, 1]) + I[c, 2, 2])
        for a in range(3):
            I[c, a, a] += reg
        l10 = I[c, 1, 0] / I[c, 0, 0]
        l20 = I[c, 2, 0] / I[c, 0, 0]
        a11 = I[c, 1, 1] - l10 * I[c, 0, 1]
        a12 = I[c, 1, 2] - l10 * I[c, 0, 2]
        a21 = I[c, 2, 1] - l20 * I[c, 0, 1]
        a22 = I[c, 2, 2] - l20 * I[c, 0, 2]
        b1 = L[c, 1] - l10 * L[c, 0]
        b2 = L[c, 2] - l20 * L[c, 0]
        l21 = a21 / a11
        a22 = a22 - l21 * a12
        b2 = b2 - l21 * b1
        x2 = b2 / a22
        x1 = (b1 - a12 * x2) / a11
        om[c, 0] = ((L[c, 0] - I[c, 0, 1] * x1) - I[c, 0, 2] * x2) / I[c, 0, 0]
        om[c, 1] = x1
        om[c, 2] = x2
    for p in range(N):
        c = cid[p]
        if c < 0:
            continue
        for a in range(3):
            r[a] = x[p, a] - xc[c, a]
        vr0 = vc[c, 0] + (om[c, 1] * r[2] - om[c, 2] * r[1])
        vr1 = vc[c, 1] + (om[c, 2] * r[0] - om[c, 0] * r[2])
        vr2 = vc[c, 2] + (om[c, 0] * r[1] - om[c, 1] * r[0])
        d0 = rig[p] * (vr0 - v[p, 0])
        d1 = rig[p] * (vr1 - v[p, 1])
        d2 = rig[p] * (vr2 - v[p, 2])
        v[p, 0] = v[p, 0] + d0
        v[p, 1] = v[p, 1] + d1
        v[p, 2] = v[p, 2] + d2
        rho = m[p] / _mx(vol0[p], 1e-12)
        mod = lam[p] + 2.0 * mu[p]
        sig = np.sqrt(_mx(mod * rho, 0.0)) * np.sqrt((d0 * d0 + d1 * d1) + d2 * d2)
        if strength[p] > 0.0 and sig > strength[p]:
            cid[p] = -1
            dmg[p] = True


@njit(cache=True, error_model="numpy")
def _corners(x0, x1, x2, inv_dx, gtop, top, lo, wtop, i0, i1, fr):
    """The eight cells and weights `MPM._sample` reads at one point."""
    for a in range(3):
        g = (x0 if a == 0 else (x1 if a == 1 else x2)) * inv_dx - 0.5
        g = g if g > 0.0 else 0.0
        g = g if g < gtop[a] else gtop[a]
        i = int(g)
        fr[a] = g - i
        j = min(i + 1, top[a])
        i0[a] = min(max(i - lo[a], 0), wtop[a])
        i1[a] = min(max(j - lo[a], 0), wtop[a])


@njit(cache=True, error_model="numpy")
def contact(x, v, phi, pgrad, lo_s, inv_dx, gtop, top, friction, deep_at):
    """`MPM._contact` for one window holding every particle: the ones
    inside a wall lose the velocity into it (with Coulomb friction along
    it) and are put back out along the axis they came in on. Writes x and
    v; returns how many were deeper than `deep_at`."""
    N = x.shape[0]
    wtop = np.empty(3, np.int64)
    for a in range(3):
        wtop[a] = phi.shape[a] - 1
    i0 = np.empty(3, np.int64)
    i1 = np.empty(3, np.int64)
    fr = np.empty(3)
    nrm = np.empty(3)
    deep = 0
    for p in range(N):
        _corners(x[p, 0], x[p, 1], x[p, 2], inv_dx, gtop, top, lo_s, wtop, i0, i1, fr)
        ph = 0.0
        for a in range(2):
            wa = fr[0] if a else 1.0 - fr[0]
            ia = i1[0] if a else i0[0]
            for b in range(2):
                wb = fr[1] if b else 1.0 - fr[1]
                ib = i1[1] if b else i0[1]
                for c in range(2):
                    wc = fr[2] if c else 1.0 - fr[2]
                    ic = i1[2] if c else i0[2]
                    ph = ph + wa * wb * wc * phi[ia, ib, ic]
        if not ph < 0.0:
            continue
        if ph < deep_at:
            deep += 1
        for k in range(3):
            nrm[k] = 0.0
        for a in range(2):
            wa = fr[0] if a else 1.0 - fr[0]
            ia = i1[0] if a else i0[0]
            for b in range(2):
                wb = fr[1] if b else 1.0 - fr[1]
                ib = i1[1] if b else i0[1]
                for c in range(2):
                    wc = fr[2] if c else 1.0 - fr[2]
                    ic = i1[2] if c else i0[2]
                    wt = wa * wb * wc
                    for k in range(3):
                        nrm[k] = nrm[k] + wt * pgrad[ia, ib, ic, k]
        ln = np.sqrt((nrm[0] * nrm[0] + nrm[1] * nrm[1]) + nrm[2] * nrm[2])
        for k in range(3):
            nrm[k] = nrm[k] / _mx(ln, 1e-12) if ln > 1e-12 else 0.0
        dom = 1 if abs(nrm[1]) > abs(nrm[0]) else 0
        if abs(nrm[2]) > abs(nrm[dom]):
            dom = 2
        ga = nrm[dom]
        va = v[p, dom]
        if (ga > 0.0 and va < 0.0) or (ga < 0.0 and va > 0.0):
            o0 = 1 if dom == 0 else 0
            o1 = 1 if dom == 2 else 2
            vn = abs(va)
            t0 = v[p, o0]
            t1 = v[p, o1]
            vtn = np.sqrt(t0 * t0 + t1 * t1)
            keep = _mx(0.0, 1.0 - friction * vn / _mx(vtn, 1e-12))
            v[p, dom] = 0.0
            v[p, o0] = t0 * keep
            v[p, o1] = t1 * keep
        sg = 1.0 if ga > 0.0 else (-1.0 if ga < 0.0 else 0.0)
        for k in range(3):
            x[p, k] = x[p, k] - ph * (sg if k == dom else 0.0)
    return deep


@njit(cache=True, error_model="numpy")
def substep(x, v, C, F, m, vol0, mu, lam, yld, eta, strength, dmg, cid, rig,
            ncl, wall, anywall, lo_s, hi_s, phi, pgrad, phi_node, shape, gtop,
            dx, inv_dx, c4, dt, grav, friction, grid_contact, particle_contact,
            window, dmg_yield, svd_sweeps, eig_sweeps, node_thr, deep_at,
            clip_lo, clip_hi):
    """`MPM.substep` for one island, whole. Writes F (where matter flowed),
    dmg and cid in place, as the numpy path does, and returns the new
    (x, v, C, F) and the count of particles read deeper than the margin."""
    N = x.shape[0]
    top = shape - 1
    base = np.empty((N, 3), np.int64)
    fx = np.empty((N, 3))
    w = np.empty((3, N, 3))
    weights(x, inv_dx, base, fx, w)
    blk = np.zeros((N, 3))
    if anywall:
        contact_faces(x, wall, inv_dx, dx, top, blk)
    affine = np.empty((N, 3, 3))
    stress(F, C, m, mu, lam, yld, eta, strength, dmg, vol0, dt, c4,
           dmg_yield, svd_sweeps, eig_sweeps, affine)
    lo = np.zeros(3, np.int64)
    hi = shape.copy()
    if window:
        for a in range(3):
            bl = base[0, a]
            bh = base[0, a]
            for p in range(1, N):
                bl = min(bl, base[p, a])
                bh = max(bh, base[p, a])
            lo[a] = max(bl, 0)
            hi[a] = min(bh + 3, shape[a])
    wy = hi[1] - lo[1]
    wz = hi[2] - lo[2]
    ncell = (hi[0] - lo[0]) * wy * wz
    gm = np.zeros(ncell)
    gp = np.zeros((ncell, 3))
    gb = np.zeros((ncell, 3))
    p2g(base, fx, w, m, v, affine, blk, lo, top, wy, wz, dx, gm, gp, gb)
    gv = grid(gm, gp, gb, lo, hi, wy, wz, shape, dt, grav, grid_contact,
              friction, phi_node, lo_s, hi_s, node_thr)
    nv = np.zeros((N, 3))
    nC = np.zeros((N, 3, 3))
    g2p(base, fx, w, gv, lo, top, wy, wz, dx, c4, nv, nC)
    xn, Fn = move(x, nv, nC, F, dt)
    if ncl:
        hold(xn, nv, m, rig, cid, ncl, vol0, lam, mu, strength, dmg)
    deep = 0
    if particle_contact and anywall:
        deep = contact(xn, nv, phi, pgrad, lo_s, inv_dx, gtop, top, friction, deep_at)
    for p in range(N):
        for a in range(3):
            xn[p, a] = _clip(xn[p, a], clip_lo, clip_hi[a])
    return xn, nv, nC, Fn, deep
