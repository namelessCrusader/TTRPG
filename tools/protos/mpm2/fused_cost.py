"""COST MODEL ONLY (not the physics, not exact): what does one MLS-MPM substep
cost when ALL of it is one numba call — stress with a per-particle 3x3
Jacobi SVD, Kelvin-Voigt, P2G on the active window, the grid update with a
contact rule, G2P, F update, and a per-cluster rigid projection — so that
no numpy call overhead is left? Measured at 66, 512 and 3662 particles."""
import time
import numpy as np
import numba as nb

@nb.njit(cache=False)
def jacobi_eig3(A, V):
    # symmetric 3x3 eigen-decomposition by cyclic Jacobi, fixed 6 sweeps
    for i in range(3):
        for j in range(3):
            V[i, j] = 1.0 if i == j else 0.0
    for sweep in range(6):
        for p in range(2):
            for q in range(p + 1, 3):
                apq = A[p, q]
                if abs(apq) < 1e-30:
                    continue
                th = 0.5 * (A[q, q] - A[p, p]) / apq
                t = (1.0 if th >= 0 else -1.0) / (abs(th) + np.sqrt(th * th + 1.0))
                c = 1.0 / np.sqrt(t * t + 1.0); s = t * c
                for k in range(3):
                    akp, akq = A[k, p], A[k, q]
                    A[k, p] = c * akp - s * akq; A[k, q] = s * akp + c * akq
                for k in range(3):
                    apk, aqk = A[p, k], A[q, k]
                    A[p, k] = c * apk - s * aqk; A[q, k] = s * apk + c * aqk
                for k in range(3):
                    vkp, vkq = V[k, p], V[k, q]
                    V[k, p] = c * vkp - s * vkq; V[k, q] = s * vkp + c * vkq

@nb.njit(cache=False)
def substep(x, v, C, F, m, vol0, mu, lam, eta, cid, rig, ncl, dt, dx, wall, gm, gp):
    N = x.shape[0]; inv = 1.0 / dx
    nx, ny, nz = wall.shape
    lo = np.full(3, 1 << 30); hi = np.zeros(3, np.int64)
    base = np.empty((N, 3), np.int64); fx = np.empty((N, 3)); w = np.empty((N, 3, 3))
    for p in range(N):
        for a in range(3):
            px = x[p, a] * inv; b = int(np.floor(px - 0.5)); f = px - b
            base[p, a] = b; fx[p, a] = f
            w[p, 0, a] = 0.5 * (1.5 - f) ** 2; w[p, 1, a] = 0.75 - (f - 1.0) ** 2; w[p, 2, a] = 0.5 * (f - 0.5) ** 2
            lo[a] = min(lo[a], max(b, 0)); hi[a] = max(hi[a], min(b + 3, (nx, ny, nz)[a]))
    wx, wy, wz = hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]
    nn = wx * wy * wz
    for i in range(nn):
        gm[i] = 0.0; gp[i, 0] = 0.0; gp[i, 1] = 0.0; gp[i, 2] = 0.0
    FtF = np.empty((3, 3)); V = np.empty((3, 3)); U = np.empty((3, 3)); R = np.empty((3, 3)); aff = np.empty((3, 3))
    for p in range(N):
        Fp = F[p]
        for i in range(3):
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += Fp[k, i] * Fp[k, j]
                FtF[i, j] = s
        jacobi_eig3(FtF, V)
        J = 1.0
        for k in range(3):
            sk = np.sqrt(max(FtF[k, k], 1e-12)); J *= sk
            for i in range(3):
                s = 0.0
                for j in range(3):
                    s += Fp[i, j] * V[j, k]
                U[i, k] = s / sk
        for i in range(3):
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += U[i, k] * V[j, k]
                R[i, j] = s
        k0 = -dt * vol0[p] * 4.0 * inv * inv
        for i in range(3):
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += (Fp[i, k] - R[i, k]) * Fp[j, k]
                pf = 2.0 * mu[p] * s + (lam[p] * (J - 1.0) * J if i == j else 0.0) + eta[p] * (C[p, i, j] + C[p, j, i])
                aff[i, j] = k0 * pf + m[p] * C[p, i, j]
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    wt = w[p, a, 0] * w[p, b, 1] * w[p, c, 2]
                    gi = ((base[p, 0] + a - lo[0]) * wy + (base[p, 1] + b - lo[1])) * wz + (base[p, 2] + c - lo[2])
                    d0 = (a - fx[p, 0]) * dx; d1 = (b - fx[p, 1]) * dx; d2 = (c - fx[p, 2]) * dx
                    gm[gi] += wt * m[p]
                    for i in range(3):
                        gp[gi, i] += wt * (m[p] * v[p, i] + aff[i, 0] * d0 + aff[i, 1] * d1 + aff[i, 2] * d2)
    for i in range(nn):
        if gm[i] > 1e-12:
            for a in range(3):
                gp[i, a] /= gm[i]
            gp[i, 2] -= 9.81 * dt
            ix = lo[0] + i // (wy * wz); iy = lo[1] + (i // wz) % wy; iz = lo[2] + i % wz
            if ix < nx and iy < ny and iz < nz and wall[ix, iy, iz] and gp[i, 2] < 0.0:
                gp[i, 2] = 0.0
    for p in range(N):
        nv0 = 0.0; nv1 = 0.0; nv2 = 0.0
        for i in range(3):
            for j in range(3):
                C[p, i, j] = 0.0
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    wt = w[p, a, 0] * w[p, b, 1] * w[p, c, 2]
                    gi = ((base[p, 0] + a - lo[0]) * wy + (base[p, 1] + b - lo[1])) * wz + (base[p, 2] + c - lo[2])
                    d = ((a - fx[p, 0]) * dx, (b - fx[p, 1]) * dx, (c - fx[p, 2]) * dx)
                    g0, g1, g2 = gp[gi, 0], gp[gi, 1], gp[gi, 2]
                    nv0 += wt * g0; nv1 += wt * g1; nv2 += wt * g2
                    for j in range(3):
                        C[p, 0, j] += 4 * inv * inv * wt * g0 * d[j]
                        C[p, 1, j] += 4 * inv * inv * wt * g1 * d[j]
                        C[p, 2, j] += 4 * inv * inv * wt * g2 * d[j]
        v[p, 0] = nv0; v[p, 1] = nv1; v[p, 2] = nv2
        for a in range(3):
            x[p, a] += dt * v[p, a]
        Fn = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += ((1.0 if i == k else 0.0) + dt * C[p, i, k]) * F[p, k, j]
                Fn[i, j] = s
        F[p] = Fn
    # rigid projection per cluster: COM velocity + spin (sums only; solve skipped)
    W = np.zeros(ncl); vc = np.zeros((ncl, 3)); xc = np.zeros((ncl, 3))
    for p in range(N):
        c = cid[p]
        if c >= 0:
            wp = m[p] * rig[p]; W[c] += wp
            for a in range(3):
                vc[c, a] += wp * v[p, a]; xc[c, a] += wp * x[p, a]
    for p in range(N):
        c = cid[p]
        if c >= 0 and W[c] > 0:
            for a in range(3):
                v[p, a] += rig[p] * (vc[c, a] / W[c] - v[p, a])

def make(N, dx=0.05):
    side = int(np.ceil(N ** (1 / 3)))
    if N == 66:
        g = np.stack(np.meshgrid(np.arange(3), np.arange(2), np.arange(11), indexing="ij"), -1).reshape(-1, 3)[:N]
    else:
        g = np.stack(np.meshgrid(*[np.arange(side)] * 3, indexing="ij"), -1).reshape(-1, 3)[:N]
    x = (g + 0.5) * dx + np.array([0.9, 0.9, 0.1])
    F = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
    return dict(x=x, v=np.zeros((N, 3)), C=np.zeros((N, 3, 3)), F=F, m=np.full(N, 0.075),
                vol0=np.full(N, dx ** 3), mu=np.full(N, 4e5), lam=np.full(N, 9e5), eta=np.full(N, 50.0),
                cid=np.zeros(N, np.int32), rig=np.ones(N), ncl=1)

wall = np.zeros((40, 40, 70), bool); wall[:, :, 0] = True
gm = np.zeros(40 * 40 * 70); gp = np.zeros((40 * 40 * 70, 3))
s = make(66)
substep(s["x"], s["v"], s["C"], s["F"], s["m"], s["vol0"], s["mu"], s["lam"], s["eta"], s["cid"], s["rig"], 1, 2.9e-4, 0.05, wall, gm, gp)
for N in (66, 512, 3662):
    s = make(N)
    K = 400 if N < 1000 else 100
    t0 = time.perf_counter()
    for _ in range(K):
        substep(s["x"], s["v"], s["C"], s["F"], s["m"], s["vol0"], s["mu"], s["lam"], s["eta"], s["cid"], s["rig"], 1, 2.9e-4, 0.05, wall, gm, gp)
    us = (time.perf_counter() - t0) / K * 1e6
    print(f"fused numba substep, {N:5d} particles: {us:8.1f} us/substep "
          f"({us / N:.2f} us/particle) -> x86 substeps = {us * 86 / 1e3:6.1f} ms/tick at 5 cm")
