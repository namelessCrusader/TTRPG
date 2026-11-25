"""Classical Hamiltonian compact U(1) lattice gauge theory + compact matter phase (R2.1).
H = sum E^2/2 + K sum_p (1-cos theta_p) + sum pi^2/(2 chi) - J sum_l cos(phi_i - phi_j - theta_ij).
Leapfrog (kick-drift-kick). Gauss law G = div E + pi is conserved to round-off by construction
(the matter kick moves E and pi by the same number; the plaquette kick has zero divergence).
Sources: Kogut-Susskind 1975 Hamiltonian; Fradkin-Shenker 1979; classical real-time LGT (Moore et al 1990s).
Imports nothing from src/ or tests/."""
import numpy as np
import numba as nb

TWO_PI = 2 * np.pi


def make_lattice(L):
    N = L ** 3
    idx = np.arange(N).reshape(L, L, L)
    fwd = np.empty((3, N), np.int64)
    bwd = np.empty((3, N), np.int64)
    for mu in range(3):
        fwd[mu] = np.roll(idx, -1, axis=mu).ravel()
        bwd[mu] = np.roll(idx, 1, axis=mu).ravel()
    return fwd, bwd


@nb.njit(cache=True)
def _wrap(a):
    return a - TWO_PI * np.floor((a + np.pi) / TWO_PI)


@nb.njit(cache=True)
def _kick(th, E, phi, pi, fwd, K, J, h, use_m):
    N = th.shape[1]
    for s in range(N):
        for mu in range(3):
            for nu in range(mu + 1, 3):
                smu = fwd[mu, s]
                snu = fwd[nu, s]
                b = th[mu, s] + th[nu, smu] - th[mu, snu] - th[nu, s]
                f = h * K * np.sin(b)
                E[mu, s] -= f
                E[nu, smu] -= f
                E[mu, snu] += f
                E[nu, s] += f
        if use_m:
            for mu in range(3):
                sn = fwd[mu, s]
                d = phi[s] - phi[sn] - th[mu, s]
                f = h * J * np.sin(d)
                E[mu, s] += f
                pi[s] -= f
                pi[sn] += f


@nb.njit(cache=True)
def _drift(th, E, phi, pi, h, chi, use_m):
    for mu in range(3):
        for s in range(th.shape[1]):
            th[mu, s] = _wrap(th[mu, s] + h * E[mu, s])
    if use_m:
        for s in range(phi.shape[0]):
            phi[s] = _wrap(phi[s] + h * pi[s] / chi)


@nb.njit(cache=True)
def advance(th, E, phi, pi, fwd, K, J, chi, dt, n, use_m):
    """n leapfrog steps; E, pi synchronised with th, phi at entry and exit."""
    if n <= 0:
        return
    _kick(th, E, phi, pi, fwd, K, J, 0.5 * dt, use_m)
    for i in range(n):
        _drift(th, E, phi, pi, dt, chi, use_m)
        if i < n - 1:
            _kick(th, E, phi, pi, fwd, K, J, dt, use_m)
        else:
            _kick(th, E, phi, pi, fwd, K, J, 0.5 * dt, use_m)


@nb.njit(cache=True)
def plaq_angles(th, fwd):
    """B[p,s]: p=0 yz, 1 zx, 2 xy; wrapped plaquette angle (outward-normal orientation)."""
    N = th.shape[1]
    B = np.empty((3, N))
    for s in range(N):
        for p in range(3):
            mu = (p + 1) % 3
            nu = (p + 2) % 3
            smu = fwd[mu, s]
            snu = fwd[nu, s]
            b = th[mu, s] + th[nu, smu] - th[mu, snu] - th[nu, s]
            B[p, s] = _wrap(b)
    return B


@nb.njit(cache=True)
def observables(th, phi, fwd, use_m):
    """returns plaq cos mean, higgs cos d mean, monopole count (abs), vortex plaquettes (abs v>0), signed vortex count in xy"""
    N = th.shape[1]
    B = plaq_angles(th, fwd)
    pc = 0.0
    for p in range(3):
        for s in range(N):
            pc += np.cos(B[p, s])
    pc /= 3 * N
    mon = 0
    for s in range(N):
        t = 0.0
        for p in range(3):
            t += B[p, fwd[p, s]] - B[p, s]
        if abs(round(t / TWO_PI)) > 0:
            mon += 1
    hc = 0.0
    nv = 0
    vxy = 0
    if use_m:
        for mu in range(3):
            for s in range(N):
                hc += np.cos(phi[s] - phi[fwd[mu, s]] - th[mu, s])
        hc /= 3 * N
        for p in range(3):
            mu = (p + 1) % 3
            nu = (p + 2) % 3
            for s in range(N):
                smu = fwd[mu, s]
                snu = fwd[nu, s]
                w = (_wrap(phi[s] - phi[smu] - th[mu, s]) + _wrap(phi[smu] - phi[fwd[nu, smu]] - th[nu, smu])
                     - _wrap(phi[snu] - phi[fwd[mu, snu]] - th[mu, snu]) - _wrap(phi[s] - phi[snu] - th[nu, s]))
                v = round((w + B[p, s]) / TWO_PI)
                if v != 0:
                    nv += 1
                if p == 2:
                    vxy += v
    return pc, hc, mon, nv, vxy


@nb.njit(cache=True)
def energy_parts(th, E, phi, pi, fwd, K, J, chi, use_m):
    N = th.shape[1]
    eE = 0.0
    for mu in range(3):
        for s in range(N):
            eE += 0.5 * E[mu, s] ** 2
    eB = 0.0
    for s in range(N):
        for mu in range(3):
            for nu in range(mu + 1, 3):
                smu = fwd[mu, s]
                snu = fwd[nu, s]
                b = th[mu, s] + th[nu, smu] - th[mu, snu] - th[nu, s]
                eB += K * (1 - np.cos(b))
    eM = 0.0
    if use_m:
        for s in range(N):
            eM += 0.5 * pi[s] ** 2 / chi
            for mu in range(3):
                eM -= J * np.cos(phi[s] - phi[fwd[mu, s]] - th[mu, s])
    return eE, eB, eM


@nb.njit(cache=True)
def divE(E, bwd):
    N = E.shape[1]
    out = np.zeros(N)
    for s in range(N):
        for mu in range(3):
            out[s] += E[mu, s] - E[mu, bwd[mu, s]]
    return out


class Sim:
    def __init__(self, L, K, J=0.0, chi=0.25, T=1.0, use_m=True, dt=0.02, seed=1):
        self.L, self.K, self.J, self.chi, self.T = L, K, J, chi, T
        self.use_m = use_m and (chi > 0)
        self.dt = dt
        self.N = L ** 3
        self.fwd, self.bwd = make_lattice(L)
        self.rng = np.random.default_rng(seed)
        self.th = np.zeros((3, self.N))
        self.E = np.zeros((3, self.N))
        self.phi = np.zeros(self.N)
        self.pi = np.zeros(self.N)
        self.rho = np.zeros(self.N)
        kk = 2 * np.pi * np.fft.fftfreq(L)
        k2 = 2 * sum((1 - np.cos(k)) for k in np.meshgrid(kk, kk, kk, indexing="ij"))
        self.kA = k2  # eigenvalues of -laplacian (lattice)
        self.G0 = None

    # --- Gauss law
    def gauss(self):
        return divE(self.E, self.bwd) + self.pi - self.rho

    def _solve(self, r):
        den = self.kA + (self.chi if self.use_m else 0.0)
        rk = np.fft.fftn(r.reshape(self.L, self.L, self.L))
        den = den.copy()
        den[0, 0, 0] = 1.0 if not self.use_m else den[0, 0, 0]
        lk = rk / den
        if not self.use_m:
            lk[0, 0, 0] = 0.0
        return np.real(np.fft.ifftn(lk)).ravel()

    def project(self, E, pi):
        """minimal-energy shift so that div E + pi = rho (in place)."""
        r = divE(E, self.bwd) + pi - self.rho
        lam = self._solve(r)
        g = lam[None, :] - lam[self.fwd]  # g_mu(s) = lam(s) - lam(s+mu)
        E -= g
        if self.use_m:
            pi -= self.chi * lam

    def refresh(self, frac=1.0):
        """Andersen-type refresh of (E,pi) from the Gauss-conditioned Gaussian at temperature T.
        frac<1: mix (OU style, keeps the measure)."""
        T = self.T
        Eg = self.rng.normal(0, np.sqrt(T), self.E.shape)
        pg = self.rng.normal(0, np.sqrt(T * self.chi), self.N) if self.use_m else np.zeros(self.N)
        if frac < 1.0:
            a = np.sqrt(1 - frac)
            b = np.sqrt(frac)
            # remove the mean-field part first (deterministic shift), mix the fluctuation, put it back
            Ec, pc = self.mean_field()
            Ef = self.E - Ec
            pf = self.pi - pc
            Eg = a * Ef + b * Eg
            pg = a * pf + b * pg
            self.E[:] = Eg
            self.pi[:] = pg
            self.project(self.E, self.pi)
            return
        self.E[:] = Eg
        self.pi[:] = pg
        self.project(self.E, self.pi)

    def mean_field(self):
        """E_c, pi_c: the Gauss-law minimiser for rho."""
        E0 = np.zeros_like(self.E)
        p0 = np.zeros(self.N)
        self.project(E0, p0)
        return E0, p0

    def hot_start(self, spread=0.0):
        self.th[:] = self.rng.normal(0, spread, self.th.shape) if spread > 0 else 0.0
        self.phi[:] = self.rng.uniform(-np.pi, np.pi, self.N) if (self.use_m and spread > 1) else 0.0
        self.refresh()
        self.G0 = self.gauss().copy()

    def step(self, n):
        advance(self.th, self.E, self.phi, self.pi, self.fwd, self.K, self.J, self.chi if self.use_m else 1.0,
                self.dt, n, self.use_m)

    def thermalise(self, n_refresh=40, steps=100, frac=1.0):
        for _ in range(n_refresh):
            self.step(steps)
            self.refresh(frac)

    def energies(self):
        return energy_parts(self.th, self.E, self.phi, self.pi, self.fwd, self.K, self.J,
                            self.chi if self.use_m else 1.0, self.use_m)

    def obs(self):
        return observables(self.th, self.phi, self.fwd, self.use_m)

    def set_charges(self, pos_a, pos_b):
        self.rho[:] = 0
        self.rho[pos_a] += 1.0
        self.rho[pos_b] -= 1.0

    def site(self, x, y, z):
        L = self.L
        return ((x % L) * L + (y % L)) * L + (z % L)


def block_err(x, nb_=10):
    x = np.asarray(x, float)
    n = (len(x) // nb_) * nb_
    if n < nb_:
        return float(np.mean(x)), float("nan")
    b = x[:n].reshape(nb_, -1).mean(axis=1)
    return float(b.mean()), float(b.std(ddof=1) / np.sqrt(nb_))
