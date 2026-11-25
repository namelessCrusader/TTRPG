"""4D Euclidean compact U(1) Wilson action, Metropolis. The quantum ground-state analogue of the Hamiltonian model
(control for confinement). Creutz, Jacobs, Rebbi 1979: first-order-like transition near beta = 1.01."""
import numpy as np, numba as nb
TWO_PI = 2*np.pi

def make_lat4(L):
    N = L**4
    idx = np.arange(N).reshape(L, L, L, L)
    f = np.empty((4, N), np.int64); b = np.empty((4, N), np.int64)
    for m in range(4):
        f[m] = np.roll(idx, -1, axis=m).ravel(); b[m] = np.roll(idx, 1, axis=m).ravel()
    return f, b

@nb.njit(cache=True)
def _wrap(a):
    return a - TWO_PI*np.floor((a+np.pi)/TWO_PI)

@nb.njit(cache=True)
def sweep(th, f, b, beta, delta, nhit):
    N = th.shape[1]; acc = 0
    for s in range(N):
        for mu in range(4):
            zr = 0.0; zi = 0.0
            for nu in range(4):
                if nu == mu: continue
                smu = f[mu, s]; snu = f[nu, s]; sbn = b[nu, s]
                a_up = th[nu, smu] - th[mu, snu] - th[nu, s]
                sm = f[mu, sbn]
                a_lo = -(th[mu, sbn] + th[nu, sm] - th[nu, sbn])
                zr += np.cos(a_up) + np.cos(a_lo); zi += np.sin(a_up) + np.sin(a_lo)
            for h in range(nhit):
                old = th[mu, s]; new = old + delta*(2*np.random.random()-1)
                # action term: -beta*Re(e^{i th} Z)
                dS = -beta*((np.cos(new)-np.cos(old))*zr - (np.sin(new)-np.sin(old))*zi)
                if dS <= 0 or np.random.random() < np.exp(-dS):
                    th[mu, s] = _wrap(new); acc += 1
    return acc

@nb.njit(cache=True)
def seed_nb(x):
    np.random.seed(x)

@nb.njit(cache=True)
def plaq(th, f, mu, nu, s):
    return th[mu, s] + th[nu, f[mu, s]] - th[mu, f[nu, s]] - th[nu, s]

@nb.njit(cache=True)
def wilson2(th, f, b, R, T):
    N = th.shape[1]; tot = 0.0; n = 0
    for s in range(N):
        for mu in range(4):
            for nu in range(mu+1, 4):
                for orient in range(2):
                    m1, m2 = (mu, nu) if orient == 0 else (nu, mu)
                    a = 0.0; p = s
                    for i in range(R):
                        a += th[m1, p]; p = f[m1, p]
                    for i in range(T):
                        a += th[m2, p]; p = f[m2, p]
                    for i in range(R):
                        p = b[m1, p]; a -= th[m1, p]
                    for i in range(T):
                        p = b[m2, p]; a -= th[m2, p]
                    tot += np.cos(a); n += 1
    return tot/n

@nb.njit(cache=True)
def monopole_density(th, f):
    """fraction of 3-cubes (dual links) carrying nonzero monopole charge"""
    N = th.shape[1]; cnt = 0; tot = 0
    for s in range(N):
        for mu in range(4):
            d = np.empty(3, np.int64); k = 0
            for nu in range(4):
                if nu != mu:
                    d[k] = nu; k += 1
            a, bb, c = d[0], d[1], d[2]
            t = 0.0
            for (x, y, z) in ((a, bb, c), (bb, c, a), (c, a, bb)):
                t += _wrap(plaq(th, f, y, z, f[x, s])) - _wrap(plaq(th, f, y, z, s))
            tot += 1
            if abs(round(t/TWO_PI)) > 0: cnt += 1
    return cnt/tot

def run(L, beta, ntherm=400, nmeas=1500, every=5, seed=1, delta=0.6, nhit=3, loops=((1,1),(1,2),(2,1),(2,2),(3,3),(2,3),(3,2))):
    f, b = make_lat4(L)
    th = np.zeros((4, L**4)); seed_nb(seed)
    rng = np.random.default_rng(seed)
    th[:] = rng.uniform(-np.pi, np.pi, th.shape) if beta < 1.0 else 0.0   # hot start below, cold above
    for i in range(ntherm): sweep(th, f, b, beta, delta, nhit)
    W = {l: [] for l in loops}; M = []; P = []
    for i in range(nmeas):
        sweep(th, f, b, beta, delta, nhit)
        if i % every == 0:
            for l in loops: W[l].append(wilson2(th, f, b, *l))
            M.append(monopole_density(th, f))
    return {"W": {f"{a}x{c}": np.array(v) for (a, c), v in W.items()}, "mono": np.array(M)}
