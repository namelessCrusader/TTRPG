"""Heat and moisture through the THICKNESS of a thing thinner than a cell (T2, spec 5.1-5.5).

A lattice cell is 5 cm. A blade is 2 mm thick, a hide 4 mm. Every law that takes time
reads the cell, so the time is wrong by (50/2)^2. This module solves the profile across
the thickness instead. It is pure: arrays in, arrays out, no World.

THE STATE IS ENTHALPY. `H` is specific enthalpy in J/kg above the 20 C reference (the
lattice's convention), one value per node. Temperature is DERIVED from it through a
material's T(H) table, which carries latent plateaus (ice at 0 C, wax at 55 C, the
magnetic bump of steel). Nothing else is state, so the books are exact.

THE STEP is backward Euler on a control volume per node:
    m_i dH_i/dt = g_(i-1)(T_(i-1) - T_i) + g_i (T_(i+1) - T_i) - port terms
with g = k A / dz and k taken per interface as the harmonic mean of the two nodes' k(T).

HOW IT IS SOLVED (wave 13 polish, decision N1; the first build failed here, see below).
 1. Conductances g are FROZEN for a solve. For frozen g the equation
        F(T) = (m/dt)(H(T) - H0) + (conduction out of each node at T) = 0
    is the gradient of a convex function of the temperatures (H(T) is increasing, the
    conduction operator is symmetric and positive), so it has ONE solution and a descent
    method cannot cycle. `_solve_fixed_g` is a damped Newton on that function: the
    direction is the linearised solve with c_app = dH/dT at the iterate, the step is the
    largest of 1, 1/2, 1/4, ... along it that has not passed the minimiser of the function on
    that line. Latent plateaus (c_app jumping by 10^4 inside a smear of hundredths of a
    kelvin) therefore cost a few extra halvings, never a cycle. Inside one linear stretch of
    the table the full step is exact and the solve ends at once.
 2. k(T) is handled by a predictor-corrector: pass 1 takes g at the start temperatures, each
    further pass at the midpoint of the start and the last pass (under-relaxed by halving
    when the passes stop contracting, which a k that jumps inside a plateau can cause),
    stopping when a pass moves no node more than OUTER_TOL or when g no longer changes (every
    constant-k row is one solve). The result is an exact backward-Euler step for the last
    pass's g; the maximum principle holds for ANY positive g (M-matrix with a monotone T(H)).
 3. H is set from the FLUXES of the final solve (flux form): every flux leaves one node and
    enters the next in the same number, so the energy books close to float64 round-off at any
    pass count. T(H) then agrees with the solved T to INNER_TOL kelvin.
THE WORK IS BOUNDED. A row spends at most WORK_MAX tridiagonal solves per substep (typically
1-10, and one for a constant-k row away from a plateau; the worst seen in 3000 random draws
and 2700 grid cases was 63). A row that runs out is redone as two half steps, each an exact step of its own,
BISECT_MAX deep at most: a hard bound of (2^(BISECT_MAX+1) - 1) x WORK_MAX solves per row per
substep, at most NSUB_CAP substeps a tick. `stats()` counts every capped, redone and
unconverged row. MEASURED (numpy 2.5, one core): a step of ONE 7-node row costs 0.35 ms
(water, one solve) to 1.7 ms (steel, 8 solves across its magnetic bump), 13 nodes 0.4 to 2 ms;
a batch of a thousand rows costs 3 to 13 microseconds a row (the Python glue is shared). Over
3000 random draws (60000 ticks; all materials, dt 5 ms to 1 s, h 2 to 160000, 0.1 to 20 mm,
start and bath from -80 to 160 C across both plateaus) a record's tick costs 0.59 ms at the
median, 0.76 ms on average and 4.1 ms at the 99th percentile; the worst tick seen was 38 ms
(a thin steel foil heated hundreds of kelvin in one tick, up to 32 substeps, the NSUB_CAP).
THE FIRST BUILD recursed to 12 halvings with no work limit when a linearised Newton fell into
a limit cycle at a plateau: a 1.4 mm water film at h 5900 crossing both the 100 C and 0 C
plateaus cost 17-22 s a tick, and extreme sweeps left up to 2333 unconverged leaf rows.

NODE MASSES and CONDUCTANCES. A SYMMETRIC record (both broad faces meet the same port)
stores the half thickness, nodes from the centre (node 0) to the face (node N-1); each
node holds the grams of the node AND its mirror image, and `mult = 2` doubles every
conductance and the port, which is the same equation as the full plate. A full record
stores 2n-1 nodes over the whole thickness, `mult = 1`, a port on each end.

SUBSTEPS. A node of 0.1 mm face thickness with 180 J/m2K and h = 3000 drops 250 K in a
25 ms tick, so a quench tick is commonly several substeps: any row whose first trial step
moves a node more than DT_MAX (50 K) is redone with ceil(dT/50) substeps, and the first
tick after a port change takes 4 (it also fixes the early surface-temperature error of
first-order Euler).

Everything takes any material with the Material interface: `T_of_H(H)`, `c_app(H)`,
`k(T)`, `rho`. `Material` is a piecewise-linear table; `CallableMaterial` wraps any
monotone T(H) function.

References: Incropera et al., Fundamentals of Heat and Mass Transfer (plate roots and
Heisler series, Table 5.1); Crank, The Mathematics of Diffusion (plate series, ch. 4);
Churchill and Chu 1975 (natural convection); Voller and Swaminathan 1991 (enthalpy
method, apparent heat capacity)."""
import math

import numpy as np

T_REF = 20.0            # C, the reference where H = 0
DT_MAX = 50.0           # K, a node that moves more than this in a tick is substepped
NSUB_CAP = 32
NEWTON = 32             # outer (conductance) passes per substep, at most (8 left a node on the melting ramp of water 0.5 K from the settled state; 16 left some; 32 left none in 400 random draws)
OUTER_TOL = 2e-3        # K: a finished row whose last CORRECTOR pass still moved a node more than this is 'unsettled'
G_TOL = 1e-9            # relative: conductances this close to the last pass's need no new solve
INNER_TOL = 1e-7        # K: a solve is done when its fluxes and its own H(T) agree to this
ROUNDOFF = 1e-15        # x the sum of the sizes of the terms of a residual (a few ulps): what its round-off allows
LS_MAX = 40             # step halvings of the line search, at most
LS_SLACK = 1e-9         # the line search accepts a step whose slope is within round-off of 0
WORK_MAX = 100          # HARD BUDGET: tridiagonal solves per row per substep
BISECT_MAX = 2          # a row that spent its budget is redone in halves, this deep at most
FIRST_TICK_SUBSTEPS = 4
STATS = {"capped": 0, "bisected": 0, "unconverged": 0, "unsettled": 0, "rows": 0, "solves": 0, "max_solves": 0}   # the convergence report (see `stats`)
QUIET_EPS = 6e-8        # ~2^-24: below the float32 resolution of the host cell's energy


# ── materials ────────────────────────────────────────────────────────────────
class Material:
    """Piecewise-linear enthalpy table H(T) (J/kg, T in C, both strictly increasing) with
    k(T). A latent change is a steep segment (a smear of a fraction of a kelvin), so H(T)
    is a bijection and T(H) is continuous: the 'plateau' is a range of H at one T."""

    def __init__(self, name, rho, Tk, Hk, k):
        self.name = name
        self.rho = float(rho)
        self.Tk = np.asarray(Tk, dtype=np.float64)
        self.Hk = np.asarray(Hk, dtype=np.float64)
        assert self.Tk.shape == self.Hk.shape and self.Tk.size >= 2
        assert (np.diff(self.Tk) > 0).all() and (np.diff(self.Hk) > 0).all()
        self.slope = np.diff(self.Hk) / np.diff(self.Tk)         # c_app per segment
        self._k = k if callable(k) else (lambda T, _k=float(k): np.full_like(T, _k))

    def _seg_H(self, H):
        return np.clip(np.searchsorted(self.Hk, H, side="right") - 1, 0, self.Hk.size - 2)

    def T_of_H(self, H):
        i = self._seg_H(H)
        return self.Tk[i] + (H - self.Hk[i]) / self.slope[i]

    def c_app(self, H):
        return self.slope[self._seg_H(H)]

    def H_of_T(self, T):
        T = np.asarray(T, dtype=np.float64)
        i = np.clip(np.searchsorted(self.Tk, T, side="right") - 1, 0, self.Tk.size - 2)
        return self.Hk[i] + (T - self.Tk[i]) * self.slope[i]

    def k(self, T):
        return self._k(T)

    @staticmethod
    def constant(name, rho, c, k, T_lo=-200.0, T_hi=3000.0):
        """Constant properties; H(20 C) = 0. The case every closed form is written for."""
        return Material(name, rho, [T_lo, T_hi],
                        [c * (T_lo - T_REF), c * (T_hi - T_REF)], k)


class CallableMaterial:
    """Any monotone T(H) callable (and k(T)), as the spec asks. c_app = dH/dT is taken by a
    central difference of the callable, clamped so a flat stretch cannot divide by zero."""

    def __init__(self, T_of_H, k, rho=1000.0, dH=10.0, c_min=1.0, name="callable"):
        self._T, self._kf, self.rho, self.dH, self.c_min, self.name = \
            T_of_H, k, float(rho), float(dH), float(c_min), name

    def T_of_H(self, H):
        return self._T(H)

    def c_app(self, H):
        d = self._T(H + self.dH) - self._T(H - self.dH)
        return np.maximum(2.0 * self.dH / np.maximum(d, 1e-300), self.c_min)

    def k(self, T):
        return self._kf(T) if callable(self._kf) else np.full_like(T, float(self._kf))

    def H_of_T(self, T):
        """The inverse of the callable, by bracketed bisection (the callable is monotone)."""
        T = np.asarray(T, dtype=np.float64)
        lo = np.full_like(T, -1.0e6)
        hi = np.full_like(T, 1.0e6)
        for _ in range(60):                                   # widen until both ends bracket
            below = self._T(lo) > T
            above = self._T(hi) < T
            if not (below.any() or above.any()):
                break
            lo = np.where(below, 2.0 * lo, lo)
            hi = np.where(above, 2.0 * hi, hi)
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            up = self._T(mid) < T
            lo = np.where(up, mid, lo)
            hi = np.where(up, hi, mid)
        return 0.5 * (lo + hi)


# ── the tridiagonal solver (its own 30 lines; skinburn keeps a separate copy) ─
def tridiag_solve(a, b, c, d):
    """Thomas algorithm, batched over rows: a (sub, a[:,0] unused), b (diag), c (super,
    c[:,-1] unused), d (rhs), all (K, N). A fixed loop over nodes, vectorised over rows:
    deterministic. Diagonally dominant systems only (every system here is)."""
    K, n = b.shape
    cp = np.empty_like(b)
    dp = np.empty_like(b)
    cp[:, 0] = c[:, 0] / b[:, 0]
    dp[:, 0] = d[:, 0] / b[:, 0]
    for i in range(1, n):
        den = b[:, i] - a[:, i] * cp[:, i - 1]
        cp[:, i] = c[:, i] / den
        dp[:, i] = (d[:, i] - a[:, i] * dp[:, i - 1]) / den
    x = np.empty_like(b)
    x[:, -1] = dp[:, -1]
    for i in range(n - 2, -1, -1):
        x[:, i] = dp[:, i] - cp[:, i] * x[:, i + 1]
    return x


# ── geometry ─────────────────────────────────────────────────────────────────
def node_fractions(n, sym):
    """Share of the record's volume held by each stored node. Symmetric (half profile
    plus mirror): [1, 2, ..., 2, 1] / (2 (n-1)); full profile: [.5, 1, ..., 1, .5]/(n-1)."""
    f = np.ones(n)
    if sym:
        f[1:-1] = 2.0
        return f / (2.0 * (n - 1))
    f[0] = f[-1] = 0.5
    return f / (n - 1)


def node_spacing(thick, n, sym):
    """Distance between neighbouring nodes (m)."""
    return np.asarray(thick, dtype=np.float64) / ((n - 1) * (2.0 if sym else 1.0))


def conduction_gk(area, thick, n, sym):
    """A * mult / dz per row: multiply by k (W/m K) for the node-to-node conductance."""
    return np.asarray(area, dtype=np.float64) * (2.0 if sym else 1.0) / node_spacing(thick, n, sym)


def port_conductance(h, area, ascale, sym):
    """W/K of a port: h A a_scale, doubled for a symmetric record (two faces)."""
    return np.asarray(h) * np.asarray(area) * np.asarray(ascale) * (2.0 if sym else 1.0)


# ── the heat step ────────────────────────────────────────────────────────────
def _k_if(kn):
    """k at the interface of neighbouring nodes: the HARMONIC mean, which is the two
    half-cells' resistances in series (exact for a step in k, such as ice against water
    across a melting front; the arithmetic mean over-conducts there by up to 6 percent)."""
    a, b = kn[:, :-1], kn[:, 1:]
    return 2.0 * a * b / (a + b)


def _k_mul(T, g, Gp, Tenv, mag=False):
    """Heat that leaves each node through its conductances, W: (K T)_i - rhs_i of the linear
    conduction operator at fixed conductances g (K,N-1) and ports Gp, Tenv (K,2). With
    mag=True it also returns the sum of the absolute values of the terms (the size the
    round-off of the sum scales with)."""
    out = np.zeros_like(T)
    f = g * (T[:, :-1] - T[:, 1:])                                    # i -> i+1
    out[:, :-1] += f
    out[:, 1:] -= f
    lo = Gp[:, 0] * (T[:, 0] - Tenv[:, 0])
    hi = Gp[:, 1] * (T[:, -1] - Tenv[:, 1])
    out[:, 0] += lo
    out[:, -1] += hi
    if not mag:
        return out
    a = np.zeros_like(T)                # round-off lives in the subtraction T_i - T_j, not in its result
    af = g * (np.abs(T[:, :-1]) + np.abs(T[:, 1:]))
    a[:, :-1] += af
    a[:, 1:] += af
    a[:, 0] += Gp[:, 0] * (np.abs(T[:, 0]) + np.abs(Tenv[:, 0]))
    a[:, -1] += Gp[:, 1] * (np.abs(T[:, -1]) + np.abs(Tenv[:, 1]))
    return out, a


def _resid(T, H0, mdt, g, Gp, Tenv, mat, mag=False):
    """F(T) = (m/dt)(H(T) - H0) + out(T) at fixed conductances: the backward-Euler residual in
    watts, and the enthalpy H(T) it was made with. F is the GRADIENT of a convex function of T
    (H(T) is increasing and the conduction operator is symmetric), so F = 0 is its unique
    minimiser and a descent method cannot cycle. mag=True also returns the round-off scale of
    F in watts (the sum of the absolute sizes of its terms)."""
    Hm = mat.H_of_T(T)
    if not mag:
        return mdt * (Hm - H0) + _k_mul(T, g, Gp, Tenv), Hm
    out, a = _k_mul(T, g, Gp, Tenv, True)
    return mdt * (Hm - H0) + out, Hm, a + mdt * (np.abs(Hm) + np.abs(H0))


def _solve_fixed_g(T, H0, mdt, g, Gp, Tenv, mat, budget):
    """Backward Euler at FIXED conductances, for K rows: find T with F(T) = 0.

    Damped Newton on the convex potential of F. The direction is the old linearised solve
    ((m c_app/dt) + conduction) dT = -F, with c_app taken from the table at the iterate. The
    step is the largest of 1, 1/2, 1/4, ... along that direction with d.F(T + s d) <= 0, i.e.
    the minimiser of the potential on the line has not been passed, so the potential falls on
    every step whatever the plateaus do (the old loop fell into limit cycles because it took
    the full step and put H, not T, on the far side of a plateau). Inside one linear stretch of
    the table the full step is exact, so the rows finish in a few solves. A row is done when
    its fluxes and its own enthalpy agree: |H(T) - H_flux| <= INNER_TOL x c_app (kelvin).

    Returns (T, converged (K,), solves (K,)). A row stops after `budget` (K,) tridiagonal
    solves (the work budget)."""
    K = T.shape[0]
    T = T.copy()
    conv = np.zeros(K, dtype=bool)
    solves = np.zeros(K, dtype=np.int64)
    act = np.arange(K)
    while act.size:
        Tc = T[act]
        F, Hm, sc = _resid(Tc, H0[act], mdt[act], g[act], Gp[act], Tenv[act], mat, True)
        ca = mat.c_app(Hm)
        # done when F is under INNER_TOL kelvin of enthalpy, or under the round-off of its own
        # terms (a stiff port: 2e6 W/K times a 1e-14 K rounding of T - Tenv is 2e-8 W)
        done = (np.abs(F) <= INNER_TOL * ca * mdt[act] + ROUNDOFF * sc).all(axis=1)
        conv[act[done]] = True
        keep = ~done & (solves[act] < budget[act])
        if not keep.any():
            break
        act = act[keep]
        Tc, F, ca = Tc[keep], F[keep], ca[keep]
        mdt_a, g_a, Gp_a, Te_a, H0_a = mdt[act], g[act], Gp[act], Tenv[act], H0[act]
        b = mdt_a * ca
        b[:, :-1] += g_a
        b[:, 1:] += g_a
        b[:, 0] += Gp_a[:, 0]
        b[:, -1] += Gp_a[:, 1]
        off = np.zeros_like(b)
        off[:, 1:] = -g_a
        sup = np.zeros_like(b)
        sup[:, :-1] = -g_a
        d = tridiag_solve(off, b, sup, -F)
        solves[act] += 1
        p0 = (d * F).sum(axis=1)                                  # < 0: Newton is a descent direction
        s = np.ones(act.size)
        todo = np.arange(act.size)
        for _ in range(LS_MAX + 1):
            Tt = Tc[todo] + s[todo, None] * d[todo]
            Ft, _h = _resid(Tt, H0_a[todo], mdt_a[todo], g_a[todo], Gp_a[todo], Te_a[todo], mat)
            ok = (d[todo] * Ft).sum(axis=1) <= -LS_SLACK * p0[todo]
            todo = todo[~ok]
            if todo.size == 0:
                break
            s[todo] *= 0.5
        T[act] = Tc + s[:, None] * d
    return T, conv, solves


def _substep(H0, m, gk, Gp, Tenv, mat, dt, newton, max_solves=WORK_MAX):
    """One backward-Euler substep of K rows. Gp, Tenv: (K,2) port conductance (W/K) and
    environment temperature on node 0 and node N-1. Returns H_new, the J that left through
    each port ((K,2), positive out), ok (K,): the row's solve finished inside its budget, and
    unsettled (K,): ok, and the last conductance pass (a corrector, pass 2 or later) still moved a
    node more than OUTER_TOL. It is False for a row that is not ok and for a one-pass call.
    Energy: sum m (H_new - H0) = -sum(Q) exactly.

    The scheme. The conductances depend on T through k(T). Pass 1 takes them at the temperatures
    at the start of the substep and solves the backward-Euler equation EXACTLY for them
    (`_solve_fixed_g`). Each further pass (at most `newton` in all; none when the new
    conductances equal the last ones, which is every constant-k row; and the passes stop as soon
    as one moves no node more than OUTER_TOL) takes them at the MIDPOINT
    of the start temperatures and the last pass's, and solves again from there. That is a
    predictor-corrector on k: second order in the change of k over the step, and unlike an
    iteration to the end-of-step k it cannot flip-flop when k jumps inside a latent plateau (a
    node on the plateau changes T by a thousandth of a kelvin and k by a factor of 4; the old
    loop and a plain lagged iteration both fall into that cycle). The step is a valid
    backward-Euler state with conductances lagged by at most the last pass: for ANY positive
    conductances the system is an M-matrix with a monotone T(H), so the maximum principle holds
    whatever the pass count is. H is then set from the FLUXES of the final solve (flux form):
    every flux leaves one node and enters the next in the same number, so the energy books
    close to round-off. A row whose last corrector pass still moved a node more than OUTER_TOL
    (k not settled inside `newton` passes) is flagged `unsettled` in the fourth return; the
    caller counts it only if it keeps the result.

    A row's result never depends on its batch mates (rows are solved in sub-batches and a
    finished row is not touched again). `max_solves` is the hard budget of tridiagonal solves
    per row for the whole substep; a row that runs out of it has ok = False."""
    K, N = H0.shape
    mdt = m / dt
    T0 = mat.T_of_H(H0)
    T = T0.copy()
    g = np.empty((K, N - 1))
    work = np.zeros(K, dtype=np.int64)
    open_ = np.ones(K, dtype=bool)              # rows still taking passes
    ok = np.ones(K, dtype=bool)
    unsettled = np.zeros(K, dtype=bool)
    omega = np.ones(K)                          # under-relaxation of the conductances (1 = none)
    last_move = np.full(K, np.inf)
    for p in range(max(int(newton), 1)):
        idx = np.flatnonzero(open_)
        if idx.size == 0:
            break
        Tk = T0[idx] if p == 0 else 0.5 * (T0[idx] + T[idx])
        gn = _k_if(mat.k(Tk)) * gk[idx, None]
        if p:
            gn = g[idx] + omega[idx, None] * (gn - g[idx])
            same = (np.abs(gn - g[idx]) <= G_TOL * g[idx]).all(axis=1)     # nothing new to solve
            open_[idx[same]] = False
            unsettled[idx[same]] = False
            idx, gn = idx[~same], gn[~same]
            if idx.size == 0:
                break
        g[idx] = gn
        Tn, conv, sv = _solve_fixed_g(T[idx], H0[idx], mdt[idx], g[idx], Gp[idx], Tenv[idx], mat,
                                      max_solves - work[idx])
        move = np.abs(Tn - T[idx]).max(axis=1)
        if p:                                                # pass 0 moves from the START state: that is the step, not a k lag
            unsettled[idx] = move > OUTER_TOL
        stuck = move > 0.5 * last_move[idx]                  # not contracting: a 2-cycle in k
        omega[idx[stuck]] *= 0.5
        last_move[idx] = move
        if p:
            open_[idx[~unsettled[idx]]] = False                  # the corrector moved nothing: done
        T[idx] = Tn
        work[idx] += sv
        ok[idx[~conv]] = False
        open_[idx[~conv]] = False
    f = g * (T[:, :-1] - T[:, 1:])                                     # i -> i+1, W
    net = np.zeros_like(T)
    net[:, :-1] -= f
    net[:, 1:] += f
    lo = Gp[:, 0] * (T[:, 0] - Tenv[:, 0])
    hi = Gp[:, 1] * (T[:, -1] - Tenv[:, 1])
    net[:, 0] -= lo
    net[:, -1] -= hi
    STATS["solves"] += int(work.sum())                                 # work done counts, kept or not
    STATS["max_solves"] = max(STATS["max_solves"], int(work.max()))
    return H0 + dt * net / m, np.stack([lo * dt, hi * dt], axis=1), ok, unsettled & ok


def _robust_substep(H0, m, gk, Gp, Tenv, mat, dt, newton, depth=0, work=WORK_MAX):
    """`_substep`, with the fall-back for a row that ran out of its work budget: it is redone
    as two half steps (recursively, BISECT_MAX deep). Each half is an exact backward-Euler step
    of its own, so the books stay exact and the state stays a valid backward-Euler state; the
    budget makes the cost bounded (the total is at most the sum of the budgets of the halves,
    2^(BISECT_MAX+1) - 1 times WORK_MAX solves, per row, per substep). A row that is still not
    finished at the depth limit is counted in STATS['unconverged'] and returned as the flux
    form of its last iterate (energy exact, the maximum principle only to its residual)."""
    H, Q, ok, uns = _substep(H0, m, gk, Gp, Tenv, mat, dt, newton, work)
    if ok.all():
        STATS["rows"] += int(ok.size)
        STATS["unsettled"] += int(uns.sum())
        return H, Q
    bad = np.flatnonzero(~ok)
    if depth >= BISECT_MAX:                          # kept as it is: counted as unconverged, not as unsettled
        STATS["unconverged"] += int(bad.size)
        STATS["rows"] += int(ok.size)
        STATS["unsettled"] += int(uns.sum())
        return H, Q
    STATS["capped"] += int(bad.size)
    if depth == 0:
        STATS["bisected"] += int(bad.size)
    good = np.flatnonzero(ok)                        # the rows kept from this attempt are counted here; `bad` rows by their halves
    STATS["rows"] += int(good.size)
    STATS["unsettled"] += int(uns[good].sum())
    h = dt / 2.0
    sub = (m[bad], gk[bad], Gp[bad], Tenv[bad])
    Hb, Qb = _robust_substep(H0[bad], *sub, mat, h, newton, depth + 1, work)
    Hb, Q2 = _robust_substep(Hb, *sub, mat, h, newton, depth + 1, work)
    H = H.copy()
    Q = Q.copy()
    H[bad] = Hb
    Q[bad] = Qb + Q2
    return H, Q


def stats(reset=False):
    """The convergence report. 'capped': rows that ran out of their work budget; 'bisected':
    rows redone as shorter steps; 'unconverged': rows still not finished at the depth limit;
    'unsettled': KEPT row-substeps (see 'rows') that finished their solves but whose last
    conductance (k) corrector pass still moved a node by more than OUTER_TOL kelvin, so k was
    not settled inside `newton` passes. The result is still an exact backward-Euler state for the
    conductances of that pass: energy exact, maximum principle holds; only the lag of k is left
    (disjoint from 'unconverged'; a constant-k row and a one-pass call are never counted); 'rows':
    the row-substeps whose results were kept (the denominator of 'unsettled'; a row redone in
    halves counts its halves); 'solves': total tridiagonal solves (including attempts that were
    thrown away); 'max_solves': the most one row used in one substep. The energy books
    cannot show a non-converged row, so this is the check for it."""
    out = dict(STATS)
    if reset:
        for k in STATS:
            STATS[k] = 0
    return out


def _advance(H0, m, gk, Gp, Tenv, mat, dt, nsub, newton, work=WORK_MAX):
    H = H0
    Q = np.zeros((H0.shape[0], 2))
    ds = dt / nsub
    for _ in range(nsub):
        H, q = _robust_substep(H, m, gk, Gp, Tenv, mat, ds, newton, 0, work)
        Q += q
    return H, Q


def step(H, m, gk, Gp, Tenv, mat, dt, *, newton=NEWTON, nsub_min=None,
         dT_max=DT_MAX, nsub_cap=NSUB_CAP, work=WORK_MAX):
    """Advance K rows of N nodes by dt. H (K,N) J/kg, m (K,N) kg per node, gk (K,) from
    `conduction_gk`, Gp (K,2) W/K, Tenv (K,2) C. `nsub_min` (K,) forces substeps (4 on the
    first tick after a port change). Returns (H_new, Q (K,2) J out through each port,
    nsub (K,)). Energy exact to round-off; a row's result does not depend on the other
    rows of the batch."""
    H = np.asarray(H, dtype=np.float64)
    K = H.shape[0]
    Hn, Q = _advance(H, m, gk, Gp, Tenv, mat, dt, 1, newton, work)
    jump = np.abs(mat.T_of_H(Hn) - mat.T_of_H(H)).max(axis=1)
    need = np.clip(np.ceil(jump / dT_max), 1, nsub_cap).astype(np.int64)
    if nsub_min is not None:
        need = np.maximum(need, np.asarray(nsub_min, dtype=np.int64))
    for n in np.unique(need[need > 1]):                       # sorted: deterministic
        r = np.flatnonzero(need == n)
        Hn[r], Q[r] = _advance(H[r], m[r], gk[r], Gp[r], Tenv[r], mat, dt, int(n), newton, work)
    return Hn, Q, need


def node_energy(H, m):
    """J above the reference, per row."""
    return (H * m).sum(axis=1)


def quiet_mask(H, m, gk, Gp, Tenv, mat, dt, scale_J, eps=QUIET_EPS):
    """True where the energy a step COULD move is below the float32 resolution of the
    host cell: eps x scale_J, scale_J being the host's |E| plus its capacity per kelvin.
    (v1's '1e-6 K' never held in a room with a fire: float32 neighbours wander by 1e-5 K.)
    The caller also requires no species gradient (`species_flat`)."""
    T = mat.T_of_H(H)
    kn = mat.k(T)
    g = _k_if(kn) * gk[:, None]
    inner = (g * np.abs(T[:, :-1] - T[:, 1:])).max(axis=1)
    lo = Gp[:, 0] * np.abs(T[:, 0] - Tenv[:, 0])
    hi = Gp[:, 1] * np.abs(T[:, -1] - Tenv[:, 1])
    move = np.maximum(np.maximum(inner, lo), hi) * dt
    return move <= eps * np.asarray(scale_J)


def species_flat(X, tol=1e-12):
    """True where the moisture content (or any fraction) is flat across the nodes."""
    return (X.max(axis=1) - X.min(axis=1)) <= tol * (1.0 + np.abs(X).max(axis=1))


# ── scales and numbers ───────────────────────────────────────────────────────
def biot(h, L, k):
    return np.asarray(h) * np.asarray(L) / np.asarray(k)


def l_char(volume, area):
    """V/A: the characteristic length of Bi = h L/k. A plate of thickness t: t/2."""
    return np.asarray(volume) / np.asarray(area)


def sigma_sv(volume, area):
    """Surface to volume, 1/m (what burning reads in place of the BURN_AREA fudge)."""
    return np.asarray(area) / np.asarray(volume)


def t_diff(L, alpha):
    """Diffusion time L^2/alpha (s), for heat (alpha = k/rho c) or species (D)."""
    return np.asarray(L) ** 2 / np.asarray(alpha)


def t_lumped(rho, c, L, h):
    """rho c L / h (s): the time constant of a thermally thin body."""
    return rho * c * L / np.asarray(h)


FLUIDS = {   # properties at 20 C: rho, mu, k, cp, beta (1/K). Incropera A.4, A.6 (beta of water u)
    "air": dict(rho=1.204, mu=1.825e-5, k=0.0257, cp=1007.0, beta=1.0 / 293.15),
    "water": dict(rho=998.0, mu=1.002e-3, k=0.598, cp=4182.0, beta=2.07e-4),
}


def h_conv(speed, Lc, dT, fluid):
    """Convection coefficient (W/m2K) of a wall in a fluid: forced (laminar flat plate,
    Nu = 0.664 Re^0.5 Pr^(1/3), Incropera 7.30, Re < 5e5) PLUS natural (Churchill-Chu,
    vertical plate, laminar, 0.68 + 0.67 Ra^0.25 / [1 + (0.492/Pr)^(9/16)]^(4/9)). The
    relative speed of host and fluid is in `speed`: a swung blade or a thrown cup cools
    faster. Properties come from the FLUID row (data), not a branch on the fluid."""
    f = FLUIDS[fluid] if isinstance(fluid, str) else fluid
    Pr = f["mu"] * f["cp"] / f["k"]
    Re = f["rho"] * np.abs(speed) * Lc / f["mu"]
    nu_f = 0.664 * np.sqrt(Re) * Pr ** (1.0 / 3.0)
    Ra = 9.81 * f["beta"] * np.abs(dT) * Lc ** 3 * f["rho"] ** 2 * f["cp"] / (f["mu"] * f["k"])
    nu_n = 0.68 + 0.67 * Ra ** 0.25 / (1.0 + (0.492 / Pr) ** (9.0 / 16.0)) ** (4.0 / 9.0)
    return (nu_f + nu_n) * f["k"] / Lc


# ── closed forms: the plate series ───────────────────────────────────────────
def plate_roots(Bi, nterms=12):
    """Roots zeta_n of zeta tan(zeta) = Bi, n = 0..nterms-1 (Bi = inf: (n + 1/2) pi)."""
    if math.isinf(Bi):
        return (np.arange(nterms) + 0.5) * math.pi
    out = np.empty(nterms)
    for n in range(nterms):
        lo, hi = n * math.pi + 1e-12, n * math.pi + math.pi / 2 - 1e-12
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if mid * math.tan(mid) < Bi:
                lo = mid
            else:
                hi = mid
        out[n] = 0.5 * (lo + hi)
    return out


def plate_theta(xi, Fo, Bi, nterms=12):
    """Dimensionless (T - Tinf)/(Ti - Tinf) at x/L = xi in a plate of half thickness L, both
    faces in a fluid with Biot number Bi = h L/k, Fourier number Fo = alpha t/L^2:
        theta = sum C_n exp(-zeta_n^2 Fo) cos(zeta_n xi),  C_n = 4 sin z/(2 z + sin 2z).
    Bi = inf is the Dirichlet plate (species with a sorption boundary)."""
    z = plate_roots(Bi, nterms)
    C = 4.0 * np.sin(z) / (2.0 * z + np.sin(2.0 * z))
    xi = np.asarray(xi, dtype=np.float64)
    return sum(C[n] * math.exp(-z[n] ** 2 * Fo) * np.cos(z[n] * xi) for n in range(nterms))


def plate_mean_theta(Fo, Bi, nterms=12):
    """Slab-average theta (the moisture or heat content left): sum C_n sin(z)/z exp(-z^2 Fo)."""
    z = plate_roots(Bi, nterms)
    C = 4.0 * np.sin(z) / (2.0 * z + np.sin(2.0 * z))
    return sum(C[n] * math.sin(z[n]) / z[n] * math.exp(-z[n] ** 2 * Fo) for n in range(nterms))


def stefan_plate_time(rho, Lf, L, h, k_liquid, dT):
    """Quasi-steady Stefan time (s) to melt (or freeze) a plate of half thickness L at its
    transition temperature from one face, h to an environment dT beyond it, the other face
    adiabatic or the mirror: rho Lf L (1/h + L/(2 k)) / dT. Exact as the Stefan number
    c dT/Lf goes to 0 (the sensible heat of the liquid layer is neglected)."""
    return rho * Lf * (L / h + L * L / (2.0 * k_liquid)) / dT


# ── species through the thickness (moisture in hide, wood, grain) ────────────
def diffuse_step(Wd, Md, rho_dry, gk, D_of, T, Gm, Xeq, Hn, dt, picard=2):
    """Moisture (any one mobile species) through the thickness, backward Euler, flux form.

    Wd (K,N) kg of the mobile species per node; Md (K,N) kg of everything else (the
    matrix, held fixed); rho_dry (K,N) kg of matrix per m3 of node; gk (K,) = A mult/dz;
    D_of(X, T) -> m2/s with X = Wd/Md the dry-basis content (the table row: moisture in
    hide, wood and grain changes D by 10-100x); T (K,N) C; Gm (K,2) = beta A mult rho_dry
    kg/s per unit X at each port and Xeq (K,2) the equilibrium content there, from the
    SORPTION ISOTHERM, not an equal fraction (Gm = 0 closes a port); Hn (K,N) specific
    enthalpy the water carries when it moves (upwind), so energy is exact too.

    Returns (W_new, H_new, out (K,2) kg through the ports, X_new). Mass: sum W_new + out
    = sum Wd. Energy: the water carries H of its source node; the total m H is unchanged
    except for what the ports take, out_i H_end, which the caller books."""
    K, N = Wd.shape
    X0 = Wd / Md
    Xk = X0
    for _ in range(picard):
        Dn = D_of(Xk, T)
        rho_if = 0.5 * (rho_dry[:, :-1] + rho_dry[:, 1:])
        c = 0.5 * (Dn[:, :-1] + Dn[:, 1:]) * rho_if * gk[:, None]          # kg/s per unit X
        C = Md / dt
        b = C.copy()
        b[:, :-1] += c
        b[:, 1:] += c
        b[:, 0] += Gm[:, 0]
        b[:, -1] += Gm[:, 1]
        d = C * X0
        d[:, 0] += Gm[:, 0] * Xeq[:, 0]
        d[:, -1] += Gm[:, 1] * Xeq[:, 1]
        off = np.zeros_like(b)
        off[:, 1:] = -c
        sup = np.zeros_like(b)
        sup[:, :-1] = -c
        Xk = tridiag_solve(off, b, sup, d)
    F = c * (Xk[:, :-1] - Xk[:, 1:]) * dt                                  # kg, i -> i+1
    out = np.stack([Gm[:, 0] * (Xk[:, 0] - Xeq[:, 0]) * dt,
                    Gm[:, 1] * (Xk[:, -1] - Xeq[:, 1]) * dt], axis=1)
    dm = np.zeros_like(Wd)
    dm[:, :-1] -= F
    dm[:, 1:] += F
    dm[:, 0] -= out[:, 0]
    dm[:, -1] -= out[:, 1]
    # clip round-off only: the implicit solve keeps X >= 0
    Wn = np.maximum(Wd + dm, 0.0)
    # energy: the water carries the enthalpy of its source node
    Hup = np.where(F > 0, Hn[:, :-1], Hn[:, 1:])
    dE = np.zeros_like(Wd)
    dE[:, :-1] -= F * Hup
    dE[:, 1:] += F * Hup
    dE[:, 0] -= out[:, 0] * Hn[:, 0]
    dE[:, -1] -= out[:, 1] * Hn[:, -1]
    mt = Md + Wd
    mn = Md + Wn
    Hnew = (mt * Hn + dE) / mn
    return Wn, Hnew, out, Xk
