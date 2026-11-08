"""TRANSPORT IN POROUS SOLIDS (wave 13, T6): water and what it carries, cell to cell.

Pore water between touching solid cells moves by a WATER POTENTIAL psi (= ln a_w in
the bound range), not by content, so two unlike porous solids at the same a_w
exchange nothing (wood at 12 % beside hide or soil). The flux across a face is

    F = kappa_f (psi_i - psi_j) / dx   [g/(m2 s)],   kappa_f = harmonic(kappa_i, kappa_j)

with kappa = D_app rho_dry (dw / d ln a_w) in g/(m s) per unit psi. The step is in
flux form, so what leaves one cell enters the next: the total is conserved to
rounding, and it keeps the maximum principle while dt is under the stability
limit. Above that limit the step substeps (it never extrapolates).

Solutes ride the water flux: the concentration of the DONOR cell times the water that
crosses (upwind). Drying a salty wet solid carries salt to the surface, wicking
carries sugar up a rope. One apparent diffusivity D_app is used and the vapour path
is inside it (Wood Handbook figures already contain it); R_int, the resistance the
dry skin adds to the surface law, comes from the same D.

FLOAT32 BITE RULE. Pore water sits in a float32 array (fvol); a 9 ml cell has an
ulp of 9.5e-7 g and wood-scale fluxes are 6e-8 to 1e-6 g/s, so a one-second step
rounds away to nothing. A transport box therefore runs on its own clock, N ticks
long, chosen so the largest bite is at least 16 ulp of the receiving cell
(bite_clock), capped by the stability limit and by one hour; below that the box sleeps.

Plain numpy is the definition. Units: grams, metres, seconds. No World.
"""
from typing import NamedTuple, Optional

import numpy as np

from .matterdata import D25, E_D, R, TORTUOSITY


def harmonic(a, b):
    """Harmonic mean at a face; zero if either side is zero."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    s = a + b
    return np.where(s > 0.0, 2.0 * a * b / np.where(s > 0.0, s, 1.0), 0.0)


def arrhenius(D_ref, E, T_c, T_ref_c=20.0):
    """D at T_c from D_ref at T_ref_c: D_ref exp(-E/R (1/T - 1/T_ref)). Factor 1.6 from
    20 to 40 C at 18 kJ/mol (c)."""
    T = np.asarray(T_c, np.float64) + 273.15
    return np.asarray(D_ref, np.float64) * np.exp(-E / R * (1.0 / T - 1.0 / (T_ref_c + 273.15)))


def host_D(species, host, theta, phi, T_c):
    """Effective diffusivity (m2/s) of a dissolved species in a host.
    Free water: D25 with the Arrhenius law. Inside a pore network the water-filled
    fraction theta of porosity phi gives Millington-Quirk theta^(10/3)/phi^2; for a
    tissue host with a measured row in matterdata.TORTUOSITY that row is used."""
    Dw = arrhenius(D25[species], E_D[species], T_c, 25.0)
    t = TORTUOSITY.get((species, host))
    if t is not None:
        return Dw * t
    theta = np.asarray(theta, np.float64)
    phi = np.asarray(phi, np.float64)
    return Dw * theta ** (10.0 / 3.0) / np.maximum(phi, 1e-12) ** 2


def r_int(dx, D_app, rho_dry, dwda, rho_sat_g_m3):
    """Interior resistance (s/m) a drying solid adds to the surface law:
    (dx/2) / D_int,  D_int = D_app rho_dry (dw/da_w) / rho_sat  in vapour-density
    units. rho_dry in kg/m3, rho_sat in g/m3. Wood at 10 %, 20 C: 23,100 s/m against
    1/h_m = 120 s/m (c): wood dries from the inside."""
    D_int = D_app * rho_dry * dwda / (np.asarray(rho_sat_g_m3, np.float64) * 1e-3)
    return 0.5 * dx / D_int


def series_resistance(*Rs):
    """Resistances in series (s/m) add."""
    return sum(np.asarray(r, np.float64) for r in Rs)


def ulp(x, dtype=np.float32):
    """Spacing of the float grid at x: the smallest bite a cell can feel."""
    return np.spacing(np.abs(np.asarray(x, dtype)))


def bite_clock(flux_per_tick, ulp_g, cap_ticks, n_ulp=16):
    """Ticks between steps so one bite is at least n_ulp ulp of the receiving cell.
    flux_per_tick: grams per tick (array); returns int ticks in [1, cap_ticks]. A zero
    flux gives cap_ticks (the box sleeps)."""
    f = np.abs(np.asarray(flux_per_tick, np.float64))
    need = n_ulp * np.asarray(ulp_g, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        n = np.where(f > 0.0, np.ceil(need / np.where(f > 0.0, f, 1.0)), np.inf)
    return np.clip(n, 1, cap_ticks).astype(np.int64)


class Step(NamedTuple):
    m: np.ndarray                 # new water, same dtype as the input
    s: Optional[np.ndarray]       # new solutes (or None)
    flux: list                    # per axis: grams moved from index i to i+1, whole dt
    n_sub: int                    # substeps taken
    dt_done: float                # seconds actually advanced (== dt unless max_sub hit)


def _axis_slices(nd, ax):
    lo = [slice(None)] * nd
    hi = [slice(None)] * nd
    lo[ax] = slice(0, -1)
    hi[ax] = slice(1, None)
    return tuple(lo), tuple(hi)


def _rates(psi, kap, on, dx):
    """Per axis: water rate g/s from index i to i+1, and the conductance g/s per
    unit psi of each face."""
    nd = psi.ndim
    rates, cond = [], []
    for ax in range(nd):
        lo, hi = _axis_slices(nd, ax)
        g = np.where(on[ax], harmonic(kap[lo], kap[hi]) * dx, 0.0)
        cond.append(g)
        rates.append(g * (psi[lo] - psi[hi]))
    return rates, cond


def stable_dt(cap, cond, safety=0.5):
    """Largest explicit dt (s): safety * min(cap_i / sum of conductances of cell i)."""
    nd = len(cond)
    G = np.zeros(cap.shape)
    for ax in range(nd):
        lo, hi = _axis_slices(nd, ax)
        G[lo] += cond[ax]
        G[hi] += cond[ax]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(G > 0.0, cap / np.where(G > 0.0, G, 1.0), np.inf)
    return safety * float(t.min())


def potential_flux_step(m, psi, kappa, cap, mask, dx, dt, solutes=None,
                        psi_of_m=None, max_sub=20000):
    """Advance the water m (g per cell) by dt seconds.

    psi      water potential per cell (ln a_w), the driver
    kappa    per cell, g/(m s) per unit psi; harmonic mean at each face
    cap      per cell, grams per unit psi (dm/dpsi); sets the stability limit and
             linearises psi between substeps when psi_of_m is None
    mask     bool, the cells that take part (porous solid); a face is open only
             if both sides are in the mask
    solutes  optional grams, shape m.shape + (n,); rides the water flux (upwind)
    psi_of_m optional callable m -> psi, for a nonlinear isotherm (called per substep)

    Returns a Step. Conserves total water and solute to rounding (float64 inside, one
    cast back to m.dtype at the end). A step never takes more than half of a donor's
    water (the faces it feeds are scaled down when needed), so
    nothing goes negative, whatever psi and cap the caller linearised with."""
    dt_in = m.dtype
    mf = np.asarray(m, np.float64).copy()
    ps = np.asarray(psi, np.float64).copy()
    kap = np.asarray(kappa, np.float64)
    cap = np.asarray(cap, np.float64)
    nd = mf.ndim
    sf = None if solutes is None else np.asarray(solutes, np.float64).copy()
    on = []
    for ax in range(nd):
        lo, hi = _axis_slices(nd, ax)
        on.append(mask[lo] & mask[hi])
    tot = [np.zeros(tuple(mf.shape[i] - (1 if i == ax else 0) for i in range(nd)))
           for ax in range(nd)]
    t_left, n_sub = float(dt), 0
    while t_left > 0.0 and n_sub < max_sub:
        _r0, cond = _rates(ps, kap, on, dx)
        rates = _r0
        h = min(t_left, stable_dt(cap, cond))
        out = np.zeros(mf.shape)                       # g/s leaving each cell
        for ax in range(nd):
            lo, hi = _axis_slices(nd, ax)
            r = rates[ax]
            out[lo] += np.maximum(r, 0.0)
            out[hi] += np.maximum(-r, 0.0)
        # A donor never gives more than half of what it holds this substep: scale the
        # faces it feeds (both sides see the same scaled flux, so nothing is lost). A
        # cell with no water gives nothing. The step itself is limited by stability only,
        # so a dry cell cannot stall the clock.
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(out > 0.0, np.minimum(1.0, 0.5 * np.maximum(mf, 0.0)
                                                   / np.where(out > 0.0, out * h, 1.0)), 1.0)
        rates = []
        for ax in range(nd):
            lo, hi = _axis_slices(nd, ax)
            r = _r0[ax]
            rates.append(np.where(r > 0.0, r * scale[lo], r * scale[hi]))
        dm = np.zeros(mf.shape)
        ds = None if sf is None else np.zeros(sf.shape)
        for ax in range(nd):
            lo, hi = _axis_slices(nd, ax)
            f = rates[ax] * h
            dm[lo] -= f
            dm[hi] += f
            tot[ax] += f
            if sf is not None:
                donor_lo = f > 0.0
                c_lo = sf[lo] / np.maximum(mf[lo], 1e-300)[..., None]
                c_hi = sf[hi] / np.maximum(mf[hi], 1e-300)[..., None]
                sflow = f[..., None] * np.where(donor_lo[..., None], c_lo, c_hi)
                ds[lo] -= sflow
                ds[hi] += sflow
        mf += dm
        if sf is not None:
            sf += ds
        ps = psi_of_m(mf) if psi_of_m is not None else ps + dm / np.maximum(cap, 1e-300)
        t_left -= h
        n_sub += 1
    return Step(mf.astype(dt_in), None if sf is None else sf.astype(dt_in), tot,
                n_sub, float(dt) - t_left)
