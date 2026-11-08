"""VOLATILE MIXTURES AND THE BOIL STEP (wave 13, T4, law L14).

Raoult with a real activity model (Van Laar, the ethanol pair refitted to the
real azeotrope), so a pot enriches spirit with no still code in it. The steam
carries its own latent heat per species (a wash's vapour is 46 wt percent
ethanol, so 1598 J/g, not 2260). Pure numpy, no World. T in C, P in Pa, grams.

  bubble_point(comp, P) -> (T_boil, y mass fractions of the vapour, L_eff J/g)
  rayleigh(comp, boiled_fraction, P, steps) -> (distillate, residue)
  boil_step(comp, T, C_JK, P) -> BoilOut (steam_comp, comp_new, T_new, dE, boiled_J)

boil_step is a linearised step plus a Newton pass (the 25-step bisection cost
4 ms for 16 cells). It books the joules the steam took (boiled_J) and returns
the steam by species so the host can book it; condense_J(steam) gives the joules
the same vapour returns on a cold surface (the same Lv table).

Not modelled here (stated): non-volatile solutes do not enter the bubble point of a
volatile mixture (they only raise it slightly); a ternary water + ethanol + methanol
mixture takes the water activity coefficient as the mole-share mean of the two
binaries (u); ethanol-methanol is ideal (u).
"""
import numpy as np

from . import mixdata as D
from . import mixsol as S
from . import mixwater as W
from .mixwater import prep, boiling_point, tsat_water, _psat_k

VOLATILES = ("h2o", "ethanol", "methanol")
_I = {n: D.INDEX[n] for n in VOLATILES}
_ANT = {n: D.row("antoine", n) for n in ("ethanol", "methanol")}
_VL = {n: D.row("vanlaar", n) for n in ("ethanol", "methanol")}
MMHG = D.MMHG


def psat_antoine(name, T):
    """Vapour pressure of a volatile from its Antoine row, Pa, T in C."""
    A, B, C = _ANT[name]
    return 10.0 ** (A - B / (C + np.asarray(T, dtype=float))) * MMHG


def vanlaar(x1, A12, A21):
    """Van Laar activity coefficients (g1, g2) of a binary at mole fraction x1."""
    x1 = np.asarray(x1, dtype=float)
    x2 = 1.0 - x1
    d = A12 * x1 + A21 * x2
    return np.exp(A12 * (A21 * x2 / d) ** 2), np.exp(A21 * (A12 * x1 / d) ** 2)


def _moles(c):
    return {n: c[..., _I[n]] / D.SPECIES[n].M for n in VOLATILES}


def gammas(nw, ne, nm):
    """Activity coefficients (g_w, g_e, g_m) at the given moles of water, ethanol, methanol."""
    tot = nw + ne + nm
    with np.errstate(invalid="ignore", divide="ignore"):
        gw_e = np.ones_like(tot)
        ge = np.ones_like(tot)
        gm = np.ones_like(tot)
        lnw = np.zeros_like(tot)
        share_e = np.where(ne + nm > 0.0, ne / np.where(ne + nm > 0.0, ne + nm, 1.0), 0.0)
        x1e = np.where(ne + nw > 0.0, ne / np.where(ne + nw > 0.0, ne + nw, 1.0), 0.0)
        g1, g2 = vanlaar(x1e, *_VL["ethanol"])
        ge = np.where(ne > 0.0, g1, 1.0)
        lnw = lnw + share_e * np.log(g2)
        x1m = np.where(nm + nw > 0.0, nm / np.where(nm + nw > 0.0, nm + nw, 1.0), 0.0)
        g1, g2 = vanlaar(x1m, *_VL["methanol"])
        gm = np.where(nm > 0.0, g1, 1.0)
        lnw = lnw + (1.0 - share_e) * np.log(g2) * (nm > 0.0)
        gw_e = np.exp(np.where(ne + nm > 0.0, lnw, 0.0))
    return gw_e, ge, gm


_BP_ITERS = 12


def _partials(T, nw, ne, nm, gw, ge, gm):
    """Partial pressures (Pa) of water, ethanol, methanol over the liquid at T."""
    x = nw + ne + nm
    with np.errstate(invalid="ignore", divide="ignore"):
        xw, xe, xm = nw / x, ne / x, nm / x
    pw = xw * gw * np.where(xw > 0.0, _psat_k(T + D.T0K), 0.0)
    pe = xe * ge * np.where(xe > 0.0, psat_antoine("ethanol", T), 0.0)
    pm = xm * gm * np.where(xm > 0.0, psat_antoine("methanol", T), 0.0)
    return pw, pe, pm


def bubble_point(comp, P=D.P_ATM):
    """(T_boil C, y, L_eff): the temperature where the partial pressures sum to P,
    y the vapour as MASS fractions on the full species axis, L_eff = sum y Lv in
    J/g of vapour. Only water, ethanol and methanol are volatile here; a cell with
    none of them gives NaN. Secant on ln(sum p / P)."""
    c = prep(comp)
    lead = c.shape[:-1]
    c = c.reshape(-1, c.shape[-1])
    n = _moles(c)
    nw, ne, nm = n["h2o"], n["ethanol"], n["methanol"]
    gw, ge, gm = gammas(nw, ne, nm)
    P = np.asarray(P, dtype=float)

    def F(T):
        pw, pe, pm = _partials(T, nw, ne, nm, gw, ge, gm)
        return np.log(np.maximum(pw + pe + pm, 1e-300) / P)

    t0 = np.full(nw.shape, 80.0)
    t1 = t0 + 1.0
    f0, f1 = F(t0), F(t1)
    for _ in range(_BP_ITERS):
        den = f1 - f0
        step = np.where(np.abs(den) > 1e-14, f1 * (t1 - t0) / np.where(den == 0.0, 1.0, den), 0.0)
        t2 = np.clip(t1 - step, 1.0, 300.0)
        t0, f0 = t1, f1
        t1, f1 = t2, F(t2)
    T = t1
    pw, pe, pm = _partials(T, nw, ne, nm, gw, ge, gm)
    mass = np.zeros(c.shape)
    mass[:, _I["h2o"]] = pw / P * D.SPECIES["h2o"].M
    mass[:, _I["ethanol"]] = pe / P * D.SPECIES["ethanol"].M
    mass[:, _I["methanol"]] = pm / P * D.SPECIES["methanol"].M
    tot = mass.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = mass / tot[:, None]
    Lv = np.array([D.SPECIES[x].Lv for x in D.ORDER])
    L = (y * Lv).sum(axis=1)
    bad = ~(nw + ne + nm > 0.0)
    T = np.where(bad, np.nan, T)
    return T.reshape(lead), y.reshape(lead + (c.shape[-1],)), L.reshape(lead)


def condense_J(steam_comp):
    """Joules a vapour gives back on a cold surface: sum grams * Lv by species. The
    same table boil_step books, so a still's books close."""
    return (prep(steam_comp) * np.array([D.SPECIES[x].Lv for x in D.ORDER])).sum(axis=-1)


def rayleigh(comp, boiled_fraction, P=D.P_ATM, steps=40):
    """Batch boiling off `boiled_fraction` of the non-water volatile (ethanol plus
    methanol moles), vapour never returned. Steps of equal volatile removal with a
    midpoint predictor. Returns (distillate grams, residue grams); residue = comp -
    distillate exactly, so moles are conserved to rounding."""
    c0 = prep(comp)
    lead = c0.shape[:-1]
    c0 = c0.reshape(-1, c0.shape[-1])
    M = np.array([D.SPECIES[x].M for x in D.ORDER])
    iv = [_I["ethanol"], _I["methanol"]]
    V0 = (c0[:, iv] / M[iv]).sum(axis=1)
    dV = np.asarray(boiled_fraction, dtype=float) * V0 / steps
    cur = c0.copy()
    dist = np.zeros_like(c0)
    vol_idx = [_I[x] for x in VOLATILES]
    for _ in range(steps):
        _T, y, _L = bubble_point(cur, P)
        ymol = np.zeros_like(y)
        ymol[:, vol_idx] = y[:, vol_idx] / M[vol_idx]
        ymol = ymol / np.maximum(ymol.sum(axis=1, keepdims=True), 1e-300)
        yV = ymol[:, iv].sum(axis=1)
        dn = np.where(yV > 1e-12, dV / np.where(yV > 1e-12, yV, 1.0), 0.0)
        half = np.maximum(cur - 0.5 * dn[:, None] * ymol * M, 0.0)
        _T, y2, _L = bubble_point(half, P)
        ymol2 = np.zeros_like(y2)
        ymol2[:, vol_idx] = y2[:, vol_idx] / M[vol_idx]
        ymol2 = ymol2 / np.maximum(ymol2.sum(axis=1, keepdims=True), 1e-300)
        yV2 = ymol2[:, iv].sum(axis=1)
        dn = np.where(yV2 > 1e-12, dV / np.where(yV2 > 1e-12, yV2, 1.0), 0.0)
        g = dn[:, None] * ymol2 * M
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(g > 0.0, 0.999 * cur / np.where(g > 0.0, g, 1.0), np.inf)
        g = g * np.minimum(1.0, ratio.min(axis=1))[:, None]
        cur = cur - g
        dist = dist + g
    res = c0 - dist
    shape = lead + (c0.shape[-1],)
    return dist.reshape(shape), res.reshape(shape)


class BoilOut(tuple):
    """(steam_comp, comp_new, T_new, dE, boiled_J) plus .crystal_new (the crystal
    grams after the nucleation rule), .crystal_J (joules the crystallising gave
    back, + = released; part of dE, not of boiled_J) and .L_eff (J/g of the steam)."""


def _make_out(five, crystal_new, crystal_J, L_eff):
    o = BoilOut(five)
    o.crystal_new = crystal_new
    o.crystal_J = crystal_J
    o.L_eff = L_eff
    return o


_NEWTON_MAX = 3
_NEWTON_TOL_K = 1e-4


def _tb_y(c, P, tb_water, volmask):
    """(Tb, y mass fractions, L_eff) for cells c (N,S). Non-volatile cells boil
    water out of a solution; volatile cells use the bubble point."""
    N = c.shape[0]
    tb = np.empty(N)
    y = np.zeros(c.shape)
    L = np.full(N, D.LATENT_SIM)
    nv = ~volmask
    if nv.any():
        tb[nv] = boiling_point(c[nv], P, tb_water)
        y[nv, _I["h2o"]] = 1.0
    if volmask.any():
        t, yy, ll = bubble_point(c[volmask], P)
        if tb_water is not None:
            t = t + (tb_water - float(tsat_water(P)))
        tb[volmask] = t
        y[volmask] = yy
        L[volmask] = ll
    return tb, y, L


def boil_step_general(comp, T, C_JK, P=D.P_ATM, crystal=None, tb_water=None):
    """The general boil step (volatile cells use the bubble point). Boil the superheat off cells. comp: DISSOLVED grams (N,S); T the cell
    temperature (C); C_JK its heat capacity (J/K); crystal the crystal grams (zeros
    if None); tb_water the host's pure-water boiling point (the sim's BOIL; None
    uses IF97). Returns BoilOut:
      (steam_comp, comp_new, T_new, dE, boiled_J), .crystal_new, .crystal_J, .L_eff
    steam_comp + comp_new + (crystal_new - crystal) = comp (mass exact); dE =
    C (T_new - T) = -boiled_J + crystal_J, so energy is exact. The solute stays, so
    the boiling point climbs as syrup thickens. Crystal rule: after the step, where
    dissolved / saturation exceeds the species' nucleation ratio, the excess over
    saturation moves to crystal in this call (a boiled-dry brine leaves its salt as
    crystal; a syrup stays dissolved up to ratio 1.3). The same rule runs first so a
    cell handed over supersaturated still has a boiling point."""
    c_in = prep(comp)
    lead = c_in.shape[:-1]
    c = c_in.reshape(-1, c_in.shape[-1]).copy()
    N = c.shape[0]
    T = np.broadcast_to(np.asarray(T, dtype=float), lead).reshape(-1).copy()
    C = np.broadcast_to(np.asarray(C_JK, dtype=float), lead).reshape(-1)
    cry = np.zeros_like(c) if crystal is None else np.broadcast_to(prep(crystal), c_in.shape).reshape(c.shape).copy()
    T_in = T.copy()
    # nucleation first
    c, cry, rel0 = S.nucleate(c, cry, T)
    T = T + rel0 / C
    vm = (c[:, _I["ethanol"]] + c[:, _I["methanol"]]) > 0.0
    tb0, y, L = _tb_y(c, P, tb_water, vm)
    ex = T - tb0
    act = np.isfinite(ex) & (ex > 0.0) & (c[:, _I["h2o"]] > 0.0)
    steam = np.zeros_like(c)
    if act.any():
        ia = np.nonzero(act)[0]
        ca, ya, La, Ca, Ta, ea, tba, va = c[ia], y[ia], L[ia], C[ia], T[ia], ex[ia], tb0[ia], vm[ia]
        with np.errstate(invalid="ignore", divide="ignore"):
            cap = 0.999 * np.where(ya > 0.0, ca / np.where(ya > 0.0, ya, 1.0), np.inf).min(axis=1)
        # vapour-able mass for the finite difference
        vol_mass = np.maximum(ca[:, _I["h2o"]] + ca[:, _I["ethanol"]] + ca[:, _I["methanol"]], 1e-12)
        h = 1e-3 * vol_mass
        tb_h = _tb_y(np.maximum(ca - h[:, None] * ya, 0.0), P, tb_water, va)[0]
        s = (tb_h - tba) / h
        dm = np.minimum(ea * Ca / (La + Ca * s), cap)
        newton = (ea > 2.0) | va
        for _ in range(_NEWTON_MAX):
            if not newton.any():
                break
            c1 = np.maximum(ca - dm[:, None] * ya, 0.0)
            tb1 = _tb_y(c1, P, tb_water, va)[0]
            g = (Ta - tb1) * Ca - La * dm
            hh = np.maximum(1e-3 * (vol_mass - dm), 1e-12)
            tb2 = _tb_y(np.maximum(c1 - hh[:, None] * ya, 0.0), P, tb_water, va)[0]
            s2 = (tb2 - tb1) / hh
            dm_new = np.clip(dm + g / (La + Ca * s2), 0.0, cap)
            dm = np.where(newton, dm_new, dm)
            newton = newton & (np.abs(g) / Ca > _NEWTON_TOL_K)
        steam[ia] = dm[:, None] * ya
    boiled = (steam * np.array([D.SPECIES[x].Lv for x in D.ORDER])).sum(axis=1)
    c_new = c - steam
    T_new = T - boiled / C
    c_new, cry_new, rel1 = S.nucleate(c_new, cry, T_new)
    T_new = T_new + rel1 / C
    dE = C * (T_new - T_in)
    shape = lead + (c.shape[-1],)
    five = (steam.reshape(shape), c_new.reshape(shape), T_new.reshape(lead),
            dE.reshape(lead), boiled.reshape(lead))
    return _make_out(five, cry_new.reshape(shape), (rel0 + rel1).reshape(lead), L.reshape(lead))


_LAT = D.LATENT_SIM
_DELTA_K = 0.05                    # step for the temperature derivatives (K)


_CAPINFO = {}


def _cap_info(keys):
    """[(group index, species name, nucleation ratio, M)] for the groups that are a
    single solute with a solubility row; cached per registry revision."""
    k = (D.revision(), tuple(keys))
    t = _CAPINFO.get(k)
    if t is None:
        t = []
        for j, key in enumerate(keys):
            mem = [n for n in D.ORDER if D.SPECIES[n].aq and D.SPECIES[n].model == key]
            if len(mem) == 1 and S.has_solubility(mem[0]):
                sp = D.SPECIES[mem[0]]
                t.append((j, mem[0], sp.nucleation_ratio, sp.M))
        _CAPINFO[k] = t
    return t


def _sat_caps(grp, T):
    """(N,g) molality at which a group's solute would have crystallised (nucleation
    ratio times saturation at T): inf for a group with no single solute that has a
    solubility row. A trial state in the step solve is held there, because the
    crystal rule leaves the solution at saturation, so the boiling point stops
    climbing and the solve cannot run away on a boiled-dry cell."""
    cap = np.full(grp[1].shape, np.inf)
    if grp[1].shape[1] == 0:
        return cap
    for j, name, nuc, M in _cap_info(grp[0]["keys"]):
        if grp[2][:, j].any():
            cap[:, j] = nuc * S.solubility_water(name, T) * 10.0 / M
    return cap


def _boil_fast(c, cry, T, C, P, tb_water, lead):
    """The step for cells with no ethanol or methanol: the steam is water, so the
    arrays shrink to the few species the cells hold, the molalities are grouped once
    (they go as 1/water, so taking water out is a scale), and L is the sim's LATENT."""
    N, S_ = c.shape
    colmask = c.any(axis=0)
    if cry is not None:
        colmask |= cry.any(axis=0)
    colmask[_I["h2o"]] = True
    cols = np.nonzero(colmask)[0]
    iw = int(np.searchsorted(cols, _I["h2o"]))
    cc = c[:, cols]
    cr = np.zeros((N, len(cols))) if cry is None else cry[:, cols]
    T_in = T
    dc, cr, rel0 = S.nucleate_cols(cc, cr, cols, T)
    T = T + rel0 / C
    Pf = np.broadcast_to(np.asarray(P, dtype=float), (N,))
    t0 = tsat_water(Pf)
    grp = W._groups(dc, cols)
    Wt = grp[4]

    capG = []                      # the crystal-rule caps, only built if a Newton pass needs them

    def tbf(sel, scale):
        cap = None
        if scale is not None:
            if not capG:
                capG.append(_sat_caps(grp, T))
            cap = capG[0][sel]
        t, la = W.tb_from_groups(grp.take(sel), Pf[sel], t0[sel], scale, cap)
        if tb_water is None:
            return t
        return tb_water + np.where(la != 0.0, t - t0[sel], 0.0)

    allc = slice(None)
    tb = tbf(allc, None)
    ex = T - tb
    act = np.isfinite(ex) & (ex > 0.0) & (Wt > 0.0)
    dm_full = np.zeros(N)
    if act.any():
        ia = np.nonzero(act)[0]
        ga = grp.take(ia)
        Wa, Ca, Ta, ea, tba = Wt[ia], C[ia], T[ia], ex[ia], tb[ia]
        # slope dTb / d(water removed), by implicit differentiation of
        # ln aw(W, T) + ln psat(T) = ln P at the answer: two ln aw evaluations and
        # one vapour-pressure call instead of a second fixed point
        h = 1e-4 * Wa
        Tr = tba if tb_water is None else tba - (tb_water - t0[ia])      # the IF97 T of the answer
        la0 = W._lnaw_g(ga, Tr)
        laW = W._lnaw_g(ga.scaled(Wa / (Wa - h)), Tr)
        dl = _DELTA_K
        laT = W._lnaw_g(ga, Tr + dl)
        lp = np.log(_psat_k(np.concatenate([Tr, Tr + dl]) + D.T0K))
        qT = (laT - la0) / dl + (lp[len(Tr):] - lp[:len(Tr)]) / dl
        s = -((laW - la0) / h) / qT
        cap = 0.999 * Wa
        dm = np.minimum(ea * Ca / (_LAT + Ca * s), cap)
        newton = ea > 2.0
        for _ in range(_NEWTON_MAX):
            if not newton.any():
                break
            tb1 = tbf(ia, Wa / np.maximum(Wa - dm, 1e-300))
            g = (Ta - tb1) * Ca - _LAT * dm
            hh = np.maximum(1e-3 * (Wa - dm), 1e-12)
            tb2 = tbf(ia, Wa / np.maximum(Wa - dm - hh, 1e-300))
            s2 = (tb2 - tb1) / hh
            dm = np.where(newton, np.clip(dm + g / (_LAT + Ca * s2), 0.0, cap), dm)
            newton = newton & (np.abs(g) / Ca > _NEWTON_TOL_K)
        dm_full[ia] = np.where(np.isfinite(dm), dm, 0.0)
    boiled = dm_full * _LAT
    T_new = T - boiled / C
    dc = dc.copy()
    dc[:, iw] -= dm_full
    dc2, cr2, rel1 = S.nucleate_cols(dc, cr, cols, T_new)
    T_new = T_new + rel1 / C
    dE = C * (T_new - T_in)
    comp_new = np.zeros((N, S_))
    comp_new[:, cols] = dc2
    cry_new = np.zeros((N, S_))
    cry_new[:, cols] = cr2
    steam = np.zeros((N, S_))
    steam[:, _I["h2o"]] = dm_full
    shape = lead + (S_,)
    five = (steam.reshape(shape), comp_new.reshape(shape), T_new.reshape(lead),
            dE.reshape(lead), boiled.reshape(lead))
    return _make_out(five, cry_new.reshape(shape), (rel0 + rel1).reshape(lead), np.full(lead, _LAT))


def boil_step(comp, T, C_JK, P=D.P_ATM, crystal=None, tb_water=None):
    """Boil the superheat off cells. comp: DISSOLVED grams (..., S); T the cell
    temperature (C); C_JK its heat capacity (J/K); crystal the crystal grams (zeros
    if None); tb_water the host's pure-water boiling point (the sim's BOIL; None
    uses IF97). Returns BoilOut:
      (steam_comp, comp_new, T_new, dE, boiled_J), .crystal_new, .crystal_J, .L_eff
    steam_comp + comp_new + (crystal_new - crystal) = comp (mass exact); dE =
    C (T_new - T) = -boiled_J + crystal_J, so energy is exact. The solute stays, so
    the boiling point climbs as syrup thickens. Crystal rule: after the step, where
    dissolved / saturation exceeds the species' nucleation ratio, the excess over
    saturation moves to crystal in this call (a boiled-dry brine leaves its salt as
    crystal; a syrup stays dissolved up to ratio 1.3). The same rule runs first so a
    cell handed over supersaturated still has a boiling point. Cells holding ethanol
    or methanol take the bubble-point route (boil_step_general)."""
    c_in = prep(comp)
    if c_in[..., _I["ethanol"]].any() or c_in[..., _I["methanol"]].any():
        return boil_step_general(c_in, T, C_JK, P, crystal, tb_water)
    lead = c_in.shape[:-1]
    c = c_in.reshape(-1, c_in.shape[-1])
    N = c.shape[0]
    Tf = np.broadcast_to(np.asarray(T, dtype=float), lead).reshape(-1)
    Cf = np.broadcast_to(np.asarray(C_JK, dtype=float), lead).reshape(-1)
    cry = None if crystal is None else np.broadcast_to(prep(crystal), c_in.shape).reshape(c.shape)
    return _boil_fast(c, cry, Tf, Cf, P, tb_water, lead)
