"""PSYCHROMETRY AND THE SURFACE LAW (wave 13, T6): pure functions on arrays.

Water vapour in air, the latent heat and its books, the one surface law (evaporate,
condense, sorb) with the Stefan (blowing) factor, the wet bulb and dew point,
the saturation adjustment that makes mist, the exact boil step, the GAB
isotherm of wood, and the closed forms erf and Crank that the slow clock jumps with.

No World, no state: every function takes numbers and returns numbers. Units: grams,
joules, seconds, metres, degC, Pa; E of a cell is measured from 20 C.

THE BOOKS (T6 spec section 3). One gram of water going between liquid at Tv and
vapour:  cell E changes by -(L0 + cv (Tv - 20));  the gas voxel's E by +cv (Tv - 20);
the stored latent by +L0. The three add to zero. This equals the true latent heat
h_fg(Tv) = L0 - (cw - cv)(Tv - 20) plus the sensible heat the departing gram carries.

Vapour pressure comes from T4's mixwater.psat_water when that file exists and agrees
with the stand-in to 0.2 % at 20 C; otherwise from _standin.py (Wagner-Pruss). The
solute functions (a_w, boil, freeze, VLE) always come from _standin until phase B.
"""
import math

import numpy as np

from . import _standin
from .matterdata import (AMBIENT, CI, CV, CW, L0, L_CONV20, M_WATER, P_ATM, R)

try:
    from . import mixwater as _mw
except ImportError:
    _mw = None

AIR_RHO_CP = 1200.0     # J/(m3 K): rho_air cp; h_m = h/AIR_RHO_CP is Lewis exponent 0
STEFAN_CAP = 0.9        # largest mole fraction the blowing factor is evaluated at
RHO_WATER_G_M3 = 1.0e6


def _pick_provider():
    """'mixwater' if T4's file is importable and agrees with the stand-in, else 'standin'."""
    if _mw is None or not hasattr(_mw, "psat_water"):
        return "standin"
    try:
        v = float(_mw.psat_water(20.0))
    except Exception:
        return "standin"
    ok = bool(np.isfinite(v)) and abs(v / float(_standin.psat_water(20.0)) - 1.0) < 2e-3
    return "mixwater" if ok else "standin"


PROVIDER = _pick_provider()


def p_sat_water(Tc):
    """Saturation pressure over liquid water, Pa."""
    Tc = np.asarray(Tc, np.float64)
    if PROVIDER == "mixwater":
        # T4's IF97 from the triple point up; below it (and wherever it gives NaN) the
        # stand-in. The two agree to 1e-7 at the seam.
        p = np.asarray(_mw.psat_water(Tc), np.float64)
        return np.where(np.isfinite(p) & (Tc >= 0.01), p, _standin.psat_water(Tc))
    return _standin.psat_water(Tc)


def p_sat_ice(Tc):
    """Saturation pressure over ice, Pa."""
    Tc = np.asarray(Tc, np.float64)
    if PROVIDER == "mixwater" and hasattr(_mw, "psat_ice"):
        p = np.asarray(_mw.psat_ice(Tc), np.float64)
        return np.where(np.isfinite(p), p, _standin.psat_ice(Tc))
    return _standin.psat_ice(Tc)


# forwarders: the solute functions are the stand-in's until T4's comp-based API is wired
aw = _standin.aw
boil_point = _standin.boil_point
freeze_point = _standin.freeze_point
vle_ethanol = _standin.vle_ethanol


def rho_sat(Tc, over_ice=False):
    """Saturation vapour density, g/m3: p M/(R T). 17.3 at 20 C, 589 at 100 C."""
    Tc = np.asarray(Tc, np.float64)
    p = p_sat_ice(Tc) if over_ice else p_sat_water(Tc)
    return p * M_WATER / (R * (Tc + 273.15))


def h_fg(Tc):
    """Latent heat of vaporisation, J/g, Kirchhoff form (c): within 0.8 % of the
    steam table to 100 C and 2 % to 150 C. The books use this form on purpose: it is
    what makes them close."""
    return L0 - (CW - CV) * (np.asarray(Tc, np.float64) - AMBIENT)


def evap_energy(Tv):
    """Per gram evaporated at Tv: (cell E loss, gas E gain, stored latent gain)."""
    Tv = np.asarray(Tv, np.float64)
    sens = CV * (Tv - AMBIENT)
    return L0 + sens, sens, L0 + 0.0 * Tv


def phase_transfer(E_cell, E_gas, stored, g, Tv):
    """Move g grams from liquid (g > 0 evaporates, g < 0 condenses) at Tv.
    Returns the new (E_cell, E_gas, stored). E_cell + E_gas + stored is unchanged."""
    cell, gas, st = evap_energy(Tv)
    return E_cell - g * cell, E_gas + g * gas, stored + g * st


def stefan_factor(x, cap=STEFAN_CAP):
    """Blowing factor ln(1/(1-x))/x with x capped (c: 1.03, 1.19, 1.72, 2.56 at
    x = 0.05, 0.3, 0.7, 0.9)."""
    x = np.minimum(np.asarray(x, np.float64), cap)
    xs = np.maximum(x, 1e-12)
    return np.where(x < 1e-9, 1.0, -np.log1p(-xs) / xs)


def mass_coeff(h_conv):
    """Mass-transfer coefficient h_m, m/s, from the convective h (Lewis exponent 0)."""
    return np.asarray(h_conv, np.float64) / AIR_RHO_CP


def surface_flux(h_conv, Ts, a_s, rho_air, R_int=0.0, P=P_ATM, T_air=None,
                 over_ice=False):
    """Mass flux at a surface, g/(m2 s); positive = evaporation.

    J = F_s (rho_eq - rho_air) / (1/h_m + R_int),  rho_eq = rho_sat(Ts) a_s
    F_s = Stefan factor of the surface vapour fraction x_s = p_sat(Ts) a_s / P for
    evaporation, of the gas side x_g for condensation. Zero at equilibrium, flips
    sign at the dew point, scales with h (while R_int is small)."""
    Ts = np.asarray(Ts, np.float64)
    a_s = np.asarray(a_s, np.float64)
    rho_air = np.asarray(rho_air, np.float64)
    ps = p_sat_ice(Ts) if over_ice else p_sat_water(Ts)
    rho_eq = rho_sat(Ts, over_ice) * a_s
    x_s = ps * a_s / P
    Ta = Ts if T_air is None else np.asarray(T_air, np.float64)
    x_g = rho_air * R * (Ta + 273.15) / M_WATER / P
    F = np.where(rho_eq >= rho_air, stefan_factor(x_s), stefan_factor(x_g))
    res = 1.0 / mass_coeff(h_conv) + np.asarray(R_int, np.float64)
    return F * (rho_eq - rho_air) / res


def surface_step(rho_eq, rho_air, V_m3, A_m2, h_conv, F=1.0, R_int=0.0, dt=1.0):
    """Grams moved in one step by the EXPONENTIAL form (positive = evaporation):
    m = (rho_eq - rho_air) V (1 - exp(-dt/tau)),  tau = V / (A F / (1/h_m + R_int)).
    It never passes equilibrium for any tau and dt, so no spurious mist is made
    (c: tau is 6 s for a 5 cm voxel, 1.5 s at 2.5 cm)."""
    res = 1.0 / mass_coeff(h_conv) + np.asarray(R_int, np.float64)
    tau = V_m3 * res / (A_m2 * F)
    return (rho_eq - rho_air) * V_m3 * (-np.expm1(-dt / tau))


def dew_point(rho_g_m3, over_ice=False):
    """Temperature (C) at which air holding rho g/m3 of vapour is saturated."""
    rho = np.asarray(rho_g_m3, np.float64)
    return _standin.bisect(lambda T: rho_sat(T, over_ice) - rho, -80.0, 200.0)


def wet_bulb(Ta, rh, h_conv=10.0, P=P_ATM, stefan=True):
    """Temperature of a wet surface in still air, C. The heat balance
    F h_m (rho_sat(Ts) - rh rho_sat(Ta)) h_fg(Ts) = h (Ta - Ts) does not depend on h
    (Lewis exponent 0): 17.7 at 25 C / 50 % (chart 17.9), 13.6 at 20 C / 50 %."""
    Ta = np.asarray(Ta, np.float64)
    rho_a = np.asarray(rh, np.float64) * rho_sat(Ta)

    def f(Ts):
        F = stefan_factor(p_sat_water(Ts) / P) if stefan else 1.0
        flux = F * (rho_sat(Ts) - rho_a) / (AIR_RHO_CP / h_conv) * h_fg(Ts)
        return flux - h_conv * (Ta - Ts)
    return _standin.bisect(f, -40.0, np.maximum(Ta, -39.0))


# ---- the boil step ----------------------------------------------------------
def boil_g(C, T, m_w, T_b):
    """Grams boiled in one tick: min(water, C (T - T_b) / h_fg(T_b)). The steam
    leaves at T_b, which makes this EXACT: the cell ends at T_b (c: residual 0.0)."""
    g = C * (T - T_b) / h_fg(T_b)
    return np.minimum(np.maximum(g, 0.0), m_w)


def boil_step(E, C, m_w, T_b):
    """One boil step of a cell with heat capacity C (J/K, including its water),
    energy E (J from 20 C) and m_w grams of water. Returns
    (g, E_new, C_new, E_gas_gain, stored_gain): the steam is made at T_b."""
    E = np.asarray(E, np.float64)
    C = np.asarray(C, np.float64)
    T = AMBIENT + E / C
    g = boil_g(C, T, m_w, T_b)
    cell, gas, st = evap_energy(T_b)
    return g, E - g * cell, C - g * CW, g * gas, g * st


# ---- mist by saturation adjustment ------------------------------------------
def _drho_dT(Tc):
    return (rho_sat(Tc + 0.01) - rho_sat(Tc - 0.01)) / 0.02


_T_SOLVE_MAX = 370.0    # the vapour table ends near 374 C; the solve never looks past this


def saturation_adjust(m_vap, T0, V_m3, C_other, iters=12):
    """Condense the surplus of a gas voxel at once. Vapour m0 g at T0 C in V m3, with
    heat capacity C_other (air, J/K, without the new mist) plus cv per gram of vapour
    left plus cw per gram of the new mist. Solve
    F(mc) = (m0 - mc) - rho_sat(T1) V = 0,  T1 = T0 + L0 mc / (C_other + cv (m0 - mc) + cw mc)
    (c: a 5 cm voxel at 150 % ends 25.2 C with 0.33 mg condensed; the naive bite would
    have set about 38 C). F falls steadily from F(0) > 0 to F(m0) < 0 for a voxel with a
    surplus, so the root is BRACKETED in [0, m0] and every step is a Newton step that
    falls back to bisection when it leaves the bracket. The answer is therefore always
    finite and inside [0, m0], and T1 is looked up no higher than 370 C, so a huge
    surplus (a quench or flash boil putting grams of steam into one voxel) cannot make
    NaN. 12 steps converge to better than 1e-6 of the surplus for any surplus
    (tests sweep 1.1x to 100x). Returns (mc, T1). The books: gas E += L0 mc,
    vapour -= mc, mist += mc."""
    m0 = np.asarray(m_vap, np.float64)
    T0 = np.asarray(T0, np.float64)
    Co = np.asarray(C_other, np.float64)
    over = m0 > rho_sat(T0) * V_m3

    def T_of(mc):
        return T0 + L0 * mc / (Co + CV * (m0 - mc) + CW * mc)

    def F_of(mc):
        return (m0 - mc) - rho_sat(np.clip(T_of(mc), -80.0, _T_SOLVE_MAX)) * V_m3

    lo = np.zeros_like(m0 + T0 + Co)
    hi = np.broadcast_to(m0, lo.shape).copy()
    mc = lo.copy()
    for _ in range(iters):
        T1 = np.clip(T_of(mc), -80.0, _T_SOLVE_MAX)
        Cg = Co + CV * (m0 - mc) + CW * mc
        F = (m0 - mc) - rho_sat(T1) * V_m3
        lo = np.where(F > 0.0, mc, lo)
        hi = np.where(F <= 0.0, mc, hi)
        dT = L0 / Cg - L0 * mc * (CW - CV) / Cg ** 2
        dF = -1.0 - V_m3 * _drho_dT(T1) * dT
        with np.errstate(divide="ignore", invalid="ignore"):
            step = mc - F / dF
        ok = np.isfinite(step) & (step >= lo) & (step <= hi)
        mc = np.where(ok, step, 0.5 * (lo + hi))
    mc = np.where(over, mc, 0.0)
    return mc, T_of(mc)


def mist_evaporate(m_mist, m_vap, T, V_m3, dt, tau=10.0):
    """Grams of mist that go back to vapour in dt: the deficit to saturation,
    relaxed with tau (10 s, u), never more than the mist there is. The gas E pays
    L0 per gram."""
    deficit = np.maximum(rho_sat(T) * V_m3 - m_vap, 0.0)
    return np.minimum(m_mist, deficit * (-np.expm1(-dt / tau)))


def mist_extinction(lwc_g_m3, r_eff_m=5e-6):
    """Extinction per metre, K = 3 LWC / (2 rho_w r_eff): 0.1 g/m3 gives 0.03/m,
    visibility about 130 m (c)."""
    return 3.0 * np.asarray(lwc_g_m3, np.float64) / (2.0 * RHO_WATER_G_M3 * r_eff_m)


# ---- the vapour mixing bound for steam added to a small voxel ----------------
def mix_steam(T_gas, C_other, m_vap, g, T_steam):
    """Temperature of a gas voxel after g grams of steam at T_steam join it. The
    vapour's heat capacity is in C, so the answer is the weighted mean, always
    between T_gas and T_steam."""
    C0 = C_other + CV * m_vap
    return (C0 * T_gas + CV * g * T_steam) / (C0 + CV * g)


# ---- sorption isotherm (GAB) -------------------------------------------------
def gab_w(a, wm=0.065, C=9.0, K=0.78):
    """Equilibrium moisture (g water per g dry) at water activity a. Wood default:
    fitted (c) to the Wood Handbook anchors (u): 6.3, 9.2, 12.0, 16.0, 21.0, 24.0 %
    at 30, 50, 65, 80, 90, 95 % RH."""
    y = K * np.asarray(a, np.float64)
    return wm * C * y / ((1.0 - y) * (1.0 - y + C * y))


def gab_dwda(a, wm=0.065, C=9.0, K=0.78):
    y = K * np.asarray(a, np.float64)
    g = (1.0 - y) * (1.0 + (C - 1.0) * y)
    gp = (C - 2.0) - 2.0 * (C - 1.0) * y
    return K * wm * C * (g - y * gp) / g ** 2


def gab_aw(w, wm=0.065, C=9.0, K=0.78):
    """The inverse: water activity at moisture w (the isotherm, read backwards)."""
    w = np.asarray(w, np.float64)
    B = w * (C - 2.0) - wm * C
    y = (B + np.sqrt(B * B + 4.0 * (C - 1.0) * w * w)) / (2.0 * (C - 1.0) * np.maximum(w, 1e-300))
    return np.where(w <= 0.0, 0.0, y / K)


# ---- closed forms (T5 jumps with these) --------------------------------------
_erf = np.vectorize(math.erf, otypes=[np.float64])


def erf(x):
    return _erf(np.asarray(x, np.float64))


def fick_erf(x, t, D, c0, cs):
    """Semi-infinite solid, surface held at cs, start c0: c(x, t)."""
    x = np.asarray(x, np.float64)
    return cs + (c0 - cs) * erf(x / (2.0 * np.sqrt(D * t)))


def crank_slab(x, t, D, l, c0, c1, terms=60):
    """Plane sheet of thickness 2 l (x from the centre), both faces held at c1,
    initial c0 (Crank, The Mathematics of Diffusion, 4.18)."""
    x = np.asarray(x, np.float64)
    s = np.zeros_like(x)
    for n in range(terms):
        k = (2 * n + 1) * math.pi / (2.0 * l)
        s = s + (-1.0) ** n / (2 * n + 1) * math.exp(-D * k * k * t) * np.cos(k * x)
    return c1 + (c0 - c1) * (4.0 / math.pi) * s


def crank_slab_mean(t, D, l, c0, c1, terms=60):
    """Mean content of the same slab: the drying curve."""
    s = 0.0
    for n in range(terms):
        k = (2 * n + 1) * math.pi / (2.0 * l)
        s += math.exp(-D * k * k * t) / (2 * n + 1) ** 2
    return c1 + (c0 - c1) * 8.0 / math.pi ** 2 * s


def ice_enthalpy(E, m_ice):
    """True enthalpy of a cell: E - L_conv20 m_ice (c). Constant through freezing and
    melting at no outside heat."""
    return np.asarray(E, np.float64) - L_CONV20 * np.asarray(m_ice, np.float64)


def sensible_ice(m_ice, T):
    """E of m_ice grams of ice at T (J from 20 C): cp_ice (T - 20)."""
    return CI * m_ice * (np.asarray(T, np.float64) - AMBIENT)
