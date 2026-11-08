"""PRIVATE STAND-IN for T4's mixture functions (wave 13, T6, phase A).

T4 (mixwater.py, mixvle.py, mixsol.py) owns water vapour pressure, water activity,
the boiling rise, the freezing drop, the solubility lines and Van Laar. T6 is built
in parallel, so this file keeps a small copy of each, good to the same few tenths
of a percent, and a skip-marked cross test (tests/test_t6_psychro.py) compares the
two once mixwater.py exists. At wiring (phase B) this file is deleted and the
callers import T4's.

Every function takes a SOLUTE LOAD `b` = grams of solute per gram of water, so no
species table is needed. Temperatures in degC, pressure in Pa, arrays or scalars.
All numbers are (u) unless marked (c); see matterdata.py.
"""
import numpy as np

from .matterdata import (ANTOINE, HYDROHALITE, LF0, LF_SLOPE, M, M_WATER, NACL_PHI,
                         NORRISH_K, P_ATM, R, SOLUBILITY, SOLUBILITY_T, VAN_LAAR)

# ---- water vapour -----------------------------------------------------------
_TC, _PC = 647.096, 22.064e6
_WP = (-7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719, 1.80122502)


def psat_water(Tc):
    """Saturation pressure over liquid water, Pa. Wagner-Pruss (u); agrees with
    IAPWS-IF97 region 4 to 0.1 % from 0.01 to 150 C (c)."""
    T = np.asarray(Tc, dtype=np.float64) + 273.15
    t = 1.0 - T / _TC
    s = (_WP[0] * t + _WP[1] * t ** 1.5 + _WP[2] * t ** 3 + _WP[3] * t ** 3.5
         + _WP[4] * t ** 4 + _WP[5] * t ** 7.5)
    return _PC * np.exp(_TC / T * s)


def psat_ice(Tc):
    """Saturation pressure over ice, Pa. Murphy & Koop 2005 (u)."""
    T = np.asarray(Tc, dtype=np.float64) + 273.15
    return np.exp(9.550426 - 5723.265 / T + 3.53068 * np.log(T) - 0.00728332 * T)


def lfus(Tc):
    """Latent heat of fusion, J/g, at T C (u: 333.55 + 2.09 T)."""
    return LF0 + LF_SLOPE * np.asarray(Tc, dtype=np.float64)


def bisect(f, lo, hi, n=64):
    """Vectorised bisection for an INCREASING f on [lo, hi] (arrays broadcast)."""
    lo, hi = np.broadcast_arrays(np.asarray(lo, np.float64), np.asarray(hi, np.float64))
    lo, hi = lo.copy(), hi.copy()
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        up = f(mid) > 0.0
        hi = np.where(up, mid, hi)
        lo = np.where(up, lo, mid)
    return 0.5 * (lo + hi)


# ---- water activity ---------------------------------------------------------
_PHI_M = np.array([p[0] for p in NACL_PHI])
_PHI_V = np.array([p[1] for p in NACL_PHI])


def molality_nacl(b):
    return np.asarray(b, np.float64) / M["NaCl"] * 1000.0


def ln_aw_nacl(b):
    """ln a_w of NaCl solution: -phi nu m / 55.508 (Robinson & Stokes table, u)."""
    m = molality_nacl(b)
    phi = np.interp(m, _PHI_M, _PHI_V)
    return -phi * 2.0 * m / (1000.0 / M_WATER)


def ln_aw_sucrose(b):
    """Norrish: ln a_w = ln x_w + K x_s^2, K = -6.47 (u). Good to about 80 wt %."""
    ns = np.asarray(b, np.float64) / M["sucrose"]
    nw = 1.0 / M_WATER
    xs = ns / (ns + nw)
    return np.log(1.0 - xs) + NORRISH_K["sucrose"] * xs ** 2


_LN_AW = {"NaCl": ln_aw_nacl, "sucrose": ln_aw_sucrose}


def aw(species, b):
    return np.exp(_LN_AW[species](b))


def freeze_point(species, b):
    """Freezing point, C: 1/T_f = 1/273.15 - R ln(a_w)/6010 (c). Meaningless for
    sucrose above about 40 wt % (supersaturated); bound it with the solubility."""
    ln = _LN_AW[species](b)
    return 1.0 / (1.0 / 273.15 - R * ln / 6010.0) - 273.15


def boil_point(P=P_ATM, species=None, b=0.0):
    """Boiling point, C, of a solution at pressure P: psat(T) a_w = P (c)."""
    a = 1.0 if species is None else aw(species, b)
    P = np.asarray(P, np.float64)
    return bisect(lambda T: psat_water(T) * a - P, 1.0, 200.0)


# ---- solubility and the NaCl eutectic ---------------------------------------
def solubility(species, Tc):
    """Saturation load, grams per 100 g water. NaCl below 0.1 C is the hydrohalite
    line (u)."""
    Tc = np.asarray(Tc, np.float64)
    s = np.interp(Tc, SOLUBILITY_T, SOLUBILITY[species])
    if species == "NaCl":
        (t0, s0), (t1, s1) = HYDROHALITE
        hh = np.interp(Tc, [t0, t1], [s0, s1])
        s = np.where(Tc < t1, hh, s)
    return s


def solubility_load(species, Tc):
    """The same as a load b = g per g water."""
    return solubility(species, Tc) / 100.0


def _find_eutectic():
    """Where the NaCl liquidus meets the hydrohalite solubility line: (T C, b).
    c: -21.58 C at 23.31 wt %; the real value is -21.1 C at 23.3 wt %."""
    def g(T):
        return freeze_point("NaCl", solubility_load("NaCl", T)) - T
    # g > 0 above the crossing (the liquidus at the solubility is still above T)
    T = bisect(lambda T: -g(T), -26.0, -18.0)
    return float(T), float(solubility_load("NaCl", T))


EUTECTIC_NACL = _find_eutectic()


# ---- ethanol-water VLE ------------------------------------------------------
def _p_ethanol(Tc):
    A, B, C = ANTOINE["ethanol"]
    return 133.322 * 10.0 ** (A - B / (C + np.asarray(Tc, np.float64)))


def van_laar(x1, A12=None, A21=None):
    """Activity coefficients (g1, g2) of ethanol (1) and water (2) at mole fraction x1."""
    if A12 is None:
        A12, A21 = VAN_LAAR[("ethanol", "H2O")]
    x1 = np.asarray(x1, np.float64)
    x2 = 1.0 - x1
    den = A12 * x1 + A21 * x2
    g1 = np.exp(A12 * (A21 * x2 / den) ** 2)
    g2 = np.exp(A21 * (A12 * x1 / den) ** 2)
    return g1, g2


def vle_ethanol(x1, P=P_ATM):
    """Bubble point of ethanol-water at mole fraction x1: (T C, y1)."""
    x1 = np.asarray(x1, np.float64)
    g1, g2 = van_laar(x1)

    def f(T):
        return x1 * g1 * _p_ethanol(T) + (1.0 - x1) * g2 * psat_water(T) - P
    T = bisect(f, 40.0, 110.0)
    y1 = x1 * g1 * _p_ethanol(T) / P
    return T, y1


def abv_mass(y1):
    """Mass fraction of ethanol for mole fraction y1."""
    y1 = np.asarray(y1, np.float64)
    return y1 * M["ethanol"] / (y1 * M["ethanol"] + (1.0 - y1) * M_WATER)


def mole_fraction_ethanol(g_ethanol, g_water):
    ne = np.asarray(g_ethanol, np.float64) / M["ethanol"]
    nw = np.asarray(g_water, np.float64) / M_WATER
    return ne / (ne + nw)


def azeotrope():
    """(x, T) where y = x: scan then refine (c: 0.894, 78.15 C)."""
    x = np.linspace(0.80, 0.99, 20001)
    T, y = vle_ethanol(x)
    d = y - x
    i = int(np.argmin(np.abs(d)))
    return float(x[i]), float(T[i])


def abv_vol(y1):
    """Volume fraction of ethanol for mole fraction y1 (densities 0.789, 0.998 g/ml,
    no mixing contraction). A 10 % ABV wash gives a 52 % ABV vapour by this count
    (46.6 % by mass); the T6 spec's "52 %" is the volume figure."""
    y1 = np.asarray(y1, np.float64)
    ve = y1 * M["ethanol"] / 0.789
    vw = (1.0 - y1) * M_WATER / 0.998
    return ve / (ve + vw)
