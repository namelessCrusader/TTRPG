"""GAS-SOLID EQUILIBRIA, CARBON POTENTIAL AND ELECTRODES (wave 13, T4, law L14).

Small tables with one formula each, pure numpy, no World. T in C, K in atm (the
dimensionless equilibrium constant on an atm basis), partial pressures in Pa unless
a function says atm. Nothing here is a rate: `affinity` only says which way a
reaction would go (positive = forward) and how hard; L3 moves the gas and L4 sets
the speed.

  reaction_K, eq_pressure, affinity, co_fraction     gas-solid rows (calcite, ...)
  carbon_activity, austenite_carbon, carbon_activity_of
  E0, nernst, cell_emf                                electrode potentials

E0 is the DRIVING FORCE only. Aluminium has a passive oxide film; the film (not E0)
gates the rate, and it is L4/L7 data, not a fact of this law.
"""
import numpy as np

from . import mixdata as D

R = D.R_GAS
ATM = D.P_ATM


def _row(rid):
    return D.row("gassolid", rid)


def reaction_K(rid, T):
    """Equilibrium constant (atm basis) of a gas-solid row at T (C): ln K = a - b/T
    for kind 'lnk'; K = exp(-dG / (R T)) with dG = a - b T (J/mol) for kind 'dg'."""
    kind, a, b, _form = _row(rid)
    Tk = np.asarray(T, dtype=float) + D.T0K
    if kind == "lnk":
        return np.exp(a - b / Tk)
    return np.exp(-(a - b * Tk) / (R * Tk))


def eq_pressure(rid, T):
    """Equilibrium pressure of the one gas a decomposition gives off, Pa (calcite:
    1 atm at 898 C)."""
    if _row(rid)[3] != "p_single":
        raise ValueError(f"{rid} is not a single-gas decomposition row")
    return reaction_K(rid, T) * ATM


def affinity(rid, T, p):
    """A = R T ln(K / Q) in J/mol, positive = forward. p: the gas pressure in Pa for a
    single-gas row; (pCO, pCO2) in Pa for the CO/CO2 rows (carbon activity 1)."""
    form = _row(rid)[3]
    K = reaction_K(rid, T)
    if form == "p_single":
        Q = np.asarray(p, dtype=float) / ATM
    else:
        pco, pco2 = (np.asarray(x, dtype=float) / ATM for x in p)
        Q = pco * pco / pco2 if form == "co_co2" else pco2 / pco
    return R * (np.asarray(T, dtype=float) + D.T0K) * np.log(K / Q)


def co_fraction(rid, T, P=ATM, a_C=1.0):
    """CO share (by moles) of a CO/CO2 gas at equilibrium with the row, total
    pressure P in Pa. 'boudouard': C + CO2 = 2 CO at carbon activity a_C;
    'wuestite': FeO + CO = Fe + CO2."""
    form = _row(rid)[3]
    K = reaction_K(rid, T)
    if form == "co2_over_co":
        return 1.0 / (1.0 + K)
    if form != "co_co2":
        raise ValueError(f"{rid} is not a CO/CO2 row")
    Ka = K * a_C
    Pa = np.asarray(P, dtype=float) / ATM
    return (-Ka + np.sqrt(Ka * Ka + 4.0 * Pa * Ka)) / (2.0 * Pa)


def carbon_activity(pCO, pCO2, T):
    """Carbon activity of a CO/CO2 gas: a_C = pCO^2 / (K_B(T) pCO2), partial
    pressures in atm. If a_C(gas) < a_C(iron) carbon leaves the iron."""
    return np.asarray(pCO, dtype=float) ** 2 / (reaction_K("boudouard", T) * np.asarray(pCO2, dtype=float))


_MC, _MFE = 12.011, 55.845


def carbon_activity_of(wt_c, T):
    """Activity of carbon in austenite for wt% C: log10 a = log10(y/(1-y)) + a0/T +
    a1 + (a2/T + a3) N, N the atom fraction of C and y = N/(1-N) the filled site share."""
    a0, a1, a2, a3 = D.row("austenite", "a")
    w = np.asarray(wt_c, dtype=float)
    Tk = np.asarray(T, dtype=float) + D.T0K
    n_c = w / _MC
    N = n_c / (n_c + (100.0 - w) / _MFE)
    y = N / (1.0 - N)
    return 10.0 ** (np.log10(y / (1.0 - y)) + a0 / Tk + a1 + (a2 / Tk + a3) * N)


def austenite_carbon(aC, T):
    """Inverse of carbon_activity_of: wt% C in austenite at carbon activity aC
    (bisection on the atom fraction, 80 steps; round-trips to 1e-9)."""
    a0, a1, a2, a3 = D.row("austenite", "a")
    aC = np.asarray(aC, dtype=float)
    Tk = np.asarray(T, dtype=float) + D.T0K
    lo = np.full(np.broadcast(aC, Tk).shape, 1e-12)
    hi = np.full(lo.shape, 0.33)
    target = np.log10(aC)
    for _ in range(80):
        N = 0.5 * (lo + hi)
        y = N / (1.0 - N)
        g = np.log10(y / (1.0 - y)) + a0 / Tk + a1 + (a2 / Tk + a3) * N - target
        hi = np.where(g > 0.0, N, hi)
        lo = np.where(g > 0.0, lo, N)
    N = 0.5 * (lo + hi)
    return 100.0 * N * _MC / (N * _MC + (1.0 - N) * _MFE)


# -- electrodes -----------------------------------------------------------------------------
def E0(couple):
    """Standard potential vs SHE at 25 C, V (for a couple with a pH term: its value at its own pH)."""
    return D.row("e0", couple)[0]


def nernst(couple, a_ox=1.0, a_red=1.0, T=25.0, pH=None):
    """E = E0 - (R T / n F) ln(a_red / a_ox), plus the pH term for a couple that
    consumes h protons per reduction: - (R T ln10 / F)(h / n)(pH - pH_of_E0).
    59.16 mV per decade at n = 1, 25 C."""
    e0, n, h, ph0 = D.row("e0", couple)
    Tk = np.asarray(T, dtype=float) + D.T0K
    e = e0 - (R * Tk / (n * D.F_FAR)) * np.log(np.asarray(a_red, dtype=float) / np.asarray(a_ox, dtype=float))
    if pH is not None and h:
        e = e - (R * Tk * np.log(10.0) / D.F_FAR) * (h / n) * (np.asarray(pH, dtype=float) - ph0)
    return e


def _acts(a):
    return a if isinstance(a, tuple) else (a, 1.0)


def cell_emf(anode, cathode, a_a=1.0, a_c=1.0, T=25.0, pH=None):
    """E_cathode - E_anode, V; the couple with the lower E dissolves. a_a, a_c are
    the activity of the oxidised form (the metal's ion) or a tuple (a_ox, a_red)."""
    ea = nernst(anode, *_acts(a_a), T=T, pH=pH)
    ec = nernst(cathode, *_acts(a_c), T=T, pH=pH)
    return ec - ea
