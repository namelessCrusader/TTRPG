"""SOLUTIONS (wave 13, T6): the gram arithmetic of dissolving, precipitating and
concentrating. Pure functions on gram arrays; species on the LAST axis. No storage:
T1's lumps hold the grams, T4 holds the equilibria (the saturation load b_sat comes
in as an argument, g solute per g water).

THE VOLUME RULE IS INJECTED. fvol of a solution cell is the SOLUTION volume (what
the fluid hydraulics always meant), so water grams = V - sum v_i m_i. Which rule gives
V from the grams is not decided here: every function takes `rule`, a VolumeRule with
volume(mw, ms, v) and water(V, ms, v). The default is the stand-in LINEAR rule
V = m_w + sum v_i m_i with v_i the apparent specific volume (ml/g): NaCl 0.355,
sucrose 0.625 (c, from the 20 wt % densities). It gives (c) density 1.148 for 20 wt %
NaCl, 1.081 for 20 wt % sucrose, 1.336 for saturated sucrose (real 1.33), and the
saturated sucrose solution is 2.27 times the water volume. T1's Masson rule
(phiV0, Sv) replaces it at wiring by passing a different VolumeRule.

Concentration c is g solute per ml SOLUTION; density rho = (m_w + m_s) / V.
"""
from typing import Callable, NamedTuple

import numpy as np

from .matterdata import H_SOL, S_CRIT, V_APP


class VolumeRule(NamedTuple):
    volume: Callable      # (mw, ms, v) -> V
    water: Callable       # (V, ms, v) -> mw


def _sum_vm(ms, v):
    return np.sum(np.asarray(ms, np.float64) * np.asarray(v, np.float64), axis=-1)


def linear_volume(mw, ms, v):
    return np.asarray(mw, np.float64) + _sum_vm(ms, v)


def linear_water(V, ms, v):
    return np.asarray(V, np.float64) - _sum_vm(ms, v)


LINEAR = VolumeRule(linear_volume, linear_water)


def v_of(*species):
    """Apparent specific volumes (ml/g) for a species list, T6's private copy."""
    return np.array([V_APP[s] for s in species], np.float64)


def solution_volume(mw, ms, v, rule=LINEAR):
    """Volume (ml) of mw grams of water with ms grams of solutes (last axis)."""
    return rule.volume(mw, ms, v)


def water_grams(V, ms, v, rule=LINEAR):
    """The inverse: grams of water in a solution of volume V."""
    return rule.water(V, ms, v)


def conc(m, V):
    """g solute per ml of solution."""
    return np.asarray(m, np.float64) / np.asarray(V, np.float64)


def density(mw, ms, v, rule=LINEAR):
    """g/ml of the solution."""
    V = rule.volume(mw, ms, v)
    return (np.asarray(mw, np.float64) + np.sum(ms, axis=-1)) / V


# ---- dissolving and precipitating -------------------------------------------
def dissolve_step(m_solid, m_dissolved, mw, b_sat, kappa, dt):
    """Crystals dissolve toward saturation. The target is
    min(all the solute there is, b_sat * mw); the approach is exponential,
    1 - exp(-kappa dt), so it NEVER passes c_sat for any dt. kappa is a rate (1/s):
    contact area and stirring are in it. Supersaturation is left alone (that is
    precipitate_step). Returns (m_solid, m_dissolved, grams dissolved)."""
    m_solid = np.asarray(m_solid, np.float64)
    m_dissolved = np.asarray(m_dissolved, np.float64)
    target = np.minimum(m_solid + m_dissolved, np.asarray(b_sat) * np.asarray(mw))
    d = np.maximum(target - m_dissolved, 0.0) * (-np.expm1(-np.asarray(kappa) * dt))
    d = np.minimum(d, m_solid)
    return m_solid - d, m_dissolved + d, d


def precipitate_step(m_dissolved, mw, b_sat, s_crit, tau, dt, seeded=False):
    """Grams that crystallise in dt. Nothing happens until the load passes
    s_crit * b_sat (sugar 1.1-1.2, salt 1.0-1.02, u); a seed (a crystal already
    there, or True) removes the barrier. Then the excess over b_sat relaxes with tau
    (s). Returns grams precipitated (>= 0, never below saturation)."""
    m_dissolved = np.asarray(m_dissolved, np.float64)
    sat = np.asarray(b_sat) * np.asarray(mw)
    limit = np.where(seeded, sat, np.asarray(s_crit) * sat)
    excess = np.maximum(m_dissolved - sat, 0.0)
    return np.where(m_dissolved > limit, excess * (-np.expm1(-dt / np.asarray(tau))), 0.0)


def evaporate_concentrate(mw, m_dissolved, m_crystal, dw, b_sat):
    """Water leaves, the solute stays. Whatever the water left behind cannot hold
    comes out as crystals at once. Mass is exact. Returns (mw, m_dissolved, m_crystal).
    A 3.5 wt % salt water starts to crystallise with about 10 % of its water left (c)."""
    mw = np.asarray(mw, np.float64)
    dw = np.minimum(np.asarray(dw, np.float64), mw)
    mw2 = mw - dw
    d2 = np.minimum(np.asarray(m_dissolved, np.float64), np.asarray(b_sat) * mw2)
    x2 = np.asarray(m_crystal, np.float64) + (np.asarray(m_dissolved, np.float64) - d2)
    return mw2, d2, x2


def heat_of_solution(species, g):
    """Energy change of the cell, J, when g grams dissolve (g < 0: crystallise).
    NaCl +66 J/g and sucrose +17 J/g TAKE heat, so the change is negative."""
    return -H_SOL[species] * np.asarray(g, np.float64)


def s_crit(species):
    return S_CRIT[species]
