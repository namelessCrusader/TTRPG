"""MATSTRESS: free strain and self-stress (L15), and what replaces the 3-cell
TSHOCK stencil (spec T9-material v2, sections 3.7 and 4.8).

STAND-ALONE. Pure numpy and math; no World. `combine` comes from matdamage
inside try/except with a local copy.

WHY NOT A STENCIL. A second difference of three cell temperatures changes by
two times between 5 cm and 10 cm cells (measured against a 1-D conduction
oracle: granite, 800 K step, 126 s: exact surface stress -385 MPa; the stencil
gives 54 MPa at 5 cm and 27 MPa at 10 cm). The law here uses PHYSICAL
temperatures, so it does not depend on the voxel size:

    T_surface = T_cell_mean + skin          (the stored skin, SKIN_T, is the
                                             skin node's temperature above
                                             the cell mean; flux driven)
    T_body    = mean of T along the flux line through the thing (at most 0.5 m)
    sigma_surface = - E alpha (T_surface - T_body) / (1 - nu)
                    compression when the surface is hotter than the body mean

The skin node is the layer sqrt(a t0) deep, so it is cooler than the very
surface: the law reads about 23 percent low against the oracle for granite
and glass (tested to stay inside 30 percent).

Numbers marked (u) are unverified.
"""
import math

import numpy as np

def _combine_copy(D, dD):
    """Independent causes: 1 - (1 - D)(1 - dD), clipped to [0, 0.999]. A local
    copy of matdamage.combine, kept equal to it by a cross test; used when
    matdamage cannot be imported."""
    r = np.clip(1.0 - (1.0 - np.asarray(D, float)) * (1.0 - np.asarray(dD, float)), 0.0, 0.999)
    return float(r) if r.ndim == 0 else r


try:
    from .matdamage import combine
except ImportError:
    combine = _combine_copy

SKIN_T0 = 126.0             # s: the skin's depth time (materials.SKIN_T[0])
BODY_MAX_M = 0.5            # m: longest flux line read for the body mean
ICE_STRAIN = 0.09           # volume fraction of water that becomes ice
K_WATER = 2.2e9             # Pa: bulk modulus of water
P_ATM = 1.0e5               # Pa
BETA_WATER = 2.07e-4        # 1/K at 20 C (u)
KINGERY_C = 3.25            # Hasselman/Kingery thin-body coefficient (u for the exact value)
DAMAGE_RATE = 0.01          # D_other added per event at u = 1, scaled by u^4 (game choice, u)

# volume jump at a temperature crossing: key -> (T_C, dV as a fraction of the volume, direction)
# 'cooling': +dV when T falls through T_C, -dV when it rises; 'heating': the reverse.
PHASE = {
    "ice": (0.0, ICE_STRAIN, "cooling"),             # water to ice, +9 % (verified)
    "quartz": (573.0, 0.004, "heating"),             # alpha to beta quartz, about +0.4 % (u)
}
# volume of the product over the volume it came from (not a temperature step)
REACTION_RATIO = {"rust": 3.0, "slaking": 2.0}       # rust 2-4 times the iron (u); lime slaking x2 (u)


# ── free strain and restrained stress ───────────────────────────────────────
def free_strain(alpha, dT, beta=0.0, dw=0.0, dV_phase=0.0):
    """Strain a piece would have if nothing held it, per axis: alpha dT +
    beta d(moisture) + (volume fraction change of a phase change)/3."""
    return np.asarray(alpha, float) * dT + beta * dw + dV_phase / 3.0


def restrained_stress(E, nu, eps, constraint="uniaxial"):
    """Stress where the strain eps is held back. Uniaxial E eps, biaxial (a plate
    or a surface layer) E eps/(1-nu), triaxial E eps/(1-2 nu). Compression is
    negative when eps is an expansion."""
    E = np.asarray(E, float)
    nu = np.asarray(nu, float)
    if constraint == "uniaxial":
        return E * eps
    if constraint == "biaxial":
        return E * eps / (1.0 - nu)
    if constraint == "triaxial":
        return E * eps / (1.0 - 2.0 * nu)
    raise ValueError(f"unknown constraint {constraint!r}")


def surface_stress(E, nu, alpha, T_surface, T_body_mean):
    """- E alpha (T_surface - T_body)/(1 - nu): negative (compression) for a hot
    surface, positive (tension) for a cold one."""
    return -np.asarray(E, float) * alpha * (np.asarray(T_surface, float) - T_body_mean) / (1.0 - np.asarray(nu, float))


def surface_temperature(T_cell_mean, skin_dT):
    """The stored skin is the skin above the cell mean (SKIN_T)."""
    return np.asarray(T_cell_mean, float) + np.asarray(skin_dT, float)


def body_mean(T_line, fill_line, h=None, max_m=BODY_MAX_M):
    """Mean temperature along the flux line through the thing, weighted by how
    much solid each cell holds, over at most max_m (needs h to count cells)."""
    T = np.asarray(T_line, float)
    f = np.asarray(fill_line, float)
    n = len(T) if h is None else min(len(T), max(1, int(round(max_m / h))))
    w = f[:n]
    return float((T[:n] * w).sum() / w.sum()) if w.sum() > 0 else float(T[0])


def interior_tension(E, nu, alpha, T_int, T_body_mean):
    """Interior cooler than the body mean (a heated body): + E alpha (T_body - T_int)/(1 - nu)
    in tension. Writes damage, not a shatter."""
    return np.asarray(E, float) * alpha * (T_body_mean - np.asarray(T_int, float)) / (1.0 - np.asarray(nu, float))


def skin_depth(k, rho, cp, t=SKIN_T0):
    """sqrt(a t), a = k/(rho cp): the heated skin's depth, metres."""
    return math.sqrt(k / (rho * cp) * t)


# ── failure ─────────────────────────────────────────────────────────────────
def spall(rec_eff, sigma_surface, skin_depth_m, exposed, pore_p=0.0):
    """Does the surface flake? Compression at the surface (plus the steam
    pressure in the pores, pore_p) reaching the effective compressive strength
    sc_eff, and ONLY where a face is exposed to a cell that is not solid. A face
    against another solid does not spall; the caller adds damage instead
    (`stress_damage`). Returns (fails, depth_m): the flake is the heated skin."""
    sc = np.asarray(rec_eff["sc"], float)
    drive = np.maximum(-np.asarray(sigma_surface, float), 0.0) + pore_p
    fails = np.asarray(exposed, bool) & (drive >= sc) & (sc > 0)
    return fails, np.where(fails, skin_depth_m, 0.0)


def spall_flake(smass_g, E_J, rho, depth_m, area_m2):
    """A flake of depth `depth_m` off a face of `area_m2`: mass rho x depth x area
    (never more than the cell holds), the heat it carries off by mass share, as
    `_shatter` splits a cell. Returns (flake_g, flake_J, left_g, left_J); the
    flake plus what is left is exactly what the cell held."""
    m = np.asarray(smass_g, float)
    fl = np.minimum(m, rho * depth_m * area_m2 * 1000.0)
    share = np.where(m > 0, fl / np.where(m > 0, m, 1.0), 0.0)
    eflake = np.asarray(E_J, float) * share
    return fl, eflake, m - fl, np.asarray(E_J, float) - eflake


def stress_damage(u, D=0.0):
    """Damage from a stress that is under the limit (u < 1), or a piece at u >= 1.
    dD = DAMAGE_RATE u^4 per event (a hot rock face cracking a little each time,
    (u)), combined with the damage already there. Returns (D_new, piece)."""
    u = np.asarray(u, float)
    dD = np.where(u >= 1.0, 0.0, DAMAGE_RATE * u ** 4)
    return combine(D, dD), u >= 1.0


# ── thermal shock of a thin piece ───────────────────────────────────────────
def R_shock(sf, nu, E, alpha):
    """Thermal shock parameter sf (1 - nu)/(E alpha), kelvin."""
    return sf * (1.0 - nu) / (E * alpha)


def biot(h_c, a, k):
    """Bi = h_c a / k, a the half thickness."""
    return h_c * a / k


def kingery_dT(R, Bi):
    """Hasselman/Kingery: the quench temperature difference a thin piece takes,
    R (1 + 3.25/Bi). Rises as Bi falls; Bi -> infinity returns R."""
    return R * (1.0 + KINGERY_C / np.asarray(Bi, float))


# ── a sealed vessel: freezing and heating are two inputs of ONE law ─────────
def vessel_compliance(r, t, E, nu):
    """Volume compliance of a thin closed cylinder: (r/(t E)) (5/2 - 2 nu), 1/Pa."""
    return (r / (t * E)) * (2.5 - 2.0 * nu)


def vessel_free_strain(contents, alpha_wall=0.0):
    """Net free volume strain of the contents against the wall: ice forming
    (0.09 x frozen fraction) plus liquid expansion minus the wall's own
    (beta - 3 alpha) dT. contents: frozen (0..1), dT (K), beta (1/K)."""
    return (ICE_STRAIN * contents.get("frozen", 0.0)
            + (contents.get("beta", BETA_WATER) - 3.0 * alpha_wall) * contents.get("dT", 0.0))


def sealed_vessel_pressure(vessel, contents):
    """Pressure in a sealed vessel, hoop stress and whether it bursts.

    The free strain eps of the contents (frozen fraction, rise dT) is taken up by
    the headspace gas (fraction phi of the volume, air at P_ATM: phi p/(p0 + p)),
    the liquid's compressibility (1/K) and the wall's compliance C_v:

        eps = phi p/(p0 + p) + p (1/K + C_v)

    which is a quadratic in p and is solved exactly. Hoop stress = (p + p_vapour)
    r/t; it bursts when the hoop reaches sf_eff. A vent never builds pressure:
    pass vent=True in contents. vessel: r, t, E, nu, sf (and alpha for the wall);
    contents: frozen, dT, beta, phi, p_vapour, K, vent."""
    r, t = vessel["r"], vessel["t"]
    eps = vessel_free_strain(contents, vessel.get("alpha", 0.0))
    phi = contents.get("phi", 0.0)
    pv = contents.get("p_vapour", 0.0)
    if contents.get("vent", False) or eps <= 0.0:
        p = 0.0
    else:
        c = 1.0 / contents.get("K", K_WATER) + vessel_compliance(r, t, vessel["E"], vessel["nu"])
        Bq = c * P_ATM + phi - eps
        p = (-Bq + math.sqrt(Bq * Bq + 4.0 * c * eps * P_ATM)) / (2.0 * c)
    hoop = (p + pv) * r / t
    return p, hoop, hoop >= vessel["sf"]


# ── phase strains ───────────────────────────────────────────────────────────
def phase_strain(key, T_old, T_new):
    """Linear free strain of a temperature crossing of the phase `key`: dV/3 when
    T goes through the transition the way the table says, -dV/3 the other way,
    0 when it does not cross. (rust and slaking are reactions: `reaction_strain`.)"""
    Tt, dV, direction = PHASE[key]
    down = T_old > Tt >= T_new
    up = T_old <= Tt < T_new
    fwd, back = (down, up) if direction == "cooling" else (up, down)
    return dV / 3.0 if fwd else (-dV / 3.0 if back else 0.0)


def reaction_strain(key, fraction):
    """Linear free strain when `fraction` of a solid has turned into the product
    that takes REACTION_RATIO[key] times the volume."""
    return (REACTION_RATIO[key] - 1.0) * fraction / 3.0
