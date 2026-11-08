"""SELECTORS (wave 13, T1): how much of each component a physical rule moves.

Every function here is a pure function on arrays, one number per ENTRY (a
component of a row): it returns a share phi in [0, 1] of that component that
moves or changes form in this step, plus (for density) which way. A law then
multiplies phi by the entry's grams and calls take/put (mixtable.py). The
selectors know nothing about cells, rows or the world.

THE USER'S QUESTION (spec 4.6): can a mix be separated by real laws? The answer
is the FORM of each component. Free forms (dissolved, crystal, grain) are pulled
apart by physics; BOUND is held in the solid body and is not touched by any
SEPARATION rule. Three kinds of rule:
  SEPARATION   never touch BOUND: boil-off, density (settle, float), size
               (sieve, winnow), wash and dissolve, magnet.
  DIFFUSION    may move BOUND, slowly, by its own Arrhenius rate.
  PHASE_CHANGE may convert BOUND, by phase data (the lever rule).
The form masks are DATA (MASKS below); each selector applies its own mask on
the way out, so a property test over random entries can check phi = 0 for BOUND.

Formulas (spec 4.6; (u) = unverified, from memory):
  boil-off    y_i = x_i gamma_i P_i / P, P_i from the Antoine row.
  density     Stokes v = (rho_p - rho_f) g d^2 / (18 mu); above Re 1 the
              Schiller-Naumann drag Cd = 24/Re (1 + 0.15 Re^0.687), with Cd not
              below 0.44 (the Newton plateau, beyond its 800 range); crowded
              settling v0 (1 - phi)^n, Richardson-Zaki n by Re. NOTE the speed
              jumps up to 13 percent where the law switches at Re 1 (the spec's
              switch); the Richardson-Zaki n joins within 2 percent.
  size        sieve: 0 for d >= a, else 1 - exp(-nu dt open (1 - d/a)^2);
              winnow: min(1, (u - v_t) dt / dz), 0 for u <= v_t.
  dissolve    still liquid, Sherwood 2: d^2 = d0^2 - 8 D (c_sat - c) t / rho_p.
  melt        a crystal or grain melts as heat allows; BOUND only by the lever
              rule fraction (T4's phase data supplies the compositions).
  magnet      sigma(B, T) = sigma_s(T) tanh(B / B_k), B_k = mu0 N Ms, N = 1/3;
              captured when sigma gradB > g (1 + drag/weight).
  diffusion   D = D0 exp(-Q / (R T)), share 1 - exp(-D dt / dz^2).
Units: SI (m, kg, s, Pa) except mu in mPa s and T in degC, as in mixrules.py.
"""
import numpy as np

from . import mixrules as MR
from . import species as SP
from .species_seed import BOUND, CRYSTAL, DISSOLVED, GRAIN

__all__ = ["SEPARATION", "DIFFUSION", "PHASE_CHANGE", "MASKS", "rule_kind", "form_mask",
           "apply_mask", "psat", "relative_volatility", "vapour_fraction",
           "boiling_point_ideal", "boil_off_phi", "terminal_speed", "richardson_zaki_n",
           "hindered_speed", "settle_phi", "density_phi", "sieve_phi", "winnow_phi",
           "air_terminal_speed", "wash_phi", "dissolve_d", "dissolve_time", "dissolve_phi",
           "lever_fraction", "melt_phi", "phase_change_phi", "sigma_sat_T", "magnet_sigma",
           "magnet_min_gradient", "magnet_phi", "register_diffusion", "DIFFUSION_DATA",
           "arrhenius_D", "diffusion_phi", "diffusion_phi_pair", "G", "MU0", "R_GAS",
           "MMHG"]

G = 9.81                      # m/s^2
MU0 = 4e-7 * np.pi            # T m / A
R_GAS = 8.314462618           # J/mol/K
MMHG = 133.322                # Pa per mmHg
KELVIN = 273.15
AIR_RHO, AIR_MU = 1.2, 0.0181             # kg/m^3, mPa s (20 C)
MIN_CD = 0.44                 # Newton plateau, used past the Schiller-Naumann range

SEPARATION, DIFFUSION, PHASE_CHANGE = "separation", "diffusion", "phase_change"
# rule -> (kind, mask over [DISSOLVED, CRYSTAL, GRAIN, BOUND]); GRAIN may be the
# string "soluble": allowed only for a grain that dissolves (spec 4.6 table).
MASKS = {
    "boil_off": (SEPARATION, (True, False, False, False)),
    "density": (SEPARATION, (False, True, True, False)),
    "size": (SEPARATION, (False, True, True, False)),
    "wash": (SEPARATION, (True, True, "soluble", False)),
    "magnet": (SEPARATION, (False, True, True, False)),
    "diffusion": (DIFFUSION, (False, False, False, True)),
    "melt": (PHASE_CHANGE, (False, True, True, True)),
}


def rule_kind(rule):
    """SEPARATION, DIFFUSION or PHASE_CHANGE."""
    return MASKS[rule][0]


def form_mask(rule, form, soluble=None):
    """Which entries a rule may act on, from their form (bool array)."""
    form = np.asarray(form, np.int64)
    out = np.zeros(form.shape, bool)
    for f, allowed in enumerate(MASKS[rule][1]):
        if allowed == "soluble":
            sol = np.zeros(form.shape, bool) if soluble is None else np.asarray(soluble, bool)
            out |= (form == f) & sol
        elif allowed:
            out |= form == f
    return out


def apply_mask(rule, form, phi, soluble=None):
    """phi where the rule may act on the form, 0 elsewhere."""
    return np.where(form_mask(rule, form, soluble), phi, 0.0)


# ---- (a) boiling: vapour pressure and the composition of what leaves ----------
_ANT = {}


def _antoine_table(slot):
    t = _ANT.get(slot)
    if t is None:
        n = len(SP.SPECIES)
        t = np.full((n, 3), np.nan)
        for s in SP.SPECIES:
            if s is not None:
                a = s.get("antoine", slot, None)
                if a:
                    t[s.id] = a[:3]
        _ANT[slot] = t
    return t


def psat(species, T_C, slot="ref"):
    """Vapour pressure in Pa from the species' Antoine row,
    log10 P(mmHg) = A - B / (C + T). NaN for a species with no row (it is taken as
    not volatile by boil_off_phi). Outside the row's fitted range it extrapolates."""
    a = _antoine_table(slot)[np.asarray(species, np.int64)]
    T = np.asarray(T_C, np.float64)
    return 10.0 ** (a[..., 0] - a[..., 1] / (a[..., 2] + T)) * MMHG


def relative_volatility(sp_a, sp_b, T_C, slot="ref"):
    """P_a / P_b at T (the ideal relative volatility; gamma is T4's)."""
    return psat(sp_a, T_C, slot) / psat(sp_b, T_C, slot)


def vapour_fraction(x, species, T_C, P_Pa, gamma=1.0, slot="ref"):
    """Mole fractions y of the vapour over liquid mole fractions x [..., K]:
    y_i = x_i gamma_i P_i / P (not normalised: sum y = 1 only at the boiling
    point). A species with no Antoine row has P_i = 0."""
    p = np.nan_to_num(psat(species, np.asarray(T_C)[..., None], slot))
    return np.asarray(x, np.float64) * gamma * p / np.asarray(P_Pa)[..., None]


def boiling_point_ideal(x, species, P_Pa=101325.0, gamma=1.0, slot="ref", lo=-50.0, hi=400.0):
    """The temperature (degC) at which sum x_i gamma_i P_i(T) = P, by bisection.
    x: mole fractions [n, K]; species: ids [n, K] (or [K])."""
    x = np.atleast_2d(np.asarray(x, np.float64))
    species = np.broadcast_to(np.asarray(species, np.int64), x.shape)
    lo = np.full(len(x), float(lo))
    hi = np.full(len(x), float(hi))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        tot = (x * gamma * np.nan_to_num(psat(species, mid[:, None], slot))).sum(axis=1)
        hot = tot > P_Pa
        hi = np.where(hot, mid, hi)
        lo = np.where(hot, lo, mid)
    return 0.5 * (lo + hi)


def boil_off_phi(cell, species, form, grams, T_C, P_Pa, boiled_g, gamma=1.0, slot="ref"):
    """Share of each DISSOLVED entry that leaves as vapour when boiled_g[row] grams
    boil off: the vapour has mole fractions y_i ~ x_i gamma_i P_i, the mass leaving
    of entry i is boiled_g * y_i M_i / sum(y_j M_j), phi = min(1, that / grams_i).
    Entries are flat (cell = row of each entry); T_C, P_Pa, boiled_g per row. A
    species with no vapour pressure data, and any form but DISSOLVED, gives 0."""
    cell = np.asarray(cell, np.int64)
    species = np.asarray(species, np.int64)
    form = np.asarray(form, np.int64)
    grams = np.asarray(grams, np.float64)
    n = int(cell.max()) + 1 if len(cell) else 0
    M = MR.sp_arrays(slot)["M"][species]
    dis = form == DISSOLVED
    mol = np.where(dis & np.isfinite(M), grams / np.where(np.isfinite(M), M, 1.0), 0.0)
    xs = mol / np.maximum(np.bincount(cell, weights=mol, minlength=n)[cell], 1e-300)
    p = np.nan_to_num(psat(species, np.asarray(T_C, np.float64)[cell], slot))
    y = xs * gamma * p
    ym = y * np.where(np.isfinite(M), M, 0.0)
    tot = np.bincount(cell, weights=ym, minlength=n)[cell]
    with np.errstate(divide="ignore", invalid="ignore"):
        leave = np.asarray(boiled_g, np.float64)[cell] * ym / tot
        phi = np.where((tot > 0) & (grams > 0), np.minimum(1.0, leave / grams), 0.0)
    return apply_mask("boil_off", form, phi)


# ---- (b) density: settle and float ----------------------------------------------
def richardson_zaki_n(Re):
    """The exponent n of v = v0 (1 - phi)^n: 4.65 below Re 0.2, 4.4 Re^-0.03 to 1,
    4.4 Re^-0.1 to 500, 2.4 above (u, wall effect ignored; the end values agree
    with Richardson and Zaki 1954 as quoted: 4.65 and 2.39)."""
    Re = np.maximum(np.asarray(Re, np.float64), 1e-30)
    return np.where(Re < 0.2, 4.65, np.where(Re < 1.0, 4.4 * Re ** -0.03,
                                             np.where(Re < 500.0, 4.4 * Re ** -0.1, 2.4)))


def _cd(Re):
    Re = np.maximum(Re, 1e-30)
    return np.maximum(24.0 / Re * (1.0 + 0.15 * Re ** 0.687), MIN_CD)


def terminal_speed(d, rho_p, rho_f, mu_mPas, g=G):
    """Signed terminal speed in m/s (positive: the particle sinks, negative: it
    floats). Stokes where its Reynolds number is at most 1, else the
    Schiller-Naumann drag solved for v by bisection. d in m, densities in kg/m^3."""
    d, rho_p, rho_f, mu = np.broadcast_arrays(*(np.asarray(a, np.float64) for a in
                                                 (d, rho_p, rho_f, mu_mPas)))
    mu = mu * 1e-3
    dr = rho_p - rho_f
    vs = dr * g * d * d / (18.0 * mu)
    Re_s = rho_f * np.abs(vs) * d / mu
    # |v| solves  Cd(Re) v^2 = 4 |dr| g d / (3 rho_f)
    rhs = 4.0 * np.abs(dr) * g * d / (3.0 * np.maximum(rho_f, 1e-12))
    lo = np.full(d.shape, np.log(1e-9))
    hi = np.full(d.shape, np.log(1e4))
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        v = np.exp(mid)
        f = _cd(rho_f * v * d / mu) * v * v - rhs
        hi = np.where(f > 0, mid, hi)
        lo = np.where(f > 0, lo, mid)
    vn = np.exp(0.5 * (lo + hi))
    out = np.where(Re_s <= 1.0, vs, np.sign(dr) * vn)
    return np.where(d > 0, out, 0.0)


def hindered_speed(v0, phi_vol, Re):
    """Crowded settling v0 (1 - phi)^n (Richardson-Zaki)."""
    return np.asarray(v0) * (1.0 - np.clip(phi_vol, 0.0, 1.0)) ** richardson_zaki_n(Re)


def settle_phi(d, rho_p, rho_f, mu_mPas, dt, dz, phi_vol=0.0, g=G):
    """(phi, direction): the share that crosses a cell face of height dz in dt,
    min(1, |v| dt / dz), and +1 (sinks) or -1 (floats, 0 if neutral)."""
    v0 = terminal_speed(d, rho_p, rho_f, mu_mPas, g)
    mu = np.asarray(mu_mPas, np.float64) * 1e-3
    Re = np.asarray(rho_f, np.float64) * np.abs(v0) * np.asarray(d, np.float64) / mu
    v = hindered_speed(np.abs(v0), phi_vol, Re)
    return np.minimum(1.0, v * dt / dz), np.sign(v0)


def density_phi(form, d, rho_p, rho_f, mu_mPas, dt, dz, phi_vol=0.0, g=G):
    """settle_phi under the density form mask (crystal and grain only)."""
    phi, direction = settle_phi(d, rho_p, rho_f, mu_mPas, dt, dz, phi_vol, g)
    return apply_mask("density", form, phi), direction


# ---- (c) size: sieve and winnow ------------------------------------------------------
def sieve_phi(form, d, aperture, open_frac, nu_intensity, dt, aspect=1.0):
    """Share of each crystal or grain entry that passes a square aperture in dt:
    a sphere of effective diameter d * aspect with d >= aperture cannot pass (a hard
    0); smaller ones pass with chance open_frac (1 - d/a)^2 per contact, at
    nu_intensity contacts per second."""
    d_eff = np.asarray(d, np.float64) * aspect
    a = np.asarray(aperture, np.float64)
    p = np.where(d_eff < a, open_frac * (1.0 - d_eff / a) ** 2, 0.0)
    phi = 1.0 - np.exp(-nu_intensity * dt * p)
    return apply_mask("size", form, phi)


def air_terminal_speed(d, rho_p):
    """Terminal speed in still air (m/s), same drag law with rho_f = 1.2."""
    return terminal_speed(d, rho_p, AIR_RHO, AIR_MU)


def winnow_phi(form, u, v_t, dt, dz):
    """Share carried up by an air stream of speed u over one cell of height dz:
    min(1, (u - v_t) dt / dz), 0 when u <= v_t."""
    phi = np.minimum(1.0, np.maximum(np.asarray(u, np.float64) - v_t, 0.0) * dt / dz)
    return apply_mask("size", form, phi)


# ---- wash, leach, dissolve -------------------------------------------------------------
def wash_phi(form, soluble, f_fluid, dissolve_share=0.0):
    """DISSOLVED entries leave with the fluid share f_fluid; CRYSTAL entries (and
    GRAIN ones only if soluble) dissolve by dissolve_share (from dissolve_phi);
    BOUND is never washed out (a desorption row, L3, if wanted)."""
    form = np.asarray(form, np.int64)
    phi = np.where(form == DISSOLVED, f_fluid, dissolve_share)
    return apply_mask("wash", form, phi, soluble)


def dissolve_d(d0, rho_p, D, c_sat, c, t):
    """Diameter after t seconds of dissolving in still liquid (Sherwood 2):
    d^2 = d0^2 - 8 D (c_sat - c) t / rho_p, floored at 0. SI units; c and c_sat in
    kg/m^3 of solution."""
    drive = np.maximum(np.asarray(c_sat, np.float64) - c, 0.0)
    d2 = np.asarray(d0, np.float64) ** 2 - 8.0 * D * drive * t / rho_p
    return np.sqrt(np.maximum(d2, 0.0))


def dissolve_time(d0, rho_p, D, c_sat, c):
    """Seconds to dissolve a particle of diameter d0 completely (inf if c >= c_sat)."""
    drive = np.maximum(np.asarray(c_sat, np.float64) - c, 0.0)
    with np.errstate(divide="ignore"):
        return np.where(drive > 0, rho_p * np.asarray(d0, np.float64) ** 2 / (8.0 * D * np.where(drive > 0, drive, 1.0)),
                        np.inf)


def dissolve_phi(d0, rho_p, D, c_sat, c, dt):
    """Mass share of a particle that dissolves in dt: 1 - (d(dt) / d0)^3 (the number
    of particles stays the same). 0 when the liquid is saturated (c >= c_sat)."""
    d0 = np.asarray(d0, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = dissolve_d(d0, rho_p, D, c_sat, c, dt) / d0
    return np.where(d0 > 0, 1.0 - r ** 3, 0.0)


# ---- (e) melting by composition -------------------------------------------------------------
def lever_fraction(x_bulk, x_solid, x_liquid):
    """The liquid fraction by the lever rule between solidus and liquidus
    compositions: (x_bulk - x_solid) / (x_liquid - x_solid), clipped to [0, 1]."""
    x_bulk, x_s, x_l = (np.asarray(a, np.float64) for a in (x_bulk, x_solid, x_liquid))
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (x_bulk - x_s) / (x_l - x_s)
    return np.clip(np.nan_to_num(f), 0.0, 1.0)


def melt_phi(form, Tm, T_C, grams, Lf, heat_J=np.inf):
    """Share that melts: a CRYSTAL or GRAIN entry with Tm < T melts as heat allows
    (min(1, heat / (grams Lf))); a BOUND entry never melts by this call (use
    phase_change_phi with the lever rule)."""
    Tm, T = np.asarray(Tm, np.float64), np.asarray(T_C, np.float64)
    need = np.maximum(np.asarray(grams, np.float64) * Lf, 1e-300)
    phi = np.where(Tm < T, np.minimum(1.0, heat_J / need), 0.0)
    form = np.asarray(form, np.int64)
    return np.where((form == CRYSTAL) | (form == GRAIN), phi, 0.0)


def phase_change_phi(form, Tm, T_C, grams, Lf, lever=None, heat_J=np.inf):
    """The PHASE_CHANGE rule on every form: CRYSTAL and GRAIN by melt_phi; BOUND only
    through the lever-rule liquid fraction at T (`lever`, from T4's phase data; 0
    where none is given); DISSOLVED does not melt."""
    form = np.asarray(form, np.int64)
    lv = np.zeros(form.shape) if lever is None else np.asarray(lever, np.float64)
    phi = np.where(form == BOUND, lv, melt_phi(form, Tm, T_C, grams, Lf, heat_J))
    return apply_mask("melt", form, phi)


# ---- (f) magnet --------------------------------------------------------------------------------
_MAG = {}


def _mag_table(slot):
    t = _MAG.get(slot)
    if t is None:
        n = len(SP.SPECIES)
        t = np.full((n, 4), np.nan)
        for s in SP.SPECIES:
            if s is not None:
                for j, f in enumerate(("sigma_s", "Tc", "beta", "rho_s")):
                    v = s.get(f, slot, None)
                    if v is not None:
                        t[s.id, j] = v
        _MAG[slot] = t
    return t


def sigma_sat_T(sigma_s0, Tc, beta, T_C):
    """Saturation moment per kg at T: sigma_s0 g(T), g = ((1 - T/Tc)/(1 - 293.15/Tc))^beta
    (kelvin), exactly 1 at 20 C and 0 at and above the Curie point."""
    Tk, Tck = np.asarray(T_C, np.float64) + KELVIN, np.asarray(Tc, np.float64) + KELVIN
    with np.errstate(invalid="ignore", divide="ignore"):
        g = ((1.0 - Tk / Tck) / (1.0 - 293.15 / Tck)) ** beta
    return np.where(Tk < Tck, sigma_s0 * np.nan_to_num(g), 0.0)


def magnet_sigma(sigma_s0, Tc, beta, rho_s, B, T_C, N=1.0 / 3.0):
    """Moment per kg in field B (tesla): sigma_s(T) tanh(B / B_k), B_k = mu0 N Ms,
    Ms = sigma_s(T) rho (demagnetising factor N = 1/3 for a filing)."""
    s = sigma_sat_T(sigma_s0, Tc, beta, T_C)
    Bk = MU0 * N * s * rho_s
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(s > 0, s * np.tanh(np.asarray(B, np.float64) / np.where(Bk > 0, Bk, 1.0)), 0.0)


def magnet_min_gradient(sigma_s0, Tc, beta, rho_s, B, T_C, drag_ratio=0.0, g=G):
    """The least field gradient (T/m) that holds a grain: g (1 + drag/weight) / sigma.
    inf for a grain with no moment."""
    sg = magnet_sigma(sigma_s0, Tc, beta, rho_s, B, T_C)
    with np.errstate(divide="ignore"):
        return np.where(sg > 0, g * (1.0 + drag_ratio) / np.where(sg > 0, sg, 1.0), np.inf)


def magnet_phi(species, form, B, gradB, T_C, drag_ratio=0.0, slot="ref"):
    """1.0 for a crystal or grain entry that the magnet captures (sigma gradB >
    g (1 + drag/weight)), else 0. A species with no sigma_s is never captured."""
    sp = np.asarray(species, np.int64)
    t = _mag_table(slot)[sp]
    need = magnet_min_gradient(np.nan_to_num(t[..., 0]), np.nan_to_num(t[..., 1]),
                               np.nan_to_num(t[..., 2], nan=0.36), np.nan_to_num(t[..., 3], nan=1.0),
                               B, T_C, drag_ratio)
    got = (np.asarray(gradB, np.float64) > need) & np.isfinite(t[..., 0])
    return apply_mask("magnet", form, got.astype(np.float64))


# ---- diffusion in the solid -------------------------------------------------------------------------
# (solute species, host species) -> (D0 m^2/s, Q J/mol). Carbon in austenite: D is
# about 1e-11 m^2/s at 900 C (spec 4.6); D0 = 4e-5 and Q = 148 kJ/mol give 1.0e-11 (u).
DIFFUSION_DATA = {(20, 3): (4.0e-5, 148e3)}


def register_diffusion(solute, host, D0, Q):
    """Add or replace the Arrhenius row of a solute in a host."""
    DIFFUSION_DATA[(int(solute), int(host))] = (float(D0), float(Q))


def arrhenius_D(D0, Q, T_C):
    """D = D0 exp(-Q / (R T)), m^2/s."""
    return D0 * np.exp(-Q / (R_GAS * (np.asarray(T_C, np.float64) + KELVIN)))


def diffusion_phi(form, D, dt, dz):
    """Share of a BOUND entry that moves by solid-state diffusion in dt over a cell
    of size dz: 1 - exp(-D dt / dz^2). Zero for every other form."""
    phi = 1.0 - np.exp(-np.asarray(D, np.float64) * dt / (dz * dz))
    return apply_mask("diffusion", form, phi)


def diffusion_phi_pair(solute, host, form, T_C, dt, dz):
    """diffusion_phi for a (solute, host) pair from DIFFUSION_DATA; 0 where the pair
    has no row (no data, no motion)."""
    solute, host = np.broadcast_arrays(np.asarray(solute, np.int64), np.asarray(host, np.int64))
    D0 = np.zeros(solute.shape)
    Q = np.zeros(solute.shape)
    has = np.zeros(solute.shape, bool)
    for (s, h), (d0, q) in DIFFUSION_DATA.items():
        m = (solute == s) & (host == h)
        D0[m], Q[m], has[m] = d0, q, True
    D = np.where(has, arrhenius_D(np.where(has, D0, 0.0), np.where(has, Q, 1.0), T_C), 0.0)
    return diffusion_phi(form, D, dt, dz)
