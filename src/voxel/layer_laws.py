"""SURFACE-LAYER LAWS (wave 13, T3): pure functions, no World, no pool.

Sections follow the spec (specs/T3-layers_v2.txt) numbers: 4.1 thickness and
rheology, 4.2 deposition from air, 4.3 condensation, 4.4 hold limits, 4.5 drain
and drip, 4.6 soak and imbibition, 4.7 phase points, 4.8 the joint latent pin,
4.9 kinetics and growth, 4.10 contact, 4.12 optics cover, 4.13 friction.

SI unless a name says otherwise (g, cm3, ml). T in C. Compositions are lists of
(component key, grams). Three things here are STAND-INS behind one call each
(F12): freeze_point, boil_point and psat (T4's mixwater owns them; a cross test
in tests/test_layers_flow.py compares them when mixwater exists), and
viscosity (T1 species).
"""
import bisect
import math

import numpy as np

from .layer_data import (AMBIENT, BOUND, CRYSTAL, DISSOLVED, G, GRAIN, K_B_WATER,
                         K_F_WATER, LIQUIDUS_T, LIQUIDUS_WT, R_GAS, SHEAR_RATIO,
                         T_EUTECTIC, TE_MAX, P0_TRANSFER, L_SLIDE_MIN, ROWS, d_of_sz,
                         unpack_key)

TWO_PI = 2.0 * math.pi


# =====================================================================================
# 4.1 THICKNESS, COVER, POROSITY, VISCOSITY, YIELD STRESS
# =====================================================================================
def comp_volumes(comp):
    """(V_liq, V_cry, G_vol, G_mass) of a composition, cm3 and g. DISSOLVED solvent
    and liquid species by rho_l, dissolved solutes by rho_d; CRYSTAL and BOUND by
    rho_c in V_cry; GRAIN by rho_c in G_vol."""
    vl = vc = vg = gm = 0.0
    for ck, g in comp:
        if g <= 0.0:
            continue
        sp, form, _sz = unpack_key(ck)
        r = ROWS[sp]
        if form == DISSOLVED:
            vl += g / (r["rho_d"] if "solute" in r["tags"] else r["rho_l"])
        elif form == GRAIN:
            vg += g / r["rho_c"]
            gm += g
        else:
            vc += g / r["rho_c"]
    return vl, vc, vg, gm


def bulk_volume(v_liq, v_cry, g_vol, por):
    """V = max(V_liq + V_cry + G, V_cry + G/(1 - por)), cm3 (4.1). Liquid fills the
    voids of the grain framework until they are full."""
    por = min(max(por, 0.0), 0.999)
    return max(v_liq + v_cry + g_vol, v_cry + g_vol / (1.0 - por))


def void_capacity(g_vol, por):
    """cm3 of liquid the voids of the grain framework hold."""
    por = min(max(por, 0.0), 0.999)
    return g_vol * (1.0 / (1.0 - por) - 1.0)


def porosity_collapse(por, por_dep, por_wet, v_liq, g_vol):
    """New porosity of a grain framework after wetting (4.1): capillary force
    collapses a fluffy deposit toward por_wet as liquid arrives,
    por -> por_dep + (por_wet - por_dep)*w with w = min(1, V_liq/(G*(1/(1-por_wet)-1))).
    Monotone: it never rises again (dried soot stays compact)."""
    if g_vol <= 0.0 or por_wet >= por_dep:
        return por
    need = g_vol * (1.0 / (1.0 - por_wet) - 1.0)
    w = 1.0 if need <= 0.0 else min(1.0, v_liq / need)
    return min(por, por_dep + (por_wet - por_dep) * w)


def merge_porosity(g1, por1, g2, por2):
    """por' = 1 - G_total / V_framework_total (4.1, merge)."""
    gt = g1 + g2
    if gt <= 0.0:
        return 0.0
    fw = g1 / (1.0 - min(por1, 0.999)) + g2 / (1.0 - min(por2, 0.999))
    return 1.0 - gt / fw


def cover_particulate(kappa_eff, m_a):
    """c = 1 - exp(-kappa_eff * m_a): kappa m2/g, m_a g per m2 of FACE (4.1a, 4.12)."""
    return 1.0 - math.exp(-kappa_eff * m_a)


def cover_overlap(c1, c2):
    """Random overlap of two covers (merge)."""
    return 1.0 - (1.0 - c1) * (1.0 - c2)


def cover_liquid(volume_m3, area_m2, h_eq):
    """c = min(1, V/(A*h_eq)): a lump below the film does not thin out, it beads."""
    if h_eq <= 0.0:
        return 1.0
    return min(1.0, volume_m3 / (area_m2 * h_eq))


def viscosity_water(T_C):
    """Pa s. Vogel fit: 1.002 mPa s at 20 C."""
    return 2.414e-5 * 10.0 ** (247.8 / (T_C + 273.15 - 140.0))


def viscosity_andrade(eta20_mPas, B, T_C):
    """Pa s. eta = eta20*exp(B*(1/T_K - 1/293.15)); B 3500 K for oils (u)."""
    return 1e-3 * eta20_mPas * math.exp(B * (1.0 / (T_C + 273.15) - 1.0 / 293.15))


def liquid_viscosity(row, T_C):
    """Pa s of a liquid species row at T."""
    if row["eta_B"] == 0.0 and row["liq_class"] == "water":
        return viscosity_water(T_C)
    return viscosity_andrade(row["eta20"], row["eta_B"], T_C)


def krieger_dougherty(phi, phi_max=0.6, expo=1.5, cap=1.0e6):
    """eta/eta0 = (1 - phi/phi_max)^(-1.5), capped at 1e6 as phi -> phi_max."""
    x = 1.0 - phi / phi_max
    if x <= 0.0:
        return cap
    return min(cap, x ** (-expo))


def yield_stress(phi, tau_ref, phi_ref):
    """tau_y = tau_ref*(phi/phi_ref)^3 for phi >= 0.02, else 0 (4.1)."""
    if phi < 0.02 or tau_ref <= 0.0:
        return 0.0
    return tau_ref * (phi / phi_ref) ** 3


def lambda_c(sigma, rho_kgm3):
    """Capillary length, m."""
    return math.sqrt(sigma / (rho_kgm3 * G))


# =====================================================================================
# 4.4 HOLD LIMITS
# =====================================================================================
def hold_thickness(orient, rough_m, tau_y, rho_kgm3, sigma, theta_deg, film_m):
    """Thickness a face holds before the liquid leaves (4.4). orient: 'wall',
    'up' or 'down'. film_m is FILM*dx (0.5 mm at 5 cm), the lattice's own meaning
    of a wet cell."""
    hy = tau_y / (rho_kgm3 * G)
    lc = lambda_c(sigma, rho_kgm3)
    h_up = min(2.0 * lc * math.sin(math.radians(theta_deg) / 2.0), film_m)
    if orient == "wall":
        return max(rough_m, hy)
    if orient == "up":
        return h_up
    return min(0.2 * lc, h_up) + hy


# =====================================================================================
# 4.5 DRAIN AND DRIP
# =====================================================================================
def film_flow(h, eta, rho_kgm3, tau_y=0.0):
    """Flow per metre of width, m3/s: Nusselt film with the Bingham correction,
    Q = rho g h^3/(3 eta) * (1 - 1.5 (hy/h) + 0.5 (hy/h)^3), 0 for h <= hy."""
    hy = tau_y / (rho_kgm3 * G)
    if h <= hy or h <= 0.0:
        return 0.0
    r = hy / h
    return rho_kgm3 * G * h ** 3 / (3.0 * eta) * (1.0 - 1.5 * r + 0.5 * r ** 3)


def film_speed(h, eta, rho_kgm3, tau_y=0.0):
    """Mean speed Q/h, m/s."""
    return film_flow(h, eta, rho_kgm3, tau_y) / h if h > 0.0 else 0.0


def jeffreys_thickness(eta, z, rho_kgm3, t):
    """h = sqrt(eta z/(rho g t)): a film draining from the top of a wall, at depth
    z below the top after t seconds (Jeffreys 1930)."""
    return math.sqrt(eta * z / (rho_kgm3 * G * t))


def drain_volume(h, h_hold, eta, rho_kgm3, tau_y, dx, dt, area, cover):
    """Volume m3 that leaves a wall layer of thickness h in one step:
    dV = min(Q*dx*dt, (h - h_hold)*A*cover). Zero at or under the hold limit."""
    if h <= h_hold:
        return 0.0
    q = film_flow(h, eta, rho_kgm3, tau_y)
    return min(q * dx * dt, (h - h_hold) * area * cover)


def tate_drop_volume(sigma, rho_kgm3):
    """Volume of a pendant drop, m3: (4/3) pi (0.85 lambda_c)^3 (Tate)."""
    return 4.0 / 3.0 * math.pi * (0.85 * lambda_c(sigma, rho_kgm3)) ** 3


def make_drop(x, y, z, ml, fluid, pot=0.0):
    """One entry of World.drops in the repo's real 9-field format
    [x, y, z, vx, vy, vz, ml, fluid, pot] (fluids.py). A test fails if it changes."""
    return [float(x), float(y), float(z), 0.0, 0.0, 0.0, float(ml), int(fluid),
            float(pot)]


DROP_FIELDS = 9
DROP_MAX_ML = 5.0               # drops of one tick merge into one parcel of at most this


def pendant_drops(v_excess_m3, v_drop_m3, max_ml=DROP_MAX_ML):
    """(n drops, ml in the parcel, m3 left over). Drops of one tick merge into a
    parcel of at most max_ml; what does not fit waits for the next tick."""
    if v_drop_m3 <= 0.0 or v_excess_m3 < v_drop_m3:
        return 0, 0.0, v_excess_m3
    n = int(v_excess_m3 // v_drop_m3)
    n = min(n, max(1, int(max_ml / (v_drop_m3 * 1e6))))
    ml = n * v_drop_m3 * 1e6
    return n, ml, v_excess_m3 - n * v_drop_m3


# face order, the same as heat._DIRS: +x, -x, +y, -y, +z, -z
DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
UP_FACE, DOWN_FACE = 4, 5


def drain_target(solid, x, y, z, face):
    """Where a wall layer's run goes (4.5), from an injected solid mask (3-D bool).
    Returns (kind, (x, y, z, face)):
      'wall'    the same face of the cell below (the wall goes on)
      'step'    the up face of the solid cell in front and below (a step or the
                floor at the foot): that face is the lattice's, hand-over by 4.4
      'pendant' the down face of this cell (an overhang or edge: it drips)
    A wall layer is on a side face (0-3) of a solid cell whose neighbour in that
    direction is open."""
    nx, ny, nz = solid.shape
    dx, dy, _ = DIRS[face]
    # the cell below
    if z - 1 >= 0 and solid[x, y, z - 1]:
        bx, by = x + dx, y + dy
        open_b = not (0 <= bx < nx and 0 <= by < ny and solid[bx, by, z - 1])
        if open_b:
            return "wall", (x, y, z - 1, face)
        return "step", (bx, by, z - 1, UP_FACE)
    return "pendant", (x, y, z, DOWN_FACE)


# =====================================================================================
# 4.6 SOAK AND IMBIBITION
# =====================================================================================
def soak_increment(S, t_w, dt):
    """Closed-form increment of I(t) = S*sqrt(t_w), g/m2 (no singularity at 0)."""
    return S * (math.sqrt(t_w + dt) - math.sqrt(t_w))


def soak_capacity(rho_l_gcm3, porosity, thickness_m, s_host):
    """g per m2 of face the substrate can still take: rho*porosity*thickness*(1 - s)."""
    return rho_l_gcm3 * 1e6 * porosity * thickness_m * (1.0 - s_host)


def sieve_passes(d_grain, d_pore):
    """GRAIN goes in only if d_grain < 0.2*d_pore (a sieve)."""
    return d_grain < 0.2 * d_pore


def inside_depth(v_liq_m3, porosity, area):
    """d_in = V_liq_in/(porosity*A), m."""
    return v_liq_m3 / (porosity * area) if porosity > 0.0 else 0.0


def imbibition_S(rho_kgm3, por, d_grain, sigma, theta_deg, eta):
    """Sorptivity of a liquid into a grain layer, g/m2/s^0.5:
    S = rho*phi*sqrt(sigma*cos(theta)*r/(2 eta)), r = 0.2 d_grain. cos(theta) <= 0
    (hydrophobic soot) wicks nothing."""
    c = math.cos(math.radians(theta_deg))
    if c <= 0.0 or por <= 0.0 or d_grain <= 0.0:
        return 0.0
    r = 0.2 * d_grain
    return 1000.0 * rho_kgm3 * por * math.sqrt(sigma * c * r / (2.0 * eta))


# =====================================================================================
# 4.7 PHASE POINTS (stand-ins for T4 mixwater, F12)
# =====================================================================================
_WAGNER = (-7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719, 1.80122502)
_TC, _PC = 647.096, 22.064e6


def psat_water(T_C):
    """Saturation pressure over liquid water, Pa. Wagner-Pruss (IAPWS) form: the
    private stand-in for mixwater.psat_water; 101.418 kPa at 100 C."""
    T = T_C + 273.15
    tau = 1.0 - T / _TC
    a = _WAGNER
    s = (a[0] * tau + a[1] * tau ** 1.5 + a[2] * tau ** 3 + a[3] * tau ** 3.5
         + a[4] * tau ** 4 + a[5] * tau ** 7.5)
    return _PC * math.exp(_TC / T * s)


def tsat_water(P):
    """Boiling point of pure water at P Pa, C (Newton on psat_water)."""
    T = 100.0 + 28.0 * math.log(max(P, 1.0) / 101325.0) / (1.0 + 0.2 * abs(math.log(max(P, 1.0) / 101325.0)))
    for _ in range(40):
        f = psat_water(T) - P
        d = (psat_water(T + 1e-4) - psat_water(T - 1e-4)) / 2e-4
        step = f / d
        T -= step
        if abs(step) < 1e-10:
            break
    return T


def psat_sonntag(T_C, ice=False):
    """Pa. Sonntag 1990: over water 611.2 exp(17.62 T/(243.12+T)), over ice
    611.2 exp(22.46 T/(272.62+T)). Used by the condensation law (4.3)."""
    if ice:
        return 611.2 * math.exp(22.46 * T_C / (272.62 + T_C))
    return 611.2 * math.exp(17.62 * T_C / (243.12 + T_C))


def dew_point(T_C, rh):
    """C. Inverse of psat_sonntag at rh * psat(T).

    The limit at zero humidity is physical: with no vapour in the air there is no
    temperature at which a surface gains water, so the dew point is minus infinity
    and this returns -math.inf (a surface at any finite temperature is above it, so
    nothing condenses). The log in the formula is not defined there; the limit is
    taken instead of raising. Humidity above 1 is not refused (it is
    supersaturated air, dew point above T), but a negative or NaN rh is a bad input
    and raises ValueError."""
    if not rh >= 0.0:                                  # also catches NaN
        raise ValueError(f"relative humidity must be >= 0, got {rh!r}")
    if rh == 0.0:
        return -math.inf
    e = rh * psat_sonntag(T_C)
    q = e / 611.2
    if q <= 0.0:                      # rh so small the vapour pressure (or its ratio to
        return -math.inf              # the 611.2 Pa reference) underflows: the limit
    x = math.log(q)
    if x >= 17.62:                                     # the formula's pole (4.5e10 Pa): not air
        raise ValueError(f"vapour pressure {e:.3g} Pa is outside the Sonntag range")
    return 243.12 * x / (17.62 - x)


_LIQ_T_ASC = tuple(reversed(LIQUIDUS_T))          # ascending temperatures
_LIQ_WT_ASC = tuple(reversed(LIQUIDUS_WT))


def liquidus_wt(T_C):
    """Weight fraction of NaCl in a brine on the liquidus at T (0 at 0 C, 0.233 at
    the eutectic), from the CRC table (u)."""
    if T_C >= 0.0:
        return 0.0
    t = max(T_C, T_EUTECTIC)
    i = bisect.bisect_left(_LIQ_T_ASC, t)
    if i <= 0:
        return _LIQ_WT_ASC[0] / 100.0
    t0, t1 = _LIQ_T_ASC[i - 1], _LIQ_T_ASC[i]
    w0, w1 = _LIQ_WT_ASC[i - 1], _LIQ_WT_ASC[i]
    return (w0 + (w1 - w0) * (t - t0) / (t1 - t0)) / 100.0


def brine_freeze_point(wt_frac):
    """C. Table lookup by weight fraction of salt, clamped at the eutectic."""
    return float(np.interp(wt_frac * 100.0, LIQUIDUS_WT, LIQUIDUS_T))


def freeze_point(water_g, solutes):
    """Freezing point, C, of water with dissolved solutes {species: grams} (the
    private stand-in for mixwater.freeze_point). NaCl-like species ('salt' tag) go
    by the CRC table; others by Raoult T_f = -K_f*b*i."""
    if water_g <= 0.0:
        return 0.0
    salt = other = 0.0
    for sp, g in solutes.items():
        r = ROWS[sp]
        if "salt" in r["tags"]:
            salt += g
        else:
            other += K_F_WATER * (g / r["M"]) / (water_g / 1000.0) * r["nu"]
    tf = brine_freeze_point(salt / (water_g + salt)) if salt > 0.0 else 0.0
    return tf - other


def raoult_freeze_drop(molality, i):
    return K_F_WATER * molality * i


def boil_point(solutes, water_g, P=101325.0):
    """Boiling point, C, of water with dissolved solutes {species: grams} at P Pa:
    T_b(P) + K_b*b*i (the private stand-in for mixwater.boiling_point)."""
    t0 = tsat_water(P)
    if water_g <= 0.0:
        return t0
    n = sum(g / ROWS[sp]["M"] * ROWS[sp]["nu"] for sp, g in solutes.items())
    return t0 + K_B_WATER * n / (water_g / 1000.0)


def solubility(row, T_C):
    """g solute per g water, or None when the row has no solubility."""
    if row["sol0"] is None:
        return None
    return row["sol0"] + row["sol_slope"] * T_C


def lv_water(T_C):
    """J/g. Lv(T) = 2501 - 2.37 T."""
    return 2501.0 - 2.37 * T_C


def lf_kirchhoff(row, T_C):
    """J/g. Lf(T) = Lf0 + (cp_l - cp_c)*T: water at -10 C is 312, not 333.6."""
    return row["Lf0"] + (row["cp_l"] - row["cp_c"]) * T_C


def lf_at_ambient(row):
    """Lf extrapolated to the 20 C reference: what a crystal gram stores."""
    return row["Lf0"] + (row["cp_l"] - row["cp_c"]) * AMBIENT


# =====================================================================================
# ENTHALPY: what a gram of (species, form) holds relative to liquid at 20 C
#   h(T) = cp_form*(T - 20) + offset_form
# CRYSTAL of a melting species: -Lf(20 C); DISSOLVED solute: +heat of solution;
# BOUND: -(heat released on binding). Sensible heat lives in the host node (it is
# part of its capacity); the offsets are what the layer STORES (F2).
# =====================================================================================
def cp_of(row, form):
    return row["cp_l"] if form == DISSOLVED else row["cp_c"]


def form_offset(row, form):
    """J/g stored in a gram of this form, relative to its liquid at 20 C."""
    if form == CRYSTAL and row["Lf0"] > 0.0:
        return -lf_at_ambient(row)
    if form == DISSOLVED and row["hsol"] != 0.0:
        return row["hsol"]
    if form == BOUND and row["kin"] is not None:
        return -row["kin"]["dH"]
    return 0.0


def enthalpy(row, form, T_C):
    """J/g of a gram of species/form at T relative to liquid at 20 C."""
    return cp_of(row, form) * (T_C - AMBIENT) + form_offset(row, form)


# =====================================================================================
# 4.8 THE JOINT LATENT PIN
#   A cell has ONE skin node (Csur at Ts). Every layer on its faces that can change
#   phase offers eq(T) = the change in its entries at equilibrium temperature T. The
#   node finds the common T' with   Csur*(T' - Ts) + sum dH(T') = 0   where dH is the
#   enthalpy change of the layers' entries (and of the exported vapour). The function
#   is monotone; bisection; a jump (eutectic, boiling point of pure water) is a
#   plateau, resolved by interpolating between the two sides.
# =====================================================================================
class PhaseFace:
    """One layer's phase offer. `keys` are the entries it can change (component
    keys; VAPOUR = -1 is exported); eq(T) -> array of grams change aligned with
    keys; h(T) -> array of J/g, entry enthalpies at T (cp and offset)."""
    VAPOUR = -1

    def __init__(self, keys, eq, h):
        self.keys = list(keys)
        self.eq = eq
        self.h = h

    def dH(self, T):
        d = np.asarray(self.eq(T), dtype=np.float64)
        return float(np.dot(d, self.h(T))), d


def solve_pin(Csur, Ts, faces, iters=70):
    """(T', [dm per face], plateau) of the joint solve. Csur J/K, Ts C.
    dm[i] is faces[i].eq(...) at the solution (interpolated across a jump)."""
    if not faces:
        return Ts, [], False

    def total(T):
        s = 0.0
        ds = []
        for f in faces:
            e, d = f.dH(T)
            s += e
            ds.append(d)
        return Csur * (T - Ts) + s, ds

    g0, d0 = total(Ts)
    if g0 == 0.0 and all(not np.any(d) for d in d0):
        return Ts, d0, False
    if g0 > 0.0:                      # too much heat released: root below Ts
        hi, ghi = Ts, g0
        step = 4.0
        lo = Ts - step
        glo, _ = total(lo)
        while glo > 0.0 and step < 4000.0:
            hi, ghi = lo, glo
            step *= 2.0
            lo = Ts - step
            glo, _ = total(lo)
    else:
        lo, glo = Ts, g0
        step = 4.0
        hi = Ts + step
        ghi, _ = total(hi)
        while ghi < 0.0 and step < 4000.0:
            lo, glo = hi, ghi
            step *= 2.0
            hi = Ts + step
            ghi, _ = total(hi)
    # faces that do not change anywhere in the bracket are inert: leave them out
    zero = [np.zeros(len(f.keys)) for f in faces]
    live = []
    for i, f in enumerate(faces):
        if np.any(f.eq(lo)) or np.any(f.eq(hi)) or np.any(f.eq(Ts)):
            live.append(i)
    if len(live) < len(faces):
        sub = [faces[i] for i in live]
        T_n, dm_sub, pl = solve_pin(Csur, Ts, sub, iters) if sub else (Ts, [], False)
        out = list(zero)
        for i, d in zip(live, dm_sub):
            out[i] = d
        return T_n, out, pl
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        gm, _ = total(mid)
        if gm < 0.0:
            lo, glo = mid, gm
        else:
            hi, ghi = mid, gm
        if hi - lo < 1e-11:
            break
    glo, dlo = total(lo)
    ghi, dhi = total(hi)
    span = ghi - glo
    th = 0.0 if span <= 0.0 else min(1.0, max(0.0, -glo / span))
    Tn = lo + (hi - lo) * th
    dms = [a + th * (b - a) for a, b in zip(dlo, dhi)]
    plateau = span > 1e-6 * max(1.0, abs(Csur))
    return Tn, dms, plateau


def _bounded(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class WaterFace(PhaseFace):
    """Water with its solutes in one layer: freeze, melt, salt lever rule, salt
    dissolution and crystallisation, boiling with elevation. Entries:
    [water DISSOLVED, water CRYSTAL, VAPOUR, then (solute DISSOLVED, solute
    CRYSTAL) for every solute with a solubility]. `water` is (ck_liquid, ck_crystal,
    g_liquid, g_crystal); `solutes` a list of (ck_d, ck_c, g_d, g_c, species)."""

    def __init__(self, water, solutes, Ts, P=101325.0, freeze_cap=None,
                 q_cap_g=None, boil=True):
        ckl, ckc, wl, wc = water
        self.W = wl + wc
        self.wc0 = wc
        self.solutes = solutes
        self.P = P
        self.Tb0 = tsat_water(P)
        self.freeze_cap = freeze_cap          # g per step the Stefan cap allows
        self.q_cap_g = q_cap_g                # g per step a boiling curve allows
        self.boil = boil
        self.row_w = ROWS[ckl >> 10]
        self.salts = [s for s in solutes if "salt" in ROWS[s[4]]["tags"]]
        self.others = [s for s in solutes if "salt" not in ROWS[s[4]]["tags"]]
        self.S_tot = sum(s[2] + s[3] for s in self.salts)
        # n_eff: moles times van't Hoff i of every dissolved solute (for the boil)
        self.n_tot = sum((s[2] + s[3]) / ROWS[s[4]]["M"] * ROWS[s[4]]["nu"]
                         for s in solutes)
        keys = [ckl, ckc, self.VAPOUR]
        for s in solutes:
            keys += [s[0], s[1]]
        super().__init__(keys, self._eq, self._h)
        self._rows = [self.row_w, self.row_w, None] + [
            ROWS[s[4]] for s in solutes for _ in (0, 1)]
        self._forms = [DISSOLVED, CRYSTAL, -1] + [f for s in solutes
                                                  for f in (DISSOLVED, CRYSTAL)]
        self._start = np.array([wl, wc, 0.0] + [x for s in solutes for x in (s[2], s[3])])
        self._cp = np.array([self.row_w["cp_l"] if f < 0 else cp_of(r, f)
                             for r, f in zip(self._rows, self._forms)])
        self._off = np.array([0.0 if f < 0 else form_offset(r, f)
                              for r, f in zip(self._rows, self._forms)])

    # enthalpies of the entries at T: cp (T - 20) + offset, the vapour adds Lv(T)
    def _h(self, T):
        out = self._cp * (T - AMBIENT) + self._off
        out[2] += lv_water(T)
        return out

    def ice_eq(self, T):
        """Equilibrium ice, g, at temperature T (lever rule with salt)."""
        W = self.W
        if W <= 0.0 or T >= 0.0:
            return 0.0
        S = self.S_tot
        if S <= 0.0:
            return W
        if T <= T_EUTECTIC:
            return W
        c_eq = liquidus_wt(T)
        wt0 = S / (W + S)
        if wt0 >= c_eq:
            return 0.0
        return max(0.0, min(W, W - S * (1.0 - c_eq) / c_eq))

    def _eq(self, T):
        W = self.W
        ice = self.ice_eq(T)
        ice = _bounded(ice, 0.0, W)
        if ice > self.wc0 and self.freeze_cap is not None:
            ice = min(ice, self.wc0 + self.freeze_cap)
        vap = 0.0
        if self.boil and T > self.Tb0 and W > 0.0:
            wl0 = W - ice
            if self.n_tot <= 0.0:
                vap = wl0
            else:
                b = (T - self.Tb0) / K_B_WATER                       # mol/kg
                w_rem = 1000.0 * self.n_tot / b if b > 0.0 else 1e30
                # a brine cannot hold more salt than it dissolves: past saturation the
                # film boils dry and the salt crystallises
                sat_w = sum((s[2] + s[3]) / solubility(ROWS[s[4]], T) for s in self.salts)
                if w_rem < sat_w:
                    w_rem = 0.0
                vap = _bounded(wl0 - w_rem, 0.0, wl0)
            if self.q_cap_g is not None:
                vap = min(vap, self.q_cap_g)
        wl = W - ice - vap
        out = [wl - self._start[0], ice - self._start[1], vap]
        for s in self.solutes:
            tot = s[2] + s[3]
            sol = solubility(ROWS[s[4]], T)
            if sol is None:
                d = tot                       # does not crystallise: stays dissolved
            elif wl <= 0.0:
                d = 0.0
            else:
                d = min(tot, sol * wl)
            out += [d - s[2], (tot - d) - s[3]]
        return out


class MeltFace(PhaseFace):
    """A melting species with a RANGE (wax, tallow, tar): the liquid fraction is an
    enthalpy curve between Tsol and Tliq, linear in T. Entries [liquid, crystal]."""

    def __init__(self, species, ck_l, ck_c, g_l, g_c):
        self.row = ROWS[species]
        self.tot = g_l + g_c
        self.g_l0, self.g_c0 = g_l, g_c
        super().__init__([ck_l, ck_c], self._eq, self._h)
        self._cp = np.array([cp_of(self.row, DISSOLVED), cp_of(self.row, CRYSTAL)])
        self._off = np.array([form_offset(self.row, DISSOLVED),
                              form_offset(self.row, CRYSTAL)])

    def liquid_fraction(self, T):
        a, b = self.row["Tsol"], self.row["Tliq"]
        if b is None or b <= a:
            return 1.0 if T >= (a if a is not None else 1e9) else 0.0
        return _bounded((T - a) / (b - a), 0.0, 1.0)

    def _eq(self, T):
        fl = self.liquid_fraction(T)
        return [fl * self.tot - self.g_l0, (1.0 - fl) * self.tot - self.g_c0]

    def _h(self, T):
        return self._cp * (T - AMBIENT) + self._off


def stefan_cap_grams(area_cover, k_cry, t_liq, Ts, Lf, h_cry, dt):
    """Most a film can freeze in dt through the ice already formed (4.8d):
    dm <= A*k*(T_liq - Ts)/(Lf*max(h_cry, 0.1 mm))*dt, g (k W/m/K, Lf J/g)."""
    if Ts >= t_liq:
        return 0.0
    return area_cover * k_cry * (t_liq - Ts) / (Lf * max(h_cry, 1e-4)) * dt


def stefan_time(rho_kgm3, Lf_Jkg, h, k, dT):
    """Seconds for a film of thickness h to freeze at dT: rho Lf h^2/(2 k dT)."""
    return rho_kgm3 * Lf_Jkg * h * h / (2.0 * k * dT)


# =====================================================================================
# 4.9 KINETICS, GROWTH, ADHESION, SPALL
# =====================================================================================
def arrhenius(k_ref, T_ref_C, Ea, T_C):
    """k(T) = k_ref*exp(-(Ea/R)*(1/T_K - 1/T_ref_K))."""
    return k_ref * math.exp(-(Ea / R_GAS) * (1.0 / (T_C + 273.15) - 1.0 / (T_ref_C + 273.15)))


def first_order_step(X, k, dt):
    """X' = 1 - (1 - X) exp(-k dt): exact for any dt at constant k."""
    return 1.0 - (1.0 - X) * math.exp(-k * dt)


def q10(Ea, T_C=10.0):
    """k(T+10)/k(T) from the row's own Ea, taken from 10 to 20 C: 50 kJ/mol gives
    2.06 and 60 kJ/mol 2.39."""
    return math.exp(Ea / R_GAS * (1.0 / (T_C + 273.15) - 1.0 / (T_C + 283.15)))


def cure_k_eff(k, h, h_ref):
    """Oxygen-diffusion limit: a coat thicker than h_ref cures slower."""
    if h_ref is None or h <= h_ref:
        return k
    return k * h_ref / h


# two-regime oxide growth (iron row, fitted to the temper chart, not to growth data (u))
OXIDE = dict(A=1120e-9, B=2282.0, tau=1.0, T_x=400.0, t_join=10.0,
             kp_x=5e-16, Q=100e3)      # parabolic constant at T_x m2/s, activation J/mol


def oxide_xL(T_C, p=OXIDE):
    return p["A"] * math.exp(-p["B"] / (T_C + 273.15))


def oxide_kp(T_C, p=OXIDE):
    """Parabolic constant, m2/s, Arrhenius from its value at the crossover."""
    return p["kp_x"] * math.exp(-p["Q"] / R_GAS * (1.0 / (T_C + 273.15)
                                                   - 1.0 / (p["T_x"] + 273.15)))


def oxide_thickness(T_C, t, p=OXIDE):
    """Film thickness, m, after t seconds at constant T from a clean surface. Below
    the crossover T_x: logarithmic x = x_L ln(1 + t/tau). Above: the log law until
    t_join, then parabolic x^2 = x_c^2 + k_p (t - t_join); continuous at the join."""
    xl = oxide_xL(T_C, p)
    if T_C < p["T_x"] or t <= p["t_join"]:
        return xl * math.log1p(t / p["tau"])
    xc = xl * math.log1p(p["t_join"] / p["tau"])
    return math.sqrt(xc * xc + oxide_kp(T_C, p) * (t - p["t_join"]))


def oxide_step(x, T_C, dt, p=OXIDE):
    """Thickness after dt more seconds at T, from a film of thickness x. The log law
    is stepped by inverting to its equivalent time, so a stepped history equals the
    closed form; above T_x the parabolic law takes over once the film is as thick as
    the log law makes it at t_join. Continuous in x and in T by construction."""
    xl = oxide_xL(T_C, p)
    xj = xl * math.log1p(p["t_join"] / p["tau"])
    if T_C < p["T_x"] or x < xj:
        u = x / xl
        t_eq = p["tau"] * math.expm1(u) if u < 700.0 else 1e300
        xn = xl * math.log1p((t_eq + dt) / p["tau"])
        if T_C >= p["T_x"] and xn > xj:             # crosses the join inside the step
            t_left = dt - max(0.0, p["t_join"] - t_eq)
            return math.sqrt(xj * xj + oxide_kp(T_C, p) * max(t_left, 0.0))
        return xn
    return math.sqrt(x * x + oxide_kp(T_C, p) * dt)


def rust_thickness(r, t, b=0.6):
    """Atmospheric rust x = r*t^b (ISO 9223 shape), b about 0.5-0.8."""
    return r * t ** b


def spall_energy(E_f, eps, h):
    """Stored elastic energy per area, J/m2: G = E_f eps^2 h / 2."""
    return 0.5 * E_f * eps * eps * h


def spalls(E_f, eps, h, W_a):
    return spall_energy(E_f, eps, h) >= W_a


def spall_h_crit(E_f, eps, W_a):
    """Thickness where a film lifts by itself: 2 W_a/(E_f eps^2)."""
    return 2.0 * W_a / (E_f * eps * eps)


def thermal_mismatch_strain(d_alpha, dT):
    return d_alpha * dT


SLOW_TICKS = 40                  # ticks between slow-tier visits
HOT_DT = 15.0                    # K from ambient that promotes a layer to the hot tier
HOT_DX = 0.1                     # share a layer may change in one slow step


def promote_hot(Ts, k, tick_s, above_hold, slow_ticks=SLOW_TICKS, dT=HOT_DT, dX=HOT_DX):
    """Slow-tier promotion rule (5.2): a layer is visited every tick when its host
    skin deviates from AMBIENT by more than dT, when k(T)*slow step would change it
    by more than dX (a flame lick of 0.5 s is seen), or when it is above its hold."""
    return bool(abs(Ts - AMBIENT) > dT or k * tick_s * slow_ticks > dX or above_hold)


def mix_strength(rows, which, phi_l):
    """Layer adhesion or cohesion, Pa, as the geometric mean exp(sum x_i ln a_i) of
    its entries (4.9). rows: list of (mass, species row, form). which: 'a' or 'c'.
    Mobile entries run from the wet value to the dry one by (1 - phi_l); BOUND
    entries use the bound value; CRYSTAL the crystal value when the row has one."""
    tot = sum(m for m, _r, _f in rows)
    if tot <= 0.0:
        return 0.0
    s = 0.0
    for m, r, f in rows:
        if f == BOUND:
            v = r["a_bnd"] if which == "a" else r["c_bnd"]
        elif f == CRYSTAL and r["a_cry" if which == "a" else "c_cry"] is not None:
            v = r["a_cry"] if which == "a" else r["c_cry"]
        else:
            wet = r["a_wet"] if which == "a" else r["c_wet"]
            dry = r["a_dry"] if which == "a" else r["c_dry"]
            v = wet + (dry - wet) * (1.0 - phi_l)
        s += (m / tot) * math.log(max(v, 0.1))
    return math.exp(s)


# =====================================================================================
# 4.10 CONTACT, WIPE, WASH, WEAR
# =====================================================================================
def liquid_split(theta_S, theta_D):
    """Share of a liquid film that goes to D at separation, by the work of adhesion
    (Young-Dupre): f_D = (1+cos thD)/((1+cos thS) + (1+cos thD)). 0.5 for equals."""
    a = 1.0 + math.cos(math.radians(theta_S))
    b = 1.0 + math.cos(math.radians(theta_D))
    return b / (a + b)


def dry_transfer(c_c, a_D, a_S, p, te_max=TE_MAX, p0=P0_TRANSFER):
    """f_st = c_c * TE_max * a_D/(a_D + a_S) * (1 - exp(-p/p0))."""
    return c_c * te_max * a_D / (a_D + a_S) * (1.0 - math.exp(-p / p0))


def slide_fraction(f_st, c_c, s, L_p):
    """f(s) = f_st + (c_c - f_st)*(1 - exp(-s/L_s)), L_s = max(L_p, 0.02 m): the
    static value at s = 0, the whole contacted share as s grows."""
    ls = max(L_p, L_SLIDE_MIN)
    return f_st + (c_c - f_st) * (1.0 - math.exp(-s / ls))


def wipe_fraction(tau, adh, coh, shear_ratio=SHEAR_RATIO):
    """Share of a layer a wipe of shear tau (Pa) lifts (4.10d). adh is a pull-off
    number; shear adhesion = shear_ratio*adh. If adh_s <= coh (ice, scale, paint,
    a skin) nothing lifts under adh_s and the WHOLE layer lifts at or over it; else
    nothing under coh, (tau-coh)/(adh_s-coh) between, all at adh_s."""
    adh_s = shear_ratio * adh
    if adh_s <= coh:
        return 1.0 if tau >= adh_s else 0.0
    if tau < coh:
        return 0.0
    if tau >= adh_s:
        return 1.0
    return (tau - coh) / (adh_s - coh)


def wall_shear_film(h, rho_kgm3=1000.0):
    """Shear on a wall under a falling film, Pa: rho g h."""
    return rho_kgm3 * G * h


def wall_shear_nusselt(eta, q, h):
    """3 eta Q/h^2 (Pa) for a Nusselt film of flow q per m width and thickness h."""
    return 3.0 * eta * q / (h * h)


def archard_dh(K, p, s, H):
    """Removed thickness, m: dh = K p s/H (Archard 1953)."""
    return K * p * s / H


# =====================================================================================
# 4.2 DEPOSITION FROM AIR
# =====================================================================================
MU_AIR = 1.81e-5
RHO_AIR = 1.2
MFP = 0.066e-6                   # mean free path of air, m


def cunningham(d):
    """Slip correction C_c = 1 + Kn*(1.257 + 0.4 exp(-1.1/Kn)), Kn = 2 lambda/d."""
    kn = 2.0 * MFP / d
    return 1.0 + kn * (1.257 + 0.4 * math.exp(-1.1 / kn))


def stokes_velocity(d, rho_p, mu=MU_AIR, rho_a=RHO_AIR):
    """Settling speed, m/s: (rho_p - rho_a) g d^2 C_c/(18 mu); d m, rho kg/m3."""
    return (rho_p - rho_a) * G * d * d * cunningham(d) / (18.0 * mu)


def kinematic_viscosity_air(T_C):
    """nu(T) = 1.5e-5*(T_K/293)^1.7, m2/s."""
    return 1.5e-5 * ((T_C + 273.15) / 293.0) ** 1.7


def thermophoretic_velocity(T_gas, T_wall, K_th=0.5, delta=0.02):
    """v_th = K_th*nu(T_film)*(T_gas - T_wall)/(delta*T_gas_K), m/s toward the colder
    wall; nu at the film temperature (mean of gas and wall). A 500 K gas over a
    300 K wall gives 2.5e-4 m/s."""
    nu = kinematic_viscosity_air(0.5 * (T_gas + T_wall))
    return K_th * nu * (T_gas - T_wall) / (delta * (T_gas + 273.15))


V_BROWN = 1.0e-5                 # Brownian plus turbulent wall deposition, m/s (u)


def deposition_velocity(d, rho_p, orient, T_gas=None, T_wall=None, v_b=V_BROWN):
    """v_d = max(0, v_b + v_th + v_s*cos(orient)); orient +1 up face, 0 wall, -1
    ceiling. Floored at 0: a big particle on a ceiling does not go negative."""
    vth = 0.0
    if T_gas is not None and T_wall is not None:
        vth = thermophoretic_velocity(T_gas, T_wall)
    return max(0.0, v_b + vth + stokes_velocity(d, rho_p) * orient)


def deposit_mass(v_d, conc, area, dt, cap_g=None):
    """Grams that settle on a face in dt: J*A*dt with J = v_d*C (C g/m3); capped
    (the caller passes 0.5 of the cell's particles shared over its exposed faces)."""
    m = v_d * conc * area * dt
    return m if cap_g is None else min(m, cap_g)


def sink_fraction(v_d, area_s, volume, t):
    """Well-mixed sink fraction 1 - exp(-v_d*A_s/V*t) of a room's particles."""
    return 1.0 - math.exp(-v_d * area_s / volume * t)


# =====================================================================================
# 4.3 CONDENSATION
# =====================================================================================
def rho_vapour(e_Pa, T_C):
    """g/m3 of water vapour at partial pressure e and temperature T."""
    return 1000.0 * e_Pa * 0.018015 / (R_GAS * (T_C + 273.15))


def condense_flux(rho_v_air, Ts, h_c, a_w=1.0, T_air=25.0, Le=0.88):
    """g/m2/s onto a face at Ts (positive condenses): h_m*(rho_v,air - a_w*rho_sat(Ts)),
    h_m = h_c/(rho_a c_p Le^(2/3)) (Chilton-Colburn). Over ice below 0 C."""
    rho_a = 101325.0 / (287.05 * (T_air + 273.15))
    h_m = h_c / (rho_a * 1006.0 * Le ** (2.0 / 3.0))
    sat = rho_vapour(psat_sonntag(Ts, ice=Ts < 0.0), Ts)
    return h_m * (rho_v_air - a_w * sat)


def condense_latent(Ts):
    """J/g into the skin: Lv(T), or Lv + Lf(T) below 0 C (frost, Ls = 2834 at 0 C)."""
    if Ts < 0.0:
        return lv_water(Ts) + 333.55 + (4.18 - 2.09) * Ts
    return lv_water(Ts)


def tar_condenses(p_part, T_wall, p_ref, T_ref, L_Jg, M):
    """Creosote: p_sat = p_ref*exp(-(L M/R)(1/T_K - 1/T_ref_K)); it condenses where
    p_sat(T_wall) < the partial pressure."""
    ps = p_ref * math.exp(-(L_Jg * M / R_GAS) * (1.0 / (T_wall + 273.15)
                                                 - 1.0 / (T_ref + 273.15)))
    return ps < p_part


# =====================================================================================
# 4.12 / 4.13 OPTICS COVER AND FRICTION
# =====================================================================================
def friction_mu(mu_dry, mu_lube, h, Ra):
    """mu_eff = mu_dry*(1 - f_w) + mu_lube*f_w with f_w = 1 - exp(-h/Ra)."""
    fw = 1.0 - math.exp(-h / Ra) if Ra > 0.0 else (1.0 if h > 0.0 else 0.0)
    return mu_dry * (1.0 - fw) + mu_lube * fw


def wet_fraction(h, Ra):
    return 1.0 - math.exp(-h / Ra) if Ra > 0.0 else (1.0 if h > 0.0 else 0.0)


def series_resistance(thicknesses_over_k):
    return float(sum(thicknesses_over_k))


def grain_diameter(species, sz):
    """m: a row's fixed size d0 (soot, pigments) else the size class of the key."""
    d0 = ROWS[species]["d0"]
    return d0 if d0 else d_of_sz(sz)
