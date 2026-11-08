"""Clothing as records with real insulation (wave 13, T7). Pure functions.

A garment is a plain dict (deep-copy and snapshot safe):
    name, fibre, layer (0 = against the skin, higher = outer), thickness_m,
    void (void fraction of the bulk), dry_mass_g_m2, covers {zone: coverage 0..1},
    area_m2 {zone: area of the zone it covers}, water_g (state).

Insulation: R = t / k_eff in m2K/W (1 clo = 0.155). Layers add in series, with
an air gap between them; covered fractions act in PARALLEL with bare skin. A
soaked garment collapses by fibre: k_eff(s) = k_dry + beta s phi (k_water -
k_dry), s the wetness (water held over the soaked capacity, which is g of water per g of dry
fabric, SOAKED_G_PER_G, not the pore volume). Wind is a correction to the TOTAL insulation
(garment plus boundary air), applied once (ISO 9920 / Havenith).

The ensemble gives ONE series resistance per zone for the lattice's H3 hook:
the skin node meets the air through 1/G = 1/(h A f_cl) + R_cl/A, where h is the
combined convective + linearised radiative coefficient. Marks: (u) unverified.
"""
import math

import numpy as np

from .bodyair import latent_heat

CLO = 0.155                              # m2K/W
K_DRY = 0.042                            # bulk fabric still-air conductivity, W/(m K) (u 0.040-0.045)
K_WATER = 0.6
AIR_GAP_R = 0.10 * CLO                   # air trapped between layers, m2K/W (u 0.1-0.2 clo)
H_STATIC = 8.3                           # combined h of a still-air surface, W/m2K (4.5 conv + 3.8 rad)
V_FREE = 0.15                            # wind below this does not pump the clothing, m/s
D_MAX = 0.281 / (2.0 * 0.044)            # the correction is a parabola in (v - 0.15): past its minimum
                                         # (3.19 m/s above V_FREE) it is held, wind cannot add insulation
RHO_WATER_G_M3 = 1.0e6
# fibre: beta (how much of the wet pore volume bridges the insulation) (u), fibre density g/m3 (u)
K_FIBRE = {"cotton": 0.40, "wool": 0.05, "synthetic": 0.15, "leather": 0.30}
# Water a fabric holds, per gram of DRY fabric (the cause of the old bug: the capacity was the
# whole pore volume, void x thickness x area, 5859 g per m2 for a 1 clo shirt, 23 times what a
# soaked shirt carries; pores of a fabric are not all filled by water that wicks in).
#   REGAIN_G_PER_G: moisture regain at 20 C and 65 percent RH, the water the fibre itself takes
#     from air: cotton 8.5 %, wool 16 %, polyester 0.4 % (ASTM D1909 / textile handbooks, as
#     tabulated in Morton and Hearle, Physical Properties of Textile Fibres; leather 12 % (u)).
#   SOAKED_G_PER_G: water held by a fabric soaked and left to drip (wicking capacity).
#     Centrifuged retention is 0.40-0.50 g/g for cotton and wool (DIN 53814 water retention
#     value; US patent 4,371,485 quotes 40 to 50 percent); a soaked and dripping garment holds
#     more than that, so the dripping values are (u): cotton 1.5, wool 2.0 (wool crimp and
#     swelling hold more), synthetic 0.8 (little fibre uptake, mostly between yarns), leather 1.0.
REGAIN_G_PER_G = {"cotton": 0.085, "wool": 0.16, "synthetic": 0.004, "leather": 0.12}
SOAKED_G_PER_G = {"cotton": 1.5, "wool": 2.0, "synthetic": 0.8, "leather": 1.0}
WICK_TAU_S = {"cotton": 10.0, "wool": 25.0, "synthetic": 15.0, "leather": 30.0}   # (u 5-30 s)


def garment(name, fibre, thickness_m, covers, area_m2=None, layer=0, void=0.9,
            dry_mass_g_m2=250.0, water_g=0.0):
    """Build a garment record. `covers` {zone: coverage}; `area_m2` {zone: m2 of
    that zone it could cover} (default 1 m2 per zone, so the water capacity
    is per square metre)."""
    if fibre not in K_FIBRE:
        raise ValueError(f"unknown fibre {fibre!r}")
    zs = sorted(covers)
    return {"name": name, "fibre": fibre, "layer": int(layer), "thickness_m": float(thickness_m),
            "void": float(void), "dry_mass_g_m2": float(dry_mass_g_m2),
            "covers": {z: float(covers[z]) for z in zs},
            "area_m2": {z: float((area_m2 or {}).get(z, 1.0)) for z in zs},
            "water_g": float(water_g)}


def thickness_for_clo(clo):
    """Bulk thickness that gives `clo` of dry insulation, m (6.5 mm per clo)."""
    return clo * CLO * K_DRY


def water_capacity_g(g):
    """Grams of water the garment holds when soaked: SOAKED_G_PER_G of its dry mass, never more
    than its pore volume (void x thickness x area) can contain."""
    area = math.fsum(g["covers"][z] * g["area_m2"][z] for z in sorted(g["covers"]))
    pores = g["void"] * g["thickness_m"] * area * RHO_WATER_G_M3
    return min(SOAKED_G_PER_G[g["fibre"]] * g["dry_mass_g_m2"] * area, pores)


def regain_g(g):
    """Water the dry garment already holds in a 20 C, 65 percent RH room, g."""
    area = math.fsum(g["covers"][z] * g["area_m2"][z] for z in sorted(g["covers"]))
    return REGAIN_G_PER_G[g["fibre"]] * g["dry_mass_g_m2"] * area


def saturation(g):
    """Wetness s = water held / soaked capacity, 0..1 (1 = soaked and dripping; the beta of K_FIBRE
    is the conductivity bridge of that state)."""
    cap = water_capacity_g(g)
    return 0.0 if cap <= 0.0 else min(max(g["water_g"] / cap, 0.0), 1.0)


def k_eff(g, s=None):
    """Conductivity of the garment's bulk at saturation s (default its own)."""
    s = saturation(g) if s is None else s
    beta = K_FIBRE[g["fibre"]]
    return K_DRY + beta * s * g["void"] * (K_WATER - K_DRY)


def resistance(g, s=None):
    """Thermal resistance of the garment, m2K/W."""
    return g["thickness_m"] / k_eff(g, s)


def clo(g, s=None):
    """Insulation of the garment alone in clo; dry cotton shirt about 0.25."""
    return resistance(g, s) / CLO


def mass_g(g):
    """Wet mass of the garment: dry fabric plus the water it holds."""
    area = math.fsum(g["covers"][z] * g["area_m2"][z] for z in sorted(g["covers"]))
    return g["dry_mass_g_m2"] * area + g["water_g"]


def wind_factor(v):
    """I_total(v) / I_total(still): exp(-0.281 (v - 0.15) + 0.044 (v - 0.15)^2) for
    v > 0.15 m/s (ISO 9920 / Havenith), else 1. A correction to TOTAL insulation."""
    d = min(max(float(v) - V_FREE, 0.0), D_MAX)
    return math.exp(-0.281 * d + 0.044 * d * d)


def series_resistance(layers):
    """Resistances in series: the sum."""
    return math.fsum(layers)


def h3_conductance(h, area_m2, f_cl, R):
    """The H3 hook: the skin node meets the air through
    G = 1 / (1/(h A f_cl) + R/A), W/K."""
    return 1.0 / (1.0 / (h * area_m2 * f_cl) + R / area_m2)


def _band_resistances(clothes, zone, wet):
    """Split the zone into bands by garment coverage (garments nested: each covers
    the fraction [0, c] of the zone). Returns [(width, R_garments)] with a bare
    band as R = 0. Layers in a band add in series, with an air gap between."""
    here = [g for g in clothes if g["covers"].get(zone, 0.0) > 0.0]
    here.sort(key=lambda g: (g["layer"], g["name"]))
    if not here:
        return [(1.0, 0.0)]
    cov = sorted({min(g["covers"][zone], 1.0) for g in here})
    edges = [0.0] + cov + ([1.0] if cov[-1] < 1.0 else [])
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # garments present over (lo, hi): those covering at least hi
        on = [g for g in here if min(g["covers"][zone], 1.0) >= hi - 1e-12]
        if on:
            R = math.fsum(resistance(g, None if wet is None else wet.get(g["name"])) for g in on)
            R += AIR_GAP_R * (len(on) - 1)
        else:
            R = 0.0
        out.append((hi - lo, R))
    return out


def _fcl(R):
    """ISO 7730 f_cl for a scalar (the same formula as bodyair.f_cl_iso, without the array overhead)."""
    return 1.0 + 1.29 * R if R <= 0.078 else 1.05 + 0.645 * R


def total_insulation(R_cl, h=H_STATIC):
    """I_total = R_cl + 1/(f_cl h), m2K/W: garment plus boundary air, still."""
    return R_cl + 1.0 / (_fcl(R_cl) * h)


def _solve_series(I_tot, h):
    """The R_cl with R + 1/(f_cl(R) h) = I_tot, in closed form. With f = a + b R
    the equation is a quadratic: b h R^2 + (a - b I_tot) h R - (a h I_tot - 1) = 0."""
    for a, b, lo, hi in ((1.0, 1.29, 0.0, 0.078), (1.05, 0.645, 0.078, math.inf)):
        A = b * h
        Bq = (a - b * I_tot) * h
        Cq = -(a * h * I_tot - 1.0)
        R = (-Bq + math.sqrt(Bq * Bq - 4.0 * A * Cq)) / (2.0 * A)
        if lo <= R <= hi:
            return R
    return 0.078                         # ISO's two branches differ by 3e-4 at the join


def ensemble_resistance(clothes, zone, v=0.0, wet=None, h=H_STATIC):
    """The series resistance R_cl (m2K/W) for the H3 hook of one zone.

    Each band of the zone has I_total = R_band + 1/(f_cl h); the wind
    correction multiplies I_total ONCE; bands combine in parallel (bare skin
    is a band with R = 0). The result is the R_cl that, put in
    G = 1/(1/(h A f_cl(R)) + R/A) with the SAME h, gives that combined
    conductance, so the lattice needs only R_cl and f_cl(R_cl). `wet` is
    {garment name: saturation} to override a garment's own state."""
    bands = _band_resistances(clothes, zone, wet)
    wf = wind_factor(v)
    U = math.fsum(w / (total_insulation(R, h) * wf) for w, R in bands)
    I_tot = 1.0 / U
    if I_tot <= total_insulation(0.0, h):    # bare, or wind has stripped the garment's worth: no series R
        return 0.0
    return _solve_series(I_tot, h)


def ensemble_total(clothes, zone, v=0.0, wet=None, h=H_STATIC):
    """I_total of the zone (m2K/W) with the wind applied once."""
    return total_insulation(ensemble_resistance(clothes, zone, v, wet, h), h)


# ── wetting and drying (mass-exact) ──────────────────────────────────────────
def wet_step(g, available_g, dt, contact=True):
    """Water the garment takes from `available_g` (rain, a puddle, sweat on
    the skin under it) in `dt` s. With contact the pores fill toward capacity
    with the fibre's wicking time; without, nothing moves. Returns
    (garment, taken_g): the garment is a new dict; the caller removes
    `taken_g` from the source."""
    cap = water_capacity_g(g)
    room = max(cap - g["water_g"], 0.0)
    if not contact or room <= 0.0 or available_g <= 0.0:
        return g, 0.0
    frac = 1.0 - math.exp(-dt / WICK_TAU_S[g["fibre"]])
    take = min(room * frac, available_g)
    g2 = dict(g)
    g2["water_g"] = g["water_g"] + take
    return g2, take


def dry_step(g, e_max_w, T, dt):
    """Evaporation from the garment surface with the same E_max form as skin.
    Returns (garment, evap_g, latent_J). Condensation (e_max < 0) is refused
    here (a cold garment in humid air is the caller's mixing problem)."""
    if e_max_w <= 0.0 or g["water_g"] <= 0.0:
        return g, 0.0, 0.0
    L = float(latent_heat(T))
    ev = min(g["water_g"], e_max_w * dt / L * 1000.0)
    g2 = dict(g)
    g2["water_g"] = g["water_g"] - ev
    return g2, ev, ev * L / 1000.0
