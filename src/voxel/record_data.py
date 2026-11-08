"""Stand-in DATA for the record pools (T2): ten species, their enthalpy tables, moisture
diffusion and sorption rows, joint kinds, and strength and fracture rows.

PHASE A STAND-ALONE. The records hold DENSE grams per species per node; this module is
the species table they read. At wiring (slice 1) T1's registry replaces `SP_NAMES` and the
rows, through one adapter that maps dense grams to T1 pool row ids per node (one
direction), and T9's material record supplies the strength rows. Until then every number
here carries a reference or is marked (u) = unverified: independent of the code that uses
it, but its source has not been checked (A-S marks it sourced).

Temperatures are C, enthalpy J/kg above 20 C (the lattice convention), densities kg/m3."""
import functools
import math

import numpy as np

from .fasteners import INCH, dowel_Rd  # noqa: F401  (the ONE copy of the NDS Rd: see the end of this file)
from .thinheat import Material, T_REF

SP_NAMES = ("iron", "steel", "wood", "water", "wax", "hide", "glass", "grain", "glue", "salt")
NS = len(SP_NAMES)
SP = {n: i for i, n in enumerate(SP_NAMES)}
FLUID_NAMES = ("water", "oil", "blood", "brine", "milk", "acid")        # ml slots of a vessel
NF = len(FLUID_NAMES)
FORMS = ("liquid", "dissolved", "crystal", "grain", "bound")             # T1 forms, for the adapter

# species -> (rho kg/m3, source)
RHO = np.array([7800.0,    # iron: the sim's table (materials.py)
                7801.0,    # 1% C steel, Incropera A.1
                700.0,     # dry oak (u; Wood Handbook 0.6-0.75)
                998.0,     # water 20 C
                900.0,     # paraffin (u)
                950.0,     # dry hide / leather (u)
                2500.0,    # soda-lime glass, Incropera A.1
                1350.0,    # wheat kernel true density (u)
                1270.0,    # dry hide glue (u)
                2165.0])   # NaCl, CRC
# swelling factor on the volume a species contributes (1 = none). Water in a hide swells it
# by about the volume of the water (u).
SWELL = np.ones(NS)


# ── enthalpy tables ──────────────────────────────────────────────────────────
def _bump(T, T0, amp, width):
    return amp * np.exp(-0.5 * ((T - T0) / width) ** 2)


def build_material(name, rho, cp, k, latents=(), T_lo=-80.0, T_hi=1700.0, dT=5.0, extra=()):
    """Tabulate H(T) from a cp(T) function and latent changes ((Tm, L J/kg, width K), ...)
    into a Material. Each latent is one segment of width `width` whose slope carries L, so
    T(H) is a continuous plateau of height L at Tm. H(20 C) = 0."""
    knots = list(np.arange(T_lo, T_hi + dT, dT)) + list(extra)
    for Tm, _, w in latents:
        knots += [Tm - 0.5 * w, Tm + 0.5 * w]
    Tk = np.unique(np.round(np.array(knots, dtype=np.float64), 9))
    c = cp(Tk)
    dH = 0.5 * (c[:-1] + c[1:]) * np.diff(Tk)
    for Tm, L, w in latents:
        i = int(np.argmin(np.abs(Tk - (Tm - 0.5 * w))))
        dH[i] += L
    Hk = np.concatenate([[0.0], np.cumsum(dH)])
    Hk -= np.interp(T_REF, Tk, Hk)
    return Material(name, rho, Tk, Hk, k)


_WATER_SMEAR = (0.03,)     # K, the width of the melting plateau (the latent below)


def _steel_cp(T):   # Incropera A.1: 473 at 300 K; rises toward the magnetic change; bump 900 J/kg K, 25 K wide at 770 C (u)
    return 460.0 + 0.30 * (T - 20.0) + _bump(T, 770.0, 900.0, 25.0)


def _steel_k(T):    # 43 at 20 C falling to about 27 at 800 C (u above 300 K)
    return np.maximum(43.3 - 0.021 * (T - 20.0), 25.0)


def _iron_cp(T):    # the sim's table says 450 (u beyond)
    return 450.0 + 0.30 * (T - 20.0)


def _water_cp(T):   # liquid 4182 at 20 C; ice 2100 below 0 C; steam 2010 above 100 C (CRC)
    return np.where(T < 0.0, 2100.0, np.where(T > 100.0, 2010.0, 4182.0))


def _water_k(T):    # liquid 0.598; ice 2.2 (CRC); mixed by the ICE FRACTION across the melting plateau
    # The plateau is a range of H at one T. A sharp step of k at 0 C inside it makes the
    # conductance of a node that sits on the plateau flip with the sign of a thousandth of a
    # kelvin, and no lagged-conductance pass can settle that (the node ends 6 K away from the
    # state it was solved for). The ice fraction is LINEAR in T across the smeared plateau
    # (T is linear in H there), so k is too: a series/parallel mix would also do, the ramp is
    # the simple one (u: the real mush k depends on the ice texture).
    f = np.clip((T - (0.0 - 0.5 * _WATER_SMEAR[0])) / _WATER_SMEAR[0], 0.0, 1.0)   # 0 ice .. 1 liquid
    return 2.2 + (0.598 - 2.2) * f


def _wax_cp(T):
    return np.where(T < 55.0, 2100.0, 2200.0)           # paraffin (u)


def _specs():
    return {
        "iron": dict(cp=_iron_cp, k=lambda T: np.full_like(T, 55.0)),       # materials.py: 55
        "steel": dict(cp=_steel_cp, k=_steel_k),
        # Wood Handbook eq. 4-13: cp = 0.1031 + 0.003867 T_K kJ/kg K (dry wood)
        "wood": dict(cp=lambda T: 1000.0 * (0.1031 + 0.003867 * (T + 273.15)),
                     k=lambda T: np.full_like(T, 0.15)),
        "water": dict(cp=_water_cp, k=_water_k, latents=((0.0, 333.55e3, _WATER_SMEAR[0]),      # Lf 333.55 J/g at 0 C
                                                      (100.0, 2256.4e3, 0.05))),   # Lv 2256.4 J/g at 100 C, 1 atm (CRC)
        "wax": dict(cp=_wax_cp, k=lambda T: np.full_like(T, 0.25),
                    latents=((55.0, 210e3, 0.05),)),                                   # paraffin Tm, Lf (u)
        "hide": dict(cp=lambda T: np.full_like(T, 1700.0), k=lambda T: np.full_like(T, 0.15)),   # (u)
        "glass": dict(cp=lambda T: np.full_like(T, 840.0), k=lambda T: np.full_like(T, 1.0)),    # Incropera A.3
        "grain": dict(cp=lambda T: np.full_like(T, 1800.0), k=lambda T: np.full_like(T, 0.15)),  # (u)
        "glue": dict(cp=lambda T: np.full_like(T, 1700.0), k=lambda T: np.full_like(T, 0.2)),    # (u)
        "salt": dict(cp=lambda T: np.full_like(T, 880.0), k=lambda T: np.full_like(T, 6.5)),     # CRC
    }


MATERIALS = {}
for _n, _s in _specs().items():
    MATERIALS[_n] = build_material(_n, RHO[SP[_n]], _s["cp"], _s["k"], _s.get("latents", ()))
del _n, _s

_MIX = {}
_MIX_MAX = 256


def mix_material(w):
    """Material of a fixed MASS-fraction vector w (length NS): H(T) = sum w_s H_s(T) on the
    union of the species' knots (exact for piecewise-linear tables), k by volume fraction
    (parallel, an upper bound), rho = 1/sum(w/rho). Cached by w rounded to 3 places."""
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    key = tuple(np.round(w, 3))
    got = _MIX.get(key)
    if got is not None:
        return got
    w = np.array(key) / sum(key)
    live = [i for i in range(NS) if w[i] > 0.0]
    if len(live) == 1:
        mat = MATERIALS[SP_NAMES[live[0]]]
    else:
        Tk = np.unique(np.concatenate([MATERIALS[SP_NAMES[i]].Tk for i in live]))
        Hk = sum(w[i] * MATERIALS[SP_NAMES[i]].H_of_T(Tk) for i in live)
        rho = 1.0 / sum(w[i] / RHO[i] for i in live)
        phi = np.array([w[i] / RHO[i] for i in live]) * rho
        mats = [MATERIALS[SP_NAMES[i]] for i in live]

        def kmix(T, _m=mats, _p=phi):
            return sum(p * m.k(T) for p, m in zip(_p, _m))
        mat = Material("mix", rho, Tk, Hk, kmix)
    if len(_MIX) >= _MIX_MAX:
        _MIX.pop(next(iter(_MIX)))
    _MIX[key] = mat
    return mat


# THE LATTICE'S OWN HEAT CAPACITY (wave 13c, C5). The lattice holds one constant c per solid
# (materials._CSOLID_ARR). A record that rides on a lattice thing and has to agree with its
# cells in joules must use the same c, or reconcile would move the whole difference every
# tick (iron: 450 J/kg K flat against the record table's 450 + 0.3 (T-20)). A world asks for
# it with `recs.lattice_cp = True`; the tables above are kept for records that own matter.
_LATTICE_SPECIES = {"iron": "IRON", "wood": "WOOD", "glass": "GLASS"}


@functools.lru_cache(maxsize=256)
def _lattice_material(key):
    from . import materials as _M
    w = np.array(key) / sum(key)
    c = 0.0
    for i in range(NS):
        if w[i] <= 0.0:
            continue
        lat = _LATTICE_SPECIES.get(SP_NAMES[i])
        ci = 1000.0 * float(_M._CSOLID_ARR[getattr(_M, lat)]) if lat else float(
            MATERIALS[SP_NAMES[i]].c_app(np.array([0.0]))[0])
        c += w[i] * ci
    ref = mix_material(w)
    return Material.constant("lattice mix", ref.rho, c, float(ref.k(np.array([20.0]))[0]))


def lattice_material(w):
    """Constant-c Material of a mass-fraction vector, with the lattice's c for the
    species the lattice knows (same rounding as mix_material, so the cache is shared)."""
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    return _lattice_material(tuple(np.round(w, 3)))


def row_material(g):
    """Material for a row's grams per species (the mean composition of its nodes)."""
    return mix_material(np.asarray(g, dtype=np.float64))


# ── moisture: diffusion and sorption rows ────────────────────────────────────
# Dry-basis diffusivity D(X, T) = D0 exp(a X) exp(-(Ea/R)(1/T - 1/Tref)), T in K, Tref 293.15.
# Moisture changes D by 10-100x (spec 5.2). All (u): order of magnitude from drying
# literature (wood 1e-10..1e-9 bound water; leather and grain 1e-11..1e-10 m2/s).
DIFF = {
    "wood": (3e-10, 6.0, 4000.0),
    "hide": (5e-11, 4.0, 3500.0),
    "grain": (2e-11, 5.0, 3500.0),
    "glue": (1e-11, 3.0, 3500.0),
}
# GAB sorption isotherm X = Xm C K a / ((1 - K a)(1 - K a + C K a)), a = RH (0..1), X kg/kg
# dry basis. Typical shapes (u): wood Xm .06, hide .10, grain .08, glue .09.
SORPTION = {
    "wood": (0.06, 10.0, 0.80),
    "hide": (0.10, 8.0, 0.78),
    "grain": (0.08, 9.0, 0.75),
    "glue": (0.09, 8.0, 0.75),
}


def diffusivity(X, T, matrix):
    """D (m2/s) of the mobile species in `matrix` (a species name) at content X, T in C."""
    D0, a, Ea = DIFF[matrix]
    return D0 * np.exp(a * X) * np.exp(-Ea * (1.0 / (T + 273.15) - 1.0 / 293.15))


def eq_moisture(rh, matrix):
    """Equilibrium dry-basis moisture at relative humidity rh (the sorption boundary)."""
    Xm, C, K = SORPTION[matrix]
    a = np.clip(np.asarray(rh, dtype=np.float64), 0.0, 0.99)
    return Xm * C * K * a / ((1.0 - K * a) * (1.0 - K * a + C * K * a))


def node_volumes(Y):
    """m3 per node from grams per species Y[..., NS]: sum Y/1000/rho x swelling."""
    return (np.asarray(Y) / 1000.0 / RHO * SWELL).sum(axis=-1)


# ── strength and fracture stand-ins: THE ONE TABLE (T9 owns the real rows) ───
# E Pa, nu, sigma_y Pa (None = breaks before it yields), sigma_u Pa, eps_f (strain to
# fracture, T9 `ductility`), K_Ic Pa sqrt(m) (None = no flaw arm), Weibull m (None = no scatter),
# rho kg/m3, and optionally a0 m (the flaw the material carries; default breaking.A0).
# breaking.STRENGTH is BUILT from this table, so a number is typed once. Where T9's matrec
# states the same material, this row equals it (the lane X cross test compares them).
# Numbers: wood, glass, iron and rope follow matrec (T9) and the bond table
# (src/bonds/materials.py WOOD_L, WOOD_T); the Wood Handbook (FPL-GTR-282, ch. 5) tension
# along the grain 58-112 MPa for green clear wood (A-S wood.tension_parallel). Hard steel:
# spec 6.1 gives K_Ic 20 MPa sqrt(m); sigma_y 1.8 GPa and sigma_u 2.2 GPa are (u) (a blade at
# about HRC 58-60; no sourced entry in sourced_numbers.json). eps_f: the spec's prose says
# "0.5 percent", but a row with sigma_y 1.8 GPa and E 210 GPa yields at 0.86 percent, so it
# cannot fracture at 0.5 percent without first yielding, and the spec's own worked example
# (flaw arm at 0.68 percent) needs eps_f above 0.68 percent: eps_f = 0.02 (u; tempered tool
# steel at this hardness shows 1-3 percent total elongation). test_records_core keeps
# eps_f >= sigma_y/E for every row with a yield. Mild
# steel: spec 6.1 (eps_f 25 percent, sigma_y 250 MPa, K_Ic over 100). Everything else (u).
# The wood rows' K_Ic is the EFFECTIVE value that puts the flaw stress at a0 on the tensile
# strength (an echo): real wood fails at knots and grain, not at a 50 um crack, so wood's a0
# is the one that goes with T9's K_Ic 0.4 MPa sqrt(m): a0 = (K_Ic/(1.12 sigma_u))^2/pi.
STRENGTH = {
    "steel": dict(E=210e9, nu=0.30, sigma_y=250e6, sigma_u=400e6, eps_f=0.25, K_Ic=100e6, weibull_m=30.0, rho=7850.0),
    "steel_hard": dict(E=210e9, nu=0.30, sigma_y=1.8e9, sigma_u=2.2e9, eps_f=0.02, K_Ic=20e6, weibull_m=20.0, rho=7800.0),
    "iron": dict(E=200e9, nu=0.29, sigma_y=250e6, sigma_u=350e6, eps_f=0.20, K_Ic=50e6, weibull_m=40.0, rho=7800.0),   # = matrec iron (u beyond sigma_y)
    "glass": dict(E=70e9, nu=0.22, sigma_y=None, sigma_u=50e6, eps_f=0.0, K_Ic=0.75e6, weibull_m=7.0, rho=2500.0),     # = matrec glass; soda-lime m 5-15 (u)
    "wood": dict(E=11e9, nu=0.35, sigma_y=None, sigma_u=60e6, eps_f=0.01, K_Ic=0.4e6, weibull_m=10.0, rho=600.0, a0=1.128e-5),  # = matrec wood = bond table WOOD_L
    "wood_perp": dict(E=0.55e9, nu=0.35, sigma_y=None, sigma_u=3e6, eps_f=0.01, K_Ic=0.042e6, weibull_m=10.0, rho=600.0),      # = bond table WOOD_T
    "ash": dict(E=12e9, nu=0.35, sigma_y=None, sigma_u=100e6, eps_f=0.01, K_Ic=1.4e6, weibull_m=8.0, rho=690.0),         # (u) a hard wood along the grain
    "earthenware": dict(E=30e9, nu=0.25, sigma_y=None, sigma_u=15e6, eps_f=0.0, K_Ic=0.5e6, weibull_m=8.0, rho=2000.0), # (u)
    "hide": dict(E=0.2e9, nu=0.45, sigma_y=15e6, sigma_u=25e6, eps_f=0.4, K_Ic=5e6, weibull_m=8.0, rho=950.0),          # (u)
    "wax": dict(E=0.2e9, nu=0.4, sigma_y=3e6, sigma_u=4e6, eps_f=0.05, K_Ic=0.1e6, weibull_m=6.0, rho=900.0),           # (u)
    # a rope is a bundle of fibres: no flaw arm; T9 gives m 20, but a bundle's scatter is narrow and
    # the break test here is the tension at sigma_u (u)
    "rope": dict(E=0.94e9, nu=0.40, sigma_y=None, sigma_u=94e6, eps_f=0.15, K_Ic=None, weibull_m=None, rho=885.0),
}


def g_c(row):
    """Fracture energy G_c = K_Ic^2 (1 - nu^2)/E (J/m2), plane strain."""
    return row["K_Ic"] ** 2 * (1.0 - row["nu"] ** 2) / row["E"]


# ── substrates and joint kinds (spec 9.2; T2 joinery READS them from here) ────
# Each mode of a joint is capped by the MATCHING strength of the weaker substrate. The tension
# figure is NOT typed again: it is the sigma_u of the STRENGTH row named in `row` (one table, so a
# glass or hide number cannot differ between breaking and joinery). Shear (tau) and compression
# (sig_c) have no row in STRENGTH and are typed here once.
# wood: shear parallel to the grain, Douglas-fir 12 % MC about 7.8 MPa (Wood Handbook); tension
# PERPENDICULAR to the grain is the wood_perp row (= src/bonds/materials.py WOOD_T, 3.0 MPa; the
# Wood Handbook range is 2.1-5.5, Douglas-fir 2.3). Others (u): order of magnitude.
_SUBSTRATE_ROWS = {
    "wood": dict(row="wood_perp", tau=7.8e6, sig_c=3.0e7),
    "steel": dict(row="steel", tau=1.5e8, sig_c=2.5e8),
    "glass": dict(row="glass", tau=2.0e7, sig_c=1.0e9),
    "stone": dict(row=None, tau=5.0e6, sig_c=8.0e7, sig_t=6.0e6),         # no strength row: typed here
    "leather": dict(row="hide", tau=5.0e6, sig_c=1.0e6),
}
SUBSTRATE = {n: {"tau": r["tau"],
                 "sig_t": STRENGTH[r["row"]]["sigma_u"] if r["row"] else r["sig_t"],
                 "sig_c": r["sig_c"]}
             for n, r in _SUBSTRATE_ROWS.items()}

# ── the kinds (data) ──────────────────────────────────────────────────────────
# "cls" picks the capacity rule, "cure" the cure law and its numbers. Every
# number is (u) unless it carries a source.
JOIN_KINDS = {
    # hide glue / PVA on wood 8-14 MPa lap shear (u); epoxy on steel 10-25 MPa (u)
    "hide_glue": {"cls": "adhesive", "tau_a": 11e6, "t_ratio": 0.6, "lap0": 0.05,
                  "cure": {"law": "set_on_cool", "T_hi": 303.15, "T_lo": 298.15}},
    "pva": {"cls": "adhesive", "tau_a": 11e6, "t_ratio": 0.6, "lap0": 0.05,
            "cure": {"law": "dry"}},
    "epoxy": {"cls": "adhesive", "tau_a": 17.5e6, "t_ratio": 0.6, "lap0": 0.03,
              "cure": {"law": "maturity_exp", "tau_ref": 28800.0, "Ea_R": 6000.0}},
    # Wood Handbook withdrawal P = 54.12 G^2.5 D per mm of penetration (D in mm)
    "nail": {"cls": "nail", "cure": {"law": "none"}, "lateral_ult": 2.0},
    "screw": {"cls": "screw", "cure": {"law": "none"}, "lateral_ult": 2.0},
    # capstan T_load/T_hold = exp(mu phi); knot efficiency is a field of the kind
    "lash_overhand": {"cls": "lashing", "knot_eff": 0.50, "cure": {"law": "none"}},
    "lash_square": {"cls": "lashing", "knot_eff": 0.50, "cure": {"law": "none"}},
    "lash_bowline": {"cls": "lashing", "knot_eff": 0.675, "cure": {"law": "none"}},
    "stitch": {"cls": "sewn", "eff": 0.8, "cure": {"law": "none"}},
    # lime tensile bond 0.1-0.5, compressive 1-5 MPa; Portland over 10 (u)
    "mortar_lime": {"cls": "mortar", "sig_t": 0.3e6, "sig_c": 3.0e6, "tau": 0.5e6,
                    "cure": {"law": "carbonation", "K": 3.0e-3 / math.sqrt(3.15576e7),
                             "lo": 0.4, "hi": 0.8, "taper": 0.1}},
    "mortar_portland": {"cls": "mortar", "sig_t": 1.0e6, "sig_c": 2.0e7, "tau": 2.0e6,
                        "cure": {"law": "hydration", "Ea_R": 4500.0,
                                 "rh_lo": 0.80, "rh_hi": 0.90}},
    # weld: a rate with a window; the upper limit is the solidus (L14)
    "weld_steel_1c": {"cls": "weld", "eff": 1.0,
                      "cure": {"law": "weld", "T_bond": 1323.15, "T_sol": 1420.15,
                               "tau_ref": 10.0, "T_tref": 1423.15, "Q_R": 33700.0,
                               "oxide_max": 5e-6}},
    "weld_wrought": {"cls": "weld", "eff": 1.0,
                     "cure": {"law": "weld", "T_bond": 1373.15, "T_sol": 1773.15,
                              "tau_ref": 10.0, "T_tref": 1473.15, "Q_R": 33700.0,
                              "oxide_max": 5e-6}},
    "clamp": {"cls": "clamp", "cure": {"law": "none"}},
}
JOIN_KIND_NAMES = tuple(JOIN_KINDS)


# ── dowel and screw yield: the reduction term Rd (ONE copy, in fasteners.py) ──
# NDS-2018 Table 12.3.1B, read from the AWC free view-only NDS on 2026-10-30 (the numbers, the
# source and the K_theta factor are at the top of fasteners.py). `dowel_Rd` and `INCH` are
# imported at the top of this file and kept here so joinery and the tests keep their old import
# path: it is the same function object as fasteners.dowel_Rd, so the two modules cannot
# disagree (decision N6).
