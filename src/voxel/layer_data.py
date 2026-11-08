"""SURFACE-LAYER DATA (wave 13, T3): forms, keys, species rows, substrate rows,
the NaCl liquidus table, the label table. numpy only; no World, no heat, no sim.

A layer is a small mix of species, each in one of T1's four FORMS, on a face.
Every number a law needs is a COLUMN of a row here, so the same code reads every
row (F10: thresholds are data). Every (u) is "from memory, no source opened";
tests/data/sourced_numbers.json (lane A-S) promotes numbers one at a time.

Units (one place): rho g/cm3, cp J/(g K), k W/(m K), Lf Lv hsol J/g, pressures
Pa, lengths m (um where the name says so), angles degrees, sigma N/m, eta mPa s.

IDS. T1 owns species ids. Where `species_seed` exists its ids are used (salt is
sodium_chloride 18, clay is kaolinite 23, quartz 21, calcite 22, iron_oxide 31,
tallow 30); a species T1 has no row for yet (soot, plasma, red cells, linseed,
beeswax, tar, inks, ochre, gum) takes a provisional id 200+. ALL of that lives
in ONE dict, ID, built below, so the swap when T1 adds a row is one edit.
"""
import numpy as np

# ---- T1's forms and key (T1 species.py is the owner; a cross test compares) ----
DISSOLVED, CRYSTAL, GRAIN, BOUND = 0, 1, 2, 3
FORM_NAMES = ("dissolved", "crystal", "grain", "bound")
KEY_FORM_SHIFT, KEY_SPECIES_SHIFT = 8, 10
SZ_D0_M, SZ_PER_OCTAVE = 1e-6, 8

AMBIENT = 20.0                 # C, the reference of every sensible energy
G = 9.81
SIGMA_SB = 5.670374419e-8
R_GAS = 8.314462618
FILM = 0.01                    # heat.FILM: a cell 1% full of liquid is wet
DEPTH_CAP = 8                  # layers on a face
NCOMP = 8                      # entries in a layer's composition
H_ISO = 1.0e-3                 # m, thinner layers are isothermal with the skin
STEFAN_MIN = 0.5e-3            # m, the Stefan cap acts from here up
SOOT_TRUE = 0.025              # g true soot per sim smoke unit (u, 0.015/0.6)
SHEAR_RATIO = 0.7              # shear adhesion / pull-off adhesion (u)
TE_MAX = 0.6                   # dry transfer efficiency ceiling (u)
P0_TRANSFER = 5.0e3            # Pa (u)
L_SLIDE_MIN = 0.02             # m
WET_MEMORY = 60.0              # s, a face dry this long forgets t_wet (u)
HOLD_MIN = 1.0e-4              # m, floor of a hold thickness
BYTES_PER_LAYER = 157          # 8+4+8+4+4+4+4+8+4+1+4+4+4+32+64


def pack_key(species, form, sz=0):
    """Component key (T1): species<<10 | form<<8 | sz."""
    return (int(species) << KEY_SPECIES_SHIFT) | (int(form) << KEY_FORM_SHIFT) | int(sz)


def unpack_key(k):
    k = int(k)
    return k >> KEY_SPECIES_SHIFT, (k >> KEY_FORM_SHIFT) & 3, k & 255


def d_of_sz(sz):
    """Particle diameter of size class sz, m (0 gives 0.0)."""
    return 0.0 if sz == 0 else SZ_D0_M * 2.0 ** (sz / SZ_PER_OCTAVE)


def sz_of_d(d):
    """Nearest size class of a diameter in metres (at least 1)."""
    return int(min(255, max(1, round(SZ_PER_OCTAVE * np.log2(d / SZ_D0_M)))))


# ---- species ids: ONE remap dict -------------------------------------------------
# provisional ids for species T1's seed does not hold yet
PROVISIONAL = {
    "plasma": 201, "red_cells": 202, "soot": 206, "linseed": 209, "beeswax": 211,
    "tar": 212, "ink_carbon": 213, "ink_gall": 214, "ochre": 215, "gum": 216,
}
# my name -> T1 seed name, where the seed has the species
SEED_NAME = {"water": "water", "lamp_oil": "lamp_oil", "ash": "ash",
             "salt": "sodium_chloride", "clay": "kaolinite", "quartz": "quartz",
             "chalk": "calcite", "iron_oxide": "iron_oxide", "tallow": "tallow",
             "acid": "sulfuric_acid", "weak_acid": "acetic_acid"}
LEGACY_FALLBACK = {"water": 12, "lamp_oil": 13, "ash": 5, "acid": 14, "weak_acid": 15,
                   "blood": 16, "salt": 200, "clay": 203, "quartz": 204, "chalk": 207,
                   "iron_oxide": 208, "tallow": 210}


def _seed_ids():
    """name -> id out of T1's species_seed if the file exists, else {}."""
    try:
        from . import species_seed as seed
        return {r["name"]: int(r["id"]) for r in list(seed.LEGACY) + list(seed.SEED)}
    except (ImportError, AttributeError, KeyError, TypeError):
        return {}


def _build_ids():
    seed = _seed_ids()
    ids = {}
    for name, fallback in LEGACY_FALLBACK.items():
        ids[name] = seed.get(SEED_NAME.get(name, name), fallback)
    ids["blood"] = seed.get("blood", 16)
    ids.update(PROVISIONAL)
    return ids


ID = _build_ids()            # THE remap dict: my name -> species id
# convenience aliases for the same species at other sizes
ID["silt"] = ID["sand"] = ID["quartz"]
ID["chalk_dust"] = ID["chalk"]
NAME_OF = {}
for _n, _i in ID.items():
    NAME_OF.setdefault(_i, _n)
for _n in ("quartz", "clay", "salt", "chalk"):
    NAME_OF[ID[_n]] = _n

WATER, LAMP_OIL, ASH = ID["water"], ID["lamp_oil"], ID["ash"]
SALT, CLAY, QUARTZ, SOOT = ID["salt"], ID["clay"], ID["quartz"], ID["soot"]
PLASMA, RED_CELLS, CHALK = ID["plasma"], ID["red_cells"], ID["chalk"]
IRON_OXIDE, LINSEED, TALLOW = ID["iron_oxide"], ID["linseed"], ID["tallow"]
BEESWAX, TAR, INK_CARBON = ID["beeswax"], ID["tar"], ID["ink_carbon"]
INK_GALL, OCHRE, GUM = ID["ink_gall"], ID["ochre"], ID["gum"]
ACID_SPECIES = frozenset({ID["acid"], ID["weak_acid"]})   # rejected by add() (4.14)

# the fluid ids of materials.py (WATER 1, OIL 2, BLOOD 8) <-> species, for drops
FLUID_OF_NAME = {"water": 1, "lamp_oil": 2, "blood": 8}


def fluid_id_for(species_masses):
    """materials fluid id of a drop: blood if plasma is in it, oil if the
    liquid is mostly oil class, else water. species_masses: {species: g}."""
    if species_masses.get(PLASMA, 0.0) > 0.0:
        return 8
    oil = sum(g for s, g in species_masses.items()
              if ROWS.get(s) is not None and ROWS[s]["liq_class"] in ("oil", "wax"))
    wat = sum(g for s, g in species_masses.items()
              if ROWS.get(s) is not None and ROWS[s]["liq_class"] == "water")
    return 2 if oil > wat else 1


# ---- species rows ------------------------------------------------------------------
_DEFAULT = dict(
    name="?", M=100.0, liq_class=None, tags=(),
    rho_l=1.0, rho_c=2.0, rho_d=None,          # liquid, crystal/grain/bound, dissolved solute
    por_dep=0.0, por_wet=0.0, d0=None,         # deposit porosity, wet-collapse, fixed grain size m
    cp_l=2.0, cp_c=1.0, k_l=0.2, k_c=0.5,      # liquid/dissolved vs crystal/grain/bound
    Tsol=None, Tliq=None, Lf0=0.0,             # melting RANGE, latent at 0 C reference
    Lv0=0.0, Lv_slope=0.0, Tb0=None,           # Lv(T)=Lv0+Lv_slope*T; boil point of the pure liquid
    hsol=0.0, sol0=None, sol_slope=0.0,        # heat of solution; solubility g/g water
    nu=1.0,                                    # van't Hoff i (Raoult stand-in only)
    eps_l=0.9, eps_c=0.9, alpha_l=0.5, alpha_c=0.5, albedo=(0.5, 0.5, 0.5),
    kappa=0.0,                                 # hiding area m2/g (optics and grain cover)
    theta_wet=40.0,                            # contact angle with water, deg (soot above 90)
    mu_lube=0.3, mu_lube_c=None,
    a_wet=200.0, a_dry=200.0, a_bnd=1.0e5, a_cry=None,      # pull-off adhesion Pa
    c_wet=0.0, c_dry=100.0, c_bnd=1.0e5, c_cry=None,        # cohesion Pa
    tau_ref=0.0, phi_ref=0.3,                  # yield stress of a grain suspension
    sigma=0.03, eta20=1.0, eta_B=0.0,          # surface tension, viscosity at 20 C, Andrade B
    kin=None,                                  # irreversible kinetics (4.9)
    T_ign=None, dHc=0.0, rate_cap=0.0, soot_yield=0.0,
    K_wear=0.0, H=1.0e7,
    acid=False,
)


def _row(name, **kw):
    r = dict(_DEFAULT)
    r["name"] = name
    r.update(kw)
    if r["rho_d"] is None:
        r["rho_d"] = r["rho_l"]
    if r["mu_lube_c"] is None:
        r["mu_lube_c"] = r["mu_lube"]
    return r


# kinetics row: from_form -> BOUND, rate k_ref at T_ref (C), Ea J/mol, o2 gain g/g,
# dH J/g released on binding, h_ref (m) oxygen-limited thickness (None = none)
def kinetics(k_ref, T_ref, Ea, o2=0.0, dH=0.0, h_ref=None, src=DISSOLVED):
    return dict(k_ref=k_ref, T_ref=T_ref, Ea=Ea, o2=o2, dH=dH, h_ref=h_ref, src=src)


def calibrate_arrhenius(T1, th1, T2, th2):
    """(k_ref at T1, Ea) of a first-order law through two half-lives:
    half-life th1 s at T1 C and th2 s at T2 C. Used for blood (u)."""
    k1, k2 = np.log(2.0) / th1, np.log(2.0) / th2
    i1, i2 = 1.0 / (T1 + 273.15), 1.0 / (T2 + 273.15)
    Ea = R_GAS * np.log(k1 / k2) / (i2 - i1)
    return float(k1), float(Ea)


_K_BLOOD, _EA_BLOOD = calibrate_arrhenius(65.0, 10.0, 48.0, 24.0 * 3600.0)

ROWS = {}


def _add(name, **kw):
    sid = ID[name]
    ROWS[sid] = _row(name, **kw)


# water: its CRYSTAL is ice/frost (por_dep 0.85 for frost, u)
_add("water", M=18.015, liq_class="water", tags=("water", "solvent"),
     rho_l=0.998, rho_c=0.917, por_dep=0.85, por_wet=0.85, cp_l=4.18, cp_c=2.09,
     k_l=0.60, k_c=2.2, Tsol=0.0, Tliq=0.0, Lf0=333.55, Lv0=2501.0, Lv_slope=-2.37,
     Tb0=100.0, eps_l=0.95, eps_c=0.97, alpha_l=0.1, alpha_c=0.25, kappa=0.0,
     albedo=(0.9, 0.9, 0.9), mu_lube=0.15, mu_lube_c=0.07,
     a_wet=300.0, a_dry=300.0, a_cry=3.0e5, c_wet=0.0, c_dry=0.0, c_cry=1.5e6,
     sigma=0.0728, eta20=1.002, eta_B=0.0, nu=1.0)
_add("lamp_oil", M=170.0, liq_class="oil", tags=("oil", "fuel"),
     rho_l=0.80, rho_c=0.85, cp_l=2.0, cp_c=1.9, k_l=0.14, k_c=0.14,
     Tsol=-20.0, Tliq=-20.0, Lf0=0.0, eps_l=0.9, alpha_l=0.7, albedo=(0.3, 0.25, 0.1),
     mu_lube=0.08, a_wet=150.0, a_dry=150.0, c_wet=0.0, c_dry=0.0,
     sigma=0.028, eta20=2.5, eta_B=3500.0, T_ign=210.0, dHc=43000.0, rate_cap=10.0,
     soot_yield=0.03)
_add("linseed", M=880.0, liq_class="oil", tags=("oil", "fuel", "drying"),
     rho_l=0.93, rho_c=1.0, cp_l=1.9, cp_c=1.7, k_l=0.17, k_c=0.2,
     eps_l=0.9, alpha_l=0.7, albedo=(0.55, 0.4, 0.15), mu_lube=0.08,
     a_wet=150.0, a_dry=150.0, a_bnd=5.0e6, c_wet=0.0, c_dry=0.0, c_bnd=8.0e6,
     sigma=0.033, eta20=55.0, eta_B=3500.0, T_ign=230.0, dHc=39000.0, rate_cap=8.0,
     soot_yield=0.03,
     kin=kinetics(np.log(10.0) / 172800.0, 20.0, 60000.0, o2=0.12, dH=300.0,
                  h_ref=50e-6))
_add("tar", M=200.0, liq_class="oil", tags=("oil", "fuel", "creosote"),
     rho_l=1.05, rho_c=1.1, cp_l=1.8, cp_c=1.5, k_l=0.15, k_c=0.2,
     Tsol=20.0, Tliq=60.0, Lf0=100.0, eps_l=0.9, alpha_l=0.95, albedo=(0.05, 0.05, 0.05),
     kappa=2.0, mu_lube=0.1, a_wet=500.0, a_dry=3000.0, c_dry=2000.0,
     sigma=0.035, eta20=300.0, eta_B=4500.0, T_ign=300.0, dHc=35000.0, rate_cap=6.0,
     soot_yield=0.1)
_add("tallow", M=850.0, liq_class="wax", tags=("wax", "fuel"),
     rho_l=0.90, rho_c=0.93, cp_l=2.3, cp_c=2.0, k_l=0.17, k_c=0.23,
     Tsol=36.0, Tliq=45.0, Lf0=160.0, eps_l=0.9, alpha_l=0.5, albedo=(0.9, 0.88, 0.8),
     mu_lube=0.1, a_wet=300.0, a_dry=800.0, c_dry=1500.0,
     sigma=0.032, eta20=40.0, eta_B=3000.0, T_ign=250.0, dHc=40000.0, rate_cap=5.0,
     soot_yield=0.05)
_add("beeswax", M=700.0, liq_class="wax", tags=("wax", "fuel"),
     rho_l=0.92, rho_c=0.96, cp_l=2.4, cp_c=2.1, k_l=0.17, k_c=0.25,
     Tsol=35.0, Tliq=64.0, Lf0=177.0, eps_l=0.9, alpha_l=0.5, albedo=(0.9, 0.8, 0.5),
     mu_lube=0.1, a_wet=300.0, a_dry=1200.0, c_dry=3000.0,
     sigma=0.03, eta20=60.0, eta_B=3000.0, T_ign=250.0, dHc=42000.0, rate_cap=5.0,
     soot_yield=0.05)
_add("salt", M=58.44, liq_class="water", tags=("mineral", "salt", "solute"),
     rho_l=2.165, rho_c=2.165, rho_d=3.516, cp_l=0.28, cp_c=0.864, k_l=0.45, k_c=6.5,
     Tsol=801.0, Tliq=801.0, hsol=66.0, sol0=0.356, sol_slope=5.0e-5, nu=1.86,
     eps_c=0.9, alpha_c=0.3, albedo=(0.9, 0.9, 0.9), kappa=0.3,
     a_wet=300.0, a_dry=2000.0, c_dry=3000.0, c_cry=2000.0, a_cry=2000.0,
     mu_lube=0.3, mu_lube_c=0.35)
_add("plasma", M=66000.0, liq_class="water", tags=("protein", "solute", "blood"),
     rho_l=1.35, rho_c=1.35, rho_d=1.35, cp_l=2.5, cp_c=2.0, k_l=0.3, k_c=0.3,
     albedo=(0.45, 0.02, 0.03), kappa=3.0, mu_lube=0.2,
     a_wet=300.0, a_dry=5.0e4, a_bnd=5.0e5, c_wet=0.0, c_dry=2.0e4, c_bnd=6.0e5,
     kin=kinetics(_K_BLOOD, 65.0, _EA_BLOOD, o2=0.0, dH=50.0))
_add("red_cells", M=1.0, liq_class=None, tags=("cell", "blood", "organic"),
     rho_l=1.10, rho_c=1.10, por_dep=0.40, por_wet=0.40, cp_l=3.5, cp_c=3.0,
     k_l=0.4, k_c=0.4, albedo=(0.5, 0.02, 0.03), kappa=3.0, mu_lube=0.2,
     a_wet=300.0, a_dry=3.0e4, a_bnd=3.0e5, c_dry=1.0e4, c_bnd=3.0e5)
_add("clay", M=258.0, liq_class=None, tags=("mineral", "clay"),
     rho_l=2.6, rho_c=2.6, por_dep=0.70, por_wet=0.50, cp_l=0.9, cp_c=0.9,
     k_l=1.0, k_c=1.0, albedo=(0.55, 0.42, 0.3), kappa=0.8, mu_lube=0.25,
     eps_c=0.93, alpha_c=0.6, tau_ref=200.0, phi_ref=0.30,
     a_wet=300.0, a_dry=3000.0, a_bnd=1.0e5, c_wet=0.0, c_dry=2500.0)
_add("quartz", M=60.08, liq_class=None, tags=("mineral", "silt"),
     rho_l=2.65, rho_c=2.65, por_dep=0.45, por_wet=0.45, cp_l=0.74, cp_c=0.74,
     k_l=1.4, k_c=1.4, albedo=(0.6, 0.55, 0.45), kappa=0.3, mu_lube=0.4,
     eps_c=0.93, alpha_c=0.6, tau_ref=60.0, phi_ref=0.30,
     a_wet=100.0, a_dry=300.0, c_wet=0.0, c_dry=100.0)
_add("soot", M=12.0, liq_class=None, tags=("carbon", "soot"),
     rho_l=1.8, rho_c=1.8, por_dep=0.956, por_wet=0.70, d0=0.5e-6,
     cp_l=0.7, cp_c=0.7, k_l=0.05, k_c=0.05, eps_c=0.95, alpha_c=0.95,
     albedo=(0.03, 0.03, 0.03), kappa=8.0, mu_lube=0.3, theta_wet=110.0,
     a_wet=100.0, a_dry=100.0, c_wet=0.0, c_dry=50.0,
     T_ign=500.0, dHc=32800.0, rate_cap=1.0, K_wear=1e-3, H=5e5)
_add("ash", M=60.0, liq_class=None, tags=("mineral", "ash"),
     rho_l=2.5, rho_c=2.5, por_dep=0.94, por_wet=0.80, cp_l=0.8, cp_c=0.8,
     k_l=0.12, k_c=0.12, eps_c=0.9, alpha_c=0.8, albedo=(0.55, 0.55, 0.55),
     kappa=0.5, mu_lube=0.3, a_wet=200.0, a_dry=200.0, c_dry=80.0)
_add("chalk", M=100.09, liq_class=None, tags=("mineral", "chalk"),
     rho_l=2.71, rho_c=2.71, por_dep=0.60, por_wet=0.55, d0=5e-6, cp_l=0.82,
     cp_c=0.82, k_l=0.8, k_c=0.8, albedo=(0.92, 0.92, 0.9), kappa=0.5, mu_lube=0.35,
     a_wet=100.0, a_dry=200.0, c_dry=100.0, K_wear=1e-3, H=1e6)
_add("iron_oxide", M=159.7, liq_class=None, tags=("oxide", "rust"),
     rho_l=5.2, rho_c=5.2, por_dep=0.30, por_wet=0.30, cp_l=0.65, cp_c=0.65,
     k_l=1.0, k_c=1.0, eps_c=0.85, alpha_c=0.85, albedo=(0.45, 0.2, 0.1), kappa=1.0,
     mu_lube=0.45, a_wet=3.0e5, a_dry=3.0e5, a_bnd=3.0e5, c_dry=1.0e5, c_bnd=1.0e6)
_add("ink_carbon", M=12.0, liq_class=None, tags=("carbon", "pigment"),
     rho_l=1.8, rho_c=1.8, por_dep=0.5, por_wet=0.5, d0=0.1e-6, cp_l=0.7, cp_c=0.7,
     albedo=(0.05, 0.05, 0.05), kappa=5.0, mu_lube=0.3, a_wet=200.0, a_dry=2000.0,
     c_dry=2000.0)
_add("ink_gall", M=300.0, liq_class="water", tags=("dye", "solute"),
     rho_l=1.2, rho_c=1.2, rho_d=1.2, cp_l=2.0, cp_c=1.5, albedo=(0.08, 0.07, 0.1),
     kappa=3.0, mu_lube=0.2, a_wet=300.0, a_dry=5000.0, a_bnd=5.0e5, c_dry=5000.0,
     c_bnd=5.0e5,
     kin=kinetics(np.log(2.0) / (4.0 * 3600.0), 20.0, 50000.0, o2=0.0, dH=0.0))
_add("ochre", M=160.0, liq_class=None, tags=("mineral", "pigment"),
     rho_l=3.5, rho_c=3.5, por_dep=0.5, por_wet=0.5, d0=2e-6, cp_l=0.8, cp_c=0.8,
     albedo=(0.6, 0.25, 0.1), kappa=2.0, mu_lube=0.3, a_wet=200.0, a_dry=1000.0,
     c_dry=800.0)
_add("gum", M=200.0, liq_class="water", tags=("binder", "solute"),
     rho_l=1.3, rho_c=1.3, rho_d=1.3, cp_l=2.0, cp_c=1.4, albedo=(0.8, 0.7, 0.4),
     kappa=0.5, mu_lube=0.2, a_wet=400.0, a_dry=2.0e5, a_bnd=2.0e5, c_dry=1.0e5,
     c_bnd=2.0e5)
# rows for acid species exist only so add() can name them in its error
for _n in ("acid", "weak_acid"):
    ROWS[ID[_n]] = _row(_n, liq_class="water", tags=("acid",), acid=True)


def row(species):
    """Species row by id (KeyError for a species with none)."""
    return ROWS[int(species)]


# ---- NaCl liquidus (CRC, u): wt% salt in the brine at the freezing point ----------
# SOURCED (tests/data/sourced_numbers.json nacl.freezing_curve_C and nacl.eutectic,
# retrieved 2026-10-29): the ice liquidus by weight percent, 0 to 23, then the eutectic
_FREEZE_PCT = (-0.56, -1.11, -1.72, -2.33, -2.94, -3.61, -4.33, -5.06, -5.78, -6.56,
               -7.33, -8.17, -9.06, -9.94, -10.89, -11.89, -12.94, -14.06, -15.22,
               -16.44, -17.78, -19.17, -20.67)
LIQUIDUS_WT = (0.0,) + tuple(float(i) for i in range(1, 24)) + (23.31,)
LIQUIDUS_T = (0.0,) + _FREEZE_PCT + (-21.12,)
T_EUTECTIC = -21.12
WT_EUTECTIC = 0.2331
K_F_WATER = 1.853               # K kg/mol
K_B_WATER = 0.512               # K kg/mol

# ---- substrate rows (one per material id, plus records) ------------------------
# rough_um wall retention roughness (u); theta per liquid class (deg); porosity of
# the substrate; S sorptivity g/m2/s^0.5 along/across (u); d_pore_um; eps, alpha;
# mu_dry; a_pull pull-off adhesion Pa by layer class; H Pa; W_adh J/m2 (spall); E Pa
_SUB = dict(rough_um=100.0, theta=dict(water=60.0, oil=20.0, wax=40.0),
            porosity=0.0, S_along=0.0, S_across=0.0, d_pore_um=0.0,
            eps=0.9, alpha=0.6, mu_dry=0.4, thickness=None,
            a_pull=dict(liquid=300.0, grain=500.0, solid=5.0e5),
            H=1.0e9, W_adh=10.0, E=5.0e10, k=1.0, C_skin=24.0)
SUBSTRATE = {}


def _sub(mat, name, **kw):
    r = dict(_SUB)
    r["name"] = name
    r.update(kw)
    SUBSTRATE[mat] = r


_sub(0, "air", rough_um=0.0, eps=0.0, alpha=0.0)
_sub(1, "wood", rough_um=150.0, porosity=0.55, S_along=13.0, S_across=4.0,
     d_pore_um=10.0, eps=0.9, alpha=0.6, mu_dry=0.4, H=3.0e7, E=1.0e10, k=0.15,
     theta=dict(water=50.0, oil=15.0, wax=35.0), C_skin=4.4)
_sub(2, "stone", rough_um=100.0, porosity=0.05, S_along=3.0, S_across=3.0,
     d_pore_um=5.0, eps=0.93, alpha=0.6, mu_dry=0.5, H=1.0e9, E=5.0e10, k=2.0,
     a_pull=dict(liquid=300.0, grain=800.0, solid=1.0e6), C_skin=24.0)
_sub(3, "iron", rough_um=20.0, eps=0.8, alpha=0.7, mu_dry=0.4, H=1.2e9, E=2.0e11,
     k=55.0, theta=dict(water=70.0, oil=20.0, wax=40.0), W_adh=10.0,
     a_pull=dict(liquid=300.0, grain=300.0, solid=5.0e5))
_sub(4, "flesh", rough_um=30.0, porosity=0.1, S_along=2.0, S_across=2.0,
     d_pore_um=3.0, eps=0.98, alpha=0.6, mu_dry=0.6, H=1e6, E=1e6, k=0.37,
     theta=dict(water=70.0, oil=40.0, wax=50.0))
_sub(5, "ash", rough_um=500.0, porosity=0.8, S_along=20.0, S_across=20.0,
     d_pore_um=50.0, eps=0.9, alpha=0.8, mu_dry=0.5, k=0.1)
_sub(6, "glass", rough_um=20.0, eps=0.92, alpha=0.1, mu_dry=0.4, H=5e9, E=7e10,
     k=0.9, theta=dict(water=30.0, oil=10.0, wax=30.0))
_sub(7, "leaf", rough_um=50.0, porosity=0.3, S_along=3.0, S_across=3.0, d_pore_um=5.0,
     eps=0.95, alpha=0.5, mu_dry=0.5, k=0.1, theta=dict(water=80.0, oil=20.0, wax=40.0))
_sub(8, "char", rough_um=300.0, porosity=0.7, S_along=15.0, S_across=15.0,
     d_pore_um=20.0, eps=0.9, alpha=0.9, mu_dry=0.5, k=0.08)
_sub(9, "tin", rough_um=20.0, eps=0.1, alpha=0.4, mu_dry=0.4, k=66.8)
_sub(10, "lead", rough_um=30.0, eps=0.28, alpha=0.5, mu_dry=0.4, k=35.3)
_sub(11, "rope", rough_um=300.0, porosity=0.6, S_along=10.0, S_across=10.0,
     d_pore_um=30.0, eps=0.9, alpha=0.6, mu_dry=0.6, k=0.1)
SUB_PAPER, SUB_HIDE, SUB_SKIN = 100, 101, 102
_sub(SUB_PAPER, "paper", rough_um=10.0, porosity=0.7, S_along=30.0, S_across=30.0,
     d_pore_um=10.0, eps=0.92, alpha=0.2, mu_dry=0.5, thickness=1.0e-4, k=0.06,
     theta=dict(water=50.0, oil=20.0, wax=40.0))
_sub(SUB_HIDE, "hide", rough_um=60.0, porosity=0.4, S_along=8.0, S_across=8.0,
     d_pore_um=4.0, eps=0.9, alpha=0.5, mu_dry=0.6, thickness=2.0e-3, k=0.2)
_sub(SUB_SKIN, "skin", rough_um=30.0, porosity=0.1, S_along=2.0, S_across=2.0,
     d_pore_um=3.0, eps=0.98, alpha=0.6, mu_dry=0.6, k=0.37,
     theta=dict(water=70.0, oil=40.0, wax=50.0))


def substrate(sub):
    """A substrate row from a mat id, a name, or a row dict."""
    if isinstance(sub, dict):
        return sub
    if sub is None:
        return SUBSTRATE[2]
    if isinstance(sub, str):
        for r in SUBSTRATE.values():
            if r["name"] == sub:
                return r
        raise KeyError(sub)
    return SUBSTRATE[int(sub)]


# ---- label table (describe(): a READING of the composition, never stored) --------
# Each rule: label, then conditions (quantity, op, value). Quantities are mass
# shares of the layer: liq (DISSOLVED solvent), dis (all DISSOLVED), cry:<tag>,
# grain:<tag>, bnd:<tag>, any:<tag>. The first rule that holds names the layer.
LABELS = (
    ("frost", (("cry:water", ">=", 0.5), ("por", ">=", 0.5))),
    ("ice", (("cry:water", ">=", 0.5),)),
    ("rust", (("bnd:oxide", ">=", 0.3),)),
    ("rust", (("any:oxide", ">=", 0.5), ("liq", "<", 0.1))),
    ("set blood", (("bnd:blood", ">=", 0.3),)),
    ("dried blood", (("any:blood", ">=", 0.3), ("liq", "<", 0.10))),
    ("wet blood", (("any:blood", ">=", 0.1), ("liq", ">=", 0.10))),
    ("soot", (("any:soot", ">=", 0.5),)),
    ("ash", (("any:ash", ">=", 0.5),)),
    ("wax", (("any:wax", ">=", 0.5),)),
    ("cured oil", (("bnd:drying", ">=", 0.5),)),
    ("oil", (("any:oil", ">=", 0.5),)),
    ("creosote", (("any:creosote", ">=", 0.5),)),
    ("wet mud", (("grain:mineral", ">=", 0.40), ("liq", ">=", 0.10))),
    ("dried mud", (("grain:clay", ">=", 0.3), ("liq", "<", 0.10))),
    ("dust", (("grain:mineral", ">=", 0.5), ("liq", "<", 0.10))),
    ("chalk", (("any:chalk", ">=", 0.5),)),
    ("ink", (("any:pigment", ">=", 0.2),)),
    ("ink", (("any:dye", ">=", 0.05),)),
    ("brine", (("liq", ">=", 0.5), ("any:salt", ">=", 0.02))),
    ("salt crust", (("cry:salt", ">=", 0.5),)),
    ("water", (("liq", ">=", 0.5),)),
)
COLOUR_WET = 0.7                 # a wet surface is darker: albedo x 0.7 (u)


def fresh_arrays():
    """Zero-length pool arrays (name -> array), the one place the dtypes live.
    Their itemsizes add up to BYTES_PER_LAYER (a test checks)."""
    return dict(
        key=np.empty(0, np.int64), below=np.empty(0, np.int32),
        m=np.empty(0, np.float64), cover=np.empty(0, np.float32),
        por=np.empty(0, np.float32), adh=np.empty(0, np.float32),
        coh=np.empty(0, np.float32), t_wet=np.empty(0, np.float32),
        t_last=np.empty(0, np.float64), inside=np.empty(0, np.uint8),
        depth=np.empty(0, np.float32), pat=np.empty(0, np.int32),
        content=np.empty(0, np.int32),
        ckey=np.empty((0, NCOMP), np.uint32), cm=np.empty((0, NCOMP), np.float64))
