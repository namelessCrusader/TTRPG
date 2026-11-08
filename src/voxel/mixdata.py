"""MIXTURES, THE DATA (wave 13, T4, law L14): every table, every source key.

One module owns the numbers L14 runs on, so a number has one place and a name.
Nothing here computes a mixture quantity; mixwater, mixvle, mixphase, mixsol and
mixsolid do, and they read these tables. numpy and the standard library only.

Marks (the user's rule, "real references, mark what is unverified"):
  a SOURCES key   the number was read from that source
  "u"             unverified: from memory, not yet checked against a source.
                  A strict test refuses a (u) row (`refuse_u`).
  "fit:<what>"    a constant that was FITTED to a named number. A test of that
                  number only reproduces the fit, so it is an ECHO test.
  "pack:<name>"   a row added by a magic or game pack (`register_species`).
Run `python -m src.voxel.mixdata` for the report: rows per table, the (u) count
and the sources used.

Units: temperature C, pressure Pa, mass g, molality mol/kg water, M in g/mol,
latent heats J/g. Kelvin only inside functions.

The SPECIES registry here is the phase-A stand-in for the thermodynamic columns
(M, Lv, sigma, nucleation ratio, cp); T1's species.py owns identity and the pure
numbers. tests/test_mixtures_data.py compares the two once both exist.
"""
from collections import namedtuple
from contextlib import contextmanager

import numpy as np

# -- constants -----------------------------------------------------------------
R_GAS = 8.314462618          # J/mol/K
F_FAR = 96485.33212          # C/mol
T0K = 273.15
P_ATM = 101325.0
MW = 18.01528                # g/mol, water
VW = 18.05e-6                # m3/mol, water molar volume used in the osmotic law
BOIL_SIM = 100.0             # the sim's BOIL (materials.BOIL): a stand-in, compared in a test
LATENT_SIM = 2260.0          # the sim's LATENT (materials.LATENT), J/g
L_FUS0 = 333.55              # J/g, fusion of ice at 0 C (u; T8's ICE row owns the shared value)
CP_ICE = 2.09                # J/g/K
CP_WATER = 4.18              # J/g/K
SUPERCOOL_K = 2.0            # K, same as the 2 K the metal FREEZE branch uses
NO_DATA = float("nan")

# -- sources -------------------------------------------------------------------
U = "u"
SOURCES = {
    "crc": "CRC Handbook of Chemistry and Physics, 97th ed.",
    "if97": "Wagner et al. 2000, IAPWS-IF97 (J. Eng. Gas Turbines Power 122:150)",
    "mk05": "Murphy and Koop 2005, Q. J. R. Meteorol. Soc. 131:1539",
    "pitzer73": "Pitzer and Mayorga 1973, J. Phys. Chem. 77:2300",
    "ppb84": "Pitzer, Peiper and Busey 1984, J. Phys. Chem. Ref. Data 13:1",
    "norrish66": "Norrish 1966, J. Food Technol. 1:25",
    "ross75": "Ross 1975, J. Pharm. Sci. 64:1204 (named only: v1 rule, replaced by zsr)",
    "zsr": "Zdanovskii; Stokes and Robinson 1966, J. Phys. Chem. 70:2126",
    "perry": "Perry's Chemical Engineers' Handbook (Antoine constants)",
    "sander15": "Sander 2015, Atmos. Chem. Phys. 15:4399 (Henry's law constants)",
    "davies62": "Davies 1962, Ion Association",
    "andrews65": "Andrews 1965, J. Iron Steel Inst. 203:721",
    "woodhb": "Wood Handbook, FPL-GTR-190",
    "cp88": "Carsel and Parrish 1988, Water Resour. Res. 24:755",
    "vg80": "van Genuchten 1980, Soil Sci. Soc. Am. J. 44:892",
    "cda": "Copper Development Association data sheets",
    "repo92": "docs/realism-references-2.md R16 (and R12 for NaCl)",
    "repo84": "Pinho and Macedo 2005, J. Chem. Eng. Data 50:29 (source 84 of realism-references-2)",
    "janaf": "NIST-JANAF Thermochemical Tables (to be opened in the source-check pass)",
    "repo:materials.MELT": "read from src/voxel/materials.py at first use, never retyped",
    "fit:crc-nacl-bp": "kappa fitted so saturated NaCl boils at 108.70 C (CRC 108.7)",
    "fit:crc-azeotrope": "Van Laar pair fitted to the ethanol-water azeotrope 0.894 / 78.15 C",
    "fit:calcite-898": "calcite ln K constant shifted so 1 atm lands at 898 C (repo92 refit)",
    "fit:wood-gab": "GAB constants fitted to three Wood Handbook points at 20 C",
    "fit:ethanol-ice": "ethanol-water ice line digitised from the usual 20 and 40 percent ABV rules of thumb",
}


def src_ok(src):
    """A source mark is legal: a SOURCES key, 'u', 'fit:<what>' that is in SOURCES,
    or 'pack:<name>'."""
    return src == U or src in SOURCES or (isinstance(src, str) and src.startswith("pack:") and len(src) > 5)


# -- tables --------------------------------------------------------------------
# TABLES[name][key] = (value, src). Every number a law reads sits in one of these
# rows, and every row carries a source mark.
TABLES = {}


def _table(name, rows):
    TABLES[name] = dict(rows)
    return TABLES[name]


def row(table, key):
    return TABLES[table][key][0]


def src_of(table, key):
    return TABLES[table][key][1]


def refuse_u(table, key):
    """A strict (independent) test calls this on each row it relies on: it raises
    when the row is still marked (u)."""
    v, s = TABLES[table][key]
    if s == U:
        raise AssertionError(f"strict test refuses the (u) row {table}[{key}]")
    return v


def unverified():
    """[(table, key)] of every row still marked (u)."""
    return [(t, k) for t, rows in TABLES.items() for k, (_, s) in rows.items() if s == U]


# IAPWS-IF97 region 4 (Wagner et al. 2000, eq. 29-31): saturation line.
_table("if97", {
    "n": (np.array([0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2,
                    0.12020824702470e5, -0.32325550322333e7, 0.14915108613530e2,
                    -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849,
                    0.65017534844798e3]), "if97"),
    "p_min_pa": (611.657, "if97"),       # triple point
    "p_max_pa": (22.064e6, "if97"),      # critical point
    "t_max_c": (373.946, "if97"),
})

# Murphy and Koop 2005: ice and supercooled water vapour pressure (T in K, p in Pa).
_table("mk05", {
    "ice": (np.array([9.550426, -5723.265, 3.53068, -0.00728332]), "mk05"),
    "liquid": (np.array([54.842763, -6763.22, -4.210, 0.000367, 0.0415, 218.8,
                         53.878, -1331.22, -9.44523, 0.014025]), "mk05"),
})

# Latent heat of fusion: Lfus(T) = 333.55 + 2.09 T (Kirchhoff, cp liquid 4.18, ice 2.09).
_table("water_misc", {
    "lfus0": (L_FUS0, U),
    "cp_ice": (CP_ICE, U),
    "cp_liquid": (CP_WATER, U),
    "latent_sim": (LATENT_SIM, "repo:materials.MELT"),
})

# Debye-Hueckel slope Aphi(T), T in C (u).
_table("aphi", {
    0.0: (0.3767, U), 25.0: (0.3915, U), 50.0: (0.4102, U), 75.0: (0.4333, U),
    100.0: (0.4617, U), 126.85: (0.495, U),
})

# Pitzer parameters, 1:1 electrolytes: b0, b1, C at 25 C; slopes below 25 C;
# kappa scales the m-dependent part above 25 C (s = 1 + kappa (T - 25)).
_table("pitzer", {
    "nacl_25": ((0.0765, 0.2664, 0.00127), "pitzer73"),
    "nacl_slope": ((7.159e-4, 7.0e-4, -4.1e-5), U),       # ppb84 values, exact digits unchecked
    "nacl_kappa": (-1.064e-3, "fit:crc-nacl-bp"),
    "kcl_25": ((0.04835, 0.2122, -0.00084), "pitzer73"),
    "kcl_slope": ((5.794e-4, 10.71e-4, -5.1e-5), U),
    "kcl_kappa": (-1.064e-3, U),                          # same as NaCl, NOT fitted (no anchor)
})

# Norrish constant K: aw = x_w exp(K x_s^2).
_table("norrish", {
    "sucrose": (-6.47, "norrish66"),
    "glucose": (-2.25, U),
    "fructose": (-1.54, U),
    "glycerol": (-1.16, U),
    "urea": (0.0, U),
})

# Van Laar: ln g1 = A12 (A21 x2 / (A12 x1 + A21 x2))^2, ln g2 = A21 (A12 x1 / (...))^2.
# Volatile (1) with water (2). Valid 20-100 C only; coverage() says so outside.
_table("vanlaar", {
    "ethanol": ((1.685, 0.95), "fit:crc-azeotrope"),
    "methanol": ((0.58, 0.56), U),
})

# Antoine: log10 p[mmHg] = A - B / (C + T[C]).
_table("antoine", {
    "ethanol": ((8.20417, 1642.89, 230.300), "perry"),
    "methanol": ((8.08097, 1582.271, 239.726), "perry"),
    "water_t1": ((8.07131, 1730.63, 233.426), U),     # T1's row, used only in the cross test
})
MMHG = 133.322

# Ethanol-water ice line, digitised: wt percent ethanol -> freezing point C.
_table("ethanol_ice", {
    "points": (((0.0, 0.0), (5.0, -2.0), (10.0, -4.4), (16.4, -9.0), (33.4, -27.0)), "fit:ethanol-ice"),
    "wmax_extrapolated": (40.0, U),
})

# Ethanol-water density at 20 C against mass percent ethanol, g/ml (CRC, remembered).
_table("ethanol_density", {
    "mass_pct": ((0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0), U),
    "rho": ((0.99820, 0.98187, 0.96864, 0.95382, 0.93518, 0.91384, 0.89113, 0.86766,
             0.84344, 0.81797, 0.78924), U),
})

# Masson apparent molar volume V = V0 + S sqrt(m), cm3/mol (u).
_table("masson", {
    "nacl": ((16.6, 1.8), U),
    "kcl": ((26.7, 2.0), U),
    "sucrose": ((211.5, 0.0), U),
})

# Solubility, g per 100 g water at T = 0, 20, 40, 60, 80, 100 C (CRC, remembered: all u).
SOL_T = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)
_table("solubility", {
    "nacl": ((35.65, 35.89, 36.37, 37.04, 37.93, 39.12), U),
    "kcl": ((28.1, 34.2, 40.1, 45.8, 51.3, 56.3), U),
    "sucrose": ((179.2, 203.9, 238.1, 287.3, 362.1, 487.2), U),
    "kno3": ((13.3, 31.6, 63.9, 110.0, 169.0, 246.0), U),
    "nano3": ((73.0, 87.6, 104.9, 124.7, 148.0, 180.0), U),
    "caoh2": ((0.189, 0.173, 0.141, 0.121, 0.094, 0.076), U),
})
# Single-point solubilities (g per 100 g water at one T): constant in T, flagged.
_table("solubility_point", {
    "glucose": ((20.0, 91.0), U),
    "naoh": ((20.0, 109.0), U),
})
# Independent anchors (realism-references-2 R12, source 84): NaCl 36.0 / 36.8 / 38.1 at 25 / 50 / 80 C.
_table("solubility_anchor", {
    "nacl": (((25.0, 36.0), (50.0, 36.8), (80.0, 38.1)), "repo92"),
})
# NaCl below 0.1 C is the hydrohalite line (g per 100 g water): (T, s), linear (u).
_table("hydrohalite", {
    "nacl": (((-21.1, 30.4), (0.1, 35.66)), U),       # 35.7 at 0.1 C, matched to the table's 35.66
})
# Solubility in ethanol at 25 C, g per 100 g (cosolvent rule).
_table("solubility_ethanol", {
    "nacl": (0.065, U),
    "sucrose": (0.9, U),
})
# Heat of solution, J/g of solute, + = endothermic (absorbs heat).
_table("hsol", {
    "nacl": (66.0, "repo92"),
    "sucrose": (17.0, U),
    "kno3": (345.0, U),
    "kcl": (231.0, U),            # 17.2 kJ/mol
    "naoh": (-1110.0, U),
})
# Reference eutectics (the model COMPUTES them; these are what tests compare to).
_table("eutectic_ref", {
    "nacl": ((-21.1, 23.3), "crc"),
    "kcl": ((-10.7, 19.7), U),
    "sucrose": ((-14.0, 63.0), U),
})

# Henry's law: kH25 mol/(m3 Pa), dlnk K (sander15); u except the CO2 pair.
_table("henry", {
    "o2": ((1.2e-5, 1700.0), U),
    "n2": ((6.4e-6, 1600.0), U),
    "co2": ((3.3e-4, 2400.0), "sander15"),
    "co": ((9.5e-6, 1300.0), U),
    "ch4": ((1.4e-5, 1600.0), U),
    "h2": ((7.8e-6, 500.0), U),
    "h2s": ((1.0e-3, 2100.0), U),
    "so2": ((1.2e-2, 3100.0), U),
})
_table("salting_out", {"ks_l_per_mol": (0.1, U)})

# pKa (CRC) by acid, stepwise; the first step of sulfuric acid is complete (pKa -10).
_table("pka", {
    "acetic": ((4.756,), "crc"),
    "lactic": ((3.86,), "crc"),
    "citric": ((3.128, 4.761, 6.396), "crc"),
    "co2": ((6.352, 10.329), "crc"),          # dissolved CO2 as H2CO3*
    "h2so4": ((-10.0, 1.99), "crc"),
    "h3po4": ((2.148, 7.198, 12.375), "crc"),
    "nh4": ((9.25,), "crc"),
})
_table("pkw", {
    0.0: (14.94, "crc"), 25.0: (14.00, "crc"), 50.0: (13.26, "crc"), 100.0: (12.26, "crc"),
})
_table("davies", {"A": (0.509, "davies62")})

# Water in wood: GAB isotherm fitted to the Wood Handbook at 20 C.
_table("gab", {
    "wood": ((0.0682, 7.97, 0.763), "fit:wood-gab"),
    "wood_t_coef": (-0.0045, U),             # multiplier 1 + coef (T - 20)
})
_table("gab_points", {
    "fit_points": (((30.0, 0.062), (65.0, 0.120), (90.0, 0.206)), "woodhb"),
    "check_points": (((20.0, 0.047), (50.0, 0.092), (60.0, 0.110), (80.0, 0.162), (95.0, 0.237)), U),
})

# van Genuchten: theta_r, theta_s, alpha [1/cm], n (u, remembered from cp88).
_table("vangenuchten", {
    "sand": ((0.045, 0.43, 0.145, 2.68), U),
    "sandy_loam": ((0.065, 0.41, 0.075, 1.89), U),
    "loam": ((0.078, 0.43, 0.036, 1.56), U),
    "silt_loam": ((0.067, 0.45, 0.020, 1.41), U),
    "clay_loam": ((0.095, 0.41, 0.019, 1.31), U),
    "clay": ((0.068, 0.38, 0.008, 1.09), U),
})

# Gas-solid equilibria. kind "lnk": ln K = a - b/T (K in atm). kind "dg": dG = a - b T (J/mol).
_table("gassolid", {
    "calcite": (("lnk", 16.335, 19130.0, "p_single"), "fit:calcite-898"),     # CaCO3 = CaO + CO2
    "boudouard": (("dg", 170700.0, 174.5, "co_co2"), U),                      # C + CO2 = 2 CO
    "wuestite": (("dg", -22800.0, -24.3, "co2_over_co"), U),                  # FeO + CO = Fe + CO2
    "lime_slaking": (("dg", 109000.0, 143.0, "p_single"), U),                 # Ca(OH)2 = CaO + H2O
    "magnesite": (("dg", 117000.0, 175.0, "p_single"), U),                    # MgCO3 = MgO + CO2
})
# Carbon in austenite (Smith/Duenwald-Wagner type): constants of
# log10 a_C = log10(y/(1-y)) + 2300/T - 0.92 + (3860/T - 1.8) N.
_table("austenite", {
    "a": ((2300.0, -0.92, 3860.0, -1.8), U),
})

# Electrode potentials E0 vs SHE at 25 C, V: (E0, n electrons, H+ consumed per reduction, pH of E0).
_table("e0", {k: (v, U) for k, v in {
    "Li+/Li": (-3.040, 1, 0, 0), "K+/K": (-2.931, 1, 0, 0), "Ca2+/Ca": (-2.868, 2, 0, 0),
    "Na+/Na": (-2.71, 1, 0, 0), "Mg2+/Mg": (-2.372, 2, 0, 0), "Al3+/Al": (-1.662, 3, 0, 0),
    "Mn2+/Mn": (-1.185, 2, 0, 0), "Zn2+/Zn": (-0.762, 2, 0, 0), "Cr3+/Cr": (-0.744, 3, 0, 0),
    "Fe2+/Fe": (-0.447, 2, 0, 0), "Ni2+/Ni": (-0.257, 2, 0, 0), "Sn2+/Sn": (-0.138, 2, 0, 0),
    "Pb2+/Pb": (-0.126, 2, 0, 0), "H+/H2": (0.0, 2, 0, 0), "Cu2+/Cu": (0.342, 2, 0, 0),
    "Cu+/Cu": (0.521, 1, 0, 0), "Fe3+/Fe2+": (0.771, 1, 0, 0), "Ag+/Ag": (0.800, 1, 0, 0),
    "Au3+/Au": (1.498, 3, 0, 0), "O2/OH-": (0.401, 4, 0, 0), "O2/H2O": (1.229, 4, 4, 0),
    "Cl2/Cl-": (1.358, 2, 0, 0), "Ag2S/Ag": (-0.691, 2, 0, 0), "AgCl/Ag": (0.222, 1, 0, 0),
    # new row: aluminium in base. E is the value AT pH 14; nernst() moves it with pH.
    "Al(OH)4-/Al": (-2.33, 3, 4, 14),
}.items()})

# Pure melts not in materials.MELT: Tm C, latent J/g.
_table("pure_melts", {
    "h2o": ((0.0, L_FUS0), U),
    "cu": ((1084.6, 205.0), U),
    "zn": ((419.5, 108.0), U),
    "feo": ((1377.0, 310.0), U),
    "sio2": ((1713.0, 156.0), U),
    "c": ((3800.0, 0.0), U),
})
# Species whose pure end points are READ from materials.MELT (never retyped).
MELT_NAMES = {"sn": "TIN", "pb": "LEAD", "fe": "IRON"}

# Alloys and slags: digitised bulk data. wt percent of B; (T_s, T_l) in C.
# The pure-metal end points marked None are filled from materials.MELT.
_table("sys_pbsn", {
    "TE_wE": ((183.0, 61.9), "crc"),
    "left": (((0.0, None), (5.0, 312.0), (10.0, 302.0), (20.0, 280.0), (30.0, 255.0),
              (40.0, 238.0), (50.0, 216.0), (60.0, 190.0), (61.9, 183.0)), U),
    "right": (((61.9, 183.0), (70.0, 193.0), (90.0, 215.0), (97.5, 228.5), (100.0, None)), U),
    "alpha": (((183.0, 19.2), (327.5, 0.0)), U),       # T -> wt% Sn dissolved in the Pb-rich solid
    "beta": (((183.0, 97.5), (231.9, 100.0)), U),      # T -> wt% Sn in the Sn-rich solid
})
_table("sys_cusn", {
    "points": (((0.0, 1084.6), (5.0, 970.0, 1050.0), (8.0, 900.0, 1020.0), (10.0, 854.0, 1000.0),
                (12.0, 830.0, 985.0), (15.0, 810.0, 955.0), (20.0, 798.0, 895.0),
                (22.0, 798.0, 870.0), (25.5, 798.0, 798.0)), U),
    "c90500": (((10.0, 854.0, 1000.0)), "cda"),
})
_table("sys_cuzn", {
    "points": (((0.0, 1084.6), (10.0, 1030.0, 1045.0), (20.0, 980.0, 1000.0),
                (30.0, 915.0, 955.0), (35.0, 910.0, 940.0), (40.0, 885.0, 899.0)), U),
    "c26000": (((30.0, 915.0, 955.0)), "cda"),
})
_table("sys_fec", {
    "liquidus": (((0.0, None), (0.53, 1493.0), (4.3, 1147.0)), U),
    "solidus": (((0.0, None), (0.09, 1493.0), (0.17, 1493.0), (2.11, 1147.0), (4.3, 1147.0)), U),
    "TE_wE_wA": ((1147.0, 4.3, 2.11), U),
    "wmax": (4.3, U),
    "ae1": (727.0, U),
    "ae3": ((912.0, 203.0), "andrews65"),
    "acm_pts": (((0.77, 727.0), (2.11, 1147.0)), U),
})
_table("sys_feosio2", {
    "points": (((0.0, 1377.0), (24.0, 1177.0), (29.5, 1205.0), (37.0, 1178.0),
                (50.0, 1178.0, 1400.0), (60.0, 1178.0, 1500.0)), U),
    "wuestite": ((0.0, 1377.0), "crc"),
})


# -- the species registry (phase-A stand-in; T1 owns identity) -------------------
Species = namedtuple("Species", [
    "name", "M", "kind", "nu", "z", "Lv", "sigma", "nucleation_ratio", "cp", "rho",
    "aq", "model", "src", "unverified"])
# kind: water, salt (needs a non-ideal aw model), salt_ideal (monovalent, ideal-only),
# sugar, alcohol, acid, base, gas, metal, oxide, mineral, other.
# aq: the species can sit dissolved in water (it enters ln_aw); Lv 0 = not volatile here.
# model: the key of its water-activity binary curve (an alias shares a curve and ZSR
# adds their moles, so splitting one species into two halves cannot move aw).

ORDER = []                  # species names; the LAST axis of every comp array
SPECIES = {}                # name -> Species
INDEX = {}                  # name -> column
_U_COLS = ("Lv", "sigma", "nucleation_ratio", "cp", "rho")


def _add(name, M, kind, nu=1, z=0, Lv=0.0, sigma=1.0, nucleation_ratio=1.0, cp=1.0, rho=1.0,
         aq=True, model=None, src="crc", unverified=_U_COLS):
    if not src_ok(src):
        raise ValueError(f"species {name}: bad or missing src {src!r}")
    if name in SPECIES:
        raise ValueError(f"species {name} already registered")
    if not (M > 0):
        raise ValueError(f"species {name}: M must be > 0")
    if model is None:
        model = name if name in MODEL_KEYS else ("ideal" if aq else "")
    if aq and model != "ideal" and model not in MODEL_KEYS:
        raise ValueError(f"species {name}: unknown water-activity model {model!r}")
    sp = Species(name, float(M), kind, float(nu), z, float(Lv), float(sigma), float(nucleation_ratio),
                 float(cp), float(rho), bool(aq), model, src, tuple(unverified))
    SPECIES[name] = sp
    INDEX[name] = len(ORDER)
    ORDER.append(name)
    _REV[0] += 1
    return sp


MODEL_KEYS = ("nacl", "kcl", "sucrose", "glucose", "fructose", "glycerol", "urea", "ethanol", "methanol")
_REV = [0]                  # registry revision: bumps on every change, caches key on it


def register_species(name, M, kind, *, nu=1, z=0, Lv=0.0, sigma=1.0, nucleation_ratio=1.0, cp=1.0,
                     rho=1.0, aq=True, model=None, src=None, unverified=_U_COLS):
    """Add a species row (a pack's fantasy metal, a new solute). `src` is required:
    a SOURCES key, 'u', 'fit:<what>' or 'pack:<name>'. An alias model (an existing
    key such as 'nacl') shares that species' water-activity curve."""
    if src is None:
        raise ValueError("register_species needs a src (a SOURCES key, 'u' or 'pack:<name>')")
    return _add(name, M, kind, nu=nu, z=z, Lv=Lv, sigma=sigma, nucleation_ratio=nucleation_ratio,
                cp=cp, rho=rho, aq=aq, model=model, src=src, unverified=unverified)


# water first: column 0 is the solvent, always
_add("h2o", MW, "water", nu=0, Lv=LATENT_SIM, sigma=1.0, cp=CP_WATER, rho=0.998, aq=False, model="",
     src="if97", unverified=("sigma",))
_add("nacl", 58.443, "salt", nu=2, sigma=1.0, nucleation_ratio=1.02, cp=0.86, rho=2.165)
_add("kcl", 74.551, "salt", nu=2, sigma=1.0, nucleation_ratio=1.02, cp=0.69, rho=1.984)
_add("sucrose", 342.30, "sugar", nu=1, sigma=1.0, nucleation_ratio=1.3, cp=1.24, rho=1.587)
_add("glucose", 180.156, "sugar", nu=1, sigma=1.0, cp=1.24, rho=1.54)
_add("fructose", 180.156, "sugar", nu=1, sigma=1.0, cp=1.24, rho=1.60)
_add("glycerol", 92.094, "other", nu=1, sigma=0.3, cp=2.43, rho=1.26)
_add("urea", 60.056, "other", nu=1, sigma=0.5, cp=1.34, rho=1.32)
_add("ethanol", 46.068, "alcohol", nu=1, Lv=837.0, sigma=0.0, cp=2.44, rho=0.789)
_add("methanol", 32.042, "alcohol", nu=1, Lv=1099.0, sigma=0.0, cp=2.53, rho=0.791)
_add("acetic", 60.052, "acid", nu=1, sigma=1.0, cp=2.05, rho=1.049)
_add("lactic", 90.078, "acid", nu=1, sigma=1.0, cp=2.0, rho=1.206)
_add("citric", 192.124, "acid", nu=1, sigma=1.0, cp=1.2, rho=1.665)
_add("h2so4", 98.079, "acid", nu=3, sigma=1.0, cp=1.4, rho=1.83)
_add("naoh", 39.997, "base", nu=2, sigma=1.0, cp=1.49, rho=2.13)
_add("koh", 56.106, "base", nu=2, sigma=1.0, cp=1.18, rho=2.12)
_add("caoh2", 74.093, "base", nu=3, sigma=1.0, cp=1.2, rho=2.21)
_add("k2co3", 138.205, "base", nu=3, sigma=1.0, cp=0.83, rho=2.43)
_add("na2co3", 105.988, "base", nu=3, sigma=1.0, cp=1.06, rho=2.54)
_add("kno3", 101.103, "salt_ideal", nu=2, sigma=1.0, cp=0.95, rho=2.11, nucleation_ratio=1.0)
_add("nano3", 84.995, "salt_ideal", nu=2, sigma=1.0, cp=1.09, rho=2.26)
_add("caco3", 100.087, "mineral", aq=False, cp=0.82, rho=2.71)
_add("cao", 56.077, "oxide", aq=False, cp=0.75, rho=3.34)
_add("co2", 44.009, "gas", aq=False, cp=0.85, rho=1.0)
_add("o2", 31.998, "gas", aq=False, cp=0.92, rho=1.0)
_add("n2", 28.013, "gas", aq=False, cp=1.04, rho=1.0)
_add("co", 28.010, "gas", aq=False, cp=1.04, rho=1.0)
_add("ch4", 16.043, "gas", aq=False, cp=2.2, rho=1.0)
_add("h2", 2.016, "gas", aq=False, cp=14.3, rho=1.0)
_add("fe", 55.845, "metal", aq=False, cp=0.45, rho=7.87)
_add("c", 12.011, "other", aq=False, cp=0.71, rho=2.26)
_add("cu", 63.546, "metal", aq=False, cp=0.385, rho=8.96)
_add("sn", 118.710, "metal", aq=False, cp=0.228, rho=7.31)
_add("pb", 207.2, "metal", aq=False, cp=0.129, rho=11.34)
_add("zn", 65.38, "metal", aq=False, cp=0.388, rho=7.14)
_add("feo", 71.844, "oxide", aq=False, cp=0.72, rho=5.74)
_add("fe3o4", 231.533, "oxide", aq=False, cp=0.65, rho=5.17)
_add("fe2o3", 159.688, "oxide", aq=False, cp=0.65, rho=5.24)
_add("sio2", 60.084, "oxide", aq=False, cp=0.74, rho=2.65)

# Species names that are NOT crafts or products: the "no craft words" test reads this.
CRAFT_WORDS = ("brine", "bronze", "distill", "cure", "solder", "syrup", "wash", "vodka")
# Salts that must carry a non-ideal aw model or stay out: CaCl2 and MgCl2 are not here.
AW_EXCLUDED = ("cacl2", "mgcl2")


def species(name):
    return SPECIES[name]


def col(name):
    return INDEX[name]


def n_species():
    return len(ORDER)


def M_of():
    """float64 array of molar masses in column order."""
    return np.array([SPECIES[n].M for n in ORDER])


def pure_metal_points():
    """{name: (Tm C, latent J/g)} for the pure metals, READ from materials.MELT
    (232, 327, 1538 C; 59, 23, 247 J/g). The import is lazy: this module stays
    numpy and stdlib only, and nothing here retypes the sim's numbers."""
    from . import materials as _m
    out = {}
    for sp, const in MELT_NAMES.items():
        tm, _fl, lat = _m.MELT[getattr(_m, const)]
        out[sp] = (float(tm), float(lat))
    return out


@contextmanager
def scratch_registry():
    """Snapshot the registry; whatever a test registers inside is removed again."""
    keep = (list(ORDER), dict(SPECIES), dict(INDEX), dict(SYSTEM_SPECS))
    try:
        yield
    finally:
        ORDER[:] = keep[0]
        SPECIES.clear()
        SPECIES.update(keep[1])
        INDEX.clear()
        INDEX.update(keep[2])
        SYSTEM_SPECS.clear()
        SYSTEM_SPECS.update(keep[3])
        _REV[0] += 1


# -- registered systems ----------------------------------------------------------
# A pack (or a later lane) adds a phase system as data. mixphase builds the built-in
# ones itself; SYSTEM_SPECS holds only the added ones.
SYSTEM_SPECS = {}
_RULES = ("eutectic", "linear", "lever_pair", "aqueous")


def register_system(name, spec, src=None):
    """Add a phase system as data. spec is a dict with 'rule' in eutectic, linear,
    lever_pair or aqueous and the keys mixphase documents; 'A' and 'B' must be
    registered species. A missing src is rejected."""
    if src is None or not src_ok(src):
        raise ValueError("register_system needs a src (a SOURCES key, 'u' or 'pack:<name>')")
    if spec.get("rule") not in _RULES:
        raise ValueError(f"unknown rule {spec.get('rule')!r}; one of {_RULES}")
    for k in ("A", "B"):
        if spec.get(k) not in SPECIES:
            raise ValueError(f"system {name}: species {spec.get(k)!r} is not registered")
    if name in SYSTEM_SPECS:
        raise ValueError(f"system {name} already registered")
    SYSTEM_SPECS[name] = dict(spec, src=src)
    _REV[0] += 1
    return SYSTEM_SPECS[name]


def revision():
    return _REV[0]


# -- the (u) report ----------------------------------------------------------------
def report():
    """Text report: rows per table, (u) count, fitted rows and the sources used."""
    lines = []
    used = {}
    n_rows = n_u = 0
    for t, rows in TABLES.items():
        nu_t = sum(1 for _, s in rows.values() if s == U)
        lines.append(f"{t:20s} rows {len(rows):3d}   (u) {nu_t:3d}")
        n_rows += len(rows)
        n_u += nu_t
        for _, s in rows.values():
            used[s] = used.get(s, 0) + 1
    lines.append("-" * 44)
    sp_u = sum(1 for sp in SPECIES.values() for _ in sp.unverified)
    lines.append(f"table rows {n_rows}, (u) rows {n_u}; species {len(ORDER)}, (u) species columns {sp_u}")
    lines.append("sources used: " + ", ".join(f"{k} x{v}" for k, v in sorted(used.items())))
    fits = sorted(k for k in used if k.startswith("fit:"))
    lines.append("fitted (echo tests only): " + (", ".join(fits) if fits else "none"))
    lines.append("(u) rows: " + ", ".join(f"{t}[{k}]" for t, k in unverified()))
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
