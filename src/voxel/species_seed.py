"""SPECIES SEED DATA: pure data, NO imports (wave 13, T1).

Three things live here and nothing else:

  1. The FORM and SIZE constants (DISSOLVED, CRYSTAL, GRAIN, BOUND), so
     materials.py can import this file at its end (phase B) without a cycle.
  2. LEGACY: the names, formulas, composite flags and REFERENCE numbers (the
     real-world values) for species 0-16. The ENGINE numbers of those species are
     never typed: species.py reads them from materials.py.
  3. SEED: the extra species 17-63 (copper, salt, ethanol, graphite, quartz...),
     each with an `engine` slot (what the sim runs on; empty means "same as
     ref") and a `ref` slot (the real value), a source tag or an unverified mark
     for EVERY numeric field.

Units (one place): por_grain 0..1 (porosity of the packed GRAIN form), rho g/L, c J/(g K), k W/(m K), Tm Tb Tc Tdec degC, Lf Lv hcomb
dH J/g, rho_e ohm m, M g/mol, phiV0 mL/mol, Sv mL kg^0.5 mol^-1.5, mu mPa s,
sigma_s A m^2/kg, ign degC, spread = the legacy per-tick viscosity column,
h_wet W/(m^2 K) (the legacy quench-transfer column).

Honesty rule (RULES.md): a number is either SOURCED (its tag is a key of SOURCES
and names a document the repo already holds) or it is in the row's `unverified`
tuple, which means "(u): from memory, no source opened". Nearly all of it is (u).
tests/data/sourced_numbers.json (lane A-S) turns (u) into sourced one number at a
time; until then a test that depends on a (u) number is named test_u_* and is not
a gate for a scene.
"""

# ---- forms and sizes -------------------------------------------------------
DISSOLVED, CRYSTAL, GRAIN, BOUND = 0, 1, 2, 3
FORM_NAMES = ("dissolved", "crystal", "grain", "bound")
NFORMS = 4
# Size class sz (uint8): sz = 0 means "no size" (DISSOLVED, BOUND). For sz >= 1
# the particle diameter is d = SZ_D0_M * 2 ** (sz / SZ_PER_OCTAVE) metres.
SZ_D0_M = 1e-6
SZ_PER_OCTAVE = 8
KEY_FORM_SHIFT = 8
KEY_SPECIES_SHIFT = 10

# ---- id blocks (stable: a pinned PREFIX test catches reordering) ----------
NMAT_PAD = 64                    # mat ids padded to this (uint8 views stay)
NFLUID_PAD = 64                  # fluid ids padded to this
MAT_ICE = 12                     # reserved: water BOUND. T6 owns its numbers.
SPECIES_LEGACY_END = 17          # species 0-16 are read from materials.py
SPECIES_SEED_END = 64            # 17-63 seed set
GAS_LO, GAS_HI = 64, 96          # 64-95 gases for L2b (the GAS lane picks the order)
PACK_LO = 96                     # 96+ packs and future species
SOLID_ROWS = (0, 64)             # pure solid rows: row r is mat r
FLUID_ROWS = (64, 128)           # pure fluid rows: row 64 + f is fluid f
MIX_BASE = 128                   # user mixes from here up

# Legacy mat id -> species id (mat ids 1-11 ARE species ids 1-11), fluid id ->
# species id. MTIN, MLEAD, MIRON are the LIQUID state of species 9, 10, 3.
LEGACY_FLUID_SPECIES = {0: 0, 1: 12, 2: 13, 3: 14, 4: 15, 5: 9, 6: 10, 7: 3, 8: 16}
# mat id -> (species id, form, size class) for the mats that are not the legacy
# 0-11 identity: ICE is water BOUND; the new skeleton mats follow, in order of
# first need. Copper and quartz come first, then the stones, salts and soil.
MAT_SLOTS = {
    12: (12, BOUND, 0),          # ICE (T6 numbers)
    13: (17, BOUND, 0),          # copper
    14: (21, BOUND, 0),          # quartz rock
    15: (21, GRAIN, 64),         # quartz sand, 0.25 mm
    16: (22, BOUND, 0),          # calcite (limestone, marble)
    17: (23, BOUND, 0),          # kaolinite clay
    18: (20, BOUND, 0),          # graphite
    19: (34, BOUND, 0),          # magnetite
    20: (31, BOUND, 0),          # iron oxide (rust, hematite)
    21: (18, BOUND, 0),          # salt block
    22: (30, BOUND, 0),          # tallow
    23: (29, BOUND, 0),          # sugar
    24: (32, BOUND, 0),          # quicklime
    25: (33, GRAIN, 40),         # slaked lime powder, 32 um
    26: (24, GRAIN, 48),         # humus, soil crumb
}
MAT_NAMES = {12: "ice", 13: "copper", 14: "quartz_rock", 15: "sand", 16: "calcite",
             17: "clay", 18: "graphite", 19: "magnetite", 20: "iron_oxide",
             21: "salt", 22: "tallow", 23: "sugar", 24: "quicklime",
             25: "slaked_lime", 26: "humus"}
FLUID_NAMES = {0: "none", 1: "water", 2: "oil", 3: "acid", 4: "weak_acid",
               5: "molten_tin", 6: "molten_lead", 7: "molten_iron", 8: "blood",
               9: "ethanol", 10: "molten_copper"}
# New fluid ids: only liquids that flow differently from water get one. Every
# other liquid uses `fluid_like`.
FLUID_SLOTS = {
    9: 19,                       # ethanol
    10: 17,                      # molten copper
}
# The legacy mat that stands in for the MECHANICAL rows (SPAN, TOUGH, TSHOCK,
# EMISS...) of a new mat until T9 gives it its own, and the legacy fluid whose
# miscibility, spread and quench rows stand in for a new fluid.
MAT_LIKE = {12: 2, 13: 3, 14: 2, 15: 5, 16: 2, 17: 2, 18: 2, 19: 3, 20: 2,
            21: 2, 22: 4, 23: 2, 24: 2, 25: 5, 26: 5}
FLUID_LIKE = {9: 1, 10: 7}

SOURCES = {
    "repo:realism-references-2:R12": (
        "docs/realism-references-2.md, row R12, sources 83-85 (Palandri and "
        "Kharaka 2004; Pinho and Macedo 2005; enthalpy of solution)"),
    "repo:materials.py": "the engine value, read from src/voxel/materials.py",
    "kestin1981": (
        "tests/data/sourced_numbers.json, key nacl.rel_viscosity_20C (Kestin, "
        "Khalifa, Correia, J. Phys. Chem. Ref. Data 10, 71, 1981, "
        "doi 10.1063/1.555641)"),
    "gwproject:porosity-range": (
        "GW-Project, Hydrogeologic Properties of Earth Materials and Principles of "
        "Groundwater Flow (2020), section 3.1 Table 1: sand total porosity 26-50 "
        "percent fine, 30-45 percent coarse (after Freeze and Cherry 1979); the "
        "value in the row is a point inside that range"),
    "limedata:hydrated-lime-bulk": (
        "bulk density of hydrated lime powder, 350 to 560 kg/m3 (loose to packed): "
        "MLC Hydrated Lime data sheet (22 and 35 lb/ft3, mlc.com), Boral hydrated "
        "lime data sheet (450 to 500 kg/m3), National Lime Association fact sheet "
        "(25 to 35 lb/ft3); the value in the row is a point inside that range"),
    "computed": "arithmetic on other fields of the same row (stoichiometry); "
                "inherits the status of its inputs",
}


def _row(id, name, formula, ref, src=None, states=None, like=0, fluid_like=None,
         composite=False, skeleton=(BOUND, GRAIN, CRYSTAL), grain_sz=0,
         engine=None, decomposition=None, note=""):
    """One seed row. `unverified` is every numeric field of ref/engine that has
    no source tag: so a new field is (u) until somebody sources it."""
    src = dict(src or {})
    engine = dict(engine or {})
    fields = sorted(set(ref) | set(engine))
    unverified = tuple(f for f in fields if f not in src)
    return {"id": id, "name": name, "formula": formula, "ref": ref,
            "engine": engine, "src": src, "unverified": unverified,
            "states": states or {}, "like": like, "fluid_like": fluid_like,
            "composite": composite, "skeleton": tuple(skeleton),
            "grain_sz": grain_sz, "decomposition": decomposition, "note": note}


# ---- LEGACY species 0-16: names and REFERENCE numbers ---------------------
# Engine numbers are NOT here (species.py reads materials.py). `ref` is the
# real-world value; most is (u). Water's melting number is the agreed single
# one: Tm 0, Lf 333.55 J/g (the ref slot).
_WATER_ANTOINE = (8.07131, 1730.63, 233.426, 1.0, 100.0)    # log10 P(mmHg), T degC
_ETHANOL_ANTOINE = (8.20417, 1642.89, 230.300, 1.0, 78.0)
# Andrade mu = A exp(B / T_K) mPa s, B fitted to two points. Water: 1.002 mPa s
# at 20 C and 0.547 at 50 C gives B = 1911.4 K (the spec rounds it to 1911). A is
# set so 20 C reproduces 1.002: 1.002 / exp(1911.4 / 293.15) = 1.4764e-3.
_WATER_ANDRADE = (1.4764e-3, 1911.4)

LEGACY = (
    _row(0, "void", "", {}, composite=False, note="air / empty; the row is empty"),
    _row(1, "wood", "", {}, composite=True, like=1,
         note="placeholder for a real mix (cellulose, lignin, bound water)"),
    _row(2, "stone", "", {}, composite=True, like=2,
         note="one STONE stands for limestone and granite alike until split"),
    _row(3, "iron", "Fe",
         {"M": 55.845, "rho_s": 7874.0, "c_s": 0.449, "k_s": 80.2, "rho_e": 9.71e-8,
          "Tm": 1538.0, "Lf": 247.0, "Tb": 2862.0, "Lv": 6090.0,
          "sigma_s": 217.75, "Tc": 770.0, "beta": 0.36},
         like=3, note="liquid state is fluid MIRON"),
    _row(4, "flesh", "", {}, composite=True, like=4),
    _row(5, "ash", "", {}, composite=True, like=5),
    _row(6, "glass", "SiO2", {"M": 60.084, "rho_s": 2500.0}, like=6,
         note="soda-lime glass is a mix; the pure row is a stand-in"),
    _row(7, "leaf", "", {}, composite=True, like=7),
    _row(8, "char", "C", {}, composite=True, like=8),
    _row(9, "tin", "Sn",
         {"M": 118.71, "rho_s": 7310.0, "c_s": 0.228, "k_s": 66.8, "rho_e": 1.15e-7,
          "Tm": 231.9, "Lf": 59.0}, like=9, note="liquid state is fluid MTIN"),
    _row(10, "lead", "Pb",
         {"M": 207.2, "rho_s": 11340.0, "c_s": 0.129, "k_s": 35.3, "rho_e": 2.08e-7,
          "Tm": 327.5, "Lf": 23.0}, like=10, note="liquid state is fluid MLEAD"),
    _row(11, "rope", "", {}, composite=True, like=11),
    _row(12, "water", "H2O",
         {"M": 18.015, "rho_l": 998.0, "c_l": 4.18, "k_l": 0.60, "Tm": 0.0,
          "Lf": 333.55, "Tb": 100.0, "Lv": 2257.0, "antoine": _WATER_ANTOINE,
          "mu_A": _WATER_ANDRADE[0], "mu_B": _WATER_ANDRADE[1],
          "rho_s": 917.0, "c_s": 2.09, "k_s": 2.2,       # ice: T6's row (icephase.ICE_ROW) owns
          "emiss_s": 0.97, "emiss_l": 0.96},             # these four; a cross test compares them
         like=2, skeleton=(),
         note="ice is water BOUND (mat 12); bound water never wins the mat view "
              "by name (empty skeleton); steam is a gas-lane species"),
    _row(13, "lamp_oil", "", {}, composite=True, skeleton=()),
    _row(14, "sulfuric_acid", "H2SO4", {"M": 98.079, "rho_l": 1830.0}, skeleton=(),
         note="the pure ACID row is the STOCK composition 0.983 / 0.017 water"),
    _row(15, "acetic_acid", "CH3COOH",
         {"M": 60.052, "rho_l": 1049.0, "Tm": 16.6, "Tb": 118.1}, skeleton=(),
         note="the pure WEAK_ACID row is the STOCK composition 0.05 / 0.95 water"),
    _row(16, "blood", "", {}, composite=True, skeleton=()),
)

# ---- SEED species 17-63 ----------------------------------------------------
_NACL_SOL = ((25.0, 36.0), (50.0, 36.8), (80.0, 38.1))      # g per 100 g water
# relative viscosity of NaCl brine (mu_solution / mu_water, both at 20 C and
# 0.1 MPa) against mass fraction. SOURCED: Kestin, Khalifa and Correia 1981 (J.
# Phys. Chem. Ref. Data 10, 71), read by lane A-S into tests/data/sourced_numbers.json
# key nacl.rel_viscosity_20C. The spec's old (u) table (1.14, 1.33, 1.62, 1.99,
# 2.50) was 6 to 33 percent high; 20 wt percent is 1.562, not 1.99. The numbers
# below are the sourced "relative" values, copied once.
_NACL_MU_REL = ((0.0, 1.0), (0.05, 1.0783), (0.10, 1.1884), (0.15, 1.3438),
                (0.20, 1.562), (0.25, 1.8725))
# Packed-bed porosity of a dry quartz sand (the GRAIN form). SOURCED RANGE: total
# porosity of sand 26 to 50 percent (fine) and 30 to 45 percent (coarse), Table 1
# of the GW-Project "Hydrogeologic Properties of Earth Materials and Principles of
# Groundwater Flow" (2020), after Freeze and Cherry 1979 and Domenico and Schwartz
# 1998; its worked example is a loose quartz sand of dry bulk density 1.43 g/cm3
# on particle density 2.65 g/cm3 = 0.46. 0.40 is the middle of the coarse range
# (our grain is 0.25 mm, medium sand); the POINT VALUE is a choice inside the
# sourced range, which is what the tag below means. 0.40 gives bulk 1590 g/L.
_SAND_POROSITY = 0.40
# Hydrated (slaked) lime powder, the GRAIN form of mat 25. The particle density of
# Ca(OH)2 is 2211 kg/m3 (rho_s). A heap of the powder is mostly air: supplier and
# trade-association data sheets give a LOOSE bulk density of about 350 kg/m3 (MLC:
# 22 lb/ft3), 450 to 500 (Boral), 400 to 560 (National Lime Association fact sheet:
# 25 to 35 lb/ft3), and a PACKED (tapped) value of about 560 (MLC: 35 lb/ft3).
# SOURCED RANGE 350 to 560 kg/m3, read from those data sheets on 2026-10-30 (the
# loose/packed label of the NLA figure was not in the text that was read, so it is
# kept only as part of the range). The POINT VALUE 450 kg/m3 (a poured, uncompacted
# heap) is a choice inside that range, so the number is the porosity that gives it:
_HYDRATED_LIME_BULK = 450.0
_HYDRATED_LIME_POROSITY = 1.0 - _HYDRATED_LIME_BULK / 2211.0
# Thermal conductivity of the gas that fills a pore (air at 300 K), W/m/K. Textbook
# (Incropera and DeWitt, Table A.4: 0.0263 at 300 K). Read by species.packed_k.
K_PORE_GAS = 0.0263
# sucrose solutions at 20 C, mPa s / 1.002: 10 wt percent 1.336, 20 1.945, 30
# 3.187, 40 6.162, 50 15.43, 60 58.37 (CRC table, from memory) (u)
_SUCROSE_MU_REL = ((0.0, 1.00), (0.10, 1.33), (0.20, 1.94), (0.30, 3.18),
                   (0.40, 6.15), (0.50, 15.4), (0.60, 58.3))

SEED = (
    _row(17, "copper", "Cu",
         {"M": 63.546, "rho_s": 8960.0, "c_s": 0.385, "k_s": 401.0, "rho_e": 1.68e-8,
          "emiss_s": 0.3, "Tm": 1085.0, "Lf": 209.0, "Tb": 2562.0, "Lv": 4730.0,
          "rho_l": 8020.0, "c_l": 0.495, "k_l": 166.0, "emiss_l": 0.3, "spread": 0.6, "h_wet": 5000.0},
         states={"solid": {BOUND: 13}, "liquid": 10}, like=3,
         note="liquid state is fluid 10 (molten copper)"),
    _row(18, "sodium_chloride", "NaCl",
         {"M": 58.44, "rho_s": 2165.0, "c_s": 0.864, "k_s": 6.5, "emiss_s": 0.9,
          "Tm": 801.0, "Lf": 482.0, "Tb": 1413.0, "Lv": 2900.0,
          "phiV0": 16.62, "Sv": 1.87, "c_app": 0.28, "k_app": 0.45,
          "mu_rel": _NACL_MU_REL,
          "dH_sol": 66.0, "solubility": _NACL_SOL},
         src={"dH_sol": "repo:realism-references-2:R12",
              "solubility": "repo:realism-references-2:R12",
              "mu_rel": "kestin1981"},
         states={"solid": {BOUND: 21}}, like=2, fluid_like=1,
         note="dissolved view is WATER; Masson 16.62 and 1.87 are the one set"),
    _row(19, "ethanol", "C2H5OH",
         {"M": 46.07, "rho_l": 789.0, "c_l": 2.44, "k_l": 0.167, "emiss_l": 0.9,
          "Tm": -114.1, "Lf": 108.0, "Tb": 78.37, "Lv": 838.0,
          "antoine": _ETHANOL_ANTOINE,
          "mu_A": 4.0277e-3, "mu_B": 1670.0,
          "spread": 0.50, "h_wet": 800.0,
          "hcomb": 29700.0, "o2_ratio": 2.0837, "soot_yield": 0.01},
         src={"o2_ratio": "computed"},
         states={"liquid": 9}, like=2, fluid_like=1,
         note="o2_ratio = 3 x 31.998 / 46.07 (C2H5OH + 3 O2)"),
    _row(20, "graphite", "C",
         {"M": 12.011, "rho_s": 2260.0, "c_s": 0.709, "k_s": 120.0, "rho_e": 8e-6,
          "emiss_s": 0.8, "hcomb": 32800.0, "o2_ratio": 2.6641, "soot_yield": 0.05},
         src={"o2_ratio": "computed"},
         states={"solid": {BOUND: 18}}, like=2, note="o2_ratio = 31.998 / 12.011"),
    _row(21, "quartz", "SiO2",
         {"M": 60.084, "rho_s": 2650.0, "c_s": 0.74, "k_s": 7.7, "emiss_s": 0.9,
          "Tm": 1713.0, "Lf": 141.0, "por_grain": _SAND_POROSITY},
         src={"por_grain": "gwproject:porosity-range"},
         states={"solid": {BOUND: 14, GRAIN: 15}}, like=2, grain_sz=64,
         note="rock is BOUND (mat 14), sand is GRAIN (mat 15, 0.25 mm). rho_s is "
              "the PARTICLE (quartz) density 2650: rock and every settling or "
              "particle law read it. por_grain is the porosity of the packed bed, "
              "so a cell of sand holds rho_s x (1 - por_grain) = 1590 g/L: "
              "species.bulk_density(quartz, GRAIN) is what the cell table reads. "
              "k_s 7.7 is SOLID quartz (the rock, and every grain); a bed of grains "
              "conducts about 0.3 (species.bulk_conductivity: the solid and the "
              "pore air joined by the Zehner-Schlunder packed-bed rule)"),
    _row(22, "calcite", "CaCO3",
         {"M": 100.087, "rho_s": 2710.0, "c_s": 0.82, "k_s": 3.6, "emiss_s": 0.9,
          "Tdec": 840.0, "dH_dec": 1780.0},
         states={"solid": {BOUND: 16}}, like=2, fluid_like=1,
         note="decomposes to quicklime and CO2; no melting point"),
    _row(23, "kaolinite", "Al2Si2O5(OH)4",
         {"M": 258.16, "rho_s": 2600.0, "c_s": 0.95, "k_s": 1.0, "emiss_s": 0.9},
         states={"solid": {BOUND: 17}}, like=2),
    _row(24, "humus", "",
         {"rho_skel": 1400.0, "porosity": 0.5, "rho_s": 700.0, "c_s": 1.5,
          "k_s": 0.2, "emiss_s": 0.9},
         states={"solid": {GRAIN: 26}}, like=5, composite=True, grain_sz=48,
         note="placeholder for a real mix of organic matter and minerals"),
    _row(25, "cellulose", "(C6H10O5)n",
         {"M": 162.14, "rho_s": 1500.0, "c_s": 1.3, "k_s": 0.25, "hcomb": 16500.0,
          "o2_ratio": 1.184, "soot_yield": 0.6}, src={"o2_ratio": "computed"},
         like=1,
         note="o2_ratio = 6 x 31.998 / 162.14 = 1.184 ((C6H10O5)n + 6 O2 -> 6 CO2 "
              "+ 5 H2O, one anhydroglucose unit, M 162.14). The old 1.26 was "
              "the rope row of materials.py (16.5 kJ/g over the rule of thumb 13.1 "
              "kJ per g of O2), a heat ratio, not the stoichiometry of cellulose"),
    _row(26, "collagen", "",
         {"rho_s": 1340.0, "c_s": 1.6, "k_s": 0.3, "hcomb": 17000.0},
         like=4, fluid_like=1, composite=True,
         note="a protein family; gelatin when hot and wet"),
    _row(27, "tannin", "C76H52O46",
         {"M": 1701.2, "rho_s": 1500.0, "c_s": 1.4, "hcomb": 17000.0},
         like=2, fluid_like=1, note="tannic acid as the stand-in"),
    _row(28, "starch", "(C6H10O5)n",
         {"M": 162.14, "rho_s": 1500.0, "c_s": 1.5, "k_s": 0.3, "hcomb": 17500.0},
         like=1, fluid_like=1),
    _row(29, "sucrose", "C12H22O11",
         {"M": 342.30, "rho_s": 1587.0, "c_s": 1.24, "k_s": 0.5, "emiss_s": 0.9,
          "Tm": 186.0, "hcomb": 16500.0,
          "phiV0": 211.5, "Sv": 0.0, "c_app": 1.2, "k_app": 0.35,
          "mu_rel": _SUCROSE_MU_REL},
         states={"solid": {BOUND: 23}}, like=2, fluid_like=1,
         note="solubility about 200 g per 100 g water at 20 C (u)"),
    _row(30, "tallow", "",
         {"rho_s": 900.0, "c_s": 2.0, "k_s": 0.2, "rho_l": 890.0, "c_l": 2.2,
          "k_l": 0.17, "Tm": 40.0, "hcomb": 39500.0, "o2_ratio": 2.891,
          "soot_yield": 0.6, "emiss_s": 0.9},
         src={"o2_ratio": "computed"},
         states={"solid": {BOUND: 22}}, like=4, fluid_like=2, composite=True,
         note="melts over a range near 40 C; o2_ratio = 80 x 31.998 / 885.4 "
              "(triolein C57H104O6 as the stand-in)"),
    _row(31, "iron_oxide", "Fe2O3",
         {"M": 159.69, "rho_s": 5240.0, "c_s": 0.65, "k_s": 3.0, "emiss_s": 0.9},
         states={"solid": {BOUND: 20}}, like=2, note="rust, hematite"),
    _row(32, "quicklime", "CaO",
         {"M": 56.077, "rho_s": 3340.0, "c_s": 0.75, "k_s": 7.0, "emiss_s": 0.9,
          "Tm": 2613.0},
         states={"solid": {BOUND: 24}}, like=2),
    _row(33, "slaked_lime", "Ca(OH)2",
         {"M": 74.093, "rho_s": 2211.0, "c_s": 1.18, "k_s": 1.3, "emiss_s": 0.9,
          "Tdec": 512.0, "por_grain": _HYDRATED_LIME_POROSITY},
         src={"por_grain": "limedata:hydrated-lime-bulk"},
         states={"solid": {GRAIN: 25}}, like=5, fluid_like=1, grain_sz=40,
         note="rho_s is the particle density 2211; the powder (GRAIN, mat 25) is a "
              "heap of bulk density 450 kg/m3 (sourced range 350 to 560)"),
    _row(34, "magnetite", "Fe3O4",
         {"M": 231.53, "rho_s": 5170.0, "c_s": 0.6, "k_s": 5.1, "emiss_s": 0.9,
          "sigma_s": 92.0, "Tc": 585.0, "beta": 0.36},
         states={"solid": {BOUND: 19}}, like=3),
)
