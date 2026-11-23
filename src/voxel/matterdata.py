"""MATTER DATA (wave 13, T6): the columns T1's species table does not have.

Pure data and a few one-line helpers. No imports but the standard library.
T1 (species.py) owns identity, molar mass and the apparent volumes. This file
holds a PRIVATE COPY of those two so the T6 process modules stand alone in
phase A; a skip-marked cross test compares them to T1's registry. It adds the
columns T1 lacks: heat of formation, diffusivity, Antoine, supersaturation limit,
tortuosity per host, heat of solution, fresh-air background.

Units (one place): grams, joules, seconds, metres, degC. Energies of a cell are
measured from AMBIENT = 20 C (the repo's E). Pressures in Pa.

Honesty (RULES.md): every number below is (u) = recalled from the handbooks (CRC,
IAPWS, Perry, Wood Handbook, Robinson & Stokes), not opened, unless a comment
says (c) = computed by this lane. tests/data/sourced_numbers.json (lane A-S)
turns a (u) into a sourced number one at a time.
"""

# ---- constants ------------------------------------------------------------
R = 8.314462618            # J/(mol K)
AMBIENT = 20.0             # C, the zero of every cell's E
T_REF_GAS = 293.15         # K: gas concentrations are kept at this air density
P_ATM = 101325.0           # Pa
M_WATER = 18.015           # g/mol
CW = 4.2                   # J/(g K) liquid water (the repo's number)
CV = 1.86                  # J/(g K) vapour
CI = 2.09                  # J/(g K) ice (u)
L0 = 2454.0                # J/g: 2453.5 (c); latent heat at 20 C
LF0 = 333.55               # J/g, fusion at 0 C (u: CRC 6.01 kJ/mol)
LF_SLOPE = 2.09            # J/(g K): Lfus(T) = LF0 + 2.09 T (u, T4's number)
L_CONV20 = LF0 + 20.0 * (CW - CI)   # 375.75 (c): ice's E jump at no heat exchange
RHO_ICE = 0.917            # g/ml
RHO_WATER = 1.0            # g/ml
FRESH_CO2_G_M3 = 0.74      # g/m3 of fresh air: 400 ppm at 20 C (c: 0.0168 mol/m3)

# ---- elements -------------------------------------------------------------
ATOMIC = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "Na": 22.990,
          "Cl": 35.45, "Ca": 40.078, "Fe": 55.845}

# name -> (formula as {element: count}, phase at 20 C and 1 atm, c J/(g K),
#          dHf kJ/mol (298 K, in the stated phase), apparent specific volume ml/g)
# phase: g gas, l liquid, s solid. Species keys follow T1's names where T1 has one
# (ALIAS below). Water has three keys, one per phase, because the energy books
# differ by phase.
_ROWS = {
    # dHf (u): CRC / NBS tables. c (u). v only for solutes (c from the 20 wt % densities)
    "H2O":      ({"H": 2, "O": 1}, "l", 4.2, -285.83, None),
    "H2O_g":    ({"H": 2, "O": 1}, "g", 1.86, -241.83, None),
    "H2O_s":    ({"H": 2, "O": 1}, "s", 2.09, -291.8, None),
    "NaCl":     ({"Na": 1, "Cl": 1}, "s", 0.85, -411.15, 0.355),
    "sucrose":  ({"C": 12, "H": 22, "O": 11}, "s", 1.24, -2226.1, 0.625),
    "glucose":  ({"C": 6, "H": 12, "O": 6}, "s", 1.25, -1273.3, 0.62),
    "ethanol":  ({"C": 2, "H": 6, "O": 1}, "l", 2.44, -277.6, 1.27),
    "CaCO3":    ({"Ca": 1, "C": 1, "O": 3}, "s", 0.82, -1206.9, None),
    "CaO":      ({"Ca": 1, "O": 1}, "s", 0.75, -635.1, None),
    "Ca(OH)2":  ({"Ca": 1, "O": 2, "H": 2}, "s", 1.19, -985.2, None),
    "CO2":      ({"C": 1, "O": 2}, "g", 0.84, -393.51, None),
    "CO":       ({"C": 1, "O": 1}, "g", 1.04, -110.53, None),
    "O2":       ({"O": 2}, "g", 0.92, 0.0, None),
    "Fe":       ({"Fe": 1}, "s", 0.45, 0.0, None),
    "FeOOH":    ({"Fe": 1, "O": 2, "H": 1}, "s", 0.9, -559.0, None),  # goethite
    "C":        ({"C": 1}, "s", 0.71, 0.0, None),
    # no heat of formation: dissolved forms and fuels (rows with these skip the
    # Hess check on purpose; their heat is measured data)
    "NaCl_aq":  ({"Na": 1, "Cl": 1}, "l", 3.3, None, None),
    "sucrose_aq": ({"C": 12, "H": 22, "O": 11}, "l", 2.8, None, None),
    "ash":      ({}, "s", 0.8, None, None),
    "wood":     ({}, "s", 1.7, None, None),
    "flesh":    ({}, "s", 3.5, None, None),
    "leaf":     ({}, "s", 2.4, None, None),
    "char":     ({}, "s", 1.0, None, None),
    "rope":     ({}, "s", 1.4, None, None),
}


def _molar(formula):
    return sum(ATOMIC[e] * n for e, n in formula.items())


M = {name: _molar(row[0]) for name, row in _ROWS.items()}   # g/mol (c)
PHASE = {name: row[1] for name, row in _ROWS.items()}
CP = {name: row[2] for name, row in _ROWS.items()}          # J/(g K)
DHF = {name: row[3] for name, row in _ROWS.items()}         # kJ/mol
V_APP = {name: row[4] for name, row in _ROWS.items() if row[4] is not None}  # ml/g
GASES = tuple(n for n, p in PHASE.items() if p == "g")

# T1's species names for the same things (documentation and the cross test).
ALIAS = {"H2O": "water", "NaCl": "salt", "sucrose": "sugar", "ethanol": "ethanol",
         "CaCO3": "calcite", "CaO": "quicklime", "Ca(OH)2": "slaked_lime",
         "FeOOH": "iron_oxide", "C": "graphite", "Fe": "iron"}

# ---- fresh-air background (g/m3 at the reference air density) --------------
# Only species that fresh air really carries. H2O is set by w.air_rh, so it is not
# here. O2 is the existing 0.28 g/L; it is left out because the O2 field is the
# repo's own.
BACKGROUND = {"CO2": FRESH_CO2_G_M3}

# ---- solutes in water -----------------------------------------------------
# D25: diffusivity in water at 25 C, m2/s (u). E_D: activation, J/mol (u):
# about 18 kJ/mol for small solutes, factor 1.6 from 20 to 40 C (c).
D25 = {"NaCl": 1.61e-9, "sucrose": 5.2e-10, "glucose": 6.7e-10,
       "ethanol": 1.2e-9, "H2O": 2.3e-9, "tannin": 5e-10}
E_D = {"NaCl": 18000.0, "sucrose": 18000.0, "glucose": 18000.0,
       "ethanol": 18000.0, "H2O": 18000.0, "tannin": 18000.0}
# Tortuosity factor (D_eff = D_water * tau) in tissue hosts (u); soil and stone use
# Millington-Quirk from the pore volume instead, see transport.host_D.
TORTUOSITY = {
    ("NaCl", "meat"): 2e-10 / 1.61e-9,    # 1-3e-10 m2/s in meat, fish, hide
    ("NaCl", "hide"): 1e-10 / 1.61e-9,
    ("sucrose", "meat"): 0.2,
    ("tannin", "hide"): 3e-11 / 5e-10,
}
# Supersaturation: crystals appear when c passes S_CRIT times saturation (u).
S_CRIT = {"NaCl": 1.01, "sucrose": 1.15}
# Heat of solution, J per g solute dissolved; + takes heat from the cell (u).
H_SOL = {"NaCl": 66.0, "sucrose": 17.0, "NaOH": -1113.0, "CaCl2": -746.0,
         "KNO3": 345.0}

# ---- vapour-liquid data (u) -----------------------------------------------
# Antoine for ethanol, mmHg and degC: log10 P = A - B/(C + T), 20-93 C.
ANTOINE = {"ethanol": (8.20417, 1642.89, 230.300)}
# Van Laar constants for ethanol (1) and water (2): T4's refit that puts the
# azeotrope at 0.894 mole fraction and 78.15 C (Perry's 1.6798/0.9227 gives 0.912).
VAN_LAAR = {("ethanol", "H2O"): (1.685, 0.95)}
# 1 atm VLE table (x, y, T C) for ethanol-water (u).
VLE_TABLE = ((0.019, 0.170, 95.5), (0.0721, 0.389, 89.0), (0.1238, 0.470, 85.3),
             (0.2608, 0.558, 82.3), (0.3965, 0.612, 80.7), (0.5732, 0.684, 79.3),
             (0.7472, 0.782, 78.41), (0.8943, 0.894, 78.15))

# ---- solubility (g solute per 100 g water, u CRC) -------------------------
SOLUBILITY_T = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)
SOLUBILITY = {"NaCl": (35.7, 35.9, 36.4, 37.1, 38.0, 39.2),
              "sucrose": (179.0, 203.9, 238.1, 287.3, 362.1, 487.2)}
# Below 0.1 C the stable NaCl solid is hydrohalite NaCl.2H2O: solubility is a line
# from 30.4 g/100 g at -21.1 C to 35.7 at 0.1 C (u, T4's mixsol owns it).
HYDROHALITE = ((-21.1, 30.4), (0.1, 35.7))

# ---- NaCl and sucrose water activity (u) ----------------------------------
# Osmotic coefficient of NaCl (Robinson & Stokes), by molality.
NACL_PHI = ((0.0, 0.9325), (0.1, 0.9325), (0.5, 0.921), (1.0, 0.936), (2.0, 0.983),
            (3.0, 1.045), (4.0, 1.116), (5.0, 1.196), (6.0, 1.271), (6.14, 1.28))
NORRISH_K = {"sucrose": -6.47}      # ln a_w = ln x_w + K x_s^2

# ---- wood moisture (u: Wood Handbook, Siau, Stamm) ------------------------
# GAB parameters fitted (c) to 30 % RH 6.3, 50 % 9.2, 65 % 12.0, 80 % 16.0,
# 90 % 21.0, 95 % 24.0 (percent moisture, dry basis).
GAB_WOOD = (0.065, 9.0, 0.78)       # wm, C, K
WOOD_DRY_DENSITY = 500.0            # kg/m3
WOOD_D_APP = 2e-10                  # m2/s across the grain, 20 C, 12 %
WOOD_E_D = 38000.0                  # J/mol

# ---- burning: elemental composition by mass (u) ---------------------------
# Each row: C, H, O, water, ash fractions summing to 1. Chosen so the O2 demand by
# mass balance is within 2 % of the repo's O2_RATIO (wood 1.33, flesh 0.6, leaf
# 1.25, char 2.6, rope 1.26). `char` = share of the carbon that leaves as soot
# instead of CO2 (the repo's visible-smoke proxy stays as it is in SOOT).
BURN_COMP = {
    "wood":  {"C": 0.50, "H": 0.06, "O": 0.43, "water": 0.0, "ash": 0.01, "char": 0.05},
    "flesh": {"C": 0.195, "H": 0.03, "O": 0.105, "water": 0.60, "ash": 0.07, "char": 0.10},
    "leaf":  {"C": 0.4985, "H": 0.06, "O": 0.4215, "water": 0.0, "ash": 0.02, "char": 0.10},
    "char":  {"C": 0.92, "H": 0.02, "O": 0.02, "water": 0.0, "ash": 0.04, "char": 0.0},
    "rope":  {"C": 0.481, "H": 0.062, "O": 0.449, "water": 0.0, "ash": 0.008,
              "char": 0.05},
}
# the repo's material ids for those names (materials.py: WOOD 1, FLESH 4, LEAF 7,
# CHAR 8, ROPE 11), so burn_row(mat) takes either.
BURN_NAME_OF_MAT = {1: "wood", 4: "flesh", 7: "leaf", 8: "char", 11: "rope"}
BURN_HEAT = {"wood": 16000.0, "flesh": 8000.0, "leaf": 15000.0, "char": 30000.0,
             "rope": 16500.0}       # J/g, copied from materials.SOLID (cross-tested)
O2_RATIO = {"wood": 1.33, "flesh": 0.6, "leaf": 1.25, "char": 2.6, "rope": 1.26}


def names():
    """All species keys, sorted."""
    return sorted(_ROWS)


def formula(name):
    return dict(_ROWS[name][0])


if __name__ == "__main__":
    print("%-9s %-3s %8s %7s %9s" % ("species", "ph", "M g/mol", "c", "dHf kJ/mol"))
    for n in names():
        h = "%9.1f" % DHF[n] if DHF[n] is not None else "%9s" % "none"
        print("%-9s %-3s %8.3f %7.2f %s" % (n, PHASE[n], M[n], CP[n], h))
    print("every number above is (u) unless a comment says (c)")
