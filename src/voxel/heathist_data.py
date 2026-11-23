"""HEAT HISTORY: the numbers (wave 15, lane 1e, smithing stage 1). DATA only.

Tags: (v) a source that was read, (k) from memory of the handbooks, (u) unverified: a fit or a
guess that a later lane must check against a published curve. No law lives here.

Stage 1 covers plain carbon steels, manganese steels and wrought iron (owner Q6).
Cr, Ni, Mo are in the formulas with zero as their default, so a later alloy is a row, not code.
"""
R_GAS = 8.314462618            # J/mol/K

# --- critical temperatures -------------------------------------------------------------
# Andrews, J.I.S.I. 203 (1965) 721-727, as printed in the Cambridge MAP page and in the
# handbook text. (v) for the form, (k) for the carbon term of Ac3.
AC1 = dict(base=723.0, Mn=-10.7, Ni=-16.9, Si=29.1, Cr=16.9)
AC3 = dict(base=910.0, sqrtC=-203.0, Ni=-15.2, Si=44.7, Mo=31.5)

# Martensite start. Andrews linear (valid below 0.6 percent C, (v) Cambridge MAP page);
# above 0.6 percent Steven and Haynes (v). The CRITIC check: the carbon coefficient is 474
# (the MAP page and most sources); some print 476. The spread between models is about 33 K
# one sigma (v), so MS_SIGMA_K is carried and used by anything that wants a band.
MS_ANDREWS = dict(base=539.0, C=-423.0, Mn=-30.4, Cr=-12.1, Ni=-17.7, Mo=-7.5)
MS_STEVEN_HAYNES = dict(base=561.0, C=-474.0, Mn=-33.0, Cr=-17.0, Ni=-17.0, Mo=-21.0)
MS_SH_C_VARIANT = -476.0       # printed by some sources; 2 K at 1 percent C
C_SWITCH = 0.6                 # wt percent C where the two Ms forms hand over (v)
MS_SIGMA_K = 33.0              # (v)

# Koistinen-Marburger 1959, Acta Metall. 7, 59-60: fM = 1 - exp(-b (Ms - T)), b about 0.011 /K (v).
KM_B = 0.011

# --- austenitising (u) --------------------------------------------------------------------
# Carbide dissolution is diffusion limited. Activation energy of carbon in austenite about
# 148 kJ/mol (the partition.py row, (u)); tau_a is 15 s at 800 C (u: a thin bar must soak for
# some seconds; a smith waits for the colour to even out).
QA_J = 148.0e3
TAU_A_REF_S = 15.0
TAU_A_REF_C = 800.0

# --- austenite grain growth, d^n - d0^n = K t, K = K0 exp(-Q/RT) (u) --------------------------
# Form (v: several fits, n 2 to 6, Q 250-570 kJ/mol). The plain-carbon row is FITTED here to a
# typical text value: about 60 um after one hour at 950 C and about 250 um after one hour at
# 1100 C, d0 = 10 um, n = 2 (u). That gives Q about 264 kJ/mol (computed in heathist).
GRAIN_N = 2.0
GRAIN_D0_UM = 10.0
GRAIN_FIT = ((950.0, 60.0), (1100.0, 250.0))     # (T C, d um after 3600 s)
GRAIN_REF_UM = 20.0            # the grain size the C-curve and ductility rows are quoted at (u)
HARDENABILITY_P = 1.0          # tau scales as (d/d_ref)^p (u)

# --- diffusional products, Scheil additivity (Kirkaldy-type nose; constants (u)) -------------
# Nose of the eutectoid (0.8 C, 0.5 Mn) C-curve: about 550 C (k, textbook TTT of a 1080 steel).
# The textbook START of transformation at the nose is about 1 s; Scheil here sums to COMPLETION, so
# NOSE_TAU_S is fitted to the critical cooling rate instead (about 150 K/s for 0.8 C through the
# nose, k): 8 s gives 0.4 martensite at 100 K/s and the full 0.8 at 300 K/s (u). The activation
# energy is DERIVED in heathist from the nose position.
NOSE_T_C = 550.0
NOSE_TAU_S = 8.0
NOSE_REF_C = 0.8
NOSE_REF_MN = 0.5
NOSE_M = 3.0                   # driving-force exponent (u)
TAU_C_SLOPE = 3.0              # tau_nose = NOSE_TAU_S exp(3 (c - 0.8)): low carbon is less hardenable (u)
MN_GROSSMANN = 3.33            # hardenability factor (1 + 3.33 Mn), Grossmann 1942 (k)

# --- hardness, Vickers (all (u) unless marked) ----------------------------------------------
HV_BASE = (95.0, 135.0)        # annealed ferrite-pearlite HV = 95 + 135 c (k, fit to the handbook 0.2-0.8 range)
HV_FINE_RATIO = 1.9            # fine pearlite or bainite formed at the nose, over the coarse base (u)
HV_RETAINED = 250.0            # retained austenite (u)
# Fresh martensite depends on carbon only (v: Grange, Hribal and Porter 1977). 210 at 0.005 and
# about 500 at 0.2 (v, a Nippon Steel report); 620, 750, 830 at 0.4, 0.6, 0.8 (k, (u)): the
# 0.2 value looks high against 400-450 HV in some handbooks, a lane must read the ASM table.
HV_M_C = (0.0, 0.005, 0.2, 0.4, 0.6, 0.8, 1.2)
HV_M = (205.0, 210.0, 500.0, 620.0, 750.0, 830.0, 880.0)

# --- tempering: Hollomon-Jaffe Hp = T (C + log10 t) / 1000, T kelvin, t hours (v: Wikipedia and
# TWI). C = 19.5 for 0.25-0.4 C, 15 for 0.9-1.2 C, linear between (v). g(P) from 1 (nothing
# happened) to 0 (soft): the table is a fit to Grange-Hribal-Porter style curves (u).
CHJ_LOW, CHJ_HIGH = 19.5, 15.0
CHJ_C_LOW, CHJ_C_HIGH = 0.4, 0.9
G_P = (0.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 20.0)
G_G = (1.0, 1.0, 0.90, 0.65, 0.35, 0.18, 0.06, 0.0)

# --- latent heat (u): pearlite to austenite on heating. The range of the pure-iron
# alpha-gamma heat (about 16 J/g) to the eutectoid steel (about 70 J/g, the order the
# calorimetry gives); martensite sits a third of the way up (u).
L_AUST_J_G = (16.0, 70.0)      # at c = 0 and c = 0.77
L_MART_SHARE = 0.3

# --- toughness readouts (u) --------------------------------------------------------------
DUCT_Q = 1.0                   # ductility and impact toughness scale (HV_base/HV)^q
GRAIN_HP = 0.5                 # Hall-Petch: toughness factor (d_ref/d)^0.5 (k)

# --- steel rows by name: composition in weight percent. A thing that names none has none. -----
ROWS = {
    "wrought_iron": dict(c=0.02, mn=0.05),     # slag fibres, almost no carbon (k)
    "mild_steel": dict(c=0.20, mn=0.50),
    "medium_steel": dict(c=0.40, mn=0.70),
    "tool_steel": dict(c=0.80, mn=0.50),       # eutectoid plain carbon: the file and the blade
}

# --- the surface of a quench: effective heat-transfer coefficient h(T_surface), W/m2/K (u).
# Orders from the handbooks (Totten, Handbook of Quenchants): a quench has a vapour blanket at the
# hot end, a nucleate-boiling peak near 250-450 C (10^4 and more), and a slow convective tail.
# These rows are for an AGITATED bath (the smith swirls the bar), which breaks the blanket early;
# oil has a lower peak, still air is convection (radiation is added by the slab). Fit: a 20 mm
# 0.8 C bar in this water row reaches martensite at the skin and pearlite at the core (the
# Grossmann H of about 1 to 2 per inch is h of about 3000 to 6000 on average).
H_QUENCH = dict(
    brine=((20.0, 3000.0), (100.0, 8000.0), (250.0, 40000.0), (450.0, 40000.0), (600.0, 12000.0), (900.0, 8000.0)),
    water=((20.0, 1000.0), (100.0, 3000.0), (250.0, 20000.0), (450.0, 20000.0), (600.0, 6000.0), (900.0, 4000.0)),
    oil=((20.0, 400.0), (200.0, 1500.0), (400.0, 1500.0), (600.0, 1100.0), (900.0, 800.0)),
    air=((20.0, 12.0), (900.0, 25.0)),
)
