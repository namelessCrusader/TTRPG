"""FASTENERS AND JOINT CAPACITIES (wave 13, T11): pure functions, SI in, SI out.

What holds a door to its frame, a strike plate to a jamb, a bar to its cleats.
Every function takes plain numbers (no World, no cells, no material ids) so
that T2's joinery can import it for the screw and dowel rows and the
mechanism records can import it for the same rows: one copy of each formula.

Units: lengths in metres, forces in newtons, stresses in pascals, unless the
name says otherwise. The two withdrawal formulas are written the way the Wood
Handbook writes them (millimetres in, newtons out); the wrappers here take
metres and convert.

NOT HERE: nails (they stay in joinery.py with T2's nail row 54.12 G^2.5 D),
and nothing that knows a material id. Wood density G (specific gravity, 0.45
for pine) and the embedment strength of the wood come in as numbers.

Sources: USDA Wood Handbook ch. 8 (withdrawal of screws, recalled, (u));
AWC NDS ch. 11-12 (withdrawal allowable 2850 G^2 D per inch of thread,
embedment 16600 G^1.84 psi, yield mode IV, recalled, (u)), except the reduction
term Rd, which is read from NDS-2018 Table 12.3.1B itself (see RD_FULL below);
Shigley for stiffness; the strengths across the grain are the bond table's (3 MPa).
"""
import math

import numpy as np

PSI = 6894.757293168           # Pa per psi
LBF = 4.4482216152605          # N per pound-force
INCH = 0.0254                  # m

# Wood Handbook ultimate withdrawal, N with D and L in mm: P = 108.25 G^2 D L
WITHDRAW_SCREW = 108.25        # (u): recalled
WITHDRAW_NDS = 2850.0          # lb per inch of thread per inch of diameter, per G^2 (allowable)
NDS_SAFETY = 5.0               # ultimate over allowable, the usual order (u)
# The yield-mode reduction term Rd, ONE copy (the function `dowel_Rd` below; record_data and joinery
# import it from here). ANSI/AWC NDS-2018 Table 12.3.1B, p. 84, read in the AWC free view-only copy
# of the 2018 NDS (Chapter 12) on 2026-10-30. This is SOURCED, no longer recalled:
#   0.25 in <= D <= 1 in:  Im, Is  4 K_theta;  II  3.6 K_theta;  IIIm, IIIs, IV  3.2 K_theta
#   D < 0.25 in:           every mode  K_D
#   K_theta = 1 + 0.25 (theta/90), theta = the largest angle between the load and the grain
#   K_D = 2.2 for D <= 0.17 in;  K_D = 10 D + 0.5 for 0.17 in < D < 0.25 in   (D in inches)
# A SMALL screw gets the SMALL term (so a higher yield value) and the term steps up to 3.2 at 1/4 in.
# This file had the two ends swapped (3.2 under 1/4 in, 2.2 above) until 2026-10-30 (decision N6).
RD_FULL = 3.2                  # modes IIIm, IIIs and IV, D from 1/4 in up, load along the grain
RD_KD_SMALL = 2.2              # K_D for D up to 0.17 in
KD_D_LO, KD_D_HI = 0.17 * INCH, 0.25 * INCH
LATERAL_ULT_OVER_YIELD = 2.0   # (u): lateral ultimate as a multiple of the yield value
FYB_SCREW = 600e6              # Pa, bending yield of a wood screw (u)
SHEAR_OVER_YIELD = 0.6         # shear yield as a fraction of tensile yield (von Mises-ish)
PUNCH_TAU = 5e6                # Pa, across the plies of a thin panel (u)


# ── embedment and withdrawal ─────────────────────────────────────────────────
def wood_embedment(G, D):
    """Dowel-bearing (embedment) strength of wood in Pa, for a fastener of
    diameter D (m) in wood of specific gravity G (no load-to-grain angle).

    NDS 12.3.3: D under 1/4 in (a screw or small nail) Fe = 16600 G^1.84 psi;
    bolts of 1/4 in and over Fe = 11200 G psi. 0.45 pine at 4.83 mm: 26 MPa."""
    if D < 0.25 * INCH:
        return 16600.0 * G ** 1.84 * PSI
    return 11200.0 * G * PSI


def screw_withdrawal(G, D, L):
    """Ultimate withdrawal of a wood screw: P = 108.25 G^2 D L (N, D and L in
    mm). `D` and `L` are in METRES here; L is the threaded length in the wood.
    A #10 (4.83 mm) with 19 mm of thread in pine G 0.45: 2.0 kN."""
    return WITHDRAW_SCREW * G * G * (D * 1000.0) * (L * 1000.0)


def screw_withdrawal_allowable(G, D, L):
    """The NDS allowable withdrawal, N: 2850 G^2 D lb per inch of thread, D in
    inches, times the thread length in inches. An ultimate is about
    NDS_SAFETY times it."""
    return WITHDRAW_NDS * G * G * (D / INCH) * (L / INCH) * LBF


# ── lateral load on a dowel: the yield modes ─────────────────────────────────
def dowel_bearing(Fe, D, l):
    """Mode I (bearing): the wood crushes under the fastener, F = Fe D l."""
    return Fe * D * l


def dowel_Rd(D, theta_deg=0.0):
    """The reduction term Rd of yield modes IIIm, IIIs and IV for a dowel-type
    fastener of diameter D (metres), NDS-2018 Table 12.3.1B (the numbers and
    their source are at the top of this file). D under 1/4 in: K_D, 2.2 up to
    0.17 in and 10 D + 0.5 (D in inches) from there, so 3.0 at 1/4 in; D from
    1/4 in: 3.2 K_theta. `theta_deg` (0 to 90) is the angle between the load
    and the grain; only a fastener of 1/4 in or more uses it (1.25 at 90).

    Scalar in gives a 0-d array (float() it), an array gives an array. The
    caller gives the diameter its equations use. Footnote 1 of the table (a
    threaded fastener of nominal 1/4 in or more with a root under 1/4 in takes
    K_D K_theta) is not modelled."""
    if not 0.0 <= theta_deg <= 90.0:
        raise ValueError("theta_deg is the angle between load and grain, 0 to 90")
    D = np.asarray(D, dtype=np.float64)
    kd = np.where(D <= KD_D_LO, RD_KD_SMALL, 10.0 * (D / INCH) + 0.5)
    return np.where(D >= KD_D_HI, RD_FULL * (1.0 + 0.25 * theta_deg / 90.0), kd)


def dowel_mode_iv(Fe, Fyb, D, Rd=None, Re=0.0):
    """Mode IV (the fastener bends in two hinges, the wood under it crushes):
    Z = D^2 / Rd * sqrt(2 Fe Fyb / (3 (1 + Re)))  (NDS Table 12.3.1A, in
    single shear). Re = Fe_main / Fe_side: 0 for a steel plate as the side
    member (the strike plate on a jamb), 1 for wood on wood. `Rd` is the
    NDS term for the diameter (`dowel_Rd`) unless the caller gives one.

    A yield VALUE, not an ultimate: see `dowel_lateral_ultimate`."""
    if Rd is None:
        Rd = float(dowel_Rd(D))
    return (D * D / Rd) * math.sqrt(2.0 * Fe * Fyb / (3.0 * (1.0 + Re)))


def dowel_lateral_yield(G, D, l, Fyb=FYB_SCREW, Rd=None, Re=0.0):
    """The lateral yield value of one screw in wood of density G: the smaller
    of full bearing (Fe D l) and mode IV. Mode IV caps it once the screw is
    long enough, so a longer screw (or a stud behind the jamb) raises nothing
    past that point."""
    Fe = wood_embedment(G, D)
    return min(dowel_bearing(Fe, D, l), dowel_mode_iv(Fe, Fyb, D, Rd, Re))


def dowel_lateral_ultimate(G, D, l, n=1, **kw):
    """n screws in lateral shear, ULTIMATE: LATERAL_ULT_OVER_YIELD x the yield
    value each. Two #10 strike screws in pine: about 4 kN (3.99 kN with the
    NDS Rd of 2.40 at 0.19 in; it was 3.0 kN with the two ends of Rd swapped)."""
    return n * LATERAL_ULT_OVER_YIELD * dowel_lateral_yield(G, D, l, **kw)


def lateral_mode(G, D, l, Fyb=FYB_SCREW, Rd=None, Re=0.0):
    """Which mode governs, "bearing" or "iv" (for a report or a test)."""
    Fe = wood_embedment(G, D)
    return "bearing" if dowel_bearing(Fe, D, l) <= dowel_mode_iv(
        Fe, Fyb, D, Rd, Re) else "iv"


# ── splitting, crushing, punching, bars ──────────────────────────────────────
def split_across_grain(sigma_t_perp, l_s, t_e):
    """A jamb splitting behind the keeper: F = sigma_t,perp l_s t_e. The
    strength across the grain is the bond table's 3 MPa; l_s the length of
    wood the screws share (0.08 m, (u)) and t_e the wood left behind the
    screws (0.012 m, (u); 0.03 m with a stud behind). 2.9 kN, and 7.2 kN
    with the stud."""
    return sigma_t_perp * l_s * t_e


def bearing_crush(sigma_perp, area):
    """The leaf edge crushing across the grain on the stop: strength x area."""
    return sigma_perp * area


def punching(tau_u, perimeter, t):
    """A thin panel punched through: F = tau_u x perimeter x thickness (of the
    PLIES that carry the shear: a hollow-core skin is a few mm, not the door's
    thickness). A foot (perimeter 0.5 m) on a 3 mm skin and 5 MPa: 7.5 kN."""
    return tau_u * perimeter * t


def shear_round(sigma_y, d, n=1):
    """A round pin or bolt of diameter d in single shear: 0.6 sigma_y pi d^2/4
    each. Steel 250 MPa, 12 mm: 17 kN."""
    return n * SHEAR_OVER_YIELD * sigma_y * math.pi * d * d / 4.0


def tension_round(sigma_y, d, n=1):
    """A round bar in tension, at its yield."""
    return n * sigma_y * math.pi * d * d / 4.0


def bar_section_modulus(b, d):
    """Z = b d^2 / 6 of a section b wide and d DEEP in the direction of the
    load (the depth is what counts, squared)."""
    return b * d * d / 6.0


def bar_bending(sigma_y, b, d, span):
    """A bar simply supported at both cleats, load at mid-span:
    P = 4 M_y / S with M_y = sigma_y Z. 38 x 89 mm pine ON EDGE (b = 0.038,
    d = 0.089) over 1.2 m at 40 MPa: 6.7 kN. On the flat (b = 0.089,
    d = 0.038) about 0.43 of that."""
    return 4.0 * sigma_y * bar_section_modulus(b, d) / span


def bar_holds(beam, cleat_screws, leaf_bearing):
    """A bar holds the smallest of its beam, its cleat screws and the leaf
    edge bearing on it. Returns (force, which) with which in ("beam",
    "cleats", "leaf")."""
    c = (("beam", beam), ("cleats", cleat_screws), ("leaf", leaf_bearing))
    w = min(c, key=lambda t: t[1])
    return w[1], w[0]


# ── stiffness, series and referred ───────────────────────────────────────────
def axial_k(E, A, L):
    """E A / L."""
    return E * A / L


def cantilever_k(E, I, L):
    """3 E I / L^3."""
    return 3.0 * E * I / L ** 3


def rect_I(b, t):
    """Second moment of a b-wide, t-deep rectangle bending about the axis
    that makes t the depth: b t^3 / 12."""
    return b * t ** 3 / 12.0


def crush_k(E_perp, A, t):
    """Crushing across the grain over a thickness t: E_perp A / t."""
    return E_perp * A / t


def series_k(*ks):
    """1/k = sum of 1/k_i for links in a row. A broken link (k = 0) makes the
    path stiffness 0."""
    if not ks or any(k <= 0.0 for k in ks):
        return 0.0
    return 1.0 / sum(1.0 / k for k in ks)


def referred_k(k, l, a):
    """A stiffness k felt at distance l from the hinge axis, seen from a
    contact at distance a: k (l/a)^2."""
    return k * (l / a) ** 2


def scale_damage(x, D):
    """A capacity or a stiffness after damage D: x (1 - D) (Kachanov strain
    equivalence; the same factor T9's matdamage.mult_t and mult_E apply)."""
    return x * (1.0 - D)
