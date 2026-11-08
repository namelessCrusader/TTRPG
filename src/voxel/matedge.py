"""THE EDGE AS STATE (wave 13, T9): geometry, wear, rolling, honing, filings.

Today `edge` is one number per cell, the contact area in m2, declared once (an
axe is 2e-4) and never changed. Here an edge is a wedge of included angle
theta with a rounded tip of radius r over a length L. Only r changes in use;
L and theta belong to the tool. The old `edge` array stays as the AREA view, so
every old reader keeps working.

Stand-alone: numpy and math only. All lengths in metres, areas in m2, work in
joules, hardness in pascals, volumes in m3.

  contact width   w_c = 2 r + 2 d_c tan(theta / 2)      area A = L w_c
  d_c = 3.5 mm is the depth at which skin gives: a calibration constant (u).
  A felling axe (theta 30 deg, r 0.1 mm, L 0.1 m) has A = 2.08e-4, the 2e-4
  the scenes declare.

  WEAR (Archard 1953): V = K W / (mu_flank H_edge). The lost cross-section is
  V / L and the tip grows as r' = sqrt(r^2 + V / L). K depends on the
  hardness ratio (Hutchings and Shipway: abrasion turns severe when the
  abrasive is harder than about 1.2 times the surface):
      K = K_soft + (K_hard - K_soft) smoothstep((H_target / H_edge - 0.8) / 0.4)
  with K_soft = 1e-6 and K_hard = 1e-2 (both (u)).

  ROLLING (a hard blow): peak force F = e / delta_s with the stopping distance
  delta_s = clip(1 mm sqrt(H_edge / H_target), 0.2 mm, 50 mm) (u). The edge
  cannot carry a pressure above its hardness, so r' = max(r, F / (2 L H_edge)).
  This is why an axe chops wood all day and not stone.

  HONING: V = K(H_abrasive / H_edge) F s / H_edge, r' = sqrt(max(r^2 - V/L,
  r_floor^2)) with r_floor = 0.25 x the grit diameter (u). A stone softer than
  the edge does next to nothing (K_soft).

  FILINGS: the volume the edge loses is owed as mass. It leaves the tool as
  whole-gram chunks, so tool + filings is constant (see `filings_step`).
"""
import math
from dataclasses import dataclass

import numpy as np

D_C = 3.5e-3                 # m (u): depth at which skin gives; the calibration of the contact width
THETA_DEFAULT = math.radians(30.0)
R_DEFAULT = 1.0e-4           # m: a new axe edge, 0.1 mm
K_SOFT = 1.0e-6              # (u) Archard coefficient, soft abrasive
K_HARD = 1.0e-2              # (u) Archard coefficient, hard abrasive
H_RATIO_LO, H_RATIO_SPAN = 0.8, 0.4
STOP_M = 1.0e-3              # (u) stopping distance scale
STOP_LO, STOP_HI = 0.2e-3, 50.0e-3
GRIT_FLOOR = 0.25            # (u) r_floor = this x the grit diameter
SKIN_J_M2_OLD = 1.0e5        # WOUND["skin_Jm2"] today: skin opens under energy per AREA
C_SKIN_J_M2 = 1.04e6         # the same law on the TIP area L 2r (rescaled so a pristine edge opens at today's energy)
FILING_CHUNK_G = 1.0         # filings leave the tool in whole grams
REQUANT = 0.01               # the `edge` array is rewritten when r moved by 1 % or more


@dataclass(frozen=True)
class EdgeGeom:
    """An edge: length L (m), included angle theta (rad), tip radius r (m)."""
    L_m: float
    theta_rad: float
    r_m: float


def contact_width(r, theta, d_c=D_C):
    """w_c = 2 r + 2 d_c tan(theta / 2)."""
    return 2.0 * np.asarray(r, float) + 2.0 * d_c * np.tan(np.asarray(theta, float) / 2.0)


def area(geom, d_c=D_C):
    """The edge area L w_c (what the old `edge` array holds)."""
    return geom.L_m * float(contact_width(geom.r_m, geom.theta_rad, d_c))


def geom_from_area(A, theta=THETA_DEFAULT, r0=R_DEFAULT, d_c=D_C):
    """A declared edge area becomes (L, theta, r0). edge = 2e-4 gives
    L = 0.096 m and r = 100 um. `theta` comes from the thing record when it has
    one (an arrow head, a knife), else 30 degrees."""
    w = float(contact_width(r0, theta, d_c))
    return EdgeGeom(float(A) / w, float(theta), float(r0))


def area_after_wear(A, dr, L):
    """A' = A + 2 L dr (exact: only the 2 r term moves)."""
    return A + 2.0 * L * dr


def tip_area(geom):
    """L 2 r: the area under which skin opens (the wound split, step W8)."""
    return geom.L_m * 2.0 * geom.r_m


def skin_constant(theta=THETA_DEFAULT, r0=R_DEFAULT, d_c=D_C):
    """The tip-area skin constant that makes a pristine edge open skin at
    today's energy: 1e5 w_c / (2 r0). 1.038e6 for the axe; C_SKIN_J_M2 is the
    spec's rounded 1.04e6."""
    return SKIN_J_M2_OLD * float(contact_width(r0, theta, d_c)) / (2.0 * r0)


def skin_open_J(geom, c_skin=C_SKIN_J_M2):
    """Energy a blow needs to open skin: c_skin x tip area. 20.8 J for the new
    axe, 97 J for the rolled one."""
    return c_skin * tip_area(geom)


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def k_archard(h_ratio):
    """K from the hardness ratio (abrasive or target over edge): K_soft well
    below 0.8, K_hard above 1.2, the switch in between (half way at 1.0)."""
    s = _smoothstep((np.asarray(h_ratio, float) - H_RATIO_LO) / H_RATIO_SPAN)
    return K_SOFT + (K_HARD - K_SOFT) * s


def wear(geom, work_J, H_edge, H_target, mu_flank=0.4):
    """Archard wear of the edge from `work_J` joules done against a target.
    Returns (r_new, volume_lost_m3). Monotone in work, falls as 1/H_edge."""
    K = k_archard(np.asarray(H_target, float) / np.asarray(H_edge, float))
    V = K * np.asarray(work_J, float) / (mu_flank * np.asarray(H_edge, float))
    r_new = np.sqrt(geom.r_m ** 2 + V / geom.L_m)
    return _s(r_new), _s(V)


def stopping_distance(H_edge, H_target):
    """delta_s = clip(1 mm sqrt(H_edge / H_target), 0.2 mm, 50 mm)."""
    return np.clip(STOP_M * np.sqrt(np.asarray(H_edge, float) / np.asarray(H_target, float)),
                   STOP_LO, STOP_HI)


def blow_force(blow_J, H_edge, H_target):
    """Peak force F = e / delta_s."""
    return np.asarray(blow_J, float) / stopping_distance(H_edge, H_target)


def roll(geom, blow_J, H_edge, H_target):
    """The tip radius after a blow of `blow_J`: the edge rolls over until its
    pressure falls to its hardness, r' = max(r, F / (2 L H_edge)). A soft
    target (wood, flesh) never rolls an iron edge."""
    F = blow_force(blow_J, H_edge, H_target)
    return _s(np.maximum(geom.r_m, F / (2.0 * geom.L_m * np.asarray(H_edge, float))))


def hone(geom, strokes, force_N, stroke_m, H_edge, H_abrasive, grit_m):
    """Hone with an abrasive: returns (r_new, volume_removed_m3).
    The edge cannot go below r_floor = 0.25 grit, and an edge already finer
    than the floor is left alone (honing never dulls it; the volume removed
    stays non-negative so the filings ledger closes)."""
    K = k_archard(np.asarray(H_abrasive, float) / np.asarray(H_edge, float))
    V = K * np.asarray(force_N, float) * np.asarray(stroke_m, float) * np.asarray(strokes, float) \
        / np.asarray(H_edge, float)
    r_floor = GRIT_FLOOR * grit_m
    r2 = geom.r_m ** 2 - V / geom.L_m
    r_new = np.minimum(geom.r_m, np.sqrt(np.maximum(r2, r_floor ** 2)))
    vol = geom.L_m * (geom.r_m ** 2 - r_new ** 2)           # what actually left (clamped at the floor)
    return _s(r_new), _s(vol)


def needs_write(r_written, r_now, tol=REQUANT):
    """The `edge` array is rewritten only when r moved by 1 % or more since the
    last write (the thing keeps the exact r): tiny wear never changes the array."""
    return abs(r_now - r_written) >= tol * r_written


def filings_step(owed_g, lost_g, chunk_g=FILING_CHUNK_G):
    """Book `lost_g` grams worn off. Returns (owed_g, chunks): the grams still
    owed (below one chunk) and the number of whole chunks that leave the tool
    now as rubble of the tool's material. Tool mass drops by exactly
    chunks x chunk_g, so tool + filings (+ what is owed, still in the tool) is
    constant."""
    owed = float(owed_g) + float(lost_g)
    n = int(math.floor(owed / chunk_g))
    return owed - n * chunk_g, n


def volume_to_g(volume_m3, rho_kg_m3):
    """Grams in a volume of material."""
    return float(volume_m3) * float(rho_kg_m3) * 1000.0


def _s(x):
    """A numpy 0-d result as a float; arrays unchanged."""
    return float(x) if np.ndim(x) == 0 else x
