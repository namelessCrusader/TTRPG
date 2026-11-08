"""ICE AS A PHASE OF WATER (wave 13, T6): the ice row, the settle, the closed forms.

T6 owns the ice LAW and its numbers. T9's matrec ingests ICE_ROW (a plain dict) in
phase B; T1's mat 12 is the view of water's BOUND form; T4 supplies the liquidus
and Lfus(T) (here: a private stand-in, passed in as a `Liquidus`).

Why L_CONV20 and not 333.55. A cell's E is measured from 20 C and its heat capacity
CHANGES when water becomes ice (4.2 to 2.09 J/g K). One isolated gram at 0 C becoming
ice, with no heat exchange, gains 333.55 + 20 (cw - ci) = 375.75 J in E (c).
The true enthalpy is therefore H = E - L_CONV20 * m_ice, and the ledger subtracts
L_CONV20 times the ice mass. settle() conserves H exactly (E is rebuilt from H).

SETTLE is the enthalpy method with a liquidus. A cell holds liquid water m_w, ice m_i,
dissolved solute s and crystals x (S = s + x), other heat capacity C_other, energy E.
Total water Mw = m_w + m_i does not change here. The final liquid water W solves
"the temperature after freezing W' equals the liquidus of what is left", by
bisection on W (g is monotone). Salt is rejected into less liquid, so the freezing
point falls (brine rejection). When the liquid reaches the eutectic load b_e (the
liquidus meets the solubility line, hydrohalite below 0.1 C), T is pinned at T_e,
the liquid keeps b_e, and the rest freezes while crystals come out (the crystal is
kept as dry NaCl plus ice: mass exact). The eutectic is not stored, it falls out
of the two curves: -21.6 C at 23.3 wt % (c; real -21.1).

NUCLEATION is a number per situation, not a global 2 K. A cell freezes only if ice is
already in it (nucleation 0) or T is below the liquidus by dT_nuc. Melting relaxes
toward the liquidus with tau 600 s (u).

Ice in a PARTIAL cell is a gram count (ice_g: frost, a pond skin). The cell becomes
mat ICE only past ICE_FILL of its volume and reverts below ICE_UNFILL.
No snow, no sublimation, free ice does not float yet (wave 15). Said, not hidden.
"""
from typing import Callable, NamedTuple, Optional

import numpy as np

from . import _standin
from .matterdata import AMBIENT, CI, CW, LF0, L_CONV20, RHO_ICE

# ---- the row (plain dict; T9's matrec ingests it) ----------------------------
# Every number is (u). SPAN is generous: ice is carried by the water under it, there
# is no solid buoyancy until wave 15.
ICE_ROW = {
    "name": "ice",
    "rho": 917.0,              # g/L
    "c": 2.09,                 # J/(g K)
    "k": 2.2,                  # W/(m K)
    "emissivity": 0.97,
    "Lf": 333.55,              # J/g at 0 C
    "Tm": 0.0,                 # C, pure water
    "expansion": 1000.0 / 917.0,   # volume of 1 g of ice per volume of 1 g of water
    "transmit": 0.8,           # clear ice
    "tough": 0.3,
    "span": 40,
    "E_young": 9.3e9,          # Pa
    "poisson": 0.32,
    "tensile": 1.0e6,          # Pa
    "solid": (917.0, 2.09, 2.2, None, 0.0, 0.0),   # the SOLID tuple of materials.py
    "owner": "T6",
}
def ice_row():
    """A copy of ICE_ROW for T9's matrec to ingest (the spec's ice_row())."""
    return dict(ICE_ROW)


ICE_FILL = 0.5                 # volume share of ice at which a cell becomes mat ICE
ICE_UNFILL = 0.3               # below this it reverts (hysteresis, no flicker)
TAU_MELT = 600.0               # s, melting relaxes to the liquidus (u)
NUCLEATION = {                 # K below the liquidus before ice forms (data)
    "ice_in_cell": 0.0, "ice_in_neighbour": 0.0, "cold_wall": 0.0,
    "free_surface": 0.0, "isolated_bulk": 2.0}


def nucleation_k(ice_in_cell=False, ice_in_neighbour=False, cold_wall=False,
                 free_surface=False):
    """Supercooling (K) a cell can hold in this situation: 0 with ice in the cell or a
    neighbour, on a cold solid wall, or at the free surface under cold air
    (heterogeneous); 2 K only for isolated clean bulk water (u)."""
    hetero = (np.asarray(ice_in_cell) | np.asarray(ice_in_neighbour)
              | np.asarray(cold_wall) | np.asarray(free_surface))
    return np.where(hetero, NUCLEATION["ice_in_cell"], NUCLEATION["isolated_bulk"])


# ---- the liquidus -------------------------------------------------------------
class Liquidus(NamedTuple):
    T_of_b: Callable                     # freezing point (C) at load b (g solute per g water)
    eutectic: Optional[tuple]            # (T_e, b_e) or None
    solubility: Optional[Callable]       # b_sat(T), used only if all water is liquid


def _nacl_T(b):
    return _standin.freeze_point("NaCl", b)


def _sucrose_T(b):
    return _standin.freeze_point("sucrose", b)


NACL = Liquidus(_nacl_T, _standin.EUTECTIC_NACL,
                lambda T: _standin.solubility_load("NaCl", T))
SUCROSE = Liquidus(_sucrose_T, None, None)
PURE = Liquidus(lambda b: 0.0 * np.asarray(b, np.float64), None, None)


class Settled(NamedTuple):
    E: np.ndarray
    m_w: np.ndarray
    m_i: np.ndarray
    x: np.ndarray            # crystals
    T: np.ndarray
    frozen: np.ndarray       # signed grams frozen this call (negative = melted)


def enthalpy(E, m_i):
    """True enthalpy E - L_CONV20 m_ice: constant through freezing and melting."""
    return np.asarray(E, np.float64) - L_CONV20 * np.asarray(m_i, np.float64)


def _equilibrium(H, Mw, S, Co, liq):
    """Equilibrium (W liquid water, crystals x, T) of cells with enthalpy H. Arrays."""
    L = L_CONV20
    tiny = 1e-12

    def H_at(W, T):
        return (Co + CW * W + CI * (Mw - W)) * (T - AMBIENT) - L * (Mw - W)

    def T_after(W):
        return AMBIENT + (H + L * (Mw - W)) / (Co + CW * W + CI * (Mw - W))

    b_top = S / np.maximum(Mw, tiny)
    eut = liq.eutectic
    if eut is None:
        T_e, b_e = None, np.inf
    else:
        T_e, b_e = eut
    saturated_top = b_top >= b_e
    T_top = np.where(saturated_top, T_e if T_e is not None else 0.0,
                     liq.T_of_b(np.where(saturated_top, 0.0, b_top)))
    H_top = H_at(Mw, T_top)
    all_liquid = H >= H_top

    # regime A: liquid + ice on the liquidus, W from W_e (or ~0) up to Mw
    if eut is None:
        W_lo = np.full_like(Mw, tiny) * np.maximum(Mw, tiny)
    else:
        W_lo = np.where(S > 0, S / b_e, 0.0)
        W_lo = np.minimum(W_lo, Mw)

    def f(W):                                    # increasing in W
        b = S / np.maximum(W, tiny * np.maximum(Mw, tiny))
        return liq.T_of_b(b) - T_after(W)
    Wa = _standin.bisect(f, W_lo, np.maximum(Mw, W_lo), 70)
    W = Wa
    x = np.zeros_like(Mw)
    T = liq.T_of_b(S / np.maximum(W, tiny * np.maximum(Mw, tiny)))
    if eut is not None:
        T_ec = np.where(S > 0, T_e, liq.T_of_b(0.0))    # no solute: the pin is Tm
        H1 = H_at(W_lo, T_ec)
        H0 = H_at(0.0 * Mw, T_ec)
        in_B = (H < H1) & (H >= H0)
        in_C = H < H0
        with np.errstate(divide="ignore", invalid="ignore"):
            WB = np.where(H1 > H0, W_lo * (H - H0) / (H1 - H0), 0.0)
        W = np.where(in_B, WB, np.where(in_C, 0.0, W))
        x = np.where(in_B | in_C, S - b_e * W, x)
        T = np.where(in_B, T_ec, T)
        T_c = AMBIENT + (H + L * Mw) / (Co + CI * Mw)
        T = np.where(in_C, T_c, T)
    # all liquid: no ice, T from H
    T_l = AMBIENT + H / (Co + CW * Mw)
    W = np.where(all_liquid, Mw, W)
    T = np.where(all_liquid, T_l, T)
    if liq.solubility is not None:
        x_l = np.maximum(S - liq.solubility(T_l) * Mw, 0.0)
        x = np.where(all_liquid, x_l, x)
    else:
        x = np.where(all_liquid, 0.0, x)
    return W, x, T


def settle(E, C_other, m_w, m_i, s, x, dT_nuc, dt, liquidus=NACL, tau_melt=TAU_MELT):
    """Bring cells to (or toward) phase equilibrium at constant enthalpy.

    E, C_other, m_w, m_i, s (dissolved), x (crystals): arrays (grams, J, J/K). C_other
    is every heat capacity except liquid water and ice (solutes, crystals, the
    vessel). dT_nuc: K of supercooling the cell can hold (nucleation_k). dt: seconds,
    the melting relaxation uses it (np.inf = straight to equilibrium). liquidus: a
    Liquidus (T4's when it exists).
    Freezing is immediate once nucleated (the enthalpy method). Returns Settled.
    The enthalpy E - L_CONV20 m_i is conserved exactly; the heat of solution is T4's
    dissolved_J book, not here (a 100 g ice + 10 g salt cell ends within a few grams
    of the T4 figure with it left out)."""
    E = np.asarray(E, np.float64)
    Co = np.asarray(C_other, np.float64)
    m_w = np.asarray(m_w, np.float64)
    m_i = np.asarray(m_i, np.float64)
    s = np.asarray(s, np.float64)
    x = np.asarray(x, np.float64)
    Mw = m_w + m_i
    S = s + x
    H = enthalpy(E, m_i)
    C_cur = Co + CW * m_w + CI * m_i
    T_cur = AMBIENT + E / C_cur
    with np.errstate(divide="ignore", invalid="ignore"):
        W_eq, x_eq, _T = _equilibrium(H, Mw, S, Co, liquidus)
        b_cur = np.where(m_w > 1e-12, s / np.maximum(m_w, 1e-300), 0.0)
        T_f = liquidus.T_of_b(b_cur)
    freezing = W_eq < m_w - 1e-12 * np.maximum(Mw, 1e-300)
    melting = W_eq > m_w + 1e-12 * np.maximum(Mw, 1e-300)
    allowed = (m_i > 0.0) | (T_cur < T_f - np.asarray(dT_nuc, np.float64))
    fac = -np.expm1(-np.asarray(dt, np.float64) / tau_melt)
    fac = np.where(np.isfinite(fac), fac, 1.0)
    W_new = np.where(freezing & allowed, W_eq,
                     np.where(melting, m_w + (W_eq - m_w) * fac, m_w))
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(np.abs(W_eq - m_w) > 0, (W_new - m_w) / (W_eq - m_w), 0.0)
    x_new = x + (x_eq - x) * frac
    ice = Mw - W_new
    E_new = H + L_CONV20 * ice
    T_new = AMBIENT + E_new / (Co + CW * W_new + CI * ice)
    return Settled(E_new, W_new, ice, x_new, T_new, ice - m_i)


# ---- expansion on freezing -------------------------------------------------------
def expand_on_freeze(M, m_f, V_ml, other_ml=0.0, room_ml=np.inf):
    """A cell of V ml holds M grams of water (plus other_ml of something else) and m_f
    grams of it freeze. Ice takes 1/0.917 ml per gram, water 1 ml. The surplus volume
    is EXPELLED as liquid to a neighbour with room (up first). Returns
    (m_frozen, m_expelled, pending_ml): mass is exact (frozen + expelled never exceeds M).
    With nowhere to go, less freezes and the ml that could not leave is the pending
    pressure (L15 bursts the jar; a full sealed 1000 ml cell: 83 ml pending). 1000 g in a 1000 ml cell freezes 917 g and expels 83 g."""
    M = np.asarray(M, np.float64)
    m_f = np.minimum(np.asarray(m_f, np.float64), M)
    avail = np.asarray(V_ml, np.float64) - other_ml
    extra = 1.0 / RHO_ICE - 1.0
    need = m_f / RHO_ICE + (M - m_f)
    over = np.maximum(need - avail, 0.0)
    liquid_left = M - m_f
    # expel liquid first; if there is not enough liquid, the ice itself is limited
    exp1 = np.minimum(over, liquid_left)
    m_ice_cap = avail * RHO_ICE                       # ice that fits if all liquid is out
    m_f2 = np.where(over > liquid_left, np.minimum(m_f, m_ice_cap), m_f)
    exp2 = np.where(over > liquid_left, M - m_f2, exp1)
    # the neighbour's room limits what can be expelled
    room = np.asarray(room_ml, np.float64)
    ok = exp2 <= room
    # shrink the frozen mass until the expelled volume fits the room
    m_f3 = np.where(ok, m_f2, (avail - M + room) / extra)
    m_f3 = np.clip(m_f3, 0.0, m_f2)
    exp3 = np.where(ok, exp2, room)
    pending = np.maximum(exp2 - exp3, 0.0)           # ml that had to leave and could not
    return m_f3, exp3, pending


def ice_volume_ml(ice_g):
    return np.asarray(ice_g, np.float64) / RHO_ICE


def promote(ice_g, vox_l, was_ice, fill=ICE_FILL, unfill=ICE_UNFILL):
    """Is a cell mat ICE? True past `fill` of its volume; once ICE it stays ICE until it
    drops below `unfill` (hysteresis: +/-1e-3 noise cannot flicker it)."""
    f = ice_volume_ml(ice_g) / (np.asarray(vox_l, np.float64) * 1000.0)
    return np.where(np.asarray(was_ice), f >= unfill, f >= fill)


# ---- closed forms ------------------------------------------------------------------
def stefan(dT, t, k=2.2, rho=917.0, Lf=333.55e3):
    """Ice thickness (m) after t seconds with the surface held dT below freezing,
    no water-side heat, no snow: h = sqrt(2 k dT t / (rho Lf)). SI units. 11 cm in a
    day at 10 K, 30 cm in a week (c)."""
    return np.sqrt(2.0 * k * np.asarray(dT, np.float64) * np.asarray(t, np.float64) / (rho * Lf))


def stefan_film(dT, t, h_air, k=2.2, rho=917.0, Lf=333.55e3):
    """The same with an air film h_air (W/m2K) in series (T8's air-film form, the closed
    form the slow clock uses): h/(2k)... solves h^2/(2k) + h/h_air = dT t/(rho Lf)."""
    a = np.asarray(dT, np.float64) * np.asarray(t, np.float64) / (rho * Lf)
    return k * (-1.0 / h_air + np.sqrt(1.0 / h_air ** 2 + 2.0 * a / k))


def ice_rise_ratio():
    """Level rise of a pond per metre of ice formed (0.09): 1/0.917 - 1."""
    return 1.0 / RHO_ICE - 1.0


def melt_extent_pure(C, T):
    """Grams of ice made when a supercooled pure-water cell of heat capacity C (J/K, its
    value before freezing) at T < 0 freezes to the pin: C (0 - T) / 333.55."""
    return C * (0.0 - np.asarray(T, np.float64)) / LF0
