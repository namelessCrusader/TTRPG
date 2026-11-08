"""Skin and air physics for a body (wave 13, T7): vapour, heat transfer
coefficients, evaporation, breath. Pure functions; no World.

References: ISO 7730 / Fanger / ASHRAE (comfort conventions); Magnus
(vapour pressure); JOS-3 (Lewis ratio for evaporation). Marks: (u) unverified.

Units: C for temperature, kPa for vapour pressure, W and m2 for heat, g and s
for water, L/min for ventilation.
"""
import math

import numpy as np

SIGMA = 5.670374e-8                      # W/(m2 K4)
P_ATM = 101.325                          # kPa
MAGNUS_A, MAGNUS_B = 17.62, 243.12       # Magnus constants (over water)
CP_AIR = 1006.0                          # J/(kg K), dry air
CP_VAP = 1860.0                          # J/(kg K), water vapour
RHO_AIR_20 = 1.204                       # kg/m3 of dry air at 20 C, 1 atm
LEWIS_K_PER_KPA = 16.5                   # Lewis ratio, K/kPa (JOS-3)
I_M_CLOTH = 0.38                         # moisture permeability index of a garment (u)
CLO = 0.155                              # m2 K/W
EMISS_SKIN = 0.98                        # (u 0.95-0.99)
F_EFF = 0.71                             # effective radiation area factor of a standing body (ISO; the
                                         # voxel path uses per-zone view instead and passes 1.0)


def p_sat(T):
    """Saturation vapour pressure over water, kPa (Magnus): 2.333 at 20 C, 6.265 at 37 C.

    Domain: T must be finite and above -B (-243.12 C), where the fit has its pole
    and is meaningless; otherwise ValueError. The fit is good to about 0.5 percent
    from -45 to 60 C and 1.6 percent high at 100 C (103 against 101.325 kPa); it
    is not clipped, the caller compares it with the total pressure (see
    vapour_pressure and humidity_ratio, which refuse a vapour pressure that
    reaches the total)."""
    T = np.asarray(T, dtype=np.float64)
    if not np.all(np.isfinite(T)) or np.any(T <= -MAGNUS_B):
        raise ValueError(f"p_sat: T must be finite and above {-MAGNUS_B} C")
    return 0.6112 * np.exp(MAGNUS_A * T / (MAGNUS_B + T))


def vapour_pressure(T, rh, P=P_ATM):
    """Partial pressure of vapour at air temperature T and relative humidity rh (0..1), kPa.

    Physical limit: the vapour cannot be at or above the total pressure P. Where
    p_sat(T) >= P water BOILS (about 99.6 C at 1 atm; 36 C at 6 kPa) and air
    cannot hold a fraction rh of it as a mixture, so a result >= P raises
    ValueError, as does rh outside 0..1 (a tiny float overshoot is allowed)."""
    rh = np.asarray(rh, dtype=np.float64)
    if not np.all(np.isfinite(rh)) or np.any(rh < 0.0) or np.any(rh > 1.0 + 1e-9):
        raise ValueError("vapour_pressure: rh must be in 0..1")
    pv = rh * p_sat(T)
    if np.any(pv >= P):
        raise ValueError(f"vapour_pressure: vapour pressure reaches the total pressure {P} kPa (water boils)")
    return pv


def humidity_ratio(p_v, P=P_ATM):
    """Mass of vapour per mass of dry air: 0.622 p / (P - p).

    Domain: 0 <= p_v < P. At or above the total pressure there is no dry air
    left (the ratio is infinite, then negative), so that raises ValueError."""
    p_v = np.asarray(p_v, dtype=np.float64)
    if not np.all(np.isfinite(p_v)) or np.any(p_v < 0.0) or np.any(p_v >= P):
        raise ValueError(f"humidity_ratio: need 0 <= p_v < P ({P} kPa)")
    return 0.622 * p_v / (P - p_v)


def latent_heat(T):
    """Latent heat of vaporisation of water at T, J/kg: (2501 - 2.37 T) kJ/kg (2.42 MJ/kg at 33 C)."""
    return (2501.0 - 2.37 * np.asarray(T, dtype=np.float64)) * 1000.0


# ── heat transfer coefficients (W/m2K) ───────────────────────────────────────
def h_c_nat(dT):
    """Natural convection on a standing body, ISO 7730: 2.38 |dT|^0.25 (4.5 at dT = 13)."""
    return 2.38 * np.abs(np.asarray(dT, dtype=np.float64)) ** 0.25


def h_c_forced(v):
    """Forced convection, ISO 7730: 12.1 sqrt(v), v the relative air speed m/s."""
    return 12.1 * np.sqrt(np.maximum(np.asarray(v, dtype=np.float64), 0.0))


def h_c(dT, v=0.0):
    """ISO 7730: the larger of natural and forced."""
    return np.maximum(h_c_nat(dT), h_c_forced(v))


def h_r(T_surf, T_rad, eps=EMISS_SKIN, f_eff=F_EFF):
    """Linearised radiative coefficient, 4 eps sigma f_eff (T_m + 273.15)^3 (about 4.5
    at 22 C and f_eff 0.71 -> 3.3; 4.7 for f_eff 1)."""
    Tm = 0.5 * (np.asarray(T_surf, dtype=np.float64) + np.asarray(T_rad, dtype=np.float64))
    return 4.0 * eps * SIGMA * f_eff * (Tm + 273.15) ** 3


def f_cl_iso(I_cl):
    """Clothing area factor (ISO 7730) for a clothing insulation I_cl in m2K/W:
    1 + 1.29 I (I <= 0.078), 1.05 + 0.645 I above. 1.15 at 1 clo."""
    I = np.asarray(I_cl, dtype=np.float64)
    return np.where(I <= 0.078, 1.00 + 1.29 * I, 1.05 + 0.645 * I)


def operative_temperature(h_c_, T_a, h_r_, T_r):
    return (h_c_ * T_a + h_r_ * T_r) / (h_c_ + h_r_)


def dry_heat_loss(T_sk, T_a, T_r, I_cl=0.0, v=0.0, f_eff=F_EFF, eps=EMISS_SKIN):
    """Dry heat from the skin per m2, W/m2 (ISO 7730): (T_sk - T_op) / (I_cl + 1/(f_cl h)).
    The clothing surface temperature is approximated by the skin's for h_c."""
    hc = h_c(T_sk - T_a, v)
    hr = h_r(T_sk, T_r, eps, f_eff)
    h = hc + hr
    top = operative_temperature(hc, T_a, hr, T_r)
    return (T_sk - top) / (I_cl + 1.0 / (f_cl_iso(I_cl) * h))


# ── material-row data: the face law is DATA, not a body test (plan 3, T7) ────
# A face material row carries its convection law and its emissivity. The
# default is the lattice's old constants; the FLESH row gets the ISO law and
# 0.98. The lattice reads the row; nothing here knows it is looking at a body.
LEGACY_FACE_ROW = {"conv_law": "const", "conv_const": 10.0, "conv_nat_a": 0.0,
                   "conv_nat_exp": 0.25, "conv_forced_a": 0.0, "emissivity": 0.95}
FLESH_FACE_ROW = {"conv_law": "iso", "conv_const": 10.0, "conv_nat_a": 2.38,
                  "conv_nat_exp": 0.25, "conv_forced_a": 12.1, "emissivity": EMISS_SKIN}


def face_h_conv(row, dT, v=0.0):
    """The convective coefficient a face material row asks for, W/m2K."""
    if row["conv_law"] == "const":
        return np.full_like(np.asarray(dT, dtype=np.float64), row["conv_const"])
    nat = row["conv_nat_a"] * np.abs(np.asarray(dT, dtype=np.float64)) ** row["conv_nat_exp"]
    forced = row["conv_forced_a"] * np.sqrt(max(v, 0.0))
    return np.maximum(nat, forced)


# ── evaporation ──────────────────────────────────────────────────────────────
def evap_max(T_sk, T_a, rh, h_c_, I_cl=0.0, area_m2=1.0, p_a=None):
    """Maximum evaporation (W, SIGNED) from a skin patch of `area_m2`:

        E_max = A (p_sk,s - p_a) / (1/(Lewis h_c) + R_e,cl)

    Lewis: h_e = 16.5 K/kPa x h_c. R_e,cl = I_cl / (i_m L_R) (m2 kPa/W) with
    i_m = 0.38 (u). NOT clipped: when the air holds more vapour than a skin at
    T_sk can (40 C at 90 percent RH against 35 C skin), E_max is negative,
    which is CONDENSATION on the skin (a small heat gain). Positive = the air
    can take water."""
    p_sk = p_sat(T_sk)
    pa = vapour_pressure(T_a, rh) if p_a is None else p_a
    he = LEWIS_K_PER_KPA * np.asarray(h_c_, dtype=np.float64)
    R_cl = np.asarray(I_cl, dtype=np.float64) / (I_M_CLOTH * LEWIS_K_PER_KPA)
    return area_m2 * (p_sk - pa) / (1.0 / he + R_cl)


DRY_WETTEDNESS = 0.06                    # diffusion through dry skin (u: 0.06 .. 0.1)


def evaporation_step(sweat_g, film_g, e_max_w, dt, T_sk, film_cap_g=None):
    """Water exchange of a skin patch over `dt` s. `sweat_g` is produced this
    step, `film_g` already on the skin. Returns a dict:
      evap_g     evaporated (>0) from the film, or condensed (<0) onto it, g
      film_g     the film after, g
      drip_g     water above the film capacity that drips off, g
      latent_w   the heat taken from the skin (>0) or given to it (<0), W
    Evaporated mass = min(produced + film, E_max dt / L); diffusion through
    dry skin gives at least DRY_WETTEDNESS of E_max even with no film (that
    water comes from the body, not the film, and is reported in evap_g).
    Mass closes: sweat_g + film_in = evap_g (positive part from the film) + film_out + drip_g
    (+ body diffusion g, `diffuse_g`)."""
    L = float(latent_heat(T_sk))             # J/kg
    cap = e_max_w * dt / L * 1000.0          # grams the air could take (signed)
    avail = sweat_g + film_g
    diffuse = 0.0
    if cap >= 0.0:
        ev = min(avail, cap)
        left = avail - ev
        diffuse = max(DRY_WETTEDNESS * cap - ev, 0.0)     # the skin's own diffusion
        evap = ev
    else:                                    # condensation: the film grows
        evap = cap
        left = avail - cap
    drip = 0.0
    if film_cap_g is not None and left > film_cap_g:
        drip = left - film_cap_g
        left = film_cap_g
    total = evap + diffuse
    return {"evap_g": evap, "diffuse_g": diffuse, "film_g": left, "drip_g": drip,
            "latent_w": total * L / 1000.0 / dt}


# ── breath ───────────────────────────────────────────────────────────────────
def exhale_temperature(T_in, T_core):
    """T_ex = min(T_core, 34 + 0.2 (T_in - 20)) (u): 34 C in a 20 C room, rising
    toward the core when the inhaled air is hot. (The spec wrote the intercept
    as 32 but its own gloss as 34 C at 20 C; the gloss is the number the
    acceptance bands 1.5-3 W / 8-13 W / 0.2-0.3 g/min fit, 32 gives 8.5 W and
    0.21 g/min at the edge, so the intercept is 34.)"""
    return np.minimum(T_core, 34.0 + 0.2 * (np.asarray(T_in, dtype=np.float64) - 20.0))


def breath_loss(V_E_l_min, T_in, rh_in, T_core=37.0, P=P_ATM):
    """Heat and water lost to breathing. `V_E_l_min` is the ventilation (BTPS-ish,
    L/min of air at room conditions). The exhaled air leaves at T_ex saturated.

        Q_sens = m_dry (cp_a + w_in cp_v)(T_ex - T_in)
        Q_lat  = m_dry (w_ex - w_in) L(T_ex)

    Returns dict(sens_w, lat_w, water_g_s, T_ex, w_in, w_ex). At rest (7.5 L/min)
    in 20 C, 50 percent RH: about 2 W sensible, 10 W latent, 0.26 g/min."""
    T_ex = float(exhale_temperature(T_in, T_core))
    rho = RHO_AIR_20 * 293.15 / (T_in + 273.15)
    m_dry = V_E_l_min / 1000.0 / 60.0 * rho          # kg/s of dry air (the humid part is small)
    w_in = float(humidity_ratio(vapour_pressure(T_in, rh_in, P), P))
    w_ex = float(humidity_ratio(p_sat(T_ex), P))
    sens = m_dry * (CP_AIR + w_in * CP_VAP) * (T_ex - T_in)
    lat = m_dry * (w_ex - w_in) * float(latent_heat(T_ex))
    return {"sens_w": sens, "lat_w": lat, "water_g_s": m_dry * (w_ex - w_in) * 1000.0,
            "T_ex": T_ex, "w_in": w_in, "w_ex": w_ex}
