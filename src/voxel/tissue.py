"""Tissue and body-sheet data for body heat (wave 13, T7), and Pennes in a voxel.

Pure functions and small tables. No World. Everything here is a number or a
formula; the law that applies it to the lattice (bodyheat.py) comes later.

Marks: (u) = unverified number from memory of the literature, to be checked
(A-S marks the sourced ones); (d) = a design choice, not physics.

What is in this file
  TISSUE         seven tissues: density, capacity, conductivity, basal heat,
                 basal blood flow (IT'IS 4.x, Duck, Fiala, Elia, ICRP 89; all u)
  ZONES          the 14 body zones, and ZONE_OF_BONE, the ONE place that says
                 which zone a bone (and the distal end of a bone) belongs to
  WINTER_SHARE   zone mass shares (Winter 1990); AREA_SHARE zone skin-area
                 shares (the segment shares JOS-3 uses, u). Both come from
                 `bodysheet` when that module exists, else the local copy.
  sheet(...)     one person's numbers; zone_params(sheet) the per-zone
                 q_m, perfusion, R_sc, counter-current factor, area
  pennes_step    one exact step of Pennes in the phi form, the central blood
                 pool solved with the voxels in closed form
"""
import math

import numpy as np

try:                                     # the G10 body numbers, when they exist
    from . import bodysheet as _BS
except ImportError:                      # pragma: no cover - lane A-G10 builds it
    _BS = None

# ── constants ────────────────────────────────────────────────────────────────
RHO_B = 1050.0                           # blood density kg/m3
C_B = 3617.0                             # blood capacity J/(kg K)
RHOC_B = RHO_B * C_B                     # 3.80e6 J/(m3 K), the perfusion currency
MET_W_PER_KG = 1.22                      # the SIM's MET: ACSM 3.5 ml O2/kg/min = 1.22 W/kg
ISO_MET_W_PER_M2 = 58.15                 # the ISO 7730 comfort convention (never mixed with the above)
Q10 = 2.0                                # tissue metabolism, per 10 K (Fiala)
T_REF = 37.0                             # core-zone reference temperature
CENTRAL_SHARE = 0.4                      # share of the blood that is the central pool (u)
BLOOD_ML_PER_KG = {"male": 70.0, "female": 65.0}     # (u)
SKIN_M = 0.002                           # dermis + epidermis thickness for R_sc, m (u)
KCAL_DAY_W = 4184.0 / 86400.0            # 1 kcal/day in watts


class Tissue:
    """One row of the table. `w` is basal blood flow in ml/(100 g min)."""
    __slots__ = ("rho", "c", "k", "q_m", "w")

    def __init__(self, rho, c, k, q_m, w):
        self.rho, self.c, self.k, self.q_m, self.w = rho, c, k, q_m, w

    @property
    def w_b(self):
        """Basal perfusion per tissue VOLUME, 1/s (ml blood per ml tissue per s)."""
        return self.w * (self.rho / 1000.0) / 6000.0


# rho kg/m3, c J/(kg K), k W/(m K), q_m basal W/m3, w ml/(100 g min). All (u).
# The ORGAN q_m is the least certain row (liver 10 W/kg, heart 21, kidney 21,
# gut 3-5, lung low, by Elia 1992): it was re-fitted so the reference man and
# woman land within 10 percent of Mifflin-St Jeor with hands, feet and neck
# in (the draft table summed to 96 W for 73 kg, not 82 W). The fit is
# `ORGAN_QM_FIT` below and test_u_reference_people_match_mifflin.
ORGAN_QM_FIT = 6500.0
TISSUE = {
    "skin":   Tissue(1109.0, 3391.0, 0.37, 368.0, 7.0),
    "fat":    Tissue(911.0, 2348.0, 0.21, 58.0, 2.0),
    "muscle": Tissue(1090.0, 3421.0, 0.49, 684.0, 3.0),
    "bone":   Tissue(1500.0, 1313.0, 0.32, 85.0, 2.5),
    "organs": Tissue(1060.0, 3600.0, 0.52, ORGAN_QM_FIT, 48.0),
    "brain":  Tissue(1045.0, 3696.0, 0.51, 13400.0, 50.0),
    "blood":  Tissue(RHO_B, C_B, 0.52, 0.0, 0.0),
}
TISSUES = ("skin", "fat", "muscle", "bone", "organs", "brain", "blood")

# ── zones ────────────────────────────────────────────────────────────────────
# 14 zones: head (with the neck), torso, and for each side upper arm, forearm,
# hand, thigh, shin, foot. The names equal the bone names scenes._person uses,
# so a bone's zone is its own name; hand and foot are the DISTAL part of the
# forearm and shin bones.
_SIDE_PARTS = ("upper arm", "forearm", "hand", "thigh", "shin", "foot")
ZONES = ("head", "torso") + tuple(f"{s} {p}" for s in ("left", "right") for p in _SIDE_PARTS)
ZONE_INDEX = {z: i for i, z in enumerate(ZONES)}
NZONES = len(ZONES)


def _kind(zone):
    """'left forearm' -> 'forearm'; 'head' -> 'head'."""
    return zone.split(" ", 1)[1] if zone[:4] in ("left", "righ") else zone

ZONE_OF_BONE = {"head": "head", "torso": "torso"}
for _s in ("left", "right"):
    for _p in ("upper arm", "forearm", "thigh", "shin"):
        ZONE_OF_BONE[f"{_s} {_p}"] = f"{_s} {_p}"
# bone -> (zone of the distal end, fraction of the bone length that is distal),
# measured along the bone axis from the proximal joint in the REST pose.
# Distal 27 percent of the forearm is the hand, distal 24 percent of the shin
# is the foot (Winter, u).
DISTAL_OF_BONE = {}
for _s in ("left", "right"):
    DISTAL_OF_BONE[f"{_s} forearm"] = (f"{_s} hand", 0.27)
    DISTAL_OF_BONE[f"{_s} shin"] = (f"{_s} foot", 0.24)


def zone_of(bone, along=None):
    """The zone of a point on `bone`. `along` is 0..1 from the proximal joint
    (None = the bone as a whole, which is its main zone). Unknown bones give
    None: the caller falls back to its own map."""
    z = ZONE_OF_BONE.get(bone)
    if z is None:
        return None
    d = DISTAL_OF_BONE.get(bone)
    if d is not None and along is not None and along >= 1.0 - d[1]:
        return d[0]
    return z


_WINTER_LOCAL = {"head": 0.081, "torso": 0.497}
_AREA_LOCAL = {"head": 0.0744, "torso": 0.2982}
for _s in ("left", "right"):
    for _p, _w, _a in (("upper arm", 0.028, 0.0514), ("forearm", 0.016, 0.0337),
                       ("hand", 0.006, 0.0268), ("thigh", 0.100, 0.1119),
                       ("shin", 0.0465, 0.0600), ("foot", 0.0145, 0.0300)):
        _WINTER_LOCAL[f"{_s} {_p}"] = _w
        _AREA_LOCAL[f"{_s} {_p}"] = _a


def _normed(d):
    s = math.fsum(d[z] for z in ZONES)
    return {z: d[z] / s for z in ZONES}


def _bs_winter():
    """Winter's fractions from `bodysheet.WINTER` (its names: trunk, shank), keyed by our
    zones; None if bodysheet is absent or does not have them all."""
    if _BS is None:
        return None
    try:
        w = _BS.WINTER
        alias = {"torso": "trunk", "shin": "shank"}
        return {z: float(w[alias.get(_kind(z), _kind(z))]) for z in ZONES}
    except (AttributeError, KeyError, TypeError):
        return None


def _shares(local, bs=None):
    """`bodysheet`'s table when it has one for exactly our zones, else ours."""
    return _normed(bs if bs is not None else local)


WINTER_SHARE = _shares(_WINTER_LOCAL, _bs_winter())       # mass fractions, sum 1 (bodysheet.WINTER when present)
AREA_SHARE = _shares(_AREA_LOCAL)                          # skin-area fractions, sum 1 (u; bodysheet has no area table)

# Lean (non-fat) composition of each zone, fractions of the zone's non-fat
# mass, sum 1 (from the draft composition of the first T7 check, hands and
# feet split off the forearm and shin, the neck in the head; u).
_LEAN = {
    "head":      {"skin": .084, "muscle": .217, "bone": .265, "organs": .0, "brain": .337, "blood": .097},
    "torso":     {"skin": .065, "muscle": .405, "bone": .205, "organs": .215, "brain": .0, "blood": .110},
    "upper arm": {"skin": .093, "muscle": .617, "bone": .216, "organs": .0, "brain": .0, "blood": .074},
    "forearm":   {"skin": .126, "muscle": .559, "bone": .245, "organs": .0, "brain": .0, "blood": .070},
    "hand":      {"skin": .200, "muscle": .300, "bone": .380, "organs": .0, "brain": .0, "blood": .120},
    "thigh":     {"skin": .057, "muscle": .683, "bone": .211, "organs": .0, "brain": .0, "blood": .049},
    "shin":      {"skin": .086, "muscle": .593, "bone": .247, "organs": .0, "brain": .0, "blood": .074},
    "foot":      {"skin": .150, "muscle": .280, "bone": .450, "organs": .0, "brain": .0, "blood": .120},
}
# share of the body's fat that sits in each zone (u); normalised over the 14
_FAT_LOCAL = {"head": .04, "torso": .47, "upper arm": .035, "forearm": .02,
              "hand": .004, "thigh": .115, "shin": .035, "foot": .008}
# subcutaneous fat thickness at the reference fat fraction REF_FAT, m (u)
_FAT_M_LOCAL = {"head": .003, "torso": .009, "upper arm": .005, "forearm": .003,
                "hand": .002, "thigh": .008, "shin": .004, "foot": .002}
# half the muscle thickness under the fat on the way to the core, m (u); the
# lattice default is a 10 mm path
_MUSCLE_HALF_M = {"head": .004, "torso": .015, "upper arm": .012, "forearm": .008,
                  "hand": .004, "thigh": .018, "shin": .010, "foot": .004}
# counter-current exchange factor eps_cc (arterial blood warmed by the vein
# beside it); 0 for head and torso (u)
_EPS_CC = {"head": 0.0, "torso": 0.0, "upper arm": 0.15, "forearm": 0.30,
           "hand": 0.55, "thigh": 0.15, "shin": 0.40, "foot": 0.55}
REF_FAT = 0.19
PERF_ALLOMETRY = -0.25                   # blood flow per kg falls with size: cardiac output ~ mass^0.75 (u)
PERF_REF_KG = 70.0
REF_LEAN_KG = 59.1                        # lean mass of the 73 kg reference man
ORGAN_ALLOMETRY = -0.35                  # organ share of the lean torso ~ lean^-0.35 (u, fitted)


_FAT_SHARE = _normed({z: _FAT_LOCAL[_kind(z)] for z in ZONES})


class Sheet:
    """One person's numbers. Plain attributes; `as_dict()` is p["sheet"]."""
    __slots__ = ("sex", "mass_kg", "height_m", "fat_frac", "age_y", "bmr_scale")

    def __init__(self, sex, mass_kg, height_m, fat_frac, age_y, bmr_scale=1.0):
        self.sex, self.mass_kg, self.height_m = sex, float(mass_kg), float(height_m)
        self.fat_frac, self.age_y, self.bmr_scale = float(fat_frac), float(age_y), float(bmr_scale)

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}

    @classmethod
    def from_dict(cls, d):
        return cls(d["sex"], d["mass_kg"], d["height_m"], d["fat_frac"],
                   d.get("age_y", 30.0), d.get("bmr_scale", 1.0))

    def __repr__(self):
        return "Sheet(%s)" % ", ".join(f"{k}={getattr(self, k)!r}" for k in self.__slots__)


REFERENCE = {"male": (73.0, 1.75), "female": (62.0, 1.65)}   # kg, m: used only when bodysheet is absent


def bodysheet_sheet(name="adult male", **override):
    """A Sheet from `bodysheet.get_sheet(name, **override)` (the G10 numbers), or
    None when bodysheet is not there. Extra keys of its dict (name, strength_N)
    are kept out of the Sheet; they stay in bodysheet."""
    if _BS is None:
        return None
    try:
        return Sheet.from_dict(_BS.get_sheet(name, **override))
    except (AttributeError, KeyError, TypeError):
        return None


def bsa_dubois(mass_kg, height_m):
    """Du Bois body surface area, m2: 0.007184 m^0.425 h_cm^0.725."""
    return 0.007184 * mass_kg ** 0.425 * (height_m * 100.0) ** 0.725


def fat_frac_deurenberg(sex, mass_kg, height_m, age_y=30.0):
    """Body fat fraction from BMI, age and sex (Deurenberg 1991; u: +-4 points)."""
    bmi = mass_kg / (height_m * height_m)
    s = 1.0 if sex == "male" else 0.0
    f = (1.20 * bmi + 0.23 * age_y - 10.8 * s - 5.4) / 100.0
    return float(min(max(f, 0.04), 0.50))


def bmr_mifflin(sex, mass_kg, height_m, age_y=30.0):
    """Basal metabolic power by Mifflin-St Jeor, W (73 kg, 175 cm, 30 y: 81 W)."""
    kcal = 10.0 * mass_kg + 6.25 * height_m * 100.0 - 5.0 * age_y + (5.0 if sex == "male" else -161.0)
    return kcal * KCAL_DAY_W


def met_to_w(met, mass_kg):
    """The SIM's metabolic power: ACSM 1 MET = 1.22 W/kg (85 W at 70 kg)."""
    return met * MET_W_PER_KG * mass_kg


def iso_met_to_w(met, bsa_m2):
    """The ISO 7730 comfort convention, 58.15 W/m2 per met (107 W at 1.85 m2).
    Used only in comfort checks; never mixed with `met_to_w`."""
    return met * ISO_MET_W_PER_M2 * bsa_m2


def sheet(sex="male", mass_kg=None, height_m=None, fat_frac=None, age_y=30.0, bmr_scale=1.0):
    """Build a Sheet. Wraps `bodysheet` (G10): with no mass and no height given, the
    person is its adult man or woman (fat fraction included, unless given).
    Otherwise a missing number comes from the reference adult scaled by mass^(1/3),
    and from Deurenberg's fat fraction. Without bodysheet the same, from
    REFERENCE."""
    if mass_kg is None and height_m is None:
        ref = bodysheet_sheet("adult " + sex)
        if ref is not None:
            return Sheet(sex, ref.mass_kg, ref.height_m, ref.fat_frac if fat_frac is None else fat_frac,
                         age_y, bmr_scale)
    rm, rh = REFERENCE[sex]
    mass_kg = rm if mass_kg is None else mass_kg
    height_m = (rh * (mass_kg / rm) ** (1.0 / 3.0)) if height_m is None else height_m
    if fat_frac is None:
        fat_frac = fat_frac_deurenberg(sex, mass_kg, height_m, age_y)
    return Sheet(sex, mass_kg, height_m, fat_frac, age_y, bmr_scale)


def blood_mass_kg(sh):
    return sh.mass_kg * BLOOD_ML_PER_KG[sh.sex] / 1000.0 * RHO_B / 1000.0 * 1.0


def pool_capacity(sh):
    """C_pool, J/K: the central share of the blood times c_blood (7.7 kJ/K at 73 kg)."""
    return blood_mass_kg(sh) * CENTRAL_SHARE * C_B


def composition(sh):
    """Mass of each tissue in each zone, kg: {zone: {tissue: kg}}. Fat follows
    fat_frac and the zone fat shares; the rest of each zone is its lean mix.
    The tissues of a zone sum to the zone's Winter mass exactly."""
    fat_total = sh.fat_frac * sh.mass_kg
    lean_total = sh.mass_kg - fat_total
    out = {}
    for z in ZONES:
        zm = WINTER_SHARE[z] * sh.mass_kg
        fat = min(fat_total * _FAT_SHARE[z], 0.85 * zm)
        lean = zm - fat
        lw = _LEAN[_kind(z)]
        if lw["organs"] > 0.0:           # organs grow slower than the body (u)
            f = (lean_total / REF_LEAN_KG) ** ORGAN_ALLOMETRY
            lw = dict(lw)
            lw["muscle"] += lw["organs"] * (1.0 - f)
            lw["organs"] *= f
        row = {"fat": fat}
        acc = fat
        names = [t for t in TISSUES if t != "fat"]
        for t in names[:-1]:
            row[t] = lean * lw[t]
            acc += row[t]
        row[names[-1]] = zm - acc            # the last one closes the sum exactly
        out[z] = {t: row[t] for t in TISSUES}
    return out


class ZoneParams:
    """Per-zone numbers for the law. SI units.
    q_m        W/m3 over the whole zone (tissue-mixed, includes skin)
    power_w    q_m x volume, basal
    w_b_core   1/s perfusion of the non-skin tissue (per its volume)
    w_b_skin   1/s perfusion of the skin tissue (per its volume), basal
    flow_ml_s  basal blood flow of the zone, ml/s
    R_sc       core-to-skin resistance, m2 K/W
    dermal_frac skin volume / zone volume
    eps_cc     counter-current exchange factor
    area_m2    true skin area of the zone"""
    __slots__ = ("zone", "mass_kg", "volume_m3", "c_mix", "q_m", "power_w", "w_b_core",
                 "w_b_skin", "flow_ml_s", "R_sc", "fat_m", "skin_m", "dermal_frac",
                 "eps_cc", "area_m2", "fat_kg")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw[k])

    def __repr__(self):
        return "ZoneParams(%s)" % ", ".join(f"{k}={getattr(self, k)!r}" for k in self.__slots__)


def zone_params(sh):
    """{zone: ZoneParams} for a Sheet. A lean person has thin fat (R_sc falls),
    a fat one thick (R_sc rises): fat_m scales with fat_frac."""
    comp = composition(sh)
    bsa = bsa_dubois(sh.mass_kg, sh.height_m)
    pf = (sh.mass_kg / PERF_REF_KG) ** PERF_ALLOMETRY
    out = {}
    for z in ZONES:
        c = comp[z]
        kind = _kind(z)
        vol = {t: c[t] / TISSUE[t].rho for t in TISSUES}
        V = math.fsum(vol.values())
        mass = math.fsum(c.values())
        P = math.fsum(TISSUE[t].q_m * vol[t] for t in TISSUES) * sh.bmr_scale
        cap = math.fsum(c[t] * TISSUE[t].c for t in TISSUES)
        flow = pf * math.fsum(TISSUE[t].w_b * vol[t] for t in TISSUES if t != "skin")
        Vc = V - vol["skin"]
        fat_m = _FAT_M_LOCAL[kind] * sh.fat_frac / REF_FAT
        R = (SKIN_M / TISSUE["skin"].k + fat_m / TISSUE["fat"].k
             + _MUSCLE_HALF_M[kind] / TISSUE["muscle"].k)
        out[z] = ZoneParams(
            zone=z, mass_kg=mass, volume_m3=V, c_mix=cap / mass, q_m=P / V, power_w=P,
            w_b_core=flow / Vc, w_b_skin=pf * TISSUE["skin"].w_b,
            flow_ml_s=(flow + pf * TISSUE["skin"].w_b * vol["skin"]) * 1e6,
            R_sc=R, fat_m=fat_m, skin_m=SKIN_M, dermal_frac=vol["skin"] / V,
            eps_cc=_EPS_CC[kind], area_m2=AREA_SHARE[z] * bsa, fat_kg=c["fat"])
    return out


def basal_power_w(sh):
    """Sum of q_m V over the body at 37 C, W."""
    return math.fsum(zp.power_w for zp in zone_params(sh).values())


def cardiac_output_l_min(sh):
    """Sum of the basal tissue flows, L/min (4.5-6.5 expected)."""
    return math.fsum(zp.flow_ml_s for zp in zone_params(sh).values()) * 60.0 / 1000.0


# ── gates and metabolism of one voxel ────────────────────────────────────────
def smoothstep(x, lo, hi):
    """0 at lo, 1 at hi, smooth in between."""
    t = np.clip((np.asarray(x, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def perfusion_factor(T, omega=0.0, circ=1.0, vaso=1.0):
    """The state factor on a voxel's perfusion: (1 - Omega) x smoothstep(T;
    0 C, 2 C) x circ x vaso. Omega is the burn damage integral (1 = coagulated:
    a scald is not pulled back to the pool); the smoothstep stops flow in
    frozen flesh; circ is blood volume times heart function (0 when the heart
    has stopped), so a dead body has no flow and no flag says so."""
    return (1.0 - np.asarray(omega, dtype=np.float64)) * smoothstep(T, 0.0, 2.0) * circ * vaso


def perfusion_g(w_b, volume_m3, factor=1.0):
    """Perfusion conductance of a voxel, W/K: w_b rho_b c_b V x factor."""
    return w_b * RHOC_B * volume_m3 * factor


def metab_power(q_m, volume_m3, T, bmr_scale=1.0, T_ref=T_REF):
    """Basal heat of a voxel, W: q V x scale x Q10^((T - T_ref)/10), stopped in
    frozen flesh with the same smoothstep as the flow."""
    T = np.asarray(T, dtype=np.float64)
    return q_m * volume_m3 * bmr_scale * Q10 ** ((T - T_ref) / 10.0) * smoothstep(T, -2.0, 2.0)


# ── Pennes in a voxel, the pool solved together (spec 3.1) ───────────────────
def _phi(x):
    """a/x with a = 1 - exp(-x): 1 at x = 0 (series where x is tiny)."""
    x = np.asarray(x, dtype=np.float64)
    a = -np.expm1(-x)
    small = x < 1e-8
    safe = np.where(small, 1.0, x)
    phi = np.where(small, 1.0 - 0.5 * x + x * x / 6.0, a / safe)
    return a, phi


def pennes_step(T, C, g, Q, dt, T_p, C_p, Q_o=0.0, eps=None):
    """One exact step of Pennes bioheat for voxels coupled to the central pool.

        rho c dT/dt = g (T_p - T) + Q          (per voxel; conduction is not here)

    T, C (J/K), g (W/K), Q (W: metabolism + work + shiver - sweat, all that is
    not blood) are arrays over voxels; T_p, C_p (J/K) and Q_o (W; breath and
    other pool sinks) are scalars. `eps` (array or None) is the counter-current
    factor: the voxel sees arterial blood T_a = (1-eps) T_p + eps T, which is
    just g(1-eps) as the conductance, the same on both sides, so energy is
    conserved by construction.

    With x = g dt / C, a = 1 - exp(-x), phi = a / x:
        T'  = T + a (T_p' - T) + phi Q dt / C         (END-of-step pool)
        B   = C (T' - T) - Q dt                       (what the pool loses)
        T_p' = [C_p T_p + sum(C a T - (phi-1) Q dt) - Q_o dt] / [C_p + sum(C a)]
    Backward Euler on the pool: stable for any dt (an explicit pool coupling
    blows up above about 2 x 21 s). g = 0 gives a = 0, phi = 1, no division.
    Returns (T_new, T_p_new, B). Sum(C (T' - T)) + C_p (T_p' - T_p) = sum(Q) dt
    - Q_o dt to rounding."""
    T = np.asarray(T, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if eps is not None:
        g = g * (1.0 - np.asarray(eps, dtype=np.float64))
    x = g * dt / C
    a, phi = _phi(x)
    Qdt = Q * dt
    num = C_p * T_p + np.sum(C * a * T - (phi - 1.0) * Qdt) - Q_o * dt
    den = C_p + np.sum(C * a)
    Tp1 = num / den
    Tn = T + a * (Tp1 - T) + phi * Qdt / C
    B = C * (Tn - T) - Qdt
    return Tn, float(Tp1), B


DT_MAX_ADVANCE = 20.0                    # s: longest single backward-Euler pool step `pennes_advance` takes


def pennes_advance(T, C, g, Q, dt, T_p, C_p, Q_o=0.0, eps=None, dt_max=DT_MAX_ADVANCE):
    """`pennes_step` over a long `dt` (the night skip, a slow tick) with the
    sources held constant, cut into equal steps of at most `dt_max`.

    One backward-Euler pool step is stable for ANY dt but first-order: with a
    -100 W cooling step the core error after one 600 s step is about 0.12 K,
    after 30 steps of 20 s about 0.006 K (measured, test_codeonly_*). Energy
    closes exactly at every step, so it closes over the whole advance.
    Returns (T_new, T_p_new, B_total): B_total is the blood exchange summed
    over the steps."""
    n = max(1, int(math.ceil(dt / dt_max)))
    h = dt / n
    T = np.asarray(T, dtype=np.float64)
    Btot = np.zeros_like(T)
    for _ in range(n):
        T, T_p, B = pennes_step(T, C, g, Q, h, T_p, C_p, Q_o, eps)
        Btot = Btot + B
    return T, T_p, Btot
