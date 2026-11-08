"""ANATOMY: the data tables behind injury (wave 13, T10, lane A-T10b).

A pure data module. No World, no lattice, no random numbers. It answers
"what is in a body, how strong is it, and where does it sit" for the bone,
fall and bleeding laws in `bonestate`, `impact` and `vasculature`.

WHAT IS IN IT
  draw(pid, name, key, lo, hi)   a deterministic value that is this person's
                                 own: the crc32 hash `reflex._draw` uses, so
                                 the same person gets the same number from
                                 both modules.
  make_sheet(...)                the person's numbers: T7's fields (sex,
                                 mass_kg, height_m, fat_frac, age_y,
                                 bmr_scale) plus bone_scale, muscle_scale,
                                 pain_scale, ko_scale, bsa_m2. It wraps
                                 `bodysheet` (G10); the sex factor 0.90 is
                                 applied ONCE, on top of the size rule.
  REGIONS (11)                   head, chest, abdomen, and for each side
                                 upper arm, forearm, thigh, shin. The hand
                                 rides in the forearm, the foot in the shin,
                                 the neck in the head, and the torso bone is
                                 split in the tables: chest (upper 55 % of
                                 its height) and abdomen with the pelvis.
  REGION_OF_ZONE                 the 14 body-heat zones of T7 -> these regions
  MEMBERS                        the structures that carry load or break
                                 (calcaneus, tibia, femur, pelvis, lumbar
                                 spine, skull, ...): failure load, bone
                                 geometry, crush stroke, healing time.
  VESSELS, ORGANS                the cuttable vessels and the internal organs
  SKIN_LAYERS                    the 8-node skin profile of the burn law
  WINTER                         segment mass fractions (sum 1.0)

Numbers marked (u) are independent but unverified; `sourced_bone_loads()`
reads tests/data/sourced_numbers.json (lane A-S) and says which of them a
source now backs, with the status it saw.
"""
import json
import math
import os
import zlib

try:
    from . import bodysheet as _BS
except ImportError:                    # pragma: no cover - stand-in below
    _BS = None

G = 9.81                          # m/s2

# ── the one hash draw ────────────────────────────────────────────────────────


def draw(pid, name, key, lo, hi):
    """A value in [lo, hi) that is this person's own: crc32 of key, pid and
    name. Identical to `reflex._draw` when the person carries no value of the
    key. No random numbers: same person, same answer, in any order."""
    u = (zlib.crc32(f"{key}|{pid}|{name}".encode()) & 0xFFFFFFFF) / 2.0 ** 32
    return float(lo + (hi - lo) * u)


# ── regions ──────────────────────────────────────────────────────────────────
_SIDES = ("left", "right")
_LIMB_REGIONS = ("upper arm", "forearm", "thigh", "shin")
REGIONS = ("head", "chest", "abdomen") + tuple(
    f"{s} {p}" for s in _SIDES for p in _LIMB_REGIONS)
assert len(REGIONS) == 11

# The scene builder's bone each region lives in (the torso is one bone).
BONE_OF_REGION = {r: ("torso" if r in ("chest", "abdomen") else r) for r in REGIONS}

TORSO_SPLIT = 0.55            # the chest is the upper 55 % of the torso's height

# The 14 zones of T7 (tissue.ZONES). The hand rides in the forearm bone and
# the foot in the shin bone. The torso zone is ONE zone: it maps to the chest;
# `region_of_zone(zone, along)` splits it by height when the caller knows it.
_ZONES = ("head", "torso") + tuple(
    f"{s} {p}" for s in _SIDES
    for p in ("upper arm", "forearm", "hand", "thigh", "shin", "foot"))
REGION_OF_ZONE = {"head": "head", "torso": "chest"}
for _s in _SIDES:
    for _p, _r in (("upper arm", "upper arm"), ("forearm", "forearm"),
                   ("hand", "forearm"), ("thigh", "thigh"),
                   ("shin", "shin"), ("foot", "shin")):
        REGION_OF_ZONE[f"{_s} {_p}"] = f"{_s} {_r}"
ZONES = _ZONES


def region_of_zone(zone, along=None):
    """The region of a body-heat zone. `along` (0 at the pelvis, 1 at the
    shoulders) splits the torso zone: above 1 - TORSO_SPLIT is the chest, the
    rest the abdomen. For every other zone `along` is ignored."""
    if zone == "torso" and along is not None:
        return "chest" if along >= 1.0 - TORSO_SPLIT else "abdomen"
    return REGION_OF_ZONE[zone]


# ── Winter 1990 segment fractions ────────────────────────────────────────────
# One segment (limbs are per side). `bodysheet.WINTER` when present.
_WINTER_LOCAL = {"head": 0.081, "trunk": 0.497, "upper arm": 0.028,
                 "forearm": 0.016, "hand": 0.006, "thigh": 0.100,
                 "shank": 0.0465, "foot": 0.0145}


def _winter():
    if _BS is not None:
        try:
            w = dict(_BS.WINTER)
            if all(k in w for k in _WINTER_LOCAL):
                return {k: float(w[k]) for k in _WINTER_LOCAL}
        except (AttributeError, TypeError):
            pass
    return dict(_WINTER_LOCAL)


WINTER = _winter()


def _region_fraction_table():
    w = WINTER
    t = {"head": w["head"],
         "chest": w["trunk"] * TORSO_SPLIT,
         "abdomen": w["trunk"] * (1.0 - TORSO_SPLIT)}
    for s in _SIDES:
        t[f"{s} upper arm"] = w["upper arm"]
        t[f"{s} forearm"] = w["forearm"] + w["hand"]
        t[f"{s} thigh"] = w["thigh"]
        t[f"{s} shin"] = w["shank"] + w["foot"]
    return t


REGION_FRAC = _region_fraction_table()          # sums to 1.0 over the 11 regions


def segment_mass(sheet, region):
    """Mass of one region in kg (Winter). 'thigh' with no side means one side;
    'torso' is chest plus abdomen."""
    if region == "torso":
        return sheet["mass_kg"] * (REGION_FRAC["chest"] + REGION_FRAC["abdomen"])
    if region in REGION_FRAC:
        return sheet["mass_kg"] * REGION_FRAC[region]
    return sheet["mass_kg"] * REGION_FRAC["left " + region]


# ── Du Bois ──────────────────────────────────────────────────────────────────
def bsa_dubois(mass_kg, height_cm):
    """Body surface area, Du Bois and Du Bois 1916: 0.007184 W^0.425 H^0.725
    (W kg, H cm). 70 kg and 175 cm give 1.85 m2."""
    return 0.007184 * mass_kg ** 0.425 * height_cm ** 0.725


# ── the sheet ────────────────────────────────────────────────────────────────
SEX_FACTOR = {"male": 1.0, "female": 0.90}      # bone, applied ONCE
AGE_BONE_START_Y = 50.0
AGE_BONE_PER_DECADE = 0.05
BONE_REF_KG = 70.0
DRAW_RANGES = {"bone_scale": (0.92, 1.08), "pain_scale": (0.8, 1.2),
               "ko_scale": (0.85, 1.15)}
REF_HEIGHT_M = 1.75


def _sex(s):
    s = str(s).strip().lower()
    return "female" if s in ("f", "female", "woman") else "male"


def _bone_scale(sheet, d):
    """bodysheet.bone_scale when present, else the same rule locally:
    (m/70)^(2/3) x sex factor x minus 5 % a decade over 50 x the draw."""
    if _BS is not None:
        try:
            return float(_BS.bone_scale(sheet, d))
        except (AttributeError, KeyError, TypeError):
            pass
    size = (sheet["mass_kg"] / BONE_REF_KG) ** (2.0 / 3.0)
    sex = SEX_FACTOR.get(sheet.get("sex", "male"), 1.0)
    over = max(0.0, sheet.get("age_y", 30.0) - AGE_BONE_START_Y) / 10.0
    return size * sex * max(0.0, 1.0 - AGE_BONE_PER_DECADE * over) * d


def make_sheet(sheet_in=None, mass_kg=None, height_m=None, age_y=30, sex="m",
               pid=0, name="", draws=True):
    """One person's injury sheet, a plain dict.

    `sheet_in` is T7's p["sheet"] ({sex, mass_kg, height_m, fat_frac, age_y,
    bmr_scale}); when it is given its numbers win. Otherwise mass and height
    come from the arguments, and with neither the person is the stock 43.9 kg
    figure (`bodysheet.STOCK`). A missing height scales by the cube root of
    mass from the 70 kg, 1.75 m man.

    Added here (T10 owns them): bone_scale = (m/70)^(2/3) x sex_factor (0.90
    for a woman, ONCE) x age x draw(0.92-1.08); pain_scale draw 0.8-1.2;
    ko_scale draw 0.85-1.15; muscle_scale 1.0; bsa_m2 (Du Bois). With
    `draws=False` every draw is 1.0 (the neutral person, for tables and
    tests)."""
    if sheet_in is not None:
        s = dict(sheet_in)
        s["sex"] = _sex(s.get("sex", sex))
        s.setdefault("age_y", float(age_y))
        s.setdefault("fat_frac", 0.18)
        s.setdefault("bmr_scale", 1.0)
        if "mass_kg" not in s or "height_m" not in s:
            raise KeyError("a sheet needs mass_kg and height_m")
    elif mass_kg is None and height_m is None:
        base = (_BS.get_sheet("stock") if _BS is not None
                else {"name": "stock", "sex": "male", "mass_kg": 43.9, "height_m": 1.5,
                      "fat_frac": 0.15, "age_y": 30.0, "bmr_scale": 1.0})
        s = dict(base)
        s["sex"] = _sex(sex) if sex != "m" else s["sex"]
        s["age_y"] = float(age_y)
    else:
        m = float(mass_kg if mass_kg is not None else BONE_REF_KG * (height_m / REF_HEIGHT_M) ** 3)
        h = float(height_m if height_m is not None else REF_HEIGHT_M * (m / BONE_REF_KG) ** (1.0 / 3.0))
        s = {"name": name or "person", "sex": _sex(sex), "mass_kg": m, "height_m": h,
             "fat_frac": 0.18, "age_y": float(age_y), "bmr_scale": 1.0}
    s["mass_kg"] = float(s["mass_kg"])
    s["height_m"] = float(s["height_m"])
    d = {k: (draw(pid, name, k, lo, hi) if draws else 1.0)
         for k, (lo, hi) in sorted(DRAW_RANGES.items())}
    s["bone_scale"] = _bone_scale(s, d["bone_scale"])
    s["pain_scale"] = d["pain_scale"]
    s["ko_scale"] = d["ko_scale"]
    s["muscle_scale"] = float(s.get("muscle_scale", 1.0))
    s["bsa_m2"] = bsa_dubois(s["mass_kg"], s["height_m"] * 100.0)
    return s


# ── structural members: what breaks ──────────────────────────────────────────
# F_FAIL_N is the table's FAILURE LOAD under typical loading (it already holds
# the usual small eccentricity: published tolerances such as calcaneus 6.8 kN
# and lumbar 6-8 kN are failure loads) for ONE such structure, adult 70 kg
# (bone_scale 1.0). All (u) unless `sourced_bone_loads()` says otherwise.
#   Do, Di  outer and inner diameter of the shaft, mm (None: no shaft bending
#           model; M_fail then comes from F_fail x span / 4)
#   area_mm2 (members with no shaft model) load-bearing bone area for the
#           healing stiffness, mm2 (u)
#   n_par   structures in parallel inside ONE record (the forearm pair)
#   L_m     the span a blow bends the bone over (3-point bend), m
#   crush_m the stroke it yields over once it gives way (for the chain)
#   tau_wk  secondary-healing time constant, weeks (u)
#   pool_ml blood that pools at infinity when it breaks (ATLS ranges, u)
#   region  where the member lives (a bone name of the scene builder)
SIGMA_B_PA = 180e6               # cortical bone bending strength (Reilly and Burstein 1975)
E_BONE_PA = 17e9                 # cortical bone modulus
DYN = 1.25                       # loading-rate strengthening on all bone capacities (u)

MEMBERS = {
    "calcaneus":  dict(region="shin", F_FAIL_N=6500.0, Do=None, Di=None, area_mm2=1200.0, n_par=1, L_m=0.08,
                       crush_m=0.020, tau_wk=20.0, pool_ml=150.0),
    "tibia":      dict(region="shin", F_FAIL_N=8500.0, Do=27.0, Di=17.0, n_par=1, L_m=0.37,
                       crush_m=0.030, tau_wk=20.0, pool_ml=750.0),
    "femur":      dict(region="thigh", F_FAIL_N=12000.0, Do=28.0, Di=14.0, n_par=1, L_m=0.45,
                       crush_m=0.030, tau_wk=18.0, pool_ml=1500.0),
    "femoral_neck": dict(region="thigh", F_FAIL_N=3500.0, Do=None, Di=None, area_mm2=700.0, n_par=1, L_m=0.10,
                       crush_m=0.010, tau_wk=18.0, pool_ml=1000.0),
    "pelvis":     dict(region="torso", F_FAIL_N=5500.0, Do=None, Di=None, area_mm2=600.0, n_par=1, L_m=0.30,
                       crush_m=0.040, tau_wk=16.0, pool_ml=2500.0),
    "lumbar":     dict(region="torso", F_FAIL_N=8000.0, Do=None, Di=None, area_mm2=1700.0, n_par=1, L_m=0.15,
                       crush_m=0.015, tau_wk=16.0, pool_ml=500.0),
    "ribs":       dict(region="torso", F_FAIL_N=3000.0, Do=None, Di=None, area_mm2=300.0, n_par=1, L_m=0.20,
                       crush_m=0.020, tau_wk=8.0, pool_ml=125.0),
    "humerus":    dict(region="upper arm", F_FAIL_N=6000.0, Do=22.0, Di=12.0, n_par=1, L_m=0.31,
                       crush_m=0.025, tau_wk=12.0, pool_ml=600.0),
    "clavicle":   dict(region="upper arm", F_FAIL_N=2500.0, Do=None, Di=None, area_mm2=150.0, n_par=1, L_m=0.15,
                       crush_m=0.010, tau_wk=8.0, pool_ml=150.0),
    "forearm":    dict(region="forearm", F_FAIL_N=2800.0, Do=13.0, Di=7.0, n_par=2, L_m=0.25,
                       crush_m=0.020, tau_wk=10.0, pool_ml=400.0),
    "radius_distal": dict(region="forearm", F_FAIL_N=2800.0, Do=None, Di=None, area_mm2=350.0, n_par=1, L_m=0.06,
                       crush_m=0.010, tau_wk=10.0, pool_ml=200.0),
    "skull":      dict(region="head", F_FAIL_N=5000.0, Do=None, Di=None, area_mm2=600.0, n_par=1, L_m=0.18,
                       crush_m=0.008, tau_wk=16.0, pool_ml=150.0),
}

# The member a plain bone name means when its record names none.
MEMBER_OF_BONE = {"head": "skull", "torso": "lumbar"}
for _s in _SIDES:
    MEMBER_OF_BONE[f"{_s} shin"] = "tibia"
    MEMBER_OF_BONE[f"{_s} thigh"] = "femur"
    MEMBER_OF_BONE[f"{_s} upper arm"] = "humerus"
    MEMBER_OF_BONE[f"{_s} forearm"] = "forearm"
MEMBER_OF_BONE.update({"chest": "ribs", "abdomen": "pelvis"})

# Healing time by bone kind (weeks; the member's tau_wk is the same table).
TAU_WEEKS = {"shin": 20.0, "thigh": 18.0, "upper arm": 12.0, "forearm": 10.0, "head": 16.0}

# Mass above each member that decelerates with the body (everything above it
# rides down on it). Winter fractions: the feet carry nearly all, the spine the
# trunk, head and arms.
M_ABOVE = {"calcaneus": 0.99, "tibia": 0.96, "femur": 0.90, "pelvis": 0.68,
           "lumbar": 0.62, "organ_feet": 0.50}


def m_above(x):
    """Fraction of the body mass carried by member (or region) x: the table
    value for a member, the value of the region's main member for a region
    ('shin' -> tibia, 'thigh' -> femur, 'abdomen' -> pelvis, 'chest' -> lumbar)."""
    if x in M_ABOVE:
        return M_ABOVE[x]
    key = MEMBER_OF_BONE.get(x)
    if key is None:
        key = MEMBER_OF_BONE.get("left " + x)
    return M_ABOVE.get(key, 1.0)


def member_geometry(member, sheet=None):
    """(Do_m, Di_m, L_m) of a member for this person, or None for a member with
    no shaft model. Diameters scale with bone_scale^(1/2) (strength follows the
    section), the span with the height."""
    mem = MEMBERS[member]
    if mem["Do"] is None:
        return None
    s = 1.0
    h = 1.0
    if sheet is not None:
        s = math.sqrt(sheet.get("bone_scale", 1.0))
        h = sheet["height_m"] / REF_HEIGHT_M
    return (mem["Do"] * 1e-3 * s, mem["Di"] * 1e-3 * s, mem["L_m"] * h)


def section_modulus_m3(member):
    """Z = pi (Do^4 - Di^4) / (32 Do) of the adult shaft, m^3 (times n_par);
    None when the member has no shaft model."""
    mem = MEMBERS[member]
    if mem["Do"] is None:
        return None
    Do, Di = mem["Do"] * 1e-3, mem["Di"] * 1e-3
    return mem["n_par"] * math.pi * (Do ** 4 - Di ** 4) / (32.0 * Do)


def section_area_m2(member):
    """Cortical section area pi/4 (Do^2 - Di^2) of the adult shaft, m^2 (times
    n_par). For a member with no shaft model: its load-bearing bone area
    `area_mm2` (u; orders from anatomy texts: a lumbar body's end plate is about
    1700 mm2, a calcaneal section 1200, a femoral neck 700, a clavicle shaft 150,
    the rib group 300). The old F_fail / 100 MPa rule gave a lumbar 80 mm2, which
    made the healing stiffness 20x too soft and left a broken lumbar, pelvis,
    rib, skull or neck a non-union even at bed rest."""
    mem = MEMBERS[member]
    if mem["Do"] is None:
        return mem["area_mm2"] * 1e-6
    Do, Di = mem["Do"] * 1e-3, mem["Di"] * 1e-3
    return mem["n_par"] * math.pi / 4.0 * (Do ** 2 - Di ** 2)


def moment_of_inertia_m4(member):
    """I = pi/64 (Do^4 - Di^4) of the adult shaft (times n_par), for the splint
    stiffness. A member with no shaft model gets the I that would give its
    M_fail at sigma_b, so a splint ratio still has an answer."""
    mem = MEMBERS[member]
    if mem["Do"] is None:
        Z = mem["F_FAIL_N"] * mem["L_m"] / 4.0 / SIGMA_B_PA
        return Z * 0.014
    Do, Di = mem["Do"] * 1e-3, mem["Di"] * 1e-3
    return mem["n_par"] * math.pi / 64.0 * (Do ** 4 - Di ** 4)


def m_fail_adult(member):
    """Bending capacity M_fail = sigma_b Z, N m, of the adult member (bone_scale
    1). A member with no shaft model: F_fail x span / 4, the bending moment
    that equals its failure load as a 3-point bend."""
    Z = section_modulus_m3(member)
    if Z is None:
        mem = MEMBERS[member]
        return mem["F_FAIL_N"] * mem["L_m"] / 4.0
    return SIGMA_B_PA * Z


# ── vessels ──────────────────────────────────────────────────────────────────
# One row per vessel. d_mm and depth_mm are for the adult. az is degrees around
# the limb in the BONE'S OWN FRAME (0 anterior in anatomical position, 90 toward
# the midline); hw its half-width. kind a (artery) or v (vein). Q is the flow of
# an open end at rest pressure, ml/s. (u): orders of magnitude from anatomy texts.
# D_ADULT_MM is the adult limb thickness the depth fraction refers to.
D_ADULT_MM = {"thigh": 150.0, "shin": 100.0, "upper arm": 90.0, "forearm": 70.0,
              "head": 180.0, "chest": 280.0, "abdomen": 260.0}


def _v(name, region, kind, d, depth, az, hw, q):
    return dict(name=name, region=region, kind=kind, d_mm=d, depth_mm=depth,
                az=az, hw=hw, Q_ml_s=q)


VESSELS = (
    _v("femoral artery", "thigh", "a", 8.0, 30.0, 60.0, 20.0, 15.0),
    _v("femoral vein", "thigh", "v", 10.0, 35.0, 75.0, 20.0, 6.0),
    _v("great saphenous", "thigh", "v", 4.0, 5.0, 90.0, 25.0, 1.5),
    _v("posterior tibial", "shin", "a", 4.0, 20.0, 160.0, 20.0, 4.0),
    _v("anterior tibial", "shin", "a", 3.0, 25.0, 340.0, 20.0, 2.5),
    _v("saphenous vein", "shin", "v", 3.0, 4.0, 90.0, 20.0, 0.8),
    _v("brachial artery", "upper arm", "a", 4.5, 15.0, 100.0, 20.0, 6.0),
    _v("basilic vein", "upper arm", "v", 4.0, 4.0, 100.0, 20.0, 1.0),
    _v("radial artery", "forearm", "a", 2.5, 6.0, 30.0, 15.0, 2.0),
    _v("ulnar artery", "forearm", "a", 2.5, 8.0, 60.0, 15.0, 1.8),
    _v("cephalic vein", "forearm", "v", 3.0, 3.0, 0.0, 25.0, 0.8),
    _v("carotid artery", "head", "a", 6.0, 15.0, 90.0, 20.0, 12.0),
    _v("jugular vein", "head", "v", 10.0, 8.0, 80.0, 20.0, 5.0),
    _v("scalp arteries", "head", "a", 1.5, 3.0, 0.0, 180.0, 0.8),
    _v("intercostal artery", "chest", "a", 2.0, 10.0, 0.0, 180.0, 1.5),
)
VESSEL_OF = {v["name"]: v for v in VESSELS}

# Structures that are not vessels but are cut by the same wound arc.
TENDONS = (dict(name="flexor tendons", region="forearm", depth_mm_lo=8.0,
                depth_mm_hi=25.0, az=45.0, hw=45.0),)

# Internal organs: only an impact reaches them. Q_ml_s is the flow of a full
# rupture at rest pressure; low is a low-pressure organ (lung).
ORGANS = {
    "liver":  dict(region="abdomen", Q_ml_s=8.0, kind="a", d_mm=6.0),
    "spleen": dict(region="abdomen", Q_ml_s=5.0, kind="a", d_mm=5.0),
    "lung":   dict(region="chest", Q_ml_s=1.5, kind="v", d_mm=6.0),
    "aorta":  dict(region="chest", Q_ml_s=30.0, kind="a", d_mm=22.0),
    "heart":  dict(region="chest", Q_ml_s=25.0, kind="a", d_mm=20.0),
}

# Blood that pools at infinity into a closed cavity, ml, adult (ATLS ranges;
# midpoints, u). The pelvis is the abdomen's.
CAVITY_ML = {"chest": 125.0, "abdomen": 2500.0, "head": 150.0}
for _s in _SIDES:
    CAVITY_ML[f"{_s} upper arm"] = 600.0
    CAVITY_ML[f"{_s} forearm"] = 400.0
    CAVITY_ML[f"{_s} thigh"] = 1500.0
    CAVITY_ML[f"{_s} shin"] = 750.0

# ── skin (the burn law's profile) ────────────────────────────────────────────
# Depths to layer bottoms in mm, with the layer's properties (Torvi and Dale
# 1994, u): k W/mK, rho*c J/m3K, and the Pennes perfusion term w_b rho_b c_b
# W/m3K toward arterial 37 C. 8 nodes.
SKIN_LAYERS = (
    # name,         bottom_mm, k,     rho_c,  perfusion
    ("epi_top",       0.05, 0.255, 3.96e6, 0.0),
    ("basal",         0.10, 0.255, 3.96e6, 0.0),
    ("dermis_1",      0.40, 0.523, 3.96e6, 4750.0),
    ("dermis_2",      1.00, 0.523, 3.96e6, 4750.0),
    ("dermis_3",      1.50, 0.523, 3.96e6, 4750.0),
    ("dermis_base",   2.00, 0.523, 3.96e6, 4750.0),
    ("subcutis_1",    4.00, 0.167, 2.50e6, 1500.0),
    ("subcutis_2",   10.00, 0.167, 2.50e6, 1500.0),
)


# ── sourced numbers (lane A-S) ───────────────────────────────────────────────
DEFAULT_SOURCED = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tests", "data", "sourced_numbers.json")


def sourced_bone_loads(path=None, age_y=30.0, sex="male"):
    """What tests/data/sourced_numbers.json says about the bone failure loads.
    Returns {member: {"value_N", "status", "kind", "table_N"}} for the members a
    source covers, and {} when the file is absent or unreadable. status is the
    file's own ('sourced', or 'not_found' which leaves the member out).
    kind is 'bound' for a lower bound (the femur: failure load exceeds 12 kN)
    or 'value'. The lumbar value is the Jager compilation's regression for this
    age and sex (kN: male 10.43 - 0.923 per decade, female 7.65 - 0.685)."""
    out = {}
    try:
        with open(path or DEFAULT_SOURCED) as f:
            doc = json.load(f)
        ent = doc.get("entries", doc)
    except (OSError, ValueError, TypeError, AttributeError):
        return out
    try:
        e = ent.get("bone.femur_fracture_load")
        if e and e.get("status") == "sourced" and e.get("value"):
            v = float(e["value"]["proximal_femur_failure_kN_greater_than"]) * 1e3
            out["femur"] = dict(value_N=v, status="sourced", kind="bound",
                                table_N=MEMBERS["femur"]["F_FAIL_N"])
        e = ent.get("bone.vertebra_fracture_load")
        if e and e.get("status") == "sourced" and e.get("value"):
            a, b = (10.43, 0.923) if _sex(sex) == "male" else (7.65, 0.685)
            v = (a - b * (age_y / 10.0)) * 1e3
            out["lumbar"] = dict(value_N=v, status="sourced", kind="value",
                                 table_N=MEMBERS["lumbar"]["F_FAIL_N"])
        for key, member in (("bone.tibia_fracture_load", "tibia"),
                            ("bone.calcaneus_fracture_load", "calcaneus")):
            e = ent.get(key)
            if e and e.get("status") == "sourced" and e.get("value") is not None:
                out[member] = dict(value_N=float(e["value"]) * 1e3, status="sourced",
                                   kind="value", table_N=MEMBERS[member]["F_FAIL_N"])
    except (KeyError, TypeError, ValueError, AttributeError):
        pass
    return out
