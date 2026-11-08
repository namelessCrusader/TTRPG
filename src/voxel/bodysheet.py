"""BODY SHEET: the one source of body numbers (wave 13, lane G10).

A pure data module. No World, no lattice, no random numbers, and no imports
from the rest of the engine. Everything that needs "how heavy is a thigh" or
"how hard can this person push" asks here, so there is one answer.

WHAT IS IN IT
  * Sheets: plain dicts {name, sex, mass_kg, height_m, fat_frac, age_y,
    bmr_scale}. The same keys as p["sheet"] in the body-heat topic (T7), so
    `tissue.sheet` and `anatomy.make_sheet` can wrap `get_sheet` through a
    try/except ImportError and fall back to their own stand-in.
      ADULT_MALE    73 kg, 1.75 m, fat 0.18, age 30  (Mifflin-St Jeor check in
                    the T7 spec uses this man)
      ADULT_FEMALE  60 kg, 1.62 m, fat 0.27, age 30
      STOCK         the figure `scenes._person` builds TODAY: 43.9 kg and 1.5 m
                    (396 flesh cells at 5 cm; 43.875 kg counted, stored as
                    43.9). It also keeps today's flat strength_N of 400, so a
                    scene that asks the stock sheet for strength gets what
                    BODY["strength_N"] gives now.
  * Winter's segment mass fractions (D. A. Winter, Biomechanics and Motor
    Control of Human Movement, 4th ed., table 4.1): head and neck 8.1 %, trunk
    49.7 %, upper arm 2.8 %, forearm 1.6 %, hand 0.6 %, thigh 10 %, shank
    4.65 %, foot 1.45 %. Limbs are per side, so the sum is
    8.1 + 49.7 + 2 x (2.8 + 1.6 + 0.6 + 10 + 4.65 + 1.45) = 100.0 %.
  * bone_scale, strength_N, grip_N, kick mass and foot speed, a default
    per-person skill record, and `adult_layout(voxel_cm)` (bone lengths and
    radii in metres for a scene builder).

WHICH FIGURE SHIPS
  Wave 13 ships on the STOCK 43.9 kg figure unless the user takes decision D7
  (adult default). Nothing in this module changes a scene by being imported;
  a scene is only different when a later step builds it from ADULT_MALE or
  ADULT_FEMALE, and that step is D7's. The tables in the injury topic that are
  calibrated for the adult say so, and the stock figure shifts every fracture
  height by 1 to 2 m.

NUMBERS THAT ARE NOT SOURCED YET
  Tagged (u) in the comments: independent but unverified. The two kick numbers
  are stored as (u); `kick_numbers()` replaces them by the sourced values the
  moment tests/data/sourced_numbers.json (lane A-S) carries them and says
  which status it used.

PLAIN-ENGLISH CHOICES (each is data you can change)
  * The sex factor 0.90 is applied ONCE, on top of the size rule
    (m / 70) ** (2/3). The earlier 0.75 double-counted size.
  * strength_N is mu_foot * m * g, because a person cannot push harder than
    the foot grips: mu_foot is 400 / (68 g) = 0.5996, so 68 kg gives exactly
    400 N. mu_foot in the literature is 0.5 to 0.9 (u).
  * Body density: the world's FLESH row is 1000 kg/m3, so the layout is sized
    so that volume x 1000 = sheet mass. (Living tissue is nearer 1010 to
    1060; the lattice uses its own number everywhere and so does this.)
"""
import json
import math
import os
import zlib

G = 9.81                      # m/s2 (materials.GRAVITY; copied, pure module)
FLESH_KG_M3 = 1000.0          # materials.py FLESH row; the lattice's number

# ── Winter 1990 segment mass fractions (percent of total body mass) ──────────
# value is for ONE segment; "sided" ones appear on both sides.
WINTER_PCT = {
    "head": 8.1,          # head and neck
    "trunk": 49.7,
    "upper arm": 2.8,
    "forearm": 1.6,
    "hand": 0.6,
    "thigh": 10.0,
    "shank": 4.65,
    "foot": 1.45,
}
SIDED = ("upper arm", "forearm", "hand", "thigh", "shank", "foot")
CENTRAL = ("head", "trunk")
WINTER = {k: v / 100.0 for k, v in WINTER_PCT.items()}   # fractions

# Names the rest of the engine uses for the same piece.
ALIAS = {"torso": "trunk", "shin": "shank", "head and neck": "head",
         "neck": "head", "upperarm": "upper arm", "upper_arm": "upper arm"}

# The 9 (11 with sides) bones scenes._person names. The hand rides in the
# forearm bone, the foot in the shin bone and the neck in the head.
BONE_PARTS = {
    "head": ("head",),
    "torso": ("trunk",),
    "upper arm": ("upper arm",),
    "forearm": ("forearm", "hand"),
    "thigh": ("thigh",),
    "shin": ("shank", "foot"),
}

# Segment length as a fraction of standing height (Drillis and Contini 1966,
# as tabulated in Winter table 4.1) (u until A-S marks it sourced).
LENGTH_FRAC = {
    "upper arm": 0.186, "forearm": 0.146, "hand": 0.108,
    "thigh": 0.245, "shank": 0.246, "foot_len": 0.152,
    "ankle_z": 0.039, "shoulder_z": 0.818,
}

SEX_FACTOR = {"male": 1.0, "female": 0.90}     # bone, applied once
AGE_BONE_START_Y = 50.0                        # bone loss starts here (u)
AGE_BONE_PER_DECADE = 0.05                     # minus 5 % a decade (u)
BONE_REF_KG = 70.0

STRENGTH_REF_N = 400.0        # BODY["strength_N"]: an unremarkable adult
STRENGTH_REF_KG = 68.0        # ... of this mass (T11 spec: 68 kg at mu 0.6)
MU_FOOT = STRENGTH_REF_N / (STRENGTH_REF_KG * G)   # 0.5996 (0.5 to 0.9 (u))
OFF_HAND = 0.8                # BODY["off_hand"]: what the other hand is worth
MUSCLE_STRESS_PA = 63e3       # (u) realism-map G10: muscle force per section

# (u) effective striker for a kick at strength 400 N: 5.6 kg x 6 m/s = 100 J.
# Replaced by sourced values when lane A-S has them (see kick_numbers).
KICK_REF = {"mass_kg": 5.6, "speed_ms": 6.0}
KICK_EXP = {"mass_kg": 1.0, "speed_ms": 1.0}   # both scale with strength/400

DEFAULT_SOURCED = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tests", "data", "sourced_numbers.json")


def _sheet(name, sex, mass_kg, height_m, fat_frac, age_y, **extra):
    d = {"name": name, "sex": sex, "mass_kg": float(mass_kg),
         "height_m": float(height_m), "fat_frac": float(fat_frac),
         "age_y": float(age_y), "bmr_scale": 1.0}
    d.update(extra)
    return d


ADULT_MALE = _sheet("adult male", "male", 73.0, 1.75, 0.18, 30.0)
ADULT_FEMALE = _sheet("adult female", "female", 60.0, 1.62, 0.27, 30.0)
# strength_N 400 is today's flat figure: the stock scenes keep it.
STOCK = _sheet("stock", "male", 43.9, 1.5, 0.15, 30.0, strength_N=STRENGTH_REF_N)

SHEETS = {"adult male": ADULT_MALE, "male": ADULT_MALE,
          "adult female": ADULT_FEMALE, "female": ADULT_FEMALE,
          "stock": STOCK}


def get_sheet(name="stock", **override):
    """A COPY of a named sheet (stock, adult male, adult female), with any key
    overridden. Never the stored dict, so a caller cannot change the table."""
    d = dict(SHEETS[name])
    d.update(override)
    return d


def sheet_of(p):
    """The sheet of person `p`: p["sheet"] if the person has one, else the
    stock figure (what a hand-built body is today)."""
    s = p.get("sheet") if p else None
    return s if s else get_sheet("stock")


def bsa_m2(sheet):
    """Body surface area, Du Bois and Du Bois 1916: 0.007184 m^0.425 h_cm^0.725.
    70 kg, 175 cm gives 1.85."""
    return 0.007184 * sheet["mass_kg"] ** 0.425 * (sheet["height_m"] * 100.0) ** 0.725


# ── segment masses ───────────────────────────────────────────────────────────
def _norm(segment):
    """Split a name into (side or None, segment). Accepts 'left thigh',
    'right_shin', 'torso', 'thigh'; unsided limbs mean ONE side."""
    s = segment.replace("_", " ").strip().lower()
    side = None
    for pre in ("left ", "right "):
        if s.startswith(pre):
            side, s = pre.strip(), s[len(pre):]
    s = ALIAS.get(s, s)
    if s not in WINTER:
        raise KeyError(f"no body segment {segment!r}")
    if side is not None and s in CENTRAL:
        raise KeyError(f"{s} has no side ({segment!r})")
    return side, s


def segment_mass(sheet, segment):
    """Mass in kg of one segment of this person. A limb segment is ONE side's
    mass ('left thigh', or just 'thigh' for either). Winter fractions."""
    _, s = _norm(segment)
    return sheet["mass_kg"] * WINTER[s]


def segment_masses(sheet):
    """Every segment, both sides: {'head': kg, 'trunk': kg, 'left thigh': kg,
    ...}. Sums to the sheet mass (to float rounding)."""
    out = {k: segment_mass(sheet, k) for k in CENTRAL}
    for k in SIDED:
        for side in ("left", "right"):
            out[f"{side} {k}"] = segment_mass(sheet, k)
    return out


def bone_mass(sheet, bone):
    """Mass in kg of what scenes._person calls a bone: 'forearm' carries the
    hand and 'shin' the foot, 'head' the neck. Sided names ('left shin') and
    unsided names both mean one bone."""
    s = bone.replace("_", " ").strip().lower()
    for pre in ("left ", "right "):
        if s.startswith(pre):
            s = s[len(pre):]
    s = ALIAS.get(s, s)
    if s == "trunk":
        s = "torso"
    if s == "shank":
        s = "shin"
    return sum(segment_mass(sheet, part) for part in BONE_PARTS[s])


# ── bone scale ───────────────────────────────────────────────────────────────
def bone_scale(sheet, draw=1.0):
    """How strong this person's bones are, against the 70 kg man (1.0):
    (m / 70) ** (2/3), times the sex factor ONCE (female 0.90), times minus 5 %
    a decade over 50, times the person's own draw (0.92 to 1.08, T10 makes it).
    The stock 43.9 kg figure gives 0.732 (its tables quote 0.73)."""
    size = (sheet["mass_kg"] / BONE_REF_KG) ** (2.0 / 3.0)
    sex = SEX_FACTOR.get(sheet.get("sex", "male"), 1.0)
    over = max(0.0, sheet.get("age_y", 30.0) - AGE_BONE_START_Y) / 10.0
    age = max(0.0, 1.0 - AGE_BONE_PER_DECADE * over)
    return size * sex * age * float(draw)


# ── strength ─────────────────────────────────────────────────────────────────
def strength_N(sheet):
    """The most force this person can put on a thing, rested and unhurt: what
    the foot can grip, mu_foot * m * g. A sheet that carries its own
    "strength_N" (the stock figure's flat 400, or a rescue scene's 900) keeps
    it. 68 kg gives 400 N. A "muscle_scale" key (default 1) multiplies."""
    own = sheet.get("strength_N")
    if own is not None:
        return float(own)
    return MU_FOOT * sheet["mass_kg"] * G * float(sheet.get("muscle_scale", 1.0))


def grip_N(sheet, dominant=True):
    """What one hand can hold: strength_N for the dominant hand, 0.8 of it for
    the other (BODY["off_hand"], the same rule as skeleton._hand_strength)."""
    s = strength_N(sheet)
    return s if dominant else s * OFF_HAND


def kick_numbers(path=None):
    """(mass_kg, speed_ms, status) for a kick at 400 N. status is 'u' when the
    values are the stored unsourced ones, or the status string found in the
    sourced-numbers file (lane A-S) when it carries them. Never raises."""
    mass, speed, status = KICK_REF["mass_kg"], KICK_REF["speed_ms"], "u"
    try:
        with open(path or DEFAULT_SOURCED) as f:
            doc = json.load(f)
        got_m, got_v = _find(doc, "kick_mass_kg"), _find(doc, "kick_speed_ms")
        if got_m is not None and got_v is not None:
            mass, speed = got_m[0], got_v[0]
            status = got_m[1] if got_m[1] == got_v[1] else "mixed"
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return float(mass), float(speed), status


def _find(doc, key):
    """(value, status) for `key` anywhere in a nested sourced-numbers file, or
    None. An entry is a bare number (status 'sourced') or a dict with a
    'value' and a 'status' or 'tag'."""
    if isinstance(doc, dict):
        if key in doc:
            e = doc[key]
            if isinstance(e, dict) and "value" in e:
                return float(e["value"]), str(e.get("status", e.get("tag", "sourced")))
            if isinstance(e, (int, float)):
                return float(e), "sourced"
        for v in doc.values():
            r = _find(v, key)
            if r is not None:
                return r
    elif isinstance(doc, list):
        for v in doc:
            if isinstance(v, dict) and v.get("name", v.get("id")) == key and "value" in v:
                return float(v["value"]), str(v.get("status", v.get("tag", "sourced")))
            r = _find(v, key)
            if r is not None:
                return r
    return None


def kick_mass_kg(sheet, path=None):
    """Effective mass of the striking foot and leg, kg: 5.6 at 400 N, scaled
    by strength / 400 (the T11 rule; the exponent is data)."""
    mass = kick_numbers(path)[0]
    return mass * (strength_N(sheet) / STRENGTH_REF_N) ** KICK_EXP["mass_kg"]


def kick_speed_ms(sheet, path=None):
    """Foot speed at contact, m/s: 6 at 400 N, scaled by strength / 400."""
    speed = kick_numbers(path)[1]
    return speed * (strength_N(sheet) / STRENGTH_REF_N) ** KICK_EXP["speed_ms"]


def kick_energy_J(sheet, path=None):
    """1/2 m v^2 of the kick: 100 J for the 400 N person."""
    return 0.5 * kick_mass_kg(sheet, path) * kick_speed_ms(sheet, path) ** 2


# ── skill ────────────────────────────────────────────────────────────────────
# A skill is a level 0..1 for an act. Everybody has a default for every act,
# drawn from a hash of (act, pid, name) like reflex._draw: same person, same
# answer, no random numbers, and no dependence on dict or call order. A
# stored p["skill"][act] always wins. All ranges are (u) starting points.
SKILL_RANGE = {
    "operate": (0.70, 0.95),     # latch, bolt, handle: everyday
    "kick": (0.40, 0.80),
    "carry": (0.40, 0.80),
    "throw": (0.30, 0.70),
    "climb": (0.30, 0.70),
    "swim": (0.20, 0.80),
    "pick": (0.00, 0.15),        # a lock: a trained trade, so low by default
}
SKILL_OTHER = (0.20, 0.60)       # an act the table does not know


def _u(key, p):
    """A number in [0, 1) from the crc32 of key, pid and name."""
    return (zlib.crc32(f"{key}|{p.get('pid')}|{p.get('name')}".encode())
            & 0xFFFFFFFF) / 2.0 ** 32


def skill_default(p, act):
    """The drawn default level of person `p` at `act`. Does not touch p."""
    lo, hi = SKILL_RANGE.get(act, SKILL_OTHER)
    return float(lo + (hi - lo) * _u(f"skill|{act}", p))


def skill_of(p, act):
    """p["skill"][act] if set, else the drawn default. Read only."""
    v = (p.get("skill") or {}).get(act)
    return float(v) if v is not None else skill_default(p, act)


def ensure_skill(p, act=None):
    """Make the record exist: p["skill"][act] for one act, or for every act in
    the table (sorted, so the dict is built the same way every time). Returns
    p["skill"]. Existing entries are kept."""
    rec = p.setdefault("skill", {})
    for a in ([act] if act is not None else sorted(SKILL_RANGE)):
        if a not in rec:
            rec[a] = skill_default(p, a)
    return rec


# ── layout for a scene builder ───────────────────────────────────────────────
# Body frame, metres: x across (left is -x, as in the stock builder), y front,
# z up from the floor, origin at the midline between the feet. Arms hang at
# the sides, the foot points along +y.
TRUNK_DEPTH_OVER_WIDTH = 0.7     # the trunk section's y-radius / x-radius (u)
FOOT_HEEL_M = 0.06               # heel behind the ankle (u)
MAX_CELLS = 60                   # widest section the box search tries


def _section(vol_m3, nz, v, aspect):
    """The whole-cell section (nx, ny) of a block nz cells tall that holds at
    least `vol_m3` of flesh at 1 g/cm3 in as few cells as possible and with
    nx / ny near `aspect`. Returns (nx, ny, frac) with frac = true volume /
    box volume, 0 < frac <= 1, which is the partial fill that keeps the mass."""
    need = vol_m3 / (nz * v ** 3)               # section area, in cells
    best = None
    for nx in range(1, MAX_CELLS + 1):
        ny = max(1, math.ceil(need / nx - 1e-9))
        n = nx * ny
        score = (n / need - 1.0) + 0.3 * abs(math.log((nx / ny) / aspect))
        if best is None or score < best[0] - 1e-12:
            best = (score, nx, ny)
    _, nx, ny = best
    return nx, ny, need / (nx * ny)


def adult_layout(voxel_cm=5.0, sheet=None):
    """Bone lengths and radii of a person, in metres, and the cell boxes to
    draw them, for a scene builder. Pure numbers.

    `sheet` defaults to ADULT_MALE. Every segment is a straight piece with an
    elliptic section (radius rx across, ry front-to-back; for the foot, which
    lies along y, ry is the radius in z), and Winter's mass: volume x 1000 kg/m3
    is the segment's mass, so the true shapes weigh the sheet.

    The lattice has only whole cells, and a forearm is thinner than one at
    10 cm. So each segment also comes as a BOX of whole cells with a partial
    fill `frac` (what `fill(frac=...)` takes) that gives the box exactly the
    segment's mass, the same trick the stock builder uses with 0.85 and 0.9.
    The box has the nearest whole-cell length and the fewest cells that hold
    the volume. Lengths round to whole cells, so the drawn height differs from
    the sheet's by up to a cell or two ("height_drawn_m").

    Returns {"voxel_cm", "height_m", "height_drawn_m", "mass_kg", "segments",
    "joints", "bones"}:
      segments  {name: {axis, a, b, length, rx, ry, mass_kg, cells, frac}};
                a and b are the centres of the end faces (x, y, z) in metres;
                cells = (x0, x1, y0, y1, z0, z1) are cell indices (half open)
                with x = 0 and y = 0 on a cell boundary at the midline, z = 0
                the floor; names are 'head', 'trunk', 'left thigh' ...
      joints    {name: (x, y, z)} metres: shoulder, elbow, wrist, hip, knee,
                ankle of each side
      bones     {name: length_m} for the stock builder's bones (the forearm
                bone includes the hand, the shin bone the foot)
    No two boxes share a cell.
    """
    sh = dict(ADULT_MALE if sheet is None else sheet)
    v = float(voxel_cm) / 100.0
    H = sh["height_m"]
    f = LENGTH_FRAC
    rho = FLESH_KG_M3
    m = {k: sh["mass_kg"] * w for k, w in WINTER.items()}
    vol = {k: x / rho for k, x in m.items()}
    z_ank = f["ankle_z"] * H
    z_knee = z_ank + f["shank"] * H
    z_hip = z_knee + f["thigh"] * H
    z_sh = f["shoulder_z"] * H
    L = {k: f[k] * H for k in ("upper arm", "forearm", "hand", "thigh", "shank")}
    L["foot"] = f["foot_len"] * H
    L["trunk"] = z_sh - z_hip
    L["head"] = H - z_sh

    # true shapes, metres
    rx, ry = {}, {}
    for k in ("head", "upper arm", "forearm", "hand", "thigh", "shank", "foot"):
        rx[k] = ry[k] = math.sqrt(vol[k] / (math.pi * L[k]))
    rx["trunk"] = math.sqrt(vol["trunk"] / (math.pi * L["trunk"] * TRUNK_DEPTH_OVER_WIDTH))
    ry["trunk"] = TRUNK_DEPTH_OVER_WIDTH * rx["trunk"]

    # heights in whole cells
    c_ank = max(1, round(z_ank / v))
    c_knee = c_ank + max(1, round(L["shank"] / v))
    c_hip = c_knee + max(1, round(L["thigh"] / v))
    c_sh = c_hip + max(1, round(L["trunk"] / v))
    c_top = c_sh + max(1, round(L["head"] / v))
    c_ua = max(1, round(L["upper arm"] / v))
    c_fa = max(1, round(L["forearm"] / v))
    c_hn = max(1, round(L["hand"] / v))
    c_ft = max(1, round(L["foot"] / v))
    heel = max(1, round(FOOT_HEEL_M / v))

    cells, frac = {}, {}

    def box(name, key, nz, aspect, x_in=None, x_side=0):
        """Size the box of segment `name` from the volume of `key`. x_in is the
        cell its INNER edge sits on (side -1 puts the box to the left of it, 1
        to the right); with no x_in the box is centred on the midline."""
        nx, ny, fr = _section(vol[key], nz, v, aspect)
        if x_in is None:
            x0 = -(nx // 2)
        elif x_side < 0:
            x0 = x_in - nx
        else:
            x0 = x_in
        y0 = -(ny // 2)
        cells[name] = [x0, x0 + nx, y0, y0 + ny, None, None]
        frac[name] = fr
        return nx

    box("trunk", "trunk", c_sh - c_hip, 1.0 / TRUNK_DEPTH_OVER_WIDTH)
    cells["trunk"][4:] = [c_hip, c_sh]
    box("head", "head", c_top - c_sh, 1.0)
    cells["head"][4:] = [c_sh, c_top]
    leg_out = 0
    for side, sg in (("left", -1), ("right", 1)):
        for part, key, nz, z0 in (("thigh", "thigh", c_hip - c_knee, c_knee),
                                  ("shank", "shank", c_knee - c_ank, c_ank)):
            nx = box(f"{side} {part}", key, nz, 1.0, x_in=0, x_side=sg)
            cells[f"{side} {part}"][4:] = [z0, z0 + nz]
            leg_out = max(leg_out, nx)
        # the foot: heel behind the ankle, as wide as its volume asks
        n_len = c_ft                                          # cells along y
        need = vol["foot"] / (n_len * v ** 3)                 # section cells, x by z
        nx = max(1, math.ceil(need / c_ank - 1e-9))
        frac[f"{side} foot"] = need / (nx * c_ank)
        x0 = -nx if sg < 0 else 0
        cells[f"{side} foot"] = [x0, x0 + nx, -heel, n_len - heel, 0, c_ank]
        leg_out = max(leg_out, nx)
    tr = cells["trunk"]
    arm_in = {-1: min(tr[0], -leg_out), 1: max(tr[1], leg_out)}
    for side, sg in (("left", -1), ("right", 1)):
        z = c_sh
        for part, nz in (("upper arm", c_ua), ("forearm", c_fa), ("hand", c_hn)):
            box(f"{side} {part}", part, nz, 1.0, x_in=arm_in[sg], x_side=sg)
            cells[f"{side} {part}"][4:] = [z - nz, z]
            z -= nz

    # true geometry in metres
    r_ua = max(rx["upper arm"], rx["forearm"], rx["hand"])
    x_arm = rx["trunk"] + r_ua
    x_leg = max(rx["thigh"], rx["shank"])
    segs, joints = {}, {}

    def add(name, key, axis, a, b):
        segs[name] = {"axis": axis, "a": tuple(a), "b": tuple(b),
                      "length": L[key], "rx": rx[key], "ry": ry[key],
                      "mass_kg": m[key], "cells": tuple(cells[name]), "frac": frac[name]}

    add("trunk", "trunk", "z", (0, 0, z_hip), (0, 0, z_sh))
    add("head", "head", "z", (0, 0, z_sh), (0, 0, H))
    for side, sg in (("left", -1.0), ("right", 1.0)):
        xa, xl = sg * x_arm, sg * x_leg
        z1, z2 = z_sh - L["upper arm"], z_sh - L["upper arm"] - L["forearm"]
        z3 = z2 - L["hand"]
        add(f"{side} upper arm", "upper arm", "z", (xa, 0, z_sh), (xa, 0, z1))
        add(f"{side} forearm", "forearm", "z", (xa, 0, z1), (xa, 0, z2))
        add(f"{side} hand", "hand", "z", (xa, 0, z2), (xa, 0, z3))
        add(f"{side} thigh", "thigh", "z", (xl, 0, z_hip), (xl, 0, z_knee))
        add(f"{side} shank", "shank", "z", (xl, 0, z_knee), (xl, 0, z_ank))
        rf = ry["foot"]
        add(f"{side} foot", "foot", "y", (xl, -FOOT_HEEL_M, rf),
            (xl, L["foot"] - FOOT_HEEL_M, rf))
        joints.update({f"{side} shoulder": (xa, 0.0, z_sh), f"{side} elbow": (xa, 0.0, z1),
                       f"{side} wrist": (xa, 0.0, z2), f"{side} hip": (xl, 0.0, z_hip),
                       f"{side} knee": (xl, 0.0, z_knee), f"{side} ankle": (xl, 0.0, z_ank)})
    bones = {"head": L["head"], "torso": L["trunk"], "upper arm": L["upper arm"],
             "forearm": L["forearm"] + L["hand"], "thigh": L["thigh"],
             "shin": L["shank"] + z_ank}
    return {"voxel_cm": float(voxel_cm), "height_m": H, "height_drawn_m": c_top * v,
            "mass_kg": sh["mass_kg"], "segments": segs, "joints": joints, "bones": bones}
