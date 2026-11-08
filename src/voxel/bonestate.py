"""BONE STATE: records that carry load (wave 13, T10, lane A-T10b).

A bone is a record: {integrity, angle, gap_m, callus, splint, peak_N} (and, once
it is broken, the structural `member` that limits it). A bone carries load by
its cross-section and breaks by stress. A broken shin bears no load because its
record says so, not because of a flag.

THE ONE CAPACITY FUNCTION
  `capacity(rec, name, sheet)` returns (F_fail, M_fail): the failure load and
  the bending capacity of this person's bone as it is NOW. Every other law asks
  it: `check` (does it break), `bears` (does it carry), and `impact` (the
  referred capacities of the fuse chain are built from `break_force`, which is
  this function seen through the elliptic rule). So a bone breaks at ONE force
  whichever module asks, and a test holds the two equal.

THE LAWS
  capacity   F_fail = table value x bone_scale x max(integrity, residual)
             M_fail = sigma_b Z x bone_scale x max(integrity, residual)
             residual = 0.02 + 0.5 splint + 0.98 callus^3  (capped at 1)
  break rule (F / F_fail)^2 + (M_extra / M_fail)^2 >= 1, M_extra a LATERAL
             moment the caller supplies (a blow's bending, a lever); zero for a
             plain axial load. There is no hidden eccentricity.
  healing    strain eps = delta / gap, delta = peak_N / k_fx,
             k_fx = (E A / L) min(1, integrity + callus + splint + 0.01): the
             part of the section that is still whole carries the load too, so
             a partial break is stiffer than a full one and heals by the SAME
             callus law (integrity is 0 for a full break); callus grows
             d(c)/dt = (1 - c) / tau while eps <= 0.10, and stands still above
             it (non-union). One exact step for any dt: eps falls as the callus
             grows, so the window test at the start decides the whole step.
  gait       leg ratio = min(1, cap / (1.2 W)): walk, limp (pace divided by
             0.4 + 0.6 ratio), hop (under 0.3), crawl (both legs).

Pure functions on plain dicts. No World. Numbers marked (u) are unverified.
"""
import math

try:
    from . import anatomy as _AN
except ImportError:                    # pragma: no cover - the module is part of this lane
    _AN = None

G = 9.81
RESIDUAL_BASE = 0.02          # a broken shin offers 2 % of its capacity
RESIDUAL_SPLINT = 0.5
RESIDUAL_CALLUS = 0.98
CALLUS_POWER = 3.0            # S(c) = c^3 (c^2 returned weight bearing at 3 weeks)
EPS_WINDOW = 0.10             # secondary healing window: strain up to 10 %
K_FX_FLOOR = 0.01             # stiffness of the fibrous gap, as a fraction of intact
GAP_MIN_M = 0.5e-3            # least gap of a broken bone
GAP_SPAN_M = 2.5e-3           # so a full crush has 3 mm
ANGLE_RAD = 0.2               # angle at integrity 0 (u)
WEEK_S = 7.0 * 86400.0

# Load classes for a time skip, in body weights on the bone (u). bonestate.peak_for.
# EACH LIMB HAS ITS OWN (decision N7). LOAD_BW is the LEG's (the T10 spec's numbers, kept under
# their old name). An arm bone at rest does not carry a leg's load: it passes what its limb
# weighs. The arm numbers are the weights of the segments, Winter's mass fractions (Winter,
# Biomechanics and Motor Control of Human Movement, 4th ed., 2009, Table 4.1; anatomy.WINTER):
#   lying and sitting  0.022 = forearm 0.016 + hand 0.006 (the arm rests on the bed or the lap:
#                      at most the forearm and hand hang off the bone)
#   standing           0.050 = upper arm 0.028 + forearm 0.016 + hand 0.006 (the whole arm hangs)
# These are (u): a bound from the segment weights, not a measurement of load in the bone (in vivo
# shoulder loads exist, Bergmann et al., J Biomech 40(10):2139-2149, 2007, doi
# 10.1016/j.jbiomech.2006.10.037, but are not used). The tall heavy man whose full forearm break did
# not heal at the leg's 0.05 now has a strain of 0.047 against the 0.10 window. In a seeded sweep of
# 3000 random builds (35 to 150 kg, 1.45 to 2.05 m, every arm member, full break) 0 leave the window
# at 0.022 (worst strain 0.062) and 473 leave it at 0.05 (worst 0.141).
# The torso and the head are not limbs: they keep the leg's classes as before (u, not reviewed).
LOAD_BW = {"lying": 0.05, "sitting": 0.1, "standing": 0.5}
ARM_LOAD_BW = {"lying": 0.022, "sitting": 0.022, "standing": 0.050}
LIMB_LOAD_BW = {"leg": LOAD_BW, "arm": ARM_LOAD_BW}
LIMB_OF_PART = {"shin": "leg", "thigh": "leg", "upper arm": "arm", "forearm": "arm"}
STANDING_ONE_LEG_BW = 1.0
WALK_BW = 1.2
HOP_BW = 2.0
RUN_BW = 2.5
CRAWL_BW = 0.25

LIMP_FROM = 0.3               # leg ratio under this: hop on the other leg
WALK_LOAD_BW = 1.2            # a leg must carry 1.2 BW to walk

UNITED = 0.999               # callus at which the break counts as healed


def _member_data():
    if _AN is None:                    # pragma: no cover
        raise ImportError("bonestate needs anatomy")
    return _AN.MEMBERS


def member_of(name, rec=None, member=None):
    """The structural member that limits `name`: the argument, else the record's
    own (a broken calcaneus), else the plain bone's default (a shin is a tibia).
    A name that is itself a member ('femur') means that member."""
    mem = _member_data()
    if member is not None:
        return member
    if rec is not None and rec.get("member"):
        return rec["member"]
    if name in mem:
        return name
    m = _AN.MEMBER_OF_BONE.get(name)
    if m is None:
        raise KeyError(f"no structural member for {name!r}")
    return m


def new_bone(sheet=None, name=None):
    """A fresh, intact bone record. (The sheet and name are for the symmetry of
    the interface: an intact record is the same for everybody, the person is in
    the sheet that `capacity` reads.)"""
    return {"integrity": 1.0, "angle": 0.0, "gap_m": 0.0, "callus": 0.0,
            "splint": 0.0, "peak_N": 0.0}


def residual(rec):
    """What a broken bone still carries as a fraction of intact:
    0.02 + 0.5 splint + 0.98 callus^3, at most 1."""
    c = min(max(rec.get("callus", 0.0), 0.0), 1.0)
    s = min(max(rec.get("splint", 0.0), 0.0), 1.0)
    return min(1.0, RESIDUAL_BASE + RESIDUAL_SPLINT * s + RESIDUAL_CALLUS * c ** CALLUS_POWER)


def strength_factor(rec):
    """max(integrity, residual): the part of the intact capacity the record keeps."""
    return max(min(max(rec.get("integrity", 1.0), 0.0), 1.0), residual(rec))


def capacity(rec, name, sheet, member=None):
    """(F_fail_N, M_fail_Nm) of this bone now. THE ONE capacity function.
    Both scale with the person's bone_scale; both fall with the record's
    strength factor (intact: 1)."""
    m = member_of(name, rec, member)
    row = _member_data()[m]
    bs = sheet.get("bone_scale", 1.0) if sheet else 1.0
    k = bs * strength_factor(rec if rec is not None else {})
    return row["F_FAIL_N"] * k, _AN.m_fail_adult(m) * k


def break_force(rec, name, sheet, lateral_Nm=0.0, dyn=1.0, m_above=1.0, member=None):
    """The contact force at which this bone breaks, N: the axial load on the bone
    at which the elliptic rule reaches 1, referred to the contact (divide by the
    fraction of body mass the bone carries) and with the dynamic strengthening
    `dyn` on the bone. Zero when the lateral moment alone is enough. `check`
    flips at exactly this load; `impact` builds its chain from it."""
    F, M = capacity(rec, name, sheet, member)
    F, M = F * dyn, M * dyn
    if M > 0.0:
        r = lateral_Nm / M
        if r >= 1.0:
            return 0.0
        F *= math.sqrt(1.0 - r * r)
    return F / m_above


def check(rec, name, sheet, load_N, lateral_Nm=0.0, dyn=1.0, member=None):
    """True when the bone breaks: (F / F_fail)^2 + (M_extra / M_fail)^2 >= 1. The
    load is the axial load ON THE BONE (a contact force is multiplied by the
    mass fraction above it first). `dyn` strengthens both capacities."""
    F, M = capacity(rec, name, sheet, member)
    F, M = F * dyn, M * dyn
    x = (load_N / F) ** 2 if F > 0.0 else math.inf
    y = (lateral_Nm / M) ** 2 if M > 0.0 else (math.inf if lateral_Nm > 0.0 else 0.0)
    return x + y >= 1.0


def break_(rec, name, sheet, load_N, crush_frac, member=None, side=1.0):
    """Record a break. `crush_frac` is the part of the stroke the bone used (1:
    crushed through, 0.37: a partial compression fracture). integrity falls to
    1 - crush_frac (kept if already lower), gap_m is 0.5 mm plus 2.5 mm of the
    crush, angle 0.2 rad x (1 - integrity) toward the load side `side` (+1 or
    -1), and peak_N keeps the largest load seen. A second break of another
    member of the same bone leaves the more limiting member named. Returns the
    record (mutated)."""
    if crush_frac <= 0.0:
        rec["peak_N"] = max(rec.get("peak_N", 0.0), load_N)
        return rec
    crush_frac = min(crush_frac, 1.0)
    mem = member_of(name, rec, member)
    new_int = 1.0 - crush_frac
    old = rec.get("integrity", 1.0)
    if old < 1.0 and rec.get("member") and rec["member"] != mem:
        mine = _member_data()[mem]["F_FAIL_N"] * new_int
        theirs = _member_data()[rec["member"]]["F_FAIL_N"] * old
        if mine < theirs:
            rec["member"] = mem
        new_int = min(new_int, old)
    else:
        rec["member"] = mem
        new_int = min(new_int, old)
    rec["integrity"] = new_int
    rec["gap_m"] = max(rec.get("gap_m", 0.0), GAP_MIN_M + GAP_SPAN_M * (1.0 - new_int))
    rec["angle"] = math.copysign(ANGLE_RAD * (1.0 - new_int), side)
    rec["peak_N"] = max(rec.get("peak_N", 0.0), load_N)
    return rec


def is_broken(rec):
    """True while the bone is broken and not yet carrying its intact load."""
    return rec.get("integrity", 1.0) < 1.0 and residual(rec) < 1.0 - 1e-9 \
        and strength_factor(rec) < 1.0 - 1e-9


def cap_N(rec, name, sheet, member=None):
    """The axial load this bone carries now: F_fail x bone_scale x
    max(integrity, residual)."""
    return capacity(rec, name, sheet, member)[0]


def bears(rec, name, sheet, N, member=None):
    """Can the bone take an axial load of N newtons? The footing law asks this
    before it counts a sole as firm."""
    return cap_N(rec, name, sheet, member) >= N


def leg_ratio(bones, side, sheet, weight_N):
    """Gait capacity of one leg, 0..1: min(1, cap / (1.2 W)) of the weaker of
    its shin and thigh records (a missing record is intact). `bones` is the
    body's {bone name: record}."""
    caps = []
    for part in ("shin", "thigh"):
        nm = f"{side} {part}"
        rec = (bones or {}).get(nm)
        if rec is None:
            continue
        caps.append(cap_N(rec, nm, sheet))
    if not caps:
        return 1.0
    return min(1.0, min(caps) / (WALK_LOAD_BW * weight_N))


def pace_factor(ratio):
    """Pace divided by this for a limping leg: 0.4 + 0.6 ratio; a ratio of 1 walks
    at full pace."""
    return 1.0 / (0.4 + 0.6 * min(max(ratio, 0.0), 1.0)) if ratio < 1.0 else 1.0


def gait_class(left, right):
    """walk, limp, hop or crawl from the two leg ratios: both full is a walk;
    one between 0.3 and 1 a limp; one under 0.3 with the other able a hop on the
    other; both under 0.3 a crawl on arms and knees. (One leg under 0.3 and the
    other between 0.3 and 1 is a hop.)"""
    lo, hi = min(left, right), max(left, right)
    if lo >= 1.0:
        return "walk"
    if lo >= LIMP_FROM:
        return "limp"
    if hi >= LIMP_FROM:
        return "hop"
    return "crawl"


def tau_s(rec, name, member=None):
    """Healing time constant of this bone, seconds (weeks by the member's table)."""
    m = member_of(name, rec, member)
    return _member_data()[m]["tau_wk"] * WEEK_S


def k_fx(rec, name, sheet, member=None):
    """Axial stiffness of the fracture, N/m: (E A / L) min(1, integrity + callus +
    splint + 0.01). The unbroken part of a partial fracture is in parallel with
    the callus and the splint; a full break (integrity 0) has none of it."""
    m = member_of(name, rec, member)
    row = _member_data()[m]
    bs = sheet.get("bone_scale", 1.0) if sheet else 1.0
    h = (sheet["height_m"] / _AN.REF_HEIGHT_M) if sheet else 1.0
    EA_L = _AN.E_BONE_PA * _AN.section_area_m2(m) * bs / (row["L_m"] * h)
    whole = min(max(rec.get("integrity", 1.0), 0.0), 1.0)
    return EA_L * min(1.0, whole + rec.get("callus", 0.0) + rec.get("splint", 0.0) + K_FX_FLOOR)


def strain(rec, name, sheet, peak_N, member=None):
    """Interfragmentary strain eps = delta / gap, delta = peak_N / k_fx. A bone
    with no gap (an intact one) has none."""
    gap = rec.get("gap_m", 0.0)
    if gap <= 0.0:
        return 0.0
    return (peak_N / k_fx(rec, name, sheet, member)) / gap


def heal_step(rec, name, sheet, dt, peak_N=None, member=None):
    """Advance healing by dt seconds (one hour per schedule step, or a whole
    skip: the step is exact for any dt). peak_N is the largest load the bone saw
    since the last step (default: the record's own peak_N). Callus grows while
    the strain is inside the 10 % window and stands still above it
    (non-union). Resets peak_N. Returns the record."""
    if rec.get("integrity", 1.0) >= 1.0 and rec.get("gap_m", 0.0) <= 0.0:
        return rec
    peak = rec.get("peak_N", 0.0) if peak_N is None else peak_N
    eps = strain(rec, name, sheet, peak, member)
    rec["strain"] = eps
    rec["nonunion"] = eps > EPS_WINDOW
    if eps <= EPS_WINDOW:
        tau = tau_s(rec, name, member)
        c = rec.get("callus", 0.0)
        rec["callus"] = 1.0 - (1.0 - c) * math.exp(-dt / tau)
        if rec["callus"] >= UNITED:            # united: the record is intact again
            rec["integrity"], rec["gap_m"], rec["nonunion"] = 1.0, 0.0, False
            rec.pop("member", None)             # a united bone is a plain bone again
            rec["angle"] = 0.0
    rec["peak_N"] = 0.0
    return rec


def splint_fraction(EI_plank, EI_bone):
    """The share of the bending stiffness a plank lashed along the bone adds:
    EI_s / (EI_s + EI_b) (u form)."""
    return EI_plank / (EI_plank + EI_bone)


def bone_EI(name, sheet, member=None):
    """E I of this person's bone, N m^2 (cortical modulus 17 GPa; sections scale
    with bone_scale^2 so that strength scales with bone_scale)."""
    m = member_of(name, None, member)
    bs = sheet.get("bone_scale", 1.0) if sheet else 1.0
    return _AN.E_BONE_PA * _AN.moment_of_inertia_m4(m) * bs ** 2


def limb_of(bone):
    """"leg" or "arm" for a limb bone name ("left shin", "right upper arm"); None for a bone that
    is not a limb ("torso", "head") and for None: those keep the leg's load classes."""
    if bone is None:
        return None
    part = bone.split(" ", 1)[1] if bone.startswith(("left ", "right ")) else bone
    return LIMB_OF_PART.get(part)


def peak_for(posture, gait=None, mass_kg=70.0, ratio=1.0, legs=2, bone=None):
    """The peak load class of a person for a time skip, N on one bone.
    Legs (and any bone given no name): lying 0.05 BW, sitting 0.1, standing 0.5 per leg (1.0 on
    one leg), walking 1.2 x the gait ratio, hopping 2.0, running 2.5, crawling 0.25. A gait wins
    over a posture. `bone` ("left forearm") picks the bone's own limb: an ARM bone has its own
    classes (lying and sitting 0.022 BW, standing 0.05: LIMB_LOAD_BW) and does not take the leg's
    gait loads (when the legs walk, hop or run the arm hangs: its standing class; crawling is
    0.25 BW on every limb). The torso and the head keep the leg's classes."""
    W = mass_kg * G
    limb = limb_of(bone) or "leg"
    cls = LIMB_LOAD_BW[limb]
    if gait == "crawl":
        return CRAWL_BW * W
    if gait is not None and limb == "arm":
        return cls["standing"] * W
    if gait in ("walk", "limp"):
        return WALK_BW * ratio * W
    if gait == "hop":
        return HOP_BW * W
    if gait == "run":
        return RUN_BW * W
    if posture == "standing" and limb == "leg" and legs < 2:
        return STANDING_ONE_LEG_BW * W
    return cls[posture] * W


def reduce(rec):
    """Traction (an act that hurts): the angle goes to zero and the gap to its
    least. Returns the record."""
    rec["angle"] = 0.0
    if rec.get("gap_m", 0.0) > 0.0:
        rec["gap_m"] = GAP_MIN_M
    return rec
