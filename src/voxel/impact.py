"""IMPACT: the fuse chain for falls, thrown bodies and blows (wave 13, T10, lane A-T10b).

One law for a fall, a thrown body, a blow and a kick. A body stops against a
contact. The stopping system is a set of ELEMENTS IN SERIES, each with a force
at which it gives way (capacity C_i referred to the contact force), a stroke it
yields over (crush c_i), and what happens when it gives way. A yielding element
holds the force at C_i while it crushes. The contact and the stiff body are a
spring k_el that loads between yields.

THE BALANCE, with gravity's work during the stroke. W = m g. The body falls
through h, then goes on descending by the whole stroke X = F/k_el + (the crush
used). Work by the contact force = KE at contact + W X, so
    (F - W)^2 / (2 k_el) + sum over yielded elements (C_i - W) c_i
                                         = W h + W^2 / (2 k_el)
Three closed-form stages in a loop over the elements sorted by capacity:
  1. ELASTIC: F = W + sqrt(2 k (E - plateau sum)), while the energy is below the
     next element's yield.
  2. PLATEAU: if the energy falls inside the step of element i, F = C_i exactly
     and element i used crush (E - E_before) / (C_i - W): a PARTIAL failure.
  3. PAST: element i is crushed through; go on to the next.
An element with C_i <= W cannot hold the body: it has already failed (a bone
broken by standing). No bisection.

EVERY JOULE HAS A SINK. The work in, W (h + X), is booked to: the elastic
spring (tissue and contact cells; the contusion share), each bone's crush heat,
each organ's, and the control element's work (the legs flexing, heat in the leg
muscle). What a contact does not take stays as the body's motion for the next.

REFERRED CAPACITY. C_i = 1.25 x (sum over the loaded bones of the bone's break
force) / m_above_i, the break force from bonestate (THE ONE capacity function),
because everything above a structure decelerates with the body. Both feet use
both calcanei in parallel; a one-foot or tilted landing uses the feet that carry
load and scales k_el by the same share.

The MODE TABLES (feet, side, back, shoulder, head, arms) are DATA. If the
random-orientation targets are not met (survivors average 5.2 m, LD50 12-15 m),
the table is retuned, not the law. `fit_report()` prints the shares.

Pure functions on plain dicts. No World. Numbers marked (u) are unverified.
"""
import math

try:
    from . import anatomy as _AN
except ImportError:                    # pragma: no cover
    _AN = None
try:
    from . import bonestate as _BN
except ImportError:                    # pragma: no cover
    _BN = None

G = 9.81

# ── the control element: the muscles brake the body ──────────────────────────
PREP_T0_S = 0.10               # airborne time before the legs get ready
PREP_RAMP_S = 0.20             # and the time it takes them
CTL_CAP_X = 1.5                # eccentric strength is 1.5 x the legs' push
CTL_STROKE_X = 1.4             # stroke is 1.4 x the crouch depth
SOFT_CAP_BW = 4.0              # an unprepared body keeps heel, ankle and soft tissue:
SOFT_STROKE_M = 0.04           # 4 BW over 4 cm
LEGS_N_70 = 2060.0             # legs' push of the 70 kg adult (u)
CROUCH_M_175 = 0.286           # crouch depth of the 1.75 m adult (u)
REF_KG = 70.0

# ── structure tables (data) ──────────────────────────────────────────────────
# An element: name; member (bonestate/anatomy structural member) or organ F_N;
# bones (the records it lives in; {S} is the side the landing code gives);
# n (structures of the member in parallel inside the element); m_above (the
# fraction of the body mass that decelerates through it); crush_m.
_PAIR_SHIN = ("left shin", "right shin")
_PAIR_THIGH = ("left thigh", "right thigh")


def _el(name, member, bones, m_above, n=1, kind="bone", crush=None, F_N=None, organs=()):
    return dict(name=name, member=member, bones=bones, m_above=m_above, n=n, kind=kind,
                crush=crush, F_N=F_N, organs=organs)


MODES = {
    # Landing on the feet: the best orientation, the long stroke.
    "feet": dict(
        k_el_70=70e3, ctl="legs", f_E=None, pairs=True,
        elements=(
            _el("calcaneus", "calcaneus", _PAIR_SHIN, 0.99),
            _el("tibia", "tibia", _PAIR_SHIN, 0.96),
            _el("femur", "femur", _PAIR_THIGH, 0.90),
            _el("pelvis", "pelvis", ("torso",), 0.68, n=2),
            _el("lumbar", "lumbar", ("torso",), 0.62),
            _el("chest organs", None, ("torso",), 0.50, kind="organ", crush=0.10,
                F_N=24000.0, organs=("heart", "aorta", "liver")),
        )),
    # Landing on the hip or side. The contact energy is a share of the drop
    # (f_E: aware, limp) unless the landing code gives the effective-mass energy.
    "side": dict(
        k_el_70=50e3, ctl=None, f_E=(0.15, 0.35), pairs=False,
        elements=(
            _el("femoral neck", "femoral_neck", ("{S} thigh",), 1.0),
            _el("ribs lateral", "ribs", ("torso",), 1.0),
            _el("pelvis lateral", "pelvis", ("torso",), 1.0, crush=0.020),
            _el("side organs", None, ("torso",), 1.0, kind="organ", crush=0.06,
                F_N=8000.0, organs=("liver", "spleen", "lung")),
        )),
    "back": dict(
        k_el_70=60e3, ctl=None, f_E=(0.20, 0.40), pairs=False,
        elements=(
            _el("pelvis", "pelvis", ("torso",), 1.0, n=2),
            _el("lumbar", "lumbar", ("torso",), 0.62),
            _el("ribs posterior", "ribs", ("torso",), 1.0, n=2),
            _el("back organs", None, ("torso",), 1.0, kind="organ", crush=0.06,
                F_N=10000.0, organs=("spleen", "lung", "aorta")),
        )),
    "shoulder": dict(
        k_el_70=40e3, ctl=None, f_E=(0.12, 0.25), pairs=False,
        elements=(
            _el("clavicle", "clavicle", ("{S} upper arm",), 1.0),
            _el("humerus", "humerus", ("{S} upper arm",), 1.0),
            _el("ribs lateral", "ribs", ("torso",), 1.0),
        )),
    # Landing on the outstretched hands (FOOSH). The arm is its own control element.
    "arms": dict(
        k_el_70=30e3, ctl="arms", f_E=(0.30, 0.30), pairs=True,
        elements=(
            _el("distal radius", "radius_distal", ("left forearm", "right forearm"), 1.0),
            _el("clavicle", "clavicle", ("left upper arm", "right upper arm"), 1.0),
            _el("humerus", "humerus", ("left upper arm", "right upper arm"), 1.0),
        )),
    # Landing on the head: a small effective mass, a short stiff stroke.
    "head": dict(
        k_el=500e3, ctl=None, f_E=None, pairs=False, head=True,
        elements=(_el("skull", "skull", ("head",), 1.0),)),
}
HEAD_MASS_KG_70 = 4.5          # the head itself (u)
NECK_LINK_X = 0.3              # and this share of the trunk, through the neck (u)
ARM_STROKE_X = 0.2             # the arms' control stroke: this share of arm length
ARM_SOFT_BW = 2.0
ARM_SOFT_STROKE_M = 0.02
ROLL_ORDER = ("feet", "side", "back", "shoulder")

# ── the head: acceleration, HIC, knock-out ───────────────────────────────────
KO_G = 80.0                    # knock-out at this peak head acceleration (u) ...
KO_T_S = 0.003                 # ... for this long
T_KO_S = 20.0                  # knock-out time at the threshold, seconds; goes as (a/KO)^2
T_KO_MAX_S = 1800.0
HIC_T_MAX_S = 0.015            # HIC15
HIC_SKULL = 1000.0             # skull fracture risk
HIC_FATAL = 2500.0             # fatal brain injury (u)

# ── what kills (the outcome of a landing, no treatment) ──────────────────────
ORGAN_LETHAL_FRAC = 0.5        # a chest or belly organ crushed this far is fatal
DEATH_LOSS_FRAC = 0.504        # the existing death line: 50.4 % of the blood
BLEED_TAMPONADE = 0.35         # a closed bleed pools this share of its cavity (u)
V_ML_PER_KG = 70.0

# ── the random-orientation ensemble (data; the lane prints the fit) ──────────
# weight, mode, aware, one_foot. Weights sum to 1; tuned so survivors of a
# height-weighted population average 5.2 m and LD50 is 12-15 m.
ENSEMBLE = (
    (0.25, "feet", True, False),
    (0.10, "feet", True, True),
    (0.10, "feet", False, False),
    (0.15, "side", True, False),
    (0.10, "side", False, False),
    (0.10, "back", True, False),
    (0.08, "back", False, False),
    (0.12, "head", False, False),
)
FALL_HEIGHT_SCALE_M = 8.3      # fall heights in the population: exponential with this scale (u)
FALL_GRID = tuple(0.5 * i for i in range(1, 61))     # 0.5 .. 30 m


# ── the chain ────────────────────────────────────────────────────────────────
def chain(W_N, h_m, k_el, elements, E_J=0.0):
    """Solve the fuse chain. `elements` is a list of dicts with at least
    name, C (referred capacity, N), crush (m) and optional kind ('bone',
    'organ', 'ctl'); other keys ride along. `E_J` is extra energy in joules on
    top of W h (a blow gives W = 0, h = 0 and all its energy here).

    Returns (F_peak, failed, work):
      F_peak  peak contact force, N
      failed  the elements that gave way, ordered by capacity, each the element
              dict plus crush_used (m) and frac (crush_used / crush, 0..1); the
              control element is NOT listed (it has no consequence)
      work    the sinks: elastic_J, bone_J, organ_J, control_J, X_m (the whole
              stroke), W_h_X_J (the work in) and total_J (the sinks' sum), and
              ctl_used (the part of the control stroke used, 0..1)
    A bone whose C is at or below W has already failed by standing: listed with
    pre_failed True and nothing crushed."""
    W = float(W_N)
    k = float(k_el)
    target = W * h_m + E_J + W * W / (2.0 * k)
    els = sorted(elements, key=lambda e: (e["C"], e["name"]))
    failed = []
    used_sum = 0.0                    # sum of (C_i - W) c_i of crushed-through elements
    crush_sum = 0.0
    plateau = 0.0                     # sum of (C_i - W) x crush used, over every element
    sinks = {"bone": 0.0, "organ": 0.0, "control": 0.0}
    ctl_used = 0.0
    F = None

    def book(e, used):
        nonlocal plateau
        kind = e.get("kind", "bone")
        key = "control" if kind == "ctl" else ("organ" if kind == "organ" else "bone")
        sinks[key] += e["C"] * used
        plateau += (e["C"] - W) * used
        if kind == "ctl":
            return
        d = dict(e)
        d["crush_used"] = used
        d["frac"] = used / e["crush"] if e["crush"] > 0.0 else 1.0
        failed.append(d)

    for e in els:
        C, c = e["C"], e["crush"]
        if C <= W:
            if e.get("kind", "bone") != "ctl":
                d = dict(e)
                d.update(crush_used=0.0, frac=1.0, pre_failed=True)
                failed.append(d)
            continue
        E_minus = (C - W) ** 2 / (2.0 * k) + used_sum
        if target <= E_minus:
            F = W + math.sqrt(max(2.0 * k * (target - used_sum), 0.0))
            break
        E_plus = E_minus + (C - W) * c
        if target < E_plus:
            used = (target - E_minus) / (C - W)
            book(e, used)
            crush_sum += used
            if e.get("kind") == "ctl":
                ctl_used = used / c if c > 0.0 else 1.0
            F = C
            break
        book(e, c)
        crush_sum += c
        used_sum += (C - W) * c
        if e.get("kind") == "ctl":
            ctl_used = 1.0
    if F is None:
        F = W + math.sqrt(max(2.0 * k * (target - used_sum), 0.0))
    elastic = F * F / (2.0 * k)
    X = F / k + crush_sum
    work = dict(elastic_J=elastic, bone_J=sinks["bone"], organ_J=sinks["organ"],
                control_J=sinks["control"], X_m=X, ctl_used=ctl_used, plateau_J=plateau)
    work["W_h_X_J"] = W * h_m + E_J + W * X
    work["total_J"] = elastic + sinks["bone"] + sinks["organ"] + sinks["control"]
    return F, failed, work


def prep_of(t_air_s, aware=True):
    """How ready the legs are: clip((t_air - 0.10 s) / 0.20 s, 0, 1) for an awake
    body, from the AIRBORNE time (feet left support to feet meet it), not from
    the drop height. An unconscious body has none."""
    if not aware:
        return 0.0
    return min(max((t_air_s - PREP_T0_S) / PREP_RAMP_S, 0.0), 1.0)


def contact_energy(m_kg, v_n, r_cross_n=0.0, I_kgm2=None):
    """Energy taken at a contact point: 1/2 m_eff v_n^2 with 1/m_eff = 1/m +
    (r x n)^2 / I, v_n the speed along the contact normal at the contact cell
    (linear plus angular). With no inertia given, m_eff = m."""
    inv = 1.0 / m_kg
    if I_kgm2:
        inv += r_cross_n ** 2 / I_kgm2
    return 0.5 * v_n * v_n / inv


def legs_default(sheet):
    return LEGS_N_70 * (sheet["mass_kg"] / REF_KG) ** (2.0 / 3.0)


def crouch_default(sheet):
    return CROUCH_M_175 * sheet["height_m"] / _AN.REF_HEIGHT_M


def _bone_rec(recs, name):
    r = recs.get(name) if recs else None
    return r if r is not None else _BN.new_bone()


def _build_elements(mode, sheet, W, side, share_feet, recs):
    """The elements of one mode for this person: referred capacities from
    bonestate's break force, 1.25 dynamic factor, and the loaded feet."""
    spec = MODES[mode]
    bs = sheet.get("bone_scale", 1.0)
    out = []
    for e in spec["elements"]:
        bones = [b.replace("{S}", side) for b in e["bones"]]
        if spec["pairs"] and len(bones) == 2 and bones[0] != bones[1] and mode == "feet":
            bones = [b for b, s in zip(bones, share_feet) if s]
        elif spec["pairs"] and len(bones) == 2 and mode == "arms":
            bones = [b for b, s in zip(bones, share_feet) if s]
        if not bones:
            continue
        if e["kind"] == "organ":
            C = _AN.DYN * e["F_N"] * bs / e["m_above"]
            crush = e["crush"]
            bones_loaded = bones
        else:
            F = 0.0
            for b in bones:
                F += _BN.break_force(_bone_rec(recs, b), b, sheet, dyn=_AN.DYN,
                                     member=e["member"])
            C = F * e["n"] / e["m_above"]
            crush = e["crush"] if e["crush"] is not None else _AN.MEMBERS[e["member"]]["crush_m"]
            bones_loaded = bones
        out.append(dict(name=e["name"], C=C, crush=crush, kind=e["kind"], member=e["member"],
                        bones=tuple(bones_loaded), organs=e["organs"], m_above=e["m_above"]))
    return out


def head_result(sheet, m_eff, F_peak, k_el, ko_scale=None):
    """The head's numbers from the chain's peak force on it: peak acceleration
    a = F / m_eff (in g), the pulse a half-sine of length T_c = pi sqrt(m_eff/k),
    HIC15 = T (mean a)^2.5 with T at most 15 ms, knock-out when a >= 80 g x
    ko_scale for 3 ms (time out t_ko = 20 s (a / ko_g)^2, at most 30 min),
    skull-fracture risk at HIC 1000 and fatal brain injury at 2500.
    Rotational acceleration, the main cause of concussion, is NOT modelled."""
    ko_s = sheet.get("ko_scale", 1.0) if ko_scale is None else ko_scale
    a_peak = F_peak / (m_eff * G)
    Tc = math.pi * math.sqrt(m_eff / k_el)
    a_mean = a_peak * 2.0 / math.pi
    hic_v = hic(a_mean, Tc)
    thr = KO_G * ko_s
    if a_peak > thr:
        above = Tc * (1.0 - (2.0 / math.pi) * math.asin(thr / a_peak))
    else:
        above = 0.0
    ko = above >= KO_T_S
    t_ko = min(T_KO_S * (a_peak / thr) ** 2, T_KO_MAX_S) if ko else 0.0
    return dict(a_peak_g=a_peak, a_mean_g=a_mean, T_s=Tc, hic=hic_v, ko=ko, t_ko_s=t_ko,
                skull_risk=hic_v >= HIC_SKULL, dead=hic_v >= HIC_FATAL, m_eff=m_eff)


def hic(a_g, T_s):
    """HIC15 = T (mean a)^2.5, a in g, T in s, T at most 15 ms."""
    return min(T_s, HIC_T_MAX_S) * a_g ** 2.5


def _stage(sheet, mode, E_or_h, energy, m_kg, legs_N, crouch_m, t_air_s, aware, share_feet,
           side, prep, E_contact_J, arms_N, recs):
    spec = MODES[mode]
    if spec["pairs"] and sum(share_feet) <= 0:
        # No foot carries load (both off the ground): a feet or arms contact has
        # no spring, so the body lands on its side instead.
        return _stage(sheet, "side", E_or_h, energy, m_kg, legs_N, crouch_m, t_air_s, aware,
                      (1, 1), side, prep, E_contact_J, arms_N, recs)
    scale = m_kg / REF_KG
    W = m_kg * G
    ctl = None
    h_eff, W_eff = None, W
    m_eff = None
    if spec.get("head"):
        trunk = _AN.segment_mass(sheet, "torso")
        m_eff = HEAD_MASS_KG_70 * scale + NECK_LINK_X * trunk
        W_eff = m_eff * G
        h_eff = (E_or_h / W_eff) if energy else E_or_h
        k = spec["k_el"] * scale
    else:
        if E_contact_J is not None:
            h_eff = E_contact_J / W
        elif energy:
            h_eff = E_or_h / W
        elif spec["f_E"] is not None:
            h_eff = spec["f_E"][0 if aware else 1] * E_or_h
        else:
            h_eff = E_or_h
        k = spec["k_el_70"] * scale
        if spec["pairs"]:
            k *= sum(share_feet) / 2.0
    elements = _build_elements(mode, sheet, W, side, share_feet, recs)
    p = prep_of(t_air_s, aware) if prep is None else prep
    if spec["ctl"] == "legs":
        lg = legs_N if legs_N is not None else legs_default(sheet)
        cr = crouch_m if crouch_m is not None else crouch_default(sheet)
        C = (1.0 - p) * SOFT_CAP_BW * W + p * CTL_CAP_X * lg
        S = SOFT_STROKE_M + p * (CTL_STROKE_X * cr - SOFT_STROKE_M)
        elements.append(dict(name="control", C=C, crush=S, kind="ctl"))
    elif spec["ctl"] == "arms":
        an = arms_N if arms_N is not None else 2.0 * _arm_strength(sheet)
        arm_len = (0.186 + 0.146 + 0.108) * sheet["height_m"]
        C = (1.0 - p) * ARM_SOFT_BW * W + p * CTL_CAP_X * an
        S = ARM_SOFT_STROKE_M + p * (ARM_STROKE_X * arm_len - ARM_SOFT_STROKE_M)
        elements.append(dict(name="control", C=C, crush=S, kind="ctl"))
    F, failed, work = chain(W_eff, h_eff, k, elements)
    res = dict(mode=mode, F_peak=F, a_body_g=F / W_eff, failed=failed, work=work, prep=p,
               k_el=k, h_eff=h_eff, W_N=W_eff,
               contusion_J=work["elastic_J"],
               E_contact_J=W_eff * h_eff)
    if spec.get("head"):
        res["head"] = head_result(sheet, m_eff, F, k)
    return res


def _arm_strength(sheet):
    """One arm's push, N: the sheet's own strength_N, else the foot-grip rule of
    bodysheet, else the stock 400 N."""
    if "strength_N" in sheet:
        return float(sheet["strength_N"])
    try:
        from . import bodysheet as _bs
        return float(_bs.strength_N(sheet))
    except ImportError:                    # pragma: no cover
        return 400.0


def landing(sheet, mode, h_or_E, m_kg=None, legs_N=None, crouch_m=None, t_air_s=None,
            aware=True, share_feet=(1, 1), shares=1, roll_order=ROLL_ORDER, energy=False,
            side="left", prep=None, E_contact_J=None, arms_N=None, recs=None):
    """One landing. `h_or_E` is the drop height in metres, or contact energy in
    joules with energy=True. Modes: feet, side, back, shoulder, arms, head.
    The feet chain gets the control element (legs flexing) with prep from the
    airborne time `t_air_s` (default the free fall time of the height). The
    other modes take f_E of the drop energy (0.15 aware / 0.35 limp for the
    hip) unless the landing code passes `E_contact_J` (1/2 m_eff v_n^2, see
    `contact_energy`). `shares` > 1 is a roll: the energy splits into that many
    sequential landings, each through the next mode of `roll_order`.

    Returns {mode, F_peak, a_body_g, failed, work, prep, contusion_J,
    residual_J, stages, head?}. `failed` entries carry name, member, bones,
    kind, organs, crush_used (m), frac. `residual_J` is the energy this contact
    did not take (the body's motion for the next one). For a roll the failed
    list merges the stages (largest frac per element) and work is summed."""
    m = float(m_kg if m_kg is not None else sheet["mass_kg"])
    tair = t_air_s
    if tair is None:
        tair = math.sqrt(2.0 * max(h_or_E, 0.0) / G) if not energy else 0.0
    if shares <= 1:
        r = _stage(sheet, mode, h_or_E, energy, m, legs_N, crouch_m, tair, aware, share_feet,
                   side, prep, E_contact_J, arms_N, recs)
        total = (h_or_E if energy else m * G * h_or_E)
        r["residual_J"] = max(total - r["E_contact_J"], 0.0)
        r["stages"] = [r]
        return r
    stages = []
    for i in range(shares):
        md = mode if i == 0 else roll_order[i % len(roll_order)]
        part = h_or_E / shares
        stages.append(_stage(sheet, md, part, energy, m, legs_N, crouch_m,
                             tair / shares if i == 0 else 0.0, aware if i == 0 else False,
                             share_feet, side, prep if i == 0 else None, None, arms_N, recs))
    merged = {}
    for s in stages:
        for f in s["failed"]:
            key = (f["name"], f["bones"])
            if key not in merged or f["frac"] > merged[key]["frac"]:
                merged[key] = f
    work = {k: math.fsum(s["work"][k] for s in stages) for k in stages[0]["work"]
            if k not in ("ctl_used",)}
    work["ctl_used"] = max(s["work"]["ctl_used"] for s in stages)
    # the mode is the region the body FIRST touched, not the one that was asked for
    # (a feet landing with no foot down is a side landing in the first stage)
    out = dict(mode=stages[0]["mode"], F_peak=max(s["F_peak"] for s in stages),
               a_body_g=max(s["a_body_g"] for s in stages),
               failed=sorted(merged.values(), key=lambda f: (f["C"], f["name"])),
               work=work, prep=stages[0]["prep"], contusion_J=work["elastic_J"],
               residual_J=0.0, stages=stages, E_contact_J=math.fsum(s["E_contact_J"] for s in stages))
    for s in stages:
        if "head" in s:
            out["head"] = s["head"]
    return out


# ── what a landing does to a person ──────────────────────────────────────────
def hemorrhage_ml(res, sheet):
    """Blood that would pool at infinity from the breaks of a landing, ml,
    after tamponade: sum over broken bones of the member's cavity x frac."""
    ml = 0.0
    scale = sheet["mass_kg"] / REF_KG
    for st in res.get("stages", [res]):
        for f in st["failed"]:
            if f["kind"] != "bone":
                continue
            pool = _AN.MEMBERS[f["member"]]["pool_ml"]
            ml += pool * f["frac"] * max(1, len(f["bones"])) * scale * BLEED_TAMPONADE
    return ml


def outcome(res, sheet):
    """What a landing does to the person, untreated: {dead, why, ko, t_ko_s,
    hemorrhage_ml, bleed_frac}. Dead for a fatal head result (HIC 2500), a
    crushed organ (frac 0.5 of its stroke) or a closed bleed past the death
    line. This is the data the random-orientation fit is judged by; the live
    law reads the same records and lets perfusion decide."""
    why = None
    ko, t_ko = False, 0.0
    for st in res.get("stages", [res]):
        h = st.get("head")
        if h:
            ko = ko or h["ko"]
            t_ko = max(t_ko, h["t_ko_s"])
            if h["dead"]:
                why = why or "brain"
        for f in st["failed"]:
            if f["kind"] == "organ" and f["frac"] >= ORGAN_LETHAL_FRAC:
                why = why or "organ"
    ml = hemorrhage_ml(res, sheet)
    V0 = V_ML_PER_KG * sheet["mass_kg"]
    if ml >= DEATH_LOSS_FRAC * V0:
        why = why or "blood"
    return dict(dead=why is not None, why=why, ko=ko, t_ko_s=t_ko, hemorrhage_ml=ml,
                bleed_frac=ml / V0)


# ── blows ────────────────────────────────────────────────────────────────────
BLOW_K = {("club", "thigh"): 100e3, ("fist", "cheek"): 50e3, ("fist", "torso"): 60e3,
          ("club", "torso"): 80e3, ("club", "forearm"): 90e3}     # N/m (u)


def blow_on_bone(sheet, bone, joules, area_m2, edge, k_c, m_w, rec=None, member=None):
    """A blow on a bone, by ONE criterion, force. The share of the weapon's
    energy that goes into the struck segment: E_c = joules m_l / (m_w + m_l)
    (m_w the weapon's effective mass, m_l the segment's Winter mass). Contact
    force of a mass-spring contact: F_c = sqrt(2 k_c E_c). The bone breaks when
    F_c reaches its bending force F_bend = 4 M_fail / L (3-point bend)
    combined with the elliptic rule in bonestate; an edge that cuts to bone
    depth (`edge` = the fraction of the bone's section removed, 0 for a blunt
    blow) lowers integrity by that fraction and F_bend with it. After a break the
    bone soaks crush energy as in a fall.

    Returns {F_c, F_bend, E_c, broke, crush_used, frac, integrity, pressure_Pa,
    absorbed_J, rec}. `rec` is mutated when given."""
    m_l = _AN.segment_mass(sheet, bone)
    rec = rec if rec is not None else _BN.new_bone(sheet, bone)
    mem = _BN.member_of(bone, rec, member)
    row = _AN.MEMBERS[mem]
    if edge > 0.0:
        rec["integrity"] = min(rec.get("integrity", 1.0), 1.0 - min(edge, 1.0))
    E_c = joules * m_l / (m_w + m_l)
    F_c = math.sqrt(2.0 * k_c * E_c)
    L = row["L_m"] * sheet["height_m"] / _AN.REF_HEIGHT_M
    M_extra = F_c * L / 4.0                                  # the 3-point bend moment
    F_bend = 4.0 * _BN.capacity(rec, bone, sheet, mem)[1] / L
    broke = _BN.check(rec, bone, sheet, 0.0, lateral_Nm=M_extra, member=mem)
    out = dict(F_c=F_c, F_bend=F_bend, E_c=E_c, broke=bool(broke), crush_used=0.0, frac=0.0,
               integrity=rec["integrity"], pressure_Pa=F_c / area_m2 if area_m2 > 0.0 else math.inf,
               absorbed_J=0.0, member=mem)
    if broke:
        el = dict(name=mem, C=max(F_bend, 1e-6), crush=row["crush_m"], kind="bone")
        _, failed, work = chain(0.0, 0.0, k_c, [el], E_J=E_c)
        used = failed[0]["crush_used"] if failed else 0.0
        frac = failed[0]["frac"] if failed else 0.0
        # the part of the stroke that breaks the bone at all: a broken bone
        # has at least a small crush, so the record is never "broken, intact"
        frac = max(frac, 0.02)
        _BN.break_(rec, bone, sheet, F_c, frac, member=mem)
        out.update(crush_used=used, frac=frac, integrity=rec["integrity"],
                   absorbed_J=work["bone_J"] + work["elastic_J"])
    out["rec"] = rec
    return out


# ── the random-orientation ensemble ──────────────────────────────────────────
def _case(sheet, mode, aware, one_foot, h):
    kw = dict(aware=aware)
    if mode == "feet":
        kw["share_feet"] = (1, 0) if one_foot else (1, 1)
        kw["prep"] = 1.0 if aware else 0.0
    return landing(sheet, mode, h, **kw)


def survival_curve(sheet, ensemble=None, grid=FALL_GRID):
    """For each height of the grid: the fraction of the ensemble that survives
    (no treatment), by the outcome rule. Returns (heights, survival list)."""
    ens = ensemble or ENSEMBLE
    tot = math.fsum(e[0] for e in ens)
    S = []
    for h in grid:
        alive = 0.0
        for w, mode, aware, one in ens:
            r = _case(sheet, mode, aware, one, h)
            if not outcome(r, sheet)["dead"]:
                alive += w
        S.append(alive / tot)
    return list(grid), S


def fit_numbers(sheet, ensemble=None, scale_m=FALL_HEIGHT_SCALE_M, grid=FALL_GRID):
    """The two numbers the realism map asks of falls: the mean drop of the
    survivors of a population whose heights fall off exponentially with
    `scale_m` (target 5.2 m), and LD50, the height at which half of the
    random orientations die (target 12-15 m, linear between grid points).
    Returns {mean_survivable_m, ld50_m}."""
    hs, S = survival_curve(sheet, ensemble, grid)
    num = math.fsum(math.exp(-h / scale_m) * s * h for h, s in zip(hs, S))
    den = math.fsum(math.exp(-h / scale_m) * s for h, s in zip(hs, S))
    ld50 = None
    for i in range(1, len(hs)):
        if S[i] <= 0.5 < S[i - 1] or (S[i] <= 0.5 and S[i - 1] <= 0.5 and i == 1):
            f = (S[i - 1] - 0.5) / max(S[i - 1] - S[i], 1e-12)
            ld50 = hs[i - 1] + f * (hs[i] - hs[i - 1])
            break
    return dict(mean_survivable_m=num / den if den else 0.0, ld50_m=ld50)


def fit_report(sheet, ensemble=None):
    """A printable text of the fitted shares and the two numbers."""
    ens = ensemble or ENSEMBLE
    fit = fit_numbers(sheet, ens)
    lines = ["landing shares (weight, mode, aware, one foot):"]
    for w, mode, aware, one in ens:
        lines.append(f"  {w:5.3f}  {mode:9s} aware={aware!s:5s} one_foot={one}")
    lines.append(f"mean survivable drop {fit['mean_survivable_m']:.2f} m (target 5.2), "
                 f"LD50 {fit['ld50_m']} m (target 12-15)")
    return "\n".join(lines)
