"""SIGNS: what one person can see of another (wave 13, T10 3.7).

Each sign is a row of data: the visible state it reads, how strong it is
(0..1), the range at which a person looking can see it, and whether it needs
the target moving or the face turned to the viewer. The signs of a target are
worked out once per target per look from a plain dict, so every viewer sees
the same thing. A sign is SEEN when the target is in the viewer's focus and in
line of sight (the existing look pass) and the distance is within
range x light. This module only answers "which signs, how strong" and "is it
visible at this distance"; the look pass owns focus and line of sight.

RANGES ARE CLAMPED TO the sight range (`SEE_M`, WILL["see_m"] = 12 m): the look
pass only builds person pairs inside it, so a longer range could never fire.

HONESTY: a sign reads only the fields in INPUTS, things a person could show
(never the vessel table, the burn profile or the cause). The belief a viewer
forms does not name the cause.

Pure functions on plain dicts. No World import. The numbers are (u).
"""
try:
    from .policy import WILL as _WILL
    _SEE = float(_WILL["see_m"])
except ImportError:  # pragma: no cover - the stand-in
    _SEE = 12.0
SEE_M = min(_SEE, 12.0)

# The only keys of the state dict a sign may read. A hidden field (a vessel,
# a cut, an omega profile) is not here, so it cannot leak.
INPUTS = (
    "bleed_ml_s",      # flow from the worst open, unsealed cut
    "blood_on_clothes",
    "leg_capacity",    # gait capacity of the worst leg, 0..1
    "leg_pain",        # pain on the worst leg, 0..1
    "bone_angle",      # radians, the worst bend
    "open_fracture",
    "f",               # circulating blood fraction
    "skin_cold",
    "shiver",          # shiver power as a fraction of its maximum
    "drunk",
    "burn_third",      # any third-degree burn
    "burn_second_exposed",   # second degree on the face or hands
    "awake",
    "blood_o2",
    "lying",
    "arm_pain",        # pain on the worst arm, 0..1
    "fat_gam",         # the worst limb fatigue, 0..1
)

# thresholds (u)
BLEED_SEEN_ML_S = 0.05
BLEED_FULL_ML_S = 2.0
LIMP_PAIN = 0.3
BEND_RAD = 0.2
PALE_F = 0.8
PALE_FULL_F = 0.5
SHIVER_MIN = 0.10
DRUNK_MIN = 0.1
FAINT_O2 = 0.55            # BODY["faint_o2"]
CRADLE_PAIN = 0.5
TIRED_GAM = 0.6
STRONG = 0.9               # a strength at or over this uses range_strong_m


def _row(name, word, range_m, needs_moving=False, needs_face=False,
         range_strong_m=None):
    r = min(float(range_m), SEE_M)
    rs = r if range_strong_m is None else min(float(range_strong_m), SEE_M)
    return {"name": name, "word": word, "range_m": r, "range_strong_m": rs,
            "needs_moving": needs_moving, "needs_face": needs_face}


# Table order is the order of words. Every range is clamped to SEE_M.
SIGNS = (
    _row("limp", "limping", 12.0, needs_moving=True),
    _row("bleeding", "bleeding", 12.0),
    _row("deformed", "deformed", 10.0),
    _row("pale", "pale", 4.0, needs_face=True),
    _row("shivering", "shivering", 10.0),
    _row("drunk", "swaying", 12.0, needs_moving=True),
    _row("burned", "burned", 5.0, range_strong_m=12.0),
    _row("still", "still", 8.0),
    _row("cradling", "cradling an arm", 10.0),
    _row("tired", "tired", 6.0),
)
BY_NAME = {r["name"]: r for r in SIGNS}


def _clip01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def signs_of(state):
    """The signs a target shows: [(name, strength 0..1)] in table order.
    Reads only the keys in INPUTS."""
    g = state.get
    out = []
    cap = g("leg_capacity", 1.0)
    lp = g("leg_pain", 0.0)
    if cap < 1.0 or lp >= LIMP_PAIN:
        out.append(("limp", _clip01(max(1.0 - cap, lp))))
    flow = g("bleed_ml_s", 0.0)
    if flow >= BLEED_SEEN_ML_S:
        out.append(("bleeding", _clip01(flow / BLEED_FULL_ML_S)))
    elif g("blood_on_clothes", False):
        out.append(("bleeding", 0.3))
    ang = g("bone_angle", 0.0)
    if ang >= BEND_RAD or g("open_fracture", False):
        out.append(("deformed", 1.0 if g("open_fracture", False)
                    else _clip01(ang)))
    f = g("f", 1.0)
    if f < PALE_F:
        out.append(("pale", _clip01((PALE_F - f) / (PALE_F - PALE_FULL_F))))
    elif g("skin_cold", False):
        out.append(("pale", 0.5))
    sh = g("shiver", 0.0)
    if sh >= SHIVER_MIN:
        out.append(("shivering", _clip01(sh)))
    dr = g("drunk", 0.0)
    if dr >= DRUNK_MIN:
        out.append(("drunk", _clip01(dr)))
    if g("burn_third", False):
        out.append(("burned", 1.0))
    elif g("burn_second_exposed", False):
        out.append(("burned", 0.5))
    awake = g("awake", True)
    o2 = g("blood_o2", 1.0)
    if (not awake or o2 < FAINT_O2) and g("lying", False):
        out.append(("still", 1.0))
    ap = g("arm_pain", 0.0)
    if ap >= CRADLE_PAIN:
        out.append(("cradling", _clip01(ap)))
    gam = g("fat_gam", 0.0)
    if gam >= TIRED_GAM:
        out.append(("tired", _clip01(gam)))
    return out


def range_of(sign):
    """The range (m) at which a sign, given as a name or a (name, strength)
    pair, can be seen. A strong sign (strength 0.9 or more) may carry a
    longer range, as a third-degree burn does."""
    name, strength = (sign, 0.0) if isinstance(sign, str) else sign
    row = BY_NAME[name]
    return row["range_strong_m"] if strength >= STRONG else row["range_m"]


def visible(sign, dist_m, light=1.0, moving=False, facing_ok=True):
    """Can a viewer at dist_m see this sign? `sign` is a name or a
    (name, strength) pair. light scales the range (1.0 in a lit room). The
    look pass has already checked focus and line of sight."""
    name = sign if isinstance(sign, str) else sign[0]
    row = BY_NAME[name]
    if row["needs_moving"] and not moving:
        return False
    if row["needs_face"] and not facing_ok:
        return False
    return dist_m <= range_of(sign) * light


def visible_signs(state, dist_m, light=1.0, moving=False, facing_ok=True):
    """The names of the signs of `state` a viewer at dist_m can see."""
    return tuple(n for n, s in signs_of(state)
                 if visible((n, s), dist_m, light, moving, facing_ok))


def words(names):
    """The words for a set of sign names, in table order: 'limping and
    bleeding'. Stable for the same set; unknown names are refused."""
    want = set(names)
    for n in want:
        if n not in BY_NAME:
            raise KeyError("no sign %r" % (n,))
    ws = [r["word"] for r in SIGNS if r["name"] in want]
    if len(ws) <= 1:
        return "".join(ws)
    return ", ".join(ws[:-1]) + " and " + ws[-1]
