"""INJURY LAWS (wave 13c, lane C7): what knocks a person out, and what a fall,
a burn, a hot hold and a loud noise do to one.

The old rule was one number: p["hurt"] + p["burn"] past 0.2 knocked a person
out, past 0.45 killed them. A bruise (hurt) is not a cause of unconsciousness
in a real body, so the number is gone and the real causes stand in its place.
Every cause below is a state on the body that takes the SAME road to the
brain that blood and oxygen already take (`delivered_o2`), or is a timed
knock-out. None of them is a flag on a person.

  blood loss      unchanged: the blood fraction sets the perfusion.
  low oxygen      unchanged: the air and the carbon monoxide.
  a head blow     `ko_until` (a tick). A head result of the landing chain
                  (impact.head_result) or a blow on the head sets it. The
                  person is out until then and wakes only after it.
  burn shock      the burn integral p["burn"] is a stand-in for the burnt
                  area (the face-by-face skin model, skinburn, waits for the
                  heat law to publish its flux). It takes plasma out of the
                  circulation, so it is a blood-loss equivalent (`shock_loss`)
                  read through the same perfusion curve. It is calibrated so
                  that a burn ALONE faints and kills at the old lines (0.2 and
                  0.45), and so that nothing else changes for burns.
  fatal organ     `organ_dead`: a landing that crushed a chest organ past half
                  its stroke, or a fatal head result.
  the vasovagal   pain or fear -> vagal outflow -> heart rate and pressure
  reflex          fall -> flow to the brain falls -> the person faints. The
                  drive is ACUTE pain (new hurt and new burn, fading with a
                  30 s time constant) plus fear. Each person has a threshold
                  drawn from a log-normal distribution (`vagal_threshold`):
                  most people need a great deal, a few faint at a stubbed toe.
                  The flow drop reaches the brain through `flow_factor`, so
                  the faint is the ordinary oxygen faint with its ordinary
                  hysteresis; lying flat takes the vagal drive away and the
                  person comes round.

Falls: `landing_effects` runs the closed-form landing chain (impact.py) for a
person and writes what it found: a head knock-out, a crushed organ, bones
that failed (kept as a list for the support law, not yet read by it) and the
blood the breaks will pool into the limb and the trunk over half an hour.
Hot holds: `withdraw_check` is the pain withdrawal reflex of a hand on a hot
thing. A loud noise on a sleeper is `ReflexMixin.noise_wakes` (reflex.py).
Both use the D2 law of reflex.py: sleep raises the threshold, only the dead
are off, and whether a stimulus fires the reflex is a chance."""
import math
import zlib

import numpy as np

from . import anatomy as AN
from . import bodysheet as BS
from . import impact as IM
from . import skinburn as SB
from . import vasculature as VA
from .materials import BODY, SOLID, TICK_S

# ── the vasovagal reflex ──────────────────────────────────────────────────────
# Median threshold in hurt units (1.0 = 2500 J absorbed): 0.6 is a hard
# landing's worth of pain arriving at once. The spread is log-normal with
# sigma 0.6 and clipped at +-2.5 sigma: 5 percent of people faint below 0.22,
# 1 percent below 0.15, and one in a hundred needs more than 1.5 (u: the
# numbers; real: a minority faint from pain or the sight of blood, Kapoor 2000
# and Wieling et al. 2009 give syncope of this kind in roughly one person in
# three over a lifetime, almost always from a quite strong trigger).
VAGAL_MEDIAN = 0.6
VAGAL_SIGMA = 0.6
VAGAL_CLIP = 2.5
PAIN_TAU_S = 30.0          # acute pain fades (habituation and the flow answer)
VAGAL_RISE_S = 4.0         # the vagal outflow builds over seconds
VAGAL_FALL_S = 10.0        # and lying flat takes it away over seconds
VAGAL_SPAN = 0.25          # full response at 1.25 x threshold
FLOW_DROP = 0.7            # at full response the brain gets 30 percent (never alone a death)
UPRIGHT_M = 0.8            # a body shorter than this is lying down


def _normal(p, key):
    """A standard-normal number that is the person's own (Box-Muller from two
    hashes of the key, the pid and the name; no random stream is touched)."""
    h = f"{key}|{p.get('pid')}|{p.get('name')}"
    u1 = (zlib.crc32((h + "|a").encode()) & 0xFFFFFFFF) / 2.0 ** 32
    u2 = (zlib.crc32((h + "|b").encode()) & 0xFFFFFFFF) / 2.0 ** 32
    return math.sqrt(-2.0 * math.log(max(u1, 1e-12))) * math.cos(2.0 * math.pi * u2)


def vagal_threshold(p):
    """The acute pain (hurt units) that makes THIS person's vagal outflow
    answer in full. The sheet's own number if it has one (p["vagal_dial"]),
    else a log-normal draw."""
    v = p.get("vagal_dial")
    if v is not None:
        return float(v)
    z = max(-VAGAL_CLIP, min(VAGAL_CLIP, _normal(p, "vagal")))
    return VAGAL_MEDIAN * math.exp(VAGAL_SIGMA * z)


def vagal_step(p, upright, tick):
    """Advance the acute pain and the vagal outflow of one person by a tick.
    Writes nothing for a person who has never been hurt, burnt or afraid."""
    h, b = p.get("hurt", 0.0), p.get("burn", 0.0)
    seen = p.get("_pain_seen")
    if seen is None:
        if h == 0.0 and b == 0.0 and not p.get("fear"):
            return
        if tick <= 1:                       # a body that STARTS hurt is not in a new pain
            p["_pain_seen"] = (h, b)
            return
        seen = (0.0, 0.0)
    acute = p.get("pain_acute", 0.0)
    v = p.get("vagal", 0.0)
    if (h, b) == seen and acute < 1e-4 and v < 1e-4 and not p.get("fear"):
        return
    dt = TICK_S
    acute *= math.exp(-dt / PAIN_TAU_S)
    ps = float(p.get("pain_scale", 1.0))
    acute += max(h - seen[0], 0.0) * ps + max(b - seen[1], 0.0)
    p["_pain_seen"] = (h, b)
    drive = acute + float(p.get("fear", 0.0))
    thr = vagal_threshold(p)
    if upright and p["awake"]:
        target, tau = min(max((drive - thr) / (VAGAL_SPAN * thr), 0.0), 1.0), VAGAL_RISE_S
    else:
        target, tau = 0.0, VAGAL_FALL_S
    v += (target - v) * (1.0 - math.exp(-dt / tau))
    p["pain_acute"] = acute if acute >= 1e-4 else 0.0
    p["vagal"] = v if v >= 1e-4 else 0.0


# ── burn shock ────────────────────────────────────────────────────────────────
_SHOCK_X = (0.0, BODY["faint_burn"], BODY["death_burn"], 1.0)
_SHOCK_Y = (0.0, VA.faint_loss_frac(), VA.death_loss_frac(), 0.8)


def shock_loss(burn):
    """The blood-loss equivalent of a burn integral: plasma that has left the
    circulation. Through (0.2, 36 percent) and (0.45, 50.4 percent), the
    fractions at which perfusion reaches the old faint and death lines."""
    return float(np.interp(burn, _SHOCK_X, _SHOCK_Y))


def flow_factor(p):
    """The share of the brain's flow that burn shock and the vagal outflow
    leave (1.0 for a body with neither). Multiplies `delivered_o2`."""
    f = 1.0
    b = p.get("burn", 0.0)
    if b > 0.0:
        f *= SB.perfusion_after_loss(shock_loss(b))
    v = p.get("vagal", 0.0)
    if v > 0.0:
        f *= 1.0 - FLOW_DROP * v
    return f


def out_from_a_blow(p, tick):
    """True while a head blow keeps this person out (`ko_until`)."""
    return tick < p.get("ko_until", 0)


# ── internal bleeding: blood that has left the circulation but not the body ─
def pool_step(p):
    """Move the blood a fracture will pool (`pool_left_ml`) into the tissue
    (`pooled_ml`) at the vessels' own time constant. The blood stays in the
    body (`blood_ml` is unchanged; the books close) but it no longer perfuses:
    `physiology._perfusion` reads blood_ml - pooled_ml."""
    left = p.get("pool_left_ml", 0.0)
    if left <= 0.0:
        return
    d = left * (1.0 - math.exp(-TICK_S / VA.POOL_TAU_S))
    d = min(d, left)
    p["pooled_ml"] = p.get("pooled_ml", 0.0) + d
    p["pool_left_ml"] = left - d if left - d > 1e-6 else 0.0


# ── falls ─────────────────────────────────────────────────────────────────────
def sheet_of(p):
    """The injury sheet of person `p` (anatomy.make_sheet over its p["sheet"],
    or the stock figure), with its own draws."""
    # (p["sheet"] is also the Mind's harm/traits sheet in some scenes; only a
    # sheet with the body's numbers is a body sheet)
    s = p.get("sheet")
    body = s if s and "mass_kg" in s and "height_m" in s else BS.get_sheet("stock")
    return AN.make_sheet(body, pid=p.get("pid", 0), name=p.get("name", ""))


def fall_mode(pose, contact, eye):
    """Which part of the body came down, from the landing's own cells: the
    principal axis of the cells and the end that touched. A body within 35
    degrees of vertical lands on its feet if the low end is away from the
    head (`eye`, where the head was a moment ago) and on its head if it is
    the head end. A tilted body lands on its side."""
    pts = np.asarray(pose, float)
    if len(pts) < 3:
        return "feet"
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    ax = vt[0]
    if abs(ax[2]) < math.cos(math.radians(35.0)):
        return "side"
    z_lo = pts[:, 2].min()
    low = pts[(pts[:, 2] <= z_lo + 1.0)]
    if eye is None:
        return "feet"
    d_head = float(np.linalg.norm(low.mean(axis=0) - np.asarray(eye, float)))
    d_far = float(np.linalg.norm(pts[pts[:, 2] >= pts[:, 2].max() - 1.0].mean(axis=0)
                                 - np.asarray(eye, float)))
    return "head" if d_head < d_far else "feet"


def landing_effects(p, joules, pose, contact, shares, tick):
    """What a landing of `joules` did to the person `p`: runs the chain and
    writes ko_until, organ_dead, fractures and the pooling blood on the person.
    Returns the result, or None when the landing is too small to matter."""
    sheet = sheet_of(p)
    m = float(sheet["mass_kg"])
    h = joules / (m * IM.G)
    if h < 0.25:                          # under a knee-high step nothing breaks
        return None
    mode = fall_mode(pose, contact, p.get("_eye"))
    res = IM.landing(sheet, mode, h, m_kg=m, aware=bool(p["awake"]), shares=max(int(shares), 1))
    out = IM.outcome(res, sheet)
    if out["ko"]:
        p["ko_until"] = max(p.get("ko_until", 0), tick + int(out["t_ko_s"] / TICK_S))
    if out["dead"] and out["why"] in ("brain", "organ"):
        p["organ_dead"] = out["why"]
    ml = out["hemorrhage_ml"]
    if ml > 0.0:
        p["pool_left_ml"] = p.get("pool_left_ml", 0.0) + ml
    failed = [f["name"] for st in res.get("stages", [res]) for f in st["failed"]
              if f["kind"] == "bone"]
    if failed:
        p["fractures"] = sorted(set(p.get("fractures", [])) | set(failed))
    return dict(res=res, out=out, mode=mode, h=h)


def head_blow(p, joules, tick, m_w_kg=1.0, k_el=500e3):
    """A blow on the head of `joules` from a weapon or fist of effective mass
    `m_w_kg` (u: 1 kg), through the head's contact spring: the same head
    result as a landing. Sets ko_until or organ_dead and returns the result."""
    sheet = sheet_of(p)
    m_h = AN.segment_mass(sheet, "head")
    e_c = joules * m_h / (m_w_kg + m_h)
    f = math.sqrt(2.0 * k_el * e_c)
    res = IM.head_result(sheet, m_h, f, k_el)
    if res["ko"]:
        p["ko_until"] = max(p.get("ko_until", 0), tick + int(res["t_ko_s"] / TICK_S))
    if res["dead"]:
        p["organ_dead"] = "brain"
    return res


# ── the hot hold: the pain withdrawal ────────────────────────────────────────
def effusivity(mat):
    """sqrt(k rho c) of a solid material id, J/(m2 K s^0.5), from the table."""
    row = SOLID.get(int(mat))
    if row is None:
        return SB.SKIN_E
    rho, c, k = row[0], row[1], row[2]
    return math.sqrt(max(k, 1e-6) * max(rho, 1.0) * max(c, 1e-3) * 1000.0)


def hold_pain(mat, T_thing, T_skin=SB.T_SET):
    """Heat pain 0..1 of a hand on a thing: the interface temperature of two
    semi-infinite bodies (skinburn.contact_T), read by the heat pain scale."""
    return SB.pain_heat(SB.contact_T(SB.SKIN_E, T_skin, effusivity(mat), T_thing))
