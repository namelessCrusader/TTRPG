"""EXERTION FROM THE BODY (wave 14b, lane G).

How hard a person can strike, shove or throw comes from THEIR BODY: muscle
force, the mass of the swinging segment, technique, training and state. No
fixed constant and no trait flag. Pure functions on a person's dict; no World.

THE LAW (one for every act)
    E = F_muscle x stroke x technique x z_person x state        (joules)
  F_muscle   the person's strength_N (p["strength_N"], or the sheet's, or the
             stock 400 N), raised by TRAINING with diminishing returns.
  stroke     the distance the segment is driven while the muscle pushes it,
             per act (KINDS). A kick of 400 N over 0.2875 m is 115 J.
  technique  a curve of the skill level 0..1 with diminishing returns
             (bodysheet.skill_of): tech(0.6) = 1, tech(0.4) 0.82, tech(1) 1.25.
  z_person   a per person lognormal draw (hash of pid and name, so the same
             person gives the same answer): the part of muscle quality the
             sheet does not carry (fibre type, leverage, leg length).
  state      fatigue of the limb group (fatigue.capacity and push_factor) times
             the leg capacity of the bones (bonestate.leg_ratio, for a kick),
             times a situation factor the caller may give (footing, a cramped
             room: `situation`, 0..1).
  The effective mass of the segment is a fraction of the body mass (KINDS), and
  the speed at contact follows: v = sqrt(2 E / m_eff). The energy is paid from
  the body's books by the caller at the efficiency of muscle (energetics.ETA).

CALIBRATION (the ordinary adult kick is 115 J, decision 6 of wave 14)
  Ramakrishnan, Wang, Shankar, Fien 2018, Sports Engineering 21:53-62: 52
  volunteers, open-stance front kick on a rig: two thirds averaged over 100 J
  and about one adult in a hundred above 215 J. A lognormal with median 115 J
  and sigma 0.27 gives P(E > 100) = 70 percent and P(E > 215) = 1.0 percent.
  The built population (population_kick: z sigma 0.22, skills 0.4 to 0.8, 5
  percent trained) has median 114 J, P(E > 215) = 1.0 percent and P(E > 100)
  = 80 percent: the tail is matched, the low side is a little narrow (u). Effective striker mass 5.6 kg is 8 percent of 70 kg
  (leg 16.1 percent of body mass, Winter 2009 / Dempster, about half of it
  moving at foot speed) (u). Training: strength training raises force 20 to
  35 percent (Hakkinen; Aagaard) (u), with a diminishing curve here.
  The values marked (u) are unverified. Where the full text was not read, the
  abstract's numbers are used.
"""
import math
from functools import lru_cache
from statistics import NormalDist

from . import bodysheet as _bs
from . import fatigue as _fat
from . import energetics as _en

STOCK_STRENGTH_N = 400.0
REF_MASS_KG = 70.0
REF_SKILL = 0.6                 # the middle of bodysheet.SKILL_RANGE["kick"]
Z_SIGMA = 0.22                  # (u) tuned: see population_kick
Z_CLIP = 3.0                    # draws beyond 3 sigma are clipped
TRAIN_GAIN = 0.35               # (u) most force a lifetime of training adds
TRAIN_RATE = 3.0                # diminishing returns: 1 - exp(-3 t)

# technique: T0 + (T1 - T0) * (1 - exp(-s / TAU)) / (1 - exp(-1 / TAU)), then
# divided by its value at REF_SKILL. A skilled kicker is better, with less
# and less to gain near the top. (u) shape; the range 0.55 .. 2.3 is a 4x
# spread of raw technique, which is about what a novice-expert gap is.
TECH_T0, TECH_T1, TECH_TAU = 0.55, 2.3, 0.8

# act: stroke (m) of the muscle, effective mass fraction of body mass, limb group
KINDS = {
    "kick":  {"stroke_m": 0.2875, "mass_frac": 0.080, "limb": "legs"},   # 115 J at 400 N
    "punch": {"stroke_m": 0.100, "mass_frac": 0.040, "limb": "right arm"},   # 40 J (u)
    "shove": {"stroke_m": 0.300, "mass_frac": 0.500, "limb": "right arm"},   # 120 J (u)
    "throw": {"stroke_m": 0.500, "mass_frac": 0.020, "limb": "right arm"},   # arm 2 percent (u)
}
# a lift, a haul, a shove of a thing, a grip is a FORCE, not an energy (`force_N`): the back
# and legs carry it
FORCE_KINDS = {"lift": {"limb": "back"}}
THROW_V_MAX_MS = 30.0           # fastest an untrained arm releases a light ball (u)
M_ARM_EFF_KG = 1.5              # (u) the arm and shoulder that move with the thing
_N = NormalDist()


def _raw_tech(s):
    s = min(max(float(s), 0.0), 1.0)
    return TECH_T0 + (TECH_T1 - TECH_T0) * (1.0 - math.exp(-s / TECH_TAU)) \
        / (1.0 - math.exp(-1.0 / TECH_TAU))


_TECH_REF = _raw_tech(REF_SKILL)


def technique(skill):
    """Technique factor, 1.0 at the middle skill (0.6), diminishing returns."""
    return _raw_tech(skill) / _TECH_REF


def training_factor(train):
    """Force gain of training level 0..1: 1 + 0.35 (1 - exp(-3 t)) / (1 - exp(-3))."""
    t = min(max(float(train or 0.0), 0.0), 1.0)
    return 1.0 + TRAIN_GAIN * (1.0 - math.exp(-TRAIN_RATE * t)) / (1.0 - math.exp(-TRAIN_RATE))


def person_z(p, act="kick", sigma=Z_SIGMA):
    """The person's own muscle-quality draw: lognormal, median 1, from the hash
    of act, pid and name. A stored p["exert_z"][act] wins."""
    own = (p.get("exert_z") or {}).get(act)
    if own is not None:
        return float(own)
    return _z_hashed(act, p.get("pid"), p.get("name"), sigma)


@lru_cache(maxsize=4096)
def _z_hashed(act, pid, name, sigma):
    """The hashed draw (pure in its arguments, so it is cached: the lift limit is
    asked of a hauler every tick)."""
    u = min(max(_bs._u("exert|" + act, {"pid": pid, "name": name}), 1e-6), 1.0 - 1e-6)
    n = min(max(_N.inv_cdf(u), -Z_CLIP), Z_CLIP)
    return math.exp(sigma * n)


def strength_of(p):
    own = p.get("strength_N")
    if own is not None:
        return float(own)
    sh = p.get("sheet")
    # a sheet that says neither its own strength nor a mass (a scene's partial
    # sheet, holding only some other number) leaves the stock adult
    if isinstance(sh, dict) and ("strength_N" in sh or "mass_kg" in sh):
        return _bs.strength_N(sh)
    return STOCK_STRENGTH_N


def mass_of(p):
    sh = p.get("sheet")
    return float(sh["mass_kg"]) if isinstance(sh, dict) and "mass_kg" in sh else REF_MASS_KG


def state_factor(p, kind="kick", situation=1.0):
    """0..1: what fatigue, injury and the situation leave of the force."""
    limb = (KINDS.get(kind) or FORCE_KINDS[kind])["limb"]
    st = (p.get("body") or {}).get("fat", {}).get(limb)
    f = 1.0
    if st is not None:
        f = _fat.capacity(st, 0.0) * _fat.push_factor(st)
    f *= min(max(float(situation), 0.0), 1.0)
    bones = (p.get("body") or {}).get("bones")
    if kind == "kick" and bones:
        sh = p.get("sheet") if isinstance(p.get("sheet"), dict) else _bs.STOCK
        w = mass_of(p) * 9.81
        from . import bonestate as _bn
        f *= _bn.leg_ratio(bones, str(p.get("striking_leg", "right")), sh, w)
    return f


def strike(p, kind="kick", situation=1.0, skill=None):
    """The strike of this person now: {"E_J", "m_kg", "v_ms", "F_N", ...}.
    Pure: reads p, writes nothing."""
    k = KINDS[kind]
    sk = _bs.skill_of(p, kind if kind in _bs.SKILL_RANGE else "kick") if skill is None else skill
    F = strength_of(p) * training_factor(p.get("training"))
    E = F * k["stroke_m"] * technique(sk) * person_z(p, kind) * state_factor(p, kind, situation)
    m = k["mass_frac"] * mass_of(p)
    return {"E_J": E, "m_kg": m, "v_ms": math.sqrt(2.0 * E / m) if E > 0 else 0.0,
            "F_N": F, "skill": sk}


def throw_speed_ms(p, thing_kg, situation=1.0):
    """Release speed of a thrown thing of `thing_kg`: the arm's energy shared
    with the arm's own moving mass, capped by the fastest the arm moves."""
    e = strike(p, "throw", situation)["E_J"]
    v = math.sqrt(2.0 * e / (max(float(thing_kg), 1e-6) + M_ARM_EFF_KG))
    return min(v, THROW_V_MAX_MS)


def force_N(p, situation=1.0):
    """The most force this body puts on a load (lifts, hauls, shoves, holds): its
    muscle force raised by training, times the person's own spread and what
    fatigue, injury and the situation leave. The stock adult at rest gives 400 N,
    so a world with the gate off (400 N flat) and this one agree for the person
    the stock sheet describes."""
    return strength_of(p) * training_factor(p.get("training")) * person_z(p, "lift") \
        * state_factor(p, "lift", situation)


def torque_scale(p, kind="punch", situation=1.0):
    """How much stronger than the stock arm this body's swing is: its strike
    energy over the stock adult's (400 N, middle technique, no spread, rested).
    The arm's torque (materials.BODY arm_Nm) is the stock adult's; a body's own
    muscle scales it by this, so the speed a swing or a throw arrives with is
    the person's (the same law as `strike`)."""
    ref = STOCK_STRENGTH_N * KINDS[kind]["stroke_m"]
    return strike(p, kind, situation)["E_J"] / ref


def reserve_J(p, mass_kg=None):
    """What a body with no heat record has to pay a burst with, metabolic joules:
    the limb's reserve above critical power (W', fatigue.WP) times the body's
    own. The reserve comes back with the recovery of fatigue.WP_TAU_S."""
    m = mass_kg if mass_kg is not None else mass_of(p)
    return _fat.W_PRIME_J_KG * m


def pay_reserve(p, kind, J, now_s):
    """Pay a strike of J mechanical joules from the limb's reserve when the body
    has no heat record (so no stores). The muscle burns J / eta; the reserve
    `wp` falls by that over the body's own W' and recovers (tau 300 s) between
    strikes, counted from the last. Returns the share that could be paid: 1.0,
    or less when the reserve ran dry (the strike is weaker). Writes the limb's
    fatigue state, so the next strike's `state_factor` reads the tiredness."""
    if J <= 0.0:
        return 1.0
    limb = KINDS[kind]["limb"]
    fat = p.setdefault("body", {}).setdefault("fat", {})
    st = fat.setdefault(limb, _fat.new_state())
    last = st.get("t_s")
    wp = float(st["wp"])
    if last is not None and now_s > last:
        wp = 1.0 - (1.0 - wp) * math.exp(-(now_s - last) / _fat.WP_TAU_S)
    cost = pay_J(J)
    w_prime = reserve_J(p)
    have = wp * w_prime
    paid = min(cost, have)
    st["wp"] = (have - paid) / w_prime
    st["t_s"] = float(now_s)
    return paid / cost


def pay_J(E_J, eta=_en.ETA):
    """What the muscle burns to deliver E_J of work."""
    return E_J / eta


def population_kick(n=20000, trained_share=0.05, seed=7):
    """A deterministic population (a van der Corput walk, no RNG): the kick
    energies of n adults with stock strength, skills in the kick range, a
    trained minority with the highest skill and training. Returns a sorted
    list. Used by the calibration test and by the report."""
    out = []
    lo, hi = _bs.SKILL_RANGE["kick"]
    for i in range(n):
        p = {"pid": i, "name": "pop%d" % i, "strength_N": STOCK_STRENGTH_N}
        if _vdc(i + seed, 3) < trained_share:
            p["skill"] = {"kick": 0.85 + 0.15 * _vdc(i + seed, 5)}
            p["training"] = 0.3 + 0.7 * _vdc(i + seed, 7)
        out.append(strike(p, "kick")["E_J"])
    out.sort()
    return out


def _vdc(i, b):
    f, r = 1.0, 0.0
    i = int(i)
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r
