"""FATIGUE: endurance and recovery per limb (wave 13, T10 3.5).

Per limb group (legs, left arm, right arm, back) two numbers, kept in a plain
dict {"gam": 0..1, "wp": 0..1}:

  gam  STATIC fatigue (a grip, a held load, standing on one leg, pressing a
       wound). Endurance at a constant fraction f of the RESTED maximum
       voluntary force is ET(f) = 21.9 f^-1.98 seconds (the realism map's
       fit: f 0.5 -> 86 s, f 0.2 -> 9 min). MINER'S RULE: while loaded
       d(gam)/dt = 1 / ET(load / MVC_rested), at rest d(gam)/dt = -gam / 240 s,
       and the hold FAILS when gam >= 1. Because the rate depends on the
       RESTED maximum, not on the already-tired one, a constant load ends at
       exactly ET(f) (v1 used the fatigued capacity and was off by 2x). The
       capacity shown to the rest of the sim is
       c = 1 - gam (1 - f_load), floor 0.1: it equals the load fraction at
       gam = 1, which is where the hold fails.
  wp   DYNAMIC reserve (gait, climbing, rowing): critical power. CP is 6.5 MET
       and W' is 250 J/kg (u); above CP the reserve falls linearly, below it
       recovers toward 1 with tau 300 s (Skiba-style, u).

Energy: mechanical power F v costs F v / eta of metabolic power (eta 0.22), a
lowering (negative work) costs |F v| / 1.2, and a STATIC hold, which does no
external work, still costs P = 60 W f^1.5 per active limb group (about 10 W at
0.3 MVC; published isometric costs are 5-15 W for 10-30 percent, (u)). The
module returns joules; the energy store of the person debits them.

Pure functions on plain dicts, no World. The constants eta, MET and the
eccentric divisor come from energetics.py when it is there; the literals below
are a stand-in with a skip-marked cross test.
"""
import math

try:
    from . import energetics as _EN
    ETA = _EN.ETA
    MET_W_PER_KG = _EN.MET_W_PER_KG
    ECC_DIV = _EN.ECC_DIV
except ImportError:  # pragma: no cover - the stand-in numbers
    ETA = 0.22
    MET_W_PER_KG = 1.22
    ECC_DIV = 1.2
STAND_IN = {"ETA": 0.22, "MET_W_PER_KG": 1.22, "ECC_DIV": 1.2}

LIMBS = ("legs", "left arm", "right arm", "back")

# Static: Rohmert-style endurance fit (the realism map).
ET_A = 21.9                    # s at f = 1
ET_N = 1.98
F_REST = 0.05                  # below this load fraction the limb rests
F_FLOOR = 0.10                 # below this the rate falls off fast (u)
FLOOR_EXP = 6.0                # how fast (u)
REC_TAU_S = 240.0              # recovery of gam at rest (u)
CAP_FLOOR = 0.10               # the shown capacity never goes under this
F_MAX = 3.0                    # loads over this fraction are rated as this

# Dynamic: critical power.
CP_MET = 6.5                   # (u)
W_PRIME_J_KG = 250.0           # (u)
WP_TAU_S = 300.0               # recovery below CP (u)
WP_FULL = 0.2                  # above this reserve the push is full
PUSH_AT_EMPTY = 0.6            # push with an empty reserve (CP level), (u)

# Muscle energy.
STATIC_W = 60.0                # W per active limb group at f = 1
STATIC_EXP = 1.5


def new_state():
    """A rested limb."""
    return {"gam": 0.0, "wp": 1.0}


def et_s(f):
    """Endurance time (s) of a constant hold at load fraction f of the rested
    maximum: 21.9 f^-1.98. Below 10 percent the time grows faster still, so a
    light hold is held very long (the soft floor near Rohmert's limit)."""
    f = min(max(float(f), 1e-6), F_MAX)
    if f >= F_FLOOR:
        return ET_A * f ** -ET_N
    return ET_A * F_FLOOR ** -ET_N * (F_FLOOR / f) ** FLOOR_EXP


def step_static(state, load_N, mvc_N, dt):
    """Advance one limb's static fatigue by dt seconds (a closed form for any
    dt: the rate is constant while the load is). Returns a NEW dict; the
    input is not changed. gam stays in 0..1; at 1 the hold has failed."""
    gam = state["gam"]
    f = load_N / mvc_N if mvc_N > 0.0 else F_MAX
    if f < F_REST:
        gam = gam * math.exp(-dt / REC_TAU_S)
    else:
        gam = min(gam + dt / et_s(f), 1.0)
    out = dict(state)
    out["gam"] = gam
    return out


def capacity(state, f_load):
    """Shown capacity as a fraction of the rested maximum, never under 0.1."""
    f = min(max(f_load, 0.0), 1.0)
    return max(1.0 - state["gam"] * (1.0 - f), CAP_FLOOR)


def holds(state, load_N, mvc_N):
    """True while the hold is on: not worn out and not over the capacity."""
    if mvc_N <= 0.0:
        return load_N <= 0.0
    f = load_N / mvc_N
    return state["gam"] < 1.0 and f <= capacity(state, f)


def step_dynamic(state, met, dt, mass_kg):
    """Advance the dynamic reserve wp. `met` is the person's current MET (the
    sim's ACSM number). Above critical power wp falls at (P - CP) / W',
    below it wp recovers toward 1 with tau 300 s. Returns a NEW dict."""
    p_kg = met * MET_W_PER_KG                    # W/kg
    cp_kg = CP_MET * MET_W_PER_KG
    wp = state["wp"]
    if p_kg > cp_kg:
        wp = max(wp - (p_kg - cp_kg) / W_PRIME_J_KG * dt, 0.0)
    else:
        wp = 1.0 - (1.0 - wp) * math.exp(-dt / WP_TAU_S)
    out = dict(state)
    out["wp"] = wp
    return out


def push_factor(state):
    """How much of the legs' push and sprint speed the reserve allows: full
    above wp 0.2, falling to the critical-power level (0.6, u) when empty."""
    wp = state["wp"]
    if wp >= WP_FULL:
        return 1.0
    return PUSH_AT_EMPTY + (1.0 - PUSH_AT_EMPTY) * wp / WP_FULL


def static_power_w(static_f):
    """Metabolic power (W) of holding at load fraction f with no movement."""
    return STATIC_W * max(static_f, 0.0) ** STATIC_EXP


def metabolic_J(F_N, v_ms, dt, static_f=0.0):
    """Metabolic joules of one limb group over dt: positive mechanical power
    F v costs F v / eta, negative (lowering, a fall caught) costs |F v| / 1.2,
    and a static hold adds 60 W f^1.5."""
    p = F_N * v_ms
    cost = p / ETA if p >= 0.0 else -p / ECC_DIV
    return (cost + static_power_w(static_f)) * dt


def fatigue_of(p, limb):
    """The fatigue the reflex slowing reads for one limb: the larger of the
    person's sleep fatigue p["fatigue"] and the limb's gam, clipped to 0..1.
    A tired hand slows that hand's reflex and nothing else."""
    base = p.get("fatigue", 0.0) or 0.0
    gam = p.get("body", {}).get("fat", {}).get(limb, {}).get("gam", 0.0)
    return min(max(base, gam), 1.0)
