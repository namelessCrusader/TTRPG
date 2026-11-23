"""GRIP: force and slip of a held thing (wave 15, lane 2c, spec S2 item 2).

A held thing is a load the hand's pads must carry by friction. ONE law for a tool, a
rope, a person's wrist, a ledge. Pure functions: no World, no state of their own; the
hand's fatigue lives in p["body"]["fat"][limb] (fatigue.py) and this module only steps it.

THE LAW
    N_req   = F_t / (mu_s * N_EFF)       the squeeze that just prevents a slide
    N_use   = min(N_max, SM * N_req)     the person squeezes with a safety margin SM
    N_max   = G_hand * c_fat * (1 - tendon_cut) * c_state
    HOLDS iff N_req <= N_max and the hand has not worn out (fatigue gam < 1).
  G_hand is the rested hand's maximum voluntary squeeze: the person's own muscle force
  (exertion.force_N without the limb state: strength, training, the person's draw), the
  off hand at 0.8 of it. With the dry skin pair (mu 0.5, two opposed pads) the hand
  holds exactly G_hand of tangential load, which is the number the old hold test used
  (weight <= strength), so a dry hold the old law allowed is still allowed. A wet or
  oiled pair needs a harder squeeze for the same load: the squeeze is the load fraction f
  that fatigue.step_static spends (Rohmert: ET = 21.9 f^-1.98 s), so wet hands tire sooner
  with nothing else written.
  ROPE ROUND A POST OR POLE: T_hold = T_load exp(-mu phi) (Euler-Eytelwein 1781, capstan).
  SLIDE: once the static test fails the thing slides against the kinetic friction
  mu_k N_max; the work mu_k N s is heat, booked by the caller where the sliding is done.

SOURCES
  SM 1.3: people grip with the least force that avoids slip plus a margin, nearly constant
    over surfaces (Johansson and Westling 1984, Exp Brain Res 56:550 and 53:393; the
    margin's value 1.2-1.4 is (u): from abstracts, not read in full).
  Grip strength: about 47 kg-force (460 N) for men 30-39, 29 kg for women (Mathiowetz et al.
    1985, Arch Phys Med Rehabil 66:69) (u: from memory). The sheet's strength_N of 400 N for
    the stock adult sits 13 percent under the male mean: kept (u).
  Skin friction on hard objects 0.4-0.5 dry, falling when wet (Adams et al. 2007, J R Soc
    Interface 4:329) (u); the pair numbers are matfric's, so there is ONE friction table.
  Reaction delay before a slipping grip is corrected: 75 ms (Johansson and Westling) (u).
"""
import math

from . import bodysheet as _bs
from . import exertion as _ex
from . import fatigue as _fat

N_EFF = 2.0                     # opposed pads of a pinch or wrap
MU_REF = 0.5                    # dry skin on a hard object (matfric); N_EFF * MU_REF = 1
SM = 1.3                        # safety margin of the squeeze over the least that holds (u)
OFF_HAND = _bs.OFF_HAND         # 0.8
REACT_DELAY_S = 0.075           # (u)
HOLD_LIMBS = {"left arm": "left arm", "right arm": "right arm"}


def dominant_arm(p):
    return str(p.get("handed", "right")) + " arm"


def hand_rested_N(p, arm):
    """G_hand: the rested maximum squeeze of that hand. The person's muscle force
    (strength, training, their own draw; no fatigue, no injury), the off hand 0.8 of it."""
    g = _ex.strength_of(p) * _ex.training_factor(p.get("training")) * _ex.person_z(p, "lift")
    return g if arm == dominant_arm(p) else g * OFF_HAND


def tendon_cut(p, arm):
    """The share of that forearm's flexor tendons cut (0..1), read from the body record
    (p["body"]["tendon_cut"][arm]; the wound code of the bleeding lane writes it)."""
    return min(max(float(((p.get("body") or {}).get("tendon_cut") or {}).get(arm, 0.0)), 0.0), 1.0)


def state_factor(p, arm):
    """c_state: what a hurt, cold or burned hand leaves of its squeeze, 0..1. Reads
    p["hand_state"][arm] when the injury code writes it; a whole hand otherwise."""
    return min(max(float((p.get("hand_state") or {}).get(arm, 1.0)), 0.0), 1.0)


def n_required(load_N, mu_s, n_eff=N_EFF):
    """The squeeze that just prevents a slide of a tangential load."""
    return float(load_N) / max(float(mu_s) * n_eff, 1e-9)


def capstan_hold(T_load_N, mu, phi_rad):
    """The tension the hand must supply to hold a rope that pulls T_load, round a post by
    phi radians: T_load exp(-mu phi) (Euler-Eytelwein)."""
    return float(T_load_N) * math.exp(-float(mu) * float(phi_rad))


def squeeze_use(N_req, N_max):
    return min(float(N_max), SM * float(N_req))


def capacity(p, arm, f_load):
    """N_max now: G_hand times the limb's fatigue capacity at that load fraction, the cut
    tendons and the state of the hand."""
    st = ((p.get("body") or {}).get("fat") or {}).get(arm) or _fat.new_state()
    return (hand_rested_N(p, arm) * _fat.capacity(st, f_load)
            * (1.0 - tendon_cut(p, arm)) * state_factor(p, arm))


def holds(p, arm, load_N, mu_s, wraps_rad=0.0):
    """Does that hand hold that tangential load against that pair of surfaces now?
    Needs only reading. An unconscious or dead holder is the caller's test."""
    st = ((p.get("body") or {}).get("fat") or {}).get(arm) or _fat.new_state()
    if st["gam"] >= 1.0:
        return False
    G = hand_rested_N(p, arm)
    load = capstan_hold(load_N, mu_s, wraps_rad) if wraps_rad > 0.0 else float(load_N)
    f_req = n_required(load, mu_s) / G if G > 0.0 else _fat.F_MAX
    return n_required(load, mu_s) <= capacity(p, arm, min(f_req, 1.0)) + 1e-12


def step(p, arm, load_N, mu_s, dt_s, wraps_rad=0.0):
    """One tick of a hold: the squeeze the person uses, the fatigue clock advanced (the
    limb's state is written to p["body"]["fat"][arm], the one place), and whether it holds.
    Returns {"holds", "N_req", "N_use", "f", "gam", "static_W"}."""
    G = hand_rested_N(p, arm)
    load = capstan_hold(load_N, mu_s, wraps_rad) if wraps_rad > 0.0 else float(load_N)
    N_req = n_required(load, mu_s)
    fat = p.setdefault("body", {}).setdefault("fat", {})
    st = fat.get(arm) or _fat.new_state()
    f_req = N_req / G if G > 0.0 else _fat.F_MAX
    N_max = capacity(p, arm, min(f_req, 1.0))
    ok = st["gam"] < 1.0 and N_req <= N_max + 1e-12
    N_use = squeeze_use(N_req, N_max)
    # the fatigue RATE reads the squeeze the task asks (against the RESTED maximum), not the one
    # the tired hand could still give: a constant hold ends at exactly ET(f) (fatigue.py, Miner)
    f = min(SM * N_req, G) / G if G > 0.0 else _fat.F_MAX
    new = _fat.step_static(st, f * G, G, dt_s)
    if "t_s" in st:
        new["t_s"] = st["t_s"]
    fat[arm] = new
    return {"holds": ok, "N_req": N_req, "N_use": N_use, "f": f, "gam": new["gam"],
            "static_W": _fat.static_power_w(f)}


def slide_heat_J(mu_k, N, slide_m):
    """The heat of a rope or handle running through a hand: mu_k N s."""
    return float(mu_k) * float(N) * float(slide_m)


def slide_accel(F_t, mu_k, N_max, m_kg):
    """a = (F_t - mu_k N_EFF N_max) / m of a thing slipping from a hand."""
    return (float(F_t) - float(mu_k) * N_EFF * float(N_max)) / max(float(m_kg), 1e-9)
