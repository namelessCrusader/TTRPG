"""LOCKS AND KEYS (wave 13, T11): a pin-tumbler lock as geometry.

A lock is a plug in a cylinder, held by pin stacks that cross the shear line.
A key lifts each stack by its cut; a pick lifts them one at a time while a
wrench turns the plug a hair, so that one pin at a time BINDS. Nothing here
rolls a die for success: whether a lock opens is its geometry, the tension
and the hand (a skill number that changes how well the hand feels and holds).

THE NUMBERS. Rises are QUANTISED: a factory cuts depths in whole steps (0.38
mm, Schlage; Kwikset 0.58), which is what keeps two random locks from
accepting each other's key. ONE tolerance window w_i per pin (0.05 to 0.09
mm) says how far the pin interface may sit from the shear line and still let
the plug turn: the key and the pick both use it. A key off by one depth step
is 4 to 7 windows out.

All lengths here are millimetres (a lock is small), forces newtons, torques
N m. Random draws use `hash01` (events.py) on the lock's seed or the picker's
id, probe count and lock id: the same ids replay the same probes.

(u) = unverified. Sources: Tobias, Locks, Safes and Security (pin-tumbler);
Blaze 2003; lockwiki (Kwikset depth step 0.023 in).
"""
import math
from statistics import NormalDist

from . import mechanisms as mech
from .events import hash01
from .materials import IRON

try:                                    # A-G10: the person's skill at an act
    from . import bodysheet as _bodysheet
except ImportError:
    _bodysheet = None

# geometry (Schlage-type, (u) from lock-picking references)
PLUG_R = 6.35            # mm
PIN_D = 2.92             # mm
PITCH = 3.96             # mm between pin centres
STEP_MM = 0.38           # depth step
BASE_MM = 1.60           # (u) shallowest rise
W_LO, W_HI = 0.05, 0.09  # mm, one window per pin
E_LO, E_HI = 0.005, 0.05  # mm, the misalignment of each bore: sets the binding order
MU_PIN = 0.3             # (u) friction of a pin on its bore
SPRING_GEOM = {"d": 0.30e-3, "D": 2.5e-3, "n": 8, "G": 79e9}   # 0.64 N/mm
SPRING_PRELOAD_MM = 1.0
BOLT_TORQUE = 0.15       # N m to throw the bolt (u)
TAU_BIND = 0.01          # N m: under it no pin binds (u)
TAU_HOLD = 0.005         # N m: under it a set pin falls back (u)
TIP_FORCE_MAX = 4.0      # N the pick tip can push before it bends (u)
WRENCH_YIELD = 0.175     # N m, a 3 x 0.5 mm spring steel wrench (u)
SHEAR_BRASS = 200e6      # Pa (u)
SY_BRASS = 200e6         # Pa, key bow yield (u: 100 to 250)
KEY_BOW_Z_MM3 = 15.0     # section modulus of a key bow: M_y = 3 N m
CLICK_DX_MM = 0.3        # a pin dropping to the shear line
ETA_CLICK = 8e-3         # fraction of the pin's F dx radiated as a click (u, FITTED):
#                          # chosen so the real _noise law gives 36.9 dB at 1 m
ND = NormalDist()
SPOOL_EXTRA = 4.0        # tension must fall under TAU_HOLD x this to clear a false set


def pin_spring():
    """The pin spring from its geometry: a coil of 0.30 mm wire, 2.5 mm coil,
    8 turns: 0.64 N/mm."""
    g = SPRING_GEOM
    return mech.new_spring("pin", "coil", "steel", g)


def spring_force_N():
    """F_spring on a seated pin: rate x preload, 0.64 N."""
    return mech.coil_rate(SPRING_GEOM["d"], SPRING_GEOM["D"], SPRING_GEOM["n"],
                          SPRING_GEOM["G"]) * SPRING_PRELOAD_MM * 1e-3


def tau_max(F_pick_max=TIP_FORCE_MAX, mu=MU_PIN, r=PLUG_R * 1e-3):
    """The top of the tension window: the tension at which lifting the binding
    pin needs the whole tip force. (F_pick_max - F_spring) r / mu = 0.071 N m."""
    return (F_pick_max - spring_force_N()) * r / mu


def window_Nm():
    """(tau_bind, tau_max): the tensions at which a pick can work."""
    return TAU_BIND, tau_max()


def gauss(u):
    """A standard normal from a uniform in (0, 1)."""
    return ND.inv_cdf(min(max(u, 1e-9), 1.0 - 1e-9))


def skill_pick(p=None, default=0.1):
    """The person's skill at picking: the bodysheet's drawn default or stored
    p["skill"]["pick"] when bodysheet exists, else `default`."""
    if p is not None and _bodysheet is not None:
        try:
            return float(_bodysheet.skill_of(p, "pick"))
        except (AttributeError, KeyError, TypeError):
            pass
    return default


# ── the lock and the key ─────────────────────────────────────────────────────
def make_lock(seed, n=5, lock_id=None, spool=0.0, seize_Nm=0.0, step_mm=STEP_MM,
              base_mm=BASE_MM, keyway=0b0111_1110):
    """A pin-tumbler lock from a seed. Pin i: n_step (0..9), the bore error
    e (mm) that sets which pin binds first, the window w (mm), a driver
    ("spool" with probability `spool`), and the picking state (lift_mm, set).
    The rise of its stack is base + n_step x step."""
    pins = []
    for i in range(n):
        pins.append({"n_step": int(math.floor(10 * hash01(seed, "n", i))),
                     "e_mm": E_LO + (E_HI - E_LO) * hash01(seed, "e", i),
                     "w_mm": W_LO + (W_HI - W_LO) * hash01(seed, "w", i),
                     "driver": "spool" if hash01(seed, "d", i) < spool else "plain",
                     "lift_mm": 0.0, "set": False, "overset": False,
                     "sheared": False})
    return {"id": lock_id if lock_id is not None else "lock-%s" % (seed,),
            "kind": "pin", "n": n, "seed": seed, "pitch_mm": PITCH,
            "plug_r_mm": PLUG_R, "pin_d_mm": PIN_D, "step_mm": step_mm,
            "base_mm": base_mm, "mu_pin": MU_PIN, "pins": pins,
            "keyway": int(keyway), "bolt_Nm": BOLT_TORQUE,
            "seize_Nm": float(seize_Nm), "false_set": False,
            "obstructed": False, "rot_rad": 0.0}


def rise_mm(lock, i):
    """Where pin i's interface sits on a seated key cut: base + n_step x step."""
    return lock["base_mm"] + lock["pins"][i]["n_step"] * lock["step_mm"]


def make_key(cuts, wear_mm=None, blade=0b0111_1110, mat="brass",
             bow_Z_mm3=KEY_BOW_Z_MM3):
    """A key. `cuts` are depth steps; `wear_mm` (per cut) is how much shorter
    a worn or badly copied cut is than its step says; `blade` is the profile
    as a bitmask (the wards)."""
    return {"cuts": [int(c) for c in cuts],
            "wear_mm": list(wear_mm) if wear_mm is not None else [0.0] * len(cuts),
            "blade": int(blade), "mat": mat, "bow_Z_mm3": float(bow_Z_mm3)}


def key_for(lock):
    """The key a locksmith cuts for this lock: its steps, a fitting blade."""
    return make_key([p["n_step"] for p in lock["pins"]], blade=lock["keyway"])


def copy_key(lock, key, sigma_mm, seed):
    """A copy of `key` cut by a machine with error sigma_mm per cut (wear
    drawn from hash01(seed, i)): the same steps, a different bitting in mm."""
    out = dict(key)
    out["wear_mm"] = [key["wear_mm"][i] + sigma_mm * gauss(hash01(seed, "copy", i))
                      for i in range(len(key["cuts"]))]
    return out


def cut_mm(lock, key, i):
    """The lift key gives pin i: base + cut x step - wear."""
    return lock["base_mm"] + key["cuts"][i] * lock["step_mm"] - key["wear_mm"][i]


def blade_fits(lock, key):
    """The wards: the blade enters if it has no part outside the keyway."""
    return (key["blade"] & ~lock["keyway"]) == 0


def key_matches(lock, key):
    """Every pin within ITS window: |cut - rise| <= w_i."""
    if len(key["cuts"]) != lock["n"]:
        return False
    return all(abs(cut_mm(lock, key, i) - rise_mm(lock, i))
               <= lock["pins"][i]["w_mm"] for i in range(lock["n"]))


def key_yield_Nm(key):
    """M_y of the key bow: sigma_y (brass) x Z."""
    return SY_BRASS * key["bow_Z_mm3"] * 1e-9


def plug_need_Nm(lock):
    """The torque to turn the plug: the bolt spring plus any seize (rust)."""
    return lock["bolt_Nm"] + lock["seize_Nm"]


def try_key(lock, key, T_hand):
    """Put `key` in and turn with up to T_hand N m. Returns a dict with
    "result": "wrong_blade" (never enters), "wrong_cuts" (the pins hold the
    plug), "stuck" (right key, the lock needs more than the hand gives),
    "snaps" (the torque on the key reaches its yield first: the bow is gone,
    no chance involved) or "turns". The key carries min(T_hand, torque the
    lock needs); a wrong key meets the pins, which hold against any torque the
    key survives."""
    if key.get("bow_Z_mm3", 0.0) <= 0.0:
        return {"result": "no_bow", "T_on_key": 0.0}
    if not blade_fits(lock, key):
        return {"result": "wrong_blade", "T_on_key": 0.0}
    match = key_matches(lock, key) and not lock["obstructed"]
    need = plug_need_Nm(lock) if match else math.inf
    T_on = min(T_hand, need)
    My = key_yield_Nm(key)
    if T_on > My:
        key["bow_Z_mm3"] = 0.0
        key["broken"] = True
        lock["obstructed"] = True               # a piece stays in the keyway
        return {"result": "snaps", "T_on_key": My}
    if not match:
        return {"result": "wrong_cuts", "T_on_key": T_on}
    if T_hand >= need:
        return {"result": "turns", "T_on_key": T_on}
    return {"result": "stuck", "T_on_key": T_on}


# ── picking ──────────────────────────────────────────────────────────────────
def reset_pins(lock):
    for p in lock["pins"]:
        p["lift_mm"], p["set"], p["overset"] = 0.0, False, False
    lock["false_set"] = False
    lock["rot_rad"] = 0.0


def apply_tension(lock, tau):
    """What the wrench does to the pins' state. Under TAU_HOLD nothing holds
    them: set pins fall back, an overset pin falls (tension released), a
    false set clears. Returns True if the state was lost."""
    if tau >= TAU_HOLD:
        return False
    lost = any(p["set"] or p["overset"] for p in lock["pins"])
    reset_pins(lock)
    return lost


def binding_pin(lock, tau):
    """The pin that binds at this tension: the unset pin of largest bore error
    e (it is the first the turning plug presses on), if tau >= TAU_BIND."""
    if tau < TAU_BIND:
        return None
    unset = [i for i, p in enumerate(lock["pins"])
             if not p["set"] and not p["sheared"]]
    if not unset:
        return None
    return max(unset, key=lambda i: lock["pins"][i]["e_mm"])


def lift_force(lock, i, tau):
    """Force to lift pin i: the spring, plus friction of the plug's pressure
    N = tau / r on the BINDING pin only."""
    F = spring_force_N()
    if i == binding_pin(lock, tau):
        F += lock["mu_pin"] * tau / (lock["plug_r_mm"] * 1e-3)
    return F


def feel_ratio(lock, i, tau):
    """The resistance of pin i over a free pin's, minus 1: what the picker
    senses (mu N / F_spring on the binding pin, 0 elsewhere)."""
    return lift_force(lock, i, tau) / spring_force_N() - 1.0


def weber(skill):
    """The smallest extra resistance a hand notices: 0.4 (novice) to 0.1
    (expert) (u)."""
    return 0.4 - 0.3 * skill


def tip_force_max(tool=None):
    """What the pick tip can push: its rating times (1 - D)."""
    if tool is None:
        return TIP_FORCE_MAX
    return tool.get("tip_force_max_N", TIP_FORCE_MAX) * (1.0 - tool.get("D", 0.0))


def make_pick(tip_force_max_N=TIP_FORCE_MAX, wrench_yield_Nm=WRENCH_YIELD):
    """A pick set as a thing's tool record. Nothing breaks by chance."""
    return {"kind": "pick", "tip_force_max_N": tip_force_max_N,
            "wrench_yield_Nm": wrench_yield_Nm, "D": 0.0}


def load_wrench(tool, tau):
    """The wrench carries tau; past its yield it bends: damage dD = 1 - yield/
    tau (T9's combine), and the tip force it can then steer falls with (1 - D)."""
    y = tool["wrench_yield_Nm"] * (1.0 - tool["D"])
    if tau <= y:
        return tool
    tool["D"] = mech.combine(tool["D"], 1.0 - y / tau)
    return tool


def probe(lock, i, tau, lift_mm, tool=None):
    """Lift pin i to `lift_mm` under tension `tau`. Returns the feel:
    "set_already", "springy" (a free pin: it springs back), "jammed" (the
    binding pin needs more than the tip can push), "short" (lifted under its
    window: falls back), "click" (set: the interface is at the shear line to
    within its window), "overset" (past the window: it blocks the plug until
    the tension is released), or "false_set" (a spool pin is holding; ease the
    tension)."""
    pins = lock["pins"]
    p = pins[i]
    if p["set"] or p["sheared"]:
        return "set_already"
    if tau < TAU_HOLD:
        apply_tension(lock, tau)
        return "springy"
    if lock["false_set"]:
        if tau < TAU_HOLD * SPOOL_EXTRA:
            lock["false_set"] = False          # the spool's waist clears
            return "springy"
        return "false_set"
    if i != binding_pin(lock, tau):
        return "springy"
    if lift_force(lock, i, tau) > tip_force_max(tool):
        return "jammed"
    err = lift_mm - rise_mm(lock, i)
    if abs(err) <= p["w_mm"]:
        p["set"], p["lift_mm"] = True, lift_mm
        nxt = binding_pin(lock, tau)
        lock["rot_rad"] = (max(q["e_mm"] for q in pins) -
                           (pins[nxt]["e_mm"] if nxt is not None else 0.0)
                           ) / lock["plug_r_mm"]
        if p["driver"] == "spool":
            lock["false_set"] = True
        return "click"
    if err > p["w_mm"]:
        p["overset"], p["lift_mm"] = True, lift_mm
        return "overset"
    return "short"


def is_open(lock, tau):
    """The plug turns: every pin set (or sheared), none overset, tension held,
    no false set."""
    return (tau >= TAU_HOLD and not lock["false_set"] and not lock["obstructed"]
            and all(p["set"] or p["sheared"] for p in lock["pins"])
            and not any(p["overset"] for p in lock["pins"]))


def probe_time_s(skill):
    """Seconds per probe: 3.0 (1 - 0.8 skill) (u)."""
    return 3.0 * (1.0 - 0.8 * skill)


def pick_session(lock, tau, skill, pid="p", tool=None, max_probes=60):
    """A picker with wrench tension `tau` and `skill` (0..1) works the lock,
    one probe at a time, until it opens or the budget is gone.

    The tension wobbles by sigma = 0.5 (1 - skill) tau (drawn from hash01 of
    pid, lock id, probe count); the picker feels the binding pin as extra
    resistance when it exceeds weber(skill), and then lifts it, overshooting
    the shear line by sigma_lift = 0.12 (1 - skill) mm; an overset is felt at
    once and the picker releases the tension and starts over; a false set
    (spool pin) is met by easing the tension for one probe. Returns a dict:
    opened, probes, time_s, jams, oversets, restarts."""
    sigma_t = 0.5 * (1.0 - skill) * tau
    sigma_l = 0.12 * (1.0 - skill)
    res = {"opened": False, "probes": 0, "time_s": 0.0, "jams": 0,
           "oversets": 0, "restarts": 0}
    ptr = 0
    n = lock["n"]
    tool = tool if tool is not None else make_pick()
    k = 0
    while k < max_probes:
        tau_eff = max(0.0, tau + sigma_t * gauss(hash01(pid, lock["id"], k, "tau")))
        if lock["false_set"]:
            tau_eff = min(tau_eff, 0.6 * TAU_HOLD * SPOOL_EXTRA)   # ease off
        load_wrench(tool, tau_eff)
        apply_tension(lock, tau_eff)
        k += 1
        res["probes"], res["time_s"] = k, res["time_s"] + probe_time_s(skill)
        if is_open(lock, tau_eff):
            res["opened"] = True
            return res
        unset = [i for i in range(n) if not lock["pins"][i]["set"]]
        if not unset:
            if lock["false_set"] and tau_eff < TAU_HOLD * SPOOL_EXTRA:
                lock["false_set"] = False       # eased: the spool's waist passes
            continue
        i = unset[ptr % len(unset)]
        if lock["false_set"] or feel_ratio(lock, i, tau_eff) >= weber(skill):
            lift = rise_mm(lock, i) + sigma_l * gauss(hash01(pid, lock["id"], k, "lift"))
            r = probe(lock, i, tau_eff, lift, tool)
        else:
            r = "springy"
            ptr += 1
        if r == "click":
            ptr = 0
        elif r == "jammed":
            res["jams"] += 1
            ptr += 1
        elif r == "overset":
            res["oversets"] += 1
            reset_pins(lock)                    # the picker lets go and starts over
            res["restarts"] += 1
            ptr = 0
        elif r == "short":
            ptr += 1
        if is_open(lock, tau_eff):
            res["opened"] = True
            return res
    return res


# ── forcing ──────────────────────────────────────────────────────────────────
def forcing_torque_Nm(lock):
    """The plug torque that shears one binding pin at the shear line:
    N = tau_plug / r <= S_shear A, A = pi d^2/4 of the pin, brass 200 MPa:
    8.5 N m. Pins bind one at a time, so this is the most the plug needs."""
    A = math.pi * (lock["pin_d_mm"] * 1e-3) ** 2 / 4.0
    return lock["plug_r_mm"] * 1e-3 * SHEAR_BRASS * A


def force_open(lock, tau_plug, tool=None):
    """Turn the plug by brute torque. The tool is checked against its own
    yield first: a pick set's wrench bends at WRENCH_YIELD long before 8.5
    N m, so forcing needs a tool that can carry it (tool=None means a stout
    one, a screwdriver). Returns a dict: opened, sheared (pin indices), and
    "tool_yields" if the wrench gave out."""
    if tool is not None:
        y = tool["wrench_yield_Nm"] * (1.0 - tool["D"])
        if tau_plug > y:
            load_wrench(tool, tau_plug)
            return {"opened": False, "sheared": [], "tool_yields": True}
    if tau_plug < forcing_torque_Nm(lock) or lock["obstructed"]:
        return {"opened": False, "sheared": [], "tool_yields": False}
    sheared = []
    for i, p in enumerate(lock["pins"]):
        if not p["set"]:
            p["sheared"] = True                 # the pin is gone, no flag on the lock
            sheared.append(i)
    return {"opened": True, "sheared": sheared, "tool_yields": False}


# ── sound ────────────────────────────────────────────────────────────────────
def click_energy_J(eta=ETA_CLICK, F_spring=None, dx_mm=CLICK_DX_MM):
    """The radiated energy of a pin dropping to the shear line: a small
    fraction eta of the mechanical F dx (0.64 N x 0.3 mm = 1.9e-4 J). eta is
    DATA, fitted so that the real `_noise` law (whose ring constants
    over-predict small clicks) gives a click of 30 to 40 dB at 1 m."""
    F = spring_force_N() if F_spring is None else F_spring
    return eta * F * dx_mm * 1e-3


def emit_click(w, at, mat=IRON, ev=None, pid=None, eta=ETA_CLICK):
    """Make the click through the world's own `_noise` (picks is not a
    sounding verb; the click is passed as `ev`). Returns the sound dict."""
    return w._noise(at, click_energy_J(eta), [int(mat)], ev=ev, pid=pid)
