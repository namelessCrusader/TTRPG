"""Thermoregulation as controllers (wave 13, T7): sweat, skin vessels, shiver.

Structure after Stolwijk and Hardy 1966, Gordon 1974, Fiala 1999/2001, JOS-3
(same family). Pure functions and plain-dict state; no World. All constants
are (u) until fitted to the oracle on STEADY cases and tested on held-out ones.

The reflex contract (reflex.py, wave 12 BODY lane) is a table of triggers with
a delay. A 1 Hz controller is not a trigger, so these rows are a new row KIND,
"continuous": they keep `_draw`'s per-person values and the `_reflex_off`
test, but not the delay gate. Autonomic control works asleep and unconscious
(people shiver and sweat in their sleep; only the dead do not), which is data
in the row (`works_when`), not a flag in code.

Behaviour (put on a coat, move to shade, drink) is the mind's; `heat_signals`
only turns the numbers into 0..1 menu inputs.
"""
import copy
import math
import zlib

import numpy as np

from .tissue import ZONES, NZONES, AREA_SHARE

# ── rows for the reflex contract (kind "continuous") ─────────────────────────
THERMO_ROWS = {
    "sweat":  {"kind": "continuous", "band": (3.0, 8.0), "tau_s": (60.0, 120.0),
               "works_when": "alive", "yields_to": (), "verb": "sweats"},
    "vessel": {"kind": "continuous", "band": (2.0, 5.0), "tau_s": (20.0, 60.0),
               "works_when": "alive", "yields_to": (), "verb": None},
    "shiver": {"kind": "continuous", "band": (3.0, 8.0), "tau_s": (10.0, 30.0),
               "works_when": "alive", "yields_to": (), "verb": "shivers"},
}
SCHMITT_ON, SCHMITT_OFF = 0.10, 0.05     # fraction of max: one event per onset, one per offset

# Neutral skin set points per zone, C (u). Mean (area weighted) about 33.5.
SET_SKIN = {"head": 35.0, "torso": 34.2}
for _s in ("left", "right"):
    for _p, _t in (("upper arm", 33.6), ("forearm", 33.4), ("hand", 33.0),
                   ("thigh", 33.6), ("shin", 33.2), ("foot", 32.8)):
        SET_SKIN[f"{_s} {_p}"] = _t
SET_CORE = 37.0                          # hypothalamus set point, C (u)

# Where sweat is made: share of the whole-body rate by zone (u), sums to 1.
_SWEAT_KIND = {"head": .10, "torso": .40, "upper arm": .04, "forearm": .035,
               "hand": .01, "thigh": .07, "shin": .035, "foot": .01}
# Vessel range of the skin blood-flow multiplier, (min, max) by zone kind. The
# hands and feet carry the widest range (arteriovenous shunts). All inside
# [VESSEL_MIN, VESSEL_MAX] = [0.1, 60].
VESSEL_MIN, VESSEL_MAX = 0.1, 60.0
_VESSEL_RANGE = {"head": (0.5, 5.0), "torso": (0.2, 8.0), "upper arm": (0.15, 15.0),
                 "forearm": (0.12, 25.0), "hand": (0.1, 60.0), "thigh": (0.15, 15.0),
                 "shin": (0.12, 25.0), "foot": (0.1, 60.0)}

# Controller constants (u): ranges drawn per person.
PARAMS = {
    # name: (low, high)
    "set_core":    (36.8, 37.2),
    "sw_k_c":      (0.08, 0.13),         # g/s per K core error (Stolwijk 0.37 L/h/K = 0.103)
    "sw_k_s":      (0.007, 0.012),       # g/s per K skin error (0.034 L/h/K = 0.0094)
    "sw_k_cs":     (0.02, 0.04),         # g/s per K^2
    "sw_max":      (0.35, 0.55),         # g/s whole body (1.3-2.0 L/h)
    "vd_k_dil":    (60.0, 140.0),        # dilation gain per K core warm error
    "vc_k_con":    (2.0, 5.0),           # constriction gain per K skin cold error
    "sh_max_w_kg": (3.5, 4.5),           # W/kg at full shivering (sustained 2-3x rest)
    "sh_k1":       (0.04, 0.06),         # per K^2
    "sh_k2":       (0.008, 0.012),       # per K^2
    "tau_sweat":   THERMO_ROWS["sweat"]["tau_s"],
    "tau_vessel":  THERMO_ROWS["vessel"]["tau_s"],
    "tau_shiver":  THERMO_ROWS["shiver"]["tau_s"],
    "delay_sweat": THERMO_ROWS["sweat"]["band"],
    "delay_vessel": THERMO_ROWS["vessel"]["band"],
    "delay_shiver": THERMO_ROWS["shiver"]["band"],
}
CORE_COLD_WEIGHT = 3.0                   # C-core error counts this many times a cold-skin error in vasoconstriction (u, fitted on zone_31C)
DEHYD_SHIFT_K_PER_PCT = 0.2              # core threshold shift for sweat, per percent mass lost (Sawka, u)
SW_MAX_LOSS_PER_PCT = 0.04               # sweat capacity lost per percent dehydration (u)
CIRCADIAN_K = 0.3                        # amplitude of the set point (u), peak at 17:00
SLEEP_SHIFT_K = -0.3                     # asleep lowers the core set point (u)
DT_BRAIN = 0.2                           # hypothalamus minus pool, K (u)

# Core-temperature depression of shivering: a DATA CURVE, not Q10 (the Q10
# law stays in tissue metabolism only). (core C, factor).
SHIVER_CURVE = ((30.0, 0.0), (32.5, 0.5), (35.5, 1.0))


def _kind(zone):
    return zone.split(" ", 1)[1] if zone[:4] in ("left", "righ") else zone


_SW_SHARE_RAW = {z: _SWEAT_KIND[_kind(z)] for z in ZONES}
_SW_SUM = math.fsum(_SW_SHARE_RAW.values())
SWEAT_SHARE = {z: v / _SW_SUM for z, v in _SW_SHARE_RAW.items()}
VESSEL_RANGE = {z: _VESSEL_RANGE[_kind(z)] for z in ZONES}


# ── per-person draws, the same recipe as reflex._draw ────────────────────────
def draw(pid, name, param, lo=None, hi=None):
    """A value in [lo, hi] that is this person's own: a hash of the parameter
    name, the pid and the name. Independent of dict order and of any random
    number generator."""
    if lo is None:
        lo, hi = PARAMS[param]
    u = (zlib.crc32(f"{param}|{pid}|{name}".encode()) & 0xFFFFFFFF) / 2.0 ** 32
    return float(lo + (hi - lo) * u)


def person_params(pid, name):
    """All of a person's controller constants (a plain dict, deep-copy safe)."""
    return {k: draw(pid, name, k) for k in sorted(PARAMS)}


def set_points(prm, hour=12.0, asleep=False, offset=0.0):
    """Core set point now (C): the person's own, plus the daily rhythm
    (0.3 K, peak 17:00), minus 0.3 K asleep, plus a fever offset the infection
    law can write."""
    return (prm["set_core"] + CIRCADIAN_K * math.cos(2.0 * math.pi * (hour - 17.0) / 24.0)
            + (SLEEP_SHIFT_K if asleep else 0.0) + offset)


# ── shiver depression curve ──────────────────────────────────────────────────
def shiver_depression(T_core):
    """1.0 at >= 35.5 C, 0.5 at 32.5 C, 0 at <= 30 C; linear between (u)."""
    xs = [c[0] for c in SHIVER_CURVE]
    ys = [c[1] for c in SHIVER_CURVE]
    return float(np.interp(T_core, xs, ys))


# ── error signals ────────────────────────────────────────────────────────────
def signals(T_hyp, T_sk, set_core, set_sk=None, area=None, dehyd_pct=0.0):
    """Error signals, K. `T_sk` is {zone: skin C}. Warm/cold parts:
    W_c = max(E_c, 0), C_c = max(-E_c, 0), W_s = sum a_z max(E_s,z, 0),
    C_s = sum a_z max(-E_s,z, 0), a_z the true-area weights (sum 1).
    `E_c_sweat` is the core error with the dehydration shift on the sweat
    threshold (+0.2 K per percent of body mass lost)."""
    set_sk = SET_SKIN if set_sk is None else set_sk
    area = AREA_SHARE if area is None else area
    a_sum = math.fsum(area[z] for z in ZONES)
    E_c = T_hyp - set_core
    E_s = {z: T_sk[z] - set_sk[z] for z in ZONES}
    W_s = math.fsum(area[z] / a_sum * max(E_s[z], 0.0) for z in ZONES)
    C_s = math.fsum(area[z] / a_sum * max(-E_s[z], 0.0) for z in ZONES)
    E_cs = E_c - DEHYD_SHIFT_K_PER_PCT * dehyd_pct
    return {"E_c": E_c, "W_c": max(E_c, 0.0), "C_c": max(-E_c, 0.0),
            "W_c_sweat": max(E_cs, 0.0), "E_s": E_s, "W_s": W_s, "C_s": C_s,
            "dehyd_pct": dehyd_pct}


class Outputs:
    """What the controllers want right now (before the lags). All plain
    numbers: sweat_g_s {zone: g/s}, vessel {zone: multiplier of basal skin
    flow, 1 = no change}, shiver_w (W, whole body)."""
    __slots__ = ("sweat_g_s", "vessel", "shiver_w")

    def __init__(self, sweat_g_s, vessel, shiver_w):
        self.sweat_g_s, self.vessel, self.shiver_w = sweat_g_s, vessel, shiver_w

    def sweat_total(self):
        return math.fsum(self.sweat_g_s.values())


def targets(sig, prm, mass_kg, T_core, fatigue=1.0):
    """The instantaneous demand, before lags.

    sweat   sw = sw_max' min(raw/sw_max', 1), raw = k_c W_c + k_s W_s + k_cs W_c W_s
            (g/s), where W_c carries the dehydration shift and sw_max' falls with
            dehydration. Shared out by SWEAT_SHARE x 2^(E_s,z/10), rescaled so
            the total never passes sw_max'.
    vessel  m_z = (1 + k_dil W_c')/(1 + k_con C_s) x 2^(E_s,z/6), clipped to the
            zone's range (all inside [0.1, 60])
    shiver  P = sh_max m clip(k1 C_c C_s + k2 C_s^2, 0, 1) x fatigue x D(T_core)"""
    Wc = sig["W_c_sweat"]
    sw_max = prm["sw_max"] * max(1.0 - SW_MAX_LOSS_PER_PCT * sig["dehyd_pct"], 0.2)
    # TWO-SIDED (wave 13b, C6; the JOS-3 finding zone_31C). The core term is SIGNED:
    # a core under its set point takes sweat away from what warm skin asks for
    # (Stolwijk 1971 NASA CR-1855; Gagge, Stolwijk and Nishi 1971; Fiala et al.
    # 1999/2001: the sweat command is a sum of the signed hypothalamic error and the
    # warm skin signal, kept at or above zero) (u: read from memory). With only the warm
    # part, a body 0.4 K under its set point at 31 C still sweated and kept its skin
    # vessels open, and the core sagged by 0.4 K in the neutral zone.
    Ec_s = sig["E_c"] - DEHYD_SHIFT_K_PER_PCT * sig["dehyd_pct"]
    raw = max(prm["sw_k_c"] * Ec_s + prm["sw_k_s"] * sig["W_s"]
              + prm["sw_k_cs"] * Wc * sig["W_s"], 0.0)
    total = min(raw, sw_max)
    loc = {z: SWEAT_SHARE[z] * 2.0 ** (sig["E_s"][z] / 10.0) for z in ZONES}
    ls = math.fsum(loc.values())
    # the local factor moves the sweat toward warm zones without changing the total
    sweat = {z: total * loc[z] / ls for z in ZONES}
    dil = 1.0 + prm["vd_k_dil"] * sig["W_c"]
    # a core under its set point shuts the skin vessels as cold skin does (Fiala 1999: the
    # vasoconstriction command has a core term as well as a skin term) (u); the core is the
    # stronger sensor (CORE_COLD_WEIGHT K of skin error per K of core error, u)
    con = 1.0 + prm["vc_k_con"] * (sig["C_s"] + CORE_COLD_WEIGHT * sig["C_c"])
    vessel = {}
    for z in ZONES:
        lo, hi = VESSEL_RANGE[z]
        m = dil / con * 2.0 ** (sig["E_s"][z] / 6.0)
        vessel[z] = min(max(m, lo), hi)
    drive = min(max(prm["sh_k1"] * sig["C_c"] * sig["C_s"] + prm["sh_k2"] * sig["C_s"] ** 2, 0.0), 1.0)
    shiver = prm["sh_max_w_kg"] * mass_kg * drive * fatigue * shiver_depression(T_core)
    return Outputs(sweat, vessel, shiver)


# ── lags with dead time ──────────────────────────────────────────────────────
def new_state():
    """Controller state: current (lagged) outputs and the history of targets
    for the dead time. Plain lists and dicts: deep-copy and snapshot safe."""
    return {"t": 0.0,
            "sweat": {z: 0.0 for z in ZONES},
            "vessel": {z: 1.0 for z in ZONES},
            "shiver": 0.0,
            "hist": []}                   # (t, sweat dict, vessel dict, shiver)


def _lag(y, target, dt, tau):
    """Exact first-order lag over dt: no overshoot for any dt."""
    a = 1.0 - math.exp(-dt / tau)
    return y + (target - y) * a


def controllers(sig, prm, state, dt, mass_kg, T_core, fatigue=1.0):
    """Advance the controllers by `dt` seconds (any dt; the lag is exact).
    Targets go through a dead time (a transport delay per channel), then a
    first-order lag with the person's time constant. Mutates and returns
    `state`; returns (state, lagged Outputs). The history is trimmed to the
    longest dead time."""
    tgt = targets(sig, prm, mass_kg, T_core, fatigue)
    start = state["t"]
    now = start + dt
    # an entry is (t_start, targets): the demand applies from t_start on
    state["hist"].append((start, dict(tgt.sweat_g_s), dict(tgt.vessel), tgt.shiver_w))
    horizon = max(prm["delay_sweat"], prm["delay_vessel"], prm["delay_shiver"])
    # keep the newest entry at or before (now - horizon), and everything later
    while len(state["hist"]) > 1 and state["hist"][1][0] <= now - horizon:
        state["hist"].pop(0)
    neutral = (None, {z: 0.0 for z in ZONES}, {z: 1.0 for z in ZONES}, 0.0)

    def seen(delay):
        """The demand as it was `delay` seconds ago: the newest entry that had
        started by then, or the neutral output if none had (the dead time)."""
        t_want = now - delay
        pick = neutral
        for h in state["hist"]:
            if h[0] <= t_want:
                pick = h
            else:
                break
        return pick

    hs, hv, hh = seen(prm["delay_sweat"]), seen(prm["delay_vessel"]), seen(prm["delay_shiver"])
    for z in ZONES:
        state["sweat"][z] = _lag(state["sweat"][z], hs[1][z], dt, prm["tau_sweat"])
        state["vessel"][z] = _lag(state["vessel"][z], hv[2][z], dt, prm["tau_vessel"])
    state["shiver"] = _lag(state["shiver"], hh[3], dt, prm["tau_shiver"])
    state["t"] = now
    return state, Outputs(dict(state["sweat"]), dict(state["vessel"]), state["shiver"])


# ── events with hysteresis (a Schmitt trigger) ───────────────────────────────
def schmitt(was_on, value, vmax, on=SCHMITT_ON, off=SCHMITT_OFF):
    """(is_on, event) with event in (None, "on", "off"). On above `on` x vmax,
    off below `off` x vmax: one event per onset and one per offset."""
    if vmax <= 0.0:
        return (False, "off" if was_on else None)
    f = value / vmax
    if not was_on and f > on:
        return True, "on"
    if was_on and f < off:
        return False, "off"
    return was_on, None


def events(prm, out, mass_kg, flags):
    """Run the two Schmitt triggers for sweat and shiver. `flags` is
    {"sweat": bool, "shiver": bool} (mutated). Returns a list of
    (verb, "on"|"off") in a fixed order: ("shivers", ...), ("sweats", ...)."""
    evs = []
    for name, val, vmax in (("shiver", out.shiver_w, prm["sh_max_w_kg"] * mass_kg),
                            ("sweat", out.sweat_total(), prm["sw_max"])):
        is_on, ev = schmitt(flags.get(name, False), val, vmax)
        flags[name] = is_on
        if ev:
            evs.append((THERMO_ROWS[name]["verb"], ev))
    return evs


def works(row, p):
    """Whether a continuous row works for this person: by `works_when`. "alive"
    = not dead (asleep and unconscious still shiver); "awake" = also not
    asleep or unconscious."""
    if p.get("alive") is False:
        return False
    if row.get("works_when", "awake") == "alive":
        return True
    return not (p.get("asleep") or p.get("unconscious"))


# ── what the mind may see ────────────────────────────────────────────────────
HEAT_THRESH = {"cold_core_k": 1.5, "cold_skin_k": 6.0, "hot_core_k": 1.5, "hot_skin_k": 3.0,
               "thirst_lo_pct": 0.5, "thirst_span_pct": 2.0}


def heat_signals_from(T_core, T_sk_mean, set_core, set_sk_mean, dehyd_pct=0.0, hunger=0.0):
    """cold, hot, thirsty, hungry, each in 0..1, from numbers and data thresholds."""
    th = HEAT_THRESH
    cold = max((set_core - T_core) / th["cold_core_k"], (set_sk_mean - T_sk_mean) / th["cold_skin_k"], 0.0)
    hot = max((T_core - set_core) / th["hot_core_k"], (T_sk_mean - set_sk_mean) / th["hot_skin_k"], 0.0)
    thirsty = (dehyd_pct - th["thirst_lo_pct"]) / th["thirst_span_pct"]
    c = lambda v: float(min(max(v, 0.0), 1.0))
    return {"cold": c(cold), "hot": c(hot), "thirsty": c(thirsty), "hungry": c(hunger)}


def heat_signals(p):
    """The same from a person's p["heat"] record (numbers only; missing keys read as comfortable)."""
    h = p.get("heat") or {}
    return heat_signals_from(h.get("T_pool", SET_CORE), h.get("T_skin_mean", 33.5),
                             h.get("set_core", SET_CORE), h.get("set_skin_mean", 33.5),
                             h.get("dehyd_pct", 0.0), h.get("hunger", 0.0))
