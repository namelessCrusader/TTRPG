"""TWINS: the closed form of a tick law's OWN recurrence, applied at jump time
(wave 13, T5, spec 3.9).

Why a twin and not a Process: `_breathe` writes blood_o2, met, smoke and cohb
every tick for every alive person who is not safe, so a persistent folded
process would have two owners while the world ticks. A twin is APPLIED AT JUMP
TIME ONLY. The person stays live, and the tick law keeps all the thresholds
(a man wakes by the existing wake_o2 line in `_law_life`, not because a
scheduler woke him).

The breath twin uses the tick law's OWN discrete factor, not the continuous
exp(): x_n = x* + (x0 - x*)(1 - r)^n. That is exact to float rounding (1.8e-12
over 432,000 ticks, measured), so golden stays SAME and nothing needs a
tolerance. The continuous exp() form would differ by 1.8e-7.

The constants are read from base18 `physiology.BREATH`, `materials.BODY` and
`materials.TICK_S` (read only). The mouth-air triple (gas, frac, smoke g/L) is
what `_law_life` computes before `_breathe`. Until the wiring extracts
`_mouth_air` (W3), `mouth_air()` here repeats that arithmetic; a test compares
it with a spy on `_breathe` during a real tick.

THE TWIN PROTOCOL
  keys                            person-dict keys this twin writes
  applicable(w, p) -> (ok, why)   the tier-0 test for this person
  enter(w, p) -> ctx              read the targets once
  horizon(w, p, ctx) -> [(k, tag)]   the first tick count k at which a fine-law
                                  threshold on these keys flips
  advance(w, p, ctx, n) -> fluxes write the keys at tick +n
  power_w(w, p, ctx) -> watts     heat this person puts into the room (tier 1)
A jump goes to k - 2 at most (the margin covers rounding) and the tick law
does the rest.

SLOT: the T7 body_heat twin (STEP class, implicit steps of 30-300 s) is
registered under "body_heat" by the body-heat lane with `register_twin`. Its
flux dict is {("E","heat","room"): J, ("mass","h2o","room"): g,
("mass","co2","room"): g, ("mass","o2","room"): -g, ("E","stores","body"): J}.
Nothing here depends on it; with none registered a living person is simply
whatever the breath twin says.
"""
import math

import numpy as np

from .faintrates import o2_target
from .materials import BODY, TICK_S
from .physiology import BREATH, PhysiologyMixin

LN2 = math.log(2.0)


def _pow1m(r, n):
    """(1 - r)^n without the rounding of forming 1 - r first: 1 - r for r near
    1e-6 has an absolute error of 1e-16, which n = 432,000 powers up to 5e-11.
    exp(n log1p(-r)) is exact to a few ulps. r >= 1 gives 0, r <= 0 gives 1."""
    if n <= 0 or r <= 0.0:
        return 1.0
    if r >= 1.0:
        return 0.0
    return math.exp(n * math.log1p(-r))


def geo_relax(x0, target, r, n):
    """The closed form of the discrete relaxation x <- x + (target - x) r
    applied n times: target + (x0 - target)(1 - r)^n. Generic: the breath
    twin's blood_o2 and met are this, so is any per-tick relaxation."""
    if n <= 0:
        return x0
    return target + (x0 - target) * _pow1m(r, n)


def geo_decay(x0, k, n):
    """x <- x (1 - k) applied n times (cohb, smoke)."""
    if n <= 0:
        return x0
    return x0 * _pow1m(k, n)


def first_tick(pred, n_max=1 << 28):
    """Smallest n in [1, n_max] with pred(n) true, or None. `pred` must be
    monotone from some point on (it is, for a relaxation toward a target):
    exponential search for a true value, then bisection on the integer tick."""
    if pred(1):
        return 1
    lo, hi = 1, 2
    while hi <= n_max and not pred(hi):
        lo, hi = hi, hi * 2
    if hi > n_max:
        if not pred(n_max):
            return None
        hi = n_max
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if pred(mid):
            hi = mid
        else:
            lo = mid
    return hi


class _Pred(PhysiologyMixin):
    """The real predicate `_delivered_o2` without a World: it reads only
    BLEED and the person's own keys."""


_PRED = _Pred()


def delivered_o2(p):
    """What `_law_life` compares with wake_o2 and faint_o2, by the real code."""
    return _PRED._delivered_o2(p)


class Twin:
    """Base class and protocol. Plain class: found by NAME in TWINS, never
    stored in a snapshot."""
    name = "twin"
    keys = ()

    def applicable(self, w, p):
        return False, "not implemented"

    def enter(self, w, p):
        return {}

    def horizon(self, w, p, ctx):
        return []

    def advance(self, w, p, ctx, n):
        return {}

    def power_w(self, w, p, ctx):
        return 0.0

    def o2_use_g(self, w, p, ctx, n):
        """Grams of O2 this person would use in n ticks (the V2 check)."""
        return 0.0


def mouth_air(w, p):
    """(gas, frac, smoke_gl) at this person's mouth: the arithmetic of
    `_law_life` before it calls `_breathe`, read only. None when the body has
    no flesh on the lattice."""
    flesh, sl = w._person_cells(p)
    if flesh is None or not flesh.any():
        return None
    return w._mouth_air(p, flesh, sl)


class BreathTwin(Twin):
    """blood_o2, met, smoke and cohb for a person in clean still air, by the
    discrete recurrences of `physiology._breathe`."""
    name = "breath"
    keys = ("blood_o2", "met", "smoke", "cohb")

    def applicable(self, w, p):
        if not p["alive"] or p["safe"]:
            return True, "dead or safe: nothing runs"
        if w.o2 is None:
            return False, "the first step has not been taken"
        if any(x.get("open") for x in (p.get("wounds") or ())):
            return False, "bleeding"
        if p["awake"] and not p.get("asleep"):
            return False, "awake"
        if p.get("skin", 0.0) > BODY["hurt_T"] or p["burn"] >= 1.0:
            return False, "burn integral rising"
        if (p.get("ko_until", 0) > w.tick or p.get("organ_dead")
                or p.get("vagal") or p.get("pain_acute") or p.get("pool_left_ml")):
            return False, "an injury clock is running"
        eye, last = p.get("_eye"), p.get("_met_at")
        if p["awake"] and (last is None or eye is None
                           or (float(eye[0]), float(eye[1])) != tuple(last)):
            return False, "moving"
        air = mouth_air(w, p)
        if air is None:
            return False, "no flesh on the lattice"
        gas, frac, sm = air
        if not gas:
            return False, "no gas at the mouth"
        if sm > 0.0:
            return False, "smoke at the mouth"
        return True, ""

    def enter(self, w, p):
        gas, frac, sm = mouth_air(w, p)
        dt = TICK_S
        want = BREATH["out_met"] if not p["awake"] else \
            (3.5 + 0.1 * 0.0) / 3.5
        return {"x": o2_target(frac),
                "r_b": dt / BREATH["tau_s"],
                "want": want,
                "r_m": min(dt / BREATH["met_s"], 1.0),
                "f_s": 0.5 ** (dt / BREATH["cough_half_s"]),
                "k_c": min(max(LN2 * dt / (BREATH["co_half_min"] * 60.0), 0.0), 1.0)}

    def _keys_at(self, p, ctx, n):
        b = geo_relax(p["blood_o2"], ctx["x"], ctx["r_b"], n)
        c = geo_decay(p.get("cohb", 0.0), ctx["k_c"], n)
        return b, c

    def horizon(self, w, p, ctx):
        """First tick count at which a threshold in `_law_life` flips. The
        predicate is the REAL `_delivered_o2` on a copy of the person's keys."""
        if not p["alive"] or p["safe"]:
            return []
        q = dict(p)
        out = []

        def at(n):
            q["blood_o2"], q["cohb"] = self._keys_at(p, ctx, n)
            return delivered_o2(q)
        if not p["awake"]:
            k = first_tick(lambda n: at(n) > BODY["wake_o2"])
            if k is not None:
                out.append((k, "wakes"))
        else:
            k = first_tick(lambda n: at(n) < BODY["faint_o2"])
            if k is not None:
                out.append((k, "faints"))
        k = first_tick(lambda n: at(n) < BODY["death_o2"])
        if k is not None:
            out.append((k, "dies"))
        return sorted(out)

    def advance(self, w, p, ctx, n):
        if n <= 0:
            return {}
        b, c = self._keys_at(p, ctx, n)
        p["blood_o2"] = b
        p["cohb"] = float(min(max(c, 0.0), 1.0))
        p["met"] = geo_relax(p.get("met", 1.0), ctx["want"], ctx["r_m"], n)
        p["smoke"] = max(0.0, p["smoke"] * _pow1m(1.0 - ctx["f_s"], n))
        return {}


class ConstSourceTwin(Twin):
    """A TEST DOUBLE for the T7 body_heat twin: a person is a constant source
    of heat and an O2 sink. 80 W and 0.25 L O2 per minute for a sleeper. It
    writes no person key. Register it under any name with register_twin."""
    name = "const_source"
    keys = ()

    def __init__(self, watts=80.0, o2_l_min=0.25, co2_l_min=0.2, name="const_source"):
        self.watts, self.o2_l_min, self.co2_l_min = watts, o2_l_min, co2_l_min
        self.name = name

    def applicable(self, w, p):
        if not p["alive"] or p["safe"]:
            return True, ""
        return True, ""

    def enter(self, w, p):
        return {}

    def power_w(self, w, p, ctx):
        return self.watts if (p["alive"] and not p["safe"]) else 0.0

    def o2_use_g(self, w, p, ctx, n):
        if not p["alive"] or p["safe"]:
            return 0.0
        return self.o2_l_min / 60.0 * n * TICK_S * 1.429

    def advance(self, w, p, ctx, n):
        if not p["alive"] or p["safe"] or n <= 0:
            return {}
        s = n * TICK_S
        return {("E", "heat", "room"): self.watts * s,
                ("mass", "o2", "room"): -self.o2_l_min / 60.0 * s * 1.429,
                ("mass", "co2", "room"): self.co2_l_min / 60.0 * s * 1.977}


TWINS = {"breath": BreathTwin()}
BODY_HEAT = "body_heat"             # the T7 slot: TWINS.get(BODY_HEAT) is None until T7


def register_twin(name, twin):
    TWINS[name] = twin
    return twin


def unregister_twin(name):
    TWINS.pop(name, None)


def person_twins(w, p):
    """[(name, twin, ctx)] for a person, or (None, reason) when any registered
    twin refuses. Dead and safe persons have no twin and are fine."""
    if not p["alive"] or p["safe"]:
        return [], ""
    out = []
    for name in sorted(TWINS, key=lambda n: (n != "breath", n)):      # the breath twin first, as it always was
        tw = TWINS[name]
        if tw is None:
            continue
        ok, why = tw.applicable(w, p)
        if not ok:
            return None, f"{name}: {why}"
        out.append((name, tw, tw.enter(w, p)))
    return out, ""
