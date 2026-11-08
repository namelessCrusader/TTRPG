"""THE ONE REFLEX CONTRACT: what a body does by itself, below the mind.

A reflex is not a decision. It runs every tick, costs the model nothing, and
is written once, here, so that balance today and the pain withdrawal, the
breath-hold, the startle, the raised voice in noise and the rest tomorrow all
behave the same way. A new reflex is a new ROW in `REFLEX` plus the law that
says when it is wanted; it does not get a contract of its own.

The contract, in plain words:

1. IT HAS A DELAY, and the delay is the person's own. Each row names a band in
   seconds (balance: 90 to 120 ms). A person's sheet may carry its own value
   (`p["balance_s"]`); without one the value is drawn from the band by a hash
   of the key, the pid and the name, the same every run and different for two
   people (the same draw `policy.sheet` makes for the reaction times). The
   delay is at least one tick: nothing is answered on the tick it is seen.

2. THE CHOSEN ACT WINS. A reflex is the body's answer when nobody chose
   otherwise. If the person has chosen an act that covers the reflex (a leap,
   a deliberate step off an edge), the reflex stays quiet. The act says so by
   `_chose(p, tag)`; a row lists the tags it yields to.

3. DRINK, FATIGUE AND SLEEP DEBT SLOW IT. Each is a number from 0 to 1 on the
   person (`drunk`, `fatigue`, `sleep_debt`; 0 when absent) and each adds its
   share to the delay: delay = base * (1 + 0.5 drunk + 0.3 fatigue + 0.4
   sleep_debt). The same factor shortens a step (a tired person reaches less
   far). Nothing sets these numbers yet; they are the seam for the body-clock
   and alcohol laws of later waves.

4. SLEEP RAISES THE THRESHOLD; ONLY THE DEAD ARE OFF (wave 13b, decision D2).
   A reflex row that names a stimulus scale (`threshold`) is never switched off
   by sleep or unconsciousness. The stimulus a sleeper needs is the awake
   threshold times (1 + `depth_gain` x depth), times the person's own dial, and
   whether a stimulus of that strength fires the reflex (or wakes the person)
   is a chance, drawn from a hash like every other per-person number here (see
   `fire_chance`). Deep unconsciousness has a high threshold but not an
   infinite one. A row with NO stimulus scale (balance: a reflex of the body
   against gravity, with nothing to be woken by) keeps the wave 12 rule: it is
   a thing the waking nervous system does, and it is out when the person is
   dead, unconscious or asleep. The thermoregulation rows (shiver, sweat,
   vessels) are `continuous` and `works_when` "alive": they run asleep and
   unconscious, and are off only when dead. None of this is a flag on a person.

5. EVERY FIRING IS AN EVENT. What the body did is written down as an ordinary
   typed event (`_reflex_fire`), with `why = "reflex:<kind>"` and the kind in
   its data, so the person's own mind perceives what its body did and other
   people see it. The verb is one of the closed VERBS; no reflex invents one.

6. NO MODEL CALLS. A reflex reads the state of the body and the lattice and
   writes the lattice. It never asks a chooser, a policy or a model.

7. A LOST CAPABILITY COMES FROM STRUCTURE. A broken shin bears no load, so the
   foot it carries is not support; a cut tendon gives no grip force. The reflex
   itself has no "disabled" switch: the law that asks what the body can do
   (support, grip) answers from the body, and the reflex simply finds nothing
   to do.

Balance is the first row. Its trigger and its three rungs (step, grab, fall)
live in motion.py (`_balance_reflex`); this file is only the contract."""
import math
import zlib

from .materials import SOUND, TICK_S
from .thermoreg import THERMO_ROWS

# One row a reflex.
#   band      the delay in seconds, drawn per person (sheet key `<kind>_s`)
#   reach_m   the longest step (balance) in metres, drawn per person
#             (sheet key `<kind>_step_m`)
#   yields_to the chosen-act tags that silence it
# INERT DATA FIELDS OF WAVE 13a (decisions D2; nothing reads them yet, the
# defaults are the old behaviour):
#   kind       "discrete" (fires once, after a delay) or "continuous"
#   works_when None = the old rule (`_reflex_off`); C6 and C7 give it a meaning
#   tau_s      lag of a continuous row, in seconds (0 for a discrete one)
#   threshold, depth, dial, chance  the sleep / unconscious law of D2 (None)
# "fall_grab" is a chosen-act tag a catch sets; the withdraw row of C7 lists
# it in its yields_to. No row names it in 13a.
# WAVE 13b (C6): the data of the D2 law is read now.
#   threshold  the awake stimulus strength that fires the reflex half the time
#              (in the units of the stimulus the row is about; None = no scale)
#   depth      "sleep": how deep the sleep or unconsciousness raises it; the
#              name of the per-person depth key is DEPTH_KEYS
#   dial       the band the person's own sensitivity is drawn from (key
#              `<kind>_dial`); a LOW dial is a light sleeper / a touchy reflex
#   chance     the width of the chance curve, as a share of the threshold
REFLEX_DEFAULTS = {"kind": "discrete", "works_when": None, "tau_s": 0.0,
                   "threshold": None, "depth": None, "dial": None,
                   "chance": None}
REFLEX = {
    "balance": {"band": (0.09, 0.12), "reach_m": (0.35, 0.60),
                "yields_to": ("leap", "step_off"), **REFLEX_DEFAULTS},
    # WAVE 13c (C7), the D2 law at work. Both rows name a stimulus scale, so
    # sleep and unconsciousness raise their threshold and never switch them
    # off.
    #   withdraw  the hand lets go of a hot thing. Stimulus: heat pain 0..1
    #             (0 at 44 C of the basal layer, 1 at 55 C); half the time at
    #             0.5 (spec T10: a 60 C mug is held, hot iron is dropped). The
    #             reflex arc is spinal: 30 to 60 ms (u). A chosen hold or a
    #             catch in progress silences it.
    #   startle   a loud noise wakes a sleeper (or startles one awake). Stimulus:
    #             the sound pressure at the ear relative to 1 (20 log10 of it
    #             is the level in dB). The awake threshold is set so that at the
    #             default depth of sleep (0.6) and the default dial a sleeper
    #             wakes half the time at `SOUND["wake_db"]` (60 dB, Basner et
    #             al. 2011 and the sleep studies behind it: reliable
    #             awakening at 55 to 65 dB). Narrow chance curve (0.1): a few
    #             dB either way.
    "withdraw": {"band": (0.03, 0.06), "yields_to": ("hold_hot", "fall_grab"),
                 **{**REFLEX_DEFAULTS, "threshold": 0.5, "depth": "sleep",
                    "dial": (0.8, 1.25)}},
    "startle": {"band": (0.04, 0.10), "yields_to": (),
                **{**REFLEX_DEFAULTS,
                   "threshold": 10.0 ** (SOUND["wake_db"] / 20.0) / (1.0 + 4.0 * 0.6),
                   "depth": "sleep", "dial": (0.5, 2.0), "chance": 0.1}},
}
# The thermoregulation rows (thermoreg.THERMO_ROWS) are reflex rows of kind
# "continuous": they take the same per-person draws and the same `_reflex_off`.
for _k, _v in THERMO_ROWS.items():
    REFLEX[_k] = {**REFLEX_DEFAULTS, **_v}

# THE SLEEP LAW (decision D2, the user's refinement). Depth runs from 0 (awake)
# to 1 (the deepest unconsciousness). A sleeper's depth is `p["sleep_depth"]`
# (0.6 if it is not written); an unconscious body's is `p["out_depth"]` (0.9).
# The threshold rises as  1 + depth_gain x depth  (depth_gain 4: a deeply
# unconscious body needs a stimulus 4.6 times as strong as a waking one, not
# an infinite one; u: the order of arousal thresholds in sleep, Busby et al.
# 1994, auditory arousal in deep sleep needs 20-30 dB more than in light sleep,
# which is about 4-8 times the pressure).
SLEEP_DEPTH = 0.6
OUT_DEPTH = 0.9
DEPTH_GAIN = 4.0
CHANCE_WIDTH = 0.35            # the chance curve goes from ~5 to ~95 percent over +-0.35 x threshold x 3
DIAL_BAND = (0.6, 1.6)         # every person's own sensitivity multiplier, drawn once (u)

# How much each slowing state adds to a delay at its worst (value 1).
SLOWS = {"drunk": 0.5, "fatigue": 0.3, "sleep_debt": 0.4}


def _draw(p, key, lo, hi):
    """A value in [lo, hi] that is this person's own: the sheet's if it has
    one, else a hash of the key, the pid and the name (no random numbers)."""
    v = p.get(key)
    if v is not None:
        return float(v)
    u = (zlib.crc32(f"{key}|{p.get('pid')}|{p.get('name')}".encode())
         & 0xFFFFFFFF) / 2.0 ** 32
    return float(lo + (hi - lo) * u)


def fatigue_of(p, limb=None):
    """The fatigue that slows `limb` (0 to 1): the limb's own number in
    p["limb_fatigue"] (a dict the effort laws write, wave 13b) if the person
    has one for it, else the whole person's p["fatigue"]."""
    if limb is not None:
        v = (p.get("limb_fatigue") or {}).get(limb)
        if v is not None:
            return v
    return p.get("fatigue", 0.0)


def depth_of(p):
    """How deeply this person is out: None for the dead, 0.0 for a waking
    body, else the depth of its sleep or unconsciousness (0 to 1)."""
    if not p["alive"]:
        return None
    if p["awake"] and not p.get("asleep"):
        return 0.0
    if not p["awake"]:
        return float(p.get("out_depth", OUT_DEPTH))
    return float(p.get("sleep_depth", SLEEP_DEPTH))


def threshold_now(p, kind):
    """The stimulus strength that fires reflex `kind` for this person NOW: the
    awake threshold of the row, raised by depth, times the person's own dial.
    None for the dead and for a row with no stimulus scale."""
    row = REFLEX[kind]
    thr = row.get("threshold")
    d = depth_of(p)
    if thr is None or d is None:
        return None
    lo, hi = row.get("dial") or DIAL_BAND
    dial = _draw(p, f"{kind}_dial", lo, hi)
    return float(thr) * (1.0 + DEPTH_GAIN * d) * dial


def fire_chance(p, kind, strength):
    """The chance, 0 to 1, that a stimulus of `strength` fires reflex `kind`
    (or wakes the person) in the second it lasts: a logistic curve in the
    stimulus, half at the threshold, width CHANCE_WIDTH x threshold. 0 for the
    dead and for a row with no scale; the same function awake and asleep, only
    the threshold differs."""
    thr = threshold_now(p, kind)
    if thr is None or strength <= 0.0:
        return 0.0
    w = (REFLEX[kind].get("chance") or CHANCE_WIDTH) * thr
    z = (strength - thr) / max(w, 1e-12)
    if z < -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def _unit(p, kind, tick):
    """A number in [0, 1) that is the same every run: a hash of the kind, the
    person and the tick (no random-number stream is read or advanced)."""
    return (zlib.crc32(f"fire|{kind}|{p.get('pid')}|{p.get('name')}|{tick}".encode())
            & 0xFFFFFFFF) / 2.0 ** 32


def slowed(p, limb=None):
    """The factor drink, fatigue and sleep debt put on a delay: 1.0 for a
    rested sober body."""
    return 1.0 + sum(
        k * min(max(float((fatigue_of(p, limb) if n == "fatigue"
                           else p.get(n, 0.0)) or 0.0), 0.0), 1.0)
        for n, k in SLOWS.items())


class ReflexMixin:

    def reflex_delay_s(self, p, kind):
        """This person's delay for reflex `kind`, in seconds, as it is now
        (slowed by drink, fatigue and sleep debt)."""
        lo, hi = REFLEX[kind]["band"]
        return _draw(p, f"{kind}_s", lo, hi) * slowed(p)

    def reflex_reach_m(self, p, kind):
        """The longest step this person's reflex takes, in metres: their own
        draw, shortened by the same slowing that lengthens the delay."""
        lo, hi = REFLEX[kind]["reach_m"]
        return _draw(p, f"{kind}_step_m", lo, hi) / slowed(p)

    def _chose(self, p, tag, ticks=1):
        """The person has CHOSEN an act that covers a reflex (the tag names
        it) for this many ticks, counted from now."""
        p.setdefault("_chose", {})[tag] = self.tick + int(ticks)

    def reflex_fires(self, p, kind, strength):
        """Does a stimulus of `strength` fire reflex `kind` THIS tick? The
        per-second chance (`fire_chance`) is spread over the ticks of the
        second so that a stimulus that lasts a second fires with that chance;
        the draw is a hash of the tick, so it is the same every run."""
        c = fire_chance(p, kind, strength)
        if c <= 0.0:
            return False
        if c >= 1.0:
            return True
        per_tick = 1.0 - (1.0 - c) ** (TICK_S / 1.0)
        return _unit(p, kind, self.tick) < per_tick

    def noise_wakes(self, p, level_db):
        """Does a sound of `level_db` at this sleeper's ear wake it, this
        tick? The D2 law: the startle threshold is raised by the depth of the
        sleep and scaled by the person's own dial, and the answer is a
        chance (`reflex_fires`). The old rule was a fixed line at 60 dB."""
        return self.reflex_fires(p, "startle", 10.0 ** (level_db / 20.0))

    def _reflex_off(self, p, kind):
        """Why reflex `kind` cannot fire for this person now: "out" (dead; or,
        for a row with no stimulus scale and no `works_when`, unconscious or
        asleep too), "chosen" (they chose something that covers it), or None
        when it may run. A row WITH a scale is never out by sleep: sleep raises
        its threshold (`threshold_now`), and `reflex_fires` asks the chance."""
        row = REFLEX[kind]
        if not p["alive"]:
            return "out"
        if row.get("works_when") != "alive" and row.get("threshold") is None \
                and (not p["awake"] or p.get("asleep")):
            return "out"
        chose = p.get("_chose")
        if chose:
            for tag in REFLEX[kind]["yields_to"]:
                if chose.get(tag, -1) >= self.tick:
                    return "chosen"
        return None

    def _reflex_gate(self, p, kind):
        """One look at reflex `kind` for a person whose body has just asked
        for it: "off" (it may not run: see `_reflex_off`), "wait" (the delay
        has not passed) or "go" (fire it now, then call `_reflex_done`).
        The first call that finds the trigger starts the delay; the trigger
        going away is `_reflex_done`'s business, called by the law."""
        if self._reflex_off(p, kind):
            self._reflex_done(p, kind)
            return "off"
        pend = p.setdefault("_reflex", {})
        if kind not in pend:
            n = max(1, int(round(self.reflex_delay_s(p, kind) / TICK_S)))
            pend[kind] = self.tick + n
        return "go" if self.tick >= pend[kind] else "wait"

    def _reflex_done(self, p, kind):
        """The trigger is over (answered, or gone by itself): the next one
        starts a fresh delay."""
        pend = p.get("_reflex")
        if pend and kind in pend:
            del pend[kind]
            if not pend:
                del p["_reflex"]

    def _reflex_fire(self, p, kind, verb, text, result=None, target=None,
                     place=None, **data):
        """Write down that reflex `kind` fired, as an event the mind
        perceives (`verb` is one of the closed VERBS), and return it."""
        extra = tuple(sorted((k, v) for k, v in data.items())
                      ) + (("reflex", kind),)
        return self.emit(verb, actor=p["pid"], target=target, result=result,
                         why=f"reflex:{kind}", place=place, data=extra,
                         text=text, to=p, site=f"reflex:{kind}")
