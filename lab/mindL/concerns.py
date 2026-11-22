"""E3. Acquired concerns (SV1 and SV2, step 3 of SV5, 2026-11-21; learned-mind-target 9.2).

A mind is born with NO concerns. A concern is a target (a person, or a thing the mind perceives; never an act) with a learned weight
w >= 0. A concern adds to the reward, on top of the drop in drive error:

    reward  +=  sum over targets T of  w_T * ( need_T before  -  need_T after )

where need_T is how far the target is from a good state AS THE MIND PERCEIVES IT. For a person that is read through SIGNS (SV2: the
world's perceived_need, hurt, fear and hunger, each 0..1, shown in proportion to the target's expressiveness and read with noise that
falls with the observer's acuity). So a stoic is under-read and gets less care than an expressive person with the same true need.
For a thing, the need is 1 if I do not hold it, 0 if I do. Reward is still only a drop in error. The body is never switched off: a
concern wins a choice only when w times the predicted relief beats the body terms in the same lookahead.

THE INTERFACE (Concerns). The mind talks to the module only through
    learn(step)          at a step's close; step is a StepObs; returns the concern reward of that step (read BEFORE this step's learning)
    pull(T)              the weight now (0 for a target never met)
    expected(verb, T, need)   the predicted concern gain of my act `verb` aimed at T when T's need reads `need` (the choice term)
    relief(verb, need)  how far the act is predicted to relieve a need that reads `need`; set_prior(verb, rho, eta, n0): a cultural belief
    night()              the night's end: disuse
    table() / set_table(t)   the weights, for inspection and for the swap control
    observe_read / did   what the mind sees (readings of other people's needs) and does (acts aimed at a person): these feed the
                         model of what my acts do to their needs (OtherEffects, below)
Any learning rule that fits this can replace RWConcerns (a three-factor local rule: the eligibility is the contact, the third factor
the drive-error drop; see "WHERE A LOCAL RULE PLUGS IN").

THE DEFAULT RULE (RWConcerns). Per target T it keeps
    a_T  a Rescorla-Wagner association between the target's presence and the homeostatic reward of the step MINUS ITS RUNNING AVERAGE
         (the existing valence rule V: "drive-error drops that followed contact"). Taking off the average means a target that is always
         there, or there at random, predicts nothing; cues present share the prediction, so a placebo is held near 0.
         CONDITIONED REINFORCEMENT.
    e_T  the effort justification: when I acted for T (my act aimed at T, and it relieved T's need, measured in the step or predicted by
         the learned act effects) and the step COST me (drive error rose), e_T += cost * (error rise).   COST PAID ON ITS BEHALF.
    h_T  the habituation: each turn of contact multiplies it by (1 - hab); it recovers toward 1 with time constant 1/rec. It scales
         what a contact teaches (the step of a_T and the effort credit), so repeated contact pays less.
    w_T = min(ceil, gain * max(a_T, 0) + e_T).
It falls by DISUSE (a_T and e_T fade at night, `fade` a night), DISILLUSION (a_T goes negative when error rises after contact, and the
effort credit then shrinks by `dis` a step) and HABITUATION (above). `ceil` and `gain` are capacities (the genome); the content is not.
The numbers (lr, gain, ceil, cost, fade, dis, hab, rec, JND) are set before any run, not tuned, and none is sourced yet: a request for
sources (attachment formation and decay, habituation of social reward) goes to the library.

OtherEffects: what my acts do to the needs of the people I act at. For each act verb: rho, the fraction of the target's need that it
relieves (a meal for the hungry), and eta, the change it makes in the need as a plain number (a blow raises it); the choice uses the
positive part of each (the help and the harm). Learned from the target's perceived need at my act and when the window closes (at my next
act aimed at the same person, or CREDIT turns on), minus the natural change of that need over the same time (a per-turn rate learned
from the stretches in which I aimed nothing at the person). The change is credited to the LATEST act only (as in step 2, where sharing
it among overlapping acts made a placebo look as good as the real act). The same delta rule as effects.DeltaMeanings, on the target's
needs. A cultural prior can be set with set_prior(verb, rho, eta, n0).

WHERE A LOCAL RULE PLUGS IN. (1) a_T is a delta rule on the contact cue: a three-factor rule with the contact as the eligibility trace
and the drive-error drop minus the prediction as the third factor; the same one a Hebbian rule with a context baseline would give.
(2) h_T is a per-synapse habituation trace (a fast-decaying gain). (3) OtherEffects is one delta-rule row per verb, like the step 2 rows.
None of that is built.

NEEDS MORE WORK (noted, not done): things are targets of formation and reward only (no choice term for acts at a thing: the menu does
not name the thing); the effect of my act on a target who is NOT aimed at is not learned; hearsay and absent targets are not read (a
concern for someone in another room has no pull); grief is the fade of w after the death, with the need left as last read; no
contest/status targets (9.3).
"""
from __future__ import annotations

import math
from collections import namedtuple

JND = 0.15       # a sign reading below this is not a need (about 1.25 SD of the noise at acuity 1)
CREDIT = 12      # turns: the window of an act's effect on another's need (as effects.CREDIT)
NMIN = 0.2       # a window counts for the relief fraction only if the need was at least this
FLOOR = 0.02

# a step of the mind, as the concerns see it. r: the HOMEOSTATIC reward of the step (drop of drive error); before/after: {target: need}
# at its start and close (only targets perceived then); contact: targets perceived at the start; act: the verb; aimed: the target of the
# act (a person) or None; helped: how far the act is predicted to relieve the aimed target (0 if not known); dt: turns.
StepObs = namedtuple("StepObs", "r before after contact act aimed helped dt")


def need_of(sign):
    """One number from the three sign channels: the sum of what shows above the just-noticeable difference."""
    return sum(max(0.0, v - JND) for v in sign.values())


class Concerns:
    """The interface."""

    def pull(self, target):
        raise NotImplementedError

    def expected(self, verb, target, need):
        raise NotImplementedError

    def learn(self, step):
        raise NotImplementedError

    def night(self):
        raise NotImplementedError

    def table(self):
        raise NotImplementedError

    def set_table(self, table):
        raise NotImplementedError

    def relief(self, verb, need):
        raise NotImplementedError

    def set_prior(self, verb, rho=0.0, eta=0.0, n0=3):
        raise NotImplementedError

    def observe_read(self, t, readings):
        raise NotImplementedError

    def did(self, target, verb, t, need):
        raise NotImplementedError

    def flush(self):
        raise NotImplementedError


class _Row:
    __slots__ = ("a", "e", "h")

    def __init__(self):
        self.a, self.e, self.h = 0.0, 0.0, 1.0


class OtherEffects:
    """What my acts do to the perceived needs of the people I act at (see the module docstring).

    An act aimed at T opens a window. It closes at my next act aimed at T, or CREDIT turns on, whichever is first, and the reading of T
    then is compared with the one at the act. The window's change is credited to the LATEST act only (as step 2 credits an answer to my
    latest act: sharing it among overlapping acts made a placebo look as good as the real one), after the natural change of the need,
    a per-turn rate learned from the stretches in which I aimed nothing at the person (a ratio of decayed sums, so a long stretch
    weighs more than a short one)."""

    def __init__(self, credit=CREDIT, floor=FLOOR, n0=5):
        self.credit, self.floor, self.n0 = credit, floor, n0
        self.rho: dict = {}              # verb -> [value, n]: the fraction of the need relieved beyond the natural change
        self.eta: dict = {}              # verb -> [value, n]: the rise caused beyond the natural change
        self.open: dict = {}             # target -> [t, verb, need_before]: the window of my latest act at T
        self.last: dict = {}             # target -> (t, need) of its last reading
        self.s_rel = self.s_tr = self.s_h = self.s_th = 0.0
        self.log: list = []

    def set_prior(self, verb, rho=0.0, eta=0.0, n0=3):
        self.rho[verb] = [float(rho), n0]
        self.eta[verb] = [float(eta), n0]

    def predict(self, verb):
        """(the fraction of the need my act relieves, the rise it causes): the parts that help and the parts that hurt."""
        r, e = self.rho.get(verb), self.eta.get(verb)
        return max(0.0, r[0]) if r else 0.0, max(0.0, e[0]) if e else 0.0

    def ctx(self):
        """The natural change per turn of a need: (as a fraction relieved, as a signed change)."""
        return (self.s_rel / self.s_tr if self.s_tr > 1e-9 else 0.0), (self.s_h / self.s_th if self.s_th > 1e-9 else 0.0)

    def did(self, target, verb, t, need):
        self._close(target, t)                                       # my latest act at T closes the window of the one before
        if need is not None:
            self.open[target] = [t, verb, need]

    def flush(self):
        self.open, self.last = {}, {}

    def _close(self, T, t):
        e = self.open.pop(T, None)
        prev = self.last.get(T)
        if e is None or prev is None or prev[0] != t or prev[0] <= e[0]:
            return                                                   # no reading of T now: nothing learned
        t0, verb, before = e
        now, L = prev[1], t - t0
        c_rel, c_h = self.ctx()
        if before >= NMIN:
            self._move(self.rho, verb, (before - now) / before - c_rel * L)
        self._move(self.eta, verb, (now - before) - c_h * L)
        self.log.append((verb, T, before, now))

    def observe_read(self, t, readings):
        for T, need in readings.items():
            if T.startswith("thing:"):
                continue
            prev = self.last.get(T)
            if prev is not None and t > prev[0] and T not in self.open:
                dt = t - prev[0]                                     # nothing aimed at T in between: the need's natural change
                for k in ("s_rel", "s_tr", "s_h", "s_th"):
                    setattr(self, k, getattr(self, k) * 0.999)
                if prev[1] >= NMIN:
                    self.s_rel += (prev[1] - need) / prev[1] * dt
                    self.s_tr += dt * dt
                self.s_h += need - prev[1]
                self.s_th += dt
            self.last[T] = (t, need)
        for T in [T for T, e in self.open.items() if t - e[0] >= self.credit]:
            self._close(T, t)

    def _move(self, table, verb, obs):
        """The delta rule on one row: a step toward what the window showed (n0 pseudo-observations of 'no effect' to start)."""
        r = table.setdefault(verb, [0.0, self.n0])
        r[0] = max(-1.0, min(1.0, r[0] + max(1.0 / (r[1] + 1), self.floor) * (obs - r[0])))
        r[1] += 1


class RWConcerns(Concerns):
    def __init__(self, gain=20.0, ceil=1.0, cost=1.0, lr=0.2, lr_base=0.05, fade=0.02, dis=0.05, hab=0.005, rec=1.0 / 300, helped_min=JND,
                 effects=None):
        self.gain, self.ceil, self.cost, self.lr, self.lr_base, self.fade = gain, ceil, cost, lr, lr_base, fade
        self.dis, self.hab, self.rec, self.helped_min = dis, hab, rec, helped_min
        self.rows: dict = {}                 # target -> _Row; empty at birth
        self.base = 0.0                      # the average homeostatic reward of a step (the context)
        self.fx = effects if effects is not None else OtherEffects()
        self.last_cue_gain: dict = {}

    # ------------------------------------------------------------ reading
    def _w(self, r):
        return min(self.ceil, self.gain * max(r.a, 0.0) + r.e)

    def pull(self, target):
        r = self.rows.get(target)
        return self._w(r) if r is not None else 0.0

    def table(self):
        return {T: self._w(r) for T, r in self.rows.items()}

    def set_table(self, table):
        """The swap control: replace the weights by another mind's (kept as effort credit; the rest of the state is cleared)."""
        self.rows = {}
        for T, w in table.items():
            r = self.rows[T] = _Row()
            r.e = float(w)

    def expected(self, verb, target, need):
        w = self.pull(target)
        if w <= 0.0:
            return 0.0
        rho, eta = self.fx.predict(verb)
        return w * (rho * need - eta)

    def relief(self, verb, need):
        """How far my act `verb` is predicted to relieve a target whose need reads `need` (0 if unknown)."""
        return self.fx.predict(verb)[0] * need

    def set_prior(self, verb, rho=0.0, eta=0.0, n0=3):
        """A cultural belief about what an act does to another's need (the hook for the knowledge layer)."""
        self.fx.set_prior(verb, rho, eta, n0)

    def observe_read(self, t, readings):
        self.fx.observe_read(t, readings)

    def did(self, target, verb, t, need):
        self.fx.did(target, verb, t, need)

    def flush(self):
        self.fx.flush()

    # ------------------------------------------------------------ learning
    def learn(self, st):
        r_c = 0.0
        for T, b in st.before.items():
            a = st.after.get(T)
            if a is not None:
                r_c += self.pull(T) * (b - a)
        cues = set(st.contact)
        if st.aimed is not None:
            cues.add(st.aimed)
        for T in cues:
            self.rows.setdefault(T, _Row())
        pred = sum(self.rows[T].a for T in cues)
        d = (st.r - self.base) - pred
        self.base += self.lr_base * (st.r - self.base)
        rate = self.lr / max(1, len(cues))              # the cues share one prediction: its total step is lr, however many there are
        for T in cues:
            row = self.rows[T]
            row.a += rate * row.h * d
            if row.a < 0 and row.e > 0:
                row.e *= 1.0 - self.dis                      # disillusion: error rises after contact
        if st.aimed is not None and st.r < 0:                # effort justification: it cost me, and it was for them
            b, a = st.before.get(st.aimed), st.after.get(st.aimed)
            seen = (b - a) if (b is not None and a is not None) else 0.0
            if max(seen, st.helped) >= self.helped_min:
                row = self.rows[st.aimed]
                row.e = min(self.ceil, row.e + self.cost * row.h * (-st.r))
        for T, row in self.rows.items():
            if T in cues:
                row.h *= (1.0 - self.hab) ** st.dt
            else:
                row.h += (1.0 - row.h) * (1.0 - math.exp(-st.dt * self.rec))
        return r_c

    def night(self):
        for T in list(self.rows):
            r = self.rows[T]
            r.a *= 1.0 - self.fade
            r.e *= 1.0 - self.fade
            if abs(r.a) < 1e-5 and r.e < 1e-5:
                del self.rows[T]
