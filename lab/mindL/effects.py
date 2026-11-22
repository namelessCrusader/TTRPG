"""E2. Act meanings learned from their EFFECTS (SV3, step 2 of SV5, 2026-11-21).

The KIND and HARM verb lists said what an act MEANS. This module learns it, per partner and per act that
partner can do to me, from the MEASURED change of my own drive needs after that act. It replaces the labels
in the partner term of the choice (mind option "effects", off by default; the labelled version stays as the
control arm). It is the partner model of learned-mind-target 9.3 with two outputs:

    predict(partner, my_act, state) -> {their act: chance they do it in answer, within WINDOW turns}
    effect(partner, their_act, state) -> expected change of my drive needs (a vector in DRIVES order)

The worth of an act to me is read OUTSIDE the model, from the state: sum over answers of
chance * (my weighted drive error now - my weighted drive error after the effect). So food is worth a lot to
the hungry and nothing to the fed, and coin to the poor, with no rule saying so. (State dependence comes in two
places: the weighted error has a set point, and a need cannot go below 0 or above 1: effect() clips.)

THE INTERFACE (ActMeanings). The mind talks to a model only through
    predict(partner, my_act, state), effect(partner, their_act, state), update(obs), set_prior(...)
and update takes one of four observations: Did (I acted on a partner), Got (a partner acted on me, with
the needs at the start of the step it landed in), Tick (a choice point: time passes, due windows close),
Flush (the night ends: windows still open are closed for answers and dropped for effects). Any learning rule
that fits this can replace DeltaMeanings; see "WHERE A LOCAL RULE PLUGS IN".

THE DEFAULT LEARNING RULE (DeltaMeanings) is the simplest delta rule.
  EFFECTS. Credit window CREDIT = 12 turns (the longest act in DUR is 12, a sleep; a meal is 4 turns, a tending 6; a
  gift given to be eaten pays in about the next choice or two; chosen before any run, one global value, not tuned).
  When a partner's act at me is perceived it opens a credit entry. The entry's measured effect is
      need(now) - need(start of the step the act landed in) - drift * (turns between),
  taken once CREDIT turns have passed since the act, where drift is the mean change of the needs per turn when
  NOTHING was done to me (learned by its own delta rule from stretches with no open entry): so the placebo, an act
  that does nothing, measures 0, and hunger's rise is not charged to the person who spoke to me. Entries that overlap
  share the credit as in Rescorla-Wagner: each entry is corrected by what is left after the OTHER entries in its
  window are taken off, with its own step 1/(n+1), never below FLOOR (so it is a sample mean at first, then a
  slow moving average that follows change). Two layers: per (partner, act), and pooled over partners per act
  (the mind's own shared prior across partners, 9.3): a new (partner, act) starts at the pooled value.
  ANSWERS. An answer is credited to the LATEST act of mine at that partner still inside its window (not to every open
  window, which would make a wave look as likely to be served as an ask when both are done often). One logistic regression per (partner, my act, their act) on the option's numbers x, trained at the
  close of each WINDOW (30 turns) on "did they do it", blended with the pooled-over-partners one by the number of
  windows seen (K_POOL pseudo-windows for the pool). An answer never seen from a partner is not listed: a deaf
  partner has nothing, and nothing is learned about it.
  A CULTURAL PRIOR (the hook for the knowledge layer, K1-K12): set_prior(act, delta, n0, p) hands in an
  initial effect estimate for an act (and, if p is given, an initial chance of that act as an answer) with a
  pseudo-count n0, so the first n0 observations are outweighed by it, and experience overrides it:
  the weight of the prior after n observations is n0/(n0+n).

WHERE A LOCAL RULE PLUGS IN. (1) The effect estimate is one row per (partner, act) updated only by that row's
own residual and a step size: the form of a three-factor rule, row += step * eligibility * (error signal), where the
credit entry IS the eligibility trace (a fixed-length window, 1 while open) and the drive change minus the others'
shares is the third factor. An eligibility trace that decays (e^(-age/tau)) with the update applied every step
would replace the window; the entry list and Tick are the only places that know about it. (2) The answer model is
a delta-rule (logistic) layer per (partner, my act): a delta-rule memory (the nesformer idea) or a Hebbian
outer product of (my act, option numbers) with (their act) would replace _learn_did and _p. (3) predictive coding
would replace the two layers by one net whose prediction error trains both. None of that is built.

NEEDS MORE WORK (noted, not done): (a) my own acts during a window confound the measured effect (eating after
a gift); they average out only if they do not depend on the partner's act. A residual against the world model's
prediction of my own act would remove them. (b) A gift of food shows its effect only after I eat it; a window
of 12 turns catches the next meal, not a meal tomorrow. (c) the drift is one vector for all states. (d) effects
are means; a partner who sometimes helps and sometimes harms with the same act is one number per drive. (e) the
drive vector holds no injury other than safety (hp enters it) and no concern for another person (step 3).
"""
from __future__ import annotations

import math
from collections import namedtuple

import numpy as np

CREDIT = 12          # turns: the credit window of an effect
WINDOW = 30          # turns: the window of an answer (as partners.WINDOW)
FLOOR = 0.02         # the smallest step of an effect estimate (recency)
DRIFT_FLOOR = 0.02
K_POOL = 3           # pseudo-windows the pooled answer model counts for
K_INIT = 2           # a new (partner, act) starts at the pooled value with this many pseudo-observations
P0 = 0.05            # the chance given to an answer never yet seen
VMAX = 96            # room for the verbs

Did = namedtuple("Did", "partner act x t")             # I did `act` to `partner`; x = the option's numbers
Got = namedtuple("Got", "partner act t tref need0")    # `partner` did `act` to me at t; need0 = my needs at tref
Tick = namedtuple("Tick", "t need")                    # a choice point
Flush = namedtuple("Flush", "t need")                  # the night's end
State = namedtuple("State", "x need")                  # x: option numbers (for predict); need: my needs (for effect)


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def error_drop(kind, need, delta):
    """The worth of a predicted change `delta` of my needs: the drop of my weighted drive error."""
    from lab.mindL import drives as H
    return float(H.errors(kind, need).sum() - H.errors(kind, np.clip(need + delta, 0.0, 1.0)).sum())


class ActMeanings:
    """The interface. predict -> {their act: chance}; effect -> change of my drive needs; update(obs) -> [(partner, error)]."""

    def predict(self, partner, my_act, state):
        raise NotImplementedError

    def effect(self, partner, their_act, state):
        raise NotImplementedError

    def update(self, obs):
        raise NotImplementedError

    def set_prior(self, act, delta, n0=3, p=None, partner=None):
        raise NotImplementedError


class _Cr:
    __slots__ = ("partner", "act", "t", "tref", "need0")

    def __init__(self, partner, act, t, tref, need0):
        self.partner, self.act, self.t, self.tref, self.need0 = partner, act, t, tref, need0


class DeltaMeanings(ActMeanings):
    def __init__(self, ndrives, nx=7, lr_pred=0.3, credit=CREDIT, window=WINDOW, floor=FLOOR, keep_log=False, turn_drift=False):
        self.nd, self.nx, self.lr_pred, self.credit, self.window, self.floor = ndrives, nx, lr_pred, credit, window, floor
        self.turn_drift = turn_drift             # step 3 fix (option fx_fix): the drift is a per-TURN rate, not a mean over ticks
        self.sd, self.st = np.zeros(ndrives), 0.0
        self.vocab: dict = {}
        self.verbs: list = []
        self.bias0 = np.full(VMAX, math.log(P0 / (1 - P0)))
        self.seen: dict = {}                 # partner -> bool mask over the vocabulary (answers ever seen)
        self.gseen = np.zeros(VMAX, bool)
        self.cw: dict = {}                   # (partner or "*", my act) -> [W (VMAX x nx), windows seen]
        self.eff: dict = {}                  # (partner, their act) -> [vector, n]
        self.pool: dict = {}                 # their act -> [vector, n]
        self.drift, self.n_drift = np.zeros(ndrives), 0
        self.open: list = []                 # credit entries
        self.recent: list = []               # entries closed lately (their share is taken off overlapping windows)
        self.dids: list = []                 # [partner, act, x, t0, set of answer indices]
        self.last = None                     # (t, need) of the last tick
        self.keep_log = keep_log
        self.credit_log: list = []           # (partner, act, measured change, [(partner, act) of the overlapping entries])

    # ------------------------------------------------------------ vocabulary
    def _ix(self, verb):
        j = self.vocab.get(verb)
        if j is None:
            assert len(self.verbs) < VMAX, "raise VMAX"
            j = self.vocab[verb] = len(self.verbs)
            self.verbs.append(verb)
        return j

    def _cw(self, key):
        c = self.cw.get(key)
        if c is None:
            W = np.zeros((VMAX, self.nx))
            W[:, 0] = self.bias0
            c = self.cw[key] = [W, 0]
        return c

    # ------------------------------------------------------------ the prior hook
    def set_prior(self, act, delta, n0=3, p=None, partner=None):
        """A cultural belief: `act` (done to me by anyone, or by `partner`) changes my needs by `delta` (a vector in DRIVES
        order), held with the weight of n0 observations; p: the chance that someone does this act in answer to mine."""
        vec = np.asarray(delta, float).copy()
        if partner is None:
            self.pool[act] = [vec, n0]
        else:
            self.eff[(partner, act)] = [vec, n0]
        if p is not None:
            j = self._ix(act)
            self.bias0[j] = math.log(p / (1 - p))
            for W, n in self.cw.values():
                if n == 0:
                    W[j, 0] = self.bias0[j]
            self.gseen[j] = True                 # the belief puts the act on every partner's list
            self.prior_ans = getattr(self, "prior_ans", set()) | {j}

    def _seen(self, partner):
        s = self.seen.get(partner)
        extra = getattr(self, "prior_ans", None)
        if extra:
            s = np.zeros(VMAX, bool) if s is None else s.copy()
            s[list(extra)] = True
        return s

    # ------------------------------------------------------------ reading
    def _p(self, key, idx, x):
        c = self.cw.get(key)
        if c is None:
            return sig(self.bias0[idx]), 0
        return sig(c[0][idx] @ x), c[1]

    def predict(self, partner, my_act, state):
        seen = self._seen(partner)
        if seen is None:
            return {}
        idx = np.flatnonzero(seen)
        if not idx.size:
            return {}
        x = state.x
        pp, n = self._p((partner, my_act), idx, x)
        ps, _ = self._p(("*", my_act), idx, x)
        p = (n * pp + K_POOL * ps) / (n + K_POOL)
        return {self.verbs[j]: float(q) for j, q in zip(idx, p)}

    def effect(self, partner, their_act, state):
        e = self.eff.get((partner, their_act)) or self.pool.get(their_act)
        need = np.asarray(state.need, float)
        if e is None:
            return np.zeros(self.nd)
        return np.clip(need + e[0], 0.0, 1.0) - need

    def raw_effect(self, partner, their_act):
        """The stored estimate, before the state clips it (None if never seen)."""
        e = self.eff.get((partner, their_act))
        return None if e is None else e[0].copy()

    # ------------------------------------------------------------ learning
    def update(self, obs):
        if isinstance(obs, Did):
            self.dids.append([obs.partner, obs.act, obs.x, obs.t, set()])
            return []
        if isinstance(obs, Got):
            j = self._ix(obs.act)
            self.seen.setdefault(obs.partner, np.zeros(VMAX, bool))[j] = True
            self.gseen[j] = True
            mine = [d for d in self.dids if d[0] == obs.partner and obs.t >= d[3]]
            if mine:                                      # the answer goes to the latest act of mine at that partner
                max(mine, key=lambda d: d[3])[4].add(j)
            self.open.append(_Cr(obs.partner, obs.act, obs.t, obs.tref, np.asarray(obs.need0, float)))
            return []
        if isinstance(obs, Tick):
            return self._tick(obs.t, np.asarray(obs.need, float), False)
        if isinstance(obs, Flush):
            return self._tick(obs.t, np.asarray(obs.need, float), True)
        raise TypeError(obs)

    def _est(self, e, layer):
        """The current estimate of one entry (without creating it)."""
        if layer == "p":
            r = self.eff.get((e.partner, e.act)) or self.pool.get(e.act)
        else:
            r = self.pool.get(e.act)
        return r[0] if r is not None else 0.0

    def _tick(self, t, need, flush):
        out = []
        if self.last is not None and t < self.last[0]:      # a new night began without a flush
            self.open, self.recent, self.last = [], [], None
        if not flush and self.last is not None and not self.open and t > self.last[0]:
            dt = t - self.last[0]
            self.n_drift += 1
            if self.turn_drift:
                # a ratio of decayed sums (change over turns), so a 12-turn sleep weighs 12 times a 1-turn ask. The old rule below
                # averaged the per-turn rate over TICKS: the many short ticks (asks, looks) outweighed the few long ones (meals,
                # sleep), and the mean rate of rising need came out 1.3 to 2.7 times the true one (step 3 diagnosis).
                self.sd = self.sd * (1.0 - DRIFT_FLOOR) + (need - self.last[1])
                self.st = self.st * (1.0 - DRIFT_FLOOR) + dt
                self.drift = self.sd / self.st
            else:
                lr = max(1.0 / self.n_drift, DRIFT_FLOOR)
                self.drift += lr * ((need - self.last[1]) / dt - self.drift)
        if flush:
            self.open, self.recent, self.last = [], [], None    # incomplete windows measure nothing
        else:
            self.last = (t, need)
            due = sorted((e for e in self.open if t - e.t >= self.credit), key=lambda e: e.t)
            for e in due:
                others = [f for f in self.open + self.recent if f is not e and e.tref <= f.t <= t]
                base = need - e.need0 - self.drift * (t - e.tref)
                for layer in ("p", "pool"):
                    pred = self._est(e, layer) + sum((self._est(f, layer) for f in others), np.zeros(self.nd))
                    resid = base - pred
                    if layer == "p":
                        r = self.eff.get((e.partner, e.act))
                        if r is None:
                            q = self.pool.get(e.act)
                            r = self.eff[(e.partner, e.act)] = [q[0].copy() if q else np.zeros(self.nd),
                                                                min(q[1], K_INIT) if q else 0]
                    else:
                        r = self.pool.setdefault(e.act, [np.zeros(self.nd), 0])
                    r[0] = r[0] + max(1.0 / (r[1] + 1), self.floor) * resid
                    r[1] += 1
                if self.keep_log:
                    self.credit_log.append((e.partner, e.act, base.copy(), [(f.partner, f.act) for f in others]))
            if due:
                gone = {id(e) for e in due}
                self.open = [e for e in self.open if id(e) not in gone]
                self.recent = [f for f in self.recent + due if f.t >= t - 60]
        done = [d for d in self.dids if flush or t - d[3] >= self.window]
        if done:
            self.dids = [d for d in self.dids if not (flush or t - d[3] >= self.window)]
            out = [r for r in (self._learn_did(d) for d in done) if r is not None]
        return out

    def _learn_did(self, d):
        partner, act, x, _, ans = d
        seen = self.seen.get(partner)
        if seen is None or not seen.any():
            return None                                    # a partner that never answered: nothing to learn
        err = None
        for key, mask in (((partner, act), seen), (("*", act), self.gseen)):
            idx = np.flatnonzero(mask)
            c = self._cw(key)
            y = np.array([1.0 if j in ans else 0.0 for j in idx])
            p = sig(c[0][idx] @ x)
            c[0][idx] += self.lr_pred * np.outer(y - p, x)
            c[1] += 1
            if key[0] != "*":
                err = float(np.abs(y - p).mean())
        return partner, err
