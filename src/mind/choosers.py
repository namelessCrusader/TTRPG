"""The choosers: what picks one option of a menu (docs/sim-mind-design.md 3.5,
step M5).

A chooser gets a DecisionRequest and gives back a Decision (api.Chooser).
`submit` sends the request, and `poll` gives the answer. Every chooser in
this file is synchronous: the answer is ready before `submit` returns. A
chooser reads only the request: the texts, and the rule scores of each
option. A model reads the texts only: Combined removes the scores from the
request before it asks the model.

    Rules      draws from softmax((prior + habit) / T_RULES). This is the
               lab's Rules policy (lab/world2/policy.py:144-150).
    Random     every option the same. The floor.
    Gate       the rules pick when their top probability is at least g.
               The model picks the others.
    Combined   the model's view of the options (z), plus the traits' part of
               the rule scores, behind the plausibility guard. Only z comes
               from the model. With the guard off, this is the lab's bias
               mode (lab/world2/policy.py:233-243).
    Replay     the picks of an earlier run, by request id. This is a stub:
               step M6 reads the picks from a decisions file.
    Player     a person. It picks from the full menu (the loop never cuts a
               player's menu), and it can chain picks (Job.then).

THE DRAW. The sim has no random number generator (design 2.4). A chooser
that draws makes a new generator for each request, seeded from (world seed,
who, request id) with `seed_of`. So a draw does not depend on the requests
before it, or on the process: a run with these choosers replays exactly.

THE TWO DRAWS OF A CHOICE. lab/judge/signals.py reads two draws for each
choice: the draw of this character (`p_char`, the Decision's `probs`) and
the draw of an average character over the same options (`p_avg`): the same
chooser with every trait at the middle. `average(req)` gives p_avg. For the
rules it is softmax((prior_avg + habit) / T). For Combined it is the model's
view with no trait bias, as in the lab's bias mode. A chooser that has no
average gives None, and the backend uses the rules' average.

A REMOTE chooser (a model in another process, step M11) has `remote = True`.
The loop counts each request that goes to the model, tick by tick (loop.py).
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
import random

from .api import Decision
from .core import T_RULES, softmax

TAU = 0.5           # Combined: the temperature of the model's view (the lab's Model.TAU)
BETA = 1.0          # Combined: the weight of the trait bias (the lab's Model beta)
# Combined: the plausibility guard. The trait bias moves an option only when
# the model gives it at least GUARD times the probability of its best option
# (lab/results/methods/combine.md, A2: d = 0.5). 0 turns the guard off.
GUARD = 0.5


def seed_of(*parts):
    """A seed made from `parts`: the same in every process. (The hash() of a
    str changes from one process to the next.)"""
    data = "\x1f".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def draw(p, seed, req):
    """An index drawn with the probabilities `p`, by the generator of request
    `req`: seeded from (seed, who, request id)."""
    return random.Random(seed_of(seed, req.who, req.id)).choices(range(len(p)), weights=p)[0]


def one_hot(n, i):
    """The probabilities of a pick that is certain."""
    return tuple(1.0 if j == i else 0.0 for j in range(n))


def zscore(scores):
    """A model's scores as log-weights (when they are probabilities), then
    moved to mean 0 and spread 1, so that every choice has the same spread.
    This is the lab's _z (lab/world2/policy.py)."""
    if all(0 <= x <= 1 for x in scores):
        scores = [math.log(max(x, 1e-9)) for x in scores]
    mu = sum(scores) / len(scores)
    sd = math.sqrt(sum((x - mu) ** 2 for x in scores) / len(scores)) or 1.0
    return [(x - mu) / sd for x in scores]


class _Sync:
    """Common code of a synchronous chooser. `decide(req)` gives the
    Decision. `submit` keeps it, and `poll` gives it out once."""
    remote = False

    def __init__(self):
        self._done = {}

    def submit(self, req):
        if req.id in self._done:
            raise ValueError(f"{self.name}: request {req.id} was submitted already")
        self._done[req.id] = self.decide(req)

    def poll(self, req_id, block=True):
        return self._done.pop(req_id, None)

    def average(self, req):
        return None


class Rules(_Sync):
    """The rules: draw from softmax((prior + habit) / t). With `argmax`, take
    the best option (the first of two equal ones) and draw nothing."""
    name = "rules"

    def __init__(self, seed=0, t=T_RULES, argmax=False):
        super().__init__()
        self.seed, self.t, self.argmax = seed, t, argmax

    def probs(self, req):
        return tuple(softmax([a + h for a, h in zip(req.prior, req.habit)], self.t))

    def average(self, req):
        return tuple(softmax([a + h for a, h in zip(req.prior_avg, req.habit)], self.t))

    def decide(self, req):
        p = self.probs(req)
        i = max(range(len(p)), key=p.__getitem__) if self.argmax else draw(p, self.seed, req)
        return Decision(req.id, i, self.name, p)


class Random(_Sync):
    """Every option the same, for this character and for the average one."""
    name = "random"

    def __init__(self, seed=0):
        super().__init__()
        self.seed = seed

    def probs(self, req):
        return (1 / len(req.options),) * len(req.options)

    average = probs

    def decide(self, req):
        p = self.probs(req)
        return Decision(req.id, draw(p, self.seed, req), self.name, p)


class Gate:
    """The rules pick when their top probability is at least `g`. Else the
    model picks. The design measured g = 0.7 on the inn: there the rules
    answer 35% of the decisions, and the model is not asked. The Decision
    names the chooser that picked: the rules, or the model."""
    name = "gate"
    remote = False

    def __init__(self, g, rules, model):
        if not 0 < g <= 1:
            raise ValueError(f"the gate is a probability above 0, not {g}")
        self.g, self.rules, self.model = g, rules, model
        self._asked = {}            # request id -> the chooser that has it

    def route(self, req):
        """The chooser that answers `req`."""
        return self.rules if max(self.rules.probs(req)) >= self.g else self.model

    def submit(self, req):
        c = self._asked[req.id] = self.route(req)
        c.submit(req)

    def poll(self, req_id, block=True):
        c = self._asked.get(req_id)
        d = None if c is None else c.poll(req_id, block)
        if d is not None:
            del self._asked[req_id]
        return d

    def average(self, req):
        return self.route(req).average(req)


class Combined:
    """The model says what makes sense; the traits push some options up or
    down. The probability of option i is

        softmax(z_i / tau + beta * (prior_i - prior_avg_i) / t + habit_i / t)

    z are the model's scores (zscore). prior - prior_avg is the traits' part
    of the rule score: the score of this character less the score of an
    average one. Habit is counted once.

    THE PLAUSIBILITY GUARD. The bias moves only the options that the model
    already finds plausible: those with at least `guard` times the
    probability of the model's best option, in softmax(z / tau). Another
    option keeps the model's view (with `hard`, it cannot be picked). Thus
    the traits choose among the options that make sense, and never make an
    option sensible (design 1.1: traits are a guard and a bias inside the
    legal menu). `guard` 0 turns the guard off: then this is the lab's bias
    mode, float for float.

    `sense` is the model, a chooser whose Decision gives a score for each
    option in `probs`. It gets the request without the rule scores."""
    name = "combined"
    remote = False

    def __init__(self, sense, seed=0, tau=TAU, beta=BETA, t=T_RULES, guard=GUARD, hard=False):
        if not 0 <= guard <= 1:
            raise ValueError(f"the guard is a share from 0 to 1, not {guard}")
        self.sense, self.seed = sense, seed
        self.tau, self.beta, self.t, self.guard, self.hard = tau, beta, t, guard, hard
        self._asked = {}            # request id -> the request
        self._done = {}             # request id -> a Decision made without the model
        self._z = (None, None)      # (request id, z) of the last answer, for `average`

    def submit(self, req):
        if len(req.options) == 1:   # one option: nothing to ask (the lab asks nothing too)
            self._done[req.id] = Decision(req.id, 0, self.name, (1.0,))
            self._z = (req.id, [0.0])
            return
        zero = (0.0,) * len(req.options)
        self._asked[req.id] = req
        self.sense.submit(dataclasses.replace(req, prior=zero, prior_avg=zero, habit=zero))

    def poll(self, req_id, block=True):
        if req_id in self._done:
            return self._done.pop(req_id)
        req = self._asked.get(req_id)
        d = None if req is None else self.sense.poll(req_id, block)
        if d is None:
            return None
        del self._asked[req_id]
        if d.probs is None or len(d.probs) != len(req.options):
            raise ValueError(f"{self.sense.name} gave no score for each option of request {req_id}")
        z = zscore(d.probs)
        self._z = (req_id, z)
        p = self.probs(req, z)
        return Decision(req_id, draw(p, self.seed, req), self.name, p, fallback=d.fallback)

    def probs(self, req, z):
        """The draw of this character, given the model's z."""
        view = [x / self.tau for x in z]
        top = max(view)
        floor = top + math.log(self.guard) if self.guard > 0 else -math.inf
        out = []
        for v, a, b, h in zip(view, req.prior, req.prior_avg, req.habit):
            if v >= floor:
                out.append(v + self.beta * (a - b) / self.t + h / self.t)
            else:
                out.append(-math.inf if self.hard else v + h / self.t)
        return tuple(softmax(out))

    def average(self, req):
        """The draw of an average character: the model's view and habit, with
        no trait bias."""
        rid, z = self._z
        if rid != req.id:
            return None
        return tuple(softmax([x / self.tau + h / self.t for x, h in zip(z, req.habit)]))


class Missing(LookupError):
    """Replay has no pick for a request."""


class Replay(_Sync):
    """The picks of an earlier run, by request id. A STUB: step M6 reads the
    picks from a decisions file (src/mind/record.py).

    `picks` maps a request id to the index of its pick, or to a Decision. A
    run that replays asks the same questions, so a request with no pick is an
    error (Missing). With `fallback` (a synchronous chooser), that chooser
    answers it instead, and the Decision says that it is a fallback."""
    name = "replay"

    def __init__(self, picks, fallback=None):
        super().__init__()
        self.picks, self.fallback = picks, fallback

    def decide(self, req):
        got = self.picks.get(req.id)
        if got is None:
            if self.fallback is None:
                raise Missing(f"no recorded pick for request {req.id}")
            self.fallback.submit(req)
            return dataclasses.replace(self.fallback.poll(req.id), fallback=True)
        if isinstance(got, Decision):
            if got.request != req.id:
                raise ValueError(f"the decision for {got.request} was given for request {req.id}")
            return got
        if not 0 <= got < len(req.options):
            raise ValueError(f"the recorded pick {got} is not an option of request {req.id}")
        return Decision(req.id, got, self.name)


class Player(_Sync):
    """A person picks. `ask(req)` shows the request (the scene text and the
    options, with the group of each) and gives back the number of the pick,
    or a list of numbers: the pick, then more options to start one after the
    other (a chain, Job.then). The loop gives a player the full menu, and
    never asks the model for it."""
    name = "player"

    def __init__(self, ask):
        super().__init__()
        self.ask = ask

    def decide(self, req):
        got = self.ask(req)
        picks = [got] if isinstance(got, int) else list(got)
        n = len(req.options)
        if not picks or not all(isinstance(i, int) and 0 <= i < n for i in picks):
            raise ValueError(f"a pick is the number of an option, from 0 to {n - 1}, not {got!r}")
        return Decision(req.id, picks[0], self.name, one_hot(n, picks[0]), then=tuple(picks[1:]))
