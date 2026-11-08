"""PROTOTYPE of the backend-neutral mind interface (design only, scratch).

Dataclasses + chooser protocol + the neutral functions (menu pre-rank,
describe rules). No backend code here."""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Protocol

SPELLING = "mark1-mind-v0"
MENU_MAX = 16


@dataclass(frozen=True)
class Ref:
    id: str
    name: str
    kind: str                    # person | thing | place | self | direction


@dataclass(frozen=True)
class Percept:
    t: int
    channel: str                 # saw | heard | felt | told | noticed
    text: str                    # engine-written, from the perceiver's side
    event_id: str
    actor: str | None = None
    verb: str = ""
    target: str | None = None
    phase: str = "done"          # start | done | stopped
    salience: float = 0.0


@dataclass(frozen=True)
class Belief:
    kind: str                    # where | took | ...
    about: str
    value: str
    source: str                  # event id, or sight/own/start
    t: int = 0


@dataclass
class Option:
    verb: str
    roles: dict                  # role -> Ref
    text: str
    duration: int
    features: dict = field(default_factory=dict)
    ok: bool = True
    missing: tuple = ()          # capability phrases, when ok is False
    hint: str = ""
    raw: dict | None = None      # backend payload (never shown to a chooser)

    @property
    def key(self):
        return self.verb + "|" + "|".join(f"{k}={r.id}" for k, r in sorted(self.roles.items()))


@dataclass
class Why:
    decision_id: str
    chooser: str
    reason: str
    percepts: tuple = ()
    beliefs: tuple = ()
    score: float = 0.0
    margin: float = 0.0


@dataclass
class Job:
    who: str
    option: Option
    started: int
    left: int
    why: Why
    status: str = "running"      # running | done | failed | stopped
    interruptible: bool = True
    result: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionRequest:
    id: str
    who: str
    t: int
    situation: str
    question: str
    options: tuple
    prior: tuple                 # rules scores of the options, same order
    reason: str
    apply_at: int


@dataclass(frozen=True)
class Decision:
    request_id: str
    pick: int
    scores: tuple | None
    chooser: str


class Chooser(Protocol):
    name: str
    def submit(self, req: DecisionRequest) -> None: ...
    def poll(self, req_id: str, block: bool) -> Decision | None: ...


def softmax(z, t=1.0):
    m = max(z)
    e = [math.exp((x - m) / t) for x in z]
    s = sum(e)
    return [x / s for x in e]


class PriorChooser:
    """Samples from softmax(prior / T) with a seeded rng: the rules chooser
    when `prior` is the rule utility. Synchronous."""
    def __init__(self, name="rules", T=0.2, rng=None, seed=0):
        self.name, self.T, self.rng, self.seed = name, T, rng, seed
        self._done = {}

    def submit(self, req):
        pc = softmax(list(req.prior), self.T)
        rng = self.rng or random.Random(_h(self.seed, req.id))
        i = rng.choices(range(len(pc)), weights=pc)[0]
        self._done[req.id] = Decision(req.id, i, tuple(pc), self.name)

    def poll(self, req_id, block=True):
        return self._done.pop(req_id, None)


def _h(*xs):
    return int(hashlib.sha1("|".join(map(str, xs)).encode()).hexdigest()[:12], 16)


def prerank(options, scores, k=MENU_MAX, group=lambda o: o.verb):
    """Keep at most k: the best of each verb first, then the best of the rest.
    The null act (verb 'wait') never falls off. Returns kept indices in a
    stable order (by score)."""
    n = len(options)
    if n <= k:
        return list(range(n))
    order = sorted(range(n), key=lambda i: -scores[i])
    keep, seen = [], set()
    for i in order:
        if options[i].verb == "wait":
            keep.append(i)
            seen.add("wait")
    for i in order:
        g = group(options[i])
        if g not in seen and len(keep) < k:
            keep.append(i)
            seen.add(g)
    for i in order:
        if i not in keep and len(keep) < k:
            keep.append(i)
    return sorted(keep, key=lambda i: -scores[i])
