"""The types of the mind interface: what a backend gives the mind, and what the
mind gives back (docs/sim-mind-design.md 3.5).

The mind works with TASKS ("take the axe"), not with limb moves. A backend (the
inn in `src/rooms`, the voxel world through `src/voxel/mindlink.py`) writes its
world into these types. The neutral functions of `src/mind` make a menu and a
scene text from them. A chooser picks an option. The backend runs the pick as
a job and reports what happened.

RULES THAT THESE TYPES KEEP.
- This module uses only the standard library. It imports no backend and no
  `lab` code.
- A frozen type holds no dict, list or set. A mapping is kept as a tuple of
  (name, value) pairs, sorted by name, and a list is kept as a tuple. The
  constructor does this conversion, so a caller can give a dict or a list.
  Thus every frozen value hashes, and two equal values have the same hash.
- A chooser gets a `DecisionRequest` and nothing more. A request holds texts
  and rule scores. It holds no `Ref`, no tick, and no id other than its own
  id and the id of who decides. A model chooser sends only the situation, the
  question and the option texts.
- A `Job` is plain data, so a snapshot can copy and pickle it.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Protocol, runtime_checkable

# How a mind got a percept: it saw, heard or felt it, somebody told it, it did
# the act itself, or it noticed a state ("the purse is gone").
CHANNELS = ("saw", "heard", "felt", "told", "did", "noticed")
# The phase of an event: an act starts, then it is done or it is stopped.
PHASES = ("start", "done", "stopped")
# The status of a job, and the phases a job reports in a JobEvent.
JOB_STATUS = ("running", "done", "failed", "stopped")
JOB_PHASES = ("start", "act", "done", "failed", "stopped")


def pairs(items):
    """A mapping (or an iterable of pairs) as a tuple of (name, value) pairs,
    sorted by name. A name that occurs two times is an error."""
    got = items.items() if isinstance(items, Mapping) else items
    out = tuple(sorted(((str(k), v) for k, v in got), key=lambda kv: kv[0]))
    names = [k for k, _ in out]
    if len(set(names)) != len(names):
        raise ValueError(f"a name occurs more than once in {names}")
    return out


class _Frozen:
    """Common code of the frozen types. After the dataclass constructor, it
    changes each mapping to sorted pairs and each list or set to a tuple. Then
    it calls `_check`. A type uses `_check` to refuse values that break its
    rules."""

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, Mapping):
                object.__setattr__(self, f.name, pairs(v))
            elif isinstance(v, list):
                object.__setattr__(self, f.name, tuple(v))
            elif isinstance(v, (set, frozenset)):
                object.__setattr__(self, f.name, tuple(sorted(v)))
        self._check()

    def _check(self):
        pass


@dataclass(frozen=True)
class Ref(_Frozen):
    """Something a role can name: a person, a thing or a place.

    `id` is the key of the backend (a pid, an eid, a cell, a room). A chooser
    never sees it. `name` is the words for it: "the axe", "Sir Ada", "the
    cellar"."""
    kind: str                       # person | thing | place, or a word of the backend
    id: str
    name: str


@dataclass(frozen=True)
class Percept(_Frozen):
    """One thing that a mind perceived. The engine writes it from the side of
    the perceiver: a sound that the perceiver cannot place has no actor."""
    t: int
    channel: str                    # one of CHANNELS
    text: str                       # the words the perceiver can say about it
    source: str                     # the event id; NEWS is a source not had recently
    verb: str = ""                  # the verb of the event
    actor: Ref | None = None        # None when the perceiver cannot tell who did it
    target: Ref | None = None
    phase: str = "done"             # one of PHASES
    kind: str = ""                  # the interrupt class (catalog `interrupts_known`), or ""
    salience: float = 0.0

    def _check(self):
        if self.channel not in CHANNELS:
            raise ValueError(f"percept channel {self.channel!r} is not one of {CHANNELS}")
        if self.phase not in PHASES:
            raise ValueError(f"percept phase {self.phase!r} is not one of {PHASES}")


@dataclass(frozen=True)
class Belief(_Frozen):
    """What a mind thinks is true. It can be wrong. It keeps the event it came
    from, so the cause log can show why the mind thinks it."""
    kind: str                       # at | holds | state | hurt_by | took | ...
    about: Ref
    value: Ref | str                # a Ref for a person, thing or place; else a word
    text: str                       # the words the engine writes for it
    source: str                     # the event id, or "sight", "own", "start"
    how: str = "saw"                # saw | heard | told | looked | own
    t: int = 0


@dataclass(frozen=True)
class SelfState(_Frozen):
    """The facts a character knows about itself, in words."""
    name: str
    public: str = ""                # how others see it: "a knight"
    place: str = ""                 # where it is: "the cellar"
    holds: tuple[str, ...] = ()     # the names of what it holds
    feels: tuple[str, ...] = ()     # words from the engine: "hungry", "angry at Tom"
    drives: tuple[str, ...] = ()    # catalog drives that are above their threshold now


@dataclass(frozen=True)
class Option(_Frozen):
    """One task a character can start now: a catalog verb with its roles filled.

    `ok` is False for a BLOCKED option: the world permits the act, but the
    actor lacks something. Then `missing` gives the words of each capability
    that is missing ("a free hand"). A blocked option never goes to a chooser.
    It goes into the scene text as what is missing.

    `key` is the identity of the option: the verb and the id of each role.
    A backend uses it to find its own act for the option that was picked."""
    verb: str                       # a catalog verb id
    roles: tuple[tuple[str, Ref], ...]   # (role, Ref) pairs; a dict is accepted
    text: str                       # the words a chooser reads: "take the axe"
    group: str                      # the catalog group of the verb
    duration: int                   # in the backend's units (turns or ticks)
    interruptible: bool = True
    features: tuple[tuple[str, float], ...] = ()   # numbers the rule scores read
    ok: bool = True
    missing: tuple[str, ...] = ()

    def _check(self):
        # pairs given as a tuple keep the caller's order; sort them, so that
        # `key` and equality do not depend on that order
        object.__setattr__(self, "roles", pairs(self.roles))
        object.__setattr__(self, "features", pairs(self.features))
        for r, ref in self.roles:
            if not isinstance(ref, Ref):
                raise ValueError(f"option {self.text!r}: role {r!r} is not a Ref")
        if self.ok == bool(self.missing):
            raise ValueError(f"option {self.text!r}: a blocked option says what is missing,"
                             f" and only a blocked one does (ok={self.ok}, missing={self.missing})")

    @property
    def key(self):
        return "|".join([self.verb] + [f"{r}={ref.id}" for r, ref in self.roles])

    def role(self, name):
        """The Ref that fills role `name`, or None."""
        return next((ref for r, ref in self.roles if r == name), None)


@dataclass(frozen=True)
class Menu(_Frozen):
    """What one decision chooses from.

    `options` are the legal options a chooser sees, in canonical order (group,
    then verb, then text; never by score). `prior`, `prior_avg` and `habit`
    give one number for each option: the rule score for this character, the
    rule score for an average character, and the pull of habit, which is
    counted once. `full` holds every legal option before the pre-rank cut,
    and `cut` holds the legal options that the cut removed, so the cut is on
    record. A player's menu is never cut. `blocked` holds the options that
    lack a capability."""
    options: tuple[Option, ...]
    blocked: tuple[Option, ...]
    full: tuple[Option, ...]
    cut: tuple[Option, ...]
    prior: tuple[float, ...]
    prior_avg: tuple[float, ...]
    habit: tuple[float, ...]

    def _check(self):
        n = len(self.options)
        if not n:
            raise ValueError("a menu has at least one option")
        if not len(self.prior) == len(self.prior_avg) == len(self.habit) == n:
            raise ValueError("prior, prior_avg and habit have one number for each option")
        texts = [o.text for o in self.options]
        if len(set(texts)) != n:
            raise ValueError(f"two options of a menu have the same text: {texts}")
        if not all(o.ok for o in self.options + self.cut):
            raise ValueError("a blocked option is never offered and never cut")
        if any(o.ok for o in self.blocked):
            raise ValueError("an option in `blocked` has ok=False")
        kept = sorted(o.key for o in self.options + self.cut)
        if kept != sorted(o.key for o in self.full):
            raise ValueError("`options` and `cut` together are exactly `full`")

    @property
    def groups(self):
        """((group, (index, ...)), ...): the options of each group, in menu order."""
        out = {}
        for i, o in enumerate(self.options):
            out.setdefault(o.group, []).append(i)
        return tuple((g, tuple(ix)) for g, ix in out.items())


@dataclass(frozen=True)
class Why(_Frozen):
    """Why a job was started: the cause-log entry of one decision."""
    decision: str                   # the id of the DecisionRequest
    chooser: str                    # the name of the chooser that picked
    reason: str                     # why the mind was asked: idle | done | failed:... | interrupt:...
    percepts: tuple[str, ...] = ()  # the sources of the percepts in the scene text
    beliefs: tuple[str, ...] = ()   # the sources of the beliefs in the scene text
    score: float = 0.0              # the chooser's probability of the pick
    margin: float = 0.0             # the gap from the pick to the next best option
    fallback: bool = False          # True when a realtime model was late and the rules picked


@dataclass
class Job:
    """A task that a mind chose and the backend runs over many ticks.

    This is plain data, so a snapshot can copy and pickle it. `state` is for
    the backend's progress (the current step, when a limb was asked). It holds
    only plain values: numbers, strings, tuples, lists and dicts of them."""
    id: str                         # the id of the decision that started it
    who: str
    option: Option
    started: int
    why: Why
    status: str = "running"         # one of JOB_STATUS
    ended: int | None = None
    reason: str = ""                # why it ended: "interrupt:struck", "no took after asking"
    then: tuple[Option, ...] = ()   # options to run next: a player's chain
    state: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in JOB_STATUS:
            raise ValueError(f"job status {self.status!r} is not one of {JOB_STATUS}")
        self.then = tuple(self.then)


@dataclass(frozen=True)
class JobEvent(_Frozen):
    """One step in the life of a job: it started, it acted, it ended."""
    t: int
    who: str
    job: str                        # Job.id
    verb: str
    phase: str                      # one of JOB_PHASES
    text: str = ""                  # the told text that others can perceive, or ""
    reason: str = ""

    def _check(self):
        if self.phase not in JOB_PHASES:
            raise ValueError(f"job event phase {self.phase!r} is not one of {JOB_PHASES}")


@dataclass(frozen=True)
class DecisionRequest(_Frozen):
    """What a chooser gets: one question about one character.

    `options` are the option texts in canonical order, and `groups` gives the
    group of each. `prior`, `prior_avg` and `habit` are the rule numbers of
    each option, for the rules and for the trait bias. A model never gets
    them. `who` and `id` only seed a chooser that draws: the seed comes from
    (world seed, who, id)."""
    id: str
    who: str
    reason: str
    situation: str
    question: str
    options: tuple[str, ...]
    groups: tuple[str, ...]
    prior: tuple[float, ...]
    prior_avg: tuple[float, ...]
    habit: tuple[float, ...]

    def _check(self):
        n = len(self.options)
        if not n:
            raise ValueError(f"request {self.id}: no options")
        if len(set(self.options)) != n:
            raise ValueError(f"request {self.id}: two options have the same text")
        if not len(self.groups) == len(self.prior) == len(self.prior_avg) == len(self.habit) == n:
            raise ValueError(f"request {self.id}: groups and scores have one entry for each option")


@dataclass(frozen=True)
class Decision(_Frozen):
    """A chooser's answer to one request."""
    request: str                    # DecisionRequest.id
    pick: int                       # an index into the request's options
    chooser: str
    probs: tuple[float, ...] | None = None   # the chooser's probability of each option
    then: tuple[int, ...] = ()      # more picks to run after it: a player's chain
    fallback: bool = False


@runtime_checkable
class Chooser(Protocol):
    """What picks an option from a menu. `submit` sends a request. `poll`
    returns the decision, or None when it is not ready and `block` is False.
    A synchronous chooser (Rules) has the answer before `submit` returns."""
    name: str

    def submit(self, req: DecisionRequest) -> None: ...

    def poll(self, req_id: str, block: bool = True) -> Decision | None: ...


@runtime_checkable
class Backend(Protocol):
    """A world that the mind can drive. A backend can have more methods.

    `name` is the key of the backend in the verb catalog ("rooms", "voxel").
    `who` is the backend's id of a person."""
    name: str

    def needs_decision(self) -> list[tuple[str, str]]:
        """(who, reason) for each person that must choose now."""
        ...

    def self_state(self, who: str) -> SelfState: ...

    def percepts(self, who: str) -> tuple[Percept, ...]: ...

    def beliefs(self, who: str) -> tuple[Belief, ...]: ...

    def bind(self, who: str) -> tuple[Option, ...]:
        """Every option the catalog verbs bind to now, the blocked ones too."""
        ...

    def rule_scores(self, who: str, options: tuple[Option, ...]) -> tuple[tuple[float, ...], ...]:
        """(prior, prior_avg, habit), each with one number for each option."""
        ...

    def start(self, who: str, option: Option, why: Why) -> Job:
        """Check the option again against the live world, and start it."""
        ...
