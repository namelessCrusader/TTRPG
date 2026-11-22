"""The neutral functions of the mind: from what a backend gives to the menu a
chooser sees and to the facts that the scene text shows
(docs/sim-mind-design.md 3.5, step M4).

A backend gives the character's own state, its percepts, its beliefs, every
option that the catalog verbs bind to (the blocked ones too), and three rule
scores for each option. These functions know no backend. They read only the
types of `api` and the verb catalog, and they are pure: the same input gives
the same output, in every process.

    perceive   the last few percepts, oldest first: at most PERCEPTS
    menu       the menu: pre-ranked to MENU_MAX or fewer options, in
               canonical order, always with the null act; the blocked
               options apart, and the cut on record
    people     the people the scene names, and what they hold
    relevant   the beliefs that bear on the menu: at most BELIEFS
    needs      the drives that nothing on the menu serves, and what is missing
    scene      all of the facts above for one decision. describe.py writes
               them as words.

THE MENU (design 1.1 and 3.5). The model that picks reads a list of option
texts. The lab measured 62% right picks with 16 options and 47% with 64, so
an NPC menu has 16 options or fewer. The rules cut it first (the pre-rank):

1. Score each legal option with max(prior, prior_avg) + habit. That keeps the
   options that this character OR an average character would pick, so the
   trait bias of a chooser can still reach them. Habit is counted once.
2. Keep options by score until the kept ones hold MASS of the probability
   that the rules give with that score (softmax at T_RULES), or FIRST options.
3. Add the null act, then the best option of each verb that is not kept yet,
   then the best of the rest, up to MENU_MAX.

The kept options go to the chooser in CANONICAL ORDER: the group of the verb
(in the order of the catalog's `groups`), then the verb (in the order of the
catalog file), then the text. Never by score: a list sorted by score puts the
rules' favourite first every time, and that tells a model the answer. Ties of
score go in canonical order too, so the menu does not depend on the order in
which the backend lists its options.

Two options with the same text are one choice to a reader. The better one
stays and the others are cut. A player's menu is never cut by the pre-rank.
A blocked option (ok False) is never offered and never cut: it goes to
`Menu.blocked`, and the scene says what it lacks.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass

from . import catalog as _catalog
from .api import Belief, Menu, Option, Percept, Ref, SelfState

MENU_MAX = 16       # the most options an NPC menu has (design 1.1)
BATCH = 4           # the options a chooser weighs in one pick when a long menu is MERGED (tourney.py)
MASS = 0.95         # step 2 of the pre-rank: the share of the rules' probability to keep
FIRST = 12          # step 2 of the pre-rank: the most options it keeps by score
T_RULES = 0.2       # the temperature of the rules (the Rules chooser draws with it)
PERCEPTS = 5        # the most percepts a scene shows
BELIEFS = 2         # the most beliefs a scene shows
# the kinds of belief that only say where a thing is: "the axe is in the
# yard" (at: about the thing), "Oren has the lute" (holds: about the person)
LOCATING = ("at", "holds")


@functools.lru_cache(maxsize=1)
def catalog():
    """The shipped verb catalog, read once."""
    return _catalog.load()


def softmax(z, t=1.0):
    m = max(z)
    e = [math.exp((x - m) / t) for x in z]
    s = sum(e)
    return [x / s for x in e]


# ---------------------------------------------------------------- the types of a scene
@dataclass(frozen=True)
class Present:
    """A person the scene names: someone the character can act on now. `holds`
    are the names of what the character believes that person holds, and
    `state` the words of the person's state ("lying on the floor"), both from
    the character's beliefs."""
    who: Ref
    holds: tuple[str, ...] = ()
    state: tuple[str, ...] = ()


@dataclass(frozen=True)
class Need:
    """A drive above its threshold that no legal option serves.

    When a BLOCKED option would serve it, `blocked` is that option, and `says`
    are the words of everything it lacks: all of them are needed. Else `says`
    are the words of the capabilities that the verbs which serve the drive
    hint at: any one of them would do ("something to eat")."""
    drive: str
    says: tuple[str, ...]
    blocked: Option | None = None


@dataclass(frozen=True)
class Scene:
    """The facts that the scene text of one decision shows. `seen` are the
    beliefs behind `people` (what they hold, their state); `beliefs` are the
    others that the text states, at most BELIEFS."""
    state: SelfState
    people: tuple[Present, ...]
    needs: tuple[Need, ...]
    percepts: tuple[Percept, ...]
    beliefs: tuple[Belief, ...]
    seen: tuple[Belief, ...] = ()

    def __post_init__(self):
        if len(self.percepts) > PERCEPTS:
            raise ValueError(f"a scene shows at most {PERCEPTS} percepts, not {len(self.percepts)}")
        if len(self.beliefs) > BELIEFS:
            raise ValueError(f"a scene shows at most {BELIEFS} beliefs, not {len(self.beliefs)}")

    @property
    def sources(self):
        """(percept sources, belief sources): the events behind what the text
        says, for the Why of the decision. Each source is given once."""
        bel = dict.fromkeys(b.source for b in self.beliefs + self.seen)
        return tuple(x.source for x in self.percepts), tuple(bel)


# ---------------------------------------------------------------- perceive
def perceive(percepts, k=PERCEPTS):
    """The last `k` percepts, oldest first. Percepts of one tick keep the
    backend's order. A source given more than once counts once, at its last
    place."""
    got = {}
    for x in sorted(percepts, key=lambda x: x.t):
        got.pop(x.source, None)
        got[x.source] = x
    return tuple(got.values())[-k:] if k > 0 else ()


# ---------------------------------------------------------------- the menu
def canonical(cat, o):
    """The place of an option in canonical order: the group, then the verb,
    then the text."""
    try:
        return cat.order(o.verb), o.text
    except KeyError:
        raise ValueError(f"option {o.text!r}: verb {o.verb!r} is not in the catalog") from None


def menu(options, prior, prior_avg, habit, cat=None, player=False, cap=MENU_MAX):
    """The Menu of one decision. `options` are every option the backend bound,
    blocked ones too. `prior`, `prior_avg` and `habit` give one rule score
    for each of them (a blocked option's scores are not read). `player`:
    the menu of a player, which the pre-rank never cuts."""
    cat = cat or catalog()
    n = len(options)
    if not len(prior) == len(prior_avg) == len(habit) == n:
        raise ValueError("prior, prior_avg and habit have one number for each option")
    if len({o.key for o in options}) != n:
        raise ValueError("two options have the same key")
    if cap < 2:
        raise ValueError("a menu has room for the null act and one more option")
    # canonical order; the key orders two options with the same text
    place = [canonical(cat, o) + (o.key,) for o in options]
    score = [max(a, b) + h for a, b, h in zip(prior, prior_avg, habit)]
    # the legal options, best first; ties of score in canonical order
    best = sorted((i for i in range(n) if options[i].ok), key=lambda i: (-score[i], place[i]))
    uniq, texts = [], set()
    for i in best:
        if options[i].text not in texts:
            uniq.append(i)
            texts.add(options[i].text)
    null = next((i for i in uniq if options[i].verb == cat.null_act.id), None)
    if null is None:
        raise ValueError(f"a menu always has the null act ({cat.null_act.id}); bind it")
    keep = uniq if player or len(uniq) <= cap else _prerank(options, uniq, score, null, cap)
    kept = set(keep)
    keep = sorted(keep, key=place.__getitem__)

    def canon(ix):
        return tuple(options[i] for i in sorted(ix, key=place.__getitem__))

    return Menu(tuple(options[i] for i in keep),
                canon(i for i in range(n) if not options[i].ok),
                canon(best),
                canon(i for i in best if i not in kept),
                tuple(prior[i] for i in keep), tuple(prior_avg[i] for i in keep),
                tuple(habit[i] for i in keep))


def _prerank(options, uniq, score, null, cap):
    """The indices that the pre-rank keeps. `uniq` holds the legal options
    with distinct texts, best first; `null` is the null act among them."""
    p = softmax([score[i] for i in uniq], T_RULES)
    first = min(FIRST, cap - 1)                 # leave room for the null act
    keep, mass = [], 0.0
    for i, pi in zip(uniq, p):
        if mass >= MASS or len(keep) >= first:
            break
        keep.append(i)
        mass += pi
    if null not in keep:
        keep.append(null)
    verbs = {options[i].verb for i in keep}
    for i in uniq:                              # the best of each verb not kept yet
        if len(keep) >= cap:
            break
        if options[i].verb not in verbs:
            keep.append(i)
            verbs.add(options[i].verb)
    for i in uniq:                              # then the best of the rest
        if len(keep) >= cap:
            break
        if i not in keep:
            keep.append(i)
    return keep


# ---------------------------------------------------------------- the facts of a scene
def people(m, beliefs):
    """The people the scene names, and the beliefs it uses for them.

    A person is named when it fills the `target` role of a legal or blocked
    option. In the catalog, `target` is what an act is done to, and a backend
    binds a person to it only when the character can act on that person now
    (the inn: in the same room). A person in another role can be away: in
    "tell Marta that Oren took the purse", Oren can be in the cellar. People
    are named in the order of their names.

    What a person holds and its state come from the beliefs: an `at` belief
    about a thing whose value is that person, a `holds` belief about that
    person, and a `state` belief about that person."""
    who = {}
    for o in m.full + m.blocked:
        r = o.role("target")
        if r is not None and r.kind == "person":
            who.setdefault(r, None)
    out, seen = [], []
    for r in sorted(who, key=lambda r: (r.name.casefold(), r.name, r.id)):
        holds, state = {}, {}
        for b in beliefs:
            if b.kind == "at" and b.about.kind == "thing" and b.value == r:
                holds.setdefault(b.about.name, None)
            elif b.kind == "holds" and b.about == r and isinstance(b.value, Ref):
                holds.setdefault(b.value.name, None)
            elif b.kind == "state" and b.about == r and isinstance(b.value, str):
                state.setdefault(b.value, None)
            else:
                continue
            seen.append(b)
        out.append(Present(r, tuple(holds), tuple(state)))
    return tuple(out), tuple(dict.fromkeys(seen))


def _located(b):
    """The thing that a LOCATING belief places, or None for another belief."""
    if b.kind not in LOCATING or not isinstance(b.value, Ref):
        return None
    return b.about if b.kind == "at" else b.value


def relevant(state, m, beliefs, seen=(), k=BELIEFS):
    """The beliefs the scene states: at most `k`, the most relevant first.

    Left out are the beliefs that the scene already says (`seen`, and what
    the character itself holds) and every belief that places a thing in the
    character's own place: those are the things that lie about, and the lab
    measured that a sentence which names a useless thing makes the model
    pick that thing (100%). The others, in this order:
    1. beliefs about something on the menu: a LOCATING belief whose thing
       fills a role of an offered option, or another belief whose `about` or
       `value` does ("Oren took the purse", when "accuse Oren of taking the
       purse" is offered);
    2. other beliefs that are not LOCATING ("the lute is gone").
    A LOCATING belief about a thing that is not on the menu is never stated.
    Nor is a belief about the character's own deed ("Fenn took the purse"),
    unless someone told it: then it says what others think.
    Within each class the fresher belief goes first, then the backend's
    order."""
    seen = set(seen)
    on_menu = {r for o in m.options for _, r in o.roles}
    ranked = []
    for i, b in enumerate(beliefs):
        if b in seen:
            continue
        thing, v = _located(b), b.value
        if thing is not None:
            holder = b.about if b.kind == "holds" else v
            if holder.kind == "person" and holder.name == state.name:
                continue                        # what it holds: the scene says it
            if b.kind == "at" and v.kind == "place" and v.name == state.place:
                continue                        # a thing that lies here
            if thing not in on_menu:
                continue
            tier = 0
        else:
            if isinstance(v, Ref) and v.kind == "person" and v.name == state.name and b.how != "told":
                continue                        # its own deed ("Fenn took the purse"): it knows
            tier = 0 if b.about in on_menu or (isinstance(v, Ref) and v in on_menu) else 1
        ranked.append((tier, -b.t, i, b))
    return tuple(b for *_, b in sorted(ranked, key=lambda x: x[:3])[:k])


def needs(state, m, cat=None, backend=None):
    """What is missing for each drive of the character that is above its
    threshold and that no legal option serves (`Need`). A verb serves a drive
    when the catalog says so; with `backend`, only the verbs that this
    backend runs count. A drive that a legal option serves needs nothing:
    the way to meet it is on the menu, or was cut with the rest. A drive whose
    verbs give no words (no blocked option, no hint) gives no Need."""
    cat = cat or catalog()
    for d in state.drives:
        if d not in cat.drives:
            raise ValueError(f"drive {d!r} is not in the catalog")
    out = []
    for d in cat.drives:
        if d not in state.drives:
            continue
        serving = [v for v in cat.verbs.values()
                   if d in v.serves and (backend is None or backend in v.runs_on)]
        ids = {v.id for v in serving}
        if any(o.verb in ids for o in m.full):
            continue
        stuck = next((o for o in m.blocked if o.verb in ids), None)
        if stuck is not None:
            out.append(Need(d, stuck.missing, stuck))
            continue
        says = tuple(dict.fromkeys(cat.capabilities[v.hint].says for v in serving if v.hint))
        if says:
            out.append(Need(d, says))
    return tuple(out)


def scene(state, percepts, beliefs, m, cat=None, backend=None):
    """The facts of one decision's scene: who is here, what is missing, the
    last few percepts and the beliefs that bear on the menu."""
    cat = cat or catalog()
    who, seen = people(m, beliefs)
    return Scene(state, who, needs(state, m, cat, backend), perceive(percepts),
                 relevant(state, m, beliefs, seen), seen)
