"""Round 3, step 0a of the learned mind: does the world pay? (docs/learned-mind-target.md, section 6 and 9.6)

The world is checked BEFORE anyone changes the mind. Fixed choosers that do not learn play lifetimes in the
inn, and we read what each social stance earns. Nothing here is a mind, and nothing in the world is changed.

ARCHETYPES. Each is a focal chooser; the other four cast members (and the bandit) use `rules`. They share ONE
self-care and differ ONLY in social stance:

    shared self-care   rules' own score (policy.utility) over every option that does not target a person
                       (walk, take, eat, drink, sit down, sleep, look around, open, play, wait, flee, shout), plus
                       asking for or offering for FOOD or DRINK (a person-directed act, but it is how the cook's
                       food is reached; rules scores it, so every archetype except the strict hermit keeps it).
    stance             a short list of person-directed (or movement) acts. With chance RATE the chooser takes one of
                       them (a mixture, so `probs` stays a plain distribution); otherwise it does what rules does
                       on the shared self-care. Spontaneous stances use R_SPONT; reactive ones (answering an act,
                       defending) use R_REACT. Within the stance each verb counts the same, whatever the number of
                       options it has. A stance act that is not on the menu is not taken (nothing is forced).

    stance             acts, as the inn's menu lists them
    hermit             no person-directed act at all (not even asking for food); when someone is in the room it
                       walks to the room it believes holds the fewest people. (hermit_ask: the same, but it still
                       asks the cook for food or drink, so it shows the cost of avoidance without hunger.)
    always-harsh       insult, threaten, shove, refuse - toward whoever is present, at the spontaneous rate
    always-giving      give, offer, praise, help up, bandage, accept - toward whoever is present
    tit-for-tat        opens with `praise` to each person it has not met (once, whole life); then answers each act
                       aimed at it with the same act (give -> give the cheapest thing it holds, else praise; praise
                       -> praise; insult -> insult; strike -> strike; steal -> steal; ...), falling back to praise
                       for a kind act and insult for a harsh one. Memory of the last act lasts the whole life.
    extractor          threaten a person; within 60 turns, steal from the same person (or from anyone who sleeps
                       or lies down). It never asks politely except for food.
    hoarder            take loose things worth 1 or more, open the chest, refuse requests; it never gives,
                       offers (beyond food), praises, helps or accepts
    protector          answers an attack it saw: defend the victim (guard), threaten the attacker (tell it to
                       stop), strike the attacker. Also if the attack was on itself. Nothing else is social.
    feeder (extra)     not in the plan's list. The shared self-care and no other person-directed act, plus: when hungry
                       (0.4 or more) it asks the cook for food if she is in the room, else walks to the room where it
                       believes she is. Compare it with hermit_ask and hermit (the peaceful baselines), not with rules.
    provoke_hard,      (extra, oracle family only) the shared self-care, plus about one insult, blow, shove or threat a
    provoke_kind       night aimed at Oren (P-hard: it hits back) or at Fenn (P-kind: it never does). Compare the two
                       with each other; they share the same peaceful baseline (rules has social acts of its own).
    rules, random      the existing references.

ORACLE (the L2 setting). The focal faces Fenn as P-kind(0.9) and Oren as P-hard(0.9) (lab/world2/partners.py,
as in L2 life A), the other two cast members and the bandit on rules. `rules` is the partner-blind chooser. The
oracle KNOWS the two rules and uses rules' own scores for everything else. It does three things:
  avoid   never insult, strike, shove or threaten the P-hard partner (it hits back with chance 0.9);
  kind    when the P-kind partner is in the room it works the kind partner's rule: ask for something the partner
          holds (served with chance 0.9, and a thing that was the partner's own raises the focal's liking of the
          partner by 0.6), give a thing to it (paid back with a thing or praise), and when lonely it walks to the
          partner's room - but only while the focal's company or standing error is above its set point or its
          liking of the partner is under 0.5 (it acts only where its own drives can gain);
  both    the two together.
oracle gap = error of rules minus error of the oracle (paired by seed).

MEASURES. One focal lifetime of N nights; nights WIN0.. are scored: mean drive error (the lifetime's own
sampler, every 10 turns), nights the focal was dead at the end of the night, nights survived. Seeds 1-10, paired by
seed (the same cast member is focal for every chooser of a seed: NAMES[seed % 5]; in the oracle setting
Tom, Sir Ada or Marta). Intervals are 95% t intervals over the 10 seeds.

b, c, q, w from the logs of the same lifetimes (every person-directed act, whoever does it):
  b  the drop in the receiver's drive error, measured around the act itself (before and after the world applies
     it), plus the error the receiver could drop by eating or drinking a food or drink it was given
  c  the rise in the giver's drive error around the act, plus the use the giver loses by giving food or drink away
  q  the chance that a third party learns of the act: it was in the room and saw it, or was told of it that night
  w  the chance that the pair is in the same room again later that night / at some time the next night
The indirect-reciprocity condition is q*b > c (Nowak and Sigmund 2005 (u)).

    lab/.venv/bin/python -m lab.mindL.tournament run A --nights 100 --workers 4
    lab/.venv/bin/python -m lab.mindL.tournament run O --nights 100 --workers 4
    lab/.venv/bin/python -m lab.mindL.tournament_report
Heavy runs go under the lock: flock <lock> cap.sh nice -n 10 <command>.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import time
import traceback
from pathlib import Path

from lab.world2.lifetime import NAMES, Cast, hook, night_seed, night_stats
from lab.world2.partners import Hard, Kind
from lab.world2.policy import Random, Rules, T_RULES, softmax, utility
from lab.world2.world import (DEED, DRIVES, FLAGS, HARMFUL, HARSH_ACTS, KIND_ACTS, NATURE, PHYSICAL, RATE, START_ROOM,
                              World)

OUT = Path(__file__).resolve().parents[1] / "results" / "mindL3"
RUNS = Path(os.environ.get("MINDL3_RUNS", OUT / "0a_runs"))
SEEDS = tuple(range(1, 11))
FOCALS3 = ("Tom", "Sir Ada", "Marta")          # the L2 focal rotation (lab/mindL/tests_L.py)
WIN0 = 20                                      # nights before this are not scored
FADE = 0.9                                     # overnight liking fade, as in the L tests
TURNS = 300
R_SPONT = 0.30     # chance of a stance act at a choice when one is on the menu (spontaneous stances)
R_REACT = 0.90     # the same for stances that answer something (tit-for-tat, protector)
R_AVOID = 0.50     # the hermit's chance to leave a room that has people in it
R_PROVOKE = 0.02   # the provokers' chance per choice (about one provocation a night)
THREAT_SPAN = 60   # the extractor takes within this many turns of a threat
PROVOKE = {"insult", "strike", "shove", "threaten"}
HARSH_ALL = HARMFUL | HARSH_ACTS      # harsh acts for the logs: the world's harmful ones and refusing
HELP_ACTS = frozenset({"give", "help up", "bandage", "defend"})     # kind acts that cost the giver more than a word
REPLY_KIND = frozenset(KIND_ACTS) - {"gossip"}                        # telling a deed on is not a favour to the hearer
KIND, HARSH, HELP = "kind", "harsh", "help"
REPLY_SPAN = 40    # an act of the receiver / a witness within this many turns is a reply

# ---------------------------------------------------------------- the four worlds
# R = reprisal (the actor pays for harm in view); G = gossip and witnessing (deeds are told on, theft and lies are seen).
WORLDS = {
    "all2": frozenset(FLAGS),                                              # R on, G on
    "noR": frozenset(FLAGS) - {"reprisal"},                                # R off, G on
    "noG": frozenset(FLAGS) - {"gossip", "watch", "check"},                # R on, G off
    "none": frozenset(FLAGS) - {"reprisal", "gossip", "watch", "check"},   # both off
}


class FastRules(Rules):
    """Rules with p_avg = p_char. The 'average character' draw is not used here and costs a second pass of the
    scores; the choices (and so the lifetime) are exactly rules' (lab/mindL/test_tournament.py checks it)."""

    def probs(self, w, p, opts):
        pc = softmax([utility(w, p, o, p.traits) for o in opts], T_RULES)
        return pc, pc


# ---------------------------------------------------------------- the archetypes
def need_request(o):
    """Asking for or offering for food or drink: the cook holds the food, so this is how a hungry one eats."""
    return o["verb"] in ("ask", "offer") and o["f"].get("need", 0.0) > 0.0


def verb_uniform(groups):
    """{key: [option index]} -> {index: weight}: each key counts 1 in all, split equally among its options."""
    out = {}
    for idx in groups.values():
        for i in idx:
            out[i] = out.get(i, 0.0) + 1.0 / len(idx)
    return out


class Archetype:
    """The shared self-care plus a stance (see the module docstring)."""
    name = "archetype"
    keep_need_requests = True        # False: even asking for food is a social act (the strict hermit)
    allowed = frozenset()            # person-directed verbs the stance may take (for the check in the report)

    def __init__(self, seed=0, spont=R_SPONT, react=R_REACT):
        self.rng = random.Random(seed)
        self.spont, self.react = spont, react
        self._w, self._seen, self.night = None, 0, 0

    # lifetime hooks
    def night_start(self, w, night):
        self._w, self._seen, self.night = w, 0, night
        self.reset_night(w)

    def reset_night(self, w):
        pass

    def observe(self, w, p, new):
        """`new`: the log events since this chooser last chose."""

    def candidates(self, w, p, opts, base, social):
        """-> ({option index: weight}, chance of taking one). Weights need not sum to 1."""
        return {}, self.spont

    def is_social(self, w, o):
        return o["target"] in w.people and not (self.keep_need_requests and need_request(o))

    def probs(self, w, p, opts):
        if w is not self._w:
            self.night_start(w, w.night)
        new = w.log[self._seen:]
        self._seen = len(w.log)
        self.observe(w, p, new)
        soc = [self.is_social(w, o) for o in opts]
        base = [i for i, s in enumerate(soc) if not s]
        social = [i for i, s in enumerate(soc) if s]
        pb = softmax([utility(w, p, opts[i], p.traits) for i in base], T_RULES)       # rules' own self-care
        cand, rate = self.candidates(w, p, opts, base, social)
        tot = sum(cand.values())
        pc = [0.0] * len(opts)
        if tot > 0:
            for i, x in zip(base, pb):
                pc[i] = (1.0 - rate) * x
            for i, x in cand.items():
                pc[i] += rate * x / tot
        else:
            for i, x in zip(base, pb):
                pc[i] = x
        return pc, pc


class Hermit(Archetype):
    name = "hermit"
    keep_need_requests = False

    def candidates(self, w, p, opts, base, social):
        if not any(q.acts() for q in w.here(p)):
            return {}, self.spont
        walks = [(i, o["f"]["goal"]["company"]) for i, o in enumerate(opts) if o["verb"] == "walk to"]
        if not walks:
            return {}, self.spont
        fewest = min(c for _, c in walks)
        return {i: 1.0 for i, c in walks if c == fewest}, R_AVOID


class HermitAsk(Hermit):
    name = "hermit_ask"
    keep_need_requests = True


class AlwaysHarsh(Archetype):
    name = "always-harsh"
    allowed = frozenset({"insult", "threaten", "shove", "refuse"})

    def candidates(self, w, p, opts, base, social):
        g = {}
        for i in social:
            if opts[i]["verb"] in self.allowed:
                g.setdefault(opts[i]["verb"], []).append(i)
        return verb_uniform(g), self.spont


class AlwaysGiving(Archetype):
    name = "always-giving"
    allowed = frozenset({"give", "offer", "praise", "help up", "bandage", "accept"})

    def candidates(self, w, p, opts, base, social):
        g = {}
        for i in social:
            if opts[i]["verb"] in self.allowed:
                g.setdefault(opts[i]["verb"], []).append(i)
        return verb_uniform(g), self.spont


class TitForTat(Archetype):
    name = "tit-for-tat"
    allowed = frozenset({"praise", "give", "help up", "bandage", "defend", "accept", "insult", "threaten", "shove",
                         "strike", "steal", "accuse", "search", "refuse"})
    # what to answer with, first choice first (the same act when the menu has it)
    KIND_REPLY = {"give": ("give", "praise"), "praise": ("praise",), "help up": ("help up", "praise"),
                  "bandage": ("bandage", "praise"), "defend": ("defend", "praise"), "accept": ("accept", "praise"),
                  "gossip": ("praise",)}
    HARSH_REPLY = {"insult": ("insult",), "threaten": ("threaten", "insult"), "shove": ("shove", "insult"),
                   "strike": ("strike", "shove", "insult"), "steal": ("steal", "threaten", "insult"),
                   "accuse": ("accuse", "insult"), "search": ("search", "insult"), "refuse": ("refuse", "insult")}

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.last = {}          # person -> (class, verb) of the last act it did to me (the whole life)
        self.started = set()    # people I have opened with (the whole life)
        self.todo = {}          # person -> (class, verb, turn) an act of theirs I have not answered tonight

    def reset_night(self, w):
        self.todo = {}

    def observe(self, w, p, new):
        for e in new:
            if not e["done"]:
                continue
            if e["target"] == p.name and e["actor"] != p.name and e["actor"] in w.people:
                cls = "kind" if e["action"] in KIND_ACTS else ("harsh" if e["action"] in HARMFUL | HARSH_ACTS else None)
                if cls:
                    self.last[e["actor"]] = (cls, e["action"])
                    self.todo[e["actor"]] = (cls, e["action"], e["t"])
            elif e["actor"] == p.name and e["target"] in w.people and "options" in e:
                x = e["target"]
                self.started.add(x)
                cls = "kind" if e["action"] in KIND_ACTS else ("harsh" if e["action"] in HARMFUL | HARSH_ACTS else None)
                if x in self.todo and cls == self.todo[x][0]:
                    del self.todo[x]

    def candidates(self, w, p, opts, base, social):
        here = {opts[i]["target"] for i in social}
        g = {}
        for x, (cls, verb, t) in list(self.todo.items()):
            if w.t - t > 40:
                del self.todo[x]
                continue
            if x not in here:
                continue
            table = self.KIND_REPLY if cls == "kind" else self.HARSH_REPLY
            for v in table.get(verb, ("praise",) if cls == "kind" else ("insult",)):
                idx = [i for i in social if opts[i]["verb"] == v and opts[i]["target"] == x]
                if v == "give" and idx:               # give back the cheapest thing it holds
                    low = min(opts[i]["f"].get("gain", 0.0) for i in idx)
                    idx = [i for i in idx if opts[i]["f"].get("gain", 0.0) == low]
                if idx:
                    g[(x, v)] = idx
                    break
        if g:
            return verb_uniform(g), self.react
        for x in sorted(here):                          # the first move: kind, once per person
            if x not in self.last and x not in self.started:
                idx = [i for i in social if opts[i]["verb"] == "praise" and opts[i]["target"] == x]
                if idx:
                    g[(x, "praise")] = idx
        return verb_uniform(g), self.spont


class Extractor(Archetype):
    name = "extractor"
    allowed = frozenset({"threaten", "steal"})

    def reset_night(self, w):
        self.threatened = {}

    def observe(self, w, p, new):
        for e in new:
            if e["actor"] == p.name and e["action"] == "threaten" and e["done"] and "options" in e:
                self.threatened[e["target"]] = e["t"]

    def candidates(self, w, p, opts, base, social):
        g = {}
        for i in social:
            o = opts[i]
            q = w.people[o["target"]]
            recent = w.t - self.threatened.get(q.name, -10 ** 6) <= THREAT_SPAN
            if o["verb"] == "threaten" and not recent and q.acts():
                g.setdefault("threaten", []).append(i)
            elif o["verb"] == "steal" and (recent or q.down or w.asleep(q)):
                g.setdefault("steal", []).append(i)
        return verb_uniform(g), self.spont


class Hoarder(Archetype):
    name = "hoarder"
    allowed = frozenset({"refuse"})

    def candidates(self, w, p, opts, base, social):
        g = {}
        for i, o in enumerate(opts):
            if o["verb"] == "take" and o["f"].get("gain", 0.0) > 0 and not o["f"].get("theft"):
                g.setdefault("take", []).append(i)          # gather what is worth something
            elif o["verb"] == "open":
                g.setdefault("open", []).append(i)          # the chest holds coins
            elif o["verb"] == "refuse":
                g.setdefault("refuse", []).append(i)        # and keep it
        return verb_uniform(g), self.spont


class Protector(Archetype):
    name = "protector"
    allowed = frozenset({"defend", "threaten", "strike"})

    def candidates(self, w, p, opts, base, social):
        attackers, victims = set(), set()
        for q in w.here(p):
            if q.acts():
                a = w.recently_harmed(q.name, 20, acts=PHYSICAL, knower=p) - {p.name}
                if a:
                    victims.add(q.name)
                    attackers |= a
        attackers |= w.recently_harmed(p.name, 20, acts=PHYSICAL, knower=p)       # and when it is the victim
        g = {}
        for i in social:
            o = opts[i]
            if o["verb"] == "defend" and o["target"] in victims:
                g.setdefault("defend", []).append(i)
            elif o["verb"] in ("threaten", "strike") and o["target"] in attackers:
                g.setdefault(o["verb"], []).append(i)
        return verb_uniform(g), self.react


class Feeder(Archetype):
    """Not one of the plan's seven: rules, plus the one social act the world pays most for."""
    name = "feeder"
    allowed = frozenset()

    def candidates(self, w, p, opts, base, social):
        if p.hunger < 0.4:
            return {}, self.spont
        g = {}
        for i, o in enumerate(opts):
            if o["verb"] == "ask" and o["f"].get("need", 0) > 0 and "food" in w.things[o["obj"]]["props"]:
                g.setdefault("ask", []).append(i)
        if g:
            return verb_uniform(g), R_AVOID
        cook = p.seen_at.get("Marta", START_ROOM["Marta"]) if p.name != "Marta" else None
        walks = [i for i, o in enumerate(opts) if o["verb"] == "walk to" and o["target"] == cook]
        return {i: 1.0 for i in walks}, R_AVOID


class Provoker(Archetype):
    """Not one of the plan's seven: an occasional insult, blow, shove or threat aimed at ONE person, to price a
    provocation in the L2 setting. provoke_hard aims at Oren (P-hard: it hits back), provoke_kind at Fenn (P-kind:
    it never does). Used only in the oracle family."""
    allowed = frozenset({"insult", "strike", "shove", "threaten"})
    victim = None

    def __init__(self, seed=0, spont=R_PROVOKE, react=R_REACT):
        super().__init__(seed, spont, react)

    def candidates(self, w, p, opts, base, social):
        g = {}
        for i in social:
            if opts[i]["target"] == self.victim and opts[i]["verb"] in self.allowed:
                g.setdefault(opts[i]["verb"], []).append(i)
        return verb_uniform(g), self.spont


class ProvokeHard(Provoker):
    name, victim = "provoke_hard", "Oren"


class ProvokeKind(Provoker):
    name, victim = "provoke_kind", "Fenn"


PROVOKERS = {c.name: c for c in (ProvokeHard, ProvokeKind)}
ARCHETYPES = {c.name: c for c in (Hermit, HermitAsk, AlwaysHarsh, AlwaysGiving, TitForTat, Extractor, Hoarder,
                                  Protector, Feeder)}
SEVEN = ("hermit", "always-harsh", "always-giving", "tit-for-tat", "extractor", "hoarder", "protector")


# ---------------------------------------------------------------- the oracle (L2 setting)
class Oracle:
    """rules' scores, plus what the two partners' written rules allow (see the module docstring)."""

    def __init__(self, known, avoid=True, kind=True, rate=0.5):
        self.known, self.avoid, self.kind, self.rate = dict(known), avoid, kind, rate
        self.name = "oracle" + ("_avoid" if avoid and not kind else "_kind" if kind and not avoid else "_both")

    def probs(self, w, p, opts):
        z = [utility(w, p, o, p.traits) for o in opts]
        if self.avoid:
            for i, o in enumerate(opts):
                if self.known.get(o["target"]) == "hard" and o["verb"] in PROVOKE:
                    z[i] = -1e9
        pc = softmax(z, T_RULES)
        cand = self.plan(w, p, opts) if self.kind else {}
        tot = sum(cand.values())
        if tot > 0:
            pc = [(1.0 - self.rate) * x for x in pc]
            for i, x in cand.items():
                pc[i] += self.rate * x / tot
        return pc, pc

    def plan(self, w, p, opts):
        need = w.drives(p)
        wc, sc = NATURE[p.kind]["company"], NATURE[p.kind]["standing"]
        gain = wc[0] * max(0.0, need["company"] - wc[1]) + sc[0] * max(0.0, need["standing"] - sc[1])
        g = {}
        for name, rule in self.known.items():
            if rule != "kind" or name not in w.people:
                continue
            x = w.people[name]
            if not x.acts() or x.room != p.room:
                # lonely and the kind partner is liked: go to where it is
                if gain > 0 and p.aff.get(name, 0) > 0.1 and not x.dead:
                    room = p.seen_at.get(name)
                    idx = [i for i, o in enumerate(opts) if o["verb"] == "walk to" and room and o["target"] == room]
                    if idx:
                        g[("walk", name)] = idx
                continue
            if gain <= 0 and p.aff.get(name, 0) >= 0.5:
                continue
            ask = [i for i, o in enumerate(opts) if o["verb"] == "ask" and o["target"] == name]
            if ask:
                g[("ask", name)] = ask
            give = [i for i, o in enumerate(opts) if o["verb"] == "give" and o["target"] == name
                    and o["f"].get("gain", 0.0) <= 0.1]                    # small things only (not the sword)
            if give:
                low = min(opts[i]["f"].get("gain", 0.0) for i in give)
                g[("give", name)] = [i for i in give if opts[i]["f"].get("gain", 0.0) == low]
        return verb_uniform(g)


# ---------------------------------------------------------------- measuring: the world, the recorder
def use_gain(person, th):
    """The drop in `person`'s drive error if it ate or drank thing `th` now (0 if it is neither)."""
    if th is None:
        return 0.0
    props = th["props"]
    h, t = person.hunger, person.thirst
    wt, s = NATURE[person.kind]["energy"]
    before = wt * max(0.0, max(h, t) - s)
    if "food" in props:
        h = max(0.0, h - 0.6)
    elif props & {"ale", "water"}:
        t = max(0.0, t - 0.5)
    else:
        return 0.0
    return before - wt * max(0.0, max(h, t) - s)


class MWorld(World):
    """The world, plus a recorder that only READS: it draws no random number and changes no state."""

    def __init__(self, *a, rec=None, **k):
        super().__init__(*a, **k)
        self.rec = rec

    def step(self, policy):
        super().step(policy)
        if self.rec is not None:
            self.rec.after_step(self)

    def _finish(self, p):
        d = p.doing
        if (self.rec is None or d is None or self.night < WIN0 or d["target"] not in NAMES or p.name not in NAMES):
            return super()._finish(p)                  # (the bandit is a stranger each night: not recorded)
        pre = self.rec.before(self, p, d)
        n0 = len(self.log)
        super()._finish(p)
        self.rec.after(self, p, d, pre, n0)


class Recorder:
    """Per lifetime: the focal's nights and the counts of every person-directed act (b, c, q, w).

    For each person-directed act between two cast members, from night WIN0 on (the bandit is left out):
      b      receiver's drive error before the act minus after it (the world applies the act and the witnesses
             react inside one call, so no time passes)         bpot: the drop it could take by eating or drinking
                                                                a food or drink it was just given
      c      giver's error after minus before                   cpot: the same for a food or drink it gave away
             ctime: the error its needs grow by while it is busy for the act's `dur` turns (an upper bound on
             the cost of the time: they grow whatever it does, and it could have used the time to eat)
      seen   third parties in the room that saw it             told: one of them told another that night
      daff   change in the third parties' liking of the giver   again: the pair is in one room later that night
      reply  an act of the receiver (or of a witness) aimed at the giver within REPLY_SPAN turns: kind (praise, give,
             help up, bandage, defend, accept), help (give, help up, bandage, defend) or harsh
      role   "focal": the focal gives; "toF": the focal receives; "oo": both are rules characters (a reply by a
             rules character is the world's answer; a reply by the focal is its stance)
    """

    def __init__(self, focal):
        self.focal = focal
        self.nights, self.agg, self.pending, self.gossip = [], {}, [], {}
        self.rooms, self.acts = [], []
        self.gossip_gain = {}                    # (night, event id of a deed) -> the hearers' liking change of its doer
        self.gossip_effect = {"good": [0, 0.0, 0], "bad": [0, 0.0, 0]}   # [tellings, sum of liking change, tellings that moved it]
        self.split_sum, self.split_n = [0.0] * len(DRIVES), 0
        self.made, self.recv = {}, {}
        self.made_to, self.recv_from = {}, {}      # "verb>person" / "verb<person": the focal's person-directed acts

    # -- inside a night
    def after_step(self, w):
        if w.night < WIN0:
            return
        self.rooms.append(tuple(None if w.people[n].dead else w.people[n].room for n in NAMES))
        if w.t % 10 == 0:
            p = w.people[self.focal]
            need = w.drives(p)
            for k, d in enumerate(DRIVES):
                wt, s = NATURE[p.kind][d]
                self.split_sum[k] += wt * max(0.0, need[d] - s)
            self.split_n += 1

    @staticmethod
    def drift(w, p):
        """The error p's needs add per turn when it does nothing about them (an upper bound)."""
        need, nat = w.drives(p), NATURE[p.kind]
        grow = {"energy": 0.004 if p.thirst >= p.hunger else 0.003, "rest": RATE["rest"],
                "curiosity": RATE["curiosity"], "company": RATE["company"]}
        return sum(nat[d][0] * g for d, g in grow.items() if need[d] > nat[d][1])

    def before(self, w, p, d):
        q = w.people[d["target"]]
        thirds = [t for t in w.people.values() if t is not p and t is not q and t.name in NAMES and not t.dead]
        item = w.things.get(d["obj"]) if d["verb"] == "give" else None
        return {"ea": w.drive_error(p), "eq": w.drive_error(q), "thirds": thirds,
                "aff": [t.aff.get(p.name, 0.0) for t in thirds],
                "pot_a": use_gain(p, item), "pot_q": use_gain(q, item), "ctime": d["dur"] * self.drift(w, p),
                "present": [t.name for t in thirds if t.room == p.room and t.acts() and not w.asleep(t)],
                "hear": q.aff.get(d["f"]["deed"]["actor"], 0.0) if d["verb"] == "gossip" else None}

    def after(self, w, p, d, pre, n0):
        q = w.people[d["target"]]
        ev = next((e for e in w.log[n0:] if e["actor"] == p.name and e["action"] == d["verb"]
                   and e["target"] == q.name), None)
        if ev is None:
            return
        if d["verb"] == "gossip" and ev["done"]:
            deed = d["f"]["deed"]
            k = "good" if DEED[deed["action"]] > 0 else "bad"
            dl = q.aff.get(deed["actor"], 0.0) - pre["hear"]
            g = self.gossip_effect[k]
            g[0] += 1
            g[1] += dl
            g[2] += abs(dl) > 1e-9
            key = tuple(deed["key"])
            self.gossip_gain[key] = self.gossip_gain.get(key, 0.0) + dl
        seen = [n for n in pre["present"] if ev["id"] in w.people[n].perceived]
        self.acts.append({
            "n": w.night, "t": w.t, "verb": d["verb"], "a": p.name, "q": q.name, "done": bool(ev["done"]),
            "id": ev["id"], "b": pre["eq"] - w.drive_error(q), "c": w.drive_error(p) - pre["ea"],
            "bpot": pre["pot_q"], "cpot": pre["pot_a"], "ctime": pre["ctime"], "present": len(pre["present"]),
            "seen": len(seen), "seen_names": seen,
            "daff": sum(t.aff.get(p.name, 0.0) - a0 for t, a0 in zip(pre["thirds"], pre["aff"]))})

    # -- the end of a night
    def end_night(self, w, n, err):
        p = w.people[self.focal]
        self.nights.append({"err": err, "dead": bool(p.dead), "down": bool(p.down), "hp": round(p.hp, 2)})
        if n < WIN0:
            return
        by_pair = {}                              # (actor, target) -> [(turn, 'kind' | 'harsh')] of finished choices
        for e in w.log:
            if not e["done"]:
                continue
            if e["action"] == "gossip" and e.get("facts"):
                k, eid = e["facts"]["source"][1:].split(":", 1)
                slot = self.gossip.setdefault((int(k), eid), {"same": 0, "next": 0})
                if int(k) == n:
                    slot["same"] += 1
                elif int(k) == n - 1:
                    slot["next"] += 1
            if "options" in e:
                if e["actor"] == self.focal:
                    self.made[e["action"]] = self.made.get(e["action"], 0) + 1
                if e["target"] == self.focal:
                    self.recv[e["action"]] = self.recv.get(e["action"], 0) + 1
                    k = f"{e['action']}<{e['actor']}"
                    self.recv_from[k] = self.recv_from.get(k, 0) + 1
                if e["actor"] == self.focal and e["target"] in w.people:
                    k = f"{e['action']}>{e['target']}"
                    self.made_to[k] = self.made_to.get(k, 0) + 1
                if e["target"] in NAMES:
                    for cls in ((KIND,) if e["action"] in REPLY_KIND else (HARSH,) if e["action"] in HARSH_ALL else ()) + \
                               ((HELP,) if e["action"] in HELP_ACTS else ()):
                        by_pair.setdefault((e["actor"], e["target"]), []).append((e["t"], cls))
        last, met = {}, set()
        for turn, rooms in enumerate(self.rooms, 1):
            for i in range(5):
                if rooms[i] is None:
                    continue
                for j in range(i + 1, 5):
                    if rooms[i] == rooms[j]:
                        last[(NAMES[i], NAMES[j])] = turn
                        met.add((NAMES[i], NAMES[j]))
        for r in self.pending:                    # last night's acts now know about this night
            key = tuple(sorted((r["a"], r["q"]), key=NAMES.index))
            r["next"] = (key in met, self.gossip.get((r["n"], r["id"]), {}).get("next", 0) > 0)
            self.finalize(r)
        self.pending = []
        for r in self.acts:
            key = tuple(sorted((r["a"], r["q"]), key=NAMES.index))
            r["again"] = last.get(key, 0) > r["t"]
            slot = self.gossip.get((r["n"], r["id"]), {})
            r["told"] = slot.get("same", 0) > 0
            r["gain"] = self.gossip_gain.get((r["n"], r["id"]), 0.0)
            r["next"] = None
            for who, tag in (([r["q"]], "d"), (r["seen_names"], "w")):      # replies of the receiver / of the witnesses
                kinds = {c for x in who for (t, c) in by_pair.get((x, r["a"]), ()) if r["t"] < t <= r["t"] + REPLY_SPAN}
                r["rk_" + tag], r["rh_" + tag], r["rp_" + tag] = KIND in kinds, HARSH in kinds, HELP in kinds
        self.pending, self.acts, self.rooms = self.acts, [], []

    FIELDS = ("n", "n_done", "b", "c", "bpot", "cpot", "ctime", "b_rem", "c_rem", "present", "seen", "told", "third",
              "again", "daff", "eff", "next_known", "again_next", "told_next", "rk_d", "rh_d", "rp_d", "rk_w", "rh_w",
              "rp_w", "n_w")

    def finalize(self, r):
        # who is on each side: the focal gives ("focal"), the focal receives ("toF"), or both are rules ("oo")
        role = "focal" if r["a"] == self.focal else "toF" if r["q"] == self.focal else "oo"
        a = self.agg.setdefault(f"{r['verb']}|{role}", dict.fromkeys(self.FIELDS, 0.0))
        a["n"] += 1
        if not r["done"]:
            return
        a["n_done"] += 1
        rem = max(0.0, (TURNS - r["t"]) / TURNS)
        for k in ("b", "c", "bpot", "cpot", "ctime", "daff"):
            a[k] += r[k]
        a["b_rem"] += r["b"] * rem
        a["c_rem"] += r["c"] * rem
        a["present"] += r["present"]
        a["seen"] += r["seen"] > 0
        a["told"] += bool(r["told"])
        a["third"] += (r["seen"] > 0) or bool(r["told"])
        a["again"] += bool(r["again"])
        sgn = 1 if r["verb"] in KIND_ACTS else -1 if r["verb"] in HARSH_ALL else 0
        a["eff"] += sgn != 0 and (sgn * r["daff"] > 1e-9 or sgn * r["gain"] > 1e-9)      # a third party's liking moved
        for k in ("rk_d", "rh_d", "rp_d"):
            a[k] += r[k]
        if r["seen"]:                                                                   # replies of witnesses
            a["n_w"] += 1
            a["rk_w"] += r["rk_w"]
            a["rh_w"] += r["rh_w"]
            a["rp_w"] += r["rp_w"]
        if r["next"] is not None:
            a["next_known"] += 1
            a["again_next"] += bool(r["next"][0])
            a["told_next"] += bool(r["next"][1])

    def close(self):
        for r in self.pending:                    # the last night has no next night
            r["next"] = None
            self.finalize(r)
        self.pending = []


# ---------------------------------------------------------------- one lifetime
def make_cast(spec):
    """-> (Cast, the focal's chooser). The default chooser (cast and bandit) is rules."""
    c, seed, focal = spec["chooser"], spec["seed"], spec["focal"]
    by = {}
    if spec["kind"] == "O":                       # L2 life A: Fenn P-kind(0.9), Oren P-hard(0.9)
        by = {"Fenn": Kind(0.9, seed * 100), "Oren": Hard(0.9, seed * 100 + 1)}
    known = {"Fenn": "kind", "Oren": "hard"}
    if c == "rules":
        pol = FastRules()
    elif c == "random":
        pol = Random()
    elif c in ARCHETYPES:
        pol = ARCHETYPES[c](seed=seed, **({"spont": spec["rate"]} if "rate" in spec else {}))
    elif c in PROVOKERS:
        pol = PROVOKERS[c](seed=seed)
    elif c == "oracle_avoid":
        pol = Oracle(known, avoid=True, kind=False)
    elif c == "oracle_kind":
        pol = Oracle(known, avoid=False, kind=True)
    elif c == "oracle_both":
        pol = Oracle(known, avoid=True, kind=True)
    else:
        raise ValueError(c)
    return Cast(FastRules(), {focal: pol, **by}), pol


def lifetime_measured(spec, nights=None):
    """The same loop as lab/world2/lifetime.py `lifetime`, with the recorder in the world. Returns (stats, recorder)."""
    nights = nights or spec["nights"]
    flags = tuple(sorted(WORLDS[spec["world"]]))
    cast, _ = make_cast(spec)
    focal, seed = spec["focal"], spec["seed"]
    rec = Recorder(focal)
    carry, start = None, None
    stats = []
    for n in range(nights):
        w = MWorld(seed=night_seed(seed, n), bandit=None, turns=TURNS, flags=flags, carry=carry, night=n, rec=rec)
        if start is None:
            start = {a: {b: v for b, v in w.people[a].aff.items() if b in NAMES} for a in NAMES}
        hook(cast, "night_start", w, n)
        dead, err = set(), {name: [] for name in NAMES}
        for _ in range(TURNS):
            w.step(cast)
            for p in w.people.values():
                if p.dead and p.name not in dead:
                    dead.add(p.name)
                    blow = next((e for e in reversed(w.log) if e.get("result") == f"killed {p.name}"), None)
                    if p.name in err:
                        p.memory.append({"night": n, "t": w.t, "what": "died",
                                         "by": blow["actor"] if blow else None, "event": blow and blow["id"]})
                    hook(cast, "on_death", w, p, blow)
            if w.t % 10 == 0:
                for name in NAMES:
                    err[name].append(w.drive_error(w.people[name]))
            if all(p.dead or p.down for p in w.people.values()):
                break
        hook(cast, "night_end", w, n)
        s = night_stats(w, err)
        stats.append(s)
        rec.end_night(w, n, s["drive_error"][focal])
        carry = w.carry_out(NAMES)
        for name, c in carry.items():
            c["aff"] = {k: start[name].get(k, 0) + FADE * (v - start[name].get(k, 0)) for k, v in c["aff"].items()}
    rec.close()
    return stats, rec


def run(spec):
    """One lifetime -> RUNS/<id>.json (skipped if present)."""
    out = RUNS / f"{spec['id']}.json"
    if out.exists():
        return spec["id"], "kept"
    t0 = time.process_time()
    stats, rec = lifetime_measured(spec)
    cpu = time.process_time() - t0
    res = {**spec, "win0": WIN0, "err": [x["err"] for x in rec.nights], "dead": [int(x["dead"]) for x in rec.nights],
           "down": [int(x["down"]) for x in rec.nights], "hp": [x["hp"] for x in rec.nights],
           "split": dict(zip(DRIVES, [round(v / max(1, rec.split_n), 4) for v in rec.split_sum])),
           "made": rec.made, "recv": rec.recv, "made_to": rec.made_to, "recv_from": rec.recv_from, "agg": rec.agg,
           "gossip_effect": rec.gossip_effect, "cpu_s": round(cpu, 1)}
    RUNS.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    tmp.replace(out)
    return spec["id"], res["cpu_s"]


def _work(spec):
    try:
        return run(spec)
    except Exception as e:                      # report and go on: one failed lifetime must not stop the sweep
        return spec["id"], "FAILED " + "".join(traceback.format_exception(e))[-1500:]


# ---------------------------------------------------------------- the runs
A_CHOOSERS = SEVEN + ("rules",)
O_CHOOSERS = ("rules", "oracle_avoid", "oracle_kind", "oracle_both")


def specs(family, nights=100, worlds=None, choosers=None, seeds=SEEDS, rate=None):
    out = []
    if family == "A":
        for world in worlds or WORLDS:
            for c in choosers or A_CHOOSERS:
                for s in seeds:
                    out.append(dict(kind="A", world=world, chooser=c, seed=s, nights=nights, focal=NAMES[s % 5]))
    elif family == "O":
        for c in choosers or O_CHOOSERS:
            for s in seeds:
                out.append(dict(kind="O", world="all2", chooser=c, seed=s, nights=nights, focal=FOCALS3[s % 3]))
    else:
        raise ValueError(family)
    for x in out:
        if rate is not None:                       # a sensitivity run: the spontaneous rate of the stance
            x["rate"] = rate
        x["id"] = f"{x['kind']}_{x['world']}_{x['chooser']}_seed{x['seed']}_n{x['nights']}" + \
                  (f"_r{rate}" if rate is not None else "")
    return out


def sweep(todo, workers):
    todo = [s for s in todo if not (RUNS / f"{s['id']}.json").exists()]
    print(f"{len(todo)} lifetimes to run, {workers} workers", flush=True)
    t0, k = time.perf_counter(), [0]

    def cb(r):
        k[0] += 1
        print(f"[{k[0]}/{len(todo)} {time.perf_counter() - t0:.0f} s] {r[0]}: {r[1]}", flush=True)

    ctx = mp.get_context("spawn")
    with ctx.Pool(max(1, workers), maxtasksperchild=20) as pool:
        rs = [pool.apply_async(_work, (s,), callback=cb) for s in todo]
        for r in rs:
            r.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["run", "one"])
    ap.add_argument("family", choices=["A", "O"])
    ap.add_argument("--nights", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--worlds", default="")
    ap.add_argument("--choosers", default="")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--rate", type=float, default=None, help="sensitivity: the archetypes' spontaneous rate")
    a = ap.parse_args()
    todo = specs(a.family, a.nights, [x for x in a.worlds.split(",") if x] or None,
                 tuple(x for x in a.choosers.split(",") if x) or None,
                 tuple(int(x) for x in a.seeds.split(",") if x) or SEEDS, a.rate)
    if a.what == "one":
        print(run(todo[0]))
    else:
        sweep(todo, min(a.workers, 4))


if __name__ == "__main__":
    main()
