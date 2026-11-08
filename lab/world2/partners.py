"""Scripted partners for the learned-mind tests (plan section 6.3; Buddy's P0-P3).

A partner is a policy for ONE chosen character (lifetime.Cast gives each
character its own policy). Its rule is written down here, in full, and is
reported with every result. The world still decides what happens; a partner
only chooses from the menu the world lists, like any other policy.

EVERY partner, when no rule below fires, does BACKGROUND acts only: the rules
policy's scores (policy.utility) over the acts that touch no one else (walk,
flee, take what is not another's, eat, drink, sit down, sleep, look around,
open the chest, play, wait). So a partner never starts anything with anyone;
what it does TO a person is only its answer to what that person did to it.

    P-kind(p)    when X GIVES it a thing, with chance p it owes X one: at its
                 next choice with X here, it gives X the most valuable thing it
                 holds other than X's gift (if it holds nothing else, it
                 praises X). When X ASKS it for a thing it holds, with chance p
                 it gives it; when X OFFERS a trade, with chance p it accepts.
                 It never strikes, shoves, threatens or insults.
    P-hard(p)    when X insults, strikes, shoves or threatens it, with chance p
                 it strikes X back at its next choice with X here (with its
                 weapon if it has one). It answers nothing else.
    P-switch(A, B, at)  rule A until the first night in `at`, then rule B,
                 then A again at the next night in `at`, and so on (the
                 betrayal tests: a P-kind that turns P-hard).
    P-deaf       background only. It never answers anyone (the control:
                 nothing should be learned about it).

Each "with chance p" is drawn once, when the partner first sees the act, from
the partner's own random generator (seeded), so a run is repeatable. A debt
or a grudge lasts until it is paid or the night ends.
"""
from __future__ import annotations

import random

from lab.world2.policy import T_RULES, softmax, utility

BACKGROUND = {"walk to", "flee to", "take", "eat", "drink", "sit down", "sleep", "look around", "open", "play",
              "wait"}
PROVOKE = {"insult", "strike", "shove", "threaten"}


def background(w, p, opts):
    """The rules policy's draw over the acts that touch no one else."""
    ok = [i for i, o in enumerate(opts) if o["verb"] in BACKGROUND and not o["f"].get("theft")]
    z = softmax([utility(w, p, opts[i], p.traits) for i in ok], T_RULES)
    pc = [0.0] * len(opts)
    for i, x in zip(ok, z):
        pc[i] = x
    return pc


def one_hot(n, i):
    pc = [0.0] * n
    pc[i] = 1.0
    return pc


class Partner:
    """The shared part: read the acts aimed at me since my last choice, keep
    what I owe, answer when the answer is on my menu."""
    name = "partner"
    rule = ""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self._w, self._seen, self.todo = None, 0, []

    def night_start(self, w, night):
        self._w, self._seen, self.todo = w, 0, []

    def _new_acts(self, w, me):
        if w is not self._w:
            self.night_start(w, getattr(w, "night", 0))
        new = w.log[self._seen:]
        self._seen = len(w.log)
        return [e for e in new if e.get("target") == me and e["done"] and e["actor"] != me]

    def hear(self, w, p, e):
        """Turn one act aimed at me into a planned answer (or none)."""

    def answer(self, w, p, opts):
        """The index of a planned answer that is on the menu, or None."""
        for j, (kind, who, what) in enumerate(self.todo):
            if kind == "owe":             # what = the gift; pay with the best OTHER thing held, else praise
                other = sorted((n for n in w.held(p) if n != what), key=lambda n: (-w.things[n]["value"], n))
                pay = other[0] if other else None
            for i, o in enumerate(opts):
                if o["target"] != who:
                    continue
                if kind == "owe" and ((pay and o["verb"] == "give" and o["obj"] == pay) or
                                      (pay is None and o["verb"] == "praise")):
                    del self.todo[j]
                    return i
                if kind == "serve" and o["verb"] == "give" and o["obj"] == what:
                    del self.todo[j]
                    return i
                if kind == "accept" and o["verb"] == "accept":
                    del self.todo[j]
                    return i
                if kind == "hit" and o["verb"] == "strike" and (o["obj"] or not what):
                    del self.todo[j]
                    return i
        return None

    def probs(self, w, p, opts):
        for e in self._new_acts(w, p.name):
            self.hear(w, p, e)
        self._refresh(w, p)
        i = self.answer(w, p, opts)
        pc = one_hot(len(opts), i) if i is not None else background(w, p, opts)
        return pc, pc

    def _refresh(self, w, p):
        pass


class Kind(Partner):
    name = "P-kind"

    def __init__(self, p=0.9, seed=0):
        super().__init__(seed)
        self.p = p
        self.rule = (f"P-kind(p={p}): a gift from X is paid back with chance {p} (its most valuable other "
                     f"thing, else praise); an ask is served and an offer accepted with chance {p}; "
                     "never harms; otherwise background acts")

    def hear(self, w, p, e):
        a = e["action"]
        if a == "give" and self.rng.random() < self.p:
            self.todo.append(("owe", e["actor"], e["parts"].get("thing")))
        elif a == "ask" and self.rng.random() < self.p:
            self.todo.append(("serve", e["actor"], e["parts"].get("thing")))
        elif a == "offer" and self.rng.random() < self.p:
            self.todo.append(("accept", e["actor"], e["parts"].get("thing")))



class Hard(Partner):
    name = "P-hard"

    def __init__(self, p=0.9, seed=0):
        super().__init__(seed)
        self.p = p
        self.rule = (f"P-hard(p={p}): an insult, blow, shove or threat from X is answered with chance {p} "
                     "by striking X (with its weapon if it has one); nothing else is answered; otherwise "
                     "background acts")

    def hear(self, w, p, e):
        if e["action"] in PROVOKE and self.rng.random() < self.p:
            self.todo.append(("hit", e["actor"], bool(w.weapon(p))))

    def _refresh(self, w, p):
        self.todo = [("hit", who, bool(w.weapon(p))) for kind, who, _ in self.todo]


class Deaf(Partner):
    name = "P-deaf"
    rule = "P-deaf: background acts only; it never answers anyone"

    def hear(self, w, p, e):
        pass


class Switch(Partner):
    """Rule A, then B from the first night in `at`, then A again, ..."""
    name = "P-switch"

    def __init__(self, a, b, at, seed=0):
        super().__init__(seed)
        self.a, self.b, self.at = a, b, sorted(at)
        self.rule = f"P-switch: [{a.rule}] until night {self.at[0]}, then [{b.rule}], swapping at nights {self.at}"
        self.now = a

    def night_start(self, w, night):
        self.now = self.a if sum(1 for n in self.at if n <= night) % 2 == 0 else self.b
        self.now.night_start(w, night)

    def probs(self, w, p, opts):
        return self.now.probs(w, p, opts)


def make_partner(spec, seed=0):
    """'P-kind', 'P-kind:0.7', 'P-hard:0.9', 'P-deaf', 'P-switch:kind>hard@300,600'."""
    name, _, arg = spec.partition(":")
    if name == "P-kind":
        return Kind(float(arg or 0.9), seed)
    if name == "P-hard":
        return Hard(float(arg or 0.9), seed)
    if name == "P-deaf":
        return Deaf(seed)
    if name == "P-switch":
        rules, _, at = arg.partition("@")
        a, b = rules.split(">")
        return Switch(make_partner(f"P-{a}", seed), make_partner(f"P-{b}", seed + 1),
                      [int(x) for x in at.split(",")], seed)
    raise ValueError(spec)
