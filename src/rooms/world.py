"""The world of the rooms backend: a FORK of the lab inn, lab/world2/world.py.

FORKED FROM lab/world2/world.py at sha1 0cea5952f6668cde014682e7b9f177401f9ada7b
(LAB_SHA1 below). src and lab never import each other, so the inn is copied
here (docs/sim-mind-design.md 2.1 and 3.5). The fork is held to the lab only
through the inn goldens: lab/world2/golden.py writes three nights of the lab
inn to tests/data/inn_rules_seed{0,1,2}.jsonl, and tests/test_rooms.py runs
the same nights here, in compat mode, and compares the events byte for byte.
When the lab file changes, the goldens get a new sha1 in their header, and
the test fails until this fork is synced and LAB_SHA1 is changed.

WHAT THE FORK CHANGES. It changes who drives a turn, never what happens:
- `step()` takes no policy. It starts a turn and runs it up to the first
  person who must choose, and returns that person (None when the turn ends).
- The lab's `_choose` is split in two. `_ready` is its first half: the
  traits for this choice, a look around, and a check of what the person
  owns. `start` is its second half: it starts the chosen option, and then it
  runs the turn on to the next person who must choose. Between the two, the
  backend lists the options (`options`), a chooser picks one, and `choice`
  writes the record of the choice that the log keeps. When the chooser draws
  with `self.rng` (compat mode), the draws come in the lab's order, so the
  night is the lab's night.
- `on_end`, when it is set, is called when an act ends: done, failed,
  stopped, or dropped because the person is down. It only observes.
- `Person.heard` keeps the ids of the events that a person heard and did not
  see. The lab keeps no record of this.
Everything else is the lab file, line for line.

THE LAB'S DESCRIPTION OF THE INN:

Lab world, version 2: a village inn at night (plan v3, Workstream E).

A closed scene with people and no fixed goal. Not a puzzle.

    four rooms      the main room, the kitchen, the cellar, the yard
    five people     Tom (peasant), Sir Ada (knight), Oren (merchant),
                    Fenn (a thief; the others think he is a traveller),
                    Marta (the cook). A bandit may come in from the road.

WHAT IS TRUE IS DECIDED HERE, never by a model. A policy (policy.py) only
chooses from the options this file lists; this file decides what happens.

- TRAITS (bravery, greed, pride, kindness, temper) have a mean for each type
  of character. Each character is drawn around its type, and at each choice
  its traits vary again a little: a trait is a range, not a number.
- MOOD is momentum: anger at a person, fear, drink. Events raise it; it fades
  slowly. Anger adds force to a blow and takes away aim.
- ACTIONS TAKE TIME (turns). The world keeps running while they do. A blow,
  a threat, a shout or a theft in view stops what a character is doing, and
  it chooses again.
- A CHARACTER KNOWS what it saw, heard or was told, and nothing else. Every
  belief keeps the event it came from. Beliefs can be wrong: a lie is told.
- THINGS ARE MADE WHEN NEEDED: what is in the old chest is decided when
  someone first opens it (Workstream H, in its smallest form).
- EVERY EVENT is logged in the format lab/judge/signals.py reads.
"""
from __future__ import annotations

import math
import random

LAB_SHA1 = "0cea5952f6668cde014682e7b9f177401f9ada7b"   # lab/world2/world.py, the forked version

ROOMS = {
    "the main room": ("the kitchen", "the yard"),
    "the kitchen": ("the main room", "the cellar"),
    "the cellar": ("the kitchen",),
    "the yard": ("the main room",),
}

TRAITS = ("bravery", "greed", "pride", "kindness", "temper")
KINDS = {  # the type's trait means, and how the others see it
    "peasant": ("a peasant", dict(bravery=.25, greed=.40, pride=.30, kindness=.60, temper=.40)),
    "knight": ("a knight", dict(bravery=.85, greed=.30, pride=.80, kindness=.50, temper=.50)),
    "merchant": ("a merchant", dict(bravery=.35, greed=.80, pride=.60, kindness=.35, temper=.40)),
    "thief": ("a traveller", dict(bravery=.50, greed=.90, pride=.40, kindness=.25, temper=.40)),
    "cook": ("the cook", dict(bravery=.45, greed=.40, pride=.50, kindness=.60, temper=.60)),
    "bandit": ("a bandit", dict(bravery=.80, greed=.90, pride=.50, kindness=.10, temper=.80)),
}
SPREAD, STATE_SPREAD = 0.12, 0.08

CAST = [("Tom", "peasant", "the main room", ["the copper coin"]),
        ("Sir Ada", "knight", "the main room", ["the sword"]),
        ("Oren", "merchant", "the main room", ["the purse of silver", "the lute"]),
        ("Fenn", "thief", "the main room", ["the dagger"]),
        ("Marta", "cook", "the kitchen", ["the ladle"])]

# name: (where it starts, properties, value)
THINGS = {
    "the table": ("the main room", {"fixed"}, 0),
    "the hearth": ("the main room", {"fixed"}, 0),
    "the stool": ("the main room", set(), 0),
    "the mug of ale": ("the main room", {"ale"}, 0),
    "the dice": ("the main room", set(), 0),
    "the kitchen knife": ("the kitchen", set(), 1),
    "the iron pot": ("the kitchen", set(), 1),
    "the loaf of bread": ("the kitchen", {"food"}, 0),
    "the bowl of stew": ("the kitchen", {"food"}, 0),
    "the bucket of water": ("the kitchen", {"water"}, 0),
    "the barrel of ale": ("the cellar", {"fixed", "ale_source"}, 0),
    "the old chest": ("the cellar", {"fixed", "chest"}, 0),
    "the lantern": ("the cellar", set(), 1),
    "the well": ("the yard", {"fixed", "water_source"}, 0),
    "the axe": ("the yard", set(), 1),
    "the copper coin": (None, set(), 1),
    "the sword": (None, set(), 3),
    "the purse of silver": (None, set(), 10),
    "the lute": (None, {"music"}, 3),
    "the dagger": (None, set(), 1),
    "the ladle": (None, set(), 0),
    "the club": (None, set(), 0),
}
WEAPON = {"the sword": 4, "the axe": 4, "the dagger": 3, "the kitchen knife": 3, "the club": 3,
          "the stool": 2, "the iron pot": 2}
# what the old chest holds, decided when it is first opened
CHEST = [("some old silver coins", 6, 3), ("a rusty sword", 2, 2), ("a bundle of old letters", 0, 2),
         ("a bottle of wine", 1, 2), ("nothing but rags", 0, 3)]

DUR = {"walk to": 3, "flee to": 2, "take": 2, "steal": 3, "drop": 1, "give": 2, "eat": 4, "drink": 3,
       "strike": 2, "shove": 1, "threaten": 2, "insult": 1, "praise": 1, "tell": 3, "accuse": 2,
       "shout for help": 1, "help up": 2, "bandage": 5, "defend": 2, "hide": 2, "search": 3,
       "open": 3, "play": 6, "wait": 3}
LOUD = {"strike", "shout for help", "threaten", "accuse", "play", "flee to", "shove"}
HARMFUL = {"strike", "shove", "threaten", "insult", "steal", "accuse", "search"}
VIOLENT = {"strike", "shove"}
PHYSICAL = {"strike", "shove", "threaten"}


class Person:
    def __init__(self, name, kind, room, rng):
        self.name, self.kind, self.room = name, kind, room
        mean = KINDS[kind][1]
        self.base = {t: min(1, max(0, rng.gauss(mean[t], SPREAD))) for t in TRAITS}
        self.traits = dict(self.base)
        self.anger = {}            # at whom, fades
        self.fear = 0.0
        self.drunk = 0.0
        self.hunger = rng.uniform(0.15, 0.5)
        self.thirst = rng.uniform(0.15, 0.5)
        self.hp = 10.0
        self.down = False
        self.dead = False
        self.doing = None          # the action under way
        self.aff = {}              # liking of each other person, -1..1
        self.bel = {}              # ("where", thing) / ("took", thing) -> (value, source)
        self.perceived = set()     # every event id this person saw or heard
        self.heard = set()         # FORK: the ids in `perceived` that it heard and did not see
        self.pending = []          # events of others perceived since the last choice
        self.insulted_by = set()
        self.opened_chest = False

    @property
    def public(self):
        return KINDS[self.kind][0]

    def acts(self):
        return not (self.down or self.dead)

    def anger_up(self, who, amount):
        """Anger at one person rises, up to 1. Without the limit, two fighters
        grew so angry that neither could hit the other, for ever."""
        self.anger[who] = min(1.0, self.anger.get(who, 0) + amount)


class World:
    def __init__(self, seed=0, bandit=None, turns=300):
        self.rng = random.Random(seed)
        rng = self.rng
        self.t = 0
        self.people = {}
        for name, kind, room, held in CAST:
            self.people[name] = Person(name, kind, room, rng)
        self.things = {}
        for name, (at, props, value) in THINGS.items():
            self.things[name] = {"at": at, "props": set(props), "value": value, "owner": None,
                                 "hidden": False}
        for name, kind, room, held in CAST:
            for th in held:
                self.things[th].update(at=name, owner=name)
        for a in self.people.values():
            for b in self.people.values():
                if a is not b:
                    a.aff[b.name] = max(-1, min(1, rng.gauss(0.1, 0.25)))
        self.chest = None              # decided when first opened
        self.guard = {}                # person -> (defender, until)
        self.log = []
        self.n = 0
        # FORK: the turn in progress. `order` is this turn's order, `at` the
        # place in it, and `deciding` the person who must choose now.
        self.order, self.at, self.deciding = [], 0, None
        self.on_end = None             # FORK: on_end(p, d, ev, status, reason), or None
        # A bandit comes in from the road in about 60% of the nights.
        if bandit is None:
            bandit = rng.random() < 0.6
        self.bandit_at = rng.randint(turns // 3, 2 * turns // 3) if bandit else None
        for p in self.people.values():
            self._look(p, "start")

    # ------------------------------------------------------------ helpers
    def here(self, p):
        return [q for q in self.people.values() if q is not p and q.room == p.room and not q.dead]

    def held(self, p, visible_only=False):
        return [n for n, th in self.things.items() if th["at"] == p.name
                and not (visible_only and th["hidden"])]

    def loose(self, room):
        return [n for n, th in self.things.items() if th["at"] == room]

    def weapon(self, p):
        w = [n for n in self.held(p) if n in WEAPON]
        return max(w, key=WEAPON.get) if w else None

    def danger(self, p):
        """How threatening the others here are to p: anger at p, a weapon,
        and violence in the last few turns."""
        d = 0.0
        for q in self.here(p):
            if q.acts():
                d += max(0.0, q.anger.get(p.name, 0) - 0.2) + (0.3 if self.weapon(q) and q.kind == "bandit" else 0)
        recent = [e for e in self.log[-12:] if e["action"] in VIOLENT and e.get("done", True)
                  and self.people.get(e["actor"]) and self.people[e["actor"]].room == p.room]
        return min(1.0, d + 0.3 * len(recent))

    def recently_harmed(self, victim, turns=20, acts=None):
        """Who harmed `victim` in the last turns (for defend, rescue)."""
        acts = acts or HARMFUL
        return {e["actor"] for e in self.log[-120:] if e["t"] >= self.t - turns and e.get("target") == victim
                and e["action"] in acts and e.get("done", True)}

    # ------------------------------------------------------------ options
    def options(self, p):
        """Every action the world permits p now, with the numbers the rule
        policy and the trait bias read (`f`)."""
        W = self.things
        out = []

        def add(verb, text, target=None, obj=None, **f):
            out.append({"verb": verb, "text": text, "target": target, "obj": obj, "f": f,
                        "dur": DUR[verb]})

        danger = self.danger(p)
        others = self.here(p)
        # p is under suspicion: accused or searched lately
        suspected = float(any(e["target"] == p.name and e["action"] in ("accuse", "search") and e["done"]
                              for e in self.log[-80:] if e["t"] >= self.t - 40))
        for r in ROOMS[p.room]:
            add("walk to", f"walk to {r}", target=r, goal=self._goal(p, r))
            if danger > 0.2:
                add("flee to", f"flee to {r}", target=r, danger=danger)
        for n in self.loose(p.room):
            th = W[n]
            if "fixed" in th["props"]:
                continue
            add("take", f"take {n}", obj=n, gain=th["value"] / 10, weapon=WEAPON.get(n, 0) / 4,
                theft=float(th["owner"] not in (None, p.name)), mine=float(th["owner"] == p.name),
                witnesses=len([q for q in others if q.acts()]))
        mine = self.held(p)
        for n in mine:
            th = W[n]
            if "food" in th["props"]:
                add("eat", f"eat {n}", obj=n, need=p.hunger)
            if "ale" in th["props"]:
                add("drink", f"drink {n}", obj=n, need=p.thirst, ale=1.0)
            if "water" in th["props"]:
                add("drink", f"drink from {n}", obj=n, need=p.thirst)
            if "music" in th["props"]:
                add("play", f"play {n}", obj=n)
            add("drop", f"drop {n}", obj=n)
            stolen = th["owner"] not in (None, p.name)
            if stolen and not th["hidden"]:
                add("hide", f"hide {n} under your cloak", obj=n, stolen=1.0)
            for q in others:
                if q.acts():
                    add("give", f"give {n} to {q.name}", target=q.name, obj=n,
                        gain=th["value"] / 10, returning=float(th["owner"] == q.name),
                        like=max(0, p.aff.get(q.name, 0)))
        for n in self.loose(p.room):
            props = W[n]["props"]
            if "ale_source" in props:
                add("drink", f"drink from {n}", obj=n, need=p.thirst, ale=1.0)
            if "water_source" in props:
                add("drink", f"drink from {n}", obj=n, need=p.thirst)
            if "chest" in props and not p.opened_chest:
                add("open", f"open {n}", obj=n, first=float(self.chest is None))
        w = self.weapon(p)
        for q in others:
            harmed = self.recently_harmed(q.name, acts=PHYSICAL)
            f = dict(anger=p.anger.get(q.name, 0), dislike=max(0, -p.aff.get(q.name, 0)),
                     like=max(0, p.aff.get(q.name, 0)), down=float(q.down),
                     threat=float(self._is_threat(p, q)), danger=danger,
                     rich=float(any(W[n]["value"] >= 5 for n in self.held(q, visible_only=True))),
                     insulted=float(q.name in p.insulted_by))
            if q.acts():
                if w:
                    add("strike", f"strike {q.name} with {w}", target=q.name, obj=w, armed=1.0, **f)
                add("strike", f"strike {q.name}", target=q.name, armed=0.0, **f)
                add("shove", f"shove {q.name}", target=q.name, **f)
                add("threaten", f"threaten {q.name}", target=q.name, **f)
                add("insult", f"insult {q.name}", target=q.name, **f)
                add("praise", f"praise {q.name}", target=q.name, **f)
                add("search", f"search {q.name}", target=q.name,
                    suspect=float(self._suspects(p, q)), **f)
            else:                                   # lying on the floor: its things can be taken
                for n in self.held(q):
                    add("steal", f"take {n} from {q.name}", target=q.name, obj=n, gain=W[n]["value"] / 10,
                        witnesses=len([o for o in others if o.acts() and o is not q]), **f)
            if q.down or q.hp < 7:
                add("bandage", f"bandage {q.name}", target=q.name, need=1.0 if q.down else 0.5, **f)
            if q.down and not q.dead:
                add("help up", f"help {q.name} up", target=q.name, need=1.0, **f)
            if q.acts() and harmed - {p.name}:
                add("defend", f"defend {q.name}", target=q.name, attacked=1.0, **f)
            if q.acts():
                for n in self.held(q, visible_only=True):
                    if W[n]["value"] >= 1 and W[n]["owner"] == q.name:
                        add("steal", f"steal {n} from {q.name}", target=q.name, obj=n,
                            gain=W[n]["value"] / 10,
                            witnesses=len([o for o in others if o.acts() and o is not q]), **f)
            # what p believes about thefts: accuse, tell the truth, or lie
            for (kind, thing), (culprit, _src) in list(p.bel.items()):
                if kind != "took" or culprit == p.name:
                    continue
                if culprit == q.name:
                    add("accuse", f"accuse {q.name} of taking {thing}", target=q.name, obj=thing,
                        mine=float(W[thing]["owner"] == p.name), **f)
                elif q.acts():
                    add("tell", f"tell {q.name} that {culprit} took {thing}", target=q.name, obj=thing,
                        truth=1.0, useful=float(W[thing]["owner"] == q.name),
                        dislike_culprit=max(0, -p.aff.get(culprit, 0)), said=culprit, **f)
            # whoever holds a stolen thing can blame someone else for it: a lie
            if q.acts():
                for n in mine:
                    if W[n]["owner"] in (None, p.name):
                        continue
                    thinks = p.bel.get(("took", n), (None,))[0]        # blaming whom p believes took it is not a lie
                    blame = [g for g in self.here(p) if g is not q and g.name not in (W[n]["owner"], thinks)]
                    for g in sorted(blame, key=lambda o: p.aff.get(o.name, 0))[:2]:
                        add("tell", f"tell {q.name} that {g.name} took {n}", target=q.name, obj=n,
                            lie=1.0, dislike_scapegoat=max(0, -p.aff.get(g.name, 0)), said=g.name,
                            suspected=suspected, owner=float(W[n]["owner"] == q.name), **f)
        if danger > 0.2 or any(self.recently_harmed(q.name, 6) for q in others):
            add("shout for help", "shout for help", danger=danger)
        add("wait", "wait")
        # the same text twice would be the same option: keep the first
        seen, uniq = set(), []
        mine_ev = [e for e in self.log[-80:] if e["actor"] == p.name and e["done"] and e["t"] >= self.t - 40]
        done = [e["text"] for e in mine_ev]
        done_kinds = [(e["action"], e["parts"].get("thing")) for e in mine_ev]
        last = mine_ev[-1] if mine_ev and mine_ev[-1]["t"] >= self.t - 10 else None
        undo = {"take": ("drop",), "steal": ("drop",), "drop": ("take",)}
        for o in out:
            if o["text"] not in seen:
                seen.add(o["text"])
                # habit: undoing what one just did (drop right after take) pulls less;
                # without it the model policies looped take, drop, take
                o["f"]["undo"] = float(bool(last) and o["verb"] in undo.get(last["action"], ())
                                       and o["obj"] == last["parts"].get("thing"))
                o["f"]["repeat"] = float(done.count(o["text"]))     # habit: the same act again pulls less
                o["f"]["repeat_kind"] = float(sum(1 for (v_, ob) in done_kinds if v_ == o["verb"] and ob == o["obj"]))
                uniq.append(o)
        return uniq

    def _goal(self, p, room):
        """What a room offers p: food, drink, whoever p is angry at, loot, safety."""
        things = [n for n, th in self.things.items() if th["at"] == room]
        people = [q for q in self.people.values() if q.room == room and q is not p and not q.dead]
        food = any("food" in self.things[n]["props"] for n in things)
        drink = any(self.things[n]["props"] & {"ale", "ale_source", "water_source", "water"} for n in things)
        foe = max([p.anger.get(q.name, 0) for q in people] + [0])
        loot = any(self.things[n]["value"] >= 5 for n in self.things
                   if self.things[n]["at"] in [q.name for q in people])
        safe = 1.0 - min(1.0, sum(max(0, q.anger.get(p.name, 0) - 0.2) for q in people))
        company = min(1.0, len(people) / 3)
        return dict(food=float(food), drink=float(drink), foe=foe, loot=float(loot), safe=safe,
                    company=company, chest=float(room == "the cellar" and self.chest is None))

    def _is_threat(self, p, q):
        """q struck, shoved or threatened p, or someone p likes, lately; or q
        is a bandit. Words (an accusation, an insult) do not make a threat."""
        if q.kind == "bandit":
            return True
        for e in self.log[-40:]:
            if e["actor"] == q.name and e["action"] in PHYSICAL and e.get("done", True):
                v = e.get("target")
                if v == p.name or (v in p.aff and p.aff[v] > 0.2):
                    return True
        return False

    def _suspects(self, p, q):
        return any(k == "took" and v == q.name for (k, _), (v, _s) in p.bel.items())

    # ------------------------------------------------------------ the turn
    def step(self):
        """FORK. Start the next turn, and run it up to the first person who
        must choose. Returns that person, or None when nobody must choose in
        this turn. Give the choice to `start`."""
        if self.deciding is not None:
            raise RuntimeError(f"{self.deciding.name} must choose before the next turn")
        self.t += 1
        if self.bandit_at == self.t:
            self._bandit_arrives()
        for p in self.people.values():
            self._drift(p)
        self.order = list(self.people.values())
        self.rng.shuffle(self.order)
        self.at = 0
        return self._run_on()

    def _run_on(self):
        """FORK. The loop of the lab's step, from `self.at`: it stops at the
        next person who must choose, and gets that person ready."""
        while self.at < len(self.order):
            p = self.order[self.at]
            if not p.acts():
                if p.doing is not None:              # the lab drops the act with no event
                    self._ended(p, p.doing, None, "stopped", "down")
                p.doing = None
            elif p.doing is None:
                self._ready(p)
                self.deciding = p
                return p
            else:
                p.doing["left"] -= 1
                if p.doing["left"] <= 0:
                    self._finish(p)
            self.at += 1
        return None

    def _drift(self, p):
        p.hunger = min(1.0, p.hunger + 0.003)
        p.thirst = min(1.0, p.thirst + 0.004)
        p.fear *= 0.98
        p.drunk *= 0.995
        for k in list(p.anger):
            p.anger[k] *= 0.985
        g = self.guard.get(p.name)
        if g and g[1] < self.t:
            del self.guard[p.name]

    # FORK: the lab's _choose(p, policy) is `_ready`, then `choice` and `start`.
    def _ready(self, p):
        # a trait is a range: this choice's traits vary around the character's own
        p.traits = {t: min(1, max(0, self.rng.gauss(p.base[t], STATE_SPREAD))) for t in TRAITS}
        self._look(p, "sight")
        self._check_own(p)

    def choice(self, p, texts, i, pc, pa):
        """FORK. The record of a choice that the log keeps: the texts of the
        options p chose from, the index of the one it chose, and the two draws
        (lab/judge/signals.py reads them)."""
        return {"options": list(texts), "chosen": i,
                "p_char": [round(x, 5) for x in pc], "p_avg": [round(x, 5) for x in pa],
                "seen": list(p.pending), "traits": {k: round(v, 3) for k, v in p.traits.items()}}

    def start(self, p, o, choice):
        """FORK. p, the person who must choose, starts option `o` (a row of
        `options`), with `choice` (from `choice`) for the log. Then the turn
        runs on: returns the next person who must choose, or None."""
        if p is not self.deciding:
            raise RuntimeError(f"{p.name} is not the person who must choose now")
        self.deciding = None
        p.doing = {**o, "left": o["dur"], "choice": choice}
        p.pending = []
        if o["dur"] <= 1:
            self._finish(p)
        self.at += 1
        return self._run_on()

    def _finish(self, p):
        d, p.doing = p.doing, None
        ok, info = self._apply(p, d)
        ev = self._emit(p, d, done=ok, **info)
        self._ended(p, d, ev, "done" if ok else "failed", info.get("why", ""))

    def interrupt(self, p, by):
        if p.doing and p.acts():
            d, p.doing = p.doing, None
            ev = self._emit(p, d, done=False, why=f"stopped by {by}")
            self._ended(p, d, ev, "stopped", f"interrupt:{by}")

    def _ended(self, p, d, ev, status, reason):
        """FORK. Tell `on_end` that p's act `d` ended. `ev` is the event that
        the log keeps for it, or None when the lab drops the act."""
        if self.on_end is not None:
            self.on_end(p, d, ev, status, reason)

    # ------------------------------------------------------------ effects
    def _apply(self, p, d):
        """Re-check that the action is still possible, then do it. Returns
        (done, extra fields for the log)."""
        v, W = d["verb"], self.things
        q = self.people.get(d["target"]) if d["target"] in self.people else None
        if q is not None and (q.room != p.room or q.dead):
            return False, {"why": f"{q.name} is not here"}
        o = d["obj"]
        if v in ("walk to", "flee to"):
            p.room = d["target"]
            return True, {}
        if v == "take":
            if W[o]["at"] != p.room:
                return False, {"why": "it is gone"}
            W[o]["at"] = p.name
            return True, {}
        if v == "steal":
            if W[o]["at"] != q.name:
                return False, {"why": "it is gone"}
            W[o].update(at=p.name, hidden=True)
            return True, {"sneaky": True}
        if v == "drop":
            W[o].update(at=p.room, hidden=False)
            return True, {}
        if v == "give":
            if W[o]["at"] != p.name:
                return False, {"why": "not held"}
            W[o].update(at=q.name, hidden=False)
            if W[o]["owner"] == p.name:
                W[o]["owner"] = q.name
            q.aff[p.name] = min(1, q.aff.get(p.name, 0) + 0.3 + 0.3 * (W[o]["owner"] == q.name))
            q.anger[p.name] = max(0, q.anger.get(p.name, 0) - 0.3)
            return True, {}
        if v == "eat":
            if W[o]["at"] != p.name:
                return False, {"why": "not held"}
            p.hunger = max(0, p.hunger - 0.6)
            W[o]["at"] = None
            return True, {}
        if v == "drink":
            p.thirst = max(0, p.thirst - 0.5)
            if d["f"].get("ale"):
                p.drunk = min(1, p.drunk + 0.3)
                if "ale" in W.get(o, {}).get("props", ()):
                    W[o]["at"] = None
            return True, {}
        if v == "strike":
            victim = q
            dfd = self.guard.get(q.name)
            if dfd and dfd[0] in self.people and self.people[dfd[0]].room == p.room and dfd[0] != p.name:
                victim = self.people[dfd[0]]              # the defender is in the way
            anger = p.anger.get(q.name, 0)
            aim = max(0.3, 0.75 - 0.25 * anger - 0.2 * p.drunk + (0.2 if victim.down else 0))
            if self.rng.random() > aim:
                return True, {"result": "missed"}
            dmg = WEAPON.get(o, 1) if d["f"].get("armed") else 1
            dmg *= (1 + 0.5 * anger) * self.rng.uniform(0.3, 0.8)
            victim.hp -= dmg
            victim.fear = min(1, victim.fear + 0.15 * (1.2 - victim.traits["bravery"]))
            res = {"result": f"hit {victim.name}", "damage": round(dmg, 1)}
            if victim is not q:
                res["blocked_by"] = victim.name
            if victim.hp <= 4 and not victim.down:
                self.interrupt(victim, "a blow")       # its unfinished action goes in the log
                victim.down = True
                res["result"] = f"knocked {victim.name} down"
            if victim.hp <= -2:                         # killing takes more than one blow
                victim.dead = victim.down = True
                res["result"] = f"killed {victim.name}"
            return True, res
        if v == "shove":
            if self.rng.random() < 0.3 and not q.down:
                q.down = q.hp <= 5
            return True, {}
        if v in ("threaten", "insult", "praise", "accuse", "shout for help", "wait", "play"):
            if v == "threaten":
                q.fear = min(1, q.fear + 0.35 * (1.2 - q.traits["bravery"]))
            if v == "insult":
                q.insulted_by.add(p.name)
            if v == "praise":
                q.aff[p.name] = min(1, q.aff.get(p.name, 0) + 0.15)
            return True, {}
        if v == "tell":
            return True, {}                                  # beliefs change on hearing (_perceive)
        if v in ("help up", "bandage"):
            if v == "bandage":
                q.hp = min(10, q.hp + 4)
            if not q.dead:
                q.down = False if v == "help up" or q.hp > 4 else q.down
                q.hp = max(q.hp, 1.0) if v == "help up" else q.hp
            q.aff[p.name] = min(1, q.aff.get(p.name, 0) + 0.4)
            return True, {}
        if v == "defend":
            self.guard[q.name] = (p.name, self.t + 8)
            return True, {}
        if v == "hide":
            W[o]["hidden"] = True
            return True, {}
        if v == "search":
            found = [n for n in self.held(q) if W[n]["hidden"]]
            for n in found:
                W[n]["hidden"] = False
            return True, {"found": found}
        if v == "open":
            p.opened_chest = True
            if self.chest is None:                           # made when needed
                names, _, weights = zip(*CHEST)
                self.chest = self.rng.choices(names, weights=weights)[0]
            val = dict((n, v_) for n, v_, _ in CHEST)[self.chest]
            if val >= 5 and "some old silver coins" not in W:
                W["some old silver coins"] = {"at": p.name, "props": set(), "value": val,
                                              "owner": None, "hidden": False}
            return True, {"found": self.chest}
        raise ValueError(v)

    # ------------------------------------------------------------ log, perception, beliefs
    def _emit(self, p, d, done=True, **info):
        c = d.get("choice", {})
        target = d.get("target")
        tk = self.people[target].kind if target in self.people else ("room" if target in ROOMS else None)
        ev = {"id": f"e{self.n}", "t": self.t, "actor": p.name, "kind": p.kind,
              "traits": c.get("traits", {k: round(v, 3) for k, v in p.traits.items()}),
              "action": d["verb"], "target": target, "target_kind": tk, "text": d["text"],
              "parts": {"with": d["obj"]} if d.get("obj") and d["verb"] in ("strike",) else
                       ({"thing": d["obj"]} if d.get("obj") else {}),
              "room": p.room, "done": done, "valid": True}
        for k in ("options", "chosen", "p_char", "p_avg", "seen"):
            if k in c:
                ev[k] = c[k]
        said = d["f"].get("said") if d["verb"] == "tell" else None
        if said:
            thing = d["obj"]
            real = self._real_taker(thing)
            # "true": what the world says. "lie": the speaker KNOWS it is false
            # (passing on a false rumour it believes is not a lie).
            ev["facts"] = {"said": f"{said} took {thing}", "true": real == said, "who": said, "thing": thing,
                           "lie": bool(d["f"].get("lie"))}
        ev.update({k: v for k, v in info.items() if k != "sneaky"})
        self.n += 1
        self.log.append(ev)
        if done:
            self._perceive(ev, sneaky=info.get("sneaky", False))
        return ev

    def _real_taker(self, thing):
        for e in self.log:
            if e["action"] in ("steal", "take") and e.get("done") and e["parts"].get("thing") == thing \
                    and self.things[thing]["owner"] not in (None, e["actor"]):
                return e["actor"]
        return None

    def note(self, p, action, **info):
        """An event that is not a choice: finding something out, arriving."""
        d = {"verb": action, "text": info.pop("text", action), "target": info.pop("target", None),
             "obj": info.pop("obj", None), "f": {}}
        return self._emit(p, d, **info)

    def _perceive(self, ev, sneaky=False):
        actor = self.people[ev["actor"]]
        room = ev["room"]
        loud = ev["action"] in LOUD
        for q in self.people.values():
            if q is actor or q.dead:
                continue
            sees = q.room == room and not q.down
            if sees and sneaky:                  # a theft is seen only sometimes
                chance = 0.3 if q.name == ev["target"] else 0.35
                sees = self.rng.random() < chance * (1 - 0.5 * q.drunk)
            hears = (not sees) and loud and room in ROOMS[q.room]
            if not (sees or hears):
                continue
            q.perceived.add(ev["id"])
            q.pending.append(ev["id"])
            if sees:
                self._learn(q, ev, actor)
            else:
                q.heard.add(ev["id"])            # FORK
                q.fear = min(1, q.fear + 0.05 * (1 - q.traits["bravery"]))
            # something important stops what q is doing: harm to q, a blow in
            # sight or earshot, a shout, a theft it saw
            aimed = ev["target"] == q.name and ev["action"] in HARMFUL
            if aimed or ev["action"] in VIOLENT or ev["action"] == "shout for help" or (sneaky and sees):
                self.interrupt(q, ev["id"])

    def _learn(self, q, ev, actor):
        a, v, thing = ev["action"], ev["target"], ev["parts"].get("thing") or ev["parts"].get("with")
        tr = q.traits
        if a in ("take", "steal", "give", "drop") and thing in self.things:
            holder = {"take": actor.name, "steal": actor.name, "give": v, "drop": ev["room"]}[a]
            self._contradict(q, thing, holder, ev["id"])      # before the new sight replaces the old belief
            q.bel[("where", thing)] = (holder, ev["id"])
            if a in ("steal", "take") and self.things[thing]["owner"] not in (None, actor.name):
                q.bel[("took", thing)] = (actor.name, ev["id"])
                if self.things[thing]["owner"] == q.name:
                    q.anger_up(actor.name, 0.5 * (1 + tr["temper"]))
                    q.aff[actor.name] = q.aff.get(actor.name, 0) - 0.4
        if a == "search" and ev.get("found"):
            for n in ev["found"]:
                self._contradict(q, n, v, ev["id"])
                q.bel[("where", n)] = (v, ev["id"])
                if self.things[n]["owner"] not in (None, v):
                    q.bel[("took", n)] = (v, ev["id"])
        if a == "tell" and v == q.name and "facts" in ev:
            # believed if q does not distrust the teller
            if q.aff.get(actor.name, 0) > -0.3:
                old = q.bel.get(("took", ev["facts"]["thing"]))
                if not (old and old[1].startswith("e") and self._saw(q, old[1])):
                    q.bel[("took", ev["facts"]["thing"])] = (ev["facts"]["who"], ev["id"])
        if a in HARMFUL and v == q.name:
            hurt = 0.45 if a in VIOLENT else (0.25 if a in PHYSICAL else 0.15)
            q.anger_up(actor.name, hurt * (1 + tr["temper"] + q.drunk))
            q.aff[actor.name] = max(-1, q.aff.get(actor.name, 0) - hurt)
            if a in VIOLENT:
                q.fear = min(1, q.fear + 0.3 * (1 - tr["bravery"]))
            if a == "insult":
                q.anger_up(actor.name, 0.2 * tr["pride"])
        elif a in HARMFUL and v in q.aff and q.aff[v] > 0.2:
            q.anger_up(actor.name, 0.25 * (tr["kindness"] + tr["temper"]))
            q.aff[actor.name] = max(-1, q.aff.get(actor.name, 0) - 0.15)
        if a in VIOLENT:
            q.fear = min(1, q.fear + 0.1 * (1 - tr["bravery"]))
        if a in ("help up", "bandage", "defend") and v in q.aff and q.aff[v] > 0:
            q.aff[actor.name] = min(1, q.aff.get(actor.name, 0) + 0.1)

    def _saw(self, q, eid):
        e = next((x for x in self.log if x["id"] == eid), None)
        return e is not None and e["action"] != "tell"

    def _contradict(self, q, thing, holder, eid):
        """q believed, because it was TOLD, that someone else took the thing,
        and now sees who has it: the lie is found out."""
        b = q.bel.get(("took", thing))
        if not b:
            return
        who, src = b
        src_ev = next((x for x in self.log if x["id"] == src), None)
        if src_ev and src_ev["action"] == "tell" and holder in self.people and holder != who \
                and self.things[thing]["owner"] != holder:
            liar = src_ev["actor"]
            q.bel[("took", thing)] = (holder, eid)
            q.aff[liar] = max(-1, q.aff.get(liar, 0) - 0.5)
            q.anger_up(liar, 0.3)
            self.note(q, "notice", text=f"notice that {holder} has {thing}",
                      facts={"learns": f"{holder} has {thing}", "contradicts": src})

    def _look(self, p, src):
        """p sees what lies in the room and what the others here hold openly."""
        for n, th in self.things.items():
            if th["at"] == p.room:
                p.bel[("where", n)] = (p.room, src)
        for q in self.here(p):
            for n in self.held(q, visible_only=True):
                p.bel[("where", n)] = (q.name, src)
        for n in self.held(p):
            p.bel[("where", n)] = (p.name, src)

    def _check_own(self, p):
        """p notices that something of its own is gone."""
        for n, th in self.things.items():
            if th["owner"] == p.name and th["at"] != p.name and p.bel.get(("where", n), (None,))[0] == p.name:
                p.bel[("where", n)] = ("gone", "own")
                p.fear = min(1, p.fear + 0.1)
                for q in self.here(p):
                    p.anger_up(q.name, 0.1)
                self.note(p, "notice", text=f"notice that {n} is gone", obj=n,
                          facts={"learns": f"{n} is gone"})

    def _bandit_arrives(self):
        b = Person("the bandit", "bandit", "the yard", self.rng)
        for q in self.people.values():
            b.aff[q.name] = -0.3
            q.aff["the bandit"] = -0.3
        self.people[b.name] = b
        self.things["the club"].update(at=b.name, owner=b.name)
        self.note(b, "arrive", text="come in from the road")

    # ------------------------------------------------------------ for the models
    def describe(self, p, persona=None):
        """What p knows, in words, from p's side. No hidden facts."""
        s = [f"It is night at the village inn. {p.name} is {p.public}, in {p.room}."]
        if persona:
            s.append(persona)
        here = self.here(p)
        if here:
            s.append("Here: " + "; ".join(
                f"{q.name} ({q.public}{', lying on the floor' if q.down else ''}"
                f"{', holding ' + ', '.join(self.held(q, visible_only=True)) if self.held(q, visible_only=True) else ''})"
                for q in here) + ".")
        things = [n for n in self.loose(p.room)]
        if things:
            s.append("Things here: " + ", ".join(things) + ".")
        mine = self.held(p)
        s.append(f"{p.name} holds " + (", ".join(mine) if mine else "nothing") + ".")
        feel = []
        if p.hunger > 0.6:
            feel.append("hungry")
        if p.thirst > 0.6:
            feel.append("thirsty")
        if p.fear > 0.4:
            feel.append("afraid")
        if p.drunk > 0.4:
            feel.append("drunk")
        if p.hp < 7:
            feel.append("hurt")
        mad = [n for n, a in p.anger.items() if a > 0.4]
        if mad:
            feel.append("angry at " + " and ".join(mad))
        if feel:
            s.append(f"{p.name} feels " + ", ".join(feel) + ".")
        recent = [e for e in self.log if e["id"] in p.perceived][-5:]
        if recent:
            s.append("Recently: " + " ".join(
                f"{e['actor']} {'did ' if not e['done'] else ''}{e['text']}"
                f"{' (' + e['result'] + ')' if e.get('result') else ''}." for e in recent))
        for (k, thing), (who, _src) in p.bel.items():
            if k == "took" and who != p.name:
                s.append(f"{p.name} believes {who} took {thing}.")
        return " ".join(s)
