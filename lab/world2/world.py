"""Lab world, version 2: a village inn at night (plan v3, Workstream E).

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

FLAGS (learned-mind plan, section 6). Each is OFF by default; with none on,
the inn is exactly what it was (lab/world2/golden.py, and the digests in
lab/results/mindL/world_flags_off.json). A flag adds facts and menu acts to
the world; it never chooses for anyone.

    drives   seven needs, each a level in 0..1 (0 = met) with a set point and a
             weight per type (NATURE) and a drift per turn (RATE): energy
             (hunger, thirst), safety (fear, hp), company, standing, wealth,
             rest (fatigue) and curiosity (want of the new). New acts: sit
             down, sleep, look around. A sleeper sees nothing, hears what is
             loud in its own room and wakes on it.
    gossip   a person who SAW a deed (a harm, a kindness, a lie found out)
             may tell another of it; a hearer who does not distrust the
             teller changes its liking of the doer (reputation).
    trade    the cook holds the food and serves it. Anyone may ask another
             for a thing, or offer a thing of its own for it; the one asked
             answers by giving, accepting, or refusing (its own choice).
    watch    a theft by someone the watcher distrusts is seen more often.
    check    lies are found out more often: seeing the thing held openly, or
             being hidden, by another than the one named; searching the
             named culprit and finding nothing; a tale that contradicts what
             the hearer saw itself.
    known    the menu's numbers (and the menu itself) come only from what the
             chooser knows. Without it, some option features read hidden
             facts: `danger` sums the OTHERS' private anger, a room's `goal`
             counts the food, people and anger in a room the chooser cannot
             see, `first` on the chest says whether anyone opened it, and the
             flee and shout options appear on that hidden danger. With it:
             danger is the hostility the chooser PERCEIVED (harsh acts aimed
             at it, fading as anger fades; kind acts lower it) plus a visibly
             armed bandit and violence it perceived; a room's goal comes from
             where it last saw each thing and person (else the inn's layout at
             the start of the night); `first` from whether it saw the chest
             opened; threat, defend and shout read only perceived events;
             a sleeper's or a fallen person's things are the ones visible or
             believed held. A chooser also notices the loss of ANY thing it
             believed it held (not only its own) at its next choice, so the
             wealth need never drops unnoticed.

    reprisal  the interrupt loophole closed (2026-10-29). A harmful act aimed at
             a person (HARMFUL: strike, shove, threaten, insult, steal, accuse,
             search) takes at least HARM_MIN turns, whatever it is: the actor is
             committed and spends its time. Everyone who SEES one lowers its
             liking of the actor a little (WITNESS_COST), and with `drives` the
             actor's own standing and company suffer for each witness. An act
             that is interrupted keeps the turns already done: choosing the same
             act again within RESUME_SPAN turns takes only what is left, so being
             interrupted costs the interrupted one nothing but the choice. The
             target's anger, liking, the gossip and its own reply were there
             before; this flag adds the actor's cost and the resume. One rule for
             every harmful act, none for a single verb.
    wear     the same words from the same person mean less each time in a short
             span (diminishing returns). For `praise` and `insult` (WEAR_VERBS):
             each earlier one by the same actor to the same target within
             WEAR_SPAN turns multiplies the effect by WEAR (liking, anger, the
             slight to standing, lonely, the hostility perceived).
    NEW FLAGS (2026-11-08, the inn fix; all four OFF in "all" and "all2", ON in "all3"):
    meals    food in shareable PORTIONS (a pot of stew, a loaf). Whoever holds the pot may eat a portion,
             SERVE one to somebody (sharing costs a fed giver little and feeds a hungry one a lot), and the inn's
             house rules sell a meal for a thing of small value (`buy`) or for work (`work`). The supply is a
             dial (World(supply=...)). Needs `trade`.
    partner  one rule for everybody: every act aimed at a person carries the actor's TRUST in that person (its
             liking, the deeds it heard of, lies and robberies it remembers). The rules score reads it the same
             for all: kind acts go to the trusted, restraint toward the trusted, less for the distrusted.
    news     gossip by SALIENCE, not by sign: a deed is worth telling by how unusual it is among the deeds the
             teller has seen, how much it goes against its own opinion of the doer, and how big it is. A witness
             warms to one who helps in view as it cools to one who harms (the mirror of `reprisal`).
    calm     the causes of the rules cast's quarrels, found in step 0a2 (all-rules cast, 5 seeds, 30 nights):
             the cast made about 52 harsh acts a night (24 charges, 16 blows, 7 searches), mean liking -0.77.
             (1) A charge was repeated: habit forgot what one had said after 40 turns, so every believer accused
                 again and again. Now the memory of what one said lasts the night and fades slowly (ECHO_HALF), and
                 saying the same kind of thing to the same person pulls less, as the same words do.
             (2) A blow never spent the anger behind it, so a fight ran until someone fell. Acting on a grievance
                 now spends part of it (DISCHARGE), as eating spends hunger.
             (3) The rules score read neither one's wounds nor the company drive. A hurt one is now held back
                 from blows and threats, and a lonely one is drawn to warm acts (policy.flag_terms).
             (4) A charge about a thing one no longer thinks the culprit holds is moot (feature `moot`, and a
                 search only of one still suspected).
             (5) Reprisal charged an answer to a wrong as much as the wrong: every witness cooled to the one who hit
                 back. A witness now charges an answer a quarter of a first harm (PROVOKED_SHARE), which is the
                 main cause of the liking spiral (liking -0.77 -> -0.33 with the rest).
    The first six flags are what "all" means (as in the first test round, so
    its numbers can be redone); "all2" is every flag, these two included.

LIFETIME (lab/world2/lifetime.py): `carry` brings in each cast member's
traits, liking and memories from the night before; `night` numbers the night.
"""
from __future__ import annotations

import math
import random

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

# ---------------------------------------------------------------- flags
BASE_FLAGS = ("drives", "gossip", "trade", "watch", "check", "known")      # what "all" means
FLAGS = BASE_FLAGS + ("reprisal", "wear")                                  # "all2" is all of them
NEW_FLAGS = ("meals", "partner", "news", "calm")                          # the inn fix; "all3" is FLAGS + these
ALL3 = FLAGS + NEW_FLAGS
HARM_MIN = 3             # reprisal: the least turns a harmful act aimed at a person takes
WITNESS_COST = 0.04      # reprisal: a witness's liking of the actor falls by this
WITNESS_SLIGHT = 0.03    # reprisal (+drives): the actor's standing / company worsen by this per witness
RESUME_SPAN = 20         # reprisal: an interrupted act can be taken up again within this many turns
# meals: a ration takes PORTION off the hunger need. One need unit is about 1,100 kcal a day-night (a night's drift of
# 0.9 is about 1,000 kcal), so a ration of 0.25 is about 275 kcal (u): a bowl of stew of 290 g at 95 kcal/100 g, or a
# 100 g slab of bread at 265 kcal/100 g. The pot holds 12 bowls (3.5 kg), the loaf 8 slabs (0.8 kg): 20 rations = 5.0 need units, against about 4.6 that
# five people need to stay at their set point through a night (0.9 drift + 0.3 start - 0.3 set point, each). All (u).
PORTION = 0.25
FULL = 0.10             # meals: below this hunger a person is full and eats no more
STOCK = {"the pot of stew": 12, "the loaf of bread": 8}
MEAL_PORTIONS, WORK_PORTIONS = 2, 1     # a thing paid buys a meal (two rations); an hour's chores earn one
WORK_TIRE, CHORE_RELIEF = 0.06, 0.04    # work tires the worker, and eases the one it helps
PAYABLE = (1, 3)                        # the values a house will take for a meal (no change is given)
# partner: the weight of trust in the score of an act aimed at a person (the same for every actor)
PARTNER_W = 1.5
BENEFIT_ACTS = {"give", "praise", "help up", "bandage", "defend", "gossip", "offer", "accept", "serve", "buy", "work"}
RESTRAINT_ACTS = {"strike", "shove", "threaten", "insult", "steal", "accuse", "search"}
HELPED = {"give", "help up", "bandage", "defend", "serve"}   # news: a witness warms to the one who does these
SURPRISE_BITS = 3.0     # news: a deed 1 in 8 of what the teller has seen is fully surprising
# calm
WOUND_W = 2.0           # calm: the rules score reads one's own wounds (the safety drive), as it reads hunger
PROVOKED_SHARE, PROVOKED_SPAN = 0.25, 60   # calm: a witness charges an answer to a wrong this share of a first harm
ECHO_HALF = 150         # calm: the memory of what one has said halves every 150 turns (half a night)
DISCHARGE = 0.5         # calm: a harmful act aimed at a person spends this share of the actor's anger at it
LONELY_W, LONELY_SET = 1.5, 0.3      # calm: weight of the company drive (above the set point) on warm acts
WARM_ACTS = {"praise", "gossip", "serve", "give"}
WEAR_VERBS = {"praise", "insult"}
WEAR, WEAR_SPAN = 0.5, 40   # wear: each repeat within the span halves the effect
DUR.update({"sit down": 4, "sleep": 12, "look around": 2, "gossip": 3, "ask": 1, "offer": 1,
            "accept": 2, "refuse": 1, "serve": 2, "buy": 2, "work": 10})
DRIVES = ("energy", "safety", "company", "standing", "wealth", "rest", "curiosity")
# NATURE: for each type, each drive's (weight, set point). The error of a drive
# is max(0, need - set point); a weight of 0 means the type does not care.
NATURE = {
    "peasant": dict(energy=(1.0, .30), safety=(1.2, .15), company=(0.8, .30), standing=(0.5, .40),
                    wealth=(0.0, 1.0), rest=(0.8, .40), curiosity=(0.4, .50)),
    "knight": dict(energy=(0.8, .35), safety=(0.6, .30), company=(0.5, .40), standing=(1.2, .20),
                   wealth=(0.0, 1.0), rest=(0.6, .45), curiosity=(0.5, .40)),
    "merchant": dict(energy=(0.9, .30), safety=(1.0, .20), company=(0.6, .35), standing=(0.8, .30),
                     wealth=(1.2, .20), rest=(0.7, .40), curiosity=(0.3, .50)),
    "thief": dict(energy=(0.9, .30), safety=(0.9, .25), company=(0.3, .50), standing=(0.3, .50),
                  wealth=(1.3, .20), rest=(0.6, .45), curiosity=(0.8, .30)),
    "cook": dict(energy=(0.8, .30), safety=(0.9, .20), company=(1.1, .25), standing=(0.7, .30),
                 wealth=(0.0, 1.0), rest=(0.9, .35), curiosity=(0.4, .45)),
    "bandit": dict(energy=(0.8, .35), safety=(0.6, .35), company=(0.1, .70), standing=(0.6, .30),
                   wealth=(1.4, .10), rest=(0.5, .50), curiosity=(0.5, .40)),
}
WANT = {"merchant": 13, "thief": 6, "bandit": 6}      # held value that meets the wealth need
# drift per turn, when nothing serves the need (hunger and thirst keep their own)
RATE = {"rest": 0.002, "curiosity": 0.004, "company": 0.004, "company_met": 0.006, "standing_fade": 0.995}
EXERT = {"walk to": 0.01, "flee to": 0.02, "strike": 0.02, "shove": 0.01, "play": 0.02, "search": 0.01}
# how being the TARGET of an act moves standing (slight) and company (lonely)
SLIGHT = {"insult": .20, "accuse": .25, "search": .20, "threaten": .15, "strike": .10, "shove": .10,
          "refuse": .15, "praise": -.20, "give": -.15, "help up": -.10, "bandage": -.10, "defend": -.10,
          "accept": -.10, "gossip": -.05, "serve": -.15}
KIND_ACTS = {"praise", "give", "help up", "bandage", "defend", "accept", "gossip", "serve"}
HARSH_ACTS = {"insult", "refuse", "threaten", "strike", "shove", "accuse", "search"}
# deeds a witness may tell of (gossip): +1 kind, -1 harm, -1.5 a lie found out
DEED = {**{a: -1.0 for a in HARMFUL}, "give": 1.0, "help up": 1.0, "bandage": 1.0, "defend": 1.0,
        "praise": 0.5, "refuse": -0.5, "lie": -1.5, "serve": 1.0}
PAST = {"strike": "struck", "shove": "shoved", "threaten": "threatened", "insult": "insulted",
        "steal": "robbed", "accuse": "accused", "search": "searched", "give": "gave a gift to",
        "help up": "helped up", "bandage": "bandaged", "defend": "defended", "praise": "praised",
        "refuse": "refused", "lie": "lied to", "serve": "fed"}
# known: how much a perceived act aimed at me says about the actor's hostility
# toward me (a kind act lowers it); it fades as anger fades (0.985 a turn)
HOSTILE = {"strike": .6, "shove": .5, "threaten": .4, "steal": .4, "insult": .25, "accuse": .25,
           "search": .25, "refuse": .1, "give": -.3, "help up": -.3, "bandage": -.3, "defend": -.2,
           "praise": -.1, "accept": -.1, "serve": -.3}
START_ROOM = {n: room for n, _, room, _ in CAST}


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
        self.pending = []          # events of others perceived since the last choice
        self.insulted_by = set()
        self.opened_chest = False
        # used only when a flag is on (no random draws here, so the night is unchanged)
        self.fatigue, self.boredom, self.lonely, self.slight = 0.10, 0.30, 0.20, 0.10
        self.glimpsed, self.examined, self.visited, self.kinds_seen = set(), set(), {room}, set()
        self.deeds = []            # deeds this person SAW (and may tell of): dicts, newest last
        self.told = set()          # (hearer, deed key) already told
        self.rep = {}              # name -> sum of deed signs heard about them (gossip)
        self.liars = {}            # name -> times caught in a falsehood
        self.memory = []           # lasting records across nights (deaths, lies found out, thefts)
        self.requests = {}         # asker -> {"thing", "pay", "t"}
        self.doubted = set()       # told-beliefs already dropped after a failed search
        # used only under the `known` flag: what p perceived of the others
        self.seen_at = {}          # name -> room where p last saw or heard it (None: unknown, dead)
        self.seen_open = False     # p saw (or did) the old chest opened tonight
        self.hostile = {}          # name -> (perceived hostility toward p, turn it was set)
        self.partial = {}          # reprisal: (verb, target, obj) -> (turns done, turn interrupted)
        self.said = {}             # calm: option text -> turns it was done tonight
        self.said_kind = {}        # calm: (verb, object) -> turns

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
    def __init__(self, seed=0, bandit=None, turns=300, flags=(), carry=None, night=0, supply=1.0):
        self.flags = frozenset(BASE_FLAGS if flags == "all" else FLAGS if flags == "all2" else
                               ALL3 if flags == "all3" else flags)
        bad = self.flags - set(FLAGS) - set(NEW_FLAGS)
        assert not bad, f"unknown flags {sorted(bad)}"
        self.night = night
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
        self.words = []                # wear: (turn, actor, target, verb) of the words said
        self.guard = {}                # person -> (defender, until)
        self.log = []
        self.n = 0
        # A bandit comes in from the road in about 60% of the nights.
        if bandit is None:
            bandit = rng.random() < 0.6
        self.bandit_at = rng.randint(turns // 3, 2 * turns // 3) if bandit else None
        if carry:                       # the same cast as last night: no random draw here
            for name, c in carry.items():
                p = self.people[name]
                p.base, p.traits = dict(c["base"]), dict(c["base"])
                p.aff.update(c["aff"])
                p.memory, p.rep, p.liars = c["memory"], dict(c["rep"]), dict(c["liars"])
                p.deeds, p.told = list(c["deeds"]), set(c["told"])
        if "trade" in self.flags:       # the cook holds the food and serves it
            cook = next(p.name for p in self.people.values() if p.kind == "cook")
            for n in ("the loaf of bread", "the bowl of stew"):
                self.things[n].update(at=cook, owner=cook)
        self.portions = {}             # meals: thing -> rations left
        if "meals" in self.flags:
            assert "trade" in self.flags, "meals needs trade (the cook holds the food)"
            cook = next(p.name for p in self.people.values() if p.kind == "cook")
            del self.things["the bowl of stew"]          # a pot to share instead of one bowl
            self.things["the pot of stew"] = {"at": cook, "props": {"food"}, "value": 0, "owner": cook,
                                              "hidden": False}
            self.portions = {n: max(1, round(k * supply)) for n, k in STOCK.items()}
        self.start_at = {n: th["at"] for n, th in self.things.items()}   # the inn as each night sets it
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
        and violence in the last few turns. Under `known`: the hostility p
        perceived, a visibly armed bandit, violence p perceived."""
        if "known" in self.flags:
            return self._danger_known(p)
        d = 0.0
        for q in self.here(p):
            if q.acts():
                d += max(0.0, q.anger.get(p.name, 0) - 0.2) + (0.3 if self.weapon(q) and q.kind == "bandit" else 0)
        recent = [e for e in self.log[-12:] if e["action"] in VIOLENT and e.get("done", True)
                  and self.people.get(e["actor"]) and self.people[e["actor"]].room == p.room]
        return min(1.0, d + 0.3 * len(recent))

    def recently_harmed(self, victim, turns=20, acts=None, knower=None):
        """Who harmed `victim` in the last turns (for defend, rescue). With a
        `knower` under `known`: only the harms the knower perceived or did."""
        acts = acts or HARMFUL
        return {e["actor"] for e in self.log[-120:] if e["t"] >= self.t - turns and e.get("target") == victim
                and e["action"] in acts and e.get("done", True) and self._knows(knower, e)}

    def _knows(self, p, e):
        """Under `known`: p perceived the event or did it. Else always true."""
        return p is None or "known" not in self.flags or e["actor"] == p.name or e["id"] in p.perceived

    def hostility(self, p, name):
        """How hostile p has SEEN `name` be toward it (known flag), 0..1."""
        h, t0 = p.hostile.get(name, (0.0, self.t))
        return h * 0.985 ** (self.t - t0)

    def _danger_known(self, p):
        d = 0.0
        for q in self.here(p):
            if q.acts():
                armed = any(n in WEAPON for n in self.held(q, visible_only=True))
                d += max(0.0, self.hostility(p, q.name) - 0.2) + (0.3 if armed and q.kind == "bandit" else 0)
        recent = [e for e in self.log[-12:] if e["action"] in VIOLENT and e.get("done", True) and self._knows(p, e)
                  and self.people.get(e["actor"]) and self.people[e["actor"]].room == p.room]
        return min(1.0, d + 0.3 * len(recent))

    def _believed_held(self, p, q):
        """What p sees q hold or believes q holds (known flag)."""
        seen = set(self.held(q, visible_only=True))
        return [n for n in self.things if n in seen or p.bel.get(("where", n), (None,))[0] == q.name]

    # ------------------------------------------------------------ flags: drives, trust, the lifetime
    @staticmethod
    def asleep(p):
        return bool(p.doing) and p.doing["verb"] == "sleep"

    def wealth(self, p):
        """What p holds of value. Under `known`, what p BELIEVES it holds:
        a loss counts once p notices it (at its next choice, _check_own)."""
        if "known" in self.flags:
            return sum(th["value"] for n, th in self.things.items()
                       if p.bel.get(("where", n), (None,))[0] == p.name)
        return sum(self.things[n]["value"] for n in self.held(p))

    def drives(self, p):
        """Each drive's need, 0..1 (0 = met). Read by a mind; written by no one
        but this file."""
        return {"energy": max(p.hunger, p.thirst),
                "safety": 1.0 if p.dead else max(p.fear, min(1.0, max(0.0, (10 - p.hp) / 10))),
                "company": p.lonely, "standing": p.slight,
                "wealth": max(0.0, 1 - self.wealth(p) / WANT.get(p.kind, 10)),
                "rest": p.fatigue, "curiosity": p.boredom}

    def drive_error(self, p, need=None):
        """The weighted error of p's drives, by its type's NATURE. A drop in it
        is the homeostatic reward (Keramati and Gutkin)."""
        need = need or self.drives(p)
        return sum(w * max(0.0, need[d] - s) for d, (w, s) in NATURE[p.kind].items())

    def distrust(self, p, name):
        """How much p distrusts `name`, from p's own memories: dislike, bad
        deeds heard of (gossip), falsehoods caught. 0..1."""
        return min(1.0, max(0.0, -p.aff.get(name, 0)) + 0.2 * max(0.0, -p.rep.get(name, 0))
                   + 0.5 * p.liars.get(name, 0))

    def carry_out(self, names=None):
        """What each cast member takes into the next night (lifetime.py)."""
        names = names or [n for n, _, _, _ in CAST]
        cast = set(n for n, _, _, _ in CAST)
        out = {}
        for n in names:
            p = self.people[n]
            deeds = [d for d in p.deeds if d["actor"] in cast and d["target"] in cast][-6:]
            keys = {d["key"] for d in deeds}
            out[n] = {"base": dict(p.base), "aff": {k: v for k, v in p.aff.items() if k in cast},
                      "memory": p.memory, "rep": {k: v for k, v in p.rep.items() if k in cast},
                      "liars": {k: v for k, v in p.liars.items() if k in cast}, "deeds": deeds,
                      "told": sorted(t for t in p.told if t[0] in cast and t[1] in keys)}
        return out

    def _rested(self, p, amount):
        p.fatigue = min(1.0, max(0.0, p.fatigue - amount))

    def _novel(self, p, amount):
        p.boredom = max(0.0, p.boredom - amount)

    # ------------------------------------------------------------ options
    def options(self, p):
        """Every action the world permits p now, with the numbers the rule
        policy and the trait bias read (`f`)."""
        W = self.things
        out = []

        tcache = {}

        def add(verb, text, target=None, obj=None, **f):
            if partner and target in self.people and "trust" not in f:
                if target not in tcache:
                    tcache[target] = self.trust(p, target)
                f["trust"] = tcache[target]
            dur = DUR[verb]
            if reprisal and verb in HARMFUL and target in self.people:
                dur = max(dur, HARM_MIN)
            out.append({"verb": verb, "text": text, "target": target, "obj": obj, "f": f,
                        "dur": dur})

        danger = self.danger(p)
        others = self.here(p)
        # p is under suspicion: accused or searched lately
        suspected = float(any(e["target"] == p.name and e["action"] in ("accuse", "search") and e["done"]
                              for e in self.log[-80:] if e["t"] >= self.t - 40 and self._knows(p, e)))
        known = "known" in self.flags
        reprisal = "reprisal" in self.flags
        partner = "partner" in self.flags
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
            if "food" in th["props"] and not (n in self.portions and p.hunger <= FULL):   # a full stomach holds no more
                add("eat", f"eat a portion of {n}" if n in self.portions else f"eat {n}", obj=n, need=p.hunger)
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
            if n in self.portions:                  # meals: a pot is shared by the portion (serve), not handed over
                continue
            for q in others:
                if q.acts():
                    add("give", f"give {n} to {q.name}", target=q.name, obj=n,
                        gain=th["value"] / 10, returning=float(th["owner"] == q.name),
                        like=max(0, p.aff.get(q.name, 0)))
                    if "trade" in self.flags and p.requests.get(q.name, {}).get("thing") == n \
                            and p.requests[q.name]["pay"] is None:
                        out[-1]["f"].update(asked=1.0, distrust=self.distrust(p, q.name))
        for n in self.loose(p.room):
            props = W[n]["props"]
            if "ale_source" in props:
                add("drink", f"drink from {n}", obj=n, need=p.thirst, ale=1.0)
            if "water_source" in props:
                add("drink", f"drink from {n}", obj=n, need=p.thirst)
            if "chest" in props and not p.opened_chest:
                add("open", f"open {n}", obj=n,
                    first=float(not p.seen_open) if known else float(self.chest is None))
        w = self.weapon(p)
        for q in others:
            harmed = self.recently_harmed(q.name, acts=PHYSICAL, knower=p)
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
                for n in (self._believed_held(p, q) if known else self.held(q)):
                    add("steal", f"take {n} from {q.name}", target=q.name, obj=n, gain=W[n]["value"] / 10,
                        witnesses=len([o for o in others if o.acts() and o is not q]), **f)
            if q.down or q.hp < 7:
                add("bandage", f"bandage {q.name}", target=q.name, need=1.0 if q.down else 0.5, **f)
            if q.down and not q.dead:
                add("help up", f"help {q.name} up", target=q.name, need=1.0, **f)
            if q.acts() and harmed - {p.name}:
                add("defend", f"defend {q.name}", target=q.name, attacked=1.0, **f)
            if q.acts():
                sleeping = "drives" in self.flags and self.asleep(q)     # a sleeper's things can all be taken
                for n in ((self._believed_held(p, q) if known else self.held(q)) if sleeping
                          else self.held(q, visible_only=True)):
                    if sleeping or (W[n]["value"] >= 1 and W[n]["owner"] == q.name):
                        add("steal", f"steal {n} from {q.name}", target=q.name, obj=n,
                            gain=W[n]["value"] / 10,
                            witnesses=len([o for o in others if o.acts() and o is not q]), **f)
            # what p believes about thefts: accuse, tell the truth, or lie
            for (kind, thing), (culprit, _src) in list(p.bel.items()):
                if kind != "took" or culprit == p.name:
                    continue
                if culprit == q.name:
                    extra = {}
                    if "calm" in self.flags:     # a charge about a thing p no longer thinks the culprit holds is moot
                        extra["moot"] = float(p.bel.get(("where", thing), (None,))[0] != q.name)
                    add("accuse", f"accuse {q.name} of taking {thing}", target=q.name, obj=thing,
                        mine=float(W[thing]["owner"] == p.name), **extra, **f)
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
        if self.flags:
            self._flag_options(p, add, others, danger)
        if danger > 0.2 or any(self.recently_harmed(q.name, 6, knower=p) for q in others):
            add("shout for help", "shout for help", danger=danger)
        add("wait", "wait")
        # the same text twice would be the same option: keep the first
        seen, uniq = set(), []
        mine_ev = [e for e in self.log[-80:] if e["actor"] == p.name and e["done"] and e["t"] >= self.t - 40]
        done = [e["text"] for e in mine_ev]
        done_kinds = [(e["action"], e["parts"].get("thing")) for e in mine_ev]
        last = mine_ev[-1] if mine_ev and mine_ev[-1]["t"] >= self.t - 10 else None
        undo = {"take": ("drop",), "steal": ("drop",), "drop": ("take",)}
        calm = "calm" in self.flags
        for o in out:
            if o["text"] not in seen:
                seen.add(o["text"])
                # habit: undoing what one just did (drop right after take) pulls less;
                # without it the model policies looped take, drop, take
                o["f"]["undo"] = float(bool(last) and o["verb"] in undo.get(last["action"], ())
                                       and o["obj"] == last["parts"].get("thing"))
                o["f"]["repeat"] = float(done.count(o["text"]))     # habit: the same act again pulls less
                o["f"]["repeat_kind"] = float(sum(1 for (v_, ob) in done_kinds if v_ == o["verb"] and ob == o["obj"]))
                if calm:                    # calm: one remembers what one said all night, fading slowly
                    o["f"]["repeat"] = self._echo(p.said.get(o["text"]))
                    o["f"]["repeat_kind"] = self._echo(p.said_kind.get((o["verb"], o["obj"]))) + \
                        (self._echo(p.said_kind.get((o["verb"], o["target"]))) if o["target"] else 0.0)
                uniq.append(o)
        return uniq

    def _flag_options(self, p, add, others, danger):
        """The menu acts the flags add. Facts only: whether to take them is
        the chooser's business."""
        W = self.things
        awake = [q for q in others if q.acts() and not self.asleep(q)]
        if "drives" in self.flags:
            add("sit down", "sit down", fatigue=p.fatigue, danger=danger)
            add("sleep", "sleep", fatigue=p.fatigue, danger=danger, others=float(len(awake)),
                distrust=max([self.distrust(p, q.name) for q in awake] + [0.0]))
            new = [n for n in self.loose(p.room) if n not in p.examined] + \
                  [q.name for q in others if q.name not in p.examined]
            room_things = len(self.loose(p.room)) + len(others)
            add("look around", "look around", unseen=len(new) / max(1, room_things), boredom=p.boredom)
        if "gossip" in self.flags:
            for q in awake:
                fresh = [d for d in reversed(p.deeds)
                         if not (q.name in (d["actor"], d["target"]) or (q.name, d["key"]) in p.told)]
                news = "news" in self.flags
                if news:        # the most salient deeds, whatever their sign (a stable sort: ties keep the newest first)
                    fresh.sort(key=lambda d: -self._salience(p, d))
                for d in fresh[:3]:
                    sign = DEED[d["action"]]
                    extra = {"sal": self._salience(p, d)} if news else {}
                    add("gossip", f"tell {q.name} that {d['actor']} {PAST[d['action']]} {d['target']}",
                        target=q.name, deed=d, bad=float(sign < 0), good=float(sign > 0),
                        dislike_actor=max(0, -p.aff.get(d["actor"], 0)), like_actor=max(0, p.aff.get(d["actor"], 0)),
                        like=max(0, p.aff.get(q.name, 0)), dislike=max(0, -p.aff.get(q.name, 0)), **extra)
        if "trade" in self.flags:
            mine = [n for n in self.held(p) if W[n]["value"] >= 1]
            for q in awake:
                dq = self.distrust(p, q.name)
                like = max(0, p.aff.get(q.name, 0))
                for n in self.held(q, visible_only=True):
                    props = W[n]["props"]
                    food, drink = float("food" in props), float(bool(props & {"ale", "water"}))
                    need = max(food * p.hunger, drink * p.thirst)
                    if n in self.portions:       # meals: one asks for a portion; the counter sells (buy, work)
                        add("ask", f"ask {q.name} for a portion of {n}", target=q.name, obj=n, need=need,
                            gain=0.0, like=like, distrust=dq)
                        continue
                    if food or drink or W[n]["value"] >= 1:
                        add("ask", f"ask {q.name} for {n}", target=q.name, obj=n, need=need,
                            gain=W[n]["value"] / 10, like=like, distrust=dq)
                    for m in mine:
                        if m != n:
                            add("offer", f"offer {q.name} {m} for {n}", target=q.name, obj=n, pay=m, need=need,
                                gain=W[n]["value"] / 10, cost=W[m]["value"] / 10, like=like, distrust=dq)
                # the answers to what q asked of p (giving is the "give" option)
                r = p.requests.get(q.name)
                if r and r["thing"] in self.held(p):
                    if r["pay"] is not None and r["pay"] in self.held(q):
                        add("accept", f"accept {q.name}'s {r['pay']} for {r['thing']}", target=q.name,
                            obj=r["thing"], pay=r["pay"], gain=W[r["thing"]]["value"] / 10,
                            get=W[r["pay"]]["value"] / 10, like=like, distrust=dq)
                    add("refuse", f"refuse {q.name}", target=q.name, obj=r["thing"], like=like,
                        dislike=max(0, -p.aff.get(q.name, 0)), distrust=dq)

        if "meals" in self.flags:
            self._meal_options(p, add, awake)

    def _meal_options(self, p, add, awake):
        """meals: serve a portion from a pot p holds; buy a meal or work for one at a pot another holds."""
        W = self.things
        for n in self.held(p):
            if n not in self.portions or self.portions[n] <= 0:
                continue
            left = self.portions[n] / STOCK[n]
            for q in awake:
                r = p.requests.get(q.name, {})
                asked = float(r.get("thing") == n and r.get("pay") is None)
                add("serve", f"serve {q.name} a portion of {n}", target=q.name, obj=n, asked=asked,
                    seen_need=q.hunger if q.hunger > 0.6 else 0.0, own=p.hunger, left=left,
                    like=max(0, p.aff.get(q.name, 0)), distrust=self.distrust(p, q.name))
        pay = [m for m in self.held(p) if PAYABLE[0] <= W[m]["value"] <= PAYABLE[1] and m not in self.portions]
        for q in awake:
            for n in self.held(q, visible_only=True):
                if n not in self.portions or self.portions[n] <= 0:
                    continue
                dq = self.distrust(p, q.name)
                for m in pay:
                    add("buy", f"pay {q.name} {m} for a meal from {n}", target=q.name, obj=n, pay=m, need=p.hunger,
                        cost=W[m]["value"] / 10, distrust=dq)
                add("work", f"work for {q.name} for a share of {n}", target=q.name, obj=n, need=p.hunger,
                    fatigue=p.fatigue, distrust=dq)

    def _goal(self, p, room):
        """What a room offers p: food, drink, whoever p is angry at, loot, safety."""
        if "known" in self.flags:
            return self._goal_known(p, room)
        things = [n for n, th in self.things.items() if th["at"] == room]
        people = [q for q in self.people.values() if q.room == room and q is not p and not q.dead]
        food = any("food" in self.things[n]["props"] for n in things)
        drink = any(self.things[n]["props"] & {"ale", "ale_source", "water_source", "water"} for n in things)
        foe = max([p.anger.get(q.name, 0) for q in people] + [0])
        loot = any(self.things[n]["value"] >= 5 for n in self.things
                   if self.things[n]["at"] in [q.name for q in people])
        safe = 1.0 - min(1.0, sum(max(0, q.anger.get(p.name, 0) - 0.2) for q in people))
        company = min(1.0, len(people) / 3)
        g = dict(food=float(food), drink=float(drink), foe=foe, loot=float(loot), safe=safe,
                 company=company, chest=float(room == "the cellar" and self.chest is None))
        if "drives" in self.flags:
            g["new"] = float(room not in p.visited)
        return g

    def _goal_known(self, p, room):
        """The same, from what p knows: where it last saw each thing (fixed
        things, and anything it has not seen tonight: where the night put it),
        where it last saw or heard each person (else where the cast starts),
        and the hostility it perceived."""
        def at(n):
            b = p.bel.get(("where", n))
            return b[0] if b else self.start_at.get(n)
        things = [n for n in self.things if at(n) == room]
        people = [q for q in self.people.values() if q is not p
                  and p.seen_at.get(q.name, START_ROOM.get(q.name)) == room]
        food = any("food" in self.things[n]["props"] for n in things)
        drink = any(self.things[n]["props"] & {"ale", "ale_source", "water_source", "water"} for n in things)
        foe = max([p.anger.get(q.name, 0) for q in people] + [0])
        names = {q.name for q in people}
        loot = any(self.things[n]["value"] >= 5 for n in self.things if at(n) in names)
        safe = 1.0 - min(1.0, sum(max(0, self.hostility(p, q.name) - 0.2) for q in people))
        g = dict(food=float(food), drink=float(drink), foe=foe, loot=float(loot), safe=safe,
                 company=min(1.0, len(people) / 3),
                 chest=float(room == "the cellar" and not (p.seen_open or p.opened_chest)))
        if "drives" in self.flags:
            g["new"] = float(room not in p.visited)
        return g

    def _is_threat(self, p, q):
        """q struck, shoved or threatened p, or someone p likes, lately; or q
        is a bandit. Words (an accusation, an insult) do not make a threat."""
        if q.kind == "bandit":
            return True
        for e in self.log[-40:]:
            if e["actor"] == q.name and e["action"] in PHYSICAL and e.get("done", True) and self._knows(p, e):
                v = e.get("target")
                if v == p.name or (v in p.aff and p.aff[v] > 0.2):
                    return True
        return False

    def _suspects(self, p, q):
        if "calm" in self.flags:         # still suspects q only of what it believes q still holds
            return any(k == "took" and v == q.name and p.bel.get(("where", th), (None,))[0] == q.name
                       for (k, th), (v, _s) in p.bel.items())
        return any(k == "took" and v == q.name for (k, _), (v, _s) in p.bel.items())

    # ------------------------------------------------------------ the turn
    def step(self, policy):
        self.t += 1
        if self.bandit_at == self.t:
            self._bandit_arrives()
        for p in self.people.values():
            self._drift(p)
        order = list(self.people.values())
        self.rng.shuffle(order)
        for p in order:
            if not p.acts():
                p.doing = None
                continue
            if p.doing is None:
                self._choose(p, policy)
            else:
                p.doing["left"] -= 1
                if p.doing["left"] <= 0:
                    self._finish(p)

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
        if "drives" in self.flags and not p.dead:
            p.fatigue = min(1.0, p.fatigue + RATE["rest"])
            p.boredom = min(1.0, p.boredom + RATE["curiosity"])
            liked = sum(1 for q in self.here(p) if q.acts() and p.aff.get(q.name, 0) > 0.1)
            p.lonely = max(0.0, p.lonely - RATE["company_met"] * min(2, liked)) if liked else \
                min(1.0, p.lonely + RATE["company"])
            p.slight *= RATE["standing_fade"]
        if p.requests:                  # a request not answered in 30 turns lapses
            p.requests = {k: r for k, r in p.requests.items() if r["t"] >= self.t - 30}

    def _choose(self, p, policy):
        # a trait is a range: this choice's traits vary around the character's own
        p.traits = {t: min(1, max(0, self.rng.gauss(p.base[t], STATE_SPREAD))) for t in TRAITS}
        self._look(p, "sight")
        self._check_own(p)
        opts = self.options(p)
        pc, pa = policy.probs(self, p, opts)
        i = self.rng.choices(range(len(opts)), weights=pc)[0]
        o = opts[i]
        left = o["dur"]
        if "reprisal" in self.flags:             # an interrupted act is taken up where it stopped
            done_, t_ = p.partial.pop((o["verb"], o["target"], o["obj"]), (0, -RESUME_SPAN - 1))
            if self.t - t_ <= RESUME_SPAN:
                left = max(1, left - done_)
        p.doing = {**o, "left": left, "choice": {
            "options": [x["text"] for x in opts], "chosen": i,
            "p_char": [round(x, 5) for x in pc], "p_avg": [round(x, 5) for x in pa],
            "seen": list(p.pending), "traits": {k: round(v, 3) for k, v in p.traits.items()}}}
        p.pending = []
        if left <= 1:
            self._finish(p)

    def _finish(self, p):
        d, p.doing = p.doing, None
        ok, info = self._apply(p, d)
        ev = self._emit(p, d, done=ok, **info)
        if "calm" in self.flags and ok:
            p.said.setdefault(d["text"], []).append(self.t)
            p.said_kind.setdefault((d["verb"], d["obj"]), []).append(self.t)
            if d["target"]:
                p.said_kind.setdefault((d["verb"], d["target"]), []).append(self.t)
        if self.flags and ok:
            self._after(p, d, ev)

    def interrupt(self, p, by):
        if p.doing and p.acts():
            d, p.doing = p.doing, None
            if d["verb"] == "sleep":    # woken: the turns slept still rested it
                self._rested(p, 0.05 * (d["dur"] - d["left"]))
            elif "reprisal" in self.flags and d["dur"] - d["left"] > 0:
                p.partial[(d["verb"], d["target"], d["obj"])] = (d["dur"] - d["left"], self.t)
            self._emit(p, d, done=False, why=f"stopped by {by}")

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
            if o in self.portions:                       # meals: one portion of the pot
                if not self._feed(p, 1, o):
                    return False, {"why": "none left"}
                return True, {}
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
            aim = max(0.3, 0.75 - 0.25 * anger - 0.2 * p.drunk + (0.2 if victim.down else 0)
                      - (0.25 * max(0.0, p.fatigue - 0.6) if "drives" in self.flags else 0)
                      + (0.2 if "drives" in self.flags and self.asleep(victim) else 0))
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
                q.aff[p.name] = min(1, q.aff.get(p.name, 0) + 0.15 * self._wear(p.name, q.name, v))
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
        if self.flags:
            return self._apply_flag(p, d, q)
        raise ValueError(v)

    def _apply_flag(self, p, d, q):
        v, W, o = d["verb"], self.things, d["obj"]
        if v == "sit down":
            self._rested(p, 0.15)
            return True, {}
        if v == "sleep":
            self._rested(p, 0.6)
            return True, {}
        if v == "look around":
            new = [n for n in self.loose(p.room) if n not in p.examined] + \
                  [x.name for x in self.here(p) if x.name not in p.examined]
            p.examined.update(new)
            self._novel(p, 0.02 + 0.04 * len(new))
            self._look(p, "sight")
            return True, {"found": new}
        if v == "gossip":
            return True, {}                                  # the hearer changes on hearing (_learn)
        if v in ("ask", "offer"):
            q.requests[p.name] = {"thing": o, "pay": d["f"].get("pay"), "t": self.t}
            return True, {}
        if v == "accept":
            pay = d["f"]["pay"]
            if W[o]["at"] != p.name or W[pay]["at"] != q.name:
                return False, {"why": "not held"}
            W[o].update(at=q.name, owner=q.name, hidden=False)
            W[pay].update(at=p.name, owner=p.name, hidden=False)
            q.requests.pop(p.name, None)
            p.requests.pop(q.name, None)
            for a, b in ((p, q), (q, p)):
                a.aff[b.name] = min(1, a.aff.get(b.name, 0) + 0.1)
            return True, {"traded": pay}
        if v == "refuse":
            p.requests.pop(q.name, None)
            q.aff[p.name] = max(-1, q.aff.get(p.name, 0) - 0.1)
            return True, {}
        if v == "serve":                                     # meals: a portion from the pot p holds
            if o not in self.portions or W[o]["at"] != p.name or not self._feed(q, 1, o):
                return False, {"why": "none left"}
            q.aff[p.name] = min(1, q.aff.get(p.name, 0) + 0.3)        # as a gift (give)
            q.anger[p.name] = max(0, q.anger.get(p.name, 0) - 0.3)
            return True, {}
        if v == "buy":                                       # meals: a thing for a meal, at the house's price
            m = d["f"]["pay"]
            if W[m]["at"] != p.name or W[o]["at"] != q.name or not self._feed(p, MEAL_PORTIONS, o):
                return False, {"why": "not held"}
            W[m].update(at=q.name, owner=q.name, hidden=False)
            for a, b in ((p, q), (q, p)):
                a.aff[b.name] = min(1, a.aff.get(b.name, 0) + 0.1)
            return True, {"traded": m}
        if v == "work":                                      # meals: chores for a ration
            if W[o]["at"] != q.name or not self._feed(p, WORK_PORTIONS, o):
                return False, {"why": "none left"}
            p.fatigue = min(1.0, p.fatigue + WORK_TIRE)
            self._rested(q, CHORE_RELIEF)
            for a, b in ((p, q), (q, p)):
                a.aff[b.name] = min(1, a.aff.get(b.name, 0) + 0.1)
            return True, {}
        raise ValueError(v)

    def _feed(self, person, n, item):
        """meals: `person` eats up to n rations of `item`. Returns how many it got (0 if the pot is empty)."""
        n = min(n, self.portions.get(item, 0))
        if n <= 0:
            return 0
        self.portions[item] -= n
        person.hunger = max(0.0, person.hunger - PORTION * n)
        if self.portions[item] <= 0:
            self.things[item]["at"] = None
        return n

    def trust(self, p, name):
        """partner: how much p trusts `name`, -1..1, from what p already has: its liking, the deeds it heard of
        (rep), the lies it caught, the robberies and the killing it remembers. One rule for everybody."""
        bad = sum(1 for m in p.memory if m.get("who") == name and m.get("what") == "robbed")
        bad += 2 * sum(1 for m in p.memory if m.get("by") == name and m.get("what") == "died")
        t = p.aff.get(name, 0.0) + 0.1 * p.rep.get(name, 0) - 0.5 * p.liars.get(name, 0) - 0.2 * min(3, bad)
        return max(-1.0, min(1.0, t))

    def _salience(self, p, d):
        """news: how worth telling a deed is. Unusual among the deeds p has seen (its surprisal, in bits), against
        p's own opinion of the doer, and big. The same for a good deed and a bad one."""
        sign = DEED[d["action"]]
        same = sum(1 for x in p.deeds if (DEED[x["action"]] > 0) == (sign > 0))
        surprise = min(1.0, -math.log2((same + 1) / (len(p.deeds) + 2)) / SURPRISE_BITS)
        against = min(1.0, 2 * max(0.0, -math.copysign(1, sign) * p.aff.get(d["actor"], 0)))
        return abs(sign) / 1.5 * (0.6 * surprise + 0.4 * against)

    def _echo(self, turns):
        """calm: how much of what p said before still weighs on it, from the turns it said it."""
        return sum(0.5 ** ((self.t - t0) / ECHO_HALF) for t0 in turns) if turns else 0.0

    def _after(self, p, d, ev):
        """What a done act does to the actor itself, under the flags."""
        v = d["verb"]
        if "wear" in self.flags and v in WEAR_VERBS and d["target"] in self.people:
            self.words.append((self.t, p.name, d["target"], v))
        if "drives" in self.flags:
            p.fatigue = min(1.0, p.fatigue + EXERT.get(v, 0))
            if v == "drink":
                self._rested(p, 0.03)                         # drinking slowly rests
            if v in ("walk to", "flee to") and p.room not in p.visited:
                p.visited.add(p.room)
                self._novel(p, 0.15)
            if v == "open":
                self._novel(p, 0.4 if ev.get("found") and self.chest and p.name == self._first_opener() else 0.15)
        if "calm" in self.flags and v in HARMFUL and d["target"] in self.people and d["target"] in p.anger:
            p.anger[d["target"]] *= 1.0 - DISCHARGE           # acting on a grievance spends part of it
        if v in ("give", "serve") and "trade" in self.flags:
            q = self.people[d["target"]]
            if p.requests.get(q.name, {}).get("thing") == d["obj"]:
                del p.requests[q.name]                         # served
        if v == "search" and "check" in self.flags:
            self._checked(p, self.people[d["target"]], ev)
        if v == "gossip":
            p.told.add((d["target"], d["f"]["deed"]["key"]))
        if "known" in self.flags:            # p knows where the things it just handled went
            for n in (d.get("obj"), d["f"].get("pay")):
                if n in self.things and v in ("take", "steal", "drop", "give", "eat", "drink", "hide", "accept", "buy"):
                    at = self.things[n]["at"]
                    if at is None:
                        p.bel.pop(("where", n), None)
                    else:
                        p.bel[("where", n)] = (at, ev["id"])
            if v == "open":
                p.seen_open = True
            for n in self.held(p):           # and what it holds after it (the chest's coins)
                p.bel[("where", n)] = (p.name, ev["id"])

    def _wear(self, actor, target, verb):
        """The effect of these words now: WEAR to the power of how many of the
        same words the same actor said to the same target within WEAR_SPAN."""
        if "wear" not in self.flags or verb not in WEAR_VERBS:
            return 1.0
        n = sum(1 for (t, a, b, v) in self.words if a == actor and b == target and v == verb
                and self.t - t <= WEAR_SPAN)
        return WEAR ** n

    def _first_opener(self):
        return next((e["actor"] for e in self.log if e["action"] == "open" and e["done"]), None)

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
        if d["verb"] == "gossip":
            dd = d["f"]["deed"]
            ev["facts"] = {"said": f"{dd['actor']} {PAST[dd['action']]} {dd['target']}", "true": True,
                           "who": dd["actor"], "deed": dd["action"], "victim": dd["target"],
                           "source": f"n{dd['key'][0]}:{dd['key'][1]}"}
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
        drives = "drives" in self.flags
        for q in self.people.values():
            if q is actor or q.dead:
                continue
            sees = q.room == room and not q.down
            asleep = drives and self.asleep(q)
            if asleep:
                sees = False                     # a sleeper sees nothing
            if sees and sneaky:                  # a theft is seen only sometimes
                chance = 0.3 if q.name == ev["target"] else 0.35
                if "watch" in self.flags and self.distrust(q, actor.name) >= 0.3:
                    chance = 0.6                 # a distrusted one is watched
                sees = self.rng.random() < chance * (1 - 0.5 * q.drunk)
            hears = (not sees) and loud and (room in ROOMS[q.room] or (asleep and q.room == room))
            if not (sees or hears):
                continue
            q.perceived.add(ev["id"])
            q.pending.append(ev["id"])
            if "known" in self.flags:
                self._perceive_known(q, ev)
            if drives and (ev["actor"], ev["action"]) not in q.kinds_seen:
                q.kinds_seen.add((ev["actor"], ev["action"]))
                self._novel(q, 0.03)
            if asleep and hears:                 # a loud noise wakes a sleeper
                self.interrupt(q, ev["id"])
            if sees:
                self._learn(q, ev, actor)
            else:
                q.fear = min(1, q.fear + 0.05 * (1 - q.traits["bravery"]))
            # something important stops what q is doing: harm to q, a blow in
            # sight or earshot, a shout, a theft it saw
            aimed = ev["target"] == q.name and ev["action"] in HARMFUL
            if aimed or ev["action"] in VIOLENT or ev["action"] == "shout for help" or (sneaky and sees):
                self.interrupt(q, ev["id"])

    def _perceive_known(self, q, ev):
        """What q takes from any event it saw or heard (known flag): where the
        actor is, who died, the chest opened, hostility aimed at q."""
        a = ev["action"]
        q.seen_at[ev["actor"]] = ev["room"]
        res = ev.get("result") or ""
        if res.startswith("killed "):
            q.seen_at[res[len("killed "):]] = None
        if a == "open":
            q.seen_open = True
        if ev["target"] == q.name and a in HOSTILE:
            h = max(0.0, min(1.0, self.hostility(q, ev["actor"]) + HOSTILE[a] * self._wear(ev["actor"], q.name, a)))
            q.hostile[ev["actor"]] = (h, self.t)
        if a == "accept" and ev["target"] == q.name:       # q's pay went, the thing came
            for n, holder in ((ev.get("traded"), ev["actor"]), (ev["parts"].get("thing"), q.name)):
                if n in self.things:
                    q.bel[("where", n)] = (holder, ev["id"])

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
                    q.memory.append({"night": self.night, "t": self.t, "what": "robbed", "who": actor.name,
                                     "thing": thing})
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
            wear = self._wear(actor.name, q.name, a)
            hurt = (0.45 if a in VIOLENT else (0.25 if a in PHYSICAL else 0.15)) * wear
            q.anger_up(actor.name, hurt * (1 + tr["temper"] + q.drunk))
            q.aff[actor.name] = max(-1, q.aff.get(actor.name, 0) - hurt)
            if a in VIOLENT:
                q.fear = min(1, q.fear + 0.3 * (1 - tr["bravery"]))
            if a == "insult":
                q.anger_up(actor.name, 0.2 * tr["pride"] * wear)
        elif a in HARMFUL and v in q.aff and q.aff[v] > 0.2:
            q.anger_up(actor.name, 0.25 * (tr["kindness"] + tr["temper"]))
            q.aff[actor.name] = max(-1, q.aff.get(actor.name, 0) - 0.15)
        if a in VIOLENT:
            q.fear = min(1, q.fear + 0.1 * (1 - tr["bravery"]))
        if a in ("help up", "bandage", "defend") and v in q.aff and q.aff[v] > 0:
            q.aff[actor.name] = min(1, q.aff.get(actor.name, 0) + 0.1)
        if self.flags:
            self._learn_flag(q, ev, actor)

    def _learn_flag(self, q, ev, actor):
        """What q takes from an event it SAW, under the flags."""
        a, v, thing = ev["action"], ev["target"], ev["parts"].get("thing")
        if "drives" in self.flags and v == q.name:
            wear = self._wear(actor.name, q.name, a)
            q.slight = min(1.0, max(0.0, q.slight + SLIGHT.get(a, 0) * wear))
            if a in KIND_ACTS:
                q.lonely = max(0.0, q.lonely - 0.15 * wear)
            elif a in HARSH_ACTS:
                q.lonely = min(1.0, q.lonely + 0.05 * wear)
        if "reprisal" in self.flags and a in HARMFUL and v in self.people and v != q.name and ev.get("done"):
            # a witness thinks the less of one who does harm; the doer pays for being seen
            share = PROVOKED_SHARE if "calm" in self.flags and self._provoked(q, actor, v, ev) else 1.0
            q.aff[actor.name] = max(-1, q.aff.get(actor.name, 0) - WITNESS_COST * share)
            if "drives" in self.flags:
                actor.slight = min(1.0, actor.slight + WITNESS_SLIGHT * share)
                actor.lonely = min(1.0, actor.lonely + WITNESS_SLIGHT * share)
        if "news" in self.flags and a in HELPED and v in self.people and v != q.name and ev.get("done"):
            # the mirror of reprisal: a witness warms to one who helps in view, and the helper gains standing
            q.aff[actor.name] = min(1, q.aff.get(actor.name, 0) + WITNESS_COST)
            if "drives" in self.flags:
                actor.slight = max(0.0, actor.slight - WITNESS_SLIGHT)
                actor.lonely = max(0.0, actor.lonely - WITNESS_SLIGHT)
        if "gossip" in self.flags and a in DEED and v in self.people and v != q.name and ev.get("done"):
            self._witness(q, actor.name, a, v, ev["id"])
        if a == "gossip" and v == q.name and "facts" in ev:
            f = ev["facts"]
            if self.distrust(q, actor.name) < 0.3 and f["who"] != q.name:
                # reputation: a bad deed told lowers q's liking of the doer, a
                # good one raises it; more when the victim is someone q likes
                sign = DEED[f["deed"]]
                w = 0.12 * (1 + (q.aff.get(f["victim"], 0) > 0.2))
                q.aff[f["who"]] = max(-1, min(1, q.aff.get(f["who"], 0) + w * sign))
                q.rep[f["who"]] = q.rep.get(f["who"], 0) + sign
                q.aff[actor.name] = min(1, q.aff.get(actor.name, 0) + 0.03)     # confided in
            elif f["who"] == q.name and DEED[f["deed"]] < 0:
                q.aff[actor.name] = max(-1, q.aff.get(actor.name, 0) - 0.1)
        if a == "gossip" and v != q.name and ev["facts"]["who"] == q.name and DEED[ev["facts"]["deed"]] < 0:
            q.aff[actor.name] = max(-1, q.aff.get(actor.name, 0) - 0.1)       # overheard being run down
        if "check" in self.flags and a == "hide" and thing in self.things:
            self._contradict(q, thing, actor.name, ev["id"])
            q.bel[("where", thing)] = (actor.name, ev["id"])
        if "check" in self.flags and a == "tell" and v == q.name and "facts" in ev:
            old = q.bel.get(("took", ev["facts"]["thing"]))
            if old and old[0] != ev["facts"]["who"] and old[1].startswith("e") and self._saw(q, old[1]):
                self._caught(q, actor.name, ev["facts"]["thing"], ev["id"], "it contradicts what I saw")

    def _provoked(self, q, actor, target, ev):
        """calm: witness q saw `actor` answer a wrong: the target had harmed the actor within PROVOKED_SPAN turns
        (q saw it), or, for a charge or a search, q itself believes the target took the thing. A witness
        condemns the one who began a quarrel, not the one who answers it."""
        if ev["action"] == "accuse":
            return q.bel.get(("took", ev["parts"].get("thing")), (None,))[0] == target
        if ev["action"] == "search" and any(k == "took" and v == target for (k, _), (v, _s) in q.bel.items()):
            return True
        return any(e["actor"] == target and e.get("target") == actor.name and e["action"] in HARMFUL and e.get("done")
                   and e["id"] in q.perceived for e in self.log[-80:] if e["t"] >= self.t - PROVOKED_SPAN)

    def _witness(self, q, actor, action, target, eid):
        q.deeds.append({"key": (self.night, eid), "actor": actor, "action": action, "target": target})
        if len(q.deeds) > 12:
            if "news" in self.flags:     # forget the oldest of the commoner kind, so a rare good deed is kept
                good = [x for x in q.deeds if DEED[x["action"]] > 0]
                bad = [x for x in q.deeds if DEED[x["action"]] <= 0]
                q.deeds.remove((good if len(good) > len(bad) else bad)[0])
            else:
                del q.deeds[0]

    def _caught(self, q, liar, thing, eid, why):
        """q finds that `liar` told it something false."""
        q.liars[liar] = q.liars.get(liar, 0) + 1
        q.memory.append({"night": self.night, "t": self.t, "what": "caught lying", "who": liar, "thing": thing,
                         "why": why})
        if "gossip" in self.flags:
            self._witness(q, liar, "lie", q.name, eid)

    def _checked(self, p, q, ev):
        """The searcher's own lesson (check flag): what it found, or that a
        tale it was told about q does not hold."""
        for n in ev.get("found", []):
            self._contradict(p, n, q.name, ev["id"])
            p.bel[("where", n)] = (q.name, ev["id"])
            if self.things[n]["owner"] not in (None, q.name):
                p.bel[("took", n)] = (q.name, ev["id"])
        for (k, n), (who, src) in list(p.bel.items()):
            if k != "took" or who != q.name or n in self.held(q) or (n, src) in p.doubted:
                continue
            src_ev = next((x for x in self.log if x["id"] == src), None)
            if src_ev and src_ev["action"] == "tell":
                p.doubted.add((n, src))
                del p.bel[("took", n)]
                p.aff[src_ev["actor"]] = max(-1, p.aff.get(src_ev["actor"], 0) - 0.2)
                self.note(p, "doubt", text=f"doubt that {q.name} took {n}", obj=n,
                          facts={"learns": f"{q.name} does not have {n}", "contradicts": src})

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
            self._caught(q, liar, thing, src, f"saw that {holder} has it")   # memory only; no draw
            self.note(q, "notice", text=f"notice that {holder} has {thing}",
                      facts={"learns": f"{holder} has {thing}", "contradicts": src})

    def _look(self, p, src):
        """p sees what lies in the room and what the others here hold openly."""
        for n, th in self.things.items():
            if th["at"] == p.room:
                p.bel[("where", n)] = (p.room, src)
        check = "check" in self.flags and src == "sight"
        for q in self.here(p):
            for n in self.held(q, visible_only=True):
                if check:                     # a tale is checked against who holds it openly
                    self._contradict(p, n, q.name, "sight")
                p.bel[("where", n)] = (q.name, src)
        if "drives" in self.flags:
            new = [n for n, th in self.things.items() if th["at"] == p.room and n not in p.glimpsed] + \
                  [q.name for q in self.here(p) if q.name not in p.glimpsed]
            if new:
                p.glimpsed.update(new)
                self._novel(p, 0.02 * len(new))
        for n in self.held(p):
            p.bel[("where", n)] = (p.name, src)
        if "known" in self.flags:
            here = {q.name for q in self.here(p)}
            for name in here:
                p.seen_at[name] = p.room
            for name in list(p.seen_at):
                if p.seen_at[name] == p.room and name not in here:
                    p.seen_at[name] = None           # gone, p does not know where
            for name, room in START_ROOM.items():    # not here, where the night started
                if room == p.room and name not in here and name != p.name and name not in p.seen_at:
                    p.seen_at[name] = None

    def _check_own(self, p):
        """p notices that something of its own is gone (under `known`:
        anything it believed it held)."""
        known = "known" in self.flags
        for n, th in self.things.items():
            if (known or th["owner"] == p.name) and th["at"] != p.name \
                    and p.bel.get(("where", n), (None,))[0] == p.name:
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
