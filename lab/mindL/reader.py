"""The frozen reader: the engine's own option features, turned into numbers
(learned-mind plan, section 4; the user's answer 4: o["f"], the verb, the
target's identity; GLiNER features only later).

Nothing here learns, and nothing here chooses. Every chooser that is compared
(the mind, the bandit, PPO) reads the SAME numbers:

    verb      one of VERBS (every act the world can list, flags included)
    target    one of TARGETS: a cast member, the bandit, a room, or none
    z         [1, the numeric keys of o["f"] (KEYS), the chooser's own drive
              needs (DRIVES)]: ZDIM numbers

The linear learners cross z with the verb (one weight row per verb) and add a
weight per (verb, target). PPO gets the same numbers flat (`raw`): the verb
and target one-hot, then z without its bias; its network finds the crossings.

String and dict fields of o["f"] that are not numbers are left out: "said"
(a name, already in the target or the scapegoat's dislike), "pay" (its value
is "cost"), "deed" (its sign is "bad"/"good"). Any other key that is not in
KEYS is recorded in UNKNOWN (the tests check that it stays empty).
"""
from __future__ import annotations

import numpy as np

from lab.world2.world import CAST, DRIVES, DUR, ROOMS

VERBS = tuple(sorted(DUR))
VI = {v: i for i, v in enumerate(VERBS)}
PEOPLE = tuple(n for n, _, _, _ in CAST) + ("the bandit",)
TARGETS = PEOPLE + ("room", "none")
TI = {t: i for i, t in enumerate(TARGETS)}
GOAL = ("food", "drink", "foe", "loot", "safe", "company", "chest", "new")
KEYS = ("ale", "anger", "armed", "asked", "attacked", "bad", "boredom", "cost", "danger", "dislike",
        "dislike_actor", "dislike_culprit", "dislike_scapegoat", "distrust", "down", "fatigue", "first", "gain",
        "get", "good", "insulted", "lie", "like", "like_actor", "mine", "need", "others", "owner", "repeat",
        "repeat_kind", "returning", "rich", "stolen", "suspect", "suspected", "theft", "threat", "truth", "undo",
        "unseen", "useful", "weapon", "witnesses") + tuple("goal_" + g for g in GOAL)
KI = {k: 1 + i for i, k in enumerate(KEYS)}          # column in z (0 is the bias)
SKIP = {"said", "pay", "deed"}
COUNTS = {"witnesses", "others", "repeat", "repeat_kind"}     # counts: min(x, 3) / 3
NK, ND = len(KEYS), len(DRIVES)
ZDIM = 1 + NK + ND
NV, NT = len(VERBS), len(TARGETS)
RAW = NV + NT + NK + ND
UNKNOWN: set = set()


class Menu:
    """One choice's options as arrays: verb index, target index, z rows."""
    __slots__ = ("v", "t", "Z")

    def __init__(self, v, t, Z):
        self.v, self.t, self.Z = v, t, Z

    def __len__(self):
        return len(self.v)


def target_name(target):
    """A person's name, 'room', or 'none'."""
    if target is None:
        return "none"
    if target in ROOMS:
        return "room"
    return target if target in TI else "none"


def _num(k, x):
    if k in COUNTS:
        return min(float(x), 3.0) / 3.0
    return max(-2.0, min(2.0, float(x)))


def encode(opts, need):
    """The options as a Menu. `need` is the chooser's drive needs (DRIVES order)."""
    n = len(opts)
    v = np.empty(n, dtype=np.int64)
    t = np.empty(n, dtype=np.int64)
    Z = np.zeros((n, ZDIM))
    Z[:, 0] = 1.0
    Z[:, 1 + NK:] = need
    for i, o in enumerate(opts):
        v[i] = VI[o["verb"]]
        t[i] = TI[target_name(o["target"])]
        for k, x in o["f"].items():
            if k == "goal":
                for g, y in x.items():
                    Z[i, KI["goal_" + g]] = float(y)
            elif k in KI:
                Z[i, KI[k]] = _num(k, x)
            elif k not in SKIP:
                UNKNOWN.add(k)
    return Menu(v, t, Z)


def raw(menu):
    """The same numbers, flat for a network: verb one-hot, target one-hot, z[1:]."""
    n = len(menu)
    X = np.zeros((n, RAW), dtype=np.float32)
    X[np.arange(n), menu.v] = 1.0
    X[np.arange(n), NV + menu.t] = 1.0
    X[:, NV + NT:] = menu.Z[:, 1:]
    return X


def key(o):
    """What an act is, for habit, curiosity and episodes: (verb, whom, where
    or what): the target if there is one, else the thing."""
    return (o["verb"], o["target"] if o["target"] is not None else o["obj"])


def cue(o):
    """The person, place or thing an option is about (for valence): a person
    target, the room walked to, else the thing acted on."""
    tg = o["target"]
    if tg in PEOPLE:
        return tg
    if tg in ROOMS:
        return tg
    return o["obj"]
