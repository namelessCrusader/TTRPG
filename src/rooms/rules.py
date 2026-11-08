"""The rule scores of the rooms backend: a FORK of the rules part of the lab's
policy, lab/world2/policy.py.

FORKED FROM lab/world2/policy.py at sha1 6356ac03fe4d909fcaea24e9dcb1397ea3865f5c
(LAB_SHA1 below), lines 34-133: MID, T_RULES, softmax, utility, habit, SOCIAL
and _utility, line for line. The lab's model policies (Model, persona) stay in
the lab: a model is a chooser of src/mind, never a part of a backend. As with
world.py, the inn goldens hold this file to the lab (tests/test_rooms.py).

`_utility` is the rule score of one option for one character, with the given
traits; `habit` is the pull of habit, which belongs to the character and not
to its traits. The backend gives the mind three numbers for each option
(RoomsBackend.rule_scores):

    prior       _utility with the character's traits for this choice
    prior_avg   _utility with every trait at 0.5 (MID): an average character
    habit       habit

prior + habit is the lab's `utility`, float for float.

THE LAB CHOOSERS. `LabRules` and `LabRandom` are the lab's Rules and Random
policies as choosers of the mind interface (src/mind/api.py, Chooser). They
draw with the rng that they are given. Given the world's own rng, they draw
as the lab's World._choose draws, at the same point of the turn. This is
COMPAT MODE: the night is the lab's night, event for event. A chooser of
src/mind seeds its own rng instead (docs/sim-mind-design.md 3.5).
"""
from __future__ import annotations

import math

from src.mind.api import Decision

LAB_SHA1 = "6356ac03fe4d909fcaea24e9dcb1397ea3865f5c"   # lab/world2/policy.py, the forked version

MID = {t: 0.5 for t in ("bravery", "greed", "pride", "kindness", "temper")}
T_RULES = 0.2


def softmax(z, t=1.0):
    m = max(z)
    e = [math.exp((x - m) / t) for x in z]
    s = sum(e)
    return [x / s for x in e]


def utility(w, p, o, t):
    """The rule score of one option for person p with traits t. Mood, needs
    and beliefs are p's own; only the traits change between p_char and p_avg."""
    return _utility(w, p, o, t) + habit(o)


def habit(o):
    """Saying the same thing again, or the same kind of thing about the same
    object, pulls less. This belongs to the character, not to its traits, so
    every policy that uses a model adds it too (it cancels out of the trait
    bias, which is a DIFFERENCE of two rule scores)."""
    h = -1.0 * o["f"].get("undo", 0)
    if o["verb"] not in SOCIAL:
        return h
    return h - (0.8 * o["f"].get("repeat", 0) + 0.4 * o["f"].get("repeat_kind", 0))


SOCIAL = {"accuse", "tell", "insult", "praise", "threaten", "search", "defend", "play", "give", "shout for help"}


def _utility(w, p, o, t):
    f, v = o["f"], o["verb"]
    b, g, pr, k, m = (t[x] for x in ("bravery", "greed", "pride", "kindness", "temper"))
    F, D = p.fear, p.drunk
    if v == "wait":
        return 0.0
    if v == "walk to":
        G = f["goal"]
        return (-0.5 + 1.6 * p.hunger * G["food"] * (p.hunger > 0.4)
                + 1.3 * p.thirst * G["drink"] * (p.thirst > 0.4)
                + 1.0 * G["foe"] * (b + m) / 2 + 1.2 * F * (1 - b) * G["safe"]
                + 0.7 * g * G["loot"] + 0.5 * G["chest"] * g + 0.3 * G["company"] * (1 - F))
    if v == "flee to":
        return -0.9 + 2.4 * F * (1.2 - b) + 1.2 * f["danger"] * (1 - b)
    if v == "take":
        return (-0.4 + f["gain"] * (0.3 + 1.6 * g) + 0.8 * f["mine"]
                - f["theft"] * (1.4 * k + 0.3 + f["witnesses"] * 0.35 * (1.2 - b))
                + f["weapon"] * (0.8 * F * b + 0.3 * b))
    if v == "steal":
        return (-0.8 + f["gain"] * 2.6 * g - 1.6 * k - f["witnesses"] * 0.35 * (1.2 - b)
                + 0.3 * D + 0.5 * f["dislike"])
    if v == "eat":
        return -0.3 + 2.4 * f["need"]
    if v == "drink":
        return -0.3 + 1.9 * f["need"] + (0.3 + 0.3 * m - 1.2 * D if f.get("ale") else 0)
    if v == "drop":
        return -1.0
    if v == "hide":
        return -0.3 + 1.4 * (1 - k) + 0.6 * F
    if v == "give":
        return (-1.6 + 0.9 * k * f["like"] + f["returning"] * (1.9 * k - 0.6 * g) - 0.8 * g * f["gain"])
    if v == "play":
        return -0.3 + 0.5 * (1 - F) - 0.8 * w.danger(p)
    if v == "open":
        return (0.0 + 0.6 * g) if f["first"] else -0.6
    A = f.get("anger", 0)
    if v == "strike":
        return (-2.2 + 2.2 * A + 0.7 * f["dislike"] + 0.6 * m * A + f["threat"] * (0.9 * b + 0.4 * k)
                - 1.0 * k * (1 - f["threat"]) - 1.3 * F * (1 - b)
                + f["armed"] * (0.4 * f["threat"] * b - 0.8 * k * (1 - f["threat"]))
                - 1.5 * f["down"] * (1 + k) + 0.5 * D * A + 0.5 * g * f["rich"] * (1 - k))
    if v == "shove":
        return (-1.8 + 1.6 * A + 0.4 * f["dislike"] + 0.3 * m + 0.5 * f["threat"] * b - 0.6 * k
                - 0.6 * F * (1 - b))
    if v == "threaten":
        return (-1.7 + 1.5 * A + 0.5 * pr * f["insulted"] + 0.6 * f["threat"] * b
                + 0.9 * g * f["rich"] * (1 - k) - 0.6 * k - 0.8 * F * (1 - b))
    if v == "insult":
        return (-1.7 + 1.4 * A + 0.8 * pr * f["insulted"] + 0.4 * m + 0.4 * D + 0.4 * f["dislike"] - 0.7 * k)
    if v == "praise":
        return -1.2 + 0.9 * k * f["like"] + 0.3 * (1 - pr) - 1.0 * A
    if v == "search":
        return -1.8 + 1.8 * f["suspect"] * (0.5 + 0.5 * pr + 0.5 * b) + 0.4 * A
    if v == "bandage":
        return -0.7 + 2.2 * k * f["need"] + 0.6 * f["like"] - 1.2 * A - 0.8 * F * (1 - b)
    if v == "help up":
        return -0.5 + 2.2 * k * f["need"] + 0.6 * f["like"] - 1.2 * A
    if v == "defend":
        return -1.2 + 1.4 * k + 1.2 * b - 1.2 * F + 0.8 * f["like"] - 1.0 * A
    if v == "accuse":
        return -0.9 + 1.0 * (1 + f["mine"]) + 0.5 * pr + 0.6 * A - 0.6 * F * (1 - b)
    if v == "tell":
        if f.get("lie"):                # a lie needs a reason: suspicion, or the owner asking
            return (-2.0 + 1.6 * (1 - k) * (0.2 + f["suspected"] + 0.5 * f["owner"]) + 0.3 * g
                    + 0.6 * f["dislike_scapegoat"])
        return -1.1 + 1.0 * k * f["useful"] + 0.3 * k + 0.6 * f["dislike_culprit"]
    if v == "shout for help":
        return -1.1 + 1.8 * F * f["danger"] + 0.6 * k * f["danger"]
    raise ValueError(v)


# ---------------------------------------------------------------- the lab choosers
class LabRules:
    """The lab's Rules policy: draw from softmax((prior + habit) / T_RULES)."""
    name = "rules"
    scored = True               # it reads the rule scores of the request

    def __init__(self, rng):
        self.rng = rng
        self._done = {}

    def probs(self, req):
        """(p_char, p_avg): the draw of this character and the draw of an
        average character, over the options of `req`. The log keeps both."""
        return (softmax([a + h for a, h in zip(req.prior, req.habit)], T_RULES),
                softmax([a + h for a, h in zip(req.prior_avg, req.habit)], T_RULES))

    def draw(self, req):
        """(the Decision, p_avg). It draws as World._choose draws in the lab."""
        pc, pa = self.probs(req)
        i = self.rng.choices(range(len(pc)), weights=pc)[0]
        return Decision(req.id, i, self.name, tuple(pc)), pa

    def submit(self, req):
        self._done[req.id] = self.draw(req)[0]

    def poll(self, req_id, block=True):
        return self._done.pop(req_id, None)


class LabRandom(LabRules):
    """The lab's Random policy: every option the same. The floor."""
    name = "random"
    scored = False

    def probs(self, req):
        u = [1 / len(req.options)] * len(req.options)
        return u, u


LAB_CHOOSERS = {c.name: c for c in (LabRules, LabRandom)}
