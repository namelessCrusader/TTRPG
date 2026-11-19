"""How a character chooses: four policies, compared in the same world.

Each returns two draws over the same options:
    p_char   what THIS character draws from (its traits)
    p_avg    what an AVERAGE character would (every trait at 0.5)
The layer 3 surprise measure (lab/judge/signals.py) compares the two.

    random      every option the same. The floor.
    rules       no model: a score for each option from traits, mood, needs
                and what the character believes (utility AI). The
                literature's no-model baseline.
    text:M      model M reads the scene; personality is a sentence in it
                ("Tom is timid and kind"). The average character's scene has
                no such sentence. Two model calls per choice.
    appraise:M  model M reads the scene with no personality (the sense), and
                then answers two small questions with the persona sentence:
                "How does X feel?" and "What does X want to do?". The scene
                with that feeling and urge is ranked again, and the two are
                combined behind the plausibility guard (lab/methods/combine.py,
                A2): only options the sense already rates near the best can
                move. Four model calls per choice.
    bias:M      model M reads the scene with no personality, and the traits
                ADD a weight to each option (the rules' trait part). The model
                says what makes sense; the traits push some options up or
                down. One model call per choice. The weight could later be
                learned instead of written by hand.
"""
from __future__ import annotations

import math
import os
import time

from lab.world2.world import (BENEFIT_ACTS, LONELY_SET, LONELY_W, PARTNER_W, PHYSICAL, RESTRAINT_ACTS, WARM_ACTS,
                              WOUND_W)

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
    u = _utility(w, p, o, t) + habit(o)
    if w.flags & _NEW:
        u += flag_terms(w, p, o)
    return u


_NEW = frozenset(("partner", "calm"))


def flag_terms(w, p, o):
    """What the inn fix (world flags `partner`, `calm`) adds to the score of an act. The same terms for every
    character; none of them reads a trait (a trait is a dial, not a flag), so p_char and p_avg share them.
      partner  an act aimed at a person carries the trust in that person (world.trust): kind acts go to the
               trusted and are withheld from the distrusted; harm is held back toward the trusted.
      calm     the company drive: a lonely one (above its set point) is drawn to warm acts; the wounds: a hurt
               one is held back from blows and threats (the rules score read neither before)."""
    f, v = o["f"], o["verb"]
    u = 0.0
    if "partner" in w.flags and "trust" in f:
        if v in BENEFIT_ACTS:
            u += PARTNER_W * f["trust"]
        elif v in RESTRAINT_ACTS:
            u -= PARTNER_W * max(0.0, f["trust"])
    if "calm" in w.flags:
        if "drives" in w.flags and v in WARM_ACTS:
            u += LONELY_W * max(0.0, p.lonely - LONELY_SET)
        if v in PHYSICAL:                       # a wounded one does not pick a fight
            u -= WOUND_W * max(0.0, (10.0 - p.hp) / 10.0)
    return u


def habit(o):
    """Saying the same thing again, or the same kind of thing about the same
    object, pulls less. This belongs to the character, not to its traits, so
    every policy that uses a model adds it too (it cancels out of the trait
    bias, which is a DIFFERENCE of two rule scores)."""
    h = -1.0 * o["f"].get("undo", 0)
    if o["verb"] not in SOCIAL:
        return h
    return h - (0.8 * o["f"].get("repeat", 0) + 0.4 * o["f"].get("repeat_kind", 0))


SOCIAL = {"accuse", "tell", "insult", "praise", "threaten", "search", "defend", "play", "give", "shout for help",
          "gossip", "ask", "offer", "accept", "refuse", "look around"}


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
                + 0.7 * g * G["loot"] + 0.5 * G["chest"] * g + 0.3 * G["company"] * (1 - F)
                + 0.8 * p.boredom * G.get("new", 0))            # drives flag only (else + 0.0)
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
        return (-1.6 + 0.9 * k * f["like"] + f["returning"] * (1.9 * k - 0.6 * g) - 0.8 * g * f["gain"]
                + f.get("asked", 0) * (1.4 + 0.6 * k - 2.4 * f.get("distrust", 0)))   # trade flag only
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
        return (-0.9 + 1.0 * (1 + f["mine"]) + 0.5 * pr + 0.6 * A - 0.6 * F * (1 - b)
                - 1.5 * f.get("moot", 0.0))             # world flag `calm`: no charge about what was given back
    if v == "tell":
        if f.get("lie"):                # a lie needs a reason: suspicion, or the owner asking
            return (-2.0 + 1.6 * (1 - k) * (0.2 + f["suspected"] + 0.5 * f["owner"]) + 0.3 * g
                    + 0.6 * f["dislike_scapegoat"])
        return -1.1 + 1.0 * k * f["useful"] + 0.3 * k + 0.6 * f["dislike_culprit"]
    if v == "shout for help":
        return -1.1 + 1.8 * F * f["danger"] + 0.6 * k * f["danger"]
    # ---- the acts the world's flags add (lab/world2/world.py, FLAGS)
    if v == "sit down":
        return -0.7 + 1.6 * f["fatigue"] - 0.8 * f["danger"]
    if v == "sleep":
        return (-1.6 + 3.0 * max(0.0, f["fatigue"] - 0.4) - 2.0 * f["danger"] - 1.5 * f["distrust"] * (1 - b)
                - 0.2 * f["others"])
    if v == "look around":
        return -0.6 + 1.6 * f["boredom"] * f["unseen"] + 0.2 * f["boredom"]
    if v == "gossip" and "sal" in f:        # world flag `news`: worth telling by salience, not by sign
        return -1.4 + 2.6 * f["sal"] + 0.4 * f["like"] - 0.4 * f["dislike"]
    if v == "gossip":
        return (-1.5 + f["bad"] * (0.8 * f["dislike_actor"] + 0.4 * (1 - k) + 0.3)
                + f["good"] * (0.6 * f["like_actor"] + 0.3 * k) + 0.4 * f["like"] - 0.4 * f["dislike"])
    if v == "ask":
        return -1.0 + 2.2 * f["need"] + 0.4 * f["like"] + 0.6 * g * f["gain"] - 1.2 * f["distrust"] - 0.3 * pr
    if v == "offer":
        return -1.2 + 2.0 * f["need"] + (f["gain"] - f["cost"]) * (0.5 + 2 * g) + 0.3 * f["like"] - f["distrust"]
    if v == "accept":
        return -0.5 + (f["get"] - f["gain"]) * (0.5 + 2 * g) + 0.4 * f["like"] + 0.5 * k - 2.2 * f["distrust"]
    if v == "refuse":
        return -0.6 + 2.2 * f["distrust"] + 0.4 * f["dislike"] - 0.5 * k
    # ---- world flag `meals`
    if v == "serve":       # sharing costs a fed giver little: only its own hunger and a short pot weigh against it
        return (-1.3 + 2.0 * f["seen_need"] + f["asked"] * (1.4 + 0.6 * k) + 0.8 * k * f["like"]
                - 2.0 * f["distrust"] - 1.2 * max(0.0, f["own"] - 0.45) - 0.8 * (1 - f["left"]) ** 2)
    if v == "buy":
        return -0.9 + 2.2 * f["need"] - f["cost"] * (0.5 + 2 * g) - f["distrust"]
    if v == "work":
        return -1.1 + 2.2 * f["need"] - 1.2 * f["fatigue"] - 0.5 * f["distrust"]
    raise ValueError(v)


class Random:
    name = "random"

    def probs(self, w, p, opts):
        u = [1 / len(opts)] * len(opts)
        return u, u


class Rules:
    name = "rules"

    def probs(self, w, p, opts):
        pc = softmax([utility(w, p, o, p.traits) for o in opts], T_RULES)
        pa = softmax([utility(w, p, o, MID) for o in opts], T_RULES)
        return pc, pa


ADJ = {"bravery": ("timid", "brave"), "greed": ("generous", "greedy"), "pride": ("humble", "proud"),
       "kindness": ("cruel", "kind"), "temper": ("calm", "hot-tempered")}


def persona(p):
    """The traits as a sentence: only the ones that are clearly low or high."""
    words = [ADJ[t][0] if v < 0.35 else ADJ[t][1] for t, v in p.traits.items() if v < 0.35 or v > 0.65]
    if not words:
        return f"{p.name} is an ordinary sort."
    return f"{p.name} is " + (", ".join(words[:-1]) + " and " + words[-1] if len(words) > 1 else words[0]) + "."


def _z(scores):
    """Model scores as log-weights, then to the same spread for every choice."""
    if all(0 <= x <= 1 for x in scores):
        scores = [math.log(max(x, 1e-9)) for x in scores]
    mu = sum(scores) / len(scores)
    sd = math.sqrt(sum((x - mu) ** 2 for x in scores) / len(scores)) or 1.0
    return [(x - mu) / sd for x in scores]


class Model:
    """A Workstream A scorer as a character's chooser. TAU sets how sharply it
    prefers its top options (in units of the spread of its own scores)."""
    TAU = 0.5

    def __init__(self, mode, scorer, beta=1.0):
        from lab.methods.score import make
        self.mode, self.scorer_name, self.beta = mode, scorer, beta
        self.name = f"{mode}:{scorer}"
        self.s = make(scorer)
        self.calls, self.ms = 0, 0.0

    def load(self):
        import torch
        from lab.methods.run import gpu_check
        gpu_check(self.s.params)
        self.s.load()
        torch.cuda.reset_peak_memory_stats()

    def _ask(self, text, question, labels):
        t = time.perf_counter()
        r = self.s.rank(text, question, labels)
        dt = time.perf_counter() - t
        self.calls += 1
        self.ms += dt * 1000
        duty = float(os.environ.get("LAB_DUTY", "0.5"))
        if duty < 1:
            time.sleep(dt * (1 / duty - 1))
        return labels[r["pick"]]

    def _rank(self, text, p, opts):
        t = time.perf_counter()
        r = self.s.rank(text, f"What does {p.name} do next?", [o["text"] for o in opts])
        dt = time.perf_counter() - t
        self.calls += 1
        self.ms += dt * 1000
        # rest, as the Workstream A runner does, so the laptop stays cool
        duty = float(os.environ.get("LAB_DUTY", "0.5"))
        if duty < 1:
            time.sleep(dt * (1 / duty - 1))
        return _z(r["scores"])

    def probs(self, w, p, opts):
        if len(opts) == 1:
            return [1.0], [1.0]
        hab = [habit(o) / T_RULES for o in opts]
        z0 = self._rank(w.describe(p), p, opts)
        pa = softmax([z / self.TAU + h for z, h in zip(z0, hab)])
        if self.mode == "text":
            z1 = self._rank(w.describe(p, persona(p)), p, opts)
            pc = softmax([z / self.TAU + h for z, h in zip(z1, hab)])
        elif self.mode == "appraise":
            # the persona answers two small questions; the answers, not the
            # persona, go into the scene; the guard keeps the choice sensible
            g, d, hard = GUARD
            sit = w.describe(p, persona(p))
            feel = self._ask(sit, f"How does {p.name} feel?", FEEL)
            urge = self._ask(sit, f"What does {p.name} want to do?", URGE)
            z1 = self._rank(f"{w.describe(p)} {p.name} feels {feel}. {p.name} wants to {urge}.", p, opts)
            l0 = [math.log(x) for x in softmax([z / self.TAU for z in z0])]
            l1 = [math.log(x) for x in softmax([z / self.TAU for z in z1])]
            top = max(l0)
            keep = [x >= top + math.log(d) if d > 0 else True for x in l0]
            z = [(a + g * (b - a)) if k else (a if not hard else -1e9) for a, b, k in zip(l0, l1, keep)]
            pc = softmax([x + h for x, h in zip(z, hab)])
            self.last = (feel, urge)
        else:
            # the traits' part of the rule score, added to the model's view
            bias = [(utility(w, p, o, p.traits) - utility(w, p, o, MID)) / T_RULES for o in opts]
            pc = softmax([z / self.TAU + self.beta * bb + h for z, bb, h in zip(z0, bias, hab)])
        return pc, pa


FEEL = ["angry", "afraid", "calm", "sad", "glad"]
URGE = ["stand up to them", "get away from here", "call for someone", "keep out of it", "help whoever is hurt",
        "get something out of this"]
# the guard (g, d, hard): lab/results/methods/combine.md, A2 (g 2, d 0.5, hard kept valid picks near 70%)
GUARD = tuple(float(x) if i < 2 else x == "hard" for i, x in
              enumerate(os.environ.get("LAB_GUARD", "2,0.5,hard").split(",")))


def make_policy(name):
    if name == "random":
        return Random()
    if name == "rules":
        return Rules()
    mode, scorer = name.split(":", 1)
    return Model(mode, scorer)
