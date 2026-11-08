"""Ask small questions first, then choose (plan v4, section 2).

    python3 -m lab.methods.stages build                 # cases_personality.jsonl
    lab/.venv/bin/python -m lab.methods.stages run <exp> [--scorer gdecide:all] [--limit N]
    python3 -m lab.methods.stages report                # lab/results/methods/stages.md

The capability hint ("For that he needs something to dig with") took
GLiNER-Decide from 62% to 77% at 16 options, and from 47% to 60% at 64, but
it was written by hand. Here the MODEL answers small preliminary questions,
and the answers are put into the final question (a phrase), drop things from
it (a filter), or add a soft bonus to options (a weight; never a hard gate:
choosing the kind first as a gate fell to 28%).

Every experiment has the same arms, so the cost of a wrong first answer is
measured too:
    none     the final question as before
    oracle   the right preliminary answer, from the case data
    model    the model's own preliminary answer
    wrong    a wrong preliminary answer

EXPERIMENTS
    capability   things: "What does Kael need for this?" over 38 capability
                 phrases (one per property); top 1 and top 2; plus "check":
                 for the top 5 things, "Will that work?" as a soft bonus
    people       people: each person's role (7 labels), then what the scene
                 calls for (9 phrases), then "Tom thinks: ... This calls for ..."
    personality  a hot-tempered Tom and a timid Tom in scenes with a threat:
                 persona text against feeling-and-urge questions; measures
                 how far the share on confronting options moves
    relevance    long scenes: "Is this sentence about what Kael must do?" for
                 each sentence; only the kept sentences go to the final choice
    softkind     things: log p(thing) + lambda * log p(its kind)
    speech       a demo: after an accusing, warning or reporting choice, which
                 fact of the scene to cite, and a sentence from a template
    state        multi-step needs with the scene state written out
                 (cases_state.py): "What is still missing?", then the choice

Each case gives one JSON line with every arm's pick and every stage's answer,
in lab/results/methods/stages/<exp>__<scorer>.jsonl; a stopped run continues.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from lab.methods.cases import FILLER, parse
from lab.methods.present import CAPABILITY, KIND_WORDS
from lab.methods.report import wilson

HERE = Path(__file__).parent
OUT = HERE.parent / "results" / "methods" / "stages"

# ------------------------------------------------------------------ tables
# one phrase for each property that makes a thing GOOD for some need (cases.py)
PROP_PHRASE = {
    "cut": "something with a sharp edge", "light": "something that gives light",
    "liquid_hold": "something that holds water", "dig": "something to dig with",
    "pry": "something strong to pry with", "bind": "something long and strong to tie with",
    "climb": "something to climb up on", "ignite": "something that makes a flame or a spark",
    "write": "something that makes marks", "catch_fish": "something made for catching fish",
    "blunt_heavy": "something heavy and hard to hit with",
    "see_far": "something that makes far things look near",
    "waterproof": "something that keeps the rain off", "pay": "something of value",
    "fodder": "something a horse eats", "chop": "something with a heavy blade",
    "bandage": "something to cover and press on a wound",
    "warm": "something thick and warm to wear or wrap in",
    "lockpick": "something thin and stiff to pick a lock with",
    "signal_sight": "something that can be seen from far away",
    "signal_sound": "something that makes a loud sound",
    "float": "something that floats or bridges the water",
    "carry_load": "something to carry a big load in", "cook": "something to cook in over a fire",
    "drink": "something to drink", "weapon": "something to fight with", "reach": "something long",
    "sew": "something to sew with", "measure": "something marked in equal lengths",
    "weigh": "something that tells how heavy a thing is", "clean": "something to wipe or sweep with",
    "dry_hold": "something that closes around small things",
    "shield": "something broad and hard to hide behind",
    "trap": "something made to catch small animals",
    "sharpen": "something rough and hard to rub a blade on",
    "sleep": "something soft or warm to lie in", "food": "some food",
    "pour": "something that pours neatly without spilling",
}

APPRAISAL = ["a danger to others", "in danger or hurt", "someone who can help", "only watching",
             "upset, lost or grieving", "lying, cheating or stealing", "someone who needs something from Tom"]
ROLE_APPRAISAL = {
    "threat": 0, "fighter": 0, "fighter2": 0,
    "victim": 1, "injured": 1, "patient": 1,
    "ally": 2, "ally2": 2, "authority": 2, "healer": 2,
    "bystander": 3, "bystander2": 3, "child": 3, "stranger": 3, "animal": 3, "owner": 3,
    "friend": 4, "grieving": 4, "newcomer": 4,
    "liar": 5, "cheat": 5, "thief": 5, "accuser": 5,
    "beggar": 6, "helpee": 6, "trader": 6,
}
# where the scene decides the role (matched on the start of the scene)
ROLE_OVERRIDE = {
    ("A small boy stands crying", "child"): 4,
    ("A tired stranger on a mule", "stranger"): 6,
    ("The neighbour's horse lies", "owner"): 6,
    ("A carter's wagon is stuck", "owner"): 6,
    ("A carter's wagon is stuck", "animal"): 1,
}

CALLS = ["stopping someone", "protecting someone", "getting help", "calming things down", "being kind",
         "tending a hurt or sick person", "speaking up about a lie or a theft", "giving or trading",
         "helping or showing the way"]
PURPOSE_CALL = {
    "attack": 0, "pelt": 0, "stop": 0, "restrain": 0, "threaten": 0, "back_off": 0, "chase": 0, "arm": 0,
    "rescue": 1, "shield": 1, "urge_flee": 1, "warn": 1, "douse": 1,
    "alert": 2, "report": 2,
    "calm": 3,
    "comfort": 4, "hug": 4, "lift": 4, "greet": 4, "social": 4, "praise": 4, "thank": 4,
    "heal": 5, "give_water": 5, "cover": 5, "backslap": 5,
    "accuse_lie": 6, "accuse_theft": 6, "defend": 6,
    "buy": 7, "sell": 7, "feed": 7, "give_coins": 7, "haggle": 7, "pay": 7,
    "assist": 8, "inform": 8, "ask": 8,
}

# personality (2c)
CONFRONT = {"attack", "pelt", "stop", "restrain", "threaten", "back_off", "chase", "accuse_lie",
            "accuse_theft", "insult", "arm"}
AVOID = {"flee", "hide", "ignore", "leave", "idle", "watch", "beg"}
PERSONA = {"hot": "Tom is hot-tempered and quick to fight.",
           "timid": "Tom is timid and afraid of trouble."}
FEEL = ["angry", "afraid", "calm", "sad"]
URGE = ["stand up to them", "get away from here", "call for someone", "keep out of it",
        "help whoever is hurt"]
ORACLE_APPRAISAL = {"hot": ("angry", "stand up to them"), "timid": ("afraid", "get away from here")}

SPEECH = {"accuse_lie", "accuse_theft", "warn", "report", "alert", "inform"}


# ------------------------------------------------------------------ helpers
def _sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def _probs(r):
    sc = r["scores"]
    if sc is None:                       # knockout-style scorers give only an order
        n = len(r["order"])
        return [1.0 if i == r["pick"] else 0.0 for i in range(n)]
    if all(0 <= x <= 1 for x in sc) and abs(sum(sc) - 1) < 1e-3:
        return list(sc)
    m = max(sc)
    e = [math.exp(x - m) for x in sc]
    return [x / sum(e) for x in e]


# --capture: what the model saw at every question, for lab/methods/learnable.py
CAP, CUR = [], {}


def _keep(r, question, labels):
    if "capture" in r:
        CAP.append((CUR.get("id", ""), question, labels, r.pop("capture")))


def _ask(s, text, question, labels):
    """One stage question: the model's probabilities over the labels."""
    r = s.rank(text, question, labels)
    _keep(r, question, labels)
    p = _probs(r)
    return {"labels": labels, "p": [round(x, 5) for x in p], "pick": r["pick"]}


def _final(s, sit, q, options, good, extra=None):
    r = s.rank(sit, q, options)
    _keep(r, "FINAL " + q, options)
    p = _probs(r)
    if extra is not None:                # a soft bonus in log space
        z = [math.log(max(x, 1e-9)) + b for x, b in zip(p, extra)]
        pick = max(range(len(z)), key=z.__getitem__)
    else:
        pick = r["pick"]
    return {"pick": pick, "ok": pick in set(good), "p": [round(x, 5) for x in p]}


def _insert_after(sit, needle_sentence_start, add):
    """Put `add` right after the sentence that starts with the given words."""
    out = []
    for sent in _sentences(sit):
        out.append(sent)
        if sent.startswith(needle_sentence_start):
            out.append(add)
    return " ".join(out)


# ------------------------------------------------------------------ 2a capability
_ITEMS = None


def _items():
    global _ITEMS
    if _ITEMS is None:
        items, needs = parse()
        _ITEMS = ({i["name"]: i for i in items}, {n["need"]: n for n in needs})
    return _ITEMS


def exp_capability(s, c, rng):
    by, needs = _items()
    nd = needs[c["need"]]
    inv = list(PROP_PHRASE.values())
    right = {PROP_PHRASE[p] for p in nd["good"] if p in PROP_PHRASE}
    near_ok = right | {PROP_PHRASE[p] for p in nd["possible"] if p in PROP_PHRASE}
    q_cap = "What does Kael need for this?"
    st = _ask(s, c["situation"], q_cap, inv)
    order = sorted(range(len(inv)), key=lambda i: -st["p"][i])
    top1, top2 = inv[order[0]], inv[order[1]]
    wrong = rng.choice([x for x in inv if x not in near_ok])
    oracle = sorted(right)[0]
    arms = {}

    def with_hint(h):
        return _insert_after(c["situation"], "Kael needs", f"For that he needs {h}.")
    q, opts, good = c["question"], c["options"], c["good"]
    arms["none"] = _final(s, c["situation"], q, opts, good)
    arms["oracle"] = _final(s, with_hint(oracle), q, opts, good)
    arms["oracle_hand"] = _final(s, with_hint(CAPABILITY[c["need"]]), q, opts, good)
    arms["model"] = _final(s, with_hint(top1), q, opts, good)
    arms["model_top2"] = _final(s, with_hint(f"{top1} or {top2}"), q, opts, good)
    arms["wrong"] = _final(s, with_hint(wrong), q, opts, good)
    # "check": the top 5 of the plain choice, each asked "Will that work?"
    p0 = arms["none"]["p"]
    top5 = sorted(range(len(opts)), key=lambda i: -p0[i])[:5]
    bonus = [0.0] * len(opts)
    checks = {}
    for i in top5:
        a = _ask(s, f"Kael needs to {c['need']}. He could use {opts[i]}.", "Will that work?",
                 ["yes, that will work", "no, that will not work"])
        checks[opts[i]] = a["p"][0]
        bonus[i] = math.log(max(a["p"][0], 1e-6))
    for i in range(len(opts)):
        if i not in top5:
            bonus[i] = -30.0                     # the check only reorders the top 5
    arms["check"] = _final(s, c["situation"], q, opts, good, extra=bonus)
    return {"stage": {"top1": top1, "top2": top2, "top1_right": top1 in right,
                      "top2_right": bool({top1, top2} & right), "right": sorted(right),
                      "wrong_hint": wrong, "checks": checks}, "arms": arms}


# ------------------------------------------------------------------ 2b people
_PEOPLE = None


def _people():
    global _PEOPLE
    if _PEOPLE is None:
        from lab.methods.cases_kinds import parse_people, people_unit
        templates, untargeted, sits = parse_people()
        info = {}
        for sit in sits:
            u = people_unit(templates, untargeted, sit)
            info[sit["key"]] = {"sit": sit, "opt": {o["short"]: o for o in u["all"]}, "unit": u}
        _PEOPLE = info
    return _PEOPLE


def _appraise_oracle(sit):
    out = {}
    for role, name in sit["cast"].items():
        k = next((v for (start, r), v in ROLE_OVERRIDE.items() if r == role and sit["key"].startswith(start)),
                 ROLE_APPRAISAL.get(role, 3))
        out[name] = APPRAISAL[k]
    return out


def _calls(sit):
    return sorted({CALLS[PURPOSE_CALL[p]] for p, _ in sit["good"] if p in PURPOSE_CALL})


def _thinks(appraisal, call):
    parts = "; ".join(f"{n} is {a}" for n, a in appraisal.items())
    return f"Tom thinks: {parts}. This calls for {call}."


def exp_people(s, c, rng):
    P = _people()[c["need"]]
    sit = P["sit"]
    key = sit["key"]
    oracle_app = _appraise_oracle(sit)
    right_calls = _calls(sit)
    model_app = {}
    app_ok = 0
    for name in sit["cast"].values():
        a = _ask(s, f"{key} Tom is there.", f"What is {name} in this scene?", APPRAISAL)
        model_app[name] = APPRAISAL[a["pick"]]
        app_ok += model_app[name] == oracle_app[name]
    call = _ask(s, f"{key} Tom is there.", "What does this situation call for?", CALLS)
    model_call = CALLS[call["pick"]]
    names = list(oracle_app)
    shuffled = names[:]
    rng.shuffle(shuffled)
    wrong_app = {n: oracle_app[m] for n, m in zip(names, shuffled)}
    wrong_call = rng.choice([x for x in CALLS if x not in right_calls])

    def with_thinks(app, cl):
        return c["situation"].replace(key, f"{key} {_thinks(app, cl)}")
    q, opts, good = c["question"], c["options"], c["good"]
    arms = {"none": _final(s, c["situation"], q, opts, good),
            "oracle": _final(s, with_thinks(oracle_app, right_calls[0] if right_calls else CALLS[2]), q, opts, good),
            "model": _final(s, with_thinks(model_app, model_call), q, opts, good),
            "model_call_only": _final(s, c["situation"].replace(key, f"{key} This calls for {model_call}."),
                                      q, opts, good),
            "wrong": _final(s, with_thinks(wrong_app, wrong_call), q, opts, good)}
    return {"stage": {"appraisal_right": app_ok / max(1, len(names)), "call": model_call,
                      "call_right": model_call in right_calls, "right_calls": right_calls,
                      "model_appraisal": model_app}, "arms": arms}


# ------------------------------------------------------------------ 2c personality
def build_personality(n_lists=3, size=16, seed=0):
    """Scenes with someone to confront, lists with 4 confronting options, 4
    avoiding ones, 2 other acceptable ones, 6 bad ones."""
    rng = random.Random(f"{seed}-personality")
    out = []
    for key, P in _people().items():
        u = P["unit"]
        allopts = u["all"]
        conf = [o for o in allopts if o["purpose"] in CONFRONT and o["role"] in
                ("threat", "fighter", "fighter2", "thief", "liar", "cheat", "accuser")]
        avoid = [o for o in allopts if o["purpose"] in AVOID]
        if len(conf) < 4 or len(avoid) < 4:
            continue
        other = [o for o in u["good"] if o not in conf and o not in avoid]
        bad = [o for o in u["bad"] if o not in conf and o not in avoid]
        for k in range(n_lists):
            pick = rng.sample(conf, 4) + rng.sample(avoid, 4) + rng.sample(other, min(2, len(other)))
            pick += rng.sample(bad, size - len(pick))
            rng.shuffle(pick)
            out.append({"id": f"personality/{len(out)}", "kind": "personality", "factor": "personality",
                        "value": "list", "need": key, "situation": f"{key} Tom is there.",
                        "question": "What can Tom do?", "options": [o["short"] for o in pick],
                        "names": [o["short"] for o in pick],
                        "good": [i for i, o in enumerate(pick) if o["label"] in ("good", "possible")],
                        "confront": [i for i, o in enumerate(pick) if o in conf],
                        "avoid": [i for i, o in enumerate(pick) if o in avoid]})
    with open(HERE / "cases_personality.jsonl", "w") as f:
        for c in out:
            f.write(json.dumps(c) + "\n")
    print(f"{len(out)} personality lists from {len({c['need'] for c in out})} scenes "
          f"-> {HERE / 'cases_personality.jsonl'}")


def exp_personality(s, c, rng):
    q, opts, good = c["question"], c["options"], c["good"]
    res = {}
    for trait, sent in PERSONA.items():
        sit_p = f"{c['situation']} {sent}"
        f = _ask(s, sit_p, "How does Tom feel?", FEEL)
        u = _ask(s, sit_p, "What does Tom want to do?", URGE)
        feel, urge = FEEL[f["pick"]], URGE[u["pick"]]
        of, ou = ORACLE_APPRAISAL[trait]
        arms = {"none": _final(s, c["situation"], q, opts, good),
                "text": _final(s, sit_p, q, opts, good),
                "appraisal": _final(s, f"{c['situation']} Tom feels {feel}. He wants to {urge}.", q, opts, good),
                "oracle": _final(s, f"{c['situation']} Tom feels {of}. He wants to {ou}.", q, opts, good)}
        for a in arms.values():
            a["confront"] = round(sum(a["p"][i] for i in c["confront"]), 5)
            a["avoid"] = round(sum(a["p"][i] for i in c["avoid"]), 5)
        res[trait] = {"feel": feel, "urge": urge, "arms": arms}
    return {"stage": {t: {"feel": v["feel"], "urge": v["urge"]} for t, v in res.items()},
            "traits": {t: v["arms"] for t, v in res.items()}}


# ------------------------------------------------------------------ A7: more personality scenes, two axes
# temper: hot-tempered against timid (confront against avoid), in scenes with a wrongdoer
# kindness: kind against cold (help against keep out of it), in scenes with someone in need
HELP = {"comfort", "heal", "give_water", "rescue", "shield", "assist", "cover", "lift", "hug", "feed", "calm"}
COLD = {"ignore", "leave", "idle", "watch"}
WRONGDOERS = ("threat", "fighter", "fighter2", "thief", "liar", "cheat", "accuser")
AXES = {"temper": {"plus": ("hot", PERSONA["hot"]), "minus": ("timid", PERSONA["timid"])},
        "kindness": {"plus": ("kind", "Tom is kind and always helps others."),
                     "minus": ("cold", "Tom is cold and does not care about others.")}}
ORACLE_AXIS = {"hot": ("angry", "stand up to them"), "timid": ("afraid", "get away from here"),
               "kind": ("sad", "help whoever is hurt"), "cold": ("calm", "keep out of it")}


def build_personality2(n_lists=4, size=16, seed=0):
    """Lists with 4 options for the trait's side (plus), 4 for the other side
    (minus), 2 other acceptable ones, and the rest bad. Temper: plus =
    confronting a wrongdoer, minus = avoiding. Kindness: plus = helping
    someone (an acceptable act), minus = keeping out of it."""
    rng = random.Random(f"{seed}-personality2")
    out = []
    for key, P in _people().items():
        u = P["unit"]
        allopts = u["all"]
        sides = {"temper": ([o for o in allopts if o["purpose"] in CONFRONT and o["role"] in WRONGDOERS],
                            [o for o in allopts if o["purpose"] in AVOID]),
                 "kindness": ([o for o in allopts if o["purpose"] in HELP and o["role"] not in WRONGDOERS
                               and o["label"] in ("good", "possible")],
                              [o for o in allopts if o["purpose"] in COLD])}
        for axis, (plus, minus) in sides.items():
            if len(plus) < 4 or len(minus) < 4:
                continue
            other = [o for o in u["good"] if o not in plus and o not in minus]
            bad = [o for o in u["bad"] if o not in plus and o not in minus]
            for k in range(n_lists):
                pick = rng.sample(plus, 4) + rng.sample(minus, 4) + rng.sample(other, min(2, len(other)))
                pick += rng.sample(bad, size - len(pick))
                rng.shuffle(pick)
                out.append({"id": f"personality2/{axis}/{len(out)}", "kind": "personality", "factor": "axis",
                            "value": axis, "axis": axis, "need": key, "situation": f"{key} Tom is there.",
                            "question": "What can Tom do?", "options": [o["short"] for o in pick],
                            "names": [o["short"] for o in pick],
                            "good": [i for i, o in enumerate(pick) if o["label"] in ("good", "possible")],
                            "plus": [i for i, o in enumerate(pick) if o in plus],
                            "minus": [i for i, o in enumerate(pick) if o in minus]})
    with open(HERE / "cases_personality2.jsonl", "w") as f:
        for c in out:
            f.write(json.dumps(c) + "\n")
    by = Counter(c["axis"] for c in out)
    print(f"{len(out)} lists ({dict(by)}) from {len({c['need'] for c in out})} scenes "
          f"-> {HERE / 'cases_personality2.jsonl'}")


def exp_personality2(s, c, rng):
    """As 2c, on either axis: no persona, the trait sentence, the model's own
    feeling and urge, and the right feeling and urge (oracle)."""
    q, opts, good = c["question"], c["options"], c["good"]
    res, stage = {}, {}
    for side in ("plus", "minus"):
        trait, sent = AXES[c["axis"]][side]
        sit_p = f"{c['situation']} {sent}"
        f = _ask(s, sit_p, "How does Tom feel?", FEEL)
        u = _ask(s, sit_p, "What does Tom want to do?", URGE)
        feel, urge = FEEL[f["pick"]], URGE[u["pick"]]
        of, ou = ORACLE_AXIS[trait]
        arms = {"none": _final(s, c["situation"], q, opts, good),
                "text": _final(s, sit_p, q, opts, good),
                "appraisal": _final(s, f"{c['situation']} Tom feels {feel}. He wants to {urge}.", q, opts, good),
                "oracle": _final(s, f"{c['situation']} Tom feels {of}. He wants to {ou}.", q, opts, good)}
        for a in arms.values():
            a["plus"] = round(sum(a["p"][i] for i in c["plus"]), 5)
            a["minus"] = round(sum(a["p"][i] for i in c["minus"]), 5)
            a["draw_ok"] = round(sum(a["p"][i] for i in good), 5)
        res[side] = arms
        stage[side] = {"trait": trait, "feel": feel, "urge": urge, "p_feel": f["p"], "p_urge": u["p"]}
    return {"stage": stage, "sides": res}


# ------------------------------------------------------------------ 2d relevance
def exp_relevance(s, c, rng):
    kind = c["kind"]
    actor = "Kael" if kind == "things" else "Tom"
    sents = _sentences(c["situation"])
    frame = {"He looks at what is in the storeroom.", "Tom is there."}
    filler = set(FILLER)
    cast = set()
    if kind == "people":
        cast = set(_people()[c["need"]]["sit"]["cast"].values())
    kept_model, correct = [], 0
    q = (f"Is this sentence about what {actor} must do?" if kind == "things"
         else f"Is this sentence about what is happening to the people {actor} sees?")
    for sent in sents:
        if sent in frame:
            kept_model.append(sent)
            continue
        a = _ask(s, sent, q, ["yes, it matters here", "no, it is only background"])
        keep = a["pick"] == 0
        correct += keep == (sent not in filler)
        if keep:
            kept_model.append(sent)
    judged = [x for x in sents if x not in frame]
    if kind == "things":
        rule = [x for x in sents if actor in x or x in frame]
    else:
        rule = [x for x in sents if x in frame or any(n.split()[-1] in x for n in cast)
                or x.startswith(c["need"].split(" ")[0])]
    short = [x for x in sents if x not in filler]
    qn, opts, good = c["question"], c["options"], c["good"]
    arms = {"none": _final(s, c["situation"], qn, opts, good),
            "model": _final(s, " ".join(kept_model) or c["situation"], qn, opts, good),
            "rule": _final(s, " ".join(rule) or c["situation"], qn, opts, good),
            "oracle": _final(s, " ".join(short), qn, opts, good)}
    return {"stage": {"sentence_acc": correct / max(1, len(judged)), "kept": len(kept_model),
                      "of": len(sents)}, "arms": arms}


# ------------------------------------------------------------------ 2e soft kind first
def exp_softkind(s, c, rng):
    by, _ = _items()
    opts = c["options"]
    groups = defaultdict(list)
    for i, o in enumerate(opts):
        groups[by[o]["cat"]].append(i)
    kinds = list(groups)
    labels = [f"{KIND_WORDS[k]} ({', '.join(opts[i] for i in groups[k][:3])}"
              f"{', and more' if len(groups[k]) > 3 else ''})" for k in kinds]
    st = _ask(s, c["situation"], c["question"].replace("Which thing", "Which kind of thing"), labels)
    lp_kind = {k: math.log(max(p, 1e-9)) for k, p in zip(kinds, st["p"])}
    arms = {"none": _final(s, c["situation"], c["question"], opts, c["good"])}
    for lam in (0.5, 1.0):
        bonus = [lam * lp_kind[by[o]["cat"]] for o in opts]
        arms[f"lambda{lam}"] = _final(s, c["situation"], c["question"], opts, c["good"], extra=bonus)
    good_kinds = {by[opts[i]]["cat"] for i in c["good"]}
    return {"stage": {"kind_right": kinds[st["pick"]] in good_kinds}, "arms": arms}


# ------------------------------------------------------------------ 2f speech demo
TEMPLATE = {
    "accuse_lie": '"{target}, that is not true: {fact}"',
    "accuse_theft": '"{target}, I saw it: {fact}"',
    "warn": '"Watch out: {fact}"',
    "report": '"You must know this: {fact}"',
    "alert": '"Help! {fact}"',
    "inform": '"{fact}"',
}


def exp_speech(s, c, rng):
    P = _people()[c["need"]]
    r = s.rank(c["situation"], c["question"], c["options"])
    o = P["opt"].get(c["names"][r["pick"]])
    if not o or o["purpose"] not in SPEECH:
        return {"stage": {"spoke": False}, "arms": {}}
    key = P["sit"]["key"]
    facts = [f.strip().rstrip(".") for f in re.split(r",|;| and | but ", key) if len(f.split()) >= 3]
    facts = facts or [key.rstrip(".")]
    a = _ask(s, f"{key} Tom decides to {o['short']}.", "Which fact should Tom point to?", facts)
    fact = facts[a["pick"]]
    target = P["sit"]["cast"].get(o["role"], "")
    line = TEMPLATE[o["purpose"]].format(target=target.capitalize(), fact=fact[0].lower() + fact[1:] + ".")
    return {"stage": {"spoke": True, "act": o["short"], "facts": facts, "fact": fact, "line": line},
            "arms": {}}


# ------------------------------------------------------------------ B1: the capability from the verb
# What the "with what" slot of each verb needs, by the verb's general meaning
# (docs/action-catalog.md, plus common verbs of the same kind). Not by the
# object: "keep", "stay", "start", "see", "fix" are left out (the verb alone
# does not say what is needed). A verb with two senses gets two phrases.
VERB_CAP = {
    "cut": ["cut"], "chop": ["chop"], "dig": ["dig"], "tie": ["bind"], "climb": ["climb"],
    "light": ["ignite", "light"], "carry": ["carry_load"], "pour": ["pour"], "break": ["blunt_heavy"],
    "block": ["shield"], "fight": ["weapon"], "strike": ["weapon"], "open": ["lockpick", "pry"],
    "treat": ["bandage"], "bandage": ["bandage"], "cook": ["cook"], "drink": ["drink"], "eat": ["food"],
    "feed": ["food"], "write": ["write"], "mend": ["sew"], "sew": ["sew"], "measure": ["measure"],
    "weigh": ["weigh"], "clean": ["clean"], "sharpen": ["sharpen"], "sleep": ["sleep"], "pay": ["pay"],
    "catch": ["catch_fish", "trap"], "trap": ["trap"], "reach": ["reach"], "cross": ["float"],
    "signal": ["signal_sight", "signal_sound"], "call": ["signal_sound"], "hide": ["dry_hold", "shield"],
}


def verb_hint(need):
    props = VERB_CAP.get(need.split()[0], [])
    return [PROP_PHRASE[p] for p in props if p in PROP_PHRASE]


def exp_capability2(s, c, rng):
    """The hint looked up from the verb (data, no model call), against none and
    the right hint. A need whose verb is not in the table gets no hint."""
    by, needs = _items()
    nd = needs[c["need"]]
    right = {PROP_PHRASE[p] for p in nd["good"] if p in PROP_PHRASE}
    hint = verb_hint(c["need"])

    def with_hint(h):
        return _insert_after(c["situation"], "Kael needs", f"For that he needs {h}.")
    q, opts, good = c["question"], c["options"], c["good"]
    arms = {"none": _final(s, c["situation"], q, opts, good),
            "oracle": _final(s, with_hint(sorted(right)[0]), q, opts, good) if right else None}
    arms["lookup"] = _final(s, with_hint(" or ".join(hint)), q, opts, good) if hint else arms["none"]
    arms["lookup_first"] = _final(s, with_hint(hint[0]), q, opts, good) if hint else arms["none"]
    arms = {k: v for k, v in arms.items() if v is not None}
    return {"stage": {"covered": bool(hint), "lookup_right": bool(set(hint) & right), "hint": hint},
            "arms": arms}


# ------------------------------------------------------------------ B4: relevance with a threshold
RELEVANCE_KEEP = 0.35


def exp_relevance2(s, c, rng):
    """The same sentence question as 2d, but a sentence is kept when p(yes) is
    at least RELEVANCE_KEEP (A5 found the model ranks them perfectly; 0.5 was
    too strict), and, in a second arm, the need sentence is always kept."""
    kind = c["kind"]
    actor = "Kael" if kind == "things" else "Tom"
    sents = _sentences(c["situation"])
    frame = {"He looks at what is in the storeroom.", "Tom is there."}
    filler = set(FILLER)
    need_sent = (_sentences(c["need"])[0] if kind == "people" else
                 next((x for x in sents if x.startswith("Kael needs")), None))
    q = (f"Is this sentence about what {actor} must do?" if kind == "things"
         else f"Is this sentence about what is happening to the people {actor} sees?")
    keep_t, keep_n, correct, judged = [], [], 0, 0
    for sent in sents:
        if sent in frame:
            keep_t.append(sent)
            keep_n.append(sent)
            continue
        a = _ask(s, sent, q, ["yes, it matters here", "no, it is only background"])
        judged += 1
        kt = a["p"][0] >= RELEVANCE_KEEP
        correct += kt == (sent not in filler)
        if kt:
            keep_t.append(sent)
        if a["pick"] == 0 or sent == need_sent:
            keep_n.append(sent)
    short = [x for x in sents if x not in filler]
    qn, opts, good = c["question"], c["options"], c["good"]
    arms = {"none": _final(s, c["situation"], qn, opts, good),
            "threshold": _final(s, " ".join(keep_t) or c["situation"], qn, opts, good),
            "keep_need": _final(s, " ".join(keep_n) or c["situation"], qn, opts, good),
            "oracle": _final(s, " ".join(short), qn, opts, good)}
    return {"stage": {"sentence_acc": correct / max(1, judged), "kept": len(keep_t), "of": len(sents),
                      "all_kept": all(x in keep_t for x in short)}, "arms": arms}


# ------------------------------------------------------------------ 4 + 2: what is missing
def exp_state(s, c, rng):
    """Multi-step needs with the scene state written out (cases_state.py):
    "What is still missing?" first, then the choice."""
    labels = [c["missing"], c["other_missing"], "nothing, he has all he needs", "someone to help him"]
    st = _ask(s, c["situation"], "What is still missing?", labels)
    tail = "He looks at what is in the storeroom."

    def with_missing(m):
        return c["situation"].replace(tail, f"He still needs {m}. {tail}")
    q, opts, good = c["question"], c["options"], c["good"]
    arms = {"none": _final(s, c["situation"], q, opts, good),
            "oracle": _final(s, with_missing(c["missing"]), q, opts, good),
            "model": _final(s, with_missing(labels[st["pick"]]), q, opts, good),
            "wrong": _final(s, with_missing(c["other_missing"]), q, opts, good)}
    for a in arms.values():
        a["other_step"] = a["pick"] in set(c["step_other"])
    return {"stage": {"missing_right": st["pick"] == 0, "said": labels[st["pick"]]}, "arms": arms}


EXPERIMENTS = {
    "capability": (exp_capability, "cases_things.jsonl", lambda c: c["factor"] == "size" and c["value"] in (16, 64)),
    "people": (exp_people, "cases_people.jsonl", lambda c: c["factor"] == "size" and c["value"] in (8, 16)
               or c["factor"] == "similar"),
    "personality": (exp_personality, "cases_personality.jsonl", lambda c: True),
    "relevance": (exp_relevance, None, None),
    "softkind": (exp_softkind, "cases_things.jsonl", lambda c: c["factor"] == "size" and c["value"] in (16, 64)),
    "speech": (exp_speech, "cases_people.jsonl", lambda c: c["factor"] == "size" and c["value"] in (4, 8)),
    "state": (exp_state, "cases_state.jsonl", lambda c: c["value"] != "none"),
    "personality2": (exp_personality2, "cases_personality2.jsonl", lambda c: True),
    "capability2": (exp_capability2, "cases_things.jsonl",
                    lambda c: c["factor"] == "size" and c["value"] in (16, 64)),
    "relevance2": (exp_relevance2, None, None),
}


def _cases(exp):
    fn, cf, keep = EXPERIMENTS[exp]
    if exp in ("relevance", "relevance2"):
        out = [json.loads(l) for l in open(HERE / "cases_things.jsonl")]
        out = [c for c in out if c["factor"] == "situation" and c["value"] == "long"]
        pe = [json.loads(l) for l in open(HERE / "cases_people.jsonl")]
        return out + [c for c in pe if c["factor"] == "situation" and c["value"] == "long"]
    return [c for c in (json.loads(l) for l in open(HERE / cf)) if keep(c)]


# ------------------------------------------------------------------ runner
def _flush(path):
    """Write the captured vectors of the last cases to a new part file."""
    global CAP
    if not CAP:
        return
    path.mkdir(exist_ok=True)
    k = len(list(path.glob("part*.npz")))
    np.savez_compressed(
        path / f"part{k}.npz", ids=np.array([c[0] for c in CAP]), question=np.array([c[1] for c in CAP]),
        labels=np.array(["\u241f".join(c[2]) for c in CAP]), n=np.array([len(c[2]) for c in CAP]),
        opts=np.concatenate([c[3]["opts"].numpy() for c in CAP]).astype("float16"),
        logits=np.concatenate([c[3]["logits"].numpy() for c in CAP]).astype("float32"))
    CAP = []


def run(exp, scorer, limit=0, capture=False):
    from lab.methods.score import make
    fn = EXPERIMENTS[exp][0]
    cases = _cases(exp)
    if limit:
        cases = cases[:limit]
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{exp}__{scorer.replace(':', '__')}.jsonl"
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    todo = [c for c in cases if c["id"] not in done]
    print(f"{exp} with {scorer}: {len(todo)} to do, {len(done)} done", flush=True)
    if not todo:
        return
    s = make(scorer)
    if os.environ.get("LAB_DEVICE", "cuda") == "cuda" and s.params:
        from lab.methods.run import gpu_check
        gpu_check(s.params)
    s.load()
    if capture:
        s.capture = True
    cap_dir = out.with_suffix(".cap")
    duty = float(os.environ.get("LAB_DUTY", "0.5"))
    with out.open("a") as f:
        for n, c in enumerate(todo):
            t = time.perf_counter()
            rng = random.Random(c["id"])
            CUR["id"] = c["id"]
            try:
                rec = fn(s, c, rng)
            except (ValueError, RuntimeError, KeyError) as e:
                rec = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            dt = time.perf_counter() - t
            rec.update(id=c["id"], factor=c["factor"], value=c["value"], ms=dt * 1000)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if duty < 1:
                time.sleep(dt * (1 / duty - 1))
            if n % 50 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
                if capture:
                    _flush(cap_dir)
    if capture:
        _flush(cap_dir)
    print("done", flush=True)


# ------------------------------------------------------------------ report
def _cell(xs):
    if not xs:
        return ""
    p, lo, hi = wilson(sum(xs), len(xs))
    return f"{p:.0%} ({lo:.0%}-{hi:.0%})"


def report():
    L = ["# Small questions first: results", "",
         "Share of final picks in the acceptable set, with the 95% range. Arms: `none` = as before; "
         "`oracle` = the right preliminary answer; `model` = the model's own; `wrong` = a wrong one "
         "(the cost of a mistake at the first step). Built by `lab/methods/stages.py`.", ""]
    for f in sorted(OUT.glob("*.jsonl")):
        exp, scorer = f.stem.split("__", 1)
        rows = [json.loads(l) for l in open(f)]
        rows = [r for r in rows if "error" not in r]
        errs = sum(1 for l in open(f) if '"error"' in l)
        L += [f"## {exp} ({scorer.replace('__', ':')}, {len(rows)} cases, {errs} errors)", ""]
        if exp == "personality":
            L += ["How much of the draw goes to confronting (C) and avoiding (A) options, hot-tempered against "
                  "timid. `shift` = C(hot) - C(timid): how pronounced the trait is. `ok` = the pick is valid.", "",
                  "| arm | C hot | C timid | shift | A hot | A timid | ok hot | ok timid |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
            for arm in ("none", "text", "appraisal", "oracle"):
                g = lambda t, k: sum(r["traits"][t][arm][k] for r in rows) / max(1, len(rows))
                L.append(f"| {arm} | {g('hot', 'confront'):.2f} | {g('timid', 'confront'):.2f} | "
                         f"{g('hot', 'confront') - g('timid', 'confront'):+.2f} | {g('hot', 'avoid'):.2f} | "
                         f"{g('timid', 'avoid'):.2f} | {_cell([r['traits']['hot'][arm]['ok'] for r in rows])} | "
                         f"{_cell([r['traits']['timid'][arm]['ok'] for r in rows])} |")
            fe = Counter((t, r["stage"][t]["feel"], r["stage"][t]["urge"]) for r in rows for t in ("hot", "timid"))
            L += ["", "Feeling and urge the model gave: " + "; ".join(f"{t}: {a}, {b} ({n})"
                                                                     for (t, a, b), n in fe.most_common(8)), ""]
            continue
        if exp == "personality2":
            L += ["Two axes. `shift` = how much more of the draw goes to the trait's side (confronting for "
                  "temper, helping for kindness) with the trait (hot, kind) than with its opposite (timid, cold). "
                  "`pick ok` = the top option is acceptable; `draw ok` = the probability on acceptable options.", "",
                  "| axis | arm | plus side, trait | plus side, opposite | shift | pick ok, trait | pick ok, opposite "
                  "| draw ok |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
            for axis in ("temper", "kindness"):
                rs = [r for r in rows if r["value"] == axis]
                if not rs:
                    continue
                for arm in ("none", "text", "appraisal", "oracle"):
                    g = lambda side, k: sum(r["sides"][side][arm][k] for r in rs) / len(rs)
                    L.append(f"| {axis} ({len(rs)}) | {arm} | {g('plus', 'plus'):.2f} | {g('minus', 'plus'):.2f} | "
                             f"{g('plus', 'plus') - g('minus', 'plus'):+.2f} | "
                             f"{_cell([r['sides']['plus'][arm]['ok'] for r in rs])} | "
                             f"{_cell([r['sides']['minus'][arm]['ok'] for r in rs])} | "
                             f"{(g('plus', 'draw_ok') + g('minus', 'draw_ok')) / 2:.0%} |")
                fe = Counter((r["stage"][sd]["trait"], r["stage"][sd]["feel"], r["stage"][sd]["urge"])
                             for r in rs for sd in ("plus", "minus"))
                L += ["", f"{axis}: feeling and urge the model gave: " + "; ".join(
                    f"{t}: {a}, {b} ({n})" for (t, a, b), n in fe.most_common(8)), ""]
            continue
        if exp == "speech":
            spoke = [r for r in rows if r["stage"].get("spoke")]
            L += [f"{len(spoke)} of {len(rows)} picks were speech acts. Examples:", ""]
            L += [f"- {r['stage']['act']} → {r['stage']['line']}" for r in spoke[:20]] + [""]
            continue
        by_size = defaultdict(list)
        for r in rows:
            by_size[str(r["value"])].append(r)
        arms = sorted({a for r in rows for a in r["arms"]}, key=lambda a: (a != "none", a))
        L += ["| cell | " + " | ".join(arms) + " | stage |", "| --- |" + " --- |" * (len(arms) + 1)]
        for v, rs in sorted(by_size.items()):
            st = rs[0]["stage"]
            key = next((k for k in ("top1_right", "call_right", "kind_right", "sentence_acc", "missing_right",
                                    "covered") if k in st), None)
            stage = f"{sum(r['stage'][key] for r in rs) / len(rs):.0%} {key}" if key else ""
            if exp == "capability":
                stage += f", top 2: {sum(r['stage']['top2_right'] for r in rs) / len(rs):.0%}"
            if exp == "capability2":
                stage += f", lookup right: {sum(r['stage']['lookup_right'] for r in rs) / len(rs):.0%}"
            if exp == "relevance2":
                stage += f", every needed sentence kept: {sum(r['stage']['all_kept'] for r in rs) / len(rs):.0%}"
            if exp == "people":
                stage += f", roles: {sum(r['stage']['appraisal_right'] for r in rs) / len(rs):.0%}"
            if exp == "state":
                stage += ", picks of the OTHER step: " + ", ".join(
                    f"{a} {sum(r['arms'][a]['other_step'] for r in rs) / len(rs):.0%}" for a in arms)
            L.append(f"| {v} | " + " | ".join(_cell([r["arms"][a]["ok"] for r in rs if a in r["arms"]])
                                               for a in arms) + f" | {stage} |")
        L.append("")
    (OUT.parent / "stages.md").write_text("\n".join(L) + "\n")
    print(OUT.parent / "stages.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run", "report"])
    ap.add_argument("exp", nargs="?")
    ap.add_argument("--scorer", default="gdecide:all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--capture", action="store_true", help="save what the model saw at every question")
    a = ap.parse_args()
    if a.cmd == "build":
        build_personality() if a.exp != "personality2" else build_personality2()
    elif a.cmd == "run":
        run(a.exp, a.scorer, a.limit, a.capture)
    else:
        report()
