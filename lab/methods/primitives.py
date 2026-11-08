"""Primitive questions: which KINDS of question can GLiNER-Decide answer?

    python3 -m lab.methods.primitives            # writes lab/methods/cases_primitives.jsonl (CPU)
    python3 -m lab.methods.primitives report     # lab/results/methods/primitives.md (after the GPU run)

Plan v4 asked a few preliminary questions, and most were not right often
enough to help. A wrong first answer costs more than a right one gains. One
kind was nearly always right: "Is this sentence about what Kael must do?"
(97%). So before we build a line of questioning, we find the kinds of
question that this model answers reliably, and build only on those.

THE GROUPS. The questions a person asks themselves, grouped by what they are
about. Each group has a KNOWN answer (from the inn world's state, or from
the case data) or an UNKNOWN one (a feeling, a predisposition).

    group           example                                          answer from
    presence        "Who is here with Tom?"                          the inn world
    possession      "Who holds the sword?"                           the inn world
    place           "Where is Tom?"                                  the inn world
    who did it      "Who did something to Oren just now?"            the inn's event log
    to whom         "To whom did Fenn do something just now?"        the inn's event log
    inner state     "Is Tom angry at someone?" (the 'feels' sentence
                    taken out; inferred from what happened)          the inn world
    danger          "Is Tom in danger here?"                         the inn world
    property        "Is the kitchen knife something with a sharp
                    edge?"                                           item properties
    use             "Kael needs to cut the rope. Can he do it with
                    the axe?"                                        the need labels
    goal            "What is Kael trying to do?"                     the case
    event kind      "What is happening?" (a theft / a fight / ...)   the scene's cast roles
    roles           "Who is in danger or hurt?" (people named in the
                    scene only)                                      the cast roles
    fault           "Whose fault is this?"                           the cast roles
    norms           "Would it be wrong for Tom to hit the child?"    by convention (clear cases only)
    calls for       "What does this situation call for?"             the good actions
    false belief    "Where will Marta look for the ladle?"           the story
    next step       "What is still missing?"                         the state cases
    --- unknown answer: the direction is known, not the answer ---
    feeling         "How does Tom feel?" with a trait sentence
    urge            "What does Tom want to do?"
    coping          "Can Tom deal with this alone?"
    approach        "How will Kael go about it?"

FORMATS. The same knowledge asked as: `pick` (one of a few phrases), `yesno`
(one candidate: yes or no), and for roles `entity` (pick the person) against
`category` (pick the role for a named person).

WHEN THE ANSWER IS UNKNOWN (and also for the known groups):
    wording     the same question in 3 wordings; the top answer should not change
    order       the answers listed in reverse; the top answer should not change
    filler      two unrelated sentences added; the top answer should not change
    direction   (unknown groups) a trait sentence ("Tom is hot-tempered")
                against no trait sentence: the expected answer ("angry")
                must gain probability. The pairs are minimal: only the trait
                sentence differs.
    confidence  how often the answer is right when the model is sure (top
                probability 0.8 or more), and how often it is that sure: a
                question kind can be used when sure and skipped when not.

Every case is in the runner's format (lab/methods/run.py): one GLiNER-Decide
call each, scores kept.
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
RES = HERE.parent / "results" / "methods"
OUT = HERE / "cases_primitives.jsonl"

KNOWN = ["presence", "possession", "place", "who did it", "to whom", "inner state", "danger", "property", "use",
         "goal", "event kind", "roles", "fault", "norms", "calls for", "false belief", "next step"]
UNKNOWN = ["feeling", "urge", "coping", "approach"]

FILLER2 = "Rain drums on the roof. A cat watches from the top of a shelf."

# the same question in three wordings; {x} is filled per case
WORDINGS = {
    "presence": ["Who is here with {p}?", "Who is in the room with {p}?", "Which of these people can {p} see here?"],
    "presence_yn": ["Is {q} here with {p}?", "Is {q} in the same room as {p}?", "Can {p} see {q} here?"],
    "possession": ["Who holds {t}?", "Who has {t}?", "Whose hands is {t} in?"],
    "place": ["Where is {p}?", "In which room is {p}?", "Where is {p} now?"],
    "who did it": ["Who did something to {q} just now?", "Who acted on {q} a moment ago?",
                   "Who was it that just did something to {q}?"],
    "to whom": ["To whom did {q} do something just now?", "Who did {q} act on a moment ago?",
                "Who was on the receiving end of what {q} just did?"],
    "angry_yn": ["Is {p} angry at someone?", "Is {p} angry now?", "Does {p} feel anger towards anyone here?"],
    "angry_who": ["Who is {p} angry at?", "Who has made {p} angry?", "At whom is {p} angry?"],
    "danger_yn": ["Is {p} in danger here?", "Is {p} in danger?", "Could {p} get hurt here?"],
    "property_yn": ["Is {x} {prop}?", "Would you call {x} {prop}?", "Can {x} be described as {prop}?"],
    "use_yn": ["Can he do it with {x}?", "Will {x} do the job?", "Could {x} be used for this?"],
    "goal": ["What is Kael trying to do?", "What does Kael want to get done?", "What is Kael's task?"],
    "event kind": ["What is happening?", "What kind of event is this?", "What is going on here?"],
    "roles_entity_hurt": ["Who is in danger or hurt?", "Who is hurt or at risk?", "Who needs protecting here?"],
    "roles_entity_wrong": ["Who is doing wrong?", "Who is the wrongdoer here?", "Who is behaving badly?"],
    "roles_category": ["What is {q} in this scene?", "What best describes {q} here?", "Which of these is {q}?"],
    "roles_yn_hurt": ["Is {q} in danger or hurt?", "Is {q} hurt or at risk?", "Does {q} need protecting?"],
    "fault": ["Whose fault is this?", "Who is to blame for this?", "Who caused this?"],
    "norms_yn": ["Would it be wrong for Tom to {a}?", "Is it wrong to {a} here?", "Would people blame Tom if he chose to {a}?"],
    "calls for": ["What does this situation call for?", "What is needed here?", "What would help most here?"],
    "look": ["Where will {a} look for {t}?", "Where does {a} think {t} is?", "Where will {a} go to get {t}?"],
    "where_now": ["Where is {t} now?", "Where is {t} really?", "Where would you find {t} now?"],
    "know_yn": ["Does {a} know that {t} was moved?", "Is {a} aware that {t} was moved?",
                "Has {a} seen {t} being moved?"],
    "missing": ["What is still missing?", "What does Kael still need?", "What is Kael still without?"],
    "feeling": ["How does Tom feel?", "What is Tom feeling now?", "How does this make Tom feel?"],
    "urge": ["What does Tom want to do?", "What is Tom's first urge?", "What would Tom like to do now?"],
    "coping": ["Can Tom deal with this alone?", "Is Tom able to handle this by himself?",
               "Could Tom manage this on his own?"],
    "approach": ["How will Kael go about it?", "How would Kael try to do it?", "What will Kael do first?"],
}
YESNO = ["yes", "no"]

EVENT_KINDS = ["a theft", "a fight or an attack", "an injury or an illness", "a lie or a cheat",
               "someone lost, sad or grieving", "a trade", "someone asking for help", "an accident or a fire"]
ROLE_EVENT = {"threat": 1, "fighter": 1, "fighter2": 1, "thief": 0, "liar": 3, "cheat": 3, "accuser": 3,
              "injured": 2, "patient": 2, "grieving": 4, "trader": 5, "beggar": 6, "helpee": 6, "stranger": 6}
HURT_ROLES = {"victim", "injured", "patient"}
WRONG_ROLES = {"threat", "thief", "liar", "cheat", "fighter", "fighter2"}
HARM_PURPOSES = {"attack", "pelt", "insult", "steal", "threaten"}
KIND_PURPOSES = {"comfort", "heal", "give_water", "rescue", "shield", "praise", "thank", "greet", "calm"}

# unknown answers: trait sentences and the answer each should raise
PERSONAS = {"hot": "Tom is hot-tempered and quick to fight.", "timid": "Tom is timid and afraid of trouble.",
            "kind": "Tom is kind and always helps others.", "greedy": "Tom is greedy and thinks only of money.",
            "calm": "Tom is calm and patient."}
FEEL = ["angry", "afraid", "calm", "sad", "glad"]
URGE = ["stand up to them", "get away from here", "call for someone", "keep out of it", "help whoever is hurt",
        "get something for himself"]
COPE = YESNO
EXPECT = {("feeling", "hot"): "angry", ("feeling", "timid"): "afraid", ("feeling", "calm"): "calm",
          ("urge", "hot"): "stand up to them", ("urge", "timid"): "get away from here",
          ("urge", "kind"): "help whoever is hurt", ("urge", "greedy"): "get something for himself",
          ("coping", "hot"): "yes", ("coping", "timid"): "no"}
GRID_SCENES = [("A friend gives Tom a warm loaf of bread as a gift.", "glad"),
               ("Tom's old dog died in the night.", "sad"),
               ("A drunk man shoves Tom and laughs at him.", "angry"),
               ("A big man with a knife walks towards Tom.", "afraid"),
               ("Tom sits by the fire and nothing much is happening.", "calm"),
               ("Tom is there.", None)]                      # the empty scene
KAEL = {"impatient": "Kael is strong and impatient.", "careful": "Kael is careful and clever.",
        "unsure": "Kael is shy and unsure of himself.", "lazy": "Kael is lazy and gives up easily."}
APPROACH = ["force it", "look for the right tool", "ask someone to help", "give up for now"]
EXPECT.update({("approach", "impatient"): "force it", ("approach", "careful"): "look for the right tool",
               ("approach", "unsure"): "ask someone to help", ("approach", "lazy"): "give up for now"})

TOM_STORIES = [("Marta", "the ladle", "the pot", "the shelf", "Fenn"),
               ("Oren", "the purse", "his bag", "the chest", "Fenn"),
               ("Sir Ada", "the sword", "the rack", "the barrel", "Tom"),
               ("Tom", "the copper coin", "his boot", "the jar", "Oren"),
               ("the cook", "the knife", "the drawer", "the basket", "the boy"),
               ("the miller", "the key", "the box", "the sack", "his wife"),
               ("the shepherd", "the bell", "the hook", "the bench", "the girl"),
               ("the pedlar", "the ring", "the pouch", "the cup", "the thief"),
               ("the smith", "the hammer", "the anvil", "the trough", "the apprentice"),
               ("the priest", "the candle", "the altar", "the window", "the boy")]


# ------------------------------------------------------------------ building
LURE_GROUPS = {"presence", "possession", "place", "who did it", "to whom", "roles", "fault", "false belief"}


def _case(out, group, fmt, text, qkey, fill, labels, good, known=True, base=None, **extra):
    """The base case and its variants: wordings 1 and 2, the reversed order,
    two filler sentences, and (for pick questions over names, things and
    places) a LURE: one sentence that names a wrong answer."""
    base = base or f"{group}/{len(out)}"
    wrong = [x for i, x in enumerate(labels) if i not in good and x not in ("no one", "nobody")]
    lure = group in LURE_GROUPS and fmt in ("pick", "entity") and good and wrong
    variants = [("base", 0, False, ""), ("wording1", 1, False, ""), ("wording2", 2, False, ""),
                ("order", 0, True, ""), ("filler", 0, False, FILLER2)]
    if lure:
        variants.append(("lure", 0, False, f"Someone mentions {wrong[0]}."))
    for var, w, rev, add in variants:
        opts = labels[::-1] if rev else list(labels)
        gi = [opts.index(labels[i]) for i in good]
        out.append({"id": f"prim/{base}/{var}", "kind": "primitive", "factor": group, "value": fmt,
                    "group": group, "fmt": fmt, "known": known, "variant": var, "base": base,
                    "situation": (f"{text} {add}" if add else text),
                    "question": WORDINGS[qkey][w].format(**fill), "options": opts, "names": opts, "good": gi,
                    **extra})


def inn_cases(out, seeds=12, per_seed=12, seed=0):
    from lab.world2.policy import make_policy
    from lab.world2.world import HARMFUL, ROOMS, VIOLENT, World
    rng = random.Random(f"{seed}-inn")
    pol = make_policy("rules")
    for s in range(seeds):
        w = World(seed=5000 + s)
        times = sorted(rng.sample(range(15, 300), per_seed))
        for t in range(300):
            w.step(pol)
            if t not in times:
                continue
            live = [p for p in w.people.values() if p.acts()]
            if not live:
                break
            p = rng.choice(live)
            P = p.name
            text = w.describe(p)
            no_feel = ". ".join(x for x in text.split(". ") if not x.startswith(f"{P} feels")).strip()
            no_feel = no_feel if no_feel.endswith(".") else no_feel + "."
            here = [q.name for q in w.here(p)]
            others = [n for n in w.people if n != P]
            tag = f"inn{s}t{t}"
            # presence
            absent = [n for n in others if n not in here]
            if here and absent:
                labels = rng.sample(here, min(2, len(here))) + rng.sample(absent, min(2, len(absent)))
                rng.shuffle(labels)
                _case(out, "presence", "pick", text, "presence", {"p": P}, labels,
                      [i for i, n in enumerate(labels) if n in here], base=f"presence/{tag}")
                for q, ans in ((rng.choice(here), 0), (rng.choice(absent), 1)):
                    _case(out, "presence", "yesno", text, "presence_yn", {"p": P, "q": q}, YESNO, [ans],
                          base=f"presence_yn/{tag}/{q}")
            # possession: a thing held openly by someone here, or by P
            held = [(th, q) for q in [P] + here for th in w.held(w.people[q], visible_only=True)]
            if held:
                th, holder = rng.choice(held)
                labels = [P] + ([holder] if holder != P else []) + [n for n in here if n != holder][:3] + ["nobody"]
                _case(out, "possession", "pick", text, "possession", {"t": th}, labels, [labels.index(holder)],
                      base=f"possession/{tag}")
            # place
            rooms = list(ROOMS)
            _case(out, "place", "pick", text, "place", {"p": P}, rooms, [rooms.index(p.room)], base=f"place/{tag}")
            # who did it, to whom: from the events the text lists
            recent = [e for e in w.log if e["id"] in p.perceived][-5:]
            aimed = [e for e in recent if e.get("target") in w.people and e["done"]]
            if aimed:
                e = aimed[-1]
                tgt = e["target"]
                actors = {x["actor"] for x in aimed if x["target"] == tgt}
                pool = [n for n in w.people if n != tgt]
                labels = sorted(actors) + [n for n in rng.sample(pool, len(pool)) if n not in actors][:4 - len(actors)]
                rng.shuffle(labels)
                _case(out, "who did it", "pick", text, "who did it", {"q": tgt}, labels,
                      [i for i, n in enumerate(labels) if n in actors], base=f"who/{tag}")
                act = e["actor"]
                tgts = {x["target"] for x in aimed if x["actor"] == act}
                pool = [n for n in w.people if n != act]
                labels = sorted(tgts) + [n for n in rng.sample(pool, len(pool)) if n not in tgts][:4 - len(tgts)]
                rng.shuffle(labels)
                _case(out, "to whom", "pick", text, "to whom", {"q": act}, labels,
                      [i for i, n in enumerate(labels) if n in tgts], base=f"whom/{tag}")
            # inner state, from what happened (the 'feels' sentence taken out)
            harmed_by = {x["actor"] for x in recent if x.get("target") == P and x["action"] in HARMFUL and x["done"]}
            mad = {n for n, a in p.anger.items() if a > 0.4}
            if mad & harmed_by:
                _case(out, "inner state", "yesno", no_feel, "angry_yn", {"p": P}, YESNO, [0], base=f"angry/{tag}")
                labels = sorted(mad & harmed_by)[:1] + [n for n in here if n not in mad][:3]
                if len(labels) > 1:
                    rng.shuffle(labels)
                    _case(out, "inner state", "pick", no_feel, "angry_who", {"p": P}, labels,
                          [i for i, n in enumerate(labels) if n in mad], base=f"angry_who/{tag}")
            elif not p.anger or max(p.anger.values()) < 0.1:
                if not harmed_by:
                    _case(out, "inner state", "yesno", no_feel, "angry_yn", {"p": P}, YESNO, [1],
                          base=f"angry/{tag}")
            # danger
            dng = w.danger(p)
            violent = any(x["action"] in VIOLENT for x in recent)
            if dng >= 0.5:
                _case(out, "danger", "yesno", text, "danger_yn", {"p": P}, YESNO, [0], base=f"danger/{tag}")
            elif dng == 0 and not violent and "bandit" not in text:
                _case(out, "danger", "yesno", text, "danger_yn", {"p": P}, YESNO, [1], base=f"danger/{tag}")


def people_cases(out, seed=0):
    from lab.methods.cases_kinds import parse_people, people_unit
    from lab.methods.stages import APPRAISAL, CALLS, PURPOSE_CALL, _appraise_oracle
    rng = random.Random(f"{seed}-people")
    templates, untargeted, sits = parse_people()
    for k, sit in enumerate(sits):
        key = sit["key"]
        text = f"{key} Tom is there."
        low = key.lower()
        named = {r: n for r, n in sit["cast"].items() if (n[4:] if n.startswith("the ") else n).lower() in low}
        # event kind
        kinds = sorted({ROLE_EVENT[r] for r in sit["cast"] if r in ROLE_EVENT and r in named})
        if "fire" in low:
            kinds = sorted(set(kinds) | {7})
        if "falls" in low or "collapses" in low:
            kinds = sorted(set(kinds) | {2, 7})
        if kinds:
            _case(out, "event kind", "pick", text, "event kind", {}, EVENT_KINDS, kinds, base=f"kind/{k}")
        # roles: named people only
        hurt = [n for r, n in named.items() if r in HURT_ROLES]
        wrong = [n for r, n in named.items() if r in WRONG_ROLES]
        people = list(named.values()) + ["Tom", "no one"]
        if hurt:
            _case(out, "roles", "entity", text, "roles_entity_hurt", {}, people, [people.index(n) for n in hurt],
                  base=f"roles_hurt/{k}")
        if wrong:
            _case(out, "roles", "entity", text, "roles_entity_wrong", {}, people, [people.index(n) for n in wrong],
                  base=f"roles_wrong/{k}")
        orc = _appraise_oracle(sit)
        for n in named.values():
            _case(out, "roles", "category", text, "roles_category", {"q": n}, APPRAISAL, [APPRAISAL.index(orc[n])],
                  base=f"roles_cat/{k}/{n}")
            _case(out, "roles", "yesno", text, "roles_yn_hurt", {"q": n}, YESNO, [0 if n in hurt else 1],
                  base=f"roles_yn/{k}/{n}")
        # fault
        # fault: the named wrongdoer; "no one" only where nobody in the cast did wrong
        if wrong or (hurt and not any(r in WRONG_ROLES for r in sit["cast"])):
            _case(out, "fault", "pick", text, "fault", {}, people,
                  [people.index(n) for n in wrong] if wrong else [people.index("no one")], base=f"fault/{k}")
        # norms: clear cases only (harm to someone who did no wrong; a kind act)
        unit = people_unit(templates, untargeted, sit)
        wrongs = [o for o in unit["all"] if o["purpose"] in HARM_PURPOSES and o["role"] not in WRONG_ROLES
                  and o["role"] in sit["cast"]]
        kinds_ = [o for o in unit["all"] if o["purpose"] in KIND_PURPOSES and o["role"] not in WRONG_ROLES]
        for o, ans in [(o, 0) for o in rng.sample(wrongs, min(3, len(wrongs)))] + \
                      [(o, 1) for o in rng.sample(kinds_, min(3, len(kinds_)))]:
            _case(out, "norms", "yesno", text, "norms_yn", {"a": o["short"]}, YESNO, [ans],
                  base=f"norms/{k}/{o['short']}")
        # calls for
        right = sorted({PURPOSE_CALL[p] for p, _ in sit["good"] if p in PURPOSE_CALL})
        if right:
            _case(out, "calls for", "pick", text, "calls for", {}, CALLS, right, base=f"calls/{k}")


def things_cases(out, seed=0):
    from lab.methods.cases import label, parse
    from lab.methods.stages import PROP_PHRASE
    rng = random.Random(f"{seed}-things")
    items, needs = parse()
    # property: one thing that has it, one that does not
    for prop, phrase in PROP_PHRASE.items():
        has = [i for i in items if prop in i["props"]]
        hasnt = [i for i in items if prop not in i["props"]]
        for i in rng.sample(has, min(3, len(has))):
            _case(out, "property", "yesno", f"Kael looks at {i['name']}.", "property_yn",
                  {"x": i["name"], "prop": phrase}, YESNO, [0], base=f"prop/{prop}/{i['name']}")
        for i in rng.sample(hasnt, 3):
            _case(out, "property", "yesno", f"Kael looks at {i['name']}.", "property_yn",
                  {"x": i["name"], "prop": phrase}, YESNO, [1], base=f"prop/{prop}/{i['name']}")
    near = {}
    from lab.methods.cases import NEARMISS
    for n, s in NEARMISS.items():
        near[n] = {x.strip() for x in s.split(",")}
    for nd in needs:
        text = f"Kael needs to {nd['need']}."
        good = [i for i in items if label(i, nd) == "good"]
        bad = [i for i in items if label(i, nd) == "bad" and i["name"] not in near.get(nd["need"], ())]
        nm = [i for i in items if i["name"] in near.get(nd["need"], ())]
        for i, ans, kind in [(i, 0, "good") for i in rng.sample(good, min(2, len(good)))] + \
                            [(i, 1, "bad") for i in rng.sample(bad, 2)] + \
                            [(i, 1, "near") for i in rng.sample(nm, min(2, len(nm)))]:
            _case(out, "use", "yesno", text, "use_yn", {"x": i["name"]}, YESNO, [ans],
                  base=f"use/{nd['need']}/{i['name']}", item_kind=kind)
        others = rng.sample([x["need"] for x in needs if x is not nd], 4)
        labels = [nd["need"]] + others
        rng.shuffle(labels)
        _case(out, "goal", "pick", f"{text} He looks at what is in the storeroom.", "goal", {}, labels,
              [labels.index(nd["need"])], base=f"goal/{nd['need']}")


def belief_cases(out):
    """False belief: someone moves a thing while its owner is away (or
    watching). Where will the owner look? Where is it now? Does the owner
    know? The two versions differ in one sentence."""
    for k, (a, t, p1, p2, b) in enumerate(TOM_STORIES):
        for saw in (False, True):
            story = (f"{a[0].upper() + a[1:]} puts {t} in {p1} and goes out. " if not saw else
                     f"{a[0].upper() + a[1:]} puts {t} in {p1}. ") + \
                    (f"While {a} is away, {b} moves {t} to {p2}. {a[0].upper() + a[1:]} comes back." if not saw else
                     f"{a[0].upper() + a[1:]} watches {b} move {t} to {p2}.")
            places = [p1, p2]
            tag = f"{k}/{'saw' if saw else 'away'}"
            _case(out, "false belief", "pick", story, "look", {"a": a, "t": t}, places, [1 if saw else 0],
                  base=f"look/{tag}", saw=saw)
            _case(out, "false belief", "pick", story, "where_now", {"t": t}, places, [1], base=f"now/{tag}",
                  saw=saw, control=True)
            _case(out, "false belief", "yesno", story, "know_yn", {"a": a, "t": t}, YESNO, [0 if saw else 1],
                  base=f"know/{tag}", saw=saw)


def state_cases(out):
    from lab.methods.cases_state import STEPS
    for need, steps in STEPS.items():
        for si, (missing, state, _) in enumerate(steps):
            other = steps[1 - si][0]
            labels = [missing, other, "nothing, he has all he needs", "someone to help him"]
            _case(out, "next step", "pick", f"Kael needs to {need}. {state}", "missing", {}, labels, [0],
                  base=f"missing/{need}/{si}")


def disposition_cases(out, seed=0):
    """Unknown answers. Each scene is asked with no trait sentence and with
    each trait sentence; only that sentence differs."""
    from lab.methods.cases_state import STEPS
    scenes = sorted({json.loads(l)["need"] for l in open(HERE / "cases_personality.jsonl")})
    for k, key in enumerate(scenes):
        for trait, sent in [("none", "")] + list(PERSONAS.items()):
            text = f"{key} Tom is there." + (f" {sent}" if sent else "")
            for grp, labels in (("feeling", FEEL), ("urge", URGE), ("coping", COPE)):
                _case(out, grp, "pick" if grp != "coping" else "yesno", text, grp, {}, labels, [], known=False,
                      base=f"{grp}/{k}/{trait}", scene=key, trait=trait,
                      expect=EXPECT.get((grp, trait)))
    # the trait x scene grid: does the scene still move the feeling once a
    # trait is given? (a 'state-blind' model follows the trait only), and the
    # empty scene (what the model says with nothing to go on)
    for k, (key, feel) in enumerate(GRID_SCENES):
        for trait, sent in [("none", "")] + list(PERSONAS.items()):
            text = key + (f" {sent}" if sent else "")
            _case(out, "feeling", "grid", text, "feeling", {}, FEEL, [FEEL.index(feel)] if feel else [],
                  known=False, base=f"grid/{k}/{trait}", scene=key, trait=trait, scene_feel=feel,
                  expect=EXPECT.get(("feeling", trait)))
    needs = sorted(STEPS) + ["cut the rope", "cross the cold river", "open the locked chest", "catch the loose goat",
                             "get the cat down from the roof", "light the dark cellar"]
    for k, need in enumerate(dict.fromkeys(needs)):
        for trait, sent in [("none", "")] + list(KAEL.items()):
            text = f"Kael needs to {need}." + (f" {sent}" if sent else "")
            _case(out, "approach", "pick", text, "approach", {}, APPROACH, [], known=False,
                  base=f"approach/{k}/{trait}", scene=need, trait=trait, expect=EXPECT.get(("approach", trait)))


def build(seed=0):
    out = []
    inn_cases(out, seed=seed)
    people_cases(out, seed=seed)
    things_cases(out, seed=seed)
    belief_cases(out)
    state_cases(out)
    disposition_cases(out, seed=seed)
    with open(OUT, "w") as f:
        for c in out:
            f.write(json.dumps(c) + "\n")
    by = defaultdict(int)
    for c in out:
        if c["variant"] == "base":
            by[c["group"]] += 1
    print(f"{len(out)} cases ({sum(by.values())} questions x 5 variants) -> {OUT}")
    for g in KNOWN + UNKNOWN:
        print(f"  {g:13s} {by.get(g, 0)}")


# ------------------------------------------------------------------ report
def _p(scores):
    import numpy as np
    s = np.asarray(scores, dtype=float)
    if s.min() >= 0 and s.max() <= 1 and abs(s.sum() - 1) < 1e-3:
        return s
    e = np.exp(s - s.max())
    return e / e.sum()


def report(scorer="gdecide:all", tag="prim", sure=0.8):
    import numpy as np
    from lab.methods.report import wilson
    cases = {json.loads(l)["id"]: json.loads(l) for l in open(OUT)}
    f = RES / f"{scorer.replace(':', '__')}.{tag}.jsonl"
    if not f.exists():
        print(f"no results yet: {f}")
        return
    got = {}
    for l in open(f):
        r = json.loads(l)
        c = cases.get(r["id"])
        if c and "scores" in r:
            p = _p(r["scores"])
            top = c["options"][int(np.argmax(p))]
            got[r["id"]] = {"c": c, "p": p, "top": top, "ok": int(np.argmax(p)) in set(c["good"]),
                            "conf": float(p.max())}
    base = defaultdict(dict)
    for cid, g in got.items():
        base[g["c"]["base"]][g["c"]["variant"]] = g

    def ci(k, n):
        _, lo, hi = wilson(k, n)
        return f"{k / n:.0%} ({lo:.0%}-{hi:.0%})" if n else "-"

    def agree(vs, var):
        pairs = [(v["base"]["top"], v[var]["top"]) for v in vs if "base" in v and var in v]
        return f"{np.mean([a == b for a, b in pairs]):.0%}" if pairs else "-"
    L = ["# Primitive questions: which kinds can GLiNER-Decide answer?", "",
         f"Built by `lab/methods/primitives.py`. `right` = the top answer is right (base wording). `chance` = the "
         f"share right when picking at random. `same: wording / order / filler` = how often the top answer stays "
         f"the same with another wording, the answers in reverse order, or two unrelated sentences added. `sure` "
         f"= the share of questions where the top probability is {sure} or more, and how often those are right.",
         "", "## Known answers", "",
         "| group | format | questions | right | chance | same: wording | same: order | same: filler | "
         "right with a lure | sure: share | sure: right |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    groups = defaultdict(list)
    for b, v in base.items():
        if "base" in v:
            c = v["base"]["c"]
            groups[(c["group"], c["fmt"])].append(v)
    for g in KNOWN:
        for fmt in ("pick", "yesno", "entity", "category"):
            vs = groups.get((g, fmt))
            if not vs:
                continue
            ok = [v["base"]["ok"] for v in vs]
            chance = np.mean([len(v["base"]["c"]["good"]) / len(v["base"]["c"]["options"]) for v in vs])
            sure_v = [v["base"] for v in vs if v["base"]["conf"] >= sure]
            lu = [v["lure"]["ok"] for v in vs if "lure" in v]
            L.append(f"| {g} | {fmt} | {len(vs)} | {ci(sum(ok), len(ok))} | {chance:.0%} | {agree(vs, 'wording1')}"
                     f", {agree(vs, 'wording2')} | {agree(vs, 'order')} | {agree(vs, 'filler')} | "
                     f"{ci(sum(lu), len(lu)) if lu else '-'} | "
                     f"{len(sure_v) / len(vs):.0%} | {ci(sum(x['ok'] for x in sure_v), len(sure_v))} |")
    # splits worth seeing
    L += ["", "### Splits", ""]
    use = groups.get(("use", "yesno"), [])
    for kind in ("good", "bad", "near"):
        vs = [v for v in use if v["base"]["c"].get("item_kind") == kind]
        if vs:
            L.append(f"- use, {kind} things: {ci(sum(v['base']['ok'] for v in vs), len(vs))} right")
    fb = groups.get(("false belief", "pick"), []) + groups.get(("false belief", "yesno"), [])
    for q in ("look", "now", "know"):
        for saw in (False, True):
            vs = [v for v in fb if v["base"]["c"]["base"].startswith(q) and v["base"]["c"]["saw"] == saw]
            if vs:
                L.append(f"- false belief, '{q}', {'watched' if saw else 'was away'}: "
                         f"{ci(sum(v['base']['ok'] for v in vs), len(vs))} right")
    for g in ("presence", "inner state", "danger", "roles"):
        vs = groups.get((g, "yesno"), [])
        for ans, name in ((0, "yes"), (1, "no")):
            s = [v for v in vs if v["base"]["c"]["good"] == [v["base"]["c"]["options"].index(YESNO[ans])]]
            if s:
                L.append(f"- {g}, yes/no, the answer is '{name}': {ci(sum(v['base']['ok'] for v in s), len(s))} right")
    # unknown answers
    L += ["", "## Unknown answers: direction and stability", "",
          "`direction` = the share of scenes where the trait sentence raises the probability of the expected answer "
          "over no trait sentence (0.5 = no effect). `shift` = the mean rise. `top` = the most common top answer "
          "with that trait.", "",
          "| group | trait | expected | direction | shift | top | same: wording | same: order | same: filler |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for g in UNKNOWN:
        vs = [v for (gg, fm), x in groups.items() if gg == g and fm != "grid" for v in x]
        none = {v["base"]["c"]["scene"]: v["base"] for v in vs if v["base"]["c"]["trait"] == "none"}
        traits = sorted({v["base"]["c"]["trait"] for v in vs} - {"none"})
        for t in traits:
            tv = [v for v in vs if v["base"]["c"]["trait"] == t]
            exp = tv[0]["base"]["c"].get("expect")
            dirs, shifts = [], []
            if exp:
                for v in tv:
                    b = v["base"]
                    n0 = none.get(b["c"]["scene"])
                    if n0:
                        i, j = b["c"]["options"].index(exp), n0["c"]["options"].index(exp)
                        dirs.append(b["p"][i] > n0["p"][j])
                        shifts.append(b["p"][i] - n0["p"][j])
            tops = defaultdict(int)
            for v in tv:
                tops[v["base"]["top"]] += 1
            top = max(tops, key=tops.get)
            L.append(f"| {g} | {t} | {exp or '-'} | {ci(sum(dirs), len(dirs)) if dirs else '-'} | "
                     f"{np.mean(shifts):+.2f} | {top} ({tops[top]}/{len(tv)}) | "
                     if shifts else f"| {g} | {t} | {exp or '-'} | - | - | {top} ({tops[top]}/{len(tv)}) | ")
            L[-1] += f"{agree(tv, 'wording1')}, {agree(tv, 'wording2')} | {agree(tv, 'order')} | {agree(tv, 'filler')} |"
    # the trait x scene grid
    grid = groups.get(("feeling", "grid"), [])
    if grid:
        L += ["", "## Trait x scene: does the scene still matter once a trait is given?", "",
              "Each row is one trait sentence (or none) over the same scenes. `scene right` = the share of scenes "
              "whose top feeling is the scene's own (glad for a gift, sad for a loss, ...). `scene moves it` = the "
              "share of scenes where the scene's own feeling is more likely than in the other scenes (0.5 = the "
              "scene does not matter: 'state-blind'). `same as empty` = the share of scenes whose top feeling is "
              "the same as with the empty scene ('Tom is there.').", "",
              "| trait | top feeling in each scene | scene right | scene moves it | same as empty |",
              "| --- | --- | --- | --- | --- |"]
        by_t = defaultdict(dict)
        for v in grid:
            c = v["base"]["c"]
            by_t[c["trait"]][c["scene"]] = v["base"]
        for t, sc in by_t.items():
            empty = sc.get("Tom is there.")
            real = {k: b for k, b in sc.items() if b["c"]["scene_feel"]}
            right = [b["top"] == b["c"]["scene_feel"] for b in real.values()]
            moves = []
            for k, b in real.items():
                i = FEEL.index(b["c"]["scene_feel"])
                others = [o["p"][i] for kk, o in real.items() if kk != k]
                moves.append(b["p"][i] > np.mean(others))
            same = [b["top"] == empty["top"] for b in real.values()] if empty else []
            tops = ", ".join(f"{b['c']['scene_feel']}: {b['top']}" for b in real.values())
            L.append(f"| {t} | {tops} | {np.mean(right):.0%} | {np.mean(moves):.0%} | "
                     f"{np.mean(same):.0%} |" if same else f"| {t} | {tops} | {np.mean(right):.0%} | "
                     f"{np.mean(moves):.0%} | - |")
    (RES / "primitives.md").write_text("\n".join(L) + "\n")
    print(RES / "primitives.md")


if __name__ == "__main__":
    report() if sys.argv[1:] == ["report"] else build()
