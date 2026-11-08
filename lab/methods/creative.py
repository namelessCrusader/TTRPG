"""Creativity tests, layers 1 and 2 (layer 3 is lab/judge/signals.py).

    python3 -m lab.methods.creative cases     # writes lab/methods/cases_creative.jsonl
                                              # and lab/results/methods/creative_labels.md
    lab/.venv/bin/python -m lab.methods.run qwen17:one --cases lab/methods/cases_creative.jsonl --tag creative
    python3 -m lab.methods.creative report    # writes lab/results/methods/creative.md

CREATIVE = UNEXPECTED AND STILL WORKS. Random is not creative. So each test
pairs "unusual" with "valid".

LAYER 1, RESOURCEFULNESS. The obvious tools are taken away. Can the model
find the unusual thing that still does the job? The rope is cut with the
chisel, the arrows block behind the hay bale, the silver is weighed on the
yoke. Three cells at 8 options, the same needs in each:

    obvious        1 good (obvious) item + 7 bad         (the control)
    unusual        1 unusual-but-workable item + 7 bad
    unusual_near   1 unusual-but-workable item + 7 hand-picked near misses
                   (things that SEEM to do the job and cannot)

LAYER 2, VARIETY WITHOUT NONSENSE. Characters draw their action from the
scores; they do not always take the best. One cell at 16 options, 2 obvious +
3 unusual + 11 bad, with every score kept. From one pass per case, the report
works out what the draw does at each temperature: how often it is still
workable, how many DIFFERENT workable things 10 draws give, how often the
draw picks an unusual one, and how often nonsense. The curve (variety
against nonsense) does not depend on how a model scales its scores.

LABELS. The rule labels in cases.py are too loose for this (they say a frying
pan sharpens a blade), so the unusual items are picked by hand, one need at a
time. DOUBT items are arguable either way and are left out of every list, as
the rule's "possible" items are. creative_labels.md is for the human audit.
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from lab.methods.cases import NEARMISS, label, parse, situation_text

HERE = Path(__file__).parent
RES = HERE.parent / "results" / "methods"

# need: (UNUSUAL but workable, and how) ; DOUBT (arguable: in no list)
UNUSUAL = {
    "cut the rope": (
        "the chisel, the spear, the arrows, the torch, the candle, the brazier, the glass bottle",
        "the file, the lantern, the oil lamp, the shell, the mirror, the pickaxe, the fish hook, the pliers, the wire, the flat stone, the rock, the bone"),
    "light the dark cellar": (
        "the tinderbox, the flint and steel",
        "the mirror, the magnifying lens"),
    "carry water from the well": (
        "the helmet, the tarp, the oilskin coat, the vase, the churn, the sponge, the honey jar, the boots",
        "the frying pan, the salt jar, the ink pot, the shell, the wheelbarrow, the mortar and pestle, the bottle of wine, the bottle of spirits"),
    "dig up the buried box": (
        "the frying pan, the iron pot, the helmet, the crowbar, the flat stone, the shell",
        "the bucket, the bowl, the plate, the tin cup, the axe, the sword, the spear, the kitchen knife, the dagger, the chisel, the cauldron, the hatchet"),
    "tie up the thief": (
        "the scarf, the bandages, the bolt of cloth, the tea cloth, the fishing rod, the snare, the blanket",
        "the cage, the sack, the tarp, the leather patch, the yoke, the padlock"),
    "climb to the high window": (
        "the barrel, the wooden box, the hay bale, the plank, the log, the wheelbarrow, the cauldron, the tent pegs",
        "the pole, the net, the knotted cord, the chain, the lasso, the churn, the pickaxe"),
    "start a fire in the hearth": (
        "the lantern, the oil lamp, the brazier",
        "the magnifying lens, the bottle of spirits, the file, the rock"),
    "write a letter home": (
        "the needle and thread, the chisel, the kitchen knife, the dagger, the brick",
        "the stick, the candle, the ink pot, the feather, the bone, the rock, the nails, the sealing wax"),
    "catch fish in the river": (
        "the pitchfork, the needle and thread, the wire, the bolt of cloth, the sack",
        "the bow, the arrows, the hairpin, the sword, the dagger, the club, the tea cloth, the scarf"),
    "fix the loose nail in the door": (
        "the axe, the hatchet, the horseshoe, the flat stone, the boots, the mortar and pestle, the helmet, the cleaver",
        "the kettle, the cauldron, the log, the grindstone, the shield, the bottle of wine, the sword, the dagger"),
    "see who is far down the road": (
        "the ladder, the barrel, the wooden box, the hay bale",
        "the horn, the whistle, the bell, the flag, the kite"),
    "stay dry in the storm": (
        "the shield, the boat, the barrel, the wheelbarrow, the raft",
        "the plank, the painting, the cutting board, the helmet, the cauldron, the wool cloak, the blanket, the bolt of cloth, the wooden box"),
    "pay the ferryman": (
        "the bottle of wine, the bottle of spirits, the dried meat, the cheese, the honey jar, the lute, the sword, the painting",
        "the salt jar, the dice, the flute, the fur hat, the boots, the book, the loaf of bread, the salted fish, the dagger"),
    "feed the hungry horse": (
        "the herbs",
        "the loaf of bread, the salt jar, the honey jar"),
    "chop firewood": (
        "the sword, the pickaxe, the sickle, the mace",
        "the crowbar, the dagger, the scythe, the club, the spear"),
    "treat a bleeding cut": (
        "the tea cloth, the bolt of cloth, the towel, the honey jar, the bottle of spirits, the salve, the needle and thread, the herbs, the torch",
        "the scarf, the apron, the blanket, the sponge, the leather patch, the soap, the bottle of wine, the salt jar"),
    "keep warm on a freezing night": (
        "the brazier, the hay bale, the bolt of cloth, the tent, the flint and steel, the tinderbox",
        "the torch, the lantern, the candle, the oil lamp, the sack, the kettle, the bottle of spirits, the tarp"),
    "open a locked chest whose key is lost": (
        "the axe, the hatchet, the pickaxe, the saw, the dagger, the pliers, the file, the fish hook, the rock, the mace",
        "the kitchen knife, the sword, the brick, the club, the torch, the iron pot, the frying pan, the chain, the cleaver, the spear"),
    "signal a ship far out at sea": (
        "the torch, the lantern, the brazier, the bolt of cloth, the blanket, the tarp, the shield",
        "the helmet, the horn, the bell, the oil lamp, the candle, the flint and steel, the tinderbox, the firewood"),
    "call the village for help": (
        "the iron pot, the frying pan, the kettle, the cauldron, the shield, the hammer, the torch, the flag",
        "the chain, the bow, the lute, the horseshoe, the mirror"),
    "cross the flooded stream": (
        "the barrel, the pole, the waterskin",
        "the lasso, the rope, the ladder, the wooden box, the shield, the empty flask, the hay bale, the cauldron, the tarp, the churn"),
    "carry the harvest to the barn": (
        "the barrel, the wooden box, the apron, the net",
        "the cauldron, the cage, the tarp, the blanket, the bucket, the bolt of cloth, the raft, the boat"),
    "cook the stew": (
        "the helmet, the bucket, the clay jug",
        "the shovel, the tin cup, the brazier, the shield, the bowl"),
    "drink on a long walk": (
        "the apple",
        "the tin cup, the empty flask, the glass bottle, the clay jug, the eggs, the honey jar"),
    "fight off the wolf": (
        "the frying pan, the shovel, the crowbar, the rock, the lantern, the horn, the drum, the dried meat, the pole, the shield",
        "the iron pot, the rolling pin, the brick, the chain, the brazier, the candle, the bell, the whistle, the salted fish, the stick, the lasso"),
    "reach the apple high in the tree": (
        "the sling, the rock, the pebble, the bow, the barrel, the wooden box, the axe",
        "the ball, the lasso, the rope, the club, the saw, the arrows, the hay bale, the chain"),
    "mend the torn cloak": (
        "the hairpin, the twine",
        "the wire, the fish hook, the leather patch, the sealing wax, the rope"),
    "measure the length of the field": (
        "the chain, the wheelbarrow, the plank, the twine",
        "the rope, the stick, the pole, the ladder, the boots"),
    "weigh the silver": (
        "the yoke, the plank",
        "the measuring rod, the copper coins, the fishing rod, the bucket, the pole"),
    "clean the muddy floor": (
        "the shovel, the hoe, the apron, the scarf, the bolt of cloth, the sack",
        "the bucket, the rake, the hay bale, the wool cloak, the pillow, the blanket"),
    "hide the stolen coins": (
        "the shovel, the boots, the pillow, the honey jar, the hay bale, the vase, the loaf of bread, the hoe",
        "the book, the tarp, the cauldron, the iron pot, the kettle, the clay jug, the churn, the bedroll"),
    "block the arrows": (
        "the barrel, the plank, the cauldron, the wheelbarrow, the boat, the log, the hay bale, the frying pan",
        "the cutting board, the helmet, the raft, the wooden box, the tarp, the iron pot"),
    "break the ice on the pond": (
        "the axe, the hatchet, the spear, the log, the chisel, the sword",
        "the torch, the kettle, the pole, the pitchfork, the cleaver, the flat stone"),
    "trap a rabbit": (
        "the basket, the wooden box, the sack, the wire, the twine",
        "the bucket, the rope, the lasso, the bow, the sling, the barrel, the fishing rod"),
    "sharpen the blade": (
        "the brick, the clay jug, the plate, the leather patch, the boots, the bowl",
        "the rock, the pebble, the whetstone, the tin cup, the bone"),
    "sleep outside": (
        "the hay bale, the wool cloak, the net, the boat, the bolt of cloth, the sack",
        "the tarp, the oilskin coat, the raft, the saddle, the fur hat, the scarf"),
    "eat something before the journey": (
        "the herbs, the fishing rod, the bow, the sling",
        "the sack of oats, the pinecone, the snare, the fish hook, the net, the bottle of wine"),
    "pour the oil into the lamp": (
        "the tin cup, the ladle, the kettle, the parchment, the empty flask, the glass bottle",
        "the bowl, the wooden spoon, the bottle of wine, the bottle of spirits, the waterskin"),
}

SIZE1, SIZE2 = 8, 16          # layer 1 lists, layer 2 lists
MIX_GOOD, MIX_UNUSUAL = 2, 3  # layer 2: obvious and unusual items in each list


def _split(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def sets(items, needs):
    """For each need: good, unusual, bad (usable as a distractor), near (hand
    near misses that are bad here), and the problems the audit should see."""
    by = {i["name"]: i for i in items}
    out, problems = {}, []
    for nd in needs:
        name = nd["need"]
        uns, dts = (_split(x) for x in UNUSUAL[name])
        for n in uns + dts:
            if n not in by:
                problems.append(f"{name}: unknown item '{n}'")
        uns = [n for n in uns if n in by]
        dts = [n for n in dts if n in by]
        good = [i["name"] for i in items if label(i, nd) == "good"]
        for n in uns:
            if n in good:
                problems.append(f"{name}: '{n}' is already good")
        near_all = _split(NEARMISS.get(name, ""))
        for n in uns:      # only near misses the rule calls bad were ever used as one
            if n in near_all and label(by[n], nd) == "bad":
                problems.append(f"{name}: '{n}' is unusual here but a hand near miss in cases.py")
        keep_out = set(good) | set(uns) | set(dts)
        bad = [i["name"] for i in items if label(i, nd) == "bad" and i["name"] not in keep_out]
        near = [n for n in near_all if n in bad]
        rule_possible = [i["name"] for i in items if label(i, nd) == "possible"]
        out[name] = {"good": good, "unusual": [n for n in uns if n not in good],
                     "doubt": dts, "bad": bad, "near": near, "rule_possible": rule_possible}
    return out, problems


def _case(cid, value, need, names, good_names, unusual_names, rng):
    rng.shuffle(names)
    return {"id": cid, "kind": "creative", "factor": "resource", "value": value, "need": need,
            "situation": situation_text(need, "short", rng),
            "question": "Which thing can Kael use for this?",
            "options": names, "names": names,
            "good": [i for i, n in enumerate(names) if n in good_names or n in unusual_names],
            "obvious": [i for i, n in enumerate(names) if n in good_names],
            "unusual": [i for i, n in enumerate(names) if n in unusual_names],
            "keep_scores": True}


def build(n_per_cell=60, seed=0):
    items, needs = parse()
    S, problems = sets(items, needs)
    cases = []
    # LAYER 1: the same needs and the same bad items in "obvious" and
    # "unusual"; only the one workable item differs.
    l1 = [n for n, s in S.items() if s["unusual"] and s["good"] and len(s["bad"]) >= SIZE1]
    rng = random.Random(f"{seed}-l1")
    for k in range(n_per_cell):
        need = l1[k % len(l1)]
        s = S[need]
        bad = rng.sample(s["bad"], SIZE1 - 1)
        g, u = rng.choice(s["good"]), rng.choice(s["unusual"])
        cases.append(_case(f"creative/obvious/{k}", "obvious", need, bad + [g], {g}, set(), random.Random(f"{seed}-o{k}")))
        cases.append(_case(f"creative/unusual/{k}", "unusual", need, bad + [u], set(), {u}, random.Random(f"{seed}-u{k}")))
    ln = [n for n, s in S.items() if s["unusual"] and len(s["near"]) >= SIZE1 - 1]
    rng = random.Random(f"{seed}-ln")
    for k in range(n_per_cell):
        need = ln[k % len(ln)]
        s = S[need]
        u = rng.choice(s["unusual"])
        cases.append(_case(f"creative/unusual_near/{k}", "unusual_near", need,
                           rng.sample(s["near"], SIZE1 - 1) + [u], set(), {u}, rng))
    # LAYER 2
    l2 = [n for n, s in S.items() if len(s["good"]) >= MIX_GOOD and len(s["unusual"]) >= MIX_UNUSUAL
          and len(s["bad"]) >= SIZE2 - MIX_GOOD - MIX_UNUSUAL]
    rng = random.Random(f"{seed}-mix")
    for k in range(n_per_cell):
        need = l2[k % len(l2)]
        s = S[need]
        g = rng.sample(s["good"], MIX_GOOD)
        u = rng.sample(s["unusual"], MIX_UNUSUAL)
        b = rng.sample(s["bad"], SIZE2 - MIX_GOOD - MIX_UNUSUAL)
        cases.append(_case(f"creative/mixed/{k}", "mixed", need, g + u + b, set(g), set(u), rng))
    return cases, S, problems, {"obvious/unusual": l1, "unusual_near": ln, "mixed": l2}


def write_cases():
    cases, S, problems, used = build()
    out = HERE / "cases_creative.jsonl"
    with out.open("w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    lines = ["# Creativity labels (for your audit)", "",
             "UNUSUAL = not the obvious tool, but it does the job. DOUBT = arguable, so it is in no list. "
             "Mark any you disagree with.", ""]
    if problems:
        lines += ["## Problems found by the check", ""] + [f"- {p}" for p in problems] + [""]
    lines += ["## Needs used by each cell", ""]
    lines += [f"- {k} ({len(v)}): {', '.join(v)}" for k, v in used.items()] + [""]
    lines += ["## Each need", ""]
    for need, s in S.items():
        lines.append(f"- **{need}**")
        lines.append(f"  - obvious: {', '.join(s['good'])}")
        lines.append(f"  - unusual: {', '.join(s['unusual']) or '(none)'}")
        lines.append(f"  - doubt: {', '.join(s['doubt']) or '(none)'}")
        rest = [n for n in s["rule_possible"] if n not in s["unusual"] and n not in s["doubt"]]
        if rest:
            lines.append(f"  - rule said possible, not picked (in no list): {', '.join(rest)}")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "creative_labels.md").write_text("\n".join(lines) + "\n")
    print(f"{len(cases)} cases -> {out}")
    for k, v in used.items():
        print(f"  {k}: {len(v)} needs")
    for p in problems:
        print("  PROBLEM", p)


# ---------------------------------------------------------------- report
def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0, c - h), min(1, c + h))


def logits(rec):
    """Scores as log-weights for a draw. Probabilities (and Laya's yes-chance
    for each option) are logged; log-probabilities and raw scores are used as
    they are."""
    sc = rec["scores"]
    if all(0 <= x <= 1 for x in sc) and not rec["scorer"].endswith((":apart", ":solo")):
        return [math.log(max(x, 1e-12)) for x in sc]
    return sc


def draw(z, t):
    m = max(z)
    e = [math.exp((x - m) / t) for x in z]
    s = sum(e)
    return [x / s for x in e]


K_DRAWS = 10
TEMPS = [10 ** (x / 10) for x in range(-20, 21)]      # 0.01 .. 100, in units of the model's own spread


def curve(rows):
    """For each temperature: the average over cases of what the draw does."""
    spread = sorted(_std(logits(r)) for r in rows)
    unit = spread[len(spread) // 2] or 1.0             # a model's own scale, so T=1 means "about its usual gap"
    out = []
    for t in TEMPS:
        acc = defaultdict(float)
        for r in rows:
            p = draw(logits(r), t * unit)
            work = set(r["good"])
            uns = set(r["unusual"])
            pw = sum(p[i] for i in work)
            acc["workable"] += pw
            acc["unusual"] += sum(p[i] for i in uns)
            acc["distinct"] += sum(1 - (1 - p[i]) ** K_DRAWS for i in work)
            acc["distinct_unusual"] += sum(1 - (1 - p[i]) ** K_DRAWS for i in uns)
        n = len(rows)
        out.append({"t": t, **{k: v / n for k, v in acc.items()}, "nonsense": 1 - acc["workable"] / n})
    return out


def _std(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def at_nonsense(cv, level):
    """The draw's behaviour at the temperature where nonsense reaches `level`
    (interpolated). None if the model is past it even when it is greedy."""
    if cv[0]["nonsense"] > level:
        return None
    for a, b in zip(cv, cv[1:]):
        if a["nonsense"] <= level <= b["nonsense"]:
            w = 0 if b["nonsense"] == a["nonsense"] else (level - a["nonsense"]) / (b["nonsense"] - a["nonsense"])
            return {k: a[k] + w * (b[k] - a[k]) for k in a}
    return cv[-1]


def report():
    cases = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "cases_creative.jsonl")}
    runs = {}
    for f in sorted(glob.glob(str(RES / "*.creative.jsonl"))):
        name = os.path.basename(f)[: -len(".creative.jsonl")].replace("__", ":")
        rows = []
        for l in open(f):
            r = json.loads(l)
            if r["id"] in cases:
                r.update({k: cases[r["id"]][k] for k in ("obvious", "unusual", "names")}, scorer=name)
                rows.append(r)
        runs[name] = rows
    out = ["# Creativity tests, layers 1 and 2", "",
           "Creative = unexpected AND still works. Labels: `creative_labels.md` (audit them). "
           "60 cases per cell, so a cell is good to about +/-12 points.", "",
           "## Layer 1: resourcefulness (8 options)", "",
           "Share of picks that do the job. `obvious`: one ordinary tool among 7 useless things. "
           "`unusual`: the ordinary tools are gone; one unusual-but-workable thing among the same 7. "
           "`unusual, near misses`: the 7 others SEEM useful and are not. "
           "`drop` = obvious minus unusual: how much the model relies on the ordinary tool.", "",
           "| scorer | obvious | unusual | unusual, near misses | drop | rank of the unusual thing (median) |",
           "| --- | --- | --- | --- | --- | --- |",
           "| chance | 12% | 12% | 12% | | 4.5 |"]
    for name, rows in runs.items():
        cells, ps = [], {}
        for v in ("obvious", "unusual", "unusual_near"):
            rs = [r for r in rows if r["value"] == v and "error" not in r]
            p, lo, hi = wilson(sum(r["ok"] for r in rs), len(rs))
            ps[v] = p
            cells.append(f"{p:.0%} ({lo:.0%}-{hi:.0%})" if rs else "")
        ranks = sorted(r["first"] for r in rows if r["value"] == "unusual" and "first" in r)
        med = ranks[len(ranks) // 2] if ranks else ""
        out.append(f"| {name} | " + " | ".join(cells) + f" | {ps['obvious'] - ps['unusual']:+.0%} | {med} |")
    out += ["", "## Layer 2: variety without nonsense (16 options: 2 obvious, 3 unusual, 11 useless)", "",
            "The character DRAWS its action from the scores. A higher temperature gives more variety and "
            "more nonsense. Each row reads the draw at the temperature where nonsense reaches that level, "
            "so models are compared at the same amount of nonsense. `different` = how many different "
            f"workable things {K_DRAWS} draws give (of 5). `unusual` = share of draws that are unusual and workable. "
            "`greedy` = always the top score. A dash: the model makes more nonsense than that even when greedy.", "",
            "| scorer | greedy: workable | greedy: unusual | 10% nonsense: different / unusual | "
            "20%: different / unusual | 30%: different / unusual |",
            "| --- | --- | --- | --- | --- | --- |",
            "| uniform draw | 31% | 19% | - | - | - |"]
    curves = {}
    for name, rows in runs.items():
        rs = [r for r in rows if r["value"] == "mixed" and "scores" in r]
        if len(rs) < 10:
            continue
        cv = curve(rs)
        curves[name] = cv
        g = [r for r in rows if r["value"] == "mixed" and "pick" in r]
        gw = sum(r["pick"] in r["good"] for r in g) / len(g)
        gu = sum(r["pick"] in r["unusual"] for r in g) / len(g)
        cells = []
        for lv in (0.10, 0.20, 0.30):
            a = at_nonsense(cv, lv)
            cells.append("-" if a is None else f"{a['distinct']:.1f} / {a['unusual']:.0%}")
        out.append(f"| {name} | {gw:.0%} | {gu:.0%} | " + " | ".join(cells) + " |")
    no_scores = [n for n, rows in runs.items() if n not in curves]
    if no_scores:
        out += ["", f"No layer 2 row (the method gives no score for each option): {', '.join(no_scores)}."]
    out += ["", "## Layer 3", "",
            "Scene-level creativity needs the lab world (Workstream E) and the judge (Workstream C). "
            "The measures are in `lab/judge/signals.py`."]
    (RES / "creative.md").write_text("\n".join(out) + "\n")
    (RES / "creative_curves.json").write_text(json.dumps(curves))
    print(RES / "creative.md")


if __name__ == "__main__":
    {"cases": write_cases, "report": report}[sys.argv[1] if len(sys.argv) > 1 else "cases"]()
