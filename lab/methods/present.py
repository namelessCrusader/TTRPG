"""How should the problem be SHOWN to a model? The presentation battery.

    python3 -m lab.methods.present build      # cases_present.jsonl
    lab/methods/present_queue.sh              # every model, one at a time
    python3 -m lab.methods.present report     # lab/results/methods/present.md

Workstream A changed the model and the scoring method. Every test gave the
model the same kind of input: a sentence of situation, a question, and each
option as a short name. This battery changes HOW the problem is presented,
one thing at a time from that default, on the same 120 lists (16 and 64
options) that the ablation uses.

THE NEED (the situation and the question)
    default     "Kael needs to cut the rope. He looks at what is in the storeroom."
                + "Which thing can Kael use for this?"
    ctx_q       the need moves into the question: "Which thing can Kael use to cut the rope?"
    ctx_kv      key and value lines: "Who: Kael / Goal: cut the rope / Where: the storeroom"
    ctx_json    the same as JSON
    ctx_you     second person: "You need to cut the rope." / "Which thing can you use?"
    ctx_instr   no story, an order: "Pick one thing that can be used to cut the rope."
    ctx_cap     a hint of the capability needed: "... For that he needs something with
                a sharp edge." Hand-written, no item names. An ORACLE: if it helps a
                lot, a first step that turns a need into a capability is worth learning.

THE OPTIONS
    opt_desc    name and description: "the kitchen knife: a short blade with a wooden handle"
    opt_props   name and the world's tags: "the kitchen knife (cut, pierce, sharp, weapon)".
                These are the tags the labels come from, so this measures whether a
                model can USE capabilities it is given, not whether it knows them.
                The word-overlap baseline shows how much is plain string matching.
    opt_cat     name and kind: "the kitchen knife (kitchen things)"
    opt_num     numbered: "7. the kitchen knife"

HOW QUERY AND OPTIONS COMBINE (model-specific, on the default input only)
    onli2b:use, onli2b:work     other hypothesis wordings for the entailment model
    qwen17:cloze, cloze_cal     a sentence to complete ("... He takes the kitchen
                                knife.") instead of question and answer
    <model>:hier                two steps, as the grouped pick-list: choose the kind
                                of thing first, then the thing within that kind

A WORD-OVERLAP BASELINE (overlap:all) runs on every variant: a variant that
only helps because it leaks words of the answer shows up there.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

from lab.methods.cases import parse
from lab.methods.report import wilson

HERE = Path(__file__).parent
RES = HERE.parent / "results" / "methods"
SIZES = (16, 64)
VARIANTS = ("default", "ctx_q", "ctx_kv", "ctx_json", "ctx_you", "ctx_instr", "ctx_cap",
            "opt_desc", "opt_props", "opt_cat", "opt_num")

# the capability each need asks for, in words that name no item
CAPABILITY = {
    "cut the rope": "something with a sharp edge",
    "light the dark cellar": "something that gives light",
    "carry water from the well": "something that holds water",
    "dig up the buried box": "something to dig with",
    "tie up the thief": "something long and strong to tie with",
    "climb to the high window": "something to climb on",
    "start a fire in the hearth": "something that makes a flame or a spark",
    "write a letter home": "something that makes marks",
    "catch fish in the river": "something made for catching fish",
    "fix the loose nail in the door": "something heavy and hard to hit with",
    "see who is far down the road": "something that makes far things look near",
    "stay dry in the storm": "something that keeps the rain off",
    "pay the ferryman": "something of value",
    "feed the hungry horse": "something a horse eats",
    "chop firewood": "something with a heavy blade",
    "treat a bleeding cut": "something to cover and press on a wound",
    "keep warm on a freezing night": "something thick and warm to wear or wrap in",
    "open a locked chest whose key is lost": "something to pry the lid or pick the lock with",
    "signal a ship far out at sea": "something that can be seen from far away",
    "call the village for help": "something that makes a loud sound",
    "cross the flooded stream": "something that floats or bridges the water",
    "carry the harvest to the barn": "something to carry a big load in",
    "cook the stew": "something to cook in over a fire",
    "drink on a long walk": "something to drink",
    "fight off the wolf": "something to hit or stab with",
    "reach the apple high in the tree": "something long",
    "mend the torn cloak": "something to sew with",
    "measure the length of the field": "something marked in equal lengths",
    "weigh the silver": "something that tells how heavy a thing is",
    "clean the muddy floor": "something to wipe or sweep with",
    "hide the stolen coins": "something that closes around small things",
    "block the arrows": "something broad and hard to hide behind",
    "break the ice on the pond": "something heavy and hard to hit with",
    "trap a rabbit": "something made to catch small animals",
    "sharpen the blade": "something rough and hard to rub a blade on",
    "sleep outside": "something soft or warm to lie in",
    "eat something before the journey": "some food",
    "pour the oil into the lamp": "something that pours neatly without spilling",
}
KIND_WORDS = {
    "tool": "tools", "kitchen": "kitchen things", "light": "lights and lamps", "farm": "farm things",
    "travel": "travel gear", "weapon": "weapons", "writing": "writing things", "nature": "natural things",
    "clothing": "clothing", "food": "food", "container": "containers", "music": "musical instruments",
    "sewing": "sewing things", "cleaning": "cleaning things", "medical": "medical supplies",
    "money": "money and valuables", "toy": "toys and games", "fishing": "fishing gear",
    "religious": "holy things", "decor": "decorations",
}
PLACE = "the storeroom"


def _variant(c, v, by):
    need = c["need"]
    sit, q = c["situation"], c["question"]
    names = c["names"]
    opts = list(names)
    if v == "ctx_q":
        sit, q = "Kael looks at what is in the storeroom.", f"Which thing can Kael use to {need}?"
    elif v == "ctx_kv":
        sit = f"Who: Kael\nGoal: {need}\nWhere: {PLACE}"
    elif v == "ctx_json":
        sit = json.dumps({"who": "Kael", "goal": need, "where": PLACE})
    elif v == "ctx_you":
        sit, q = f"You need to {need}. You look at what is in the storeroom.", "Which thing can you use for this?"
    elif v == "ctx_instr":
        sit, q = "The things in the storeroom are listed.", f"Pick one thing that can be used to {need}."
    elif v == "ctx_cap":
        sit = f"Kael needs to {need}. For that he needs {CAPABILITY[need]}. He looks at what is in the storeroom."
    elif v == "opt_desc":
        opts = [f"{n}: {by[n]['desc']}" for n in names]
    elif v == "opt_props":
        opts = [f"{n} ({', '.join(sorted(p.replace('_', ' ') for p in by[n]['props']))})" for n in names]
    elif v == "opt_cat":
        opts = [f"{n} ({KIND_WORDS[by[n]['cat']]})" for n in names]
    elif v == "opt_num":
        opts = [f"{i + 1}. {n}" for i, n in enumerate(names)]
    return sit, q, opts


def build():
    items, _ = parse()
    by = {i["name"]: i for i in items}
    src = [json.loads(l) for l in open(HERE / "cases_things.jsonl")]
    src = [c for c in src if c["factor"] == "size" and c["value"] in SIZES]
    out = []
    for c in src:
        for v in VARIANTS:
            sit, q, opts = _variant(c, v, by)
            out.append(dict(c, id=f"present/{v}/{c['id']}", factor="present", value=v, size=c["value"],
                            orig=c["id"], situation=sit, question=q, options=opts))
    with open(HERE / "cases_present.jsonl", "w") as f:
        for c in out:
            f.write(json.dumps(c) + "\n")
    print(f"{len(out)} cases -> {HERE / 'cases_present.jsonl'}")


def report():
    cases = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "cases_present.jsonl")}
    runs = {}
    for f in sorted(glob.glob(str(RES / "*.present.jsonl"))):
        name = os.path.basename(f)[: -len(".present.jsonl")].replace("__", ":")
        runs[name] = [json.loads(l) for l in open(f)]
    L = ["# How to present the problem: the battery", "",
         "Share of picks in the acceptable set, 60 lists per cell (about +/-12 points). Each column changes ONE "
         "thing from `default` (see the top of `lab/methods/present.py`). Bold = at least 10 points above the "
         "same model's default. `overlap:all` is the word-overlap baseline: a column where it jumps leaks words "
         "of the answer.", ""]
    for size in SIZES:
        L += [f"## {size} options", "", "| model | " + " | ".join(VARIANTS) + " |",
              "| --- |" + " --- |" * len(VARIANTS)]
        for name, rows in runs.items():
            by = {}
            for r in rows:
                c = cases.get(r["id"])
                if c and c["size"] == size and "pick" in r:
                    by.setdefault(c["value"], []).append(r["ok"])
            if not by:
                continue
            base = sum(by.get("default", [])) / max(1, len(by.get("default", [])))
            cells = []
            for v in VARIANTS:
                xs = by.get(v)
                if not xs:
                    cells.append("")
                    continue
                p = sum(xs) / len(xs)
                t = f"{p:.0%}"
                cells.append(f"**{t}**" if p >= base + 0.10 and v != "default" else t)
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L.append("")
    (RES / "present.md").write_text("\n".join(L) + "\n")
    print(RES / "present.md")


if __name__ == "__main__":
    {"build": build, "report": report}[sys.argv[1]]()
