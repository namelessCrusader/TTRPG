"""What does each model actually use? Take parts of the need away.

    python3 -m lab.methods.ablate build      # cases_ablate.jsonl
    run_all.sh --cases lab/methods/cases_ablate.jsonl --tag ablate --keep-scores -- <scorers>
    python3 -m lab.methods.ablate report     # lab/results/methods/ablate.md

The failure analysis (failures.py) found that the models pick what the need
is ABOUT (the rope, for "cut the rope") instead of what does the job. This
test finds out why, one part of the need at a time. The same 16- and
64-option lists as Workstream A, asked five ways:

    full     "Kael needs to cut the rope."                     (as before)
    verb     "Kael needs to cut something."                    (what to do, no object)
    object   "Kael needs to do something with the rope."       (the object, no verb)
    blank    no need at all                                    (the model's prior)
    slot     each option is the whole action: "cut the rope with the kitchen knife"

What it tells us:
    blank pick = full pick in a failure  -> the need was ignored (a prior)
    object pick = full pick in a failure, verb alone is right
                                         -> the object pulled the model away
    verb better than full                -> the object HURTS: it is a lure
    slot better than full                -> phrasing the whole action fixes it
                                            (an inference change, no training)
    calibrated: full score minus blank score, from the kept scores
                                         -> removing the prior fixes it
                                            (contextual calibration, offline)
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from pathlib import Path

from lab.methods.report import wilson

HERE = Path(__file__).parent
RES = HERE.parent / "results" / "methods"
SIZES = (16, 64)
VARIANTS = ("full", "verb", "object", "blank", "slot")

# need: (the verb alone, the object alone)
PARTS = {
    "cut the rope": ("cut something", "the rope"),
    "light the dark cellar": ("light something up", "the dark cellar"),
    "carry water from the well": ("carry something", "the water in the well"),
    "dig up the buried box": ("dig something up", "the buried box"),
    "tie up the thief": ("tie someone up", "the thief"),
    "climb to the high window": ("climb up high", "the high window"),
    "start a fire in the hearth": ("start a fire", "the hearth"),
    "write a letter home": ("write something", "a letter home"),
    "catch fish in the river": ("catch something", "the fish in the river"),
    "fix the loose nail in the door": ("fix something", "the loose nail in the door"),
    "see who is far down the road": ("see far", "whoever is down the road"),
    "stay dry in the storm": ("stay dry", "the storm"),
    "pay the ferryman": ("pay someone", "the ferryman"),
    "feed the hungry horse": ("feed someone", "the hungry horse"),
    "chop firewood": ("chop something", "the firewood"),
    "treat a bleeding cut": ("treat something", "a bleeding cut"),
    "keep warm on a freezing night": ("keep warm", "the freezing night"),
    "open a locked chest whose key is lost": ("open something", "a locked chest whose key is lost"),
    "signal a ship far out at sea": ("signal someone far away", "a ship out at sea"),
    "call the village for help": ("call for help", "the village"),
    "cross the flooded stream": ("cross something", "the flooded stream"),
    "carry the harvest to the barn": ("carry something", "the harvest"),
    "cook the stew": ("cook something", "the stew"),
    "drink on a long walk": ("drink", "the long walk"),
    "fight off the wolf": ("fight something off", "the wolf"),
    "reach the apple high in the tree": ("reach something high up", "the apple in the tree"),
    "mend the torn cloak": ("mend something", "the torn cloak"),
    "measure the length of the field": ("measure something", "the field"),
    "weigh the silver": ("weigh something", "the silver"),
    "clean the muddy floor": ("clean something", "the muddy floor"),
    "hide the stolen coins": ("hide something", "the stolen coins"),
    "block the arrows": ("block something", "the arrows"),
    "break the ice on the pond": ("break something", "the ice on the pond"),
    "trap a rabbit": ("trap something", "a rabbit"),
    "sharpen the blade": ("sharpen something", "the blade"),
    "sleep outside": ("sleep", "the night outside"),
    "eat something before the journey": ("eat", "the journey"),
    "pour the oil into the lamp": ("pour something", "the oil and the lamp"),
}
TAIL = " He looks at what is in the storeroom."


def build():
    src = [json.loads(l) for l in open(HERE / "cases_things.jsonl")]
    src = [c for c in src if c["factor"] == "size" and c["value"] in SIZES]
    out = []
    for c in src:
        verb, obj = PARTS[c["need"]]
        sit = {"full": c["situation"],
               "verb": f"Kael needs to {verb}." + TAIL,
               "object": f"Kael needs to do something with {obj}." + TAIL,
               "blank": "Kael looks at what is in the storeroom.",
               "slot": c["situation"]}
        for v in VARIANTS:
            d = dict(c, id=f"ablate/{v}/{c['id']}", factor="ablate", value=v, size=c["value"],
                     orig=c["id"], situation=sit[v])
            if v == "slot":
                d["question"] = "What should Kael do?"
                d["options"] = [f"{c['need']} with {n}" for n in c["names"]]
            out.append(d)
    with open(HERE / "cases_ablate.jsonl", "w") as f:
        for c in out:
            f.write(json.dumps(c) + "\n")
    print(f"{len(out)} cases -> {HERE / 'cases_ablate.jsonl'}")


def _z(sc):
    """Scores as log-weights (probabilities are logged)."""
    if all(0 <= x <= 1 for x in sc):
        return [math.log(max(x, 1e-9)) for x in sc]
    return list(sc)


def _pct(k, n):
    if not n:
        return ""
    p, lo, hi = wilson(k, n)
    return f"{p:.0%} ({lo:.0%}-{hi:.0%})"


def report():
    cases = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "cases_ablate.jsonl")}
    L = ["# What does each model use? The need, taken apart", "",
         "Share of picks in the acceptable set, 60 lists per size. `calibrated` = the full scores minus the "
         "blank scores (each option's prior removed), worked out from the kept scores. The failure columns "
         "look only at the lists the model got WRONG with the full need: `prior` = the pick did not change when "
         "the need was removed; `object pull` = the object alone gives the same wrong pick while the verb alone "
         "is right.", ""]
    for size in SIZES:
        L += [f"## {size} options", "",
              "| model | full | verb only | object only | blank | whole action (slot) | calibrated | "
              "failures: prior | failures: object pull |",
              "| --- |" + " --- |" * 8]
        for f in sorted(glob.glob(str(RES / "*.ablate.jsonl"))):
            name = os.path.basename(f)[: -len(".ablate.jsonl")].replace("__", ":")
            rows = {}
            for l in open(f):
                r = json.loads(l)
                c = cases.get(r["id"])
                if c and c["size"] == size and "pick" in r:
                    rows[(c["value"], c["orig"])] = r
            origs = sorted({o for (_, o) in rows})
            cells = []
            for v in VARIANTS:
                rs = [rows[(v, o)] for o in origs if (v, o) in rows]
                cells.append(_pct(sum(r["ok"] for r in rs), len(rs)))
            # calibrated: full minus blank, per option
            k = n = 0
            for o in origs:
                a, b = rows.get(("full", o)), rows.get(("blank", o))
                if a and b and "scores" in a and "scores" in b:
                    za, zb = _z(a["scores"]), _z(b["scores"])
                    cal = [x - y for x, y in zip(za, zb)]
                    pick = max(range(len(cal)), key=cal.__getitem__)
                    k += pick in set(a["good"])
                    n += 1
            cells.append(_pct(k, n) if n else "-")
            # failure causes
            fails = [o for o in origs if ("full", o) in rows and not rows[("full", o)]["ok"]]
            prior = sum(1 for o in fails if ("blank", o) in rows and rows[("blank", o)]["pick"] == rows[("full", o)]["pick"])
            pull = sum(1 for o in fails if ("object", o) in rows and ("verb", o) in rows
                       and rows[("object", o)]["pick"] == rows[("full", o)]["pick"] and rows[("verb", o)]["ok"])
            cells.append(f"{prior / len(fails):.0%} of {len(fails)}" if fails else "")
            cells.append(f"{pull / len(fails):.0%}" if fails else "")
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L.append("")
    (RES / "ablate.md").write_text("\n".join(L) + "\n")
    print(RES / "ablate.md")


if __name__ == "__main__":
    {"build": build, "report": report}[sys.argv[1]]()
