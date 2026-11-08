"""Scene state and the next step (plan v4, section 4).

    python3 -m lab.methods.cases_state          # writes lab/methods/cases_state.jsonl
    python3 -m lab.methods.cases_state report   # lab/results/methods/state.md (after the queue)

"Start a fire in the hearth": the firewood was marked wrong, but you do need
firewood, if the hearth is empty. Many needs take steps (fuel, then a flame;
a pot, then a fire), so the right thing depends on the state of the scene,
which the old cases never said. Here each multi-step need has two states,
written out, and the acceptable things follow the state:

    none     "Kael needs to start a fire in the hearth."            either step is acceptable
    step A   "... The hearth is cold and empty."                    fuel
    step B   "... Dry wood is laid in the hearth, but it is not lit." a flame

Each list holds 1-2 things for the current step, 2 things for the OTHER step
(the key distractors: right later, wrong now) and 5 useless things. The same
lists are asked with no state, so the shift the state causes is measured.
Fields: step_good (the current step), step_other (the other step).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from lab.methods.cases import label, parse

HERE = Path(__file__).parent

# need: [(what is missing, the state sentence, the things for this step)]
STEPS = {
    "start a fire in the hearth": [
        ("fuel", "The hearth is cold and empty.", ["the firewood", "the charcoal", "the log"]),
        ("a flame", "Dry wood is laid in the hearth, but it is not lit.",
         ["the flint and steel", "the tinderbox", "the candle", "the torch"])],
    "cook the stew": [
        ("a pot", "A fire is burning in the hearth, but there is nothing to cook the stew in.",
         ["the iron pot", "the cauldron", "the kettle"]),
        ("a fire", "The stew is ready in the pot, but the hearth is cold and holds no wood.",
         ["the firewood", "the charcoal", "the log"])],
    "write a letter home": [
        ("something to write on", "Kael has a quill and ink, but nothing to write on.",
         ["the parchment", "the wax tablet"]),
        ("something to write with", "Kael has a sheet of parchment, but nothing to write with.",
         ["the quill and ink", "the charcoal", "the chalk"])],
    "light the dark cellar": [
        ("a flame to light the lamp", "Kael has an oil lamp full of oil, but it is not lit.",
         ["the flint and steel", "the tinderbox", "the candle", "the torch"]),
        ("something that gives light", "Kael has nothing that gives light.",
         ["the lantern", "the candle", "the torch", "the oil lamp"])],
    "catch fish in the river": [
        ("a hook", "Kael has a fishing rod, but its hook is lost.", ["the fish hook"]),
        ("something to fish with", "Kael has nothing for fishing.", ["the net", "the fishing rod", "the spear"])],
    "mend the torn cloak": [
        ("a patch", "The hole in the cloak is too big to sew shut; it needs a piece of cloth or leather over it.",
         ["the leather patch", "the bolt of cloth"]),
        ("something to sew with", "Kael has a patch, but nothing to sew it on with.", ["the needle and thread"])],
}
N_LISTS, SIZE = 30, 8

# B2: the same states written as ONLY what is missing (nothing present or
# done is named), as the engine could write them from the preconditions
MISSING_ONLY = {
    ("start a fire in the hearth", 0): "There is no fuel in the hearth.",
    ("start a fire in the hearth", 1): "The fire in the hearth still needs a flame to light it.",
    ("cook the stew", 0): "There is nothing to cook the stew in.",
    ("cook the stew", 1): "There is no fire to cook on.",
    ("write a letter home", 0): "Kael has nothing to write on.",
    ("write a letter home", 1): "Kael has nothing to write with.",
    ("light the dark cellar", 0): "Kael still needs a flame to light his lamp.",
    ("light the dark cellar", 1): "Kael has nothing that gives light.",
    ("catch fish in the river", 0): "Kael has no fish hook.",
    ("catch fish in the river", 1): "Kael has nothing to fish with.",
    ("mend the torn cloak", 0): "The hole is too big to sew shut without a piece of cloth or leather.",
    ("mend the torn cloak", 1): "Kael has nothing to sew with.",
}


def build(seed=0):
    items, needs = parse()
    by = {i["name"]: i for i in items}
    nd = {n["need"]: n for n in needs}
    for need, steps in STEPS.items():
        for _, _, things in steps:
            for t in things:
                assert t in by, f"unknown thing {t!r}"
    out = []
    for need, steps in STEPS.items():
        in_steps = {t for _, _, things in steps for t in things}
        bad = [i["name"] for i in items if label(i, nd[need]) == "bad" and i["name"] not in in_steps]
        rng = random.Random(f"{seed}-{need}")
        for si, (missing, state, things) in enumerate(steps):
            other = [t for t in steps[1 - si][2] if t not in things]
            for k in range(N_LISTS):
                g = rng.sample(things, min(len(things), rng.choice([1, 2])))
                o = rng.sample(other, min(2, len(other)))
                opts = g + o + rng.sample(bad, SIZE - len(g) - len(o))
                rng.shuffle(opts)
                base = {"kind": "state", "factor": "state", "need": need, "question": "Which thing can Kael use for this?",
                        "options": opts, "names": opts, "missing": missing,
                        "other_missing": steps[1 - si][0],
                        "step_good": [i for i, n in enumerate(opts) if n in g],
                        "step_other": [i for i, n in enumerate(opts) if n in o]}
                tail = " He looks at what is in the storeroom."
                out.append(dict(base, id=f"state/{need}/step{si}/{k}", value=f"step{si}",
                                situation=f"Kael needs to {need}. {state}{tail}", good=base["step_good"],
                                state=state))
                out.append(dict(base, id=f"state/{need}/none{si}/{k}", value="none",
                                situation=f"Kael needs to {need}.{tail}",
                                good=base["step_good"] + base["step_other"], state=""))
    with open(HERE / "cases_state.jsonl", "w") as f:
        for c in out:
            f.write(json.dumps(c) + "\n")
    print(f"{len(out)} cases ({len(STEPS)} needs, 2 states, {N_LISTS} lists each, asked with and without "
          f"the state) -> {HERE / 'cases_state.jsonl'}")


def build_wordings():
    """B2: each state case (not the no-state ones) in three wordings:
    as_now         the state as before (what is there and what is missing)
    missing_only   only what is missing; nothing present or done is named
    missing_first  "He still needs X." before the state as before"""
    base = [json.loads(l) for l in open(HERE / "cases_state.jsonl")]
    out = []
    for c in base:
        if c["value"] == "none":
            continue
        si = int(c["value"][-1])
        tail = " He looks at what is in the storeroom."
        head = f"Kael needs to {c['need']}."
        for w, text in (("as_now", c["situation"]),
                        ("missing_only", f"{head} {MISSING_ONLY[(c['need'], si)]}{tail}"),
                        ("missing_first", f"{head} He still needs {c['missing']}. {c['state']}{tail}")):
            out.append(dict(c, id=c["id"].replace("state/", f"state2/{w}/"), factor="wording", value=w,
                            situation=text, step=c["value"]))
    with open(HERE / "cases_state2.jsonl", "w") as f:
        for c in out:
            f.write(json.dumps(c) + "\n")
    print(f"{len(out)} cases -> {HERE / 'cases_state2.jsonl'}")


def build_lure(seed=0):
    """B2: the general lure. Plain things lists (8 and 16 options), and the same
    list with one sentence that names a USELESS thing from the list. How often
    is that thing picked, with and without the sentence?"""
    rng = random.Random(f"{seed}-lure")
    base = [json.loads(l) for l in open(HERE / "cases_things.jsonl")]
    base = [c for c in base if c["factor"] == "size" and c["value"] in (8, 16)]
    out = []
    for c in base:
        bad = [i for i in range(len(c["options"])) if i not in c["good"]]
        out.append(dict(c, id=f"lure/{c['id']}/plain", factor="lure", value="plain", lure=None))
        for k, i in enumerate(rng.sample(bad, min(2, len(bad)))):
            text = c["situation"].replace(" He looks at", f" There is {c['names'][i]} by the door. He looks at")
            out.append(dict(c, id=f"lure/{c['id']}/named{k}", factor="lure", value="named", lure=i,
                            situation=text))
    with open(HERE / "cases_lure.jsonl", "w") as f:
        for c in out:
            f.write(json.dumps(c) + "\n")
    print(f"{len(out)} cases -> {HERE / 'cases_lure.jsonl'}")


def report2(scorer="gdecide:all"):
    """B2 report: the three wordings, and the lure."""
    from lab.methods.report import wilson
    res = HERE.parent / "results" / "methods"
    L = ["# B2. The mention lure", ""]

    def ci(xs):
        if not xs:
            return "-"
        p, lo, hi = wilson(sum(xs), len(xs))
        return f"{p:.0%} ({lo:.0%}-{hi:.0%})"
    f = res / f"{scorer.replace(':', '__')}.state2.jsonl"
    if f.exists():
        cases = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "cases_state2.jsonl")}
        rows = [(cases[r["id"]], r) for r in map(json.loads, open(f)) if r["id"] in cases and "pick" in r]
        L += ["## The state in three wordings", "",
              "`right step` = the pick is for the step that is missing; `other step` = for the step that is "
              "already done or comes later. 30 lists for each state.", "",
              "| wording | step A: right | step A: other | step B: right | step B: other |",
              "| --- | --- | --- | --- | --- |"]
        for w in ("as_now", "missing_only", "missing_first"):
            cell = []
            for st in ("step0", "step1"):
                rs = [(c, r) for c, r in rows if c["value"] == w and c["step"] == st]
                cell += [ci([r["pick"] in c["step_good"] for c, r in rs]),
                         ci([r["pick"] in c["step_other"] for c, r in rs])]
            L.append(f"| {w} | " + " | ".join(cell) + " |")
        L += ["", "By need (right step, step A / step B):", "",
              "| need | as_now | missing_only | missing_first |", "| --- | --- | --- | --- |"]
        for need in STEPS:
            cell = []
            for w in ("as_now", "missing_only", "missing_first"):
                parts = []
                for st in ("step0", "step1"):
                    rs = [(c, r) for c, r in rows if c["value"] == w and c["step"] == st and c["need"] == need]
                    parts.append(f"{sum(r['pick'] in c['step_good'] for c, r in rs) / max(1, len(rs)):.0%}")
                cell.append(" / ".join(parts))
            L.append(f"| {need} | " + " | ".join(cell) + " |")
        L.append("")
    f = res / f"{scorer.replace(':', '__')}.lure.jsonl"
    if f.exists():
        cases = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "cases_lure.jsonl")}
        rows = {r["id"]: r for r in map(json.loads, open(f)) if r["id"] in cases and "pick" in r}
        named, plain_same, right_plain, right_named = [], [], [], []
        for cid, c in cases.items():
            if c["value"] != "named" or cid not in rows:
                continue
            pid = cid.rsplit("/", 1)[0] + "/plain"
            if pid not in rows:
                continue
            named.append(rows[cid]["pick"] == c["lure"])
            plain_same.append(rows[pid]["pick"] == c["lure"])
            right_named.append(rows[cid]["ok"])
            right_plain.append(rows[pid]["ok"])
        L += ["## The general lure", "",
              "One sentence names a useless thing from the list (\"There is the bucket by the door.\").", "",
              "| | the named useless thing is picked | the pick is acceptable |", "| --- | --- | --- |",
              f"| plain list | {ci(plain_same)} | {ci(right_plain)} |",
              f"| with the sentence | {ci(named)} | {ci(right_named)} |", ""]
    (res / "lure.md").write_text("\n".join(L) + "\n")
    print(res / "lure.md")


def report():
    """With the state written out: how often the pick is the current step, and
    how often the OTHER step (right later, wrong now). Without it: acceptable."""
    import glob
    import os
    from collections import defaultdict
    cases = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "cases_state.jsonl")}
    res = HERE.parent / "results" / "methods"
    L = ["# Scene state and the next step", "",
         "`right step` = the pick is for the step the state says is missing; `other step` = it is for the other "
         "step (right later, wrong now). Without the state either step is acceptable. 30 lists per state.", ""]
    for f in sorted(glob.glob(str(res / "*.state.jsonl"))):
        name = os.path.basename(f)[: -len(".state.jsonl")].replace("__", ":")
        rows = [json.loads(l) for l in open(f)]
        by = defaultdict(lambda: defaultdict(list))
        for r in rows:
            c = cases.get(r["id"])
            if not c or "pick" not in r:
                continue
            key = "no state" if c["value"] == "none" else c["value"]
            by[c["need"]][key].append((r["pick"] in c["step_good"], r["pick"] in c["step_other"], r["ok"]))
        L += [f"## {name}", "", "| need | step A: right / other | step B: right / other | no state: acceptable |",
              "| --- | --- | --- | --- |"]
        for need, d in by.items():
            m = lambda xs, j: f"{sum(x[j] for x in xs) / len(xs):.0%}" if xs else ""
            L.append(f"| {need} | {m(d['step0'], 0)} / {m(d['step0'], 1)} | {m(d['step1'], 0)} / {m(d['step1'], 1)} | "
                     f"{m(d['no state'], 2)} |")
        L.append("")
    (res / "state.md").write_text("\n".join(L) + "\n")
    print(res / "state.md")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if sys.argv[1:] else "build"
    {"build": build, "report": report, "wordings": build_wordings, "lure": build_lure,
     "report2": report2}[cmd]()
