"""What the models are actually asked: two real cases from every task space.

    python3 -m lab.methods.examples        # writes lab/results/methods/examples.md

Every case is shown IN FULL: every sentence, the question and every option,
with the acceptable options marked. Then what each model literally receives
(the same strings the scorers in score.py build), and what GLiNER2.5-Decide
picked in the runs so far, with its top three when its scores were kept.
Nothing here runs a model: picks come from the result files.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from lab.methods.score import NLI

HERE = Path(__file__).parent
RES = HERE.parent / "results" / "methods"
OUT = RES / "examples.md"

# (title, case file, result file for GLiNER-Decide, case ids, what the space tests)
SPACES = [
    ("Things", "cases_things.jsonl", "gdecide__all.jsonl",
     ["things/size=8/3", "things/size=16/4"], "Which thing can do the job? One need, a list of things."),
    ("Things, long scene", "cases_things.jsonl", "gdecide__all.jsonl",
     ["things/situation=long/2", "things/situation=long/7"],
     "The same, with six unrelated sentences around the need."),
    ("Things, hand-picked near misses", "cases_things.jsonl", "gdecide__all.jsonl",
     ["things/nearmiss=hand/4", "things/nearmiss=hand/9"],
     "The other seven things only LOOK as if they would do the job."),
    ("Creative (the obvious tools removed)", "cases_creative.jsonl", "gdecide__all.creative.jsonl",
     ["creative/unusual/0", "creative/mixed/7"],
     "Layer 1: find the unusual thing that still works. Layer 2: obvious, unusual and useless mixed."),
    ("People", "cases_people.jsonl", "gdecide__all.people.jsonl",
     ["people/size=8/3", "people/similar=same/5"],
     "Actions aimed at the people present. `same`: the right actions aimed at the wrong people."),
    ("Places", "cases_places.jsonl", "gdecide__all.places.jsonl",
     ["places/size=8/5", "places/similar=near/12"], "Where can she go for this?"),
    ("Presentation battery", "cases_present.jsonl", "gdecide__all.present.jsonl",
     ["present/ctx_kv/things/size=16/4", "present/ctx_cap/things/size=16/9"],
     "The same lists, presented another way (here: key-value lines; a capability hint)."),
    ("Ablation", "cases_ablate.jsonl", "gdecide__all.ablate.jsonl",
     ["ablate/object/things/size=16/4", "ablate/slot/things/size=16/9"],
     "Parts of the need taken away (here: the object only; each option as the whole action)."),
]


def _load(path):
    p = HERE / path if not str(path).startswith(str(RES)) else Path(path)
    if not p.exists():
        return {}
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)}


def _top3(r, options):
    if not r or "scores" not in r:
        return ""
    sc = r["scores"]
    p = sc if abs(sum(sc) - 1) < 1e-3 else [math.exp(x - max(sc)) for x in sc]
    s = sum(p)
    top = sorted(range(len(p)), key=lambda i: -p[i])[:3]
    return "; top three: " + ", ".join(f"{options[i]} {p[i] / s:.0%}" for i in top)


def _case_block(c, r):
    good = set(c["good"])
    L = ["**Situation:** " + c["situation"].replace("\n", " / "), "",
         f"**Question:** {c['question']}", "",
         f"**Options ({len(c['options'])}; \\* = acceptable):** " +
         " · ".join(("\\*" if i in good else "") + o for i, o in enumerate(c["options"])), ""]
    if r and "pick" in r:
        L.append(f"**GLiNER-Decide picked:** {c['options'][r['pick']]} "
                 f"({'acceptable' if r['pick'] in good else 'NOT acceptable'}){_top3(r, c['options'])}")
    elif r and "error" in r:
        L.append(f"**GLiNER-Decide:** could not run ({r['error'][:80]})")
    else:
        L.append("**GLiNER-Decide:** not run on this case yet")
    L.append("")
    return L


def _model_inputs(sit, q, options):
    """The strings each scorer builds (score.py), for the first options."""
    labels = [o.replace(" (", ", ").replace("(", "").replace(")", "") for o in options]
    show = options[:3]
    more = f" … ({len(options)} in all)" if len(options) > 3 else ""
    L = ["<details><summary>What each model receives</summary>", "",
         "- **GLiNER2.5-Decide** (one pass over the whole list): text = the situation; "
         f"instruction = `{q}`; labels = `{labels[:3]}`{more}",
         "- **bev-decider, Von, Laya** (one pass over the whole list): state = the situation; question = "
         f"`{{\"type\": \"choice\", \"instructions\": \"{q}\", \"criteria\": {{{', '.join(repr(o) + ': None' for o in show)}{', …' if more else ''}}}}}`",
         f"- **openjev** (one option at a time): premise = the situation; hypothesis = "
         + " / ".join(f"`{NLI.HYP['one'].format(q=q, o=o)}`" for o in show) + more,
         f"- **Qwen3, one at a time** (log-probability of the words after the prompt): prompt = "
         f"`{sit.replace(chr(10), ' / ')}\\nQuestion: {q}\\nAnswer:`; continuations = "
         + " / ".join(f"`␣{o}.`" for o in show) + more
         + f"; the calibrated version subtracts the same continuations after `Question: {q}\\nAnswer:`",
         "", "</details>", ""]
    return L


def _inn(seed=3, turns=60, who="Tom"):
    """One decision in the inn, rebuilt with the rules policy (the logs do not
    keep the description text)."""
    from lab.world2.policy import Rules, persona
    from lab.world2.world import World
    w = World(seed=seed, turns=300)
    pol = Rules()
    for _ in range(turns):
        w.step(pol)
    p = w.people[who]
    opts = [o["text"] for o in w.options(p)]
    return w.describe(p), persona(p), f"What does {who} do next?", opts


def main():
    L = ["# What the models are asked", "",
         "Two real cases from every task space, in full. Picks are GLiNER2.5-Decide's, from the runs so far. "
         "Built by `lab/methods/examples.py`.", ""]
    for title, cf, rf, ids, what in SPACES:
        cases, res = _load(cf), _load(RES / rf)
        L += [f"## {title}", "", what, ""]
        for cid in ids:
            c = cases.get(cid)
            if not c:
                L += [f"(case {cid} not found)", ""]
                continue
            L += [f"### `{cid}`", ""] + _case_block(c, res.get(cid)) + _model_inputs(
                c["situation"], c["question"], c["options"])
    L += ["## The inn (lab world v2)", "",
          "No acceptable set: the inn has no single right action. This is what a model policy reads at one "
          "choice (rebuilt with the rules policy, seed 3, turn 60).", ""]
    for turns, who in ((60, "Tom"), (140, "Sir Ada")):
        desc, pers, q, opts = _inn(turns=turns, who=who)
        L += [f"### {who}, turn {turns}", "", f"**Description:** {desc}", "",
              f"**The personality sentence (text policy only):** {pers}", "", f"**Question:** {q}", "",
              f"**Options ({len(opts)}):** " + " · ".join(opts), ""] + _model_inputs(desc, q, opts)
    OUT.write_text("\n".join(L) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
