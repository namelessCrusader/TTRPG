"""Why do the models pick wrong? Every wrong pick in Workstream A, examined.

    python3 -m lab.methods.failures          # writes lab/results/methods/failures.md

A wrong pick is compared with what a RANDOM wrong option from the same list
would look like. A lure counts only when the model falls for it more often
than chance: LIFT = (share of wrong picks with the property) / (share of the
list's wrong options with it). Lift 1 = the property does not attract the
model; lift 3 = the model is drawn to it three times more than chance.

    verb word    the pick shares a word with the need's verb ("the cutting board"
                 for "cut the rope")
    object word  the pick shares a word with the need's object ("the salted
                 fish" for "catch fish in the river"): the model matched the
                 thing to be acted on, not the tool
    filler word  the pick shares a word with the unrelated scene text (long
                 situations only)
    neighbor     the pick is the same kind of thing as a good option, or shares
                 a property with one (the rule pools in cases.py)
    first, last  the pick's place in the list
    shortest, longest   the pick's name length in the list
    like object  the pick is among the fifth of the list's wrong options closest
                 IN MEANING to the need's object (a small embedding model,
                 bge-small, on the CPU): "the padlock" for "a locked chest",
                 "the boat" for "a ship". Lists of 16 or more only.
    like verb    the same, closest to the verb ("the axe" for "chop")
    favourite    the pick is one of this model's ten most-picked wrong items
                 (normalised by how often each item is on a list): a prior
                 that ignores the need. The favourites are found on one half
                 of the cases and counted on the other half, so the measure
                 does not find what it put in.

It also finds CASES THAT ALMOST EVERY MODEL FAILS. Those are either truly
hard, or a label is wrong: they are listed for a human to check.
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from lab.methods.cases import FILLER, parse, pools

HERE = Path(__file__).parent
RES = HERE.parent / "results" / "methods"

# each need as (verb, object): the words that say what to DO and what it is done TO
SPLIT = {
    "cut the rope": ("cut", "the rope"), "light the dark cellar": ("light", "the dark cellar"),
    "carry water from the well": ("carry", "water from the well"),
    "dig up the buried box": ("dig up", "the buried box"), "tie up the thief": ("tie up", "the thief"),
    "climb to the high window": ("climb", "the high window"),
    "start a fire in the hearth": ("start a fire", "the hearth"), "write a letter home": ("write", "a letter home"),
    "catch fish in the river": ("catch", "fish in the river"),
    "fix the loose nail in the door": ("fix", "the loose nail in the door"),
    "see who is far down the road": ("see far", "who is down the road"),
    "stay dry in the storm": ("stay dry", "the storm"), "pay the ferryman": ("pay", "the ferryman"),
    "feed the hungry horse": ("feed", "the hungry horse"), "chop firewood": ("chop", "firewood"),
    "treat a bleeding cut": ("treat", "a bleeding cut"),
    "keep warm on a freezing night": ("keep warm", "a freezing night"),
    "open a locked chest whose key is lost": ("open", "a locked chest whose key is lost"),
    "signal a ship far out at sea": ("signal", "a ship far out at sea"),
    "call the village for help": ("call for help", "the village"),
    "cross the flooded stream": ("cross", "the flooded stream"),
    "carry the harvest to the barn": ("carry", "the harvest to the barn"), "cook the stew": ("cook", "the stew"),
    "drink on a long walk": ("drink", "a long walk"), "fight off the wolf": ("fight off", "the wolf"),
    "reach the apple high in the tree": ("reach high", "the apple in the tree"),
    "mend the torn cloak": ("mend", "the torn cloak"),
    "measure the length of the field": ("measure", "the length of the field"),
    "weigh the silver": ("weigh", "the silver"), "clean the muddy floor": ("clean", "the muddy floor"),
    "hide the stolen coins": ("hide", "the stolen coins"), "block the arrows": ("block", "the arrows"),
    "break the ice on the pond": ("break", "the ice on the pond"), "trap a rabbit": ("trap", "a rabbit"),
    "sharpen the blade": ("sharpen", "the blade"), "sleep outside": ("sleep", "outside"),
    "eat something before the journey": ("eat", "something before the journey"),
    "pour the oil into the lamp": ("pour", "the oil into the lamp"),
}
STOP = set("the a an of to in on for with and is from at up off out far who whose before into "
           "something down high his her it its be".split())


def words(text):
    out = set()
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in STOP:
            continue
        for suf in ("ing", "ed", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                w = w[: -len(suf)]
                if suf in ("ing", "ed") and len(w) > 3 and w[-1] == w[-2]:
                    w = w[:-1]                     # cutting -> cut
                break
        out.add(w)
    return out


FILLER_WORDS = set().union(*(words(f) for f in FILLER))


def props(case, i, ctx):
    """The properties of option i in a case (cached: every model sees the same lists)."""
    key = (case["id"], i)
    if key not in ctx["cache"]:
        ctx["cache"][key] = _props(case, i, ctx)
    return ctx["cache"][key]


def _props(case, i, ctx):
    name = case["names"][i]
    verb, obj = SPLIT[case["need"]]
    ow = words(name)
    lens = [len(n) for n in case["names"]]
    n = len(case["names"])
    sim = ctx["sim"][case["need"]]
    bad = [j for j in range(n) if j not in case["good"]]
    top = max(1, round(len(bad) / 5))
    def near(which):
        if n < 16:
            return None
        order = sorted(bad, key=lambda j: -sim[which][case["names"][j]])
        return i in order[:top]
    return {
        "like object": near("object"),
        "like verb": near("verb"),
        "verb word": bool(ow & words(verb)),
        "object word": bool(ow & words(obj)),
        "filler word": bool(ow & FILLER_WORDS) if case["value"] == "long" and case["factor"] == "situation" else None,
        "neighbor": name in ctx["neighbors"][case["need"]],
        "first": i == 0,
        "last": i == n - 1,
        "shortest": len(name) == min(lens),
        "longest": len(name) == max(lens),
    }


HARD = 0.8


def similarities(items):
    """Cosine similarity of every item to each need's verb and object, with
    bge-small on the CPU (the GPU stays free for the queue)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    m = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").eval()

    @torch.no_grad()
    def emb(texts):
        b = tok(texts, padding=True, truncation=True, return_tensors="pt")
        v = m(**b).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(v, dim=-1)

    names = [i["name"] for i in items]
    E = emb(names)
    out = {}
    for need, (verb, obj) in SPLIT.items():
        q = emb([verb, obj])
        s = (q @ E.T).tolist()
        out[need] = {"verb": dict(zip(names, s[0])), "object": dict(zip(names, s[1]))}
    return out


def _favourites(rows, cases, keep):
    present, picked = Counter(), Counter()
    for r in rows:
        if "pick" not in r or r.get("ok") or r["id"] not in cases or not keep(r["id"]):
            continue
        c = cases[r["id"]]
        for i, n in enumerate(c["names"]):
            if i not in c["good"]:
                present[n] += 1
        picked[c["names"][r["pick"]]] += 1
    score = {n: picked[n] / present[n] for n in picked if present[n] >= 5}
    return [n for n, _ in sorted(score.items(), key=lambda x: -x[1])[:10]]


def load_runs():
    runs = {}
    for f in sorted(glob.glob(str(RES / "*.jsonl"))):
        name = os.path.basename(f)[:-6]
        if "." in name:
            continue
        runs[name.replace("__", ":")] = [json.loads(l) for l in open(f)]
    return runs


def main():
    cases = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "cases_things.jsonl")}
    items, needs = parse()
    neighbors = {}
    for nd in needs:
        _, _, pl = pools(items, nd)
        neighbors[nd["need"]] = {i["name"] for i in pl["same"]} | {i["name"] for i in pl["near"]}
    ctx = {"neighbors": neighbors, "sim": similarities(items), "cache": {}}
    runs = load_runs()
    keys = ["like object", "like verb", "verb word", "object word", "filler word", "neighbor", "first", "last", "shortest", "longest",
            "favourite"]

    # each model's favourite wrong items: picked when wrong, per time on a list,
    # found on one half of the cases (by case number) for use on the other
    def half(cid):
        return int(cid.rsplit("/", 1)[1]) % 2

    fav, fav_half = {}, {}
    for name, rows in runs.items():
        for h in (0, 1, None):
            fav_half[(name, h)] = _favourites(rows, cases, lambda cid: h is None or half(cid) == h)
        fav[name] = fav_half[(name, None)]

    out = ["# Why the models pick wrong (Workstream A, things)", "",
           "Each cell: the LIFT of a property among wrong picks (see the top of `lab/methods/failures.py`). "
           "Above 1.5 (bold) = the model is drawn to it. The share of wrong picks with the property is in brackets. "
           "`wrong` = the number of wrong picks examined.", "",
           "| model | wrong | " + " | ".join(keys) + " |", "| --- | --- |" + " --- |" * len(keys)]
    fail_by_case = defaultdict(list)
    examples = defaultdict(lambda: defaultdict(list))
    order = sorted(runs, key=lambda n: -sum(bool(r.get("ok")) for r in runs[n]) / max(1, len(runs[n])))
    for name in order:
        rows = runs[name]
        hit, base, cnt = Counter(), Counter(), Counter()
        nwrong = 0
        for r in rows:
            if "pick" not in r or r["id"] not in cases:
                continue
            c = cases[r["id"]]
            fail_by_case[r["id"]].append((name, r.get("ok"), c["names"][r["pick"]]))
            if r.get("ok"):
                continue
            nwrong += 1
            bad = [i for i in range(len(c["names"])) if i not in c["good"]]
            pp = props(c, r["pick"], ctx)
            other = fav_half[(name, 1 - half(r["id"]))]          # favourites from the other half
            pp["favourite"] = c["names"][r["pick"]] in other
            bp = [props(c, i, ctx) for i in bad]
            for k in keys:
                if pp.get(k) is None:
                    continue
                cnt[k] += 1
                hit[k] += bool(pp[k])
                # chance: a random wrong option from the same list
                if k == "favourite":
                    base[k] += sum(c["names"][i] in other for i in bad) / len(bad)
                else:
                    base[k] += sum(bool(b[k]) for b in bp) / len(bad)
                if pp[k] and len(examples[k][name]) < 3:
                    examples[k][name].append(f"{c['need']} -> {c['names'][r['pick']]}")
        cells = []
        for k in keys:
            if not cnt[k]:
                cells.append("")
                continue
            rate, chance = hit[k] / cnt[k], base[k] / cnt[k]
            if chance == 0:
                cells.append("-")
                continue
            lift = rate / chance
            txt = f"{lift:.1f} ({rate:.0%})"
            cells.append(f"**{txt}**" if lift >= 1.5 and rate >= 0.1 else txt)
        out.append(f"| {name} | {nwrong} | " + " | ".join(cells) + " |")

    out += ["", "## Examples of each lure", ""]
    for k in keys:
        ex = [e for n in order for e in examples[k][n]][:6]
        if ex:
            out.append(f"- **{k}**: " + "; ".join(dict.fromkeys(ex)))
    out += ["", "## Each model's favourite wrong items (picked when wrong, per time on a list)", ""]
    for name in order[:14]:
        out.append(f"- {name}: {', '.join(fav[name][:6])}")

    # cases most models fail: hard, or a label problem
    out += ["", f"## Cases that almost every model fails ({HARD:.0%} or more; check the labels)", "",
            "| case | need | good options | models wrong | most common wrong pick |", "| --- | --- | --- | --- | --- |"]
    hard = []
    for cid, res in fail_by_case.items():
        if len(res) < 20:
            continue
        wrong = [p for n, ok, p in res if not ok]
        if len(wrong) / len(res) >= HARD:
            c = cases[cid]
            hard.append((len(wrong) / len(res), cid, c, Counter(wrong).most_common(1)[0]))
    hard.sort(key=lambda x: -x[0])
    for share, cid, c, (pick, k) in hard[:40]:
        out.append(f"| {cid} | {c['need']} | {', '.join(c['names'][i] for i in c['good'])} | "
                   f"{share:.0%} of {len(fail_by_case[cid])} | {pick} ({k}) |")
    shared = Counter(Counter(p for n, ok, p in res if not ok).most_common(1)[0][0]
                     for cid, res in fail_by_case.items() if len(res) >= 20 and
                     sum(not ok for n, ok, p in res) / len(res) >= HARD)
    out += ["", f"{len(hard)} cases are failed by {HARD:.0%} or more of the models. "
            f"Wrong picks that recur across them: {', '.join(f'{n} ({k})' for n, k in shared.most_common(10))}."]
    out += people_section()
    (RES / "failures.md").write_text("\n".join(out) + "\n")
    print(RES / "failures.md")


def people_section():
    """The "people" kind: is a wrong pick the right action at the wrong
    person, the right person with the wrong action, or neither?"""
    files = sorted(glob.glob(str(RES / "*.people.jsonl")))
    if not files:
        return []
    from lab.methods.cases_kinds import parse_people, people_unit
    templates, untargeted, sits = parse_people()
    info = {}                                   # (situation, option text) -> (purpose, role)
    good = {}
    for sit in sits:
        u = people_unit(templates, untargeted, sit)
        good[sit["key"]] = sit["good"]
        for o in u["all"]:
            info[(sit["key"], o["short"])] = (o["purpose"], o["role"])
    cases = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "cases_people.jsonl")}

    def kind_of(need, name):
        pur, role = info[(need, name)]
        gp = {p for p, _ in good[need]}
        gr = {r for _, r in good[need]} - {"-"}
        if role != "-" and pur in gp:
            return "wrong person"
        if role in gr:
            return "wrong action"
        return "neither"

    kinds = ["wrong person", "wrong action", "neither"]
    out = ["", "## People: whom the action is aimed at", "",
           "Of the wrong picks: `wrong person` = an action that is right in this scene, aimed at the wrong "
           "person (\"pull the bandit to safety\"); `wrong action` = the right person, the wrong action "
           "(\"praise the bandit\"); `neither`. Lift against a random wrong option from the same list.", "",
           "| model | right | wrong picks | " + " | ".join(kinds) + " |", "| --- | --- | --- |" + " --- |" * len(kinds)]
    for f in files:
        name = os.path.basename(f)[: -len(".people.jsonl")].replace("__", ":")
        rows = [json.loads(l) for l in open(f)]
        rows = [r for r in rows if "pick" in r and r["id"] in cases]
        hit, base = Counter(), Counter()
        wrong = [r for r in rows if not r["ok"]]
        for r in wrong:
            c = cases[r["id"]]
            hit[kind_of(c["need"], c["names"][r["pick"]])] += 1
            bad = [i for i in range(len(c["names"])) if i not in c["good"]]
            for i in bad:
                base[kind_of(c["need"], c["names"][i])] += 1 / len(bad)
        n = max(1, len(wrong))
        cells = [f"{hit[k] / base[k]:.1f} ({hit[k] / n:.0%})" if base[k] else "-" for k in kinds]
        out.append(f"| {name} | {sum(r['ok'] for r in rows) / max(1, len(rows)):.0%} | {len(wrong)} | "
                   + " | ".join(cells) + " |")
    return out


if __name__ == "__main__":
    main()
