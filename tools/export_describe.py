"""THE SRC DESCRIBE OF THE LAB'S LABELED CASES, AS A FILE (wave 10, LAB10).

    python3 tools/export_describe.py                 # lab/results/describe_ab/src_describe.jsonl
    python3 tools/export_describe.py --root DIR      # the src/ of another tree (a snapshot)

The user decided (2026-10-10) that a describe is judged on the lab's labeled
cases: does the model pick an option in the case's acceptable set? The lab
wrote each case's situation in its own words. This tool writes the SAME
facts in the words of src/mind/describe.py (its SPELLING: mark1-mind-v2), so the
lab can score both texts on the same options (lab/methods/describe_ab.py).

THE ISOLATION RULE. lab/ never imports src/, and src/ never imports lab/.
This tool imports src (it is system python, like the rest of tools/) and
only READS the lab's case files as data. The text crosses as a file.

WHICH CASES CAN BE RENDERED. src's describe takes a core.Scene: who decides,
where it is, what it holds, what it needs (said as what is MISSING), the
last percepts and a few beliefs. The lab's decision cases keep the facts of
their situation as fields, and these map onto a Scene:

    who        the named character of the case set (Kael, Mara, Tom); the
               lab text names it too
    goal       `need` of a things or places case ("cut the rope"): a Need
               that a blocked option would meet, "Kael needs something to
               cut the rope" ("a place to" for places)
    missing    `missing` of a state case at a step ("fuel"): the Need says
               it, "Kael needs fuel to start a fire in the hearth"
    state      `state` of a state case ("The hearth is cold and empty."): a
               percept the character saw
    event      `need` of a people or personality case ("A bandit grabs a
               child in the market square."): a percept the character saw
    where      "the storeroom" for the things cases (the lab's own fields
               say so: cases_present ctx_kv "Where: the storeroom")
    holds      nothing, unless a state says "Kael has a fishing rod, but ...";
               v2 says nothing of an empty hand (v1 said "Tom holds nothing.")

What a lab text has and the src describe does not say, by its own rules:
the filler sentences of the "long" situations (no fact behind them), and a
lure ("There is the tongs by the door."): src never names a thing that lies
about. Those cases are rendered from their facts, without those sentences.

NOT RENDERED: cases_die (a fair die: every face is acceptable, so a pick
measures nothing) and cases_primitives (questions about the scene, "Who
holds the bowl of stew?", not decisions; their inn texts keep no fields,
and the groups that do keep one, `scene`, have no acceptable set).

Each output row: id, set (the case file), situation (the src text), and
the spelling. The first row is a header with the describe file's sha256,
so a result says which describe it judged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CASES = REPO / "lab" / "methods"
OUT = REPO / "lab" / "results" / "describe_ab" / "src_describe.jsonl"

# the case sets this tool renders, and the character each one is about
WHO = {"things": "Kael", "lure": "Kael", "ablate": "Kael", "present": "Kael", "pipe": "Kael",
       "creative": "Kael", "state": "Kael", "state2": "Kael", "places": "Mara",
       "people": "Tom", "personality": "Tom", "personality2": "Tom"}
SKIP = {"die": "a fair die: every face is acceptable",
        "primitives": "questions about the scene, not decisions; no fields to render"}
EVENT = ("people", "personality", "personality2")     # `need` is an event the character saw
HAS = re.compile(r"Kael has (?!nothing)(.+?), but")


def scene_of(c, kind, api, core):
    """The core.Scene that holds the facts of lab case `c` of set `kind`."""
    who = WHO[kind]
    place = "the storeroom" if kind not in ("places",) + EVENT else ""
    # "Kael has a fishing rod, but its hook is lost.": the state says what he holds
    has = HAS.match(c.get("state") or "")
    state = api.SelfState(name=who, place=place, holds=(has.group(1),) if has else ())
    seen, needs = [], []
    if kind in EVENT:
        seen.append(c["need"])
    else:
        goal = api.Option(verb="goal", roles={}, text=c["need"], group="goal", duration=1,
                          ok=False, missing=("something",))
        if kind.startswith("state") and c.get("state"):
            seen.append(c["state"])
            says = (c["missing"],)
        else:
            says = ("a place",) if kind == "places" else ("something",)
        needs.append(core.Need("goal", says, goal))
    percepts = tuple(api.Percept(t=0, channel="saw", text=x, source=f"case:{n}")
                     for n, x in enumerate(seen))
    return core.Scene(state, (), tuple(needs), percepts, ())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 tools/export_describe.py",
                                 description="Write the src describe of the lab's labeled cases.")
    ap.add_argument("--root", default=str(REPO),
                    help="the tree whose src/ renders the text (default: this repo)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    sys.path.insert(0, str(root))
    from src.mind import api, core, describe

    src_file = Path(describe.__file__).resolve()
    if root not in src_file.parents:
        raise SystemExit(f"src.mind.describe came from {src_file}, not from {root}")
    rows, counts, refused = [], {}, []
    for kind in WHO:
        n = 0
        for line in (CASES / f"cases_{kind}.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            # a present case may say "you" or nobody: that wording is the lab's experiment
            if WHO[kind] not in c["situation"] + " " + c["question"] and kind != "present":
                raise SystemExit(f"{c['id']}: the case does not name {WHO[kind]}")
            try:
                text = describe.describe(scene_of(c, kind, api, core))
            except ValueError as e:               # a digit: the describe refuses it
                refused.append((c["id"], str(e)))
                continue
            rows.append({"id": c["id"], "set": kind, "situation": text, "spelling": describe.SPELLING})
            n += 1
        counts[kind] = n
    head = {"header": True, "spelling": describe.SPELLING, "root": str(root),
            "describe_sha256": hashlib.sha256(src_file.read_bytes()).hexdigest(),
            "rendered": counts, "refused": len(refused), "not_rendered": SKIP}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in [head] + rows))
    print(json.dumps(head))
    for i, why in refused[:10]:
        print(f"  refused {i}: {why}")
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
