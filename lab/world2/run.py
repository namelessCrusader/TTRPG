"""Run the inn: many nights (seeds) under one policy, then measure them.

    lab/.venv/bin/python -m lab.world2.run rules --seeds 20 --turns 300
    lab/.venv/bin/python -m lab.world2.run bias:gdecide:all --seeds 10
    lab/.venv/bin/python -m lab.world2.run compare         # one table over every policy run so far

Each night is written to lab/results/world2/<policy>/seed<k>.events.jsonl as
it finishes; a night already written is skipped, so a stopped run continues.
Then lab/judge/signals.py (creativity, layer 3) and the numbers below go to
lab/results/world2/<policy>/report.md.

TYPE BEHAVIOUR (plan, Workstream F). When a character sees another harmed
(a blow, a threat) by someone else, what does it do next? BRAVE = it strikes,
shoves or threatens the attacker, or defends, bandages or helps up the
victim. AWAY = it flees or walks off. The plan's proposed ranges: a peasant
brave in 10-25% of threats, a knight in more than 70%.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from lab.judge.signals import analyse, report as signals_report
from lab.world2.policy import make_policy
from lab.world2.world import World

# LAB_WORLD_OUT: another results folder (world2b = after the take/drop fix of 2026-09-30)
OUT = Path(__file__).parent.parent / "results" / os.environ.get("LAB_WORLD_OUT", "world2")


def night(policy, seed, turns):
    w = World(seed=seed, turns=turns)
    for _ in range(turns):
        w.step(policy)
        if all(p.dead or p.down for p in w.people.values()):
            break
    return w


def responses(events, window=12):
    """For each harm seen by a third character: its next finished choice."""
    out = defaultdict(Counter)
    by = {e["id"]: e for e in events}
    for e in events:
        if not e.get("done", True) or not e.get("options"):
            continue
        for sid in e.get("seen", []):
            h = by.get(sid)
            if not h or h["action"] not in ("strike", "shove", "threaten") or not h.get("done"):
                continue
            if e["actor"] in (h["actor"], h["target"]) or e["t"] - h["t"] > window:
                continue
            a, tgt = e["action"], e.get("target")
            if (a in ("strike", "shove", "threaten") and tgt == h["actor"]) or \
                    (a in ("defend", "bandage", "help up") and tgt == h["target"]) or a == "shout for help":
                kind = "brave"
            elif a in ("flee to", "walk to"):
                kind = "away"
            else:
                kind = "other"
            out[e["kind"]][kind] += 1
            break
    return out


def summary(runs):
    ev = [e for evs in runs.values() for e in evs]
    done = [e for e in ev if e.get("done", True) and e.get("options")]
    acts = Counter(e["action"] for e in done)
    resp = defaultdict(Counter)
    for evs in runs.values():
        for k, c in responses(evs).items():
            resp[k].update(c)
    deaths = sum(1 for e in ev if str(e.get("result", "")).startswith("killed"))
    lies = sum(1 for e in ev if e.get("facts", {}).get("lie") and e.get("done", True))
    found = sum(1 for e in ev if e["action"] == "notice" and "contradicts" in e.get("facts", {}))
    stopped = sum(1 for e in ev if e.get("options") and not e.get("done", True))
    return {"choices": len([e for e in ev if e.get("options")]), "acts": acts, "resp": resp,
            "deaths": deaths / max(1, len(runs)), "lies": lies, "lies_found": found,
            "stopped": stopped / max(1, len([e for e in ev if e.get("options")]))}


def world_section(runs):
    s = summary(runs)
    L = ["", "## The world", "",
         f"- choices: {s['choices']}; stopped before done: {s['stopped']:.0%}; "
         f"deaths per night: {s['deaths']:.2f}; lies told: {s['lies']}, found out: {s['lies_found']}",
         "- most common actions: " + ", ".join(f"{a} {n}" for a, n in s["acts"].most_common(12)), "",
         "## Type behaviour: what a witness does next after seeing a blow or a threat", "",
         "| type | brave | away | other | n |", "| --- | --- | --- | --- | --- |"]
    for k, c in sorted(s["resp"].items()):
        n = sum(c.values())
        L.append(f"| {k} | {c['brave'] / n:.0%} | {c['away'] / n:.0%} | {c['other'] / n:.0%} | {n} |")
    return "\n".join(L) + "\n"


def run(policy_name, seeds, turns, tag=""):
    pol = make_policy(policy_name)
    if hasattr(pol, "load"):
        pol.load()
    d = OUT / (policy_name.replace(":", "__") + (f".{tag}" if tag else ""))
    d.mkdir(parents=True, exist_ok=True)
    for seed in range(seeds):
        f = d / f"seed{seed}.events.jsonl"
        if f.exists():
            continue
        t0 = time.time()
        w = night(pol, seed, turns)
        tmp = f.with_suffix(".tmp")
        tmp.write_text("\n".join(json.dumps(e) for e in w.log) + "\n")
        tmp.rename(f)
        print(f"  seed {seed}: {len(w.log)} events, {time.time() - t0:.0f}s"
              + (f", {pol.calls} model calls, {pol.ms / max(1, pol.calls):.0f} ms each" if hasattr(pol, "calls") else ""),
              flush=True)
    text = signals_report(str(d)) + world_section(_load(d))
    (d / "report.md").write_text(text)
    print(d / "report.md")


def _load(d):
    return {f.name: [json.loads(l) for l in open(f)] for f in sorted(d.glob("*.events.jsonl"))}


def compare():
    """One table over every policy: the layer 3 numbers side by side."""
    L = ["# Scene creativity (layer 3) and type behaviour, by policy", "",
         "`surprising, in character` = choices an average character would make less than 1 time in 20, "
         "that this character's own traits make MORE likely. `odd` = the same, but its traits do not explain "
         "them. Story patterns: matches per night (different people each). `variety` = how different two nights "
         "are in what each type of character did (0 = the same, 1 = nothing shared). `reactions` = choices aimed at "
         "the people of an event the character just saw. Brave = share of a type's reactions to a blow or threat it saw "
         "that confront the attacker or help the victim (plan: peasant 10-25%, knight over 70%).", "",
         "| policy | nights | surprising, in character | odd | revenge | betrayal | rescue | lie found out "
         "| coward brave | gift returned | longest chain | reactions | map covered | variety | "
         "brave: peasant | knight | merchant | thief | cook |",
         "| --- |" + " --- |" * 18]
    for d in sorted(p for p in OUT.iterdir() if p.is_dir() and "." not in p.name):
        runs = _load(d)
        if not runs:
            continue
        a = analyse(runs, top=10 ** 6)
        ch = [e for evs in runs.values() for e in evs if e.get("options") and e.get("p_avg")]
        n = max(1, len(ch))
        inc = sum(1 for u, f, _, _ in a["surprise_in"] if u > 3.0) / n
        off = sum(1 for u, f, _, _ in a["surprise_off"] if u > 3.0) / n
        pf = a["pattern_freq"]
        c = list(a["chains"].values())
        s = summary(runs)
        br = []
        for k in ("peasant", "knight", "merchant", "thief", "cook"):
            r = s["resp"].get(k, Counter())
            m = sum(r.values())
            br.append(f"{r['brave'] / m:.0%} ({m})" if m else "-")
        pr = a["pattern_rate"]
        L.append(f"| {d.name.replace('__', ':')} | {len(runs)} | {inc:.1%} | {off:.1%} | "
                 + " | ".join(f"{pr.get(p, 0):.1f}" for p in ("revenge", "betrayal", "rescue", "lie found out",
                                                             "coward acts bravely", "gift returned"))
                 + f" | {max(x['longest'] for x in c)} | {sum(x['reactive'] for x in c) / len(c):.0%}"
                 + f" | {a['variety']['coverage']:.0%} | {a['variety']['run_distance']:.2f} | " + " | ".join(br) + " |")
    (OUT / "compare.md").write_text("\n".join(L) + "\n")
    print(OUT / "compare.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--turns", type=int, default=300)
    ap.add_argument("--tag", default="", help="suffix for the output folder (e.g. smoke)")
    a = ap.parse_args()
    if a.policy == "compare":
        compare()
    else:
        run(a.policy, a.seeds, a.turns, a.tag)
