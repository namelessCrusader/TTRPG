"""Creativity tests, layer 3: creativity in SCENES, measured on a world's
event log. These are also the signals the judge reads (plan v3, Workstream C).

    python3 -m lab.judge.signals <dir of runs>     # one events.jsonl per run (seed)

Creative = unexpected AND makes sense AND in character. The log gives the
first and the third; the judge (or you, in a rating round) gives the second.

THE EVENT LOG. The lab world (Workstream E) writes one JSON object per event:

    id        unique event id
    t         turn
    actor     who acted                      "mara"
    kind      the actor's type               "peasant"
    traits    the actor's traits now         {"bravery": 0.2, ...}
    action    a catalog verb                 "strike", "give", "tell" (docs/action-catalog.md)
    target    who or what, or null
    target_kind  the target's type, or null
    parts     optional parts                 {"with": "the iron pot", "how": "in anger"}
    options   what the actor could choose from at that moment (strings)
    chosen    index into options
    p_char    the probabilities the actor drew from
    p_avg     the probabilities an AVERAGE character (the same model, traits
              set to the middle) gives the same options
    seen      ids of other characters' events the actor perceived since its last choice
    valid     the world permitted it
    done      false when it was stopped before it happened (optional; true if absent)
    facts     optional: {"said": f, "true": false, "lie": true} when it speaks a lie
              ("true" = what the world says; "lie" = the speaker knows it is false);
              {"learns": f, "contradicts": <event id>} when it finds one out

WHAT IT MEASURES

1. Surprise (after "Select the Unexpected", Kreminski et al. 2022). For each
   choice: how unlikely it was for an average character (-log p_avg), and how
   much this character's own personality explains it (log p_char - log p_avg).
   Unlikely AND in character (the character itself gives it at least a 5%
   chance, twice the average's) = creative in character. Unlikely and NOT in
   character = the draw did something odd: "noise", or a surprise for the
   judge to rate.
2. Story patterns (story sifting, after Felt [Kreminski et al. 2019]): revenge,
   betrayal, rescue, a lie found out, a coward who acts bravely, a gift
   returned. Each match is ranked by how RARE that pattern is across all runs
   and how surprising its events were.
3. Chains: one character's action changes what another does next. The
   longest chain, and the share of choices that were reactions to another.
4. Variety between runs (expressive range, Smith & Whitehead 2010): each run
   gets two numbers (how much harm, how much help); the report shows how much
   of that map the runs cover, how different the runs are, and, per type of
   character, how varied its actions are. A knight's surprises should differ
   from a thief's.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations

HARM = {"strike", "shoot", "throw at", "shove", "grapple", "disarm", "threaten", "insult", "hold",
        "steal", "accuse"}
HELP = {"give", "help up", "bandage", "press on a wound", "carry", "give food", "give drink",
        "praise", "offer", "wake", "warn", "defend"}


# ---------------------------------------------------------------- 1. surprise
def surprise(ev):
    """-log p_avg of the choice, and how much the character's own draw
    explains it. None when the log has no probabilities for this event."""
    try:
        pa = ev["p_avg"][ev["chosen"]]
        pc = ev["p_char"][ev["chosen"]]
    except (KeyError, IndexError, TypeError):
        return None
    u = -math.log(max(pa, 1e-9))
    fit = math.log(max(pc, 1e-9)) - math.log(max(pa, 1e-9))
    return {"unexpected": u, "fit": fit, "p_char": pc}


# IN CHARACTER: the character itself finds it plausible (at least a 5% draw)
# and at least twice as likely as the average character does. A tail draw of
# 1 in 1000 is not in character even when the traits made it 27x likelier.
IN_CHAR_P, IN_CHAR_FIT = 0.05, math.log(2)


def in_character(s):
    return s["p_char"] >= IN_CHAR_P and s["fit"] >= IN_CHAR_FIT


def surprising(events, top=10):
    """The most unexpected choices, split into in character (fit > 0) and not."""
    rows = []
    for ev in events:
        s = surprise(ev)
        if s and ev.get("valid", True):
            rows.append((s["unexpected"], s["fit"], ev, in_character(s)))
    rows.sort(key=lambda r: -r[0])
    inc = [r[:3] for r in rows if r[3]][:top]
    off = [r[:3] for r in rows if not r[3]][:top]
    return inc, off


# ---------------------------------------------------------------- 2. patterns
def _get(ev, path):
    x = ev
    for k in path.split("."):
        if not isinstance(x, dict) or k not in x:
            return None
        x = x[k]
    return x


def _step_ok(ev, step, b):
    """Does event `ev` satisfy one pattern step under bindings `b`? Returns the
    new bindings, or None. A value is a literal, a set of literals, "?var"
    (bind or check), or a function of the value."""
    if ev.get("done") is False:          # stopped before it happened: not part of a story
        return None
    nb = dict(b)
    for key, want in step.items():
        if key in ("within", "where", "as"):
            continue
        got = _get(ev, key)
        if isinstance(want, str) and want.startswith("?"):
            if got is None or nb.setdefault(want, got) != got:
                return None
        elif isinstance(want, (set, frozenset)):
            if got not in want:
                return None
        elif callable(want):
            if got is None or not want(got):
                return None
        elif got != want:
            return None
    if "as" in step:
        nb[step["as"]] = ev["id"]
    if "where" in step and not step["where"](ev, nb):
        return None
    return nb


def match(events, pattern, limit=50):
    """All matches of a pattern: a list of steps in time order. A step may be a
    list of alternatives. `within` = at most this many turns after the
    previous step. `where` = a test on (event, bindings), for example that
    ?x, ?v and ?h are three different characters."""
    steps = pattern["steps"]
    out = {}                     # one match per binding of the characters: the earliest

    def go(i, start, b, used, t_prev):
        if len(out) >= limit:
            return
        if i == len(steps):
            key = tuple(sorted((k, str(v)) for k, v in b.items() if not k.startswith("?e")))
            out.setdefault(key, {"pattern": pattern["name"], "events": used, "bind": b})
            return
        alts = steps[i] if isinstance(steps[i], list) else [steps[i]]
        for j in range(start, len(events)):
            ev = events[j]
            for st in alts:
                if t_prev is not None and "within" in st and ev["t"] - t_prev > st["within"]:
                    continue
                nb = _step_ok(ev, st, b)
                if nb is not None:
                    go(i + 1, j + 1, nb, used + [ev["id"]], ev["t"])
                    break

    go(0, 0, {}, [], None)
    return list(out.values())


def _diff(*names):
    return lambda ev, b: len({b[n] for n in names if n in b}) == len([n for n in names if n in b])


PATTERNS = [
    {"name": "revenge",
     "text": "?x harms ?a; later ?a harms ?x",
     "steps": [{"actor": "?x", "action": HARM, "target": "?a", "where": _diff("?x", "?a")},
               {"actor": "?a", "action": HARM, "target": "?x"}]},
    {"name": "betrayal",
     "text": "?a helps ?b; later ?a harms ?b",
     "steps": [{"actor": "?a", "action": HELP, "target": "?b", "where": _diff("?a", "?b")},
               {"actor": "?a", "action": HARM, "target": "?b"}]},
    {"name": "rescue",
     "text": "?x harms ?v; soon a third, ?h, helps ?v or turns on ?x",
     "steps": [{"actor": "?x", "action": HARM, "target": "?v", "where": _diff("?x", "?v")},
               [{"actor": "?h", "action": HELP, "target": "?v", "within": 20, "where": _diff("?x", "?v", "?h")},
                {"actor": "?h", "action": HARM, "target": "?x", "within": 20, "where": _diff("?x", "?v", "?h")}]]},
    {"name": "lie found out",
     "text": "?a tells ?b something false; later ?b learns it is not so",
     "steps": [{"actor": "?a", "target": "?b", "facts.lie": True, "as": "?e_lie"},
               {"actor": "?b", "where": lambda ev, b: _get(ev, "facts.contradicts") == b["?e_lie"]}]},
    {"name": "coward acts bravely",
     "text": "a character with low bravery attacks a threat",
     "steps": [{"actor": "?a", "action": HARM, "traits.bravery": lambda v: v < 0.35,
                "target_kind": {"bandit", "wolf", "dragon", "knight", "thief"}}]},
    {"name": "gift returned",
     "text": "?a gives ?b something; later ?b helps ?a",
     "steps": [{"actor": "?a", "action": "give", "target": "?b", "where": _diff("?a", "?b")},
               {"actor": "?b", "action": HELP, "target": "?a"}]},
]


# ---------------------------------------------------------------- 3. chains
def _responds(ev, s):
    """ev answers s: made after seeing it, by someone else, and aimed at the
    people in it. Doing anything at all after seeing it is not enough: in a
    shared room everything would be a reaction to everything."""
    return (s["actor"] != ev["actor"] and ev.get("target") is not None
            and ev.get("target") in (s["actor"], s.get("target")))


def chains(events):
    """A choice that answers another character's event it saw is a reaction.
    The longest chain of reactions, and the characters in it."""
    by = {ev["id"]: ev for ev in events}
    depth, path = {}, {}
    for ev in events:                                   # the log is in time order
        best, via = 1, None
        for sid in ev.get("seen", []):
            s = by.get(sid)
            if s and _responds(ev, s) and depth.get(sid, 0) + 1 > best:
                best, via = depth[sid] + 1, sid
        depth[ev["id"]] = best
        path[ev["id"]] = via
    if not events:
        return {"reactive": 0, "longest": 0, "actors": []}
    end = max(depth, key=depth.get)
    who, e = [], end
    while e is not None:
        who.append(by[e]["actor"])
        e = path[e]
    reactive = sum(any(by.get(s) and _responds(ev, by[s]) for s in ev.get("seen", []))
                   for ev in events) / len(events)
    return {"reactive": reactive, "longest": depth[end], "actors": who[::-1]}


# ---------------------------------------------------------------- 4. variety
def run_numbers(events):
    events = [ev for ev in events if ev.get("done", True)]
    n = max(len(events), 1)
    return {"harm": sum(ev["action"] in HARM for ev in events) / n,
            "help": sum(ev["action"] in HELP for ev in events) / n}


def _profile(events):
    """What each type of character did in one run: shares of (type, action)."""
    c = Counter((ev.get("kind"), ev["action"]) for ev in events if ev.get("done", True))
    n = sum(c.values()) or 1
    return {k: v / n for k, v in c.items()}


def _js(p, q):
    """Jensen-Shannon distance between two profiles: 0 the same, 1 nothing shared."""
    keys = set(p) | set(q)
    m = {k: (p.get(k, 0) + q.get(k, 0)) / 2 for k in keys}
    kl = lambda a: sum(a[k] * math.log2(a[k] / m[k]) for k in keys if a.get(k, 0) > 0)
    return math.sqrt(max(0.0, (kl(p) + kl(q)) / 2))


def _entropy(counter):
    n = sum(counter.values())
    return -sum(c / n * math.log2(c / n) for c in counter.values()) if n else 0.0


def variety(runs, bins=5):
    """Expressive range over runs that differ only in the seed."""
    pts = [run_numbers(evs) for evs in runs.values()]
    cells = {(min(int(p["harm"] * bins), bins - 1), min(int(p["help"] * bins), bins - 1)) for p in pts}
    prof = [_profile(evs) for evs in runs.values()]
    dist = [_js(a, b) for a, b in combinations(prof, 2)]
    per_kind = defaultdict(Counter)
    for evs in runs.values():
        for ev in evs:
            per_kind[ev.get("kind")][ev["action"]] += 1
    return {"points": pts, "coverage": len(cells) / bins ** 2, "cells": sorted(cells),
            "run_distance": sum(dist) / len(dist) if dist else 0.0,
            "kind_entropy": {k: _entropy(c) for k, c in per_kind.items()},
            "kind_top": {k: c.most_common(3) for k, c in per_kind.items()}}


# ---------------------------------------------------------------- report
def load_runs(root):
    runs = {}
    for f in sorted(glob.glob(os.path.join(root, "*", "events.jsonl"))) + \
             sorted(glob.glob(os.path.join(root, "*.events.jsonl"))):
        runs[os.path.relpath(f, root)] = [json.loads(l) for l in open(f)]
    return runs


def analyse(runs, top=10):
    """Everything above, over all runs. Pattern matches are ranked by how rare
    the pattern is across runs (share of runs that have it) and how surprising
    their events were."""
    found = {}
    for name, evs in runs.items():
        by = {ev["id"]: ev for ev in evs}
        found[name] = []
        for p in PATTERNS:
            for m in match(evs, p):
                s = [surprise(by[e]) for e in m["events"]]
                m["surprise"] = sum(x["unexpected"] for x in s if x)
                found[name].append(m)
    n_runs = max(len(runs), 1)
    freq = Counter(p for name in found for p in {m["pattern"] for m in found[name]})
    per_run = Counter(m["pattern"] for name in found for m in found[name])
    ranked = []
    for name, ms in found.items():
        for m in ms:
            rarity = -math.log(freq[m["pattern"]] / n_runs) if freq[m["pattern"]] < n_runs else 0.0
            ranked.append((rarity + m["surprise"], rarity, name, m))
    ranked.sort(key=lambda r: -r[0])
    inc, off = [], []
    for name, evs in runs.items():
        a, b = surprising(evs, top)
        inc += [(u, f, name, ev) for u, f, ev in a]
        off += [(u, f, name, ev) for u, f, ev in b]
    inc.sort(key=lambda r: -r[0])
    off.sort(key=lambda r: -r[0])
    return {"patterns": ranked, "pattern_freq": {k: v / n_runs for k, v in freq.items()},
            "pattern_rate": {k: v / n_runs for k, v in per_run.items()},
            "surprise_in": inc[:top], "surprise_off": off[:top],
            "chains": {name: chains(evs) for name, evs in runs.items()},
            "variety": variety(runs)}


def _say(ev):
    if ev.get("text"):
        return f"t{ev['t']} {ev['actor']} ({ev.get('kind')}): {ev['text']}" + \
            (f" ({ev['result']})" if ev.get("result") else "")
    s = f"t{ev['t']} {ev['actor']} ({ev.get('kind')}) {ev['action']}"
    if ev.get("target"):
        s += f" {ev['target']}"
    for k, v in (ev.get("parts") or {}).items():
        s += f", {k} {v}"
    return s


def report(root, out=None):
    runs = load_runs(root)
    if not runs:
        raise SystemExit(f"no event logs under {root}")
    a = analyse(runs)
    L = [f"# Scene creativity: {len(runs)} runs", "",
         "## Unexpected choices that fit the character (for rating: do they make sense?)", ""]
    L += [f"- {u:.1f} unexpected, {f:+.1f} fit: {_say(ev)}  ({name})" for u, f, name, ev in a["surprise_in"]]
    L += ["", "## Unexpected choices the character does not explain (odd draws: sense, or noise?)", ""]
    L += [f"- {u:.1f} unexpected, {f:+.1f} fit: {_say(ev)}  ({name})" for u, f, name, ev in a["surprise_off"]]
    L += ["", "## Story patterns", "", "| pattern | share of runs | matches per run |", "| --- | --- | --- |"]
    L += [f"| {p['name']} | {a['pattern_freq'].get(p['name'], 0):.0%} | {a['pattern_rate'].get(p['name'], 0):.1f} |"
          for p in PATTERNS]
    L += ["", "Rarest and most surprising matches:", ""]
    for score, rarity, name, m in a["patterns"][:10]:
        L.append(f"- {m['pattern']} ({name}): " + ", ".join(f"{k}={v}" for k, v in m["bind"].items()
                                                         if not k.startswith("?e")))
    ch = list(a["chains"].values())
    L += ["", "## Chains", "",
          f"- choices that react to another character: {sum(c['reactive'] for c in ch) / len(ch):.0%}",
          f"- longest chain: {max(c['longest'] for c in ch)} "
          f"({' > '.join(max(ch, key=lambda c: c['longest'])['actors'])})"]
    v = a["variety"]
    L += ["", "## Variety between runs", "",
          f"- map covered (harm x help, 5x5): {v['coverage']:.0%}",
          f"- how different two runs are, by what each type did (0 = the same, 1 = nothing shared): "
          f"{v['run_distance']:.2f}", "",
          "| type | variety of actions (bits) | most common |", "| --- | --- | --- |"]
    for k, e in sorted(v["kind_entropy"].items(), key=lambda x: str(x[0])):
        L.append(f"| {k} | {e:.2f} | {', '.join(f'{x} {n}' for x, n in v['kind_top'][k])} |")
    text = "\n".join(L) + "\n"
    if out:
        open(out, "w").write(text)
    return text


if __name__ == "__main__":
    print(report(sys.argv[1]))
