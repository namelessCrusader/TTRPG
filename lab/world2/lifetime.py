"""A lifetime: the same cast, night after night (learned-mind plan, section 6.1).

Each night is a fresh World: the room, the food and the things reset, hunger
and wounds too. What carries (World.carry_out): each cast member's traits
(drawn on night 0, then kept), its liking of the others, its memories (deaths,
thefts suffered, lies found out), what it heard of others (reputation), the
deeds it saw and to whom it told them. A character who died comes back the
next night as itself, with the memory of dying ({"what": "died", "by": ...}).
The bandit is a stranger each night: nothing of it carries.

The policy's hooks, if it has them, are called:
    night_start(world, night)        after the night is made, before turn 1
    night_end(world, night)          after the last turn
    on_death(world, person, event)   when a person dies (event = the blow)
so a learning mind keeps its own state across nights in the policy object.

    lab/.venv/bin/python -m lab.world2.lifetime rules --nights 1000 --seed 0
    lab/.venv/bin/python -m lab.world2.lifetime rules --nights 50 --partner Fenn=P-kind:0.9
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lab.world2.partners import make_partner
from lab.world2.policy import make_policy
from lab.world2.world import CAST, DRIVES, FLAGS, World

NAMES = [n for n, _, _, _ in CAST]
OUT = Path(__file__).resolve().parents[1] / "results" / "mindL"


class Cast:
    """One policy per character: `by` names the ones that differ."""

    def __init__(self, default, by=None):
        self.default, self.by = default, dict(by or {})
        self.name = default.name + "".join(f"+{k}={v.name}" for k, v in sorted(self.by.items()))

    def probs(self, w, p, opts):
        return self.by.get(p.name, self.default).probs(w, p, opts)

    def _all(self):
        seen = []
        for x in [self.default, *self.by.values()]:
            if all(x is not y for y in seen):
                seen.append(x)
        return seen

    def night_start(self, w, night):
        for x in self._all():
            hook(x, "night_start", w, night)

    def night_end(self, w, night):
        for x in self._all():
            hook(x, "night_end", w, night)

    def on_death(self, w, person, event):
        for x in self._all():
            hook(x, "on_death", w, person, event)


def hook(policy, name, *args):
    f = getattr(policy, name, None)
    if f is not None:
        f(*args)


def night_seed(seed, night):
    return seed * 1_000_003 + night


def night_stats(w, err):
    """One night in numbers (for the lifetime's report)."""
    ev = [e for e in w.log if e["done"]]
    lies = [e for e in ev if e.get("facts", {}).get("lie")]
    lie_ids = {e["id"] for e in lies}
    found = {e["facts"]["contradicts"] for e in w.log
             if e["action"] in ("notice", "doubt") and e.get("facts", {}).get("contradicts") in lie_ids}
    caught = sum(1 for e in w.log if e["action"] == "notice" and "contradicts" in e.get("facts", {}))
    # a lie believed: found out, or still the hearer's belief at the end
    believed = {e["id"] for e in lies if e["id"] in found or
                w.people[e["target"]].bel.get(("took", e["facts"]["thing"]), (None, None))[1] == e["id"]}
    acts = {}
    for e in ev:
        acts[e["action"]] = acts.get(e["action"], 0) + 1
    return {"turns": w.t, "events": len(w.log), "deaths": sum(1 for p in w.people.values() if p.dead),
            "lies": len(lies), "lies_found": len(found), "lies_believed": len(believed), "notices": caught, "acts": acts,
            "drive_error": {n: round(sum(x) / max(1, len(x)), 4) for n, x in err.items()},
            "bandit": w.bandit_at is not None,
            "liking_mean": round(sum(v for n in NAMES for k, v in w.people[n].aff.items() if k in NAMES)
                                 / (len(NAMES) * (len(NAMES) - 1)), 4)}


def lifetime(policy, nights=1000, seed=0, turns=300, flags="all", bandit=None, on_night=None, fade=1.0):
    """Run the cast through `nights` nights. Returns (carry, stats per night).
    `fade` < 1: overnight, each liking moves (1 - fade) of the way back to its
    night-0 value (forgetting; 1.0 = none, the default)."""
    carry, stats, start = None, [], None
    for n in range(nights):
        w = World(seed=night_seed(seed, n), bandit=bandit, turns=turns, flags=flags, carry=carry, night=n)
        if start is None:               # the likings as drawn on night 0 (what `fade` returns to)
            start = {a: {b: v for b, v in w.people[a].aff.items() if b in NAMES} for a in NAMES}
        hook(policy, "night_start", w, n)
        dead, err = set(), {name: [] for name in NAMES}
        drives = "drives" in w.flags
        for _ in range(turns):
            w.step(policy)
            for p in w.people.values():
                if p.dead and p.name not in dead:
                    dead.add(p.name)
                    blow = next((e for e in reversed(w.log) if e.get("result") == f"killed {p.name}"), None)
                    if p.name in err:
                        p.memory.append({"night": n, "t": w.t, "what": "died",
                                         "by": blow["actor"] if blow else None, "event": blow and blow["id"]})
                    hook(policy, "on_death", w, p, blow)
            if drives and w.t % 10 == 0:
                for name in NAMES:
                    err[name].append(w.drive_error(w.people[name]))
            if all(p.dead or p.down for p in w.people.values()):
                break
        hook(policy, "night_end", w, n)
        stats.append(night_stats(w, err))
        if on_night:                    # before the carry: a test may set what goes into tomorrow
            on_night(w, n)
        carry = w.carry_out(NAMES)
        if fade < 1.0:
            for n, c in carry.items():
                c["aff"] = {k: start[n].get(k, 0) + fade * (v - start[n].get(k, 0)) for k, v in c["aff"].items()}
    return carry, stats


def summarise(stats, block=100):
    """The nights in blocks: mean deaths, lies, lies found, drive error, acts."""
    out = []
    for i in range(0, len(stats), block):
        s = stats[i:i + block]
        k = len(s)
        acts = {}
        for x in s:
            for a, c in x["acts"].items():
                acts[a] = acts.get(a, 0) + c
        lies = sum(x["lies"] for x in s)
        believed = sum(x["lies_believed"] for x in s)
        out.append({"nights": f"{i}-{i + k - 1}", "deaths": round(sum(x["deaths"] for x in s) / k, 3),
                    "lies": round(lies / k, 3), "lies_found_share": round(sum(x["lies_found"] for x in s) / max(1, lies), 3),
                    "lies_believed_share": round(believed / max(1, lies), 3),
                    "believed_lies_found_share": round(sum(x["lies_found"] for x in s) / max(1, believed), 3),
                    "liking_mean": round(sum(x["liking_mean"] for x in s) / k, 3),
                    "drive_error": {n: round(sum(x["drive_error"][n] for x in s) / k, 4) for n in NAMES},
                    "acts_per_night": {a: round(c / k, 2) for a, c in sorted(acts.items(), key=lambda t: -t[1])}})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("policy", help="rules, random")
    ap.add_argument("--nights", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--turns", type=int, default=300)
    ap.add_argument("--flags", default="all", help="'all' (the first six), 'all2' (every flag), 'none', or a comma "
                    "list of " + ",".join(FLAGS))
    ap.add_argument("--partner", action="append", default=[], help="NAME=SPEC, e.g. Fenn=P-kind:0.9")
    ap.add_argument("--fade", type=float, default=1.0, help="overnight liking fade (1.0 = none)")
    ap.add_argument("--out", default=None, help="json file (default lab/results/mindL/lifetime_<...>.json)")
    a = ap.parse_args()
    flags = a.flags if a.flags in ("all", "all2") else tuple(x for x in a.flags.split(",") if x and x != "none")
    by = {}
    for i, spec in enumerate(a.partner):
        name, _, ps = spec.partition("=")
        by[name] = make_partner(ps, seed=a.seed * 100 + i)
    policy = Cast(make_policy(a.policy), by)
    t0 = time.perf_counter()
    carry, stats = lifetime(policy, a.nights, a.seed, a.turns, flags, fade=a.fade)
    dt = time.perf_counter() - t0
    res = {"policy": policy.name, "partners": {k: v.rule for k, v in by.items()}, "seed": a.seed, "fade": a.fade,
           "nights": a.nights, "turns": a.turns, "flags": sorted(World(flags=flags).flags),
           "seconds": round(dt, 2), "seconds_per_night": round(dt / a.nights, 4),
           "blocks": summarise(stats), "memory_sizes": {n: len(c["memory"]) for n, c in carry.items()},
           "liking_at_end": {n: {k: round(v, 3) for k, v in c["aff"].items()} for n, c in carry.items()},
           "drives": list(DRIVES)}
    out = Path(a.out) if a.out else OUT / f"lifetime_{a.policy}_seed{a.seed}_n{a.nights}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"{a.nights} nights in {dt:.1f} s ({dt / a.nights:.3f} s a night) -> {out}")


if __name__ == "__main__":
    main()
