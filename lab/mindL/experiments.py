"""The learned-mind tests (plan, section 8). Built so far: L0 (smoke).

FOCAL design: in each lifetime ONE cast member (the focal one) uses the
chooser under test; the rest of the cast (and the bandit from the road) use
`rules`. So every chooser meets the same kind of company, and PPO (which
learns over one character's decisions) is compared like for like. Each
chooser is run once with each of the five cast members as the focal one.

    L0  keeps itself well: a lifetime in the inn (flags all), the focal one's
        drive error per night (lab/world2/lifetime.py samples it every 10
        turns). Pass (plan): as low as `rules` or lower, and far below random.

    lab/.venv/bin/python -m lab.mindL.experiments l0 mind --nights 100 --seed 1
    lab/.venv/bin/python -m lab.mindL.experiments l0-report --nights 100 --seed 1

DIAGNOSTIC --ban "shout for help,shove": the same run with those verbs taken
off the focal chooser's menu (the world is unchanged). It asks whether a
result stands without the 1-turn acts that interrupt everyone in sight or
earshot (world.py _perceive): repeated every turn, they stop every act of 2
turns or more near the actor, a blow included. Results: L0ban-<verbs>_*.json.

Choosers: mind, mind:<letters> (an ablation, e.g. mind:AK), bandit, ppo,
rules, random. Seeds: the chooser's own seed is the lifetime's seed.

ABLATION (plan 9.2, one module added at a time): mind:AK, mind:AKV,
mind:AKVP, mind:AKVPE, mind:AKVPEW, then the full mind (AKVWPEC, i.e. + C).

THE FIXES (2026-10-29): chooser "mind2" is the mind with mind.FIXES (the partner models
weigh in the choice and follow change); the world flags are "all2" (= "all" plus `reprisal`
and `wear`: the interrupt loophole closed, diminishing returns for praise and insult).
"all" and "mind" are exactly the first round's.

Flags "all" include `known`: every chooser's menu numbers come only from what
its character perceived (lab/world2/world.py). Results made before that flag
are in lab/results/mindL/before_known/.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from lab.mindL import drives as H
from lab.mindL.mind import DEFAULT, FIXES, make_mind
from lab.mindL.nulls import PPO_CFG, make_bandit, run_ppo
from lab.world2.lifetime import NAMES, Cast, lifetime
from lab.world2.policy import Random, Rules
from lab.world2.world import DRIVES, World

OUT = Path(__file__).resolve().parents[1] / "results" / "mindL"
CHOOSERS = ("mind", "bandit", "ppo", "rules", "random")


def make_chooser(name, seed):
    if name == "mind":
        return make_mind(seed)
    if name.startswith("mind:"):
        return make_mind(seed, modules=name.split(":", 1)[1])
    if name == "mind2":                  # the 2026-10-29 fixes (mind.FIXES); meant for flags "all2"
        return make_mind(seed, **FIXES)
    if name == "bandit":
        return make_bandit(seed)
    if name == "rules":
        return Rules()
    if name == "random":
        return Random()
    raise ValueError(name)


def settings(name):
    if name == "mind2":
        return {**DEFAULT, **FIXES}
    if name.startswith("mind"):
        return {**DEFAULT, **({"modules": name.split(":", 1)[1]} if ":" in name else {})}
    if name == "bandit":
        return {"tau": 0.005, "lr": 0.1}
    if name == "ppo":
        return dict(PPO_CFG)
    return {}


class Ban:
    """A DIAGNOSTIC wrapper, not a world change: the focal chooser never sees
    the banned verbs on its menu (if nothing else is left, it sees all).
    Used to ask whether a result stands without one act (an exploit)."""

    def __init__(self, inner, verbs):
        self.inner, self.verbs = inner, frozenset(verbs)
        self.name = f"{inner.name}-no[{','.join(sorted(self.verbs))}]"

    def probs(self, w, p, opts):
        keep = [i for i, o in enumerate(opts) if o["verb"] not in self.verbs] or list(range(len(opts)))
        pc, pa = self.inner.probs(w, p, [opts[i] for i in keep])
        out_c, out_a = [0.0] * len(opts), [0.0] * len(opts)
        for j, i in enumerate(keep):
            out_c[i], out_a[i] = pc[j], pa[j]
        return out_c, out_a

    def __getattr__(self, k):              # the lifetime's hooks reach the chooser inside
        return getattr(self.inner, k)


class SplitCast(Cast):
    """A Cast that also samples the focal one's error per drive, at the first
    choice made on each turn divisible by 10 (an approximation of the
    lifetime's own sampler, used only for the split by drive)."""

    def __init__(self, default, by, focal, split):
        super().__init__(default, by)
        self.focal, self.split, self._t = focal, split, None

    def probs(self, w, p, opts):
        if w.t % 10 == 0 and w.t != self._t:
            self._t = w.t
            q = w.people[self.focal]
            self.split.append(H.errors(q.kind, H.needs(w, q)))
        return super().probs(w, p, opts)


def focal_run(chooser, focal, nights, seed, flags="all", ban=()):
    dead, acts, split, night_split = [], [], [], []

    def on_night(w, n):
        p = w.people[focal]
        dead.append(bool(p.dead))
        a = {}
        for e in w.log:
            if e["actor"] == focal and "options" in e:
                a[e["action"]] = a.get(e["action"], 0) + 1
        acts.append(a)
        night_split.append(np.mean(split, axis=0) if split else np.zeros(len(DRIVES)))
        split.clear()

    t0 = time.perf_counter()
    info = {}
    wrap = (lambda x: Ban(x, ban)) if ban else (lambda x: x)
    if chooser == "ppo":
        stats, info = run_ppo(focal, nights, seed, flags, on_night=on_night,
                              cast=lambda d, by: SplitCast(d, {k: wrap(v) for k, v in by.items()}, focal, split))
    else:
        pol = make_chooser(chooser, seed)
        _, stats = lifetime(SplitCast(Rules(), {focal: wrap(pol)}, focal, split), nights, seed, flags=flags,
                            on_night=on_night)
        a = getattr(pol, "agents", {}).get(focal)
        if a is not None:
            info["decisions"] = a.decisions
    err = [s["drive_error"][focal] for s in stats]
    return {"drive_error": err, "dead": dead, "acts": acts,
            "split": [[round(float(x), 4) for x in s] for s in night_split],
            "seconds": round(time.perf_counter() - t0, 1), **info}


def block(xs, a, b):
    x = np.array(xs[a:b], dtype=float)
    return round(float(x.mean()), 4), round(float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0, 4)


def summarise(r, nights, q):
    e = r["drive_error"]
    acts = {}
    for a in r["acts"][-q:]:
        for k, v in a.items():
            acts[k] = acts.get(k, 0) + v
    split = np.mean(r["split"][-q:], axis=0)
    return {"mean": block(e, 0, nights), f"first{q}": block(e, 0, q), f"last{q}": block(e, nights - q, nights),
            "deaths": int(sum(r["dead"])), f"deaths_last{q}": int(sum(r["dead"][-q:])),
            f"split_last{q}": {d: round(float(x), 4) for d, x in zip(DRIVES, split)},
            f"acts_per_night_last{q}": {k: round(v / q, 1) for k, v in sorted(acts.items(), key=lambda t: -t[1])[:12]},
            "seconds": r["seconds"], **{k: r[k] for k in ("decisions", "truncated_menus") if k in r}}


def l0(chooser, nights, seed, focals=NAMES, flags="all", ban=()):
    q = max(1, nights // 5)
    tag = chooser + (f"-no[{','.join(sorted(ban))}]" if ban else "")
    res = {"test": "L0" + ("-ban" if ban else ""), "chooser": tag, "seed": seed, "nights": nights,
           "flags": sorted(World(flags=flags).flags), "ban": sorted(ban),
           "design": "focal: one cast member uses the chooser, the rest of the cast and the bandit use rules"
                     + ("; DIAGNOSTIC: the banned verbs are taken off the focal chooser's menu" if ban else ""),
           "settings": settings(chooser), "block": q, "focal": {}}
    out = OUT / f"{prefix(ban)}_{chooser.replace(':', '-')}_seed{seed}_n{nights}.json"
    for f in focals:
        r = focal_run(chooser, f, nights, seed, flags, ban)
        res["focal"][f] = {"summary": summarise(r, nights, q), **r}
        s = res["focal"][f]["summary"]
        print(f"{chooser:8s} {f:8s} mean {s['mean'][0]:.3f}  first{q} {s[f'first{q}'][0]:.3f}  "
              f"last{q} {s[f'last{q}'][0]:.3f}  deaths {s['deaths']}  {r['seconds']} s", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res))
    return res


def prefix(ban):
    return "L0" + ("ban-" + "-".join(sorted(v.replace(" ", "") for v in ban)) if ban else "")


def l0_report(nights, seed, choosers=None, ban=()):
    """Merge the per-chooser files into one table (json + markdown text)."""
    q = max(1, nights // 5)
    rows, table = {}, {}
    pre = prefix(ban)
    files = sorted(OUT.glob(f"{pre}_*_seed{seed}_n{nights}.json"))
    for fp in files:
        r = json.loads(fp.read_text())
        if r.get("test") != ("L0-ban" if ban else "L0") or "focal" not in r:
            continue
        c = r["chooser"]
        if choosers and c not in choosers:
            continue
        per = {f: v["summary"] for f, v in r["focal"].items()}
        # the unit is the FOCAL character: nights within one focal's life are
        # correlated, and the spread between focals dominates, so the error
        # bar is the standard error over the focal means (n = 5), not over
        # the pooled focal-nights
        allf = {k: [float(np.mean(v["drive_error"][a:b])) for v in r["focal"].values()]
                for k, (a, b) in (("mean", (0, nights)), ("first", (0, q)), ("last", (nights - q, nights)))}
        rows[c] = {"focals": sorted(per), "mean": block(allf["mean"], 0, len(allf["mean"])),
                   f"first{q}": block(allf["first"], 0, len(allf["first"])),
                   f"last{q}": block(allf["last"], 0, len(allf["last"])),
                   "deaths": sum(v["deaths"] for v in per.values()),
                   f"deaths_last{q}": sum(v[f"deaths_last{q}"] for v in per.values()),
                   "per_focal_last": {f: v[f"last{q}"][0] for f, v in per.items()},
                   "per_focal_mean": {f: v["mean"][0] for f, v in per.items()},
                   "split_last": {d: round(float(np.mean([v[f"split_last{q}"][d] for v in per.values()])), 4)
                                  for d in DRIVES},
                   "seconds": round(sum(v["seconds"] for v in per.values()), 1),
                   "settings": r["settings"]}
        table[c] = per
    rank = {"mind": 0, "bandit": 2, "ppo": 3, "rules": 4, "random": 5}
    order = sorted(rows, key=lambda c: (rank.get(c.split("-no[")[0], 1), len(c.split("-no[")[0]), c))
    lines = [("DIAGNOSTIC (banned verbs off the focal menu). " if ban else "") + f"L0, seed {seed}, {nights} nights, focal design (the other four on rules). Drive error of the focal "
             f"character, sampled every 10 turns; mean over the 5 focal characters' own means, +- the standard error "
             f"over those 5 (the focal is the unit: its nights are not independent).", "",
             f"| chooser | all nights | nights 0-{q - 1} | nights {nights - q}-{nights - 1} | deaths (all / last {q}) "
             f"| " + " | ".join(f"{f} last {q}" for f in NAMES) + " | time |",
             "| --- " * (6 + len(NAMES)) + "|"]
    for c in order:
        r = rows[c]
        lines.append(f"| {c} | {r['mean'][0]:.3f} +- {r['mean'][1]:.3f} | {r[f'first{q}'][0]:.3f} +- {r[f'first{q}'][1]:.3f} "
                     f"| {r[f'last{q}'][0]:.3f} +- {r[f'last{q}'][1]:.3f} | {r['deaths']} / {r[f'deaths_last{q}']} | "
                     + " | ".join(f"{r['per_focal_last'].get(f, float('nan')):.3f}" for f in NAMES)
                     + f" | {r['seconds']} s |")
    lines += ["", f"Error by drive, nights {nights - q}-{nights - 1}, mean over the focals:", "",
              "| chooser | " + " | ".join(DRIVES) + " |", "| --- " * (1 + len(DRIVES)) + "|"]
    for c in order:
        lines.append(f"| {c} | " + " | ".join(f"{rows[c]['split_last'][d]:.3f}" for d in DRIVES) + " |")
    # paired: the same focal character, the same seed, the same company
    base = {b: next((c for c in rows if c.split("-no[")[0] == b), None) for b in ("rules", "ppo", "bandit")}
    lines += ["", f"Paired difference of nights {nights - q}-{nights - 1} (chooser minus null, per focal; mean +- standard "
              f"error over the 5 focals; below 0 = the chooser keeps itself better):", "",
              "| chooser | minus rules | minus PPO | minus bandit |", "| --- | --- | --- | --- |"]
    for c in order:
        cells = []
        for b in ("rules", "ppo", "bandit"):
            o = base[b]
            fs = [f for f in NAMES if o and f in rows[c]["per_focal_last"] and f in rows[o]["per_focal_last"]]
            if c == o or len(fs) < 2:
                cells.append("—")
                continue
            d = [rows[c]["per_focal_last"][f] - rows[o]["per_focal_last"][f] for f in fs]
            m, se = block(d, 0, len(d))
            rows[c].setdefault("paired_last", {})[b] = [m, se]
            cells.append(f"{m:+.3f} +- {se:.3f}")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    out = OUT / f"{pre}_seed{seed}_n{nights}.json"
    out.write_text(json.dumps({"test": "L0" + ("-ban" if ban else ""), "seed": seed, "nights": nights, "rows": rows, "per_focal": table,
                               "markdown": text}, indent=1))
    print(text)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("test", choices=["l0", "l0-report"])
    ap.add_argument("chooser", nargs="?", default="mind")
    ap.add_argument("--nights", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--focal", default=",".join(NAMES))
    ap.add_argument("--ban", default="", help="DIAGNOSTIC: verbs taken off the focal menu, comma list")
    a = ap.parse_args()
    ban = tuple(x for x in a.ban.split(",") if x)
    if a.test == "l0":
        l0(a.chooser, a.nights, a.seed, a.focal.split(","), ban=ban)
    else:
        l0_report(a.nights, a.seed, ban=ban)


if __name__ == "__main__":
    main()
