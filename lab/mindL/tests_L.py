"""The learned-mind tests L0-L5 (plan, section 8), with the controls.

Choosers: mind (the full AKVWPEC mind), bandit (N1), ppo (N2), rules, random.
Seeds: 1-10 (seed 0 was the tuning seed). Every lifetime: flags "all" (with
`known`), the full menu, 300 turns a night, and the overnight liking fade
FADE (env MINDL_FADE, default 0.9: each liking moves 10% of the way back to
its night-0 draw overnight; why: lab/results/mindL/report.md, "The fade").
Settings of the learners: as tuned (lab/mindL/mind.py DEFAULT, nulls.py), not
retuned. A focal character uses the chooser; the scripted
partners named in the run use their written rule; everyone else, and the
bandit from the road, uses rules.

Three run families share the lifetimes between the tests:

  L04  plain inn, N04 nights. Focal = the cast member NAMES[seed % 5].
       reward "drives" (homeostatic, plan section 5) for every chooser: L0.
       reward "designer" for mind, bandit, ppo: L4. The designer reward is
       the drop of the error -0.1 * points, where an act the character
       finishes scores +1 if kind (KIND_ACTS), -1 if harsh or a theft,
       +0.5 for eat, drink, sit down, sleep, look around: the right acts, but
       paid the same whatever the character needs.
  L12  partners Fenn and Oren, N12 nights; focal = ("Tom", "Sir Ada",
       "Marta")[seed % 3]. Life A: Fenn P-kind(0.9), Oren P-hard(0.9) until
       2/3 of the life, then both P-deaf (the shared world, and the
       contingency broken mid-run). Life B: the same with the rules swapped
       (the clone of A: same seed, same nature, the other life). shufA, shufB:
       as A, B, but each answer is credited to a cast member drawn at random
       (the shuffled-answers control). deaf: both P-deaf all life (P-deaf
       control and the baseline for L2).
  L35  partner Fenn, N35 nights; focal as L12. Twin A: Fenn P-kind, P-hard
       from 0.2 N (betrayal 1), P-deaf from 0.4 N (the shared test window and
       the contingency broken), P-kind from 0.5 N, P-hard from 0.7 N
       (betrayal 2). Twin B: P-hard, then P-kind from 0.2 N, then the same as
       A from 0.4 N. So at 0.4 N the twins have lived the same events in the
       other order (L5), and at 0.7 N twin A meets its SECOND betrayal and
       twin B its FIRST, at the same age and after the same last 0.3 N (L3).

    lab/.venv/bin/python -m lab.mindL.tests_L run L04 --workers 6 --ppo-workers 2
    lab/.venv/bin/python -m lab.mindL.tests_L report

Each lifetime is one file in lab/results/mindL/runs/ (skipped if present:
the sweep restarts where it stopped). The per-decision rows that L4 needs are
in ~/.cache/mark1-build/mindL/runs_raw/.
"""
from __future__ import annotations

import argparse
import json
import os
import multiprocessing as mp
import random
import time
from pathlib import Path

import numpy as np

from lab.mindL import drives as H
from lab.world2.world import DRIVES, HARSH_ACTS, KIND_ACTS

OUT = Path(__file__).resolve().parents[1] / "results" / "mindL"
RUNS = Path(os.environ.get("MINDL_RUNS", OUT / "runs"))
RAW = Path.home() / ".cache" / "mark1-build" / "mindL" / "runs_raw"
SEEDS = tuple(range(1, 11))
CHOOSERS = ("mind", "bandit", "ppo", "rules", "random")
LEARNERS = ("mind", "bandit", "ppo")
FOCALS3 = ("Tom", "Sir Ada", "Marta")
N04, N12, N35 = 300, 300, 400
FADE = float(os.environ.get("MINDL_FADE", "0.9"))
# the settings of the learners retuned on seed 0 in the fixed world (round 2): {"mind2": {...}, "bandit": {...},
# "ppo": {...}}; the file named by MINDL_TUNED. Unset: the settings of round 1 (mind.DEFAULT, nulls.py).
TUNED = json.loads(Path(os.environ["MINDL_TUNED"]).read_text()) if os.environ.get("MINDL_TUNED") else {}
P = 0.9
PROVOKE = {"insult", "strike", "shove", "threaten"}
KINDA = set(KIND_ACTS)
HARSHA = set(HARSH_ACTS) | {"steal"}
# the acts that serve a drive (for L4's context index)
CATS = {"energy": ("eat", "drink"), "rest": ("sit down", "sleep"), "curiosity": ("look around", "open"),
        "safety": ("flee to",)}
DESIGNER = {**{v: 1.0 for v in KINDA}, **{v: -1.0 for v in HARSHA},
            "eat": .5, "drink": .5, "sit down": .5, "sleep": .5, "look around": .5}


# ------------------------------------------------------------------ the runs
def specs(family, nights=None, choosers=CHOOSERS, flags="all"):
    out = []
    if family == "L04":
        n = nights or N04
        for c in choosers:
            for s in SEEDS:
                out.append(dict(family="L04", variant="drives", chooser=c, seed=s, nights=n,
                                focal=NAMES()[s % 5], partners={}, reward="drives"))
        for c in [x for x in choosers if x in LEARNERS + ("mind2",)]:
            for s in SEEDS:
                out.append(dict(family="L04", variant="designer", chooser=c, seed=s, nights=n,
                                focal=NAMES()[s % 5], partners={}, reward="designer"))
    elif family == "L12":
        n = nights or N12
        cut = (2 * n) // 3
        lives = {"A": ("kind", "hard"), "B": ("hard", "kind")}
        for c in choosers:
            for s in SEEDS:
                for v, (f, o) in lives.items():
                    for shuf in (("", "shuf") if c in LEARNERS + ("mind2",) else ("",)):   # the shuffle control only for learners
                        tag = "~shuf" if shuf else ""
                        out.append(dict(family="L12", variant=shuf + v, chooser=c, seed=s, nights=n,
                                        focal=FOCALS3[s % 3], reward="drives",
                                        partners={"Fenn": [[0, f + tag], [cut, "deaf"]],
                                                  "Oren": [[0, o + tag], [cut, "deaf"]]}))
                out.append(dict(family="L12", variant="deaf", chooser=c, seed=s, nights=n, focal=FOCALS3[s % 3],
                                reward="drives", partners={"Fenn": [[0, "deaf"]], "Oren": [[0, "deaf"]]}))
    elif family == "L35":
        n = nights or N35
        b = [round(n * x) for x in (0.2, 0.4, 0.5, 0.7)]
        twins = {"A": ("kind", "hard"), "B": ("hard", "kind")}
        for c in choosers:
            for s in SEEDS:
                for v, (x1, x2) in twins.items():
                    out.append(dict(family="L35", variant=v, chooser=c, seed=s, nights=n, focal=FOCALS3[s % 3],
                                    reward="drives",
                                    partners={"Fenn": [[0, x1], [b[0], x2], [b[1], "deaf"], [b[2], "kind"],
                                                       [b[3], "hard"]]}))
    else:
        raise ValueError(family)
    for x in out:
        x["fade"] = FADE
        x["id"] = f"{x['family']}_{x['variant']}_{x['chooser']}_seed{x['seed']}_n{x['nights']}"
        if flags != "all":              # the 2026-10-29 fixes: a world with reprisal and wear ("all2")
            x["flags"], x["id"] = flags, x["id"] + "_" + flags
    return out


def NAMES():
    from lab.world2.lifetime import NAMES as N
    return N


def make_one(spec, seed):
    from lab.world2.partners import Deaf, Hard, Kind
    base, _, tag = spec.partition("~")
    p = {"kind": lambda: Kind(P, seed), "hard": lambda: Hard(P, seed), "deaf": lambda: Deaf(seed)}[base]()
    if tag == "shuf":
        rng = random.Random(seed + 7777)
        inner = p.hear

        def hear(w, me, e):
            others = [n for n in NAMES() if n != me.name]
            inner(w, me, {**e, "actor": rng.choice(others)})
        p.hear = hear
        p.rule = (p.rule + "; SHUFFLED: each act it would answer is credited to one of the other four cast "
                  "members drawn at random, and the answer goes to that one")
        p.name += "~shuf"
    return p


class Seq:
    """A partner whose rule changes at set nights: phases [(from night, spec)]."""

    def __init__(self, phases, seed):
        self.parts = [(n, make_one(s, seed + 10 * i)) for i, (n, s) in enumerate(phases)]
        self.name = ">".join(p.name for _, p in self.parts)
        self.rule = " | ".join(f"from night {n}: {p.rule}" for n, p in self.parts)
        self.now = self.parts[0][1]

    def night_start(self, w, night):
        self.now = [p for n, p in self.parts if n <= night][-1]
        self.now.night_start(w, night)

    def probs(self, w, p, opts):
        return self.now.probs(w, p, opts)


class Designer:
    """L4's designer reward, as an error whose drop is the reward."""
    scale = 0.1

    def __init__(self):
        self.w = None

    def __call__(self, w, p):
        if w is not self.w:
            self.w, self.ptr, self.pts = w, 0, {}
        for e in w.log[self.ptr:]:
            if e["done"] and "options" in e:
                self.pts[e["actor"]] = self.pts.get(e["actor"], 0.0) + DESIGNER.get(e["action"], 0.0)
        self.ptr = len(w.log)
        return -self.scale * self.pts.get(p.name, 0.0)


def make_recorder(default, by, focal):
    from lab.world2.lifetime import Cast

    class Recorder(Cast):
        """A Cast that records each choice of the focal character: its needs,
        who is here, and for each drive in CATS whether an act serving it was
        on the menu and the chance the chooser put on such acts."""

        def __init__(self):
            super().__init__(default, by)
            self.focal, self.night, self.near, self.ndec, self.dec = focal, 0, {}, 0, []

        def night_start(self, w, night):
            self.night, self.near, self.ndec = night, {}, 0
            super().night_start(w, night)

        def probs(self, w, p, opts):
            pc, pa = super().probs(w, p, opts)
            if p.name == self.focal:
                self.ndec += 1
                for q in w.here(p):
                    self.near[q.name] = self.near.get(q.name, 0) + 1
                row = [self.night, *H.needs(w, p)]
                for verbs in CATS.values():
                    idx = [i for i, o in enumerate(opts) if o["verb"] in verbs]
                    row += [1.0 if idx else 0.0, float(sum(pc[i] for i in idx)), len(idx) / len(opts)]
                self.dec.append(row)
            return pc, pa

    return Recorder()


def night_row(w, n, rec, focal, agent):
    p = w.people[focal]
    others = [x for x in NAMES() if x != focal]
    acts, to, frm = {}, {x: [0, 0, 0, 0] for x in others}, {x: [0, 0] for x in others}
    for e in w.log:
        if e["actor"] == focal and "options" in e:
            a = e["action"]
            acts[a] = acts.get(a, 0) + 1
            if e["target"] in to:
                c = to[e["target"]]
                c[0] += a in KINDA
                c[1] += a in PROVOKE
                c[2] += a in HARSHA
                c[3] += 1
        elif e["target"] == focal and e["actor"] in frm and e["done"]:
            frm[e["actor"]][0] += e["action"] in KINDA
            frm[e["actor"]][1] += e["action"] in HARSHA
    row = {"n": n, "dead": bool(p.dead), "acts": acts, "to": to, "from": frm,
           "aff": {x: round(float(p.aff.get(x, 0.0)), 3) for x in others},
           "near": {x: round(rec.near.get(x, 0) / max(1, rec.ndec), 3) for x in others}, "ndec": rec.ndec}
    if agent is not None and hasattr(agent, "pm"):
        from lab.mindL.partners import X_KEYS
        x0 = np.zeros(1 + len(X_KEYS))
        x0[0] = 1.0
        mi = {}
        for x in others:
            pk = agent.pm.predict(x, "give", x0)[0]
            ph = [agent.pm.predict(x, v, x0)[1] for v in PROVOKE if (x, v) in agent.pm.w]
            mi[x] = [round(float(pk), 3), round(float(np.mean(ph)) if ph else 0.5, 3),
                     round(float(agent.val.value(x)), 4)]
        row["mind"] = mi
    return row


def context_index(dec, nights):
    """L4: per drive, over the choices in the last half of the life with an act
    serving it on the menu: the Spearman correlation of the drive's need with
    the LIFT, the chance the chooser put on such acts minus their share of the
    menu (what a uniform draw would put on them; the menu itself changes with
    the state, so the raw chance is confounded: random shows it). Also the
    chance and the menu share in the bottom and top third of the need."""
    from scipy.stats import spearmanr
    D = np.asarray(dec, dtype=float)
    out = {}
    if len(D) == 0:
        return out
    late = D[:, 0] >= nights // 2
    for ci, c in enumerate(CATS):
        x = D[:, 1 + DRIVES.index(c)]
        a, m, sh = D[:, 8 + 3 * ci], D[:, 9 + 3 * ci], D[:, 10 + 3 * ci]
        sel = late & (a > 0)
        xs, ys, ss = x[sel], m[sel], sh[sel]
        lift = ys - ss
        if len(xs) < 30 or xs.std() == 0 or lift.std() < 1e-9:
            out[c] = {"n": int(len(xs)), "rho": None}
            continue
        lo, hi = np.quantile(xs, [1 / 3, 2 / 3])
        out[c] = {"n": int(len(xs)), "rho": round(float(spearmanr(xs, lift)[0]), 4),
                  "chance_low_third": round(float(ys[xs <= lo].mean()), 4),
                  "chance_top_third": round(float(ys[xs >= hi].mean()), 4),
                  "share_low_third": round(float(ss[xs <= lo].mean()), 4),
                  "share_top_third": round(float(ss[xs >= hi].mean()), 4)}
    return out


def run(spec):
    """One lifetime; writes RUNS/<id>.json."""
    from lab.mindL.mind import make_mind
    from lab.mindL.nulls import make_bandit, run_ppo
    from lab.world2.lifetime import lifetime
    from lab.world2.policy import Random, Rules

    out = RUNS / f"{spec['id']}.json"
    if out.exists():
        return spec["id"], "kept"
    t0 = time.perf_counter()
    c, seed, focal, nights = spec["chooser"], spec["seed"], spec["focal"], spec["nights"]
    partners = {name: Seq(ph, seed * 100 + i) for i, (name, ph) in enumerate(sorted(spec["partners"].items()))}
    designer = Designer() if spec["reward"] == "designer" else None
    rows, box, pol = [], {}, None

    def on_night(w, n):
        agent = getattr(pol, "agents", {}).get(focal) if pol is not None else None
        rows.append(night_row(w, n, box["rec"], focal, agent))

    def cast(d, by):
        box["rec"] = make_recorder(d, {**partners, **by}, focal)
        return box["rec"]

    info = {}
    if c == "ppo":
        stats, info = run_ppo(focal, nights, seed, spec.get("flags", "all"), on_night=on_night, threads=1, cast=cast, err=designer,
                              fade=spec["fade"], **TUNED.get("ppo", {}))
        info = {k: v for k, v in info.items() if k in ("decisions", "truncated_menus")}
    else:
        if c == "mind":
            pol = make_mind(seed)
        elif c == "mind2":              # the fixes (mind.FIXES); meant for flags "all2"
            from lab.mindL.mind import FIXES
            pol = make_mind(seed, **{**FIXES, **TUNED.get("mind2", {})})
        elif c == "bandit":
            pol = make_bandit(seed, **TUNED.get("bandit", {}))
        else:
            pol = Rules() if c == "rules" else Random()
        if designer is not None:
            make = pol.make

            def mk(n, k, make=make):
                a = make(n, k)
                a.err_fn = designer
                return a
            pol.make = mk
        _, stats = lifetime(cast(Rules(), {focal: pol}), nights, seed, flags=spec.get("flags", "all"), on_night=on_night,
                            fade=spec["fade"])
        a = getattr(pol, "agents", {}).get(focal)
        if a is not None:
            info["decisions"] = a.decisions
    rec = box["rec"]
    res = {**spec, "rules": {k: v.rule for k, v in partners.items()},
           "drive_error": [s["drive_error"][focal] for s in stats], "nights_rows": rows,
           "context": context_index(rec.dec, nights), "seconds": round(time.perf_counter() - t0, 1), **info}
    if spec["family"] == "L04":
        RAW.mkdir(parents=True, exist_ok=True)
        np.save(RAW / f"{spec['id']}.npy", np.asarray(rec.dec, dtype=np.float32))
    RUNS.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    tmp.replace(out)
    return spec["id"], res["seconds"]


def _work(spec, deadline=None):
    if deadline and time.time() > deadline:     # a block of the sweep is over: start no new lifetime
        return spec["id"], "deferred"
    try:
        return run(spec)
    except Exception as e:  # report and go on: one failed lifetime must not stop the sweep
        import traceback
        return spec["id"], "FAILED " + "".join(traceback.format_exception(e))[-1500:]


def sweep(family, workers, nights=None, only=None, ppo_workers=2, choosers=CHOOSERS, flags="all", seeds=None,
          stop_after=None):
    """PPO lifetimes (torch, about 700 MB each) run in their own pool of
    `ppo_workers`; the rest in a pool of `workers` - ppo_workers."""
    todo = [s for s in specs(family, nights, choosers, flags) if (not seeds or s["seed"] in seeds) and not (RUNS / f"{s['id']}.json").exists()]
    if only:
        todo = [s for s in todo if s["chooser"] in only]
    cost = {"mind2": 4, "mind": 4, "bandit": 3, "ppo": 3, "rules": 1, "random": 1}
    todo.sort(key=lambda s: -cost[s["chooser"]] * s["nights"])
    ppo = [s for s in todo if s["chooser"] == "ppo"]
    rest = [s for s in todo if s["chooser"] != "ppo"]
    pw = min(ppo_workers, max(1, workers - 1)) if ppo else 0
    print(f"{family}: {len(todo)} lifetimes to run ({len(ppo)} PPO), {workers} workers ({pw} for PPO)", flush=True)
    t0, k = time.perf_counter(), [0]
    deadline = time.time() + 60 * stop_after if stop_after else None     # minutes: no lifetime STARTS after it

    def cb(r):
        k[0] += 1
        print(f"[{k[0]}/{len(todo)} {time.perf_counter() - t0:.0f} s] {r[0]}: {r[1]}", flush=True)

    ctx = mp.get_context("spawn")
    with ctx.Pool(max(1, workers - pw), maxtasksperchild=1) as pa, ctx.Pool(max(1, pw), maxtasksperchild=1) as pb:
        rs = [pa.apply_async(_work, (s, deadline), callback=cb) for s in rest]
        rs += [pb.apply_async(_work, (s, deadline), callback=cb) for s in ppo]
        for r in rs:
            r.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["run", "report", "one"])
    ap.add_argument("family", nargs="?", default="L04")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--ppo-workers", type=int, default=2)
    ap.add_argument("--nights", type=int, default=None)
    ap.add_argument("--only", default="")
    ap.add_argument("--id", default="")
    ap.add_argument("--choosers", default="", help="comma list (default: the five); mind2 = the fixes")
    ap.add_argument("--flags", default="all", help="the world: all (first round) or all2 (reprisal and wear too)")
    ap.add_argument("--seeds", default="", help="comma list (default 1-10)")
    ap.add_argument("--stop-after", type=float, default=None, help="minutes: start no new lifetime after this")
    a = ap.parse_args()
    if a.what == "run":
        sweep(a.family, a.workers, a.nights, [x for x in a.only.split(",") if x], a.ppo_workers,
              tuple(x for x in a.choosers.split(",") if x) or CHOOSERS, a.flags,
              {int(x) for x in a.seeds.split(",") if x}, a.stop_after)
    elif a.what == "one":
        s = next(s for s in specs(a.family, a.nights, tuple(x for x in a.choosers.split(",") if x) or CHOOSERS, a.flags)
                 if s["id"] == a.id)
        print(run(s))
    else:
        from lab.mindL.report_L import report
        report()


if __name__ == "__main__":
    main()
