"""PROTOTYPE: the lab inn behind the mind interface (scratch only; lab is
imported read-only to check parity, the real port copies the file into src).

    python3 rooms_proto.py
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/home/nisargparikh/Desktop/Fun Stuff/Mark_1")
sys.path.insert(0, __file__.rsplit("/", 1)[0])

from lab.world2.policy import MID, T_RULES, Rules, softmax, utility  # noqa: E402
from lab.world2.world import ROOMS, TRAITS, STATE_SPREAD, World  # noqa: E402
from mind_api import (Belief, DecisionRequest, Job, Option, Percept, PriorChooser,  # noqa: E402
                      Ref, Why, prerank)

VERB_ID = {"walk to": "walk_to", "flee to": "flee_to", "shout for help": "shout_for_help",
           "help up": "help_up"}
INSTRUMENT = {"strike"}          # obj is the instrument for these verbs; the target otherwise
CAP = {"the sword": "sharp", "the axe": "sharp", "the dagger": "sharp", "the kitchen knife": "sharp"}
SAYS = {"sharp": "something with a sharp edge"}


def to_option(w, p, o):
    roles = {}
    if o["target"] in w.people:
        roles["target"] = Ref(o["target"], o["target"], "person")
    elif o["target"] in ROOMS:
        roles["place"] = Ref(o["target"], o["target"], "place")
    if o["obj"]:
        role = "instrument" if o["verb"] in INSTRUMENT else ("thing" if "target" in roles else "target")
        roles[role] = Ref(o["obj"], o["obj"], "thing")
    f = {k: v for k, v in o["f"].items() if isinstance(v, (int, float))}
    return Option(VERB_ID.get(o["verb"], o["verb"]), roles, o["text"], o["dur"], f, raw=o)


class Rooms:
    """Backend A over the lab inn. `compat=True` reproduces lab World.step(Rules())
    draw for draw (no pre-rank, the world's own rng picks)."""

    def __init__(self, seed, turns=300, compat=False, chooser=None, k=16):
        self.w = World(seed=seed, turns=turns)
        self.compat, self.k = compat, k
        self.chooser = chooser or PriorChooser(rng=self.w.rng if compat else None, seed=seed)
        self.n_req = 0
        self.stats = {"full": [], "kept": [], "rank": [], "desc_words": [], "beliefs": [], "ms": []}

    # ---------------------------------------------------------------- the loop
    def step(self):
        w = self.w
        w.t += 1
        if w.bandit_at == w.t:
            w._bandit_arrives()
        for p in w.people.values():
            w._drift(p)
        order = list(w.people.values())
        w.rng.shuffle(order)
        for p in order:
            if not p.acts():
                p.doing = None
                continue
            if p.doing is None:
                self.decide(p, "idle")
            else:
                p.doing["left"] -= 1
                if p.doing["left"] <= 0:
                    w._finish(p)

    def decide(self, p, reason):
        w = self.w
        t0 = time.perf_counter()
        p.traits = {t: min(1, max(0, w.rng.gauss(p.base[t], STATE_SPREAD))) for t in TRAITS}
        w._look(p, "sight")
        w._check_own(p)
        raw = w.options(p)
        opts = [to_option(w, p, o) for o in raw]
        u = [utility(w, p, o, p.traits) for o in raw]
        ua = [utility(w, p, o, MID) for o in raw]
        keep = list(range(len(opts))) if self.compat else \
            prerank(opts, [max(a, b) for a, b in zip(u, ua)], self.k)
        sit = self.describe(p, [opts[i] for i in keep]) if not self.compat else ""
        self.n_req += 1
        req = DecisionRequest(f"{w.t}:{p.name}:{self.n_req}", p.name, w.t, sit,
                              f"What does {p.name} do next?", tuple(opts[i] for i in keep),
                              tuple(u[i] for i in keep), reason, w.t)
        self.chooser.submit(req)
        dec = self.chooser.poll(req.id, block=True)
        j = keep[dec.pick]
        o = raw[j]
        pc = softmax(u, T_RULES)
        pa = softmax(ua, T_RULES)
        self.stats["full"].append(len(opts))
        self.stats["kept"].append(len(keep))
        self.stats["rank"].append(sorted(range(len(u)), key=lambda i: -u[i]).index(j))
        self.stats["ms"].append(1000 * (time.perf_counter() - t0))
        if sit:
            self.stats["desc_words"].append(len(sit.split()))
        job = Job(p.name, opts[j], w.t, o["dur"],
                  Why(req.id, self.chooser.name, reason, tuple(p.pending)))
        p.doing = {**o, "left": o["dur"], "job": job, "choice": {
            "options": [x["text"] for x in raw], "chosen": j,
            "p_char": [round(x, 5) for x in pc], "p_avg": [round(x, 5) for x in pa],
            "seen": list(p.pending), "traits": {k: round(v, 3) for k, v in p.traits.items()}}}
        p.pending = []
        if o["dur"] <= 1:
            w._finish(p)

    # ---------------------------------------------------------------- inputs
    def percepts(self, p, n=5):
        evs = [e for e in self.w.log if e["id"] in p.perceived][-n:]
        out = []
        for e in evs:
            txt = f"{e['actor']} {'tried to ' if not e['done'] else ''}{e['text']}"
            if e.get("result"):
                txt += f" ({e['result']})"
            out.append(Percept(e["t"], "saw", txt + ".", e["id"], e["actor"], e["action"],
                               e.get("target"), "done" if e["done"] else "stopped"))
        return out

    def beliefs(self, p, about=(), n=2):
        """At most n 'took' beliefs, those about things/people on the menu first."""
        bs = [Belief(k, th, who, src) for (k, th), (who, src) in p.bel.items()
              if k == "took" and who != p.name]
        bs.sort(key=lambda b: (b.about not in about and b.value not in about))
        return bs[:n]

    def describe(self, p, opts):
        w = self.w
        s = [f"It is night at the village inn. {p.name} is {p.public}, in {p.room}."]
        here = w.here(p)
        if here:
            s.append("Here: " + "; ".join(
                f"{q.name} ({q.public}{', lying on the floor' if q.down else ''}"
                f"{', holding ' + ', '.join(w.held(q, visible_only=True)) if w.held(q, visible_only=True) else ''})"
                for q in here) + ".")
        mine = w.held(p)
        s.append(f"{p.name} holds " + (", ".join(mine) if mine else "nothing") + ".")
        feel = [x for x, c in (("hungry", p.hunger > .6), ("thirsty", p.thirst > .6), ("afraid", p.fear > .4),
                               ("drunk", p.drunk > .4), ("hurt", p.hp < 7)) if c]
        mad = [n for n, a in p.anger.items() if a > .4]
        if mad:
            feel.append("angry at " + " and ".join(mad))
        if feel:
            s.append(f"{p.name} feels " + ", ".join(feel) + ".")
        # WHAT IS MISSING, never what is there: a need with no way to meet it here
        if p.hunger > .6 and not any(o.verb == "eat" for o in opts):
            s.append(f"{p.name} has no food to hand.")
        pc = self.percepts(p)
        if pc:
            s.append("Recently: " + " ".join(x.text for x in pc))
        about = {r.id for o in opts for r in o.roles.values()}
        bs = self.beliefs(p, about)
        self.stats["beliefs"].append(len(bs))
        for b in bs:
            s.append(f"{p.name} believes {b.value} took {b.about}.")
        return " ".join(s)


def lab_night(seed, turns):
    w = World(seed=seed, turns=turns)
    pol = Rules()
    for _ in range(turns):
        w.step(pol)
    return w.log


def port_night(seed, turns, **kw):
    r = Rooms(seed, turns, **kw)
    for _ in range(turns):
        r.step()
    return r


if __name__ == "__main__":
    turns = 300
    # 1. parity: the port in compat mode is the lab, draw for draw
    for seed in range(3):
        a = lab_night(seed, turns)
        b = port_night(seed, turns, compat=True).w.log
        same = json.dumps(a, sort_keys=True, default=list) == json.dumps(b, sort_keys=True, default=list)
        print(f"seed {seed}: lab events {len(a)}, port events {len(b)}, identical={same}")
    # 2. the interface mode: pre-rank to 16, clean description
    import statistics as st
    agg = {k: [] for k in ("full", "kept", "rank", "desc_words", "beliefs", "ms")}
    t0 = time.perf_counter()
    nreq = 0
    for seed in range(6):
        r = port_night(seed, turns)
        nreq += r.n_req
        for k in agg:
            agg[k] += r.stats[k]
    dt = time.perf_counter() - t0
    full = sorted(agg["full"])
    print(f"decisions {nreq} over {6 * turns} turns ({nreq / (6 * turns):.2f}/turn), {dt:.2f}s total, "
          f"{1000 * dt / (6 * turns):.2f} ms/turn")
    print(f"full menu: median {st.median(full)}, p90 {full[int(.9 * len(full))]}, max {full[-1]}, "
          f"share > 16: {sum(x > 16 for x in full) / len(full):.2f}")
    print(f"kept: max {max(agg['kept'])}; rules pick within the kept 16 by construction; "
          f"pick rank top1 {sum(x == 0 for x in agg['rank']) / len(agg['rank']):.2f}")
    print(f"decide() ms: median {st.median(agg['ms']):.2f}, p99 {sorted(agg['ms'])[int(.99 * len(agg['ms']))]:.2f}")
    print(f"describe words: median {st.median(agg['desc_words'])}, max {max(agg['desc_words'])}; "
          f"beliefs shown max {max(agg['beliefs'])}")
    r = port_night(0, 120)
    p = r.w.people["Fenn"]
    raw = r.w.options(p)
    opts = [to_option(r.w, p, o) for o in raw]
    print("\nEXAMPLE describe (Fenn, t=120):\n" + r.describe(p, opts))
