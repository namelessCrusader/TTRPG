"""A PRIMED CHILDHOOD (mindL6 stage B1; owner decision 2026-11-23 (a): "a primed childhood in the safe inn, plus what evolution gives, plus cultural priors").

One childhood per person and seed: the SAME learner (the same mind option set the adult will use) lives from age 1 to age 12 in the SAFE family inn (no lethal, no perma)
in a CHILD body (the world's `young` argument), then the whole learner is SAVED (Mind.dump) and loaded into the adult (PerCharacter.load).

WHO RAISES IT (the carers; u). The other six people of the family inn are the CARERS and are cast as the RULES policy: the previous generation, the culture's established
practice (u: no source says how a culture's practice should be cast; the rules are the bar the learned mind has to beat, and they are the stand-in for "what adults of this
culture do"). Pip and Gran keep their body sheets, so a child (7) and an old woman are among the carers.

WHAT THE CHILD IS GIVEN (all from the library answer, D-c to D-e; (u) where marked):
  GIVEN. A dependent (under world.DEPENDENT_AGE = 4, u) cries (world.cry_level: graded by the unmet needs, cause blind) and the carers' serve options for it are driven by the cry:
         they bring it water or a ration without knowing which it needs (the world, flag body2). Nothing here: the world does it.
  SENT.  From the age of peak delegation (water carrying: girls 4, boys 9; Stieglitz et al. 2012, Tsimane) a child is SENT for water: on a night with an errand (chance = the
         peak share, 0.83 girls and 0.79 boys, times a bell of width ERR_SIGMA = 3 years (u), around the peak age; no errand under 4) the child's choice at some turn between 30 and 270
         is replaced by a step toward the water (the yard) until it is there; there it chooses freely. The act it was told to do is learned as its own act (the answer, L7: "the
         learner already counts that at full weight"). Sex is a table (u) of the harness; the world has no sex.
  TAUGHT. Teaching episodes happen at TEACH_PER_HOUR = 7 an hour of game time (Hadza and BaYaka children 6-8 an hour, Lew-Levy 2019); 75 percent come from another child, 25 percent
         from an adult (same paper). A teacher passes on an act it does itself (one of its last 12 acts), through drives2.told (lossy by design: take-up chance 0.6, noise 0.25,
         all u). The ADULT teaches what it KNOWS: each carer has a shadow world model trained on its own acts' outcomes by the same delta rule as the mind (a carer is not learned,
         it is practice; the shadow is what it has experienced). A CHILD teacher knows only what it was told (two lossy hops: adult -> peer -> the child; 2 peers; each peer counts a
         told verb as known). The weight of what is told never grows past TOLD_CAP = 9 pseudo-counts a verb (u: three tellings), so experience always overrides.
  WATCHED. The mind's own watching (letter W) is on, with the weight of the watcher's AGE (drives2.watch_weight: nothing at birth; the child's outcome weight, falling to the adult's).
LENGTH. The childhood is CHILD_NIGHTS nights long (30 or 90 here; a night is a day, so the real childhood is thousands: u; run two lengths as a sensitivity check). Ages run linearly
from AGE0 = 1 to AGE1 = 12 over the nights.

    lab/.venv/bin/python -m lab.mindL.childhood  <seed> <person> <nights>      (one childhood, cached)
"""
from __future__ import annotations

import collections
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from lab.mindL import reader
from lab.mindL.drives2 import Drives2, told
from lab.mindL.household import FLAGS as FAMILY_FLAGS_SAFE, NAMES, make_policy
from lab.mindL.world_model import WorldModelP
from lab.world2.lifetime import hook, night_seed
from lab.world2.policy import Rules
from lab.world2.world import DEPENDENT_AGE, FAMILY_CAST, ROOMS, World

# the learner of the stage B work: the SAME option set in the childhood and in the adult's run
LEARNER = dict(drives2="SCBDW", d2={"dial": True}, watch_age=True, taste=True, concerns=False)     # for make_policy
ADULT = {k: v for k, v in LEARNER.items() if k != "concerns"}                                          # for household.run_full(arm="off", extra=ADULT): arm "off" sets concerns False
WFLAGS = ("drives2", "body2")
SEX = {"Jon": "m", "Mara": "f", "Pip": "m", "Gran": "f", "Tom": "m", "Sir Ada": "f", "Oren": "m"}      # (u) the harness's, not the world's
PEAK_AGE = {"f": 4.0, "m": 9.0}              # Stieglitz et al. 2012, Table 1: the age at which the chance of being delegated water carrying peaks
PEAK_SHARE = {"f": 0.83, "m": 0.79}          # the share who carried water on the sampled days
ERR_SIGMA = 3.0                              # (u) width of the bell of the delegation chance around the peak age, years
AGE0, AGE1 = 1.0, 12.0
TEACH_PER_HOUR, TURN_MIN, PEER_SHARE = 7.0, 4.8, 0.75
N_PEERS, RECENT, TOLD_CAP = 2, 12, 9.0
WATER_ROOM = "the yard"
SCR = Path("/home/nisargparikh/.cache/mark1-build/mindL6/B1/childhoods")


def errand_chance(sex, age):
    """The chance that a night has a water errand for a child of this sex and age: 0 under the dependent age, else the sex's peak share times a bell around its peak age."""
    if age < DEPENDENT_AGE:
        return 0.0
    return PEAK_SHARE[sex] * math.exp(-0.5 * ((age - PEAK_AGE[sex]) / ERR_SIGMA) ** 2)


def next_hop(src, dst):
    """The next room on the shortest path from src to dst (None if there already or no path)."""
    if src == dst:
        return None
    seen, edge = {src: None}, [src]
    while edge:
        nxt = []
        for x in edge:
            for r in ROOMS[x]:
                if r not in seen:
                    seen[r] = x
                    nxt.append(r)
                    if r == dst:
                        while seen[r] != src:
                            r = seen[r]
                        return r
        edge = nxt
    return None


class Shadow:
    """What a carer has experienced of its own acts, by the same delta rule as the mind's world model (a carer is not learned: this is its practice, the thing it can tell)."""

    def __init__(self, letters):
        self.D = Drives2(letters)
        self.wm = WorldModelP(reader.NV, self.D.ND, 1 + reader.NK + self.D.ND + 1, 0.2)
        self.open = None
        self.recent = collections.deque(maxlen=RECENT)


class Childhood:
    """The policy of one childhood: the focal learner, the carers as Rules, the shadows, the errands and the teaching."""

    def __init__(self, seed, person, nights, learner_cfg=None, letters="SCBDW"):
        self.seed, self.person, self.nights = seed, person, nights
        self.idx = NAMES.index(person)
        self.sex = SEX[person]
        self.mind = make_policy(seed, **{**LEARNER, **(learner_cfg or {})})
        self.rules = Rules()
        self.shadow = {n: Shadow(letters) for n in NAMES if n != person}
        self.peers = [WorldModelP(reader.NV, self.shadow[next(iter(self.shadow))].D.ND, 1 + reader.NK + self.shadow[next(iter(self.shadow))].D.ND + 1, 0.2)
                      for _ in range(N_PEERS)]
        self.rng = np.random.default_rng([seed, self.idx, 99])       # teaching and errands
        self.crng = np.random.default_rng([seed, self.idx, 77])      # the carers' draws (their own generator: the world's draws stay as many as before)
        self.errand_turn, self.errand_on = None, False
        self.stats = dict(episodes=0, via_child=0, taken=0, errands=0, errand_nights=0, given=0, own_drinks=[], taught_drink=0)
        nd = self.shadow[next(iter(self.shadow))].D.ND
        self.d_cols = [1 + reader.NK + d for d in range(nd)]
        self.name = "childhood"

    # ------------------------------------------------------------------ the policy interface
    def age(self, night):
        return AGE0 + (AGE1 - AGE0) * night / max(1, self.nights)

    def night_start(self, w, night):
        age = self.age(night)
        self.cur_age = age
        self.errand_on = False
        self.errand_turn = None
        if self.rng.random() < errand_chance(self.sex, age):
            self.errand_turn = int(self.rng.integers(30, 271))
            self.stats["errand_nights"] += 1
        self.mind.agent(w.people[self.person])                      # made now so that told() can write to it
        hook(self.mind, "night_start", w, night)

    def night_end(self, w, night):
        hook(self.mind, "night_end", w, night)
        for sh in self.shadow.values():
            sh.open = None

    def on_death(self, w, person, event):
        hook(self.mind, "on_death", w, person, event)

    def probs(self, w, p, opts):
        if p.name == self.person:
            ag = self.mind.agent(p)
            a = ag.decide(w, p, opts)
            f = self._errand(w, p, opts)
            if f is not None and f != a:                            # sent: the imposed act is the one that is done and learned
                o = opts[f]
                st = ag.step
                st.a, st.act, st.cue, st.opt = f, reader.key(o), reader.cue(o), o
                a = f
            pc = [0.0] * len(opts)
            pc[a] = 1.0
            return pc, pc
        pc, pa = self.rules.probs(w, p, opts)
        sh = self.shadow.get(p.name)
        if sh is None:                                              # the bandit from the road: not a carer, nothing it does is taught (the world's own draw, as in any rules run)
            return pc, pa
        need = sh.D.needs(w, p)
        if sh.open is not None:
            v, z, n0 = sh.open
            sh.wm.update(v, z, need - n0)
        menu = reader.encode(opts, need, extra=1)
        i = int(self.crng.choice(len(pc), p=np.asarray(pc) / sum(pc)))      # the carer's act is drawn HERE (so its shadow knows what it did) and handed to the world one-hot
        sh.open = (int(menu.v[i]), menu.Z[i].copy(), need.copy())
        sh.recent.append(int(menu.v[i]))
        return _one_hot(len(opts), i), pa

    def _errand(self, w, p, opts):
        """The index of the option that is a step toward the water, if the child is on an errand now, else None."""
        if self.errand_turn is None:
            return None
        if not self.errand_on and w.t >= self.errand_turn:
            self.errand_on = True
            self.stats["errands"] += 1
        if not self.errand_on:
            return None
        if p.room == WATER_ROOM:
            self.errand_on, self.errand_turn = False, None
            return None
        hop = next_hop(p.room, WATER_ROOM)
        for i, o in enumerate(opts):
            if o["verb"] == "walk to" and o["target"] == hop:
                return i
        return None

    # ------------------------------------------------------------------ teaching (after each turn)
    def teach(self, w):
        lam = TEACH_PER_HOUR * TURN_MIN / 60.0
        k = int(self.rng.poisson(lam))
        ag = self.mind.agents.get(self.person)
        if ag is None:
            return
        for _ in range(k):
            carers = [n for n, sh in self.shadow.items() if sh.recent]
            if not carers:
                return
            c = self.shadow[carers[int(self.rng.integers(len(carers)))]]
            v = c.recent[int(self.rng.integers(len(c.recent)))]
            self.stats["episodes"] += 1
            if self.rng.random() < PEER_SHARE:                      # through another child: two lossy hops
                self.stats["via_child"] += 1
                peer = self.peers[int(self.rng.integers(N_PEERS))]
                if told(c.wm, peer, [v], self.rng, self.d_cols, n0_max=TOLD_CAP):
                    peer.cnt[v] = max(peer.cnt[v], 3.0)             # what a child has been told counts as known to it
                    got = told(peer, ag.wm, [v], self.rng, self.d_cols, n0_max=TOLD_CAP)
                else:
                    got = []
            else:
                got = told(c.wm, ag.wm, [v], self.rng, self.d_cols, n0_max=TOLD_CAP)
            if got:
                self.stats["taken"] += 1
                self.stats["taught_drink"] += int(v == reader.VI["drink"])


def _one_hot(n, i):
    pc = [0.0] * n
    pc[i] = 1.0
    return pc


def run_childhood(seed, person, nights, cache=True, learner_cfg=None, tag="", teach=True, wflags=None, scr=None):
    """One childhood. Returns dict(blob, stats, nights); cached per (seed, person, nights). B2 options (defaults = B1): tag/scr name the cache, teach=False has no teaching
    episodes, wflags the world flags of the childhood (learner_cfg replaces keys of LEARNER; drives2=False is the old learner)."""
    scr = Path(scr) if scr else SCR
    scr.mkdir(parents=True, exist_ok=True)
    out = scr / f"child{tag}_{seed}_{person.replace(' ', '_')}_{nights}.pkl"
    if cache and out.exists():
        return pickle.loads(out.read_bytes())
    reader.use_cast(FAMILY_CAST)
    ch = Childhood(seed, person, nights, learner_cfg)
    cseed = seed + 5000 + 17 * ch.idx
    carry, t0 = None, time.perf_counter()
    for n in range(nights):
        age = ch.age(n)
        w = World(seed=night_seed(cseed, n), flags=tuple(FAMILY_FLAGS_SAFE) + (WFLAGS if wflags is None else tuple(wflags)), carry=carry, night=n, turns=300, young={person: age})
        ch.night_start(w, n)
        for _ in range(300):
            w.step(ch)
            if teach:
                ch.teach(w)
        ch.stats["own_drinks"].append(sum(1 for e in w.log if e["done"] and e["action"] == "drink" and e["actor"] == person))
        ch.stats["given"] += sum(1 for e in w.log if e["done"] and e["action"] == "serve" and e.get("target") == person)
        ch.night_end(w, n)
        carry = w.carry_out(NAMES)
    ag = ch.mind.agents[person]
    v = reader.VI["drink"]
    res = dict(blob=ag.dump(), nights=nights, seed=seed, person=person, seconds=round(time.perf_counter() - t0, 1),
               stats={**ch.stats, "drink_cnt": float(getattr(ag.wm, "cnt", np.zeros(v + 1))[v]), "drink_n0": float(getattr(ag.wm, "n0", np.zeros(v + 1))[v]), "watched": int(getattr(ag, "watched", 0)),
                                                "decisions": int(ag.decisions)})
    out.write_bytes(pickle.dumps(res))
    return res


if __name__ == "__main__":
    r = run_childhood(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]))
    print({k: v for k, v in r["stats"].items() if k != "own_drinks"}, "seconds", r["seconds"])
