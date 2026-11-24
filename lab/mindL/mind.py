"""The learned mind as a lab/world2 policy (learned-mind plan, section 4).

One mind for each character, made the first time that character chooses.
Each module has its own rule and speed (see each file):

    H  drives.py        the reward: the drop in weighted drive error (written)
    V  valence.py       Rescorla-Wagner on the people, places, things present
    A  actor_critic.py  actor-critic on the TD error, over the menu features
    K  habit.py         value-free: grows with repeats, fades with disuse
    W  world_model.py   one-step prediction of the needs' change, per verb
    P  partners.py      logistic regression per partner and act kind
    E  episodic.py      one-shot store, nearest-neighbour lookup, night replay
    C  curiosity.py     learning progress of W and P, with habituation

THE CHOICE: a softmax, at temperature tau, over the menu of
    actor score + b_habit * habit + b_val * valence of the person, room or
    thing + b_wm * W lookahead (predicted drop of drive error) + b_partner *
    P lookahead (p_kind * value of a kind answer + p_harm * value of a harsh
    one, both from V) + b_ep * episodic lookup + b_cur * curiosity bonus.
The mind draws the act with its own random generator and hands the world a
one-hot draw (as the scripted partners do), so the world's own draws are
unchanged in number. `modules` (a string of letters from "AKVWPEC") switches
terms off for the ablations; V's values are still learned when V is off,
because P's lookahead reads the worth of an answer from them.

A STEP runs from one choice to the next (or to death, or to the night's
end): reward r = E(before) - E(after), dt turns. At its close: P learns from
the windows that ended, V from the cues present, W from the needs' change, the
critic and actor from delta = r + gamma^dt V(s') - V(s), and E stores the
step. At the night's end: the open step closes (terminal), E replays the
night's steps with the largest |delta| into A and K, habits fade overnight.

What the mind sees: its own drives, the menu and its numbers, and the events
it PERCEIVED (w.log filtered by p.perceived). That is free of hidden facts
only with the world flag `known` (on in flags "all"): without it the menu's
numbers read the others' private anger (`danger`), unseen rooms (`goal`) and
the chest (`first`), and danger also enters the critic's state. The test
test_mind_inputs_hold_no_hidden_fact checks the state, the menu, PPO's rows and
the reward against changes of unperceived facts. The reward's wealth need is
what the character believes it holds (a loss counts once it is noticed, at its
next choice), also under `known`.
"""
from __future__ import annotations

import zlib

import numpy as np

from lab.mindL import drives as H
from lab.mindL import reader
from lab.mindL.actor_critic import ActorCritic
from lab.mindL.concerns import RWConcerns, StepObs, need_of
from lab.mindL.concerns2 import CARE_VERBS, KinConcerns, StepObs2
from lab.mindL.curiosity import Curiosity
from lab.mindL.effects import Did, DeltaMeanings, Flush, Got, State, Tick, error_drop
from lab.mindL.episodic import Episodic
from lab.mindL.habit import Habit
from lab.mindL.partners import HARM, KIND, PartnerModels, features
from lab.mindL.valence import Valence
from lab.mindL.drives2 import DEFAULT_LETTERS, IE, IT, ND2, Drives2, exponent_dials, watch_weight
from lab.mindL.world_model import WorldModel, WorldModelP
from lab.world2.world import DRIVES, DUR, ROOMS

# Settings: chosen from short runs on SEED 0 (40 nights; Tom, Oren, Fenn; the
# grids are in the L0 report). The test runs use other seeds.
DEFAULT = dict(modules="AKVWPEC", tau=0.01, gamma=0.98,
               lr_actor=0.01, lr_critic=0.1, lr_val=0.2, lr_habit=0.02, habit_fade=0.98, lr_wm=0.2,
               lr_partner=0.3, k_episodes=5, cap=200, replay=0.3, replay_scale=0.5, hab=30.0,
               b_habit=0.05, b_val=0.5, b_wm=1.0, b_partner=0.5, b_ep=0.5, b_cur=0.5,
               # the 2026-10-29 fixes, all off here so the first round can be redone (see FIXES)
               b_trust=0.0, pm_forget=0.0, pm_detect=False,
               # TRACES (2026-11-20, off so every older run reproduces): true online TD(lambda) with dutch
               # traces on the critic, an eligibility trace on the actor's U[verb, target], and the
               # critic's value carried across the night's end (V(s') is not zeroed while the life goes on).
               # lam is per turn, one global value, fixed before any run: (gamma*lam)^30 = 0.22, so the
               # trace still holds a fifth of its weight at the end of P's 30-turn window.
               traces=False, lam=0.97,
               # ACT MEANINGS FROM EFFECTS (2026-11-21, SV3; off so every older run reproduces): the partner term reads
               # what an act would DO to my needs (lab/mindL/effects.py) instead of the KIND and HARM labels. fx_prior:
               # cultural beliefs {act: {"delta": {drive: change}, "n0": 3, "p": 0.3}}; fx_log keeps the credit log.
               effects=False, fx_prior=None, fx_log=False,
               # fx_fix (step 3, off): the drift of the needs is a per-turn rate (turn-weighted), see effects.DeltaMeanings
               fx_fix=False,
               # ACQUIRED CONCERNS (2026-11-21, SV1/SV2, learned-mind-target 9.2; off so every older run reproduces): the mind
               # starts with none; a person or thing it perceives can become a concern, with a weight learned from contact that
               # paid (conditioned reinforcement) and cost paid on its behalf (effort justification); the reward adds weight x the
               # drop of the target's need as perceived through signs. lab/mindL/concerns.py. cn: RWConcerns settings;
               # cn_things: held and loose things (value >= 1) are targets too; b_concern: its weight in the choice (1: the
               # same units as the world-model lookahead, a predicted drop of error).
               concerns=False, cn=None, cn_things=True, b_concern=1.0, cn_fx_prior=None,
               # DRIVES2 (mindL6 stage A, lab/mindL/drives2.py; off so every older run reproduces): True = "SCFD"; letters S separate thirst, C convex drive
               # (Keramati and Gutkin), F felt depletion, D need by destination, W learning by watching. d2: Drives2 settings (n, m, felt_gain).
               # wm_prior: {verb: dict(const={drive: x}, slope={drive: x}, n0=3)}, a prior on the world model of my own acts (what being TOLD writes).
               # w_obs: the weight of a watched act against my own experience (u), copy_kin and copy_succ: dials that raise it for kin and for the
               # successful (u, 0). b_dest: weight of the destination's expected relief in a walk option's score (u, 1: the units of the lookahead).
               drives2=False, d2=None, wm_prior=None, w_obs=0.3, copy_kin=0.0, copy_succ=0.0, b_dest=1.0,
               # B1 (mindL6 stage B1; all off so every older run reproduces): d2={"dial": True} gives each person its own exponents n > m > 1 (drives2.exponent_dials);
               # the letter B of drives2 is the staged felt ladder; watch_age: the weight of a watched act follows the watcher's AGE (drives2.watch_weight: 0 at birth,
               # the child's outcome weight, falling to the adult's) instead of w_obs; taste: the first valence of what is consumed is the world's signed taste (sweet good,
               # bitter bad) times taste_gain (u), a prior that experience overrides at the valence's own rate.
               watch_age=False, taste=False, taste_gain=0.3)
# THE FIXES (user, 2026-10-29), "mind2" in lab/mindL/experiments.py: the partner models weigh in the
# choice (the one-step lookahead gets four times the weight, and the person's recent treatment of me
# is part of its valence: kind acts toward one who treats me well pull more, harsh acts toward one
# who treats me well pull less), and they follow change (recency weighting and change detection,
# lab/mindL/partners.py). The world side is the flags reprisal and wear (flags "all2").
FIXES = dict(b_partner=2.0, b_trust=1.0, pm_forget=0.02, pm_detect=True)
# the acts toward a person that open a partner-model window
SOCIAL = {"give", "praise", "insult", "threaten", "strike", "shove", "search", "accuse", "tell", "gossip", "ask",
          "offer", "accept", "refuse", "help up", "bandage", "defend", "steal"}


def c_taste(cfg):
    return bool(cfg.get("taste"))


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class Step:
    __slots__ = ("menu", "a", "pi", "E", "need", "s", "t", "act", "cue", "room", "opt", "cn")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class Agent:
    """One character's chooser: keeps the open step, reads what it perceived,
    and learns when a step closes. Subclasses: Mind, nulls.Bandit."""

    def __init__(self, name, kind, seed=0):
        self.name, self.kind = name, kind
        self.rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
        self.step = None
        self._w, self._ptr = None, 0
        self.decisions = 0
        self.D = H                   # the drive model: the module drives (old) or a drives2.Drives2
        self.err_fn = H.error        # the error whose drop is the reward (L4 swaps in a designer's)

    def _sync(self, w):
        if w is not self._w:
            if self.step is not None and self._w is not None:
                self.finish(self._w, self._w.people[self.name])
            self._w, self._ptr = w, 0

    def perceived_at_me(self, w, p):
        """Acts by others aimed at me, done, that I perceived, since last read."""
        new = w.log[self._ptr:]
        self._ptr = len(w.log)
        return [e for e in new if e.get("target") == p.name and e["actor"] != p.name and e["done"]
                and e["id"] in p.perceived]

    def decide(self, w, p, opts):
        self._sync(w)
        evs = self.perceived_at_me(w, p)
        need = self.D.needs(w, p)
        E, s = self.err_fn(w, p), self.D.state(w, p, need)
        self.observe(w, evs)
        if self.step is not None:
            self.learn(w, p, evs, need, E, s, terminal=False)
        menu = self.make_menu(w, p, opts, need)
        a, pi = self.pick(w, p, opts, menu, need, s)
        o = opts[a]
        self.step = Step(menu=menu, a=a, pi=pi, E=E, need=need, s=s, t=w.t, act=reader.key(o),
                         cue=reader.cue(o), room=p.room, opt=o)
        self.decisions += 1
        self.after_pick(w, p, opts, menu, a)
        return a

    def finish(self, w, p):
        """Close the open step as terminal (death, or the night's end)."""
        self._sync(w)
        evs = self.perceived_at_me(w, p)
        self.observe(w, evs)
        if self.step is not None:
            need = self.D.needs(w, p)
            self.learn(w, p, evs, need, self.err_fn(w, p), self.D.state(w, p, need), terminal=self.ends_life(p))
            self.step = None

    def make_menu(self, w, p, opts, need):
        return reader.encode(opts, need)

    def ends_life(self, p):
        """Is the step closed by finish() the last one of this life? Always, for a chooser that
        treats a night's end like a death (the first rounds); a Mind with traces only if dead."""
        return True

    def night_end(self, w, p):
        self.finish(w, p)

    # the parts a chooser fills in
    def observe(self, w, evs):
        pass

    def learn(self, w, p, evs, need, E, s, terminal):
        raise NotImplementedError

    def pick(self, w, p, opts, menu, need, s):
        raise NotImplementedError

    def after_pick(self, w, p, opts, menu, a):
        pass


class Mind(Agent):
    def __init__(self, name, kind, seed=0, **cfg):
        super().__init__(name, kind, seed)
        bad = set(cfg) - set(DEFAULT)
        assert not bad, f"unknown settings {sorted(bad)}"
        c = self.cfg = {**DEFAULT, **cfg}
        self.mods = set(c["modules"])
        self.letters = (DEFAULT_LETTERS if c["drives2"] is True else (c["drives2"] or ""))
        self.use_dest, self.use_watch = "D" in self.letters, "W" in self.letters
        if self.letters:
            d2 = dict(c["d2"] or {})
            if d2.pop("dial", False):
                d2["n"], d2["m"] = exponent_dials(seed, name)
            self.D = Drives2(self.letters, **d2)
            self.err_fn = self.D.error
            zdim = 1 + reader.NK + self.D.ND + int(self.use_dest)
        else:
            assert not c["wm_prior"], "wm_prior needs drives2"
            zdim = reader.ZDIM
        self.zdim = zdim
        self.gamma_walk = c["gamma"] ** DUR["walk to"]      # (kept: older callers; the dest term uses each option's own duration)
        self.ac = ActorCritic(reader.NV, zdim, reader.NT, self.D.SDIM, c["lr_actor"], c["lr_critic"], c["gamma"])
        self.habit = Habit(c["lr_habit"], c["habit_fade"])
        self.val = Valence(c["lr_val"])
        self.wm = (WorldModelP if self.letters else WorldModel)(reader.NV, len(self.D.DRIVES), zdim, c["lr_wm"])
        for verb, b in (c["wm_prior"] or {}).items():
            dix = {d: i for i, d in enumerate(self.D.DRIVES)}
            self.wm.set_prior(reader.VI[verb], {dix[d]: x for d, x in b.get("const", {}).items()}, {dix[d]: x for d, x in b.get("slope", {}).items()},
                              b.get("n0", 3.0), reader.NK)
        self._aff: dict = {}         # drives2 D: room -> {act: (verb, z, strongest need then)}: what I have seen on offer there
        self._ww, self._wptr, self._sign0, self.watched = None, 0, {}, 0
        self.pm = PartnerModels(c["lr_partner"], forget=c["pm_forget"], detect=c["pm_detect"])
        self.ep = Episodic(c["k_episodes"], c["cap"])
        self.cur = Curiosity(c["hab"])
        self.last = None             # the last choice's terms, for inspection
        self.fx = None
        if c["effects"]:
            assert not c["b_trust"], "the effects arm uses no KIND/HARM labels: b_trust must be 0"
            self.fx = DeltaMeanings(self.D.ND, nx=len(features({})), keep_log=c["fx_log"], turn_drift=c["fx_fix"])
            self.fx.cap = self.D.cap if self.letters else 1.0
            for act, b in (c["fx_prior"] or {}).items():
                self.fx.set_prior(act, [b["delta"].get(d, 0.0) for d in self.D.DRIVES], b.get("n0", 3), b.get("p"))
        self.ac.lam = c["lam"] if c["traces"] else 0.0
        self.cn = None
        self._rd: dict = {}
        self._got: frozenset = frozenset()
        self._cw, self._cptr = None, 0
        if c["concerns"] == "cn2":           # mindL5 stage A: the second constants and the kin rule (concerns2.py)
            self.cn = KinConcerns(name=name, seed=seed, **(c["cn"] or {}))
        elif c["concerns"]:
            self.cn = RWConcerns(**(c["cn"] or {}))
        if self.cn is not None:
            for verb, b in (c["cn_fx_prior"] or {}).items():
                self.cn.set_prior(verb, b.get("rho", 0.0), b.get("eta", 0.0), b.get("n0", 3))

    def ends_life(self, p):
        return bool(p.dead) or not self.cfg["traces"]

    # ------------------------------------------------------------ drives2: need by destination (D)
    @staticmethod
    def moves_me(o, room):
        """An option moves me by its EFFECT: its target is a place other than the one I am in (not by the name of its verb: walk to, flee to, or any other)."""
        return o["target"] in ROOMS and o["target"] != room

    def make_menu(self, w, p, opts, need):
        if not self.use_dest:
            return reader.encode(opts, need)
        menu = reader.encode(opts, need, extra=1)
        raw = self.D.raw_needs(w, p)
        self._remember(p.room, opts, menu, need, raw)
        rel: dict = {}
        for i, o in enumerate(opts):
            if self.moves_me(o, p.room):
                if o["target"] not in rel:
                    rel[o["target"]] = self.relief_at(o["target"], p.kind, need, raw)
                menu.Z[i, -1] = rel[o["target"]]
        return menu

    def _remember(self, room, opts, menu, need, raw=None):
        """What I see on offer here, kept for what it could do for me later (not movement: a place is what can be DONE in it). Each entry keeps the option's own
        need number then and the column of MY needs it is about (the need it equals: an eat is about hunger, a drink about thirst, any act the same way), so that
        the number can be read again for how needy I am now in THAT need."""
        if room not in ROOMS:
            return
        mem = self._aff.setdefault(room, {})
        raw = need if raw is None else raw
        k = reader.KI["need"]
        for i, o in enumerate(opts):
            if not self.moves_me(o, room):
                own, col = float(menu.Z[i, k]), None
                if own > 1e-9:
                    diff = np.abs(np.asarray(raw) - own)
                    j = int(np.argmin(diff))
                    col = j if diff[j] < 1e-6 else None
                mem[reader.key(o)] = (int(menu.v[i]), menu.Z[i].copy(), own, col)

    def relief_at(self, room, kind, need, raw=None):
        """The relief my OWN world model expects from the best act I know is on offer in `room`, for my CURRENT needs: the drop of the drive
        (a one-step lookahead over known affordances, every need, no rule about thirst). A remembered option's numbers are read again for now:
        the needs columns are my needs now, and its own `need` number scales with how needy I am now against then in the need it is ABOUT (its own need column;
        u: a general stand-in for recomputing the option's numbers, which only the world can do). An entry with no such column is not rescaled."""
        mem = self._aff.get(room)
        if not mem:
            return 0.0
        raw = need if raw is None else raw
        ents = list(mem.values())
        vs = np.array([x[0] for x in ents])
        Z = np.array([x[1] for x in ents])
        c0 = 1 + reader.NK
        nd = self.D.ND
        Z[:, c0:c0 + nd] = need
        Z[:, -1] = 0.0
        k = reader.KI["need"]
        scale = np.ones(len(ents))
        for r, x in enumerate(ents):
            col = x[3] if len(x) > 3 else None
            if col is not None:
                scale[r] = np.clip(float(raw[col]) / max(float(x[2]), 0.1), 0.0, 4.0)
        Z[:, k] *= scale
        pred = np.einsum("ndz,nz->nd", self.wm.M[vs], Z)
        return float(max(0.0, self.D.drop(kind, need, pred).max()))

    # ------------------------------------------------------------ drives2: learning by watching (W)
    def _watch(self, w, p):
        """Another person's act that I perceived, and the change in their signs (hunger, thirst, hurt or fear) between my last reading of them and
        now, teach my world model for that act at weight w_obs (u) of an own experience, on the needs I can see only. Only an act that is
        the only one of that person in the interval counts (a clean credit); a reading is as noisy as the world's signs. Lossy by design."""
        c, me = self.cfg, self.name
        if w is not self._ww:
            self._ww, self._wptr, self._sign0 = w, 0, {}
        new = w.log[self._wptr:]
        self._wptr = len(w.log)
        now = {}
        for q in w.here(p):
            if q.name != me:
                sg = w.perceived_need(me, q.name)
                if sg is not None:
                    now[q.name] = sg
        by: dict = {}
        for e in new:
            if e["done"] and e["actor"] != me and e["actor"] in reader.PEOPLE and e["id"] in p.perceived and e["action"] in reader.VI:
                by.setdefault(e["actor"], []).append(e)
        for X, es in by.items():
            if len(es) != 1 or X not in now or X not in self._sign0:
                continue
            t0, s0 = self._sign0[X]
            if es[0]["t"] >= t0 and w.t - t0 <= 40:
                self._learn_watched(es[0], s0, now[X], X, age=getattr(p, "age", 30.0) if c["watch_age"] else None)
        self._sign0.update({X: (w.t, sg) for X, sg in now.items()})

    def _learn_watched(self, e, s0, s1, X, age=None):
        nd, c0 = self.D.ND, 1 + reader.NK
        dv, mask, z = np.zeros(nd), np.zeros(nd), np.zeros(self.zdim)
        z[0] = 1.0
        for d, key in ((IE, "hunger"), (IT, "thirst")):
            if key in s0 and key in s1:
                dv[d], mask[d], z[c0 + d] = s1[key] - s0[key], 1.0, s0[key]
        sf0, sf1 = max(s0["hurt"], s0["fear"]), max(s1["hurt"], s1["fear"])
        dv[1], mask[1], z[c0 + 1] = sf1 - sf0, 1.0, sf0
        c = self.cfg
        wt = c["w_obs"] if age is None else watch_weight(age)
        if wt <= 0.0:
            return
        if c["copy_kin"] and self.cn is not None and X in getattr(self.cn, "rows", {}) and hasattr(self.cn, "khat"):
            wt *= 1.0 + c["copy_kin"] * self.cn.khat(self.cn.rows[X])
        if c["copy_succ"]:
            wt *= 1.0 + c["copy_succ"] * max(0.0, 1.0 - 0.5 * (s1.get("hunger", 0.0) + s1.get("thirst", 0.0)))
        self.wm.update(reader.VI[e["action"]], z, dv, scale=min(1.0, wt), mask=mask)
        self.watched += 1

    # ------------------------------------------------------------ save and load (the whole learner)
    PARTS = ("ac", "habit", "val", "wm", "pm", "ep", "cur", "fx", "cn", "_aff", "_sign0", "decisions", "watched", "_rd", "_got")

    def dump(self):
        """The whole learner as bytes (actor, critic, world model with its priors, effects, concerns, valence, habit, partner models, memory, curiosity,
        the generator): a primed childhood can be cached per seed and loaded into an adult. Call between nights (no open step)."""
        import pickle
        assert self.step is None, "save between nights: the open step is not kept"
        return pickle.dumps({"name": self.name, "kind": self.kind, "parts": {k: getattr(self, k) for k in self.PARTS},
                             "rng": self.rng.bit_generator.state})

    def load(self, blob):
        import pickle
        d = pickle.loads(blob)
        assert d["name"] == self.name and d["kind"] == self.kind, "a learner is loaded into the same person"
        for k, v in d["parts"].items():
            setattr(self, k, v)
        self.rng.bit_generator.state = d["rng"]
        self.step, self._w, self._ptr, self._ww, self._wptr = None, None, 0, None, 0

    # ------------------------------------------------------------ choosing
    def terms(self, w, p, opts, menu, need):
        """Each term of the score, per option (a dict of arrays)."""
        c, m, n = self.cfg, self.mods, len(opts)
        acts = [reader.key(o) for o in opts]
        T = {}
        T["actor"] = self.ac.scores(menu) if "A" in m else np.zeros(n)
        if "K" in m:
            T["habit"] = c["b_habit"] * np.array([self.habit.strength(p.room, a) for a in acts])
        if "V" in m:
            T["valence"] = c["b_val"] * np.array([self.val.value(reader.cue(o)) for o in opts])
            if c["b_trust"]:
                # the person's recent treatment of me (partner models, recency weighted) is part of how
                # good the person is: kind acts toward one who treats me well pull more, harsh acts less
                vk, vh = self.val.value("got_kind"), self.val.value("got_harm")
                tr = np.zeros(n)
                for i, o in enumerate(opts):
                    if o["target"] in reader.PEOPLE and (o["verb"] in KIND or o["verb"] in HARM):
                        t = self.pm.trust(o["target"], vk, vh)
                        tr[i] = t if o["verb"] in KIND else -max(0.0, t)
                T["valence"] = T["valence"] + c["b_trust"] * tr
        if "W" in m:
            if self.letters:                       # drives2: the exact predicted drop of the drive (D is convex: a gradient overstates a big relief)
                T["world"] = c["b_wm"] * self.D.drop(p.kind, need, self.wm.predict(menu))
            else:
                T["world"] = c["b_wm"] * self.wm.lookahead(menu, H.grad(p.kind, need))
        if self.use_dest:
            T["dest"] = c["b_dest"] * np.array([c["gamma"] ** o.get("dur", DUR.get(o["verb"], 1)) * menu.Z[i, -1] if self.moves_me(o, p.room) else 0.0
                                                for i, o in enumerate(opts)])
        if "P" in m and self.fx is not None:
            T["partner"] = c["b_partner"] * self.fx_term(opts, need, p.kind)
        elif "P" in m:
            vk, vh = self.val.value("got_kind"), self.val.value("got_harm")
            x = np.zeros(n)
            for i, o in enumerate(opts):
                if o["verb"] in SOCIAL and o["target"] in reader.PEOPLE:
                    pk, ph = self.pm.predict(o["target"], o["verb"], features(o["f"]))
                    x[i] = pk * vk + ph * vh
            T["partner"] = c["b_partner"] * x
        if self.cn is not None:
            T["concern"] = c["b_concern"] * np.array([
                self.cn.expected(o["verb"], o["target"], self._rd.get(o["target"], 0.0)) if o["target"] in self._rd else 0.0
                for o in opts])
        if "E" in m:
            x = np.zeros(n)
            groups: dict = {}
            for i, a in enumerate(acts):
                groups.setdefault(a, []).append(i)
            for a, idx in groups.items():
                x[idx] = self.ep.lookup(a, menu.Z[idx])
            T["episodic"] = c["b_ep"] * x
        if "C" in m:
            T["curiosity"] = c["b_cur"] * np.array([
                self.cur.bonus(o["verb"], o["target"] if o["target"] in reader.PEOPLE else None, a)
                for o, a in zip(opts, acts)])
        return T

    def fx_term(self, opts, need, kind):
        """The partner term from effects: for an act at a person, the sum over their likely answers of (chance of the answer
        times the drop of my weighted drive error that the answer's effect would bring from the state I am in)."""
        x = np.zeros(len(opts))
        worth: dict = {}
        for i, o in enumerate(opts):
            who = o["target"]
            if who not in reader.PEOPLE:
                continue
            for a, pa in self.fx.predict(who, o["verb"], State(features(o["f"]), need)).items():
                k = (who, a)
                if k not in worth:
                    worth[k] = error_drop(kind, need, self.fx.effect(who, a, State(None, need)), self.D if self.letters else None)
                x[i] += pa * worth[k]
        return x

    def pick(self, w, p, opts, menu, need, s):
        T = self.terms(w, p, opts, menu, need)
        score = sum(T.values())
        pi = softmax(score / self.cfg["tau"])
        a = int(self.rng.choice(len(opts), p=pi))
        self.last = {"terms": T, "pi": pi, "a": a}
        return a, pi

    def after_pick(self, w, p, opts, menu, a):
        o = opts[a]
        act = reader.key(o)
        if self.cn is not None:
            self.step.cn = dict(self._rd)
            if o["target"] in reader.PEOPLE:
                self.cn.did(o["target"], o["verb"], w.t, self._rd.get(o["target"]))
        self.habit.update(p.room, [reader.key(x) for x in opts], act)
        self.cur.visit(act)
        if self.fx is not None:
            if o["target"] in reader.PEOPLE:
                self.fx.update(Did(o["target"], o["verb"], features(o["f"]), w.t))
        elif o["verb"] in SOCIAL and o["target"] in reader.PEOPLE:
            self.pm.probe(o["target"], o["verb"], features(o["f"]), w.t)

    # ------------------------------------------------------------ learning
    def _readings(self, w, p):
        """What I perceive of the needs of those I could care for: people in my room by their signs, things I hold (need 0) or
        that lie loose in my room (need 1)."""
        rd = {}
        for q in w.here(p):
            sg = w.perceived_need(self.name, q.name)
            if sg is not None:
                rd[q.name] = need_of(sg)
        if self.cfg["cn_things"] and not p.dead:
            for n in w.held(p):
                if w.things[n]["value"] >= 1:
                    rd["thing:" + n] = 0.0
            for n in w.loose(p.room):
                if w.things[n]["value"] >= 1 and not w.things[n]["hidden"]:
                    rd["thing:" + n] = 1.0
        return rd

    def observe(self, w, evs):
        if self.use_watch:
            self._watch(w, w.people[self.name])
        if self.cn is not None:
            self._rd = self._readings(w, w.people[self.name])
            self.cn.observe_read(w.t, self._rd)
            if isinstance(self.cn, KinConcerns):             # cn2: the care I saw others give (the observed-care cue)
                if w is not self._cw:
                    self._cw, self._cptr = w, 0
                me = w.people[self.name]
                for e in w.log[self._cptr:]:
                    if (e["done"] and e["action"] in CARE_VERBS and e.get("target") and e["actor"] != self.name and e["target"] != self.name
                            and e["id"] in me.perceived and e["target"] in reader.PEOPLE):
                        self.cn.observe_care(e["actor"], e["target"], e["t"])
                self._cptr = len(w.log)
        if self.fx is not None:
            need = self.D.needs(w, w.people[self.name])
            tref, need0 = (self.step.t, self.step.need) if self.step is not None else (w.t, need)
            for e in evs:
                self.fx.update(Got(e["actor"], e["action"], e["t"], tref, need0))
            for partner, err in self.fx.update(Tick(w.t, need)):
                self.cur.p.observe(partner, err)
            return
        self.pm.observe(evs)
        for partner, err in self.pm.close_due(w.t):
            self.cur.p.observe(partner, err)

    def learn(self, w, p, evs, need, E, s, terminal):
        st = self.step
        r = self.D.reward(st.E, E)
        dt = max(1, w.t - st.t)
        if self.cn is not None:
            self._got = frozenset(e["actor"] for e in evs)
            r += self.cn_learn(st, r, dt)
        # V: the cues present over the step
        cues = {"ctx": 1.0}
        for e in evs:
            cues[e["actor"]] = 1.0
        if self.fx is None and any(e["action"] in KIND for e in evs):
            cues["got_kind"] = 1.0
        if self.fx is None and any(e["action"] in HARM for e in evs):
            cues["got_harm"] = 1.0
        for cue in (st.cue, st.room):
            if cue is not None and cue not in cues:
                cues[cue] = 0.5
        self._taste_prior(w, p)
        self.val.update(cues, r)
        # W: what the act did to my needs
        v, z = st.menu.v[st.a], st.menu.Z[st.a]
        err = self.wm.update(v, z, need - st.need)
        self.cur.w.observe(st.opt["verb"], err)
        # A: critic and actor on the TD error
        if self.cfg["traces"]:
            delta = self.ac.trace_step(st.menu, st.a, st.pi, st.s, s, r, dt, terminal, self.cfg["lam"])
            self.ac.actor_update(st.menu, st.a, st.pi, delta, tables="W")
        else:
            delta = self.ac.delta(r, st.s, s, dt, terminal)
            self.ac.critic_update(st.s, delta)
            self.ac.actor_update(st.menu, st.a, st.pi, delta)
        # E: one shot, and tonight's steps for replay
        self.ep.store(st.act, z, delta)
        self.ep.tonight.append((st.menu, st.a, st.pi, st.s, s, r, dt, terminal, st.room, st.act))

    def _taste_prior(self, w, p):
        """A first taste: what was just on my tongue (the world's sense of taste: person.tasted, set when something is consumed, however it came to me) starts with
        the world's signed taste (sweet and savoury good, bitter bad) times taste_gain as its valence, an innate prior that the next updates override at the valence's
        own rate (about 1/lr_val = 5 experiences). It reads the sense, not the name of any act."""
        item = getattr(p, "tasted", None)
        if item is None or not c_taste(self.cfg):
            return
        p.tasted = None
        tf = getattr(w, "taste", None)
        if tf is not None and item not in self.val.v:
            self.val.v[item] = self.cfg["taste_gain"] * tf(item)

    def cn_learn(self, st, r_h, dt):
        """The concerns learn from the step that closes (st.cn: what I read at its start; self._rd: what I read now) and return the
        concern reward of the step. r_h is the homeostatic reward only (a concern never teaches itself)."""
        o = st.opt
        aimed = o["target"] if o["target"] in reader.PEOPLE else None
        helped = 0.0
        if aimed is not None and aimed in st.cn:
            helped = self.cn.relief(o["verb"], st.cn[aimed])
        if isinstance(self.cn, KinConcerns):
            return self.cn.learn(StepObs2(r_h, st.cn, self._rd, tuple(st.cn), o["verb"], aimed, helped, dt, self._got))
        return self.cn.learn(StepObs(r_h, st.cn, self._rd, tuple(st.cn), o["verb"], aimed, helped, dt))

    def night_end(self, w, p):
        self.finish(w, p)
        if self.cn is not None:
            self.cn.flush()
            if isinstance(self.cn, KinConcerns):
                self.cn.set_household([q for q in w.people if q != self.name and q in reader.PEOPLE and q != "the bandit"
                                       and q not in getattr(w, "gone", ())])
                if self.cn.nrs == 0:
                    self.cn.age = float(getattr(p, "age", 30.0))
            self.cn.night()
        if self.fx is not None:
            for partner, err in self.fx.update(Flush(w.t, self.D.needs(w, p))):
                self.cur.p.observe(partner, err)
        for partner, err in self.pm.close_all():
            self.cur.p.observe(partner, err)
        if "E" in self.mods and self.ep.tonight:
            self.replay()
        self.ep.tonight = []
        self.habit.night()

    def replay(self):
        """Sleep: the night's steps with the largest |delta| (now), again into
        the actor and critic, and into habit, at a lower rate."""
        c = self.cfg
        steps = self.ep.tonight
        d = [abs(self.ac.delta(r, s0, s1, dt, term)) for (_, _, _, s0, s1, r, dt, term, _, _) in steps]
        k = max(1, int(round(c["replay"] * len(steps))))
        for j in sorted(range(len(steps)), key=lambda i: -d[i])[:k]:
            menu, a, pi, s0, s1, r, dt, term, room, act = steps[j]
            delta = self.ac.delta(r, s0, s1, dt, term)
            self.ac.critic_update(s0, delta, c["replay_scale"])
            self.ac.actor_update(menu, a, pi, delta, c["replay_scale"])
            if "K" in self.mods:
                self.habit.update(room, [act], act, c["replay_scale"])


class PerCharacter:
    """A lab/world2 policy: one agent per character, made on first choice."""

    def __init__(self, make, name):
        self.make, self.name = make, name
        self.agents: dict = {}

    def agent(self, p):
        a = self.agents.get(p.name)
        if a is None:
            a = self.agents[p.name] = self.make(p.name, p.kind)
            blob = getattr(self, "preload", {}).get(p.name)
            if blob is not None:
                a.load(blob)
        return a

    def save(self):
        """{name: the whole learner as bytes} (Mind only), between nights."""
        return {n: a.dump() for n, a in self.agents.items() if hasattr(a, "dump")}

    def load(self, blobs):
        """Learners to load into each person the first time they choose (or now, if they already exist)."""
        self.preload = dict(blobs)
        for n, a in self.agents.items():
            if n in self.preload:
                a.load(self.preload[n])

    def probs(self, w, p, opts):
        i = self.agent(p).decide(w, p, opts)
        pc = [0.0] * len(opts)
        pc[i] = 1.0
        return pc, pc

    def night_start(self, w, night):
        pass

    def night_end(self, w, night):
        for name, a in self.agents.items():
            if name in w.people:
                a.night_end(w, w.people[name])

    def on_death(self, w, person, event):
        a = self.agents.get(person.name)
        if a is not None:
            a.finish(w, person)


def make_mind(seed=0, **cfg):
    mods = cfg.get("modules", DEFAULT["modules"])
    name = "mind" if mods == DEFAULT["modules"] else f"mind:{mods}"
    return PerCharacter(lambda n, k: Mind(n, k, seed, **cfg), name)
