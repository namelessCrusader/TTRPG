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
from lab.mindL.curiosity import Curiosity
from lab.mindL.episodic import Episodic
from lab.mindL.habit import Habit
from lab.mindL.partners import HARM, KIND, PartnerModels, features
from lab.mindL.valence import Valence
from lab.mindL.world_model import WorldModel
from lab.world2.world import DRIVES

# Settings: chosen from short runs on SEED 0 (40 nights; Tom, Oren, Fenn; the
# grids are in the L0 report). The test runs use other seeds.
DEFAULT = dict(modules="AKVWPEC", tau=0.01, gamma=0.98,
               lr_actor=0.01, lr_critic=0.1, lr_val=0.2, lr_habit=0.02, habit_fade=0.98, lr_wm=0.2,
               lr_partner=0.3, k_episodes=5, cap=200, replay=0.3, replay_scale=0.5, hab=30.0,
               b_habit=0.05, b_val=0.5, b_wm=1.0, b_partner=0.5, b_ep=0.5, b_cur=0.5,
               # the 2026-10-29 fixes, all off here so the first round can be redone (see FIXES)
               b_trust=0.0, pm_forget=0.0, pm_detect=False)
# THE FIXES (user, 2026-10-29), "mind2" in lab/mindL/experiments.py: the partner models weigh in the
# choice (the one-step lookahead gets four times the weight, and the person's recent treatment of me
# is part of its valence: kind acts toward one who treats me well pull more, harsh acts toward one
# who treats me well pull less), and they follow change (recency weighting and change detection,
# lab/mindL/partners.py). The world side is the flags reprisal and wear (flags "all2").
FIXES = dict(b_partner=2.0, b_trust=1.0, pm_forget=0.02, pm_detect=True)
# the acts toward a person that open a partner-model window
SOCIAL = {"give", "praise", "insult", "threaten", "strike", "shove", "search", "accuse", "tell", "gossip", "ask",
          "offer", "accept", "refuse", "help up", "bandage", "defend", "steal"}


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class Step:
    __slots__ = ("menu", "a", "pi", "E", "need", "s", "t", "act", "cue", "room", "opt")

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
        need = H.needs(w, p)
        E, s = self.err_fn(w, p), H.state(w, p, need)
        self.observe(w, evs)
        if self.step is not None:
            self.learn(w, p, evs, need, E, s, terminal=False)
        menu = reader.encode(opts, need)
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
            need = H.needs(w, p)
            self.learn(w, p, evs, need, self.err_fn(w, p), H.state(w, p, need), terminal=True)
            self.step = None

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
        self.ac = ActorCritic(reader.NV, reader.ZDIM, reader.NT, H.SDIM, c["lr_actor"], c["lr_critic"], c["gamma"])
        self.habit = Habit(c["lr_habit"], c["habit_fade"])
        self.val = Valence(c["lr_val"])
        self.wm = WorldModel(reader.NV, len(DRIVES), reader.ZDIM, c["lr_wm"])
        self.pm = PartnerModels(c["lr_partner"], forget=c["pm_forget"], detect=c["pm_detect"])
        self.ep = Episodic(c["k_episodes"], c["cap"])
        self.cur = Curiosity(c["hab"])
        self.last = None             # the last choice's terms, for inspection

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
            T["world"] = c["b_wm"] * self.wm.lookahead(menu, H.grad(p.kind, need))
        if "P" in m:
            vk, vh = self.val.value("got_kind"), self.val.value("got_harm")
            x = np.zeros(n)
            for i, o in enumerate(opts):
                if o["verb"] in SOCIAL and o["target"] in reader.PEOPLE:
                    pk, ph = self.pm.predict(o["target"], o["verb"], features(o["f"]))
                    x[i] = pk * vk + ph * vh
            T["partner"] = c["b_partner"] * x
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
        self.habit.update(p.room, [reader.key(x) for x in opts], act)
        self.cur.visit(act)
        if o["verb"] in SOCIAL and o["target"] in reader.PEOPLE:
            self.pm.probe(o["target"], o["verb"], features(o["f"]), w.t)

    # ------------------------------------------------------------ learning
    def observe(self, w, evs):
        self.pm.observe(evs)
        for partner, err in self.pm.close_due(w.t):
            self.cur.p.observe(partner, err)

    def learn(self, w, p, evs, need, E, s, terminal):
        st = self.step
        r = H.reward(st.E, E)
        dt = max(1, w.t - st.t)
        # V: the cues present over the step
        cues = {"ctx": 1.0}
        for e in evs:
            cues[e["actor"]] = 1.0
        if any(e["action"] in KIND for e in evs):
            cues["got_kind"] = 1.0
        if any(e["action"] in HARM for e in evs):
            cues["got_harm"] = 1.0
        for cue in (st.cue, st.room):
            if cue is not None and cue not in cues:
                cues[cue] = 0.5
        self.val.update(cues, r)
        # W: what the act did to my needs
        v, z = st.menu.v[st.a], st.menu.Z[st.a]
        err = self.wm.update(v, z, need - st.need)
        self.cur.w.observe(st.opt["verb"], err)
        # A: critic and actor on the TD error
        delta = self.ac.delta(r, st.s, s, dt, terminal)
        self.ac.critic_update(st.s, delta)
        self.ac.actor_update(st.menu, st.a, st.pi, delta)
        # E: one shot, and tonight's steps for replay
        self.ep.store(st.act, z, delta)
        self.ep.tonight.append((st.menu, st.a, st.pi, st.s, s, r, dt, terminal, st.room, st.act))

    def night_end(self, w, p):
        self.finish(w, p)
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
        return a

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
