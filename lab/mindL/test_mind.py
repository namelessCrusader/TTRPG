"""Tests for the learned mind (lab/mindL): one or more per module, then the
mind as a world policy, then the nulls. Plain asserts (the lab venv has no
pytest).

    lab/.venv/bin/python -m lab.mindL.test_mind
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from lab.mindL import drives as H
from lab.mindL import reader
from lab.mindL.actor_critic import ActorCritic
from lab.mindL.curiosity import Curiosity, Progress
from lab.mindL.episodic import Episodic
from lab.mindL.habit import Habit
from lab.mindL.mind import FIXES, Mind, make_mind, softmax
from lab.mindL.nulls import K, make_bandit, make_env, menu_obs, run_ppo
from lab.mindL.partners import PartnerModels, features
from lab.mindL.valence import Valence
from lab.mindL.world_model import WorldModel
from lab.world2.lifetime import Cast, lifetime
from lab.world2.policy import Random, Rules
from lab.world2.world import DRIVES, NATURE, World


# ------------------------------------------------------------ reader
def test_reader_reads_every_option_of_a_night():
    reader.UNKNOWN.clear()
    menus = []

    class Spy(Random):
        def probs(self, w, p, opts):
            m = reader.encode(opts, H.needs(w, p))
            menus.append((opts, m))
            return super().probs(w, p, opts)

    w = World(seed=4, flags="all", bandit=True)
    for _ in range(300):
        w.step(Spy())
    assert len(menus) > 200
    assert not reader.UNKNOWN, reader.UNKNOWN
    for opts, m in menus[:300]:
        assert m.Z.shape == (len(opts), reader.ZDIM) and np.isfinite(m.Z).all()
        for o, v, t in zip(opts, m.v, m.t):
            assert reader.VERBS[v] == o["verb"]
            assert reader.TARGETS[t] == reader.target_name(o["target"])
        X = reader.raw(m)
        assert X.shape == (len(opts), reader.RAW)
        assert (X[:, :reader.NV].sum(1) == 1).all() and (X[:, reader.NV:reader.NV + reader.NT].sum(1) == 1).all()


# ------------------------------------------------------------ H
def test_drives_reward_is_the_drop_of_weighted_error():
    w = World(seed=0, flags="all")
    p = w.people["Oren"]
    need = H.needs(w, p)
    assert abs(H.errors(p.kind, need).sum() - w.drive_error(p)) < 1e-9
    e0 = w.drive_error(p)
    p.hunger, p.thirst = 0.9, 0.9
    e1 = w.drive_error(p)
    assert e1 > e0 and abs(H.reward(e0, e1) - (e0 - e1)) < 1e-12 and H.reward(e1, e0) > 0
    g = H.grad(p.kind, H.needs(w, p))
    for i, d in enumerate(DRIVES):
        wt, sp = NATURE[p.kind][d]
        assert g[i] == (wt if H.needs(w, p)[i] > sp else 0.0)
    assert H.state(w, p).shape == (H.SDIM,)


# ------------------------------------------------------------ V
def test_valence_delta_rule_separates_people_and_absorbs_drift():
    v = Valence(lr=0.2)
    for i in range(300):
        who, r = ("Fenn", -1.0) if i % 2 else ("Oren", 1.0)
        v.update({"ctx": 1.0, who: 1.0}, r - 0.1)          # -0.1: the needs' drift, on every step
    assert v.value("Fenn") < -0.6 and v.value("Oren") > 0.6, v.v
    assert abs(v.value("ctx") + 0.1) < 0.1, v.value("ctx")
    assert v.value("Marta") == 0.0


# ------------------------------------------------------------ A
def test_actor_critic_learns_which_act_pays():
    ac = ActorCritic(reader.NV, reader.ZDIM, reader.NT, 3, lr_actor=0.3, lr_critic=0.2, gamma=0.9)
    rng = np.random.default_rng(0)
    opts = [{"verb": "eat", "target": None, "obj": "the loaf of bread", "f": {"need": 0.8}},
            {"verb": "wait", "target": None, "obj": None, "f": {}}]
    menu = reader.encode(opts, np.zeros(len(DRIVES)))
    s = np.array([1.0, 0.0, 0.0])
    for _ in range(300):
        pi = softmax(ac.scores(menu) / 0.1)
        a = int(rng.choice(2, p=pi))
        r = 0.5 if a == 0 else 0.0
        d = ac.delta(r, s, s, 1, terminal=True)
        ac.critic_update(s, d)
        ac.actor_update(menu, a, pi, d)
    pi = softmax(ac.scores(menu) / 0.1)
    assert pi[0] > 0.9, pi
    assert 0.2 < ac.value(s) < 0.55, ac.value(s)
    # a semi-Markov step: the next state's value is discounted by gamma ** dt, and ignored at the end
    ac.c[:] = [1.0, 0, 0]
    assert abs(ac.delta(0.0, s, s, 2, False) - (0.81 - 1.0)) < 1e-9
    assert abs(ac.delta(0.0, s, s, 2, True) - (-1.0)) < 1e-9


# ------------------------------------------------------------ K
def test_habit_grows_with_repeats_and_fades_with_disuse():
    h = Habit(lr=0.1, fade=0.9)
    acts = [("sit down", None), ("wait", None)]
    for _ in range(30):
        h.update("the main room", acts, acts[0])
    s = h.strength("the main room", acts[0])
    assert s > 0.9 and h.strength("the main room", acts[1]) < 0.05
    assert h.strength("the kitchen", acts[0]) == 0.0           # tied to the place
    for _ in range(10):
        h.update("the main room", acts, acts[1])
    assert h.strength("the main room", acts[0]) < s * 0.4      # passed over: fades
    before = h.strength("the main room", acts[1])
    h.night()
    assert abs(h.strength("the main room", acts[1]) - 0.9 * before) < 1e-9


# ------------------------------------------------------------ W
def test_world_model_learns_what_an_act_does_to_the_needs():
    wm = WorldModel(reader.NV, len(DRIVES), reader.ZDIM, lr=0.3)
    v = reader.VI["eat"]
    rng = np.random.default_rng(1)
    errs = []
    for _ in range(400):
        z = np.zeros(reader.ZDIM)
        z[0] = 1.0
        z[reader.KI["need"]] = rng.uniform(0, 1)
        d = np.zeros(len(DRIVES))
        d[0] = -0.6 * z[reader.KI["need"]]                      # eating lowers energy by more when hungrier
        errs.append(wm.update(v, z, d))
    assert np.mean(errs[-50:]) < 0.1 * np.mean(errs[:10]), (np.mean(errs[:10]), np.mean(errs[-50:]))
    opts = [{"verb": "eat", "target": None, "obj": "x", "f": {"need": 0.9}},
            {"verb": "eat", "target": None, "obj": "x", "f": {"need": 0.1}}]
    menu = reader.encode(opts, np.zeros(len(DRIVES)))
    g = np.zeros(len(DRIVES))
    g[0] = 1.0                                                  # energy above its set point
    la = wm.lookahead(menu, g)
    assert la[0] > la[1] > 0, la


# ------------------------------------------------------------ P
def test_partner_models_keep_each_person_apart():
    pm = PartnerModels(lr=0.3, window=30)
    x = features({"like": 0.2})
    t = 0
    for i in range(40):
        pm.probe("Fenn", "give", x, t)
        pm.probe("Oren", "give", x, t)
        pm.observe([{"actor": "Fenn", "action": "give", "t": t + 5}])        # Fenn pays back
        pm.observe([{"actor": "Oren", "action": "strike", "t": t + 6}])      # Oren hits
        t += 31
        pm.close_due(t)
    fk, fh = pm.predict("Fenn", "give", x)
    ok, oh = pm.predict("Oren", "give", x)
    assert fk > 0.85 and fh < 0.15, (fk, fh)
    assert ok < 0.15 and oh > 0.85, (ok, oh)
    assert pm.predict("Fenn", "insult", x) == (0.5, 0.5)                    # no data: no guess
    assert pm.predict("Marta", "give", x) == (0.5, 0.5)
    # a window stays open until it ends; an answer before the act does not count
    pm2 = PartnerModels(window=30)
    pm2.probe("Fenn", "praise", x, 100)
    pm2.observe([{"actor": "Fenn", "action": "praise", "t": 99}])
    assert pm2.close_due(120) == [] and len(pm2.open) == 1
    (who, err), = pm2.close_due(130)
    assert who == "Fenn" and pm2.predict("Fenn", "praise", x)[0] < 0.5


# ------------------------------------------------------------ P: the fixes (2026-10-29)
def _answers(pm, partner, kind, n, t0, verb="give"):
    """n windows on `partner`: my act, then (if `kind`) a kind answer from it."""
    x = features({})
    for i in range(n):
        t = t0 + 100 * i
        pm.probe(partner, verb, x, t)
        if kind:
            pm.observe([{"actor": partner, "t": t + 1, "action": "praise"}])
        pm.close_due(t + 100)
    return t0 + 100 * n


def test_partner_model_follows_a_betrayal_within_a_few_nights():
    """A partner paid back 100 times in a row, then stops (3 windows a night).
    The first-round model is still sure of it after 2 nights; recency
    weighting and change detection move it below one half."""
    x = features({})
    after = {}
    for name, kw in (("plain", {}), ("fixed", dict(forget=0.02, detect=True))):
        pm = PartnerModels(0.3, **kw)
        t = _answers(pm, "Fenn", True, 100, 0)
        assert pm.predict("Fenn", "give", x)[0] > 0.85
        _answers(pm, "Fenn", False, 6, t + 10_000)         # two nights of 3 windows with no payback
        after[name] = pm.predict("Fenn", "give", x)[0]
    assert after["plain"] > 0.75, after
    assert after["fixed"] < 0.5 and after["fixed"] < after["plain"] - 0.3, after


def test_change_detection_does_not_fire_on_a_steady_partner():
    """Few false alarms: a partner that pays back with chance 0.9, 0.5 or 0.1,
    for 300 windows, is taken for changed in under 3% of them (a false alarm
    only cuts the weights and costs a few windows of relearning)."""
    import random
    x = features({})
    for p in (0.9, 0.5, 0.1):
        rng = random.Random(4)
        pm = PartnerModels(0.3, forget=0.02, detect=True)
        for i in range(300):
            pm.probe("Fenn", "give", x, 100 * i)
            if rng.random() < p:
                pm.observe([{"actor": "Fenn", "t": 100 * i + 1, "action": "praise"}])
            pm.close_due(100 * i + 100)
        assert pm.changes.get("Fenn", 0) <= 9, (p, pm.changes)
    # and a model that has not been told the partner is reliable has no trust to show
    assert PartnerModels().trust("Marta", 0.2, -0.2) == 0.0


def _two_partners(cfg):
    w = World(seed=0, flags="all2", bandit=False)
    for p in w.people.values():
        p.room, p.doing = "the main room", None
    tom = w.people["Tom"]
    w.things["the copper coin"].update(at="Tom", owner="Tom")
    m = Mind("Tom", "peasant", 1, **cfg)
    m.val.v.update({"got_kind": 0.2, "got_harm": -0.2})   # what an answer is worth (V learns this in a life)
    return w, tom, m


def _chance_to_give(w, tom, m):
    """The mind's chance of giving the coin to Fenn and to Oren (the two options differ
    only in the person), and the terms that moved them."""
    opts = w.options(tom)
    need = H.needs(w, tom)
    T = m.terms(w, tom, opts, reader.encode(opts, need), need)
    sc = sum(T.values())
    pi = softmax(sc / m.cfg["tau"])
    ix = {o["text"]: i for i, o in enumerate(opts)}
    a, b = ix["give the copper coin to Fenn"], ix["give the copper coin to Oren"]
    assert abs(sum(T["actor"][[a, b]])) < 1e-12 and (T["world"][a] == T["world"][b])
    return float(pi[a]), float(pi[b])


def test_partner_models_weigh_in_the_choice_and_follow_a_betrayal():
    """Two options that differ only in the person: give the coin to Fenn (who
    pays gifts back) or to Oren (who never does). Fenn is preferred
    by a wide margin; then they swap for 6 windows. The first-round mind still
    prefers Fenn (the partner term is small and the model slow); the fixed mind
    has turned to Oren."""
    got = {}
    for name, cfg in (("first", {}), ("fixed", FIXES)):
        w, tom, m = _two_partners(cfg)
        t = _answers(m.pm, "Fenn", True, 30, 0)
        t = _answers(m.pm, "Oren", False, 30, t)
        f0, o0 = _chance_to_give(w, tom, m)
        t = _answers(m.pm, "Fenn", False, 6, t + 10_000)
        t = _answers(m.pm, "Oren", True, 6, t)
        f1, o1 = _chance_to_give(w, tom, m)
        got[name] = (f0, o0, f1, o1)
    f0, o0, f1, o1 = got["fixed"]
    assert f0 > 0.9 and o0 < 0.1, got                       # trusts the one who pays back
    assert o1 > 0.9 and f1 < 0.1, got                       # and follows the swap within 6 windows each
    f0, o0, f1, o1 = got["first"]
    assert f0 > 0.9 and f1 > o1 + 0.5, got                  # the first round: still stuck on Fenn


# ------------------------------------------------------------ E
def test_episodic_one_shot_and_window():
    ep = Episodic(k=3, cap=5)
    z = np.zeros(reader.ZDIM)
    z[0] = 1.0
    act = ("give", "Fenn")
    assert ep.lookup(act, z[None])[0] == 0.0
    ep.store(act, z, -1.0)                                  # one betrayal
    assert ep.lookup(act, z[None])[0] == -1.0               # counts at once
    assert ep.lookup(("give", "Oren"), z[None])[0] == 0.0   # only for that act
    far = z.copy()
    far[5] = 10.0
    for i in range(3):
        ep.store(act, far, 1.0)
    assert ep.lookup(act, z[None])[0] < ep.lookup(act, far[None])[0]
    for i in range(10):
        ep.store(act, far, 0.5)
    assert len(ep.mem[act]) == 5 and -1.0 not in [v for _, v in ep.mem[act]]   # oldest dropped


# ------------------------------------------------------------ C
def test_curiosity_is_learning_progress_with_habituation():
    learn, flat, noise = Progress(), Progress(), Progress()
    rng = np.random.default_rng(0)
    lp_learn, lp_noise = [], []
    for i in range(200):
        learn.observe("x", 1.0 * 0.98 ** i)
        flat.observe("x", 0.3)
        noise.observe("x", rng.uniform(0, 1))
        lp_learn.append(learn.progress("x"))
        lp_noise.append(noise.progress("x"))
    assert max(lp_learn[:100]) > 0.2 and flat.progress("x") == 0.0
    assert np.mean(lp_learn[20:100]) > 2 * np.mean(lp_noise[20:100]), (np.mean(lp_learn[20:100]), np.mean(lp_noise[20:100]))
    c = Curiosity(hab=10)
    for i in range(30):
        c.w.observe("give", 1.0 * 0.9 ** i)
    b0 = c.bonus("give", None, ("give", "Fenn"))
    for _ in range(20):
        c.visit(("give", "Fenn"))
    assert b0 > 0 and c.bonus("give", None, ("give", "Fenn")) < b0 / 2.5
    assert c.bonus("give", None, ("give", "Oren")) == b0


# ------------------------------------------------------------ the mind as a policy
def _digest(w):
    return hashlib.sha1(json.dumps([(e["actor"], e["text"], e["done"]) for e in w.log]).encode()).hexdigest()


def test_mind_runs_a_lifetime_learns_and_repeats():
    digests = []
    for _ in range(2):
        m = make_mind(seed=3)
        last = {}

        def grab(w, n):
            last["d"] = _digest(w)
            last["w"] = w

        _, stats = lifetime(Cast(Rules(), {"Tom": m}), nights=3, seed=2, flags="all", on_night=grab)
        digests.append(last["d"])
        a = m.agents["Tom"]
        assert a.decisions > 100, a.decisions
        assert np.isfinite(a.ac.W).all() and np.isfinite(a.ac.c).all() and np.isfinite(a.wm.M).all()
        assert np.abs(a.ac.W).sum() > 0 and a.val.v and a.ep.size > 0 and a.habit.h
        assert a.step is None                                  # the night's end closed the last step
        assert set(a.last["terms"]) == {"actor", "habit", "valence", "world", "partner", "episodic", "curiosity"}
        for e in last["w"].log:                                # every act of Tom's was on its menu
            if e["actor"] == "Tom" and "options" in e:
                assert e["options"][e["chosen"]] == e["text"] and e["p_char"][e["chosen"]] == 1.0
    assert digests[0] == digests[1], "the same seeds gave a different night"


def test_mind_modules_switch_off():
    m = make_mind(seed=0, modules="AK")
    assert m.name == "mind:AK"
    lifetime(Cast(Rules(), {"Fenn": m}), nights=1, seed=0, flags="all")
    assert set(m.agents["Fenn"].last["terms"]) == {"actor", "habit"}
    try:
        Mind("Tom", "peasant", 0, temperature=1)
    except AssertionError:
        return
    raise AssertionError("an unknown setting was accepted")


def test_mind_reads_only_what_it_perceived():
    """The acts aimed at me that the mind learns from are only the ones I
    perceived (a theft from me that I did not see is not among them)."""
    hidden_total = 0
    for seed in range(6):
        w = World(seed=seed, flags="all", bandit=True)
        for _ in range(300):
            w.step(Rules())
        for p in w.people.values():
            a = Mind(p.name, p.kind, 0)
            a._sync(w)
            got = a.perceived_at_me(w, p)
            aimed = [e for e in w.log if e.get("target") == p.name and e["actor"] != p.name and e["done"]]
            assert all(e["id"] in p.perceived for e in got)
            assert [e["id"] for e in got] == [e["id"] for e in aimed if e["id"] in p.perceived]
            hidden_total += len(aimed) - len(got)
            assert a.perceived_at_me(w, p) == []                # each event is read once
    assert hidden_total > 0, "no unperceived act in 6 nights: the test checks nothing"


def test_mind_inputs_hold_no_hidden_fact():
    """Everything a learner reads (the critic's state, the encoded menu, the
    raw rows PPO gets, the drive error that is its reward) stays the same
    when facts the character has not perceived change: the others' private
    anger at it, the contents of rooms it cannot see, the chest (the
    verifier's leak; world flag `known`, on in "all")."""
    import random
    from lab.world2.test_world import _hide_change
    rng = random.Random(3)
    n = 0
    for seed in range(3):
        w = World(seed=seed, flags="all", bandit=True)
        for t in range(200):
            w.step(Rules())
            if t % 25 != 10:
                continue
            for p in list(w.people.values()):
                if not p.acts():
                    continue
                w2, p2 = _hide_change(w, p, rng)
                need, need2 = H.needs(w, p), H.needs(w2, p2)
                assert np.array_equal(need, need2) and H.error(w, p) == H.error(w2, p2)
                assert np.array_equal(H.state(w, p, need), H.state(w2, p2, need2))
                m1, m2 = reader.encode(w.options(p), need), reader.encode(w2.options(p2), need2)
                assert np.array_equal(m1.v, m2.v) and np.array_equal(m1.t, m2.t) and np.array_equal(m1.Z, m2.Z)
                assert np.array_equal(reader.raw(m1), reader.raw(m2))
                n += 1
    assert n > 30, n


# ------------------------------------------------------------ nulls
def test_bandit_runs_and_learns_the_immediate_reward():
    b = make_bandit(seed=0)
    lifetime(Cast(Rules(), {"Marta": b}), nights=2, seed=0, flags="all")
    a = b.agents["Marta"]
    assert a.decisions > 50 and np.abs(a.W).sum() > 0 and np.isfinite(a.W).all()
    # the delta rule moves the prediction toward the reward it saw
    opts = [{"verb": "eat", "target": None, "obj": "x", "f": {"need": 0.5}}]
    menu = reader.encode(opts, np.zeros(len(DRIVES)))
    before = a.predict(menu)[0]
    from lab.mindL.mind import Step
    a.step = Step(menu=menu, a=0, pi=np.ones(1), E=1.0, need=None, s=None, t=0, act=None, cue=None, room=None,
                  opt=opts[0])
    a.learn(None, None, [], None, 0.5, None, True)             # reward 0.5
    assert abs(a.predict(menu)[0] - 0.5) < abs(before - 0.5)


def test_ppo_env_hands_over_each_choice_and_ends_each_night():
    env = make_env("Sir Ada", nights=1, seed=0)
    obs, _ = env.reset()
    assert obs.shape == (K * (reader.RAW + 1),)
    steps, total, done = 0, 0.0, False
    while not done:
        rows = obs.reshape(K, reader.RAW + 1)
        n = int(rows[:, -1].sum())
        assert n >= 1
        a = int(np.flatnonzero(rows[:n, reader.VI["wait"]] == 1)[0])       # always wait
        obs, r, done, trunc, _ = env.step(a)
        total += r
        steps += 1
    assert steps > 50 and np.isfinite(total)
    obs, _ = env.reset()                                    # one night only: the lifetime is over
    assert env.finished and len(env.stats) == 1 and not obs.any()


def test_menu_obs_keeps_wait_when_the_menu_is_long():
    opts = [{"verb": "take", "target": None, "obj": f"t{i}", "f": {}} for i in range(K + 10)]
    opts.append({"verb": "wait", "target": None, "obj": None, "f": {}})
    obs, idx = menu_obs(opts, np.zeros(len(DRIVES)))
    assert len(idx) == K and idx[-1] == len(opts) - 1
    rows = obs.reshape(K, reader.RAW + 1)
    assert rows[-1, reader.VI["wait"]] == 1.0 and rows[:, -1].all()


def test_ppo_learns_online_over_a_short_lifetime():
    t0 = time.perf_counter()
    stats, info = run_ppo("Tom", nights=2, seed=0, n_steps=64, batch_size=32)
    assert len(stats) == 2 and info["decisions"] > 100, info
    assert time.perf_counter() - t0 < 120


def test_ban_diagnostic_hides_a_verb_and_keeps_the_hooks():
    from lab.mindL.experiments import Ban
    m = make_mind(seed=0)
    b = Ban(m, ["shout for help", "wait"])
    w = {}
    lifetime(Cast(Rules(), {"Oren": b}), nights=2, seed=1, flags="all", on_night=lambda x, n: w.update(w=x))
    mine = [e for e in w["w"].log if e["actor"] == "Oren" and "options" in e]
    assert mine and all(e["action"] not in ("shout for help", "wait") for e in mine)
    assert m.agents["Oren"].step is None                   # night_end reached the mind through the wrapper


# ------------------------------------------------------------ the tests' parts (tests_L)
def test_seq_partner_changes_its_rule_at_the_set_night():
    from lab.mindL.tests_L import Seq
    q = Seq([[0, "kind"], [2, "hard"], [3, "deaf"]], seed=0)
    seen = []
    lifetime(Cast(Rules(), {"Fenn": q}), nights=4, seed=1, flags="all",
             on_night=lambda w, n: seen.append(q.now.name))
    assert seen == ["P-kind", "P-kind", "P-hard", "P-deaf"], seen


def test_shuffled_partner_answers_someone_drawn_at_random():
    from lab.mindL.tests_L import make_one
    p = make_one("kind~shuf", seed=3)
    got = set()
    for _ in range(40):
        p.todo = []
        p.hear(None, type("Me", (), {"name": "Fenn"})(), {"action": "give", "actor": "Tom", "parts": {"thing": "x"}})
        got |= {who for _, who, _ in p.todo}
    assert "Fenn" not in got and len(got) >= 3, got        # credited to the other four, not only the giver


def test_designer_reward_pays_finished_acts_the_same_whatever_the_need():
    from lab.mindL.tests_L import DESIGNER, Designer
    d = Designer()
    w = World(seed=0, flags="all")
    p = w.people["Tom"]
    assert d(w, p) == 0.0
    w.log.append({"actor": "Tom", "action": "praise", "done": True, "options": []})
    w.log.append({"actor": "Tom", "action": "shove", "done": False, "options": []})     # not finished: no pay
    w.log.append({"actor": "Tom", "action": "eat", "done": True, "options": []})
    w.log.append({"actor": "Oren", "action": "give", "done": True, "options": []})     # someone else's
    assert abs(d(w, p) - (-0.1 * (DESIGNER["praise"] + DESIGNER["eat"]))) < 1e-12


if __name__ == "__main__":
    n = 0
    t0 = time.perf_counter()
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            n += 1
    print(f"{n} passed in {time.perf_counter() - t0:.1f} s")
