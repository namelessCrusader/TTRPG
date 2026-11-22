"""Mechanism tests for act meanings learned from effects (SV3, step 2 of SV5). Plain asserts, like test_mind.

    lab/.venv/bin/python -m lab.mindL.test_effects [out.json]

Each test records its numbers in TABLE (the report prints them). The tests do not need the world to pay.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np
import scipy.stats as st

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from lab.mindL import drives as H
from lab.mindL import reader
from lab.mindL.effects import ActMeanings, DeltaMeanings, Did, Flush, Got, State, Tick, error_drop
from lab.mindL.mind import FIXES, Mind, make_mind
from lab.mindL.partners import features
from lab.world2.lifetime import Cast, lifetime
from lab.world2.partners import Deaf, Hard, Need
from lab.world2.policy import Rules
from lab.world2.world import ALL6, DRIVES, SIGNS_FLAGS, World

TABLE: dict = {}
ALL6S = tuple(ALL6) + tuple(SIGNS_FLAGS)
EFX = {**FIXES, "b_trust": 0.0, "effects": True}
X0 = np.zeros(len(features({})))
X0[0] = 1.0


def ci(xs):
    x = np.asarray(xs, float)
    if len(x) < 2:
        return [float(x.mean()), None, None]
    h = st.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h)]


def fm(c, d=3):
    return f"{c[0]:+.{d}f} [{c[1]:+.{d}f}, {c[2]:+.{d}f}]"


# ------------------------------------------------------------ a synthetic stream with KNOWN effects
E = {("Fenn", "serve"): (-0.30, 0), ("Oren", "serve"): (-0.10, 0), ("Oren", "poke"): (0.25, 1), ("Fenn", "nod"): (0.0, 0)}
DRIFT = np.array([0.002, 0.0005, 0, 0, 0, 0.001, 0])


def synth(seed, ticks=2500, noise=0.01, shuffle=False, model=None):
    """A little world: five seconds-long steps; I ask Fenn (who serves, eases energy -0.30), wave at Fenn (who nods, no effect),
    Oren serves me too (-0.10) and pokes me now and then (safety +0.25). Needs drift. Returns the model."""
    rng = np.random.default_rng(seed)
    m = model or DeltaMeanings(len(DRIVES), keep_log=True)
    need = np.full(len(DRIVES), 0.5)
    t, due, last_obs = 0, [], need.copy()
    for _ in range(ticks):
        dt = int(rng.integers(3, 9))
        need0, tref = last_obs, t
        t += dt
        need = need + DRIFT * dt
        evs = []
        for (when, who, act) in [d for d in due if d[0] <= t]:
            evs.append((when, who, act))
            eff = np.zeros(len(DRIVES))
            eff[E[(who, act)][1]] = E[(who, act)][0]
            need = need + eff
        due = [d for d in due if d[0] > t]
        if rng.random() < 0.04:
            evs.append((t - 1, "Oren", "poke"))
            need = need.copy()
            need[1] += 0.25
        if rng.random() < 0.04:
            evs.append((t - 1, "Oren", "serve"))
            need = need.copy()
            need[0] += -0.10
        obs_need = need + rng.normal(0, noise, len(need))
        for (when, who, act) in evs:
            who_c = who
            if shuffle:                                   # answers credited to a random member of the cast
                who_c = str(rng.choice(["Fenn", "Oren", "Marta", "Sir Ada"]))
            m.update(Got(who_c, act, when, tref, need0))
        m.update(Tick(t, obs_need))
        last_obs = obs_need                               # what the mind saw
        # my acts: ask Fenn, wave at Fenn; answers come 1-3 turns later
        for my, ans, p_do, p_ans in (("ask", "serve", 0.12, 0.8), ("wave", "nod", 0.12, 0.9)):
            if rng.random() < p_do:
                m.update(Did("Fenn", my, X0, t))
                if rng.random() < p_ans:
                    due.append((t + int(rng.integers(1, 4)), "Fenn", ans))
    return m


def test_accuracy_synthetic():
    """The learned effect of each (partner, act) converges to the true effect (known here), with the drift taken off."""
    err, rows = {k: [] for k in E}, {}
    for seed in range(1, 9):
        m = synth(seed)
        for k, (v, j) in E.items():
            r = m.raw_effect(*k)
            assert r is not None, k
            err[k].append(r[j] - v)
            other = np.delete(r, j)
            err[k + ("off",)] = err.get(k + ("off",), []) + [float(np.abs(other).max())]
    for k in E:
        c = ci(err[k])
        rows[f"{k[0]}:{k[1]}"] = {"true": E[k][0], "error": c}
        assert c[1] <= 0 <= c[2] or abs(c[0]) < 0.02, (k, c)         # unbiased within noise, or tiny
        assert abs(c[0]) < 0.04, (k, c)
        assert np.mean(err[k + ("off",)]) < 0.06, (k, err[k + ("off",)])
    TABLE["accuracy_synthetic"] = rows


def _batch_effects(log, ridge=1.0):
    """The reference: the best linear separation of the measured changes into per-(partner, act) effects over the whole log, by ridge
    least squares (each entry's change = its own effect + the effects of the entries overlapping its window). The online rule
    should land on this. Returns {cell: vector}, {cell: number of entries}."""
    cells = sorted({(w, a) for w, a, _, _ in log} | {o for _, _, _, os_ in log for o in os_})
    ix = {c: i for i, c in enumerate(cells)}
    A = np.zeros((len(log), len(cells)))
    Y = np.array([b for _, _, b, _ in log])
    for r, (w, a, _, others) in enumerate(log):
        A[r, ix[(w, a)]] += 1
        for o in others:
            A[r, ix[o]] += 1
    th = np.linalg.solve(A.T @ A + ridge * np.eye(len(cells)), A.T @ Y)
    n = {c: sum(1 for w, a, _, _ in log if (w, a) == c) for c in cells}
    return {c: th[i] for c, i in ix.items()}, n


def test_accuracy_on_logged_lives():
    """On real lives (a mind in the all6s inn with Fenn P-need and Oren P-hard), the learned effect of each (partner, act) matches the
    measured effect: the batch least-squares separation of the logged credit windows (overlapping acts shared out), for the cells with
    at least 15 entries. Also reported: the plain mean of the entries with no other act in their window, where there are 8 or more."""
    slopes, maes, iso_s, cells, ncell = [], [], [], 0, 0
    for seed in (1, 2, 3, 4):
        pol = make_mind(seed, **EFX, fx_log=True)
        lifetime(Cast(Rules(), {"Tom": pol, "Fenn": Need(0.9, seed), "Oren": Hard(0.9, seed + 50)}), 40, seed, flags=ALL6S, fade=0.9)
        fx = pol.agents["Tom"].fx
        th, cnt = _batch_effects(fx.credit_log)
        xs, ys = [], []
        for k, n in cnt.items():
            if n >= 15 and k in fx.eff:
                xs += list(th[k])
                ys += list(fx.eff[k][0])
                cells += 1
        xs, ys = np.array(xs), np.array(ys)
        assert len(xs) >= 14, len(xs)
        slopes.append(float((xs @ ys) / (xs @ xs)))
        maes.append(float(np.abs(xs - ys).mean()))
        by: dict = {}
        for who, act, base, others in fx.credit_log:
            if not others:
                by.setdefault((who, act), []).append(base)
        a = [(np.mean(v, axis=0), fx.eff[k][0]) for k, v in by.items() if len(v) >= 8 and k in fx.eff]
        if a:
            ncell += len(a)
            x = np.concatenate([p for p, _ in a])
            y = np.concatenate([q for _, q in a])
            iso_s.append(float((x @ y) / (x @ x)))
    c1, c2, c3 = ci(slopes), ci(maes), ci(iso_s) if len(iso_s) > 1 else [None] * 3
    TABLE["accuracy_logged"] = {"slope_learned_on_batch_measured": c1, "mean_abs_error": c2, "cells": cells, "seeds": 4,
                                "slope_on_isolated_mean": c3, "isolated_cells": ncell}
    assert c1[1] < 1.0 < c1[2] or abs(c1[0] - 1) < 0.2, c1
    assert c2[0] < 0.03, c2


def _hungry_mind():
    return Mind("Tom", "peasant", 0, **EFX)


def _fake_opts(who="Fenn", verbs=("ask",)):
    return [{"verb": v, "target": who, "obj": None, "f": {}} for v in verbs]


def test_state_dependence_same_act_different_worth():
    """A gift of food is worth more to the hungry than to the fed; coin more to the poor than to the rich. Same model, same act."""
    m = _hungry_mind()
    ix = {d: i for i, d in enumerate(DRIVES)}
    food, coin = np.zeros(len(DRIVES)), np.zeros(len(DRIVES))
    food[ix["energy"]], coin[ix["wealth"]] = -0.4, -0.5
    rows = {}
    for who, act, vec in (("Fenn", "serve", food), ("Marta", "give coin", coin)):
        # Fenn serves in answer to an ask: learn the answer and its effect from experience (as in the stream)
        t = 0
        for i in range(40):
            need0 = np.full(len(DRIVES), 0.5)
            m.fx.update(Did(who, "ask", X0, t))
            m.fx.update(Got(who, act, t + 1, t, need0))
            m.fx.update(Tick(t + 14, np.clip(need0 + vec, 0, 1)))
            m.fx.update(Tick(t + 31, np.clip(need0 + vec, 0, 1)))
            t += 40
    opts = [{"verb": "ask", "target": "Fenn", "obj": None, "f": {}}, {"verb": "ask", "target": "Marta", "obj": None, "f": {}}]

    def worth(kind, need, i):
        mm = Mind("Tom", kind, 0, **EFX)
        mm.fx = m.fx
        return float(mm.fx_term(opts, need, kind)[i])
    hungry, fed = np.full(len(DRIVES), 0.3), np.full(len(DRIVES), 0.3)
    hungry[ix["energy"]], fed[ix["energy"]] = 0.85, 0.05
    poor, rich = np.full(len(DRIVES), 0.3), np.full(len(DRIVES), 0.3)
    poor[ix["wealth"]], rich[ix["wealth"]] = 0.9, 0.05
    wh, wf = worth("peasant", hungry, 0), worth("peasant", fed, 0)
    wp, wr = worth("merchant", poor, 1), worth("merchant", rich, 1)
    rows = {"food: hungry": wh, "food: fed": wf, "coin: poor": wp, "coin: rich": wr}
    TABLE["state_dependence"] = rows
    assert wh > 0.2 and abs(wf) < 0.02, rows
    assert wp > 0.15 and abs(wr) < 0.02, rows


def test_shuffled_answers_do_not_separate_partners():
    """A prober (random choices, the mind's own learning) lives with Fenn P-need and Oren P-hard. For each partner the learned worth of my act
    (from the learned answers and effects, in a middling state) is a function of the act: Oren's blow follows my provocations, Fenn's
    payment follows my gifts. What must separate is the CONTINGENCY: the spread (max minus min) of the worth of my acts over the acts I did at
    that partner 3 times or more, averaged over the two partners: whether what a partner does to me depends on what I do to it. With
    the answers credited to random members of the cast the spread falls (paired by seed, real minus shuffled, 95 percent t interval over 6
    seeds). Also reported, not asserted: the mean absolute difference of the two partners' worth of the same acts. It does NOT fall, because
    Fenn and Oren differ in what they do to me whatever I do (their own background acts), which a mind should learn too."""
    from lab.mindL.tests_L import make_one
    real, shuf, spread_r, spread_s = [], [], [], []
    for seed in range(1, 7):
        for tag, out, sp in (("", real, spread_r), ("~shuf", shuf, spread_s)):
            class Prober(Mind):
                def pick(self, w, p, opts, menu, need, s):
                    a = int(self.rng.integers(len(opts)))
                    pi = np.zeros(len(opts))
                    pi[a] = 1.0
                    return a, pi
            pol = type(make_mind(seed))(lambda n, k, s=seed: Prober(n, k, s, **EFX), "prober")
            fen = make_one("need" + tag, seed)
            ore = make_one("hard" + tag, seed + 50)
            lifetime(Cast(Rules(), {"Tom": pol, "Fenn": fen, "Oren": ore}), 40, seed, flags=ALL6S, fade=0.9)
            a = pol.agents["Tom"]
            need = np.full(len(DRIVES), 0.5)
            acts = {who: {k[1] for k in a.fx.cw if k[0] == who and a.fx.cw[k][1] >= 3} for who in ("Fenn", "Oren")}
            common = sorted(acts["Fenn"] & acts["Oren"])
            assert common, acts

            def v(who, act):
                return sum(pa * error_drop("peasant", need, a.fx.effect(who, ans, State(None, need)))
                           for ans, pa in a.fx.predict(who, act, State(X0, need)).items())
            F = np.array([v("Fenn", x) for x in common])
            O = np.array([v("Oren", x) for x in common])
            out.append(float(np.abs(F - O).mean()))
            sp.append(float(np.mean([F.max() - F.min(), O.max() - O.min()])))
    d = [r - s for r, s in zip(spread_r, spread_s)]
    TABLE["shuffled_answers"] = {"contingency (spread of worth over my acts), real": ci(spread_r), "shuffled": ci(spread_s),
                                 "real minus shuffled": ci(d),
                                 "(info) partner difference, real": ci(real), "(info) partner difference, shuffled": ci(shuf)}
    assert ci(d)[1] > 0, TABLE["shuffled_answers"]


def test_deaf_partner_teaches_nothing():
    for seed in (1, 2):
        pol = make_mind(seed, **EFX)
        lifetime(Cast(Rules(), {"Tom": pol, "Fenn": Deaf(seed), "Oren": Deaf(seed + 5)}), 15, seed, flags=ALL6S, fade=0.9)
        fx = pol.agents["Tom"].fx
        need = np.full(len(DRIVES), 0.5)
        for who in ("Fenn", "Oren"):
            assert not any(k[0] == who for k in fx.eff), who          # no effect estimate for a deaf partner
            assert fx.predict(who, "ask", State(X0, need)) == {}, who  # and no answers expected from it
            assert not any(k[0] == who and fx.cw[k][1] for k in fx.cw if k[0] != "*"), who
        o = [{"verb": "ask", "target": "Fenn", "obj": None, "f": {}}]
        assert pol.agents["Tom"].fx_term(o, need, "peasant")[0] == 0.0
    TABLE["deaf"] = {"effect entries for Fenn/Oren": 0, "answers expected": 0, "worth of an act at them": 0.0}


def test_placebo_act_has_no_worth():
    """An act whose answer has no effect (a wave answered by a nod) is worth about 0; an ask answered by food is worth a lot (hungry state)."""
    ix = DRIVES.index("energy")
    need = np.full(len(DRIVES), 0.3)
    need[ix] = 0.85
    w_wave, w_ask = [], []
    for seed in range(1, 9):
        m = synth(seed)
        for act, out in (("wave", w_wave), ("ask", w_ask)):
            out.append(sum(pa * error_drop("peasant", need, m.effect("Fenn", a, State(None, need)))
                           for a, pa in m.predict("Fenn", act, State(X0, need)).items()))
    c0, c1 = ci(w_wave), ci(w_ask)
    TABLE["placebo"] = {"worth of wave (no effect)": c0, "worth of ask (food)": c1}
    assert c0[1] <= 0 <= c0[2] or abs(c0[0]) < 0.01, c0
    assert abs(c0[0]) < 0.02 and c1[1] > 0.15, (c0, c1)


def test_cultural_prior_shifts_early_choices_and_experience_overrides():
    ix = DRIVES.index("energy")
    prior = {"serve": {"delta": {"energy": -0.5}, "n0": 3, "p": 0.5}}
    w = World(seed=4, flags=ALL6S, bandit=True, turns=300)
    seen = {}

    class Spy(Rules):
        def probs(self, w_, p, opts):
            if p.name == "Tom" and "o" not in seen and any(o["target"] == "Fenn" for o in opts):
                seen["o"], seen["p"], seen["w"] = opts, p, w_
            return super().probs(w_, p, opts)
    for _ in range(300):
        w.step(Spy())
        if "o" in seen:
            break
    opts, p, w_ = seen["o"], seen["p"], seen["w"]
    need = H.needs(w_, p)
    need[ix] = 0.85
    menu = reader.encode(opts, need)
    plain = Mind("Tom", p.kind, 0, **EFX)
    primed = Mind("Tom", p.kind, 0, **EFX, fx_prior=prior)
    t0, t1 = plain.terms(w_, p, opts, menu, need)["partner"], primed.terms(w_, p, opts, menu, need)["partner"]
    at_fenn = np.array([o["target"] == "Fenn" for o in opts])
    assert np.allclose(t0, 0) and t1[at_fenn].min() > 0.05, (t0.max(), t1[at_fenn].min())
    pi0 = plain.pick(w_, p, opts, menu, need, None)[1]
    pi1 = primed.pick(w_, p, opts, menu, need, None)[1]
    shift = float(pi1[at_fenn].sum() - pi0[at_fenn].sum())
    assert shift > 0, shift
    # experience: the "serve" answers come and do nothing to my needs; the prior fades with n0/(n0+n)
    m = primed.fx
    v0 = float(primed.fx_term(opts, need, p.kind)[at_fenn].mean())
    t = 0
    for i in range(30):
        n0 = np.full(len(DRIVES), 0.5)
        m.update(Did("Fenn", "ask", X0, t))
        m.update(Got("Fenn", "serve", t + 1, t, n0))
        m.update(Tick(t + 14, n0))
        m.update(Tick(t + 31, n0))
        t += 40
    v1 = float(primed.fx_term(opts, need, p.kind)[at_fenn].mean())
    left = abs(m.raw_effect("Fenn", "serve")[ix]) / 0.5
    TABLE["prior"] = {"partner term at Fenn, no prior": float(t0[at_fenn].mean()), "with prior": float(t1[at_fenn].mean()),
                      "chance on acts at Fenn, shift": shift, "worth before experience": v0, "after 30 contrary experiences": v1,
                      "share of the prior left in the effect": left}
    assert left < 0.15 and v1 < 0.25 * v0, TABLE["prior"]


def test_flag_off_is_the_old_mind_and_effects_uses_no_labels():
    def sig(pol):
        _, stats = lifetime(Cast(Rules(), {"Tom": pol}), 3, 1, flags="all6", fade=0.9)
        a = pol.agents["Tom"]
        h = hashlib.sha256(np.concatenate([a.ac.U.ravel(), a.ac.W.ravel(), a.ac.c]).tobytes()).hexdigest()[:16]
        return [x["drive_error"]["Tom"] for x in stats], h, a.decisions, a.val.v
    a = sig(make_mind(1, **FIXES))
    b = sig(make_mind(1, **FIXES, effects=False, fx_prior=None, fx_log=True))
    assert a == b
    pol = make_mind(1, **EFX)
    lifetime(Cast(Rules(), {"Tom": pol}), 3, 1, flags="all6", fade=0.9)
    v = pol.agents["Tom"].val.v
    assert "got_kind" not in v and "got_harm" not in v
    assert pol.agents["Tom"].fx is not None and pol.agents["Tom"].pm.n == {}
    TABLE["flag_off"] = {"same drive error, weights hash, decisions and valences as the explicit-off mind": True,
                         "plus the golden check in test_mind": "run separately"}


def test_the_interface_can_be_swapped():
    class Null(ActMeanings):
        def predict(self, partner, my_act, state):
            return {}

        def effect(self, partner, their_act, state):
            return np.zeros(len(DRIVES))

        def update(self, obs):
            return []

        def set_prior(self, *a, **k):
            pass
    pol = make_mind(1, **EFX)
    pol.make = (lambda make: (lambda n, k: _swap(make(n, k))))(pol.make)

    def _swap(a):
        a.fx = Null()
        return a
    _, stats = lifetime(Cast(Rules(), {"Tom": pol}), 2, 1, flags="all6", fade=0.9)
    assert len(stats) == 2


if __name__ == "__main__":
    t0 = time.perf_counter()
    n = 0
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            t1 = time.perf_counter()
            f()
            n += 1
            print(f"ok {name} ({time.perf_counter() - t1:.0f} s)", flush=True)
    print(f"{n} passed in {time.perf_counter() - t0:.0f} s")
    if len(sys.argv) > 1:
        json.dump(TABLE, open(sys.argv[1], "w"), indent=1, default=float)
    print(json.dumps(TABLE, indent=1, default=float))
