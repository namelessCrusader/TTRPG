"""Controls for ACQUIRED CONCERNS (SV1, SV2; step 3 of SV5; learned-mind-target 9.2). Plain asserts, like test_effects.

    lab/.venv/bin/python -m lab.mindL.test_concerns [out.json]

These are the gate. They do not need the world to pay: the formation controls (placebo, habituation, formed then extinguished, grief, the
clones, the model of what my acts do to others) run on synthetic streams whose payer is known, through the same module the mind uses;
the swap, the stoic and the no-loop controls run in the real inn, through the real mind and the real options and signs. Each test
records its numbers in TABLE. Seeds 1-8, 95 percent t intervals, paired by seed.
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
from lab.mindL.concerns import JND, NMIN, Concerns, OtherEffects, RWConcerns, StepObs, need_of
from lab.mindL.mind import FIXES, Mind, make_mind, softmax
from lab.world2.lifetime import Cast, lifetime
from lab.world2.partners import Hard, Need
from lab.world2.policy import Rules
from lab.world2.world import ALL6, DRIVES, SIGNS_FLAGS, World

TABLE: dict = {}
ALL6S = tuple(ALL6) + tuple(SIGNS_FLAGS)
SEEDS = range(1, 9)


def ci(xs):
    x = np.asarray(xs, float)
    if len(x) < 2:
        return [float(x.mean()), None, None]
    h = st.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h)]


def fm(c, d=3):
    return f"{c[0]:+.{d}f} [{c[1]:+.{d}f}, {c[2]:+.{d}f}]"


# ------------------------------------------------------------ a synthetic life with a KNOWN payer
PAY = 0.02          # the mean homeostatic reward of a step with the payer present (a meal is 0.3, an answer in the inn 0.001)
SD = 0.02


def life(seed, pay, present=None, steps=3000, per_night=100, c=None, dt=3, always=(), rng_seed=None, bias=0.0):
    """steps of the mind: each target is present with its chance; the step's reward is noise plus the sum of the pay of the targets
    present. pay: {target: mean reward when present}; present: {target: chance} (default 0.5); always: present every step."""
    rng = np.random.default_rng([seed, 0 if rng_seed is None else rng_seed])
    c = c or RWConcerns()
    present = present or {}
    names = sorted(set(pay) | set(present) | set(always))
    for k in range(steps):
        here = [T for T in names if T in always or rng.random() < present.get(T, 0.5)]
        r = float(rng.normal(0.0, SD) + bias + sum(pay.get(T, 0.0) for T in here))
        c.learn(StepObs(r, {T: 0.0 for T in here}, {T: 0.0 for T in here}, tuple(here), "look around", None, 0.0, dt))
        if k % per_night == per_night - 1:
            c.night()
    return c


def test_placebo_targets_keep_weights_near_zero():
    """A payer (reward +PAY when present) and placebo targets (present at random, or always: they predict nothing)."""
    pay, plc, alw = [], [], []
    for s in SEEDS:
        c = life(s, {"Fenn": PAY}, {"P1": 0.5, "P2": 0.3, "P3": 0.7}, always=("Everpresent",))
        t = c.table()
        pay.append(t["Fenn"])
        plc.append(np.mean([t.get(p, 0.0) for p in ("P1", "P2", "P3")]))
        alw.append(t.get("Everpresent", 0.0))
    p, q, a = ci(pay), ci(plc), ci(alw)
    TABLE["placebo"] = {"weight of the payer": p, "mean weight of 3 random placebos": q, "weight of an always-present placebo": a}
    assert p[1] > 0.15, p
    assert q[2] < 0.05 and a[2] < 0.05 and q[2] < 0.25 * p[1], (p, q, a)


def test_placebo_needs_the_baseline_taken_off():
    """Owed from step 3's verifier. When every step pays a little (a good mood: bias > 0), a target that is ALWAYS there predicts nothing
    ONLY because the running average of the reward is subtracted. With it, the always-present placebo stays near 0 and the payer is
    still found. Without it (lr_base = 0: the average never moves from 0) the placebo soaks up the mood: the test must FAIL then."""
    BIAS = 0.05
    with_b, without_b, payer = [], [], []
    for s in SEEDS:
        c = life(s, {"Fenn": PAY}, {"Fenn": 0.5}, always=("Everpresent",), bias=BIAS)
        with_b.append(c.table().get("Everpresent", 0.0))
        payer.append(c.table()["Fenn"])
        m = life(s, {"Fenn": PAY}, {"Fenn": 0.5}, always=("Everpresent",), bias=BIAS, c=RWConcerns(lr_base=0.0))
        without_b.append(m.table().get("Everpresent", 0.0))
    A, B, P = ci(with_b), ci(without_b), ci(payer)
    TABLE["placebo_baseline"] = {"always-present, baseline on": A, "always-present, baseline OFF (the mutation)": B, "payer, baseline on": P}
    assert A[2] < 0.05, A                      # near 0 only WITH the baseline
    assert B[1] > 0.3, B                       # the mutation is caught: without it the placebo is loved
    assert P[1] > 0.1, P


def test_formed_then_extinguished():
    """E-A: a payer's weight is formed, then the payer is present and pays nothing (it is extinguished), then absent (it fades)."""
    peak, mid, end, e_cost = [], [], [], []
    for s in SEEDS:
        c = life(s, {"Fenn": PAY}, {"Fenn": 0.5}, steps=2000)
        peak.append(c.table()["Fenn"])
        life(s + 100, {"Fenn": -PAY}, {"Fenn": 0.5}, steps=800, c=c)        # present, and now the error rises after contact
        mid.append(c.table().get("Fenn", 0.0))
        life(s + 200, {}, {"Fenn": 0.0}, steps=3000, c=c)                    # absent: 30 nights of disuse
        end.append(c.table().get("Fenn", 0.0))
    P, M, E_ = ci(peak), ci(mid), ci(end)
    TABLE["formed_extinguished"] = {"weight after forming": P, "after 8 nights of disillusion": M, "after 30 nights of disuse": E_}
    assert P[1] > 0.15 and M[2] < 0.25 * P[1], (P, M)
    # disuse alone: formed, then absent
    d = []
    for s in SEEDS:
        c = life(s, {"Fenn": PAY}, {"Fenn": 0.5}, steps=2000)
        w0 = c.table()["Fenn"]
        life(s + 300, {}, {"Fenn": 0.0}, steps=3000, c=c)
        d.append(c.table().get("Fenn", 0.0) / w0)
    D = ci(d)
    TABLE["formed_extinguished"]["share left after 30 nights of disuse alone"] = D
    assert D[2] < 0.6 and D[1] > 0.0 - 1e-9, D


def test_habituation_repeated_contact_pays_less():
    """h falls under unbroken contact and recovers apart; and one paid step teaches a target met all the time much less than a fresh one."""
    c = RWConcerns()
    rows = c.rows.setdefault("Fenn", __import__("lab.mindL.concerns", fromlist=["_Row"])._Row())
    for _ in range(100):
        c.learn(StepObs(0.0, {"Fenn": 0.0}, {"Fenn": 0.0}, ("Fenn",), "x", None, 0.0, 3))
    h_in = rows.h
    for _ in range(100):
        c.learn(StepObs(0.0, {}, {}, (), "x", None, 0.0, 3))
    h_back = rows.h
    # one paid step teaches less to a target met all the time than to a fresh one: the same step, the increment of a
    inc_fresh, inc_hab = [], []
    for s in SEEDS:
        for contact, out in ((0, inc_fresh), (300, inc_hab)):
            c = RWConcerns()
            c.rows["Fenn"] = __import__("lab.mindL.concerns", fromlist=["_Row"])._Row()
            for _ in range(contact // 3):
                c.learn(StepObs(0.0, {"Fenn": 0.0}, {"Fenn": 0.0}, ("Fenn",), "x", None, 0.0, 3))
            a0 = c.rows["Fenn"].a
            c.learn(StepObs(PAY * (1 + 0.1 * s), {"Fenn": 0.0}, {"Fenn": 0.0}, ("Fenn",), "x", None, 0.0, 3))
            out.append((c.rows["Fenn"].a - a0) / (PAY * (1 + 0.1 * s)))
    on, off = ci(inc_hab), ci(inc_fresh)
    d40 = ci(np.array(inc_fresh) - np.array(inc_hab))
    TABLE["habituation"] = {"h after 300 turns of contact": h_in, "h after 300 more turns apart": h_back,
                            "increment of a from one paid step, fresh target (per unit of pay)": off,
                            "after 300 turns of contact": on, "fresh minus habituated (paired)": d40, "ratio": float(on[0] / off[0])}
    assert h_in < 0.35 and h_back > 0.6, (h_in, h_back)
    assert d40[1] > 0 and on[0] < 0.4 * off[0], (on, off)


def _scene(seed=0, hunger=0.8, flags=ALL6S):
    """Tom (the mind) with Fenn and Oren in one room, Tom holding the coin; both of them hungry, all showing it."""
    w = World(seed=seed, flags=flags, bandit=False)
    for p in w.people.values():
        p.room, p.doing = "the main room", None
    w.things["the copper coin"].update(at="Tom", owner="Tom")
    for n in ("Fenn", "Oren"):
        w.people[n].hunger, w.people[n].express = hunger, 1.0
    return w, w.people["Tom"]


def _mind(table=None, seed=1, **kw):
    m = Mind("Tom", "peasant", seed, concerns=True, cn_fx_prior={"give": {"rho": 0.7, "n0": 50}}, **kw)
    if table is not None:
        m.cn.set_table(table)
    return m


def _logodds(w, tom, m):
    """log chance of giving the coin to Fenn minus to Oren: the two options differ only in the person."""
    m._rd = m._readings(w, tom)
    opts = w.options(tom)
    need = H.needs(w, tom)
    T = m.terms(w, tom, opts, reader.encode(opts, need), need)
    pi = softmax(sum(T.values()) / m.cfg["tau"])
    ix = {o["text"]: i for i, o in enumerate(opts)}
    a, b = ix["give the copper coin to Fenn"], ix["give the copper coin to Oren"]
    assert abs(T["actor"][a] - T["actor"][b]) < 1e-12 and T["world"][a] == T["world"][b]
    return float(np.log(pi[a]) - np.log(pi[b])), T["concern"][[a, b]]


def test_swapping_learned_weights_flips_the_choice():
    """Two minds with different lives: A learned to care for Fenn (he paid), B for Oren. Same scene, same state, same mind: put A's
    learned weights in, then B's. The choice between the two otherwise identical options flips."""
    a_tab, b_tab, d = [], [], []
    for s in SEEDS:
        A = life(s, {"Fenn": PAY, "Oren": 0.0}, {"Fenn": 0.5, "Oren": 0.5}).table()
        B = life(s, {"Fenn": 0.0, "Oren": PAY}, {"Fenn": 0.5, "Oren": 0.5}).table()
        w, tom = _scene(seed=s)
        m = _mind(seed=s)
        m.cn.set_table({k: v for k, v in A.items() if k in ("Fenn", "Oren")})
        la, ta = _logodds(w, tom, m)
        m.cn.set_table({k: v for k, v in B.items() if k in ("Fenn", "Oren")})
        lb, tb = _logodds(w, tom, m)
        m.cn.set_table({})
        l0, _ = _logodds(w, tom, m)
        a_tab.append(la)
        b_tab.append(lb)
        d.append(la - lb)
        assert abs(l0) < 1e-9                          # no concerns: no difference at all between the two (the null)
    D = ci(d)
    TABLE["swap"] = {"log odds Fenn over Oren with A's weights": ci(a_tab), "with B's weights": ci(b_tab),
                     "A minus B (paired by seed)": D, "no concerns": 0.0}
    assert ci(a_tab)[1] > 0 and ci(b_tab)[2] < 0 and D[1] > 0, (ci(a_tab), ci(b_tab), D)


def test_swap_exercises_the_eta_sign():
    """Owed from step 3's verifier. The swap test above runs on a prior with eta = 0 (only the help part, rho). Here the prior is a HARM
    (strike: eta > 0, rho = 0): the concern term of a blow at the cared-for person is NEGATIVE (w * (rho * need - eta)), so A (cares for Fenn)
    gets a lower term for striking Fenn than for striking Oren, B the opposite, and without concerns the two terms are equal. If the eta
    sign were dropped (or flipped), A would gain from striking Fenn and this fails."""
    ds, ta_, tb_ = [], [], []
    for s in SEEDS:
        A = life(s, {"Fenn": PAY, "Oren": 0.0}, {"Fenn": 0.5, "Oren": 0.5}).table()
        B = life(s, {"Fenn": 0.0, "Oren": PAY}, {"Fenn": 0.5, "Oren": 0.5}).table()
        w, tom = _scene(seed=s)
        m = Mind("Tom", "peasant", s, concerns=True, cn_fx_prior={"strike": {"rho": 0.0, "eta": 0.5, "n0": 50}})

        def gap(tab):
            m.cn.set_table({k: v for k, v in tab.items() if k in ("Fenn", "Oren")})
            m._rd = m._readings(w, tom)
            opts = w.options(tom)
            need = H.needs(w, tom)
            T = m.terms(w, tom, opts, reader.encode(opts, need), need)
            ix = {o["text"]: i for i, o in enumerate(opts)}
            a, b = ix["strike Fenn"], ix["strike Oren"]
            return float(T["concern"][a] - T["concern"][b])

        ga, gb, g0 = gap(A), gap(B), gap({})
        ta_.append(ga)
        tb_.append(gb)
        ds.append(ga - gb)
        assert abs(g0) < 1e-12
    TABLE["swap_eta"] = {"concern term, strike Fenn minus strike Oren, A's weights": ci(ta_), "B's weights": ci(tb_), "A minus B": ci(ds)}
    assert ci(ta_)[2] < 0 and ci(tb_)[1] > 0 and ci(ds)[2] < 0, (ci(ta_), ci(tb_), ci(ds))


def test_clones_with_different_lives_come_to_different_concerns():
    """Clones (same seed, same nature, same presence draws) lived in different worlds (A: Fenn pays, B: Oren pays) are farther apart
    than the same life with other draws: D_life / D_rng."""
    dl, dr = [], []
    names = ("Fenn", "Oren", "Tom", "Marta")
    for s in SEEDS:
        pres = {n: 0.5 for n in names}
        A = life(s, {"Fenn": PAY}, pres).table()
        B = life(s, {"Oren": PAY}, pres).table()
        A2 = life(s, {"Fenn": PAY}, pres, rng_seed=7).table()
        dist = lambda x, y: float(sum(abs(x.get(n, 0.0) - y.get(n, 0.0)) for n in names))
        dl.append(dist(A, B))
        dr.append(dist(A, A2))
    L, R = ci(dl), ci(dr)
    TABLE["clones"] = {"D_life": L, "D_rng": R, "D_life / D_rng": float(L[0] / R[0]), "paired D_life - D_rng": ci(np.array(dl) - np.array(dr))}
    assert L[0] / R[0] >= 2.0 and ci(np.array(dl) - np.array(dr))[1] > 0, (L, R)


def test_a_stoic_gets_less_care_than_an_expressive_person():
    """Same true hunger, same weight: the stoic (expressiveness 0.25) is read lower, so the pull to feed it is smaller."""
    exp, sto, rd_e, rd_s = [], [], [], []
    for s in SEEDS:
        w, tom = _scene(seed=s, hunger=0.8)
        w.people["Fenn"].express, w.people["Oren"].express = 1.0, 0.25
        m = _mind({"Fenn": 0.6, "Oren": 0.6}, seed=s)
        lo, t = _logodds(w, tom, m)
        exp.append(float(t[0]))
        sto.append(float(t[1]))
        rd_e.append(m._rd["Fenn"])
        rd_s.append(m._rd["Oren"])
    TABLE["stoic"] = {"need read, expressive": ci(rd_e), "need read, stoic": ci(rd_s), "concern term for feeding the expressive": ci(exp),
                      "for feeding the stoic": ci(sto), "expressive minus stoic (paired)": ci(np.array(exp) - np.array(sto))}
    d = ci(np.array(exp) - np.array(sto))
    assert d[1] > 0 and ci(sto)[0] < 0.5 * ci(exp)[0], d


def test_grief_lasts_longer_for_heavier_weights():
    """Report only (not a gate on a number): after the target dies the weight fades with disuse; nights until it is under 0.05."""
    out = {}
    last = 0
    for w0 in (0.2, 0.5, 0.9):
        c = RWConcerns()
        c.set_table({"Fenn": w0})
        nights = 0
        while c.pull("Fenn") >= 0.05 and nights < 1000:
            c.night()
            nights += 1
        out[f"weight {w0}"] = nights
        assert nights > last
        last = nights
    TABLE["grief"] = {"nights until the weight is under 0.05": out, "rule": "exponential disuse fade: t = ln(w0 / 0.05) / -ln(1 - fade)"}


def test_no_standing_next_to_the_loved_one_loop():
    """A mind that cares a lot (w = 0.9) for Fenn and Oren turns to them when they need something, and hardly otherwise: the pull is the
    weight times the relief they need, and a healthy person needs none (the small rest is sign noise), so there is no loop of standing next to the loved one.
    Measured on the same decisions of one ordinary life (the mind has no concern; at each decision the chance on acts aimed at Fenn
    or Oren is also computed as if it held w = 0.9 for both), split by whether either of them shows a need (a reading of at least NMIN)."""
    healthy, needy, share_needy = [], [], []
    for s in range(1, 9):
        pol = make_mind(s, **{**FIXES, "b_trust": 0.0, "concerns": True, "cn_fx_prior": {"give": {"rho": 0.7, "n0": 50}}})
        rec = {"h": [], "n": []}
        orig = Mind.pick

        def pick(self, w, p, opts, menu, need, sc, orig=orig, rec=rec):
            a, pi = orig(self, w, p, opts, menu, need, sc)
            if p.name == "Tom":
                T = dict(self.last["terms"])
                heavy = RWConcerns(effects=self.cn.fx)
                heavy.set_table({"Fenn": 0.9, "Oren": 0.9})
                T["concern"] = self.cfg["b_concern"] * np.array([heavy.expected(o["verb"], o["target"], self._rd.get(o["target"], 0.0))
                                                                  if o["target"] in self._rd else 0.0 for o in opts])
                pi1 = softmax(sum(T.values()) / self.cfg["tau"])
                at = np.array([o["target"] in ("Fenn", "Oren") for o in opts])
                key = "n" if any(self._rd.get(n, 0.0) >= NMIN for n in ("Fenn", "Oren")) else "h"
                rec[key].append((float(pi1[at].sum() - pi[at].sum()), float(pi[at].sum())))
            return a, pi
        Mind.pick = pick
        try:
            lifetime(Cast(Rules(), {"Tom": pol, "Fenn": Need(0.9, s), "Oren": Hard(0.9, s + 50)}), 10, s, flags=ALL6S, fade=0.9)
        finally:
            Mind.pick = orig
        keys = set(pol.agents["Tom"].cn.table())          # targets are persons and things the mind perceives, never acts
        assert keys <= set(reader.PEOPLE) | {k for k in keys if k.startswith("thing:")}, keys
        assert not keys & set(reader.VERBS)
        healthy.append(float(np.mean([d for d, _ in rec["h"]])) if rec["h"] else 0.0)
        needy.append(float(np.mean([d for d, _ in rec["n"]])) if rec["n"] else 0.0)
        share_needy.append(len(rec["n"]) / max(1, len(rec["n"]) + len(rec["h"])))
    h, n = ci(healthy), ci(needy)
    TABLE["no_loop"] = {"rise in the chance on acts at Fenn/Oren from w = 0.9, when both look healthy": h,
                        "when one of them shows a need": n, "share of decisions with a need showing": ci(share_needy)}
    assert h[2] < 0.05 and n[1] > 0.0 and h[0] < 0.1 * n[0], (h, n)      # a pull when needy; about a twentieth of it when healthy (sign noise below NMIN)


def test_model_of_what_my_acts_do_to_others():
    """OtherEffects on a synthetic person: 'give' relieves 70 percent of the need, 'praise' does nothing, 'strike' raises it by 0.3;
    the needs are read with noise and recover on their own. The placebo (praise) must come out 0."""
    G, P_, S_ = [], [], []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        fx = OtherEffects()
        need, t = 0.0, 0
        log = []
        for k in range(6000):
            t += 1
            if need < 0.05 and rng.random() < 0.02:
                need = float(rng.uniform(0.3, 0.9))                    # an episode starts
            elif need > 0 and rng.random() < 0.01:
                need = 0.0                                             # it ends on its own
            for e in [e for e in log if e[0] == t]:
                if e[1] == "give":
                    need *= 0.3
                elif e[1] == "strike":
                    need = min(1.0, need + 0.3)
            log = [e for e in log if e[0] > t]
            read = max(0.0, need + rng.normal(0, 0.05))
            fx.observe_read(t, {"Fenn": read})
            if rng.random() < 0.15:
                verb = ("give", "praise", "strike")[int(rng.integers(0, 3))]
                fx.did("Fenn", verb, t, read)
                if verb != "praise":
                    log.append((t + int(rng.integers(1, 4)), verb))
        G.append(fx.rho["give"][0])
        P_.append(fx.rho["praise"][0])
        S_.append(fx.eta["strike"][0])
    g, p, sx = ci(G), ci(P_), ci(S_)
    TABLE["other_effects"] = {"rho give (true 0.7 of the need, minus the episodes that end anyway)": g, "rho praise (true 0)": p,
                              "eta strike (true +0.3 when it lands: about 0.3 x the chance it lands before the window closes)": sx}
    assert g[1] > 0.3 and p[1] < 0.1 and p[2] > -0.1 and p[0] < 0.2 * g[0] and sx[1] > 0.08, (g, p, sx)


def test_concerns_off_is_the_old_mind_and_the_interface_can_be_swapped():
    def sig(pol):
        _, stats = lifetime(Cast(Rules(), {"Tom": pol}), 3, 1, flags="all6", fade=0.9)
        a = pol.agents["Tom"]
        h = hashlib.sha256(np.concatenate([a.ac.U.ravel(), a.ac.W.ravel(), a.ac.c]).tobytes()).hexdigest()[:16]
        return [x["drive_error"]["Tom"] for x in stats], h, a.decisions, a.val.v
    assert sig(make_mind(1, **FIXES)) == sig(make_mind(1, **FIXES, concerns=False, cn=None, cn_things=False, b_concern=3.0))
    assert make_mind(1, **FIXES).make("Tom", "peasant").cn is None

    class Null(Concerns):
        def pull(self, target):
            return 0.0

        def expected(self, verb, target, need):
            return 0.0

        def learn(self, step):
            return 0.0

        def night(self):
            pass

        def table(self):
            return {}

        def set_table(self, table):
            pass

        def relief(self, verb, need):
            return 0.0

        def set_prior(self, verb, rho=0.0, eta=0.0, n0=3):
            pass

        def observe_read(self, t, readings):
            pass

        def did(self, target, verb, t, need):
            pass

        def flush(self):
            pass
    pol = make_mind(1, **{**FIXES, "concerns": True})
    mk = pol.make

    def swap(n, k):
        a = mk(n, k)
        a.cn = Null()
        return a
    pol.make = swap
    _, stats = lifetime(Cast(Rules(), {"Tom": pol}), 2, 1, flags=ALL6S, fade=0.9)
    assert len(stats) == 2
    TABLE["off_and_swap"] = {"explicit off equals default (drive error, weights hash, decisions, valences)": True,
                             "a null Concerns plugged in runs a lifetime": True, "plus the golden check in test_mind": "run separately"}


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
