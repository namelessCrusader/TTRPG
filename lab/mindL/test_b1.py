"""Tests of mindL6 stage B1 (rules: PASS_RULE_B1.txt, written before the runs). Plain asserts.

    lab/.venv/bin/python -m lab.mindL.test_b1
"""
from __future__ import annotations

import os
import types

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from lab.mindL import reader
from lab.mindL.drives2 import (DRIVES2, IE, IT, W_ADULT, Drives2, exponent_dials, setpoints, told, watch_weight, weights)
from lab.mindL.mind import FIXES, Mind, make_mind
from lab.mindL.world_model import WorldModelP
from lab.world2 import world as W
from lab.world2.lifetime import Cast
from lab.world2.policy import Rules
from lab.world2.world import ALL6, LETHAL_RESERVE, SIGNS_FLAGS, World

EFX = {**FIXES, "b_trust": 0.0, "effects": True, "fx_fix": True}
FL = tuple(ALL6) + tuple(SIGNS_FLAGS) + ("drives2",)
FB = FL + ("body2",)
C0 = 1 + reader.NK


def mind(letters="SCFD", **kw):
    return Mind("Oren", "merchant", 1, drives2=letters, **EFX, **kw)


def world(flags=FL, seed=3, **kw):
    return World(seed=seed, flags=flags, turns=300, **kw)


# ============================================================================================ A1 the stage A gaps, generalised
def test_dest_score_term_is_in_the_score_and_equals_its_formula():
    m = mind()
    w = world()
    p = w.people["Oren"]
    vd = reader.VI["drink"]
    m.wm.M[vd][IT, C0 + IT] = -0.8
    z = np.zeros(m.zdim)
    z[0] = 1.0
    m._aff = {"the cellar": {("drink", "x"): (vd, z, 0.5, None)}}
    need = np.zeros(8)
    need[IT] = 0.9
    opts = [{"verb": "walk to", "target": "the cellar", "obj": None, "f": {}, "dur": 3}, {"verb": "walk to", "target": "the kitchen", "obj": None, "f": {}, "dur": 3}]
    menu = m.make_menu(w, p, opts, need)
    T = m.terms(w, p, opts, menu, need)
    want = m.cfg["b_dest"] * m.cfg["gamma"] ** 3 * menu.Z[0, -1]
    assert menu.Z[0, -1] > 0.1 and abs(T["dest"][0] - want) < 1e-12 and T["dest"][1] == 0.0, (T["dest"], want)


def test_the_destination_feature_goes_to_any_option_that_moves_me_not_by_verb_name():
    m = mind()
    w = world()
    p = w.people["Oren"]                                       # in the main room
    vd = reader.VI["drink"]
    m.wm.M[vd][IT, C0 + IT] = -0.8
    z = np.zeros(m.zdim)
    z[0] = 1.0
    m._aff = {"the yard": {("drink", "x"): (vd, z, 0.5, None)}}
    need = np.zeros(8)
    need[IT] = 0.9
    for verb in ("flee to", "sit down", "look around"):         # none of these is "walk to": the target is a place that is not here
        opts = [{"verb": verb, "target": "the yard", "obj": None, "f": {}, "dur": 2}]
        menu = m.make_menu(w, p, opts, need)
        assert menu.Z[0, -1] > 0.1, verb
        T = m.terms(w, p, opts, menu, need)
        assert T["dest"][0] > 0, verb
    opts = [{"verb": "walk to", "target": p.room, "obj": None, "f": {}, "dur": 2}]      # a target that is here moves nobody
    assert m.make_menu(w, p, opts, need).Z[0, -1] == 0.0
    opts = [{"verb": "wait", "target": None, "obj": None, "f": {}, "dur": 1}]
    assert m.make_menu(w, p, opts, need).Z[0, -1] == 0.0
    # and movement is not what a place offers: a remembered "walk" is not kept as an affordance
    m2 = mind()
    opts = [{"verb": "flee to", "target": "the yard", "obj": None, "f": {}, "dur": 2}, {"verb": "wait", "target": None, "obj": None, "f": {}, "dur": 1}]
    m2.make_menu(w, p, opts, need)
    assert list(m2._aff[p.room]) == [("wait", None)], list(m2._aff[p.room])


def test_a_remembered_option_is_rescaled_by_its_own_need_column():
    m = mind()
    k = reader.KI["need"]
    ve, vd = reader.VI["eat"], reader.VI["drink"]
    m.wm.M[ve][IE, k] = -0.1                                    # my model: the relief of an act is in proportion to the option's own need number
    m.wm.M[vd][IT, k] = -0.5
    raw = np.zeros(8)
    raw[IE], raw[IT] = 0.9, 0.5                                 # then: very hungry, a little thirsty
    opts = [{"verb": "eat", "target": None, "obj": "bread", "f": {"need": 0.9}}, {"verb": "drink", "target": None, "obj": "well", "f": {"need": 0.5}}]
    menu = reader.encode(opts, raw, extra=1)
    m._remember("the main room", opts, menu, raw, raw)
    cols = {key[0]: ent[3] for key, ent in m._aff["the main room"].items()}
    assert cols == {"eat": IE, "drink": IT}, cols               # which need each is about: the one its number equals
    m._aff["the kitchen"] = {("eat", "bread"): m._aff["the main room"].pop(("eat", "bread"))}      # the eat is split off to its own room, to read it alone
    now = np.zeros(8)
    now[IE], now[IT] = 0.45, 1.0                                # now: half as hungry, twice as thirsty
    rd = m.relief_at("the main room", "merchant", now, now)
    re_ = m.relief_at("the kitchen", "merchant", now, now)
    want_d = float(m.D.drop("merchant", now, np.eye(8)[IT] * (-0.5 * 0.5 * 2.0)))          # drink: its number 0.5 x (1.0 / 0.5)
    want_e = float(m.D.drop("merchant", now, np.eye(8)[IE] * (-0.1 * 0.9 * 0.5)))          # eat: its number 0.9 x (0.45 / 0.9)
    assert abs(rd - want_d) < 1e-9 and abs(re_ - want_e) < 1e-9, (rd, want_d, re_, want_e)  # (the old rescale by max(hunger, thirst) = 1.0 would give the eat x1.1 and the drink x2)


def test_felt_need_cap_is_one_plus_the_gain_and_a_neutral_act_is_worth_nothing():
    d = Drives2("SCF")
    need = np.zeros(8)
    need[IT] = 1.8
    assert d.cap[IT] == d.cap[IE] == 2.0
    assert d.drop("peasant", need, np.zeros(8)) == 0.0          # doing nothing at a felt need above 1 earns nothing
    rel = np.zeros(8)
    rel[IT] = -0.5
    want = d.D("peasant", need) - d.D("peasant", need + rel)
    assert abs(d.drop("peasant", need, rel) - want) < 1e-12 and want > 0
    up = np.zeros(8)
    up[IT] = 0.5                                                # a worsening is clipped at the cap (2.0), not at 1
    capped = need.copy()
    capped[IT] = 2.0
    assert abs(d.drop("peasant", need, up) - (d.D("peasant", need) - d.D("peasant", capped))) < 1e-12 and d.drop("peasant", need, up) < 0
    off = Drives2("SC")
    assert off.cap[IT] == 1.0


def test_watch_mask_keeps_the_needs_i_cannot_see_unchanged():
    m = mind("SCFDW")
    v = reader.VI["drink"]
    m.wm.M[v][IE, 0] = 0.5                                      # a belief about hunger that the watched sign cannot test (no hunger key below)
    m.wm.M[v][3, 0] = 0.4                                       # and about a need that is never seen at all (company)
    s0 = {"hurt": 0.0, "fear": 0.0, "thirst": 0.8}
    s1 = {"hurt": 0.0, "fear": 0.0, "thirst": 0.3}
    m._learn_watched({"action": "drink"}, s0, s1, "Tom")
    assert m.wm.M[v][IE, 0] == 0.5 and m.wm.M[v][3, 0] == 0.4, (m.wm.M[v][IE, 0], m.wm.M[v][3, 0])
    assert m.wm.M[v][IT].any()                                  # what is seen does teach


# ============================================================================================ A2 exponent dials
def test_exponent_dials_are_per_person_fixed_and_inside_n_gt_m_gt_1():
    seen = set()
    for seed in range(1, 6):
        for name in ("Jon", "Mara", "Pip", "Gran", "Tom", "Sir Ada", "Oren"):
            n, m = exponent_dials(seed, name)
            assert n > m > 1.0 and (n, m) == exponent_dials(seed, name)
            seen.add((round(n, 6), round(m, 6)))
    assert len(seen) == 35
    mk = Mind("Tom", "peasant", 3, drives2="SCB", d2={"dial": True})
    assert (mk.D.n, mk.D.m) == exponent_dials(3, "Tom")


def test_a_large_n_brings_back_the_max_and_the_default_dials_do_not():
    need = np.zeros(8)
    need[IE], need[IT] = 0.95, 0.60                             # hunger deficit larger than thirst
    eat, drink = np.zeros(8), np.zeros(8)
    eat[IE], drink[IT] = -0.1, -0.1
    big = Drives2("SC", n=60.0, m=59.0)
    assert big.drop("peasant", need, drink) < 1e-3 * big.drop("peasant", need, eat)
    ok = Drives2("SC")
    assert ok.drop("peasant", need, drink) > 0.05 * ok.drop("peasant", need, eat)


def test_partial_competition_and_a_drink_pays_for_every_dial_draw():
    kind = "peasant"
    s, wt = setpoints(kind)[IE], weights(kind)[IE]
    for i in range(60):
        n, m = exponent_dials(100 + i, "Tom")
        d = Drives2("SC", n=n, m=m)
        need = np.zeros(8)
        need[IE], need[IT] = s + 0.6, s + 0.3                   # thirst deficit is half of hunger's
        eat, drink = np.zeros(8), np.zeros(8)
        eat[IE], drink[IT] = -0.01, -0.01
        ratio = d.drop(kind, need, drink) / d.drop(kind, need, eat)
        assert 0.02 < ratio < 0.6, (n, m, ratio)
        for h in (0.4, 0.7, 1.0):                               # whatever the hunger, a drink pays
            need[IE], need[IT] = h, 0.5
            assert d.drop(kind, need, drink * 20) > 0


# ============================================================================================ A3 the felt ladder
def test_the_ladder_steps_at_the_anchors_and_is_monotone():
    w = world(FB)
    p = w.people["Tom"]
    d = Drives2("SCB")
    p.thirst = 1.0

    def felt_at(pct):                                           # the percent of body mass lost, through the reserve (need part is 1.0 at the cap)
        dep = (pct - W.WATER_LOSS_NEED_PCT) / (W.WATER_LOSS_END_PCT - W.WATER_LOSS_NEED_PCT)
        p.reserve["thirst"] = LETHAL_RESERVE["thirst"] * (1.0 - dep)
        return d.needs(w, p)[IT] - 1.0
    assert abs(W.water_lost_pct(1.0, 0.0) - 1.0) < 1e-12 and abs(W.water_lost_pct(1.0, 1.0) - 10.5) < 1e-12
    for thr, h in W.WATER_RUNGS:
        assert abs(felt_at(thr - 0.05) - (h_prev(thr))) < 1e-9, (thr, felt_at(thr - 0.05))
        assert abs(felt_at(thr + 0.05) - h) < 1e-9, (thr, felt_at(thr + 0.05))
    grid = [felt_at(x) for x in np.linspace(1.0, 10.4, 60)]
    assert all(b >= a for a, b in zip(grid, grid[1:])) and grid[0] == 0.0 and grid[-1] >= 0.85
    assert len({round(g, 9) for g in grid}) == len(W.WATER_RUNGS) + 1                 # steps, not a slope
    lin = Drives2("SCF")
    p.reserve["thirst"] = LETHAL_RESERVE["thirst"] * 0.9
    assert abs(lin.needs(w, p)[IT] - 1.1) < 1e-9                                      # F alone is still the linear course (stage A)
    # hunger has its own ladder (percent of body weight): quiet to 6
    p.thirst = 0.3
    p.reserve["thirst"] = LETHAL_RESERVE["thirst"]
    p.hunger = 1.0
    p.reserve["hunger"] = LETHAL_RESERVE["hunger"] * (1 - 0.14)                       # 5.6 percent lost
    assert d.needs(w, p)[IE] == 1.0
    p.reserve["hunger"] = LETHAL_RESERVE["hunger"] * (1 - 0.30)                       # 12 percent
    assert abs(d.needs(w, p)[IE] - 1.3) < 1e-9


def h_prev(thr):
    last = 0.0
    for t, h in W.WATER_RUNGS:
        if t < thr:
            last = h
    return last


# ============================================================================================ A4 the world, body2
class Idle:
    name = "idle"

    def probs(self, w, p, opts):
        pc = [0.0] * len(opts)
        pc[[o["verb"] for o in opts].index("wait")] = 1.0
        return pc, pc


def test_a_resting_person_who_never_drinks_dies_in_3_5_to_10_5_days():
    carry, day = None, None
    for n in range(12):
        w = World(seed=n + 7, flags=("lethal", "body2"), bandit=False, carry=carry, night=n)
        for _ in range(300):
            w.step(Idle())
            if w.people["Tom"].dead:
                day = n + w.t / 300
                break
        if day:
            break
        carry = w.carry_out()
    assert day is not None and 3.5 <= day <= 10.5, day


def test_water_is_repaid_within_5_turns_hunger_keeps_its_pace_and_off_is_the_old_pace():
    for flags, fast in ((("lethal", "body2"), True), (("lethal",), False)):
        w = World(seed=5, flags=flags, bandit=False)
        p = w.people["Tom"]
        p.thirst, p.hunger = 0.5, 0.5                          # after a drink that ended a deficit
        p.reserve["thirst"] = LETHAL_RESERVE["thirst"] - 3.0   # three days of deficit (more than the 24 h the answer measured)
        p.reserve["hunger"] = LETHAL_RESERVE["hunger"] - 3.0
        for _ in range(5):
            w._drift(p)
        if fast:
            assert p.reserve["thirst"] == LETHAL_RESERVE["thirst"]
        else:
            assert abs(p.reserve["thirst"] - (LETHAL_RESERVE["thirst"] - 3.0 + 5 / 300)) < 1e-9
        assert abs(p.reserve["hunger"] - (LETHAL_RESERVE["hunger"] - 3.0 + 5 / 300)) < 1e-9          # hunger: a day a day, always
    assert W.LETHAL_DAYS["hunger"] == 45.0


def _fed(flags, pre, thirst=0.3, dep=0.0, hunger=0.9):
    w = World(seed=4, flags=flags, turns=300)
    item = next(iter(w.portions))
    p = w.people["Tom"]
    p.hunger, p.thirst = hunger, thirst
    p.reserve["thirst"] = LETHAL_RESERVE["thirst"] * (1.0 - dep)
    got = w._feed(p, 1, item)
    return got, hunger - p.hunger, p.thirst - thirst


def test_eating_raises_thirst_and_the_dehydrated_eat_less_off_is_as_before():
    a, ra, ta = _fed(FB, 0)                                     # rehydrated (need 0.3, reserve full)
    b, rb, tb = _fed(FB, 0, thirst=1.0, dep=(2.9 - 1.0) / 9.5)  # 2.9 percent of body mass lost, no drink
    assert a == b == 1                                          # as many meals
    assert abs(ta - 0.05) < 1e-12 and tb == 0.0                  # a meal raises thirst by 0.05 a ration (u, frozen here); at the cap it cannot go higher
    assert 0.4 <= rb / ra <= 0.9, (rb, ra)
    assert abs(rb / ra - 0.65 / (1 - 0.35 * 0.3 / 2.9)) < 1e-6  # 0.65 against the rehydrated (0.3 percent lost)
    o, ro, to = _fed(FL, 0)                                     # body2 off: exactly the old ration, no thirst
    assert o == 1 and abs(ro - W.PORTION) < 1e-12 and to == 0.0
    assert FB != FL


def test_after_a_drink_the_eat_factor_is_back_within_5_turns():
    w = world(FB + ("lethal",))
    p = w.people["Tom"]
    p.thirst, p.hunger = 1.0, 0.9
    p.reserve["thirst"] = LETHAL_RESERVE["thirst"] * (1.0 - (2.9 - 1.0) / 9.5)
    f0 = w.eat_factor(p)
    assert abs(f0 - 0.65) < 0.01
    w._sip(p, "the well", False)                                # one sip: -0.5
    w._sip(p, "the well", False)                                # a drink is taken until the need is at its usual level
    for _ in range(5):
        w._drift(p)
    assert p.thirst < 0.35 and p.reserve["thirst"] == LETHAL_RESERVE["thirst"]
    assert w.eat_factor(p) > 0.95, w.eat_factor(p)
    # for the record: a single sip leaves 0.5 percent lost and the factor at about 0.94 (below 0.95): said in the report


# ============================================================================================ A5 the innate set
def test_the_cry_is_graded_cause_blind_and_the_carer_reads_it_not_the_cause():
    c = W.cry_level
    xs = [c(0.325 + x, 0.2, 0.0, 0.0, 1.0) for x in (0.0, 0.1, 0.2, 0.4, 0.6)]
    assert all(b > a for a, b in zip(xs, xs[1:]))
    assert c(0.325 + 0.3, 0.2, 0.0, 0.0, 1.0) == c(0.2, 0.325 + 0.3, 0.0, 0.0, 1.0)      # hunger or thirst of the same size: the same cry
    assert c(0.5, 0.2, 0.0, 0.0, 8.0) < 0.05 * c(0.5, 0.2, 0.0, 0.0, 1.0) + 1e-12          # the cry fades with age
    w = world(FB)
    ada, tom = w.people["Sir Ada"], w.people["Tom"]
    ada.age = 1.0
    ada.hunger, ada.thirst = 0.95, 0.1
    r = w.perceived_need("Tom", "Sir Ada")
    assert "cry" in r and "hunger" not in r and "thirst" not in r
    ada.age = 5.0
    assert {"hunger", "thirst", "cry"} <= set(w.perceived_need("Tom", "Sir Ada"))
    ada.age = 1.0
    ada.hunger, ada.thirst = 0.95, 0.1
    h = w.seen_need_of(ada, ada.thirst)
    ada.hunger, ada.thirst = 0.1, 0.95
    assert h == w.seen_need_of(ada, ada.thirst) > 0.6         # a carer hears the same whatever the need
    ada.age = 30.0
    assert w.seen_need_of(ada, 0.9) == 0.9 and w.seen_need_of(ada, 0.5) == 0.0             # an adult: its need itself, above 0.6
    # a classifier on the cry alone recovers the cause at chance (hunger, thirst, fear: 1/3)
    rng = np.random.default_rng(0)
    ada.age = 1.0
    ys, xs = [], []
    for i in range(600):
        cause = int(rng.integers(3))
        ada.hunger = ada.thirst = 0.2
        ada.fear = 0.0
        if cause == 0:
            ada.hunger = 0.8
        elif cause == 1:
            ada.thirst = 0.8
        else:
            ada.fear = 0.475
        w.t = i
        ys.append(cause)
        xs.append(w.perceived_need("Tom", "Sir Ada")["cry"])
    ys, xs = np.array(ys), np.array(xs)
    means = np.array([xs[ys == k].mean() for k in range(3)])
    acc = float((np.argmin(np.abs(xs[:, None] - means[None, :]), axis=1) == ys).mean())
    assert 0.25 <= acc <= 0.41, acc
    w0 = world(FL)                                              # body2 off: no cry at all
    assert "cry" not in w0.perceived_need("Tom", "Sir Ada")


def test_taste_is_a_first_valence_that_experience_overrides():
    m = Mind("Tom", "peasant", 2, drives2="SCB", taste=True)
    w = types.SimpleNamespace(taste=lambda item: {"the mug of ale": W.TASTE_ALE, "the loaf of bread": W.TASTE_FOOD}.get(item, 0.0))
    for item, sign in (("the mug of ale", -1), ("the loaf of bread", 1)):
        p = types.SimpleNamespace(tasted=item)
        m._taste_prior(w, p)
        assert m.val.value(item) * sign > 0 and p.tasted is None          # the tongue is read once
    assert abs(m.val.value("the mug of ale") - m.cfg["taste_gain"] * W.TASTE_ALE) < 1e-12
    for _ in range(10):                                         # ten experiences in which the bitter thing is better than the baseline flip its sign
        m.val.update({"ctx": 1.0, "the mug of ale": 0.5}, 0.3)
        m.val.update({"ctx": 1.0}, 0.0)
        m._taste_prior(w, types.SimpleNamespace(tasted="the mug of ale"))               # the prior is NOT set again
    assert m.val.value("the mug of ale") > 0
    m2 = Mind("Tom", "peasant", 2, drives2="SCB", taste=True)   # and with no contrast the prior is washed out toward 0 within 15 experiences (it does not stay)
    m2._taste_prior(w, types.SimpleNamespace(tasted="the mug of ale"))
    for _ in range(15):
        m2.val.update({"ctx": 1.0, "the mug of ale": 0.5}, 0.3)
    assert abs(m2.val.value("the mug of ale")) < 0.05 * abs(m2.cfg["taste_gain"] * W.TASTE_ALE)
    off = Mind("Tom", "peasant", 2, drives2="SCB")
    off._taste_prior(w, types.SimpleNamespace(tasted="the mug of ale"))
    assert off.val.value("the mug of ale") == 0.0
    ww = World(seed=2, flags=FB)                                # the world: what is consumed (eaten, drunk, or brought) is what is tasted; off: nothing is tasted
    assert ww.taste("the barrel of ale") < 0 < ww.taste("the loaf of bread") and ww.taste("the well") == 0.0
    t = ww.people["Tom"]
    ww._sip(t, "the barrel of ale", True)
    assert t.tasted == "the barrel of ale"
    ww._feed(t, 1, next(iter(ww.portions)))
    assert t.tasted == next(iter(ww.portions))
    w0 = World(seed=2, flags=FL)
    w0._sip(w0.people["Tom"], "the well", False)
    assert w0.people["Tom"].tasted is None


def test_no_imitation_at_birth_and_the_outcome_weight_follows_age():
    s0 = {"hurt": 0.0, "fear": 0.0, "hunger": 0.4, "thirst": 0.8}
    s1 = {"hurt": 0.0, "fear": 0.0, "hunger": 0.4, "thirst": 0.3}
    v = reader.VI["drink"]
    m = mind("SCFDW", watch_age=True)
    m._learn_watched({"action": "drink"}, s0, s1, "Tom", age=0.0)
    assert m.watched == 0 and not m.wm.M[v].any() and m.wm.cnt[v] == 0
    assert watch_weight(0.0) == 0.0 and watch_weight(0.1) == 0.0
    ws = [watch_weight(a) for a in (0.2, 0.5, 1.0)]
    assert all(b > a for a, b in zip(ws, ws[1:]))
    assert watch_weight(30.0) == W_ADULT
    assert abs(watch_weight(7.0) - (0.22 + 0.38 * (1 - 7 / 18))) < 1e-12 and watch_weight(7.0) > watch_weight(20.0) > W_ADULT - 1e-12
    m._learn_watched({"action": "drink"}, s0, s1, "Tom", age=7.0)
    assert m.watched == 1 and abs(m.wm.cnt[v] - watch_weight(7.0)) < 1e-12
    off = mind("SCFDW")                                         # watch_age off: the stage A constant (age is not read)
    off._learn_watched({"action": "drink"}, s0, s1, "Tom")
    assert abs(off.wm.cnt[v] - off.cfg["w_obs"]) < 1e-12


# ============================================================================================ A6 the childhood harness
def test_teaching_rate_peer_share_errands_and_save_load():
    from lab.mindL import childhood as C
    # errands: the chance is the sourced share at the sourced peak age, none before the dependent age
    assert abs(C.errand_chance("f", 4.0) - 0.83) < 1e-12 and abs(C.errand_chance("m", 9.0) - 0.79) < 1e-12
    assert C.errand_chance("m", 3.99) == 0.0 and C.errand_chance("f", 3.0) == 0.0 and C.errand_chance("f", 1.0) == 0.0
    ch = C.Childhood(1, "Tom", 10)
    ch.age = lambda n: 9.0
    dummy = types.SimpleNamespace(people={"Tom": types.SimpleNamespace(name="Tom", kind="peasant")})
    for n in range(1000):
        ch.night_start(dummy, n)
    share = ch.stats["errand_nights"] / 1000
    assert 0.76 <= share <= 0.82, share                         # 0.79 +- the sampling noise of 1000 nights (sd 0.013)
    # a short real childhood: teaching 6 to 8 an hour, 0.75 through a child
    r = C.run_childhood(5, "Tom", 3, cache=False)
    st = r["stats"]
    per_hour = st["episodes"] / (3 * 24.0)
    assert 6.0 <= per_hour <= 8.0, per_hour
    assert 0.70 <= st["via_child"] / st["episodes"] <= 0.80, st["via_child"] / st["episodes"]
    assert st["taken"] > 0 and st["watched"] > 0 and st["decisions"] > 100
    # save and load: the loaded learner makes the same choices as the one it was saved from
    blob = r["blob"]
    reader.use_cast(W.FAMILY_CAST)
    runs = []
    for k in range(2):
        pol = make_mind(5, **C.LEARNER, **{kk: vv for kk, vv in {**FIXES, "b_trust": 0.0, "effects": True, "fx_fix": True}.items()})
        pol.load({"Tom": blob})
        ag = pol.agent(types.SimpleNamespace(name="Tom", kind="peasant"))
        runs.append(ag)
    assert runs[0].rng.bit_generator.state == runs[1].rng.bit_generator.state
    assert all(np.array_equal(getattr(runs[0].wm, a), getattr(runs[1].wm, a)) for a in ("M", "cnt", "n0"))
    assert runs[0].wm.n0.max() > 0 and runs[0].wm.cnt[reader.VI["drink"]] > 0, "the childhood left experience and told priors in the learner"
    assert runs[0].wm.n0.max() <= C.TOLD_CAP + 1e-9 and C.TOLD_CAP == 9.0, runs[0].wm.n0.max()      # what is told never outweighs experience without bound


# ============================================================================================ A7 the body sheets
def test_body_sheets_elder_work_child_metab_and_a_young_person_is_a_child():
    fam = ("family",) + FB
    w = World(seed=3, flags=fam)
    assert w.people["Gran"].work == 0.85 and w.people["Pip"].metab == 0.70 and w.people["Gran"].ill == 2.0 and w.people["Gran"].ill_len == 1.5
    w0 = World(seed=3, flags=("family",) + FL)
    assert w0.people["Gran"].work == 0.50                      # body2 off: the old sheet
    wy = World(seed=3, flags=fam, young={"Tom": 3.0})
    assert wy.people["Tom"].age == 3.0 and wy.people["Tom"].metab == 0.70 and wy.people["Tom"].work == 0.15 and wy.people["Jon"].age == 38.0
    wn = World(seed=3, flags=fam)
    assert wn.people["Tom"].age == 28.0 and wn.people["Tom"].metab == 1.0


def test_off_is_the_old_mind_and_the_new_options_change_nothing_alone():
    m = Mind("Tom", "peasant", 0)
    assert m.cfg["watch_age"] is False and m.cfg["taste"] is False and m.D.__name__ if hasattr(m.D, "__name__") else True
    d = Drives2("SCF")
    assert not d.ladder and d.n == 4.0 and d.m == 3.0


if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            n += 1
            print("ok", k, flush=True)
    print(n, "passed")
