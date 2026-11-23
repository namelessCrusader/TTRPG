"""Tests of the mindL6 stage A mechanism (lab/mindL/drives2.py, mind option drives2): one group of tests per piece. Plain asserts.

    lab/.venv/bin/python -m lab.mindL.test_drives2
"""
from __future__ import annotations

import os
import random

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from lab.mindL import drives as H
from lab.mindL import reader
from lab.mindL.drives2 import DRIVES2, IE, IT, Drives2, told
from lab.mindL.mind import FIXES, Mind, make_mind
from lab.mindL.world_model import WorldModel, WorldModelP
from lab.world2.policy import Rules
from lab.world2.world import ALL6, LETHAL_FLAGS, LETHAL_RESERVE, SIGNS_FLAGS, World

FL = tuple(ALL6) + tuple(SIGNS_FLAGS) + ("drives2",) + tuple(LETHAL_FLAGS)
EFX = {**FIXES, "b_trust": 0.0, "effects": True, "fx_fix": True}


def world(seed=3, flags=FL):
    return World(seed=seed, flags=flags, turns=300)


def person(w, name="Tom"):
    return w.people[name]


# ---------------------------------------------------------------------------------------------- 1 separate drives
def test_thirst_is_its_own_drive_and_a_drink_pays_when_hunger_is_higher():
    w = world()
    p = person(w)
    p.hunger, p.thirst = 0.9, 0.7
    old = H.needs(w, p)[0]
    assert old == 0.9                                   # the old energy is the max: thirst is hidden behind hunger
    d2 = Drives2("S" + "C")
    n = d2.needs(w, p)
    assert n[IE] == 0.9 and n[IT] == 0.7 and len(n) == len(DRIVES2) == 8
    e0 = d2.error(w, p)
    p.thirst = 0.2                                      # a drink
    e1 = d2.error(w, p)
    assert e0 - e1 > 0.01, (e0, e1)
    p.thirst = 0.7
    o0 = w.drive_error(p)
    p.thirst = 0.2
    assert abs(o0 - w.drive_error(p)) < 1e-12           # the old drive: the same drink earned exactly nothing


def test_thirst_weight_and_set_point_are_hungers_for_each_kind():
    from lab.world2.world import NATURE
    from lab.mindL.drives2 import setpoints, weights
    for k in NATURE:
        assert weights(k)[IT] == weights(k)[IE] and setpoints(k)[IT] == setpoints(k)[IE]


# ---------------------------------------------------------------------------------------------- 2 convex drive
def test_exponents_obey_keramati_gutkin_condition():
    d = Drives2("SC")
    assert d.n > d.m > 1
    try:
        Drives2("SC", n=2.0, m=3.0)
        raise SystemExit("n < m must be refused")
    except AssertionError:
        pass


def test_alliesthesia_relief_is_worth_more_at_greater_deficit():
    d = Drives2("SC")
    for kind in ("peasant", "innkeeper", "child"):
        lo = np.zeros(8)
        hi = np.zeros(8)
        lo[IT], hi[IT] = 0.45, 0.95
        rel = np.zeros(8)
        rel[IT] = -0.15
        a, b = d.drop(kind, lo, rel), d.drop(kind, hi, rel)
        assert b > 1.3 * a > 0, (kind, a, b)
    lin = Drives2("S")                                  # without C: the same relief is worth the same
    assert abs(lin.drop("peasant", lo, rel) - lin.drop("peasant", hi, rel)) < 1e-9


def test_the_larger_deficit_wins_when_two_compete():
    d = Drives2("SC")
    need = np.zeros(8)
    need[IE], need[IT] = 0.9, 0.55                      # very hungry, a little thirsty
    g = d.grad("peasant", need)
    assert g[IE] > 2 * g[IT] > 0, g
    rel_eat, rel_drink = np.zeros(8), np.zeros(8)
    rel_eat[IE], rel_drink[IT] = -0.25, -0.25
    assert d.drop("peasant", need, rel_eat) > d.drop("peasant", need, rel_drink)
    need[IE], need[IT] = 0.55, 0.9                      # and the other way round
    assert d.drop("peasant", need, rel_drink) > d.drop("peasant", need, rel_eat)


# ---------------------------------------------------------------------------------------------- 3 felt depletion
def test_felt_depletion_grows_with_the_reserve_lost_and_reads_only_the_body():
    w = world()
    p, q = person(w), person(w, "Oren")
    d = Drives2("SCF")
    p.thirst = 1.0
    out = []
    for frac in (1.0, 0.75, 0.5, 0.25, 0.05):
        p.reserve["thirst"] = LETHAL_RESERVE["thirst"] * frac
        out.append(d.needs(w, p)[IT])
    assert all(b > a for a, b in zip(out, out[1:])) and out[0] == 1.0 and out[-1] > 1.8, out
    s0 = d.state(w, p)
    q.reserve["thirst"] = 0.01                           # someone else's body: nothing changes for me
    q.thirst = 1.0
    assert np.array_equal(s0, d.state(w, p))
    # the channel is in the state and in the drive
    assert s0[-2] > 0.9 and s0[-1] == 0.0
    e0 = d.error(w, p)
    p.reserve["thirst"] = LETHAL_RESERVE["thirst"] * 0.5
    assert d.error(w, p) < e0                            # less depleted, less pressure
    off = Drives2("SC")
    assert off.needs(w, p)[IT] == 1.0                    # without F the need stops at its cap
    p.reserve["thirst"] = LETHAL_RESERVE["thirst"]
    p.thirst = 0.4
    assert d.needs(w, p)[IT] == 0.4                      # a body with its reserve is not "felt"


def test_felt_urgency_beats_the_unfelt_drive_near_death():
    d, o = Drives2("SCF"), Drives2("SC")
    w = world()
    p = person(w)
    p.thirst = 1.0
    p.reserve["thirst"] = LETHAL_RESERVE["thirst"] * 0.15
    rel = np.zeros(9 - 1)
    rel[IT] = -0.5
    assert d.drop(p.kind, d.needs(w, p), rel) > 1.2 * o.drop(p.kind, o.needs(w, p), rel)       # the same drink is worth more
    assert d.D(p.kind, d.needs(w, p)) > 2.0 * o.D(p.kind, o.needs(w, p))                    # and the pressure is greater


# ---------------------------------------------------------------------------------------------- 4 own-act priors, being told
def _wm(n0=0.0):
    wm = WorldModelP(reader.NV, 8, 1 + reader.NK + 8, 0.2)
    return wm


def _z(thirst):
    z = np.zeros(1 + reader.NK + 8)
    z[0] = 1.0
    z[1 + reader.NK + IT] = thirst
    return z


def test_prior_is_used_then_overridden_by_experience():
    v = reader.VI["drink"]
    wm = _wm()
    wm.set_prior(v, {}, {IT: -0.8}, 3.0, reader.NK)
    assert abs((wm.M[v] @ _z(0.8))[IT] - (-0.64)) < 1e-9          # the told effect is what it predicts
    w0 = []
    for n in range(40):                                            # the world says: a drink does nothing
        z = _z(0.8)
        wm.update(v, z, np.zeros(8))
        w0.append(abs((wm.M[v] @ z)[IT]))
    assert w0[-1] < 0.05 * 0.64, w0[-1]                             # experience overrides it
    # the prior holds longer the larger its pseudo-count
    a, b = _wm(), _wm()
    a.set_prior(v, {}, {IT: -0.8}, 1.0, reader.NK)
    b.set_prior(v, {}, {IT: -0.8}, 10.0, reader.NK)
    for wm in (a, b):
        for _ in range(3):
            wm.update(v, _z(0.8), np.zeros(8))
    assert abs((b.M[v] @ _z(0.8))[IT]) > abs((a.M[v] @ _z(0.8))[IT])


def test_no_prior_is_the_old_world_model_bit_for_bit():
    old, new = WorldModel(reader.NV, 8, 1 + reader.NK + 8, 0.2), WorldModelP(reader.NV, 8, 1 + reader.NK + 8, 0.2)
    rng = np.random.default_rng(1)
    for _ in range(50):
        v, z, d = int(rng.integers(0, reader.NV)), rng.random(1 + reader.NK + 8), rng.random(8) - 0.5
        old.update(v, z, d)
        new.update(v, z, d)
    assert np.array_equal(old.M, new.M)


def test_told_is_lossy_and_lighter_than_experience():
    v = reader.VI["drink"]
    sp = _wm()
    for _ in range(30):
        sp.update(v, _z(0.8), np.eye(8)[IT] * -0.5)
    cols = [1 + reader.NK + d for d in range(8)]
    rng = np.random.default_rng(5)
    got, diffs = 0, []
    for k in range(200):
        ls = _wm()
        r = told(sp, ls, [v], rng, cols)
        got += len(r)
        if r:
            diffs.append(np.abs(ls.M[v] - sp.M[v]).max())
    assert 0.4 < got / 200 < 0.8 and min(diffs) > 0, (got, min(diffs))        # taken up with a chance below 1; never exactly lossless
    assert ls.n0[v] < sp.cnt[v]                                                # a smaller weight than the teller's own experience
    ig = _wm()
    assert told(ig, _wm(), [v], rng, cols) == []                               # one who knows nothing passes on nothing


# ---------------------------------------------------------------------------------------------- 5 learning by watching
def _mind(letters="SCFDW", **kw):
    reader.use_cast(__import__("lab.world2.world", fromlist=["CAST"]).CAST)
    return Mind("Oren", "merchant", 1, drives2=letters, **EFX, **kw)


def test_watching_updates_my_model_at_a_weight_below_my_own():
    s0 = {"hurt": 0.0, "fear": 0.0, "hunger": 0.4, "thirst": 0.8}
    s1 = {"hurt": 0.0, "fear": 0.0, "hunger": 0.4, "thirst": 0.3}
    e = {"action": "drink"}
    own, seen = _mind(), _mind()
    v = reader.VI["drink"]
    z = np.zeros(own.zdim)
    z[0] = 1.0
    z[1 + reader.NK + IT] = 0.8
    own.wm.update(v, z, np.eye(8)[IT] * -0.5)
    seen._learn_watched(e, s0, s1, "Tom")
    a, b = np.abs(own.wm.M[v][IT]).sum(), np.abs(seen.wm.M[v][IT]).sum()
    assert 0 < b < a, (a, b)
    assert b > 0.2 * a * 0.3                                             # it is w_obs of it, not nothing
    assert seen.watched == 1 and seen.wm.cnt[v] == seen.cfg["w_obs"]
    kin = _mind(copy_kin=2.0)                                            # the dial: copy kin more (khat from the concern rows)
    assert kin.cfg["copy_kin"] == 2.0
    off = _mind("SCFD")
    assert not off.use_watch


def test_watching_in_the_world_teaches_a_mind_that_never_drank():
    """A mind that has not drunk, in a room with a person who drinks: its model of drink gets the sign of the relief (integration)."""
    from lab.world2.lifetime import night_seed
    reader.use_cast(__import__("lab.world2.world", fromlist=["CAST"]).CAST)
    pol = make_mind(1, drives2="SCFDW", **EFX)
    seen = 0
    for n in range(3):
        w = World(seed=night_seed(1, n), flags=tuple(ALL6) + tuple(SIGNS_FLAGS) + ("drives2",), turns=300, night=n)
        for _ in range(300):
            w.step(pol)
        pol.night_end(w, n)
    seen = sum(a.watched for a in pol.agents.values())
    assert seen > 0, "no person was ever seen to act with a change in their signs"
    v = reader.VI["drink"]
    assert any(a.wm.cnt[v] > 0 for a in pol.agents.values())


# ---------------------------------------------------------------------------------------------- 6 need by destination
def test_thirst_above_hunger_does_not_leak_into_the_energy_need():
    w = world()
    p = person(w)
    p.hunger, p.thirst = 0.2, 0.9
    n = Drives2("SC").needs(w, p)
    assert n[IE] == 0.2 and n[IT] == 0.9                # a mutation folding thirst into energy (max) shows here


def test_walk_option_carries_destination_relief_in_the_menu():
    m = _mind("SCFD")
    w = World(seed=3, flags=tuple(ALL6) + tuple(SIGNS_FLAGS) + ("drives2",), turns=300)
    p = w.people["Oren"]
    c0 = 1 + reader.NK
    vd = reader.VI["drink"]
    m.wm.M[vd][IT, c0 + IT] = -0.8
    z = np.zeros(m.zdim)
    z[0] = 1.0
    m._aff = {"the cellar": {("drink", "x"): (vd, z, 0.5)}}
    need = np.zeros(8)
    need[IT] = 0.9
    opts = [{"verb": "walk to", "target": "the cellar", "f": {}}, {"verb": "walk to", "target": "the kitchen", "f": {}}]
    menu = m.make_menu(w, p, opts, need)
    assert menu.Z[0, -1] > 0.1 and menu.Z[1, -1] == 0.0, menu.Z[:, -1]


def test_walk_carries_the_relief_my_model_expects_at_the_destination_for_any_need():
    m = _mind("SCFD")
    c0 = 1 + reader.NK
    vd, vs = reader.VI["drink"], reader.VI["sleep"]
    m.wm.M[vd][IT, c0 + IT] = -0.8                      # my model: a drink takes thirst down in proportion to the thirst
    m.wm.M[vs][5, c0 + 5] = -0.8                        # a sleep takes rest down in proportion
    zd, zs = np.zeros(m.zdim), np.zeros(m.zdim)
    zd[0] = zs[0] = 1.0
    m._aff = {"the cellar": {("drink", "x"): (vd, zd, 0.5)}, "the main room": {("sleep", "y"): (vs, zs, 0.5)}, "the kitchen": {}}
    need = np.zeros(8)
    need[IT] = 0.9
    cell, main, kit = (m.relief_at(r, "merchant", need) for r in ("the cellar", "the main room", "the kitchen"))
    assert cell > 0.1 and main == 0.0 and kit == 0.0, (cell, main, kit)           # thirsty: the water room
    need = np.zeros(8)
    need[5] = 0.9
    cell, main = (m.relief_at(r, "merchant", need) for r in ("the cellar", "the main room"))
    assert main > 0.1 and cell == 0.0, (cell, main)                               # tired: the other room, the same law
    need = np.zeros(8)
    assert m.relief_at("the cellar", "merchant", need) == 0.0                     # no need, no relief
    # a mind that has never learned what a drink does expects nothing there
    assert _mind("SCFD").relief_at("the cellar", "merchant", np.eye(8)[IT] * 0.9) == 0.0


def test_the_walk_option_gets_the_feature_and_the_term():
    from lab.world2.lifetime import night_seed
    reader.use_cast(__import__("lab.world2.world", fromlist=["CAST"]).CAST)
    pol = make_mind(2, drives2="SCFD", **EFX)
    w = World(seed=night_seed(2, 0), flags=tuple(ALL6) + tuple(SIGNS_FLAGS) + ("drives2",), turns=300)
    for _ in range(120):
        w.step(pol)
    a = pol.agents["Oren"]
    assert a.last is not None and "dest" in a.last["terms"]
    assert a.zdim == 1 + reader.NK + 8 + 1


# ---------------------------------------------------------------------------------------------- 7 save and load
def test_save_load_round_trip_gives_identical_choices():
    from lab.world2.lifetime import night_seed
    reader.use_cast(__import__("lab.world2.world", fromlist=["CAST"]).CAST)
    flags = tuple(ALL6) + tuple(SIGNS_FLAGS) + ("drives2",)
    pol = make_mind(4, drives2="SCFDW", concerns="cn2", **{k: v for k, v in EFX.items()}, wm_prior={"drink": dict(slope={"thirst": -0.5}, n0=2.0)})
    for n in range(2):
        w = World(seed=night_seed(4, n), flags=flags, turns=300, night=n)
        for _ in range(300):
            w.step(pol)
        pol.night_end(w, n)
    blobs = pol.save()
    assert set(blobs) >= {"Tom", "Oren"}
    # continue the original on a fixed night; load the saved learner into a new policy and run the same night
    runs = []
    for k in range(2):
        p2 = pol if k == 0 else make_mind(4, drives2="SCFDW", concerns="cn2", **EFX, wm_prior={"drink": dict(slope={"thirst": -0.5}, n0=2.0)})
        if k == 1:
            p2.load(blobs)
        w = World(seed=night_seed(4, 7), flags=flags, turns=300, night=7)
        for _ in range(250):
            w.step(p2)
        runs.append([(e["t"], e["actor"], e["action"], e.get("target"), e.get("obj")) for e in w.log])
    assert len(runs[0]) > 100 and runs[0] == runs[1]


# ---------------------------------------------------------------------------------------------- 8 concern saturation
def _ties(rel):
    from lab.mindL.concerns2 import KinConcerns, _Row2
    cn = KinConcerns(name="x", seed=0, dial=dict(cap=1.0, lr=1.0, fade=1.0), params={"rel": rel})
    for T, n_obs, n_ej in (("A", 300, 12), ("B", 100, 6), ("C", 10, 1)):
        r = cn.rows[T] = _Row2()
        r.hrs, r.ky = 2000.0, 0.0                          # the same time together: only the care seen and paid for differ
    cn.rows["carer"] = _Row2()
    cn.rows["carer"].e = 1.0                               # someone I care about fully: the care cue counts in full
    for T, n_obs, n_ej in (("A", 300, 12), ("B", 100, 6), ("C", 10, 1)):
        for _ in range(n_obs):
            cn.observe_care("carer", T)
        cn.rows[T].ejn = float(n_ej)
    return cn


def test_relative_grading_keeps_ties_ordered_where_the_absolute_one_saturates():
    ab, rl = _ties(False), _ties(True)
    wa, wr = ab.table(), rl.table()
    # absolute: effort and cue are near their ceilings for A and B alike, little to tell them apart
    assert wa["A"] - wa["B"] < 0.4 * (wr["A"] - wr["B"]) + 0.05, (wa, wr)
    assert wr["A"] > wr["B"] > wr["C"] > 0, wr
    sa, sr = ab.state(), rl.state()
    assert sr["B"]["EJ"] < 0.7 * sr["A"]["EJ"] and sa["B"]["EJ"] == sa["A"]["EJ"] == 1.0, (sa["B"]["EJ"], sr["B"]["EJ"])
    assert sr["A"]["khat"] > sr["B"]["khat"] > sr["C"]["khat"], sr
    # the strongest tie keeps its full effect (the anchors are as before)
    assert abs(sr["A"]["EJ"] - sa["A"]["EJ"]) < 1e-12 and abs(sr["A"]["khat"] - sa["A"]["khat"]) < 1e-9


def test_rel_off_is_mindL5_and_coarse_agrees_with_the_same_law():
    from lab.mindL.concerns2 import CN2, _Row2, coarse_advance
    off = _ties(False)
    assert all(abs(off.rows[T].kn - n) < 1e-9 for T, n in (("A", 300), ("B", 100), ("C", 10)))   # kn only counts: weights unchanged
    rows = {"carer": _Row2()}
    rows["carer"].e = 1.0
    rows["A"], rows["B"] = _Row2(), _Row2()
    e = {T: dict(eng=0.0, con=0.0, bouts=0.0, dbar=0.0, cbar=0.0, ej=0.0, feat={}, res=1.0, obs_by={"carer": r}) for T, r in (("A", 3.0), ("B", 1.0))}
    coarse_advance(rows, e, 30.0, 30.0, dict(cap=1.0, lr=1.0, fade=1.0), c={**CN2, "rel": True})
    assert rows["A"].kn > 2.5 * rows["B"].kn > 0, (rows["A"].kn, rows["B"].kn)


def test_off_is_the_old_mind():
    m = Mind("Tom", "peasant", 0)
    assert m.D is H and m.zdim == reader.ZDIM and not m.use_dest and not m.use_watch and type(m.wm) is WorldModel


if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            n += 1
            print("ok", k, flush=True)
    print(n, "passed")
