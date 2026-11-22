"""MENUS BATCHED THEN MERGED (wave 14b, lane F): src/mind/tourney.py, the mind's
`batch`, and the voxel `menu_batch`.

What is held:
- a menu of the batch size or fewer is asked EXACTLY as before (the same
  request, the same Decision);
- the merge does not depend on the order the options come in;
- the merge of an exact chooser (a max) is the exact answer;
- the probabilities of a tournament add to 1;
- reactive mode costs 0 model calls, with the batch on;
- a world with `menu_batch` set picks as the reflex table does on the full menu.
"""
import dataclasses
import itertools
import random

import pytest

from src.mind import core
from src.mind.api import Decision, DecisionRequest
from src.mind.choosers import Rules
from src.mind.loop import MindConfig, Mind, run
from src.mind.tourney import Tourney, chunks, joint, merge
from src.rooms import RoomsBackend
from src.rooms.__main__ import SETTING


def req(n, rid="r", who="Fenn", seed=1):
    rng = random.Random(seed)
    pr = tuple(round(rng.gauss(0, 1), 3) for _ in range(n))
    return DecisionRequest(rid, who, "idle", "s", "q", tuple(f"option {i:02d}" for i in range(n)),
                           tuple("g%d" % (i // 5) for i in range(n)), pr, pr, (0.0,) * n)


def shuffled(r, order):
    pick = lambda t: tuple(t[i] for i in order)
    return dataclasses.replace(r, options=pick(r.options), groups=pick(r.groups),
                               prior=pick(r.prior), prior_avg=pick(r.prior_avg), habit=pick(r.habit))


def ask(chooser, r):
    chooser.submit(r)
    return chooser.poll(r.id)


# ---------------------------------------------------------------- the pure merge
def test_CHUNKS_are_even_and_never_a_lone_last_one():
    for n in range(5, 60):
        for size in (2, 3, 4, 8):
            if n <= size:
                continue
            c = chunks(list(range(n)), size)
            assert [x for b in c for x in b] == list(range(n))
            assert max(map(len, c)) <= size and max(map(len, c)) - min(map(len, c)) <= 1
            assert len(c) == -(-n // size)


def test_MERGE_of_a_max_is_the_max_at_every_size():
    rng = random.Random(3)
    for n in (2, 3, 5, 9, 16, 33, 100):
        v = [rng.random() for _ in range(n)]
        for size in (2, 3, 4, 7, 16, 200):
            best = lambda m, r: max(m, key=v.__getitem__)
            assert merge(list(range(n)), size, best) == max(range(n), key=v.__getitem__)


def test_MERGE_is_one_pick_when_the_set_fits():
    calls = []
    merge(list(range(4)), 4, lambda m, r: calls.append((m, r)) or m[0])
    assert calls == [([0, 1, 2, 3], 0)]


def test_MERGE_does_not_depend_on_the_order_of_the_options():
    """The batches are cut from the canonical order, so a chooser that is only
    a function of the batch gives the same winner for any arrangement."""
    names = [f"act {i:02d}" for i in range(23)]
    score = {n: random.Random(n).random() for n in names}

    def run_(order, size):
        keys = [names[i] for i in order]
        # a noisy chooser: it depends on the batch's contents, not on a max
        def pick(m, r):
            tag = "|".join(keys[i] for i in m)
            return m[random.Random(tag).randrange(len(m))]
        return keys[merge(keys, size, pick)]

    rng = random.Random(9)
    for size in (2, 3, 4, 5):
        base = run_(list(range(23)), size)
        for _ in range(25):
            o = list(range(23))
            rng.shuffle(o)
            assert run_(o, size) == base


def test_JOINT_adds_to_one():
    r = req(23)
    t = Tourney(Rules(seed=2), 4)
    d = ask(t, r)
    assert len(d.probs) == 23 and sum(d.probs) == pytest.approx(1.0)
    assert d.probs[d.pick] > 0 and all(p >= 0 for p in d.probs)
    a = t.average(r)
    assert len(a) == 23 and sum(a) == pytest.approx(1.0)


# ---------------------------------------------------------------- the chooser
def test_a_request_that_fits_the_batch_is_asked_as_it_always_was():
    for n in (1, 2, 3, 4):
        r = req(n)
        plain = ask(Rules(seed=5), r)
        wrapped = Tourney(Rules(seed=5), 4)
        assert ask(wrapped, r) == plain
        assert wrapped.average(r) == Rules(seed=5).average(r)


def test_the_TOURNAMENT_of_argmax_rules_is_the_best_option():
    for n in (5, 9, 16, 40):
        r = req(n, seed=n)
        best = max(range(n), key=r.prior.__getitem__)
        for size in (2, 3, 4, 8):
            assert ask(Tourney(Rules(seed=0, argmax=True), size), r).pick == best


def test_the_TOURNAMENT_pick_is_the_same_for_the_options_in_any_order():
    r = req(21, seed=4)
    rng = random.Random(1)
    for size in (3, 4, 6):
        base = ask(Tourney(Rules(seed=8), size), r)
        for _ in range(20):
            o = list(range(21))
            rng.shuffle(o)
            s = shuffled(r, o)
            d = ask(Tourney(Rules(seed=8), size), s)
            assert s.options[d.pick] == r.options[base.pick]
            # and the probability of each option is the same one
            by = {s.options[i]: p for i, p in enumerate(d.probs)}
            assert [by[t] for t in r.options] == pytest.approx(list(base.probs))


# ---------------------------------------------------------------- the mind
def _night(seed, turns=120, model=None, **cfg):
    b = RoomsBackend(seed, turns)
    mind = Mind(MindConfig(seed=seed, setting=SETTING, **cfg), model=model)
    run(b, mind, turns)
    return b, mind


class Sense:
    name, remote = "fake", True

    def __init__(self):
        self.seen, self._done = [], {}

    def submit(self, r):
        self.seen.append(r)
        s = tuple((hash(t) % 97 + 1) / 98 for t in r.options)
        self._done[r.id] = Decision(r.id, max(range(len(s)), key=s.__getitem__), self.name, s)

    def poll(self, req_id, block=True):
        return self._done.pop(req_id, None)

    def average(self, r):
        return None


def test_the_BATCH_in_the_loop_is_the_old_decisions_when_every_menu_fits():
    """With a batch no menu is longer than, nothing is merged and nothing is
    cut: the rows are those of an uncut mind (cap 200) to the byte."""
    b0, m0 = _night(0, cap=200)
    b1, m1 = _night(0, batch=200)
    assert m0.rows and m1.rows == m0.rows
    assert max(len(r["options"]) for r in m0.rows) > core.MENU_MAX     # long menus, all asked whole


def test_the_BATCH_merges_the_long_menus_and_keeps_the_log():
    b, m = _night(1, batch=3)
    assert m.rows
    longest = max(len(r["options"]) for r in m.rows)
    assert longest > 3                                  # there were long menus
    for r in m.rows:
        assert r["cut"] == [] and 0 <= r["pick"] < len(r["options"])
        assert sum(r["probs"]) == pytest.approx(1.0, abs=1e-4)


def test_REACTIVE_mode_with_the_batch_costs_no_model_call():
    sense = Sense()
    for seed in range(2):
        b, mind = _night(seed, model=sense, batch=3)
        assert sorted(mind.remote) == list(range(1, b.world.t + 1))
        assert set(mind.remote.values()) == {0}
    assert sense.seen == []


def test_FEATURED_mode_with_the_batch_asks_the_model_in_small_batches_only():
    sense = Sense()
    b, mind = _night(0, model=sense, mode="featured", featured=("Oren",), batch=3, gate=None)
    assert sense.seen and {r.who for r in sense.seen} == {"Oren"}
    assert max(len(r.options) for r in sense.seen) <= 3
    assert sum(mind.remote.values()) == len(sense.seen)


def test_the_config_refuses_a_batch_of_one():
    with pytest.raises(ValueError):
        MindConfig(batch=1)


# ---------------------------------------------------------------- the voxel body
from types import SimpleNamespace

from src.voxel.policy import MENU_BATCH, MENU_CAP, REFLEXES, TablePolicy
from src.voxel.will import WillMixin


def _legs_menu(rng, n_exits, n_roam):
    menu = [{"key": "stay", "tag": "stay", "verb": "stay"}]
    for i in range(n_exits):
        menu.append({"key": f"go to exit {i}", "tag": "go:exit", "verb": "go",
                     "danger": rng.choice([0.0, 0.0, 1.0, 3.0]), "away": round(rng.uniform(1, 40), 2)})
    for i in range(n_roam):
        menu.append({"key": f"roam {i}", "tag": "go:roam", "verb": "go", "away": round(rng.uniform(1, 40), 2)})
    for i in range(5):
        menu.append({"key": f"take thing {i}", "tag": "take:dear", "verb": "take", "away": float(i)})
    return menu


def _sit(percept):
    return {"limb": "legs", "percept": percept, "reflexes": REFLEXES, "chosen": {}}


def test_the_BATCH_of_the_voxel_is_below_the_cap_and_the_mind_has_the_same_one():
    assert MENU_BATCH == core.BATCH and 3 <= MENU_BATCH <= MENU_CAP


def test_a_voxel_TOURNAMENT_of_the_table_is_the_whole_menu_pick_at_every_batch():
    rng = random.Random(11)
    me = SimpleNamespace(policy=TablePolicy())
    for _ in range(60):
        menu = _legs_menu(rng, rng.randint(0, 9), rng.randint(0, 8))
        for percept in (None, "sees_fire", "hears_alarm"):
            sit = _sit(percept)
            full = me.policy.pick(sit, menu)
            for b in (3, 4, 5, 7):
                if len(menu) <= b:
                    continue
                i = WillMixin._merged_pick(me, sit, "legs", {}, menu, b)
                # the same row, or one that is a tie of the table's own key (tag, danger, away)
                key = lambda o: (o["tag"], o.get("danger", 0.0), o.get("away", 0.0))
                assert key(menu[i]) == key(menu[full]), (percept, b, menu[i], menu[full])


def test_a_voxel_TOURNAMENT_does_not_depend_on_the_order_of_the_menu():
    rng = random.Random(5)
    me = SimpleNamespace(policy=TablePolicy())
    for _ in range(40):
        menu = _legs_menu(rng, rng.randint(2, 9), rng.randint(2, 8))
        tail = menu[1:]
        for percept in (None, "sees_fire"):
            base = menu[WillMixin._merged_pick(me, _sit(percept), "legs", {}, menu, 4)]["key"]
            for _ in range(8):
                rng.shuffle(tail)
                m = [menu[0]] + tail
                assert m[WillMixin._merged_pick(me, _sit(percept), "legs", {}, m, 4)]["key"] == base


def test_a_world_with_the_batch_off_is_untouched_and_a_script_keeps_its_cut():
    from src.voxel.policy import MENU_BATCH
    assert WillMixin.menu_batch == MENU_BATCH      # on for every world since 14bF; None is the old cut
    assert getattr(TablePolicy, "batchable", False)
    assert not getattr(SimpleNamespace(), "batchable", False)


def test_a_world_with_menu_batch_offers_the_whole_menu_and_picks_from_it():
    from src.voxel.materials import AIR, STONE, WOOD
    from src.voxel.scenes import _person
    from src.voxel.sim import World
    w = World(90, 60, 40, voxel_cm=5)
    w.open_sky = False
    w.fill(0, 90, 0, 60, 0, 40, STONE)
    w.mat[1:89, 1:59, 1:39] = AIR
    w.smass[1:89, 1:59, 1:39] = 0.0
    w.exits = [(88, 12), (88, 30), (88, 48), (1, 12), (1, 48), (45, 58)]
    w.fill(10, 18, 20, 30, 1, 3, WOOD, frac=0.8)
    w.E[10, 20, 1] = 9.0e5
    for n, (x, y) in enumerate(((30, 10), (60, 40), (20, 50), (70, 12))):
        _person(w, x, y)["name"] = f"P{n}"
    w.menu_batch = MENU_BATCH
    for _ in range(60):
        w.step()
    longest = max((len(m) for r in w.traces for m in r["menus"].values()), default=0)
    assert longest > MENU_CAP, "nothing was cut: the menus are whole"
    for r in w.traces:
        for limb, keys in r["menus"].items():
            assert r["pick"][limb] in keys
