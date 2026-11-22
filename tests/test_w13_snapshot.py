"""OLD SAVED WORLDS STILL LOAD (wave 13, lane B1; OpenWorld item 5).

`snapshot()` is a deep copy of `vars(world)` and `restore()` first DELETES every
instance attribute the world has, then puts the snapshot's back. A world saved
before wave 13 does not carry the wave 13 names, so any name that `__init__`
makes as an INSTANCE attribute and the code reads without a default would be
missing after `restore` and the next tick would fail. That is why the lazy state
of wave 13 (`slow`, `climate`, `mixes`, `scomp`, `fcomp`, `dmg`, `bzone`, `ice_g`,
`records`, `layers`, `face_env`, ...) lives on the World CLASS with its default.

This file proves it the only way a test in the repo can: tests/data/
world_attrs_base19.json holds the 108 instance names a base19 world has (made
from the base19 tree itself), and a snapshot of a wave 13 world with every name
outside that list taken out IS the snapshot base19 would have made. It must
restore into a world that has all the new members allocated (so `restore` has to
delete them), the next 100 ticks must be the ticks the world would have had, and
a snapshot of a world that HAS the new members must round-trip.

(The run against the real base19 tree, in one process, with the old tree's
snapshot carried over, is in lane B1's report: it needs the old source, which a
test in the repo cannot have.)
"""
import json
import os

import numpy as np
import pytest

from src.voxel import faceenv as FE
from src.voxel.demo import build
from src.voxel.harvest import world as harvest_world
from src.voxel.scenes import _person
from src.voxel.sim import STONE, WATER, WOOD, World

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "data", "world_attrs_base19.json")) as _f:
    BASE19 = set(json.load(_f)["keys"])

# INSTANCE names a wave 13 lane may add to a world that uses none of the new features, each with
# the reason and the way an old snapshot (which lacks it) still works. Empty today: every new name
# is a class attribute. A lane that really needs an instance attribute made in `__init__` adds it
# HERE, in the same change, and says how the code copes when `restore` has deleted it.
# Wave 14a lane A: a humid world (HumidMixin: every scene world, the harvest world) makes its WetState
# in __init__. "wet" and "air_rh" are those two names. An old snapshot lacks them; restore() only
# rewrites the names the snapshot holds, so the restored world keeps its own WetState (the
# vapour store is not snapshotted yet: open problem in the lane A report). The plain demo and
# fire-room worlds still have neither name (checked below).
# Wave 14a refinement R4: HumidMixin is in World's bases, so EVERY world makes both in __init__
# (enable_wet). A snapshot that lacks them restores into a dry world (the world it was saved as).
ALLOWED_NEW = {"wet": "HumidMixin.humid_default (R4)", "air_rh": "HumidMixin.humid_default (R4)"}
HUMID_ONLY = {"wet", "air_rh"}

# every name wave 13 puts on a World that base19 does not have, as the class says it
LAZY = ("slow", "climate", "t_out", "t_sky", "mixes", "mixes_live", "scomp", "fcomp",
        "dmg", "bzone", "ice_g", "records", "layers", "face_env")


def _fire_room():
    w = World(34, 26, 24, voxel_cm=5)
    w.open_sky = False
    w.fill(0, 34, 0, 26, 0, 1, STONE)
    for (x0, x1, y0, y1) in ((0, 1, 0, 26), (33, 34, 0, 26), (0, 34, 0, 1), (0, 34, 25, 26)):
        w.fill(x0, x1, y0, y1, 0, 24, STONE)
    w.fill(5, 13, 10, 16, 1, 3, WOOD, frac=0.6)
    for x in range(14, 17):
        w.pour(x, 13, 4, WATER, 200.0)
    _person(w, 26, 13, z0=1)["name"] = "the witness"
    return w


def _poke(w, t):
    """Heat the fire room's logs for 30 ticks (the other worlds are left alone)."""
    if t < 30 and w.shape == (34, 26, 24):
        w.E[8, 13, 2] += 2500.0


BUILDS = {"demo": build, "harvest 3": lambda: harvest_world(3)[0], "fire room": _fire_room}


def _dry(make):
    """A world as it was saved before humidity was on by default: no wet state, no air_rh.
    An old-shape snapshot restores into exactly this (restore deletes what it lacks)."""
    def f():
        w = make()
        w.__dict__.pop("wet", None)
        w.__dict__.pop("air_rh", None)
        return w
    return f


def _state(w):
    out = {k: np.ascontiguousarray(v).tobytes() for k, v in vars(w).items()
           if isinstance(v, np.ndarray)}
    out["shed"] = repr(w.shed)
    out["tick"] = w.tick
    out["people"] = repr([(p["name"], p.get("anchor"), p.get("blood_ml"), p.get("alive"))
                          for p in w.persons])
    return out


def _fill_every_lazy_member(w):
    """Everything wave 13 can hang on a world, made and not empty."""
    mixes = w.ensure_mixes()
    w.scomp[3:6, 3:6, 1] = 7
    w.fcomp[2, 2, 2] = 9
    w.ensure_damage()[1, 1, 1] = 0.25
    w.ensure_bzone()[4, 4, 1] = 3
    w.ensure_ice()[5, 5, 1] = 12.5
    w.ensure_records()
    w.ensure_layers()
    w.face_env = FE.FaceEnv(bz=w.bzone, ov=FE.overlay_table([{"az": 1.2}]))
    return mixes


def test_the_lazy_names_are_class_attributes_and_nothing_is_on_the_instance():
    w = World(10, 10, 10, voxel_cm=5)
    for name in LAZY:
        assert name in vars(World), name
        assert name not in vars(w), name
    assert w.slow is None and w.climate is None and w.mixes_live is False
    assert w.scomp is w.fcomp is w.dmg is w.bzone is w.ice_g is None
    assert w.records is None and w.layers is None and w.face_env is None
    assert w.lazy_bytes() == {}


def test_a_world_that_uses_none_of_it_has_exactly_the_attribute_names_base19_had():
    for name, make in BUILDS.items():
        w = make()
        for t in range(25):
            _poke(w, t)
            w.step()
        assert set(vars(w)) <= BASE19 | set(ALLOWED_NEW), (name, sorted(set(vars(w)) - BASE19))
        assert set(vars(w)) & HUMID_ONLY == HUMID_ONLY, name    # R4: every world is humid


@pytest.mark.parametrize("name", list(BUILDS))
def test_a_snapshot_of_the_old_shape_restores_and_the_next_100_ticks_are_the_same(name):
    make = _dry(BUILDS[name])
    a = make()
    for t in range(30):
        _poke(a, t)
        a.step()
    snap = a.snapshot()
    old = {k: v for k, v in snap.items() if k in BASE19}                 # what base19 saved
    assert set(old) <= set(snap)
    # a world that holds every new member is wound back to a world that never had them
    b = make()
    _fill_every_lazy_member(b)
    assert b.lazy_bytes()
    b.restore(old)
    assert b.lazy_bytes() == {}, "restore must delete the members the old snapshot never had"
    for lazy in LAZY:
        assert lazy not in vars(b), f"{lazy} left on the instance"
    assert b.mixes is None and b.scomp is None and b.slow is None and b.face_env is None
    assert _state(a) == _state(b)
    for t in range(30, 130):
        _poke(a, t)
        _poke(b, t)
        a.step()
        b.step()
        if t % 10 == 0 or t == 129:
            assert _state(a) == _state(b), f"{name}: tick {t}"
    # and the new world can take the members up again after the old snapshot
    b.ensure_mixes()
    b.step()


def test_a_snapshot_of_a_world_with_the_new_members_round_trips_and_is_a_copy():
    w = _fire_room()
    for _ in range(10):
        w.step()
    _fill_every_lazy_member(w)
    n_rows = w.mixes.n_rows
    snap = w.snapshot()
    for name in ("scomp", "fcomp", "dmg", "bzone", "ice_g", "mixes", "records", "layers", "face_env"):
        assert name in snap, name
    # the snapshot is a copy, not the world's own arrays
    w.scomp[3, 3, 1] = 99
    w.bzone[4, 4, 1] = 0
    assert snap["scomp"][3, 3, 1] == 7 and snap["bzone"][4, 4, 1] == 3
    w.restore(snap)
    assert w.scomp[3, 3, 1] == 7 and w.bzone[4, 4, 1] == 3 and w.fcomp[2, 2, 2] == 9
    assert w.dmg[1, 1, 1] == np.float32(0.25) and w.ice_g[5, 5, 1] == np.float32(12.5)
    assert w.mixes.n_rows == n_rows and w.records.n == 0 and w.layers is not None
    # the restored bundle points at the restored overlay ids, not at a ghost
    assert w.face_env.bz is w.bzone
    # into a second world as well, and it steps
    w2 = _fire_room()
    w2.restore(snap)
    assert np.array_equal(w2.scomp, w.scomp) and w2.scomp is not w.scomp
    for _ in range(5):
        w.step()
        w2.step()
    assert _state(w) == _state(w2)


def test_a_scheduler_on_the_world_is_part_of_its_state_and_an_empty_one_changes_nothing():
    from src.voxel import slowtime as ST
    a, b = _fire_room(), _fire_room()
    b.slow = ST.Scheduler()
    for t in range(80):
        _poke(a, t)
        _poke(b, t)
        a.step()
        b.step()
    sa, sb = _state(a), _state(b)
    assert sa == sb, "an installed, empty scheduler moved the world"
    snap = b.snapshot()
    assert "slow" in snap
    c = _fire_room()
    c.restore(snap)
    assert isinstance(c.slow, ST.Scheduler) and c.slow is not b.slow
    d = _fire_room()
    d.slow = ST.Scheduler()
    d.restore({k: v for k, v in snap.items() if k in BASE19})
    assert d.slow is None
