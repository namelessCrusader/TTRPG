"""THE BLOW SEAM in bodies._spend (wave 13, part 13a, lane B2).

Every blow a moving thing lands is spent in `_spend`. Wave 13 puts ONE seam in
it, with the order the plan fixes (plan 3.2):

    mechanism  ->  record cut  ->  shatter threshold x (1 - D)  ->  cell law

Each of the first two is a method of the world, found by name (BLOW_STAGES), called
once per struck thing, and able only to take joules; the third reads the lazy `dmg`
array; the fourth is what `_spend` always was. In 13a no world has a stage and no
`dmg` array exists, so the seam is inert, and these tests hold both halves of that:

  * INERT: no stage, or a stage that owns nothing and hands the joules back, or an
    all-zero `dmg`, changes nothing, to the bit (same shatters with the same
    intensity, same world);
  * LIVE: with stages and damage the order, the arguments, the remainder arithmetic
    and the threshold are what the plan says, and the REAL T11 handler
    (MechanismsMixin.mech_blow) is found by name and takes a door's blow.
Tags: [I] identity and closed forms."""
import inspect

import numpy as np
import pytest

from src.voxel.bodies import BLOW_STAGES
from src.voxel.materials import FLESH, GLASS, STONE, WOOD, _TOUGH_ARR
from src.voxel.events import plain
from src.voxel.matdamage import shatter_threshold
from src.voxel.scenes import _person
from src.voxel.sim import World

STAGE_METHODS = [m for _, m in BLOW_STAGES]


class Spy(World):
    """A World that records what the cell law does: every cell it shatters, with
    the intensity it passes (the blow over the cell's threshold)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.shatters = []

    def _shatter(self, x, y, z, k):
        self.shatters.append(((x, y, z), k))


def world(cls=Spy, nx=30, ny=16, nz=14):
    w = cls(nx, ny, nz)                                   # 10 cm voxels
    w.open_sky = False
    w.fill(0, nx, 0, ny, 0, 1, STONE)
    return w


def thing(w, name, box, m):
    """Register a thing and fill its box with it: its cells carry its eid."""
    e = w.register(name)
    w._stamping = e
    w.fill(*box, m)
    w._stamping = 0
    return e


def strike(w, cells, joules, vel=(5.0, 0.0, 0.0), by=None):
    """Spend `joules` of a loose stone's on `cells`; returns the stone."""
    cells = [tuple(int(v) for v in c) for c in cells]
    b = w.loose(cells[0], vel, STONE, 500.0)
    if by is not None:
        b["by_pid"] = by
    w._spend(b, cells, joules)
    return b


def thr_of(w, cell, m, a_hit=0.01):
    """The cell law's threshold for `cell`, in `_spend`'s own operand order."""
    return (_TOUGH_ARR[m] * 1000.0 * a_hit * (0.7 + 0.6 * w._flaw01(*cell)))


def state(w):
    """Everything a blow can touch, as bytes: the lattice fields, the event log, sounds.
    (Events and sounds go through `plain`, which turns a not-yet-numbered event box into
    its id: a box prints its memory address.)"""
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    for f in ("mat", "smass", "E", "fl", "fvol", "eid", "rubble"):
        h.update(np.ascontiguousarray(getattr(w, f)).tobytes())
    h.update(repr([e.to_json() for e in w.events.rows]).encode())
    h.update(repr(plain(w.sounds)).encode())
    h.update(repr(sorted(w.gone.items(), key=repr)).encode())
    h.update(repr(w._pending and [repr(plain(x)) for x in w._pending]).encode())
    return h.hexdigest()


# ═════════════════════════════════════════════════════════════════════════════
# 1. the contract
# ═════════════════════════════════════════════════════════════════════════════
def test_the_stage_table_is_the_plans_order_and_a_plain_world_has_no_stage():
    assert BLOW_STAGES == (("mechanism", "mech_blow"), ("record_cut", "record_blow"))
    w = World(8, 8, 8)
    # B7 put MechanismsMixin in World, so a plain World has the mech_blow stage (inert on a
    # thing that is no mechanism, see the test below); record_blow has no owner yet.
    assert w._blow_stages() == (w.mech_blow,)
    assert getattr(w, "dmg", None) is None, "no damage array until T9 turns it on"
    for m in STAGE_METHODS:
        if m != "mech_blow":
            assert not hasattr(World, m), f"{m} is not a method of the plain World in 13a"


def test_the_real_T11_handler_is_the_one_the_seam_looks_for_by_name():
    from src.voxel.mechlaw import MechanismsMixin
    assert "mech_blow" in STAGE_METHODS and callable(MechanismsMixin.mech_blow)
    params = list(inspect.signature(MechanismsMixin.mech_blow).parameters)
    assert params == ["self", "eid", "cells", "joules", "a_hit", "direction", "striker"]


# ═════════════════════════════════════════════════════════════════════════════
# 2. INERT by default
# ═════════════════════════════════════════════════════════════════════════════
def test_with_no_stage_the_cell_law_judges_each_cell_on_its_share():
    w = world()
    e = thing(w, "pane", (10, 13, 5, 6, 2, 4), GLASS)    # 3 x 1 x 2 = 6 cells
    cells = [tuple(c) for c in np.argwhere(w.eid == e).tolist()]
    strike(w, cells, 60.0)
    per = 60.0 / len(cells)
    assert [c for c, _ in w.shatters] == cells, "every cell was judged, in order"
    for c, k in w.shatters:
        assert k == per / max(thr_of(w, c, GLASS), 1e-9), "intensity = share over threshold"


def test_a_stage_that_owns_nothing_and_hands_the_joules_back_changes_nothing_to_the_bit():
    class Hands(Spy):
        def mech_blow(self, eid, cells, joules, a_hit=None, direction=None, striker=None):
            return float(joules)

        def record_blow(self, eid, cells, joules, a_hit=None, direction=None, striker=None):
            return float(joules)

    outs = []
    for cls in (Spy, Hands):
        w = world(cls)
        e = thing(w, "pane", (10, 14, 5, 7, 2, 4), GLASS)
        thing(w, "block", (18, 20, 5, 7, 2, 3), STONE)
        cells = [tuple(c) for c in np.argwhere(w.eid != 0).tolist()]
        cells.append((25, 3, 3))                          # and a cell that is no thing's
        strike(w, cells, 123.456789)
        outs.append((w.shatters, state(w)))
    assert outs[0][0] == outs[1][0] and outs[0][1] == outs[1][1]
    assert len(outs[0][0]) > 0


def test_an_all_zero_damage_array_changes_nothing_either():
    outs = []
    for dmg in (None, "zeros"):
        w = world()
        e = thing(w, "pane", (10, 14, 5, 7, 2, 4), GLASS)
        if dmg:
            w.dmg = np.zeros(w.shape, np.float32)
        cells = [tuple(c) for c in np.argwhere(w.eid == e).tolist()]
        strike(w, cells, 32.0)                            # 2 J a cell: about a glass cell's threshold
        outs.append((w.shatters, state(w)))
    assert outs[0] == outs[1]
    per = 32.0 / 16
    broke = sum(per > thr_of(w, c, GLASS) for c in cells)
    assert 0 < broke < 16 and len(outs[0][0]) == broke, "some cells go and some hold, the same ones"


def test_the_real_mechanisms_mixin_is_inert_on_a_thing_that_is_no_mechanism():
    """When the mixin joins the class list (lane B7), every world has `mech_blow`. It must
    hand the joules of an ordinary wall or pane back to the bit, or golden moves."""
    from src.voxel.mechlaw import MechanismsMixin

    assert issubclass(Spy, MechanismsMixin), "B7: the mixin is already in World"

    class Mech(Spy):
        pass

    class NoStage(Spy):
        mech_blow = None        # the world as it was before the mixin: no stage

    outs = []
    for cls in (NoStage, Mech):
        w = world(cls)
        thing(w, "pane", (10, 14, 5, 7, 2, 4), GLASS)
        thing(w, "wall", (3, 8, 2, 3, 1, 8), WOOD)
        cells = [tuple(c) for c in np.argwhere(w.eid != 0).tolist()]
        strike(w, cells, 400.0)
        outs.append((w.shatters, state(w)))
    assert outs[0] == outs[1] and len(outs[0][0]) > 0


# ═════════════════════════════════════════════════════════════════════════════
# 3. LIVE: the order, the arguments, the arithmetic
# ═════════════════════════════════════════════════════════════════════════════
class Staged(Spy):
    """Stages that log what they were offered and take a fixed fraction."""
    take = {"mech": 0.0, "record": 0.0}

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.log = []

    def mech_blow(self, eid, cells, joules, a_hit=None, direction=None, striker=None):
        self.log.append(("mech", eid, np.array(cells), joules, a_hit, direction, striker))
        return joules * (1.0 - self.take["mech"])

    def record_blow(self, eid, cells, joules, a_hit=None, direction=None, striker=None):
        self.log.append(("record", eid, np.array(cells), joules, a_hit, direction, striker))
        return joules * (1.0 - self.take["record"])

    def _shatter(self, x, y, z, k):
        self.log.append(("cell", (x, y, z), k))
        super()._shatter(x, y, z, k)


def test_the_order_is_mechanism_then_record_cut_then_the_lowered_threshold_then_the_cell_law():
    w = world(Staged)
    w.take = {"mech": 0.25, "record": 0.5}
    e = thing(w, "pane", (10, 13, 5, 6, 2, 4), GLASS)
    cells = [tuple(c) for c in np.argwhere(w.eid == e).tolist()]
    w.dmg = np.zeros(w.shape, np.float32)
    for c in cells:
        w.dmg[c] = 0.5
    strike(w, cells, 80.0)
    names = [r[0] for r in w.log]
    assert names == ["mech", "record"] + ["cell"] * len(cells), names
    mech, rec = w.log[0], w.log[1]
    n = len(cells)
    assert mech[1] == rec[1] == e
    assert mech[3] == (80.0 / n) * n and mech[3] == pytest.approx(80.0)
    assert rec[3] == pytest.approx(80.0 * 0.75), "the record stage sees what the mechanism left"
    left = 80.0 * 0.75 * 0.5
    for (_k, c, k) in w.log[2:]:
        thr = shatter_threshold(thr_of(w, c, GLASS), np.float32(0.5))
        assert k == (left / n) / max(thr, 1e-9), "the cell law judges the remainder on thr x (1 - D)"


def test_a_stage_is_offered_each_struck_thing_once_with_its_own_cells_and_joules():
    w = world(Staged)
    ea = thing(w, "a", (10, 13, 5, 6, 2, 4), GLASS)       # 6 cells
    eb = thing(w, "b", (18, 20, 5, 6, 2, 3), GLASS)       # 2 cells
    ca = [tuple(c) for c in np.argwhere(w.eid == ea).tolist()]
    cb = [tuple(c) for c in np.argwhere(w.eid == eb).tolist()]
    bare = [(25, 3, 3)]                                   # no thing: never offered
    strike(w, ca + cb + bare, 90.0)
    per = 90.0 / 9
    offers = [r for r in w.log if r[0] == "mech"]
    assert [r[1] for r in offers] == [ea, eb]
    assert sorted(map(tuple, offers[0][2].tolist())) == sorted(ca) and offers[0][2].shape == (6, 3)
    assert sorted(map(tuple, offers[1][2].tolist())) == sorted(cb)
    assert offers[0][3] == per * 6 and offers[1][3] == per * 2
    assert all(r[4] == pytest.approx(0.01) for r in offers), "a_hit: the cell face, in m2"


def test_a_cell_nobody_took_from_keeps_its_share_exactly():
    w = world(Staged)
    w.take = {"mech": 0.5, "record": 0.0}
    ea = thing(w, "a", (10, 12, 5, 6, 2, 3), GLASS)       # 2 cells: half of their share goes
    w.fill(25, 26, 3, 4, 3, 4, GLASS)                     # a bare glass cell: no thing
    ca = [tuple(c) for c in np.argwhere(w.eid == ea).tolist()]
    bare = (25, 3, 3)
    assert w.eid[bare] == 0
    strike(w, ca + [bare], 30.0)
    cell_k = {c: k for (_n, c, k) in [r for r in w.log if r[0] == "cell"]}
    per = 30.0 / 3
    assert cell_k[bare] == per / max(thr_of(w, bare, GLASS), 1e-9), "no thing, no stage: its own share"
    for c in ca:
        assert cell_k[c] == (per * 0.5) / max(thr_of(w, c, GLASS), 1e-9), "half of what it was"


def test_a_stage_that_takes_it_all_ends_the_chain_and_nothing_shatters():
    w = world(Staged)
    w.take = {"mech": 1.0, "record": 0.0}
    e = thing(w, "pane", (10, 13, 5, 6, 2, 4), GLASS)
    cells = [tuple(c) for c in np.argwhere(w.eid == e).tolist()]
    strike(w, cells, 500.0)
    assert [r[0] for r in w.log] == ["mech"], "the record stage is not offered nothing"
    assert w.shatters == []


def test_a_stage_cannot_add_joules_or_return_a_negative_amount():
    class Liar(Spy):
        seen = None

        def mech_blow(self, eid, cells, joules, a_hit=None, direction=None, striker=None):
            return joules * 50.0                          # tries to add energy

        def record_blow(self, eid, cells, joules, a_hit=None, direction=None, striker=None):
            self.seen = joules                            # what the NEXT stage is offered
            return joules
    w = world(Liar)
    e = thing(w, "pane", (10, 13, 5, 6, 2, 4), GLASS)
    cells = [tuple(c) for c in np.argwhere(w.eid == e).tolist()]
    strike(w, cells, 12.0)
    assert w.seen == pytest.approx(12.0), "the record stage was offered what the blow carried, not 50 times it"
    ref = world()
    thing(ref, "pane", (10, 13, 5, 6, 2, 4), GLASS)
    strike(ref, cells, 12.0)
    assert w.shatters == ref.shatters, "no more than it was offered reached the cells"

    class Neg(Spy):
        def mech_blow(self, eid, cells, joules, a_hit=None, direction=None, striker=None):
            return -5.0
    w = world(Neg)
    thing(w, "pane", (10, 13, 5, 6, 2, 4), GLASS)
    strike(w, cells, 500.0)
    assert w.shatters == [], "a negative remainder is nothing left"


def test_flesh_is_never_offered_to_a_stage_and_the_person_is_still_wounded():
    w = world(Staged, 40, 30, 40)
    w.open_sky = False
    p = _person(w, 12, 12)
    w.step()
    flesh = [tuple(c) for c in np.argwhere(w.mat == FLESH).tolist()][:3]
    e = thing(w, "pane", (30, 33, 5, 6, 2, 4), GLASS)
    pane = [tuple(c) for c in np.argwhere(w.eid == e).tolist()]
    hurt = p["hurt"]
    strike(w, flesh + pane, 60.0)
    assert [r[1] for r in w.log if r[0] == "mech"] == [e], "the person's eid is not a thing's"
    assert p["hurt"] >= hurt


def test_the_striker_reaches_the_stage_as_the_persons_own_dict_and_direction_is_none():
    w = world(Staged, 40, 30, 40)
    p = _person(w, 12, 12)
    w.step()
    e = thing(w, "pane", (30, 33, 5, 6, 2, 4), GLASS)
    cells = [tuple(c) for c in np.argwhere(w.eid == e).tolist()]
    strike(w, cells, 40.0, vel=(3.0, 4.0, 0.0), by=p["pid"])
    mech = next(r for r in w.log if r[0] == "mech")
    assert mech[5] is None, "no vector is passed: the door handler would take it for a side name"
    assert mech[6] is p, "the striker is the person's own dict"
    # a stone nobody threw
    w2 = world(Staged)
    e2 = thing(w2, "pane", (10, 13, 5, 6, 2, 4), GLASS)
    c2 = [tuple(c) for c in np.argwhere(w2.eid == e2).tolist()]
    strike(w2, c2, 40.0, vel=(0.0, 0.0, 0.0))
    m2 = next(r for r in w2.log if r[0] == "mech")
    assert m2[5] is None and m2[6] is None


def test_damage_lowers_the_threshold_so_a_blow_the_whole_cell_would_shrug_off_breaks_it():
    cell = (12, 5, 2)
    outs = {}
    for D in (None, 0.0, 0.75):
        w = world()
        thing(w, "pane", (12, 13, 5, 6, 2, 3), GLASS)
        if D is not None:
            w.dmg = np.zeros(w.shape, np.float32)
            w.dmg[cell] = D
        thr = thr_of(w, cell, GLASS)
        strike(w, [cell], float(thr) * 0.5)               # half what a whole pane takes
        outs[D] = (w.shatters, float(thr))
    assert outs[None][0] == [] and outs[0.0][0] == [], "undamaged: it holds"
    (c, k), = outs[0.75][0]
    assert c == cell and k == pytest.approx(0.5 / 0.25, rel=1e-5), "75 percent damaged: thr x 0.25"


def test_the_real_door_handler_takes_the_blow_before_the_cells_do():
    """MechanismsMixin.mech_blow, found by name: a struck door leaf is a thing whose record
    is a mechanism, the blow goes through its links and none of it reaches the cell law."""
    from src.voxel.mechlaw import MechanismsMixin
    from src.voxel.policy import TablePolicy

    class Stay(TablePolicy):
        def pick(self, situation, menu):
            return 0

    assert issubclass(Spy, MechanismsMixin), "B7: the mixin is already in World"

    class DoorWorld(Spy):
        pass

    nx, ny, nz, wall_y = 50, 40, 50, 25
    w = DoorWorld(nx, ny, nz, voxel_cm=5)
    w.fill(0, nx, 0, ny, 0, 1, STONE)
    w.fill(0, nx, wall_y, wall_y + 1, 1, 45, STONE)
    w.fill(9, 31, wall_y, wall_y + 1, 1, 45, 0)
    mid = w.add_door((10, wall_y), 20, 40, swing=-1, along=(1, 0), corner=(-1, -1))
    w.policy = Stay()
    w.step()
    rec = w.mechs[mid]
    assert w._blow_stages() == (w.mech_blow,)
    calls = []
    real = w.mech_blow

    def spy(eid, cells, joules, a_hit=None, direction=None, striker=None):
        calls.append((eid, len(cells), joules))
        return real(eid, cells, joules, a_hit, direction, striker)
    w.mech_blow = spy
    leaf = [(15, wall_y, z) for z in (8, 9, 10)]
    n_before = len(w.shatters)
    strike(w, leaf, 450.0, vel=(0.0, 12.0, 0.0))
    assert calls and calls[0][0] == rec["leaf"] and calls[0][1] == 3
    assert calls[0][2] == pytest.approx(450.0)
    assert any(e.verb == "strikes" and e.target == ("thing", rec["leaf"]) for e in w.events.rows), \
        "the leaf took the blow through its record, and said so"
    assert len(w.shatters) == n_before, "and the cell law got none of it"
    # the same blow on the same leaf with no handler shatters wood cells
    plain = Spy(nx, ny, nz, voxel_cm=5)
    plain.fill(0, nx, 0, ny, 0, 1, STONE)
    thing(plain, "leaf", (10, 30, wall_y, wall_y + 1, 1, 41), WOOD)
    strike(plain, leaf, 450.0, vel=(0.0, 12.0, 0.0))
    assert len(plain.shatters) == 3
