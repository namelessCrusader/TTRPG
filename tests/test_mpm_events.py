"""What the particle matter did, as the seam reports it (`Matter.events`).

The voxel side cannot see inside the solver: a block that fell through the
seam landed without a sound, and a pane that cracked there cracked in
silence. The tap reads each step off the particles, piece by piece, and
hands back plain records: a blow (`struck`), two pieces meeting (`touch`),
matter cracking (`crack`) and bonds parting (`part`). These tests pin that
the records say what happened, when it happened, and nothing else, and that
reading them changes nothing.

Every number here was measured before it was asserted.
"""

import numpy as np
import pytest

from src.mpm.core import GRAVITY
from src.mpm.matter import HIT, _off_fall
from src.voxel.sim import World, AIR, STONE, WOOD, GLASS


def _plain(v):
    """A record holds only what any reader can take: no numpy anywhere."""
    if isinstance(v, dict):
        return all(isinstance(k, str) and _plain(x) for k, x in v.items())
    if isinstance(v, tuple):
        return all(_plain(x) for x in v)
    return type(v) in (int, float, str, bool)


def _run(w, ticks):
    """Step `w` and collect every record, checking each one is plain and
    carries the tick it was made in."""
    out = []
    for _ in range(ticks):
        t = w.tick
        w.step()
        for r in w.matter.events():
            assert _plain(r), r
            assert r["tick"] == t, r
            out.append(r)
    return out


def _dropped(mat=WOOD, z0=15):
    """A 2x2x2 block of `mat` in the air over a stone floor (10 cm cells).
    The floor's top is at 0.1 m and the block's bottom at z0 cells."""
    w = World(12, 12, 30)
    w.fill(0, 12, 0, 12, 0, 1, STONE)
    w.fill(5, 7, 5, 7, z0, z0 + 2, mat)
    cells = np.argwhere(w.mat == mat)
    w.matter.promote(cells[cells[:, 2] >= 1])            # not the floor
    return w


def test_a_BLOW_IS_WHAT_FALLING_AND_RESTING_DO_NOT_EXPLAIN():
    """The jolt behind every `struck`. Free flight changes the velocity by
    exactly g dt, a thing lying on a floor not at all, and one tipping over
    or sliding by something in between: none of those is a blow. A landing
    takes off the speed it arrived with, and that is the jolt."""
    gdt = GRAVITY * 0.025
    assert _off_fall(gdt, gdt) == pytest.approx(0.0, abs=1e-15)          # flying
    assert _off_fall(np.zeros(3), gdt) == 0.0                            # lying
    assert _off_fall(0.4 * gdt, gdt) == pytest.approx(0.0, abs=1e-15)    # tipping
    assert _off_fall(np.array([0.0, 0.0, 3.0]), gdt) == pytest.approx(3.0)  # landed
    assert _off_fall(np.array([1.0, 0.0, 0.0]), gdt) == pytest.approx(1.0)  # knocked
    many = np.array([gdt, [0.0, 0.0, 3.0]])
    assert _off_fall(many, gdt) == pytest.approx([0.0, 3.0])


def test_a_DROPPED_BLOCK_IS_STRUCK_ONCE_at_the_SPEED_IT_FELL():
    """A wooden block falls 1.4 m onto a stone floor: one blow, in the tick
    it lands, at the speed it fell (measured 5.242 m/s against
    sqrt(2gh) = 5.241), losing what the fall was worth (measured 67.1 J
    against m g h = 65.9 J: the last few centimetres of the tick count too),
    against the lattice cells under it."""
    w = _dropped()
    kg = float(w.matter.mpm.m.sum())
    h = 1.5 - 0.1
    ev = _run(w, 40)
    struck = [r for r in ev if r["kind"] == "struck"]
    big = [r for r in struck if r["joules"] > 0.5 * kg * 9.81 * h]
    assert len(big) == 1, struck
    s = big[0]
    dx = w.matter.dx
    assert s["speed"] * dx == pytest.approx(np.sqrt(2 * 9.81 * h), rel=0.10)
    assert s["joules"] == pytest.approx(kg * 9.81 * h, rel=0.10)
    assert s["jolt"] * dx > HIT
    assert (s["eid"], s["batch"]) == (0, 0)
    assert s["against"] == ()
    assert s["mats"] == (int(WOOD),)
    # it lies on the floor cells under its own 2x2 footprint
    assert sorted(s["hit"]) == [(x, y, 0) for x in (5, 6) for y in (5, 6)]
    assert s["cell"][2] == 1 and 5 <= s["cell"][0] <= 6 and 5 <= s["cell"][1] <= 6
    # and the tick it is reported in is the tick it stopped falling
    ent = w.matter.entities()[0]
    assert not ent["aloft"]
    assert s["tick"] == 21, "it lands in its 22nd tick: sqrt(2h/g) = 0.534 s"


def test_a_BLOCK_ON_A_BLOCK_is_reported_in_the_SAME_TICK():
    """A wooden block (thing 9) falls onto a stone block (thing 7) that lies
    on the floor. The tick it lands, the two pieces touch and the wooden
    one is struck AGAINST THE STONE ONE, not against the lattice. The
    prototype tap judged contact by whole cells, a tick late: particles of
    two things push on each other through the grid while they are still a
    cell apart, so the blow came first and named the floor."""
    w = World(16, 12, 24)
    w.fill(0, 16, 0, 12, 0, 1, STONE)
    w.fill(5, 9, 4, 8, 1, 3, STONE)
    w.eid[5:9, 4:8, 1:3] = 7
    w.fill(6, 8, 5, 7, 12, 14, WOOD)
    w.eid[6:8, 5:7, 12:14] = 9
    w.matter.promote(np.argwhere(w.eid == 7))
    w.matter.promote(np.argwhere(w.eid == 9))
    ev = _run(w, 40)
    struck = [r for r in ev if r["kind"] == "struck"]
    touch = [r for r in ev if r["kind"] == "touch"]
    assert len(struck) == 1 and len(touch) == 1, ev
    s, t = struck[0], touch[0]
    assert s["tick"] == t["tick"]
    assert (s["eid"], s["batch"]) == (9, 1)
    assert s["against"] == ((7, 0),)
    assert s["hit"] == (), "it lies on the stone block, not on the lattice"
    assert (t["eid"], t["batch"], t["other"]) == (7, 0, (9, 1))
    # where they meet: between the stone block's top (z = 3) and the
    # wooden block's bottom, which rests about a cell above it
    assert 2.0 < t["at"][2] < 5.0
    assert 6.0 <= t["at"][0] <= 8.0 and 5.0 <= t["at"][1] <= 7.0


def test_THINGS_AT_REST_SAY_NOTHING():
    """A pane promoted where it lies, on the floor and on a table, at both
    cell sizes, settles the centimetre the solver's contact needs and then
    lies still: no blow, no crack, no touch. Nor do two blocks promoted
    together while one rests on the other: they were already touching when
    they came in, and that is not a contact they made."""
    def floor_pane():
        w = World(16, 12, 10)
        w.fill(0, 16, 0, 12, 0, 1, STONE)
        w.fill(4, 10, 4, 8, 1, 2, GLASS)
        w.matter.promote(np.argwhere(w.mat == GLASS))
        return w

    def table_pane():
        w = World(30, 20, 20, voxel_cm=5)
        w.fill(0, 30, 0, 20, 0, 1, STONE)
        w.fill(5, 25, 5, 15, 1, 8, STONE)
        w.fill(8, 20, 7, 13, 8, 9, GLASS)
        w.matter.promote(np.argwhere(w.mat == GLASS))
        return w

    def stacked():
        w = World(16, 12, 16)
        w.fill(0, 16, 0, 12, 0, 1, STONE)
        w.fill(5, 9, 4, 8, 1, 3, STONE)
        w.eid[5:9, 4:8, 1:3] = 7
        w.fill(6, 8, 5, 7, 3, 5, WOOD)
        w.eid[6:8, 5:7, 3:5] = 9
        w.matter.promote(np.argwhere(w.eid > 0))
        return w

    for build in (floor_pane, table_pane, stacked):
        w = build()
        assert len(w.matter) > 0
        ev = _run(w, 40)
        assert ev == [], (build.__name__, ev)
        assert not w.matter.aloft(), build.__name__
    assert len(w.matter._touching) == 1, "the stacked blocks touch all along"


def test_a_STONE_THAT_BREAKS_says_so_ONCE_in_the_TICK_IT_LANDS():
    """Stone dropped 1.4 m lands hard enough to break (measured: 6 of its 8
    particles crack and 2 come loose from the cluster). The blow, the
    cracks and the parted bonds are all in the tick it lands, and the
    records add up to the particles: every crack and every parted bond is
    reported once."""
    w = _dropped(STONE)
    m = w.matter.mpm
    bound = int((m.cid >= 0).sum())
    ev = _run(w, 40)
    struck = [r for r in ev if r["kind"] == "struck"]
    crack = [r for r in ev if r["kind"] == "crack"]
    part = [r for r in ev if r["kind"] == "part"]
    assert crack and part, ev
    assert sum(r["n"] for r in crack) == int(m.dmg.sum())
    assert sum(r["n"] for r in part) == bound - int((m.cid >= 0).sum())
    assert sum(r["n"] for r in crack) > sum(r["n"] for r in part), \
        "some cracked without coming loose, so the two kinds are two facts"
    assert {r["tick"] for r in crack} == {struck[0]["tick"]}
    assert {r["tick"] for r in part} == {struck[0]["tick"]}
    assert all(r["mats"] == (int(STONE),) for r in crack + part)
    # the order within a tick: the blow, then what it broke
    kinds = [r["kind"] for r in ev if r["tick"] == struck[0]["tick"]]
    assert kinds == ["struck", "crack", "part"]


def test_a_SECOND_FALL_IS_A_SECOND_BLOW():
    """The hysteresis lets one landing be one blow, and must not swallow the
    next one. The block from the drop test lands, lies still, and is thrown
    straight up at 3 m/s: it flies, falls back, and is struck again, at
    about the speed it was thrown. Nothing in between."""
    w = _dropped()
    first = _run(w, 40)
    assert [r["kind"] for r in first] == ["struck"]
    w.matter.mpm.v[:, 2] += 3.0
    again = _run(w, 40)
    struck = [r for r in again if r["kind"] == "struck"]
    assert len(struck) == 1, again
    assert struck[0]["speed"] * w.matter.dx == pytest.approx(3.0, rel=0.10)
    assert struck[0]["tick"] > first[0]["tick"] + 20


def test_the_TAP_CHANGES_NOTHING():
    """The pole that golden.py's pole_mpm topples, twice: once with the tap
    reading every step and once with it off. Every particle and every cell
    the laws read are the same bits after every tick, and the tap saw the
    pole land."""
    def pole():
        w = World(24, 8, 16)
        w.fill(0, 24, 0, 8, 0, 1, STONE)
        w.fill(4, 7, 3, 5, 1, 12, WOOD)
        w.mat[5:7, 3:5, 1] = AIR
        w.smass[5:7, 3:5, 1] = 0.0
        return w
    a, b = pole(), pole()
    b.matter.tap = False
    seen = []
    for t in range(60):
        a.step()
        b.step()
        seen += a.matter.events()
        assert b.matter.events() == []
        ma, mb = a.matter.mpm, b.matter.mpm
        for f in ("x", "v", "C", "F", "Jp", "m", "dmg", "cid", "E", "eid", "batch"):
            assert np.array_equal(getattr(ma, f), getattr(mb, f)), (t, f)
        for f in ("mat", "smass", "E", "fvol", "eid", "moving"):
            assert np.array_equal(getattr(a, f), getattr(b, f)), (t, f)
    assert len(a.matter), "nothing went through the seam"
    assert any(r["kind"] == "struck" and r["joules"] > 50.0 for r in seen), seen
