"""Reference tests for SOUND (docs/realism-references.md, 4.8; gap G1).

Each test builds a small scene at 5 cm voxels and checks one measured
real-world number. The pass band and its source are in each docstring; the
source numbers are the references of that file's section 5.8. Levels are dB
at the ear; a shout is SHOUT_DB (90) at a metre.

Hearing is `_at_ear` (senses.py): a sound from where it is made to the ear
on the outside of the listener's head, with neither end's own body in the
way. `_hears` (a line understood) and `_law_hear` (a noise noticed, a
sleeper woken) both ask it.
"""
import gc

import numpy as np
import pytest

from src.voxel import senses
from src.voxel.sim import (HEAR_DB, IRON, LEAF, SHOUT_DB, SOUND, STONE, WOOD,
                           TablePolicy, World)
from src.voxel.scenes import _person


class _Still(TablePolicy):
    """Everyone keeps their place: the legs pick their first option."""
    def pick(self, sit, menu):
        if sit["limb"] == "legs":
            return 0
        return super().pick(sit, menu)


def _ground(nx, ny, nz=44, roof=False):
    """A stone floor, and a stone ceiling at 2.1 m if `roof`."""
    w = World(nx, ny, nz, voxel_cm=5)
    w.fill(0, nx, 0, ny, 0, 1, STONE)
    if roof:
        w.fill(0, nx, 0, ny, nz - 1, nz, STONE)
    w.exits = []
    w.policy = _Still()
    return w


def _two(w, a_at, b_at, names=("shouter", "listener")):
    """Two standing people, and one tick so they have heads and bodies."""
    a = _person(w, *a_at)
    b = _person(w, *b_at)
    a["name"], b["name"] = names
    w.step()
    return a, b


def _at(w, a, b, db=SHOUT_DB):
    """The level of a sound from `a`'s head at `b`'s ear, as `_law_hear`
    and `_hears` both hear it."""
    own = set(a["_own"]) | set(b["_own"])
    return w._at_ear(a["_eye"], b["_eye"], db, own)


# ── A-T1 ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dx,dy", [(60, 0), (-60, 0), (0, 60), (0, -60),
                                   (42, 42), (-42, -42), (42, -42), (-42, 42)])
def test_A_T1_a_shout_is_HEARD_THE_SAME_by_hears_and_by_law_hear(dx, dy):
    """A-T1 (high). Two standing people 3 m apart in an open room, along +-x,
    +-y and both diagonals. Both `_hears` and `_law_hear` give
    90 - 9.5 = 80.5 +- 1 dB, and the line is understood.

    Source: internal consistency, roadmap 97. Before: `_hears` charged the
    mass law for both heads (40-46 dB) and gave about 34-40 dB at 3 m."""
    w = _ground(160, 160)
    a, b = _two(w, (78, 78), (78 + dx, 78 + dy))
    lvl = _at(w, a, b)
    assert 79.5 <= lvl <= 81.5, lvl
    assert w._hears(a["_eye"], b["_eye"]) and w._hears(b["_eye"], a["_eye"])
    # and `_law_hear` hears a shout at exactly that level
    w._say(a, "fire")
    rows = [r for r in w.minds[b["pid"]].perceived if r.how == "heard"]
    w._law_hear()
    got = [r for r in w.minds[b["pid"]].perceived
           if r.how == "heard" and r not in rows]
    assert got and abs(got[-1].db - lvl) < 1e-3, (got, lvl)
    assert dict(got[-1].data)["line"] == "fire", "its words made out"


def test_A_T1_a_shout_carries_THIRTY_METRES_outdoors():
    """A-T1 (high), the far half: 30 m apart on open ground the shout
    arrives at 90 - 29.5 = 60.5 dB, at least 40 dB (HEAR_DB), and is
    understood. Source: spreading loss, row A4 (6 dB a doubling)."""
    w = _ground(620, 16, 40)
    a, b = _two(w, (6, 8), (606, 8))
    lvl = _at(w, a, b)
    assert lvl >= 40.0 and abs(lvl - 60.5) <= 1.0, lvl
    assert w._hears(a["_eye"], b["_eye"])


def test_two_people_with_THE_SAME_NAME_hear_each_other():
    """Found by V3: `_law_hear` told a person's own sounds by NAME, and
    default names repeat ("the person"), so two of them were deaf to each
    other's every footstep and shout. A sound's maker is a pid."""
    w = _ground(100, 60)
    a, b = _two(w, (20, 30), (60, 30), names=("the person", "the person"))
    w._say(a, "fire")
    w._law_hear()
    assert b["noise"] is not None and b["noise"]["ev_id"] == \
        w.minds[a["pid"]].said_ev, "he heard the other one's shout"
    assert (a.get("noise") or {}).get("ev_id") != w.minds[a["pid"]].said_ev, \
        "and nobody hears his own shout as news"


def test_a_BLOW_is_its_STRIKER_S_OWN_noise_when_NAMES_REPEAT():
    """Found by the R2 verifier. A blow's sound is made before its `hits`
    event has an id, so `_maker` could match it to its maker only by NAME,
    and with two people of the same name it gave up: the striker was not
    skipped and heard his own blow as news. A body's noise now carries the
    pid of whose flesh the body is (`_body_pid`: the eid the flesh carries
    where flesh is stamped, and otherwise, of the people with that name,
    the one whose flesh is nearest where the body left the lattice)."""
    w = _ground(100, 60)
    a, b = _two(w, (20, 30), (60, 30), names=("the person", "the person"))
    top = max(c[2] for c in a["_own"])
    piece = np.array(sorted(c for c in a["_own"] if c[2] == top), np.int64)
    swung = w._launch(piece, (0.0, 0.0, 0.0), owner=a["name"])
    swung["part"] = True                  # a piece of him, swung as an arm is
    assert w._body_pid(swung) == a["pid"], "his flesh says whose it is"
    w.fill(26, 27, 30, 31, 1, 2, IRON)
    w._spend(swung, np.array([(26, 30, 1)], np.int64), 20.0)
    snd = w.sounds[-1]
    assert snd["who"] == "the person" and snd["pid"] == a["pid"]
    w._law_hear()
    assert b["noise"] is not None and b["noise"]["at"] == snd["at"], \
        "the other one hears the blow"
    assert (a.get("noise") or {}).get("at") != snd["at"], \
        "and the striker does not hear his own blow as news"


# ── A-T2 ────────────────────────────────────────────────────────────────────

def _two_rooms():
    """Two rooms under one ceiling, a 10 cm stone wall between them at
    x = 5 m, and an open doorway 1 m wide and 2.1 m tall in it."""
    w = _ground(200, 120, roof=True)
    w.fill(100, 102, 0, 120, 1, 43, STONE)
    w.mat[100:102, 70:90, 1:43] = 0
    w.smass[100:102, 70:90, 1:43] = 0.0
    return w


def test_A_T2_a_shout_goes_ROUND_AN_OPEN_DOORWAY():
    """A-T2 (high). The shouter is in room A, 2 m from the doorway, with no
    line of sight; the listener is in room B, 3 m past the doorway, round
    the corner. Expected 50-68 dB +- 10 (a rough band), and `_hears` is
    True. Source: 205 (an adjacent room through an open doorway is 10-17 dB
    down), 214 and 207 (ISO 9613-2: one edge costs at most 20 dB). Before:
    about 20 dB through `_level` (the 10 cm wall's mass law, 55.6 dB)."""
    w = _two_rooms()
    a, b = _two(w, (84, 42), (110, 22))
    assert not w._sees(a["_eye"], b["_eye"]), "round the corner"
    lvl = _at(w, a, b)
    assert 40.0 <= lvl <= 78.0, lvl
    assert w._hears(a["_eye"], b["_eye"])


def test_a_doorway_SEALED_MID_TICK_is_sealed_to_sound_at_once():
    """Found by the R2 verifier. The ways round (`_flood`) were kept for the
    tick, so a doorway filled in during a tick still let sound round for
    the rest of it: 59 dB through stone, where a fresh flood gave 30. They
    are now kept only while the cells stay the same. Filling the doorway
    and then opening it again, in the same tick, each gives what a world
    with no kept floods at all gives."""
    w = _two_rooms()
    a, b = _two(w, (84, 42), (110, 22))
    door = _at(w, a, b)
    w.fill(100, 102, 70, 90, 1, 43, STONE)               # the same tick
    shut = _at(w, a, b)
    senses._ROUND.clear()
    assert shut == _at(w, a, b), "the same as a fresh flood"
    assert shut < door - 20.0, (door, shut)
    w.mat[100:102, 70:90, 1:43] = 0                      # and open again
    w.smass[100:102, 70:90, 1:43] = 0.0
    assert _at(w, a, b) == door


# ── A-T3, A-T8 ──────────────────────────────────────────────────────────────

def _bedroom(door, gap):
    """A bedroom behind a 10 cm stone wall with a doorway 1 m wide and
    2.1 m tall. `door`: a 5 cm pine plank hung in it; `gap`: one voxel
    (5 cm, 2.4% of the door) left open under it. The shouter stands 1 m
    outside, the sleeper 3 m inside, both on the line through the door's
    middle: off the line of the gap."""
    w = _ground(200, 120, roof=True)
    w.fill(100, 102, 0, 120, 1, 43, STONE)
    w.mat[100:102, 50:70, 1:43] = 0
    w.smass[100:102, 50:70, 1:43] = 0.0
    if door:
        w.fill(100, 101, 50, 70, 2 if gap else 1, 43, WOOD)
    a, b = _two(w, (77, 60), (159, 60), names=("shouter", "sleeper"))
    return w, a, b


def test_A_T3_a_shout_through_a_CLOSED_DOOR_wakes_a_SLEEPER():
    """A-T3 (high). A bedroom with a closed 5 cm plank door, hung as doors
    are, with a gap under it. The shouter stands 1 m outside and shouts
    "Fire!" again and again; an adult sleeps 3 m inside.

    Expected: the door costs 10-25 dB +- 5; the level at the pillow is
    50-70 dB; the sleeper wakes within 30 s of the first shout. Source:
    205 (closing a door adds about 10 dBA), 212 (hollow-core door STC
    20-25), 210 (gaps drop a door's rating). Before: the door cost 36.5 dB,
    at most 53.5 dB reached the room, and he never woke.

    Waking is one sample against `wake_db` today (G18), so a later shout
    can do nothing an earlier one did not: three shouts are enough."""
    w, a, b = _bedroom(door=False, gap=False)
    open_lvl = _at(w, a, b)
    w, a, b = _bedroom(door=True, gap=True)
    lvl = _at(w, a, b)
    assert 5.0 <= open_lvl - lvl <= 30.0, (open_lvl, lvl)
    assert 50.0 <= lvl <= 70.0, lvl
    b["asleep"] = True
    woke = None
    for t in range(120):
        if t % 40 == 0:
            w._say(a, "fire")
        w.step()
        if not b["asleep"]:
            woke = t
            break
    assert woke is not None and woke * 0.025 <= 30.0, "the shout woke him"


@pytest.mark.xfail(strict=True, reason=(
    "an AIRTIGHT 5 cm pine door costs its mass law, 36.5 dB, and A-T3 asks "
    "10-25 +- 5; A-T8 (i) allows 25-37 for the same door, so the two bands "
    "disagree, and the mass law sits 2-5 dB above measured timber (row A9). "
    "Not changed in R2 (no gap, no way round: it is a mass-law question)"))
def test_A_T3_an_AIRTIGHT_door_costs_no_more_than_30_dB():
    """A-T3 (high), the airtight half of its scene: the door's loss within
    10-25 dB +- 5. Source: 205, 212, 210."""
    w, a, b = _bedroom(door=False, gap=False)
    open_lvl = _at(w, a, b)
    w, a, b = _bedroom(door=True, gap=False)
    assert open_lvl - _at(w, a, b) <= 30.0


def test_A_T8_a_GAP_UNDER_THE_DOOR():
    """A-T8 (medium). The pine door of A-T3, (i) airtight and (ii) with a
    one-voxel gap at the bottom (2.4% of its area); the listener is off the
    line of the gap. Expected: (i) 25-37 dB, (ii) 12-20 dB. Source: 213
    (composite transmission loss: a 1% opening caps a wall near 20 dB),
    210. Before: the gap changed the loss by less than 1 dB."""
    w, a, b = _bedroom(door=False, gap=False)
    open_lvl = _at(w, a, b)
    w, a, b = _bedroom(door=True, gap=False)
    tight = open_lvl - _at(w, a, b)
    w, a, b = _bedroom(door=True, gap=True)
    gapped = open_lvl - _at(w, a, b)
    assert 25.0 <= tight <= 37.0, tight
    assert 12.0 <= gapped <= 20.0, gapped


def test_A_T8_a_door_PART_FULL_leaks_by_its_gaps_not_its_mass():
    """Row A11, A12: a cell that is partly full is partly a hole, and a
    hole passes sound whole. The scene's "shut door" is 97% full: a 3% gap
    caps it near 15 dB (composite TL, source 213), where its mass alone said
    36.2 dB. Sealed round, so the only way is through it."""
    def through(frac):
        w = _ground(80, 20, 30)
        w.fill(40, 41, 0, 20, 1, 30, WOOD, frac=frac)
        return w._level((20, 10, 15), (60, 10, 15), SHOUT_DB)
    clear = _ground(80, 20, 30)._level((20, 10, 15), (60, 10, 15), SHOUT_DB)
    loss = clear - through(0.97)
    assert 13.0 <= loss <= 18.0, loss
    assert clear - through(1.0) > 30.0, "and a full one is a wall"


# ── A12: a hole ─────────────────────────────────────────────────────────────

def test_A12_a_PINHOLE_passes_its_share_and_not_a_doorway_s():
    """Row A12. One 5 cm voxel missing from a sealed 20 cm stone wall 3 m
    wide and 2.1 m tall (6.3 m2), with the shouter and the listener 2.5 m
    either side of it. A hole passes the sound that falls on it and no
    more: composite TL 10 log10(6.3 / 0.0025) = 34 dB (source 213; the
    wall's own 61.6 dB adds nothing), +- 5. Before: about 6 dB, nearly an
    open doorway, because the way round did not know how big a hole was.

    The sim charges the hole against the first Fresnel zone, about 2.6 m2
    here, not the whole wall (`_opening`), and comes out near 30 dB. A
    doorway in the same wall must still cost what A-T2 allows.

    And the hole costs the same however it is met. The same hole in a 5 cm
    wall, with the straight line from mouth to ear going right through it,
    cost NOTHING before (only a way round was ever charged for a hole), so
    a peephole passed a shout whole. A hole four times the area passes
    four times the sound: 6 dB less, as composite TL says."""
    def scene(wall, hole=None, thick=4):
        w = _ground(200, 60, roof=True)
        if wall:
            w.fill(100, 100 + thick, 0, 60, 1, 43, STONE)
        if hole is not None:
            (y0, y1), (z0, z1) = hole
            w.mat[100:100 + thick, y0:y1, z0:z1] = 0
            w.smass[100:100 + thick, y0:y1, z0:z1] = 0.0
        return w, *_two(w, (50, 30), (150, 30))
    w, a, b = scene(False)
    clear = _at(w, a, b)
    w, a, b = scene(True)
    sealed = clear - _at(w, a, b)
    w, a, b = scene(True, hole=((30, 31), (30, 31)))
    pinhole = clear - _at(w, a, b)
    w, a, b = scene(True, hole=((20, 40), (1, 43)))
    doorway = clear - _at(w, a, b)
    assert sealed > 55.0, sealed
    assert 29.0 <= pinhole <= 39.0, pinhole
    assert doorway <= 17.0, doorway
    # head on, through a 5 cm wall: the straight line is open all the way
    w, a, b = scene(True, hole=((30, 31), (30, 31)), thick=1)
    p, q = np.asarray(a["_eye"], float), np.asarray(w._ear(b["_eye"]), float)
    line = np.round(p[None] + np.linspace(0.0, 1.0, 101)[:, None] * (q - p))
    assert all(w.mat[tuple(c)] == 0 for c in line.astype(int)
               if c[0] == 100), "the line goes through the hole"
    thin = clear - _at(w, a, b)
    assert abs(thin - pinhole) <= 1.0, (thin, pinhole)
    w, a, b = scene(True, hole=((29, 31), (29, 31)), thick=1)
    four = clear - _at(w, a, b)
    assert 4.0 <= thin - four <= 8.0, (thin, four)


def test_A12_a_hole_MET_HEAD_ON_costs_its_share_in_a_PLANK_DOOR_or_a_THICK_WALL():
    """Row A12 again, with the straight line from mouth to ear going right
    through the hole, so no way round is ever looked for. A 20 cm stone
    wall across a room 2.1 m tall, the shouter and the listener 2.5 m
    either side of it, and in it:

    - a doorway 1 m wide shut by a 5 cm pine plank door, with one 5 cm
      knot hole in the plank on the line;
    - no doorway, and one 5 cm hole bored through all 20 cm of stone on
      the line;
    - the doorway left open.

    Expected: both holes cost the composite TL of a 5 cm hole, 34 dB +- 5
    (source 213; the sim gets about 30, see the A12 test above), and the
    same to within 1 dB: it is how narrow a hole is that keeps sound out,
    not how deep. The open doorway costs 0-3 dB: it is most of the
    Fresnel zone. Before: the knot hole cost nothing at all (a hole was
    charged only on a way round, and the line through it was open), while
    the hole through 20 cm of stone cost 30 dB."""
    def scene(door=None, bore=False):
        w = _ground(200, 60, roof=True)
        w.fill(100, 104, 0, 60, 1, 43, STONE)
        if door is not None:
            w.mat[100:104, 20:40, 1:43] = 0
            w.smass[100:104, 20:40, 1:43] = 0.0
            if door == "plank":
                w.fill(103, 104, 20, 40, 1, 43, WOOD)
        a, b = _two(w, (50, 30), (150, 30))
        p = np.asarray(a["_eye"], np.float32)
        q = np.asarray(w._ear(b["_eye"]), np.float32)
        n = int(np.ceil(np.abs(q - p).max())) + 1
        line = np.round(p[None] + np.linspace(0.0, 1.0, n)[:, None]
                        * (q - p)[None]).astype(int)
        through = [tuple(c) for c in line if 100 <= c[0] < 104]
        if bore:                                     # the hole, on the line
            for c in through:                        # as `_level` samples it
                w.mat[c] = 0
                w.smass[c] = 0.0
        assert all(w.mat[c] == 0 for c in through) == (bore or door == "open")
        return w, a, b
    w = _ground(200, 60, roof=True)
    a, b = _two(w, (50, 30), (150, 30))
    clear = _at(w, a, b)
    w, a, b = scene(door="plank")
    plank = clear - _at(w, a, b)
    w, a, b = scene(door="plank", bore=True)
    knot = clear - _at(w, a, b)
    w, a, b = scene(bore=True)
    bored = clear - _at(w, a, b)
    w, a, b = scene(door="open")
    doorway = clear - _at(w, a, b)
    assert 29.0 <= knot <= 39.0, knot
    assert 29.0 <= bored <= 39.0, bored
    assert abs(knot - bored) <= 1.0, (knot, bored)
    assert knot < plank, (knot, plank)
    assert 0.0 <= doorway <= 3.0, doorway


def test_a_CORRIDOR_is_not_a_hole_to_the_people_in_it():
    """The other side of the pinhole rule. Two people 6.5 m apart down a
    stone corridor 30 cm wide: the corridor is far narrower than the
    Fresnel zone between them (about 1 m across at 500 Hz), but its walls
    are mirrors and carry the sound along, so it is no hole to them (it has
    no mouths between them). They hear each other exactly as in a corridor
    1 m wide. Joined to two rooms, the same 30 cm is a hole, with both its
    mouths on the way, and costs.

    The worlds are kept small and freed one at a time: a World refers to
    itself through its particle solver, so a world dropped by rebinding
    lives on until a collection, and three 1.5 M-cell worlds at once took
    the suite past 3 GB."""
    def corridor(width, rooms=False):
        w = _ground(160, 60, roof=True)
        w.fill(0, 160, 0, 60, 1, 43, STONE)
        y0 = 30 - width // 2
        x0, x1 = (65, 95) if rooms else (10, 150)
        w.mat[x0:x1, y0:y0 + width, 1:43] = 0
        w.smass[x0:x1, y0:y0 + width, 1:43] = 0.0
        if rooms:
            for r0, r1 in ((10, 65), (95, 150)):
                w.mat[r0:r1, 1:59, 1:43] = 0
                w.smass[r0:r1, 1:59, 1:43] = 0.0
        return w, *_two(w, (20, 30), (140, 30))

    def level(width, rooms=False):
        w, a, b = corridor(width, rooms)
        db = _at(w, a, b)
        del w, a, b
        gc.collect()
        return db
    wide, narrow = level(20), level(6)
    assert abs(wide - narrow) < 0.5, (wide, narrow)
    joined = level(6, rooms=True)
    assert narrow - joined > 5.0, (narrow, joined)


# ── A-T4 ────────────────────────────────────────────────────────────────────

def test_A_T4_a_BYSTANDER_is_not_a_wall():
    """A-T4 (high). A tavern: shouter and listener 4 m apart, then one and
    two people on the line between them. Expected: 0-6 dB for one (a
    head-shadow proxy, unverified as an insertion loss), at most 10 dB for
    two, and the line still understood at 4 m. Source: 215 (head shadow
    1.6 dB at 500 Hz, 2.7-4 dB at 2-4 kHz), 214. Before: 46-50 dB a head."""
    def scene(between):
        w = _ground(140, 40, roof=True)
        for x in between:
            _person(w, x, 20)["name"] = "drinker"
        a, b = _two(w, (20, 20), (100, 20))
        return w, a, b
    w, a, b = scene(())
    alone = _at(w, a, b)
    w, a, b = scene((60,))
    one = alone - _at(w, a, b)
    w, a, b = scene((45, 75))
    two = alone - _at(w, a, b)
    assert 0.0 <= one <= 6.0, one
    assert 0.0 <= two <= 10.0, two
    assert w._hears(a["_eye"], b["_eye"]), "understood past two of them"


# ── A-T5 ────────────────────────────────────────────────────────────────────

@pytest.mark.xfail(strict=True, reason=(
    "needs a fire sound: fire never calls `_noise` (row A21, G18), so a "
    "sleeper is woken by a fire only when something falls or breaks"))
def test_A_T5_a_FIRE_is_LOUD_ENOUGH_to_wake_a_sleeper_3_m_off():
    """A-T5 (high). An adult asleep with a wood fire lit 3 m away in the
    same room should wake from its crackle, well before about 217 s (NIST
    time to untenable conditions, flaming fires). The crackle's real level
    is unverified: the lab played 42-48 dBA at the pillow and woke 91% of
    adults within 30 s. Source: 204, 247, 248, 80.

    What is missing is the sound, not the waking (`_law_hear` wakes anyone
    a sound reaches loud enough), so this asks only that: a crib fire,
    while it burns, makes a noise that is at least 42 dB 3 m away. No
    sleeper in the scene: a body would only slow the run."""
    w = _ground(60, 60, 40)
    w.fill(26, 32, 26, 32, 1, 6, WOOD, frac=0.7)
    w.E[28, 28, 2] = 9.0e5
    loudest = None
    for _ in range(60):
        w.step()
        for snd in w.sounds:
            if snd["tick"] == w.tick - 1:
                lvl = w._level(snd["at"], (snd["at"][0] + 60.0,
                                           snd["at"][1], 30.0), snd["db"])
                loudest = lvl if loudest is None else max(loudest, lvl)
    assert w.burning().any() or any(e.verb == "ignites"
                                    for e in w.events.rows), "it burned"
    assert loudest is not None and loudest >= 42.0, loudest


# ── A-T6 ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("deep,frac", [(4, 0.30), (20, 0.30), (20, 1.0)])
def test_A_T6_a_HEDGE_does_not_silence_a_shout(deep, frac):
    """A-T6 (high). Two people 4 m apart outdoors with a hedge between them,
    taller than they are, blocking the view: the scene hedge (LEAF at frac
    0.3, 4 voxels deep), a 1 m hedge at frac 0.3 and a 1 m dense one.
    Expected: the hedge costs 0-3 dB. Source: 207 (ISO 9613-2 counts
    foliage only on paths of 10 m or more: 1 dB for 10-20 m). Before:
    28.6 dB (the scene hedge) to 42.5 dB."""
    def scene(hedge):
        w = _ground(140, 40, 60)
        if hedge:
            x0 = 70 - deep // 2
            w.fill(x0, x0 + deep, 0, 40, 1, 40, LEAF, frac=frac)
        return w, *_two(w, (20, 20), (100, 20))
    w, a, b = scene(False)
    clear = _at(w, a, b)
    w, a, b = scene(True)
    if frac == 1.0:
        assert not w._sees(a["_eye"], b["_eye"]), "the dense one hides them"
    assert 0.0 <= clear - _at(w, a, b) <= 3.0, (clear, _at(w, a, b))


# ── A-T7 ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mat,thick,lo,hi", [(WOOD, 1, 28.0, 34.0),
                                             (STONE, 2, 44.0, 52.0),
                                             (STONE, 4, 52.0, 58.0)])
def test_A_T7_a_WALL_costs_what_its_MASS_says(mat, thick, lo, hi):
    """A-T7 (medium). Shouter and listener 2 m apart with one partition,
    sealed to the floor, the ceiling and both side walls: (a) 5 cm pine,
    (b) 10 cm stone, (c) 20 cm stone. Expected (a) 28-34 dB, (b) 44-52,
    (c) 52-58, each +- 5. Source: 209 (25 mm floor 28 dB, 50 mm boarding
    32), 210 (precast concrete STC 49 / 55 / 58 at 100 / 150 / 200 mm),
    211. The mass law here sits 2-5 dB above those (rows A9, A10: "ok for
    game"): 36.5, 55.6 and 61.6 dB."""
    def scene(wall):
        w = _ground(100, 30, roof=True)
        if wall:
            w.fill(50, 50 + thick, 0, 30, 1, 43, mat)
        return w, *_two(w, (27, 15), (67, 15))
    w, a, b = scene(False)
    clear = _at(w, a, b)
    w, a, b = scene(True)
    loss = clear - _at(w, a, b)
    assert lo - 5.0 <= loss <= hi + 5.0, loss


# ── A-T9 ────────────────────────────────────────────────────────────────────

def test_A_T9_a_BRAWL_masks_speech():
    """A-T9 (medium). A tavern with a crowd or brawl at 70-75 dBA. A shout
    from 5 m (SNR about 0 to +6 dB) is understood; a normal voice (58 dBA at
    1 m) from 2 m (SNR about -18 to -23 dB) is not; lines start to fail
    below about -3 dB SNR. Source: 200, 199 (SNR 0-3 dB "satisfactory",
    below -3 "insufficient"). Before: no masking; every line was heard.

    The crowd is the room's ambient level (`w.ambient_db`): a brawl in the
    sim makes 50 ms knocks, not a sustained noise (G18), so it is set as
    the room's level, as a scene would set it."""
    def scene(gap, ambient=None):
        w = _ground(40 + gap, 30, roof=True)
        if ambient is not None:
            w.ambient_db = ambient
        a, b = _two(w, (10, 15), (10 + gap, 15))
        return w, a, b
    w, a, b = scene(100, ambient=72.0)                   # a shout from 5 m
    assert w._hears(a["_eye"], b["_eye"]), "the shout is made out"
    w, a, b = scene(40, ambient=72.0)                    # a voice from 2 m
    assert not w._hears(a["_eye"], b["_eye"], 58.0), "the voice is lost"
    w, a, b = scene(40)                                  # in a quiet room
    assert w._hears(a["_eye"], b["_eye"], 58.0), "and heard in a quiet one"
    # where the line is lost: about -3 dB under the noise
    for gap, heard in ((150, True), (300, False)):       # 7.5 m: -1 dB,
        w, a, b = scene(gap, ambient=72.0)               # 15 m: -7.5 dB
        snr = _at(w, a, b) - 72.0
        assert w._hears(a["_eye"], b["_eye"]) is heard, (gap, snr)
    assert SOUND["ambient_db"] < HEAR_DB, "a quiet room masks nothing"


# ── MINDS AT REST (wave 12): batched hearing, the voice, idle wandering ────

def _house_of_sounds():
    """A house with every kind of thing sound meets: a stone wall with a
    doorway, a pine door with a gap under it, a wall with a pinhole, a hedge,
    and three people standing about in it. Small, so the whole law runs in a
    second."""
    from src.voxel.sim import LEAF
    w = _ground(100, 50, nz=30, roof=True)
    w.fill(30, 32, 0, 50, 1, 29, STONE)
    w.mat[30:32, 20:30, 1:25] = 0
    w.smass[30:32, 20:30, 1:25] = 0.0
    w.fill(30, 31, 20, 30, 2, 25, WOOD)                    # a door, gap below
    w.fill(60, 61, 0, 50, 1, 29, WOOD)
    w.mat[60:61, 10:11, 12:13] = 0                          # a pinhole
    w.smass[60:61, 10:11, 12:13] = 0.0
    w.fill(75, 80, 0, 50, 1, 20, LEAF, frac=0.4)            # a hedge
    people = []
    for x, y in ((10, 25), (45, 25), (90, 40)):
        people.append(_person(w, x, y))
    for i, p in enumerate(people):
        p["name"] = f"hearer{i}"
    w.step()
    return w, people


def test_BATCHED_hearing_is_the_PLAIN_law_to_the_last_bit():
    """The compiled batch (`_levels`) gives the very numbers `_level` gives
    pair by pair: walls by their mass, gaps, bodies, a hedge, a hole, the
    way round a doorway; with and without the speaker's and the listener's
    own bodies in the way. Bitwise, over many pairs. The plain law is the
    definition (`fused=False`)."""
    w, people = _house_of_sounds()
    if not w._fused_hearing():
        pytest.skip("numba's logs round differently from numpy's here, so "
                    "the plain law is the one in use")
    by_pid = {q["pid"]: q for q in w.persons}
    rng = np.random.RandomState(5)
    pairs = []
    for _ in range(120):
        at = (float(rng.randint(2, 98)), float(rng.randint(2, 48)),
              float(rng.randint(1, 26)))
        p = people[rng.randint(3)]
        src = people[rng.randint(3)]["pid"] if rng.rand() < 0.5 else None
        pairs.append((p, p["_eye"], {"at": at, "db": float(
            rng.choice([30.0, 40.0, 60.0, 75.0, 90.0, 105.0]))}, src))
    w.fused = True
    batch = w._hear_pairs(pairs, by_pid)
    w.fused = False
    plain = w._hear_pairs(pairs, by_pid)
    assert batch == plain
    assert len(set(plain)) > 20, "the pairs are not all alike"
    # and the plain batch is `_level` itself, pair by pair
    p, ear, snd, src = pairs[0]
    own = set(p["_own"]) | (set(by_pid[src]["_own"]) if src is not None
                            else set())
    assert plain[0] == w._at_ear(snd["at"], ear, snd["db"], own,
                                 floor=SOUND["notice_db"])


def test_BATCHED_hearing_changes_nothing_a_listener_notices():
    """`_law_hear` reads the batch in the order it always read pairs: the
    same person hears the same loudest sound, whichever path computed it."""
    def run(fused):
        w, people = _house_of_sounds()
        w.fused = fused
        for i, p in enumerate(people):
            w._noise((20.0 + 30 * i, 25.0, 5.0), 3.0, [STONE, STONE])
        w._noise((70.0, 30.0, 4.0), 10.0, [IRON, STONE])
        w._law_hear()
        return [(p["name"], (p.get("noise") or {}).get("db")) for p in people]
    assert run(True) == run(False)


def _voices(db_name, gap, wall=False):
    """Two people `gap` voxels apart in a quiet room (and a pine door of
    5 cm between them if `wall`); is the line `db_name` understood?"""
    from src.voxel.policy import VOICE_DB
    w = _ground(40 + gap, 30, roof=True)
    if wall:
        mid = 20 + gap // 2
        w.fill(mid, mid + 1, 0, 30, 1, 43, WOOD)
    a, b = _two(w, (10, 15), (10 + gap, 15))
    return w, a, b, w._hears(a["_eye"], b["_eye"], VOICE_DB[db_name])


def test_a_WHISPER_is_heard_across_a_TABLE_and_NOT_through_a_DOOR():
    """A whisper (about 40 dB at a metre) is made out by someone a table's
    width away in a quiet room, and is lost the moment a door is between
    them; the same door lets a call out through, and a shout. The sound law
    decides, the voice only says how loud the line starts."""
    w, a, b, ok = _voices("whisper", 12)                 # 0.6 m of floor
    assert ok, "a whisper across a table"
    # not through a door (a pine door in the way, 1 m apart)
    w, a, b, ok = _voices("whisper", 20, wall=True)
    assert not ok, "a whisper does not go through a door"
    w, a, b, ok = _voices("shout", 20, wall=True)
    assert ok, "a shout does"
    # and by the real act, not the helper: the listener's heard row has the
    # words only when the line was made out
    for voice, words in (("whisper", False), ("shout", True)):
        w = _ground(80, 30, roof=True)
        w.fill(30, 31, 0, 30, 1, 43, WOOD)
        a, b = _two(w, (20, 15), (45, 15))
        a["lines"] = {"fire": "Fire!"}
        w._say(a, "fire", voice)
        w._law_hear()
        rows = [r for r in w.minds[b["pid"]].perceived if r.how == "heard"]
        got = bool(rows) and dict(rows[-1].data).get("line") == "fire"
        assert got is words, (voice, rows)


def test_a_VOICE_carries_as_far_as_its_level_says():
    """The ladder: whisper, say quietly, say, call out, shout, each
    understood further than the last in open air (6 dB a doubling to the
    floor of 40 dB), at about 1, 3, 10, 56 and 316 m."""
    from src.voxel.policy import VOICE_DB
    reach = {}
    for name, db in VOICE_DB.items():
        reach[name] = 10.0 ** ((db - HEAR_DB) / 20.0)
    ordered = [reach[n] for n in ("whisper", "say quietly", "say",
                                  "call out", "shout")]
    assert ordered == sorted(ordered)
    assert 0.9 <= reach["whisper"] <= 1.1
    assert 2.5 <= reach["say quietly"] <= 4.0
    assert 8.0 <= reach["say"] <= 12.0
    # and the world agrees with the arithmetic: a normal voice 8 m off is
    # understood, 14 m off is not (a quiet room)
    for gap, heard in ((160, True), (280, False)):
        w = _ground(40 + gap, 30, roof=True)
        a, b = _two(w, (10, 15), (10 + gap, 15))
        assert w._hears(a["_eye"], b["_eye"], VOICE_DB["say"]) is heard, gap


def test_a_SHOUT_is_heard_in_the_NEXT_ROOM_and_a_whisper_is_not():
    """Through a closed pine door with a gap under it (the bedroom of A-T3):
    a shout arrives at 50-70 dB, well over the floor, and is made out; a
    whisper or a quiet word is not."""
    from src.voxel.policy import VOICE_DB
    w, a, b = _bedroom(door=True, gap=True)
    heard = {v: w._hears(a["_eye"], b["_eye"], db)
             for v, db in VOICE_DB.items()}
    assert heard["shout"] and heard["call out"]
    assert not heard["whisper"] and not heard["say quietly"]


def test_a_spoken_line_is_a_SOUND_of_its_own_level():
    """`_say` puts the line into the sound record at the level of the voice,
    and its `says` event carries that level."""
    from src.voxel.policy import VOICE_DB
    w = _ground(60, 30, roof=True)
    a, b = _two(w, (10, 15), (40, 15))
    a["lines"] = {"fire": "Fire!", "hello": "Hello."}
    for voice in VOICE_DB:
        w.sounds.clear()
        w._say(a, "hello", voice)
        assert w.sounds[-1]["db"] == VOICE_DB[voice]
        ev = [e for e in w.events.rows if e.verb == "says"][-1]
        assert ev.db == VOICE_DB[voice]
    # a line with no level given is said at its own: an alarm shouted, a
    # reply called, anything else said
    w._say(a, "fire")
    assert w.sounds[-1]["db"] == VOICE_DB["shout"]
    w._say(a, "hello")
    assert w.sounds[-1]["db"] == VOICE_DB["say"]


def test_the_LOMBARD_reflex_raises_the_voice_in_noise_and_not_in_quiet():
    """A speaker in a loud room speaks up without choosing to: half a dB
    for each dB of noise over 45, as far as a throat goes (100 dB). In the
    quiet nothing is added. A chosen whisper is not lifted (the chosen act
    wins over the reflex), and the lift is in the `says` event."""
    from src.voxel.policy import LOMBARD, VOICE_DB
    w = _ground(60, 30, roof=True)
    a, b = _two(w, (10, 15), (40, 15))
    a["lines"] = {"hello": "Hello."}
    w._say(a, "hello", "say")
    assert w.sounds[-1]["db"] == VOICE_DB["say"], "a quiet room adds nothing"
    w.ambient_db = 75.0                                   # a brawl
    w._say(a, "hello", "say")
    want = VOICE_DB["say"] + LOMBARD["slope"] * (75.0 - LOMBARD["from_db"])
    assert abs(w.sounds[-1]["db"] - want) < 1e-9 and want > 70.0
    ev = [e for e in w.events.rows if e.verb == "says"][-1]
    assert dict(ev.data)["lombard_db"] == 15.0 and ev.db == w.sounds[-1]["db"]
    w._say(a, "hello", "whisper")                         # chosen quiet wins
    assert w.sounds[-1]["db"] == VOICE_DB["whisper"]
    w.ambient_db = 110.0
    w._say(a, "hello", "shout")                           # never past a throat
    assert w.sounds[-1]["db"] == LOMBARD["max_db"]


def test_the_MOUTH_menu_offers_every_line_at_every_loudness():
    """Whisper to shout, a row each, the line's own loudness first and under
    the tag the reflex table takes (`say:fire`); the rest are
    `say:fire@whisper` and so on. The reflex shouts the alarm as it did."""
    from src.voxel.policy import VOICE, VOICE_DB

    class Rec(TablePolicy):
        menus = []

        def pick(self, sit, menu):
            if sit["limb"] == "mouth":
                self.menus.append((sit["percept"], menu))
            return super().pick(sit, menu)
    w = _ground(60, 30, roof=True)
    w.policy = Rec()
    w.exits = [(55, 15)]
    p = _person(w, 10, 15)
    p["name"] = "caller"
    p["lines"] = {"fire": "Fire!"}
    w.step()
    assert Rec.menus, "the mouth was asked"
    menu = Rec.menus[0][1]
    rows = [m for m in menu if m["verb"] == "say"]
    assert [m["voice"] for m in rows][0] == "shout", "its own loudness first"
    assert {m["voice"] for m in rows} == {v for v, _ in VOICE}
    assert {m["db"] for m in rows} == set(VOICE_DB.values())
    assert [m["tag"] for m in rows][0] == "say:fire"
    assert "say:fire@whisper" in {m["tag"] for m in rows}
    assert len({m["key"] for m in rows}) == len(rows), "each row reads differently"


def test_a_REPLY_is_offered_only_to_a_body_that_HEARD_something_to_answer():
    """Answering presupposes something to answer. The mouth keys now carry the
    loudness (`say(coming, call out)`), so the old guard in test_voxel (the
    string `say(coming)` is not in the menu) can no longer fail. This is the
    same guard written against the new keys, with a control that shows the
    reply IS offered once an alarm has been heard."""
    class Rec(TablePolicy):
        def __init__(self):
            self.mouth = {}

        def pick(self, sit, menu):
            if sit["limb"] == "mouth":
                self.mouth[len(self.mouth)] = menu
            return super().pick(sit, menu)

    def mouth_keys(pol):
        return {m["key"] for menu in pol.mouth.values() for m in menu}

    # (1) a body that SEES a fire and heard nobody: no reply row at any loudness
    w = _ground(60, 16, roof=False)
    pol = Rec()
    w.policy = pol
    w.exits = [(55, 8)]
    w.fill(48, 50, 6, 8, 1, 4, WOOD)
    w.E[48, 6, 1] = 9.0e5
    p = _person(w, 40, 7)
    p["name"] = "A"
    for _ in range(40):
        w.step()
    assert pol.mouth, "the mouth was asked"
    assert not any(k.startswith("say(coming") for k in mouth_keys(pol)), \
        "this body heard nobody: there is nothing to answer"
    del w

    # (2) control: a body that HEARS the alarm is offered the reply
    w = _ground(60, 16, roof=False)
    pol = Rec()
    w.policy = pol
    w.exits = [(55, 8)]
    a = _person(w, 6, 7)
    a["name"] = "A"
    b = _person(w, 16, 7)
    b["name"] = "B"
    a["surprise_s"] = 0.1
    w.step()
    w._say(b, "fire")
    for _ in range(30):
        w.step()
    assert any(k.startswith("say(coming") for k in mouth_keys(pol)), \
        "a body that heard the alarm may answer it; the control for the guard"


HALL = (44, 28, 40)                                       # x, y, z cells inside


def _hall():
    """A hall with a roof and one doorway in its east wall onto open ground:
    the house of fire_alarm, small: 44 by 28 cells of floor (2.2 m by 1.4 m)
    and 2 m under the roof."""
    nx, ny, nz = HALL
    w = World(nx + 30, ny + 20, nz + 4, voxel_cm=5)
    w.fill(0, nx + 30, 0, ny + 20, 0, 1, STONE)
    w.fill(10, 10 + nx, 10, 10 + ny, 1, nz + 1, STONE)
    w.mat[11:9 + nx, 11:9 + ny, 1:nz] = 0
    w.smass[11:9 + nx, 11:9 + ny, 1:nz] = 0.0
    mid = 10 + ny // 2
    w.mat[9 + nx:10 + nx, mid - 6:mid + 6, 1:36] = 0       # the doorway
    w.smass[9 + nx:10 + nx, mid - 6:mid + 6, 1:36] = 0.0
    w.exits = [(11 + nx, mid)]
    return w


def _in_hall(x, y):
    nx, ny, _ = HALL
    return 11 <= x < 10 + nx and 11 <= y < 10 + ny


def _wanderers(ticks, n=2):
    """`n` idle people in the hall, all strangers to it, for `ticks`. Returns
    the world, them, and every cell any of them stood in."""
    w = _hall()
    ps = []
    for i in range(n):
        p = _person(w, 16 + 9 * i, 16 + 4 * i)
        p["name"] = f"idler{i}"
        ps.append(p)
    seen = set()
    for _ in range(ticks):
        w.step()
        for p in ps:
            seen.add(tuple(int(round(v)) for v in p["anchor"][:2]))
    return w, ps, seen


def test_IDLE_people_stay_in_their_ROOM_and_do_not_wander_out_of_the_door():
    """An idle body wanders in the room it is in. fire_alarm's Berel walked
    out of the open front door at the first decision, because the spot
    `_places` named was any spot he had seen, and the yard is in view of the
    door. Now idle strangers in a hall with its door open, for half a minute
    of sim time, never step past the doorway; and they do move about."""
    w, ps, seen = _wanderers(800)
    out = sorted(c for c in seen if not _in_hall(*c))
    assert not out, out[:8]
    assert len(seen) > 30, "and they did wander"
    assert not any(p["safe"] for p in ps)


def test_leaving_the_room_is_still_on_the_menu_by_choice():
    """Holding the idle table inside the room is not a gate on the character:
    the way out is a row of the menu, a drive's reflex takes it, and a mind
    that wants to can pick it. The wander rows say whether the whole way
    stays in the room; the rest is the chooser's."""
    class Rec(TablePolicy):
        legs = []

        def pick(self, sit, menu):
            if sit["limb"] == "legs":
                self.legs.append((sit["percept"], menu))
            return super().pick(sit, menu)
    w = _hall()
    w.policy = Rec()
    p = _person(w, 44, 24)                       # by the door
    p["name"] = "idler"
    p["known"][:] = True                         # knows the whole place
    for _ in range(12):
        w.step()
    idle = [m for pc, m in Rec.legs if pc is None]
    assert idle, "the legs were asked while idle"
    tags = {r["tag"] for menu in idle for r in menu}
    assert "go:exit" in tags, "the door is a row"
    rows = [r for menu in idle for r in menu if r["tag"] in
            ("go:roam", "go:step", "go:frontier")]
    assert rows and all("inside" in r for r in rows)
    # the way out is a row that is not marked as staying in
    assert not any(r.get("inside") is False and r["tag"] == "go:roam"
                   for r in rows), "the roam spot is drawn from the room"


def test_N_IDLE_people_for_TEN_SIM_MINUTES_never_leave_their_room():
    """The same as above for ten minutes of sim time (24,000 ticks at 25 ms)
    and four people. Slow tier."""
    w, ps, seen = _wanderers(24000, n=4)
    out = sorted(c for c in seen if not _in_hall(*c))
    assert not out, out[:8]
    assert not any(p["safe"] for p in ps)
