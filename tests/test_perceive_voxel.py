"""New in V3 (docs/sim-mind-design.md, 3.4 and 4.4): perception.

Every tick `_law_perceive` writes what reached each person's senses into
their Mind, as rows (`Rec`) that each name the event that caused them. The
decision models will read only these rows, so what they must get right is
what a body could and could not have known: a sound heard through a wall
says nothing of who made it, the unconscious perceive nothing at all, and
each percept of `_law_will` points at the event it came from.
"""
import numpy as np

from src.voxel.events import _Pending
from src.voxel.sim import (BODY, GLASS, LEAF, STONE, WOOD, TablePolicy,
                           World)
from src.voxel.scenes import _person


def _rows(w, p, how=None):
    return [r for r in w.minds[p["pid"]].perceived
            if how is None or r.how == how]


# ── hearing ─────────────────────────────────────────────────────────────────

class _Post(TablePolicy):
    """The guard keeps his post; everyone else does as the table says."""
    def pick(self, sit, menu):
        if sit["who"] == "guard" and sit["limb"] == "legs":
            return 0
        return super().pick(sit, menu)


def _footsteps_past(partition):
    """A man walks along one side of a stone partition; a guard stands
    2.5 m off on the other side, facing it. The partition is hung short:
    a 15 cm slot runs under it, like the gap under a door."""
    w = World(90, 40, 40, voxel_cm=5)
    w.fill(0, 90, 0, 40, 0, 1, STONE)
    if partition:
        w.fill(30, 31, 0, 40, 1, 40, STONE)
        w.mat[30, 6:24, 1:4] = 0                # the slot (AIR)
        w.smass[30, 6:24, 1:4] = 0.0
    w.exits = []
    guard = _person(w, 80, 15)
    guard["name"] = "guard"
    guard["facing"] = (-1.0, 0.0)              # looking toward the walker
    walker = _person(w, 27, 8)
    walker["name"] = "walker"
    walker["goal"] = (27, 22)                  # along the partition
    w.policy = _Post()
    for _ in range(50):
        w.step()
    return w, guard, walker


def test_a_FOOTSTEP_THROUGH_A_WALL_is_heard_but_says_NOT_WHO_and_places_NOBODY():
    """The footsteps come under the partition and the guard hears them. He
    cannot see who it is: at eye height the partition is solid. So the rows
    say a footstep and where it seemed to be, never the walker, and nothing
    gives the guard a belief about where the walker is. Without the
    partition the same guard does see him -- so it is the wall that hides
    him, not the distance or the way the guard faces."""
    w, guard, walker = _footsteps_past(partition=True)
    steps = [r for r in _rows(w, guard, "heard") if r.verb == "steps"]
    assert steps, "the footsteps come under the partition and are heard"
    for r in steps:
        assert r.actor_seen is None, "a footstep never says whose it is"
        src = w.events.get(r.ev)
        assert src is not None and src.verb == "steps" \
            and src.actor == walker["pid"], "but the row knows its cause"
    assert guard["noise"]["ev_id"] in {r.ev for r in steps}
    assert ("at", walker["pid"]) not in w.minds[guard["pid"]].bel, \
        "hearing someone is not knowing where they are"
    assert not [r for r in _rows(w, guard)
                if r.how in ("saw", "looked") and r.actor_seen == walker["pid"]]
    w, guard, walker = _footsteps_past(partition=False)
    assert ("at", walker["pid"]) in w.minds[guard["pid"]].bel, \
        "in the open the guard sees him"


def _bang_beside(w, p):
    """A crash (about 100 dB at a metre: test_sound.py) on the floor beside
    the person, with no event behind it."""
    e = p["_eye"]
    return w._noise((e[0] + 3.0, e[1], 1.0), 10.0, [GLASS, STONE])


def test_a_SLEEPER_WOKEN_BY_A_CRASH_knows_the_crash_is_why():
    """A pane falls onto the floor in the room where a man is asleep. The
    crash wakes him, and the `wakes` event names the event the crash was the
    noise of -- a `lands` or `breaks` the physical laws queue only at the
    end of the tick, after he has already woken. His `heard` row points at
    the same event."""
    w = World(50, 20, 30, voxel_cm=5)
    w.fill(0, 50, 0, 20, 0, 1, STONE)
    w.exits = [(48, 10)]
    sleeper = _person(w, 38, 10)
    sleeper["name"] = "sleeper"
    sleeper["asleep"] = True
    w.fill(10, 13, 8, 12, 20, 21, GLASS)       # a pane, falling from 1 m
    for _ in range(60):
        w.step()
        if not sleeper["asleep"]:
            break
    assert not sleeper["asleep"], "the crash woke him"
    woke = [e for e in w.events.rows if e.verb == "wakes"
            and e.actor == sleeper["pid"]]
    assert len(woke) == 1
    cause = w.events.get(woke[0].cause) if woke[0].cause is not None else None
    assert cause is not None and cause.verb in ("lands", "breaks"), woke[0]
    assert cause.tick == woke[0].tick or cause.tick == woke[0].tick - 1
    heard = [r for r in _rows(w, sleeper, "heard") if r.ev == cause.id]
    assert heard and dict(heard[0].data).get("asleep"), \
        "the row of what woke him, heard in his sleep"
    assert not any(isinstance(r.ev, _Pending) for r in _rows(w, sleeper)), \
        "nothing a tick leaves behind still holds a box"


def test_an_UNCONSCIOUS_LISTENER_gets_NO_ROWS_but_the_NOISE_still_reaches_him():
    """3.4, "who perceives": the unconscious get no rows of any kind. But
    `p["noise"]` is today's channel and is not perception's to change: it
    still updates, as it always has, and now carries the event it came
    from (here none: a bare noise)."""
    w = World(40, 40, 30, voxel_cm=5)
    w.fill(0, 40, 0, 40, 0, 1, STONE)
    w.exits = []
    p = _person(w, 20, 20)
    q = _person(w, 10, 20)                      # someone to see, if he could
    p["name"], q["name"] = "p", "q"
    w.step()
    before = len(_rows(w, p))
    p["awake"] = False
    snd = _bang_beside(w, p)
    w.emit("says", actor=q["pid"], result="fire", text="")   # in plain view
    w._law_perceive()
    w._tick_end()
    assert len(_rows(w, p)) == before, _rows(w, p)[before:]
    assert p["noise"] is not None and p["noise"]["tick"] == snd["tick"] \
        and p["noise"]["at"] == snd["at"], "the noise reached him all the same"
    assert "ev_id" in p["noise"]


# ── fire ────────────────────────────────────────────────────────────────────

def test_ONE_FIRE_is_ONE_ignites_and_ONE_burns_out():
    """A crib lit at one cell burns for a while and goes out (fire_proto.py
    measured: lit at t0, out at t59). The fire bookkeeping must call that
    one fire from start to end -- not a new fire each tick a cell flickers
    -- and the `burns_out` names the fire it ends."""
    w = World(60, 60, 40, voxel_cm=5)
    w.fill(0, 60, 0, 60, 0, 1, STONE)
    w.fill(26, 32, 26, 32, 1, 6, WOOD, frac=0.7)
    w.E[28, 28, 2] = 9.0e5
    _person(w, 10, 30)                          # someone awake: the gate
    _person(w, 48, 40)
    for t in range(90):
        if t == 30:                             # put out: a fire on the real
            wood = w.mat == WOOD                # clock (G4) burns for minutes
            w.E[wood] = 0.0                     # (a douse, as if all of it
            w._skin_field()[wood] = 0.0         # were cold at once)
        w.step()
    lit = [e for e in w.events.rows if e.verb == "ignites"]
    out = [e for e in w.events.rows if e.verb == "burns_out"]
    assert len(lit) == 1 and len(out) == 1, (lit, out)
    assert out[0].target == ("fire", lit[0].id) and out[0].cause == lit[0].id
    assert out[0].tick > lit[0].tick
    assert w._fires == [], "and nothing is left burning on the books"


def test_SEES_FIRE_points_at_the_fire_it_saw():
    """The `sees_fire` percept of `_law_will` now says which fire. He stands
    2.7 m off: `_sees` starts its ray at the top of his head, and a fire at
    his feet is behind his own body to it (the percept chain is not this
    step's to change)."""
    w = World(100, 60, 40, voxel_cm=5)
    w.fill(0, 100, 0, 60, 0, 1, STONE)
    w.fill(26, 32, 26, 32, 1, 6, WOOD, frac=0.7)
    w.E[31, 29, 5] = 9.0e5                      # its corner nearest him
    p = _person(w, 85, 29)
    p["facing"] = (-1.0, 0.0)                   # toward the crib
    for _ in range(20):
        w.step()
        if p.get("emergency"):
            break
    got = w.minds[p["pid"]].noticed
    lit = [e for e in w.events.rows if e.verb == "ignites"]
    assert got is not None and got[1] == "sees_fire", got
    assert lit and got[2] == lit[0].id


# ── sight ───────────────────────────────────────────────────────────────────

def test_RAYS_agree_with_SEES_on_every_ray():
    """`rays` is `_sees` for many rays at once, and must answer the same:
    through stone (no), glass (yes), a half-full crib (dimmer), a hedge, a
    smoke bank, rays shorter than three cells, rays along an axis and
    corner to corner. 64 of 64."""
    w = World(40, 30, 20, voxel_cm=5)
    w.fill(0, 40, 0, 30, 0, 1, STONE)
    w.fill(15, 16, 0, 30, 1, 12, STONE)         # a wall, with a window
    w.fill(15, 16, 10, 16, 4, 9, GLASS)
    w.fill(25, 28, 5, 25, 1, 6, WOOD, frac=0.4)
    w.fill(8, 10, 2, 28, 1, 15, LEAF, frac=0.3)
    w.smoke[30:38, :, 6:14] = 3.0               # a smoke bank
    rng = np.random.default_rng(3)
    A = rng.uniform([0, 0, 1], [40, 30, 20], (60, 3)).astype(np.float32)
    B = rng.uniform([0, 0, 1], [40, 30, 20], (60, 3)).astype(np.float32)
    A = np.vstack([A, [[5, 5, 5], [5, 5, 5], [2, 13, 6], [1, 1, 1]]])
    B = np.vstack([B, [[6, 5, 5], [7, 6, 5], [35, 13, 6], [39, 29, 19]]])
    want = np.array([w._sees(a, b) for a, b in zip(A, B)])
    got = w.rays(A, B)
    assert (got == want).sum() == 64, np.flatnonzero(got != want)
    assert want.any() and not want.all(), "the rays test both answers"


# ── what each percept came from ─────────────────────────────────────────────

def test_HEARS_ALARM_points_at_the_SHOUT():
    """A man shouts "Fire!" in one room; another stands in the next, 5 m
    off, through an open doorway in the stone wall between them. The
    hearer's `hears_alarm` names the shouter's `says`, not the shouter, not
    the tick. (This was a metre apart in the open while `_hears` charged
    both heads as walls, docs/realism-references.md G1; R2 fixed that.)"""
    w = World(140, 60, 44, voxel_cm=5)
    w.fill(0, 140, 0, 60, 0, 1, STONE)
    w.fill(70, 72, 0, 60, 1, 44, STONE)         # the wall, and its doorway
    w.mat[70:72, 20:40, 1:42] = 0               # (AIR), 1 m by 2.05 m
    w.smass[70:72, 20:40, 1:42] = 0.0
    w.exits = []
    shouter = _person(w, 17, 30)
    shouter["name"] = "shouter"
    hearer = _person(w, 117, 30)
    hearer["name"] = "hearer"
    w.step()
    w._say(shouter, "fire")
    said = w.events.rows[-1]
    assert said.verb == "says" and said.actor == shouter["pid"]
    w.step()
    got = w.minds[hearer["pid"]].noticed
    assert got is not None and got[1] == "hears_alarm", got
    assert got[2] == said.id
    # and the shout is a heard row of his too, its words made out
    heard = [r for r in _rows(w, hearer, "heard") if r.ev == said.id]
    assert heard and dict(heard[0].data)["line"] == "fire"


def test_a_VOICE_is_learned_by_SEEING_and_HEARING_one_shout():
    """The first shout he both sees and hears teaches him the voice; from
    then on a shout he only hears names who it was, and before it none
    did (3.4, "no identity leaks")."""
    w = World(40, 40, 30, voxel_cm=5)
    w.fill(0, 40, 0, 40, 0, 1, STONE)
    w.exits = []
    a, b = _person(w, 10, 20), _person(w, 22, 20)
    a["name"], b["name"] = "a", "b"             # a sound whose maker has the
    b["facing"] = (-1.0, 0.0)                   # listener's name is his own
    w.step()
    w._say(a, "coming")
    first = w.events.rows[-1]
    w.step()
    mb = w.minds[b["pid"]]
    assert ("voice", a["pid"]) in mb.bel
    row = next(r for r in _rows(w, b, "heard") if r.ev == first.id)
    assert row.actor_seen is None, "not yet known when it was heard"
    for _ in range(3):
        w.step()
    w._say(a, "coming")
    second = w.events.rows[-1]
    w.step()
    row = next(r for r in _rows(w, b, "heard") if r.ev == second.id)
    assert row.actor_seen == a["pid"], "a known voice, its words made out"


# ── what is hurting a body ──────────────────────────────────────────────────

def test_a_FAINT_and_a_DEATH_from_WOUNDS_name_the_wound():
    """A blow leaves a wound (`_blow`), the Mind keeps it as harm, and the
    faint and the death it brings name that `wounds` event as their cause
    -- the start of `stops breathing <- wounds <- hits` (3.4)."""
    w = World(40, 40, 30, voxel_cm=5)
    w.fill(0, 40, 0, 40, 0, 1, STONE)
    w.exits = []
    p = _person(w, 20, 20)
    w.step()
    cell = tuple(int(v) for v in np.argwhere(w.mat == 4)[0])   # FLESH
    w._blow(p, cell, 400.0, 1e-4)
    assert isinstance(w.minds[p["pid"]].harm["wounds"], _Pending)
    w._tick_end()
    wound = [e for e in w.events.rows if e.verb == "wounds"][-1]
    assert w.minds[p["pid"]].harm["wounds"] == wound.id, "boxes are unboxed"
    # C7: a bruise no longer faints anyone. The wound is followed by a REAL
    # cause: a head blow (ko_until), then a crushed organ (organ_dead).
    p["hurt"] = BODY["death_burn"] + 0.01
    w._law_life()
    assert p["awake"] and p["alive"], "hurt alone knocks nobody out"
    p["ko_until"] = w.tick + 100
    w._law_life()
    faint = w.events.rows[-1]
    assert faint.verb == "faints" and faint.why == "a blow"
    assert faint.cause == wound.id and faint.link == "carried"
    p["organ_dead"] = "organ"
    w._law_life()
    died = w.events.rows[-1]
    assert died.verb == "dies" and died.cause == wound.id


# ── the particle matter ─────────────────────────────────────────────────────

def test_WHAT_THE_PARTICLES_DID_becomes_ONE_EVENT_A_RECORD():
    """`_after_raster` turns each record of `matter.events()` into one typed
    event (struck -> strikes, and so on), in the tick it happened, at its
    cell. A wooden block dropped 1.4 m strikes the floor (test_mpm_events)."""
    w = World(12, 12, 30)
    w.fill(0, 12, 0, 12, 0, 1, STONE)
    w.fill(5, 7, 5, 7, 15, 17, WOOD)
    cells = np.argwhere(w.mat == WOOD)
    w.matter.promote(cells)
    recs = []
    for _ in range(30):
        w.step()
        recs += w.matter.events()
    verb = {"touch": "touches", "struck": "strikes", "crack": "cracks",
            "part": "parts"}
    evs = [e for e in w.events.rows if e.verb in verb.values()]
    assert any(r["kind"] == "struck" for r in recs)
    assert [(verb[r["kind"]], r["tick"], r["cell"]) for r in recs] == \
        [(e.verb, e.tick, e.place) for e in evs]


# ── snapshots ───────────────────────────────────────────────────────────────

def test_a_SNAPSHOT_round_trip_keeps_what_was_perceived():
    w = World(40, 40, 30, voxel_cm=5)
    w.fill(0, 40, 0, 40, 0, 1, STONE)
    w.exits = []
    a, b = _person(w, 10, 20), _person(w, 22, 20)
    a["name"], b["name"] = "a", "b"
    b["facing"] = (-1.0, 0.0)
    w.step()
    w._say(a, "coming")
    w.step()
    assert w.minds[b["pid"]].perceived and w.minds[b["pid"]].bel
    snap = w.snapshot()
    rows = list(w.minds[b["pid"]].perceived)
    bel = dict(w.minds[b["pid"]].bel)
    for _ in range(3):
        w.step()
    w.restore(snap)
    assert list(w.minds[b["pid"]].perceived) == rows
    assert w.minds[b["pid"]].bel == bel
