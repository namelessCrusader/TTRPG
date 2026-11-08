"""New in V0 (docs/sim-mind-design.md, 4.4): the event core. V1 adds the
tests below the line, for the sites `will.py`, `senses.py` and
`physiology.py` convert.

`Ev`, `EventLog`, `Mind`, `emit`, `because` and `why` -- nothing in the sim
calls any of this yet (that starts at V1), so these tests build the pieces
directly and check them as a small library, the same way `test_snapshot.py`
tests the ghost fix on top of `src.voxel.demo.build()`.
"""
import copy
import json

import numpy as np
import pytest

from src.voxel.demo import build
from src.voxel.events import EventLog, Mind
from src.voxel.scenes import _person
from src.voxel.sim import STONE, World


def test_the_ring_TRIMS_and_ids_and_get_still_work():
    log = EventLog(cap=8)
    for t in range(20):
        log.append(_ev(log, t))
    assert len(log.rows) <= 8
    # every id ever handed out is either still in the ring, or gone for good
    for ev_id in range(log.next_id):
        row = log.get(ev_id)
        assert row is None or row.id == ev_id
    last = log.rows[-1]
    assert log.get(last.id) is last
    assert log.get(log.next_id) is None            # not written yet


def test_since_SKIPS_everything_up_to_and_including_the_given_id():
    log = EventLog(cap=64)
    for t in range(10):
        log.append(_ev(log, t))
    mid = log.rows[4].id
    rest = log.since(mid)
    assert [e.id for e in rest] == [e.id for e in log.rows[5:]]
    assert log.since(-1) == log.rows                # before the ring: everything


def test_deepcopy_of_an_EV_SHARES_the_record():
    log = EventLog()
    ev = log.append(_ev(log, 0))
    assert copy.deepcopy(ev) is ev, \
        "Ev.__deepcopy__ must return self, so a snapshot copies the ring for free"


def test_deepcopy_of_the_WHOLE_LOG_shares_every_row():
    log = EventLog()
    for t in range(5):
        log.append(_ev(log, t))
    cp = copy.deepcopy(log)
    assert cp.rows == log.rows
    assert all(a is b for a, b in zip(cp.rows, log.rows))


def test_snapshot_restore_KEEPS_THE_LOG_and_bumps_the_epoch():
    w = build()
    w.add_person(2, 2)
    w.emit("steps", actor=0, text="x")
    before_epoch = w._epoch
    snap = w.snapshot()
    w.step()
    w.emit("steps", actor=0, text="y")
    w.restore(snap)
    assert [e.verb for e in w.events.rows] == ["steps"], \
        "restore must put the event log back, not keep what ran after the snapshot"
    assert w._epoch == before_epoch + 1, \
        "_epoch is excluded from state, so restore must bump it, not restore it"
    w.restore(snap)
    assert w._epoch == before_epoch + 2, "every restore bumps it again"


def test_restore_also_keeps_ONE_MIND_PER_PERSON():
    w = build()
    w.add_person(2, 2)
    w.add_person(3, 3)
    w.minds[0].bel["x"] = (1, 0, "saw", 0)
    snap = w.snapshot()
    w.minds[0].bel["x"] = (2, 0, "saw", 0)
    w.restore(snap)
    assert w.minds[0].bel["x"] == (1, 0, "saw", 0)
    assert len(w.minds) == 2


def test_add_person_APPENDS_A_MIND_and_stamps_pid():
    w = build()
    p0 = w.add_person(2, 2)
    p1 = w.add_person(3, 3)
    assert p0["pid"] == 0 and p1["pid"] == 1
    assert len(w.minds) == 2
    assert all(isinstance(m, Mind) for m in w.minds)


def test_emit_APPENDS_LEGACY_TEXT_ONLY_WHEN_GIVEN_to():
    w = build()
    p = w.add_person(2, 2)
    n0 = len(p["events"])
    w.emit("steps", actor=0, text="steps east")           # no `to`
    assert len(p["events"]) == n0
    w.emit("steps", actor=0, text="steps east", to=p)
    assert p["events"][-1] == f"t{w.tick}: steps east"


def test_emit_DEFAULTS_CAUSE_from_the_current_because():
    w = build()
    root = w.emit("decides", actor=0, text="")
    with w.because(root.id):
        child = w.emit("steps", actor=0, text="")
        assert child.cause == root.id
    after = w.emit("steps", actor=0, text="")
    assert after.cause is None, "the cause must not leak past the block"


def test_emit_LETS_A_CALLER_OVERRIDE_the_current_cause():
    w = build()
    root = w.emit("decides", actor=0, text="")
    other = w.emit("decides", actor=0, text="")
    with w.because(root.id):
        child = w.emit("steps", actor=0, text="", cause=other.id)
    assert child.cause == other.id


def test_because_RESTORES_after_an_exception():
    w = build()
    root = w.emit("decides", actor=0, text="")
    assert w._cause is None
    with pytest.raises(ValueError):
        with w.because(root.id):
            assert w._cause == root.id
            raise ValueError("boom")
    assert w._cause is None, "because() must restore the cause even on an exception"


def test_tick_start_RESETS_the_cause():
    w = build()
    root = w.emit("decides", actor=0, text="")
    w._cause = root.id
    w._tick_start()
    assert w._cause is None


def test_why_WALKS_THE_CHAIN_oldest_first():
    w = build()
    a = w.emit("decides", actor=0, text="")
    with w.because(a.id):
        b = w.emit("swings", actor=0, text="")
    with w.because(b.id):
        c = w.emit("hits", actor=0, text="")
    chain = w.why(c.id)
    assert [e.id for e in chain] == [a.id, b.id, c.id]


def test_why_STOPS_at_a_cause_the_ring_no_longer_holds():
    w = build()
    a = w.emit("decides", actor=0, text="")
    with w.because(a.id):
        b = w.emit("swings", actor=0, text="")
    w.events.first = b.id                 # pretend `a` fell off the ring
    w.events.rows = w.events.rows[1:]
    chain = w.why(b.id)
    assert [e.id for e in chain] == [b.id]


def test_to_json_IS_PLAIN_JSON(tmp_path):
    w = build()
    ev = w.emit("hits", actor=0, target=("person", 1), instrument=None,
                phase="done", result=None, why=None, place=(1, 2, 3),
                db=40.0, dur=0.1, text="hits")
    s = json.dumps(ev.to_json())
    back = json.loads(s)
    assert back["verb"] == "hits"
    assert back["target"] == ["person", 1]   # json has no tuples; round-trips as a list
    assert "text" not in back, "Ev.text is the legacy line; to_json must leave it out"


# ── V1: the sites converted in will.py, senses.py and physiology.py ─────────
# Golden and the two named `test_voxel.py` tests exercise most of these, but
# four sites never come up in any golden case or in the FAST tier: `let_go`
# and `pull_in` (the hand is only ever asked to hold or reach in the cases
# that run), `stand` (nothing leans and then stands up straight), and `gone`
# (a body whose flesh has entirely left the lattice without ever becoming a
# flying body -- the ordinary way to vanish is to burn to ash over hundreds
# of ticks, which no quick case runs). Pinned here, directly, the same way
# the rest of this file builds pieces directly rather than through a scene.

class _Wants:
    """The same fixed-choice policy `test_voxel.py` tests bodies with: one
    named tag per limb, the null act otherwise."""
    name = "wants"

    def __init__(self, **want):
        self.want = want

    def pick(self, situation, menu):
        tag = self.want.get(situation["limb"])
        for i, opt in enumerate(menu):
            if opt["tag"] == tag:
                return i
        return 0


def _standing_person(x=10, y=10):
    w = World(40, 40, 30, voxel_cm=5)
    w.fill(0, 40, 0, 40, 0, 1, STONE)
    w.exits = []
    p = _person(w, x, y)
    comp, sl = w._person_cells(p)
    cells = np.argwhere(comp)
    cells[:, 0] += sl[0].start
    cells[:, 1] += sl[1].start
    ax_ = float(cells[:, 0].mean())
    ay_ = float(cells[:, 1].mean())
    eye = (ax_, ay_, float(cells[:, 2].max()))
    return w, p, cells, ax_, ay_, eye


def test_LETTING_GO_emits_releases():
    w, p, cells, ax_, ay_, eye = _standing_person()
    # `_held_cells` re-floods from the gripped cell and checks it against the
    # material held: a mismatch reads as "gone" and clears `held` on sight
    # (hands.py), so the fake grip has to point at real stone to survive the
    # legs menu's own look at what this body is carrying.
    p["held"] = {"cell": (0, 0, 0), "mat": int(STONE), "label": "stick", "arm": None}
    w.policy = _Wants(hands="let_go")
    w._decide(p, None, cells, ax_, ay_, eye, 0.1 * w.scale)
    assert p["held"] is None and p.get("dragging") is None
    assert p["events"][-1].endswith("lets go")
    assert w.events.rows[-1].verb == "releases"


def test_PULLING_THE_ARM_BACK_IN_emits_pulls_in():
    w, p, cells, ax_, ay_, eye = _standing_person()
    arm = next(a for a in w._limbs(p) if "arm" in a)
    p["drawn"] = {arm: np.array([1.2])}       # an arm already out
    w.policy = _Wants(hands="pull_in")
    w._decide(p, None, cells, ax_, ay_, eye, 0.1 * w.scale)
    assert p["events"][-1].endswith("lowers the arm")
    assert w.events.rows[-1].verb == "pulls_in"


def test_STANDING_UP_STRAIGHT_emits_stands():
    w, p, cells, ax_, ay_, eye = _standing_person()
    p["drawn"] = {"lean": np.array([0.5])}    # bent, so there is something to undo
    w.policy = _Wants(waist="upright")
    w._decide(p, None, cells, ax_, ay_, eye, 0.1 * w.scale)
    assert p["events"][-1].endswith("straightens up")
    assert w.events.rows[-1].verb == "stands"


def test_A_BODY_WITH_NO_FLESH_LEFT_and_NOTHING_IN_THE_AIR_emits_gone():
    w, p, cells, ax_, ay_, eye = _standing_person()
    w.step()                                  # fills w.o2, which _law_life needs
    # the flesh is gone, and nothing flew off to account for it: the same
    # claim `_left_world` leaves behind once a body truly has nowhere left
    # on the lattice (3.3's census), without re-running the flood that finds
    # it (persons.py's own business, not this step's).
    p["_claim_mask"], p["_claim_sl"] = None, None
    p["_claim_tick"] = w.tick
    w._law_life()
    assert p["events"][-1].endswith("is gone")
    assert w.events.rows[-1].verb == "gone"


def test_A_FALLING_BODY_THAT_ROLLS_emits_rolls():
    w, p, cells, ax_, ay_, eye = _standing_person()
    w.bodies.append({"owner": p["name"]})     # in the air, as far as this law can tell
    w.policy = _Wants(legs="roll")
    w._decide_falling(p)
    assert p["events"][-1].endswith("tucks to roll")
    assert w.events.rows[-1].verb == "rolls"


def test_A_LOUD_SOUND_WAKES_A_SLEEPER():
    w, p, cells, ax_, ay_, eye = _standing_person()
    p["asleep"] = True
    w.sounds.append({"tick": w.tick, "at": (ax_, ay_, eye[2]),
                      "db": 90.0, "who": None})
    w._law_hear()
    assert not p["asleep"]
    assert p["events"][-1].endswith("dB)")
    assert w.events.rows[-1].verb == "wakes"


def test_DEATH_BY_SUFFOCATION_emits_dies():
    w, p, cells, ax_, ay_, eye = _standing_person()
    w.step()                                  # fills w.o2
    p["blood_o2"] = 0.0                        # below BODY["death_o2"]
    w._law_life()
    assert not p["alive"]
    assert p["events"][-1].endswith("stops breathing")
    assert w.events.rows[-1].verb == "dies"


def _ev(log, t):
    from src.voxel.events import Ev
    return Ev(id=log.next_id, tick=t, verb="steps", actor=None, target=None,
              instrument=None, phase=None, result=None, why=None, place=None,
              origin=None, db=None, visible=True, dur=None, cause=None,
              link=None, text="")


# ── V2: events in the physical modules (bodies, hands, motion, support,
# heat, fluids, pose, physiology) ────────────────────────────────────────────
# `hits`, `wounds`, `lands` and `topples` are new events, not conversions of
# legacy text -- and every one of them is raised from inside a PHYSICAL law,
# which runs before `_law_will` in a tick (2.2). Emitting them the moment
# they happen would consume an id earlier in the tick than it used to be
# consumed, and a shout later in the SAME tick (`_say`, senses.py) stamps the
# id of its own `says` event onto the shout's sound record -- so that number
# would shift for no reason of its own. `_defer` (events.py) holds every one
# of these back to `_tick_end`, after the tick's shouts have already taken
# whatever id they were going to take. Measured against this: harvest seed 7
# DIFFERED at `sounds`, not just `events`, before `_defer` existed -- the
# clearest sign the ids were leaking where they should not.

def test_A_THROWN_IRON_NAMES_ITS_THROWER_IN_hits():
    from tests.test_voxel import _pitch
    w, b, ev = _pitch("hands free")           # he does not try to catch it
    thrower = next(r for r in w.persons if r["name"] == "thrower")
    hits = [e for e in w.events.rows if e.verb == "hits"
            and e.target == ("person", b["pid"])]
    assert hits, f"the iron must strike him somewhere ({ev[:6]})"
    assert hits[0].actor == thrower["pid"], \
        "a thrown iron's hits names its thrower, not nobody"


def test_A_WHOLE_BODY_FALL_hits_WITH_NO_ACTOR_the_FALLER_AS_INSTRUMENT_and_FALLS_AS_CAUSE():
    w, p, cells, ax_, ay_, eye = _standing_person(x=10, y=10)
    q = _person(w, 25, 10)
    comp, sl = w._person_cells(q)
    c0 = np.argwhere(comp)[0]
    qcell = (int(c0[0] + (sl[0].start or 0)), int(c0[1] + (sl[1].start or 0)),
             int(c0[2] + (sl[2].start or 0)))
    # the body itself never decided to swing anything -- it just fell, the
    # same way `_law_footing` launches one: a `falls` event made first, its
    # id carried on the body for whatever it eventually lands on.
    falls_ev = w.emit("falls", actor=p["pid"], why="loses their footing",
                      text=f"{p['name']} loses their footing")
    b = {"owner": p["name"], "part": False, "fly": True,
         "mats": np.array([STONE], np.int16), "edges": np.zeros(1, np.float32),
         "falls_ev": falls_ev.id}
    w._spend(b, [qcell], 500.0)
    w._tick_end()                              # flush the deferred `hits`
    hits = [e for e in w.events.rows if e.verb == "hits"
            and e.target == ("person", q["pid"])]
    assert hits, "a whole-body fall landing on someone must still be a hit"
    assert hits[0].actor is None, \
        "nobody swung this body at him; it only fell"
    assert hits[0].instrument == ("person", p["pid"]), \
        "the faller is the instrument, not the actor"
    assert hits[0].cause == falls_ev.id, \
        "the hit is caused by the fall, which happened first"


def test_A_PANE_DROP_GIVES_EXACTLY_ONE_breaks():
    from src.voxel.sim import GLASS
    w = World(12, 12, 14)
    w.fill(0, 12, 0, 12, 0, 1, STONE)
    w.fill(5, 6, 5, 6, 9, 10, GLASS)            # one glass cell, high in the air
    for _ in range(60):
        w.step()
    breaks = [e for e in w.events.rows if e.verb == "breaks"]
    assert len(breaks) == 1, f"one pane dropped once must break once ({breaks})"


def test_EVENTS_ARE_THE_SAME_with_skip_quiet_true_and_false():
    from src.voxel.sim import GLASS

    def build():
        w = World(12, 12, 14)
        w.fill(0, 12, 0, 12, 0, 1, STONE)
        w.fill(5, 6, 5, 6, 9, 10, GLASS)
        return w
    slow, fast = build(), build()
    fast.restore(slow.snapshot())               # identical starting state
    slow.skip_quiet, fast.skip_quiet = False, True
    for _ in range(60):
        slow.step()
        fast.step()
    a = [(e.verb, e.actor, e.target, e.cause) for e in slow.events.rows]
    b = [(e.verb, e.actor, e.target, e.cause) for e in fast.events.rows]
    assert a == b, "skip_quiet must not change which events happen, or to whom"


def test_A_WEAK_HOLDER_GOES_OVER_WITH_THE_MAN_HE_CANNOT_HOLD():
    """hands.py's `_haul` ("goes over, and the grip is not enough to hold a
    hanging man"): a holder who closes a hand on a falling man can still be
    too weak to go ON holding him once he is hanging clear of the ground."""
    from src.voxel.scenes import _person as _scene_person, _Wants
    w = World(50, 16, 70, voxel_cm=5)
    w.open_sky = True
    w.fill(0, 50, 0, 16, 0, 1, STONE)
    w.fill(0, 26, 0, 16, 1, 40, STONE)           # ledge top z39, drop at x26
    w.exits = []
    a = _scene_person(w, 22, 8, z0=40)
    a["name"], a["facing"] = "the catcher", (1.0, 0.0)
    a["strength_N"] = 4000.0                     # strong enough to CATCH him
    a["ready"] = True                            # and there to: ready (R3)
    b = _scene_person(w, 27, 8, z0=40)
    b["name"], b["facing"] = "the faller", (-1.0, 0.0)
    w.policy = _Wants(each={
        "the catcher": {"hands": "catch", "legs": "stay", "waist": "stand"},
        "the faller": {"hands": "hands free", "legs": "stay"}})
    for _ in range(60):
        w.step()
        a["events"].clear()
        b["events"].clear()
        if a.get("dragging") == "the faller":
            break
    assert a.get("dragging") == "the faller", "he must close a hand on him first"
    a["strength_N"] = 1.0                        # ...but too weak to GO ON
    went_over = False                            # holding him, once he hangs
    for _ in range(40):
        w._law_grips()
        w._haul(a, 0, 0)
        a["events"].clear()
        if a.get("dragging") is None:
            went_over = True
            break
    assert went_over, "too weak to hold him, he must let go"
    assert w.events.rows[-1].verb == "releases", \
        f"and it must be the 'goes over' release, not some other one " \
        f"({w.events.rows[-1]})"


def test_TICK_END_leaves_NO_BOX_in_a_minds_ATTENTION_or_NEWS():
    """The onset attention law (`_attend`, will.py) keys what a mind
    attended to by source, and a source raised this same tick is still a
    `_Pending` box. `_unbox` puts the real id in at the tick's end, the way
    it does for `noticed`: the key and `news` become the id, a key the id
    already had keeps the newer tick, and a box that never got an id (a
    sound that matched nothing) is dropped. Nothing a tick leaves behind
    holds a box, so a snapshot between ticks copies plain ints."""
    from src.voxel.events import _Pending
    w = World(20, 20, 20, voxel_cm=5)
    w.fill(0, 20, 0, 20, 0, 1, STONE)
    p = _person(w, 10, 10)
    m = w.minds[p["pid"]]
    w._tick_start()
    box = w._defer("ignites", place=(5, 5, 1), text="")
    lost = _Pending()
    t = w.tick
    m.attn = {"chokes": t - 3, w.events.next_id: t - 2, box: t, lost: t}
    m.news = box
    w._tick_end()
    assert box.id is not None
    assert m.attn == {"chokes": t - 3, box.id: t}, m.attn
    assert m.news == box.id
    assert not any(isinstance(k, _Pending) for k in m.attn)
    snap = w.snapshot()
    w.restore(snap)
    assert w.minds[p["pid"]].attn == {"chokes": t - 3, box.id: t}
