"""Reference tests for sight and reaction (docs/realism-references.md, 4.9 and
gap G2): P-T1, P-T5, P-T7 and P-T10, plus the pieces they stand on.

Before step R3 a percept was answered on the tick it fired, by everybody at
once; a burning bench 5 m away, looked at, was never noticed (a fixed 4 m
radius); the field of view was 120 degrees and saw everything in it in full;
a runner was "seen" by reading the other body's `fleeing` flag; a thrown thing
was perceived without sight; and a seen fire was forgotten with a 4.6 s
half-life. Each test below builds a small world (10 cm voxels where the
scene allows) and measures one of these against the reference.
"""
import math

import numpy as np

from src.voxel.materials import IRON, STONE, TICK_S, WOOD
from src.voxel.policy import (PREMOVE_MIN_S, REACT, WILL, TablePolicy, rank,
                              sheet)
from src.voxel.scenes import _person, thing
from src.voxel.sim import World


class _Still(TablePolicy):
    """The reflex table, except that a body with nothing happening to it
    stands still: an idle walk would move the watcher off its mark."""

    def pick(self, sit, menu):
        if sit["percept"] is None and sit["limb"] == "legs":
            return 0
        return super().pick(sit, menu)


def _room(w):
    """A floor, and nothing else: open air on every side."""
    w.fill(0, w.shape[0], 0, w.shape[1], 0, 1, STONE)


# ── the pieces ───────────────────────────────────────────────────────────────

def test_the_FIELD_OF_VIEW_detects_wide_and_identifies_narrow():
    """Rows P1 and P2: something is DETECTED anywhere in about +/-100
    degrees, and IDENTIFIED only within about +/-30. The whole field was 120
    degrees, and everything in it was seen in full."""
    eye, face = (0.0, 0.0, 10.0), (1.0, 0.0)

    def at(deg):
        r = math.radians(deg)
        return (20.0 * math.cos(r), 20.0 * math.sin(r), 10.0)

    w = World(4, 4, 4, voxel_cm=10)              # 20 cells off is 2 m
    for deg in (0, 60, 95, -95):
        assert w._in_cone(face, eye, at(deg)), f"{deg} degrees is in view"
    for deg in (110, 150, 180, -120):
        assert not w._in_cone(face, eye, at(deg)), f"{deg} degrees is not"
    for deg in (0, 25, -25):
        assert w._in_focus(face, eye, at(deg)), f"{deg} degrees is told apart"
    for deg in (40, 90, -60, 150):
        assert not w._in_focus(face, eye, at(deg)), \
            f"{deg} degrees is seen, not told apart"
    near = (10.0 * math.cos(math.radians(80)), 10.0 * math.sin(math.radians(80)),
            10.0)
    assert w._in_focus(face, eye, near), "but at 1 m the edge tells it apart"
    assert 190.0 <= WILL["fov_deg"] <= 220.0 and WILL["focus_deg"] <= 70.0


def test_SIGHT_STARTS_IN_THE_HEAD_not_in_the_cell_in_front_of_it():
    """A head against a low ceiling does not see through it. Every cell next
    to the eye used to be skipped (the "eyelashes"), so a slab one cell
    above the crown was not there for a ray going up past it. Now only the
    looker's own flesh is skipped (`rays` with `skip`, `_sees` with `own`)."""
    w = World(40, 30, 40, voxel_cm=5)
    _room(w)
    p = _person(w, 15, 15)
    p["facing"] = (1.0, 0.0)
    w.step()
    eye = p["_eye"]
    ez = int(round(eye[2]))
    w.fill(0, 40, 0, 30, ez + 1, ez + 2, STONE)     # one cell over the crown
    target = (eye[0] + 3.0, eye[1], float(ez + 6))
    assert w._sees(eye, target), "the old rule: the slab is an eyelash"
    assert not w.rays([eye], [target], skip=[(p["pid"], -1)])[0], \
        "the ray starts in the head, and the slab stops it"
    own = set(p["_own"])
    assert not w._sees(eye, target, own=own), "and the same, one ray"
    level = (eye[0] + 8.0, eye[1], float(ez))
    assert w.rays([eye], [level], skip=[(p["pid"], -1)])[0] \
        and w._sees(eye, level, own=own), "its own head is never in the way"


def test_REACTION_TIMES_are_a_persons_own_and_inside_their_bands():
    """Rows P12-P15: each person has their own times, from the sheet or
    drawn from the band for that person, so two people do not react on the
    same tick. Drawn without a random number generator: the same every run."""
    people = [{"pid": i, "name": "the person"} for i in range(40)]
    for key, (lo, hi) in REACT.items():
        got = [sheet(p, key) for p in people]
        assert all(lo <= v <= hi for v in got), key
        assert len({round(v, 6) for v in got}) > 30, f"{key}: one each"
        assert got == [sheet(p, key) for p in people], f"{key}: the same again"
        assert lo + 0.2 * (hi - lo) <= float(np.median(got)) \
            <= hi - 0.2 * (hi - lo) or key == "premove_s"
    pre = sorted(sheet(p, "premove_s") for p in people)
    assert 7.0 <= pre[len(pre) // 2] <= 20.0, "pre-movement: a game median"
    assert sheet({"pid": 0, "name": "x", "ready_s": 0.2}, "ready_s") == 0.2, \
        "a sheet value wins"
    assert 0.15 <= REACT["ready_s"][0] and REACT["ready_s"][1] <= 0.35
    assert REACT["surprise_s"] == (1.0, 1.5) and REACT["flinch_s"][1] < 0.2
    assert REACT["premove_s"][0] >= 5.0 and PREMOVE_MIN_S >= 0.25
    assert rank("hears_noise") < rank("sees_fire") < rank("scorched")


def test_the_FORGET_FACTOR_is_derived_from_the_look_period():
    """Row P21: a 60 s half-life, from the look period and the tick, not a
    typed-in factor (0.985 a look was a 4.6 s half-life)."""
    looks = WILL["forget_half_s"] / (WILL["look_every"] * TICK_S)
    assert abs(WILL["forget"] ** looks - 0.5) < 1e-9
    assert abs(WILL["forget"] - 0.99885) < 1e-4


# ── P-T1: a visible fire across the room ─────────────────────────────────────

def _bench_fire(dist, bearing, name, ticks):
    """An 8 x 7 m room at 10 cm, a watcher at its west end facing east, a
    bench lit `dist` m off at `bearing` degrees. Returns the ticks of the
    first burning cell, of the first `sees_fire`, of the legs moving off
    for the door, and the watcher."""
    w = World(80, 70, 25, voxel_cm=10)
    _room(w)
    w.policy = _Still()
    px, py = 10, 15
    w.exits = [(1, py)]
    a = _person(w, px, py)
    a["name"], a["facing"] = name, (1.0, 0.0)
    r = math.radians(bearing)
    bx = int(round(px + dist * 10 * math.cos(r)))
    by = int(round(py + dist * 10 * math.sin(r)))
    w.fill(bx, bx + 4, by, by + 2, 1, 5, WOOD)
    lit = noticed = moved = None
    for t in range(ticks):
        if t == 0:
            w.E[bx, by, 1] = 9.0e5 * 8
        w.step()
        if lit is None and w.burning().any():
            lit = w.tick
        m = w.minds[a["pid"]]
        if noticed is None and m.noticed and m.noticed[1] == "sees_fire":
            noticed = m.noticed[0]
        for e in a["events"]:
            if moved is None and "makes for the door" in e:
                moved = w.tick
        a["events"].clear()
        if moved is not None:
            break
    pre = [r for r in w.traces if r["who"] == name
           and r["situation"].get("phase") == "premove"]
    return w, a, lit, noticed, moved, pre


def test_P_T1_a_FIRE_ACROSS_THE_ROOM_is_noticed_and_people_take_time_to_go():
    """P-T1: a bench lit 3 m straight ahead and 6 m off at 45 degrees -- the
    old rule never noticed a fire beyond 4 m. Noticed within 5 s of the first
    flame; the legs move off for the door inside the pre-movement band
    (5-30 s, `REACT["premove_s"]`; the reference accepts 5-60 s), never under
    0.25 s; and in between the body does a pre-movement act: it looks toward
    the fire (and warns, by its own row). The watcher is one whose drawn
    pre-movement is short, so the test is quick; its time is checked
    against its draw, and the band itself in the test above."""
    name = next(f"watcher {i}" for i in range(200)
                if sheet({"pid": 0, "name": f"watcher {i}"}, "premove_s") < 5.6)
    for dist, bearing in ((3.0, 0.0), (6.0, 45.0)):
        w, a, lit, noticed, moved, pre = _bench_fire(dist, bearing, name, 340)
        assert lit is not None and noticed is not None, (dist, bearing)
        assert (noticed - lit) * TICK_S <= 5.0, \
            f"noticed {(noticed - lit) * TICK_S:.2f} s after the first flame"
        assert moved is not None, "and leaves"
        off = (moved - noticed) * TICK_S
        lo, hi = REACT["premove_s"]
        assert lo <= off <= hi and off >= PREMOVE_MIN_S, f"moved off at {off:.2f} s"
        assert abs(off - sheet(a, "premove_s")) <= 2 * TICK_S, \
            "when this person's own pre-movement was over"
        assert pre and any(r["pick"].get("eyes", "").startswith("look toward the fire")
                           for r in pre), "it looked toward the fire first"
        assert all(r["pick"].get("legs") in (None, "stay") for r in pre), \
            "and its legs stayed while it made sense of it"
        del w


def test_a_PERSON_TAKES_UP_WHAT_THEY_HOLD_DEAR_before_leaving():
    """The pre-movement can be more than a look: a person whose sheet says
    what they hold dear (`dear`, a thing's eid) takes it up before going."""
    w = World(60, 40, 25, voxel_cm=10)
    _room(w)
    w.policy = _Still()
    w.exits = [(1, 20)]
    a = _person(w, 10, 20)
    a["name"], a["facing"] = "keeper", (1.0, 0.0)
    a["premove_s"] = 3.0                       # an explicit sheet value
    with thing(w, "the box") as e:
        w.fill(14, 15, 20, 21, 1, 3, WOOD)
    a["dear"] = e
    w.fill(30, 34, 20, 22, 1, 5, WOOD)          # the bench, 2 m ahead
    w.E[30, 20, 1] = 9.0e5 * 8
    took = None
    for _ in range(200):
        w.step()
        for r in w.traces:
            if took is None and r["who"] == "keeper" \
                    and r["situation"].get("phase") == "premove" \
                    and "take hold of" in r["pick"].get("hands", ""):
                took = r["pick"]["hands"]
        if took is not None:
            break
    assert took is not None and "box" in took, took


# ── P-T5: ready against surprised ────────────────────────────────────────────

def _mug(ready):
    """B stands; an iron mug is tossed at him from 1 m: from in front when B
    is ready (he has just swung an arm: a body in a fight), from his side
    when he is not. Returns the release tick and B's decisions and events."""
    from src.voxel.scenes import _Wants
    w = World(40, 30, 25, voxel_cm=10)
    _room(w)
    b = _person(w, 20, 15)
    b["name"], b["facing"], b["strength_N"] = "B", (-1.0, 0.0), 3000.0
    w.exits = []
    if ready:
        w.policy = _Wants(each={"B": {"hands": "swing", "legs": "stay"}})
        for _ in range(30):
            w.step()
        assert any("swings" in e for e in b["events"]), "B has struck out"
    w.policy = _Still()                         # the reflex table: catch it
    for _ in range(10):
        w.step()
    b["events"].clear()
    eye = b["_eye"]
    z = int(round(eye[2])) - 3
    if ready:                                   # from in front of him
        x0, y0, vel = int(eye[0]) - 10, int(eye[1]), (2.5, 0.0, 1.0)
    else:                                       # from his side
        x0, y0, vel = int(eye[0]), int(eye[1]) - 10, (0.0, 2.5, 1.0)
    w.fill(x0, x0 + 1, y0, y0 + 1, z, z + 1, IRON, frac=0.05)   # 0.4 kg
    cells = np.argwhere(w.mat == IRON)
    w._launch(cells, vel)
    t0 = w.tick
    ev, rows = [], []
    for _ in range(80):
        w.step()
        ev += [(w.tick, e) for e in b["events"]]
        b["events"].clear()
    rows = [r for r in w.traces if r["who"] == "B" and r["tick"] >= t0
            and r["percept"] == "sees_thrown"]
    return w, t0, ev, rows


def test_P_T5_a_READY_body_answers_fast_a_SURPRISED_one_flinches_first():
    """P-T5. Ready (already fighting): the first catching motion 0.15-0.35 s
    after the release. Surprised: a flinch -- the eyes going to it -- under
    0.2 s, and the organised answer only after 1-1.5 s, by when the mug has
    gone past: a fast surprise lands first."""
    w, t0, ev, rows = _mug(ready=True)
    grab = [t for t, e in ev if "catches" in e or "grabs at" in e
            or "cannot hold" in e]
    assert grab, ev
    assert 0.15 <= (grab[0] - t0) * TICK_S <= 0.35, (grab[0] - t0) * TICK_S
    del w
    w, t0, ev, rows = _mug(ready=False)
    flinch = [r["tick"] for r in rows if r["situation"].get("phase") == "flinch"]
    act = [r["tick"] for r in rows if r["situation"].get("phase") is None]
    assert flinch and (flinch[0] - t0) * TICK_S < 0.2, rows
    assert any("toward what is coming" in e for _, e in ev), ev
    assert act and 1.0 <= (act[0] - t0) * TICK_S <= 1.5 + 2 * TICK_S, rows
    assert not any("catches" in e for _, e in ev), "it was past him first"


# ── P-T7: a runner at the edge of sight ──────────────────────────────────────

def _runner(bearing):
    """O stands facing +y with his gaze held there. R runs 3 m off along the
    tangent at `bearing` degrees from O's facing. Returns the tick R comes
    into O's field of view, the tick of O's first `sees_runner`, and both."""
    w = World(90, 90, 25, voxel_cm=10)
    _room(w)
    w.policy = _Still()
    ox, oy = 45, 45
    o = _person(w, ox, oy)
    o["name"], o["facing"] = "O", (0.0, 1.0)
    r = math.radians(bearing)
    # the point at `bearing`, 3 m off (bearing measured from +y toward +x),
    # and the tangent through it
    cx, cy = ox + 30 * math.sin(r), oy + 30 * math.cos(r)
    tx, ty = math.cos(r), -math.sin(r)
    sx, sy = int(round(cx + 22 * tx)), int(round(cy + 22 * ty))   # from
    gx, gy = int(round(cx - 22 * tx)), int(round(cy - 22 * ty))   # behind
    q = _person(w, sx, sy)
    q["name"] = "R"
    q["facing"] = (-tx, -ty)
    w.exits = [(gx, gy)]
    w.step()
    q["emergency"], q["fleeing"], q["goal"] = True, True, (gx, gy)
    entered = seen = None
    for _ in range(90):
        o["look"] = (0.0, 1.0, w.tick)
        w.step()
        if q.get("_eye") is not None and entered is None \
                and w._in_cone(o["gaze"], o["_eye"], q["_eye"]):
            entered = w.tick - 1                # the tick just stepped
        m = w.minds[o["pid"]]
        if seen is None and m.noticed and m.noticed[1] == "sees_runner":
            seen = m.noticed[0]
        if q["safe"]:
            break
    return w, entered, seen, o, q


def test_P_T7_a_RUNNER_is_seen_at_90_degrees_and_not_at_150():
    """P-T7. A runner crossing 3 m off at 90 degrees is seen within 0.2-0.5
    s of coming into view; at 150 degrees, never. Seen by SPEED: the
    runner's `fleeing` flag is set in both runs, and it is the speed that
    is seen, from the edge of the view where nobody is told apart."""
    w, entered, seen, o, q = _runner(90.0)
    assert entered is not None and seen is not None, (entered, seen)
    assert 0.2 <= (seen - entered) * TICK_S <= 0.5, (seen - entered) * TICK_S
    del w
    w, entered, seen, o, q = _runner(150.0)
    assert q["fleeing"] and seen is None, "behind him: not seen"
    del w


# ── P-T10: remember the fire ─────────────────────────────────────────────────

def test_P_T10_a_SEEN_FIRE_is_REMEMBERED_for_about_a_minute():
    """P-T10. A body looks at a burning bench, then looks away for good. The
    danger it remembers at the fire's columns is at least half 60 s after
    the last sight (a 60 s half-life, the code's own stated intent): it was
    0.22 at 10 s and 1e-4 at 60 s. And it does fade: a quarter at 120 s.
    Looked at directly, one look every `look_every` ticks, without stepping
    the world through two minutes."""
    w = World(60, 30, 25, voxel_cm=10)
    _room(w)
    w.policy = _Still()
    w.exits = []
    p = _person(w, 10, 15)
    p["facing"] = (1.0, 0.0)
    w.fill(30, 34, 14, 16, 1, 5, WOOD)
    w.E[30, 14, 1] = 9.0e5 * 8
    for _ in range(12):
        w.step()
    assert w.burning().any(), "the bench burns"
    lit = w.burning().any(axis=2)
    comp, sl = w._person_cells(p)
    cells = np.argwhere(comp)
    cells[:, 0] += sl[0].start
    cells[:, 1] += sl[1].start
    eye = p["_eye"]
    p["gaze"] = (1.0, 0.0)
    w._look_around(p, eye, cells)
    assert float(p["danger"][lit].max()) == 1.0, "it saw the fire"
    p["gaze"] = (-1.0, 0.0)                     # and looks away, for good
    per_s = 1.0 / (WILL["look_every"] * TICK_S)
    got = {}
    for k in range(1, int(120 * per_s) + 1):
        w._look_around(p, eye, cells)
        if k in (int(10 * per_s), int(60 * per_s), int(120 * per_s)):
            got[round(k / per_s)] = float(p["danger"][lit].max())
    assert got[10] >= 0.85, got
    assert got[60] >= 0.5 - 1e-4, got           # 0.5 to float32 rounding
    assert 0.2 <= got[120] <= 0.3, got
