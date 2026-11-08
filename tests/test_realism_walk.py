"""WALKING AND RUNNING AT REAL SPEEDS: reference test B-T7 and the checks
around it (gap G3 and rows B3, B4 of docs/realism-references.md).

The pace was one voxel every 3 ticks whatever a voxel was: 0.67 m/s on a grid
axis at 5 cm (half a real walk), about 0.47 m/s on a diagonal, 1.33 m/s at
10 cm, and a body fleeing a fire went no faster than one strolling. Now the
pace is set in metres a second and turned into ticks with the voxel size; a
diagonal step is sqrt(2) of a voxel long and is paid for as that; and a body
fleeing runs.

Each scene is a flat floor with a door at the far end, and the walker is a
resident (it knows the whole floor, so the route is the planner's and not a
search). Speed is measured over the STEADY part of the walk: from when the
body is 1 m out to when it is 1 m short of the furthest it got, so the
reaction time, the first turn and the arrival do not count. The routes are
10 m (8 m on the diagonal), not the 20 m of B-T7: a steady speed does not
need the extra length, and the test stays quick.

Peak memory of this file: about 290 MB (measured, /usr/bin/time -v). About 55 s.
"""
import gc

import numpy as np

from src.voxel.sim import World, FLESH, STONE, TICK_S, WOOD
from src.voxel.policy import TablePolicy
from src.voxel.scenes import _person, _Wants


def _floor(cm, metres, diag):
    """A stone floor `metres` long for a walk, along x or corner to corner,
    with 0.6 m to spare at each end; a door at the far end. Returns the
    world, the start column and the door."""
    vox = 0.01 * cm
    pad = int(round(0.6 / vox))
    if diag:
        side = int(round(metres / np.sqrt(2.0) / vox)) + 2 * pad
        nx, ny = side, side
        start, door = (pad, pad), (nx - pad, ny - pad)
    else:
        nx, ny = int(round(metres / vox)) + 2 * pad, int(round(1.0 / vox))
        start, door = (pad, ny // 2), (nx - pad, ny // 2)
    # 40 voxels high: `_person` builds the same 30-voxel body at any voxel
    # size (1.5 m at 5 cm, 3 m at 10 cm), and it must fit in the room.
    w = World(nx, ny, 40, voxel_cm=cm)
    w.fill(0, nx, 0, ny, 0, 1, STONE)
    w.exits = [door]
    return w, start, door


def _speed(w, p, ticks, before=None, since=None):
    """Steps the world and returns the walker's steady speed in m/s (see the
    module docstring), measured on the centre of its flesh, from the first
    tick `since(p)` is true (from the start, without one)."""
    vox = 0.1 * w.scale
    track = []
    for t in range(ticks):
        if before is not None:
            before(t)
        w.step()
        p["events"].clear()
        own = np.argwhere(w.mat == FLESH)
        if not len(own) or p["safe"]:
            break
        if since is None or track or since(p):
            track.append((w.tick, float(own[:, 0].mean()),
                          float(own[:, 1].mean())))
    assert len(track) > 1, "the walker never set off"
    xs = np.array(track)
    d = np.hypot(xs[:, 1] - xs[0, 1], xs[:, 2] - xs[0, 2]) * vox
    assert d.max() > 4.0, f"the walker hardly moved ({d.max():.2f} m)"
    a = int(np.nonzero(d > 1.0)[0][0])
    b = int(np.nonzero(d > d.max() - 1.0)[0][0])
    return float((d[b] - d[a]) / ((xs[b, 0] - xs[a, 0]) * TICK_S))


def _walk(cm, diag=False, flee=False, sneak=False):
    """The walker goes to the door, picking "go(the door ...)" off its own
    menu. `flee` gives it a linked mind's intent to flee (`intent["flee"]`),
    `sneak` makes it creep."""
    metres = 8.0 if diag else 10.0
    w, start, _door = _floor(cm, metres, diag)
    p = _person(w, *start)
    p["name"] = "walker"
    p["known"][:] = True
    if flee:
        p["intent"] = {"flee": True}
    p["sneak"] = sneak
    w.policy = _Wants(each={"walker": {"legs": "go(the door"}})
    v = _speed(w, p, int(40 * metres / 0.3))
    del w, p
    gc.collect()
    return v


def test_B_T7_a_WALK_is_a_real_walk_and_the_same_at_5_and_10_cm():
    """B-T7, the walk on a grid axis. Pass: 1.2-1.5 m/s at both voxel sizes
    (the B-T7 band; the source band is 1.10-1.65 m/s for a comfortable walk:
    Bohannon & Andrews 2011, refs 172-173), and the two sizes within 5% of
    each other. Before: 0.67 m/s at 5 cm, 1.33 m/s at 10 cm."""
    v5, v10 = _walk(5), _walk(10)
    for cm, v in ((5, v5), (10, v10)):
        assert 1.2 <= v <= 1.5, f"a walk at {cm} cm is {v:.2f} m/s"
    assert abs(v5 - v10) <= 0.05 * max(v5, v10), \
        f"the voxel size changes the walk: {v5:.2f} m/s at 5 cm, {v10:.2f} at 10 cm"


def test_B_T7_a_DIAGONAL_walk_is_as_fast_as_one_along_the_grid():
    """B-T7, the walk corner to corner at 45 degrees. A diagonal step is
    sqrt(2) of a voxel and is paid for as that, so the speed over the ground
    is the walk's. Pass: 1.2-1.5 m/s at 5 and 10 cm (refs 172-173), and
    within 5% of the same walk along a grid axis. Before: about 0.47 m/s at
    5 cm, because a 45 degree route was a staircase of grid steps."""
    for cm in (5, 10):
        axis, diag = _walk(cm), _walk(cm, diag=True)
        assert 1.2 <= diag <= 1.5, f"a diagonal walk at {cm} cm is {diag:.2f} m/s"
        assert abs(diag - axis) <= 0.05 * axis, \
            f"at {cm} cm the diagonal is {diag:.2f} m/s and the axis {axis:.2f}"


class _StandUnlessStartled(TablePolicy):
    """The reflex table, except that a body with nothing happening to it
    stands where it is: a stroll before the fire is seen would be measured
    as part of the flight."""
    def pick(self, sit, menu):
        if sit["percept"] is None and sit["limb"] == "legs":
            return 0
        return super().pick(sit, menu)


def test_B_T7_FLEEING_A_FIRE_is_a_RUN_at_5_and_10_cm():
    """B-T7, the flight. A man faces a lit crib 3.5 m in front of him (at
    10 cm the same 30-voxel body is 3 m tall, and from nearer its own
    shoulder hides a fire on the floor), with the door 6.8 m behind; he sees the flame and the reflex table sends him for the
    door. The speed is measured from the tick he starts to flee. Pass:
    2.1-7 m/s at both voxel sizes, so a running gait, not a fast walk:
    people change from a walk to a run at 2.06 m/s (Hreljac 1993, ref
    175); the top of the band is far under the elite top speed of 12.4 m/s
    (ref 174), as ordinary running speeds have no checked source. And the
    same at both sizes, within 5%. Before: no faster than a walk."""
    out = {}
    for cm in (5, 10):
        vox = 0.01 * cm
        w, start, door = _floor(cm, 10.0, False)
        n = lambda m: max(1, int(round(m / vox)))
        cx, cy = door[0] + n(0.3), door[1]          # 3.5 m ahead of him
        w.exits = [start]                           # and the door behind him
        w.fill(cx, cx + n(0.3), cy - n(0.15), cy + n(0.15), 1, 1 + n(0.1),
               WOOD, frac=0.8)
        p = _person(w, door[0] - n(3.2), door[1])
        p["name"], p["facing"] = "walker", (1.0, 0.0)
        p["known"][:] = True
        p["premove_s"] = 0.25                       # no pre-movement (R3)
        w.policy = _StandUnlessStartled()

        # A FLAME HELD TO THE CRIB until he has seen it burning: heat in
        # proportion to the voxel's volume, held on, because a crib lit for
        # a fixed time catches at one size and goes out at the other.
        def light(t, w=w, p=p, at=(cx + 1, cy, 1)):
            if not p["fleeing"]:
                w.E[at] += 2500.0 * w.vox_l / 0.125

        out[cm] = _speed(w, p, 400, before=light,
                         since=lambda p: p["fleeing"])
        assert p["safe"], "he ran from the fire he saw, out of the door"
        del w, p
        gc.collect()
    for cm, v in out.items():
        assert 2.1 <= v <= 7.0, f"fleeing a fire at {cm} cm is {v:.2f} m/s"
    assert abs(out[5] - out[10]) <= 0.05 * max(out.values()), out


def test_a_LINKED_MIND_that_means_to_flee_runs_too():
    """A run is for anything a body is fleeing, not for fire alone: a linked
    mind that says it is running from something (`intent["flee"]`) gets the
    same gait. Pass: 2.1-7 m/s (refs 174, 175, as above), at 5 cm."""
    v = _walk(5, flee=True)
    assert 2.1 <= v <= 7.0, f"a body meaning to flee goes {v:.2f} m/s"


def test_CREEPING_is_slower_than_WALKING():
    """A creep is a walk slowed on purpose to be quiet (`SOUND["sneak_slow"]`,
    half a walk; the footstep tests in tests/test_sound.py hang on it). No
    source gives a speed for walking quietly, so the pass range is the
    model's: 0.4-0.6 of the walk at 5 cm, and slower than any comfortable
    walk (1.10 m/s, refs 172-173)."""
    walk, creep = _walk(5), _walk(5, sneak=True)
    assert 0.4 * walk <= creep <= 0.6 * walk, (walk, creep)
    assert creep < 1.10, f"a creep at {creep:.2f} m/s is a walk"
