"""Realism tests for breathing: the clock a body runs on without air (realism
gap G9 in docs/realism-references.md; tests B-T1, B-T2 and B-T8 of 4.7 and
P-T4 of 4.9). Every band in a docstring is the document's, with its source.

THE CLOCK IS MINUTES LONG, and a world stepped for minutes is too slow for a
test. So the body tests FAST-FORWARD a still world (`_breathe_only`): while
nothing is moving, only the body's chemistry runs a tick (`_law_life`), with
the tick counted as `step` counts it; the moment anything is in the air (a
body keeling over), the whole world steps until it lands. A still pool, a
sealed room and a set smoke field do not change while nobody moves, so this
is the same clock, not a shortcut through it. The bodies are columns of
flesh at 10 cm (`_person` is a 5 cm body), standing still, so they work at
one MET.

Each test is under ~1 GB and ~20 s.
"""
import numpy as np

from src.voxel.materials import (AIR, BODY, FLESH, O2_PER_L, SMOKE_EXT, STONE,
                                 TICK_S, WATER)
from src.voxel.physiology import BREATH
from src.voxel.scenes import _Wants
from src.voxel.sim import World

S = 1.0 / TICK_S                 # ticks in a second
MIN = 60.0 * S                   # ticks in a minute


def _column(w, x, y, tall_m, name):
    """A body: a 2 x 1 column of flesh `tall_m` high, standing on z = 1."""
    v = 0.1 * w.scale
    w.fill(x, x + 2, y, y + 1, 1, 1 + int(round(tall_m / v)), FLESH, frac=0.9)
    return w.add_person(x, y, name=name)


def _boxes(nx, ny, nz, rooms, voxel_cm=10):
    """A sealed stone block hollowed into `rooms` side by side along x, each
    `nx` voxels wide inside, with one stone wall between neighbours."""
    w = World(rooms * (nx + 1) + 1, ny + 2, nz + 2, voxel_cm=voxel_cm)
    w.open_sky = False
    w.fill(0, w.shape[0], 0, w.shape[1], 0, w.shape[2], STONE)
    for r in range(rooms):
        x0 = 1 + r * (nx + 1)
        w.mat[x0:x0 + nx, 1:ny + 1, 1:nz + 1] = AIR
        w.smass[x0:x0 + nx, 1:ny + 1, 1:nz + 1] = 0.0
    w.exits = []
    w.policy = _Wants(legs="stay", hands="hands free")
    return w


def _room(w, nx, r):
    """The x slice of room `r` of a `_boxes` world."""
    x0 = 1 + r * (nx + 1)
    return slice(x0, x0 + nx)


def _breathe_only(w, ticks, each=None):
    """Run `ticks` ticks, fast-forwarding while nothing moves (see the module
    docstring). `each(w)` runs before every tick: a scene's own change."""
    for _ in range(ticks):
        if each is not None:
            each(w)
        if w.bodies:
            w.step()
        else:
            w._tick_start()
            w._law_life()
            w._tick_end()
            w.tick += 1


class _Clock:
    """When each body went down, came round and died (ticks from `t0`)."""

    def __init__(self, w, people):
        self.w, self.people, self.t0 = w, people, w.tick
        self.faint, self.wake, self.dead = {}, {}, {}
        self.was = {p["name"]: p["awake"] for p in people}

    def __call__(self, w):
        t = w.tick - self.t0
        for p in self.people:
            n = p["name"]
            if self.was[n] and not p["awake"]:
                self.faint.setdefault(n, t)
            if not self.was[n] and p["awake"]:
                self.wake.setdefault(n, t)
            if not p["alive"]:
                self.dead.setdefault(n, t)
            self.was[n] = p["awake"]
            p["events"].clear()


def test_B_T1_DROWNING_CLOCK_a_held_breath_lasts_minutes():
    """B-T1 (source 194, row B22): a person with the head 20 cm under water
    is unconscious at 60-180 s (accept 30-240 s) and dead at 4-10 min
    (accept 3-15 min); a body pulled out before ~4 min survives and comes
    round; a face pressed into a solid follows the same clock. Before G9:
    nobody drowned (a water cell held fresh oxygen) and a buried face died
    in 1.7 s.

    Reduced: four 1.6 m columns in four pools at 10 cm, standing still (one
    MET, no struggle). The doc's pull-out at 90 s comes BEFORE this body
    faints (it faints at ~2.5 min at rest), so the pulled copy is pulled out
    at 3.5 min instead: after the faint and before the 4 min line, which is
    what "pulled out before ~4 min survives" asks. Pulled out is its pool
    drained. The doc's second copy, "unconscious face-down in a 10 cm
    puddle", is a body knocked out (a head blow: `ko_until`) lying under
    the water: the sim has no face, so a lying body breathes through its
    whole top and a 10 cm puddle would not cover it."""
    nx = 4
    w = _boxes(nx, 6, 20, rooms=4)
    held = _column(w, 2, 3, 1.6, "held")              # pool 0
    pulled = _column(w, 7, 3, 1.6, "pulled")          # pool 1
    out = _column(w, 12, 3, 1.6, "knocked out")       # pool 2
    buried = _column(w, 17, 3, 1.6, "buried")         # room 3: no water
    for r in range(3):                                # water to 1.8 m: 20 cm
        xs = _room(w, nx, r)                          # over a 1.6 m head
        wet = np.zeros(w.shape, bool)
        wet[xs, 1:7, 1:19] = True
        wet &= w.mat == AIR
        w.fl[wet], w.fvol[wet] = WATER, w.cap
    w.fill(16, 20, 2, 5, 14, 18, STONE)               # the face in a solid
    w.fill(17, 19, 3, 4, 1, 17, FLESH, frac=0.9)      # (refilled: one body)
    out["ko_until"] = 10 ** 9   # a head blow keeps him out (C7: a bruise no longer would)
    out["awake"] = False
    for _ in range(40):                               # the water settles and
        w.step()                                      # the knocked-out man
    for _ in range(400):                              # lies down
        if not w.bodies:
            break
        w.step()
    assert not w.bodies
    people = (held, pulled, out, buried)
    clock = _Clock(w, people)
    t_pull = int(3.5 * MIN)

    def scene(w):
        if w.tick - clock.t0 == t_pull:               # pulled out: drained
            xs = _room(w, nx, 1)
            w.fvol[xs] = 0.0
            w.fl[xs] = 0
        clock(w)
    _breathe_only(w, int(9 * MIN), scene)
    clock(w)
    f, d = clock.faint, clock.dead
    assert 60 * S <= f["held"] <= 180 * S, f"unconscious at {f['held'] / S:.0f} s"
    assert 4 * MIN <= d["held"] <= 10 * MIN, f"dead at {d['held'] / MIN:.1f} min"
    assert abs(f["buried"] - f["held"]) <= 2 * S, \
        "a face in a solid runs the same clock as a face in water"
    assert abs(d["buried"] - d["held"]) <= 5 * S
    assert "pulled" in f and f["pulled"] < t_pull, "out before the pull"
    assert pulled["alive"], "pulled out before ~4 min, he lives"
    assert "pulled" in clock.wake and clock.wake["pulled"] - t_pull <= 2 * MIN, \
        "and comes round in the air"
    assert 4 * MIN <= d["knocked out"] <= 10 * MIN, \
        f"out from the start, dead at {d['knocked out'] / MIN:.1f} min"


def test_B_T2_a_ROOM_WITH_LITTLE_OXYGEN():
    """B-T2 (source 193, row B21): in air of 14% O2 nobody faints within
    10 min; at 10% they do NOT faint within 10 min (wave 14bF, declared: the
    row said 1-10 min, from a coarse hazard table; the haemoglobin curve of
    wave 14b lane H, Severinghaus 1979, and the FAA time of useful
    consciousness at 18000 ft, which is 10.4% sea-level-equivalent oxygen,
    20-30 min, agree that 10% is not a faint within 10 min); at 5% coma comes
    in 10-60 s
    and death not before ~2 min. Before G9: at 5% faint 1.1 s, death 5.3 s.

    Reduced: three small sealed rooms at 10 cm in place of 3 x 3 x 2.5 m
    ones. The body does not draw the room's oxygen down (no exhaled breath
    in the sim), so the room's size does not enter."""
    nx = 4
    w = _boxes(nx, 4, 20, rooms=3)
    people = [_column(w, 1 + r * (nx + 1) + 1, 2, 1.6, f"{pct}%")
              for r, pct in enumerate((14, 10, 5))]
    w.step()
    for r, pct in enumerate((14, 10, 5)):
        w.o2[_room(w, nx, r)] = O2_PER_L * w.vox_l * pct / 20.9
    clock = _Clock(w, people)
    _breathe_only(w, int(10 * MIN), clock)
    clock(w)
    f, d = clock.faint, clock.dead
    assert "14%" not in f, f"14% must not faint in 10 min ({f.get('14%')})"
    assert "10%" not in f, f"10% must not faint in 10 min ({f.get('10%')})"
    assert "10%" not in d
    assert 10 * S <= f["5%"] <= 60 * S, f"5%: coma at {f['5%'] / S:.0f} s"
    assert d.get("5%", 10 * MIN) >= 2 * MIN, \
        f"5%: dead at {d.get('5%', -1) / S:.0f} s, before ~2 min"


def _smoke_ramp(t):
    """Smoke at the head of a standing occupant of a bedroom whose bed
    burns, g/L, at `t` seconds after flaming ignition. A t^2 fire (the
    medium growth curve) fills its room with the mass it has burned, so
    the smoke goes as t^3; it reaches the 0.12 g/L the sim calls a
    smoke-logged room (SMOKE_EXT) at three minutes."""
    return 0.12 * (t / 180.0) ** 3


def test_B_T8_SMOKE_IN_THE_BEDROOM_and_the_RESCUE():
    """B-T8 (sources 191, 196; rows B23, B24): the standing occupant of a
    burning bedroom collapses 3-6 min after flaming ignition (accept 2-10);
    one who crawls lasts longer; the rescued victim stays down for at least
    a minute in clean air (COHb half-life 320 min). Before G9: a smoke load
    of 0.00 at 21 s, and recovery in 5-12 s.

    Reduced, because the sim's fire runs on the wrong clock (gap G4) and a
    burning bedroom stepped for six minutes is far too slow for a test: the
    air the two occupants breathe is SET, from `_smoke_ramp`, in two sealed
    rooms. Above 1 m it is the ramp; below, a quarter of it (crawl low; the
    quarter is a stand-in -- the doc asks only that the crawler last
    longer). The oxygen is left fresh, so what puts them down is the
    carbon monoxide the smoke carries (BREATH co_per_smoke). The rescue
    is the standing occupant's room cleared of smoke the tick he goes down:
    carried into clean air."""
    nx = 4
    w = _boxes(nx, 6, 24, rooms=2)
    stands = _column(w, 2, 3, 1.6, "stands")
    crawls = _column(w, 7, 3, 0.5, "crawls")         # head ~0.5 m up
    w.step()
    clock = _Clock(w, (stands, crawls))
    zi = 1 + 10                                       # 1.0 m
    rescued = {}

    def scene(w):
        clock(w)
        c = _smoke_ramp((w.tick - clock.t0) * TICK_S) * w.vox_l
        for r, p in enumerate((stands, crawls)):
            xs = _room(w, nx, r)
            if p is stands and "stands" in clock.faint:
                rescued.setdefault("t", w.tick - clock.t0)
                rescued.setdefault("cohb", p["cohb"])
                w.smoke[xs] = 0.0                     # carried into clean air
                continue
            w.smoke[xs, 1:-1, zi:-1] = c
            w.smoke[xs, 1:-1, 1:zi] = 0.25 * c
    _breathe_only(w, int(10 * MIN), scene)
    clock(w)
    f = clock.faint
    assert 3 * MIN <= f["stands"] <= 6 * MIN, \
        f"standing: down at {f['stands'] / MIN:.2f} min"
    assert f.get("crawls", 10 * MIN) > f["stands"], "the crawler lasts longer"
    down = rescued["t"]
    assert "stands" not in clock.wake or clock.wake["stands"] - down >= 1 * MIN, \
        "the rescued victim stays down for at least a minute"
    assert not stands["awake"] and stands["alive"]
    # and it is the 320 min half-life that is clearing him
    gone = (w.tick - clock.t0 - down) / MIN
    want = rescued["cohb"] * 0.5 ** (gone / BREATH["co_half_min"])
    assert abs(stands["cohb"] - want) < 0.002 * want, (stands["cohb"], want)


def test_a_DOSE_IS_PER_LITRE_not_per_voxel():
    """The same air, per litre, gives the same dose at 5 cm and at 10 cm.
    Before G9 the smoke load read grams per voxel, 8x more at 10 cm."""
    got = []
    for cm in (5, 10):
        n = 40 // cm                                  # 0.4 m wide rooms
        w = _boxes(n, n, 200 // cm, rooms=1, voxel_cm=cm)
        p = _column(w, 1 + n // 2 - 1, 1 + n // 2, 1.6, "breathes")
        w.step()
        w.smoke[w.mat == AIR] = 0.05 * w.vox_l        # 0.05 g/L everywhere
        _breathe_only(w, int(20 * S))
        got.append((p["cohb"], p["smoke"]))
        del w
    (a, sa), (b, sb) = got
    assert a > 0.0 and abs(a - b) < 1e-3 * a, got
    assert abs(sa - sb) < 1e-3 * sa, got


def _side_rooms(cm):
    """P-T4's scene at `cm`: a closed side room with a smouldering source,
    its door ajar (20 cm) into a room where a person sits with their back to
    it. Sizes in metres, so the same scene at any voxel size."""
    v = cm / 100.0
    n = lambda m: int(round(m / v))
    X1, X2, Y, Z = n(0.8), n(0.9), n(0.8), n(1.6)
    nx, ny, nz = X1 + X2 + 3, Y + 2, Z + 2
    w = World(nx, ny, nz, voxel_cm=cm)
    w.open_sky = False
    w.fill(0, nx, 0, ny, 0, nz, STONE)
    w.mat[1:nx - 1, 1:ny - 1, 1:nz - 1] = AIR
    w.smass[1:nx - 1, 1:ny - 1, 1:nz - 1] = 0.0
    wx = 1 + X1                                      # the dividing wall
    w.fill(wx, wx + 1, 1, ny - 1, 1, nz - 1, STONE)
    door = (wx, slice(1 + n(0.3), 1 + n(0.5)), slice(1, 1 + n(1.5)))
    w.mat[door], w.smass[door] = AIR, 0.0
    px, py = wx + 1 + n(0.6), 1 + n(0.35)
    p = _column(w, px, py, 1.2, "sitter")            # the nose at sitting height
    p["facing"] = (1.0, 0.0)                         # back to the door
    w.exits = []
    w.policy = _Wants(legs="stay", hands="hands free")
    x0, z0 = 1 + n(0.2), 1 + n(0.1)                  # a 20 cm cube on the
    return w, p, (slice(x0, x0 + n(0.2)),) * 2 + (slice(z0, z0 + n(0.2)),)


SMOULDER_G_S = 3.0          # smoke, g/s
SMOULDER_O2 = 0.5           # g of oxygen drawn per g of smoke


def _smoulder(w, box):
    """One tick of the smoulder in `box`: smoke in, oxygen out, both spread
    over the box in proportion to what each cell holds, so the same grams
    per litre at any voxel size."""
    w.smoke[box] += SMOULDER_G_S * TICK_S / w.smoke[box].size
    if w.o2 is not None:
        o2 = w.o2[box]
        have = float(o2.sum())
        if have > 0.0:
            o2 *= np.float32(max(1.0 - SMOULDER_O2 * SMOULDER_G_S * TICK_S
                                 / have, 0.0))


def _at_nose(w, p):
    """Extinction coefficient (1/m) of the air round the top 20 cm of a
    body, read by the test on its own: every air cell in a box one cell
    round the head."""
    comp, sl = w._person_cells(p)
    fw = np.argwhere(comp)
    x0, y0 = fw[:, 0].min() + sl[0].start, fw[:, 1].min() + sl[1].start
    x1, y1 = fw[:, 0].max() + sl[0].start, fw[:, 1].max() + sl[1].start
    zt = int(fw[:, 2].max())
    box = (slice(x0 - 1, x1 + 2), slice(y0 - 1, y1 + 2),
           slice(zt + 1 - int(round(0.2 / (0.1 * w.scale))), zt + 2))
    air = w.mat[box] == AIR
    return SMOKE_EXT * float(w.smoke[box][air].mean()) / w.vox_l


def test_P_T4_SMELL_BEFORE_SIGHT_at_any_voxel_size():
    """P-T4 (source 234, row P11): the first smoke percept fires while
    visibility is still 9-10 m or more (sigma <= ~0.3 /m at C = 3), and the
    sigma it fires at agrees within 20% at 5 and 10 cm. Before G9: ~0.4 g/L
    at 5 cm against 0.05 g/L at 10 cm, an 8x gap.

    Reduced: the smoulder is a set source (`_smoulder`), not a burning
    object: the sim's fire runs on its own clock (G4) and differs by voxel
    size, which is not what this asks. 3 g/s of the sim's smoke is far more
    than a real smoulder, so the run is short; what is measured does not
    depend on how fast the smoke came, only on its being read per litre. It
    draws a little oxygen (a smoulder makes much more smoke per gram of
    oxygen than a flame); with none, the air would not be stirred at all,
    because the sim stirs a room only while its oxygen is uneven."""
    fired = {}
    for cm in (10, 5):
        w, p, src = _side_rooms(cm)
        mind = w.minds[p["pid"]]
        for t in range(int(60 * S)):
            _smoulder(w, src)
            w.step()
            p["events"].clear()
            if mind.noticed and mind.noticed[1] == "chokes":
                fired[cm] = (t, _at_nose(w, p))
                break
        del w, p, mind
    assert set(fired) == {5, 10}, f"the smoke must be noticed: {fired}"
    s5, s10 = fired[5][1], fired[10][1]
    assert max(s5, s10) <= 0.3, f"noticed only at sigma {s5:.2f}, {s10:.2f} /m"
    assert 3.0 / max(s5, s10) >= 10.0, "while visibility is still >= 10 m"
    assert abs(s5 - s10) <= 0.2 * min(s5, s10), \
        f"the same smell at both voxel sizes: {s5:.3f} vs {s10:.3f} /m"
