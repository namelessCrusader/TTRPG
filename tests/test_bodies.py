"""Whose flesh is whose: `_claim_bodies` (src/voxel/persons.py) under the
moves that used to hand one body's voxels to another. A body is the flesh
that carries its eid (step E7); these tests were written against the claim
that grew bodies from their deeds, and still say what must hold.

Each test is one root cause, set up with nothing else going on, so that a
failure names the cause and not a scene. Every body here is `scenes._person`,
which is 396 voxels at 5 cm."""
import numpy as np

from src.voxel import harvest
from src.voxel.materials import BODY, CHEST0, TICK_S
from src.voxel.scenes import _person, _room_for, _Wants
from src.voxel.sim import AIR, FLESH, IRON, STONE, World


def _room():
    """A closed stone box with a flat floor and nobody in it."""
    w = World(40, 30, 40, voxel_cm=5)
    w.open_sky = False
    w.fill(0, 40, 0, 30, 0, 40, STONE)
    w.mat[1:39, 1:29, 1:39] = AIR
    w.smass[1:39, 1:29, 1:39] = 0.0
    return w


def _cells(w, q):
    comp, sl = w._person_cells(q)
    if comp is None:
        return np.zeros((0, 3), np.int64)
    got = np.argwhere(comp)
    got[:, 0] += sl[0].start or 0
    got[:, 1] += sl[1].start or 0
    return got


def _flesh(w):
    """Every flesh cell on the lattice, as a set of (x, y, z)."""
    return {tuple(c) for c in np.argwhere(w.mat == FLESH).tolist()}


def _lumps(cells):
    """How many 6-connected pieces a set of cells is in."""
    left, n = set(map(tuple, cells)), 0
    while left:
        n += 1
        stack = [left.pop()]
        while stack:
            x, y, z = stack.pop()
            for c in ((x + 1, y, z), (x - 1, y, z), (x, y + 1, z),
                      (x, y - 1, z), (x, y, z + 1), (x, y, z - 1)):
                if c in left:
                    left.remove(c)
                    stack.append(c)
    return n


def _shape(cells):
    """A set of cells moved to the origin: equal for two sets that are one
    whole-cell move apart."""
    c = np.asarray(cells, np.int64).reshape(-1, 3)
    return sorted(map(tuple, (c - c.min(axis=0)).tolist()))


def _fly_out(w, p, ticks=80):
    """Run the bodies law a tick at a time, claiming between ticks as the sim
    does, until nothing is in the air."""
    for _ in range(ticks):
        if not w.bodies:
            return
        _next_tick(w)
        w._person_cells(p)
        w._law_bodies()
    assert not w.bodies, "everything came down"


def _next_tick(w):
    """The claim is settled once a tick; move the clock, not the world."""
    w.tick += 1


def _side_by_side():
    """Two whole people, A and B, 30 cm apart along y (so each is built and
    first claimed as exactly 396 voxels, not touching), with B standing
    where a quarter turn of his swings his shoulders into A's flank."""
    w = _room()
    a = _person(w, 15, 15)
    a["name"] = "A"
    b = _person(w, 15, 21)
    b["name"] = "B"
    whole = len(_cells(w, a))
    assert whole == len(_cells(w, b)) == 396
    assert int((w.mat == FLESH).sum()) == 2 * whole
    return w, a, b, whole


def _touching():
    """The same two, with B walked over to A by his own steps (deed and all,
    `_carry_own`) until his front is flat against A's back."""
    w, a, b, whole = _side_by_side()
    ca, cb = _cells(w, a), _cells(w, b)
    while int(ca[:, 1].max()) + 1 < int(cb[:, 1].min()):
        assert w._walk(cb, 0, -1), "B can step towards A"
        w._carry_own(b, 0, -1)
        _next_tick(w)
        ca, cb = _cells(w, a), _cells(w, b)
    assert (len(ca), len(cb)) == (whole, whole)
    assert int(ca[:, 1].max()) + 1 == int(cb[:, 1].min()), "they touch"
    return w, a, b, whole


def test_a_body_that_TURNS_beside_another_keeps_ALL_of_itself():
    """A quarter turn moves a body's flesh and not its deed, so the next
    claim seeded B with only the part of him that did not move. The cells
    he turned into were nobody's, and the grow gave the ones A could reach
    in the same round to A, because ties go to the lower index. Measured in
    the crowd test: 414 against 378 of a 396-voxel body, after two turns.

    B is short by exactly the cells he turned into, so he grows first and
    gets those back; A is whole, and gets nothing of him."""
    w, a, b, whole = _side_by_side()
    flesh0 = int((w.mat == FLESH).sum())
    for quarters in (1, 3, 3, 1):
        assert w._turn(b, quarters), "B has room to turn"
        _next_tick(w)
        ca, cb = _cells(w, a), _cells(w, b)
        assert (len(ca), len(cb)) == (whole, whole), \
            f"after a turn of {quarters}: A {len(ca)}, B {len(cb)}"
        assert not ({tuple(c) for c in ca} & {tuple(c) for c in cb})
        assert int((w.mat == FLESH).sum()) == flesh0


def test_flesh_the_PARTICLE_SOLVER_holds_is_nobodys_body():
    """`moving` flesh is drawn by the particle solver for one tick; every
    mover refuses it, and so must the claim, or a man standing against a
    toppled corpse claims its particles as himself. Here even a body with no
    deed yet (its first claim, which may grow freely) must not take them."""
    w = _room()
    a = _person(w, 15, 15)
    a["name"] = "A"
    cells = _cells(w, a)
    body = len(cells)
    x1 = int(cells[:, 0].max()) + 1
    z0, z1 = int(cells[:, 2].min()), int(cells[:, 2].max()) + 1
    y0, y1 = int(cells[:, 1].min()), int(cells[:, 1].max()) + 1
    w.mat[x1, y0:y1, z0:z1] = FLESH                 # pressed against him
    w.smass[x1, y0:y1, z0:z1] = 0.1
    w.moving[x1, y0:y1, z0:z1] = True
    a["_own"] = frozenset()                         # no deed: first claim
    _next_tick(w)
    got = _cells(w, a)
    assert len(got) == body, f"claimed {len(got)}, the body is {body}"
    assert not w.moving[tuple(got.T)].any()


def test_a_man_IN_THE_AIR_holds_no_deed_on_the_cells_he_LEFT():
    """A man launched whole keeps his deed so a staggered landing is still
    his, but the cells he stood in are air the moment he leaves, and anyone
    can move into air. A kept deed on them meant two people owning one voxel:
    9 of the first 30 fuzzer worlds, every time with one of the two aloft.
    What he keeps is only flesh he really left on the lattice."""
    w, a, b, whole = _touching()
    cb = _cells(w, b)
    w._launch(cb, (0.0, 0.0, 2.0), owner="B")
    assert w.bodies, "B is in the air"
    _next_tick(w)
    assert _cells(w, a).shape[0] == whole
    assert w._person_cells(b)[0] is None, "B is aloft, so he claims nothing"
    own_b = b.get("_own") or frozenset()
    assert all(int(w.mat[c]) == FLESH for c in own_b), \
        "an aloft deed names only flesh on the lattice"
    # A steps into where B stood: two people never own the same voxel
    for _ in range(3):
        ca = _cells(w, a)
        if not w._walk(ca, 0, 1):
            break
        w._carry_own(a, 0, 1)
        _next_tick(w)
        assert len(_cells(w, a)) == whole
        assert not (a["_own"] & (b.get("_own") or frozenset()))


def test_a_TURNED_REACHING_or_CROUCHING_body_s_DEED_follows_its_FLESH():
    """The movers in pose.py move a body's voxels inside a tick. Each now
    tells the deed where they went (`_move_own`), so between the move and
    the next claim the deed is the body and not where it used to be."""
    w = _room()
    p = _person(w, 15, 15)
    _cells(w, p)
    assert w._turn(p, 1)
    assert p["_own"] == _flesh(w), "after a quarter turn"
    _next_tick(w)
    assert w._repose(p, "right arm", [0.8, 0.0]) == "moved"
    assert p["_own"] == _flesh(w), "after a reach"
    _next_tick(w)
    assert w._crouch(p, 0.8) == "moved"
    assert p["_own"] == _flesh(w), "after a crouch"


def test_a_man_LAUNCHED_is_aloft_for_the_REST_OF_THE_TICK():
    """The claim this tick was made with him standing. Launched whole, he
    has no body on the lattice, and anything later in the same tick that
    asks must hear that, not the cells he left (fuzzer seed 16: a man in
    the air walked 318 voxels of air)."""
    w, a, b, whole = _side_by_side()
    cb = _cells(w, b)                               # this tick's claim
    w._launch(cb, (0.0, 0.0, 2.0), owner="B")
    assert w._person_cells(b)[0] is None, "B is in the air now"
    assert len(_cells(w, a)) == whole


def test_a_blow_on_flesh_NOBODY_OWNS_wounds_no_one():
    """A dead man's deed is cleared, so his flesh is nobody's, and a blow
    that lands on it has no one to wound. It used to raise instead (fuzzer
    seed 58, at t5)."""
    w = _room()
    a = _person(w, 15, 8)
    a["name"] = "A"
    dead = _person(w, 15, 20)
    dead["name"] = "dead"
    dead["alive"] = False
    _next_tick(w)
    _cells(w, a)
    assert not dead.get("_own")
    cell = next(tuple(c) for c in np.argwhere(w.mat == FLESH).tolist()
                if w._person_at(tuple(c)) is None)
    stone = w.loose(cell, (5.0, 0.0, 0.0), STONE, 500.0)
    hurt = a["hurt"]
    w._spend(stone, [cell], 50.0)
    assert int(w.mat[cell]) == FLESH and a["hurt"] == hurt


def test_a_body_that_KEELS_OVER_comes_down_in_ONE_PIECE():
    """A topple was turned and then rounded to the lattice, which sends two
    voxels into one cell and two neighbours apart: one standing body came
    out in 3 to 18 pieces at every lean between 0.05 and 1.4 rad, and the
    claim, which floods one lump, kept the biggest. The landing is now the
    skeleton's three shears, and a pose in pieces is not taken."""
    for axis in (0, 1):
        for theta in (0.3, 0.6, 1.0):
            w = World(70, 70, 40, voxel_cm=5)
            w.open_sky = False
            w.fill(0, 70, 0, 70, 0, 1, STONE)
            p = _person(w, 30, 30)
            cells = _cells(w, p)
            whole = len(cells)
            foot = cells[cells[:, 2] <= cells[:, 2].min() + 1]   # as `_collapse`
            w._topple(cells.astype(np.float32), axis, 1,
                      float(foot[:, axis].max()), float(cells[:, 2].min()),
                      owner=p["name"])
            assert len(w.bodies) == 1, "he keels over"
            w._land_body(w.bodies.pop(), theta)
            assert len(_flesh(w)) == whole
            assert _lumps(_flesh(w)) == 1, \
                f"axis {axis}, {theta} rad: {_lumps(_flesh(w))} pieces"
            assert p["_own"] == _flesh(w)
            if theta > np.pi / 4:      # nearer flat than upright: not stood up
                tall = max(c[2] for c in _flesh(w)) - min(c[2] for c in _flesh(w))
                assert tall + 1 < int(np.ptp(cells[:, 2])) + 1, \
                    f"axis {axis}, {theta} rad: stood back up ({tall + 1} cells tall)"
            del w


def test_a_man_who_FALLS_any_way_ends_up_LYING_on_the_floor():
    """Turned about his feet, a man whose shoulders are wider than his stance
    put a shoulder through the floor at pi/2, so flat was never clear, the
    pose the fall stopped at was a staircase, and he was redrawn STANDING
    where he started: every fall across the shoulder line, in an open room.
    Flat now rests on the floor it fell to."""
    for axis in (0, 1):
        for s in (1, -1):
            w = World(70, 70, 40, voxel_cm=5)
            w.open_sky = False
            w.fill(0, 70, 0, 70, 0, 1, STONE)
            p = _person(w, 35, 35)
            cells = _cells(w, p)
            tall = int(np.ptp(cells[:, 2])) + 1
            foot = cells[cells[:, 2] <= cells[:, 2].min() + 1]
            w._topple(cells.astype(np.float32), axis, s,
                      float(foot[:, axis].max() if s > 0 else foot[:, axis].min()),
                      float(cells[:, 2].min()), owner=p["name"])
            _fly_out(w, p, ticks=200)
            got = np.array(sorted(_flesh(w)))
            assert len(got) == len(cells)
            assert _lumps(got) == 1
            assert p["_own"] == _flesh(w)
            assert int(np.ptp(got[:, 2])) + 1 < tall // 2, \
                f"axis {axis}, side {s}: {int(np.ptp(got[:, 2])) + 1} of {tall} cells tall"
            del w


def test_a_man_who_FAINTS_in_an_open_room_stays_DOWN(monkeypatch):
    """Stood back up by his own landing, an unconscious man was collapsed
    again every 8 ticks for as long as he was out, in the middle of an
    empty room."""
    from src.voxel import physiology
    from src.voxel.scenes import _Wants
    monkeypatch.setitem(physiology.BODY, "wake_o2", 9e9)   # he stays out
    w = World(60, 60, 40, voxel_cm=5)
    w.open_sky = False
    w.fill(0, 60, 0, 60, 0, 40, STONE)
    w.mat[1:59, 1:59, 1:39] = AIR
    w.smass[1:59, 1:59, 1:39] = 0.0
    w.exits = []
    p = _person(w, 30, 30)
    p["name"] = "out"
    w.policy = _Wants(each={"out": {"hands": "hands free", "legs": "stay"}})
    p["awake"] = False
    falls = []
    real = w._crumple          # what the slack body does (was `_collapse`)
    w._crumple = lambda *a: (falls.append(w.tick), real(*a))[1]
    for _ in range(60):
        w.step()
    got = np.array(sorted(_flesh(w)))
    assert len(falls) == 1, f"keeled over at ticks {falls}"
    assert not w.bodies and int(np.ptp(got[:, 2])) + 1 < 15, "lying down"


def test_a_LANDING_deeds_his_FLESH_and_not_what_he_HOLDS():
    """A landing wrote the deed as every cell it put down, so a man who came
    down holding iron owned the iron as part of himself."""
    w = _room()
    p = _person(w, 15, 15)
    cells = _cells(w, p)
    top = cells[cells[:, 2] == cells[:, 2].max()][0]
    iron = (int(top[0]), int(top[1]), int(top[2]) + 1)
    w.fill(iron[0], iron[0] + 1, iron[1], iron[1] + 1, iron[2], iron[2] + 1,
           IRON)
    w._launch(np.vstack([cells, [iron]]), (0.0, 0.0, 1.0), owner=p["name"])
    _fly_out(w, p)
    assert int((w.mat == IRON).sum()) == 1
    assert p["_own"] == _flesh(w), "the deed is his flesh, and only that"


def test_a_SWUNG_ARM_that_comes_back_is_in_his_DEED_where_his_BONES_say():
    """The claims made while an arm is out of the lattice write him without
    it, and its landing never wrote it back, so the arm came down owned by
    nobody. Its bones were also re-based on the corner of the body WITHOUT
    the arm, which is the wrong corner when the arm is the body's edge."""
    w = _room()
    p = _person(w, 15, 15)
    _cells(w, p)
    assert w._swing(p, "left arm") is not None
    _fly_out(w, p)
    assert p["_own"] == _flesh(w), "the arm came back to the deed"
    origin = np.array(sorted(p["_own"])).min(axis=0)
    for bone in ("left upper arm", "left forearm"):
        assert all(int(w.mat[tuple(c)]) == FLESH
                   for c in np.asarray(p["segs"][bone]) + origin), bone


def test_an_arm_out_when_he_STEPS_comes_back_to_HIM_not_to_the_PLACE():
    """A man can step while his arm is out, and the arm came back to the
    cells it left, a step away from him: 54 voxels of arm lying beside him
    for the rest of the run (fuzzer seed 134). It comes back moved as the
    rest of him moved, and a step later in the same tick takes it along."""
    w = _room()
    p = _person(w, 15, 15)
    _cells(w, p)
    assert w._swing(p, "left arm") is not None
    stepped = 0
    for _ in range(80):
        if not w.bodies:
            break
        _next_tick(w)
        if stepped < 2:
            body = _cells(w, p)
            if w._walk(body, 1, 0):           # away from his left arm
                w._carry_own(p, 1, 0)
                stepped += 1
        w._person_cells(p)
        w._law_bodies()
    assert not w.bodies and stepped >= 1, "he stepped with his arm out"
    assert _lumps(_flesh(w)) == 1, "the arm is on the man"
    assert p["_own"] == _flesh(w)
    body = _cells(w, p)                       # and the next step takes it
    assert w._walk(body, -1, 0)
    w._carry_own(p, -1, 0)
    assert _lumps(_flesh(w)) == 1


def test_a_man_who_comes_down_in_TWO_PIECES_owns_BOTH():
    """Each landing replaced the deed with what it put down, so of a man who
    came down in two landings only the last piece was his (roadmap 87)."""
    w = _room()
    p = _person(w, 15, 15)
    cells = _cells(w, p)
    whole = len(cells)
    w._launch(cells[cells[:, 2] < 16], (0.0, 0.0, 0.0), owner=p["name"])
    w._launch(cells[cells[:, 2] >= 16], (0.0, 0.0, 0.0), owner=p["name"])
    _fly_out(w, p)
    assert len(_flesh(w)) == whole
    assert p["_own"] == _flesh(w), \
        f"he owns {len(p['_own'])} of his {whole} voxels"


# ── BODY11: a body lands in one piece ───────────────────────────────────────

def test_a_man_who_JUMPS_mid_SWING_takes_his_ARM_with_him():
    """A swung arm is a free body of its own, and it stayed out while the
    rest of him jumped: it came back to where he had been, he came down
    somewhere else, and he was a man and an arm apart from then on (fuzzer
    seeds 7, 27, 37, 45). The arm comes home when he leaves the ground."""
    w = _room()
    p = _person(w, 15, 15)
    whole = len(_cells(w, p))
    assert w._swing(p, "left arm") is not None
    w._launch(_cells(w, p), (1.5, 0.0, 1.0), owner=p["name"])
    assert not any(b.get("part") for b in w.bodies), "the arm came home"
    _fly_out(w, p)
    assert len(_flesh(w)) == whole and _lumps(_flesh(w)) == 1
    assert p["_own"] == _flesh(w)
    origin = np.array(sorted(p["_own"])).min(axis=0)
    for bone in ("left upper arm", "left forearm"):
        assert all(int(w.mat[tuple(c)]) == FLESH
                   for c in np.asarray(p["segs"][bone]) + origin), bone


def test_a_man_who_KEELS_OVER_mid_SWING_takes_his_ARM_with_him():
    """The same for a topple: the body went over, the arm came back to
    where he had stood."""
    w = World(70, 70, 40, voxel_cm=5)
    w.open_sky = False
    w.fill(0, 70, 0, 70, 0, 1, STONE)
    p = _person(w, 30, 30)
    whole = len(_cells(w, p))
    assert w._swing(p, "left arm") is not None
    cells = _cells(w, p)
    foot = cells[cells[:, 2] <= cells[:, 2].min() + 1]
    w._topple(cells.astype(np.float32), 1, 1, float(foot[:, 1].max()),
              float(cells[:, 2].min()), owner=p["name"])
    assert len(w.bodies) == 1 and not w.bodies[0].get("part")
    assert len(w.bodies[0]["cells"]) == whole, "all of him is going over"
    _fly_out(w, p, ticks=200)
    assert len(_flesh(w)) == whole and _lumps(_flesh(w)) == 1


def test_a_man_who_goes_over_with_his_ARM_OUT_takes_ALL_of_himself():
    """Going over, the arm held out comes down at once (`_law_footing`),
    and the body was then launched from the cells read BEFORE it came down:
    the cells the arm had left (air) went into the air, and the arm stayed
    on the floor (fuzzer seed 6: 44 cells of him left behind, then
    [342, 18]). Every cell of him leaves, and he lands in one piece."""
    w = _room()
    p = _person(w, 15, 15)
    whole = len(_cells(w, p))
    for th in (1.2, 1.0, 0.8, 1.4, 0.6, 0.4):
        if w._repose(p, "right arm", th) == "moved":
            break
    else:
        raise AssertionError("no drawable arm-out pose")
    p["pose"] = {"right arm": [th, 0.0, 0.0, 0.0]}
    out = _cells(w, p)
    assert len(out) == whole
    # WHY he goes over is not the question here: he does.
    w._pulled_over = lambda q, own: None
    w._overbalanced = lambda q, own: (2.0, 0.0)
    w._law_footing()
    flier = [b for b in w.bodies if b.get("owner") == p["name"]]
    assert len(flier) == 1 and not flier[0].get("part")
    mine = (flier[0]["mats"] == FLESH) & (flier[0]["eids"] == p["eid"])
    assert int(mine.sum()) == whole, f"{int(mine.sum())} of {whole} left"
    assert not ((w.mat == FLESH) & (w.eid == p["eid"])).any(), \
        "none of him stayed on the floor"
    assert (flier[0]["mats"] != AIR).all(), "and no air went with him"
    _fly_out(w, p)
    got = _flesh(w)
    assert len(got) == whole and _lumps(got) == 1


def test_a_man_who_LANDS_where_something_now_STANDS_comes_down_WHOLE():
    """A flight stops at its last clear pose, but the room can change under
    it within the tick (a man steps in, something falls there): the cells in
    the way were piled up their columns as debris, a man in pieces (fuzzer
    seed 14: 11 cells of him piled, then 22). He is moved, the nearest whole
    cell move that is clear."""
    w = _room()
    p = _person(w, 15, 15)
    cells = _cells(w, p)
    whole = len(cells)
    w._launch(cells, (0.0, 0.0, 0.0), owner=p["name"])
    x = int(cells[:, 0].min())                # a post where his edge was
    y = int(cells[cells[:, 0] == x][0, 1])
    w.fill(x, x + 1, y, y + 1, 1, 39, STONE)
    stone = int((w.mat == STONE).sum())
    _fly_out(w, p)
    got = _flesh(w)
    assert len(got) == whole and _lumps(got) == 1, \
        f"{len(got)} of {whole} cells, in {_lumps(got)} pieces"
    assert int((w.mat == STONE).sum()) == stone and p["_own"] == got


def test_an_ARM_whose_place_was_TAKEN_comes_back_WHOLE_and_ON_HIM():
    """An arm comes back to where it left from, and the cells of it that
    were taken by then were piled up their columns as debris: an arm in
    pieces, or beside the man. It is moved a cell or two, to where it is
    clear and every piece of it touches him (the same rule brings back the
    raised arm of fuzzer seeds 2, 12, 19 and 36 after he moved)."""
    w = _room()
    p = _person(w, 15, 15)
    whole = len(_cells(w, p))
    b = w._swing(p, "left arm")
    assert b is not None
    rest = _flesh(w)                             # the rest of him
    arm = np.round(b["cells"]).astype(np.int64)
    low = arm[arm[:, 2] == arm[:, 2].min()][0]   # the hand: something is
    w.fill(int(low[0]), int(low[0]) + 1, int(low[1]), int(low[1]) + 1,  # there
           int(low[2]), int(low[2]) + 1, IRON)
    _fly_out(w, p)
    got = _flesh(w)
    assert len(got) == whole and _lumps(got) == 1, \
        f"{len(got)} of {whole} cells, in {_lumps(got)} pieces"
    assert int((w.mat == IRON).sum()) == 1 and p["_own"] == got
    back = np.array(sorted(got - rest))
    assert _shape(back) == _shape(arm), "the arm, whole: moved, not piled"


def test_a_man_with_an_ARM_OUT_in_a_swing_does_not_TURN_till_it_is_BACK():
    """A stride squares a body up to where it walks (`_square_up`), and a
    turn moves the flesh on the lattice: a swung arm is not on it. After a
    half turn his other arm stood where the swung one had left from, and
    the swung arm came back into it, every cell piled up its column onto
    him (fuzzer seeds 17, 111, 134: 54 cells each). He turns once the arm
    is back, and the arm comes back whole, on him."""
    w = _room()
    p = _person(w, 15, 15)
    whole = len(_cells(w, p))
    b = w._swing(p, "left arm")
    assert b is not None
    arm = np.round(b["cells"]).astype(np.int64)
    rest = _flesh(w)                             # the rest of him
    cx, cy = p.get("chest", CHEST0)
    p["facing"] = (-float(cx), -float(cy))       # he means to walk back
    assert not w._square_up(p), "not with his arm out"
    _fly_out(w, p)
    got = _flesh(w)
    assert len(got) == whole and _lumps(got) == 1, \
        f"{len(got)} of {whole} cells, in {_lumps(got)} pieces"
    assert p["_own"] == got
    assert _shape(np.array(sorted(got - rest))) == _shape(arm), \
        "the arm, whole: not piled"
    _next_tick(w)
    assert w._square_up(p), "and with it back, he turns"
    assert len(_flesh(w)) == whole and _lumps(_flesh(w)) == 1


def test_a_KEELING_body_with_NOWHERE_CLEAR_to_lie_is_MOVED_not_PILED():
    """A topple that found no clear pose took one with a graze in it, and
    each grazing cell was piled up its column: a cell of him piled is a cell
    of him somewhere else (fuzzer seeds 5, 43). Here a low wall runs the
    length of the room under his near side, which lifting him (`rest`) and
    sliding him back along the fall (`slump`) cannot clear, and the place
    his head was has had something put in it. The pose is moved, a whole
    cell at a time, until it is clear: here one cell to the side."""
    for axis in (0, 1):
        w = World(70, 70, 40, voxel_cm=5)
        w.open_sky = False
        w.fill(0, 70, 0, 70, 0, 1, STONE)
        p = _person(w, 30, 30)
        cells = _cells(w, p)
        whole = len(cells)
        foot = cells[cells[:, 2] <= cells[:, 2].min() + 1]
        w._topple(cells.astype(np.float32), axis, 1,
                  float(foot[:, axis].max()), float(cells[:, 2].min()),
                  owner=p["name"])
        b = w.bodies.pop()
        flat = w._lay_down(dict(b), np.pi / 2)
        assert not w._in_the_way(flat)
        shapes = [_shape(flat), _shape(cells)]
        side = 1 - axis                        # the wall runs along the fall,
        wall = [0, 70, 0, 70, 1, int(flat[:, 2].max()) + 2]   # just over him
        wall[2 * side] = int(flat[:, side].min())
        wall[2 * side + 1] = wall[2 * side] + 1
        w.fill(*wall, IRON)
        top = cells[np.argmax(cells[:, 2])]    # and his head's place is taken
        w.fill(int(top[0]), int(top[0]) + 1, int(top[1]), int(top[1]) + 1,
               int(top[2]), int(top[2]) + 1, IRON)
        iron = int((w.mat == IRON).sum())
        assert w._in_the_way(flat) and w._in_the_way(cells), "neither is clear"
        w._land_body(b, np.pi / 2)
        got = _flesh(w)
        assert len(got) == whole and _lumps(got) == 1, \
            f"axis {axis}: {len(got)} of {whole} cells, in {_lumps(got)} pieces"
        assert _shape(np.array(sorted(got))) in shapes, \
            f"axis {axis}: lying or standing, moved whole, not piled"
        assert int((w.mat == IRON).sum()) == iron, "and nothing it hit moved"
        del w


def test_a_SOLE_PAST_THE_EDGE_of_the_world_stands_on_NOTHING():
    """`_underfoot` reads every sole at once since the optimization wave,
    so a body partly off the lattice raised an IndexError (the old loop
    stopped at the first sole that stood on something), and a sole at
    x = -1 read the far side of the world."""
    w = World(30, 12, 44, voxel_cm=5)
    w.fill(0, 30, 0, 12, 0, 1, STONE)
    p = _person(w, 10, 6)
    cells = _cells(w, p)
    soles = cells[cells[:, 2] == cells[:, 2].min()]
    past = cells.copy()
    past[:, 0] += 30 - int(soles[:, 0].max())  # the last sole column is off
    assert (past[:, 0] >= 30).any() and (past[past[:, 2] == past[:, 2].min(), 0] < 30).any()
    assert w._underfoot(past), "the soles still in the world stand on stone"
    off = cells.copy()
    off[:, 0] += 40                            # all of him past the edge
    assert not w._underfoot(off)
    left = cells.copy()
    left[:, 0] -= int(soles[:, 0].max()) + 1   # every sole at x < 0
    assert not w._underfoot(left), "not the floor at the far side of the world"


# ── F9: the catch gate, a corpse's flesh, and room for a whole body ─────────

def _catcher():
    """One ready man on a stone floor, stepped once so he has a body, and an
    iron stone 100 g dropping past his right hand: a catch his arm can stop,
    so only WHEN he may close his hand is in question."""
    w = World(40, 30, 44, voxel_cm=5)
    w.fill(0, 40, 0, 30, 0, 1, STONE)
    p = _person(w, 12, 10)
    p["name"], p["ready"], p["ready_s"] = "him", True, 0.15
    w.policy = _Wants()
    w.step()
    b = w.loose((20, 10, 20), (0.0, 0.0, -1.0), IRON, 100.0)
    return w, p, b


def test_a_CATCH_counts_from_when_he_SAW_it_not_from_the_THROW():
    """`_catch` refused anything thrown less than BODY['react_s'] (0.2 s)
    ago, however long the man had been watching it come. The gate is the
    sighting now (`_in_time`, R3): a ready man who saw it leave the hand
    closes on it 0.15 s later, sooner than 0.2 s after the throw."""
    w, p, b = _catcher()
    b["born"] = w.tick - 6                    # thrown 6 ticks ago...
    b["_seen"] = {p["pid"]: (w.tick - 6, True)}   # ...and seen as it left
    assert (w.tick - b["born"]) * TICK_S < BODY["react_s"], \
        "this is sooner after the throw than the old gate allowed"
    assert w._catch(p, b, "right arm"), "seen in time, so it is caught"
    assert p.get("held") and p["held"]["mat"] == IRON


def _matter(w):
    """Every cell of matter, on the lattice or in the air, as sorted
    (material, grams, edge) rows: the books a move must keep exactly."""
    on = w.mat != AIR
    rows = [np.stack([w.mat[on].astype(np.float64),
                      w.smass[on].astype(np.float64),
                      w.edge[on].astype(np.float64)], 1)]
    for b in w.bodies:
        rows.append(np.stack([b["mats"].astype(np.float64),
                              b["masses"].astype(np.float64),
                              b["edges"].astype(np.float64)], 1))
    a = np.concatenate(rows)
    return a[np.lexsort(a.T[::-1])]


def test_a_THING_HELD_in_his_hand_LEAVES_and_LANDS_with_a_JUMPER():
    """A man who jumped holding a stone left it in the air where his hand
    had been: it fell there, and his grip named a cell it was no longer in,
    so he had let go without deciding to. What hangs from his hand goes up
    with him through the same `_lift`, comes down through the same `_put`,
    and is still in his hand when he lands; every gram of it is accounted
    for, on the lattice or in the air, the whole way."""
    w, p, b = _catcher()
    b["born"] = w.tick - 6
    b["_seen"] = {p["pid"]: (w.tick - 6, True)}
    assert w._catch(p, b, "right arm")
    w._take_up(p, _cells(w, p))                  # in his hand, off the floor
    stone = w._held_cells(p)
    assert stone is not None and not len(w._contact(stone, ignore=_cells(w, p)))
    e, books = p["held"]["eid"], _matter(w)
    whole = len(_cells(w, p))
    w._launch(_cells(w, p), (1.5, 0.0, 1.5), owner=p["name"])
    assert not (w.mat == IRON).any(), "the stone left the floor with him"
    assert len(w.bodies) == 1 and int((w.bodies[0]["eids"] == e).sum()) == 1
    assert np.array_equal(_matter(w), books), "nothing made or lost going up"
    _fly_out(w, p)
    assert np.array_equal(_matter(w), books), "nothing made or lost coming down"
    got = w._held_cells(p)
    assert got is not None and p.get("held"), "still in his hand"
    assert int(w.eid[tuple(got[0])]) == e and int(w.mat[tuple(got[0])]) == IRON
    assert w._in_reach(_cells(w, p), got), "beside him, not where he left"
    assert len(_flesh(w)) == whole and _lumps(_flesh(w)) == 1


def test_a_HELD_THING_that_lands_without_the_GRIPPED_CELL_is_still_HELD():
    """The grip followed only the one cell in his hand. If that cell did
    not land (past the edge of the world) and the rest of the thing did,
    the grip still named the old cell, `_held_cells` found the thing gone
    from it, and he let go without deciding to. The hand is on what is
    left of the thing, found by its eid. And where he came down is where
    HE is: the thing in his hand is not counted in his anchor."""
    w, p, b = _catcher()
    b["born"] = w.tick - 6
    b["_seen"] = {p["pid"]: (w.tick - 6, True)}
    assert w._catch(p, b, "right arm")
    w._take_up(p, _cells(w, p))
    x, y, z = map(int, w._held_cells(p)[0])
    e = p["held"]["eid"]
    for c in ((x + 1, y, z), (x + 2, y, z)):      # a longer iron thing
        assert w.mat[c] == AIR
        w.mat[c], w.smass[c], w.eid[c] = IRON, 100.0, e
    w._thing_gen = getattr(w, "_thing_gen", 0) + 1
    assert len(w._held_cells(p)) == 3
    w._launch(_cells(w, p), (0.0, 0.0, 1.5), owner=p["name"])
    fly = w.bodies[0]
    assert int((fly["eids"] == e).sum()) == 3, "all of it went up with him"
    fly["cells"][fly["grip"]] = (w.shape[0] + 40, y, z)   # will not land
    _fly_out(w, p)
    assert int(((w.mat == IRON) & (w.eid == e)).sum()) == 2
    got = w._held_cells(p)
    assert got is not None and p.get("held"), "he still holds what is left"
    assert int(w.eid[tuple(got[0])]) == e and len(got) == 2
    fl = _cells(w, p)
    assert p["anchor"] == (int(round(float(fl[:, 0].mean()))),
                           int(round(float(fl[:, 1].mean())))), \
        "his anchor is his own flesh"


def test_a_thing_seen_LATE_or_NEVER_is_not_caught_however_long_it_flew():
    """The other side of the same gate: a stone that has been in the air
    for 25 s is no easier to catch for that if he saw it only 2 ticks ago,
    and one he never saw at all is not caught."""
    w, p, b = _catcher()
    b["born"] = w.tick - 1000                 # long in the air
    b["_seen"] = {p["pid"]: (w.tick - 2, True)}   # seen just now
    assert not w._catch(p, b, "right arm"), "seen too late to answer"
    del b["_seen"]
    assert not w._catch(p, b, "right arm"), "never seen"
    assert any(x is b for x in w.bodies) and not p.get("held"), \
        "it goes on past him"


def test_a_living_man_does_not_CLAIM_the_CORPSE_lying_against_him():
    """The dead lose their deed, so a corpse's flesh was free, and a living
    man lying against it grew into all of it: 792 cells, 396 of them the
    dead man's, for the rest of the run (9 of 150 fuzzer worlds, E6). A
    body is its eid now (E7), and the corpse's flesh still carries the dead
    man's: it is nobody's body to move, and still his flesh."""
    w, a, b, whole = _touching()
    corpse = {tuple(c) for c in _cells(w, a).tolist()}
    a["alive"] = False
    for _ in range(3):
        _next_tick(w)
        mine = {tuple(c) for c in _cells(w, b).tolist()}
        assert len(mine) == whole, f"he owns {len(mine)} of his {whole}"
        assert not mine & corpse, "and none of it is the dead man's"
        assert a["_own"] == frozenset() and _cells(w, a).size == 0, \
            "the dead man is no body anyone moves"
        assert {tuple(c) for c in np.argwhere(
            w.body_by_eid(a["eid"])).tolist()} == corpse, \
            "the corpse is still his flesh"


def test_a_FORGOTTEN_DEED_costs_a_body_nothing():
    """The claim used to grow from last tick's deed, so a deed that forgot
    part of a body (a mover that did not say where it went) left that part
    to whoever reached it first, and a corpse had to be kept from taking it
    back. The claim reads the eid now: what the deed says a moment ago does
    not decide anything."""
    w, a, b, whole = _touching()
    a["alive"] = False
    _next_tick(w)
    _cells(w, b)
    arm = {c for c in b["_own"] if c[2] >= 25}       # his top, deed forgotten
    b["_own"] = b["_own"] - arm
    _next_tick(w)
    mine = {tuple(c) for c in _cells(w, b).tolist()}
    assert arm <= mine and len(mine) == whole, "he has all of himself"
    assert not ({tuple(c) for c in np.argwhere(
        w.body_by_eid(a["eid"])).tolist()} & arm), "and the corpse none of it"


# ── E7: a body is the flesh that carries its eid ─────────────────────────────

def test_a_neighbour_s_flesh_LAUNCHED_with_a_man_is_not_HIS_when_it_lands():
    """Fuzzer seed 108 (E6): the claim's grow handed a neighbour's swung arm
    to the man it touched, he was launched carrying it, and the landing
    deeded it to him. Here 18 cells of B's top are sent up with A, as that
    launch did. They come down as B's, because they carry B's eid; while
    they are in the air B is in the air too, and has no body to move."""
    w, a, b, whole = _touching()
    ca, cb = _cells(w, a), _cells(w, b)
    piece = cb[cb[:, 2] >= int(cb[:, 2].max()) - 1][:18]
    w._launch(np.vstack([ca, piece]), (0.0, 0.0, 1.0), owner="A")
    assert w._person_cells(b)[0] is None, "some of B is in the air"
    on, up = w.body_of(b)
    assert len(on) + len(up) == whole, "and all of him is still his"
    _fly_out(w, a)
    _next_tick(w)
    mine_a = {tuple(c) for c in _cells(w, a).tolist()}
    mine_b = {tuple(c) for c in _cells(w, b).tolist()}
    assert len(mine_a) == whole, f"A owns {len(mine_a)} of his {whole}"
    assert len(mine_b) == whole, f"B owns {len(mine_b)} of his {whole}"
    assert not mine_a & mine_b
    assert a["_own"] == frozenset(mine_a), "the landing deeded only his own"


def test_TWO_MEN_who_keel_over_as_ONE_LUMP_come_down_as_two_whole_men():
    """The support law topples a lump, and two people lying in contact are
    one lump with one owner (`_law_torque`'s `whose`). The eid on the flesh
    says the other man is in the air too, and the landing keeps EACH of
    them in one piece (`_lay_down`), not merely the lump."""
    w, a, b, whole = _touching()
    a["awake"] = b["awake"] = False
    ca, cb = _cells(w, a), _cells(w, b)
    both = np.vstack([ca, cb])
    foot = both[both[:, 2] <= both[:, 2].min() + 1]
    w._topple(both.astype(np.float32), 0, 1, float(foot[:, 0].max()),
              float(both[:, 2].min()), owner="A")
    assert len(w.bodies) == 1, "they keel over"
    _next_tick(w)
    assert w._person_cells(b)[0] is None, "B goes over with A"
    _fly_out(w, a, ticks=200)
    _next_tick(w)
    for q in (a, b):
        got = _cells(w, q)
        assert len(got) == whole and _lumps(got) == 1, \
            f"{q['name']}: {len(got)} cells in {_lumps(got)} pieces"


def test_a_man_HALF_LANDED_is_one_body_on_the_floor_and_in_the_air():
    """A landing is not all at once. Between the two landings of a man in
    two free bodies he has no body to move (the free body owns the flight),
    his deed is what he has put down, and `body_of` finds all of him: the
    flesh on the floor and the flesh still in the air."""
    w = _room()
    p = _person(w, 15, 15)
    cells = _cells(w, p)
    whole = len(cells)
    w._launch(cells[cells[:, 2] < 16], (0.0, 0.0, 0.0), owner=p["name"])
    w._launch(cells[cells[:, 2] >= 16], (0.0, 0.0, 3.0), owner=p["name"])
    half = False
    for _ in range(80):
        if not w.bodies:
            break
        _next_tick(w)
        w._person_cells(p)
        w._law_bodies()
        if w.bodies and (w.mat == FLESH).any():
            half = True
            _next_tick(w)
            assert w._person_cells(p)[0] is None, "half landed: aloft"
            assert p["_own"] == _flesh(w), "his deed is what he put down"
            on, up = w.body_of(p)
            assert len(on) == len(_flesh(w)) and len(on) + len(up) == whole
    assert half, "he came down in two landings"
    _next_tick(w)
    assert len(_cells(w, p)) == whole


def test_a_body_IN_PIECES_is_still_ONE_PERSON():
    """Identity is the eid, not connectivity. A man whose head is moved off
    his shoulders (by hand here; a clipped build or a bad landing in the
    fuzzer) is two lumps, and both are him: the claim gives him both, and
    a blow on either finds him."""
    w = _room()
    p = _person(w, 15, 15)
    cells = _cells(w, p)
    whole = len(cells)
    head = cells[cells[:, 2] >= int(cells[:, 2].max()) - 3]
    to = head + np.array([8, 0, 0])
    for arr in (w.mat, w.smass, w.eid):
        v = arr[tuple(head.T)].copy()
        arr[tuple(head.T)] = 0
        arr[tuple(to.T)] = v
    _next_tick(w)
    got = _cells(w, p)
    assert len(got) == whole and _lumps(got) == 2
    assert w._person_at(tuple(int(v) for v in to[0])) is p
    assert w._person_at(tuple(int(v) for v in got[0])) is p


def test_FLESH_built_by_hand_is_ADOPTED_once_and_then_is_the_body():
    """The older tests build a body with `fill` and then `add_person`, so
    none of its flesh carries an eid. The first claim adopts it, split
    between two such bodies in contact by who is nearer, and stamps it;
    after that the eid is the body and the claim does no growing."""
    w = _room()
    w.fill(10, 13, 10, 13, 1, 20, FLESH, frac=0.9)
    w.fill(13, 16, 10, 13, 1, 20, FLESH, frac=0.9)      # touching
    a = w.add_person(11, 11, "A")
    b = w.add_person(14, 11, "B")
    w.audit_bodies = True
    _next_tick(w)
    ca, cb = _cells(w, a), _cells(w, b)
    assert len(ca) == len(cb) == 3 * 3 * 19
    assert (ca[:, 0] < 13).all() and (cb[:, 0] >= 13).all()
    assert (w.eid[tuple(ca.T)] == a["eid"]).all()
    adopted = [r for r in w.id_audit if "adopted" in r]
    assert sorted(r["adopted"] for r in adopted) == [171, 171]
    _next_tick(w)
    _cells(w, a)
    assert len([r for r in w.id_audit if "adopted" in r]) == 2, \
        "nothing is left to adopt"


def test_nobody_is_built_where_his_WHOLE_BODY_does_not_fit():
    """`fill` clamps at the lattice edge, so a man put on a tall post in a
    36-high room was built as two legs with no pelvis (harvest seed 31, 84
    cells of 396). `_room_for` asks first, and the harvest places a man only
    where it says yes."""
    w = World(26, 18, 36, voxel_cm=5)
    w.fill(0, 26, 0, 18, 0, 1, STONE)
    assert _room_for(w, 10, 9, 1), "a clear floor has room"
    w.fill(10, 12, 9, 11, 1, 22, STONE)        # a post 1.05 m tall
    assert not _room_for(w, 10, 9, 22), "on top of it his head is past the roof"
    assert not _room_for(w, 6, 9, 1), "and his right arm would be in it"
    assert not _room_for(w, 1, 9, 1), "and he does not fit off the edge"
    for seed in (21, 31):
        hw, people, _lip = harvest.world(seed)
        for q in people:
            n = sum(len(v) for v in q["segs"].values())
            assert len(q["segs"]) == 10 and n == whole_body(), \
                f"seed {seed}: {q['name']} built as {n} cells"
        del hw


def whole_body():
    """How many cells `_person` builds a man of in open air (396)."""
    w = World(12, 10, 34, voxel_cm=5)
    p = _person(w, 4, 5)
    return sum(len(v) for v in p["segs"].values())
