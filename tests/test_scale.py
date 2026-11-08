"""THE LAWS WORK WHERE THE MATTER IS, AND IT CHANGES NOTHING (wave 12, lane
SCALE).

One idle person in a yard of 1.6 million voxels cost 77 ms a tick, and about
a thirtieth of that world is anything but air. The torque law, the support
law, the temperature and fire fields, the airspace labels and the reach of a
hand now work from a list of the cells that are matter (or a box round the
heat), not from whole-lattice arrays. Every one of these is EXACT, and these
tests say so: the same worlds are run with `w.windows` on and off and must
agree to the bit, tick by tick, in every field a law may write; and each
piece is checked alone against the whole-lattice code it stands in for.
"""
import numpy as np
import pytest

from src.voxel import label as labelmod
from src.voxel.hands import _reach_distance
from src.voxel.kernels import HAVE_NUMBA, _relax_nb
from src.voxel.label import label, regions
from src.voxel.materials import (AIR, AMBIENT, IRON, OIL, STONE, WOOD)
from src.voxel.scenes import _person, thing
from src.voxel.sim import World
from src.voxel.window import _piece_bounds, _relax_list, compact_labels

pytestmark = pytest.mark.skipif(not HAVE_NUMBA,
                                reason="the windows are compiled lists")

_FIELDS = ("mat", "smass", "fl", "fvol", "fpot", "E", "smoke", "o2", "skin",
           "edge", "rubble", "eid", "p_add", "fallh", "moving")


def _bits(w):
    """Every field a law may write, as raw bytes, and the world's books."""
    out = {k: np.ascontiguousarray(getattr(w, k)).tobytes()
           for k in _FIELDS if getattr(w, k, None) is not None}
    out["shed"] = repr(w.shed)
    out["hrr"] = repr(w.hrr)
    out["bodies"] = len(w.bodies)
    out["air_region"] = (None if w.air_region is None
                         else w.air_region.tobytes())
    out["people"] = repr([(p["name"], p.get("anchor"), p.get("blood_ml"))
                          for p in w.persons])
    return out


def _lockstep(build, ticks, poke=None, fused=True):
    """Two copies of one world, windows on and windows off, stepped side by
    side. Returns the first (tick, field) where their bits part, or None; and
    how many topples the windowed copy had (the tip law working)."""
    on, off = build(), build()
    off.restore(on.snapshot())
    on.windows, off.windows = True, False
    on.fused = off.fused = fused
    tips = []
    real = on._topple
    on._topple = lambda *a, **k: (tips.append(len(a[0])), real(*a, **k))[1]
    tipped = 0
    for t in range(ticks):
        for w in (on, off):
            if poke is not None:
                poke(w, t)
            w.step()
        tipped = len(tips)
        a, b = _bits(on), _bits(off)
        assert a.keys() == b.keys(), f"t{t}: different fields exist"
        for k in a:
            if a[k] != b[k]:
                return (t, k), tipped
    return None, tipped


def _room(nx=44, ny=36, nz=26):
    w = World(nx, ny, nz, voxel_cm=5)
    w.fill(0, nx, 0, ny, 0, nz, STONE)
    w.mat[1:nx - 1, 1:ny - 1, 1:nz - 1] = AIR
    w.smass[1:nx - 1, 1:ny - 1, 1:nz - 1] = 0.0
    w.mat[0:1, 14:20, 1:20] = AIR                         # a doorway
    w.smass[0:1, 14:20, 1:20] = 0.0
    return w


# ── the pieces, one at a time ───────────────────────────────────────────────
def test_the_PIECES_of_a_LIST_of_cells_are_the_PIECES_of_the_LATTICE():
    """Numbered by the largest cell, with the piece below each cell, as the
    numpy flood and `np.unique` number them. Keys that split touching cells,
    one-cell-thick worlds and a column as tall as the world are all here."""
    rng = np.random.default_rng(11)
    for shape in ((9, 7, 5), (1, 6, 4), (12, 3, 1), (5, 5, 5), (2, 2, 40)):
        for dens in (0.3, 0.55, 0.8):
            mask = rng.random(shape) < dens
            key = rng.integers(0, 3, shape).astype(np.int8)
            idx = np.flatnonzero(mask)
            if not len(idx):
                continue
            comp, below, n = compact_labels(idx, key.reshape(-1)[idx], shape)
            lab = label(mask, key, canon="max", fused=False)
            _, want = np.unique(lab[mask], return_inverse=True)
            assert np.array_equal(comp, want)
            assert n == int(want.max()) + 1
            whole = np.full(mask.size, -1)
            whole[idx] = want
            z = idx % shape[2]
            assert np.array_equal(below, np.where(z > 0, whole[idx - 1], -1))


def test_the_BOUNDS_of_each_PIECE_are_the_UFUNC_AT_ones():
    rng = np.random.default_rng(12)
    for shape in ((9, 7, 5), (6, 6, 12), (3, 8, 2)):
        mask = rng.random(shape) < 0.6
        idx = np.flatnonzero(mask)
        ny, nz = shape[1], shape[2]
        cx, cy, cz = idx // (ny * nz), (idx // nz) % ny, idx % nz
        key = rng.integers(0, 2, shape).astype(np.int8)
        comp, below, n = compact_labels(idx, key.reshape(-1)[idx], shape)
        got = _piece_bounds(comp, cx, cy, cz, below, n, nz)
        big = 10 ** 6
        bx0 = np.full((n, nz), big); bx1 = np.full((n, nz), -big)
        by0 = np.full((n, nz), big); by1 = np.full((n, nz), -big)
        np.minimum.at(bx0, (comp, cz), cx); np.maximum.at(bx1, (comp, cz), cx)
        np.minimum.at(by0, (comp, cz), cy); np.maximum.at(by1, (comp, cz), cy)
        contact = (cz == 0) | ((below >= 0) & (below != comp))
        cL = comp[contact]
        fx0 = np.full(n, big); fx1 = np.full(n, -big)
        fy0 = np.full(n, big); fy1 = np.full(n, -big)
        fz = np.full(n, big)
        np.minimum.at(fx0, cL, cx[contact]); np.maximum.at(fx1, cL, cx[contact])
        np.minimum.at(fy0, cL, cy[contact]); np.maximum.at(fy1, cL, cy[contact])
        np.minimum.at(fz, cL, cz[contact])
        sup_lo = np.full(n, 10 ** 9); sup_hi = np.full(n, -1)
        cnz = contact & (cz > 0)
        np.minimum.at(sup_lo, comp[cnz], below[cnz])
        np.maximum.at(sup_hi, comp[cnz], below[cnz])
        has_ground = np.zeros(n, bool)
        cg = contact & (cz == 0)
        if cg.any():
            has_ground[np.unique(comp[cg])] = True
        want = (bx0, bx1, by0, by1, fx0, fx1, fy0, fy1, fz, sup_lo, sup_hi,
                has_ground)
        for g, w_ in zip(got, want):
            assert g.dtype == w_.dtype and np.array_equal(g, w_)


def test_the_SUPPORT_RELAXED_from_a_LIST_is_the_SUPPORT_RELAXED_over_the_LATTICE():
    rng = np.random.default_rng(13)
    for shape in ((10, 9, 8), (5, 5, 20), (16, 3, 3)):
        solid = rng.random(shape) < 0.45
        solid[:, :, 0] |= rng.random(shape[:2]) < 0.5
        span = np.where(solid, rng.integers(0, 6, shape), 0).astype(np.int16)
        rubble = solid & (rng.random(shape) < 0.15)
        slack = np.full(shape, -1, np.int16)
        slack[:, :, 0][solid[:, :, 0]] = span[:, :, 0][solid[:, :, 0]]
        a = _relax_nb(solid, span, rubble, slack.copy())
        b = _relax_list(np.flatnonzero(solid), solid, span, rubble, slack.copy())
        assert a.dtype == b.dtype and np.array_equal(a, b)


def test_a_WINDOWED_T_and_BURNING_are_the_WHOLE_ONES_in_a_cold_a_hot_and_a_skinned_world():
    for skin in (False, True):
        w = _room()
        w.fill(10, 16, 10, 16, 1, 3, WOOD, frac=0.7)
        w.pour(20, 20, 1, OIL, 100.0)
        w.skip_quiet = True
        w.heat_capacity()
        C = w.heat_capacity()
        for state in ("cold", "warm", "hot", "skin only", "everywhere"):
            w.E[...] = 0.0
            if skin:
                w.skin = np.zeros(w.shape, np.float32)
            if state == "warm":
                w.E[10:14, 10:14, 1:3] = 30.0
            if state == "hot":
                w.E[10:14, 10:14, 1:3] = (450.0 - AMBIENT) * C[10:14, 10:14, 1:3]
                w.E[20, 20, 1] = (300.0 - AMBIENT) * C[20, 20, 1]
            if state == "skin only" and skin:
                w.skin[12:16, 12:16, 2] = 400.0
            if state == "everywhere":
                w.E[...] = 5.0 * C
                w.E[11:13, 11:13, 1:3] = (400.0 - AMBIENT) * C[11:13, 11:13, 1:3]
            w.windows = True
            t_on, b_on = w.T(), w.burning()
            w.windows = False
            t_off, b_off = w.T(), w.burning()
            assert t_on.dtype == t_off.dtype
            assert t_on.tobytes() == t_off.tobytes(), (skin, state)
            assert b_on.tobytes() == b_off.tobytes(), (skin, state)
            if state == "hot":
                assert b_on.any(), "the hot case must have fire in it"
        del w.windows
        w.skin = None


def test_the_KEPT_COPIES_of_the_derived_fields_follow_every_change_to_the_bit():
    """`_matter_moved` brings its kept copies up to date cell by cell, in
    place. After any run of changes they are the fields, and whoever holds an
    older heat capacity still holds the older state."""
    rng = np.random.default_rng(14)
    w = _room()
    w.fill(10, 16, 10, 16, 1, 3, WOOD, frac=0.7)
    held = w.heat_capacity()
    was = held.copy()
    for step in range(6):
        for _ in range(int(rng.integers(1, 30))):
            x, y, z = (int(rng.integers(2, s - 2)) for s in w.shape)
            w.smass[x, y, z] = float(rng.random()) * 20.0
            w.fvol[x, y, z] = float(rng.random()) * 5.0
            w.mat[x, y, z] = int(rng.choice([AIR, WOOD, STONE, IRON]))
        cap = w.heat_capacity()
        k = w._cap_key
        for kept, now in zip(k, (w.mat, w.smass, w.fl, w.fvol)):
            assert kept.tobytes() == now.tobytes()
        want = w._cap_of(w.mat, w.smass, w.fl, w.fvol)
        assert cap.tobytes() == want.tobytes()
        assert held.tobytes() == was.tobytes(), "an older array is never written"
        held, was = cap, cap.copy()


def test_the_JOIN_KEY_of_a_LIST_of_cells_is_the_KEY_of_the_LATTICE_there():
    """A named thing is one lump whatever it is made of: its cells key on the
    eid, every other cell on its material."""
    w = _walkers_and_a_table()
    assert w._thing_lut.any(), "the world must have a named thing in it"
    idx = np.flatnonzero(w.mat != AIR)
    assert np.array_equal(w._key_at(idx), w._key().reshape(-1)[idx])
    assert not np.array_equal(w._key_at(idx), w.mat.reshape(-1)[idx])
    plain = _room()
    plain.fill(10, 16, 10, 16, 1, 3, WOOD)
    idx = np.flatnonzero(plain.mat != AIR)
    assert np.array_equal(plain._key_at(idx), plain.mat.reshape(-1)[idx])


def test_the_REACH_of_a_BODY_is_the_PAIR_BY_PAIR_DISTANCE():
    rng = np.random.default_rng(15)
    for _ in range(12):
        shape = (int(rng.integers(8, 22)), int(rng.integers(8, 22)),
                 int(rng.integers(8, 30)))
        origin = (int(rng.integers(0, 5)),) * 3
        R = int(rng.integers(1, 7))
        n = int(rng.integers(1, 40))
        cells = np.array([[int(rng.integers(0, shape[a])) + origin[a]
                           for a in range(3)] for _ in range(n)])
        rel = np.argwhere(rng.random(shape) < 0.3)
        got = _reach_distance(rel, cells, origin, shape, R)
        want = np.abs((rel + np.array(origin))[:, None, :] - cells[None, :, :]
                      ).max(axis=2).min(axis=1)
        keep = want <= R
        assert np.array_equal(got[keep], want[keep])
        assert (got[~keep] > R).all()


def test_the_KEPT_AIR_LABELS_are_the_LABELS_from_scratch(monkeypatch):
    """The airspaces are patched into the last labelling when a few cells
    change and nothing joins, splits, appears or loses its largest cell, and
    labelled again otherwise. Either way: the array a fresh labelling gives,
    and a copy nobody else holds."""
    monkeypatch.setattr(labelmod, "_INCR_MIN", 0)
    monkeypatch.setattr(labelmod, "_KEPT_REGIONS", {})
    seen = []
    real = labelmod._patch_regions

    def spy(*a):
        r = real(*a)
        seen.append(int(r))
        return r

    monkeypatch.setattr(labelmod, "_patch_regions", spy)
    rng = np.random.default_rng(16)
    shape = (14, 12, 10)
    mask = rng.random(shape) < 0.75
    held = []
    for step in range(60):
        mask = mask.copy()
        for _ in range(int(rng.integers(1, 6))):
            i = tuple(int(rng.integers(0, s)) for s in shape)
            mask[i] = not mask[i]
        if step % 20 == 19:                              # and a wall at once
            mask[:, int(rng.integers(3, 9)), :] = False
        got, n = regions(mask)
        want, m = regions(mask, fused=False)
        assert n == m and got.dtype == want.dtype and np.array_equal(got, want)
        held.append((got, want))
    assert 0 in seen and 1 in seen, "both the patch and the full labelling ran"
    assert all(np.array_equal(g, w_) for g, w_ in held), \
        "a copy handed out is never written by a later call"


# ── whole worlds: windows on and off ─────────────────────────────────────────
def _walkers_and_a_table():
    """A room with a person walking about in it and a table whose two
    east legs go at tick 6: the torque law must find the table tipping, the
    support law must carry what is on it, and people moving every tick keep
    the airspace labels and the derived fields busy."""
    w = _room()
    w.fill(26, 36, 20, 26, 8, 9, WOOD, frac=0.8)                # the top
    for (lx, ly) in ((26, 20), (26, 24), (35, 20), (35, 24)):
        w.fill(lx, lx + 1, ly, ly + 2, 1, 8, WOOD)               # four legs
    w.fill(28, 31, 21, 24, 9, 12, IRON)                          # a load on it
    w.fill(5, 8, 28, 31, 1, 7, WOOD, frac=0.5)                   # a stack
    with thing(w, "axe", kind="tool", parts={"head": IRON, "haft": WOOD}):
        w.fill(14, 20, 30, 31, 1, 2, WOOD)                       # a named axe
        w.fill(20, 22, 30, 31, 1, 2, IRON, edge=2e-4)            # iron and wood
    _person(w, 10, 10)
    return w


def _legs_go(w, t):
    if t == 6:
        for (lx, ly) in ((35, 20), (35, 24)):
            w.mat[lx, ly:ly + 2, 1:8] = AIR
            w.smass[lx, ly:ly + 2, 1:8] = 0.0


def test_a_ROOM_of_WALKERS_and_a_TIPPING_TABLE_gives_the_SAME_BITS_with_windows_on_and_off():
    got, tipped = _lockstep(_walkers_and_a_table, 40, _legs_go)
    assert got is None, f"windows on and off part at tick {got[0]}, in {got[1]}"
    assert tipped, "the table must have gone over, or the tip law was not run"


def _burning_room():
    """A bed alight over a spilled lamp with a person in the room: the lit
    cells, the fire books, the temperature field and the burning cells all
    work every tick."""
    w = _room()
    w.fill(28, 38, 24, 30, 4, 6, WOOD, frac=0.7)
    w.fill(28, 38, 24, 30, 12, 13, WOOD, frac=0.8)
    for (x, y) in ((27, 24), (27, 25), (26, 24)):
        w.pour(x, y, 1, OIL, 100.0)
    _person(w, 10, 10)
    C = w.heat_capacity()
    w.fill(4, 10, 4, 9, 1, 3, WOOD, frac=0.7)            # a second fire
    C = w.heat_capacity()
    bed = np.zeros(w.shape, bool)
    bed[28:38, 24:30, 4:6] = True                        # more lit cells than
    bed[4:10, 4:9, 1:3] = True                           # the band needs
    bed &= w.mat == WOOD
    w.E[bed] = (450.0 - AMBIENT) * C[bed]
    pool = w.fl == OIL
    w.E[pool] = (270.0 - AMBIENT) * C[pool]
    return w


def test_a_BURNING_ROOM_with_PEOPLE_gives_the_SAME_BITS_with_windows_on_and_off():
    w = _burning_room()
    assert int(w.burning().sum()) > 64, "enough lit cells to be read by band"
    got, _ = _lockstep(_burning_room, 50)
    assert got is None, f"windows on and off part at tick {got[0]}, in {got[1]}"


def test_the_PLAIN_numpy_PATH_with_windows_on_is_the_PLAIN_numpy_PATH_with_them_off():
    """No compiled kernels at all: the cells listed for the span and the
    box round the heat are still the whole-lattice sums, bit for bit."""
    got, _ = _lockstep(_burning_room, 20, fused=False)
    assert got is None, f"windows on and off part at tick {got[0]}, in {got[1]}"


def test_a_KEPT_PIECE_that_GROWS_past_ANOTHER_ONES_largest_cell_is_NUMBERED_AGAIN(monkeypatch):
    """The pieces are numbered by their largest cell, so patching a cell into
    a piece is only safe while that cell is not larger than the piece's
    largest. Two rooms, the first with the lower largest cell; its wall is
    opened toward the far side of the second's, so the first outranks it."""
    monkeypatch.setattr(labelmod, "_INCR_MIN", 0)
    monkeypatch.setattr(labelmod, "_KEPT_REGIONS", {})
    seen = []
    real = labelmod._patch_regions
    monkeypatch.setattr(labelmod, "_patch_regions",
                        lambda *a: (seen.append(int(real(*a))), seen[-1])[1])
    mask = np.ones((10, 8, 6), bool)
    mask[7, :, :] = False                         # a wall
    mask[9, :, :] = False                         # and another
    mask[8, :, :] = False
    mask[8, 3, 2] = True                          # a one-cell room beyond it
    first, n = regions(mask)
    assert n == 2 and first[0, 0, 0] == 0 and first[8, 3, 2] == 1
    opened = mask.copy()
    opened[7, 7, 5] = True                        # the first room grows into the wall
    opened[8, 7, 5] = True                        # and past the one-cell room's cell
    got, n = regions(opened)
    want, m = regions(opened, fused=False)
    assert n == m == 2 and np.array_equal(got, want)
    assert got[8, 3, 2] == 0 and got[0, 0, 0] == 1, "the far room is now the lower"
    assert seen[-1] == 1, "that change could not be patched in"


def test_a_BIG_yard_with_one_idle_PERSON_gives_the_SAME_BITS_with_the_airspace_labels_KEPT(monkeypatch):
    """The kept labels in a real run: a person walking about in open air,
    labelled by patching, against labelled from scratch every tick."""
    monkeypatch.setattr(labelmod, "_INCR_MIN", 0)
    monkeypatch.setattr(labelmod, "_KEPT_REGIONS", {})
    seen = []
    real = labelmod._patch_regions

    def spy(*a):
        seen.append(int(real(*a)))
        return seen[-1]

    monkeypatch.setattr(labelmod, "_patch_regions", spy)

    def yard():
        w = World(60, 48, 40, voxel_cm=5)
        w.fill(0, 60, 0, 48, 0, 1, STONE)
        _person(w, 30, 24)
        return w

    got, _ = _lockstep(yard, 40)
    assert got is None, f"windows on and off part at tick {got[0]}, in {got[1]}"
    assert 0 in seen, "a walking person is patched into the labels"


def test_a_BODY_in_REACH_of_THINGS_lists_the_SAME_THINGS_with_windows_on_and_off():
    w = _room()
    w.fill(11, 14, 10, 12, 1, 3, WOOD)
    w.fill(13, 15, 15, 17, 1, 2, IRON)
    w.fill(10, 12, 6, 8, 1, 6, STONE)
    p = _person(w, 12, 13)
    w.step()
    comp, sl = w._person_cells(p)
    own = np.argwhere(comp)
    own[:, 0] += sl[0].start
    own[:, 1] += sl[1].start
    w.windows = True
    on = w._objects_within_reach(p, own)
    w.windows = False
    off = w._objects_within_reach(p, own)
    assert on == off and on, "some things are in reach, and the same ones"
