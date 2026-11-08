"""THE FAST PATHS AND THE PLAIN ONES GIVE THE SAME ARRAYS (wave O, lane SIM1).

Each speedup here is EXACT: a compiled loop (numba) that must leave the same
numbers, to the bit, as the numpy code it replaces. The numpy code stays as
the plain path (`w.fused = False`) and is the definition of what the law
means. Golden says so for the scenes it plays; these tests say so on worlds
picked to reach the corners: the edge of the world, smoke, fire, glass, and
eyes that sit exactly half way between two cells.
"""
import numpy as np
import pytest

from src.voxel.kernels import _look_rays
from src.voxel.label import label
from src.voxel.materials import GLASS, SEE_TAU, SOLID, STONE, WOOD
from src.voxel.scenes import _person
from src.voxel.sim import HAVE_NUMBA, World

if not HAVE_NUMBA:
    pytest.skip("numba is optional; without it there is only the plain path",
                allow_module_level=True)


def _cells(w, p):
    comp, sl = w._person_cells(p)
    cells = np.argwhere(comp)
    cells[:, 0] += sl[0].start
    cells[:, 1] += sl[1].start
    return cells


def _belief(p):
    return {k: (None if p.get(k) is None else p[k].copy())
            for k in ("known", "danger", "free")}


# ── C5: the look ─────────────────────────────────────────────────────────────
def _look_both(w, p, eye, cells, gaze):
    """One look, taken by both paths from the same belief; their beliefs after."""
    out = []
    was = _belief(p)
    for fused in (True, False):
        q = dict(p)
        q.update({k: (None if v is None else v.copy()) for k, v in was.items()})
        q["gaze"] = gaze
        w.fused = fused
        w._look_around(q, eye, cells)
        out.append(_belief(q))
    w.fused = True
    return out


def _gazes(rng):
    fixed = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (0.6, 0.8),
             (-0.7071, 0.7071)]
    return fixed + [tuple(rng.normal(size=2)) for _ in range(4)]


def _check_look(w, rng, eyes=None):
    for p in w.persons:
        cells = _cells(w, p)
        eye0 = p.get("_eye") or (float(cells[:, 0].mean()),
                                 float(cells[:, 1].mean()),
                                 float(cells[:, 2].max()))
        for eye in [eye0] + list(eyes or ()):
            for gaze in _gazes(rng):
                fast, plain = _look_both(w, p, eye, cells, gaze)
                for k in ("known", "danger", "free"):
                    assert (fast[k] is None) == (plain[k] is None), k
                    if fast[k] is not None:
                        assert fast[k].dtype == plain[k].dtype, k
                        assert np.array_equal(fast[k], plain[k]), \
                            f"{k} differs at eye {eye}, gaze {gaze}"


def test_look_FAST_and_PLAIN_mark_the_same_cells_in_a_SMOKY_ROOM_on_FIRE():
    """A walled room with a burning bench, smoke under the ceiling, a glass
    pane and a doorway: rays stop on stone, dim in smoke, pass the glass, and
    the light band sees the fire under the smoke."""
    rng = np.random.default_rng(5)
    w = World(60, 34, 25, voxel_cm=10)
    w.fill(0, 60, 0, 34, 0, 1, STONE)
    w.fill(0, 60, 0, 1, 1, 24, STONE)
    w.fill(0, 60, 33, 34, 1, 24, STONE)
    w.fill(40, 41, 1, 33, 1, 24, STONE)
    w.mat[40, 15:19, 1:20] = 0                          # a doorway
    w.smass[40, 15:19, 1:20] = 0.0
    w.fill(40, 41, 24, 30, 5, 15, GLASS)                # and a window
    w.exits = []
    _person(w, 10, 15)
    _person(w, 50, 8)
    w.fill(30, 34, 14, 16, 1, 5, WOOD)
    w.E[30, 14, 1] = 9.0e5 * 8
    for _ in range(12):
        w.step()
        for p in w.persons:
            p["events"].clear()
    w.smoke[:, :, 12:] += rng.uniform(0.0, 0.4, w.smoke[:, :, 12:].shape
                                      ).astype(w.smoke.dtype) * w.vox_l
    assert w.burning().any(), "the bench burns"
    _check_look(w, rng, eyes=[(10.5, 15.5, 14.0), (20.5, 14.0, 9.0)])


def test_look_FAST_and_PLAIN_agree_at_the_EDGE_of_the_world():
    """An open floor, bodies near the edges: rays leave the world and are
    clipped to its last cells, which are still read and marked."""
    rng = np.random.default_rng(11)
    w = World(30, 26, 30, voxel_cm=5)
    w.fill(0, 30, 0, 26, 0, 1, STONE)
    w.exits = []
    _person(w, 3, 3)
    _person(w, 24, 21)
    w.step()
    _check_look(w, rng, eyes=[(0.5, 25.0, 20.0), (29.0, 0.5, 3.0),
                              (2.5, 2.5, 18.0)])


def test_look_FAST_and_PLAIN_agree_in_a_CROWD():
    """The CROWD test's world: three bodies in a closed hall, a few ticks in,
    so each one's flesh is in the others' way."""
    rng = np.random.default_rng(3)
    w = World(50, 26, 40, voxel_cm=5)
    w.open_sky = False
    w.fill(0, 50, 0, 26, 0, 40, STONE)
    w.mat[1:49, 1:25, 1:39] = 0
    w.smass[1:49, 1:25, 1:39] = 0.0
    w.exits = [(46, 12)]
    for x, y in ((22, 12), (26, 17), (34, 8)):
        _person(w, x, y)
    for _ in range(6):
        w.step()
        for p in w.persons:
            p["events"].clear()
    _check_look(w, rng)


def _rays_plain(ex, ey, cs, sn, R, nlog, nlog_band):
    """`_look_around`'s numpy rays, on given depth grids."""
    nx, ny = nlog.shape
    steps = np.arange(1, R + 1, dtype=np.float32)
    xi = np.clip(np.round(ex + cs[:, None] * steps[None]), 0, nx - 1).astype(np.int32)
    yi = np.clip(np.round(ey + sn[:, None] * steps[None]), 0, ny - 1).astype(np.int32)
    out = []
    for g in (nlog, nlog_band):
        tau = np.cumsum(g[xi, yi], axis=1)
        before = np.concatenate([np.zeros((len(cs), 1), np.float32),
                                 tau[:, :-1]], axis=1) < SEE_TAU
        m = np.zeros((nx, ny), bool)
        m[xi[before], yi[before]] = True
        out.append(m)
    return out


def test_the_RAY_KERNEL_matches_numpy_on_random_depths_even_NEGATIVE_ones():
    """The kernel alone, on random depth grids. Some grids hold small
    negative depths (a cell a hair over fully clear, which float32 rounding
    can make): then no ray may stop early, and the kernel is told so."""
    rng = np.random.default_rng(7)
    for trial in range(40):
        nx, ny = rng.integers(5, 40, size=2)
        nlog = rng.exponential(0.08, (nx, ny)).astype(np.float32)
        band = rng.exponential(0.05, (nx, ny)).astype(np.float32)
        if trial % 3 == 0:
            nlog[rng.random((nx, ny)) < 0.2] = np.float32(-1.2e-7)
            band[rng.random((nx, ny)) < 0.2] = np.float32(-0.0)
        if trial % 4 == 1:
            nlog[rng.random((nx, ny)) < 0.05] = np.float32(6.9)
        ang = np.linspace(0.0, 2 * np.pi, 97, endpoint=False)[rng.random(97) < 0.6]
        cs, sn = np.cos(ang), np.sin(ang)
        ex = float(rng.integers(0, nx)) + rng.choice([0.0, 0.5, 0.25])
        ey = float(rng.integers(0, ny)) + rng.choice([0.0, 0.5, -0.5])
        R = int(rng.integers(4, 60))
        want = _rays_plain(ex, ey, cs, sn, R, nlog, band)
        stop = bool(nlog.min() >= 0.0) and bool(band.min() >= 0.0)
        reach = np.zeros((nx, ny), bool)
        lit = np.zeros((nx, ny), bool)
        _look_rays(ex, ey, cs, sn, R, nlog, band, np.float32(SEE_TAU), stop,
                   reach, lit)
        assert np.array_equal(reach, want[0]), trial
        assert np.array_equal(lit, want[1]), trial


# ── C3: the labeller, a run at a time ───────────────────────────────────────
def _mats(seed, n=40):
    """Random lattices: blocks of three materials, air between, runs that
    start and stop anywhere up a column, and shapes one cell thin."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        shape = (1, 1, 1) if i == 0 else \
            tuple(int(v) for v in rng.integers(1 if i < 6 else 3, 22, 3))
        mat = np.zeros(shape, np.uint8)
        for _ in range(3 * max(shape)):
            lo = rng.integers(0, np.array(shape))
            sz = rng.integers(1, 7, 3)
            sl = tuple(slice(int(l), int(min(l + s, m)))
                       for l, s, m in zip(lo, sz, shape))
            mat[sl] = rng.integers(0, 4)
        mat[rng.random(shape) < 0.05] = rng.integers(0, 4)
        out.append(mat)
    return out


def test_RUN_LABELS_are_the_CELL_LABELS_array_for_array_key_min_and_max():
    """`label`'s run path against the cell-at-a-time union-find it replaces
    (itself held to the old floods by test_label.py): with and without a key,
    the largest and the smallest index as the name."""
    from src.voxel.label import _union_find
    for mat in _mats(4):
        mask = mat != 0
        for key, use_key in ((mask, False), (mat, True)):
            for canon in ("max", "min"):
                got = label(mask, mat if use_key else None, canon=canon)
                want = _union_find(mask, key, use_key, canon == "max", False)
                assert got.dtype == want.dtype
                assert np.array_equal(got, want), (mat.shape, use_key, canon)
                # and WRAP, which has no run path, still is the cell one
                got = label(mask, mat if use_key else None, canon=canon, wrap=True)
                want = _union_find(mask, key, use_key, canon == "max", True)
                assert np.array_equal(got, want), (mat.shape, use_key, canon)


def test_AIR_REGIONS_are_numbered_as_np_unique_numbers_them():
    """`regions`, both paths, against what `_air_regions` did: the label,
    then np.unique's inverse as the region number."""
    from src.voxel.label import regions
    masks = [m == 0 for m in _mats(9)] + [np.zeros((3, 4, 5), bool),
                                          np.ones((3, 4, 5), bool)]
    house = np.ones((100, 80, 50), bool)
    house[:, :, 0] = False
    house[10:90, 10:70, 0:45] = False
    house[11:89, 11:69, 1:44] = True
    house[10:11, 35:45, 1:38] = True
    for px, py in ((60, 30), (32, 58), (45, 20), (70, 50)):
        house[px - 2:px + 2, py - 2:py + 2, 1:31] = False
    for mask in masks + [house]:
        lab = label(mask, canon="max", fused=False)
        _, compact = np.unique(lab[mask], return_inverse=True)
        want = np.full(mask.shape, -1, np.int32)
        want[mask] = compact.astype(np.int32)
        n_want = int(compact.max()) + 1 if compact.size else 0
        for fused in (True, False):
            got, n = regions(mask, fused=fused)
            assert got.dtype == np.int32 and np.array_equal(got, want), fused
            assert n == n_want and type(n) is int, fused


# ── C2: the coarse air grid ─────────────────────────────────────────────────
def _grid(shape, fused):
    from src.voxel.air import AirMixin

    class G(AirMixin):
        pass
    g = G()
    g.shape, g.fused = shape, fused
    return g


def _same_bits(a, b):
    """Equal to the bit: the same dtype, the same numbers, the same zeros."""
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape \
        and np.array_equal(a, b) and np.array_equal(np.signbit(a), np.signbit(b))


def _field(rng, shape, dtype):
    a = (rng.standard_normal(shape) * 10.0 ** rng.uniform(-4, 4, shape)).astype(dtype)
    a[rng.random(shape) < 0.3] = 0.0
    a[rng.random(shape) < 0.1] = -0.0
    return a


GRID_SHAPES = [(100, 80, 50), (8, 8, 8), (9, 7, 6), (5, 3, 13), (3, 6, 9),
               (16, 4, 5), (4, 16, 17), (6, 6, 4), (7, 5, 3), (2, 2, 2),
               (13, 11, 1), (21, 2, 30), (1, 9, 12)]


def test_the_AIR_GRID_fast_and_plain_agree_TO_THE_BIT_padded_shapes_and_all():
    """`_cells` (mean and sum), `_cells_where`, `_blur_cells`, `_upcell` and
    `_upcell_where`, the compiled path against numpy, on random float32 and
    float64 fields with zeros of both signs, on worlds whose sides do and do
    not divide by 4 (the house is 50 high, and pads)."""
    rng = np.random.default_rng(17)
    for shape in GRID_SHAPES:
        fast, plain = _grid(shape, True), _grid(shape, False)
        for dtype in (np.float32, np.float64):
            for _ in range(3):
                a = _field(rng, shape, dtype)
                mask = rng.random(shape) < rng.uniform(0.1, 0.9)
                for how in ("mean", "sum"):
                    assert _same_bits(fast._cells(a, how), plain._cells(a, how)), \
                        (shape, dtype, how)
                assert _same_bits(fast._cells_where(mask, a),
                                  plain._cells(np.where(mask, a, 0.0))), (shape, dtype)
                assert _same_bits(fast._cells_where(mask),
                                  plain._cells(mask.astype(np.float32))), shape
                c = plain._cells(a)
                for passes in (0, 1, 2, 5):
                    assert _same_bits(fast._blur_cells(c, passes),
                                      plain._blur_cells(c, passes)), (shape, passes)
                assert _same_bits(fast._upcell(c), plain._upcell(c)), shape
                assert _same_bits(fast._upcell_where(c, mask),
                                  plain._upcell(c)[mask]), shape


# ── wave 11b (SLEEP): the wind carrying smoke ───────────────────────────────
GAS_SHAPES = [(9, 7, 6), (5, 3, 13), (16, 4, 5), (2, 2, 2), (13, 11, 1),
              (1, 9, 12), (40, 30, 20)]


def _gas_world(seed, shape, fused, open_sky):
    """A world of stone and wood blocks, some of them part full (so the pores
    differ), smoke of both signs and both zeros everywhere, and a coarse wind
    and pressure field already blowing."""
    rng = np.random.default_rng(seed)
    w = World(*shape)
    w.fused, w.open_sky = fused, open_sky
    for m, share in ((STONE, 0.2), (WOOD, 0.1)):
        cells = (rng.random(shape) < share) & (w.mat == 0)
        w.mat[cells] = m
        w.smass[cells] = (SOLID[m][0] * w.vox_l
                          * rng.choice([1.0, 0.99, 0.5, 0.03], size=int(cells.sum())))
    w.smoke = np.abs(_field(rng, shape, np.float32)) * 1e-3
    w.smoke[rng.random(shape) < 0.05] = -0.0
    cshape = tuple((n + w._CS - 1) // w._CS for n in shape)
    w.pcell = _field(rng, cshape, np.float32)
    w.vcell = [_field(rng, cshape, np.float32) for _ in range(3)]
    w.E = np.where(rng.random(shape) < 0.3,
                   rng.uniform(0, 300, shape), 0.0).astype(np.float32)
    return w


def test_the_WIND_carries_smoke_the_same_FAST_and_PLAIN():
    """`_law_air`, compiled (`_carry_gas`) against numpy, to the bit: the
    smoke it carries and the pressure and wind it leaves."""
    for k, shape in enumerate(GAS_SHAPES):
        for open_sky in (True, False):
            fast = _gas_world(k, shape, True, open_sky)
            plain = _gas_world(k, shape, False, open_sky)
            for _ in range(3):
                fast._law_air()
                plain._law_air()
                assert _same_bits(fast.smoke, plain.smoke), (shape, open_sky)
                assert _same_bits(fast.pcell, plain.pcell), (shape, open_sky)
                for a, b in zip(fast.vcell, plain.vcell):
                    assert _same_bits(a, b), (shape, open_sky)


def test_ANY_IN_is_numpys_table_lookup():
    """`_any_in` (the melt law's "is anything here that melts or sets?")
    against `table[a].any()`, on codes with no hit, one hit at either end,
    and many."""
    from src.voxel.kernels import _any_in
    rng = np.random.default_rng(29)
    table = rng.random(12) < 0.3
    table[0] = False
    for n in (0, 1, 7, 1000):
        for _ in range(20):
            a = rng.integers(0, 12, n).astype(np.uint8)
            a[rng.random(n) < rng.uniform(0, 1)] = 0
            assert _any_in(a, table) == bool(table[a].any())
    miss = np.zeros(50, np.uint8)
    hit = int(np.flatnonzero(table)[0])
    for at in (0, 49):
        a = miss.copy()
        a[at] = hit
        assert _any_in(a, table) and not _any_in(miss, table)


# ── C1: the support slack ───────────────────────────────────────────────────
def _watch_slack(w, seen):
    """Make every `_relax_slack` of this world run BOTH paths on the same
    input and compare them: the slack and what is left of `rubble`."""
    def both(solid, grip=frozenset()):
        rub = w.rubble.copy()
        w.fused = False
        plain = World._relax_slack(w, solid, grip)
        rub_plain = w.rubble.copy()
        np.copyto(w.rubble, rub)
        w.fused = True
        fast = World._relax_slack(w, solid, grip)
        assert fast.dtype == plain.dtype and np.array_equal(fast, plain), \
            f"slack differs at tick {w.tick}"
        assert np.array_equal(w.rubble, rub_plain)
        seen.append(bool(grip))
        return fast
    w._relax_slack = both


def _step(w, n):
    for _ in range(n):
        w.step()
        for p in w.persons:
            p["events"].clear()


def test_SLACK_fast_and_plain_agree_on_RANDOM_lattices_with_RUBBLE_and_GRIPS():
    """Random blocks of every material, part-full cells (a thinned beam has
    less span), cells flagged as broken pieces, and grips seeded anywhere,
    on solid or not: the fixpoint the queue reaches is numpy's, cell for cell."""
    from src.voxel.materials import ASH, FLESH, IRON
    rng = np.random.default_rng(23)
    for trial in range(25):
        shape = tuple(int(v) for v in rng.integers(3, 26, 3))
        w = World(*shape, voxel_cm=int(rng.choice([5, 10])))
        for _ in range(4 * max(shape)):
            lo = rng.integers(0, np.array(shape))
            sz = rng.integers(1, 8, 3)
            hi = np.minimum(lo + sz, shape)
            w.fill(int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1]),
                   int(lo[2]), int(hi[2]),
                   int(rng.choice([0, STONE, WOOD, IRON, GLASS, ASH, FLESH])),
                   frac=float(rng.choice([1.0, 0.6, 0.3, 0.1])))
        w.rubble = (rng.random(shape) < 0.15) & (rng.random(shape) < 0.9)
        solid = w.mat != 0
        cells = np.argwhere(rng.random(shape) < 0.02)
        for grip in (frozenset(), frozenset(map(tuple, cells))):
            rub = w.rubble.copy()
            w.fused = False
            plain = w._relax_slack(solid, grip)
            rub_plain = w.rubble.copy()
            w.rubble = rub
            w.fused = True
            fast = w._relax_slack(solid, grip)
            assert fast.dtype == plain.dtype and np.array_equal(fast, plain), trial
            assert np.array_equal(w.rubble, rub_plain), trial
            assert (fast >= 0).any() or not solid[:, :, 0].any()


def test_SLACK_fast_and_plain_agree_in_a_HOUSE_a_CROWD_a_HARVEST_and_a_GRIP():
    """Real worlds, stepped, both paths run on every call: a house of people
    walking (the case that misses the cache most), the crowd, a harvest world,
    and a man held over a drop by a grip."""
    from src.voxel import harvest
    from src.voxel.scenes import _Wants
    seen = []
    w = World(100, 80, 50, voxel_cm=5)                 # the house
    w.fill(0, 100, 0, 80, 0, 1, STONE)
    w.fill(10, 90, 10, 70, 0, 45, STONE)
    w.mat[11:89, 11:69, 1:44] = 0
    w.smass[11:89, 11:69, 1:44] = 0.0
    w.mat[10:11, 35:45, 1:38] = 0
    w.smass[10:11, 35:45, 1:38] = 0.0
    w.exits = [(13, 40)]
    for px, py in ((60, 30), (32, 58)):
        _person(w, px, py)
    _watch_slack(w, seen)
    _step(w, 4)
    counts = [len(seen)]
    del w
    w = World(50, 26, 40, voxel_cm=5)                  # the crowd
    w.open_sky = False
    w.fill(0, 50, 0, 26, 0, 40, STONE)
    w.mat[1:49, 1:25, 1:39] = 0
    w.smass[1:49, 1:25, 1:39] = 0.0
    w.exits = [(46, 12)]
    for x, y in ((22, 12), (26, 17), (34, 8)):
        _person(w, x, y)
    _watch_slack(w, seen)
    _step(w, 12)
    counts.append(len(seen))
    del w
    w, people, _ = harvest.world(3)                    # a harvest world
    w.policy = harvest.Roam(3)
    _watch_slack(w, seen)
    _step(w, 25)
    counts.append(len(seen))
    del w
    w = World(40, 40, 46, voxel_cm=5)                  # held over a drop
    w.fill(0, 40, 0, 40, 0, 1, STONE)
    w.fill(16, 40, 0, 40, 1, 11, STONE)
    w.exits = []
    a = _person(w, 20, 20, z0=11)
    a["name"], a["strength_N"] = "the holder", 600.0
    b = _person(w, 28, 20, z0=11)
    b["name"] = "the mark"
    w.policy = _Wants(each={"the holder": {"hands": "take hold of the mark"}})
    _watch_slack(w, seen)
    _step(w, 20)
    w.fill(26, 34, 0, 40, 1, 11, 0)
    _step(w, 10)
    counts.append(len(seen))
    assert all(b > a for a, b in zip([0] + counts, counts)), \
        f"every world must reach the relax at least once ({counts})"
    assert any(seen), "and some calls must seed a grip"


# ── C4: a body's cells, found once a tick ───────────────────────────────────
def _underfoot_old(w, cells):
    """`_underfoot` as it was: a Python loop over the cells."""
    from src.voxel.materials import AIR
    zlo = int(cells[:, 2].min())
    if zlo <= 0:
        return True
    return any(w.mat[int(c[0]), int(c[1]), zlo - 1] != AIR
               for c in cells if int(c[2]) == zlo)


def _borne_old(w):
    out = set()
    for p in w.persons:
        comp, sl = w._person_cells(p)
        if comp is None or not comp.any():
            continue
        c = np.argwhere(comp)
        c[:, 0] += sl[0].start or 0
        c[:, 1] += sl[1].start or 0
        if _underfoot_old(w, c):
            out.update(map(tuple, c))
    return frozenset(out)


def test_a_BODYS_CELLS_from_the_claim_are_what_ARGWHERE_gives_and_stay_right():
    """The cells a claim found are handed to every law that asks, in place of
    another np.argwhere over the whole lattice; the borne set reuses the
    claim's own frozenset; `_owners` is kept while the deeds are the same
    objects. Each against the old way, in a crowd that walks, after a deed
    is moved by hand (a mover), and after a claim mask is taken away."""
    w = World(50, 26, 40, voxel_cm=5)
    w.open_sky = False
    w.fill(0, 50, 0, 26, 0, 40, STONE)
    w.mat[1:49, 1:25, 1:39] = 0
    w.smass[1:49, 1:25, 1:39] = 0.0
    w.exits = [(46, 12)]
    for x, y in ((22, 12), (26, 17), (34, 8)):
        _person(w, x, y)
    for t in range(14):
        _step(w, 1)
        for p in w.persons:
            comp, sl = w._person_cells(p)
            if comp is None:
                continue
            want = np.argwhere(comp)
            got = w._claim_cells(p, comp, sl)
            assert got.dtype == want.dtype and np.array_equal(got, want)
            got[:] = -7                          # a caller may change its copy
            assert np.array_equal(w._claim_cells(p, comp, sl, shift=False), want)
            assert p["_own"] == frozenset(map(tuple, want))
            assert list(p["_own"]) == list(frozenset(map(tuple, want))), \
                "the deed iterates in the same order as it always did"
            assert all(type(v) is np.int64 for c in list(p["_own"])[:5] for v in c)
            for cells in (want, want.astype(np.float64) + 0.4):
                assert w._underfoot(cells) == _underfoot_old(w, cells)
        borne = w._borne_cells()
        assert borne == _borne_old(w)
        assert all(type(v) is np.int64 for c in list(borne)[:5] for v in c)
        k, p_ = w._owners()
        k2, p2 = w._owners_of()
        assert np.array_equal(k, k2) and np.array_equal(p_, p2), t
        if t == 6:                               # a mover rewrites one deed
            w._carry_own(w.persons[1], 1, 0)
            k, p_ = w._owners()
            k2, p2 = w._owners_of()
            assert np.array_equal(k, k2) and np.array_equal(p_, p2)
    p = w.persons[0]
    comp, sl = w._person_cells(p)
    fake = comp.copy()                           # a mask the claim never made
    fake[tuple(np.argwhere(comp)[0])] = False
    assert np.array_equal(w._claim_cells(p, fake, sl), np.argwhere(fake))
