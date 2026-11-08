"""THE LAWS WORK ONLY WHERE SOMETHING CAN CHANGE, AND IT CHANGES NOTHING
(wave 11b, lane SLEEP).

`skip_quiet` passes over a whole law when the thing it moves is absent
everywhere. `sleep_regions` goes one step further: a law that is busy
somewhere does its work only where it can change something (the columns a
flame stands in, the cells that hold a liquid, the cells whose matter moved).
Every such skip is EXACT: these tests run the same world with the regions on
and off and ask for the same bits, tick by tick, in every field a law may
touch.
"""
import numpy as np
import pytest

from src.voxel.heat import _count_nb, _count_open_at
from src.voxel.kernels import HAVE_NUMBA, _diff4
from src.voxel.materials import AIR, AMBIENT, CHAR, OIL, STONE, WATER, WOOD
from src.voxel.sim import World

_FIELDS = ("mat", "smass", "fl", "fvol", "fpot", "E", "smoke", "o2", "skin",
           "edge", "rubble", "eid", "p_add")


def _bits(w):
    """Every field a law may write, as raw bytes, and the world's books."""
    out = {k: np.ascontiguousarray(getattr(w, k)).tobytes()
           for k in _FIELDS if getattr(w, k, None) is not None}
    out["shed"] = repr(w.shed)
    out["hrr"] = repr(w.hrr)
    out["gen"] = w._matter_gen
    if w.pcell is not None:
        out["pcell"] = w.pcell.tobytes()
        out["vcell"] = b"".join(v.tobytes() for v in w.vcell)
    return out


def _lockstep(build, ticks, poke=None):
    """Two copies of one world, regions on and regions off, stepped side by
    side. Returns the first (tick, field) where their bits part, or None."""
    on, off = build(), build()
    off.restore(on.snapshot())
    on.sleep_regions, off.sleep_regions = True, False
    for t in range(ticks):
        for w in (on, off):
            if poke is not None:
                poke(w, t)
            w.step()
        a, b = _bits(on), _bits(off)
        assert a.keys() == b.keys(), f"t{t}: different fields exist"
        for k in a:
            if a[k] != b[k]:
                return t, k
    return None


def _corner_fire():
    """A shut room, 2 x 1.6 x 1.4 m at 5 cm, with a doorway: a bed burning in
    the far corner over a spilled lamp, a shelf above it that the flame
    presses against, a pool of water in the near corner, and a wooden
    bench nobody has lit."""
    w = World(40, 32, 28, voxel_cm=5)
    w.fill(0, 40, 0, 32, 0, 28, STONE)
    w.mat[1:39, 1:31, 1:27] = AIR
    w.smass[1:39, 1:31, 1:27] = 0.0
    w.mat[0:1, 12:18, 1:20] = AIR                         # a doorway
    w.smass[0:1, 12:18, 1:20] = 0.0
    w.fill(28, 38, 22, 30, 4, 6, WOOD, frac=0.7)          # the bed
    w.fill(28, 38, 22, 30, 12, 13, WOOD, frac=0.8)        # a shelf above it
    w.fill(4, 12, 4, 8, 1, 6, WOOD, frac=0.6)             # a bench, cold
    for (x, y) in ((27, 22), (27, 23), (26, 22)):
        w.pour(x, y, 1, OIL, 100.0)
    for x in range(3, 9):                                 # a pool, still settling
        for y in range(20, 26):
            w.pour(x, y, 1, WATER, 900.0)
    for y in range(20, 26):
        w.pour(3, y, 2, WATER, 700.0)
    C = w.heat_capacity()
    bed = np.zeros(w.shape, bool)
    bed[28:32, 22:26, 4:6] = True
    bed &= w.mat == WOOD
    w.E[bed] = (450.0 - AMBIENT) * C[bed]
    pool = w.fl == OIL
    w.E[pool] = (270.0 - AMBIENT) * C[pool]
    return w


@pytest.mark.skipif(not HAVE_NUMBA, reason="the derived fields' skip is compiled")
def test_a_SMALL_FIRE_in_a_CORNER_gives_the_SAME_BITS_with_regions_on_and_off():
    """200 ticks of a corner fire: flame, smoke, a settling pool, burning
    wood. Not one bit of any field may differ, on any tick."""
    got = _lockstep(_corner_fire, 200)
    assert got is None, f"regions on and off part at tick {got[0]}, in {got[1]}"


@pytest.mark.skipif(not HAVE_NUMBA, reason="the derived fields' skip is compiled")
def test_a_QUIET_REGION_WAKES_when_heat_smoke_or_liquid_reach_its_edge():
    """The far side of the room is quiet until things arrive there (a bed of
    embers glows apart from it all, and the bed's pores are emptied of air at
    tick 10, so it must draw air in from the room): a hot
    block at tick 20, a puff of smoke at 30, liquids poured at 40, a second
    flame at 50 and a burning board under the open sky's last layer at 60.
    Each must be worked on from the tick it arrives, exactly as a whole-world
    sweep would."""
    def build():
        w = _corner_fire()
        w.mat[1:39, 1:31, 26:28] = AIR                   # an open roof, so a
        w.smass[1:39, 1:31, 26:28] = 0.0                 # flame can reach the
        w.fill(2, 6, 2, 6, 26, 27, WOOD, frac=0.5)       # top of the world
        w.fill(20, 23, 3, 6, 1, 2, CHAR, frac=0.5)       # and a bed of embers
        C = w.heat_capacity()
        w.E[20:23, 3:6, 1] = (700.0 - AMBIENT) * C[20:23, 3:6, 1]
        return w

    def poke(w, t):
        C = w.heat_capacity()
        if t == 10:                                      # the bed's own pores run
            w.o2[28:38, 22:30, 4:6] = 0.0                # dry: it must draw air in
        if t == 20:
            w.E[2:4, 28:30, 2:4] += (600.0 - AMBIENT) * C[2:4, 28:30, 2:4]
        if t == 30:
            w.smoke[20, 2, 20] += 3.0
        if t == 40:                                      # oil, and water poured
            for y in range(2, 6):                        # on top of it: the
                w.pour(36, y, 1, OIL, 800.0)             # heavier one sinks
                w.pour(36, y, 2, WATER, 800.0)
                w.pour(37, y, 3, OIL, 800.0)
        if t == 50:
            cold = w.mat[4:12, 4:8, 1:6] == WOOD
            E = w.E[4:12, 4:8, 1:6]
            E[cold] = (500.0 - AMBIENT) * C[4:12, 4:8, 1:6][cold]
        if t == 60:
            top = w.mat[2:6, 2:6, 26] == WOOD
            E = w.E[2:6, 2:6, 26]
            E[top] = (600.0 - AMBIENT) * C[2:6, 2:6, 26][top]

    got = _lockstep(build, 90, poke)
    assert got is None, f"regions on and off part at tick {got[0]}, in {got[1]}"


def test_REGION_AT_is_the_SURFACE_REGION_read_at_those_cells():
    """The airspace each burning cell breathes, asked of the burning cells
    alone, is what the whole-world field says there: edges and walls too."""
    rng = np.random.default_rng(3)
    for shape in ((9, 7, 5), (1, 6, 4), (12, 3, 1)):
        w = World(*shape)
        w.mat[...] = np.where(rng.random(shape) < 0.45, STONE, AIR)
        w.skip_quiet = False
        w._air_regions()
        full = w._region_of_surface()
        for _ in range(5):
            mask = rng.random(shape) < 0.3
            got = w._region_at(mask)
            assert got.dtype == full.dtype
            assert np.array_equal(got, full[mask])


def test_OPEN_FACES_counted_at_a_few_cells_are_the_whole_world_counts():
    rng = np.random.default_rng(4)
    for shape in ((8, 6, 5), (1, 1, 7), (5, 1, 1)):
        mat = np.where(rng.random(shape) < 0.5, STONE, AIR).astype(np.uint8)
        fvol = np.where(rng.random(shape) < 0.3, 30.0, 0.0).astype(np.float32)
        film = 0.01 * 125.0
        i = np.flatnonzero(rng.random(shape) < 0.4)
        n_out, n_gas = _count_open_at(mat, fvol, film, i)
        assert n_out.dtype == n_gas.dtype == np.int8
        assert np.array_equal(n_out, _count_nb(mat == AIR).reshape(-1)[i])
        assert np.array_equal(n_gas, _count_nb((mat == AIR) & (fvol <= film)).reshape(-1)[i])


@pytest.mark.skipif(not HAVE_NUMBA, reason="the derived fields' skip is compiled")
def test_DERIVED_FIELDS_kept_cell_by_cell_are_the_FULL_BUILD_to_the_bit():
    """Heat capacity and porosity, kept up where matter moved, against a
    build from scratch: a few cells, a NaN, a -0.0, and so many cells that
    it rebuilds everything."""
    rng = np.random.default_rng(6)
    w = _corner_fire()
    fresh = _corner_fire()
    w.heat_capacity(), w.porosity()
    gen = w._matter_gen

    def check(moved):
        cap, por = w.heat_capacity(), w.porosity()
        fresh.mat, fresh.smass, fresh.fl, fresh.fvol = (
            w.mat.copy(), w.smass.copy(), w.fl.copy(), w.fvol.copy())
        fresh._cap_key = None
        want_cap, want_por = fresh.heat_capacity(), fresh.porosity()
        assert cap.tobytes() == want_cap.tobytes()
        assert por.tobytes() == want_por.tobytes()
        return w._matter_gen

    held = w.heat_capacity()
    was = held.copy()
    w.smass[29, 23, 4] *= 0.5                             # a little wood burns
    w.fvol[5, 21, 1] -= 10.0                              # a little water moves
    gen2 = check(True)
    assert gen2 == gen + 1, "a change is a new generation, as before"
    assert w.heat_capacity() is not held and np.array_equal(held, was), \
        "the old array is never written: whoever holds it keeps that state"
    assert check(False) == gen2, "no change, no new generation"
    w.smass[30, 24, 4] = -0.0                             # equal to 0.0 by `!=`
    w.smass[31, 24, 4] = np.nan                           # never equal to itself
    check(True)
    cells = rng.random(w.shape) < 0.3                     # too many: a full build
    w.smass[cells & (w.mat == STONE)] *= np.float32(0.9)
    check(True)
    out = np.empty(4, np.int64)
    a = np.arange(10, dtype=np.float32)
    b = a.copy()
    b[[2, 7]] = -1
    z = np.zeros(10, np.uint8)
    assert _diff4(a, z, z, a, b, z, z, a, out) == 2 and list(out[:2]) == [2, 7]
    assert _diff4(a, z, z, a, a, z, z, a, out) == 0
    b[:] = -1
    assert _diff4(a, z, z, a, b, z, z, a, out) == -1
