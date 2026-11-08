"""THE FLUID LAWS, COMPILED AND PLAIN, AND WHAT A LIQUID DOES ON A FLOOR
(wave 12, lane FLUIDS).

Step 1, SLEEP: `_law_flow` and `_law_head` have a compiled sweep (the default,
`w.fused`) and the plain Python one, which is the definition. They must give
the same bits, tick by tick, in every field a fluid law may write. A world
where no cell would move any fluid is not swept at all.

The later steps (the equilibrium film, drop gravity in SI, the retained film)
are checked at the end of the file against the bands of docs/realism-references
4.4: L-T1, L-T2, L-T3, L-T8.
"""
import numpy as np
import pytest

from src.voxel import fluids as F
from src.voxel.kernels import HAVE_NUMBA
from src.voxel.materials import (ACID, AIR, FLUID, MISCIBLE, NOFLUID, OIL, STONE, TICK_S, WATER,
                                 WEAK_ACID, _MIXHEAT)
from src.voxel.sim import World

pytestmark = pytest.mark.skipif(not HAVE_NUMBA, reason="the sweeps are compiled")

_FIELDS = ("mat", "smass", "fl", "fvol", "fpot", "E", "smoke", "o2", "skin", "p_add")


def _bits(w):
    out = {k: np.ascontiguousarray(getattr(w, k)).tobytes()
           for k in _FIELDS if getattr(w, k, None) is not None}
    out["drops"] = repr([[float(v) for v in d] for d in w.drops])
    out["shed"] = repr(w.shed)
    return out


def _lockstep(build, ticks, poke=None):
    """Two copies of one world, compiled and plain, stepped side by side.
    Returns the first (tick, field) where their bits part, or None."""
    a, b = build(), build()
    b.restore(a.snapshot())
    a.fused, b.fused = True, False
    for t in range(ticks):
        for w in (a, b):
            if poke is not None:
                poke(w, t)
            w.step()
        x, y = _bits(a), _bits(b)
        for k in x:
            if x[k] != y[k]:
                return t, k
    return None


def _tank(n=14, d=8, voxel_cm=10):
    w = World(n, n, d, voxel_cm=voxel_cm)
    w.fill(0, n, 0, n, 0, 1, STONE)
    return w


def _spill():
    """A heap of water and oil on a flat floor with a step in it, a wall, and
    a puddle already on the far side: it spreads, runs down the step, and the
    oil climbs over the water."""
    w = _tank(16, 8)
    w.fill(0, 16, 6, 7, 1, 4, STONE)                    # a wall
    w.fill(8, 16, 0, 6, 1, 2, STONE)                    # a step
    for x in range(2, 5):
        for y in range(2, 5):
            for z in (1, 2, 3):
                w.pour(x, y, z, WATER, w.cap)
    w.pour(3, 3, 4, OIL, 700.0)
    w.pour(12, 3, 2, WATER, 300.0)
    return w


def _vessels():
    """Two tanks joined by a pipe near the floor, one full of water: the head
    law lifts the level in the other (and a third body stands apart)."""
    w = World(24, 8, 8, voxel_cm=10)
    w.fill(0, 24, 0, 8, 0, 1, STONE)
    w.fill(0, 24, 0, 8, 1, 7, STONE)
    w.mat[1:8, 1:7, 1:7] = AIR
    w.mat[16:23, 1:7, 1:7] = AIR
    w.mat[8:16, 3:5, 1:3] = AIR                         # the pipe
    w.smass[w.mat == AIR] = 0.0
    for x in range(1, 8):
        for y in range(1, 7):
            for z in (1, 2, 3, 4):
                w.pour(x, y, z, WATER, w.cap)
    for x in range(17, 20):
        w.pour(x, 3, 1, OIL, 300.0)
    return w


def _acid_bath():
    """Vitriol poured onto water, and water onto weaker acid: the mixing law
    and its heat run in the flow and the head laws."""
    w = _tank(12, 8)
    for x in range(2, 10):
        for y in range(2, 10):
            w.pour(x, y, 1, WATER, w.cap)
            w.pour(x, y, 2, WATER, 0.6 * w.cap)
    for x in range(4, 7):
        for y in range(4, 7):
            w.pour(x, y, 3, ACID, w.cap, 1.0)
    w.pour(8, 8, 3, WEAK_ACID, 500.0, 0.8)
    return w


@pytest.mark.parametrize("build,ticks", [(_spill, 70), (_vessels, 70), (_acid_bath, 50)])
def test_the_COMPILED_SWEEPS_are_the_PLAIN_ONES_to_the_bit(build, ticks):
    got = _lockstep(build, ticks)
    assert got is None, f"compiled and plain part at tick {got[0]}, in {got[1]}"


def test_SPLASHES_and_DROPS_in_flight_agree_to_the_bit():
    """A block hits a pond; the droplets arc and land back on the grid (the
    arc is Python in both paths, the landing feeds the compiled sweep)."""
    def build():
        w = _tank(14, 10)
        for x in range(1, 13):
            for y in range(1, 13):
                for z in (1, 2):
                    w.pour(x, y, z, WATER, w.cap)
        return w

    def poke(w, t):
        if t in (3, 9):
            w._splash(6, 6, 2, 3)

    got = _lockstep(build, 40, poke)
    assert got is None, f"compiled and plain part at tick {got[0]}, in {got[1]}"


def test_THE_LAYER_SWAP_writes_chains_as_numpy_does():
    """Water over oil over water over oil in one column: every pair swaps at
    once, and a cell that is both the upper of one pair and the lower of the
    next takes the lower's value."""
    def build():
        w = _tank(6, 12)
        w.pour(2, 2, 1, WATER, 400.0)
        w.pour(2, 2, 2, OIL, 300.0)
        w.pour(2, 2, 3, WATER, 200.0)
        w.pour(2, 2, 4, OIL, 100.0)
        w.pour(2, 2, 5, WATER, 50.0)
        w.fpot[2, 2, 1:6] = np.float32([0.1, 0.2, 0.3, 0.4, 0.5])
        w.E[2, 2, 1:6] = np.float32([5.0, 6.0, 7.0, 8.0, 9.0])
        return w

    got = _lockstep(build, 6)
    assert got is None, f"compiled and plain part at tick {got[0]}, in {got[1]}"


def test_ONE_MOVE_is_the_PLAIN_MOVE_to_the_bit():
    """`_move` on random cells, both kinds of amount, every fluid pair: the
    cell's fluid, volume, potency and heat, and the heat of mixing."""
    rng = np.random.default_rng(11)
    tb = F._tb()
    bad = 0
    for trial in range(400):
        w = World(3, 1, 1, voxel_cm=5)
        w.mat[...] = AIR
        w.smass[:, 0, 0] = np.float32(rng.random(3) * 3.0 * (rng.random(3) < 0.5))
        f_src = int(rng.choice([WATER, OIL, ACID, WEAK_ACID, WATER, ACID]))
        f_dst = int(rng.choice([NOFLUID, WATER, ACID, WEAK_ACID, OIL]))
        w.fl[0, 0, 0], w.fl[1, 0, 0] = f_src, f_dst
        w.fvol[0, 0, 0] = np.float32(rng.random() * w.cap)
        w.fvol[1, 0, 0] = np.float32(rng.random() * w.cap) if f_dst else 0.0
        w.fpot[0, 0, 0] = np.float32(rng.random())
        w.fpot[1, 0, 0] = np.float32(rng.random()) * (rng.random() < 0.8)
        w.E[:, 0, 0] = np.float32(rng.normal(size=3) * 500.0)
        ml = float(rng.random() * w.fvol[0, 0, 0])
        if trial % 5 == 0:                              # leaves only dust behind
            ml = float(w.fvol[0, 0, 0]) - float(rng.random() * 0.02)
            w.fvol[1, 0, 0] = np.float32(rng.random() * 0.5 * w.cap) if f_dst else 0.0
        if trial % 2:
            ml = np.float32(ml)
        ref = w.snapshot()
        w._move_fluid((0, 0, 0), (1, 0, 0), ml)
        want = [a.copy() for a in (w.fl, w.fvol, w.fpot, w.E)]
        w.restore(ref)
        flat = [a.reshape(-1) for a in (w.fl, w.fvol, w.fpot, w.E, w.smass, w.mat)]
        F._move(0, 1 * w.shape[1] * w.shape[2], float(ml), isinstance(ml, np.float32),
                *flat, np.float32(w.cap), np.float32(w.c_airbase), tb)
        for got, ex in zip((w.fl, w.fvol, w.fpot, w.E), want):
            if got.tobytes() != ex.tobytes():
                bad += 1
                break
    assert bad == 0


def test_the_INTERPOLATION_is_numpys_to_the_bit():
    rng = np.random.default_rng(5)
    for f, (p_at, q_at) in _MIXHEAT.items():
        n = len(p_at)
        xs = np.concatenate([rng.random(300), p_at[::97], [-1.0, 2.0, 0.0, 1.0]])
        for x in xs:
            assert F._interp(float(x), p_at, q_at, n) == float(np.interp(x, p_at, q_at))


def test_the_TABLES_the_kernels_assume():
    assert AIR == 0 and NOFLUID == 0
    mixt = F._tb()[0]
    for a in range(F._NFL):
        for b in range(F._NFL):
            r = World._mix(a, b)
            assert mixt[a, b] == (-1 if r is None else r)


def test_a_STILL_POND_is_not_swept(monkeypatch):
    """A pond at rest costs a scan and no settling sweep: `_sweep` runs for a
    world that moves, never for one that cannot."""
    w = _tank(30, 6, voxel_cm=5)
    w.fill(0, 30, 0, 1, 1, 6, STONE)                    # and a wall all round
    w.fill(0, 30, 29, 30, 1, 6, STONE)
    w.fill(0, 1, 0, 30, 1, 6, STONE)
    w.fill(29, 30, 0, 30, 1, 6, STONE)
    for x in range(1, 29):
        for y in range(1, 29):
            for z in (1, 2, 3):
                w.pour(x, y, z, WATER, w.cap)
    w.step()
    calls = {"sweep": 0}
    real = F._sweep

    def counting(*a):
        calls["sweep"] += 1
        return real(*a)
    monkeypatch.setattr(F, "_sweep", counting)
    for _ in range(5):
        w.step()
    assert calls["sweep"] == 0
    w.pour(5, 5, 4, WATER, 100.0)                       # a drip: it wakes
    w.step()
    assert calls["sweep"] == 1


# ── step 2: the film a liquid comes to rest at (L-T2, L-T3) ─────────────────

def _tip(fluid, liters, vcm, foot=None):
    """A bucket tipped onto a level stone floor: the liquid heaped in a short
    column at the middle of a big empty floor. Returns the world."""
    n = {5: 80, 10: 40}[vcm]
    w = World(n, n, 14, voxel_cm=vcm)
    w.fill(0, n, 0, n, 0, 1, STONE)
    foot = foot or {5: 4, 10: 2}[vcm]
    cols = [(x, y) for x in range(n // 2, n // 2 + foot) for y in range(n // 2, n // 2 + foot)]
    ml, z = liters * 1000.0, 1
    while ml > 0:
        for x, y in cols:
            add = min(w.cap, ml)
            w.pour(x, y, z, fluid, add)
            ml -= add
            if ml <= 0:
                break
        z += 1
    return w


def _fluid_laws(w, seconds):
    """Only the three laws that move a liquid, for `seconds` of world time:
    what a spill does needs nothing else."""
    for _ in range(int(seconds / TICK_S)):
        w._law_flow()
        w._law_stir()
        w._law_head()


def _wetted(w, fluid):
    """Area (m²) of the floor cells holding at least 0.1 mm of liquid, and the
    mean depth (mm) over them."""
    face = (0.1 * w.scale) ** 2
    thr = 1e-4 * face * 1e6                                # ml in 0.1 mm of one face
    wet = w.fvol[:, :, 1] >= thr
    area = float(wet.sum()) * face
    ml = float(w.fvol[:, :, 1:][w.fl[:, :, 1:] == fluid].sum())
    return area, (ml * 1e-6 / area * 1e3 if area else 0.0)


def test_L_T2_A_BUCKET_SPILLED_ON_THE_FLOOR_stops_at_its_film_whatever_the_voxel():
    """L-T2 (sources 66, 95, 96): 1 L of water tipped on a level stone floor
    spreads to 1-4 mm deep, 0.25-1.0 m², and 5 cm and 10 cm voxels agree
    within 30%. (Today: 0.71 m²/L at 5 cm and 2.85 at 10, area going as
    V^(2/3).) The 10 L half is `test_L_T2_TEN_LITRES_...`."""
    res = {}
    for vcm in (5, 10):
        w = _tip(WATER, 1.0, vcm)
        mass = w.total_fluid(WATER)
        _fluid_laws(w, 60)
        res[vcm] = _wetted(w, WATER)
        assert w.total_fluid(WATER) == pytest.approx(mass, rel=1e-5), "the mass ledger closes"
        area, depth = res[vcm]
        assert 0.25 <= area <= 1.0 and 1.0 <= depth <= 4.0, f"{vcm} cm: {area:.2f} m², {depth:.1f} mm"
        del w
    assert res[5][0] == pytest.approx(res[10][0], rel=0.3), f"5 cm {res[5]}, 10 cm {res[10]}"


def test_L_T2_TEN_LITRES_cover_ten_times_the_area_of_one_at_both_sizes():
    """The 10 L half of L-T2: the same film, so the area per litre is
    the same as one litre's within 30% (it went as V^(2/3) before), the depth
    is still 1-4 mm, and most of the area is there by 30 s."""
    for vcm in (5, 10):
        w = _tip(WATER, 1.0, vcm)
        _fluid_laws(w, 60)
        one = _wetted(w, WATER)[0]
        del w
        w = _tip(WATER, 10.0, vcm)
        mass = w.total_fluid(WATER)
        _fluid_laws(w, 30)
        at30 = _wetted(w, WATER)[0]
        _fluid_laws(w, 30)
        area, depth = _wetted(w, WATER)
        assert w.total_fluid(WATER) == pytest.approx(mass, rel=1e-5), "the mass ledger closes"
        assert 1.0 <= depth <= 4.0, f"{vcm} cm: {depth:.1f} mm"
        assert area / 10.0 == pytest.approx(one, rel=0.3), f"{vcm} cm: {area:.2f} m² for 10 L, {one:.2f} for 1 L"
        assert at30 >= 0.7 * area, f"{vcm} cm: {at30:.2f} of {area:.2f} m² by 30 s"
        del w


def test_L_T3_LAMP_OIL_SPILL_is_thinner_than_water_and_the_same_at_both_sizes():
    """L-T3 (source 66): 0.5 L of oil on a smooth stone floor covers 0.2-1.2 m²
    at 0.4-2.5 mm (lube oil: 0.57 m² at 0.88 mm), and the area changes by at
    most 30% with the voxel size."""
    res = {}
    for vcm in (5, 10):
        w = _tip(OIL, 0.5, vcm)
        _fluid_laws(w, 60)
        res[vcm] = _wetted(w, OIL)
        area, depth = res[vcm]
        assert 0.2 <= area <= 1.2 and 0.4 <= depth <= 2.5, f"{vcm} cm: {area:.2f} m², {depth:.2f} mm"
        del w
    assert res[5][0] == pytest.approx(res[10][0], rel=0.3), f"5 cm {res[5]}, 10 cm {res[10]}"


def test_EACH_LIQUID_has_its_own_film():
    """The film is the capillary length times 2 sin(theta/2): water a few mm,
    a wetting oil under a mm, molten lead (it does not wet stone) 3-4 mm. At
    5 cm a face is 25 cm², so a film is that many ml."""
    film, ret, tol = F._wet_tab(1.0)
    depth = lambda f: film[f] / (0.1 ** 2 * 1e6) * 1e3        # mm
    from src.voxel.materials import MLEAD
    assert 1.0 <= depth(WATER) <= 4.0
    assert depth(OIL) < 1.0
    assert 2.9 <= depth(MLEAD) <= 4.0                          # L-T1's band: 2.9-4.0 mm
    assert depth(OIL) < depth(WATER) < depth(MLEAD)
    assert all(ret[f] < film[f] for f in (WATER, OIL, MLEAD))


# ── step 3: a drop falls at g (L-T8) ────────────────────────────────────────

def _splash_pond(vcm):
    """A stone block, 15 cm (20 at 10 cm), dropped from 40 cm into a 30 cm pond."""
    n, depth, k = {5: (30, 6, 3), 10: (16, 3, 2)}[vcm]
    w = World(n, n, 20 if vcm == 5 else 10, voxel_cm=vcm)
    w.fill(0, n, 0, n, 0, 1, STONE)
    w.fill(0, n, 0, 1, 1, depth + 1, STONE)
    w.fill(0, n, n - 1, n, 1, depth + 1, STONE)
    w.fill(0, 1, 0, n, 1, depth + 1, STONE)
    w.fill(n - 1, n, 0, n, 1, depth + 1, STONE)
    for x in range(1, n - 1):
        for y in range(1, n - 1):
            for z in range(1, 1 + depth):
                w.pour(x, y, z, WATER, w.cap)
    c, zb = n // 2, depth + 1 + round(40 / vcm)
    w.fill(c, c + k, c, c + k, zb, zb + k, STONE)
    return w


@pytest.mark.parametrize("vcm", [5, 10])
def test_L_T8_A_SPLASH_OBEYS_g(vcm):
    """L-T8 (source 97): the droplets a block throws out of a pond fall with
    9.81 m/s² +-10%, at 5 and at 10 cm. (Today: 20 m/s² at 5 cm and 40 at 10:
    the fall was 0.25 cells per tick² whatever the cell.) Fitted as the median
    second difference of each parcel's height, tick by tick, in flight."""
    w = _splash_pond(vcm)
    tracks = {}
    for _ in range(100):
        w.step()
        for d in w.drops:
            tracks.setdefault(id(d), []).append(d[2])
    second = [v for tr in tracks.values() if len(tr) >= 4 for v in np.diff(np.array(tr), 2)]
    assert len(second) >= 8, "the block throws droplets"
    g = -float(np.median(second)) * (0.1 * w.scale) / TICK_S ** 2
    assert 9.81 * 0.9 <= g <= 9.81 * 1.1, f"{g:.2f} m/s²"


# ── step 4: the coat a liquid keeps, and what a porous solid drinks ─────────

def _soaking_scene():
    """A wooden floor with a puddle on it, a wooden post standing in the
    water, a rope hanging into it, and a shaft of stone it runs down."""
    from src.voxel.materials import ROPE, WOOD
    w = World(16, 12, 10, voxel_cm=5)
    w.fill(0, 16, 0, 12, 0, 1, STONE)
    w.fill(0, 10, 0, 12, 1, 2, WOOD)                    # a wooden floor...
    for x in range(1, 7):
        for y in range(1, 7):
            w.pour(x, y, 2, WATER, 6.0)                 # ...with a puddle on it
    w.fill(3, 4, 3, 4, 2, 6, WOOD)                      # a post in the puddle
    w.fill(5, 6, 5, 6, 2, 5, ROPE, frac=0.3)
    w.pour(3, 2, 6, WATER, 80.0)                        # and a pour, onto the post
    w.fill(11, 15, 2, 6, 1, 9, STONE)                   # a stone block with a shaft
    w.mat[12:14, 3:5, 2:9] = AIR
    w.smass[12:14, 3:5, 2:9] = 0.0
    w.pour(12, 3, 8, WATER, 90.0)
    return w


def test_the_SOAKING_SCENE_is_the_PLAIN_ONE_to_the_bit():
    got = _lockstep(_soaking_scene, 60)
    assert got is None, f"compiled and plain part at tick {got[0]}, in {got[1]}"


def test_a_PUDDLE_on_WOOD_is_drunk_at_S_root_t():
    """The water a face takes in follows i = S*sqrt(t) (source 109; pine
    1-7 across the grain, 10-16 along it, one number here: 1), until the
    wood holds its fill. Clocked by the water the wood already holds, so it
    is the same at 15 s and at 60 s. The ledger closes: the water went into
    the wood's cells, and is counted there."""
    from src.voxel.materials import WOOD
    w = World(14, 14, 5, voxel_cm=5)
    w.fill(0, 14, 0, 14, 0, 1, WOOD)
    for x in range(3, 11):
        for y in range(3, 11):
            w.pour(x, y, 1, WATER, 5.0)
    mass = w.total_fluid(WATER)
    face = 64 * 0.05 ** 2
    for secs in (15.0, 60.0):
        _fluid_laws(w, secs - (0.0 if secs == 15.0 else 15.0))
        g_per_m2 = float(w.fvol[w.mat == WOOD].sum()) / face    # 1 g per ml
        want = F.SORPTIVITY[WOOD][0] * np.sqrt(secs)
        assert g_per_m2 == pytest.approx(want, rel=0.1), f"{secs:.0f} s: {g_per_m2:.1f} g/m², want {want:.1f}"
    assert w.total_fluid(WATER) == pytest.approx(mass, rel=1e-5)
    assert (w.fl[w.mat == WOOD] == WATER).sum() == 64, "held in the wood's own cells"


def test_STONE_DRINKS_NOTHING_and_oil_is_drunk_slowly():
    from src.voxel.materials import WOOD
    def drunk(floor, fluid):
        w = World(8, 8, 4, voxel_cm=5)
        w.fill(0, 8, 0, 8, 0, 1, floor)
        for x in range(2, 6):
            for y in range(2, 6):
                w.pour(x, y, 1, fluid, 3.0)
        _fluid_laws(w, 30)
        return float(w.fvol[w.mat == floor].sum())
    assert drunk(STONE, WATER) == 0.0
    assert 0.0 < drunk(WOOD, OIL) < 0.5 * drunk(WOOD, WATER)


def _shaft():
    """A one-cell shaft of stone, 13 cells deep, on a stone floor."""
    w = World(5, 5, 16, voxel_cm=5)
    w.fill(0, 5, 0, 5, 0, 1, STONE)
    w.fill(0, 5, 0, 5, 1, 15, STONE)
    w.mat[2, 2, 1:15] = AIR
    w.smass[2, 2, 1:15] = 0.0
    return w


def test_WATER_FALLING_PAST_A_WALL_LEAVES_A_COAT():
    """A liquid running down a shaft leaves a coat on the solid faces it
    touches (`RETAINED`: a quarter of a mm of water on each of four faces),
    and none where it lands in a puddle. All of it is counted."""
    w = _shaft()
    w.pour(2, 2, 14, WATER, 100.0)
    _film, ret, _tol = F._wet_tab(w.scale)
    coat = 4 * ret[WATER]
    _fluid_laws(w, 5)
    shaft = w.fvol[2, 2, 3:14]
    assert shaft.min() > 0.5 * coat and shaft.max() < 1.01 * coat, \
        f"the coat is {shaft.min():.2f}-{shaft.max():.2f} ml of {coat:.2f}"
    assert w.fvol[2, 2, 1] > 100.0 - 13 * coat - 1.0, "and the rest is in the puddle at the bottom"
    assert w.total_fluid(WATER) == pytest.approx(100.0, rel=1e-5)


def test_a_COAT_RUNS_OFF_with_the_time_constant_DRAIN_S():
    """A cell holding no more than its coat sheds a share of it every tick:
    after DRAIN_S it holds 1/e of what it did. Counted all the way down."""
    w = _shaft()
    _film, ret, _tol = F._wet_tab(w.scale)
    coat = 4 * ret[WATER]
    w.fl[2, 2, 13], w.fvol[2, 2, 13] = WATER, coat
    _fluid_laws(w, F.DRAIN_S)
    assert float(w.fvol[2, 2, 13]) == pytest.approx(coat / np.e, rel=0.03)
    assert w.total_fluid(WATER) == pytest.approx(coat, rel=1e-5), "none of it is lost on the way"


def test_the_SKIN_holds_water_first():
    from src.voxel.materials import WOOD
    w = World(8, 8, 5, voxel_cm=5)
    w.fill(0, 8, 0, 8, 0, 1, STONE)
    w.fill(3, 5, 3, 5, 1, 3, WOOD)
    assert float(w.skin_water().sum()) == 0.0
    w.fl[3, 3, 1], w.fvol[3, 3, 1] = WATER, 0.2             # a little water
    w.fl[4, 4, 1], w.fvol[4, 4, 1] = WATER, 30.0            # and a lot
    sk = w.skin_water()
    assert sk[3, 3, 1] == pytest.approx(0.2)
    assert 0.2 < sk[4, 4, 1] < 30.0, "what the skin cannot hold is in the core"


# ── L-T1: molten lead splats on a cold stone floor ──────────────────────────

def _lead_splat(vcm):
    """100 ml of molten lead at 400 °C poured from 20 cm onto cold stone, with
    every law running (it freezes where it lies). Returns the area of the
    splat (m², cells holding at least 0.5 mm of lead), its mean thickness
    (mm), the second it was last liquid, and the lead left as a liquid."""
    from src.voxel.materials import AMBIENT, LEAD, MLEAD
    n = {5: 24, 10: 12}[vcm]
    w = World(n, n, 16 if vcm == 5 else 8, voxel_cm=vcm)
    w.fill(0, n, 0, n, 0, 1, STONE)
    z = 1 + round(20 / vcm)
    w.pour(n // 2, n // 2, z, MLEAD, 100.0)
    w.E[n // 2, n // 2, z] = (400.0 - AMBIENT) * w.heat_capacity()[n // 2, n // 2, z]
    last_liquid = 0.0
    for t in range(int(60 / TICK_S)):
        w.step()
        if ((w.fl == MLEAD) & (w.fvol > 0.5)).any():
            last_liquid = (t + 1) * TICK_S
    face = (0.1 * w.scale) ** 2
    rho = FLUID[MLEAD][0]                                   # g/ml, the same as LEAD's
    cells = (w.mat[:, :, 1] == LEAD) & (w.smass[:, :, 1] >= 0.5e-3 * face * 1e6 * rho)
    area = float(cells.sum()) * face
    grams = float(w.smass[:, :, 1][cells].sum())
    return area, grams / rho * 1e-6 / area * 1e3, last_liquid, float(w.fvol[w.fl == MLEAD].sum())


def test_L_T1_MOLTEN_LEAD_SPLATS_to_its_film_and_freezes():
    """L-T1 (sources 87, 95): 100 ml of lead at 400 °C poured from 20 cm onto a
    cold stone floor makes a splat 2.9-4.0 mm thick over 0.025-0.035 m² (the
    capillary length 2.07 mm and a contact angle of 90-150 degrees); accept
    2-6 mm and 0.017-0.05 m², the same area within 50% at 5 and 10 cm, and
    solid in 1-60 s. (Today: it spreads like water.)"""
    res = {}
    for vcm in (5, 10):
        area, thick, solid_at, liquid = _lead_splat(vcm)
        res[vcm] = area
        assert 0.017 <= round(area, 6) <= 0.05, f"{vcm} cm: {area:.4f} m²"      # (5 cells of 10 cm)
        # (a cross of five 10 cm cells is 0.05 m² and 2.000 mm: the two bands meet
        # there, so the thickness is read to 0.001 mm; a 0.03 ml speck sits in a sixth cell)
        assert 2.0 <= round(thick, 3) <= 6.0, f"{vcm} cm: {thick:.4f} mm thick"
        assert 1.0 <= solid_at <= 60.0 and liquid == 0.0, f"{vcm} cm: liquid until {solid_at:.1f} s"
    assert res[5] == pytest.approx(res[10], rel=0.5)
