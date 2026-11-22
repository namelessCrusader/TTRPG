"""The seam between the voxel world and the particle solver.

What crosses it must arrive whole, the solver must see the lattice as it is
now, and the compiled solver must be the plain one, here inside a World
rather than on a bare particle cloud (tests/test_mpm.py does that part).
"""

import numpy as np
import pytest

from src.mpm import kernels
from src.mpm.core import SDF_MARGIN
from src.mpm.matter import _NAMES
from src.voxel.sim import World, AIR, STONE, WOOD, GLASS, FLESH, WATER, IRON, CHAR, ASH
from src.voxel.materials import FLUID, MIRON


def _block_cells(x0, x1, y0, y1, z0, z1):
    return np.stack(np.meshgrid(np.arange(x0, x1), np.arange(y0, y1),
                                np.arange(z0, z1), indexing="ij"), -1).reshape(-1, 3)


def test_EVERY_MATERIAL_THAT_CAN_BE_SOLID_CAN_LEAVE_THE_LATTICE():
    """Ash, char and tin had six fields in the solver's table where every
    reader unpacks seven, so a charred stump or a tin ingot that toppled
    raised ValueError inside `promote` (docs/realism-references.md G6). Every
    material with a particle row is promoted here, and its grams are counted
    on both sides of the seam for five ticks."""
    for vid in sorted(_NAMES):
        w = World(12, 12, 14)
        w.fill(0, 12, 0, 12, 0, 1, STONE)
        w.fill(5, 7, 5, 7, 6, 8, vid)
        before = w.total_mass(vid)
        assert before > 0.0, vid
        n = w.matter.promote(_block_cells(5, 7, 5, 7, 6, 8))
        assert n == 8, (vid, n)
        # The lattice keeps grams in float32, and `promote` used to turn them
        # into kilograms in float32 too, so the crossing itself rounded each
        # cell: 4800 g of wood came out 4800.00019 g. It converts in float64
        # now, so the crossing is as exact as the five ticks after it.
        after = w.total_mass(vid)
        assert after == pytest.approx(before, rel=1e-12), vid
        for _ in range(5):
            w.step()
            assert w.total_mass(vid) == pytest.approx(after, rel=1e-12), vid


def _pole(compiled):
    """golden.py's pole_mpm: a notched pole that the torque law sends
    through the seam, and that falls onto a stone floor."""
    w = World(24, 8, 16)
    w.fill(0, 24, 0, 8, 0, 1, STONE)
    w.fill(4, 7, 3, 5, 1, 12, WOOD)
    w.mat[5:7, 3:5, 1] = AIR
    w.smass[5:7, 3:5, 1] = 0.0
    w.matter.mpm.compiled = compiled
    return w


def _vessel(compiled):
    """An open glass box holding water, promoted in mid-air: it falls, lands
    on stone, and its glass is brittle, so whatever it carries is poured."""
    w = World(20, 20, 24, voxel_cm=5)
    w.wet = None        # kept dry on purpose: this test is about what the solver hands back untouched; the humid surface law (evaporation) writes the energy and volume of water cells the solver holds, a different question from the boil
    w.fill(0, 20, 0, 20, 0, 1, STONE)
    w.fill(7, 12, 7, 12, 10, 15, GLASS)
    w.mat[8:11, 8:11, 11:15] = AIR
    w.smass[8:11, 8:11, 11:15] = 0.0
    for x in range(8, 11):
        for y in range(8, 11):
            w.pour(x, y, 11, WATER, 100.0)
    w.matter.promote(np.argwhere(w.mat == GLASS))
    w.matter.mpm.compiled = compiled
    return w


_LATTICE = ("mat", "smass", "E", "fl", "fvol", "fpot", "edge", "rubble", "moving")
_PARTICLES = ("x", "v", "C", "F", "Jp", "dmg", "cid", "fvol", "E")


@pytest.mark.skipif(not kernels.HAVE_NUMBA, reason="numba is not installed")
@pytest.mark.parametrize("build,ticks", [(_pole, 80), (_vessel, 40)],
                         ids=["pole", "vessel"])
def test_a_WORLD_IS_THE_SAME_WORLD_with_the_COMPILED_SOLVER(build, ticks):
    """The same world twice, one carried by the numpy solver and one by the
    compiled one, compared after every tick: the lattice the laws read, and
    every particle. The pole is the cheapest world the torque law sends
    through the seam; the vessel carries water and can break."""
    a, b = build(False), build(True)
    for t in range(ticks):
        a.step()
        b.step()
        for f in _LATTICE:
            assert np.array_equal(getattr(a, f), getattr(b, f)), (t, f)
        ma, mb = a.matter.mpm, b.matter.mpm
        for f in _PARTICLES:
            assert np.array_equal(getattr(ma, f), getattr(mb, f)), (t, "mpm." + f)
    assert len(a.matter), "nothing went through the seam"


@pytest.mark.skipif(not kernels.HAVE_NUMBA, reason="numba is not installed")
def test_a_WORLD_THAT_IS_NOT_FUSED_does_not_fuse_its_SOLVER(monkeypatch):
    """`w.fused` is the world's one switch for its compiled paths, and
    the particle solver obeys it too: with it off, the fused substep
    (`kernels.substep`) is never called, and the solver's own switch is
    left as it was. With it on, it runs every substep. Both worlds are the
    same bits, particle for particle."""
    calls = []
    fused = kernels.substep

    def counted(*a):
        calls.append(1)
        return fused(*a)
    monkeypatch.setattr(kernels, "substep", counted)
    plain, fast = _vessel(True), _vessel(True)
    plain.fused = False
    subs = 0
    for _ in range(8):
        plain.step()
        assert not calls and plain.matter.mpm.fused
        n = len(calls)
        fast.step()
        subs = len(calls) - n
        assert subs > 0
        calls.clear()
        for f in _PARTICLES:
            assert np.array_equal(getattr(plain.matter.mpm, f),
                                  getattr(fast.matter.mpm, f)), f


@pytest.mark.parametrize("sleep", [False, True], ids=["awake", "sleep"])
def test_the_SOLVER_SEES_THE_LATTICE_AS_IT_IS_where_the_particles_can_reach(sleep):
    """A man walks past a plank that is falling beside him. Every tick his
    cells move and their number does not change, which is exactly the change
    the old whole-world count could not see: it left the solver colliding
    with where he had been. Checked as the solver starts each step, over the
    particles' box and the distance field's margin around it.

    With sleep on, the plank lands and sleeps for the few ticks (3,
    measured) before he comes within reach, and on those ticks the solver
    is not called at all; every tick it is, it sees the lattice as it is,
    the tick it wakes included."""
    from src.voxel.scenes import _person, _Wants
    w = World(48, 20, 40, voxel_cm=5)
    w.fill(0, 48, 0, 20, 0, 1, STONE)
    w.exits = []
    p = _person(w, 8, 6)
    p["name"], p["facing"] = "walker", (1.0, 0.0)
    w.policy = _Wants(each={"walker": {"legs": "straight on"}})
    w.fill(14, 30, 10, 12, 14, 16, WOOD)          # beside his path, in the air
    w.matter.promote(np.argwhere(w.mat == WOOD))
    m = w.matter
    m.sleep = sleep
    seen, dozed = [], []
    solve = m.mpm.step

    def checked(dt, *a, **k):
        c = np.floor(m.mpm.x * m.mpm.inv_dx).astype(np.int64)
        lo = np.maximum(c.min(axis=0) - 2 - SDF_MARGIN, 0)
        hi = np.minimum(c.max(axis=0) + 3 + SDF_MARGIN, w.shape)
        box = tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))
        here = w.mat[box] != AIR
        seen.append((np.array_equal(m.mpm.wall[box], here),
                     (np.argwhere(w.mat[box] == FLESH) + lo).tobytes()))
        return solve(dt, *a, **k)

    m.mpm.step = checked
    for t in range(40):
        n = len(seen)
        w.step()
        if len(seen) == n:
            dozed.append(t)
            assert m.asleep().all(), f"tick {t}: the solver skipped matter that was awake"
    assert len(seen) == 40 - len(dozed)
    assert bool(dozed) == sleep, dozed
    stale = [t for t, (ok, _) in enumerate(seen) if not ok]
    assert not stale, f"the solver saw a stale lattice on ticks {stale}"
    # and the test tested something: he was in the box, and he moved in it
    flesh = [f for _, f in seen]
    assert sum(1 for f in flesh if f) >= 30, "he was never near the plank"
    assert sum(1 for a, b in zip(flesh, flesh[1:]) if a != b) >= 10, \
        "he did not move where the particles could reach"


# ------------------------------------------------------- which thing it is
#
# Particles carry the eid of the thing they came from (design 3.0, item 6), and
# rasterize hands each drawn cell to one thing. The lattice's own `eid` field
# arrives with the entity steps; until then a test gives its World one, which
# is exactly what `promote`, `rasterize` and `unrasterize` look for.


def _rasterize_world_sized(w, m, drawn, drawn_fluid):
    """`Matter.rasterize` as it was before it went sparse: every sum over an
    array the size of the world. Kept here as the definition the sparse one
    must equal bit for bit. `w` and `m` need only the arrays it reads."""
    cell = np.clip((m.x * m.inv_dx).astype(np.int64), 0,
                   np.array(w.shape) - 1)
    free = w.mat[tuple(cell.T)] == AIR
    cell = cell[free]
    if not len(cell):
        return 0
    flat = np.ravel_multi_index(tuple(cell.T), w.shape)
    n = int(np.prod(w.shape))
    grams = np.bincount(flat, m.m[free] * 1000.0, n)
    best_mat = np.zeros(n, np.int16)
    best_g = np.zeros(n)
    for vm in np.unique(m.vmat[free]):
        g = np.bincount(flat, np.where(m.vmat[free] == vm, m.m[free], 0.0), n)
        win = g > best_g
        best_mat[win], best_g[win] = vm, g[win]
    E = np.bincount(flat, m.E[free], n)
    whole = ~m.dmg[free]
    fvol = np.bincount(flat, np.where(whole, m.fvol[free], 0.0), n)
    fpot = np.bincount(flat, np.where(whole, m.fpot[free] * m.fvol[free], 0.0), n)
    edge = np.zeros(n)
    np.maximum.at(edge, flat, m.edge[free])
    fl = np.zeros(n, np.int16)
    best_v = np.zeros(n)
    for f in np.unique(m.fl[free]):
        if f == 0:
            continue
        v = np.bincount(flat, np.where((m.fl[free] == f) & whole,
                                       m.fvol[free], 0.0), n)
        win = v > best_v
        fl[win], best_v[win] = f, v[win]
    hit = grams > 0.0
    w.mat.ravel()[hit] = best_mat[hit]
    w.smass.ravel()[hit] = grams[hit]
    w.E.ravel()[hit] = E[hit]
    empty = w.fvol.ravel() <= 0.0
    draw_f = hit & empty
    w.fvol.ravel()[draw_f] = fvol[draw_f]
    w.fpot.ravel()[draw_f] = np.where(fvol[draw_f] > 0, fpot[draw_f] / np.maximum(fvol[draw_f], 1e-12), 0.0)
    w.fl.ravel()[draw_f] = fl[draw_f]
    drawn_fluid.ravel()[draw_f] = True
    poured = 0
    if m.dmg.any():
        sp = m.dmg[free] & (m.fvol[free] > 0.0)
        if sp.any():
            poured = 1
            sflat = flat[sp]
            add = np.bincount(sflat, m.fvol[free][sp], n)
            apot = np.bincount(sflat, (m.fpot[free] * m.fvol[free])[sp], n)
            aid = np.zeros(n, np.int16)
            bestv = np.zeros(n)
            for f in np.unique(m.fl[free][sp]):
                if f == 0:
                    continue
                v = np.bincount(sflat, np.where(m.fl[free][sp] == f,
                                                m.fvol[free][sp], 0.0), n)
                win = v > bestv
                aid[win], bestv[win] = f, v[win]
            got = add > 0.0
            w.fvol.ravel()[got] += add[got]
            w.fpot.ravel()[got] = np.where(
                w.fvol.ravel()[got] > 0,
                apot[got] / np.maximum(add[got], 1e-12), 0.0)
            w.fl.ravel()[got] = aid[got]
            drawn_fluid.ravel()[got] = False
            idxs = np.flatnonzero(free)[sp]
            m.fvol[idxs] = 0.0
            m.fl[idxs] = 0
    w.edge.ravel()[hit] = edge[hit]
    rub = np.bincount(flat, m.dmg[free].astype(float), n) > 0
    w.rubble.ravel()[hit] = rub[hit]
    w.moving.ravel()[hit] = True
    drawn.ravel()[hit] = True
    return poured


def _thrown_vessel(compiled):
    """The vessel, thrown down at 6 m/s so that it breaks on landing and
    pours what it held."""
    w = _vessel(compiled)
    w.matter.mpm.v[:, 2] = -6.0
    return w


@pytest.mark.parametrize("build,ticks", [(_pole, 80), (_thrown_vessel, 40)],
                         ids=["pole", "vessel"])
def test_DRAWING_ONLY_THE_CELLS_THAT_HOLD_PARTICLES_draws_the_SAME_LATTICE(build, ticks):
    """Rasterize used to sum over arrays the size of the world (33.8 ms a
    tick at 400k cells) and now sums over the cells the particles are in.
    Every tick of a falling pole and of a vessel that breaks and pours, the
    world-sized version is run on a copy of what the new one is given, and
    the lattice, the seam's own masks and the particles must come out the
    same bits."""
    from types import SimpleNamespace as NS
    w = build(True)
    mt = w.matter
    real = mt.rasterize
    seen = {"ticks": 0, "poured": 0}

    def checked():
        lat = NS(shape=w.shape, **{f: getattr(w, f).copy() for f in _LATTICE})
        p = mt.mpm
        part = NS(x=p.x, inv_dx=p.inv_dx, m=p.m, vmat=p.vmat, E=p.E, dmg=p.dmg,
                  edge=p.edge, fpot=p.fpot, fvol=p.fvol.copy(), fl=p.fl.copy())
        drawn, dfl = mt.drawn.copy(), mt.drawn_fluid.copy()
        seen["poured"] += _rasterize_world_sized(lat, part, drawn, dfl)
        real()
        for f in _LATTICE:
            assert np.array_equal(getattr(w, f), getattr(lat, f)), f
        assert np.array_equal(mt.drawn, drawn) and np.array_equal(mt.drawn_fluid, dfl)
        assert np.array_equal(p.fvol, part.fvol) and np.array_equal(p.fl, part.fl)
        seen["ticks"] += 1

    mt.rasterize = checked
    for _ in range(ticks):
        w.step()
    assert seen["ticks"] >= ticks // 2, "the seam was hardly used"
    if build is _thrown_vessel:
        assert seen["poured"], "the vessel never poured, so the spill path was not tested"


def _brute_cell_eids(cells, vmat, kg, eid):
    """The rule of design 3.0 item 6 written as plainly as it reads: per cell,
    the material with the most mass (a tie to the lower id), then among that
    material's particles the eid with the most mass (a tie to the lower eid).
    Masses are added one particle at a time, in particle order."""
    out = {}
    for c in sorted(set(cells)):
        here = [i for i, ci in enumerate(cells) if ci == c]
        g = {}
        for i in here:
            g[int(vmat[i])] = g.get(int(vmat[i]), 0.0) + float(kg[i])
        mat = min(v for v in g if g[v] == max(g.values()))
        ge = {}
        for i in here:
            if vmat[i] == mat:
                ge[int(eid[i])] = ge.get(int(eid[i]), 0.0) + float(kg[i])
        out[c] = (mat, min(e for e in ge if ge[e] == max(ge.values())))
    return out


def test_a_DRAWN_CELL_BELONGS_TO_the_HEAVIEST_THING_OF_ITS_OWN_MATERIAL():
    """Thirty random clouds, three materials and five eids (0 is "no
    thing") packed several to a cell, with masses that tie exactly as often
    as they can (quarters of a kilogram add up without rounding). Every drawn
    cell's material and eid must be what the rule says, and every cell not
    drawn must have no eid."""
    rng = np.random.default_rng(3)
    for trial in range(30):
        w = World(10, 10, 10)
        w.eid = np.zeros(w.shape, np.int32)
        m = w.matter.mpm
        k = int(rng.integers(20, 400))
        pts = rng.uniform(0.2, 0.6, (k, 3))          # cells 2..5 on each axis
        m._append(pts, 0.0, 1e-3, 1.0, 0.3, 0.0, 0.0, 1.0, np.zeros(k), 0)
        m.m[:] = rng.choice([0.25, 0.5, 0.75], k)
        m.vmat[:] = rng.integers(1, 4, k)
        m.eid[:] = rng.integers(0, 5, k)
        w.matter.rasterize()
        cells = [tuple(int(v) for v in c) for c in (pts * m.inv_dx).astype(int)]
        want = _brute_cell_eids(cells, m.vmat, m.m, m.eid)
        for c, (mat, e) in want.items():
            assert (int(w.mat[c]), int(w.eid[c])) == (mat, e), (trial, c)
        drawn = np.zeros(w.shape, bool)
        drawn[tuple(np.array(list(want)).T)] = True
        assert np.array_equal(drawn, w.matter.drawn), trial
        assert not w.eid[~drawn].any(), trial


def test_TWO_PROMOTED_THINGS_STAY_TWO_THINGS():
    """Two blocks, stamped as two things, leave the lattice in two promote
    calls; one falls onto the other and both come to rest. For 100 ticks
    every particle keeps its thing and its batch, the drawn cells carry
    exactly those two eids and no eid is left on an empty cell, and
    `entities()` reports each thing whole: its grams, its cells, whether it
    is still moving."""
    w = World(16, 12, 16)
    w.fill(0, 16, 0, 12, 0, 1, STONE)
    w.eid = np.zeros(w.shape, np.int32)
    w.fill(5, 9, 4, 8, 1, 3, STONE)
    w.eid[5:9, 4:8, 1:3] = 7
    w.fill(6, 8, 5, 7, 8, 10, WOOD)
    w.eid[6:8, 5:7, 8:10] = 9
    mt = w.matter
    mt.promote(np.argwhere(w.eid == 7))
    mt.promote(np.argwhere(w.eid == 9))
    m = mt.mpm
    first, second = m.eid == 7, m.eid == 9
    g7, g9 = float(m.m[first].sum()) * 1000.0, float(m.m[second].sum()) * 1000.0
    assert first.sum() == 32 and second.sum() == 8
    assert (m.batch[first] == 0).all() and (m.batch[second] == 1).all()
    for t in range(100):
        w.step()
        assert (m.eid[first] == 7).all() and (m.eid[second] == 9).all(), t
        assert set(np.unique(w.eid[mt.drawn]).tolist()) == {7, 9}, t
        assert not w.eid[w.mat == AIR].any(), t
        assert not w.eid[~mt.drawn].any(), t
        ent = mt.entities()
        assert set(ent) == {7, 9}, t
        assert ent[7]["mix"] == {int(STONE): pytest.approx(g7, rel=1e-12)}, t
        assert ent[9]["mix"] == {int(WOOD): pytest.approx(g9, rel=1e-12)}, t
        for e in (7, 9):
            got = {tuple(c) for c in ent[e]["cells"].tolist()}
            assert got == {tuple(c) for c in np.argwhere(w.eid == e).tolist()}, (t, e)
            assert ent[e]["particles"] == (32 if e == 7 else 8)
        if t == 2:
            assert ent[9]["aloft"], "the upper block should still be falling"
    assert not ent[7]["aloft"] and not ent[9]["aloft"], "both should lie still"
    # the upper block came to rest on the lower one, not through it
    assert ent[9]["centroid"][2] > ent[7]["centroid"][2] + 1.0
    lo, hi = ent[9]["lo"], ent[9]["hi"]
    assert ((hi - lo) >= 2).all() and ((hi - lo) <= 4).all(), (lo, hi)


def test_a_VESSEL_REPORTS_WHAT_IT_CARRIES():
    """`entities()` counts the fluid riding on a promoted vessel, which is
    what the ledger counts, and nothing the lattice still holds."""
    w = _vessel(True)
    ent = w.matter.entities()
    assert set(ent) == {0}
    assert ent[0]["fluid"] == {int(WATER): pytest.approx(w.matter.fluid(WATER), rel=1e-12)}
    assert ent[0]["fluid"][int(WATER)] > 800.0
    assert ent[0]["mix"] == {int(GLASS): pytest.approx(w.total_mass(GLASS), rel=1e-12)}
    assert len(ent[0]["cells"]) == 0, "nothing is drawn before the first rasterize"


# ------------------------------------------------- what a promoted thing is


def test_the_SAME_THING_IS_EQUALLY_STRONG_WHEREVER_IT_LEAVES_THE_LATTICE():
    """A promoted cluster's flaws were read off one field at its cells' WORLD
    position, so the same pane toppled from a shelf and from the floor was
    made of different glass (the bug test_mpm.py pins for `add_box`). They
    are read from the thing's own corner now, in a field seeded by its eid:
    the same pane at two heights is the same glass, and a different thing
    of the same shape is not."""
    def pane(x0, y0, z0, eid=None):
        w = World(20, 14, 20)
        w.fill(0, 20, 0, 14, 0, 1, STONE)
        w.fill(x0, x0 + 5, y0, y0 + 3, z0, z0 + 2, GLASS)
        if eid is not None:
            w.eid = np.zeros(w.shape, np.int32)
            w.eid[w.mat == GLASS] = eid
        w.matter.promote(np.argwhere(w.mat == GLASS))
        return w.matter.mpm.strength
    low, high = pane(3, 3, 2), pane(11, 8, 13)
    assert len(low) == 30 and (low > 0.0).all()
    assert low == pytest.approx(high, rel=1e-12)
    assert np.unique(low).size > 10, "the pane has flaws, not one strength"
    a, b, c = pane(3, 3, 2, eid=5), pane(11, 8, 13, eid=5), pane(3, 3, 2, eid=6)
    assert a == pytest.approx(b, rel=1e-12)
    assert not np.allclose(a, c), "another thing is another piece of glass"


def test_a_HALF_FULL_CELL_IS_LESS_WOOD_NOT_LIGHTER_WOOD():
    """A promoted particle used to take its whole cell's volume whatever the
    cell held, so the felled tree's 40%-full spokes and canopy were
    materials of 0.4 times their density: sound in them ran 1.58 times too
    fast, their viscous step was 2.5 times too short, and the tree cost 215
    substeps a tick (design 3.2, P4). Each particle's density is now its
    material's, and the tree costs what its materials cost: 108 substeps,
    which is wood's viscous limit (86) with the margin `cfl_dt` keeps."""
    from src.voxel.scenes import _grow_tree
    from src.mpm.core import MAT
    from src.voxel.sim import TICK_S, LEAF
    w = World(40, 40, 70, voxel_cm=5)
    w.fill(0, 40, 0, 40, 0, 1, STONE)
    _grow_tree(w, 20, 20)
    full = w.smass[w.mat == WOOD].max()
    assert (w.smass[w.mat == WOOD] < 0.5 * full).any(), "no part-full spokes"
    # this is the CONTINUUM's rule: a crown would go to bonds (src/bonds),
    # so the tree is sent to the particles on purpose
    w.matter.bonds_on = False
    w.matter.promote(np.argwhere(w.mat[:, :, 1:] != AIR) + [0, 0, 1])
    m = w.matter.mpm
    for vid, name in ((int(WOOD), "wood"), (int(LEAF), "leaf")):
        rho = m.m[m.vmat == vid] / m.vol0[m.vmat == vid]
        assert rho == pytest.approx(MAT[name][0], rel=1e-12), name
    assert int(np.ceil(TICK_S / m.cfl_dt())) == 108
    # the same as two full cells of the same two materials
    v = World(8, 8, 8, voxel_cm=5)
    v.fill(2, 3, 2, 3, 2, 3, WOOD)
    v.fill(3, 4, 2, 3, 2, 3, LEAF)
    v.matter.bonds_on = False
    v.matter.promote(np.argwhere(v.mat != AIR))
    assert v.matter.mpm.cfl_dt() == m.cfl_dt()


# ------------------------------------------------ what the lattice gives back
#
# Between `rasterize` and `unrasterize` the laws work on the drawn cells like
# any others: conduction, fire, melting. `unrasterize` hands each change back
# to the particles drawn there (design 3.2, P6). The scenes are the design's
# prototypes (design/mpm2/writeback_probe.py, review/drawn_shove.py).


def _torch_block(promote):
    """A 3x2x2 block of wood (900 g at 5 cm) lying on a stone floor, left
    on the lattice or promoted, and given ten ticks to settle."""
    w = World(16, 10, 12, voxel_cm=5)
    w.fill(0, 16, 0, 10, 0, 1, STONE)
    w.fill(6, 9, 4, 6, 1, 3, WOOD)
    if promote:
        w.matter.promote(np.argwhere(w.mat == WOOD))
    for _ in range(10):
        w.step()
    return w


def _burned(w):
    """Grams of the block, whatever fire has made of it."""
    return sum(w.total_mass(m) for m in (WOOD, CHAR, ASH))


def test_the_SEAM_KEEPS_THE_HEAT_IT_IS_GIVEN():
    """A torch held to the block's end, 300 J a tick into its end cell for
    60 ticks: too little to light it (about 160 C at most), so the energy
    ledger is all there is to read (the heat of burning is not in
    `total_energy`). A promoted block drifted -22,883 J: the heat put into
    its drawn cells was emptied with them every tick. Now its drift is the
    lattice block's to within a joule, read between every two ticks, with
    the hand-back still pending.

    The torch used to put 1000 J a tick into the AIR cell beside the block.
    Air holds 1.2 J/(L K) now (realism G5), 0.15 J/K in a 5 cm cell, so that
    was some 6,700 K a tick; the heat goes into the wood instead. The
    lattice's own ledger used to drift (-2,889 J, design 3.2): it was the
    plume leaving by the open top, uncounted. The sky loss is counted now
    (`shed`), so the lattice ledger must close too."""
    a, b = _torch_block(False), _torch_block(True)
    ea, eb = a.total_energy(), b.total_energy()
    put, worst = 0.0, 0.0
    for t in range(80):
        if t < 60:
            a.E[6, 4, 1] += 300.0
            b.E[6, 4, 1] += 300.0
            put += 300.0
        a.step()
        b.step()
        worst = max(worst, abs((b.total_energy() - eb) - (a.total_energy() - ea)))
    assert worst < 1.0, worst
    assert abs(a.total_energy() - ea - put) < 1.0, \
        "the lattice ledger is exact: its drift was the plume leaving by the open top, uncounted"
    assert len(b.matter) == 12 and int(b.matter.drawn.sum()) == 12
    assert b.matter.energy() > 10000.0, "the particles did not take the heat"


def test_a_PROMOTED_BLOCK_BURNS():
    """The same block, alight. The particles were drawn again every tick
    from matter that had not changed, so a promoted block never burned: 900 g
    of wood before and after, no smoke. Now the lattice burns the drawn cells
    and the particles lose what burned, and a cell the fire turned to char
    turns its particles to char, in the solver too.

    On the real fire clock (realism G4) this block burns about 1.3 g a
    second, so it cannot lose the 100 g it lost on the old clock (the
    prototype: 900 -> 737.5 g) in a test's few seconds. So the block starts
    alight through, as a fire some minutes old, and it is held to its TWIN
    left on the lattice in the same fire: the promoted block must burn the
    grams the lattice block burns and make the smoke it makes. Measured in
    120 ticks: 3.895 g burned and 0.266 g of smoke in both, to 1e-4 g."""
    def lit_through(w):
        block = w.mat == WOOD
        w.E[block] = 430.0 * w.heat_capacity()[block]

    a, w = _torch_block(False), _torch_block(True)
    g0, smoke0 = _burned(w), float(w.smoke.sum())
    assert g0 == pytest.approx(900.0, rel=1e-9)
    assert _burned(a) == pytest.approx(g0, rel=1e-9)   # the same block
    assert float(a.smoke.sum()) == smoke0
    lit_through(a)
    lit_through(w)
    lit = 0
    for t in range(120):
        a.step()
        w.step()
        lit = max(lit, int((w.burning() & w.matter.drawn).sum()))
    assert lit > 0, "no drawn cell ever burned"
    gone, twin = g0 - _burned(w), g0 - _burned(a)
    assert twin > 1.0, "the lattice block did not burn: the scene is wrong"
    assert gone == pytest.approx(twin, abs=0.01), (gone, twin)
    left = float(w.matter.mpm.m.sum()) * 1000.0
    assert left < g0 - 1.0, "the particles kept their grams"
    assert left == pytest.approx(_burned(w), abs=0.1), "the particles are not what the lattice left"
    smoke, smoke_twin = float(w.smoke.sum()) - smoke0, float(a.smoke.sum()) - smoke0
    assert smoke_twin > 0.1
    assert smoke == pytest.approx(smoke_twin, rel=1e-3), (smoke, smoke_twin)


def test_MOVING_DRAWN_MATTER_ON_THE_LATTICE_DOES_NOT_COPY_IT():
    """A hand-move (`_shove`) of a promoted block's drawn cells took 600 g of
    wood to 1,200 g: the lattice kept the moved copy, and the particles were
    never told. The movers refuse a drawn cell now (E0). And a law that moved
    one anyway would now move the matter, not copy it: the cells it left
    are air, so the particles drawn there are removed, and the lattice keeps
    what arrived where it arrived."""
    w = World(20, 12, 16, voxel_cm=5)
    w.fill(0, 20, 0, 12, 0, 1, STONE)
    w.fill(8, 10, 5, 7, 1, 3, WOOD)
    w.matter.promote(np.argwhere(w.mat == WOOD))
    for _ in range(15):
        w.step()
    m0 = w.total_mass(WOOD)
    assert m0 == pytest.approx(600.0, rel=1e-9)
    cells = np.argwhere((w.mat == WOOD) & w.moving).astype(np.int32)
    assert len(cells) == 8
    assert not w._shove(cells, 3, 0, 0), "a hand moved drawn matter"
    w.step()
    assert w.total_mass(WOOD) == pytest.approx(m0, rel=1e-12)
    # the guard taken away: the move goes through on the lattice
    cells = np.argwhere((w.mat == WOOD) & w.moving).astype(np.int32)
    w.moving[tuple(cells.T)] = False
    assert w._shove(cells, 3, 0, 0)
    assert w.total_mass(WOOD) == pytest.approx(m0, rel=1e-6)    # still pending
    for t in range(5):
        w.step()
        assert w.total_mass(WOOD) == pytest.approx(m0, rel=1e-6), t
    assert len(w.matter) == 0, "the particles the move left behind are still there"
    assert int((w.mat == WOOD).sum()) == 8


def _iron_slab():
    """A 2x2 slab of iron (975 g a cell at 5 cm) promoted onto a stone
    floor, settled, and still drawn."""
    w = World(12, 12, 14, voxel_cm=5)
    w.fill(0, 12, 0, 12, 0, 1, STONE)
    w.fill(5, 7, 5, 7, 1, 2, IRON)
    w.matter.promote(np.argwhere(w.mat == IRON))
    for _ in range(5):
        w.step()
    assert int(w.matter.drawn.sum()) == 4
    return w


@pytest.mark.parametrize("hot", [1650.0, 2200.0], ids=["part", "whole"])
def test_MELTING_A_DRAWN_IRON_CELL_KEEPS_THE_IRON(hot):
    """The drawn cells of an iron slab are heated past iron's melting point.
    The melt law turns grams of iron into millilitres of molten iron in
    those cells; the particles were never told, so the slab never melted.
    Now the particles lose the grams and carry the melt, and a cell that
    melted through is handed to the lattice with the melt in it. Grams of
    iron plus grams of molten iron hold at every tick."""
    w = _iron_slab()
    d = w.matter.drawn.copy()
    w.E[d] = (hot - 20.0) * w.heat_capacity()[d]

    def iron():
        return w.total_mass(IRON) + FLUID[MIRON][0] * w.total_fluid(MIRON)

    g0 = iron()
    melted, carried, n0 = 0.0, 0.0, len(w.matter)
    for t in range(30):
        w.step()
        assert iron() == pytest.approx(g0, abs=0.5), t
        melted = max(melted, FLUID[MIRON][0] * w.total_fluid(MIRON))
        carried = max(carried, float(w.matter.mpm.fvol[w.matter.mpm.fl == MIRON].sum()))
    assert melted > 100.0, "nothing melted"
    if hot < 2000.0:
        assert carried > 10.0, "the particles never carried the melt"
    else:
        assert len(w.matter) < n0, "a cell melted through and its particles stayed"


def test_with_NOTHING_DONE_TO_IT_THE_SEAM_HANDS_BACK_NOTHING():
    """No heat and no fire: a vessel of water thrown down so that it breaks
    and pours. Nothing the lattice does between two ticks touches a drawn
    cell, so there is nothing to hand back, and every tick the particles
    are exactly what the solver left (the golden cases pole_mpm and
    seam_fall hold this bit for bit)."""
    w = _thrown_vessel(True)
    for t in range(40):
        w.step()
        assert w.matter._written() is None, t
    assert w.matter.mpm.broken(), "the vessel did not break"


# ------------------------------------------------------- what a vessel holds


def test_a_VESSELS_CAVITY_is_WHAT_IT_CARRIES_or_COULD_HOLD():
    """`_cavity` is the rule `promote` carries a vessel's fluid by
    (tests/test_vessels.py pins what it carries, shape by shape), asked on
    its own. With `need_fluid` it is the cells of the hollow that hold
    fluid now; without, the whole hollow, whatever is in it: what the
    vessel could hold. A lid changes neither. A tube open at both ends
    walls nothing in."""
    def cup(lid=False):
        w = World(10, 10, 20)
        w.fill(0, 10, 0, 10, 0, 1, STONE)
        w.fill(2, 8, 2, 8, 1, 2, GLASS)
        w.fill(2, 8, 2, 8, 2, 7 if lid else 6, GLASS)
        w.mat[3:7, 3:7, 2:6] = AIR
        w.smass[3:7, 3:7, 2:6] = 0.0
        return w

    hollow = np.zeros((10, 10, 20), bool)
    hollow[3:7, 3:7, 2:6] = True
    for lid in (False, True):
        w = cup(lid)
        glass = np.argwhere(w.mat == GLASS)
        assert np.array_equal(w.matter._cavity(glass, need_fluid=False), hollow)
        assert not w.matter._cavity(glass, need_fluid=True).any(), "it is empty"
        for x in range(3, 7):
            for y in range(3, 7):
                for z in (2, 3):
                    w.pour(x, y, z, WATER, w.cap)
        wet = np.zeros_like(hollow)
        wet[3:7, 3:7, 2:4] = True
        assert np.array_equal(w.matter._cavity(glass, need_fluid=True), wet)
        assert np.array_equal(w.matter._cavity(glass, need_fluid=False), hollow)
        # in any order, and it is what `promote` then carries
        assert np.array_equal(w.matter._cavity(glass[::-1], need_fluid=True), wet)
        held = float(w.fvol[wet].sum())
        w.matter.promote(glass)
        assert w.matter.fluid(WATER) == pytest.approx(held, rel=1e-12)
    w = World(20, 10, 10)
    w.fill(0, 20, 0, 10, 0, 1, STONE)
    w.fill(2, 18, 3, 7, 2, 6, GLASS)
    w.mat[2:18, 4:6, 3:5] = AIR
    w.smass[2:18, 4:6, 3:5] = 0.0
    pipe = np.argwhere(w.mat == GLASS)
    assert not w.matter._cavity(pipe, need_fluid=False).any()
    assert not w.matter._cavity(np.zeros((0, 3), np.int64), need_fluid=False).any()


# ------------------------------------------------------- sleep (design P8)
#
# A thing that has come to rest leaves the solve and is still drawn on the
# lattice and still handed what the laws do to it there (`Matter._doze`,
# `_wake`). The four checks are the design's (step X5); the last two pin the
# wake rules the design names that the four do not reach.


def _asleep_from(w, ticks, each=None):
    """Step `w` `ticks` times; the first tick after which all of its
    particle matter sleeps, or None."""
    first = None
    for t in range(ticks):
        if each is not None:
            each(t)
        w.step()
        if first is None and len(w.matter) and w.matter.asleep().all():
            first = t
    return first


def test_the_POLE_GOES_TO_SLEEP_and_then_COSTS_THE_SOLVER_NOTHING():
    """The hinged pole lands at tick 37 and slides to rest; its mean speed
    is under STILL from tick 50 (measured 0.0077 m/s), and after REST ticks
    of that it sleeps: from tick 52 on (measured). From then on the solver
    is never called. `matter.step` cost 37 ms a tick while it moved and
    0.04 ms asleep (measured), and the pole is still on the lattice, every
    gram of it, where the laws and the minds can see it."""
    import time
    w = _pole(kernels.HAVE_NUMBA)
    m = w.matter
    solved, cost, now = [], [], [0]
    solve, step = m.mpm.step, m.step

    def counted(dt, *a, **k):
        solved.append(now[0])
        return solve(dt, *a, **k)

    def timed(dt):
        t0 = time.perf_counter()
        n = step(dt)
        cost.append((time.perf_counter() - t0, n))
        return n

    m.mpm.step, m.step = counted, timed
    g0 = None
    first = None
    for t in range(80):
        now[0] = t
        w.step()
        if g0 is None and len(m):
            g0 = w.total_mass(WOOD)
        if first is None and len(m) and m.asleep().all():
            first = t
            drawn = int(m.drawn.sum())
    assert first is not None and 49 <= first <= 56, first
    assert max(solved) == first, "the solver ran after the pole slept"
    after = [s for s, _ in cost[first + 1:]]
    assert all(n == 0 for _, n in cost[first + 1:])
    assert float(np.median(after)) < 2e-3, np.median(after)
    assert float(np.median([s for s, _ in cost[40:50]])) > 10 * float(np.median(after))
    ent = m.entities()[0]
    assert ent["asleep"] and not ent["aloft"] and not m.aloft()
    assert int(m.drawn.sum()) == drawn > 0, "a sleeper is still drawn"
    assert w.total_mass(WOOD) == pytest.approx(g0, rel=1e-12)


def test_a_SLEEPING_POLE_LIES_WHERE_a_POLE_THAT_NEVER_SLEEPS_LIES():
    """At tick 80, 28 ticks after it fell asleep, every particle of the
    pole is within a cell of where the same pole lies with sleep off
    (measured: 0.02 of a cell). Sleep only stops a twitch of a fifth of a
    millimetre a second."""
    a, b = _pole(kernels.HAVE_NUMBA), _pole(kernels.HAVE_NUMBA)
    b.matter.sleep = False
    assert _asleep_from(a, 80) is not None
    assert _asleep_from(b, 80) is None
    off = np.abs(a.matter.mpm.x - b.matter.mpm.x).max() * a.matter.mpm.inv_dx
    assert off < 1.0, off


def _warm_cup(sleep):
    """A glass cup holding 800 ml of water, its glass warmed to 90 C and
    promoted where it stands on a stone floor."""
    w = World(12, 12, 14)
    w.fill(0, 12, 0, 12, 0, 1, STONE)
    w.fill(4, 8, 4, 8, 1, 4, GLASS)
    w.mat[5:7, 5:7, 2:4] = AIR
    w.smass[5:7, 5:7, 2:4] = 0.0
    for x in (5, 6):
        for y in (5, 6):
            w.pour(x, y, 2, WATER, 200.0)
    glass = np.argwhere(w.mat == GLASS)
    w.E[tuple(glass.T)] = 70.0 * w.heat_capacity()[tuple(glass.T)]   # 90 °C
    w.matter.promote(glass)
    w.matter.sleep = sleep
    return w


def test_the_LEDGERS_ARE_EXACT_ACROSS_SLEEP_AND_WAKE():
    """The warm cup sleeps (tick 7, measured), is thrown up 1 m/s at tick
    35, which wakes it, and lands and sleeps again (tick 48). All the while
    it gives its heat to the floor: a sleeper is still handed every joule
    the lattice moves, and heat does not wake it (the solver never reads
    it). Every tick, its glass and its water are the grams and millilitres
    it started with, exactly, and the whole world's energy is the same
    number, bit for bit, as in the same world with sleep off. (That number
    drifts in both: the lattice's own drift, design 3.2.)"""
    a, b = _warm_cup(True), _warm_cup(False)
    g0, f0 = a.total_mass(GLASS), a.total_fluid(WATER)
    slept, heat = [], []
    for t in range(70):
        if t == 35:
            a.matter.mpm.v[:, 2] += 1.0
            b.matter.mpm.v[:, 2] += 1.0
        a.step()
        b.step()
        slept.append(bool(a.matter.asleep().all()))
        heat.append(a.matter.energy())
        assert a.total_mass(GLASS) == g0, t
        assert a.total_fluid(WATER) == f0, t
        assert a.total_energy() == b.total_energy(), t
    assert not any(b.matter.asleep())
    assert slept[34] and not slept[35], "the throw did not wake it"
    assert any(slept[:35]) and any(slept[36:]), slept
    nap = [t for t in range(35) if slept[t]]
    assert heat[nap[-1]] < heat[nap[0]] - 100.0, "the sleeper was not handed its heat"


def _plank_on_post(sleep):
    """A 4x2 plank of wood promoted where it lies on a stone post of its
    own footprint, five cells tall."""
    w = World(16, 10, 16)
    w.fill(0, 16, 0, 10, 0, 1, STONE)
    w.fill(5, 9, 4, 6, 1, 6, STONE)
    w.fill(5, 9, 4, 6, 6, 7, WOOD)
    w.matter.promote(np.argwhere(w.mat == WOOD))
    w.matter.sleep = sleep
    return w


def test_REMOVING_THE_SHELF_UNDER_A_SLEEPING_PLANK_makes_it_FALL_THE_SAME_TICK():
    """The plank sleeps on its post. The post is taken away between two
    ticks, and in the very next tick the plank falls, as fast as the same
    plank that never slept (one tick of free fall, g dt = 0.245 m/s), and
    both land on the floor."""
    a, b = _plank_on_post(True), _plank_on_post(False)
    assert _asleep_from(a, 15) is not None, "it never slept"
    _asleep_from(b, 15)
    for w in (a, b):
        w.mat[5:9, 4:6, 1:6] = AIR
        w.smass[5:9, 4:6, 1:6] = 0.0
        w.step()
    va, vb = (float(w.matter.mpm.v[:, 2].mean()) for w in (a, b))
    assert not a.matter.asleep().any()
    assert va < -0.2 and abs(va - vb) < 0.01, (va, vb)
    for w in (a, b):
        for _ in range(30):
            w.step()
        assert int(np.argwhere(w.mat == WOOD)[:, 2].max()) == 1, "it did not land"


def test_a_BLOCK_DROPPED_ON_A_SLEEPING_BLOCK_WAKES_IT_and_SAYS_WHAT_IT_DID():
    """A stone block settles on the floor, asleep after tick 6 (measured:
    promoted where it lies, it first settles onto the floor), and a wooden
    block falls on it from about a metre. The stone wakes before the wood can reach
    it (an awake particle comes within APART nodes, plus the step's travel),
    and what the tap reports is what it reports with sleep off: one touch
    and one blow, in the same tick, the blow against the stone."""
    def build(sleep):
        w = World(16, 12, 24)
        w.fill(0, 16, 0, 12, 0, 1, STONE)
        w.fill(5, 9, 4, 8, 1, 3, STONE)
        w.eid[5:9, 4:8, 1:3] = 7
        w.matter.promote(np.argwhere(w.eid == 7))
        w.matter.sleep = sleep
        return w

    a, b = build(True), build(False)
    for w in (a, b):
        for _ in range(8):
            w.step()
    assert a.matter.asleep().all(), "the stone block never slept"
    for w in (a, b):
        w.fill(6, 8, 5, 7, 12, 14, WOOD)
        w.eid[6:8, 5:7, 12:14] = 9
        w.matter.promote(np.argwhere(w.eid == 9))
    seen, nap = {}, []
    for w in (a, b):
        ev = []
        for _ in range(30):
            w.step()
            ev += [(r["kind"], r["tick"], r["eid"], r.get("against"))
                   for r in w.matter.events() if r["kind"] in ("touch", "struck")]
            if w is a:
                stone = a.matter.mpm.eid == 7
                nap.append((a.tick - 1, bool(a.matter.asleep()[stone].all())))
        seen[w is a] = ev
    assert seen[True] == seen[False], seen
    assert sorted(k for k, *_ in seen[True]) == ["struck", "touch"]
    hit = seen[True][0][1]
    assert ("struck", hit, 9, ((7, 0),)) in seen[True]
    assert nap[0][1], "the stone woke before the wood came near"
    assert not dict(nap)[hit], "the stone slept through the blow"


def test_what_the_LATTICE_DOES_TO_A_SLEEPER_WAKES_IT_ONLY_WHEN_THE_SOLVER_WOULD_FEEL_IT():
    """The block from the torch tests, asleep on the floor. Heat held to it
    for ten ticks is handed to it and it sleeps on. A drawn cell that loses
    half its grams (as burning takes them) changes what the solver reads,
    and wakes it in the next tick."""
    w = _torch_block(True)
    m = w.matter
    assert m.asleep().all(), "it never slept"
    e0 = m.energy()
    for _ in range(10):
        w.E[5, 4, 1] += 1000.0
        w.step()
    assert m.asleep().all() and m.energy() > e0 + 1000.0
    cell = m.drawn_at[0]
    w.smass.ravel()[cell] *= 0.5
    w.step()
    assert not m.asleep().any()
