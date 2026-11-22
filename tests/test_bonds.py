"""BONDS behind the seam: things that must part into pieces.

The user's pick (2026-10-10) is BOTH. A promoted cluster that must PART
(a crown of leaves, a rope, a pane) is carried as rigid pieces joined by
bonds that bend and break (src/bonds); everything else stays on the
continuum (src/mpm), which lands a solid thing whole. These tests pin the
rule that sends a thing one way or the other, what each kind does, and that
the ledgers stay exact through promote, step, break and write-back.

Every number here was measured before it was asserted.
"""

import numpy as np
import pytest

from src.bonds.materials import PETIOLE_R
from src.bonds.store import claims
from src.mpm.core import GRAVITY
from src.voxel.materials import ROPE
from src.voxel.sim import World, AIR, STONE, WOOD, GLASS, IRON, LEAF, WATER

# 12 mm manila is 5 g in a 5 cm cell: this share of a cell of ROPE
MANILA = 5.0 / (885.0 * 0.125)


def _cells(w, mats):
    return np.argwhere(np.isin(w.mat, mats))


def _plain(v):
    """A record holds only what any reader can take: no numpy anywhere."""
    if isinstance(v, dict):
        return all(isinstance(k, str) and _plain(x) for k, x in v.items())
    if isinstance(v, tuple):
        return all(_plain(x) for x in v)
    return type(v) in (int, float, str, bool)


# ── the rule ─────────────────────────────────────────────────────────────────

def _small_tree(w, cx, cy, z0, top, r, sap=True):
    """A trunk two cells square up to `top`, four spokes, and a leaf ball
    of radius `r` round them (5 cm cells): `_grow_tree` in small. Its
    leaves are fresh (holding sap) or, without `sap`, dry."""
    w.fill(cx - 1, cx + 1, cy - 1, cy + 1, z0, top, WOOD)
    X, Y, Z = np.ogrid[:w.shape[0], :w.shape[1], :w.shape[2]]
    ball = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - top - 1) ** 2 <= r * r
    ball &= w.mat == AIR
    w.mat[ball] = LEAF
    w.smass[ball] = 200.0 * 0.4 * w.vox_l
    if sap:
        w.fl[ball] = WATER
        w.fvol[ball] = 0.06 * w.cap
    for bx, by in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for t in range(1, r):
            px, py = cx + bx * t, cy + by * t
            w.fill(px, px + 1, py, py + 1, top - 1, top, WOOD, frac=0.4)


def test_THE_RULE_sends_what_must_PART_to_bonds_and_the_rest_to_the_continuum():
    """A crown (leaf on wood), a rope (ROPE, the material) and a pane (a
    thin full sheet of glass) go to bonds. A plank, a pole, a thin dowel (a
    wood strand as thin as the rope: rope is a material, not a fill), a
    glass bar, a vat, a leaf hedge with no wood, and a named crown when
    named things are not let in, do not."""
    w = World(40, 40, 40, voxel_cm=5)
    _small_tree(w, 10, 10, 1, 12, 4)
    assert claims(w, _cells(w, (WOOD, LEAF))) == "crown"
    w = World(30, 10, 30, voxel_cm=5)
    w.fill(10, 11, 5, 6, 5, 25, ROPE, frac=MANILA)         # 12 mm manila
    w.fill(10, 12, 5, 7, 4, 5, IRON)
    assert claims(w, _cells(w, (ROPE, IRON))) == "rope"
    w = World(30, 10, 30, voxel_cm=5)
    w.fill(10, 11, 5, 6, 5, 25, WOOD, frac=0.067)          # a dowel, as thin
    w.fill(10, 12, 5, 7, 4, 5, IRON)
    assert claims(w, _cells(w, (WOOD, IRON))) is None
    w = World(30, 10, 30, voxel_cm=5)
    w.fill(10, 11, 5, 6, 5, 25, WOOD, frac=0.4)            # a stick, not a rope
    assert claims(w, _cells(w, (WOOD,))) is None
    w = World(30, 20, 30, voxel_cm=5)
    w.fill(5, 25, 5, 13, 10, 11, GLASS)                    # a pane
    assert claims(w, _cells(w, (GLASS,))) == "pane"
    w = World(30, 20, 30, voxel_cm=5)
    w.fill(5, 8, 5, 6, 1, 25, GLASS)                       # 1 x 3 x 24: a bar
    assert claims(w, _cells(w, (GLASS,))) is None
    w = World(30, 20, 30, voxel_cm=5)
    w.fill(5, 9, 5, 9, 5, 9, GLASS)                        # a vat
    w.mat[6:8, 6:8, 6:9] = AIR
    assert claims(w, _cells(w, (GLASS,))) is None
    w = World(30, 20, 30, voxel_cm=5)
    w.fill(5, 25, 5, 13, 10, 12, WOOD)                     # a plank
    assert claims(w, _cells(w, (WOOD,))) is None
    w = World(30, 20, 30, voxel_cm=5)
    w.fill(5, 25, 5, 8, 1, 20, LEAF, frac=0.3)             # a hedge
    assert claims(w, _cells(w, (LEAF,))) is None
    w = World(40, 40, 40, voxel_cm=5)
    _small_tree(w, 10, 10, 1, 12, 4)
    w.eid[w.mat != AIR] = 5                                # a named tree
    assert claims(w, _cells(w, (WOOD, LEAF))) is None
    assert claims(w, _cells(w, (WOOD, LEAF)), named=True) == "crown"


# ── the crown ────────────────────────────────────────────────────────────────

def _fell(sap):
    """A small tree, cut at its foot and pushed over onto its crown. Which
    leaf pieces came off it (no wood in their fragment) and when, how fast
    each was falling once it had been loose for 0.25 s and was still in the
    air, and the bonds."""
    w = World(60, 30, 40, voxel_cm=5)
    w.fill(0, 60, 0, 30, 0, 1, STONE)
    _small_tree(w, 12, 15, 1, 22, 4, sap)
    tree = _cells(w, (WOOD, LEAF))
    w.matter.promote(tree[tree[:, 2] >= 2])
    b = w.matter.bonds
    leafp = np.unique(b.mp[b.vmat == LEAF])
    b.kick(np.flatnonzero(~b.pkin), omega=(0.0, 0.6, 0.0),
           about=np.array([12.0, 15.0, 2.0]) * w.matter.dx)
    off_at, vz = {}, []
    for t in range(80):
        w.step()
        fr = b.fragments()
        wood = set(fr[b.mp[b.vmat == WOOD]].tolist())
        for p in leafp:
            if fr[p] not in wood:
                off_at.setdefault(int(p), t)
        loose = [p for p, t0 in off_at.items() if t - t0 >= 10 and b.px[p, 2] > 0.1]
        vz += b.pv[loose, 2].tolist()
    return b, leafp, off_at, vz


def test_a_FELLED_CROWN_SHEDS_SPRAYS_THAT_DRIFT_not_one_block():
    """The user's complaint: a felled tree's crown came down as ONE block of
    leaf. Here a small tree, cut at its foot and pushed over, lands on its
    crown. Its leaves are pieces, each on its own stalk. A dry crown sheds
    them: many separate pieces (a spray, or a twig with a few), which come
    down no faster than a leaf can (air on a leaf's own area, about
    3.3 m/s) and spread out round the crown. A green crown keeps nearly all
    of its leaves, as a felled green tree does: what it loses on a hard
    landing comes off with a snapped twig, a few sprays at a time."""
    b, leafp, off_at, vz = _fell(sap=False)
    off = np.array(sorted(off_at))
    fr = b.fragments()
    sizes = np.bincount(fr[off]) if len(off) else np.zeros(0, int)
    sizes = sizes[sizes > 0]
    assert len(off) >= 0.15 * len(leafp), (len(off), len(leafp))
    assert len(sizes) >= 10, sizes
    assert sizes.max() <= 0.3 * len(off), sizes
    when = np.array(list(off_at.values()))
    assert (when >= 30).mean() >= 0.9, "they come off when it lands"
    assert vz and min(vz) > -3.6, (len(vz), min(vz) if vz else None)
    assert b.px[off, 1].std() > 0.1 and b.px[off, 0].std() > 0.2
    assert max(b.residual()) < 1e-9
    g, gleaf, goff, _ = _fell(sap=True)
    assert len(goff) <= 0.2 * len(gleaf), (len(goff), len(gleaf))
    # what a green crown loses on landing it loses with its twigs; its own
    # leaf stalks (petioles) hold, where a dry crown's part
    pet = lambda x: (x.btag == 2) & (x.A <= np.pi * PETIOLE_R ** 2 * 1.000001) & ~x.alive
    assert pet(g).sum() <= 0.1 * pet(b).sum(), (pet(g).sum(), pet(b).sum())


def _hinge(sap):
    """tree_hinge's tree and axe (src/voxel/scenes.py), under bonds: the
    big voxel crown, 2,877 leaf cells on a trunk that runs up into it.
    Which leaf pieces came off (no wood in their fragment) and when, the
    bonds, and the tick the tree first left the lattice with what it took
    (tick, wood members, leaf members)."""
    from src.voxel import scenes
    w = World(90, 40, 70, voxel_cm=5)
    w.fill(0, 90, 0, 40, 0, 1, STONE)
    scenes._grow_tree(w, 20, 20)
    if not sap:
        lm = w.mat == LEAF
        w.fl[lm] = 0
        w.fvol[lm] = 0.0
    b = w.matter.bonds
    off_at = {}
    first = None
    for t in range(120):
        if t in (15, 30, 45):
            k = (t // 15) - 1
            w.mat[21 - k, 18:22, 6:9] = AIR
            w.smass[21 - k, 18:22, 6:9] = 0.0
        w.step()
        if len(b.pm):
            if first is None:
                first = (t, int((b.vmat == WOOD).sum()), int((b.vmat == LEAF).sum()))
            fr = b.fragments()
            wood = set(fr[b.mp[b.vmat == WOOD]].tolist()) | set(fr[b.pkin].tolist())
            for p in np.unique(b.mp[b.vmat == LEAF]):
                if fr[p] not in wood:
                    off_at.setdefault(int(p), t)
    return b, off_at, first


@pytest.mark.slow
def test_the_REAL_SCENE_crown_a_dry_one_sheds_a_green_one_breaks_into_boughs():
    """The verifier's point: the small tree above is built for the test.
    tree_hinge's own crown (2,877 leaf cells, softened to fit the budget).
    Its trunk runs up into it and its branches grow from the trunk, so wood
    holds wood (the user's decision, 2026-10-16: the scene is fixed, not
    the torque law): the crown stays on the tree until the axe's third
    stroke, and the tree leaves the lattice whole, trunk and crown. Dry, it
    sheds hundreds of sprays as single pieces that spread round the tree.
    Green, it keeps its leaves; landing, it snaps boughs off where its
    branches break, and the rest of the crown stays on its trunk.

    (Before the fix the crown was let go at t12, alone, held to the trunk
    only by leaves, and its green crown landed in five boughs of which the
    biggest held at most 70% of it. That last bound was this test's; it
    belonged to a crown with no wood holding it.)"""
    b, off_at, first = _hinge(sap=False)
    assert first[0] == 45 and first[1] >= 600, first
    leafp = np.unique(b.mp[b.vmat == LEAF])
    off = np.array(sorted(off_at))
    assert len(off) >= 0.15 * len(leafp), (len(off), len(leafp))
    fr = b.fragments()
    sizes = np.bincount(fr[off])
    sizes = sizes[sizes > 0]
    assert (sizes == 1).mean() >= 0.9, sizes.max()
    assert b.px[off, 0].std() > 0.2 and b.px[off, 1].std() > 0.2
    assert max(b.residual()) < 1e-9
    g, goff, first = _hinge(sap=True)
    assert first[0] == 45 and first[1] >= 600, first
    gleaf = np.unique(g.mp[g.vmat == LEAF])
    assert len(goff) <= 0.01 * len(gleaf), (len(goff), len(gleaf))
    fr = g.fragments()
    lumps = np.bincount(fr[g.mp[g.vmat == LEAF]])
    assert (lumps >= 100).sum() >= 3, np.sort(lumps)[::-1][:6]
    trunk = np.bincount(fr[g.mp[g.vmat == WOOD]]).argmax()
    assert lumps.argmax() == trunk, "the crown came off its trunk"


# ── the rope ─────────────────────────────────────────────────────────────────

def _rope():
    """1 m of 12 mm manila (20 cells of ROPE, 5 g each) tied to a stone
    beam 1.8 m up on two posts, with a 3.9 kg iron weight (four cells) on
    its end, promoted where it hangs."""
    w = World(40, 12, 40, voxel_cm=5)
    w.fill(0, 40, 0, 12, 0, 1, STONE)
    w.fill(10, 31, 5, 7, 36, 38, STONE)
    w.fill(10, 12, 5, 7, 1, 36, STONE)           # two posts: a beam the lattice
    w.fill(29, 31, 5, 7, 1, 36, STONE)           # neither drops nor tips
    w.fill(20, 21, 6, 7, 16, 36, ROPE, frac=MANILA)
    w.fill(20, 22, 6, 8, 15, 16, IRON)
    c = _cells(w, (ROPE, IRON))
    n = w.matter.promote(c[c[:, 2] < 36])
    assert n == 24
    return w


def test_a_ROPE_SWINGS_AT_ITS_PERIOD():
    """Set swinging 10 degrees as one rigid body about its knot, the rope
    swings at the period of a compound pendulum made of its own pieces (the
    rope hardly stretches under 38 N, so it is nearly rigid)."""
    w = _rope()
    b = w.matter.bonds
    pivot = np.array([20.5, 6.5, 36.0]) * w.matter.dx
    free = np.flatnonzero(~b.pkin)
    r = b.px[free] - pivot
    M = b.pm[free].sum()
    d = float(-(b.pm[free] * r[:, 2]).sum() / M)
    I = float((b.pm[free] * (r[:, 0] ** 2 + r[:, 2] ** 2)).sum() + b.pI[free, 1].sum())
    th = np.radians(10.0)
    T = 2 * np.pi * np.sqrt(I / (M * 9.81 * d)) * (1 + th ** 2 / 16)
    b.kick(free, omega=(0.0, th * np.sqrt(M * 9.81 * d / I), 0.0), about=pivot)
    bob = int(b.mp[np.flatnonzero(b.vmat == IRON)[0]])
    xs = []
    for _ in range(int(2.3 * T / 0.025)):
        w.step()
        xs.append(float(b.px[bob, 0] - pivot[0]))
    xs = np.array(xs)
    # the bob passes under the knot going forward once a period
    up = np.flatnonzero((xs[:-1] < 0) & (xs[1:] >= 0))
    t = (up + xs[up] / (xs[up] - xs[up + 1]) + 1) * 0.025
    period = float(np.diff(t).mean())
    assert len(up) >= 2
    assert period == pytest.approx(T, rel=0.01), (period, T)
    assert not (~b.alive).any(), "a swing breaks nothing"
    assert max(b.residual()) < 1e-9


def test_a_ROPE_BREAKS_AT_ITS_WEAKEST_LINK():
    """A pull on the weight that grows by 10 kN a second: the first bond to
    part is the weakest one on the line, and it parts at its strength."""
    w = _rope()
    b = w.matter.bonds
    line = np.flatnonzero(b.btag == 1)
    bob = int(b.mp[np.flatnonzero(b.vmat == IRON)[0]])
    for _ in range(4):
        w.step()
    rope_kg = float(b.pm[~b.pkin].sum())
    F = 0.0
    while b.alive[line].all() and F < 5e4:
        F += 1e4 * 0.025
        b.pfext[bob] = (0.0, 0.0, -F)
        w.step()
    gone = line[~b.alive[line]]
    assert len(gone) >= 1
    first = gone[np.argmin(b.tbreak[gone])]
    weakest = line[np.argmin(b.sig[line] * b.A[line])]
    assert first == weakest, (first, weakest, b.tbreak[gone])
    hold = float(b.sig[weakest] * b.A[weakest])
    assert F + rope_kg * 9.81 == pytest.approx(hold, rel=0.05), (F, hold)
    assert max(b.residual()) < 1e-6 * hold


# ── the pane ─────────────────────────────────────────────────────────────────

def test_a_PANE_DROPPED_ON_AN_EDGE_SPLITS_AND_PART_STAYS():
    """A 1.0 x 0.4 m pane of glass, 70% of it over a table, dropped 0.7 m
    flat: it cracks along the table's edge, right across, and the part over
    the table stays on it while the rest falls to the floor (where that part
    may crack again as it lands)."""
    w = World(40, 16, 40, voxel_cm=5)
    w.fill(0, 40, 0, 16, 0, 1, STONE)
    w.fill(4, 20, 2, 14, 1, 16, STONE)        # the table: top 0.8 m, edge x = 1.0 m
    w.fill(6, 26, 4, 12, 30, 31, GLASS)
    g0 = w.total_mass(GLASS)
    w.matter.promote(_cells(w, (GLASS,)))
    b = w.matter.bonds
    kinds = []
    first = None                         # the first crack, where it was
    for _ in range(60):
        t = w.tick
        w.step()
        assert w.total_mass(GLASS) == g0
        for r in w.matter.events():
            assert r["tick"] == t and _plain(r), r
            kinds.append(r["kind"])
        broke = np.flatnonzero(~b.alive)
        if first is None and len(broke):
            first = 0.5 * (b.px[b.bi[broke]] + b.px[b.bj[broke]])
    fr = b.fragments()
    assert len(np.unique(fr)) >= 2, "the pane did not split"
    top = b.px[:, 2] > 0.8
    assert top.sum() >= 40, "too little stayed on the table"
    assert (b.px[top, 0] < 1.1).all(), "what stays must be over the table"
    assert (b.px[:, 2] < 0.3).sum() >= 10, "nothing fell to the floor"
    # where it cracked first: at the table's edge, across all 8 strips
    assert first is not None
    assert np.median(first[:, 0]) == pytest.approx(1.0, abs=0.15), first
    assert len(np.unique(np.floor(first[:, 1] / w.matter.dx))) == 8, first
    # and it said so: the blow, the crack, the parting (src/mpm's records)
    assert {"struck", "crack", "part"} <= set(kinds), kinds
    assert max(b.residual()) < 1e-8


# ── the plank ────────────────────────────────────────────────────────────────

def test_a_PLANK_STAYS_ON_THE_CONTINUUM_and_LANDS_WHOLE():
    """A plank is not claimed: it goes to the particles, as it always did,
    and lands on the floor in one piece."""
    w = World(30, 10, 20)
    w.fill(0, 30, 0, 10, 0, 1, STONE)
    w.fill(8, 20, 3, 6, 6, 7, WOOD)
    c = _cells(w, (WOOD,))
    w.matter.promote(c[c[:, 2] > 0])
    assert len(w.matter.bonds) == 0 and len(w.matter.mpm.x) > 0
    for _ in range(40):
        w.step()
    m = w.matter.mpm
    assert not m.dmg.any(), "the plank cracked"
    assert m.x[:, 2].max() < 0.35


# ── a plank on stone: the contact's step bound and friction ─────────────────

STONE_TOP = 0.05          # the floor is one 5 cm cell of stone


def _plank_drop(z0, vx=0.0, nt=40):
    """A wooden plank (80 x 20 x 5 cm: sixteen members to a piece), carried
    as bonded pieces, with its underside `z0` cells above a stone floor and
    thrown sideways (+x) at `vx` m/s. Per tick: its mechanical energy
    (movement, turning and height, J), its centre of mass's height, its
    sideways speed (m/s) and its sideways place (m), and the plank."""
    w = World(30, 12, 30, voxel_cm=5)
    w.fill(0, 30, 0, 12, 0, 1, STONE)
    w.fill(5, 21, 4, 8, z0, z0 + 1, WOOD)
    mt = w.matter
    mt._promote_bonds(np.argwhere(w.mat == WOOD), (0.0, 0.0, 0.0), "pane")
    b = mt.bonds
    ps = np.flatnonzero(~b.pkin)
    b.kick(ps, dv=(vx, 0.0, 0.0))
    M = b.pm[ps].sum()
    rows = []
    for _ in range(nt):
        w.step()
        e = b.kinetic() + (b.pm[ps] * b.g * b.px[ps, 2]).sum()
        com = lambda a: (b.pm[ps, None] * a[ps]).sum(0) / M
        rows.append((e, com(b.px)[2], com(b.pv)[0], com(b.px)[0]))
    return b, np.array(rows)


def test_a_PLANK_DROPPED_ON_STONE_NEVER_GAINS_ENERGY_and_BOUNCES_AS_WOOD_DOES():
    """A plank lands flat on stone with sixteen members pressing at once.
    Each carried a whole face's share of the contact's spring and damper,
    so the plank met the floor two and a half times too stiff and too
    damped for the step it was bound to, the damper overshot, and it left
    the floor with three times the energy it fell with (3.7 m/s in, 6.8 m/s
    out). The contact's damping is now part of the step bound, and a face is
    one contact however many members press on it. So: no tick has more
    mechanical energy than the plank was dropped with, and it rebounds at
    wood-on-stone's restitution, 0.3-0.45 (the handbook has about 0.4), on
    average over drops from several heights."""
    es = []
    for z0 in (6, 8, 10, 12, 15):
        b, r = _plank_drop(z0, nt=45)
        e0 = b.pm.sum() * b.g * (z0 * 0.05 + 0.025)
        assert r[:, 0].max() < e0 * (1 + 1e-6), (z0, r[:, 0].max() / e0)
        # the rebound, from the height of the first flight's apex over where
        # the plank comes to rest, against the height it fell (the apex from
        # a flight tick's height and speed: the impact falls between ticks)
        z = r[:, 1]
        rest = z[-1]
        assert rest == pytest.approx(STONE_TOP + 0.025, abs=0.003), rest
        k = int(np.argmax(np.diff(z) > 0)) + 1          # first tick going up
        j = k + 3
        vj = (z[j + 1] - z[j - 1]) / 0.05
        apex = z[j] + vj * vj / (2 * b.g)
        es.append(np.sqrt((apex - rest) / (z[0] - rest)))
    # one drop scatters by about 0.1 (a contact lasts three or four
    # substeps, and where in one the blow lands changes what it gives
    # back), so the band is for the mean of five, and no drop is far out
    assert 0.3 <= np.mean(es) <= 0.45, es
    assert 0.15 <= min(es) and max(es) <= 0.55, es


def test_a_PLANK_THROWN_SIDEWAYS_NEVER_REVERSES_and_STOPS_WITHIN_A_CELL():
    """Friction stops a slide and does not reverse it. Each of a plank's
    sixteen members on the floor was capped at what stops the slide in one
    substep, and all sixteen pushed together: the plank skidding at 0.5 m/s
    came back at 1 m/s. The cap is the piece's, shared by the members of its
    face. A plank on the floor thrown along it at 0.5 m/s (it would slide
    3 cm) stops within a cell; one that lands with sideways speed loses it
    and never reverses it (but for a ringing of a few mm/s, the bending of
    its four pieces: 1% of the throw is the floor), and once it has come
    down for the last time travels no more than a cell. The drops are kept
    short of the world's wall, which would bounce it back."""
    dx = 0.05
    rest = STONE_TOP + 0.025
    b, r = _plank_drop(1, vx=0.5, nt=30)
    start = 13 * dx                      # the plank's centre, cells 5 to 21
    assert r[:, 2].min() > -1e-3 and r[:, 2].max() <= 0.5 + 1e-9, r[:, 2]
    assert abs(r[-1, 2]) < 1e-3
    assert 0.0 < r[-1, 3] - start <= dx, r[-1, 3] - start
    assert (b.pv[:, 0] > -1e-3).all(), b.pv[:, 0]
    for z0, vx in ((15, 0.5), (10, 0.7)):
        b, r = _plank_drop(z0, vx=vx, nt=70)
        assert r[:, 3].max() + 0.4 < 1.45, "the plank would reach the wall"
        assert r[:, 2].min() > -0.01 * vx, (z0, vx, r[:, 2].min())
        assert abs(r[-1, 2]) < 1e-3, (z0, vx)
        last = int(np.flatnonzero(r[:, 1] > rest + 0.003)[-1])   # last clear of the floor
        assert 0.0 <= r[-1, 3] - r[last, 3] <= dx, (z0, vx, r[-1, 3] - r[last, 3])


def test_the_CONTACT_STEP_BOUND_counts_every_MEMBER_that_can_press_at_once():
    """A face is the most cells a piece shows along one axis: 16 for a
    plank of 16 cells one deep, 1 for a lone cell. And
    the substep the bound gives holds for the whole face with its damper:
    (w h)^2 + 2 (c/m) h < 4 (symplectic Euler), which the plank was past."""
    w = World(30, 12, 30, voxel_cm=5)
    w.fill(0, 30, 0, 12, 0, 1, STONE)
    w.fill(5, 21, 4, 8, 15, 16, WOOD)
    w.matter._promote_bonds(np.argwhere(w.mat == WOOD), (0.0, 0.0, 0.0), "pane")
    b = w.matter.bonds
    assert (b.pface == 16).all(), b.pface
    free = ~b.pkin
    for p in np.flatnonzero(free):
        mem = b.mp == p
        k = b.pface[p] * b.kcm[mem][0]
        c = b.pface[p] * b.ccm[mem][0]
        assert (k / b.pm[p]) * b.h ** 2 + 2 * (c / b.pm[p]) * b.h < 4.0, p
    # a block and a lone cell: counted again from the cells' own places
    w2 = World(30, 12, 30, voxel_cm=5)
    w2.fill(10, 14, 4, 8, 5, 9, WOOD)
    w2.fill(20, 21, 4, 5, 5, 6, WOOD)
    w2.matter._promote_bonds(np.argwhere(w2.mat == WOOD), (0.0, 0.0, 0.0), "pane")
    b2 = w2.matter.bonds
    cell = np.floor(b2.x / w2.matter.dx + 1e-6).astype(int)
    for p in range(len(b2.pm)):
        c = cell[b2.mp == p]
        by_axis = [len({tuple(np.delete(r, a)) for r in c}) for a in range(3)]
        assert b2.pface[p] == max(by_axis), (p, b2.pface[p], by_axis)
    assert 1.0 in b2.pface and b2.pface.max() > 4.0, b2.pface


# ── the two solvers (cross-solver walls) ─────────────────────────────────────

def _plank_on_block(cross, kick=0.0, continuum=True):
    """A stone block (24 x 16 x 20 cm, on the continuum: a block is not
    claimed; or left on the lattice) on the floor, and a wooden plank
    (80 x 20 x 5 cm) 0.5 m above it, square over it, carried as bonded
    pieces (the seam's own door to bonds, as a crown's boughs are carried),
    thrown down at `kick` m/s: 60 ticks. The plank members' lowest height
    over the block, tick by tick, in cells; and how far the plank's centre
    went sideways (x, y), in cells."""
    w = World(30, 12, 30, voxel_cm=5)
    w.fill(0, 30, 0, 12, 0, 1, STONE)
    w.fill(10, 16, 4, 8, 1, 5, STONE)
    w.fill(5, 21, 4, 8, 15, 16, WOOD)
    mt = w.matter
    mt.cross_walls = cross
    if continuum:
        mt.promote(np.argwhere(w.mat[:, :, 1:5] == STONE) + [0, 0, 1])
    mt._promote_bonds(np.argwhere(w.mat == WOOD), (0.0, 0.0, 0.0), "pane")
    b = mt.bonds
    if kick:
        b.kick(np.flatnonzero(~b.pkin), dv=(0.0, 0.0, kick))
    c0 = b.x.mean(0)
    low = []
    for _ in range(60):
        w.step()
        over = (b.x[:, 0] > 0.5) & (b.x[:, 0] < 0.8)
        low.append(float(b.x[over, 2].min() / mt.dx))
    return w, low, (b.x.mean(0) - c0)[:2] / mt.dx


def test_a_BONDED_PLANK_LANDS_ON_A_CONTINUUM_BLOCK_not_through_it():
    """The two solvers see each other as walls (the user's decision,
    2026-10-16). A plank in bonds dropped onto a block on the continuum
    lands on it and lies on it: its members' centres half a cell above the
    block's top, never in it. With the walls off it goes into the block,
    as both solvers did before (and there, measured the same before this
    change, it is thrown back out at about 7 m/s). The block is not pushed
    into the floor, and the books of the bonds are exact."""
    w, low, side = _plank_on_block(True)
    b, m = w.matter.bonds, w.matter.mpm
    assert min(low) > 4.9, min(low)
    assert low[-1] == pytest.approx(5.5, abs=0.05), low[-1]
    assert np.abs(side).max() < 0.1, side          # it landed square: straight down
    assert m.x[:, 2].max() / w.matter.dx < 4.3
    assert m.x[:, 2].min() / w.matter.dx > 0.95
    assert max(b.residual()) < 1e-9
    assert b.cross[2] > 0.0 and w.matter.cross[2] < 0.0, "pushed, both ways"
    assert b.cross[:3] + w.matter.cross[:3] == pytest.approx(0.0, abs=1e-9)
    _, low, _ = _plank_on_block(False)
    assert min(low) < 4.0, min(low)


def test_a_PLANK_THROWN_DOWN_SQUARE_ON_A_CONTINUUM_BLOCK_lands_STRAIGHT_as_on_a_lattice_one():
    """A plank thrown down hard and square onto a block lands straight on
    it, whichever solver carries the block: it does not slide off sideways.
    It did, 1 to 4 cells and onto the floor, when the floor's grip on the
    shivering block was booked in part on the plank (`Matter._cross_hook`:
    a push along a face is friction, shared by what each wall pushed
    square to it, not by contact mass)."""
    for kick in (-3.0, -6.0):
        lat = _plank_on_block(True, kick, continuum=False)
        con = _plank_on_block(True, kick)
        for w, low, side in (lat, con):
            assert np.abs(side).max() < 0.1, (kick, side)
            assert min(low) > 4.9, (kick, min(low))       # on it, never off it
            b = w.matter.bonds
            assert np.ptp(b.x[:, 2]) / w.matter.dx < 0.1, kick   # not tilted
            assert max(b.residual()) < 1e-9
        assert con[0].matter.bonds.cross[:3] + con[0].matter.cross[:3] \
            == pytest.approx(0.0, abs=1e-9)


def _stone_on_pane(cross):
    """A pane of glass (80 x 40 x 5 cm, in bonds) on two stone posts, and a
    stone (10 cm cube, on the continuum) 15 cm over its middle."""
    w = World(30, 12, 50, voxel_cm=5)
    w.fill(0, 30, 0, 12, 0, 1, STONE)
    w.fill(6, 8, 2, 10, 1, 10, STONE)
    w.fill(20, 22, 2, 10, 1, 10, STONE)
    w.fill(6, 22, 2, 10, 10, 11, GLASS)
    w.fill(13, 15, 5, 7, 14, 16, STONE)
    mt = w.matter
    mt.cross_walls = cross
    mt.sleep = False                 # see below
    mt.promote(np.argwhere(w.mat == GLASS))
    mt.promote(np.argwhere(w.mat[:, :, 14:16] == STONE) + [0, 0, 14])
    assert len(mt.bonds) == 128 and len(mt.mpm.x) == 8
    return w


def test_a_CONTINUUM_STONE_LANDS_ON_A_BONDED_PANE_and_the_TWO_SOLVERS_BOOKS_ARE_EXACT():
    """A stone on the continuum dropped onto a pane in bonds lands on it
    and stays on it (it hops no more than a few millimetres, where a stone
    landing on a pane that cannot give would throw it back up: the reason
    a continuum wall gives, `kernel.run`). With the walls off it falls
    through to the floor.

    THE MOMENTUM OF THE TWO SOLVERS TOGETHER is exact through the contact.
    The stone touches nothing but the pane, so its momentum is what it had,
    plus gravity, plus what the pane's cells gave it (`Matter.cross`); the
    pane's is what came in plus every outside push (`Bonds.residual`); and
    the push each gave the other is booked on both sides, the same impulse
    reversed. (Sleep is off: putting the stone to sleep stops what little
    motion it has left, as a sleeper's floor does, and the continuum keeps
    no book of that.)"""
    w = _stone_on_pane(True)
    mt = w.matter
    m, b = mt.mpm, mt.bonds
    M = float(m.m.sum())
    P0 = (m.m[:, None] * m.v).sum(0)
    low = []
    for t in range(40):
        w.step()
        low.append(float(m.x[:, 2].min() / mt.dx))
        P = (m.m[:, None] * m.v).sum(0)
        want = P0 + M * GRAVITY * 0.025 * (t + 1) + mt.cross[:3]
        assert np.abs(P - want).max() < 1e-9, (t, P - want)
        assert np.abs(b.cross[:3] + mt.cross[:3]).max() < 1e-12, t
        assert max(b.residual()) < 1e-9, t
    assert b.cross[2] < -10.0, "the stone pushed the pane down"
    assert min(low) > 10.95 and max(low[10:]) < 11.6, low
    assert low[-1] == pytest.approx(11.1, abs=0.05)
    assert (b.x[:, 2] / mt.dx > 10.4).all(), "the pane held"
    w = _stone_on_pane(False)
    for _ in range(20):
        w.step()
    assert w.matter.mpm.x[:, 2].min() / w.matter.dx < 2.0


# ── the ledgers ──────────────────────────────────────────────────────────────

def test_the_LEDGERS_ARE_EXACT_through_PROMOTE_STEP_BREAK_and_WRITE_BACK():
    """A small tree, warm, wet with sap, knocked over: every gram of wood
    and leaf, every millilitre of sap and every joule of heat is where the
    books say, every tick, while bonds break on landing, while the lattice
    burns half of one drawn leaf cell and takes another away, and on until
    it all lies still and sleeps. The pieces' momentum and angular momentum
    are what came in, plus every outside push, less what left with the
    matter (to rounding); the heat is exact to the lattice's float32."""
    w = World(40, 24, 30, voxel_cm=5)
    w.wet = None        # kept dry on purpose: this book counts liquid water to 1e-12 and an open humid surface gives the air a little vapour (booked in total_water, not in total_fluid)
    w.fill(0, 40, 0, 24, 0, 1, STONE)
    _small_tree(w, 10, 12, 1, 14, 3)
    w.E[w.mat != AIR] += 30.0
    w.E[:, :, 0] = 0.0
    tree = _cells(w, (WOOD, LEAF))
    tree = tree[tree[:, 2] >= 2]
    wood0, leaf0 = w.total_mass(WOOD), w.total_mass(LEAF)
    sap0, E0 = w.total_fluid(WATER), w.total_energy()
    w.matter.promote(tree)
    b = w.matter.bonds
    assert len(b) == len(tree) and len(w.matter.mpm.x) == 0
    assert w.total_mass(WOOD) == pytest.approx(wood0, rel=1e-12)
    assert w.total_mass(LEAF) == pytest.approx(leaf0, rel=1e-12)
    assert w.total_fluid(WATER) == pytest.approx(sap0, rel=1e-12)
    assert w.total_energy() == pytest.approx(E0, rel=1e-12)
    b.kick(np.flatnonzero(~b.pkin), omega=(0.0, 1.5, 0.0),
           about=np.array([10.0, 12.0, 2.0]) * w.matter.dx)
    took = 0.0
    for t in range(200):
        w.step()
        if t == 20:
            # the lattice burns half of one drawn leaf cell, and takes a
            # second one back (as if it had become something else)
            at = np.unravel_index(b.drawn.drawn_at, w.shape)
            leafy = np.flatnonzero(w.mat[at] == LEAF)
            c1 = tuple(a[leafy[0]] for a in at)
            c2 = tuple(a[leafy[-1]] for a in at)
            took = float(w.smass[c1]) / 2.0
            w.smass[c1] -= took
            w.mat[c2] = WOOD
        lw = w.total_mass(WOOD) + w.total_mass(LEAF)
        assert lw == pytest.approx(wood0 + leaf0 - took, rel=1e-12), t
        assert w.total_fluid(WATER) == pytest.approx(sap0, rel=1e-12), t
        assert w.total_energy() == pytest.approx(E0, rel=1e-6), t
        P = max(b.residual())
        assert P < 1e-9, (t, b.residual())
        if t > 50 and b.asleep:
            break
    assert (~b.alive).any(), "nothing broke"
    assert b.asleep, "it never came to rest"
