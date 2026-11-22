"""NAMED THINGS IN BONDS (the user's decision, 2026-10-16).

A named thing that must part (a crown, a rope, a pane) goes to bonds like
any other. things.py finds a thing's matter through the seam
(`Matter.points`: the particles and the bonds' members as one set), so a
named tree or rope keeps its name and its eid through promotion, bonding,
a break and the write-back, and the pieces a break makes are new things
named from it ("a piece of the oak"). The ledgers stay exact on the way.

Every number here was measured before it was asserted.
"""

import numpy as np
import pytest

from src.voxel.materials import ROPE
from src.voxel.scenes import thing
from src.voxel.sim import World, AIR, STONE, WOOD, IRON, LEAF, WATER


def _cells(w, mats):
    return np.argwhere(np.isin(w.mat, mats))


def _small_tree(w, cx, cy, z0, top, r):
    """tests/test_bonds.py's small tree (fresh leaves): a trunk two cells
    square up to `top`, four spokes, and a leaf ball of radius `r`."""
    w.fill(cx - 1, cx + 1, cy - 1, cy + 1, z0, top, WOOD)
    X, Y, Z = np.ogrid[:w.shape[0], :w.shape[1], :w.shape[2]]
    ball = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - top - 1) ** 2 <= r * r
    ball &= w.mat == AIR
    w.mat[ball] = LEAF
    w.smass[ball] = 200.0 * 0.4 * w.vox_l
    w.fl[ball] = WATER
    w.fvol[ball] = 0.06 * w.cap
    for bx, by in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for t in range(1, r):
            px, py = cx + bx * t, cy + by * t
            w.fill(px, px + 1, py, py + 1, top - 1, top, WOOD, frac=0.4)


def _family(w, e):
    """Thing `e` and every thing made from it, alive or not."""
    out, todo = {e}, [e]
    while todo:
        k = todo.pop()
        for c, t in w.things.items():
            if t["parent"] == k and c not in out:
                out.add(c)
                todo.append(c)
    return out


def _grams(w, ks):
    return sum(w.props(k)["mass_g"] for k in ks if w.things[k]["dead"] is None)


def test_a_NAMED_TREE_KEEPS_ITS_NAME_through_BONDS_a_BREAK_and_the_WRITE_BACK():
    """A named oak (a small tree, warm, wet with sap) is cut at its foot,
    goes to bonds as "the oak", and is knocked over. Every tick: each of
    its members carries the oak's eid or the eid of a piece made from it;
    the members of one fragment (what unbroken bonds hold together) are one
    thing; every gram of the oak is in the oak or its pieces; the cells the
    bonds draw say which thing they are; and the books are exact (grams,
    sap, heat, momentum and angular momentum). Its landing breaks it: the
    oak keeps its eid and its name ("the broken oak"), and each piece is
    "a piece of the oak". At tick 20 the lattice burns half of one drawn
    leaf cell, and that is what the oak family loses."""
    w = World(60, 30, 40, voxel_cm=5)
    w.wet = None        # kept dry on purpose: this book counts liquid water to 1e-12 and an open humid surface gives the air a little vapour (booked in total_water, not in total_fluid)
    w.fill(0, 60, 0, 30, 0, 1, STONE)
    with thing(w, "the oak", kind="tree") as e:
        _small_tree(w, 12, 15, 1, 22, 4)
    assert w.things[e]["called"] == "the oak"
    w.E[w.mat != AIR] += 30.0
    w.E[:, :, 0] = 0.0
    tree = _cells(w, (WOOD, LEAF))
    assert (w.eid[tuple(tree.T)] == e).all()
    cut = tree[tree[:, 2] >= 2]
    g0 = _grams(w, [e])
    wl0 = w.total_mass(WOOD) + w.total_mass(LEAF)
    sap0, E0 = w.total_fluid(WATER), w.total_energy()
    w._things_moved(w.eid[tuple(cut.T)])      # as the laws that promote do
    w.matter.promote(cut)
    b = w.matter.bonds
    assert len(b) == len(cut) and len(w.matter.mpm.x) == 0, "named: to bonds"
    assert (b.eid == e).all()
    assert _grams(w, [e]) == pytest.approx(g0, rel=1e-12)
    b.kick(np.flatnonzero(~b.pkin), omega=(0.0, 0.6, 0.0),
           about=np.array([12.0, 15.0, 2.0]) * w.matter.dx)
    took = 0.0
    for t in range(80):
        w.step()
        if t == 20:
            at = np.unravel_index(b.drawn.drawn_at, w.shape)
            leafy = np.flatnonzero(w.mat[at] == LEAF)
            c1 = tuple(a[leafy[0]] for a in at)
            assert int(w.eid[c1]) in _family(w, e)
            took = float(w.smass[c1]) / 2.0
            w.smass[c1] -= took
        fam = _family(w, e)
        assert set(np.unique(b.eid).tolist()) <= fam, t
        fr = b.fragments()[b.mp]
        for f in np.unique(fr):
            assert len(np.unique(b.eid[fr == f])) == 1, (t, f)
        # the drawn cells are the members' things, as the minds read them
        at = b.drawn.drawn_at
        assert set(w.eid.ravel()[at].tolist()) <= fam, t
        # (a thing's grams are what its matter holds: the burn reaches the
        # members at the next write-back, the tick after; the ledgers below
        # count it at once, as pending)
        felt = took if t > 20 else 0.0
        assert _grams(w, fam) == pytest.approx(g0 - felt, rel=1e-9), t
        lw = w.total_mass(WOOD) + w.total_mass(LEAF)
        assert lw == pytest.approx(wl0 - took, rel=1e-12), t
        assert w.total_fluid(WATER) == pytest.approx(sap0, rel=1e-12), t
        assert w.total_energy() == pytest.approx(E0, rel=1e-6), t
        P, L = b.residual()
        assert P < 1e-9 and L < 1e-9, (t, P, L)
    assert took > 0.0
    assert (~b.alive).any(), "nothing broke"
    oak = w.things[e]
    assert oak["dead"] is None and oak["broken"]
    assert w.name(e, "x") == "the broken oak"
    kids = [k for k in _family(w, e) if k != e]
    assert len(kids) >= 2, kids
    assert {w.things[k]["called"] for k in kids} == {"a piece of the oak"}
    # the oak is the heaviest piece, and it is in bonds
    assert w.props(e)["in_solver"]
    assert w.props(e)["mass_g"] >= max(w.props(k)["mass_g"] for k in kids
                                       if w.things[k]["dead"] is None)


def _rope(named):
    """tests/test_bonds.py's rope: 1 m of 12 mm manila (20 cells of ROPE,
    5 g each) knotted to a stone beam on two posts, a 3.9 kg iron weight on
    its end; named "rope" and "weight" when `named`. Its rope grams too."""
    w = World(40, 12, 40, voxel_cm=5)
    w.fill(0, 40, 0, 12, 0, 1, STONE)
    w.fill(10, 31, 5, 7, 36, 38, STONE)
    w.fill(10, 12, 5, 7, 1, 36, STONE)
    w.fill(29, 31, 5, 7, 1, 36, STONE)
    frac = 5.0 / (885.0 * w.vox_l)
    if named:
        with thing(w, "rope", kind="rope") as r:
            w.fill(20, 21, 6, 7, 16, 36, ROPE, frac=frac)
        with thing(w, "weight") as k:
            w.fill(20, 22, 6, 8, 15, 16, IRON)
    else:
        r = k = 0
        w.fill(20, 21, 6, 7, 16, 36, ROPE, frac=frac)
        w.fill(20, 22, 6, 8, 15, 16, IRON)
    c = _cells(w, (ROPE, IRON))
    g = float(w.smass[w.mat == ROPE].sum(dtype=np.float64))
    w._things_moved(w.eid[tuple(c.T)])        # as the laws that promote do
    assert w.matter.promote(c) == 24
    return w, r, k, g


def test_a_NAMED_ROPE_THAT_BREAKS_is_the_ROPE_and_A_PIECE_OF_THE_ROPE():
    """The named rope and its weight go to bonds as two things. Pulled
    until a link parts, the rope breaks as an unnamed rope does: at its
    weakest link, at that link's strength (its flaws are its own, seeded by
    its eid, so it is not the same link as the unnamed rope's). The heavier
    part keeps the rope's eid and name ("the broken rope"), the other is "a
    piece of the rope", and the weight is still the weight, whole. Not a
    gram of rope is lost, and the books are exact."""
    w, r, k, g0 = _rope(True)
    b = w.matter.bonds
    assert b.kind_of([0]) == ["rope"]
    assert set(np.unique(b.eid).tolist()) == {r, k}
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
    first = gone[np.argmin(b.tbreak[gone])]
    weakest = line[np.argmin(b.sig[line] * b.A[line])]
    assert first == weakest, (first, weakest)
    hold = float(b.sig[weakest] * b.A[weakest])
    assert F + rope_kg * 9.81 == pytest.approx(hold, rel=0.05), (F, hold)
    b.pfext[bob] = 0.0
    for _ in range(3):
        w.step()
    P, L = b.residual()
    assert P < 1e-6 * hold and L < 1e-6 * hold, (P, L)
    rope = b.vmat == ROPE
    fam = _family(w, r)
    assert set(np.unique(b.eid[rope]).tolist()) <= fam
    assert (b.eid[~rope] == k).all(), "the weight keeps its own name"
    assert w.things[r]["dead"] is None and w.things[k]["dead"] is None
    assert w.name(r, "x") == "the broken rope"
    assert w.name(k, "x") == "the weight"
    kids = [c for c in fam if c != r]
    assert [w.things[c]["called"] for c in kids] == ["a piece of the rope"]
    fr = b.fragments()[b.mp]
    assert len(np.unique(fr[rope])) == 2
    for f in np.unique(fr[rope]):
        assert len(np.unique(b.eid[rope & (fr == f)])) == 1
    assert _grams(w, fam) == pytest.approx(g0, rel=1e-12)
    assert g0 == pytest.approx(100.0, rel=1e-6)
