"""Pin what a promoted vessel carries with it, before X4's `_cavity` refactor
touches the code that decides it (design: docs/sim-mind-design.md, step EB).

The rule lives in the "AND A VESSEL CARRIES WHAT IS IN IT" block of
`src/mpm/matter.py::Matter.promote` (the carry block; see docs/sim-split-map.md
for where it sits against the design's old `sim.py` line numbers). From the
bounding box of the solid cells a `promote()` call takes off the lattice, it
decides which AIR cells holding fluid are "inside" that shape, and moves their
fluid onto the particles instead of leaving it to fall on its own.

These tests pin TODAY'S rule, not an ideal one. `test_an_OPEN_ENDED_PIPE_...`
below pins a real gap in it (a tube with no end caps is not "walled in" and so
is not carried) on purpose: whoever rewrites `_cavity` must decide to keep that
gap or close it, never drift across it unnoticed.
"""
import numpy as np
import pytest

from src.voxel.sim import AIR, GLASS, STONE, WATER, World


def _promote_glass(w):
    """Promote every GLASS cell as one cluster, the way the topple and
    free-fall paths do (support.py:456), and return what rode along."""
    cells = np.argwhere(w.mat == GLASS)
    w.matter.promote(cells)
    return w.matter.fluid(WATER)


def test_a_PROMOTED_VAT_CARRIES_WHAT_IT_HELD():
    """Right after the promote, the particles hold about what was poured, and
    the hollow it came from holds nothing."""
    w = World(10, 10, 20, voxel_cm=10)
    w.fill(0, 10, 0, 10, 0, 1, STONE)
    w.fill(2, 8, 2, 8, 1, 2, GLASS)                      # vat base
    w.fill(2, 8, 2, 8, 2, 6, GLASS)                      # vat walls...
    w.mat[3:7, 3:7, 2:6] = AIR                           # ...cavity
    w.smass[3:7, 3:7, 2:6] = 0.0
    poured = 0.0
    for x in range(3, 7):
        for y in range(3, 7):
            for z in (2, 3):
                w.pour(x, y, z, WATER, w.cap)
                poured += w.cap

    carried = _promote_glass(w)

    assert carried == pytest.approx(poured, rel=1e-6)
    assert float(w.fvol[3:7, 3:7, 2:6].sum()) == 0.0


# ── the carry mask, pinned per shape (matter.py:141-170 as of base3) ─────────
#
# Each shape is poured, promoted, and checked against the amount measured on
# today's code. `lattice_after` is what is left behind in the AIR cells the
# vessel used to stand over — zero where the carry block claimed the fluid,
# unchanged where it did not (the block never duplicates fluid: it either
# moves it whole onto the particles, or leaves it exactly where it was).

def _cup():
    """An open-topped glass cup, two layers of water, both walled on 4 sides
    and resting on the base. Expected: carried."""
    w = World(10, 10, 20, voxel_cm=10)
    w.fill(0, 10, 0, 10, 0, 1, STONE)
    w.fill(2, 8, 2, 8, 1, 2, GLASS)
    w.fill(2, 8, 2, 8, 2, 6, GLASS)
    w.mat[3:7, 3:7, 2:6] = AIR
    w.smass[3:7, 3:7, 2:6] = 0.0
    poured = 0.0
    for x in range(3, 7):
        for y in range(3, 7):
            for z in (2, 3):
                w.pour(x, y, z, WATER, w.cap)
                poured += w.cap
    return w, poured, (slice(3, 7), slice(3, 7), slice(2, 6))


def _open_box():
    """A wider, shallower open-topped box, one layer, not filled to the cap.
    Expected: carried, same as the cup."""
    w = World(14, 14, 20, voxel_cm=10)
    w.fill(0, 14, 0, 14, 0, 1, STONE)
    w.fill(2, 12, 2, 12, 1, 2, GLASS)
    w.fill(2, 12, 2, 12, 2, 5, GLASS)
    w.mat[3:11, 3:11, 2:5] = AIR
    w.smass[3:11, 3:11, 2:5] = 0.0
    poured = 0.0
    for x in range(3, 11):
        for y in range(3, 11):
            w.pour(x, y, 2, WATER, 400.0)
            poured += 400.0
    return w, poured, (slice(3, 11), slice(3, 11), slice(2, 5))


def _chair():
    """A seat, four legs and a backrest: solid glass, no cavity, nothing to
    carry. Pins that promoting a shape with no vessel in it does not crash
    and carries nothing."""
    w = World(14, 10, 14, voxel_cm=10)
    w.fill(0, 14, 0, 10, 0, 1, STONE)
    w.fill(4, 10, 4, 7, 4, 5, GLASS)                     # seat
    for (lx, ly) in ((4, 4), (9, 4), (4, 6), (9, 6)):
        w.fill(lx, lx + 1, ly, ly + 1, 1, 4, GLASS)      # legs
    w.fill(4, 10, 7, 8, 5, 9, GLASS)                     # backrest
    return w, 0.0, (slice(0, 1), slice(0, 1), slice(0, 1))  # nothing poured


def _pipe():
    """A straight glass tube, bored through and OPEN AT BOTH ENDS along x: no
    end caps. Expected: NOT carried — the known gap. At the hollow bore's
    (y, z) position there is no glass at any x, so the carry block's walled
    check (own cells on both sides in x) never fires, and the water is left
    stranded in the lattice when the glass flies off."""
    w = World(20, 10, 10, voxel_cm=10)
    w.fill(0, 20, 0, 10, 0, 1, STONE)
    w.fill(2, 18, 3, 7, 2, 6, GLASS)
    w.mat[2:18, 4:6, 3:5] = AIR
    w.smass[2:18, 4:6, 3:5] = 0.0
    poured = 0.0
    for x in range(3, 17):
        for y in (4, 5):
            w.pour(x, y, 3, WATER, w.cap)
            poured += w.cap
    return w, poured, (slice(2, 18), slice(4, 6), slice(3, 5))


def _closed_box():
    """The same cup, sealed with a glass lid on top too. Expected: carried,
    exactly as the open cup — the carry rule only looks for support from
    below and walls to the side, never for an open top."""
    w = World(10, 10, 20, voxel_cm=10)
    w.fill(0, 10, 0, 10, 0, 1, STONE)
    w.fill(2, 8, 2, 8, 1, 2, GLASS)
    w.fill(2, 8, 2, 8, 2, 7, GLASS)                      # walls, sealed at z=6
    w.mat[3:7, 3:7, 2:6] = AIR
    w.smass[3:7, 3:7, 2:6] = 0.0
    poured = 0.0
    for x in range(3, 7):
        for y in range(3, 7):
            for z in (2, 3):
                w.pour(x, y, z, WATER, w.cap)
                poured += w.cap
    return w, poured, (slice(3, 7), slice(3, 7), slice(2, 6))


# (builder, expected carried, expected fluid left in the lattice after)
_SHAPES = {
    "cup": (_cup, 32000.0, 0.0),
    "open_box": (_open_box, 25600.0, 0.0),
    "chair": (_chair, 0.0, 0.0),
    "pipe": (_pipe, 0.0, 28000.0),
    "closed_box": (_closed_box, 32000.0, 0.0),
}


@pytest.mark.parametrize("name", sorted(_SHAPES))
def test_the_CARRY_MASK_is_pinned_per_shape(name):
    build, expect_carried, expect_left = _SHAPES[name]
    w, poured, box = build()
    assert float(w.fvol[box].sum()) == pytest.approx(poured, rel=1e-6)

    carried = _promote_glass(w)

    assert carried == pytest.approx(expect_carried, rel=1e-6)
    assert float(w.fvol[box].sum()) == pytest.approx(expect_left, rel=1e-6)
