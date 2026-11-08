"""What a BOND is made of: its own table, with handbook numbers.

The particle solver's table (src/mpm/core.py MAT) is calibrated for the
continuum, and it stays as it is: its wood is 10 MPa, which the continuum's
softening was tuned around. A bond is a piece of a real member (a trunk, a
twig, a leaf stalk, a strand of rope, a strip of glass), so it reads the
handbook (docs/realism-references.md S2-S4, and the BD lane's notes in
tools/protos/bonds/bonds.py, where each number was first measured).

Each row: Young's modulus E (Pa), Poisson's ratio, tensile strength (Pa),
damping of the bond as a fraction of critical, density (kg/m^3). Wood has
two rows, ALONG the grain and ACROSS it, and a bond between the two reads
Hankinson's formula for its own direction.

UNVERIFIED (order of magnitude, from the BD prototype, not opened here):
the petiole (a green leaf stalk), and the rope (12 mm manila: about 10.6 kN
to break and 10% stretch at break).
"""

import numpy as np

from ..voxel.materials import (ASH, CHAR, FLESH, GLASS, IRON, LEAD, LEAF,
                               ROPE as _ROPE, STONE, TIN, WOOD)

# along the grain: E 11 GPa; strength 60 MPa, the low end of bending
# strength (S3: 59-99 MPa). Across: 1/20 as stiff (S2) and 3 MPa (S4:
# 2.1-5.5 MPa).
WOOD_L = (1.1e10, 0.35, 6.0e7, 0.10, 600.0)
WOOD_T = (5.5e8, 0.35, 3.0e6, 0.10, 600.0)
# annealed glass: 70 GPa; about 50 MPa in bending (30-90 by surface flaws)
GLASS_B = (7.0e10, 0.22, 5.0e7, 0.02, 2500.0)
# 12 mm manila: E*A about 1.06e5 N over a 1.13e-4 m^2 section, and 10.6 kN
# to break (Engineering ToolBox's manila table; the voxel ROPE material
# reads the same density)
ROPE = (9.4e8, 0.40, 9.4e7, 0.10, 885.0)
# a green leaf stalk (UNVERIFIED)
PETIOLE = (3.0e8, 0.40, 2.0e7, 0.10, 900.0)
# ...and the stalk of a DRY leaf (one holding no sap): it has grown the
# layer it drops by, and holds a tenth as much (the BD prototype's autumn
# petiole: an assumption, for contrast)
PETIOLE_DRY = (3.0e8, 0.40, 2.0e6, 0.10, 900.0)
# a green twig (UNVERIFIED): green wood is about 0.7 as stiff as dry and
# about 40 MPa in bending
TWIG = (8.0e9, 0.35, 4.0e7, 0.10, 900.0)
# the rest of what can be solid, when it is part of a bonded thing (an iron
# weight on a rope, a stone in a crown): handbook orders of magnitude.
# Granite 50 GPa, 10 MPa in tension; wrought iron 200 GPa, 250 MPa; tin and
# lead soft metals; flesh a soft solid; char a weak, brittle wood.
STONE_B = (5.0e10, 0.25, 1.0e7, 0.05, 2600.0)
IRON_B = (2.0e11, 0.29, 2.5e8, 0.05, 7870.0)
TIN_B = (5.0e10, 0.36, 2.0e7, 0.05, 7300.0)
LEAD_B = (1.6e10, 0.44, 1.2e7, 0.05, 11340.0)
FLESH_B = (1.0e6, 0.45, 1.0e6, 0.30, 1050.0)
CHAR_B = (1.0e9, 0.30, 1.0e6, 0.10, 300.0)

# voxel material -> its row (wood is special: two rows, chosen by grain)
ROW = {int(GLASS): GLASS_B, int(STONE): STONE_B, int(IRON): IRON_B,
       int(TIN): TIN_B, int(LEAD): LEAD_B, int(FLESH): FLESH_B,
       int(CHAR): CHAR_B, int(LEAF): PETIOLE, int(_ROPE): ROPE}
# what holds nothing: ash is a heap, not a member
NO_BOND = {int(ASH)}

# A LEAF STALK. Each leaf cell (a spray of leaves) hangs on ONE stalk, to
# the next cell on its way to the wood: a petiole this thick for a spray
# with nothing beyond it, and a twig for one that carries others, as thick
# as all the stalks it carries together (Leonardo's rule: a branch's
# section is the sum of its twigs'), up to a full face of the cell.
PETIOLE_R = 0.002                       # m
# What air pushes on, per gram of leaf: the BD prototype's spray, 30 g with
# 0.04 m^2 (it falls at about 3.3 m/s).
LEAF_AREA = 0.04 / 30.0                 # m^2 per g


def hankinson(a, b, cos2):
    """Hankinson's formula: a property `a` along the grain and `b` across
    it, for a direction whose cosine to the grain, squared, is `cos2`."""
    return a * b / (a * (1.0 - cos2) + b * cos2)


def wood_row(cos2):
    """The wood row for a bond whose direction makes cos^2 = `cos2` with
    the grain: E and strength by Hankinson, the rest as along it."""
    E = hankinson(WOOD_L[0], WOOD_T[0], cos2)
    S = hankinson(WOOD_L[2], WOOD_T[2], cos2)
    return (E, WOOD_L[1], S, WOOD_L[3], WOOD_L[4])


def pair_row(ma, mb, cos2=1.0):
    """The row of a bond between a cell of voxel material `ma` and one of
    `mb`, or None when they do not bond. Wood with wood reads the grain;
    a leaf with anything is a stalk; otherwise the weaker of the two rows
    (lower strength), since a joint is as strong as its weaker side."""
    ma, mb = int(ma), int(mb)
    if ma in NO_BOND or mb in NO_BOND:
        return None
    if ma == int(LEAF) or mb == int(LEAF):
        return PETIOLE
    ra = wood_row(cos2) if ma == int(WOOD) else ROW.get(ma)
    rb = wood_row(cos2) if mb == int(WOOD) else ROW.get(mb)
    if ra is None or rb is None:
        return None
    return ra if ra[2] <= rb[2] else rb


def flaw_field(shape, seed):
    """Where a thing is weak: the same field src/mpm reads (a uniform field
    smoothed twice along each axis, scaled to [0, 1]), seeded by the thing,
    so its weak places go wherever it goes."""
    rng = np.random.default_rng(seed)
    f = rng.random(shape)
    for _ in range(2):
        for a in range(3):
            pad = np.pad(f, [(1, 1) if b == a else (0, 0) for b in range(3)],
                         mode="edge")
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[a], hi[a] = slice(0, -2), slice(2, None)
            f = (f + pad[tuple(lo)] + pad[tuple(hi)]) / 3.0
    f -= f.min()
    return f / max(float(f.max()), 1e-12)


def flaw_factor(f):
    """src/mpm's spread of strength over a flaw value: 0.45 to 2.0 times."""
    return 0.45 + 1.55 * f
