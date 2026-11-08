"""The four scenes, as bonded objects. continuum.py builds the same four in
src/mpm; run.py runs either and writes what it measured.

Each builder returns a `Bonded` (bonds.py), not yet finished, plus a dict of
the piece and bond indices the measurement needs. Geometry is in metres and
copies the continuum scene it is compared with (src/mpm/scenes.py, or the
voxel `tree_fell`)."""

import numpy as np

import bonds as B

TICK = 0.025


# ── 1. the tree ──────────────────────────────────────────────────────────────
# `tree_fell`'s tree, in metres: a 20 cm square trunk from the ground (0.05)
# to 2.05 m; a crown about 0.45 m round centred at 2.3 m, on eight wood
# spokes of 5 cm voxels at 40% fill (so about 3.2 cm square). The voxel crown
# is SOLID LEAF at 40% fill: 2698 cells, about 27 kg of leaf plus sap. A real
# crown is mostly air, so here the leaves are 30 g sprays on petioles, each
# with the area air pushes on (0.04 m^2: it falls at about 3.3 m/s), hung on
# the spokes and on a short leader above the trunk.
SPRAY_M, SPRAY_A, SPRAY_HALF = 0.03, 0.04, (0.06, 0.06, 0.005)
PETIOLE_R, PETIOLE_L = 0.002, 0.07


def tree(cx=1.0, cy=1.0, seed=0, petiole=B.PETIOLE, wood=(B.WOOD_L_HB, B.WOOD_T_HB)):
    b = B.Bonded()
    b.wall((-10, -10, -1.0), (10, 10, 0.05))               # the ground
    s = 0.2
    stump = b.piece((cx, cy, 0.05 - 0.1), 600 * s ** 3, (0.1,) * 3, kin=True, tag=9)
    # THE BUTT'S CONTACT SPHERE IS HALF THE TRUNK'S. A full 10 cm sphere
    # sat on the ground exactly at the root, pressed into it as soon as the
    # trunk leaned on the root, and held the tree square: the 2 Hz root below
    # could not sway, and a felled tree on a free hinge stood. Half the size,
    # the butt meets the ground only past 60 degrees of lean.
    trunk = [b.piece((cx, cy, 0.05 + s * (k + 0.5)), 600 * s ** 3, (0.1,) * 3,
                     rc=0.05 if k == 0 else None, tag=1)
             for k in range(10)]
    A, I = s * s, s ** 4 / 12
    root = b.bond(stump, trunk[0], wood, A, I, s / 2, grain=(0, 0, 1), tag=9)
    for k in range(9):
        b.bond(trunk[k], trunk[k + 1], wood, A, I, s / 2, grain=(0, 0, 1), tag=1)
    # THE SHAKER'S GRIP, on the trunk piece centred at 0.55 m: a pinned
    # clamp beside the trunk, joined to it by a ball joint that pulls the
    # two together every way, never lets go, and holds no turning. A clamp
    # that also held the trunk piece square (a trunk piece pinned outright)
    # bent the trunk between it and the root by the whole 1.5 cm stroke and
    # snapped it below the clamp in the first 0.14 s of shaking; a real
    # clamp's rubber pads let the trunk turn in them.
    clamp = b.piece((cx, cy - s, 0.05 + s * 2.5), 1.0, (0.05,) * 3, kin=True, tag=8)
    grip = b.bond(clamp, trunk[2], wood, A, I, s / 2, grain=(0, 0, 1), ball=True,
                  bend=0.0, flaw=1e30, tag=8)
    # the leader: 5 cm square, four 15 cm pieces up to 2.65 m
    sl, hl = 0.05, 0.075
    ml = 600 * sl * sl * 2 * hl
    lead = [b.piece((cx, cy, 2.05 + hl * (2 * k + 1)), ml, (sl / 2, sl / 2, hl), tag=2)
            for k in range(4)]                         # a box standing up
    Al, Il = sl * sl, sl ** 4 / 12
    b.bond(trunk[-1], lead[0], wood, Al, Il, sl / 2, grain=(0, 0, 1), tag=2)
    for k in range(3):
        b.bond(lead[k], lead[k + 1], wood, Al, Il, sl / 2, grain=(0, 0, 1), tag=2)
    # eight spokes of three 12 cm pieces, rising 4 cm a piece
    ss, hs = 0.032, 0.06
    ms = 600 * ss * ss * 2 * hs
    As, Is = ss * ss, ss ** 4 / 12
    twigs = list(lead)
    for k in range(8):
        a = np.pi / 4 * k
        d = np.array((np.cos(a), np.sin(a), 0.0))
        prev = lead[0]
        for p in range(3):
            r = 0.16 + 0.12 * p
            x = np.array((cx, cy, 2.13 + 0.04 * p)) + r * d
            i = b.piece(x, ms, (hs, ss / 2, ss / 2), tag=3, along=d)
            b.bond(prev, i, wood, As, Is, ss / 2, grain=d, tag=3)
            twigs.append(i)
            prev = i
    # leaf sprays: four round each twig piece, two more at each spoke tip
    rng = np.random.default_rng(seed)
    fl = B.flaw_field((8, 8, 8), seed).ravel()
    Ap, Ip = np.pi * PETIOLE_R ** 2, np.pi * PETIOLE_R ** 4 / 4
    sprays, petioles = [], []
    for t in twigs:
        xt = b._p[t]["x"]
        if t in lead:
            dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
        else:
            d = xt - np.array((cx, cy, xt[2]))
            d /= np.linalg.norm(d)
            side = np.cross(d, (0, 0, 1))
            dirs = [side, -side, (0, 0, 1), (0, 0, -1)]
            if (twigs.index(t) - 4) % 3 == 2:          # a spoke's tip
                dirs += [d, d + np.array((0, 0, 0.7))]
        for dd in dirs:
            dd = np.asarray(dd, float) / np.linalg.norm(dd)
            xs = xt + PETIOLE_L * dd
            # NO SPRAY INSIDE ANOTHER. Sprays of neighbouring spokes met
            # where the spokes do, started overlapping, and the contact that
            # pushed them apart snapped their petioles in the first
            # millisecond. A spray that would touch one already hung is left
            # off.
            if any(np.linalg.norm(xs - b._p[o]["x"]) < 0.07 for o in sprays):
                continue
            j = b.piece(xs, SPRAY_M, SPRAY_HALF, rc=0.03, area=SPRAY_A, tag=4)
            petioles.append(b.bond(t, j, petiole, Ap, Ip, PETIOLE_R,
                                   flaw=B.flaw_factor(fl[rng.integers(len(fl))]),
                                   tag=4))
            sprays.append(j)
    return b, dict(stump=stump, trunk=trunk, lead=lead, root=root, clamp=clamp,
                   grip=grip, sprays=np.array(sprays), petioles=np.array(petioles))


def root(b, ix, f_sway=2.0, zeta=0.05):
    """The stump bond is the ROOT PLATE in its soil, not wood: it turns
    under the tree as soil gives, so the tree sways at about `f_sway` Hz,
    and it does not break. (A rigid root and a shaker clamp 0.5 m above it
    measured 200 MPa in the trunk between them: a shaker would have snapped
    the trunk, which a real one does not, because the roots move.)"""
    r = ix["root"]
    xb = b.x[ix["stump"]] + b.ai[r]
    d = b.x - xb
    free = ~b.kin
    Ib = float((b.m[free] * (d[free, 0] ** 2 + d[free, 2] ** 2)).sum())
    k = Ib * (2 * np.pi * f_sway) ** 2
    b.kb[r] = b.kt[r] = k
    b.cb[r] = b.ct[r] = 2 * zeta * np.sqrt(k * Ib)
    b.sig[r] = b.tau[r] = 1e30


def fell(b, ix):
    """The notch: the stump bond becomes a HINGE. A felling hinge is about a
    tenth of the trunk thick and holds the tree for most of its fall
    (docs/realism-references.md S24); its fibres bend and give rather than
    spring back, so it is near zero in bending, and it lets go at about a
    radian (57 degrees) of lean."""
    r = ix["root"]
    b.kb[r] *= 1e-5
    b.kt[r] *= 1e-5
    b.cb[r] *= 1e-3
    b.ct[r] *= 1e-3
    # it lets go by turning, not by being pulled: its section is made too
    # big for the push along it to count, and it parts at one radian
    b.A[r] = 1e9
    b.sig[r] = b.kb[r] * 1.0 * b.rb[r] / b.Ib[r]
    b.tau[r] = 1e30


# ── 2. the rope ──────────────────────────────────────────────────────────────
# 1 m of 12 mm manila (0.1 kg/m) in ten 10 cm links from a beam at 2 m,
# with a 5 kg iron weight on the end.
ROPE_R, LINK, BOB = 0.006, 0.1, 5.0


def rope(top=(0.0, 0.0, 2.0), angle=0.0, nlink=10, seed=0):
    b = B.Bonded(pp=False)
    b.wall((-10, -10, -1.0), (10, 10, 0.0))
    top = np.asarray(top, float)
    A, I = np.pi * ROPE_R ** 2, np.pi * ROPE_R ** 4 / 4
    ml = 0.1 * LINK
    side = (5.0 / 7800.0) ** (1 / 3)
    d = np.array((np.sin(angle), 0.0, -np.cos(angle)))
    # the beam's centre a half-link above the knot, so the rope turns
    # about `top` itself
    beam = b.piece(top - d * (LINK / 2), 1.0, (0.05, 0.05, 0.05), kin=True, tag=9)
    links, bnds = [], []
    prev = beam
    fl = B.flaw_field((nlink + 3, 3, 3), seed)[1:-1, 1, 1]
    for k in range(nlink + 1):
        if k < nlink:
            x = top + d * (LINK * (k + 0.5))
            i = b.piece(x, ml, (LINK / 2, ROPE_R, ROPE_R), rc=ROPE_R, tag=1,
                        along=d)
            links.append(i)
        else:
            x = top + d * (LINK * nlink + side / 2)
            i = b.piece(x, BOB, (side / 2,) * 3, area=side * side, tag=2)
        # a rope bends as easily as a chain: ball joints, with a thousandth
        # of a rod's stiffness against turning
        bnds.append(b.bond(prev, i, B.ROPE, A, I, ROPE_R, ball=True,
                           bend=1e-3, flaw=B.flaw_factor(fl[min(k, nlink)]), tag=1))
        prev = i
    return b, dict(beam=beam, links=links, bob=i, bonds=np.array(bnds), side=side)


def rope_period(nlink=10, angle=np.radians(20)):
    """The period of the same rope taken as a rigid compound pendulum about
    the beam, with the finite-amplitude series to the fourth power."""
    side = (5.0 / 7800.0) ** (1 / 3)
    ml = 0.1 * LINK
    I = sum(ml * ((LINK * (k + 0.5)) ** 2 + LINK ** 2 / 12) for k in range(nlink))
    db = LINK * nlink + side / 2
    I += BOB * (db ** 2 + side ** 2 / 6)
    Md = sum(ml * LINK * (k + 0.5) for k in range(nlink)) + BOB * db
    T0 = 2 * np.pi * np.sqrt(I / (Md * B.GRAVITY))
    return T0 * (1 + angle ** 2 / 16 + 11 * angle ** 4 / 3072)


# ── 3. the glass pane ────────────────────────────────────────────────────────
# src/mpm scenes.glass_edge: a 1.0 x 0.4 x 0.15 m pane, 0.7 m of it over a
# table whose edge is at x = 1.5, arriving at 6.3 m/s (a 2 m drop), started
# 5 cm above the table as tests/test_mpm.py's _glass starts it.
def glass(field=None, s=0.05):
    b = B.Bonded()
    b.wall((0.0, 0.0, -1.0), (3.2, 1.4, 0.2))           # floor
    b.wall((0.3, 0.3, 0.2), (1.5, 1.1, 1.2))            # table, edge at x=1.5
    _mpm_edges(b, (3.2, 1.4, 4.0))
    lo = (0.8, 0.5, 1.25)
    idx = B.lattice(b, lo, (1.8, 0.9, 1.40), s, B.GLASS, field=field)
    return b, dict(idx=idx, lo=np.array(lo), s=s)


# ── 4. the plank ─────────────────────────────────────────────────────────────
# (a) src/mpm scenes.plank: 1.0 x 0.4 x 0.2 m of wood, 0.35 m of it on a
# table whose top is 1.6 m up and 0.65 m off it, let go.
def _mpm_edges(b, size, dx=0.1):
    """src/mpm clamps every particle to [dx/2, size - 3dx/2] on each axis,
    which is a wall at the edge of its world; the same walls here, so that
    neither model gets further than the other."""
    X, Y, Z = size
    lo, hi = 0.5 * dx, np.array(size) - 1.5 * dx
    b.wall((-1, -1, -1), (lo, Y + 1, Z + 1))
    b.wall((hi[0], -1, -1), (X + 1, Y + 1, Z + 1))
    b.wall((-1, -1, -1), (X + 1, lo, Z + 1))
    b.wall((-1, hi[1], -1), (X + 1, Y + 1, Z + 1))


def plank_tip(s=0.05, field=None, wood=(B.WOOD_L, B.WOOD_T)):
    b = B.Bonded()
    b.wall((0.0, 0.0, -1.0), (3.2, 1.4, 0.2))           # floor
    b.wall((0.3, 0.3, 0.2), (1.3, 1.1, 1.6))            # table
    _mpm_edges(b, (3.2, 1.4, 2.6))
    idx = B.lattice(b, (0.95, 0.5, 1.6), (1.95, 0.9, 1.8), s,
                    wood, grain=(1, 0, 0), field=field)
    return b, dict(idx=idx)


# (b) a 2.0 x 0.2 x 0.1 m plank on two 20 cm supports 1.8 m apart (centre
# to centre), loaded in the middle by a force that doubles every 0.1 s.
BEND_L, BEND_B, BEND_H = 1.8, 0.2, 0.1


def plank_bend(s=0.05, field=None):
    b = B.Bonded()
    b.wall((0.0, 0.0, -1.0), (3.0, 0.6, 0.0))           # floor
    b.wall((0.5, 0.0, 0.0), (0.7, 0.6, 1.0))            # support
    b.wall((2.3, 0.0, 0.0), (2.5, 0.6, 1.0))            # support
    idx = B.lattice(b, (0.5, 0.2, 1.0), (2.5, 0.4, 1.1), s,
                    (B.WOOD_L, B.WOOD_T), grain=(1, 0, 0), field=field)
    nx = idx.shape[0]
    load = idx[nx // 2 - 1:nx // 2 + 1, :, -1].ravel()
    return b, dict(idx=idx, load=load)


def bend_theory(sig=B.WOOD_L[2], E=B.WOOD_L[0]):
    """Simply supported beam, point load in the middle: the load at which
    the outer fibre reaches `sig`, and the middle's sag per newton."""
    I = BEND_B * BEND_H ** 3 / 12
    return 4 * sig * I / (BEND_H / 2 * BEND_L), BEND_L ** 3 / (48 * E * I)
