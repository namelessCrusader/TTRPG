"""BONDED MATTER: what the seam hands to bonds, and what it reads back.

The user's pick (2026-10-10) is BOTH: the continuum (src/mpm) keeps solid
impacts and flow, and BONDS take the promoted things that must PART into
pieces: trees and their leaves, ropes, panes that split. This file is the
bonds side of the seam (src/mpm/matter.py): `claims` is the rule that says
which promoted cluster comes here, `Bonds.add` builds it, and the rest is
what `Matter` asks of any matter it carries.

A PIECE is a rigid lump of voxel cells; a MEMBER is one of its cells. The
members are what the lattice sees: each is drawn into the cell its centre is
in, and carries what that cell carried (its grams, heat, fluid, edge, the
thing it belongs to) exactly as a particle does in src/mpm. So the seam
draws, takes back and counts members with the code it uses for particles.

Every quantity here is in SI, as in src/mpm. Cells cross at the seam.
"""

import numpy as np

from . import kernel as K
from .materials import (LEAF_AREA, PETIOLE, PETIOLE_DRY, PETIOLE_R, ROPE,
                        TWIG, flaw_factor, flaw_field, pair_row, wood_row)
from ..voxel.materials import (AIR, ASH, GLASS, LEAF, ROPE as _ROPE_MAT,
                               TICK_S, WOOD, _DENS_ARR)

GRAVITY = 9.81
AIR_DENS = 1.2           # kg/m^3, the voxel sim's
DRAG_CD = 1.1            # the voxel sim's
MU = 0.4                 # friction, the voxel sim's FRICTION
# How much of a contact's spring is damped (of critical): what an impact
# loses at the surface. A face counts as ONE contact whatever number of
# members press on it (`_faces`), so a plank and a cube lose the same share.
# 0.38 gives a plank landing on stone a restitution of 0.35 on average over
# five heights (the handbook has wood on stone at about 0.4; the test's band
# is 0.3-0.45), a green crown landing still breaks its boughs off (the test
# of the real scene wants three lumps of 100 leaves or more), and the small
# tree of the ledger test lies still within its 200 ticks. 0.36 to 0.41 give
# the boughs; 0.42 and up leave the crown whole bar one bough. It was 0.5
# while a plank's sixteen members each carried a whole face's damping,
# which they no longer do. A single drop scatters about 0.1 either way: a
# contact lasts three or four substeps, so where in a substep the blow
# lands changes how much it gives back, and the tree's tumble is as touchy
# (0.39 and 0.40 leave it rolling at tick 200).
ZETA_C = 0.38
# THE CONTACT'S DAMPING IS PART OF ITS STEP BOUND. A spring of frequency w and
# a damper of rate g = c/m, stepped explicitly (symplectic Euler) at h, hold
# while (w*h)^2 + 2*g*h < 4. That is the plain bound w*h < 2 for the frequency
# w' = (g + sqrt(g^2 + 4 w^2)) / 2, and `_rows` gives each free piece's
# squared frequencies at w' for the push the contact adds. A face is damped at
# its share ZETA_C of critical, g = 2*ZETA_C*sqrt(kc/m) (the faces that press
# at once push along different axes, each a damper of its own, so one face's
# rate holds for the lot). Without it a plank, whose sixteen members press
# together, was stepped past the point where the damper itself overshoots,
# and landed with three times the energy it fell with.
# THE COST OF THE BONDED MATTER, in piece-substeps per tick. An explicit
# bond solver pays for its stiffest, lightest bond in substeps, and real wood
# at 5 cm would take thousands a tick. So all the bonded matter of a world is
# given at most BUDGET piece-substeps a tick: a rope of twenty links can
# afford real stiffness, a crown of 2,700 sprays cannot, and its stiffest
# bonds are softened just enough to fit (`_soften`). A bond of MATTER (a
# block of wood, a strip of glass) keeps the speed at which it breaks, as
# src/mpm's softening does: its strength goes down with the square root of
# its softening (an impact makes a stress rho*c*v, and c goes as sqrt(E)).
# A STALK or a ROPE keeps its handbook strength: what breaks it is the
# weight and the swing of what hangs on it, a force that softening does not
# change. Scaled like matter, a twig softened a thousandfold could not hold
# the sprays beyond it, and a crown shed 400 sprays standing still.
BUDGET = 2.0e5
NSUB_MIN = 4
NSUB_MAX = 4000
SAFETY = 0.8             # of the stable step (Gershgorin's bound)
# A lump of one material is cut into pieces about this big: a trunk into
# 20 cm blocks, as the BD prototype cut it. Glass in a pane, rope, leaves
# and broken bits are one cell to a piece.
BLOCK_M = 0.2
# A rope bends as easily as a chain: ball joints with a thousandth of a
# rod's stiffness against turning.
ROPE_BEND = 1e-3
# A PANE is glass at most this many cells thick, at least four times as
# wide as it is thick, and a full sheet (not a frame or a vessel).
PANE_THICK = 2
# The voxel ids, as ints (lattice arrays hold them as uint8).
_LEAF, _WOOD, _GLASS, _ASH = int(LEAF), int(WOOD), int(GLASS), int(ASH)
_ROPE = int(_ROPE_MAT)
_FACES = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
# the 10 diagonal directions of one half-space (6 edges, 4 corners)
_DIAG = np.array([d for d in np.ndindex(3, 3, 3)
                  if sum(abs(c - 1) for c in d) >= 2]) - 1
_DIAG = _DIAG[[tuple(d) > (0, 0, 0) for d in _DIAG]]


# `kernel.run` with no continuum cells to collide with
_NO_XW = np.full((1, 1, 1), -1, np.int64)


def _fill(mats, grams, vox_l):
    """How full each cell is, 0..1: its grams over a full cell's."""
    return np.clip(grams / np.maximum(_DENS_ARR[mats] * vox_l, 1e-9), 0.0, 1.0)


def _box(cells):
    """A dense index box over `cells`: (lo, shape, lut) where lut holds each
    cell's row, -1 elsewhere, with a margin of one cell all round."""
    lo = cells.min(axis=0) - 1
    shape = tuple(int(v) for v in cells.max(axis=0) - lo + 2)
    lut = np.full(shape, -1, np.int64)
    lut[tuple((cells - lo).T)] = np.arange(len(cells))
    return lo, lut


def claims(w, cells, named=False):
    """THE RULE: does this promoted cluster go to bonds, and as what?

    - "crown": a LEAF cell shares a face with a WOOD cell of the cluster.
      Leaves hang on stalks, and a crown that falls must be able to shed
      them one spray at a time.
    - "rope": it has ROPE in it (the material: twisted fibre, which holds
      in tension and hangs). It was "a strand of wood under 15% full", and
      that caught a thin dowel, an arrow shaft or a stick burned thin.
    - "pane": it is all GLASS, at most PANE_THICK cells thick, at least four
      times as wide as that, and a full sheet. A pane must split where it is
      bent and leave part of itself behind.
    Anything else (a plank, a pole, a vat, a stone, a body) is None: it stays
    on the continuum, which lands a solid thing whole.

    A NAMED THING comes here only when `named` (the seam's `bonds_named`,
    True since things.py finds a thing's matter through `Matter.points`);
    without it every cell must have eid 0."""
    cells = np.asarray(cells, np.int64).reshape(-1, 3)
    if not len(cells):
        return None
    idx = tuple(cells.T)
    mats = w.mat[idx].astype(np.int64)
    keep = mats != int(AIR)
    cells, mats = cells[keep], mats[keep]
    if len(cells) < 2:
        return None
    weid = getattr(w, "eid", None)
    if not named and weid is not None and (weid[tuple(cells.T)] != 0).any():
        return None
    lo, lut = _box(cells)
    rel = cells - lo
    leaf = mats == _LEAF
    if leaf.any() and (mats == _WOOD).any():
        for e in _FACES:
            for s in (1, -1):
                nb = lut[tuple((rel + s * e).T)]
                ok = nb >= 0
                if (leaf[ok] & (mats[nb[ok]] == _WOOD)).any():
                    return "crown"
    if (mats == _ROPE).any():
        return "rope"
    if (mats == _GLASS).all():
        ext = np.sort(cells.max(axis=0) - cells.min(axis=0) + 1)
        if (ext[0] <= PANE_THICK and ext[1] >= 4 * ext[0]
                and len(cells) >= 0.75 * float(np.prod(ext))):
            return "pane"
    return None


def _grain(cells, wood, r=3):
    """The grain of each wood cell: the long axis of the wood of this
    thing within `r` cells of it (the principal axis of their positions).
    A trunk's grain runs up it, a branch's along it. Not a lattice field:
    it is read off the promoted cells, every time."""
    out = np.zeros((len(cells), 3))
    out[:, 2] = 1.0
    wc = cells[wood]
    if not len(wc):
        return out
    lo, lut = _box(wc)
    occ = lut >= 0
    for i in np.flatnonzero(wood):
        c = cells[i] - lo
        sl = tuple(slice(max(int(c[a]) - r, 0), int(c[a]) + r + 1) for a in range(3))
        pts = np.argwhere(occ[sl]).astype(np.float64)
        if len(pts) < 2:
            continue
        pts -= pts.mean(axis=0)
        val, vec = np.linalg.eigh(pts.T @ pts)
        if val[2] - val[1] < 1e-9:
            continue                     # no long axis: keep "up"
        g = vec[:, 2]
        out[i] = g if g[np.argmax(np.abs(g))] > 0 else -g
    return out


def _qmat(Rm):
    """The unit quaternion (w, x, y, z) of a rotation matrix."""
    t = np.trace(Rm)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = (0.25 * s, (Rm[2, 1] - Rm[1, 2]) / s, (Rm[0, 2] - Rm[2, 0]) / s,
             (Rm[1, 0] - Rm[0, 1]) / s)
    elif Rm[0, 0] > Rm[1, 1] and Rm[0, 0] > Rm[2, 2]:
        s = np.sqrt(1.0 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2]) * 2
        q = ((Rm[2, 1] - Rm[1, 2]) / s, 0.25 * s, (Rm[0, 1] + Rm[1, 0]) / s,
             (Rm[0, 2] + Rm[2, 0]) / s)
    elif Rm[1, 1] > Rm[2, 2]:
        s = np.sqrt(1.0 + Rm[1, 1] - Rm[0, 0] - Rm[2, 2]) * 2
        q = ((Rm[0, 2] - Rm[2, 0]) / s, (Rm[0, 1] + Rm[1, 0]) / s, 0.25 * s,
             (Rm[1, 2] + Rm[2, 1]) / s)
    else:
        s = np.sqrt(1.0 + Rm[2, 2] - Rm[0, 0] - Rm[1, 1]) * 2
        q = ((Rm[1, 0] - Rm[0, 1]) / s, (Rm[0, 2] + Rm[2, 0]) / s,
             (Rm[1, 2] + Rm[2, 1]) / s, 0.25 * s)
    q = np.array(q)
    return q / np.linalg.norm(q)


def _qmul(a, b):
    return np.array(K._qmul(*a, *b))


def _qconj(a):
    return np.array((a[0], -a[1], -a[2], -a[3]))


def _rotations(q):
    R = np.zeros((len(q), 3, 3))
    if len(q):
        K.q2R(np.ascontiguousarray(q), R)
    return R


def _frame(pos, kg, dx):
    """A rigid lump of member cubes (side `dx`) at world points `pos`, of
    masses `kg`: its centre, its principal moments, and the rotation from
    its principal frame to the world (det +1)."""
    M = float(kg.sum())
    c = (kg[:, None] * pos).sum(axis=0) / M
    r = pos - c
    Iw = np.eye(3) * (kg * (r * r).sum(axis=1)).sum() - (kg[:, None, None]
                                                         * r[:, :, None] * r[:, None, :]).sum(0)
    Iw += np.eye(3) * M * dx * dx / 6.0
    val, vec = np.linalg.eigh(Iw)
    if np.linalg.det(vec) < 0:
        vec[:, 2] = -vec[:, 2]
    return c, np.maximum(val, 1e-12), vec


# Every per-member array, in the order `keep` filters them (the seam reads
# the particle-shaped ones by the names src/mpm gives its particles).
MEMBER_FIELDS = ("x", "v", "m", "vol0", "vmat", "eid", "batch", "E", "fl",
                 "fvol", "fpot", "edge", "dmg", "nap", "mp", "off", "mrad",
                 "kcm", "ccm", "mtie")
# OPTIONAL MEMBER FIELDS (wave 13), the same four src/mpm gives a particle
# (`PARTICLE_OPT`): mix ids of the solid and of the fluid, damage as a float,
# grams of ice. They are not in MEMBER_FIELDS: an array that does not exist
# costs nothing. `ensure_opt` makes one (zeros); `add` and `keep` carry every
# one that exists (`opt_fields`).
MEMBER_OPT = (("scomp", np.uint32), ("fcomp", np.uint32),
              ("dscal", np.float32), ("ice_g", np.float32),
              ("wood_sp", np.uint8), ("wood_grain", np.uint8),
              ("wood_bound", np.float32))
PIECE_FIELDS = ("px", "pv", "pq", "pell", "pm", "pI", "pIbinv", "pcda",
                "pkin", "pfext", "pbatch", "pkind", "panchor", "pnsub", "pface")
BOND_FIELDS = ("bi", "bj", "ai", "aj", "Q0", "kn", "ks", "kb", "kt", "cn",
               "cs", "cb", "ct", "A", "rb", "Ib", "Jb", "sig", "tau", "alive",
               "tbreak", "smax", "soft", "ball", "btag", "zeta")
_KINDS = ("crown", "rope", "pane")


class _Drawn:
    """What the seam's last `rasterize` gave the lattice from bonds: the
    same bookkeeping `Matter` keeps for the particles (see `_written`)."""

    def __init__(self):
        self.drawn_at = np.zeros(0, np.int64)
        self.drawn_eid = np.zeros(0, np.int32)
        self.drawn_was = {}
        self.drawn_who = np.zeros(0, np.int64)
        self.drawn_in = np.zeros(0, np.int64)


class Bonds:
    """All the bonded matter of one world: pieces, their members (cells),
    and the bonds between pieces. Plain numpy arrays, so the world's
    snapshot copies it and golden hashes it like anything else."""

    def __init__(self, shape, dx):
        self.shape = tuple(int(s) for s in shape)
        self.dx = float(dx)
        self.inv_dx = 1.0 / self.dx
        self.t = 0.0
        self.substeps = 0
        self.h = TICK_S
        self.g = GRAVITY
        z3 = lambda: np.zeros((0, 3))
        # members (cells): what the seam reads, by src/mpm's particle names
        self.x, self.v, self.m, self.vol0 = z3(), z3(), np.zeros(0), np.zeros(0)
        self.vmat = np.zeros(0, np.int16)
        self.eid = np.zeros(0, np.int32)
        self.batch = np.zeros(0, np.int32)
        self.E, self.fvol, self.fpot = np.zeros(0), np.zeros(0), np.zeros(0)
        self.fl = np.zeros(0, np.int16)
        self.edge = np.zeros(0)
        self.dmg = np.zeros(0, bool)
        self.nap = np.zeros(0, np.int32)         # -1: awake (src/mpm's meaning)
        self.mp = np.zeros(0, np.int64)          # the piece each is in
        self.off = z3()                          # where in it, in its own frame
        self.mrad, self.kcm, self.ccm = np.zeros(0), np.zeros(0), np.zeros(0)
        self.mtie = np.zeros((0, 3), np.int64)   # the lattice cell it is tied to
        self._opt = ()                           # the MEMBER_OPT fields that exist
        # pieces
        self.px, self.pv, self.pell = z3(), z3(), z3()
        self.pq = np.zeros((0, 4))
        self.pm, self.pcda = np.zeros(0), np.zeros(0)
        self.pI, self.pIbinv, self.pfext = z3(), z3(), z3()
        self.pkin = np.zeros(0, bool)
        self.pbatch = np.zeros(0, np.int32)
        self.pkind = np.zeros(0, np.int8)        # index into _KINDS
        self.panchor = np.zeros((0, 3), np.int64)
        self.pnsub = np.zeros(0, np.int64)       # the substeps its thing can afford
        self.pface = np.zeros(0)                 # members one face of it presses with
        # bonds
        self.bi, self.bj = np.zeros(0, np.int64), np.zeros(0, np.int64)
        self.ai, self.aj, self.Q0 = z3(), z3(), np.zeros((0, 4))
        for f in ("kn", "ks", "kb", "kt", "cn", "cs", "cb", "ct", "A", "rb",
                  "Ib", "Jb", "sig", "tau", "tbreak", "smax", "soft", "zeta"):
            setattr(self, f, np.zeros(0))
        self.alive = np.zeros(0, bool)
        self.ball = np.zeros(0, bool)
        self.btag = np.zeros(0, np.int8)         # 0 matter, 1 rope, 2 stalk
        self.adj_ptr = np.zeros(1, np.int64)
        self.adj_b = np.zeros(0, np.int64)
        # THE LEDGERS, [P(3), L(3)] about the world's origin: what came in
        # through `add`, what outside pushes gave (the kernel's ledger), and
        # what left with matter the lattice took back (`_reweigh`). The
        # momentum of the pieces is always in + pushed - carried.
        self.inflow = np.zeros(6)
        self.ledger = np.zeros(6)
        self.carried = np.zeros(6)
        # ...and the part of `ledger` the continuum's particles gave, as
        # walls and as the push back for what these pieces did to them
        # (`Matter.cross`): the seam books the same impulse, reversed, on
        # the particles' side, so the two solvers' momentum adds up exactly
        self.cross = np.zeros(6)
        self.drawn = _Drawn()
        self.asleep = False
        self._still = 0
        self._nap_walls = None                   # (lo, hi, walls) asleep
        self.broken_last = np.zeros(0, np.int64)  # bonds the last step broke

    def __len__(self):
        return len(self.m)

    # ------------------------------------------------------------ building

    def add(self, cells, mats, grams, E, fl, fvol, fpot, edge, eids, rubble,
            vel, batch, kind, vox_l, anchors=()):
        """Make pieces, members and bonds of one promoted cluster (the
        lattice has already let go of it). `vel` is m/s. `anchors` are
        (row of a cell, the lattice cell it is tied to, 1 for a rope's knot
        or 2 for a leaf's stalk): see `_anchor`. A cell some bonded thing is
        already tied to joins that thing (`_join`). Returns how many members
        it made."""
        cells = np.asarray(cells, np.int64)
        mats = np.asarray(mats, np.int64)
        grams = np.asarray(grams, np.float64)
        fvol = np.asarray(fvol, np.float64)
        n = len(cells)
        dx = self.dx
        kg = grams / 1000.0
        fill = _fill(mats, grams, vox_l)
        pos = (cells + 0.5) * dx
        leaf = mats == _LEAF
        loose = np.asarray(rubble, bool) | (mats == _ASH)
        thin = mats == _ROPE             # a rope's cells: one link each
        single = leaf | thin | loose | ((mats == _GLASS) & (kind == "pane"))
        # ---- pieces: single cells, and lumps cut into blocks
        piece = np.full(n, -1, np.int64)
        piece[single] = np.arange(int(single.sum()))
        npc = int(single.sum())
        lo, lut = _box(cells)
        rel = cells - lo
        B = max(1, int(round(BLOCK_M / dx)))
        blk = (cells - cells.min(axis=0)) // B
        up = np.arange(n)

        def root(a):
            while up[a] != a:
                up[a] = up[up[a]]
                a = up[a]
            return a
        for e in _FACES:
            nb = lut[tuple((rel + e).T)]
            for a in np.flatnonzero(nb >= 0):
                b = int(nb[a])
                if (single[a] or single[b] or mats[a] != mats[b]
                        or (blk[a] != blk[b]).any()):
                    continue
                up[root(a)] = root(b)
        lumps = {}
        for a in np.flatnonzero(~single):
            r = root(a)
            if r not in lumps:
                lumps[r] = npc
                npc += 1
            piece[a] = lumps[r]
        # ---- each piece's frame
        P0 = len(self.pm)
        cen = np.zeros((npc, 3))
        Ip = np.zeros((npc, 3))
        qp = np.zeros((npc, 4))
        mpc = np.zeros(npc)
        off = np.zeros((n, 3))
        nmem = np.bincount(piece, minlength=npc)
        order = np.argsort(piece, kind="stable")
        cut = np.searchsorted(piece[order], np.arange(npc + 1))
        for p in range(npc):
            s = order[cut[p]:cut[p + 1]]
            w_ = np.maximum(kg[s], 1e-12)
            c, I, Rm = _frame(pos[s], w_, dx)
            cen[p], Ip[p], qp[p] = c, I, _qmat(Rm)
            mpc[p] = float(kg[s].sum())
            off[s] = (pos[s] - c) @ Rm
        mpc = np.maximum(mpc, 1e-9)
        nface = np.maximum(nmem.astype(float) ** (2.0 / 3.0), 1.0)
        area = np.where(np.bincount(piece, leaf.astype(float), npc) > 0,
                        np.bincount(piece, np.where(leaf, LEAF_AREA * grams, 0.0), npc),
                        dx * dx * nface)
        # ---- bonds between pieces
        grain = _grain(cells, mats == _WOOD)
        field = flaw_field(tuple(int(v) for v in rel.max(axis=0) + 3),
                           int(eids.min()) if len(eids) else 0)
        pairs = []
        for e in _FACES:
            nb = lut[tuple((rel + e).T)]
            for a in np.flatnonzero(nb >= 0):
                if not (leaf[a] or leaf[int(nb[a])]):
                    pairs.append((a, int(nb[a])))
        # LEAVES HANG ON STALKS, ONE EACH. From the wood outwards (the
        # cluster's wood, and the lattice's wood it is tied to), each leaf
        # cell's stalk goes to the neighbour one step nearer the wood; a
        # leaf the wood cannot reach that way hangs on nothing.
        once = set()                     # a leaf hangs on ONE stalk
        anchors = [(cl, lat, tag) for cl, lat, tag in anchors
                   if tag != 2 or not (cl in once or once.add(cl))]
        parent = np.full(n, -1, np.int64)
        tied = np.zeros(n, bool)
        for cl, _lat, tag in anchors:
            if tag == 2:
                tied[cl] = True
        reach = np.zeros(n, bool)
        reach[(mats == _WOOD) & ~loose] = True
        front = list(np.flatnonzero(tied & leaf & ~loose))
        reach[front] = True
        while True:
            new = []
            for e in _FACES:
                for sg in (1, -1):
                    nb = lut[tuple((rel + sg * e).T)]
                    for a in np.flatnonzero((nb >= 0) & leaf & ~loose & ~reach):
                        if reach[nb[a]] and parent[a] < 0 and not (tied[a]):
                            parent[a] = nb[a]
                            new.append(a)
            if not new:
                break
            reach[new] = True
        # each stalk carries its own spray and every one beyond it
        carry = np.where(leaf, 1.0, 0.0)
        hang = np.flatnonzero(parent >= 0)
        depth = np.zeros(n, np.int64)
        for a in hang:
            d, c = 0, a
            while parent[c] >= 0 and leaf[parent[c]]:
                c = parent[c]
                d += 1
            depth[a] = d
        for a in hang[np.argsort(-depth[hang], kind="stable")]:
            if leaf[parent[a]]:
                carry[parent[a]] += carry[a]
        for a in hang:
            pairs.append((a, int(parent[a])))
        stick = ~leaf & ~loose & ~thin
        for d in _DIAG:
            nb = lut[tuple((rel + d).T)]
            for a in np.flatnonzero((nb >= 0) & stick):
                b = int(nb[a])
                if not stick[b]:
                    continue
                # only where the two touch by an edge or a corner alone: no
                # face path of this thing between them
                between = False
                for k in range(3):
                    if d[k] == 0:
                        continue
                    for c in (rel[a] + d * (np.arange(3) == k), rel[b] - d * (np.arange(3) == k)):
                        j = lut[tuple(c)]
                        if j >= 0 and stick[j]:
                            between = True
                if not between:
                    pairs.append((a, b))
        agg = {}
        for a, b in pairs:
            pa, pb = int(piece[a]), int(piece[b])
            if pa == pb or loose[a] or loose[b]:
                continue
            if pa > pb:
                a, b, pa, pb = b, a, pb, pa
            if leaf[a] or leaf[b]:
                kid = a if leaf[a] and parent[a] == b else b
                ar = min(np.pi * PETIOLE_R ** 2 * carry[kid], dx * dx)
                tag = 3 if carry[kid] > 1.0 else (2 if fvol[kid] > 0.0 else 4)
            elif thin[a] or thin[b]:
                g_ = min(grams[a] if thin[a] else 1e30, grams[b] if thin[b] else 1e30)
                ar, tag = g_ / 1000.0 / (ROPE[4] * dx), 1
            else:
                ar, tag = dx * dx * min(fill[a], fill[b]), 0
            mid = np.minimum((rel[a] + rel[b]) // 2, np.array(field.shape) - 1)
            fl_ = flaw_factor(field[tuple(mid)])
            gsum = np.zeros(3)
            for c in (a, b):
                if mats[c] == _WOOD:
                    gc = grain[c]
                    gsum += gc if gsum @ gc >= 0 else -gc
            r = agg.setdefault((pa, pb), [0.0, 0.0, 0, np.zeros(3), tag,
                                          int(mats[a]), int(mats[b])])
            r[0] += ar
            r[1] += fl_
            r[2] += 1
            r[3] += gsum
        rows = []
        for (pa, pb), (ar, fsum, cnt, gsum, tag, ma, mb) in sorted(agg.items()):
            d = cen[pb] - cen[pa]
            L = float(np.linalg.norm(d))
            if L < 1e-9:
                continue
            nvec = d / L
            gn = np.linalg.norm(gsum)
            cos2 = float((nvec @ gsum / gn) ** 2) if gn > 0 else 1.0
            if tag in (2, 3, 4):
                row = {2: PETIOLE, 3: TWIG, 4: PETIOLE_DRY}[tag]
                I_ = ar * ar / (4.0 * np.pi)
                rb_ = np.sqrt(ar / np.pi)
                tag = 2
            elif tag == 1:
                row = ROPE
                I_ = ar * ar / (4.0 * np.pi)
                rb_ = np.sqrt(ar / np.pi)
            else:
                row = wood_row(cos2) if ma == mb == _WOOD else pair_row(ma, mb, cos2)
                if row is None:
                    continue
                I_ = ar * ar / 12.0
                rb_ = np.sqrt(ar) / 2.0
            rows.append((pa, pb, row, ar, I_, rb_, L, fsum / cnt, tag))
        # ---- into the arrays: pieces, members, bonds
        self._extend_pieces(cen, np.repeat(np.asarray(vel, float)[None], npc, 0),
                            qp, mpc, Ip, DRAG_CD * area, batch, _KINDS.index(kind),
                            face=self._faces(rel, piece, npc))
        mem = dict(x=pos.copy(), v=np.repeat(np.asarray(vel, float)[None], n, 0),
                   m=kg.copy(), vol0=np.where(kg > 0, kg / np.maximum(
                       _DENS_ARR[mats], 1e-9), dx ** 3),
                   vmat=mats.astype(np.int16), eid=np.asarray(eids, np.int32),
                   batch=np.full(n, batch, np.int32),
                   E=np.asarray(E, np.float64), fl=np.asarray(fl, np.int16),
                   fvol=np.asarray(fvol, np.float64), fpot=np.asarray(fpot, np.float64),
                   edge=np.asarray(edge, np.float64), dmg=np.asarray(rubble, bool).copy(),
                   nap=np.full(n, -1, np.int32), mp=piece + P0, off=off,
                   mrad=np.full(n, 0.5 * dx), kcm=np.zeros(n), ccm=np.zeros(n),
                   mtie=np.full((n, 3), -1, np.int64))
        for f in MEMBER_FIELDS:
            setattr(self, f, np.concatenate([getattr(self, f), mem[f]]))
        for f, dt in self.opt_fields():
            setattr(self, f, np.concatenate([getattr(self, f), np.zeros(n, dt)]))
        if rows:
            cols = list(zip(*rows))
            self._bonds(P0 + np.array(cols[0], np.int64), P0 + np.array(cols[1], np.int64),
                        np.array(cols[2]), np.array(cols[3]), np.array(cols[4]),
                        np.array(cols[5]), np.array(cols[6]), np.array(cols[7]),
                        np.array(cols[8], np.int8))
        for cl, lat, tag in anchors:
            at = np.minimum(rel[cl], np.array(field.shape) - 1)
            self._anchor(int(piece[cl]) + P0, lat, grams[cl], tag, carry[cl],
                         bool(fvol[cl] <= 0.0), flaw_factor(field[tuple(at)]))
        self._join(cells, piece + P0)
        self._adjacency()
        self._soften()
        self.retime()
        P, L = self._momentum(np.arange(P0, P0 + npc))
        self.inflow += np.concatenate([P, L])
        self.asleep = False
        self._still = 0
        return n

    def _anchor(self, p, lat, grams, tag, carry=1.0, dry=False, flaw=1.0):
        """A BONDED THING CAN BE TIED TO THE LATTICE, by what holds in
        tension, which the lattice itself has none of (support never comes
        from above):
        - a ROPE's cell that touches a solid cell of the lattice is knotted
          to it (tag 1): a ball joint of the rope itself, so a rope can hang
          from a beam;
        - a LEAF cell that touches WOOD of the lattice hangs on it by a
          stalk (tag 2), as it hangs on the wood it came with: a crown that
          leaves the lattice still hangs on the trunk that stays.
        The other end is an ANCHOR, a pinned piece at that cell's centre
        that never moves; what its bond pulls is an outside push on the
        ledger. When the cell goes (cut, burned), the bond parts (`untie`);
        when it is promoted, it joins this thing (`_join`). Wood is never
        tied to wood here: the lattice decided that wood let go."""
        lat = np.asarray(lat, np.int64)
        old = np.flatnonzero(self.pkin & (self.panchor == lat).all(axis=1))
        if len(old):
            a = int(old[0])
        else:
            self._extend_pieces(((lat + 0.5) * self.dx)[None], np.zeros((1, 3)),
                                np.array([[1.0, 0.0, 0.0, 0.0]]), np.ones(1),
                                np.ones((1, 3)), np.zeros(1), -1, self.pkind[p])
            a = len(self.pm) - 1
            self.pkin[a] = True
            self.panchor[a] = lat
        self.mtie[self.mp == p] = lat
        L = float(np.linalg.norm(self.px[p] - self.px[a]))
        if tag == 1:
            ar = float(grams) / 1000.0 / (ROPE[4] * self.dx)
            row, I_, rb_ = ROPE, ar * ar / (4.0 * np.pi), np.sqrt(ar / np.pi)
        else:
            ar = min(np.pi * PETIOLE_R ** 2 * carry, self.dx * self.dx)
            row = TWIG if carry > 1.0 else (PETIOLE if dry is False else PETIOLE_DRY)
            I_, rb_ = ar * ar / (4.0 * np.pi), np.sqrt(ar / np.pi)
        self._bonds(np.array([a]), np.array([p]), np.array([row]), np.array([ar]),
                    np.array([I_]), np.array([rb_]), np.array([L]), np.array([flaw]),
                    np.array([tag], np.int8))

    def joins(self, shape, cells):
        """The kind of the bonded thing tied to one of `cells` (a cluster
        about to be promoted), or None: such a cluster joins it."""
        a = np.flatnonzero(self.pkin & (self.panchor >= 0).all(axis=1))
        if not len(a) or not len(cells):
            return None
        at = np.ravel_multi_index(tuple(self.panchor[a].T), shape)
        hit = np.isin(at, np.ravel_multi_index(tuple(np.asarray(cells).T), shape))
        if not hit.any():
            return None
        return _KINDS[int(self.pkind[a[np.argmax(hit)]])]

    def _join(self, cells, piece):
        """The new cells some anchors stood for are bonded matter now: each
        bond to such an anchor is moved onto the new piece that holds the
        cell, at the same point and at rest as things stand, and the anchor
        is retired (pinned, tied to nothing). Returns the bonds it moved."""
        moved = []
        a_all = np.flatnonzero(self.pkin & (self.panchor >= 0).all(axis=1))
        if not len(a_all):
            return np.zeros(0, np.int64)
        lut = {tuple(c): int(p) for c, p in zip(cells.tolist(), piece)}
        R = _rotations(self.pq)
        for a in a_all:
            p = lut.get(tuple(self.panchor[a].tolist()))
            if p is None:
                continue
            for b in np.flatnonzero(self.alive & ((self.bi == a) | (self.bj == a))):
                if self.bi[b] == a:
                    at = self.px[a] + R[a] @ self.ai[b]
                    self.bi[b] = p
                    self.ai[b] = R[p].T @ (at - self.px[p])
                else:
                    at = self.px[a] + R[a] @ self.aj[b]
                    self.bj[b] = p
                    self.aj[b] = R[p].T @ (at - self.px[p])
                self.Q0[b] = _qmul(_qconj(self.pq[self.bi[b]]), self.pq[self.bj[b]])
                moved.append(b)
            self.mtie[(self.mtie == self.panchor[a]).all(axis=1)] = -1
            self.panchor[a] = -1
        moved = np.array(moved, np.int64)
        if len(moved):
            self._damp(moved)
        return moved

    def untie(self, wall):
        """Knots whose lattice cell is no longer solid part. Returns the
        bonds that parted."""
        a = np.flatnonzero(self.pkin & (self.panchor >= 0).all(axis=1))
        if not len(a):
            return np.zeros(0, np.int64)
        c = self.panchor[a]
        inside = ((c >= 0) & (c < np.array(wall.shape))).all(axis=1)
        held = np.zeros(len(a), bool)
        held[inside] = wall[tuple(c[inside].T)]
        if held.all():
            return np.zeros(0, np.int64)
        for cc in c[~held]:
            self.mtie[(self.mtie == cc).all(axis=1)] = -1
        self.panchor[a[~held]] = -1
        return self.cut(a[~held])

    def opt_fields(self):
        """The MEMBER_OPT rows whose arrays exist (none, unless asked)."""
        return tuple(r for r in MEMBER_OPT if r[0] in self._opt)

    def ensure_opt(self, name):
        """Make the optional per-member array `name` (zeros) if it is not
        there yet, and return it."""
        if name not in self._opt:
            for f, dt in MEMBER_OPT:
                if f == name:
                    setattr(self, f, np.zeros(len(self.x), dt))
                    self._opt = self._opt + (f,)
                    break
            else:
                raise KeyError(name)
        return getattr(self, name)

    def ensure_opt_all(self):
        for f, _ in MEMBER_OPT:
            self.ensure_opt(f)

    def _extend_pieces(self, x, v, q, m, I, cda, batch, kind, face=None):
        k = len(m)
        new = dict(px=x, pv=v, pq=q, pell=np.zeros((k, 3)), pm=m, pI=I,
                   pIbinv=1.0 / I, pcda=cda, pkin=np.zeros(k, bool),
                   pfext=np.zeros((k, 3)), pbatch=np.full(k, batch, np.int32),
                   pkind=np.full(k, kind, np.int8),
                   panchor=np.full((k, 3), -1, np.int64),
                   pnsub=np.full(k, NSUB_MAX, np.int64),
                   pface=np.ones(k) if face is None else np.asarray(face, float))
        for f in PIECE_FIELDS:
            setattr(self, f, np.concatenate([getattr(self, f), new[f]]))

    def _bonds(self, i, j, rows, area, inertia, radius, L, flaw, tag):
        """Beams from pieces `i` to pieces `j` (arrays), each held at the
        midpoint of its two pieces' centres, at REAL stiffness (`_soften`
        softens them later, if it must). `rows` is (k, 5) material rows."""
        k = len(i)
        if not k:
            return
        rows = np.asarray(rows, np.float64).reshape(k, 5)
        E, nu, S, zeta = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
        G = E / (2.0 * (1.0 + nu))
        J = 2.0 * inertia
        ball = tag == 1
        R = _rotations(self.pq)
        c = 0.5 * (self.px[i] + self.px[j])
        ai = np.einsum("nji,nj->ni", R[i], c - self.px[i])
        aj = np.einsum("nji,nj->ni", R[j], c - self.px[j])
        Q0 = np.array([_qmul(_qconj(self.pq[a]), self.pq[b])
                       for a, b in zip(i, j)]).reshape(k, 4)
        kn = E * area / L
        ks = np.where(ball, kn, G * area / L)
        bend = np.where(ball, ROPE_BEND, 1.0)
        z = np.zeros(k)
        new = dict(bi=i, bj=j, ai=ai, aj=aj, Q0=Q0, kn=kn, ks=ks,
                   kb=E * inertia / L * bend, kt=G * J / L * bend,
                   cn=z, cs=z, cb=z, ct=z, A=area, rb=radius, Ib=inertia, Jb=J,
                   sig=S * flaw, tau=S * flaw, alive=np.ones(k, bool),
                   tbreak=np.full(k, np.nan), smax=z, soft=np.ones(k),
                   ball=ball, btag=tag, zeta=zeta)
        b0 = len(self.bi)
        for f in BOND_FIELDS:
            a = getattr(self, f)
            setattr(self, f, np.concatenate([a, np.asarray(new[f], a.dtype)]))
        self._damp(np.arange(b0, len(self.bi)))

    def _damp(self, bs):
        """A dashpot beside each spring of bonds `bs`: its row's share of
        critical for the two pieces' reduced mass (and inertia)."""
        if not len(bs):
            return
        i, j = self.bi[bs], self.bj[bs]
        mi = np.where(self.pkin[i], 1e30, self.pm[i])
        mj = np.where(self.pkin[j], 1e30, self.pm[j])
        mr = mi * mj / (mi + mj)
        Ii = np.where(self.pkin[i], 1e30, self.pI[i].min(axis=1))
        Ij = np.where(self.pkin[j], 1e30, self.pI[j].min(axis=1))
        Ir = Ii * Ij / (Ii + Ij)
        z = self.zeta[bs]
        self.cn[bs] = 2 * z * np.sqrt(self.kn[bs] * mr)
        self.cs[bs] = 2 * z * np.sqrt(self.ks[bs] * mr)
        self.cb[bs] = 2 * z * np.sqrt(self.kb[bs] * Ir)
        self.ct[bs] = 2 * z * np.sqrt(self.kt[bs] * Ir)

    def _adjacency(self):
        """Which bonds each piece has (the "already bonded" contact test)."""
        n = len(self.pm)
        deg = np.bincount(np.concatenate([self.bi, self.bj]), minlength=n)
        self.adj_ptr = np.concatenate([[0], np.cumsum(deg)]).astype(np.int64)
        fillp = self.adj_ptr[:-1].copy()
        self.adj_b = np.zeros(2 * len(self.bi), np.int64)
        for b in range(len(self.bi)):
            for e in (self.bi[b], self.bj[b]):
                self.adj_b[fillp[e]] = b
                fillp[e] += 1

    def _contacts(self, ps):
        """Contact stiffness of pieces `ps`, spread over their members: as
        stiff as twice the piece's stiffest bond, so an impact meets the
        matter's own stiffness (a piece with no bond: a 300 Hz contact); a
        face of members resting on something adds up to that."""
        if not len(ps):
            return
        kc, nface = self._kc()
        mine = np.isin(self.mp, ps)
        p = self.mp[mine]
        self.kcm[mine] = kc[p] / nface[p]
        self.ccm[mine] = 2 * ZETA_C * np.sqrt(self.kcm[mine] * self.pm[p] / nface[p])

    def _kc(self):
        """Each piece's contact stiffness (N/m, see `_contacts`) and how
        many of its members one face of it holds."""
        kk = np.maximum(self.kn, self.ks) * self.alive
        kmax = np.zeros(len(self.pm))
        np.maximum.at(kmax, self.bi, kk)
        np.maximum.at(kmax, self.bj, kk)
        kc = np.where(kmax > 0, 2.0 * kmax, self.pm * (2 * np.pi * 300.0) ** 2)
        return kc, self.pface

    @staticmethod
    def _faces(rel, piece, npc):
        """How many members one face of each new piece presses on a surface
        with at once: the most cells it shows along any of the lattice's
        three axes (`rel` are its members' cells, `piece` their pieces). A
        cube of 64 shows 16; a plank of 16 cells one deep shows all 16,
        where a cube's rule (the 2/3 power of the count) said 6, and sixteen
        members each pushing with a sixth of the face's stiffness and
        damping made the plank two and a half times stiffer and damped than
        its step was bound for. Counted in the lattice's own cells, where a
        piece's principal frame is no guide (a cube has no principal axes)."""
        face = np.ones(npc)
        side = int(rel.max()) + 1 if len(rel) else 1
        for a in range(3):
            b, c = (a + 1) % 3, (a + 2) % 3
            key = (piece * side + rel[:, b]) * side + rel[:, c]
            seen = np.bincount(np.unique(key) // (side * side), minlength=npc)
            face = np.maximum(face, seen[:npc])
        return face

    def _rows(self, per_bond=False):
        """Gershgorin's bound on each free piece's squared frequencies, per
        degree of freedom (n, 6): from each bond's exact 12 x 12 stiffness
        at rest (both pieces' push and turn), weighted by mass and by
        inertia about each piece's own axes, plus three times the piece's
        contact spring (three faces against something at once). A true
        upper bound for the bonds, so the step it gives is stable.
        With `per_bond`, also (live bonds, what each adds to its first
        piece's largest row, and to its second's)."""
        n = len(self.pm)
        R = _rotations(self.pq)
        free = ~self.pkin
        m = np.where(free, self.pm, 1.0)
        im = np.where(free, 1.0 / np.sqrt(m), 0.0)
        I = np.where(free[:, None], self.pI, 1.0)
        W = np.einsum("nij,nj,nkj->nik", R, 1.0 / np.sqrt(I), R) * free[:, None, None]
        rows = np.zeros((n, 6))
        kc, _ = self._kc()
        rows[:, :3] += (3.0 * kc / m)[:, None] * free[:, None]
        on = np.flatnonzero(self.alive)
        if len(on):
            bi, bj = self.bi[on], self.bj[on]
            nv = self.px[bj] - self.px[bi]
            nv /= np.maximum(np.linalg.norm(nv, axis=1, keepdims=True), 1e-12)
            nn = nv[:, :, None] * nv[:, None, :]
            I3 = np.eye(3)[None]
            Kg = self.kn[on, None, None] * nn + self.ks[on, None, None] * (I3 - nn)
            Kt = self.kb[on, None, None] * (I3 - nn) + self.kt[on, None, None] * nn
            ri = np.einsum("nij,nj->ni", R[bi], self.ai[on])
            rj = np.einsum("nij,nj->ni", R[bj], self.aj[on])

            def skew(r):
                z = np.zeros(len(r))
                return np.stack([np.stack([z, -r[:, 2], r[:, 1]], 1),
                                 np.stack([r[:, 2], z, -r[:, 0]], 1),
                                 np.stack([-r[:, 1], r[:, 0], z], 1)], 1)
            k = len(on)
            E3 = np.repeat(I3, k, 0)
            Bm = np.concatenate([-E3, skew(ri), E3, -skew(rj)], 2)
            Cm = np.concatenate([np.zeros((k, 3, 3)), -E3, np.zeros((k, 3, 3)), E3], 2)
            Km = np.einsum("nai,nab,nbj->nij", Bm, Kg, Bm) + \
                np.einsum("nai,nab,nbj->nij", Cm, Kt, Cm)
            D = np.zeros((k, 12, 12))
            for blk, (p, w) in enumerate(((bi, None), (bi, 1), (bj, None), (bj, 1))):
                sl = slice(3 * blk, 3 * blk + 3)
                D[:, sl, sl] = (np.eye(3)[None] * im[p][:, None, None]) if w is None else W[p]
            Kn = np.abs(np.einsum("nij,njk,nkl->nil", D, Km, D)).sum(2)
            np.add.at(rows, bi, Kn[:, :6])
            np.add.at(rows, bj, Kn[:, 6:])
        # the contact's damper, in the same rows: see "THE CONTACT'S DAMPING"
        g_ = 2.0 * ZETA_C * np.sqrt(kc / m)[:, None] * free[:, None]
        rows[:, :3] = (0.5 * (g_ + np.sqrt(g_ * g_ + 4.0 * rows[:, :3]))) ** 2
        if per_bond:
            if len(on):
                return rows, on, Kn[:, :6].max(axis=1), Kn[:, 6:].max(axis=1)
            z = np.zeros(0)
            return rows, on, z, z
        return rows

    def _soften(self):
        """Fit all the bonded matter into BUDGET. It is solved together, at
        one substep, so it shares one budget: BUDGET over the pieces there
        are now. Where a piece's fastest mode needs a shorter step than
        that, its stiffest bonds are softened, just enough (`_refit`), so no
        piece is left faster than the target. A bond of matter loses
        strength with the square root of its softening (the breaking speed
        is kept); a stalk or a rope keeps its strength. A thing that
        joins makes the share smaller, and the things already there are
        softened to it too; nothing is ever stiffened back."""
        free = np.flatnonzero(~self.pkin)
        if not len(free):
            return
        target = int(np.clip(BUDGET / len(free), NSUB_MIN, NSUB_MAX))
        self.pnsub[free] = np.minimum(self.pnsub[free], target)
        self._refit(free)

    def _refit(self, ps):
        """Soften bonds until pieces `ps` fit the substeps their thing can
        afford (`pnsub`). `_soften` does it when a thing is built; `reweigh`
        does it again when burning makes a piece lighter, and so faster, so
        a burning crown costs what it cost standing.

        EACH BOND IS SOFTENED ONLY AS FAR AS ITS OWN STIFFNESS NEEDS. A leaf
        cell inside a crown holds a thick twig (Leonardo's rule) and a thin
        petiole. Softening every bond of the piece by one factor, set by the
        twig, softened the petiole a thousandfold too, and a stalk that soft
        meets an impact with a thirtieth of its real force (the force of a
        blow goes as the square root of the stiffness): a dry spray then
        stayed on through a landing that tears it off a real tree. So each
        piece's frequency budget is shared out like water filling a tank:
        a bond that adds less than the level `lvl` to the piece's bound
        keeps its own stiffness, and a stiffer one is cut to the level. The
        level is the highest for which the bonds, capped there, plus the
        contact, stay inside the piece's bound. The contact is six times
        the stiffest bond over the mass (`_kc`, `_rows`), which is at most
        `rho` times the most any capped bond adds, `rho` being the largest
        share that stiffness is of what its bond adds. A bond keeps the
        smaller cut of its two pieces, so no piece is left over its bound,
        and a last pass with one factor per piece catches anything the
        estimate missed."""
        ps = np.asarray(ps, np.int64)
        if not len(ps):
            return
        n = len(self.pm)
        w2 = np.full(n, np.inf)
        w2[ps] = (2.0 * SAFETY / (TICK_S / self.pnsub[ps])) ** 2
        self._contacts(ps)
        _, on, ci, cj = self._rows(per_bond=True)
        if len(on):
            ends = np.concatenate([self.bi[on], self.bj[on]])
            c = np.concatenate([ci, cj])
            kk = np.maximum(self.kn[on], self.ks[on])
            mass = np.where(self.pkin, np.inf, self.pm)
            share = np.concatenate([kk, kk]) / mass[ends] / np.maximum(c, 1e-300)
            rho = np.zeros(n)
            np.maximum.at(rho, ends, share)
            lvl = np.full(n, np.inf)
            order = np.lexsort((c, ends))
            e_s, c_s = ends[order], c[order]
            cut = np.searchsorted(e_s, np.arange(n + 1))
            for p in ps:
                cp = c_s[cut[p]:cut[p + 1]]
                d = len(cp)
                C = 6.0 * rho[p]
                if not d or cp.sum() + C * cp[-1] <= w2[p]:
                    continue
                # the bonds below the level keep theirs (`below`); the
                # level is what is left, over the rest and the contact
                below = np.concatenate([[0.0], np.cumsum(cp)])
                k = np.arange(d)
                th = (w2[p] - below[:-1]) / (d - k + C)
                lo = np.concatenate([[0.0], cp[:-1]])
                ok = (th >= lo) & (th <= cp)
                lvl[p] = th[np.argmax(ok)] if ok.any() else w2[p] / (d + C)
            f = np.minimum(1.0, np.concatenate([lvl[self.bi[on]] / np.maximum(ci, 1e-300),
                                                lvl[self.bj[on]] / np.maximum(cj, 1e-300)]))
            s = np.ones(len(self.bi))
            s[on] = np.minimum(f[:len(on)], f[len(on):])
            self._scale(np.flatnonzero(s < 1.0), s[s < 1.0], ps)
        # the last pass: one factor for every bond of a piece still over
        rows = self._rows().max(axis=1)
        f = np.ones(n)
        f[ps] = np.minimum(1.0, w2[ps] / np.maximum(rows[ps], 1e-30))
        bs = np.flatnonzero(self.alive & (f[self.bi] < 1.0) | self.alive & (f[self.bj] < 1.0))
        if len(bs):
            self._scale(bs, np.minimum(f[self.bi[bs]], f[self.bj[bs]]), ps)

    def _scale(self, bs, s, ps):
        """Soften bonds `bs` by factors `s`. Matter (a block of wood, a
        strip of glass) keeps its breaking speed, a stalk or a rope its
        breaking load (see BUDGET)."""
        for k in ("kn", "ks", "kb", "kt"):
            getattr(self, k)[bs] *= s
        self.soft[bs] *= s
        k = np.where(self.btag[bs] == 0, np.sqrt(s), 1.0)
        self.sig[bs] *= k
        self.tau[bs] *= k
        self._damp(bs)
        self._contacts(ps)

    def retime(self):
        """Choose the substep again (after anything changed a stiffness or
        a mass)."""
        if not len(self.pm) or (~self.pkin).sum() == 0:
            self.h = TICK_S
            return
        self._contacts(np.arange(len(self.pm)))
        top = float(self._rows().max())
        self.h = min(TICK_S, SAFETY * 2.0 / np.sqrt(max(top, 1e-12)))

    # ------------------------------------------------------------ running

    def walls(self, w, drawn):
        """The collider: the lattice's solid cells, less those drawn from
        moving matter (the seam's `drawn`)."""
        return (w.mat != AIR) & ~drawn

    def step(self, dt, wall, xw=None, xbody=None, bmass=None, bvel=None, ximp=None):
        """Carry the bonded matter `dt` seconds against the lattice `wall`.
        Returns the substeps it took (0 when asleep or empty). `xw`,
        `xbody`, `bmass`, `bvel` and `ximp` are the continuum's cells in
        `wall`, the body each is part of, the bodies' masses and velocities,
        and the impulse each cell gives (see `kernel.run`); None when there
        are none."""
        self.broken_last = np.zeros(0, np.int64)
        if not len(self.pm) or self.asleep:
            self.t += dt
            return 0
        nsub = max(1, int(np.ceil(dt / self.h - 1e-9)))
        h = dt / nsub
        alive0 = self.alive.copy()
        if xw is None:
            xw, xbody, ximp = _NO_XW, np.zeros(0, np.int64), np.zeros((0, 3))
            bmass, bvel = np.ones(1), np.zeros((1, 3))
        K.run(nsub, h, self.px, self.pv, self.pq, self.pell, self.pm,
              self.pIbinv, self.pcda, self.pkin, self.pfext, self.bi, self.bj,
              self.ai, self.aj, self.Q0, self.kn, self.ks, self.kb, self.kt,
              self.cn, self.cs, self.cb, self.ct, self.A, self.rb, self.Ib,
              self.Jb, self.sig, self.tau, self.alive, self.tbreak, self.smax,
              self.t, self.adj_ptr, self.adj_b, self.mp, self.off, self.mrad,
              self.kcm, self.ccm, self.mtie, np.ascontiguousarray(wall), self.dx, MU,
              ZETA_C, self.g, AIR_DENS, self.ledger, xw, xbody, bmass, bvel, ximp, self.cross,
              self.pface)
        self.t += dt
        self.substeps += nsub
        self.broken_last = np.flatnonzero(alive0 & ~self.alive)
        self.place()
        return nsub

    def place(self):
        """The members' world positions and velocities, from their pieces."""
        if not len(self.m):
            return
        x = np.zeros((len(self.m), 3))
        K.members_at(self.px, self.pq, self.mp, self.off, x)
        self.x = x
        om = self.omega()
        r = x - self.px[self.mp]
        self.v = self.pv[self.mp] + np.cross(om[self.mp], r)

    def omega(self):
        """Each piece's angular velocity in the world frame."""
        R = _rotations(self.pq)
        lb = np.einsum("nji,nj->ni", R, self.pell) * self.pIbinv
        om = np.einsum("nij,nj->ni", R, lb)
        om[self.pkin] = 0.0
        return om

    # ------------------------------------------------------------ the ledgers

    def _momentum(self, ps=None):
        """(P, L about the origin) of the free pieces `ps` (all of them
        when None)."""
        d = ~self.pkin if ps is None else np.asarray(ps)[~self.pkin[ps]]
        P = (self.pm[d, None] * self.pv[d]).sum(0)
        L = (np.cross(self.px[d], self.pm[d, None] * self.pv[d]) + self.pell[d]).sum(0)
        return P, L

    def residual(self):
        """What the totals moved by that nothing on the books accounts for:
        (|P - in - pushed + carried|, |L - ...|). Zero in exact arithmetic."""
        P, L = self._momentum()
        want = self.inflow + self.ledger - self.carried
        return (float(np.linalg.norm(P - want[:3])),
                float(np.linalg.norm(L - want[3:])))

    def kinetic(self):
        d = ~self.pkin
        R = _rotations(self.pq)
        lb = np.einsum("nji,nj->ni", R[d], self.pell[d])
        return float(0.5 * (self.pm[d] * (self.pv[d] ** 2).sum(1)).sum()
                     + 0.5 * (lb * lb * self.pIbinv[d]).sum())

    # ------------------------------------------------------- what the lattice did

    def reweigh(self, idx, kg):
        """Members `idx` now weigh `kg` (the lattice burned or melted some
        of their cells). Each piece they are in keeps its place and its
        turning: its centre moves to the new centre of mass, its velocity
        to the velocity that point had, and its angular momentum is what
        its new inertia gives at the same spin. What the lost matter
        carried (its momentum, its angular momentum) is booked as leaving
        (`carried`), so the ledger stays exact."""
        idx = np.asarray(idx, np.int64)
        if not len(idx):
            return
        new = self.m.copy()
        new[idx] = kg
        ps = np.unique(self.mp[idx])
        ps = ps[~self.pkin[ps]]
        if not len(ps):
            self.m = new
            return
        P0, L0 = self._momentum(ps)
        self.place()
        om = self.omega()
        R = _rotations(self.pq)
        for p in ps:
            s = np.flatnonzero(self.mp == p)
            w_ = new[s]
            if float(w_.sum()) <= 0.0:
                continue                         # `keep` takes it away
            c, I, Rm = _frame(self.x[s], np.maximum(w_, 1e-12), self.dx)
            Ro = R[p]
            # bonds of p: their attachment points, in the new frame
            mine_i = np.flatnonzero(self.bi == p)
            mine_j = np.flatnonzero(self.bj == p)
            att_i = self.px[p] + self.ai[mine_i] @ Ro.T
            att_j = self.px[p] + self.aj[mine_j] @ Ro.T
            qn = _qmat(Rm)
            # the turn from the old frame to the new, for the bonds' rest turns
            e = _qmul(_qconj(self.pq[p]), qn)
            for b in mine_i:
                self.Q0[b] = _qmul(_qconj(e), self.Q0[b])
            for b in mine_j:
                self.Q0[b] = _qmul(self.Q0[b], e)
            self.pv[p] = self.pv[p] + np.cross(om[p], c - self.px[p])
            self.px[p] = c
            self.pq[p] = qn
            self.pm[p] = max(float(w_.sum()), 1e-9)
            self.pface[p] = min(self.pface[p], float((w_ > 0).sum()))   # a face only shrinks
            self.pI[p] = I
            self.pIbinv[p] = 1.0 / I
            # same spin, new inertia: ell = R I R^T omega
            self.pell[p] = Rm @ (I * (Rm.T @ om[p]))
            self.off[s] = (self.x[s] - c) @ Rm
            self.ai[mine_i] = (att_i - c) @ Rm
            self.aj[mine_j] = (att_j - c) @ Rm
            leaf = self.vmat[s] == _LEAF
            if leaf.any():
                self.pcda[p] = DRAG_CD * LEAF_AREA * 1000.0 * float(w_[leaf].sum())
        self.m = new
        P1, L1 = self._momentum(ps)
        self.carried += np.concatenate([P0 - P1, L0 - L1])
        self._damp(np.flatnonzero(np.isin(self.bi, ps) | np.isin(self.bj, ps)))
        self._refit(ps)
        self.retime()
        self.place()

    def keep(self, mask):
        """Keep only the members `mask`; a piece left with none goes, with
        its bonds. What leaves is booked as carried away."""
        mask = np.asarray(mask, bool)
        if mask.all():
            return
        gone = np.flatnonzero(~mask)
        # members leaving a piece that stays: its mass goes to zero first
        self.reweigh(gone, np.zeros(len(gone)))
        for f in MEMBER_FIELDS + tuple(f for f, _ in self.opt_fields()):
            setattr(self, f, getattr(self, f)[mask])
        alive_p = np.bincount(self.mp, minlength=len(self.pm)) > 0
        alive_p |= self.pkin
        if alive_p.all():
            return
        P0, L0 = self._momentum(np.flatnonzero(~alive_p))
        self.carried += np.concatenate([P0, L0])
        remap = np.cumsum(alive_p) - 1
        for f in PIECE_FIELDS:
            setattr(self, f, getattr(self, f)[alive_p])
        bk = alive_p[self.bi] & alive_p[self.bj]
        for f in BOND_FIELDS:
            setattr(self, f, getattr(self, f)[bk])
        self.bi, self.bj = remap[self.bi], remap[self.bj]
        self.mp = remap[self.mp]
        self._adjacency()
        self.retime()

    def kick(self, ps, dv=(0.0, 0.0, 0.0), omega=None, about=None):
        """Push pieces `ps` from outside: add `dv` (m/s) to their velocity,
        and, with `omega` (rad/s, world frame), set them turning as one
        rigid body about the point `about`: each piece moves at
        omega x (x - about) and spins at omega. The change of momentum and
        angular momentum is an outside push, booked on the ledger, and it
        wakes the bonded matter."""
        ps = np.asarray(ps, np.int64)
        ps = ps[~self.pkin[ps]]
        P0, L0 = self._momentum(ps)
        self.pv[ps] += np.asarray(dv, float)
        if omega is not None:
            om = np.asarray(omega, float)
            self.pv[ps] += np.cross(om, self.px[ps] - np.asarray(about, float))
            R = _rotations(self.pq[ps])
            self.pell[ps] = np.einsum("nij,nj,nkj,k->ni", R, self.pI[ps], R, om)
        P1, L1 = self._momentum(ps)
        self.ledger += np.concatenate([P1 - P0, L1 - L0])
        self.asleep = False
        self._still = 0
        self.place()

    def push(self, members, J, at):
        """Impulses `J` (k, 3), N s, from outside, on the pieces of
        members `members`, each at the world point `at` (k, 3). Each piece
        takes the sum of its pushes and their turning about its centre. On
        the ledger, and on `cross` (the continuum's pushes: see `Matter`).
        Returns what it booked, [P(3), L(3)] about the origin. (A pinned
        piece takes nothing. It is an ANCHOR (`_anchor`), which has no
        members, so no member's push lands on one: the filter below is
        only a guard.)"""
        members = np.asarray(members, np.int64)
        if not len(members):
            return np.zeros(6)
        J = np.asarray(J, np.float64).reshape(-1, 3)
        at = np.asarray(at, np.float64).reshape(-1, 3)
        p = self.mp[members]
        n = len(self.pm)
        dP = np.stack([np.bincount(p, J[:, a], n) for a in range(3)], 1)
        H = np.cross(at, J)
        dL = np.stack([np.bincount(p, H[:, a], n) for a in range(3)], 1)
        ps = np.unique(p)
        ps = ps[~self.pkin[ps]]
        self.pv[ps] += dP[ps] / self.pm[ps, None]
        # the turning about the piece's own centre: what the pushes give
        # about the origin, less what the centre's new motion carries
        self.pell[ps] += dL[ps] - np.cross(self.px[ps], dP[ps])
        out = np.concatenate([dP[ps].sum(0), dL[ps].sum(0)])
        self.ledger += out
        self.cross += out
        self.asleep = False
        self._still = 0
        self.place()
        return out

    def dried(self, members):
        """Leaf members that have lost their last sap: the stalk each one
        hangs on by (a petiole, carrying no other spray) weakens to a dry
        leaf's (PETIOLE_DRY), as it does when a fire dries a crown."""
        ps = np.unique(self.mp[np.asarray(members, np.int64)])
        if not len(ps):
            return
        pet = np.pi * PETIOLE_R ** 2 * 1.000001
        hit = self.alive & (self.btag == 2) & (self.A <= pet) \
            & (np.isin(self.bi, ps) | np.isin(self.bj, ps))
        self.sig[hit] = np.minimum(self.sig[hit], PETIOLE_DRY[2])
        self.tau[hit] = np.minimum(self.tau[hit], PETIOLE_DRY[2])

    def cut(self, ps):
        """Every bond of pieces `ps` parts (matter that cracked, or burned to
        ash: it holds nothing any more)."""
        ps = np.asarray(ps, np.int64)
        hit = self.alive & (np.isin(self.bi, ps) | np.isin(self.bj, ps))
        self.alive[hit] = False
        self.tbreak[hit] = self.t
        return np.flatnonzero(hit)

    # ------------------------------------------------------------ reading

    def fragments(self):
        """Each piece's fragment: the lowest piece its unbroken bonds join it
        to."""
        lab = np.arange(len(self.pm), dtype=np.int64)
        on = np.flatnonzero(self.alive)
        i, j = self.bi[on], self.bj[on]
        # each piece takes the lowest label across its bonds, then the label
        # of its label, until nothing changes (a label never rises)
        while True:
            m = np.minimum(lab[i], lab[j])
            new = lab.copy()
            np.minimum.at(new, i, m)
            np.minimum.at(new, j, m)
            new = new[new]
            if np.array_equal(new, lab):
                return lab
            lab = new

    def kind_of(self, members):
        """The rule's name for the thing members belong to."""
        return [_KINDS[int(k)] for k in self.pkind[self.mp[members]]]
