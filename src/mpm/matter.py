"""THE THING THAT MOVES MATTER — the voxel sim's one door to it.

The voxel sim keeps the world: fields, fire, fluid, sight, and every mind in
it. What it cannot do is move a solid believably — a felled tree SINKS nine
voxels instead of toppling, a vat bounces for ever, a toppled thing lands where
it started (roadmap 91). So a cluster that has to move LEAVES the lattice
through this door, is carried by a particle solver, and is drawn back onto the
lattice every tick for the laws that read it.

This is a SEAM, not a plugin system (roadmap 93, decision 7). It has one side
today, and the surface is exactly what that side needs:

    promote(cells, vel, owner)   this cluster leaves the lattice
    step(dt)                     carry it
    unrasterize() / rasterize()  take it off the lattice before the laws that
                                 would try to move it; put it back before the
                                 laws that need to see it; and take back what
                                 those laws did to it (the write-back)
    collide(cells, vel)          a person, as a moving surface (later)
    wake(eid)                    something is about to push it: if it
                                 lay asleep (REST), carry it again
    aloft() / mass() / energy()  what the ledgers and the minds ask
    events()                     what the particle matter did this tick:
                                 blows, touches, cracks, parted bonds

Units cross here and nowhere else. The lattice speaks grid cells and grams;
the solver speaks metres and kilograms. Every conversion is in this file so
that "mass is a closed ledger" stays a thing you can check on both sides.
"""

import numpy as np

from .core import GRAVITY, MPM, MAT, SDF_MARGIN
from ..bonds.store import Bonds, claims
from ..voxel.materials import ASH as _ASH, LEAF, ROPE, WOOD

# voxel material id -> solver material name. Everything the voxel sim can make
# solid has a row; a material with no row cannot leave the lattice, which is
# an error to raise, not a case to paper over.
_NAMES = {1: "wood", 2: "stone", 3: "iron", 4: "flesh", 5: "ash", 6: "glass",
          7: "leaf", 8: "char", 9: "tin", 10: "lead", 11: "rope"}
AIR = 0

# below this speed, for the ledgers and the minds, a particle is not "in the
# air": it has landed and is lying there. m/s.
STILL = 0.02

# What fire leaves of a solid. A drawn cell the lattice turned into one of
# these is the same matter, burned, and its particles turn with it.
_BURNT = tuple(v for v, n in _NAMES.items() if n in ("char", "ash"))
# The fields of a drawn cell the laws may change, and that the seam hands
# back to the particles (`_written`). The fluid ones count only in a cell
# whose fluid the seam drew (`drawn_fluid`). Not `edge`: only a hand makes
# an edge, and no hand reaches a drawn cell.
_SOLID_FIELDS = ("mat", "smass", "E", "rubble")
_FLUID_FIELDS = ("fl", "fvol", "fpot")

# THE OPTIONAL CARRIED FIELDS (wave 13): the lattice array that holds each on
# the world (None until a lane makes it) and the name the particle and the
# bonded member give it. Every line that touches them is skipped while the
# lattice array is None, so a world that never made one is unchanged.
# (`dmg` on the lattice is a float; on a particle `dmg` is the bool "cracked",
# so the float is `dscal` there.)
_OPT_LATTICE = (("scomp", "scomp"), ("fcomp", "fcomp"), ("dmg", "dscal"),
                ("ice_g", "ice_g"))


def _opt_live(w):
    """The (lattice name, particle name) pairs whose lattice array exists."""
    return [(lf, pf) for lf, pf in _OPT_LATTICE if getattr(w, lf, None) is not None]

# WHAT THE PARTICLE MATTER DID, read off the solve by a tap that never writes
# to it (`Matter.step`, `events`). A BLOW is a change in the speed of a
# piece's centre of mass that neither falling nor lying still explains (its
# JOLT, `_off_fall`). Measured over one tick, in m/s: past HIT it is a blow,
# and the piece reports no other blow until its jolt has fallen under QUIET.
# That gap is the hysteresis: one landing that takes two ticks to stop is
# one blow, and a bounce, after a flight with no jolt at all, is another.
HIT = 0.5
QUIET = 0.1
# Two pieces TOUCH when a particle of each reaches the same grid node, which
# is how the solver lets one push the other: their stencils' lowest nodes
# (`MPM._weights`) are at most TOUCH apart on every axis. They have let go
# again only when they are more than APART away, so two things resting on
# each other do not touch afresh every time one of them shivers.
TOUCH = 2
APART = 3

# SLEEP (design P8). A thing that has come to rest costs the solver as much
# as one that is falling: the pole lying on the floor paid 37 ms a tick, for
# ever, to move a fifth of a millimetre a second. So a piece that has lain
# still (the `aloft` rule, mean speed under STILL) for REST ticks in a row is
# put to sleep: the solver stops carrying it, and it is still drawn on the
# lattice and still handed back what the laws did to it there. A body in
# free flight cannot fool REST: its speed changes by g dt = 0.25 m/s a tick,
# so it is never under STILL two ticks running. It wakes when anything could
# move it: an awake particle comes near (within APART nodes, as two pieces
# that have let go are apart, plus as far as it can travel in the step), the
# lattice changes within WAKE cells of it (a support cell removed), the lattice
# changes what the solver reads about it (its mass, its material, a crack,
# a cell lost), or something writes a speed into it (an impulse).
REST = 3
WAKE = 3

# CROSS-SOLVER WALLS (the user's decision, 2026-10-16). The continuum's
# particles and the bonds' members see each other as walls, ONE WAY PER TICK:
# each solver collides against where the other's matter was at the start
# of the tick, as it collides against the lattice. The cells the members
# are in join the particles' collider for the step (`MPM.wall`), so the
# solver holds a stone up on a pane exactly as it holds it up on a floor;
# the cells the particles were in join the members' collider (`kernel.run`).
# Neither is a lattice wall, and what each pushes is PUSHED BACK on the
# other solver: what the members' cells gave the particles (`_cross_hook`)
# goes onto those members' pieces, at the particles, reversed, and what the
# particles' cells gave the members goes onto the particles that were in
# them. Both are booked on both sides (`Matter.cross`, `Bonds.cross`), so
# the momentum of the two solvers together changes only by what the
# lattice, gravity and the air give them.


# a cell and its six face neighbours, in the order the cross hook reads them
_SIDES = np.array([(0, 0, 0), (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0),
                   (0, 0, -1), (0, 0, 1)], np.int64)


def _off_fall(d, gdt):
    """How far each change of velocity `d` (..., 3) is from any change that
    falling and resting can make in the same time: the nearest point of the
    line from 0 (held still) to `gdt` (falling free, gravity times the
    time). A thing in free flight, one lying on a floor, and one sliding or
    tipping over (partly held) all read about zero; a thing that hit
    something reads the speed it lost."""
    gg = max(float(gdt @ gdt), 1e-300)
    s = np.clip((d @ gdt) / gg, 0.0, 1.0)
    return np.linalg.norm(d - s[..., None] * gdt, axis=-1)


def _near_mask(a, b, r):
    """Which of the grid points `a` (k, 3) lie within `r` of some point of
    `b` on every axis. The points of `b` are marked on a small grid around
    them, grown by `r` along each axis in turn (which grows them into
    cubes), and `a` is looked up in it."""
    lo = b.min(axis=0) - r
    shape = tuple(int(v) for v in b.max(axis=0) - lo + r + 1)
    occ = np.zeros(shape, bool)
    occ[tuple((b - lo).T)] = True
    for ax in range(3):
        grown = occ.copy()
        for k in range(1, r + 1):
            up = [slice(None)] * 3
            down = [slice(None)] * 3
            up[ax], down[ax] = slice(k, None), slice(None, -k)
            grown[tuple(up)] |= occ[tuple(down)]
            grown[tuple(down)] |= occ[tuple(up)]
        occ = grown
    at = a - lo
    ok = ((at >= 0) & (at < np.array(shape))).all(axis=1)
    out = np.zeros(len(a), bool)
    out[ok] = occ[tuple(at[ok].T)]
    return out


class _Tap:
    """One `step` of the particles, watched piece by piece. It reads the
    particles and never writes them, so the solve with the tap on is the
    solve without it, bit for bit.

    A PIECE is the matter of one thing that came in through one `promote`:
    the particles with one (eid, batch). Today no scene names its things, so
    every particle has eid 0, and two lumps the support law sent through the
    seam one after the other would otherwise be one "thing" whose blows
    average out; the batch keeps them apart. A named thing promoted in one
    call is one piece."""

    def __init__(self, m):
        self.m = m
        key = (m.eid.astype(np.int64) << 32) | (m.batch.astype(np.int64) + 1)
        uk, g = np.unique(key, return_inverse=True)
        self.g = g.reshape(-1)
        self.n = len(uk)
        self.keys = [(int(k >> 32), int(k & 0xFFFFFFFF) - 1) for k in uk]
        order = np.argsort(self.g, kind="stable")
        cut = np.searchsorted(self.g[order], np.arange(self.n + 1))
        self.members = [order[cut[i]:cut[i + 1]] for i in range(self.n)]
        self.M = np.maximum(np.bincount(self.g, m.m, self.n), 1e-300)
        self.u = [self._com_v()]          # the centre's velocity, substep by substep
        self.e0 = self._energy()
        self.dmg0, self.cid0 = m.dmg.copy(), m.cid.copy()

    def _com_v(self):
        """Each piece's centre-of-mass velocity, m/s: (pieces, 3)."""
        m = self.m
        if self.n == 1:
            p = (m.m @ m.v)[None]
        else:
            p = np.stack([np.bincount(self.g, m.m * m.v[:, a], self.n)
                          for a in range(3)], 1)
        return p / self.M[:, None]

    def _energy(self):
        """Each piece's kinetic plus potential energy, J (height from 0)."""
        m = self.m
        e = m.m * (0.5 * (m.v * m.v).sum(axis=1) - m.x @ GRAVITY)
        return np.bincount(self.g, e, self.n)

    def watch(self):
        """After every substep (`MPM.step(each=)`)."""
        self.u.append(self._com_v())

    def base(self):
        """The lowest node of each particle's stencil, as `MPM._weights`."""
        m = self.m
        return np.floor(m.x * m.inv_dx - 0.5).astype(np.int64)

    def pairs(self, base, r, among=None):
        """The pairs of pieces (i < j), from `among` (all pairs when None),
        that have a particle each within `r` nodes of the other."""
        if among is None:
            among = [(i, j) for i in range(self.n) for j in range(i + 1, self.n)]
        if not among:
            return set()
        lo = np.array([base[s].min(axis=0) for s in self.members])
        hi = np.array([base[s].max(axis=0) for s in self.members])
        out = set()
        for i, j in among:
            if (lo[i] - r > hi[j]).any() or (lo[j] - r > hi[i]).any():
                continue
            a, b = base[self.members[i]], base[self.members[j]]
            a = a[((a >= lo[j] - r) & (a <= hi[j] + r)).all(axis=1)]
            b = b[((b >= lo[i] - r) & (b <= hi[i] + r)).all(axis=1)]
            if len(a) and len(b) and _near_mask(a, b, r).any():
                out.add((i, j))
        return out


def _cell_eids(flat, n, vmat, kg, eid, best_mat):
    """The thing each of `n` drawn cells belongs to (design 3.0, item 6).

    ONE THING PER CELL, AND IT IS MADE OF THE CELL'S MATERIAL. Of the
    particles in a cell (`flat` says which cell each is in), only those of
    the material the cell was given count, and the eid with the most mass
    among them wins; a tie goes to the lower eid. Counting every particle
    instead could give an iron cell to the wooden thing beside it. The
    masses are summed per (cell, eid) by `np.bincount`, in particle order,
    which is the order a plain loop over the particles would add them."""
    out = np.zeros(n, np.int32)
    ok = vmat == best_mat[flat]
    if not ok.any():
        return out
    c, e = flat[ok].astype(np.int64), eid[ok].astype(np.int64)
    e0 = int(e.min())
    span = int(e.max()) - e0 + 1
    pair, which = np.unique(c * span + (e - e0), return_inverse=True)
    g = np.bincount(which, kg[ok], len(pair))
    pc, pe = pair // span, pair % span + e0
    o = np.lexsort((pe, -g, pc))         # by cell, then heaviest, then lowest eid
    pc, pe = pc[o], pe[o]
    first = np.ones(len(pc), bool)
    first[1:] = pc[1:] != pc[:-1]
    out[pc[first]] = pe[first]
    return out


def _mix(f, v, pot, fls, vols, pots):
    """One cell's fluid (`f`, volume `v`, potency `pot`) with more poured in.
    The volumes add; the cell keeps the fluid with the most volume (a tie to
    the lower id), which is the rule `rasterize` gives a cell; potency mixes
    by volume."""
    by = {int(f): float(v)} if v > 0.0 else {}
    for fi, vi in zip(fls, vols):
        by[int(fi)] = by.get(int(fi), 0.0) + float(vi)
    tot = float(v) + float(np.sum(vols))
    best = max(by, key=lambda k: (by[k], -k))
    mixed = (float(pot) * float(v) + float(np.sum(pots * vols))) / tot
    return best, tot, mixed


class Matter:
    def __init__(self, world, per_cell=1, stiff=1e-4):
        self.w = world
        self.per_cell = int(per_cell)
        self.dx = 0.1 * world.scale                   # metres per voxel
        self.mpm = MPM(world.shape, dx=self.dx, stiff=stiff)
        self.drawn = np.zeros(world.shape, bool)      # cells rasterize wrote
        self.drawn_fluid = np.zeros(world.shape, bool)  # ...and whose fluid is ours
        # NOBODY OWNS PARTICLE MATTER. These two named a person per particle
        # for `aloft_owners`, and no caller ever passed one: bodies stay on
        # the lattice. They are never read; they stay only because
        # tools/golden.py hashes every attribute and counts a REMOVED one as
        # a change in every world. Drop them with the dense distance fields
        # in core.py when the golden base is next recorded.
        self.owner = np.zeros(0, np.int32)
        self.owners = []
        self._wall_key = None                         # the box it was last rebuilt for
        self.promoted = 0                             # promote calls: the next batch
        # the cells the last `rasterize` drew (flat indices into the world)
        # and the thing each one was given (see `_cell_eids`)
        self.drawn_at = np.zeros(0, np.int64)
        self.drawn_eid = np.zeros(0, np.int32)
        # ...what each of those cells held as it was drawn, and which
        # particles were drawn into which of them (`_written` reads these)
        self.drawn_was = {}
        self.drawn_who = np.zeros(0, np.int64)
        self.drawn_in = np.zeros(0, np.int64)
        # WHAT THE PARTICLE MATTER DID (`events`). The tap reads every step
        # and changes nothing; `tap = False` skips it. Two flags say whether
        # the voxel side should turn the blows into sound (`_noise`) and
        # into damage (`_spend`). Both are off: until the voxel side reads
        # them, a falling particle thing is silent and hurts no one, as it
        # always was, and turning them on changes what people hear and
        # suffer, so every harvest made before has to be made again.
        self.tap = True
        self.seam_sounds = False
        self.seam_blows = False
        self._events = []                             # the last step's records
        self._armed = {}                              # piece -> may it report a blow
        self._touching = set()                        # pairs of pieces in contact
        # SLEEP (REST, WAKE). `sleep = False` never puts anything to sleep.
        # The dormant store is the particles' own `nap` field and one record
        # per sleeper: the box of lattice it watches and the walls in that
        # box as it fell asleep. `_still` counts each awake unit's still ticks.
        self.sleep = True
        self._naps = {}                               # record -> (lo, hi, walls)
        self._still = {}                              # unit -> still ticks in a row
        self._nap_next = 0                            # the next record's number
        # THE OTHER SIDE: BONDS (the user's pick, BOTH, 2026-10-10). A
        # promoted cluster that must PART into pieces (a crown of leaves, a
        # rope, a pane: `claims` says which) is carried as rigid pieces
        # joined by bonds that bend and break (src/bonds), and everything
        # else stays on the continuum above. The seam draws, takes back and
        # counts both with the same code. `bonds_on = False` sends
        # everything to the continuum, as before.
        self.bonds = Bonds(world.shape, self.dx)
        self.bonds_on = True
        # A NAMED THING comes here too (the user's decision, 2026-10-16):
        # things.py finds a thing's matter through `points`, the particles
        # and the bonds' members as one set, so a named crown, rope or pane
        # keeps its name and its eid in bonds. False keeps named things on
        # the continuum, as before (`claims`).
        self.bonds_named = True
        self._bond_armed = {}                         # (eid, batch) -> may it report a blow
        # CROSS-SOLVER WALLS (see above). `cross_walls = False` lets the
        # two solvers pass through each other, as they did. `cross` is what
        # the bonds gave the particles, [P(3), L(3)] about the origin: the
        # same impulse as `bonds.cross`, reversed.
        self.cross_walls = True
        self.cross = np.zeros(6)
        self._xw = None                               # cell -> continuum cell, or -1

    # ------------------------------------------------------------ the door in

    def promote(self, cells, vel=(0.0, 0.0, 0.0)):
        """A cluster of lattice cells becomes particles and leaves the grid.

        Lifts everything the cell carried — material, mass, heat, fluid, edge
        and, when the world has one, the thing it belongs to (`eid`) —
        exactly as `_topple` and `_launch` did, and clears the cells. There
        is no pivot, axis or angle to hand over: gravity and the contact
        surface decide how it falls, which is the point. Every particle is
        stamped with this call's serial (`mpm.batch`).
        """
        w = self.w
        cells = np.asarray(cells, np.int64)
        if not len(cells):
            return 0
        idx = tuple(cells.T)
        mats = w.mat[idx].astype(np.int16)
        bad = set(int(m) for m in np.unique(mats)) - set(_NAMES) - {AIR}
        if bad:
            raise ValueError(f"no particle material for voxel id(s) {sorted(bad)}")
        kind = claims(w, cells, self.bonds_named) if self.bonds_on else None
        if kind is None and self.bonds_on and len(self.bonds.pm):
            # a cell a bonded thing is tied to joins that thing (a named
            # thing only when named things may come, as for `claims`)
            weid = getattr(w, "eid", None)
            if self.bonds_named or weid is None or not weid[idx].any():
                kind = self.bonds.joins(w.shape, cells[mats != AIR])
        if kind is not None:
            return self._promote_bonds(cells, vel, kind)
        keep = mats != AIR
        cells, mats = cells[keep], mats[keep]
        idx = tuple(cells.T)
        n_per = self.per_cell ** 3
        step = self.dx / self.per_cell
        # sub-positions inside each cell, jittered as add_box jitters, so a
        # promoted block does not start as a perfect standing-wave lattice
        offs = np.stack(np.meshgrid(*[np.arange(step * 0.5, self.dx, step)] * 3,
                                    indexing="ij"), -1).reshape(-1, 3)
        rng = np.random.default_rng(self.promoted)
        pts = (cells[:, None, :] * self.dx + offs[None]).reshape(-1, 3)
        # A VOXEL IS FLAT-FACED, AND PROMOTED MATTER KEEPS THAT SHAPE. `add_box`
        # jitters a fresh lattice by a quarter of a spacing so it does not sit
        # ringing on its own symmetry; the same quarter here turned the bottom
        # of a felled pole into a saw-blade of 2.5 cm teeth, and measured, it
        # rested on the floor through ONE particle at a time — one hit per
        # substep out of twenty-nine along the face — and friction on a
        # sixty-second of a body stops nothing. Five percent breaks the
        # symmetry and keeps the face.
        pts += rng.uniform(-0.05 * step, 0.05 * step, pts.shape)
        rep = np.repeat(np.arange(len(cells)), n_per)
        # the thing each cell belongs to, read off the lattice when it says
        weid = getattr(w, "eid", None)
        eids = np.zeros(len(cells), np.int32) if weid is None \
            else weid[idx].astype(np.int32)
        # where each thing's own cells start: its flaws are read from here
        corner = {int(e): cells[eids == e].min(axis=0) for e in np.unique(eids)}
        start = len(self.mpm.x)
        for m in np.unique(mats):
            sel = rep[mats[rep] == m]
            name = _NAMES[int(m)]
            dens, E, nu, yld, damp, strength, rigid = MAT[name]
            k = self._kind(name)
            # THIS CELL'S SHARE, IN FLOAT64 FROM THE START. The lattice keeps
            # grams in float32, and dividing there turned a cell's 600 g into
            # 0.60000002 kg: every crossing of the seam moved the mass by up
            # to 4e-8 of itself.
            kg = w.smass[idx][sel].astype(np.float64) / n_per / 1000.0
            # A HALF-FULL CELL IS HALF AS MUCH WOOD, NOT A LIGHTER WOOD. The
            # volume at rest is the volume of the mass it carries. It was the
            # whole cell's, which made a branch spoke at 40% fill a material
            # of 240 kg/m^3: its sound ran 1.58 times too fast, a 40% canopy's
            # viscous step was 2.5 times too short, and the felled tree paid
            # 215 substeps a tick where its materials need 108. The same
            # density makes a spoke part at the jolt real wood parts at (its
            # parting stress rho * c * dv rises 1.58 times). A cell with no
            # mass keeps the cell's volume, as before: it is no matter at all.
            vol = np.where(kg > 0.0, kg / dens, (self.dx ** 3) / n_per)
            c0 = 0.0 if strength is None else strength * np.sqrt(self.mpm.stiff)
            crit = np.full(len(sel), c0)
            if c0 > 0.0:
                # A THING'S FLAWS ARE ITS OWN. They were read off one field at
                # the cells' WORLD position, so the same stick toppled from a
                # shelf and from the floor was made of different wood — the
                # bug `add_box` already fixed for particles placed by hand
                # (test_WHERE_A_THING_IS_DOES_NOT_CHANGE_WHAT_IT_IS_MADE_OF).
                # Each thing reads a field seeded by its own eid, from its own
                # corner, so its strengths go wherever it goes.
                for e in np.unique(eids[sel]):
                    mine = eids[sel] == e
                    fl = self.mpm._flaw_field(int(e))
                    ij = cells[sel][mine] - corner[int(e)]
                    crit[mine] = c0 * (0.45 + 1.55 * fl[ij[:, 0], ij[:, 1], ij[:, 2]])
            s0 = len(self.mpm.x)
            self.mpm._append(pts[mats[rep] == m], kg, vol, E * self.mpm.stiff,
                             nu, 0.0 if yld is None else yld, damp, dens, crit, k,
                             rigid)
            s1 = len(self.mpm.x)
            self.mpm.E[s0:s1] = w.E[idx][sel].astype(np.float64) / n_per
            self.mpm.fl[s0:s1] = w.fl[idx][sel]
            self.mpm.fvol[s0:s1] = w.fvol[idx][sel].astype(np.float64) / n_per
            self.mpm.fpot[s0:s1] = w.fpot[idx][sel]
            self.mpm.edge[s0:s1] = w.edge[idx][sel]
            self.mpm.vmat[s0:s1] = m
            self.mpm.eid[s0:s1] = eids[sel]
            for lf, pf in _opt_live(w):
                self.mpm.ensure_opt(pf)[s0:s1] = getattr(w, lf)[idx][sel]
        n = len(self.mpm.x) - start
        self.mpm.batch[start:] = self.promoted
        # AND A VESSEL CARRIES WHAT IS IN IT. The liquid in a vat sits in AIR
        # cells inside a glass shell; only the shell is solid, so only the
        # shell was ever promoted — and the acid was left hanging in mid-air
        # the instant the vat let go, to pour down on its own while the glass
        # fell separately beside it. Measured on a dropped vial: 0 ml aboard,
        # all of it on the floor before the glass arrived. What it carries is
        # the fluid in its cavity (`_cavity`). The liquid rides on the
        # shell's particles, which is not where it would be in life, but it
        # is the right mass in the right place to within the vessel itself.
        if n:
            carry = self._cavity(cells, need_fluid=True)
            if carry.any():
                vol = float(w.fvol[carry].sum())
                fid = int(np.bincount(w.fl[carry].astype(np.int64),
                                      w.fvol[carry]).argmax())
                pot = float((w.fpot[carry] * w.fvol[carry]).sum() / max(vol, 1e-12))
                self.mpm.fvol[start:] += vol / n
                self.mpm.fl[start:] = fid
                self.mpm.fpot[start:] = pot
                for arr in (w.fl, w.fvol, w.fpot):
                    arr[carry] = 0
        # AND IT LEAVES AS ONE THING. Everything in this call came off one
        # connected lump of lattice, so the grouping is already known and
        # nothing has to rediscover it: a trunk keeps its canopy, a vat keeps
        # its walls, and neither is a cloud of independent points that
        # happens to be flying in formation.
        self.mpm.bind(start)
        # CELLS PER SECOND in, metres per second out: `dx` is metres a cell.
        # The note here read "cells/tick", and no caller had ever passed
        # anything but zero, so nothing had tested the claim.
        self.mpm.v[start:] = np.asarray(vel, float) * self.dx
        # and the lattice lets go
        for arr in (w.mat, w.smass, w.E, w.fl, w.fvol, w.fpot, w.edge, w.fallh,
                    w.rubble):
            arr[idx] = 0
        if weid is not None:
            weid[idx] = 0
        for lf, _ in _opt_live(w):
            getattr(w, lf)[idx] = 0
        self.promoted += 1
        return n

    def _promote_bonds(self, cells, vel, kind):
        """`promote`, for a cluster `claims` gave to bonds: everything the
        cells carried goes onto the members (one per cell), the lattice lets
        go exactly as for particles, and this call's serial is the batch.
        A thing in bonds carries no cavity: no crown, rope or pane holds a
        liquid."""
        w = self.w
        mats = w.mat[tuple(cells.T)]
        cells = cells[mats != AIR]
        idx = tuple(cells.T)
        weid = getattr(w, "eid", None)
        eids = np.zeros(len(cells), np.int32) if weid is None \
            else weid[idx].astype(np.int32)
        # what it is tied to on the lattice (`Bonds._anchor`): a rope's
        # cells to any solid cell they touch, a leaf to the wood it touches
        anchors = []
        mats = w.mat[idx]
        rope = mats == ROPE
        tie = rope | (mats == LEAF)
        if tie.any():
            mine = set(map(tuple, cells.tolist()))
            shape = np.array(w.shape)
            for r in np.flatnonzero(tie):
                c = cells[r]
                for a in range(3):
                    for s in (-1, 1):
                        nb = c.copy()
                        nb[a] += s
                        if (nb < 0).any() or (nb >= shape).any() or tuple(nb) in mine:
                            continue
                        t = tuple(nb)
                        if w.mat[t] == AIR or self.drawn[t] or w.moving[t]:
                            continue
                        if rope[r]:
                            anchors.append((r, nb, 1))
                        elif w.mat[t] == WOOD:
                            anchors.append((r, nb, 2))
        n = self.bonds.add(cells, w.mat[idx].astype(np.int64),
                           w.smass[idx].astype(np.float64),
                           w.E[idx].astype(np.float64), w.fl[idx],
                           w.fvol[idx].astype(np.float64),
                           w.fpot[idx].astype(np.float64),
                           w.edge[idx].astype(np.float64), eids,
                           w.rubble[idx].copy(),
                           np.asarray(vel, float) * self.dx, self.promoted, kind,
                           w.vox_l, anchors)
        live = _opt_live(w)
        if live:
            self.bonds.ensure_opt_all()
            tail = slice(len(self.bonds.x) - n, None)
            for lf, pf in live:
                getattr(self.bonds, pf)[tail] = getattr(w, lf)[idx]
        for arr in (w.mat, w.smass, w.E, w.fl, w.fvol, w.fpot, w.edge, w.fallh,
                    w.rubble):
            arr[idx] = 0
        if weid is not None:
            weid[idx] = 0
        for lf, _ in _opt_live(w):
            getattr(w, lf)[idx] = 0
        self.promoted += 1
        return n

    def _cavity(self, cells, need_fluid=True):
        """The cells a vessel made of `cells` holds, as a mask the size of
        the world.

        A cell is in the vessel when the vessel does not own it, the vessel
        walls it in on all four sides at its own height, and it stands on
        the vessel or on another cell the vessel holds. That is "in the
        cup". It does not need the cup to be closed, which matters because
        an open vat is exactly the case. A tube open at both ends walls
        nothing in, so it holds nothing (tests/test_vessels.py pins that
        gap on purpose).

        With `need_fluid` (what `promote` asks), only cells that hold fluid
        count: what the vessel carries now. Without it, every cell of the
        hollow counts, whatever is in it: what the vessel could hold. A
        caller that wants only the empty part masks it with `mat == AIR`."""
        w = self.w
        out = np.zeros(w.shape, bool)
        cells = np.asarray(cells, np.int64).reshape(-1, 3)
        if not len(cells):
            return out
        lo, hi = cells.min(axis=0), cells.max(axis=0) + 1
        box = tuple(slice(int(lo[a]), int(hi[a])) for a in range(3))
        ins = np.zeros(tuple(int(v) for v in hi - lo), bool)
        ins[tuple((cells - lo).T)] = True
        ax = np.cumsum(ins, axis=0) > 0
        bx = np.cumsum(ins[::-1], axis=0)[::-1] > 0
        ay = np.cumsum(ins, axis=1) > 0
        by = np.cumsum(ins[:, ::-1], axis=1)[:, ::-1] > 0
        walled = ax & bx & ay & by & ~ins
        if need_fluid:
            walled &= w.fvol[box] > 0.0
        held = np.zeros_like(ins)
        for z in range(1, ins.shape[2]):
            held[:, :, z] = walled[:, :, z] & (ins[:, :, z - 1] | held[:, :, z - 1])
        out[box] = held
        return out

    def _kind(self, name):
        """The solver's index for the material `name`, added on first use."""
        if name not in self.mpm.kinds:
            self.mpm.kinds.append(name)
        return self.mpm.kinds.index(name)

    def _become(self, idx, vid):
        """The particles `idx` are voxel material `vid` now, in the solver
        too: its row in MAT, at the softened modulus `promote` uses. Char and
        ash have no tensile strength, so there are no flaws to read."""
        name = _NAMES[int(vid)]
        dens, E, nu, yld, damp, strength, rigid = MAT[name]
        c0 = 0.0 if strength is None else strength * np.sqrt(self.mpm.stiff)
        self.mpm.become(idx, E * self.mpm.stiff, nu, 0.0 if yld is None else yld,
                        damp, dens, c0, self._kind(name), rigid)
        self.mpm.vmat[idx] = vid

    # ------------------------------------------------------------ carrying it

    def _walls(self, dt):
        """The static lattice is the collider. Rebuilt when it changes WHERE
        THE PARTICLES CAN REACH, and only then.

        This was keyed on a count of the whole world's solid cells, and a
        count is the wrong question both ways. It is blind to a cell that
        MOVED (one gone here, one come there: the same count and a stale
        collider), and it rebuilt for a fire burning out across the house
        that no particle could touch: 1.7 s a time on the falling tree's
        grid. So the lattice is compared with the collider inside the box the
        particles can reach in this step: their cells, the stencil and the
        distance field's margin around them, and as far as the fastest of
        them can travel in `dt`. A change in there rebuilds; a change
        anywhere else waits until something comes near it."""
        w, m = self.w, self.mpm
        reach = 2 + SDF_MARGIN + int(np.ceil(np.abs(m.v).max() * dt * m.inv_dx)) + 2
        c = np.floor(m.x * m.inv_dx).astype(np.int64)
        lo = np.maximum(c.min(axis=0) - reach, 0)
        hi = np.minimum(c.max(axis=0) + 1 + reach, w.shape)
        box = tuple(slice(int(lo[a]), int(hi[a])) for a in range(3))
        if not np.array_equal((w.mat[box] != AIR) & ~self.drawn[box], m.wall[box]):
            m.wall[:] = (w.mat != AIR) & ~self.drawn
            self._wall_key = (tuple(int(v) for v in lo), tuple(int(v) for v in hi))

    def step(self, dt):
        """Carry the particles, then the bonded matter, through `dt`
        seconds. The particles' substeps are returned, as before. When both
        hold matter that can meet in this step, each is a wall to the other
        (CROSS-SOLVER WALLS, above)."""
        x = self._cross_start(dt)
        sub = self._step_mpm(dt, x)
        if x is not None and x["J"]:
            self._cross_push_bonds(x)
        if len(self.bonds.pm):
            self._step_bonds(dt, x)
        return sub

    # ------------------------------------------------- cross-solver walls

    def _cross_start(self, dt):
        """Where each solver's matter is at the start of the tick, when
        the two can meet in it; None when they cannot (one of them is
        empty, or they are further apart than either can go in `dt`), and
        then nothing here costs anything."""
        m, b = self.mpm, self.bonds
        if not self.cross_walls or not len(m.x) or not len(b.m):
            return None
        g = 0.5 * float(np.abs(GRAVITY).max()) * dt * dt
        far = lambda v: int(np.ceil((float(np.abs(v).max()) * dt + g) * m.inv_dx)) + 2
        pc = np.floor(m.x * m.inv_dx).astype(np.int64)
        mc = np.floor(b.x * b.inv_dx).astype(np.int64)
        r = far(m.v) + far(b.v)
        if ((pc.min(axis=0) - r > mc.max(axis=0)).any()
                or (mc.min(axis=0) - r > pc.max(axis=0)).any()):
            return None
        shape = np.array(self.w.shape)
        pc = np.clip(pc, 0, shape - 1)
        mc = np.clip(mc, 0, shape - 1)
        # the members' cells, as a box: which member each is (the first)
        lo = mc.min(axis=0) - 1
        dims = tuple(int(v) for v in mc.max(axis=0) - lo + 2)
        who = np.full(dims, -1, np.int64)
        rel = mc - lo
        order = np.arange(len(mc))[::-1]                # the first one wins
        who[tuple(rel[order].T)] = order
        return {"lo": lo, "who": who, "pc": pc, "mc": mc,
                "J": [], "k": [], "at": []}

    def _cross_hook(self, x, idx):
        """What the members' cells push the particles with, read after
        every substep (the solver itself pushes, as at any wall). Particles
        `idx` (of the solver's arrays as they are in this step) can reach a
        member's cell; each bound cluster with one of them in it is watched
        whole, and the loose ones together, as one GROUP. Nothing else
        changes a group's momentum but gravity and the walls (the solver
        conserves it, `_hold_shape` too), so what it gained in a substep,
        less gravity, is what the walls gave it. Along each axis that push
        came from the walls on one side of it, and it is shared between the
        members' cells and the lattice's on that side by the mass of the
        group's particles beside each (a particle inside a member's cell is
        beside it on every side). The members' share is noted for the
        member each of those particles is beside, at the particle
        (`_cross_push_bonds`).

        AN AXIS WITH NO WALL ON THE SIDE THE PUSH CAME FROM is rubbing
        along a face: friction, and a wall's friction is as big as the push
        it gives square to that face (the grid's friction is sized by the
        normal impulse each node removed). So that push is shared between
        the members and the lattice by what each pushed along the OTHER
        axes in the same substep, not by contact mass. By contact mass, a
        block on the floor with a plank landing on it handed half of the
        floor's grip on the block's shivering to the plank, and the plank
        slid 1 to 4 cells sideways off a block it had landed on square (a
        lattice block, the same drop, lands straight).

        That share is the cross push by the books, not by a measurement at
        each wall: a group pressed on a pane and a floor at once has the
        push from below split by contact mass. The books stay exact either
        way, since the same impulse is booked, reversed, on the members."""
        m = self.mpm
        lo, who = x["lo"], x["who"]
        dims = np.array(who.shape)
        cid = m.cid[idx]
        clus = np.unique(cid[cid >= 0])
        I = np.flatnonzero(np.isin(m.cid, clus)) if len(clus) else np.zeros(0, np.int64)
        I = np.union1d(I, idx[cid < 0])
        gk = np.where(m.cid[I] >= 0, m.cid[I], -1)
        _, gid = np.unique(gk, return_inverse=True)
        gid = gid.reshape(-1)
        ng = int(gid.max()) + 1
        kg = m.m[I]
        M = np.bincount(gid, kg, ng)
        state = {"v": m.v[I].copy()}
        shape = np.array(m.wall.shape)

        def member(c):
            ok = ((c >= 0) & (c < dims)).all(axis=1)
            out = np.full(len(c), -1, np.int64)
            out[ok] = who[tuple(c[ok].T)]
            return out

        def hook(h):
            v1 = m.v[I]
            dv = v1 - state["v"]
            state["v"] = v1.copy()
            dP = np.stack([np.bincount(gid, kg * dv[:, a], ng) for a in range(3)], 1) \
                - M[:, None] * GRAVITY[None] * h
            pos = m.x[I]
            cell = np.floor(pos * m.inv_dx).astype(np.int64)
            # beside what, on each side: the member there, and the lattice
            # (its own cell, then -x, +x, -y, +y, -z, +z, all in one go)
            nb = (cell[None] + _SIDES[:, None]).reshape(-1, 3)
            k7 = member(nb - lo).reshape(7, -1)
            inside = k7[0]
            k6 = np.where(inside >= 0, inside, k7[1:])
            ok = ((nb[len(cell):] >= 0) & (nb[len(cell):] < shape)).all(axis=1)
            lat = np.ones(len(ok), bool)                    # past the edge: the world's
            lat[ok] = m.wall[tuple(nb[len(cell):][ok].T)]
            l6 = lat.reshape(6, -1) & (k6 < 0)
            ks = {(a, s): k6[2 * a + (s > 0)] for a in range(3) for s in (-1, 1)}
            lats = {(a, s): l6[2 * a + (s > 0)] for a in range(3) for s in (-1, 1)}
            hit = k6 >= 0
            first = hit.argmax(axis=0)                      # the first side, in that order
            anyk = np.where(hit.any(axis=0), k6[first, np.arange(len(I))], -1)
            anyl = l6.any(axis=0)
            mxa = np.bincount(gid, np.where(anyk >= 0, kg, 0.0), ng)
            if not mxa.any():
                return
            # the push along a came from the walls on the side it points
            # away from (each group's own side, by the sign of its push)
            fric = np.zeros((ng, 3), bool)      # nothing beside it there
            norm = np.zeros((ng, 3, 2))         # pushed square: members, lattice
            plan = []
            for a in range(3):
                for s in (-1, 1):
                    want = (s * dP[:, a]) < 0.0
                    if not want.any():
                        continue
                    k, lat = ks[a, s], lats[a, s]
                    mx = np.bincount(gid, np.where(k >= 0, kg, 0.0), ng)
                    ml = np.bincount(gid, np.where(lat, kg, 0.0), ng)
                    side = (mx + ml) > 0.0
                    fric[:, a] |= want & ~side
                    on = want & side
                    share = np.where(on & (mx > 0.0), mx / np.maximum(mx + ml, 1e-300), 0.0)
                    norm[on, a, 0] = share[on] * np.abs(dP[on, a])
                    norm[on, a, 1] = (1.0 - share[on]) * np.abs(dP[on, a])
                    plan.append((a, k, mx, share))
            if fric.any():
                # rubbing along a face: shared by what each pushed square
                # to it, along the other two axes, in this substep (with
                # no such push, by contact mass, all it touches)
                mla = np.bincount(gid, np.where(anyl, kg, 0.0), ng)
                bym = np.where(mxa > 0.0, mxa / np.maximum(mxa + mla, 1e-300), 0.0)
                for a in range(3):
                    f = fric[:, a]
                    if not f.any():
                        continue
                    nm = norm[:, :, 0].sum(1) - norm[:, a, 0]
                    nl = norm[:, :, 1].sum(1) - norm[:, a, 1]
                    share = np.where(nm + nl > 0.0, nm / np.maximum(nm + nl, 1e-300), bym)
                    share = np.where(f, share, 0.0)
                    plan.append((a, anyk, mxa, share))
            for a, k, mx, share in plan:
                if not share.any():
                    continue
                r = np.flatnonzero((k >= 0) & (share[gid] > 0.0))
                j = np.zeros((len(r), 3))
                j[:, a] = (share * dP[:, a])[gid[r]] * kg[r] / mx[gid[r]]
                x["J"].append(j)
                x["k"].append(k[r])
                x["at"].append(pos[r].copy())
        return hook

    def _cross_push_bonds(self, x):
        """What the particles lost at the members' cells this step goes
        onto those members' pieces, at the particles, reversed; booked on
        both sides."""
        J = np.concatenate(x["J"])
        k = np.concatenate(x["k"])
        at = np.concatenate(x["at"])
        out = self.bonds.push(k, -J, at)
        self.cross -= out

    def _cross_walls(self, x, wall):
        """The lattice `wall` and the cells the particles were in at the
        start of the tick, for the bonds to collide with: (walls, xw, the
        body each cell is part of, the bodies' masses and velocities, ximp,
        the cells) for `Bonds.step` (see `kernel.run`). A cell's body is the
        bound cluster with the most mass in it, whole (a push on a cluster
        moves all of it: `_cross_push_mpm`), or else the loose particles in
        it; its velocity is its particles' by mass, as their own step this
        tick left them (a stone the members' cells just stopped is still
        now, not falling)."""
        m = self.mpm
        flat = np.ravel_multi_index(tuple(x["pc"].T), self.w.shape)
        cells, of = np.unique(flat, return_inverse=True)
        of = of.reshape(-1)
        x["of"] = of
        n = len(cells)
        if self._xw is None or self._xw.shape != self.w.shape:
            self._xw = np.full(self.w.shape, -1, np.int64)
        self._xw.ravel()[cells] = np.arange(n)
        both = wall.copy()
        both.ravel()[cells] = True
        # each cell's heaviest cluster, or -1 (its loose particles)
        owner = np.full(n, -1, np.int64)
        bound = m.cid >= 0
        if bound.any():
            pair, inv = np.unique(np.stack([of[bound], m.cid[bound]], 1),
                                  axis=0, return_inverse=True)
            g = np.bincount(inv.reshape(-1), m.m[bound], len(pair))
            o = np.lexsort((pair[:, 1], -g, pair[:, 0]))
            first = np.ones(len(o), bool)
            first[1:] = pair[o[1:], 0] != pair[o[:-1], 0]
            owner[pair[o[first], 0]] = pair[o[first], 1]
        key = np.where(owner >= 0, owner, -1 - np.arange(n))
        uk, xbody = np.unique(key, return_inverse=True)
        xbody = xbody.reshape(-1)
        nb = len(uk)
        # each particle's body: its cluster, when that cluster is one; a
        # loose particle, its cell's loose particles, when they are one
        pk = np.where(bound, m.cid, -1 - of)
        ok = np.isin(pk, uk)
        pb = np.full(len(pk), -1, np.int64)
        pb[ok] = np.searchsorted(uk, pk[ok])
        bmass = np.bincount(pb[ok], m.m[ok], nb)
        bvel = np.stack([np.bincount(pb[ok], m.m[ok] * m.v[ok, a], nb) for a in range(3)], 1) \
            / np.maximum(bmass, 1e-300)[:, None]
        # A BODY THE LATTICE HOLDS does not give: a block on the floor under
        # a plank is a wall to it, as the floor is. (Given, it sank under the
        # plank's weight within the step, and the plank went into it.)
        c = np.floor(m.x * m.inv_dx).astype(np.int64)
        shp = np.array(self.w.shape)
        held = np.zeros(len(c), bool)
        for a in range(3):
            for sg in (-1, 1):
                nbc = c.copy()
                nbc[:, a] += sg
                inn = ((nbc >= 0) & (nbc < shp)).all(axis=1)
                hit = ~inn
                hit[inn] = m.wall[tuple(nbc[inn].T)]
                held |= hit
        firm = np.bincount(pb[ok & held], minlength=nb) > 0
        bmass = np.where(firm, 1e30, np.maximum(bmass, 1e-300))
        bvel[firm] = 0.0
        return both, self._xw, xbody, bmass, bvel, np.zeros((n, 3)), cells

    def _cross_push_mpm(self, x, ximp, cells):
        """What the members got from the particles' cells this step goes
        back onto the particles that were in them, reversed: each cell's
        share by mass, and a share that falls on a bound cluster spread over
        the whole cluster (a rigid push, not a jolt to one particle, which
        the solver's shape holding would read as a blow). Booked here; the
        bonds booked theirs in the kernel."""
        self._xw.ravel()[cells] = -1
        if not ximp.any():
            return
        m = self.mpm
        of = x["of"].reshape(-1)
        M = np.bincount(of, m.m, len(cells))
        J = -ximp[of] * (m.m / np.maximum(M[of], 1e-300))[:, None]   # each particle's
        dv = J / np.maximum(m.m, 1e-300)[:, None]
        bound = m.cid >= 0
        if bound.any():
            n = int(m.cid.max()) + 1
            c = m.cid[bound]
            Jc = np.stack([np.bincount(c, J[bound, a], n) for a in range(3)], 1)
            Mc = np.bincount(c, m.m[bound], n)
            dv[bound] = Jc[c] / np.maximum(Mc[c], 1e-300)[:, None]
        moved = (dv != 0.0).any(axis=1)
        m.v[moved] += dv[moved]
        P = (m.m[moved, None] * dv[moved]).sum(0)
        L = np.cross(m.x[moved], m.m[moved, None] * dv[moved]).sum(0)
        self.cross += np.concatenate([P, L])

    def _step_mpm(self, dt, x=None):
        """Carry the particles through `dt` seconds. With `tap` on, the
        step is watched as it goes and `events` says what it did. Matter
        asleep (REST) is not carried, and costs nothing; with all of it
        asleep the solver is not called at all."""
        self._events = []
        m = self.mpm
        if not len(m.x):
            self._armed, self._touching = {}, set()
            self._naps, self._still = {}, {}
            return 0
        if self._naps and not self.sleep:
            self.wake()                               # sleep turned off: all up
        if self._naps:
            self._wake(dt)
        awake = m.nap < 0
        if not awake.any():                           # all of it asleep
            m.t += dt
            return 0
        # the sleepers leave the solve for this step (`MPM.set_aside`), and
        # what the tap remembers about them waits for them
        held = None if awake.all() else m.set_aside(awake)
        if held is not None:
            sk = (held[1]["eid"].astype(np.int64) << 32) | (held[1]["batch"] + 1)
            dozing = {(int(k >> 32), int(k & 0xFFFFFFFF) - 1) for k in np.unique(sk)}
            armed0, touching0 = self._armed, self._touching
        # the fused substep runs only when the world's own switch allows it:
        # with `w.fused = False` the solver goes stage by stage, as the
        # world's other laws then run their plain numpy paths
        fused = m.fused
        m.fused = fused and bool(getattr(self.w, "fused", True))
        hook = None
        if x is not None:
            # the awake particles that can reach a member's cell this step
            c = np.floor(m.x * m.inv_dx).astype(np.int64) - x["lo"]
            g = 0.5 * float(np.abs(GRAVITY).max()) * dt * dt
            r = int(np.ceil((float(np.abs(m.v).max()) * dt + g) * m.inv_dx)) + 2
            near = ((c >= -r) & (c < np.array(x["who"].shape) + r)).all(axis=1)
            if near.any():
                hook = self._cross_hook(x, np.flatnonzero(near))
        added = None
        try:
            self._walls(dt)
            if x is not None:
                # the members' cells, walls for this step only
                at = np.ravel_multi_index(tuple(x["mc"].T), self.w.shape)
                added = at[~m.wall.ravel()[at]]
                m.wall.ravel()[added] = True
            if not self.tap:
                sub = m.step(dt, after=hook)
            else:
                tap = _Tap(m)
                self._first_touches(tap)
                sub = m.step(dt, each=tap.watch, after=hook)
                self._events = self._read(tap, dt)
                if held is not None:
                    for k, a in armed0.items():
                        if k in dozing:
                            self._armed.setdefault(k, a)
                    self._touching |= {p for p in touching0
                                       if p[0] in dozing and p[1] in dozing}
        finally:
            m.fused = fused
            if added is not None:
                m.wall.ravel()[added] = False
            if held is not None:
                m.take_back(held)
        if self.sleep:
            self._doze(dt)
        return sub

    # ------------------------------------------------------------ bonds

    def _step_bonds(self, dt, x=None):
        """Carry the bonded matter `dt` seconds against the lattice as it is
        now (everything is off the lattice while the solvers run, so the
        collider is the lattice's solid cells). It reports what it did as
        the particles do: `struck` for a blow to a thing (edge-triggered by
        HIT and QUIET), and `crack` and `part` for bonds that broke, with the
        cells of the pieces they held. It does not report `touch`.

        BONDS SLEEP AS ONE. When every FRAGMENT of the bonded matter (the
        pieces its unbroken bonds hold together) has had a mean member speed
        under STILL for REST ticks, the solver stops carrying it until
        the lattice it lies in changes, a write-back changes what it weighs
        or is made of, something writes a speed or a load (`pfext`) into
        it, a knot's cell goes, new matter joins it, or awake particles come
        near. While a load is on it, it does not sleep. The little motion
        it still had is stopped by what it lies on: an outside push, booked
        on the ledger like any other, so the books stay exact."""
        b, w = self.bonds, self.w
        wall = b.walls(w, self.drawn)
        cut = b.untie(wall)
        if b.asleep:
            lo, hi, was = b._nap_walls
            box = tuple(slice(int(lo[a]), int(hi[a])) for a in range(3))
            near = False
            m = self.mpm
            if len(m.x) and (m.nap < 0).any():
                c = np.floor(m.x[m.nap < 0] * m.inv_dx).astype(np.int64)
                r = self._reach(np.flatnonzero(m.nap < 0), dt)
                near = bool(((c.max(axis=0) + r >= lo) & (c.min(axis=0) - r < hi)).all())
            if (len(cut) or near or not np.array_equal(wall[box], was)
                    or (b.pv != 0.0).any() or (b.pell != 0.0).any()
                    or (b.pfext != 0.0).any()):
                b.asleep = False
                b._still = 0
        if b.asleep:
            b.t += dt
            return 0
        keys0, u0, e0 = self._bond_groups()
        if x is None:
            sub = b.step(dt, wall)
        else:
            both, xw, xbody, bmass, bvel, ximp, cells = self._cross_walls(x, wall)
            sub = b.step(dt, both, xw, xbody, bmass, bvel, ximp)
            self._cross_push_mpm(x, ximp, cells)
        broken = np.concatenate([cut, b.broken_last])
        if self.tap:
            self._events.extend(self._bond_events(keys0, u0, e0, broken, dt))
        if self.sleep:
            speed = self._bond_speed()
            pulled = bool((b.pfext != 0.0).any())     # a load keeps it awake
            b._still = b._still + 1 if speed <= STILL and not pulled else 0
            if b._still >= REST:
                b.asleep = True
                P, L = b._momentum()
                b.ledger -= np.concatenate([P, L])
                b.pv[:] = 0.0
                b.pell[:] = 0.0
                b.place()
                c = np.floor(b.x * b.inv_dx).astype(np.int64)
                lo = np.maximum(c.min(axis=0) - WAKE, 0)
                hi = np.minimum(c.max(axis=0) + 1 + WAKE, w.shape)
                box = tuple(slice(int(lo[a]), int(hi[a])) for a in range(3))
                b._nap_walls = (lo, hi, wall[box].copy())
        return sub

    def _bond_speed(self):
        """The `aloft` statistic for bonded matter, fragment by fragment:
        the highest mean member speed of any fragment. One mean over all
        of it let a crown lying still put a spray still sliding to sleep;
        the fastest member would call a resting crown's twitch a flight."""
        b = self.bonds
        if not len(b.v):
            return 0.0
        fr = b.fragments()[b.mp]
        n = int(fr.max()) + 1
        cnt = np.bincount(fr, minlength=n)
        sp = np.bincount(fr, np.linalg.norm(b.v, axis=1), n)
        return float((sp[cnt > 0] / cnt[cnt > 0]).max())

    def _bond_groups(self):
        """The bonded matter thing by thing, as the tap counts pieces of
        particles: (eid, batch) keys, each one's centre-of-mass velocity
        (m/s) and its kinetic plus potential energy (J)."""
        b = self.bonds
        key = (b.eid.astype(np.int64) << 32) | (b.batch.astype(np.int64) + 1)
        uk, g = np.unique(key, return_inverse=True)
        g = g.reshape(-1)
        n = len(uk)
        M = np.maximum(np.bincount(g, b.m, n), 1e-300)
        u = np.stack([np.bincount(g, b.m * b.v[:, a], n) for a in range(3)], 1) / M[:, None]
        e = np.bincount(g, b.m * (0.5 * (b.v * b.v).sum(axis=1) - b.x @ GRAVITY), n)
        keys = [(int(k >> 32), int(k & 0xFFFFFFFF) - 1) for k in uk]
        return keys, u, e

    def _bond_events(self, keys0, u0, e0, broken, dt):
        """The records of one bonds step (see `events`)."""
        b, w = self.bonds, self.w
        tick = int(getattr(w, "tick", 0))
        shape = np.array(w.shape)
        keys, u1, e1 = self._bond_groups()
        out = []
        at0 = {k: i for i, k in enumerate(keys0)}
        key = (b.eid.astype(np.int64) << 32) | (b.batch.astype(np.int64) + 1)

        def where(sel):
            at = b.x[sel].mean(axis=0) * b.inv_dx
            return (tuple(float(v) for v in at),
                    tuple(int(v) for v in np.clip(np.floor(at), 0, shape - 1)))
        armed = {}
        for i, k in enumerate(keys):
            j = at0.get(k)
            if j is None:
                continue
            jolt = float(_off_fall((u1[i] - u0[j])[None], GRAVITY * dt)[0])
            ready = self._bond_armed.get(k, True)
            if not (ready and jolt > HIT):
                armed[k] = ready or jolt < QUIET
                continue
            armed[k] = False
            sel = key == ((k[0] << 32) | (k[1] + 1))
            at, c = where(sel)
            out.append({"kind": "struck", "tick": tick, "eid": k[0], "batch": k[1],
                        "joules": max(float(e0[j] - e1[i]), 0.0),
                        "speed": float(np.linalg.norm(u0[j])) * b.inv_dx,
                        "jolt": jolt * b.inv_dx, "at": at, "cell": c,
                        "against": (), "hit": (),
                        "mats": tuple(int(v) for v in np.unique(b.vmat[sel]))})
        self._bond_armed = armed
        if len(broken):
            ps = np.unique(np.concatenate([b.bi[broken], b.bj[broken]]))
            did = np.isin(b.mp, ps)
            for kind in ("crack", "part"):
                for k in sorted(set(zip((b.eid[did]).tolist(), (b.batch[did]).tolist()))):
                    sel = did & (b.eid == k[0]) & (b.batch == k[1])
                    at, c = where(sel)
                    out.append({"kind": kind, "tick": tick, "eid": int(k[0]),
                                "batch": int(k[1]), "n": int(sel.sum()),
                                "at": at, "cell": c,
                                "mats": tuple(int(v) for v in np.unique(b.vmat[sel]))})
        return out

    # ------------------------------------------------------------ sleep (P8)

    def _units(self):
        """The particles in units that sleep and wake together: each
        particle's unit (n,) and each unit's name, the lowest (eid, batch)
        of the pieces in it. A unit is a piece (as the tap counts them, see
        `_Tap`) joined to every piece it shares a bound cluster with: half a
        cluster asleep would hold the other half to a shape that no longer
        moves with it."""
        m = self.mpm
        key = (m.eid.astype(np.int64) << 32) | (m.batch.astype(np.int64) + 1)
        uk, g = np.unique(key, return_inverse=True)
        g = g.reshape(-1)
        up = list(range(len(uk)))

        def root(i):
            while up[i] != i:
                up[i] = up[up[i]]
                i = up[i]
            return i

        bound = m.cid >= 0
        if bound.any():
            first = {}
            for c, i in np.unique(np.stack([m.cid[bound], g[bound]], 1), axis=0):
                j = first.setdefault(int(c), int(i))
                up[root(int(i))] = root(j)
        r = np.array([root(i) for i in range(len(uk))])
        names_r, unit = np.unique(r[g], return_inverse=True)
        unit = unit.reshape(-1)
        names = []
        for k in names_r:
            low = int(uk[r == k].min())
            names.append((low >> 32, (low & 0xFFFFFFFF) - 1))
        return unit, names

    def _within(self, a, b, r):
        """Does any particle of `a` (indices) have its stencil within `r`
        nodes of the stencil of some particle of `b`, on every axis (the
        rule TOUCH and APART are measured by)?"""
        if not len(a) or not len(b):
            return False
        m = self.mpm
        ba = np.floor(m.x[a] * m.inv_dx - 0.5).astype(np.int64)
        bb = np.floor(m.x[b] * m.inv_dx - 0.5).astype(np.int64)
        lo, hi = ba.min(axis=0) - r, ba.max(axis=0) + r
        bb = bb[((bb >= lo) & (bb <= hi)).all(axis=1)]
        return bool(len(bb)) and bool(_near_mask(bb, ba, r).any())

    def _reach(self, idx, dt):
        """How many nodes the particles `idx` can come within of something
        else in one step of `dt`: APART, and as far as the fastest of them
        can go in the step, falling the whole of it, and one node more."""
        m = self.mpm
        v = float(np.abs(m.v[idx]).max()) if len(idx) else 0.0
        go = (v * dt + 0.5 * float(np.abs(GRAVITY).max()) * dt * dt) * m.inv_dx
        return APART + int(np.ceil(go)) + 1

    def _box(self, idx):
        """The lattice a sleeper watches: its particles' cells and WAKE
        cells around them, [lo, hi), and the walls in it as the solver
        would see them now."""
        w, m = self.w, self.mpm
        c = np.floor(m.x[idx] * m.inv_dx).astype(np.int64)
        lo = np.maximum(c.min(axis=0) - WAKE, 0)
        hi = np.minimum(c.max(axis=0) + 1 + WAKE, w.shape)
        box = tuple(slice(int(lo[a]), int(hi[a])) for a in range(3))
        return lo, hi, (w.mat[box] != AIR) & ~self.drawn[box]

    def _doze(self, dt):
        """After a step: count each awake unit's still ticks, and put to
        sleep the units that have lain still for REST ticks, unless an awake
        particle that is not going to sleep with them could reach them in
        the next step (it would only wake them again). A sleeper keeps its
        place, its shape (`F`) and everything it carries; its speed and its
        velocity gradient become zero, which is what being at rest means,
        and what lets `_wake` see an impulse as a speed that is not zero."""
        m = self.mpm
        unit, names = self._units()
        n = len(names)
        asleep = np.bincount(unit, (m.nap >= 0).astype(float), n) > 0
        speed = np.bincount(unit, np.linalg.norm(m.v, axis=1), n) \
            / np.maximum(np.bincount(unit, minlength=n), 1)
        still, ready = {}, []
        for u, k in enumerate(names):
            if asleep[u]:
                continue
            still[k] = self._still.get(k, 0) + 1 if speed[u] <= STILL else 0
            if still[k] >= REST:
                ready.append(u)
        self._still = still
        if not ready:
            return
        # a unit stays up while something that stays up can reach it
        while ready:
            up = np.flatnonzero(~np.isin(unit, ready) & (m.nap < 0))
            r = self._reach(up, dt)
            keep = [u for u in ready
                    if not self._within(np.flatnonzero(unit == u), up, r)]
            if len(keep) == len(ready):
                break
            ready = keep
        for u in ready:
            idx = np.flatnonzero(unit == u)
            rid = self._nap_next
            self._nap_next += 1
            self._naps[rid] = self._box(idx)
            m.nap[idx] = rid
            m.v[idx] = 0.0
            m.C[idx] = 0.0
            del self._still[names[u]]

    def _wake(self, dt):
        """Before a step: wake every sleeper that something could move.
        Its lattice box changed (a support cell gone, a wall come), or a
        speed was written into it (an impulse), or an awake particle could
        reach it in this step; and then whatever the woken ones could
        reach, until nothing more wakes. What the lattice did to a sleeper's
        mass or material wakes it in `_write_back`, before this."""
        w, m = self.w, self.mpm
        live = set(int(r) for r in np.unique(m.nap[m.nap >= 0]))
        for rid in [r for r in self._naps if r not in live]:
            del self._naps[rid]                       # all of it removed
        rouse = set()
        for rid, (lo, hi, walls) in self._naps.items():
            box = tuple(slice(int(lo[a]), int(hi[a])) for a in range(3))
            if not np.array_equal((w.mat[box] != AIR) & ~self.drawn[box], walls):
                rouse.add(rid)
        moved = (m.nap >= 0) & ((m.v != 0.0).any(axis=1)
                                | (m.C != 0.0).any(axis=(1, 2)))
        rouse |= set(int(r) for r in np.unique(m.nap[moved]))
        self._rouse(rouse)
        up = np.flatnonzero(m.nap < 0)
        if not len(up):
            return
        r = self._reach(up, dt)
        while self._naps:
            near = {rid for rid in self._naps
                    if self._within(np.flatnonzero(m.nap == rid), up, r)}
            if not near:
                break
            self._rouse(near)
            up = np.flatnonzero(m.nap < 0)

    def _rouse(self, rids):
        """Wake the sleepers `rids`: their particles are solved again from
        the next step, starting from rest."""
        m = self.mpm
        for rid in rids:
            m.nap[m.nap == rid] = -1
            self._naps.pop(rid, None)

    def wake(self, eid=None):
        """Wake every sleeping particle of thing `eid`, or every sleeper.
        For a caller that is about to push particle matter by some way the
        wake rules do not watch; writing a speed into `mpm.v` is watched."""
        m = self.mpm
        mine = m.nap >= 0 if eid is None else (m.nap >= 0) & (m.eid == eid)
        self._rouse(set(int(r) for r in np.unique(m.nap[mine])))
        b = self.bonds
        if b.asleep and (eid is None or (b.eid == eid).any()):
            b.asleep = False
            b._still = 0

    def asleep(self):
        """Which particles sleep (a mask over the particles). Bonded matter
        sleeps as one: `bonds.asleep` (and `entities` reports both)."""
        return self.mpm.nap >= 0

    # ------------------------------------------------------- what it did

    def events(self):
        """What the particle matter did in the last `step`, as plain records:
        dicts of ints, floats, strings and tuples, in the lattice's units
        (places in cells, speeds in cells per second, energy in joules).
        Empty when the tap is off or nothing happened. The voxel side reads
        them after `rasterize` and turns them into its own events (design
        3.0, ruling 7: one producer, one consumer).

        Every record has `kind`, `tick`, and the piece it is about: `eid`
        (0 = no thing) and `batch` (the `promote` call it came in by; see
        `_Tap`). Then, by kind, in this order within a step:

          touch   two pieces came into contact. `other` is the second piece,
                  as (eid, batch); `at`, `cell` where they meet.
          struck  a blow (edge-triggered, see HIT and QUIET). `joules` is
                  the energy the piece lost in this step, `speed` how fast
                  its centre was going before the blow, `jolt` the speed
                  the blow took off; `against` the pieces it is touching as
                  (eid, batch) pairs, `hit` the lattice cells it lies against
                  (both empty: it bounced clear before the step ended);
                  `at`, `cell` where; `mats` its voxel materials.
          crack   `n` of its particles cracked (lost cohesion);
          part    `n` of its particles came loose from the cluster that
                  held their shape. A bond that parts cracks too, so a part
                  usually comes with a crack. `at`, `cell` and `mats` are
                  those particles'."""
        return list(self._events)

    def _first_touches(self, tap):
        """A piece the tap has not seen before may already lie against
        another: two things promoted together, or a thing promoted onto one
        already in the solver. That is how it came in, not a contact it
        made, so those pairs are taken as touching without a `touch`."""
        new = [i for i, k in enumerate(tap.keys) if k not in self._armed]
        if not new or tap.n < 2:
            return
        among = [(i, j) for i in range(tap.n) for j in range(i + 1, tap.n)
                 if i in new or j in new]
        for i, j in tap.pairs(tap.base(), TOUCH, among):
            self._touching.add((tap.keys[i], tap.keys[j]))

    def _read(self, tap, dt):
        """The step's records (`events`), from what `tap` watched, and the
        tap's memory carried to the next step: which pieces are touching,
        and which may report a blow."""
        m, w = self.mpm, self.w
        tick = int(getattr(w, "tick", 0))
        keys = tap.keys
        u = np.array(tap.u)                           # (substeps + 1, pieces, 3)
        lost = tap.e0 - tap._energy()
        jolt = _off_fall(u[-1] - u[0], GRAVITY * dt)
        h = dt / max(len(u) - 1, 1)
        shape = np.array(w.shape)
        out = []

        def where(pts):
            at = pts.mean(axis=0) * m.inv_dx
            return (tuple(float(v) for v in at),
                    tuple(int(v) for v in np.clip(np.floor(at), 0, shape - 1)))

        # TOUCH: a pair touches when its pieces share a node now (TOUCH), or
        # when they touched already and are not yet APART (the hysteresis).
        # A pair that shares a node and did not touch before is an onset.
        base = tap.base() if tap.n > 1 else None
        was = self._touching
        near = tap.pairs(base, APART) if tap.n > 1 else set()
        close = tap.pairs(base, TOUCH, sorted(near)) if near else set()
        now = {(i, j) for (i, j) in near
               if (i, j) in close or (keys[i], keys[j]) in was}
        for i, j in sorted(close, key=lambda p: (keys[p[0]], keys[p[1]])):
            if (keys[i], keys[j]) in was:
                continue
            # where they meet: halfway between the particles of each that
            # reach the other
            a, b = tap.members[i], tap.members[j]
            pa = _near_mask(base[a], base[b], TOUCH)
            pb = _near_mask(base[b], base[a], TOUCH)
            at, cell = where(np.stack([m.x[a][pa].mean(axis=0),
                                       m.x[b][pb].mean(axis=0)]))
            out.append({"kind": "touch", "tick": tick, "eid": keys[i][0],
                        "batch": keys[i][1], "other": keys[j],
                        "at": at, "cell": cell})
        self._touching = {(keys[i], keys[j]) for (i, j) in now}
        # STRUCK: a jolt past HIT from a piece that is armed
        armed = {}
        for i, k in enumerate(keys):
            ready = self._armed.get(k, True)
            if not (ready and jolt[i] > HIT):
                armed[k] = ready or bool(jolt[i] < QUIET)
                continue
            armed[k] = False
            s = tap.members[i]
            # how fast it went in: the fastest its centre moved before the
            # hardest jolt of the step (one blow is spread over substeps)
            hardest = int(np.argmax(_off_fall(u[1:, i] - u[:-1, i], GRAVITY * h)))
            speed = float(np.linalg.norm(u[:hardest + 1, i], axis=1).max())
            others = sorted({b for (a, b) in now if a == i}
                            | {a for (a, b) in now if b == i})
            contact = np.zeros(len(s), bool)
            for j in others:
                contact |= _near_mask(base[s], base[tap.members[j]], TOUCH)
            cell = np.clip(np.floor(m.x[s] * m.inv_dx).astype(np.int64), 0, shape - 1)
            walls, hit = self._against(cell, 1)
            if not walls.any() and not others:
                walls, hit = self._against(cell, 2)
            contact |= walls
            at, c = where(m.x[s][contact] if contact.any() else m.x[s])
            out.append({"kind": "struck", "tick": tick, "eid": k[0], "batch": k[1],
                        "joules": max(float(lost[i]), 0.0),
                        "speed": speed * m.inv_dx,
                        "jolt": float(jolt[i]) * m.inv_dx,
                        "at": at, "cell": c,
                        "against": tuple(keys[j] for j in others),
                        "hit": tuple(tuple(int(v) for v in q) for q in hit),
                        "mats": tuple(int(v) for v in np.unique(m.vmat[s]))})
        self._armed = armed
        # CRACK and PART: what the step broke. Both only ever go one way (a
        # crack does not heal, a parted bond is not made again), so the
        # step's start and end say all of it
        for kind, did in (("crack", m.dmg & ~tap.dmg0),
                          ("part", (tap.cid0 >= 0) & (m.cid < 0))):
            if not did.any():
                continue
            for i in np.unique(tap.g[did]):
                s = did & (tap.g == i)
                at, c = where(m.x[s])
                out.append({"kind": kind, "tick": tick, "eid": keys[i][0],
                            "batch": keys[i][1], "n": int(s.sum()),
                            "at": at, "cell": c,
                            "mats": tuple(int(v) for v in np.unique(m.vmat[s]))})
        return out

    def _against(self, cell, k):
        """Which of the particles whose cells are `cell` (n, 3) have a
        wall of the collider `k` cells away along an axis, and those wall
        cells (the collider is the lattice the solver stepped against)."""
        wall = self.mpm.wall
        touch = np.zeros(len(cell), bool)
        hit = [np.zeros((0, 3), np.int64)]
        for a in range(3):
            for s in (-k, k):
                nb = cell.copy()
                nb[:, a] += s
                ok = (nb[:, a] >= 0) & (nb[:, a] < wall.shape[a])
                there = np.zeros(len(cell), bool)
                there[ok] = wall[tuple(nb[ok].T)]
                touch |= there
                hit.append(nb[there])
        return touch, np.unique(np.concatenate(hit), axis=0)

    def unrasterize(self):
        """Take particle matter OFF the lattice before the laws that would try
        to move it run. Support relaxes columns, torque tips clusters, footing
        launches bodies — all of them are the solver's job now for anything
        that is already the solver's.

        FIRST IT TAKES BACK WHAT THE LATTICE DID TO IT. The seam was one-way:
        a block drawn beside a torch took the torch's heat into its cells,
        and this then emptied the cells and the heat was gone. Measured, a
        promoted block's energy ledger drifted -22,883 J where the same block
        left on the lattice drifted -2,889, and a promoted block in a fire
        never burned at all, because every tick it was drawn again from
        particles that had not changed. So the change in every drawn cell is
        handed back to the particles that were drawn there (`_written`),
        and only then are the cells emptied."""
        if not self.drawn.any():
            return
        plan = self._written()
        if plan is not None:
            self._write_back(plan)
        if len(self.bonds.drawn.drawn_at):
            bk = self.bonds.drawn
            plan = self._written(self.bonds, bk)
            if plan is not None:
                self._write_back_bonds(plan)
            self.bonds.drawn = type(bk)()
        w = self.w
        d = self.drawn
        for arr in (w.mat, w.smass, w.E, w.edge, w.rubble):
            arr[d] = 0
        for lf, _ in _opt_live(w):
            getattr(w, lf)[d] = 0
        weid = getattr(w, "eid", None)
        if weid is not None:
            weid[d] = 0
        df = self.drawn_fluid
        for arr in (w.fl, w.fvol, w.fpot):
            arr[df] = 0
        w.moving[d] = False
        self.drawn[:] = False
        self.drawn_fluid[:] = False
        self.drawn_at = np.zeros(0, np.int64)
        self.drawn_eid = np.zeros(0, np.int32)
        self.drawn_was = {}
        self.drawn_who = np.zeros(0, np.int64)
        self.drawn_in = np.zeros(0, np.int64)

    def _written(self, mp=None, bk=None):
        """What the lattice did to the drawn cells since `rasterize` drew
        them, as the change it makes to the particles. None when it did
        nothing, so a tick with no edits leaves every particle as it was.

        Between `rasterize` and `unrasterize` the laws read and write the
        drawn cells like any others. They may change WHAT a cell holds: heat
        conducted in or radiated away, wood burned, iron melted, a cell
        cracked by heat. They never change WHERE it is, because every mover
        refuses a `moving` cell. So each change belongs to the particles
        drawn into that cell, and goes back to them:
          - heat, shared by mass;
          - grams, by one ratio for the whole cell, and the volume at rest
            by the same ratio, so the matter keeps its density;
          - the fluid the seam drew there (`drawn_fluid`), shared by what
            each particle carried, or by mass if none of them carried any;
          - a cell turned to char or ash turns its particles of the cell's
            material to char or ash;
          - a cell that became rubble cracks its particles (`dmg`).
        A cell that became air, or some other material, or has no grams
        left, is not the particles' any more. They are removed, and the
        lattice keeps the cell as it is now (its heat, its fluid, whatever
        moved in). Any fluid those particles carried that was never drawn
        there is poured into it.

        The answer is a dict: `_write_back` applies it, and the ledgers read
        it, so a total taken between two ticks already counts the change.
        `mp` and `bk` are the matter and its drawing's bookkeeping: the
        particles and this seam's own by default, or the bonds' members and
        theirs (`Bonds.drawn`)."""
        if mp is None:
            mp, bk = self.mpm, self
        at = bk.drawn_at
        if not len(at):
            return None
        w = self.w
        was = bk.drawn_was
        now = {f: getattr(w, f).ravel()[at] for f in _SOLID_FIELDS + _FLUID_FIELDS}
        ours = self.drawn_fluid.ravel()[at]
        solid = np.zeros(len(at), bool)
        for f in _SOLID_FIELDS:
            solid |= now[f] != was[f]
        wet = np.zeros(len(at), bool)
        for f in _FLUID_FIELDS:
            wet |= now[f] != was[f]
        wet &= ours
        changed = solid | wet
        if not changed.any():
            return None
        n = len(at)
        sel = changed[bk.drawn_in]
        p, c = bk.drawn_who[sel], bk.drawn_in[sel]       # those particles, and
        kg = mp.m[p]                                     # which cell each is in
        M = np.maximum(np.bincount(c, kg, n), 1e-300)    # kg drawn into each cell
        s0 = was["smass"].astype(np.float64)
        s1 = now["smass"].astype(np.float64)
        left = M + (s1 - s0) / 1000.0                    # kg each one holds now
        m0, m1 = was["mat"], now["mat"]
        burnt = (m1 != m0) & np.isin(m1, _BURNT)
        gone = changed & (((m1 != m0) & ~burnt) | (s1 <= 0.0) | (left <= 0.0))
        ratio = (left / M)[c]
        dE = now["E"].astype(np.float64) - was["E"].astype(np.float64)
        vmat = mp.vmat[p].copy()
        turn = (burnt & ~gone)[c] & (vmat == m0[c])
        vmat[turn] = m1[c][turn]
        plan = {"p": p, "drop": gone[c],
                "m": kg * ratio, "vol0": mp.vol0[p] * ratio,
                "E": mp.E[p] + dE[c] * (kg / M[c]),
                "vmat": vmat, "turn": turn,
                "dmg": mp.dmg[p] | (now["rubble"] & ~was["rubble"])[c]}
        fl, fvol, fpot = mp.fl[p].copy(), mp.fvol[p].copy(), mp.fpot[p].copy()
        q = (wet & ~gone)[c]
        if q.any():
            held = np.bincount(c, fvol, n)               # what they carry there
            v1 = now["fvol"].astype(np.float64)
            dv = v1 - was["fvol"].astype(np.float64)
            by = np.where(held[c] > 0.0, fvol / np.maximum(held[c], 1e-300),
                          kg / M[c])
            new = np.where(v1[c] > 0.0, np.maximum(fvol + dv[c] * by, 0.0), 0.0)
            fvol[q] = new[q]
            fl[q] = np.where(new[q] > 0.0, now["fl"][c][q], 0)
            fpot[q] = np.where(new[q] > 0.0, now["fpot"][c][q], 0.0)
        plan.update(fl=fl, fvol=fvol, fpot=fpot)
        # the cells handed to the lattice: as they are now, and as they will
        # be once the fluid that was never drawn there is poured in
        h = np.flatnonzero(gone)
        hfl = now["fl"][h].astype(np.int64)
        hv = now["fvol"][h].astype(np.float64)
        hp = now["fpot"][h].astype(np.float64)
        plan.update(hand=at[h], hand_eid=bk.drawn_eid[h], hand_mat=m1[h],
                    hand_smass=s1[h], hand_E=now["E"][h].astype(np.float64),
                    hand_ours=ours[h], hand_fl0=hfl.copy(), hand_fvol0=hv.copy())
        pour = gone[c] & ~ours[c] & (mp.fvol[p] > 0.0)
        if pour.any():
            for i in np.flatnonzero(np.isin(h, c[pour])):
                mine = pour & (c == h[i])
                hfl[i], hv[i], hp[i] = _mix(hfl[i], hv[i], hp[i], mp.fl[p][mine],
                                            mp.fvol[p][mine], mp.fpot[p][mine])
        plan.update(hand_fl=hfl, hand_fvol=hv, hand_fpot=hp)
        return plan

    def _write_back(self, plan):
        """Apply what `_written` found: to the particles, then to the cells
        the lattice keeps, then remove the particles those cells held."""
        w, mp = self.w, self.mpm
        p = plan["p"]
        # A SLEEPER WAKES when this changes what the solver reads about it:
        # its mass (and with it its volume), its material, a crack, a
        # particle gone. Heat and the fluid it carries the solver never
        # reads, so a sleeping block warmed by a fire sleeps on, and is
        # still handed every joule.
        nap = mp.nap[p]
        if (nap >= 0).any():
            felt = plan["drop"] | plan["turn"] | (plan["m"] != mp.m[p]) \
                | (plan["dmg"] != mp.dmg[p])
            self._rouse(set(int(r) for r in np.unique(nap[felt & (nap >= 0)])))
        for f in ("m", "vol0", "E", "dmg", "fl", "fvol", "fpot"):
            getattr(mp, f)[p] = plan[f]
        turn = plan["turn"]
        for v in np.unique(plan["vmat"][turn]):
            self._become(p[turn & (plan["vmat"] == v)], int(v))
        cells = plan["hand"]
        if len(cells):
            w.fl.ravel()[cells] = plan["hand_fl"]
            w.fvol.ravel()[cells] = plan["hand_fvol"]
            w.fpot.ravel()[cells] = plan["hand_fpot"]
            w.moving.ravel()[cells] = False
            self.drawn.ravel()[cells] = False
            self.drawn_fluid.ravel()[cells] = False
            weid = getattr(w, "eid", None)
            if weid is not None:
                # our thing's mark goes, unless what is there now set its own
                ours = weid.ravel()[cells] == plan["hand_eid"]
                weid.ravel()[cells[ours]] = 0
        if plan["drop"].any():
            keep = np.ones(len(mp.x), bool)
            keep[p[plan["drop"]]] = False
            mp.keep(keep)

    def _hand(self, plan):
        """The cells `_written` gives back to the lattice: its fluid as it
        will be, and no longer drawn or the seam's thing."""
        w = self.w
        cells = plan["hand"]
        if len(cells):
            w.fl.ravel()[cells] = plan["hand_fl"]
            w.fvol.ravel()[cells] = plan["hand_fvol"]
            w.fpot.ravel()[cells] = plan["hand_fpot"]
            w.moving.ravel()[cells] = False
            self.drawn.ravel()[cells] = False
            self.drawn_fluid.ravel()[cells] = False
            weid = getattr(w, "eid", None)
            if weid is not None:
                ours = weid.ravel()[cells] == plan["hand_eid"]
                weid.ravel()[cells[ours]] = 0

    def _write_back_bonds(self, plan):
        """`_write_back` for the bonds' members. Heat and fluid go on the
        members as they do on particles. A change of grams goes through
        `Bonds.reweigh`, so each piece's mass, centre and turning stay
        exact. A cell that cracked (rubble) or burned to ash holds nothing:
        every bond of its piece parts. Anything the solver would feel wakes
        the bonded matter."""
        b = self.bonds
        p = plan["p"]
        felt = plan["drop"] | plan["turn"] | (plan["m"] != b.m[p]) \
            | (plan["dmg"] != b.dmg[p])
        if felt.any():
            b.asleep = False
            b._still = 0
        dry = (b.vmat[p] == LEAF) & (b.fvol[p] > 0.0) & (plan["fvol"] <= 0.0)
        for f in ("vol0", "E", "fl", "fvol", "fpot"):
            getattr(b, f)[p] = plan[f]
        if dry.any():
            b.dried(p[dry])
        cracked = plan["dmg"] & ~b.dmg[p]
        b.dmg[p] = plan["dmg"]
        turn = plan["turn"]
        b.vmat[p[turn]] = plan["vmat"][turn]
        ash = turn & (plan["vmat"] == int(_ASH))
        if (cracked | ash).any():
            b.cut(np.unique(b.mp[p[cracked | ash]]))
        heavy = (plan["m"] != b.m[p]) & ~plan["drop"]
        if heavy.any():
            b.reweigh(p[heavy], plan["m"][heavy])
        self._hand(plan)
        if plan["drop"].any():
            keep = np.ones(len(b.m), bool)
            keep[p[plan["drop"]]] = False
            b.keep(keep)

    def rasterize(self):
        """Draw particle matter back onto the lattice for the laws that read
        it, then the bonded matter, by the same rule (`_raster`)."""
        self._raster(self.mpm, self)
        if len(self.bonds):
            self.bonds.drawn = type(self.bonds.drawn)()
            self._raster(self.bonds, self.bonds.drawn)

    def _raster(self, m, bk):
        """Draw particle matter back onto the lattice for the laws that read it
        — conduction, burning, sight, the minds. A cell gets the material of
        most of its mass, the thing (`eid`) that has most of that material
        there, the sum of everything else, and `moving` set so no law
        mistakes it for something that stands still. A cell another matter
        drew this tick is not free, as no lattice cell is. `m` is the matter
        (src/mpm's particles, or the bonds' members) and `bk` keeps what was
        drawn (this seam itself for the particles, `Bonds.drawn` for bonds)."""
        if not len(m.x):
            return
        w = self.w
        cell = np.clip((m.x * m.inv_dx).astype(np.int64), 0,
                       np.array(w.shape) - 1)
        free = w.mat[tuple(cell.T)] == AIR
        cell = cell[free]
        if not len(cell):
            return
        # ONLY THE CELLS THAT HOLD A PARTICLE ARE SUMMED. Every sum below ran
        # over an array the size of the world, and a 400k-cell house paid
        # 33.8 ms a tick to add up 3,662 particles. `at` is the cells the
        # particles are in, and `flat` says which of them each particle is
        # in. `np.bincount` adds a cell's particles in particle order either
        # way, so these are the world-sized sums, bit for bit.
        at, flat = np.unique(np.ravel_multi_index(tuple(cell.T), w.shape),
                             return_inverse=True)
        n = len(at)
        grams = np.bincount(flat, m.m[free] * 1000.0, n)
        # material: whichever has the most mass in the cell
        best_mat = np.zeros(n, np.int16)
        best_g = np.zeros(n)
        for vm in np.unique(m.vmat[free]):
            g = np.bincount(flat, np.where(m.vmat[free] == vm, m.m[free], 0.0), n)
            win = g > best_g
            best_mat[win], best_g[win] = vm, g[win]
        eid = _cell_eids(flat, n, m.vmat[free], m.m[free], m.eid[free], best_mat)
        # THE AIR'S HEAT GOES INTO WHAT MOVES IN. The cells drawn into are
        # air, and air holds heat (`E` above ambient) like any cell. Writing
        # the matter's heat over it deleted that heat: a warm crown falling
        # through the air it had warmed lost 1.5 J in 60 ticks, the same
        # crown on the particles 0.66 J. So what the air held is shared out
        # to the matter drawn there, by mass, and the cell is the sum.
        air = w.E.ravel()[at].astype(np.float64)
        if (air != 0.0).any():
            idx = np.flatnonzero(free)
            by = np.where(grams[flat] > 0.0, m.m[free] * 1000.0 / np.maximum(grams[flat], 1e-300), 0.0)
            m.E[idx] = m.E[idx] + air[flat] * by
        E = np.bincount(flat, m.E[free], n)
        # ONLY WHAT IS STILL IN A VESSEL is drawn-and-reclaimed. A cracked
        # particle's liquid is poured below instead, and summing it in both
        # places counted every millilitre twice (1000 ml of acid read as
        # 2000 the tick the vial broke).
        whole = ~m.dmg[free]
        fvol = np.bincount(flat, np.where(whole, m.fvol[free], 0.0), n)
        fpot = np.bincount(flat, np.where(whole, m.fpot[free] * m.fvol[free], 0.0), n)
        edge = np.zeros(n)
        np.maximum.at(edge, flat, m.edge[free])
        # one fluid id per cell: the one carrying the most volume
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
        cells = at[hit]                    # the world's cells this draws into
        w.mat.ravel()[cells] = best_mat[hit]
        w.smass.ravel()[cells] = grams[hit]
        w.E.ravel()[cells] = E[hit]
        bk.drawn_at, bk.drawn_eid = cells, eid[hit]
        weid = getattr(w, "eid", None)
        if weid is not None:
            weid.ravel()[cells] = eid[hit]
        # FLUID THE LATTICE ALREADY HOLDS IS NOT OURS TO OVERWRITE. A vat is
        # glass cells around AIR cells holding acid; the glass is promoted,
        # the acid stays on the lattice (as it did under `_topple`), and the
        # falling vat passes through its own hollow. Writing the particles'
        # fluid (none) onto those cells deleted 1000 ml of acid from the
        # world in two ticks, and with it the whole alchemist cascade. The
        # seam draws its fluid only into cells carrying none, and remembers
        # which, so the lattice's fluid law can pour the rest as it should.
        empty = w.fvol.ravel()[at] <= 0.0
        draw_f = hit & empty
        into = at[draw_f]
        w.fvol.ravel()[into] = fvol[draw_f]
        w.fpot.ravel()[into] = np.where(fvol[draw_f] > 0, fpot[draw_f] / np.maximum(fvol[draw_f], 1e-12), 0.0)
        w.fl.ravel()[into] = fl[draw_f]
        self.drawn_fluid.ravel()[into] = True
        # A SHARD DOES NOT CARRY ACID. What an intact vessel holds is drawn on
        # to the lattice each tick and taken back the next, because it is
        # still in the vessel. The moment a piece cracks it is not a vessel
        # any more, so its liquid is POURED: handed to the lattice for good,
        # where the fluid law can run it downhill and the acid law can eat
        # what it lands on. Without this the vial broke and its acid stayed
        # glued to the flying glass for ever.
        if m.dmg.any():
            sp = m.dmg[free] & (m.fvol[free] > 0.0)
            if sp.any():
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
                into = at[got]
                w.fvol.ravel()[into] += add[got]
                w.fpot.ravel()[into] = np.where(
                    w.fvol.ravel()[into] > 0,
                    apot[got] / np.maximum(add[got], 1e-12), 0.0)
                w.fl.ravel()[into] = aid[got]
                self.drawn_fluid.ravel()[into] = False  # the lattice owns it now
                idxs = np.flatnonzero(free)[sp]
                m.fvol[idxs] = 0.0
                m.fl[idxs] = 0
        w.edge.ravel()[cells] = edge[hit]
        # THE OPTIONAL FIELDS (wave 13), only where both sides have the array:
        # a cell gets the mix of its heaviest particle, the mass-weighted
        # damage, and all the ice.
        for lf, pf in _opt_live(w):
            if pf not in m._opt:
                continue
            a = getattr(m, pf)[free]
            if pf in ("scomp", "fcomp"):
                head = np.zeros(n, a.dtype)
                order = np.argsort(m.m[free], kind="stable")   # heaviest last
                for k in order:                  # (a loop: the last write wins)
                    head[flat[k]] = a[k]
                val = head
            elif pf == "dscal":
                val = np.bincount(flat, a * m.m[free], n) / np.maximum(grams / 1000.0, 1e-300)
            else:
                val = np.bincount(flat, a, n)
            getattr(w, lf).ravel()[cells] = val[hit]
        # a cracked particle is rubble: it rests on what is under it and
        # holds nothing up, which is what `rubble` already means
        rub = np.bincount(flat, m.dmg[free].astype(float), n) > 0
        w.rubble.ravel()[cells] = rub[hit]
        w.moving.ravel()[cells] = True
        self.drawn.ravel()[cells] = True
        # WHAT THE LATTICE WAS GIVEN, so that `_written` can tell what the
        # laws did with it: each field they may change, as the lattice holds
        # it (in its own float32), and which particles went into which cell
        bk.drawn_was = {f: getattr(w, f).ravel()[cells].copy()
                        for f in _SOLID_FIELDS + _FLUID_FIELDS}
        pick = hit[flat]
        bk.drawn_who = np.flatnonzero(free)[pick]
        bk.drawn_in = (np.cumsum(hit) - 1)[flat[pick]]

    def collide(self, cells, vel):
        """A person as a moving surface. Not yet: bodies stay on the lattice
        until `arms` and `crouch` migrate (roadmap 93, order of work 5)."""
        raise NotImplementedError

    # ---------------------------------------------------------- what is asked

    def aloft(self):
        """Is anything still moving? What `not w.bodies` used to mean.

        A BODY QUESTION, ANSWERED WITH A BODY STATISTIC. Explicit contact
        leaves a resting body with a microscopic twitch — measured, a settled
        pole at a mean of 0.006 m/s with one particle now and then past 0.02
        — and asking whether ANY particle moves calls that pole airborne. The
        mean over the cluster falls two orders of magnitude between falling
        and lying still, which is the separation a threshold needs."""
        m = self.mpm
        if bool(len(m.x)) and float(np.linalg.norm(m.v, axis=1).mean()) > STILL:
            return True
        return self._bond_speed() > STILL

    def aloft_owners(self):
        """The people whose bodies are in the air as particles: nobody.
        Bodies stay on the lattice (roadmap 93, decision 4), so no particle
        has an owner. Kept so that the laws that ask (`persons.py`,
        `motion.py`) need not change; a thing's own state is `entities()`."""
        return set()

    def entities(self):
        """The particle matter thing by thing: {eid: facts}. Eid 0, matter
        that no thing owns, is one entry like any other.

        In the lattice's units, as everything this file hands back is:
            cells     (k, 3) the cells the last `rasterize` gave this eid
                      (empty while nothing is drawn)
            centroid  centre of mass, in cells
            vel       velocity of the centre of mass, cells per second
            speed     mean speed of its particles, cells per second
            aloft     still moving, by the rule `aloft()` uses for everything
            mix       {voxel material id: grams}
            fluid     {fluid id: volume} carried on its particles
            lo, hi    the box its particles span, in whole cells [lo, hi)
            particles how many
            asleep    the solver is not carrying it (REST)
        """
        m = self.mpm
        out = {}
        if not len(m.x) and not len(self.bonds):
            return out
        drawn_at, drawn_eid = self.drawn_at, self.drawn_eid
        if len(self.bonds):
            # the particles and the bonds' members, as one set of points
            b, bk = self.bonds, self.bonds.drawn
            m = type("Points", (), {})()
            for f in ("x", "v", "m", "vmat", "fl", "fvol", "eid"):
                setattr(m, f, np.concatenate([getattr(self.mpm, f), getattr(b, f)]))
            # bonded matter sleeps as one (`_step_bonds`): its members nap
            # when it does
            m.nap = np.concatenate([self.mpm.nap, np.full(len(b), 0 if b.asleep else -1,
                                                          self.mpm.nap.dtype)])
            m.inv_dx = self.mpm.inv_dx
            drawn_at = np.concatenate([drawn_at, bk.drawn_at])
            drawn_eid = np.concatenate([drawn_eid, bk.drawn_eid])
        sp = np.linalg.norm(m.v, axis=1)
        for e in np.unique(m.eid):
            s = m.eid == e
            kg = m.m[s]
            tot = max(float(kg.sum()), 1e-12)
            pos = m.x[s] * m.inv_dx
            vm, fl = m.vmat[s], m.fl[s]
            out[int(e)] = {
                "cells": np.stack(np.unravel_index(
                    drawn_at[drawn_eid == e], self.w.shape), -1),
                "centroid": (kg[:, None] * pos).sum(axis=0) / tot,
                "vel": (kg[:, None] * m.v[s]).sum(axis=0) / tot * m.inv_dx,
                "speed": float(sp[s].mean()) * m.inv_dx,
                "aloft": float(sp[s].mean()) > STILL,
                "mix": {int(v): float(kg[vm == v].sum()) * 1000.0
                        for v in np.unique(vm)},
                "fluid": {int(f): float(m.fvol[s][fl == f].sum())
                          for f in np.unique(fl) if f != 0},
                "lo": np.floor(pos.min(axis=0)).astype(np.int64),
                "hi": np.floor(pos.max(axis=0)).astype(np.int64) + 1,
                "particles": int(s.sum()),
                "asleep": bool((m.nap[s] >= 0).all())}
        return out

    # THE LEDGERS COUNT WHAT IS PENDING. Between two ticks the lattice has
    # already burned, heated or melted the drawn cells, and the particles
    # hear of it only at the next `unrasterize`. A total read now counts the
    # particles as that will leave them, and the cells it will hand to the
    # lattice (which the lattice's own sums skip while they are `moving`),
    # so the total before the hand-back and after it is the same number.

    def _pending(self, names, m=None, bk=None):
        """The particle fields `names` as `_written` would leave them (a
        removed particle holds nothing), and the plan; or the live fields
        and None when nothing is pending. `m` and `bk` as for `_written`."""
        if m is None:
            m, bk = self.mpm, self
        plan = self._written(m, bk)
        if plan is None:
            return [getattr(m, f) for f in names], None
        out = []
        for f in names:
            a = getattr(m, f).copy()
            a[plan["p"]] = plan[f]
            if f in ("m", "E", "fvol"):
                a[plan["p"][plan["drop"]]] = 0.0
            out.append(a)
        return out, plan

    def mass(self):
        """Grams per voxel material id, for the ledger."""
        (kg, vmat), plan = self._pending(("m", "vmat"))
        out = {}
        if plan is not None:
            for v, g in zip(plan["hand_mat"], plan["hand_smass"]):
                if v != AIR:
                    out[int(v)] = out.get(int(v), 0.0) + float(g)
        for vm in np.unique(vmat):
            out[int(vm)] = out.get(int(vm), 0.0) + float(kg[vmat == vm].sum()) * 1000.0
        if len(self.bonds) or len(self.bonds.drawn.drawn_at):
            (kg, vmat), plan = self._pending(("m", "vmat"), self.bonds, self.bonds.drawn)
            if plan is not None:
                for v, g in zip(plan["hand_mat"], plan["hand_smass"]):
                    if v != AIR:
                        out[int(v)] = out.get(int(v), 0.0) + float(g)
            for vm in np.unique(vmat):
                out[int(vm)] = out.get(int(vm), 0.0) + float(kg[vmat == vm].sum()) * 1000.0
        return out

    def energy(self):
        out = self._energy()
        if len(self.bonds) or len(self.bonds.drawn.drawn_at):
            out += self._energy(self.bonds, self.bonds.drawn)
        return out

    def _energy(self, m=None, bk=None):
        (E,), plan = self._pending(("E",), m, bk)
        if plan is None:
            return float(E.sum())
        return float(E.sum()) + float(plan["hand_E"].sum())

    def fluid(self, f):
        out = self._fluid(f)
        if len(self.bonds) or len(self.bonds.drawn.drawn_at):
            out += self._fluid(f, self.bonds, self.bonds.drawn)
        return out

    def _fluid(self, f, m=None, bk=None):
        (fl, fvol), plan = self._pending(("fl", "fvol"), m, bk)
        if plan is None:
            return float(fvol[fl == f].sum())
        # a handed cell's fluid starts to count; the lattice already counted
        # the fluid of one whose fluid was never the seam's
        now = plan["hand_fvol"][plan["hand_fl"] == f].sum()
        was = plan["hand_fvol0"][(plan["hand_fl0"] == f) & ~plan["hand_ours"]].sum()
        return float(fvol[fl == f].sum()) + float(now) - float(was)

    def points(self):
        """Every point of matter the seam carries, particles first and then
        the bonds' members, as one object with the fields a reader of
        `matter.mpm` asks for: x, v, m, vmat, eid, cid. A member's `cid` is
        its fragment (the pieces its unbroken bonds hold together),
        numbered after the particles' clusters, so one fragment is one
        "piece" as a bound cluster of particles is. With no bonded matter
        this is `mpm` itself."""
        m, b = self.mpm, self.bonds
        if not len(b):
            return m
        out = type("Points", (), {})()
        for f in ("x", "v", "m", "vmat", "eid"):
            setattr(out, f, np.concatenate([getattr(m, f), getattr(b, f)]))
        fr = b.fragments()[b.mp]
        top = int(m.cid.max()) + 1 if len(m.cid) else 0
        out.cid = np.concatenate([m.cid, top + fr]).astype(m.cid.dtype)
        return out

    def set_eid(self, idx, e):
        """Give the points `idx` (as `points` numbers them) to thing `e`."""
        idx = np.asarray(idx, np.int64)
        n = len(self.mpm.x)
        self.mpm.eid[idx[idx < n]] = e
        self.bonds.eid[idx[idx >= n] - n] = e

    def __len__(self):
        return len(self.mpm.x) + len(self.bonds)
