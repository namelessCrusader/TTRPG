"""Sight, hearing, sound levels, speech, and noticing things in flight."""
import weakref

import numpy as np
from .materials import (AIR, FLESH, HEAR_DB, LEAF, SEE_TAU, SHOUT_DB, SMOKE_EXT, SOUND, _DENS_ARR, _TRANSMIT_ARR)
from . import smokeopt
from .events import _Pending
from .kernels import HAVE_NUMBA, _hear_levels, _look_rays, exact_math, njit
from .policy import (LINES, LINE_VOICE, LOMBARD, VOICE_DB, VOICE_VERB, WILL)


# THE WAYS ROUND, per world: the cells sound can pass through, and how far
# each cell is from an ear by them (`_flood`). Kept beside the World and not
# in it, so a snapshot does not copy it and the golden hash does not see it.
#
# It is kept for as long as THE CELLS ARE THE SAME, and no longer. It used to
# be kept for the tick, so a doorway filled in during a tick still let sound
# round for the rest of it. Now every use first checks the cells against the
# ones the floods were made from (`_same_open`, a fraction of a millisecond),
# so a wall built or knocked through, a world put back, or a test that edits
# cells by hand is seen at once. And a flood is kept ACROSS ticks while
# nothing changes, which is most ticks: people walking do not change it, as
# flesh and air both let sound by.
_ROUND = weakref.WeakKeyDictionary()
_FLOODS_KEPT = 16                # floods kept per world, the oldest dropped,
_FLOOD_BYTES = 64 << 20          # and no more than this much of them
_FACES = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                   (0, 0, 1), (0, 0, -1)], np.int64)


def _through(mat, smass, vox_l):
    """What gets past each of these cells, for sight: the (1-f) that misses
    the stuff in it, plus the f that hits it and comes through anyway (glass).
    Elementwise, so a cell gives the same number wherever it sits."""
    packed = np.clip(smass / np.maximum(_DENS_ARR[mat] * vox_l, 1e-9), 0.0, 0.999)
    return (1.0 - packed) + packed * _TRANSMIT_ARR[mat]


@njit(cache=True, inline="always")
def _cell_open(mat, smass, dvl, lim, x, y, z):
    """Can sound pass cell (x, y, z)? Air, flesh and leaves, or anything
    less than `lim` full (`lim` a float32, `dvl` each material's full mass
    in a voxel). Every test of a cell for sound goes through here."""
    m = mat[x, y, z]
    return m == AIR or m == FLESH or m == LEAF or \
        np.float32(smass[x, y, z] / dvl[m]) < lim


@njit(cache=True)
def _open_mask(mat, smass, dvl, lim):
    """The cells sound can pass through (`_cell_open`), all of them. The
    mask is made here and checked by `_same_open` with the very same test,
    so the two can never disagree over a cell whose fill sits within
    rounding of `lim` (numpy's own test was made in whatever float the
    fill came out in, and the check in float32)."""
    nx, ny, nz = mat.shape
    out = np.empty(mat.shape, np.bool_)
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                m = mat[x, y, z]                 # `_cell_open`, written out:
                out[x, y, z] = m == AIR or m == FLESH or m == LEAF or \
                    np.float32(smass[x, y, z] / dvl[m]) < lim
    return out


@njit(cache=True)
def _same_open(mat, smass, dvl, lim, open_):
    """Are the cells sound can pass through still exactly `open_` (made by
    `_open_mask`)? Cell by cell, stopping at the first cell that differs."""
    if mat.shape != open_.shape:
        return False
    nx, ny, nz = mat.shape
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                m = mat[x, y, z]                 # `_cell_open`, written out:
                o = m == AIR or m == FLESH or m == LEAF or \
                    np.float32(smass[x, y, z] / dvl[m]) < lim
                if o != open_[x, y, z]:          # a call a cell is 10x slower
                    return False
    return True


@njit(cache=True)
def _narrow(mat, smass, dvl, lim, c, j, w):
    """Is there wall within `w` cells of cell `c` on BOTH sides along axis
    `j`? The edge of the world is not wall."""
    nj = mat.shape[j]
    q = np.empty(3, np.int64)
    for s in (-1, 1):
        q[0], q[1], q[2] = c[0], c[1], c[2]
        hit = False
        for t in range(1, w + 1):
            q[j] = c[j] + s * t
            if q[j] < 0 or q[j] >= nj:
                break
            if not _cell_open(mat, smass, dvl, lim, q[0], q[1], q[2]):
                hit = True
                break
        if not hit:
            return False
    return True


@njit(cache=True)
def _opening(mat, smass, dvl, lim, path, n, a, b, lam):
    """How open the TIGHTEST place on a way is: the share of the first
    Fresnel zone that is open, where the way passes through a hole. The
    way is `path[:n]`, cells from `a`'s end to `b`'s. It is asked of the
    way round (`_round_length`) and of the straight line (`_ray_opening`)
    alike: a hole costs the same whether the sound goes straight through
    it or has to go round to it.

    A hole passes the sound that falls on it, and no more (composite
    transmission loss, row A12). What falls on it is what crosses its
    area, so the question is how big it is next to the patch of wall that
    would carry the sound if the wall were not there. That patch is the
    first Fresnel zone, a disc of radius sqrt(wavelength d1 d2 / (d1 +
    d2)) round the way, d1 and d2 the distances to the two ends: about a
    metre across a room. A doorway fills most of it and costs nothing; a
    5 cm hole is a thousandth of it and costs 30 dB.

    A cell of the way is in a hole if the way goes straight through it
    (moves along one axis there, and is open on both sides along it) and,
    across that axis, there is wall on BOTH sides of it within the disc,
    along one direction at least (a slit) or both (a hole). A way along a
    floor or past one wall is not in a hole: a floor or a wall beside the
    way is a mirror, not a hole's edge. The disc is counted only where it
    is inside the world, so a slit at the bottom of the world (a gap under
    a door) is compared with the half of the disc above it, which is
    roughly what the floor's mirror image does to it: it doubles the slit.

    The share does not depend on how thick the wall is. A hole is as
    narrow in a 5 cm plank as in a 20 cm wall, and it is the narrowness,
    not the depth, that keeps the sound out.

    And a hole has TWO MOUTHS on the way: the sound comes into it from the
    open and goes out of it into the open. A corridor is not a hole to
    someone standing in it. Its walls are two mirrors, and between them
    the sound is carried along, not cut down to the corridor's share of
    the zone. So the way is walked from the hole to each end, and if it is
    still between walls as close as the hole's own when it gets there,
    that end is inside, and the place is not charged as a hole.

    `mat`, `smass`, `dvl` and `lim` are what `_cell_open` reads. Returns
    1.0 if the way goes through no hole."""
    dims = np.array(mat.shape, np.int64)
    c = np.empty(3, np.int64)
    q = np.empty(3, np.int64)
    e = np.empty(3, np.int64)
    least = 1.0
    for i in range(n):
        for x in range(3):
            c[x] = int(path[i, x])
        d1 = np.sqrt((path[i, 0] - a[0]) ** 2 + (path[i, 1] - a[1]) ** 2
                     + (path[i, 2] - a[2]) ** 2)
        d2 = np.sqrt((path[i, 0] - b[0]) ** 2 + (path[i, 1] - b[1]) ** 2
                     + (path[i, 2] - b[2]) ** 2)
        if d1 + d2 <= 0.0:
            continue
        r2 = lam * d1 * d2 / (d1 + d2)           # a zone no wider than the
        r = int(np.sqrt(r2))                     # cell (at a sound made IN a
        if r < 1:                                # hole) is all hole
            continue
        full = 0                                 # cells in the whole disc
        for u in range(-r, r + 1):
            full += 2 * int(np.sqrt(r2 - u * u)) + 1
        lo = max(i - 1, 0)
        hi = min(i + 1, n - 1)
        for k in range(3):
            # the way goes THROUGH here along k: it moves along k here, and
            # the cells either side along k are open
            if c[k] < 1 or c[k] >= dims[k] - 1 or path[lo, k] == path[hi, k]:
                continue
            q[0], q[1], q[2] = c[0], c[1], c[2]
            q[k] = c[k] - 1
            if not _cell_open(mat, smass, dvl, lim, q[0], q[1], q[2]):
                continue
            q[k] = c[k] + 1
            if not _cell_open(mat, smass, dvl, lim, q[0], q[1], q[2]):
                continue
            j1 = (k + 1) % 3
            j2 = (k + 2) % 3
            # and there is wall on both sides of it, across k (along
            # `held`, the farther side `w` cells off)
            held = -1
            w = 0
            for j in (j1, j2):
                sides = 0
                far = 0
                for s in (-1, 1):
                    q[0], q[1], q[2] = c[0], c[1], c[2]
                    for t in range(1, r + 1):
                        q[j] = c[j] + s * t
                        if q[j] < 0 or q[j] >= dims[j]:
                            break
                        if not _cell_open(mat, smass, dvl, lim,
                                          q[0], q[1], q[2]):
                            sides += 1
                            far = max(far, t)
                            break
                if sides == 2:
                    held = j
                    w = far
                    break
            if held < 0:
                continue
            opn = 0
            tot = 0
            wide = False
            q[0], q[1], q[2] = c[0], c[1], c[2]
            for u in range(-r, r + 1):
                q[j1] = c[j1] + u
                if q[j1] < 0 or q[j1] >= dims[j1]:
                    continue
                h = int(np.sqrt(r2 - u * u))
                for v in range(-h, h + 1):
                    q[j2] = c[j2] + v
                    if q[j2] < 0 or q[j2] >= dims[j2]:
                        continue
                    tot += 1
                    if _cell_open(mat, smass, dvl, lim, q[0], q[1], q[2]):
                        opn += 1
                if 3 * opn > full:               # a third open costs less
                    wide = True                  # than any bend: stop counting
                    break
            if wide or tot == 0:
                continue
            f = opn / tot
            if f >= least:
                continue
            # both mouths on the way, or it is a corridor someone is in
            mouths = 0
            for step in (-1, 1):
                m = i + step
                while 0 <= m < n:
                    for x in range(3):
                        e[x] = int(path[m, x])
                    if not _narrow(mat, smass, dvl, lim, e, held, w):
                        mouths += 1
                        break
                    m += step
            if mouths == 2:
                least = f
    return least


@njit(cache=True)
def _ray_opening(mat, smass, dvl, lim, a, b, lam):
    """`_opening` of the STRAIGHT line from `a` to `b`, sampled the way
    `_level` samples its ray, if all of it is open. A sound that goes
    straight through a hole is charged for the hole's size as much as one
    that has to go round to it; before, a 5 cm hole in a 5 cm wall, met
    head on, cost nothing at all. 1.0 if anything is in the way: then the
    wall is charged, and a hole off the line is found by the way round."""
    nx, ny, nz = mat.shape
    n = int(np.ceil(max(abs(b[0] - a[0]), abs(b[1] - a[1]),
                        abs(b[2] - a[2])))) + 1
    if n <= 2:
        return 1.0
    path = np.empty((n, 3), np.float64)
    for i in range(n):
        t = i / (n - 1.0)
        x = min(max(int(np.rint(a[0] + t * (b[0] - a[0]))), 0), nx - 1)
        y = min(max(int(np.rint(a[1] + t * (b[1] - a[1]))), 0), ny - 1)
        z = min(max(int(np.rint(a[2] + t * (b[2] - a[2]))), 0), nz - 1)
        if 0 < i < n - 1 and not _cell_open(mat, smass, dvl, lim, x, y, z):
            return 1.0
        path[i, 0], path[i, 1], path[i, 2] = x, y, z
    return _opening(mat, smass, dvl, lim, path, n, a, b, lam)


@njit(cache=True)
def _hear_openings(mat, smass, dvl, lim, A, B, lam, floor, hasfloor, level,
                   spread, need):
    """The hole a straight line may pass through, for every pair `_hear_levels`
    left with `need` 2 or 3: where the line is open all the way, the level is
    no more than the free level less the share of the sound the hole passes
    (`_level`, the same lines, pair by pair)."""
    for k in range(A.shape[0]):
        if need[k] < 2 or (hasfloor and not level[k] >= floor):
            continue
        share = _ray_opening(mat, smass, dvl, lim, A[k], B[k], lam)
        if share < 1.0:
            v = spread[k] + 10.0 * np.log10(max(share, 1e-12))
            if v < level[k]:
                level[k] = v


@njit(cache=True)
def _flood(open_, sx, sy, sz):
    """How many face steps each cell is from (sx, sy, sz), through cells
    sound can pass (`open_`); -1 where it cannot get. A plain breadth-first
    flood: each cell once, so a whole house costs a few milliseconds."""
    nx, ny, nz = open_.shape
    dist = np.full(open_.shape, -1, np.int32)
    q = np.empty(open_.size, np.int64)
    head = 0
    tail = 1
    dist[sx, sy, sz] = 0
    q[0] = (sx * ny + sy) * nz + sz
    while head < tail:
        c = q[head]
        head += 1
        z = c % nz
        y = (c // nz) % ny
        x = c // (ny * nz)
        d = dist[x, y, z] + 1
        if x > 0 and open_[x - 1, y, z] and dist[x - 1, y, z] < 0:
            dist[x - 1, y, z] = d
            q[tail] = c - ny * nz
            tail += 1
        if x < nx - 1 and open_[x + 1, y, z] and dist[x + 1, y, z] < 0:
            dist[x + 1, y, z] = d
            q[tail] = c + ny * nz
            tail += 1
        if y > 0 and open_[x, y - 1, z] and dist[x, y - 1, z] < 0:
            dist[x, y - 1, z] = d
            q[tail] = c - nz
            tail += 1
        if y < ny - 1 and open_[x, y + 1, z] and dist[x, y + 1, z] < 0:
            dist[x, y + 1, z] = d
            q[tail] = c + nz
            tail += 1
        if z > 0 and open_[x, y, z - 1] and dist[x, y, z - 1] < 0:
            dist[x, y, z - 1] = d
            q[tail] = c - 1
            tail += 1
        if z < nz - 1 and open_[x, y, z + 1] and dist[x, y, z + 1] < 0:
            dist[x, y, z + 1] = d
            q[tail] = c + 1
            tail += 1
    return dist


@njit(cache=True)
def _clear(open_, p, q):
    """Is the straight line from p to q all cells sound can pass? Sampled
    the way `_level` samples its ray, ends left out."""
    nx, ny, nz = open_.shape
    n = int(np.ceil(max(abs(q[0] - p[0]), abs(q[1] - p[1]),
                        abs(q[2] - p[2])))) + 1
    for i in range(1, n - 1):
        t = i / (n - 1.0)
        x = min(max(int(np.rint(p[0] + t * (q[0] - p[0]))), 0), nx - 1)
        y = min(max(int(np.rint(p[1] + t * (q[1] - p[1]))), 0), ny - 1)
        z = min(max(int(np.rint(p[2] + t * (q[2] - p[2]))), 0), nz - 1)
        if not open_[x, y, z]:
            return False
    return True


@njit(cache=True)
def _round_length(open_, dist, a, b, lam, mat, smass, dvl, lim):
    """The length, in voxels, of the shortest way from point `a` to point
    `b` through cells sound can pass, or -1 if there is none, and how open
    the tightest hole on it is (`_opening`; `lam` is the wavelength in
    voxels, and `mat`, `smass`, `dvl`, `lim` are what `_cell_open` reads).
    `dist` is the flood from `b`'s cell.

    The flood's own path is made of face steps, a staircase, and is longer
    than the real way. So the path is walked back from `a` (each step to
    the neighbour one nearer `b`, the one closest to `b` when several are),
    and then PULLED TIGHT like a string: it runs straight for as long as
    the straight line stays in the open, and turns only where it has to,
    at a corner. What is left is the way round a door jamb or over a wall."""
    nx, ny, nz = open_.shape
    sx = min(max(int(np.rint(a[0])), 0), nx - 1)
    sy = min(max(int(np.rint(a[1])), 0), ny - 1)
    sz = min(max(int(np.rint(a[2])), 0), nz - 1)
    if dist[sx, sy, sz] < 0:               # a sound made IN a solid (a pane
        best = -1                          # landing): leave from the nearest
        bx, by, bz = sx, sy, sz            # cell next to it that is open
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    x, y, z = sx + dx, sy + dy, sz + dz
                    if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                        d = dist[x, y, z]
                        if d >= 0 and (best < 0 or d < best):
                            best, bx, by, bz = d, x, y, z
        if best < 0:
            return -1.0, 1.0
        sx, sy, sz = bx, by, bz
    n = dist[sx, sy, sz] + 1
    path = np.empty((n, 3), np.float64)
    x, y, z = sx, sy, sz
    for i in range(n):
        path[i, 0], path[i, 1], path[i, 2] = x, y, z
        d = dist[x, y, z]
        if d == 0:
            break
        bx, by, bz = x, y, z
        best = 1e30
        for k in range(6):
            X = x + _FACES[k, 0]
            Y = y + _FACES[k, 1]
            Z = z + _FACES[k, 2]
            if 0 <= X < nx and 0 <= Y < ny and 0 <= Z < nz \
                    and dist[X, Y, Z] == d - 1:
                e = (X - b[0]) ** 2 + (Y - b[1]) ** 2 + (Z - b[2]) ** 2
                if e < best:
                    best, bx, by, bz = e, X, Y, Z
        x, y, z = bx, by, bz
    length = 0.0
    anchor = a.copy()
    last = a.copy()
    for i in range(n):
        c = path[i]
        if not _clear(open_, anchor, c):
            length += np.sqrt(((last - anchor) ** 2).sum())
            anchor = last
        last = c.copy()
    if not _clear(open_, anchor, b):
        length += np.sqrt(((last - anchor) ** 2).sum())
        anchor = last
    return (length + np.sqrt(((b - anchor) ** 2).sum()),
            _opening(mat, smass, dvl, lim, path, n, a, b, lam))


class SensesMixin:
    # SMOKE OPACITY (wave 13c, C4b). The gate. Sight reads the sim's smoke
    # scalar (self.smoke, g per voxel of "smoke units") as soot: K = sigma_e
    # * 0.025 g soot per unit * 1000 L/m3 = 217.5 1/m per g/L (smokeopt), 38.8
    # times the old 5.6 of materials.SMOKE_EXT. False = the old clear smoke.
    # Per world: set `w.smoke_opacity = False`. Cough and CO (physiology) keep
    # their own numbers; they are data hooks in smokeopt.Hooks, not this lane.
    smoke_opacity = True

    def _smoke_ext(self):
        """Extinction, 1/m per (g/L) of smoke unit, that sight uses."""
        return (smokeopt.extinction_from_smoke(1.0)
                if self.smoke_opacity else SMOKE_EXT)


    def _look_around(self, p, eye, cells):
        """Sweep the eyes and REMEMBER what they fell on.

        A body's map of the world is what it has actually looked at — nothing
        else. Rays go out through the facing cone and stop where the view
        stops, so a wall casts a shadow of ignorance behind it and a doorway
        lets knowledge through. Turning the head fills the map in over time,
        and walking somewhere fills it in properly.

        This is what makes the exit list honest. Before it, every body picked
        the nearest door from the world's own register and pathed to it across
        rooms it had never entered — the sim's knowledge wearing a person's
        face. Now a stranger has to find the door."""
        nx, ny, _ = self.shape
        vox_m = 0.1 * self.scale
        R = max(int(WILL["see_m"] / max(vox_m, 1e-9)), 4)
        z = int(np.clip(round(eye[2]), 0, self.shape[2] - 1))
        nrays = max(48, int(2.0 * np.pi * R / 1.5))
        ang = np.linspace(0.0, 2.0 * np.pi, nrays, endpoint=False)
        fx, fy = p.get("gaze", p.get("facing", (1.0, 0.0)))
        fn = float(np.hypot(fx, fy)) or 1.0
        inside = (np.cos(ang) * fx + np.sin(ang) * fy) / fn >= \
            np.cos(np.radians(WILL["fov_deg"] / 2.0))
        ang = ang[inside]
        if not len(ang):
            return
        # TWO WAYS TO RUN THE SAME RAYS. The plain one (numpy) builds every
        # sample of every ray, about 560 x 240 of them, and adds up the depth
        # along each. The fast one (`_look_rays`, numba) works out each
        # column's depth once and marches the rays one by one, stopping a ray
        # once nothing further along it can be seen. Both mark the same cells
        # (test_look_FAST_and_PLAIN_mark_the_same_cells); the plain one is the
        # definition, and `w.fused = False` runs it.
        fast = self.fused and HAVE_NUMBA
        if fast:
            ex, ey = float(eye[0]), float(eye[1])
            cs, sn = np.cos(ang), np.sin(ang)
            nlog = -np.log(np.maximum(_through(
                self.mat[:, :, z], self.smass[:, :, z], self.vox_l), 1e-9))
        else:
            steps = np.arange(1, R + 1, dtype=np.float32)
            xi = np.clip(np.round(eye[0] + np.cos(ang)[:, None] * steps[None]),
                         0, nx - 1).astype(np.int32)
            yi = np.clip(np.round(eye[1] + np.sin(ang)[:, None] * steps[None]),
                         0, ny - 1).astype(np.int32)
            through = _through(self.mat[xi, yi, z], self.smass[xi, yi, z],
                               self.vox_l)
            tau = np.cumsum(-np.log(np.maximum(through, 1e-9)), axis=1)
            # a cell is seen if the view REACHED it: the depth BEFORE it is clear
            reach = np.concatenate([np.zeros((len(ang), 1), np.float32),
                                    tau[:, :-1]], axis=1) < SEE_TAU
            p["known"][xi[reach], yi[reach]] = True
        p["known"][int(round(eye[0])), int(round(eye[1]))] = True
        # AND WHAT WAS THERE, not merely that there was a there. Belief recorded
        # geometry only: a body could remember the shape of a room it had walked
        # through and nothing whatever about the fire in it, so when it came to
        # choose somewhere to go, every place it knew looked equally good. That
        # is why an idle body would stroll toward a blaze — not because it was
        # brave, but because nothing it remembered said otherwise.
        #
        # Read by COLUMN rather than at eye height, because light from a fire at
        # your feet reaches your eyes: what the ray decides is whether you can
        # SEE that far, which is the occlusion the loop above already did.
        #
        # And it FADES. A body that never forgets treats an hour-old fire as a
        # fire; one that forgets at once has no memory to speak of.
        danger = p.get("danger")
        if danger is None:
            danger = p["danger"] = np.zeros(self.shape[:2], np.float32)
        #
        # WHAT IT SAW, AND NOT WHAT IS THERE. This used to run
        # `np.maximum(danger, alight)` over the WHOLE lattice one line before
        # the sight-gated write below, which quietly set every burning column
        # in the world to full danger whether or not this body had any way of
        # knowing. Measured: a witness sealed in a room at x10, behind solid
        # stone, carried danger 1.0 over a fire at x30 it could not possibly
        # see — and `_places` scores routes against exactly this map, so bodies
        # were routing around fires nobody had told them about. It is the same
        # god-channel the ablation took out of the percept, one layer down:
        # belief must come from the eyes or it is not belief.
        danger *= WILL["forget"]
        alight = self.burning().any(axis=2)
        # LIGHT GETS ITS OWN REACH, because `reach` above is a single
        # horizontal plane at EYE HEIGHT and a fire is on the floor. Smoke
        # fills a room from the ceiling down, so a witness standing in the
        # smoke layer had every ray stopped at the first cell — measured, 335
        # rays reaching 335 cells — while the bench burning at its feet was in
        # plain view under the smoke. The whole of `danger` was coming from the
        # global line above; delete that and a body in an open room with a fire
        # twelve cells away remembers nothing at all.
        #
        # So the occlusion for LIGHT is taken over the band from this body's
        # feet to its eyes, and a column counts as clear if ANY height in that
        # band is clear. That is what the comment above always claimed and only
        # half did. It sees under smoke, over a bench and through a doorway,
        # and not through a full-height wall, which is opaque at every height.
        # The band stops at the feet so that a body upstairs cannot see a fire
        # through its own floor, and at the eyes so it cannot see through the
        # ceiling either.
        fz0 = int(np.clip(cells[:, 2].min(), 0, self.shape[2] - 1))
        fz1 = int(np.clip(round(eye[2]) + 1, fz0 + 1, self.shape[2]))
        bmat = self.mat[:, :, fz0:fz1]
        bpack = np.clip(self.smass[:, :, fz0:fz1] / np.maximum(
            _DENS_ARR[bmat] * self.vox_l, 1e-9), 0.0, 0.999)
        clear = ((1.0 - bpack) + bpack * _TRANSMIT_ARR[bmat]).max(axis=2)
        # AND A BODY IS NOT A WALL TO ITSELF — the same rule belief already
        # keeps for the floor, and it has to be kept here or the looker is the
        # thing that blinds it. Taken over a band, its own torso is opaque at
        # every height in that band, so every ray died on the first cell and a
        # body could see nothing whatever. Its own flesh is denser than smoke.
        clear[cells[:, 0], cells[:, 1]] = 1.0
        if fast:
            nlog_band = -np.log(np.maximum(clear, 1e-9))
            reach = np.zeros((nx, ny), bool)
            lit = np.zeros((nx, ny), bool)
            # a ray may stop early only while no depth is negative (kernels.py)
            stop = bool(nlog.min() >= 0.0) and bool(nlog_band.min() >= 0.0)
            _look_rays(ex, ey, cs, sn, R, nlog, nlog_band, np.float32(SEE_TAU),
                       stop, reach, lit)
            p["known"][reach] = True
            danger[lit & alight] = 1.0
        else:
            ltau = np.cumsum(-np.log(np.maximum(clear[xi, yi], 1e-9)), axis=1)
            lit = np.concatenate([np.zeros((len(ang), 1), np.float32),
                                  ltau[:, :-1]], axis=1) < SEE_TAU
            danger[xi[lit], yi[lit]] = np.where(
                alight[xi[lit], yi[lit]], 1.0, danger[xi[lit], yi[lit]])
        # AND WHETHER IT COULD GET THERE. This is the same god-channel as the
        # door register, one layer down and much better hidden: belief gated
        # WHERE a body had looked, and then the planner read the floor itself,
        # live, every time. So a corridor blocked by a beam that fell behind a
        # body's back was routed around instantly, by a body that never turned
        # its head — and a wall breached out of sight became a way through the
        # moment it was breached. What a body can plan over is now what it SAW
        # it could plan over, and nothing else.
        #
        # It does not fade. A floor is not a fire: the useful error is a body
        # that trusts a way it can no longer take and finds out by going, not
        # one that forgets the shape of a room it was standing in a moment ago.
        walk_now = self._walkable(cells)
        free = p.get("free")
        if free is None:                  # built on the FIRST look, because a
            free = p["free"] = (          # body's height decides which columns
                walk_now.copy() if p.get("knew")   # it fits down, and there is
                else np.zeros(self.shape[:2], bool))  # no flesh yet at signup
        if fast:
            free[reach] = walk_now[reach]
        else:
            free[xi[reach], yi[reach]] = walk_now[xi[reach], yi[reach]]
        # A BODY IS NOT A WALL TO ITSELF — the same rule the planner already
        # keeps, and it has to be kept HERE too or belief is poisoned by the
        # looker. Its own flesh is denser than PUSH_THROUGH, so every look
        # wrote the ground under its own feet down as blocked, it walked on,
        # and nothing ever looked back to correct it. Measured before this
        # line: a body crossing an empty 4 m hall laid 109 phantom walls
        # behind itself and believed every one of them. What it carries goes
        # with it, for the same reason and by the same mistake.
        free[cells[:, 0], cells[:, 1]] = True
        if p.get("held"):
            obj = self._held_cells(p)
            if obj is not None and len(obj):
                free[obj[:, 0], obj[:, 1]] = True

    def _sees(self, a, b, own=None):
        """Sight as OPTICAL DEPTH along the ray, not a list of allowed materials.

        Every voxel on the way dims the view by how much stuff is in it: the
        material's own opacity scaled by how full the cell actually is, plus the
        smoke hanging in it. Beer-Lambert; you can see if enough light survives.

        What this buys over the old material whitelist: a GLASS window is a
        window (it was opaque before), a sparse hedge dims instead of walling
        off, a doorway half-choked with rubble is half-blind, and — the one that
        was simply missing — SMOKE BLINDS. You could previously spot a runner
        clean through a smoke bank. The old "bodies don't wall off sight" case
        is gone with it: a body between you and the fire really does block it,
        and your own is skipped by geometry (you don't see your own eyelashes)
        rather than by exempting the material.

        `own`, when given, is the looker's own flesh (cells): THE RAY STARTS
        IN THE HEAD. Without it every cell next to the eye is skipped, which
        let a body see through a wall its face was against; with it, only
        its own flesh is skipped (`rays` does the same with `skip`)."""
        a = np.asarray(a, np.float32)
        b = np.asarray(b, np.float32)
        n = int(np.ceil(np.abs(b - a).max())) + 1
        if n <= 2:
            return True
        ts = np.linspace(0.0, 1.0, n)[1:-1]
        pts = np.round(a[None] + ts[:, None] * (b - a)[None]).astype(np.int64)
        if own is None:
            near = (np.abs(pts - a[None]).max(axis=1) > 1.0)  # skip own eyelashes
        else:
            near = np.array([tuple(c) not in own for c in pts.tolist()], bool)
        pts = pts[near]
        if not len(pts):
            return True
        nx, ny, nz = self.shape
        pts[:, 0] = np.clip(pts[:, 0], 0, nx - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, ny - 1)
        pts[:, 2] = np.clip(pts[:, 2], 0, nz - 1)
        idx = tuple(pts.T)
        vox_m = 0.1 * self.scale
        packed = np.clip(self.smass[idx] / np.maximum(
            _DENS_ARR[self.mat[idx]] * self.vox_l, 1e-9), 0.0, 0.999)
        # what gets past one voxel: the (1-f) that misses the stuff entirely,
        # plus the f that hits it and comes through anyway if it is glass
        through = (1.0 - packed) + packed * _TRANSMIT_ARR[self.mat[idx]]
        tau = float(-np.log(np.maximum(through, 1e-9)).sum())
        tau += float((self._smoke_ext() * (self.smoke[idx] / self.vox_l) * vox_m).sum())
        return tau < SEE_TAU

    @staticmethod
    def _in_cone(facing, eye, target, deg=None):
        """Eyes point where the face points: the target must sit inside the
        field-of-view cone around `facing` (a 2D heading). The cone is the
        whole field (`fov_deg`), where a thing is DETECTED, unless `deg`
        names a narrower one (`_in_focus`)."""
        vx, vy = target[0] - eye[0], target[1] - eye[1]
        n = float(np.hypot(vx, vy))
        if n < 1.5:
            return True                              # at arm's reach you just know
        cosang = (facing[0] * vx + facing[1] * vy) / n
        return cosang >= np.cos(np.radians(
            (WILL["fov_deg"] if deg is None else deg) / 2.0))

    def _in_focus(self, facing, eye, target):
        """Is the target where it can be IDENTIFIED -- who it is, which
        thing it is -- and not only detected (rows P1, P2, P6)? In the
        middle of the view (`focus_deg`), or anywhere in it within
        `know_m`. A percept from the rest of the field says that something
        is there, and not what."""
        if not self._in_cone(facing, eye, target):
            return False
        n = float(np.hypot(target[0] - eye[0], target[1] - eye[1]))
        return n * 0.1 * self.scale <= WILL["know_m"] \
            or self._in_cone(facing, eye, target, WILL["focus_deg"])

    def _hears(self, a, b, db1m=SHOUT_DB):
        """Can a SHOUT at `a` be made out at `b`? `a` and `b` are where two
        heads are (their `_eye`), and the line is heard the way `_law_hear`
        hears every other sound: at the ear on the outside of the head, with
        neither speaker's nor listener's own body in the way (`_at_ear`).

        It used to ask `_level` straight from one eye to the other, and both
        eyes are the TOP CELL OF A HEAD: the ray ran out through the
        speaker's skull and in through the listener's, and the mass law
        charged 40-46 dB for the two of them. A shout was understood only
        within 2-3 m in an empty room (docs/realism-references.md, G1)."""
        own = set()
        for q in self.persons:
            if self._is_at(q, a) or self._is_at(q, b):
                own |= set(q.get("_own") or ())
        lvl = self._at_ear(a, b, db1m, own,
                           floor=max(HEAR_DB, self._ambient() + SOUND["snr_db"]))
        return self._understood(lvl)

    def _ambient(self):
        """The noise a line has to be heard over: a quiet room, unless the
        scene says otherwise (`w.ambient_db`, a crowd or a brawl)."""
        return float(getattr(self, "ambient_db", SOUND["ambient_db"]))

    def _understood(self, lvl):
        """Are words arriving at `lvl` made out? Loud enough to be words at
        all (HEAR_DB), and not drowned by the room: a shout is understood
        down to about -3 dB under the noise around it, and lost below that."""
        return bool(lvl >= HEAR_DB
                    and lvl - self._ambient() >= SOUND["snr_db"])

    @staticmethod
    def _is_at(q, pt):
        """Is `pt` this person's head: their `_eye`, or a cell of their body?"""
        e = q.get("_eye")
        if e is not None and max(abs(float(e[i]) - float(pt[i]))
                                 for i in range(3)) < 0.5:
            return True
        own = q.get("_own")
        return bool(own) and tuple(int(round(float(v))) for v in pt) in own

    def _ear(self, eye):
        """THE EAR IS ON THE OUTSIDE OF THE HEAD. `_eye` is the top cell of
        the flesh, so a ray to it ran up through the listener's own skull
        and the mass law charged him 46 dB for his own head: a pane
        shattering two metres away arrived at 53 dB."""
        return (eye[0], eye[1], min(eye[2] + 1.0, self.shape[2] - 1.0))

    def _at_ear(self, at, eye, db1m, own, floor=None):
        """THE ONE WAY A SOUND IS HEARD: how loud a sound made at `at` is at
        the ear of a head whose `_eye` is `eye`, with the cells in `own`
        (the bodies of whoever made it and whoever hears it) not counted as
        in the way. `_law_hear` and `_hears` both ask this, so a noise and
        a line are heard by the same law (roadmap 97 fixed the first and
        not the second)."""
        return self._level(at, self._ear(eye), db1m, own=own or None,
                           floor=floor)

    def _maker(self, snd):
        """The pid of the person who made a sound with a body (a footstep,
        a shout), or None. Asked BY PID: names repeat ("the person"), and a
        sound matched by name was deaf to everyone else of the same name.
        The pid is the actor of the event the sound is the noise of; a name
        is used only if the event is gone and the name is not shared.

        A sound made before its event exists (a blow, a landing: their
        event is queued and has no id until the tick ends) carries its
        maker's pid itself (`_noise(pid=)`), so that a blow is its
        striker's own noise even when another person has his name."""
        if snd.get("pid") is not None:
            return snd["pid"]
        if snd.get("who") is None:
            return None
        src = snd.get("ev_id", snd.get("src"))
        ev = self.events.get(src) if isinstance(src, int) else None
        if ev is not None and ev.actor is not None:
            return ev.actor
        named = [q["pid"] for q in self.persons if q["name"] == snd["who"]]
        return named[0] if len(named) == 1 else None

    def _law_hear(self):
        """Every noise of the last two ticks, heard by every ear it reaches.

        A sleeper wakes when a noise at the ear passes the arousal level; an
        awake body NOTICES one that is new and above the level of a quiet
        room, and remembers where it came from — which is all "I heard
        something over there" is. Nobody hears their own footsteps as news."""
        self.sounds = [snd for snd in self.sounds if snd["tick"] >= self.tick - 1]
        if not self.sounds or not self.persons:
            return
        makers = [self._maker(snd) for snd in self.sounds]
        by_pid = {q["pid"]: q for q in self.persons}
        # WHO HEARS WHAT, asked all at once. A level depends on the cells and
        # on where two heads are, and nothing `_heard` or the wake-up below
        # does changes either, so every (ear, sound) pair is worked out first,
        # in one call (`_levels`), and then read in the order it always was.
        ears = []
        for p in self.persons:
            if not p["alive"] or p["safe"]:
                continue
            ear = p.get("_eye")
            if ear is None:
                comp, sl = self._person_cells(p)
                if comp is None or not comp.any():
                    continue
                fw = self._claim_cells(p, comp, sl, shift=False)
                ear = (float(fw[:, 0].mean()) + (sl[0].start or 0),
                       float(fw[:, 1].mean()) + (sl[1].start or 0),
                       float(fw[:, 2].max()))
            ears.append((p, ear))
        pairs = [(p, ear, snd, src) for p, ear in ears
                 for snd, src in zip(self.sounds, makers)
                 if not (src is not None and src == p["pid"])]
        lvls = iter(self._hear_pairs(pairs, by_pid))
        for p, ear in ears:
            best = None
            for snd, src in zip(self.sounds, makers):
                if src is not None and src == p["pid"]:
                    continue
                lvl = next(lvls)
                self._heard(p, snd, lvl)              # a `heard` row (V3)
                if best is None or lvl > best[0]:
                    best = (lvl, snd)
            if best is None:
                continue
            lvl, snd = best
            if p.get("asleep"):
                if self.noise_wakes(p, lvl):      # D2: a chance, not a fixed line
                    p["asleep"] = False
                    self.emit("wakes", actor=p["pid"], db=float(lvl),
                              cause=self._src(snd),
                              text=f"{p['name']} wakes ({lvl:.0f} dB)",
                              to=p, site="hear:wakes")
                continue
            if lvl >= SOUND["notice_db"] and \
                    snd["tick"] > (p.get("noise") or {}).get("tick", -1):
                p["noise"] = {"tick": snd["tick"], "at": snd["at"], "db": lvl,
                              "ev_id": self._src(snd)}

    def _hear_pairs(self, pairs, by_pid):
        """The level at each ear of each sound in `pairs`, a list of (listener,
        ear, sound, maker's pid or None), in that order: what `_at_ear` gives
        one pair at a time, with neither the listener's body nor the maker's
        in the way. Done as one batch (`_levels`) when the compiled path is on,
        and pair by pair, the definition, when it is not."""
        if not pairs:
            return []
        nz1 = self.shape[2] - 1.0
        owns, rows = {}, []
        for p, ear, snd, src in pairs:
            q = by_pid.get(src) if src is not None else None
            owns[id(p)] = p
            if q is not None:
                owns[id(q)] = q
            rows.append((p, q))
        if not self._fused_hearing():
            out = []
            for (p, ear, snd, src), (_, q) in zip(pairs, rows):
                own = set(p.get("_own") or ())
                if q is not None:
                    own |= set(q.get("_own") or ())
                out.append(self._at_ear(snd["at"], ear, snd["db"], own,
                                        floor=SOUND["notice_db"]))
            return out
        # each body's own cells, as sorted flat indices
        ny, nz = self.shape[1], self.shape[2]
        index, keys, starts = {}, [], [0]
        for ident, p in owns.items():
            cells = [c for c in (p.get("_own") or ())
                     if 0 <= c[0] < self.shape[0] and 0 <= c[1] < ny
                     and 0 <= c[2] < nz]
            k = np.unique((np.array(cells, np.int64).reshape(-1, 3)
                           @ np.array([ny * nz, nz, 1], np.int64))) \
                if cells else np.zeros(0, np.int64)
            index[ident] = len(starts) - 1
            keys.append(k)
            starts.append(starts[-1] + len(k))
        okeys = np.concatenate(keys) if keys else np.zeros(0, np.int64)
        A = np.array([snd["at"] for _, _, snd, _ in pairs], np.float32)
        B = np.array([(e[0], e[1], min(e[2] + 1.0, nz1)) for _, e, _, _ in pairs],
                     np.float32)
        DB = np.array([snd["db"] for _, _, snd, _ in pairs], np.float64)
        PI = np.array([index[id(p)] for p, _ in rows], np.int64)
        QI = np.array([index[id(q)] if q is not None else -1 for _, q in rows],
                      np.int64)
        return self._levels(A, B, DB, PI, QI, okeys,
                            np.array(starts, np.int64), SOUND["notice_db"])

    def _fused_hearing(self):
        """Is the compiled batch to be used? It is on with the rest of the
        fused laws, and only where numba's logs and power round as numpy's."""
        return bool(self.fused and HAVE_NUMBA and exact_math())

    def _levels(self, A, B, DB, PI, QI, okeys, ostart, floor):
        """`_level` for a batch: pair k is a sound of `DB[k]` dB at a metre,
        made at `A[k]` and heard at `B[k]` (float32 points, the ear already
        out on the outside of the head), with the bodies of person `PI[k]` and
        `QI[k]` (-1 for none) not in the way (`okeys[ostart[i]:ostart[i+1]]`
        are the sorted flat cell indices of person i's body). The same
        numbers `_level` gives, to the last bit (tests/test_fused_paths.py).

        The straight line and its walls, bodies and leaves are one compiled
        call (`_hear_levels`); the hole in a line, a second (`_hear_openings`);
        and the way round, which needs the flood kept for an ear, is asked
        only of the pairs where it could matter, and the kept floods are
        checked against the cells once for the whole batch, not once a pair."""
        vox_m = 0.1 * self.scale
        K = len(A)
        DV = np.empty(K, np.float64)
        for k in range(K):                       # numpy's own norm, as `_level`
            DV[k] = float(np.linalg.norm(B[k] - A[k]))
        level = np.empty(K, np.float64)
        spread = np.empty(K, np.float64)
        need = np.zeros(K, np.int8)
        dvl = self._full_mass()
        lim = np.float32(SOUND["open_fill"])
        least = 10.0 * np.log10(3.0)
        _hear_levels(self.mat, self.smass, dvl, A, B, DB, DV, PI, QI, okeys,
                     ostart, vox_m, SOUND["shadow_db"], SOUND["bend_cap_db"],
                     SOUND["leaf_db_m"], float(floor), True, least,
                     AIR, FLESH, LEAF, level, spread, need)
        _hear_openings(self.mat, self.smass, dvl, lim, A.astype(np.float64),
                       B.astype(np.float64),
                       SOUND["wave_m"] / max(vox_m, 1e-9), float(floor), True,
                       level, spread, need)
        state = None
        for k in np.flatnonzero(need == 3):
            if state is None:
                state = self._round_state()
            around = self._round_level(A[k], B[k], float(DB[k]), float(DV[k]),
                                       state)
            if around is not None and around > level[k]:
                level[k] = around
        return level.tolist()

    def _body_pid(self, b):
        """The pid of the person a free body is, or is part of (a swung arm,
        a jump, a fall), or None. `owner` is a NAME, and names repeat ("the
        person"), so the name only narrows it down:

        - the eid the flesh carries, where people's flesh is stamped
          (`stamp_bodies`), says whose it is outright;
        - else, if one person has the name, it is theirs;
        - else, of those who have it, the one whose own flesh is nearest
          where the body left the lattice (`cells`): an arm leaves from
          its owner's shoulder, a faller from where he stood. A tie is
          nobody's."""
        if b.get("owner") is None:
            return None
        eids, mats = b.get("eids"), b.get("mats")
        if eids is not None and mats is not None and len(eids):
            mine = eids[(mats == FLESH) & (eids > 0)]
            if len(mine):
                by_eid = {q["eid"]: q["pid"] for q in self.persons
                          if q.get("eid")}
                vals, counts = np.unique(mine, return_counts=True)
                for e in vals[np.argsort(-counts, kind="stable")]:
                    if int(e) in by_eid:                 # most of it is his
                        return by_eid[int(e)]
        named = [q for q in self.persons if q["name"] == b["owner"]]
        if len(named) <= 1:
            return named[0]["pid"] if named else None
        left = np.asarray(b.get("cells", ()), np.float64).reshape(-1, 3)
        if not len(left):                        # nowhere it left from: a tie
            return None
        near = []
        for q in named:
            own = q.get("_own")
            at = np.array(list(own), np.float64) if own else (
                np.array([[*q["anchor"], left[:, 2].mean()]], np.float64)
                if q.get("anchor") is not None else None)
            if at is None:
                continue
            gap = min(float(np.abs(at - c[None]).max(axis=1).min())
                      for c in left)
            near.append((gap, q["pid"]))
        near.sort()
        if not near or (len(near) > 1 and near[1][0] == near[0][0]):
            return None
        return near[0][1]

    def _noise(self, at, joules, mats, who=None, ev=None, pid=None):
        """Something struck something: a sound, at `at`, this loud at 1 m.

        `ev` is the event this is the noise of (V3), so whoever hears it can
        be told what caused it. A caller that has the event in hand passes
        it (a footstep, `_law_will`). The physical laws make their sound a
        line BEFORE they queue their `hits` or `lands`, which has no id until
        the tick ends; their sound gets a box, and `_unbox` matches it to
        that event by tick and place. `pid` is who made it, when the box
        cannot say yet (`_maker`)."""
        if joules <= 0.0 or not len(mats):
            return None
        ring = min(SOUND["ring"].get(int(m), 1e-7) for m in mats)
        watts = ring * float(joules) / SOUND["impact_s"]
        if watts <= 1e-12:
            return None
        db = 10.0 * np.log10(watts / 1e-12) - 8.0
        snd = {"tick": self.tick, "at": tuple(float(v) for v in at),
               "db": float(db), "who": who}
        if pid is not None:
            snd["pid"] = pid
        if ev is None:
            ev = _Pending()
            self._sound_boxes.append((ev, self.tick, snd["at"]))
            self._boxed = True
        snd["src"] = ev
        self.sounds.append(snd)
        return snd

    def _level(self, a, b, db1m, own=None, floor=None):
        """Hearing in DECIBELS: how loud a sound that is `db1m` at a metre
        from `a` is at `b`. The cells in `own` are not in the way.

        Distance spends it by spherical spreading (6 dB per doubling). Then
        whatever is on the straight line between them, each by what it is:

        - A WALL by the acoustic MASS LAW: its mass per area, read off
          `smass`, so derived, not declared. 5 cm of masonry is ~135 kg/m2
          and eats ~50 dB, the same thickness of pine ~30 kg/m2 and ~37 dB.
          The mass is summed over the whole path and the law applied ONCE --
          a wall's loss comes from its total mass, not per slice, or six
          voxels of stone would silence a thunderclap.
        - THE GAPS IN A WALL by the energy they let through, not by the
          mass they leave out. A partly filled cell is mostly stuff and
          partly a hole (`fill`), and a hole passes sound whole: a door 97%
          full is a door with a 3% gap, and costs ~15 dB, not the 36 dB its
          mass would say (composite transmission loss, row A12).
        - A BODY is a shadow, ~3 dB each, not 46 dB of masonry (row A14).
        - LEAVES are charged by the ISO foliage rule: nothing under 10 m of
          them (row A15). A hedge hides you; it does not silence you.

        - A HOLE the line goes straight through (a peephole, a slit, a
          knot hole in a plank door) by its size: it passes its share of
          the sound and no more (`_ray_opening`, row A12), the same charge
          as when the sound has to go round to it. Before, a 5 cm hole met
          head on cost nothing in a 5 cm wall and 30 dB in a 20 cm one.

        And sound BENDS. If a wall is in the way, it can also go round --
        through a doorway, over a partition, under a door -- and the louder
        way wins. The way round is the shortest path through cells it can
        pass (`_round_length`), charged its extra length and the ISO
        9613-2 bending loss, never more than 20 dB, or the size of the hole
        it goes through if that costs more (`_round_level`). Before, an
        open doorway a metre round a corner cost 55.6 dB; a real one costs
        10-17.

        `floor` is the level below which the caller does not care: the way
        round, which costs a flood of the house, is looked for only when it
        could come out above it. Ears have no cone."""
        a = np.asarray(a, np.float32)
        b = np.asarray(b, np.float32)
        vox_m = 0.1 * self.scale
        d_vox = float(np.linalg.norm(b - a))
        spread = db1m - 20.0 * np.log10(max(d_vox * vox_m, 1.0))
        n = int(np.ceil(np.abs(b - a).max())) + 1
        if n <= 2:
            return float(spread)
        ts = np.linspace(0.0, 1.0, n)[1:-1]
        pts = np.round(a[None] + ts[:, None] * (b - a)[None]).astype(np.int64)
        nx, ny, nz = self.shape
        pts[:, 0] = np.clip(pts[:, 0], 0, nx - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, ny - 1)
        pts[:, 2] = np.clip(pts[:, 2], 0, nz - 1)
        if own:                                          # neither end's own
            keep = np.array([tuple(c) not in own        # body is a wall
                             for c in pts.tolist()], bool)  # between them
            pts = pts[keep]
        if not len(pts):
            return float(spread)
        idx = tuple(pts.T)
        mat = self.mat[idx]
        mass = self.smass[idx]
        fill = np.clip(mass / np.maximum(_DENS_ARR[mat] * self.vox_l, 1e-9),
                       0.0, 1.0)
        body = (mat == FLESH) & (fill > 0.0)
        leaf = (mat == LEAF) & (fill > 0.0)
        wall = (mat != AIR) & (mat != FLESH) & (mat != LEAF) & (fill > 0.0)
        loss = 0.0
        if wall.any():
            sigma = float(mass[wall].sum()) / 1000.0 / max(vox_m * vox_m, 1e-9)
            tl = max(0.0, 20.0 * np.log10(max(sigma, 1e-12) * 500.0) - 47.0)
            # what gets through the gaps: each run of wall cells along the
            # ray is one barrier, as open as its least open cell (a hole has
            # to go all the way through); barriers one behind another pass
            # the product of their openings
            run = np.cumsum(np.diff(np.concatenate([[0], wall.astype(np.int8)]))
                            == 1)
            gap = 1.0 - fill
            gap[gap < 1e-4] = 0.0                        # full is full
            leak = 1.0
            for r in np.unique(run[wall]):
                leak *= float(gap[wall & (run == r)].min())
            loss = -10.0 * np.log10(min(leak + 10.0 ** (-tl / 10.0), 1.0))
        if body.any():                                   # one shadow a body
            bodies = int((np.diff(np.concatenate([[0], body.astype(np.int8)]))
                          == 1).sum())
            loss += min(SOUND["shadow_db"] * bodies, SOUND["bend_cap_db"])
        if leaf.any():                                   # ISO 9613-2 foliage
            leaf_m = float(fill[leaf].sum()) * vox_m
            if leaf_m >= 20.0:
                loss += SOUND["leaf_db_m"] * min(leaf_m, 200.0)
            elif leaf_m >= 10.0:
                loss += 1.0
        level = spread - loss
        # THROUGH A HOLE. A line that is open all the way may still pass
        # through a hole in a wall, and a hole passes its share of the sound
        # and no more (`_ray_opening`): the same charge the way round makes.
        # (Open as the way round counts it: a cell less than half full is
        # charged above for its gap, and is a hole here.)
        if floor is None or level >= floor:
            share = float(_ray_opening(
                self.mat, self.smass, self._full_mass(),
                np.float32(SOUND["open_fill"]), a.astype(np.float64),
                b.astype(np.float64), SOUND["wave_m"] / max(vox_m, 1e-9)))
            if share < 1.0:
                level = min(level, spread + 10.0 * np.log10(max(share, 1e-12)))
        # THE WAY ROUND. It can only win where a wall costs more than the
        # least bending ever costs (10 log10 3, a path that only grazes an
        # edge), and it is worth a flood only if it could matter to the
        # caller.
        least = 10.0 * np.log10(3.0)
        if loss - least > 1e-6 and (floor is None or spread - least >= floor):
            around = self._round_level(a, b, db1m, d_vox)
            if around is not None and around > level:
                level = around
        return float(level)

    def _full_mass(self):
        """Each material's mass when it fills a voxel, as `_cell_open`
        reads it (float32, like `smass`)."""
        return np.maximum(_DENS_ARR * self.vox_l, 1e-9)

    def _round_state(self):
        """The cells sound can pass and the floods kept from them, checked
        against the cells as they are NOW (see `_ROUND`), with the full mass
        of each material and the fill below which a cell is open. A batch
        asks this once and hands it to each `_round_level`."""
        got = _ROUND.get(self)
        dvl = self._full_mass()
        lim = np.float32(SOUND["open_fill"])
        if got is None or not _same_open(self.mat, self.smass, dvl, lim,
                                         got[0]):
            got = _ROUND[self] = (_open_mask(self.mat, self.smass, dvl, lim),
                                  {})
        return got[0], got[1], dvl, lim

    def _round_level(self, a, b, db1m, d_vox, state=None):
        """The level at `b` of the sound from `a` by the shortest way round
        through the open, or None if there is no way round. The flood is
        from `b`'s cell (an ear, so asked again and again) and kept while
        the cells stay the same (`_ROUND`).

        The way round is charged two things, and the dearer one counts:
        BENDING (its extra length and the ISO 9613-2 edge loss), and the
        SIZE OF THE HOLE it goes through (`_opening`). Bending alone did
        not know how big a hole was: one missing 5 cm voxel in a sealed
        wall let a sound through only a few dB below an open doorway. A
        hole passes its share of the sound and no more, whatever way it is
        reached, so the way round is never louder than the free level less
        that share (composite transmission loss, row A12): about 30 dB for
        one voxel's hole. A doorway is most of its zone, and a gap under a
        door is charged more by bending than by its size, so both come out
        as they did."""
        open_, floods, dvl, lim = state or self._round_state()
        nx, ny, nz = self.shape
        cell = (min(max(int(np.rint(b[0])), 0), nx - 1),
                min(max(int(np.rint(b[1])), 0), ny - 1),
                min(max(int(np.rint(b[2])), 0), nz - 1))
        dist = floods.pop(cell, None)
        if dist is None:
            dist = _flood(open_, *cell)
            keep = max(1, min(_FLOODS_KEPT, _FLOOD_BYTES // dist.nbytes))
            while len(floods) >= keep:
                del floods[next(iter(floods))]          # the least recent
        floods[cell] = dist                             # the most recent
        vox_m = 0.1 * self.scale
        length, share = _round_length(
            open_, dist, np.asarray(a, np.float64), np.asarray(b, np.float64),
            SOUND["wave_m"] / max(vox_m, 1e-9), self.mat, self.smass, dvl,
            lim)
        length = float(length)
        if length < 0.0:
            return None
        extra = max(length - d_vox, 0.0) * vox_m
        bend = min(10.0 * np.log10(3.0 + 20.0 * extra / SOUND["wave_m"]),
                   SOUND["bend_cap_db"])
        level = db1m - 20.0 * np.log10(max(length * vox_m, 1.0)) - bend
        if share < 1.0:                                 # through a hole
            free = db1m - 20.0 * np.log10(max(d_vox * vox_m, 1.0))
            level = min(level, free + 10.0 * np.log10(max(float(share), 1e-12)))
        return float(level)

    def _lombard(self, voice):
        """THE LOMBARD REFLEX: how many dB a speaker who means to speak at
        `voice` (a name in `policy.VOICE`) lifts it by, in the noise around
        them. Above `from_db` of background the voice rises `slope` dB for
        every dB of noise, and no further than a throat goes (`max_db`). In a
        quiet room it adds nothing. A WHISPER is the one voice the speaker
        chose to keep down: the chosen act wins over the reflex (docs/
        realism-map.md, the one contract), so it is not lifted, and in a loud
        room it is lost, which is why people lean in."""
        db = VOICE_DB[voice]
        if voice == "whisper":
            return 0.0
        lift = LOMBARD["slope"] * max(self._ambient() - LOMBARD["from_db"], 0.0)
        return min(db + lift, max(LOMBARD["max_db"], db)) - db

    def _say(self, p, response, voice=None):
        """A line is spoken, at a loudness: `voice` is one of the names in
        `policy.VOICE` (whisper, say quietly, say, call out, shout), by default
        the line's own (`LINE_VOICE`: an alarm is shouted, a reply is called,
        anything else is said). The level is a sound of that many dB at a
        metre from the mouth, raised by the Lombard reflex in a noisy room, and
        the sound law (`_law_hear`, `_hears`) decides who hears it. The lift
        is written into the `says` event, so a mind can perceive that its body
        raised its voice."""
        text = p.get("lines", LINES).get(response, LINES.get(response))
        if text:
            voice = voice if voice in VOICE_DB \
                else LINE_VOICE.get(response, "say")
            lift = self._lombard(voice)
            db = VOICE_DB[voice] + lift
            self.speech.append((self.tick, p["name"], text))
            ev = self.emit("says", actor=p["pid"], result=response, db=db,
                           data=(("lombard_db", lift),) if lift > 0.0 else (),
                           text=f"{p['name']} {VOICE_VERB[voice]}: \"{text}\"",
                           to=p, site="say:shouts")
            # WHAT was said, not just that something was. A hearer needs the
            # line to know what the noise meant to it.
            p["_shouted"] = (self.tick, response, db)   # and how loud
            self.minds[p["pid"]].said_ev = ev.id   # and which event it was
            eye = p.get("_eye")
            if eye is not None:                  # and a line is a NOISE, which
                self.sounds.append({"tick": self.tick, "at": tuple(eye),
                                    "db": db, "who": p["name"],
                                    "ev_id": ev.id})  # wakes

    def _in_flight_near(self, p, own):
        """Things in the air that a hand of this body could close on now.

        A catch is a grab at something that will not wait. So the test is the
        same reach a grab uses, asked of a body that has left the grid — which
        is why this needed nothing new once things could be thrown: a flier
        already has a position and a velocity, and a fist already has a place."""
        out = []
        if not self.bodies:
            return out
        reach = self._reach_of(p) / max(0.1 * self.scale, 1e-9)
        fists = np.asarray(self._fists(p, own), np.float64)
        for b in self.bodies:
            # A THING, NOT A BODY. An owned body is somebody's flesh — a person
            # falling, or a person hanging from your own fist — and neither of
            # those is a thing thrown at you. Without this, holding a man over
            # a drop made you perceive him as a missile every tick, which
            # crowded out every other percept you might have had about the
            # situation you were actually in. Catching a falling PERSON is a
            # real act and a different one (item 56).
            if not b.get("fly") or b.get("part") or b.get("owner") is not None:
                continue
            at = self._fly_pose(b)
            d = np.abs(at[None, :, :] - fists[:, None, :]).max(axis=2)
            if float(d.min()) <= reach:
                out.append(b)
        return out

    def _falling_near(self, p, own):
        """PEOPLE in the air a hand of this body could close on now.

        The twin of `_in_flight_near`, and separate from it on purpose: what
        you do about a falling friend is not what you do about a thrown stone,
        so they are two percepts and two rows. The arithmetic underneath is the
        same — a body in the air has a position and a velocity, a fist has a
        place, and `m v / t` says whether an arm can stop it."""
        out = []
        if not self.bodies:
            return out
        reach = self._reach_of(p) / max(0.1 * self.scale, 1e-9)
        fists = np.asarray(self._fists(p, own), np.float64)
        for b in self.bodies:
            who = b.get("owner")
            if not b.get("fly") or b.get("part") or who is None \
                    or who == p["name"]:
                continue
            at = self._fly_pose(b)
            d = np.abs(at[None, :, :] - fists[:, None, :]).max(axis=2)
            if float(d.min()) <= reach:
                out.append(b)
        return out
