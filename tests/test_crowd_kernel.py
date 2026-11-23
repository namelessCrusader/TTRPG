"""S4-3c crowd kernel: the blocked-step candidate scan in one call.

`routing.step_clear` (numba, with the plain python of the same source as the
definition) replaces the per-candidate numpy loop in `pathlaw._step_to_k`.
The equivalence test keeps the OLD method (copied from base25, below) and
compares (dz, k) and the whole-world hash after the move, bit for bit, over
random rooms, bodies, directions and ranges. MUTATION: reverse the dz order
(down before up) in the new method and the test fails (checked by hand, see
test_mutation_order_matters).
"""
import hashlib

import numpy as np
import pytest

from src.voxel import routing, scenes
from src.voxel.materials import AIR, BODY, PUSH_THROUGH, STONE, WOOD, _DENS_ARR
from src.voxel.pathlaw import PathingMixin, WADE_FRAC
from src.voxel.sim import World


class W(World):
    step_up_m = 0.30
    step_down_m = 0.30


class OldW(W):
    # base25's _step_to_k, verbatim
    def _step_to_k(self, cells, sx, sy, p=None, kmax=1):
        """Translate a body one cell sideways AND up or down by whatever rise
        its legs can manage, shoving aside what is light enough to shove, by
        the same exchange `_walk` uses. Returns dz in cells (0 for a flat step)
        or None when no step in range works.

        The arriving cells at the new height must be clear (packed under
        PUSH_THROUGH, no real puddle, nothing the solver holds) EXCLUDING the
        body's own; everything above the soles for the whole body is among the
        arriving cells, so a low ceiling refuses a step up; and the soles must
        be supported (the cell under them is not air and not the body's own).
        The smallest |dz| that works is taken, up before down. A drop beyond
        R_dn is not offered: the legs do not step off a lip, the footing law
        takes a body that walks off one.

        WITH `kmax` > 1 (the legs: `_stride`) a step that RISES or DROPS may
        reach k = 1..kmax cells along (sx, sy), and it must leave the body
        BALANCED: its centre of mass over the base its soles make
        (`_overbalanced`, the footing law's own question, asked of the place).
        A body is one rigid piece here, so a 15 cm kerb cannot be mounted by a
        one-cell shuffle that leaves the body's weight behind the edge, where
        the footing law would topple it; a real stride puts the foot up and
        the weight over it. The flat step is never reached over more than one
        cell and is not balance-tested (it never was). Returns (dz, k) or None."""
        up, dn = self._step_range(p)
        cells = np.asarray(cells, np.int64)
        nx, ny, nz = self.shape
        base = cells.copy()
        base[:, 0] += sx
        base[:, 1] += sy
        if (base[:, 0] < 0).any() or (base[:, 0] >= nx).any() \
                or (base[:, 1] < 0).any() or (base[:, 1] >= ny).any():
            return None
        own = set(map(tuple, cells.tolist()))
        if self.moving[tuple(cells.T)].any():
            return None
        order = [(0, 1)]
        for r in range(1, max(up, dn) + 1):
            for dz in ((r,) if r <= up else ()) + ((-r,) if r <= dn else ()):
                order += [(dz, k) for k in range(1, kmax + 1)]
        ghost = None
        own_lin = None
        blk = None
        for dz, k in order:
            if k > 1:
                base = cells.copy()
                base[:, 0] += sx * k
                base[:, 1] += sy * k
                if (base[:, 0] < 0).any() or (base[:, 0] >= nx).any() \
                        or (base[:, 1] < 0).any() or (base[:, 1] >= ny).any():
                    continue
            else:
                base = cells.copy()
                base[:, 0] += sx
                base[:, 1] += sy
            tgt = base.copy()
            tgt[:, 2] += dz
            if (tgt[:, 2] < 0).any() or (tgt[:, 2] >= nz).any():
                continue
            sh = self.shape
            lin = (tgt[:, 0] * sh[1] + tgt[:, 1]) * sh[2] + tgt[:, 2]
            if own_lin is None:
                own_lin = (cells[:, 0] * sh[1] + cells[:, 1]) * sh[2] + cells[:, 2]
            fresh = ~np.isin(lin, own_lin)
            ent_a = tgt[fresh]
            ent_t = tuple(ent_a.T)
            if len(ent_a):
                if blk is None:
                    # the cells a step could enter (the body's box grown by
                    # the longest reach and the rise), judged once: moving,
                    # packed past the push, or a real puddle
                    ex = kmax * abs(sx)
                    ey = kmax * abs(sy)
                    rz = max(up, dn)
                    lo = np.maximum(cells.min(axis=0) - (ex, ey, rz), 0)
                    hi = np.minimum(cells.max(axis=0) + (ex, ey, rz) + 1,
                                    (nx, ny, nz))
                    bx = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
                    packed = self.smass[bx] / np.maximum(
                        _DENS_ARR[self.mat[bx]] * self.vox_l, 1e-9)
                    blk = (self.moving[bx] | (packed >= PUSH_THROUGH)
                           | (self.fvol[bx] > WADE_FRAC * self.cap))
                bad_ = blk[tuple((ent_a - lo).T)].any()
                if bad_:
                    continue
            zmin = int(tgt[:, 2].min())
            if zmin > 0:
                low = tgt[tgt[:, 2] == zmin]
                bz = (low[:, 2] - 1,)
                below = (low[:, 0], low[:, 1], bz[0])
                lo_lin = (low[:, 0] * sh[1] + low[:, 1]) * sh[2] + bz[0]
                if not ((self.mat[below] != AIR)
                        & ~np.isin(lo_lin, own_lin)).any():
                    continue
            if dz and kmax > 1 and p is not None:
                if ghost is None:
                    ghost = dict(p, held=None, dragging=None, _v=None,
                                 _ghost_from=cells,
                                 _ghost_kg=self.smass[tuple(cells.T)].copy())
                if self._overbalanced(ghost, tgt) is not None:
                    continue
            new = set(map(tuple, tgt.tolist()))
            entered = ent_a
            vacated = [c for c in map(tuple, cells.tolist()) if c not in new]
            ent = ent_t
            vac = tuple(np.array(vacated, np.int64).reshape(-1, 3).T)
            body = self._lift(tuple(cells.T))
            shoved = self._lift(ent)
            self._put(tuple(tgt.T), body)
            self._put(vac, shoved)
            self._torque_solid = None
            return int(dz), int(k)
        return None


def _room(seed):
    r = np.random.default_rng(seed)
    w = W(30, 24, 40, voxel_cm=5)
    w.fill(0, 30, 0, 24, 0, 1, STONE)
    for _ in range(int(r.integers(0, 3))):       # blocks and walls
        x0, y0 = int(r.integers(18, 26)), int(r.integers(2, 20))
        dx, dy, h = int(r.integers(1, 6)), int(r.integers(1, 6)), int(r.integers(1, 8))
        w.fill(x0, min(x0 + dx, 30), y0, min(y0 + dy, 24), 1, 1 + h,
               STONE if r.random() < .7 else WOOD)
    if r.random() < .6:                          # a kerb (or a drop) across the way
        xk, h = int(r.integers(12, 16)), int(r.integers(1, 5))
        w.fill(xk, 30, 0, 24, 1, 1 + h, STONE)
        if r.random() < .3:
            w.fill(xk + 4, xk + 6, 0, 24, 0, 1, AIR)
    scenes._person(w, 6, 10)
    w.step()
    return w


def _hash(w):
    h = hashlib.sha1()
    for a in (w.mat, w.smass, w.moving):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _body(w):
    p = w.persons[0]
    comp, sl = w._person_cells(p)
    return p, w._claim_cells(p, comp, sl)


def test_kernel_equals_old_scan_bit_for_bit_over_random_worlds():
    n_step = nrise = 0
    for seed in range(300):
        r = np.random.default_rng(1000 + seed)
        a, b = _room(seed), _room(seed)
        pa, ca = _body(a)
        pb, cb = _body(b)
        assert np.array_equal(ca, cb)
        for _ in range(14):
            sx, sy = (1, 0) if r.random() < .7 else [int(v) for v in r.integers(-1, 2, 2)]
            if not (sx or sy):
                continue
            kmax = int(r.integers(1, 4))
            ra = OldW._step_to_k(a, ca, sx, sy, pa, kmax)
            rb = b._step_to_k(cb, sx, sy, pb, kmax)
            assert ra == rb, (seed, sx, sy, kmax)
            assert _hash(a) == _hash(b)
            if ra is not None:
                n_step += 1
                nrise += ra[0] != 0
                d = np.array([sx * ra[1], sy * ra[1], ra[0]], np.int64)
                ca, cb = ca + d, cb + d
    assert n_step > 100 and nrise > 30


def test_kernel_numba_equals_plain_python(monkeypatch):
    """Both paths of routing.step_clear (compiled, and the same source run as
    plain python) give the same mask on every call the step makes."""
    real = routing.step_clear
    calls = []

    def both(*a, **k):
        x = real(*a, numba=True)
        y = real(*a, numba=False)
        assert np.array_equal(x, y)
        calls.append(int(x.sum()))
        return x
    monkeypatch.setattr(W, "_step_clear", staticmethod(both))
    for seed in range(60):
        w = _room(seed)
        p, c = _body(w)
        w._step_to_k(c, 1, 0, p, 3)
    assert len(calls) >= 60 and max(calls) > 0


def test_footprint_cache_gives_the_same_arrays():
    for seed in range(20):
        w = _room(seed)
        _, c = _body(w)
        routing._FOOT.clear()
        f1 = routing.footprint(c)
        f2 = routing.footprint(c + (3, 2, 1))        # same shape, other place: a hit
        cx = int(round(float(c[:, 0].mean())))
        cy = int(round(float(c[:, 1].mean())))
        offs = sorted({(int(x) - cx, int(y) - cy) for x, y in c[:, :2]} - {(0, 0)})
        assert f1[0].tolist() == [o[0] for o in offs]
        assert f1[1].tolist() == [o[1] for o in offs]
        assert f2[0] is f1[0] and f2[1] is f1[1]


def test_mutation_reversed_order_fails_the_equivalence():
    """MUTATION: try the candidates in the reverse order (longest reach and
    down first). The new method then disagrees with the old scan in some room,
    so the equivalence test above would fail. (An up and a down step are never
    both clear in a real column, so the reversal is what shows the order.)"""
    from src.voxel import pathlaw
    real = pathlaw._candidates
    pathlaw._candidates = lambda u, d, k: real(u, d, k)[::-1]
    differ = 0
    try:
        for seed in range(300):
            a, b = _room(seed), _room(seed)
            pa, ca = _body(a)
            pb, cb = _body(b)
            for _ in range(14):
                ra = OldW._step_to_k(a, ca, 1, 0, pa, 3)
                rb = b._step_to_k(cb, 1, 0, pb, 3)
                differ += ra != rb
                if ra is None or ra != rb:
                    break
                d = np.array([ra[1], 0, ra[0]], np.int64)
                ca, cb = ca + d, cb + d
    finally:
        pathlaw._candidates = real
    assert differ > 0
