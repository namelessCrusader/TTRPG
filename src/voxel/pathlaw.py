"""PathingMixin: routes over levels, and the step that goes with them (T11 6.5).

A mixin for World that REPLACES `_fit_grid` and `_routes` (and adds
`_step_to`), written on World's public methods only, so phase B changes the
class list and nothing else:

    class World(..., MotionMixin, ...):  ->  class World(PathingMixin, ..., MotionMixin, ...)

WHAT IT DOES

  * The STEP RANGE is 0.30 m up and down since C9 (wave 13c: class
    attributes `step_up_m`, `step_down_m` below; a person's own entry wins). With
    the range at 0 it is the old behaviour. `_fit_grid`
    calls super(); `_routes` gives exactly the same paths as the base class
    (tie for tie) by the compiled sweep in routing.py, or by super() itself
    when `fast_sweep` is False. `_walk` and `_walkable` are NOT overridden.
  * With a range above 0 (R cells = round(step_m / d), d = 0.1 x scale metres)
    the planner runs over LEVELS: nodes (level, x, y), an edge when the next
    column has a level within [-R_dn, +R_up] and the whole footprint fits
    there. A body plans over what it BELIEVES: `p["fz"]`, an int16 plane
    [K, nx, ny] of the standing z it last SAW per level, written by
    `_see_levels` (phase B calls that from the look). A body with no `fz` reads
    the world (the honest default for a bare route query), masked by `known`
    and `free`.
  * `_step_to(cells, sx, sy)` is the leg's step with a rise: it finds the dz in
    range whose arriving cells are clear and whose soles are supported, and
    moves the body by the same exchange `_walk` uses (the cells it enters go
    to the cells it leaves). It returns the dz, or None.
  * `step_along(p, cells, sx, sy)` is `_step_to` plus the bookkeeping a stride
    owes (time from the table, metabolic work, the heat of a descent). In
    phase B `_stride` does the same inline; here it lets a test, or a scene,
    walk a flight and close the books.

A drop larger than R_dn is not an edge and not a step: a body that walks off a
lip falls by the footing and landing laws, as before.
"""
import numpy as np

from . import routing, stairs
from .materials import AIR, BODY, PUSH_THROUGH, _DENS_ARR

# step cost tables by (voxel size in mm, range): one per voxel size, built once
_FTAB = {}
# the depth of liquid a step will not wade, as a share of a cell: the SAME
# number `_walk` keeps (motion.WADE_CELLS), so a flat step is refused or allowed
# by one rule whichever leg takes it.
WADE_FRAC = 0.5


class PathingMixin:
    # The world-wide defaults of the step range (m). None = read BODY, whose
    # entries phase B adds; a person's own p["step_up_m"] wins over both.
    # C9 (wave 13c): THE STEP RANGE IS ON. 0.30 m up and down (T11 spec 6.5;
    # the 2026-09 test found 5 to 30 cm worked and 40 cm stopped). A world, or
    # one person (p["step_up_m"]), can set a smaller number; 0 is the old flat
    # legs exactly. Switch ledger: delete when BODY in materials.py carries it.
    step_up_m = 0.30
    step_down_m = 0.30
    levels_K = 2                          # standing levels per column (D3)
    fast_sweep = True                     # the exact kernel for the flat case

    # ── the step range ──────────────────────────────────────────────────────
    def _step_range(self, p=None):
        """(R_up, R_dn) in cells for this body: 0 means today's behaviour."""
        d = 0.1 * self.scale
        up = dn = None
        if p is not None:
            up, dn = p.get("step_up_m"), p.get("step_down_m")
        if up is None:
            up = self.step_up_m if self.step_up_m is not None \
                else BODY.get("step_up_m", 0.0)
        if dn is None:
            dn = self.step_down_m if self.step_down_m is not None \
                else BODY.get("step_down_m", 0.0)
        return int(round(float(up) / d)), int(round(float(dn) / d))

    def _step_ftab(self, r_max):
        """The integer step cost table for this voxel size, built once."""
        key = (round(self.scale * 100.0), max(int(r_max), 6))
        tab = _FTAB.get(key)
        if tab is None:
            tab = _FTAB[key] = np.array(
                stairs.step_table(self.scale * 10.0, key[1]), np.int64)
        return tab

    # ── belief: what the body last saw it could stand on ───────────────────
    def _body_h(self, cells):
        c = np.asarray(cells)
        return int(c[:, 2].max()) - int(c[:, 2].min()) + 1

    def _see_levels(self, p, cells, reach=None, box=None, write_free=True):
        """Write what this body can see it could stand on into `p["fz"]`, for
        the columns in `reach` (a bool plane the shape of the world, or None
        for the whole `box`), and keep `p["free"]` in step: free is "some level
        is standable". Its own flesh counts as neither floor nor wall.

        `write_free=False` leaves the flat `free` plane alone (C9 does that: the
        flat belief is what every flat reader and the golden hashes read, and
        the planner over levels reads `fz`, not `free`)."""
        nx, ny, _nz = self.shape
        K = int(p.get("levels") or self.levels_K)
        fz = p.get("fz")
        if fz is None:
            fz = p["fz"] = np.full((K, nx, ny), -1, np.int16)
        if reach is not None and box is None:
            xs, ys = np.nonzero(reach)
            if not len(xs):
                return fz
            box = (int(xs.min()), int(xs.max()) + 1,
                   int(ys.min()), int(ys.max()) + 1)
        s = routing.surfaces(self.mat, self.smass, self.vox_l,
                             self._body_h(cells), K=fz.shape[0], box=box,
                             own=cells)
        x0, x1, y0, y1 = s.box
        if reach is None:
            fz[:, x0:x1, y0:y1] = s.top
        else:
            m = reach[x0:x1, y0:y1]
            sub = fz[:, x0:x1, y0:y1]
            sub[:, m] = s.top[:, m]
        if not write_free:
            return fz
        free = p.get("free")
        any_level = (fz[:, x0:x1, y0:y1] >= 0).any(axis=0)
        if free is None:
            free = p["free"] = np.zeros((nx, ny), bool)
        if reach is None:
            free[x0:x1, y0:y1] = any_level
        else:
            m = reach[x0:x1, y0:y1]
            sub = free[x0:x1, y0:y1]
            sub[m] = any_level[m]
        return fz

    def _look_levels(self, p, eye, cells):
        """The look (`_look_around`), and then WHAT IT COULD STAND ON (C9, wave 13c). With the
        step range at 0 this is the old look and nothing more. With a range,
        the standing levels (`p["fz"]`) are written for the columns this look
        told the body something new about: the first look writes every column
        it knows, a later one the columns it has just come to know or whose
        flat belief flipped. A column it saw before and whose floor changed
        without flipping keeps the height it last saw (one stale height per
        level, as the spec says: the legs read the world when they step)."""
        if not any(self._step_range(p)):
            return self._look_around(p, eye, cells)
        known0 = p["known"].copy()
        free0 = None if p.get("free") is None else p["free"].copy()
        self._look_around(p, eye, cells)
        known, free = p["known"], p.get("free")
        if p.get("fz") is None:
            reach = known | (free if free is not None else False)
        else:
            reach = known & ~known0
            if free0 is not None and free is not None:
                reach |= free != free0
        if reach.any():
            self._see_levels(p, cells, reach=reach, write_free=False)

    def _levels_for(self, cells, p):
        """The top[K, nx, ny] this body plans over: its belief if it has one,
        else the world as it is now, masked by what it has been to / found
        free. Always a COPY, because A BODY IS NOT A WALL TO ITSELF: every
        column its own flesh stands in counts as standable at the height of
        its soles (the flat planner's `walk_ok[own] = True`), so an arm
        hanging over a counter does not stop the body from stepping."""
        fz = p.get("fz") if p is not None else None
        if fz is not None:
            top = fz.copy()
        else:
            K = int((p or {}).get("levels") or self.levels_K)
            s = routing.surfaces(self.mat, self.smass, self.vox_l,
                                 self._body_h(cells), K=K, own=cells)
            top = s.top
            if p is not None:
                free = p.get("free")
                if free is not None:
                    # the flat belief says "standable at my own height" and
                    # that is all it says: where it holds and the world now
                    # shows no level, the body still trusts it (a stale belief
                    # is its own to find out by going), at the height of its
                    # soles
                    zlo = int(np.asarray(cells)[:, 2].min())
                    blank = free & (top[0] < 0)
                    top[0][blank] = zlo
                    top[:, ~free] = -1
                top[:, ~p["known"]] = -1
        c = np.asarray(cells)
        zlo = int(c[:, 2].min())
        if fz is not None and p is not None and p.get("free") is not None:
            # THE FLAT BELIEF STILL COUNTS: where the body believes "I can
            # stand here at my own height" and the levels it saw hold no level
            # at that height, it trusts the flat belief, as it always did
            # (and as the no-belief branch above does).
            self._put_level(top, p["free"] & ~(top == zlo).any(axis=0), zlo)
        own = np.zeros(top.shape[1:], bool)
        own[c[:, 0], c[:, 1]] = True
        self._put_level(top, own & ~(top == zlo).any(axis=0), zlo)
        return top

    @staticmethod
    def _put_level(top, mask, z):
        """Write standing height z into the columns of `mask`: in a free slot,
        else over the slot whose height is nearest."""
        if not mask.any():
            return
        xs, ys = np.nonzero(mask)
        t = top[:, xs, ys].astype(np.int32)
        blank = t < 0
        slot = np.where(blank.any(axis=0), blank.argmax(axis=0),
                        np.abs(t - z).argmin(axis=0))
        top[slot, xs, ys] = z

    # ── the planner ─────────────────────────────────────────────────────────
    def _fit_grid(self, cells, p=None):
        up, dn = self._step_range(p)
        if up == 0 and dn == 0:
            return super()._fit_grid(cells, p)
        top = self._levels_for(cells, p)
        fox, foy = routing.footprint(cells)
        # THE PLANE THE FLAT READERS GET (the menu's places, the leap's
        # landings) is the flat plane, plus the columns that fit at a level
        # OTHER than the soles' own and within the step range of it. It never
        # loses a column the flat rule gave, and a far level (the ground two
        # metres below a lip) is not offered: a leap aimed at it would be a
        # new option in a scene with no step in it. A far level is still
        # routed to (the planner's own aim uses every level when it is given
        # no plane); it is only not offered here.
        flat, _ = super()._fit_grid(cells, p)
        zlo = int(np.asarray(cells)[:, 2].min())
        near = (top >= zlo - dn) & (top <= zlo + up) & (top != zlo)
        fit = flat | (routing.fit_levels(top, up, dn, fox, foy) & near).any(axis=0)
        cx = int(round(float(cells[:, 0].mean())))
        cy = int(round(float(cells[:, 1].mean())))
        fit[cx, cy] = True                    # wherever it is now, it fits
        return fit, (cx, cy)

    def _routes(self, cells, targets, p=None, fit=None):
        up, dn = self._step_range(p)
        if up == 0 and dn == 0:
            if not self.fast_sweep:
                return super()._routes(cells, targets, p, fit)
            if fit is None:
                fit, start = self._fit_grid(cells, p)
            else:
                start = (int(round(float(cells[:, 0].mean()))),
                         int(round(float(cells[:, 1].mean()))))
            return routing.routes_flat(fit, start, targets)
        top = self._levels_for(cells, p)
        fox, foy = routing.footprint(cells)
        cx = int(round(float(cells[:, 0].mean())))
        cy = int(round(float(cells[:, 1].mean())))
        zlo = int(np.asarray(cells)[:, 2].min())
        paths, _cost = routing.routes25(
            top, (cx, cy, zlo), up, dn, fox, foy, self._step_ftab(max(up, dn)),
            targets, fit2d=fit)
        return paths

    # ── the leg's step with a rise ──────────────────────────────────────────
    def _step_to(self, cells, sx, sy, p=None):
        """One cell sideways and the rise it needs (see `_step_to_k`)."""
        r = self._step_to_k(cells, sx, sy, p, 1)
        return None if r is None else r[0]

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

    def step_along(self, p, cells, sx, sy):
        """One stride's step with its books closed: `_step_to`, then the
        person's own deed shifted (`_carry_own`), the TIME the move takes (from
        the table: max(d / v_walk, rise / vz_max)), the METABOLIC work, and the
        potential a descent loses as heat.

        The work goes where the body-heat lane's store will read it, `p["_work_J"]`
        (mechanical, metabolic and time totals are kept in `p["_stairs"]`).
        The heat of a descent is booked through the world's one `deposit()` when
        the mechanisms mixin is there (so `total_energy` and `mech_in` move
        together); without it the joules wait in `p["_fall_heat_J"]`.

        Returns (new_cells, dz) or (cells, None) when the step is refused."""
        cells = np.asarray(cells, np.int64)
        dz = self._step_to(cells, sx, sy, p)
        if dz is None:
            return cells, None
        new = cells + np.array([sx, sy, dz], np.int64)
        self._carry_own(p, sx, sy, dz)
        self._step_books(p, new, sx, sy, dz)
        return new, dz

    def _step_books(self, p, new, sx, sy, dz):
        """What a step with a rise owes, once it is taken (`new` = the body's
        cells after it): time, metabolic work, and the heat of a descent.
        Returns the seconds the move takes."""
        d = 0.1 * self.scale
        run = d * float(np.hypot(sx, sy))
        mass_kg = float(self.smass[tuple(new.T)].sum()) / 1000.0
        w = stairs.step_work(mass_kg, dz * d, run)
        t = stairs.step_time(abs(dz) * d, d * float(np.hypot(sx, sy)), mass_kg,
                             p.get("strength_N", BODY.get("strength_N", 400.0)))
        book = p.setdefault("_stairs", {"time_s": 0.0, "metabolic_J": 0.0,
                                        "mech_J": 0.0, "heat_J": 0.0,
                                        "up_m": 0.0, "down_m": 0.0})
        book["time_s"] += t
        book["metabolic_J"] += w["metabolic_J"]
        book["mech_J"] += w["mech_J"] if dz > 0 else 0.0
        book["heat_J"] += w["heat_J"]
        book["up_m" if dz > 0 else "down_m"] += abs(dz) * d
        p["_work_J"] = p.get("_work_J", 0.0) + w["metabolic_J"]
        if dz < 0:
            fall = w["fall_heat_J"]
            dep = getattr(self, "deposit", None)
            if dep is None:
                p["_fall_heat_J"] = p.get("_fall_heat_J", 0.0) + fall
            else:
                zmin = int(new[:, 2].min())
                soles = new[new[:, 2] == zmin]
                feet = tuple(soles.T)
                floor = soles.copy()
                floor[:, 2] -= 1
                floor = floor[(floor[:, 2] >= 0)
                              & (self.mat[tuple(floor.T)] != AIR)]
                sp = stairs.DESCENT_HEAT_SPLIT
                if len(floor):
                    dep(feet, fall * sp["feet"])
                    dep(tuple(floor.T), fall * sp["floor"])
                else:
                    dep(feet, fall)
        return t
