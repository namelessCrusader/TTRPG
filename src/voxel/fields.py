"""Derived fields (heat capacity, temperature, porosity), the authoring API, activity windows."""
import weakref

import numpy as np
from .kernels import HAVE_NUMBA, _diff4
from .window import box_volume
from .materials import (AIR, AIR_C, AMBIENT, FLUID, MISCIBLE, NOFLUID, OIL, SOLID, _CSOLID_ARR, _DENS_ARR, _FC_ARR, _FDENS_ARR, _REACTIVE)


# world -> (the `_cap_key` they were made for, heat capacity, porosity): the
# derived fields `_matter_moved` kept up cell by cell. Outside the world, so
# its state (`_cap`, `_por`, None until asked for) is just what it always was.
_KEPT = weakref.WeakKeyDictionary()


# Below this, the cells of fuel catch no flame: even a fuel with the lowest
# ignition temperature in the tables needs a cell hotter than ambient to burn.
# `burning` leans on it to leave every cell at ambient out of the sum.
_BURN_FLOOR = min([spec[3] for spec in SOLID.values() if spec[3] is not None]
                  + [FLUID[OIL][3]])


class FieldsMixin:

    # THE LAWS WORK WHERE THE MATTER IS (wave 12, `window.py`). On, a law that
    # can find its work from a list of cells, or from the box round the
    # heat, does not sweep the lattice, and gives the same bits as if it had.
    # Off is the whole-lattice code, the definition: tests/test_scale.py runs
    # worlds both ways.
    windows = True

    # ── derived fields ───────────────────────────────────────────────────────
    def heat_capacity(self):
        """What it costs to warm a voxel by a degree: its solid, its fluid, and
        the air in the space left over.

        CACHED against the four fields it is made of. It was being rebuilt
        TWELVE TIMES A TICK — a fifth of all sim time, burning or not — for a
        quantity that only changes when matter does: something burns away,
        melts, is poured, or walks. Comparing the four costs about a fifth of
        one rebuild, and it is right by construction because it compares the
        real state, rather than by everyone who writes a voxel remembering to
        invalidate it. Same bargain as the slack field, and the same reason.

        Callers only ever read it — checked — so one array is handed to all."""
        self._matter_moved()
        if self._cap is None:
            kept = self._kept(1)
            self._cap = (kept if kept is not None
                         else self._cap_of(self.mat, self.smass, self.fl, self.fvol,
                                           *self._mix_args()))
        return self._cap

    def _kept(self, n):
        """Field `n` (1 heat capacity, 2 porosity) as `_matter_moved` kept it
        up for the matter there is now, or None."""
        kept = _KEPT.get(self)
        if kept is not None and kept[0] is self._cap_key:
            return kept[n]
        return None

    def _cap_of(self, mat, smass, fl, fvol, scomp=None, fcomp=None):
        """Heat capacity, J/K, of the cells given: the whole lattice, or a list
        of cells (every step is the cell's own, so both give the same bits).

        `scomp` and `fcomp` (T1) are the mix ids of those same cells; None is the
        old path. Only the cells that hold a mix read the table: every other
        cell gets the very same float32 it always got."""
        c_solid = _CSOLID_ARR[mat]
        fdens = _FDENS_ARR[fl]
        c_fluid = _FC_ARR[fl]
        if scomp is not None:
            sel = scomp > 0
            if sel.any():
                c_solid = c_solid.copy()
                c_solid[sel] = self.mixes.prop("c", scomp[sel])
        if fcomp is not None:
            sel = fcomp > 0
            if sel.any():
                fdens, c_fluid = fdens.copy(), c_fluid.copy()
                fdens[sel] = self.mixes.prop("rho", fcomp[sel]) / 1000.0     # g/ml
                c_fluid[sel] = self.mixes.prop("c", fcomp[sel])
        fmass = fvol * fdens
        return smass * c_solid + fmass * c_fluid + self.c_airbase

    def _por_of(self, mat, smass, fvol, scomp=None):
        """Porosity (`porosity`) of the cells given, cell by cell like
        `_cap_of`. A cell that holds a solid mix packs against the mix's own
        density (`scomp`, T1), not the density of the material it views as."""
        dens = _DENS_ARR[mat]
        if scomp is not None:
            sel = scomp > 0
            if sel.any():
                dens = dens.copy()
                dens[sel] = self.mixes.prop("rho", scomp[sel])
        packed = smass / np.maximum(dens * self.vox_l, 1e-9)
        void = np.clip(1.0 - packed, 0.0, 1.0)
        drowned = fvol > 0.5 * self.cap          # a flooded gap does not breathe
        return np.where(drowned, np.minimum(void, 0.02), void)

    def _mix_args(self):
        """(scomp, fcomp) for `_cap_of` and `_por_of`: both None (the old path)
        until a cell holds a mix."""
        if not self.mixes_live:
            return (None, None)
        return (self.scomp, self.fcomp)

    def _windows_on(self):
        """Do the laws work from lists and boxes (`windows`)? Not when the
        skipping is off: that setting means every law over everything."""
        return self.skip_quiet and self.windows

    def _windowed(self):
        """`_windows_on`, for a law whose list kernel is compiled."""
        return self._windows_on() and self.fused and HAVE_NUMBA

    def _sleeps(self):
        """Do the laws skip the parts of the world they cannot change
        (`sleep_regions`)? Only when quiet things are skipped at all."""
        return self.skip_quiet and getattr(self, "sleep_regions", True)

    def _matter_moved(self):
        """Has any matter changed since the derived fields were last built? If
        it has, throw them away and remember the new state.

        One comparison serves both heat capacity and porosity, because both are
        made of the same four fields — so the price of checking is paid once
        however many derived fields come to depend on it.

        ONLY WHERE IT MOVED (`sleep_regions`). A fire eats a little wood every
        tick, and that threw both fields away and built them again for the
        whole world, a dozen passes over it for a few hundred cells. Each
        cell's capacity and porosity are made of its own four fields alone, so
        the cells that changed are worked out again (`_cap_of`, `_por_of`) and
        every other cell keeps what it had, which is what a full build would
        give it: the same bits. They are kept beside the world (`_KEPT`) and
        handed out when asked for, as NEW arrays, as a full build would be, so
        whoever still holds the old one keeps the old state, and
        `_surface_geometry` still sees a change by identity."""
        k = self._cap_key
        now = (self.mat, self.smass, self.fl, self.fvol)
        # A MIX CHANGES WHAT A CELL HOLDS WITHOUT CHANGING THESE FOUR (T1): a
        # convert() moves grams between forms and leaves mat, smass, fl and
        # fvol where they were. So a world with mixes keeps a fifth and a
        # sixth copy beside the key (`_mix_key`), and a change in either throws
        # the fields away. `_cap_key` itself stays four arrays: it is hashed.
        mix_same = True
        if self.mixes_live:
            mk = self.__dict__.get("_mix_key")
            mix_same = (mk is not None and np.array_equal(mk[0], self.scomp)
                        and np.array_equal(mk[1], self.fcomp))
        if (mix_same and k is not None and np.array_equal(k[0], self.mat)
                and np.array_equal(k[1], self.smass)
                and np.array_equal(k[2], self.fl)
                and np.array_equal(k[3], self.fvol)):
            return False
        if (k is not None and not self.mixes_live
                and self.fused and HAVE_NUMBA and self._sleeps()
                and all(a.shape == b.shape and a.dtype == b.dtype
                        for a, b in zip(k, now))):
            flat = [np.ascontiguousarray(a).reshape(-1) for a in now]
            out = np.empty(max(flat[0].size // 8, 1), np.int64)
            n = _diff4(*flat, *(a.reshape(-1) for a in k), out)
            if n > 0:                            # a few cells: just those
                i = out[:n]
                cells = [b[i] for b in flat]
                cap = self._cap if self._cap is not None else self._kept(1)
                por = self._por if self._por is not None else self._kept(2)
                if cap is not None:
                    cap = cap.copy()
                    cap.reshape(-1)[i] = self._cap_of(*cells)
                if por is not None:
                    por = por.copy()
                    por.reshape(-1)[i] = self._por_of(cells[0], cells[1], cells[3])
                self._cap = self._por = None
                # THE KEPT COPIES ARE BROUGHT UP TO DATE CELL BY CELL, too: they
                # are ours alone, so only the cells that changed are written
                # (four copies of the lattice were a fifth of this law's
                # cost). The tuple is a new object, because `_kept` knows
                # which state a field was made for by its identity.
                for a, kk in zip(cells, k):
                    kk.reshape(-1)[i] = a
                self._cap_key = tuple(k)
                _KEPT[self] = (self._cap_key, cap, por)
                self._matter_gen = getattr(self, "_matter_gen", 0) + 1
                return True
        self._cap = self._por = None
        self._cap_key = (self.mat.copy(), self.smass.copy(),
                         self.fl.copy(), self.fvol.copy())
        if self.mixes_live:
            self._mix_key = (self.scomp.copy(), self.fcomp.copy())
        # AND A GENERATION, because this RESETS as it answers: whoever asks
        # second in a tick is told nothing moved, which is true of the interval
        # between the two questions and false of the tick. A cache that wants
        # "has matter moved since I last looked" has to compare a stamp rather
        # than ask, or it goes stale exactly when a second caller appears.
        self._matter_gen = getattr(self, "_matter_gen", 0) + 1
        return True

    @property
    def c_airbase(self):
        """What the air in one voxel holds per kelvin, J/K: real air, 1.2 J/(L·K)
        (AIR_C, H12), in every voxel whatever else is in it. (It was 50
        J/(L·K), a "thermal floor" 41 times real air.)"""
        return AIR_C * self.vox_l

    def T(self):
        return self._T_in()[0]

    def _T_in(self, fill=True):
        """The temperature field, and the box of the world where it is not
        ambient: None when it is ambient everywhere, else a box of slices
        (the whole world when the laws are not windowed).

        `AMBIENT + E / heat_capacity` for every cell, except that a cell
        holding no heat above ambient (E == 0: 0 / capacity is 0, whatever
        the capacity) is AMBIENT exactly, so only the box round the heat is
        worked out and the rest of the field is filled in. The same bits as
        the sum over the lattice, which costs 2 ms at 1.6 M voxels even in a
        room where nothing is warm. Where it is warm nowhere the field is not
        even made when the caller only wants the box (`fill=False`, field
        None).

        (The heat capacity is asked for first, as it always was, though a
        room at ambient has no use for its values: asking is also what
        notices that matter has moved, and the derived fields it keeps are
        part of the state a world is compared by.)"""
        whole = (slice(None), slice(None), slice(None))
        cap = self.heat_capacity()
        if not self._windows_on():
            return AMBIENT + self.E / cap, whole
        box = self._unsettled(0)[0]
        if box is None:
            if not fill:
                return None, None
            return np.full(self.shape, AMBIENT, self._T_dtype()), None
        if box_volume(box) * 2 > self.E.size:
            return AMBIENT + self.E / cap, whole
        t = np.full(self.shape, AMBIENT, self._T_dtype())
        t[box] = AMBIENT + self.E[box] / cap[box]
        return t, box

    def _T_dtype(self):
        """The type `AMBIENT + E / heat_capacity` comes out as."""
        cap = np.result_type(self.smass.dtype, _CSOLID_ARR.dtype,
                             self.fvol.dtype, _FDENS_ARR.dtype, _FC_ARR.dtype)
        return (np.zeros(1, self.E.dtype) / np.ones(1, cap) + AMBIENT).dtype

    def porosity(self):
        """How freely gas crosses a voxel: THE SPACE THAT IS LEFT IN IT.

        Not a material row. `fill()` has always said it — "frac < 1 is a PARTIAL
        voxel: a stick is mostly air inside its cube" — and `smass` has always
        carried it; the gas laws simply threw it away by asking `mat == AIR`, a
        binary question about a continuous field. Solid mass over what the
        material would weigh packed solid IS the occupied fraction; what is left
        is void, and void is what gas moves through.

        So a shut door leaks because the SCENE says it fills 99% of its cells —
        a true statement about that door's geometry, at the one resolution the
        lattice cannot draw (a 1 cm undercut in a 5 cm voxel). Wood is not
        declared leaky; this door is. Nothing here knows what a door is.

        What falls out for free, none of it written: rubble and thatch breathe,
        a wall half-eaten by acid breathes through the loss, and wood GROWS more
        permeable as it burns away, feeding the fire that is thinning it."""
        self._matter_moved()                      # cached beside heat capacity:
        if self._por is None:                     # same four fields, one check
            kept = self._kept(2)
            self._por = (kept if kept is not None
                         else self._por_of(self.mat, self.smass, self.fvol,
                                           *self._mix_args()[:1]))
        return self._por

    def burning(self):
        """Where combustion is HAPPENING right now (derived, not stored).

        A cell at ambient burns nothing (`_BURN_FLOOR`), so only the box
        round the heat (and the skin's) is looked at, and the rest is False:
        the same cells as the sum over the lattice, which was six masks over
        1.6 M voxels in a room where nothing was warm."""
        T, box = self._T_in(fill=False)
        s = getattr(self, "skin", None)              # fuel burns at its SURFACE
        skin = s is not None and s.shape == self.shape   # (`_surfaces`): the skin's
        whole = box is not None and box[0] == slice(None)   # temperature, not
        if skin and not whole and self._windows_on():       # the cell's
            there = self._window(s != 0, 0)
            if there is not None:
                box = there if box is None else tuple(
                    slice(min(a.start, c.start), max(a.stop, c.stop))
                    for a, c in zip(box, there))
        if whole or _BURN_FLOOR <= AMBIENT or not self._windows_on():
            if T is None:
                T = self.T()
            if skin:
                T = T + s
            out = (self.fl == OIL) & (self.fvol > 0) & (T >= FLUID[OIL][3])
            for m, spec in SOLID.items():
                if spec[3] is not None:
                    out |= (self.mat == m) & (self.smass > 0) & (T >= spec[3])
            return out
        out = np.zeros(self.shape, bool)
        if box is None:
            return out
        T = (np.full(tuple(a.stop - a.start for a in box), AMBIENT,
                     self._T_dtype()) if T is None else T[box])
        if skin:
            T = T + s[box]
        fl, mat, sm = self.fl[box], self.mat[box], self.smass[box]
        here = (fl == OIL) & (self.fvol[box] > 0) & (T >= FLUID[OIL][3])
        for m, spec in SOLID.items():
            if spec[3] is not None:
                here |= (mat == m) & (sm > 0) & (T >= spec[3])
        out[box] = here
        return out

    # ── placing things (scene setup) ─────────────────────────────────────────
    def fill(self, x0, x1, y0, y1, z0, z1, material, frac=1.0, edge=0.0, mix=None):
        """frac < 1 is a PARTIAL voxel: a stick is mostly air inside its cube.
        edge > 0 declares a working edge: the m² this thing concentrates a blow
        into (an axe bit ~2e-4). A property of the OBJECT, not the material.

        The matter put here is part of whichever thing is being made
        (`_stamping`, 0 for none), and air is part of no thing.

        `mix` (T1) is a row of the mix table (an int, or the list `MixTable.parse`
        takes): the solid is made of that mix. `material` is then the mat the
        cells view as (None: the table's own view), and the mass is the mix's
        own density times the volume. Without `mix`, the cells are pure, and a
        cell that held a mix before is set back to pure through `set_mat`."""
        sl = (slice(x0, x1), slice(y0, y1), slice(z0, z1))
        if mix is not None:
            row = self._as_row(mix)
            if material is None:
                material = int(self.mixes.mat_view(np.array([row]))[0])
            self.mat[sl] = material
            self.set_solid(sl, row)
            self.smass[sl] = float(self.mixes.prop("rho", np.array([row]))[0]) * frac * self.vox_l
        else:
            self.set_mat(sl, material)
            self.smass[sl] = SOLID[material][0] * frac * self.vox_l
        self.edge[sl] = edge
        self._things_moved(self.eid[sl])
        self.eid[sl] = self._stamping if material != AIR else 0
        self._things_moved(self.eid[sl])

    def pour(self, x, y, z, fluid, ml, pot=1.0, mix=None):
        """Put `ml` of fluid in a cell of air. `mix` (T1) is a row of the mix
        table (a fluid-slot row, or what `MixTable.parse` takes): the fluid is
        that mix, and `fluid` is the fl it views as (None: the table's view).
        A cell that holds a mix, or a pour of one, joins by the exact blend
        (`_fluid_blend`); a pure pour into a pure cell is the old code."""
        at = (x, y, z)
        if mix is None and not (self.mixes_live and self._frow(at)):
            if fluid not in _REACTIVE:
                pot = 0.0                    # water has no potency to bring
            if self.mat[x, y, z] == AIR and self._mix(fluid, self.fl[x, y, z]) is not None:
                old = float(self.fvol[x, y, z])
                add = min(self.cap - old, ml)
                if add <= 0:
                    return
                self.fpot[x, y, z] = (self.fpot[x, y, z] * old + pot * add) / (old + add)
                self.fl[x, y, z] = self._mix(fluid, self.fl[x, y, z])
                self.fvol[x, y, z] = old + add
            return
        row = self._as_row(mix) if mix is not None else 64 + int(fluid)
        if fluid is None:
            fluid = int(self.mixes.fl_view(np.array([row]))[0])
        if self.mat[x, y, z] != AIR or self._mix_ids(
                fluid, mix is not None, self.fl[at], self._frow(at) > 0) is None:
            return                                   # a solid, or a fluid that refuses it
        if fluid not in _REACTIVE:
            pot = 0.0
        plan = self._fluid_plan(at, row, float(ml))
        if plan is None or plan[0] <= 0:
            return
        old = float(self.fvol[at])
        self._fluid_commit(at, plan, fluid)
        self.fpot[at] = (self.fpot[at] * old + pot * plan[0]) / (old + plan[0])

    # ── COMPOSITION IN THE LATTICE (wave 13, T1; plan section 1) ────────────
    # THE RULE: once any cell holds a mix, `mat` and `fl` are VIEWS of what the
    # mix is made of, and the only writers of a cell's matter are the three
    # below (and the fluid hand-over, which calls `_fluid_commit`). Nothing in
    # 13a puts a mix in a cell on its own: a scene does, through `fill(mix=)`,
    # `pour(mix=)` or these. `mixes_live` is False until then, every method
    # here is one `if` and the old statement, and golden cannot move.
    def _as_row(self, mix):
        """A mix table row for `mix`: an int, or a list `MixTable.parse` takes."""
        t = self.ensure_mixes()
        if isinstance(mix, (int, np.integer)):
            return int(mix)
        return t.parse(mix)

    def set_mat(self, idx, new_mat):
        """THE writer of `mat` where the matter changes kind (burn, char, melt,
        a shard, an acid residue). Pure world: `mat[idx] = new_mat`, the old
        statement. With mixes: the cells become pure `new_mat`, and the grams a
        mix held there are booked to `recast` (old negative, new positive), so
        the species totals stay exact up to that one named ledger."""
        if not self.mixes_live:
            self.mat[idx] = new_mat
            return
        self.mixes.set_mat(self.mat, self.scomp, self.smass, idx, new_mat)

    def set_solid(self, idx, rows):
        """Write the solid mix of the cells `idx` (an int row or an array),
        and `mat` as the view of it (with the table's hysteresis: the old mat
        is kept while its share is within 5 percent of the leader's). Row 0
        means pure: scomp becomes 0 and mat is left alone. `smass` is the
        caller's own to write (mass and composition are two facts)."""
        rows = np.asarray(rows, np.int64)
        if not self.mixes_live:
            if not rows.any():
                return
            self.ensure_mixes()
        cur = self.mat[idx]
        rows = np.broadcast_to(rows, cur.shape)
        self.ensure_mixes()
        view = self._view_of(self.mixes.mat_view, "prev_mat", rows, cur)
        self.scomp[idx] = rows
        self.mat[idx] = np.where(rows == 0, cur, view)
        if rows.any():
            self.mixes_live = True

    def set_fluid(self, idx, rows):
        """`set_solid` for the fluid: write `fcomp`, and `fl` as the solvent
        view of it. `fvol` and `fpot` are the caller's."""
        rows = np.asarray(rows, np.int64)
        if not self.mixes_live:
            if not rows.any():
                return
            self.ensure_mixes()
        cur = self.fl[idx]
        rows = np.broadcast_to(rows, cur.shape)
        self.ensure_mixes()
        view = self._view_of(self.mixes.fl_view, "prev_fl", rows, cur)
        self.fcomp[idx] = rows
        self.fl[idx] = np.where(rows == 0, cur, view)
        if rows.any():
            self.mixes_live = True

    @staticmethod
    def _view_of(fn, key, rows, cur):
        """`mat_view` / `fl_view` (`fn`, whose old-value argument is `key`) of
        whole arrays of cells. One row for many cells (a fill) is looked up once
        per distinct old value."""
        flat, prev = rows.ravel(), cur.ravel()
        if not flat.size:
            return flat.reshape(cur.shape)
        if (flat == flat[0]).all():
            out = np.empty(flat.size, np.int64)
            for pv in np.unique(prev):
                out[prev == pv] = fn(flat[:1], **{key: pv})[0]
            return out.reshape(cur.shape)
        return fn(flat, **{key: prev}).reshape(cur.shape)

    def legacy_ok(self, solid=True, fluid=True):
        """D6: the cells a legacy matter-changing law may work on. None when no
        cell holds a mix (the law runs as it always did, untouched); else a
        bool lattice, True where the solid is pure (`solid`) and the fluid is
        pure (`fluid`). Every law that WRITES smass, fvol, fl, mat or fpot
        multiplies its mask by this until it is moved to take/put; the skip is
        in the switch ledger with its deleting condition."""
        if not self.mixes_live:
            return None
        ok = np.ones(self.shape, bool)
        if solid:
            ok &= self.scomp == 0
        if fluid:
            ok &= self.fcomp == 0
        return ok

    def _frow(self, at):
        """The fluid mix row of ONE cell, 0 if pure or empty (a row left behind
        by a law that emptied the cell without knowing it is not read)."""
        if not self.mixes_live or self.fl[at] == NOFLUID or self.fvol[at] <= 0:
            return 0
        return int(self.fcomp[at])

    def _fluid_row(self, at):
        """The table row of the fluid in ONE cell: its mix, else the pure row."""
        r = self._frow(at)
        return r if r else 64 + int(self.fl[at])

    def _fdens_grid(self):
        """Fluid density, g/ml, of every cell: `_FDENS_ARR[fl]`, except where
        the fluid is a mix, which has its own (brine sinks under fresh water
        though both are `fl` WATER). Not a mix world: that very expression."""
        d = _FDENS_ARR[self.fl]
        if not self.mixes_live:
            return d
        sel = self.fcomp > 0
        if sel.any():
            d = d.copy()
            d[sel] = self.mixes.prop("rho", self.fcomp[sel]) / 1000.0
        return d

    def _solid_rho(self, mat, scomp=None):
        """Solid density, g/L, of the cells given: `_DENS_ARR[mat]`, and the
        mix's own where `scomp` (their mix ids, or None) says so."""
        d = _DENS_ARR[mat]
        if scomp is not None and self.mixes_live:
            sel = scomp > 0
            if sel.any():
                d = d.copy()
                d[sel] = self.mixes.prop("rho", scomp[sel])
        return d

    @staticmethod
    def _mix_ids(f_in, mixed_in, f_at, mixed_at):
        """`_mix` for two fluids of which either may be a mix. WHAT "REFUSED"
        MEANS: acid (the family whose truth is `fpot`, `_REACTIVE`) and a mix
        do not blend in wave 13, because `_law_acid` lets spent acid pale into
        inert sludge at constant volume, which is not a mixing process and has
        no home in a table of species yet (T1 4.7). So the answer is None,
        exactly like oil into water: nothing moves, nothing is booked, no heat
        is made, and both cells stay as they were. They may still swap places by
        density. Pouring acid on brine, and brine on acid, are both no-ops."""
        f_out = FieldsMixin._mix(f_in, f_at)
        if f_out is None:
            return None
        mixed_in = mixed_in and f_in != NOFLUID           # (nothing in an empty cell is a mix)
        mixed_at = mixed_at and f_at != NOFLUID
        if (mixed_in or mixed_at) and (int(f_in) in _REACTIVE or int(f_at) in _REACTIVE):
            return None
        return f_out

    def _mix_cells(self, a, b):
        """`_mix` for the fluids in cells `a` (the giver) and `b` (the taker).
        Not a mix world: `_mix(fl[a], fl[b])`, the old call."""
        if not self.mixes_live:
            return self._mix(self.fl[a], self.fl[b])
        return self._mix_ids(self.fl[a], self._frow(a) > 0, self.fl[b], self._frow(b) > 0)

    def _fluid_plan(self, dst, row_in, ml):
        """THE MASS-EXACT BLEND (T1 4.7). `ml` of the fluid of table row
        `row_in` joins the cell `dst`. Mass is exact and volume follows:
            m_move = ml * rho(row_in);   M_new = fvol_dst * rho(dst) + m_move;
            row_new = blend;             vol_new = M_new / rho(row_new).
        If vol_new passes the cell's capacity, `ml` is scaled by
        (cap - fvol_dst) / (vol_new - fvol_dst) and the blend made again once;
        any overfill left (float noise) is clamped, and its mass is booked to
        `squeeze`. Returns (ml accepted, row_new, vol_new, squeezed grams), or
        None. Writes nothing except new rows in the table."""
        T = self.mixes
        old = float(self.fvol[dst]) if self.fl[dst] != NOFLUID else 0.0
        rho_in = float(T.prop("rho", np.array([row_in]))[0]) / 1000.0
        room = self.cap - old
        if room <= 0 or ml <= 0:
            return None
        ml = min(ml, room)
        if old <= 0:
            return (ml, row_in, ml, 0.0)
        r_dst = self._fluid_row(dst)
        rho_d = float(T.prop("rho", np.array([r_dst]))[0]) / 1000.0
        sq = 0.0
        for _ in range(2):
            rows, M = T.blend(np.array([r_dst]), np.array([old * rho_d]),
                              np.array([row_in]), np.array([ml * rho_in]))
            rho_new = float(T.prop("rho", rows)[0]) / 1000.0
            vol = float(M[0]) / rho_new
            if vol <= self.cap:
                break
            ml *= (self.cap - old) / (vol - old)
        else:
            sq = (vol - self.cap) * rho_new
            vol = self.cap
        return (ml, int(rows[0]), vol, sq)

    def _fluid_commit(self, dst, plan, f_hint):
        """Write the result of `_fluid_plan` to the cell `dst`: `fcomp` (0 when
        the result is a pure fluid row), `fl` as the view, `fvol` as float32.
        What float32 cannot hold, and the clamp, are booked (`f32`, `squeeze`)."""
        T = self.mixes
        ml, row, vol, sq = plan
        v32 = np.float32(vol)
        rho = float(T.prop("rho", np.array([row]))[0]) / 1000.0
        ent = T.entries(np.array([row]))
        T.book("f32", ent.species, ent.frac * (vol - float(v32)) * rho)
        if sq:
            T.book("squeeze", ent.species, ent.frac * sq)
        if 64 <= row < 128:                     # a pure fluid row: back to the old path
            self.fcomp[dst] = 0
            self.fl[dst] = row - 64
        else:
            self.fcomp[dst] = row
            prev = int(self.fl[dst]) if self.fl[dst] != NOFLUID else int(f_hint)
            self.fl[dst] = int(T.fl_view(np.array([row]), prev_fl=prev)[0])
            self.mixes_live = True
        self.fvol[dst] = v32

    @staticmethod
    def _mix(f_in, f_at):
        """Species the target voxel holds after f_in joins f_at — or None if the
        two refuse each other (oil into water). Same fluid or empty: trivial."""
        if f_at in (NOFLUID, f_in):
            return f_in
        return MISCIBLE.get(frozenset({int(f_in), int(f_at)}))

    def _hot_window(self, pad=1):
        """The slab of the world heat can be in, grown by one voxel of room to
        move into — or None when nothing anywhere is above ambient.

        Heat travels one voxel per LAW THAT MOVES IT, and no further: every
        term of `_law_conduct` is across a face between TOUCHING cells, and its
        only other path is radiation from a surface out of the world, which
        cannot warm a neighbour. So a box around everything above ambient,
        grown by one, contains everything ONE such law can write. (A flame's
        heat is placed by the burn law, which runs over the whole world.)

        Not one voxel a TICK, which is what this first claimed and is wrong:
        conduction and buoyancy both move heat and both run in the same tick, so
        heat really does get two voxels from home before the tick is out. The
        measured symptom was a single cell — the sim left 3.4 J at z4 that the
        windowed run left at zero, on the very first tick, because conduction
        carried heat up to z3 and buoyancy then carried some of it to z4, one
        past the edge of a box drawn before either ran.

        So the window is asked for again between the two, rather than padded by
        two once. That costs one more scan of a field we have already touched,
        and it stays right when a third law that moves heat is added — which a
        comment saying "pad by two" would not.

        This is the half that matters for a big world: a town with one room
        alight answers "yes, something is hot" and would otherwise get no skip
        at all, because presence alone is a question about the WHOLE world."""
        return self._unsettled(pad)[0]

    # ── THE SEAMS OF "WHAT IS NOT AT REST" (wave 13, plan 3.3) ──────────────
    # Today the world rests at 20 °C and "heat is here" means `E != 0`. Cold
    # (T8) gives a world another rest, and the scheduler (T5) asks the same
    # question to decide whether a night can be skipped. Both hang off these
    # three methods, so that the sim, the fields and the quiet gate all mean
    # the same thing. At their defaults each returns exactly what the code it
    # stands in for computed before they existed.
    def _heat_busy(self):
        """Is there any heat to move at all? The gate in front of burning,
        boiling, melting, the oxygen law and the dev heaters (`step`)."""
        return self._busy(self.E)

    def _unsettled(self, pad=0):
        """(box, reasons): the slab of the world that is not at rest, grown by
        `pad`, or None if all of it is; and why, a list of words that is empty
        here. At the old rest (every cell at 20 °C, E == 0) that is the window
        of `E != 0` and no reasons, which is `_hot_window`'s old answer."""
        return self._window(self.E != 0, pad), []

    def _rest_cell(self):
        """The E a settled cell holds, in kelvin above 20 °C: 0 everywhere.
        (An array, cell by cell, in a world with an outside.)"""
        return np.float32(0.0)

    # ── THE LAZY STATE OF WAVE 13 (plan section 1) ──────────────────────────
    # Every one of these is None (or False) on a World until a scene or a law
    # asks for it, and costs nothing until then (`sim.World` holds the names as
    # CLASS attributes, so an old snapshot, which does not carry them, restores
    # cleanly and golden does not see them). The `ensure_*` methods make one,
    # once; `lazy_bytes` says what is held.
    LAZY_ARRAYS = (("scomp", np.uint32), ("fcomp", np.uint32), ("dmg", np.float32),
                   ("bzone", np.uint16), ("ice_g", np.float32))

    def _ensure_array(self, name):
        a = getattr(self, name)
        if a is None:
            dtype = dict(self.LAZY_ARRAYS)[name]
            a = np.zeros(self.shape, dtype)
            setattr(self, name, a)
        return a

    def ensure_mixes(self):
        """T1: the mix table and the two mix-id arrays (scomp, fcomp: uint32,
        0 = a pure cell, read the old way). `mixes_live` stays False until a
        writer puts a mix id in a cell. Returns the table."""
        if self.mixes is None:
            from .mixtable import MixTable
            self.mixes = MixTable()
        self._ensure_array("scomp")
        self._ensure_array("fcomp")
        return self.mixes

    def ensure_damage(self):
        """T9: damage per cell, float32, 0 .. 0.999 (one writer, `add_damage`)."""
        return self._ensure_array("dmg")

    def ensure_bzone(self):
        """T7: the overlay id of each cell (person slot x 16 + zone), uint16,
        0 = none. A carried field (CARRIED_OPT): it moves with the matter."""
        return self._ensure_array("bzone")

    def ensure_ice(self):
        """T6: grams of ice in a cell that is not (yet) an ICE cell, float32."""
        return self._ensure_array("ice_g")

    def ensure_records(self):
        """T2: the record pools (`record.Records`), empty."""
        if self.records is None:
            from .record import Records
            self.records = Records()
        return self.records

    def ensure_layers(self):
        """T3: the surface-layer pool (`layers.Layers`), empty, at this
        world's voxel size."""
        if self.layers is None:
            from .layers import Layers
            self.layers = Layers(dx=0.1 * self.scale)
        return self.layers

    def mix_holders(self):
        """name -> array of everything that holds a mix id, for
        `MixTable.collect`. For now the two lattice arrays. A lane that puts a
        mix id anywhere else (a thrown body, a drop, a particle, a record, a
        layer) adds it HERE, in the same change, and registers it with
        `MixTable.register_holder`, or a collection would renumber the rows
        and leave that holder pointing at the wrong ones."""
        return {n: getattr(self, n) for n in ("scomp", "fcomp") if getattr(self, n) is not None}

    def collect_mixes(self):
        """The automatic hook for a QUIET tick of the slow clock (T1): free the
        mix rows nothing refers to and renumber the holders, when the table has
        grown past twice what is live. Does nothing without a mix table, and
        nothing while matter is off the lattice (a body mid-topple, a drop in
        flight, cells the particle solver moves: their ids are not in
        `mix_holders`). Returns the old-to-new row array, or None."""
        if self.mixes is None or self.bodies or self.drops or self.moving.any():
            return None
        return self.mixes.maybe_collect(self.mix_holders())

    def lazy_bytes(self):
        """{name: bytes} of the lazy arrays that exist. A world that has made
        none of them returns {}."""
        return {n: int(getattr(self, n).nbytes) for n, _d in self.LAZY_ARRAYS
                if getattr(self, n) is not None}

    def _window(self, there, pad):
        """A box around everything `there` is true of, grown by `pad` — or None
        if it is nowhere. With skipping off, the whole world."""
        if not self.skip_quiet:
            return (slice(None), slice(None), slice(None))
        out = []
        for ax in range(3):
            line = there.any(axis=tuple(a for a in range(3) if a != ax))
            nz = np.flatnonzero(line)
            if not len(nz):
                return None
            out.append(slice(max(int(nz[0]) - pad, 0),
                             min(int(nz[-1]) + 1 + pad, self.shape[ax])))
        return tuple(out)

    def _busy(self, *fields):
        """Is there anything here at all for a law to move?

        A whole-array reduce costs about a tenth of a millisecond on a 300k
        world; the laws it stands in front of cost tens. Each question is asked
        IMMEDIATELY before the law it gates, never hoisted to the top of the
        tick — a fire that lights this tick makes its smoke this tick, and a
        stale answer would leave that smoke sitting still for a tick."""
        if not self.skip_quiet:
            return True
        return any(bool(f.any()) for f in fields)
