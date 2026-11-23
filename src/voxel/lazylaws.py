"""LazyLawsMixin: EXACT SPEEDUPS FOR A WORLD WITH ONE WARM THING IN IT (wave 15, S4-3).

A warm body keeps "heat is here" true for ever (E != 0), so every law that
asks it wakes and sweeps the whole lattice: the burn law, the oxygen law, the
pneumatic air law, a dozen derived-field checks. The person's own cells are a
few thousand voxels of 1.6 million; the rest of the world is at rest and
contributes nothing. This mixin answers each of those questions from the cells
that can matter, and gives the SAME bits as the sweep.

THE GATE. `lazy_exact` is False on a plain World: every method here is then
`super()`, and an old scene takes the old path bit for bit. OutsideMixin (cold
and body-heat worlds) turns it on. Flipping it for every world is a declared
step of its own (the default-on flips, wave 15 batch 3).

WHAT IT DOES, each one an exact answer:
  * the matter memo (`fields._matter_moved`): once the kept copy of the matter
    has been compared, the answer stays "no" until a law that can write matter
    has run. The laws are wrapped here (`__init_subclass__`); the ones that
    cannot write matter are listed in MATTER_PURE, every other `_law_*` is
    assumed to write.
  * the summary of the heat (`_hot`): the box round the heat and the skin, the
    temperatures in it, and the three yes-or-no answers the quiet laws need, made
    once and kept until a law that can write E or matter has run.
  * laws left out when a proof says they have nothing to do (PROOFS): crack, boil,
    melt and the wet skin are not run, burn runs its quiet exit, `_law_o2` returns
    early. A law is left out ONLY when the memo is verified (`_lz_verified`), so
    that every ask it would have made is a no-op: `_matter_gen`, `_cap` and `_por`
    are state a golden check compares, and a skipped law must not move an ask.
  * `_burn_cells`, `burning()`, `_law_o2`: the quiet tests read over the box
    round the heat, not over three whole-lattice arrays.
  * `_warm_cells`, `_cells`, `_law_air`: the coarse push of warm gas worked
    out over the blocks the heat is in; the coarse mean of the porosity kept
    until matter moves; no smoke to carry means no smoke carried.

THE AUDIT. With MARK1_AUDIT_LAZY=1 every one of these answers is also made the
long way, by the unchanged code, and a difference raises
`lazyaudit.LazyMismatch` (lazyaudit.py). The golden check, the fuzzer and the
suite are run once with it on.

What is NOT lazy: a world with a climate keeps its own gates (`_burn_cells`,
`_law_o2` and `_warm_cells` fall through to the full code there, because the
rest of a cell outside the box is not 20 C); a world with a gas flow keeps its
own law; a world with mixes never uses the matter memo.
"""
import functools
import types

import numpy as np

from . import heat as H
from . import lazyaudit as audit
from . import lazymemo as memo
from .air import _NO_MASK
from .kernels import HAVE_NUMBA, _cells_sum
from .fields import _BURN_FLOOR
from .kernels import _any_in
from .window import box_volume
from .materials import (AIR, AMBIENT, BOIL, FLUID, FREEZE, MELT, OIL, O2_PER_L, SOLID, TSHOCK, _IGN_ARR)

# The lowest temperature at which anything catches (a solid's skin or an oil). A cell at
# ambient is under it: the reason a quiet cell can be left unread by `_burn_asleep`.
_CATCH_MIN = min(float(np.min(_IGN_ARR)), float(FLUID[OIL][3]))
# A cell at ambient (20 C) neither melts nor boils: it is under the lowest melting point and
# under the boiling point. (False for some other table, and the proofs then never fire.)
_AMBIENT_INERT = AMBIENT < min(mt for mt, _f, _l in MELT.values()) and AMBIENT < BOIL

# THE LAWS THAT CANNOT WRITE MATTER (mat, smass, fl, fvol, scomp, fcomp). Every other
# `_law_*` is treated as a writer: the memo is off while it runs and ends when it does.
# Conduction and buoyancy move heat; the oxygen law moves oxygen; the pneumatic law
# moves pressure and smoke; the senses read. The audit mode is what checks this list.
# (Not the senses: `_law_perceive` looks round a person by taking the person's own cells
# off the lattice for a moment and putting them back, and the first golden check of this
# lane caught it: four cases moved `_matter_gen`.)
MATTER_PURE = frozenset((
    "_law_conduct", "_law_rise", "_law_o2", "_law_air",
    "_law_radiate", "_law_outdoor", "_law_ground", "_law_climate"))
# THE LAWS THAT DO NOT WRITE E (heat). The rest end the summary of the heat (`_hot`).
E_PURE = frozenset()
# A LAW THAT CAN PROVE IT HAS NOTHING TO DO: the method that says so (before it runs), and
# whether the law is then not called at all (SKIP) or called as it is (the proof keeps the memo).
PROOFS = {"_law_crack": "_lz_quiet_crack", "_law_boil": "_lz_quiet_boil",
          "_law_melt": "_lz_quiet_melt", "_law_burn": "_lz_quiet_burn",
          "_law_skin_water": "_lz_quiet_skin"}
SKIP = frozenset(("_law_crack", "_law_boil", "_law_melt", "_law_skin_water"))

_DUMMY_SMOKE = np.zeros((1, 1, 1), np.float32)      # see `_law_air`
_CHAINS = {}


def _plain_chain(cls, name):
    """Is `name` of this class the law of HeatMixin or SupportMixin and nothing else,
    apart from the memo-clearing wrappers of OutsideMixin? A proof of "this law has
    nothing to do" is the proof about THAT law, so a mixin that gives the world another
    `_law_boil` has none."""
    key = (cls, name)
    hit = _CHAINS.get(key)
    if hit is None:
        from .heat import HeatMixin
        from .support import SupportMixin
        from .wetmix import WetMixin
        from .outside_world import OutsideMixin
        owners = []
        for c in cls.__mro__:
            if c is LazyLawsMixin or c is OutsideMixin:
                continue                                     # (the memo wrappers and the gate)
            for d in (c.__dict__.get(name), c.__dict__.get("_lz_own", {}).get(name)):
                if (d is not None and not getattr(d, "_lz_wrapper", False)
                        and not d.__qualname__.startswith("_clears_memo")):
                    owners.append(c)
        hit = _CHAINS[key] = bool(owners) and all(o in (HeatMixin, SupportMixin, WetMixin) for o in owners)
    return hit


def _wrap(name, f):
    """`f`, with the memo told when it runs. A law that can write matter switches the
    matter memo off while it runs and arms it again when it ends; a law that writes E (or
    matter) ends the summary of the heat. A law with a PROOF that it has nothing to do this
    tick is, this tick, neither (and, for SKIP, is not run)."""
    writes = name not in MATTER_PURE
    heats = name not in E_PURE
    proof = PROOFS.get(name)

    @functools.wraps(f)
    def law(self, *a, **k):
        st = memo.get(self)
        if st is None or not st.live:
            return f(self, *a, **k)
        quiet = False
        if proof is not None and st.wdepth == 0 and self._lz_on() and _plain_chain(type(self), name):
            quiet = getattr(self, proof)()
            if quiet and name in SKIP:
                return None
        w = writes and not quiet
        if w:
            st.wdepth += 1
            st.mm = 0
        try:
            return f(self, *a, **k)
        finally:
            if w:
                st.wdepth -= 1
                if st.wdepth == 0 and st.live:
                    st.mm = 1
            if (w or (heats and not quiet)) and st.live:
                if st.skipbump:
                    st.skipbump = False
                else:
                    st.hot = None
    law._lz_wrapper = True
    return law


class _Hot:
    """What the cells that are not at rest say, for the laws that look at them (kept
    until a law that can change them has run; `LazyLawsMixin._hot`)."""
    __slots__ = ("ebox", "box", "T", "burn_asleep", "skin_quiet", "hot45")

    def sig(self):
        return (self.ebox, self.box, self.burn_asleep, self.skin_quiet, self.hot45,
                None if self.T is None else self.T)


class LazyLawsMixin:
    lazy_exact = False          # the gate: off here, on in OutsideMixin (cold, body heat)

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        # every `_law_*` this class resolves is wrapped on the class itself, so a law
        # defined in a mixin, in a subclass or in a test is told to the memo
        # from the OUTSIDE (a nested law is counted by `wdepth`)
        for name in dir(cls):
            if not name.startswith("_law_"):
                continue
            f = getattr(cls, name, None)
            if (isinstance(f, types.FunctionType) and not (name in MATTER_PURE and name in E_PURE)
                    and not getattr(f, "_lz_wrapper", False)):
                if cls.__dict__.get(name) is f:              # its own: kept, for `_plain_chain`
                    cls._lz_own = dict(cls.__dict__.get("_lz_own", {}), **{name: f})
                setattr(cls, name, _wrap(name, f))

    # ------------------------------------------------------------ the tick
    def _tick_start(self):
        super()._tick_start()
        st = memo.state(self)
        st.live, st.mm, st.wdepth, st.hot, st.skipbump = False, 0, 0, None, False

    def _after_raster(self):
        """The particles have landed on the lattice: from here the matter is
        written only by the laws that follow, and the memo may start."""
        super()._after_raster()
        if self.lazy_exact and self._windows_on():
            st = memo.state(self)
            st.live, st.mm, st.wdepth, st.hot, st.skipbump = True, 1, 0, None, False

    def _tick_end(self):
        st = memo.get(self)
        if st is not None:
            st.live, st.mm, st.wdepth, st.hot, st.skipbump = False, 0, 0, None, False
        super()._tick_end()

    # ------------------------------------------------------------ helpers
    def _lz_on(self):
        """Is the windowed answer allowed here? A lazy world, laws worked from
        windows, and no climate (a climate world's cells outside the box are at
        their own rest, not at 20 C: those keep the full code)."""
        return (self.lazy_exact and self._windows_on()
                and getattr(self, "climate", None) is None)

    def _lz_verified(self):
        """May a law be left out because a proof says it has nothing to do? Only when every
        ask that law would have made is a no-op: the matter has been compared since the last
        law that could change it (`mm == 2`) and the heat capacity is built, so the
        asks (the laws left out ask for the heat capacity and nothing else: never for the
        porosity) would neither find moved matter nor build a field. Anything else runs the law,
        asking where it always asked. (`_matter_gen` and the kept fields are state a golden
        check compares: a law that is skipped must not move an ask.)"""
        st = memo.get(self)
        return (st is not None and st.live and st.mm == 2 and not self.mixes_live
                and self._cap is not None)

    def _lz_hot_box(self):
        """The box round the cells with E != 0 (`_unsettled(0)`), or None."""
        return self._hot().ebox

    # ------------------------------------------------------------ the summary of the heat
    def _hot(self):
        """What the cells that are not at rest say, for the laws that look at them:
        their box, their temperatures, and the three yes-or-no answers the quiet laws
        need. Made once and kept (in a tick, in a lazy world) until a law that can
        write E or matter has ended."""
        st = memo.state(self)
        h = st.hot if st.wdepth == 0 else None           # (inside a law that can write: made afresh, never kept)
        if h is not None:
            if audit.enabled():
                audit.check("hot", True, lambda: self._same_hot(h, self._hot_make()))
            return h
        h = self._hot_make()
        if st.live and st.wdepth == 0:
            st.hot = h
        return h

    @staticmethod
    def _same_hot(a, b):
        s, t = a.sig(), b.sig()
        return (s[:5] == t[:5] and ((s[5] is None) == (t[5] is None))
                and (s[5] is None or s[5].tobytes() == t[5].tobytes()))

    def _hot_make(self):
        """The summary, worked out over the box round the heat and round the skin: a
        cell outside it has E == 0 and no skin, so it is at ambient exactly (20 C), which is
        under every threshold the quiet laws ask about."""
        cap = self.heat_capacity()                       # (what `T()` asks first: it notices moved matter)
        skin = getattr(self, "skin", None)               # (not made here: `_skin_field` makes it where the burn law does)
        if skin is not None and skin.shape != self.shape:
            skin = None
        h = _Hot()
        h.ebox = self._unsettled(0)[0]
        there = self._nz_window(skin, 0) if skin is not None else None
        box = h.ebox
        if there is not None:
            box = there if box is None else tuple(
                slice(min(a.start, c.start), max(a.stop, c.stop)) for a, c in zip(box, there))
        h.box = box
        h.T = None
        h.burn_asleep, h.skin_quiet, h.hot45 = True, True, False
        if box is not None:
            T = AMBIENT + self.E[box] / cap[box]
            h.T = T
            sk = skin[box] if skin is not None else 0.0     # (no skin field yet: no skin)
            hi = T + (np.abs(sk) if skin is not None else 0.0)
            mat = self.mat[box]
            warm = (hi >= _IGN_ARR[mat]) & (self.smass[box] > 0)
            oil = (self.fl[box] == OIL) & (self.fvol[box] > 0) & (hi >= FLUID[OIL][3])
            h.burn_asleep = not (warm.any() or oil.any())
            h.skin_quiet = not ((T + sk) > BOIL).any()
            h.hot45 = bool(((T > 45.0) & (mat == AIR)).any())
        return h

    # ------------------------------------------------------------ proofs of "nothing to do"
    def _lz_quiet_crack(self):
        """`_law_crack` starts with `np.isin(mat, TSHOCK).any()`: no brittle material
        anywhere, nothing to do. (The law's own test, asked here instead of there.)"""
        return self._lz_verified() and not np.isin(self.mat, list(TSHOCK)).any()

    def _lz_quiet_boil(self):
        """`_law_boil` can only act on a water cell hotter than the boiling point
        (the pressure only raises it): no cell is, so it acts on nothing. Not with mixes
        (a mixed liquid boils at its own point) and not where another law of this name
        stands in front (`_plain_chain`)."""
        if not self._lz_verified() or not _AMBIENT_INERT:
            return False
        h = self._hot()
        quiet = h.T is None or not (h.T > BOIL).any()
        return audit.proves("quiet_boil", quiet, self._boil_idle_full)

    def _boil_idle_full(self):
        """The long way: would `_law_boil` change a cell? (Its own tests.)"""
        from .materials import WATER
        wet = (self.fl == WATER) & (self.fvol > 0)
        if not wet.any():
            return True
        T = self.T()
        boil_at = BOIL + (1.0 * np.clip(self._upcell(self.pcell), 0.0, 18.0) if self.pcell is not None else 0.0)
        return not ((self.fl == WATER) & (self.fvol > 0) & (T > boil_at)).any()

    def _lz_quiet_melt(self):
        """`_law_melt` acts on a meltable solid above its melting point, or on a molten
        pool (a freezable fluid) below its setting point. Cells at ambient are under every
        melting point, so the first is read over the box round the heat; the second is
        the law's own question, asked of the fluid ids (`fl`): none anywhere. Not with mixes."""
        if not self._lz_verified() or not _AMBIENT_INERT:
            return False
        h = self._hot()
        T = h.T
        quiet = (not _any_in(np.ascontiguousarray(self.fl).reshape(-1), H._FREEZABLE_ARR)
                 and (T is None or not any((T > mt).any() for mt, _f, _l in MELT.values())))
        return audit.proves("quiet_melt", quiet, self._melt_idle_full)

    def _melt_idle_full(self):
        """The long way: would `_law_melt` change a cell? (Its own conditions, over the lattice.)"""
        from .materials import NOFLUID
        T = self.T()
        for m, (mt, fl_id, lat) in MELT.items():
            if (((self.mat == m) & (self.smass > 0) & (T > mt)
                 & ((self.fl == NOFLUID) | (self.fl == fl_id)))).any():
                return False
        for fl_id, (mt, m, lat) in FREEZE.items():
            if ((self.fl == fl_id) & (self.fvol > 0) & (T < mt - 2.0)
                    & ((self.mat == AIR) | (self.mat == m))).any():
                return False
        return True

    def _lz_quiet_burn(self):
        """`_law_burn` writes no matter when no skin is past the boil (`_law_skin_water`)
        and nothing can catch (`_burn_cells`)."""
        if _CATCH_MIN <= AMBIENT or not self._lz_verified():
            return False
        h = self._hot()
        return h.skin_quiet and h.burn_asleep

    # ------------------------------------------------------------ where it burns
    def _lz_burning(self):
        """`burning()` read from the summary of the heat: the box round the heat and the
        skin, and the temperatures in it, which the quiet laws have just made (the cells
        outside it burn nothing, as the law's own comment says). None where the old code
        must run: a climate, a world with no skin field yet, a box that is most of the world."""
        s = getattr(self, "skin", None)
        if (not self._lz_on() or s is None or s.shape != self.shape or _BURN_FLOOR <= AMBIENT):
            return None
        h = self._hot()
        if h.ebox is not None and box_volume(h.ebox) * 2 > self.E.size:
            return None
        out = np.zeros(self.shape, bool)
        box = h.box
        if box is not None:
            T = h.T + s[box]
            fl, mat, sm = self.fl[box], self.mat[box], self.smass[box]
            here = (fl == OIL) & (self.fvol[box] > 0) & (T >= FLUID[OIL][3])
            for m, spec in SOLID.items():
                if spec[3] is not None:
                    here |= (mat == m) & (sm > 0) & (T >= spec[3])
            out[box] = here
        return audit.check("burning", out, self._burning_scan)

    # ------------------------------------------------------------ burning
    def _lz_quiet_skin(self):
        """`_law_skin_water` holds a wet skin at the boil: with no skin past the boil there
        is nothing to hold (the law's own test, `Ts > BOIL`, asked over the box round the
        heat and the skin; an unwet cell is stricter still)."""
        if not self._lz_verified() or not _AMBIENT_INERT:
            return False
        quiet = self._hot().skin_quiet
        return audit.proves("skin_water", quiet, self._skin_water_idle_full)

    def _skin_water_idle_full(self):
        """The long way: would `_law_skin_water` change a cell?"""
        from .materials import WATER
        wet = (self.fl == WATER) & (self.fvol > 0) & (self.mat != AIR) & (self.smass > 0)
        if not wet.any():
            return True
        geo = self._surface_geometry()
        skin = self._skin_field()
        at = np.nonzero(wet & geo.has & (skin > 0))
        if not len(at[0]):
            return True
        Ts = AMBIENT + self.E[at] / geo.C[at] + skin[at]
        return not (Ts > BOIL).any()

    def _burn_cells(self):
        if self._lz_on() and _CATCH_MIN > AMBIENT and self._hot().burn_asleep:
            self._skin_field()                           # (the law makes it on its first line)
            audit.check("burn_asleep", True, self._burn_asleep_full)
            # the quiet exit of the law, line for line
            self._hrr = (self.tick, 0.0)
            H._FIRES[self] = (self.tick, None)
            self._blasts = []
            return
        return super()._burn_cells()

    def _burn_asleep_full(self):
        """The test `_burn_cells` starts with, over the whole lattice."""
        T0 = self.T()
        hi = T0 + np.abs(self._skin_field())
        warm = (hi >= _IGN_ARR[self.mat]) & (self.smass > 0)
        return not (warm.any()
                    or ((self.fl == OIL) & (self.fvol > 0) & (hi >= FLUID[OIL][3])).any())

    # ------------------------------------------------------------ oxygen
    def _law_o2(self):
        if self._lz_on() and self.skip_quiet:
            self.heat_capacity()                         # T() asks it,
            self.porosity()                              # and the law asks this before its gate
            if self._o2_asleep_win():
                memo.state(self).skipbump = True         # it changed nothing: the heat summary stands
                return
        return super()._law_o2()

    def _o2_asleep_win(self):
        """`_o2_asleep(T)` with T read over the box round the heat only: a cell
        outside it is at ambient exactly, and 20 C is not above 45."""
        fresh = np.float32(O2_PER_L * self.vox_l)
        o2 = self.o2
        if o2.min() != fresh or o2.max() != fresh:
            asleep = False
        elif self.smoke.any():
            asleep = False
        else:
            asleep = not self._hot().hot45
        return audit.check("o2_asleep", asleep, lambda: self._o2_asleep(self.T()))

    # ------------------------------------------------------------ the pneumatic law
    def _warm_cells(self):
        if not self._lz_on() or not self._cells_fast(self.E):
            return super()._warm_cells()
        cs = self._CS
        cap = self.heat_capacity()
        cshape = tuple((s + cs - 1) // cs for s in self.shape)
        out = np.zeros(cshape, np.float32)
        box = self._hot().ebox
        if box is not None:
            lo = [(b.start // cs) * cs for b in box]
            sub = tuple(slice(l, min(-(-b.stop // cs) * cs, n))
                        for l, b, n in zip(lo, box, self.shape))
            T = AMBIENT + self.E[sub] / cap[sub]
            gas = self.mat[sub] == AIR
            warm = np.where(gas, np.clip(T - AMBIENT, 0.0, 500.0), 0.0).astype(np.float32)
            c = _cells_sum(warm, _NO_MASK, False, False, cs, True)
            out[tuple(slice(l // cs, l // cs + n) for l, n in zip(lo, c.shape))] = c
        return audit.check("warm_cells", out, lambda: self._warm_cells_full())

    def _warm_cells_full(self):
        """The definition (`AirMixin._warm_cells`), over the whole lattice."""
        gas = self.mat == AIR
        return self._cells(np.where(gas, np.clip(self.T() - AMBIENT, 0.0, 500.0), 0.0).astype(np.float32))

    def _cells(self, arr, how="mean"):
        if self.lazy_exact:
            if how == "mean" and arr is self._por and arr is not None:
                # the porosity array is handed out as one object until matter moves, and
                # read only; its coarse mean changes only with it
                st = memo.state(self)
                hit = st.cpor
                if hit is not None and hit[0] is arr:
                    return audit.check("cells_por", hit[1], lambda: super(LazyLawsMixin, self)._cells(arr, how))
                out = super()._cells(arr, how)
                st.cpor = (arr, out)
                return out
            if how == "sum" and arr is self.p_add and not arr.view(np.uint8).any():
                # nothing was injected: the block sums are +0.0 and so is this
                zero = np.zeros(tuple((s + self._CS - 1) // self._CS for s in arr.shape), arr.dtype)
                return audit.check("cells_padd", zero, lambda: super(LazyLawsMixin, self)._cells(arr, how))
        return super()._cells(arr, how)

    def _law_air(self):
        """The pneumatic law, with no smoke to carry left uncarried. The wind
        carries SMOKE only, and carrying a field that is zero in every bit
        changes no bit of it, but the fused kernel sweeps the whole lattice three
        times to learn that (about 14 ms at 1.6 M). So when the smoke array is all
        zero the law is handed a one-cell stand-in for it: the kernel then
        has no face to work on, and the real array is put back. (The clean form is
        a test in air.py; see the S4-3b report, needs_other_files.)"""
        sm = self.smoke
        if (self._lz_on() and not self.gas_flow and self.fused and HAVE_NUMBA
                and sm is not None and sm.dtype == np.float32 and self.pcell is not None
                and self.vcell[0].dtype == np.float32 and self.porosity().dtype == np.float32
                and not sm.view(np.uint8).any()):
            if audit.enabled():
                return self._law_air_audited(sm)
            self.smoke = _DUMMY_SMOKE
            try:
                return super()._law_air()
            finally:
                self.smoke = sm
        return super()._law_air()

    def _law_air_audited(self, sm):
        """Audit: the lazy law and the full one from the same start; every
        array the law writes must come out the same."""
        names = ("pcell", "vcell", "p_add")

        def keep():
            return (None if self.pcell is None else self.pcell.copy(),
                    None if self.vcell is None else [v.copy() for v in self.vcell],
                    self.p_add.copy())

        def put(s):
            self.pcell = None if s[0] is None else s[0].copy()
            self.vcell = None if s[1] is None else [v.copy() for v in s[1]]
            self.p_add[...] = s[2]

        start = keep()
        self.smoke = _DUMMY_SMOKE
        try:
            super()._law_air()
        finally:
            self.smoke = sm
        lazy = keep()
        put(start)
        super()._law_air()
        full = keep()
        audit.check("law_air", (lazy[0], lazy[1], lazy[2], sm), lambda: (full[0], full[1], full[2], sm))
        return None
