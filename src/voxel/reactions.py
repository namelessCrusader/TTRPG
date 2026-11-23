"""THE REACTION AND SORPTION ROW RUNNER (wave 15, S1 slice 1).

The row engine (rxn.py) is pure data: a Row, its rate, drivers, gates and books. This file
is the ONE adapter that lets a World run those rows on its lattice cells. There is no craft
word here: lime, tanning and the rest are rows in rxn.catalogue(); a scene lists the row ids
it wants (`set_reactions`) and the runner does the rest.

WHERE A CELL'S CHEMISTRY LIVES. A cell holds up to two LUMPS in the mix table (mixtable.py):
the SOLID lump (row `scomp`, or the pure solid `mat`; grams `smass`) and the FLUID lump (row
`fcomp`, or the pure fluid 64 + `fl`; ml `fvol`, so grams = ml x rho / 1000). A row name of
rxn.py ("CaO", "tannin_aq", "H2O") maps to a place and a form in NAMES: S solid lump, F fluid
lump, G the gas beside the cell (the GasStore). A cell is a candidate for a row when the
row's key reactant is present in its lump (a cached look-up per mix row id, so a bronze bar
of 10,000 cells costs one entry). Nothing outside the active set is touched.

THE ADAPTER. `LatticeState` implements the state protocol rxn.extent and rxn.advance use
(get, avail, room, add, heat, p, T, aw, area, made, chem). The row engine runs UNCHANGED.

THE PENDING STORE (the sub-floor rule). A mix holds a species only above about 6e-8 of a
lump (2^-24, the quantum), and a float32 cell has an ulp. A slow row (or a slow diffusion)
that moves less than that per step would round to nothing, or worse, change a lump by
rounding and book the rest to "quant". So every change goes to a float64 PENDING array first
(per place, species, form and cell; signed). The laws READ stored + pending, so the physics
sees the true amount. A pending amount is committed to the lump only when it reaches a
threshold, 1024 quanta of the lump (2^-14, so the table's own rounding of about one
quantum per edit is under 0.2 percent of what it commits) and 16 float32 ulp of its mass (the bite rule of
transport.py), and then in one edit. Pending is MASS: it is in `rx_mass_g()`, so total
grams = lumps + gas + pending + the quant and f32 ledgers is exact. Heat has its own pending
(pend_E, in the world's E units) for the same reason (float32 E).

THE ONE CLOCK AND THE ONE STEP ORDER. `ChemClockMixin` owns the box clock for BOTH this runner
and the Fick kernel of fick_solids.py: one method, `_chem_box(dt)`, runs, in this fixed order,
half a Fick step (arrival), the rows (reaction), half a Fick step (Strang). There is no second
clock. The clock sleeps for N ticks chosen so the largest bite reaches its commit threshold
(bite_clock), at least one second (EVERY ticks) and at most one hour; with nothing active it
re-tests once a second (one boolean gather over the lattice), and with the gates off it does
nothing at all (the world is the old world bit for bit).

THE BOOKS. Reactants fall, products rise, the heat dH xi goes to the cell's E and to the
`chem_J` ledger, so (sum E + pending E) minus chem_J is constant under the runner. A gas that
has no cell to go to (no open face) is booked in `gas_gone`, never dropped.

THE GATE. `w.rx` is None on every world (a CLASS attribute, so an old snapshot restores and
golden sees nothing). `set_reactions(rows=[ids])` makes it; None means the old world.
Mixin, no World subclass: the merge agent adds `ReactionMixin` and `FickSolidsMixin` to
World's bases (until then scenes_reactions.make_world composes them).

Sources: Langmuir 1918, J Am Chem Soc 40, 1361 (k); Levenspiel, Chemical Reaction
Engineering (k); Covington, Tanning Chemistry, RSC 2009 (k); slaking -63.7 kJ/mol (Hess from
NBS heats of formation, rxn.check, 2 %); calcine p_eq per Stanmore and Gilot 2005 (u).
"""
import math

import numpy as np

from . import rxn
from . import species as SP
from .gasstore import GasStore, gas_volume
from .materials import AMBIENT, TICK_S
from .matterdata import BACKGROUND, PHASE
from .species_seed import BOUND, DISSOLVED, FLUID_ROWS, GRAIN, MIX_BASE
from .transport import bite_clock

EVERY = 40                      # ticks of the shortest box step (1 s)
CAP_TICKS = int(3600 / TICK_S)  # the longest sleep: one hour
QFLOOR = 2.0 ** -14             # 1024 quanta of a mix lump (the commit threshold, fraction):
                                # a commit's own rounding is about 1-2 quanta, so a commit at
                                # this size is right to about 0.2 percent
N_ULP = 16                      # the bite rule: a commit is at least 16 float32 ulp
SOAKED = 0.05                   # share of a cell's volume in liquid above which its water activity
                                # is 1 (a soaked cell); below it the cell takes the air's humidity (u)
TINY_G = 1.0e-12                # absolute floor of a commit, grams

# rxn.py name -> (place, species name, form). Place S solid lump, F fluid lump, G the gas
# store (name = the store's key). A name not here makes set_reactions refuse the row.
NAMES = {
    "H2O": ("F", "water", DISSOLVED),
    "tannin_aq": ("F", "tannin", DISSOLVED),
    "CaO": ("S", "quicklime", BOUND),
    "Ca(OH)2": ("S", "slaked_lime", GRAIN),
    "CaCO3": ("S", "calcite", BOUND),
    "tannin_b": ("S", "tannin", BOUND),
    "collagen": ("S", "collagen", BOUND),
    "CO2": ("G", "CO2", None),
    "CO": ("G", "CO", None),
    "H2O_g": ("G", "H2O", None),
}
FLUID0 = FLUID_ROWS[0]          # row 64: the pure fluids start here


def _loc(name):
    p, sp, fo = NAMES[name]
    return (p, SP.BY_NAME[sp] if p != "G" else sp, fo)


def row_supported(row):
    """True when every name the row uses has a place and its key reactant is not a gas."""
    names = [s for s, _g in row.reactants + row.products]
    return all(n in NAMES for n in names) and NAMES[row.key][0] != "G"


class ChemBooks:
    """The books both gates share (the runner and the Fick kernel): the pending store, the
    chemical heat, the gas that had no cell. Plain python and numpy: it copies with the world."""

    def __init__(self):
        self.pend = {}               # (place, species, form) -> float64 array (world shape)
        self.pend_E = None           # float64 array (world shape), in E units
        self.chem_J = 0.0            # chemical heat released, J (the energy identity)
        self.gas_gone = {}           # gas name -> grams made with no cell to take them
        self.store = None            # an own GasStore when the world has no wet store


class RxState:
    """What a world with the runner holds: the rows and what they have done."""

    def __init__(self, rows):
        self.rows = list(rows)
        self.extents = {}            # row id -> total grams of key reacted
        self.lut = {}                # (row id, place) -> bool per mix row id
        self.runs = 0
        self.cells = 0


class ChemClock:
    """The shared box clock (one per world)."""

    def __init__(self, tick):
        self.last = tick             # the tick this box last ran up to
        self.due = tick + EVERY      # run when the world's tick reaches this
        self.done = -1
        self.dt_max = 300.0          # s, the longest single step of the closed-explicit rows
        self.runs = 0


# ---- gathering the lumps of some cells ------------------------------------------------
class LumpView:
    """The solid and fluid lumps, the energy and the gas neighbour of a set of cells
    (flat indices `idx`), with the PENDING store folded in. Writes go to `delta` and are
    committed by `commit()`."""

    def __init__(self, w, idx, gas=True):
        self.w, self.idx = w, np.asarray(idx, np.int64)
        n = len(self.idx)
        self.n = n
        sh = w.shape
        self.ix = tuple(np.unravel_index(self.idx, sh))
        mx = w.ensure_mixes()
        self.mx = mx
        mat = w.mat.reshape(-1)[self.idx]
        sc = w.scomp.reshape(-1)[self.idx].astype(np.int64)
        self.sid = np.where(sc > 0, sc, mat.astype(np.int64))
        self.sm = np.where(mat > 0, w.smass.reshape(-1)[self.idx].astype(np.float64), 0.0)
        self.sid = np.where(self.sm > 0, self.sid, 0)
        fv = w.fvol.reshape(-1)[self.idx].astype(np.float64)
        fc = w.fcomp.reshape(-1)[self.idx].astype(np.int64)
        fl = w.fl.reshape(-1)[self.idx].astype(np.int64)
        self.fid = np.where(fv > 0, np.where(fc > 0, fc, FLUID0 + fl), 0)
        self.rho_f = np.zeros(n)
        ok = self.fid > 0
        if ok.any():
            self.rho_f[ok] = np.nan_to_num(mx.prop("rho", self.fid[ok]).astype(np.float64))
        self.fm = np.where(ok, fv * self.rho_f / 1000.0, 0.0)
        self.fid = np.where(self.fm > 0, self.fid, 0)
        self.E0 = w.E.reshape(-1)[self.idx].astype(np.float64)
        self.delta = {}              # (place, sp, form) -> float64 array, uncommitted
        self.dE = np.zeros(n)
        self._ent = {}
        self._gram = {}
        self.cb = w._chem_books()
        self._gas(gas)

    # -- the gas beside each cell
    def _gas(self, look=True):
        """The first open gas cell beside each cell, and the count of open faces."""
        w = self.w
        nx, ny, nz = w.shape
        x, y, z = self.ix
        self.nbr = np.full(self.n, -1, np.int64)
        self.nopen = np.zeros(self.n, np.int64)
        self.dx = (w.vox_l / 1000.0) ** (1.0 / 3.0)
        if not look:
            return
        matf, fvf = w.mat.reshape(-1), w.fvol.reshape(-1)
        for a, b, c in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            X, Y, Z = x + a, y + b, z + c
            ok = (X >= 0) & (X < nx) & (Y >= 0) & (Y < ny) & (Z >= 0) & (Z < nz)
            f = np.where(ok, (np.clip(X, 0, nx - 1) * ny + np.clip(Y, 0, ny - 1)) * nz
                         + np.clip(Z, 0, nz - 1), 0)
            g = ok & (matf[f] == 0) & (fvf[f] < 0.9 * w.cap)
            self.nopen += g
            take = g & (self.nbr < 0)
            self.nbr[take] = f[take]

    @property
    def has_gas(self):
        return self.nbr >= 0

    def store(self):
        w = self.w
        if w.wet is not None:
            return w.wet.store
        if self.cb.store is None:
            self.cb.store = GasStore(w.shape, w.vox_l)
        return self.cb.store

    # -- grams of a species in a form in a lump (summed over size classes)
    def _entries(self, place):
        e = self._ent.get(place)
        if e is None:
            ids = self.sid if place == "S" else self.fid
            e = self._ent[place] = self.mx.entries(ids)
        return e

    def stored(self, place, sp, form):
        k = (place, sp, form)
        g = self._gram.get(k)
        if g is None:
            if place == "G":
                st = self.store()
                nb = np.maximum(self.nbr, 0)
                ex = st.excess(sp).reshape(-1)[nb].astype(np.float64)
                g = np.where(self.nbr >= 0, ex, 0.0)
            else:
                e = self._entries(place)
                m = (e.species == sp) & (e.form == form)
                mass = self.sm if place == "S" else self.fm
                g = np.bincount(e.cell[m], weights=e.frac[m] * mass[e.cell[m]],
                                minlength=self.n)
            self._gram[k] = g
        return g

    def pending(self, k):
        a = self.cb.pend.get(k)
        return 0.0 if a is None else a.reshape(-1)[self.idx]

    def grams(self, k):
        """stored + pending + uncommitted change of one (place, species, form)."""
        d = self.delta.get(k)
        return self.stored(*k) + self.pending(k) + (0.0 if d is None else d)

    def add(self, k, dg):
        d = self.delta.get(k)
        self.delta[k] = (0.0 if d is None else d) + np.asarray(dg, np.float64)

    # -- the cell's temperature and the free gas volume
    def T(self):
        w = self.w
        sc, fc = (None, None)
        if w.mixes_live:
            sc = w.scomp.reshape(-1)[self.idx]
            fc = w.fcomp.reshape(-1)[self.idx]
        C = w._cap_of(w.mat.reshape(-1)[self.idx], w.smass.reshape(-1)[self.idx],
                      w.fl.reshape(-1)[self.idx], w.fvol.reshape(-1)[self.idx], sc, fc)
        pe = self.cb.pend_E
        E = self.E0 + self.dE + (0.0 if pe is None else pe.reshape(-1)[self.idx])
        return AMBIENT + E / np.asarray(C, np.float64)

    def gas_vol(self):
        w = self.w
        fv = np.where(self.nbr >= 0, w.fvol.reshape(-1)[np.maximum(self.nbr, 0)],
                      w.fvol.reshape(-1)[self.idx])
        return gas_volume(w.vox_l, fv)

    # -- thresholds and the commit
    def _thr(self, place, mass):
        """The smallest change a lump of `mass` grams can hold: 4 quanta, 16 float32 ulp."""
        m32 = np.asarray(mass, np.float32)
        return np.maximum(np.maximum(QFLOOR * mass, N_ULP * np.spacing(m32).astype(np.float64)),
                          TINY_G)

    def commit(self, force=False):
        """Fold `delta` into pending and commit what reached its threshold, in one edit
        per lump. Returns the largest committed-or-held bite as a per-cell max of
        delta / threshold (for the clock)."""
        w, cb = self.w, self.cb
        shape = w.shape
        worst = 0.0
        biggest = 0.0
        for k, d in self.delta.items():
            if k not in cb.pend:
                cb.pend[k] = np.zeros(shape)
            P = cb.pend[k].reshape(-1)
            tot = P[self.idx] + d
            place = k[0]
            if place == "S":
                thr = self._thr("S", self.sm + np.maximum(tot, 0.0))
            elif place == "F":
                thr = self._thr("F", self.fm + np.maximum(tot, 0.0))
            else:
                nb = np.maximum(self.nbr, 0)
                ex = self.store().excess(k[1]).reshape(-1)[nb]
                thr = np.maximum(N_ULP * np.spacing(np.asarray(ex, np.float32)).astype(np.float64),
                                 TINY_G)
            go = (np.abs(tot) >= thr) | force
            P[self.idx] = np.where(go, 0.0, tot)
            self.delta[k] = np.where(go, tot, 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                biggest = max(biggest, float(np.max(np.abs(d) / thr)) if self.n else 0.0)
        self._write()
        # whatever a commit could not apply (a lump emptied by rounding) stays pending
        return biggest

    def _write(self):
        w = self.w
        for place in ("S", "F"):
            items = [(k, d) for k, d in self.delta.items() if k[0] == place and np.any(d != 0.0)]
            if items:
                self._edit_lump(place, items)
        gas = [(k, d) for k, d in self.delta.items() if k[0] == "G" and np.any(d != 0.0)]
        for k, d in gas:
            self._gas_commit(k, d)
        # energy: pending E in E units, committed at 16 ulp of float32 E
        if np.any(self.dE != 0.0):
            cb = self.cb
            if cb.pend_E is None:
                cb.pend_E = np.zeros(w.shape)
            PE = cb.pend_E.reshape(-1)
            tot = PE[self.idx] + self.dE
            Ef = w.E.reshape(-1)
            cur = Ef[self.idx]
            thr = N_ULP * np.spacing(np.asarray(np.abs(cur) + 1.0, np.float32)).astype(np.float64)
            if w.E.dtype == np.float64:
                thr = np.zeros_like(thr)
            go = np.abs(tot) >= thr
            Ef[self.idx] = np.where(go, cur.astype(np.float64) + tot, cur).astype(w.E.dtype)
            PE[self.idx] = np.where(go, 0.0, tot)
            self.dE[:] = 0.0

    def _edit_lump(self, place, items):
        w, mx = self.w, self.mx
        ids = (self.sid if place == "S" else self.fid).astype(np.int32)
        mass = (self.sm if place == "S" else self.fm).copy()
        ent = self._entries(place)
        cells, keys, dgs = [], [], []
        for (pl, sp, fo), d in items:
            pos = d > 0
            if pos.any():
                s = SP.SPECIES[sp]
                sz = (s.grain_sz if (fo == GRAIN and s.grain_sz > 0) else 0)
                c = np.flatnonzero(pos)
                cells.append(c)
                keys.append(np.full(len(c), SP.pack_key(sp, fo, sz), np.uint32))
                dgs.append(d[c])
            neg = d < 0
            if neg.any():
                m = (ent.species == sp) & (ent.form == fo)
                have = np.bincount(ent.cell[m], weights=ent.frac[m] * mass[ent.cell[m]],
                                   minlength=self.n)
                share = np.where(have[ent.cell[m]] > 0,
                                 ent.frac[m] * mass[ent.cell[m]] / np.where(have[ent.cell[m]] > 0,
                                                                             have[ent.cell[m]], 1.0), 0.0)
                cc = ent.cell[m]
                keep = neg[cc]
                cells.append(cc[keep])
                keys.append(ent.key[m][keep])
                dgs.append((d[cc] * share)[keep])
        if not cells or sum(len(c) for c in cells) == 0:
            return
        cell_i = np.concatenate(cells)
        key = np.concatenate(keys)
        dg = np.concatenate(dgs)
        i2, m2, applied = mx.apply(ids, mass, cell_i, key, dg)
        # what a clamp refused (a reactant a hair short) goes back to pending, not to nowhere
        short = dg - applied
        if np.any(np.abs(short) > 0.0):
            # attribute the refusal to its (species, form) pending entry
            sp_of = np.asarray(SP.unpack_key(key)[0], np.int64)
            fo_of = np.asarray(SP.unpack_key(key)[1], np.int64)
            for (pl, sp, fo), _d in items:
                sel = (sp_of == sp) & (fo_of == fo)
                if sel.any():
                    P = self.cb.pend.setdefault((pl, sp, fo), np.zeros(w.shape)).reshape(-1)
                    np.add.at(P, self.idx[cell_i[sel]], short[sel])
        changed = np.flatnonzero((i2 != ids) | (m2 != mass))
        if len(changed) == 0:
            return
        ix = tuple(a[changed] for a in self.ix)
        ni, nm = i2[changed].astype(np.int64), m2[changed]
        if place == "S":
            mx.store(w.smass, ix, ni, nm)
            pure = ni < FLUID0
            fl_ix = tuple(a[~pure] for a in ix)
            if pure.any():
                px = tuple(a[pure] for a in ix)
                w.scomp[px] = 0
                w.mat[px] = ni[pure]
            if (~pure).any():
                old_mat = w.mat[fl_ix].copy()
                w.set_solid(fl_ix, ni[~pure])
                # a mix with no solid view of its own (collagen, tannin) keeps the cell's old
                # mat: the cell stays solid (mat 0 would read as air)
                w.mat[fl_ix] = np.where(w.mat[fl_ix] == 0, old_mat, w.mat[fl_ix])
                w.mixes_live = True
            gone = nm <= 0.0
            if gone.any():
                gx = tuple(a[gone] for a in ix)
                w.scomp[gx] = 0
                w.mat[gx] = 0
        else:
            rho = np.nan_to_num(mx.prop("rho", np.maximum(ni, 1)).astype(np.float64))
            rho = np.where(ni > 0, rho, 1.0)
            ml = np.where(nm > 0, nm * 1000.0 / np.where(rho > 0, rho, 1.0), 0.0)
            m32 = ml.astype(np.float32)
            diff = (ml - m32.astype(np.float64)) * rho / 1000.0
            e2 = mx.entries(ni)
            mx.book("f32", e2.species, e2.frac * diff[e2.cell])
            w.fvol[ix] = m32
            pure = (ni >= FLUID0) & (ni < MIX_BASE)
            lone = ni >= MIX_BASE
            if pure.any():
                px = tuple(a[pure] for a in ix)
                w.fcomp[px] = 0
                w.fl[px] = ni[pure] - MIX_BASE // 2
            if lone.any():
                w.set_fluid(tuple(a[lone] for a in ix), ni[lone])
                w.mixes_live = True
            gone = (ml <= 0.0)
            if gone.any():
                gx = tuple(a[gone] for a in ix)
                w.fcomp[gx] = 0
                w.fl[gx] = 0
                w.fvol[gx] = 0.0

    def _gas_commit(self, k, d):
        st = self.store()
        name = k[1]
        has = self.nbr >= 0
        pos = np.flatnonzero(has & (d != 0.0))
        if len(pos):
            nb = self.nbr[pos]
            ix = tuple(np.unravel_index(nb, self.w.shape))
            V = gas_volume(self.w.vox_l, self.w.fvol.reshape(-1)[nb])
            got = np.zeros(len(pos))
            for v in np.unique(V):                    # the store takes one volume per call
                m = V == v
                got[m] = st.add(name, tuple(a[m] for a in ix), d[pos][m], float(v))
            short = d[pos] - got
            if np.any(short != 0.0):
                P = self.cb.pend.setdefault(k, np.zeros(self.w.shape)).reshape(-1)
                np.add.at(P, self.idx[pos], short)
        lost = (~has) & (d != 0.0)
        if lost.any():
            g = self.cb.gas_gone
            g[name] = g.get(name, 0.0) + float(d[lost].sum())


# ---- the state protocol of rxn.py on the lattice ---------------------------------------
class LatticeState:
    """The dict-view protocol of rxn.ArrayState, on the cells of a LumpView."""

    def __init__(self, view):
        self.v = view
        n = view.n
        self.shape = (n,)
        self.x = None
        self.made = {"water": np.zeros(n), "latent": np.zeros(n)}
        self.chem = np.zeros(n)
        self._T = None
        w = view.w
        rh = getattr(w, "air_rh", None)
        wet = w.fvol.reshape(-1)[view.idx] > SOAKED * w.cap      # a film is not a soak
        self.aw = np.where(wet, 1.0, 0.5 if rh is None else float(rh))
        self.area = view.nopen * view.dx ** 2
        self.V = view.gas_vol()

    @property
    def T(self):
        if self._T is None:
            self._T = self.v.T()
        return self._T

    def _k(self, name):
        return _loc(name)

    def get(self, name):
        return self.v.grams(_loc(name))

    def avail(self, name):
        g = self.get(name)
        if PHASE.get(name) == "g":
            g = g + BACKGROUND.get(_loc(name)[1], 0.0) * self.V
        return g

    def room(self, name):
        return rxn.sorb_room(self, name)

    def conc(self, name, dg=0.0):
        sp = _loc(name)[1]
        return BACKGROUND.get(sp, 0.0) + (self.get(name) + dg) / self.V

    def p(self, name, dg=0.0):
        return self.conc(name, dg) / rxn.M[name] * rxn.R * rxn.T_REF_GAS

    def add(self, name, dg):
        self.v.add(_loc(name), dg)

    def heat(self, J):
        self.v.dE = self.v.dE + J

    def book(self, water=0.0, latent=0.0):
        self.made["water"] = self.made["water"] + water
        self.made["latent"] = self.made["latent"] + latent


# ---- the mixin --------------------------------------------------------------------------
class ChemClockMixin:
    """The box clock and the one step order, shared by ReactionMixin and FickSolidsMixin."""
    rx = None                     # the runner state (None = off: the old world)
    fk = None                     # the Fick kernel state (fick_solids.py)
    chemclk = None
    cb = None                     # the shared books (ChemBooks)

    def _chem_on(self):
        return self.rx is not None or self.fk is not None

    def _chem_books(self):
        if self.cb is None:
            self.cb = ChemBooks()
        return self.cb

    def _chem_clock(self):
        if self.chemclk is None:
            self.chemclk = ChemClock(self.tick)
        return self.chemclk

    def _tick_end(self):
        super()._tick_end()
        if self.rx is None and self.fk is None:
            return
        c = self._chem_clock()
        if c.done == self.tick or self.tick < c.due:
            return
        c.done = self.tick
        dt = (self.tick + 1 - c.last) * TICK_S
        c.last = self.tick + 1
        wait = self._chem_box(dt)
        ticks = int(np.clip(math.ceil(wait / TICK_S), EVERY, CAP_TICKS)) if wait > 0 else EVERY
        c.due = self.tick + ticks

    def _chem_box(self, dt):
        """ONE box step of dt seconds, in the one order: half Fick, rows, half Fick.
        Returns the seconds the box may sleep (0 = nothing moving)."""
        waits = []
        if self.fk is not None:
            waits.append(self._fick_pass(0.5 * dt))
        if self.rx is not None:
            waits.append(self._rx_pass(dt))
        if self.fk is not None:
            waits.append(self._fick_pass(0.5 * dt))
        waits = [x for x in waits if x and x > 0]
        return min(waits) if waits else 0.0

    def chem_run(self, seconds, dt=None):
        """Advance the chemistry (and the Fick kernel) by `seconds` WITHOUT ticking the world:
        the slow-clock twin used by scenes and tests for days of tanning. Steps of `dt`
        (default the clock's dt_max, 300 s)."""
        c = self._chem_clock()
        dt = c.dt_max if dt is None else dt
        t = 0.0
        while t < seconds - 1e-9:
            h = min(dt, seconds - t)
            self._chem_box(h)
            t += h
        c.runs += 1

    # -- the books (every gram in one place)
    def rx_mass_g(self, by_species=False):
        """Grams of every species in the world: lumps + gas + pending + gas that had no cell.
        (The quant and f32 ledgers are in `self.mixes.ledger`.)"""
        mx = self.ensure_mixes()
        out = {}
        s = (self.smass.reshape(-1) > 0) & (self.mat.reshape(-1) > 0)
        sid = np.where(self.scomp.reshape(-1) > 0, self.scomp.reshape(-1), self.mat.reshape(-1))
        for ids, m in ((sid[s], self.smass.reshape(-1)[s].astype(np.float64)),):
            for sp, g in mx.totals(ids.astype(np.int64), m).items():
                out[sp] = out.get(sp, 0.0) + g
        f = self.fvol.reshape(-1) > 0
        fid = np.where(self.fcomp.reshape(-1) > 0, self.fcomp.reshape(-1),
                       FLUID0 + self.fl.reshape(-1).astype(np.int64))[f]
        fm = (self.fvol.reshape(-1)[f].astype(np.float64)
              * mx.prop("rho", fid.astype(np.int64)).astype(np.float64) / 1000.0)
        for sp, g in mx.totals(fid.astype(np.int64), fm).items():
            out[sp] = out.get(sp, 0.0) + g
        return out if by_species else sum(out.values())


class ReactionMixin(ChemClockMixin):
    """The reaction and sorption row runner. Off (rx None) the world is the old world."""

    def set_reactions(self, rows=None):
        """Turn the runner on for the catalogue row ids (or Row objects) in `rows`
        (default: none, which still makes the state). A row with a name the adapter does not
        place (an O2 or glucose row) is refused. Returns the RxState."""
        cat = rxn.catalogue()
        out = []
        for r in (rows or []):
            row = cat[r] if isinstance(r, str) else r
            if not row_supported(row):
                raise ValueError("row %s uses a name the lattice adapter does not place" % row.id)
            out.append(row)
        self.ensure_mixes()
        self.rx = RxState(out)
        self._chem_books()
        c = self._chem_clock()
        c.last, c.due = self.tick, self.tick + EVERY
        return self.rx

    def reactions_off(self):
        self.rx = None

    def rx_pending_g(self, species=None):
        """Grams held in the pending store (signed), total or of one species name."""
        tot = 0.0
        if self.cb is None:
            return tot
        for (pl, sp, fo), a in self.cb.pend.items():
            if pl == "G":
                continue
            if species is None or species == SP.SPECIES[sp].name:
                tot += float(a.sum())
        return tot

    def _rx_lut(self, row, place):
        rx, mx = self.rx, self.mixes
        n = mx.n_rows
        key = (row.id, place)
        lut = rx.lut.get(key)
        if lut is None or len(lut) < n:
            start = 0 if lut is None else len(lut)
            ent = mx.entries(np.arange(start, n))
            _p, sp, fo = _loc(row.key)
            m = (ent.species == sp) & (ent.form == fo)
            new = np.zeros(n - start, bool)
            new[ent.cell[m]] = True
            lut = new if lut is None else np.concatenate([lut, new])
            rx.lut[key] = lut
        return lut

    def _rx_pass(self, dt):
        """Run every enabled row on its active cells for dt seconds. Returns the seconds
        the box may sleep (the bite clock), 0 when nothing moved."""
        rx = self.rx
        if not rx.rows:
            return 0.0
        mx = self.ensure_mixes()
        if rx.lut and rx.lut.get("_n", 0) > mx.n_rows:       # the table was collected: renumbered
            rx.lut = {}
        rx.lut["_n"] = mx.n_rows
        mat = self.mat.reshape(-1).astype(np.int64)
        sc = self.scomp.reshape(-1).astype(np.int64)
        sid = np.where(sc > 0, sc, mat)
        fv = self.fvol.reshape(-1)
        fid = np.where(fv > 0, np.where(self.fcomp.reshape(-1) > 0,
                                        self.fcomp.reshape(-1).astype(np.int64),
                                        FLUID0 + self.fl.reshape(-1).astype(np.int64)), 0)
        c = self._chem_clock()
        wait = 0.0
        for row in rx.rows:
            place = NAMES[row.key][0]
            lut = self._rx_lut(row, place)
            mask = lut[sid if place == "S" else fid]
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            view = LumpView(self, idx)
            st = LatticeState(view)
            nsub = 1 if (row.first_order and row.equilibrium is None) else \
                max(1, int(math.ceil(dt / c.dt_max)))
            xi = np.zeros(view.n)
            for _ in range(nsub):
                st._T = None
                xi = xi + rxn.advance([row], st, None, dt / nsub)[row.id]
            view.commit()
            self._chem_books().chem_J += float(st.chem.sum())
            rx.extents[row.id] = rx.extents.get(row.id, 0.0) + float(xi.sum())
            rx.runs += 1
            rx.cells += view.n
            flux = np.abs(xi) / dt
            j = int(np.argmax(flux)) if view.n else 0
            if view.n and flux[j] > 0.0:
                m_ref = max(view.sm[j], view.fm[j], 1e-6)
                need = float(view._thr("S", np.array([m_ref]))[0])
                ticks = int(bite_clock(np.array([flux[j] * TICK_S]), need / N_ULP, CAP_TICKS)[0])
                w_s = ticks * TICK_S
                wait = w_s if wait == 0.0 else min(wait, w_s)
        return wait
