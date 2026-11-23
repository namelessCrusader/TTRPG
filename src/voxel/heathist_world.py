"""HEAT HISTORY IN A WORLD (wave 15, lane 1e): the mixin, the gate and the books.

THE GATE: `world.heat_history` (class default False). Off, nothing here runs and no thing has
a temper from the law: every old world is bit for bit what it was. On, each THING that was
registered with a steel composition (`hh_register`) keeps one heathist state per CELL, stepped each
tick from the cell's own temperature. A thing with no composition, or one that has never passed
Ac1 or Ms, reads a temper of exactly 1.0.

  hh_register(eid, row | c, mn)   give a thing a composition (heathist_data.ROWS or numbers)
  heat_temper(eid, edge=False)    HV / HV_base: the multiplier matrec.at and edgewear read. With
                                  edge=True, the mean over the cells that carry the thing's edge
                                  (the tip is what wears), else the mean over the thing
  heat_hv(eid)                    per-cell Vickers hardness (a skin and a core differ)
  hh_toughness(eid)               per-cell ductility multiplier

THE BOOKS: the law's latent heat is taken out of (or put into) each cell's own E and booked on
`phase_J` (J held in the state, positive when stored as enthalpy of austenite or martensite), so
`total_energy() + phase_J` is constant. `phase_J` is a class attribute at zero and becomes an
instance value on the first booking (a `restore` takes it away again); the merge step adds it to
the ledger BOOKS and to total_energy.

THE ESTIMATE (ledger row heat_history.cell_mean): a cell's T is its mean, so the skin of a thick
cell is not seen; heathist_slab.Slab is the through-thickness sampler a blade or a bar scene uses.
"""
import numpy as np

from . import heathist as H
from . import heathist_data as D
from .materials import AMBIENT

SCALE_MIN_C = 150.0          # below this the oxide film grows less than a nanometre in a day (the log law)


class HeatHistoryMixin:
    heat_history = False
    phase_J = 0.0
    _hh = None               # {eid: dict(idx, state, x)} lazily made

    def hh_register(self, eid, row=None, c=None, mn=None):
        if row is not None:
            c, mn = D.ROWS[row]["c"], D.ROWS[row]["mn"]
        idx = np.asarray(self._cells_now(eid), np.int64)
        s = H.new_state(c, 0.5 if mn is None else mn, T0=float(self.T().ravel()[idx].mean()) if len(idx) else 20.0, n=len(idx))
        s["Tp"] = self.T().ravel()[idx].astype(float)
        if self._hh is None:
            self._hh = {}
        self._hh[eid] = dict(idx=np.sort(idx), c=float(c), mn=float(0.5 if mn is None else mn), s=s, x=np.zeros(len(idx)))
        return eid

    # -- per-tick
    def hh_book_soak(self, joules):
        """Book the enthalpy a bar's state already holds when a scene starts it soaked (the fire put
        it there before the scene began). The only writer of phase_J outside the tick is here, so the
        ledger has one owner."""
        self.phase_J = float(joules)

    def _tick_end(self):
        # World.step is defined in sim.py and wins over a mixin's step, so the
        # per-tick state step hangs on the end-of-tick seam, like reactions.py.
        super()._tick_end()
        if self.heat_history and self._hh:
            self._hh_tick()

    def _hh_sync(self, eid, rec):
        idx = np.sort(np.asarray(self._cells_now(eid), np.int64))
        if len(idx) == len(rec["idx"]) and (idx == rec["idx"]).all():
            return
        old = dict(zip(rec["idx"].tolist(), range(len(rec["idx"]))))
        fresh = H.new_state(rec["c"], rec["mn"], n=len(idx))
        T = self.T().ravel()[idx]
        fresh["Tp"] = np.asarray(T, float)
        x = np.zeros(len(idx))
        for j, i in enumerate(idx.tolist()):
            o = old.get(i)
            if o is not None:
                for key in fresh:
                    fresh[key][j] = rec["s"][key][o]
                x[j] = rec["x"][o]
        rec["idx"], rec["s"], rec["x"] = idx, fresh, x

    def _hh_tick(self):
        from .materials import TICK_S
        Ef, mf = self.E.ravel(), self.smass.ravel()
        Tf = None
        booked = 0.0
        for eid, rec in self._hh.items():
            s0 = rec["s"]
            if (not len(rec["idx"]) or (float(Ef[rec["idx"]].max()) <= 0.0 and not s0["fg"].any()
                                        and float(s0["Tp"].max()) <= 100.0)):
                continue                       # at or below ambient and nothing remembered: nothing to step
            if Tf is None:
                Tf = self.T().ravel()
            self._hh_sync(eid, rec)
            idx, s = rec["idx"], rec["s"]
            if not len(idx):
                continue
            T = Tf[idx].astype(float)
            dh = H.step(s, T, TICK_S)
            nz = dh != 0.0
            if nz.any():
                dE = mf[idx[nz]].astype(float) * dh[nz]        # J, the state's gain (or loss)
                Ef[idx[nz]] -= dE.astype(Ef.dtype)
                booked += float(dE.sum())
            hot = np.flatnonzero(T > SCALE_MIN_C)
            if len(hot):                                       # the film grows on the hot cells (law: layer_laws)
                rec["x"][hot] = H.oxide_step_v(rec["x"][hot], T[hot], TICK_S)
        if booked:
            self.phase_J = self.phase_J + booked

    # -- readouts
    def _hh_rec(self, eid):
        return None if not self._hh else self._hh.get(eid)

    def heat_hv(self, eid):
        rec = self._hh_rec(eid)
        return None if rec is None else H.hardness(rec["s"])

    def hh_toughness(self, eid):
        rec = self._hh_rec(eid)
        return None if rec is None else H.toughness_factor(rec["s"])

    def hh_scale_nm(self, eid):
        rec = self._hh_rec(eid)
        return None if rec is None else rec["x"] * 1e9

    def heat_temper(self, eid, edge=False):
        """The temper multiplier of a thing, or None when the gate is off or it has no
        composition (the caller then keeps the hand-set value)."""
        rec = self._hh_rec(eid) if self.heat_history else None
        if rec is None or not len(rec["idx"]):
            return None
        r = H.temper_ratio(rec["s"])
        if edge:
            e = self.edge.ravel()[rec["idx"]] > 0
            if e.any():
                return float(r[e].mean())
        return float(r.mean())
