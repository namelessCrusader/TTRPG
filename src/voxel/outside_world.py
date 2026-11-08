"""OutsideMixin: a DRAFT of how the cold topic plugs into a World (wave 13, T8).

NOT RUN ON A REAL WORLD IN PHASE A. Most of the laws it serves are written
inline in sim.py and heat.py, so a subclass cannot reach them: the 0 C floor
and the 'radiate to 20 C' block would run underneath it. Phase B step 0
(golden SAME) adds the overridable seams on base19: _floor(), _env_radiation(u,
sl), _top_sink(), _heat_busy(), _unsettled(pad), _rest_cell(). This file is
written against the base18 API (E, mat, fvol, smoke, heat_capacity(), tick,
shed, open_sky, air_region, scale, vox_l, cap, skip_quiet, persons) and is
exercised only against a stub host in tests/test_cold_world.py; the World-level
tests there are skipped with the reason 'needs phase B seams'.

State (small; the big caches live OUTSIDE the world like heat._GEO, so
snapshot() and restore() stay cheap and exact):

    w.climate         Climate or None (None = every new path is off)
    w.t_out, w.t_sky  floats in force now (T_out a multiple of step_k)
    w.air_rh          RH of fresh outside air (T6 reads it)
    w.outside_booked  dict cause -> joules (shed's sign: heat that LEFT)

Every method has a climate-None branch that does nothing or the old thing.
"""
import weakref

import numpy as np

from . import climate as K
from . import outside as O
from .materials import AIR, AMBIENT, SMOKE_STILL, TICK_S
from .window import box_volume

OUTSIDE_CAUSES = ("rad", "edge", "air_film", "air_far", "shift", "top", "ground", "ref_shift",
                  "mrt_resid")
_OUT = weakref.WeakKeyDictionary()      # world -> its outdoor caches (cls, face env, Hs ...)


class OutsideMixin:
    _uns_quiet = None                   # the memo of a QUIET answer: (tick, reasons), while a tick runs
    _in_step = False
    CLIMATE_EVERY = 40                  # ticks (1 s) between reads of T_out
    ENV_EVERY = 8                       # ticks between face-cache refreshes
    STATIC_EVERY = 40                   # ticks between heightmap checks

    # ------------------------------------------------------------ setup
    def set_climate(self, climate, soak=True):
        """Put a Climate on the world (or None to take it off). With soak, every
        cell starts at its rest: solids and enclosed air at T_rest, outdoor air
        at T_out. Scenes that want a settled start set rest_c = T_out(0)."""
        _OUT.pop(self, None)
        self.climate = climate
        self.outside_booked = {k: 0.0 for k in OUTSIDE_CAUSES}
        self.t_out = self.t_sky = None
        if climate is None:
            return
        self._publish_climate(first=True)
        self._env_refresh(force=True)
        if soak:
            self.soak()

    def _clim(self):
        return getattr(self, "climate", None)

    def _rest_k(self):
        c = self._clim()
        return 0.0 if c is None else K.rest_temp(c) - AMBIENT

    def _th_out(self):
        return 0.0 if getattr(self, "t_out", None) is None else self.t_out - AMBIENT

    # ------------------------------------------------------------ publish T_out
    def _publish_climate(self, first=False):
        """Read T_out and T_sky; if T_out (rounded to step_k) changed, ONE exact
        shift of the outdoor cells, booked. A settled cell lands on the new
        value bit for bit; a plume keeps its deviation."""
        c = self._clim()
        x = float(K.day_coord(c, None, self.tick))
        tq = float(K.quantize(K.day_curve(c, x), c.step_k))
        td = float(K.dew_point_state(c, x))
        self.t_sky = float(K.t_sky(c, tq, td, 24.0 * (x - np.floor(x))))
        self.air_rh = float(K.rh_of(tq, td))
        old = getattr(self, "t_out", None)
        self.t_out = tq
        if old is not None and old != tq and not first:
            cache = _OUT.get(self)
            if cache is not None and "cls" in cache:
                gain = O.outdoor_shift(self.E, self.heat_capacity(), cache["cls"],
                                       old - AMBIENT, tq - AMBIENT)
                self.shed -= gain
                self.outside_booked["shift"] -= gain
        if not first:
            self._env_temps()

    def _law_climate(self):
        """Tick start: the slow clock of the outside."""
        if self._clim() is not None and self.tick % self.CLIMATE_EVERY == 0:
            self._publish_climate()
            self._uns_quiet = None

    # ------------------------------------------------------------ where the laws plug in
    # World.step calls these names, so the mixin hangs its laws on them. (C1 does
    # not own sim.py; once the mixin is among World's bases the same hooks work.)
    def _tick_start(self):
        super()._tick_start()
        if self._clim() is not None:
            self._uns_quiet, self._in_step = None, True
            self._law_climate()

    def _tick_end(self):
        super()._tick_end()
        if self._in_step:
            self._uns_quiet, self._in_step = None, False

    def _law_rise(self, sl=None):
        """The old buoyancy law, then (climate worlds only) the outdoor air
        relaxing toward T_out and the deep ground under the floor."""
        super()._law_rise(sl)
        if self._clim() is not None:
            self._uns_quiet = None
            self._law_outdoor()
            self._law_ground()

    def _floor(self):
        """THE 0 C FLOOR IS GONE (T8 3.9). B0's floor probe: no golden case ever
        goes below 0 C. A cold cell stays cold and the books close.

        It still ASKS for the heat capacity, as the old floor did on its first
        line: asking is what notices that matter has moved, and the derived
        caches that keeps (`_cap`, `_cap_key`, `_matter_gen`) are part of the
        state a world is compared by (golden), so they are touched at the same
        place in the tick as they always were."""
        self.heat_capacity()

    def _top_sink(self):
        """The open top: the plume's excess over the OUTSIDE air leaves (3.8a).
        Without a climate it is the old law, line for line."""
        if self._clim() is None:
            return super()._top_sink()
        top = (slice(None), slice(None), -1)
        sky = (self.mat[top] == AIR) & (self.fvol[top] < 0.1 * self.cap)
        if sky.any():
            was = float(self.E[top].sum(dtype=np.float64))
            Et = self.E[top]
            C = self.heat_capacity()[top]
            Tt = AMBIENT + Et / C
            ex = np.maximum(Tt - self.t_out, 0.0)
            share = self._rise_share(ex, self.t_out)[sky].astype(np.float32)
            Et[sky] -= share * (Et[sky] - np.float32(self._th_out()) * C[sky])
            gone = was - float(self.E[top].sum(dtype=np.float64))
            self.shed += gone
            self.outside_booked["top"] += gone

    def _rest_ref(self, sl):
        """What the eddy term measures a gas cell's warmth against: K above
        20 C, per cell of the box `sl` (3.8d). None with no climate, so the old
        line `T - AMBIENT` stays verbatim."""
        r = self._rest_cell()
        return r[sl] if isinstance(r, np.ndarray) else None

    # ------------------------------------------------------------ the matter rule (T8 4.3)
    def _rest_pre(self, mask):
        if self._clim() is None:
            return None
        return self.heat_capacity()[mask].astype(np.float64)

    def _rest_post(self, mask, c_old):
        """Matter has left the cells of `mask` at a fixed E. The REST part of the
        energy travels with the capacity (booked as ref_shift, with shed's sign),
        and cold matter takes its cold with it (to `gone`, so the total closes):
        a -5 C stone cell that is dissolved does not stay on a far lighter cell as
        -43,000 C."""
        c_new = self.heat_capacity()[mask].astype(np.float64)
        e_new, ref, carried = O.matter_change(self.E[mask], c_old, c_new, AMBIENT + self._rest_k())
        self.E[mask] = e_new.astype(np.float32)
        r, c = float(ref.sum()), float(carried.sum())
        self.shed += r
        self.outside_booked["ref_shift"] += r
        self.gone["E"] = self.gone.get("E", 0.0) + c

    def _warm_cells(self):
        """The legacy pressure field's buoyant push in a climate world (3.8c): the
        warmth of GAS above its own rest, as the coarse cell's mean over all its
        voxels (the average the old law took, so a reference climate at 20 C
        gives the old field bit for bit wherever the solids are at rest). A hot
        stone is not a buoyant gas (only gas cells count) and a cold wall at rest
        beside outdoor air at T_out is no wind (the reference is the cell's own
        rest, not 20 C). Spec 3.8c divides by the gas cells only; that would make
        a coarse cell that is mostly wall push as hard as open air (reference
        continuity R1 measured 4 percent on torch_pillar), so the old mean is kept."""
        if self._clim() is None:
            return super()._warm_cells()
        gas = self.mat == AIR
        warm = np.where(gas, np.clip(self.T() - (AMBIENT + self._rest_cell()), 0.0, 500.0), 0.0)
        return self._cells(warm.astype(np.float32))

    def _T_in(self, fill=True):
        """The temperature field and its box. In a climate world the cells
        outside the unsettled box are at THEIR rest, not at 20 C (3.8f)."""
        if self._clim() is None:
            return super()._T_in(fill)
        whole = (slice(None), slice(None), slice(None))
        cap = self.heat_capacity()
        if not self._windows_on():
            return AMBIENT + self.E / cap, whole
        box = self._unsettled(0)[0]
        if box is None and not fill:
            return None, None
        if box is not None and box_volume(box) * 2 > self.E.size:
            return AMBIENT + self.E / cap, whole
        t = np.empty(self.shape, self._T_dtype())
        t[...] = AMBIENT + self._rest_cell()
        if box is not None:
            t[box] = AMBIENT + self.E[box] / cap[box]
        return t, box

    # ------------------------------------------------------------ the caches
    def _static_heights(self):
        """Hs: the highest cell of static solid structure in each column (solid
        materials; not flesh, not fluid), -1 where none. A person walking never
        changes it."""
        solid = self.mat != AIR
        flesh = getattr(self, "_FLESH_ID", None)
        if flesh is not None:
            solid = solid & (self.mat != flesh)
        nz = solid.shape[2]
        top = nz - 1 - np.argmax(solid[:, :, ::-1], axis=2)
        return np.where(solid.any(axis=2), top, -1).astype(np.int64)

    def _env_refresh(self, force=False):
        """Rebuild the outdoor class and the face environment when the static
        structure or the airspaces changed; otherwise at most every ENV_EVERY
        ticks (stale up to 0.2 s; conservation holds because emission and
        absorption use the same cached weights)."""
        c = self._clim()
        if c is None:
            return None
        cache = _OUT.setdefault(self, {})
        if not force and self.tick - cache.get("tick", -(10 ** 9)) < self.ENV_EVERY:
            return cache
        cache["tick"] = self.tick
        # THE CHEAP KEY (cost): matter that has not moved gives the world's own cached heat
        # capacity back as the SAME array object (fields.heat_capacity hands out a new one when
        # mat, smass, fl or fvol changed), so asking "did the structure change?" is an identity
        # test, not two passes over the lattice. The airspaces are labelled by the world.
        reg_id = id(getattr(self, "air_region", None))
        quick = (id(self.heat_capacity()), reg_id)
        if quick == cache.get("quick"):
            return cache
        cache["quick"] = quick
        Hs = self._static_heights()
        gas = (self.mat == AIR) & (self.fvol < 0.01 * self.cap)
        key = (hash(Hs.tobytes()), hash(gas.tobytes()), reg_id)      # a person walking changes `gas`
        if key == cache.get("key"):
            return cache
        cache["key"] = key
        solid = self.mat != AIR
        nb = np.zeros(solid.shape, bool)
        for ax in range(3):
            for sgn in (1, -1):
                nb |= O._shift_nb_free(solid, ax, sgn, False)
        film = gas & nb
        R = O.view_radius_cells(Hs, 0.1 * self.scale)
        cls = O.outdoor_class(Hs, gas, film, R, open_sky=self.open_sky)
        out = (cls & 1) != 0
        cells = np.argwhere(out & film)
        psi = np.zeros(solid.shape, np.float32)
        if len(cells):
            psi[tuple(cells.T)] = O.sky_view(Hs, cells, R).astype(np.float32)
        reg = getattr(self, "air_region", None)
        room = np.where(gas & ~out, (reg + 2) if reg is not None else 1, 0).astype(np.int32)
        ground_k = int(round(c.ground_top_m / (0.1 * self.scale)))
        fe = O.face_env(solid, gas, cls, psi, room, open_sky=self.open_sky, ground_k=ground_k)
        cache.update(Hs=Hs, gas=gas, film=film, cls=cls, psi=psi, room=room, fe=fe, solid=solid,
                     exposed=(solid & (fe.n_out > 0)) | (out & film))
        self._env_temps()
        return cache

    def _env_temps(self):
        cache = _OUT.get(self)
        if cache is None or "fe" not in cache or self.t_out is None:
            return
        cache["tenv"] = O.env_temperature(cache["fe"].n_out, cache["fe"].w_sky, self.t_sky, self.t_out)

    def env_t4(self, cell):
        """T^4 (K^4) of the environment a surface cell radiates to: the one the
        lattice face law uses, so the body law's 1/s correction does not count
        the same radiation twice (T7's seam)."""
        cache = _OUT.get(self)
        if cache is None or "fe" not in cache:
            return float((AMBIENT + 273.15) ** 4)
        fe = cache["fe"]
        return float(O.env_t4(fe.n_out[cell], fe.w_sky[cell], self.t_sky, self.t_out))

    # ------------------------------------------------------------ the laws
    def _law_outdoor(self):
        """Outdoor air relaxes toward T_out: k_film where it touches a surface,
        k_far in free air. Exact exponential; booked."""
        c = self._clim()
        cache = _OUT.get(self)
        if c is None or cache is None or "cls" not in cache:
            return
        a, b = O.outdoor_relax(self.E, self.heat_capacity(), cache["cls"], self._th_out(),
                               O.relax_factor(c.k_film, TICK_S), O.relax_factor(c.k_far, TICK_S),
                               O.SETTLE_K, hot_k=c.plume_k, smoke=self.smoke, smoke_still=SMOKE_STILL)
        self.shed += a + b
        self.outside_booked["air_film"] += a
        self.outside_booked["air_far"] += b
        self._uns_quiet = None

    def _env_radiation(self, u, eps, sl):
        """The seam in heat._law_conduct: the radiation of every surface cell of
        the box `sl`, replacing the 'radiate to 20 C' block. Returns float32
        joules that LEAVE each cell, and books its parts. Outdoor faces see the
        sky and the outdoor air; indoor faces exchange through their room's
        radiant node (zero-sum); edge faces add a film."""
        if self._clim() is None:
            return super()._env_radiation(u, eps, sl)
        Ts, af, Csur = u.Ts, u.af, u.Csur
        parts = self._rad_parts = {"rad": 0.0, "edge": 0.0, "mrt_resid": 0.0, "hot": 0.0}
        cache = self._env_refresh()
        fe, solid = cache["fe"], cache["solid"][sl]
        dx = 0.1 * self.scale
        A, dt = dx * dx, TICK_S
        n_out = np.where(solid, fe.n_out[sl], 0)
        loss = O.radiation_out(Ts, eps, af, n_out, cache["tenv"][sl], A, dt, 0.0, Csur)
        parts["rad"] = float(loss.sum(dtype=np.float64))
        edge = O.edge_film(Ts, np.where(solid, fe.n_edge[sl], 0), self.t_out, af, A, dt, Csur)
        parts["edge"] = float(edge.sum(dtype=np.float64))
        loss = loss + edge
        n_in = np.where(solid, fe.n_in[sl], 0)
        if (n_in > 0).any():
            m = O.mrt_star(Ts, eps, af, n_in, fe.room[sl], A, dt, Csur)
            loss = loss + m.q
            parts["mrt_resid"] = m.resid
            # surfaces at or above RAD_HOT are not in the star: they take the old
            # route, radiating to their room's node, and the loss leaves (G7 lands
            # what it can). It is the one indoor leak, hot surfaces only (3.7).
            hot = (n_in > 0) & (Ts >= O.RAD_HOT)
            if hot.any():
                hl = O.radiation_out(Ts, eps, af, np.where(hot, n_in, 0), m.tmr_cell, A, dt, 0.0, Csur)
                loss = loss + hl
                parts["hot"] = float(hl.sum(dtype=np.float64))
            tm = cache.get("tmr")
            if tm is None:
                tm = cache["tmr"] = np.full(self.shape, AMBIENT, np.float32)
            tm[sl] = m.tmr_cell
        return loss.astype(np.float32)

    def _radiated(self, gone):
        """What the radiation law took out of the box, by cause. The nominal
        parts come from the kernels; what float32 rounding added to or took from
        the cells is booked with the outdoor radiation, so the books equal the
        part of shed that T8 wrote. The hot route (surfaces at or above RAD_HOT
        radiating to their room node) stays in shed and G7's bank, as in base18."""
        parts = getattr(self, "_rad_parts", None)
        if self._clim() is None or parts is None:
            return
        self._rad_parts = None
        b = self.outside_booked
        b["edge"] += parts["edge"]
        b["mrt_resid"] += parts["mrt_resid"]
        b["rad"] += gone - parts["edge"] - parts["mrt_resid"] - parts["hot"]

    def _hot_env_k4(self, box, hm):
        """G7's environment for a hot surface cell: the sky and outdoor air for
        its outdoor faces and its room's radiant node for its indoor faces,
        weighted by face count (T^4, K^4). The fixed ambient with no climate."""
        cache = _OUT.get(self)
        if self._clim() is None or cache is None or "fe" not in cache or self.t_out is None:
            return super()._hot_env_k4(box, hm)
        fe = cache["fe"]
        no, ni = fe.n_out[box][hm].astype(np.float64), fe.n_in[box][hm].astype(np.float64)
        t4o = (cache["tenv"][box][hm].astype(np.float64) + 273.15) ** 4
        tm = cache.get("tmr")
        t4i = ((tm[box][hm].astype(np.float64) if tm is not None else AMBIENT) + 273.15) ** 4
        w = no + ni
        return np.where(w > 0, (no * t4o + ni * t4i) / np.where(w > 0, w, 1.0), (AMBIENT + 273.15) ** 4)

    def _law_ground(self):
        """The bottom layer of solid cells exchanges with the deep ground over
        ground_depth_m (an adiabatic bottom when it is None)."""
        c = self._clim()
        cache = _OUT.get(self)
        if c is None or c.ground_depth_m is None or cache is None or "solid" not in cache:
            return
        C = self.heat_capacity()[:, :, 0]
        Ts = AMBIENT + self.E[:, :, 0] / C
        x = float(K.day_coord(c, None, self.tick))
        tg = float(K.t_ground(c, c.ground_depth_m, x))
        dx = 0.1 * self.scale
        k = np.full(Ts.shape, 2.0, np.float32)                    # TODO(B): the cell's own k
        q = O.ground_flux(Ts, tg, k, dx * dx, TICK_S, c.ground_depth_m + dx / 2.0, C)
        q = np.where(cache["solid"][:, :, 0], q, np.float32(0.0)).astype(np.float32)
        was = float(self.E.sum(dtype=np.float64))
        self.E[:, :, 0] -= q
        gone = was - float(self.E.sum(dtype=np.float64))
        self.shed += gone
        self.outside_booked["ground"] += gone
        self._uns_quiet = None

    # ------------------------------------------------------------ the gate
    def _unsettled(self, pad=0):
        """(box, reasons): the cells not at rest, joined with the exposed list
        when the world is driven. With no climate: exactly the old
        _window(E != 0, pad) and no reasons.

        THE MEMO (cost, spec 3.8e): the gate is asked five or six times a tick and
        a scan of 1.6 M cells costs about a millisecond. WHILE A TICK RUNS, an
        answer of QUIET stays true until a law that can write E has run, so the
        scan is made once and the laws in `_E_WRITERS` clear it when they finish.
        It never lives across ticks (`_tick_start` clears it, `_tick_end` ends it),
        so a write between two ticks is always seen; and a busy answer is never
        kept, because a busy world's laws change E all the time."""
        c = self._clim()
        if c is None:
            return self._window(self.E != 0, pad), []
        if not self.skip_quiet:
            return (slice(None), slice(None), slice(None)), ["whole"]
        if self._uns_quiet is not None and self._uns_quiet[0] == self.tick:
            return None, list(self._uns_quiet[1])
        cache = self._env_refresh()
        why = O.driven_reasons(K.rest_temp(c), self.t_out, self.t_sky)
        box, reasons = O.unsettled_box(self.E, self.heat_capacity(), cache["cls"], self._rest_k(),
                                       self._th_out(), O.SETTLE_K, exposed=cache["exposed"],
                                       driven=why, pad=pad)
        if box is None and self._in_step:
            self._uns_quiet = (self.tick, tuple(reasons))
        return box, reasons

    def _heat_busy(self):
        if self._clim() is None:
            return super()._heat_busy()
        return self._unsettled(0)[0] is not None

    def _rest_cell(self):
        """The per-cell rest of E in K above 20 C: T_out outdoors, T_rest else."""
        cache = _OUT.get(self)
        if self._clim() is None or cache is None or "cls" not in cache:
            return np.float32(0.0)
        return np.where((cache["cls"] & 1) != 0, np.float32(self._th_out()),
                        np.float32(self._rest_k())).astype(np.float32)

    # ------------------------------------------------------------ scenes
    def soak(self):
        """Every cell at its rest. Call LAST, after fill() and fixtures, and
        never on the cells of a living person (T7 puts heat on a person)."""
        cache = self._env_refresh(force=True)
        C = self.heat_capacity()
        E = O.soak_fill(C, cache["cls"], self._rest_k(), self._th_out())
        keep = getattr(self, "_living_cells", None)
        if keep is not None:
            E = np.where(keep, self.E, E)
        self.E[...] = E

    def outside_totals(self):
        """The books, and whether they add up to the part of shed T8 wrote."""
        b = dict(getattr(self, "outside_booked", {}))
        return dict(b, total=float(sum(b.values())))


# The laws that can write E between the first look at "what is not at rest" and the gates after
# them. Each clears the memo of a quiet answer when it finishes (`OutsideMixin._unsettled`).
# Laws that run BEFORE the first look of a tick (particles, bodies landing, igniters) need no
# entry: the first look of every tick is fresh.
_E_WRITERS = ("_law_ignite", "_law_bodies", "_law_flow", "_law_stir", "_law_head", "_law_conduct",
              "_law_crack", "_law_rise", "_law_burn", "_law_boil", "_law_melt", "_law_acid",
              "_law_smoke", "_law_o2", "_law_air", "_law_gasflow", "_law_outdoor", "_law_ground",
              "_law_climate")


def _next_law(world, name):
    """The method `name` of the class after OutsideMixin in the world's MRO."""
    mro = type(world).__mro__
    for c in mro[mro.index(OutsideMixin) + 1:]:
        f = c.__dict__.get(name)
        if f is not None:
            return f.__get__(world)
    return None


def _clears_memo(name):
    def law(self, *a, **k):
        f = _next_law(self, name)
        r = f(*a, **k) if f is not None else None
        if self._uns_quiet is not None:                 # (a world with no climate never makes the name)
            self._uns_quiet = None
        return r
    law.__name__ = name
    return law


for _n in _E_WRITERS:
    if _n not in vars(OutsideMixin):                    # (the mixin's own laws clear it below)
        setattr(OutsideMixin, _n, _clears_memo(_n))
