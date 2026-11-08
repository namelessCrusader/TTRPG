"""WetMixin / WetWorld: the matter processes of water (wave 13c, lane C3, T6).

Water leaves the liquid and comes back. Steam is KEPT (boiling hands the grams to a
vapour store instead of deleting them), a wet surface gives vapour to the air or takes
it from the air by ONE surface law, a supersaturated gas voxel makes mist, water that
is cold freezes by the enthalpy method and ice melts, and a moist solid dries toward
the humidity of the air round it.

THE GATE. `w.wet` is None on every world. `w.enable_wet(rh=...)` makes a `WetState`
(the gas store, the clock) and turns the laws on; with `w.wet` None every method here
falls back to `super()` and the world is the old world, bit for bit (tests/test_wet_on.py
runs golden-style scenes both ways). It composes with the other capabilities (cold, body
heat, mixtures, material laws): `class MyWorld(WetMixin, BodyWorld)` is one world with
all of them. `WetWorld = WetMixin + ColdWorld` is the plain name.

THE BOOKS (T6 spec section 3 and 8). One gram going between liquid at Tv and vapour
moves the cell's E by -(L0 + cv (Tv - 20)), the gas voxel's E by +cv (Tv - 20), and the
STORED LATENT by +L0, so the three add to zero. `total_energy()` therefore reads
E + ... + L0 * (vapour grams above the fixed background). `total_water()` is liquid +
vapour + mist + ice + what has left through the open edges (`gone['vapour']`). Ice
carries the enthalpy of icephase: `- L_CONV20 * ice grams` in `total_energy()`.

WHAT IS CHEAP ON PURPOSE (the user's cheaper-laws preference; each is in the C3
report's ledger with its deleting condition):
  * the slow laws (surface, seep, mist, ice) run once a second (EVERY ticks), with
    exponential steps that cannot overshoot, so a longer step is still exact in
    the books and never makes mist from nothing;
  * the vapour and mist heat capacity are NOT in the voxel's C (a 5 cm voxel holds
    0.15 J/K, steam in it adds up to 25 percent); the latent heat is booked exactly,
    the sensible part of a gram of steam is carried by the voxel it is put in;
  * with gas_flow OFF the vapour spreads by a plain diffusion (no rise); with gas_flow
    ON the flow carries it (gasflow.py hooks `_gf_extra_*`), so steam rises with the
    warm air it comes with.

NOT HERE (the report says so): frost on a cold pane (needs face layers, T3), Fick
transport INSIDE a solid (SCALE's list), sorption into DRY cells, mixed liquids (they
keep the legacy skip, D6: brine boils in heatmix and its steam is booked there).
"""
import numpy as np

from . import heatmix as _heatmix
from . import icephase as IP
from . import psychro
from .coldworld import ColdWorld
from .gasstore import GasStore, gas_volume
from .materials import (AIR, AMBIENT, BOIL, H_CONV, ICE, NOFLUID, TICK_S, WATER,
                        FLUID, WOOD)
from .matterdata import CI, CV, CW, L0, L_CONV20, P_ATM, R, M_WATER

EVERY = 40                   # ticks between the slow laws (1 s)
R_INT_SOLID = 23000.0        # s/m, internal resistance of a moist wood cell (T6 spec 5.5, c)
SEEP_K = 1.0 / 6.0           # share of a difference a diffusion step moves (stability limit)
EDGE_RELAX = 0.5             # share of the vapour on an open world edge that leaves per slow step
HEADSPACE = 0.9              # a cell fuller than this has no gas to speak of
TINY_G = 1.0e-9              # grams below which vapour is called none
D_WATER = FLUID[WATER][0]    # g/ml
_DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


class WetState:
    """What a wet world holds: the gas store, the clock, the switches."""

    def __init__(self, store, every=EVERY):
        self.store = store
        self.every = int(every)
        self.dt = self.every * TICK_S
        self.surface = True      # the surface law
        self.mist = True         # saturation adjustment
        self.seep = True         # diffusion of vapour when gas_flow is off
        self.ice = True          # freezing and melting
        self.edges = True        # open world edges let vapour out and in
        self.keep_steam = True   # boiling keeps its steam
        self.stats = {"evap_g": 0.0, "cond_g": 0.0, "boil_g": 0.0, "mist_g": 0.0}
        self._cap_key = None
        self._cap_val = None
        self.ice_gen = 0
        self.quiet_gen = -1       # store generation at the last verdict 'no vapour, no mist'
        self.has_ice = False      # any ICE cell or ice gram count (set by enable_wet and by the ice law)


class WetMixin:
    wet = None

    # ---------------------------------------------------------------- setup
    def enable_wet(self, rh=0.5, t_c=None, every=EVERY):
        """Turn the matter processes on. `rh` is the relative humidity of fresh air; the
        background vapour density B0 is fixed from it at `t_c` (default 20 C, or the
        outside air with a climate). Returns the WetState."""
        t_ref = AMBIENT if t_c is None else float(t_c)
        if t_c is None and getattr(self, "t_out", None) is not None:
            t_ref = float(self.t_out)
        store = GasStore(self.shape, self.vox_l)
        store.set_background("H2O", float(rh) * float(psychro.rho_sat(t_ref)))
        self.wet = WetState(store, every)
        self.air_rh = float(rh)
        self.wet_refresh()
        return self.wet

    def wet_refresh(self):
        """Look the world over once for ICE (cells or gram counts) and for vapour. Call it
        after a scene fills ICE into the lattice by hand: the idle gate trusts its flags."""
        w = self.wet
        w.has_ice = bool((self.mat == ICE).any()) or (self.ice_g is not None and bool(self.ice_g.any()))
        w.quiet_gen = -1

    # ---------------------------------------------------------------- readings
    def _gas_cells(self):
        """Cells that are gas: air with at most HEADSPACE of liquid in them."""
        return (self.mat == AIR) & (self.fvol < HEADSPACE * self.cap)

    def vapour(self):
        """Grams of vapour ABOVE the background, per voxel (float32). Allocated on use."""
        return self.wet.store.excess("H2O")

    def mist(self):
        return self.wet.store.excess("MIST")

    def rh_at(self, cell):
        """Relative humidity of the air of one cell (fraction)."""
        st = self.wet.store
        x, y, z = cell
        V = float(gas_volume(self.vox_l, self.fvol[x, y, z]))
        rho = st.bg("H2O") + float(st.excess("H2O")[x, y, z]) / V
        T = AMBIENT + float(self.E[x, y, z]) / float(self.heat_capacity()[x, y, z])
        return rho / float(psychro.rho_sat(T))

    def total_water(self):
        """Liquid water + vapour + mist + ice + vapour that left by the open edges.
        Constant in a world with no reaction rows (T6 spec section 8)."""
        w = self.total_fluid(WATER)
        if self.wet is None:
            return w
        st = self.wet.store
        w += st.total("H2O") + st.total("MIST") + self.wet_gone()
        w += self.ice_grams()
        return w

    def wet_gone(self):
        return float(self.gone.get("vapour", 0.0))

    def ice_grams(self):
        g = float(self.ice_g.sum(dtype=np.float64)) if self.ice_g is not None else 0.0
        return g + float(self.smass[self.mat == ICE].sum(dtype=np.float64))

    def total_energy(self):
        e = super().total_energy()
        if self.wet is None:
            return e
        return e + L0 * self.wet.store.total("H2O") - L_CONV20 * self.ice_grams()

    def heat_capacity(self):
        """The old heat capacity, plus what the T6 matter adds: the vapour above the
        background (cv per gram), the mist (cw) and the ice held as a gram count in a
        cell that is not an ICE cell (ci). T6 spec 4: without them a 5 cm voxel (0.15 J/K)
        would swing 16 K per milligram of condensate. Cached against the base array
        and the store, and worked out only inside the box where any of it is."""
        cap = super().heat_capacity()
        w = self.wet
        if w is None:
            return cap
        st = w.store
        key = (id(cap), st.generation, w.ice_gen)
        if w._cap_key == key:
            return w._cap_val
        ex = st.excess("H2O") if st.has("H2O") else None
        mi = st.excess("MIST") if st.has("MIST") else None
        ig = self.ice_g
        there = None
        for a in (ex, mi, ig):
            if a is not None:
                m = (a > 0)
                there = m if there is None else (there | m)
        box = None if there is None else self._window(there, 0)
        if box is None:
            out = cap
        else:
            out = cap.copy()
            add = np.zeros(out[box].shape, np.float64)
            if ex is not None:
                add += CV * np.maximum(ex[box], 0.0)
            if mi is not None:
                add += CW * mi[box]
            if ig is not None:
                add += CI * ig[box]
            out[box] += add.astype(out.dtype)
        w._cap_key, w._cap_val = key, out
        return out

    # ---------------------------------------------------------------- the clock
    def _tick_end(self):
        super()._tick_end()
        w = self.wet
        if w is not None and self.tick % w.every == 0:
            self._law_wet()

    def _law_wet(self):
        """The slow matter laws, in this order: the surface law (evaporate / condense /
        sorb), the vapour spread, mist, the open edges, ice.

        THE IDLE GATE. A world with no liquid water, no vapour or mist and no ice does
        one reduction over the liquid volumes (about 1.7 ms at 5.1 M voxels) and stops;
        the store's generation says whether anything touched the vapour since the last
        quiet verdict, so a quiet sky costs nothing more."""
        w = self.wet
        dt = w.dt
        st = w.store
        water = bool(self.fvol.any())
        vap = st.generation != w.quiet_gen
        if not water and not vap and not w.has_ice:
            return
        if w.surface:
            self._wet_surface(dt, water)
        if w.seep and not getattr(self, "gas_flow", False):
            self._wet_seep()
        if w.mist:
            self._wet_mist(dt)
        if w.edges:
            self._wet_edges()
        if w.ice:
            self._wet_ice(dt)
        if vap and not water and not w.has_ice:
            quiet = not (st.excess("H2O").any() or st.excess("MIST").any())
            w.quiet_gen = st.generation if quiet else -1

    # ---------------------------------------------------------------- boiling, steam kept
    def _law_boil(self):
        w = self.wet
        if w is None or not w.keep_steam:
            return super()._law_boil()
        if self.mixes_live and getattr(self, "mix_laws", True):
            _heatmix.boil_mixed(self, self.T(), self.heat_capacity(), BOIL)
        cand = (self.fl == WATER) & (self.fvol > 0)
        ok = self.legacy_ok(solid=False)
        if ok is not None:
            cand &= ok
        if not cand.any():
            return
        T, C = self.T(), self.heat_capacity()
        boil_at = BOIL + (1.0 * np.clip(self._upcell(self.pcell), 0.0, 18.0)
                          if self.pcell is not None else 0.0)
        hot = cand & (T > boil_at)
        if not hot.any():
            return
        idx = np.nonzero(hot)
        Tb = boil_at[idx] if isinstance(boil_at, np.ndarray) else np.full(len(idx[0]), boil_at)
        m_w = self.fvol[idx].astype(np.float64) * D_WATER
        g, E_new, _C_new, gain, _st = psychro.boil_step(self.E[idx], C[idx], m_w, Tb)
        sliver = (m_w - g) <= 0.01                       # the old law clears fvol <= 0.01
        if sliver.any():
            g = np.where(sliver, m_w, g)
            cell, gas, _s = psychro.evap_energy(Tb)
            E_new = self.E[idx].astype(np.float64) - g * cell
            gain = g * gas
        self.fvol[idx] = np.maximum(self.fvol[idx] - (g / D_WATER).astype(self.fvol.dtype), 0.0)
        self.E[idx] = E_new.astype(self.E.dtype)
        self.p_add[idx] += (g * 0.1).astype(self.p_add.dtype)
        dry = idx[0][self.fvol[idx] <= 0.0], idx[1][self.fvol[idx] <= 0.0], idx[2][self.fvol[idx] <= 0.0]
        if len(dry[0]):
            self.fl[dry] = NOFLUID
            self.fpot[dry] = 0.0
        self._deposit(idx, g, gain)
        w.stats["boil_g"] += float(g.sum())

    def _deposit(self, src, g, gain):
        """Put g grams of vapour made at cells `src` into the gas, with `gain` joules of
        sensible heat. The steam goes to the cell itself while it has headspace, else
        up the column to the first cell that does. Out of the open top it is gone
        (booked, with its latent heat). Under a lid it stays in the cell it was made in
        (a sealed pot builds pressure through p_add)."""
        st = self.wet.store
        x, y, z = (np.asarray(a).copy() for a in src)
        g = np.asarray(g, np.float64)
        gain = np.asarray(gain, np.float64)
        nz = self.shape[2]
        state = np.zeros(len(x), np.int8)          # 0 looking, 1 placed, 2 out the top, 3 lid
        for _ in range(nz + 1):
            act = np.nonzero(state == 0)[0]
            if not len(act):
                break
            xs, ys, zs = x[act], y[act], z[act]
            here = (self.mat[xs, ys, zs] == AIR) & (self.fvol[xs, ys, zs] < HEADSPACE * self.cap)
            state[act[here]] = 1
            go = act[~here]
            if not len(go):
                break
            top = z[go] + 1 >= nz
            state[go[top]] = 2
            go = go[~top]
            lid = self.mat[x[go], y[go], z[go] + 1] != AIR
            state[go[lid]] = 3
            z[go[~lid]] += 1
        out = state == 2
        if out.any():
            gl = float(g[out].sum())
            self.gone["vapour"] = self.gone.get("vapour", 0.0) + gl
            self.shed += L0 * gl + float(gain[out].sum())
        for code, cell in ((1, (x, y, z)), (3, tuple(np.asarray(a) for a in src))):
            m = state == code
            if m.any():
                t = tuple(a[m] for a in cell)
                st.add("H2O", t, g[m])
                np.add.at(self.E, t, gain[m].astype(self.E.dtype))

    # ---------------------------------------------------------------- the surface law
    def _wet_surface(self, dt, water=True):
        """Evaporation, condensation and sorption at the faces between a wet cell and gas,
        and condensation of vapour on cold solid faces. One law, exponential step."""
        st = self.wet.store
        wcell = (self.fl == WATER) & (self.fvol > 0) if water else np.zeros(self.shape, bool)
        ex = st.excess("H2O")
        vap_any = bool(ex.any())
        if not wcell.any() and not vap_any:
            return
        ok = self.legacy_ok(solid=False)
        if ok is not None:
            wcell &= ok
        if wcell.any():
            self._surface_sources(wcell, dt)
        if vap_any:
            self._surface_sinks(dt)

    def _face(self):
        dx = 0.1 * self.scale
        return dx * dx, 0.1 * 0.1 * 0.1 * self.scale ** 3

    def _surface_sources(self, wcell, dt):
        st = self.wet.store
        A, _V = self._face()
        idx = np.nonzero(wcell)
        n = len(idx[0])
        if not n:
            return
        C = self.heat_capacity()
        pool = self.mat[idx] == AIR
        head = pool & (self.fvol[idx] < HEADSPACE * self.cap)
        Ts = np.clip(AMBIENT + self.E[idx].astype(np.float64) / C[idx], -80.0, 370.0)
        sm = self.smass[idx].astype(np.float64)
        wc = np.where(sm > 0, self.fvol[idx] * D_WATER / np.maximum(sm, 1e-9), 0.0)
        a_s = np.where(pool, 1.0, np.where(wc >= 0.30, 1.0, psychro.gab_aw(np.minimum(wc, 0.30))))
        Rint = np.where(pool, 0.0, R_INT_SOLID)
        for d in ((0, 0, 0),) + _DIRS:
            if d == (0, 0, 0):
                use = head
                tx, ty, tz = idx
            else:
                nx = idx[0] + d[0]
                ny = idx[1] + d[1]
                nzz = idx[2] + d[2]
                inb = ((nx >= 0) & (nx < self.shape[0]) & (ny >= 0) & (ny < self.shape[1])
                       & (nzz >= 0) & (nzz < self.shape[2]))
                nx, ny, nzz = (np.clip(nx, 0, self.shape[0] - 1), np.clip(ny, 0, self.shape[1] - 1),
                               np.clip(nzz, 0, self.shape[2] - 1))
                gas_t = (self.mat[nx, ny, nzz] == AIR) & (self.fvol[nx, ny, nzz] < HEADSPACE * self.cap)
                # a pool with headspace has its own surface; a full pool shows its top face;
                # a moist solid shows every face that touches gas
                use = inb & gas_t & np.where(pool, (~head) & (d == (0, 0, 1)), True)
                tx, ty, tz = nx, ny, nzz
            if not use.any():
                continue
            i = np.nonzero(use)[0]
            t = (tx[i], ty[i], tz[i])
            s = tuple(a[i] for a in idx)
            self._apply_face(s, t, Ts[i], a_s[i], Rint[i], dt, A, pool[i])

    def _apply_face(self, s, t, Ts, a_s, Rint, dt, A, pool):
        """One batch of faces: source cells `s` (a wet cell), target gas cells `t`."""
        st = self.wet.store
        C = self.heat_capacity()
        Vg = gas_volume(self.vox_l, self.fvol[t].astype(np.float64))
        Tg = np.clip(AMBIENT + self.E[t].astype(np.float64) / C[t], -80.0, 370.0)
        rho_air = st.bg("H2O") + st.excess("H2O")[t].astype(np.float64) / Vg
        rho_eq = psychro.rho_sat(Ts) * a_s
        evap = rho_eq >= rho_air
        x_s = psychro.p_sat_water(Ts) * a_s / P_ATM
        x_g = rho_air * R * (Tg + 273.15) / M_WATER / P_ATM
        F = np.where(evap, psychro.stefan_factor(x_s), psychro.stefan_factor(x_g))
        g = psychro.surface_step(rho_eq, rho_air, Vg, A, H_CONV, F, Rint, dt)
        mw = self.fvol[s].astype(np.float64) * D_WATER
        g = np.minimum(g, mw)                                       # cannot give more than it has
        room = (self.cap - self.fvol[s].astype(np.float64)) * D_WATER
        # condensing: the liquid must fit
        g = np.where(g < 0, -np.minimum(-g, np.maximum(room, 0.0)), g)
        # a different liquid in the target cell? only plain targets are used (gas cells)
        applied = st.add("H2O", t, g)
        g = applied
        Tv = np.where(g >= 0, Ts, Tg)
        cell, gas, _stored = psychro.evap_energy(Tv)
        self.fvol[s] = np.maximum(self.fvol[s] - (g / D_WATER).astype(self.fvol.dtype), 0.0)
        np.add.at(self.E, s, (-g * cell).astype(self.E.dtype))
        np.add.at(self.E, t, (g * gas).astype(self.E.dtype))
        gone = self.fvol[s] <= 0.0
        if gone.any():
            sg = tuple(a[gone] for a in s)
            self.fl[sg] = NOFLUID
            self.fpot[sg] = 0.0
        self.wet.stats["evap_g"] += float(g[g > 0].sum())
        self.wet.stats["cond_g"] += float(-g[g < 0].sum())

    def _surface_sinks(self, dt):
        """Vapour condenses on a solid face colder than its dew point: the water forms in
        the gas voxel (it runs off by the fluid law), the wall takes the latent heat."""
        st = self.wet.store
        A, _V = self._face()
        ex = st.excess("H2O")
        there = np.abs(ex) > TINY_G
        box = self._window(there, 1)
        if box is None:
            return
        gas = self._gas_cells()
        C = self.heat_capacity()
        sub = box
        ge = gas[sub]
        for d in _DIRS:
            gi = np.nonzero(ge)
            gx, gy, gz = (gi[0] + sub[0].start, gi[1] + sub[1].start, gi[2] + sub[2].start)
            nx, ny, nz = gx + d[0], gy + d[1], gz + d[2]
            inb = ((nx >= 0) & (nx < self.shape[0]) & (ny >= 0) & (ny < self.shape[1])
                   & (nz >= 0) & (nz < self.shape[2]))
            gx, gy, gz, nx, ny, nz = (a[inb] for a in (gx, gy, gz, nx, ny, nz))
            solid = ((self.mat[nx, ny, nz] != AIR) & (self.mat[nx, ny, nz] != ICE)
                     & ~((self.fl[nx, ny, nz] == WATER) & (self.fvol[nx, ny, nz] > 0)))
            if not solid.any():
                continue
            gx, gy, gz, nx, ny, nz = (a[solid] for a in (gx, gy, gz, nx, ny, nz))
            g_ = (gx, gy, gz)
            n_ = (nx, ny, nz)
            Ts = np.clip(AMBIENT + self.E[n_].astype(np.float64) / C[n_], -80.0, 370.0)
            Vg = gas_volume(self.vox_l, self.fvol[g_].astype(np.float64))
            Tg = np.clip(AMBIENT + self.E[g_].astype(np.float64) / C[g_], -80.0, 370.0)
            rho_air = st.bg("H2O") + ex[g_].astype(np.float64) / Vg
            cold = (Ts > 0.0) & (psychro.rho_sat(Ts) < rho_air)
            if not cold.any():
                continue
            g_ = tuple(a[cold] for a in g_)
            n_ = tuple(a[cold] for a in n_)
            Ts, Vg, Tg, rho_air = Ts[cold], Vg[cold], Tg[cold], rho_air[cold]
            x_g = rho_air * R * (Tg + 273.15) / M_WATER / P_ATM
            gm = psychro.surface_step(psychro.rho_sat(Ts), rho_air, Vg, A, H_CONV,
                                      psychro.stefan_factor(x_g), 0.0, dt)      # negative
            room = (self.cap - self.fvol[g_].astype(np.float64)) * D_WATER
            plain = (self.fl[g_] == NOFLUID) | (self.fl[g_] == WATER)
            gm = np.where(plain, -np.minimum(-gm, np.maximum(room, 0.0)), 0.0)
            applied = st.add("H2O", g_, gm)                # removes vapour (negative)
            cond = -applied                                  # grams that became liquid
            hit = cond > 0
            if not hit.any():
                continue
            g2 = tuple(a[hit] for a in g_)
            n2 = tuple(a[hit] for a in n_)
            c = cond[hit]
            Tgh, Tsh = Tg[hit], Ts[hit]
            self.fl[g2] = WATER
            np.add.at(self.fvol, g2, (c / D_WATER).astype(self.fvol.dtype))
            # solid: h_fg(Tg); gas voxel: net (cw - cv)(Tg - 20); stored -L0 (in the store)
            hfg = psychro.h_fg(Tgh)
            np.add.at(self.E, n2, (c * hfg).astype(self.E.dtype))
            np.add.at(self.E, g2, (c * (CW - CV) * (Tgh - AMBIENT)).astype(self.E.dtype))
            self.wet.stats["cond_g"] += float(c.sum())

    # ---------------------------------------------------------------- vapour spread, edges
    def _wet_box(self, pad=1):
        st = self.wet.store
        there = None
        for nm in ("H2O", "MIST"):
            if st.has(nm):
                m = np.abs(st.excess(nm)) > TINY_G
                there = m if there is None else (there | m)
        return None if there is None else self._window(there, pad)

    def _wet_seep(self):
        """Conservative diffusion of the vapour over gas cells (gas_flow off). Windowed to
        the box the vapour is in; the total is exact to float32 rounding."""
        box = self._wet_box(1)
        if box is None:
            return
        a = self.wet.store.excess("H2O")
        e = a[box].astype(np.float64)
        air = self._gas_cells()[box]
        d = np.zeros_like(e)
        for ax in range(3):
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[ax] = slice(0, -1)
            hi[ax] = slice(1, None)
            lo, hi = tuple(lo), tuple(hi)
            open_ = air[lo] & air[hi]
            f = np.where(open_, SEEP_K * (e[lo] - e[hi]), 0.0)
            d[lo] -= f
            d[hi] += f
        a[box] = (e + d).astype(a.dtype)
        self.wet.store.generation += 1

    def _wet_edges(self):
        """The world is open at its edges: vapour above the background leaves (or comes
        in, if the cell is below it) and is booked to gone['vapour'] with its latent."""
        st = self.wet.store
        a = st.excess("H2O")
        gas = None
        got = 0.0
        for ax in range(3):
            for end in (0, -1):
                sl = [slice(None)] * 3
                sl[ax] = end
                sl = tuple(sl)
                slab = a[sl]
                if not slab.any():
                    continue
                if gas is None:
                    gas = self._gas_cells()
                m = gas[sl] & (slab != 0)
                if not m.any():
                    continue
                v = slab[m].astype(np.float64)
                move = v * EDGE_RELAX
                slab[m] = (v - move).astype(a.dtype)
                got += float(move.sum())
        if got != 0.0:
            self.gone["vapour"] = self.gone.get("vapour", 0.0) + got
            self.shed += L0 * got
            st.generation += 1

    # ---------------------------------------------------------------- mist
    def _wet_mist(self, dt):
        """Saturation adjustment in gas voxels that hold more vapour than their air can,
        and the evaporation of mist back into voxels that can take it."""
        st = self.wet.store
        box = self._wet_box(0)
        if box is None:
            return
        ex = st.excess("H2O")
        mi = st.excess("MIST")
        gas = self._gas_cells()[box]
        C = self.heat_capacity()[box]
        E = self.E[box]
        fv = self.fvol[box]
        V = gas_volume(self.vox_l, fv.astype(np.float64))
        bg = st.bg("H2O")
        m_vap = bg * V + ex[box].astype(np.float64)
        T0 = np.clip(AMBIENT + E.astype(np.float64) / C, -80.0, 370.0)
        # condense
        sup = gas & (m_vap > psychro.rho_sat(T0) * V * (1.0 + 1e-9))
        if sup.any():
            ii = np.nonzero(sup)
            mc, _T1 = psychro.saturation_adjust(m_vap[ii], T0[ii], V[ii], C[ii].astype(np.float64))
            mc = np.minimum(mc, m_vap[ii])
            gi = tuple(a + b.start for a, b in zip(ii, box))
            applied = -st.add("H2O", gi, -mc)
            st.add("MIST", gi, applied)
            np.add.at(self.E, gi, (L0 * applied).astype(self.E.dtype))
            self.wet.stats["mist_g"] += float(applied.sum())
        # evaporate mist where there is room
        mm = mi[box]
        if mm.any():
            m_vap = bg * V + ex[box].astype(np.float64)
            T0 = np.clip(AMBIENT + self.E[box].astype(np.float64) / C, -80.0, 370.0)
            ii = np.nonzero(mm > 0)
            dm = psychro.mist_evaporate(mm[ii].astype(np.float64), m_vap[ii], T0[ii], V[ii], dt)
            gi = tuple(a + b.start for a, b in zip(ii, box))
            took = -st.add("MIST", gi, -dm)
            st.add("H2O", gi, took)
            np.add.at(self.E, gi, (-L0 * took).astype(self.E.dtype))

    # ---------------------------------------------------------------- ice
    def _wet_ice(self, dt):
        """Pure water below its freezing point freezes (enthalpy method, supercooling by
        situation) and ice above it melts. Ice is a gram count in a cell until it fills
        ICE_FILL of the cell, then the cell is mat ICE (hysteresis ICE_UNFILL).
        Levels are worked bottom-up, so water pushed out by freezing at one level is
        seen by the level above it."""
        w = self.wet
        isice = (self.mat == ICE) if w.has_ice else np.zeros(self.shape, bool)
        has_g = self.ice_g is not None and w.has_ice and bool(self.ice_g.any())
        wcell = (self.fl == WATER) & (self.fvol > 0)
        if self.mixes_live:
            wcell &= (self.fcomp == 0)
        if not (wcell.any() or isice.any() or has_g):
            w.has_ice = False
            return
        C = self.heat_capacity()
        cand = isice
        if has_g:
            cand = cand | (self.ice_g > 0)
        if wcell.any():
            idxw = np.nonzero(wcell)
            Tw = AMBIENT + self.E[idxw].astype(np.float64) / C[idxw]
            cold = Tw <= 0.0
            if cold.any():
                cand = cand.copy()
                cand[tuple(a[cold] for a in idxw)] = True
        if not cand.any():
            return
        idx = np.nonzero(cand)
        for z in np.unique(idx[2]):
            m = idx[2] == z
            self._ice_level(tuple(a[m] for a in idx), dt)
        w.ice_gen += 1
        w.has_ice = bool((self.mat == ICE).any()) or bool(self.ensure_ice().any())

    def _expel_target(self, idx):
        """For each cell: the z of the first cell above it, in its column, that has room
        for water (walking up through cells that are full of water), and the room in ml.
        z = -1 where there is none (a lid, ice, the top of the world)."""
        nz = self.shape[2]
        x, y = idx[0], idx[1]
        tz = idx[2] + 1
        out = np.full(len(x), -1, np.int64)
        room = np.zeros(len(x))
        searching = np.ones(len(x), bool)
        for _ in range(nz):
            inb = searching & (tz < nz)
            if not inb.any():
                break
            zc = np.minimum(tz, nz - 1)
            m = inb & (self.mat[x, y, zc] == AIR) & ((self.fl[x, y, zc] == NOFLUID)
                                                     | (self.fl[x, y, zc] == WATER))
            r = np.maximum(self.cap - self.fvol[x, y, zc].astype(np.float64), 0.0)
            hit = m & (r > 1e-3 * self.cap)
            out = np.where(hit, tz, out)
            room = np.where(hit, r, room)
            searching = m & ~hit
            tz = np.where(searching, tz + 1, tz)
        return out, room

    def _ice_level(self, idx, dt):
        w = self.wet
        C = self.heat_capacity()
        ice_g = self.ensure_ice()
        ismat = self.mat[idx] == ICE
        m_i = np.where(ismat, self.smass[idx], ice_g[idx]).astype(np.float64)
        m_w = np.where(self.fl[idx] == WATER, self.fvol[idx], 0.0).astype(np.float64) * D_WATER
        E0 = self.E[idx].astype(np.float64)
        Cc = C[idx].astype(np.float64)
        C_other = Cc - CW * m_w - CI * m_i
        # nucleation: ice in the cell, or a solid / the free surface beside it
        hetero = (m_i > 0)
        for d in _DIRS:
            nx = np.clip(idx[0] + d[0], 0, self.shape[0] - 1)
            ny = np.clip(idx[1] + d[1], 0, self.shape[1] - 1)
            nz = np.clip(idx[2] + d[2], 0, self.shape[2] - 1)
            hetero |= (self.mat[nx, ny, nz] != AIR)
            if d == (0, 0, 1):
                hetero |= (self.mat[nx, ny, nz] == AIR) & (self.fvol[nx, ny, nz] < 0.5 * self.cap)
        dT = IP.nucleation_k(ice_in_cell=hetero)
        out = IP.settle(E0, C_other, m_w, m_i, 0.0, 0.0, dT, dt, liquidus=IP.PURE)
        d_ice = out.m_i - m_i                                   # positive: freezing
        # FREEZING NEEDS ROOM. Ice (917 g/L) takes 1.09 times the volume of its water. The
        # surplus is pushed up the column to the first cell with room; what has nowhere to
        # go is not frozen (mass stays exact, the enthalpy is rebuilt from the ice it has).
        expelled = np.zeros_like(d_ice)
        tz = np.full(len(d_ice), -1, np.int64)
        frz = d_ice > 1e-12
        if frz.any():
            extra = 1.0 / IP.RHO_ICE - 1.0
            tz, room = self._expel_target(idx)
            vol0 = m_w / D_WATER + m_i / IP.RHO_ICE
            d_ok = np.minimum(d_ice, IP.RHO_ICE * self.cap - m_i)
            d_ok = np.minimum(d_ok, (self.cap - vol0 + room) / extra)
            d_ice = np.where(frz, np.clip(d_ok, 0.0, None), d_ice)
            expelled = np.where(frz, np.maximum(vol0 + extra * d_ice - self.cap, 0.0), 0.0)
            expelled = np.minimum(expelled, np.maximum(m_w - d_ice, 0.0))
        ice_new = m_i + d_ice
        # enthalpy E - L_CONV20 * ice is kept: E rises by L_CONV20 for every gram frozen
        E_new = E0 + L_CONV20 * d_ice
        W_new = m_w - d_ice
        push = None
        if (expelled > 0).any():
            e_i = np.nonzero(expelled > 0)[0]
            ex_g = expelled[e_i]
            Tn = AMBIENT + E_new[e_i] / (C_other[e_i] + CW * W_new[e_i] + CI * ice_new[e_i])
            gx = (idx[0][e_i], idx[1][e_i], tz[e_i])
            W_new[e_i] -= ex_g
            sens = CW * (Tn - AMBIENT) * ex_g
            E_new[e_i] -= sens
            push = (gx, ex_g, sens)
        self.E[idx] = E_new.astype(self.E.dtype)
        wl = np.maximum(W_new, 0.0)
        self.fvol[idx] = (wl / D_WATER).astype(self.fvol.dtype)
        liq = wl > 0
        self.fl[tuple(a[liq] for a in idx)] = WATER
        gone_l = ~liq & (self.fl[idx] == WATER)
        if gone_l.any():
            gi = tuple(a[gone_l] for a in idx)
            self.fl[gi] = NOFLUID
            self.fpot[gi] = 0.0
        if push is not None:                 # the expelled water arrives above
            gx, ex_g, sens = push
            self.fl[gx] = WATER
            np.add.at(self.fvol, gx, (ex_g / D_WATER).astype(self.fvol.dtype))
            np.add.at(self.E, gx, sens.astype(self.E.dtype))
        # promote / demote
        vl = self.vox_l * 1000.0
        frac = (ice_new / IP.RHO_ICE) / vl
        now = np.where(ismat, frac >= IP.ICE_UNFILL, frac >= IP.ICE_FILL)
        to_ice = now & ~ismat & (self.mat[idx] == AIR)
        to_gas = ismat & ~now
        stay = ismat & now
        if stay.any():
            si = tuple(a[stay] for a in idx)
            self.smass[si] = ice_new[stay].astype(self.smass.dtype)
        if to_ice.any():
            ti = tuple(a[to_ice] for a in idx)
            self.mat[ti] = ICE
            self.smass[ti] = ice_new[to_ice].astype(self.smass.dtype)
            ice_g[ti] = 0.0
        keep_g = ~ismat & ~to_ice
        if keep_g.any():
            kg = tuple(a[keep_g] for a in idx)
            ice_g[kg] = ice_new[keep_g].astype(ice_g.dtype)
        if to_gas.any():
            gi = tuple(a[to_gas] for a in idx)
            self.mat[gi] = AIR
            self.smass[gi] = 0.0
            ice_g[gi] = ice_new[to_gas].astype(ice_g.dtype)

    # ---------------------------------------------------------------- the gasflow hooks
    def _gf_extra_there(self):
        """Cells with vapour to carry (gasflow.py asks this every tick)."""
        w = self.wet
        if w is None or not w.store.has("H2O"):
            return None
        return np.abs(w.store.excess("H2O")) > 1e-6

    def _gf_extra_fields(self, reg):
        """[(name, array)] of the extra fields the flow carries in region `reg`."""
        w = self.wet
        if w is None or not w.store.has("H2O"):
            return []
        a = w.store.excess("H2O")[reg]
        if not a.any():
            return []
        return [("H2O", a.astype(np.float32))]

    def _gf_extra_done(self, reg, name, arr, left):
        """Write the carried field back; `left` grams went out through the world's edge."""
        w = self.wet
        w.store.excess(name)[reg] = arr
        w.store.generation += 1
        if left:
            self.gone["vapour"] = self.gone.get("vapour", 0.0) + left
            self.shed += L0 * left


class WetWorld(WetMixin, ColdWorld):
    pass
