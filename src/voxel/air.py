"""Air laws: coarse pressure/wind field, smoke transport, airspace regions, oxygen."""
import numpy as np
from .kernels import (HAVE_NUMBA, _blur_nb, _carry_gas, _cells_sum, _gas_overturn, _o2_diffuse,
                      _upcell_at, _upcell_nb)
from .gasflow import GAS_DIFFUSIVITY, GasFlowMixin
from .label import regions
from .materials import (AIR, AMBIENT, O2_PER_L, PLUME_REACH_M, SMOKE_STILL, TICK_S)

# THE GAS-FLOW SWITCH (docs/realism-map.md, section 6, lane 7). Off: the lossy
# pressure field, `_law_rise`, the smoke rise and ceiling jet, and the two-zone
# stir below are the laws, exactly as they were. On: the buoyant flow of
# gasflow.py carries heat, smoke and oxygen, and those stand-ins step aside.
# A world's own `gas_flow` is read first; this is what a new world starts with.
# Turning it on for good also means deleting `_law_rise` and its call.
GAS_FLOW = False


# HOW FAST PRESSURE FOLLOWS THE FLOW (wave 13b, C6). It was 0.08. With the velocity
# gain 0.12 below, that made the pneumatic field LINEARLY UNSTABLE in three
# dimensions: a random seed of pressure and velocity in a 6x6x6 block of cells
# grows by 1.0164 a tick (1.025 at 8x8x8), a 1-D or 2-D field is stable (0.965,
# 0.967), and any plume that lasted longer than about a thousand ticks drove the
# field to float32 overflow and NaN (a 40 C heater voxel in a small room: wind
# 6e6 at tick 2000, NaN at 7000; measured on base20 as well). Nobody saw it
# because a fire lasts a few hundred ticks in a test and a plume of any other
# kind did not exist; a standing person is a plume that never ends (T7). At 0.03
# the field decays in every size tried (100x80x20 cells: 7e-8 over 1500 ticks;
# 0.04 grows 1.19 there and 0.045 grows 5000-fold). The forced wind over a warm
# column is within 15 percent of what it was (0.23 against 0.20); the pressure
# it builds is smaller (0.03 against 0.07), and pressure that a fire INJECTS
# (`p_add`) drives the wind with the same gain as before.
P_FOLLOW = 0.03


def _no_rise(sl=None):
    """`_law_rise`, while the gas flow is on: the flow lifts the heat."""
    return None


# placeholders for `_cells_sum`'s unused arguments (a typed array each)
_NO_MASK = np.zeros((1, 1, 1), bool)
_ONE32 = np.ones((1, 1, 1), np.float32)


class AirMixin(GasFlowMixin):

    _CS = 4                                              # air-field cell = 4^3 voxels

    @property
    def gas_flow(self):
        """Is the buoyant gas flow (gasflow.py) the law of the air? The heat
        law's own `_law_rise` is the other half of the old stand-ins; it is
        put aside on this world while the flow is on (it would lift the same
        heat twice)."""
        on = self.__dict__.get("_gas_flow", GAS_FLOW)
        if on and "_law_rise" not in self.__dict__:
            self.__dict__["_law_rise"] = _no_rise
        return on

    @gas_flow.setter
    def gas_flow(self, on):
        self.__dict__["_gas_flow"] = bool(on)
        if on:
            self.__dict__["_law_rise"] = _no_rise
        else:
            self.__dict__.pop("_law_rise", None)
            self.__dict__.pop("_gf", None)

    def _blur_cells(self, carr, passes):
        """Neighbor-mean blur on the coarse cell grid — the shape of a plume's
        entrainment field: strongest at the fire, fading over `passes` cells.

        Compiled when the world is fused (`_blur_nb`, the same sums in the
        same order); the numpy below is the plain path and the definition."""
        if self.fused and HAVE_NUMBA and passes > 0 \
                and carr.dtype in (np.float32, np.float64):
            return _blur_nb(carr, passes)
        out = carr
        for _ in range(passes):
            m2 = out.copy(); c2 = np.ones_like(out)
            for axis in range(3):
                a = [slice(None)] * 3; b = [slice(None)] * 3
                a[axis], b[axis] = slice(None, -1), slice(1, None)
                a, b = tuple(a), tuple(b)
                m2[a] += out[b]; c2[a] += 1
                m2[b] += out[a]; c2[b] += 1
            out = m2 / c2
        return out

    def _cells_fast(self, arr):
        """Can `_cells_sum` give numpy's block sums for this array? A float
        array, and a world more than one block high (kernels.py says why)."""
        return self.fused and HAVE_NUMBA and self.shape[2] > self._CS \
            and arr.dtype in (np.float32, np.float64)

    def _cells_where(self, mask, arr=None):
        """`_cells(np.where(mask, arr, 0.0))`, or `_cells(mask.astype(
        np.float32))` with no `arr`, without the whole-lattice temporary."""
        a = _ONE32 if arr is None else arr
        if self._cells_fast(a):
            return _cells_sum(a, mask, True, arr is None, self._CS, True)
        return self._cells(mask.astype(np.float32) if arr is None
                           else np.where(mask, arr, 0.0))

    def _cells(self, arr, how="mean"):
        """Downsample a voxel array onto the coarse cell grid."""
        if self._cells_fast(arr):
            return _cells_sum(arr, _NO_MASK, False, False, self._CS, how == "mean")
        cs = self._CS
        nx, ny, nz = self.shape
        px, py, pz = (-nx) % cs, (-ny) % cs, (-nz) % cs
        # A WORLD THAT ALREADY DIVIDES EVENLY NEEDS NO PADDING, and np.pad copies
        # the whole lattice whether it has anything to add or not. Most worlds
        # divide evenly — this was a spare full-size copy of a 4 MB array, made
        # eleven times a tick, for nothing.
        a = arr if (px or py or pz) == 0 else np.pad(arr, ((0, px), (0, py), (0, pz)))
        a = a.reshape(a.shape[0] // cs, cs, a.shape[1] // cs, cs, a.shape[2] // cs, cs)
        return a.mean(axis=(1, 3, 5)) if how == "mean" else a.sum(axis=(1, 3, 5))

    def _upcell(self, carr):
        """Broadcast a cell array back to voxel resolution."""
        cs = self._CS
        nx, ny, nz = self.shape
        if self.fused and HAVE_NUMBA:
            return _upcell_nb(carr, nx, ny, nz, cs)
        return np.repeat(np.repeat(np.repeat(carr, cs, 0), cs, 1), cs, 2)[:nx, :ny, :nz]

    def _upcell_where(self, carr, mask):
        """`_upcell(carr)[mask]`: only the cells that are read."""
        if self.fused and HAVE_NUMBA:
            return _upcell_at(carr, mask, self._CS)
        return self._upcell(carr)[mask]

    def _law_air(self):
        """PRESSURE, pneumatic half — a coarse LOSSY gas field (the falling-sand
        lineage's design, in 3D): pressure and velocity live on 4-voxel cells.
        Fires and boiling INJECT pressure (hot gas expands); velocity follows the
        pressure gradient; pressure follows the velocity divergence; both decay.
        Walls block. The open sky bleeds pressure away. The field then CARRIES
        smoke and oxygen (advection) — a fire now pushes its smoke through a
        doorway instead of politely diffusing.

        With the gas flow on, none of this is the law: the flow solves its own
        pressure and carries the gases (gasflow.py)."""
        if self.gas_flow:
            return self._law_gasflow()
        if self.pcell is None:
            cshape = tuple((s + self._CS - 1) // self._CS for s in self.shape)
            self.pcell = np.zeros(cshape, np.float32)
            self.vcell = [np.zeros(cshape, np.float32) for _ in range(3)]
        # what the coarse pressure cells can breathe through, read from POROSITY
        # rather than a headcount of non-air voxels: a shut door that fills 97%
        # of its cells is a path, and the old binary read rounded its gap away.
        # The block stays HARD, not graded — these cells are 20 cm wide, so a
        # cell half-filled by a wall has that wall across its whole face, and
        # grading it by volume fraction let sealed rooms bleed pressure through
        # solid masonry (measured: a sealed fire stopped out-pressuring an open
        # one). Coarse geometry cannot support a soft answer here.
        cpor = self._cells(self.porosity())
        blocked = cpor < 0.05
        self.pcell += self._cells(self.p_add, "sum")
        self.p_add[:] = 0.0
        p, v = self.pcell, self.vcell
        # convection: warm cells RISE — this closes the loop that gives a room a
        # real breeze: up over the fire, out along the ceiling, down the cool
        # walls, back along the floor. Without it the wind only pushes outward
        # and smoke piles into corners.
        warm = self._warm_cells()
        v[2] += np.where(~blocked, warm * 0.0015, 0.0)
        for axis in range(3):                            # velocity chases the gradient
            a = [slice(None)] * 3; b = [slice(None)] * 3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            g = (p[a] - p[b]) * 0.12
            ok = ~blocked[a] & ~blocked[b]
            v[axis][a] += np.where(ok, g, 0.0)
            v[axis][b] += np.where(ok, g, 0.0)
            div_pair = v[axis][a] - v[axis][b]           # and pressure follows the flow
            p[a] -= P_FOLLOW * np.where(ok, div_pair, 0.0)
            p[b] += P_FOLLOW * np.where(ok, div_pair, 0.0)
        for arr in (p, *v):                              # diffuse + decay (lossy on
            m = arr.copy()                               # purpose — game-feel field)
            cnt = np.ones_like(arr)
            for axis in range(3):
                a = [slice(None)] * 3; b = [slice(None)] * 3
                a[axis], b[axis] = slice(None, -1), slice(1, None)
                a, b = tuple(a), tuple(b)
                m[tuple(a)] += arr[tuple(b)]; cnt[tuple(a)] += 1
                m[tuple(b)] += arr[tuple(a)]; cnt[tuple(b)] += 1
            arr *= 0.8
            arr += 0.2 * (m / cnt)
            arr *= 0.965
        p[blocked] = 0.0
        for axis in range(3):
            v[axis][blocked] = 0.0
        if self.open_sky:
            sky = ~blocked[:, :, -1]
            p[:, :, -1][sky] *= 0.3
        # advection: the wind carries the gases, upwind. A face passes what its
        # PORES allow — open air freely, a shut door at a hundredth. This is the
        # half that matters for a closed door: diffusion through a 1%-porous leaf
        # is nearly nil (measured), but a fire PRESSURISES the room it is in and
        # pushes its smoke through the gaps. That push is why a shut door is a
        # delay and not a seal.
        air = self.mat == AIR
        por = self.porosity()
        if (self.fused and HAVE_NUMBA and self.smoke is not None
                and self.smoke.dtype == por.dtype == v[0].dtype == np.float32):
            for axis in range(3):                        # (the loop below, fused:
                d = [0, 0, 0]; d[axis] = 1               # `_carry_gas`)
                _carry_gas(self.smoke, por, v[axis], d[0], d[1], d[2], self._CS)
            return
        for axis in range(3):
            w = np.clip(self._upcell(v[axis]) * 0.25, -0.4, 0.4)
            a = [slice(None)] * 3; b = [slice(None)] * 3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            pair_ok = np.minimum(por[a], por[b])
            wface = w[a]
            # the wind carries SMOKE only, still. Second attempt at O2-advection
            # (2026-09-04), measured and rejected for a DIFFERENT reason than the
            # first: the lossy compressible field plus set-value boundaries acts
            # as a mass PUMP — the house-fire room sloshed itself down to 2% O2
            # with no fire burning at all. Composition moves by diffusion and by
            # buoyant OVERTURN in _law_o2 (hot-poor gas trades places with
            # cool-rich gas), which is local, conservative, and can't pump.
            for field in (self.smoke,):
                if field is None:
                    continue
                fwd = np.where(wface > 0, wface * field[a], 0.0) * pair_ok
                back = np.where(wface < 0, -wface * field[b], 0.0) * pair_ok
                field[a] += back - fwd
                field[b] += fwd - back
        # (the old 1.5x-fresh O2 clip is gone: clipping DESTROYED whatever the
        # wind compressed past the cap — the world was quietly losing oxygen
        # every tick. Compressed pockets are fine: burn health caps at 1, so
        # rich air is a reservoir, never a rate boost.)

    # The most voxels smoke moves along ONE axis in a single pass of the law
    # below: a lateral escape, a seep, and two turns of the ceiling jet. Counted
    # off the code, not guessed. Add a phase that shifts smoke sideways and this
    # must go up — and if you forget, the tick-by-tick guard says so, which is
    # why it is safe to have a number here at all.
    _SMOKE_REACH = 4

    def _law_smoke(self, sl=None):
        """Smoke is a gas: it seeps to neighbors, RISES hard through open air, and
        leaves at the open sky. The mass that burning removes from wood and oil
        travels here — conservation made visible.

        Run over a box around the smoke there IS, because this law makes some
        thirteen passes over a field that is empty nearly everywhere: a fire in
        one hut of a town had every pass sweeping the whole town."""
        sl = sl if sl is not None else (slice(None),) * 3
        smoke = self.smoke[sl]                    # a view: writing writes the world
        air = self.mat[sl] == AIR
        por = self.porosity()[sl]
        # smoke is born INSIDE burning solids — it must escape through the surface
        # first (up if it can, sideways if it must), or it stays trapped in the wood
        lo = (slice(None), slice(None), slice(None, -1))
        up = (slice(None), slice(None), slice(1, None))
        esc = np.where(~air[lo] & air[up], 0.9 * smoke[lo], 0.0).astype(np.float32)
        smoke[lo] -= esc
        smoke[up] += esc
        for a, b in ((( slice(None, -1), slice(None), slice(None)), (slice(1, None), slice(None), slice(None))),
                     ((slice(None), slice(None, -1), slice(None)), (slice(None), slice(1, None), slice(None)))):
            for s_, d_ in ((a, b), (b, a)):
                esc = np.where(~air[s_] & air[d_], 0.5 * smoke[s_], 0.0).astype(np.float32)
                smoke[s_] -= esc
                smoke[d_] += esc
        flow = self.gas_flow          # the flow lifts smoke and runs the ceiling jet
        # a flow carries smoke far better than a share of the difference does:
        # what is left to seep is a diffusivity in m^2/s (no lattice's number)
        seep = (min(GAS_DIFFUSIVITY * TICK_S / (0.1 * self.scale) ** 2, 0.1)
                if flow else 0.08)
        for axis in range(3):                                # seep: even out with neighbors
            a = [slice(None)] * 3; b = [slice(None)] * 3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            q = seep * (smoke[a] - smoke[b]) * np.minimum(por[a], por[b])
            smoke[a] -= q
            smoke[b] += q
        at_roof = sl[2].stop in (None, self.shape[2])     # is the window's top
        if flow:
            if at_roof and self.open_sky:        # the open sky takes it, and says so
                top = (slice(None), slice(None), -1)
                was = float(smoke[top][air[top]].sum(dtype=np.float64))
                smoke[top][air[top]] *= 0.5
                self.gone["smoke"] = self.gone.get("smoke", 0.0) + was * 0.5
            return
        lo = (slice(None), slice(None), slice(None, -1))
        up = (slice(None), slice(None), slice(1, None))
        risable = air[lo] & air[up]                          # rise: buoyant, fast
        q = np.where(risable, 0.4 * smoke[lo], 0.0).astype(np.float32)
        smoke[lo] -= q
        smoke[up] += q
        # the CEILING JET: smoke that can rise no further pools against the
        # ceiling and RACES along it — buoyancy makes it an upside-down liquid
        # seeking an upside-down level. Without this the bank crawled at
        # diffusion pace and the far half of a burning room read 0.00 g
        # forever (measured; a real ceiling jet runs meters per second).
        pinned = np.zeros_like(air)
        pinned[:, :, :-1] = air[:, :, :-1] & ~air[:, :, 1:]  # air with solid above
        pinned[:, :, -1] = air[:, :, -1] & (not self.open_sky) & at_roof
        for _ in range(2):
            for axis in (0, 1):
                a = [slice(None)] * 3; b = [slice(None)] * 3
                a[axis], b[axis] = slice(None, -1), slice(1, None)
                a, b = tuple(a), tuple(b)
                m = pinned[a] & pinned[b]
                q = np.where(m, 0.25 * (smoke[a] - smoke[b]),
                             0.0).astype(np.float32)
                smoke[a] -= q
                smoke[b] += q
        if at_roof:                                          # the open sky takes it
            top = (slice(None), slice(None), -1)             # — but only the real
            smoke[top][air[top]] *= 0.5                      # sky, not a box edge

    def _air_regions(self):
        """Label the connected airspaces (6-connectivity, each airspace named by
        its largest flat index). A fire can only breathe the air ITS OWN airspace
        holds — a sealed wall is a real wall, a doorway joins two rooms into one
        lung. Recomputed only when the set of air voxels changes (a wall burns
        through, a collapse opens a gap — or, far more often, a person moves:
        flesh is not air). That last case made this flood half of a house tick,
        so it runs through the shared labeller (`label.py`), whose compiled path
        gives the same array as the numpy max-propagation it replaced."""
        air = self.mat == AIR
        if self._region_air is not None and bool((air == self._region_air).all()):
            return
        self._aircells = self._airblur = None    # the walls moved: so did the air
        self.air_region, self._region_count = regions(air, fused=self.fused)
        self._region_air = air

    def _region_of_surface(self):
        """For every voxel, the airspace it touches (max over the 6 neighbors) —
        this is the air a burning SOLID voxel breathes from. -1: touches none."""
        r = self.air_region
        out = r.copy()
        out[1:, :, :] = np.maximum(out[1:, :, :], r[:-1, :, :])
        out[:-1, :, :] = np.maximum(out[:-1, :, :], r[1:, :, :])
        out[:, 1:, :] = np.maximum(out[:, 1:, :], r[:, :-1, :])
        out[:, :-1, :] = np.maximum(out[:, :-1, :], r[:, 1:, :])
        out[:, :, 1:] = np.maximum(out[:, :, 1:], r[:, :, :-1])
        out[:, :, :-1] = np.maximum(out[:, :, :-1], r[:, :, 1:])
        return out

    def _region_at(self, mask):
        """`_region_of_surface()[mask]`: the same max over each cell and its six
        neighbours, worked out for the cells of `mask` alone, in the order
        boolean indexing gives them. `mask` may also be those cells' indices
        (a tuple, as np.nonzero gives them)."""
        r = self.air_region
        at = mask if isinstance(mask, tuple) else np.nonzero(mask)
        out = r[at]
        for axis in range(3):
            for d in (-1, 1):
                c = list(at)
                c[axis] = at[axis] + d
                ok = (c[axis] >= 0) & (c[axis] < r.shape[axis])
                out[ok] = np.maximum(out[ok], r[tuple(a[ok] for a in c)])
        return out

    def _law_o2(self):
        """Oxygen: everywhere air reaches, including a solid's pores; fire eats it (in
        the burn law); movement evens it out. The mixing rate RISES with temperature —
        a fire violently stirs the air that feeds it (convective entrainment), cold air
        barely moves. An open sky refills the top; in a SEALED space (open_sky=False)
        the fire eats what the room holds, then stops — fuel or no fuel.

        MIXING IS WINDOWED, and unlike smoke it needs no judgement to be. Air
        that is everywhere at the same composition has nothing to even out —
        every pair cancels — so the box around where the field is NOT fresh
        contains every exchange there is, and the skip is exact. Measured in a
        burning town, that box is 21% of the world at tick 20 and 63% at tick
        69: the deficit really does stay where the fire is, which is exactly
        what smoke does NOT do (an infinitesimal trace of soot reaches every
        cell within ten ticks, so `smoke != 0` is useless and smoke needs a
        stated threshold instead)."""
        T = self.T()
        por = self.porosity()          # asked before the gate, as it always was
        if self._o2_asleep(T):
            return
        air = self.mat == AIR
        fused = self.fused and HAVE_NUMBA
        # one hop per axis for diffusion; the overturn below adds one more in z
        mix = self._window(self.o2 != O2_PER_L * self.vox_l, 1)
        for axis in range(3):
            if self.gas_flow:          # the flow carries the oxygen; what is left
                break                  # is the sky refilling the edges, below
            sl = mix if mix is not None else (slice(None),) * 3
            if fused:
                d = [0, 0, 0]; d[axis] = 1
                _o2_diffuse(self.o2[sl], T[sl], por[sl], d[0], d[1], d[2])
                continue
            a = [slice(None)] * 3; b = [slice(None)] * 3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            o2, Tw, pw = self.o2[sl], T[sl], por[sl]
            k = np.clip(0.1 + (Tw[a] + Tw[b]) / 1600.0, 0.1, 0.45)
            k = k * np.minimum(pw[a], pw[b])         # a barrier passes what its PORES
            q = k * (o2[a] - o2[b])                  # allow: a shut door breathes a
            o2[a] -= q                               # little, masonry almost nothing,
            o2[b] += q                               # a pane nothing at all
        # (the old 0.1-per-tick pull toward the REGION MEAN is gone: it made every
        # airspace an instantly-stirred lung — head air always equaled knee air,
        # and a fire's deficit teleported to every nose in the room.)
        # BUOYANT OVERTURN — the vertical transport that builds the two-layer
        # fire room (CFAST's physics, done per-pair): where warmer gas sits
        # UNDER cooler gas the column is unstable and the parcels trade places,
        # each keeping its composition — so the fire's hot, oxygen-poor exhaust
        # climbs to the ceiling and the cool fresh air it displaces slides down
        # to the floor. Crawl-low is this law's shadow. A stable column (cool
        # below hot) doesn't move: stratification, once built, PERSISTS.
        lo = (slice(None), slice(None), slice(None, -1))
        hi = (slice(None), slice(None), slice(1, None))
        # ASKED AGAIN. The diffusion above has just moved oxygen, so the box it
        # was drawn around is one voxel out of date in z — the same mistake the
        # heat window made on its first day, when conduction carried heat up one
        # cell and buoyancy carried it up another, one past the edge of a box
        # drawn before either ran. Smoke rides along, so its own spread counts
        # too.
        flow = self.gas_flow           # the flow IS the overturn and the stir
        turn = None if flow else self._window(
            (self.o2 != O2_PER_L * self.vox_l) | (self.smoke > SMOKE_STILL), 1)
        tsl = turn if turn is not None else (slice(None),) * 3
        if flow:
            pass
        elif fused:
            _gas_overturn(self.o2[tsl], self.smoke[tsl], T[tsl], air[tsl])
        else:
            Tw, aw = T[tsl], air[tsl]
            dT = Tw[lo] - Tw[hi]                         # >0 = unstable: hot below
            f = np.where(aw[lo] & aw[hi],
                         np.clip(dT * 0.02, 0.0, 0.45), 0.0).astype(np.float32)
            for gas in (self.o2[tsl], self.smoke[tsl]):  # a parcel is ONE gas: its
                q = f * (gas[lo] - gas[hi])              # oxygen and its soot
                gas[lo] -= q                             # travel together
                gas[hi] += q
        # TWO-ZONE stirring — the CFAST room, made literal: every airspace is a
        # HOT SMOKY zone (the plume and the exhaust bank it feeds under the
        # ceiling) and a COOL CLEAR zone (everything else), each stirred fast
        # internally by its own buoyant circulation, exchanging across the
        # interface only by the diffusion and overturn above. The fire's debt
        # is hot-weighted (see the burn law), so the FOUL zone is the hot one —
        # and as smoke and heat bank downward the foul zone DESCENDS onto a
        # standing head while the floor stays breathable: crawl-low is this
        # structure's shadow, not a rule. It is also why a bed fire keeps
        # breathing: its cool zone is the whole room's fresh pool (a fire that
        # dug a local oxygen hole and died beside fresh air was the measured
        # alternative).
        if air.any() and not flow:
            # the stirring is LOCAL — a plume circulates ITS room, not every
            # airspace its doorway connects to. (Zone-global mixing was tried
            # and it teleported the room's smoke into the whole yard outside,
            # diluting it to 0.00 g — the same crime as the old region mean.)
            # Each voxel relaxes toward the blurred mean of its OWN zone over
            # a room-scale reach, so a doorway is a gradient, not a wormhole.
            # 0.2 g of smoke PER LITRE: it read 0.2 g per voxel, which is
            # 1.6 g/L at 5 cm and 0.2 g/L at 10 cm, so the same smoke made
            # a smoky zone at one voxel size and not the other (P-T4: the
            # 10 cm sitter's air was cut off from the room's as the bank
            # grew, and the 5 cm one's was not).
            hotzone = (T > 45.0) | (self.smoke > 0.2 * self.vox_l)
            reach = max(2, int(round(PLUME_REACH_M / (0.1 * self.scale * self._CS))))
            sizes = np.bincount(self.air_region[air], minlength=self._region_count)
            # AN AIRSPACE WITH NOTHING IN IT TO STIR IS NOT STIRRED. Bullet's
            # island rule, which is the right one here for the same reason it is
            # right there: the thing that may sleep is a whole connected body,
            # never part of one, and anything that disturbs it wakes all of it.
            # An airspace IS an island — gas mixes room-wide, so a room is the
            # smallest thing that can be called still.
            #
            # A room every cell of which holds fresh air, no soot and no warmth
            # has one empty zone and one uniform zone, and relaxing a uniform
            # thing toward its own mean moves nothing. The cost of asking is two
            # comparisons and a count; the cost of not asking was a blur of the
            # WHOLE lattice, three times per zone, for every room in the world.
            # Measured in a building of twenty sealed rooms with a fire in one:
            # nineteen were asleep, holding 95% of all the air.
            live = np.ones(self._region_count, bool)
            if self.skip_quiet:
                still = air & ((self.o2 != O2_PER_L * self.vox_l)
                               | (self.smoke > 0.0) | (T > 45.0))
                live = np.bincount(self.air_region[still],
                                   minlength=self._region_count) > 0
            for r in np.nonzero(sizes > 64)[0]:  # a pocket too small to hold a
                if not live[r]:                  # circulation isn't stirred, and
                    continue                     # neither is a room at rest
                in_r = air & (self.air_region == r)          # circulation isn't stirred
                for zone in (True, False):
                    mz = in_r & (hotzone == zone)            # zone AND region: the blur
                    if not mz.any():                         # cannot see a thin wall,
                        continue                             # so the mask must
                    wt_c = np.maximum(self._blur_cells(
                        self._cells_where(mz), reach), 1e-6)
                    for gas in (self.o2, self.smoke):
                        gm_c = self._blur_cells(
                            self._cells_where(mz, gas), reach)
                        local = self._upcell_where(
                            (gm_c / wt_c).astype(np.float32), mz)
                        d = 0.15 * (local - gas[mz])
                        if gas is self.smoke:
                            d = self._balanced(d)
                        gas[mz] += d
        if self.open_sky:
            # an open world floats in an INFINITE fresh atmosphere: every
            # lattice edge that is air sits at ambient composition. (Top-only
            # refresh let a long fire drag the whole map toward extinction —
            # half-oxygen "open air", which no real backyard ever becomes.
            # Sealed scenes wall their edges in stone and never feel this.)
            fresh = O2_PER_L * self.vox_l
            for edge in ((slice(None), slice(None), -1),
                         (0, slice(None), slice(None)), (-1, slice(None), slice(None)),
                         (slice(None), 0, slice(None)), (slice(None), -1, slice(None))):
                self.o2[edge][air[edge]] = fresh

    @staticmethod
    def _balanced(d):
        """The stir's changes to one zone, made to add up to nothing.

        Relaxing toward a BLURRED mean is not conservative: the blur weighs a
        coarse cell by how much of it is the zone, so what a peaked cell
        gives up is not what its neighbours take. For smoke near a source the
        gap was most of it -- measured, a 0.5 g/s source in a closed room
        lost 3.3 of every 5 g it made in this one line, so a smoulder could
        never fill the next room enough for anyone to smell it (P-T4). The
        side with more is scaled down to the side with less: nothing is
        made, nothing is lost, and no cell is moved further than the stir
        asked."""
        give = d < 0.0
        out = float(-d[give].sum())
        into = float(d[~give].sum())
        if out <= 0.0 or into <= 0.0:
            return np.zeros_like(d)
        if out > into:
            d[give] *= np.float32(into / out)
        else:
            d[~give] *= np.float32(out / into)
        return d

    def _o2_asleep(self, T):
        """THE EXACT GATE: is there nothing at all for the oxygen law to do?

        Yes when every cell holds the fresh value, there is no soot anywhere and
        no air cell is above 45 C. Then (law by law): diffusion and the
        overturn move `k * (a - b)` of two equal numbers, which is 0.0; the
        two-zone stir only runs in a room that holds something not fresh, not
        sooty and not warm (`live`, above), and no room does; the open sky sets
        its edge cells to the fresh value they already have. So the law changes
        no bit, and the whole tick of it can be skipped. Only with `skip_quiet`
        (without it the stir runs on every room and a blur of a flat field may
        differ in the last bit of a float32). The temperature test is a
        scalar `T.max()` first, so a cool world pays one reduce. Measured
        cost is in WORK/w13b/B6/report.txt."""
        if not self.skip_quiet:
            return False
        o2 = self.o2
        fresh = np.float32(O2_PER_L * self.vox_l)    # what `o2 != fresh` compares with
        if o2.min() != fresh or o2.max() != fresh:
            return False
        if self.smoke.any():
            return False
        if T.max() > 45.0 and bool(((T > 45.0) & (self.mat == AIR)).any()):
            return False
        return True

    def _mixing(self):
        """Has the air anything to even out? Oxygen moves down a gradient, and
        a flat field has none — every pair cancels. The open sky is the other
        way in: it refills the top, which matters only if the world is below
        what fresh air holds."""
        if not self.skip_quiet:
            return True
        lo, hi = float(self.o2.min()), float(self.o2.max())
        return lo != hi or (self.open_sky and hi < O2_PER_L * self.vox_l)

    def _warm_cells(self):
        """THE SEAM for the legacy buoyant push: how warm each coarse cell is, K
        (the pressure field `_law_air` lifts by). The default measures against
        the fixed ambient; cold (T8) measures gas against its own rest (3.8c).

        WAVE 13b, C6 (T7 step 2b): ONLY GAS PUSHES. A hot stone, a hot plank or a
        warm body is not a buoyant gas: it warms the air it touches by conduction,
        and that warm air is what rises. Reading the solids' own temperature as
        wind gave a warm body a draught of its own and made every hot wall blow.
        The coarse cell's mean over ALL its voxels is kept (the old average), with
        a solid voxel counting as zero, so a cell that is mostly wall pushes in
        proportion to the gas it holds. A declared golden diff (the C6 report)."""
        gas = self.mat == AIR
        return self._cells(np.where(gas, np.clip(self.T() - AMBIENT, 0.0, 500.0), 0.0).astype(np.float32))

    def _gas_busy(self):
        """Is any GAS warm enough to push (wave 13b, C6)? `_warm_cells` reads gas
        voxels only, so a hot solid or a warm body cannot wake the pneumatic law
        by itself: this asks the unsettled box for a gas voxel above 20 C and
        nothing else. EXACT: with no such voxel the buoyant term is zero and the
        law, with its fields at zero, changes nothing. A world with a climate
        keeps the wider question (`_heat_busy`): there the push is measured
        against each cell's own rest, which is not 20 C."""
        if getattr(self, "climate", None) is not None:
            return self._heat_busy()
        box = self._unsettled(0)[0]
        if box is None:
            return False
        return bool((self.E[box][self.mat[box] == AIR] > 0).any())

    def _pushed(self):
        """Is any gas moving, being made to move, or about to be?

        Pressure and velocity decay toward nothing and stay there, and the
        gradient of a flat field is zero, so a still world carries nothing. But
        pressure is not only injected by `p_add`: WARM AIR MAKES ITS OWN WIND,
        and the law reads `T - AMBIENT` directly, because a plume is buoyancy
        before it is anything else.

        Found by the tick-by-tick guard, which is the whole reason to have one.
        Gated on the pressure fields alone, the first wisp of smoke a fire ever
        made came out wrong by 1.7e-5 on the very tick it appeared — because the
        world had been skipping its buoyant wind since tick zero, and the gate
        looked perfectly reasonable until something measured it."""
        if not self.skip_quiet:
            return True
        if self._gas_busy() or bool(self.p_add.any()):
            return True
        if self.gas_flow:                       # a flow already going carries on
            return self._gf_live() or self._push_live()   # and a push (wind, a moving surface)
        if self.pcell is None:
            return False
        return bool(self.pcell.any()) or any(bool(v.any()) for v in self.vcell)
