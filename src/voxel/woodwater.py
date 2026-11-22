"""WOOD WATER IN TWO POOLS (wave 14a refinement, lane R3).

One wetness number per cell cannot fit a drying curve. A wood cell that is named by
FluidsMixin.set_wood(..., pools=True) holds its water in three places, all booked:
  vapour  in the gas beside it (the wet store's H2O),
  free    liquid in the cell cavities (the cell's own fvol, as before),
  bound   water in the cell walls (this module's array `wood_bound`, grams).
Free water leaves in days; bound water desorbs and reabsorbs over weeks (Multi-phase model for
moisture transport in wood, doi:10.1007/s00226-025-01635-9: separate phases with conversion terms
fit the measured profiles where one equation does not; doi:10.1007/bf02483291: the transfer
between vapour and bound water is a finite exchange).

THE LAWS (every `wood_every` ticks, one exponential step, exact for a linear relaxation):
  1 free -> bound   a cell with liquid water in it and bound water below the fibre saturation
                    point relaxes toward saturation:  dB = (Bsat - B)(1 - exp(-k dt)), at most F.
  2 bound <-> vapour  a cell with no free water, beside gas, relaxes toward the equilibrium moisture
                    of that gas (the isotherm, psychro.gab_w, capped at saturation). The gas cell
                    never goes past the equilibrium vapour density (it is a small pot).
  3 hot wood        above BOIL the target is zero and the rate is fast (HOT_TAU_S): wood dries in a fire.
  4 a cell that is no longer wood (char, ash, burnt away) gives its bound water to the gas.
k = pi^2 D / dx^2 (the first mode of a slab of one cell), D the bound-water diffusion coefficient,
faster along the grain than across it when the cell has an open face on its grain axis, with an
Arrhenius factor on temperature. Each conversion adds to `wood_book` (grams) and moves the heat of
sorption: the cell holds -W(u) of enthalpy per gram of dry wood (W the integral heat of wetting),
so `bound_enthalpy_J()` + total_energy() is conserved by these laws.
"""
import numpy as np

from .materials import AIR, AMBIENT, BOIL, NOFLUID, TICK_S, WATER, WOOD

EVERY = 40                      # ticks between applications (1 s)
FSP = 0.30                      # g/g, fibre saturation point at room temperature. Wood Handbook ch.4
                                # (Glass and Zelinka): "about 30 percent"; species range 0.21-0.32 (u)
D_ACROSS = 2.0e-10              # m2/s, bound-water diffusion across the grain near 12 percent MC and
                                # 25 C. Siau, Transport Processes in Wood, order 1e-10 to 1e-9 (u)
GRAIN_RATIO = 3.0               # along over across for the bound-water exchange (u). The 500x of the
                                # liquid permeability (doi:10.1007/s00226-025-01635-9) is the soak
                                # law's business (wooddata S_along/S_across), not this slow exchange
EA = 38000.0                    # J/mol, activation energy of bound diffusion, Skaar 1988 (u)
HS0 = 1000.0                    # J/g, differential heat of sorption at zero moisture, falling to 0 at
                                # the fibre saturation point as (1 - u/FSP)^2; Wood Handbook ch.4 (u)
HOT_TAU_S = 10.0                # s, bound water above BOIL leaves with this time constant (u)
T_PSY = 150.0                   # C, the steam fits (psychro) end near here; hotter wood reads as 150
D_W = 1.0                       # g/ml, water
_R = 8.314


def wetting_heat(u, fsp=FSP):
    """W(u), J per gram of dry wood: the integral heat of sorption from dry to moisture u."""
    x = np.clip(np.asarray(u, np.float64) / fsp, 0.0, 1.0)
    return HS0 * fsp / 3.0 * (1.0 - (1.0 - x) ** 3)


def rate(T, along, dx):
    """k, 1/s: the bound-water relaxation rate of a cell at T (C), with an open grain-axis face
    (`along`) or without, for a voxel of side dx metres."""
    D = D_ACROSS * np.where(along, GRAIN_RATIO, 1.0)
    D = D * np.exp(EA / _R * (1.0 / 298.15 - 1.0 / (np.asarray(T, np.float64) + 273.15)))
    return np.pi ** 2 * D / (dx * dx)


def relax(B, target, k, dt):
    """New B after dt of relaxation toward target at rate k (exact)."""
    return B + (target - B) * (-np.expm1(-k * dt))


class WoodWaterMixin:
    wood_bound = None            # g of bound water per cell, float32; lazy
    wood_every = EVERY
    wood_book = None             # cumulative grams per conversion; lazy dict

    def _wood_pools_on(self):
        return self.wood_bound is not None

    def _tick_end(self):
        super()._tick_end()
        if self.wood_bound is not None and self.tick % self.wood_every == 0:
            self._law_wood_pools(self.wood_every * TICK_S)

    # ---------------------------------------------------------------- readings
    def bound_water_g(self):
        """Grams of bound water in the world (the third ledger, beside total_fluid and the vapour)."""
        return 0.0 if self.wood_bound is None else float(self.wood_bound.sum(dtype=np.float64))

    def bound_enthalpy_J(self):
        """Enthalpy held by the bound water, J (negative: wetting gave the heat out)."""
        if self.wood_bound is None:
            return 0.0
        m = self.smass.astype(np.float64)
        u = np.where(m > 0, self.wood_bound / np.maximum(m, 1e-12), 0.0)
        return float(-(m * wetting_heat(u)).sum())

    def moisture(self):
        """Bound moisture u, g per g of dry wood, per cell (0 where none)."""
        m = self.smass.astype(np.float64)
        return np.where(m > 0, self.wood_bound / np.maximum(m, 1e-12), 0.0)

    def _book(self, key, g):
        if self.wood_book is None:
            self.wood_book = {}
        self.wood_book[key] = self.wood_book.get(key, 0.0) + float(g)

    # ---------------------------------------------------------------- the law
    def _gas_neighbour(self, x, y, z):
        """For each cell, (found, tx, ty, tz, open_faces[6]): the first neighbour that is gas, and
        which of the six faces are open (not wood: gas, liquid or another solid)."""
        nx, ny, nz = self.shape
        n = len(x)
        found = np.zeros(n, bool)
        t = [np.zeros(n, np.int64) for _ in range(3)]
        opn = np.zeros((6, n), bool)
        head = self.cap * 0.9
        for d, (ex, ey, ez) in enumerate(((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                                          (0, 0, 1), (0, 0, -1))):
            qx, qy, qz = x + ex, y + ey, z + ez
            ins = (qx >= 0) & (qx < nx) & (qy >= 0) & (qy < ny) & (qz >= 0) & (qz < nz)
            qx, qy, qz = np.clip(qx, 0, nx - 1), np.clip(qy, 0, ny - 1), np.clip(qz, 0, nz - 1)
            opn[d] = ins & (self.mat[qx, qy, qz] != WOOD)
            g = ins & (self.mat[qx, qy, qz] == AIR) & (self.fvol[qx, qy, qz] < head) & ~found
            for a, q in zip(t, (qx, qy, qz)):
                a[g] = q[g]
            found |= g
        return found, t, opn

    def _law_wood_pools(self, dt):
        from . import psychro
        from .gasstore import gas_volume
        B3 = self.wood_bound
        named = (self.wood_sp > 0) & (self.mat == WOOD)
        idx = np.nonzero(named)
        st = self.wet.store if getattr(self, "wet", None) is not None else None
        dx = 0.1 * self.scale
        C = self.heat_capacity()
        if len(idx[0]):
            x, y, z = idx
            m = self.smass[idx].astype(np.float64)
            B = B3[idx].astype(np.float64)
            T = AMBIENT + self.E[idx].astype(np.float64) / C[idx]
            F = np.where(self.fl[idx] == WATER, self.fvol[idx].astype(np.float64) * D_W, 0.0)
            found, t, opn = self._gas_neighbour(x, y, z)
            ax = self.wood_grain[idx].astype(np.int64) - 1
            along = np.zeros(len(x), bool)
            for a in range(3):
                along |= (ax == a) & (opn[2 * a] | opn[2 * a + 1])
            k = rate(T, along, dx)
            hot = T >= BOIL
            Bsat = FSP * m
            # 1 free -> bound
            dB = np.where((F > 0) & ~hot, np.minimum((Bsat - B) * (-np.expm1(-k * dt)), F), 0.0)
            dB = np.maximum(dB, 0.0)
            if dB.any():
                u1, u2 = B / np.maximum(m, 1e-12), (B + dB) / np.maximum(m, 1e-12)
                self.fvol[idx] = np.maximum(self.fvol[idx] - (dB / D_W).astype(self.fvol.dtype), 0.0)
                gone = self.fvol[idx] <= 0
                if gone.any():
                    g_ = tuple(a[gone] for a in idx)
                    self.fl[g_], self.fpot[g_] = NOFLUID, 0.0
                self.E[idx] += (m * (wetting_heat(u2) - wetting_heat(u1))).astype(self.E.dtype)
                B = B + dB
                self._book("free_to_bound", dB.sum())
            # 2 and 3: bound <-> vapour (only where there is gas beside and no free water)
            if st is not None and found.any():
                F2 = np.where(self.fl[idx] == WATER, self.fvol[idx].astype(np.float64) * D_W, 0.0)
                use = found & (F2 <= 0)
                if use.any():
                    i = np.nonzero(use)[0]
                    tg = tuple(a[i] for a in t)
                    Vg = gas_volume(self.vox_l, self.fvol[tg].astype(np.float64))
                    Tg = np.minimum(AMBIENT + self.E[tg].astype(np.float64) / C[tg], T_PSY)
                    rho_air = st.bg("H2O") + st.excess("H2O")[tg].astype(np.float64) / Vg
                    Ts = np.minimum(T[i], T_PSY)
                    rs = psychro.rho_sat(Ts)
                    h = np.clip(rho_air / rs, 0.0, 1.0)
                    u = B[i] / np.maximum(m[i], 1e-12)
                    ut = np.where(hot[i], 0.0, np.minimum(psychro.gab_w(h), FSP))
                    ki = np.where(hot[i], 1.0 / HOT_TAU_S, k[i])
                    g = (u - ut) * m[i] * (-np.expm1(-ki * dt))         # + leaves the wood
                    rho_eq = rs * np.minimum(psychro.gab_aw(np.clip(u, 1e-9, FSP)), 1.0)
                    # the gas pot is refreshed by the vapour spread once per wet cycle, so a step
                    # longer than that may move that many pots-full
                    room = np.abs(rho_eq - rho_air) * Vg * max(1.0, dt / float(self.wet.dt))
                    g = np.clip(g, -room, room)
                    g = np.where(g > 0, np.minimum(g, B[i]), g)
                    applied = st.add("H2O", tg, g)
                    g = np.asarray(applied, np.float64)
                    u2 = np.maximum(B[i] - g, 0.0) / np.maximum(m[i], 1e-12)
                    Tv = np.where(g >= 0, Ts, Tg)
                    cell, gas, _s = psychro.evap_energy(Tv)
                    sx = tuple(a[i] for a in idx)
                    dH = m[i] * (wetting_heat(u) - wetting_heat(u2))       # heat taken from the cell
                    np.add.at(self.E, sx, (-g * cell - dH).astype(self.E.dtype))
                    np.add.at(self.E, tg, (g * gas).astype(self.E.dtype))
                    B[i] = B[i] - g
                    self._book("bound_to_vapour", g[g > 0].sum())
                    self._book("vapour_to_bound", -g[g < 0].sum())
            B3[idx] = B.astype(B3.dtype)
        # 4 a cell that is no longer wood gives its bound water away
        gone = (B3 > 0) & (self.mat != WOOD)
        if gone.any() and st is not None and hasattr(self, "_deposit"):
            src = np.nonzero(gone)
            g = B3[src].astype(np.float64)
            m = np.maximum(self.smass[src].astype(np.float64), 1e-12)
            Ts = np.minimum(AMBIENT + self.E[src].astype(np.float64) / C[src], T_PSY)
            cell, gas, _s = psychro.evap_energy(Ts)
            np.add.at(self.E, src, (-g * cell - m * wetting_heat(g / m)).astype(self.E.dtype))
            B3[src] = 0.0
            self._deposit(src, g, g * gas)
            self._book("released", g.sum())
