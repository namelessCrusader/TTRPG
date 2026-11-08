"""THE GAS STORE (wave 13, T6): named gas grams per voxel, EXCESS over a fixed background.

Water vapour, mist and any gas product of a reaction (CO2, CO, ethanol) live here as
lazy float32 arrays, one per name, in grams per voxel. The number kept is the grams
ABOVE a FIXED reference background B0 (g/m3, per species; zero for a gas fresh air
does not carry, 0.74 g/m3 for CO2, and for H2O the value set when w.air_rh is first
set). Fresh air at B0 is exactly zero and sleeps. Why fixed: a MOVING background
(follow the climate) would make every untouched cell look changed. Outdoor cells
follow the climate by an exact booked shift (shift_outdoor); a sealed room keeps its
grams.

    concentration (g/m3) = B0 + excess / free gas volume (m3)

The free gas volume of a cell is what the liquid and the solid leave: see gas_volume.
Excess may go down to minus the background grams of that cell and no lower.

The store never touches a World. Nothing is allocated until a name is used.
Indices are anything numpy accepts: a boolean mask or a tuple of index arrays.
"""
import numpy as np

from .matterdata import BACKGROUND, R
from . import psychro


def gas_volume(vox_l, fvol_ml, solid_l=0.0):
    """Free gas volume of a cell, m3: the voxel minus its liquid and its solid.
    Floored at 1 percent of the voxel so a full cell never divides by zero."""
    free_l = np.maximum(vox_l - np.asarray(fvol_ml, np.float64) / 1000.0 - solid_l,
                        0.01 * vox_l)
    return free_l / 1000.0


class GasStore:
    def __init__(self, shape, vox_l, dtype=np.float32, background=None):
        self.shape = tuple(shape)
        self.vox_l = float(vox_l)
        self.dtype = np.dtype(dtype)
        self._ex = {}                      # name -> grams excess
        self._bg = dict(BACKGROUND)        # name -> g/m3, FIXED once set
        if background:
            self._bg.update(background)
        self.generation = 0                # bumps when an array is created or changed
                                           # in a way the heat-capacity cache must see

    # ---- the background ---------------------------------------------------
    def set_background(self, name, g_m3):
        """Set B0 once. Changing it later would move every untouched cell's meaning,
        so a second call with a different value is refused."""
        g_m3 = float(g_m3)
        if name in self._ex and name in self._bg and self._bg[name] != g_m3:
            raise ValueError("B0 of %r is fixed once the store holds it" % name)
        self._bg[name] = g_m3

    def bg(self, name):
        """Fixed background of a species, g/m3 (0.0 if it has none)."""
        return self._bg.get(name, 0.0)

    def names(self):
        return sorted(self._ex)

    def has(self, name):
        return name in self._ex

    # ---- the excess arrays ------------------------------------------------
    def excess(self, name):
        """The excess grams array (allocated on first use)."""
        a = self._ex.get(name)
        if a is None:
            a = np.zeros(self.shape, self.dtype)
            self._ex[name] = a
            self.generation += 1
        return a

    def _vol(self, vol, idx):
        """Free gas volume (m3) of the selected cells: None = a whole voxel, a
        scalar, or an array of the store's shape."""
        v = np.asarray(self.vox_l / 1000.0 if vol is None else vol, np.float64)
        return v if v.ndim == 0 else v[idx]

    def add(self, name, idx, g, gas_vol_m3=None):
        """Add g grams (array or scalar, signed). A removal is clamped so the excess
        never goes below minus the background grams of the cell. Returns the signed
        amount actually applied (same shape as the selected cells)."""
        a = self.excess(name)
        g = np.asarray(g, np.float64)
        floor = -self.bg(name) * self._vol(gas_vol_m3, idx)
        if isinstance(idx, tuple) and all(isinstance(i, (np.ndarray, list)) for i in idx):
            # repeated indices must see each other: sum the requests per cell first, so
            # the floor clamp is applied once to the total for that cell
            ix = np.broadcast_arrays(*[np.asarray(i) for i in idx])
            flat = np.ravel_multi_index(tuple(ix), a.shape)
            gb = np.broadcast_to(g, flat.shape)
            fl = np.broadcast_to(floor, flat.shape)
            uniq, inv = np.unique(flat, return_inverse=True)
            inv = inv.reshape(flat.shape)
            tot = np.zeros(uniq.shape)
            np.add.at(tot, inv.ravel(), gb.ravel())
            fl_u = np.zeros(uniq.shape)
            fl_u[inv.ravel()] = fl.ravel()
            cur_u = a.reshape(-1)[uniq].astype(np.float64)
            app_u = np.maximum(tot, fl_u - cur_u)
            a.reshape(-1)[uniq] = (cur_u + app_u).astype(a.dtype)
            # each request gets its pro-rata share of the clamped total
            ratio = np.where(tot != 0.0, app_u / np.where(tot != 0.0, tot, 1.0), 1.0)
            applied = gb * ratio[inv]
        elif isinstance(idx, tuple):
            cur = a[idx].astype(np.float64)
            applied = np.maximum(g, floor - cur)
            np.add.at(a, idx, applied.astype(a.dtype))
        else:
            cur = a[idx].astype(np.float64)
            applied = np.maximum(g, floor - cur)
            a[idx] = (cur + applied).astype(a.dtype)
        self.generation += 1
        return applied

    def take(self, name, idx, g, gas_vol_m3=None):
        """Remove up to g grams (positive). Returns what was taken. Never takes
        more than the cell holds counting its background (the excess floor)."""
        got = -self.add(name, idx, -np.asarray(g, np.float64), gas_vol_m3)
        return got

    # ---- readings ---------------------------------------------------------
    def actual(self, name, gas_vol_m3=None):
        """Concentration, g/m3: B0 + excess / free gas volume."""
        v = self.vox_l / 1000.0 if gas_vol_m3 is None else gas_vol_m3
        return self.bg(name) + self.excess(name).astype(np.float64) / v

    def rh(self, name, T, gas_vol_m3=None):
        """Relative humidity of a vapour: actual concentration over the saturation
        vapour density at T (over the FREE gas volume)."""
        return self.actual(name, gas_vol_m3) / psychro.rho_sat(T)

    def total(self, name):
        """Sum of the excess grams (float64 accumulation)."""
        a = self._ex.get(name)
        return 0.0 if a is None else float(a.sum(dtype=np.float64))

    def total_actual(self, name, gas_vol_m3=None):
        """Grams in the store counting the background of every cell."""
        v = self.vox_l / 1000.0 if gas_vol_m3 is None else gas_vol_m3
        n = int(np.prod(self.shape)) if np.ndim(v) == 0 else v.size
        bgg = self.bg(name) * (float(v) * n if np.ndim(v) == 0 else float(np.sum(v)))
        return self.total(name) + bgg

    def partial_pressure(self, name, M, gas_vol_m3=None, T_ref=293.15):
        """Pa, from the actual concentration at the reference air density (so 0.74
        g/m3 of CO2 reads 41 Pa, 400 ppm)."""
        return self.actual(name, gas_vol_m3) / M * R * T_ref

    # ---- the climate follows the outside ----------------------------------
    def shift_outdoor(self, name, mask, b_old, b_new, gas_vol_m3=None):
        """Outdoor cells hold fresh air whose vapour (or gas) concentration is b_old
        g/m3 and the climate moves it to b_new. The excess of every masked cell shifts
        by (b_new - b_old) V, exactly. Returns the booked grams (the net grams that
        entered the world; negative = left): the caller books gone['vapour']. A
        sealed room is simply not in the mask and keeps its grams. The floor is
        respected: a cell cannot go below minus the fixed background."""
        a = self.excess(name)
        v = self.vox_l / 1000.0 if gas_vol_m3 is None else np.asarray(gas_vol_m3, np.float64)
        d = (float(b_new) - float(b_old)) * (v if np.ndim(v) == 0 else v[mask])
        cur = a[mask].astype(np.float64)
        floor = -self.bg(name) * (v if np.ndim(v) == 0 else v[mask])
        new = np.maximum(cur + d, floor)
        a[mask] = new.astype(a.dtype)
        self.generation += 1
        return float((new - cur).sum())

    def legacy_seep(self, name, por, air, k):
        """Gas leaves a cell for its neighbours when the gas flow is OFF (no rise,
        no buoyancy: a stand-in, said so). Conservative diffusion of the excess
        grams over gas cells `air` (bool), each face open by min(por_i, por_j),
        k the share of the difference moved per step (k <= 1/6 for stability).
        The total is exact to rounding. Returns the grams moved."""
        a = self.excess(name)
        if a.ndim != 3:
            raise ValueError("legacy_seep needs a 3-D store")
        if k > 1.0 / 6.0 + 1e-12:
            raise ValueError("k above 1/6 is unstable")
        e = a.astype(np.float64)
        d = np.zeros_like(e)
        moved = 0.0
        for ax in range(3):
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[ax] = slice(0, -1)
            hi[ax] = slice(1, None)
            lo, hi = tuple(lo), tuple(hi)
            open_ = air[lo] & air[hi]
            f = np.where(open_, k * np.minimum(por[lo], por[hi]) * (e[lo] - e[hi]), 0.0)
            d[lo] -= f
            d[hi] += f
            moved += float(np.abs(f).sum())
        a[...] = (e + d).astype(a.dtype)
        self.generation += 1
        return moved
