"""LayersMixin: the surface layers on the world (wave 13c, lane C4, T3).

A face of the world can carry a thin stack of matter (soot, dew, mud, blood). The
pool (`layers.Layers`) was built and tested alone in phase A. This mixin is the glue.
What is on in this lane:

  * SOOT DEPOSITION. Smoke that touches a solid face settles on it at the rate
    v_d * C (T3 spec 4.2): v_d is the sum of the Brownian/turbulent wall speed, the
    thermophoretic speed toward a cooler wall (hot gas over a cold ceiling) and the
    Stokes settling speed with the face orientation (down on a floor, none on a wall,
    negative and floored at zero on a ceiling). The air gives up `smoke` units; the
    face gets SOOT_TRUE (0.025) grams of true soot per unit (the sim's smoke is the
    "visible products", about 40 times the true soot); the rest of the removed units
    is booked in `gone["smoke"]`, as the open sky already does.
  * THE EMISSIVITY OVERLAY. A sooty face radiates like soot (eps 0.95) in proportion
    to its cover. The overlay is handed to the face heat law through `_eps_over`.
    Series resistance and heat capacity of a soot film are NOT wired (a 0.1 mm film
    is R 0.002 m2K/W and 0.005 J/K a face against 260 J/K of a stone cell; the report
    gives the numbers).

THE GATE. `w.surface_laws` is False on a world that has not asked. With it False,
every method here falls back to `super()` or returns the plain value and the world is
the old world bit for bit. `w.enable_layers()` turns it on for one world. ONE
CONNECTED WORLD: LayersMixin is a plain mixin, and the soot gate is data on the world
(`layer_soot`), so a world can have layers with or without soot, and with the cold,
body heat, mixture, material and wet capabilities in any combination.

CHEAP ON PURPOSE (the user's cheaper-laws preference): the pass runs once every
DEP_EVERY ticks over the smoke box only; the pool takes all faces of a pass in one
array step (`Layers.deposit_soot_batch`); the emissivity overlay is rebuilt only when
the pool changed (its generation counter), by array steps, and read as a handful of
indexed writes.
"""
import numpy as np

from . import layer_laws as LL
from .layer_data import AMBIENT, GRAIN, SOOT, SOOT_TRUE, pack_key
from .layers import Layers
from .materials import AIR, TICK_S

DEP_EVERY = 40           # ticks between soot passes (1 s, the slow tick). k dt stays under 6e-3 at a 5 cm cell, so the Euler step is exact to 3e-3 of what a pass takes (measured against EVERY=8 in the report)
EPS_EVERY = 40           # ticks between rebuilds of the emissivity overlay (a film does not change in under a second)
DEP_MIN = 1.0e-5         # g of smoke in a cell below which it is called clear
D_SOOT = 0.5e-6          # effective soot particle diameter, m (T3 spec 4.2)
RHO_SOOT = 1800.0        # kg/m3 (u)
SHARE_CAP = 0.5          # most of a cell's smoke one pass may take
EPS_SOOT = 0.95          # soot emissivity (T3 spec 4.12)
_DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
# cos(orientation) of the host face a cell-to-host step lands on: the host face that
# looks back at the air is d ^ 1, so a host ABOVE the air cell (d = 4) is a ceiling.
_COS = np.array([0.0, 0.0, 0.0, 0.0, -1.0, 1.0])


class LayersMixin:
    """Put it first in the bases: `class MyWorld(LayersMixin, BodyWorld)`."""

    surface_laws = True        # the gate (wave 13c C4: ON, soot deposition passed P2); False = the old world
    layer_soot = True          # soot deposition (needs surface_laws)

    # ---------------------------------------------------------------- the switch
    def enable_layers(self, soot=True, every=DEP_EVERY):
        """Turn the surface laws on for this world. Idempotent."""
        self.surface_laws = True
        self.layer_soot = bool(soot)
        self.layer_every = int(every)
        self.ensure_layers()
        return self.layers

    def _layers_on(self):
        return self.surface_laws and self.layers is not None

    # ---------------------------------------------------------------- the books
    def layer_mass(self):
        """Book T3: grams in and out of the pool, per (species, tag)."""
        L = self.layers
        if L is None:
            return {}
        return {"led_in": {f"{k[0]}:{k[1]}": v for k, v in L.led_in.items()},
                "led_out": {f"{k[0]}:{k[1]}": v for k, v in L.led_out.items()}}

    def layer_enthalpy(self):
        L = self.layers
        return 0.0 if L is None else float(L.stored_J())

    def soot_on_faces(self):
        """Grams of true soot on all faces."""
        L = self.layers
        return 0.0 if L is None else float(L.total(SOOT))

    # ---------------------------------------------------------------- the pass
    def _law_smoke(self, sl=None):
        super()._law_smoke(sl)
        if not (self.surface_laws and self.layer_soot):
            return
        if self.tick % getattr(self, "layer_every", DEP_EVERY):
            return
        self._soot_pass(sl)

    def _soot_pass(self, sl=None):
        """Smoke settles on the solid faces beside it (T3 spec 4.2). Returns the
        smoke units taken out of the air."""
        L = self.ensure_layers()
        sl = sl if sl is not None else (slice(None),) * 3
        nx, ny, nz = self.shape
        sm = self.smoke[sl]
        there = (sm > DEP_MIN) & (self.mat[sl] == AIR)
        cx, cy, cz = np.nonzero(there)
        if not len(cx):
            return 0.0
        ox, oy, oz = (s.start or 0 for s in sl)
        cx, cy, cz = cx + ox, cy + oy, cz + oz
        smoke_c = self.smoke[cx, cy, cz].astype(np.float64)
        dt = getattr(self, "layer_every", DEP_EVERY) * TICK_S
        dx = 0.1 * self.scale
        conc = smoke_c * (1000.0 / self.vox_l)               # smoke units per m3
        E, C = self.E, self.heat_capacity()           # T = AMBIENT + E/C, read where needed
        def temp(x, y, z):
            # clipped to where the film-temperature viscosity law is defined (a cell left
            # with a stray negative E would otherwise give a NaN speed and a NaN smoke field)
            return np.clip(AMBIENT + E[x, y, z].astype(np.float64) / C[x, y, z], -150.0, 5000.0)
        Tg = temp(cx, cy, cz)
        v_s = LL.stokes_velocity(D_SOOT, RHO_SOOT)
        hosts, faces, owner, vds = [], [], [], []
        ci = np.arange(len(cx))
        for d, (ddx, ddy, ddz) in enumerate(_DIRS):
            hx, hy, hz = cx + ddx, cy + ddy, cz + ddz
            ok = (hx >= 0) & (hx < nx) & (hy >= 0) & (hy < ny) & (hz >= 0) & (hz < nz)
            hx, hy, hz, c_ = hx[ok], hy[ok], hz[ok], ci[ok]
            solid = (self.mat[hx, hy, hz] != AIR) & ~self.moving[hx, hy, hz]
            hx, hy, hz, c_ = hx[solid], hy[solid], hz[solid], c_[solid]
            if not len(hx):
                continue
            Tw = temp(hx, hy, hz)
            v_th = LL.thermophoretic_velocity(Tg[c_], Tw)
            vd = np.maximum(0.0, LL.V_BROWN + v_th + v_s * _COS[d])
            hosts.append((hx * ny + hy) * nz + hz)
            faces.append(np.full(len(hx), d ^ 1, np.int64))
            owner.append(c_)
            vds.append(vd)
        if not hosts:
            return 0.0
        hosts = np.concatenate(hosts)
        faces = np.concatenate(faces)
        owner = np.concatenate(owner)
        vd = np.concatenate(vds)
        take = vd * conc[owner] * (dx * dx) * dt              # smoke units per face
        want = np.bincount(owner, weights=take, minlength=len(cx))
        room = SHARE_CAP * smoke_c
        scale = np.where(want > room, room / np.maximum(want, 1e-300), 1.0)
        take = take * scale[owner]
        gone = np.bincount(owner, weights=take, minlength=len(cx))
        self.smoke[cx, cy, cz] = (smoke_c - gone).astype(self.smoke.dtype)
        total = float(gone.sum())
        self.gone["smoke"] = self.gone.get("smoke", 0.0) + total * (1.0 - SOOT_TRUE)
        keys = (hosts << 3) | faces                          # tag 0: a cell face
        o = np.argsort(keys, kind="stable")
        L.now = self.tick * TICK_S
        L.deposit_soot_batch(keys[o], take[o] * SOOT_TRUE)
        return total

    # ---------------------------------------------------------------- the heat read-out
    def _eps_over(self, eps, sl):
        """The emissivity the face law reads, with the layers on it. A cell's value is
        its substrate's, moved toward the layer's by the layer's cover, averaged over
        the cell's open faces (the approximation the spec states in 4.13)."""
        L = self.layers
        if not self.surface_laws or L is None or L.n == 0:
            return eps
        c = getattr(L, "eps_cache", None)
        if c is None or (c[0] != L.gen and self.tick - c[1] >= EPS_EVERY):
            hosts, vals = self._eps_build(L)
            _nx, ny, nz = self.shape
            c = (L.gen, self.tick, hosts // (ny * nz), (hosts // nz) % ny, hosts % nz, vals)
            L.eps_cache = c
        _gen, _t, hx, hy, hz, vals = c
        if not len(vals):
            return eps
        ox, oy, oz = (s.start or 0 for s in sl)
        ex, ey, ez = eps.shape
        if ox == 0 and oy == 0 and oz == 0 and (ex, ey, ez) == self.shape:
            inn = None                                  # the whole world: every host is in
            lx, ly, lz = hx, hy, hz
        else:
            lx, ly, lz = hx - ox, hy - oy, hz - oz
            inn = (lx >= 0) & (lx < ex) & (ly >= 0) & (ly < ey) & (lz >= 0) & (lz < ez)
            if not inn.any():
                return eps
        out = eps.copy()
        if inn is None:
            out[lx, ly, lz] = vals.astype(out.dtype)
        else:
            out[lx[inn], ly[inn], lz[inn]] = vals[inn].astype(out.dtype)
        return out

    def _eps_build(self, L):
        """(hosts, eps) of every cell with a layer on a cell face: array steps for the
        faces whose whole stack is one pure soot layer, the pool's own `thermal` for any
        other face (rare in this lane)."""
        keys = L.keys()
        keys = keys[(keys >> 58) == 0]
        if not len(keys):
            return np.empty(0, np.int64), np.empty(0, np.float32)
        nx, ny, nz = self.shape
        host = keys >> 3
        face = keys & 7
        hx, hy, hz = host // (ny * nz), (host // nz) % ny, host % nz
        sub = self._emiss_table()[self.mat[hx, hy, hz]].astype(np.float64)
        top = L.lookup(keys)
        ck = pack_key(SOOT, GRAIN, 1)                         # what deposit_soot_batch writes
        pure = ((L.ckey[top, 0] == ck) & (L.cm[top, 1:].sum(axis=1) == 0.0)
                & (L.below[top] < 0) & (L.inside[top] == 0))
        cover = L.cover[top].astype(np.float64)
        eps_f = np.where(pure, EPS_SOOT * cover + (1.0 - cover) * sub, sub)
        for i in np.nonzero(~pure)[0]:
            eps_f[i] = L.thermal(int(keys[i]))[1]
        # open faces of each host: neighbours that are not solid
        open_n = np.zeros(len(keys), np.float64)
        for d, (ddx, ddy, ddz) in enumerate(_DIRS):
            x, y, z = hx + ddx, hy + ddy, hz + ddz
            ok = (x >= 0) & (x < nx) & (y >= 0) & (y < ny) & (z >= 0) & (z < nz)
            gas = np.zeros(len(keys), bool)
            gas[ok] = self.mat[x[ok], y[ok], z[ok]] == AIR
            open_n += gas
        open_n = np.maximum(open_n, 1.0)
        uh, inv = np.unique(host, return_inverse=True)
        delta = np.bincount(inv, weights=(eps_f - sub), minlength=len(uh))
        n_open = np.bincount(inv, weights=open_n, minlength=len(uh)) / np.maximum(
            np.bincount(inv, minlength=len(uh)), 1)
        sub_h = np.bincount(inv, weights=sub, minlength=len(uh)) / np.maximum(
            np.bincount(inv, minlength=len(uh)), 1)
        return uh.astype(np.int64), (sub_h + delta / n_open).astype(np.float32)
