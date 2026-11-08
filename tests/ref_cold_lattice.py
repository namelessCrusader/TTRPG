"""A small REFERENCE LATTICE for the cold topic (T8): pure numpy, no World.

It exists to test COMPOSITION: that the kernels of src/voxel/outside.py, put
together as the World will put them, keep the facts the spec lists (a quiet
reference climate stays bit-still, a -40 C stone is not clamped, a lumped cube
cools like a lumped cube, the ledger closes, a roofed room is not the outside).
It is not the sim. One node per cell, explicit Euler, 5 cm cells by default,
and a time step that may be a few ticks long (the outdoor relaxation is an
exact exponential, so a longer step does not make it unstable).

Per step, in this order (the order the spec gives the World laws):
    1. publish T_out / T_sky (every `publish_every_s`): one exact outdoor_shift
    2. sources (W per cell), booked as injected energy
    3. the quiet gate: unsettled_box; if nothing is off rest and nothing is
       driven, STOP (the cell values are bit-still)
    4. conduction and the film at solid|gas faces (clipped by FACE_CLIP)
    5. radiation: outdoor faces to their sky/air environment, indoor faces
       through the room's MRT star, edge faces' film, ground flux
    6. outdoor_relax (two rates)
Every exchange with the outside is booked into `shed` (heat that LEFT, signed)
and `booked[cause]`, from the actual change of the float32 lattice, so
total_energy() = sum(E) + shed - injected is a conserved quantity.

OVERLAY TABLES (for the T7 lane's body hooks): `overlay` may hold per-cell
arrays 'h_conv' (zone film conductance, W/m2K, default H_CONV), 'eps'
(emissivity, default the material's), 'r_series' (series resistance in
m2K/W added to the film, default 0), each lattice-shaped or None.
"""
from collections import deque

import numpy as np

from src.voxel import climate as K
from src.voxel import outside as O

F32 = np.float32
AIR_CVOL = 1200.0              # J/(m3 K): materials.AIR_C = 1.2 J/(L K)
AIR_K = 0.026                  # W/(m K)

# id -> (volumetric heat capacity J/(m3 K), conductivity W/(m K), emissivity)
MATS = {
    1: (2.1e6, 2.0, 0.90),     # stone
    2: (1.0e6, 0.12, 0.90),    # wood  (effusivity sqrt(k rho c) ~ 350-400)
    3: (3.5e6, 50.0, 0.30),    # iron
    4: (5.0e4, 0.03, 0.90),    # foam (an insulating pad)
    5: (2.0e6, 0.90, 0.90),    # glass
}
STONE, WOOD, IRON, FOAM, GLASS = 1, 2, 3, 4, 5


class RefLattice:
    def __init__(self, shape, climate=None, dx=0.05, dt=0.25, t_rest=None, open_sky=True,
                 publish_every_s=1.0, overlay=None, clock=None, fused=None):
        self.shape = tuple(shape)
        self.dx, self.dt = float(dx), float(dt)
        self.A = self.dx * self.dx
        self.open_sky = bool(open_sky)
        self.climate = climate if climate is not None else K.Climate.reference()
        self.clock = clock if clock is not None else K.clock_of(self.climate)
        self.t_rest = K.rest_temp(self.climate) if t_rest is None else float(t_rest)
        self.rest_k = self.t_rest - O.AMBIENT
        self.publish_every_s = float(publish_every_s)
        self.overlay = dict(overlay or {})
        self.fused = fused
        self.mat = np.zeros(self.shape, np.uint8)
        self.src = np.zeros(self.shape, np.float32)          # W per cell
        self.E = np.zeros(self.shape, np.float32)
        self.shed = 0.0
        self.injected = 0.0
        self.booked = {k: 0.0 for k in ("rad", "edge", "air_film", "air_far", "shift", "top",
                                        "ground", "ref_shift", "mrt_resid")}
        self.t = 0.0
        self.steps = 0
        self.t_out = None
        self.t_sky = None
        self.last_box = None
        self.last_why = []
        self.awake_steps = 0
        self._last_publish = -1e9
        self.built = False

    # ---------------------------------------------------------- building
    def fill(self, x0, x1, y0, y1, z0, z1, mat):
        self.mat[x0:x1, y0:y1, z0:z1] = mat
        self.built = False

    def source(self, cell, watts):
        self.src[cell] = watts

    def finish(self):
        """Build the static caches: C, class, sky view, face environment."""
        shp = self.shape
        self.solid = self.mat > 0
        self.gas = ~self.solid
        cv = np.full(shp, AIR_CVOL)
        kk = np.full(shp, AIR_K)
        ee = np.full(shp, 0.0)
        for mid, (c, k, e) in MATS.items():
            m = self.mat == mid
            cv[m], kk[m], ee[m] = c, k, e
        if self.overlay.get("eps") is not None:
            ee = np.where(self.solid, self.overlay["eps"], ee)
        self.C = (cv * self.dx ** 3).astype(np.float32)
        self.k = kk.astype(np.float32)
        self.eps = ee.astype(np.float32)
        hc = self.overlay.get("h_conv")
        rs = self.overlay.get("r_series")
        h = np.full(shp, O.H_CONV) if hc is None else np.asarray(hc, float)
        if rs is not None:
            h = 1.0 / (1.0 / h + np.asarray(rs, float))
        self.h_eff = h.astype(np.float32)
        self.Hs = np.full(shp[:2], -1, np.int64)
        for z in range(shp[2]):
            self.Hs = np.where(self.solid[:, :, z], z, self.Hs)
        nb = np.zeros(shp, bool)
        for ax in range(3):
            for sgn in (1, -1):
                nb |= O._shift_nb_free(self.solid, ax, sgn, False)
        self.film = self.gas & nb
        R = O.view_radius_cells(self.Hs, self.dx)
        self.cls = O.outdoor_class(self.Hs, self.gas, self.film, R, open_sky=self.open_sky,
                                   fused=self.fused)
        out = (self.cls & 1) != 0
        cells = np.argwhere(out & self.film)
        self.psi = np.zeros(shp, np.float32)
        if len(cells):
            self.psi[tuple(cells.T)] = O.sky_view(self.Hs, cells, R, fused=self.fused).astype(np.float32)
        self.room = self._label_rooms(self.gas & ~out)
        ground_k = int(round(self.climate.ground_top_m / self.dx))
        self.fe = O.face_env(self.solid, self.gas, self.cls, self.psi, self.room,
                             open_sky=self.open_sky, ground_k=ground_k, fused=self.fused)
        self.af = np.ones(shp, np.float32)
        self.exposed = ((self.solid & (self.fe.n_out > 0)) | (out & self.film))
        self.built = True
        self._set_env()

    def _label_rooms(self, mask):
        lab = np.zeros(mask.shape, np.int32)
        nxt = 0
        for start in map(tuple, np.argwhere(mask)):
            if lab[start]:
                continue
            nxt += 1
            q = deque([start])
            lab[start] = nxt
            while q:
                x, y, z = q.popleft()
                for dx, dy, dz in O._DIRS6:
                    a, b, c = x + dx, y + dy, z + dz
                    if 0 <= a < mask.shape[0] and 0 <= b < mask.shape[1] and 0 <= c < mask.shape[2] \
                            and mask[a, b, c] and not lab[a, b, c]:
                        lab[a, b, c] = nxt
                        q.append((a, b, c))
        return lab

    def soak(self):
        """Every cell at its rest (outdoor air at T_out). Call after finish()."""
        if not self.built:
            self.finish()
        self._publish(force=True, shift=False)
        self.E = O.soak_fill(self.C, self.cls, self.rest_k, self.th)

    # ---------------------------------------------------------- climate
    def _x(self):
        return float(self.clock.day_coord(self.t / K.TICK_S))

    def _publish(self, force=False, shift=True):
        if not force and self.t - self._last_publish < self.publish_every_s - 1e-9:
            return
        self._last_publish = self.t
        x = self._x()
        tq = float(K.quantize(K.day_curve(self.climate, x), self.climate.step_k))
        ts = float(K.t_sky_x(self.climate, x))
        self.t_sky = ts
        new_th = tq - O.AMBIENT
        if self.t_out is None or tq != self.t_out:
            if shift and self.t_out is not None and self.built:
                gain = O.outdoor_shift(self.E, self.C, self.cls, self.t_out - O.AMBIENT, new_th,
                                       fused=self.fused)
                self.shed -= gain
                self.booked["shift"] -= gain
            self.t_out = tq
        self.th = self.t_out - O.AMBIENT
        self._set_env()

    def _set_env(self):
        if not self.built:
            return
        t_sky = self.t_sky if self.t_sky is not None else self.climate.t_ann_c
        t_out = self.t_out if self.t_out is not None else self.climate.t_ann_c
        self.tenv = O.env_temperature(self.fe.n_out, self.fe.w_sky, t_sky, t_out)

    # ---------------------------------------------------------- the step
    def T(self):
        return (O.AMBIENT + self.E / self.C).astype(np.float32)

    def total_energy(self):
        return float(self.E.sum(dtype=np.float64)) + self.shed - self.injected

    def _book(self, cause, before):
        gone = before - float(self.E.sum(dtype=np.float64))
        self.shed += gone
        self.booked[cause] += gone

    def step(self):
        if not self.built:
            self.finish()
        if self.t_out is None:
            self._publish(force=True, shift=False)
        else:
            self._publish()
        dt = self.dt
        if np.any(self.src):
            add = (self.src * F32(dt)).astype(np.float32)
            self.E += add
            self.injected += float(add.sum(dtype=np.float64))
        t_sky_eff = self.t_sky if self.t_sky is not None else self.t_out
        driven = O.driven_reasons(self.t_rest, self.t_out, t_sky_eff)
        box, why = O.unsettled_box(self.E, self.C, self.cls, self.rest_k, self.th, O.SETTLE_K,
                                   exposed=self.exposed, driven=driven, fused=self.fused)
        self.last_box, self.last_why = box, why
        self.t += dt
        self.steps += 1
        if box is None:
            return
        self.awake_steps += 1
        self._conduct()
        self._radiate()
        ff = O.relax_factor(self.climate.k_film, dt)
        fa = O.relax_factor(self.climate.k_far, dt)
        a, b = O.outdoor_relax(self.E, self.C, self.cls, self.th, ff, fa, O.SETTLE_K, fused=self.fused)
        self.shed += a + b
        self.booked["air_film"] += a
        self.booked["air_far"] += b

    def _conduct(self):
        E, C = self.E, self.C
        T = O.AMBIENT + E / C
        A, dt = F32(self.A), F32(self.dt)
        dxh = F32(self.dx / 2.0)
        for ax in range(3):
            a = [slice(None)] * 3
            b = [slice(None)] * 3
            a[ax], b[ax] = slice(0, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            sa, sb = self.solid[a], self.solid[b]
            ka, kb = self.k[a], self.k[b]
            g_ss = A / (dxh / ka + dxh / kb)
            ks = np.where(sa, ka, kb)
            hs = np.where(sa, self.h_eff[a], self.h_eff[b])
            g_sg = A / (dxh / ks + F32(1.0) / hs)
            g_gg = A * F32(AIR_K) / F32(self.dx)
            G = np.where(sa & sb, g_ss, np.where(sa | sb, g_sg, g_gg)).astype(np.float32)
            dT = T[a] - T[b]
            q = G * dt * dT
            lim = F32(O.FACE_CLIP) * np.minimum(C[a], C[b]) * np.abs(dT)
            q = np.minimum(np.maximum(q, -lim), lim).astype(np.float32)
            E[a] -= q
            E[b] += q

    def _radiate(self):
        E, C = self.E, self.C
        Ts = O.AMBIENT + E / C
        solid = self.solid
        eps = self.eps
        fe = self.fe
        # outdoor faces: to their sky and air environment
        before = float(E.sum(dtype=np.float64))
        loss = O.radiation_out(Ts, eps, self.af, np.where(solid, fe.n_out, 0), self.tenv,
                               self.A, self.dt, 0.0, C, fused=self.fused)
        E -= loss
        self._book("rad", before)
        # lattice-edge faces' film
        before = float(E.sum(dtype=np.float64))
        Ts = O.AMBIENT + E / C
        q = O.edge_film(Ts, np.where(solid, fe.n_edge, 0), self.t_out, self.af, self.A, self.dt, C)
        E -= q
        self._book("edge", before)
        # indoor faces: through each room's radiant node (zero-sum before the clip)
        if (fe.n_in > 0).any():
            Ts = O.AMBIENT + E / C
            m = O.mrt_star(Ts, eps, self.af, np.where(solid, fe.n_in, 0), fe.room, self.A, self.dt, C,
                           fused=self.fused)
            before = float(E.sum(dtype=np.float64))
            E -= m.q
            self._book("mrt_resid", before)
        # the ground under the floor layer
        gd = self.climate.ground_depth_m
        if gd is not None:
            before = float(E.sum(dtype=np.float64))
            Ts = (O.AMBIENT + E / C)[:, :, 0]
            tg = float(K.t_ground(self.climate, gd, self._x()))
            gq = O.ground_flux(Ts, tg, self.k[:, :, 0], self.A, self.dt, gd + self.dx / 2.0, C[:, :, 0])
            E[:, :, 0] -= np.where(solid[:, :, 0], gq, F32(0))
            self._book("ground", before)

    def run(self, n):
        for _ in range(n):
            self.step()

    # ---------------------------------------------------------- readouts
    def booked_total(self):
        return float(sum(self.booked.values()))
