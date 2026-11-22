"""ONE PUSH LAW: any moving surface pushes the gas it sweeps (wave 14a, lane D).

A bellows board, a fan blade, a hand, a body, a blowing mouth and the wind are
one thing: a surface that moves through gas and so moves the gas. There is no
bellows code and no fan code here, and there never will be: each is a scene that
puts SURFACE ELEMENTS in the world and moves them.

A SURFACE ELEMENT is a little flat patch of a moving thing: where it is, its area,
its unit normal, and how fast it moves (all SI). A board is a grid of patches,
a blade is a row of them tilted by its pitch, a spinning blade has a velocity that
grows with its radius. The law reads only those four things plus a drag
coefficient (a flat plate across the stream: 1.2, Hoerner 1965, Fluid-Dynamic Drag).

What a patch does to the gas, both from the one fact that the gas touching it
cannot go through it:

  1. DISPLACEMENT. A patch moving along its normal at v_n sweeps a_n = a v_n
     cubic metres a second out of the way: the gas on the side it moves toward
     is squeezed (a source of volume, +a v_n), the gas on the other side is let
     in (a sink, -a v_n). These go to the gas flow's pressure solve as one more
     term beside the expansion of hot gas. In a closed chamber with one way out
     that is a jet out of the way out; across a thin plate in open air it is the
     flow round the edge. A sealed airspace has no way to take it (it keeps no
     pressure, see gasflow.py): the mean is taken off, as it is for a fire.
  2. DRAG. Moving across the gas it also drags it along: the force on the gas is
     F = 1/2 rho Cd a |w| w, in the direction of the normal, where w is the
     speed of the patch relative to the gas, along the normal. This is the
     only thing that gives a steady thrust (a pitched blade pushes air along its
     normal though its own motion is sideways; d'Alembert: displacement alone
     gives none). The impulse goes into the speeds of the faces of the cells it
     is in, never more than closing the gap between the patch and the gas.

The law reports what the gas did to each patch (the reaction, `force`, in N), so
whatever drives the surface (a motor, a person's arm) pays for it, and the work
it did on the gas, in the flow's stats (`push_work_J`; not a ledger term:
the kinetic energy of the flow is not on the books in this solver, as before).

WIND is the same law at the edge of the world: air outside the lattice is a surface
(permeable) that moves at the wind's speed. Its windward faces carry the wind's
speed in, the leeward faces and the sky let the gas out. See `wind_faces`.

All of it runs only inside the gas flow (World.gas_flow), only when a surface or a
wind is present, and a world with neither is not touched.
"""
import numpy as np

from .materials import AIR_DENS

CD_PLATE = 1.2           # drag coefficient of a flat plate across the stream (Hoerner 1965)


class Surface:
    """A set of surface elements that move together or each on their own.
    `pos` (N,3) metres, `area` (N,) m^2, `normal` (N,3) unit, `vel` (N,3) m/s.
    The driver (a scene, a body, a mechanism) overwrites `pos`, `normal` and `vel`
    each tick as the thing moves; the law fills `force` (N,3), the force of the
    gas on each patch, last tick."""

    def __init__(self, pos, area, normal, vel, cd=CD_PLATE):
        self.pos = np.array(pos, np.float64).reshape(-1, 3)
        n = len(self.pos)
        self.area = np.broadcast_to(np.asarray(area, np.float64), (n,)).copy()
        nv = np.broadcast_to(np.asarray(normal, np.float64), (n, 3)).copy()
        self.normal = nv / np.maximum(np.linalg.norm(nv, axis=1, keepdims=True), 1e-12)
        self.vel = np.broadcast_to(np.asarray(vel, np.float64), (n, 3)).copy()
        self.cd = float(cd)
        self.force = np.zeros((n, 3))


def patch_grid(lo, hi, axis, dx, vel, sign=1.0, cd=CD_PLATE):
    """A flat rectangle of patches, one per dx x dx square, between the corners
    `lo` and `hi` (metres) that share their `axis` coordinate, facing +sign along
    it: the shape of a board. Returns a Surface."""
    ax1, ax2 = [a for a in range(3) if a != axis]
    n1 = max(int(round((hi[ax1] - lo[ax1]) / dx)), 1)
    n2 = max(int(round((hi[ax2] - lo[ax2]) / dx)), 1)
    g1 = lo[ax1] + (np.arange(n1) + 0.5) * dx
    g2 = lo[ax2] + (np.arange(n2) + 0.5) * dx
    G1, G2 = np.meshgrid(g1, g2, indexing="ij")
    pos = np.zeros((n1 * n2, 3))
    pos[:, axis] = lo[axis]
    pos[:, ax1] = G1.ravel()
    pos[:, ax2] = G2.ravel()
    nrm = np.zeros(3)
    nrm[axis] = sign
    return Surface(pos, dx * dx, nrm, vel, cd)


class PushMixin:
    """The world's surfaces and wind. Mixed into the gas flow."""

    # a world with none has an empty list and no wind (class defaults, so nothing
    # is added to its __dict__ and a snapshot is unchanged)
    gas_surfaces = ()
    wind = None              # (wx, wy) m/s at `wind_ref_h`, or None
    wind_ref_h = 10.0        # m, the height the wind speed is given at
    wind_alpha = 0.0         # exponent of the wind's rise with height (0 = uniform; 0.14 open country)

    def add_surface(self, surface):
        """Put a Surface in the world (it must be in a world with gas_flow on to do anything)."""
        if "gas_surfaces" not in self.__dict__:
            self.gas_surfaces = []
        self.gas_surfaces.append(surface)
        return surface

    def remove_surface(self, surface):
        if "gas_surfaces" in self.__dict__ and surface in self.gas_surfaces:
            self.gas_surfaces.remove(surface)

    def _push_live(self):
        """Is something pushing: a wind, or a surface that is moving?"""
        return self._wind_vec() is not None or any(
            bool(np.any(s.vel)) for s in self.gas_surfaces)

    def _wind_vec(self):
        w = self.wind
        if w is None:
            return None
        wx, wy = float(w[0]), float(w[1])
        return (wx, wy) if (wx != 0.0 or wy != 0.0) else None

    # ── wind: the windward faces of the world carry the wind's speed in ──────
    def wind_faces(self, F, nc, H):
        """For each lateral axis with wind along it: (axis, side, speeds) where
        `side` is 0 (the low edge) or -1 (the high) and `speeds` the signed
        speed on each face of that edge (shape of the edge, openness-weighted
        later by the solver). The wind's rise with height is a power law."""
        wv = self._wind_vec()
        out = []
        if wv is None or not self.open_sky:
            return out
        for ax in (0, 1):
            w = wv[ax]
            if w == 0.0:
                continue
            side = 0 if w > 0 else -1
            other = [a for a in range(3) if a != ax]
            zc = (np.arange(nc[2]) + 0.5) * H
            prof = np.ones(nc[2]) if self.wind_alpha == 0.0 else \
                (np.maximum(zc, 0.05) / self.wind_ref_h) ** self.wind_alpha
            sp = np.zeros(tuple(nc[a] for a in other))
            sp[...] = prof.reshape([-1 if a == 2 else 1 for a in other]) * w
            out.append((ax, side, sp.astype(np.float32)))
        return out

    # ── surfaces: the sources and impulses the solver adds ───────────────────
    def _push_terms(self, vel_c, nc, H, dt):
        """What the moving surfaces do this tick. `vel_c` is the gas speed at the
        middle of every cell (three arrays). Returns (Q, dU, work, boxes):
        Q the volume the surfaces sweep out of each cell in m^3/s (positive: gas
        must leave), dU three arrays of the speed each cell's gas gains from the
        drag, the work done on the gas (J) and the cells the surfaces touch (a
        box of slices) so the flow will not call that part of the world quiet.
        Fills each surface's `force`."""
        Q = np.zeros(nc, np.float64)
        dU = [np.zeros(nc, np.float64) for _ in range(3)]
        work = 0.0
        lo = np.array([10 ** 9] * 3)
        hi = np.array([-1] * 3)
        top = np.array(nc) - 1
        V = H ** 3
        for s in self.gas_surfaces:
            if len(s.pos) == 0:
                continue
            c0 = np.clip(np.floor(s.pos / H).astype(np.intp), 0, top)
            vn = np.einsum("ij,ij->i", s.vel, s.normal)
            # the gas next to the patch, and its speed along the normal
            g = np.stack([vel_c[a][c0[:, 0], c0[:, 1], c0[:, 2]] for a in range(3)], axis=1)
            w = vn - np.einsum("ij,ij->i", g, s.normal)           # patch speed relative to gas
            lead = np.clip(np.floor((s.pos + s.normal * (0.5 * H)) / H).astype(np.intp), 0, top)
            trail = np.clip(np.floor((s.pos - s.normal * (0.5 * H)) / H).astype(np.intp), 0, top)
            # 1. displacement
            q = s.area * vn
            np.add.at(Q, tuple(lead.T), q)
            np.add.at(Q, tuple(trail.T), -q)
            # 2. drag, half on each side of the patch, never more than closes the gap
            per = np.bincount(np.ravel_multi_index(tuple(c0.T), nc), minlength=int(np.prod(nc)))
            share = 1.0 / np.maximum(per[np.ravel_multi_index(tuple(c0.T), nc)], 1)
            du = 0.5 * s.cd * s.area * np.abs(w) * w * dt / V      # speed gained along +normal
            cap = np.abs(w) * share
            du = np.clip(du, -cap, cap)
            for sd in (lead, trail):
                for a in range(3):
                    np.add.at(dU[a], tuple(sd.T), 0.5 * du * s.normal[:, a])
            f = AIR_DENS * V * du / dt                             # N, on the gas, along the normal
            s.force[:] = -(f[:, None] * s.normal)
            work += float(np.sum(f * vn)) * dt
            for c in (c0, lead, trail):
                lo = np.minimum(lo, c.min(axis=0))
                hi = np.maximum(hi, c.max(axis=0))
        if hi[0] < 0:
            return Q, dU, 0.0, None
        return Q, dU, work, tuple(slice(max(int(l) - 1, 0), min(int(h) + 2, n))
                                  for l, h, n in zip(lo, hi, nc))
