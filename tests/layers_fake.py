"""A faithful two-node host for the layer tests (wave 13, T3). No World.

The real host (heat.py) keeps, per cell: C (J/K, the whole cell), Cs (the skin
node's share), E (J above AMBIENT, float32), skin (K, float32: the skin's
temperature above the cell average) and `has` (False for iron, tin, lead, molten
metal, a cell with no open face: no skin node, the whole cell is one node, and
Csur = C). Then T = AMBIENT + E/C, Ts = T + skin where has else T. A partly
filled solid has `af` < 1 (a thinner thing in the cell, face area x af), and cells
in `moving` are not touched by any layer pass.

The booking idiom for energy J into the skin node (heat._burn_cells and others):
    E += J ; skin += J * (1/Cnode - 1/C)        (Cnode = Cs if has else C)
`apply` takes the F2 adjustments (key, dE, dC) of a layer call and also changes the
capacity (dC goes to BOTH Cs and C, so Cb = C - Cs is unchanged); it does it in
float64 and stores E and skin in the host dtype.
"""
import numpy as np

from src.voxel.layers import host_of, make_key

AMBIENT = 20.0


class FakeHost:
    def __init__(self, n=4, C=100.0, Cs=20.0, dtype=np.float32, has=True, dx=0.05):
        self.n = n
        self.dtype = dtype
        self.C = np.full(n, C, np.float64)
        self.Cs = np.full(n, Cs, np.float64)
        self.has = np.full(n, bool(has))
        self.af = np.ones(n, np.float32)
        self.moving = np.zeros(n, bool)
        self.E = np.zeros(n, dtype)
        self.skin = np.zeros(n, dtype)
        self.dx = dx
        self.lost = 0.0                      # float32 rounding the host swallowed

    # ---- reads --------------------------------------------------------------------
    def T(self, i):
        return AMBIENT + float(self.E[i]) / self.C[i]

    def Ts(self, i):
        return self.T(i) + float(self.skin[i]) if self.has[i] else self.T(i)

    def Cnode(self, i):
        return self.Cs[i] if self.has[i] else self.C[i]

    Csur = Cnode

    def keys_of(self, i):
        return [make_key(0, i, f) for f in range(6)]

    def total_E(self):
        return float(self.E.astype(np.float64).sum())

    # ---- writes -------------------------------------------------------------------
    def set_T(self, i, T, skin_T=None):
        """Cell average T and (optionally) a skin temperature; E to match."""
        self.E[i] = (T - AMBIENT) * self.C[i]
        if self.has[i]:
            self.skin[i] = 0.0 if skin_T is None else skin_T - T

    def set_Ts(self, i, Ts):
        """Put the skin node at Ts and the body node at the cell's current mean."""
        T = self.T(i)
        if not self.has[i]:
            self.set_T(i, Ts)
            return
        Es = self.Cs[i] * (Ts - AMBIENT)
        Eb = self.Cb(i) * (T - AMBIENT)
        self.E[i] = Es + Eb
        Tm = AMBIENT + float(self.E[i]) / self.C[i]
        self.skin[i] = Ts - Tm

    def Cb(self, i):
        return self.C[i] - self.Cs[i]

    def book(self, i, J):
        """The repo idiom for J into the skin node (capacity unchanged)."""
        Cn = self.Cnode(i)
        before = float(self.E[i])
        self.E[i] = self.dtype(before + J)
        self.lost += (before + J) - float(self.E[i])
        if self.has[i]:
            self.skin[i] = self.dtype(float(self.skin[i]) + J * (1.0 / Cn - 1.0 / self.C[i]))

    def apply(self, adj):
        """Apply layer adjustments [(key, dE, dC)]."""
        for key, dE, dC in adj:
            i = (host_of(key)) & ((1 << 55) - 1)
            if i >= self.n or self.moving[i]:
                continue
            self._apply_one(i, dE, dC)

    def _apply_one(self, i, dE, dC):
        C, E = self.C[i], float(self.E[i])
        T = AMBIENT + E / C
        if self.has[i]:
            Ts = T + float(self.skin[i])
            Cs = self.Cs[i]
            Es = Cs * (Ts - AMBIENT)
            Eb = E - Es
            Es2 = Es + dE
            Cs2 = Cs + dC
            C2 = C + dC
            Ts2 = AMBIENT + Es2 / Cs2
            E2 = Es2 + Eb
            T2 = AMBIENT + E2 / C2
            self.Cs[i], self.C[i] = Cs2, C2
            self._store(i, E2, Ts2 - T2)
        else:
            self.C[i] = C + dC
            self._store(i, E + dE, 0.0)

    def _store(self, i, E2, skin2):
        e32 = self.dtype(E2)
        self.lost += E2 - float(e32)
        self.E[i] = e32
        self.skin[i] = self.dtype(skin2)

    # ---- one latent step on a cell's faces --------------------------------------------
    def latent(self, layers, i, dt, **kw):
        """Run Layers.latent on cell i's six faces against its ONE skin node and
        apply the result. Returns (T_new, vapour_g)."""
        if self.moving[i]:                       # every pass masks ~w.moving
            return self.Ts(i), 0.0
        kw.setdefault("area", self.dx * self.dx * float(self.af[i]))
        Tn, adj, vap = layers.latent(self.keys_of(i), self.Csur(i), self.Ts(i), dt, **kw)
        self.apply(adj)
        return Tn, vap
