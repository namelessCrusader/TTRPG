"""FICK TRANSPORT IN SOLIDS (wave 15, S1 slice 2): a dissolved species spreading through the
pore water of a solid, cell to cell, by its own gradient.

transport.py moves pore WATER by a potential, and a solute only rides that flux. Nothing
spread a solute through STILL pore water: a salt gradient in a ham or a tannin gradient in a
hide never evened out. This is that law, one kernel, no craft word:

    flux from cell i to cell j  =  G_ij (c_i - c_j)            [g/s]
    G_ij = D_face A / dx  =  D_face dx   [m3/s]  x 1e6  [ml/s]
    D_face = harmonic(D_i, D_j)
    c = grams of the species DISSOLVED in the cell's fluid lump / ml of that fluid (pore water)

(Fick's first law, Crank, The Mathematics of Diffusion 2nd ed. 1975 (k).) Face based and in
flux form: what leaves one cell enters the next, so the total is conserved exactly (float64
inside the kernel, committed through the pending store of reactions.py).

WHERE IT RUNS. A face is open when both cells hold pore liquid (fvol > 0) and at least one
is a solid (a porous solid); a free liquid beside a porous solid is the surface case. Two
free liquid cells are NOT a face here (the fluid laws own them). Only the DISSOLVED form
moves; BOUND, CRYSTAL and GRAIN stay (so a wash takes the dissolved tannin out of a hide and
leaves what is bound). Water itself keeps its own laws.

D_eff. Free-water D25 with an Arrhenius law (transport.arrhenius), times the host's
measured tortuosity row (an APPARENT diffusivity of the tissue, so the flux per pore-water gradient is
D_app x theta, theta = pore water share of the cell) when matterdata.TORTUOSITY has one (tannin in hide, NaCl in hide or
meat), else Millington-Quirk theta^(10/3)/phi^2 of the cell (transport.host_D's rule). The host
of a cell is its dominant solid species through HOST_OF (data). A free liquid cell has D_free.

THE STEP IS EXPLICIT with substeps under dt_sub = 0.5 min(V_i / sum G_i) (3-D stability and
the maximum principle; the substep count is capped and a pass that hits the cap says so in
`fk.short_s`). It runs inside the ONE box step of reactions.ChemClockMixin (half before the
rows, half after: Strang), never on a clock of its own.

BOUNDARY: a BATH. `fick_hold(cell)` keeps a free-liquid cell at the concentration it has
(a large bath of liquor): the grams it supplies are booked in `fk.bath_g` per species, never
hidden. This is the only boundary condition.

Sources: Crank 1975 (k); Siau, Transport Processes in Wood, Springer 1984 (k) for the
wood case (slice 3); the D rows are matterdata.py's (u), tannin's free-water D25 5e-10 m2/s
and its 18 kJ/mol are (u) kept here until matterdata.py takes them (needs_other_files).
"""
import math

import numpy as np

from . import species as SP
from . import transport as TR
from .materials import TICK_S
from .matterdata import D25, E_D, TORTUOSITY
from .reactions import (CAP_TICKS, N_ULP, ChemClockMixin, LumpView)
from .species_seed import DISSOLVED

# species name -> the key of D25 / E_D / TORTUOSITY (matterdata's own names)
D_KEY = {"tannin": "tannin", "sodium_chloride": "NaCl", "sucrose": "sucrose",
         "ethanol": "ethanol"}
D25_X = dict(D25)
D25_X.setdefault("tannin", 5e-10)         # (u): the row of matterdata.TORTUOSITY is 3e-11/5e-10
E_D_X = dict(E_D)
E_D_X.setdefault("tannin", 18000.0)       # (u) small-solute activation; a tannin is bigger
# dominant solid species -> the host key of TORTUOSITY (data, not code)
HOST_OF = {"collagen": "hide"}
MAX_SUB = 50000


class FickState:
    def __init__(self, species):
        self.species = list(species)                 # species names
        self.ids = [SP.BY_NAME[s] for s in self.species]
        self.fixed = {}                              # flat cell -> {species id: grams}
        self.bath_g = {s: 0.0 for s in self.species}  # grams the baths supplied (+) or took (-)
        self.short_s = 0.0
        self.runs = 0
        self.substeps = 0


class FickSolidsMixin(ChemClockMixin):
    """The Fick kernel. Off (fk None) the world is the old world."""

    def set_fick(self, species=("tannin",)):
        """Turn the kernel on for these species (names in the species registry that have
        a diffusivity). Returns the FickState."""
        for s in species:
            if D_KEY.get(s) not in D25_X:
                raise ValueError("no diffusivity for %r" % s)
        self.ensure_mixes()
        self.fk = FickState(species)
        self._chem_books()
        c = self._chem_clock()
        c.last, c.due = self.tick, self.tick + 40
        return self.fk

    def fick_off(self):
        self.fk = None

    def fick_hold(self, cell):
        """Keep the free-liquid cell `cell` at its present concentration (a bath)."""
        fk = self.fk
        f = int(np.ravel_multi_index(cell, self.shape))
        v = LumpView(self, np.array([f]), gas=False)
        fk.fixed[f] = {sp: float(v.stored("F", sp, DISSOLVED)[0]) for sp in fk.ids}

    def fick_mass_g(self, species):
        """Grams of a dissolved species in every fluid lump, plus its pending store, plus its
        quant and f32 ledgers: the number that is exactly conserved (a bath's supply is
        `fk.bath_g`)."""
        sp = SP.BY_NAME[species]
        v = LumpView(self, np.flatnonzero(self.fvol.reshape(-1) > 0), gas=False)
        tot = float(v.stored("F", sp, DISSOLVED).sum() + np.sum(v.pending(("F", sp, DISSOLVED))))
        if self.mixes is not None:           # the rounding of the quantiser and of float32
            tot += self.mixes.ledger("quant").get(sp, 0.0) + self.mixes.ledger("f32").get(sp, 0.0)
        return tot

    def _fick_pass(self, dt):
        """Diffuse for dt seconds. Returns the seconds the box may sleep (0 = nothing moved)."""
        fk = self.fk
        wetf = self.fvol.reshape(-1) > 0
        if not wetf.any() or dt <= 0.0:
            return 0.0
        shape = self.shape
        wet = wetf.reshape(shape)
        solid = self.mat != 0
        P = wet & solid
        if not P.any():
            return 0.0
        near = P.copy()
        for ax in range(3):
            a = [slice(None)] * 3
            b = [slice(None)] * 3
            a[ax], b[ax] = slice(0, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            near[a] |= P[b]
            near[b] |= P[a]
        part = (P | (wet & ~solid & near))
        idx = np.flatnonzero(part.reshape(-1))
        n = len(idx)
        view = LumpView(self, idx, gas=False)
        K = len(fk.ids)
        m = np.stack([view.grams(("F", sp, DISSOLVED)) for sp in fk.ids], axis=1)   # (n, K)
        if not np.any(m != 0.0):
            return 0.0
        pos = np.full(self.shape, -1, np.int64).reshape(-1)
        pos[idx] = np.arange(n)
        pos = pos.reshape(shape)
        Pi = P.reshape(-1)[idx]
        ia, ib = [], []
        for ax in range(3):
            a = [slice(None)] * 3
            b = [slice(None)] * 3
            a[ax], b[ax] = slice(0, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            both = (pos[a] >= 0) & (pos[b] >= 0)
            pa, pb = pos[a][both], pos[b][both]
            ok = Pi[pa] | Pi[pb]                      # two free liquids: not a face here
            ia.append(pa[ok])
            ib.append(pb[ok])
        ia, ib = np.concatenate(ia), np.concatenate(ib)
        if len(ia) == 0:
            return 0.0
        V = self.fvol.reshape(-1)[idx].astype(np.float64)            # pore ml
        T = view.T()
        # D per cell and species
        mx = self.mixes
        dom = mx.dominant(view.sid)
        host = np.array([HOST_OF.get(SP.SPECIES[int(d)].name, "") if d > 0 else ""
                         for d in dom])
        theta = np.clip(V / self.cap, 1e-6, 1.0)
        phi = np.maximum(np.asarray(self._por_of(self.mat.reshape(-1)[idx],
                                                 self.smass.reshape(-1)[idx],
                                                 self.fvol.reshape(-1)[idx],
                                                 None if not self.mixes_live
                                                 else self.scomp.reshape(-1)[idx]), np.float64),
                         theta)
        D = np.zeros((n, K))
        for k, s in enumerate(fk.species):
            key = D_KEY[s]
            Dw = TR.arrhenius(D25_X[key], E_D_X[key], T, 25.0)
            tau = theta ** (10.0 / 3.0) / np.maximum(phi, 1e-12) ** 2        # Millington-Quirk
            for h in np.unique(host):
                t = TORTUOSITY.get((key, h)) if h else None
                if t is not None:
                    tau = np.where(host == h, t * theta, tau)       # a measured D is APPARENT:
                    # flux = D_app theta grad(c pore), so tissue content theta c obeys D_app
            tau = np.where(Pi, tau, 1.0)                                    # free liquor: D_free
            D[:, k] = Dw * tau
        G = TR.harmonic(D[ia], D[ib]) * view.dx * 1e6                      # ml/s, (nf, K)
        Gsum = np.zeros(n)
        np.add.at(Gsum, ia, G.max(axis=1))
        np.add.at(Gsum, ib, G.max(axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            tmax = np.where(Gsum > 0, V / np.where(Gsum > 0, Gsum, 1.0), np.inf)
        dt_sub = 0.5 * float(tmax.min())
        nsub = int(math.ceil(dt / dt_sub)) if np.isfinite(dt_sub) else 1
        short = 0.0
        if nsub > MAX_SUB:
            short = dt - MAX_SUB * dt_sub
            nsub = MAX_SUB
        h = min(dt, nsub * dt_sub) / nsub
        fixed = [(int(np.searchsorted(idx, f)), d) for f, d in fk.fixed.items()
                 if len(idx) and np.searchsorted(idx, f) < n and idx[np.searchsorted(idx, f)] == f]
        m0 = m.copy()
        Vn = np.maximum(V, 1e-300)[:, None]
        for _ in range(nsub):
            c = m / Vn
            f = G * (c[ia] - c[ib])
            for k in range(K):
                m[:, k] -= np.bincount(ia, weights=f[:, k], minlength=n) * h
                m[:, k] += np.bincount(ib, weights=f[:, k], minlength=n) * h
            for j, d in fixed:                        # a bath keeps its concentration
                for k, sp in enumerate(fk.ids):
                    fk.bath_g[fk.species[k]] += d[sp] - m[j, k]
                    m[j, k] = d[sp]
        fk.runs += 1
        fk.substeps += nsub
        fk.short_s += short
        dm = m - m0
        for k, sp in enumerate(fk.ids):
            view.add(("F", sp, DISSOLVED), dm[:, k])
        view.commit()
        flux = np.abs(dm).max(axis=1) / dt if n else np.zeros(0)
        j = int(np.argmax(flux)) if n else 0
        if n and flux[j] > 0.0:
            need = float(view._thr("F", np.array([max(view.fm[j], 1e-6)]))[0])
            ticks = int(TR.bite_clock(np.array([flux[j] * TICK_S]), need / N_ULP, CAP_TICKS)[0])
            return ticks * TICK_S
        return 0.0
