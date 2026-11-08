"""Is the seam one-way? A promoted wood block rests on a stone floor beside a
torch. Conduction, radiation and burning act on its DRAWN cells; unrasterize
then zeroes those cells. Measure the energy ledger and the wood/smoke books
today, and with a prototype WRITE-BACK (lattice deltas on drawn cells applied
to the particles in that cell, by mass share, before unrasterize zeroes them)."""
import sys
sys.path.insert(0, "/home/nisargparikh/Desktop/Fun Stuff/Mark_1")
import numpy as np
from src.voxel.sim import World, STONE, WOOD, AIR
from src.mpm.matter import Matter


class WriteBack(Matter):
    _rec = None

    def rasterize(self):
        m, w = self.mpm, self.w
        self._rec = None
        if not len(m.x):
            return
        cell = np.clip((m.x * m.inv_dx).astype(np.int64), 0, np.array(w.shape) - 1)
        free = w.mat[tuple(cell.T)] == AIR             # who will be drawn (matter.py:234)
        super().rasterize()
        dflat = np.flatnonzero(self.drawn)
        pidx = np.flatnonzero(free)
        pflat = np.ravel_multi_index(tuple(cell[free].T), w.shape)
        self._rec = (dflat, pidx, np.searchsorted(dflat, pflat),
                     w.E.ravel()[dflat].copy(), w.smass.ravel()[dflat].copy())

    def pending(self):
        """lattice change on drawn cells not yet handed to the particles"""
        if self._rec is None:
            return 0.0, 0.0
        dflat, _, _, E0, S0 = self._rec
        w = self.w
        return (float((w.E.ravel()[dflat].astype(np.float64) - E0).sum()),
                float((w.smass.ravel()[dflat].astype(np.float64) - S0).sum()))

    def unrasterize(self):
        if self._rec is not None:
            w, m = self.w, self.mpm
            dflat, pidx, pos, E0, S0 = self._rec
            dE = w.E.ravel()[dflat].astype(np.float64) - E0.astype(np.float64)
            Snow = w.smass.ravel()[dflat].astype(np.float64)
            Mc = np.bincount(pos, m.m[pidx], len(dflat))
            share = m.m[pidx] / np.maximum(Mc[pos], 1e-300)
            m.E[pidx] += dE[pos] * share
            ratio = np.where(S0 > 0, Snow / np.maximum(S0.astype(np.float64), 1e-300), 1.0)
            m.m[pidx] *= ratio[pos]
            self._rec = None
        super().unrasterize()


def run(wb, ticks=120, torch=20000.0, torch_ticks=60):
    w = World(16, 10, 12, voxel_cm=5)
    if wb:
        w.matter = WriteBack(w)
    w.fill(0, 16, 0, 10, 0, 1, STONE)
    w.fill(6, 9, 4, 6, 1, 3, WOOD)
    w.matter.promote(np.argwhere(w.mat == WOOD))
    for _ in range(10):                   # let it settle onto the floor
        w.step()
    e0, put = w.total_energy(), 0.0
    wood0 = w.total_mass(WOOD)
    smoke0 = float(w.smoke.sum())
    worst = 0.0
    for t in range(ticks):
        if t < torch_ticks:
            w.E[5, 4, 1] += torch; put += torch       # a torch held against its side
        w.step()
        pe = w.matter.pending()[0] if wb else 0.0
        worst = max(worst, abs(w.total_energy() + pe - e0 - put))
    pe, ps = w.matter.pending() if wb else (0.0, 0.0)
    drawn = w.matter.drawn
    print(f"{'write-back' if wb else 'today     '}: energy ledger drift {w.total_energy() + pe - e0 - put:+10.1f} J "
          f"(worst {worst:9.1f}) of {put:.0f} J put in | wood {wood0:.1f} -> {w.total_mass(WOOD) + ps:.1f} g | "
          f"smoke made {float(w.smoke.sum()) - smoke0:.2f} g | particle E {w.matter.energy():.0f} J | "
          f"drawn cells burning {int((w.burning() & drawn).sum())}")


if __name__ == "__main__":
    run(False)
    run(True)
