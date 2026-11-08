"""Where does the last 5% go? Hypothesis: rasterize OVERWRITES the heat that
the AIR cells it draws into already held (matter.py:272 assigns, not adds)."""
import numpy as np
import writeback_probe as wp
from src.voxel.sim import AIR

over = [0.0]
class WB2(wp.WriteBack):
    absorb = False
    def rasterize(self):
        m, w = self.mpm, self.w
        if len(m.x):
            cell = np.clip((m.x * m.inv_dx).astype(np.int64), 0, np.array(w.shape) - 1)
            free = w.mat[tuple(cell.T)] == AIR
            flat = np.unique(np.ravel_multi_index(tuple(cell[free].T), w.shape))
            e_air = w.E.ravel()[flat].astype(np.float64)
            over[0] += float(e_air.sum())
            if self.absorb and len(flat):          # the particle takes the air's heat
                pf = np.ravel_multi_index(tuple(cell[free].T), w.shape)
                pos = np.searchsorted(flat, pf)
                pidx = np.flatnonzero(free)
                Mc = np.bincount(pos, m.m[pidx], len(flat))
                m.E[pidx] += e_air[pos] * m.m[pidx] / Mc[pos]
        super().rasterize()

for absorb in (False, True):
    over[0] = 0.0
    WB2.absorb = absorb
    wp.WriteBack = WB2
    print(f"-- write-back{' + absorb air heat' if absorb else ''}:")
    wp.run(True, ticks=80, torch=1000.0, torch_ticks=60)
    print(f"   lattice heat overwritten by rasterize in AIR cells it drew into: {over[0]:.1f} J")
