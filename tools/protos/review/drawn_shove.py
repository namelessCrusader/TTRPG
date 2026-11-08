"""Review probe: what happens when a hand-move (_shove) acts on DRAWN particle cells."""
import sys, numpy as np
sys.path.insert(0, "/home/nisargparikh/Desktop/Fun Stuff/Mark_1")
from src.voxel.sim import World, STONE, WOOD, AIR
w = World(20, 12, 16, voxel_cm=5)
w.fill(0, 20, 0, 12, 0, 1, STONE)
w.fill(8, 10, 5, 7, 1, 3, WOOD)
w.matter.promote(np.argwhere(w.mat == WOOD))
for _ in range(15):
    w.step()
m0 = w.total_mass(WOOD)
cells = np.argwhere((w.mat == WOOD) & w.moving)
print("drawn wood cells", len(cells), "total wood", round(m0, 1), "objects in reach would see them:",
      bool(len(cells)))
ok = w._shove(cells.astype(np.int32), 3, 0, 0)
print("shove ok", ok)
for _ in range(3):
    w.step()
print("total wood after", round(w.total_mass(WOOD), 1),
      "lattice wood not moving", round(float(w.smass[(w.mat == WOOD) & ~w.moving].sum()), 1),
      "particles", round(w.matter.mass().get(WOOD, 0.0), 1))
