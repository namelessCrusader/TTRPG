"""Control: the same scene with the block left ON the lattice (never promoted).
How much does the energy ledger drift with no seam at all?"""
import sys
sys.path.insert(0, "/home/nisargparikh/Desktop/Fun Stuff/Mark_1")
import numpy as np
from src.voxel.sim import World, STONE, WOOD
w = World(16, 10, 12, voxel_cm=5)
w.fill(0, 16, 0, 10, 0, 1, STONE)
w.fill(6, 9, 4, 6, 1, 3, WOOD)
for _ in range(10):
    w.step()
e0, put = w.total_energy(), 0.0
for t in range(80):
    if t < 60:
        w.E[5, 4, 1] += 1000.0; put += 1000.0
    w.step()
print(f"lattice only: energy ledger drift {w.total_energy() - e0 - put:+.1f} J of {put:.0f} J put in")
