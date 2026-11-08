import sys
sys.path.insert(0, sys.argv[1])
from src.voxel.sim import World, STONE, WATER


class Heavy:
    """A mind that must never be copied (a model whose weights dwarf the lattice)."""
    def pick(self, sit, menu):
        return 0

    def __deepcopy__(self, memo):
        raise AssertionError("snapshot copied the MIND")


w = World(24, 14, 16, voxel_cm=10)
w.fill(0, 24, 0, 14, 0, 1, STONE)
w.pour(10, 7, 3, WATER, 500.0)
w.policy = Heavy()
for _ in range(3):
    w.step()
try:
    snap = w.snapshot()
except AssertionError as e:
    print("FAIL:", e); sys.exit(1)
w.step()
w.restore(snap)
ok1 = w.matter.w is w
w2 = World(24, 14, 16, voxel_cm=10)
w2.restore(snap)
ok2 = w2.matter.w is w2 and w.matter is not w2.matter
w.step(); w2.step()
print("after restore matter.w is the world:", ok1, "| restored into another world:", ok2,
      "| policy kept:", type(w.policy).__name__, type(w2.policy).__name__)
sys.exit(0 if ok1 and ok2 else 1)
