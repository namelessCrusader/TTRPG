"""Does a declared edge survive being held and reached with? (review probe)"""
import sys, numpy as np
sys.path.insert(0, "/home/nisargparikh/Desktop/Fun Stuff/Mark_1")
from src.voxel.sim import World, STONE, IRON, AIR, FLESH
from src.voxel.scenes import _person, _Wants

w = World(64, 20, 40, voxel_cm=5)
w.fill(0, 64, 0, 20, 0, 1, STONE)
a = _person(w, 17, 10); a["name"] = "attacker"
w.fill(22, 24, 10, 11, 1, 2, IRON, edge=2e-4)
w.policy = _Wants(each={"attacker": {"hands": "take", "legs": "stay"}})
def report(tag):
    e = w.edge > 0
    held = a.get("held")
    hc = w._held_cells(a) if held else None
    he = float(w.edge[tuple(hc.T)].max()) if hc is not None and len(hc) else None
    print(f"{tag} t{w.tick}: edged cells={int(e.sum())} "
          f"in AIR={int((e & (w.mat == AIR)).sum())} in FLESH={int((e & (w.mat == FLESH)).sum())} "
          f"in IRON={int((e & (w.mat == IRON)).sum())} held={bool(held)} held_edge={he}", flush=True)
report("start")
for t in range(60):
    w.step()
    if t % 10 == 9:
        report("take")
w.policy = _Wants(each={"attacker": {"hands": "reach", "legs": "stay"}})
for t in range(80):
    w.step()
    if t % 10 == 9:
        report("reach")
