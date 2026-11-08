import sys, numpy as np
sys.path.insert(0, "/home/nisargparikh/Desktop/Fun Stuff/Mark_1")
from src.voxel.sim import World, STONE, IRON, AIR, FLESH
from src.voxel.scenes import _person, _Wants
w = World(64, 20, 40, voxel_cm=5)
w.fill(0, 64, 0, 20, 0, 1, STONE)
a = _person(w, 17, 10); a["name"] = "attacker"
w.fill(22, 24, 10, 11, 1, 2, IRON, edge=2e-4)
w.policy = _Wants(each={"attacker": {"hands": "take", "legs": "stay"}})
for t in range(12):
    w.step()
arm = a["held"]["arm"]
w.policy = _Wants(each={"attacker": {"hands": "keep", "legs": "stay"}})
def rep(tag):
    e = w.edge > 0
    print(tag, "iron_at", np.argwhere(w.mat == IRON).tolist(), "edged:", 
          [(tuple(c), int(w.mat[tuple(c)])) for c in np.argwhere(e)], flush=True)
rep("before")
for th in (0.3, 0.6, 0.9, 1.2, 1.5):
    r = w._repose(a, arm, th)
    rep(f"repose {arm} {th} -> {r}")
for th in (1.0, 0.5, 0.0):
    r = w._repose(a, arm, th)
    rep(f"back {arm} {th} -> {r}")
other = [n for n in w._limbs(a) if "arm" in n and n != arm][0]
for th in (0.4, 0.8, 1.2):
    r = w._repose(a, other, th)
    rep(f"other {other} {th} -> {r}")
