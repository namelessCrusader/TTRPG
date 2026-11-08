import sys, time
sys.path.insert(0, sys.argv[1])
from src.voxel.sim import World, STONE, IRON, TablePolicy
from src.voxel.scenes import _person, _OneBlow
for name, pol in (("table", TablePolicy()), ("one_blow", _OneBlow())):
    w = World(46, 20, 40, voxel_cm=5)
    w.fill(0, 46, 0, 20, 0, 1, STONE); w.exits = [(44, 10)]
    a = _person(w, 17, 10); a["name"] = "attacker"
    b = _person(w, 31, 10); b["name"] = "defender"
    w.fill(22, 24, 10, 11, 1, 2, IRON, edge=2e-4)
    w.policy = pol
    t0 = time.perf_counter()
    for _ in range(150):
        w.step(); a["events"].clear(); b["events"].clear()
    menus = sum(len(r["menus"]) for r in w.traces)
    print(f"{name}: decides {len(w.traces)}, limb menus {menus}, wound open {any(wd['open'] for wd in b.get('wounds') or [])}, wall {time.perf_counter()-t0:.1f}s")
