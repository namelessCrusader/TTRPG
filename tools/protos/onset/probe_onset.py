"""Decides per tick under three attention laws, on the review's smoky room.
latch  = today; nolatch = the old design's edit 4; onset = the proposed law."""
import sys, time, collections
sys.path.insert(0, sys.argv[1])
from src.voxel.sim import World, STONE, WOOD
from src.voxel.scenes import _person

def smoky_room(known):
    w = World(34, 26, 32, voxel_cm=5)
    w.open_sky = False
    w.fill(0, 34, 0, 26, 0, 1, STONE)
    w.fill(0, 34, 0, 26, 31, 32, STONE)
    for (x0, x1, y0, y1) in ((0, 1, 0, 26), (33, 34, 0, 26), (0, 34, 0, 1), (0, 34, 25, 26)):
        w.fill(x0, x1, y0, y1, 0, 32, STONE)
    w.fill(5, 13, 10, 16, 1, 3, WOOD, frac=0.6)
    w.exits = [(32, 13)]
    p = _person(w, 22, 13, z0=1); p["name"] = "the witness"
    if not known:
        p["known"][:] = False
    return w, p

T = int(sys.argv[3]) if len(sys.argv) > 3 else 120
for known in (False, True):
    for law in sys.argv[2].split(","):
        w, p = smoky_room(known)
        if law == "onset":
            w.attention = "onset"
        t0 = time.perf_counter()
        for t in range(T):
            if t < 60:
                w.E[8, 13, 2] += 2500.0
            p["smoke"] = max(p["smoke"], 0.2)
            if law == "nolatch":
                p["emergency"] = False
            w.step()
            p["events"].clear()
        dt = time.perf_counter() - t0
        rows = [r for r in w.traces if r["who"] == "the witness"]
        said = sum(1 for s in w.speech if s[1] == "the witness")
        perc = collections.Counter(str(r["percept"]) + ("*" if r["situation"].get("onset") else "") for r in rows)
        print(f"known_exit={known} law={law:8s} decisions {len(rows):3d}/{T} ticks, shouts {said:3d}, "
              f"safe={p['safe']}, wall {dt:.1f}s, percepts {dict(perc)}", flush=True)
