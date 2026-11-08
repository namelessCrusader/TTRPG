"""Close the evidence gap the review found: the WINDOW (numpy, einsum kept)
against the UNMODIFIED MPM.substep, on cases that actually FRACTURE, with
array_equal on every per-particle field after every substep."""
import sys
sys.path.insert(0, "/home/nisargparikh/Desktop/Fun Stuff/Mark_1")
sys.path.insert(0, "/tmp/claude-1000/-home-nisargparikh-Desktop-Fun-Stuff-Mark-1/88440bcd-3d95-4f09-8a13-ebe021f962f7/scratchpad/design/mpm")
import numpy as np
import fast_mpm
from compare import CASES, FIELDS

def bound_glass():
    w = CASES["glass"]()
    w.bind(0)            # bound AND brittle: the _hold_shape parting path
    w.rig[:] = 1.0
    return w

for name, mk, K in (("glass", CASES["glass"], 600), ("bound_glass", bound_glass, 600)):
    o = mk()
    win = fast_mpm.as_win(mk(), "window", explicit_dot=False)
    dt = o.cfl_dt()
    first_bad, first_broken = None, None
    for s in range(K):
        o.substep(dt); win.substep(dt)
        if first_bad is None and not all(np.array_equal(getattr(o, f), getattr(win, f)) for f in FIELDS + ("strength", "m")):
            first_bad = s
        if first_broken is None and o.broken():
            first_broken = s
    print(f"{name:12s} K={K}: window == original every substep: {first_bad is None}"
          f"{'' if first_bad is None else f' (first diff at {first_bad})'}; first crack at substep {first_broken}; "
          f"broken {o.broken()}/{win.broken()}; parted (cid<0 & dmg) {int(((o.cid<0)&o.dmg).sum())}/{int(((win.cid<0)&win.dmg).sum())}")

# an empty solver
from src.mpm import MPM
e = MPM((8, 8, 8), dx=0.1)
print("empty MPM: original step(0.025) substeps =", e.step(0.025))
e2 = fast_mpm.as_win(MPM((8, 8, 8), dx=0.1), "window")
try:
    e2.step(0.025); print("empty window: ok")
except Exception as ex:
    print("empty window:", type(ex).__name__, ex)
