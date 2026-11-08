"""numba vs the numpy window path with the explicit left-to-right dot: compare
EVERY per-particle field after EVERY substep (the voxel fast-path discipline)."""
import sys, time
sys.path.insert(0, "/home/nisargparikh/Desktop/Fun Stuff/Mark_1")
sys.path.insert(0, "/tmp/claude-1000/-home-nisargparikh-Desktop-Fun-Stuff-Mark-1/88440bcd-3d95-4f09-8a13-ebe021f962f7/scratchpad/design/mpm")
import numpy as np
import fast_mpm
from compare import CASES, FIELDS

for name, K in (("block", 200), ("plank", 200), ("glass", 400), ("seamlike", 100)):
    a = fast_mpm.as_win(CASES[name](), "window", explicit_dot=True)
    b = fast_mpm.as_win(CASES[name](), "numba")
    dt = a.cfl_dt()
    first_bad = None
    for s in range(K):
        a.substep(dt); b.substep(dt)
        if first_bad is None and not all(np.array_equal(getattr(a, f), getattr(b, f)) for f in FIELDS):
            first_bad = s
    print(f"{name:9s} {K} substeps: numba == numpy(explicit dot) every substep: {first_bad is None}"
          f"{'' if first_bad is None else f' (first diff at substep {first_bad})'}; "
          f"broken {a.broken()}/{b.broken()}, clusters live {int((a.cid>=0).sum())}/{int((b.cid>=0).sum())}")
# and how far the explicit-dot definition moves from today's einsum one
o = CASES["glass"](); e = fast_mpm.as_win(CASES["glass"](), "window", explicit_dot=True)
dt = o.cfl_dt()
for _ in range(400):
    o.substep(dt); e.substep(dt)
print("glass, 400 substeps: today's einsum path vs explicit-dot path, max|dx| =",
      f"{np.abs(o.x - e.x).max():.2e} m, broken {o.broken()} vs {e.broken()}")
