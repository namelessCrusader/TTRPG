"""New in S4 (docs/sim-mind-design.md, 4.4): the `_crouch` fix (C5).

`_crouch` reads `joints` in the branch taken only when the drawn body's
bounding corner (`base`) ends up somewhere other than where the body started
(`origin`) -- the knee-crease `_bunch` path can shift cells sideways, not
just down, and in general a crouch need not. Before the fix, `joints` was
never bound in that branch; a crouch that actually took it would raise
NameError. The hygiene test (test_module_hygiene.py) catches the missing
binding statically; this test forces the branch at runtime, so a regression
is also caught by running the body, not only by reading it.

The branch is rare in ordinary play (a standing crouch usually keeps its
`own.min(axis=0)`), so it is forced here rather than hunted for with a
sweep of thetas and facings -- `np.array_equal` is the only one called
inside `_crouch`, so patching it to say "different" for the one call drives
the method into the branch without faking anything else about the draw.
"""
import numpy as np

from src.voxel.scenes import _person, _Wants
from src.voxel.sim import STONE, World


def test_a_CROUCH_that_MOVES_still_has_its_JOINTS(monkeypatch):
    w = World(40, 30, 50, voxel_cm=5)
    w.open_sky = False
    w.fill(0, 40, 0, 30, 0, 1, STONE)
    _person(w, 12, 15)
    w.policy = _Wants()
    w.step()
    p = w.persons[0]
    p["facing"] = (0.0, 1.0)
    assert p.get("joints"), "the body must have joints for this test to mean anything"
    before = {b: np.asarray(v).copy() for b, v in p["joints"].items()}

    import src.voxel.pose as pose_mod
    monkeypatch.setattr(pose_mod.np, "array_equal", lambda a, b: False)
    result = w._crouch(p, 1.0)

    assert result == "moved", f"a forced base-moved crouch should still draw ({result})"
    after = p["joints"]
    assert all(b in after for b in before), "the branch must not drop a joint"
    assert any(not np.array_equal(before[b], np.asarray(after[b])) for b in before), \
        "the shift must have actually carried the joints, or the branch did nothing"

