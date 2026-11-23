"""Wave 15 S4-3d: the MPM's four whole-world distance arrays are lazy.

A lazy array must give the same answers as the eager one: the solver reads the
windowed field, so a drop onto a floor is bit-equal whether or not the whole
field was built first, and the audit mode (MARK1_AUDIT_LAZY=1) compares the
window with the definition.
"""
import sys

import numpy as np
import pytest

from src.mpm import MPM
import src.mpm.core as core

NAMES = ("phi", "pgrad", "phi_node", "pgrad_node")


def _drop(build_first=False):
    w = MPM((24, 24, 24), dx=0.1, stiff=1e-4)
    w.add_wall((0.0, 0.0, 0.0), (2.4, 2.4, 0.2))
    w.add_box((0.8, 0.8, 0.6), (1.2, 1.2, 1.0), "wood")
    w.compiled = False
    if build_first:
        w._build_sdf()
    for _ in range(6):
        w.step(0.025)
    return w


def test_the_four_arrays_are_None_until_asked_for():
    w = MPM((20, 20, 20), dx=0.1)
    assert all(getattr(w, n) is None for n in NAMES)
    w.wall[:, :, :2] = True
    w._build_sdf()
    assert w.phi.shape == (20, 20, 20) and w.pgrad.shape == (20, 20, 20, 3)
    assert w.phi_node.shape == (20, 20, 20) and w.pgrad_node.shape == (20, 20, 20, 3)


def test_a_particle_scene_steps_with_the_arrays_None_and_matches_the_eager_run():
    lazy, eager = _drop(False), _drop(True)
    assert all(getattr(lazy, n) is None for n in NAMES)   # the solver never built them
    assert np.array_equal(lazy.x, eager.x) and np.array_equal(lazy.v, eager.v)


def test_a_solver_path_reading_phi_without_building_it_fires(monkeypatch):
    w = MPM((20, 20, 20), dx=0.1)
    with pytest.raises((TypeError, AttributeError)):
        w._sample(w.phi, np.zeros((1, 3)) + 0.5)


def test_audit_mode_passes_on_a_drop_and_catches_a_wrong_window(monkeypatch):
    monkeypatch.setattr(core, "_AUDIT_LAZY", True)
    w = _drop(False)                                     # every window built is checked
    assert all(getattr(w, n) is None for n in NAMES)     # and the arrays dropped again
    near = w._near_walls()                               # a real, correct window
    w._audit_window(near)                                # passes
    bad = near[:3] + (near[3] + 0.01,) + near[4:]        # one wrong by 1 cm
    with pytest.raises(AssertionError, match="lazy distance window"):
        w._audit_window(bad)
    assert all(getattr(w, n) is None for n in NAMES)


_GOLDEN_CHECK = """
import sys, numpy as np
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import golden as g
from src.mpm import MPM
def mk(eager):
    w = MPM((20, 20, 20), dx=0.1)
    w.wall[:, :, :2] = True
    if eager:
        w._build_sdf()
    class M: pass
    class H: pass
    h = H(); h.matter = M(); h.matter.mpm = w; w.w = None
    return h
a, b = mk(False), mk(True)
assert b.matter.mpm.phi is not None and a.matter.mpm.phi is None
assert g._digest(a.matter, a, "matter") == g._digest(b.matter, b, "matter")
print("ok")
"""


def test_golden_ignores_the_four_names_on_either_side():
    """golden.py re-execs itself under PYTHONHASHSEED=0 on import, so it runs
    in a child process."""
    import os, subprocess
    env = dict(os.environ, PYTHONHASHSEED="0")
    r = subprocess.run([sys.executable, "-c", _GOLDEN_CHECK], env=env,
                       capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)) or ".")
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-800:]
