"""Wave 14b, lane H: the balance reflex reads the floor. A recovery step needs
friction (d / h of the lean, contact.step_friction_needed); on oil it slides out
and the man goes over, on a wet floor it grips and he recovers, on a dry floor
nothing changes. A fall from a slip goes through the same tip and landing as any."""
import numpy as np
import pytest

from src.voxel import contact, scenes_contact as S
from tests.test_contact_laws import _cells, _world


def test_a_step_needs_d_over_h_of_friction():
    assert contact.step_friction_needed(0.0, 0.9) == 0.0
    assert contact.step_friction_needed(0.3, 0.9) == pytest.approx(1 / 3)
    assert contact.step_friction_needed(0.1, 0.0) == pytest.approx(0.4)   # the 0.25 m floor on h
    assert contact.step_holds(None, 5.0, 0.9)                              # no floor: not asked
    assert contact.step_holds(0.6, 0.3, 0.9) and not contact.step_holds(0.08, 0.3, 0.9)
    assert contact.step_holds(0.15, 0.1, 0.9)                              # a small lean on a wet floor


def _walk_on(setup, ticks=900, grips=True):
    w, p = _world(nx=70)
    setup(w)
    if not grips:
        w._step_grips = lambda *a, **k: True
    n = 0
    for n in range(ticks):
        comp, _sl = w._person_cells(p)
        if comp is None or not comp.any():
            break
        c = _cells(w, p)
        if c[:, 0].mean() >= 60:
            break
        w._stride(p, c, (65, 7), float(c[:, 0].mean()), float(c[:, 1].mean()))
        w.step()
    went = [e for e in w.events.rows if e.verb == "topples"]
    return w, p, len(went), p.get("_slips", 0)


def _pool(fid):
    def f(w):
        for x in range(16, 50):
            for y in range(5, 10):
                w.pour(x, y, 1, fid, 1.5)
    return f


def test_oil_slips_become_a_fall_and_the_old_reflex_recovered():
    _w, _p, falls_old, slips_old = _walk_on(_pool(2), grips=False)
    w, p, falls, slips = _walk_on(_pool(2))
    assert slips_old > 10 and falls_old == 0           # as before: slides, never goes over
    assert slips > 5 and falls == 1                    # now: the step slides out, he goes over
    why = [e.text for e in w.events.rows if e.result == "slip" and e.why == "reflex:balance"]
    assert why and "slides out" in why[-1]


def test_a_wet_floor_is_many_slips_and_no_fall():
    _w, _p, falls, slips = _walk_on(_pool(1))
    assert slips > 10 and falls == 0                   # wet shoe 0.15 grips a small lean


def test_a_dry_floor_changes_nothing():
    w, p, falls, slips = _walk_on(lambda w: None)
    assert slips == 0 and falls == 0


def test_the_gate_is_contact_laws():
    def setup(w):
        _pool(2)(w)
        w.contact_laws = False
    _w, _p, falls, slips = _walk_on(setup)
    assert slips == 0 and falls == 0                   # off: no slip law, no friction check


def test_the_same_step_grips_on_the_dry_floor_and_slides_on_the_pool():
    w, p = _world(nx=70)
    _pool(2)(w)
    c = _cells(w, p)
    lean = (4.0, 0.0)                                  # 4 cells = 0.2 m past the base
    assert w._step_grips(p, c, (3, 0), lean) is True   # x = 6: the floor is dry there
    c2 = c.copy()
    c2[:, 0] += 20
    assert w._step_grips(p, c2, (3, 0), lean) is False
