"""C13 (wave 13c): contact events and the slip law. T3's P3 (Holmes), P8 (seal), P9 (grease).

[I] exact against the old behaviour, [C] code against its own table.
The footfall and the grasp are CONTACT EVENTS (contact.touch): the layers move
between two faces, the contact is logged, and a typed "touches" event is said when
matter moved. The slip law asks for the friction a step needs and compares it with
the friction the floor gives (layers and lattice film), through the one formula.
"""
import numpy as np
import pytest

from src.voxel import contact, marks as M, scenes_contact as S
from src.voxel import layer_data as D
from src.voxel.layers import cell_key
from src.voxel.materials import FRICTION


def _cells(w, p):
    comp, sl = w._person_cells(p)
    return w._claim_cells(p, comp, sl)


def _walk(w, p, x_stop, ticks=400, goal=(55, 7)):
    """Drive the legs (the will is off): until the body's centre passes x_stop."""
    for i in range(ticks):
        comp, _sl = w._person_cells(p)
        if comp is None or not comp.any():
            return i                       # he went over: no longer a body on his feet
        c = _cells(w, p)
        ax, ay = float(c[:, 0].mean()), float(c[:, 1].mean())
        if ax >= x_stop:
            return i
        w._stride(p, c, goal, ax, ay)
        w.step()
    return ticks


def _world(**kw):
    w, p = S.walkway(**kw)
    w._law_will = lambda: None
    return w, p


def _species(L, key):
    return [sorted({sp >> 10 for sp, _g in L._comp(s)}) for s in L.stack(key)]


def _grams(L, key):
    return sum(g for s in L.stack(key) for _sp, g in L._comp(s))


# ---------------------------------------------------------------- P3 Holmes
def test_P3_a_walk_through_mud_then_chalk_leaves_chalk_tracks_first_then_mud():
    w, p = _world(nx=140)
    S.lay(w, 14, 26, S.mud(1.0))
    S.lay(w, 26, 38, S.chalk(0.3))
    L = w.layers
    m0 = L.mass()
    _walk(w, p, 36, goal=(130, 7))
    # the soles carry chalk over mud: counting the layers on the boot tells the route
    for foot in (0, 1):
        sp = _species(L, contact.sole_key(p, foot))
        assert len(sp) >= 2 and sp[0] == [D.CHALK] and set(sp[-1]) == {D.CLAY, D.WATER}
    _walk(w, p, 110, ticks=600, goal=(130, 7))
    assert L.ledger_closes() and L.mass() == pytest.approx(m0, rel=1e-9)
    # the tracks on the bare floor beyond the chalk: chalk near the chalk, mud further on
    track = {x: [_species(L, cell_key(x, y, 0, 4, w.shape)) for y in (6, 7, 8)]
             for x in range(38, 108)}
    near = [s for x in range(38, 60) for s in track[x] if s]
    far = [s for x in range(90, 108) for s in track[x] if s]
    assert near and all(s == [[D.CHALK]] for s in near)                 # chalk first
    assert far and all(set(s[0]) == {D.CLAY, D.WATER} for s in far)    # then mud
    assert not any(D.CHALK in layer for x in range(90, 108) for s in track[x] for layer in s)


def test_P3_a_clean_booted_man_leaves_none_and_a_footfall_costs_nothing_on_a_clean_floor():
    w, p = _world(nx=60)
    L = w.layers
    _walk(w, p, 40)
    assert L.mass() == 0.0 and not L.stack(contact.sole_key(p, 0))
    assert getattr(w, "contacts", None) is None or all(c["moved_g"] == 0 for c in w.contacts)


def test_footfall_is_a_typed_event_with_a_contact_record():
    w, p = _world(nx=60)
    S.lay(w, 10, 30, S.mud(1.0))
    _walk(w, p, 24)
    rec = [c for c in w.contacts if c["moved_g"] > 0]
    assert rec and rec[0]["kind"] == "footfall" and 1e3 < rec[0]["p"] < 1e5   # a body's weight on a sole
    assert rec[0]["actor"] == p["pid"] and 0.0 < rec[0]["c_c"] <= 1.0


def test_the_contact_gate_off_is_the_old_world():
    w, p = _world(nx=60)
    w.contact_laws = False
    S.lay(w, 10, 30, S.mud(1.0))
    m = w.layers.mass()
    _walk(w, p, 24)
    assert w.layers.mass() == m and not getattr(w, "contacts", None)
    assert not w.layers.stack(contact.sole_key(p, 0))


# ---------------------------------------------------------------- P9 grease
def test_P9_grip_falls_with_oil_in_the_layers_and_in_the_lattice():
    w, p = _world(nx=60)
    dry = contact.floor_mu(w, 20, 7, 1)
    assert dry == pytest.approx(0.6)                     # flesh on stone under the friction law
    S.lay(w, 18, 24, S.oil())
    oily = contact.floor_mu(w, 20, 7, 1)
    assert oily < 0.15 and oily == pytest.approx(0.08, abs=0.03)
    # the lattice: a pool of oil on a clean floor reads the same way
    w2, _p2 = _world(nx=60)
    w2.pour(30, 7, 1, 2, 3.0)                             # fluid id 2: oil
    assert w2.fvol[30, 7, 1] > 0.01 * w2.cap
    assert contact.floor_mu(w2, 30, 7, 1) < 0.3 < dry
    w2.mat_laws = frozenset()                             # without the pair law: the 0.40 floor
    assert contact.floor_mu(w2, 5, 7, 1) == FRICTION


def test_P9_the_slip_law_holds_on_a_dry_floor_and_lets_go_on_oil():
    assert contact.slip_speed(0.6, 1.0) == 0.0
    assert contact.slip_speed(0.4, 1.0) == 0.0
    assert contact.slip_speed(0.08, 1.0) > 0.5
    assert contact.slip_speed(0.19, 1.0) < contact.slip_speed(0.08, 1.0)
    # at the boundary it flips exactly: mu = required
    assert contact.slip_speed(contact.required_friction(1.0), 1.0) == 0.0
    # running asks for more
    assert contact.required_friction(2.0) == pytest.approx(2 * contact.required_friction(1.0))


def test_P9_a_man_walking_on_oil_slips_and_the_balance_reflex_pays_for_it_in_time():
    ticks, slips = {}, {}
    for oil in (False, True):
        w, p = _world(nx=60)
        if oil:
            S.lay(w, 16, 34, S.oil())
        ticks[oil] = _walk(w, p, 45)
        slips[oil] = p.get("_slips", 0)
    assert slips[False] == 0 and slips[True] > 10
    # (wave 14b: oil gives 0.08 against the 0.2 to 0.3 a recovery step needs, so
    # the reflex's step slides out too and the man goes over, sooner than the
    # walk would have ended; see tests/test_slip_falls.py)
    assert ticks[True] != ticks[False]


def test_the_toppling_foot_reads_the_floor_it_stands_on():
    w, p = _world(nx=60)
    b = {"cells": np.array([[20, 7, 1], [20, 8, 1], [20, 7, 2]], float)}
    assert w._pivot_mu(b) == pytest.approx(FRICTION)      # dry: the calibrated 0.40
    S.lay(w, 18, 24, S.oil())
    assert w._pivot_mu(b) < 0.15
    w.contact_laws = False
    assert w._pivot_mu(b) == FRICTION


# ---------------------------------------------------------------- P8 seal and the hand
def test_P8_a_hand_with_wax_meets_a_clean_thing_and_a_clean_hand_takes_the_wax():
    w, p = _world(nx=40)
    L = w.layers
    hk = contact.hand_key(p, "right")
    key = cell_key(20, 7, 5, 4, w.shape)
    L.add(hk, {(D.BEESWAX, D.GRAIN, 4): 0.2}, cover=1.0, Ts=60.0)
    cells = np.array([[20, 7, 5]])
    w.mat[20, 7, 5] = 2                                   # something to hold
    m0 = L.mass()
    moved = contact.grasp(w, p, cells, "right")
    assert moved > 0.0 and L.mass() == pytest.approx(m0) and L.stack(key)
    assert w.contacts[-1]["kind"] == "grasp"
    back = contact.grasp(w, p, cells, "right")             # the face has a stack now: the hand takes
    assert back >= 0.0 and L.ledger_closes()


def test_P8_a_seal_is_the_mirror_of_the_die_and_a_reseal_no_longer_fits():
    w, p = _world(nx=40)
    L = w.layers
    die = M.Bitmap(np.pad(np.eye(6), 1) * 200.0)
    seal = M.impress(die, 2e4, M.wax_yield(65.0), 1e-3)
    assert seal is not None
    page = cell_key(10, 7, 0, 4, w.shape)
    L.add(page, {(D.BEESWAX, D.GRAIN, 4): 0.3}, cover=1.0, Ts=65.0, pattern=seal)
    hand = contact.hand_key(p, "left")
    moved = contact.touch(w, "press", p["pid"], page, [hand], 0.9, 2e4, 0.0, 0.03)
    assert moved > 0.0
    lifted = L.patterns[int(L.pat[L.stack(hand)[0]])]
    assert M.ncc(M.mirror(lifted), M.mirror(seal)) == pytest.approx(1.0, abs=1e-9)
    reseal = M.smear(M.blur(seal, 1.0e-9, 5000.0), 2e-3)
    assert M.ncc(reseal, seal) < 0.97
    assert L.ledger_closes()


def test_no_pool_no_work():
    w, p = _world(nx=40)
    w.layers = None
    assert contact.touch(w, "x", 0, 1, [2], 0.5, 1.0, 0.0, 0.1) == 0.0
    assert contact.footfall(w, p, 10, 7, 1, (1, 0), 70.0) == 0.0
