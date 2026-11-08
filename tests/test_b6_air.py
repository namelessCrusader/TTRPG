"""Lane B6 (wave 13a): the exact o2 gate, the _mouth_air extraction, scenes P1/P2/P4.
Fast tier. The 44,000-tick twin comparison is in test_slowtwins.py (slow)."""
import numpy as np
import pytest

from src.voxel import scenes_slow as SS
from src.voxel import slowtwins as TW
from src.voxel.materials import AMBIENT, O2_PER_L, STONE, TICK_S
from src.voxel.sim import World


def _bits(w):
    return {k: getattr(w, k).copy() for k in ("o2", "smoke", "E")}


def _same(a, b):
    for k, v in _bits(a).items():
        assert np.array_equal(v, getattr(b, k)), k


def _pair(build, ticks=30):
    a, b = build(), build()
    a._o2_asleep = lambda T: False          # the law with the gate removed
    for _ in range(ticks):
        a.step()
        b.step()
    return a, b


def test_gate_is_exact_on_a_still_empty_room():
    a, b = _pair(lambda: SS._room())
    assert b._o2_asleep(b.T())
    _same(a, b)


def test_gate_is_exact_with_a_sleeper_and_a_hot_air_cell():
    def build():
        w, _ = SS.scene_p2(2)
        w.E[5:7, 5:7, 20:22] += 3000.0       # a hot patch: the gate must stay open
        return w
    a, b = _pair(build, 20)
    assert not b._o2_asleep(b.T())
    _same(a, b)


def test_gate_stays_open_for_smoke_and_a_deficit():
    w = SS._room()
    w.step()
    assert w._o2_asleep(w.T())
    w.smoke[8, 8, 8] = 0.01
    assert not w._o2_asleep(w.T())
    w.smoke[...] = 0
    w.o2[8, 8, 8] *= 0.9
    assert not w._o2_asleep(w.T())
    w.skip_quiet = False
    w.o2[...] = np.float32(O2_PER_L * w.vox_l)
    assert not w._o2_asleep(w.T())            # without skip_quiet the gate is never taken


def test_mouth_air_is_what_the_twin_computes_and_what_breathe_gets():
    w, p = SS.scene_p1(0.3, settle=5)
    flesh, sl = w._person_cells(p)
    ours = w._mouth_air(p, flesh, sl)
    assert ours == TW.mouth_air(w, p) and ours[0] is True
    seen = []
    real = w._breathe
    w._breathe = lambda pp, g, f, s: (seen.append((g, f, s)), real(pp, g, f, s))[1]
    w.step()
    assert seen and seen[0] == ours


def test_p1_wakes_at_187_2_and_258_4_minutes():
    r = SS.run_p1(0.30)
    assert abs(r["minutes"] - 187.2) < 0.05 and r["event"] == "wakes"
    assert abs(SS.run_p1(0.35)["minutes"] - 258.4) < 0.05


def test_p2_jumps_most_of_a_short_night():
    r = SS.run_p2(3000, settle=60)
    assert r["halted_by"] == "target" and r["jumped"] > 1500


def test_p4_look_is_free_touch_unfolds_and_it_folds_200_ticks_after():
    r = SS.run_p4()
    assert r["look_changed_nothing"] and r["mode_look"] == "folded"
    assert r["mode_touch"] == "live" and 190 <= r["ticks_to_fold"] <= 260
