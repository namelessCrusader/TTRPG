"""SMITHING STAGE 1 in a World (wave 15, lane 1e): the gate, the books, the edge cell, the scenes."""
import numpy as np
import pytest

from src.voxel import edgewear as EW
from src.voxel import heathist as H
from src.voxel import scenes_smith as S
from src.voxel.heathist_world import HeatHistoryMixin
from src.voxel.materials import AMBIENT, IRON, STONE
from src.voxel.sim import World

AXE = 2e-4


def _hot_world(cls, gate):
    w = cls(12, 8, 8, voxel_cm=5)
    w.fill(0, 12, 0, 8, 0, 1, STONE)
    w.fill(4, 7, 3, 6, 1, 4, IRON)
    eid = w.register("bar")
    w.stamp(eid, w.mat == IRON)
    if hasattr(w, "hh_register"):
        w.heat_history = gate
        w.hh_register(eid, row="tool_steel")
    idx = np.flatnonzero((w.mat == IRON).ravel())
    w.E.ravel()[idx] = (900.0 - AMBIENT) * w.heat_capacity().ravel()[idx]
    return w, eid


def test_GATE_OFF_the_world_is_bit_for_bit_the_old_one():
    plain, _ = _hot_world(World, False)
    cls = World
    off, eid = _hot_world(cls, False)
    for _ in range(40):
        plain.step()
        off.step()
    assert np.array_equal(plain.E, off.E) and plain.total_energy() == off.total_energy()
    assert off.heat_temper(eid) is None and off.phase_J == 0.0              # the hand-set temper stays in charge


def test_GATE_ON_never_hot_things_read_temper_one_and_the_edge_keeps_the_hand_number():
    cls = World
    w = cls(12, 8, 8, voxel_cm=5)
    w.fill(0, 12, 0, 8, 0, 1, STONE)
    w.fill(4, 7, 3, 6, 1, 4, IRON)
    eid = w.register("bar")
    w.stamp(eid, w.mat == IRON)
    w.heat_history = True
    w.hh_register(eid, row="tool_steel")
    for _ in range(20):
        w.step()
    assert w.heat_temper(eid) == 1.0
    assert EW._law_temper_thing(w, eid, 0.7) == 1.0                         # the law, not the typed 0.7, once it has a composition
    assert EW._law_temper_thing(w, eid + 99, 0.7) == 0.7                    # a thing with no composition: the hand number


def test_the_EDGE_reads_the_cell_that_carries_it_not_the_thing_scalar():
    cls = World
    w = cls(12, 8, 8, voxel_cm=5)
    w.fill(0, 12, 0, 8, 0, 1, STONE)
    w.fill(4, 7, 3, 4, 1, 2, IRON, edge=AXE)         # the edge cell(s)
    w.fill(4, 7, 4, 6, 1, 2, IRON)                   # the spine, no edge
    eid = w.register("blade")
    w.stamp(eid, w.mat == IRON)
    w.heat_history = True
    w.hh_register(eid, row="tool_steel")
    rec = w._hh[eid]
    e = w.edge.ravel()[rec["idx"]] > 0
    assert e.any() and (~e).any()
    hard = _hard_state()
    for k in rec["s"]:
        rec["s"][k][e] = hard[k][0]                  # only the edge cells were quenched
    r_edge, r_all = w.heat_temper(eid, edge=True), w.heat_temper(eid)
    assert r_edge > 3.0 and r_all < r_edge - 1.0     # a thing-scalar mean would lose the edge's hardness


def _hard_state():
    s = H.new_state(0.8, 0.5)
    H.step(s, 850.0, 60.0)
    H.step(s, 850.0, 300.0)
    T = 850.0
    while T > 20.0:
        T = max(20.0, T - 75.0)
        H.step(s, T, 0.05)
    return s


def test_a_hot_bar_in_the_voxel_world_books_its_latent_heat_and_closes():
    r = S.forge_quench(seconds=30.0)
    assert 0.0 < r["phase_J"] < r["phase_J_start"] and abs(r["energy_drift_J"]) < 1e-6 * r["energy_J"]
    t = r["through"]                                                          # a 5 cm cell is one lump; the through-thickness run sees the gradient
    assert t["skin_HV"] > t["core_HV"] + 100.0 and abs(t["books_drift"]) < 1e-9
    assert r["scale_nm_skin"] > 100.0                                       # scale grew on the hot cells


def test_the_SCENES_run_and_say_what_a_smith_expects():
    a = S.anneal_vs_quench()
    assert a["quench"]["HV"] > 3.0 * a["furnace"]["HV"] and a["air"]["HV"] < 1.2 * a["furnace"]["HV"]
    t = S.temper_ladder()
    hv = [t[T]["HV"] for T in (200, 300, 400, 500, 600)]
    assert hv == sorted(hv, reverse=True) and hv[0] > 0.85 * t["as_quenched"]
    o = S.overheat()
    assert o[1010.0]["d_um"] > 4.0 * o[850.0]["d_um"] and o[1010.0]["tough"] < 0.6 * o[850.0]["tough"]
    b = S.bar_quench()
    assert b["skin_HV"] > b["core_HV"] + 100.0 and abs(b["books_drift"]) < 1e-9
