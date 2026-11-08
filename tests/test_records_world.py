"""RECORDS ON THE WORLD (wave 13c, lane C5, T2): the books, the heat readout, the split and the
death of a host, snapshot round trip, and joints. The record laws themselves are tested stand-alone
in tests/test_records_*.py; this file tests that the world carries them."""
import numpy as np

from src.voxel import record as R
from src.voxel.materials import AIR
from src.voxel.scenes_records import glued_pair, quench_blade
from src.voxel.sim import STONE, World


def _gap(w, rid):
    r = w.rec_read[rid]
    return r["T_core"] - r["T_surf"]


def test_no_records_is_the_old_world():
    w = World(10, 10, 6)
    w.fill(0, 10, 0, 10, 0, 1, STONE)
    for _ in range(5):
        w.step()
    assert w.records is None and w.rec_read is None and w.composites is None
    assert w.records_owned() == 0.0 and w.records_owned_E() == 0.0
    assert w.books()["records_owned"] == 0.0
    assert w.records_audit_world() == []
    assert "records" not in vars(w) and "rec_read" not in vars(w)


def test_record_follows_its_host_and_the_audit_is_clean():
    w, rid = quench_blade(6.0)
    for _ in range(60):
        w.step()
        if w.tick % 20 == 0:
            assert w.records_audit_world() == []
    st = w._host_state(int(w.eid[2, 4, 1]))
    g, E = w.records.row_totals(rid)
    assert abs(g.sum() - st[1].sum()) < 1e-6 * st[1].sum()
    assert abs(E - st[2]) < 1e-3 * abs(st[2])
    assert abs(w.rec_read[rid]["T_surf"] - st[3]) < 25.0


def test_thicker_means_a_bigger_surface_to_core_gap_in_water():
    gaps = []
    for th in (2.0, 6.0, 20.0):
        w, rid = quench_blade(th)
        for _ in range(60):
            w.step()
        gaps.append(_gap(w, rid))
    assert gaps[0] < gaps[1] < gaps[2]
    assert gaps[0] >= 0.0


def test_a_port_is_not_reset_every_tick():
    w, rid = quench_blade(2.0)
    for _ in range(40):
        w.step()
    assert w.rec_read[rid]["nsub"] == 1                  # settled ports: one substep


def test_books_close_with_records_in_the_world():
    w, rid = quench_blade(2.0)
    E0 = w.total_energy()
    for _ in range(40):
        w.step()
    # the record owns nothing (a shape-only record): the total is the world's own
    assert w.records_owned() == 0.0 and w.records_owned_E() == 0.0
    w2, _ = quench_blade(2.0, record=False)
    for _ in range(40):
        w2.step()
    assert w.total_energy() == w2.total_energy()         # a record never moves a joule of the lattice
    assert E0 > 0.0


def test_snapshot_round_trip_is_exact():
    w, rid = quench_blade(6.0)
    for _ in range(10):
        w.step()
    snap = w.snapshot()
    for _ in range(20):
        w.step()
    want = (w.rec_read[rid]["T_surf"], w.records.row_totals(rid)[1])
    w.restore(snap)
    for _ in range(20):
        w.step()
    assert (w.rec_read[rid]["T_surf"], w.records.row_totals(rid)[1]) == want


def test_a_thing_that_comes_apart_takes_its_record_with_it():
    w, rid = quench_blade(6.0, water=False)
    e = int(w.eid[2, 4, 1])
    for _ in range(3):
        w.step()
    # cut the blade in two: remove the middle cell (air, nothing in it)
    w.mat[5, 4, 1] = AIR
    w.smass[5, 4, 1] = 0.0
    w.E[5, 4, 1] = 0.0
    w._things_moved(np.array([e]))
    w._id_dirty.add(e)
    for _ in range(3):
        w.step()
    ids = {int(x) for x in w.eid[2:10, 4, 1].ravel() if x}
    assert len(ids) == 2
    hosts = {int(h) for h in w.records.hosts()}
    assert hosts == ids                                   # a record on each piece
    for h in hosts:
        st = w._host_state(h)
        got = sum(w.records.row_totals(r)[0].sum() for r in w.records.rids() if
                  w.records.get(r, "host_id") == h)
        assert abs(got - st[1].sum()) < 1e-6 * st[1].sum()
    assert w.records_audit_world() == []


def test_a_thing_that_is_gone_loses_its_record():
    w, rid = quench_blade(2.0, water=False)
    e = int(w.eid[2, 4, 1])
    w.step()
    w.mat[2:10, 4, 1] = AIR
    w.smass[2:10, 4, 1] = 0.0
    w.E[2:10, 4, 1] = 0.0
    w._things_moved(np.array([e]))
    w._id_dirty.add(e)
    for _ in range(3):
        w.step()
    assert w.records.n == 0 and not any(k != "_stat" for k in w.rec_read)


def test_glue_cures_and_the_composite_comes_apart_at_its_joint():
    w, a, b = glued_pair(grams=2.0)
    (gid,) = w.composites
    rid = w.composites[gid]["joints"][0]["rid"]
    assert w.records_owned() == 2.0                       # the glue line is the record's own matter
    assert w.joint_capacity(gid, rid).Fs == 0.0           # uncured: it holds nothing
    p = w.props(a)
    assert p["parts"] == (a, b) and p["weakest_joint_N"] == 0.0 and p["status"] == "whole"
    # an uncured joint under any load fails: the parts stay, the joint stays, the group is apart
    failed, comps = w.composite_check(gid, {rid: (0.0, 1.0, 0.0)})
    assert failed == rid and len(comps) == 2
    assert w.composites[gid]["status"] == "apart"
    assert w.props(a)["status"] == "apart" and w.props(b) is not None   # both still things


def test_a_cured_joint_holds_below_its_capacity_and_fails_above():
    w, a, b = glued_pair(grams=2.0)
    (gid,) = w.composites
    j = w.composites[gid]["joints"][0]
    for _ in range(40):
        w.step()
    assert 0.0 < j["alpha"] <= 1.0                        # set by cooling, with time
    j["alpha"] = 1.0                                      # (the full cure is days; tested in test_records_joinery)
    cap = w.joint_capacity(gid, j["rid"])
    assert cap.Fs > 0.0
    ok, comps = w.composite_check(gid, {j["rid"]: (0.0, 0.5 * cap.Fs, 0.0)})
    assert ok is None and len(comps) == 1
    bad, comps = w.composite_check(gid, {j["rid"]: (0.0, 1.5 * cap.Fs, 0.0)})
    assert bad == j["rid"] and len(comps) == 2


def test_a_thing_in_no_composite_has_no_composite_props():
    w, a, b = glued_pair()
    other = w.register("stone", "thing")
    assert w.composite_props(other) is None
