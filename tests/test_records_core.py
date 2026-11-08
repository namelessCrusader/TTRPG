"""T2 records: pools, census, audit, absorb/consume, exact split, the ledger term, header
views (spec 3, 4.3, 10; tests 1, 8, 9, 17, 19, 20; critic 21).

Tags (plan section 9): test_echo_* echo, test_codeonly_* code-only, test_u_* unverified data.
All tests below are conservation, exactness, identity or contract tests and may gate a scene.
"""
import copy
import os
import subprocess
import sys
import time

import math

import numpy as np
import pytest

from src.voxel import record as R
from src.voxel import record_data as rd
from src.voxel import record_view as RV

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _recs(n=0):
    recs = R.Records()
    return recs, [R.make_thin(recs, R.SECTION, {"steel": 1.0}, 0.002, 0.04, 0.4, T=100.0 + k,
                              host_id=k + 1) for k in range(n)]


# ── 1. pools ─────────────────────────────────────────────────────────────────
def test_pool_create_free_list_and_rid_never_reused():
    recs, rids = _recs(5)
    assert recs.n == 5 and rids == [1, 2, 3, 4, 5]
    row2 = recs.loc[rids[2]][1]
    recs.kill(rids[2])
    assert recs.n == 4 and rids[2] not in recs.loc
    new = R.make_thin(recs, R.SECTION, {"wood": 1.0}, 0.01, 0.05, 0.2, host_id=9)
    assert new == 6                                           # rid is monotonic, never reused
    assert recs.loc[new][1] == row2                           # the ROW is reused (free list)
    assert set(recs.rids(R.SECTION)) == {1, 2, 4, 5, 6}
    # a recycled row carries nothing of the old record
    assert recs.get(new, "Y").sum() != recs.get(rids[0], "Y").sum()
    assert recs.get(new, "H")[6] == pytest.approx(float(rd.MATERIALS["wood"].H_of_T(20.0)))


def test_pools_grow_past_their_capacity():
    recs = R.Records(cap=4)
    ids = [R.make_thin(recs, R.SECTION, {"steel": 1.0}, 0.002, 0.04, 0.4, host_id=k) for k in range(40)]
    assert recs.n == 40 and recs.pools[R.SECTION].cap >= 40
    assert recs.get(ids[0], "area") == pytest.approx(0.016)


def test_every_class_can_be_made_with_defaults():
    recs = R.Records()
    for cls in R.SCHEMA:
        rid = recs.add(cls, R.HOST_THING, 1)
        assert recs.get(rid, "cls") == cls and recs.get(rid, "alive")
    assert recs.n == len(R.SCHEMA)
    # a bare thin record has no geometry: that is the one thing the audit objects to
    assert {f["cls"] for f in R.records_audit(recs)} == set(R.THIN)


def test_deepcopy_equals_and_caches_stay_outside():
    recs, rids = _recs(6)
    R.heat_tick(recs, 0.025)
    R.cache_of(recs)["scratch"] = np.arange(5)
    dup = copy.deepcopy(recs)
    for cls in recs.pools:
        for name, arr in recs.pools[cls].a.items():
            assert np.array_equal(arr, dup.pools[cls].a[name]), (cls, name)
    assert dup.loc == recs.loc and dup.next_rid == recs.next_rid
    assert "scratch" not in R.cache_of(dup)                    # caches are not state; rebuilt lazily
    # snapshot / restore / step / compare: the copy steps to the same bits
    R.heat_tick(recs, 0.025)
    R.heat_tick(dup, 0.025)
    assert np.array_equal(recs.pools[R.SECTION].a["H"], dup.pools[R.SECTION].a["H"])


_DIGEST = """
import sys
sys.path.insert(0, %r)
import numpy as np
from src.voxel import record as R
recs = R.Records(seed=7)
for k in range(30):
    R.make_thin(recs, R.SECTION if k %% 2 else R.SHEET, {"steel": 0.5, "water": 0.5} if k %% 3 else {"wood": 1.0},
                0.002 + 0.0001 * k, 0.04, 0.3, T=20.0 + 10 * k, host_id=k %% 5 + 1,
                ports=dict(h=(0.0, 100.0 + k)))
for _ in range(5):
    recs.begin_tick(); R.heat_tick(recs, 0.025)
print(float(recs.pools[R.SECTION].a["H"].sum()), float(recs.pools[R.SHEET].a["H"].sum()),
      int(recs.pools[R.SECTION].a["seed"].sum()))
"""


def test_result_is_independent_of_PYTHONHASHSEED():
    outs = []
    for hs in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.append(subprocess.run([sys.executable, "-c", _DIGEST % ROOT], capture_output=True,
                                   text=True, env=env, cwd=ROOT, check=True).stdout)
    assert outs[0] == outs[1] == outs[2] and outs[0].strip()


# ── census (3.10, resolution E10) ────────────────────────────────────────────
def _sources(n_eid, rng):
    def src(n):
        return (rng.integers(0, n_eid, n).astype(np.int32), rng.integers(0, rd.NS, n).astype(np.int8),
                (rng.random(n) * 100).astype(np.float32), (rng.random(n) * 1e4).astype(np.float32))
    return src


def test_census_equals_the_per_thing_scan():
    rng = np.random.default_rng(1)
    n_eid = 50
    src = _sources(n_eid, rng)
    a, b = src(3000), src(700)
    cen = R.census(n_eid, a, b, count=True)
    ref_g = np.zeros((n_eid, rd.NS))
    ref_E = np.zeros(n_eid)
    for eid, sp, m, E in (a, b):
        for e in range(n_eid):
            sel = eid == e
            for s in range(rd.NS):
                ref_g[e, s] += m[sel & (sp == s)].astype(np.float64).sum()
            ref_E[e] += E[sel].astype(np.float64).sum()
    assert np.allclose(cen.g, ref_g, rtol=1e-12, atol=1e-9)
    assert np.allclose(cen.E, ref_E, rtol=1e-12)
    assert cen.count.sum() == 3700
    # stray entries (unregistered cells, species -1) are skipped, not fatal
    bad = (np.array([-1, 3, 999]), np.array([0, -1, 0]), np.ones(3), np.ones(3))
    c2 = R.census(n_eid, bad)
    assert c2.g.sum() == 0.0


def test_benchmark_100_active_records_census_50k_particles_200k_lattice_under_2ms():
    rng = np.random.default_rng(2)
    n_eid = 2000
    src = _sources(n_eid, rng)
    lattice, particles = src(200000), src(50000)
    ids = np.arange(100) * 7
    scratch = {}
    best = 1.0
    for _ in range(40):
        t = time.perf_counter()
        cen = R.census(n_eid, lattice, particles, scratch=scratch)
        g, E = cen.g[ids], cen.E[ids]
        best = min(best, time.perf_counter() - t)
    print("grouped census, 250k entries, 100 records: %.3f ms" % (best * 1e3))
    assert best < 2e-3
    assert g.shape == (100, rd.NS)


def test_lattice_energy_share_is_the_capacity_share():
    share = R.lattice_energy_share(np.array([1000.0]), np.array([2.0]), 500.0, np.array([4000.0]))
    assert share[0] == pytest.approx(1000.0 * 2.0 * 500.0 / 4000.0)


# ── 17. audit ────────────────────────────────────────────────────────────────
def _hosted(n):
    recs, rids = _recs(n)
    cen = R.Census(n + 2)
    for rid in rids:
        g, E = recs.row_totals(rid)
        h = int(recs.get(rid, "host_id"))
        cen.g[h] += g
        cen.E[h] += E
    return recs, rids, cen


def test_audit_clean_world_is_clean():
    recs, rids, cen = _hosted(8)
    assert R.records_audit(recs, host=cen) == []


def test_audit_flags_a_corrupted_record():
    recs, rids, cen = _hosted(8)
    p, r = recs.row(rids[3])
    p.a["Y"][r, 2, rd.SP["steel"]] *= 1.0001                  # a grain of steel appears
    f = R.records_audit(recs, host=cen)
    assert any(x["kind"] == "host" and x["host"] == 4 for x in f), f
    p.a["Y"][r, 2, rd.SP["steel"]] = np.nan
    assert any(x["kind"] == "corrupt" for x in R.records_audit(recs))
    recs2, rids2, cen2 = _hosted(3)
    recs2.set(rids2[0], "H", np.full(R.NN, 1.0e6))            # an energy the host never had
    assert any(x["kind"] == "host" and "energy" in x["detail"] for x in R.records_audit(recs2, host=cen2))
    recs3, rids3, _ = _hosted(3)
    recs3.set(rids3[1], "thick", 0.02)                        # thickness no longer follows the grams
    assert any("thickness" in x["detail"] for x in R.records_audit(recs3))


def test_audit_tolerance_follows_the_dtype():
    assert R.tol_for(np.float64) == 1e-12
    assert R.tol_for(np.float32, 8) == pytest.approx(8 * 2.0 ** -23)
    recs, rids, cen = _hosted(2)
    cen.E[1] *= 1.0 + 1e-9                                     # float32-sized disagreement
    assert R.records_audit(recs, host=cen)                     # a float64 host: a finding
    assert R.records_audit(recs, host=cen, tol_host=R.tol_for(np.float32, 4)) == []


def test_audit_writes_to_the_sink():
    recs, rids, cen = _hosted(2)
    cen.g[1, rd.SP["steel"]] += 1.0
    sink = []
    R.records_audit(recs, host=cen, sink=sink)
    assert sink and sink[0]["kind"] == "host"


def test_audit_absorb_meter_counts_what_reconcile_fixed():
    recs, rids, cen = _hosted(3)
    recs.begin_tick()
    cen.E[2] += 7.0                                            # something changed the host
    R.reconcile(recs, cen)
    assert recs.meter[(2, "reconcile")][2] == pytest.approx(7.0)
    assert any(f["kind"] == "absorb" and f["host"] == 2 for f in R.records_audit(recs, host=cen))
    recs.touch(2)                                              # an external law owns up to it
    assert not any(f["kind"] == "absorb" for f in R.records_audit(recs, host=cen))
    recs.begin_tick()
    assert recs.meter == {}                                    # a fresh meter every tick


# ── 8. absorb / consume ──────────────────────────────────────────────────────
def test_absorb_conserves_and_spreads_by_capacity():
    recs, rids = _recs(1)
    rid = rids[0]
    e0 = recs.energy_J(rid)
    assert R.absorb(recs, rid, 1234.5) == 1234.5
    assert recs.energy_J(rid) - e0 == pytest.approx(1234.5, rel=1e-12)
    assert R.absorb(recs, rid, -300.0) == -300.0
    assert recs.energy_J(rid) - e0 == pytest.approx(934.5, rel=1e-12)
    H = recs.get(rid, "H")[:7]
    assert np.allclose(H - H[0], 0.0, atol=1e-9)               # uniform capacity: uniform rise
    assert recs.meter[(1, "absorb")][0] == pytest.approx(1534.5)


def test_consume_surface_takes_outer_nodes_first_and_grows_loss():
    recs = R.Records()
    rid = R.make_thin(recs, R.SECTION, {"wood": 1.0}, 0.01, 0.05, 0.2, T=200.0, host_id=1)
    Y0 = recs.get(rid, "Y").copy()
    wood = rd.SP["wood"]
    e0 = recs.energy_J(rid)
    # a small bite comes off the face node only (no regrid yet)
    got, J, left = R.consume(recs, rid, "wood", Y0[6, wood] * 0.4)
    Y = recs.get(rid, "Y")
    assert Y[6, wood] == pytest.approx(Y0[6, wood] * 0.6) and np.array_equal(Y[:6], Y0[:6])
    assert J == pytest.approx(got / 1000.0 * float(recs.get(rid, "H")[6]), rel=1e-9)
    loss1 = recs.get(rid, "loss")
    assert loss1 == pytest.approx(got / 1000.0 / 700.0 / 0.01 / 2.0, rel=1e-9)   # m per face
    # more than the face node holds: the next node is eaten, the profile is regridded
    take = Y0[6, wood] * 1.5
    got2, J2, left2 = R.consume(recs, rid, "wood", take)
    assert got2 == pytest.approx(take) and left2 == 0.0
    assert recs.get(rid, "loss") > loss1
    Y1 = recs.get(rid, "Y")
    assert Y1.sum() == pytest.approx(Y0.sum() - got - got2, rel=1e-12)
    assert recs.energy_J(rid) == pytest.approx(e0 - J - J2, rel=1e-9)   # the matter took its enthalpy with it
    want_thick = 0.01 * (1.0 - (got + got2) / Y0.sum())
    assert recs.get(rid, "thick") == pytest.approx(want_thick, rel=1e-9)
    assert Y1[:7, wood].min() > 0.0                              # no empty node is left for the solver
    assert not R.records_audit(recs)
    R.heat_tick(recs, 0.025)                                    # and the solve is finite
    assert np.isfinite(recs.get(rid, "H")).all()


def test_regrid_is_conservative():
    recs = R.Records()
    rid = R.make_thin(recs, R.SECTION, {"wood": 0.6, "water": 0.4}, 0.01, 0.05, 0.2, T=50.0, host_id=1)
    p, r = recs.row(rid)
    rng = np.random.default_rng(3)
    p.a["Y"][r, :7] *= rng.uniform(0.2, 1.8, (7, 1))
    p.a["H"][r, :7] = np.sort(rng.uniform(0, 2e5, 7))
    R.refresh_geometry(recs, rid)
    g0, E0 = recs.row_totals(rid)
    th0 = recs.get(rid, "thick")
    assert R.regrid(recs, rid)
    g1, E1 = recs.row_totals(rid)
    assert np.allclose(g1, g0, rtol=1e-13) and E1 == pytest.approx(E0, rel=1e-12)
    assert recs.get(rid, "thick") == pytest.approx(th0, rel=1e-12)
    v = rd.node_volumes(recs.get(rid, "Y")[:7])
    assert np.allclose(v / v.sum(), R.th.node_fractions(7, True), rtol=1e-12)    # uniform again


def test_consume_bulk_takes_the_wettest_node_first():
    recs = R.Records()
    rid = R.make_thin(recs, R.SHEET, {"hide": 0.7, "water": 0.3}, 0.004, 0.2, 0.2, host_id=1)
    w = rd.SP["water"]
    recs.get(rid, "Y")[3, w] *= 3.0                            # node 3 is the wet one
    before = recs.get(rid, "Y")[:, w].copy()
    R.consume(recs, rid, "water", before[3] * 0.5, where="bulk")
    after = recs.get(rid, "Y")[:, w]
    assert after[3] == pytest.approx(before[3] * 0.5) and np.array_equal(after[[0, 1, 2, 4, 5, 6]], before[[0, 1, 2, 4, 5, 6]])


def test_over_consume_returns_the_remainder_and_never_goes_negative():
    recs = R.Records()
    rid = R.make_thin(recs, R.SECTION, {"water": 0.5, "hide": 0.5}, 0.004, 0.1, 0.1, host_id=1)
    w = rd.SP["water"]
    have = float(recs.get(rid, "Y")[:, w].sum())
    got, J, left = R.consume(recs, rid, "water", have * 3.0)
    assert got == pytest.approx(have, rel=1e-12) and left == pytest.approx(2.0 * have, rel=1e-9)
    assert (recs.get(rid, "Y") >= 0.0).all() and recs.get(rid, "Y")[:, w].sum() == 0.0
    assert R.consume(recs, rid, "water", 5.0)[0] == 0.0       # nothing left to take
    assert R.consume(recs, rid, "water", -5.0)[0] == 0.0      # a negative ask takes nothing
    # a class (b) record: never negative either
    lot = recs.add(R.LOT, R.HOST_THING, 2)
    recs.get(lot, "g")[rd.SP["grain"]] = 100.0
    recs.set(lot, "E", 50.0)
    got, J, left = R.consume(recs, lot, "grain", 160.0)
    assert (got, left) == (100.0, 60.0) and recs.get(lot, "g").min() >= 0.0
    assert J == pytest.approx(50.0)


def test_reconcile_sheds_a_burnt_host_loss_surface_first_and_audit_holds():
    recs = R.Records()
    rid = R.make_thin(recs, R.SECTION, {"wood": 1.0}, 0.01, 0.05, 0.2, T=300.0, host_id=1)
    g, E = recs.row_totals(rid)
    cen = R.Census(3)
    burnt = 8.0                                               # grams the burn law took
    cen.g[1] = g
    cen.g[1, rd.SP["wood"]] -= burnt
    cen.E[1] = E - burnt / 1000.0 * float(recs.get(rid, "H")[6])   # and the energy that went with it
    recs.begin_tick()
    recs.touch(1)                                              # the burn law touched this host
    R.reconcile(recs, cen)
    assert R.records_audit(recs, host=cen) == []
    assert recs.get(rid, "loss") > 0.0                          # shed from the surface: the slab is thinner
    assert recs.get(rid, "thick") == pytest.approx(0.01 * (1.0 - burnt / g.sum()), rel=1e-9)
    # a gain goes to the surface node
    cen.g[1, rd.SP["wood"]] += 5.0
    cen.E[1] += 5.0 / 1000.0 * float(recs.get(rid, "H")[6])
    before = recs.get(rid, "Y")[6, rd.SP["wood"]]
    R.reconcile(recs, cen)
    assert recs.get(rid, "Y")[:, rd.SP["wood"]].sum() == pytest.approx(g[rd.SP["wood"]] - burnt + 5.0, rel=1e-12)
    assert R.records_audit(recs, host=cen) == []


# ── 9. exact row split ───────────────────────────────────────────────────────
def test_row_split_extensive_sums_to_1e15_and_second_is_whole_minus_first():
    rng = np.random.default_rng(5)
    worst = 0.0
    for k in range(60):
        recs = R.Records()
        comp = {"steel": float(rng.random()), "wood": float(rng.random()), "water": float(rng.random())}
        rid = R.make_thin(recs, R.SECTION, comp, float(rng.uniform(0.001, 0.01)), 0.05,
                          float(rng.uniform(0.1, 0.5)), T=float(rng.uniform(0, 800)), host_id=3)
        g0, E0 = recs.row_totals(rid)
        len0, area0, Y0 = recs.get(rid, "seg_len"), recs.get(rid, "area"), recs.get(rid, "Y").copy()
        child = recs.split(rid, float(rng.uniform(0.05, 0.95)))
        g1, E1 = recs.row_totals(rid)
        g2, E2 = recs.row_totals(child)
        worst = max(worst, abs(E1 + E2 - E0) / abs(E0), float(np.max(np.abs(g1 + g2 - g0)[g0 > 0] / g0[g0 > 0])),
                    abs(recs.get(rid, "seg_len") + recs.get(child, "seg_len") - len0) / len0)
        # the second share is whole minus first, exactly
        assert np.array_equal(recs.get(child, "Y"), Y0 - recs.get(rid, "Y"))
        assert recs.get(child, "host_id") == 3 and recs.get(child, "H")[3] == recs.get(rid, "H")[3]
        assert not R.records_audit(recs)
    assert worst < 1e-15, worst


def test_split_counts_stay_integers_and_add_exactly():
    recs = R.Records()
    lot = recs.add(R.LOT, R.HOST_THING, 1, n_items=1000)
    recs.get(lot, "hist")[:] = np.arange(256).reshape(16, 16) * 3 + 1
    recs.get(lot, "g")[rd.SP["grain"]] = 777.0
    h0 = recs.get(lot, "hist").copy()
    for f in (0.1, 0.333, 0.5, 0.999):
        l2 = recs.add(R.LOT, R.HOST_THING, 1, n_items=1000)
        recs.get(l2, "hist")[:] = h0
        recs.get(l2, "g")[rd.SP["grain"]] = 777.0
        c2 = recs.split(l2, f)
        assert recs.get(l2, "n_items") + recs.get(c2, "n_items") == 1000
        assert (recs.get(l2, "hist") + recs.get(c2, "hist") == h0).all()
        assert recs.get(l2, "n_items").dtype.kind == "i"
        assert recs.get(l2, "g")[rd.SP["grain"]] + recs.get(c2, "g")[rd.SP["grain"]] == 777.0


def test_split_rejects_the_endpoints():
    recs, rids = _recs(1)
    for f in (0.0, 1.0, -0.2):
        with pytest.raises(ValueError):
            recs.split(rids[0], f)


# ── 20. the ledger term ──────────────────────────────────────────────────────
def test_records_owned_is_exactly_zero_when_empty():
    recs = R.Records()
    o = R.records_owned(recs)
    assert o.E == 0.0 and not o.g.any() and not o.ml.any() and o.mass_total() == 0.0
    assert R.records_owned(None).E == 0.0
    # shape-only records never enter the ledger
    R.make_thin(recs, R.SECTION, {"steel": 1.0}, 0.002, 0.04, 0.4, T=500.0, host_id=1)
    assert R.records_owned(recs).E == 0.0 and R.records_owned(recs).mass_total() == 0.0
    # and a killed owner leaves the ledger at exact zero again
    lot = recs.add(R.LOT, R.HOST_THING, 1)
    recs.get(lot, "g")[rd.SP["grain"]] = 12.5
    assert R.records_owned(recs).mass_total() == 12.5
    recs.kill(lot)
    assert R.records_owned(recs).mass_total() == 0.0 and R.records_owned(recs).E == 0.0


def test_records_owned_sums_a_lot_a_glue_line_and_a_cup_of_tea():
    recs = R.Records()
    lot = recs.add(R.LOT, R.HOST_THING, 1, n_items=1000)
    recs.get(lot, "g")[rd.SP["grain"]] = 1000 * 0.04 * 1000.0 / 1000.0     # 1000 seeds of 40 mg: 40 g
    recs.set(lot, "E", 3.0)
    glue = recs.add(R.JOIN, R.HOST_THING, 2)
    recs.get(glue, "g")[rd.SP["glue"]] = 0.8                  # a 0.8 g glue line
    recs.set(glue, "E", 0.5)
    cup = recs.add(R.VESSEL, R.HOST_THING, 3)                 # a cup of tea: 250 ml
    recs.get(cup, "ml")[0] = 250.0
    recs.get(cup, "g")[rd.SP["water"]] = 249.5
    recs.set(cup, "E", 5000.0)
    voxel = recs.add(R.VESSEL, R.HOST_THING, 4)               # a voxel-scale vessel: the lattice owns it
    recs.set(voxel, "owns", False)
    recs.get(voxel, "ml")[0] = 9999.0
    recs.get(voxel, "g")[rd.SP["water"]] = 9999.0
    o = R.records_owned(recs)
    assert o.g[rd.SP["grain"]] == pytest.approx(40.0) and o.g[rd.SP["glue"]] == 0.8
    assert o.g[rd.SP["water"]] == 249.5 and o.ml[0] == 250.0
    assert o.E == pytest.approx(5003.5) and o.mass_total() == pytest.approx(290.3)
    assert o.species()["glue"] == 0.8
    # the audit checks class (b) against the ledger
    ledger = {"g": o.g.copy(), "E": o.E}
    assert R.records_audit(recs, ledger=ledger) == []
    ledger["g"][rd.SP["glue"]] += 0.1
    assert any(f["kind"] == "ledger" for f in R.records_audit(recs, ledger=ledger))


def test_remainders_are_part_of_the_term():
    recs = R.Records()
    recs.rem_add(5, ("m", rd.SP["water"]), 1e-9)
    recs.rem_add(5, "E", 2e-7)
    o = R.records_owned(recs)
    assert o.g[rd.SP["water"]] == 1e-9 and o.E == 2e-7
    assert recs.rem_take(5, "E") == 2e-7 and recs.rem_take(5, "E") == 0.0


# ── record_view (critic 21) ──────────────────────────────────────────────────
LAYERS = {"rows": [dict(rid=11, tag=0, host=500, mass_g=12.0, E_J=30.0),
                   dict(rid=12, tag=1, host=3, mass_g=0.4, E_J=1.0),
                   dict(rid=13, tag=2, host=17, mass_g=2.0, E_J=0.0),
                   dict(rid=14, tag=3, host=9001, mass_g=100.0, E_J=-5.0)],
          "totals": {"mass_g": 114.4, "E_J": 26.0}}
BODY = {"person": 2, "bones": [dict(rid=21, zone=0, mass_g=900.0), dict(rid=22, zone=5, mass_g=400.0)],
        "wounds": [dict(rid=23, zone=5, blood_ml=12.5)], "totals": {"mass_g": 1300.0, "ml": 12.5}}
CLOTHES = [dict(rid=31, name="coat", mass_g=1100.0, E_J=40.0), dict(rid=32, name="hat", mass_g=90.0)]
MECH = {"rid": 41, "host": 8, "parts": [{"mass_g": 30.0}, {"mass_g": 12.0}], "stored_J": 3.5,
        "totals": {"mass_g": 42.0, "E_J": 3.5}}


def test_foreign_stores_each_yield_a_header():
    lay = RV.view_layers(LAYERS).headers()
    assert [h.cls for h in lay] == ["LAYER"] * 4 and [h.rid for h in lay] == [11, 12, 13, 14]
    assert [h.host_kind for h in lay] == [R.HOST_CELL, R.HOST_THING, R.HOST_REGION, R.HOST_PLOT]
    assert lay[1].host_id == 3 and lay[3].totals["mass_g"] == 100.0
    body = RV.view_body(BODY).headers()
    assert [h.cls for h in body] == ["BONE", "BONE", "WOUND"]
    assert body[1].host_id == 2 * 16 + 5 and body[2].totals == {"ml": 12.5}
    clo = RV.view_clothes(CLOTHES, host_id=2).headers()
    assert [h.rid for h in clo] == [31, 32] and clo[0].host_id == 2
    mech = RV.view_mechanism(MECH).headers()
    assert mech[0].cls == "MECHANISM" and mech[0].totals["mass_g"] == 42.0 and mech[0].host_id == 8
    # the header has the pools' own fields
    recs, rids = _recs(2)
    assert RV.headers_of(recs)[0]._fields == lay[0]._fields
    assert RV.headers_of(recs)[0].cls == "SECTION" and len(RV.headers_of(recs)) == 2


def test_one_audit_covers_every_store_and_flags_a_corrupted_foreign_total():
    recs, rids, cen = _hosted(3)
    views = [RV.view_layers(LAYERS), RV.view_body(BODY), RV.view_clothes(CLOTHES, 2, {"mass_g": 1190.0, "E_J": 40.0}),
             RV.view_mechanism(MECH)]
    assert R.records_audit(recs, host=cen, views=views) == []
    bad_layers = copy_dict(LAYERS)
    bad_layers["totals"]["mass_g"] = 999.0                    # the store's totals field is corrupted
    f = R.records_audit(recs, host=cen, views=[RV.view_layers(bad_layers)] + views[1:])
    assert [x["kind"] for x in f] == ["foreign_totals"] and f[0]["store"] == "layers"
    bad_body = copy_dict(BODY)
    bad_body["wounds"][0]["blood_ml"] = float("nan")
    assert R.records_audit(recs, views=[RV.view_body(bad_body)])
    bad_mech = copy_dict(MECH)
    bad_mech["stored_J"] = 9.0
    f = R.records_audit(recs, views=[RV.view_mechanism(bad_mech)])
    assert f and f[0]["store"] == "mechanism" and f[0]["field"] == "E_J"
    bad_cl = [dict(rid=31, name="coat", mass_g=-5.0)]
    assert any(x["kind"] == "corrupt" for x in R.records_audit(recs, views=[RV.view_clothes(bad_cl, 2)]))


def copy_dict(d):
    return copy.deepcopy(d)


def test_views_are_read_only_snapshots():
    v = RV.view_layers(LAYERS)
    hs = v.headers()
    hs.clear()                                                 # mutating the list does not touch the view
    assert len(v.headers()) == 4
    assert v.totals()["mass_g"] == pytest.approx(114.4)


# ── data rows (record_data) ──────────────────────────────────────────────────
def test_codeonly_strength_rows_reproduce_the_spec_worked_example():
    """Spec 6.1: hardened blade K_Ic 20 MPa sqrt(m), E 210 GPa, nu 0.3 -> G_c = 1.73 kJ/m2;
    the mild-steel row yields (eps_f 25 percent, K_Ic over 100). Code against its own table.
    The hard row's eps_f is 2 percent, not the spec prose's 0.5: see record_data."""
    hard, mild = rd.STRENGTH["steel_hard"], rd.STRENGTH["steel"]
    assert rd.g_c(hard) == pytest.approx(1.73e3, rel=0.01)
    assert hard["eps_f"] == 0.02 and mild["eps_f"] == 0.25 and mild["K_Ic"] >= 100e6
    for name, row in rd.STRENGTH.items():
        assert row["sigma_y"] is None or row["sigma_u"] >= row["sigma_y"] > 0, name
        assert row["weibull_m"] is None or row["weibull_m"] > 1, name


def test_every_strength_row_can_reach_its_yield_before_it_fractures():
    """A row that yields cannot fracture at a strain below its yield strain (sigma_y / E):
    eps_f is 0 (brittle, no tear arm) or at least that. The hard steel row once had eps_f
    0.5 percent against a yield strain of 0.86 percent."""
    for name, row in rd.STRENGTH.items():
        if row["sigma_y"] is not None and row["eps_f"] > 0.0:
            assert row["eps_f"] >= row["sigma_y"] / row["E"], name


def test_one_source_of_truth_breaking_reads_record_data():
    """Every number of breaking.STRENGTH is the record_data row's (the table is typed once),
    and the two modules agree on the rows that used to be typed twice."""
    from src.voxel import breaking as B
    rows = {"steel_mild": "steel"}
    for name, st in B.STRENGTH.items():
        r = rd.STRENGTH[rows.get(name, name)]
        assert (st.E, st.nu, st.sigma_y, st.sigma_u, st.eps_f, st.K_Ic, st.m, st.rho) == (
            r["E"], r["nu"], r["sigma_y"], r["sigma_u"], r["eps_f"], r["K_Ic"], r["weibull_m"], r["rho"]), name
        assert st.a0 == r.get("a0", B.A0), name
    assert B.STRENGTH["steel_hard"].eps_f == rd.STRENGTH["steel_hard"]["eps_f"] == 0.02


def test_dowel_Rd_is_one_copy_and_follows_the_nds_by_diameter():
    """NDS 12.3.1B mode IV: 2.2 up to 0.17 in, 10 D + 0.5 to 0.25 in, 3.2 K_theta from 0.25 in
    (K_theta = 1 along the grain). joinery reads it from here."""
    from src.voxel import joinery as J
    inch = rd.INCH
    assert float(rd.dowel_Rd(0.10 * inch)) == 2.2 and float(rd.dowel_Rd(0.17 * inch)) == 2.2
    assert float(rd.dowel_Rd(0.20 * inch)) == pytest.approx(2.5)
    assert float(rd.dowel_Rd(0.25 * inch)) == 3.2 and float(rd.dowel_Rd(0.5 * inch)) == 3.2
    D = np.linspace(0.002, 0.012, 11)
    assert (np.diff(rd.dowel_Rd(D)) >= 0).all()               # a bigger dowel never gets a smaller Rd
    for G, Dmm in ((0.45, 4.83), (0.6, 3.5), (0.35, 6.0), (0.5, 8.0)):
        want = (Dmm * 1e-3) ** 2 / float(rd.dowel_Rd(Dmm * 1e-3)) * math.sqrt(
            2.0 * (16600.0 * 6894.757293168) * G ** 1.84 * 600e6 / 3.0)       # Fe = 16600 G^1.84 psi
        assert J.dowel_lateral_yield_standin(G, Dmm) == pytest.approx(want, rel=1e-12)


def test_codeonly_species_and_join_tables_are_complete():
    assert len(rd.SP_NAMES) == rd.NS == len(rd.RHO) == 10 and set(rd.MATERIALS) == set(rd.SP_NAMES)
    assert rd.DIFF.keys() == rd.SORPTION.keys()
    # the sorption isotherm is monotone and the equilibrium content is data, not a fraction of the air's
    rh = np.linspace(0.0, 0.95, 20)
    for m in rd.SORPTION:
        x = rd.eq_moisture(rh, m)
        assert (np.diff(x) > 0).all() and x[0] == 0.0
    assert "nail" in rd.JOIN_KINDS and rd.JOIN_KINDS["nail"]["cure"]["law"] == "none"
    assert rd.JOIN_KINDS["pva"]["cure"]["law"] == "dry" and len(rd.JOIN_KIND_NAMES) == len(rd.JOIN_KINDS)
    # a mixed material's table is the mass-weighted sum of the parts (energy-exact mixing)
    w = np.zeros(rd.NS)
    w[rd.SP["water"]], w[rd.SP["hide"]] = 0.5, 0.5
    mix = rd.mix_material(w)
    T = np.linspace(-30, 200, 60)
    assert np.allclose(mix.H_of_T(T), 0.5 * rd.MATERIALS["water"].H_of_T(T) + 0.5 * rd.MATERIALS["hide"].H_of_T(T),
                       rtol=0, atol=1e-6)
    assert mix.rho == pytest.approx(1.0 / (0.5 / 998.0 + 0.5 / 950.0))
