"""TIER 1: the thermal network of a region (wave 13, T5, spec 4.10 and 12
T31-T37). The lattice's own heat law is coarse-grained into an rc_net; the jump
is exact in HOW MUCH heat and approximate in WHERE, and it refuses where it
cannot be trusted. Comparisons against the tick law are bounded, and the bound
is a stated number (u), not a promise: see each test.

@pytest.mark.slow tests run minutes of ticks; they go under flock + cap.sh.
"""
import numpy as np
import pytest

from src.voxel import quiet as Q
from src.voxel import slowforms as F
from src.voxel import slownet as N
from src.voxel import slowtime as ST
from src.voxel import slowtwins as TW
from src.voxel.materials import AIR, AMBIENT, IRON, STONE, TICK_S, WATER, WOOD
from src.voxel.scenes import _person
from src.voxel.sim import World

HOUR = 144000


class SlowWorld(World):
    slow = None
    heater = None                   # (cells (n,3), watts): a TEST-ONLY source added each tick

    def _tick_start(self):
        super()._tick_start()
        if self.slow is not None:
            self.slow.pre(self)
        if self.heater is not None:
            cells, watts = self.heater
            ix = tuple(np.asarray(cells).T)
            self.E[ix] += np.float32(watts * TICK_S / len(cells))


def _room(nx=20, ny=20, nz=16, vc=5):
    w = SlowWorld(nx, ny, nz, voxel_cm=vc)
    w.fill(0, nx, 0, ny, 0, 1, STONE)
    w.fill(0, nx, 0, 1, 0, nz, STONE)
    w.fill(0, nx, ny - 1, ny, 0, nz, STONE)
    w.fill(0, 1, 0, ny, 0, nz, STONE)
    w.fill(nx - 1, nx, 0, ny, 0, nz, STONE)
    w.fill(0, nx, 0, ny, nz - 1, nz, STONE)
    w.exits = []
    return w


def _hearth(T0=60.0, k=5, dims=(20, 20, 16)):
    w = _room(*dims)
    w.fill(7, 7 + k, 7, 7 + k, 1, 1 + k, STONE)
    for _ in range(3):
        w.step()
    C = w.heat_capacity()
    sl = (slice(7, 7 + k), slice(7, 7 + k), slice(1, 1 + k))
    w.E[sl] = (T0 - AMBIENT) * C[sl]
    return w


def _block_mean(w, k=5):
    sl = (slice(7, 7 + k), slice(7, 7 + k), slice(1, 1 + k))
    return float(AMBIENT + w.E[sl].sum() / w.heat_capacity()[sl].sum())


# ──────────────────────────────────────── T31 the net is the lattice ──
def test_net_capacity_and_stored_energy_sum_to_the_lattice_totals():
    w = _hearth()
    res = N.thermal_net(w, max_nodes=10000)
    box = res.box
    assert res.B == 1 and res.refusals == []
    assert abs(res.Cn.sum() - float(w.heat_capacity()[box].sum())) < 1e-6 * res.Cn.sum()
    assert abs(float((res.Cn * (res.T0 - AMBIENT)).sum()) - float(w.E[box].sum(dtype=np.float64))) < 1e-3
    # a coarser net (blocks of 2) still holds the same capacity and energy
    res2 = N.thermal_net(w, max_nodes=300)
    assert res2.B >= 2 and res2.spec.n < res.spec.n / 3
    assert abs(res2.Cn.sum() - res.Cn.sum()) < 1e-6 * res.Cn.sum()
    assert abs(float((res2.Cn * (res2.T0 - AMBIENT)).sum()) - float(w.E[box].sum(dtype=np.float64))) < 1e-3
    # the links are symmetric, positive and between different nodes
    s = res.spec
    ge = s.at(res.T0, AMBIENT).g                         # the temperature-dependent links read at the start
    assert (ge > 0).all() and (s.g >= 0).all() and (s.rows < s.cols).all() and s.rad.min() >= 0.0
    assert s.Gb.min() >= 0.0


def test_a_single_exposed_face_has_the_heat_laws_own_conductance():
    """Solid | gas is (H_CONV + radiation of hot gas) A af af; here the gas is cool,
    so it is H_CONV A. Solid | solid is the harmonic k times dx."""
    from src.voxel.materials import H_CONV, SOLID
    w = SlowWorld(6, 6, 6, voxel_cm=5)
    w.fill(2, 3, 2, 3, 2, 3, STONE)
    w.fill(3, 4, 2, 3, 2, 3, IRON)
    w.step()
    faces, a_i, u, T = N.face_arrays(w, (slice(0, 6),) * 3)
    dx = 0.05
    ids = np.arange(216).reshape(6, 6, 6)
    stone, iron, air = ids[2, 2, 2], ids[3, 2, 2], ids[2, 2, 3]
    got = {}
    for ia, ib, G in faces:
        for a, b, g in zip(ia, ib, G):
            got[(int(a), int(b))] = float(g)
    kS, kI = SOLID[STONE][2], SOLID[IRON][2]
    assert abs(got[(int(stone), int(iron))] - 2 * kS * kI / (kS + kI) * dx) < 1e-9
    fill_s = float(np.sqrt(np.clip(SOLID[STONE][0] * w.vox_l / (SOLID[STONE][0] * w.vox_l), 0, 1)))
    assert abs(got[(int(stone), int(air))] - H_CONV * dx * dx * fill_s * 1.0) < 1e-9
    # radiation to the universe: e sigma A af n_gas, 5 gas faces of the stone cell
    from src.voxel.materials import SIGMA
    assert abs(a_i.ravel()[stone] - 0.9 * SIGMA * dx * dx * 5) < 1e-15


# ───────────────────────────────────────────────── T32 write-back ──
def test_write_back_adds_c_dT_to_every_node_exactly_in_energy():
    w = _hearth()
    res = N.thermal_net(w, max_nodes=10000)
    rng = np.random.default_rng(11)
    dT = rng.normal(0.0, 2.0, res.spec.n)
    e_before = w.E[res.box].astype(np.float64).copy()
    stats = N.write_back(w, res, res.T0 + dT, snap=False)
    de = w.E[res.box].astype(np.float64) - e_before
    per_node = np.bincount(res.node_of.ravel(), de.ravel(), res.spec.n)
    want = res.Cn * dT
    assert float(np.abs(per_node - want).max()) < 1e-3 * (1.0 + float(np.abs(want).max()) * 1e-4)
    assert abs(stats["expected"] - float(want.sum())) < 1e-6 * (1 + abs(float(want.sum())))
    assert abs(stats["actual"] - stats["expected"]) < 0.05              # float32 cells
    assert abs(stats["rounding"]) < 0.05
    # the sub-node shape survives: two cells of one node keep their difference
    assert float(np.abs(de - (res.cell_C * dT[res.node_of])).max()) < 5e-3


def test_snapping_books_the_residue_into_shed_and_leaves_warming_cells_alone():
    w = _hearth(T0=20.02)
    res = N.thermal_net(w, max_nodes=10000)
    # the block is 20 mK warm and cools to 4 mK: inside the 10 mK settle row
    T1 = np.where(res.T0 > AMBIENT + 0.01, AMBIENT + 0.004, res.T0)
    shed0 = w.shed
    box = res.box
    stats = N.write_back(w, res, T1, snap=True)
    # no source and a steady state at the boundary: every cooling cell within 0.01 K is exactly 0
    cooled = w.E[box] == 0.0
    assert stats["cells_snapped"] > 0
    assert stats["snapped"] > 0.0 and abs(w.shed - shed0 - stats["snapped"]) < 1e-9
    # energy is conserved: E + shed is what it was plus the (exact) write-back
    # cells that were warming (dT > 0) are not snapped: the same call with T1 below T0
    w2 = _hearth(T0=20.0)
    r2 = N.thermal_net(w2, max_nodes=10000)
    assert float(np.abs(w2.E).max()) == 0.0
    s2 = N.write_back(w2, r2, r2.T0 + 0.004, snap=True)
    assert s2["cells_snapped"] == 0                    # everything is warming: nothing is snapped


def test_tier_0_resumes_after_a_cooled_hearth():
    w = _hearth(T0=20.4, dims=(14, 14, 12))
    sch = ST.Scheduler()
    w.slow = sch
    assert not Q.quiet(w, 0).ok and Q.quiet(w, 1).ok
    rep = ST.advance(w, seconds=60 * 86400.0, approx=True, sched=sch)
    assert rep.tier_used == "reduced" and rep.jumped > 1_000_000 and rep.halted_by == "target"
    assert int(np.count_nonzero(w.E)) == 0, float(np.abs(w.E).max())
    q = Q.quiet(w, 0)
    assert q.ok                                            # E is 0 again: the exact tier is back
    assert int(np.count_nonzero(w.E)) == 0                  # and stays at zero for the exact tier
    rep0 = ST.advance(w, seconds=3600.0, sched=sch)        # now the exact tier jumps
    assert rep0.tier_used == "exact" and rep0.jumped > 100000


# ───────────────────────────────────────────────── T33 refusals ──
def _refusal_codes(w, **kw):
    res = N.thermal_net(w, max_nodes=10000, **kw)
    return {c for c, _p, _w in N.refusals(w, res)}


def test_refuses_when_an_ignition_row_lies_inside_the_trajectory():
    w = _room()
    w.fill(8, 10, 8, 10, 1, 3, WOOD, frac=0.8)
    w.step()
    C = w.heat_capacity()
    w.E[8:10, 8:10, 1:3] = (350.0 - AMBIENT) * C[8:10, 8:10, 1:3]
    assert "threshold" in _refusal_codes(w)
    why = [r for r in N.refusals(w, N.thermal_net(w, max_nodes=10000)) if r[0] == "threshold"][0]
    assert "ignition" in why[2] and 8 <= why[1][0] < 10
    # the same wood cooler than its ignition row never crosses it
    w2 = _room()
    w2.fill(8, 10, 8, 10, 1, 3, WOOD, frac=0.8)
    w2.step()
    C2 = w2.heat_capacity()
    w2.E[8:10, 8:10, 1:3] = (200.0 - AMBIENT) * C2[8:10, 8:10, 1:3]
    assert "threshold" not in _refusal_codes(w2)


def test_refuses_when_a_source_would_boil_the_water():
    w = _room()
    for x in range(6, 12):
        for y in range(6, 12):
            w.pour(x, y, 1, WATER, 600.0)
    for _ in range(3):
        w.step()
    C = w.heat_capacity()
    w.E[6:12, 6:12, 1] = (95.0 - AMBIENT) * C[6:12, 6:12, 1]
    cells = np.argwhere(w.E != 0)
    codes = _refusal_codes(w, sources=[(cells, 2000.0)])
    assert "threshold" in codes
    res = N.thermal_net(w, max_nodes=10000, sources=[(cells, 2000.0)])
    assert any("water boils" in r[2] for r in N.refusals(w, res))


def test_refuses_a_small_hot_stone_by_its_radiative_biot_and_accepts_a_layered_one():
    """A 5 cm stone at 500 K: h = 10 + 4 e sigma T^3 = 35.5 W/m2K, L = V/A = 8 mm,
    Bi = 0.15 > 0.1: refused as a lump. A 30 cm block resolves three layers
    through its thickness: accepted."""
    def world(k, T_c):
        w = _room(24, 24, 24)
        w.mat[8:8 + k, 8:8 + k, 3:3 + k] = STONE            # floating: only the net is built
        w.smass[8:8 + k, 8:8 + k, 3:3 + k] = 2700.0 * w.vox_l
        w.step()
        C = w.heat_capacity()
        sl = (slice(8, 8 + k), slice(8, 8 + k), slice(3, 3 + k))
        w.E[sl] = (T_c - AMBIENT) * C[sl]
        return w
    w1 = world(1, 227.0)
    res = N.thermal_net(w1, max_nodes=10000)
    bi = [r for r in N.refusals(w1, res) if r[0] == "biot" and r[1] == (8, 8, 3)]
    assert bi and "Biot 0.1" in bi[0][2] and "1 layer" in bi[0][2]
    # (the cool stone floor of the room is a one-cell plate too, but nothing moves there)
    assert not [r for r in N.refusals(w1, res) if r[0] == "biot" and r[1] != (8, 8, 3)]
    w6 = world(6, 227.0)
    res6 = N.thermal_net(w6, max_nodes=10000)
    assert [r for r in N.refusals(w6, res6) if r[0] == "biot"] == []
    # a cool small stone (not thermally active) is not asked
    wc = world(1, 20.3)
    assert [r for r in N.refusals(wc, N.thermal_net(wc, max_nodes=10000)) if r[0] == "biot"] == []


def test_refuses_a_region_that_hosts_a_process_needing_its_history():
    w = _hearth()
    sch = ST.Scheduler()
    hist = F.Form("quench", "STEP", lambda s, p, d, a, b: (s, {}), lambda s, p, k: s[0],
                  lambda p: (), needs_history=True)
    F.register(hist)
    try:
        dh = ST.DictHost("bar", np.linspace(0, 0.02, 21), np.full(21, 0.5), 0.02, box=(6, 6, 0, 13, 13, 7))
        sch.add_host("bar", dh)
        sch.add("quench", "shadow", ("dict", "bar"), {}, None, 0)
        res = N.thermal_net(w, max_nodes=10000)
        codes = {c for c, _p, _w in N.refusals(w, res, sch)}
        assert "history" in codes
        far = ST.DictHost("far", np.linspace(0, 0.02, 21), np.full(21, 0.5), 0.02, box=(100, 100, 100, 101, 101, 101))
        sch2 = ST.Scheduler()
        sch2.add_host("far", far)
        sch2.add("quench", "shadow", ("dict", "far"), {}, None, 0)
        assert "history" not in {c for c, _p, _w in N.refusals(w, res, sch2)}
    finally:
        F.FORMS.pop("quench", None)


def test_refuses_when_the_sleepers_would_use_too_much_of_the_rooms_oxygen():
    """4 sleepers use about 480 L of O2 in 8 h: 1.2 % of a 40 m3 room. In a
    1 m3 box that is 48 %: refused."""
    big = _room(40, 40, 40, 10)                         # 40 x 40 x 40 cells of 1 L = 64 m3... of cells
    assert N.v2_o2_ok(big, (slice(0, 40),) * 3, 480.0 * 1.429)[0]
    small = _room(10, 10, 10, 10)                       # 1 m3 of cells (8 m3 inside walls is not: 8^3 L)
    ok, why = N.v2_o2_ok(small, (slice(0, 10),) * 3, 480.0 * 1.429)
    assert not ok and "limit" in why


# ─────────────────────────────── T34 / T37 against the tick law ──
def _jump(w, secs, max_nodes=10000, **kw):
    ok, info = N.advance_region(w, int(secs / TICK_S), max_nodes=max_nodes, **kw)
    assert ok, info
    return info


def test_hearth_cool_down_for_60_s_agrees_with_ticks_within_15_percent_of_the_drop():
    """The stated bound (u): the net's drop of the block's mean temperature is
    within 15 % of the tick law's, and the heat the world sheds within 10 %."""
    a, b = _hearth(), _hearth()
    e0 = a.total_energy()
    for _ in range(int(60 / TICK_S)):
        a.step()
    info = _jump(b, 60.0)
    da, db = 60.0 - _block_mean(a), 60.0 - _block_mean(b)
    assert da > 0.2 and abs(db - da) < 0.15 * da, (da, db)
    shed_a, shed_b = a.shed, b.shed
    assert abs(shed_b - shed_a) < 0.10 * shed_a
    assert abs(b.total_energy() - e0) < 1.0                            # the net books every joule (float32 cells)
    assert info["B"] == 1


def test_a_jump_then_600_real_ticks_has_no_discontinuity():
    """T37: after the jump the THERMAL_DRIVEN fields (skin, wind) are recomputed
    by the first ticks. 60 s by net then 15 s of ticks agrees with 75 s of ticks."""
    a, b = _hearth(), _hearth()
    for _ in range(int(75 / TICK_S)):
        a.step()
    _jump(b, 60.0)
    mid = _block_mean(b)
    for _ in range(600):
        b.step()
    da, db = 60.0 - _block_mean(a), 60.0 - _block_mean(b)
    assert abs(db - da) < 0.15 * da
    assert 60.0 > mid > _block_mean(b)                                 # still falling, smoothly
    assert np.isfinite(b.E).all() and float(b.skin.max()) < 5.0


@pytest.mark.slow
def test_hearth_cool_down_for_10_minutes_agrees_within_15_percent_of_the_drop():
    a, b = _hearth(), _hearth()
    for _ in range(int(600 / TICK_S)):
        a.step()
    _jump(b, 600.0)
    da, db = 60.0 - _block_mean(a), 60.0 - _block_mean(b)
    assert abs(db - da) < 0.15 * da, (da, db)
    assert abs(b.shed - a.shed) < 0.10 * a.shed


@pytest.mark.slow
@pytest.mark.parametrize("max_nodes,bound", [(10000, 0.05), (6000, 0.15)])
def test_constant_source_room_agrees_with_ticks_within_15_percent_of_the_rise(max_nodes, bound):
    """T36: a 300 W stand-in for four sleepers (an iron block, so the heat is
    one body) in a closed stone room. Mean air temperature and the heater's own
    rise after 5 minutes of ticks, against the net; the total energy is exact.
    One cell per node (8,712 nodes) is within 5 %; blocks of two cells (about
    1,600 nodes, the default size) within 15 % (u: measured 0.1 % and 9 %).
    The net holds the whole room: the plume crosses it in seconds."""
    def build():
        w = _room(22, 22, 18)
        w.fill(9, 13, 9, 11, 1, 6, IRON)
        for _ in range(3):
            w.step()
        cells = np.argwhere(w.mat[9:13, 9:11, 1:6] == IRON) + np.array([9, 9, 1])
        return w, cells
    a, ca = build()
    b, cb = build()
    a.heater = (ca, 300.0)
    secs = 300.0
    for _ in range(int(secs / TICK_S)):
        a.step()
    info = _jump(b, secs, max_nodes=max_nodes, sources=[(cb, 300.0)])
    air = lambda w: float(np.mean(w.T()[(w.mat == AIR)]))              # noqa: E731
    sl = (slice(9, 13), slice(9, 11), slice(1, 6))
    heat = lambda w: float(w.T()[sl].mean())                           # noqa: E731
    ra, rb = air(a) - AMBIENT, air(b) - AMBIENT
    ha, hb = heat(a) - AMBIENT, heat(b) - AMBIENT
    assert ha > 3.0 and ra > 0.001
    assert abs(hb - ha) < bound * ha, (ha, hb)
    assert abs(rb - ra) < bound * ra, (ra, rb)
    assert info["B"] == (1 if max_nodes == 10000 else 2)
    assert abs(b.total_energy() - (300.0 * secs)) < 1.0                # exact: source minus nothing lost
    assert abs(info["source_J"] - 300.0 * secs) < 1e-6


# ───────────────────── the air in the net: the plume and the room ──
def test_the_net_reads_the_lattices_own_heat_rates_in_a_plume():
    """No tuned number in the air: at a state with a developed plume, the net
    (gas | gas eddies, surface | gas, buoyant rise, all read at the node
    temperatures) gives each layer of air the heat rate the tick law gives it
    over the next second, within 8 % of the total |rate| (u: 3 % measured)."""
    w = _room(22, 22, 18)
    w.fill(9, 13, 9, 11, 1, 6, IRON)
    for _ in range(3):
        w.step()
    cells = np.argwhere(w.mat[9:13, 9:11, 1:6] == IRON) + np.array([9, 9, 1])
    w.heater = (cells, 300.0)
    for _ in range(int(40.0 / TICK_S)):
        w.step()
    res = N.thermal_net(w, max_nodes=20000)
    assert res.B == 1 and res.spec.dyn is not None
    e0 = w.E[res.box].astype(np.float64).copy()
    w.heater = None
    nt = 40
    for _ in range(nt):
        w.step()
    tick = (w.E[res.box].astype(np.float64) - e0) / (nt * TICK_S)      # W, includes the radiated loss
    sp = res.spec.at(res.T0, AMBIENT)
    net = -sp.matvec_lap(res.T0) + sp.Gb * (AMBIENT - res.T0)
    air = (w.mat[res.box] == AIR)
    nd = res.node_of
    layers_t = np.array([tick[:, :, z][air[:, :, z]].sum() for z in range(1, 17)])
    layers_n = np.array([net[nd[:, :, z][air[:, :, z]]].sum() for z in range(1, 17)])
    assert layers_t[:12].min() > 0.0 and layers_n[:12].min() > 0.0      # heat reaches high up
    assert np.abs(layers_n - layers_t).sum() < 0.08 * np.abs(layers_t).sum(), (layers_t, layers_n)


def test_the_region_grows_to_the_closed_room_and_not_past_a_cap(monkeypatch):
    w = _room(22, 22, 18)
    w.fill(9, 13, 9, 11, 1, 6, IRON)
    for _ in range(3):
        w.step()
    cells = np.argwhere(w.mat[9:13, 9:11, 1:6] == IRON) + np.array([9, 9, 1])
    box = N.region_for(w, [(cells, 300.0)], 300.0)
    assert [(s.start, s.stop) for s in box] == [(0, 22), (0, 22), (0, 18)]
    monkeypatch.setattr(N, "GAS_CELLS_MAX", 100)                       # a room too big to take whole
    small = N.region_for(w, [(cells, 300.0)], 300.0)
    assert all(s.stop - s.start < 22 for s in small[:2])
    assert N.gas_reach(w, small, cap=100) is None


def test_a_plume_that_would_leave_the_box_is_refused_not_lost(monkeypatch):
    """The ring round the box is held at the boundary value. A jump whose heat
    would walk into the air of that ring says so and ticks instead."""
    w = _room(22, 22, 18)
    w.fill(9, 13, 9, 11, 1, 6, IRON)
    for _ in range(3):
        w.step()
    cells = np.argwhere(w.mat[9:13, 9:11, 1:6] == IRON) + np.array([9, 9, 1])
    monkeypatch.setattr(N, "GAS_CELLS_MAX", 100)
    ok, info = N.advance_region(w, int(300.0 / TICK_S), sources=[(cells, 300.0)], max_nodes=10000)
    assert not ok and any(r[0] == "escape" for r in info["refusals"]), info


# ─────────────────────────────────────────── tier 1 in the detector ──
def test_tier_1_owns_the_hot_window_and_tier_0_does_not():
    w = _hearth(dims=(14, 14, 12))
    q0, q1 = Q.quiet(w, 0), Q.quiet(w, 1)
    assert not q0.ok and any(b.reason.startswith("E != 0") for b in q0.blockers)
    assert q1.ok and q1.owned and q1.owned[0][0].start == 0
    C = w.heat_capacity()
    w.E[5:7, 5:7, 2:4] += 600.0 * C[5:7, 5:7, 2:4]                     # a flame is still a blocker
    w.fill(5, 7, 5, 7, 2, 4, WOOD)
    C = w.heat_capacity()
    w.E[5:7, 5:7, 2:4] = (500.0 - AMBIENT) * C[5:7, 5:7, 2:4]
    assert not Q.quiet(w, 1).ok


def test_advance_without_approx_ticks_a_hot_world_and_with_approx_jumps_it():
    a = _hearth(dims=(14, 14, 12))
    sch = ST.Scheduler()
    a.slow = sch
    rep = ST.advance(a, 1500, sched=sch)                               # tier 0: the hot block is a blocker
    assert rep.jumped == 0 and rep.stepped == 1500
    b = _hearth(dims=(14, 14, 12))
    sch2 = ST.Scheduler()
    b.slow = sch2
    t0 = b.tick
    rep = ST.advance(b, 6000, approx=True, sched=sch2)
    assert rep.tier_used == "reduced" and rep.jumped > 4000 and rep.t_to == t0 + 6000
    assert _block_mean(b) < 60.0


def test_sleepers_warm_a_sealed_room_by_exactly_their_watts_and_use_its_oxygen():
    """The const-source twin stands in for body_heat: 4 sleepers of 80 W for 2 h
    in a sealed room. The energy the world holds rises by 4 x 80 W x t to float
    rounding (the net books every joule); the twins' fluxes go to the sinks."""
    tw = TW.register_twin("const_source", TW.ConstSourceTwin(watts=80.0, o2_l_min=0.25))
    try:
        w = _room(30, 24, 40)
        sch = ST.Scheduler()
        w.slow = sch
        for k, (x, y) in enumerate(((6, 6), (6, 15), (15, 6), (15, 15))):
            p = _person(w, x, y, z0=1)
            p["awake"], p["asleep"], p["blood_o2"] = True, True, 0.95
        for _ in range(60):
            w.step()
        e0 = w.total_energy()
        secs = 2 * 3600.0
        t0 = w.tick
        rep = ST.advance(w, seconds=secs, approx=True, sched=sch)
        assert rep.tier_used == "reduced" and rep.jumped > 0.9 * secs / TICK_S and rep.t_to == t0 + int(secs / TICK_S)
        gained = w.total_energy() - e0
        assert abs(gained - 4 * 80.0 * secs) < 0.02 * 4 * 80.0 * secs       # float32 cells, 2 %: the lattice drifts
        assert sch.sinks[("E", "source", "twin")] < 0.0
        o2 = -sch.sinks[("mass", "o2", "room")]
        assert abs(o2 - 4 * 0.25 / 60.0 * secs * 1.429) < 0.02 * o2 + 1.0
        assert float(np.mean(w.T()[w.mat == AIR])) > AMBIENT + 0.001
        # the persons are alive and the twin keys moved by the breath twin as well
        assert all(p["alive"] for p in w.persons)
    finally:
        TW.unregister_twin("const_source")


def test_a_refused_tier_1_jump_falls_back_to_ticking_and_says_why():
    tw = TW.register_twin("const_source", TW.ConstSourceTwin(watts=80.0, o2_l_min=40.0))
    try:
        w = _room(12, 12, 36)                          # a small sealed room
        sch = ST.Scheduler()
        w.slow = sch
        p = _person(w, 5, 5, z0=1)
        p["awake"], p["asleep"] = True, True
        for _ in range(60):
            w.step()
        rep = ST.advance(w, seconds=8 * 3600.0, approx=True, max_fine_ticks=1700, sched=sch)
        assert rep.halted_by == "blocked" and rep.jumped == 0
        assert any(g == "thermal" and "O2" in r for g, _p, r in rep.blockers_seen) or \
            any("O2" in r for _g, _p, r in rep.blockers_seen)
    finally:
        TW.unregister_twin("const_source")
