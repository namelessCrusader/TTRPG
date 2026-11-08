"""THE GAS FLOW (src/voxel/gasflow.py; docs/realism-map.md, section 6, lane 7).

Hot air rises, and takes its smoke and its heat with it. The old laws did
that with stand-ins (a share of heat handed up, smoke "rising hard", a stir
toward a zone mean); the flow does it with a pressure solve and a carried
flux, and these tests hold it against what rooms really do:

  * a hot layer builds under a ceiling, and a burning room is hotter at the
    top than at the floor (the thing the old laws got backwards);
  * the smoke layer's interface falls the way Zukoski's plume says it does;
  * the ceiling jet under an unconfined ceiling is Alpert's;
  * a doorway has a neutral plane, hot out above it, cool in below;
  * heat, smoke and oxygen are carried EXACTLY: what leaves one voxel arrives
    in the next, and what leaves the world is booked;
  * the compiled path is the numpy one, to the bit;
  * the switch is off by default and, off, changes nothing.

Rooms here are 5 cm voxels (20 cm cells). Heat is put in by hand (a power
into a patch of floor voxels, with a trace of smoke) so the test is about the
GAS and not about a fire's chemistry.
"""
import numpy as np
import pytest

from src.voxel import gasflow, heat
from src.voxel.gasflow import _flux, _gf_apply_nb, _gf_flux_nb, face_open
from src.voxel.materials import AIR, AMBIENT, O2_PER_L, OIL, STONE, TICK_S, WOOD
from src.voxel.sim import World

DX = 0.05                                            # m, a voxel here


@pytest.fixture(autouse=True)
def _no_eddy(monkeypatch):
    """The heat law's EDDY is a stand-in for the very convection the flow now
    carries (a gas-to-gas conductivity that grows with the excess temperature,
    0.01 m^2/s at 100 K: it smears a layer in seconds). With the flow as the
    law of the air it has to go (see the lane's report); until then these
    tests turn it off, so they hold with or without that edit."""
    monkeypatch.setattr(heat, "EDDY", 1e12)


def _box(n, nz=None, flow=True, fused=True):
    """A sealed stone room, n voxels a side inside, 5 cm voxels."""
    nz = nz or n
    w = World(n + 2, n + 2, nz + 2, voxel_cm=5)
    w.open_sky = False
    w.fill(0, n + 2, 0, n + 2, 0, nz + 2, STONE)
    w.mat[1:n + 1, 1:n + 1, 1:nz + 1] = AIR
    w.smass[1:n + 1, 1:n + 1, 1:nz + 1] = 0
    w.fused = fused
    w.gas_flow = flow
    return w


def _burn(w, watts, x, y, z=1, smoke=0.0):
    """Put `watts` into a 20 cm cube of voxels for one tick (a flame is a
    volume: the same power in one layer of voxels is a 3000 K sheet that
    radiates into the floor), with `smoke` grams of soot."""
    w.E[x:x + 4, y:y + 4, z:z + 4] += watts * TICK_S / 64
    if smoke:
        w.smoke[x:x + 4, y:y + 4, z:z + 4] += smoke / 64


def _layer_height(smoke, n, skip=None):
    """The smoke layer's interface, from the two-layer fit that keeps both the
    smoke and its square (Cetegen): thickness = (sum s)^2 / sum s^2, with the
    column sums s taken over the room outside the plume."""
    m = np.ones((n, n), bool)
    if skip:
        m[skip[0]:skip[1], skip[0]:skip[1]] = False
    col = smoke[1:n + 1, 1:n + 1, 1:n + 1][m].sum(axis=0).astype(float)
    return (n - col.sum() ** 2 / (col ** 2).sum()) * DX


# ── the switch ───────────────────────────────────────────────────────────────
def test_the_SWITCH_is_OFF_by_default_and_OFF_leaves_the_old_laws_alone():
    w = World(30, 30, 30, voxel_cm=5)
    assert w.gas_flow is False
    assert "_law_rise" not in w.__dict__                  # the heat law is the world's
    w.gas_flow = True
    assert w.__dict__["_law_rise"] is not None            # put aside while on
    w.gas_flow = False
    assert "_law_rise" not in w.__dict__ and w.gas_flow is False
    # a world switched on and off again is the world that never was
    a, b = World(30, 30, 30, voxel_cm=5), World(30, 30, 30, voxel_cm=5)
    for w_ in (a, b):
        w_.fill(0, 30, 0, 30, 0, 1, STONE)
    b.gas_flow = True
    b.gas_flow = False
    for t in range(12):
        for w_ in (a, b):
            _burn(w_, 800.0, 10, 10, smoke=0.016)
            w_.step()
    assert np.array_equal(a.E, b.E) and np.array_equal(a.smoke, b.smoke)
    assert np.array_equal(a.o2, b.o2)


# ── the geometry: faces read through the voxels ──────────────────────────────
def test_a_WALL_ONE_VOXEL_THICK_inside_a_cell_shuts_the_face_the_cell_mean_would_leave_open():
    por = np.ones((16, 8, 8), np.float32)
    por[5, :, :] = 0.0                                    # a wall, 5 cm thick,
    F = face_open(por, 4, False)                          # in the middle of a cell
    assert F[0].shape == (5, 2, 2)
    assert np.all(F[0][2] == 1.0) and np.all(F[0][3] == 1.0)
    assert np.all(F[0][1] == 0.0)                         # the face it sits in: SHUT
    assert np.all(F[0][0] == 0.0) and np.all(F[0][-1] == 0.0)   # a sealed world's edge
    # (the mean porosity of the cell it is in is 0.75: the coarse read calls
    # that open)
    assert float(por[4:8].mean()) == 0.75
    por[5, 0:4, 0:4] = 1.0                                # a doorway, one cell square
    G = face_open(por, 4, False)
    assert G[0][1][0, 0] == 1.0 and G[0][1][1, 1] == 0.0
    por[5, 4:8, 4:8] = 0.03                               # a leaky door: 3% of the pores
    H = face_open(por, 4, False)
    assert abs(H[0][1][1, 1] - 4 / (3 + 1 / 0.03)) < 1e-6   # one lane of four voxels


def test_an_OPEN_SKY_opens_the_edge_but_never_the_ground():
    por = np.ones((8, 8, 8), np.float32)
    F = face_open(por, 4, True)
    assert np.all(F[0][0] == 1.0) and np.all(F[0][-1] == 1.0)
    assert np.all(F[2][:, :, 0] == 0.0) and np.all(F[2][:, :, -1] == 1.0)


# ── the solve ────────────────────────────────────────────────────────────────
def test_the_PROJECTION_leaves_no_cell_with_more_gas_than_it_started_with():
    """A hot bubble in a sealed room, the flow run alone: the speeds it solves
    for are free of divergence to the solver's tolerance, they are not small,
    and the solve takes a few tens of iterations."""
    w = _box(24)
    w.o2 = np.full(w.shape, O2_PER_L * w.vox_l, np.float32)
    w.E[6:14, 6:14, 1:5] += 8.0                           # 50 K or so, at the floor
    for t in range(30):
        w.tick = t
        w._law_gasflow()
    st = w._gf
    assert w.gas_flow_stats["res"] <= gasflow.GAS_P_TOL * 1.01
    assert w.gas_flow_stats["iters"] < gasflow.GAS_P_ITERS
    F = w._gf_geometry()["F"]
    A = (4 * DX) ** 2
    div = 0.0
    for ax in range(3):
        fl = A * F[ax] * st["u"][ax]
        lo = tuple(slice(None, -1) if a == ax else slice(None) for a in range(3))
        hi = tuple(slice(1, None) if a == ax else slice(None) for a in range(3))
        div = div + fl[hi] - fl[lo]
    speed = max(float(np.abs(u).max()) for u in st["u"])
    assert speed > 0.1                                    # it moves...
    assert float(np.abs(div).max()) < 1e-3 * speed * A    # ...and keeps the gas it has


# ── hot air goes UP ──────────────────────────────────────────────────────────
def test_a_hot_room_is_HOTTER_AT_THE_TOP_and_cool_at_the_floor():
    w = _box(32)
    for t in range(int(10 / TICK_S)):
        _burn(w, 5000.0, 16, 16, smoke=0.008)
        w.step()
    T = w.T()
    far = (slice(2, 10), slice(2, 10))                    # away from the plume
    top = float(T[far + (slice(28, 32),)].mean())
    floor = float(T[far + (slice(2, 6),)].mean())
    assert top > floor + 8.0, (top, floor)
    # the plume itself: warmest at its foot, still warm at the ceiling
    assert T[15:17, 15:17, 30].mean() > AMBIENT + 15.0


def test_there_is_NO_LIFT_without_heat():
    """Smoke in a room at ambient temperature goes nowhere on its own (it
    seeps, a little): there is no wind to carry it."""
    w = _box(24)
    w.smoke[10:14, 10:14, 4:8] = 0.5
    z0 = float((w.smoke[1:25, 1:25, 1:25].sum((0, 1)) * np.arange(1, 25)).sum()
               / w.smoke.sum())
    for t in range(80):
        w.step()
    z1 = float((w.smoke[1:25, 1:25, 1:25].sum((0, 1)) * np.arange(1, 25)).sum()
               / w.smoke.sum())
    assert abs(z1 - z0) < 0.5                             # voxels
    assert not w._gf_live()


# ── exact carrying ───────────────────────────────────────────────────────────
def test_in_a_SEALED_ROOM_heat_smoke_and_oxygen_are_carried_EXACTLY():
    """The flow is one law; run alone, it moves heat, smoke and the oxygen
    deficit around a closed room and gains and loses none of them."""
    w = _box(28)
    fresh = np.float32(O2_PER_L * w.vox_l)
    w.o2 = np.full(w.shape, fresh, np.float32)
    rng = np.random.default_rng(5)
    w.smoke[8:20, 8:20, 2:6] = rng.random((12, 12, 4)).astype(np.float32)
    w.o2[8:20, 8:20, 2:10] -= (rng.random((12, 12, 8)) * 0.05).astype(np.float32)
    w.E[8:20, 8:20, 1:3] += rng.random((12, 12, 2)).astype(np.float32) * 30.0
    e0 = float(w.E.sum(dtype=np.float64))
    s0 = float(w.smoke.sum(dtype=np.float64))
    d0 = float((fresh - w.o2).sum(dtype=np.float64))
    for t in range(60):
        w.tick = t
        w._law_gasflow()
    assert float(np.abs(w._gf["u"][2]).max()) > 0.2        # it really moved
    assert abs(float(w.E.sum(dtype=np.float64)) + w.shed - e0) < 1e-6 * e0
    assert abs(float(w.smoke.sum(dtype=np.float64)) - s0) < 1e-6 * s0
    d1 = float((fresh - w.o2).sum(dtype=np.float64))
    assert abs(d1 - d0) < 1e-5 * d0
    # smoke was neither made nor lost, and it went up
    zs = (w.smoke.sum((0, 1)) * np.arange(w.shape[2])).sum() / w.smoke.sum()
    assert zs > 3.5


def test_HEAT_RIDES_THROUGH_THE_PORES_of_a_solid_that_has_them_and_not_through_one_that_has_none():
    """A lid across a sealed room, a hot patch under it. Half-wood planks (half
    their space is air) let the hot air through, and the air gives its heat to
    the wood it passes (the air leaves a plank at the plank's temperature, and
    a plank is a lot of thermal mass, so most of it stays there); a lid of stone
    lets nothing through and is never warmed. The books balance either way:
    what the planks are warmed by comes out of the air. (A lid this thick keeps
    what it takes: it is a lot more thermal mass than the air it is warmed by.)"""
    seen = {}
    for name, mat, frac in (("planks", WOOD, 0.5), ("stone", STONE, 1.0)):
        w = _box(16)
        w.fill(1, 17, 1, 17, 6, 8, mat, frac=frac)
        w.o2 = np.full(w.shape, O2_PER_L * w.vox_l, np.float32)
        w.E[6:10, 6:10, 1:4] += 6.0                       # 40 K or so of air under the lid
        e0 = float(w.E.sum(dtype=np.float64))
        went = 0.0
        for t in range(400):
            w.tick = t
            w._law_gasflow()
            went = max(went, float(w._gf["u"][2][1:3, 1:3, 1:3].max()))
        seen[name] = (float(w.E[:, :, 6:8].sum(dtype=np.float64)) / e0,      # the lid
                      float(w.E[:, :, 8:].sum(dtype=np.float64)) / e0)       # over it
        assert abs(float(w.E.sum(dtype=np.float64)) + w.shed - e0) < 1e-5 * e0, name
        if name == "planks":
            assert went > 0.05                            # air really goes up through
    assert seen["planks"][0] > 0.5, seen                  # the wood took most of it (0.9 in 10 s)
    assert seen["stone"] == (0.0, 0.0), seen


def test_what_LEAVES_THROUGH_THE_SKY_is_BOOKED_heat_into_shed_smoke_into_gone():
    w = World(30, 30, 40, voxel_cm=5)
    w.fill(0, 30, 0, 30, 0, 1, STONE)
    w.gas_flow = True
    w.o2 = np.full(w.shape, O2_PER_L * w.vox_l, np.float32)
    rng = np.random.default_rng(9)
    w.smoke[10:20, 10:20, 20:30] = rng.random((10, 10, 10)).astype(np.float32)
    w.E[10:20, 10:20, 20:30] += rng.random((10, 10, 10)).astype(np.float32) * 40.0
    e0 = float(w.E.sum(dtype=np.float64))
    s0 = float(w.smoke.sum(dtype=np.float64))
    for t in range(90):
        w.tick = t
        w._law_gasflow()
    assert w.shed > 0.05 * e0 and w.gone["smoke"] > 0.05 * s0     # a lot left
    assert abs(float(w.E.sum(dtype=np.float64)) + w.shed - e0) < 1e-5 * e0
    left = s0 - float(w.smoke.sum(dtype=np.float64))
    assert abs(left - w.gone["smoke"]) < 1e-5 * s0


# ── the carrying scheme ──────────────────────────────────────────────────────
def test_the_LIMITED_SCHEME_makes_no_new_extremes_loses_nothing_and_keeps_an_edge():
    """A top hat of smoke carried 16 voxels at a Courant number of 0.4: it
    never goes above its top or below zero, none of it is lost, and its edge is
    a handful of voxels wide (a first-order upwind scheme smears it over twice
    that)."""
    n = 60
    f = np.zeros((n, 3, 3), np.float32)
    f[20:30] = 1.0
    s = np.full((n + 1, 3, 3), 0.4, np.float32)
    s[0] = s[-1] = 0.0
    ok = np.ones(s.shape, bool)
    for _ in range(40):
        f = f - np.diff(_flux(f, s, ok, 0), axis=0)
    assert f.min() >= 0.0 and f.max() <= 1.0 + 1e-6
    assert abs(float(f.sum()) - 90.0) < 1e-3
    edge = int(((f[:, 0, 0] > 0.1) & (f[:, 0, 0] < 0.9)).sum())
    assert edge <= 10


def test_the_COMPILED_KERNELS_equal_the_NUMPY_FLUX_on_random_faces_to_the_bit():
    rng = np.random.default_rng(3)
    shp, nsub = (9, 8, 7), 3
    inv = np.float32(1 / nsub)
    f = (rng.random(shp) * 10).astype(np.float32)
    fl_p, fl_n = [], []
    for ax in range(3):
        fs = tuple(n + 1 if a == ax else n for a, n in enumerate(shp))
        ok = rng.random(fs) > 0.2
        s = np.where(ok, (rng.random(fs) - 0.5) * 1.2, 0.0).astype(np.float32)
        ff = rng.random(fs) > 0.6
        fa = rng.random(fs) > 0.3                         # faces with something moving
        for p in (0, 1):                                  # the first pass and a later one
            s_eff = np.where(ff, s * inv, s if p == 0 else np.float32(0.0))
            a = np.where(fa, _flux(f, s_eff, ok, ax), np.float32(0.0))
            b = np.empty(fs, np.float32)
            _gf_flux_nb(f, s, ok, ff, fa, inv, p, nsub, ax, b)
            assert _same_bits(a, b), (ax, p)
        fl_p.append(a)
        fl_n.append(b)
    new_p = f.copy()
    for ax in range(3):
        new_p -= np.diff(fl_p[ax], axis=ax)
    new_n = np.empty(shp, np.float32)
    _gf_apply_nb(f, *fl_n, new_n)
    assert _same_bits(new_p, new_n)


# ── the compiled path IS the numpy one ───────────────────────────────────────
def _run_pair(ticks, watts, wall):
    out = []
    for fused in (True, False):
        w = World(40, 40, 44, voxel_cm=5)
        w.fill(0, 40, 0, 40, 0, 1, STONE)
        if wall:
            w.fill(24, 27, 6, 34, 1, 18, STONE)
        w.fused = fused
        w.gas_flow = True
        for t in range(ticks):
            _burn(w, watts, 16, 16, smoke=0.03)
            w.step()
        out.append(w)
    return out


def _same_bits(a, b):
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def test_the_COMPILED_FLOW_and_the_NUMPY_ONE_agree_bit_for_bit():
    """Including the cells that need substeps (a hot plume, 10 kW into 20 cm)
    and a wall in the way."""
    fast, plain = _run_pair(70, 10000.0, True)
    assert fast.gas_flow_stats["passes"] >= 2             # the fast zone ran
    for name in ("E", "smoke", "o2"):
        assert _same_bits(getattr(fast, name), getattr(plain, name)), name
    for a, b in zip(fast._gf["u"], plain._gf["u"]):
        assert _same_bits(a, b)
    assert fast.shed == plain.shed and fast.gone == plain.gone


# ── what rooms really do ─────────────────────────────────────────────────────
def _heskestad(watts, z):
    """Heskestad's plume over a fire of `watts`, at z metres above it: the
    centreline excess temperature (K), the centreline speed (m/s) and the
    volume flow (m^3/s, McCaffrey's entrainment)."""
    kw = watts / 1000.0
    return (25.0 * kw ** (2 / 3) * z ** (-5 / 3), 1.032 * kw ** (1 / 3) * z ** (-1 / 3),
            0.071 * kw ** (1 / 3) * z ** (5 / 3) / 1.2)


def test_the_PLUME_over_a_fire_is_Heskestad_s_within_a_factor_of_one_and_a_half():
    """3 kW into 20 cm of floor, open air all round. The plume the flow makes
    over it is the plume the correlations give: the speed on its axis, the
    flow it carries (which GROWS with height: air is drawn in at its sides)
    and its centreline temperature (which falls).

    Judged from 1.5 m up. A 3 kW plume is narrower than one 20 cm cell below
    about 1.2 m, and the cells cannot show a plume thinner than themselves:
    at 1 m the cell holds the same heat in twice the gas (half the
    temperature, twice the flow), which is the grid's limit and not the law's."""
    n, q = 36, 3000.0
    w = World(n, n, 64, voxel_cm=5)
    w.fill(0, n, 0, n, 0, 1, STONE)
    w.gas_flow = True
    for t in range(int(10 / TICK_S)):
        _burn(w, q, 16, 16)
        w.step()
    T, uz = w.T(), w._gf["u"][2]
    flow, speed, rise = {}, {}, {}
    for z in (1.0, 1.5, 2.0):
        k = int(z / DX) + 1
        flow[z] = float(uz[:, :, k // 4].sum()) * (4 * DX) ** 2
        speed[z] = float(uz[4:6, 4:6, k // 4].max())
        rise[z] = float(T[16:20, 16:20, k].mean()) - AMBIENT
    for z in (1.5, 2.0):
        dT, u, v = _heskestad(q, z)
        assert u / 1.5 < speed[z] < u * 1.5, (z, speed[z], u)
        assert v / 1.5 < flow[z] < v * 1.5, (z, flow[z], v)
        assert dT / 1.5 < rise[z] < dT * 1.5, (z, rise[z], dT)
    # drawn-in air makes the flow grow and the temperature fall with height
    # (all three heights, the one the grid cannot resolve too)
    assert flow[2.0] > 1.5 * flow[1.0] and rise[2.0] < 0.85 * rise[1.0]
    assert flow[1.0] < flow[1.5] < flow[2.0] and rise[1.0] > rise[1.5] > rise[2.0]


def _zukoski(t, watts, n):
    """Where Zukoski's plume puts the smoke layer's interface after t seconds in
    a closed room n voxels a side: the plume entrains 0.071 Q^(1/3) z^(5/3) kg/s
    (Q in kW, z the height above the fire), and the room's cool air is what it
    entrains, so dz/dt = -0.071 Q^(1/3) z^(5/3) / (rho A)."""
    H = n * DX
    c = 0.071 * (watts / 1000.0) ** (1 / 3) / 1.2 / (H * H)
    return (H ** (-2 / 3) + (2 / 3) * c * t) ** -1.5


def test_the_SMOKE_LAYER_falls_the_way_a_plume_fills_a_room():
    """Zukoski. A 1.5 kW fire in a closed 1.6 m room, its smoke a passive
    trace. The layer's interface (from the two-layer fit on the smoke outside
    the plume) must follow the plume's entrainment, and stay a LAYER: a bank
    of smoke over clear air, not smoke spread through the room."""
    n, q = 32, 1500.0
    w = _box(n)
    seen = {}
    for t in range(int(30 / TICK_S)):
        _burn(w, q, 16, 16, smoke=0.008)
        w.step()
        if (t + 1) % 200 == 0:
            seen[round((t + 1) * TICK_S)] = _layer_height(w.smoke, n, (10, 22))
    for sec in (15, 20, 25, 30):
        want = _zukoski(sec, q, n)
        assert 0.7 * want < seen[sec] < 1.3 * want, (sec, seen[sec], want)
    assert seen[30] < seen[15] - 0.1                       # it keeps falling
    # sharp: above the interface the smoke is thick, below it there is almost none
    col = w.smoke[1:n + 1, 1:n + 1, 1:n + 1][:10, :10].sum(axis=(0, 1))
    k = int(seen[30] / DX)
    assert col[k + 4:].mean() > 8 * col[:max(k - 8, 1)].mean()


def _alpert(watts, r, height):
    """Alpert's ceiling jet (r/H > 0.18): the excess temperature (K) and the
    speed (m/s) at radius r (m) under an unconfined ceiling `height` (m) over a
    fire of `watts`."""
    kw = watts / 1000.0
    return 5.38 * (kw / r) ** (2 / 3) / height, 0.197 * kw ** (1 / 3) * height ** 0.5 / r ** (5 / 6)


def test_the_CEILING_JET_under_an_unconfined_ceiling_is_Alpert_s_order_and_shape():
    """A 10 kW fire 2.15 m under a stone plate with the open sky all round. The
    plate stands on a grid of thin posts (a free span of stone this wide would
    fall: the support law is right about that), so the gas leaves freely
    between them. Under the plate a hot jet runs OUTWARD, hottest near the
    axis, and at 1 m from it the gas is Alpert's temperature to a factor of two
    and a half. It is hotter than Alpert's, not cooler, and the jet is slower
    than his (0.1 to 0.3 m/s against 0.6 to 0.9): the flow has no momentum of
    its own, so the plume does not turn the corner at speed and the jet piles
    up. That is the one known gap of this law (see the module's last
    paragraph), which is why the band is not tighter."""
    nx, nz, q = 72, 56, 10000.0
    w = World(nx, nx, nz, voxel_cm=5)
    w.fill(0, nx, 0, nx, 0, 1, STONE)
    w.fill(4, nx - 4, 4, nx - 4, 44, 48, STONE)
    posts = (8, 24, 48, 62)
    for x in posts:
        for y in posts:
            w.fill(x, x + 2, y, y + 2, 1, 44, STONE)
    w.gas_flow = True
    c = nx // 2                                          # the axis is between voxels c-1 and c
    for t in range(int(10 / TICK_S)):
        _burn(w, q, c - 2, c - 2)
        w.step()
    assert float((w.mat[4:nx - 4, 4:nx - 4, 44:48] == STONE).mean()) == 1.0   # it stood
    T = w.T()

    def ring(r):
        k = int(round(r / DX - 0.5))
        pts = [T[c + k, c - 1, 43], T[c - 1 - k, c - 1, 43],
               T[c - 1, c + k, 43], T[c - 1, c - 1 - k, 43]]
        return float(np.mean(pts)) - AMBIENT
    H = 43 * DX
    want, _ = _alpert(q, 1.0, H)
    assert 0.5 * want < ring(1.0) < 2.5 * want, (ring(1.0), want)
    assert ring(0.6) > ring(1.0) > ring(1.4) > 0.0
    ux = w.vcell[0][:, c // 4 - 1:c // 4 + 1, 10].mean(axis=1)   # the layer under the plate
    assert ux[c // 4 + 1:c // 4 + 5].mean() > 0.1 and ux[c // 4 - 5:c // 4 - 1].mean() < -0.1


def test_a_DOORWAY_HAS_A_NEUTRAL_PLANE_hot_gas_out_above_it_cool_air_in_below():
    """A 6 kW fire in a closed 1.6 m room with a 0.6 m x 1.0 m doorway to the
    open. Through the door the gas goes OUT above one height and IN below it;
    the volume in is about the volume out; and the height of the change is the
    one the pressures give: z_n / H = 1 / (1 + (T_hot / T_amb)^(1/3)), for the
    temperature of the gas that goes out."""
    n, X0, q = 32, 8, 6000.0
    wx, wy, wz = X0 + n + 10, n + 18, n + 10
    w = World(wx, wy, wz, voxel_cm=5)
    w.fill(0, wx, 0, wy, 0, 1, STONE)
    w.fill(X0, X0 + n + 2, 8, 8 + n + 2, 1, n + 2, STONE)
    w.mat[X0 + 1:X0 + n + 1, 9:9 + n, 1:n + 1] = AIR
    w.smass[X0 + 1:X0 + n + 1, 9:9 + n, 1:n + 1] = 0
    dy = 8 + 1 + n // 2 - 6
    w.mat[X0:X0 + 1, dy:dy + 12, 1:21] = AIR                 # the door, 1 m high
    w.smass[X0:X0 + 1, dy:dy + 12, 1:21] = 0
    w.gas_flow = True
    for t in range(int(25 / TICK_S)):
        _burn(w, q, X0 + 20, 8 + 1 + 14)
        w.step()
    # the door plane: x-speed of every lane, for the door's z cells (0..5) and
    # y cells (4..7), and what share of a face each lane is
    u = w._gf["u"][0][X0 // 4, 4:8, :6]
    F = w._gf_geometry()["F"][0][X0 // 4, 4:8, :6]
    flux = (4 * DX) ** 2 * F * u                         # m^3/s, + = into the room
    per_row = flux.sum(axis=0)                           # by height
    into, out = float(per_row[per_row > 0].sum()), float(-per_row[per_row < 0].sum())
    assert (per_row[:3] > 0).all() and per_row[5] < 0    # in below, out above
    assert 0.8 < out / into < 1.6, (into, out)           # what comes in goes out (a little more:
                                                         # the gas has warmed and taken room)
    k = int(np.flatnonzero(np.diff(np.sign(per_row)) != 0)[0])
    assert np.count_nonzero(np.diff(np.sign(per_row))) == 1       # one neutral plane
    here, there = per_row[k], per_row[k + 1]
    zc = lambda i: (4 * i + 1) * DX                      # a cell's height above the floor
    z_n = zc(k) + (zc(k + 1) - zc(k)) * here / (here - there)
    hot = float(w.T()[X0 + 1:X0 + n + 1, 9:9 + n, 17:21].mean()) - AMBIENT   # gas going out
    want = 1.0 / (1.0 + ((AMBIENT + 273.15 + hot) / (AMBIENT + 273.15)) ** (1 / 3))
    assert 0.4 < z_n / 1.0 < 0.95, z_n
    assert abs(z_n / 1.0 - want) < 0.25, (z_n, want)     # a quarter of the door: the
                                                         # room is stratified, not one hot
                                                         # box, and a cell is a fifth of it


def _burning_room():
    """The house-fire room (scenes.house_fire's, without the person): a stone
    room 3.9 x 2.9 x 2.15 m with a doorway, a bed, a table and a wardrobe, and
    the bed's underside alight over a spill of oil at 270 C. It is built here
    and not by the scene so that the check does not move when the scene's
    ignition does (the scene now lets the lamp take half a minute to light the
    bed; this one starts with the bed burning, as the scene did)."""
    w = World(100, 80, 50, voxel_cm=5)
    w.fill(0, 100, 0, 80, 0, 1, STONE)
    w.fill(10, 90, 10, 70, 0, 45, STONE)
    w.mat[11:89, 11:69, 1:44] = AIR
    w.smass[11:89, 11:69, 1:44] = 0.0
    w.mat[10:11, 35:45, 1:38] = AIR                       # the doorway
    w.smass[10:11, 35:45, 1:38] = 0.0
    w.fill(66, 86, 14, 28, 5, 7, WOOD, frac=0.7)          # the bed
    for (lx, ly) in ((67, 15), (67, 25), (83, 15), (83, 25)):
        w.fill(lx, lx + 2, ly, ly + 2, 1, 5, WOOD)
    w.fill(30, 44, 50, 62, 10, 11, WOOD, frac=0.8)        # the table
    for (lx, ly) in ((31, 51), (31, 59), (41, 51), (41, 59)):
        w.fill(lx, lx + 2, ly, ly + 2, 1, 10, WOOD)
    w.fill(84, 88, 55, 67, 1, 30, WOOD, frac=0.5)         # the wardrobe
    for (ox, oy) in ((64, 16), (65, 17), (64, 18)):
        w.pour(ox, oy, 1, OIL, 100.0)                     # the dropped lamp
    C = w.heat_capacity()
    underside = np.zeros(w.shape, bool)
    underside[66:86, 14:28, 5:6] = True
    underside &= w.mat == WOOD
    w.E[underside] = (450.0 - AMBIENT) * C[underside]
    pool = w.fl == OIL
    w.E[pool] = (270.0 - AMBIENT) * C[pool]
    return w


def test_the_BURNING_ROOM_is_HOTTER_UNDER_THE_CEILING_than_near_the_floor_at_30_s():
    """The house fire, on the real clock: at 30 s the gas under the ceiling is
    hotter than the gas near the floor, in the half of the room away from the
    bed. (With the old laws it was the other way about: the floor read 127 C
    and the ceiling 24 C.)"""
    w = _burning_room()
    w.gas_flow = True
    for t in range(1200):
        w.step()
    T, air = w.T(), w.mat == AIR
    half = (slice(16, 56), slice(16, 64))

    def mean(z0, z1):
        a = air[half + (slice(z0, z1),)]
        return float(T[half + (slice(z0, z1),)][a].mean())
    top, floor = mean(38, 43), mean(2, 5)
    assert top > floor + 5.0, (top, floor)
    assert int(w.burning().sum()) > 50                    # and it is a real fire


def test_a_QUIET_world_costs_the_flow_nothing_and_a_dying_flow_dies():
    w = _box(24)
    w.step()
    assert not w._pushed() and w.gas_flow_stats == {}
    for t in range(60):
        _burn(w, 3000.0, 8, 8, smoke=0.016)
        w.step()
    assert w._gf_live()
    for t in range(500):                                  # the fire is out
        w.step()
    assert max(float(np.abs(u).max()) for u in w._gf["u"]) < 0.05      # nothing drives it


def test_SKIPPING_THE_QUIET_AIR_changes_nothing_that_matters():
    """The flow sweeps a box around where gas is not at rest (and not the whole
    world), and does not carry a trace of heat under GAS_QUIET_K. Against the
    same world swept whole, every temperature agrees to a tenth of a kelvin and
    the smoke to a thousandth of a gram a voxel."""
    ends = []
    for skip in (True, False):
        w = World(40, 40, 40, voxel_cm=5)
        w.fill(0, 40, 0, 40, 0, 1, STONE)
        w.fill(24, 27, 6, 34, 1, 18, STONE)
        w.skip_quiet = skip
        w.gas_flow = True
        for t in range(120):
            _burn(w, 3000.0, 16, 16, smoke=0.03)
            w.step()
        ends.append(w)
    a, b = ends
    assert float(np.abs(a.T() - b.T()).max()) < 0.1
    assert float(np.abs(a.smoke - b.smoke).max()) < 1e-3
