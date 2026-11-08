"""Kernel tests for src/voxel/outside.py (T8, spec 6.2, K1-K11). Arrays only.

Tags (plan section 9): test_echo_* = the constant was fitted to that number,
test_codeonly_* = tests the code against its own table, test_u_* = independent
but source unverified. The rest are conservation, exactness, closed-form and
arithmetic-identity lines, which may gate scenes.
"""
import numpy as np
import pytest

from src.voxel import outside as O

F = np.float32
HAVE_NUMBA = O.HAVE_NUMBA


def _old_window(there, pad):
    out = []
    for ax in range(3):
        line = there.any(axis=tuple(a for a in range(3) if a != ax))
        nz = np.flatnonzero(line)
        if not len(nz):
            return None
        out.append(slice(max(int(nz[0]) - pad, 0), min(int(nz[-1]) + 1 + pad, there.shape[ax])))
    return tuple(out)


FUSED = [False] + ([True] if HAVE_NUMBA else [])


# ----------------------------------------------------------------- K1, K2
@pytest.mark.parametrize("fused", FUSED)
def test_K1_settle_zero_is_exactly_the_old_E_nonzero_box(fused):
    rng = np.random.default_rng(1)
    shape = (14, 11, 9)
    C = rng.uniform(0.1, 400.0, shape).astype(np.float32)
    cls = np.zeros(shape, np.uint8)
    for trial in range(200):
        E = np.zeros(shape, np.float32)
        k = rng.integers(0, 6)
        for _ in range(k):
            E[tuple(rng.integers(0, s) for s in shape)] = rng.normal() * 5
        pad = int(rng.integers(0, 3))
        box, why = O.unsettled_box(E, C, cls, 0.0, 0.0, 0.0, pad=pad, fused=fused)
        want = _old_window(E != 0, pad)
        assert box == want
        assert (why == []) == (want is None)


@pytest.mark.parametrize("fused", FUSED)
def test_K2_rest_and_driven(fused):
    shape = (10, 10, 8)
    C = np.full(shape, 0.15, np.float32)
    cls = np.zeros(shape, np.uint8)
    cls[:, :, 4:] = 1
    rest_k, th = -25.0, -32.0
    E = O.soak_fill(C, cls, rest_k, th)
    box, why = O.unsettled_box(E, C, cls, rest_k, th, O.SETTLE_K, fused=fused)
    assert box is None and why == []
    E2 = E.copy()
    E2[3, 4, 6] += F(0.01) * C[3, 4, 6]          # 10 mK, over the 5 mK tolerance
    box, why = O.unsettled_box(E2, C, cls, rest_k, th, O.SETTLE_K, fused=fused)
    assert box == (slice(3, 4), slice(4, 5), slice(6, 7)) and why == ["off_rest"]
    E3 = E.copy()
    E3[3, 4, 6] += F(0.002) * C[3, 4, 6]         # 2 mK, inside it
    assert O.unsettled_box(E3, C, cls, rest_k, th, O.SETTLE_K, fused=fused)[0] is None
    exposed = np.zeros(shape, bool)
    exposed[2:5, 2:5, 3:6] = True
    why_driven = O.driven_reasons(-25.0, -32.0, -32.0)
    assert why_driven == ["interface"]
    box, why = O.unsettled_box(E, C, cls, rest_k, th, O.SETTLE_K, exposed=exposed,
                               driven=why_driven, fused=fused)
    assert box == (slice(2, 5), slice(2, 5), slice(3, 6)) and why == ["interface"]
    # not driven: the exposed list is not added
    assert O.unsettled_box(E, C, cls, rest_k, th, O.SETTLE_K, exposed=exposed, driven=[],
                           fused=fused)[0] is None
    assert O.driven_reasons(20.0, 20.0, 20.0) == []
    assert O.driven_reasons(20.0, 20.0, -4.0) == ["sky_drive"]


# ----------------------------------------------------------------- K3
@pytest.mark.parametrize("fused", FUSED)
def test_K3_outdoor_shift_settled_bit_equal_and_books_joules(fused):
    rng = np.random.default_rng(3)
    shape = (12, 12, 10)
    C = rng.uniform(0.1, 3.0, shape).astype(np.float32)
    cls = (rng.random(shape) < 0.5).astype(np.uint8)
    th_old, th_new = -32.0, -31.35
    E = O.soak_fill(C, cls, -25.0, th_old)
    E[2, 2, 2] = F(0.7) * C[2, 2, 2] + E[2, 2, 2]          # a plume cell (if outdoor)
    cls[2, 2, 2] = 1
    E[2, 2, 2] = F(th_old) * C[2, 2, 2] + F(0.7) * C[2, 2, 2]
    before = E.copy()
    tot0 = float(E.sum(dtype=np.float64))
    booked = O.outdoor_shift(E, C, cls, th_old, th_new, fused=fused)
    out = (cls & 1) != 0
    settled = out & (before == F(th_old) * C)
    assert np.array_equal(E[settled], (F(th_new) * C)[settled])
    assert np.array_equal(E[~out], before[~out])
    dev_before = (before - F(th_old) * C)[2, 2, 2]
    dev_after = (E - F(th_new) * C)[2, 2, 2]
    assert abs(dev_after - dev_before) < 1e-5 * abs(dev_before) + 1e-6
    tot1 = float(E.sum(dtype=np.float64))
    assert booked == pytest.approx(tot1 - tot0, rel=1e-12, abs=1e-9)
    assert O.outdoor_shift(E, C, cls, th_new, th_new, fused=fused) == 0.0


# ----------------------------------------------------------------- K4
@pytest.mark.parametrize("fused", FUSED)
def test_K4_outdoor_relax_exact_exponential_two_rates_never_overshoots(fused):
    dt = 0.025
    shape = (4, 4, 4)
    C = np.full(shape, 0.15, np.float32)
    cls = np.ones(shape, np.uint8)               # free air
    cls[:2] = 3                                  # film cells
    th = -32.0
    E = (F(th) * C).astype(np.float32)
    E += F(50.0) * C                             # 50 K above the target everywhere
    kf, ka = 1.0, 0.1
    ff, fa = O.relax_factor(kf, dt), O.relax_factor(ka, dt)
    assert ff == pytest.approx(0.0247, abs=2e-4) and fa == pytest.approx(0.0025, abs=1e-4)
    booked_f = booked_a = 0.0
    e0 = float(E.sum(dtype=np.float64))
    for n in range(1, 101):
        a, b = O.outdoor_relax(E, C, cls, th, ff, fa, O.SETTLE_K, fused=fused)
        booked_f += a
        booked_a += b
        if n in (10, 50, 100):
            dev_f = float(((E - F(th) * C) / C)[0, 0, 0])
            dev_a = float(((E - F(th) * C) / C)[3, 0, 0])
            assert dev_f == pytest.approx(50.0 * np.exp(-kf * n * dt), rel=1e-4)
            assert dev_a == pytest.approx(50.0 * np.exp(-ka * n * dt), rel=1e-4)
    # the joules booked are exactly what the lattice lost
    assert booked_f + booked_a == pytest.approx(e0 - float(E.sum(dtype=np.float64)), rel=1e-12)
    # k_far cells decay ten times slower
    assert np.log(50.0 / ((E - F(th) * C) / C)[0, 0, 0]) / np.log(
        50.0 / ((E - F(th) * C) / C)[3, 0, 0]) == pytest.approx(10.0, rel=0.01)


@pytest.mark.parametrize("fused", FUSED)
def test_K4_never_overshoots_even_at_100_per_second_and_snaps(fused):
    dt = 0.025
    f = O.relax_factor(100.0, dt)
    assert 0.0 < f < 1.0
    rng = np.random.default_rng(4)
    shape = (8, 8, 8)
    C = rng.uniform(0.05, 5.0, shape).astype(np.float32)
    cls = np.full(shape, 3, np.uint8)
    th = -12.0
    for sign in (+1, -1):
        E = (F(th) * C + sign * rng.uniform(0.0, 60.0, shape).astype(np.float32) * C).astype(np.float32)
        et = F(th) * C
        s0 = np.sign(E - et)
        for _ in range(400):
            O.outdoor_relax(E, C, cls, th, f, f, O.SETTLE_K, fused=fused)
            s = np.sign(E - et)
            assert np.all((s == 0) | (s == s0))      # never crosses the target
        assert np.all(E == et)                       # and every cell snapped onto it
    # a pure factor of 1 would jump to the target but not past it
    E = (F(th) * C + F(10.0) * C).astype(np.float32)
    O.outdoor_relax(E, C, cls, th, F(1.0), F(1.0), 0.0, fused=fused)
    assert np.all(E == F(th) * C)


# ----------------------------------------------------------------- K5, K6
def _yard_lattice(nx=60, ny=60, nz=40):
    Hs = np.full((nx, ny), -1, np.int64)
    return Hs, np.ones((nx, ny, nz), bool)


def _room(nx=60, ny=60, nz=40, w=20, h=12, door=False, hole=False):
    """A box room (walls 2 cells thick) with a flat roof at z=h. Solid cells
    are returned as a mask; Hs is the highest solid per column."""
    solid = np.zeros((nx, ny, nz), bool)
    x0, y0 = nx // 2 - w // 2, ny // 2 - w // 2
    solid[x0:x0 + w, y0:y0 + w, :h + 1] = True
    solid[x0 + 2:x0 + w - 2, y0 + 2:y0 + w - 2, :h] = False    # hollow, floor at z=0..-1: no floor
    solid[x0 + 2:x0 + w - 2, y0 + 2:y0 + w - 2, 0] = True       # floor
    if door:
        solid[x0:x0 + 2, y0 + w // 2 - 2:y0 + w // 2 + 2, 1:9] = False
    if hole:
        solid[x0 + w // 2, y0 + w // 2, h] = False
    Hs = np.full((nx, ny), -1, np.int64)
    for k in range(nz):
        Hs = np.where(solid[:, :, k], k, Hs)
    return solid, Hs, (x0, y0, w, h)


@pytest.mark.parametrize("fused", FUSED)
def test_K5_outdoor_class_by_sky_view_not_connectivity(fused):
    for kind in ("closed", "door", "hole"):
        solid, Hs, (x0, y0, w, h) = _room(door=(kind == "door"), hole=(kind == "hole"))
        air = ~solid
        R = O.view_radius_cells(Hs, 0.05)
        cls = O.outdoor_class(Hs, air, np.zeros_like(air), R, fused=fused)
        out = (cls & 1) != 0
        interior = out[x0 + 2:x0 + w - 2, y0 + 2:y0 + w - 2, 1:h - 2]
        assert not interior.any(), kind                  # never outdoor, door open or roof holed
        assert not out[x0 + 3:x0 + w - 3, y0 + 3:y0 + w - 3, 1:h].sum() > 1   # at most the cell under a hole
        assert out[5, 5, 5] and out[x0 - 3, y0 + w // 2, 2]       # the yard is
        assert out[x0 + w // 2, y0 + w // 2, h + 3]               # the air above the roof is
        if kind == "hole":
            assert out[x0 + w // 2, y0 + w // 2, h]               # the air IN the hole may be
            assert not out[x0 + w // 2, y0 + w // 2, 3]           # but not the room under it
    off = O.outdoor_class(Hs, air, np.zeros_like(air), R, open_sky=False, fused=fused)
    assert not off.any()


@pytest.mark.parametrize("fused", FUSED)
def test_K5_courtyard_has_outdoor_air_only_high_and_near_the_middle(fused):
    nx = 80
    solid = np.zeros((nx, nx, 60), bool)
    solid[20:60, 20:60, :40] = True
    solid[28:52, 28:52, :40] = False               # a 24-cell (1.2 m) courtyard, 2 m walls
    solid[28:52, 28:52, 0] = True
    Hs = np.full((nx, nx), -1, np.int64)
    for k in range(60):
        Hs = np.where(solid[:, :, k], k, Hs)
    air = ~solid
    R = O.view_radius_cells(Hs, 0.05)
    cls = O.outdoor_class(Hs, air, np.zeros_like(air), R, fused=fused)
    out = (cls & 1) != 0
    assert not out[40, 40, 1] and not out[30, 40, 5]          # the floor is a canyon
    assert out[40, 40, 45]                                    # above the wall tops
    # sky view only rises up a column, so what is outdoor stays outdoor going up
    col = out[40, 40, 1:]
    assert not np.any(col[:-1] & ~col[1:])


@pytest.mark.parametrize("fused", FUSED)
def test_K6_sky_view_flat_canyons_and_roof(fused):
    nx, ny = 900, 100
    flat = np.full((nx, ny), -1, np.int64)
    c = np.array([[450, 50, 0], [450, 50, 7]])
    assert np.allclose(O.sky_view(flat, c, 240, fused=fused), 1.0, atol=0)
    # canyon W 2 m, H 4 m (5 cm cells): 40 wide, walls 80 high
    Hs = np.full((nx, ny), -1, np.int64)
    Hs[:, :30] = 79
    Hs[:, 70:] = 79
    psi = [float(O.sky_view(Hs, np.array([[450, 50, 0]]), 240, 16, ph, fused=fused)[0])
           for ph in (0.0, np.pi / 32, np.pi / 16)]
    exact = 1.0 / np.sqrt(1.0 + (2 * 4.0 / 2.0) ** 2)
    assert exact == pytest.approx(0.2425, abs=1e-3)
    assert all(abs(p - exact) < 0.03 for p in psi), psi
    assert max(psi) - min(psi) < 0.03                         # compass phase
    # canyon W 4 m, H 2 m
    Hs = np.full((nx, ny), -1, np.int64)
    Hs[:, :10] = 39
    Hs[:, 90:] = 39
    psi = [float(O.sky_view(Hs, np.array([[450, 49, 0]]), 240, 16, ph, fused=fused)[0])
           for ph in (0.0, np.pi / 32, np.pi / 16)]
    exact = 1.0 / np.sqrt(1.0 + (2 * 2.0 / 4.0) ** 2)
    assert all(abs(p - exact) < 0.03 for p in psi), psi
    # under a roof: 0
    roof = np.full((nx, ny), -1, np.int64)
    roof[400:500, 20:80] = 20
    assert O.sky_view(roof, np.array([[450, 50, 5]]), 240, fused=fused)[0] == 0.0
    # monotone up a column
    up = O.sky_view(Hs, np.array([[450, 49, z] for z in range(0, 60)]), 240, fused=fused)
    assert np.all(np.diff(up) >= 0)


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba absent: nothing to compare")
def test_K8_sky_view_and_class_fused_equal_plain_bits():
    rng = np.random.default_rng(8)
    Hs = rng.integers(-1, 30, (50, 50)).astype(np.int64)
    cells = np.stack([rng.integers(0, 50, 300), rng.integers(0, 50, 300), rng.integers(0, 40, 300)], 1)
    a = O.sky_view(Hs, cells, 40, fused=False)
    b = O.sky_view(Hs, cells, 40, fused=True)
    assert np.array_equal(a, b)
    air = np.ones((50, 50, 40), bool)
    ca = O.outdoor_class(Hs, air, air, 30, fused=False)
    cb = O.outdoor_class(Hs, air, air, 30, fused=True)
    assert np.array_equal(ca, cb)


# ----------------------------------------------------------------- K7
@pytest.mark.parametrize("fused", FUSED)
def test_K7_radiation_zero_at_equal_clear_sky_flux_and_shortwave(fused):
    rng = np.random.default_rng(7)
    shape = (6, 6, 6)
    ts = rng.uniform(-30, 40, shape).astype(np.float32)
    ones = np.ones(shape, np.float32)
    n = np.ones(shape, np.uint8)
    cs = np.full(shape, 50.0, np.float32)
    z = O.radiation_out(ts, 0.9 * ones, ones, n, ts.copy(), 0.0025, 0.025, 0.0, cs, fused=fused)
    assert np.all(z == 0.0)                                   # exactly 0 when Ts = Tenv
    # a clear-sky up face at air temperature (0 C, sky -24.3 C): 80-95 W/m2
    A, dt = 0.0025, 0.025
    t = np.zeros(shape, np.float32)
    loss = O.radiation_out(t, 0.9 * ones, ones, n, np.full(shape, -24.3, np.float32), A, dt, 0.0,
                           cs * 1e3, fused=fused)
    w_m2 = float(loss[0, 0, 0]) / (A * dt)
    assert 80.0 < w_m2 < 95.0
    # a cold surface under a warm environment GAINS (negative loss)
    cold = O.radiation_out(np.full(shape, -30, np.float32), 0.9 * ones, ones, n,
                           np.full(shape, 0.0, np.float32), A, dt, 0.0, cs * 1e3, fused=fused)
    assert np.all(cold < 0)
    # shortwave reduces the loss one for one (after the clip)
    sw = 100.0
    loss_sw = O.radiation_out(t, 0.9 * ones, ones, n, np.full(shape, -24.3, np.float32), A, dt, sw,
                              cs * 1e3, fused=fused)
    assert np.allclose(loss - loss_sw, sw * A * dt, rtol=1e-4)
    # no outdoor face, no radiation, no shortwave
    none = O.radiation_out(t, 0.9 * ones, ones, np.zeros(shape, np.uint8),
                           np.full(shape, -24.3, np.float32), A, dt, sw, cs, fused=fused)
    assert np.all(none == 0)


def test_codeonly_K7_clip_limits_the_radiative_part():
    shape = (2, 2, 2)
    ones = np.ones(shape, np.float32)
    ts = np.full(shape, 500.0, np.float32)
    te = np.zeros(shape, np.float32)
    cs = np.full(shape, 1.0, np.float32)
    loss = O.radiation_out(ts, ones, ones, np.ones(shape, np.uint8), te, 1.0, 1.0, 0.0, cs)
    assert np.allclose(loss, O.FACE_CLIP * 1.0 * 500.0)       # clip*Csur*|Ts-Tenv|


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba absent: nothing to compare")
def test_K8_radiation_and_relax_and_box_fused_equal_plain_bits():
    rng = np.random.default_rng(81)
    shape = (9, 8, 7)
    ts = rng.uniform(-40, 600, shape).astype(np.float32)
    e = rng.uniform(0.1, 1.0, shape).astype(np.float32)
    af = rng.uniform(0.3, 1.0, shape).astype(np.float32)
    n = rng.integers(0, 4, shape).astype(np.uint8)
    te = rng.uniform(-40, 20, shape).astype(np.float32)
    cs = rng.uniform(0.1, 300, shape).astype(np.float32)
    sw = rng.uniform(0, 300, shape).astype(np.float32)
    a = O.radiation_out(ts, e, af, n, te, 0.0025, 0.025, sw, cs, fused=False)
    b = O.radiation_out(ts, e, af, n, te, 0.0025, 0.025, sw, cs, fused=True)
    assert np.array_equal(a, b)
    C = rng.uniform(0.1, 5.0, shape).astype(np.float32)
    cls = rng.integers(0, 4, shape).astype(np.uint8)
    E0 = (rng.normal(0, 3, shape) * C).astype(np.float32)
    Ea, Eb = E0.copy(), E0.copy()
    ba = O.outdoor_relax(Ea, C, cls, -5.0, 0.0247, 0.0025, 0.005, fused=False)
    bb = O.outdoor_relax(Eb, C, cls, -5.0, 0.0247, 0.0025, 0.005, fused=True)
    assert np.array_equal(Ea, Eb) and ba == pytest.approx(bb, rel=1e-12, abs=1e-12)
    Ea, Eb = E0.copy(), E0.copy()
    sa = O.outdoor_shift(Ea, C, cls, -5.0, -4.0, fused=False)
    sb = O.outdoor_shift(Eb, C, cls, -5.0, -4.0, fused=True)
    assert np.array_equal(Ea, Eb) and sa == pytest.approx(sb, rel=1e-12, abs=1e-12)
    for pad in (0, 1):
        assert (O.unsettled_box(E0, C, cls, 0.5, -5.0, 0.005, pad=pad, fused=False)
                == O.unsettled_box(E0, C, cls, 0.5, -5.0, 0.005, pad=pad, fused=True))


# ----------------------------------------------------------------- K9
@pytest.mark.parametrize("fused", FUSED)
def test_K9_mrt_star_zero_sum_uniform_hot_patch_and_rooms(fused):
    rng = np.random.default_rng(9)
    n = 300
    ts = rng.uniform(-10, 40, n)
    room = np.where(np.arange(n) < 200, 1, 2).astype(np.int32)
    eps = rng.uniform(0.3, 0.95, n)
    af = rng.uniform(0.5, 1.0, n)
    nin = rng.integers(1, 4, n).astype(np.uint8)
    cs = np.full(n, 1e12)                                    # clip off
    r = O.mrt_star(ts, eps, af, nin, room, 0.0025, 0.025, cs, dtype=np.float64, fused=fused)
    for rid in (1, 2):
        m = room == rid
        gross = np.abs(r.q[m]).sum()
        assert abs(r.q[m].sum()) < 1e-12 * gross          # zero-sum per room: nothing exchanged
    # float32 rounding goes to the residue book, well under 1e-6 of what was exchanged
    r32 = O.mrt_star(ts, eps, af, nin, room, 0.0025, 0.025, cs, fused=fused)
    assert abs(r32.resid) < 1e-6 * np.abs(r32.q).sum()
    # a uniform room gives exactly 0
    u = O.mrt_star(np.full(n, 7.25), eps, af, nin, room, 0.0025, 0.025, cs, fused=fused)
    assert np.all(u.q == 0.0) and u.resid == 0.0
    # a hot patch loses and the rest gain
    t2 = np.full(n, 10.0)
    t2[:5] = 300.0
    h = O.mrt_star(t2, eps, af, nin, room, 0.0025, 0.025, cs, dtype=np.float64, fused=fused)
    assert np.all(h.q[:5] > 0) and np.all(h.q[5:200] < 0)
    assert np.all(h.q[200:] == 0.0)                           # the other room is untouched
    # surfaces at or above RAD_HOT are out of the pool: they neither give nor take
    t3 = np.full(n, 10.0)
    t3[:5] = 900.0
    p = O.mrt_star(t3, eps, af, nin, room, 0.0025, 0.025, cs, dtype=np.float64, fused=fused)
    assert np.all(p.q == 0.0)                                 # the rest is uniform, the hot are excluded
    assert np.allclose(p.tmr_cell[:5], 10.0, atol=1e-4)       # they can radiate to this node
    # the clip's residue is what breaks the zero sum, and it is booked
    cl = O.mrt_star(t2, eps, af, nin, room, 0.0025, 0.025, np.full(n, 1e-3), dtype=np.float64,
                    fused=fused)
    assert cl.resid == pytest.approx(cl.q.sum())


@pytest.mark.parametrize("fused", FUSED)
def test_K9_mrt_star_zero_sum_for_any_spread_cold_first_hot_later(fused):
    # the radiant node T_mr must be the exact mean of T^4 whatever the order and
    # the spread: a cold first member with a hearth between RAD_HOT and 100 C later
    def zs(ts, k_room=1):
        ts = np.asarray(ts, float)
        n = ts.size
        r = O.mrt_star(ts, np.ones(n), np.ones(n), np.ones(n, np.uint8),
                       np.full(n, k_room, np.int32), 1.0, 1.0, np.full(n, 1e18),
                       dtype=np.float64, fused=fused)
        tk = ts + 273.15
        exact = (np.mean(tk ** 4)) ** 0.25
        return abs(r.q.sum()) / np.abs(r.q).sum(), r.tmr_room[k_room], exact
    cases = [[-12] * 4 + [400], [-12] * 3 + [449], [20, 449], [300] + [-12] * 5,
             [-5] * 30 + [300], [449, -12, -12, -12], [-60, 440, 20, -60, 300]]
    for ts in cases:
        zsum, tmr, exact = zs(ts)
        assert zsum < 1e-12, (ts, zsum)
        assert tmr == pytest.approx(exact, rel=1e-12), (ts, tmr, exact)
    rng = np.random.default_rng(31)
    for _ in range(50):
        zsum, tmr, exact = zs(rng.uniform(-60, 440, 40))
        assert zsum < 1e-12 and tmr == pytest.approx(exact, rel=1e-12)
    # a nearly uniform room: T+273.15 itself rounds at ~6e-14 K, so a 1e-6 K spread
    # can only close to ~1e-7 of what was exchanged (the Newton polish adds nothing)
    zsum, tmr, exact = zs([20.0, 20.0 + 1e-6, 20.0 - 1e-6, 20.0])
    assert zsum < 1e-7


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba absent: nothing to compare")
def test_K8_mrt_fused_equal_plain_bits():
    rng = np.random.default_rng(10)
    n = 400
    args = (rng.uniform(-10, 60, n), rng.uniform(0.3, 0.95, n), rng.uniform(0.5, 1, n),
            rng.integers(1, 4, n).astype(np.uint8), rng.integers(1, 6, n).astype(np.int32),
            0.0025, 0.025, rng.uniform(1, 100, n))
    a = O.mrt_star(*args, fused=False)
    b = O.mrt_star(*args, fused=True)
    assert np.array_equal(a.q, b.q) and a.resid == b.resid


# ----------------------------------------------------------------- K10
def test_K10_rest_shift_and_matter_change_close_the_books():
    assert O.rest_shift(5.0, 5.0, -12.0) == 0.0
    assert O.rest_shift(1.0, 9.0, 20.0) == 0.0               # at 20 C the rule is silent
    rng = np.random.default_rng(10)
    for _ in range(200):
        co = rng.uniform(0.1, 400.0)
        cn = rng.uniform(0.01, 400.0)
        tr = rng.uniform(-40, 30)
        E = rng.normal(0, 50.0)
        En, ref, carried = O.matter_change(E, co, cn, tr)
        assert En + ref + carried == pytest.approx(E, rel=1e-12, abs=1e-9)
        if cn >= co:
            assert carried == 0.0
        # a conversion in place at rest stays AT rest: T = rest before and after
    co, cn, tr = 260.0, 0.15, -5.0
    E = (tr - 20.0) * co                                      # a stone cell exactly at rest
    En, ref, carried = O.matter_change(E, co, cn, tr)
    assert 20.0 + En / cn == pytest.approx(tr, abs=1e-9)      # the air left behind is at rest
    # a COLD stone (3 K below rest) dissolved to air does not leave air at -43,000 C
    E = (tr - 20.0 - 3.0) * co
    En, ref, carried = O.matter_change(E, co, cn, tr)
    t_after = 20.0 + En / cn
    assert t_after == pytest.approx(tr - 3.0, abs=1e-9) and t_after > -273.15
    assert carried < 0.0                                       # the cold left with the matter
    # hot stuff keeps the old behaviour: the heat stays in the cell
    E = (tr - 20.0 + 100.0) * co
    En, ref, carried = O.matter_change(E, co, cn, tr)
    assert carried == 0.0


def test_K10_soak_fill_and_t_fill():
    C = np.full((4, 4, 4), 0.15, np.float32)
    cls = np.zeros((4, 4, 4), np.uint8)
    cls[:, :, 2:] = 1
    E = O.soak_fill(C, cls, -25.0, -32.0)
    assert np.all(E[:, :, :2] == F(-25.0) * C[:, :, :2]) and np.all(E[:, :, 2:] == F(-32.0) * C[:, :, 2:])
    T = O.t_fill(cls, -5.0, -12.0)
    assert np.all(T[:, :, :2] == -5.0) and np.all(T[:, :, 2:] == -12.0)


# ----------------------------------------------------------------- K11
def _brute_face_env(surf, gas, outdoor, room, open_sky, ground_k):
    nx, ny, nz = surf.shape
    n_out = np.zeros(surf.shape, int)
    n_in = np.zeros(surf.shape, int)
    n_edge = np.zeros(surf.shape, int)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not surf[i, j, k]:
                    continue
                for d, (dx, dy, dz) in enumerate(O._DIRS6):
                    a, b, c = i + dx, j + dy, k + dz
                    if not (0 <= a < nx and 0 <= b < ny and 0 <= c < nz):
                        if open_sky and dz != -1 and k >= ground_k:
                            n_out[i, j, k] += 1
                            n_edge[i, j, k] += 1
                    elif gas[a, b, c]:
                        if outdoor[a, b, c]:
                            n_out[i, j, k] += 1
                        else:
                            n_in[i, j, k] += 1
    return n_out, n_in, n_edge


@pytest.mark.parametrize("fused", FUSED)
def test_K11_face_env_counts_match_a_brute_force_scan(fused):
    rng = np.random.default_rng(11)
    for trial in range(6):
        shape = (7, 6, 5)
        solid = rng.random(shape) < 0.4
        gas = ~solid
        surf = solid & (rng.random(shape) < 0.9)
        cls = np.where(gas & (rng.random(shape) < 0.5), 1, 0).astype(np.uint8)
        psi = (rng.random(shape) * (cls & 1)).astype(np.float32)
        room = rng.integers(1, 4, shape).astype(np.int32)
        for open_sky, gk in ((True, 0), (True, 2), (False, 0)):
            fe = O.face_env(surf, gas, cls, psi, room, open_sky=open_sky, ground_k=gk, fused=fused)
            no, ni, ne = _brute_face_env(surf, gas, (cls & 1) != 0, room, open_sky, gk)
            assert np.array_equal(fe.n_out, no) and np.array_equal(fe.n_in, ni)
            assert np.array_equal(fe.n_edge, ne)
            assert np.all(fe.w_sky[~surf] == 0) and np.all(fe.w_sky <= fe.n_out + 1e-6)
            # the room of a cell is the id of one of its neighbours' rooms
            has = surf & (ni > 0)
            assert np.all(fe.room[has] > 0) and np.all(fe.room[~has] == 0)


def test_K11_edge_faces_below_ground_top_are_not_outdoor():
    surf = np.zeros((4, 4, 4), bool)
    surf[0, 1, 0] = surf[0, 1, 3] = True                  # a low and a high cell on the x=0 edge
    gas = np.zeros((4, 4, 4), bool)
    cls = np.zeros((4, 4, 4), np.uint8)
    fe = O.face_env(surf, gas, cls, np.zeros(surf.shape), np.zeros(surf.shape, np.int32),
                    open_sky=True, ground_k=2, fused=False)
    assert fe.n_out[0, 1, 0] == 0 and fe.n_out[0, 1, 3] == 2     # side edge + top edge for the high one
    assert fe.w_sky[0, 1, 3] == pytest.approx(1.5)               # 0.5 side, 1.0 top
    fe = O.face_env(surf, gas, cls, np.zeros(surf.shape), np.zeros(surf.shape, np.int32),
                    open_sky=True, ground_k=0, fused=False)
    assert fe.n_out[0, 1, 0] == 1                                # bottom edge is never open


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba absent: nothing to compare")
def test_K8_face_env_fused_equal_plain_bits():
    rng = np.random.default_rng(12)
    shape = (9, 8, 7)
    solid = rng.random(shape) < 0.4
    gas = ~solid
    cls = np.where(gas & (rng.random(shape) < 0.5), 1, 0).astype(np.uint8)
    psi = (rng.random(shape) * (cls & 1)).astype(np.float32)
    room = rng.integers(1, 4, shape).astype(np.int32)
    a = O.face_env(solid, gas, cls, psi, room, True, 1, fused=False)
    b = O.face_env(solid, gas, cls, psi, room, True, 1, fused=True)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_env_temperature_exact_ends_and_mix():
    n = np.array([1, 1, 2, 0], np.uint8)
    w = np.array([0.0, 1.0, 1.0, 0.0], np.float32)
    t = O.env_temperature(n, w, -24.0, -4.0)
    assert t[0] == np.float32(-4.0) and t[1] == np.float32(-24.0) and t[3] == np.float32(-4.0)
    assert -24.0 < t[2] < -4.0
    # T^4 mixes linearly: the mid value is the fourth root of the mean of fourths
    want = np.sqrt(np.sqrt(0.5 * (249.15 ** 4 + 269.15 ** 4))) - 273.15
    assert t[2] == pytest.approx(want, abs=1e-4)
    # a sky equal to the air: exactly the air, no power round trip
    assert np.all(O.env_temperature(n, w, 3.3, 3.3) == np.float32(3.3))
    assert np.allclose(O.env_t4(n, w, 3.3, 3.3), (3.3 + 273.15) ** 4, rtol=1e-6)


def test_edge_film_and_ground_flux_signs_and_clip():
    ts = np.array([10.0, -20.0, -4.0], np.float32)
    ne = np.array([1, 2, 0], np.uint8)
    af = np.ones(3, np.float32)
    cs = np.full(3, 1e6, np.float32)
    q = O.edge_film(ts, ne, -4.0, af, 0.0025, 0.025, cs)
    assert q[0] > 0 and q[1] < 0 and q[2] == 0                 # warm leaves, cold gains, no edge
    assert q[0] == pytest.approx(10.0 * 0.0025 * 0.025 * 14.0, rel=1e-5)
    g = O.ground_flux(np.array([5.0, -5.0], np.float32), 0.0, np.array([2.0, 2.0], np.float32),
                      0.0025, 0.025, 1.025, np.full(2, 1e6, np.float32))
    assert g[0] > 0 and g[1] < 0
    assert g[0] == pytest.approx(2.0 * 0.0025 / 1.025 * 0.025 * 5.0, rel=1e-5)


def test_column_max_matches_a_brute_force_window():
    rng = np.random.default_rng(13)
    a = rng.integers(-1, 30, (23, 17)).astype(np.int64)
    for r in (0, 1, 3, 9, 40):
        got = O.column_max(a, r)
        want = np.empty_like(a)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                want[i, j] = a[max(i - r, 0):i + r + 1, max(j - r, 0):j + r + 1].max()
        assert np.array_equal(got, want)
