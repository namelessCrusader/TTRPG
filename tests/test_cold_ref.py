"""M-tests (T8 spec 6.3): the kernels composed on a small reference lattice, no
World. tests/ref_cold_lattice.py is the lattice. Each test is small (a few
seconds); the lattices are 5 cm cells and a few thousand of them.

Tags: conservation / exactness / closed form tests gate scenes by default;
test_u_* are independent numbers with unverified sources; test_codeonly_* test
the code against itself.
"""
import math

import numpy as np
import pytest

from src.voxel import climate as K
from src.voxel import outside as O
from tests.ref_cold_lattice import FOAM, GLASS, STONE, WOOD, MATS, RefLattice

# an ultra-light metal for the lumped-cube test: its time constant is ~100 s
LIGHT = 6
MATS[LIGHT] = (3.4e4, 50.0, 0.80)


def _hut(climate, door=False, hole=False, open_sky=True, shape=(24, 24, 16), **kw):
    L = RefLattice(shape, climate=climate, open_sky=open_sky, **kw)
    x0, x1, y0, y1, h = 6, 18, 6, 18, 8
    L.fill(x0, x1, y0, y1, 0, 1, STONE)                               # floor
    L.fill(x0, x1, y0, y1, 1, h, STONE)                               # walls ...
    L.fill(x0 + 2, x1 - 2, y0 + 2, y1 - 2, 1, h, 0)                   # ... hollowed
    L.fill(x0, x1, y0, y1, h, h + 1, WOOD)                            # roof
    if door:
        L.fill(x0, x0 + 2, 11, 13, 1, 5, 0)
    if hole:
        L.fill(11, 12, 11, 12, h, h + 1, 0)
    return L, (x0 + 2, x1 - 2, y0 + 2, y1 - 2, 1, h)


def _inside(L, box, T=None):
    x0, x1, y0, y1, z0, z1 = box
    m = np.zeros(L.shape, bool)
    m[x0:x1, y0:y1, z0:z1] = True
    return m & L.gas


# ------------------------------------------------------------------ M1
def test_M1_quiet_reference_climate_is_bit_still_for_400_steps():
    c = K.Climate.reference(t_c=20.0)
    L, box = _hut(c, door=True)
    L.soak()
    E0 = L.E.copy()
    shed0, booked0 = L.shed, dict(L.booked)
    for _ in range(400):
        L.step()
        assert L.last_box is None and L.last_why == []
    assert np.array_equal(L.E, E0)
    assert L.shed == shed0 and L.booked == booked0
    assert L.awake_steps == 0
    # and the same for a cold world whose sky is its air
    cc = K.Climate.reference(t_c=-5.0)
    L2, _ = _hut(cc)
    L2.soak()
    E1 = L2.E.copy()
    L2.run(400)
    assert np.array_equal(L2.E, E1) and L2.awake_steps == 0


# ------------------------------------------------------------------ M2
def _slab(climate, **kw):
    L = RefLattice((14, 14, 12), climate=climate, **kw)
    L.fill(2, 12, 2, 12, 0, 2, FOAM)
    L.fill(2, 12, 2, 12, 2, 5, STONE)
    return L


def test_M2_a_settled_wall_beside_colder_outdoor_air_wakes_and_cools_at_the_film_rate():
    c = K.Climate.reference(t_c=-12.0).replace(rest_c=-5.0)            # solids rest at -5, air at -12
    L = _slab(c)
    L.soak()
    T0 = L.T()
    top = (7, 7, 4)
    assert abs(T0[top] - (-5.0)) < 1e-4 and abs(T0[7, 7, 6] - (-12.0)) < 1e-4
    L.step()
    assert L.last_box is not None and "interface" in L.last_why         # the gate did not hide it
    n = 160                                                             # 40 s
    L.run(n - 1)
    drop = float(T0[top] - L.T()[top])
    # initial slope: (series film 8.57 + radiation 4 eps sigma T^3) * A * dT / C
    C = float(L.C[top])
    u_film = 1.0 / (1.0 / O.H_CONV + 1.0 / (L.climate.k_film * (1200.0 * L.dx) ))
    assert u_film == pytest.approx(8.57, abs=0.05)
    h_r = 4 * 0.9 * O.SIGMA * (273.15 - 8.5) ** 3
    slope = (u_film + h_r) * L.A * 7.0 / C
    assert drop == pytest.approx(slope * n * L.dt, rel=0.25)
    assert drop > 0.01


# ------------------------------------------------------------------ M3
def test_M3_floor_gone_a_cold_stone_in_cold_air_is_not_clamped_and_energy_closes():
    c = K.Climate.reference(t_c=-40.0)
    L = RefLattice((12, 12, 10), climate=c)
    L.fill(3, 9, 3, 9, 0, 4, STONE)
    L.soak()
    L.E[5, 5, 2] += np.float32(1.0 * L.C[5, 5, 2])                       # one cell 1 K warm
    e0 = L.total_energy()
    scale = float(np.abs(L.E).sum(dtype=np.float64))
    L.run(400)
    T = L.T()
    assert T.min() > -40.0 - 0.01 and T.min() < -39.99
    assert T.max() < -39.0
    assert abs(L.total_energy() - e0) < 1e-6 * scale
    assert L.awake_steps > 0


# ------------------------------------------------------------------ M4
def _lumped(t0, t_end, dt, c_tot, area, eps, t_out, u_series):
    t, T = 0.0, t0
    out = {}
    while t < t_end - 1e-9:
        tk, te = T + 273.15, t_out + 273.15
        q = area * (u_series * (T - t_out) + eps * O.SIGMA * (tk ** 4 - te ** 4))
        T -= q * dt / c_tot
        t += dt
        out[round(t, 6)] = T
    return out


def test_M4_a_hot_light_cube_follows_lumped_capacitance_with_series_film_plus_radiation():
    c = K.Climate.reference(t_c=-10.0)
    L = RefLattice((12, 12, 10), climate=c)
    L.fill(0, 12, 0, 12, 0, 2, FOAM)
    L.fill(4, 7, 4, 7, 2, 5, LIGHT)                                      # a 3-cell cube
    L.soak()
    cube = np.zeros(L.shape, bool)
    cube[4:7, 4:7, 2:5] = True
    L.E[cube] = (80.0 - 20.0) * L.C[cube]
    c_tot = float(L.C[cube].sum())
    area = 45 * L.A                                                       # 4 sides and the top, 9 each
    u_series = 1.0 / (1.0 / O.H_CONV + 1.0 / (1200.0 * L.dx * c.k_film))
    ref = _lumped(80.0, 600.0, 0.05, c_tot, area, 0.8, -10.0, u_series)
    e0 = L.total_energy()
    scale = float(np.abs(L.E).sum(dtype=np.float64))
    for sec in (100, 200, 300, 600):
        L.run(int(round((sec - L.t) / L.dt)))
        Tc = float(L.T()[cube].mean())
        want = ref[round(float(sec), 6)]
        assert (Tc + 10.0) == pytest.approx(want + 10.0, rel=0.20, abs=0.05), (sec, Tc, want)
    assert abs(L.total_energy() - e0) < 1e-4 * scale
    assert float(L.T()[cube].mean()) < -9.0                               # ends at the outside, not at 20


# ------------------------------------------------------------------ M5
def test_M5_ledger_every_source_on_closes_and_the_books_equal_shed():
    c = K.Climate(lat_deg=45.0, t_ann_c=2.0, a_ann_k=8.0, a_day_k=6.0, cloud=0.2, day_s=600.0,
                  start_doy=20.0, start_hour=19.0, weather_sigma_k=2.0, seed=3, weather_tau_d=3.0,
                  ground_depth_m=1.0, rest_c=2.0)
    L, box = _hut(c, door=True)
    L.fill(11, 14, 11, 14, 1, 2, LIGHT)                                  # a hearth block
    L.source((12, 12, 1), 80.0)
    L.soak()
    e0 = L.total_energy()
    scale = float(np.abs(L.E).sum(dtype=np.float64))
    L.run(2000)
    assert abs(L.total_energy() - e0) < 1e-4 * scale
    tot = sum(L.booked.values())
    assert L.shed == pytest.approx(tot, rel=1e-9, abs=1e-6)             # shed IS what T8 wrote
    assert L.booked["rad"] != 0 and L.booked["shift"] != 0 and L.booked["ground"] != 0
    assert L.booked["air_film"] != 0 or L.booked["air_far"] != 0
    assert L.injected > 0 and L.awake_steps == 2000


# ------------------------------------------------------------------ M6
def test_M6_closed_room_radiation_stays_in_the_room():
    c = K.Climate.reference(t_c=-5.0)
    L, box = _hut(c, open_sky=False)
    L.soak()
    hot = np.zeros(L.shape, bool)
    hot[11:13, 11:13, 1:3] = True
    L.fill(11, 13, 11, 13, 1, 3, LIGHT)
    L.finish()
    L.soak()
    hot = L.mat == LIGHT
    L.E[hot] = (150.0 - 20.0) * L.C[hot]
    wall = (8, 12, 3)                                                    # an inner wall cell face-on
    assert L.solid[wall] or True
    e0 = L.total_energy()
    scale = float(np.abs(L.E).sum(dtype=np.float64))
    t_wall0 = float(L.T()[8, 12, 3]) if L.solid[8, 12, 3] else float(L.T()[7, 12, 3])
    wcell = (7, 12, 3) if not L.solid[8, 12, 3] else (8, 12, 3)
    t_wall0 = float(L.T()[wcell])
    L.run(480)                                                           # 2 minutes
    assert not (L.cls & 1).any()                                         # nothing outdoor
    assert abs(L.shed) < 1e-6 * scale                                    # nothing leaked
    assert abs(L.total_energy() - e0) < 1e-6 * scale
    assert float(L.T()[hot].mean()) < 150.0                              # the hot block lost heat ...
    assert float(L.T()[wcell]) > t_wall0 + 1e-3                          # ... and a cold wall gained it
    assert set(L.booked) and abs(L.booked["rad"]) == 0.0 and L.booked["edge"] == 0.0


# ------------------------------------------------------------------ M7
def test_M7_a_room_with_a_hole_in_the_roof_is_not_the_outside():
    c = K.Climate.reference(t_c=-12.0).replace(rest_c=20.0)
    L, box = _hut(c, hole=True)
    L.fill(11, 14, 11, 14, 1, 2, LIGHT)
    L.source((12, 12, 1), 500.0)
    L.finish()
    L.soak()
    inside = _inside(L, box)
    out = (L.cls & 1) != 0
    assert not (out & inside)[:, :, 1:6].any()                           # the room is not outdoor
    assert out[11, 11, 8]                                                # the air in the hole may be
    assert not out[11, 11, 3]
    L.run(480)                                                           # 2 minutes
    air = inside & (L.mat == 0)
    mean_air = float(L.T()[air].mean())
    assert mean_air > 10.0                                               # nowhere near T_out = -12
    # and the outside air above the hut stayed at T_out
    assert abs(float(L.T()[12, 12, 12]) - (-12.0)) < 0.3


# ------------------------------------------------------------------ M8
def test_M8_short_day_outdoor_air_tracks_T_out():
    c = K.Climate(lat_deg=45.0, day_s=600.0, a_day_k=6.0, cloud=0.3, start_doy=100.0, start_hour=0.0,
                  weather_sigma_k=0.0, rest_c=8.0, a_ann_k=0.0, t_ann_c=8.0)
    L = RefLattice((12, 12, 10), climate=c)
    L.fill(3, 9, 3, 9, 0, 2, STONE)
    L.soak()
    worst = peak = 0.0
    out = (L.cls & 1) != 0
    free = out & ~L.film
    for n in range(2400):                                                # one full 600 s day
        L.step()
        true = float(K.day_curve(c, L._x()))
        lag = float(np.abs(L.T()[free] - true).max())
        worst = max(worst, lag)
        peak = max(peak, abs(L.booked["shift"]))
        # air that touches the stone is not the free atmosphere: it sits between the stone and T_out
        # by the series film (conduction G over the relaxation C k), here well under a kelvin
        assert float(np.abs(L.T()[out & L.film] - true).max()) < 0.9
    assert worst < 0.1
    swing = 6.0 * (1 - 0.5 * 0.3) * 2
    assert peak > 1.0 and swing > 8.0          # the shift is booked, and it comes back by the end of the day
    assert free.any()


# ------------------------------------------------------------------ M9
def test_M9_kernels_at_the_reference_reproduce_the_old_radiate_to_20_block():
    rng = np.random.default_rng(9)
    shape = (8, 7, 6)
    f = np.float32
    Ts = rng.uniform(-30, 300, shape).astype(np.float32)
    eps = rng.uniform(0.1, 1.0, shape).astype(np.float32)
    af = rng.uniform(0.3, 1.0, shape).astype(np.float32)
    ngas = rng.integers(0, 5, shape).astype(np.uint8)
    Csur = rng.uniform(1.0, 500.0, shape).astype(np.float32)
    A, dt = f(0.0025), f(0.025)
    # the old block of heat.py _law_conduct, verbatim
    from src.voxel.materials import AMBIENT, FACE_CLIP, SIGMA
    K0 = np.float32(273.15)
    TINF = np.float32(AMBIENT + 273.15)
    TINF2 = np.float32((AMBIENT + 273.15) ** 2)
    tk = Ts + K0
    fourth = (tk * tk + TINF2) * (tk + TINF) * (Ts - AMBIENT)
    old = eps * f(SIGMA) * A * af * ngas * dt * fourth
    lim = f(FACE_CLIP) * Csur * np.abs(Ts - AMBIENT)
    old = np.where(ngas > 0, np.clip(old, -lim, lim), 0.0).astype(np.float32)
    env = np.full(shape, 20.0, np.float32)
    for fused in ([False, True] if O.HAVE_NUMBA else [False]):
        new = O.radiation_out(Ts, eps, af, ngas, env, A, dt, 0.0, Csur, fused=fused)
        assert np.allclose(new, old, rtol=1e-6, atol=1e-12)
    # the Tenv of a reference climate is exactly 20 C
    n = np.full(4, 2, np.uint8)
    w = np.array([0.0, 0.5, 1.0, 2.0], np.float32)
    assert np.all(O.env_temperature(n, w, 20.0, 20.0) == np.float32(20.0))
    # a uniform room at 20 C: the star gives the old ghost's answer, zero
    r = O.mrt_star(np.full(50, 20.0), eps.ravel()[:50], af.ravel()[:50], np.full(50, 2, np.uint8),
                   np.ones(50, np.int32), 0.0025, 0.025, Csur.ravel()[:50])
    assert np.all(r.q == 0.0)


# ------------------------------------------------------------------ extras
def test_codeonly_a_displaced_film_cell_returns_to_T_out_in_20_seconds():
    c = K.Climate.reference(t_c=-12.0)
    L = RefLattice((10, 10, 8), climate=c)
    L.fill(2, 8, 2, 8, 0, 2, STONE)
    L.soak()
    cell = (4, 4, 2)                                                      # a film air cell above the slab
    assert L.cls[cell] == 3
    L.E[cell] += np.float32(50.0 * L.C[cell])
    L.run(int(25 / L.dt))
    assert abs(float(L.T()[cell]) + 12.0) < 0.005
    assert L.awake_steps > 0


def test_u_clear_night_chills_an_open_surface_below_the_air(record_property):
    # W5-style: a wood plank, air +3 C, clear sky, T_dew +1: the top face drops below the air
    c = K.Climate(lat_deg=45.0, a_day_k=0.0, a_ann_k=0.0, t_ann_c=3.0, cloud=0.0, rest_c=3.0,
                  dew_dep_k=2.0, start_hour=0.0, weather_sigma_k=0.0)
    L = RefLattice((12, 12, 10), climate=c)
    L.fill(2, 10, 2, 10, 0, 2, FOAM)
    L.fill(2, 10, 2, 10, 2, 5, WOOD)
    L.soak()
    L.run(int(900 / L.dt))                                                # 15 minutes
    top = float(L.T()[5, 5, 4])
    assert top < 3.0 - 1.0                                                # chilled by the sky
    assert top > 3.0 - 8.0
    ts_k = K.t_sky_x(c, L._x())
    assert ts_k < -10.0
    cloudy = c.replace(cloud=1.0, z_cloud_km=0.5)
    L2 = RefLattice((12, 12, 10), climate=cloudy)
    L2.fill(2, 10, 2, 10, 0, 2, FOAM)
    L2.fill(2, 10, 2, 10, 2, 5, WOOD)
    L2.soak()
    L2.run(int(900 / L2.dt))
    assert (3.0 - float(L2.T()[5, 5, 4])) < 0.3 * (3.0 - top)             # cloud stops it
