"""T6 transport in porous solids (wave 13, lane A-T6). Tag convention as in
test_t6_psychro.py: test_u_* physical data, test_echo_*, test_codeonly_*; the rest are
conservation / exactness / closed-form tests and gate scenes."""
import numpy as np
import pytest

from src.voxel import psychro as P
from src.voxel import transport as T


def _chain(N, D=1e-6, dx=0.01, boundary=()):
    """1-D diffusion of a concentration c (psi = c): cap = dx^3, kappa = D."""
    cap = np.full(N, dx ** 3)
    for i in boundary:
        cap[i] = 1e30
    return np.full(N, D), cap, np.ones(N, bool), dx


def test_conserves_in_float32_to_1e6_and_keeps_the_maximum_principle():
    rng = np.random.default_rng(7)
    shape = (7, 6, 5)
    m = rng.uniform(5.0, 12.0, shape).astype(np.float32)
    cap = rng.uniform(4.0, 9.0, shape)
    kap = rng.uniform(1e-6, 1e-3, shape)
    mask = rng.random(shape) < 0.85
    psi = rng.uniform(-0.5, -0.1, shape)
    lo, hi = psi[mask].min(), psi[mask].max()
    tot0 = float(m.sum(dtype=np.float64))
    # a dt well over the stability limit forces substeps
    r = T.potential_flux_step(m, psi, kap, cap, mask, 0.05, 5e5)
    assert r.m.dtype == np.float32 and r.n_sub > 1
    assert abs(float(r.m.sum(dtype=np.float64)) - tot0) / tot0 < 1e-6
    assert np.all(r.m[~mask] == m[~mask])                 # off-mask cells untouched
    psi_new = psi + (r.m.astype(np.float64) - m) / cap
    assert psi_new[mask].min() >= lo - 1e-6 and psi_new[mask].max() <= hi + 1e-6


def test_step_is_symmetric_under_swapping_the_two_cells():
    rng = np.random.default_rng(3)
    m = rng.uniform(5.0, 9.0, 12)
    cap = rng.uniform(4.0, 9.0, 12)
    kap = rng.uniform(1e-5, 1e-3, 12)
    psi = rng.uniform(-0.5, -0.1, 12)
    mask = np.ones(12, bool)
    a = T.potential_flux_step(m, psi, kap, cap, mask, 0.05, 1e4)
    b = T.potential_flux_step(m[::-1].copy(), psi[::-1].copy(), kap[::-1].copy(),
                              cap[::-1].copy(), mask, 0.05, 1e4)
    assert np.allclose(a.m, b.m[::-1], rtol=0, atol=1e-12)
    assert np.allclose(a.flux[0], -b.flux[0][::-1], atol=1e-12)


def test_large_dt_substeps_and_matches_the_fine_step():
    kap, cap, mask, dx = _chain(15)
    psi = np.linspace(1.0, 0.0, 15)
    m = psi * cap
    big = T.potential_flux_step(m, psi, kap, cap, mask, dx, 60.0)
    assert big.n_sub >= 3 and big.dt_done == pytest.approx(60.0)
    assert T.stable_dt(cap, T._rates(psi, kap, [mask[:-1] & mask[1:]], dx)[1]) < 40.0
    fine = m.copy()
    for _ in range(3000):
        fine = T.potential_flux_step(fine, fine / cap, kap, cap, mask, dx, 0.02).m
    assert fine.shape == m.shape
    assert np.allclose(big.m, fine, atol=3e-2 * np.abs(m).max())


def test_matches_erf_semi_infinite_within_2_percent_at_20_nodes():
    N, D, dx = 21, 1e-6, 0.01
    kap, cap, mask, _ = _chain(N, D, dx, boundary=(0,))
    c = np.zeros(N)
    c[0] = 1.0
    t = (5 * dx) ** 2 / (4 * D)
    r = T.potential_flux_step(c * cap, c.copy(), kap, cap, mask, dx, t)
    got = r.m / cap
    got[0] = 1.0
    ref = P.fick_erf(np.arange(N) * dx, t, D, 0.0, 1.0)
    assert np.abs(got - ref).max() < 0.02


def test_matches_the_crank_slab_series_within_2_percent_at_20_nodes():
    N, D, dx = 21, 1e-6, 0.01
    kap, cap, mask, _ = _chain(N, D, dx, boundary=(0, N - 1))
    c = np.ones(N)
    c[0] = c[-1] = 0.0
    l = 10 * dx
    t = 0.15 * l * l / D
    r = T.potential_flux_step(c * cap, c.copy(), kap, cap, mask, dx, t)
    got = (r.m / cap)[1:-1]
    ref = P.crank_slab((np.arange(N) - 10) * dx, t, D, l, 1.0, 0.0)[1:-1]
    assert np.abs(got - ref).max() < 0.02
    # and the mean (the drying curve)
    mean = got.sum() / 20.0                           # trapezoid, both faces at 0
    assert abs(mean - P.crank_slab_mean(t, D, l, 1.0, 0.0)) < 0.02


def test_u_diffusivity_factor_from_20_to_40_C_at_18_kJ():
    f = float(T.arrhenius(1.0, 18000.0, 40.0, 20.0))
    assert abs(f - 1.6) < 0.02
    assert float(T.arrhenius(2.0, 18000.0, 20.0, 20.0)) == 2.0


def test_host_D_rows():
    D20 = float(T.host_D("NaCl", "free", 1.0, 1.0, 20.0))
    D40 = float(T.host_D("NaCl", "free", 1.0, 1.0, 40.0))
    assert 1.5 < D40 / D20 < 1.7
    # Millington-Quirk: half-full pores, porosity 0.4
    mq = float(T.host_D("NaCl", "stone", 0.2, 0.4, 25.0))
    assert mq == pytest.approx(1.61e-9 * 0.2 ** (10 / 3) / 0.4 ** 2, rel=1e-9)
    meat = float(T.host_D("NaCl", "meat", 0.7, 0.8, 25.0))
    assert 1e-10 < meat < 3e-10                                   # 1-3e-10 in meat


def test_two_unlike_porous_slabs_at_equal_a_w_exchange_nothing():
    N = 20
    aw = 0.65
    wood = dict(wm=0.065, C=9.0, K=0.78)
    hide = dict(wm=0.09, C=14.0, K=0.88)
    dry = np.where(np.arange(N) < N // 2, 75.0, 40.0)
    w = np.where(np.arange(N) < N // 2, float(P.gab_w(aw, **wood)), float(P.gab_w(aw, **hide)))
    m = w * dry                                           # grams of water
    assert abs(w[0] - w[-1]) > 0.01                       # the contents differ ...
    def psi_of_m(mm):
        ww = mm / dry
        left = np.log(P.gab_aw(ww, **wood))
        right = np.log(P.gab_aw(ww, **hide))
        return np.where(np.arange(N) < N // 2, left, right)
    psi = psi_of_m(m)
    kap = np.full(N, 2e-5)
    cap = np.full(N, 7.0)
    r = T.potential_flux_step(m, psi, kap, cap, np.ones(N, bool), 0.05, 1e5,
                              psi_of_m=psi_of_m)
    flux = max(np.abs(f).max() for f in r.flux) / 1e5     # g/s
    assert flux < 1e-12
    assert np.abs(r.m - m).max() < 1e-9
    # whereas driving by content would have moved water
    assert abs(w[N // 2 - 1] - w[N // 2]) * dry.min() > 0.4


def test_float32_bite_rule_wood_scale_drying_within_5_percent_of_float64():
    N, dx = 20, 0.05
    cap = np.full(N, 7.5)
    kap = np.full(N, 1.87e-5)                     # D_app rho_dry dw/da at wood scale
    mask = np.ones(N, bool)
    psi0 = np.where(np.arange(N) < N // 2, -0.30, -0.20)
    m0 = np.full(N, 9.0)                          # 9 g: float32 ulp 9.5e-7
    assert float(T.ulp(9.0)) == pytest.approx(9.5e-7, rel=0.02)
    Ttot = 4e5
    # float64 reference with big stable steps
    m, t = m0.copy(), 0.0
    while t < Ttot:
        m = T.potential_flux_step(m, psi0 + (m - m0) / cap, kap, cap, mask, dx, 2e4).m
        t += 2e4
    ref = m - m0
    assert abs(ref[N // 2]) > 0.02
    # float32, one-second steps: the bite rounds away to nothing
    naive = m0.astype(np.float32)
    for _ in range(1500):
        psi = psi0 + (naive.astype(np.float64) - m0) / cap
        naive = T.potential_flux_step(naive, psi, kap, cap, mask, dx, 1.0).m
    assert np.abs(naive.astype(np.float64) - m0).max() == 0.0
    # float32 with the bite clock: bites of at least 16 ulp
    m32, t, n, bites = m0.astype(np.float32), 0.0, 0, []
    while t < Ttot:
        psi = psi0 + (m32.astype(np.float64) - m0) / cap
        one = T.potential_flux_step(m32.astype(np.float64), psi, kap, cap, mask, dx, 1.0)
        f = max(np.abs(x).max() for x in one.flux)
        nt = float(T.bite_clock(f, T.ulp(m32.max()), 3600))
        bites.append(f * nt / float(T.ulp(m32.max())))
        m32 = T.potential_flux_step(m32, psi, kap, cap, mask, dx, nt).m
        t += nt
        n += 1
    assert min(bites[:-1]) >= 15.9
    d = m32.astype(np.float64) - m0
    assert np.abs(d - ref).max() / np.abs(ref).max() < 0.05


def test_bite_clock_rules():
    u = float(T.ulp(9.0))
    assert int(T.bite_clock(16 * u / 100.0, u, 3600)) == 100
    assert int(T.bite_clock(0.0, u, 3600)) == 3600                # settled: sleeps
    assert int(T.bite_clock(1e3, u, 3600)) == 1                   # big flux: every tick
    assert int(T.bite_clock(1e-30, u, 3600)) == 3600              # capped at one hour
    n = T.bite_clock(np.array([1e-7, 1e-6, 0.0]), u, 3600)
    assert n.dtype.kind == "i" and n[0] > n[1] > 0


def test_salt_rides_the_water_flux_and_is_conserved():
    N = 20
    dx = 0.05
    cap = np.full(N, 7.5)
    cap[0] = 1e3
    kap = np.full(N, 2e-4)
    mask = np.ones(N, bool)
    psi = np.full(N, -0.10)
    psi[0] = -0.60                                    # the drying surface pulls water out
    m = np.full(N, 9.0, np.float32)
    m[0] = 50.0
    s = np.zeros((N, 1), np.float32)
    s[1:, 0] = 0.9                                    # salty pore water inside
    tot_s = float(s.sum(dtype=np.float64))
    tot_m = float(m.sum(dtype=np.float64))
    r = T.potential_flux_step(m, psi, kap, cap, mask, dx, 3e4, solutes=s)
    assert abs(float(r.s.sum(dtype=np.float64)) - tot_s) / tot_s < 1e-6
    assert abs(float(r.m.sum(dtype=np.float64)) - tot_m) / tot_m < 1e-6
    assert float(r.s[0, 0]) > 0.0                      # salt arrived at the surface
    assert float(r.s[1, 0]) < 0.9                      # and left the cell beside it
    assert np.all(r.s >= 0)
    # the same run with no water flux moves no salt
    r0 = T.potential_flux_step(m, np.full(N, -0.10), kap, cap, mask, dx, 3e4, solutes=s)
    assert np.allclose(r0.s, s, atol=1e-6)


def test_salt_follows_the_donor_concentration_not_the_average():
    m = np.array([10.0, 10.0])
    s = np.array([[1.0], [0.0]])                      # salty donor on the left
    r = T.potential_flux_step(m, np.array([0.0, -1.0]), np.array([1e-3, 1e-3]),
                              np.array([10.0, 10.0]), np.ones(2, bool), 0.05, 100.0,
                              solutes=s)
    moved_w = r.flux[0][0]
    assert moved_w > 0
    assert float(r.s[1, 0]) == pytest.approx(moved_w * 0.1, rel=0.2)   # ~c_donor, falling


def test_u_wood_interior_resistance_is_about_23000_s_per_m():
    rho = float(P.rho_sat(20.0))
    R = float(T.r_int(0.05, 2e-10, 500.0, 0.187, rho))
    assert abs(R / 23000.0 - 1) < 0.10
    assert 1200.0 / 10.0 < R / 100                                    # 1/h_m = 120 s/m
    # with the GAB slope at 10 % moisture the answer stays in the band
    dwda = float(P.gab_dwda(float(P.gab_aw(0.10))))
    assert abs(float(T.r_int(0.05, 2e-10, 500.0, dwda, rho)) / 23000.0 - 1) < 0.12


def test_series_resistance_adds():
    assert float(T.series_resistance(120.0, 23000.0)) == 23120.0
    assert float(T.harmonic(2.0, 2.0)) == 2.0
    assert float(T.harmonic(0.0, 5.0)) == 0.0


def test_inconsistent_linearisation_never_drives_water_negative():
    from src.voxel import transport as T
    for seed in range(6):
        rng = np.random.default_rng(seed)
        shp = (8, 7, 6)
        m = rng.uniform(0.0, 1.0, shp).astype(np.float32)
        m[rng.random(shp) < 0.1] = 0.0                       # some empty cells
        psi = rng.uniform(-3.0, 0.0, shp)
        kap = rng.uniform(0.5, 2.0, shp)
        cap = rng.uniform(0.5, 5.0, shp)
        sol = rng.uniform(0.0, 0.1, shp + (1,)).astype(np.float32) * (m[..., None] > 0)
        mask = np.ones(shp, bool)
        st = T.potential_flux_step(m, psi, kap, cap, mask, 0.05, 50.0, solutes=sol)
        assert st.m.min() >= 0.0
        assert np.all(np.isfinite(st.m)) and np.all(np.isfinite(st.s)) and st.dt_done == pytest.approx(50.0)
        assert st.s.min() >= -1e-7
        assert float(st.m.sum(dtype=np.float64)) == pytest.approx(
            float(m.sum(dtype=np.float64)), rel=1e-5)
