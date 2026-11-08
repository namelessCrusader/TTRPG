"""T2 records: heat and moisture through the thickness (spec 5.1-5.5, tests 2-8, 17).

Tags (plan section 9): test_echo_* = echo (constant fitted to the number), test_codeonly_* =
code against its own table, test_u_* = independent but source unverified. Everything else
is conservation, exactness, a closed form or an arithmetic identity, and may gate a scene.
"""
import numpy as np
import pytest

from src.voxel import record as R
from src.voxel import record_data as rd
from src.voxel import thinheat as th


# ── helpers ──────────────────────────────────────────────────────────────────
def _plate(mat, thick, nn=7, sym=True, T0=850.0, area=1.0):
    thick_a = np.array([thick])
    m = (mat.rho * thick * area) * th.node_fractions(nn, sym)[None, :]
    H = np.full((1, nn), float(mat.H_of_T(T0)))
    gk = th.conduction_gk(np.array([area]), thick_a, nn, sym)
    return H, m, gk


def _run(mat, thick, h, t_end, dt, nn=7, T0=850.0, Tenv=20.0, dT_max=1e9, newton=3, probe=None):
    """Quench a symmetric plate; returns the history of (t, centre T, surface T)."""
    H, m, gk = _plate(mat, thick, nn, True, T0)
    Gp = np.array([[0.0, 2.0 * h]])
    Te = np.array([[Tenv, Tenv]])
    out = []
    t = 0.0
    for _ in range(int(round(t_end / dt))):
        H, Q, ns = th.step(H, m, gk, Gp, Te, mat, dt, dT_max=dT_max, newton=newton)
        t += dt
        T = mat.T_of_H(H[0])
        out.append((t, T[0], T[-1]))
        if probe and probe(T):
            break
    return np.array(out)


def _cross(hist, col, level):
    """Interpolated time a falling column crosses `level`."""
    i = int(np.argmax(hist[:, col] <= level))
    assert hist[i, col] <= level, "never reached"
    t0, t1 = hist[i - 1, 0], hist[i, 0]
    a, b = hist[i - 1, col], hist[i, col]
    return t0 + (a - level) / (a - b) * (t1 - t0)


def _drop_time(thick, h, dt=0.005, mat=None):
    mat = mat or rd.MATERIALS["steel"]
    hist = _run(mat, thick, h, 400.0, dt, dT_max=th.DT_MAX,
                probe=lambda T: T[0] <= 549.0)
    return _cross(hist, 1, 550.0) - _cross(hist, 1, 700.0)


# ── 2. energy exactness ──────────────────────────────────────────────────────
def _random_world(K, seed=0):
    rng = np.random.default_rng(seed)
    recs = R.Records()
    comps = [{"steel": 1.0}, {"wood": 1.0}, {"water": 0.5, "hide": 0.5}, {"wax": 1.0},
             {"glass": 1.0}, {"iron": 0.7, "salt": 0.3}]
    rids = []
    for k in range(K):
        sym = bool(rng.integers(0, 2))
        rid = R.make_thin(recs, R.SECTION if k % 2 else R.SHEET, comps[int(rng.integers(0, len(comps)))],
                          thick=float(rng.uniform(0.0005, 0.006)), width=0.05,
                          length=float(rng.uniform(0.05, 0.4)), T=float(rng.uniform(-10, 700)),
                          nn=7 if sym else 13, sym=sym, host_id=k + 1,
                          ports=dict(h=(float(rng.uniform(10, 3000)) * (not sym), float(rng.uniform(10, 3000))),
                                     T=(float(rng.uniform(0, 80)), float(rng.uniform(0, 80)))))
        rids.append(rid)
    return recs, rids


def test_energy_exact_1000_records_40_ticks():
    recs, rids = _random_world(1000)
    E0 = sum(recs.energy_J(r) for r in rids)
    scale = sum(abs(recs.energy_J(r)) for r in rids)
    Qtot = 0.0
    worst = 0.0
    for tick in range(40):
        recs.begin_tick()
        e_before = np.array([recs.energy_J(r) for r in rids])
        deltas = R.heat_tick(recs, 0.025)
        e_after = np.array([recs.energy_J(r) for r in rids])
        q = {rid: Q for rid, _, Q in deltas}
        Qvec = np.array([q.get(r, 0.0) for r in rids])
        Qtot += Qvec.sum()
        # per record per tick: energy lost = energy through the ports
        worst = max(worst, float(np.max(np.abs((e_before - e_after) - Qvec) / (np.abs(e_before) + 1.0))))
    E1 = sum(recs.energy_J(r) for r in rids)
    assert abs((E0 - E1) - Qtot) / scale < 1e-12
    assert worst < 1e-12, worst


def test_node_sums_reconcile_with_the_host():
    """The host applies each typed delta once; the node sums then agree with it and
    reconcile has nothing to absorb."""
    recs, rids = _random_world(60, seed=3)
    cen = R.Census(len(rids) + 2)
    for rid in rids:
        g, E = recs.row_totals(rid)
        hid = int(recs.get(rid, "host_id"))
        cen.g[hid] += g
        cen.E[hid] += E
    for _ in range(10):
        recs.begin_tick()
        for rid, hid, Q in R.heat_tick(recs, 0.025):
            cen.E[hid] -= Q                                  # the host applies the delta ONCE
        R.reconcile(recs, cen)
        assert not R.records_audit(recs, host=cen), R.records_audit(recs, host=cen)
    assert all(v[0] < 1e-6 for v in recs.meter.values())      # nothing needed fixing beyond round-off


def test_audit_absorb_meter_flags_a_double_applied_flux():
    recs = R.Records()
    rid = R.make_thin(recs, R.SECTION, {"steel": 1.0}, 0.002, 0.04, 0.4, T=800.0,
                      ports=dict(h=(0.0, 3000.0), T=(20.0, 20.0)), host_id=1)
    cen = R.Census(3)
    g, E = recs.row_totals(rid)
    cen.g[1], cen.E[1] = g, E
    recs.begin_tick()
    (_, _, Q), = R.heat_tick(recs, 0.025)
    cen.E[1] -= Q                                            # applied once: books close
    R.reconcile(recs, cen)
    assert R.records_audit(recs, host=cen) == []
    # now the host law ALSO applies the face flux (double-applied): reconcile must absorb it
    recs.begin_tick()
    (_, _, Q2), = R.heat_tick(recs, 0.025)
    cen.E[1] -= 2.0 * Q2
    R.reconcile(recs, cen)
    found = R.records_audit(recs, host=cen)
    assert any(f["kind"] == "absorb" for f in found), found
    # ... and it is NOT flagged when an external law (a burn) really touched the host
    recs.begin_tick()
    R.heat_tick(recs, 0.025)
    cen.E[1] -= 5.0
    recs.touch(1)
    R.reconcile(recs, cen)
    assert not any(f["kind"] == "absorb" for f in R.records_audit(recs, host=cen))


# ── 3. analytic and convergence ──────────────────────────────────────────────
def test_biot_0033_matches_lumped_within_3_percent():
    mat = th.Material.constant("steel_c", 7800.0, 470.0, 45.0)
    L, h = 0.001, 1500.0                                      # 2 mm plate: Bi = h L/k = 0.0333
    assert abs(th.biot(h, L, 45.0) - 0.0333) < 1e-3
    tau = 7800.0 * 470.0 * L / h
    hist = _run(mat, 2 * L, h, 2.0 * tau, 0.025, T0=820.0, Tenv=20.0)
    alpha = 45.0 / (7800.0 * 470.0)
    for frac in (0.5, 1.0, 2.0):
        i = int(round(frac * tau / 0.025)) - 1
        theta_c = (hist[i, 1] - 20.0) / 800.0
        if frac <= 1.0:                    # the lumped model itself drifts ~1 % per time constant at Bi 0.033
            assert abs(theta_c / np.exp(-hist[i, 0] / tau) - 1.0) < 0.03
        # and the exact series, which the solver must match whatever the lumped model does
        assert abs(theta_c / th.plate_theta(0.0, alpha * hist[i, 0] / L ** 2, 0.0333333) - 1.0) < 0.01


def test_biot_083_matches_heisler_series_within_2_percent():
    mat = th.Material.constant("steel_c", 7800.0, 470.0, 45.0)
    L, h = 0.025, 1500.0                                      # 50 mm bar
    Bi = float(th.biot(h, L, 45.0))
    assert abs(Bi - 0.833) < 5e-3
    alpha = 45.0 / (7800.0 * 470.0)
    hist = _run(mat, 2 * L, h, 70.0, 0.25, T0=850.0, Tenv=20.0)
    for Fo in (0.3, 0.6, 1.0):
        i = int(round(Fo * L * L / alpha / 0.25)) - 1
        Fo_i = alpha * hist[i, 0] / L ** 2
        want = th.plate_theta(0.0, Fo_i, Bi)
        got = (hist[i, 1] - 20.0) / 830.0
        assert abs(got / want - 1.0) < 0.02, (Fo, got, want)
        wsurf = th.plate_theta(1.0, Fo_i, Bi)
        gsurf = (hist[i, 2] - 20.0) / 830.0
        assert abs(gsurf / wsurf - 1.0) < 0.03, (Fo, gsurf, wsurf)


def _converge(mat):
    res = {}
    for dt in (0.025, 0.005, 0.001):
        hist = _run(mat, 0.002, 3000.0, 0.5, dt)
        res[dt] = (hist[-1, 1], hist[-1, 2])
    return [res[d][0] for d in (0.025, 0.005, 0.001)], [res[d][1] for d in (0.025, 0.005, 0.001)]


def _first_order(v):
    e25, e5 = abs(v[0] - v[2]), abs(v[1] - v[2])
    assert e25 < 3.0 and e5 < e25                              # within 3 K at 25 ms, monotone
    assert 3.0 < e25 / e5 < 8.0                                # first order: ratio ~ (25-1)/(5-1) = 6


def test_convergence_dt_25_5_1_ms_first_order_centre_and_surface_constant_properties():
    """The spec's quoted centre numbers (584.55 / 582.77 / 582.41 C at 0.5 s) are for the
    constant-property plate (k 45, rho 7800, c 470), 2 mm, h 3000, 850 C start."""
    c, s = _converge(th.Material.constant("steel_c", 7800.0, 470.0, 45.0))
    print("const: centre 25/5/1 ms", [round(float(x), 2) for x in c], "surface", [round(float(x), 2) for x in s])
    assert [round(float(x), 2) for x in c] == [584.55, 582.77, 582.41]
    _first_order(c)
    _first_order(s)                                            # the SURFACE temperature converges too
    assert 564.0 < s[2] < 564.5 and s[0] > s[2]


def test_convergence_with_variable_properties_is_first_order_too():
    c, s = _converge(rd.MATERIALS["steel"])
    print("c(T),k(T): centre 25/5/1 ms", [round(float(x), 2) for x in c], "surface", [round(float(x), 2) for x in s])
    _first_order(c)
    _first_order(s)
    assert 600.0 < c[2] < 800.0


def test_first_tick_after_a_port_change_is_four_substeps():
    recs = R.Records()
    rid = R.make_thin(recs, R.SECTION, {"steel": 1.0}, 0.002, 0.04, 0.4, T=500.0, host_id=1,
                      ports=dict(h=(0.0, 100.0)))
    recs.begin_tick()
    R.heat_tick(recs, 0.025)
    assert R.cache_of(recs)["nsub"][rid] == 1
    assert R.set_port(recs, rid, 1, h=3000.0) is True
    assert R.set_port(recs, rid, 1, h=3000.0) is False         # no change, no wake
    R.heat_tick(recs, 0.025)
    assert R.cache_of(recs)["nsub"][rid] == th.FIRST_TICK_SUBSTEPS
    R.heat_tick(recs, 0.025)
    assert not recs.get(rid, "port_changed")                   # one tick only


def test_substeps_when_a_node_would_move_more_than_50k():
    mat = rd.MATERIALS["steel"]
    H, m, gk = _plate(mat, 0.0012, 7, True, 800.0)            # 0.1 mm node spacing
    Gp = np.array([[0.0, 120000.0]])                          # h 60000: a water film breaking at the wall
    Te = np.array([[20.0, 20.0]])
    Hn, Q, ns = th.step(H, m, gk, Gp, Te, mat, 0.025)
    assert ns[0] > 1
    assert abs(float(((Hn - H) * m).sum() + Q.sum())) / abs(float((H * m).sum())) < 1e-13


# ── 4. enthalpy with c(T), k(T) and a latent plateau ─────────────────────────
def test_codeonly_tables_are_consistent_T_of_H():
    for name, mat in rd.MATERIALS.items():
        T = np.linspace(-60.0, 1500.0, 407)
        assert np.allclose(mat.T_of_H(mat.H_of_T(T)), T, atol=1e-9), name
        assert (np.diff(mat.H_of_T(T)) > 0).all(), name
    assert abs(rd.MATERIALS["water"].H_of_T(20.0)) < 1e-9
    # latent plateau of water: 333.55 kJ/kg between -0.05 and +0.05 C
    w = rd.MATERIALS["water"]
    assert abs(float(w.H_of_T(0.06) - w.H_of_T(-0.06)) - 333.55e3) < 700.0


def test_energy_exact_through_a_latent_plateau_and_variable_k():
    mat = rd.mix_material(np.eye(rd.NS)[rd.SP["water"]] * 0.5 + np.eye(rd.NS)[rd.SP["hide"]] * 0.5)
    H, m, gk = _plate(mat, 0.004, 7, True, -8.0)
    Gp = np.array([[0.0, 2.0 * 400.0]])
    Te = np.array([[30.0, 30.0]])
    E0 = float((H * m).sum())
    out = 0.0
    for _ in range(400):
        H, Q, _ = th.step(H, m, gk, Gp, Te, mat, 0.25)
        out += float(Q.sum())
    assert abs((E0 - float((H * m).sum())) - out) / (abs(E0) + out) < 1e-12
    T = mat.T_of_H(H)
    assert np.allclose(mat.H_of_T(T), H, rtol=0, atol=1e-6)


def _stefan_case(mat, thick, h, dT, Tm, nn, dt, fr):
    near = np.abs(0.5 * (mat.Tk[:-1] + mat.Tk[1:]) - Tm) < 1.0           # the latent segment AT Tm
    i = int(np.flatnonzero(near)[np.argmax(mat.slope[near])])
    Ha, Hb = mat.Hk[i], mat.Hk[i + 1]
    H = np.full((1, nn), Ha)
    m = (mat.rho * thick) * th.node_fractions(nn, True)[None, :]
    gk = th.conduction_gk(np.array([1.0]), np.array([thick]), nn, True)
    Gp = np.array([[0.0, 2.0 * h]])
    Te = np.array([[Tm + dT, Tm + dT]])
    got, t = {}, 0.0
    while len(got) < len(fr) and t < 20000.0:
        H, _, _ = th.step(H, m, gk, Gp, Te, mat, dt)
        t += dt
        phi = np.clip((H[0] - Ha) / (Hb - Ha), 0.0, 1.0)
        mean = float((phi * m[0]).sum() / m[0].sum())
        for f in fr:
            if f not in got and mean >= f:
                got[f] = t
    kl = float(mat.k(np.array([Tm + 1.0]))[0])
    L = thick / 2.0
    want = {f: mat.rho * (Hb - Ha) * (f * L / h + (f * L) ** 2 / (2.0 * kl)) / dT for f in fr}
    return got, want


@pytest.mark.parametrize("comp,thick,h,dT,Tm,nn", [
    ({"water": 0.5, "hide": 0.5}, 0.004, 100.0, 3.0, 0.0, 7),     # frozen, water-soaked 4 mm hide
    ({"water": 0.5, "hide": 0.5}, 0.004, 100.0, 3.0, 0.0, 13),
    ({"wax": 1.0}, 0.004, 100.0, 5.0, 55.0, 7),                   # melting wax plate
    ({"wax": 1.0}, 0.004, 100.0, 5.0, 55.0, 13),
])
def test_stefan_plateau_time_within_5_percent(comp, thick, h, dT, Tm, nn):
    w = np.zeros(rd.NS)
    for k, v in comp.items():
        w[rd.SP[k]] = v
    got, want = _stefan_case(rd.mix_material(w), thick, h, dT, Tm, nn, 0.3, (0.5, 0.9))
    for f in (0.5, 0.9):
        assert abs(got[f] / want[f] - 1.0) < 0.05, (comp, nn, f, got[f], want[f])


def test_stefan_hide_through_the_record_pool_and_heat_tick():
    recs = R.Records()
    rid = R.make_thin(recs, R.SHEET, {"water": 0.5, "hide": 0.5}, 0.004, 0.2, 0.2, T=-0.06,
                      host_id=1, ports=dict(h=(0.0, 100.0), T=(0.0, 3.0)))
    mat = rd.mix_material(np.eye(rd.NS)[rd.SP["water"]] * 0.5 + np.eye(rd.NS)[rd.SP["hide"]] * 0.5)
    near = np.abs(0.5 * (mat.Tk[:-1] + mat.Tk[1:])) < 1.0               # the 0 C latent segment
    i = int(np.flatnonzero(near)[np.argmax(mat.slope[near])])
    Ha, Hb = mat.Hk[i], mat.Hk[i + 1]
    t, dt = 0.0, 0.5
    while t < 4000.0:
        recs.begin_tick()
        R.heat_tick(recs, dt)
        t += dt
        p, r = recs.row(rid)
        phi = np.clip((p.a["H"][r, :7] - Ha) / (Hb - Ha), 0.0, 1.0)
        m = p.a["Y"][r, :7].sum(axis=1)
        if (phi * m).sum() / m.sum() >= 0.9:
            break
    kl = float(mat.k(np.array([1.0]))[0])
    want = mat.rho * (Hb - Ha) * (0.9 * 0.002 / 100.0 + (0.9 * 0.002) ** 2 / (2 * kl)) / 3.0
    assert abs(t / want - 1.0) < 0.06, (t, want)               # one start offset of 0.06 K, dt 0.5 s


# ── 5. time scaling at the SAME h ────────────────────────────────────────────
def test_u_thickness_ratio_6mm_over_2mm_same_h():
    t2, t6 = _drop_time(0.002, 1500.0), _drop_time(0.006, 1500.0)
    print("700->550 C at h 1500: 2 mm %.3f s, 6 mm %.3f s, ratio %.2f" % (t2, t6, t6 / t2))
    assert 2.5 <= t6 / t2 <= 3.5


def test_u_2mm_at_h3000_700_to_550_in_a_quarter_to_six_tenths_second():
    """[U] the c(T), k(T) table and the 900 J/kg K bump are (u); the band is wide on purpose."""
    t = _drop_time(0.002, 3000.0)
    print("2 mm, h 3000: 700->550 C in %.3f s" % t)
    assert 0.25 <= t <= 0.6


def test_u_half_millimetre_is_faster_than_two():
    assert _drop_time(0.0005, 3000.0, dt=0.002) < _drop_time(0.002, 3000.0)


# ── 6. stability ─────────────────────────────────────────────────────────────
def test_implicit_stable_at_25ms_with_point_one_mm_nodes_and_explicit_is_not():
    mat = rd.MATERIALS["steel"]
    thick = 0.0012                                            # 7 nodes over 0.6 mm: dz = 0.1 mm
    assert abs(float(th.node_spacing(thick, 7, True)) - 1e-4) < 1e-12
    hist = _run(mat, thick, 3000.0, 1.0, 0.025, dT_max=1e9)
    assert np.isfinite(hist).all() and (hist[:, 1:] >= 20.0 - 1e-9).all()
    assert (np.diff(hist[:, 1]) <= 1e-9).all() and (np.diff(hist[:, 2]) <= 1e-9).all()   # monotone cooling
    # explicit forward Euler on the same nodes and tick (constant properties)
    alpha, dz, dt = 45.0 / (7800.0 * 470.0), 1e-4, 0.025
    assert dt > dz * dz / (2.0 * alpha) * 10                  # far beyond the explicit limit
    T = np.full(7, 850.0)
    for _ in range(40):
        Tn = T.copy()
        Tn[1:-1] += alpha * dt / dz ** 2 * (T[:-2] - 2 * T[1:-1] + T[2:])
        Tn[0] += 2 * alpha * dt / dz ** 2 * (T[1] - T[0])
        Tn[-1] += 2 * alpha * dt / dz ** 2 * (T[-2] - T[-1]) - 2 * 3000.0 * dt / (7800 * 470 * dz) * (T[-1] - 20.0)
        T = Tn
        if not np.isfinite(T).all() or np.abs(T).max() > 1e9:
            break
    assert (not np.isfinite(T).all()) or np.abs(T).max() > 1e6


# ── 7. species through the thickness ─────────────────────────────────────────
def _drying_setup(nn, X0, Dconst):
    thick, area = 0.004, 1.0
    rho_dry = 950.0
    vol = thick * area * th.node_fractions(nn, True)
    Md = rho_dry * vol
    Wd = Md * X0
    gk = th.conduction_gk(np.array([area]), np.array([thick]), nn, True)
    rd_ = np.full((1, nn), rho_dry)
    return Wd[None, :], Md[None, :], rd_, gk, vol


def test_drying_slab_matches_the_plate_series_and_conserves_water():
    nn, X0, D = 13, 0.3, 2e-10
    Wd, Md, rho_dry, gk, vol = _drying_setup(nn, X0, D)
    L = 0.002
    Gm = np.array([[0.0, 1e6]])
    Xeq = np.zeros((1, 2))
    Hn = np.zeros((1, nn))
    T = np.full((1, nn), 20.0)
    W0 = Wd.sum()
    out_tot = 0.0
    t, dt = 0.0, 5.0
    for Fo in (0.1, 0.3, 0.6, 1.0):
        while D * t / L ** 2 < Fo - 1e-12:
            Wd, Hn, out, _ = th.diffuse_step(Wd, Md, rho_dry, gk, lambda X, T_: np.full_like(X, D), T,
                                             Gm, Xeq, Hn, dt)
            out_tot += float(out.sum())
            t += dt
        mean = float(Wd.sum() / Md.sum()) / X0
        want = th.plate_mean_theta(D * t / L ** 2, math_inf())
        assert abs(mean / want - 1.0) < 0.02, (Fo, mean, want)
    assert abs((W0 - Wd.sum()) - out_tot) / W0 < 1e-12        # water out = boundary flux integral


def math_inf():
    return float("inf")


def test_record_drying_uses_the_D_table_follows_mass_and_thickness():
    def dried(rh, hours=1.0):
        recs = R.Records()
        rid = R.make_thin(recs, R.SHEET, {"hide": 0.7, "water": 0.3}, 0.004, 0.2, 0.2, T=20.0,
                          host_id=1, ports=dict(h=(0.0, 0.0)))
        recs.set(rid, "port_rh", [-1.0, rh])
        recs.set(rid, "port_hm", [1.0, 1e3])                   # near-Dirichlet surface
        w0 = float(recs.get(rid, "Y")[:, rd.SP["water"]].sum())
        th0 = float(recs.get(rid, "thick"))
        gone = 0.0
        for _ in range(int(hours * 3600 / 20)):
            recs.begin_tick()
            for _, _, dm, dE in R.species_tick(recs, 20.0):
                gone += dm
        w1 = float(recs.get(rid, "Y")[:, rd.SP["water"]].sum())
        return recs, rid, w0, w1, gone, th0
    recs, rid, w0, w1, gone, th0 = dried(0.4)
    assert w1 < w0 and abs((w0 - w1) - gone) / w0 < 1e-12
    assert float(recs.get(rid, "thick")) < th0                 # drying shrinks it: geometry from the nodes
    assert not R.records_audit(recs)                           # thickness follows the node volumes
    # a drier air dries it further; the sorption boundary (not an equal fraction) sets the limit
    _, _, _, w_dry, _, _ = dried(0.1)
    _, _, _, w_wet, _, _ = dried(0.8)
    assert w_dry < w1 < w_wet
    # the D table: D at X = 0.3 is more than D at X = 0 (moisture raises D), used by the solver
    assert rd.diffusivity(0.3, 20.0, "hide") > 3.0 * rd.diffusivity(0.0, 20.0, "hide")


def test_moisture_energy_leaves_with_the_water():
    recs = R.Records()
    rid = R.make_thin(recs, R.SHEET, {"hide": 0.7, "water": 0.3}, 0.004, 0.2, 0.2, T=60.0, host_id=1)
    recs.set(rid, "port_rh", [-1.0, 0.2])
    recs.set(rid, "port_hm", [1.0, 1e3])
    e0 = recs.energy_J(rid)
    tot = 0.0
    for _ in range(50):
        recs.begin_tick()
        for _, _, dm, dE in R.species_tick(recs, 20.0):
            tot += dE
    assert abs((e0 - recs.energy_J(rid)) - tot) / abs(e0) < 1e-12


# ── quiet ────────────────────────────────────────────────────────────────────
def test_quiet_when_the_step_cannot_move_float32_energy_and_wakes_on_change():
    recs = R.Records()
    rid = R.make_thin(recs, R.SECTION, {"steel": 1.0}, 0.002, 0.04, 0.4, T=20.0, host_id=1,
                      ports=dict(h=(0.0, 10.0), T=(20.0, 20.0)))
    recs.begin_tick()
    R.heat_tick(recs, 0.025)
    assert recs.get(rid, "quiet")                              # nothing to move
    before = recs.get(rid, "H").copy()
    R.heat_tick(recs, 0.025)
    assert np.array_equal(before, recs.get(rid, "H"))          # skipped
    R.set_port(recs, rid, 1, T=400.0)
    assert not recs.get(rid, "quiet")                          # a port change wakes it
    R.heat_tick(recs, 0.025)
    assert recs.get(rid, "H")[6] > before[6]
    R.heat_tick(recs, 0.025)
    recs.set(rid, "quiet", True)
    R.absorb(recs, rid, 1.0)
    assert not recs.get(rid, "quiet")                          # an absorb wakes it


def test_any_T_of_H_callable_works():
    c, k = 500.0, 30.0
    mat = th.CallableMaterial(lambda H: 20.0 + H / c, k, rho=7000.0)
    ref = th.Material.constant("c", 7000.0, c, k)
    H, m, gk = _plate(ref, 0.004, 7, True, 400.0)
    Gp = np.array([[0.0, 2000.0]])
    Te = np.array([[20.0, 20.0]])
    a, Qa, _ = th.step(H, m, gk, Gp, Te, mat, 0.05)
    b, Qb, _ = th.step(H, m, gk, Gp, Te, ref, 0.05)
    assert np.allclose(a, b, rtol=1e-6, atol=1.0)


def test_codeonly_scales_and_forced_convection():
    assert th.biot(3000.0, 0.001, 45.0) == pytest.approx(0.0667, abs=1e-4)
    assert th.t_diff(0.001, 45.0 / (7800 * 470)) == pytest.approx(0.0815, rel=0.01)
    still = th.h_conv(0.0, 0.05, 30.0, "air")
    swung = th.h_conv(10.0, 0.05, 30.0, "air")
    assert 2.0 < still < 15.0 and swung > 4 * still            # a swung blade cools faster
    assert th.h_conv(0.5, 0.05, 30.0, "water") > 50 * still    # fluid is data, not a branch
    assert th.l_char(0.002 * 0.04 * 0.4, 2 * 0.04 * 0.4) == pytest.approx(0.001, rel=1e-9)


# ── the maximum principle across a latent plateau, spent records, boiling ────
def _extremes(comp, thick, h, nn, sym, T_start, T_env, dt=0.025, secs=2.5):
    w = np.zeros(rd.NS)
    for k, v in comp.items():
        w[rd.SP[k]] = v
    mat = rd.row_material(w / w.sum())
    recs = R.Records()
    rid = R.make_thin(recs, R.SECTION, comp, thick, 0.1, 0.1, T=T_start, nn=nn, sym=sym,
                      ports=dict(h=(h, h), T=(T_env, T_env)))
    p, r = recs.row(rid)
    lo, hi = min(T_start, T_env), max(T_start, T_env)
    worst = 0.0
    E0, Q = recs.energy_J(rid), 0.0
    for t in range(int(secs / dt)):
        recs.begin_tick(t)
        Q += sum(d[2] for d in R.heat_tick(recs, dt))
        T = mat.T_of_H(p.a["H"][r, :nn])
        worst = max(worst, lo - T.min(), T.max() - hi)
    return worst, (recs.energy_J(rid) - E0 + Q) / abs(E0)


@pytest.mark.parametrize("comp,thick,h,nn,T0,Te", [
    ({"hide": .6, "water": .4}, 5e-4, 1000, 13, -30.0, 90.0),     # 6.5 K over, with 3 fixed passes
    ({"hide": .6, "water": .4}, 2e-4, 20000, 13, -30.0, 90.0),    # 469 K over
    ({"water": 1.0}, 2e-4, 50000, 13, -30.0, 90.0),               # 254 K over
    ({"wax": 1.0}, 2e-4, 5000, 13, 90.0, -30.0),                  # 58 K over
    ({"water": 1.0}, 2e-4, 5000, 7, 5.0, -5.0),
])
def test_latent_crossing_stays_inside_the_range_of_the_inputs(comp, thick, h, nn, T0, Te):
    over, eerr = _extremes(comp, thick, h, nn, True, T0, Te)
    assert over < 0.05                                    # no node outside [min, max] of start and bath
    assert abs(eerr) < 1e-12                              # and the books are still exact


@pytest.mark.parametrize("comp,thick,h,nn,T0,Te", [
    ({"water": 1.0}, 1e-3, 20000, 7, -30.0, 90.0),                # first tick gave -73 C from a -30 C start
    ({"hide": .6, "water": .4}, 2e-4, 20000, 13, -30.0, 90.0),    # sweep: 3022 K over
    ({"hide": .6, "water": .4}, 2e-4, 50000, 13, -30.0, 90.0),    # sweep: 437 K over
    ({"water": 1.0}, 5e-4, 5000, 13, -30.0, 90.0),                # sweep: 36.6 K over
    ({"water": 1.0}, 2e-3, 50000, 7, -30.0, 90.0),                # sweep: 43-45 K over
    ({"hide": .6, "water": .4}, 6e-3, 50000, 7, -30.0, 90.0),     # sweep: 9.98 K over
    ({"hide": .6, "water": .4}, 5e-4, 50000, 7, -3.0, 3.0),       # sweep: 36 K over, a 6 K range
])
def test_newton_limit_cycle_across_a_plateau_is_redone_in_shorter_steps(comp, thick, h, nn, T0, Te):
    """The linearised solve can fall into a limit cycle that no pass count releases; the
    row must then be redone in shorter steps, never accepted. The energy books cannot see
    this, so the convergence report is asserted too: nothing may stay unconverged."""
    th.stats(reset=True)
    over, eerr = _extremes(comp, thick, h, nn, True, T0, Te)
    assert over < 0.05
    assert abs(eerr) < 1e-12
    assert th.stats()["unconverged"] == 0


def test_the_plateau_row_that_used_to_hit_the_cap_now_solves_inside_its_budget():
    """The row the old Newton loop could not finish (a 1 mm water film, -30 C start, 90 C bath
    at h 20000): the damped solve finishes it, with no fall-back, and the answer is inside the
    range of the inputs with the books exact."""
    mat = rd.MATERIALS["water"]
    H, m, gk = _plate(mat, 1e-3, 7, True, -30.0)
    Gp = np.array([[0.0, 2.0 * 20000.0]])
    Te = np.full((1, 2), 90.0)
    th.stats(reset=True)
    Hn, Q, ok, _u = th._substep(H, m, gk, Gp, Te, mat, 0.025, th.NEWTON)
    assert ok[0]
    T = mat.T_of_H(Hn[0])
    assert -30.0 - 1e-6 <= T.min() and T.max() <= 90.0 + 1e-6
    s = th.stats()
    assert s["capped"] == 0 and s["solves"] <= th.WORK_MAX
    dE = (m * (Hn - H)).sum() + Q.sum()
    assert abs(dE) <= 1e-12 * abs((m * H).sum() + 1.0) + 1e-9


def test_a_row_that_spends_its_budget_is_redone_in_halves_and_counted():
    """The fall-back: a tiny budget forces it. The row is redone as two half steps (each an
    exact backward-Euler step), the report counts it, and the books stay exact."""
    mat = rd.MATERIALS["water"]
    H, m, gk = _plate(mat, 1e-3, 7, True, -30.0)
    Gp = np.array([[0.0, 2.0 * 20000.0]])
    Te = np.full((1, 2), 90.0)
    th.stats(reset=True)
    Hn, Q = th._robust_substep(H, m, gk, Gp, Te, mat, 0.025, th.NEWTON, 0, 3)
    s = th.stats()
    assert s["capped"] >= 1 and s["bisected"] == 1 and s["unconverged"] == 0
    T = mat.T_of_H(Hn[0])
    assert -30.0 - 1e-6 <= T.min() and T.max() <= 90.0 + 1e-6      # still a valid state
    dE = (m * (Hn - H)).sum() + Q.sum()
    assert abs(dE) <= 1e-12 * abs((m * H).sum() + 1.0) + 1e-9      # exact books through the halves
    # and a budget too small even for the halves: counted as unconverged, books still exact
    th.stats(reset=True)
    Hn, Q = th._robust_substep(H, m, gk, Gp, Te, mat, 0.025, th.NEWTON, 0, 1)
    assert th.stats()["unconverged"] >= 1
    assert abs((m * (Hn - H)).sum() + Q.sum()) <= 1e-12 * abs((m * H).sum() + 1.0) + 1e-9
    # the hard bound on work: at most (2^(BISECT_MAX+1) - 1) budgets per substep
    assert th.stats()["solves"] <= (2 ** (th.BISECT_MAX + 1) - 1) * 1


def test_newton_freezes_converged_rows_so_batch_mates_do_not_matter():
    mat = rd.MATERIALS["water"]
    nn, sym = 13, True
    rows = []
    for thick, h, T0 in ((2e-4, 50000.0, -30.0), (3e-3, 100.0, 60.0)):     # hard row, easy row
        rows.append((_plate(mat, thick, nn, sym, T0), h))
    def run(sel):
        H = np.vstack([rows[i][0][0] for i in sel]); m = np.vstack([rows[i][0][1] for i in sel])
        gk = np.concatenate([rows[i][0][2] for i in sel])
        Gp = np.array([[0.0, 2.0 * rows[i][1]] for i in sel]); Te = np.full((len(sel), 2), 90.0)
        return th.step(H, m, gk, Gp, Te, mat, 0.025)[0]
    alone = run([1])
    together = run([0, 1])
    assert np.array_equal(alone[0], together[1])           # the easy row is bit-identical alone or with the hard one


def test_a_record_burnt_to_nothing_does_not_stop_the_batch_and_is_listed_as_spent():
    recs = R.Records()
    leaf = R.make_thin(recs, R.SECTION, {"wood": 1.0}, 0.004, 0.1, 0.1, T=300.0, host_id=1,
                       ports=dict(h=(20, 20), T=(20, 20)))
    blade = R.make_thin(recs, R.SECTION, {"steel": 1.0}, 0.002, 0.1, 0.1, T=700.0, host_id=2,
                        ports=dict(h=(3000, 3000), T=(20, 20)))
    R.consume(recs, leaf, "wood", 1e9)
    T_before = recs.get(blade, "H").copy()
    recs.begin_tick(1)
    out = R.heat_tick(recs, 0.025)                          # used to raise ValueError on the leaf
    assert [d[0] for d in out] == [blade]                   # the blade still cools
    assert recs.get(blade, "H")[6] < T_before[6]
    assert R.spent(recs) == [leaf]
    assert R.absorb(recs, leaf, 5.0) == 0.0                 # nothing to hold it: refused, not NaN
    kinds = {f["kind"] for f in R.records_audit(recs)}
    assert "spent" in kinds and "corrupt" not in kinds
    recs.kill(leaf)
    assert R.spent(recs) == [] and not R.records_audit(recs)


def test_liquid_water_in_a_hide_holds_at_100_C_until_it_is_boiled_off():
    comp = {"hide": .7, "water": .3}
    w = np.zeros(rd.NS)
    for k, v in comp.items():
        w[rd.SP[k]] = v
    mat = rd.row_material(w)
    recs = R.Records()
    rid = R.make_thin(recs, R.SECTION, comp, 0.004, 0.1, 0.1, T=20.0, nn=7, sym=True,
                      ports=dict(h=(100.0, 100.0), T=(300.0, 300.0)))
    p, r = recs.row(rid)
    tmax = 0.0
    for t in range(int(40 / 0.025)):
        recs.begin_tick(t)
        R.heat_tick(recs, 0.025)
        T = mat.T_of_H(p.a["H"][r, :7])
        tmax = max(tmax, T[0])
        if t * 0.025 in (10.0,) or abs(t * 0.025 - 20.0) < 1e-9:
            assert T[0] < 100.1                                   # the plateau: no superheat while it lasts
    # the latent heat is stored in the node's enthalpy: the plateau holds 0.3 x 2256 kJ/kg
    assert mat.H_of_T(100.05) - mat.H_of_T(99.95) == pytest.approx(0.3 * 2256.4e3, rel=1e-3)
