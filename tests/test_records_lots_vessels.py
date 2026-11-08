"""T2 LOTS, VESSELS, CONDUITS (src/voxel/lots.py, src/voxel/vessel.py):
float/sink in brine, winnow, thresh, sieve, exact integer splits; layered cup
contents, drain, weir spill, pipe flow and the siphon. Stand-alone: no World.

Tags: test_echo_* (fitted), test_codeonly_* (code against its own table or
formula), test_u_* (independent, source unverified).
"""
import math

import numpy as np
import pytest

from src.voxel import lots as L
from src.voxel import vessel as V


def _seeds(n=1000, lo=1000.0, hi=1250.0, size=3.5e-3, m=3e-5, **kw):
    """n seeds with densities stratified uniformly over [lo, hi] kg/m3."""
    d = lo + (hi - lo) * (np.arange(n) + 0.5) / n
    return L.make_lot(np.full(n, m), d, np.full(n, size), **kw)


# ── float and sink ────────────────────────────────────────────────────────────
def test_codeonly_brine_densities_are_the_crc_table():
    assert L.brine_density(10) == pytest.approx(1.071)
    assert L.brine_density(20) == pytest.approx(1.148)
    assert L.brine_density(26) == pytest.approx(1.199)
    assert L.brine_density(0) < 1.0 < L.brine_density(5)
    assert L.brine_density(15) == pytest.approx(0.5 * (1.071 + 1.148))        # linear between


def test_brine_float_shares_at_10_20_26_percent():
    """Seeds spread uniformly over 1000-1250 kg/m3: the float share is the
    integral of the density distribution below the brine's density."""
    lot = _seeds(1000)
    for pct, rho in ((10, 1.071), (20, 1.148), (26, 1.199)):
        floaters, sinkers = L.float_split(lot, rho * 1000.0)
        want = (rho * 1000.0 - 1000.0) / 250.0
        assert floaters["n"] / 1000 == pytest.approx(want, abs=1e-3)
        assert floaters["n"] + sinkers["n"] == 1000                           # counts exact
        assert np.all(floaters["density"] < rho * 1000.0)
        assert np.all(sinkers["density"] >= rho * 1000.0)
        assert math.fsum([L.total_mass(floaters), L.total_mass(sinkers)]) == pytest.approx(
            L.total_mass(lot), rel=1e-15)
    assert L.float_share(lot, 1071.0) < L.float_share(lot, 1148.0) < L.float_share(lot, 1199.0)
    # the sinkers are the ones that sprout: the denser ones
    f, s = L.float_split(lot, 1148.0)
    assert s["density"].min() > f["density"].max() - 1e-9


def test_histogram_mode_floats_the_same_share_and_masses_add():
    rng = np.random.default_rng(1)
    n = 6000
    d = 1000.0 + 250.0 * rng.random(n)
    lot = L.make_lot(rng.random(n) * 1e-5 + 2e-5, d, 3e-3 + 1e-3 * rng.random(n))
    assert lot["mode"] == "hist" and lot["n"] == n
    assert int(lot["counts"].sum()) == n
    for rho in (1071.0, 1148.0, 1199.0):
        f, s = L.float_split(lot, rho)
        assert f["n"] + s["n"] == n
        assert f["n"] / n == pytest.approx(float((d < rho).mean()), abs=0.01)
        assert math.fsum(np.concatenate([f["mass"].ravel(), s["mass"].ravel()])) == \
            pytest.approx(L.total_mass(lot), rel=1e-14)
        assert np.all(f["counts"] >= 0) and np.all(s["counts"] >= 0)


def test_per_item_and_histogram_agree_on_counts_at_a_bin_edge():
    rng = np.random.default_rng(2)
    n = 5000
    d = 1000.0 + 250.0 * rng.random(n)
    sz = 3e-3 + 1e-3 * rng.random(n)
    items = L.make_lot(np.full(n, 1e-5), d, sz, hist=False)
    hist = L.to_hist(items)
    assert items["mode"] == "items" and hist["mode"] == "hist"
    assert hist["n"] == items["n"] == n
    edge = float(hist["dens_edges"][7])                       # a bin edge: no straddle
    fi, _ = L.float_split(items, edge)
    fh, _ = L.float_split(hist, edge)
    assert fi["n"] == fh["n"]
    # sieve at a size edge likewise
    mesh = float(hist["size_edges"][5])
    pi, _ = L.sieve(items, mesh)
    ph, _ = L.sieve(hist, mesh)
    assert pi["n"] == ph["n"]


def test_1000_integer_splits_are_exact():
    rng = np.random.default_rng(3)
    for _ in range(1000):
        total = int(rng.integers(0, 5000))
        w = rng.random(int(rng.integers(1, 12))) + 1e-3
        parts = L.largest_remainder(total, w)
        assert parts.sum() == total and np.all(parts >= 0)
        assert np.all(np.abs(parts - total * w / w.sum()) < 1.0)
    assert list(L.largest_remainder(10, [1, 1, 1])) == [4, 3, 3]            # ties: lowest first
    assert L.largest_remainder(5, [0, 0]).sum() == 0
    # a lot split by count: items taken + items left == the lot, any k
    lot = _seeds(1000)
    for k in (0, 1, 333, 999, 1000):
        a, b = L.split_count(lot, k)
        assert a["n"] == k and b["n"] == 1000 - k
        assert math.fsum(np.concatenate([a["mass"], b["mass"]])) == L.total_mass(lot)
    hist = L.to_hist(lot)
    for k in (0, 7, 500, 1000):
        a, b = L.split_count(hist, k)
        assert a["n"] == k and a["n"] + b["n"] == 1000
        assert np.all(a["counts"] + b["counts"] == hist["counts"])
    # deterministic: the same k picks the same items
    a1, _ = L.split_count(lot, 100)
    a2, _ = L.split_count(lot, 100)
    assert np.array_equal(a1["id"], a2["id"])


def test_repeated_splits_and_merge_keep_count_and_mass():
    lot = _seeds(1000)
    rest, taken = lot, []
    for k in (100, 250, 3, 400):
        t, rest = L.split_count(rest, k)
        taken.append(t)
    assert rest["n"] == 1000 - 753
    back = rest
    for t in taken:
        back = L.merge(back, t)
    assert back["n"] == 1000
    assert math.fsum(back["mass"]) == pytest.approx(L.total_mass(lot), rel=1e-15)
    hist = L.to_hist(lot)
    a, b = L.split_count(hist, 400)
    m = L.merge(a, b)
    assert np.array_equal(m["counts"], hist["counts"])
    assert np.allclose(m["mass"], hist["mass"], rtol=1e-15)


def test_owned_counts_grams_and_integers():
    lot = _seeds(1000, species="wheat")
    o = L.owned(lot)
    assert o["count"] == 1000 and o["grams"]["wheat"] == pytest.approx(30.0)


# ── winnow ────────────────────────────────────────────────────────────────────
def test_u_terminal_speeds_wheat_chaff_and_stokes():
    """Wheat kernel (3.5 mm, 1300 kg/m3): about 10.6 m/s (c). Chaff: 2-5 m/s
    (u). A fine grain follows Stokes."""
    wheat = float(L.terminal_speed(1300.0, 3.5e-3))
    chaff = float(L.terminal_speed(200.0, 4e-3))
    assert wheat == pytest.approx(10.6, rel=0.03)
    assert 2.0 < chaff < 5.0 < wheat
    r = 5e-6
    v = float(L.terminal_speed(1300.0, 2 * r))
    stokes = 2 * (1300.0 - L.RHO_AIR) * L.G * r * r / (9 * L.MU_AIR)
    assert v == pytest.approx(stokes, rel=0.01)
    assert L.RHO_AIR * v * 2 * r / L.MU_AIR < 1.0
    # Newton at high Re: sqrt(4 g d (rho_p - rho_f) / (3 Cd rho_f)), Cd 0.44
    d = 0.02
    newton = math.sqrt(4 * L.G * d * (2000 - L.RHO_AIR) / (3 * 0.44 * L.RHO_AIR))
    assert float(L.terminal_speed(2000.0, d)) == pytest.approx(newton, rel=0.02)


def test_winnow_landing_distance_against_a_trajectory_integral():
    u, H = 3.0, 1.5
    for rho, d in ((1300.0, 3.5e-3), (200.0, 4e-3), (700.0, 2e-3)):
        vt = float(L.terminal_speed(rho, d))
        x_law = float(L.landing_distance(vt, u, H))
        x_int = L.landing_distance_integral(vt, u, H)
        assert x_int == pytest.approx(x_law, rel=0.01)
    # the spread: light chaff lands several times further than wheat
    xs_w = float(L.landing_distance(L.terminal_speed(1300.0, 3.5e-3), u, H))
    xs_c = float(L.landing_distance(L.terminal_speed(200.0, 4e-3), u, H))
    assert xs_c > 2.0 * xs_w
    assert xs_w == pytest.approx(0.42, abs=0.03)


def test_winnow_is_smooth_horizontal_wind_has_no_threshold():
    n = 600
    rho = np.linspace(200.0, 1300.0, n)
    lot = L.make_lot(np.full(n, 3e-5), rho, np.full(n, 3.5e-3), hist=False)
    x = L.landing_distance(L.terminal_speed(lot["density"], lot["size"]), 3.0, 1.5)
    assert np.all(np.diff(x) < 0)                         # monotone: lighter lands further
    assert np.all(np.isfinite(x)) and np.all(x > 0)       # every item lands; none "carried off"
    steps = np.abs(np.diff(x))
    assert steps.max() < 0.05 * (x.max() - x.min())       # no cliff
    # bins by distance: all items accounted, heavy ones nearest
    edges = np.array([0.0, 0.5, 1.0, 2.0, 100.0])
    bins = L.winnow(lot, 3.0, 1.5, edges)
    assert sum(b["n"] for b in bins) == n
    mean_rho = [float(b["density"].mean()) for b in bins if b["n"]]
    assert mean_rho == sorted(mean_rho, reverse=True)
    assert math.fsum(L.total_mass(b) for b in bins) == pytest.approx(L.total_mass(lot), rel=1e-15)
    # the hard rule belongs to a VERTICAL updraft only
    vt = L.terminal_speed(lot["density"], lot["size"])
    carried = L.updraft_carried(vt, 5.0)
    assert 0 < carried.sum() < n and np.all(carried == (vt < 5.0))
    # histogram mode winnows to the same counts in total
    hb = L.winnow(L.to_hist(lot), 3.0, 1.5, edges)
    assert sum(b["n"] for b in hb) == n


# ── thresh and sieve ──────────────────────────────────────────────────────────
def test_thresh_detaches_only_below_the_blow_stress():
    n = 400
    s = np.linspace(1e5, 9e5, n)
    lot = L.make_lot(np.full(n, 3e-5), np.full(n, 1300.0), np.full(n, 3e-3), strength=s)
    off, kept = L.thresh(lot, 4e5)
    assert off["n"] + kept["n"] == n
    assert np.all(off["strength"] < 4e5) and np.all(kept["strength"] >= 4e5)
    assert L.thresh(lot, 1e5)[0]["n"] == 0                  # nothing weaker than the weakest
    assert L.thresh(lot, 1e7)[1]["n"] == 0
    # histogram mode reads each bin's mean strength (approximate): when strength
    # follows size the bins resolve it
    sz = np.linspace(2e-3, 5e-3, n)
    lot2 = L.make_lot(np.full(n, 3e-5), np.full(n, 1300.0), sz, strength=s)
    off2, _ = L.thresh(lot2, 4e5)
    oh, kh = L.thresh(L.to_hist(lot2), 4e5)
    assert oh["n"] + kh["n"] == n and abs(oh["n"] - off2["n"]) <= n // 16


def test_sieve_by_size_against_a_mesh():
    n = 500
    sz = np.linspace(1e-3, 6e-3, n)
    lot = L.make_lot(np.full(n, 1e-5), np.full(n, 1300.0), sz)
    passed, kept = L.sieve(lot, 3e-3)
    assert np.all(passed["size"] < 3e-3) and np.all(kept["size"] >= 3e-3)
    assert passed["n"] + kept["n"] == n
    assert math.fsum([L.total_mass(passed), L.total_mass(kept)]) == L.total_mass(lot)


# ── vessel: layers, pour, spill, leak ─────────────────────────────────────────
def _cup(area=5e-3, lip=0.10):
    v = V.new_vessel(area, lip, lip_w=0.05, parcel_ml=1.0)
    V.add_layer(v, V.make_layer({"water": 120.0}, rho=1.0, T=V.T_REF))
    V.add_layer(v, V.make_layer({"oil": 60.0}, rho=0.92, T=V.T_REF))
    return v


def test_vessel_denser_layers_lie_lower_and_same_kind_merges():
    v = _cup()
    assert [list(l["sp"]) for l in v["layers"]] == [["water"], ["oil"]]
    V.add_layer(v, V.make_layer({"salt water": 30.0}, rho=1.1))
    assert [list(l["sp"]) for l in v["layers"]] == [["salt water"], ["water"], ["oil"]]
    V.add_layer(v, V.make_layer({"water": 10.0}, rho=1.0, T=V.T_REF))
    assert len(v["layers"]) == 3 and V.total_ml(v) == pytest.approx(220.0)


def test_pour_takes_the_top_layer_first_and_ml_is_exact():
    v = _cup()
    before = V.owned(v)
    frag = V.pour(v, 70.0)
    assert [list(f["sp"]) for f in frag] == [["oil"], ["water"]]              # oil first
    assert frag[0]["sp"]["oil"] == 60.0 and frag[1]["sp"]["water"] == pytest.approx(10.0)
    # every species adds back exactly
    for sp in ("water", "oil"):
        got = math.fsum([l["sp"].get(sp, 0.0) for l in v["layers"]] +
                        [f["sp"].get(sp, 0.0) for f in frag])
        assert got == before["ml"][sp]
    assert sum(V.layer_ml(f) for f in frag) == pytest.approx(70.0, rel=1e-12)
    # pour into a second cup and back: the ledger is exact
    cup2 = V.new_vessel(5e-3, 0.10)
    V.catch(cup2, frag)
    back = V.pour(cup2, 1e9)
    V.catch(v, back)
    after = V.owned(v)
    for sp in ("water", "oil"):
        assert after["ml"][sp] == pytest.approx(before["ml"][sp], rel=1e-15)
        assert after["grams"][sp] == pytest.approx(before["grams"][sp], rel=1e-14)
    assert after["J"] == pytest.approx(before["J"], abs=1e-9)
    assert V.total_ml(cup2) == 0.0


def test_catch_conserves_energy_when_two_temperatures_meet():
    v = V.new_vessel(5e-3, 0.1)
    V.add_layer(v, V.make_layer({"water": 100.0}, T=V.T_REF + 60.0))
    j0 = V.layer_J(v["layers"][0])
    V.add_layer(v, V.make_layer({"water": 50.0}, T=V.T_REF + 60.3))
    j1 = V.layer_J(V.make_layer({"water": 50.0}, T=V.T_REF + 60.3))
    assert len(v["layers"]) == 1
    assert V.owned(v)["J"] == pytest.approx(j0 + j1, rel=1e-12)
    # a much colder layer stays separate (no mixing law here)
    V.add_layer(v, V.make_layer({"water": 10.0}, T=V.T_REF))
    assert len(v["layers"]) == 2


def _weir_reference(H0, area, b, T, dt=1e-3):
    """RK4 of A dH/dt = -k H^1.5, independent of the closed form."""
    k = V.weir_k(b)
    H = H0
    for _ in range(int(round(T / dt))):
        f = lambda h: -k * max(h, 0.0) ** 1.5 / area
        k1 = f(H)
        k2 = f(H + 0.5 * dt * k1)
        k3 = f(H + 0.5 * dt * k2)
        k4 = f(H + dt * k3)
        H += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return H


def test_codeonly_weir_spill_within_10_percent_of_the_law():
    area, lip, b = 5e-3, 0.10, 0.05
    v = V.new_vessel(area, lip, lip_w=b, parcel_ml=1.0)
    V.add_layer(v, V.make_layer({"tea": area * (lip + 0.02) * 1e6}))       # 2 cm over the lip
    H0 = V.level(v) - lip
    assert H0 == pytest.approx(0.02)
    total0 = V.total_ml(v)
    spilled = 0.0
    dt, T = 0.05, 4.0
    for _ in range(int(T / dt)):
        out = V.spill_step(v, dt)
        spilled += sum(V.layer_ml(f) for f in out)
    H_ref = _weir_reference(H0, area, b, T)
    want = area * (H0 - H_ref) * 1e6                              # ml
    assert want > 20.0
    assert spilled == pytest.approx(want, rel=0.10)
    assert V.total_ml(v) + spilled == pytest.approx(total0, rel=1e-12)     # ml exact
    assert V.level(v) >= lip - 1e-9                              # never below the lip
    # Q law itself
    assert V.weir_flow(0.02, b) == pytest.approx(2 / 3 * 0.6 * b * math.sqrt(2 * V.G) * 0.02 ** 1.5)
    assert V.weir_flow(-1.0, b) == 0.0
    # nothing above the lip: nothing spills
    assert V.spill_step(_cup(lip=0.5), 1.0) == []


def test_spill_comes_off_the_top_layer_first():
    v = V.new_vessel(5e-3, 0.05, lip_w=0.05, parcel_ml=1.0)
    V.add_layer(v, V.make_layer({"water": 200.0}, rho=1.0))
    V.add_layer(v, V.make_layer({"oil": 100.0}, rho=0.92))           # level 6 cm, 1 cm over
    out = V.spill_step(v, 1.0)
    assert out and all(list(f["sp"]) == ["oil"] for f in out)


def test_leak_step_is_min_of_orifice_and_slit_and_conserves_ml():
    v = V.new_vessel(5e-3, 0.2, parcel_ml=0.05)
    V.add_layer(v, V.make_layer({"water": 5e-3 * 0.1 * 1e6}))     # 10 cm of water
    total0 = V.total_ml(v)
    out = []
    for _ in range(1000):                                          # 100 s
        out += V.leak_step(v, 0.0, 20e-6, 0.05, 6e-3, 0.1)
    leaked = sum(V.layer_ml(f) for f in out)
    # about 0.0054 ml/s x 100 s (the head falls a little as it drains)
    assert 0.3 < leaked < 0.6
    assert V.total_ml(v) + leaked == pytest.approx(total0, rel=1e-12)
    assert V.leak_step(V.new_vessel(5e-3, 0.2), 0.0, 20e-6, 0.05, 6e-3, 1.0) == []
    # a hole above the surface leaks nothing
    assert V.leak_step(v, 0.5, 20e-6, 0.05, 6e-3, 1.0) == []


# ── pipes ─────────────────────────────────────────────────────────────────────
def test_codeonly_friction_factor_values_and_the_transition_rule():
    assert V.friction_factor(1000.0) == pytest.approx(0.064)
    assert V.friction_factor(2300.0) == pytest.approx(0.0278, abs=5e-5)           # laminar side
    assert V.friction_factor(2301.0) == pytest.approx(0.316 * 2301 ** -0.25)      # max(64/Re, Blasius)
    assert 0.316 * 2300 ** -0.25 == pytest.approx(0.0456, abs=5e-5)
    assert V.friction_factor(10000.0) == pytest.approx(0.0316)
    assert V.friction_factor(3000.0) >= 64 / 3000.0


def test_laminar_turbulent_flow_is_monotone_across_2300_4000():
    D, Lp = 0.01, 1.0
    dhs = np.geomspace(1e-3, 2.0, 400)
    vs, res = [], []
    for dh in dhs:
        v, Re, f = V.conduit_speed(dh, Lp, D)
        vs.append(v)
        res.append(Re)
    assert all(b >= a - 1e-12 for a, b in zip(vs, vs[1:]))                    # flow never falls
    assert min(res) < V.RE_LAM and max(res) > V.RE_TURB                       # crosses the band
    inside = [r for r in res if V.RE_LAM <= r <= V.RE_TURB]
    assert inside                                                              # the band is sampled
    # laminar: Hagen-Poiseuille when entrance loss is negligible (long pipe, low Re)
    v, Re, f = V.conduit_speed(0.01, 50.0, 0.005, k_in=0.0, k_out=0.0)
    hp = 1000.0 * V.G * 0.01 * 0.005 ** 2 / (32 * 1e-3 * 50.0)
    assert Re < 2300 and v == pytest.approx(hp, rel=0.01)
    # turbulent: Blasius, head-loss closes
    v, Re, f = V.conduit_speed(1.0, 2.0, 0.01)
    assert Re > 4000
    assert (1 + V.K_ENTRANCE + V.K_EXIT + f * 2.0 / 0.01) * v * v / (2 * V.G) == pytest.approx(1.0, rel=1e-6)


def test_entrance_and_exit_losses_slow_the_flow():
    v0, _, _ = V.conduit_speed(0.5, 1.0, 0.01, k_in=0.0, k_out=0.0)
    v1, _, _ = V.conduit_speed(0.5, 1.0, 0.01)
    assert v1 < v0
    # a short wide pipe is nearly Torricelli
    vt, _, _ = V.conduit_speed(1.0, 0.01, 0.05, k_in=0.0, k_out=0.0)
    assert vt == pytest.approx(math.sqrt(2 * V.G * 1.0), rel=0.05)
    assert V.conduit_speed(0.0, 1.0, 0.01)[0] == 0.0
    q = V.conduit_flow(0.5, 1.0, 0.01)
    assert q == pytest.approx(v1 * math.pi * 0.01 ** 2 / 4)


# ── the siphon ────────────────────────────────────────────────────────────────
def test_codeonly_vapour_pressure_and_the_height_limit():
    assert V.vapour_pressure_water(293.15) == pytest.approx(2339.0, rel=0.02)
    assert V.vapour_pressure_water(353.15) == pytest.approx(47400.0, rel=0.02)
    assert V.vapour_pressure_water(373.15) == pytest.approx(101325.0, rel=0.02)
    assert V.siphon_max_height(293.15) == pytest.approx(10.1, abs=0.1)         # about 10 m
    assert V.siphon_max_height(353.15) < 6.0 < V.siphon_max_height(293.15)     # less when warm


def test_siphon_needs_priming_and_runs_while_primed():
    kw = dict(z_surface=1.0, z_summit=1.5, z_out=0.0, L=3.0, D=0.01)
    s = V.siphon(primed=False, **kw)
    assert s["v"] == 0.0 and s["reason"] == "not primed" and not s["primed"]
    s = V.siphon(primed=True, **kw)
    assert s["v"] > 0 and s["primed"] and s["Q"] == pytest.approx(s["v"] * math.pi * 1e-4 / 4)
    assert s["p_summit"] < V.P_ATM and s["reason"] == "ok"
    # the outlet at the same level as the surface: no head, no flow
    s = V.siphon(primed=True, **dict(kw, z_out=1.0))
    assert s["v"] == 0.0 and s["reason"] == "no head"
    # the surface falls below the inlet: air enters and clears the prime
    s = V.siphon(primed=True, z_in=1.2, **kw)
    assert not s["primed"] and s["reason"] == "air" and s["v"] == 0.0


def test_siphon_stops_at_vapour_pressure():
    base = dict(z_surface=0.0, z_out=-3.0, L=20.0, D=0.02)
    low = V.siphon(z_summit=5.0, primed=True, **base)
    assert low["v"] > 0 and low["p_summit"] > V.vapour_pressure_water(293.15)
    high = V.siphon(z_summit=11.0, primed=True, **base)                        # over 10 m
    assert high["v"] == 0.0 and high["reason"] == "vapour" and not high["primed"]
    # the summit pressure at the stopping point equals the vapour pressure
    lo, hi = 5.0, 11.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if V.siphon(z_summit=mid, primed=True, **base)["v"] > 0:
            lo = mid
        else:
            hi = mid
    stop = V.siphon(z_summit=lo, primed=True, **base)
    assert stop["p_summit"] == pytest.approx(V.vapour_pressure_water(293.15), rel=0.02)
    assert lo < V.siphon_max_height(293.15)                                    # flow lowers the limit
    # warm water stops lower: 7 m summit runs at 20 C, not at 80 C
    cool = V.siphon(z_summit=7.0, primed=True, T_K=293.15, **base)
    warm = V.siphon(z_summit=7.0, primed=True, T_K=353.15, **base)
    assert cool["v"] > 0 and warm["v"] == 0.0 and warm["reason"] == "vapour"


def test_vessel_owned_is_what_the_ledger_term_adds_up():
    v = _cup()
    o = V.owned(v)
    assert o["ml"] == {"water": 120.0, "oil": 60.0}
    assert o["grams"]["water"] == pytest.approx(120.0) and o["grams"]["oil"] == pytest.approx(55.2)
    assert o["J"] == pytest.approx(0.0, abs=1e-9)                         # at the 20 C reference
    hot = V.new_vessel(5e-3, 0.1)
    V.add_layer(hot, V.make_layer({"tea": 200.0}, rho=1.0, T=V.T_REF + 70.0))
    assert V.owned(hot)["J"] == pytest.approx(200.0 * 4.18 * 70.0)
    assert V.owned(V.new_vessel(5e-3, 0.1)) == {"ml": {}, "grams": {}, "J": 0.0}


def test_merge_refuses_other_species_and_mixes_t_and_moisture_by_mass():
    a = _seeds(10, species="wheat")
    b = _seeds(10, species="barley")
    with pytest.raises(ValueError):
        L.merge(a, b)
    c = _seeds(10, species="wheat")
    c["T"], c["moisture"] = 80.0, 0.3
    a["T"], a["moisture"] = 20.0, 0.1
    m = L.merge(a, c)
    wa = L.total_mass(a) / (L.total_mass(a) + L.total_mass(c))
    assert m["T"] == pytest.approx(wa * 20.0 + (1 - wa) * 80.0)
    assert m["moisture"] == pytest.approx(wa * 0.1 + (1 - wa) * 0.3)
