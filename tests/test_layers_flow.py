"""T3 layers: thickness, cover, flow, drain, drip, soak, imbibition, phase (freeze,
melt, boil, the joint latent pin), kinetics and growth (spec 7: T06-T18, T39, T40,
T42, T43, T45, T51, F12 cross tests).

Tags: test_codeonly_* tests the code against its own table, test_echo_* a constant
fitted to the number it is checked against, test_u_* independent but the source is
unverified (tests/data/sourced_numbers.json promotes them); the rest are
conservation, exactness and arithmetic identities and gate scenes.
"""
import json
import math
import os

import numpy as np
import pytest

from src.voxel import layer_data as D
from src.voxel import layer_laws as LL
from src.voxel.layers import Layers, cell_key, make_key
from tests.layers_fake import FakeHost

HERE = os.path.dirname(os.path.abspath(__file__))
A5 = 0.05 * 0.05


def _sourced(key):
    """(status, value) from tests/data/sourced_numbers.json, or ('absent', None)."""
    try:
        with open(os.path.join(HERE, "data", "sourced_numbers.json")) as f:
            e = json.load(f)["entries"][key]
        return e["status"], e["value"]
    except (OSError, KeyError, ValueError):
        return "absent", None


def _inv(host, L):
    return host.total_E() + L.stored_J() - L.energy_in_total()


# ================================================================ T06 thickness
def test_codeonly_dry_soot_is_22_7_times_its_solid_volume_and_wet_soot_one_sixth():
    r = D.ROWS[D.SOOT]
    G = 1.0 / r["rho_c"]                                  # cm3 of solid per gram
    dry = LL.bulk_volume(0.0, 0.0, G, r["por_dep"])
    assert dry / G == pytest.approx(22.7, rel=0.005)
    need = G * (1.0 / (1.0 - r["por_wet"]) - 1.0)         # liquid that collapses it
    por = LL.porosity_collapse(r["por_dep"], r["por_dep"], r["por_wet"], need, G)
    assert por == pytest.approx(0.70)
    wet = LL.bulk_volume(need, 0.0, G, por)
    assert dry / wet == pytest.approx(6.8, rel=0.05)


def test_wetting_collapse_is_monotone_and_never_reversed_by_drying():
    r = D.ROWS[D.SOOT]
    G = 1.0
    por = r["por_dep"]
    seen = []
    for v in (0.0, 0.5, 1.0, 2.0, 3.0, 1.0, 0.0):          # wet, then dry again
        por = LL.porosity_collapse(por, r["por_dep"], r["por_wet"], v, G)
        seen.append(por)
    assert all(b <= a + 1e-15 for a, b in zip(seen, seen[1:]))
    assert seen[-1] == seen[4]


def test_saturated_sand_is_as_thick_wet_as_dry():
    r = D.ROWS[D.QUARTZ]
    G = 1.0 / r["rho_c"]
    dry = LL.bulk_volume(0.0, 0.0, G, r["por_dep"])
    voids = LL.void_capacity(G, r["por_dep"])
    wet = LL.bulk_volume(voids, 0.0, G, LL.porosity_collapse(r["por_dep"], r["por_dep"],
                                                             r["por_wet"], voids, G))
    assert wet == pytest.approx(dry, rel=0.05)


def test_the_layers_thickness_reads_the_same_numbers():
    L = Layers()
    k = make_key(0, 1, 5)
    L.add(k, {(D.SOOT, D.GRAIN, 1): 0.02}, cover=1.0, Ts=20.0)
    s = L.stack(k)[0]
    V = 0.02 / 1.8 / (1 - 0.956)                          # cm3 of framework
    assert L.thickness(s) == pytest.approx(V * 1e-6 / A5, rel=1e-6)


# ================================================================ T07, T08
def test_codeonly_krieger_dougherty_values_and_the_cap():
    got = [LL.krieger_dougherty(p) for p in (0.0, 0.2, 0.4, 0.55)]
    assert got == pytest.approx([1.0, 1.84, 5.2, 41.6], rel=0.01)
    assert LL.krieger_dougherty(0.5999) > LL.krieger_dougherty(0.59) > LL.krieger_dougherty(0.55)
    assert LL.krieger_dougherty(0.6) == 1.0e6 and LL.krieger_dougherty(0.7) == 1.0e6


def test_codeonly_nusselt_film_speed_and_bingham_limits():
    assert LL.film_speed(1e-4, 1.0e-3, 1000.0) == pytest.approx(0.0327, rel=0.005)
    assert LL.film_speed(1e-4, LL.viscosity_water(20.0), 998.0) == pytest.approx(0.0327, rel=0.01)
    tau = 200.0
    hy = tau / (1600.0 * LL.G)
    assert LL.film_flow(hy, 1.0, 1600.0, tau) == 0.0
    assert LL.film_flow(0.5 * hy, 1.0, 1600.0, tau) == 0.0
    big = 1000 * hy
    assert LL.film_flow(big, 1.0, 1600.0, tau) == pytest.approx(
        LL.film_flow(big, 1.0, 1600.0, 0.0), rel=2e-3)
    h = np.linspace(1.01 * hy, 10 * hy, 50)
    q = [LL.film_flow(x, 1.0, 1600.0, tau) for x in h]
    assert all(b > a for a, b in zip(q, q[1:]))


def test_yield_stress_holds_mud_on_a_wall():
    ty = LL.yield_stress(0.35, 200.0, 0.30)
    assert ty / (1600.0 * LL.G) == pytest.approx(0.020, rel=0.05)        # (u) 20 mm
    assert 10.0 / (1300.0 * LL.G) == pytest.approx(0.0008, rel=0.02)     # paint
    assert LL.yield_stress(0.01, 200.0, 0.30) == 0.0


# ================================================================ T09 Jeffreys
def _column(sp, h0, nz, t_end, dt, rough_um=1.0, eta=None, rho=None, T=20.0):
    """A wall of nz cells (face +x of the x=0 column) filled to h0, drained."""
    dx = 0.05
    shape = (2, 1, nz)
    solid = np.zeros(shape, bool)
    solid[0, :, :] = True
    L = Layers(dx=dx)
    sub = dict(D.SUBSTRATE[2])
    sub["rough_um"] = rough_um
    g = h0 * dx * dx * 1e6 * D.ROWS[sp]["rho_l"]
    for z in range(nz):
        L.add(cell_key(0, 0, z, 0, shape), {(sp, D.DISSOLVED, 0): g}, cover=1.0, Ts=T, sub=sub)
    t = 0.0
    while t < t_end - 1e-9:
        L.flow_pass(solid, dt, Ts=T, sub=sub)
        t += dt
    return L, shape


def _cell_avg_jeffreys(eta, rho, t, z1, z2):
    k = math.sqrt(eta / (rho * LL.G * t))
    return k * (2.0 / 3.0) * (z2 ** 1.5 - z1 ** 1.5) / (z2 - z1)


def test_u_oil_drains_on_the_jeffreys_curve_within_15_percent():
    """Oil (50 mPa s, 920 kg/m3), 1 mm on a 1 m wall, 30 s: 304 um at z = 0.5 m."""
    old = D.ROWS[D.LAMP_OIL]["eta20"], D.ROWS[D.LAMP_OIL]["rho_l"]
    D.ROWS[D.LAMP_OIL]["eta20"], D.ROWS[D.LAMP_OIL]["rho_l"] = 50.0, 0.92
    try:
        L, shape = _column(D.LAMP_OIL, 1e-3, 20, 30.0, 0.1)
        top = 19                                           # cell z index of the top
        j = 10                                             # cell [0.50, 0.55] m from the top
        s = L.stack(cell_key(0, 0, top - j, 0, shape))[0]
        want = _cell_avg_jeffreys(0.05, 920.0, 30.0, 0.50, 0.55)
        assert L.thickness(s) == pytest.approx(want, rel=0.15)
        assert LL.jeffreys_thickness(0.05, 0.5, 920.0, 30.0) == pytest.approx(304e-6, rel=0.01)
        assert L.ledger_closes()
    finally:
        D.ROWS[D.LAMP_OIL]["eta20"], D.ROWS[D.LAMP_OIL]["rho_l"] = old


def test_u_water_drains_on_the_jeffreys_curve_within_15_percent_at_5_cm_cells():
    L, shape = _column(D.WATER, 1e-3, 20, 10.0, 0.025)
    eta = LL.viscosity_water(20.0)
    status, val = _sourced("water.viscosity_andrade")          # Kestin 1981: 1.002 mPa s at 20 C
    eta20 = val["eta_20C_mPas"] * 1e-3 if status == "sourced" else 1.002e-3
    assert eta == pytest.approx(eta20, rel=0.005)
    for j, z1, z2 in ((1, 0.05, 0.10), (2, 0.10, 0.15), (3, 0.15, 0.20)):
        s = L.stack(cell_key(0, 0, 19 - j, 0, shape))[0]
        want = _cell_avg_jeffreys(eta, 998.0, 10.0, z1, z2)
        assert L.thickness(s) == pytest.approx(want, rel=0.15), j
    assert LL.jeffreys_thickness(eta20, 0.05, 998.0, 10.0) == pytest.approx(22.6e-6, rel=0.01)


# ================================================================ T10 mass-exact drain
def _stepped_world():
    """Wall x=0 (z 1..7) with a floor at z=0 and a step block; a second wall at y=2
    that hangs over open air (an overhang)."""
    shape = (6, 4, 9)
    solid = np.zeros(shape, bool)
    solid[:, :, 0] = True                                  # the floor
    solid[0, 0, 1:8] = True                                # wall with a floor at its foot
    solid[0, 1, 1:3] = True                                # wall that ends on a step
    solid[1, 1, 1] = True                                  # the step: top at z = 1
    solid[0, 2, 4:8] = True                                # overhang: air under z < 4
    return solid, shape


def test_a_draining_column_conserves_mass_to_the_last_gram():
    solid, shape = _stepped_world()
    L = Layers()
    sub = dict(D.SUBSTRATE[2])
    sub["rough_um"] = 5.0
    g = 4e-4 * A5 * 1e6
    cols = [(0, 0, z) for z in range(1, 8)] + [(0, 1, 1), (0, 1, 2)] + [(0, 2, z) for z in range(4, 8)]
    for x, y, z in cols:
        L.add(cell_key(x, y, z, 0, shape), {(D.WATER, D.DISSOLVED, 0): g}, cover=1.0, Ts=20.0,
              sub=sub)
    total = L.mass()
    handed = 0.0
    dropped = 0.0
    for _ in range(300):
        r = L.flow_pass(solid, 0.025, Ts=20.0, sub=sub)
        handed += sum(sum(h["comp"].values()) for h in r["handover"])
        dropped += sum(d[6] * D.ROWS[D.WATER]["rho_l"] for d in r["drops"])
    assert L.ledger_closes()
    assert L.mass() + handed + dropped == pytest.approx(total, rel=1e-12)
    assert handed > 0.0                      # the foot and the step handed liquid up
    assert dropped >= 0.0


def test_the_run_reaches_the_floor_foot_which_holds_its_film_and_hands_up_the_rest():
    solid, shape = _stepped_world()
    L = Layers()
    sub = dict(D.SUBSTRATE[2])
    sub["rough_um"] = 5.0
    g = 1e-3 * A5 * 1e6
    for z in range(1, 8):
        L.add(cell_key(0, 0, z, 0, shape), {(D.WATER, D.DISSOLVED, 0): g}, cover=1.0, Ts=20.0,
              sub=sub)
    total = L.mass()
    foot = cell_key(1, 0, 0, 4, shape)
    handed_ml = 0.0
    for _ in range(300):
        r = L.flow_pass(solid, 0.025, Ts=20.0, sub=sub)
        handed_ml += sum(h["ml"] for h in r["handover"] if h["key"] == foot)
    assert handed_ml > 0.0                                     # the lattice got the excess
    hold = L._hold(L._comp(L.stack(foot)[0]), foot, sub)
    assert L.thickness(L.stack(foot)[0]) <= hold * 1.0001      # and the foot kept only its film
    assert L.mass() + handed_ml * D.ROWS[D.WATER]["rho_l"] == pytest.approx(total, rel=1e-12)


def test_hold_thicknesses_by_orientation():
    sig, rho = 0.0728, 998.0
    lc = LL.lambda_c(sig, rho)
    assert lc == pytest.approx(2.72e-3, rel=0.01)
    film = 0.5e-3
    up = LL.hold_thickness("up", 100e-6, 0.0, rho, sig, 60.0, 10e-3)
    assert up == pytest.approx(2 * lc * math.sin(math.radians(30.0)), rel=1e-9)   # 2.7 mm
    assert LL.hold_thickness("up", 100e-6, 0.0, rho, sig, 60.0, film) == film      # lattice cap
    assert LL.hold_thickness("wall", 100e-6, 0.0, rho, sig, 60.0, film) == 100e-6
    hy = 317.0 / (1600.0 * LL.G)
    assert LL.hold_thickness("wall", 100e-6, 317.0, 1600.0, sig, 60.0, film) == pytest.approx(hy)
    down = LL.hold_thickness("down", 100e-6, 0.0, rho, sig, 60.0, film)
    assert down == pytest.approx(min(0.2 * lc, film))
    assert down < up


def test_a_drain_with_a_temperature_gradient_conserves_host_energy_too():
    solid, shape = _stepped_world()
    host = FakeHost(n=int(np.prod(shape)), dtype=np.float64)
    for i in range(host.n):
        host.set_Ts(i, 20.0 + (i % 7) * 4.0)
    L = Layers()
    sub = dict(D.SUBSTRATE[2])
    sub["rough_um"] = 5.0
    g = 4e-4 * A5 * 1e6

    def Ts_of(k):
        return host.Ts(int(k >> 3) & ((1 << 55) - 1))
    for z in range(1, 8):
        k = cell_key(0, 0, z, 0, shape)
        _s, adj = L.add(k, {(D.WATER, D.DISSOLVED, 0): g}, cover=1.0, Ts=Ts_of(k),
                        T_in=40.0, sub=sub)
        host.apply(adj)
    for z in range(4, 8):
        k = cell_key(0, 2, z, 0, shape)
        _s, adj = L.add(k, {(D.WATER, D.DISSOLVED, 0): g}, cover=1.0, Ts=Ts_of(k), sub=sub)
        host.apply(adj)
    b = host.total_E() + L.stored_J() - L.energy_in_total()
    for _ in range(200):
        r = L.flow_pass(solid, 0.025, Ts=Ts_of, sub=sub)
        host.apply(r["adj"])
    assert host.total_E() + L.stored_J() - L.energy_in_total() == pytest.approx(b, abs=1e-7)
    assert L.ledger_closes()


# ================================================================ T11 drip
def test_codeonly_tate_drop_volumes():
    assert LL.tate_drop_volume(0.0728, 998.0) * 1e6 == pytest.approx(0.052, rel=0.05)
    assert LL.tate_drop_volume(0.028, 800.0) * 1e6 == pytest.approx(0.017, rel=0.05)


def _ceiling(ml, sp=D.WATER):
    shape = (3, 3, 4)
    solid = np.zeros(shape, bool)
    solid[1, 1, 2] = True                                  # a block with air under it
    L = Layers()
    key = cell_key(1, 1, 2, 5, shape)
    L.add(key, {(sp, D.DISSOLVED, 0): ml * D.ROWS[sp]["rho_l"]}, cover=1.0, Ts=20.0)
    return L, solid, shape, key


def test_a_ceiling_film_drips_tate_drops_in_the_real_9_field_format():
    L, solid, shape, key = _ceiling(3.0)
    m0 = L.mass()
    r = L.flow_pass(solid, 0.025, Ts=20.0)
    assert len(r["drops"]) == 1
    d = r["drops"][0]
    assert len(d) == LL.DROP_FIELDS == 9
    x, y, z, vx, vy, vz, ml, fl, pot = d
    assert all(isinstance(v, float) for v in (x, y, z, vx, vy, vz, ml, pot))
    assert isinstance(fl, int) and fl == 1 and pot == 0.0
    n = round(ml / (LL.tate_drop_volume(0.0728, 998.0) * 1e6))
    assert ml == pytest.approx(n * LL.tate_drop_volume(0.0728, 998.0) * 1e6, rel=1e-6)
    assert ml <= LL.DROP_MAX_ML
    # drop count times volume plus the remainder equals the mass
    assert L.mass() + ml * D.ROWS[D.WATER]["rho_l"] == pytest.approx(m0, rel=1e-12)
    assert L.ledger_closes()


def test_oil_drips_smaller_drops_and_a_big_wet_ceiling_merges_into_one_parcel():
    L, solid, shape, key = _ceiling(40.0)
    r = L.flow_pass(solid, 0.025, Ts=20.0)
    assert len(r["drops"]) == 1 and r["drops"][0][6] <= LL.DROP_MAX_ML + 1e-9
    L2, solid2, _s, _k = _ceiling(2.0, D.LAMP_OIL)
    r2 = L2.flow_pass(solid2, 0.025, Ts=20.0)
    if r2["drops"]:
        n = round(r2["drops"][0][6] / (LL.tate_drop_volume(0.028, 800.0) * 1e6))
        assert n >= 1 and r2["drops"][0][7] == 2


def test_pendant_drops_helper_counts_and_caps():
    v_d = 5.2e-8
    assert LL.pendant_drops(1e-8, v_d)[0] == 0
    n, ml, left = LL.pendant_drops(10 * v_d + 1e-9, v_d)
    assert n == 10 and left == pytest.approx(1e-9, abs=1e-15)
    n, ml, _l = LL.pendant_drops(1e-3, v_d)                # 1000 ml waits for later ticks
    assert ml <= 5.0 + 1e-9


# ================================================================ T12 mud runs or stays
def _mud_wall(h, phi=0.35):
    shape = (2, 1, 3)
    solid = np.zeros(shape, bool)
    solid[0, :, :] = True
    L = Layers()
    V = A5 * h * 1e6                                       # cm3 of mud
    clay = V * phi * D.ROWS[D.CLAY]["rho_c"]
    water = V * (1 - phi) * D.ROWS[D.WATER]["rho_l"]
    key = cell_key(0, 0, 2, 0, shape)
    L.add(key, {(D.CLAY, D.GRAIN, 8): clay, (D.WATER, D.DISSOLVED, 0): water}, cover=1.0, Ts=20.0)
    return L, solid, shape, key


def test_u_mud_above_its_yield_thickness_runs_and_below_it_stays():
    L, solid, shape, key = _mud_wall(0.040)
    s = L.stack(key)[0]
    L.por[s] = 0.5                                          # wetted: collapsed
    m0 = L.mass(key)
    for _ in range(40):
        L.flow_pass(solid, 0.025, Ts=20.0)
    assert L.mass(key) < m0                                 # 40 mm > 20 mm: it ran
    L2, solid2, shape2, key2 = _mud_wall(0.010)
    L2.por[L2.stack(key2)[0]] = 0.5
    m1 = L2.mass(key2)
    for _ in range(40):
        L2.flow_pass(solid2, 0.025, Ts=20.0)
    assert L2.mass(key2) == pytest.approx(m1, rel=1e-12)    # 10 mm < 20 mm: it stays


def test_drying_mud_stops_the_run():
    L, solid, shape, key = _mud_wall(0.040)
    L.por[L.stack(key)[0]] = 0.5
    L.flow_pass(solid, 0.025, Ts=20.0)
    L.take(key, D.WATER, D.DISSOLVED, 0.8 * L.mass(key, D.WATER), 20.0)    # dry it
    m = L.mass(key)
    for _ in range(40):
        L.flow_pass(solid, 0.025, Ts=20.0)
    assert L.mass(key) == pytest.approx(m, rel=1e-12)


# ================================================================ T40 cover
def test_drain_flux_per_metre_of_width_is_the_same_at_5_and_10_cm_cells():
    fluxes = []
    for dx in (0.05, 0.10):
        shape = (2, 1, 2)
        solid = np.zeros(shape, bool)
        solid[0, :, :] = True
        L = Layers(dx=dx)
        sub = dict(D.SUBSTRATE[2])
        sub["rough_um"] = 1.0
        key = cell_key(0, 0, 1, 0, shape)
        h = 3e-4
        g = h * dx * dx * 1e6 * D.ROWS[D.WATER]["rho_l"]
        L.add(key, {(D.WATER, D.DISSOLVED, 0): g}, cover=1.0, Ts=20.0, sub=sub)
        dt = 1e-4
        L.flow_pass(solid, dt, Ts=20.0, sub=sub)
        moved_ml = (g - L.mass(key)) / D.ROWS[D.WATER]["rho_l"]
        fluxes.append(moved_ml * 1e-6 / dx / dt)            # m3/s per metre of width
    assert fluxes[0] == pytest.approx(fluxes[1], rel=1e-3)
    assert fluxes[0] == pytest.approx(LL.film_flow(3e-4, LL.viscosity_water(20.0), 998.0), rel=2e-3)


def test_cover_merge_patch_and_thinning():
    assert LL.cover_overlap(0.5, 0.5) == 0.75
    L = Layers()
    k = make_key(0, 1, 4)
    L.add(k, {(D.SOOT, D.GRAIN, 1): 0.001}, cover=0.5, Ts=20.0)
    L.add(k, {(D.SOOT, D.GRAIN, 1): 0.001}, cover=0.5, Ts=20.0)
    assert L.cover_of(k) == pytest.approx(0.75)
    # a patch keeps its share while the rule allows it, and shrinks to the rule when thinned
    M = Layers()
    M.add(k, {(D.SOOT, D.GRAIN, 1): 0.01}, cover=0.8, Ts=20.0)
    rule_full = LL.cover_particulate(8.0, 0.01 / A5)
    assert rule_full > 0.8 and M.cover_of(k) == pytest.approx(0.8)
    M.take(k, D.SOOT, D.GRAIN, 0.0099, 20.0)
    rule_thin = LL.cover_particulate(8.0, 0.0001 / A5)
    assert M.cover_of(k) == pytest.approx(min(0.8, rule_thin), rel=1e-4)
    assert M.cover_of(k) < 0.8


# ================================================================ T39 imbibition
def test_water_on_dust_makes_mud_a_thin_film_does_not_carry_dust_and_soot_beads():
    L = Layers()
    k = make_key(0, 1, 4)
    L.add(k, {(D.CLAY, D.GRAIN, 8): 1.0}, cover=1.0, Ts=20.0)
    por0 = float(L.por[L.stack(k)[0]])
    L.add(k, {(D.WATER, D.DISSOLVED, 0): 0.6}, cover=1.0, Ts=20.0)
    assert len(L.stack(k)) == 2
    got = 0.0
    for _ in range(50):
        g, _adj = L.imbibe(k, 0.5)
        got += g
    clay = [s for s in L.stack(k) if L.mass(k, D.CLAY) and
            any((c >> 10) == D.CLAY for c, _g in L._comp(s))][0]
    assert got > 0.0
    assert L.mass(k, D.WATER) == pytest.approx(0.6, rel=1e-12)       # mass kept
    wet_in = sum(g for c, g in L._comp(clay) if (c >> 10) == D.WATER)
    assert wet_in == pytest.approx(got, rel=1e-9) and wet_in > 0.1    # it joined the grain layer
    assert float(L.por[clay]) < por0                                  # the framework collapsed
    phi, tau, _rho = L._mud(L._comp(clay))
    assert tau > 0.0                                                  # a yield stress appeared
    # a 0.1 mm film gives 1 Pa of wall shear against a dust cohesion of 50 Pa
    assert LL.wall_shear_film(1e-4) < 1.1 < float(D.ROWS[D.SOOT]["c_dry"])
    # water on soot beads: no wicking
    M = Layers()
    M.add(k, {(D.SOOT, D.GRAIN, 1): 0.02}, cover=1.0, Ts=20.0)
    M.add(k, {(D.WATER, D.DISSOLVED, 0): 0.5}, cover=1.0, Ts=20.0)
    assert M.imbibe(k, 5.0)[0] == 0.0
    assert M.mass(k, D.WATER, D.DISSOLVED) == pytest.approx(0.5)


def test_a_film_over_a_dust_layer_runs_down_and_leaves_the_dust():
    shape = (2, 1, 3)
    solid = np.zeros(shape, bool)
    solid[0, :, :] = True
    L = Layers()
    key = cell_key(0, 0, 2, 0, shape)
    sub = dict(D.SUBSTRATE[2])
    sub["rough_um"] = 5.0
    L.add(key, {(D.SOOT, D.GRAIN, 1): 0.02}, cover=1.0, Ts=20.0, sub=sub)
    L.add(key, {(D.WATER, D.DISSOLVED, 0): 0.5 * A5 * 1e6 * 1e-3}, cover=1.0, Ts=20.0, sub=sub)
    s0 = L.mass(key, D.SOOT)
    for _ in range(100):
        L.flow_pass(solid, 0.025, Ts=20.0, sub=sub)
    assert L.mass(key, D.SOOT) == pytest.approx(s0, rel=1e-12)
    assert L.mass(key, D.WATER) < 0.5 * A5 * 1e6 * 1e-3


# ================================================================ T13, T14 soak
def test_soak_increments_sum_to_s_sqrt_t_exactly():
    S, t, tot = 13.0, 0.0, 0.0
    for dt in (0.1,) * 50 + (1.0,) * 20 + (30.0,) * 10:
        tot += LL.soak_increment(S, t, dt)
        t += dt
    assert tot == pytest.approx(S * math.sqrt(t), rel=1e-12)
    assert LL.soak_increment(S, 0.0, 1e-9) > 0.0


def test_soak_stain_depth_capacity_sieve_and_no_double_book():
    L = Layers()
    k = make_key(0, 1, 4)
    wood = D.SUBSTRATE[1]
    g_ink = 0.3
    L.add(k, {(D.WATER, D.DISSOLVED, 0): g_ink, (D.INK_CARBON, D.GRAIN, 1): 0.01,
              (D.RED_CELLS, D.GRAIN, 22): 0.01}, cover=1.0, Ts=20.0)
    total = L.mass()
    soaked = 0.0
    t_prev = 0.0
    for i in range(200):
        g, _adj = L.soak(k, 0.5, 1)
        soaked += g
    assert soaked > 0.0
    ins = [s for s in L._chain(k) if L.inside[s]]
    assert len(ins) == 1
    comp = dict(L._comp(ins[0]))
    assert any((c >> 10) == D.INK_CARBON for c in comp)             # 0.1 um pigment went in
    assert not any((c >> 10) == D.RED_CELLS for c in comp)          # a 7 um cell did not
    # depth = V_liq/(porosity A)
    vl = sum(g for c, g in comp.items() if (c >> 10) == D.WATER)
    assert float(L.depth[ins[0]]) == pytest.approx(vl * 1e-6 / (wood["porosity"] * A5) / D.ROWS[D.WATER]["rho_l"],
                                                   rel=1e-4)
    assert L.mass() == pytest.approx(total, rel=1e-12)             # no mass appeared twice
    assert L.ledger_closes()
    # capacity: the plank cannot take more than rho*porosity*thickness*(1 - s) per area
    cap = LL.soak_capacity(1.0, wood["porosity"], 5e-3, 0.0) * A5
    assert vl <= cap + 1e-12


def test_a_face_dry_for_more_than_the_wetting_memory_forgets_how_long_it_was_wet():
    def run(gap):
        L = Layers()
        k = make_key(0, 1, 4)
        L.add(k, {(D.WATER, D.DISSOLVED, 0): 5.0}, cover=1.0, Ts=20.0)
        t, first = 0.0, 0.0
        for _ in range(50):                                        # soak for 25 s
            g, _ = L.soak(k, 0.5, 1, t_now=t)
            first += g
            t += 0.5
        t += gap
        g2, _ = L.soak(k, 0.5, 1, t_now=t)                         # then one more step
        return first, g2
    first, kept = run(5.0)                                          # wet again within 60 s
    _f, reset = run(300.0)                                          # dry for 5 minutes
    assert reset > 2.0 * kept                                       # S sqrt(dt) from scratch is bigger
    assert reset == pytest.approx(LL.soak_increment(13.0, 0.0, 0.5) * A5, rel=0.1)


def test_a_damp_plank_takes_less_and_a_thin_page_passes_the_rest_to_its_back_face():
    base = LL.soak_increment(13.0 * (1 - 0.0), 0.0, 4.0)
    damp = LL.soak_increment(13.0 * (1 - 0.6), 0.0, 4.0)
    assert damp == pytest.approx(0.4 * base)
    L = Layers()
    front = make_key(1, 7, 0)
    back = make_key(1, 7, 1)
    L.add(front, {(D.WATER, D.DISSOLVED, 0): 0.4, (D.INK_GALL, D.DISSOLVED, 0): 0.01},
          cover=1.0, Ts=20.0)
    for _ in range(400):
        L.soak(front, 0.5, D.SUB_PAPER, back_key=back)
    assert L.mass(back, D.WATER) > 0.0                             # it showed through
    paper = D.SUBSTRATE[D.SUB_PAPER]
    for s in L._chain(front) + L._chain(back):
        if L.inside[s]:
            assert float(L.depth[s]) <= paper["thickness"] * 1.001
    assert L.ledger_closes()


def test_a_strip_shallower_than_the_stain_leaves_it_a_deeper_one_removes_it():
    L = Layers()
    k = make_key(0, 1, 4)
    L.add(k, {(D.WATER, D.DISSOLVED, 0): 0.05, (D.PLASMA, D.DISSOLVED, 0): 0.005}, cover=1.0,
          Ts=20.0)
    for _ in range(100):
        L.soak(k, 0.5, 1)
    ins = [s for s in L._chain(k) if L.inside[s]][0]
    d_in = float(L.depth[ins])
    assert d_in > 0.0
    h_surf = sum(L.thickness(s) for s in L.stack(k))
    got, _adj = L.strip(k, h_surf + 0.5 * d_in, 20.0)          # takes the surface, not the pores
    assert any(L.inside[s] for s in L._chain(k)) and not L.stack(k)
    stain = L.mass(k, D.PLASMA)
    assert stain > 0.0
    got2, _adj = L.strip(k, 1.5 * d_in, 20.0)                  # planing past the stain
    assert sum(g for (sp, _f, _z), g in got2.items() if sp == D.PLASMA) == pytest.approx(stain)
    assert not any(L.inside[s] for s in L._chain(k))
    assert L.ledger_closes()


# ================================================================ T15 phase points
def test_u_liquidus_table_values_within_a_tenth_of_a_degree():
    status, val = _sourced("nacl.freezing_curve_C")
    for m, want in ((1.0, -3.34), (2.0, -6.96), (4.28, -16.5)):
        wt = m * 58.44 / (1000.0 + m * 58.44)
        assert LL.brine_freeze_point(wt) == pytest.approx(want, abs=0.1), (m, status)
    if status == "sourced":
        for pct, t in val["wt_pct_to_freezing_C"].items():
            assert LL.brine_freeze_point(float(pct) / 100.0) == pytest.approx(t, abs=0.02)
    st_e, eut = _sourced("nacl.eutectic")                          # -21.12 C at 23.31 wt %
    if st_e == "sourced":
        assert LL.T_EUTECTIC == pytest.approx(eut["T_C"], abs=0.01)
        assert LL.liquidus_wt(eut["T_C"]) == pytest.approx(eut["wt_pct"] / 100.0, abs=1e-4)
        assert LL.liquidus_wt(-30.0) == pytest.approx(eut["wt_pct"] / 100.0, abs=1e-4)
    assert LL.liquidus_wt(-30.0) == pytest.approx(0.2331)          # clamped at the eutectic
    assert LL.liquidus_wt(5.0) == 0.0


def test_a_fixed_van_t_hoff_factor_is_over_a_kelvin_off_at_4_28_mol_per_kg():
    """Why the table is used: Raoult with i = 1.9 misses the table by over 1 K."""
    raoult = LL.raoult_freeze_drop(4.28, 1.9)
    table = 16.5
    assert abs(table - raoult) > 1.0
    eut = LL.raoult_freeze_drop(5.2, 1.9)
    assert abs(21.12 - eut) > 2.0                                  # and 3 K at the eutectic


def test_u_a_brine_film_at_minus_10_holds_ice_fraction_0_64_by_the_lever_rule():
    host = FakeHost(n=2, C=1e9, Cs=1e9, dtype=np.float64)
    host.set_Ts(0, -10.0)
    L = Layers()
    k = make_key(0, 0, 4)
    W = 1.25
    S = 0.0584 * W                                                  # 1 mol/kg
    _s, adj = L.add(k, {(D.WATER, D.DISSOLVED, 0): W, (D.SALT, D.DISSOLVED, 0): S},
                    Ts=-10.0, cover=1.0)
    host.apply(adj)
    host.latent(L, 0, 10.0)
    ice = L.mass(k, D.WATER, D.CRYSTAL) / W
    assert ice == pytest.approx(0.64, abs=0.03)
    status, val = _sourced("nacl.freezing_curve_C")
    if status == "sourced":
        # the lever rule on the SOURCED liquidus: the brine left at -10 C has the weight
        # fraction where the curve crosses -10 C and holds all the salt, so its mass is
        # S / w_b and the ice is what is left of the water.
        pts = sorted((t, float(p) / 100.0) for p, t in val["wt_pct_to_freezing_C"].items())
        w_b = float(np.interp(-10.0, [t for t, _w in pts], [w for _t, w in pts]))
        assert ice == pytest.approx((W + S - S / w_b) / W, abs=0.01)
    assert L.mass(k, D.SALT, D.CRYSTAL) == pytest.approx(0.0, abs=1e-12)    # all in the brine
    assert L.ledger_closes()


def test_the_skin_is_held_at_the_moving_liquidus_and_energy_closes_to_1e_9():
    host = FakeHost(n=2, C=40.0, Cs=10.0, dtype=np.float64)
    host.set_Ts(0, -10.0)
    L = Layers()
    k = make_key(0, 0, 4)
    W = 1.25
    _s, adj = L.add(k, {(D.WATER, D.DISSOLVED, 0): W, (D.SALT, D.DISSOLVED, 0): 0.0584 * W},
                    Ts=-10.0, T_in=-10.0, cover=1.0)
    host.apply(adj)
    b = _inv(host, L)
    Tn, _v = host.latent(L, 0, 10.0)
    assert host.Ts(0) == pytest.approx(Tn, abs=1e-9)               # the node sits on the pin
    ice = L.mass(k, D.WATER, D.CRYSTAL)
    liq_w = W - ice
    brine = 0.0584 * W / (0.0584 * W + liq_w)
    assert LL.brine_freeze_point(brine) == pytest.approx(Tn, abs=0.02)   # on the liquidus
    assert _inv(host, L) == pytest.approx(b, rel=1e-9, abs=1e-9)
    # Kirchhoff: the latent heat is 312 J/g at -10 C, not 333.6
    assert LL.lf_kirchhoff(D.ROWS[D.WATER], -10.0) == pytest.approx(312.65, abs=0.1)


def test_a_pure_film_pins_the_skin_at_zero_while_it_freezes():
    host = FakeHost(n=1, C=40.0, Cs=10.0, dtype=np.float64)
    host.set_Ts(0, -5.0)
    L = Layers()
    k = make_key(0, 0, 4)
    _s, adj = L.add(k, {(D.WATER, D.DISSOLVED, 0): 1.25}, Ts=-5.0, T_in=-5.0, cover=1.0)
    host.apply(adj)
    b = _inv(host, L)
    Tn, _v = host.latent(L, 0, 10.0)
    assert Tn == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < L.mass(k, D.WATER, D.CRYSTAL) < 1.25
    assert _inv(host, L) == pytest.approx(b, rel=1e-9, abs=1e-9)


def test_melting_takes_heat_and_stops_at_zero():
    host = FakeHost(n=1, C=40.0, Cs=10.0, dtype=np.float64)
    L = Layers()
    k = make_key(0, 0, 4)
    host.set_Ts(0, 5.0)
    _s, adj = L.add(k, {(D.WATER, D.CRYSTAL, 40): 0.5}, Ts=5.0, T_in=-3.0, cover=1.0)
    host.apply(adj)
    b = _inv(host, L)
    Tn, _v = host.latent(L, 0, 10.0)
    assert Tn <= 5.0 + 1e-9 and Tn >= -1e-9
    assert L.mass(k, D.WATER, D.DISSOLVED) > 0.0
    assert _inv(host, L) == pytest.approx(b, rel=1e-9, abs=1e-9)


# ================================================================ T45 joint pin, Stefan
def test_two_faces_of_one_cell_freeze_together_against_one_skin_node():
    host = FakeHost(n=1, C=60.0, Cs=15.0, dtype=np.float64)
    host.set_Ts(0, -8.0)
    L = Layers()
    salty, fresh = make_key(0, 0, 0), make_key(0, 0, 4)
    for kk, comp in ((salty, {(D.WATER, D.DISSOLVED, 0): 1.0, (D.SALT, D.DISSOLVED, 0): 0.1}),
                     (fresh, {(D.WATER, D.DISSOLVED, 0): 0.1})):
        _s, adj = L.add(kk, comp, Ts=-8.0, T_in=-8.0, cover=1.0)
        host.apply(adj)
    b = _inv(host, L)
    Tn, _v = host.latent(L, 0, 10.0)
    assert host.Ts(0) == pytest.approx(Tn, abs=1e-9)               # one common T'
    ice_fresh = L.mass(fresh, D.WATER, D.CRYSTAL)
    ice_salty = L.mass(salty, D.WATER, D.CRYSTAL)
    assert ice_fresh == pytest.approx(0.1, rel=1e-9)               # the fresh film is all ice
    assert ice_salty > 0.0                                         # and the brine froze with it
    assert _inv(host, L) == pytest.approx(b, rel=1e-9, abs=1e-9)
    # the energy equation: Csur(T' - Ts) = latent released (checked through the invariant),
    # and the brine sits at its own liquidus while the fresh film is pinned at 0 or below
    assert Tn < -D.T_EUTECTIC * 0 and Tn > D.T_EUTECTIC


def test_an_iron_cell_without_a_skin_node_holds_the_pin_with_its_whole_capacity():
    host = FakeHost(n=1, C=12.0, Cs=0.0, has=False, dtype=np.float64)
    host.set_T(0, -4.0)
    L = Layers()
    k = make_key(0, 0, 4)
    _s, adj = L.add(k, {(D.WATER, D.DISSOLVED, 0): 0.5}, Ts=-4.0, T_in=-4.0, cover=1.0)
    host.apply(adj)
    assert host.Csur(0) == host.C[0]
    b = _inv(host, L)
    Tn, _v = host.latent(L, 0, 10.0)
    assert Tn == pytest.approx(0.0, abs=1e-9) and host.Ts(0) == pytest.approx(0.0, abs=1e-9)
    assert L.mass(k, D.WATER, D.CRYSTAL) > 0.0
    assert _inv(host, L) == pytest.approx(b, rel=1e-9, abs=1e-9)


def test_u_a_3_mm_film_freezes_in_63_s_within_20_percent_stefan_cap():
    dx = 0.05
    host = FakeHost(n=1, C=1e12, Cs=1e12, dtype=np.float64)
    L = Layers(dx=dx)
    k = make_key(0, 0, 4)
    W = 0.003 * dx * dx * 1e6 * D.ROWS[D.WATER]["rho_l"]
    host.set_Ts(0, -10.0)
    _s, adj = L.add(k, {(D.WATER, D.DISSOLVED, 0): W}, Ts=-10.0, T_in=-10.0, cover=1.0)
    host.apply(adj)
    t = 0.0
    while L.mass(k, D.WATER, D.CRYSTAL) < 0.99 * W and t < 600.0:
        host.set_Ts(0, -10.0)
        host.latent(L, 0, 0.05)
        t += 0.05
    assert t == pytest.approx(63.0, rel=0.20)
    assert t == pytest.approx(LL.stefan_time(917.0, 333.55e3, 0.003, 2.2, 10.0), rel=0.20)


# ================================================================ T16, T42
def test_evaporating_a_brine_film_dry_moves_all_salt_to_crystal_and_water_redissolves_it():
    L = Layers()
    k = make_key(0, 1, 4)
    L.add(k, {(D.WATER, D.DISSOLVED, 0): 1.0, (D.SALT, D.DISSOLVED, 0): 0.2}, cover=1.0, Ts=20.0)
    assert L.mass(k, D.SALT, D.CRYSTAL) == 0.0
    L.take(k, D.WATER, D.DISSOLVED, 0.9, 20.0, latent=LL.lv_water(20.0))
    assert L.mass(k, D.SALT, D.CRYSTAL) > 0.0                      # beyond saturation
    L.take(k, D.WATER, D.DISSOLVED, 1.0, 20.0, latent=LL.lv_water(20.0))
    assert L.mass(k, D.SALT, D.CRYSTAL) == pytest.approx(0.2, rel=1e-12)
    assert L.mass(k, D.SALT, D.DISSOLVED) == pytest.approx(0.0, abs=1e-15)
    L.add(k, {(D.WATER, D.DISSOLVED, 0): 2.0}, Ts=20.0)
    assert L.mass(k, D.SALT, D.CRYSTAL) == pytest.approx(0.0, abs=1e-12)    # redissolved
    assert L.mass(k, D.SALT) == pytest.approx(0.2, rel=1e-12)
    assert L.ledger_closes()


def test_dissolving_salt_books_its_heat_of_solution():
    host = FakeHost(n=1, C=40.0, Cs=10.0, dtype=np.float64)
    host.set_Ts(0, 20.0)
    L = Layers()
    k = make_key(0, 0, 4)
    _s, adj = L.add(k, {(D.WATER, D.DISSOLVED, 0): 1.0}, Ts=20.0, cover=1.0)
    host.apply(adj)
    b, T0 = _inv(host, L), host.Ts(0)
    _s, adj = L.add(k, {(D.SALT, D.CRYSTAL, 40): 0.2}, Ts=host.Ts(0), T_in=host.Ts(0))
    host.apply(adj)
    assert host.Ts(0) < T0                                         # endothermic: it cools
    assert L.mass(k, D.SALT, D.DISSOLVED) == pytest.approx(0.2, rel=1e-9)
    assert _inv(host, L) == pytest.approx(b, abs=1e-9)


def test_u_boil_pure_water_pins_100_c_and_1_molal_brine_about_100_95():
    for salt, want in ((0.0, 99.974), (0.05844, 100.95)):
        host = FakeHost(n=1, C=40.0, Cs=10.0, dtype=np.float64)
        host.set_Ts(0, 105.0)
        L = Layers()
        k = make_key(0, 0, 4)
        comp = {(D.WATER, D.DISSOLVED, 0): 1.25}
        if salt:
            comp[(D.SALT, D.DISSOLVED, 0)] = salt * 1.25 / 1.0
        _s, adj = L.add(k, comp, Ts=105.0, T_in=105.0, cover=1.0)
        host.apply(adj)
        b = _inv(host, L)
        w0 = L.mass(k, D.WATER)
        Tn, vap = host.latent(L, 0, 10.0)
        assert host.Ts(0) == pytest.approx(Tn, abs=1e-9)
        assert Tn == pytest.approx(want, abs=0.08)
        assert vap > 0.0
        assert L.mass(k, D.WATER) + vap == pytest.approx(w0, rel=1e-12)   # vapour + liquid = input
        assert L.ledger_closes()
        assert _inv(host, L) == pytest.approx(b, rel=1e-9, abs=1e-9)


def test_boiling_a_film_dry_leaves_the_salt_as_crystal_and_pins_while_it_lasts():
    host = FakeHost(n=1, C=2000.0, Cs=2000.0, dtype=np.float64)
    host.set_Ts(0, 130.0)
    L = Layers()
    k = make_key(0, 0, 4)
    _s, adj = L.add(k, {(D.WATER, D.DISSOLVED, 0): 0.4, (D.SALT, D.DISSOLVED, 0): 0.05},
                    Ts=130.0, T_in=130.0, cover=1.0)
    host.apply(adj)
    b = _inv(host, L)
    host.latent(L, 0, 10.0)
    assert L.mass(k, D.WATER, D.DISSOLVED) == pytest.approx(0.0, abs=1e-12)
    assert L.mass(k, D.SALT, D.CRYSTAL) == pytest.approx(0.05, rel=1e-9)
    assert host.Ts(0) > 100.0                                      # boiled dry, then it warms
    assert _inv(host, L) == pytest.approx(b, rel=1e-9, abs=1e-9)


def test_a_wax_with_a_melting_range_melts_over_the_range_and_conserves_energy():
    host = FakeHost(n=1, C=40.0, Cs=10.0, dtype=np.float64)
    host.set_Ts(0, 50.0)
    L = Layers()
    k = make_key(0, 0, 0)
    _s, adj = L.add(k, {(D.BEESWAX, D.CRYSTAL, 40): 0.5}, Ts=50.0, T_in=50.0, cover=1.0)
    host.apply(adj)
    b = _inv(host, L)
    Tn, _v = host.latent(L, 0, 1.0)
    row = D.ROWS[D.BEESWAX]
    assert row["Tsol"] <= Tn <= row["Tliq"] + 1e-9
    liq = L.mass(k, D.BEESWAX, D.DISSOLVED)
    assert 0.0 < liq < 0.5
    assert liq / 0.5 == pytest.approx((Tn - row["Tsol"]) / (row["Tliq"] - row["Tsol"]), abs=1e-6)
    assert _inv(host, L) == pytest.approx(b, rel=1e-9, abs=1e-9)


# ================================================================ T17, T18, T43 kinetics
def test_the_closed_form_equals_a_fine_loop_and_one_big_step():
    k, X = 3.3e-5, 0.1
    big = LL.first_order_step(X, k, 5000.0)
    x = X
    for _ in range(5000):
        x = LL.first_order_step(x, k, 1.0)
    assert x == pytest.approx(big, rel=1e-9)
    y = X
    for dt in (1000.0, 2500.0, 1500.0):
        y = LL.first_order_step(y, k, dt)
    assert y == pytest.approx(big, rel=1e-12)


def test_codeonly_q10_comes_from_the_rows_own_activation_energy():
    assert LL.q10(50e3) == pytest.approx(2.06, abs=0.01)
    assert LL.q10(60e3) == pytest.approx(2.39, abs=0.01)


def _linseed_x(L, k):
    b = L.mass(k, D.LINSEED, D.BOUND)
    s = L.mass(k, D.LINSEED, D.DISSOLVED)
    return b / (b + s)


def test_codeonly_linseed_film_is_tack_free_at_48_hours_and_sooner_warm():
    h = 10e-6
    g = h * A5 * 1e6 * D.ROWS[D.LINSEED]["rho_l"]
    for T, hours_max in ((20.0, 48.0), (40.0, 12.0)):
        L = Layers()
        k = make_key(0, 1, 4)
        L.add(k, {(D.LINSEED, D.DISSOLVED, 0): g}, cover=1.0, Ts=T)
        t = 0.0
        while t < hours_max * 3600.0:
            t += 600.0
            L.kinetics(k, T, t)
        assert _linseed_x(L, k) >= 0.9, T
        assert L.ledger_closes()
        # it gained 0.12 g of oxygen per gram that cured
        converted = g - L.mass(k, D.LINSEED, D.DISSOLVED)
        assert L.mass(k) == pytest.approx(g + 0.12 * converted, rel=1e-9)
    # a coat thicker than h_ref cures slower
    thick = Layers()
    k = make_key(0, 1, 4)
    thick.add(k, {(D.LINSEED, D.DISSOLVED, 0): 20 * g}, cover=1.0, Ts=20.0)
    t = 0.0
    while t < 48 * 3600.0:
        t += 600.0
        thick.kinetics(k, 20.0, t)
    assert _linseed_x(thick, k) < 0.5


def test_echo_blood_sets_by_heat_x_50c_day_0_88_and_40c_day_0_007():
    """Ea is fitted to two calibration points (half-life 10 s at 65 C, 24 h at 48 C),
    so these checks are echoes: they test the code against its own fit (E)."""
    kin = D.ROWS[D.PLASMA]["kin"]
    day = 86400.0
    for T, t, want in ((50.0, day, 0.88), (40.0, day, 0.007), (65.0, 10.0, 0.5),
                       (70.0, 10.0, 0.9998), (60.0, 60.0, 0.27)):
        k = LL.arrhenius(kin["k_ref"], kin["T_ref"], kin["Ea"], T)
        assert LL.first_order_step(0.0, k, t) == pytest.approx(want, abs=0.01 if want > 0.01 else 0.002)
    assert kin["Ea"] == pytest.approx(481e3, rel=0.01)
    assert D.calibrate_arrhenius(65.0, 10.0, 48.0, day)[1] == pytest.approx(481e3, rel=0.01)


def test_blood_layer_sets_in_the_composition_and_cooling_does_not_unset_it():
    L = Layers()
    k = make_key(0, 1, 4)
    L.add(k, {(D.PLASMA, D.DISSOLVED, 0): 0.05, (D.WATER, D.DISSOLVED, 0): 0.3}, cover=1.0, Ts=20.0)
    L.kinetics(k, 65.0, 0.0)
    L.kinetics(k, 65.0, 10.0)
    x1 = L.mass(k, D.PLASMA, D.BOUND) / L.mass(k, D.PLASMA)
    assert x1 == pytest.approx(0.5, abs=0.02)
    L.kinetics(k, 5.0, 3600.0)                                      # cold again
    assert L.mass(k, D.PLASMA, D.BOUND) / L.mass(k, D.PLASMA) >= x1 - 1e-9
    assert L.ledger_closes()


def test_echo_oxide_growth_is_45_61_69_80_nm_at_60_s_and_continuous_at_the_join():
    got = [LL.oxide_thickness(T, 60.0) * 1e9 for T in (220, 255, 270, 290)]
    assert got == pytest.approx([45, 61, 69, 80], rel=0.05)
    assert LL.oxide_thickness(220.0, 3600.0) * 1e9 == pytest.approx(90.0, rel=0.15)   # an hour
    tj = LL.OXIDE["t_join"]
    for T in (450.0, 600.0):
        a = LL.oxide_thickness(T, tj - 1e-9)
        b = LL.oxide_thickness(T, tj + 1e-9)
        assert a == pytest.approx(b, rel=1e-6)
    # monotone in time and in temperature across the regime change
    xs = [LL.oxide_thickness(T, 60.0) for T in range(200, 700, 10)]
    assert all(b >= a for a, b in zip(xs, xs[1:]))
    ts = [LL.oxide_thickness(500.0, t) for t in (1, 10, 30, 100, 1000)]
    assert all(b > a for a, b in zip(ts, ts[1:]))


def test_a_stepped_oxide_history_equals_the_closed_form():
    for T in (255.0, 500.0):
        x = 0.0
        for _ in range(600):
            x = LL.oxide_step(x, T, 0.1)
        assert x == pytest.approx(LL.oxide_thickness(T, 60.0), rel=1e-6)
    # switching T from below to above the crossover does not jump the thickness
    x = LL.oxide_step(0.0, 390.0, 30.0)
    y = LL.oxide_step(x, 410.0, 0.0)
    assert y == pytest.approx(x, rel=1e-12)


def test_oxide_grows_as_bound_iron_oxide_with_iron_from_the_substrate_and_air_oxygen():
    L = Layers()
    k = make_key(0, 1, 4)
    fe_total = 0.0
    for _ in range(60):
        _adj, fe = L.grow_oxide(k, 255.0, 1.0)
        fe_total += fe
    x = L.mass(k, D.IRON_OXIDE, D.BOUND) / (A5 * D.ROWS[D.IRON_OXIDE]["rho_c"] * 1e6)
    assert x * 1e9 == pytest.approx(61.0, rel=0.06)
    assert fe_total == pytest.approx(0.6994 * L.mass(k), rel=1e-6)
    assert L.ledger_closes()
    sub = sum(v for (sp, t), v in L.led_in.items() if t == "substrate")
    o2 = sum(v for (sp, t), v in L.led_in.items() if t == "o2")
    assert sub / (sub + o2) == pytest.approx(0.6994, rel=1e-6)


def test_spall_hammer_scale_flies_off_a_quenched_bar_a_thin_film_does_not():
    assert LL.spall_energy(100e9, 2.4e-3, 50e-6) == pytest.approx(14.4, rel=0.01)
    assert LL.spalls(100e9, 2.4e-3, 50e-6, 10.0)
    assert not LL.spalls(100e9, 2.4e-3, 5e-6, 10.0)
    assert LL.spall_h_crit(100e9, 5e-3, 10.0) == pytest.approx(8e-6, rel=0.01)


# ================================================================ T51 slow tier
def test_slow_tier_promotes_a_flame_licked_layer_and_leaves_a_cool_one():
    tick = 0.025
    k_cold = LL.arrhenius(1.33e-5, 20.0, 60e3, 22.0)
    assert not LL.promote_hot(22.0, k_cold, tick, False)
    assert LL.promote_hot(60.0, k_cold, tick, False)                # skin 40 K off ambient
    k_hot = LL.arrhenius(1.33e-5, 20.0, 60e3, 300.0)
    assert LL.promote_hot(21.0, k_hot, tick, False)                 # k dt would move it > 10 %
    assert LL.promote_hot(20.0, 0.0, tick, True)                    # above its hold limit


def test_a_stepped_kinetic_history_matches_one_big_step_at_constant_t():
    big = Layers()
    small = Layers()
    k = make_key(0, 1, 4)
    comp = {(D.PLASMA, D.DISSOLVED, 0): 0.05}
    for obj in (big, small):
        obj.add(k, comp, cover=1.0, Ts=20.0)
    big.kinetics(k, 55.0, 0.0)
    big.kinetics(k, 55.0, 4000.0)
    for i in range(1, 401):
        small.kinetics(k, 55.0, 0.0 if i == 1 else 10.0 * (i - 1))
    small.kinetics(k, 55.0, 4000.0)
    assert small.mass(k, D.PLASMA, D.BOUND) == pytest.approx(big.mass(k, D.PLASMA, D.BOUND), rel=1e-9)


# ================================================================ F12 cross tests (stand-ins)
def _t4():
    mw = pytest.importorskip("src.voxel.mixwater")
    md = pytest.importorskip("src.voxel.mixdata")
    return mw, md


def _dense(md, water_g, salt_g):
    c = np.zeros(md.n_species())
    c[md.INDEX["h2o"]] = water_g
    c[md.INDEX["nacl"]] = salt_g
    return c


def test_cross_psat_and_tsat_stand_ins_agree_with_mixwater():
    mw, _md = _t4()
    for T in (0.5, 25.0, 60.0, 100.0, 150.0):
        assert LL.psat_water(T) == pytest.approx(float(mw.psat_water(T)), rel=1e-3)
    assert LL.tsat_water(101325.0) == pytest.approx(float(mw.tsat_water(101325.0)), abs=0.01)


def test_cross_freeze_and_boil_stand_ins_agree_with_mixwater():
    mw, md = _t4()
    for m in (0.5, 1.0, 2.0, 4.28):
        s = 58.44 * m
        c = _dense(md, 1000.0, s)
        assert LL.freeze_point(1000.0, {D.SALT: s}) == pytest.approx(float(mw.freeze_point(c)), abs=0.25)
    for m in (0.5, 1.0, 2.0):
        s = 58.44 * m
        c = _dense(md, 1000.0, s)
        assert LL.boil_point({D.SALT: s}, 1000.0) == pytest.approx(float(mw.boiling_point(c)), abs=0.1)


def test_cross_phase_step_agrees_with_the_private_bisection():
    mp = pytest.importorskip("src.voxel.mixphase")
    md = pytest.importorskip("src.voxel.mixdata")
    cases = ((10.0, 0.0, -5.0, 20.0), (10.0, 0.0, -5.0, 60.0), (10.0, 0.5844, -10.0, 5.0),
             (10.0, 0.5844, -10.0, 50.0), (10.0, 1.2, -15.0, 30.0))
    for w, sa, T, C in cases:
        c = _dense(md, w, sa)
        try:
            out = mp.phase_step(c, 1.0, T, C)
        except Exception as e:                                     # a changed T4 signature
            pytest.skip(f"phase_step call failed: {e!r}")
        host = FakeHost(n=1, C=C, Cs=C, has=False, dtype=np.float64)
        L = Layers()
        k = make_key(0, 0, 4)
        comp = {(D.WATER, D.DISSOLVED, 0): w}
        if sa:
            comp[(D.SALT, D.DISSOLVED, 0)] = sa
        L.add(k, comp, Ts=T, T_in=T, cover=1.0)
        host.C[0] = host.Cs[0] = C                                  # T4's C already holds the film
        host.set_T(0, T)
        Tn, _v = host.latent(L, 0, 10.0)
        f_liq = 1.0 - L.mass(k, D.WATER, D.CRYSTAL) / (w + sa)
        tol_T, tol_f = (1e-6, 1e-6) if sa == 0.0 else (0.25, 0.005)
        assert Tn == pytest.approx(float(out[1]), abs=tol_T), (w, sa, T, C)
        assert f_liq == pytest.approx(float(out[0]), abs=tol_f), (w, sa, T, C)


def test_cross_viscosity_stand_in_agrees_with_mixrules():
    mr = pytest.importorskip("src.voxel.mixrules")
    for T in (20.0, 50.0):
        want = float(mr.andrade(D.WATER, T))                        # mPa s
        assert LL.viscosity_water(T) * 1e3 == pytest.approx(want, rel=0.02), T
