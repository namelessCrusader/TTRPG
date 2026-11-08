"""T9 free strain and self-stress (src/voxel/matstress.py).

Tags. Closed-form and exactness tests are the default. test_codeonly_*: [C] the
code against its own table (the granite 281 K comes out of sc = 150 MPa, which
is (u)). test_u_*: [U] independent of the code but the source is not opened
(the oracle materials, the vessel numbers). A (u) number is read from
tests/data/sourced_numbers.json when it is there, and the test says which
status it saw (it never changes the pass).
"""
import json
import math
import os

import numpy as np
import pytest

from src.voxel import matstress as S

HERE = os.path.dirname(os.path.abspath(__file__))


def _sourced(key):
    """The status A-S gave a number, or 'unsourced' (the file is optional)."""
    node = _entry(key)
    return node.get("status", "unsourced") if isinstance(node, dict) else "unsourced"


def _entry(key):
    try:
        d = json.load(open(os.path.join(HERE, "data", "sourced_numbers.json")))
    except (OSError, ValueError):
        return None
    return d.get("entries", {}).get(key) if isinstance(d, dict) else None


GRANITE = dict(rho=2700.0, cp=800.0, k=2.0, E=5e10, alpha=8e-6, nu=0.25, sc=1.5e8, sf=1.5e7)
GLASS = dict(rho=2500.0, cp=800.0, k=0.9, E=7e10, alpha=9e-6, nu=0.22, sc=1e9, sf=5e7)


# ── 1-D conduction oracle (finite differences, numpy only) ──────────────────
def oracle(m, Ts, t, Lz=0.5, n=500):
    """Slab of thickness Lz, one face held at Ts above the start, explicit FD.
    Returns the profile, the cell size, the diffusivity and the EXACT free-plate
    surface stress (Pa): the stress after taking out the mean and the linear
    part of the profile."""
    a = m["k"] / (m["rho"] * m["cp"])
    dz = Lz / n
    dt = 0.4 * dz * dz / a
    T = np.zeros(n)
    T[0] = Ts
    for _ in range(int(t / dt)):
        Tn = T.copy()
        Tn[1:-1] = T[1:-1] + a * dt / dz ** 2 * (T[2:] - 2 * T[1:-1] + T[:-2])
        Tn[0] = Ts
        Tn[-1] = Tn[-2]
        T = Tn
    z = (np.arange(n) + 0.5) * dz - Lz / 2
    slope = (T * z).sum() / (z * z).sum()
    sig = -m["E"] * m["alpha"] / (1 - m["nu"]) * (T - T.mean() - slope * z)
    return T, dz, a, sig[0]


def cell_mean(T, dz, h):
    k = int(round(h / dz))
    return T[: len(T) // k * k].reshape(-1, k).mean(1)


CASES = {"granite": (GRANITE, 800.0), "glass": (GLASS, 200.0)}


@pytest.fixture(scope="module")
def oracles():
    return {name: oracle(m, Ts, 126.0) for name, (m, Ts) in CASES.items()}


def test_u_granite_exact_surface_stress_is_the_385_MPa_of_the_spec(oracles):
    T, dz, a, exact = oracles["granite"]
    assert exact / 1e6 == pytest.approx(-385.0, rel=0.03)
    assert S.skin_depth(GRANITE["k"], GRANITE["rho"], GRANITE["cp"]) == pytest.approx(
        math.sqrt(a * 126.0))
    assert isinstance(_sourced("granite_E_alpha"), str)     # [U]: A-S has no row for E, alpha


@pytest.mark.parametrize("name", ["granite", "glass"])
@pytest.mark.parametrize("h", [0.05, 0.10])
def test_u_cell_mean_plus_skin_law_matches_the_oracle_within_30_percent(oracles, name, h):
    m, Ts = CASES[name]
    T, dz, a, exact = oracles[name]
    Tc = cell_mean(T, dz, h)
    delta = math.sqrt(a * S.SKIN_T0)
    node = T[: int(round(delta / dz))].mean()          # the skin node: mean over its depth
    skin = node - Tc[0]
    T_surface = S.surface_temperature(Tc[0], skin)
    Tb = S.body_mean(Tc, np.ones_like(Tc), h)           # the line through the 0.5 m slab
    sig = float(S.surface_stress(m["E"], m["nu"], m["alpha"], T_surface, Tb))
    assert sig < 0                                       # compression, the hot face
    assert abs(sig - exact) / abs(exact) < 0.30, (name, h, sig / 1e6, exact / 1e6)


@pytest.mark.parametrize("name", ["granite", "glass"])
def test_the_law_does_not_depend_on_the_grid(oracles, name):
    m, Ts = CASES[name]
    T, dz, a, exact = oracles[name]
    delta = math.sqrt(a * S.SKIN_T0)
    node = T[: int(round(delta / dz))].mean()
    out = []
    for h in (0.05, 0.10):
        Tc = cell_mean(T, dz, h)
        Tb = S.body_mean(Tc, np.ones_like(Tc), h)
        out.append(float(S.surface_stress(m["E"], m["nu"], m["alpha"],
                                          S.surface_temperature(Tc[0], node - Tc[0]), Tb)))
    assert abs(out[0] - out[1]) / abs(out[0]) < 0.01


def test_the_old_3_cell_stencil_is_off_by_5x_and_by_2x_between_grids(oracles):
    """The regression for why v1's rule was withdrawn."""
    m, Ts = CASES["granite"]
    T, dz, a, exact = oracles["granite"]
    got = {}
    for h in (0.05, 0.10):
        Tc = cell_mean(T, dz, h)
        sd2 = Tc[1] - (Tc[0] + Tc[2]) / 2
        got[h] = abs(m["E"] * m["alpha"] / (1 - m["nu"]) * sd2)
        assert abs(exact) / got[h] > 5.0
    assert got[0.05] / got[0.10] > 1.7


def test_restrained_stress_and_surface_stress_sign():
    E, nu, eps = 5e10, 0.25, 1e-4
    assert S.restrained_stress(E, nu, eps, "uniaxial") == pytest.approx(E * eps)
    assert S.restrained_stress(E, nu, eps, "biaxial") == pytest.approx(E * eps / 0.75)
    assert S.restrained_stress(E, nu, eps, "triaxial") == pytest.approx(E * eps / 0.5)
    with pytest.raises(ValueError):
        S.restrained_stress(E, nu, eps, "other")
    assert S.surface_stress(E, nu, 8e-6, 300.0, 100.0) < 0          # hot surface: compression
    assert S.surface_stress(E, nu, 8e-6, 20.0, 100.0) > 0           # cold surface: tension
    assert S.surface_stress(E, nu, 8e-6, 100.0, 100.0) == 0
    assert S.interior_tension(E, nu, 8e-6, 20.0, 100.0) > 0         # cool core of a heated body
    assert S.free_strain(8e-6, 100.0) == pytest.approx(8e-4)
    assert S.free_strain(8e-6, 0.0, dV_phase=0.09) == pytest.approx(0.03)
    assert S.free_strain(0.0, 0.0, beta=2e-3, dw=0.1) == pytest.approx(2e-4)


def test_body_mean_is_weighted_and_capped():
    T = np.array([100.0, 50.0, 20.0, 20.0])
    assert S.body_mean(T, np.ones(4)) == pytest.approx(47.5)
    assert S.body_mean(T, np.array([1.0, 1.0, 0.0, 0.0])) == pytest.approx(75.0)
    assert S.body_mean(np.full(30, 40.0), np.ones(30), 0.05) == 40.0
    long = np.concatenate([np.full(10, 100.0), np.zeros(20)])
    assert S.body_mean(long, np.ones(30), 0.05) == pytest.approx(100.0)   # only the first 0.5 m


def test_codeonly_granite_spalls_at_281_K_between_surface_and_body():
    m = GRANITE
    per_K = m["E"] * m["alpha"] / (1 - m["nu"])
    dT = m["sc"] / per_K
    assert dT == pytest.approx(281.0, abs=2.0)
    sig = S.surface_stress(m["E"], m["nu"], m["alpha"], 20.0 + dT + 1.0, 20.0)
    rec = {"sc": m["sc"]}
    fails, depth = S.spall(rec, sig, 0.011, True)
    assert bool(fails) and float(depth) == 0.011
    sig = S.surface_stress(m["E"], m["nu"], m["alpha"], 20.0 + dT - 1.0, 20.0)
    assert not bool(S.spall(rec, sig, 0.011, True)[0])


def test_u_granite_spall_difference_with_the_sourced_compressive_strength():
    """A-S sourced granite compressive strength (148.5 MPa); the spec's 150 MPa
    is (u). The status A-S gave is read, not assumed."""
    e = _entry("rock.granite.compressive")
    if not e or e.get("value") is None:
        pytest.skip(f"A-S has no sourced granite compressive strength (status {_sourced('rock.granite.compressive')})")
    m = GRANITE
    dT = float(e["value"]) * 1e6 / (m["E"] * m["alpha"] / (1 - m["nu"]))
    assert dT == pytest.approx(281.0, abs=3.5)               # 278 K: the E, alpha stay (u)


def test_the_local_combine_copy_equals_matdamage():
    rng = np.random.default_rng(2)
    D, dD = rng.uniform(-0.2, 1.2, 200), rng.uniform(-0.3, 1.1, 200)
    assert np.array_equal(S._combine_copy(D, dD), S.combine(D, dD))


def test_no_spall_on_a_face_against_a_solid_only_damage():
    m = GRANITE
    sig = S.surface_stress(m["E"], m["nu"], m["alpha"], 700.0, 20.0)
    fails, depth = S.spall({"sc": m["sc"]}, sig, 0.011, False)
    assert not bool(fails) and float(depth) == 0.0
    D, piece = S.stress_damage(0.8, 0.0)
    assert 0 < float(D) < 0.01 and not bool(piece)
    D2, _ = S.stress_damage(0.8, float(D))
    assert float(D2) > float(D)
    assert S.stress_damage(1.2, 0.3)[1]


def test_wet_stone_spalls_earlier_by_the_pore_pressure():
    sc = 1.5e8
    sig = -1.2e8
    assert not bool(S.spall({"sc": sc}, sig, 0.01, True)[0])
    assert bool(S.spall({"sc": sc}, sig, 0.01, True, pore_p=4e7)[0])


def test_spall_flake_conserves_mass_and_energy():
    rng = np.random.default_rng(3)
    for _ in range(200):
        m = float(rng.uniform(0.1, 400.0))
        E = float(rng.uniform(-1e3, 5e4))
        depth = float(rng.uniform(0.001, 0.05))
        fl, ef, left, eleft = S.spall_flake(m, E, 2700.0, depth, 0.0025)
        assert fl + left == pytest.approx(m, abs=1e-12) and ef + eleft == pytest.approx(E, abs=1e-9)
        assert 0 <= fl <= m
    fl, ef, left, eleft = S.spall_flake(1000.0, 100.0, 2700.0, 0.011, 0.0025)
    assert fl == pytest.approx(2700 * 0.011 * 0.0025 * 1000.0)            # rho x depth x area
    assert ef == pytest.approx(100.0 * fl / 1000.0)


# ── Kingery / Hasselman ─────────────────────────────────────────────────────
def test_kingery_glass_cases():
    R = S.R_shock(GLASS["sf"], GLASS["nu"], GLASS["E"], GLASS["alpha"])
    assert R == pytest.approx(61.9, abs=0.1)
    assert float(S.kingery_dT(R, 5.0)) == pytest.approx(102.0, abs=2.0)
    assert float(S.kingery_dT(R, 83.0)) == pytest.approx(64.0, abs=2.0)
    assert 2 * R == pytest.approx(123.8, abs=0.2)                   # tshock_face
    assert S.biot(1000.0, 0.003, 0.9) == pytest.approx(3.33, abs=0.01)


def test_kingery_rises_as_Bi_falls_and_tends_to_R():
    R = 61.9
    Bi = np.array([0.5, 1.0, 5.0, 20.0, 83.0, 1e4])
    d = S.kingery_dT(R, Bi)
    assert (np.diff(d) < 0).all()
    assert d[-1] == pytest.approx(R, rel=1e-3)
    assert S.R_shock(GRANITE["sf"], GRANITE["nu"], GRANITE["E"], GRANITE["alpha"]) == pytest.approx(
        28.1, abs=0.1)


# ── ONE sealed-vessel law ───────────────────────────────────────────────────
JAR = dict(r=0.04, t=0.002, E=7e10, nu=0.22, sf=5e7, alpha=9e-6)


def test_u_frozen_jar_table():
    # brim full: 3 percent ice bursts a 4 cm glass jar; headspace saves it
    p, hoop, burst = S.sealed_vessel_pressure(JAR, {"frozen": 0.03, "phi": 0.0})
    assert burst and hoop == pytest.approx(52e6, rel=0.03)
    _, hoop5, burst5 = S.sealed_vessel_pressure(JAR, {"frozen": 1.0, "phi": 0.05})
    assert burst5 and hoop5 == pytest.approx(770e6, rel=0.03)
    _, hoop10, burst10 = S.sealed_vessel_pressure(JAR, {"frozen": 1.0, "phi": 0.10})
    assert not burst10 and hoop10 == pytest.approx(16e6, rel=0.1)
    _, hoop15, burst15 = S.sealed_vessel_pressure(JAR, {"frozen": 1.0, "phi": 0.15})
    assert not burst15 and hoop15 == pytest.approx(3e6, rel=0.2)


def test_frozen_jar_is_continuous_in_headspace():
    phis = np.linspace(0.0, 0.2, 401)
    hoops = np.array([S.sealed_vessel_pressure(JAR, {"frozen": 1.0, "phi": f})[1] for f in phis])
    assert (np.diff(hoops) <= 1e-6).all()                            # more air, never more stress
    assert np.abs(np.diff(hoops)).max() < 0.25 * hoops[0]            # no jump between neighbours
    assert hoops[0] > JAR["sf"] > hoops[-1]


def test_the_solved_pressure_satisfies_its_own_equation():
    for phi in (0.0, 0.02, 0.07, 0.12):
        for x in (0.03, 0.5, 1.0):
            p, _, _ = S.sealed_vessel_pressure(JAR, {"frozen": x, "phi": phi})
            c = 1 / S.K_WATER + S.vessel_compliance(JAR["r"], JAR["t"], JAR["E"], JAR["nu"])
            lhs = phi * p / (S.P_ATM + p) + p * c
            assert lhs == pytest.approx(0.09 * x if lhs > 0 else 0, rel=1e-9) or p == 0.0


def test_u_heated_brim_full_jar_bursts_at_a_14_K_rise():
    def hoop(dT):
        return S.sealed_vessel_pressure(JAR, {"dT": dT, "phi": 0.0})
    lo, hi = 0.0, 100.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if hoop(mid)[2]:
            hi = mid
        else:
            lo = mid
    assert hi == pytest.approx(14.0, abs=1.5)
    assert not hoop(8.0)[2] and hoop(20.0)[2]
    # the same jar with a vent never builds pressure; with headspace it survives the same rise
    assert S.sealed_vessel_pressure(JAR, {"dT": 20.0, "vent": True})[0] == 0.0
    assert not S.sealed_vessel_pressure(JAR, {"dT": 20.0, "phi": 0.1})[2]
    # one law: freezing and heating add as free strains
    both = S.vessel_free_strain({"frozen": 0.01, "dT": 5.0}, JAR["alpha"])
    assert both == pytest.approx(0.0009 + (2.07e-4 - 27e-6) * 5.0)


def test_phase_strains():
    assert S.phase_strain("ice", 5.0, -5.0) == pytest.approx(0.03)            # freezing expands
    assert S.phase_strain("ice", -5.0, 5.0) == pytest.approx(-0.03)
    assert S.phase_strain("ice", 5.0, 3.0) == 0.0
    assert S.phase_strain("quartz", 500.0, 600.0) == pytest.approx(0.004 / 3)
    assert S.phase_strain("quartz", 600.0, 500.0) == pytest.approx(-0.004 / 3)
    assert S.reaction_strain("rust", 1.0) == pytest.approx(2.0 / 3.0)
    assert S.reaction_strain("rust", 0.0) == 0.0
