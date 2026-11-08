"""T4 mixtures, solutions (wave 13, law L14): solubility, the dissolved/crystal target,
Henry's law, pH, sorption, water potential and soil retention. No World.

Tags: test_exact_* (conservation, exactness, closed forms), test_strict_* (independent,
not (u)), test_u_* (source unverified; reads tests/data/sourced_numbers.json when
it exists), test_echo_* (a row reproducing the number it was entered from),
test_codeonly_* (the code against its own table).
"""
import json
import os

import numpy as np
import pytest

from src.voxel import mixdata as D
from src.voxel import mixsol as S
from src.voxel import mixtures as M
from src.voxel import mixwater as W

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCED = os.path.join(HERE, "data", "sourced_numbers.json")
mk = M.mix


def sourced(key):
    try:
        with open(SOURCED) as f:
            e = json.load(f)["entries"]
    except OSError:
        return None, "no-file"
    if key not in e:
        return None, "no-key"
    return e[key]["value"], e[key]["status"]


# -- solubility ----------------------------------------------------------------------------
def test_echo_solubility_rows_at_20_c():
    for name, want in (("nacl", 35.9), ("sucrose", 203.9), ("kno3", 31.6), ("caoh2", 0.173)):
        assert float(S.solubility_water(name, 20.0)) == pytest.approx(want, rel=0.01), name


def test_strict_nacl_solubility_against_the_repo_anchors_r12():
    D.refuse_u("solubility_anchor", "nacl")
    for T, s in D.row("solubility_anchor", "nacl"):                       # 36.0 / 36.8 / 38.1 at 25 / 50 / 80 C
        assert float(S.solubility_water("nacl", T)) == pytest.approx(s, rel=0.01), T


def test_exact_solubility_extends_log_linearly_without_a_clamp():
    assert float(S.solubility_water("sucrose", 112.0)) > 500.0           # candy at 112 C is still unsaturated
    assert float(S.solubility_water("sucrose", 112.0)) > float(S.solubility_water("sucrose", 100.0))
    assert float(S.solubility_water("caoh2", 100.0)) < float(S.solubility_water("caoh2", 0.0))   # retrograde
    assert float(S.solubility_water("nacl", -10.0)) < float(S.solubility_water("nacl", 0.0))     # hydrohalite line
    T = np.linspace(-20.0, 150.0, 300)
    assert np.all(np.diff(S.solubility_water("nacl", T)) > 0) and np.all(np.isfinite(S.solubility_water("kno3", T)))
    assert float(S.solubility_water("glucose", 5.0)) == pytest.approx(91.0, rel=1e-12) == float(S.solubility_water("glucose", 60.0))


def test_u_spirit_drops_the_salt_out():
    s0 = float(S.solubility("nacl", 25.0, 0.0))
    assert float(S.solubility("nacl", 25.0, 0.9)) < s0 / 100.0           # over 100 times lower at 90 % ethanol
    assert float(S.solubility("nacl", 25.0, 1.0)) == pytest.approx(0.065)
    assert float(S.solubility("sucrose", 25.0, 1.0)) == pytest.approx(0.9)
    assert float(S.solubility("kno3", 25.0, 0.5)) == float(S.solubility("kno3", 25.0, 0.0))   # no ethanol row: no effect


def test_exact_dissolved_target_conserves_the_total():
    rng = np.random.default_rng(2)
    d = np.zeros((200, D.n_species()))
    d[:, 0] = rng.uniform(10, 500, 200)
    d[:, D.INDEX["nacl"]] = rng.uniform(0, 300, 200)
    d[:, D.INDEX["sucrose"]] = rng.uniform(0, 900, 200)
    c = np.zeros_like(d)
    c[:, D.INDEX["nacl"]] = rng.uniform(0, 50, 200)
    T = rng.uniform(0, 100, 200)
    deq, ceq, ss = S.dissolved_target(d, c, T)
    assert np.allclose(deq + ceq, d + c, rtol=1e-15, atol=1e-14)
    assert np.all(deq >= 0) and np.all(ceq >= -1e-12)
    sat = S.saturation_grams(d, T)
    i = D.INDEX["nacl"]
    assert np.all(deq[:, i] <= sat[:, i] * (1 + 1e-12))
    assert np.array_equal(deq[:, 0], d[:, 0])                            # water is not moved
    # the target is a fixed point
    d2, c2, ss2 = S.dissolved_target(deq, ceq, T)
    assert np.allclose(d2, deq) and np.allclose(c2, ceq)
    assert np.all(ss2[:, i] <= 1.0 + 1e-12)


def test_exact_dissolved_split_and_supersaturation_read_the_same_solubility():
    c = mk(h2o=100.0, nacl=50.0)
    deq, ceq = S.dissolved_split(c, 25.0)
    assert deq[D.INDEX["nacl"]] == pytest.approx(36.0, rel=0.005) and ceq[D.INDEX["nacl"]] == pytest.approx(14.0, rel=0.02)
    ss = S.supersaturation(c, 25.0)
    assert ss[D.INDEX["nacl"]] == pytest.approx(50.0 / 36.0, rel=0.005)
    assert M.split_forms(c, 25.0)[0][D.INDEX["nacl"]] == deq[D.INDEX["nacl"]]
    assert np.all(S.supersaturation(mk(h2o=100.0, nacl=10.0), 25.0) <= 1.0)


def test_exact_heat_of_solution_is_booked_from_the_split():
    d0 = mk(h2o=100.0)
    d1 = mk(h2o=100.0, nacl=10.0)
    assert float(S.heat_of_split(d0, d1)) == pytest.approx(660.0)        # +66 J/g absorbed
    assert float(S.heat_of_split(d1, d0)) == pytest.approx(-660.0)
    assert S.heat_of_solution("naoh") < 0.0 < S.heat_of_solution("kno3")
    assert S.heat_of_solution("ethanol") == 0.0


def test_exact_nucleation_moves_the_excess_over_saturation_only_past_the_ratio():
    T = 100.0
    sat = float(S.solubility_water("nacl", T))
    below = mk(h2o=100.0, nacl=sat * 1.01)
    above = mk(h2o=100.0, nacl=sat * 1.10)
    d1, c1, r1 = S.nucleate(below, np.zeros_like(below), T)
    d2, c2, r2 = S.nucleate(above, np.zeros_like(above), T)
    i = D.INDEX["nacl"]
    assert c1[i] == 0.0 and r1 == 0.0                                  # NaCl ratio 1.02: 1.01 stays dissolved
    assert d2[i] == pytest.approx(sat, rel=1e-12) and c2[i] == pytest.approx(sat * 0.10, rel=1e-9)
    assert d2[i] + c2[i] == pytest.approx(sat * 1.10, rel=1e-14)
    assert r2 == pytest.approx(66.0 * c2[i], rel=1e-12)


# -- gases in water --------------------------------------------------------------------------
def test_strict_co2_in_water_at_20_c_and_one_atmosphere():
    D.refuse_u("henry", "co2")
    g_per_l = float(S.henry("co2", 20.0, D.P_ATM)) * 44.009 / 1000.0
    assert g_per_l == pytest.approx(1.69, abs=0.05)


def test_u_oxygen_from_moist_air_at_20_c():
    p_o2 = 0.2095 * (D.P_ATM - float(W.psat_water(20.0)))
    mg_per_l = float(S.henry("o2", 20.0, p_o2)) * 31.998
    assert mg_per_l == pytest.approx(9.1, abs=0.6)


def test_exact_gases_dissolve_less_when_warm_or_salty():
    for gas in ("o2", "co2", "n2", "ch4"):
        assert float(S.henry(gas, 5.0, 1e5)) > float(S.henry(gas, 25.0, 1e5)) > float(S.henry(gas, 60.0, 1e5)), gas
    assert float(S.henry("co2", 20.0, 1e5, salt_molal=1.0)) < float(S.henry("co2", 20.0, 1e5))
    assert float(S.henry("co2", 20.0, 2e5)) == pytest.approx(2 * float(S.henry("co2", 20.0, 1e5)), rel=1e-12)


# -- pH ----------------------------------------------------------------------------------------
def acid(name, mol):
    return mk(h2o=1000.0, **{name: mol * D.SPECIES[name].M})


def test_u_ph_of_the_everyday_solutions():
    assert float(S.ph(acid("acetic", 0.83))[0]) == pytest.approx(2.44, abs=0.1)        # vinegar
    assert float(S.ph(acid("acetic", 0.1))[0]) == pytest.approx(2.88, abs=0.1)
    co2_g = float(S.henry("co2", 20.0, 400e-6 * D.P_ATM)) * 44.009 / 1000.0            # rain at 400 ppm
    assert float(S.ph(mk(h2o=1000.0, co2=co2_g))[0]) == pytest.approx(5.6, abs=0.15)
    assert float(S.ph(mk(h2o=100.0, caoh2=0.173))[0]) == pytest.approx(12.45, abs=0.15)  # saturated lime water
    assert float(S.ph(acid("naoh", 0.1))[0]) == pytest.approx(12.9, abs=0.2)
    lac = mk(h2o=1000.0, lactic=0.08 * 90.078, naoh=0.04 * 39.997)                      # 0.04 M lactic + 0.04 M lactate
    assert float(S.ph(lac)[0]) == pytest.approx(3.86, abs=0.1)
    for k in ("acetic", "lactic", "citric", "co2"):
        D.refuse_u("pka", k)


def test_exact_ph_reports_its_validity_range_and_the_ionic_strength():
    pH, I, ok = S.ph(acid("acetic", 0.1))
    assert bool(ok) and 0.0 < float(I) < 0.01
    pH, I, ok = S.ph(mk(h2o=100.0, nacl=40.0, acetic=1.0))
    assert not bool(ok) and float(I) > 0.5                              # a strong brine is outside the Davies range
    pH, I, ok = S.ph(mk(h2o=1000.0))
    assert float(pH) == pytest.approx(7.0, abs=0.01) and bool(ok)       # pure water
    assert float(S.ph(mk(h2o=1000.0, h2so4=0.01 * 98.079))[0]) < 2.1     # sulfuric acid: first step complete
    two = S.ph(np.stack([acid("acetic", 0.1), acid("naoh", 0.1)]))
    assert two[0].shape == (2,) and two[0][0] < 7.0 < two[0][1]


def test_exact_ph_responds_in_the_right_direction():
    ph = [float(S.ph(acid("acetic", m))[0]) for m in (0.001, 0.01, 0.1, 1.0)]
    assert all(b < a for a, b in zip(ph, ph[1:]))
    buf = [float(S.ph(mk(h2o=1000.0, lactic=0.08 * 90.078, naoh=n * 39.997))[0]) for n in (0.0, 0.02, 0.04, 0.06)]
    assert all(b > a for a, b in zip(buf, buf[1:]))
    assert float(S.pkw(25.0)) == 14.0 and float(S.pkw(100.0)) == 12.26


# -- sorption, water potential, soil ---------------------------------------------------------------
def test_echo_wood_isotherm_hits_the_three_fit_points():
    for rh, w in D.row("gab_points", "fit_points"):
        assert float(S.moisture_at_aw("wood", rh / 100.0)) == pytest.approx(w, abs=0.002), rh


def test_u_wood_moisture_at_65_percent_and_20_percent():
    assert float(S.moisture_at_aw("wood", 0.65)) * 100.0 == pytest.approx(12.0, abs=0.6)
    assert float(S.moisture_at_aw("wood", 0.20)) * 100.0 == pytest.approx(4.7, abs=0.6)
    assert float(S.moisture_at_aw("wood", 0.90)) * 100.0 == pytest.approx(20.6, abs=0.6)
    for rh, w in D.row("gab_points", "check_points"):
        assert float(S.moisture_at_aw("wood", rh / 100.0)) == pytest.approx(w, abs=0.008), rh


def test_exact_wood_isotherm_is_monotone_bounded_and_inverts():
    a = np.linspace(0.0, 0.999, 300)
    w = S.moisture_at_aw("wood", a)
    assert np.all(np.diff(w) > 0) and w[0] == 0.0 and np.all(np.isfinite(w)) and w[-1] < 5.0
    back = S.aw_at_moisture("wood", w[1:])
    assert np.allclose(back, a[1:], rtol=1e-10, atol=1e-12)
    assert float(S.moisture_at_aw("wood", 0.5, 40.0)) < float(S.moisture_at_aw("wood", 0.5, 20.0))   # warm wood holds less
    assert float(S.aw_at_moisture("wood", 0.12, 40.0)) > float(S.aw_at_moisture("wood", 0.12, 20.0))
    with pytest.raises(KeyError):
        S.moisture_at_aw("grain", 0.5)                                  # reserved, not built


def test_u_water_potential_of_the_wilting_point():
    assert float(S.aw_from_psi(-1.5e6, 20.0)) == pytest.approx(0.9890, abs=0.0005)
    assert float(S.psi_from_aw(0.9890, 20.0)) == pytest.approx(-1.5e6, rel=0.01)
    a = np.linspace(0.5, 0.999, 20)
    assert np.allclose(S.aw_from_psi(S.psi_from_aw(a)), a, rtol=1e-12)
    assert float(S.psi_from_aw(1.0)) == 0.0


def test_u_soil_retention_at_the_wilting_point():
    assert float(S.retention("loam", -1.5e6)) == pytest.approx(0.088, abs=0.005)
    assert float(S.retention("clay", -1.5e6)) > float(S.retention("sand", -1.5e6))
    psi = -np.logspace(2, 7, 30)
    th = S.retention("loam", psi)
    assert np.all(np.diff(th) < 0) and th[0] <= 0.43 + 1e-12 and th[-1] >= 0.078
    for k in D.TABLES["vangenuchten"]:
        assert float(S.retention(k, 0.0)) == pytest.approx(D.row("vangenuchten", k)[1])    # saturated at zero suction
