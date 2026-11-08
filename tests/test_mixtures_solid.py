"""T4 mixtures, solid (wave 13, law L14): gas-solid equilibria, the carbon potential
and electrode potentials. No World.

Tags: test_exact_* (identities, closed forms), test_strict_* (independent, not (u)),
test_u_* (source unverified; reads tests/data/sourced_numbers.json when it exists),
test_echo_* (a fitted constant reproducing its fit), test_codeonly_* (code against
its own table). A sourced JANAF row that disagrees with the spec's constant is an
xfail with the ratio in the reason (reported to the user for the source-check pass).
"""
import json
import os

import numpy as np
import pytest

from src.voxel import mixdata as D
from src.voxel import mixsolid as X

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCED = os.path.join(HERE, "data", "sourced_numbers.json")
ATM = D.P_ATM


def sourced(key):
    try:
        with open(SOURCED) as f:
            e = json.load(f)["entries"]
    except OSError:
        return None, "no-file"
    if key not in e:
        return None, "no-key"
    return e[key]["value"], e[key]["status"]


def root(fn, lo, hi, n=80):
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if fn(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# -- calcite -------------------------------------------------------------------------------------
def test_echo_calcite_gives_one_atmosphere_at_898_c():
    T = root(lambda t: X.eq_pressure("calcite", t) / ATM - 1.0, 700.0, 1100.0)
    assert T == pytest.approx(898.0, abs=1.0)
    assert D.src_of("gassolid", "calcite") == "fit:calcite-898"


def test_u_calcite_pressure_at_827_and_977_and_monotone():
    assert float(X.eq_pressure("calcite", 827.0)) / ATM == pytest.approx(0.35, abs=0.05)
    assert float(X.eq_pressure("calcite", 977.0)) / ATM == pytest.approx(2.8, abs=0.4)
    T = np.linspace(500.0, 1100.0, 100)
    assert np.all(np.diff(X.eq_pressure("calcite", T)) > 0)


def test_u_calcite_against_the_sourced_robie_hemingway_k(record_property):
    v, status = sourced("janaf.calcite")
    record_property("sourced_status", status)
    if v is None:
        pytest.skip(f"sourced_numbers: {status}")
    for T, row in v["by_T_C"].items():
        assert float(X.reaction_K("calcite", float(T))) == pytest.approx(row["K"], rel=0.30), T     # up to 28 % at 600 C
    assert float(X.eq_pressure("calcite", v["T_p1bar_C"])) / ATM == pytest.approx(1.0, rel=0.20)


# -- Boudouard and wuestite ------------------------------------------------------------------------
def test_u_boudouard_k_is_one_at_705_c_and_the_co_share_follows():
    T = root(lambda t: float(X.reaction_K("boudouard", t)) - 1.0, 500.0, 900.0)
    assert T == pytest.approx(705.0, abs=3.0)
    assert float(X.co_fraction("boudouard", 700.0)) == pytest.approx(0.60, abs=0.06)
    assert float(X.co_fraction("boudouard", 800.0)) == pytest.approx(0.88, abs=0.06)
    assert float(X.co_fraction("boudouard", 900.0)) > 0.95
    f = [float(X.co_fraction("boudouard", t)) for t in range(500, 1100, 50)]
    assert all(b > a for a, b in zip(f, f[1:]))
    assert float(X.co_fraction("boudouard", 800.0, P=10 * ATM)) < float(X.co_fraction("boudouard", 800.0))
    assert float(X.co_fraction("boudouard", 800.0, a_C=0.5)) < float(X.co_fraction("boudouard", 800.0))


def test_u_boudouard_against_the_sourced_janaf_k(record_property):
    v, status = sourced("janaf.boudouard")
    record_property("sourced_status", status)
    if v is None:
        pytest.skip(f"sourced_numbers: {status}")
    for T, row in v["by_T_C"].items():
        assert float(X.reaction_K("boudouard", float(T))) == pytest.approx(row["K"], rel=0.12), T   # spec / sourced 0.91-0.94


def test_u_wuestite_needs_about_68_percent_co_at_1000_c():
    assert float(X.co_fraction("wuestite", 1000.0)) == pytest.approx(0.68, abs=0.03)
    assert float(X.co_fraction("wuestite", 1200.0)) > float(X.co_fraction("wuestite", 1000.0))


@pytest.mark.xfail(strict=True, reason=(
    "Sourced NIST-JANAF for real wuestite (Fe0.947O + CO = 0.947 Fe + CO2) gives K = 0.902 at 600 C "
    "and 0.404 at 1000 C; the spec's constants (a -22800, b -24.3) are 1.15-1.38x too high (CO share "
    "at 1000 C: spec 68 percent, sourced 71 percent). The spec band stays until the source-check pass "
    "refits the row (reported to the user)."))
def test_u_wuestite_against_the_sourced_janaf_row():
    v, status = sourced("janaf.wuestite")
    if v is None:
        pytest.skip(f"sourced_numbers: {status}")
    for T, row in v["variant_Fe0.947O_(real_wuestite)"].items():
        assert float(X.reaction_K("wuestite", float(T))) == pytest.approx(row["K"], rel=0.10), T


# -- carbon potential ----------------------------------------------------------------------------------
def test_u_endothermic_gas_carburises_to_about_point_eight_percent():
    aC = float(X.carbon_activity(0.20, 0.002, 925.0))
    assert aC == pytest.approx(0.44, abs=0.05)
    assert float(X.austenite_carbon(aC, 925.0)) == pytest.approx(0.8, abs=0.25)
    assert float(X.austenite_carbon(1.0, 925.0)) == pytest.approx(1.7, abs=0.4)         # (u)


def test_exact_carbon_potential_inverse_round_trips():
    for T in (800.0, 925.0, 1050.0):
        for w in (0.05, 0.4, 0.8, 1.2):
            a = float(X.carbon_activity_of(w, T))
            assert float(X.austenite_carbon(a, T)) == pytest.approx(w, rel=1e-9, abs=1e-10)
    w = np.linspace(0.05, 1.5, 40)
    assert np.all(np.diff(X.carbon_activity_of(w, 925.0)) > 0)


def test_exact_affinity_changes_sign_exactly_at_the_equilibrium_pressure():
    for T in (700.0, 800.0, 898.0, 1000.0):
        peq = float(X.eq_pressure("calcite", T))
        assert float(X.affinity("calcite", T, peq)) == pytest.approx(0.0, abs=1e-6)
        assert float(X.affinity("calcite", T, 0.5 * peq)) > 0.0 > float(X.affinity("calcite", T, 2.0 * peq))
    # CO / CO2 rows: forward (reduction) when CO is richer than the equilibrium share
    T = 1000.0
    x = float(X.co_fraction("wuestite", T))
    pco, pco2 = x * ATM, (1 - x) * ATM
    assert float(X.affinity("wuestite", T, (pco, pco2))) == pytest.approx(0.0, abs=1e-6)
    assert float(X.affinity("wuestite", T, (0.9 * ATM, 0.1 * ATM))) > 0.0 > float(X.affinity("wuestite", T, (0.3 * ATM, 0.7 * ATM)))
    xb = float(X.co_fraction("boudouard", 800.0))
    assert float(X.affinity("boudouard", 800.0, (xb * ATM, (1 - xb) * ATM))) == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError):
        X.eq_pressure("boudouard", 800.0)


def test_exact_gas_solid_rows_are_vectorised_and_finite():
    T = np.linspace(300.0, 1500.0, 500)
    for rid in D.TABLES["gassolid"]:
        K = X.reaction_K(rid, T)
        assert np.all(np.isfinite(K)) and np.all(K > 0.0), rid


# -- electrodes -------------------------------------------------------------------------------------------
def test_u_cell_potentials():
    assert float(X.cell_emf("Fe2+/Fe", "Cu2+/Cu")) == pytest.approx(0.789, abs=0.005)      # the nail in copper salt
    assert float(X.cell_emf("Zn2+/Zn", "Cu2+/Cu")) == pytest.approx(1.104, abs=0.005)
    assert float(X.cell_emf("Al3+/Al", "Ag2S/Ag")) == pytest.approx(0.97, abs=0.01)         # tarnish removal
    assert float(X.cell_emf("Cu2+/Cu", "Fe2+/Fe")) < 0.0                                    # copper in iron salt does nothing
    assert float(X.E0("H+/H2")) == 0.0 and len(D.TABLES["e0"]) >= 24


def test_u_aluminium_is_a_stronger_reductant_in_base():
    e14 = float(X.cell_emf("Al(OH)4-/Al", "Ag2S/Ag", pH=14.0))
    e7 = float(X.cell_emf("Al(OH)4-/Al", "Ag2S/Ag", pH=7.0))
    assert e14 > e7 > float(X.cell_emf("Al3+/Al", "Ag2S/Ag"))
    assert e14 == pytest.approx(2.33 - 0.691, abs=1e-9)
    # a couple without a proton term does not move with pH
    assert float(X.nernst("Cu2+/Cu", pH=3.0)) == float(X.nernst("Cu2+/Cu", pH=11.0))


def test_strict_nernst_slope_is_59_16_mv_per_decade():
    d = float(X.nernst("Ag+/Ag", 10.0) - X.nernst("Ag+/Ag", 1.0)) * 1000.0
    assert d == pytest.approx(59.16, abs=0.1)
    d2 = float(X.nernst("Cu2+/Cu", 10.0) - X.nernst("Cu2+/Cu", 1.0)) * 1000.0
    assert d2 == pytest.approx(59.16 / 2.0, abs=0.1)                                         # n = 2
    assert float(X.nernst("Cu2+/Cu", 1.0, 1.0)) == X.E0("Cu2+/Cu")
    assert float(X.nernst("Ag+/Ag", 1.0, 1.0, T=60.0)) == X.E0("Ag+/Ag")                     # Q = 1: no T term
    assert float(X.nernst("Ag+/Ag", 0.1, 1.0, T=60.0)) < float(X.nernst("Ag+/Ag", 0.1, 1.0, T=25.0))
    assert float(X.cell_emf("Fe2+/Fe", "Cu2+/Cu", a_a=0.01, a_c=1.0)) > 0.789               # dilute anode side drives harder
    assert float(X.cell_emf("Fe2+/Fe", "Cu2+/Cu", a_a=(1.0, 1.0), a_c=(1.0, 1.0))) == pytest.approx(0.789)
