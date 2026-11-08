"""T6 reaction rows (wave 13, lane A-T6). Tags as in test_t6_psychro.py."""
import math

import numpy as np
import pytest

from src.voxel import materials as MAT
from src.voxel import matterdata as D
from src.voxel import rxn as RX

CAT = RX.catalogue()


def _state(amounts, T=20.0, C_other=100.0, **kw):
    st = RX.ArrayState(amounts, E=0.0, C_other=C_other, **kw)
    st.E = st.C * (np.asarray(T, np.float64) - 20.0)
    return st


# ---- rates ------------------------------------------------------------------------
def test_arrhenius_50_kJ_is_1p91_per_10_K_near_300_K():
    a = RX.Arrhenius.from_point(1.0, 26.85, 50000.0)
    assert abs(float(a.value(36.85)) / float(a.value(26.85)) - 1.91) < 0.02
    assert float(a.value(26.85)) == pytest.approx(1.0)


def test_bigelow_z_10_K_at_60_C_is_212_kJ_per_mol():
    b = RX.Bigelow(100.0, 60.0, 10.0)
    assert b.Ea_equivalent(60.0) == pytest.approx(212000.0, rel=0.01)
    assert float(b.D(70.0)) == pytest.approx(10.0)                # one decade per z
    assert float(b.value(60.0)) == pytest.approx(math.log(10.0) / 100.0)


def test_points_with_and_without_ea():
    xs, ys = np.array([0.0, 0.5, 1.0]), np.array([0.0, 10.0, 40.0])
    p0 = RX.Points(xs, ys)
    assert float(p0.value(80.0, 0.75)) == float(np.interp(0.75, xs, ys))
    p1 = RX.Points(xs, ys, T_ref=20.0, Ea=50000.0)
    assert float(p1.value(20.0, 0.75)) == pytest.approx(float(p0.value(20.0, 0.75)))
    assert float(p1.value(30.0, 0.75)) > 1.5 * float(p0.value(20.0, 0.75))


# ---- the catalogue -------------------------------------------------------------------
def test_every_catalogue_row_closes_mass_to_1e9_and_matches_hess_within_2_percent():
    for name, row in CAT.items():
        mass, err = RX.check(row)
        assert mass < 1e-9, name
        if err is not None:
            assert err < 0.02, (name, err)
    assert set(CAT) >= {"calcine", "slake", "carbonate", "rust", "dissolve_nacl",
                        "dissolve_sucrose", "ferment", "boudouard"}
    # the rows that have a heat of formation for every species all got checked
    assert sum(1 for r in CAT.values() if RX.dH_hess(r) is not None) >= 6


def test_a_row_that_does_not_close_is_refused():
    bad = RX.Row("bad", "bulk", "A", (("A", 1.0),), (("B", 0.9),), 0.0, RX.Const(1.0))
    with pytest.raises(ValueError):
        RX.check(bad)
    RX.check(bad.replace(closed=False))            # an open row (legacy) is allowed


def test_slaking_and_carbonating_heats_have_the_right_sign_and_size():
    assert CAT["slake"].dH == pytest.approx(1150.0) and CAT["slake"].dH > 0     # released
    assert CAT["calcine"].dH < 0                                                # absorbed
    assert 1700.0 < -CAT["calcine"].dH < 1900.0
    assert 1.0e3 * 1.4 < CAT["carbonate"].dH < 1.0e3 * 1.7


def test_first_order_ten_seconds_equals_ten_one_second_steps_whatever_ea():
    for Ea in (0.0, 30000.0, 190000.0):
        row = CAT["slake"].replace(rate=RX.Arrhenius.from_point(0.05, 20.0, Ea))
        a = _state({"CaO": 10.0, "H2O": 100.0})
        b = _state({"CaO": 10.0, "H2O": 100.0})
        RX.advance([row], a, 20.0, 10.0)
        for _ in range(10):
            RX.advance([row], b, 20.0, 1.0)
        assert abs(float(a.get("CaO")) - float(b.get("CaO"))) < 1e-9
        assert abs(float(a.get("Ca(OH)2")) - float(b.get("Ca(OH)2"))) < 1e-9


def test_u_calcine_equilibrium_pressure_at_898_C_is_one_atmosphere():
    p = float(RX.p_eq_calcine(898.0))
    assert abs(p / 101325.0 - 1.0) < 0.05
    # the open-air threshold: the pressure of fresh-air CO2 (40 Pa) is reached 520-550 C
    pa = D.FRESH_CO2_G_M3 / D.M["CO2"] * D.R * D.T_REF_GAS
    assert abs(pa - 40.5) < 1.5
    T = 20474.0 / math.log(4.137e12 / pa) - 273.15
    assert 520.0 < T < 550.0


def _calcine_state(T, p_ratio, caco3=100.0, cao=10.0):
    """A 1 L cell whose CO2 pressure is p_ratio times the equilibrium at T."""
    p_eq = float(RX.p_eq_calcine(T))
    g_m3 = p_ratio * p_eq / (D.R * D.T_REF_GAS) * D.M["CO2"]
    V = 1e-3
    ex = (g_m3 - D.FRESH_CO2_G_M3) * V
    return _state({"CaCO3": caco3, "CaO": cao, "CO2": ex}, T=T, V_m3=V)


def test_equilibrium_forward_below_backward_above_zero_at_p_eq():
    row = CAT["calcine"]
    T = 800.0
    below, above, at = _calcine_state(T, 0.5), _calcine_state(T, 2.0), _calcine_state(T, 1.0)
    assert float(RX.extent(row, below, T, 10.0)) > 0
    assert float(RX.extent(row, above, T, 10.0)) < 0
    assert abs(float(RX.extent(row, at, T, 10.0))) < 1e-9
    # open air at 700 C is above 534 C: calcines; at 450 C it does not
    fresh = _state({"CaCO3": 100.0, "CaO": 0.0, "CO2": 0.0}, T=700.0)
    cold = _state({"CaCO3": 100.0, "CaO": 0.0, "CO2": 0.0}, T=450.0)
    assert float(RX.extent(row, fresh, 700.0, 100.0)) > 0
    assert abs(float(RX.extent(row, cold, 450.0, 1e5))) < 1e-12 or float(RX.extent(row, cold, 450.0, 1e5)) <= 0


def test_no_sign_flip_across_Q_equals_K_for_any_dt():
    row = CAT["calcine"]
    for T in (600.0, 800.0, 898.0, 1000.0):
        for ratio in (0.01, 0.5, 0.99, 1.01, 3.0, 50.0):
            for dt in (1e-3, 1.0, 1e3, 1e6, 1e12):
                st = _calcine_state(T, ratio)
                lnK = float(row.equilibrium.lnK(T))
                before = float(RX._ln_q(row, st, 0.0)) - lnK
                RX.advance([row], st, T, dt)
                after = float(RX._ln_q(row, st, 0.0)) - lnK
                assert before * after >= -1e-9 * max(1.0, abs(before)), (T, ratio, dt)
                assert abs(after) <= abs(before) + 1e-9
    # the same for a quotient with powers (Boudouard: pCO^2 / pCO2)
    row = CAT["boudouard"]
    for ratio in (0.1, 10.0):
        for dt in (1.0, 1e4, 1e10):
            T = 900.0
            lnK = float(row.equilibrium.lnK(T))
            V = 1e-3
            # choose CO2 so that Q = ratio K with CO = 5 g
            pco = (5.0 / V + 0.0) / D.M["CO"] * D.R * D.T_REF_GAS
            pco2 = pco ** 2 / (ratio * math.exp(lnK))
            ex = (pco2 / (D.R * D.T_REF_GAS) * D.M["CO2"] - D.FRESH_CO2_G_M3) * V
            st = _state({"C": 50.0, "CO": 5.0, "CO2": ex}, T=T, V_m3=V)
            before = float(RX._ln_q(row, st, 0.0)) - lnK
            RX.advance([row], st, T, dt)
            after = float(RX._ln_q(row, st, 0.0)) - lnK
            assert before * after >= -1e-9 * max(1.0, abs(before))


def test_extent_never_exceeds_the_reactant_or_the_room_and_nothing_goes_negative():
    rng = np.random.default_rng(9)
    rows = [CAT[k] for k in ("slake", "carbonate", "ferment", "rust", "calcine",
                             "boudouard", "dissolve_nacl")]
    st = _state({"CaO": 3.0, "H2O": 4.0, "Ca(OH)2": 2.0, "CO2": 0.0, "glucose": 5.0,
                 "ethanol": 0.0, "Fe": 2.0, "O2": 1.0, "FeOOH": 0.0, "CaCO3": 4.0,
                 "C": 1.0, "CO": 0.0, "NaCl": 2.0, "NaCl_aq": 0.0}, T=700.0)
    st.aw = 0.9
    for _ in range(200):
        st.aw = rng.uniform(0.3, 0.95)
        RX.advance(rows, st, rng.uniform(5, 950), 10 ** rng.uniform(-1, 4))
        for sp, v in st.a.items():
            floor = -st.bg.get(sp, 0.0) * st.V if D.PHASE[sp] == "g" else 0.0
            assert np.all(v >= floor - 1e-9), (sp, v)


def test_adiabatic_cell_temperature_rise_equals_dH_m_over_C():
    row = CAT["slake"].replace(rate=RX.Const(1e9))           # runs to completion in one step
    st = _state({"CaO": 5.0, "H2O": 10.0}, T=20.0, C_other=50.0)
    C_before = float(st.C)
    ext = RX.advance([row], st, 20.0, 1.0)["slake"]
    assert float(ext) == pytest.approx(5.0)                  # all the CaO, no more
    dE = 1150.0 * 5.0
    assert float(st.E) == pytest.approx(dE, rel=1e-12)
    assert float(st.T) == pytest.approx(20.0 + dE / float(st.C), rel=1e-12)
    # and with equal heat capacities the rise is exactly dH m / C
    assert float(st.C) != C_before                           # composition changed C: that is real


def test_codeonly_semenov_a_big_exothermic_cell_runs_away_a_small_one_does_not():
    base = RX.Row("ox", "bulk", "A", (("wood", 1.0), ("O2", 0.0)), (("ash", 1.0),), 8000.0,
                  RX.Arrhenius.from_point(1e-4, 100.0, 90000.0), first_order=True)

    def run(size):
        st = _state({"wood": 10.0 * size ** 3, "ash": 0.0}, T=100.0, C_other=0.0)
        peak = 100.0
        for _ in range(4000):
            RX.advance([base], st, st.T, 1.0)
            loss = 0.5 * 6 * size ** 2 * 1e-2 * (st.T - 20.0)        # J per s from the surface
            st.heat(-loss)
            peak = max(peak, float(st.T))
        return peak
    peaks = [run(s) for s in (0.2, 0.5, 1.0, 2.0)]
    assert peaks[0] < 120.0                                  # small: the surface carries it off
    assert peaks[-1] > peaks[0] + 100.0                      # big: it heats itself
    assert all(b >= a - 1e-6 for a, b in zip(peaks, peaks[1:]))   # monotone in size


# ---- the gates ------------------------------------------------------------------------
def test_aw_window_stops_carbonation_outside_50_to_80_percent():
    row = CAT["carbonate"]
    made = []
    for aw in (0.3, 0.45, 0.55, 0.7, 0.79, 0.85, 0.95):
        st = _state({"Ca(OH)2": 10.0, "CO2": 2.0, "CaCO3": 0.0, "H2O": 0.0}, aw=aw)
        made.append(float(RX.advance([row], st, 20.0, 100.0)["carbonate"]))
    inside = [m for aw, m in zip((0.3, 0.45, 0.55, 0.7, 0.79, 0.85, 0.95), made) if 0.5 <= aw <= 0.8]
    outside = [m for aw, m in zip((0.3, 0.45, 0.55, 0.7, 0.79, 0.85, 0.95), made) if not 0.5 <= aw <= 0.8]
    assert all(m > 0 for m in inside) and all(m == 0.0 for m in outside)


def test_rust_needs_wet_and_above_zero():
    row = CAT["rust"]
    ok = _state({"Fe": 10.0, "O2": 5.0, "H2O": 5.0, "FeOOH": 0.0}, T=15.0, aw=0.9)
    dry = _state({"Fe": 10.0, "O2": 5.0, "H2O": 5.0, "FeOOH": 0.0}, T=15.0, aw=0.5)
    ice = _state({"Fe": 10.0, "O2": 5.0, "H2O": 5.0, "FeOOH": 0.0}, T=-5.0, aw=0.9)
    assert float(RX.advance([row], ok, 15.0, 1e5)["rust"]) > 0
    assert float(RX.advance([row], dry, 15.0, 1e5)["rust"]) == 0.0
    assert float(RX.advance([row], ice, -5.0, 1e5)["rust"]) == 0.0


def test_dissolve_rows_never_pass_the_solubility():
    row = CAT["dissolve_nacl"]
    for dt in (1.0, 1e3, 1e9):
        st = _state({"NaCl": 100.0, "NaCl_aq": 0.0, "H2O": 100.0}, T=20.0)
        RX.advance([row], st, 20.0, dt)
        b = 0.359
        assert float(st.get("NaCl_aq")) <= b * 100.0 * (1 + 1e-9)
        assert float(st.get("NaCl_aq") + st.get("NaCl")) == pytest.approx(100.0, rel=1e-12)
    # the solid takes heat as it dissolves (NaCl +66 J/g)
    st = _state({"NaCl": 100.0, "NaCl_aq": 0.0, "H2O": 100.0}, T=20.0)
    E0 = float(st.E)
    d = float(RX.advance([row], st, 20.0, 1e9)["dissolve_nacl"])
    assert float(st.E) - E0 == pytest.approx(-66.0 * d)


# ---- the legacy adapter ----------------------------------------------------------------
def test_all_legacy_reactions_become_rows_bitwise_equal_to_np_interp():
    rows = RX.legacy_rows()
    assert len(rows) == len(MAT.REACTIONS) == 8
    for row in rows:
        f, s = row.legacy
        pts, spend, heat, residue = MAT.REACTIONS[(f, s)]
        xs, ys = MAT._RATE_PTS[(f, s)]
        for pot in np.linspace(0.0, 1.0, 101):
            assert row.rate.value(20.0, pot) == np.interp(pot, xs, ys)         # bitwise
            assert row.rate.value(95.0, pot) == np.interp(pot, xs, ys)         # no T term
        assert row.spend == spend and row.dH == heat and row.residue == residue
        assert row.kind == "contact" and not row.closed
    assert {r.legacy for r in rows} == set(MAT.REACTIONS)
    # explicit arguments give the same rows
    again = RX.legacy_rows(MAT.REACTIONS, MAT._RATE_PTS)
    assert [r.id for r in again] == [r.id for r in rows]


# ---- burning -------------------------------------------------------------------------------
def test_burn_rows_close_mass_and_match_the_o2_demand():
    for name, mid in (("wood", MAT.WOOD), ("flesh", MAT.FLESH), ("leaf", MAT.LEAF),
                      ("char", MAT.CHAR), ("rope", MAT.ROPE)):
        row = RX.burn_row(name)
        assert RX.check(row)[0] < 1e-12
        o2 = dict(row.reactants)["O2"]
        ref = MAT.O2_RATIO[mid]
        tol = 0.08 if name == "wood" else 0.02
        assert abs(o2 / ref - 1) < tol, (name, o2, ref)
        assert RX.burn_row(mid).id == row.id                  # by material id as well
        assert row.dH == MAT.SOLID[mid][4]                    # the repo's burn column, unchanged
    w = dict(RX.burn_row("wood").products)
    assert 0.5 <= w["H2O_g"] <= 0.6
    assert 1.6 < w["CO2"] < 1.9


def test_burning_books_the_vapour_it_makes():
    row = RX.burn_row("wood").replace(rate=RX.Const(0.5), drivers=(RX.Linear("wood"),))
    st = _state({"wood": 20.0, "O2": 40.0, "CO2": 0.0, "H2O_g": 0.0, "C": 0.0, "ash": 0.0},
                C_other=500.0)
    xi = float(RX.advance([row], st, 20.0, 1.0)["burn:wood"])
    vap = float(st.get("H2O_g"))
    assert vap == pytest.approx(xi * dict(row.products)["H2O_g"])
    assert float(st.made["water"]) == pytest.approx(vap)
    assert float(st.made["latent"]) == pytest.approx(D.L0 * vap)
    assert float(st.E) == pytest.approx(16000.0 * xi)          # the cell still gets exactly the repo's J/g
    assert float(st.water_identity()) == pytest.approx(0.0, abs=1e-12)
    # E + L0 vapour - made_latent: the vapour's latent is paid by the book
    assert float(st.total_energy()) == pytest.approx(16000.0 * xi)


# ---- the ledgers (39b) -----------------------------------------------------------------------
def test_ledgers_hold_through_random_rows_freezes_boils_and_condensations():
    rng = np.random.default_rng(21)
    burn = RX.burn_row("wood").replace(rate=RX.Const(0.02), drivers=(RX.Linear("wood"),))
    ferm = CAT["ferment"].replace(rate=RX.Const(0.01))
    rows = [CAT["slake"], CAT["carbonate"], CAT["rust"], ferm, burn]
    st = _state({"H2O": 120.0, "H2O_s": 12.0, "H2O_g": 1.0, "CaO": 15.0, "Ca(OH)2": 12.0,
                 "CaCO3": 0.0, "CO2": 6.0, "glucose": 15.0, "ethanol": 0.0, "Fe": 8.0,
                 "O2": 9.0, "FeOOH": 0.0, "wood": 12.0, "C": 0.0, "ash": 0.0},
                T=5.0, C_other=400.0)
    w0, e0 = float(st.water_identity()), float(st.energy_identity())
    ext = 0.0                                                # heat from outside, booked
    cover = {"evap": 0.0, "cond": 0.0, "freeze": 0.0, "boil": 0.0}
    scale = 1.0 + abs(e0) + 1e5
    for step in range(600):
        st.aw = rng.uniform(0.4, 0.95)
        act = rng.integers(0, 6)
        if act == 0:
            RX.advance(rows, st, None, 10 ** rng.uniform(-1, 2))
        elif act == 1:
            cover["evap"] += abs(float(st.evaporate(rng.uniform(0, 2.0))))
        elif act == 2:
            cover["cond"] += abs(float(st.evaporate(-rng.uniform(0, 2.0))))   # condense
        elif act == 3:
            cover["freeze"] += abs(float(st.freeze_melt(0.0)))
        elif act == 4:
            q = rng.uniform(-9000, 5000)                      # outside heat or cold
            st.E = st.E + q
            ext += q
            cover["freeze"] += abs(float(st.freeze_melt(0.0)))
        else:
            cover["boil"] += float(st.boil(100.0))
        assert abs(float(st.water_identity()) - w0) < 1e-9, step
        assert abs(float(st.energy_identity()) - ext - e0) < 1e-9 * scale, step
        assert float(st.get("H2O")) >= -1e-9 and float(st.get("H2O_g")) >= -1e-9
    # the books are not idle: rows made water and the ice moved
    assert abs(float(st.made["water"])) > 1e-3
    assert float(st.made["latent"]) != 0.0
    assert all(v > 1.0 for v in cover.values()), cover        # every kind of event happened


def test_evaporate_and_boil_on_a_lump_keep_their_own_books():
    st = _state({"H2O": 50.0, "H2O_g": 0.0, "H2O_s": 0.0}, T=100.0, C_other=20.0)
    e0 = float(st.total_energy())
    st.evaporate(5.0)
    assert float(st.total_energy()) == pytest.approx(e0, abs=1e-9)
    st.evaporate(-3.0)
    assert float(st.total_energy()) == pytest.approx(e0, abs=1e-9)
    st.E = st.E + 20000.0
    e1 = float(st.total_energy())
    g = float(st.boil(100.0))
    assert g > 0 and float(st.total_energy()) == pytest.approx(e1, abs=1e-9)
    assert float(st.total_water()) == pytest.approx(50.0)


# ---- closed forms -------------------------------------------------------------------------
def test_closed_forms_agree_with_stepping():
    m, k, t = 7.0, 0.003, 600.0
    stepped = m
    for _ in range(600):
        stepped -= stepped * float(-np.expm1(-k * 1.0))
    assert m - stepped == pytest.approx(float(RX.first_order_extent(m, k, t)), rel=1e-9)


def test_u_iso9223_and_carbonation_closed_forms():
    d1 = float(RX.iso9223_depth_um(1.0, 37.0, 0.55))
    d10 = float(RX.iso9223_depth_um(10.0, 37.0, 0.55))
    assert d1 == pytest.approx(37.0) and d10 / d1 == pytest.approx(10 ** 0.55)
    assert float(RX.carbonation_depth_mm(86400.0)) == pytest.approx(1.0)
    assert float(RX.carbonation_depth_mm(4 * 86400.0)) == pytest.approx(2.0)
