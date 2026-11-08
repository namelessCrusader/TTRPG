"""T7 clothing (no World). Tags: test_u_* physical data vs an independent number
(source unverified until A-S marks it), test_codeonly_* code vs its own table;
unmarked = exactness / closed-form / arithmetic identity."""
import copy
import math

import pytest

from src.voxel import bodyair as B
from src.voxel import clothing as CL


def shirt(fibre="cotton", clo=1.0, cover=1.0, zone="torso", layer=0, name=None, area=1.0):
    return CL.garment(name or f"{fibre}{layer}", fibre, CL.thickness_for_clo(clo), {zone: cover},
                      area_m2={zone: area}, layer=layer)


def full_water(g):
    g = copy.deepcopy(g)
    g["water_g"] = CL.water_capacity_g(g)
    return g


def test_dry_garment_clo_is_thickness_over_conductivity():
    g = shirt(clo=1.0)
    assert CL.clo(g) == pytest.approx(1.0, rel=1e-12)
    assert CL.resistance(g) == pytest.approx(0.155, rel=1e-12)
    assert CL.thickness_for_clo(1.0) == pytest.approx(0.00651, rel=0.01)       # about 6.5 mm per clo
    assert CL.saturation(g) == 0.0


def test_u_wet_cotton_keeps_at_most_25pct_wet_wool_at_least_50pct():
    c, w = shirt("cotton"), shirt("wool")
    assert CL.clo(c, 1.0) / CL.clo(c) <= 0.25
    assert CL.clo(w, 1.0) / CL.clo(w) >= 0.50
    # synthetic and leather sit between, and insulation falls monotonically with wetness
    for f in CL.K_FIBRE:
        g = shirt(f)
        vals = [CL.clo(g, s) for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
        assert all(b < a for a, b in zip(vals, vals[1:])), f
    assert CL.clo(shirt("synthetic"), 1.0) / CL.clo(shirt("synthetic")) > CL.clo(c, 1.0) / CL.clo(c)


def test_saturation_comes_from_the_water_it_holds():
    g = shirt("cotton", area=1.8)
    cap = CL.water_capacity_g(g)
    assert cap == pytest.approx(CL.SOAKED_G_PER_G["cotton"] * g["dry_mass_g_m2"] * 1.8)
    g["water_g"] = 0.5 * cap
    assert CL.saturation(g) == pytest.approx(0.5)
    assert CL.clo(g) == pytest.approx(CL.clo(g, 0.5))
    g["water_g"] = 3 * cap
    assert CL.saturation(g) == 1.0


def test_layers_add_in_series_with_an_air_gap():
    a, b = shirt("cotton", 0.5, layer=0, name="a"), shirt("wool", 1.0, layer=1, name="b")
    R1 = CL.ensemble_resistance([a], "torso")
    R2 = CL.ensemble_resistance([a, b], "torso")
    want = CL.resistance(a) + CL.resistance(b) + CL.AIR_GAP_R
    assert R2 == pytest.approx(want, rel=1e-9)
    assert R1 == pytest.approx(CL.resistance(a), rel=1e-9)
    assert CL.series_resistance([0.1, 0.2, 0.3]) == pytest.approx(0.6)
    # the order of the list does not matter, the layer index does
    assert CL.ensemble_resistance([b, a], "torso") == pytest.approx(R2, rel=1e-12)


def test_coverage_acts_in_parallel_with_bare_skin():
    full = shirt("wool", 2.0, cover=1.0)
    half = shirt("wool", 2.0, cover=0.5)
    h = CL.H_STATIC
    U_full = 1.0 / CL.total_insulation(CL.resistance(full), h)
    U_bare = 1.0 / CL.total_insulation(0.0, h)
    U_half = 0.5 * U_full + 0.5 * U_bare
    assert 1.0 / CL.ensemble_total([half], "torso") == pytest.approx(U_half, rel=1e-9)
    # bare skin bypasses: half-covered is far less insulated than half the clo
    assert CL.ensemble_total([half], "torso") < 0.6 * CL.ensemble_total([full], "torso")
    # a zone with no garment is bare
    assert CL.ensemble_resistance([half], "left hand") == 0.0
    assert CL.ensemble_total([], "torso") == pytest.approx(1.0 / h)


def test_nested_coverage_bands():
    coat = shirt("wool", 1.0, cover=0.6, layer=1, name="coat")
    tee = shirt("cotton", 0.3, cover=1.0, layer=0, name="tee")
    bands = CL._band_resistances([coat, tee], "torso", None)
    assert [w for w, _ in bands] == pytest.approx([0.6, 0.4])
    assert bands[0][1] == pytest.approx(CL.resistance(coat) + CL.resistance(tee) + CL.AIR_GAP_R)
    assert bands[1][1] == pytest.approx(CL.resistance(tee))


def test_wind_reduces_total_insulation_once():
    g = shirt("wool", 2.0)
    I0 = CL.ensemble_total([g], "torso", 0.0)
    for v in (0.1, 0.15):
        assert CL.ensemble_total([g], "torso", v) == pytest.approx(I0, rel=1e-12)    # below 0.15 m/s: nothing
    for v in (1.0, 3.0, 8.0):
        I = CL.ensemble_total([g], "torso", v)
        assert I / I0 == pytest.approx(CL.wind_factor(v), rel=1e-9)                  # once, not twice
        assert I < I0
    # the ISO parabola is held past its minimum: more wind never adds insulation
    assert CL.wind_factor(8.0) == pytest.approx(CL.wind_factor(20.0)) and CL.wind_factor(8.0) < 1.0
    assert CL.wind_factor(3.0) > CL.wind_factor(5.0)
    d = 3.0 - 0.15
    assert CL.wind_factor(3.0) == pytest.approx(math.exp(-0.281 * d + 0.044 * d * d))
    assert CL.wind_factor(0.0) == 1.0


def test_wet_state_override_changes_the_ensemble():
    g = shirt("cotton", 2.0)
    dry = CL.ensemble_total([g], "torso")
    soaked = CL.ensemble_total([g], "torso", wet={g["name"]: 1.0})
    assert soaked < 0.6 * dry
    g2 = full_water(g)
    assert CL.ensemble_total([g2], "torso") == pytest.approx(soaked, rel=1e-12)


def test_h3_series_resistance_is_exact():
    """G = 1/(1/(h A f) + R/A) and the ensemble's R_cl reproduces the combined conductance."""
    h, A, f, R = 8.3, 0.2, 1.15, 0.155
    assert CL.h3_conductance(h, A, f, R) == pytest.approx(1.0 / (1.0 / (h * A * f) + R / A), rel=1e-14)
    assert CL.h3_conductance(h, A, f, 0.0) == pytest.approx(h * A * f)
    g = shirt("wool", 1.5)
    Rcl = CL.ensemble_resistance([g], "torso", 0.0)
    fcl = float(B.f_cl_iso(Rcl))
    G = CL.h3_conductance(CL.H_STATIC, 1.0, fcl, Rcl)
    assert 1.0 / G == pytest.approx(CL.ensemble_total([g], "torso"), rel=1e-12)
    # with wind the same R_cl and f_cl(R_cl) reproduce the corrected total
    Rw = CL.ensemble_resistance([g], "torso", 4.0)
    Gw = CL.h3_conductance(CL.H_STATIC, 1.0, float(B.f_cl_iso(Rw)), Rw)
    assert 1.0 / Gw == pytest.approx(CL.ensemble_total([g], "torso", 4.0), rel=1e-12)
    assert Rw < Rcl


def test_u_a_1_clo_ensemble_total_is_plausible():
    """1 clo of garment + still boundary air (about 0.7 clo, ISO 9920's I_a 0.11-0.12 m2K/W)
    is 1.6-1.8 clo of total insulation."""
    g = shirt("cotton", 1.0)
    I = CL.ensemble_total([g], "torso") / CL.CLO
    assert 1.6 < I < 1.8, I


def test_wet_step_wicks_in_with_mass_conserved():
    g = shirt("cotton", 1.0, area=1.8)
    cap = CL.water_capacity_g(g)
    pool = 10.0 * cap
    for _ in range(300):
        g, taken = CL.wet_step(g, pool, 1.0)
        pool -= taken
        assert g["water_g"] <= cap + 1e-9
    assert g["water_g"] == pytest.approx(cap, rel=1e-3)
    assert pool + g["water_g"] == pytest.approx(10.0 * cap, rel=1e-12)
    assert CL.wet_step(g, 5.0, 1.0, contact=False)[1] == 0.0
    # limited by the source: a little rain wets it a little
    g0 = shirt("cotton", 1.0, area=1.8)
    g1, t = CL.wet_step(g0, 0.01, 60.0)
    assert t == pytest.approx(0.01) and g1["water_g"] == pytest.approx(0.01) and g0["water_g"] == 0.0
    # wool wicks more slowly than cotton
    gw, tw = CL.wet_step(shirt("wool", 1.0, area=1.8), 1e6, 5.0)
    gc, tc = CL.wet_step(shirt("cotton", 1.0, area=1.8), 1e6, 5.0)
    assert tw / CL.water_capacity_g(shirt("wool", 1.0, area=1.8)) < tc / CL.water_capacity_g(shirt("cotton", 1.0, area=1.8))


def test_dry_step_conserves_water_and_energy():
    g = full_water(shirt("cotton", 1.0, area=1.8))
    w0 = g["water_g"]
    L = float(B.latent_heat(30.0))
    g2, ev, J = CL.dry_step(g, 100.0, 30.0, 10.0)
    assert ev == pytest.approx(100.0 * 10.0 / L * 1000.0)
    assert g2["water_g"] + ev == pytest.approx(w0, rel=1e-12)
    assert J == pytest.approx(ev * L / 1000.0, rel=1e-12)
    assert g["water_g"] == w0                                 # the input is not mutated
    g3, ev3, J3 = CL.dry_step(g, 1e9, 30.0, 10.0)
    assert g3["water_g"] == pytest.approx(0.0, abs=1e-12) and ev3 == pytest.approx(w0)
    assert CL.dry_step(g, -5.0, 30.0, 10.0)[1:] == (0.0, 0.0)       # no condensation here


def test_garment_mass_adds_water_weight():
    g = shirt("cotton", 1.0, area=1.8)
    dry = CL.mass_g(g)
    assert dry == pytest.approx(250.0 * 1.8)
    assert CL.mass_g(full_water(g)) == pytest.approx(dry + CL.water_capacity_g(g))


def test_garment_is_a_plain_value():
    g = shirt("wool", 1.0)
    assert copy.deepcopy(g) == g and isinstance(g, dict)
    with pytest.raises(ValueError):
        CL.garment("x", "nylon", 0.01, {"torso": 1.0})


def test_codeonly_closed_form_series_solution_matches_bisection_and_bodyair_fcl():
    for R in (0.0, 0.03, 0.0779, 0.0781, 0.155, 0.4, 0.8):
        assert CL._fcl(R) == pytest.approx(float(B.f_cl_iso(R)), rel=1e-15)
        I = CL.total_insulation(R)
        lo, hi = 0.0, I
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if CL.total_insulation(mid) < I:
                lo = mid
            else:
                hi = mid
        assert CL._solve_series(I, CL.H_STATIC) == pytest.approx(0.5 * (lo + hi), abs=1e-9)


# ── water capacity from a textile number (regression: it was the whole pore volume, 23x) ──
def _wear(fibre, dry_g_m2, area_m2, clo):
    return CL.garment("x", fibre, CL.thickness_for_clo(clo), {"torso": 1.0}, area_m2={"torso": area_m2},
                      dry_mass_g_m2=dry_g_m2)


def test_u_cotton_shirt_soaked_holds_about_one_to_two_times_its_own_weight():
    """Cotton T-shirt: 160 g/m2, 1.4 m2 (both faces of a torso piece), about 0.25 clo -> 224 g dry.
    Soaked-and-dripping uptake of cotton fabric is 1 to 2 g/g (retention alone is 0.4-0.5 g/g,
    DIN 53814): 224 to 450 g. The old pore-volume formula gave about 1500 g."""
    g = _wear("cotton", 160.0, 1.4, 0.25)
    dry = CL.mass_g(g)
    cap = CL.water_capacity_g(g)
    assert dry == pytest.approx(224.0)
    assert 0.8 * dry <= cap <= 2.0 * dry
    assert cap == pytest.approx(336.0)
    assert cap > CL.REGAIN_G_PER_G["cotton"] * dry * 3          # more than retention-level water


def test_u_wool_sweater_soaked_holds_more_than_cotton_per_gram_and_up_to_three_times_its_weight():
    g = _wear("wool", 450.0, 1.6, 1.0)                  # a 720 g sweater, 1 clo
    dry = CL.mass_g(g)
    cap = CL.water_capacity_g(g)
    assert dry == pytest.approx(720.0)
    assert 1.0 * dry <= cap <= 3.0 * dry
    assert cap / dry > CL.water_capacity_g(_wear("cotton", 450.0, 1.6, 1.0)) / dry
    # wool takes the most water from air as well (regain 16 vs 8.5 percent, 65 percent RH)
    assert CL.regain_g(g) / dry == pytest.approx(0.16)
    assert CL.regain_g(_wear("cotton", 450.0, 1.6, 1.0)) / dry == pytest.approx(0.085)
    assert CL.regain_g(g) < cap                          # air moisture is a small part of the soaked load


@pytest.mark.parametrize("seed", range(40))
def test_capacity_never_exceeds_the_pores_and_scales_with_dry_mass(seed):
    rng = __import__("random").Random(seed)
    fibre = rng.choice(sorted(CL.K_FIBRE))
    mass = rng.uniform(40.0, 900.0)
    area = rng.uniform(0.1, 2.5)
    clo = rng.uniform(0.02, 3.0)
    void = rng.uniform(0.3, 0.95)
    g = CL.garment("r", fibre, CL.thickness_for_clo(clo), {"torso": rng.uniform(0.1, 1.0)},
                   area_m2={"torso": area}, void=void, dry_mass_g_m2=mass)
    cap = CL.water_capacity_g(g)
    a = g["covers"]["torso"] * area
    assert cap <= g["void"] * g["thickness_m"] * a * 1e6 * (1 + 1e-12)       # never more than the pores
    assert cap <= CL.SOAKED_G_PER_G[fibre] * mass * a * (1 + 1e-12)
    assert cap > CL.regain_g(g) or cap == pytest.approx(g["void"] * g["thickness_m"] * a * 1e6)
    g2 = CL.garment("r", fibre, g["thickness_m"], g["covers"], area_m2=g["area_m2"], void=void,
                    dry_mass_g_m2=2.0 * mass)
    # heavier fabric of the same bulk holds at least as much, and at most twice
    assert cap <= CL.water_capacity_g(g2) <= 2.0 * cap * (1 + 1e-12)


def test_a_shower_of_rain_now_reaches_wet_insulation_loss():
    """The report's point: 300 g of water on a 1 clo cotton shirt (1.4 m2) was s = 0.05 (a barely
    wet shirt) under the old capacity; now it is most of a soaked shirt."""
    g = _wear("cotton", 250.0, 1.4, 1.0)
    g["water_g"] = 300.0
    assert CL.saturation(g) == pytest.approx(300.0 / (1.5 * 250.0 * 1.4))
    assert CL.saturation(g) > 0.5
    assert CL.clo(g) < 0.6 * CL.clo(_wear("cotton", 250.0, 1.4, 1.0))
