"""Wave 13, T1: the selectors (src/voxel/partition.py).

Tags as in test_mixrules.py: [E] echo, [C] code-only, [U] independent but the
number's source is unverified (test_u_*), [I] independent exact / closed form.
Only [I] lines gate a scene.
"""
import os

import numpy as np
import pytest

from src.voxel import partition as P
from src.voxel import species as SP

D, C, G, B = SP.DISSOLVED, SP.CRYSTAL, SP.GRAIN, SP.BOUND
S = SP.BY_NAME
WATER, ETOH, NACL, FE, MAG, SN = (S[n] for n in ("water", "ethanol", "sodium_chloride", "iron",
                                                  "magnetite", "tin"))
QUARTZ = 2650.0
RHO_W, MU_W = 998.0, 1.002
# 20 wt percent NaCl at 20 C: density 1148 kg/m3 (CRC table, (u)); relative viscosity 1.562
# (Kestin, Khalifa and Correia 1981, SOURCED, tests/data/sourced_numbers.json key
# nacl.rel_viscosity_20C; the old 1.99 was a recalled (u) table 27 percent high).
RHO_BRINE, MU_BRINE = 1148.0, 1.562 * MU_W


# ---- density: Stokes and beyond (test 26) ----------------------------------------------
@pytest.mark.parametrize("d,want", [(2e-6, 3.6e-6), (20e-6, 3.6e-4), (1e-4, 9.0e-3)])
def test_i_stokes_speeds_in_water_match_the_spec_numbers(d, want):
    """[I] 2 um clay 3.6e-6 m/s, 20 um silt 3.6e-4, 0.1 mm sand 9.0e-3, within 1 percent."""
    assert P.terminal_speed(d, QUARTZ, RHO_W, MU_W) == pytest.approx(want, rel=0.01)


def test_i_stokes_closed_form_below_re_one():
    """[I] v = (rho_p - rho_f) g d^2 / (18 mu) exactly where Re <= 1."""
    d = 3e-5
    want = (QUARTZ - RHO_W) * 9.81 * d * d / (18 * 1.002e-3)
    assert P.terminal_speed(d, QUARTZ, RHO_W, MU_W) == pytest.approx(want, rel=1e-14)


def test_u_one_millimetre_sand_uses_the_corrected_drag_not_stokes():
    """[U] corrected 0.155 m/s within 15 percent of the literature value (u); Stokes alone
    would say 0.9 m/s, which is wrong."""
    v = P.terminal_speed(1e-3, QUARTZ, RHO_W, MU_W)
    assert v == pytest.approx(0.155, rel=0.15)
    stokes = (QUARTZ - RHO_W) * 9.81 * 1e-6 / (18 * 1.002e-3)
    assert stokes > 0.8 and v < 0.3 * stokes


def test_i_the_corrected_speed_solves_its_own_drag_balance():
    """[I] at the returned speed, Cd(Re) v^2 = 4 (rho_p - rho_f) g d / (3 rho_f)."""
    d = 2e-3
    v = float(P.terminal_speed(d, QUARTZ, RHO_W, MU_W))
    Re = RHO_W * v * d / 1.002e-3
    cd = max(24 / Re * (1 + 0.15 * Re ** 0.687), 0.44)
    assert cd * v * v == pytest.approx(4 * (QUARTZ - RHO_W) * 9.81 * d / (3 * RHO_W), rel=1e-9)


def test_u_silt_in_brine_settles_1_7_times_slower():
    """[U] 20 um in 1148 kg/m3 and 1.562 x 1.002 mPa s: 2.09e-4 m/s within 2 percent (Stokes,
    exact for the inputs; only the brine density is (u)); 1.72 times slower than in water."""
    v = P.terminal_speed(20e-6, QUARTZ, RHO_BRINE, MU_BRINE)
    assert v == pytest.approx(2.09e-4, rel=0.02)
    assert P.terminal_speed(20e-6, QUARTZ, RHO_W, MU_W) / v == pytest.approx(1.72, rel=0.02)


def test_the_brine_viscosity_here_is_the_seed_viscosity_not_a_second_copy():
    """[C] the test's 1.562 is the species seed's relative viscosity at 20 wt percent."""
    from src.voxel import species_seed as seed
    assert dict(seed._NACL_MU_REL)[0.20] * MU_W == pytest.approx(MU_BRINE, rel=1e-12)


def test_i_floating_reverses_the_sign_and_a_neutral_grain_does_not_move():
    """[I] an egg (rho < rho of the mix) rises; equal density gives 0; d = 0 gives 0."""
    assert P.terminal_speed(5e-3, 900.0, RHO_BRINE, MU_BRINE) < 0
    assert P.terminal_speed(5e-3, RHO_W, RHO_W, MU_W) == 0.0
    assert P.terminal_speed(0.0, QUARTZ, RHO_W, MU_W) == 0.0
    phi, direction = P.settle_phi(5e-3, 900.0, RHO_BRINE, MU_BRINE, 1.0, 0.05)
    assert direction == -1 and 0 < phi <= 1


def test_i_the_settling_share_never_exceeds_one_and_follows_v_dt_over_dz():
    """[I] property test over sizes, steps and cell heights; and the plain formula."""
    rng = np.random.default_rng(1)
    d = 10.0 ** rng.uniform(-6, -2, 500)
    dt = 10.0 ** rng.uniform(-2, 3, 500)
    dz = 10.0 ** rng.uniform(-3, 0, 500)
    phi, _ = P.settle_phi(d, QUARTZ, RHO_W, MU_W, dt, dz, phi_vol=rng.uniform(0, 0.6, 500))
    assert (phi >= 0).all() and (phi <= 1.0).all()
    one, _ = P.settle_phi(2e-5, QUARTZ, RHO_W, MU_W, 10.0, 0.05)
    assert one == pytest.approx(3.594e-4 * 10.0 / 0.05, rel=1e-3)


def test_i_richardson_zaki_exponent_is_continuous_at_the_joins_and_has_the_end_values():
    """[I] 4.65 at low Re and 2.4 at high Re (sourced: 4.65 and 2.39); the joins at Re 0.2,
    1 and 500 are within 2 percent of each other."""
    assert P.richardson_zaki_n(0.01) == 4.65 and P.richardson_zaki_n(5000.0) == 2.4
    for re_join in (0.2, 1.0, 500.0):
        a, b = P.richardson_zaki_n(re_join * (1 - 1e-9)), P.richardson_zaki_n(re_join * (1 + 1e-9))
        assert abs(a / b - 1) < 0.02
    re = np.logspace(-2, 3.5, 400)
    n = P.richardson_zaki_n(re)
    assert n.min() >= 2.35 and n.max() <= 4.66


def test_i_crowding_slows_settling_monotonically():
    """[I] v0 (1 - phi)^n falls as the volume fraction rises."""
    v = P.hindered_speed(1.0, np.linspace(0, 0.6, 20), 0.1)
    assert (np.diff(v) < 0).all() and v[0] == 1.0


def test_i_density_selector_acts_only_on_crystal_and_grain():
    """[I] form mask: DISSOLVED and BOUND entries get phi 0 from the density rule."""
    forms = np.array([D, C, G, B])
    phi, _ = P.density_phi(forms, 1e-3, QUARTZ, RHO_W, MU_W, 1.0, 0.05)
    assert list(phi > 0) == [False, True, True, False]


# ---- size: sieve and winnow (tests 27, 28) -------------------------------------------------
def test_i_sieve_passes_nothing_at_or_above_the_aperture():
    """[I] exactly 0 for d >= a, for every dt and rate."""
    d = np.array([1e-3, 1.0001e-3, 2e-3, 5e-3])
    phi = P.sieve_phi(np.full(4, C), d, 1e-3, 0.6, 50.0, 1e6)
    assert (phi == 0.0).all()


def test_i_sieve_is_monotone_in_size_and_in_time():
    """[I] smaller grains pass more; a longer shake passes more."""
    d = np.linspace(1e-5, 9.9e-4, 50)
    phi = P.sieve_phi(np.full(50, G), d, 1e-3, 0.6, 20.0, 1.0)
    assert (np.diff(phi) <= 0).all()
    t = P.sieve_phi(np.full(30, G), 3e-4, 1e-3, 0.6, 0.5, np.linspace(0.1, 3, 30))
    assert (np.diff(t) > 0).all() and (t < 1).all()


def test_i_sieve_oversize_leak_stays_zero_over_1000_shakes():
    """[I] the logistic of v1 leaked 14 percent at d = a; this one never does."""
    grams = 100.0
    for _ in range(1000):
        grams -= grams * P.sieve_phi(G, 1.0e-3, 1.0e-3, 0.6, 50.0, 1.0)
    assert grams == 100.0


def test_i_sieve_shape_factor_makes_a_long_grain_pass_less():
    """[I] a grain with aspect 2 behaves as if twice as wide."""
    a = P.sieve_phi(G, 3e-4, 1e-3, 0.6, 20.0, 1.0)
    b = P.sieve_phi(G, 3e-4, 1e-3, 0.6, 20.0, 1.0, aspect=2.0)
    assert b < a


def test_i_winnow_is_zero_up_to_terminal_speed_then_linear_and_capped():
    """[I] phi = min(1, (u - v_t) dt / dz), 0 for u <= v_t."""
    vt = 5.0
    assert P.winnow_phi(C, 3.0, vt, 1.0, 0.2) == 0.0
    assert P.winnow_phi(C, 5.0, vt, 1.0, 0.2) == 0.0
    assert P.winnow_phi(G, 6.0, vt, 0.1, 0.2) == pytest.approx(0.5)
    assert P.winnow_phi(G, 60.0, vt, 1.0, 0.2) == 1.0
    assert P.winnow_phi(B, 60.0, vt, 1.0, 0.2) == 0.0


def test_u_wheat_flies_and_sand_does_not_in_a_9_metre_per_second_wind_in_air():
    """[U] wheat v_t about 9 m/s (u): a 4 mm grain of 1300 kg/m3 in air has 11 m/s here, chaff
    (light, 80 kg/m3) about 3; sand of 0.25 mm falls at over 1 m/s."""
    assert 6.0 < float(P.air_terminal_speed(4e-3, 1300.0)) < 14.0
    assert float(P.air_terminal_speed(5e-3, 80.0)) < 4.0           # chaff: a few m/s
    assert float(P.air_terminal_speed(2.5e-4, 2650.0)) > 1.0       # sand


# ---- dissolving (test 29) --------------------------------------------------------------------
def test_i_dissolve_follows_the_d_squared_closed_form():
    """[I] d^2 = d0^2 - 8 D (c_sat - c) t / rho_p to 1e-6 relative."""
    d0, rho, Dm, cs, c, t = 1e-3, 2165.0, 1.6e-9, 316.0, 50.0, 200.0
    want = np.sqrt(d0 ** 2 - 8 * Dm * (cs - c) * t / rho)
    assert P.dissolve_d(d0, rho, Dm, cs, c, t) == pytest.approx(want, rel=1e-6)


def test_i_nacl_1_mm_in_still_water_takes_535_seconds_by_the_formula():
    """[I] a formula check, not a prediction (real dissolving has convection): 535 s within 1 percent."""
    t = P.dissolve_time(1e-3, 2165.0, 1.6e-9, 316.0, 0.0)
    assert t == pytest.approx(535.0, rel=0.01)
    assert P.dissolve_d(1e-3, 2165.0, 1.6e-9, 316.0, 0.0, t) == pytest.approx(0.0, abs=1e-12)


def test_i_dissolving_stops_when_the_liquid_is_saturated():
    """[I] zero at and above c_sat; the mass share is 1 - (d/d0)^3 and grows with dt."""
    assert P.dissolve_phi(1e-3, 2165.0, 1.6e-9, 316.0, 316.0, 100.0) == 0.0
    assert P.dissolve_phi(1e-3, 2165.0, 1.6e-9, 316.0, 400.0, 100.0) == 0.0
    assert np.isinf(P.dissolve_time(1e-3, 2165.0, 1.6e-9, 316.0, 316.0))
    a = P.dissolve_phi(1e-3, 2165.0, 1.6e-9, 316.0, 0.0, 100.0)
    b = P.dissolve_phi(1e-3, 2165.0, 1.6e-9, 316.0, 0.0, 200.0)
    d = P.dissolve_d(1e-3, 2165.0, 1.6e-9, 316.0, 0.0, 100.0)
    assert 0 < a < b <= 1 and a == pytest.approx(1 - (d / 1e-3) ** 3)


def test_i_wash_takes_dissolved_dissolves_crystal_leaves_bound_and_insoluble_grain():
    """[I] spec 4.6 table, wash row."""
    forms = np.array([D, C, G, G, B])
    soluble = np.array([False, False, True, False, False])
    phi = P.wash_phi(forms, soluble, 0.4, 0.1)
    assert list(phi) == [0.4, 0.1, 0.1, 0.0, 0.0]


# ---- boiling and Antoine (test 30) -----------------------------------------------------------------
def test_i_antoine_rows_reproduce_the_normal_boiling_points():
    """[I] water 760.1 mmHg at 100 C and ethanol 761.5 at 78.37 C, within 0.5 percent of 760."""
    assert P.psat(WATER, 100.0) / P.MMHG == pytest.approx(760.0, rel=0.005)
    assert P.psat(ETOH, 78.37) / P.MMHG == pytest.approx(760.0, rel=0.005)


def test_i_ideal_relative_volatility_of_ethanol_over_water_is_2_3_at_78_15():
    """[I] 754.9 / 328.8 = 2.30 (spec 4.6 a, computed from the rows)."""
    assert P.relative_volatility(ETOH, WATER, 78.15) == pytest.approx(2.30, abs=0.02)


def test_i_ideal_boiling_point_of_a_pure_liquid_and_of_a_mixture():
    """[I] pure water boils at 100 C at 1 atm; a 50/50 mole mixture boils between the two."""
    t = P.boiling_point_ideal([[1.0, 0.0]], [WATER, ETOH])[0]
    assert t == pytest.approx(100.0, abs=0.2)
    t = P.boiling_point_ideal([[0.5, 0.5]], [WATER, ETOH])[0]
    assert 78.0 < t < 100.0
    # at its own boiling point the vapour mole fractions sum to 1
    y = P.vapour_fraction(np.array([[0.5, 0.5]]), np.array([[WATER, ETOH]]), np.array([t]),
                          np.array([101325.0]))
    assert y.sum() == pytest.approx(1.0, abs=1e-6)


def test_i_boiling_wine_sends_ethanol_up_first_and_leaves_the_salt():
    """[I] the vapour is richer in ethanol than the liquid; a species with no vapour
    pressure (NaCl) never leaves; BOUND never leaves."""
    cell = np.zeros(4, np.int64)
    sp = np.array([WATER, ETOH, NACL, S["tin"]])
    form = np.array([D, D, D, B])
    grams = np.array([80.0, 10.0, 5.0, 5.0])
    phi = P.boil_off_phi(cell, sp, form, grams, [85.0], [101325.0], [20.0])
    assert phi[2] == 0.0 and phi[3] == 0.0
    assert phi[1] > phi[0] > 0
    left = phi[0] * grams[0] + phi[1] * grams[1]
    assert left == pytest.approx(20.0, rel=1e-9)             # the boiled mass is exactly that
    assert phi[1] * grams[1] / left > grams[1] / (grams[0] + grams[1])


# ---- magnet (test 31) -----------------------------------------------------------------------------------
def _need(sp, B_T, T=20.0):
    t = P._mag_table("ref")[sp]
    return float(P.magnet_min_gradient(t[0], t[1], t[2], t[3], B_T, T))


def test_u_saturated_lower_bounds_are_g_over_sigma_s():
    """[U] iron 9.81/217.75 = 0.045 T/m and magnetite 9.81/92 = 0.107 (sigma_s are (u))."""
    assert _need(FE, 5.0) == pytest.approx(0.0450, rel=2e-3)
    assert _need(MAG, 5.0) == pytest.approx(0.107, rel=2e-2)


def test_u_iron_filings_need_0_17_tesla_per_metre_in_a_0_2_tesla_field():
    """[U] and about 0.65 T/m in 0.05 T (spec 4.6 f)."""
    assert _need(FE, 0.2) == pytest.approx(0.17, rel=0.05)
    assert _need(FE, 0.05) == pytest.approx(0.65, rel=0.03)


def test_i_magnet_demagnetising_field_for_iron_is_0_72_tesla():
    """[I] B_k = mu0 N sigma_s rho_s with N = 1/3."""
    t = P._mag_table("ref")[FE]
    assert P.MU0 / 3.0 * t[0] * t[3] == pytest.approx(0.72, abs=0.01)


def test_i_nothing_is_captured_above_the_curie_point_or_for_a_non_magnet():
    """[I] g(T) is 1 at 20 C and 0 at Tc; filings above 770 C stay in the sand; sand and
    tin are never captured; a magnet pulls CRYSTAL and GRAIN but not BOUND."""
    assert P.sigma_sat_T(217.75, 770.0, 0.36, 20.0) == pytest.approx(217.75, rel=1e-12)
    assert P.sigma_sat_T(217.75, 770.0, 0.36, 770.0) == 0.0
    assert P.sigma_sat_T(217.75, 770.0, 0.36, 900.0) == 0.0
    hot = P.magnet_phi(np.array([FE]), np.array([G]), 0.5, 100.0, 800.0)
    cold = P.magnet_phi(np.array([FE]), np.array([G]), 0.5, 100.0, 20.0)
    assert hot[0] == 0.0 and cold[0] == 1.0
    sp = np.array([S["quartz"], SN, FE, FE, FE, MAG])
    fo = np.array([G, G, G, C, B, G])
    got = P.magnet_phi(sp, fo, 0.5, 5.0, 20.0)
    assert list(got) == [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]


def test_i_magnet_threshold_rises_with_drag_and_falls_with_field_strength():
    """[I] more drag needs a steeper gradient; a stronger field needs less (until saturation)."""
    assert _need(FE, 0.2) < P.magnet_min_gradient(217.75, 770.0, 0.36, 7874.0, 0.2, 20.0, drag_ratio=1.0)
    assert _need(FE, 0.5) < _need(FE, 0.2) < _need(FE, 0.05)


# ---- form masks (test 32) ----------------------------------------------------------------------------------
def test_i_every_separation_rule_returns_zero_for_a_bound_entry():
    """[I] property test: random entries through every SEPARATION selector; BOUND gets 0."""
    rng = np.random.default_rng(11)
    n = 400
    form = rng.integers(0, 4, n)
    d = 10.0 ** rng.uniform(-6, -2, n)
    sp = rng.choice([WATER, ETOH, NACL, FE, MAG, S["quartz"]], n)
    outs = {
        "density": P.density_phi(form, d, QUARTZ, RHO_W, MU_W, 5.0, 0.05)[0],
        "size": P.sieve_phi(form, d, 3e-4, 0.6, 20.0, 3.0),
        "size_w": P.winnow_phi(form, 12.0, 4.0, 1.0, 0.2),
        "wash": P.wash_phi(form, rng.integers(0, 2, n).astype(bool), 0.5, 0.5),
        "magnet": P.magnet_phi(sp, form, 0.3, 20.0, 20.0),
        "boil": P.boil_off_phi(np.zeros(n, np.int64), sp, form, rng.uniform(0.1, 5, n), [90.0],
                               [101325.0], [30.0]),
    }
    for name, phi in outs.items():
        assert (phi[form == B] == 0.0).all(), name
        assert ((phi >= 0) & (phi <= 1)).all(), name
    for rule in P.MASKS:
        if P.rule_kind(rule) == P.SEPARATION:
            assert not P.form_mask(rule, np.array([B]), soluble=np.array([True]))[0]


def test_i_the_kinds_and_masks_are_the_spec_table():
    """[I] spec 4.6 mask table, row by row."""
    want = {"boil_off": (1, 0, 0, 0), "density": (0, 1, 1, 0), "size": (0, 1, 1, 0),
            "magnet": (0, 1, 1, 0), "diffusion": (0, 0, 0, 1), "melt": (0, 1, 1, 1)}
    forms = np.arange(4)
    for rule, row in want.items():
        assert list(P.form_mask(rule, forms)) == [bool(x) for x in row], rule
    assert list(P.form_mask("wash", forms, soluble=np.ones(4, bool))) == [True, True, True, False]
    assert list(P.form_mask("wash", forms)) == [True, True, False, False]
    assert P.rule_kind("diffusion") == P.DIFFUSION and P.rule_kind("melt") == P.PHASE_CHANGE
    assert P.rule_kind("magnet") == P.SEPARATION


def test_i_bound_tin_leaves_a_solid_only_through_the_lever_rule_fraction():
    """[I] phase change: BOUND moves by `lever` and by nothing else (no lever, no melt, even
    above Tm); a CRYSTAL melts when Tm < T, as heat allows; DISSOLVED does not melt."""
    forms = np.array([B, B, C, C, D])
    phi = P.phase_change_phi(forms, 231.9, 300.0, 10.0, 59.0, lever=np.array([0.0, 0.07, 0, 0, 0]))
    assert list(phi) == [0.0, 0.07, 1.0, 1.0, 0.0]
    cold = P.phase_change_phi(np.array([C]), 231.9, 200.0, 10.0, 59.0)
    assert cold[0] == 0.0
    # limited by heat: 295 J melts half of 10 g with Lf 59 J/g
    half = P.phase_change_phi(np.array([G]), 231.9, 300.0, 10.0, 59.0, heat_J=295.0)
    assert half[0] == pytest.approx(0.5)
    no_lever = P.phase_change_phi(np.array([B]), 231.9, 900.0, 10.0, 59.0)
    assert no_lever[0] == 0.0


def test_i_lever_rule_fraction():
    """[I] x_bulk 0.2 between solid 0.16 and liquid 0.55: (0.2 - 0.16)/(0.55 - 0.16)."""
    assert P.lever_fraction(0.2, 0.16, 0.55) == pytest.approx(0.04 / 0.39)
    assert P.lever_fraction(0.1, 0.16, 0.55) == 0.0 and P.lever_fraction(0.9, 0.16, 0.55) == 1.0


def test_u_solid_state_diffusion_moves_bound_only_with_its_own_arrhenius_rate():
    """[U] carbon in iron: about 1e-11 m2/s at 900 C (u). A pair with no row does not move;
    a free form is not touched by the diffusion rule; the share rises with T and dt."""
    CARBON = S["graphite"]
    assert P.arrhenius_D(*P.DIFFUSION_DATA[(CARBON, FE)], 900.0) == pytest.approx(1e-11, rel=0.1)
    assert P.arrhenius_D(1.0, 1000.0, 20.0) == pytest.approx(np.exp(-1000.0 / (8.314462618 * 293.15)))
    phi = P.diffusion_phi_pair(CARBON, FE, B, 900.0, 3600.0, 0.05)
    assert 0 < phi < 1e-3
    assert P.diffusion_phi_pair(CARBON, FE, B, 1000.0, 3600.0, 0.05) > phi
    assert P.diffusion_phi_pair(S["tin"], FE, B, 900.0, 3600.0, 0.05) == 0.0
    for f in (D, C, G):
        assert P.diffusion_phi_pair(CARBON, FE, f, 900.0, 3600.0, 0.05) == 0.0
    P.register_diffusion(S["tin"], S["copper"], 1e-4, 100e3)
    try:
        assert P.diffusion_phi_pair(S["tin"], S["copper"], B, 900.0, 3600.0, 0.05) > 0
    finally:
        P.DIFFUSION_DATA.pop((S["tin"], S["copper"]), None)


def test_codeonly_partition_and_mixrules_import_no_world_and_no_mix_table():
    """[C] the selectors and rules are pure functions on arrays: importing them loads neither
    the mix table nor any lattice module."""
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = ("import sys; sys.path.insert(0, %r); import src.voxel.partition, src.voxel.mixrules; "
            "bad = [m for m in sys.modules if m.startswith('src.voxel.') and m.split('.')[-1] "
            "in ('mixtable', 'sim', 'fields', 'heat', 'fluids', 'bodies', 'support')]; "
            "print('BAD', bad) if bad else print('CLEAN')" % root)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=root, timeout=120)
    assert out.stdout.strip().endswith("CLEAN"), out.stdout + out.stderr
