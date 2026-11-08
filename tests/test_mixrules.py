"""Wave 13, T1: the property rules (src/voxel/mixrules.py).

Tags: [E] echo of the code's own constants, [C] code-only (test_codeonly_*), [U]
independent but the number's source is unverified (test_u_*), [I] independent
exact / closed-form / arithmetic. Only [I] lines gate a scene. Where a hand-book
number is (u) the test says so and reads tests/data/sourced_numbers.json when it
holds a value (the file is the A-S lane's; a test that finds it disagrees with the
species data is marked xfail(strict) so that fixing the data makes it fail loudly
and the mark gets removed).
"""
import json
import os

import numpy as np
import pytest

from src.voxel import materials as mat
from src.voxel import mixrules as MR
from src.voxel import species as SP
from src.voxel.species import pack_key

HERE = os.path.dirname(os.path.abspath(__file__))
D, C, G, B = SP.DISSOLVED, SP.CRYSTAL, SP.GRAIN, SP.BOUND
S = SP.BY_NAME
CU, SN, PB, FE = S["copper"], S["tin"], S["lead"], S["iron"]
WATER, NACL, ETOH, SUC = S["water"], S["sodium_chloride"], S["ethanol"], S["sucrose"]


def ent(*comps):
    """Entries of ONE row from (species, form, fraction) or (species, form, size, fraction)."""
    rows = []
    for c in comps:
        sp, fo, *rest = c
        sz, f = (rest[0], rest[1]) if len(rest) == 2 else (0, rest[0])
        rows.append((pack_key(S[sp] if isinstance(sp, str) else sp, fo, sz), f))
    rows.sort()
    keys = np.array([r[0] for r in rows], np.uint32)
    return MR.make_entries(1, np.zeros(len(rows), np.int64), keys, np.array([r[1] for r in rows]))


def many(rows):
    """Entries of several rows (each a list of (species, form, fraction))."""
    cells, keys, fr = [], [], []
    for i, comps in enumerate(rows):
        for sp, fo, f in sorted(comps, key=lambda c: pack_key(c[0], c[1], 0)):
            cells.append(i)
            keys.append(pack_key(sp, fo, 0))
            fr.append(f)
    return MR.make_entries(len(rows), cells, np.array(keys, np.uint32), fr)


BRONZE = ((CU, B, 0.88), (SN, B, 0.12))
BRINE = ((WATER, D, 0.8), (NACL, D, 0.2))


# ---- bronze 88/12 (spec 4.1, test 20) -------------------------------------------
@pytest.mark.parametrize("slot", ["ref", "engine"])
def test_u_bronze_density_is_in_the_handbook_range_or_just_under(slot):
    """[U] tin bronze 8.74-8.89 g/cm3 (u). The rule is ideal addition of volumes, so
    it may sit up to 2.5 percent under the range, never above it."""
    rho = MR.rho_mix(ent(("copper", B, .88), ("tin", B, .12)), slot)[0]
    assert 0.975 * 8740 <= rho <= 8890, rho


def test_i_bronze_density_is_the_ideal_volume_sum_of_the_ref_numbers():
    """[I] 1/rho = 0.88/8960 + 0.12/7310 -> 8724 kg/m3."""
    rho = MR.rho_mix(ent(("copper", B, .88), ("tin", B, .12)), "ref")[0]
    assert rho == pytest.approx(1.0 / (0.88 / 8960 + 0.12 / 7310), rel=1e-12)
    assert rho == pytest.approx(8723.7, abs=0.1)


def test_i_bronze_specific_heat_is_366_joules_per_kilogram_kelvin():
    """[I] c = 0.88*385 + 0.12*228 = 366.16 J/kg/K (spec: 366 +- 1)."""
    c = MR.c_mix(ent(("copper", B, .88), ("tin", B, .12)), "ref")[0] * 1000.0
    assert c == pytest.approx(366.0, abs=1.0)


def _maxwell_eucken(k_matrix, k_incl, phi_incl):
    """The v1 rule (it depends on which phase is the matrix)."""
    return k_matrix * (2 * k_matrix + k_incl + 2 * phi_incl * (k_incl - k_matrix)) / (
        2 * k_matrix + k_incl - phi_incl * (k_incl - k_matrix))


def test_u_bronze_conductivity_is_in_40_80_and_the_old_rules_are_not():
    """[U] k of 12 percent tin bronze: the Nordheim constant is (u) so the bound is
    wide. The plain mass mean (360) and Maxwell-Eucken (336) are far outside it."""
    k = MR.k_mix(ent(("copper", B, .88), ("tin", B, .12)), "ref")[0]
    assert 40 <= k <= 80, k
    mean = 0.88 * 401.0 + 0.12 * 66.8
    phi_sn = (0.12 / 7310) / (0.12 / 7310 + 0.88 / 8960)
    me = _maxwell_eucken(401.0, 66.8, phi_sn)
    assert mean == pytest.approx(360.9, abs=0.1) and me == pytest.approx(336, abs=3)
    assert not (40 <= mean <= 80) and not (40 <= me <= 80)


def test_i_bronze_conductivity_follows_the_electronic_rule_arithmetic():
    """[I] the Nordheim + Wiedemann-Franz arithmetic, done by hand: atoms, rho_mix,
    k_e = L T / rho_mix, plus the mass-averaged lattice share (spec 3.6: 47.6 + 0.55)."""
    x_sn = (0.12 / 118.71) / (0.12 / 118.71 + 0.88 / 63.546)
    rho_mix = (1 - x_sn) * 1.68e-8 + x_sn * 1.15e-7 + MR.NORDHEIM[(9, 17)] * x_sn * (1 - x_sn)
    L, T = 2.44e-8, 293.15
    k_e = L * T / rho_mix
    k_lat_sn = max(0.0, 66.8 - L * T / 1.15e-7)
    k = MR.k_mix(ent(("copper", B, .88), ("tin", B, .12)), "ref")[0]
    assert k == pytest.approx(k_e + 0.12 * k_lat_sn, rel=1e-9)
    assert k_e == pytest.approx(47.6, abs=0.1)


def test_u_bronze_bar_warms_about_7_to_8_times_slower_than_copper():
    """[U] P1's prediction, computed from the rule outputs (spec 4.1: about 7.7)."""
    e = ent(("copper", B, .88), ("tin", B, .12))
    k, rho, c = (f(e, "ref")[0] for f in (MR.k_mix, MR.rho_mix, MR.c_mix))
    cu = ent(("copper", B, 1.0))
    kc, rc, cc = (f(cu, "ref")[0] for f in (MR.k_mix, MR.rho_mix, MR.c_mix))
    ratio = (kc / (rc * cc)) / (k / (rho * c))
    assert 7.0 <= ratio <= 8.5, ratio


def test_i_a_pure_metal_keeps_its_own_conductivity_from_the_table():
    """[I] one electronic entry: k_e + lattice share is k (the lattice share is the
    table k minus L T / rho_e, so the sum is the table number whenever that is >= 0)."""
    for sp, name in ((SN, "tin"), (CU, "copper")):
        k = MR.k_mix(ent((name, B, 1.0)), "ref")[0]
        assert k == pytest.approx(max(SP.get(name).ref["k_s"],
                                      2.44e-8 * 293.15 / SP.get(name).ref["rho_e"]), rel=1e-9)


def test_i_nordheim_pair_term_is_symmetric_and_registered_by_data():
    """[I] register_nordheim stores the pair sorted; a pair with no data adds 0."""
    assert MR.nordheim_C(SN, CU) == MR.nordheim_C(CU, SN) == MR.NORDHEIM[(9, 17)]
    assert MR.nordheim_C(PB, FE) == 0.0
    old = dict(MR.NORDHEIM)
    try:
        k0 = MR.k_mix(ent(("lead", B, .5), ("tin", B, .5)), "ref")[0]
        MR.register_nordheim(SN, PB, 1.0e-6)
        assert MR.nordheim_C(PB, SN) == 1.0e-6
        assert MR.k_mix(ent(("lead", B, .5), ("tin", B, .5)), "ref")[0] < k0
    finally:
        MR.NORDHEIM.clear()
        MR.NORDHEIM.update(old)
        MR._NORD_MAT.clear()


def test_i_three_metal_alloy_sums_the_pair_terms_over_all_pairs():
    """[I] the Nordheim pair sum over i < j with three metals, checked by hand (registered
    coefficients for Pb-Sn and Cu-Pb on top of the shipped Cu-Sn)."""
    old = dict(MR.NORDHEIM)
    try:
        MR.register_nordheim(PB, SN, 1.5e-6)
        MR.register_nordheim(CU, PB, 0.5e-6)
        w = {CU: 0.6, SN: 0.25, PB: 0.15}
        a = MR.sp_arrays("ref")
        at = {k: v / a["M"][k] for k, v in w.items()}
        x = {k: v / sum(at.values()) for k, v in at.items()}
        rho = sum(x[k] * a["rho_e"][k] for k in x)
        for i in x:
            for j in x:
                if i < j:
                    rho += MR.nordheim_C(i, j) * x[i] * x[j]
        k_e = 2.44e-8 * 293.15 / rho
        lat = sum(w[k] * max(0.0, a["k_S"][k] - 2.44e-8 * 293.15 / a["rho_e"][k]) for k in w)
        got = MR.k_mix(ent(("copper", B, .6), ("tin", B, .25), ("lead", B, .15)), "ref")[0]
        assert got == pytest.approx(k_e + lat, rel=1e-12)
    finally:
        MR.NORDHEIM.clear()
        MR.NORDHEIM.update(old)
        MR._NORD_MAT.clear()


# ---- Bruggeman (test 21) -------------------------------------------------------------
def _brug(k, vol):
    k, vol = np.asarray(k, float), np.asarray(vol, float)
    return MR.bruggeman(k, vol, np.zeros(len(k), np.int64), 1)[0]


def test_i_bruggeman_limits_and_symmetry():
    """[I] phi -> 0 gives k1, phi -> 1 gives k2; the order of the phases is irrelevant."""
    assert _brug([401, 66.8], [1.0, 1e-9]) == pytest.approx(401, rel=1e-6)
    assert _brug([401, 66.8], [1e-9, 1.0]) == pytest.approx(66.8, rel=1e-6)
    assert _brug([401, 66.8], [0.5, 0.5]) == _brug([66.8, 401], [0.5, 0.5])


def test_i_bruggeman_50_50_copper_tin_is_188():
    """[I] spec 3.6: 188 either way (Maxwell-Eucken gives 191 or 145 by choice of matrix)."""
    assert _brug([401, 66.8], [0.5, 0.5]) == pytest.approx(188, abs=1.0)
    assert _maxwell_eucken(401, 66.8, 0.5) != pytest.approx(_maxwell_eucken(66.8, 401, 0.5), abs=10)


def test_i_bruggeman_root_satisfies_its_own_equation():
    """[I] the defining sum is zero at the returned k."""
    k = np.array([401.0, 66.8, 0.6, 7.7])
    phi = np.array([0.1, 0.2, 0.3, 0.4])
    ke = _brug(k, phi)
    assert np.sum(phi * (k - ke) / (k + 2 * ke)) == pytest.approx(0.0, abs=1e-12)


def test_i_bruggeman_stays_between_the_wiener_bounds():
    """[I] property test: series <= k_eff <= parallel for random mixes of 2-5 phases."""
    rng = np.random.default_rng(5)
    for _ in range(300):
        m = rng.integers(2, 6)
        k = 10.0 ** rng.uniform(-1, 2.6, m)
        vol = rng.uniform(0.01, 1, m)
        cell = np.zeros(m, np.int64)
        ke = MR.bruggeman(k, vol, cell, 1)[0]
        lo = MR.wiener_series(k, vol, cell, 1)[0]
        hi = MR.wiener_parallel(k, vol, cell, 1)[0]
        assert lo - 1e-9 <= ke <= hi + 1e-9


def test_i_a_mixed_solid_of_two_non_metals_uses_bruggeman_by_volume():
    """[I] quartz rock 50 percent + calcite 50 percent by mass: the k_mix equals a direct
    Bruggeman call on the volume fractions (neither has rho_e, so there is no group)."""
    e = ent(("quartz", B, 0.5), ("calcite", B, 0.5))
    a = MR.sp_arrays("ref")
    kq, kc = a["k_S"][S["quartz"]], a["k_S"][S["calcite"]]
    vq, vc = 0.5 / a["rho_S"][S["quartz"]], 0.5 / a["rho_S"][S["calcite"]]
    assert MR.k_mix(e, "ref")[0] == pytest.approx(_brug([kq, kc], [vq, vc]), rel=1e-9)


def test_i_a_sand_bed_joins_a_solid_as_a_bed_not_as_solid_quartz():
    """[I] quartz SAND (GRAIN, a packed bed) 50 percent + calcite 50 percent: the bed is one
    phase of conductivity packed_k(k_quartz, air, por) and volume = particle volume /
    (1 - por); then Bruggeman with the calcite. (It was the solid-quartz 7.7 over the
    particle volume, so the bed conducted like a rock.)"""
    e = ent(("quartz", G, 64, 0.5), ("calcite", B, 0.5))
    a = MR.sp_arrays("ref")
    por = a["por_grain"][S["quartz"]]
    kq, kc = a["k_S"][S["quartz"]], a["k_S"][S["calcite"]]
    kbed = SP.packed_k(kq, MR.K_PORE_GAS, por)
    vq, vc = 0.5 / (a["rho_S"][S["quartz"]] * (1.0 - por)), 0.5 / a["rho_S"][S["calcite"]]
    assert MR.k_mix(e, "ref")[0] == pytest.approx(_brug([kbed, kc], [vq, vc]), rel=1e-9)
    assert MR.k_mix(e, "ref")[0] < MR.k_mix(ent(("quartz", B, 0.5), ("calcite", B, 0.5)), "ref")[0]


# ---- brine 20 wt percent (test 22) ---------------------------------------------------------
def test_u_brine_density_with_masson_is_within_half_a_percent_of_the_handbook():
    """[U] handbook 1147.8 kg/m3 (u); Masson phiV = 16.62 + 1.87 sqrt(m)."""
    rho = MR.rho_mix(ent(("water", D, .8), ("sodium_chloride", D, .2)), "ref")[0]
    assert rho == pytest.approx(1147.8, rel=0.005), rho


def test_i_brine_density_matches_the_masson_arithmetic_by_hand():
    """[I] m = 4.278 mol/kg, phiV = 20.49 mL/mol, V = 80/0.998 + 0.3422 phiV mL per 100 g."""
    m = (20 / 58.44) / 0.08
    phiV = 16.62 + 1.87 * np.sqrt(m)
    V = 80 / 0.998 + (20 / 58.44) * phiV
    rho = MR.rho_mix(ent(("water", D, .8), ("sodium_chloride", D, .2)), "ref")[0]
    assert rho == pytest.approx(100 / V * 1000, rel=1e-12)


def test_u_brine_ideal_addition_is_2_5_percent_low_which_is_why_the_rule_exists():
    """[U] against the handbook 1147.8 (u) the plain volume sum is about 2.5 percent low."""
    e = ent(("water", D, .8), ("sodium_chloride", D, .2))
    plain = MR.rho_mix(e, "ref", masson=False)[0]
    assert plain == pytest.approx(1147.8 * 0.975, rel=0.002)


def test_i_masson_applies_only_when_the_largest_dissolved_species_is_water():
    """[I] NaCl in ethanol (the larger solvent) gets the ideal sum, and so does a
    solute without phiV data in water."""
    e1 = ent(("ethanol", D, .8), ("sodium_chloride", D, .2))
    assert MR.rho_mix(e1, "ref")[0] == MR.rho_mix(e1, "ref", masson=False)[0]
    e2 = ent(("water", D, .9), ("ethanol", D, .1))
    assert MR.rho_mix(e2, "ref")[0] == MR.rho_mix(e2, "ref", masson=False)[0]


def test_i_syrup_density_uses_sucrose_apparent_volume():
    """[I] 20 wt percent sucrose: V = 80/0.998 + (20/342.3) 211.5 mL per 100 g -> 1081."""
    rho = MR.rho_mix(ent(("water", D, .8), ("sucrose", D, .2)), "ref")[0]
    assert rho == pytest.approx(100 / (80 / 0.998 + 20 / 342.3 * 211.5) * 1000, rel=1e-12)
    assert rho == pytest.approx(1081, abs=3)


def test_u_brine_specific_heat_is_3_4_and_the_plain_mean_fails():
    """[U] handbook about 3.4 J/g/K (u): c_app 0.28 for dissolved NaCl gives 3.400;
    the plain mean 0.8*4.18 + 0.2*0.864 = 3.517 is 3.4 percent high."""
    c = MR.c_mix(ent(("water", D, .8), ("sodium_chloride", D, .2)), "ref")[0]
    assert c == pytest.approx(3.4, rel=0.01)
    plain = 0.8 * 4.18 + 0.2 * 0.864
    assert plain == pytest.approx(3.517, abs=1e-3) and abs(plain / 3.4 - 1) > 0.01


def test_u_brine_conductivity_is_0_57():
    """[U] k_app 0.45 for dissolved NaCl (u): 0.8*0.60 + 0.2*0.45 = 0.57 +- 5 percent."""
    assert MR.k_mix(ent(("water", D, .8), ("sodium_chloride", D, .2)), "ref")[0] == \
        pytest.approx(0.57, rel=0.05)


def test_u_brine_viscosity_is_1_565_mpa_s():
    """[U] 20 wt percent NaCl at 20 C: relative 1.562 (Kestin, Khalifa, Correia 1981, the
    sourced value; lane P-T1 replaced the spec's (u) 1.99) times water's 1.002 = 1.565 mPa s.
    Polish consolidation: this test used to expect the old (u) 1.99 and failed once the seed
    table was fixed."""
    mu = MR.mu_mix(ent(("water", D, .8), ("sodium_chloride", D, .2)), "ref", T_C=20.0)[0]
    assert mu == pytest.approx(1.002 * 1.562, rel=0.02), mu


def _sourced(key):
    path = os.path.join(HERE, "data", "sourced_numbers.json")
    if not os.path.exists(path):
        pytest.skip("tests/data/sourced_numbers.json absent")
    return json.load(open(path))["entries"][key]


def test_sourced_brine_relative_viscosity_at_20_percent():
    """[I] against the SOURCED relative viscosity (Kestin, Khalifa, Correia 1981)."""
    want = _sourced("nacl.rel_viscosity_20C")["value"]["20"]["relative"]
    mu = MR.mu_mix(ent(("water", D, .8), ("sodium_chloride", D, .2)), "ref", T_C=20.0)[0]
    assert mu / MR.andrade(WATER, 20.0, "ref") == pytest.approx(want, rel=0.10)


def test_i_andrade_water_reproduces_its_two_fit_points():
    """[I] 1.002 mPa s at 20 C and 0.547 at 50 C (the two points the fit is made from)."""
    assert MR.andrade(WATER, 20.0, "ref") == pytest.approx(1.002, rel=5e-3)
    assert MR.andrade(WATER, 50.0, "ref") == pytest.approx(0.547, rel=5e-3)


def test_i_a_pure_liquid_viscosity_is_its_andrade_law_and_falls_with_heat():
    """[I] mu_mix of pure water is Andrade; raising T lowers it."""
    e = ent(("water", D, 1.0))
    assert MR.mu_mix(e, "ref", T_C=20.0)[0] == pytest.approx(MR.andrade(WATER, 20.0, "ref"))
    assert MR.mu_mix(e, "ref", T_C=60.0)[0] < MR.mu_mix(e, "ref", T_C=20.0)[0]


def test_i_kendall_monroe_for_two_miscible_liquids_with_andrade_data():
    """[I] ethanol + water: mu^(1/3) is the mole-fraction mean of the pure cube roots."""
    w = 0.3
    e = ent(("ethanol", D, w), ("water", D, 1 - w))
    n_e, n_w = w / 46.07, (1 - w) / 18.015
    xe = n_e / (n_e + n_w)
    m_e, m_w = MR.andrade(ETOH, 20.0, "ref"), MR.andrade(WATER, 20.0, "ref")
    want = (xe * m_e ** (1 / 3) + (1 - xe) * m_w ** (1 / 3)) ** 3
    assert MR.mu_mix(e, "ref", 20.0)[0] == pytest.approx(want, rel=1e-12)


def test_i_a_liquid_with_no_viscosity_data_gets_the_named_default_and_a_solid_row_nan():
    """[I] molten bronze (no Andrade rows) gets MU_DEFAULT; a row with nothing dissolved: NaN."""
    mu = MR.mu_mix(ent(("copper", D, .88), ("tin", D, .12)), "ref")[0]
    assert mu == MR.MU_DEFAULT
    assert np.isnan(MR.mu_mix(ent(("copper", B, 1.0)), "ref")[0])


def test_i_log_linear_relative_viscosity_table_is_exact_at_its_nodes():
    """[I] at each node of the NaCl table the factor is the table value."""
    tab = SP.get("sodium_chloride").ref["mu_rel"]
    base = MR.andrade(WATER, 20.0, "ref")
    for w, rel in tab:
        mu = MR.mu_mix(ent(("water", D, 1 - w), ("sodium_chloride", D, w)) if w else
                       ent(("water", D, 1.0)), "ref", 20.0)[0]
        assert mu == pytest.approx(base * rel, rel=1e-9)


# ---- the acid family (test 23) ------------------------------------------------------------------
@pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 1.0])
def test_i_acid_weight_fraction_matches_the_materials_formula(p):
    """[I] w = STOCK rho p / (rho p + 0.998 (1 - p)) with the repo's rho and STOCK."""
    w, rho = mat.STOCK[mat.ACID], mat.FLUID[mat.ACID][0]
    want = w * rho * p / (rho * p + 0.998 * (1 - p))
    assert MR.acid_weight_fraction(mat.ACID, p) == pytest.approx(want, rel=1e-14)


def test_i_acid_map_gives_the_spec_numbers_16_63_98():
    """[I] p 0.1 -> 16.4, 0.5 -> 63.2, 1 -> 98.3 wt percent (spec 4.1, computed)."""
    got = [100 * float(MR.acid_weight_fraction(mat.ACID, p)) for p in (0.1, 0.5, 1.0)]
    assert got == pytest.approx([16.4, 63.2, 98.3], abs=0.05)


def test_i_acid_potency_round_trips_to_1e_9():
    """[I] potency -> weight fraction -> potency, over a grid, for both acids."""
    p = np.linspace(0, 1, 101)
    for f in (mat.ACID, mat.WEAK_ACID):
        assert np.abs(MR.acid_potency(f, MR.acid_weight_fraction(f, p)) - p).max() < 1e-9


def test_i_vinegar_books_5_percent_acetic_by_mass_not_100():
    """[I] a litre of vinegar at potency 1 is 5 percent acetic acid by weight."""
    (k_acid, k_water), (w, rest) = MR.acid_mix(mat.WEAK_ACID, 1.0)
    assert w == pytest.approx(0.05) and rest == pytest.approx(0.95)
    assert k_acid >> 10 == S["acetic_acid"] and k_water >> 10 == WATER


# ---- combustion, ignition, melting stand-in, emissivity, porosity -------------------------------------
def test_i_heat_of_combustion_is_the_mass_weighted_average_exactly():
    """[I] test 24: a 50/50 mix of ethanol and graphite burns the average."""
    h = MR.hcomb_mix(ent(("ethanol", D, .5), ("graphite", B, .5)), "ref")[0]
    assert h == pytest.approx(0.5 * 29700.0 + 0.5 * 32800.0, rel=1e-14)
    o2 = MR.o2_mix(ent(("ethanol", D, .5), ("graphite", B, .5)), "ref")[0]
    assert o2 == pytest.approx(0.5 * 2.0837 + 0.5 * 2.6641, rel=1e-14)


def test_i_heat_of_combustion_is_linear_in_the_fraction():
    """[I] property test: h(a x + (1-a) y) = a h(x) + (1-a) h(y) for rows over the same keys."""
    rng = np.random.default_rng(3)
    for _ in range(50):
        a = rng.uniform(0, 1)
        f1, f2 = rng.dirichlet(np.ones(3)), rng.dirichlet(np.ones(3))
        names = ("ethanol", "graphite", "tallow")
        mk = lambda f: ent(*[(n, D if n == "ethanol" else B, float(x)) for n, x in zip(names, f)])
        hm = MR.hcomb_mix(mk(a * f1 + (1 - a) * f2), "ref")[0]
        assert hm == pytest.approx(a * MR.hcomb_mix(mk(f1), "ref")[0]
                                   + (1 - a) * MR.hcomb_mix(mk(f2), "ref")[0], rel=1e-12)


def test_i_ignition_is_the_lowest_over_fuels_above_the_mass_floor():
    """[I] wood (ign 300) with 4 percent of a fuel with a lower ign: the floor ignores it."""
    wood = SP.get("wood")
    assert wood.get("ign") == 300.0
    e = ent(("wood", B, 0.96), ("graphite", B, 0.04))
    assert MR.ign_mix(e, "engine")[0] == 300.0
    assert np.isnan(MR.ign_mix(ent(("quartz", B, 1.0)), "engine")[0])


def test_i_interim_melting_point_is_the_mass_mean_of_the_parts_that_have_one():
    """[I] tin 232 / lead 327: 60/40 -> 0.6*231.9 + 0.4*327.5."""
    t = MR.tmelt_mix(ent(("tin", B, .6), ("lead", B, .4)), "ref")[0]
    assert t == pytest.approx(0.6 * 231.9 + 0.4 * 327.5, rel=1e-12)


def test_i_emissivity_is_that_of_the_dominant_component():
    """[I] 70 percent copper (0.3) and 30 percent quartz (0.9): copper decides."""
    e = MR.eps_mix(ent(("copper", B, .7), ("quartz", B, .3)), "ref")[0]
    assert e == SP.get("copper").ref["emiss_s"]


def test_i_porosity_of_a_composite_is_one_minus_skeleton_over_bulk():
    """[I] humus: rho_skel 1400, bulk 700 -> porosity 0.5; a dense mineral has 0."""
    assert MR.por_mix(ent(("humus", G, 48, 1.0)), "ref")[0] == pytest.approx(0.5)
    assert MR.por_mix(ent(("quartz", B, 1.0)), "ref")[0] == 0.0


def test_i_liquid_conductivity_of_molten_bronze_is_the_mass_weighted_sum():
    """[I] all DISSOLVED: k = sum w k_L (copper k_l 166, tin falls back to its solid 66.8)."""
    k = MR.k_mix(ent(("copper", D, .88), ("tin", D, .12)), "ref")[0]
    assert k == pytest.approx(0.88 * 166.0 + 0.12 * 66.8, rel=1e-12)


# ---- the registry -------------------------------------------------------------------------------------
def test_i_register_rule_and_derived_are_reachable_through_evaluate():
    """[I] a rule or a derived column added by data is run by evaluate()."""
    e = ent(("copper", B, 1.0))
    MR.register_rule("test_double_rho", lambda en, slot="engine", **kw: 2 * MR.rho_mix(en, slot))
    MR.register_derived("test_mass_sum", lambda en: en.sum(en.frac), deps=("rho",))
    try:
        assert MR.evaluate("test_double_rho", e, "ref")[0] == pytest.approx(2 * 8960.0)
        assert MR.evaluate("test_mass_sum", e)[0] == pytest.approx(1.0)
        with pytest.raises(KeyError):
            MR.evaluate("no_such_rule", e)
    finally:
        MR.RULES.pop("test_double_rho", None)
        MR.DERIVED.pop("test_mass_sum", None)
        MR._REV[0] += 1


def test_i_form_energy_conventions():
    """[I] NaCl dissolving absorbs +66 J/g at the 20 C reference; the reverse is the negative;
    a registered value wins; same form is 0; melting is moved to 20 C by Kirchhoff."""
    assert MR.form_energy(NACL, C, D) == pytest.approx(66.0)
    assert MR.form_energy(NACL, D, C) == pytest.approx(-66.0)
    assert MR.form_energy(NACL, G, B) == 0.0 and MR.form_energy(NACL, B, B) == 0.0
    lf, tm = 59.0, 231.9
    cl, cs = 0.24, 0.228                      # ref c_l falls back to c_s for tin
    got = MR.form_energy(SN, B, D, "ref")
    assert got == pytest.approx(lf - (SP.get("tin").ref.get("c_l", 0.228) - cs) * (tm - 20.0))
    MR.register_form_energy(CU, B, D, 123.0)
    try:
        assert MR.form_energy(CU, B, D) == 123.0 and MR.form_energy(CU, D, B) == -123.0
    finally:
        MR._FORM_ENERGY.pop((CU, B, D), None)
        MR._FORM_ENERGY.pop((CU, D, B), None)
    assert cl > 0


def test_codeonly_engine_slot_reads_the_repo_numbers_for_legacy_species():
    """[C] engine water is 1000 g/L and 4.2 J/g/K (materials.py), ref is 998 and 4.18."""
    e = ent(("water", D, 1.0))
    assert MR.rho_mix(e, "engine")[0] == pytest.approx(1000.0, rel=1e-12)
    assert MR.rho_mix(e, "ref")[0] == pytest.approx(998.0, rel=1e-12)
    assert MR.c_mix(e, "engine")[0] == pytest.approx(4.2) and MR.c_mix(e, "ref")[0] == pytest.approx(4.18)


def test_i_every_engine_species_with_a_pure_row_has_finite_cached_numbers():
    """[I] rho, c, k, eps are finite for the pure form of every species the table can hold
    in its pure rows (a NaN would poison a heat law)."""
    from src.voxel.mixtable import MixTable
    t = MixTable()
    ids = [r for r in range(128) if r not in (0, 64) and t.row_len([r])[0] > 0]
    for name in ("rho", "c", "k", "eps"):
        v = t.prop(name, ids)
        assert np.isfinite(v).all(), (name, [i for i, x in zip(ids, v) if not np.isfinite(x)])


# ---- a packed bed in a mix reads its bulk density, porosity and conductivity ---------------
PACKED = ("quartz", "slaked_lime")          # the species with a por_grain number


@pytest.mark.parametrize("slot", ["engine", "ref"])
@pytest.mark.parametrize("name", PACKED)
def test_i_a_pure_bed_cell_reads_the_bulk_density_porosity_and_k_of_the_species(name, slot):
    """[I] a mix cell of one species in the packed GRAIN form is the same cell the cell
    table holds: rho = species.bulk_density, por = form_porosity, k = bulk_conductivity."""
    e = ent((name, G, 1.0))
    assert MR.rho_mix(e, slot)[0] == pytest.approx(SP.bulk_density(name, G, slot), rel=1e-12)
    assert MR.por_mix(e, slot)[0] == pytest.approx(SP.form_porosity(name, G, slot), rel=1e-12)
    assert MR.k_mix(e, slot)[0] == pytest.approx(SP.bulk_conductivity(name, G, slot), rel=1e-12)
    rock = ent((name, B, 1.0))                       # the same species as rock has no pores
    assert MR.rho_mix(rock, slot)[0] == pytest.approx(SP.get(name).get("rho_s", slot), rel=1e-12)
    assert MR.por_mix(rock, slot)[0] == 0.0


def test_i_the_mix_cell_of_sand_equals_the_cell_table_row():
    """[I] the cell table (seed_mat_rows) and the mix rules agree on mat 15 and mat 25:
    density, and conductivity (both were two numbers apart: 2650 against 1590)."""
    rows = {r["mat"]: r for r in SP.seed_mat_rows()}
    for m in (15, 25):
        slot = SP.MATS[m]
        e = ent((slot.species, slot.form, 1.0))
        assert MR.rho_mix(e)[0] == pytest.approx(rows[m]["rho"], rel=1e-12)
        assert MR.k_mix(e)[0] == pytest.approx(rows[m]["k"], rel=1e-12)
    assert rows[15]["rho"] == pytest.approx(1590.0) and rows[14]["rho"] == 2650.0


@pytest.mark.parametrize("seed", range(20))
def test_i_random_bed_mixes_add_the_bed_volumes_and_keep_the_mass_exact(seed):
    """[I] random sand / lime-powder / rock / water mixes: the volume is the sum of each
    part's volume (beds over their bulk density, rock over its density, water over its
    liquid density), so rho = 1 / sum(w / rho_part); the porosity is the volume share of
    the pores; and 1 / rho times the mass gives the same volume however the mass is split."""
    rng = np.random.default_rng(1300 + seed)
    names = ["quartz", "slaked_lime", "calcite"]
    forms = {"quartz": G, "slaked_lime": G, "calcite": B}
    w = rng.dirichlet(np.ones(len(names) + 1))
    e = ent(*[(n, forms[n], float(w[i])) for i, n in enumerate(names)], ("water", D, float(w[-1])))
    rho = MR.rho_mix(e, "ref", masson=False)[0]
    parts = [SP.bulk_density(n, forms[n], "ref") for n in names] + [SP.get("water").get("rho_l", "ref")]
    assert rho == pytest.approx(1.0 / sum(wi / ri for wi, ri in zip(w, parts)), rel=1e-12)
    pore_v = sum(w[i] / SP.bulk_density(n, G, "ref") - w[i] / SP.get(n).get("rho_s", "ref")
                 for i, n in enumerate(names) if forms[n] == G)
    solid_v = sum(w[i] / SP.bulk_density(n, forms[n], "ref") for i, n in enumerate(names))
    assert MR.por_mix(e, "ref")[0] == pytest.approx(pore_v / solid_v, rel=1e-9)


def test_i_a_bed_without_a_porosity_number_is_unchanged():
    """[I] humus (GRAIN, bulk rho_s and a skeleton of its own, no por_grain) reads as
    before: rho = rho_s, and a species with no por_grain never gets a bed."""
    e = ent(("humus", G, 1.0))
    assert MR.rho_mix(e, "ref")[0] == pytest.approx(SP.get("humus").get("rho_s", "ref"), rel=1e-12)
    assert MR.por_mix(e, "ref")[0] == pytest.approx(0.5, rel=1e-12)
    assert MR.k_mix(e, "ref")[0] == pytest.approx(SP.get("humus").get("k_s", "ref"), rel=1e-12)
