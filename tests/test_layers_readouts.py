"""T3 layers: the read-outs other laws use (heat, fire, friction, sight, wetness) and
the exchanges with the air (deposition, condensation) (spec 7: T33-T36, P2 soot band,
P6 dew and frost, P9 grease).

Tags: test_codeonly_* tests the code against its own formula; test_u_* independent but
the source is unverified; the rest are identities and conservation.
"""
import numpy as np
import pytest

from src.voxel import layer_data as D
from src.voxel import layer_laws as LL
from src.voxel.layers import Layers, make_key
from tests.layers_fake import FakeHost

A5 = 0.05 * 0.05


def _inv(host, L):
    return host.total_E() + L.stored_J() - L.energy_in_total()


# ================================================================ T33 heat read-outs
def test_codeonly_two_cm_of_ash_is_a_series_resistance_of_0_167():
    L = Layers()
    k = make_key(0, 1, 4)
    g = 0.02 * A5 * 1e6 * 0.15                                       # 2 cm of 150 g/L ash
    L.add(k, {(D.ASH, D.GRAIN, 40): g}, cover=1.0, Ts=20.0)
    s = L.stack(k)[0]
    assert L.thickness(s) == pytest.approx(0.02, rel=0.01)
    R, eps, alpha, C = L.thermal(k)
    assert R == pytest.approx(0.1667, rel=0.01)                      # equal to 2.5 cm of wood
    assert R == pytest.approx(0.02 / 0.12, rel=0.01)
    assert eps == pytest.approx(0.9, abs=1e-6) and alpha == pytest.approx(0.8, abs=1e-6)


def test_codeonly_half_a_millimetre_of_water_holds_2_09_kj_per_m2_k():
    L = Layers()
    k = make_key(0, 1, 4)
    g = 0.5e-3 * A5 * 1e6 * D.ROWS[D.WATER]["rho_l"]
    L.add(k, {(D.WATER, D.DISSOLVED, 0): g}, cover=1.0, Ts=20.0)
    R, eps, alpha, C = L.thermal(k)
    assert C == pytest.approx(2090.0, rel=0.01)
    assert R == pytest.approx(0.5e-3 / 0.6, rel=0.02)                # k 0.6
    assert eps == pytest.approx(0.95, abs=1e-6)


def test_a_layers_heat_capacity_joins_both_cs_and_c_so_cb_is_unchanged():
    host = FakeHost(n=1, C=100.0, Cs=20.0, dtype=np.float64)
    host.set_Ts(0, 25.0)
    L = Layers()
    k = make_key(0, 0, 4)
    Cb0, T0, Ts0 = host.Cb(0), host.T(0), host.Ts(0)
    g = 0.5e-3 * A5 * 1e6 * 0.998
    _s, adj = L.add(k, {(D.WATER, D.DISSOLVED, 0): g}, cover=1.0, Ts=host.Ts(0))   # at the skin T
    host.apply(adj)
    dC = g * 4.18
    assert host.C[0] == pytest.approx(100.0 + dC) and host.Cs[0] == pytest.approx(20.0 + dC)
    assert host.Cb(0) == pytest.approx(Cb0, rel=1e-12)
    assert host.Ts(0) == pytest.approx(Ts0, abs=1e-9)                # same temperature: no jump
    assert host.T(0) >= T0                                           # the mean follows its skin
    # matter that arrives hot warms the skin node, not the body
    _s, adj = L.add(k, {(D.WATER, D.DISSOLVED, 0): g}, cover=1.0, Ts=host.Ts(0), T_in=80.0)
    Cb = host.Cb(0)
    host.apply(adj)
    assert host.Cb(0) == pytest.approx(Cb, rel=1e-12) and host.Ts(0) > Ts0


def test_the_float32_booking_idiom_agrees_with_apply_to_one_ulp():
    h32 = FakeHost(n=2, C=100.0, Cs=20.0, dtype=np.float32)
    h64 = FakeHost(n=2, C=100.0, Cs=20.0, dtype=np.float64)
    for h in (h32, h64):
        h.set_Ts(0, 30.0)
    h32.book(0, 123.456)
    h64.book(0, 123.456)
    assert h32.Ts(0) == pytest.approx(h64.Ts(0), abs=1e-4)
    assert h32.E.dtype == np.float32 and h32.skin.dtype == np.float32
    h32b = FakeHost(n=1, C=100.0, Cs=20.0, dtype=np.float32, has=False)
    h32b.book(0, 50.0)
    assert h32b.T(0) == pytest.approx(20.5, abs=1e-5) and h32b.skin[0] == 0.0


def test_emissivity_and_absorptivity_run_toward_the_soot_value_with_cover():
    sub = D.SUBSTRATE[2]
    last = sub["eps"]
    for m in (1e-5, 5e-5, 2e-4, 1e-3):
        L = Layers()
        k = make_key(0, 1, 5)
        L.add(k, {(D.SOOT, D.GRAIN, 1): m}, Ts=20.0)
        R, eps, alpha, C = L.thermal(k, 2)
        c = L.cover_of(k)
        assert eps == pytest.approx((1 - c) * sub["eps"] + c * 0.95, rel=1e-6)
        assert eps > last or abs(eps - last) < 1e-9
        last = eps
    assert last == pytest.approx(0.95, abs=0.02)
    assert L.thermal(make_key(0, 99, 5))[:3] == (0.0, D.SUBSTRATE[2]["eps"], D.SUBSTRATE[2]["alpha"])


def test_fuel_wet_and_look_read_the_same_layers():
    L = Layers()
    k = make_key(0, 1, 4)
    L.add(k, {(D.LAMP_OIL, D.DISSOLVED, 0): 0.5, (D.WATER, D.DISSOLVED, 0): 0.0 + 1e-9},
          cover=1.0, Ts=20.0)
    g, dHc, T_ign, cap = L.fuel(k)
    assert g == pytest.approx(0.5 / A5, rel=1e-6) and dHc == pytest.approx(43000.0, rel=1e-3)
    assert T_ign == pytest.approx(210.0) and cap > 0.0
    W = Layers()
    kw = make_key(0, 2, 4)
    W.add(kw, {(D.WATER, D.DISSOLVED, 0): 0.3}, cover=0.5, Ts=20.0)
    assert W.fuel(kw) == (0.0, 0.0, None, 0.0)                       # water is not fuel
    liq, share, per = W.wet(kw)
    assert liq == pytest.approx(0.3) and share == pytest.approx(0.5) and per == {D.WATER: 0.3}
    liq2, share2, _ = W.wet(kw, fvol_above=2.0)                      # a lattice film on top
    assert liq2 > liq and share2 > share
    look_dry = Layers().look(kw)["colour"]
    look_wet = W.look(kw)
    assert look_wet["gloss"] == pytest.approx(0.5)
    assert max(look_wet["colour"]) < max(look_dry)                    # wet is darker


# ================================================================ T34 deposition
def test_codeonly_stokes_matches_the_hinds_unit_density_table_within_3_percent():
    assert LL.stokes_velocity(1e-6, 1000.0) == pytest.approx(3.5e-5, rel=0.03)
    assert LL.stokes_velocity(10e-6, 1000.0) == pytest.approx(3.1e-3, rel=0.03)


def test_codeonly_soot_settling_speeds():
    for d, want in ((0.1e-6, 1.6e-6), (0.3e-6, 7.6e-6), (1e-6, 6.3e-5), (10e-6, 5.5e-3)):
        assert LL.stokes_velocity(d, 1800.0) == pytest.approx(want, rel=0.05), d
    assert LL.stokes_velocity(0.5e-6, 1800.0) == pytest.approx(2e-5, rel=0.1)


def test_u_thermophoresis_of_a_500_k_gas_over_a_300_k_wall_is_2_5e_minus_4():
    assert LL.kinematic_viscosity_air(126.85) == pytest.approx(2.5e-5, rel=0.03)
    assert LL.thermophoretic_velocity(226.85, 26.85) == pytest.approx(2.5e-4, rel=0.03)
    assert LL.thermophoretic_velocity(26.85, 226.85) < 0.0           # toward the cold: away from a hot wall


def test_up_faces_add_the_settling_speed_a_ceiling_never_goes_negative():
    d, rho = 5e-6, 1800.0
    up = LL.deposition_velocity(d, rho, +1.0)
    wall = LL.deposition_velocity(d, rho, 0.0)
    ceil = LL.deposition_velocity(d, rho, -1.0)
    assert up == pytest.approx(wall + LL.stokes_velocity(d, rho))
    assert wall == pytest.approx(LL.V_BROWN) and ceil == 0.0         # clamps at 0
    big = LL.deposition_velocity(10e-6, 1800.0, -1.0, T_gas=60.0, T_wall=20.0)
    assert big >= 0.0
    hot = LL.deposition_velocity(0.5e-6, 1800.0, -1.0, T_gas=226.85, T_wall=26.85)
    assert hot > LL.V_BROWN                                          # thermophoresis lifts it to the ceiling


def test_u_a_sooty_room_fills_the_computed_band_not_the_99_percent_of_v1():
    """4.2: 220 kW of wood for 300 s in a 5x5x2.5 m room."""
    soot_made = 0.2 * 300.0                                          # 0.2 g/s true soot: 60 g
    well = LL.sink_fraction(1e-4, 100.0, 62.5, 300.0)
    assert well == pytest.approx(0.047, abs=0.002)
    g = well * 62.0
    assert g == pytest.approx(2.9, abs=0.2)
    m_a = g / 100.0                                                  # g per m2 over 100 m2
    assert LL.cover_particulate(8.0, m_a) == pytest.approx(0.21, abs=0.02)
    hot = LL.sink_fraction(2.8e-4, 45.0, 25.0, 300.0)
    assert hot == pytest.approx(0.14, abs=0.005)
    assert hot * 62.0 == pytest.approx(8.7, abs=0.4)
    assert LL.cover_particulate(8.0, hot * 62.0 / 45.0) == pytest.approx(0.78, abs=0.02)
    # black (over 99 percent) only after about 20 to 30 minutes
    t = 1800.0
    late = LL.sink_fraction(2.8e-4, 45.0, 25.0, t) * 0.2 * t
    assert LL.cover_particulate(8.0, late / 45.0) > 0.99
    assert soot_made == 60.0


def test_deposit_soot_credits_soot_true_of_the_smoke_units_removed():
    L = Layers()
    k = make_key(0, 3, 5)
    units = 0.004
    _s, adj = L.deposit_soot(k, units, 25.0)
    assert L.mass(k, D.SOOT) == pytest.approx(units * D.SOOT_TRUE, rel=1e-12)
    assert L.led_in[(D.SOOT, "smoke")] == pytest.approx(units * D.SOOT_TRUE)
    assert L.ledger_closes()
    # and the smoke unit gives a visibly sooty ceiling: cover rises with the deposit
    c1 = L.cover_of(k)
    L.deposit_soot(k, units, 25.0)
    assert L.cover_of(k) > c1


# ================================================================ T35 condensation
def test_u_a_plate_at_5_c_in_70_percent_air_gains_0_043_g_per_m2_per_s():
    e_air = 0.7 * LL.psat_sonntag(25.0)
    rho_air = LL.rho_vapour(e_air, 25.0)
    assert rho_air == pytest.approx(16.1, abs=0.1)
    rho_sat = LL.rho_vapour(LL.psat_sonntag(5.0), 5.0)
    assert rho_sat == pytest.approx(6.8, abs=0.1)
    flux = LL.condense_flux(rho_air, 5.0, 5.0)
    rho_a = 101325.0 / (287.05 * 298.15)
    h_m = 5.0 / (rho_a * 1006.0 * 0.88 ** (2.0 / 3.0))
    assert flux == pytest.approx(h_m * (rho_air - rho_sat), rel=0.01)       # closed form
    assert flux == pytest.approx(0.043, rel=0.03)
    assert flux * LL.condense_latent(5.0) == pytest.approx(105.0, rel=0.03)  # W/m2 to the plate


def test_u_the_flux_changes_sign_at_the_dew_point_19_2_c():
    """The dew point of 25 C at 70 % RH is 19.2 C (Magnus/Sonntag). The flux is driven by
    a vapour DENSITY difference (Chilton-Colburn), and density e M/(R T) is taken at each
    surface's own temperature, so it crosses zero 0.4 K below the vapour-pressure dew
    point (18.8 C): the sign changes within 0.5 K of 19.2, and a plate at 20 C stays dry."""
    e_air = 0.7 * LL.psat_sonntag(25.0)
    rho_air = LL.rho_vapour(e_air, 25.0)
    assert LL.dew_point(25.0, 0.7) == pytest.approx(19.2, abs=0.1)
    assert LL.condense_flux(rho_air, 18.5, 5.0) > 0.0
    assert LL.condense_flux(rho_air, 19.6, 5.0) < 0.0
    assert LL.condense_flux(rho_air, 20.0, 5.0) < 0.0                # a plate at 20 C stays dry
    lo, hi = 15.0, 22.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if LL.condense_flux(rho_air, mid, 5.0) > 0.0 else (lo, mid)
    assert lo == pytest.approx(19.2, abs=0.5)


def test_dew_goes_dissolved_frost_goes_crystal_and_the_latent_heat_reaches_the_skin():
    host = FakeHost(n=1, C=100.0, Cs=20.0, dtype=np.float64)
    host.set_Ts(0, 5.0)
    L = Layers()
    k = make_key(0, 0, 4)
    b = _inv(host, L)
    g = 0.0425 * A5 * 10.0                                           # 10 s of the plate case
    T0 = host.Ts(0)
    _s, adj = L.condense(k, g, host.Ts(0))
    host.apply(adj)
    assert L.mass(k, D.WATER, D.DISSOLVED) == pytest.approx(g)
    dE = sum(e for _k, e, _c in adj)
    dC = sum(c for _k, _e, c in adj)
    assert dE - dC * (5.0 - 20.0) == pytest.approx(g * LL.lv_water(5.0), rel=1e-9)   # Lv(5 C)
    assert host.Ts(0) > T0                                           # and the plate warms
    assert _inv(host, L) == pytest.approx(b, abs=1e-9)
    # below zero it is frost, and the latent heat is Lv + Lf (about 2834 J/g at 0 C)
    F = Layers()
    hf = FakeHost(n=1, C=100.0, Cs=20.0, dtype=np.float64)
    hf.set_Ts(0, -5.0)
    b_frost = _inv(hf, F)
    _s, adj = F.condense(k, g, -5.0)
    hf.apply(adj)
    assert F.mass(k, D.WATER, D.CRYSTAL) == pytest.approx(g)
    dE = sum(e for _k, e, _c in adj)
    dC = sum(c for _k, _e, c in adj)
    assert (dE - dC * (-5.0 - 20.0)) / g == pytest.approx(2834.0, rel=0.01)
    assert F.describe(k)[0]["label"] == "frost"
    assert _inv(hf, F) == pytest.approx(b_frost, abs=1e-9)


def test_salting_an_icy_step_melts_the_ice_and_cools_the_skin_above_minus_21():
    host = FakeHost(n=1, C=60.0, Cs=15.0, dtype=np.float64)
    host.set_Ts(0, -5.0)
    L = Layers()
    k = make_key(0, 0, 4)
    _s, adj = L.add(k, {(D.WATER, D.CRYSTAL, 40): 1.0}, Ts=-5.0, T_in=-5.0, cover=1.0)
    host.apply(adj)
    _s, adj = L.add(k, {(D.SALT, D.CRYSTAL, 40): 0.15}, Ts=host.Ts(0), T_in=host.Ts(0))
    host.apply(adj)
    b = _inv(host, L)
    T0 = host.Ts(0)
    Tn, _v = host.latent(L, 0, 10.0)
    assert L.mass(k, D.WATER, D.DISSOLVED) > 0.0                     # it melted
    assert Tn < T0                                                   # endothermic: it cooled
    assert Tn >= D.T_EUTECTIC - 1e-9                                 # and not below the eutectic
    assert _inv(host, L) == pytest.approx(b, rel=1e-9, abs=1e-9)
    # at -25 C salt does nothing: below the eutectic the brine cannot exist
    cold = Layers()
    hc = FakeHost(n=1, C=1e9, Cs=1e9, dtype=np.float64)
    hc.set_Ts(0, -25.0)
    _s, adj = cold.add(k, {(D.WATER, D.CRYSTAL, 40): 1.0}, Ts=-25.0, T_in=-25.0, cover=1.0)
    hc.apply(adj)
    _s, adj = cold.add(k, {(D.SALT, D.CRYSTAL, 40): 0.15}, Ts=-25.0, T_in=-25.0)
    hc.apply(adj)
    hc.latent(cold, 0, 10.0)
    assert cold.mass(k, D.WATER, D.DISSOLVED) == pytest.approx(0.0, abs=1e-12)


# ================================================================ T36 friction
def test_codeonly_wet_fraction_is_zero_dry_and_063_at_one_roughness():
    assert LL.wet_fraction(0.0, 100e-6) == 0.0
    assert LL.wet_fraction(100e-6, 100e-6) == pytest.approx(0.632, abs=0.001)
    mu = LL.friction_mu(0.4, 0.08, 100e-6, 100e-6)
    assert 0.08 < mu < 0.4
    assert LL.friction_mu(0.4, 0.08, 0.0, 100e-6) == 0.4


def test_grip_reads_both_the_layer_and_the_lattice_film():
    L = Layers()
    k = make_key(0, 1, 4)
    assert L.grip(k, 0.4) == 0.4                                     # a clean floor
    L.add(k, {(D.LAMP_OIL, D.DISSOLVED, 0): 0.5e-3 * A5 * 1e6 * 0.8}, cover=1.0, Ts=20.0)
    mu_oil = L.grip(k, 0.4)
    assert mu_oil == pytest.approx(0.08, abs=0.02)                   # 0.4 falls to about 0.1
    dry = Layers()
    mu_lattice = dry.grip(k, 0.4, h_extra=100e-6, lube=0.15)         # a lattice film of 100 um
    assert mu_lattice == pytest.approx(LL.friction_mu(0.4, 0.15, 100e-6, 100e-6))
    assert mu_lattice < 0.4
    thin = Layers()
    thin.add(k, {(D.WATER, D.DISSOLVED, 0): 0.05e-3 * A5 * 1e6}, cover=1.0, Ts=20.0)
    assert thin.grip(k, 0.4) > L.grip(k, 0.4)
    assert thin.grip(k, 0.4, h_extra=1e-3) < thin.grip(k, 0.4)


# ================================================================ momentum: mass_on for the loads
def test_layer_mass_is_there_to_be_carried_mud_on_a_boot_has_weight():
    L = Layers()
    k = make_key(2, 3 * 64 + 5, 0)
    other = make_key(0, 8, 4)
    L.add(k, {(D.CLAY, D.GRAIN, 8): 40.0, (D.WATER, D.DISSOLVED, 0): 60.0}, cover=1.0, Ts=20.0)
    L.add(other, {(D.SOOT, D.GRAIN, 1): 0.5}, cover=1.0, Ts=20.0)
    got = L.mass_on([k, other, make_key(0, 99, 4)])
    assert got.tolist() == pytest.approx([100.0, 0.5, 0.0])
    assert L.mass() == pytest.approx(100.5)
