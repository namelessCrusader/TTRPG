"""Smoke optics (wave 13, T3, critic 10 / realism-map F19): K = sigma_e * C, optical
density per metre, visibility V = C_v / K, the soot scalar to K, units round trip.

Tags: test_codeonly_* tests the code against its own formula; test_u_* independent but
the source (Mulholland and Croce 8.7 m2/g, Jin C_v 8 and 3) is unverified; the rest
are identities.
"""
import math

import pytest

from src.voxel import layer_data as D
from src.voxel import smokeopt as S


def test_extinction_is_linear_in_soot_concentration():
    assert S.extinction(0.0) == 0.0
    k1 = S.extinction(0.01)
    for f in (2.0, 3.5, 10.0, 0.1):
        assert S.extinction(0.01 * f) == pytest.approx(f * k1, rel=1e-12)
    assert S.extinction(1.0) == S.SIGMA_E == 8.7
    assert S.extinction(0.002, sigma_e=5.0) == pytest.approx(0.01)


def test_visibility_is_c_v_over_k_exactly_and_doubling_concentration_halves_it():
    for c in (1e-4, 3e-3, 0.05, 0.4):
        K = S.extinction(c)
        assert S.visibility(K) == 8.0 / K
        assert S.visibility(K, 3.0) == 3.0 / K
        assert S.visibility_from_conc(c) == pytest.approx(8.0 / (8.7 * c), rel=1e-12)
        assert S.visibility_from_conc(2 * c) == pytest.approx(0.5 * S.visibility_from_conc(c),
                                                              rel=1e-12)
    assert S.visibility(0.0) == math.inf and S.visibility_from_conc(0.0) == math.inf


def test_a_reflecting_sign_is_seen_at_three_eighths_of_an_illuminated_one():
    for c in (1e-3, 0.02, 0.3):
        K = S.extinction(c)
        assert S.visibility_reflecting(K) / S.visibility(K) == pytest.approx(3.0 / 8.0, rel=1e-12)
    assert S.C_V_ILLUMINATED == 8.0 and S.C_V_REFLECTING == 3.0


def test_units_round_trip_g_per_m3_to_per_metre_to_optical_density():
    for c in (1e-5, 2.2e-3, 0.07, 1.5):
        K = S.extinction(c)
        assert S.conc_from_extinction(K) == pytest.approx(c, rel=1e-12)
        od = S.optical_density_per_m(K)
        assert od == pytest.approx(K / math.log(10.0), rel=1e-12)
        assert S.extinction_from_od(od) == pytest.approx(K, rel=1e-12)
        assert S.conc_from_extinction(S.extinction_from_od(od)) == pytest.approx(c, rel=1e-12)
    assert S.optical_density_per_m(math.log(10.0)) == pytest.approx(1.0)
    assert S.transmittance(0.5, 4.0) == pytest.approx(math.exp(-2.0))
    # Beer-Lambert and the decadic form agree: 10^-(OD * path) = exp(-K * path)
    K = S.extinction(0.03)
    assert 10.0 ** (-S.optical_density_per_m(K) * 3.0) == pytest.approx(S.transmittance(K, 3.0))


def test_u_flaming_soot_at_a_tenth_of_a_gram_per_cubic_metre_is_seen_at_9_metres():
    """sigma_e 8.7 m2/g and C_v 8 are (u): 0.1 g/m3 gives K 0.87 per m and V 9.2 m."""
    K = S.extinction(0.1)
    assert K == pytest.approx(0.87)
    assert S.visibility(K) == pytest.approx(9.2, abs=0.05)


def test_the_soot_scalar_gives_k_from_soot_true_and_the_old_law_was_39_times_too_clear():
    assert D.SOOT_TRUE == 0.025
    # 1 g/L of the sim's smoke unit = 1000 g/m3 of smoke = 25 g/m3 true soot
    assert S.soot_conc_from_smoke(1.0) == pytest.approx(25.0)
    assert S.extinction_from_smoke(1.0) == pytest.approx(217.5)
    assert S.extinction_from_smoke(2.0) == pytest.approx(2 * S.extinction_from_smoke(1.0))
    assert S.extinction_old(1.0) == 5.6
    assert S.clearer_by() == pytest.approx(38.8, abs=0.1)
    # the sim's smoke-logged room (~0.12 g/L) is a few metres of view under the new law
    assert S.visibility(S.extinction_from_smoke(0.12 * 0.05)) == pytest.approx(
        8.0 / (217.5 * 0.006), rel=1e-9)


def test_the_old_constant_matches_materials_when_materials_is_present():
    mat = pytest.importorskip("src.voxel.materials")
    assert S.SMOKE_EXT_OLD == mat.SMOKE_EXT


def test_hooks_are_data_not_laws():
    h = S.Hooks()
    assert h.call("co_yield", 1.0) is None and not h.has("purser")
    h.register("purser", lambda co_ppm, minutes: co_ppm * minutes / 35000.0)
    assert h.has("purser") and h.call("purser", 3500.0, 10.0) == pytest.approx(1.0)
    with pytest.raises(KeyError):
        h.register("not_a_hook", lambda: 0)
    assert set(S.HOOK_NAMES) == {"co_yield", "cough", "purser", "smell"}
    other = S.Hooks()
    assert not other.has("purser")                  # instance state, no module global
