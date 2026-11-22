"""Smoke sight by ambient light (wave 14a refinement, R1) and the oil soot yield.

Jin reappraisal 2025 (Fire Safety J 104573): K = sigma*V is 4.7-9.5 at 180 lx and 6-11 under
22 lx. The ratio 8.5/7.1 = 1.2 lets a sign be seen through denser smoke in dim light."""
import math

from src.voxel import climate as K
from src.voxel import smokeopt
from src.voxel.materials import AIR, OIL, SEE_SMOKE_TAU, SOOT_OIL
from src.voxel.sim import World


def test_ratio_is_one_at_180_and_up_and_1_2_at_22_and_down():
    assert smokeopt.light_k_ratio(180.0) == 1.0
    assert smokeopt.light_k_ratio(5000.0) == 1.0
    assert abs(smokeopt.light_k_ratio(22.0) - 8.5 / 7.1) < 1e-12
    assert smokeopt.light_k_ratio(0.1) == smokeopt.light_k_ratio(22.0)
    assert abs(smokeopt.light_k_ratio(22.0) - 1.2) < 0.01


def test_ratio_falls_smoothly_between_the_two_lights():
    r = [smokeopt.light_k_ratio(e) for e in (30, 50, 80, 120, 170)]
    assert all(a > b for a, b in zip(r, r[1:])) and 1.0 < r[-1] < r[0] < 1.2


def test_reflecting_object_stays_in_jins_band_at_bright_light_and_dim():
    for e in (300, 180, 22, 1):
        assert 2.0 <= SEE_SMOKE_TAU * smokeopt.light_k_ratio(e) <= 4.0


def test_sky_lux_day_twilight_night():
    assert smokeopt.sky_lux(1.0, 0.0) > 1e5 * 0.9
    assert smokeopt.sky_lux(math.sin(math.radians(-6)), 0.0) < 10.0       # civil twilight ~3.4 lx
    assert smokeopt.sky_lux(math.sin(math.radians(-40)), 0.0) == smokeopt.LUX_FLOOR
    assert smokeopt.sky_lux(1.0, 1.0) < smokeopt.sky_lux(1.0, 0.0)        # cloud dims it


def _hall(climate=None, tick=0):
    w = World(60, 6, 6, voxel_cm=10)
    w.open_sky = False
    w.smass[:] = 0.0
    if climate is not None:
        w.climate = climate
    w.tick = tick
    return w


def _reach(w, g_l):
    w.smoke[:] = g_l * w.vox_l
    k = 0
    while k < 55 and w._sees((1, 3, 3), (2 + k, 3, 3)):
        k += 1
    return k


def test_no_climate_is_the_reference_light_and_the_old_number():
    w = _hall()
    assert w._ambient_lux() == smokeopt.LUX_INDOOR
    assert w._smoke_sight_ext() == smokeopt.sight_ext_per_gl()


def test_night_sees_further_than_noon_in_the_same_smoke():
    c = K.Climate(lat_deg=45.0, cloud=0.0, start_doy=172.0, start_hour=12.0)
    noon = _hall(c)
    night = _hall(c, tick=int(12 * 3600 / 0.025))                           # midnight
    assert noon._ambient_lux() > 1e4 and night._ambient_lux() <= 1.0
    rn, rd = _reach(night, 0.02), _reach(noon, 0.02)
    assert rn > rd
    assert abs(night._smoke_sight_ext() * (8.5 / 7.1) - noon._smoke_sight_ext()) < 1e-9


def test_the_gate_off_is_the_old_law_in_a_dark_world():
    c = K.Climate(lat_deg=45.0, start_hour=0.0)
    w = _hall(c)
    w.smoke_light = False
    assert w._smoke_sight_ext() == smokeopt.sight_ext_per_gl()
    assert World.smoke_light is True


def test_rays_and_single_look_agree_in_the_dark():
    c = K.Climate(lat_deg=45.0, start_hour=0.0)
    w = _hall(c)
    w.smoke[:] = 0.02 * w.vox_l
    for k in (3, 6, 9, 12):
        a, b = (1, 3, 3), (1 + k, 3, 3)
        assert w._sees(a, b) == bool(w.rays([a], [b])[0])


def test_oil_soot_is_a_liquid_hydrocarbon_yield():
    g_per_g = SOOT_OIL * 0.025
    assert 0.037 <= g_per_g <= 0.042           # heptane to kerosene, Tewarson (u)
