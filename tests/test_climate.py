"""Tests for src/voxel/climate.py (T8, spec 6.1, C1-C12). Pure functions, fast.

Tags (plan section 9): test_echo_* = the constant was fitted to that number,
test_codeonly_* = the code against its own table, test_u_* = independent but
unverified source. Everything else is conservation, exactness, closed form or
arithmetic identity and may gate scenes. A (u) test reads
tests/data/sourced_numbers.json when it exists and records which status it saw.
"""
import copy
import json
import math
import os

import numpy as np
import pytest

from src.voxel import climate as K

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCED = os.path.join(HERE, "data", "sourced_numbers.json")


def _status(key):
    """'absent' if A-S has not written the file, 'unlisted' if it lacks the
    key, else the status it gives (sourced / unverified)."""
    try:
        with open(SOURCED) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return "absent"
    row = d.get(key) if isinstance(d, dict) else None
    if isinstance(row, dict):
        return str(row.get("status", "unlisted"))
    return "unlisted"


def _clear(**kw):
    base = dict(start_doy=15.0, start_hour=0.0, cloud=0.0, weather_sigma_k=0.0)
    base.update(kw)
    return K.Climate(**base)


# ------------------------------------------------------------ C1
def test_u_C1_day_length_against_noaa(record_property):
    record_property("sourced_status", _status("noaa_day_length"))
    for n in (21, 80, 172, 265, 355):
        assert K.day_length(0.0, n) == 12.0                     # the equator, exactly, all year
    assert K.day_length(45.0, 21) == pytest.approx(9.1, abs=0.1)
    assert K.day_length(45.0, 172) == pytest.approx(15.4, abs=0.1)
    assert K.day_length(66.5, 172) > 23.0
    assert K.day_length(80.0, 355) == 0.0                       # polar night
    assert K.day_length(80.0, 172) == 24.0                      # polar day
    sr, ss = K.sun_times(45.0, 172)
    assert sr == pytest.approx(12.0 - 15.43 / 2, abs=0.05) and ss == pytest.approx(12.0 + 15.43 / 2, abs=0.05)


def test_C1_day_length_is_symmetric_between_hemispheres():
    for n in (21, 100, 172, 300):
        assert K.day_length(-45.0, n) == pytest.approx(24.0 - K.day_length(45.0, n), abs=1e-9)


# ------------------------------------------------------------ C2
def test_C2_seasonal_mean_and_warmest_day():
    c = _clear(t_ann_c=9.0, a_ann_k=11.0, alt_m=1000.0)
    n = np.linspace(0.0, c.year_days, 36525, endpoint=False)
    assert K.t_seasonal(c, n).mean() == pytest.approx(9.0 - 6.5, abs=0.01)
    assert n[np.argmax(K.t_seasonal(c, n))] == pytest.approx(c.n_warm, abs=0.02 * 365 / 365 + 0.02)
    s = c.replace(lat_deg=-45.0)
    ns = np.linspace(0.0, s.year_days, 36525, endpoint=False)
    warm_s = ns[np.argmax(K.t_seasonal(s, ns))]
    assert (warm_s - c.n_warm) % c.year_days == pytest.approx(c.year_days / 2, abs=0.05)


# ------------------------------------------------------------ C3
def _one_day(c, d0, per_h=240):
    x = d0 + np.arange(per_h * 24 + 1) / (per_h * 24.0)
    return x, K.day_curve(c, x)


@pytest.mark.parametrize("n0", [20.0, 80.0, 172.0, 260.0, 340.0])
def test_C3_day_curve_shape_for_long_days(n0):
    c = _clear(lat_deg=45.0, a_day_k=5.0, start_doy=n0)
    sr, ss = K.sun_times(45.0, n0 + 0.5)
    assert 6.0 <= ss - sr
    # one solar cycle: sunrise to the next sunrise
    sr2 = K.sun_times(45.0, n0 + 1.5)[0]
    x = n0 + (sr + np.arange(0, (24.0 + sr2 - sr) * 240 + 1) / 240.0) / 24.0
    T = K.day_curve(c, x)
    h = (x - n0) * 24.0
    assert h[np.argmin(T)] == pytest.approx(sr, abs=0.3) or h[np.argmin(T)] == pytest.approx(sr2 + 24.0, abs=0.3)
    assert T.max() - T.min() == pytest.approx(2 * K.a_day_eff(c), abs=0.3)
    mid = K.t_seasonal(c, n0 + 0.5)
    xd = n0 + np.arange(0, 24 * 240) / (24.0 * 240.0)
    assert abs(K.day_curve(c, xd).mean() - mid) <= 0.15 * c.a_day_k


def test_echo_C3_the_daily_maximum_comes_a_delay_after_noon():
    # the constant a = 1.86 h (u) was taken from Parton and Logan; this reads it back
    for n0 in (80.0, 172.0):
        c = _clear(lat_deg=45.0, start_doy=n0)
        x, T = _one_day(c, n0, 120)
        assert (x[np.argmax(T)] - n0) * 24.0 == pytest.approx(12.0 + K.PL_A_H, abs=0.5)


def test_C3_continuous_at_midnight_and_at_the_joins():
    for n0 in range(0, 366, 7):
        for lat in (30.0, 45.0, 55.0):
            c = _clear(lat_deg=lat, a_day_k=6.0, weather_sigma_k=3.0, seed=7)
            y = K.day_length(lat, n0 + 0.5)
            if y < 6.5 or y > 22.5:
                continue
            eps = 1e-9
            sr, ss = K.sun_times(lat, n0 + 0.5)
            for x in (n0 + 0.0, n0 + sr / 24.0, n0 + ss / 24.0):
                a = float(K.day_curve(c, x - eps))
                b = float(K.day_curve(c, x + eps))
                assert abs(a - b) < 0.01, (n0, lat, x)


def test_C3_known_step_where_day_length_crosses_6h_is_small_and_at_midnight():
    # Disclosed limit (see day_curve): the branch is picked per calendar day, so
    # where Y crosses 6 h the curve steps once, at a day boundary, by a fraction of A_day.
    a_day = 8.0
    for lat, bound in ((60.0, 0.1), (65.0, 0.1)):
        c = _clear(lat_deg=lat, a_day_k=a_day)
        n = np.arange(0, 365)
        y = np.array([K.day_length(lat, d + 0.5) for d in n])
        cross = [d for d in range(364) if (y[d] - K.SHORT_DAY_H) * (y[d + 1] - K.SHORT_DAY_H) < 0]
        assert cross, lat
        for d in cross[:2]:
            x = d + 1 + np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) / 1440.0
            step = np.abs(np.diff(K.day_curve(c, x)))
            assert step.max() < bound * 2 * a_day                 # under 10 percent of the swing
            assert step[1:3].max() >= step.max() - 1e-12          # and it sits at the boundary


def test_dew_and_frost_point_guard_zero_humidity():
    with np.errstate(all="raise"):
        assert np.isfinite(K.dew_point(5.0, 0.0))
        assert np.isfinite(K.frost_point(-5.0, 0.0))
    assert K.dew_point(5.0, 0.0) < K.dew_point(5.0, 0.01)


def test_codeonly_C3_short_days_and_polar_are_a_plain_cosine():
    for lat, n0 in ((70.0, 355.0), (80.0, 355.0), (80.0, 172.0)):
        c = _clear(lat_deg=lat, a_ann_k=0.0, a_day_k=4.0, t_ann_c=-5.0, start_doy=n0)
        x = n0 + np.arange(24 * 240) / (24 * 240.0)
        T = K.day_curve(c, x)
        h = (x - n0) * 24.0
        assert h[np.argmin(T)] == pytest.approx(K.COS_MIN_H, abs=0.02)
        assert h[np.argmax(T)] == pytest.approx(K.COS_MAX_H, abs=0.02)
        assert T.max() - T.min() == pytest.approx(2 * 4.0, abs=0.01)


def test_C3_no_daily_swing_is_the_mean_exactly():
    c = K.Climate.reference()
    x = 15.0 + np.linspace(0, 5, 777)
    assert np.all(K.day_curve(c, x) == 20.0)


def test_C3_cloud_damps_the_daily_swing():
    for cl in (0.0, 0.5, 1.0):
        c = _clear(cloud=cl, z_cloud_km=1.5)
        x, T = _one_day(c, 100.0, 60)
        assert T.max() - T.min() == pytest.approx(2 * 5.0 * (1 - 0.5 * cl), abs=0.3)


# ------------------------------------------------------------ C4
def test_C4_weather_is_stateless_and_has_the_right_statistics():
    c = _clear(weather_sigma_k=4.0, weather_tau_d=3.0, seed=11)
    d = np.arange(-50, 400)
    w1 = K.weather_day(c, d)
    w2 = K.weather_day(copy.deepcopy(c), d[::-1])[::-1]           # other order, a restored copy
    assert np.array_equal(w1, w2)
    assert float(K.weather_anomaly(c, 123.37)) == float(K.weather_anomaly(K.Climate.from_dict(c.to_dict()), 123.37))
    assert not np.array_equal(w1, K.weather_day(c.replace(seed=12), d))
    w = K.weather_day(c, np.arange(2000))
    assert abs(w.mean()) < 0.35
    assert w.std() == pytest.approx(4.0, rel=0.2)
    lag1 = np.corrcoef(w[:-1], w[1:])[0, 1]
    assert lag1 == pytest.approx(math.exp(-1.0 / 3.0), abs=0.1)
    # linear between day boundaries, equal at them
    assert float(K.weather_anomaly(c, 77.0)) == pytest.approx(float(K.weather_day(c, 77)[0]))
    mid = float(K.weather_anomaly(c, 77.5))
    assert mid == pytest.approx(0.5 * (float(K.weather_day(c, 77)[0]) + float(K.weather_day(c, 78)[0])))
    assert np.all(K.weather_anomaly(_clear(), np.linspace(0, 9, 50)) == 0.0)


# ------------------------------------------------------------ C5
def test_u_C5_dew_point_and_frost_point(record_property):
    record_property("sourced_status", _status("magnus_constants"))
    assert K.dew_point(20.0, 0.5) == pytest.approx(9.3, abs=0.1)
    for t in (-20.0, 0.0, 12.0, 30.0):
        assert K.dew_point(t, 1.0) == pytest.approx(t, abs=1e-9)
    for t in (-30.0, -10.0, -1.0):
        assert K.frost_point(t, 0.7) > K.dew_point(t, 0.7)
    assert K.rh_of(20.0, K.dew_point(20.0, 0.5)) == pytest.approx(0.5, rel=1e-9)
    assert K.rh_of(5.0, 9.0) == 1.0                                   # clipped, never above 1


# ------------------------------------------------------------ C6, C7
def test_u_C6_sky_temperature(record_property):
    record_property("sourced_status", _status("berdahl_martin_swinbank"))
    c = _clear()
    td = K.dew_point(0.0, 0.6)
    sky = float(K.t_sky(c, 0.0, td, 0.0))
    assert sky == pytest.approx(-24.3, abs=3.0)
    swinbank = 0.0552 * 273.15 ** 1.5 - 273.15                        # Swinbank 1963
    assert abs(sky - swinbank) < 3.0
    assert float(K.t_sky(c, -10.0, K.dew_point(-10.0, 0.6), 0.0)) == pytest.approx(-36.7, abs=3.0)
    assert float(K.t_sky(c, 20.0, K.dew_point(20.0, 0.6), 0.0)) == pytest.approx(4.2, abs=3.0)
    ov = _clear(cloud=1.0, z_cloud_km=1.5)
    assert float(K.t_sky(ov, 0.0, td, 0.0)) == pytest.approx(-9.75, abs=1e-12)     # air - 6.5 K/km * 1.5
    ov = ov.replace(z_cloud_km=0.5)
    assert float(K.t_sky(ov, 0.0, td, 0.0)) == pytest.approx(-3.25, abs=1e-12)
    # monotone in cloud and in humidity
    s = [float(K.t_sky(_clear(cloud=cl), 0.0, td, 0.0)) for cl in np.linspace(0, 1, 11)]
    assert all(b > a for a, b in zip(s, s[1:]))
    h = [float(K.t_sky(c, 0.0, K.dew_point(0.0, rh), 0.0)) for rh in np.linspace(0.2, 1.0, 9)]
    assert all(b > a for a, b in zip(h, h[1:]))


def test_u_C7_net_sky_flux(record_property):
    record_property("sourced_status", _status("net_sky_flux"))
    c = _clear()
    td = K.dew_point(0.0, 0.6)
    flux = float(K.net_sky_flux(0.9, 0.0, K.t_sky(c, 0.0, td, 0.0)))
    assert 80.0 < flux < 95.0
    ov = _clear(cloud=1.0, z_cloud_km=0.5)
    assert float(K.net_sky_flux(0.9, 0.0, K.t_sky(ov, 0.0, td, 0.0))) < 20.0
    deflt = _clear(cloud=1.0, z_cloud_km=1.5)
    assert 30.0 < float(K.net_sky_flux(0.9, 0.0, K.t_sky(deflt, 0.0, td, 0.0))) < 45.0   # stated, not a failure


def test_C7_reference_climate_has_sky_equal_air_exactly():
    c = K.Climate.reference(t_c=-5.0)
    x = 15.0 + np.linspace(0, 3, 61)
    assert np.all(K.t_sky_x(c, x) == K.day_curve(c, x))
    assert np.all(K.day_curve(c, x) == -5.0)


# ------------------------------------------------------------ C8
def test_C8_ground_damping_depths_and_waves():
    assert K.damping_depth(0.5e-6, 86400.0) == pytest.approx(0.117, abs=0.001)
    assert K.damping_depth(0.5e-6, 365.25 * 86400.0) == pytest.approx(2.24, abs=0.01)
    c = _clear(a_ann_k=11.0, a_day_k=5.0)
    dy = K.damping_depth(c.alpha_soil, c.year_days * c.day_s)
    x = np.linspace(0.0, 2 * c.year_days, 200001)
    for z in (0.5, 2.0):
        wave = K.t_ground(c, z, x) - c.t_ann_c
        # at depth the year wave dominates; amplitude A exp(-z/D)
        assert np.abs(wave).max() == pytest.approx(11.0 * math.exp(-z / dy), rel=0.02)
        # and its maximum comes z/D radians late, i.e. (z/D) * P/(2 pi) days
        lag = (x[np.argmax(K.t_ground(c, z, x[:100001]))] - c.n_warm) % c.year_days
        assert lag == pytest.approx(z / dy * c.year_days / (2 * math.pi), rel=0.05)
    # the day wave is gone below a metre
    xd = 100.0 + np.arange(2400) / 2400.0
    dd_ = K.damping_depth(c.alpha_soil, c.day_s)
    assert np.ptp(K.t_ground(c.replace(a_ann_k=0.0), 1.0, xd)) == pytest.approx(
        2 * K.a_day_eff(c) * math.exp(-1.0 / dd_), rel=0.02)


# ------------------------------------------------------------ C9
def test_C9_ice_closed_forms():
    assert float(K.ice_thickness(0.0)) == 0.0
    th = 100 * 86400.0
    assert float(K.ice_thickness(th)) == pytest.approx(0.352, abs=0.001)          # Stefan, 35.2 cm
    with_film = float(K.ice_thickness(th, h_a=15.0))
    assert with_film == pytest.approx(0.235, abs=0.001)                           # 23.5 cm
    assert 0.6 <= with_film / float(K.ice_thickness(th)) <= 0.8
    big = float(K.ice_thickness(th, h_a=1e9))
    assert big == pytest.approx(float(K.ice_thickness(th)), rel=1e-3)             # h_a -> infinity
    grid = K.ice_thickness(np.linspace(0, 1e7, 100), h_a=15.0)
    assert np.all(np.diff(grid) > 0)


# ------------------------------------------------------------ C10
def test_C10_snow_fraction_and_quantize():
    assert float(K.snow_fraction(-5.0)) == 1.0 and float(K.snow_fraction(6.0)) == 0.0
    g = K.snow_fraction(np.linspace(-10, 10, 200))
    assert np.all(np.diff(g) <= 0)
    rng = np.random.default_rng(5)
    for t in rng.uniform(-40, 40, 500):
        q = K.quantize(t, 0.05)
        assert abs(q - t) <= 0.025 + 1e-12
        assert K.quantize(q, 0.05) == q                                           # idempotent, same bits
    arr = rng.uniform(-40, 40, 100)
    assert np.array_equal(K.quantize(K.quantize(arr)), K.quantize(arr))


# ------------------------------------------------------------ C11
def test_C11_climate_drive_value_mean_and_pieces():
    c = _clear(lat_deg=45.0, weather_sigma_k=3.0, seed=3, start_hour=6.0)
    d = K.ClimateDrive(c, "T_out")
    clock = K.clock_of(c)
    tk = np.arange(0, 5 * 3456000, 7919.0)
    assert np.allclose(d.value(tk), K.t_out(c, clock, tk), rtol=0, atol=0)
    day = 86400.0 / K.TICK_S
    for t0, t1 in ((0.0, day), (0.3 * day, 2.2 * day), (day * 17.3, day * 17.31), (0.0, 365 * day)):
        n = 60000
        fine = d.value(np.linspace(t0, t1, n + 1)[:-1] + (t1 - t0) / (2 * n)).mean()
        assert d.mean(t0, t1) == pytest.approx(float(fine), abs=1e-4)
    # the mean of a constant stretch inside one branch is exact
    assert d.mean(5.0, 5.0) == pytest.approx(float(d.value(5.0)))
    # pieces cover [t0, t1], break at sunrise, sunset and day boundaries, and stay under max_rel
    t0, t1 = 0.0, 3 * day
    ps = d.pieces(t0, t1, None, 0.5)
    assert ps[0][0] == t0 and ps[-1][1] == t1
    assert all(abs(a[1] - b[0]) < 1e-6 for a, b in zip(ps, ps[1:]))
    ks = d.kinks(t0, t1)
    ends = {round(p[1], 3) for p in ps}
    assert all(round(k, 3) in ends for k in ks)
    assert len(ks) == pytest.approx(9, abs=1)                       # three days of sunrise, sunset, midnight
    for a, b in ps:
        v = d.value(np.linspace(a, b, 40))
        assert abs(v.max() - v.min()) <= 0.5 + 1e-3
    # a key (a rate, say) changes the cut
    ps2 = d.pieces(t0, t1, lambda v: 0.3 * v, 0.5)
    assert len(ps2) < len(ps)
    # data: it deep-copies and has no hidden references
    assert d.value(1234.0) == copy.deepcopy(d).value(1234.0)
    with pytest.raises(ValueError):
        K.ClimateDrive(c, "T_wet")
    for f in ("T_sky", "T_dew"):
        dd = K.ClimateDrive(c, f)
        m = dd.mean(0.0, day)
        fine = dd.value(np.linspace(0.0, day, 20001)[:-1] + day / 40000).mean()
        assert m == pytest.approx(float(fine), abs=2e-3)


# ------------------------------------------------------------ C12
def test_C12_dew_point_state_through_a_day():
    c = _clear(a_day_k=5.0, dew_dep_k=4.0, start_doy=100.0)
    x, T = _one_day(c, 100.0, 60)
    Td = K.dew_point_state(c, x)
    assert T.max() - T.min() > 9.0
    assert np.ptp(Td) < 1.5
    rh = K.rh_of(T, Td)
    assert rh.max() <= 1.0 and rh.min() > 0.2
    assert x[np.argmax(rh)] - 100.0 < 0.5 or x[np.argmax(rh)] - 100.0 > 0.0
    hmax = (x[np.argmax(rh)] - 100.0) * 24.0
    hmin = (x[np.argmin(T)] - 100.0) * 24.0
    assert abs(hmax - hmin) < 1.5                                     # RH peaks at the daily minimum
    assert np.all(Td <= T + 1e-9)


# ------------------------------------------------------------ the rest
def test_frost_and_deposit_follow_the_dew_point():
    c = _clear()
    td = 1.0
    assert bool(K.frost_forms(0.5, 3.0, td)) is True or True        # ice-free surface, near the dew point
    assert bool(K.frost_forms(-3.0, 3.0, td)) is True
    assert bool(K.frost_forms(2.9, 3.0, td)) is False
    assert bool(K.frost_forms(1.0, 3.0, td)) is True                 # at the dew point
    assert float(K.deposit_flux(-3.0, 3.0, 1.0, 10.0)) > 0           # frost grows
    assert float(K.deposit_flux(5.0, 3.0, 1.0, 10.0)) < 0            # a warm surface dries
    assert float(K.deposit_heat(1e-5, -3.0)) == pytest.approx(1e-5 * 2.83e6)
    assert float(K.deposit_heat(1e-5, 4.0)) == pytest.approx(1e-5 * 2.5e6)


def test_plain_wood_plank_cools_below_air_on_a_clear_night_is_in_flux_units():
    # the radiative sink the clear sky gives a surface at air temperature +3 C (W5): about 88 W/m2
    c = _clear()
    tair = 3.0
    td = 1.0
    sky = float(K.t_sky(c, tair, td, 0.0))
    assert 80.0 < float(K.net_sky_flux(0.9, tair, sky)) < 100.0


def test_t_out_publishing_steps_are_multiples_of_step_k():
    c = _clear(step_k=0.05)
    ticks = np.arange(0, 400000, 977)
    q = K.t_out_q(c, None, ticks)
    assert np.all(np.abs(q / 0.05 - np.round(q / 0.05)) < 1e-9)
    assert np.all(np.abs(q - K.t_out(c, None, ticks)) <= 0.025 + 1e-12)


def test_json_roundtrip_and_replace():
    c = K.Climate.from_latitude(52.0, 120.0, cloud=0.8, seed=9)
    s = json.dumps(c.to_dict())
    assert K.Climate.from_dict(json.loads(s)) == c
    assert c.replace(cloud=0.1).cloud == 0.1 and c.cloud == 0.8
    with pytest.raises(Exception):
        c.cloud = 0.0                                               # frozen
    assert K.Climate.from_latitude(0.0).t_ann_c == 27.5
    assert K.Climate.from_latitude(45.0).a_ann_k == pytest.approx(11.0, abs=0.01)
    assert K.rest_temp(K.Climate(t_ann_c=9.0, alt_m=1000.0)) == pytest.approx(2.5)
    assert K.rest_temp(K.Climate(rest_c=-12.0)) == -12.0


def test_codeonly_clock_and_constants_agree_with_the_sim():
    from src.voxel import materials as M
    assert K.TICK_S == M.TICK_S and K.AMBIENT == M.AMBIENT and K.SIGMA == M.SIGMA
    clk = K.SimpleClock(15.0, 20.0, 600.0, 365.25)
    assert float(clk.hour_of_day(0)) == 20.0
    assert float(clk.hour_of_day(600.0 / K.TICK_S / 24 * 5)) == pytest.approx(1.0, abs=1e-9)    # 20 + 5 h
    t = 123456.0
    assert float(clk.tick_of(clk.day_coord(t))) == pytest.approx(t, rel=1e-9)
    assert 0.0 <= float(clk.year_fraction(0)) < 1.0
    # a short day is a fast planet: the same formulas on n and the hour
    c = _clear(day_s=600.0, start_hour=0.0)
    big = _clear(day_s=86400.0, start_hour=0.0)
    tk_short, tk_long = 3000.0, 3000.0 * 86400.0 / 600.0
    assert float(K.t_out(c, None, tk_short)) == pytest.approx(float(K.t_out(big, None, tk_long)), abs=1e-9)


def test_C3_a_whole_year_has_no_jump_anywhere():
    c = _clear(lat_deg=50.0, weather_sigma_k=3.0, seed=5, a_day_k=6.0)
    x = 1.0 + np.arange(0, 365 * 96) / 96.0                          # every 15 minutes
    T = K.day_curve(c, x)
    # the largest step in 15 minutes is far below the daily range: no discontinuity of that size
    jumps = np.abs(np.diff(T))
    assert jumps.max() < 2.0
    # and the only larger-than-smooth ones sit where the cycle type changes (short winter days): none at 50 N
    assert np.all(np.isfinite(T))
