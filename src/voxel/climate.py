"""CLIMATE: the outside of the world as DATA, with no World in it (wave 13, T8).

A frozen `Climate` record plus pure functions give, for any tick:

    T_out(t)   air temperature at screen height (seasonal + weather + a daily
               curve built per SOLAR CYCLE, Parton and Logan 1981)
    T_dew(t)   dew point (nearly constant through a day, so RH peaks at dawn)
    T_sky(t)   effective sky temperature (Berdahl and Martin 1984, plus cloud)
    ground temperature at a depth, the snow share, a frost test, the
    deposition flux of frost or dew, and the growth of ice on a pond.

It imports `math` and `numpy` and nothing else. Every number marked (u) is
unverified: from memory or a secondary source, to be checked before a test
locks it (docs of this wave: specs/T8-cold_v2.txt, sections 2.1 and 3).

WHEN. Time is a DAY COORDINATE `n` (days, 1.0 = the start of 1 January, so the
calendar day D = floor(n) covers [D, D+1) and its solar noon is D + 0.5):

    n(tick) = start_doy + start_hour/24 + tick * TICK_S / day_s

The hour of day is 24*(n - floor(n)). (Spec 3.1 wrote n without the start
hour; the two differ by under a day of season. This keeps n and the hour on
one clock.) `day_s` is data: a proof world may shorten it to 600 s and every
formula still reads n and the hour, so a short day is a fast planet.

DOCSTRING RULE, until wave 14 supplies the sun: use a diurnal climate at NIGHT
or under OVERCAST. By day the sky sink would chill sunlit surfaces below the
air, which is wrong. The shortwave seam is the `sw` argument of the radiation
kernel in outside.py (zero in T8).

DOCUMENTED LIMIT: `alt_m` changes only the temperature (6.5 K per km, ICAO
lapse). Air density, oxygen and boiling points stay at sea level.

Marked for removal: `ice_thickness` is dropped when T6 has its own `stefan()`
(one law, one place; decision D7).
"""
import math
from dataclasses import asdict, dataclass, replace

import numpy as np

TICK_S = 0.025               # seconds in a tick (materials.TICK_S; a cross test keeps them equal)
AMBIENT = 20.0               # the fixed reference of E (materials.AMBIENT)
K0 = 273.15
SIGMA = 5.670374e-8          # Stefan-Boltzmann, W/(m2 K4)
LAPSE_K_PER_M = 0.0065       # ICAO standard lapse rate (certain)
PL_A_H = 1.86                # Parton-Logan delay of the daily maximum, hours (u)
PL_B = 2.2                   # Parton-Logan night decay constant (u)
SHORT_DAY_H = 6.0            # below this many hours of daylight: plain cosine
COS_MIN_H = 4.0              # plain cosine: minimum at 04:00 (u)
COS_MAX_H = 16.0             # plain cosine: maximum at 16:00 (u)
CLOUD_LAPSE_K_PER_KM = 6.5   # cloud base is colder than the surface air by this (u)
WEATHER_TERMS = 11           # the AR(1) sum runs over j = 0..10
LE_TWO_THIRDS = 0.85         # Lewis number to the 2/3 for h_m = h_c/(rho cp Le^(2/3)) (u)
L_FROST = 2.83e6             # J/kg, latent heat of sublimation
L_DEW = 2.5e6                # J/kg, latent heat of condensation
RHO_AIR = 1.2                # kg/m3
CP_AIR = 1005.0              # J/(kg K)
R_VAPOUR = 461.5             # J/(kg K)
K_ICE, RHO_ICE, L_ICE = 2.2, 917.0, 334e3   # ice conductivity, density, latent heat (u)


@dataclass(frozen=True)
class Climate:
    """The outside, as plain numbers (JSON-able; `asdict`/`from_dict`)."""
    lat_deg: float = 45.0         # latitude; negative = southern hemisphere
    alt_m: float = 0.0            # height above sea level (temperature only)
    t_ann_c: float = 9.0          # annual mean air temperature at sea level, C
    a_ann_k: float = 11.0         # half range of the annual cycle, K
    n_warm: float = 202.0         # day of year of the warmest day (northern)
    a_day_k: float = 5.0          # half range of the daily cycle, K (clear sky)
    dew_dep_k: float = 4.0        # dew-point depression of the daily mean, K
    cloud: float = 0.3            # cloud cover, 0..1
    z_cloud_km: float = 1.5       # cloud base height (u); overcast scenes use 0.5
    wind_ms: float = 0.0          # reserved; only a hook in k_film
    weather_sigma_k: float = 0.0  # day-to-day weather swing (0 = none)
    weather_tau_d: float = 3.0    # memory of the swings, days (u)
    seed: int = 0
    start_doy: float = 15.0       # the date at tick 0
    start_hour: float = 20.0      # local solar hour at tick 0
    day_s: float = 86400.0        # seconds in a day (a proof world may shorten it)
    year_days: float = 365.25
    rest_c: object = None         # rest temperature of solids and enclosed air; None = annual mean
    k_film: float = 1.0           # 1/s, outdoor air touching a surface
    k_far: float = 0.1            # 1/s, free outdoor air (u)
    step_k: float = 0.05          # K, T_out is published in steps of this size
    ground_top_m: float = 0.0     # height of the ground surface above the lattice floor
    ground_depth_m: object = None  # depth to the deep ground reservoir; None = adiabatic bottom
    alpha_soil: float = 0.5e-6    # soil thermal diffusivity, m2/s (u)
    plume_k: float = 100.0        # K above T_out: hotter outdoor air is a plume or flame, left to the buoyancy law (not relaxed)

    def replace(self, **kw):
        return replace(self, **kw)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @classmethod
    def from_latitude(cls, lat, alt=0.0, **kw):
        """Defaults from two curves fitted by eye to a few dozen cities (u).
        Real cities spread +-6 K around them; scenes should author their own."""
        t_ann = 27.5 - 0.0090 * lat * lat
        a_ann = 22.0 * math.sin(math.radians(lat)) ** 2
        kw.setdefault("t_ann_c", t_ann)
        kw.setdefault("a_ann_k", a_ann)
        return cls(lat_deg=lat, alt_m=alt, **kw)

    @classmethod
    def reference(cls, t_c=20.0, **kw):
        """The 'no outside' record: T_out = T_sky = rest = t_c all the time.
        Used by the reference-climate continuity test (R1). A cloud of 1 with a
        base at the ground makes the sky exactly the air."""
        base = dict(t_ann_c=t_c, a_ann_k=0.0, a_day_k=0.0, cloud=1.0, z_cloud_km=0.0,
                    weather_sigma_k=0.0, rest_c=t_c, lat_deg=0.0)
        base.update(kw)
        return cls(**base)


def rest_temp(c):
    """The world's rest temperature for solids and enclosed air."""
    if c.rest_c is not None:
        return float(c.rest_c)
    return float(c.t_ann_c - LAPSE_K_PER_M * c.alt_m)


# ---------------------------------------------------------------- the clock
@dataclass(frozen=True)
class SimpleClock:
    """What climate needs from a clock: day coordinate and hour of day.
    A T5 Clock can stand in if it has `day_coord` (or year_fraction and
    hour_of_day)."""
    start_doy: float = 15.0
    start_hour: float = 20.0
    day_s: float = 86400.0
    year_days: float = 365.25

    def day_coord(self, tick):
        t = np.asarray(tick, np.float64) * TICK_S
        return self.start_doy + self.start_hour / 24.0 + t / self.day_s

    def year_fraction(self, tick):
        return ((self.day_coord(tick) - 1.0) / self.year_days) % 1.0

    def hour_of_day(self, tick):
        t = np.asarray(tick, np.float64) * TICK_S
        return (self.start_hour + 24.0 * t / self.day_s) % 24.0

    def tick_of(self, n):
        """The tick at day coordinate n (float)."""
        n0 = self.start_doy + self.start_hour / 24.0
        return (np.asarray(n, np.float64) - n0) * self.day_s / TICK_S


def clock_of(c):
    return SimpleClock(c.start_doy, c.start_hour, c.day_s, c.year_days)


def day_coord(c, clock, tick):
    """The day coordinate n at `tick`, from a clock or the climate's own."""
    clock = clock if clock is not None else clock_of(c)
    f = getattr(clock, "day_coord", None)
    if f is not None:
        return f(tick)
    yf = np.asarray(clock.year_fraction(tick), np.float64)
    hr = np.asarray(clock.hour_of_day(tick), np.float64)
    d = np.floor(1.0 + yf * c.year_days + 1e-9)
    return d + hr / 24.0


def _ret(x, scalar):
    return float(np.ravel(x)[0]) if scalar else x


# ---------------------------------------------------------------- the sun
def declination(n):
    """Solar declination, radians. Cooper 1969: 23.44 deg * sin(2 pi (284+n)/365)."""
    n = np.asarray(n, np.float64)
    return np.radians(23.44) * np.sin(2.0 * math.pi * (284.0 + n) / 365.0)


def day_length(lat_deg, n):
    """Hours of daylight: (24/pi) arccos(-tan(phi) tan(delta)), clipped, so 0
    in polar night and 24 in polar day. The equator is exactly 12 h."""
    scalar = np.ndim(lat_deg) == 0 and np.ndim(n) == 0
    lat = np.clip(np.asarray(lat_deg, np.float64), -89.999, 89.999)
    x = -np.tan(np.radians(lat)) * np.tan(declination(n))
    y = (24.0 / math.pi) * np.arccos(np.clip(x, -1.0, 1.0))
    return _ret(y, scalar)


def sun_times(lat_deg, n):
    """(sunrise, sunset) in local solar hours. The equation of time (under 16
    minutes) is ignored. Wave 14's sun replaces this one seam."""
    y = day_length(lat_deg, n)
    return 12.0 - y / 2.0, 12.0 + y / 2.0


# ---------------------------------------------------------------- seasons and weather
def t_seasonal(c, n):
    """Seasonal mean air temperature at day coordinate n."""
    scalar = np.ndim(n) == 0
    n = np.asarray(n, np.float64)
    nw = c.n_warm + (0.5 * c.year_days if c.lat_deg < 0 else 0.0)
    t = (c.t_ann_c - LAPSE_K_PER_M * c.alt_m
         + c.a_ann_k * np.cos(2.0 * math.pi * (n - nw) / c.year_days))
    return _ret(t, scalar)


_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _splitmix(x):
    """One round of splitmix64 on a uint64 array (wraps silently)."""
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        z = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _unit_normal(seed, d):
    """A unit normal from hash(seed, day): stateless, the same in any process,
    after a restore, in any call order."""
    d = np.atleast_1d(np.asarray(d, np.int64))
    key = np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF)
    with np.errstate(over="ignore"):
        k = _splitmix(d.view(np.uint64) * np.uint64(0xD6E8FEB86659FD93) + key)
        a = _splitmix(k)
        b = _splitmix(k + np.uint64(1))
    u1 = ((a >> np.uint64(11)).astype(np.float64) + 0.5) * (1.0 / 9007199254740992.0)
    u2 = (b >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * math.pi * u2)


def weather_day(c, d):
    """The weather anomaly (K) at the START of integer day d: an AR(1) with
    memory tau, written as a finite sum of hashed normals so it is stateless."""
    d = np.atleast_1d(np.asarray(d, np.int64))
    if c.weather_sigma_k == 0.0:
        return np.zeros(d.shape, np.float64)
    a = math.exp(-1.0 / c.weather_tau_d)
    out = np.zeros(d.shape, np.float64)
    for j in range(WEATHER_TERMS):
        out += (a ** j) * _unit_normal(c.seed, d - j)
    return c.weather_sigma_k * math.sqrt(1.0 - a * a) * out


def weather_anomaly(c, n):
    """Weather at day coordinate n, linear between day boundaries."""
    scalar = np.ndim(n) == 0
    n = np.atleast_1d(np.asarray(n, np.float64))
    if c.weather_sigma_k == 0.0:
        return _ret(np.zeros(n.shape), scalar)
    d = np.floor(n).astype(np.int64)
    f = n - d
    w = (1.0 - f) * weather_day(c, d) + f * weather_day(c, d + 1)
    return _ret(w, scalar)


def a_day_eff(c):
    """Daily half range after cloud damping (u)."""
    return c.a_day_k * (1.0 - 0.5 * c.cloud)


# ---------------------------------------------------------------- the daily curve
def _cycle(c, d):
    """Parameters of the solar cycle that starts at sunrise of calendar day d
    (arrays). The day's mean and weather are taken at its solar noon, d + 0.5."""
    n = np.asarray(d, np.float64) + 0.5
    mean = t_seasonal(c, n) + weather_anomaly(c, n)
    amp = a_day_eff(c)
    y = day_length(c.lat_deg, n)
    tn, tx = mean - amp, mean + amp
    s = np.sin(math.pi * y / (y + 2.0 * PL_A_H))
    return dict(Y=y, sr=12.0 - y / 2.0, ss=12.0 + y / 2.0, Tn=tn, Tx=tx, Tsun=tn + (tx - tn) * s,
                mean=mean)


def _pl_ok(y):
    return (y >= SHORT_DAY_H) & (y <= 23.0)


def _locate(c, x):
    """For day coordinates x (array): which branch applies and its parameters.
    kind 0 = plain cosine, 1 = Parton-Logan day, 2 = Parton-Logan night."""
    x = np.atleast_1d(np.asarray(x, np.float64))
    dd = np.floor(x).astype(np.int64)
    h = 24.0 * (x - dd)
    p0, pm, pp = _cycle(c, dd), _cycle(c, dd - 1), _cycle(c, dd + 1)
    pl = (_pl_ok(p0["Y"]) & (pm["Y"] > 0.5) & (pm["Y"] < 23.5)
          & (pp["Y"] > 0.5) & (pp["Y"] < 23.5))
    before = h < p0["sr"]
    during = ~before & (h < p0["ss"])
    # the cycle this instant belongs to
    cyc = np.where(before, dd - 1, dd)
    P = {k: np.where(before, pm[k], p0[k]) for k in p0}
    Pn = {k: np.where(before, p0[k], pp[k]) for k in p0}    # the NEXT cycle
    kind = np.where(~pl, 0, np.where(during, 1, 2))
    # the local coordinate: m (hours since sunrise) by day, nn (since sunset) by night
    hh = 24.0 * (x - cyc)                                     # hours since midnight of the cycle's day
    m = hh - P["sr"]
    nn = hh - P["ss"]
    night_len = (Pn["sr"] + 24.0) - P["ss"]
    eb = math.exp(-PL_B)
    tinf = (Pn["Tn"] - P["Tsun"] * eb) / (1.0 - eb)
    return dict(x=x, dd=dd, h=h, kind=kind, m=m, nn=nn, N=night_len, tinf=tinf,
                Tn=P["Tn"], Tx=P["Tx"], Tsun=P["Tsun"], Y=P["Y"])


def _cos_part(c, x, want_prim):
    """Plain cosine branch: continuous mean (seasonal + weather) minus a cosine
    with its minimum at 04:00. Returns value (and antiderivative in days,
    relative to the start of the calendar day for the weather term)."""
    dd = np.floor(x).astype(np.int64)
    f = x - dd
    amp = a_day_eff(c)
    w0, w1 = weather_day(c, dd), weather_day(c, dd + 1)
    val = (t_seasonal(c, x) + (1.0 - f) * w0 + f * w1
           - amp * np.cos(2.0 * math.pi * (f - COS_MIN_H / 24.0)))
    if not want_prim:
        return val, None
    nw = c.n_warm + (0.5 * c.year_days if c.lat_deg < 0 else 0.0)
    ty = 2.0 * math.pi / c.year_days
    prim = ((c.t_ann_c - LAPSE_K_PER_M * c.alt_m) * x
            + c.a_ann_k * np.sin(ty * (x - nw)) / ty
            + w0 * f + (w1 - w0) * f * f / 2.0
            - amp * np.sin(2.0 * math.pi * (f - COS_MIN_H / 24.0)) / (2.0 * math.pi))
    return val, prim


def _eval(c, x, want_prim=False):
    """Value of T_out at day coordinates x (and, if asked, an antiderivative in
    K*days valid for DIFFERENCES within one piece of one branch)."""
    L = _locate(c, x)
    xa = L["x"]
    cv, cp = _cos_part(c, xa, want_prim)
    amp = a_day_eff(c)
    a = PL_A_H
    y = np.maximum(L["Y"], 1e-9)
    # day branch
    arg = math.pi * L["m"] / (y + 2.0 * a)
    dv = L["Tn"] + (L["Tx"] - L["Tn"]) * np.sin(arg)
    # night branch
    nlen = np.maximum(L["N"], 1e-9)
    e = np.exp(np.clip(-PL_B * L["nn"] / nlen, -700.0, 700.0))
    nv = L["tinf"] + (L["Tsun"] - L["tinf"]) * e
    val = np.where(L["kind"] == 0, cv, np.where(L["kind"] == 1, dv, nv))
    if amp == 0.0:                       # no daily swing: the mean, exactly
        val = np.where(L["kind"] == 0, cv, np.where(L["kind"] == 1, L["Tn"], L["Tn"]))
    if not want_prim:
        return val, None
    dp = (L["Tn"] * L["m"] - (L["Tx"] - L["Tn"]) * (y + 2.0 * a) / math.pi * np.cos(arg)) / 24.0
    npim = (L["tinf"] * L["nn"] - (L["Tsun"] - L["tinf"]) * (nlen / PL_B) * e) / 24.0
    prim = np.where(L["kind"] == 0, cp, np.where(L["kind"] == 1, dp, npim))
    return val, prim


def day_curve(c, x):
    """T_out at day coordinate(s) x: seasonal + weather + the daily curve.

    By day (Y >= 6 h): Tn + (Tx - Tn) sin(pi m/(Y + 2a)), m hours since
    sunrise. By night: Tinf + (Ts - Tinf) exp(-b nn/N), nn hours since sunset,
    N the real night length, Tinf chosen so the night ends exactly at the NEXT
    cycle's minimum. The cycle runs sunrise to sunrise, so there is no step at
    midnight. For Y < 6 h, polar day and polar night: a plain cosine, minimum
    at 04:00 and maximum at 16:00, range 2*A_dayEff.

    Known limit: the branch is chosen per calendar day from that day's length
    Y (Parton-Logan only for 6 h <= Y <= 23 h). Where Y crosses 6 h (latitudes
    above about 58) or 23 h (above about 68) the curve steps at midnight of
    that one day: about 0.02 of the daily swing at 60 N, up to 0.2 of it at 70-80 N
    (measured with A_day 8 K, swing 16 K: 0.3 K, and 1.3-3.3 K). Below about 55 N there is no
    such crossing and the largest step per minute is the curve's own slope
    (under 0.07 K at A_day 8 K). Everywhere else the curve is continuous."""
    scalar = np.ndim(x) == 0
    v, _ = _eval(c, x)
    return _ret(v[0] if scalar else v, scalar)


def t_out_x(c, x):
    return day_curve(c, x)


def t_out(c, clock, tick):
    """Air temperature at `tick`."""
    return day_curve(c, day_coord(c, clock, tick))


def quantize(t, step=0.05):
    """Round to a multiple of `step`. Idempotent: quantize(quantize(t)) is the same bits."""
    return np.round(np.asarray(t, np.float64) / step) * step if np.ndim(t) else \
        float(round(float(t) / step) * step)


def t_out_q(c, clock, tick):
    return quantize(t_out(c, clock, tick), c.step_k)


# ---------------------------------------------------------------- humidity
def psat_magnus(t_c):
    """Saturation vapour pressure over water, hPa (Sonntag 1990 form; (u))."""
    t = np.asarray(t_c, np.float64)
    return 6.112 * np.exp(17.62 * t / (243.12 + t))


def psat_ice(t_c):
    t = np.asarray(t_c, np.float64)
    return 6.112 * np.exp(22.46 * t / (272.62 + t))


def rh_of(t_c, t_dew_c):
    """Relative humidity from air and dew-point temperatures, clipped to 1."""
    scalar = np.ndim(t_c) == 0 and np.ndim(t_dew_c) == 0
    r = np.clip(psat_magnus(t_dew_c) / psat_magnus(t_c), 0.0, 1.0)
    return _ret(r, scalar)


def dew_point(t_c, rh):
    """Dew point over water (Magnus): T_dew = 243.12 g / (17.62 - g)."""
    scalar = np.ndim(t_c) == 0 and np.ndim(rh) == 0
    t = np.asarray(t_c, np.float64)
    g = np.log(np.clip(np.asarray(rh, np.float64), 1e-6, 1.0)) + 17.62 * t / (243.12 + t)
    return _ret(243.12 * g / (17.62 - g), scalar)


def frost_point(t_c, rh):
    """Frost point over ice (used when T < 0 C): 272.62 g / (22.46 - g)."""
    scalar = np.ndim(t_c) == 0 and np.ndim(rh) == 0
    t = np.asarray(t_c, np.float64)
    g = np.log(np.clip(np.asarray(rh, np.float64), 1e-6, 1.0)) + 22.46 * t / (272.62 + t)
    return _ret(272.62 * g / (22.46 - g), scalar)


def dew_point_state(c, x):
    """The dew point at day coordinate x. It is nearly constant through a day
    (real dew points move slowly), so RH peaks at the daily minimum:

        T_dew = min(T_s + w - dew_dep, T_s + w - A_dayEff, T_out)

    which keeps RH at or below 1."""
    scalar = np.ndim(x) == 0
    xa = np.asarray(x, np.float64)
    base = t_seasonal(c, xa) + weather_anomaly(c, xa)
    td = base - max(c.dew_dep_k, a_day_eff(c))
    td = np.minimum(td, day_curve(c, xa))
    return _ret(td, scalar)


# ---------------------------------------------------------------- the sky
def eps_clear(t_dew_c, hour):
    """Clear-sky emissivity, Berdahl and Martin 1984 (u): 0.711 + 0.0056 Td +
    7.3e-5 Td^2 + 0.013 cos(2 pi h/24), Td in C, h the hour of day."""
    td = np.asarray(t_dew_c, np.float64)
    return (0.711 + 0.0056 * td + 7.3e-5 * td * td
            + 0.013 * np.cos(2.0 * math.pi * np.asarray(hour, np.float64) / 24.0))


def t_sky(c, t_air_c, t_dew_c, hour=0.0):
    """Effective sky temperature, C. T_sky^4 = (1-cloud) T_clear^4 + cloud T_cb^4,
    T_clear = eps_clear^(1/4) T_air, T_cb = T_air - 6.5 K/km * z_cloud. At
    cloud 1 the sky is the cloud base exactly (no power round trip)."""
    scalar = np.ndim(t_air_c) == 0 and np.ndim(t_dew_c) == 0 and np.ndim(hour) == 0
    ta = np.asarray(t_air_c, np.float64) + K0
    tcb = ta - CLOUD_LAPSE_K_PER_KM * c.z_cloud_km
    if c.cloud >= 1.0:
        return _ret(tcb - K0, scalar)
    tcl = np.power(eps_clear(t_dew_c, hour), 0.25) * ta
    if c.cloud <= 0.0:
        return _ret(tcl - K0, scalar)
    t4 = (1.0 - c.cloud) * tcl ** 4 + c.cloud * tcb ** 4
    return _ret(np.sqrt(np.sqrt(t4)) - K0, scalar)


def t_sky_x(c, x):
    scalar = np.ndim(x) == 0
    xa = np.asarray(x, np.float64)
    h = 24.0 * (xa - np.floor(xa))
    return _ret(t_sky(c, day_curve(c, xa), dew_point_state(c, xa), h), scalar)


def net_sky_flux(eps, t_surf_c, t_sky_c):
    """Net radiation from a surface of emissivity eps to the sky, W/m2 (up
    facing, the whole hemisphere is sky)."""
    ts = np.asarray(t_surf_c, np.float64) + K0
    tk = np.asarray(t_sky_c, np.float64) + K0
    return eps * SIGMA * (ts ** 4 - tk ** 4)


# ---------------------------------------------------------------- the ground
def damping_depth(alpha, period_s):
    """Depth at which a temperature wave falls to 1/e: D = sqrt(alpha P / pi)
    (Carslaw and Jaeger; Kusuda). 0.117 m (day) and 2.24 m (year) for 0.5e-6."""
    return math.sqrt(alpha * period_s / math.pi)


def t_ground(c, z_m, x):
    """Ground temperature at depth z_m (m, positive down) at day coordinate x:
    the annual mean plus a year wave and a day wave, each damped by exp(-z/D)
    and lagged by z/D radians."""
    xa = np.asarray(x, np.float64)
    dy = damping_depth(c.alpha_soil, c.year_days * c.day_s)
    dd = damping_depth(c.alpha_soil, c.day_s)
    nw = c.n_warm + (0.5 * c.year_days if c.lat_deg < 0 else 0.0)
    year = c.a_ann_k * math.exp(-z_m / dy) * np.cos(2.0 * math.pi * (xa - nw) / c.year_days - z_m / dy)
    day = a_day_eff(c) * math.exp(-z_m / dd) * np.cos(2.0 * math.pi * (xa - COS_MAX_H / 24.0) - z_m / dd)
    return c.t_ann_c - LAPSE_K_PER_M * c.alt_m + year + day


# ---------------------------------------------------------------- precipitation, frost, ice
def snow_fraction(t_air_c):
    """Share of precipitation that falls as snow: all rain above +3 C, all snow
    below -1 C (u: a linear ramp; Dai 2008 gives a sigmoid around 1.5 C)."""
    return np.clip((3.0 - np.asarray(t_air_c, np.float64)) / 4.0, 0.0, 1.0)


def frost_forms(t_surface_c, t_air_c, t_dew_c):
    """True where a surface is at or below the dew point (the frost point when
    the surface is below 0 C). The state is read from the dew point, which is
    nearly constant over a night."""
    ts = np.asarray(t_surface_c, np.float64)
    rh = np.clip(psat_magnus(t_dew_c) / psat_magnus(t_air_c), 1e-6, 1.0)
    return np.where(ts < 0.0, ts <= frost_point(t_air_c, rh), ts <= np.asarray(t_dew_c, np.float64))


def _rho_v(t_c, below_zero_ice=False):
    t = np.asarray(t_c, np.float64)
    e = np.where(below_zero_ice & (t < 0.0), psat_ice(t), psat_magnus(t)) * 100.0
    return e / (R_VAPOUR * (t + K0))


def deposit_flux(t_s_c, t_air_c, t_dew_c, h_c):
    """Mass flux of vapour onto a surface, kg/(m2 s), signed (negative = it dries):
    h_m (rho_v,air - rho_v,sat(T_s)), h_m = h_c/(rho_air cp Le^(2/3)). Heat given
    to the surface is flux * L_s (2.83 MJ/kg frost, 2.5 MJ/kg dew) (u)."""
    hm = h_c / (RHO_AIR * CP_AIR * LE_TWO_THIRDS)
    rho_air_v = psat_magnus(t_dew_c) * 100.0 / (R_VAPOUR * (np.asarray(t_air_c, np.float64) + K0))
    return hm * (rho_air_v - _rho_v(t_s_c, True))


def deposit_heat(flux, t_s_c):
    """W/m2 given to a surface by the phase change of the deposit."""
    return np.asarray(flux, np.float64) * np.where(np.asarray(t_s_c, np.float64) < 0.0, L_FROST, L_DEW)


def ice_thickness(theta_ks, h_a=None, k_i=K_ICE, rho_i=RHO_ICE, latent=L_ICE):
    """Ice on a still pond (MARKED: dropped when T6 has stefan()). theta is the
    freezing degree-seconds, integral of (T_f - T_air) dt. With no air film,
    Stefan: sqrt(2 k theta/(rho L)). With a film h_a (W/m2K, 15-25 (u)):
    h = (k/h_a)(-1 + sqrt(1 + 2 h_a^2 theta/(rho L k)))."""
    th = np.maximum(np.asarray(theta_ks, np.float64), 0.0)
    if h_a is None:
        return np.sqrt(2.0 * k_i * th / (rho_i * latent))
    return (k_i / h_a) * (-1.0 + np.sqrt(1.0 + 2.0 * h_a * h_a * th / (rho_i * latent * k_i)))


# ---------------------------------------------------------------- the Drive (T5 protocol)
_GL4_X = np.array([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526])
_GL4_W = np.array([0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538])
DRIVE_FIELDS = ("T_out", "T_sky", "T_dew")


class ClimateDrive:
    """A Drive (T5 protocol, duck typed): value(t), mean(t0, t1), kinks(t0, t1),
    pieces(t0, t1, key, max_rel), with t in TICKS. Plain data (a Climate and a
    field name): deep-copies and snapshots exactly. `mean` is analytic for
    T_out (piece by piece); T_sky and T_dew use 4-point Gauss-Legendre on the
    kink-cut pieces (class QUAD)."""

    def __init__(self, climate, field="T_out", clock=None):
        if field not in DRIVE_FIELDS:
            raise ValueError("ClimateDrive field must be one of %s" % (DRIVE_FIELDS,))
        self.climate = climate
        self.field = field
        self.clock = clock if clock is not None else clock_of(climate)

    def _x(self, t):
        return day_coord(self.climate, self.clock, t)

    def value(self, t):
        x = self._x(t)
        if self.field == "T_out":
            return day_curve(self.climate, x)
        if self.field == "T_dew":
            return dew_point_state(self.climate, x)
        return t_sky_x(self.climate, x)

    def kinks(self, t0, t1):
        """Ticks in (t0, t1) where the curve has a corner or a branch change:
        sunrise, sunset, and every day boundary (weather days)."""
        c = self.climate
        x0, x1 = float(self._x(t0)), float(self._x(t1))
        d0, d1 = int(math.floor(x0)) - 1, int(math.floor(x1)) + 1
        days = np.arange(d0, d1 + 1, dtype=np.int64)
        sr, ss = sun_times(c.lat_deg, days + 0.5)
        xs = np.concatenate([days, days + sr / 24.0, days + ss / 24.0])
        xs = np.sort(xs[(xs > x0) & (xs < x1)])
        ticks = self.clock.tick_of(xs) if hasattr(self.clock, "tick_of") else xs
        t0f, t1f = float(t0), float(t1)
        return [float(t) for t in np.atleast_1d(ticks) if t0f < t < t1f]

    def _segments(self, t0, t1):
        pts = [float(t0)] + self.kinks(t0, t1) + [float(t1)]
        return [(a, b) for a, b in zip(pts[:-1], pts[1:]) if b > a]

    def pieces(self, t0, t1, key=None, max_rel=0.5):
        """Cut [t0, t1] at the kinks, then bisect each stretch until key(value)
        (default: the value) moves by at most max_rel inside each part (checked
        on 9 samples). Returns [(t_a, t_b)] in ticks."""
        key = key if key is not None else (lambda v: v)
        # breadth first: every pending stretch of one level is judged in ONE
        # vectorised call (9 samples each), so a long horizon costs a handful of
        # curve evaluations, not one per stretch
        pending = [(a, b, 0) for a, b in self._segments(t0, t1)]
        done = []
        u = np.linspace(0.0, 1.0, 9)
        while pending:
            a = np.array([p[0] for p in pending])
            b = np.array([p[1] for p in pending])
            ts = a[:, None] + (b - a)[:, None] * u[None, :]
            kv = np.asarray(key(self.value(ts.ravel())), np.float64).reshape(ts.shape)
            spread = kv.max(axis=1) - kv.min(axis=1)
            nxt = []
            for (pa, pb, depth), sp in zip(pending, spread):
                if depth >= 40 or float(sp) <= max_rel:
                    done.append((float(pa), float(pb)))
                else:
                    m = 0.5 * (pa + pb)
                    nxt.append((pa, m, depth + 1))
                    nxt.append((m, pb, depth + 1))
            pending = nxt
        return sorted(done)

    def mean(self, t0, t1):
        t0, t1 = float(t0), float(t1)
        if t1 <= t0:
            return float(self.value(t0))
        if self.field != "T_out":
            tot = 0.0
            for a, b in self.pieces(t0, t1, None, 0.5):
                tm, th = 0.5 * (a + b), 0.5 * (b - a)
                tot += th * float(np.sum(_GL4_W * self.value(tm + th * _GL4_X)))
            return tot / (t1 - t0)
        c = self.climate
        tot = 0.0
        for a, b in self._segments(t0, t1):
            xa, xb, xm = (float(self._x(a)), float(self._x(b)), float(self._x(0.5 * (a + b))))
            # one branch inside a segment: locate it at the midpoint, then use
            # the same branch parameters at both ends via the midpoint's cycle
            _, pm = _eval_in_branch(c, xm, np.array([xa, xb]))
            tot += (pm[1] - pm[0]) * c.day_s / TICK_S      # K*days -> K*ticks
        return tot / (t1 - t0)


def _eval_in_branch(c, xm, xs):
    """Antiderivative (K*days) at xs, using the branch and cycle found at xm, so
    that a difference over one kink-free piece is exact."""
    Lm = _locate(c, np.array([xm]))
    xs = np.asarray(xs, np.float64)
    kind = int(Lm["kind"][0])
    if kind == 0:
        return _cos_part(c, xs, True)
    y = float(max(Lm["Y"][0], 1e-9))
    if kind == 1:
        loc = float(Lm["m"][0]) + 24.0 * (xs - xm)
        arg = math.pi * loc / (y + 2.0 * PL_A_H)
        tn, tx = float(Lm["Tn"][0]), float(Lm["Tx"][0])
        return None, (tn * loc - (tx - tn) * (y + 2.0 * PL_A_H) / math.pi * np.cos(arg)) / 24.0
    loc = float(Lm["nn"][0]) + 24.0 * (xs - xm)
    nlen = float(max(Lm["N"][0], 1e-9))
    e = np.exp(-PL_B * loc / nlen)
    tinf, tsun = float(Lm["tinf"][0]), float(Lm["Tsun"][0])
    return None, (tinf * loc - (tsun - tinf) * (nlen / PL_B) * e) / 24.0
