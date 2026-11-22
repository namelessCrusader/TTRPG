"""SMOKE OPTICS (wave 13, T3, critic 10 / realism-map F19): pure functions.

Smoke is 39 times too clear today. materials.SMOKE_EXT = 5.6 per (g/L) of the sim's
"smoke unit", which counts ALL visible products (about 40 times the true soot yield,
SOOT 0.6 for wood against an SFPE yield near 0.015). The standard law reads opacity
from SOOT alone:

    K = sigma_e * C           mass extinction coefficient times soot concentration
    OD/m = K / ln 10           optical density per metre
    V = C_v / K                visibility (Jin): C_v 8 for an illuminated sign,
                               3 for a reflecting one

sigma_e is about 8.7 m2/g for flaming soot (Mulholland and Croce); Jin's C_v values
are an empirical correlation. Both are (u) until tests/data/sourced_numbers.json
marks them sourced. The soot scalar is the smoke unit times SOOT_TRUE = 0.025 g of
true soot per unit (layer_data), which makes K = 8.7 * 0.025 * 1000 = 217.5 per
(g/L) of smoke, 38.8 times the old 5.6.

CO, coughing and the Purser incapacitation model are LEFT AS DATA HOOKS (Hooks):
they retune with the soot scalar but their numbers are not this module's.
"""
import math

from .layer_data import SOOT_TRUE

SIGMA_E = 8.7                   # m2/g, flaming soot (u)
C_V_ILLUMINATED = 8.0           # Jin, illuminated signs (u)
C_V_REFLECTING = 3.0            # Jin, reflecting signs (u)
LN10 = math.log(10.0)
SMOKE_EXT_OLD = 5.6             # materials.SMOKE_EXT, 1/m per (g/L) of smoke unit
# AMBIENT LIGHT AND SMOKE SIGHT (wave 14a refinement, R1). Reappraisal of Jin's visibility
# experiment (Fire Safety J 2025, doi 10.1016/j.firesaf.2025.104573): K = sigma*V is 4.7-9.5
# at 180 lx and 6-11 below 22 lx, for lit signs. Dim light lets the same sign be seen through
# denser smoke, by 8.5/7.1 = 1.2 (band middles). Between the two lights it is log-linear in
# illuminance (u: the paper gives two bands, not a curve). Below 22 lx it stays at 1.2 (u).
LUX_BRIGHT = 180.0              # lx, the paper's high light
LUX_DIM = 22.0                  # lx, the paper's low light
K_DIM_RATIO = 8.5 / 7.1         # middle of 6-11 over middle of 4.7-9.5
LUX_INDOOR = 300.0              # lx, a lit room (marks.luminance's reference): where no sun is simulated
LUX_FLOOR = 0.1                 # lx, a moonlit night (u)
SUN_LUX = 1.1e5                 # lx, overhead sun on a clear day (u, 1e5 to 1.2e5)
HOOK_NAMES = ("co_yield", "cough", "purser", "smell")


def extinction(conc_g_m3, sigma_e=SIGMA_E):
    """K, 1/m, of soot at conc g/m3: K = sigma_e * C."""
    return sigma_e * conc_g_m3


def conc_from_extinction(K, sigma_e=SIGMA_E):
    """Soot concentration, g/m3, for an extinction K, 1/m."""
    return K / sigma_e


def optical_density_per_m(K):
    """Decadic optical density per metre: OD/m = K/ln 10."""
    return K / LN10


def extinction_from_od(od_per_m):
    return od_per_m * LN10


def visibility(K, c_v=C_V_ILLUMINATED):
    """Metres: V = C_v / K (infinite in clear air)."""
    return math.inf if K <= 0.0 else c_v / K


def visibility_reflecting(K):
    return visibility(K, C_V_REFLECTING)


def visibility_from_conc(conc_g_m3, c_v=C_V_ILLUMINATED, sigma_e=SIGMA_E):
    return visibility(extinction(conc_g_m3, sigma_e), c_v)


def soot_conc_from_smoke(smoke_g_per_L, soot_true=SOOT_TRUE):
    """True soot concentration, g/m3, from the sim's smoke scalar (g per litre of
    smoke units): units * 1000 L/m3 * SOOT_TRUE."""
    return smoke_g_per_L * 1000.0 * soot_true


def extinction_from_smoke(smoke_g_per_L, soot_true=SOOT_TRUE, sigma_e=SIGMA_E):
    """K, 1/m, from the sim's smoke scalar (g/L): the new law the sight code uses."""
    return extinction(soot_conc_from_smoke(smoke_g_per_L, soot_true), sigma_e)


def extinction_old(smoke_g_per_L):
    """The old law: SMOKE_EXT * smoke (1/m)."""
    return SMOKE_EXT_OLD * smoke_g_per_L


def clearer_by(soot_true=SOOT_TRUE, sigma_e=SIGMA_E):
    """How much clearer the old law is than the soot law (about 38.8)."""
    return extinction_from_smoke(1.0, soot_true, sigma_e) / SMOKE_EXT_OLD


def sight_ext_per_gl(soot_true=SOOT_TRUE, sigma_e=SIGMA_E):
    """Depth per (g/L) per metre that the sight code adds: K scaled so that the cut
    SEE_TAU lands on Jin's smoke limit SEE_SMOKE_TAU (reflecting objects, C = 3)."""
    from .materials import SEE_SMOKE_TAU, SEE_TAU   # here, so the layer modules stay free of materials
    return extinction_from_smoke(1.0, soot_true, sigma_e) * SEE_TAU / SEE_SMOKE_TAU


def sight_range_m(smoke_g_per_L, lit=False):
    """Metres a uniform smoke of this strength lets a look reach: C / K, with C = 3 for
    a lit-by-reflection object and 8 for a flame (Jin). Infinite in clear air."""
    from .materials import SEE_SMOKE_TAU
    return visibility(extinction_from_smoke(smoke_g_per_L),
                      C_V_ILLUMINATED if lit else SEE_SMOKE_TAU)


def transmittance(K, path_m):
    """Beer-Lambert: exp(-K * path)."""
    return math.exp(-K * path_m)


class Hooks:
    """Data hooks for the soot scalar's other readers (CO yield by burning mode,
    coughing, Purser incapacitation, smell). A hook is a function registered by
    name; calling an unregistered one returns None (the caller keeps its old law).
    Instance state only: no module global is mutated."""

    def __init__(self):
        self.fns = {}

    def register(self, name, fn):
        if name not in HOOK_NAMES:
            raise KeyError(f"unknown smoke hook {name!r}; known: {HOOK_NAMES}")
        self.fns[name] = fn

    def call(self, name, *args, **kw):
        fn = self.fns.get(name)
        return None if fn is None else fn(*args, **kw)

    def has(self, name):
        return name in self.fns


def light_k_ratio(lux):
    """How much denser a smoke a sign can be seen through at this illuminance than at
    180 lx: 1.0 at 180 lx and up, 1.2 at 22 lx and down, log-linear between (Jin
    reappraisal 2025). A look's smoke depth counts 1/ratio as much."""
    if not lux > LUX_DIM:
        return K_DIM_RATIO
    if lux >= LUX_BRIGHT:
        return 1.0
    f = math.log(LUX_BRIGHT / lux) / math.log(LUX_BRIGHT / LUX_DIM)
    return 1.0 + (K_DIM_RATIO - 1.0) * f


def sky_lux(sin_elev, cloud=0.0):
    """Outdoor illuminance, lx, from the sine of the sun's elevation and the cloud
    cover 0..1: SUN_LUX * sin(elev) cut by 1 - 0.75 cloud^3.4 (Kasten-Czeplak, u); under
    the horizon it falls a decade per 2.9 degrees from 400 lx at sunrise (civil twilight,
    -6 deg, is 3.4 lx) down to a moonlit floor. The larger of day and twilight."""
    sin_elev = max(-1.0, min(1.0, float(sin_elev)))
    t = 1.0 - 0.75 * min(1.0, max(0.0, cloud)) ** 3.4
    day = SUN_LUX * sin_elev * t if sin_elev > 0.0 else 0.0
    el = math.degrees(math.asin(sin_elev))
    tw = 400.0 * t * 10.0 ** (min(el, 0.0) / 2.9)
    return max(day, tw, LUX_FLOOR)
