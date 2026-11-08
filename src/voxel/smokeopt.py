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
