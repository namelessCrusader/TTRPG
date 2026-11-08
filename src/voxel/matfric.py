"""FRICTION PER PAIR AND PER FILM (wave 13, T9).

Coulomb's law is unchanged: a body slides when the tangential force reaches
mu_s N, and once sliding it feels mu_k N (mu_k <= mu_s). What changes is the
number. Today it is 0.40 for every pair of materials, ice and grease included
(materials.FRICTION). Here it depends on the pair of surfaces and on what is
between them (nothing, water, oil, ice).

Stand-alone: numpy only. The class of each material comes from matrec.py when
it can be imported (matrec rows name a `mu_class`); the legacy ids are also
written out here as a stand-in, and a cross test compares the two.

CLASSES a material names one: rock, wood, steel, softmetal, glass, skin, fibre,
granular, ice.
FILMS: dry (0), wet (1: water, blood, acid), oiled (2), iced (3: a water film at
or below 0 C).

THE TABLE. Dry values are listed per class pair (static, kinetic); any pair not
listed uses `mu_default` (the geometric mean of the two classes' own dry values,
clamped to [0.03, 1.2]). A film REPLACES the pair's values (a film changes the
mechanism, it is not a multiplier). A film value never exceeds the film before
it, pair by pair, so the order iced <= oiled <= wet <= dry holds for EVERY pair,
listed or not. That is done once, when the table is built, with a running
minimum. The one exception is deliberate and explained at `_iced`: the iced
value rises as it gets colder and, at very low temperatures, may pass the
oiled value; it never passes the dry value.

Sources: T = Engineering ToolBox "Friction coefficients"; B = Byerlee 1978
(rock); BT = Bowden and Tabor 1950 (oil); BH = Bowden and Hughes 1939 (ice);
(u) = independent but unverified.
"""
import math

import numpy as np

try:
    from . import matrec as _matrec
except ImportError:
    _matrec = None

CLASSES = ("rock", "wood", "steel", "softmetal", "glass", "skin", "fibre",
           "granular", "ice")
_IDX = {c: i for i, c in enumerate(CLASSES)}
NC = len(CLASSES)

DRY, WET, OILED, ICED = 0, 1, 2, 3
FILMS = ("dry", "wet", "oiled", "iced")

FRICTION = 0.40                 # the old single number (materials.FRICTION), kept importable
MU_LO, MU_HI = 0.03, 1.2        # the clamp of a default dry value
WET_FACTOR = 0.7                # (u) an unlisted wet pair is min(dry, 0.7 dry)
OILED_PAIR = (0.10, 0.08)       # BT: boundary lubrication, mineral oil on steel 0.1-0.15
ICED_WARM = (0.05, 0.03)        # BH (u): at -5 C and warmer
ICED_COLD = (0.12, 0.10)        # BH (u): at -20 C and colder
ICED_T_WARM, ICED_T_COLD = -5.0, -20.0

# fluid ids, as materials.py has them (a test compares); film_of needs only these
_WATER, _OIL, _ACID, _WEAK_ACID, _BLOOD = 1, 2, 3, 4, 8
_MOLTEN = (5, 6, 7)

# the legacy material ids -> class: stand-in for matrec (cross-tested)
_STANDIN_CLASS = ("granular", "wood", "rock", "steel", "skin", "granular", "glass",
                  "fibre", "granular", "softmetal", "softmetal", "fibre")

# dry (static, kinetic), unordered class pairs
_DRY = {
    ("wood", "wood"): (0.40, 0.30),            # T: static 0.25-0.5, kinetic 0.2
    ("wood", "rock"): (0.55, 0.40),            # (u)
    ("wood", "steel"): (0.45, 0.30),           # T: 0.2-0.6
    ("wood", "glass"): (0.35, 0.25),           # (u)
    ("rock", "rock"): (0.70, 0.60),            # B: 0.85 below 200 MPa normal stress, 0.6 above
    ("rock", "steel"): (0.55, 0.40),           # (u)
    ("steel", "steel"): (0.70, 0.50),          # T: 0.74 / 0.57; wide spread with the oxide film (u)
    ("glass", "glass"): (0.90, 0.40),          # T
    ("glass", "rock"): (0.60, 0.40),           # (u)
    ("glass", "steel"): (0.60, 0.40),          # (u)
    ("skin", "wood"): (0.45, 0.35),            # (u) Derler and Gerhardt 2012: 0.3-0.9 for dry skin
    ("skin", "rock"): (0.60, 0.50),            # (u)
    ("skin", "steel"): (0.50, 0.40),           # (u)
    ("skin", "skin"): (0.50, 0.40),            # (u)
    ("fibre", "wood"): (0.50, 0.40),           # (u)
    ("fibre", "rock"): (0.55, 0.45),           # (u)
    ("fibre", "fibre"): (0.50, 0.40),          # (u)
    ("ice", "ice"): (0.10, 0.03),              # T 0.05-0.15 at 0 C (u)
}
_SOFT = (0.60, 0.50)            # (u) softmetal (tin, lead) with any class
_GRAN = (0.50, 0.45)            # (u) granular (ash, char) with any class; a heap's angle really lives in L5g
_WET = {
    ("wood", "wood"): (0.30, 0.20),            # T: wet 0.2
    ("rock", "rock"): (0.60, 0.50),            # (u)
    ("steel", "steel"): (0.50, 0.35),          # (u)
    ("glass", "glass"): (0.35, 0.20),          # (u)
    ("skin", "steel"): (0.40, 0.30),           # (u) skin on smooth surfaces: a stand-in
    ("skin", "glass"): (0.40, 0.30),           # (u)
}


def _pair_key(a, b):
    return (a, b) if _IDX[a] <= _IDX[b] else (b, a)


# the tables above are written in reading order; key them in class order
_DRY = {_pair_key(*k): v for k, v in _DRY.items()}
_WET = {_pair_key(*k): v for k, v in _WET.items()}


def _listed_dry(a, b):
    k = _pair_key(a, b)
    if k in _DRY:
        return _DRY[k]
    if "softmetal" in k:
        return _SOFT
    if "granular" in k:
        return _GRAN
    return None


def _own(c):
    """A class's own dry pair (class on itself); the base of the default."""
    if c == "softmetal":
        return _SOFT
    if c == "granular":
        return _GRAN
    return _DRY[(c, c)]


def mu_default(ca, cb, film=DRY):
    """The value for a class pair or film nobody listed.
    dry: the geometric mean of the two classes' own dry values, clamped to
    [0.03, 1.2]; wet: min(dry, 0.7 dry); oiled and iced: the any-pair values
    (capped by the pair's dry and wet values). Returns (static, kinetic)."""
    oa, ob = _own(ca), _own(cb)
    dry = tuple(min(MU_HI, max(MU_LO, math.sqrt(oa[i] * ob[i]))) for i in (0, 1))
    if film == DRY:
        return dry
    if film == WET:
        return tuple(min(d, WET_FACTOR * d) for d in dry)
    if film == OILED:
        return OILED_PAIR
    return ICED_WARM


def _build(flat=None):
    """The four tables [NC, NC, 4] of static and kinetic values, float64."""
    S = np.zeros((NC, NC, 4))
    K = np.zeros((NC, NC, 4))
    for a in CLASSES:
        for b in CLASSES:
            ia, ib = _IDX[a], _IDX[b]
            if flat is not None:
                S[ia, ib, :] = flat
                K[ia, ib, :] = flat
                continue
            dry = _listed_dry(a, b) or mu_default(a, b, DRY)
            wet = _WET.get(_pair_key(a, b)) or mu_default(a, b, WET)
            oil = OILED_PAIR
            ice = ICED_WARM
            prev = dry
            vals = [dry]
            for v in (wet, oil, ice):
                v = (min(v[0], prev[0]), min(v[1], prev[1]))   # the film order, by construction
                vals.append(v)
                prev = v
            for f in range(4):
                S[ia, ib, f] = vals[f][0]
                K[ia, ib, f] = vals[f][1]
    return S, K


MU_S64, MU_K64 = _build()
MU_S = MU_S64.astype(np.float32)           # the spec's float32 tables
MU_K = MU_K64.astype(np.float32)
FLAT_S, FLAT_K = _build(flat=FRICTION)     # the W2 step: every entry 0.40 / 0.40, golden SAME


def _class_table():
    """Class index by material id, padded to 64 ids: the matrec rows, then
    ICE (12, the reserved id) as "ice", then "granular"."""
    t = np.full(64, _IDX["granular"], np.int8)
    names = ([r.mu_class for r in _matrec.BASE.by_id_list] if _matrec is not None
             else list(_STANDIN_CLASS))
    for i, c in enumerate(names):
        t[i] = _IDX[c]
    if len(names) <= 12:
        t[12] = _IDX["ice"]
    return t


MAT_CLASS = _class_table()


def _class_idx(mat, mat_class=None):
    """Class index (array) for material ids."""
    return (MAT_CLASS if mat_class is None else mat_class)[np.asarray(mat)]


def _iced(T_C, ia, ib, dry_s, dry_k):
    """The iced pair at temperature T_C: mu_s 0.05 at -5 C and warmer, rising
    linearly to 0.12 at -20 C and colder (kinetic 0.03 to 0.10). T_C None reads
    the warm end, so the film order holds for the plain table. The rise with
    cold is real (ice at -20 C is much grippier than a wet film at 0 C), and
    it may pass the OILED value (0.10) below about -17 C; it is held at the
    pair's own dry value."""
    if T_C is None:
        return None
    f = np.clip((np.asarray(T_C, float) - ICED_T_WARM) / (ICED_T_COLD - ICED_T_WARM), 0.0, 1.0)
    s = ICED_WARM[0] + (ICED_COLD[0] - ICED_WARM[0]) * f
    k = ICED_WARM[1] + (ICED_COLD[1] - ICED_WARM[1]) * f
    return np.minimum(s, dry_s), np.minimum(k, dry_k)


def mu_class_pair(ca, cb, film=DRY, kinetic=True, T_C=None, table=None):
    """Friction for class INDEXES (scalars or arrays), the plain gather."""
    S, K = (MU_S64, MU_K64) if table is None else table
    ca, cb = np.asarray(ca), np.asarray(cb)
    film_a = np.asarray(film)
    base = (K if kinetic else S)[ca, cb, film_a]
    if T_C is not None:
        ice = film_a == ICED
        if np.any(ice):
            dry_s, dry_k = S[ca, cb, DRY], K[ca, cb, DRY]
            si, ki = _iced(T_C, ca, cb, dry_s, dry_k)
            base = np.where(ice, ki if kinetic else si, base)
    return float(base) if np.ndim(base) == 0 else base


def mu_pair(ma, mb, film=DRY, kinetic=True, T_C=None, table=None, mat_class=None):
    """Friction between material ids `ma` and `mb` (scalars or arrays; they
    broadcast). kinetic=False gives the static value that STARTS a slide.
    `T_C` is the temperature of the contact and matters only for an iced film.
    `table=(FLAT_S, FLAT_K)` gives the old flat 0.40 for the W2 step.
    `mat_class` is an alternative id -> class index array (an ingested table)."""
    return mu_class_pair(_class_idx(ma, mat_class), _class_idx(mb, mat_class),
                         film, kinetic, T_C, table)


def film_of(fluid_id, T_C=None):
    """The film a fluid makes on a surface: water at or below 0 C is ICED, oil
    is OILED, blood and acid are WET, no fluid is DRY. A molten metal is not a
    lubricating film and reads DRY (u); any other liquid reads WET."""
    f = np.asarray(fluid_id)
    cold = np.zeros(f.shape, bool) if T_C is None else (np.asarray(T_C, float) <= 0.0)
    out = np.full(f.shape, WET, np.int8)
    out = np.where(f == _OIL, OILED, out)
    out = np.where((f == _WATER) & cold, ICED, out)
    out = np.where((f == 0) | np.isin(f, _MOLTEN), DRY, out)
    return int(out) if out.ndim == 0 else out.astype(np.int8)


def capstan(mu, wraps_rad):
    """Rope round a post: T_load / T_hold = exp(mu theta). Three turns at
    mu 0.4 hold 1880 times."""
    r = np.exp(np.asarray(mu, float) * np.asarray(wraps_rad, float))
    return float(r) if r.ndim == 0 else r


def slip_at(ma, mb, film, normal_N, tangential_N, T_C=None):
    """True when the tangential force reaches the static limit mu_s N. It flips
    exactly at tangential == mu_s * normal."""
    mu_s = mu_pair(ma, mb, film, kinetic=False, T_C=T_C)
    r = np.asarray(tangential_N) >= mu_s * np.asarray(normal_N)
    return bool(r) if r.ndim == 0 else r
