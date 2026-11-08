"""MIXTURES, THE FRONT DOOR (wave 13, T4, law L14).

One import for the world laws and for T5, T6, T7: the mixture quantities of
mixwater, mixvle, mixphase, mixsol and mixsolid, plus the small helpers that make
them cheap to call per cell. Pure numpy, no World; the pool adapter is phase B.

There is NO MixCache and no interning in T4: T1's mix table rows are immutable and
exact, so `by_row` evaluates a function once per DISTINCT row id and gathers; cost
and memory follow the number of distinct rows, not the number of cells.

No function here returns a flag such as "safe" or "cured": thresholds (aw 0.91,
pH 4.6) are data in L4/L9 rows, read elsewhere.
"""
import numpy as np

from . import mixdata as D
from . import mixwater as W
from . import mixvle as V
from . import mixphase as P
from . import mixsol as S
from . import mixsolid as X

# the water functions and the steps, under one roof
psat_water = W.psat_water
tsat_water = W.tsat_water
ln_aw = W.ln_aw
aw = W.aw
boiling_rise = W.boiling_rise
boiling_point = W.boiling_point
freezing_drop = W.freezing_drop
freeze_point = W.freeze_point
osmolality = W.osmolality
tonicity = W.tonicity
osmotic_pressure = W.osmotic_pressure
density = W.density
heat_capacity = W.heat_capacity
coverage = W.coverage
water_for_aw = W.water_for_aw
water_for_boiling_rise = W.water_for_boiling_rise
abv_to_grams = W.abv_to_grams
bubble_point = V.bubble_point
rayleigh = V.rayleigh
boil_step = V.boil_step
phase_step = P.phase_step
phase_assembly = P.phase_assembly
liquid_fraction = P.liquid_fraction
freeze_split = P.freeze_split
dissolved_target = S.dissolved_target
dissolved_split = S.dissolved_split
supersaturation = S.supersaturation
ph = S.ph
henry = S.henry
register_species = D.register_species
eq_pressure = X.eq_pressure
reaction_K = X.reaction_K
affinity = X.affinity
nernst = X.nernst
cell_emf = X.cell_emf
register_system = D.register_system


def mix(**grams):
    """A comp vector (float64, species on the axis) from named grams: mix(h2o=100, nacl=10)."""
    c = np.zeros(D.n_species())
    for name, g in grams.items():
        c[D.INDEX[name]] = g
    return c


def fractions(comp):
    """Mass fractions of a comp array (zero cells stay zero)."""
    c = W.prep(comp)
    tot = c.sum(axis=-1, keepdims=True)
    return np.where(tot > 0.0, c / np.where(tot > 0.0, tot, 1.0), 0.0)


def grams(frac, total):
    """Grams from mass fractions and a total mass."""
    return np.asarray(frac, dtype=float) * np.asarray(total, dtype=float)[..., None]


def split_forms(comp_total, T):
    """(dissolved, crystal) of a total comp at T (C): the solubility target, for
    stand-alone use. In the world the form comes from T1's FORM axis."""
    return S.dissolved_split(comp_total, T)


def by_row(fn, ids, *arrays):
    """Evaluate fn once per DISTINCT row id and gather. fn(unique_ids, *picked) must
    return an array (or a tuple of arrays) with the unique ids on the first axis.
    `arrays` are per-cell values that are constant within a row (a quantised T, a
    pressure); the first cell of each row supplies them. Scalars pass through."""
    ids = np.asarray(ids)
    flat = ids.reshape(-1)
    u, first, inv = np.unique(flat, return_index=True, return_inverse=True)
    picked = []
    for a in arrays:
        a_ = np.asarray(a)
        picked.append(a_.reshape(-1)[first] if a_.ndim and a_.size == flat.size else a)
    out = fn(u, *picked)

    def gather(o):
        o = np.asarray(o)
        return o[inv.reshape(-1)].reshape(ids.shape + o.shape[1:])

    return tuple(gather(o) for o in out) if isinstance(out, tuple) else gather(out)


# -- columns for T1's pool (registered by name) and the boiling prefilter -----------------------
def _t_solidus(comp):
    c = W.prep(comp)
    sysm = P.system_for(c)
    return np.full(c.shape[:-1], np.nan) if sysm is None else P.solidus_liquidus(c, sysm)[0]


def _t_liquidus(comp):
    c = W.prep(comp)
    sysm = P.system_for(c)
    return np.full(c.shape[:-1], np.nan) if sysm is None else P.solidus_liquidus(c, sysm)[1]


def t_boil_1atm(comp, P_pa=D.P_ATM):
    """Boiling point at P (default 1 atm), C: the bubble point if ethanol or
    methanol is present, else the water-activity boiling point. Used as Tb_lo."""
    c = W.prep(comp)
    vol = (c[..., D.INDEX["ethanol"]] > 0.0) | (c[..., D.INDEX["methanol"]] > 0.0)
    tb = W.boiling_point(c, P_pa)
    if np.any(vol):
        tv = V.bubble_point(c, P_pa)[0]
        tb = np.where(vol, tv, tb)
    return tb


DERIVED = {
    "T_solidus": _t_solidus,
    "T_liquidus": _t_liquidus,
    "T_boil_1atm": t_boil_1atm,
    "aw_1atm": lambda comp: W.aw(comp, 25.0),
}


def register_derived(name, fn):
    """Register a derived column (a function of a comp array) by name."""
    DERIVED[name] = fn


def derived(name, comp):
    """A registered derived column; 'c_sat_<species>' is the saturation grams of
    that species at 25 C in the cell's solvent."""
    if name in DERIVED:
        return DERIVED[name](comp)
    if name.startswith("c_sat_"):
        return S.saturation_grams(comp, 25.0)[..., D.INDEX[name[6:]]]
    raise KeyError(name)


def boil_prefilter(ids, T, tb_lo, margin=0.5):
    """Mask of cells worth a boil evaluation: T > Tb_lo[id] - margin, by gather. Cells
    far below their row's lower bound (a brine at 25 C) cost one gather and one compare."""
    return np.asarray(T) > (np.asarray(tb_lo)[ids] - margin)


# -- the latent-energy books of a world (wave 13b, C2) --------------------------------------------
# T4 owns these four terms (ledgers.BOOKS); the world laws of heat.py reach them ONLY
# through the functions below, so that this module stays their one writer. Each term is a
# plain float on the world, made on first use (an unwritten term reads 0.0: `books()`).
# Signs (the lane C2 report restates them):
#   boiled_J    joules the steam took away (latent heat of boiling), + = left the cell
#   melted_J    joules the melting of a mix took from the heat, + = absorbed
#   frozen_J    joules the freezing of a mix gave back to the heat, + = released
#   dissolved_J signed joules of the heat of solution: + = taken from the heat (a salt
#               dissolving), - = given back (a salt crystallising out)
# Closure for a mixed cell:  d(E of the cells) + boiled_J + melted_J - frozen_J + dissolved_J
# = heat that came in (minus what `shed` took).
def book_boiled(w, joules):
    """Add `joules` (signed float) to the boiled_J book of world `w`."""
    w.boiled_J = getattr(w, "boiled_J", 0.0) + float(joules)


def book_melted(w, joules):
    """Add `joules` to the melted_J book of world `w`."""
    w.melted_J = getattr(w, "melted_J", 0.0) + float(joules)


def book_frozen(w, joules):
    """Add `joules` to the frozen_J book of world `w`."""
    w.frozen_J = getattr(w, "frozen_J", 0.0) + float(joules)


def book_dissolved(w, joules):
    """Add `joules` to the dissolved_J book of world `w`."""
    w.dissolved_J = getattr(w, "dissolved_J", 0.0) + float(joules)
