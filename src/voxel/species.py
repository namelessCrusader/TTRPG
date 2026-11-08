"""THE SPECIES REGISTRY (wave 13, T1): what substances there ARE.

One table of pure substances. A species has TWO slots for each number:
  engine  what the sim runs on. For species 0-16 it is READ from materials.py
          (SOLID, FLUID, MELT, BOIL, LATENT, STOCK), never retyped, so a fuel-row
          edit in materials.py flows through. For the seed set an empty engine
          slot means "same as ref".
  ref     the real value, with a source tag, or (u) = unverified.
Tests of physics rules run on `ref` against handbook numbers; tests of sim
behaviour run on `engine`.

This file also owns the ONE copy of the component KEY: key = species << 10 |
form << 8 | size class. mixtable.py and the layer code import pack_key and
unpack_key from here and write no copy of their own.

Forms: DISSOLVED (a liquid is the solvent dissolved in itself), CRYSTAL, GRAIN,
BOUND (gas is the L2b vector, not a form). Size class sz: 0 means "no size"; for
sz >= 1 the particle diameter is d = 1e-6 m * 2 ** (sz / 8).

Id blocks: species 0-16 legacy, 17-63 seed, 64-95 gases, 96+ packs. Mat ids 0-11
legacy, 12 ICE, 13-31 new skeleton mats, padded to 64. Fluid ids 0-8 legacy, 9
up new, padded to 64. Table rows: solids 0-63 (row r is mat r), fluids 64-127
(row 64 + f is fluid f), mixes 128 up.

Import direction, one way: materials -> species_seed (at the end of materials.py,
phase B); species imports materials, species_seed and legacy_tables; this file
imports no World and nothing from the mix table.
"""
from dataclasses import dataclass

import numpy as np

from . import materials as mat
from . import species_seed as seed
from .legacy_tables import build_tables
from .species_seed import (BOUND, CRYSTAL, DISSOLVED, FLUID_ROWS, FORM_NAMES, GAS_HI,
                           GAS_LO, GRAIN, MAT_ICE, MIX_BASE, NFLUID_PAD, NFORMS,
                           NMAT_PAD, PACK_LO, SOLID_ROWS, SPECIES_SEED_END)

__all__ = ["DISSOLVED", "CRYSTAL", "GRAIN", "BOUND", "FORM_NAMES", "NFORMS",
           "pack_key", "unpack_key", "size_to_m", "m_to_size", "Species", "SPECIES",
           "BY_NAME", "get", "MatSlot", "FluidSlot", "MATS", "FLUIDS", "pure_row",
           "pure_rows", "id_prefix", "prefix_problem", "legacy_inputs",
           "seed_mat_rows", "seed_fluid_rows", "padded_tables", "bulk_density",
           "form_porosity", "packed_k", "bulk_conductivity", "NMAT_PAD",
           "NFLUID_PAD", "MAT_ICE", "MIX_BASE", "GAS_LO", "GAS_HI", "PACK_LO"]

KEY_FORM_SHIFT, KEY_SPECIES_SHIFT = seed.KEY_FORM_SHIFT, seed.KEY_SPECIES_SHIFT
SZ_MAX = 255
SPECIES_MAX = 0xFFFF


# ---- the component key -----------------------------------------------------
def pack_key(species, form, sz=0):
    """key (uint32) = species << 10 | form << 8 | sz. Scalars give an int, arrays
    a uint32 array. Species is uint16, form 0..3, sz 0..255; out of range raises."""
    if np.ndim(species) == 0 and np.ndim(form) == 0 and np.ndim(sz) == 0:
        sp, fo, sz = int(species), int(form), int(sz)
        if not (0 <= sp <= SPECIES_MAX and 0 <= fo < NFORMS and 0 <= sz <= SZ_MAX):
            raise ValueError(f"key part out of range: {sp}, {fo}, {sz}")
        return sp << KEY_SPECIES_SHIFT | fo << KEY_FORM_SHIFT | sz
    sp, fo, sz = (np.asarray(a, np.int64) for a in (species, form, sz))
    if (sp.min(initial=0) < 0 or sp.max(initial=0) > SPECIES_MAX or fo.min(initial=0) < 0
            or fo.max(initial=0) >= NFORMS or sz.min(initial=0) < 0
            or sz.max(initial=0) > SZ_MAX):
        raise ValueError("key part out of range")
    return (sp << KEY_SPECIES_SHIFT | fo << KEY_FORM_SHIFT | sz).astype(np.uint32)


def unpack_key(key):
    """(species, form, sz). Scalars give ints, arrays give uint16, uint8, uint8."""
    if np.ndim(key) == 0:
        k = int(key)
        return k >> KEY_SPECIES_SHIFT, (k >> KEY_FORM_SHIFT) & 3, k & SZ_MAX
    k = np.asarray(key, np.uint32)
    return ((k >> KEY_SPECIES_SHIFT).astype(np.uint16),
            ((k >> KEY_FORM_SHIFT) & 3).astype(np.uint8), (k & SZ_MAX).astype(np.uint8))


def size_to_m(sz):
    """Particle diameter in metres of size class sz (0 = no size gives 0.0)."""
    d = seed.SZ_D0_M * np.exp2(np.asarray(sz, np.float64) / seed.SZ_PER_OCTAVE)
    d = np.where(np.asarray(sz) == 0, 0.0, d)
    return float(d) if np.ndim(sz) == 0 else d


def m_to_size(d):
    """The size class nearest (in log) to a diameter in metres, clipped to 1..255."""
    s = np.rint(seed.SZ_PER_OCTAVE * np.log2(np.asarray(d, np.float64) / seed.SZ_D0_M))
    s = np.clip(s, 1, SZ_MAX).astype(np.int64)
    return int(s) if np.ndim(d) == 0 else s


# ---- the registry ----------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Species:
    id: int
    name: str
    formula: str
    ref: dict
    engine: dict
    src: dict
    unverified: tuple
    solid_states: dict          # form -> mat id (BOUND and CRYSTAL use the BOUND mat)
    liquid: object              # fl id or None
    gas: object                 # gas slot or None (the GAS lane fills it)
    skeleton: frozenset
    like: int
    fluid_like: object
    composite: bool
    grain_sz: int
    decomposition: object = None
    note: str = ""
    legacy: bool = False

    @property
    def M(self):
        return self.ref.get("M")

    def get(self, field_name, slot="engine", default=None):
        """A number. slot "engine" falls back to ref when the engine slot has no
        entry (a seed species runs on its ref numbers); slot "ref" is the real
        value only."""
        if slot == "engine" and field_name in self.engine:
            return self.engine[field_name]
        return self.ref.get(field_name, default)

    def has(self, field_name, slot="engine"):
        return self.get(field_name, slot, None) is not None

    @property
    def states(self):
        return {"solid": dict(self.solid_states), "liquid": self.liquid, "gas": self.gas}


def _legacy_engine(sid):
    """Engine numbers of species 0-16, read from materials.py."""
    e = {}
    if 1 <= sid <= 11:
        rho, c, k, ign, burn, rate = mat.SOLID[sid]
        e.update(rho_s=rho, c_s=c, k_s=k, hcomb=burn, burn_rate=rate,
                 emiss_s=mat.EMISS[sid])
        if ign is not None:
            e["ign"] = ign
        if sid in mat.MELT:
            e["Tm"], _fl, e["Lf"] = mat.MELT[sid]
    fl = {3: mat.MIRON, 9: mat.MTIN, 10: mat.MLEAD, 12: mat.WATER, 13: mat.OIL,
          14: mat.ACID, 15: mat.WEAK_ACID, 16: mat.BLOOD}.get(sid)
    if fl is not None:
        rho, c, spread, ign, burn, rate, h = mat.FLUID[fl]
        e.update(rho_l=rho * 1000.0, c_l=c, spread=spread, h_wet=h,
                 k_l=mat.FLUID_K[fl], emiss_l=mat.EMISS_FLUID[fl])
        if sid >= 12:                       # the species IS the fluid
            e.update(hcomb=burn, burn_rate=rate)
            if ign is not None:
                e["ign"] = ign
        if fl in mat.STOCK:
            e["stock"] = mat.STOCK[fl]
    if sid == 12:
        e.update(Tb=mat.BOIL, Lv=mat.LATENT)
    return e


def _build():
    species = [None] * SPECIES_SEED_END
    solid_of = {}                    # species id -> {form: mat}
    for m, (sid, form, _sz) in seed.MAT_SLOTS.items():
        solid_of.setdefault(sid, {})[BOUND if form == CRYSTAL else form] = m
    for sid in range(1, 12):
        solid_of.setdefault(sid, {})[BOUND] = sid
    liquid_of = {sid: f for f, sid in seed.LEGACY_FLUID_SPECIES.items() if f}
    liquid_of.update({sid: f for f, sid in seed.FLUID_SLOTS.items()})
    # the legacy fluid species 12-16 keep their own fl; 3, 9, 10 are metals
    for r in seed.LEGACY + seed.SEED:
        sid = r["id"]
        legacy = sid < seed.SPECIES_LEGACY_END
        engine = dict(_legacy_engine(sid)) if legacy else dict(r["engine"])
        src = dict(r["src"])
        for k in engine:
            if legacy:
                src.setdefault(k, "repo:materials.py")
        unverified = tuple(f for f in sorted(set(r["ref"]) | set(engine))
                           if f not in src)
        sol = dict(r["states"].get("solid", {})) if not legacy else {}
        sol.update(solid_of.get(sid, {}))
        liq = liquid_of.get(sid, r["states"].get("liquid"))
        if sid == 0:
            sol, liq = {}, None
        like = r["like"]
        species[sid] = Species(
            id=sid, name=r["name"], formula=r["formula"], ref=dict(r["ref"]),
            engine=engine, src=src, unverified=unverified, solid_states=sol,
            liquid=liq, gas=None, skeleton=frozenset(r["skeleton"]), like=like,
            fluid_like=r["fluid_like"], composite=r["composite"],
            grain_sz=r["grain_sz"], decomposition=r["decomposition"],
            note=r["note"], legacy=legacy)
    return tuple(species)


SPECIES = _build()
BY_NAME = {s.name: s.id for s in SPECIES if s is not None}


def get(x):
    """A species by id or name; raises KeyError for an unused id."""
    sid = BY_NAME[x] if isinstance(x, str) else int(x)
    s = SPECIES[sid] if 0 <= sid < len(SPECIES) else None
    if s is None:
        raise KeyError(x)
    return s


# ---- mat ids and fluid ids -------------------------------------------------
@dataclass(frozen=True, eq=False)
class MatSlot:
    id: int
    name: str
    species: object             # species id, or None for an unlisted padded id
    form: int
    sz: int
    like: int                   # the legacy mat whose mechanical rows stand in


@dataclass(frozen=True, eq=False)
class FluidSlot:
    id: int
    name: str
    species: object
    like: int                   # the legacy fluid whose miscibility rows stand in


def _mats():
    out = []
    for m in range(NMAT_PAD):
        if m < 12:
            out.append(MatSlot(m, SPECIES[m].name, m, BOUND, 0, m))
        elif m in seed.MAT_SLOTS:
            sid, form, sz = seed.MAT_SLOTS[m]
            out.append(MatSlot(m, seed.MAT_NAMES[m], sid, form, sz, seed.MAT_LIKE[m]))
        else:
            out.append(MatSlot(m, f"pad_{m}", None, BOUND, 0, 0))
    out[0] = MatSlot(0, "air", 0, BOUND, 0, 0)
    return tuple(out)


def _fluids():
    out = []
    for f in range(NFLUID_PAD):
        if f in seed.LEGACY_FLUID_SPECIES:
            out.append(FluidSlot(f, seed.FLUID_NAMES[f], seed.LEGACY_FLUID_SPECIES[f], f))
        elif f in seed.FLUID_SLOTS:
            out.append(FluidSlot(f, seed.FLUID_NAMES[f], seed.FLUID_SLOTS[f],
                                 seed.FLUID_LIKE[f]))
        else:
            out.append(FluidSlot(f, f"pad_{f}", None, 0))
    return tuple(out)


MATS = _mats()
FLUIDS = _fluids()


# ---- the pure rows of the mix table ----------------------------------------
def pure_row(r):
    """The composition of pure table row r (0..127) as a tuple of
    (species, form, sz, fraction); () is the empty row. ACID and WEAK_ACID are
    the STOCK composition (potency 1), read from materials.STOCK. Rows 128 up are
    user mixes and raise."""
    if not 0 <= r < MIX_BASE:
        raise ValueError(f"row {r} is not a pure row")
    if r < SOLID_ROWS[1]:
        slot = MATS[r]
        if slot.species is None:                  # an unlisted pad copies its like
            return pure_row(slot.like) if slot.like != r else ()
        return () if r == 0 else ((slot.species, slot.form, slot.sz, 1.0),)
    f = r - FLUID_ROWS[0]
    slot = FLUIDS[f]
    if slot.species is None:
        return pure_row(FLUID_ROWS[0] + slot.like) if slot.like != f else ()
    if f == 0:
        return ()
    if f in mat.STOCK:
        w = mat.STOCK[f]
        return ((slot.species, DISSOLVED, 0, w), (BY_NAME["water"], DISSOLVED, 0, 1.0 - w))
    return ((slot.species, DISSOLVED, 0, 1.0),)


def pure_rows():
    """{row number: composition} for rows 0..127."""
    return {r: pure_row(r) for r in range(MIX_BASE)}


# ---- the pinned id prefix --------------------------------------------------
def id_prefix(n=seed.SPECIES_LEGACY_END):
    """[[id, name], ...] for species 0..n-1: what tests/data/species_ids.json pins."""
    return [[s.id, s.name] for s in SPECIES[:n]]


def prefix_problem(pinned, current=None):
    """None if `current` (default: this registry) starts with the pinned list;
    otherwise a sentence naming the first id that moved. Appending passes; a
    reorder, rename or removal fails."""
    current = id_prefix(max(len(pinned), seed.SPECIES_LEGACY_END)) if current is None \
        else [list(p) for p in current]
    for i, (pid, name) in enumerate(pinned):
        if i >= len(current):
            return f"id {pid} ({name}) was removed"
        cid, cname = current[i]
        if (cid, cname) != (pid, name):
            return f"id {pid} was {name}, now {cid} {cname}"
    return None


# ---- the legacy tables, padded ----------------------------------------------
def legacy_inputs():
    """The legacy dicts of materials.py, in the shape legacy_tables.build_tables
    takes. (The oracle test passes the frozen base18 copy instead.)"""
    return {"SOLID": mat.SOLID, "FLUID": mat.FLUID, "EMISS": mat.EMISS,
            "EMISS_FLUID": mat.EMISS_FLUID, "FLUID_K": mat.FLUID_K,
            "MELT": mat.MELT, "BOIL": mat.BOIL, "LATENT": mat.LATENT,
            "SKIN_T": mat.SKIN_T, "NMAT": mat.NMAT, "FLIDS": mat._FLIDS}


def form_porosity(species, form, slot="engine"):
    """Porosity of the PACKED form of a species: the void share of a bed of its
    grains. 0.0 for any form that is not packed grain (rock, crystal, dissolved) or
    for a species with no `por_grain` number (humus carries its packing inside a bulk
    rho_s already, and is left alone)."""
    if form != GRAIN:
        return 0.0
    return float(get(species).get("por_grain", slot, 0.0) or 0.0)


def bulk_density(species, form, slot="engine"):
    """Density in g/L of one cell of the species in this form, pores included (None
    if the species has no solid density in this slot).

    TWO DENSITIES, two readers. `rho_s` is the PARTICLE density of the substance
    (quartz 2650): the species itself, the BOUND form (a rock has no pores), and
    every law about one grain (settling, Stokes, sorting) read that. This function
    is the BULK density of a packed bed, rho_s x (1 - por_grain) (sand 1590): the
    cell table (seed_mat_rows -> _DENS_ARR) reads it, because a voxel of sand is a
    heap with air between the grains, and the grams in the voxel are the bed's."""
    rho = get(species).get("rho_s", slot)
    return None if rho is None else rho * (1.0 - form_porosity(species, form, slot))


def packed_k(k_solid, k_gas, porosity):
    """Conductivity of a packed bed of spherical grains (W/m/K) from the solid, the
    pore gas and the porosity: the Zehner-Schlunder unit-cell rule (Zehner and
    Schlunder 1970; VDI Heat Atlas, section D6), the standard effective-medium rule
    for a packed bed. kappa = k_gas / k_solid, B = 1.25 ((1 - e) / e)^(10/9),
        k / k_gas = 1 - s + s * 2/(1 - B kappa) *
            [ (1 - kappa) B / (1 - B kappa)^2 * ln(1 / (B kappa)) - (B + 1)/2
              - (B - 1)/(1 - B kappa) ],     s = sqrt(1 - e).
    Why not a continuous-solid rule (Bruggeman, 3 W/m/K for sand): grains touch at
    points, so the heat goes through the gas in the gaps and a little through the
    contacts; the Maxwell rule with the gas as the matrix (0.14) is the lowest the
    bed can be. For a solid at least 10 times the gas and 0.25 <= e <= 0.6 (a sweep of
    400000 draws, tests/test_species.py) the result lies between those two (the
    Hashin-Shtrikman bounds); a solid closer to the gas than that is not a bed of
    point contacts and the rule is not meant for it. e = 0 gives the solid, e = 1 the
    gas. Arrays or scalars. The one removable point B kappa = 1 (0/0 in the
    formula) is taken from points either side of it (error about 1e-10)."""
    ks, kg, e = (np.asarray(x, np.float64) for x in (k_solid, k_gas, porosity))
    if np.any(~(ks > 0)) or np.any(~(kg > 0)) or np.any(~((e >= 0) & (e <= 1))):
        raise ValueError("packed_k needs k_solid > 0, k_gas > 0 and 0 <= porosity <= 1")
    ks, kg, e = np.broadcast_arrays(ks, kg, e)
    inner = (e > 0) & (e < 1)
    e_ = np.where(inner, e, 0.5)

    def zs(kappa):
        B = 1.25 * ((1.0 - e_) / e_) ** (10.0 / 9.0)
        d = 1.0 - B * kappa
        s = np.sqrt(1.0 - e_)
        t = 2.0 / d * ((1.0 - kappa) * B / d ** 2 * np.log(1.0 / (B * kappa))
                       - (B + 1.0) / 2.0 - (B - 1.0) / d)
        return 1.0 - s + s * t, B
    kap = kg / ks
    with np.errstate(divide="ignore", invalid="ignore"):
        r, B = zs(kap)
        near = np.abs(1.0 - B * kap) < 1e-3
        # the bracket cancels like 1/d^3 near the point (d = 1 - B kappa), so a tiny
        # step would only lose digits: take the mean of two points either side at
        # a relative step h, for h and 2h, and Richardson them (error h^4, about 1e-10)
        def pair(h):
            return 0.5 * (zs(kap * (1.0 + h))[0] + zs(kap * (1.0 - h))[0])
        r2 = (4.0 * pair(3e-3) - pair(6e-3)) / 3.0
    r = np.where(near, r2, r)
    out = np.where(inner, kg * r, np.where(e <= 0, ks, kg))
    return float(out) if out.ndim == 0 else out


def bulk_conductivity(species, form, slot="engine"):
    """Conductivity (W/m/K) of one cell of the species in this form: its solid
    k_s, except a packed GRAIN form (one with a `por_grain`), which is a BED of grains
    with gas between them: packed_k(k_s, K_PORE_GAS, por_grain). Sand reads about 0.3,
    not the 7.7 of the quartz of a rock. None if the species has no solid k."""
    k = get(species).get("k_s", slot)
    if k is None:
        return None
    e = form_porosity(species, form, slot)
    return float(k) if e <= 0.0 else packed_k(k, seed.K_PORE_GAS, e)


def seed_mat_rows():
    """Seed rows for legacy_tables: one per listed mat id 12..: thermal numbers
    from the species (engine slot, falling back to ref), `like` from MAT_LIKE, a
    melt entry for a species that melts into a fluid of its own. The density is
    the BULK density of the mat's form (bulk_density): sand, a packed bed, is
    not as dense as the quartz of a rock; the conductivity is the bed's
    (bulk_conductivity), for the same reason."""
    rows = []
    for m in sorted(seed.MAT_SLOTS):
        slot = MATS[m]
        sp = SPECIES[slot.species]
        row = {"mat": m, "like": slot.like, "rho": bulk_density(sp.id, slot.form),
               "c": sp.get("c_s"), "k": bulk_conductivity(sp.id, slot.form), "ign": sp.get("ign"),
               "emiss": sp.get("emiss_s"), "melt": None}
        if (slot.form == BOUND and sp.name != "water" and sp.liquid is not None
                and sp.has("Tm") and sp.has("Lf")):
            row["melt"] = (sp.get("Tm"), sp.liquid, sp.get("Lf"))
        rows.append(row)
    return rows


def seed_fluid_rows():
    """Seed rows for legacy_tables, one per new fluid id (rho in g/ml)."""
    rows = []
    for f in sorted(seed.FLUID_SLOTS):
        sp = SPECIES[FLUIDS[f].species]
        rows.append({"fl": f, "like": FLUIDS[f].like, "rho": sp.get("rho_l") / 1000.0,
                     "c": sp.get("c_l"), "h": sp.get("h_wet"),
                     "emiss": sp.get("emiss_l"), "k": sp.get("k_l")})
    return rows


def padded_tables(legacy=None, nmat=NMAT_PAD, nfluid=NFLUID_PAD):
    """legacy_tables.build_tables on the live materials (or the dicts passed in)
    plus the seed rows."""
    return build_tables(legacy or legacy_inputs(), seed_mat_rows(), seed_fluid_rows(),
                        nmat, nfluid)
