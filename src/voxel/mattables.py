"""THE PADDED LOOKUP ARRAYS (wave 13a, lane B3): what the per-id arrays of materials.py are made of.

materials.py keeps a float32 array for each number a law gathers by the id of a cell
(`_DENS_ARR[mat]`, `_KSOLID_ARR[mat]`, ... and `_FDENS_ARR[fl]` for liquids). They were
exactly as long as the ids that existed, twelve and nine, so a new material meant a
longer array in every file that sized one. From wave 13 they are NMAT_PAD = 64 and
NFLUID_PAD = 64 entries long whatever is listed, so a new material is a ROW of the
record (src/voxel/matrec.py) and not a code change.

The first twelve (nine) entries are the old arrays byte for byte: the frozen oracle
proves it (tests/test_oracle_format.py). The rest follow ONE rule:

  an id that has a row in the record gets that row's numbers (ICE, id 12, is the first)
  and is its own `like`;
  an id that has none copies row 0, AIR (NOFLUID for a liquid), so a stray id acts
  like empty space and never like a material somebody forgot to describe.

Two builders, one per kind of number:
  * the THERMAL arrays (density, c, k, ignition, emissivity, the skin depths, the
    liquid arrays) and the dense MELT / FREEZE forms come from
    legacy_tables.build_tables, written once for exactly this;
  * the MECHANICAL arrays (carrying span, toughness, thermal shock, light through a
    voxel) are handed in by the caller, already read off the record
    (matrec.legacy_views), and are padded here by the same copy rule.

THIS FILE IS PURE: numpy, legacy_tables and species_seed (the two pad sizes). It does
not import materials, matrec or species, so it can be imported from anywhere in any
order; materials.py hands it the old tables and the record rows and binds the names
it returns.
"""
import numpy as np

from .legacy_tables import build_tables, pad_like
from .species_seed import NFLUID_PAD, NMAT_PAD

__all__ = ["MECHANICAL", "seed_rows", "build"]

# The arrays whose numbers are mechanical: build_tables does not make them.
MECHANICAL = ("_SPAN_ARR", "_TOUGH_ARR", "_TSHOCK_ARR", "_TRANSMIT_ARR")


def seed_rows(rows, molten):
    """The rows of the record past the old ids, in the shape legacy_tables.build_tables
    takes for a new material: {"mat", "like", "rho" g/L, "c" J/(g K), "k", "ign" C or
    None, "emiss", "melt" (T C, fluid id, latent J/g) or None}.

    `rows` are record rows (matrec.MatRec); `molten` maps the key a melt row names
    ("mtin", "mlead", "miron") to its fluid id. ICE has no melt row here: T6 owns the
    ice law, so it stays out of every MELT and FREEZE table."""
    out = []
    for r in rows:
        melt = None if r.melt is None else (r.melt.T_C, molten[r.melt.fluid_key],
                                            r.melt.latent_Jg)
        out.append({"mat": r.id, "like": r.id, "rho": r.rho,
                    "c": r.cp / 1000.0, "k": r.k,
                    "ign": None if r.burn is None else r.burn.ign_C,
                    "emiss": r.eps, "melt": melt})
    return out


def build(legacy, rows, mech, molten=None, nmat=NMAT_PAD, nfluid=NFLUID_PAD):
    """Every padded array and table, as one dict named as materials.py names them.

    legacy: the old tables, as legacy_tables.build_tables takes them (SOLID, FLUID,
      EMISS, EMISS_FLUID, FLUID_K, MELT, BOIL, LATENT, SKIN_T, NMAT, FLIDS)
    rows:   the record rows past the old ids, in id order: ids NMAT, NMAT + 1, ...
    mech:   {"_SPAN_ARR", "_TOUGH_ARR", "_TSHOCK_ARR", "_TRANSMIT_ARR": float32
            array} with one entry per id that has a row (the old ids, then `rows`)
    molten: key -> fluid id for the melt rows of `rows` (None when there are none)
    Returns build_tables' dict (the thermal arrays, MELT_T MELT_FL MELT_LAT FREEZE_T
    FREEZE_MAT FREEZE_LAT, LIKE_MAT, LIKE_FLUID ...) plus the four mechanical arrays,
    all `nmat` (`nfluid`) long."""
    rows = list(rows)
    n_old = legacy["NMAT"]
    if [r.id for r in rows] != list(range(n_old, n_old + len(rows))):
        raise ValueError("the record rows past the old ids must be ids %d, %d, ... in order"
                         % (n_old, n_old + 1))
    if n_old + len(rows) > nmat:
        raise ValueError("%d materials do not fit in %d ids" % (n_old + len(rows), nmat))
    t = build_tables(legacy, seed_rows(rows, molten or {}), [], nmat, nfluid)
    like = [int(x) for x in t["LIKE_MAT"]]
    for name in MECHANICAL:
        have = np.asarray(mech[name], np.float32)
        if have.shape != (n_old + len(rows),):
            raise ValueError("%s has %s entries, the record has %d materials"
                             % (name, have.shape, n_old + len(rows)))
        t[name] = np.array(pad_like(list(have), like, nmat), np.float32)
    return t
