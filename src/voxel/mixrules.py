"""MIX RULES (wave 13, T1): what a mix of species LOOKS like, from the numbers of
its parts.

A mix row is a list of components (species, form, size, mass fraction). This file
turns that list into the numbers the laws read: density, heat capacity,
conductivity, emissivity, viscosity, heat of combustion, ignition temperature,
a melting stand-in and porosity. Every rule is a plain function on an `Entries`
table (flat, aligned arrays: one line per component of each row); every rule is
chosen by DATA columns (form, rho_e, phiV0, ...) and never by a species name.

THE REGISTRY. register_rule(kind, fn) adds or replaces a rule: fn(ent, slot,
**kw) -> array[n]. register_derived(name, fn, deps) adds a column the table
computes lazily and caches (T4's T_solidus, c_sat_NaCl...). T9 registers its
mechanical kinds the same way. evaluate(kind, ent) runs one.

TWO SLOTS. Every rule takes slot="engine" (what the sim runs on) or "ref" (the
real value). Tests of a rule against a handbook number run on "ref".

THE RULES (spec 3.6), all by mass fraction w_i:
  rho   ideal volume addition by FORM (solid forms: bulk rho, a composite
        species adds its skeleton volume and its pores; a GRAIN form with a
        `por_grain` is a packed bed and adds the volume of its bed, particle
        volume over (1 - por_grain); DISSOLVED: liquid rho).
        A DISSOLVED electrolyte in WATER adds its apparent molar volume
        phiV = phiV0 + Sv sqrt(m) (Masson) instead, only when water is the
        largest DISSOLVED species and the solute has phiV0.
  c     sum w_i c_i; c_i is the apparent value (c_app) for DISSOLVED forms.
  k     solids: the electronic part (Nordheim resistivity, Wiedemann-Franz) for
        BOUND entries with rho_e data, plus lattice share; everything else
        joins by the symmetric effective-medium rule (Bruggeman); a packed GRAIN
        entry joins as a BED (species.packed_k of its solid and the pore gas, the
        volume of its bed). Liquids: the
        mass-weighted k_app of the dissolved part, Bruggeman with any particles.
  eps   that of the dominant component (a surface property).
  hcomb, o2_ratio, soot_yield   sum w_i x_i (exact).
  ign   minimum over components with hcomb > 0 and w >= IGN_MIN_FRAC (data).
  Tmelt interim stand-in: mass mean of Tm over components that have one.
  mu    Andrade of the solvent (Kendall-Monroe for two miscible liquids), times
        the relative viscosity of each dissolved solute (log-linear table).
  por   1 - (skeleton volume) / (bulk volume).

Units: rho g/L, c J/(g K), k W/(m K), mu mPa s, T in degC unless a name ends _K.
Numbers marked (u) are unverified (RULES.md). A rule that needs a number the
species has not got returns NaN for that row; it never invents one, except the
four named defaults at the top (each (u)).
"""
from typing import NamedTuple

import numpy as np

from . import materials as mat
from . import species as SP
from .species_seed import BOUND, CRYSTAL, DISSOLVED, GRAIN, K_PORE_GAS, NFORMS

__all__ = ["Entries", "make_entries", "sp_arrays", "refresh", "register_rule", "register_derived",
           "RULES", "DERIVED", "evaluate", "core_columns", "CORE", "bruggeman",
           "wiener_series", "wiener_parallel", "register_nordheim", "nordheim_C", "NORDHEIM",
           "form_energy", "register_form_energy", "comp_c", "comp_c_keys", "andrade",
           "rho_mix", "c_mix", "k_mix", "eps_mix", "hcomb_mix", "o2_mix", "soot_mix", "ign_mix",
           "tmelt_mix", "por_mix", "mu_mix", "acid_weight_fraction", "acid_potency", "acid_mix",
           "WATER_G_PER_ML", "L_LORENZ", "IGN_MIN_FRAC", "EPS_DEFAULT",
           "MU_DEFAULT", "T_REF_C", "mass_to_mole"]

# ---- named defaults and constants (each (u) unless it is an identity) ------
L_LORENZ = 2.44e-8              # W ohm / K^2, Lorenz number (Sommerfeld value)
T_REF_C = 20.0                  # the reference temperature of every cached column
IGN_MIN_FRAC = 0.05             # a fuel below this mass fraction does not set the ignition point (u, data)
EPS_DEFAULT = 0.9               # emissivity when the dominant component has none (u)
MU_DEFAULT = 1.0                # mPa s, a liquid with no viscosity data at all (u)
BRUGGEMAN_STEPS = 50
KELVIN = 273.15
WATER_G_PER_ML = 0.998          # the density the repo's potency formula uses for the water part

# Nordheim coefficients C_ij, ohm m, for the pair (i, j) of species ids, i < j.
# Cu-Sn 2.0e-6 is the spec's number, chosen to put 12 percent tin bronze at about
# 48 W/m/K (u). tests/data/sourced_numbers.json (cusn.nordheim_coefficient) found
# NO published value and derives 2.3e-6 to 3.7e-6 from room-temperature bronze
# resistivities (indicative only): a larger C makes bronze a WORSE conductor.
NORDHEIM = {(9, 17): 2.0e-6}
_NORD_MAT = {}


def register_nordheim(sp_a, sp_b, C):
    """Add or replace the Nordheim coefficient (ohm m) of a species pair."""
    a, b = sorted((int(sp_a), int(sp_b)))
    NORDHEIM[(a, b)] = float(C)
    _NORD_MAT.clear()


def nordheim_C(sp_a, sp_b):
    """C_ij for arrays of species ids (0 for a pair without data)."""
    n = len(SP.SPECIES)
    if "m" not in _NORD_MAT:
        m = np.zeros((n, n))
        for (a, b), c in NORDHEIM.items():
            m[a, b] = m[b, a] = c
        _NORD_MAT["m"] = m
    return _NORD_MAT["m"][np.asarray(sp_a, np.int64), np.asarray(sp_b, np.int64)]


# ---- the entries table ------------------------------------------------------
class Entries(NamedTuple):
    """Flat aligned arrays, one line per component of each row. A law works on
    this table; `cell` says which of the n rows each line belongs to."""
    n: int
    cell: np.ndarray            # int64[nnz]
    idx: np.ndarray             # int64[nnz] position inside its row
    key: np.ndarray             # uint32[nnz]
    species: np.ndarray         # int64[nnz]
    form: np.ndarray            # int64[nnz]
    size: np.ndarray            # int64[nnz]
    frac: np.ndarray            # float64[nnz] mass fraction, q * 2^-24

    def sum(self, values):
        """Per-row sum of a per-entry array."""
        return np.bincount(self.cell, weights=np.asarray(values, np.float64), minlength=self.n)


def make_entries(n, cell, key, frac):
    """An Entries table from (cell, key, frac), sorted by cell then key."""
    cell = np.asarray(cell, np.int64)
    key = np.asarray(key, np.uint32)
    sp, fo, sz = SP.unpack_key(key)
    first = np.zeros(len(cell), np.int64)
    if len(cell):
        starts = np.flatnonzero(np.r_[True, cell[1:] != cell[:-1]])
        first = np.arange(len(cell)) - np.repeat(starts, np.diff(np.r_[starts, len(cell)]))
    return Entries(int(n), cell, first, key, sp.astype(np.int64), fo.astype(np.int64),
                   sz.astype(np.int64), np.asarray(frac, np.float64))


# ---- the species arrays (dense by species id, NaN where a field is missing) --
FIELDS = ("M", "rho_s", "rho_l", "c_s", "c_l", "c_app", "k_s", "k_l", "k_app", "rho_e",
          "emiss_s", "emiss_l", "phiV0", "Sv", "hcomb", "o2_ratio", "soot_yield", "ign",
          "Tm", "Lf", "mu_A", "mu_B", "rho_skel", "porosity", "swell", "aspect",
          "dH_sol", "por_grain")
_CACHE = {}


def refresh():
    """Forget the cached species arrays (call after the registry changes)."""
    _CACHE.clear()


def sp_arrays(slot="engine"):
    """Dense arrays of every species field, indexed by species id, plus the
    derived per-form numbers: rho_S/c_S/k_S (a solid form's number, falling back
    to the liquid's), rho_L/c_L/k_L (a liquid's, falling back to the solid's),
    c_D/k_D (DISSOLVED: the apparent value, else the liquid's), and `skel`
    [species, form] (the skeleton flags)."""
    a = _CACHE.get(slot)
    if a is not None:
        return a
    n = len(SP.SPECIES)
    a = {f: np.full(n, np.nan) for f in FIELDS}
    a["skel"] = np.zeros((n, NFORMS), bool)
    a["mu_rel"] = [None] * n
    a["exists"] = np.zeros(n, bool)
    for s in SP.SPECIES:
        if s is None:
            continue
        a["exists"][s.id] = True
        for f in FIELDS:
            v = s.get(f, slot, None)
            if v is not None:
                a[f][s.id] = v
        for fo in s.skeleton:
            a["skel"][s.id, fo] = True
        t = s.get("mu_rel", slot, None)
        if t:
            a["mu_rel"][s.id] = np.asarray(t, np.float64)

    def pick(*names):
        out = a[names[0]].copy()
        for nm in names[1:]:
            out = np.where(np.isnan(out), a[nm], out)
        return out
    a["rho_S"], a["rho_L"] = pick("rho_s", "rho_l"), pick("rho_l", "rho_s")
    a["c_S"], a["c_L"] = pick("c_s", "c_l"), pick("c_l", "c_s")
    a["k_S"], a["k_L"] = pick("k_s", "k_l"), pick("k_l", "k_s")
    a["c_D"], a["k_D"] = pick("c_app", "c_L"), pick("k_app", "k_L")
    a["rho_skel_S"] = pick("rho_skel", "rho_S")
    _CACHE[slot] = a
    return a


def comp_c(species, form, slot="engine"):
    """Specific heat of a component (J/g/K): c_app for DISSOLVED, else the solid's."""
    A = sp_arrays(slot)
    sp = np.asarray(species, np.int64)
    return np.where(np.asarray(form) == DISSOLVED, A["c_D"][sp], A["c_S"][sp])


def comp_c_keys(keys, slot="engine"):
    """comp_c for an array of component keys."""
    sp, fo, _sz = SP.unpack_key(np.asarray(keys, np.uint32))
    return comp_c(sp, fo, slot)


def mass_to_mole(species, w, slot="engine"):
    """Moles per gram of mix of mass fractions w of the given species ids."""
    return np.asarray(w, np.float64) / sp_arrays(slot)["M"][np.asarray(species, np.int64)]


# ---- form energy (J/g at the 20 C reference, spec 4.2 and 4.5) ---------------
_FORM_ENERGY = {}


def register_form_energy(species, form_a, form_b, dH):
    """dH in J/g at 20 C for turning species from form_a into form_b (positive:
    absorbs heat). The reverse is stored as the negative."""
    _FORM_ENERGY[(int(species), int(form_a), int(form_b))] = float(dH)
    _FORM_ENERGY[(int(species), int(form_b), int(form_a))] = -float(dH)


def form_energy(species, form_a, form_b, slot="engine"):
    """dH(20 C) in J/g of form_a -> form_b of one species.
    Order: a registered value; a DISSOLVED <-> CRYSTAL change of a species that
    has dH_sol (solution enthalpy, positive when it absorbs heat); a change
    between a solid form and DISSOLVED for a species with Tm and Lf (melting
    moved to 20 C by Kirchhoff: Lf - (c_L - c_S)(Tm - 20)); a change between two
    solid forms is 0 (surface energy ignored). Anything else is NaN."""
    sp, a_, b_ = int(species), int(form_a), int(form_b)
    if a_ == b_:
        return 0.0
    if (sp, a_, b_) in _FORM_ENERGY:
        return _FORM_ENERGY[(sp, a_, b_)]
    A = sp_arrays(slot)
    if {a_, b_} == {DISSOLVED, CRYSTAL} and np.isfinite(A["dH_sol"][sp]):
        d = A["dH_sol"][sp]
        return d if a_ == CRYSTAL else -d
    if DISSOLVED in (a_, b_):
        tm, lf, cl, cs = A["Tm"][sp], A["Lf"][sp], A["c_L"][sp], A["c_S"][sp]
        if np.isfinite([tm, lf, cl, cs]).all():
            d = lf - (cl - cs) * (tm - T_REF_C)
            return float(d if a_ != DISSOLVED else -d)
        return float("nan")
    return 0.0


# ---- helpers -----------------------------------------------------------------
def _gather(A, name, ent):
    return A[name][ent.species]


def _first_per_cell(ent, order_key):
    """Index (into the entry arrays) of the entry with the smallest order_key in
    each row; a tie goes to the earlier entry. -1 for an empty row."""
    out = np.full(ent.n, -1, np.int64)
    if len(ent.cell) == 0:
        return out
    pos = np.arange(len(ent.cell))
    o = np.lexsort((pos, order_key, ent.cell))
    c = ent.cell[o]
    first = np.r_[True, c[1:] != c[:-1]]
    out[c[first]] = o[first]
    return out


def wiener_series(k, vol, cell, n):
    """The series (lowest) bound: 1 / sum(phi_i / k_i), per row."""
    tot = np.bincount(cell, weights=vol, minlength=n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return tot / np.bincount(cell, weights=vol / k, minlength=n)


def wiener_parallel(k, vol, cell, n):
    """The parallel (highest) bound: sum(phi_i k_i), per row."""
    tot = np.bincount(cell, weights=vol, minlength=n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.bincount(cell, weights=vol * k, minlength=n) / tot


def bruggeman(k, vol, cell, n):
    """The symmetric effective-medium conductivity of each row: the root of
    sum_i phi_i (k_i - k) / (k_i + 2 k) = 0. It does not depend on which phase is
    called the matrix. k, vol (any positive volume weights, normalised here) and
    cell are flat arrays; rows with no phase give NaN. The function is decreasing
    in k, so the root is unique and lies between the smallest and largest k_i;
    bisection, BRUGGEMAN_STEPS halvings."""
    k = np.asarray(k, np.float64)
    vol = np.asarray(vol, np.float64)
    cell = np.asarray(cell, np.int64)
    out = np.full(n, np.nan)
    if len(cell) == 0:
        return out
    tot = np.bincount(cell, weights=vol, minlength=n)
    phi = vol / tot[cell]
    lo = np.full(n, np.inf)
    hi = np.full(n, -np.inf)
    np.minimum.at(lo, cell, k)
    np.maximum.at(hi, cell, k)
    has = np.isfinite(lo) & (tot > 0)
    lo, hi = np.where(has, lo, 0.0), np.where(has, hi, 0.0)
    for _ in range(BRUGGEMAN_STEPS):
        mid = 0.5 * (lo + hi)
        m = mid[cell]
        f = np.bincount(cell, weights=phi * (k - m) / (k + 2.0 * m), minlength=n)
        up = f > 0
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    out[has] = (0.5 * (lo + hi))[has]
    return out


def _bed_porosity(ent, A):
    """Per entry: the porosity of its packed bed (por_grain) for a GRAIN entry that
    has one, else 0 (rock, crystal, dissolved, and grain without a number)."""
    por = np.nan_to_num(A["por_grain"][ent.species], nan=0.0)
    return np.where(ent.form == GRAIN, por, 0.0)


# ---- the rules ----------------------------------------------------------------
def _masson_volume(ent, A):
    """(volume L per g of mix of the DISSOLVED part, using Masson where it
    applies), as arrays per row; plus the mask of rows where it was used."""
    n = ent.n
    dis = ent.form == DISSOLVED
    w = ent.frac
    sp = ent.species
    # the largest DISSOLVED species of each row (ties to the lowest species id)
    big = _first_per_cell(ent, np.where(dis, -ent.frac, 1.0))
    water = SP.BY_NAME["water"]
    row_ok = np.zeros(n, bool)
    has = big >= 0
    row_ok[has] = dis[big[has]] & (sp[big[has]] == water)
    v = np.zeros(n)
    use = row_ok[ent.cell] & dis
    # the solvent: water at its liquid density
    sol = use & (sp == water)
    v += ent.sum(np.where(sol, w / A["rho_L"][sp], 0.0))
    ww = ent.sum(np.where(sol, w, 0.0))                      # g of water per g mix
    # solutes with phiV data: n_i phiV_i(m_i), phiV in mL/mol, result in L
    phi_ok = use & ~sol & np.isfinite(A["phiV0"][sp]) & np.isfinite(A["M"][sp])
    with np.errstate(divide="ignore", invalid="ignore"):
        nmol = np.where(phi_ok, w / A["M"][sp], 0.0)
        mol_kg = nmol / ww[ent.cell] * 1000.0                # molality, mol per kg water
        phiV = A["phiV0"][sp] + np.nan_to_num(A["Sv"][sp]) * np.sqrt(np.maximum(mol_kg, 0.0))
        v += ent.sum(np.where(phi_ok, nmol * phiV * 1e-3, 0.0))
    # a DISSOLVED solute without phiV data: ideal at its own liquid density
    ideal = use & ~sol & ~phi_ok
    v += ent.sum(np.where(ideal, w / A["rho_L"][sp], 0.0))
    return v, row_ok


def rho_mix(ent, slot="engine", masson=True):
    """Density, g/L: ideal addition of volumes by form, Masson for dissolved
    electrolytes in water (masson=False gives the plain ideal sum, to show why the
    rule exists). A composite species adds its bulk volume (skeleton + pores)."""
    A = sp_arrays(slot)
    sp, w = ent.species, ent.frac
    dis = ent.form == DISSOLVED
    if masson:
        vm, rowm = _masson_volume(ent, A)
    else:
        vm, rowm = np.zeros(ent.n), np.zeros(ent.n, bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        bed = 1.0 - _bed_porosity(ent, A)             # a packed bed holds (1 - por) of its volume
        v_ent = np.where(dis, w / A["rho_L"][sp], w / (A["rho_S"][sp] * bed))
        v_ent = v_ent + np.where((ent.form == BOUND) & np.isfinite(A["swell"][sp]),
                                 w * A["swell"][sp], 0.0)
        v_ent = np.where(rowm[ent.cell] & dis, 0.0, v_ent)
        v = ent.sum(v_ent) + vm
        return np.where(ent.sum(np.ones_like(w)) > 0, 1.0 / v, np.nan)


def por_mix(ent, slot="engine"):
    """Porosity of a solid mix: 1 - skeleton volume / bulk volume. A species
    without rho_skel has no pores of its own (its skeleton is its bulk)."""
    A = sp_arrays(slot)
    sp, w = ent.species, ent.frac
    solid = ent.form != DISSOLVED
    with np.errstate(divide="ignore", invalid="ignore"):
        vb = ent.sum(np.where(solid, w / (A["rho_S"][sp] * (1.0 - _bed_porosity(ent, A))), 0.0))
        vs = ent.sum(np.where(solid, w / A["rho_skel_S"][sp], 0.0))
        return np.where(vb > 0, 1.0 - vs / vb, 0.0)


def c_mix(ent, slot="engine"):
    """Specific heat, J/g/K: mass-weighted, c_app for dissolved forms."""
    return ent.sum(ent.frac * comp_c(ent.species, ent.form, slot))


def _electronic(ent, A, T_K):
    """The electronic group: BOUND entries with rho_e and M. Returns per-row
    (k_group, mass fraction of the group, has-group mask)."""
    n = ent.n
    sp = ent.species
    isA = (ent.form == BOUND) & np.isfinite(A["rho_e"][sp]) & np.isfinite(A["M"][sp])
    w = np.where(isA, ent.frac, 0.0)
    atoms = np.where(isA, w / np.where(isA, A["M"][sp], 1.0), 0.0)
    tot_at = ent.sum(atoms)
    mass = ent.sum(w)
    has = mass > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(isA, atoms / tot_at[ent.cell], 0.0)
        rho_e = np.where(isA, A["rho_e"][sp], 0.0)
        rho_mix_ = ent.sum(x * rho_e)
        # the pair term: sum over i < j of C_ij x_i x_j, entry by entry (rows are short)
        pair = np.zeros(n)
        a = np.flatnonzero(isA)
        if len(a):
            ca = ent.cell[a]
            kmax = int(ent.idx[a].max()) + 1
            for d in range(1, kmax):                  # offset between positions in a row
                # pair every A entry with the next A entry d places later in the same row
                j = np.arange(len(a) - d)
                ok = ca[j] == ca[j + d]
                if not ok.any():
                    break
                i1, i2 = a[j[ok]], a[j[ok] + d]
                pair += np.bincount(ent.cell[i1], weights=nordheim_C(sp[i1], sp[i2]) * x[i1] * x[i2],
                                    minlength=n)
        rho_mix_ = rho_mix_ + pair
        k_e = L_LORENZ * T_K / rho_mix_
        k_i = A["k_S"][sp]
        lat = np.where(isA, np.maximum(0.0, k_i - L_LORENZ * T_K / np.where(isA, rho_e, 1.0)), 0.0)
        k_lat = ent.sum(w * lat) / mass
        return k_e + k_lat, mass, has


def k_mix(ent, slot="engine", T_C=T_REF_C):
    """Conductivity, W/m/K, at T_C. A row with a BOUND entry is a solid: the
    electronic group (one phase) and every other entry (each its own phase) are
    joined by Bruggeman. A row with none is a liquid or a slurry: the dissolved
    part is one phase with the mass-weighted k_app; any particles join it by
    Bruggeman (a pure liquid has just the one phase)."""
    A = sp_arrays(slot)
    sp, w = ent.species, ent.frac
    n = ent.n
    T_K = T_C + KELVIN
    k_grp, m_grp, has_grp = _electronic(ent, A, T_K)
    dis = ent.form == DISSOLVED
    isA = (ent.form == BOUND) & np.isfinite(A["rho_e"][sp]) & np.isfinite(A["M"][sp])
    with np.errstate(divide="ignore", invalid="ignore"):
        wd = ent.sum(np.where(dis, w, 0.0))
        k_liq = ent.sum(np.where(dis, w * A["k_D"][sp], 0.0)) / wd   # mass-weighted k_app
        v_liq = ent.sum(np.where(dis, w / A["rho_L"][sp], 0.0))
        por = _bed_porosity(ent, A)
        v_ent = np.where(dis, 0.0, w / (A["rho_S"][sp] * (1.0 - por)))
        # a packed GRAIN entry is a bed: its phase conductivity is the bed's
        k_ent = A["k_S"][sp]
        bedded = (por > 0) & (k_ent > 0)
        if bedded.any():
            k_ent = k_ent.copy()
            k_ent[bedded] = SP.packed_k(k_ent[bedded], K_PORE_GAS, por[bedded])
        # phases: every non-dissolved non-group entry, plus one liquid, plus one group
        keep = ~dis & ~isA
        pk = [k_ent[keep]]
        pv = [v_ent[keep]]
        pc = [ent.cell[keep]]
        r = np.flatnonzero(wd > 0)
        pk.append(k_liq[r])
        pv.append(v_liq[r])
        pc.append(r)
        r = np.flatnonzero(has_grp)
        pk.append(k_grp[r])
        pv.append(ent.sum(np.where(isA, w / A["rho_S"][sp], 0.0))[r])
        pc.append(r)
        k = np.concatenate(pk)
        vol = np.concatenate(pv)
        cell = np.concatenate(pc)
        use = np.isfinite(vol) & (vol > 0)
        out = bruggeman(k[use], vol[use], cell[use], n)
        # a row that is only the group, or only liquid, keeps its own number exactly
        only_grp = has_grp & (ent.sum(keep.astype(float)) == 0) & (wd == 0)
        out = np.where(only_grp, k_grp, out)
        only_liq = (wd > 0) & (ent.sum(keep.astype(float)) == 0) & ~has_grp
        out = np.where(only_liq, k_liq, out)
        return out


def eps_mix(ent, slot="engine"):
    """Emissivity of the dominant component (the largest fraction)."""
    A = sp_arrays(slot)
    lead = _first_per_cell(ent, -ent.frac)
    out = np.full(ent.n, np.nan)
    ok = lead >= 0
    li = lead[ok]
    e = np.where(ent.form[li] == DISSOLVED, A["emiss_l"][ent.species[li]],
                 A["emiss_s"][ent.species[li]])
    e = np.where(np.isnan(e), np.where(ent.form[li] == DISSOLVED, A["emiss_s"][ent.species[li]],
                                       A["emiss_l"][ent.species[li]]), e)
    out[ok] = np.where(np.isnan(e), EPS_DEFAULT, e)
    return out


def _wsum(ent, A, name):
    v = np.nan_to_num(A[name][ent.species])
    return ent.sum(ent.frac * v)


def hcomb_mix(ent, slot="engine"):
    """Heat of combustion, J/g: sum w_i h_i (exact)."""
    return _wsum(ent, sp_arrays(slot), "hcomb")


def o2_mix(ent, slot="engine"):
    """Grams of O2 per gram of mix: sum w_i ratio_i."""
    return _wsum(ent, sp_arrays(slot), "o2_ratio")


def soot_mix(ent, slot="engine"):
    """Soot yield (g per g of mix burnt): sum w_i y_i."""
    return _wsum(ent, sp_arrays(slot), "soot_yield")


def ign_mix(ent, slot="engine"):
    """Ignition temperature: the lowest over components with hcomb > 0 and
    w >= IGN_MIN_FRAC. NaN if there is none."""
    A = sp_arrays(slot)
    sp = ent.species
    cand = ((np.nan_to_num(A["hcomb"][sp]) > 0) & (ent.frac >= IGN_MIN_FRAC)
            & np.isfinite(A["ign"][sp]))
    out = np.full(ent.n, np.inf)
    np.minimum.at(out, ent.cell[cand], A["ign"][sp][cand])
    return np.where(np.isinf(out), np.nan, out)


def tmelt_mix(ent, slot="engine"):
    """INTERIM melting point: the mass mean of Tm over the components that have
    one. A stand-in until T4's phase data; mixed cells may not enter the melt
    law before then (spec 6.3)."""
    A = sp_arrays(slot)
    tm = A["Tm"][ent.species]
    ok = np.isfinite(tm)
    wt = np.where(ok, ent.frac, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return ent.sum(wt * np.where(ok, tm, 0.0)) / ent.sum(wt)


def andrade(species, T_C, slot="engine"):
    """Viscosity of one pure liquid, mPa s: A exp(B / T_K). NaN without data."""
    A = sp_arrays(slot)
    sp = np.asarray(species, np.int64)
    return A["mu_A"][sp] * np.exp(A["mu_B"][sp] / (np.asarray(T_C, np.float64) + KELVIN))


def mu_mix(ent, slot="engine", T_C=T_REF_C):
    """Viscosity of the liquid part, mPa s, at T_C. Solvent: the Andrade law of
    the DISSOLVED species that have one (Kendall-Monroe on mole fractions if two or
    more). Times, for each other DISSOLVED species with a mu_rel table, the
    relative viscosity at its mass fraction of the row (log-linear). A row with no
    Andrade solvent gives MU_DEFAULT (molten metals; (u)). NaN for a row with
    nothing DISSOLVED."""
    A = sp_arrays(slot)
    sp, w, n = ent.species, ent.frac, ent.n
    dis = ent.form == DISSOLVED
    has_and = dis & np.isfinite(A["mu_A"][sp]) & np.isfinite(A["M"][sp])
    mu_i = np.where(has_and, andrade(sp, T_C, slot), 1.0)
    nmol = np.where(has_and, w / np.where(has_and, A["M"][sp], 1.0), 0.0)
    ntot = ent.sum(nmol)
    cnt = ent.sum(has_and.astype(float))
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(has_and, nmol / ntot[ent.cell], 0.0)
        km = ent.sum(x * np.cbrt(mu_i)) ** 3
    mu = np.where(cnt > 0, km, MU_DEFAULT)
    # relative viscosity of the solutes
    rel = np.ones(ent.n)
    for s in np.unique(sp[dis]):
        tab = A["mu_rel"][s]
        if tab is None:
            continue
        m = dis & (sp == s)
        lr = np.interp(w[m], tab[:, 0], np.log(tab[:, 1]))
        np.multiply.at(rel, ent.cell[m], np.exp(lr))
    mu = np.where(ent.sum(dis.astype(float)) > 0, mu * rel, np.nan)
    return mu


# ---- the registry ------------------------------------------------------------
RULES = {"rho": rho_mix, "c": c_mix, "k": k_mix, "eps": eps_mix, "hcomb": hcomb_mix,
         "o2_ratio": o2_mix, "soot_yield": soot_mix, "ign": ign_mix, "Tmelt": tmelt_mix,
         "por": por_mix, "mu": mu_mix}
CORE = ("rho", "c", "k", "eps")       # the columns the table caches (float32)
DERIVED = {}                          # name -> (fn, deps)
_REV = [0]


def register_rule(kind, fn):
    """Add or replace the rule `kind`: fn(ent, slot="engine", **kw) -> array[n].
    Replacing a core kind bumps the revision, so a table recomputes its caches."""
    RULES[str(kind)] = fn
    _REV[0] += 1


def register_derived(name, fn, deps=()):
    """Add a column the table computes lazily and caches per row: fn(ent) ->
    array[n]. deps names the columns it reads (for the reader of the registry; a
    re-registration of a dep bumps the revision and clears the cache)."""
    DERIVED[str(name)] = (fn, tuple(deps))
    _REV[0] += 1


def revision():
    return _REV[0]


def evaluate(kind, ent, slot="engine", **kw):
    """Run a registered rule or derived column on an Entries table."""
    if kind in RULES:
        return RULES[kind](ent, slot, **kw)
    if kind in DERIVED:
        return DERIVED[kind][0](ent)
    raise KeyError(f"no rule or derived column named {kind!r}")


def core_columns(ent, slot="engine"):
    """The four cached columns as float32 arrays."""
    return {k: np.asarray(RULES[k](ent, slot), np.float64).astype(np.float32) for k in CORE}


# ---- the acid family: potency <-> mix (spec 4.7) ------------------------------------
def acid_weight_fraction(fluid, p):
    """Weight fraction of the reagent in acid `fluid` (materials.ACID or
    WEAK_ACID) at potency p (the share of stock by volume, the rest water):
    w = STOCK rho p / (rho p + 0.998 (1 - p)), the formula materials.py uses.
    p = 1 gives STOCK (98.3 wt percent sulfuric acid, 5 wt percent vinegar)."""
    w, rho = mat.STOCK[fluid], mat.FLUID[fluid][0]
    p = np.asarray(p, np.float64)
    return w * rho * p / (rho * p + WATER_G_PER_ML * (1.0 - p))


def acid_potency(fluid, wt):
    """The inverse of acid_weight_fraction: the potency at which the reagent is
    `wt` by weight (1 at and above STOCK). The pair is a bijection on [0, STOCK]."""
    w, rho = mat.STOCK[fluid], mat.FLUID[fluid][0]
    wt = np.asarray(wt, np.float64)
    return np.where(wt >= w, 1.0, WATER_G_PER_ML * wt / (rho * (w - wt) + WATER_G_PER_ML * wt))


def acid_mix(fluid, p):
    """((reagent key, water key), (w, 1 - w)) for acid `fluid` at potency p: the
    mix the cell would hold if the acid were a species mix (both DISSOLVED). In
    wave 13 fpot stays the stored truth; this is the read-only view the ledger
    counts."""
    reagent = SP.BY_NAME["sulfuric_acid" if fluid == mat.ACID else "acetic_acid"]
    w = float(acid_weight_fraction(fluid, p))
    water = SP.BY_NAME["water"]
    return (SP.pack_key(reagent, DISSOLVED, 0), SP.pack_key(water, DISSOLVED, 0)), (w, 1.0 - w)
