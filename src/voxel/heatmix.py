"""THE HEAT LAWS' VIEW OF A MIX (wave 13a, lane B5; T4 plug point).

`HeatMixin` needs two things of a mix row that the pool (mixtable.MixTable) does
not hold as a cached column: the boiling-point RISE of a liquid (brine boils hotter),
and the switch ledger of the legacy laws that skip a mixed cell (`CRUTCHES`).

  boil_rise(mixes, ids)   K above pure water at one atmosphere, for mix rows `ids`
                          (float32). A row with no dissolved solute, or no water,
                          gives exactly 0.0, so a pure cell never moves.

The rise comes from mixwater.boiling_rise (the activity of water, T4 4.3) on the grams
of each species of the row. The pool names species as species.py does ("sodium_chloride");
mixdata names them by its own keys ("nacl"): the small alias table below maps the ones
that differ, and a species mixdata has no column for is left out of the sum (a solid
skeleton does not change what the water does).

PLAIN NUMPY. No World is imported.

THE SWITCH LEDGER (plan section 8, decision D6). Every legacy law that WRITES smass,
fvol, fl, mat or fpot skips the cells that hold a mix, until it is moved to take/put.
Each skip is one row of CRUTCHES with the condition that deletes it. C11 fails if a
row is still live at the close of the wave. tests/test_w13_heat_defaults.py checks
that every `legacy_ok` call in heat.py has a row here, and the reverse.
"""
import weakref

import numpy as np

from . import mixdata as D
from . import mixphase as PH
from . import mixtures as MX
from . import mixvle as VLE
from . import mixwater as W
from . import species as SP
from .materials import BLOOD, MTIN
from .species_seed import BOUND, CRYSTAL, DISSOLVED

__all__ = ["boil_rise", "CRUTCHES", "crutch_laws", "boil_mixed", "melt_mixed"]

_ALIAS = {"water": "h2o", "sodium_chloride": "nacl", "sulfuric_acid": "h2so4",
          "acetic_acid": "acetic", "copper": "cu", "tin": "sn", "lead": "pb"}
_MEMO = weakref.WeakKeyDictionary()     # mix table -> {(row id, mixdata revision): rise K}


def _column(species_id):
    """The mixdata column of a pool species id, or -1 when mixdata has none."""
    name = SP.SPECIES[int(species_id)].name
    return D.INDEX.get(_ALIAS.get(name, name), -1)


def boil_rise(mixes, ids):
    """Boiling-point rise, K, of the liquid in mix rows `ids` (any shape of ids
    above 0): float32, 0.0 for a row with nothing dissolved in its water."""
    ids = np.asarray(ids, np.int64)
    flat = ids.reshape(-1)
    out = np.zeros(flat.shape, np.float32)
    if not flat.size:
        return out.reshape(ids.shape)
    memo = _MEMO.setdefault(mixes, {})
    rev = D.revision()
    u, inv = np.unique(flat, return_inverse=True)
    need = np.array([int(i) for i in u if (int(i), rev) not in memo], np.int64)
    if need.size:
        ent = mixes.entries(need)
        comp = np.zeros((need.size, D.n_species()))
        cols = np.array([_column(s) for s in ent.species], np.int64)
        ok = (cols >= 0) & (ent.form == DISSOLVED)
        np.add.at(comp, (ent.cell[ok], cols[ok]), ent.frac[ok])
        rise = np.nan_to_num(W.boiling_rise(comp), nan=0.0)
        for i, r in zip(need, rise):
            memo[(int(i), rev)] = np.float32(r)
    vals = np.array([memo[(int(i), rev)] for i in u], np.float32)
    out = vals[inv.reshape(-1)]
    return out.reshape(ids.shape)


# ── the switch ledger: the legacy laws of heat.py that skip a mixed cell ───────
# law: the method; skips: which mix ids it steps over; why: what it writes that a mix
# would lose; deleted_when: the condition that removes the skip; owner: the lane.
CRUTCHES = (
    {"law": "_law_melt", "skips": "scomp and fcomp",
     "why": "melts a cell that VIEWS as TIN/LEAD/IRON into pure molten metal: a 60/40 "
            "lead-tin cell would lose its lead (writes smass, fl, fvol, mat)",
     "deleted_when": "C2 (13b) moved the mixes of a registered metal system to heatmix.melt_mixed "
                     "(phase_step, gate w.mix_laws). The skip stays for a mixed cell no system "
                     "governs (aqueous ice and salt, three species, a species mixdata cannot "
                     "name) until the ice law reads mixphase (brine freezes at its own point: C12 M1). "
                     "C3 (13c) built the ICE cell for pure water only and skips mixed cells (D6)",
     "owner": "C2"},
    {"law": "_burn_cells", "skips": "scomp (solid), fcomp (oil pool)",
     "why": "burns, chars and ashes by the material a mix views as (writes smass, mat, "
            "fvol, fl): the mix's other species would vanish",
     "deleted_when": "burning moves to the species fuel(key) law (take/put). C4 (13c) did soot "
                     "deposition only, not the burn: this waits for the reaction row runner (C12 M3)",
     "owner": "C2"},
    {"law": "_law_skin_water", "skips": "fcomp",
     "why": "boils the water held in a porous skin and clears fl/fvol/fpot: a brine "
            "in the skin would lose its solute",
     "deleted_when": "the skin water gets a composition (boil_step on the skin). C3 (13c) kept the "
                     "pure-water skin law and left brine in a skin alone",
     "owner": "C2"},
    {"law": "_law_boil", "skips": "fcomp",
     "why": "deletes the steam of any cell with fl == WATER: a boiled-dry salt "
            "solution would lose its NaCl grams (T4 6.2 item 1)",
     "deleted_when": "C2 (13b) boils a mixed liquid by heatmix.boil_mixed (boil_step with the "
                     "crystal rule, gate w.mix_laws); the skip stays only for the gate off and "
                     "for liquids with no dissolved water",
     "owner": "C2"},
)


# ── the WORLD laws of C2: brine boils, mixes melt (wave 13b) ──────────────────
# Both take the world `w` as an argument (this module imports no World) and are
# called by heat.py only while `w.mixes_live` and the gate `w.mix_laws` (default
# True) are on. They move matter with the pool's own verbs (take, convert,
# from_dense) so every gram stays in a species book, and they book the joules
# through mixtures.book_* (T4 is the one writer of those terms).
_COLS = {}                                   # the pool-id <-> mixdata-column maps, by registry revision


def _col_maps():
    """(pool species id per mixdata column, mixdata column per pool species id); -1 = none."""
    rev = D.revision()
    if _COLS.get("rev") != rev:
        top = max(s.id for s in SP.SPECIES if s is not None) + 1
        by_col = np.full(D.n_species(), -1, np.int64)
        by_pool = np.full(top, -1, np.int64)
        inv = {v: k for k, v in _ALIAS.items()}
        for c, name in enumerate(D.ORDER):
            sid = SP.BY_NAME.get(inv.get(name, name), -1)
            if sid >= 0:
                by_col[c] = sid
                by_pool[sid] = c
        _COLS.clear()
        _COLS.update(rev=rev, by_col=by_col, by_pool=by_pool)
    return _COLS["by_col"], _COLS["by_pool"]


def _dense(mixes, ids, mass, forms=None):
    """((n, species) grams in mixdata order, grams of species mixdata has no column for) of the
    lumps `ids` of `mass` grams, summed over the forms in `forms` (all when None)."""
    _bc, by_pool = _col_maps()
    ids = np.asarray(ids, np.int64)
    ent = mixes.entries(ids)
    col = by_pool[ent.species]
    g = ent.frac * np.asarray(mass, np.float64)[ent.cell]
    sel = col >= 0
    if forms is not None:
        sel = sel & np.isin(ent.form, forms)
    out = np.zeros((len(ids), D.n_species()))
    np.add.at(out, (ent.cell[sel], col[sel]), g[sel])
    return out, float(g[col < 0].sum())


_ROWCOL = weakref.WeakKeyDictionary()        # mix table -> {(column, row, revision): value}


def _row_col(mx, name, ids, forms=None):
    """A mixture column (T_boil_1atm, T_solidus, T_liquidus: MX.DERIVED) for mix rows `ids`
    (float64, NaN where the row has no such number or holds a species mixdata cannot name).
    Evaluated once per distinct row and registry revision. `forms` picks which forms count
    (the boiling point reads the DISSOLVED grams only: a crystal does not lower the water's
    activity)."""
    memo = _ROWCOL.setdefault(mx, {})
    rev = D.revision()
    name_key = (name, forms)
    ids = np.asarray(ids, np.int64)
    u = np.unique(ids)
    need = np.array([i for i in u.tolist() if (name_key, i, rev) not in memo], np.int64)
    if need.size:
        comp, _l = _dense(mx, need, np.ones(len(need)), forms)
        vals = np.asarray(MX.DERIVED[name](comp), np.float64)
        # a row with a species mixdata cannot name is not read at all: its mass fraction is
        # missing, so the number would be for a different liquid
        ent = mx.entries(need)
        bad = np.zeros(len(need), bool)
        np.logical_or.at(bad, ent.cell, _col_maps()[1][ent.species] < 0)
        for i, v, b in zip(need.tolist(), vals.tolist(), bad.tolist()):
            memo[(name_key, i, rev)] = float("nan") if b else v
    return np.array([memo[(name_key, i, rev)] for i in u.tolist()])[np.searchsorted(u, ids)]


def _pressure_add(w, idx):
    """The pure law's pressure term for the boiling point (K) at the cells `idx`; 0 without pcell."""
    if w.pcell is None:
        return 0.0
    return np.clip(w._upcell(w.pcell), 0.0, 18.0)[idx]


def _store_fvol(w, cells, rows, grams):
    """fvol of `cells` from the lump mass `grams` of mix `rows`; the float32 rounding is booked
    (`f32`), the way fields._fluid_commit books it."""
    mx = w.mixes
    rho = np.asarray(mx.prop("rho", rows), np.float64) / 1000.0
    rho = np.where(np.isfinite(rho) & (rho > 0), rho, 1.0)
    vol = np.asarray(grams, np.float64) / rho
    v32 = vol.astype(np.float32)
    ent = mx.entries(rows)
    mx.book("f32", ent.species, ent.frac * ((vol - v32.astype(np.float64)) * rho)[ent.cell])
    w.fvol[cells] = v32


def _clear_fluid(w, cells):
    """The cells hold no liquid any more (the grams were booked by the caller)."""
    w.fcomp[cells] = 0
    w.fl[cells] = 0
    w.fvol[cells] = 0.0
    w.fpot[cells] = 0.0


_COLD_BOIL = 60.0      # C: below this no mixed liquid of the pool (methanol 65, ethanol 78) boils


def boil_mixed(w, T, C, boil=100.0):
    """THE BOIL LAW FOR MIXED LIQUIDS (T4 6.2 items 1, 2 and 4; D6 for the liquid).
    A cell that holds a mixed liquid and is hotter than the boiling point of its own water
    activity (brine: 101.8 C at 10 percent) loses the steam `mixvle.boil_step` solves for it:
    water (and ethanol or methanol, by the bubble point) leave, the solute stays, the boiling
    point climbs as the liquid thickens, and past its nucleation ratio the salt comes out as
    CRYSTAL in the same step (a boiled-dry brine keeps its NaCl). The steam species go to the
    pool's `gone` book, the joules to `boiled_J`, the heat the crystals gave back to
    `dissolved_J`. A cell with no liquid water left puts its crystal residue in the solid
    slot when that is free. `T`, `C` are the world's T() and heat_capacity(); `boil` is the
    sim's pure-water boiling point (materials.BOIL), raised by the room pressure as in the
    pure law."""
    mx = w.mixes
    live = (w.fcomp > 0) & (w.fvol > 0)
    if not live.any():
        return
    live &= T > _COLD_BOIL                                    # (no liquid we know boils below
    if not live.any():                                        # methanol's 65 C: cheaper than a row)
        return
    idx = np.nonzero(live)
    ids = w.fcomp[idx].astype(np.int64)
    lo = np.asarray(_row_col(mx, "T_boil_1atm", ids, (DISSOLVED,)), np.float64)
    add = _pressure_add(w, idx)
    hot = np.isfinite(lo) & (T[idx] > lo - 1.0 + add)         # a prefilter: boil_step decides
    if not hot.any():
        return
    h = tuple(a[hot] for a in idx)
    ids_h = ids[hot]
    add_h = add[hot] if isinstance(add, np.ndarray) else np.zeros(len(ids_h))
    rho = np.asarray(mx.prop("rho", ids_h), np.float64) / 1000.0
    mass = w.fvol[h].astype(np.float64) * rho
    comp, _l = _dense(mx, ids_h, mass, (DISSOLVED,))
    cry, _l = _dense(mx, ids_h, mass, (CRYSTAL,))
    iw = D.INDEX["h2o"]
    Th, Ch = T[h].astype(np.float64), C[h].astype(np.float64)
    steam = np.zeros_like(comp)
    cry_new = cry.copy()
    cry_J = np.zeros(len(ids_h))
    for a in np.unique(add_h.round(1)):                       # one pressure term, one call
        g = add_h.round(1) == a
        if not (comp[g, iw] > 0).any():
            continue
        out = VLE.boil_step(comp[g], Th[g], Ch[g], tb_water=boil + float(a), crystal=cry[g])
        st = np.array(out[0], np.float64)
        rem = np.asarray(out[1])[:, iw]                              # a sliver of water the law cannot
        sliv = (rem > 0.0) & (rem <= 0.01)                     # reach is boiled too, as the pure
        st[sliv, iw] += rem[sliv]                              # law clears fvol <= 0.01
        steam[g] = st
        cry_new[g] = out.crystal_new
        cry_J[g] = out.crystal_J
    if not (steam > 0.0).any():
        return
    lv = np.array([D.SPECIES[x].Lv for x in D.ORDER])
    by_col = _col_maps()[0]
    ci, sp = np.nonzero(steam > 0.0)
    pid = by_col[sp]
    ids2, m2, took = mx.take(ids_h, mass, ci, SP.pack_key(pid, DISSOLVED, 0), steam[ci, sp],
                             return_taken=True)
    dc = cry_new - cry
    cj, cs = np.nonzero(dc > 1e-12)
    if len(cj):
        ids2, m2 = mx.convert(ids2, m2, cj, by_col[cs], DISSOLVED, CRYSTAL, dc[cj, cs])
    taken = np.zeros_like(steam)
    taken[ci, sp] = took
    boiled = taken @ lv                                        # joules the steam took, per cell
    mx.book("gone", pid, took)
    w.E[h] += (-boiled + cry_J).astype(np.float32)             # the latent heat leaves; the crystals' heat stays
    w.p_add[h] += (taken[:, iw] * 0.1).astype(w.p_add.dtype)   # steam pushes, as in the pure law
    MX.book_boiled(w, boiled.sum())
    MX.book_dissolved(w, -cry_J.sum())
    # --- write the cells back
    dis_left = _dense(mx, ids2, m2, (DISSOLVED,))[0][:, iw]
    dry = dis_left <= 0.0
    wet = ~dry & (m2 > 0)
    if wet.any():
        hw = tuple(a[wet] for a in h)
        w.set_fluid(hw, ids2[wet])
        _store_fvol(w, hw, ids2[wet], m2[wet])
    if dry.any():
        hd = tuple(a[dry] for a in h)
        rows, mm = ids2[dry], m2[dry]
        free = (w.mat[hd] == 0) & (w.scomp[hd] == 0)      # (an air cell holds air grams in smass)
        res = mm > 0
        put = free & res
        if put.any():
            hp = tuple(a[put] for a in hd)
            mx.store(w.smass, hp, rows[put], mm[put])
            w.set_solid(hp, rows[put])
            _clear_fluid(w, hp)
        if (~res).any():
            _clear_fluid(w, tuple(a[~res] for a in hd))
        stay = ~free & res                                    # the solid slot is taken: the
        if stay.any():                                        # crystal stays with the wet cell
            hs = tuple(a[stay] for a in hd)
            w.set_fluid(hs, rows[stay])
            _store_fvol(w, hs, rows[stay], mm[stay])


_SYS_MEMO = weakref.WeakKeyDictionary()      # mix table -> {(solid row, liquid row, revision): system name or ""}


def _system_of(mx, a, b, rev):
    """The phase system (a name) that governs a cell made of solid row `a` and liquid row `b`,
    or '' when it has none, holds a species mixdata has no column for, or is aqueous (the
    ice/salt systems need the ICE cell: C3)."""
    memo = _SYS_MEMO.setdefault(mx, {})
    key = (int(a), int(b), rev)
    got = memo.get(key)
    if got is None:
        ids = np.array([r for r in (a, b) if r > 0], np.int64)
        comp, lost = _dense(mx, ids, np.ones(len(ids)))
        comp = comp.sum(axis=0)
        sysm = PH.system_for(comp) if lost == 0.0 and comp.sum() > 0.0 else None
        got = "" if sysm is None or sysm.pure_solids else sysm.name
        memo[key] = got
    return got


def melt_mixed(w, T, C):
    """MIXES MELT AND FREEZE (T4 4.8; alloys by the lever rule). A cell whose solid is a mix
    (a 60/40 solder) or whose liquid is a mix of the same species is solved by
    `mixphase.phase_step`: the liquid fraction at its temperature, the new temperature and the
    energy, together. The cell then holds a solid row and a liquid row (a mush) from
    `phase_assembly`: the liquid is enriched in the low-melting species, the solid is not.
    No supercooling band: the energy equation holds the cell on its plateau. The joules go to
    `melted_J` and `frozen_J`. Aqueous systems wait for the ICE cell (C3)."""
    mx = w.mixes
    smix = (w.scomp > 0) & (w.smass > 0)
    # a mixed liquid that is water-like (a solution, a wash, blood) freezes by the ice law
    # of C3; only the molten ones (fluid ids from MTIN up) are solved here
    lmix = (w.fcomp > 0) & (w.fvol > 0) & (w.fl >= MTIN) & (w.fl != BLOOD)
    cand = smix | lmix
    if not cand.any():
        return
    idx = np.nonzero(cand)
    a = np.where(smix[idx], w.scomp[idx], 0).astype(np.int64)
    b = np.where(lmix[idx], w.fcomp[idx], 0).astype(np.int64)
    allr = np.unique(np.concatenate([a, b]))
    allr = allr[allr > 0]
    ts = dict(zip(allr.tolist(), np.asarray(_row_col(mx, "T_solidus", allr), np.float64).tolist()))
    tl = dict(zip(allr.tolist(), np.asarray(_row_col(mx, "T_liquidus", allr), np.float64).tolist()))
    Tc = T[idx].astype(np.float64)
    nan = float("nan")
    ts_c = np.array([ts.get(int(x), nan) for x in a])
    tl_c = np.array([tl.get(int(x), nan) for x in b])
    solid_only, liquid_only = (a > 0) & (b == 0), (a == 0) & (b > 0)
    act = np.where(solid_only, np.isfinite(ts_c) & (Tc > ts_c - 1e-6),
                   np.where(liquid_only, np.isfinite(tl_c) & (Tc < tl_c + 1e-6), (a > 0) & (b > 0)))
    if not act.any():
        return
    sel = np.nonzero(act)[0]
    rev = D.revision()
    names = np.array([_system_of(mx, a[i], b[i], rev) for i in sel])
    Cc = C[idx]
    for nm in np.unique(names):
        if nm:
            g = sel[names == nm]
            _melt_group(w, mx, PH.get_system(str(nm)), tuple(x[g] for x in idx), a[g], b[g], Tc[g], Cc[g])


def _melt_group(w, mx, sysm, cells, a, b, Tg, Cg):
    ms = np.where(a > 0, w.smass[cells], 0.0).astype(np.float64)
    rho_b = np.asarray(mx.prop("rho", np.maximum(b, 1)), np.float64) / 1000.0
    ml = np.where(b > 0, w.fvol[cells].astype(np.float64) * rho_b, 0.0)
    cs, _x = _dense(mx, np.maximum(a, 1), ms)
    cl, _x = _dense(mx, np.maximum(b, 1), ml)
    comp = cs + cl
    tot = ms + ml
    f_old = ml / np.where(tot > 0, tot, 1.0)
    res = PH.phase_step(comp, f_old, Tg, Cg.astype(np.float64), system=sysm)
    f_new, T_new, dE = (np.asarray(x, np.float64) for x in res)
    ok = np.isfinite(f_new) & np.isfinite(T_new) & (np.abs(f_new - f_old) > 1e-9)
    if not ok.any():
        return
    _f, liq, sol, _cr = PH.phase_assembly(comp[ok], T_new[ok], f=f_new[ok], system=sysm)
    cells = tuple(x[ok] for x in cells)
    dE = dE[ok]
    liq_m, sol_m = liq.sum(axis=1), sol.sum(axis=1)
    by_col = _col_maps()[0]
    cols = np.nonzero(by_col >= 0)[0]
    l_rows, _m = mx.from_dense(liq[:, cols], by_col[cols], form=DISSOLVED)
    s_rows, _m = mx.from_dense(sol[:, cols], by_col[cols], form=BOUND)
    E_old = w.E[cells].astype(np.float64)
    s_gone = sol_m <= 1e-9
    keep = ~s_gone
    if keep.any():
        ck = tuple(x[keep] for x in cells)
        mx.store(w.smass, ck, s_rows[keep], sol_m[keep])
        w.set_solid(ck, s_rows[keep])
    if s_gone.any():
        cg = tuple(x[s_gone] for x in cells)
        mat0 = int(w.mat[cg][0])
        m = np.zeros(w.shape, bool)
        m[cg] = True
        w.scomp[cg] = 0                  # nothing "vanishes": the grams moved to the liquid
        w._unmake(m, "melts")
        w._defer("melts", target=("mat", mat0), data=(("cells", int(s_gone.sum())),), text="")
    lk = liq_m > 1e-9
    if lk.any():
        cl_ = tuple(x[lk] for x in cells)
        w.set_fluid(cl_, l_rows[lk])
        _store_fvol(w, cl_, l_rows[lk], liq_m[lk])
    if (~lk).any():
        _clear_fluid(w, tuple(x[~lk] for x in cells))
    w.E[cells] = (E_old + dE).astype(np.float32)
    MX.book_melted(w, np.maximum(-dE, 0.0).sum())
    MX.book_frozen(w, np.maximum(dE, 0.0).sum())


def crutch_laws():
    """The method names that have a row in the ledger."""
    return tuple(c["law"] for c in CRUTCHES)
