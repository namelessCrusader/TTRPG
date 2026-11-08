"""HEAT CONSUMERS AT THEIR DEFAULTS (wave 13a, lane B5).

Heat reads the mix (T1), the boiling point of a liquid (T4) and the surface-film
columns of the material row (D4), and every one of them is OFF by default. These
tests pin both halves: with the switch off (a world with no mix, `film_cols`
False) the laws give the old bits, and with it on the data does what it says,
the compiled law and the numpy law agreeing to the bit.
"""
import inspect
import re

import numpy as np
import pytest

from src.voxel import heat as H
from src.voxel import heatmix
from src.voxel.materials import (AIR, AMBIENT, BOIL, EMISS, FLESH, H_CONV, IRON, LEAD, NOFLUID,
                                 STONE, TIN, WATER, WOOD, _EMISS_ARR, _EMISS_FILM_ARR, _HNAT_ARR,
                                 NMAT, NMAT_PAD)
from src.voxel.sim import World

SOLDER = [("lead", "bound", 0, 0.4), ("tin", "bound", 0, 0.6)]
BRINE = [("water", "dissolved", 0, 0.9), ("sodium_chloride", "dissolved", 0, 0.1)]


# ── the data columns ─────────────────────────────────────────────────────────
def test_the_film_columns_are_the_old_constants_everywhere_but_the_flesh_row():
    assert len(_HNAT_ARR) == len(_EMISS_FILM_ARR) == NMAT_PAD
    other = np.arange(NMAT_PAD) != FLESH
    assert (_HNAT_ARR[other] == 0).all()                       # 0 means: the constant H_CONV
    assert np.array_equal(_EMISS_FILM_ARR[other], _EMISS_ARR[other])
    assert _HNAT_ARR[FLESH] == np.float32(2.38)                # ISO natural-convection law
    assert _EMISS_FILM_ARR[FLESH] == np.float32(0.98)
    assert _EMISS_ARR[FLESH] == np.float32(0.95) == np.float32(EMISS[FLESH])   # the law's own stays


def test_nothing_is_switched_on_by_default():
    w = World(6, 6, 6, voxel_cm=10)
    assert w.film_cols is False and "film_cols" not in vars(w)
    assert w._face_env((slice(None),) * 3).inert()
    assert w._emiss_table() is _EMISS_ARR
    assert w._boil_points((slice(None),) * 3) is None
    assert w._mix_window((slice(None),) * 3) == (None, None)
    assert w.legacy_ok() is None


# ── film_cols: off is the old law, on with no such surface is the old law ─────
def _scene(flesh):
    w = World(10, 10, 8, voxel_cm=10)
    w.fill(0, 10, 0, 10, 0, 1, STONE)
    w.fill(2, 5, 2, 5, 1, 3, FLESH if flesh else WOOD)
    w.fill(6, 8, 6, 8, 1, 3, IRON)
    w.E[6:8, 6:8, 1:3] += 4e5
    w.E[2:5, 2:5, 1:3] += 2e4
    return w


def _run(flesh, cols, fused, n=25):
    w = _scene(flesh)
    w.film_cols, w.fused = cols, fused
    for _ in range(n):
        w.step()
    return w


def test_film_cols_on_changes_nothing_when_no_surface_has_a_column():
    a, b = _run(False, False, True), _run(False, True, True)
    assert np.array_equal(a.E, b.E) and np.array_equal(a._skin_field(), b._skin_field())
    c = _run(False, True, False)
    assert np.array_equal(b.E, c.E)


def test_film_cols_on_gives_the_flesh_surface_the_iso_law_in_both_paths():
    off, on = _run(True, False, True), _run(True, True, True)
    plain = _run(True, True, False)
    assert not np.array_equal(off.E, on.E)                     # the data does something
    assert np.array_equal(on.E, plain.E)                       # and the compiled == numpy law
    assert np.array_equal(on._skin_field(), plain._skin_field())
    # hot gas heats a warm body less through h = 2.38 |dT|^0.25 than through 10 W/m2K at 10 K
    assert 2.38 * 10.0 ** 0.25 < H_CONV


def test_film_cols_refuses_to_share_the_bundle_with_an_overlay():
    from src.voxel.faceenv import FaceEnv, overlay_table
    w = _scene(True)
    w.film_cols = True
    w.face_env = FaceEnv(bz=np.zeros(w.shape, np.uint16), ov=overlay_table([{"hnat": 1.0}]))
    with pytest.raises(ValueError):
        w._face_env((slice(None),) * 3)


# ── gathers through the mix ──────────────────────────────────────────────────
def test_mix_pick_is_the_identity_without_a_mix_and_reads_the_table_with_one():
    w = World(6, 6, 6, voxel_cm=10)
    base = np.full((3, 3), 7.0, np.float32)
    assert H._mix_pick(None, "k", base, None) is base
    mx = w.ensure_mixes()
    row = mx.parse(SOLDER)
    assert H._mix_pick(mx, "k", base, np.zeros((3, 3), np.uint32)) is base
    ids = np.zeros((3, 3), np.uint32)
    ids[1, 1] = row
    got = H._mix_pick(mx, "k", base, ids)
    assert got is not base and (got[ids == 0] == 7.0).all()
    assert got[1, 1] == pytest.approx(float(mx.prop("k", np.array([row]))[0]))
    assert got.dtype == np.float32 and (base == 7.0).all()     # the pure array is never touched


def _solder_world(extra=None):
    w = World(8, 8, 8, voxel_cm=10)
    w.fill(0, 8, 0, 8, 0, 1, STONE)
    row = w.ensure_mixes().parse(SOLDER)
    w.fill(2, 4, 2, 4, 1, 3, None, mix=row)
    w.fill(5, 7, 5, 7, 1, 3, TIN)
    return w, row


def test_the_geometry_of_a_mixed_cell_reads_the_mix_and_a_pure_cell_keeps_its_bits():
    w = World(8, 8, 8, voxel_cm=10)
    w.fill(0, 8, 0, 8, 0, 1, STONE)
    row = w.ensure_mixes().parse([("wood", "bound", 0, 0.7), ("iron_oxide", "bound", 0, 0.3)])
    w.fill(2, 4, 2, 4, 1, 3, None, mix=row)          # views as WOOD, but is denser and conducts more
    w.fill(5, 7, 5, 7, 1, 3, WOOD)
    assert w.mixes_live and w.mat[2, 2, 1] == WOOD
    g = w._surface_geometry()
    cell, pure = (2, 2, 1), (5, 5, 1)
    assert g.sc[cell] == row

    def geo(at, *comp):
        args = [x[at][None] for x in (w.mat, w.smass, w.fl, w.fvol, g.C)]
        return w._geo_cells(*args, g.n_out[at][None], g.n_gas[at][None], None, *comp)

    sk = geo(cell, w.scomp[cell][None], w.fcomp[cell][None])
    sp = geo(cell)                                    # the same cell read as pure WOOD
    assert sk["has"][0] and sp["has"][0]
    assert sk["Cs"][0] != sp["Cs"][0] and sk["Gsb"][0] != sp["Gsb"][0]    # the mix's own skin
    # a pure cell: the very bits, whether or not it is handed its (zero) mix ids
    p, q = geo(pure), geo(pure, w.scomp[pure][None], w.fcomp[pure][None])
    for name in p:
        assert np.array_equal(p[name], q[name]), name


def test_the_geometry_cache_follows_a_mix_that_appears_and_one_that_changes():
    w = World(8, 8, 8, voxel_cm=10)
    w.fill(0, 8, 0, 8, 0, 1, STONE)
    w.fill(2, 4, 2, 4, 1, 3, TIN)
    w.step()
    g0 = w._surface_geometry()
    assert getattr(g0, "sc", None) is None
    mx = w.ensure_mixes()
    row = mx.parse(SOLDER)
    w.fill(5, 7, 5, 7, 1, 3, None, mix=row)
    g1 = w._surface_geometry()
    assert g1.sc is not None and (g1.sc[5:7, 5:7, 1:3] == row).all()
    # the incremental build and the full one agree to the bit
    H._GEO.pop(w, None)
    g2 = w._surface_geometry()
    for name in ("Cs", "Gsb", "Cb", "Csur", "Ccor", "af", "void", "has"):
        assert np.array_equal(getattr(g1, name), getattr(g2, name)), name


def test_conduction_runs_on_a_world_with_a_mix_and_fused_equals_plain():
    outs = []
    for fused in (True, False):
        w, row = _solder_world()
        w.E[2:4, 2:4, 1:3] += 3e4
        w.fused = fused
        for _ in range(20):
            w.step()
        outs.append(w.E.copy())
    assert np.array_equal(*outs)


# ── the legacy_ok gates and the boil guard ───────────────────────────────────
def test_a_mixed_solder_cell_is_not_melted_by_the_tin_law_and_a_pure_tin_cell_is():
    w, row = _solder_world()
    w.mix_laws = False          # C2: with the gate off (the 13a state) the legacy skip holds
    w.E[...] = np.where(w.smass > 0, w.heat_capacity() * (300.0 - AMBIENT), 0.0).astype(np.float32)
    mix_before = w.smass[2:4, 2:4, 1:3].copy()
    tin_before = w.smass[5:7, 5:7, 1:3].copy()
    w._law_melt()
    assert np.array_equal(w.smass[2:4, 2:4, 1:3], mix_before)       # skipped (heatmix.CRUTCHES)
    assert (w.scomp[2:4, 2:4, 1:3] == row).all() and (w.mat[2:4, 2:4, 1:3] == w.mat[2, 2, 1]).all()
    assert (w.smass[5:7, 5:7, 1:3] < tin_before).all()             # pure tin melts as ever


def test_a_boiling_brine_keeps_its_salt_and_water_still_boils():
    w = World(8, 8, 8, voxel_cm=10)
    w.mix_laws = False          # C2: with the gate off (the 13a state) the guard holds
    w.fill(0, 8, 0, 8, 0, 1, STONE)
    row = w.ensure_mixes().parse(BRINE)
    w.pour(2, 2, 1, None, 300.0, mix=row)
    w.pour(5, 5, 1, WATER, 300.0)
    for cell in ((2, 2, 1), (5, 5, 1)):
        w.E[cell] = w.heat_capacity()[cell] * (130.0 - AMBIENT)
    brine = w.fvol[2, 2, 1]
    w._law_boil()
    assert w.fvol[2, 2, 1] == brine and w.fl[2, 2, 1] == WATER and w.fcomp[2, 2, 1] == row
    assert w.fvol[5, 5, 1] < np.float32(300.0)                      # pure water boils off
    # and boiled dry it still holds its row: the guard also covers the dry-out clear
    w.fvol[2, 2, 1] = 0.005
    w._law_boil()
    assert w.fl[2, 2, 1] == WATER and w.fvol[2, 2, 1] == np.float32(0.005)


def test_a_mixed_wood_cell_is_not_burnt_and_the_wood_beside_it_is():
    w = World(8, 8, 8, voxel_cm=10)
    w.fill(0, 8, 0, 8, 0, 1, STONE)
    mx = w.ensure_mixes()
    row = mx.parse([("wood", "bound", 0, 0.7), ("iron_oxide", "bound", 0, 0.3)])
    w.fill(2, 3, 2, 3, 1, 2, None, mix=row)
    w.fill(5, 6, 5, 6, 1, 2, WOOD)
    if w.mat[2, 2, 1] != WOOD:
        pytest.skip("this mix does not view as WOOD in this tree")
    for c in ((2, 2, 1), (5, 5, 1)):
        w.E[c] = w.heat_capacity()[c] * (600.0 - AMBIENT)
    m0, p0 = w.smass[2, 2, 1].copy(), w.smass[5, 5, 1].copy()
    for _ in range(5):
        w.step()
    assert w.smass[2, 2, 1] == m0 and w.mat[2, 2, 1] == WOOD        # skipped (heatmix.CRUTCHES)
    assert w.smass[5, 5, 1] < p0                                    # pure wood burns as ever


def test_every_legacy_ok_caller_in_heat_has_a_row_in_the_switch_ledger():
    src = inspect.getsource(H.HeatMixin)
    called = set()
    for m in re.finditer(r"    def (\w+)\(.*?(?=\n    def |\Z)", src, re.S):
        if "legacy_ok(" in m.group(0):
            called.add(m.group(1))
    assert called == set(heatmix.crutch_laws())
    for c in heatmix.CRUTCHES:
        assert c["deleted_when"] and c["why"] and c["owner"]


# ── the per-cell boiling point ───────────────────────────────────────────────
def test_boil_rise_of_pure_water_is_exactly_zero_and_brine_boils_hotter():
    w = World(4, 4, 4, voxel_cm=10)
    mx = w.ensure_mixes()
    r_pure = heatmix.boil_rise(mx, np.array([64 + WATER]))          # the pure water row
    brine = mx.parse(BRINE)
    r = heatmix.boil_rise(mx, np.array([brine, brine]))
    assert r_pure[0] == 0.0
    assert r[0] == r[1] and 1.5 < float(r[0]) < 2.1                 # 10 percent NaCl: +1.76 K (T4 4.3)
    assert r.dtype == np.float32


def test_a_world_without_a_mixed_liquid_gets_no_boiling_points_and_one_with_gets_them():
    w = World(8, 8, 8, voxel_cm=10)
    w.fill(0, 8, 0, 8, 0, 1, STONE)
    w.pour(5, 5, 1, WATER, 300.0)
    full = (slice(None),) * 3
    assert w._boil_points(full) is None and w._face_env(full).inert()
    row = w.ensure_mixes().parse(BRINE)
    w.pour(2, 2, 1, None, 300.0, mix=row)
    bp = w._boil_points(full)
    assert bp.dtype == np.float32 and bp[2, 2, 1] > BOIL + 1.5
    assert (np.delete(bp.reshape(-1), np.ravel_multi_index((2, 2, 1), bp.shape)) == np.float32(BOIL)).all()
    env = w._face_env(full)
    assert env.boils and not env.overlay and env.fits(w.shape)


def test_the_boiling_curve_is_measured_from_the_brines_own_point_in_both_paths():
    outs = []
    for fused in (True, False):
        w = World(8, 8, 8, voxel_cm=10)
        w.fill(0, 8, 0, 8, 0, 1, STONE)
        row = w.ensure_mixes().parse(BRINE)
        w.fill(2, 5, 2, 5, 1, 2, IRON)
        w.pour(3, 3, 2, None, 200.0, mix=row)
        w.E[2:5, 2:5, 1:2] += 1.5e5
        w.fused = fused
        for _ in range(15):
            w.step()
        outs.append((w.E.copy(), w.fvol.copy()))
    assert np.array_equal(outs[0][0], outs[1][0]) and np.array_equal(outs[0][1], outs[1][1])
