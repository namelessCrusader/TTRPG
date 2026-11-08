"""MIXTURES ON (wave 13b, lane C2; T1 and T4).

Brine boils at the point of its own water activity and keeps its salt; a solder cell melts
over its range by the lever rule and gives the heat back when it freezes; the latent books
close; and the compiled twin of the settling sweep gives the plain path's bits.
Every world here holds a mix, so none is a golden case (no golden world has one).
"""
import numpy as np
import pytest

from src.voxel import heatmix, mixtures
from src.voxel import species as SP
from src.voxel.materials import AIR, BOIL, NOFLUID, WATER
from src.voxel.scenes_mixtures import (BRINE10, BRINE20, POT_CELL, brine_pot, solder_cell,
                                       wet_pot_world)
from src.voxel.species_seed import GRAIN

C = POT_CELL


def _heat_pot(rows, watts_j, n):
    """Heat the pot cell by `watts_j` joules a tick for n ticks; returns the world, the joules put
    in and the cell temperature each tick."""
    w = brine_pot(rows)
    temps, put = [], 0.0
    for _ in range(n):
        w.E[C] += watts_j
        put += watts_j
        w.step()
        temps.append(float(w.T()[C]))
    return w, put, np.array(temps)


# ── brine boils hotter ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("rows,rise", [(None, 0.0), (BRINE10, 1.8), (BRINE20, 4.6)])
def test_a_pot_pins_at_the_boiling_point_of_its_own_water(rows, rise):
    """Two pots on one slow heat: pure water pins at 100 C, 10 percent brine at 101.8 C, 20
    percent at 104.6 C (T4 scene 1). The pin is read while half the water is still there."""
    w, _put, temps = _heat_pot(rows, 120.0, 330)
    mid = temps[(temps > 99.0)][40:90]
    assert len(mid) > 20
    # the heater runs a little ahead of the boil: the cell sits at the boiling point plus the
    # last tick's heat over its heat capacity (about 0.1 K here), never below the point
    assert (mid >= BOIL + rise - 0.25).all(), (mid.min(), rise)
    assert (mid <= BOIL + rise + 0.8).all(), (mid.max(), rise)


def test_the_brine_pot_is_hotter_than_the_water_pot_at_the_same_heat():
    pins = []
    for rows in (None, BRINE20):
        temps = _heat_pot(rows, 120.0, 330)[2]
        pins.append(float(temps[temps > 99.0][60:110].mean()))
    assert pins[1] - pins[0] > 3.5, pins


# ── the books close ────────────────────────────────────────────────────────────
def _species_total(w, species_id):
    """Grams of one species in the whole lattice (both slots), plus the pool's `gone` book."""
    mx = w.mixes
    t = 0.0
    for comp, mass in ((w.scomp, w.smass), (w.fcomp, None)):
        sel = comp > 0
        if not sel.any():
            continue
        ids = comp[sel].astype(np.int64)
        if mass is None:
            m = w.fvol[sel].astype(np.float64) * mx.prop("rho", ids) / 1000.0
        else:
            m = mass[sel].astype(np.float64)
        t += mx.totals(ids, m).get(species_id, 0.0)
    return t + mx.ledger("gone").get(species_id, 0.0)


def test_a_pot_boiled_dry_keeps_its_salt_and_every_gram_and_joule_is_booked():
    w = brine_pot(BRINE10)
    water, salt = SP.BY_NAME["water"], SP.BY_NAME["sodium_chloride"]
    w0, s0 = _species_total(w, water), _species_total(w, salt)
    e0, shed0 = float(w.E.sum()), w.shed
    put = 0.0
    for _ in range(420):
        w.E[C] += 400.0
        put += 400.0
        w.step()
    assert w.fvol[C] == 0 and w.fl[C] == NOFLUID and w.fcomp[C] == 0       # the liquid is gone
    assert w.smass[C] > 5.0 and w.scomp[C] > 0 and w.mat[C] != AIR          # and the salt is a crust
    assert abs(_species_total(w, salt) - s0) < 1e-3                          # not a gram of salt lost
    assert abs(_species_total(w, water) - w0) < 1e-3                         # the water went to `gone`
    assert w.mixes.ledger("gone").get(water, 0.0) > 0.99 * w0
    # joules: what went in = what the lattice holds more + the steam took + the crystals gave
    # back + the heat shed
    closed = (float(w.E.sum()) - e0) + w.boiled_J + w.dissolved_J + (w.shed - shed0)
    assert abs(closed - put) < 0.002 * put, (closed, put)
    assert w.boiled_J == pytest.approx(w0 * 2260.0, rel=0.01)
    assert w.dissolved_J < 0                                                 # crystallising gives heat


def test_the_steam_goes_to_the_gone_book_and_the_pressure_pushes_as_the_pure_law_does():
    w, _p, _t = _heat_pot(BRINE10, 400.0, 150)
    assert w.mixes.ledger("gone").get(SP.BY_NAME["water"], 0.0) > 1.0


def test_the_gate_off_is_the_wave_13a_guard():
    w = brine_pot(BRINE10)
    w.mix_laws = False
    w.E[C] = w.heat_capacity()[C] * 110.0
    v0 = float(w.fvol[C])
    w._law_boil()
    assert w.fvol[C] == v0 and not hasattr(w, "boiled_J")


def test_pure_water_is_not_touched_by_the_mixed_law():
    """A pure pot beside a mixed one boils by the old law: no book is written for it."""
    w = brine_pot(None)
    w.mixes_live = True                      # the switch is on, no cell holds a mix
    w.ensure_mixes()
    w.E[C] = w.heat_capacity()[C] * 110.0
    w._law_boil()
    assert not hasattr(w, "boiled_J") and w.fvol[C] < 50.0


# ── mixes melt ─────────────────────────────────────────────────────────────────
def test_a_solder_cell_melts_on_a_plateau_and_freezes_back_with_the_heat_returned():
    w = solder_cell()
    c = (3, 2, 1)
    mx = w.mixes
    tin, lead = SP.BY_NAME["tin"], SP.BY_NAME["lead"]

    def grams(sp):
        t = 0.0
        for comp, mass in ((w.scomp, w.smass), (w.fcomp, None)):
            sel = comp > 0
            ids = comp[sel].astype(np.int64)
            m = mass[sel].astype(np.float64) if mass is not None else \
                w.fvol[sel].astype(np.float64) * mx.prop("rho", ids) / 1000.0
            t += mx.totals(ids, m).get(sp, 0.0) if sel.any() else 0.0
        return t + mx.ledger("gone").get(sp, 0.0)

    t0, l0 = grams(tin), grams(lead)
    e0, sh0, put = float(w.E.sum()), w.shed, 0.0
    plateau, seen_mush = [], False
    for i in range(420):
        d = 60.0 if i < 200 else -60.0
        w.E[c] += d
        put += d
        w.step()
        T = float(w.T()[c])
        if w.scomp[c] > 0 and w.fcomp[c] > 0:                 # a mush
            seen_mush = True
            plateau.append(T)
    assert seen_mush
    # the 60/40 solder is eutectic-like: it sits at 183 C while it melts and while it freezes
    assert all(182.0 < t < 191.0 for t in plateau), (min(plateau), max(plateau))
    assert abs(grams(tin) - t0) < 1e-3 and abs(grams(lead) - l0) < 1e-3     # no metal lost
    closed = (float(w.E.sum()) - e0) + (w.shed - sh0) + w.melted_J - w.frozen_J
    assert abs(closed - put) < 1e-3 * abs(put) + 1.0, (closed, put)
    assert w.melted_J > 1000.0 and w.frozen_J > 1000.0
    assert w.frozen_J == pytest.approx(w.melted_J, rel=0.05)                 # what it took, it gave back


def test_the_melt_gate_off_leaves_a_mixed_cell_alone():
    w = solder_cell()
    w.mix_laws = False
    c = (3, 2, 1)
    w.E[c] += 8000.0
    for _ in range(5):
        w.step()
    assert w.fcomp[c] == 0 and not hasattr(w, "melted_J")


# ── T1's declared diff: sand and slaked lime read their bed numbers ──────────────
def test_sand_and_slaked_lime_cells_read_the_bed_not_the_rock():
    w = brine_pot(None)
    mx = w.ensure_mixes()
    sand = mx.parse([("quartz", "grain", 64, 1.0)])
    rock = mx.parse([("quartz", "bound", 0, 1.0)])
    lime = mx.parse([("slaked_lime", "grain", 40, 1.0)])
    ids = np.array([sand, rock, lime])
    rho, k = mx.prop("rho", ids), mx.prop("k", ids)
    assert rho[0] == 1590.0 and rho[1] == 2650.0 and rho[2] == 450.0
    assert 0.25 < k[0] < 0.4 and k[1] == pytest.approx(7.7, rel=1e-3)        # sand 7.7 -> about 0.31
    assert k[2] < 0.08
    assert SP.form_porosity("quartz", GRAIN) == pytest.approx(0.40)


# ── the compiled twin of the settling sweep ────────────────────────────────────
def test_the_compiled_sweep_of_a_mixed_world_is_the_plain_path_bit_for_bit():
    out = []
    for compiled in (True, False):
        w = wet_pot_world()
        w.mix_compiled = compiled
        for _ in range(40):
            w.step()
        out.append(w)
    a, b = out
    for name in ("fl", "fvol", "fpot", "E", "fcomp", "mat", "smass", "scomp"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert a.mixes.ledger("gone") == b.mixes.ledger("gone")


def test_the_twin_is_used_only_with_a_mix_and_a_pure_world_keeps_its_own_compiled_sweep():
    w = brine_pot(None)
    assert not w._compiled_mixed()
    w = wet_pot_world()
    assert w._compiled_mixed() == (w.fused and True)


# ── the switch ledger ──────────────────────────────────────────────────────────
def test_the_world_laws_are_in_the_ledger_and_write_only_through_mixtures():
    assert callable(heatmix.boil_mixed) and callable(heatmix.melt_mixed)
    assert hasattr(mixtures, "book_boiled") and hasattr(mixtures, "book_melted")
