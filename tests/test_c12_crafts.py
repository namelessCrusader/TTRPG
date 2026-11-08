"""CRAFT PROOFS FROM GENERAL LAWS (wave 13c, lane C12).

Every test here lays out matter and runs the laws that exist. No law knows a craft word
(the scenes live in src/voxel/scenes_craft.py). Proofs that need a law which does not exist
yet are xfail(strict) with the missing law in the reason: when the law lands, the test
must flip to a pass, and strict makes that visible.

Slow ones (a 10 minute cup, a bar heated to a profile) are marked in the report as SLOW ids.
"""
import numpy as np
import pytest

from src.voxel import lots
from src.voxel import species as SP
from src.voxel.scenes_craft import (BRONZE, COPPER, bar, brine_cup, file_removal, mix_density,
                                    seed_lot, solder_sweat, SWEAT_CELL)
from src.voxel.scenes_mixtures import BRINE20, BRINE10


# ── P1: bronze against copper ────────────────────────────────────────────────────
def _alpha(w, rows_mix):
    i = np.array([rows_mix])
    return float(w.mixes.prop("k", i)[0] / (w.mixes.prop("rho", i)[0] * w.mixes.prop("c", i)[0]))


def _crossing_times(rows, thresholds=(30.0, 40.0, 60.0), cap_s=120.0):
    """Seconds until the cell 6 from the hot end reaches each temperature (bar held at 400 C)."""
    w = bar(rows)
    probe = (3 + 6, 3, 1)
    mid = int(w.scomp[probe])
    hit = {}
    for i in range(1, int(cap_s / 0.025)):
        w.step()
        T = float(w.T()[probe])
        for th in thresholds:
            if th not in hit and T >= th:
                hit[th] = i * 0.025
        if len(hit) == len(thresholds):
            break
    return w, mid, hit


def test_a_bronze_bar_heats_at_the_speed_its_own_diffusivity_says_and_no_law_knows_bronze():
    """T1 P1. Two bars, one end held at 400 C. The prediction is alpha = k / (rho c) from the
    mix rule's own outputs for each bar (about 8 times apart); the measured ratio of the times
    to reach the same temperature at the same place must agree. Side losses to the stone and
    the air are the same for both and make the slow bar a few percent slower."""
    wc, mc, tc = _crossing_times(COPPER)
    wb, mb, tb = _crossing_times(BRONZE, cap_s=200.0)
    ratio_alpha = _alpha(wc, mc) / _alpha(wb, mb)
    assert 7.0 < ratio_alpha < 9.5                       # the rule: about 8 times slower
    assert len(tc) == 3 and len(tb) == 3
    for th in (30.0, 40.0, 60.0):
        r = tb[th] / tc[th]
        assert 0.85 * ratio_alpha < r < 1.2 * ratio_alpha, (th, r, ratio_alpha)
    # copper and bronze are the mix rule's numbers, not a table row of a bronze material
    assert mix_density(wb, mb) == pytest.approx(0.88 * 8960 + 0.12 * 7310, rel=0.03)


# ── P2: brine layers ───────────────────────────────────────────────────────────────
def _salt_fraction(w, z):
    i = np.array([w.fcomp[2, 2, z]])
    if i[0] == 0:
        return 0.0
    rho = float(w.mixes.prop("rho", i)[0])
    t = w.mixes.totals(i, np.array([float(w.fvol[2, 2, z]) * rho / 1000.0]))
    return t.get(SP.BY_NAME["sodium_chloride"], 0.0) / sum(t.values())


def _species_g(w, name):
    ids = w.fcomp[w.fcomp > 0].astype(np.int64)
    m = w.fvol[w.fcomp > 0].astype(np.float64) * w.mixes.prop("rho", ids) / 1000.0
    return w.mixes.totals(ids, m).get(SP.BY_NAME[name], 0.0)


@pytest.mark.parametrize("low", [True, False])
def test_brine_ends_under_fresh_water_and_stays_there_for_ten_minutes(low):
    """T1 P2. 20 percent brine UNDER fresh water stays under it; brine put OVER fresh water
    sinks through it. Either way the cup ends as 20 percent brine below and pure water above,
    with the salt and the water to the f32 bound. The density is the mix table's (CRC 1.148)."""
    w, m = brine_cup(low)
    assert mix_density(w, m) == pytest.approx(1148.0, rel=0.003)
    salt0, water0 = _species_g(w, "sodium_chloride"), _species_g(w, "water")
    for _ in range(24000):                                   # 10 sim minutes
        w.step()
    assert _salt_fraction(w, 1) == pytest.approx(0.20, abs=0.002)
    assert _salt_fraction(w, 2) < 0.002                       # no smearing upward
    assert _salt_fraction(w, 3) == 0.0
    assert _species_g(w, "sodium_chloride") == pytest.approx(salt0, rel=1e-5)
    assert _species_g(w, "water") == pytest.approx(water0, rel=1e-5)
    for nm in ("quant", "f32", "recast"):
        assert w.mixes.ledger_total(nm) == 0.0


# ── P6: sweating solder ────────────────────────────────────────────────────────────
def _split(w, comp, mass):
    t = w.mixes.totals(np.array([comp]), np.array([mass]))
    s = sum(t.values())
    return t.get(SP.BY_NAME["tin"], 0.0) / s, s


def test_solder_held_between_solidus_and_liquidus_splits_into_tin_rich_liquid_and_lead_richer_solid():
    """T1 P6, the composition half. 80/20 lead-tin held at 195 C is mostly solid: the lever
    rule on T4's digitised diagram gives a few percent of liquid holding about 55 to 60 percent
    tin, and the solid that is left holds less than the 20 percent it began with. The tin and
    the lead are the same in total."""
    w = solder_sweat(195.0)
    C = SWEAT_CELL
    tin0 = 0.2 * float(w.smass[C])
    g0 = float(w.smass[C])
    for _ in range(300):
        w.step()
    rho_l = float(w.mixes.prop("rho", np.array([w.fcomp[C]]))[0])
    m_liq = float(w.fvol[C]) * rho_l / 1000.0
    x_liq, _ = _split(w, w.fcomp[C], m_liq)
    x_sol, _ = _split(w, w.scomp[C], float(w.smass[C]))
    assert 0.03 < m_liq / g0 < 0.10                      # 5 to 10 percent of the voxel is liquid
    assert 0.52 < x_liq < 0.63                           # tin-rich (diagram: about 55 percent)
    assert x_sol < 0.19                                  # the solid left is lead-richer
    assert x_liq * m_liq + x_sol * float(w.smass[C]) == pytest.approx(tin0, rel=2e-3)


@pytest.mark.xfail(strict=True, reason="MISSING LAW: the liquid phase of a mushy solid does not "
                   "leave it. A porous solid whose liquid fraction passes a retention capacity "
                   "must shed the surplus to open neighbours (a liquid-phase expulsion law, "
                   "Darcy flow through the solid skeleton). Today the liquid stays in the "
                   "cell's fluid slot for ever, so solder never 'sweats' out of the joint.")
def test_the_tin_rich_liquid_leaves_the_solder_voxel():
    w = solder_sweat(195.0)
    for _ in range(600):
        w.step()
    out = sum(float(w.fvol[x, 2, 1]) for x in (2, 4))
    assert out > 0.05

# ── P8: seeds sorted in brine ──────────────────────────────────────────────────────
def test_sound_seeds_sink_in_brine_and_the_hollow_ones_float_and_nothing_is_lost():
    """T1 P8. The brine density is read from the cup's own mix table, not typed in."""
    lot, sound = seed_lot()
    w, m = brine_cup(True)
    rho = mix_density(w, m)
    floaters, sinkers = lots.float_split(lot, rho)
    assert floaters["n"] + sinkers["n"] == lot["n"] == 1000        # integer counts, exact
    assert (sinkers["density"] > rho).all() and (floaters["density"] < rho).all()
    hit = sound[sinkers["id"]].mean()                              # share of sinkers that are sound
    assert hit > 0.97                                              # the sorting works
    assert sound[sinkers["id"]].sum() / sound.sum() > 0.95         # and few sound seeds are lost
    assert np.isclose(np.sum(floaters["mass"]) + np.sum(sinkers["mass"]), np.sum(lot["mass"]),
                      rtol=1e-12, atol=0.0)
    # fresh water does not sort: the same lot all sinks (a negative control)
    f0, s0 = lots.float_split(lot, 998.2)
    assert s0["n"] > 0.9 * lot["n"]


# ── the file test (hand-set temper) ────────────────────────────────────────────────
def test_a_file_takes_less_off_a_harder_piece_and_skates_on_one_as_hard_as_itself():
    """T9 section 6. THE TEMPER IS HAND-SET (a multiplier on H), because no heat-history law
    exists: this proves the file's removal follows hardness, not that a quench makes a temper.
    Removal falls as 1/H while the file is harder than the piece, then drops about 30 times
    when the piece is as hard as the file."""
    soft, mid, hard = file_removal(1.0), file_removal(4.0), file_removal(6.0)
    assert soft == pytest.approx(4.0 * mid, rel=1e-9) and soft == pytest.approx(6.0 * hard, rel=1e-9)
    at_file, harder = file_removal(7.5), file_removal(9.0)
    assert hard / harder > 50.0                                    # a piece harder than the file
    assert at_file > harder
    assert file_removal(1.0, strokes=40) == pytest.approx(2.0 * soft, rel=1e-12)   # strokes add


# ── missing general laws (strict xfail: the day the law lands the test must pass) ─────
@pytest.mark.xfail(strict=True, reason="MISSING LAW: the ice law (wetworld._wet_ice) skips mixed "
                   "cells (D6). It must read the mix's freezing point (mixtures.freeze_point / "
                   "mixphase.freeze_split) so 10 percent brine freezes at -10 C and not at -5 C.")
def test_salt_water_freezes_below_its_own_freezing_point():
    from src.voxel.materials import AMBIENT, STONE, AIR
    from src.voxel.wetworld import WetWorld
    w = WetWorld(10, 10, 8, voxel_cm=10)
    w.fill(0, 10, 0, 10, 0, 1, STONE)
    w.fill(2, 5, 2, 5, 1, 4, STONE)
    w.fill(3, 4, 3, 4, 1, 4, AIR)
    w.pour(3, 3, 1, None, w.cap * 0.8, mix=w.ensure_mixes().parse(BRINE10))
    w.enable_wet(rh=0.5)
    w.E[3, 3, 1] = (-10.0 - AMBIENT) * w.heat_capacity()[3, 3, 1]
    for _ in range(5):
        w._wet_ice(1.0)
    assert w.ice_grams() > 1.0


@pytest.mark.xfail(strict=True, reason="MISSING LAW: a SORPTION / reaction row runner in the lattice "
                   "(rows of rxn.py are pure; the world runs none) plus a wash that moves only "
                   "DISSOLVED species out of a solid. Needed for tannin DISSOLVED + collagen BOUND "
                   "-> tannin BOUND (T1 P5) and for germination.")
def test_washing_a_tanned_hide_keeps_the_bound_tannin():
    from src.voxel.rxn import catalogue
    assert any("tann" in k or "sorb" in k for k in catalogue())
