"""Reference tests for chemistry: what acids eat, how fast, and what is left.

From docs/realism-references.md, section 4.5 (rows C1-C19 of table 2.5; the
source numbers are those of section 5.5 there). Each test names the real value
it checks and its pass range. Scenes are at voxel_cm 5, as section 4 asks.

Strength: potency is the share of stock by volume, the rest water. ACID is
vitriol, 98% sulfuric acid at potency 1, so potency 0.5 is about 63 wt% and
0.2 about 31 wt% (C12). WEAK_ACID is vinegar, 5% acetic acid at potency 1.

A test that needs a law the sim does not have yet is a strict expected failure
that says which law: it starts to pass, and fails the run, the day that law
lands, so the mark comes off with it.
"""
import numpy as np
import pytest

from src.voxel.materials import SOLID, TICK_S
from src.voxel.sim import (ACID, AIR, CHAR, FLESH, GLASS, IRON, LEAD, STONE, WATER,
                           WEAK_ACID, WOOD, World)


def _vat(mat, fluid, pot, rows=1, wall=GLASS):
    """A 4 x 4 slab of `mat`, `rows` voxels thick, inside a `wall` basin, under
    a one-voxel pool of `fluid` at potency `pot`. Only the slab's top face is
    wet: its sides and bottom touch the basin."""
    w = World(8, 8, 4 + rows, voxel_cm=5)
    w.fill(0, 8, 0, 8, 0, 1, wall)
    w.fill(1, 7, 1, 7, 1, 2 + rows, wall)
    w.fill(2, 6, 2, 6, 1, 1 + rows, mat)
    w.mat[2:6, 2:6, 1 + rows] = AIR
    w.smass[2:6, 2:6, 1 + rows] = 0.0
    for x in range(2, 6):
        for y in range(2, 6):
            w.pour(x, y, 1 + rows, fluid, w.cap, pot=pot)
    return w


def _grams(w, m):
    """Grams of material m on the lattice, summed exactly (float64)."""
    return float(w.smass[w.mat == m].astype(np.float64).sum())


def _run(w, ticks):
    for _ in range(ticks):
        w.step()


TOP_CM2 = 16 * 5.0 * 5.0          # the wet top face of a 4 x 4 slab at 5 cm


def _mg_per_cm2_h(grams, ticks):
    return grams * 1000.0 / TOP_CM2 / (ticks * TICK_S / 3600.0)


# ── C-T1 / C-T2: skin ──────────────────────────────────────────────────────────
def test_C_T1_vitriol_on_skin_removes_no_flesh():
    """C-T1 Vitriol on skin (high). A FLESH slab 4 x 4 x 2 under ACID at
    potency 1 (98 wt%) for 7,200 ticks (3 min).

    Real: an acid burn is coagulation necrosis: the dead layer stays in place,
    no tissue is removed. Full-thickness destruction takes more than 3 and up
    to 60 min (GHS Skin Corr. 1A, Packing Group II).
    Pass: no flesh voxel removed and 0 g of flesh lost. Before R1 a 5 cm voxel
    was gone in about 2-3 s.
    Not checked yet: the burn depth (0.1-1.3 mm at 3 min, >= 1.3 mm by 60 min).
    It needs a burn quantity in the body model; until body parts exist,
    rulings case 2 makes "no flesh removed" the whole pass.
    Sources: 119, 120, 121, 122, 123, 125."""
    w = _vat(FLESH, ACID, 1.0, rows=2)
    n0, g0 = int((w.mat == FLESH).sum()), _grams(w, FLESH)
    _run(w, 7200)
    assert int((w.mat == FLESH).sum()) == n0, "no flesh voxel may be removed"
    assert _grams(w, FLESH) == g0, "and no gram of flesh"


def test_C_T2_a_hand_soaked_in_vinegar_loses_nothing():
    """C-T2 Vinegar hand soak (high). A FLESH slab under WEAK_ACID at potency 1
    (5% acetic acid) for 2,000 ticks; extrapolated to 4 h it is still 0.

    Real: acetic acid at <= 20% is not classified for skin; 3.3% and 10% on
    rabbit skin for 4 h are not irritating (Health Canada).
    Pass: 0 g lost, no voxel removed. Before R1, about 2 mm/s.
    Source: 126."""
    w = _vat(FLESH, WEAK_ACID, 1.0, rows=2)
    n0, g0 = int((w.mat == FLESH).sum()), _grams(w, FLESH)
    _run(w, 2000)
    assert int((w.mat == FLESH).sum()) == n0
    assert _grams(w, FLESH) == g0, "vinegar takes nothing off a hand"


# ── C-T3 / C-T4: wood ──────────────────────────────────────────────────────────
def test_C_T3_a_vinegar_cask_holds_its_vinegar():
    """C-T3 Vinegar cask (high). An open wooden tub, one-voxel walls and floor,
    with a 3 x 3 x 3 cavity full of WEAK_ACID, for 2,000 ticks.

    Real: vinegar has no structural effect on wood on any game time scale;
    wooden tanks hold acids down to pH 1.5 for decades.
    Pass: wood loss < 0.1% of the wood per game day (extrapolated); no leak
    (the cavity still holds all of the vinegar, within 1 ml). Before R1 a
    wall went in about 25 s.
    Sources: 128, 129."""
    w = World(9, 9, 7, voxel_cm=5)
    w.fill(0, 9, 0, 9, 0, 1, STONE)
    w.fill(2, 7, 2, 7, 1, 5, WOOD)
    w.mat[3:6, 3:6, 2:5] = AIR
    w.smass[3:6, 3:6, 2:5] = 0.0
    for x in range(3, 6):
        for y in range(3, 6):
            for z in range(2, 5):
                w.pour(x, y, z, WEAK_ACID, w.cap)
    wood0, ml0 = _grams(w, WOOD), w.total_fluid(WEAK_ACID)
    _run(w, 2000)
    per_day = (wood0 - _grams(w, WOOD)) / 2000 * (86400.0 / TICK_S) / wood0
    assert per_day < 1e-3, f"the cask loses {per_day:.2%} of its wood a day"
    inside = float(w.fvol[3:6, 3:6, 2:5][w.fl[3:6, 3:6, 2:5] == WEAK_ACID].sum())
    assert inside >= ml0 - 1.0, f"the cask leaks: {inside:.0f} of {ml0:.0f} ml inside"


def test_C_T4_vitriol_on_a_plank_is_slow():
    """C-T4 Vitriol on a plank (high). A 5 cm wood slab 4 x 4 x 1 under ACID at
    potency 1, 2,400 ticks (1 min).

    Real: vitriol dehydrates wood to black carbon. 72% acid at 30 °C takes
    0.25-0.85 mm wood particles apart in <= 30 min: >= 0.014 mm/min (130).
    Pass: recession > 0 and <= 1 mm/min (the reference's generous bound); no
    wood voxel removed; at the measured rate the 5 cm plank holds >= 10 game
    minutes. Before R1: about 20 mm/s, a voxel gone in 2.5 s.
    Sources: 94, 130, 128."""
    w = _vat(WOOD, ACID, 1.0)
    g0 = _grams(w, WOOD)
    _run(w, 2400)
    lost = g0 - _grams(w, WOOD)
    cm3 = lost / (SOLID[WOOD][0] / 1000.0)               # density g/L -> g/cm³
    mm_per_min = 10.0 * cm3 / TOP_CM2 / (2400 * TICK_S / 60.0)
    assert 0.0 < mm_per_min <= 1.0, f"recession {mm_per_min:.4f} mm/min"
    assert (w.mat[2:6, 2:6, 1] == WOOD).all(), "the plank is still there"
    assert 50.0 / mm_per_min >= 10.0, "it holds for at least 10 game minutes"


def test_C_T4_vitriol_leaves_wood_as_char_and_leaves_char_alone():
    """C-T4, the residue, and C14 (high). Wood under ACID at potency 1, already
    eaten down to just above a quarter of a full voxel.

    Real: concentrated sulfuric acid dehydrates cellulose to carbon; 72% acid
    leaves 21.8-25.9% of the wood as acid-insoluble residue (130). Only HOT
    concentrated acid oxidises carbon; at room temperature charcoal is left
    alone (94, C14).
    Pass: every voxel turns to CHAR, not AIR, holding >= 20% of a full wood
    voxel; then 1,200 more ticks take 0 g of that char. Before R1 the wood
    dissolved "clean, no ash" and vitriol ate charcoal at about 80 mm/s.
    Sources: 94, 130."""
    w = _vat(WOOD, ACID, 1.0)
    full = SOLID[WOOD][0] * w.vox_l
    w.smass[2:6, 2:6, 1] = 0.25 * full + 0.002
    _run(w, 600)
    assert (w.mat[2:6, 2:6, 1] == CHAR).all(), "what vitriol leaves of wood is char"
    assert (w.smass[2:6, 2:6, 1] >= 0.2 * full).all(), ">= 20% of the wood stays"
    char = _grams(w, CHAR)
    _run(w, 1200)
    assert _grams(w, CHAR) == char, "cold vitriol does not eat charcoal"


@pytest.mark.xfail(strict=True, reason="needs a surface state per face: a whole "
                   "5 cm voxel cannot show the thin black film that forms first")
def test_C_T4_the_plank_surface_turns_black_within_seconds():
    """C-T4 (high): the surface turns black within seconds (94, 130). The sim
    chars a voxel only once most of it is eaten, which takes hours."""
    w = _vat(WOOD, ACID, 1.0)
    _run(w, 400)
    assert (w.mat[2:6, 2:6, 1] == CHAR).any()


# ── C-T5: mixing ──────────────────────────────────────────────────────────────
def _cup(lower, upper):
    """A one-voxel IRON cup at 5 cm: `lower` half fills the bottom cell and
    `upper` is poured above it, so it falls in and the two mix. Iron, not
    glass: the blend gets hot enough to crack glass, and steel is passive in
    acid this strong (C5), so nothing but the mixing makes heat."""
    w = World(5, 5, 6, voxel_cm=5)
    w.fill(0, 5, 0, 5, 0, 1, IRON)
    w.fill(1, 4, 1, 4, 1, 5, IRON)
    w.mat[2, 2, 1:5] = AIR
    w.smass[2, 2, 1:5] = 0.0
    half = 0.5 * w.cap
    w.pour(2, 2, 1, lower, half)
    w.pour(2, 2, 2, upper, half)
    return w, half


@pytest.mark.parametrize("lower, upper", [(ACID, WATER), (WATER, ACID)],
                         ids=["water_onto_vitriol", "vitriol_into_water"])
def test_C_T5_water_and_vitriol_mix_HOT(lower, upper):
    """C-T5 cases A and B (high). Equal volumes of WATER and ACID (potency 1),
    one poured onto the other, 200 ticks.

    Real: the integral heat of mixing of sulfuric acid (Felder, Rousseau &
    Bullard Table B.11) gives off about 0.86 kJ per ml of 98% acid at 1:1
    (about 3.1 mol water per mol acid).
    Pass: 0.86 kJ/ml +- 30% (0.60-1.12), whichever way it is poured, and the
    two become one liquid, acid, at potency about 0.5. Before R1: no heat.
    The energy released is the most the world's total rose over the run:
    the total then drifts down about 5% in 200 ticks through a heat path
    the energy ledger does not count (not the mixing; seen with any hot cup).
    Sources: 139, 94."""
    w, ml = _cup(lower, upper)
    e0 = w.total_energy()
    released = 0.0
    for _ in range(200):
        w.step()
        released = max(released, w.total_energy() - e0)
    kj_per_ml = released / 1000.0 / ml
    assert 0.60 <= kj_per_ml <= 1.12, f"{kj_per_ml:.2f} kJ per ml of vitriol"
    wet = w.fvol > 1.0
    assert set(np.unique(w.fl[wet]).tolist()) == {ACID}, "one liquid, and it is acid"
    pot = float((w.fpot[wet] * w.fvol[wet]).sum() / w.fvol[wet].sum())
    assert 0.45 <= pot <= 0.55, f"half strength, got {pot:.2f}"


@pytest.mark.xfail(strict=True, reason="needs a mixing rate and a boiling law for "
                   "a hot acid blend: today the blend is instant and never boils")
def test_C_T5_water_onto_vitriol_spits_and_vitriol_into_water_does_not():
    """C-T5 cases A and B (high): water poured onto acid makes a hot top layer
    that boils and throws acid out; acid poured into water does not (94, 138)."""
    w, _ml = _cup(ACID, WATER)
    thrown = 0.0
    for _ in range(200):
        w.step()
        thrown = max(thrown, sum(d[6] for d in w.drops))
    outside = float(w.fvol.sum()) - float(w.fvol[2, 2, 1:5].sum())
    assert thrown > 1.0 or outside > 1.0, "the blend boils and throws acid out"


@pytest.mark.xfail(strict=True, reason="ACID and WEAK_ACID are not MISCIBLE: one "
                   "liquid holding two reagents needs a potency per reagent")
def test_C_T5_vinegar_and_vitriol_blend_into_one_liquid():
    """C-T5 case C (high): WEAK_ACID poured onto ACID becomes one liquid (94).
    Today the vinegar floats on the vitriol as a separate layer."""
    w, _ml = _cup(ACID, WEAK_ACID)
    _run(w, 200)
    assert not ((w.fl == WEAK_ACID) & (w.fvol > 1.0)).any()


# ── C-T6: iron ─────────────────────────────────────────────────────────────────
def test_C_T6_steel_is_passive_in_strong_vitriol_and_eaten_by_dilute():
    """C-T6 Acid on iron, a concentration sweep (high). Three 4 x 4 x 1 IRON
    slabs under ACID at potency 1.0 (98 wt%), 0.5 (about 63 wt%) and 0.2
    (about 31 wt%), 2,400 ticks each.

    Real: carbon steel in 60-100% sulfuric acid below 38 °C loses < 0.02 in/yr
    (< 0.05 mg/(cm²·h)) (133); below about 60% the acid is "extremely
    corrosive" (133); mild steel in 1 M HCl loses 0.7-5.7 mg/(cm²·h) (131, 132).
    Pass: potency 1.0 and 0.5 <= 0.05 mg/(cm²·h); potency 0.2 in 0.7-600
    mg/(cm²·h); the dilute rate >= 10x the concentrated one. Before R1:
    432,000 mg/(cm²·h) at potency 1, linear in potency.
    Not checked yet: the visible H2, which needs gas species (roadmap 107 B).
    Sources: 131, 132, 133."""
    rate = {}
    for pot in (1.0, 0.5, 0.2):
        w = _vat(IRON, ACID, pot)
        g0 = _grams(w, IRON)
        _run(w, 2400)
        rate[pot] = _mg_per_cm2_h(g0 - _grams(w, IRON), 2400)
    assert rate[1.0] <= 0.05 and rate[0.5] <= 0.05, f"strong acid passivates: {rate}"
    assert 0.7 <= rate[0.2] <= 600.0, f"dilute acid eats steel: {rate}"
    assert rate[0.2] >= 10.0 * max(rate[1.0], rate[0.5]), f"{rate}"


# ── C-T11: the vessels acid is kept in ─────────────────────────────────────────
@pytest.mark.parametrize("wall", [GLASS, LEAD], ids=["glass", "lead"])
def test_C_T11_vitriol_stands_in_glass_and_lead(wall):
    """C-T11 Vitriol in glass and lead vessels (low; a regression guard). A
    vat of `wall` holding ACID at potency 1 for 2,000 ticks.

    Real: sulfuric acid is stored in glass; lead resists it (94).
    Pass: 0 g of the vessel lost. Source: 94."""
    w = _vat(wall, ACID, 1.0, wall=wall)
    g0 = _grams(w, wall)
    _run(w, 2000)
    assert _grams(w, wall) == g0
