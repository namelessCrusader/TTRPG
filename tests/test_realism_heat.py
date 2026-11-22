"""Realism tests for heat: how fast hot things go cold, how far heat travels
through a solid, and whether the conduction law can ever make a cell hotter
than what heats it (realism gap G5 in docs/realism-references.md; tests H-T1,
H-T2, H-T4 and H-T9 of section 4.2). Every band in a docstring is the
document's, with its source number.

THE CLOCK IS MINUTES LONG. A glowing bar takes ten minutes to go dark, which is
24,000 ticks, too many to step a whole world through in a test. So these tests
FAST-FORWARD a world where nothing moves (`_heat_only`): each tick runs the
laws that move heat, in the order `step` runs them, plus the thermostats and
the tick count `step` keeps. A bar on two stones or a stick in a stand moves
nothing else, so this is the same clock, not a shortcut through it; a test
that also checks it against `step` says so.

Each test is under ~1 GB and ~20 s.
"""
import numpy as np
import pytest

from src.voxel.materials import (AIR, AMBIENT, CHAR, IRON, LEAD, MLEAD,
                                 O2_PER_L, STONE, TICK_S, WOOD)
from src.voxel.sim import World

S = 1.0 / TICK_S                 # ticks in a second
MIN = 60.0 * S                   # ticks in a minute


def _heat_only(w, ticks, each=None):
    """Run `ticks` ticks of the heat laws alone (see the module docstring).
    `each(w)` runs after every tick and may return True to stop early."""
    for _ in range(ticks):
        if w.o2 is None:                         # as `step` does on its first tick
            w.o2 = np.full(w.shape, O2_PER_L * w.vox_l, np.float32)
        w._tick_start()
        hot = w._hot_window()
        if hot is not None:
            w._law_conduct(hot)
            risen = w._hot_window()
            if risen is not None:
                w._law_rise(risen)
        w._air_regions()
        if w._busy(w.E):
            w._law_burn()
            w._law_boil()
            w._law_melt()
        if w.thermostats:
            C = w.heat_capacity()
            for (tx, ty, tz, tT) in w.thermostats:
                w.E[tx, ty, tz] = (tT - AMBIENT) * C[tx, ty, tz]
        w._tick_end()
        w.tick += 1
        if each is not None and each(w):
            return


def _bar_on_stones(cm, n=8):
    """An iron bar 1 x 1 x n voxels on two stone blocks, in open air."""
    w = World(n + 6, 5, 6, voxel_cm=cm)
    w.fill(0, n + 6, 0, 5, 0, 1, STONE)
    w.fill(3, 4, 2, 3, 1, 2, STONE)
    w.fill(n + 2, n + 3, 2, 3, 1, 2, STONE)
    w.fill(3, n + 3, 2, 3, 2, 3, IRON)
    return w


def _cooling_times(cm, marks):
    """Seconds for the middle of an 800 °C bar to fall below each mark."""
    w = _bar_on_stones(cm)
    bar = (slice(3, 11), 2, 2)
    w.E[bar] = (800.0 - AMBIENT) * w.heat_capacity()[bar]
    mid = (6, 2, 2)
    out = {}

    def watch(w):
        T = float(w.T()[mid])
        for m in marks:
            if m not in out and T < m:
                out[m] = w.tick / S
        return len(out) == len(marks)
    _heat_only(w, int(40 * MIN), watch)
    return out


def test_H_T1_A_GLOWING_BAR_GOES_DARK_in_minutes_not_seconds():
    """H-T1 (sources 44, 45; row H21): an iron bar 1 x 1 x 8 voxels on two
    stones in open air, at 5 cm and at 10 cm, set to 800 °C. 800 -> 500 °C takes about
    680 s for a 50 mm square bar (twice a 25 mm round bar's 340 s, a lumped
    estimate); accept 170-1400 s. The glow (above 525 °C, the Draper point)
    lasts more than 2 minutes. Before G5 it went dark in at most 2.6 s. (The
    60 °C mark would take hours of ticks; it is not measured.)

    At 10 cm the same 1 x 1 x 8 bar is a 100 mm bar, holding twice the heat
    per square metre of skin: a lumped bar cools in proportion to its
    thickness, so the doc's "5 and 10 cm agree within 25%" is read as the
    10 cm bar taking twice as long, within 25% (1.5-2.5x). Reduced: compared
    on the first 100 K (800 -> 700 °C), where both are still lumped, so the
    10 cm bar costs 6,000 ticks rather than 30,000."""
    t5 = _cooling_times(5, (700.0, 525.0, 500.0))
    assert 500.0 in t5, f"the bar never cooled to 500 °C: {t5}"
    assert 170.0 <= t5[500.0] <= 1400.0, \
        f"800 -> 500 °C took {t5[500.0]:.0f} s (band 170-1400 s)"
    assert t5[525.0] > 120.0, f"it glowed for only {t5[525.0]:.0f} s"
    t10 = _cooling_times(10, (700.0,))
    assert 700.0 in t10, f"the 10 cm bar never cooled to 700 °C: {t10}"
    ratio = t10[700.0] / t5[700.0]
    assert 1.5 <= ratio <= 2.5, \
        f"a bar twice as thick took {ratio:.2f}x as long (lumped: 2x +- 25%)"


def test_H_T4_A_HAND_ON_THE_COLD_END():
    """H-T4 (sources 23, 25, 49, 41): (a) an iron bar 12 voxels long, one end
    held at 1100 °C for 3 min: 40 cm from the hot end it is still under 45 °C
    (textbook conduction gives under +1 K). (b) a wooden stick 6 voxels long,
    its top held at 250 °C (below ignition) for 5 min: 15 cm below the top it
    is under 30 °C (heat reaches only 31-35 mm behind a char front). Before
    G5 the decay lengths were 12 cm (iron) and 6.8 cm (wood), reached in
    seconds. At 5 cm."""
    w = World(18, 5, 5, voxel_cm=5)
    w.fill(0, 18, 0, 5, 0, 1, STONE)
    w.fill(4, 5, 2, 3, 1, 2, STONE)
    w.fill(13, 14, 2, 3, 1, 2, STONE)
    w.fill(2, 14, 2, 3, 2, 3, IRON)
    w.thermostats = [(2, 2, 2, 1100.0)]
    _heat_only(w, int(3 * MIN))
    T = w.T()
    assert T[2, 2, 2] > 1000.0, "precondition: the end is held hot"
    assert T[10, 2, 2] < 45.0, \
        f"40 cm from a 1100 °C end, iron is at {T[10, 2, 2]:.1f} °C"

    w = World(5, 5, 9, voxel_cm=5)
    w.fill(0, 5, 0, 5, 0, 1, STONE)
    w.fill(2, 3, 2, 3, 1, 7, WOOD)
    w.thermostats = [(2, 2, 6, 250.0)]
    _heat_only(w, int(5 * MIN))
    T = w.T()
    assert T[2, 2, 3] < 30.0, \
        f"15 cm below a 250 °C top, wood is at {T[2, 2, 3]:.1f} °C"
    assert not w.burning().any(), "250 °C is below ignition: nothing burns"


def _one_conduct(w):
    w._law_conduct()
    return w.T()


def test_H_T9_CONDUCTION_never_overshoots_and_is_the_same_both_ways():
    """H-T9 (internal invariant, the second law): (a) an air cell at 600 °C
    with 1000 °C iron on its +x, +y, +z faces, and the mirror image: after one
    tick no cell is hotter than its hottest neighbour was, and both give the
    same result. Before: 1763 °C against 840 °C. (b) uniform 400 °C air at
    5 cm with a +-10 K checkerboard decays over 10 ticks; before, it grew
    x1.4 a tick."""
    def cube(sign):
        w = World(5, 5, 5, voxel_cm=5)
        c = (2, 2, 2)
        for ax in range(3):
            p = list(c)
            p[ax] += sign
            w.mat[tuple(p)] = IRON
            w.smass[tuple(p)] = 7800.0 * w.vox_l
        C = w.heat_capacity()
        w.E[c] = (600.0 - AMBIENT) * C[c]
        for ax in range(3):
            p = list(c)
            p[ax] += sign
            w.E[tuple(p)] = (1000.0 - AMBIENT) * C[tuple(p)]
        return w
    up, down = cube(+1), cube(-1)
    t_up, t_down = _one_conduct(up)[2, 2, 2], _one_conduct(down)[2, 2, 2]
    assert t_up <= 1000.0 and t_down <= 1000.0, \
        f"the air cell passed its hottest neighbour: {t_up:.1f}, {t_down:.1f} °C"
    assert t_up > 600.0, "and it did warm"
    assert abs(t_up - t_down) < 1e-3, \
        f"mirror images disagree: {t_up:.4f} against {t_down:.4f} °C"

    w = World(12, 12, 12, voxel_cm=5)
    i, j, k = np.indices(w.shape)
    board = np.where((i + j + k) % 2 == 0, 1.0, -1.0)
    w.E[...] = (380.0 + 10.0 * board) * w.heat_capacity()
    inner = (slice(2, -2),) * 3

    def amp(w):
        T = w.T()
        return float(np.abs(((T - 400.0) * board)[inner]).mean())
    last = amp(w)
    for _ in range(10):
        w._law_conduct()
        now = amp(w)
        assert now < last, f"the checkerboard grew: {last:.3f} -> {now:.3f} K"
        last = now


def _bed(w, x0, x1, y0, y1, z0, z1, T, m=STONE, frac=1.0):
    """A forge bed reduced to thermostat blocks held at `T` (the doc's "or
    thermostat blocks at 850 °C"): packed stone, or a heap of coals."""
    w.fill(x0, x1, y0, y1, z0, z1, m, frac=frac)
    w.thermostats += [(x, y, z, T) for x in range(x0, x1) for y in range(y0, y1)
                      for z in range(z0, z1)]


@pytest.mark.slow
def test_H_T2_THE_FORGE_MELTS_LEAD_and_heats_iron_to_forging():
    """H-T2 (sources 47, 48, 84, 87; rows H23, L-rows): (a) one LEAD voxel
    (about 1.4 kg at 5 cm) buried in a bed at 800-900 °C passes 327.5 °C and
    melts fully in 0.3-10 min (about 89 kJ at 0.5-3 kW absorbed); a plateau
    below 327 °C fails. (b) an iron bar 8 voxels long with its last 2 voxels
    in the hottest part of the bed reaches 950 °C within about 20 min (ASM: no
    more than 5 min per inch of section, about 10 min for 5 cm); a plateau
    below 900 °C fails. Before G5: lead flat at 258-263 °C, iron at ~256 °C.

    Reduced: the bed is thermostat blocks, as the doc allows: packed stone at
    850 °C for (a); for (b) "the hottest part of the bed" is a heap of coals
    (CHAR, half full) at 1250 °C, charcoal with a blower (H28: up to about
    1260 °C). Nothing can heat iron past the temperature of what heats it, so
    a bed AT 800-900 °C cannot bring it to 950 °C: the doc's (b) needs a
    hotter zone, and the six voxels of bar outside the fire radiate away
    much of what the two inside take in. Fast-forwarded (`_heat_only`).
    (Measured: 950 °C at 294 s with the coals at 1250 °C, and at 524 s, the
    ASM ~10 min, with them at 1100 °C, which costs 23 s of test; with packed
    stone at 1150 °C the iron plateaued at 859 °C: contact with stone is a
    poor stand-in for coals, which heat what is in them by radiation across
    their gaps.)"""
    w = World(7, 7, 7, voxel_cm=5)
    w.fill(0, 7, 0, 7, 0, 1, STONE)
    _bed(w, 2, 5, 2, 5, 1, 4, 850.0)
    w.thermostats = [t for t in w.thermostats if t[:3] != (3, 3, 2)]
    w.fill(3, 4, 3, 4, 2, 3, LEAD)
    lead0 = float(w.smass[3, 3, 2])
    out = {}

    def watch(w):
        if "first" not in out and w.fvol[3, 3, 2] > 0:
            out["first"] = w.tick / S
        if w.mat[3, 3, 2] != LEAD:
            out["all"] = w.tick / S
            return True
        out["peak"] = max(out.get("peak", 0.0), float(w.T()[3, 3, 2]))
        return False
    _heat_only(w, int(12 * MIN), watch)
    assert "first" in out, f"the lead never melted (peak {out.get('peak', 0):.0f} °C)"
    assert "all" in out, "the lead never melted through"
    assert 0.3 * 60 <= out["all"] <= 10 * 60, \
        f"fully molten at {out['all']:.0f} s (band 18-600 s)"
    assert abs(float(w.fvol[w.fl == MLEAD].sum()) * 10.66 - lead0) < 0.01 * lead0, \
        "and every gram of it is there, molten"

    w = World(14, 7, 7, voxel_cm=5)
    w.fill(0, 14, 0, 7, 0, 1, STONE)
    _bed(w, 9, 13, 2, 5, 1, 4, 1250.0, CHAR, 0.5)
    w.thermostats = [t for t in w.thermostats if t[:3] not in ((9, 3, 2), (10, 3, 2))]
    w.fill(3, 4, 3, 4, 1, 2, STONE)
    w.fill(3, 11, 3, 4, 2, 3, IRON)                  # last 2 voxels in the bed
    hot = {}

    def watch_iron(w):
        T = float(w.T()[10, 3, 2])
        hot["peak"] = max(hot.get("peak", 0.0), T)
        if T >= 950.0:
            hot["t"] = w.tick / S
            return True
        return False
    _heat_only(w, int(20 * MIN), watch_iron)
    assert "t" in hot, f"the buried iron plateaued at {hot['peak']:.0f} °C"


def test_WARM_AIR_RISES_AT_ITS_BUOYANT_SPEED_not_at_one_rate():
    """G7's plume, in SI (`_law_rise`): warm gas rises at about
    sqrt(g · dx · ΔT / T), so in one tick a cell hands up v · TICK_S / dx of
    its excess, never more than half (RISE_MAX). At 5 cm: air 5 K warm about
    0.07 m/s (a 3% share), 100 K warm about 0.4 m/s (20%), and flame gas
    1000 K warm would want 1.3 m/s, past the half a tick allows. It was 35%
    a tick whatever the difference: a degree of warmth rose as fast as a
    flame."""
    shares = {}
    for dT in (5.0, 100.0, 980.0):
        w = World(3, 3, 4, voxel_cm=5)
        C = w.heat_capacity()
        w.E[1, 1, 1] = dT * C[1, 1, 1]
        w._law_rise()
        shares[dT] = float(w.E[1, 1, 2]) / (dT * float(C[1, 1, 1]))
    v = {dT: (9.81 * 0.05 * dT / 293.15) ** 0.5 for dT in shares}
    assert abs(shares[5.0] - v[5.0] * TICK_S / 0.05) < 1e-4, shares
    assert abs(shares[100.0] - v[100.0] * TICK_S / 0.05) < 1e-4, shares
    assert shares[980.0] == pytest.approx(0.5, abs=1e-6), shares
    assert shares[5.0] < shares[100.0] < shares[980.0]


def test_H23_THE_FORGE_BED_GIVES_A_BAR_WHAT_A_GLOWING_BED_RADIATES():
    """H23 (row H23, the forge diagnosis of 2026-10-29): a packed bed of
    glowing coals (charcoal, 80% full) at 680 °C round a cold iron block hands
    it at least half of what two grey surfaces (0.9 and 0.8) exchange by
    radiation across a thin gap, e σ (T⁴ - T⁴) with e = 1/(1/e1 + 1/e2 - 1):
    30 kW/m², so at least 15 kW/m² on the bar's face. Measured: 8.0 kW/m²
    (12 W/m²K), for tin 4.4 kW/m² (6.7 W/m²K, whose ideal is 12 kW/m²).
    One tick of `_law_conduct`, the bar's gain over its face area.

    Wave 14a (lane C): it passes. The pore term was the bed's open share (0.2)
    times the DIMMER surface's emissivity; it is now the CAVITY view
    (`_cavity_view`: 1 - fill^N, N = cell / lump size) times the grey-body
    exchange of the two surfaces, over the exact T^4 difference. Was 8.0 kW/m2,
    now about 19 kW/m2 (ideal 30)."""
    from src.voxel.heat import _count_nb
    w = World(16, 12, 10, voxel_cm=5)
    w.fill(0, 16, 0, 12, 0, 1, STONE)
    w.fill(2, 14, 2, 10, 1, 7, CHAR, frac=0.8)
    w.fill(6, 10, 4, 8, 3, 5, IRON)
    bed = w.mat == CHAR
    w.E[bed] = (680.0 - AMBIENT) * w.heat_capacity()[bed]
    bar = w.mat == IRON
    area = float(_count_nb(w.mat == CHAR)[bar].sum()) * (0.1 * w.scale) ** 2
    e0 = float(w.E[bar].sum())
    w._law_conduct()
    flux = (float(w.E[bar].sum()) - e0) / TICK_S / area
    e = 1.0 / (1.0 / 0.9 + 1.0 / 0.8 - 1.0)
    ideal = e * 5.670374e-8 * ((680.0 + 273.15) ** 4 - (AMBIENT + 273.15) ** 4)
    assert flux >= 0.5 * ideal, \
        f"{flux / 1000:.1f} kW/m² on the bar, a glowing bed's radiation is {ideal / 1000:.1f}"
