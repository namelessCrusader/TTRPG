"""The forge (wave 14a, lane C): a bar in a bed sees a cavity, and the bellows is the push law.

H23 itself is in test_realism_heat.py. These tests guard the pieces around it: the cavity
view as a law (a dense cell has none, a packed bed some, a loose heap nearly all), that the
bed-to-bar radiation is the grey-body exchange of the real surfaces, that dense solids
are untouched, and that the bellows blast is the volume the board really pushed.
"""
import numpy as np
import pytest

from src.voxel.materials import AMBIENT, CHAR, IRON, STONE, TICK_S
from src.voxel.sim import World


def test_the_CAVITY_VIEW_is_0_for_a_dense_cell_and_grows_with_the_gaps():
    w = World(4, 4, 4, voxel_cm=5)
    dx = 0.05
    v = np.array([0.0, 0.01, 0.2, 0.5], np.float32).reshape(4, 1, 1) * np.ones((4, 4, 4), np.float32)
    out = w._cavity_view(v, dx)
    col = out[:, 0, 0]
    assert col[0] == 0.0
    assert col[1] == pytest.approx(0.04, abs=0.001)          # a small void stays small (about 4 v)
    assert col[2] == pytest.approx(1.0 - 0.8 ** 4, abs=1e-5)  # the packed bed: 0.59
    assert col[3] > 0.9 and np.all(np.diff(col) > 0)
    # a coarser cell holds more lumps in a ray: the view is nearer 1
    assert w._cavity_view(v, 0.2)[2, 0, 0] > 0.97
    assert w._cavity_view(np.zeros((2, 2, 2), np.float32), dx).max() == 0.0


def _bar_flux(bed_frac, bar_mat=IRON):
    from src.voxel.heat import _count_nb
    w = World(16, 12, 10, voxel_cm=5)
    w.fill(0, 16, 0, 12, 0, 1, STONE)
    w.fill(2, 14, 2, 10, 1, 7, CHAR, frac=bed_frac)
    w.fill(6, 10, 4, 8, 3, 5, bar_mat)
    bed, bar = w.mat == CHAR, w.mat == bar_mat
    w.E[bed] = (680.0 - AMBIENT) * w.heat_capacity()[bed]
    area = float(_count_nb(bed)[bar].sum()) * (0.1 * w.scale) ** 2
    e0 = float(w.E[bar].sum())
    w._law_conduct()
    return (float(w.E[bar].sum()) - e0) / TICK_S / area


def test_a_LOOSER_bed_gives_the_bar_more_by_radiation_and_a_solid_block_gives_none_extra():
    tight, packed, loose = _bar_flux(0.95), _bar_flux(0.8), _bar_flux(0.5)
    assert tight < packed < loose
    assert loose < 60000.0                                   # never more than a black body's 680 C (about 46 kW/m2 net)


def test_a_DENSE_solid_pair_has_no_pore_radiation():
    """Two full cells of stone and iron: the pore term is 0, only conduction crosses."""
    w = World(6, 6, 6, voxel_cm=5)
    w.fill(0, 6, 0, 6, 0, 1, STONE)
    w.fill(1, 3, 1, 5, 1, 3, STONE)
    w.fill(3, 5, 1, 5, 1, 3, IRON)
    assert float(w._cavity_view(w._surfaces((slice(None),) * 3).void, 0.05).max()) == 0.0


def test_the_BELLOWS_BLAST_IS_THE_AIR_THE_BOARD_PUSHED_and_hotter_follows():
    """The forge scene feeds the burn law the flow the solver carried out of the nozzle
    (no number typed in). A 60 cm x 60 cm board at 0.08 m/s sweeps 28.8 L/s; the solver's
    nozzle flow is within 10 percent of it, and a bed with that blast runs at least 100 K
    hotter than the same bed on its natural draught."""
    from src.voxel.scenes_forge import forge
    peaks = {}
    for use in (False, True):
        f = forge(bellows=use, speed=0.08)
        if use:
            f.start(1.0)
        flows = []
        for _ in range(400):
            f.step()
            flows.append(f.blast_lps)
        peaks[use] = f.peak
        if use:
            swept = 0.36 * 0.08 * 1000.0
            steady = float(np.median([q for q in flows if q > 0.0]))
            assert abs(steady - swept) < 0.1 * swept, (steady, swept)
    assert peaks[True] >= peaks[False] + 100.0, peaks
