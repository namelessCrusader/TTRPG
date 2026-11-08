"""T9: a failed beam section becomes a PIECE (rubble), never dmg = 1 and never a
span change. The test that would have caught v1's design: it runs the REAL
`_relax_slack` (plain and fused) of the sandbox World, read only.

What it shows (spec T9-material v2, change 1 and section 2):
  * a mid-plank cell marked rubble gets slack -1 (it falls); the stubs on both
    sides keep slack >= 0 (they stand by their own span);
  * a mid-plank cell of a near-zero-span material (ash) stays held: span 0 is
    still held from the side, which is why "dmg = 1 gives span 0" could not work;
  * end to end: matbeam finds the plank overloaded by a lead block, the critical
    cell is made rubble and `_relax_slack` drops exactly that cell.
"""
import numpy as np
import pytest

from src.voxel import matbeam as B
from src.voxel import sim
from src.voxel.sim import AIR, ASH, LEAD, STONE, WOOD, World
from tests.test_matbeam import REC


def _plank_world(fused):
    w = World(40, 6, 12, voxel_cm=5)
    w.fused = fused
    w.fill(0, 40, 0, 6, 0, 1, STONE)                                   # floor
    w.fill(0, 3, 0, 6, 1, 6, STONE)                                    # two posts
    w.fill(37, 40, 0, 6, 1, 6, STONE)
    w.fill(3, 37, 0, 6, 5, 6, WOOD)                                    # a 34-cell plank over air
    return w


def _slack(w):
    return w._relax_slack(w.mat != AIR, frozenset())


@pytest.mark.parametrize("fused", [False, True])
def test_a_rubble_cell_mid_plank_falls_and_the_stubs_stay_held(fused):
    w = _plank_world(fused)
    slack = _slack(w)
    assert slack[3:37, 2, 5].min() >= 0                                # intact: all held
    w.rubble[20, :, 5] = True
    w._rub_gen += 1
    slack = _slack(w)
    assert slack[20, 2, 5] == -1                                       # the piece falls
    assert slack[19, 2, 5] >= 0 and slack[21, 2, 5] >= 0               # the stubs stand
    assert slack[3:20, 2, 5].min() >= 0 and slack[21:37, 2, 5].min() >= 0


@pytest.mark.parametrize("fused", [False, True])
def test_a_near_zero_span_ash_cell_mid_plank_stays_held(fused):
    w = _plank_world(fused)
    w.mat[20, :, 5] = ASH
    w.smass[20, :, 5] = w.smass[19, :, 5]
    slack = _slack(w)
    assert slack[20, 2, 5] >= 0                                        # held from the side
    assert slack[19, 2, 5] >= 0 and slack[21, 2, 5] >= 0


def test_the_fused_and_plain_relaxation_agree_on_the_broken_plank():
    out = []
    for fused in (False, True):
        w = _plank_world(fused)
        w.rubble[20, :, 5] = True
        w._rub_gen += 1
        out.append(_slack(w))
    assert np.array_equal(out[0], out[1])


def _world_arrays(w):
    dens = np.array(sim._DENS_ARR, float)                            # as long as the pad: a cell's id indexes it
    dens[:sim.NMAT] = [sim.SOLID[k][0] for k in range(sim.NMAT)]      # the old twelve, the dict's own floats
    solid = w.mat != AIR
    fill = np.clip(w.smass / np.maximum(dens[w.mat] * w.vox_l, 1e-9), 0.0, 1.0)
    return solid, fill, w.scale * 0.1


def _overloaded_world(fused):
    w = World(40, 6, 30, voxel_cm=5)
    w.fused = fused
    w.fill(0, 40, 0, 6, 0, 1, STONE)
    w.fill(0, 3, 2, 3, 1, 6, STONE)                                    # posts one lane wide
    w.fill(23, 26, 2, 3, 1, 6, STONE)
    w.fill(3, 23, 2, 3, 5, 6, WOOD, frac=0.3)                          # a thin plank, 20 cells
    w.fill(11, 15, 2, 3, 6, 26, LEAD)                                  # a lead block, about 110 kg
    return w


@pytest.mark.parametrize("fused", [False, True])
def test_a_failed_beam_section_becomes_a_piece_and_only_that_cell_falls(fused):
    w = _overloaded_world(fused)
    assert _slack(w)[3:23, 2, 5].min() >= 0                            # SPAN holds it
    solid, fill, h = _world_arrays(w)
    m = B.extract_members(solid, fill, w.mat, w.eid, None, None, w.rubble, None, None,
                          w._is_thing(w.eid), None, w.moving, h, rec=REC)
    W = np.where(solid, w.smass.astype(np.float64) / 1000.0 * B.G, 0.0)
    loads = B.accumulate_loads(m, W)
    res = B.analyse(m, REC, loads)
    assert loads.ground.sum() == pytest.approx(W.sum(), rel=1e-12)
    fail = [r for r in res.failing() if r["kind"] == "pinned"]
    assert len(fail) == 1 and fail[0]["mode"] in (B.M_BEND, B.M_SHEAR)
    x, y, z = fail[0]["cell"]
    assert z == 5 and 11 <= x < 15                                      # under the block
    before = (w.mat.copy(), w.smass.copy())
    w.rubble[x, y, z] = True                                            # what _break_member does
    w._rub_gen += 1
    slack = _slack(w)
    assert slack[x, y, z] == -1
    assert slack[x - 1, y, z] >= 0 and slack[x + 1, y, z] >= 0         # both stubs stand
    assert (w.mat == before[0]).all() and (w.smass == before[1]).all()  # no mass moved, no span touched
    # the stubs are now cantilevers; the rubble cell is no longer a member
    solid2, fill2, _ = _world_arrays(w)
    m2 = B.extract_members(solid2, fill2, w.mat, w.eid, None, None, w.rubble, None, None,
                           w._is_thing(w.eid), None, w.moving, h, rec=REC)
    assert all(r.kind == "cantilever" for r in m2.runs if r.mat[0] == WOOD)
    assert m2.orphan[x, y, z]


def test_dmg_never_enters_the_support_law():
    """A damage value below one changes no slack: the load check reads D, the
    piece state is how a failure shows (F11)."""
    w = _plank_world(False)
    base = _slack(w)
    dmg = np.full(w.shape, 0.9, np.float32)                            # never given to the World
    solid, fill, h = _world_arrays(w)
    m = B.extract_members(solid, fill, w.mat, w.eid, dmg, None, w.rubble, None, None,
                          w._is_thing(w.eid), None, w.moving, h, rec=REC)
    assert m.runs
    assert np.array_equal(base, _slack(w))
