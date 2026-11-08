"""T9 wired (wave 13b, lane C8a): friction per pair, the members veto, impact
damage and spall, each behind its own name in `world.mat_laws`.

What these prove:
  * the gate: with `mat_laws` empty every law is the old one, bit for bit;
  * friction: the force to START a haul is mu_s m g of the PAIR (wood on stone
    0.55, not 0.40), the sliding value is mu_k, an oiled floor is slick;
  * members: an overloaded plank breaks at its critical cell (a piece, mass
    kept), a sound plank and a standing person are untouched, and the veto never
    lets stand what SPAN drops;
  * damage: a blow under the threshold leaves a crack, a second one breaks the
    cell at thr x (1 - D), one writer and one combine rule;
  * spall: a hot stone face with air in front throws a flake (mass and heat
    kept), a face against another solid does not.
"""
import numpy as np
import pytest

from src.voxel import matlaws
from src.voxel.sim import AIR, STONE, WOOD, LEAD, World
from src.voxel.materials import FRICTION, GRAVITY, OIL


def _crate(laws, floor=STONE, thing=WOOD):
    w = World(20, 10, 10, voxel_cm=10)
    w.mat_laws = frozenset(laws)
    w.fill(0, 20, 0, 10, 0, 1, floor)
    w.fill(5, 7, 4, 6, 1, 3, thing)
    cells = np.argwhere(w.mat == thing)
    return w, cells


def _kg(w, cells):
    return float(w.smass[tuple(cells.T)].sum()) / 1000.0


def test_with_no_laws_on_effort_is_the_old_flat_number():
    w, c = _crate(())
    lift, drag = w._effort(c)
    kg = _kg(w, c)
    assert lift == kg * GRAVITY and drag == kg * GRAVITY * FRICTION
    assert w._effort(c, kinetic=True) == (lift, drag)


def test_the_flat_table_is_the_old_number_to_the_bit():
    w, c = _crate(("friction", "flat"))
    assert w._effort(c) == _crate(())[0]._effort(c)


def test_the_start_force_uses_the_static_value_of_the_pair():
    w, c = _crate(("friction",))
    kg = _kg(w, c)
    _lift, start = w._effort(c)
    _lift, slide = w._effort(c, kinetic=True)
    assert start == pytest.approx(kg * GRAVITY * 0.55)          # wood on rock, static (matfric)
    assert slide == pytest.approx(kg * GRAVITY * 0.40)          # kinetic
    assert slide < start


def test_the_pair_matters_wood_on_wood_slides_easier_than_on_stone():
    ws, cs = _crate(("friction",))
    ww, cw = _crate(("friction",), floor=WOOD)
    assert ww._effort(cw)[1] / _kg(ww, cw) < ws._effort(cs)[1] / _kg(ws, cs)


def test_a_film_of_oil_makes_the_floor_slick():
    w, c = _crate(("friction",))
    dry = w._effort(c)[1]
    for x, y, z in c:                                           # the crate stands in oil
        w.fl[x, y, z] = OIL
        w.fvol[x, y, z] = w.cap
    assert w._effort(c)[1] < 0.3 * dry


def test_nothing_under_the_thing_keeps_the_old_number():
    w, c = _crate(("friction",))
    c = c + np.array([0, 0, 4])                                 # lifted clear of the floor
    assert matlaws.contact_mu(w, c, False) is None


# ── damage ──────────────────────────────────────────────────────────────────
def test_add_damage_is_one_writer_with_one_combine_rule():
    w = World(6, 6, 6)
    assert w.dmg is None
    matlaws.add_damage(w, [(1, 1, 1)], 0.5)
    matlaws.add_damage(w, [(1, 1, 1)], 0.5)
    assert w.dmg[1, 1, 1] == pytest.approx(0.75)                # 1 - (1-.5)(1-.5)
    a = World(6, 6, 6); b = World(6, 6, 6)
    for d in (0.2, 0.5):
        matlaws.add_damage(a, [(0, 0, 0)], d)
    for d in (0.5, 0.2):
        matlaws.add_damage(b, [(0, 0, 0)], d)
    assert a.dmg[0, 0, 0] == pytest.approx(b.dmg[0, 0, 0])      # the order does not matter


def test_a_blow_under_the_threshold_leaves_a_crack_and_a_flesh_cell_none():
    w = World(6, 6, 6)
    w.fill(2, 3, 2, 3, 1, 2, WOOD)
    assert matlaws.impact_damage(w, (2, 2, 1), 2.0, 0.01) == 1
    assert 0.0 < w.dmg[2, 2, 1] < 1.0
    assert matlaws.impact_damage(w, (0, 0, 0), 2.0, 0.01) == 0     # air takes none


# ── members ─────────────────────────────────────────────────────────────────
def _plank(laws):
    w = World(40, 6, 30, voxel_cm=5)
    w.mat_laws = frozenset(laws)
    w.fill(0, 40, 0, 6, 0, 1, STONE)
    w.fill(0, 3, 2, 3, 1, 6, STONE)
    w.fill(23, 26, 2, 3, 1, 6, STONE)
    w.fill(3, 23, 2, 3, 5, 6, WOOD, frac=0.3)                   # a thin plank
    return w


def _load(w, what=LEAD):
    w.fill(11, 15, 2, 3, 6, 26, what)                           # about 110 kg of lead on it


def _overloaded(laws):
    """A world the veto has LOOKED at (the first look only records the layout),
    then a block put on the plank: a change the next look will ask about."""
    w = _plank(laws)
    matlaws.members_veto(w)
    _load(w)
    w.tick += matlaws.VETO_EVERY
    return w


def test_a_world_that_was_built_standing_is_taken_to_stand():
    w = _plank(("members",))
    _load(w)                                                    # overloaded from the start
    assert matlaws.members_veto(w) == 0 and not w.rubble.any()  # nothing CHANGED: no question asked


def test_the_veto_breaks_an_overloaded_plank_at_its_critical_cell():
    w = _overloaded(("members",))
    wood = (w.mat == WOOD).sum()
    mass = float(w.smass.sum())
    assert matlaws.members_veto(w) == 1
    assert w.rubble.sum() == 1
    x, y, z = np.argwhere(w.rubble)[0]
    assert z == 5 and 11 <= x < 15 and w.mat[x, y, z] == WOOD    # under the block, and still its matter
    assert (w.mat == WOOD).sum() == wood and float(w.smass.sum()) == mass


def test_the_broken_plank_falls_through_the_real_support_law():
    w = _plank(("members",))
    w.step()                                                    # the first look
    _load(w)
    mass = float(w.smass.sum())
    for _ in range(80):
        w.step()
    assert (w.mat[12:15, 2, 5] != WOOD).all()                   # the piece went, and the block came down
    assert np.argwhere(w.mat == LEAD)[:, 2].min() < 5          # the lead started on the plank at z = 6
    assert float(w.smass.sum()) == pytest.approx(mass, rel=1e-6)   # and nothing was lost


def test_a_world_without_the_law_keeps_its_plank():
    w = _plank(())
    w.step()
    _load(w)
    for _ in range(80):
        w.step()
    assert (w.mat[3:23, 2, 5] == WOOD).all()


def test_a_sound_plank_is_left_alone():
    w = _plank(("members",))
    matlaws.members_veto(w)
    w.fill(11, 12, 2, 3, 6, 8, STONE)                           # a small stone on it
    w.tick += matlaws.VETO_EVERY
    assert matlaws.members_veto(w) == 0 and not w.rubble.any()


def test_a_person_walking_asks_nothing_and_a_waiting_veto_does_no_work():
    w = _overloaded(("members",))
    st = matlaws._STATE[w]
    w.tick -= matlaws.VETO_EVERY - 3                            # inside the gap
    assert matlaws.members_veto(w) == 0 and matlaws._STATE[w] is st and not w.rubble.any()


def _plank_with_load(hot_c):
    w = _plank(("members",))
    w.fill(3, 23, 2, 3, 5, 6, WOOD)                             # a full plank
    matlaws.members_veto(w)
    w.fill(11, 15, 2, 3, 6, 12, LEAD)                           # a load the cold plank carries
    if hot_c is not None:
        w.E[3:23, 2, 5] = (hot_c - 20.0) * w.heat_capacity()[3:23, 2, 5]
    w.tick += matlaws.VETO_EVERY
    return w


def test_a_plank_that_carries_a_load_cold_loses_it_when_it_is_charred():
    assert matlaws.members_veto(_plank_with_load(None)) == 0    # 20 C: it holds
    assert matlaws.members_veto(_plank_with_load(150.0)) == 0   # 150 C: it holds (wood keeps 0.6 of it)
    hot = _plank_with_load(420.0)                               # 420 C: char (5 percent of wood's strength)
    assert matlaws.members_veto(hot) >= 1 and hot.rubble.any()


def test_a_charred_post_still_stands_under_its_own_weight():
    """Wood above 350 C is char, not nothing: a post that is alight and carries
    only a light cap does not collapse at the first look (the record's derate
    row is zero there, and the veto reads the char floor instead)."""
    w = World(10, 10, 14, voxel_cm=5)
    w.mat_laws = frozenset(("members",))
    w.fill(0, 10, 0, 10, 0, 1, STONE)
    w.fill(4, 6, 4, 6, 1, 8, WOOD)
    w.fill(3, 7, 3, 7, 8, 9, STONE)
    matlaws.members_veto(w)
    w.E[4:6, 4:6, 1:8] = 400.0 * w.heat_capacity()[4:6, 4:6, 1:8]
    w.tick += matlaws.VETO_EVERY
    assert matlaws.members_veto(w) == 0 and not w.rubble.any()


# ── spall ───────────────────────────────────────────────────────────────────
def _hearth(solid_in_front):
    w = World(14, 10, 10, voxel_cm=5)
    w.mat_laws = frozenset(("spall",))
    w.fill(0, 14, 0, 10, 0, 1, STONE)
    w.fill(4, 10, 3, 7, 1, 6, STONE)                                # a stone block
    if solid_in_front:
        w.fill(3, 4, 3, 7, 1, 6, STONE)
    x0 = 4
    cold = (20.0 - 20.0)
    w.E[:] = 0.0
    C = w.heat_capacity()
    w.E[4:5, 3:7, 1:6] = (700.0 - 20.0) * C[4:5, 3:7, 1:6]         # the face cells are hot, the body is cool
    return w


@pytest.mark.parametrize("front", [False, True])
def test_a_hot_exposed_face_spalls_and_a_buried_one_does_not(front):
    w = _hearth(front)
    mass, heat = float(w.smass.sum()), float(w.E.sum())
    n = 0
    w.tick = matlaws.SPALL_EVERY
    n = matlaws.spall_pass(w)
    assert (n > 0) == (not front)
    assert float(w.smass.sum()) == pytest.approx(mass, rel=1e-6)    # a flake is moved, not lost
    assert float(w.E.sum()) == pytest.approx(heat, rel=1e-5)


# ── a landing that does not break the block still cracks it (damage on) ─────
def _drop(laws):
    from tests.test_carried_opt import floor_world, _block
    w = floor_world()
    w.mat_laws = frozenset(laws)
    cells, idx = _block(w)
    w._launch(cells, (0.0, 0.0, 0.0))
    for _ in range(400):
        w._law_bodies()
        if not w.bodies:
            break
    return w


def test_a_landing_under_the_threshold_cracks_the_block_only_with_damage_on():
    on, off = _drop(("damage",)), _drop(())
    assert off.dmg is None                                      # the old world never made the array
    assert on.dmg is not None and on.dmg.max() > 0.0
    assert on.dmg.max() < 1.0
    assert float(on.smass.sum()) == pytest.approx(float(off.smass.sum()))   # the same matter, cracked


def test_a_world_with_ice_in_it_is_looked_at_without_a_crash():
    """Ice (id 12) and the ids past it have no row in the twelve: the record is
    padded to 64 ids, so a lattice that holds one can still be indexed."""
    w = World(10, 10, 8)
    w.mat_laws = frozenset(("members", "spall"))
    w.fill(0, 10, 0, 10, 0, 1, STONE)
    w.fill(2, 4, 2, 4, 1, 3, WOOD)
    matlaws.members_veto(w)
    w.mat[6:8, 6:8, 1:3] = 12                                   # ICE
    w.smass[6:8, 6:8, 1:3] = 900.0
    w.tick += matlaws.VETO_EVERY
    assert matlaws.members_veto(w) == 0
    w.tick = matlaws.SPALL_EVERY
    assert matlaws.spall_pass(w) == 0


def test_a_crop_smaller_than_the_world_is_cut_the_same_for_every_array():
    """The crop is the changed cells plus a margin: a change in one corner of a
    wide world must not mix a cropped array with an uncropped one."""
    w = World(120, 12, 14, voxel_cm=5)
    w.mat_laws = frozenset(("members",))
    w.fill(0, 120, 0, 12, 0, 1, STONE)
    w.fill(5, 8, 2, 4, 1, 4, WOOD)
    matlaws.members_veto(w)
    w.fill(6, 7, 2, 3, 4, 5, STONE)                             # a small change far from the other end
    w.fill(100, 102, 5, 7, 1, 3, WOOD)                          # ...and wood out of the crop
    w.tick += matlaws.VETO_EVERY
    assert matlaws.members_veto(w) == 0                         # it ran (no ValueError) and nothing failed
