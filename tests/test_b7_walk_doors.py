"""Lane B7 (wave 13a): walking and doors at range 0, wired in.

PathingMixin sits in front of the flat route base inside MotionMixin, so the
World gets it with no change to the class list; with the step range at 0 it is
the old planner (the range is 30 cm by default since lane C9). A door leaf or frame is not shoved by a hand.
"""
import numpy as np

from src.voxel.motion import FlatRoutesMixin, MotionMixin
from src.voxel.pathlaw import PathingMixin
from src.voxel.sim import World
from tests.test_mechlaw import DoorWorld, _room


def test_the_world_plans_with_pathingmixin_first_and_the_flat_base_behind_it():
    mro = World.__mro__
    assert mro.index(MotionMixin) < mro.index(PathingMixin) < mro.index(FlatRoutesMixin)
    assert World._routes is PathingMixin._routes
    # lane C9 (declared diff, ledger World.step_up_m = 0.3): legs climb 30 cm
    # by default, which is 3 cells at 10 cm voxels. 13a had it at 0.
    assert PathingMixin._step_range(World(8, 8, 8), None) == (3, 3)
    w = World(8, 8, 8)
    w.step_up_m = w.step_down_m = 0.0          # the per-world switch back to the flat planner
    assert PathingMixin._step_range(w, None) == (0, 0)


def test_a_hand_does_not_shove_a_door_leaf_or_its_frame():
    w, mid, _p = _room(W=3, H=3, latch=False)
    rec = w.mechs[mid]
    leaf = w._cells_of(rec["leaf"])
    frame = w._cells_of(rec["frame"])
    assert w._joined_by_mechanism(leaf) and w._joined_by_mechanism(frame)
    before = w.mat.copy()
    assert w._shove(leaf.astype(np.int32), 0, -1) is False
    assert (w.mat == before).all()


def test_a_world_with_no_mechanism_shoves_as_before():
    w = World(10, 10, 6)
    assert not w._joined_by_mechanism(np.array([[2, 2, 1]]))
    assert not DoorWorld(10, 10, 6)._joined_by_mechanism(np.array([[2, 2, 1]]))
