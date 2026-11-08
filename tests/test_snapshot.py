"""New in S5 (docs/sim-mind-design.md, 4.4): the snapshot ghost fix.

`snapshot`/`restore` deepcopy `self.__dict__`, and `self.matter.w` is a
reference straight back to the World. Without a memo entry telling deepcopy
that, it meets `self` again through `matter.w`, has never seen it before
(the World itself is not a key of the dict being copied), and copies the
WHOLE WORLD a second time -- a ghost hanging off `snap["matter"].w`, doubling
memory and never reachable through the restored world. `ledgers.py`'s `_THIS`
sentinel fixes it both ways: `snapshot` tells deepcopy `id(self) -> _THIS`,
and `restore` tells it `id(_THIS) -> self`, so `matter.w` always ends up
pointing at the live World, never at a copy of it.
"""
from src.voxel.demo import build


class _ExplodingPolicy:
    """A policy that must never be touched by snapshot/restore -- it is a
    mind, not world state (`LedgersMixin._NOT_STATE`), and it may one day be
    a model too large to deepcopy at all."""

    def __deepcopy__(self, memo):
        raise AssertionError("the policy must never be copied by snapshot/restore")


def test_SNAPSHOT_AND_RESTORE_do_not_touch_the_policy():
    w = build()
    w.policy = _ExplodingPolicy()
    snap = w.snapshot()               # must not raise
    w.restore(snap)                   # must not raise
    assert isinstance(w.policy, _ExplodingPolicy), \
        "restore must keep whatever mind is in place now, untouched"


def test_matter_W_IS_THE_WORLD_not_a_ghost_copy_of_it():
    w = build()
    assert w.matter.w is w, "the seam must read fields on the real, live world"
    snap = w.snapshot()
    w.step()                          # move the world on, so restore has work to do
    w.restore(snap)
    assert w.matter.w is w, \
        "restoring must repoint matter.w at the world being restored, not a ghost"


def test_restoring_INTO_A_SECOND_WORLD_repoints_matter_there_too():
    w1 = build()
    w1.step()
    snap = w1.snapshot()
    w2 = build()
    w2.restore(snap)
    assert w2.matter.w is w2, \
        "a snapshot taken on one World must restore cleanly onto another, " \
        "with matter.w pointing at the world it was restored INTO"
    assert w2.matter.w is not w1, "and never at the world the snapshot came from"
