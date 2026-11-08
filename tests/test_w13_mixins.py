"""THE PHASE A MIXINS ON THE WORLD, INERT (wave 13, lane B1).

Wave 13 builds four mixins that sit in front of the World's laws: body heat (T7),
the cold outside (T8, a draft), walking and doors (T11, two of them). B1's rule is
that each can be in the class list with golden SAME, and nothing is switched on: a
mixin that is in but has nothing to act on does nothing.

THEY ARE NOT IN THE CLASS LIST YET, and this file says why and what is true of each
TODAY, so that the lane that adds one starts from a fact:

  * the three that work (body heat, paths, mechanisms) are INERT in front of the
    World: the same bits on random worlds, compiled and numpy. Adding any of them
    changes no number;
  * but the phase A tests of each build "World plus the mixin" THEMSELVES (a
    throwaway `class W(BodyHeatMixin, World)` in tests/test_bodyheat_world.py, the
    same for the paths and the mechanisms), and a mixin that is already a base of the
    World cannot be named in front of it again: Python refuses the class (no
    consistent method resolution order). B1 put body heat in, found that the throwaway
    class of tests/test_bodyheat_world.py could no longer be built, and took it out
    again. So the lane that adds a mixin to
    `sim.World` (B7: paths and mechanisms; C6: body heat) edits the test that mixes it
    in, in the SAME change. `test_a_mixin_that_is_a_base_of_the_world_cannot_be_mixed_in_again`
    keeps that fact in front of them;
  * the cold mixin is NOT yet a plug. Its draft `_env_radiation` takes five
    arguments, the seam gives it three, and it has no branch for a world with no
    climate. C1 fixes both (the exact edit is in B1's report) and flips the strict
    xfail below to a pass in the same change.
"""
import numpy as np
import pytest

from src.voxel import sim
from src.voxel.bodyheat import BodyHeatMixin
from src.voxel.mechlaw import MechanismsMixin
from src.voxel.outside_world import OutsideMixin
from src.voxel.motion import FlatRoutesMixin
from src.voxel.pathlaw import PathingMixin
from src.voxel.scenes import _person
from src.voxel.sim import HAVE_NUMBA, World


def _world(cls, seed, fused):
    rng = np.random.default_rng(seed)
    w = cls(22, 18, 16, voxel_cm=5)
    w.fused = fused
    w.open_sky = bool(seed % 2)
    w.fill(0, 22, 0, 18, 0, 1, sim.STONE)
    mats = (sim.STONE, sim.WOOD, sim.IRON, sim.GLASS)
    for _ in range(int(rng.integers(3, 7))):
        x, y = int(rng.integers(1, 18)), int(rng.integers(1, 14))
        w.fill(x, x + int(rng.integers(1, 5)), y, y + int(rng.integers(1, 5)), 1,
               1 + int(rng.integers(1, 6)), mats[int(rng.integers(0, 4))],
               frac=float(rng.choice([0.4, 0.7, 1.0])))
    for _ in range(int(rng.integers(0, 3))):
        x, y = int(rng.integers(2, 20)), int(rng.integers(2, 16))
        w.pour(x, y, 1, sim.WATER, 90.0)
    for _ in range(int(rng.integers(1, 4))):
        x, y, z = int(rng.integers(1, 20)), int(rng.integers(1, 16)), int(rng.integers(1, 6))
        w.E[x, y, z] += float(rng.choice([300.0, 3000.0, 20000.0]))
    if seed % 2 == 0:
        _person(w, 15, 8, z0=1)
    return w


def _arrays(w):
    out = {k: v for k, v in vars(w).items() if isinstance(v, np.ndarray)}
    out["shed"] = np.float64(w.shed)
    return out


def _lockstep(cls, ticks=30, seeds=(0, 1, 2)):
    for seed in seeds:
        for fused in ([True, False] if HAVE_NUMBA else [False]):
            a, b = _world(World, seed, fused), _world(cls, seed, fused)
            for t in range(ticks):
                a.step()
                b.step()
                A, B = _arrays(a), _arrays(b)
                assert set(A) == set(B), (seed, fused, t, set(A) ^ set(B))
                for k in A:
                    assert np.array_equal(A[k], B[k]), f"seed {seed} fused {fused} tick {t}: {k}"


@pytest.mark.parametrize("mixin", [BodyHeatMixin],
                         ids=["body heat"])     # pathing and mechanisms are in World since 13a (B7, M3)
def test_the_working_mixins_in_front_of_the_world_change_nothing(mixin):
    _lockstep(type("Inert", (mixin, World), {}))


def test_body_heats_law_with_no_heat_record_in_the_world_is_a_loop_that_finds_nobody():
    class W(BodyHeatMixin, World):
        def step(self):
            World.step(self)
            self._law_body_heat()
    w = _world(W, 0, True)
    assert all(p.get("heat") is None for p in w.persons)
    for _ in range(45):                       # past the 40-tick body second: still nothing to do
        w.step()
    assert not any("_bh" in k or k == "bh_period" for k in vars(w))
    assert w.work_out == 0.0 and "work_out" not in vars(w)


def test_the_mixins_clash_with_the_world_where_they_mean_to_override():
    """Names a mixin defines that the World already has are overrides when the
    mixin goes in front. Pathing and the cold draft say what they replace; the
    mechanism mixin also repeats the ledger's names (`deposit`, `mech_in`, ...)
    that B2 puts in ledgers.py: B7 deletes the copies when it adds the mixin."""
    base = set()
    for c in World.__mro__:
        if c not in (PathingMixin, MechanismsMixin):             # it is in World since B7: it clashes with what it overrides
            base |= set(vars(c))

    def clash(m):
        return {n for n in vars(m) if not n.startswith("__")} & base
    assert clash(PathingMixin) == {"_fit_grid", "_routes"}   # World has it first (B7), so the names clash with FlatRoutesMixin
    # the cold draft overrides the seams of this lane: all four that it has
    assert {"_env_radiation", "_heat_busy", "_rest_cell", "_unsettled"} <= clash(OutsideMixin)


def test_a_mixin_that_is_a_base_of_the_world_cannot_be_mixed_in_again():
    """Why no phase A mixin is in the class list yet (module docstring): the tests
    of each build `class W(Mixin, World)`, which Python refuses once Mixin is a
    base of World. Shown on a throwaway World with its own copy of the mixin."""
    class Throwaway(BodyHeatMixin, World):
        pass
    with pytest.raises(TypeError, match="method resolution order"):
        type("Again", (BodyHeatMixin, Throwaway), {})
    assert BodyHeatMixin not in World.__mro__, "body heat went into sim.World: edit tests/test_bodyheat_world.py"
    # lane C4 put LayersMixin first (a gate-controlled base); mechanisms (M3) stay
    # ahead of the fields. The order is checked, not a fixed slot.
    from src.voxel.layers_world import LayersMixin
    from src.voxel.fields import FieldsMixin
    mro = World.__mro__
    assert mro.index(LayersMixin) < mro.index(MechanismsMixin) < mro.index(FieldsMixin)
    assert World.__mro__.index(PathingMixin) < World.__mro__.index(FlatRoutesMixin)   # B7 wired it


def test_the_cold_draft_in_front_of_the_world_is_not_yet_inert():
    _lockstep(type("Cold", (OutsideMixin, World), {}), ticks=10, seeds=(0,))
