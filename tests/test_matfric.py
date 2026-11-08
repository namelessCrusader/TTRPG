"""Friction per pair and per film (src/voxel/matfric.py, wave 13 topic T9).

Tags: test_codeonly_* checks the code against its own table; test_u_* leans on
an unverified (u) number. Untagged tests are arithmetic identities, closed
forms or structural facts of the table.
"""
import itertools
import math

import numpy as np
import pytest

from src.voxel import matfric as F
from src.voxel import matrec as M

NC = F.NC
CI = range(NC)


def _all():
    return itertools.product(CI, CI, range(4))


def test_symmetric_for_every_class_pair_and_film():
    for a, b, f in _all():
        assert F.MU_S64[a, b, f] == F.MU_S64[b, a, f]
        assert F.MU_K64[a, b, f] == F.MU_K64[b, a, f]


def test_static_is_at_least_kinetic_and_everything_is_in_range():
    for a, b, f in _all():
        s, k = F.MU_S64[a, b, f], F.MU_K64[a, b, f]
        assert s >= k, (a, b, f)
        assert 0.0 < k and s < 1.5, (a, b, f)
    assert F.MU_S.dtype == np.float32 and F.MU_S.shape == (NC, NC, 4)       # the spec's float32 tables


def test_film_order_iced_oiled_wet_dry_for_EVERY_pair_listed_or_not():
    for a, b in itertools.product(CI, CI):
        for T in (F.MU_S64, F.MU_K64):
            dry, wet, oil, ice = T[a, b, 0], T[a, b, 1], T[a, b, 2], T[a, b, 3]
            assert ice <= oil <= wet <= dry, (F.CLASSES[a], F.CLASSES[b], ice, oil, wet, dry)


def test_film_order_holds_through_mu_pair_on_material_ids():
    ids = np.arange(12)
    A, B = np.meshgrid(ids, ids)
    for kin in (False, True):
        v = [F.mu_pair(A, B, f, kinetic=kin) for f in range(4)]
        assert np.all(v[3] <= v[2]) and np.all(v[2] <= v[1]) and np.all(v[1] <= v[0])
        assert np.all(v[3] > 0)


def test_codeonly_listed_dry_values():
    cases = {("wood", "wood"): (0.40, 0.30), ("rock", "rock"): (0.70, 0.60),
             ("steel", "steel"): (0.70, 0.50), ("glass", "glass"): (0.90, 0.40),
             ("wood", "rock"): (0.55, 0.40), ("skin", "rock"): (0.60, 0.50),
             ("ice", "ice"): (0.10, 0.03), ("fibre", "fibre"): (0.50, 0.40)}
    for (a, b), (s, k) in cases.items():
        ia, ib = F.CLASSES.index(a), F.CLASSES.index(b)
        assert (F.MU_S64[ia, ib, 0], F.MU_K64[ia, ib, 0]) == (s, k)
        assert (F.MU_S64[ib, ia, 0], F.MU_K64[ib, ia, 0]) == (s, k)
    sm, gr = F.CLASSES.index("softmetal"), F.CLASSES.index("granular")
    for c in CI:
        if c != gr:
            assert (F.MU_S64[sm, c, 0], F.MU_K64[sm, c, 0]) == (0.60, 0.50) or c == sm
        assert F.MU_S64[gr, c, 0] <= 0.60                      # a granular-with-any (a soft metal is the exception)
    assert (F.MU_S64[gr, F.CLASSES.index("wood"), 0], F.MU_K64[gr, F.CLASSES.index("wood"), 0]) == (0.50, 0.45)


def test_default_fill_for_an_unlisted_dry_pair_is_the_geometric_mean_clamped():
    # skin - glass is not listed
    ca, cb = "skin", "glass"
    s, k = F.mu_default(ca, cb, F.DRY)
    assert s == pytest.approx(math.sqrt(0.50 * 0.90)) and k == pytest.approx(math.sqrt(0.40 * 0.40))
    ia, ib = F.CLASSES.index(ca), F.CLASSES.index(cb)
    assert F.MU_S64[ia, ib, 0] == s and F.MU_K64[ia, ib, 0] == k
    # ice - rock: ice is the low outlier; the clamp [0.03, 1.2] holds
    s, k = F.mu_default("ice", "rock", F.DRY)
    assert s == pytest.approx(math.sqrt(0.10 * 0.70)) and k == pytest.approx(max(0.03, math.sqrt(0.03 * 0.60)))
    for a, b in itertools.product(F.CLASSES, F.CLASSES):
        s, k = F.mu_default(a, b, F.DRY)
        assert 0.03 <= k <= s <= 1.2


def test_u_default_wet_is_at_most_seventy_percent_of_dry_and_oiled_iced_are_the_any_pair_values():
    for a, b in [("skin", "glass"), ("fibre", "steel"), ("ice", "rock")]:
        d = F.mu_default(a, b, F.DRY)
        w = F.mu_default(a, b, F.WET)
        assert w == tuple(0.7 * x for x in d)
        assert w[0] <= d[0] and w[1] <= d[1]
        assert F.mu_default(a, b, F.OILED) == F.OILED_PAIR == (0.10, 0.08)
        assert F.mu_default(a, b, F.ICED) == F.ICED_WARM == (0.05, 0.03)
    # a wet pair that nobody listed, read off the table, never exceeds its dry value
    ia, ib = F.CLASSES.index("fibre"), F.CLASSES.index("steel")
    assert F.MU_S64[ia, ib, 1] <= F.MU_S64[ia, ib, 0]


def test_codeonly_listed_wet_and_oiled_values():
    wood = F.CLASSES.index("wood")
    assert (F.MU_S64[wood, wood, 1], F.MU_K64[wood, wood, 1]) == (0.30, 0.20)
    rock = F.CLASSES.index("rock")
    assert (F.MU_S64[rock, rock, 1], F.MU_K64[rock, rock, 1]) == (0.60, 0.50)
    for a, b in itertools.product(CI, CI):                       # oiled 0.10 / 0.08 for any pair, unless dry is lower
        assert F.MU_S64[a, b, 2] == min(0.10, F.MU_S64[a, b, 1])
        assert F.MU_K64[a, b, 2] == min(0.08, F.MU_K64[a, b, 1])


def test_u_iced_is_005_at_minus_5_C_rising_to_012_at_minus_20_C():
    rock = M.STONE
    s = lambda T: F.mu_pair(rock, rock, F.ICED, kinetic=False, T_C=T)
    k = lambda T: F.mu_pair(rock, rock, F.ICED, kinetic=True, T_C=T)
    assert s(-5.0) == pytest.approx(0.05) and k(-5.0) == pytest.approx(0.03)
    assert s(-20.0) == pytest.approx(0.12) and k(-20.0) == pytest.approx(0.10)
    assert s(-12.5) == pytest.approx(0.085) and k(-12.5) == pytest.approx(0.065)        # linear
    assert s(-40.0) == s(-20.0) and s(-1.0) == s(-5.0)                                  # flat outside
    prev = 0.0
    for T in np.linspace(-5, -20, 16):
        assert s(T) >= prev
        prev = s(T)
    # with no temperature the plain table (the warm end) is read
    assert F.mu_pair(rock, rock, F.ICED, kinetic=False) == pytest.approx(0.05)
    # arrays of T broadcast
    out = F.mu_pair(rock, rock, F.ICED, kinetic=False, T_C=np.array([-5.0, -20.0]))
    assert out.shape == (2,) and out[1] > out[0]


def test_iced_never_passes_the_dry_value_and_T_only_matters_for_ice():
    for a, b in itertools.product(range(12), range(12)):
        dry = F.mu_pair(a, b, F.DRY, kinetic=False, T_C=-20.0)
        ice = F.mu_pair(a, b, F.ICED, kinetic=False, T_C=-20.0)
        assert ice <= dry + 1e-12
        assert F.mu_pair(a, b, F.WET, kinetic=False, T_C=-20.0) == F.mu_pair(a, b, F.WET, kinetic=False)
    # a mixed array: only the iced entries read the temperature
    film = np.array([F.DRY, F.ICED])
    out = F.mu_pair(M.WOOD, M.STONE, film, kinetic=False, T_C=-20.0)
    assert out[0] == F.mu_pair(M.WOOD, M.STONE, F.DRY, kinetic=False) and out[1] == pytest.approx(0.12)


def test_film_of_names_the_film_a_fluid_makes():
    from src.voxel import materials as MT
    assert (F._WATER, F._OIL, F._ACID, F._WEAK_ACID, F._BLOOD) == \
        (MT.WATER, MT.OIL, MT.ACID, MT.WEAK_ACID, MT.BLOOD)
    assert F._MOLTEN == (MT.MTIN, MT.MLEAD, MT.MIRON)
    assert F.film_of(MT.WATER, 10.0) == F.WET and F.film_of(MT.WATER, 0.0) == F.ICED
    assert F.film_of(MT.WATER, -5.0) == F.ICED and F.film_of(MT.WATER) == F.WET
    assert F.film_of(MT.OIL, -30.0) == F.OILED
    assert F.film_of(MT.BLOOD, 20.0) == F.WET and F.film_of(MT.ACID, 20.0) == F.WET
    assert F.film_of(MT.NOFLUID, 20.0) == F.DRY and F.film_of(MT.MIRON, 1500.0) == F.DRY
    out = F.film_of(np.array([0, 1, 2, 1]), np.array([20.0, -3.0, 20.0, 20.0]))
    assert list(out) == [F.DRY, F.ICED, F.OILED, F.WET]


def test_the_class_of_every_material_matches_matrec_and_the_stand_in():
    for r in M.BY_ID:
        assert F.CLASSES[F.MAT_CLASS[r.id]] == r.mu_class
        assert F._STANDIN_CLASS[r.id] == r.mu_class                      # cross test: the stand-in copy
    assert F.CLASSES[F.MAT_CLASS[12]] == "ice"                           # the reserved ICE id
    assert tuple(F.CLASSES) == tuple(M.MU_CLASSES)


def test_mu_pair_symmetric_on_material_ids_and_broadcasts():
    A, B = np.meshgrid(np.arange(12), np.arange(12), indexing="ij")
    for f in range(4):
        v = F.mu_pair(A, B, f, kinetic=False)
        assert np.array_equal(v, v.T)
    assert F.mu_pair(M.WOOD, M.WOOD, F.DRY, kinetic=False) == 0.40
    assert isinstance(F.mu_pair(M.WOOD, M.IRON, F.WET), float)
    assert F.mu_pair(np.array([1, 2, 3]), M.STONE).shape == (3,)


def test_the_flat_table_is_the_old_forty_percent_everywhere():
    assert F.FRICTION == 0.40
    from src.voxel import materials as MT
    assert F.FRICTION == MT.FRICTION
    for ma, mb, f, kin in itertools.product(range(12), range(12), range(4), (False, True)):
        assert F.mu_pair(ma, mb, f, kinetic=kin, table=(F.FLAT_S, F.FLAT_K)) == 0.40


def test_capstan_three_turns_at_mu_04_holds_about_1880_times():
    r = F.capstan(0.4, 3 * 2 * math.pi)
    assert r == pytest.approx(math.exp(0.4 * 6 * math.pi), rel=1e-12)
    assert r == pytest.approx(1880.0, rel=0.002)
    assert F.capstan(0.4, 0.0) == 1.0
    assert F.capstan(np.array([0.2, 0.4]), math.pi).shape == (2,)
    # the rope-on-post pair from the table: fibre on wood
    mu = F.mu_pair(M.ROPE, M.WOOD, F.DRY, kinetic=True)
    assert F.capstan(mu, 2 * math.pi) == pytest.approx(math.exp(mu * 2 * math.pi))


def test_slip_flips_exactly_at_the_static_limit():
    for ma, mb, f in [(M.WOOD, M.STONE, F.DRY), (M.IRON, M.IRON, F.WET), (M.WOOD, M.WOOD, F.OILED)]:
        mu_s = F.mu_pair(ma, mb, f, kinetic=False)
        n = 589.0
        lim = mu_s * n
        assert F.slip_at(ma, mb, f, n, lim) is True
        assert F.slip_at(ma, mb, f, n, np.nextafter(lim, 0.0)) is False
        assert F.slip_at(ma, mb, f, n, lim * 1.01) is True
    assert F.slip_at(M.WOOD, M.STONE, F.DRY, np.array([100.0, 100.0]), np.array([10.0, 90.0])).tolist() == [False, True]
    assert F.slip_at(M.WOOD, M.WOOD, F.ICED, 100.0, 4.9, T_C=-5.0) is False
    assert F.slip_at(M.WOOD, M.WOOD, F.ICED, 100.0, 5.1, T_C=-5.0) is True


def test_u_the_haul_numbers_of_scene_P8():
    """A 60 kg chest (589 N): start and sliding forces on stone, wet planks, oil, ice."""
    n = 60.0 * 9.81
    wood, stone = M.WOOD, M.STONE
    assert F.mu_pair(wood, stone, F.DRY, kinetic=False) * n == pytest.approx(324.0, rel=0.01)
    assert F.mu_pair(wood, stone, F.DRY, kinetic=True) * n == pytest.approx(235.0, rel=0.01)
    assert F.mu_pair(wood, wood, F.WET, kinetic=False) * n == pytest.approx(177.0, rel=0.01)
    assert F.mu_pair(wood, wood, F.WET, kinetic=True) * n == pytest.approx(118.0, rel=0.01)
    assert F.mu_pair(wood, wood, F.OILED, kinetic=False) * n == pytest.approx(59.0, rel=0.01)
    assert F.mu_pair(wood, wood, F.OILED, kinetic=True) * n == pytest.approx(47.0, rel=0.01)
    assert F.mu_pair(wood, wood, F.ICED, kinetic=False, T_C=-5.0) * n == pytest.approx(29.0, rel=0.02)
    assert F.mu_pair(wood, wood, F.ICED, kinetic=True, T_C=-5.0) * n == pytest.approx(18.0, rel=0.02)
