"""T3 layers, polish round 2 (addendum 5.B.3): input that used to slip through.

1. Layers.add reads comp ONCE: a generator of (key, grams) pairs keeps every gram.
2. Denormal numbers are handled by the physical limit, never a crash.
3. A non-finite light level is refused (ValueError), not turned into a silent NaN.
4. latent_in=None is a ValueError, like every other bad number.
5. take, move, strip, contact and condense check their input as add does, and
   change nothing when they refuse it.
6. (NFIX, wave 13 phase B 13a) What round 2 left open: lux_from_source on a denormal
   distance and a NaN source, a list of shares that is not one per layer, a generator of
   shares, and the substrate arguments (sub, sub_src, sub_dst, sub_S, sub_D).

Random sweeps over seeds. These are arithmetic identities and refusals, not tagged
(u): they gate.
"""
import copy
import math
import random

import numpy as np
import pytest

from src.voxel import layer_data as D
from src.voxel import layer_laws as LL
from src.voxel import marks as M
from src.voxel.layers import Layers, make_key

SEEDS = range(40)
SP = [D.WATER, D.SALT, D.CLAY, D.SOOT, D.CHALK, D.TALLOW, D.PLASMA, D.QUARTZ, D.LAMP_OIL]
NAN, INF = float("nan"), float("inf")


def _state(L):
    return (L.gen, L.n, copy.deepcopy(L.led_in), copy.deepcopy(L.led_out),
            copy.deepcopy(L.energy_in), copy.deepcopy(L.canonical()),
            len(L.patterns), len(L.contents))


def _good_comp(rng):
    c = {}
    for _ in range(rng.randint(1, 3)):
        sp = rng.choice(SP)
        f = rng.choice([D.DISSOLVED, D.CRYSTAL, D.GRAIN])
        c[(sp, f, 0 if f == D.DISSOLVED else rng.choice([8, 24, 40]))] = rng.random() + 1e-3
    return c


def _loaded(rng, nfaces=4):
    """A pool with matter on a few faces; returns (L, keys)."""
    L = Layers()
    keys = [make_key(0, i, rng.randint(0, 5)) for i in range(nfaces)]
    for k in keys:
        for _ in range(rng.randint(1, 2)):
            L.add(k, _good_comp(rng), Ts=rng.uniform(0.0, 40.0))
    return L, keys


# ================================================================ 1. a generator keeps its grams
@pytest.mark.parametrize("seed", SEEDS)
def test_add_with_a_one_shot_iterable_loses_no_gram(seed):
    rng = random.Random(1000 + seed)
    for wrap in (lambda c: ((k, g) for k, g in c.items()),          # a generator
                 lambda c: iter(list(c.items())),                   # an iterator
                 lambda c: list(c.items()),
                 lambda c: tuple(c.items()),
                 lambda c: c.items()):                              # a dict view
        comp = _good_comp(rng)
        key = make_key(0, rng.randint(0, 30), rng.randint(0, 5))
        T = rng.uniform(-5.0, 45.0)
        ref, got = Layers(), Layers()
        ref.add(key, comp, Ts=T)
        got.add(key, wrap(comp), Ts=T)
        want = sum(comp.values())
        assert got.mass() == pytest.approx(want, rel=1e-12)
        assert got.mass() > 0.0
        assert got.canonical() == ref.canonical()
        assert got.led_in == ref.led_in and got.energy_in == ref.energy_in
        assert got.ledger_closes()


@pytest.mark.parametrize("bad", [5, 2.5, object(), "abc", b"ab", [1, 2, 3], [(1,)],
                                 [((D.WATER, D.DISSOLVED, 0), 1.0, 2.0)]])
def test_add_refuses_a_comp_that_is_no_mapping_and_no_list_of_pairs(bad):
    L = Layers()
    L.add(make_key(0, 1, 0), {(D.WATER, D.DISSOLVED, 0): 1.0}, Ts=20.0)
    before = _state(L)
    with pytest.raises(ValueError):
        L.add(make_key(0, 2, 0), bad, Ts=20.0)
    assert _state(L) == before


@pytest.mark.parametrize("seed", SEEDS)
def test_a_generator_with_one_bad_pair_changes_nothing(seed):
    rng = random.Random(1100 + seed)
    L, _keys = _loaded(rng)
    comp = list(_good_comp(rng).items())
    comp.insert(rng.randint(0, len(comp)),
                ((D.WATER, D.DISSOLVED, 0), rng.choice([-1.0, NAN, INF])))
    before = _state(L)
    with pytest.raises(ValueError):
        L.add(make_key(0, 60, 0), (p for p in comp), Ts=20.0)
    assert _state(L) == before


# ================================================================ 2. denormals
def _denormal(rng):
    return rng.choice([5e-324, 1e-323, rng.uniform(1.0, 1000.0) * 5e-324,
                       10.0 ** rng.uniform(-323.0, -308.5)])


@pytest.mark.parametrize("seed", SEEDS)
def test_a_denormal_light_level_reads_by_the_formula_limit_and_never_raises(seed):
    rng = random.Random(1200 + seed)
    for _ in range(25):
        L = _denormal(rng)
        stroke = 10.0 ** rng.uniform(-5.0, -1.0)
        mar = M.mar_arcmin(L)
        # the same law as for ordinary L: 1 + 2.5 log10(10/L), written without 10/L
        assert mar == pytest.approx(1.0 + 2.5 * (1.0 - math.log10(L)), rel=1e-12)
        assert math.isfinite(mar) and mar > M.mar_arcmin(1e-300)
        cm = M.c_min(L)
        assert math.isfinite(cm) and cm == pytest.approx(
            0.02 * math.sqrt(3.0) / math.sqrt(L), rel=1e-12)
        d = M.read_distance(stroke, L)
        assert 0.0 < d <= M.read_distance(stroke, 1e-300) + 1e-18   # dimmer is never better
        assert d <= M.read_distance(stroke, 1e-3)
        p = M.legible(stroke, rng.uniform(0.01, 5.0), 1.0, 0.0, E_lux=L * math.pi)
        assert 0.0 <= p <= 1.0
    assert M.read_distance(1e-3, 5e-324) == pytest.approx(0.0042152, rel=1e-4)


@pytest.mark.parametrize("seed", SEEDS)
def test_a_denormal_humidity_is_the_zero_humidity_limit_at_every_temperature(seed):
    rng = random.Random(1300 + seed)
    for _ in range(25):
        T = rng.uniform(-60.0, 100.0)
        rh = _denormal(rng)
        td = LL.dew_point(T, rh)
        assert td == td                                   # never NaN
        assert td <= LL.dew_point(T, 1e-9) and td < T     # below every larger humidity
    for T in np.linspace(-60.0, 100.0, 33):
        assert LL.dew_point(float(T), 5e-324) < LL.dew_point(float(T), 1e-6)
    assert LL.dew_point(-40.0, 5e-324) == -math.inf


def test_the_ordinary_values_did_not_move():
    for L, mar, cm in ((1.0, 3.5, 0.02 * math.sqrt(3.0)), (9.99, None, 0.02), (50.0, 1.0, 0.02),
                       (0.025, None, 0.02 * math.sqrt(120.0))):
        assert M.mar_arcmin(L) == (mar if mar is not None else 1.0 + 2.5 * math.log10(10.0 / L))
        assert M.c_min(L) == pytest.approx(cm, rel=1e-15)
    assert LL.dew_point(20.0, 0.5) == pytest.approx(9.2, abs=0.1)


# ================================================================ 3. NaN and infinity in
@pytest.mark.parametrize("bad", [NAN, INF, -INF, "1", None, True])
def test_the_light_readers_refuse_a_non_finite_or_non_number_input(bad):
    for f in (M.c_min, M.mar_arcmin):
        with pytest.raises(ValueError):
            f(bad)
    with pytest.raises(ValueError):
        M.read_distance(1e-3, bad)
    with pytest.raises(ValueError):
        M.read_distance(bad, 1.0)
    with pytest.raises(ValueError):
        M.luminance(0.8, bad)
    with pytest.raises(ValueError):
        M.luminance(bad, 300.0)
    with pytest.raises(ValueError):
        M.legible(1e-3, 0.5, 0.8, 0.1, E_lux=bad)


@pytest.mark.parametrize("seed", SEEDS)
def test_the_light_readers_never_hand_back_a_nan(seed):
    rng = random.Random(1400 + seed)
    for _ in range(30):
        L = rng.choice([0.0, -rng.random(), 10.0 ** rng.uniform(-320.0, 6.0)])
        for v in (M.c_min(L), M.mar_arcmin(L), M.read_distance(10.0 ** rng.uniform(-5, -1), L)):
            assert v == v
        bad = rng.choice([NAN, INF, -INF])
        with pytest.raises(ValueError):
            M.c_min(bad)
        with pytest.raises(ValueError):
            M.mar_arcmin(bad)


# ================================================================ 4. latent_in
@pytest.mark.parametrize("bad", [None, NAN, INF, -INF, "1.0", True, [1.0]])
def test_latent_in_that_is_not_a_finite_number_is_a_value_error(bad):
    L = Layers()
    L.add(make_key(0, 1, 0), {(D.WATER, D.DISSOLVED, 0): 1.0}, Ts=20.0)
    before = _state(L)
    with pytest.raises(ValueError):
        L.add(make_key(0, 2, 0), {(D.WATER, D.DISSOLVED, 0): 1.0}, Ts=20.0, latent_in=bad)
    assert _state(L) == before


# ================================================================ 5. the other public methods
def _bad_numbers(rng):
    return rng.choice([NAN, INF, -INF, None, "1.0", True, [0.5]])


def _bad_keys(rng):
    return rng.choice([1.5, True, "k", None, -1, 1 << 63, NAN,
                       make_key(rng.randint(4, 31), 3, 0), make_key(0, 3, rng.choice([6, 7]))])


@pytest.mark.parametrize("seed", SEEDS)
def test_take_refuses_bad_input_and_changes_nothing(seed):
    rng = random.Random(1500 + seed)
    L, keys = _loaded(rng)
    k = rng.choice(keys)
    for _ in range(20):
        sp = rng.choice(SP)
        cases = [
            (ValueError, dict(key=_bad_keys(rng), species=sp, form=D.DISSOLVED, grams=1.0)),
            (KeyError, dict(key=k, species=_unknown(rng), form=D.DISSOLVED, grams=1.0)),
            (ValueError, dict(key=k, species=1.5, form=D.DISSOLVED, grams=1.0)),
            (ValueError, dict(key=k, species=None, form=D.DISSOLVED, grams=1.0)),
            (ValueError, dict(key=k, species=sp, form=rng.choice([4, -1, 99, 0.5, None]), grams=1.0)),
            (ValueError, dict(key=k, species=sp, form=D.DISSOLVED,
                              grams=rng.choice([-1e-9, -3.0, NAN, INF, -INF, None, "1", True]))),
            (ValueError, dict(key=k, species=sp, form=D.DISSOLVED, grams=1.0, Ts=_bad_numbers(rng))),
            (ValueError, dict(key=k, species=sp, form=D.DISSOLVED, grams=1.0, latent=_bad_numbers(rng))),
            (ValueError, dict(key=k, species=sp, form=D.DISSOLVED, grams=1.0, tag=rng.choice([None, 3]))),
        ]
        for exc, kw in cases:
            before = _state(L)
            with pytest.raises(exc):
                L.take(**kw)
            assert _state(L) == before, kw
    assert L.ledger_closes()


def _unknown(rng):
    while True:
        sp = rng.randint(1, 1 << 20)
        if sp not in D.ROWS:
            return sp


@pytest.mark.parametrize("seed", SEEDS)
def test_move_refuses_bad_input_and_changes_nothing(seed):
    rng = random.Random(1600 + seed)
    L, keys = _loaded(rng)
    a, b = keys[0], keys[1]
    for _ in range(20):
        cases = [
            dict(src_key=_bad_keys(rng), dst_key=b, fractions=0.5),
            dict(src_key=a, dst_key=_bad_keys(rng), fractions=0.5),
            dict(src_key=a, dst_key=b, fractions=rng.choice([1.0000001, 2.0, 1e9, -1e-9, -0.5, NAN, INF, None, "0.5", True])),
            dict(src_key=a, dst_key=b, fractions=[0.5, rng.choice([1.5, -0.1, NAN, INF, None])]),
            dict(src_key=a, dst_key=b, fractions=0.5, Ts_src=_bad_numbers(rng)),
            dict(src_key=a, dst_key=b, fractions=0.5, Ts_dst=_bad_numbers(rng)),
            dict(src_key=a, dst_key=b, fractions=0.5, cover=rng.choice([0.0, -0.1, 1.0000001, NAN, INF, "0.5"])),
        ]
        for kw in cases:
            before = _state(L)
            with pytest.raises(ValueError):
                L.move(**kw)
            assert _state(L) == before, kw
    assert L.ledger_closes()


@pytest.mark.parametrize("seed", SEEDS)
def test_move_of_a_share_over_one_is_refused_not_matter_from_nothing(seed):
    """Before the check a share of 2 booked twice what the layer held onto the
    destination while the ledger said nothing."""
    rng = random.Random(1700 + seed)
    L, keys = _loaded(rng)
    m0 = L.mass()
    with pytest.raises(ValueError):
        L.move(keys[0], keys[1], rng.uniform(1.0001, 50.0))
    assert L.mass() == m0 and L.ledger_closes()
    L.move(keys[0], keys[1], 1.0)
    assert L.mass() == pytest.approx(m0, rel=1e-12) and L.ledger_closes()


@pytest.mark.parametrize("seed", SEEDS)
def test_strip_refuses_bad_input_and_changes_nothing(seed):
    rng = random.Random(1800 + seed)
    L, keys = _loaded(rng)
    for _ in range(20):
        cases = [
            dict(key=_bad_keys(rng), depth_m=1e-3),
            dict(key=keys[0], depth_m=rng.choice([-1e-9, -1.0, NAN, INF, -INF, None, "1e-3", True])),
            dict(key=keys[0], depth_m=1e-3, Ts=_bad_numbers(rng)),
        ]
        for kw in cases:
            before = _state(L)
            with pytest.raises(ValueError):
                L.strip(**kw)
            assert _state(L) == before, kw
    comp, _adj = L.strip(keys[0], 10.0)                          # a good one still strips all
    assert sum(comp.values()) > 0.0 and L.ledger_closes()


@pytest.mark.parametrize("seed", SEEDS)
def test_contact_refuses_bad_input_and_changes_nothing(seed):
    rng = random.Random(1900 + seed)
    L, keys = _loaded(rng)
    S, Ds = keys[0], [keys[1], keys[2]]
    good = dict(S=S, Ds=Ds, c_c=0.5, p=1e4, s_slide=0.1, L_p=0.05)
    for _ in range(20):
        cases = [
            dict(S=_bad_keys(rng)),
            dict(Ds=[keys[1], _bad_keys(rng)]),
            dict(Ds=[]),
            dict(Ds=keys[1]),                                   # a key, not a list of keys
            dict(Ds=None),
            dict(c_c=rng.choice([-0.1, 1.0000001, 2.0, NAN, INF, None, "0.5"])),
            dict(p=rng.choice([-1.0, NAN, INF, None, "1"])),
            dict(s_slide=rng.choice([-1.0, NAN, INF, None, "1"])),
            dict(L_p=rng.choice([-1.0, NAN, INF, None, "1"])),
            dict(Ts_S=_bad_numbers(rng)),
            dict(Ts_D=_bad_numbers(rng)),
            dict(weights=[1.0]),                                # one weight, two faces
            dict(weights=[1.0, 2.0, 3.0]),
            dict(weights=[1.0, rng.choice([-1.0, NAN, INF, None, "1"])]),
            dict(weights=[0.0, 0.0]),                           # no path at all
            dict(weights=5.0),
        ]
        for kw in cases:
            before = _state(L)
            with pytest.raises(ValueError):
                L.contact(**{**good, **kw})
            assert _state(L) == before, kw
    assert L.ledger_closes()


@pytest.mark.parametrize("seed", SEEDS)
def test_contact_with_a_generator_of_faces_moves_the_same_as_with_a_list(seed):
    rng = random.Random(2000 + seed)
    La, keys = _loaded(rng)
    Lb = copy.deepcopy(La)
    args = (0.7, 2e4, 0.1, 0.05)
    ma, _ = La.contact(keys[0], [keys[1], keys[2]], *args, weights=[1.0, 3.0])
    mb, _ = Lb.contact(keys[0], (k for k in [keys[1], keys[2]]), *args,
                       weights=(x for x in [1.0, 3.0]))
    assert ma == mb and La.canonical() == Lb.canonical()
    assert La.ledger_closes() and Lb.ledger_closes()


@pytest.mark.parametrize("seed", SEEDS)
def test_condense_refuses_bad_input_and_changes_nothing(seed):
    rng = random.Random(2100 + seed)
    L, keys = _loaded(rng)
    for _ in range(20):
        good = dict(key=keys[0], grams=0.5, Ts=rng.uniform(-10.0, 30.0))
        cases = [
            dict(key=_bad_keys(rng)),
            dict(grams=rng.choice([-1e-9, -1.0, NAN, INF, None, "1", True])),
            dict(Ts=_bad_numbers(rng)),
            dict(freeze_c=_bad_numbers(rng)),
        ]
        for kw in cases:
            before = _state(L)
            with pytest.raises(ValueError):
                L.condense(**{**good, **kw})
            assert _state(L) == before, kw
    m0 = L.mass()
    L.condense(keys[0], 0.25, 5.0)                               # a good one still condenses
    assert L.mass() == pytest.approx(m0 + 0.25, rel=1e-12) and L.ledger_closes()


# ================================================================ 6. what round 2 left open (NFIX)
def test_lux_from_source_has_no_zero_division_and_reads_no_nan_as_darkness():
    assert M.lux_from_source(100.0, 2.0) == 25.0                         # the ordinary law did not move
    assert M.lux_from_source(0.0, 1.0) == 0.0 and M.lux_from_source(-5.0, 1.0) == 0.0
    assert M.lux_from_source(1.0, 1e-150) == pytest.approx(1e300, rel=1e-12)
    for d in (1e-170, 1e-200, 1e-300, 5e-324):                           # d * d underflowed to 0: a ZeroDivisionError
        assert M.lux_from_source(1.0, d) == INF                          # the source is on the eye
        assert M.lux_from_source(0.0, d) == 0.0 and M.lux_from_source(-1.0, d) == 0.0
    for bad in (NAN, INF, -INF, None, "1", True):
        with pytest.raises(ValueError):
            M.lux_from_source(bad, 1.0)                                  # a NaN source was read as darkness
        with pytest.raises(ValueError):
            M.lux_from_source(1.0, bad)
    for d in (0.0, -1.0, -1e-300):
        with pytest.raises(ValueError):
            M.lux_from_source(1.0, d)
    with pytest.raises(ValueError):
        M.luminance(0.8, M.lux_from_source(1.0, 5e-324))                 # the world sink must clamp the infinity first


@pytest.mark.parametrize("seed", SEEDS)
def test_move_takes_one_share_per_layer_and_reads_a_generator_once(seed):
    rng = random.Random(2200 + seed)
    L, keys = _loaded(rng)
    a, b = keys[0], keys[1]
    n = len(L.stack(a))
    shares = [rng.random() for _ in range(n)]
    ref = copy.deepcopy(L)
    ref.move(a, b, shares)
    assert ref.mass(a) < L.mass(a) and ref.ledger_closes()
    for wrap in (tuple, iter, lambda x: (v for v in x), lambda x: np.array(x)):   # a generator used to be refused
        got = copy.deepcopy(L)
        got.move(a, b, wrap(shares))
        assert got.canonical() == ref.canonical() and got.ledger_closes()
    for bad in (shares + [0.5], shares[:-1], [], shares * 2):            # not one share per layer
        before = _state(L)
        with pytest.raises(ValueError):
            L.move(a, b, bad)
        assert _state(L) == before
    nothing = make_key(0, 999, 0)                                        # an empty face has no layers to align to
    before = _state(L)
    assert L.move(nothing, b, [0.5, 0.5]) == [] and _state(L) == before


@pytest.mark.parametrize("seed", SEEDS)
def test_the_substrate_arguments_are_checked_before_anything_is_touched(seed):
    rng = random.Random(2300 + seed)
    L, keys = _loaded(rng)
    a, b = keys[0], keys[1]
    calls = [
        lambda sub: L.add(make_key(0, 90, 0), {(D.WATER, D.DISSOLVED, 0): 1.0}, Ts=20.0, sub=sub),
        lambda sub: L.take(a, D.WATER, D.DISSOLVED, 0.1, sub=sub),
        lambda sub: L.move(a, b, 0.5, sub_src=sub),
        lambda sub: L.move(a, b, 0.5, sub_dst=sub),
        lambda sub: L.strip(a, 1e-3, sub=sub),
        lambda sub: L.contact(a, [b], 0.5, 1e4, 0.1, 0.05, sub_S=sub),
        lambda sub: L.contact(a, [b], 0.5, 1e4, 0.1, 0.05, sub_D=sub),
        lambda sub: L.condense(a, 0.1, 5.0, sub=sub),
    ]
    for call in calls:
        for sub in (True, False, 2.5, 1.0, [1], (2,), object(), NAN, b"stone"):    # a float was cut to an int
            before = _state(L)
            with pytest.raises(ValueError):
                call(sub)
            assert _state(L) == before, sub
        for sub in ("nonsense", 9999, -3, np.int64(77)):                     # no such substrate: KeyError, as a species
            before = _state(L)
            with pytest.raises(KeyError):
                call(sub)
            assert _state(L) == before, sub
    for sub in (None, "stone", "wood", 2, 1, np.int64(2), 101, dict(D.SUBSTRATE[2])):   # the good ones still work
        L.take(a, D.WATER, D.DISSOLVED, 0.0, sub=sub)
        L.add(make_key(0, 91, 0), {(D.WATER, D.DISSOLVED, 0): 0.2}, Ts=20.0, sub=sub)
    assert L.ledger_closes()
