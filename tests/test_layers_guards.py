"""T3 layers: the three defects of the phase A report (section 4), fixed at the cause.

1. Reading a mark with no light is "cannot read" (0.0), never a crash.
2. dew_point at relative humidity 0 is the physical limit, -infinity.
3. Layers.add refuses bad input BEFORE it touches the store.

Random sweeps over seeds, not hand-picked cases. All of these are laws or arithmetic
identities (not tagged (u)): they gate.
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


# ================================================================ 1. marks in the dark
def _dark_cases(rng):
    """Every way of having no light: E 0, E negative, a source of 0 cd, a black page."""
    d = rng.uniform(0.01, 50.0)
    yield dict(E_lux=0.0)
    yield dict(E_lux=-rng.random() * 100.0)
    yield dict(E_lux=M.lux_from_source(0.0, d))
    yield dict(E_lux=M.lux_from_source(-1.0, d))


@pytest.mark.parametrize("seed", SEEDS)
def test_a_mark_with_no_light_cannot_be_read_at_any_distance_and_never_raises(seed):
    rng = random.Random(seed)
    for _ in range(20):
        stroke = 10.0 ** rng.uniform(-5.0, -1.0)
        d = 10.0 ** rng.uniform(-2.0, 2.0)
        rho_bg, rho_mark = rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0)
        sigma = rng.choice([0.0, 1e-4 * rng.random()])
        for kw in _dark_cases(rng):
            assert M.legible(stroke, d, rho_bg, rho_mark, sigma_m=sigma, **kw) == 0.0
            L = M.luminance(rho_bg, kw["E_lux"])
            assert L == 0.0
            assert M.read_distance(stroke, L) == 0.0
            assert M.c_min(L) == math.inf and M.mar_arcmin(L) == math.inf


def test_the_zero_light_limit_is_continuous_with_the_dim_side():
    """As the light goes to zero the answer goes to zero from above: no jump at the
    edge where the old code divided by zero (log10(10/L) and sqrt(3/L))."""
    prev = None
    for E in (1.0, 1e-1, 1e-2, 1e-4, 1e-8, 1e-12, 1e-30, 0.0):
        L = M.luminance(0.8, E)
        d = M.read_distance(0.56e-3, L)
        p = M.legible(0.56e-3, 0.3, 0.8, 0.05, E_lux=E)
        assert d >= 0.0 and 0.0 <= p <= 1.0
        if prev is not None:
            assert d <= prev[0] + 1e-15 and p <= prev[1] + 1e-15      # dimmer is never better
        prev = (d, p)
    assert prev == (0.0, 0.0)


@pytest.mark.parametrize("seed", SEEDS)
def test_more_light_never_makes_a_mark_harder_to_read(seed):
    rng = random.Random(100 + seed)
    for _ in range(20):
        stroke = 10.0 ** rng.uniform(-4.0, -2.0)
        d = 10.0 ** rng.uniform(-1.0, 1.0)
        rho_bg, rho_mark = rng.uniform(0.3, 1.0), rng.uniform(0.0, 0.3)
        Es = sorted([0.0] + [10.0 ** rng.uniform(-4.0, 3.0) for _ in range(6)])
        ps = [M.legible(stroke, d, rho_bg, rho_mark, E_lux=E) for E in Es]
        assert all(b >= a - 1e-12 for a, b in zip(ps, ps[1:])), (Es, ps)
        assert ps[0] == 0.0 and all(0.0 <= p <= 1.0 for p in ps)


def test_a_light_mark_on_a_black_page_is_read_by_its_own_light_and_the_dark_is_not():
    for E in (0.1, 1.0, 300.0):
        assert M.contrast(0.5, 0.0) == math.inf
        p = M.legible(2e-3, 0.5, 0.0, 0.9, E_lux=E)                  # chalk on a slate
        assert 0.0 < p <= 1.0
        assert M.legible(2e-3, 0.5, 0.0, 0.0, E_lux=E) == 0.0       # black on black: nothing
    assert M.legible(2e-3, 0.5, 0.0, 0.9, E_lux=0.0) == 0.0


def test_the_eye_on_the_mark_and_a_dead_lamp_do_not_divide_by_zero():
    assert M.subtense_arcmin(1e-3, 0.0) == 180.0 * 60.0
    assert M.lux_from_source(0.0, 1.0) == 0.0
    with pytest.raises(ValueError):
        M.lux_from_source(1.0, 0.0)                                  # a point source at distance 0


def test_the_lit_values_of_the_marks_are_exactly_what_they_were():
    """The darkness fix moves nothing on the lit side."""
    assert M.c_min(50.0) == 0.02 and M.mar_arcmin(50.0) == 1.0
    assert M.c_min(0.75) == pytest.approx(0.02 * math.sqrt(4.0))
    assert M.mar_arcmin(1.0) == pytest.approx(1.0 + 2.5)
    assert M.read_distance(0.56e-3, 0.025) == pytest.approx(0.257, abs=0.002)
    assert M.contrast(0.05, 0.8) == pytest.approx(0.9375)


# ================================================================ 2. dew point at RH 0
def test_dew_point_at_zero_humidity_is_minus_infinity_at_every_temperature():
    for T in np.linspace(-60.0, 120.0, 37):
        assert LL.dew_point(float(T), 0.0) == -math.inf


@pytest.mark.parametrize("seed", SEEDS)
def test_dew_point_is_monotone_down_to_the_limit_and_inverts_psat(seed):
    rng = random.Random(200 + seed)
    for _ in range(20):
        T = rng.uniform(-40.0, 60.0)
        rhs = sorted(10.0 ** rng.uniform(-9.0, 0.0) for _ in range(6)) + [1.0]
        dews = [LL.dew_point(T, rh) for rh in rhs]
        assert all(math.isfinite(x) for x in dews)
        assert all(b > a for a, b in zip(dews, dews[1:])), (T, rhs, dews)
        assert dews[-1] == pytest.approx(T, abs=1e-9)                # saturated: dew point is T
        assert LL.dew_point(T, 0.0) < dews[0]                        # the limit is below every rh > 0
        for rh, td in zip(rhs, dews):
            assert LL.psat_sonntag(td) == pytest.approx(rh * LL.psat_sonntag(T), rel=1e-9)


def test_dew_point_below_float_resolution_is_still_the_limit_not_a_crash():
    for rh in (5e-324, 1e-320, 1e-300, 1e-100):
        td = LL.dew_point(20.0, rh)
        assert td == td and td < LL.dew_point(20.0, 1e-9)       # not NaN, and below every larger rh


@pytest.mark.parametrize("rh", [-1e-9, -0.5, float("nan")])
def test_dew_point_refuses_a_negative_or_nan_humidity(rh):
    with pytest.raises(ValueError):
        LL.dew_point(20.0, rh)


@pytest.mark.parametrize("seed", SEEDS)
def test_no_surface_gains_water_from_air_that_holds_none(seed):
    """The reason -infinity is the right answer: with zero vapour density the condensation
    flux is not positive on a plate of any temperature, so it is below the dew point of
    nothing."""
    rng = random.Random(300 + seed)
    for _ in range(20):
        Ts = rng.uniform(-50.0, 100.0)
        assert LL.condense_flux(0.0, Ts, rng.uniform(1.0, 20.0)) <= 0.0
        assert Ts > LL.dew_point(25.0, 0.0)


# ================================================================ 3. Layers.add refuses bad input
def _state(L):
    return (L.gen, L.n, copy.deepcopy(L.led_in), copy.deepcopy(L.energy_in),
            copy.deepcopy(L.canonical()), len(L.patterns), len(L.contents))


def _good_comp(rng):
    c = {}
    for _ in range(rng.randint(1, 3)):
        sp = rng.choice(SP)
        f = rng.choice([D.DISSOLVED, D.CRYSTAL, D.GRAIN])
        c[(sp, f, 0 if f == D.DISSOLVED else rng.choice([8, 24, 40]))] = rng.random() + 1e-3
    return c


def _unknown_species(rng):
    while True:
        sp = rng.randint(1, 1 << 20)
        if sp not in D.ROWS:
            return sp


@pytest.mark.parametrize("seed", SEEDS)
def test_add_refuses_bad_mass_species_keys_and_numbers_and_changes_nothing(seed):
    rng = random.Random(400 + seed)
    L = Layers()
    base = make_key(0, rng.randint(0, 50), rng.randint(0, 5))
    L.add(base, _good_comp(rng), Ts=20.0)                    # something already there
    for _ in range(25):
        k = make_key(0, rng.randint(0, 50), rng.randint(0, 5))
        good = _good_comp(rng)
        sp = rng.choice(SP)
        bad_mass = rng.choice([-1e-12, -rng.random() - 1e-9, -1e9, float("nan"),
                               float("inf"), float("-inf")])
        cases = [
            (ValueError, k, {**good, (sp, D.GRAIN, 24): bad_mass}, {}),            # one bad among good
            (ValueError, k, {(sp, D.DISSOLVED, 0): bad_mass}, {}),
            (KeyError, k, {(_unknown_species(rng), D.DISSOLVED, 0): 1.0}, {}),
            (KeyError, k, {**good, (_unknown_species(rng), D.CRYSTAL, 8): 1.0}, {}),
            (ValueError, k, {(sp, rng.choice([4, 5, -1, 99]), 0): 1.0}, {}),       # form out of 0..3
            (ValueError, k, {(sp, D.GRAIN, rng.choice([256, 300, -1])): 1.0}, {}), # sz out of 0..255
            (ValueError, k, {(sp, D.GRAIN): 1.0}, {}),                              # not a triple
            (ValueError, k, {(sp, 0.5, 0): 1.0}, {}),                               # float form
            (ValueError, k, {-5: 1.0}, {}),                                         # negative packed key
            (ValueError, k, {(sp, D.DISSOLVED, 0): "1.0"}, {}),                     # a string mass
            (ValueError, k, {(sp, D.DISSOLVED, 0): True}, {}),                      # a bool mass
            (ValueError, k, {(D.WATER, D.DISSOLVED, 0): 1e308, (D.SALT, D.DISSOLVED, 0): 1e308},
             {}),                                                                   # total overflows
            (ValueError, k, None, {}),
            (ValueError, 1.5, good, {}),                                            # float key
            (ValueError, True, good, {}),
            (ValueError, "k", good, {}),
            (ValueError, None, good, {}),
            (ValueError, -1, good, {}),
            (ValueError, 1 << 63, good, {}),
            (ValueError, make_key(rng.randint(4, 31), 3, 0), good, {}),             # unknown tag
            (ValueError, make_key(0, 3, rng.choice([6, 7])), good, {}),             # a cell has no face 6
            (ValueError, k, good, {"cover": rng.choice([0.0, -0.1, 1.0000001, 2.0, float("nan")])}),
            (ValueError, k, good, {"T_in": rng.choice([float("nan"), float("inf")])}),
            (ValueError, k, good, {"Ts": rng.choice([float("nan"), float("-inf")])}),
            (ValueError, k, good, {"latent_in": float("nan")}),
        ]
        for exc, key, comp, kw in cases:
            before = _state(L)
            with pytest.raises(exc):
                L.add(key, comp, **kw)
            assert _state(L) == before, (key, comp, kw)
    assert L.ledger_closes()


@pytest.mark.parametrize("seed", SEEDS)
def test_add_still_takes_every_good_input_and_the_ledger_closes(seed):
    """The checks refuse nothing that was legal: zero grams add nothing, a packed int
    key and a tuple key are the same, numpy numbers work, cover 1.0 and tiny cover work."""
    rng = random.Random(500 + seed)
    L = Layers()
    total = 0.0
    for _ in range(30):
        k = make_key(rng.choice([0, 1, 2, 3]), rng.randint(0, 40),
                     rng.randint(0, 1))
        comp = _good_comp(rng)
        kw = {}
        if rng.random() < 0.3:
            kw["cover"] = rng.choice([1.0, 1e-6, 0.5, np.float64(0.25)])
        if rng.random() < 0.3:
            comp = {D.pack_key(*c): np.float64(g) for c, g in comp.items()}
        if rng.random() < 0.2:
            comp[(rng.choice(SP), D.DISSOLVED, 0)] = 0.0         # a zero entry is dropped
        key = np.int64(k) if rng.random() < 0.3 else k
        L.add(key, comp, Ts=rng.uniform(-10, 50), **kw)
        total += sum(float(g) for g in comp.values())
    assert L.ledger_closes()
    assert L.mass() == pytest.approx(total, rel=1e-12)
    n = L.n
    assert L.add(make_key(0, 99, 4), {}, Ts=20.0) == (-1, [])
    assert L.add(make_key(0, 99, 4), {(D.WATER, D.DISSOLVED, 0): 0.0}, Ts=20.0) == (-1, [])
    assert L.n == n
