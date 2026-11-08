"""Damage: combine, impact, shatter threshold, the creep clock (src/voxel/matdamage.py, wave 13 T9).

Tags: test_echo_* checks a number the constants were FITTED to (wood n and
t_R1 give the ten-year rule); test_codeonly_* checks the code against its own
table; test_u_* leans on an unverified number. Untagged tests are exact.
"""
import itertools
import math

import numpy as np
import pytest

from src.voxel import matdamage as D
from src.voxel import matrec as M

WOOD = M.REC["wood"]
N, TR1 = WOOD.creep_n, WOOD.t_R1_s


# ── combine ─────────────────────────────────────────────────────────────────

def test_combine_is_commutative_and_associative_and_stays_in_range():
    rng = np.random.default_rng(1)
    for _ in range(2000):
        a, b, c = rng.random(3) * rng.choice([1.0, 0.1, 1.0])
        assert D.combine(a, b) == D.combine(b, a)
        assert D.combine(D.combine(a, b), c) == pytest.approx(D.combine(a, D.combine(b, c)), abs=1e-12)
        for v in (D.combine(a, b), D.combine(D.combine(a, b), c)):
            assert 0.0 <= v <= 0.999


def test_combine_values_zero_and_the_clip():
    assert D.combine(0.0, 0.0) == 0.0
    assert D.combine(0.5, 0.5) == pytest.approx(0.75)
    assert D.combine(0.3, 0.0) == pytest.approx(0.3)
    assert D.combine(0.9, 1.0) == 0.999 and D.combine(1.0, 1.0) == 0.999
    assert D.combine(0.0, 1.0) == 0.999


def test_combine_a_repair_lowers_D_and_never_goes_below_zero():
    assert D.combine(0.5, -0.5) == pytest.approx(0.25)
    assert D.combine(0.1, -0.9) == 0.0
    # each cause keeps its own survival fraction: order does not matter
    assert D.combine(D.combine(0.2, 0.3), 0.4) == pytest.approx(1 - 0.8 * 0.7 * 0.6)


def test_combine_works_on_arrays():
    a = np.array([0.0, 0.5, 0.99])
    out = D.combine(a, np.array([0.5, 0.5, 0.5]))
    assert out.shape == (3,) and out.max() <= 0.999
    assert out[1] == pytest.approx(0.75)


def test_cross_there_is_one_copy_of_combine():
    """Plan section 3.2: matdamage.combine is the only combine."""
    import inspect
    from src.voxel import matrec, matfric
    for mod in (matrec, matfric):
        src = inspect.getsource(mod)
        assert "def combine" not in src


# ── multipliers and decay ───────────────────────────────────────────────────

def test_multipliers():
    assert D.mult_t(0.0) == 1.0 and D.mult_E(0.25) == 0.75
    assert D.mult_c(0.75) == pytest.approx(0.5)
    assert D.mult_c(0.0) == 1.0 and D.mult_t(0.999) == pytest.approx(0.001)
    r = M.REC["stone"].at(20.0, 0.36)
    assert r.sc == pytest.approx(M.REC["stone"].sc * D.mult_c(0.36))      # the record's at() uses the same law
    assert r.st == pytest.approx(M.REC["stone"].st * D.mult_t(0.36))
    assert r.E == pytest.approx(M.REC["stone"].E * D.mult_E(0.36))


def test_decay_uses_the_rot_ratio_from_the_record_and_clips():
    assert D.decay_D(0.1, M.REC["wood"].rot_ratio) == pytest.approx(0.5)
    assert D.decay_D(0.1, M.REC["iron"].rot_ratio) == pytest.approx(0.1)
    assert D.decay_D(0.5, 5.0) == 0.999 and D.decay_D(0.0, 5.0) == 0.0
    assert D.decay_D(-1.0, 5.0) == 0.0
    assert D.decay_D(np.array([0.0, 0.1, 1.0]), 5.0).tolist() == [0.0, pytest.approx(0.5), 0.999]


# ── impact damage ───────────────────────────────────────────────────────────

def test_codeonly_impact_dD_is_eta_e_over_gimp_A_and_capped_at_one():
    g, A = M.REC["glass"].gimp, 0.05 ** 2
    assert D.impact_dD(0.1, g, A, 0.3) == pytest.approx(0.3 * 0.1 / (g * A))
    assert D.impact_dD(1e9, g, A) == 1.0 and D.impact_dD(0.0, g, A) == 0.0
    assert D.impact_dD(0.1, g, A, eta=0.0) == 0.0                 # eta = 0: damage storage only (decision 6)
    assert D.ETA_IMPACT == 0.3
    a = D.impact_dD(np.array([0.1, 0.2]), g, A)
    assert a.shape == (2,) and a[1] == pytest.approx(2 * a[0])


def test_codeonly_a_sub_threshold_blow_lowers_the_next_threshold():
    g, A = M.REC["glass"].gimp, 0.05 ** 2
    thr = np.float32(g * A)                                  # 0.5 J
    d1 = D.impact_dD(0.3, g, A)
    assert 0.0 < d1 < 1.0
    assert D.shatter_threshold(thr, d1) < thr
    # blow after blow: the cell cracks enough that the same blow now shatters it
    Dm, n = 0.0, 0
    while 0.3 <= D.shatter_threshold(float(thr), Dm):
        Dm = D.combine(Dm, D.impact_dD(0.3, g, A))
        n += 1
        assert n < 100
    assert n == 3


# ── the shatter threshold is bit-identical at D = 0 (F4) ────────────────────

def test_shatter_threshold_at_D0_is_bit_identical_over_1e5_samples_float32():
    rng = np.random.default_rng(7)
    tough = rng.choice(np.array([0.2, 3.0, 8.0, 150.0, 0.8, 60.0, 40.0, 30.0], np.float32), 100_000)
    area = (rng.random(100_000) * 0.05).astype(np.float32)
    flaw = (0.7 + 0.6 * rng.random(100_000)).astype(np.float32)
    thr = tough * 1000.0 * area * flaw                       # today's operand order, float32
    assert thr.dtype == np.float32
    for Dv in (0.0, np.float32(0.0), np.zeros(100_000, np.float32), np.zeros(100_000)):
        out = D.shatter_threshold(thr, Dv)
        assert out.dtype == np.float32
        assert out.view(np.uint32).tobytes() == thr.view(np.uint32).tobytes()


def test_shatter_threshold_at_D0_is_bit_identical_for_scalars_and_python_floats():
    rng = np.random.default_rng(8)
    for _ in range(20_000):
        t32 = np.float32(rng.random() * 1e3) * np.float32(0.7 + 0.6 * rng.random())
        out = D.shatter_threshold(t32, 0.0)
        assert type(out) is type(t32) and out.tobytes() == t32.tobytes()
        t = float(rng.random() * 1e3) * (0.7 + 0.6 * rng.random())
        out = D.shatter_threshold(t, 0.0)
        assert type(out) is float and out == t
    assert D.shatter_threshold(np.float32(np.inf), 0.0) == np.inf
    assert D.shatter_threshold(np.float32(0.0), 0.0) == 0.0


def test_shatter_threshold_scales_with_the_survival_fraction_and_keeps_dtype():
    thr = np.array([1.0, 2.0, 3.0], np.float32)
    out = D.shatter_threshold(thr, np.array([0.0, 0.5, 0.75], np.float32))
    assert out.dtype == np.float32 and out.tolist() == [1.0, 1.0, 0.75]
    assert D.shatter_threshold(4.0, 0.25) == 3.0
    assert D.shatter_threshold(np.float32(4.0), 0.25) == np.float32(3.0)


# ── the creep clock (float64, path independent) ─────────────────────────────

def test_creep_y_step_is_the_closed_form():
    y = D.creep_y_step(1.0, 0.9, N, TR1, 100.0)
    assert y == pytest.approx(1.0 - 0.9 ** N * 100.0 / TR1, abs=1e-15)
    assert D.creep_y_step(1.0, 0.0, N, TR1, 1e9) == 1.0            # no load, no ageing
    assert D.creep_y_step(1.0, -2.0, N, TR1, 1e9) == 1.0
    assert isinstance(D.creep_y_step(1.0, 0.5, N, TR1, 1.0), float)


def test_creep_clock_is_path_independent_in_float64():
    u = 0.8
    t_end = 0.37 * D.time_to_failure(1.0, u, N, TR1)              # well inside the life
    one = D.creep_y_step(1.0, u, N, TR1, t_end)
    half = D.creep_y_step(D.creep_y_step(1.0, u, N, TR1, t_end / 2), u, N, TR1, t_end / 2)
    rng = np.random.default_rng(3)
    cuts = np.sort(rng.random(9)) * t_end
    steps = np.diff(np.concatenate(([0.0], cuts, [t_end])))
    y = 1.0
    for dt in steps:
        y = D.creep_y_step(y, u, N, TR1, dt)
    assert half == pytest.approx(one, abs=1e-14)
    assert y == pytest.approx(one, abs=1e-14)
    assert isinstance(one, float) and np.asarray(one).dtype == np.float64
    assert np.asarray(D.creep_y_step(np.float32(1.0), u, N, TR1, 1.0)).dtype == np.float64


def test_creep_stepping_to_tR_over_2_twice_equals_one_step():
    u = 0.9
    tr = D.time_to_failure(1.0, u, N, TR1)
    a = D.creep_y_step(D.creep_y_step(1.0, u, N, TR1, tr / 2), u, N, TR1, tr / 2)
    b = D.creep_y_step(1.0, u, N, TR1, tr)
    assert a == pytest.approx(b, abs=1e-14) and abs(b) < 1e-12 and abs(a) < 1e-12


def test_creep_piecewise_constant_load_is_exact():
    # u = 0.9 for half its life then u = 0.5: the rest of the life follows from y alone
    tr9 = D.time_to_failure(1.0, 0.9, N, TR1)
    y = D.creep_y_step(1.0, 0.9, N, TR1, tr9 / 2)
    assert y == pytest.approx(0.5, abs=1e-14)
    assert D.time_to_failure(y, 0.5, N, TR1) == pytest.approx(0.5 * TR1 * 0.5 ** -N)


def test_echo_wood_t_R_at_u_1_09_057():
    t1 = D.time_to_failure(1.0, 1.0, N, TR1)
    t9 = D.time_to_failure(1.0, 0.9, N, TR1)
    t57 = D.time_to_failure(1.0, 0.57, N, TR1)
    assert t1 == pytest.approx(300.0, rel=0.02)
    assert t9 == pytest.approx(4049.0, rel=0.02)
    assert t57 == pytest.approx(3.2e8, rel=0.02)
    assert t57 / (365.25 * 86400) == pytest.approx(10.2, abs=0.2)           # the ten-year rule


def test_u_one_held_for_tR1_gives_y_le_zero():
    assert D.creep_y_step(1.0, 1.0, N, TR1, TR1) <= 1e-15
    assert D.creep_y_step(1.0, 1.0, N, TR1, TR1 * 1.0001) < 0.0
    assert D.creep_y_step(1.0, 1.0, N, TR1, TR1 * 0.9) == pytest.approx(0.1)


def test_time_to_failure_infinite_without_load_and_arrays_work():
    assert D.time_to_failure(1.0, 0.0, N, TR1) == math.inf
    out = D.time_to_failure(np.array([1.0, 0.5]), np.array([0.9, 0.0]), N, TR1)
    assert out[0] == pytest.approx(4049.0, rel=0.01) and out[1] == np.inf
    assert D.time_to_failure(0.0, 0.9, N, TR1) == 0.0


def test_creep_D_for_reporting():
    assert D.creep_D(1.0, N) == 0.0 and D.creep_D(0.0, N) == 1.0 and D.creep_D(-1.0, N) == 1.0
    assert D.creep_D(0.5, N) == pytest.approx(1 - 0.5 ** (1 / (N + 1)))
    assert 0.0 < D.creep_D(0.9, N) < D.creep_D(0.5, N) < 1.0


def test_creep_advance_locates_the_failure_inside_a_step():
    y, failed, tf = D.creep_advance(1.0, 0.9, N, TR1, 10.0)
    assert not failed and tf is None and y < 1.0
    y, failed, tf = D.creep_advance(0.001, 0.9, N, TR1, 10.0)
    assert failed and y == 0.0 and 0.0 < tf < 10.0
    assert tf == pytest.approx(D.time_to_failure(0.001, 0.9, N, TR1))


# ── SEPARATION: the instant check and the clock never count twice ───────────

def test_member_at_u_09_fails_by_creep_at_4049_s_not_earlier():
    outcome, t = D.run_member(0.9, N, TR1, dt=1.0, t_end=10_000.0)
    assert outcome == "creep"
    assert t == pytest.approx(4049.0, abs=1.5)
    # not earlier: it holds until just before
    assert D.run_member(0.9, N, TR1, 1.0, 4040.0)[0] == "holds"
    assert D.run_member(0.9, N, TR1, 10.0, 4040.0)[0] == "holds"
    assert D.instant_util(0.9, 0.0) == 0.9                         # the instant check sees no creep
    # a coarse step lands on the same failure time (closed form, path independent)
    assert D.run_member(0.9, N, TR1, 500.0, 10_000.0)[1] == pytest.approx(t, abs=1e-6)


def test_member_with_D_other_01_at_u_09_fails_the_instant_check():
    assert D.instant_util(0.9, 0.1) == pytest.approx(1.0)
    assert D.instant_util(0.9, 0.1) >= 1.0 - 1e-12
    out = D.run_member(0.9, N, TR1, 1.0, 10_000.0, D_other=0.1000001)
    assert out == ("instant", 0.0)
    # compression takes the square root of the survival fraction
    assert D.instant_util(0.9, 0.19, 0.5) == pytest.approx(0.9 / 0.9)
    # and below that it holds, then creeps
    assert D.run_member(0.85, N, TR1, 100.0, 1e9, D_other=0.1)[0] == "creep"


# ── duration factor ─────────────────────────────────────────────────────────

def test_echo_duration_factor_at_0_2_s_is_1_34_and_capped_at_1_3():
    assert D.duration_factor(0.2, N, TR1) == pytest.approx(1.34, abs=0.005)
    assert D.duration_factor(0.2, N, TR1, cap=D.DURATION_CAP) == 1.3
    assert D.transient_factor(0.2, N, TR1) == 1.3
    assert D.DURATION_CAP == 1.3
    assert D.transient_factor(TR1, N, TR1) == 1.0                  # a load as long as t_R1 gains nothing
    assert D.transient_factor(1e6, N, TR1) == 1.0                  # never below 1
    assert 1.0 < D.transient_factor(10.0, N, TR1) < 1.3
    assert D.duration_factor(np.array([0.2, 300.0]), N, TR1).shape == (2,)


# ── a few facts across the modules ──────────────────────────────────────────

def test_codeonly_the_record_rows_that_carry_a_clock():
    for key in ("wood", "glass", "stone"):
        r = M.REC[key]
        assert r.creep_n and r.t_R1_s
    for key in ("iron", "tin", "lead", "flesh"):
        assert M.REC[key].creep_n is None                        # no static fatigue at room T
