"""Smoke opacity on sight (wave 13c, C4b): sight reads the smoke scalar as soot.

K = sigma_e * 0.025 g soot/unit * 1000 L/m3 per g/L = 217.5 1/m (smokeopt), 38.8 times the
old 5.6. A ray is seen while its optical depth is under SEE_TAU (2.0), so a uniform smoke
of s g/L lets you see 2/K metres. Checked against that hand formula in both sight paths."""
import numpy as np

from src.voxel import smokeopt
from src.voxel.materials import AIR, SEE_TAU, SMOKE_EXT
from src.voxel.sim import World


def _hall(n=60):
    w = World(n, 6, 6, voxel_cm=10)
    w.open_sky = False
    w.smass[:] = 0.0
    return w


def _fill(w, g_per_l):
    w.smoke[:] = g_per_l * w.vox_l


def _reach(w, k):
    """Is a cell k cells down the hall seen from cell 1, by both sight paths?"""
    a, b = (1, 3, 3), (1 + k, 3, 3)
    one = w._sees(a, b)
    many = bool(w.rays([a], [b])[0])
    assert one == many
    return one


def test_the_gate_defaults_on_and_the_constant_is_the_soot_law():
    assert World.smoke_opacity is True
    w = _hall()
    assert w._smoke_ext() == smokeopt.extinction_from_smoke(1.0)
    assert abs(w._smoke_ext() - 217.5) < 1e-9
    w.smoke_opacity = False
    assert w._smoke_ext() == SMOKE_EXT == 5.6          # the old world, the same float


def test_visibility_in_uniform_smoke_matches_the_hand_formula():
    w = _hall()
    s = 0.02                                           # g/L: a thin haze
    _fill(w, s)
    # optical depth over k cells (the ray skips its two ends) = K * 0.1 * (k-1) m
    reach_m = SEE_TAU / (smokeopt.extinction_from_smoke(s))  # about 0.46 m
    edge = int(round(reach_m / 0.1)) + 1
    assert _reach(w, edge - 2)
    assert not _reach(w, edge + 2)


def test_old_law_saw_through_the_same_smoke():
    w = _hall()
    _fill(w, 0.02)
    w.smoke_opacity = False
    assert _reach(w, 50)                               # K 0.11 /m: 5 m of it is tau 0.56
    w.smoke_opacity = True
    assert not _reach(w, 50)


def test_clear_air_is_unchanged():
    w = _hall()
    assert _reach(w, 55)
    w.smoke_opacity = False
    assert _reach(w, 55)


def test_visibility_function_agrees_with_see_tau():
    # Jin's illuminated-sign visibility C_v/K and the engine's 2/K differ by the
    # constant C_v 8 against SEE_TAU 2: the engine's cut is the stricter one.
    K = smokeopt.extinction_from_smoke(0.01)
    assert smokeopt.visibility(K) == 4 * (SEE_TAU / K)


def test_a_FLAME_is_seen_by_its_own_light_through_smoke_that_hides_a_lit_thing():
    """Jin: a light-emitting sign is seen K=8/sigma away, a reflecting one K=3/sigma.
    So the smoke on the way to a flame counts SEE_LIT_SMOKE = 3/8 as much; solids do not."""
    from src.voxel.materials import SEE_LIT_SMOKE, STONE
    w = _hall()
    k = 20
    vox_m = 0.1 * w.scale
    # smoke dense enough that the far cell's smoke depth is 1.5 x SEE_TAU
    per_cell = 1.5 * SEE_TAU / ((k - 2) * vox_m)          # 1/m over the cells between
    _fill(w, per_cell / w._smoke_ext())
    a, b = (1, 3, 3), (1 + k, 3, 3)
    assert not bool(w.rays([a], [b])[0])                    # a lit thing: hidden
    assert bool(w.rays([a], [b], lit=[True])[0])            # a flame: seen
    assert 1.5 * SEE_LIT_SMOKE < 1.0                       # why: 0.5625 x SEE_TAU
    w.mat[1 + k // 2, 3, 3] = STONE                         # a wall still hides it
    w.smass[1 + k // 2, 3, 3] = 2600.0 * w.vox_l
    assert not bool(w.rays([a], [b], lit=[True])[0])
