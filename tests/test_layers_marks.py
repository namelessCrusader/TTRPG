"""T3 marks: patterns, mirror, impress, ink print, degrade, ncc, legibility, thin-film
colour (spec 7: T27-T32; acceptance: mirror twice is identity, ncc 1.0 for a cast
copy, moonlight reading distance 0.257 m, an 8.7 mm letter readable at 6 m).

Tags: test_codeonly_* tests the code against its own formula; test_u_* independent but
the source is unverified; the rest are identities.
"""
import math

import numpy as np
import pytest

from src.voxel import layer_data as D
from src.voxel import marks as M
from src.voxel.layers import Layers, make_key

# an asymmetric 7 x 5 letter R (v rows, u columns)
R = np.array([[1, 1, 1, 1, 0],
              [1, 0, 0, 0, 1],
              [1, 0, 0, 0, 1],
              [1, 1, 1, 1, 0],
              [1, 0, 1, 0, 0],
              [1, 0, 0, 1, 0],
              [1, 0, 0, 0, 1]], np.uint8)


# ================================================================ T27 mirror and impress
def test_mirror_twice_is_the_identity_for_bitmaps_and_strokes():
    b = M.Bitmap(R * 200)
    assert np.array_equal(M.mirror(M.mirror(b)).data, b.data)
    assert not np.array_equal(M.mirror(b).data, b.data)            # R is asymmetric
    s = M.Strokes([(0.005, 0.01, 0.03, 0.04, 0.001, 1.0), (0.01, 0.0, 0.02, 0.03, 0.002, 0.5)],
                  [("a", 0.01, 0.02, 0.005, 1.0)], 0.05, 0.05)
    ss = M.mirror(M.mirror(s))
    assert np.allclose(ss.strokes, s.strokes)
    assert [g[0] for g in ss.glyphs] == [g[0] for g in s.glyphs]
    assert np.allclose([g[1:] for g in ss.glyphs], [g[1:] for g in s.glyphs])
    assert M.mirror(s).strokes[0][0] == pytest.approx(0.05 - 0.005)


def test_a_die_cut_in_reverse_reads_true_in_the_wax():
    die = M.mirror(M.Bitmap(R * 200))                               # cut in reverse
    wax = M.impress(die, p=2e4, tau_y=M.wax_yield(65.0), layer_thickness_m=1e-3)
    assert wax is not None
    assert np.array_equal(wax.data > 0, R > 0)                      # reads "R" correctly
    assert not np.array_equal(wax.data > 0, die.data > 0)
    # a stamp inked and pressed prints the mirrored positive too
    pr = M.ink_print(die, ink=0.8)
    assert np.array_equal(pr.data > 0, R > 0)
    assert pr.data.max() == pytest.approx(0.8 * 200)
    dep = M.deposit_patch(M.Bitmap(R * 1.0), share=0.5)
    assert np.array_equal(dep.data > 0, M.mirror(M.Bitmap(R)).data > 0)


# ================================================================ T28 impress needs 3 tau_y
def test_codeonly_impress_needs_a_pressure_of_three_yield_stresses():
    die = M.Bitmap(R * 100)
    warm, cold = M.wax_yield(65.0), M.wax_yield(20.0)
    assert warm < 1e3 < 5e6 < cold                                  # three orders from 50 to 70 C
    assert M.wax_yield(50.0) / M.wax_yield(70.0) == pytest.approx(1e3, rel=0.01)
    assert M.can_impress(2e4, warm) and not M.can_impress(2e4, cold)
    assert M.can_impress(3.0 * warm, warm) and not M.can_impress(3.0 * warm * 0.999, warm)
    assert M.impress(die, 2e4, warm, 1e-3) is not None
    assert M.impress(die, 2e4, cold, 1e-3) is None


def test_an_impression_leaves_the_wax_mass_unchanged_in_the_pool():
    L = Layers()
    k = make_key(1, 12, 0)
    L.add(k, {(D.BEESWAX, D.DISSOLVED, 0): 0.4}, cover=1.0, Ts=65.0,
          pattern=M.impress(M.Bitmap(R * 100), 2e4, M.wax_yield(65.0), 1e-3), source="seal")
    assert L.mass(k) == pytest.approx(0.4)
    s = L.stack(k)[0]
    assert L.pat[s] >= 0 and isinstance(L.patterns[L.pat[s]], M.Bitmap)
    assert L.look(k)["patterns"] == [int(L.pat[s])]
    assert L.ledger_closes()


# ================================================================ T29 blur
def test_blur_variance_is_two_d_t_and_a_blurred_stroke_loses_contrast():
    pitch = 1e-4
    imp = np.zeros((41, 201))
    imp[20, 100] = 1.0
    b = M.Bitmap(imp, pitch)
    D_coef, t = 2e-9, 40.0                                           # sigma = sqrt(2Dt) = 0.4 mm
    out = M.blur(b, D_coef, t)
    u = (np.arange(201) - 100) * pitch
    w = out.data[20, :]
    var = float((w * u * u).sum() / w.sum())
    assert var == pytest.approx(2 * D_coef * t, rel=0.03)
    assert out.data.sum() == pytest.approx(1.0, rel=1e-3)           # ink is spread, not lost
    assert out.data.max() < 0.2 * b.data.max()
    assert M.blur_sigma(D_coef, t) == pytest.approx(4e-4)
    # strokes carry the variance in closed form (variances add)
    s = M.Strokes([(0.01, 0.01, 0.03, 0.01, 0.0006, 1.0)], width_m=0.05, height_m=0.05)
    s2 = M.blur(M.blur(s, 1e-9, 1.0), 1e-9, 1.0)
    assert s2.sigma == pytest.approx(math.sqrt(2 * 1e-9 * 1.0 * 2), rel=1e-12)   # variances add
    clear = M.legible(0.56e-3, 0.3, 0.8, 0.05, E_lux=11.1, sigma_m=0.0)
    smudged = M.legible(0.56e-3, 0.3, 0.8, 0.05, E_lux=11.1, sigma_m=s2.sigma + 0.4e-3)
    assert smudged < 0.7 * clear


def test_smear_wear_and_fade_lower_the_amount():
    b = M.Bitmap(R * 100.0)
    sm = M.smear(b, 6e-3)
    assert sm.data.max() < b.data.max()
    assert sm.data.sum() == pytest.approx(b.data.sum(), rel=0.3)
    assert M.fade(b, 0.1, 10.0).data.max() == pytest.approx(100.0 * math.exp(-1.0))
    assert M.wear(b, 30.0).data.max() == pytest.approx(70.0)
    assert M.wear(b, 500.0).data.max() == 0.0


# ================================================================ T30 ncc
def test_ncc_is_one_for_a_cast_copy_and_falls_with_noise():
    rng = np.random.default_rng(3)
    base = rng.random((25, 25))
    assert M.ncc(base, base.copy()) == pytest.approx(1.0, abs=1e-6)
    assert M.ncc(M.Bitmap(base), M.Bitmap(base.copy())) == pytest.approx(1.0, abs=1e-6)
    vals = [M.ncc(base, base + rng.normal(0, s, base.shape)) for s in (0.05, 0.2, 0.5, 1.0)]
    assert all(b < a for a, b in zip(vals, vals[1:]))
    assert vals[0] < 1.0 - 1e-6
    # the same in the frame of a mirror: a mirrored copy does NOT match an asymmetric die
    b = M.Bitmap(R * 1.0)
    assert M.ncc(b, M.mirror(b)) < 0.99


def test_a_warm_blade_lifts_the_seal_and_the_reseal_no_longer_matches_the_die():
    die = M.mirror(M.Bitmap(R * 200.0))
    seal = M.impress(die, 2e4, M.wax_yield(65.0), 1e-3)
    assert M.ncc(seal, M.Bitmap(R * 200.0)) == pytest.approx(1.0, abs=1e-6)     # the seal fits
    reseal = M.blur(seal, 1.0e-9, 5000.0)             # a re-melted seal blurs (pitch 2 mm)
    reseal = M.smear(reseal, 2e-3)
    assert M.ncc(reseal, M.Bitmap(R * 200.0)) < 0.97


# ================================================================ T31 legibility
def test_u_a_5_arcmin_letter_is_8_7_mm_at_6_m_and_reads():
    assert M.letter_for_distance(6.0) * 1e3 == pytest.approx(8.7, abs=0.05)
    stroke = 8.7e-3 / 5.0
    assert M.subtense_arcmin(stroke, 6.0) == pytest.approx(1.0, abs=0.01)
    assert M.legible(stroke, 6.0, 0.8, 0.05, E_lux=300.0) >= 0.99
    assert M.legible(stroke, 12.0, 0.8, 0.05, E_lux=300.0) < 0.05     # twice as far: no
    assert M.mar_arcmin(50.0) == 1.0 and M.c_min(50.0) == 0.02


def test_u_a_candle_at_30_cm_reads_2_8_mm_print_and_moonlight_does_not_at_1_m():
    E = M.lux_from_source(1.0, 0.3)
    assert E == pytest.approx(11.1, abs=0.05)
    L = M.luminance(0.8, E)
    assert L == pytest.approx(2.8, abs=0.1)
    assert M.c_min(L) == pytest.approx(0.0206, abs=0.001)
    assert M.mar_arcmin(L) == pytest.approx(2.4, abs=0.1)
    assert M.contrast(0.05, 0.8) == pytest.approx(0.94, abs=0.01)
    stroke = 0.56e-3                                                 # 2.8 mm caps
    assert M.subtense_arcmin(stroke, 0.3) == pytest.approx(6.4, abs=0.1)
    assert M.legible(stroke, 0.3, 0.8, 0.05, E_lux=E) >= 0.99        # reads by candle
    assert M.legible(stroke, 1.0, 0.8, 0.05, E_lux=0.1) < 0.01       # not in moonlight at 1 m


def test_codeonly_the_print_reads_in_moonlight_only_within_0_257_m():
    L = M.luminance(0.8, 0.1)
    assert L == pytest.approx(0.0255, abs=0.001)
    assert M.c_min(L) == pytest.approx(0.22, abs=0.01)
    assert M.mar_arcmin(L) == pytest.approx(7.5, abs=0.1)
    d = M.read_distance(0.56e-3, L)
    assert d == pytest.approx(0.257, abs=0.002)
    assert M.legible(0.56e-3, 0.99 * d, 0.8, 0.05, E_lux=0.1) >= 0.99
    assert M.legible(0.56e-3, 1.3 * d, 0.8, 0.05, E_lux=0.1) < 0.2


def test_a_pale_chalk_mark_reads_in_moonlight_only_close_up():
    rho_chalk_mark = 0.8 * (1 - 0.3)                                 # contrast 0.3
    assert M.contrast(rho_chalk_mark, 0.8) == pytest.approx(0.3)
    near = M.legible(2e-3, 0.5, 0.8, rho_chalk_mark, E_lux=0.1)
    far = M.legible(2e-3, 3.0, 0.8, rho_chalk_mark, E_lux=0.1)
    assert near >= 0.95 and far < 0.05
    # and a fainter mark (contrast 0.1) fails even close up
    assert M.legible(2e-3, 0.5, 0.8, 0.8 * 0.9, E_lux=0.1) < 0.05


def test_relief_shows_in_raking_light_and_not_in_flat_light():
    assert M.relief_contrast(0.5e-3, 10.0) > M.relief_contrast(0.5e-3, 60.0) > \
        M.relief_contrast(0.5e-3, 89.0)
    assert M.relief_contrast(0.5e-3, 5.0) == 1.0
    assert M.relief_contrast(0.5e-3, 89.0) < 0.01


# ================================================================ T32 thin film (generic)
IRON = 2.9 + 3.0j
OXIDE = 2.3


def test_codeonly_thin_film_gives_straw_brown_purple_blue_and_grey_at_zero():
    r0 = M.thin_film(OXIDE, IRON, 0.0)
    assert r0 == pytest.approx((0.52, 0.52, 0.52), abs=0.01)
    r, g, b = M.thin_film(OXIDE, IRON, 45e-9)                      # straw: red-rich
    assert r > g > b and (r, g, b) == pytest.approx((0.46, 0.41, 0.30), abs=0.02)
    r, g, b = M.thin_film(OXIDE, IRON, 55e-9)                      # brown: darker, still red-rich
    assert r > g > b and r < 0.46 and b < 0.15
    r, g, b = M.thin_film(OXIDE, IRON, 72e-9)                      # purple: red and blue over green
    assert g < r and g < b and r < 0.2
    r, g, b = M.thin_film(OXIDE, IRON, 80e-9)                      # blue
    assert b > g and b > r and b > 0.15
    r, g, b = M.thin_film(OXIDE, IRON, 95e-9)                      # light blue
    assert b > g > r


def test_the_same_function_with_other_rows_gives_another_colour():
    iron = M.thin_film(OXIDE, IRON, 72e-9)
    copper_patina = M.thin_film(2.7, 0.2 + 3.5j, 72e-9)
    oil_on_water = M.thin_film(1.47, 1.33, 300e-9)
    assert copper_patina != pytest.approx(iron, abs=0.05)
    assert oil_on_water != pytest.approx(iron, abs=0.05)
    assert len(oil_on_water) == 3 and all(0.0 <= x <= 1.0 for x in oil_on_water)
    # a single wavelength returns a single reflectance
    assert M.thin_film(OXIDE, IRON, 45e-9, 550e-9) == pytest.approx(
        M.thin_film(OXIDE, IRON, 45e-9)[1])


def test_make_content_carries_a_pointer_only():
    c = M.make_content("cipher-3", claimed_source="the Duke", true_maker="Anne",
                       payload="meet at dawn")
    assert set(c) == {"code", "claimed_source", "true_maker", "payload"}
    L = Layers()
    k = make_key(1, 3, 0)
    L.add(k, {(D.INK_GALL, D.DISSOLVED, 0): 0.01, (D.WATER, D.DISSOLVED, 0): 0.05},
          cover=1.0, Ts=20.0, content=c)
    assert L.contents[int(L.content[L.stack(k)[0]])]["claimed_source"] == "the Duke"


def test_a_footprint_in_mud_is_the_mirror_of_the_sole():
    L = Layers()
    boot = make_key(2, 4 * 64 + 2, 0)
    sole = M.Bitmap(R * 1.0)
    L.add(boot, {(D.CLAY, D.GRAIN, 8): 0.5, (D.WATER, D.DISSOLVED, 0): 0.8}, cover=1.0,
          Ts=20.0, pattern=sole)
    floor = make_key(0, 77, 4)
    mirrored = M.mirrored_top_pattern(L, boot)
    moved, _adj = L.contact(boot, [floor], c_c=0.6, p=3e4, s_slide=0.0, L_p=0.1, pattern=mirrored)
    assert moved > 0.0
    track = L.patterns[int(L.pat[L.stack(floor)[0]])]
    assert np.array_equal(track.data, M.mirror(sole).data)
    assert M.ncc(M.mirror(track), sole) == pytest.approx(1.0, abs=1e-9)    # print it again: true
    assert M.mirrored_top_pattern(L, make_key(0, 5, 5)) is None
    assert L.ledger_closes()
