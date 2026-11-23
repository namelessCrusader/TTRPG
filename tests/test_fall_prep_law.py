"""D3 fall-preparation law (wave 15, lane 2c): brace ramp, re-orientation ramp, 62 percent floor."""
import math

import pytest

from src.voxel import anatomy as A
from src.voxel import impact as I


def _adult():
    return A.make_sheet(mass_kg=70.0, height_m=1.75, draws=False)


def test_ramp_shape_onset_full_and_cap():
    assert I.prep_from_body(0.10) == 0.0                      # nothing before the muscle onset
    assert I.prep_from_body(I.BRACE_T0_S) == 0.0
    full_t = I.BRACE_T0_S + I.BRACE_RAMP_S
    assert full_t == pytest.approx(0.40)                      # full near 0.40 s (Zhang 2017 line: 0.42)
    assert I.prep_from_body(full_t) == pytest.approx(I.BRACE_U)
    assert I.prep_from_body(2.0) == pytest.approx(I.BRACE_U)
    mid = I.prep_from_body(I.BRACE_T0_S + 0.5 * I.BRACE_RAMP_S)
    assert mid == pytest.approx(0.5 * I.BRACE_U)              # linear between the two


def test_the_ramp_is_not_the_old_flat_cap():
    assert not hasattr(I, "BRACE_UNFORESEEN")
    assert I.prep_from_body(0.3) > 0.0                        # the old law gave 0.0 before 0.25 s plus its ramp


def test_foreseen_and_unconscious():
    assert I.prep_from_body(0.05, True, True) == 1.0
    assert I.prep_from_body(0.9, False, False) == 0.0


def test_onset_follows_the_reaction_dial():
    assert I.brace_onset_s(0.25) == pytest.approx(0.125)
    assert I.brace_onset_s(0.45) > I.brace_onset_s(0.20)
    assert I.brace_onset_s(0.0) >= I.ONSET_MIN_S


def test_residual_at_the_3_m_fall_is_in_the_library_band():
    """Unforeseen / foreseen force at 3 m (0.78 s of air): 1.15 to 1.25 times."""
    sh = _adult()
    un = I.landing(sh, "feet", 3.0, foreseen=False)["F_peak"]
    fo = I.landing(sh, "feet", 3.0, foreseen=True)["F_peak"]
    assert 1.15 <= un / fo <= 1.25


def test_force_falls_with_air_time_over_the_ramp_at_a_high_drop():
    """More warning, less force (fitted drop heights where the legs are the limit)."""
    sh = _adult()
    fs = [I.landing(sh, "feet", 3.0, t_air_s=t, foreseen=False)["F_peak"] for t in (0.1, 0.2, 0.3, 0.4)]
    assert fs == sorted(fs, reverse=True) and fs[0] > fs[-1]


def test_reorient_ramp_centre_sd_and_end():
    assert I.reorient_share(0.0) == 0.0 or I.reorient_share(0.0) < 1e-3
    assert I.reorient_share(I.REORIENT_DONE_S) == 1.0
    assert I.reorient_share(1.5) == 1.0
    assert I.reorient_share(0.68) == pytest.approx(0.5 / (0.5 * (1 + math.erf((0.85 - 0.68) / 0.17 / math.sqrt(2)))))
    assert I.reorient_share(0.78, aware=False) == 0.0
    # a surprised 3 m faller (0.78 s) manages it about 70 percent of the time (library S3)
    assert 0.65 <= I.reorient_share(0.782) <= 0.90
    xs = [I.reorient_share(t / 20.0) for t in range(0, 30)]
    assert xs == sorted(xs)


def test_even_an_unprepared_fall_sheds_62_percent():
    assert I.shed_floor() == pytest.approx(0.617, abs=0.003)
    assert I.shed_floor() == pytest.approx(0.62, abs=0.01)
    assert I.REORIENT_MU_S == 0.68                            # hand impact at 680 ms


@pytest.mark.xfail(strict=True, reason="the fuse chain's soft element (4 BW over 4 cm) is tuned for 1-12 m falls: "
                   "at 30-60 cm drops it gives 7-11 BW unprepared and a prepared plateau of 4.5 BW, "
                   "so the 1.80/1.54/1.31 ratios of Zhang 2017 cannot be reproduced (lane 2c open problem)")
def test_zhang_2017_ratios_at_30_45_60_cm():
    sh = _adult()
    for h, want in ((0.30, 1.80), (0.45, 1.54), (0.60, 1.31)):
        un = I.landing(sh, "feet", h, foreseen=False)["F_peak"]
        fo = I.landing(sh, "feet", h, foreseen=True)["F_peak"]
        assert un / fo == pytest.approx(want, rel=0.1)
