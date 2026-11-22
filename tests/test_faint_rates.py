"""Wave 14b, lane H: what faints a person, held to the sources (faintrates.py)."""
import math

import pytest

from src.voxel import faintrates as F, injury as IJ, vasculature as VA


def test_blood_loss_faints_in_the_ATLS_class_three_band_and_kills_in_class_four():
    assert 0.30 <= VA.faint_loss_frac() <= 0.40       # class III: 30 to 40 percent
    assert 0.40 < VA.death_loss_frac() <= 0.55        # class IV: over 40 percent


def test_the_vasovagal_dial_most_need_a_lot_and_a_few_faint_easily():
    assert F.vagal_share_below(0.134) < 0.01          # at the clip: under one person in a hundred
    assert 0.08 <= F.vagal_share_below(0.3) <= 0.16   # a strong pain: about 12 percent (u)
    assert F.vagal_share_below(1.5) > 0.9
    # the closed form is the law's own distribution
    n = 4000
    draws = [IJ.vagal_threshold({"pid": i, "name": "n%d" % i}) for i in range(n)]
    got = sum(d < 0.3 for d in draws) / n
    assert abs(got - F.vagal_share_below(0.3)) < 0.03
    assert min(draws) >= IJ.VAGAL_MEDIAN * math.exp(-IJ.VAGAL_SIGMA * IJ.VAGAL_CLIP) - 1e-9


def test_the_oxygen_curve_is_one_in_fresh_air_flat_near_it_and_steep_low():
    assert F.o2_target(1.0) == 1.0 and F.o2_target(1.3) == 1.0 and F.o2_target(0.0) == 0.0
    assert 0.9 < F.o2_target(0.9) < 1.0              # a room at 90 percent costs little
    assert F.o2_target(0.9999998978206088) == 0.9999998978206088     # a float32 'fresh' is read as before
    assert F.o2_target(0.25) < 0.5 < F.o2_target(0.38)
    xs = [i / 50 for i in range(51)]
    assert all(F.o2_target(a) <= F.o2_target(b) for a, b in zip(xs, xs[1:]))


def test_high_altitude_times_sit_in_the_FAA_ranges_where_the_curve_reaches_the_line():
    t = {a: m for a, _e, _f, m, _o in F.hypoxia_table()}
    assert 100 <= t[28000.0] <= 200                   # FAA 2.5 to 3 minutes
    assert 60 <= t[30000.0] <= 120                    # FAA 1 to 2 minutes
    assert 30 <= t[35000.0] <= 60                     # FAA 30 to 60 seconds
    # 18000 ft: the FAA gives 20 to 30 minutes; the straight line fainted at 74 s
    old = {a: o for a, _e, _f, _m, o in F.hypoxia_table()}
    assert old[18000.0] < 100 and t[18000.0] > 600


@pytest.mark.xfail(strict=True, reason="no hyperventilation or brain-store lag: " + F.HYPOXIA_GAP)
def test_FAA_time_at_25000_ft_is_three_to_five_minutes():
    t = {a: m for a, _e, _f, m, _o in F.hypoxia_table()}
    assert 150 <= t[25000.0] <= 330


def test_breathe_chases_the_curve_not_the_fraction():
    from src.voxel.sim import World
    from src.voxel.scenes import _person
    from src.voxel.materials import STONE
    w = World(20, 20, 20, voxel_cm=5)
    w.fill(0, 20, 0, 20, 0, 1, STONE)
    p = _person(w, 10, 10)
    w.step()
    w.step()
    p["blood_o2"] = 0.5
    p["_eye"] = (10.0, 10.0, 5.0)
    p["_met_at"] = (10.0, 10.0)
    p["awake"] = True
    w._breathe(p, True, 0.9, 0.0)
    step = p["blood_o2"] - 0.5
    assert step == pytest.approx((F.o2_target(0.9) - 0.5) * 0.025 / 40.0, rel=1e-9)
