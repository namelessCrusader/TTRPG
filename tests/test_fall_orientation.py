"""Fall injury by orientation, intent and surface (wave 14a refinement, lane R2).

Source: need5 S2.1 (doi:10.1038/s41598-021-95226-w): against a feet-first landing the
survival odds ratio was 0.11 (0.02-0.64) lateral and 0.17 (0.03-0.99) dorsal; soft ground
favoured survival (OR 2.38). No ratio is typed into the law: the mode tables give them.
"""
import numpy as np
import pytest

from src.voxel import anatomy as A
from src.voxel import impact as I


def adult():
    return A.make_sheet(mass_kg=70.0, height_m=1.75, draws=False)


def _body(axis, n=40, length=17.0):
    """Cells of a body along `axis` (unit vector), feet at the origin, head at +length."""
    t = np.linspace(0.0, length, n)
    return np.outer(t, np.asarray(axis, float)) + np.array([50.0, 50.0, 3.0])


EYE = (50.0, 50.0, 20.0)         # where the head was, standing


def test_upright_lands_on_feet_and_inverted_on_head():
    assert I.orientation(_body((0, 0, 1)), EYE, (1, 0)) == "feet"
    inv = _body((0, 0, 1))
    inv[:, 2] = 3.0 + 17.0 - (inv[:, 2] - 3.0)      # head down
    assert I.orientation(inv, (50.0, 50.0, 6.0), (1, 0)) == "head"   # the head is the end that was just above


@pytest.mark.parametrize("axis,facing,want", [
    ((1, 0, 0), (1, 0), "front"),      # fell the way it faced
    ((-1, 0, 0), (1, 0), "back"),      # fell backwards
    ((0, 1, 0), (1, 0), "side"),       # fell across its facing
    ((0, -1, 0), (1, 0), "side"),
    ((0, 1, 0), (0, 1), "front"),      # same fall, another facing: another landing
])
def test_lying_body_is_told_by_the_way_it_faced(axis, facing, want):
    assert I.orientation(_body(axis), EYE, facing) == want


def test_without_a_facing_a_lying_body_is_a_side_landing_as_before():
    assert I.orientation(_body((-1, 0, 0)), EYE) == "side"
    assert I.orientation(_body((-1, 0, 0)), None, (1, 0)) == "side"


def test_u_model_odds_ratios_are_inside_the_papers_intervals():
    ors, surv = I.odds_ratios(adult())
    assert 0.02 <= ors["side"] <= 0.64        # lateral vs feet-first, paper 0.11 (0.02-0.64)
    assert 0.03 <= ors["back"] <= 0.99        # dorsal vs feet-first, paper 0.17 (0.03-0.99)
    assert surv["feet"] > surv["side"] and surv["feet"] > surv["back"]


def test_orientation_multiplies_the_same_energy():
    sh = adult()
    for h in (6.0, 10.0):
        feet = I.landing(sh, "feet", h)
        side = I.landing(sh, "side", h)
        # same drop; a lying landing breaks different, softer things at a far smaller force
        assert side["F_peak"] < feet["F_peak"]
    assert I.outcome(I.landing(sh, "back", 14.0), sh)["dead"]
    assert not I.outcome(I.landing(sh, "feet", 14.0), sh)["dead"]


def test_front_reads_the_side_table():
    sh = adult()
    a, b = I.landing(sh, "front", 6.0), I.landing(sh, "side", 6.0)
    assert a["F_peak"] == b["F_peak"] and a["mode"] == "side"


def test_a_prepared_jump_lands_softer_than_the_same_short_fall_by_accident():
    sh = adult()
    acc = I.landing(sh, "feet", 3.0, t_air_s=0.05)                 # a slip: no time to get ready
    jump = I.landing(sh, "feet", 3.0, t_air_s=0.05, intended=True)
    assert acc["prep"] == 0.0 and jump["prep"] == 1.0
    assert jump["F_peak"] < acc["F_peak"]
    assert len(jump["failed"]) <= len(acc["failed"])
    # the unconscious are never prepared, whatever they chose
    assert I.landing(sh, "feet", 3.0, t_air_s=0.05, intended=True, aware=False)["prep"] == 0.0


def test_intent_is_read_from_the_leap_stamp_and_goes_stale():
    ts = 0.025
    p = {"_chose": {"leap": 102}}                 # chosen at tick 100, cover to 102
    assert I.intended(p, 130, ts)
    assert not I.intended(p, 100 + int(I.INTENT_WINDOW_S / ts) + 50, ts)
    assert not I.intended({}, 10, ts)
    assert not I.intended({"_chose": {"grab": 5}}, 6, ts)


def test_hard_ground_is_the_old_landing_bit_for_bit():
    sh = adult()
    for mat in (1, 2, 3):                          # wood, stone, iron
        assert I.ground_of(mat, 1.0) is None
    for h in (2.0, 5.0, 9.0):
        a = I.landing(sh, "feet", h)
        b = I.landing(sh, "feet", h, ground=I.ground_of(2, 1.0))
        assert a["F_peak"] == b["F_peak"] and a["work"] == b["work"]


def test_soft_ground_is_a_sink_and_favours_feet_first_survival():
    sh = adult()
    soil = (3.0e5, None)                           # firm soil, 300 kPa bearing (u)
    hard = I.landing(sh, "feet", 8.0)
    soft = I.landing(sh, "feet", 8.0, ground=soil)
    assert soft["F_peak"] < hard["F_peak"]
    assert not soft["failed"] and hard["failed"]   # the heels and the lumbar spine hold
    assert soft["work"]["ground_J"] > 0.0
    w = soft["work"]
    assert w["total_J"] == pytest.approx(w["W_h_X_J"], rel=1e-9)    # every joule has a sink
    # a thin soft layer over a floor gives only its depth
    thin = I.landing(sh, "feet", 8.0, ground=(3.0e5, 0.02))
    assert hard["F_peak"] >= thin["F_peak"] >= soft["F_peak"]


def test_a_ground_that_cannot_hold_a_body_absorbs_nothing():
    sh = adult()
    g = I.ground_of(5, 1.0)                        # loose ash, 10 kPa
    assert g is not None
    assert I.landing(sh, "feet", 5.0, ground=g)["F_peak"] == I.landing(sh, "feet", 5.0)["F_peak"]


def test_roll_keeps_the_ground_in_every_stage():
    sh = adult()
    r = I.landing(sh, "feet", 14.0, shares=2, ground=(3.0e5, None))
    assert r["work"].get("ground_J", 0.0) > 0.0
