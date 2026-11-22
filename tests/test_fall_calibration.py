"""Fall injury against real data (wave 14a, lane H2).

The anchors are population numbers from the fall literature. Every one is from
memory of the papers and marked (u) until it is checked against the paper:
  - Goonetilleke 1980 (Hong Kong) and Warner and Demling 1986: about half of
    those who fall four floors (about 12-14 m) die, about 90 percent at seven
    floors (about 20 m).
  - Snyder 1963: the survivors' mean drop is about 5 m (the realism map's 5.2 m).
  - Calcaneal and lumbar fractures from a feet-first fall begin at about 3-5 m
    (Essex-Lopresti 1952; Lapostolle 2006).
  - Healthy adults rarely break a rib or a hip in a fall from standing height.
"""
import math

import pytest

from src.voxel import anatomy as A
from src.voxel import impact as I


def adult():
    return A.make_sheet(mass_kg=70.0, height_m=1.75, draws=False)


def _dead_pct(h):
    hs, S = I.survival_curve(adult())
    return 100.0 * (1.0 - S[hs.index(h)])


def test_u_population_mortality_follows_the_fall_literature():
    assert _dead_pct(3.0) == pytest.approx(0.0, abs=1e-9)                                # nobody dies from 3 m
    assert _dead_pct(6.0) <= 15.0                               # a few die at two floors
    assert 20.0 <= _dead_pct(10.0) <= 35.0
    assert 40.0 <= _dead_pct(14.0) <= 60.0                      # about half at four floors
    assert _dead_pct(20.0) >= 75.0                              # most at seven floors
    fit = I.fit_numbers(adult())
    assert 12.0 <= fit["ld50_m"] <= 15.0
    assert fit["mean_survivable_m"] == pytest.approx(5.2, abs=0.3)


def test_u_mortality_curve_has_no_cliff():
    """With the bone spread the population curve climbs by less than 15 points
    per half metre anywhere (one body gives steps of 30 points)."""
    hs, S = I.survival_curve(adult())
    assert max(S[i] - S[i + 1] for i in range(len(S) - 1)) < 0.15


def test_u_a_healthy_adult_keeps_his_bones_in_a_fall_from_standing():
    s = adult()
    for mode in ("side", "back"):
        assert [f["name"] for f in I.landing(s, mode, 1.0)["failed"]] == [], mode


def test_u_serious_injury_begins_at_two_to_four_metres():
    s = adult()
    any_fx = lambda r: any(f["kind"] == "bone" for f in r["failed"])
    assert any_fx(I.landing(s, "side", 2.0)) and any_fx(I.landing(s, "back", 3.5))
    assert any_fx(I.landing(s, "feet", 3.0, share_feet=(1, 0), prep=1.0, legs_N=2060.0, crouch_m=0.286))


def test_the_bone_spread_is_a_probability_and_one_body_is_the_old_curve():
    assert math.fsum(w for w, _ in I.BONE_SPREAD) == pytest.approx(1.0, abs=1e-12)
    hs, S1 = I.survival_curve(adult(), spread=((1.0, 1.0),))
    assert S1[0] == 1.0 and S1[-1] < 0.5
