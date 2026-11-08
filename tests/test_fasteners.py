"""Fasteners (src/voxel/fasteners.py, wave 13 T11): screws, dowels, splitting, bars.

Tags: test_echo_* checks a number the constants were FITTED to; test_codeonly_*
checks the code against its own table; test_u_* leans on a recalled, unverified
handbook number (it reads tests/data/sourced_numbers.json when present and says
which status it saw). Untagged tests are closed forms and identities.
"""
import json
import math
import os

import numpy as np
import pytest

from src.voxel import fasteners as F

HERE = os.path.dirname(os.path.abspath(__file__))


def _sourced():
    """The status the sourced-numbers file gives (None when absent)."""
    p = os.path.join(HERE, "data", "sourced_numbers.json")
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


MM = 1e-3
G_PINE = 0.45
D10 = 4.83 * MM          # a #10 screw
INCH10 = 0.19 * F.INCH   # its nominal diameter, in the NDS's inches


def test_u_withdrawal_matches_five_times_the_NDS_allowable_within_12_percent():
    src = _sourced()
    ult = F.screw_withdrawal(G_PINE, D10, 19 * MM)
    nds = F.NDS_SAFETY * F.screw_withdrawal_allowable(G_PINE, INCH10, 0.75 * F.INCH)
    assert ult == pytest.approx(2.0e3, rel=0.01)           # 2.0 kN per #10 in pine
    assert abs(ult / nds - 1.0) < 0.12, (ult, nds, "sourced file: %s" % ("present" if src else "absent"))


def test_withdrawal_scales_as_G_squared_diameter_and_length():
    base = F.screw_withdrawal(0.4, D10, 19 * MM)
    assert F.screw_withdrawal(0.8, D10, 19 * MM) == pytest.approx(4 * base)
    assert F.screw_withdrawal(0.4, 2 * D10, 19 * MM) == pytest.approx(2 * base)
    assert F.screw_withdrawal(0.4, D10, 38 * MM) == pytest.approx(2 * base)


def test_embedment_is_the_NDS_form_and_steps_at_a_quarter_inch():
    fe = F.wood_embedment(G_PINE, D10)
    assert fe == pytest.approx(16600 * G_PINE ** 1.84 * F.PSI)
    assert fe == pytest.approx(26e6, rel=0.03)              # 26 MPa
    assert F.wood_embedment(G_PINE, 0.3 * F.INCH) == pytest.approx(11200 * G_PINE * F.PSI)


def _z_by_hand(G, D_m, Fyb):
    """The mode IV yield value of ONE screw, typed out from the NDS numbers (it does not call the
    module): Fe = 16600 G^1.84 psi (NDS 12.3.3, D under 1/4 in), Rd = K_D = 10 D + 0.5 with D in
    inches (NDS-2018 Table 12.3.1B, 0.17 in < D < 0.25 in), a steel side plate (Re = 0):
    Z = D^2 / Rd sqrt(2 Fe Fyb / 3)."""
    Fe = 16600.0 * G ** 1.84 * 6894.757293168
    Rd = 10.0 * (D_m / 0.0254) + 0.5
    return D_m ** 2 / Rd * math.sqrt(2.0 * Fe * Fyb / 3.0)


def test_u_lateral_two_strike_screws_about_4_kN_ultimate():
    """Two #10 screws, 19 mm of thread, in pine G 0.45. The spec said 3 kN, but that came from an
    Rd of 3.2 at 0.19 in: the two ends of the NDS reduction term were swapped in this file
    (decision N6). With the table as printed, K_D = 2.40 at 0.19 in, one screw's yield value is
    997 N (Fe 26.3 MPa, Fyb 600 MPa (u)) and the pair's ultimate, 2.0 x each (u), is 3.99 kN."""
    z = _z_by_hand(G_PINE, D10, 600e6)
    assert z == pytest.approx(997.0, rel=0.002)                 # the hand arithmetic of the docstring
    two = F.dowel_lateral_ultimate(G_PINE, D10, 19 * MM, n=2)
    assert two == pytest.approx(4.0 * z, rel=1e-9)              # 2 screws x 2.0 x Z
    assert two == pytest.approx(4.0e3, rel=0.05)                # about 4 kN
    # and one screw's yield value is the 0.75 to 1.1 kN of the spec
    assert 0.7e3 < F.dowel_lateral_yield(G_PINE, D10, 19 * MM) < 1.1e3


def test_u_NDS_reduction_term_Rd_is_the_table_as_printed():
    """NDS-2018 Table 12.3.1B, p. 84 (the AWC free view-only NDS, Chapter 12, read 2026-10-30).
    D under 1/4 in: K_D, 2.2 for D <= 0.17 in and 10 D + 0.5 above it. D from 1/4 in to 1 in:
    3.2 K_theta for modes IIIm, IIIs and IV, K_theta = 1 + 0.25 theta/90. The ends were swapped
    in this file (3.2 for a small screw, 2.2 for a big one) until decision N6."""
    rd = lambda d_in, **kw: float(F.dowel_Rd(d_in * F.INCH, **kw))
    assert rd(0.10) == 2.2 and rd(0.17) == 2.2                 # K_D, flat up to 0.17 in
    assert rd(0.1701) == pytest.approx(2.201, abs=1e-9)        # and continuous where the line starts
    assert rd(0.19) == pytest.approx(2.4) and rd(0.20) == pytest.approx(2.5)   # 10 D + 0.5
    assert rd(0.249) == pytest.approx(2.99)                    # 3.0 at 1/4 in from below ...
    assert rd(0.25) == 3.2 and rd(0.5) == 3.2 and rd(1.0) == 3.2   # ... and the step to 3.2 K_theta
    assert rd(0.5, theta_deg=90.0) == pytest.approx(4.0)       # across the grain K_theta = 1.25
    assert rd(0.5, theta_deg=45.0) == pytest.approx(3.2 * 1.125)
    assert rd(0.19, theta_deg=90.0) == rd(0.19)                # under 1/4 in the table has no K_theta
    assert rd(0.19) < rd(0.5)                                  # a SMALL screw gets the SMALL term
    D = np.linspace(0.5e-3, 25e-3, 500)
    assert (np.diff(F.dowel_Rd(D)) >= 0).all()                 # a bigger dowel never gets a smaller one
    with pytest.raises(ValueError):
        F.dowel_Rd(0.01, theta_deg=120.0)


def test_the_default_Rd_follows_the_diameter_in_every_entry_point():
    for D in (3.5 * MM, D10, 6.0 * MM, 8.0 * MM, 12.7 * MM, 20.0 * MM):
        Rd = float(F.dowel_Rd(D))
        Fe = F.wood_embedment(G_PINE, D)
        z = F.dowel_mode_iv(Fe, F.FYB_SCREW, D, Rd=Rd)
        assert F.dowel_mode_iv(Fe, F.FYB_SCREW, D) == pytest.approx(z, rel=1e-12)
        assert F.dowel_lateral_yield(G_PINE, D, 1.0) == pytest.approx(z, rel=1e-12)   # a long screw: mode IV
        assert F.lateral_mode(G_PINE, D, 1.0) == "iv"


def test_u_lateral_a_number_8_screw_sits_on_the_flat_part_of_K_D():
    """A #8 is 4.2 mm = 0.165 in, under the 0.17 in break, so Rd = 2.2 exactly: one screw
    823 N, two ultimate 3.29 kN (the deadbolt strike of mechlaw.py)."""
    D8 = 4.2 * MM
    assert float(F.dowel_Rd(D8)) == 2.2
    Fe = 16600.0 * G_PINE ** 1.84 * 6894.757293168
    z = D8 ** 2 / 2.2 * math.sqrt(2.0 * Fe * 600e6 / 3.0)
    assert z == pytest.approx(823.0, rel=0.002)
    assert F.dowel_lateral_ultimate(G_PINE, D8, 75 * MM, n=2) == pytest.approx(4.0 * z, rel=1e-9)


def test_mode_IV_caps_the_screw_so_a_longer_one_adds_nothing():
    z19 = F.dowel_lateral_yield(G_PINE, D10, 19 * MM)
    z50 = F.dowel_lateral_yield(G_PINE, D10, 50 * MM)
    assert F.lateral_mode(G_PINE, D10, 50 * MM) == "iv"
    assert z50 == pytest.approx(F.dowel_mode_iv(F.wood_embedment(G_PINE, D10), F.FYB_SCREW, D10))
    assert z50 >= z19
    z100 = F.dowel_lateral_yield(G_PINE, D10, 100 * MM)
    assert z100 == z50
    # a stub of thread is bearing-limited
    assert F.lateral_mode(G_PINE, D10, 5 * MM) == "bearing"


def test_mode_IV_closed_form_and_the_side_member_term():
    fe, fy = 26e6, 600e6
    z = F.dowel_mode_iv(fe, fy, D10, Rd=3.2)
    assert z == pytest.approx(D10 ** 2 / 3.2 * math.sqrt(2 * fe * fy / 3))
    # wood on wood (Re = 1) is weaker than a steel side plate (Re = 0): 1/sqrt 2
    assert F.dowel_mode_iv(fe, fy, D10, 3.2, Re=1.0) == pytest.approx(z / math.sqrt(2))


def test_u_jamb_split_2_9_kN_and_7_2_with_a_stud():
    sigma = 3e6                                              # the bond table's wood across the grain
    assert F.split_across_grain(sigma, 0.08, 0.012) == pytest.approx(2.9e3, rel=0.01)
    assert F.split_across_grain(sigma, 0.08, 0.030) == pytest.approx(7.2e3, rel=0.01)


def test_u_bar_38_by_89_on_1200_mm_holds_6_7_kN_and_less_on_the_flat():
    edge = F.bar_bending(40e6, 0.038, 0.089, 1.2)
    flat = F.bar_bending(40e6, 0.089, 0.038, 1.2)
    assert edge == pytest.approx(6.7e3, rel=0.01)
    assert flat / edge == pytest.approx(0.038 / 0.089, rel=1e-9)        # about 0.43
    assert 0.42 < flat / edge < 0.44


def test_bar_holds_the_smallest_of_its_three_and_says_which():
    assert F.bar_holds(6.7e3, 6.0e3, 9e3) == (6.0e3, "cleats")
    assert F.bar_holds(6.7e3, 8.0e3, 9e3) == (6.7e3, "beam")
    assert F.bar_holds(6.7e3, 8.0e3, 2e3) == (2.0e3, "leaf")


def test_round_pin_shear_and_tension():
    assert F.shear_round(250e6, 12 * MM) == pytest.approx(17e3, rel=0.02)        # 17 kN
    assert F.tension_round(250e6, 12 * MM, n=2) == pytest.approx(2 * 250e6 * math.pi * 0.012 ** 2 / 4)


def test_punching_is_stress_times_perimeter_times_the_plies():
    assert F.punching(5e6, 0.5, 0.003) == pytest.approx(7.5e3)
    assert F.punching(5e6, 1.0, 0.003) == pytest.approx(2 * F.punching(5e6, 0.5, 0.003))


def test_stiffness_formulas_series_and_referred():
    E, b, t, L = 11e9, 0.04, 0.02, 0.5
    assert F.cantilever_k(E, F.rect_I(b, t), L) == pytest.approx(3 * E * b * t ** 3 / 12 / L ** 3)
    assert F.axial_k(E, b * t, L) == pytest.approx(E * b * t / L)
    assert F.crush_k(5.5e8, 1e-3, 0.045) == pytest.approx(5.5e8 * 1e-3 / 0.045)
    assert F.series_k(2.0, 2.0) == pytest.approx(1.0)
    assert F.series_k(1e6, 1e-3) == pytest.approx(1e-3, rel=1e-3)
    assert F.series_k(3.0, 0.0) == 0.0 and F.series_k() == 0.0           # a broken link holds nothing
    assert F.referred_k(1e5, 0.85, 0.85) == 1e5
    assert F.referred_k(1e5, 0.85, 1.7) == pytest.approx(0.25e5)


def test_damage_scales_capacity_and_stiffness_as_one_minus_D():
    assert F.scale_damage(1000.0, 0.0) == 1000.0
    assert F.scale_damage(1000.0, 0.25) == pytest.approx(750.0)
    assert F.scale_damage(1000.0, 1.0) == 0.0


def test_codeonly_the_weakest_of_the_strike_chain_flips_with_the_screw_mode():
    """No stud: the jamb split (2.9) is under the screws either way. With a
    stud the split is 7.2 kN. The order of the rest is not independent of the
    screw mode. With the NDS Rd the lateral reading of two #10 screws is
    3.99 kN, so it is ABOVE the 3.4 kN keeper plate (the plate goes first),
    while a withdrawal reading of the same screws with 12 mm of thread is
    2.54 kN, BELOW it (the screws go first). (With 19 mm of thread the
    withdrawal reading is 4.02 kN, too close to the 3.99 to tell apart, so
    this chain uses 12 mm; before decision N6 the lateral reading was 3.0 kN
    and the winners were the other way round.)"""
    plate = 3.4e3
    split = F.split_across_grain(3e6, 0.08, 0.030)
    lat = F.dowel_lateral_ultimate(G_PINE, D10, 12 * MM, n=2)
    wd = 2 * F.screw_withdrawal(G_PINE, D10, 12 * MM)
    assert lat == pytest.approx(3.99e3, rel=0.01) and wd == pytest.approx(2.54e3, rel=0.01)
    first = lambda scr: min((scr, "screws"), (plate, "plate"), (split, "split"))[1]
    assert first(lat) == "plate" and first(wd) == "screws"
