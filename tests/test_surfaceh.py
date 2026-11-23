"""surfaceh: the one surface coefficient (wave 15, lane 1b)."""
import numpy as np
import pytest

from src.voxel import surfaceh as S
from src.voxel.materials import IRON, WOOD, SIGMA
from src.voxel.thinheat import h_conv

ES, EL = S.effusivity(IRON), S.effusivity_water()


def test_cold_water_is_the_plain_single_phase_h():
    assert S.h_liquid(60.0, 20.0, 0.4, 0.8, ES, EL) == pytest.approx(float(h_conv(0.0, 0.4, 40.0, "water")))


def test_iron_boils_water_hard_then_rides_a_vapour_film():
    plain = float(h_conv(0.0, 0.4, 100.0, "water"))
    nuc = S.h_liquid(130.0, 20.0, 0.4, 0.8, ES, EL)        # nucleate: far above plain
    film = S.h_liquid(850.0, 95.0, 0.4, 0.8, ES, EL)       # a bath near boiling: H_FILM plus radiation
    assert nuc > 3 * plain
    assert 200.0 < film < 400.0 and film < plain     # the film insulates (Leidenfrost)
    assert S.h_liquid(500.0, 95.0, 0.4, 0.8, ES, EL) < nuc / 5


def test_a_cold_bath_collapses_the_vapour_blanket():
    """Wave 15 lane 2f: steel in 20 C water averages h of 3000-4000 (Grossmann H about 1 per inch),
    not the 250 of a steam film. The blend is 0 at 60 C (40 K below boiling), full at 20 C."""
    cold = S.h_liquid(850.0, 20.0, 0.4, 0.8, ES, EL)
    mid = S.h_liquid(850.0, 40.0, 0.4, 0.8, ES, EL)
    warm = S.h_liquid(850.0, 60.0, 0.4, 0.8, ES, EL)
    assert 3000.0 <= cold <= 4500.0
    assert warm < 400.0 < mid < cold


def test_a_hot_wood_surface_never_makes_a_film():
    es = S.effusivity(WOOD)
    assert S.h_liquid(850.0, 20.0, 0.4, 0.8, es, EL) == pytest.approx(float(h_conv(0.0, 0.4, 830.0, "water")))


def test_a_brine_boils_later():
    cool = S.h_liquid(125.0, 20.0, 0.4, 0.8, ES, EL, tbo=100.0)
    brine = S.h_liquid(125.0, 20.0, 0.4, 0.8, ES, EL, tbo=108.0)
    assert brine < cool


def test_air_adds_grey_radiation():
    conv = float(h_conv(0.0, 0.4, 830.0, "air"))
    assert S.h_air(850.0, 20.0, 0.4, 0.8) - conv == pytest.approx(S.radiation_h(850.0, 20.0, 0.8))
    assert S.radiation_h(850.0, 20.0, 0.8) > 60.0           # 70-100 W/m2K for a red-hot surface
    a, b = 1123.15, 293.15
    assert S.radiation_h(850.0, 20.0, 1.0) * 830.0 == pytest.approx(SIGMA * (a ** 4 - b ** 4), rel=1e-9)


def test_a_mixed_port_keeps_the_sum_of_the_face_fluxes():
    Ts = 600.0
    faces = [(2, 30.0, True), (4, 25.0, False)]
    h, T = S.port_of(Ts, faces, 0.4, IRON)
    want = (2 * S.h_liquid(Ts, 30.0, 0.4, 0.8, ES, EL) * (Ts - 30.0) / 1.0
            + 4 * S.h_air(Ts, 25.0, 0.4, 0.8) * (Ts - 25.0)) / 6.0
    assert h * (Ts - T) == pytest.approx(want, rel=0.02)    # eps of iron is the table's, ~ 0.8
    assert S.port_of(Ts, [], 0.4, IRON) is None
