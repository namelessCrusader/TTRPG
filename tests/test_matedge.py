"""The edge as state (src/voxel/matedge.py, wave 13 topic T9).

Tags: test_codeonly_* checks the code against its own constants (the d_c,
K_soft and K_hard calibrations); test_u_* leans on an unverified number.
Untagged tests are closed forms or exact accounting.
"""
import math

import numpy as np
import pytest

from src.voxel import matedge as E
from src.voxel import matrec as M

H_IRON = M.REC["iron"].H            # 1.2 GPa
H_STONE = M.REC["stone"].H          # 6 GPa
H_WOOD = M.REC["wood"].H            # 30 MPa
H_FLESH = M.REC["flesh"].H          # 10 kPa
AXE = E.EdgeGeom(0.1, math.radians(30.0), 1.0e-4)


# ── geometry ────────────────────────────────────────────────────────────────

def test_codeonly_axe_area_is_2_08e_4():
    assert E.area(AXE) == pytest.approx(2.08e-4, rel=3e-3)
    assert E.area(AXE) == pytest.approx(0.1 * (2e-4 + 2 * 3.5e-3 * math.tan(math.radians(15))), rel=1e-12)
    assert E.contact_width(1e-4, math.radians(30)) == pytest.approx(2.0756e-3, rel=1e-3)


@pytest.mark.parametrize("A", [2e-4, 5e-5, 1e-3, 3.3e-4, 1e-7])
def test_geom_from_area_round_trips_to_1e_9(A):
    g = E.geom_from_area(A)
    assert E.area(g) == pytest.approx(A, rel=1e-9)
    assert g.r_m == 1e-4 and g.theta_rad == pytest.approx(math.radians(30.0))
    g2 = E.geom_from_area(A, theta=math.radians(20.0), r0=5e-5)
    assert E.area(g2) == pytest.approx(A, rel=1e-9) and g2.theta_rad == pytest.approx(math.radians(20.0))


def test_a_declared_edge_2e_4_becomes_L_096_and_r_100_um():
    g = E.geom_from_area(2e-4)
    assert g.L_m == pytest.approx(0.0964, abs=2e-4) and g.r_m == pytest.approx(1e-4)


def test_codeonly_a_rolled_edge_r_466_um_has_A_2_81e_4_x_1_35():
    rolled = E.EdgeGeom(AXE.L_m, AXE.theta_rad, 466e-6)
    assert E.area(rolled) == pytest.approx(2.81e-4, rel=2e-3)
    assert E.area(rolled) / E.area(AXE) == pytest.approx(1.35, abs=0.01)
    assert E.area(rolled) < 1e-3                                  # still inside the "sharp" band for the mind


def test_area_follows_the_radius_exactly_A0_plus_2L_dr():
    r0 = AXE.r_m
    for r in (1.5e-4, 3e-4, 466e-6, 2e-3):
        g = E.EdgeGeom(AXE.L_m, AXE.theta_rad, r)
        assert E.area(g) == pytest.approx(E.area_after_wear(E.area(AXE), r - r0, AXE.L_m), rel=1e-12)
    assert E.area_after_wear(2e-4, 0.0, 0.1) == 2e-4


def test_codeonly_tip_area_and_the_skin_opening_energy():
    rolled = E.EdgeGeom(AXE.L_m, AXE.theta_rad, 466e-6)
    assert E.tip_area(AXE) == pytest.approx(2.0e-5, rel=1e-12)
    assert E.tip_area(rolled) == pytest.approx(9.3e-5, rel=5e-3)
    assert E.tip_area(rolled) / E.tip_area(AXE) == pytest.approx(4.66, rel=1e-3)
    assert E.C_SKIN_J_M2 == 1.04e6
    assert E.skin_open_J(AXE) == pytest.approx(20.8, rel=1e-9)
    assert E.skin_open_J(rolled) == pytest.approx(97.0, rel=0.005)
    # today's law: 1e5 J/m2 x the whole edge area, which the rescaled constant reproduces for a pristine edge
    assert E.SKIN_J_M2_OLD * E.area(AXE) == pytest.approx(E.skin_open_J(AXE), rel=0.005)
    assert E.skin_constant() == pytest.approx(1.04e6, rel=0.005)
    # any declared area, same calibration: the pristine edge opens skin at today's energy
    for A in (5e-5, 2e-4, 6e-4):
        g = E.geom_from_area(A)
        assert E.skin_constant() * E.tip_area(g) == pytest.approx(E.SKIN_J_M2_OLD * A, rel=1e-9)


def test_the_old_wound_constant_is_the_one_in_materials():
    from src.voxel import materials as MT
    assert E.SKIN_J_M2_OLD == MT.WOUND["skin_Jm2"]


# ── rolling ─────────────────────────────────────────────────────────────────

def test_codeonly_roll_50_J_on_stone_466_um():
    assert E.stopping_distance(H_IRON, H_STONE) == pytest.approx(0.447e-3, rel=1e-2)
    assert E.blow_force(50.0, H_IRON, H_STONE) == pytest.approx(112e3, rel=0.01)
    assert E.roll(AXE, 50.0, H_IRON, H_STONE) == pytest.approx(466e-6, rel=0.005)


def test_roll_none_on_wood_or_flesh():
    assert E.stopping_distance(H_IRON, H_WOOD) == pytest.approx(6.3e-3, rel=0.01)
    assert E.blow_force(50.0, H_IRON, H_WOOD) == pytest.approx(7.9e3, rel=0.01)
    assert E.roll(AXE, 50.0, H_IRON, H_WOOD) == AXE.r_m
    assert E.roll(AXE, 50.0, H_IRON, H_FLESH) == AXE.r_m
    assert E.stopping_distance(H_IRON, H_FLESH) == E.STOP_HI                # clipped at 50 mm


def test_roll_never_shrinks_and_grows_with_the_blow():
    rs = [E.roll(AXE, e, H_IRON, H_STONE) for e in (1.0, 10.0, 50.0, 200.0)]
    assert all(r >= AXE.r_m for r in rs) and rs == sorted(rs)
    rolled = E.EdgeGeom(AXE.L_m, AXE.theta_rad, 600e-6)
    assert E.roll(rolled, 1.0, H_IRON, H_STONE) == 600e-6                    # a mild blow leaves a rolled edge as it is
    assert E.roll(AXE, np.array([1.0, 50.0]), H_IRON, H_STONE).shape == (2,)


def test_a_harder_edge_rolls_less():
    steel = E.roll(AXE, 50.0, 4 * H_IRON, H_STONE)
    assert steel < E.roll(AXE, 50.0, H_IRON, H_STONE)


# ── wear ────────────────────────────────────────────────────────────────────

def test_wear_is_monotone_in_work_and_falls_with_edge_hardness():
    rs = [E.wear(AXE, w, H_IRON, H_STONE * 0.001 * 3 + H_WOOD)[0] for w in (0.0, 10.0, 100.0, 1000.0)]
    assert rs[0] == AXE.r_m and all(b > a for a, b in zip(rs, rs[1:]))
    vols = [E.wear(AXE, w, H_IRON, H_WOOD)[1] for w in (10.0, 20.0, 40.0)]
    assert vols[1] == pytest.approx(2 * vols[0]) and vols[2] == pytest.approx(2 * vols[1])
    v_soft = E.wear(AXE, 30.0, H_IRON, H_WOOD)[1]
    v_hard_edge = E.wear(AXE, 30.0, 4 * H_IRON, H_WOOD)[1]
    assert v_hard_edge == pytest.approx(v_soft / 4)                           # 1/H_edge at the same K


def test_the_wear_regime_switches_near_a_hardness_ratio_of_1():
    k = E.k_archard
    assert k(0.5) == E.K_SOFT and k(0.8) == E.K_SOFT
    assert k(1.0) == pytest.approx(0.5 * (E.K_SOFT + E.K_HARD))
    assert k(1.2) == E.K_HARD and k(10.0) == E.K_HARD
    ratios = np.linspace(0.5, 1.5, 101)
    assert np.all(np.diff(k(ratios)) >= 0)
    # in wear(): the same work on a soft target vs a target as hard as the edge
    soft = E.wear(AXE, 30.0, H_IRON, 0.5 * H_IRON)[1]
    hard = E.wear(AXE, 30.0, H_IRON, 1.5 * H_IRON)[1]
    assert hard / soft == pytest.approx(E.K_HARD / E.K_SOFT, rel=1e-9)


def test_wear_returns_the_volume_lost_and_the_radius_follows_from_it():
    r, V = E.wear(AXE, 30.0, H_IRON, H_WOOD, mu_flank=0.4)
    assert V == pytest.approx(1e-6 * 30.0 / (0.4 * H_IRON), rel=1e-9)
    assert AXE.L_m * (r ** 2 - AXE.r_m ** 2) == pytest.approx(V, rel=1e-9)     # L dr^2 IS the volume
    assert V > 0


def test_u_chopping_wood_adds_06_um2_of_r2_a_blow():
    r, V = E.wear(AXE, 30.0, H_IRON, H_WOOD)
    d_r2_um2 = (r ** 2 - AXE.r_m ** 2) * 1e12
    assert d_r2_um2 == pytest.approx(0.6, abs=0.05)
    chops = (AXE.r_m ** 2) / (r ** 2 - AXE.r_m ** 2)                           # r2 of a 100 um edge doubles in ...
    assert 12_000 < chops < 20_000
    sharp = E.EdgeGeom(AXE.L_m, AXE.theta_rad, 20e-6)
    r2, _ = E.wear(sharp, 30.0, H_IRON, H_WOOD)
    assert 500 < (20e-6) ** 2 / (r2 ** 2 - (20e-6) ** 2) < 800                 # a 20 um edge: about 640 chops


# ── honing ──────────────────────────────────────────────────────────────────

COARSE = 70e-6


def test_u_hone_an_axe_at_200_um_with_a_coarse_stone_takes_about_240_strokes():
    g = E.EdgeGeom(AXE.L_m, AXE.theta_rad, 200e-6)
    r1, V1 = E.hone(g, 1, 20.0, 0.1, H_IRON, H_STONE, COARSE)
    assert (g.r_m ** 2 - r1 ** 2) * 1e12 == pytest.approx(167.0, rel=0.01)       # um2 per stroke
    r, V = E.hone(g, 240, 20.0, 0.1, H_IRON, H_STONE, COARSE)
    assert r == pytest.approx(0.25 * COARSE, rel=0.05)
    assert E.hone(g, 200, 20.0, 0.1, H_IRON, H_STONE, COARSE)[0] > 0.25 * COARSE * 1.2


def test_hone_cannot_go_below_the_grit_floor():
    g = E.EdgeGeom(AXE.L_m, AXE.theta_rad, 200e-6)
    for grit, floor in ((COARSE, 17.5e-6), (7e-6 * 0 + 12e-6, 3e-6), (2e-6, 0.5e-6)):
        r, V = E.hone(g, 100_000, 20.0, 0.1, H_IRON, H_STONE, grit)
        assert r == pytest.approx(floor, rel=1e-9)
        assert V == pytest.approx(g.L_m * (g.r_m ** 2 - floor ** 2), rel=1e-9)  # only what was there left
    assert E.GRIT_FLOOR == 0.25


def test_hone_never_dulls_an_edge_already_finer_than_the_floor():
    fine = E.EdgeGeom(AXE.L_m, AXE.theta_rad, 5e-6)
    r, V = E.hone(fine, 100, 20.0, 0.1, H_IRON, H_STONE, COARSE)
    assert r == 5e-6 and V == 0.0


def test_hone_with_a_softer_abrasive_does_next_to_nothing():
    g = E.EdgeGeom(AXE.L_m, AXE.theta_rad, 200e-6)
    soft_stone = 0.3 * H_IRON
    r, V = E.hone(g, 100, 20.0, 0.1, H_IRON, soft_stone, COARSE)
    r_hard, V_hard = E.hone(g, 100, 20.0, 0.1, H_IRON, H_STONE, COARSE)
    assert (g.r_m - r) < 1e-4 * g.r_m and V <= 1.0001e-4 * V_hard
    assert V_hard / V == pytest.approx(E.K_HARD / E.K_SOFT, rel=1e-9)
    # a tin knife on a stone softer than tin
    tin = M.REC["tin"]
    knife = E.EdgeGeom(0.1, math.radians(25.0), 1e-4)
    r, V = E.hone(knife, 1, 20.0, 0.1, tin.H, 0.2 * tin.H, COARSE)
    r_hard, V_hard = E.hone(knife, 1, 20.0, 0.1, tin.H, 5.0 * tin.H, COARSE)
    assert V == pytest.approx(V_hard * E.K_SOFT / E.K_HARD, rel=1e-9)     # K_soft: a ten-thousandth of a hard stone's work
    assert (knife.r_m - r) < 0.01 * (knife.r_m - r_hard)


# ── filings accounting ──────────────────────────────────────────────────────

def test_filings_chunk_into_whole_grams_and_tool_plus_filings_is_constant_to_1e_12():
    rho = M.REC["iron"].rho
    tool_g = 1500.0
    m0 = tool_g
    owed, shed = 0.0, 0.0
    g = E.EdgeGeom(AXE.L_m, AXE.theta_rad, 100e-6)
    rng = np.random.default_rng(5)
    lost_total = 0.0
    for i in range(20_000):
        if i % 7 == 0:
            r, V = E.hone(g, 1, 20.0, 0.1, H_IRON, H_STONE, 5e-6)
            g = E.EdgeGeom(g.L_m, g.theta_rad, r)
        else:
            r, V = E.wear(g, float(rng.random() * 3e6) * 1e-0, H_IRON, H_IRON * 1.5)
            g = E.EdgeGeom(g.L_m, g.theta_rad, min(r, 5e-3))
        lost = E.volume_to_g(V, rho)
        lost_total += lost
        owed, n = E.filings_step(owed, lost)
        assert 0.0 <= owed < 1.0 and n >= 0
        tool_g -= n * E.FILING_CHUNK_G
        shed += n * E.FILING_CHUNK_G
        assert tool_g + shed == pytest.approx(m0, abs=1e-12)
    assert shed > 0 and shed + owed == pytest.approx(lost_total, rel=1e-9, abs=1e-9)
    assert tool_g + shed == m0                                                 # whole grams: exact in float64


def test_filings_below_one_gram_wait_in_the_tool():
    owed, n = E.filings_step(0.0, 0.4)
    assert (owed, n) == (0.4, 0)
    owed, n = E.filings_step(owed, 0.7)
    assert n == 1 and owed == pytest.approx(0.1)
    owed, n = E.filings_step(0.0, 3.5)
    assert n == 3 and owed == pytest.approx(0.5)
    assert E.volume_to_g(1e-6, 7800.0) == pytest.approx(7.8)


def test_needs_write_is_quantised_at_one_percent():
    assert not E.needs_write(1e-4, 1.005e-4) and E.needs_write(1e-4, 1.011e-4)
    assert E.needs_write(1e-4, 4.66e-4)
    assert not E.needs_write(1e-4, 1e-4)
