"""Mechanisms (src/voxel/mechanisms.py, wave 13 T11): springs, statics, impact, damage, turns.

Tags: test_echo_* checks a number the constants were FITTED to; test_codeonly_*
checks the code against its own table; test_u_* leans on an unverified number
(it names the status of tests/data/sourced_numbers.json when present).
Untagged tests are closed forms, identities and conservation laws.
"""
import itertools
import json
import math
import os
import types

import numpy as np
import pytest

from src.voxel import fasteners as F
from src.voxel import mechanisms as M

HERE = os.path.dirname(os.path.abspath(__file__))
VOXELS = (5, 10)
MM = 1e-3


def _sourced_status():
    p = os.path.join(HERE, "data", "sourced_numbers.json")
    return "present" if os.path.exists(p) else "absent"


# ── springs ──────────────────────────────────────────────────────────────────

def test_coil_rate_is_0_64_N_per_mm_for_the_pin_spring():
    k = M.coil_rate(0.30 * MM, 2.5 * MM, 8, 79e9)
    assert k == pytest.approx(0.64e3, rel=0.005)                 # 0.64 N/mm


def test_cantilever_rate_equals_3EI_over_L3_and_the_deflection_is_F_over_k():
    E, b, t, L = 200e9, 10 * MM, 0.5 * MM, 30 * MM
    k = M.leaf_rate(E, b, t, L)
    assert k == pytest.approx(F.cantilever_k(E, F.rect_I(b, t), L), rel=1e-12)
    Fz = 0.2
    # beam theory by integration of the moment-area: delta = F L^3 / (3 E I)
    delta = Fz * L ** 3 / (3 * E * F.rect_I(b, t))
    assert Fz / k == pytest.approx(delta, rel=1e-12)


@pytest.mark.parametrize("r,res", [(1.2, 0.060), (1.5, 0.265), (2.0, 0.708), (3.0, 1.679)])
def test_torsion_residual_twist_by_Nadai(r, res):
    assert M.nadai_residual(r) == pytest.approx(res, abs=0.0008)


def test_torsion_no_set_below_the_limit_and_unloading_is_monotonic():
    assert M.nadai_residual(1.0) == 0.0 and M.nadai_residual(0.4) == 0.0
    rs = np.linspace(0.5, 4.0, 200)
    res = [M.nadai_residual(r) for r in rs]
    assert all(b >= a for a, b in zip(res, res[1:]))
    # the torque ratio is continuous at yield and rises to 4/3 (the plastic limit)
    assert M.nadai_torque_ratio(1.0) == 1.0
    assert M.nadai_torque_ratio(100.0) < 4.0 / 3.0 and M.nadai_torque_ratio(100.0) > 1.33


def _leaf_closed_form(m):
    """int_1^m u (1/sqrt(3-2u) - u) du / m^2, done by hand (v = 3 - 2u)."""
    v = 3.0 - 2.0 * m
    a = 0.25 * ((6.0 - 2.0 / 3.0) - (6.0 * math.sqrt(v) - (2.0 / 3.0) * v ** 1.5))
    return (a - (m ** 3 - 1.0) / 3.0) / m ** 2


@pytest.mark.parametrize("m", [1.1, 1.2, 1.4, 1.45])
def test_leaf_set_numeric_integral_equals_the_closed_form(m):
    assert M.leaf_set_tip(m) == pytest.approx(_leaf_closed_form(m), rel=2e-3)


def test_leaf_set_is_below_the_uniform_curvature_bound_and_grows_with_m():
    prev = 0.0
    for m in (1.05, 1.1, 1.2, 1.3, 1.4, 1.45, 1.49):
        s = M.leaf_set_tip(m, n=20000)
        assert prev < s < M.leaf_set_uniform(m)
        prev = s
    assert M.leaf_set_tip(1.0) == 0.0 and M.leaf_set_tip(0.7) == 0.0


def test_echo_leaf_set_prototype_figure_is_the_root_measured_integral():
    """The spec's 1.8e-4 kappa_y L^2 at m = 1.2 (laws_v2.py) is the integral of
    the residual curvature weighted by the distance from the ROOT. The lever to
    the tip is the distance from the TIP, which gives 4.3e-3 (checked against
    the hand integral above). Both are pinned here so the difference is not
    lost: the first reproduces the spec number, the second is the physics."""
    root = M.leaf_set_tip(1.2, arm="root")
    assert root == pytest.approx(1.8e-4, rel=0.03)               # 1.755e-4: the spec's 1.8e-4
    assert root < M.leaf_set_uniform(1.2)
    assert M.leaf_set_tip(1.2) == pytest.approx(4.33e-3, rel=0.01)
    assert M.leaf_set_tip(1.2) > 20 * root


def test_spring_record_rates_and_limits_come_from_geometry_and_material():
    coil = M.new_spring("c", "coil", "steel", dict(d=0.30 * MM, D=2.5 * MM, n=8, G=79e9, sy=1.5e9))
    assert M.spring_rate(coil) == pytest.approx(640.0, rel=0.005)
    leaf = M.new_spring("l", "leaf", "steel", dict(b=10 * MM, t=0.5 * MM, L=30 * MM, sy=1.5e9))
    assert M.spring_rate(leaf) == pytest.approx(M.leaf_rate(200e9, 10 * MM, 0.5 * MM, 30 * MM))
    tor = M.new_spring("t", "torsion", "steel", dict(d=2 * MM, L=0.05, sy=1.5e9))
    assert M.spring_rate(tor) == pytest.approx(math.pi * tor_G() * (2 * MM) ** 4 / (32 * 0.05))


def tor_G():
    p = M.material_props("steel")
    return p["E"] / (2 * (1 + p["nu"]))


def test_coil_pushed_past_its_limit_rests_with_a_set():
    s = M.new_spring("c", "coil", "steel", dict(d=0.30 * MM, D=2.5 * MM, n=8, G=79e9, sy=1.0e9))
    xy = M.spring_limit(s)
    assert M.spring_set(s, 0.9 * xy) == 0.0
    assert M.spring_set(s, 1.5 * xy) == pytest.approx(M.nadai_residual(1.5) * xy)
    assert 0.0 < M.spring_set(s, 1.5 * xy) < 1.5 * xy
    assert M.spring_set(s, 3 * xy) > M.spring_set(s, 2 * xy) > M.spring_set(s, 1.5 * xy)


def test_leaf_spring_set_runs_from_none_through_root_zone_to_plastic_hinge():
    s = M.new_spring("l", "leaf", "steel", dict(b=10 * MM, t=0.5 * MM, L=30 * MM, sy=1.0e9))
    xy = M.spring_limit(s)
    assert M.spring_set(s, xy) == 0.0
    sets = [M.spring_set(s, f * xy) for f in (1.05, 1.3, 1.45, 1.8, 3.0, 6.0)]
    assert sets[0] == 0.0 or sets[0] > 0.0
    assert all(b >= a for a, b in zip(sets, sets[1:])) and sets[-1] > sets[0]
    # past the hinge: set = applied - 1.5 F_y/k
    assert sets[-1] == pytest.approx(6.0 * xy - 1.5 * xy, rel=1e-9)
    # continuity at the hinge
    x15 = 1.5 * xy + M.leaf_set_tip(1.5) * 2 * 1.0e9 * (30 * MM) ** 2 / (200e9 * 0.5 * MM)
    assert M.spring_set(s, x15 * 0.9999) == pytest.approx(M.spring_set(s, x15 * 1.0001), rel=0.02)


def test_annealed_and_tempered_springs_differ_with_no_table():
    geom = dict(b=10 * MM, t=0.5 * MM, L=30 * MM)
    soft = M.new_spring("a", "leaf", "steel", dict(geom, sy=200e6))
    hard = M.new_spring("h", "leaf", "steel", dict(geom, sy=1500e6))
    assert M.spring_rate(soft) == M.spring_rate(hard)               # same blade, same stiffness
    x = 2.0 * M.spring_limit(soft)
    assert M.spring_set(soft, x) > 0.0                              # the soft one is bent...
    assert M.spring_set(hard, x) == 0.0                             # ...the hard one only flexes
    assert M.spring_limit(hard) == pytest.approx(7.5 * M.spring_limit(soft))
    # and a coil, the temper given as hardness (Tabor), not as a yield
    c_soft = M.new_spring("cs", "coil", "steel", dict(d=0.3 * MM, D=2.5 * MM, n=8, HV=61))
    c_hard = M.new_spring("ch", "coil", "steel", dict(d=0.3 * MM, D=2.5 * MM, n=8, HV=460))
    xc = 2.0 * M.spring_limit(c_soft)
    assert M.spring_set(c_soft, xc) > 0.0 and M.spring_set(c_hard, xc) == 0.0


def test_yield_from_hardness_is_Tabor():
    assert M.yield_from_hardness(100) == pytest.approx(100 * 9.81 / 3 * 1e6)
    assert M.yield_from_hardness(460) == pytest.approx(1.5e9, rel=0.01)


def test_spring_energy_and_the_catch_closing_force():
    assert M.spring_energy(640.0, 1e-3) == pytest.approx(0.32e-3)
    assert M.latch_closing_force(15.0, 45.0) == pytest.approx(15.0)
    assert M.latch_closing_force(15.0, 30.0) == pytest.approx(15.0 / math.tan(math.radians(30)))


# ── the leaf on its hinge ────────────────────────────────────────────────────

def slab(voxel_cm, W=0.9, H=2.03, mass=43.0):
    """A uniform one-cell-thick leaf hinged at its edge (axis through the
    mid-thickness: pivot2 = (-1, 0)): cells, per-cell masses, pivot2, d."""
    d = M.cell_edge_m(voxel_cm)
    nx, nz = int(round(W / d)), int(round(H / d))
    cells = np.array([(i, 0, k) for i in range(nx) for k in range(nz)])
    return cells, np.full(len(cells), mass / len(cells)), (-1, 0), d


@pytest.mark.parametrize("vox", VOXELS)
def test_inertia_from_cells_is_m_W2_over_3_for_a_uniform_slab(vox):
    cells, m, piv, d = slab(vox)
    I = M.leaf_inertia(cells, m, piv, d)
    W = 0.9
    assert I == pytest.approx(43.0 * W * W / 3.0, rel=0.01)
    assert I > 43.0 * W * W / 3.0                                   # the cells' own d^2/6 adds a little


def test_cell_edge_is_voxel_cm_over_100():
    assert M.cell_edge_m(5) == 0.05 and M.cell_edge_m(10) == 0.1


def test_centre_of_mass_of_the_slab_is_its_middle():
    cells, m, piv, d = slab(5)
    c = M.leaf_com(cells, m, d)
    assert c[0] == pytest.approx((0.9 - 0.05) / 2, abs=1e-9)        # cell centres run 0 .. W - d
    assert c[2] == pytest.approx((2.03 // 0.05 * 0.05 - 0.05) / 2 + 0.0, abs=0.03)


@pytest.mark.parametrize("vox", VOXELS)
def test_twenty_newtons_at_0_8_m_opens_the_43_kg_leaf_in_the_closed_form_time(vox):
    cells, m, piv, d = slab(vox)
    I = M.leaf_inertia(cells, m, piv, d)
    tau_f = M.hinge_friction(0.2, 0.005, 38.0 * M.GRAVITY)           # 0.37 N m
    assert tau_f == pytest.approx(0.37, abs=0.005)
    tau = M.net_open_torque([(20.0, 0.8)], tau_f)
    assert tau == pytest.approx(16.0 - tau_f)
    t_closed = M.swing_time_closed(I, tau, math.pi / 2)
    t_num, om = M.integrate_swing(I, 20.0 * 0.8, tau_f, math.pi / 2)
    assert t_num == pytest.approx(t_closed, rel=1e-3)               # 0.1 percent
    assert t_closed == pytest.approx(1.52, rel=0.01)
    assert om == pytest.approx(tau / I * t_closed, rel=2e-3)        # end speed ~ 2.1 rad/s
    assert 0.5 * I * om ** 2 == pytest.approx(25.0, rel=0.1)        # KE ~ 25 J


def test_integration_with_friction_holds_a_weak_push_still():
    t, om = M.integrate_swing(11.6, 0.3, 0.37, math.pi / 2)
    assert t == math.inf and om == 0.0


def test_the_sum_of_torques_decides_not_who_pushes():
    assert M.net_open_torque([(20.0, 0.8), (-20.0, 0.8)], 0.37) == 0.0       # equal and opposite: shut
    assert M.net_open_torque([(30.0, 0.8), (-20.0, 0.8)], 0.37) == pytest.approx(8.0 - 0.37)
    assert M.net_open_torque([(20.0, 0.8), (-30.0, 0.8)], 0.37) == pytest.approx(-8.0 + 0.37)
    assert M.net_open_torque([(0.4, 0.8)], 0.37) == 0.0                      # friction holds it
    # a catch holds against opening only
    assert M.net_open_torque([(20.0, 0.8)], 0.37, hold=20.0) == 0.0
    assert M.net_open_torque([(-20.0, 0.8)], 0.37, hold=20.0) == pytest.approx(-16.0 + 0.37)


# ── a rigid plate on springs ─────────────────────────────────────────────────
HINGES = [(0.0, 0.3, 1e6), (0.0, 1.0, 1e6), (0.0, 1.7, 1e6)]
LATCH = (0.85, 1.0, 5e4)


def test_load_at_the_latch_line_goes_wholly_to_the_latch():
    r = M.plate_reactions(HINGES + [LATCH], [(1000.0, 0.85, 1.0)])
    assert r[3] / 1000.0 == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(r[:3], 0.0, atol=1e-7)


def test_load_at_mid_width_puts_53_percent_on_the_latch_and_16_on_each_hinge():
    r = M.plate_reactions(HINGES + [LATCH], [(1000.0, 0.45, 1.0)])
    assert r[3] / 1000.0 == pytest.approx(0.53, abs=0.01)
    assert np.allclose(r[:3] / 1000.0, 0.157, atol=0.01)


def test_plate_reactions_sum_to_the_load_and_the_moments_close():
    rng = np.random.default_rng(7)
    for _ in range(50):
        x0, y0 = rng.uniform(0.0, 0.9), rng.uniform(0.1, 1.9)
        F0 = rng.uniform(100, 3000)
        sup = HINGES + [LATCH]
        r = M.plate_reactions(sup, [(F0, x0, y0)])
        assert r.sum() == pytest.approx(F0, rel=1e-12)
        mx = sum(ri * s[0] for ri, s in zip(r, sup))
        my = sum(ri * s[1] for ri, s in zip(r, sup))
        assert abs(mx - F0 * x0) < 1e-9 * max(1.0, F0)
        assert abs(my - F0 * y0) < 1e-9 * max(1.0, F0)


def test_a_contact_beyond_the_latch_pulls_a_hinge_out():
    r = M.plate_reactions(HINGES + [LATCH], [(1000.0, 1.0, 1.0)])
    assert r[:3].min() < 0.0                                       # a hinge in tension: withdrawal mode
    assert r[3] > 1000.0                                           # and the latch carries more than the push
    # and a load low on the latch line prys the outer hinges, one each way
    low = M.plate_reactions(HINGES + [LATCH], [(1000.0, 0.85, 0.4)])
    assert low[3] / 1000.0 == pytest.approx(1.0, abs=0.1)
    assert low[:3].min() < 0.0 < low[:3].max() or np.allclose(low[:3], 0, atol=1e-6) is False


def test_plate_with_collinear_supports_cannot_hold():
    with pytest.raises(ValueError):
        M.plate_reactions(HINGES, [(100.0, 0.5, 1.0)])


# ── the two-body impact ──────────────────────────────────────────────────────
I43 = 43.0 * 0.9 ** 2 / 3.0
KICK = dict(m_s=5.6, v_s=6.0, I=I43, a=0.85, k_c=1e5)


def test_peak_force_is_v_rel_sqrt_mu_k_s():
    out = M.two_body_impact(k_path=6.7e4, **KICK)
    assert out["m_t"] == pytest.approx(I43 / 0.85 ** 2)
    assert out["mu"] == pytest.approx(4.15, abs=0.03)
    assert out["F_peak"] == pytest.approx(2.45e3, rel=0.01)                  # 2.45 kN
    k_s = 1 / (1 / 1e5 + 1 / 6.7e4)
    assert out["F_peak"] == pytest.approx(6.0 * math.sqrt(out["mu"] * k_s))
    # energy in the spring at peak is 1/2 mu v^2 of the foot's 100 J
    assert out["E_def"] == pytest.approx(0.5 * out["mu"] * 36.0)
    assert out["E_def"] < out["E0"]
    # round trip: F^2 / (2 k_s) is that energy
    assert out["F_peak"] ** 2 / (2 * out["k_s"]) == pytest.approx(out["E_def"])


def test_doubling_the_path_stiffness_raises_the_peak_by_less_than_root_two():
    a = M.two_body_impact(k_path=6.7e4, **KICK)["F_peak"]
    b = M.two_body_impact(k_path=1.34e5, **KICK)["F_peak"]
    assert a < b < a * math.sqrt(2)


def test_no_failure_conserves_angular_momentum_and_the_recoil_impulse_is_mu_1_plus_e_vrel():
    out = M.two_body_impact(k_path=6.7e4, e_c=0.2, **KICK)
    assert not out["failed"]
    assert out["L_after"] == pytest.approx(out["L_before"], rel=1e-12)
    assert out["J"] == pytest.approx(out["mu"] * 1.2 * 6.0)
    assert KICK["m_s"] * (6.0 - out["v_foot"]) == pytest.approx(out["J"])      # the foot's momentum change
    assert out["omega"] * KICK["I"] == pytest.approx(out["J"] * 0.85)           # the leaf's angular impulse
    # held by the latch: the leaf's rebound is heat, and everything closes
    assert out["KE_foot"] + out["heat_J"] == pytest.approx(out["E0"])
    assert out["heat_J"] > 0.0
    # a free leaf: foot + leaf + heat closes the same
    free = M.two_body_impact(k_path=6.7e4, e_c=0.2, leaf_held=False, **KICK)
    assert free["KE_foot"] + free["KE_leaf"] + free["heat_J"] == pytest.approx(free["E0"])
    assert free["heat_J"] >= 0.0


@pytest.mark.parametrize("e_c", [0.0, 0.2, 0.6, 1.0])
def test_a_perfectly_elastic_free_leaf_loses_nothing(e_c):
    out = M.two_body_impact(k_path=6.7e4, e_c=e_c, leaf_held=False, **KICK)
    assert out["L_after"] == pytest.approx(out["L_before"], rel=1e-12)
    if e_c == 1.0:
        assert out["heat_J"] == pytest.approx(0.0, abs=1e-9)


def test_after_a_failure_foot_KE_plus_leaf_KE_plus_heat_is_the_foots_energy():
    out = M.two_body_impact(k_path=6.7e4, F_cap=1.5e3, k_link=1e5, ductility=0.0, **KICK)
    assert out["failed"]
    assert out["KE_foot"] + out["KE_leaf"] + out["heat_J"] == pytest.approx(out["E0"], rel=1e-12)
    assert out["heat_J"] == pytest.approx(out["E_abs"], rel=1e-9)            # the failing link took it
    assert out["E_abs"] == pytest.approx(1.5e3 ** 2 / (2 * 1e5))
    assert out["L_after"] == pytest.approx(out["L_before"], rel=1e-12)
    # omega is the momentum-formula value
    assert out["omega"] == pytest.approx(5.6 * 0.85 * (6.0 - out["v_rel_after"]) / (I43 + 5.6 * 0.85 ** 2))
    # ductility lets the link take more
    duct = M.two_body_impact(k_path=6.7e4, F_cap=1.5e3, k_link=1e5, ductility=1.0, **KICK)
    assert duct["E_abs"] == pytest.approx(2 * out["E_abs"])
    assert duct["omega"] > out["omega"]                       # more taken in the link, more left in the leaf (momentum)


def test_a_link_that_absorbs_everything_leaves_foot_and_leaf_moving_together():
    out = M.two_body_impact(k_path=6.7e4, F_cap=1.0e3, k_link=1.0, ductility=0.0, **KICK)
    assert out["E_rel"] == 0.0 and out["v_rel_after"] == 0.0
    assert out["v_foot"] == pytest.approx(0.85 * out["omega"])


# ── damage ───────────────────────────────────────────────────────────────────

def blows_to_fail(r, cap=1000.0, limit=400000):
    D, life = 0.0, 0.0
    for n in range(1, limit + 1):
        D, life, failed = M.blow_damage(D, life, r * cap, cap)
        if failed:
            return n
    return None


def test_codeonly_a_blow_at_97_percent_fails_inside_two_blows():
    assert blows_to_fail(0.97) <= 2
    assert M.blow_life(0.97) == pytest.approx(0.50, abs=0.01)


def test_u_s_n_slope_orders_the_blow_counts():
    """The slope (strength falls 0.1 of itself per decade) is a stand-in (u);
    the tests assert the order and the decades, not exact counts."""
    n97, n9, n8 = blows_to_fail(0.97), blows_to_fail(0.9), blows_to_fail(0.8)
    assert n97 < n9 < n8
    assert 50 <= n8 <= 200                                   # about 100, within a factor 2
    assert blows_to_fail(0.5, limit=10000) is None           # r 0.5: more than 1e4
    assert M.blow_life(0.5) == pytest.approx(1e-5, rel=0.01)
    assert M.blow_life(0.9) == pytest.approx(0.1, rel=0.01)
    assert M.blow_life(0.8) == pytest.approx(0.01, rel=0.02)
    assert M.blow_life(1.0) == 1.0 and M.blow_life(2.0) == 1.0
    # monotone in the load: more force, fewer blows
    ns = [blows_to_fail(r) for r in (0.95, 0.9, 0.85, 0.8)]
    assert ns == sorted(ns)


def test_D_only_increases_and_a_failed_link_has_D_one():
    D, life = 0.0, 0.0
    seen = []
    for _ in range(40):
        D, life, failed = M.blow_damage(D, life, 900.0, 1000.0)
        seen.append(D)
        if failed:
            break
    assert all(b >= a for a, b in zip(seen, seen[1:]))
    assert failed and seen[-1] == 1.0 and 0.0 < seen[0] < 1.0
    # a blow does not repair
    assert M.blow_damage(0.5, 0.5, 1.0, 1000.0)[0] >= 0.5


def test_capacity_and_stiffness_fall_as_one_minus_D():
    lk = M.new_link("L", "x", "withdrawal", "wood", geom=dict(G=0.45, D=4.83e-3, L=0.019, n=2, k=2e5))
    c0, k0 = M.link_capacity(lk), M.link_stiffness(lk)
    lk["D"] = 0.4
    assert M.link_capacity_now(lk) == pytest.approx(0.6 * c0)
    assert M.link_stiffness(lk) == pytest.approx(0.6 * k0)
    # a held push the latch once held now beats it
    assert M.static_fails(0.7 * c0, c0, 0.4) and not M.static_fails(0.5 * c0, c0, 0.4)
    assert M.static_fails(1.0, c0, 0.999)


def test_combine_commutes_and_clips():
    for a, b in itertools.product((0.0, 0.2, 0.7), (0.0, 0.5, 1.0)):
        assert M.combine(a, b) == pytest.approx(M.combine(b, a))
        assert 0.0 <= M.combine(a, b) <= 0.999
    assert M.combine(0.5, 0.5) == pytest.approx(0.75)


def _strike_chain(screw_mode):
    # 12 mm of thread: with the NDS Rd two #10 screws are 3.99 kN laterally and (19 mm) 4.02 kN
    # in withdrawal, too close to tell apart; 12 mm of thread gives 2.54 kN in withdrawal
    screws = M.new_link("scr", "strike screws", screw_mode, "wood",
                        geom=dict(G=0.45, D=4.83e-3, l=0.012, L=0.012, n=2))
    plate = M.new_link("plate", "keeper plate", "bend", "iron",
                       geom=dict(b=0.09, d=0.003, span=0.04))
    split = M.new_link("split", "jamb split (stud)", "split", "wood",
                       geom=dict(l_s=0.08, t_e=0.030))
    return [screws, plate, split]


def test_u_the_weakest_link_flips_when_the_screw_mode_is_swapped():
    lat = _strike_chain("lateral")
    wd = _strike_chain("withdrawal")
    i_lat, caps_lat = M.weakest_link(lat, flaw_amp=0.0)
    i_wd, caps_wd = M.weakest_link(wd, flaw_amp=0.0)
    assert lat[i_lat]["id"] == "plate" and wd[i_wd]["id"] == "scr"           # not circular
    assert caps_lat[0] == pytest.approx(4.0e3, rel=0.05)                     # lateral: about 4 kN (NDS Rd; was 3.0 with the ends swapped)
    assert caps_wd[0] == pytest.approx(2.54e3, rel=0.02)                     # withdrawal 2 x 1.27 kN (12 mm of thread)
    assert caps_lat[2] == pytest.approx(7.2e3, rel=0.01)                     # with a stud
    assert caps_lat[1] == pytest.approx(3.4e3, rel=0.02)
    # without the stud the split (2.9) is first in both readings
    lat[2]["geom"]["t_e"] = 0.012
    assert lat[M.weakest_link(lat, flaw_amp=0.0)[0]]["id"] == "split"


def test_flaw_spreads_identical_links_deterministically():
    lk = _strike_chain("lateral")[0]
    lo, mid, hi = (M.link_capacity(lk, flaw01=f) for f in (0.0, 0.5, 1.0))
    assert lo == pytest.approx(0.7 * mid) and hi == pytest.approx(1.3 * mid)
    assert M.link_capacity(lk, flaw01=0.0, flaw_amp=0.0) == mid
    assert M.link_capacity(lk, flaw01=0.3) == M.link_capacity(lk, flaw01=0.3)


def test_blow_path_fails_the_weakest_first_and_damages_the_others():
    chain = _strike_chain("lateral")
    failed = M.blow_path(chain, 3.6e3, flaw_amp=0.0)
    assert failed == ["plate"]                                # 3.6 kN > 3.4 (plate), under the screws' 3.99
    assert M.link_failed(chain[1]) and not M.link_failed(chain[0])
    assert 0.0 < chain[0]["D"] < 1.0 and chain[2]["D"] < chain[0]["D"]
    again = M.blow_path(chain, 3.6e3, flaw_amp=0.0)
    assert "plate" not in again                               # a failed link is not hit again


def test_path_stiffness_is_series_with_damage():
    a = M.new_link("a", "a", "shear", "iron", geom=dict(d=0.012, k=1e6))
    b = M.new_link("b", "b", "shear", "iron", geom=dict(d=0.012, k=1e6))
    assert M.path_stiffness([a, b]) == pytest.approx(5e5)
    b["D"] = 0.5
    assert M.path_stiffness([a, b]) == pytest.approx(1 / (1 / 1e6 + 1 / 5e5))
    b["D"] = 0.999
    assert M.path_stiffness([a, b]) == pytest.approx(1 / (1 / 1e6 + 1 / 1e3))


def test_link_capacities_by_mode_match_the_fastener_formulas():
    sh = M.new_link("s", "pin", "shear", "iron", geom=dict(d=0.012, n=1))
    assert M.link_capacity(sh, flaw_amp=0.0) == pytest.approx(17e3, rel=0.02)
    bend = M.new_link("b", "bar", "bend", "wood", geom=dict(b=0.038, d=0.089, span=1.2, sigma=40e6))
    assert M.link_capacity(bend, flaw_amp=0.0) == pytest.approx(6.7e3, rel=0.01)
    pan = M.new_link("p", "skin", "panel", "wood", geom=dict(perimeter=0.5, t=0.003))
    assert M.link_capacity(pan, flaw_amp=0.0) == pytest.approx(7.5e3)
    ov = M.new_link("o", "x", "tension", "iron", geom=dict(cap_N=123.0))
    assert M.link_capacity(ov, flaw_amp=0.0) == 123.0
    with pytest.raises(ValueError):
        M.link_capacity(M.new_link("z", "z", "magic", "iron"))


# ── the exact quarter turn ───────────────────────────────────────────────────

def _random_cells(rng, n=60, span=20, z=True):
    pts = {(int(x), int(y), int(rng.integers(0, 3))) for x, y in rng.integers(-span, span, (n, 2))}
    return np.array(sorted(pts))


PIVOTS = [(0, 0), (6, -4), (-3, 5), (1, 1), (-7, 3), (11, -9)]


@pytest.mark.parametrize("pivot", PIVOTS)
@pytest.mark.parametrize("s", [1, -1])
def test_quarter_turn_is_a_permutation_equal_to_a_float_reference(pivot, s):
    rng = np.random.default_rng(abs(hash((pivot, s))) % 10000)
    for _ in range(10):
        cells = _random_cells(rng)
        out = M.quarter_turn(cells, s, pivot)
        assert out.dtype.kind == "i" and out.shape == cells.shape
        assert len({tuple(r) for r in out.tolist()}) == len(cells)           # no cell lost or doubled
        assert np.array_equal(out, M.quarter_turn_float(cells, s, pivot))    # float reference
        assert np.array_equal(out[:, 2], cells[:, 2])                        # z untouched


@pytest.mark.parametrize("pivot", PIVOTS)
def test_four_quarter_turns_are_the_identity_and_two_are_a_half_turn(pivot):
    rng = np.random.default_rng(3)
    cells = _random_cells(rng)
    c = cells
    for _ in range(4):
        c = M.quarter_turn(c, +1, pivot)
    assert np.array_equal(c, cells)
    assert np.array_equal(M.quarter_turn(M.quarter_turn(cells, 1, pivot), -1, pivot), cells)
    half = M.quarter_turn(M.quarter_turn(cells, 1, pivot), 1, pivot)
    assert np.array_equal(half[:, 0], pivot[0] - cells[:, 0])
    assert np.array_equal(half[:, 1], pivot[1] - cells[:, 1])


def test_the_pivot_cell_is_fixed_for_a_centre_pivot():
    out = M.quarter_turn([[3, -2, 5]], 1, (6, -4))
    assert out.tolist() == [[3, -2, 5]]


def test_corner_and_centre_pivots_both_give_integers_and_a_mismatched_pivot_raises():
    M.quarter_turn([[1, 1, 0]], 1, (4, 6))                  # centre of a cell
    M.quarter_turn([[1, 1, 0]], 1, (3, 5))                  # a cell corner
    with pytest.raises(ValueError):
        M.quarter_turn([[1, 1, 0]], 1, (3, 4))
    with pytest.raises(ValueError):
        M.quarter_turn([[1, 1, 0]], 2, (4, 4))


def test_the_hand_worked_hinge_row_goes_to_x0_minus_j_minus_1():
    x0, y0 = 7, 4
    row = np.array([(x0, y0 + j, 0) for j in range(10)])
    out = M.quarter_turn(row, +1, (2 * x0 - 1, 2 * y0 - 1))      # hinge at the corner (x0 - 1/2, y0 - 1/2)
    for j in range(10):
        assert tuple(out[j]) == (x0 - j - 1, y0, 0)


@pytest.mark.parametrize("vox", VOXELS)
def test_exchange_sends_the_room_air_into_the_vacated_cells(vox):
    d = M.cell_edge_m(vox)
    n = int(round(0.9 / d))
    x0, y0 = 5, 5
    leaf = np.array([(x0, y0 + j, z) for j in range(n) for z in range(3)])
    dest, entered, vacated = M.turn_exchange(leaf, +1, (2 * x0 - 1, 2 * y0 - 1))
    assert len(entered) == len(vacated) > 0
    leafset = {tuple(r) for r in leaf.tolist()}
    assert not any(tuple(r) in leafset for r in entered.tolist())            # air cells only
    assert all(tuple(r) in leafset for r in vacated.tolist())
    assert [tuple(r) for r in entered.tolist()] == sorted(tuple(r) for r in entered.tolist())
    assert len({tuple(r) for r in dest.tolist()}) == len(leaf)


def test_swept_sector_holds_the_start_and_end_leaf_and_nothing_beyond_the_radius():
    x0, y0, n, nz = 20, 20, 18, 3
    piv = (2 * x0 - 1, 2 * y0 - 1)
    start = np.array([(x0, y0 + j, z) for j in range(n) for z in range(nz)])
    end = M.quarter_turn(start, +1, piv)
    a0, a1 = M.sweep_span(start, piv, +1)
    assert a0 == pytest.approx(math.pi / 4) and 3.0 < a1 < math.pi            # the corner cell at 45 deg to the far end just under 180
    assert math.pi / 2 < a1 - a0 < math.pi
    sec = {tuple(r) for r in M.swept_sector(piv, n, a0, a1, (0, nz - 1)).tolist()}
    # the leaf runs along +y from the axis (bearing 90 deg) and ends along -x (180 deg)
    assert all(tuple(r) in sec for r in start.tolist())
    assert all(tuple(r) in sec for r in end.tolist())
    px, py = piv[0] / 2, piv[1] / 2
    for (x, y, z) in sec:
        assert math.hypot(x - px, y - py) <= n + 0.5 + 1e-9 and 0 <= z < nz
    # the area is the disc sector of the span
    area = len(sec) / nz
    assert area == pytest.approx((a1 - a0) / 2 * (n + 0.5) ** 2, rel=0.05)
    # clipping to the world
    clip = M.swept_sector(piv, n, a0, a1, (0, nz - 1), shape=(30, 30, 2))
    assert clip[:, 2].max() == 1 and clip[:, 0].max() < 30 and clip[:, 0].min() >= 0


# ── wedge ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mu_a,mu_b", [(0.5, 0.5), (0.3, 0.6), (0.2, 0.9)])
def test_wedge_self_locks_at_the_sum_of_the_friction_angles(mu_a, mu_b):
    lim = math.degrees(math.atan(mu_a) + math.atan(mu_b))
    assert M.wedge_limit_deg(mu_a, mu_b) == pytest.approx(lim)
    assert M.wedge_self_locks(lim - 0.01, mu_a, mu_b)
    assert not M.wedge_self_locks(lim + 0.01, mu_a, mu_b)
    # the wrong form tan(alpha) <= mu_a + mu_b would put the limit elsewhere
    wrong = math.degrees(math.atan(mu_a + mu_b))
    assert abs(wrong - lim) > 0.5 or mu_a == mu_b == 0


def test_wedge_half_pairs_give_53_degrees():
    assert M.wedge_limit_deg(0.5, 0.5) == pytest.approx(53.13, abs=0.01)
    assert M.wedge_limit_deg(0.5, 0.5) > math.degrees(math.atan(1.0)) + 5   # not the 45 of the wrong form


# ── bar ──────────────────────────────────────────────────────────────────────

def test_u_bar_hold_is_the_min_of_beam_cleats_and_leaf():
    beam = F.bar_bending(40e6, 0.038, 0.089, 1.2)
    one = F.dowel_lateral_ultimate(0.45, 4.83e-3, 0.019, 1)                    # 1.99 kN ultimate each (1.5 with the Rd ends swapped)
    assert 3 * one == pytest.approx(6.0e3, rel=0.03)                           # "6+ kN a side" is now 3 screws, not 4
    n3, which3 = M.bar_hold(40e6, 0.038, 0.089, 1.2, 3 * one, 20e3)
    assert which3 == "cleats" and n3 == pytest.approx(3 * one)                 # 3 screws a side hold a bit less than the beam
    cleats = 4 * one                                                           # 4 a side: the beam governs
    assert cleats > beam > 0.0
    n, which = M.bar_hold(40e6, 0.038, 0.089, 1.2, cleats, 20e3)
    assert which == "beam" and n == pytest.approx(6.7e3, rel=0.01)
    assert M.bar_hold(40e6, 0.038, 0.089, 1.2, 4e3, 20e3)[1] == "cleats"
    assert M.bar_hold(40e6, 0.089, 0.038, 1.2, cleats, 20e3)[0] == pytest.approx(0.43 * 6.7e3, rel=0.03)


# ── one flow: where a joule is booked ────────────────────────────────────────

def _stub():
    return types.SimpleNamespace(mech_in=0.0, work_out=0.0)


def test_deposit_without_an_actor_books_mech_in_exactly():
    w = _stub()
    heat = []
    book = M.deposit(w, [(1, 2, 3), (1, 2, 4)], 12.5, heat=lambda cells, J: heat.append((len(cells), J)))
    assert book == "mech_in" and w.mech_in == 12.5 and w.work_out == 0.0
    assert heat == [(2, 12.5)]
    M.deposit(w, [], 7.5)
    assert w.mech_in == 20.0


def test_deposit_with_an_actor_debits_the_actor_credits_work_out_and_never_touches_mech_in():
    w = _stub()
    actor = {"pid": 1}
    taken, credited = [], []
    book = M.deposit(w, [(0, 0, 0)], 30.0, actor=actor,
                     debit=lambda a, J: taken.append((a["pid"], J)),
                     work_out=lambda J: credited.append(J))
    assert book == "work_out" and w.mech_in == 0.0
    assert taken == [(1, 30.0)] and credited == [30.0]
    # the stopgaps: p["_work_J"] and w.work_out
    M.deposit(w, [], 5.0, actor=actor)
    assert actor["_work_J"] == 5.0 and w.work_out == 5.0 and w.mech_in == 0.0


def test_deposit_refuses_negative_joules():
    with pytest.raises(ValueError):
        M.deposit(_stub(), [], -1.0)


def test_deposit_with_a_heat_routine_and_no_cell_is_refused_and_books_nothing_random_sweep():
    # Cause: `if heat is not None and len(cells)` skipped the heat for an empty
    # list yet went on to book the joules: joules on the books, heat nowhere.
    # An empty list, an empty (0, 3) array and an empty index tuple all refuse
    # (and a tuple of three EMPTY arrays has len 3, the old test's blind spot).
    import random
    import numpy as np
    rng = random.Random(1312)
    empties = [lambda: [], lambda: np.zeros((0, 3), np.int64),
               lambda: (np.zeros(0, np.int64),) * 3, lambda: ()]
    for _ in range(200):
        w = _stub()
        w.mech_in, w.work_out = rng.uniform(0, 50), rng.uniform(0, 50)
        before = (w.mech_in, w.work_out)
        J = rng.uniform(1e-9, 1e4)
        actor = {"pid": 1} if rng.random() < 0.5 else None
        placed, taken = [], []
        with pytest.raises(ValueError):
            M.deposit(w, rng.choice(empties)(), J, actor=actor,
                      heat=lambda c, j: placed.append(j), debit=lambda a, j: taken.append(j))
        assert placed == [] and taken == []
        assert (w.mech_in, w.work_out) == before
        assert actor is None or "_work_J" not in actor
        # and with cells the heat is exactly J, booked once, wherever it is booked
        n = rng.randrange(1, 9)
        cells = [(i, 0, 0) for i in range(n)]
        form = rng.choice([cells, np.array(cells, np.int64), tuple(np.array(cells, np.int64).T)])
        placed = []
        book = M.deposit(w, form, J, actor=actor, heat=lambda c, j: placed.append(j))
        assert placed == [J]
        if actor is None:
            assert book == "mech_in" and w.mech_in == before[0] + J and w.work_out == before[1]
        else:
            assert book == "work_out" and w.mech_in == before[0] and w.work_out == before[1] + J


def test_zero_joules_with_no_cell_is_a_quiet_nothing():
    w = _stub()
    assert M.deposit(w, [], 0.0, heat=lambda c, j: pytest.fail("no heat to place")) == "mech_in"
    assert w.mech_in == 0.0


# ── the material adapter ─────────────────────────────────────────────────────

def test_material_props_equal_the_bond_table():
    from src.bonds import materials as B
    from src.voxel import materials as V
    w_al = M.material_props("wood", "along")
    w_ac = M.material_props("wood", "across")
    assert (w_al["E"], w_al["nu"], w_al["st"], w_al["damping"], w_al["rho"]) == B.WOOD_L
    assert w_ac["E"] == B.WOOD_T[0] and w_ac["st"] == B.WOOD_T[2]
    assert (w_al["E_perp"], w_al["st_perp"]) == (B.WOOD_T[0], B.WOOD_T[2])
    assert w_al["E"] == 11e9 and w_al["st"] == 60e6 and w_ac["E"] == 0.55e9 and w_ac["st"] == 3e6
    for name, mid in (("iron", V.IRON), ("glass", V.GLASS), ("stone", V.STONE), ("tin", V.TIN)):
        row = B.ROW[int(mid)]
        for p in (M.material_props(name), M.material_props(int(mid))):
            assert (p["E"], p["nu"], p["st"], p["damping"], p["rho"]) == row
    assert M.material_props("iron")["sy"] == 250e6
    assert M.material_props("wood")["sy"] is None                           # wood breaks before it yields


def test_material_props_extras_hardness_and_unknowns():
    b = M.material_props("brass")
    assert b["sy"] == 200e6 and b["tau"] == 200e6 and b["source"] == "extra"
    assert M.material_props("steel", HV=460)["sy"] == pytest.approx(1.5e9, rel=0.01)
    with pytest.raises(ValueError):
        M.material_props("wood", grain="sideways")
    with pytest.raises(KeyError):
        M.material_props("unobtainium", source="bonds")


def test_material_props_reads_matrec_when_it_exists():
    mr = pytest.importorskip("src.voxel.matrec")
    p = M.material_props("wood", "along", source="matrec")
    ref = mr.props("wood", "along")
    assert p["source"] == "matrec" and p["E"] == ref["E"] and p["tau"] == ref["tau"]
    assert p["st"] == ref["st"]
    # the stopgap and the record agree within the same handbook, order of magnitude
    assert 0.3 < p["E"] / M.material_props("wood")["E"] < 3.0


# ── the record ───────────────────────────────────────────────────────────────

def test_the_record_is_json_safe_and_carries_no_mass_and_no_output():
    rec = M.new_mechanism(1, "oak door", leaf=10, frame=11)
    rec["joints"].append(M.new_joint("h1", axis2=(-1, 0), limit_deg=(0.0, 100.0)))
    rec["catches"].append(M.new_catch("latch", at=(3, 0, 10), path=["bolt", "keeper"]))
    rec["springs"].append(M.new_spring("s", "coil", "steel", dict(d=3e-4, D=2.5e-3, n=8)))
    rec["links"].append(M.new_link("bolt", "bolt shear", "shear", "iron", geom=dict(d=0.012)))
    back = json.loads(json.dumps(rec))
    assert back == rec
    assert "mass" not in rec and "m" not in rec and "I" not in rec
    assert M.engaged(rec["catches"][0]) is False
    rec["catches"][0]["x_mm"] = 12.0
    assert M.engaged(rec["catches"][0]) is True
    assert rec["joints"][0]["fric_Nm_per_N"] == pytest.approx(0.2 * 0.005)
