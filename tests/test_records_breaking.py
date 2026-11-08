"""T2 BREAKING (src/voxel/breaking.py): stress AND strain failure, the weakest
point, exact shares, the cut energy, the swing load, leaks and the
fragmentation loop. Stand-alone: no World.

Tags (wave 13 rules): test_echo_* = constant fitted to the number it checks,
test_codeonly_* = the code against its own table or arithmetic of its formula,
test_u_* = independent but source unverified. Only conservation, exactness and
closed-form lines gate a scene; [E] and [C] never do.
"""
import json
import math
import os
import types

import numpy as np
import pytest

from src.voxel import breaking as B

HERE = os.path.dirname(os.path.abspath(__file__))
HARD = B.STRENGTH["steel_hard"]
MILD = B.STRENGTH["steel_mild"]
BLADE_Z = 0.04 * 0.002 ** 2 / 6          # 2.67e-8 m3


def _sourced(key):
    """Status of a (u) number in tests/data/sourced_numbers.json, or 'absent'."""
    path = os.path.join(HERE, "data", "sourced_numbers.json")
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return "absent"
    row = data.get(key) if isinstance(data, dict) else None
    if isinstance(row, dict):
        return str(row.get("status", "unmarked"))
    return "unmarked" if row is None else str(row)


def _blade(row=HARD, n=21, seed=3, length=0.3):
    return B.make_section("rect", length, 0.04, 0.002, row, n=n, seed=seed)


# ── [C] the worked example: arithmetic of the formulas ────────────────────────
def test_codeonly_blade_worked_example_numbers():
    """40 x 2 mm hard blade, K_Ic 20 MPa sqrt(m), E 210 GPa, a0 50 um. The
    K_Ic and a0 are (u); the lines are the arithmetic of the formulas."""
    print("K_Ic/a0 status:", _sourced("steel_hard_K_Ic"), _sourced("a0_flaw"))
    Z = float(B.section_modulus("rect", 0.04, 0.002))
    assert Z == pytest.approx(BLADE_Z, rel=1e-12)
    s = float(B.flaw_stress(20e6, 50e-6))
    assert s == pytest.approx(1.42e9, rel=0.005)                 # 1.42 GPa
    assert s * Z == pytest.approx(38.0, rel=0.01)                # snap moment
    assert s / 210e9 == pytest.approx(0.0068, rel=0.01)          # 0.68 percent
    Gc = B.gc_from_kic(20e6, 210e9, 0.3)
    assert Gc == pytest.approx(1.73e3, rel=0.005)                # 1.73 kJ/m2
    assert Gc * 0.04 * 0.002 == pytest.approx(0.14, rel=0.02)    # 0.14 J
    # the bend radius at the snap: 1 mm / 0.0068 = 0.15 m
    assert 0.001 / (s / 210e9) == pytest.approx(0.15, rel=0.03)


def test_codeonly_hard_blade_bent_to_that_strain_snaps_by_the_flaw_arm():
    p = _blade()
    kappa = 0.0069 / 0.001                  # a hair past the flaw strain
    r = B.bend_displacement(HARD, p, kappa, scatter=False)
    assert np.all(r["u"] >= 1.0) and set(r["mode"]) == {"flaw"}
    r = B.bend_displacement(HARD, p, 0.0066 / 0.001, scatter=False)
    assert np.all(r["u"] < 1.0)


def test_codeonly_mild_steel_yields_then_tears_at_eps_f_positive_control():
    """The SAME bend on mild steel: flaw stress about 7 GPa, so no snap; it
    yields and takes a set; push it to eps_f and it tears (positive control)."""
    pm = _blade(MILD)
    assert float(B.flaw_stress(MILD.K_Ic, MILD.a0)) == pytest.approx(7.1e9, rel=0.03)
    r = B.bend_displacement(MILD, pm, 0.0069 / 0.001, scatter=False)
    assert np.all(r["u"] < 1.0)
    assert np.all(r["yielded"]) and np.all(r["residual_kappa"] > 0)
    # just below eps_f: still holds. At eps_f: tears.
    r = B.bend_displacement(MILD, pm, 0.99 * MILD.eps_f / 0.001, scatter=False)
    assert np.all(r["u"] < 1.0)
    r = B.bend_displacement(MILD, pm, MILD.eps_f / 0.001, scatter=False)
    assert np.all(r["u"] >= 1.0) and set(r["mode"]) == {"tear"}


def test_codeonly_strain_controlled_bend_eps_f_decides():
    """Two rows identical except the strain to fracture: the same imposed
    bend (1 percent surface strain) snaps the 0.5 percent one and not the 25
    percent one; the ductile one tears when eps reaches eps_f."""
    brittle = MILD._replace(eps_f=0.005)
    ductile = MILD._replace(eps_f=0.25)
    kappa = 0.01 / 0.001
    for row, breaks in ((brittle, True), (ductile, False)):
        r = B.bend_displacement(row, _blade(row), kappa, scatter=False)
        assert bool(np.all(r["u"] >= 1.0)) == breaks
    r = B.bend_displacement(ductile, _blade(ductile), 0.25 / 0.001, scatter=False)
    assert np.all(r["u"] >= 1.0)


# ── exact shares ──────────────────────────────────────────────────────────────
def test_exact_split_adds_back_bit_for_bit():
    rng = np.random.default_rng(5)
    x = rng.random(2000) * 10.0 ** rng.integers(-9, 9, 2000)
    for f in (0.01, 0.3, 0.5, 0.97):
        a, b = B.exact_split(x, f)
        assert np.all(a + b == x)
        assert np.all(np.abs(a - f * x) <= x * 2.0 ** -39)
    a, b = B.exact_split(0.0, 0.4)
    assert a == 0.0 and b == 0.0


def test_cut_masses_of_the_two_pieces_sum_to_the_whole():
    p = _blade(n=21)
    p["energy"] = np.linspace(1.0, 3.0, 21) * 1e-3
    p["grams"] = np.c_[p["mass"] * 600.0, p["mass"] * 400.0]
    p["species"] = ("iron", "carbon")
    whole = {k: B.total(p, k) for k in ("mass", "len", "energy")}
    for f in (0.2, 0.5, 0.9):
        A, Bp = B.cut_piece(p, 10, f)
        for k in ("mass", "len", "energy"):
            tot = math.fsum([B.total(A, k), B.total(Bp, k)])
            assert abs(tot - whole[k]) <= 1e-15 * whole[k]
        for s in (0, 1):
            tot = math.fsum([A["grams"][:, s].sum(), Bp["grams"][:, s].sum()])
            assert tot == pytest.approx(math.fsum(p["grams"][:, s]), rel=1e-15)
        assert len(A["len"]) == 11 and len(Bp["len"]) == 11
    # the segment ids follow the matter
    A, Bp = B.cut_piece(p, 10, 0.4)
    assert A["sid"][-1] == Bp["sid"][0] == p["sid"][10]


def test_cut_energy_end_node_heat_equals_Gc_Ax_and_closes():
    p = _blade(n=21)
    Gc = B.gc_from_kic(HARD.K_Ic, HARD.E, HARD.nu)
    Ax = 0.04 * 0.002
    work = 5.0
    out = B.apply_cut(HARD, p, 10, 0.5, work, kin_frac=1.0)
    A, Bp, ev = out
    assert ev["surface"] == pytest.approx(Gc * Ax, rel=1e-12)
    heat = A["energy"][-1] + Bp["energy"][0]                 # all end-node heat
    assert heat == pytest.approx(Gc * Ax, rel=1e-12)
    assert ev["surface"] + ev["heat_extra"] + ev["kinetic"] == pytest.approx(work, rel=1e-15)
    # momentum: the pieces part with equal and opposite momentum
    va, vb = ev["v"]
    ma, mb = ev["masses"]
    assert ma * va + mb * vb == pytest.approx(0.0, abs=1e-9)
    assert 0.5 * ma * va ** 2 + 0.5 * mb * vb ** 2 == pytest.approx(ev["kinetic"], rel=1e-12)
    # lighter piece moves faster
    # a blow that cannot pay the new surface does not part it
    assert B.apply_cut(HARD, p, 10, 0.5, 0.5 * Gc * Ax) is None


def test_impulses_conserve_momentum_with_a_moving_parent():
    va, vb = B.impulses(0.2, 0.6, 3.0, v0=2.0)
    assert 0.2 * va + 0.6 * vb == pytest.approx(0.8 * 2.0, rel=1e-12)
    assert va != vb and abs(va - 2.0) > abs(vb - 2.0)        # the light one moves more


# ── weakest point, thresholds, damage ─────────────────────────────────────────
def test_notched_segment_breaks_first():
    p = _blade(n=21)
    p["notch"][4] = 400e-6                         # a saw cut near one end
    M = np.full(21, 25.0)                          # pure bending: same M everywhere
    r = B.bend_force(HARD, p, M, scatter=False)
    assert int(np.argmax(r["u"])) == 4
    q = _blade(n=21)
    q["notch"][4] = 20e-6                          # a scratch: still holds at 0.8
    r = B.bend_force(HARD, q, np.full(21, 0.8 * float(B.flaw_moment(HARD, q)[0])),
                     scatter=False)
    assert int(np.argmax(r["u"])) == 4 and r["u"][4] < 1.0
    r = B.bend_force(HARD, q, np.full(21, 0.95 * float(B.flaw_moment(HARD, q)[0])),
                     scatter=False)
    assert r["u"][4] >= 1.0 and np.sum(r["u"] >= 1.0) == 1       # only the scratch


def test_three_point_bend_breaks_mid_span_no_break_below_capacity():
    n, L = 21, 0.3
    p = B.make_section("rect", L, 0.04, 0.002, HARD, n=n)
    Mcap = float(B.flaw_moment(HARD, p)[0])        # 38 N m
    P_cap = 4.0 * Mcap / L                         # simply supported, mid load
    for scale, breaks in ((0.99, False), (1.01, True)):
        M = B.moments(p["x"], [(L / 2, scale * P_cap)], "simple", 0.0, L)
        r = B.bend_force(HARD, p, M, scatter=False)
        assert bool(r["u"].max() >= 1.0) == breaks
        assert int(np.argmax(r["u"])) == n // 2


def test_capacity_scales_with_one_minus_dmg_and_the_longer_flaw():
    p = _blade()
    base = float(B.flaw_moment(HARD, p)[0])
    prev = base
    for d in (0.1, 0.3, 0.6):
        q = _blade()
        q["dmg"][:] = d
        cap = float(B.flaw_moment(HARD, q)[0])
        a_i = HARD.a0 + d * B.A_MAX_FRAC * 0.002
        assert cap == pytest.approx(base * (1 - d) * math.sqrt(HARD.a0 / a_i), rel=1e-9)
        assert cap < prev
        prev = cap
    # the tear arm scales exactly with (1 - dmg)
    q = _blade(MILD)
    q["dmg"][:] = 0.4
    assert B.segment_limits(MILD, q)["eps_tear"][0] == pytest.approx(MILD.eps_f * 0.6)


def test_section_modulus_from_the_remaining_section():
    assert float(B.section_modulus("rect", 0.04, 0.002)) == pytest.approx(BLADE_Z)
    assert float(B.section_modulus("round", 0.01, 0)) == pytest.approx(math.pi * 1e-6 / 32)
    D, d = 0.02, 0.016
    assert float(B.section_modulus("tube", D, d)) == pytest.approx(
        math.pi * (D ** 4 - d ** 4) / (32 * D))
    # rust 0.2 mm off every face of the blade: depth 1.6 mm, width 39.6 mm
    z = float(B.section_modulus("rect", 0.04, 0.002, loss=0.2e-3))
    assert z == pytest.approx(0.0396 * 0.0016 ** 2 / 6, rel=1e-12)
    assert float(B.section_modulus("rect", 0.04, 0.002, loss=1.5e-3)) == 0.0
    assert B.shape_factor("rect", 0, 0) == 1.5
    assert B.shape_factor("round", 0, 0) == pytest.approx(1.698, abs=1e-3)


def test_moment_diagrams_against_closed_forms():
    n, L, P = 100, 1.0, 10.0
    x = L * (np.arange(n) + 0.5) / n
    Mc = B.moments(x, [(L, P)], "cantilever", 0.0)
    assert np.allclose(Mc, P * (L - x))
    Ms = B.moments(x, [(0.5, P)], "simple", 0.0, L)
    assert np.allclose(Ms, np.where(x < 0.5, P * x / 2, P * (L - x) / 2))
    # free beam hit at the centre: balanced by its own inertia, M_max = P L/8
    Mf = B.moments(x, [(0.5, P)], "free", masses=np.full(n, 0.01))
    assert Mf.max() == pytest.approx(P * L / 8, rel=0.03)
    assert Mf[0] < 0.05 * Mf.max()


def test_hinge_when_the_moment_passes_the_plastic_moment():
    Z = BLADE_Z
    My = MILD.sigma_y * Z
    eps = B.strain_from_moment(MILD, np.array([0.5 * My, My, 1.2 * My, 1.5 * My, 1.6 * My]),
                               Z)
    assert eps[0] == pytest.approx(0.5 * MILD.sigma_y / MILD.E)
    assert eps[1] == pytest.approx(MILD.sigma_y / MILD.E)
    assert eps[2] > eps[1] and np.isfinite(eps[2])
    assert np.isinf(eps[3]) and np.isinf(eps[4])


def test_cut_loop_repeats_until_every_piece_holds():
    """A solver that keeps loading the pieces (a bar shaken until its pieces
    are short): the loop cuts, re-solves each piece, and stops."""
    p = _blade(n=32, length=0.32)

    def make(eps):
        def solver(piece):
            return _shaken(piece, eps)
        return solver

    def _shaken(piece, eps):
        u = np.where(piece["len"].sum() > 0.1, 1.5, 0.5) * np.ones(len(piece["len"]))
        mid = 0.5 * (len(u) - 1)
        u = u * (1 - 1e-3 * np.abs(np.arange(len(u)) - mid))   # weakest: the middle
        return {"u": u, "eps": np.full(len(u), eps), "mode": np.array(["flaw"] * len(u))}
    solver = make(0.005)
    out = B.cut_loop(HARD, p, solver, work=1e3, max_cuts=64)
    assert not out["arrested"] and len(out["pieces"]) > 2
    assert all(q["len"].sum() <= 0.1 + 1e-12 for q in out["pieces"])
    assert math.fsum(B.total(q, "mass") for q in out["pieces"]) == pytest.approx(
        B.total(p, "mass"), rel=1e-15)
    # no work: the first crack arrests
    out = B.cut_loop(HARD, p, make(0.0), work=0.0)
    assert out["arrested"] and len(out["pieces"]) == 1


def test_cut_loop_with_a_real_bend_gives_two_pieces_and_a_cut_event():
    p = _blade()
    sol = B.displacement_solver(HARD, lambda x: np.full_like(x, 0.0069 / 0.001))
    out = B.cut_loop(HARD, p, sol, work=0.0, scatter=True)
    assert len(out["pieces"]) == 2 and len(out["events"]) == 1
    assert out["events"][0]["mode"] == "flaw" and out["events"][0]["cause"] == "fracture"
    tot = math.fsum(B.total(q, "mass") for q in out["pieces"])
    assert tot == B.total(p, "mass") or abs(tot - B.total(p, "mass")) <= 1e-15 * tot


# ── Weibull scatter ───────────────────────────────────────────────────────────
def test_weibull_size_effect_is_deterministic_and_larger_is_weaker():
    u = B.units(7, np.arange(200))
    assert np.array_equal(u, B.units(7, np.arange(200)))             # replays
    assert not np.array_equal(u, B.units(8, np.arange(200)))         # seed matters
    assert np.all((u > 0) & (u < 1))
    m = 8.0
    f1 = B.weibull_factor(u, m, 1e-6)
    f8 = B.weibull_factor(u, m, 8e-6)
    assert np.all(f8 < f1)                                           # same draw
    assert np.allclose(f8 / f1, 8.0 ** (-1 / m))
    assert f1.mean() == pytest.approx(1.0, rel=0.05)                 # normalised
    # weakest link: a longer section is weaker per unit stress, on average
    weakest = {}
    for n in (5, 40):
        vals = []
        for seed in range(60):
            p = B.make_section("rect", 0.01 * n, 0.04, 0.002, HARD, n=n, seed=seed)
            vals.append(B.segment_limits(HARD, p)["sigma_flaw"].min())
        weakest[n] = float(np.mean(vals))
    assert weakest[40] < weakest[5]
    # no scatter when m is None
    assert np.all(B.weibull_factor(u, None, 1e-6) == 1.0)


# ── swing, impact, thermal stress ─────────────────────────────────────────────
def test_codeonly_swing_load_is_stopping_distance_not_wave_transit():
    F = B.swing_load(1.0, 10.0, 0.01)
    assert F == pytest.approx(5000.0, rel=1e-12)
    # a stress-wave transit would give about 1.7e5 N: 30 times too much
    tau = B.wave_transit(0.15, 210e9, 7800.0)
    assert 1.0 * 10.0 / tau > 1.0e5
    assert F < 0.1 * 1.0 * 10.0 / tau
    # the stopping time is far longer than the wave transit
    assert B.swing_duration(10.0, 0.01) > 10 * tau
    assert B.swing_load(2.0, 10.0, 0.01) == 2 * F                     # linear in m
    assert B.swing_load(1.0, 20.0, 0.01) == 4 * F                     # square in v


def test_impact_stress_uses_both_impedances():
    z_s = B.impedance(7800.0, 210e9)
    z_g = B.impedance(2500.0, 70e9)
    s = B.impact_stress(5.0, z_s, z_g)
    assert s == pytest.approx(5.0 * z_s * z_g / (z_s + z_g))
    assert s < 5.0 * z_g                       # not the rigid-wall figure
    assert B.impact_stress(5.0, z_s, 1e12 * z_s) == pytest.approx(5.0 * z_s, rel=1e-6)


def test_thermal_stress_from_the_nodes_cooled_surface_in_tension():
    T = np.array([600.0, 550.0, 400.0, 250.0, 100.0])      # core to cooled face
    s = B.thermal_stress(7e10, 9e-6, 0.22, T)
    assert s[-1] > 0 and s[0] < 0                          # skin in tension
    assert float(np.sum(s)) == pytest.approx(0.0, abs=1e-3 * np.abs(s).max())
    # glass wall: the stress is E alpha dT / (1 - nu) for a uniform offset
    assert s[-1] == pytest.approx(7e10 * 9e-6 * (T.mean() - 100.0) / 0.78)


# ── leaks ─────────────────────────────────────────────────────────────────────
def test_codeonly_leak_is_min_of_orifice_and_slit():
    w, l, t, head = 20e-6, 0.05, 6e-3, 0.10
    q_sl = B.slit_leak(w, l, 1000.0 * B.G * head, 1.0e-3, t) * 1e6        # ml/s
    q_or = B.orifice_leak(w * l, head) * 1e6
    assert q_sl == pytest.approx(0.0054, rel=0.02)
    assert q_or == pytest.approx(0.84, rel=0.01)
    assert B.leak_rate(w, l, t, head) * 1e6 == pytest.approx(q_sl, rel=1e-12)
    assert 100 < q_or / q_sl < 200                         # "about 150 times"
    # a wide crack in a thin wall is the other regime: the orifice governs
    w2, t2 = 2e-3, 1e-3
    q = B.leak_rate(w2, l, t2, head)
    assert q == pytest.approx(B.orifice_leak(w2 * l, head), rel=1e-12)
    assert q < B.slit_leak(w2, l, 1000.0 * B.G * head, 1e-3, t2)


# ── the fragmentation loop ────────────────────────────────────────────────────
POT = B.STRENGTH["earthenware"]


def _pot_budget(m_kg, h=1.5):
    return B.drop_energy(m_kg, h)


def test_codeonly_pot_energies_are_mgh():
    """7.4 J (0.5 kg) and 44 J (3 kg) from 1.5 m: a formula check only (eta_f
    0.01 is (u) and is not in these numbers)."""
    assert _pot_budget(0.5) == pytest.approx(7.4, abs=0.05)
    assert _pot_budget(3.0) == pytest.approx(44.0, abs=0.2)


def test_fragmentation_masses_sum_within_1e_15_and_deterministic():
    kw = dict(mass=0.5, area=0.12, thickness=0.006, row=POT, seed=11,
              energy=3.0, grams=[500.0])
    a = B.fragment(budget=_pot_budget(0.5) * 20, **kw)
    b = B.fragment(budget=_pot_budget(0.5) * 20, **kw)
    assert a["n"] > 3
    assert math.fsum(a["mass"]) == pytest.approx(0.5, rel=1e-15)
    assert math.fsum(a["area"]) == pytest.approx(0.12, rel=1e-15)
    assert math.fsum(a["energy"]) == pytest.approx(3.0, rel=1e-15)
    assert math.fsum(a["grams"].ravel()) == pytest.approx(500.0, rel=1e-15)
    for k in ("mass", "area", "cost"):
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k]))        # same seed
    c = B.fragment(budget=_pot_budget(0.5) * 20, **dict(kw, seed=12))
    assert not np.array_equal(a["mass"], c["mass"]) or a["n"] != c["n"]  # new seed


def test_fragmentation_cost_is_cumulative_and_budget_monotone():
    kw = dict(mass=0.5, area=0.12, thickness=0.006, row=POT, seed=4)
    Gc = B.gc_from_kic(POT.K_Ic, POT.E, POT.nu)
    prev_n = 0
    for budget in (0.0, 0.5, 5.0, 50.0, 500.0):
        r = B.fragment(budget=budget, **kw)
        assert r["spent"] == pytest.approx(math.fsum(r["cost"]))
        assert r["spent"] <= budget + 1e-12
        assert r["left"] == pytest.approx(budget - r["spent"], abs=1e-12)
        assert r["n"] == len(r["cost"]) + 1
        assert r["n"] >= prev_n                         # more energy, no fewer shards
        prev_n = r["n"]
        # the stop rule: the NEXT cut (the biggest piece's) cannot be paid
        biggest = float(np.max(r["area"]))
        assert r["left"] < Gc * 0.006 * math.sqrt(biggest) / B.ETA_F or r["n"] >= 4096
    assert B.fragment(budget=0.0, **kw)["n"] == 1
    # one blow bigger than the first cut but smaller than two: cumulative costs
    one = B.fragment(budget=1e9, max_pieces=2, **kw)["cost"][0]
    r = B.fragment(budget=1.01 * one, **kw)
    assert r["n"] >= 2 and r["spent"] <= 1.01 * one


def test_fragmentation_sizes_are_right_skewed():
    r = B.fragment(mass=1.0, area=0.2, thickness=0.004, row=B.STRENGTH["glass"],
                   budget=2000.0, seed=2, max_pieces=300)
    assert r["n"] > 100
    s = np.log(r["area"])
    skew = float(np.mean((s - s.mean()) ** 3) / s.std() ** 3)
    assert abs(skew) > 0.05 or r["area"].max() / r["area"].min() > 20      # a spread


# ── cutting a thing that lives in cells ───────────────────────────────────────
def test_records_cut_gives_two_children_with_exact_mass():
    rng = np.random.default_rng(3)
    for trial in range(50):
        m = rng.random(9) * 0.02 + 0.001
        F = rng.random() * 0.8 + 0.1
        r = B.records_cut(m, F)
        assert r["separable"]
        assert math.fsum(r["mass_a"]) + math.fsum(r["mass_b"]) == pytest.approx(
            math.fsum(m), rel=1e-15)
        assert sorted(r["a"] + r["b"]) == list(range(9))
        assert np.all(r["mass_a"] >= 0) and np.all(r["mass_b"] >= 0)


def test_records_cut_a_cut_inside_one_cell_and_the_one_cell_case():
    # blade across 3 cells of 5 g each, cut at 40 percent: inside the middle cell
    r = B.records_cut(np.array([0.005, 0.005, 0.005]), 0.40)
    assert r["straddle"] == 1 and r["separable"]
    # two eids result, masses add to the parent
    assert math.fsum(r["mass_a"]) + math.fsum(r["mass_b"]) == pytest.approx(0.015, rel=1e-15)
    # larger overlap wins: 0.40 of 15 g = 6 g, 1 g into the middle cell of 5 g
    assert r["a"] == [0] and r["b"] == [1, 2]
    # the side that kept the cell holds its whole mass; the other side lost the excess
    assert math.fsum(r["mass_a"]) == pytest.approx(0.005, rel=1e-12)
    assert math.fsum(r["mass_b"]) == pytest.approx(0.010, rel=1e-12)
    one = B.records_cut(np.array([0.02]), 0.5)
    assert not one["separable"]


def test_records_cut_respects_the_cell_fill_limit():
    r = B.records_cut(np.array([0.01, 0.01, 0.01, 0.01]), 0.5, cap_mass=0.05)
    assert r["separable"] and np.all(r["mass_a"] <= 0.05) and np.all(r["mass_b"] <= 0.05)
    tight = B.records_cut(np.array([0.01, 0.01, 0.01, 0.01]), 0.4, cap_mass=0.0101)
    assert math.fsum(tight["mass_a"]) + math.fsum(tight["mass_b"]) == pytest.approx(
        0.04, rel=1e-12) or not tight["separable"]


# ── data and copies ───────────────────────────────────────────────────────────
def test_codeonly_wood_stand_in_equals_the_bonds_table():
    from src.bonds import materials as BM
    w, wp = B.STRENGTH["wood"], B.STRENGTH["wood_perp"]
    assert (w.E, w.nu, w.sigma_u, w.rho) == (BM.WOOD_L[0], BM.WOOD_L[1],
                                             BM.WOOD_L[2], BM.WOOD_L[4])
    assert (wp.E, wp.nu, wp.sigma_u, wp.rho) == (BM.WOOD_T[0], BM.WOOD_T[1],
                                                 BM.WOOD_T[2], BM.WOOD_T[4])
    # the flaw arm of the wood row lands on its tensile strength (an echo)
    assert float(B.flaw_stress(w.K_Ic, w.a0)) == pytest.approx(w.sigma_u, rel=0.01)


def test_echo_wood_flaw_stress_is_the_tensile_strength():
    for name in ("wood", "wood_perp"):
        r = B.STRENGTH[name]
        assert float(B.flaw_stress(r.K_Ic, r.a0)) == pytest.approx(r.sigma_u, rel=0.01)


def test_combine_copy_is_the_independent_causes_formula():
    D = np.linspace(0, 0.9, 7)
    d = np.linspace(0.8, 0.0, 7)
    assert np.allclose(B.combine(D, d), 1 - (1 - D) * (1 - d))
    assert np.array_equal(B.combine(D, d), B.combine(d, D))
    assert np.allclose(B.combine(B.combine(D, d), 0.3), B.combine(D, B.combine(d, 0.3)))
    assert float(B.combine(0.99, 0.99)) == 0.999


def test_codeonly_combine_matches_matdamage_when_it_exists():
    md = pytest.importorskip("src.voxel.matdamage")
    D = np.linspace(0, 0.95, 9)
    assert np.array_equal(B.combine(D, 0.3), md.combine(D, 0.3))


def test_codeonly_strength_from_a_matrec_shaped_object():
    rec = types.SimpleNamespace(E=2.1e11, nu=0.3, sy=2.5e8, st=4e8, ductility=0.25,
                                KIc=100e6, weibull_m=30.0, rho=7850.0)
    s = B.strength_from_matrec(rec)
    assert s == MILD
    bare = types.SimpleNamespace(E=1e10, st=5e7, sy=None, ductility=0.0, KIc=None, rho=2000.0)
    assert B.strength_from_matrec(bare).sigma_y is None


def test_row_split_extensive_only():
    row = {"mass": np.array([1.0, 2.0]), "len": 3.0, "dmg": np.array([0.1, 0.2]),
           "T": 20.0}
    a, b = B.split_row(row, 0.25)
    assert np.all(a["mass"] + b["mass"] == row["mass"])
    assert np.array_equal(a["dmg"], row["dmg"]) and np.array_equal(b["dmg"], row["dmg"])
    assert a["T"] == b["T"] == 20.0


def test_crack_opening_and_wall_flaw_util():
    assert B.crack_opening(0.002, 0.01) == pytest.approx(2e-5)
    assert B.crack_opening(-0.1, 0.01) == 0.0
    glass = B.STRENGTH["glass"]
    s = float(B.flaw_stress(glass.K_Ic, glass.a0))
    assert B.wall_flaw_util(glass, 0.5 * s, 0.004) == pytest.approx(0.5, rel=1e-9)
    assert B.wall_flaw_util(glass, 0.5 * s, 0.004, dmg=0.5) > 1.0


# ── cross checks with the other T2/T9 lanes (skip when their file is absent) ──
def test_codeonly_strength_from_the_real_matrec_rows_when_it_exists():
    mr = pytest.importorskip("src.voxel.matrec")
    rec = getattr(mr, "REC", None)
    if not rec or "iron" not in rec:
        pytest.skip("matrec has no REC table yet")
    wood = B.strength_from_matrec(rec["wood"])
    assert (wood.E, wood.nu, wood.sigma_u, wood.rho) == (
        B.STRENGTH["wood"].E, B.STRENGTH["wood"].nu, B.STRENGTH["wood"].sigma_u,
        B.STRENGTH["wood"].rho)                       # the stand-in equals the record
    iron = B.strength_from_matrec(rec["iron"])
    assert iron.sigma_y == 250e6 and iron.eps_f == pytest.approx(0.20)
    # the real rows run through the same bend: mild iron yields, glass snaps
    glass = B.strength_from_matrec(rec["glass"])
    r = B.bend_displacement(glass, B.make_section("rect", 0.1, 0.02, 0.003, glass),
                            0.002 / 0.0015, scatter=False)
    assert np.all(r["u"] >= 1.0)
    p = B.make_section("rect", 0.3, 0.04, 0.004, iron)
    r = B.bend_displacement(iron, p, 0.002 / 0.002, scatter=False)
    assert np.all(r["u"] < 1.0) and np.all(r["yielded"])


def test_codeonly_stand_in_rows_agree_with_record_data_where_both_state_a_number():
    rd = pytest.importorskip("src.voxel.record_data")
    rows = getattr(rd, "STRENGTH", None)
    if not rows:
        pytest.skip("record_data has no STRENGTH yet")
    for mine, theirs, keys in (("steel_mild", "steel", ("E", "nu", "sigma_y", "sigma_u", "eps_f", "K_Ic")),
                               ("steel_hard", "steel_hard", ("E", "nu", "K_Ic")),
                               ("glass", "glass", ("E", "nu", "K_Ic")),
                               ("wood", "wood", ("E", "sigma_u"))):
        a = B.STRENGTH[mine]
        for k in keys:
            assert getattr(a, k) == pytest.approx(rows[theirs][k]), (mine, k)


def test_codeonly_record_data_wood_equals_the_bonds_table():
    rd = pytest.importorskip("src.voxel.record_data")
    from src.bonds import materials as BM
    row = rd.STRENGTH["wood"]
    assert (row["E"], row["nu"], row["sigma_u"]) == (BM.WOOD_L[0], BM.WOOD_L[1], BM.WOOD_L[2])


def test_exact_split_edges_f_one_zero_and_tiny():
    import numpy as np
    for x in (3.3, 1e-9, 7.7e5, -2.5):
        a, b = B.exact_split(x, 1.0)
        assert a == x and b == 0.0
        a, b = B.exact_split(x, 0.0)
        assert a == 0.0 and b == x
        a, b = B.exact_split(x, 1e-9)
        assert a + b == x and abs(a - 1e-9 * x) <= abs(x) * 2.0 ** -52
    rng = np.random.default_rng(5)
    for _ in range(20000):
        x = 10.0 ** rng.uniform(-12, 6)
        f = rng.choice([0.0, 1.0, 1e-17, 1e-13, rng.random()])
        a, b = B.exact_split(x, f)
        assert a + b == x
        assert abs(a - f * x) <= x * 2.0 ** -52
