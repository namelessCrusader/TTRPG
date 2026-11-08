"""T2 thin heat: the solver itself, on RANDOM seeds (wave 13 polish, decision N1).

The old loop recursed up to 12 halvings with no work limit on rows that never converged: a
1.4 mm water film at h 5900 crossing both the 100 C and 0 C plateaus (120.7 C to -55.4 C, dt
0.1) cost 17-22 s a tick, and extreme sweeps left 15-2333 unconverged leaf rows. Hand-picked
cases had passed; random draws found it. So these tests draw their cases from seeds.

What is asserted, on every draw: the maximum principle (no node outside the range of the
start and the bath), exact books, no row over its work budget, nothing unconverged, and a
backward-Euler residual for constant-k materials.
"""
import time

import numpy as np
import pytest

from src.voxel import record as R
from src.voxel import record_data as rd
from src.voxel import thinheat as th

COMPS = [{"water": 1.0}, {"water": 1.0}, {"wax": 1.0}, {"hide": .6, "water": .4}, {"hide": .9, "water": .1},
         {"hide": 1.0}, {"steel": 1.0}, {"wax": .5, "water": .5}, {"glass": 1.0}, {"iron": .7, "salt": .3},
         {"wood": .8, "water": .2}, {"water": .5, "salt": .5}]
METAL = ("steel", "iron", "glass")


def _material(comp):
    w = np.zeros(rd.NS)
    for k, v in comp.items():
        w[rd.SP[k]] = v
    return rd.row_material(w / w.sum())


def _draw(rng):
    comp = COMPS[int(rng.integers(len(COMPS)))]
    hot = any(k in comp for k in METAL)
    lo, hi = (0.0, 900.0) if hot else (-80.0, 160.0)
    Th, Tc = rng.uniform(lo, hi), rng.uniform(lo, hi)
    if rng.random() < 0.3:
        Tc = Th + rng.uniform(-3, 3)                          # near a plateau
    if not hot and rng.random() < 0.25:
        Th, Tc = rng.uniform(105, 160), rng.uniform(-80, -20)  # across BOTH plateaus
    return dict(comp=comp, thick=10 ** rng.uniform(-4, -1.7), h=10 ** rng.uniform(0.3, 5.2),
                nn=int(rng.choice([7, 13])), sym=bool(rng.integers(2)), Th=Th, Tc=Tc,
                dt=float(rng.choice([0.005, 0.025, 0.1, 0.5, 1.0])))


def _run(c, ticks):
    mat = _material(c["comp"])
    recs = R.Records()
    rid = R.make_thin(recs, R.SECTION, c["comp"], c["thick"], .1, .1, T=c["Th"], nn=c["nn"],
                      sym=c["sym"], ports=dict(h=(c["h"], c["h"]), T=(c["Tc"], c["Tc"])))
    p, r = recs.row(rid)
    lo, hi = min(c["Th"], c["Tc"]), max(c["Th"], c["Tc"])
    E0, Q, worst, tmax = recs.energy_J(rid), 0.0, 0.0, 0.0
    for t in range(ticks):
        recs.begin_tick(t)
        t0 = time.perf_counter()
        Q += sum(d[2] for d in R.heat_tick(recs, c["dt"]))
        tmax = max(tmax, time.perf_counter() - t0)
        T = mat.T_of_H(p.a["H"][r, :c["nn"]])
        assert np.isfinite(T).all()
        worst = max(worst, lo - T.min(), T.max() - hi)
    return worst, abs(recs.energy_J(rid) - E0 + Q) / (abs(E0) + 1.0), tmax


@pytest.mark.parametrize("seed", range(8))
def test_random_draws_keep_the_range_the_books_and_the_budget(seed):
    rng = np.random.default_rng(1000 + seed)
    th.stats(reset=True)
    tmax = 0.0
    for _ in range(14):
        c = _draw(rng)
        worst, eerr, t = _run(c, 12)
        tmax = max(tmax, t)
        assert worst < 1e-3, (worst, c)                       # the old bound was 0.05 K
        assert eerr < 1e-12, (eerr, c)
    s = th.stats()
    assert s["unconverged"] == 0 and s["capped"] == 0, s
    assert s["max_solves"] <= th.WORK_MAX
    assert tmax < 1.0                                          # an order of magnitude over the 0.05 s seen


def test_the_stall_case_from_the_verifier_costs_milliseconds():
    """1.4 mm water, h 5878, 120.7 C to -55.4 C, dt 0.1, 7 nodes: ticks 5-7 cost 16-22 s each,
    with 1387 to 1868 unconverged leaf rows."""
    c = dict(comp={"water": 1.0}, thick=0.0014, h=5878.0, nn=7, sym=False, Th=120.7, Tc=-55.4, dt=0.1)
    th.stats(reset=True)
    t0 = time.perf_counter()
    worst, eerr, tmax = _run(c, 30)
    wall = time.perf_counter() - t0
    s = th.stats()
    assert worst < 1e-3 and eerr < 1e-12
    assert s["unconverged"] == 0 and s["capped"] == 0 and s["max_solves"] <= th.WORK_MAX
    assert wall < 3.0 and tmax < 0.5, (wall, tmax)             # was 17-22 s for ONE tick


@pytest.mark.parametrize("seed", range(6))
def test_the_step_is_the_backward_euler_solution_for_constant_k(seed):
    """For a material whose k does not depend on T the lagged conductances ARE the real ones,
    so the substep must satisfy m (H_new - H0)/dt + (heat out through the conductances and
    ports at T(H_new)) = 0 to about INNER_TOL kelvin, plateaus or not."""
    rng = np.random.default_rng(2000 + seed)
    for _ in range(15):
        comp = [{"wax": 1.0}, {"hide": 1.0}, {"glass": 1.0}, {"wood": 1.0}, {"wax": .5, "hide": .5}][int(rng.integers(5))]
        mat = _material(comp)
        nn = int(rng.choice([7, 13]))
        sym = bool(rng.integers(2))
        thick = 10 ** rng.uniform(-3.5, -2)
        K = 3
        T0 = rng.uniform(-20, 120, size=(K, 1)) + np.zeros((K, nn))
        T0 += rng.uniform(-5, 5, size=(K, nn))
        H0 = mat.H_of_T(T0)
        m = mat.rho * thick * 0.01 * th.node_fractions(nn, sym)[None, :] * np.ones((K, 1))
        gk = th.conduction_gk(np.full(K, 0.01), np.full(K, thick), nn, sym)
        Gp = np.stack([np.zeros(K) if sym else 10 ** rng.uniform(-1, 2, K), 10 ** rng.uniform(-1, 2, K)], axis=1)
        Te = rng.uniform(-60, 160, size=(K, 2))
        dt = float(rng.choice([0.01, 0.1, 1.0]))
        Hn, Q, ok, _u = th._substep(H0, m, gk, Gp, Te, mat, dt, th.NEWTON)
        assert ok.all()
        T = mat.T_of_H(Hn)
        g = th._k_if(mat.k(T)) * gk[:, None]
        res = m * (Hn - H0) / dt + th._k_mul(T, g, Gp, Te)             # W per node
        res_K = np.abs(res) * dt / (m * mat.c_app(Hn))                  # as a temperature error, K
        assert res_K.max() < 1e-4, (res_K.max(), comp)
        assert abs((m * (Hn - H0)).sum() + Q.sum()) <= 1e-12 * (abs((m * H0).sum()) + 1.0)


def test_water_k_is_continuous_across_the_melting_plateau():
    """k jumped from 2.2 to 0.598 at 0 C, in the middle of the plateau: a node on it flipped k
    with the sign of a thousandth of a kelvin and no pass could settle (the node ended 6 K away
    from the state it was solved for). It is the ice fraction's mix now."""
    T = np.linspace(-0.05, 0.05, 20001)
    k = rd._water_k(T)
    assert k[0] == pytest.approx(2.2, abs=1e-12) and k[-1] == pytest.approx(0.598, abs=1e-12)
    assert np.abs(np.diff(k)).max() < 2e-3                   # no step: 1.6 over 0.03 K in 6000 samples
    assert (np.diff(k) <= 1e-15).all()                       # ice conducts more than water
    mat = rd.MATERIALS["water"]
    assert mat.Tk[(mat.Tk > -0.0151) & (mat.Tk < 0.0151)].size >= 2     # the plateau knots exist


def test_work_is_bounded_and_the_report_has_its_counts():
    s = th.stats(reset=True)
    assert {"capped", "bisected", "unconverged", "unsettled", "solves", "max_solves"} <= set(s)
    mat = rd.MATERIALS["steel"]
    H = np.full((4, 7), float(mat.H_of_T(800.0)))
    thick = 0.002
    m = mat.rho * thick * 0.01 * th.node_fractions(7, True)[None, :] * np.ones((4, 1))
    gk = th.conduction_gk(np.full(4, 0.01), np.full(4, thick), 7, True)
    Hn, Q, ns = th.step(H, m, gk, np.tile([0.0, 60.0], (4, 1)), np.full((4, 2), 20.0), mat, 0.025)
    s = th.stats()
    assert 0 < s["solves"] <= 4 * int(ns.max()) * th.WORK_MAX * (2 ** (th.BISECT_MAX + 1) - 1)
    assert s["max_solves"] <= th.WORK_MAX


@pytest.mark.parametrize("seed", range(4))
def test_a_rows_result_does_not_depend_on_its_batch_mates_random(seed):
    """Rows are solved in sub-batches and a finished row is not touched again: a row alone and
    the same row among hard ones (plateaus, stiff ports, other substep counts) are bit-identical."""
    rng = np.random.default_rng(3000 + seed)
    for name in ("water", "steel", "wax"):
        mat = rd.MATERIALS[name]
        nn, sym, K = 7, True, 6
        thick = 10 ** rng.uniform(-4, -2.3, K)
        T0 = rng.uniform(-40, 130, K) if name != "steel" else rng.uniform(100, 850, K)
        Tc = rng.uniform(-60, 150, K) if name != "steel" else rng.uniform(0, 500, K)
        h = 10 ** rng.uniform(1, 5, K)
        m = mat.rho * thick[:, None] * 0.01 * th.node_fractions(nn, sym)[None, :]
        gk = th.conduction_gk(np.full(K, 0.01), thick, nn, sym)
        H = np.repeat(mat.H_of_T(T0)[:, None], nn, axis=1)
        Gp = np.stack([np.zeros(K), 2.0 * h * 0.01], axis=1)
        Te = np.stack([Tc, Tc], axis=1)
        Hb, Qb, nb = th.step(H, m, gk, Gp, Te, mat, 0.05)
        for i in range(K):
            sl = slice(i, i + 1)
            Ha, Qa, na = th.step(H[sl], m[sl], gk[sl], Gp[sl], Te[sl], mat, 0.05)
            assert np.array_equal(Ha[0], Hb[i]) and np.array_equal(Qa[0], Qb[i]) and na[0] == nb[i], (name, i)


# ── what "unsettled" counts (STATS['unsettled'], wave 13 polish 2) ───────────────────────────────
# Definition: of the row-substeps whose results were KEPT (STATS['rows']), the ones that finished
# their solves (so they are not 'unconverged') but whose last conductance (k) corrector pass still
# moved a node by more than OUTER_TOL. k(T) is lagged by `newton` passes; the state returned is
# still the exact backward-Euler step for the conductances of that last pass (energy exact, the
# maximum principle holds) and only the lag of k is left. A constant-k row, a one-pass call and a
# row that did not finish are never counted. Only a material whose k changes with T can be counted:
# in practice water on its melting ramp, where k falls from 2.2 to 0.598 inside 0.03 K.

def _thaw_batch(rng, K=8, nn=7, sym=True):
    """K water films, each starting uniform and meeting a bath within -30..30 C (the ramp is in reach)."""
    mat = rd.MATERIALS["water"]
    thick = 10 ** rng.uniform(-3.5, -2.3, K)
    h = 10 ** rng.uniform(2, 4.5, K)
    T0, Tc = rng.uniform(-30, 30, K), rng.uniform(-30, 30, K)
    m = mat.rho * thick[:, None] * 0.01 * th.node_fractions(nn, sym)[None, :]
    gk = th.conduction_gk(np.full(K, 0.01), thick, nn, sym)
    H = np.repeat(mat.H_of_T(T0)[:, None], nn, axis=1)
    Gp = np.stack([np.zeros(K), 2.0 * h * 0.01], axis=1)
    Te = np.stack([Tc, Tc], axis=1)
    return mat, H, m, gk, Gp, Te, T0, Tc


def _thaw_sweep(newton, seeds=range(20)):
    """(rows kept, unsettled, not ok) over random thaw batches, two dt each, from _substep itself."""
    n = u = bad = 0
    for seed in seeds:
        mat, H, m, gk, Gp, Te, T0, Tc = _thaw_batch(np.random.default_rng(seed))
        for dt in (0.05, 0.5):
            Hn, Q, ok, uns = th._substep(H, m, gk, Gp, Te, mat, dt, newton)
            assert not uns[~ok].any()                       # a row that did not finish is never 'unsettled'
            n += ok.size
            u += int(uns.sum())
            bad += int((~ok).sum())
    return n, u, bad


def test_unsettled_is_zero_at_the_default_pass_count_on_random_thaws():
    """With NEWTON passes every random water film across its melting ramp settles. At 8 passes 2 of
    320 row-substeps did not (a node on the ramp up to 0.57 K away from the settled state), at 2
    passes 12: so a build that cuts the pass count back makes this fail."""
    n, u, bad = _thaw_sweep(th.NEWTON)
    assert n == 320 and bad == 0 and u == 0, (n, u, bad)


def test_unsettled_can_be_positive_when_k_has_too_few_passes_and_the_state_is_still_exact():
    """Starve the passes (2): k on the melting ramp has not settled, so some rows are counted.
    Each such row is still an exact backward-Euler state for its last conductances: books exact to
    round-off, no node outside the range of the start and the bath, and nearer to the settled
    (many-pass) state than the start state is."""
    found = 0
    for seed in range(20):
        mat, H, m, gk, Gp, Te, T0, Tc = _thaw_batch(np.random.default_rng(seed))
        for dt in (0.05, 0.5):
            Hn, Q, ok, uns = th._substep(H, m, gk, Gp, Te, mat, dt, 2)
            Hs, Qs, oks, unss = th._substep(H, m, gk, Gp, Te, mat, dt, 64)
            assert ok.all() and not unss.any()
            for i in np.flatnonzero(uns):
                found += 1
                e0 = (m[i] * H[i]).sum()
                assert abs((m[i] * (Hn[i] - H[i])).sum() + Q[i].sum()) <= 1e-12 * (abs(e0) + 1.0)
                T = mat.T_of_H(Hn[i])
                lo, hi = min(T0[i], Tc[i]), max(T0[i], Tc[i])
                assert T.min() >= lo - 1e-3 and T.max() <= hi + 1e-3
                d_set = np.abs(T - mat.T_of_H(Hs[i])).max()
                d_start = np.abs(mat.T_of_H(H[i]) - mat.T_of_H(Hs[i])).max()
                assert d_set < d_start or d_start < 1e-3
    assert found >= 5, found


def test_unsettled_is_never_counted_for_a_one_pass_call_or_a_constant_k_row():
    """One pass has no corrector, so nothing is measured; a constant-k row needs no second pass.
    (The pass-0 move is the step itself, not a lag of k.)"""
    n, u, bad = _thaw_sweep(1)
    assert u == 0 and bad == 0
    rng = np.random.default_rng(7)
    for name in ("glass", "wax", "hide", "wood"):
        mat = rd.MATERIALS[name] if name in rd.MATERIALS else _material({name: 1.0})
        K, nn = 6, 7
        thick = 10 ** rng.uniform(-3.5, -2, K)
        T0, Tc = rng.uniform(-30, 130, K), rng.uniform(-30, 130, K)
        m = mat.rho * thick[:, None] * 0.01 * th.node_fractions(nn, True)[None, :]
        gk = th.conduction_gk(np.full(K, 0.01), thick, nn, True)
        H = np.repeat(mat.H_of_T(T0)[:, None], nn, axis=1)
        Gp = np.stack([np.zeros(K), 10 ** rng.uniform(0, 3, K)], axis=1)
        Te = np.stack([Tc, Tc], axis=1)
        for dt in (0.01, 1.0):
            _H, _Q, ok, uns = th._substep(H, m, gk, Gp, Te, mat, dt, th.NEWTON)
            assert ok.all() and not uns.any(), name


def test_the_counts_cover_the_kept_results_only_and_are_disjoint():
    """'rows' counts kept row-substeps; 'unsettled' is out of those; a row thrown away and redone
    in halves is counted by its halves, never twice; unconverged rows are not 'unsettled'."""
    mat, H, m, gk, Gp, Te, T0, Tc = _thaw_batch(np.random.default_rng(3))
    th.stats(reset=True)
    th._robust_substep(H, m, gk, Gp, Te, mat, 0.5, th.NEWTON)               # nothing capped: rows = K
    s = th.stats()
    assert s["rows"] == H.shape[0] and s["unsettled"] <= s["rows"] and s["bisected"] == 0
    th.stats(reset=True)
    th._robust_substep(H, m, gk, Gp, Te, mat, 0.5, th.NEWTON, 0, 2)         # a budget of 2 solves: all bisect
    s = th.stats()
    assert s["bisected"] > 0 and s["rows"] >= H.shape[0] and s["unsettled"] <= s["rows"]
    assert s["unconverged"] <= s["rows"]
    # the leaf is what is counted: with a budget no row can meet, nothing finishes and nothing is 'unsettled'
    th.stats(reset=True)
    Hn, Q, ok, uns = th._substep(H, m, gk, Gp, Te, mat, 0.5, th.NEWTON, 1)
    assert not ok.all() and not uns[~ok].any()
    # a budget of ONE solve: no row finishes at any level, so every row ends as 2^BISECT_MAX leaves,
    # each kept once and each unconverged; the thrown-away parents are not counted
    hard = ~th._substep(H, m, gk, Gp, Te, mat, 0.5, th.NEWTON, 1)[2]       # rows that need more than one solve
    K = int(hard.sum())
    assert K >= 2
    H, m, gk, Gp, Te = H[hard], m[hard], gk[hard], Gp[hard], Te[hard]
    th.stats(reset=True)
    th._robust_substep(H, m, gk, Gp, Te, mat, 0.5, th.NEWTON, 0, 1)
    s = th.stats()
    # (a half step may need only one solve, so a leaf can finish: the rows are 2^BISECT_MAX at most
    # per row and at least 2, and every row is counted once per kept leaf, never for a discarded parent)
    assert 2 * K <= s["rows"] <= K * 2 ** th.BISECT_MAX and s["unconverged"] <= s["rows"], s
    assert s["unsettled"] + s["unconverged"] <= s["rows"], s


def test_random_ticks_keep_unsettled_at_zero():
    """The 400-draw sweep the verifier saw read 9 'unsettled' at 8 passes; now zero, on draws from
    seeds that include water, brine and wood mixes across the melting ramp."""
    th.stats(reset=True)
    for seed in range(12):
        rng = np.random.default_rng(90000 + seed)
        for _ in range(10):
            c = _draw(rng)
            worst, eerr, _t = _run(c, 10)
            assert worst < 1e-3 and eerr < 1e-12, (worst, eerr, c)
    s = th.stats()
    assert s["rows"] > 0 and s["unsettled"] == 0 and s["unconverged"] == 0, s


def test_a_row_that_ran_out_of_its_budget_is_never_unsettled():
    """'unsettled' is for rows that FINISHED their solves; a row that ran out of budget part-way
    through the corrector passes is 'not ok' and goes to bisection, never into the unsettled count."""
    cut = 0
    for seed in range(10):
        mat, H, m, gk, Gp, Te, T0, Tc = _thaw_batch(np.random.default_rng(seed))
        for budget in range(1, 9):
            Hn, Q, ok, uns = th._substep(H, m, gk, Gp, Te, mat, 0.5, th.NEWTON, budget)
            assert not (uns & ~ok).any(), (seed, budget)
            cut += int((~ok).sum())
    assert cut > 50
