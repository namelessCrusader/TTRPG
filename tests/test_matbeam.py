"""T9 beams, columns, masonry sections, arches and the load path (src/voxel/matbeam.py).

Every test is against a textbook closed form or an exact identity, except the
ones named test_codeonly_* ([C]: they test the code against its own derate
table, not steel) and test_u_* ([U]: a number nobody has sourced yet).

The record is a stand-in dict of the 12 rows (spec T9-material 4.1) so these
tests do not depend on matrec: ROWS below. Tags in this file: closed-form and
exactness lines are the default; [C] 1 test; [U] 0 tests (the wood strengths
are used as inputs, never as an expected result).
"""
import math

import numpy as np
import pytest

from src.voxel import matbeam as B

AIR, WOOD, STONE, IRON, FLESH, ASH, GLASS, LEAF, CHAR, TIN, LEAD, ROPE = range(12)

_IRON_D = ((20, 1, 1), (100, 1, 1), (200, 1, .9), (300, 1, .8), (400, 1, .7), (500, .78, .6),
           (600, .47, .31), (700, .23, .13), (800, .11, .09), (900, .06, .0675),
           (1000, .04, .045), (1100, .02, .0225), (1200, 0, 0))
_STONE_D = ((20, 1, 1), (100, 1, 1), (200, .95, .95), (300, .85, .85), (400, .75, .75),
            (500, .6, .6), (600, .45, .45), (700, .3, .3), (800, .15, .15), (900, .08, .08),
            (1000, .04, .04))
_WOOD_D = ((20, 1, 1), (100, .75, .75), (200, .5, .5), (300, .2, .2), (350, 0, 0))
_GLASS_D = ((20, 1, 1), (400, 1, 1), (500, .6, .6), (550, .3, .3), (720, .02, .02))


def _row(rho, E, nu, sy, st, sc, sf, tau, duct, derate=None):
    return dict(rho=rho, E=E, nu=nu, sy=sy, st=st, sc=sc, sf=sf, tau=tau, ductility=duct,
                derate=derate)


ROWS = [
    _row(1.2, 0, 0, None, 0, 0, None, 0, 0),                                  # AIR
    _row(600, 1.1e10, .35, None, 6e7, 4e7, 7e7, 8e6, .01, _WOOD_D),          # WOOD
    _row(2700, 5e10, .25, None, 1e7, 1.5e8, 1.5e7, 2e7, .001, _STONE_D),      # STONE
    _row(7800, 2e11, .29, 2.5e8, 3.5e8, 2.5e8, None, 1.45e8, .2, _IRON_D),    # IRON
    _row(1000, 1e6, .45, None, 1e6, 1e6, 1e6, 3e5, .5),                       # FLESH
    _row(150, 1e8, .3, None, 0, 1e4, 0, 0, 0),                                # ASH
    _row(2500, 7e10, .22, None, 5e7, 1e9, 5e7, 3e7, 0, _GLASS_D),             # GLASS
    _row(200, 2e9, .3, None, 2e7, 0, 0, 0, .1),                               # LEAF
    _row(300, 1e9, .3, None, 1e6, 2e6, 1e6, 1e6, .005),                       # CHAR
    _row(7300, 5e10, .33, 1.1e7, 2e7, 1.1e7, None, 1.2e7, .4, ((20, 1, 1), (232, 0, 0))),   # TIN
    _row(11340, 1.6e10, .44, 5.5e6, 1.2e7, 5.5e6, None, 6e6, .5, ((20, 1, 1), (327, 0, 0))),  # LEAD
    _row(885, 9.4e8, .4, None, 9.4e7, 0, 0, 0, .15),                          # ROPE
]
REC = B.rec_from_rows(ROWS)
RHO = REC["rho"]
H5 = 0.05


# ── a tiny scene builder ────────────────────────────────────────────────────
class Scene:
    def __init__(self, nx, ny, nz, h=H5):
        self.h = h
        self.mat = np.zeros((nx, ny, nz), np.int64)
        self.f = np.zeros((nx, ny, nz))
        self.eid = np.zeros((nx, ny, nz), np.int64)
        self.rubble = np.zeros((nx, ny, nz), bool)
        self.grip = np.zeros((nx, ny, nz), bool)
        self.owned = np.zeros((nx, ny, nz), bool)
        self.T = np.full((nx, ny, nz), 20.0)
        self.dmg = np.zeros((nx, ny, nz))

    def put(self, x0, x1, y0, y1, z0, z1, mat, f=1.0, eid=0):
        sl = (slice(x0, x1), slice(y0, y1), slice(z0, z1))
        self.mat[sl], self.f[sl], self.eid[sl] = mat, f, eid
        return self

    def members(self):
        return B.extract_members(self.mat > 0, self.f, self.mat, self.eid, self.dmg, self.T,
                                 self.rubble, self.owned, self.grip, self.eid > 0, None, None,
                                 self.h, rec=REC)

    def weights(self):
        return RHO[self.mat] * self.f * self.h ** 3 * B.G * (self.mat > 0)

    def run(self, W=None, X=None, **kw):
        m = self.members()
        W = self.weights() if W is None else W
        W = np.where(self.grip, 0.0, W)
        loads = B.accumulate_loads(m, W, X, **kw)
        return m, loads, B.analyse(m, REC, loads)


def conserved(loads):
    return np.allclose(loads.ground + loads.edge + loads.lost, loads.total, rtol=1e-12, atol=1e-9)


def plank_on_posts(n, mat=WOOD, f=1.0, h=H5, lanes=1, zpost=3, eid=0, nz=None):
    """A plank of n cells, `lanes` wide, on two one-cell-wide (full width) stone
    posts, on a stone floor."""
    s = Scene(n + 4, lanes + 2, nz or zpost + 3, h)
    s.put(0, n + 4, 0, lanes + 2, 0, 1, STONE)
    s.put(0, 2, 1, lanes + 1, 1, zpost + 2, STONE)           # posts reach the plank's layer
    s.put(n + 2, n + 4, 1, lanes + 1, 1, zpost + 2, STONE)
    s.put(2, n + 2, 1, lanes + 1, zpost + 1, zpost + 2, mat, f, eid)
    return s


# ── the record ──────────────────────────────────────────────────────────────
def test_member_test_lists_exactly_the_structural_rows():
    got = {i for i, ok in enumerate(B.structural_mask(REC)) if ok}
    assert got == {WOOD, STONE, IRON, GLASS, TIN, LEAD, CHAR}


def test_effective_values_apply_temperature_and_damage():
    e0 = B.effective(REC, np.array([IRON]), np.array([20.0]), np.array([0.0]))
    assert e0["sy"][0] == 2.5e8 and e0["E"][0] == 2e11 and e0["sc"][0] == 2.5e8
    e = B.effective(REC, np.array([IRON]), np.array([600.0]), np.array([0.19]))
    assert e["sy"][0] == pytest.approx(2.5e8 * 0.47 * 0.81)
    assert e["sc"][0] == pytest.approx(2.5e8 * 0.47 * math.sqrt(0.81))
    assert e["E"][0] == pytest.approx(2e11 * 0.31 * 0.81)
    assert np.isnan(B.effective(REC, np.array([WOOD]), np.array([20.0]), np.array([0.0]))["sy"][0])


# ── sections ────────────────────────────────────────────────────────────────
def test_bar_and_plate_section_laws():
    f = np.array([1.0, 0.5, 0.25])
    bar = B.section(f, H5, False)
    assert np.allclose(bar["A"], f * H5 ** 2)
    assert np.allclose(bar["S"], H5 ** 3 * f ** 1.5 / 6)
    assert np.allclose(bar["I"], H5 ** 4 * f ** 2 / 12)
    pl = B.section(f, H5, True)
    assert np.allclose(pl["S"], H5 ** 3 * f ** 2 / 6)
    assert np.allclose(pl["I"], H5 ** 4 * f ** 3 / 12)
    assert np.allclose(pl["b"], H5)


def test_a_beam_twice_as_deep_carries_four_times_the_moment():
    one = B.stack_section(np.array([[1.0, 0.0]]), H5, [False])
    two = B.stack_section(np.array([[1.0, 1.0]]), H5, [False])
    assert two["S"][0] / one["S"][0] == pytest.approx(4.0, rel=1e-12)
    assert two["I"][0] == pytest.approx(2 * H5 ** 4 / 3, rel=1e-12)        # parallel axis, not 2 x I
    assert two["I"][0] > 2 * one["I"][0] * 3.9


# ── beam closed forms: L = n h + h/2 ────────────────────────────────────────
def _cells(n, h=H5):
    i = np.arange(1, n + 1)
    return (i - 0.25) * h, n * h + h / 2


def test_pinned_uniform_load_midspan_moment():
    for n, tol in ((20, 0.02), (40, 0.01)):
        x, L = _cells(n)
        w = 1000.0                                  # N/m
        W = np.full(n, w * H5)
        M, V = B.pinned_moments(W, x, L)
        assert abs(M.max() - w * L ** 2 / 8) / (w * L ** 2 / 8) < tol
        R0, R1 = B.pinned_reactions(W, x, L)
        assert R0 + R1 == pytest.approx(W.sum(), rel=1e-12)
        assert R0 == pytest.approx(R1, rel=1e-12)
        assert V.max() == pytest.approx(R0, rel=1e-12)


def test_pinned_midspan_point_load_and_one_cell():
    x, L = _cells(21)                               # the middle cell sits at L/2
    assert x[10] == pytest.approx(L / 2)
    W = np.zeros(21)
    W[10] = 500.0
    M, V = B.pinned_moments(W, x, L)
    assert M[10] == pytest.approx(500 * L / 4, rel=1e-12)
    assert V.max() == pytest.approx(250.0, rel=1e-12)
    x1, L1 = _cells(1)
    M1, _ = B.pinned_moments(np.array([100.0]), x1, L1)
    assert M1[0] == pytest.approx(0.375 * 100 * H5, rel=1e-12)


def test_cantilever_root_moment_tip_and_uniform():
    n = 20
    x = (np.arange(n) + 0.5) * H5
    L = n * H5
    W = np.zeros(n)
    W[-1] = 100.0
    M, V = B.cantilever_moments(W, x, H5)
    assert M[0] == pytest.approx(100.0 * x[-1], rel=1e-12)                 # P x of the tip cell
    assert V[0] == pytest.approx(100.0)
    w = 800.0
    Wu = np.full(n, w * H5)
    Mu, Vu = B.cantilever_moments(Wu, x, H5)
    assert Mu[0] == pytest.approx(w * L ** 2 / 2, rel=1e-12)               # at the support face
    assert Vu[0] == pytest.approx(w * L, rel=1e-12)
    assert Mu[10] == pytest.approx(w * (L - 10 * H5) ** 2 / 2, rel=1e-12)  # left face of cell 10


def test_pinned_demand_checks_the_faces_too():
    x, L = _cells(10)
    W = np.zeros(10)
    W[5] = 100.0
    M, _ = B.pinned_demand(W, x, L, H5)
    Mpt, _ = B.pinned_moments(W, x, L)
    assert (M >= Mpt - 1e-12).all()
    assert M[4] > Mpt[4]                                                     # the neighbour's face


def test_deflection_closed_forms_and_dead_load_screen():
    EI = 1.1e10 * H5 ** 4 / 12
    assert B.deflection_pinned_uniform(100.0, 1.0, EI) == pytest.approx(5 * 100 / (384 * EI))
    assert B.deflection_pinned_point(100.0, 1.0, EI) == pytest.approx(100 / (48 * EI))
    assert B.deflection_cantilever_uniform(100.0, 1.0, EI) == pytest.approx(100 / (8 * EI))
    assert B.deflection_cantilever_tip(100.0, 1.0, EI) == pytest.approx(100 / (3 * EI))
    # the screen: a beam loaded to its screen value (P at midspan) has u = 1
    x, L = _cells(21)                                # a cell sits at midspan
    W = np.full(21, 40.0)
    sec = B.section(np.array([1.0]), H5)
    Mcap = float(sec["S"][0] * 7e7)
    P = B.dead_load_capacity(Mcap, L, W.sum())
    Wp = W.copy()
    Wp[10] += P
    M, _ = B.pinned_moments(Wp, x, L)
    assert M.max() == pytest.approx(Mcap, rel=0.02)


def test_effective_width_rule_and_spread_conserves():
    assert B.effective_width(0.4, 0.1, 1.6) == pytest.approx(0.4)
    assert B.effective_width(2.0, 0.1, 0.9) == pytest.approx(0.4)
    p = np.zeros(8)
    p[3] = 1.0
    q = B.spread_lanes(p, H5, 1.6)
    assert q.sum() == pytest.approx(1.0, abs=1e-15)
    assert np.allclose(q, 1 / 8)
    q2 = B.spread_lanes(p, H5, 0.15)                 # a short span: the window is narrow
    assert q2.sum() == pytest.approx(1.0, abs=1e-15) and q2[3] > q2[0]


def _grillage_peak_share(nl, L, h, Et_over_E, d=0.05, lanes_loaded=(3, 4)):
    """numpy.linalg grillage: lanes are beams along the span, ties between lanes
    are beams across it (no twist stiffness: the cautious, less spreading case).
    Returns the biggest lane's share of the midspan moment."""
    nx = int(round(L / h)) + 1
    ndof = 3 * nx * nl                                # w, slope along x, slope along y

    def idx(i, j):
        return 3 * (i * nl + j)

    def bk(EI, l):
        return EI / l ** 3 * np.array([[12, 6 * l, -12, 6 * l], [6 * l, 4 * l * l, -6 * l, 2 * l * l],
                                       [-12, -6 * l, 12, -6 * l], [6 * l, 2 * l * l, -6 * l, 4 * l * l]])
    K = np.zeros((ndof, ndof))
    I_l = h * d ** 3 / 12
    for j in range(nl):
        for i in range(nx - 1):
            a, b = idx(i, j), idx(i + 1, j)
            dof = [a, a + 1, b, b + 1]
            K[np.ix_(dof, dof)] += bk(I_l, h)
    for i in range(nx):
        trib = h if 0 < i < nx - 1 else h / 2
        for j in range(nl - 1):
            a, b = idx(i, j), idx(i, j + 1)
            dof = [a, a + 2, b, b + 2]
            K[np.ix_(dof, dof)] += bk(Et_over_E * trib * d ** 3 / 12, h)
    F = np.zeros(ndof)
    mid = (nx - 1) // 2
    for j in lanes_loaded:
        F[idx(mid, j)] = 1.0 / len(lanes_loaded)
    fixed = [idx(0, j) for j in range(nl)] + [idx(nx - 1, j) for j in range(nl)]
    free = np.setdiff1d(np.arange(ndof), fixed)
    u = np.zeros(ndof)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    M = np.array([-I_l * (u[idx(mid - 1, j)] - 2 * u[idx(mid, j)] + u[idx(mid + 1, j)]) / h ** 2
                  for j in range(nl)])
    return float((M / M.sum()).max())


def test_effective_width_against_a_numpy_grillage():
    """A 1.6 m x 0.4 m board, 8 lanes, a 0.1 m foot at midspan: the rule's
    effective width is within 25 percent of the width the grillage's biggest
    lane implies (width = lane width / peak share)."""
    peak = _grillage_peak_share(8, 1.6, H5, 1.0)
    w_grill = H5 / peak
    w_rule = B.effective_width(8 * H5, B.FOOT_M, 1.6)
    assert abs(w_rule - w_grill) / w_grill < 0.25
    # the same through the code that spreads a foot's load over the lanes
    p = np.zeros(8)
    p[3] = p[4] = 0.5
    assert B.spread_lanes(p, H5, 1.6).max() == pytest.approx(1 / 8)
    assert abs(B.spread_lanes(p, H5, 1.6).max() - peak) / peak < 0.25


def test_effective_width_orthotropic_wood_needs_the_ortho_factor():
    """Across the grain wood is about 20 times softer; the plain rule then
    over-spreads (checked: 26 percent), with ortho = (E_perp/E)^0.25 it is inside 25."""
    peak = _grillage_peak_share(8, 1.6, H5, 0.05)
    w_grill = H5 / peak
    plain = B.effective_width(8 * H5, B.FOOT_M, 1.6)
    ortho = B.effective_width(8 * H5, B.FOOT_M, 1.6, ortho=0.05 ** 0.25)
    assert abs(ortho - w_grill) / w_grill < 0.25
    assert abs(ortho - w_grill) < abs(plain - w_grill)


# ── columns ─────────────────────────────────────────────────────────────────
def test_euler_5cm_pine_post_2m():
    E, b, H = 1.1e10, 0.05, 2.0
    I, A = b ** 4 / 12, b * b
    held = B.euler_critical_load(E, I, A, 4e7, H, B.K_HELD)
    free = B.euler_critical_load(E, I, A, 4e7, H, B.K_FREE)
    assert held == pytest.approx(14.1e3, rel=0.01)
    assert free == pytest.approx(3.5e3, rel=0.01)
    assert held == pytest.approx(4 * free, rel=1e-12)
    # 10 cm x 3 m, braced
    P = B.euler_critical_load(E, 0.1 ** 4 / 12, 0.01, 4e7, 3.0, B.K_HELD)
    assert P == pytest.approx(100e3, rel=0.01)


def test_johnson_branch_is_continuous_at_lambda_c():
    E, sc = 1.1e10, 4e7
    lc = math.pi * math.sqrt(2 * E / sc)
    lo = float(B.johnson_euler_stress(lc * (1 - 1e-9), E, sc))
    hi = float(B.johnson_euler_stress(lc * (1 + 1e-9), E, sc))
    assert lo == pytest.approx(hi, rel=1e-6)
    assert lo == pytest.approx(sc / 2, rel=1e-6)
    assert float(B.johnson_euler_stress(0.0, E, sc)) == pytest.approx(sc)
    s = B.johnson_euler_stress(np.linspace(1, 300, 400), E, sc)
    assert (np.diff(s) < 0).all()


def test_stack_of_cells_uses_the_parallel_axis_I():
    A, I = B.plan_section(np.ones(4), [(0, 0), (1, 0), (0, 1), (1, 1)], H5)
    assert A == pytest.approx(4 * H5 ** 2)
    assert I[0] == pytest.approx(0.1 ** 4 / 12, rel=1e-12)                   # a 10 cm square
    assert I[1] == pytest.approx(0.1 ** 4 / 12, rel=1e-12)
    assert I[0] > 15 * (H5 ** 4 / 12)                                        # not one cell's I
    A1, I1 = B.plan_section(np.ones(2), [(0, 0), (1, 0)], H5)               # 2 x 1: strong in x
    assert I1[0] > 3 * I1[1]


def _post_scene(top="held", brace=False, width=1, height=40, registered=False):
    nx = 12
    s = Scene(nx, 4, height + 4)
    s.put(0, nx, 0, 4, 0, 1, STONE)
    e = 7 if registered else 0
    s.put(1, 1 + width, 1, 1 + width, 1, height + 1, WOOD, 1.0, e)
    s.put(nx - 3, nx - 2, 1, 2, 1, height + 1, STONE)                         # the far post
    if top == "held":
        s.put(1, 1 + width, 1, 1 + width, height + 1, height + 2, STONE)
        s.put(nx - 3, nx - 2, 1, 2, height + 1, height + 2, STONE)
        s.put(1 + width, nx - 3, 1, 2, height + 1, height + 2, WOOD)          # a lintel on both posts
    if brace:
        s.put(1 + width, 2 + width, 1, 2, height // 2, height // 2 + 1, WOOD)
    return s


def _post_u(s, F, cell):
    W = np.zeros(s.mat.shape)
    W[cell] = F
    m, loads, res = s.run(W=W)
    col = [r for r in res.rows if r["kind"] == "column" and r["cell"][:2] == cell[:2]
           and r["mat"] == WOOD]
    return col[0], m


def test_post_buckles_at_euler_load_held_and_free_through_the_lattice():
    E, I, A = 1.1e10, 0.05 ** 4 / 12, 0.05 ** 2
    pc_held = math.pi ** 2 * E * I / 2.0 ** 2
    row, m = _post_u(_post_scene("held"), pc_held, (1, 1, 41))
    assert m.columns and any(c.held_top for c in m.columns)
    assert row["u"] == pytest.approx(1.0, rel=0.01) and row["mode"] == B.M_BUCKLE
    pc_free = math.pi ** 2 * E * I / (2 * 2.0) ** 2
    row, m = _post_u(_post_scene("free"), pc_free, (1, 1, 40))
    assert not any(c.held_top for c in m.columns if c.mat == WOOD)
    assert row["u"] == pytest.approx(1.0, rel=0.01) and row["mode"] == B.M_BUCKLE
    # crushing would take 100 kN; the post is slender, so buckling decides
    assert row["u_crush"] < 0.05


def test_post_10cm_by_3m_braced_takes_about_100kN():
    s = _post_scene("held", width=2, height=60, registered=True)
    E = 1.1e10
    pc = math.pi ** 2 * E * (0.1 ** 4 / 12) / 3.0 ** 2
    row, m = _post_u(s, pc, (1, 1, 61))
    assert len(m.columns[0 if m.columns[0].mat == WOOD else 1].xy) in (1, 4)
    wood = [c for c in m.columns if c.mat == WOOD][0]
    assert len(wood.xy) == 4                                                  # one 2 x 2 section
    assert row["u"] == pytest.approx(1.0, rel=0.02)
    assert 95e3 < pc < 105e3


def test_bracing_is_read_per_axis_and_the_weakest_governs():
    s = _post_scene("free", brace=True)
    F = 3000.0
    row, m = _post_u(s, F, (1, 1, 40))
    ub = row["u_buckle"]
    assert ub[1] > 2.5 * ub[0]                      # braced in x at mid height; free in y
    assert row["u"] == pytest.approx(ub[1], rel=1e-9)
    E, I = 1.1e10, 0.05 ** 4 / 12
    assert ub[1] == pytest.approx(F / (math.pi ** 2 * E * I / (2 * 2.0) ** 2), rel=0.01)


# ── masonry, arch, impact ───────────────────────────────────────────────────
def test_no_tension_section():
    N, d, b = 1000.0, 0.3, 0.2
    u = B.no_tension_util(N, N * d / 6, d, b, 1.0)
    assert float(u) == pytest.approx(2 * N / (b * d), rel=1e-12)
    assert float(B.no_tension_util(N, 0.0, d, b, 1.0)) == pytest.approx(N / (b * d))
    assert math.isinf(float(B.no_tension_util(N, N * d / 2, d, b, 1.0)))
    assert math.isinf(float(B.no_tension_util(0.0, 5.0, d, b, 1.0)))
    es = np.linspace(0.01, 0.49, 200) * d
    us = B.no_tension_util(N, N * es, d, b, 1.0)
    assert (np.diff(us) > 0).all()                 # continuous at d/6 and rising


def test_arch_thrust_and_pier():
    H, Nc, Ns = B.arch_thrust(3.0, 1.0, 20e3)
    assert H == pytest.approx(22.5e3, rel=1e-12) and Nc == H
    assert Ns == pytest.approx(math.hypot(22.5e3, 30e3))
    H2, _, _ = B.arch_thrust(3.0, 2.0, 20e3)
    assert H2 == pytest.approx(H / 2)
    us, uo = B.pier_check(H, 60e3, 1.0, 1.5, 0.7)
    assert us == pytest.approx(22.5e3 / (0.7 * 60e3))
    assert uo == pytest.approx((22.5e3 * 1.5 / 60e3) / 0.5)


def test_impact_factor_in_series():
    assert B.impact_factor(0.0, 1e-3, 0.0) == 2.0
    dst = 1e-3
    assert B.impact_factor(100.0, dst, 0.0) == pytest.approx(math.sqrt(2 * 100 / dst), rel=0.01)
    assert B.impact_factor(1.0, dst, 0.15) < B.impact_factor(1.0, dst, 0.0)


def test_80kg_person_stepping_off_1m_onto_three_5cm_strips():
    P = 80 * B.G
    L, E, b = 0.65, 1.1e10, 0.05
    I = b ** 4 / 12
    d_st = P * L ** 3 / (48 * E * 3 * I)               # three strips share
    Mcap = (b ** 3 / 6) * 7e7                          # one strip
    phi_series = B.impact_factor(1.0, d_st, B.D_BODY_PERSON)
    phi_rigid = B.impact_factor(1.0, d_st, 0.0)
    u_series = phi_series * P / 3 * L / 4 / Mcap
    u_rigid = phi_rigid * P / 3 * L / 4 / Mcap
    assert 4.0 < phi_series < 6.0                       # inside 3-8 body weights
    assert u_series < 0.2
    assert u_rigid > 1.0


# ── extraction ──────────────────────────────────────────────────────────────
def test_plank_on_two_posts_is_one_pinned_run_and_on_one_post_a_cantilever():
    m = plank_on_posts(12).members()
    runs = [r for r in m.runs]
    assert len(runs) == 1 and runs[0].kind == "pinned" and runs[0].nsup == 2
    assert runs[0].L == pytest.approx(12 * H5 + H5 / 2)
    assert np.allclose(runs[0].xc, (np.arange(12) + 0.75) * H5)
    s = Scene(20, 3, 6)
    s.put(0, 20, 0, 3, 0, 1, STONE)
    s.put(0, 2, 1, 2, 1, 4, STONE)
    s.put(2, 12, 1, 2, 3, 4, WOOD)                       # 10 cells out over air from the post
    m2, loads, res = s.run()
    r = m2.runs[0]
    assert r.kind == "cantilever" and tuple(r.cells[0]) == (2, 1, 3) and r.sup_lo == (1, 1)
    assert np.allclose(r.xc, (np.arange(10) + 0.5) * H5)
    # the critical section is the cell at the support face, with its own weight moment
    row = [x for x in res.rows if x["kind"] == "cantilever"][0]
    assert row["cell"] == (2, 1, 3)
    W = RHO[WOOD] * H5 ** 3 * B.G
    M0 = W * (np.arange(10) + 0.5).sum() * H5
    Sx = H5 ** 3 / 6
    assert row["u"] == pytest.approx(M0 / (Sx * 7e7), rel=1e-9)


def test_run_cells_load_the_posts_with_the_whole_weight():
    s = plank_on_posts(12)
    m, loads, res = s.run()
    assert conserved(loads)
    wp = RHO[WOOD] * H5 ** 3 * B.G
    assert loads.run_R[0].sum() == pytest.approx(12 * wp, rel=1e-12)
    assert loads.run_R[0][0, 0] == pytest.approx(loads.run_R[0][0, 1], rel=1e-12)
    # the reaction of the run lands on the post below it
    assert loads.N[0, 1, 1, 3] > 6 * wp                  # half the plank plus the post cell itself... at z=3


def test_the_critical_cell_of_a_midspan_block_is_under_the_block():
    s = plank_on_posts(20, f=0.5, nz=12)
    s.put(11, 13, 1, 2, 5, 6, LEAD)                      # 2 cells of lead near midspan
    m, loads, res = s.run()
    row = [x for x in res.rows if x["kind"] == "pinned"][0]
    assert row["cell"][0] in (11, 12) or row["cell"][0] in (10, 13)
    assert conserved(loads)
    # more block, more utilisation (monotone)
    s2 = plank_on_posts(20, f=0.5, nz=12)
    s2.put(11, 13, 1, 2, 5, 8, LEAD)
    r2 = [x for x in s2.run()[2].rows if x["kind"] == "pinned"][0]
    assert r2["u"] > row["u"]


def test_more_damage_means_more_utilisation():
    s = plank_on_posts(12, nz=10)
    s.put(6, 8, 1, 2, 5, 6, LEAD)
    u0 = [x for x in s.run()[2].rows if x["kind"] == "pinned"][0]["u"]
    s.dmg[:] = 0.2
    u1 = [x for x in s.run()[2].rows if x["kind"] == "pinned"][0]["u"]
    s.dmg[:] = 0.5
    u2 = [x for x in s.run()[2].rows if x["kind"] == "pinned"][0]["u"]
    assert u0 < u1 < u2
    assert u1 == pytest.approx(u0 / 0.8, rel=1e-9) and u2 == pytest.approx(u0 / 0.5, rel=1e-9)


def _two_planks(eid1, eid2, n=12):
    """Two planks, one on the other, on the same two posts (the posts reach both)."""
    s = plank_on_posts(n, zpost=3, eid=eid1, nz=9)
    s.put(0, 2, 1, 2, 5, 6, STONE)
    s.put(n + 2, n + 4, 1, 2, 5, 6, STONE)
    s.put(2, n + 2, 1, 2, 5, 6, WOOD, 1.0, eid2)
    return s


def test_extra_damage_combines_like_matdamage():
    s = plank_on_posts(12, nz=10)
    s.put(6, 8, 1, 2, 5, 6, LEAD)
    m, loads, res = s.run()
    u0 = [x for x in res.rows if x["kind"] == "pinned"][0]["u"]
    s.dmg[:] = 0.2
    m2 = B.with_extra_damage(s.members(), 0.375)             # 1 - 0.8 x 0.625 = 0.5
    u1 = [x for x in B.analyse(m2, REC, loads).rows if x["kind"] == "pinned"][0]["u"]
    assert u1 == pytest.approx(u0 / 0.5, rel=1e-9)


def test_the_local_combine_copy_equals_matdamage():
    rng = np.random.default_rng(1)
    D, dD = rng.uniform(-0.2, 1.2, 200), rng.uniform(-0.3, 1.1, 200)
    assert np.array_equal(B._combine_copy(D, dD), B.combine(D, dD))
    assert B._combine_copy(0.2, 0.3) == pytest.approx(1 - 0.8 * 0.7)
    assert B._combine_copy(0.99, 0.99) == 0.999


def test_two_layers_of_eid_zero_are_two_runs():
    m = _two_planks(0, 0).members()
    assert len(m.runs) == 2 and m.runs[1].stacked_on == 0
    assert m.runs[0].fills.shape[1] == 1 and m.runs[1].fills.shape[1] == 1


def test_two_layers_of_one_registered_eid_are_one_deeper_section():
    m = _two_planks(5, 5).members()
    runs = [r for r in m.runs if r.mat[0] == WOOD]
    assert len(runs) == 1 and runs[0].fills.shape == (12, 2)
    sec = B.stack_section(runs[0].fills, H5, runs[0].plate)
    one = B.section(np.ones(1), H5)
    assert sec["S"][0] / one["S"][0] == pytest.approx(4.0, rel=1e-12)


def test_stacked_runs_on_equal_supports_share_by_EI():
    # lower plank: fill 0.5 (eid 1); upper plank: full (eid 2); a lead block on the upper
    def build(fl, fu):
        s = Scene(20, 3, 9)
        s.put(0, 20, 0, 3, 0, 1, STONE)
        s.put(0, 2, 1, 2, 1, 6, STONE)
        s.put(14, 16, 1, 2, 1, 6, STONE)
        s.put(2, 14, 1, 2, 4, 5, WOOD, fl, 1)
        s.put(2, 14, 1, 2, 5, 6, WOOD, fu, 2)
        s.put(7, 9, 1, 2, 6, 9, LEAD)                     # a block on the upper plank
        return s
    s = build(0.5, 1.0)
    m, loads, res = s.run()
    assert len(m.runs) == 2 and conserved(loads)
    R, U = m.runs[0], m.runs[1]
    assert U.stacked_on == 0
    EIR, EIU = R.EI.mean(), U.EI.mean()
    assert EIU > 3 * EIR
    wl = RHO[WOOD] * H5 ** 3 * B.G
    qR = 12 * wl * 0.5
    B_ = 2 * 3 * RHO[LEAD] * H5 ** 3 * B.G
    qU = 12 * wl * 1.0 + B_
    p = (qU * EIR - qR * EIU) / (EIU + EIR)
    assert 0 < p < qU
    PR = loads.run_P[0].sum()
    PU = loads.run_P[1].sum()
    assert PR == pytest.approx(qR + p, rel=1e-9)
    assert PU == pytest.approx(qU - p, rel=1e-9)
    assert PR + PU == pytest.approx(qR + qU, rel=1e-12)
    # equal planks: the block is shared half and half
    s = build(1.0, 1.0)
    m, loads, _ = s.run()
    assert loads.run_P[0].sum() == pytest.approx(12 * wl + B_ / 2, rel=1e-9)
    assert loads.run_P[1].sum() == pytest.approx(12 * wl + B_ / 2, rel=1e-9)


# ── the member test (F10) ───────────────────────────────────────────────────
def test_people_leaves_ropes_ash_grip_bars_and_rubble_are_never_members():
    s = plank_on_posts(12, zpost=3, nz=14)
    # a standing person on the plank, with an arm held out sideways
    s.put(7, 8, 1, 2, 5, 12, FLESH)
    s.owned[7, 1, 5:12] = True
    s.put(8, 9, 1, 2, 10, 11, FLESH)                    # an arm (air below)
    s.owned[8, 1, 10] = True
    s.put(2, 5, 1, 2, 9, 10, LEAF, 0.5)                 # a leaf canopy in the air... on nothing
    s.put(4, 5, 2, 3, 4, 5, ROPE, 0.3)                 # a rope end hanging beside the plank
    s.put(2, 12, 2, 3, 1, 2, ASH)                        # an ash heap on the floor
    s.put(10, 11, 1, 2, 6, 9, IRON)                      # an iron bar held in a grip
    s.grip[10, 1, 6:9] = True
    s.put(0, 3, 0, 1, 1, 3, STONE)
    s.rubble[0:3, 0, 1:3] = True                         # a rubble heap of stone
    m = s.members()
    for mt in (FLESH, LEAF, ROPE, ASH):
        assert not m.member[s.mat == mt].any()
    assert not m.member[s.grip].any() and not m.member[s.rubble].any()
    cells = {tuple(c) for r in m.runs for c in r.cells}
    assert not cells & {tuple(c) for c in np.argwhere(~m.member & (s.mat > 0))}
    assert len(m.runs) == 1                              # only the plank


def test_a_persons_weight_reaches_the_posts_once():
    s = plank_on_posts(12, zpost=3, nz=14)
    wp = RHO[WOOD] * H5 ** 3 * B.G
    s.put(7, 8, 1, 2, 5, 12, FLESH)                       # body stood on the plank
    s.owned[7, 1, 5:12] = True
    s.put(8, 9, 1, 2, 10, 11, FLESH)                      # an arm: a cell with air below it
    s.owned[8, 1, 10] = True
    W = s.weights()
    body = W[s.mat == FLESH].sum()
    m, loads, res = s.run()
    assert conserved(loads) and loads.lost.sum() == 0
    assert loads.run_R[0].sum() == pytest.approx(12 * wp + body, rel=0, abs=1e-9)
    assert loads.ground.sum() == pytest.approx(W.sum(), abs=1e-9)
    # nothing about the person was analysed
    assert all(r["kind"] != "column" or r["mat"] != FLESH for r in res.rows)


def test_walking_extra_is_transient_and_reaches_the_posts():
    s = plank_on_posts(12)
    X = np.zeros(s.mat.shape)
    ex = B.walker_extra(80.0)
    X[7, 1, 4] = ex                                       # under the feet: the plank cell
    m, loads, res = s.run(X=X)
    assert loads.run_R[0][1].sum() == pytest.approx(ex, rel=1e-12)
    assert conserved(loads)
    row_t = [r for r in res.rows if r["kind"] == "pinned"][0]
    m1, l1, r1 = s.run(X=np.zeros_like(X))
    assert row_t["u"] > [r for r in r1.rows if r["kind"] == "pinned"][0]["u"]
    assert ex == pytest.approx(0.2 * 80 * B.G)


# ── arching over an opening ─────────────────────────────────────────────────
def _arch_scene(h, opening=1.0, wall_h=0.8, width=2.0, narrow=False):
    n = int(round(opening / h))
    ab = int(round(0.25 / h))
    nxw = 2 * ab + n
    ny = int(round(0.2 / h))
    zl = int(round(0.4 / h)) + 1
    top = zl + 1 + int(round(wall_h / h))
    s = Scene(nxw, ny, top, h)
    s.put(0, nxw, 0, ny, 0, 1, STONE)
    s.put(0, ab, 0, ny, 1, top, STONE)
    s.put(ab + n, nxw, 0, ny, 1, top, STONE)
    s.put(ab, ab + n, 0, ny, zl, zl + 1, STONE)               # the lintel
    if narrow:
        k = max(2, n // 4)
        s.put(ab + n // 2 - k // 2, ab + n // 2 + k // 2, 0, ny, zl + 1, zl + 4, STONE)
    else:
        s.put(0, nxw, 0, ny, zl + 1, top, STONE)              # the wall above the opening
    return s, n, ny, zl


def _wall_load_on_lintel(h):
    s, n, ny, zl = _arch_scene(h)
    m, loads, res = s.run()
    assert conserved(loads)
    P = sum(p.sum() for p in loads.run_P)
    own = n * ny * RHO[STONE] * h ** 3 * B.G
    return (P - own), n, ny, h, loads


def test_arching_load_in_metres_is_the_same_at_5_and_10_cm_cells():
    a, n1, ny1, h1, l1 = _wall_load_on_lintel(0.05)
    b, n2, ny2, h2, l2 = _wall_load_on_lintel(0.10)
    thick = 0.2
    exact = RHO[STONE] * B.G * (1.0 * 0.5 / 2) * thick       # the 45 degree triangle: L^2/4
    assert a == pytest.approx(exact, rel=0.1) and b == pytest.approx(exact, rel=0.1)
    assert abs(a - b) / b < 0.1
    # without arching the whole wall above would load it
    s, n, ny, zl = _arch_scene(0.05)
    m, loads, _ = s.run(arch=False)
    full = sum(p.sum() for p in loads.run_P) - n * ny * RHO[STONE] * 0.05 ** 3 * B.G
    assert full > 3 * a


def test_a_block_narrower_than_the_opening_is_not_a_wall():
    s, n, ny, zl = _arch_scene(0.05, narrow=True)
    m, loads, _ = s.run()
    assert conserved(loads)
    ab = int(round(0.25 / 0.05))
    block = int((s.mat[ab:ab + n, :, zl + 1:] > 0).sum()) * RHO[STONE] * 0.05 ** 3 * B.G
    own = n * ny * RHO[STONE] * 0.05 ** 3 * B.G
    P = sum(p.sum() for p in loads.run_P)
    assert P == pytest.approx(own + block, rel=1e-9)         # the full load, no arching


def test_reactions_sum_to_the_total_weight_with_arching_and_a_block_on_a_plank():
    s, n, ny, zl = _arch_scene(0.05)
    m, loads, _ = s.run()
    assert np.allclose(loads.ground.sum(), loads.total.sum(), rtol=1e-12)
    p = plank_on_posts(14)
    p.put(5, 10, 1, 2, 5, 8, STONE)
    m, loads, _ = p.run()
    W = p.weights()
    assert loads.ground.sum() == pytest.approx(W.sum(), abs=1e-9) and loads.lost.sum() == 0


def test_a_flat_masonry_lintel_without_tension_fails_at_once():
    s, n, ny, zl = _arch_scene(0.05)
    m, loads, _ = s.run()
    with_t = B.analyse(m, REC, loads)
    no_t = B.analyse(m, REC, loads, no_tension=True)
    assert max(r["u"] for r in no_t.rows if r["kind"] == "pinned") == math.inf
    assert max(r["u"] for r in with_t.rows if r["kind"] == "pinned") < math.inf


def test_plates_are_read_from_the_lanes_and_bars_from_a_single_lane():
    one = plank_on_posts(10, lanes=1).members()
    assert not one.runs[0].plate.any()
    many = plank_on_posts(10, lanes=4).members()
    assert len(many.runs) == 4 and all(r.plate.all() for r in many.runs)
    assert len(many.lane_groups) == 1 and len(many.lane_groups[0]) == 4


def test_a_foot_on_a_wide_plank_is_spread_over_the_lanes():
    s = plank_on_posts(32, lanes=8)
    X = np.zeros(s.mat.shape)
    X[18, 4, 4] = 700.0                                   # one foot cell, mid span
    m, loads, _ = s.run(X=X)
    peak_spread = max(p[1].max() for p in loads.run_P)
    m2, l2, _ = s.run(X=X, spread=False)
    peak_point = max(p[1].max() for p in l2.run_P)
    assert peak_point == pytest.approx(700.0)
    assert peak_spread < peak_point / 3
    assert sum(p[1].sum() for p in loads.run_P) == pytest.approx(700.0, rel=1e-12)


def test_rubble_hanging_cells_go_to_the_old_law_not_the_members_law():
    s = plank_on_posts(12)
    s.rubble[8, 1, 4] = True                             # a piece in the plank: no longer a member
    m = s.members()
    assert (8, 1, 4) not in {tuple(c) for r in m.runs for c in r.cells}
    assert m.orphan[8, 1, 4]
    assert m.runs and all(r.kind == "cantilever" for r in m.runs)     # the stubs
    m2, loads, _ = s.run()
    assert conserved(loads)


def _bridge_scene(seed):
    """A random bridge: posts, a slab over them (maybe supported on all four sides),
    maybe a second slab, maybe a stone wall, blocks and a person on top."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(6, 16))
    lanes = int(rng.integers(1, 5))
    zp = int(rng.integers(2, 5))
    s = Scene(n + 6, lanes + 4, zp + 14)
    s.put(0, n + 6, 0, lanes + 4, 0, 1, STONE)
    top = zp + 1
    ring = rng.random() < 0.25
    for x0, x1 in ((0, 2), (n + 4, n + 6)):
        s.put(x0, x1, 1, lanes + 3, 1, top + 3, STONE)
    if ring:
        s.put(2, n + 4, 0, 2, 1, top + 3, STONE)
        s.put(2, n + 4, lanes + 2, lanes + 4, 1, top + 3, STONE)
    mats = [WOOD, STONE, IRON, GLASS, WOOD, LEAD]
    m1 = int(rng.choice(mats))
    e1 = int(rng.choice([0, 1]))
    f1 = float(rng.choice([1.0, 0.6, 0.35]))
    s.put(2, n + 4, 2, lanes + 2, top, top + 1, m1, f1, e1)
    if rng.random() < 0.45:
        e2 = e1 if rng.random() < 0.5 else int(rng.choice([0, 1, 2]))
        m2 = m1 if rng.random() < 0.5 else int(rng.choice(mats))
        s.put(2, n + 4, 2, lanes + 2, top + 1, top + 2, m2, float(rng.choice([1.0, 0.6])), e2)
    if rng.random() < 0.4:
        s.put(0, n + 6, 1, lanes + 3, top + 2, top + 2 + int(rng.integers(1, 7)), STONE)
    for _ in range(int(rng.integers(0, 3))):
        x0 = int(rng.integers(2, n + 2))
        s.put(x0, x0 + int(rng.integers(1, 4)), 2, lanes + 2, top + 4, top + 4 + int(rng.integers(1, 5)),
              int(rng.choice([STONE, LEAD, IRON])))
    if rng.random() < 0.5:
        x0 = int(rng.integers(2, n + 2))
        s.put(x0, x0 + 1, 2, 3, top + 8, top + 13, FLESH)
        s.owned[x0, 2, top + 8:top + 13] = True
        if x0 + 1 < n + 4:
            s.put(x0 + 1, x0 + 2, 2, 3, top + 11, top + 12, FLESH)
            s.owned[x0 + 1, 2, top + 11] = True
    s.rubble = (rng.random(s.mat.shape) < 0.03) & (s.mat > 0)
    return s


def test_the_load_path_conserves_weight_on_random_bridges():
    """Every newton put in comes out at the floor, the window edge or the 'lost'
    ledger (cells held by nothing, which the old law owns), whatever the mix of
    posts, slabs, stacked slabs, arching walls, blocks and people."""
    seen = dict(stacked=0, arch=0, two_way=0, pinned=0, cantilever=0, stack2=0, lost=0)
    for seed in range(120):
        s = _bridge_scene(seed)
        rng = np.random.default_rng(1000 + seed)
        X = np.where((s.mat > 0) & (rng.random(s.mat.shape) < 0.05), rng.uniform(0, 800, s.mat.shape), 0.0)
        m = s.members()
        seen["stacked"] += any(r.stacked_on >= 0 for r in m.runs)
        seen["two_way"] += any(r.two_way for r in m.runs)
        seen["pinned"] += any(r.kind == "pinned" for r in m.runs)
        seen["cantilever"] += any(r.kind == "cantilever" for r in m.runs)
        seen["stack2"] += any(r.fills.shape[1] > 1 for r in m.runs)
        seen["arch"] += bool(B._arch_plan(m))
        for arch in (True, False):
            loads = B.accumulate_loads(m, s.weights(), X, arch=arch)
            assert conserved(loads), (seed, arch)
            assert (loads.N >= -1e-12).all()
            seen["lost"] += bool(loads.lost.sum() > 0)
        res = B.analyse(m, REC, loads)
        assert not np.isnan(res.u).any()
    assert seen["stacked"] >= 3 and seen["arch"] >= 3 and seen["two_way"] >= 1, seen
    assert seen["pinned"] >= 30 and seen["cantilever"] >= 1 and seen["stack2"] >= 3, seen


# ── [C] code against its own derate table ───────────────────────────────────
def _steel_girder(T):
    s = Scene(18, 3, 8)
    s.put(0, 18, 0, 3, 0, 1, STONE)
    s.put(0, 3, 1, 2, 1, 6, STONE)
    s.put(15, 18, 1, 2, 1, 6, STONE)
    s.put(3, 15, 1, 2, 5, 6, IRON)
    s.T[3:15, 1, 5] = T
    return s


def _girder_u(T, F):
    s = _steel_girder(T)
    W = np.zeros(s.mat.shape)
    W[8, 1, 5] = F
    _, _, res = s.run(W=W)
    return [r for r in res.rows if r["kind"] == "pinned"][0]["u"]


def test_codeonly_steel_girder_reaches_u1_at_the_derate_table_temperatures():
    u1 = _girder_u(20.0, 1000.0)
    for u0, want in ((0.5, 591.0), (0.3, 671.0)):
        F = 1000.0 * u0 / u1
        assert _girder_u(20.0, F) == pytest.approx(u0, rel=1e-9)
        lo, hi = 20.0, 1190.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if _girder_u(mid, F) >= 1.0:
                hi = mid
            else:
                lo = mid
        assert abs(hi - want) <= 10.0, (u0, hi)


def test_u_steel_derate_rows_equal_the_sourced_EN_1993_table_when_present():
    """A-S sourced the EN 1993-1-2 retention factors. The stand-in rows used by
    the [C] test above must equal them (k_y for strength, k_E for stiffness), which
    makes the 591 / 671 C results independent of how the table was typed."""
    import json
    import os
    try:
        d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                                        "sourced_numbers.json")))
        e = d["entries"]["steel.strength_retention_EN1993-1-2"]
    except (OSError, ValueError, KeyError):
        pytest.skip("no sourced EN 1993-1-2 table in tests/data/sourced_numbers.json")
    assert e["status"] == "sourced"
    ours = {T: (ks, kE) for T, ks, kE in _IRON_D}
    for T, ky, kE in zip(e["value"]["T_C"], e["value"]["k_y_effective_yield"], e["value"]["k_E_modulus"]):
        if T in ours:
            assert ours[T] == (pytest.approx(ky), pytest.approx(kE)), T


# ── the arch recogniser ─────────────────────────────────────────────────────
def _arch_slice(kind, r=10):
    """An x-z slice: solid everywhere except an opening of width 2 r, vertical
    jambs 4 cells high, then a round arch or a pointed one."""
    nx, nz = 2 * r + 12, r + 18
    m = np.ones((nx, nz), bool)
    zs = 4
    for x in range(nx):
        for z in range(nz):
            d = x + 0.5 - nx / 2
            if abs(d) < r and (z <= zs or ((z - zs) ** 2 + d * d < r * r if kind == "round"
                                           else (z - zs) < (r * 1.6) * (1 - abs(d) / r))):
                m[x, z] = False
    return m, zs


def test_profile_from_slice_reads_span_rise_and_thickness():
    for kind, rise in (("round", 10), ("pointed", 16)):
        m, zs = _arch_slice(kind)
        out = B.profile_from_slice(m, 0.1)
        assert out is not None
        span, rs, thick = out
        assert span == pytest.approx(2 * 10 * 0.1, rel=0.001)
        assert 0.6 * rise * 0.1 <= rs <= 1.05 * rise * 0.1, (kind, rs)
    lintel = np.ones((30, 12), bool)
    lintel[8:20, 0:4] = False
    span, rs, thick = B.profile_from_slice(lintel, 1.0)
    assert span == 12 and rs == 0 and thick == 8
    assert B.profile_from_slice(np.ones((10, 10), bool)) is None
