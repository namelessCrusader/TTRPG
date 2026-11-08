"""What the particle solver must not get wrong.

Every number here was measured before it was asserted, and the ones with a
tolerance have the tolerance because the measurement moved, not because a
round number looked tidy.
"""

import numpy as np
import pytest

from src.mpm import MPM, kernels
import src.mpm.core as core


def _block(stiff=1e-4, material="wood", n=24):
    w = MPM((n, n, n), dx=0.1, stiff=stiff)
    # A REAL FLOOR. This fixture had none, and the block came to rest on the
    # domain clamp two cells up — the same invisible, frictionless floor that
    # later held a felled pole ten centimetres off the ground. The test passed
    # on the bug. `floor_top = 0.2` below is now a surface, not an edge.
    w.add_wall((0.0, 0.0, 0.0), (n * 0.1, n * 0.1, 0.2))
    w.add_box((0.8, 0.8, 1.4), (1.2, 1.2, 1.8), material)
    return w


def test_a_BLOCK_IN_FREE_FALL_falls_at_exactly_g():
    """The one test that catches a damping term applied to rigid motion.

    Viscous stress reads the strain RATE, and a body that is merely falling has
    none — so gravity must arrive undiluted. Written because the obvious way to
    stop a block bouncing is to damp its velocity, which also makes it fall
    slowly, and nothing else here would have noticed."""
    w = _block()
    z0 = w.x[:, 2].mean()
    for _ in range(9):
        w.step(0.025)
    assert w.x[:, 2].mean() == pytest.approx(z0 - 0.5 * 9.81 * w.t ** 2, abs=5e-3)


def test_MASS_IS_A_CLOSED_LEDGER_through_an_impact():
    w = _block()
    m0 = w.total_mass()
    for _ in range(60):
        w.step(0.025)
    assert w.total_mass() == pytest.approx(m0, rel=1e-12)
    assert np.isfinite(w.x).all()


def test_a_DROPPED_BLOCK_COMES_TO_REST_on_the_floor_and_stays():
    """Pure elasticity cannot lose energy. Measured with none, the block bounced
    HIGHER the stiffer it was made and was still in the air after a second and
    a half — so matter here has internal friction, and this is what says so."""
    w = _block()
    zs = []
    for _ in range(80):
        w.step(0.025)
        zs.append(w.x[:, 2].mean())
    floor_top, half = 0.2, 0.2
    assert zs[-1] == pytest.approx(floor_top + half, abs=0.03)


def test_DAMPING_DECIDES_HOW_MUCH_A_THING_BOUNCES_and_nothing_else_does():
    """Restitution must fall as damping rises, or `damp` is not a property of
    the material, it is a knob that happens to change the answer."""
    out = []
    keep = core.MAT["wood"]
    try:
        for damp in (0.3, 1.0, 3.0):
            core.MAT["wood"] = keep[:4] + (damp,) + keep[5:]
            w = _block()
            zs = []
            for _ in range(80):
                w.step(0.025)
                zs.append(w.x[:, 2].mean())
            zs = np.array(zs)
            land = int(np.argmin(zs[:30]))
            out.append(np.sqrt(max(zs[land:].max() - zs[land], 0.0)
                               / (zs[0] - zs[land])))
    finally:
        core.MAT["wood"] = keep
    assert out[0] > out[1] > out[2], out
    assert out[2] < 0.35, out


def test_a_VISCOUS_STEP_HAS_ITS_OWN_LIMIT_and_cfl_knows_it():
    """Damping is a diffusion, and an explicit diffusion has a stability limit
    that has nothing to do with the wave speed. Without this the deformation
    gradient went to infinity in a dozen substeps and the SVD stopped
    converging — a blow-up that read as a bad material and was a bad step."""
    keep = core.MAT["wood"]
    try:
        core.MAT["wood"] = keep[:4] + (8.0,) + keep[5:]
        w = _block()
        rho = w.m / w.vol0
        assert w.cfl_dt() <= 0.25 * w.dx ** 2 / np.max(w.eta / rho) + 1e-18
        for _ in range(20):
            w.step(0.025)
        assert np.isfinite(w.x).all() and np.isfinite(w.F).all()
    finally:
        core.MAT["wood"] = keep


def test_A_TABLE_HOLDS_THINGS_UP_IT_DOES_NOT_HOLD_ON_TO_THEM():
    """A plank with most of itself off a table must TIP. Zeroing the velocity
    at every wall node instead glues what touches it, and the measured symptom
    was a plank bobbing between 4 and 19 degrees for three seconds: a clamped
    cantilever oscillates, a resting one falls off."""
    w = MPM((32, 14, 26), dx=0.1, stiff=1e-4)
    w.add_wall((0, 0, 0), (3.2, 1.4, 0.2))
    w.add_wall((0.3, 0.3, 0.2), (1.3, 1.1, 1.6))
    w.add_box((0.95, 0.5, 1.6), (1.95, 0.9, 1.8), "wood")
    for _ in range(60):
        w.step(0.025)
    assert w.x[:, 2].mean() < 1.0, "it never left the table"


def test_OVERSTEPPING_IS_AN_ERROR_not_a_budget():
    """Clipping the substep count does not make the sim cheaper, it makes it
    wrong — so hitting the cap says so instead of quietly detonating."""
    w = _block(stiff=1e-2)
    with pytest.raises(RuntimeError, match="substeps"):
        w.step(0.025, max_sub=4)


def test_A_PANE_BREAKS_WHERE_IT_IS_BENT_and_nowhere_else():
    """Damage must LOCALISE. That is the whole difference between breaking and
    dissolving, and the first version of this model did the latter: 474 of 480
    pieces cracked at once and the slab poured off the table like sand,
    because a critical-STRAIN threshold fires everywhere a shock reaches.

    Read off the stress instead, and with strength varying over a smoothed
    flaw field, the crack goes where the bending is: measured at 67% broken in
    the 10 cm at the table edge and 0% across the other 90 cm of the slab.

    Known limit, deliberately not asserted: the crack ARRESTS rather than
    running through — 2 of 92 columns sever. Per-particle damage has no
    crack tip and so no 1/sqrt(r) concentration to drive one. Refining the
    discretisation threefold moved severance from 2.2% to 3.1%, which is what
    says it is the model and not the resolution. The fix is a non-local
    damage term (CD-MPM, Wolper et al. 2019)."""
    from src.mpm.scenes import glass_edge
    w = glass_edge()
    x0 = w.x[:, 0].copy()
    for _ in range(90):
        w.step(0.025)
    d = w.dmg
    assert d.sum() > 0, "it did not break at all"
    edge = 1.5
    near = np.abs(x0 - edge) <= 0.2
    assert d[near].sum() >= 0.8 * d.sum(), (
        f"damage is not at the edge: {int(d[near].sum())} of {int(d.sum())}")
    far = np.abs(x0 - edge) > 0.4
    # One stray far crack is the elastic bounce (e = 0.26 now that the grid
    # supports elastically) meeting the weakest flaw; measured 1 of 56.
    assert d[far].sum() <= 0.05 * d.sum(), f"{int(d[far].sum())} broke far from the bend"
    assert w.total_mass() == pytest.approx(150.0, rel=1e-9)


def test_A_TABLE_DOES_NOT_SHOVE_THINGS_SIDEWAYS():
    """A slab dropped straight down onto a table, landing within a cell of
    its edge, must not acquire sideways speed. The node at a table's corner
    has two faces; holding both handed a 45-degree normal to everything
    near it, including matter resting on the flat top, and turned a vertical
    6 m/s landing into 1.1 m/s of drift. Now a flat face is held by the
    grid and a corner refuses only the directions its own particles were
    stopped in, so the top reads `up` right out to the edge.

    No overhang here on purpose. A slab that straddles the edge bends over
    it after the bounce and drags its compressed underside across the top,
    and that friction is a real force, sized by how soft the slab is (a
    third as much at stiff=1e-2 as at 1e-3). This test is about the
    collider, not the bending."""
    w = MPM((32, 14, 40), dx=0.1, stiff=1e-3)
    w.add_wall((0.0, 0.0, 0.0), (3.2, 1.4, 0.2))
    w.add_wall((0.3, 0.3, 0.2), (1.5, 1.1, 1.2))         # table, edge at x=1.5
    w.add_box((0.5, 0.5, 3.2), (1.45, 0.9, 3.35), "glass")  # ends 5 cm short of the edge
    for _ in range(24):
        w.step(0.025)
    before = abs(float(w.v[:, 0].mean()))
    worst = 0.0
    for _ in range(20):
        w.step(0.025)
        worst = max(worst, abs(float(w.v[:, 0].mean())))
    assert before < 0.05, f"drifting before it even lands: {before}"
    # Measured 0.21 m/s: the strip nearest the edge has less table under it
    # than the rest, so the impact bulge's friction does not quite cancel.
    # A fifth of the 1.12 the corner node gave, and it stays on the table.
    assert worst < 0.25, f"the table shoved it sideways at {worst} m/s"
    assert (w.x[:, 0] < 1.5).mean() > 0.95, "and it is still on the table"


def test_A_RESTING_PANE_DOES_NOT_BREAK_UNDER_ITS_OWN_WEIGHT():
    """The other half of choosing a threshold, and the half a shatter test on
    its own would never catch."""
    from src.mpm.scenes import resting_slab
    w = resting_slab()
    for _ in range(40):
        w.step(0.025)
    assert w.broken() == 0, w.broken()


def test_WHERE_A_THING_IS_DOES_NOT_CHANGE_WHAT_IT_IS_MADE_OF():
    """Flaws are a property of the OBJECT, not of the space it occupies.

    Indexed by world position, the same slab dropped from a greater height
    samples a different part of the flaw field and is made of stronger glass.
    That inverted a whole experiment: a slab dropped from 3.2 m broke LESS
    than one dropped from 2.0 m, which is not possible. It was not a crumple
    zone or a shielding effect — both of which were available and plausible —
    it was 15% stronger glass. Found only because the strength was printed
    beside the result instead of the result alone."""
    a = MPM((32, 14, 50), dx=0.1, stiff=1e-3)
    a.add_box((1.0, 0.5, 2.0), (2.0, 0.9, 2.15), "glass")
    b = MPM((32, 14, 50), dx=0.1, stiff=1e-3)
    b.add_box((1.0, 0.5, 4.4), (2.0, 0.9, 4.55), "glass")
    assert a.strength.mean() == pytest.approx(b.strength.mean(), rel=1e-12)
    assert sorted(a.strength) == pytest.approx(sorted(b.strength), rel=1e-12)


def test_A_HARDER_FALL_BREAKS_MORE_OF_IT():
    """Impact stress is rho c v, so it rises with the speed of arrival and
    more of the material passes its strength. The first version of this was
    NON-MONOTONIC and that is what uncovered the flaw-anchoring bug — so this
    is the regression test for the bug AND the evidence the stress criterion
    responds to the world rather than to a fitted constant."""
    out = []
    for ztop in (2.0, 3.2, 4.4):
        w = MPM((32, 14, 50), dx=0.1, stiff=1e-3)
        w.add_wall((0, 0, 0), (3.2, 1.4, 0.2))
        w.add_wall((0.3, 0.3, 0.2), (1.5, 1.1, 1.2))
        w.add_box((1.0, 0.5, ztop), (2.0, 0.9, ztop + 0.15), "glass")
        for _ in range(70):
            w.step(0.025)
        out.append(w.broken())
    # Measured 8, 89, 83 for 4.0, 6.3 and 7.9 m/s. The rise is an order of
    # magnitude; the top two are within noise of each other because the
    # hinge is only so wide — at 7.9 m/s the damage moved INTO it (81 of 83
    # within 20 cm of the edge, against 39 of 89) rather than growing.
    assert out[0] < 0.3 * out[1] and out[0] < 0.3 * out[2], out
    assert out[2] > 0.8 * out[1], out


# ------------------------------------------------------------ exact speed
#
# The solver is solved over a window and its loops are compiled, and neither
# is allowed to change one bit of the answer. Each test below runs two copies
# of a world side by side and compares every particle field after EVERY
# substep, not at the end: a difference that a bounce happens to hide at the
# end is still a difference.

FIELDS = ("x", "v", "C", "F", "Jp", "dmg", "cid", "strength")


def _glass():
    """glass_edge, started 5 cm above the table at its 6.3 m/s arrival
    speed, so it breaks within a few hundred substeps instead of a few
    thousand."""
    from src.mpm.scenes import glass_edge
    w = glass_edge()
    w.x[:, 2] -= 3.2 - 1.25
    w.v[:, 2] = -6.3
    return w


def _bound_glass():
    """The same pane as ONE cluster, so `_hold_shape` runs and bonds part."""
    w = _glass()
    w.bind(0)
    return w


def _plank():
    from src.mpm.scenes import plank
    return plank()


CASES = {"block": _block, "plank": _plank, "glass": _glass,
         "bound glass": _bound_glass}


def _lockstep(make, n, first=None, **flags):
    """Two copies of a world, the first with `first` set and the second
    with `flags`, `n` substeps side by side. Returns (first substep that
    differs or None, the field, the first copy)."""
    a, b = make(), make()
    for k, v in (first or {}).items():
        setattr(a, k, v)
    for k, v in flags.items():
        setattr(b, k, v)
    dt = a.cfl_dt()
    for s in range(n):
        a.substep(dt)
        b.substep(dt)
        for f in FIELDS:
            if not np.array_equal(getattr(a, f), getattr(b, f)):
                return s, f, a
    return None, None, a


@pytest.mark.parametrize("case", list(CASES))
def test_ACTIVE_WINDOW_changes_NOTHING(case):
    """Solving over the nodes the particles reach, instead of the whole
    grid, must give the same bits: the same nodes get the same sums in the
    same order, and a node outside the window has no mass to move. The pane
    has to BREAK in here, because cracks are where a near miss would show."""
    bad, field, w = _lockstep(CASES[case], 600, window=False)
    assert bad is None, f"{case}: {field} differs at substep {bad}"
    if "glass" in case:
        assert w.broken() >= 1, "the pane never broke, so cracking was not tested"


@pytest.mark.skipif(not kernels.HAVE_NUMBA, reason="numba is not installed")
@pytest.mark.parametrize("case", list(CASES))
def test_the_COMPILED_SOLVER_and_the_PLAIN_ONE_agree(case):
    """The numpy loops are the definition of the solver; the numba kernels
    must add the same numbers in the same order. This is the test that
    caught einsum: it sums a 3-vector as (a0 d0 + a2 d2) + a1 d1, and a
    kernel that summed left to right was 1e-15 m out after one substep."""
    bad, field, w = _lockstep(CASES[case], 300, {"fused": False}, compiled=False)
    assert bad is None, f"{case}: {field} differs at substep {bad}"


@pytest.mark.skipif(not kernels.HAVE_NUMBA, reason="numba is not installed")
@pytest.mark.parametrize("case", list(CASES))
def test_the_FUSED_SUBSTEP_is_the_PLAIN_ONE_bit_for_bit(case, monkeypatch):
    """The whole substep in one compiled call (`kernels.substep`, P10)
    against the numpy definition, after every substep: the stress with its
    fixed-sweep SVD and eigenvalue, the grid, the cracks, the bonds parting
    and contact with the table. It holds only while numba neither reorders
    a sum nor fuses a multiply and an add, so this is the guard on both."""
    calls = []
    fused = kernels.substep

    def counted(*a):
        calls.append(1)
        return fused(*a)
    monkeypatch.setattr(kernels, "substep", counted)
    bad, field, w = _lockstep(CASES[case], 300, compiled=False)
    assert bad is None, f"{case}: {field} differs at substep {bad}"
    assert len(calls) == 300, "the fused substep was not the one that ran"
    if "glass" in case:
        assert w.broken() >= 1, "the pane never broke, so cracking was not tested"
    if case == "bound glass":
        assert (w.cid < 0).any(), "no bond parted, so parting was not tested"


def _awkward_matrices():
    """Deformation gradients as the solver meets them (a turn times a small
    stretch), large ones, and the cases a fixed recipe must still get right:
    nothing at all, a line, a plane, a mirror, equal stretches, and a
    matrix that is diagonal already."""
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.standard_normal((150, 3, 3)))
    F = q @ (np.eye(3) + 0.05 * rng.standard_normal((150, 3, 3)))
    big = rng.standard_normal((50, 3, 3)) * 3.0
    odd = [np.zeros((3, 3)), np.outer([1.0, 2.0, 3.0], [1.0, 0.0, 1.0]),
           np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 0]]), np.diag([1.0, 1.0, -1.0]),
           np.eye(3), q[0], np.diag([0.9, 1.1, 1.0]), 2.0 * q[1],
           np.array([[1.0, 1e-9, 0], [0, 1.0, 0], [0, 0, 1.0]])]
    return np.concatenate([F, big, np.array(odd)])


def test_the_FIXED_SWEEP_ALGEBRA_is_LAPACKS_to_rounding():
    """The solver's SVD, top eigenvalue and 3x3 solve are fixed recipes
    (core._svd3, _top_eig3, _solve3), so that a compiled loop can repeat
    them exactly. They are a redefinition of LAPACK's, and must give its
    answers: measured on every scene in this file they agreed to 6e-15."""
    F = _awkward_matrices()
    U, S, V = core._svd3(F)
    Ul, Sl, Vtl = np.linalg.svd(F)
    assert np.abs(S - Sl).max() < 1e-13 * max(1.0, Sl.max())
    rec = core._mm_t(U * S[:, None, :], V)
    assert np.abs(rec - F).max() < 1e-13 * max(1.0, np.abs(F).max())
    for M in (U, V):
        assert np.abs(np.einsum("nki,nkj->nij", M, M) - np.eye(3)).max() < 1e-13
    full = Sl[:, 2] > 1e-6 * Sl[:, 0]       # the turn is unique only there
    assert np.abs(core._mm_t(U, V) - Ul @ Vtl)[full].max() < 1e-12
    A = F + np.swapaxes(F, 1, 2)
    top = np.linalg.eigvalsh(A)[:, -1]
    assert np.abs(core._top_eig3(A) - top).max() < 1e-13 * max(1.0, np.abs(top).max())
    I = np.einsum("nij,nkj->nik", F, F) + np.eye(3)
    b = np.random.default_rng(4).standard_normal((len(F), 3))
    want = np.linalg.solve(I, b[..., None])[..., 0]
    assert np.abs(core._solve3(I, b) - want).max() < 1e-12


@pytest.mark.skipif(not kernels.HAVE_NUMBA, reason="numba is not installed")
def test_the_COMPILED_ALGEBRA_is_the_NUMPY_ALGEBRA_bit_for_bit():
    """Matrix by matrix, the kernel's SVD and top eigenvalue are the numpy
    recipe's bits, the odd cases included (a matrix of zeros, a line, a
    plane, a mirror), whose branches the scenes above may never take."""
    F = _awkward_matrices()
    n = len(F)
    U, S, V = core._svd3(F)
    Bs = np.ascontiguousarray(F.reshape(n, 9).T)
    Vs = np.zeros((9, n))
    Vs[[0, 4, 8]] = 1.0
    kernels._sweeps(Bs, Vs, core.SVD_SWEEPS)
    B, Vk, Uk, Sk = np.empty((3, 3)), np.empty((3, 3)), np.empty((3, 3)), np.empty(3)
    for k in range(n):
        kernels._finish_svd(Bs[:, k], Vs[:, k], B, Vk, Sk, Uk)
        assert np.array_equal(Sk, S[k]) and np.array_equal(Uk, U[k]) \
            and np.array_equal(Vk, V[k]), k
    A = F + np.swapaxes(F, 1, 2)
    D = np.stack([A[:, 0, 0], A[:, 1, 1], A[:, 2, 2], A[:, 0, 1], A[:, 0, 2], A[:, 1, 2]])
    assert np.array_equal(kernels._eig_sweeps(D, core.EIG_SWEEPS), core._top_eig3(A))


def _apart():
    """Two things falling at opposite corners of a room, at 4 m/s: a bound
    wood block onto the floor, and a glass block onto a shelf 1.2 m up.
    Far enough apart for the default ISLAND_NODES to split them."""
    w = MPM((64, 64, 30), dx=0.1)
    w.add_wall((0.0, 0.0, 0.0), (6.4, 6.4, 0.2))
    w.add_wall((5.4, 5.4, 0.2), (6.4, 6.4, 1.2))
    w.add_box((0.6, 0.6, 0.5), (1.0, 1.0, 0.9), "wood", seed=1)
    w.bind(0)
    w.add_box((5.6, 5.6, 1.25), (6.0, 6.0, 1.45), "glass", seed=2)
    w.v[:, 2] = -4.0
    return w


def _parting():
    """Two wood blocks two cells apart, thrown away from each other at 8 m/s
    each: one island at first, two once a gap opens, and both end against
    the edges of the world. Split as soon as they share no node at all
    (ISLAND_NODES = 0), which puts the two windows a node apart: the
    tightest case for exactness."""
    w = MPM((48, 16, 20), dx=0.1)
    w.add_wall((0.0, 0.0, 0.0), (4.8, 1.6, 0.2))
    n = w.add_box((1.8, 0.6, 0.3), (2.2, 1.0, 0.7), "wood", seed=1)
    w.add_box((2.4, 0.6, 0.3), (2.8, 1.0, 0.7), "wood", seed=2)
    w.bind(n)
    w.v[:n, 0] = -8.0
    w.v[n:, 0] = 8.0
    return w


def _buried():
    """A wood block buried 1.6 m deep in a 3 m slab, deeper than its own
    island's distance-field window reaches, and a stone block far off and
    higher up, which stretches the one window to the slab's top. Split as
    soon as they share no node (ISLAND_NODES = 0). Before the deep particles
    read the one window (`MPM._contact`), the buried block was pushed out by
    another distance with islands than without, from the first substep."""
    w = MPM((120, 20, 44), dx=0.1)
    w.add_wall((0.0, 0.0, 0.0), (12.0, 2.0, 0.2))
    w.add_wall((0.0, 0.0, 0.2), (6.0, 2.0, 3.0))
    w.add_box((2.8, 0.8, 1.0), (3.2, 1.2, 1.4), "wood", seed=1)
    w.add_box((10.5, 0.8, 3.5), (10.9, 1.2, 3.9), "stone", seed=2)
    return w


@pytest.mark.parametrize("compiled", [True, False])
@pytest.mark.parametrize("case", ["apart", "parting", "buried"])
def test_ISLANDS_change_NOTHING(case, compiled, monkeypatch):
    """Things far apart are solved on separate windows (`MPM._islands`),
    and that must give the same bits as one window over all of them,
    compared after every substep: through the landing, the crack, a bond
    holding its shape, the world's edge, the substep where one island
    becomes two, and a particle buried deeper than its island's window."""
    if compiled and not kernels.HAVE_NUMBA:
        pytest.skip("numba is not installed")
    if case in ("parting", "buried"):
        monkeypatch.setattr(core, "ISLAND_NODES", 0)
    make = {"apart": _apart, "parting": _parting, "buried": _buried}[case]
    one, isl = make(), make()
    one.islands = False
    one.compiled = isl.compiled = compiled
    dt = one.cfl_dt()
    counts, low = [], []
    for s in range(60 if case == "buried" else 500 if compiled else 150):
        counts.append(len(isl._islands(isl._weights()[0],
                                       np.array(isl.shape) - 1)))
        one.substep(dt)
        isl.substep(dt)
        low.append(isl.x[isl.kind == 1, 2].min())
        for f in FIELDS:
            assert np.array_equal(getattr(one, f), getattr(isl, f)), \
                f"{case}: {f} differs at substep {s}"
    # and the island path was the one taken: two islands, from the start
    # when they start apart, and from the substep the gap opens otherwise
    assert counts[-1] == 2 and set(counts) == ({1, 2} if case == "parting" else {2})
    if case == "buried":
        assert isl.deep_hits > 0, "nothing was read from deeper than the margin"
    if case == "apart":
        assert isl.x[isl.kind == 0, 2].min() < 0.25, "the block never landed"
        assert min(low) < 1.22, "the glass never reached the shelf"
    if case == "parting" and compiled:
        assert isl.broken() > 0, "nothing cracked"
        assert isl.x[:, 0].min() < 0.1 and isl.x[:, 0].max() > 4.5, "never reached the edges"


def test_ISLANDS_cost_the_PARTICLES_not_the_DISTANCE(monkeypatch):
    """Two blocks falling at opposite corners of a room, one 1.1 m above the
    other: how many grid nodes each substep solves and how many cells the
    distance field is built over must not depend on how far apart they
    are. The single window is the control: it grows with the room, which
    is what islands are for.

    Counted rather than timed, so it cannot flake. Timed under the lock
    (build/X6), islands cost the same per substep at any distance and the
    single window does not; see the step's report."""
    nodes, cells = [], []
    if kernels.HAVE_NUMBA:
        p2g, edt2 = kernels.p2g, kernels.edt2

        def count_p2g(*a):
            nodes.append(len(a[-3]))
            return p2g(*a)

        def count_edt2(seed):
            cells.append(seed.size)
            return edt2(seed)
        monkeypatch.setattr(kernels, "p2g", count_p2g)
        monkeypatch.setattr(kernels, "edt2", count_edt2)
    else:
        p2g, edt2 = MPM._p2g, MPM._edt2

        def count_p2g(self, *a):
            nodes.append(len(a[-3]))
            return p2g(self, *a)

        def count_edt2(self, seed):
            cells.append(seed.size)
            return edt2(self, seed)
        monkeypatch.setattr(MPM, "_p2g", count_p2g)
        monkeypatch.setattr(MPM, "_edt2", count_edt2)

    def run(span, islands):
        n = 12 + span                        # cells along x and along y
        w = MPM((n, n, 40), dx=0.1)
        w.add_wall((0.0, 0.0, 0.0), (n * 0.1, n * 0.1, 0.2))
        w.add_box((0.4, 0.4, 1.5), (0.8, 0.8, 1.9), "wood", seed=1)
        far = (n - 8) * 0.1
        w.add_box((far, far, 2.6), (far + 0.4, far + 0.4, 3.0), "wood", seed=2)
        w.v[:, 2] = -4.0
        w.islands = islands
        # counted on the stage-by-stage path, whose P2G is a call that can
        # be counted; the fused substep solves the same window inside itself
        w.fused = False
        nodes.clear()
        cells.clear()
        dt = w.cfl_dt()
        for _ in range(40):
            w.substep(dt)
        assert w.x[:, 2].max() < 2.95, "they did not fall"
        return sum(nodes), sum(cells)

    near, far = run(50, True), run(120, True)
    assert near == far, (near, far)
    near1, far1 = run(50, False), run(120, False)
    assert far1[0] > 4 * near1[0] and far1[1] > 4 * near1[1], (near1, far1)
    assert far[0] < far1[0] / 10, (far, far1)


def test_the_DISTANCE_FIELD_NEAR_THE_PARTICLES_is_the_WHOLE_WORLDS(monkeypatch):
    """The window's distance field against the whole world's, on random
    walls and random particle clouds, at exactly what a substep reads: the
    distance and direction at each particle, and the distance at each node
    the momentum grid uses.

    Near a surface the two are the same bits. Further off they may differ
    (the window measures only to its own edge), but never in what the solver
    DECIDES: whether a particle is inside a wall, and whether a node is held
    still. And a particle buried deeper than SDF_MARGIN is counted."""
    rng = np.random.default_rng(7)
    near = (core.SDF_MARGIN - 1) * 0.1
    modes = (False, True) if kernels.HAVE_NUMBA else (False,)
    tested = windowed = 0
    for trial in range(12):
        # half the trials with no slack, so the window's edges sit as close
        # to the particles as they ever can
        monkeypatch.setattr(core, "SDF_SLACK", 0 if trial % 2 else 6)
        shape = [int(s) for s in rng.integers(18, 26, 3)]
        shape[trial % 3] = int(rng.integers(36, 48))     # long on one axis
        shape = tuple(shape)
        w = MPM(shape, dx=0.1)
        mid = rng.uniform(0.3, 0.7, 3) * np.array(shape) * 0.1
        for _ in range(int(rng.integers(3, 8))):        # walls in and around
            lo = np.maximum((mid * 10).astype(int) + rng.integers(-8, 3, 3), 0)
            hi = lo + rng.integers(1, 9, 3)
            w.wall[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
        pts = mid + rng.uniform(-0.25, 0.25, (80, 3))
        w.x = np.clip(pts, 0.05, (np.array(shape) - 1.5) * 0.1)
        w.compiled = False
        w._build_sdf()                                 # the definition
        dense = w._sample(w.phi, w.x)
        base = np.floor(w.x * w.inv_dx - 0.5).astype(np.int64)
        nodes = np.unique(np.clip((base[:, None, :] + np.array(
            [[i, j, k] for i in range(3) for j in range(3) for k in range(3)])
        ).reshape(-1, 3), 0, np.array(shape) - 1), axis=0)
        dnode = w.phi_node[tuple(nodes.T)]
        for compiled in modes:
            w.compiled = compiled
            w._near = None
            lo, hi, _, phi, pgrad, node, walled = w._near_walls()
            got = w._sample(phi, w.x, lo)
            assert ((got < 0.0) == (dense < 0.0)).all(), (trial, compiled)
            close = np.abs(dense) <= near
            assert np.array_equal(got[close], dense[close]), (trial, compiled)
            hit = close & (dense < 0.0)
            assert np.array_equal(w._sample(pgrad, w.x[hit], lo),
                                  w._sample(w.pgrad, w.x[hit])), (trial, compiled)
            wnode = node[tuple((nodes - lo).T)]
            held = -1.5 * w.dx                         # as `substep` asks it
            assert ((wnode < held) == (dnode < held)).all(), (trial, compiled)
            close = np.abs(dnode) <= near
            assert np.array_equal(wnode[close], dnode[close]), (trial, compiled)
            tested += int(hit.sum())
            windowed += int((hi - lo < np.array(shape)).any())
    assert tested > 50, "too few particles touched a wall to test anything"
    assert windowed >= 6, "the window was the whole world: nothing was tested"
    # and a particle six cells inside a thick wall is counted, not hidden
    w = MPM((24, 24, 24), dx=0.1)
    w.add_wall((0.0, 0.0, 0.0), (2.4, 2.4, 1.0))
    w.add_box((1.0, 1.0, 0.3), (1.2, 1.2, 0.5), "stone")
    w.substep(1e-4)
    assert w.deep_hits > 0


def test_EVERY_PER_PARTICLE_ARRAY_IS_IN_THE_TABLE():
    """`_append` grows exactly the arrays `PARTICLE_FIELDS` names. An array
    with one entry per particle that the table missed would fall out of step
    with the particle count the first time particles were added, and code
    that keeps or drops particles by walking the table would leave it
    behind. So: every array whose length is the particle count is in the
    table, every one in the table has that length, and a new particle
    starts where the table says."""
    w = MPM((12, 12, 12), dx=0.1)
    names = [f[0] for f in core.PARTICLE_FIELDS]
    assert len(set(names)) == len(names)
    for k, box in enumerate((((0.2, 0.2, 0.2), (0.6, 0.6, 0.4)),
                             ((0.6, 0.6, 0.6), (0.8, 1.0, 0.8)))):
        w.add_box(*box, "glass" if k else "wood")
        n = len(w.x)
        assert n not in w.shape
        per = {a for a, v in vars(w).items()
               if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == n}
        assert per == set(names), per ^ set(names)
    for name, dtype, tail, start in core.PARTICLE_FIELDS:
        a = getattr(w, name)
        assert a.dtype == dtype and a.shape[1:] == tail, name
    assert (w.F == np.eye(3)).all() and (w.Jp == 1.0).all()
    assert (w.cid == -1).all() and (w.batch == -1).all() and (w.eid == 0).all()
    assert (w.m > 0.0).all() and (w.vol0 > 0.0).all() and (w.strength > 0.0).all()


def test_a_PARTICLE_ALMOST_DEEP_reads_the_WHOLE_WORLDS_DIRECTION_OUT(monkeypatch):
    """A particle up to SDF_MARGIN cells inside a wall is not counted in
    `deep_hits`, so the window owes it the world's exact distance AND
    direction. Its direction comes from cells two away from it, and the
    nearest free cell of one of those can lie further off than the window
    used to reach. Found by search: a solid world with two free cells, the
    particle's nearest at (-3, -2, -1) cells and another at (8, 0, 1), the
    particle 3.9 cells deep. With the window as tight as it gets the old
    margin read a different direction; DEEP_READ more cells read the same
    bits as the whole world's field."""
    monkeypatch.setattr(core, "SDF_SLACK", 0)
    c = 13
    w = MPM((26, 26, 26), dx=0.1)
    w.wall[:] = True
    w.wall[c - 3, c - 2, c - 1] = False
    w.wall[c + 8, c, c + 1] = False
    w.x = np.full((1, 3), (c + 0.6) * 0.1)
    w.compiled = False
    w._build_sdf()                                   # the definition
    dense, dg = w._sample(w.phi, w.x), w._sample(w.pgrad, w.x)
    assert -core.SDF_MARGIN * w.dx <= dense[0] < -3.5 * w.dx, "not the case"
    modes = (False, True) if kernels.HAVE_NUMBA else (False,)
    for deep_read in (core.DEEP_READ, 0):
        monkeypatch.setattr(core, "DEEP_READ", deep_read)
        for compiled in modes:
            w.compiled = compiled
            w._near = None
            lo, hi, _, phi, pgrad, _, _ = w._near_walls()
            same = (np.array_equal(w._sample(phi, w.x, lo), dense)
                    and np.array_equal(w._sample(pgrad, w.x, lo), dg))
            # and the case is a real one: without the extra cells it is wrong
            assert same == (deep_read > 0), (deep_read, compiled)
