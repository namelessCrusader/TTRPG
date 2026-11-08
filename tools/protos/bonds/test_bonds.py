"""What the bond prototype must get right before any comparison means anything.

Run by path: `python3 -m pytest tools/protos/bonds/test_bonds.py`. The folder
above (tools/protos/conftest.py) keeps pytest from collecting prototypes on
its own, and nothing in tests/ imports this. Each test builds a few dozen
pieces and runs well under a second of sim, except the tree's (150 pieces,
2.7 s of sim at real stiffness: about 10 s)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bonds as B          # noqa: E402
import scenes as S         # noqa: E402


def _beam(n=10, s=0.05, stiff=1.0, g=0.0):
    """A single row of `n` free cube pieces of wood, grain along x, bonded to
    a pinned piece at x = 0: a cantilever of length n*s."""
    b = B.Bonded(pp=False)
    m = 600 * s ** 3
    root = b.piece((0, 0, 1), m, (s / 2,) * 3, kin=True)
    prev = root
    for k in range(n):
        i = b.piece(((k + 1) * s, 0, 1), m, (s / 2,) * 3)
        b.bond(prev, i, (B.WOOD_L, B.WOOD_T), s * s, s ** 4 / 12, s / 2, grain=(1, 0, 0))
        prev = i
    b.finish(stiff=stiff, strength_scale=1.0)
    b.g = g
    return b


def test_FREE_SPACE_keeps_MOMENTUM_and_TURNING_exactly():
    """Every bond pushes its two pieces equally and oppositely at one point,
    so nothing inside an object can change its momentum or its angular
    momentum. A spinning, shaking pane that also BREAKS (bonds going in the
    middle of a step) is the hard case."""
    b, ix = S.glass()
    b.finish(stiff=1e-3)
    b.g = 0.0
    b.wlo, b.whi = b.wlo[:0], b.whi[:0]
    c = b.x.mean(0)
    rng = np.random.default_rng(1)
    b.sig *= 0.05                    # weak glass, so that it breaks as it spins
    b.v[:] = np.cross((0.5, 30.0, 4.0), b.x - c) + rng.normal(0, 0.5, b.x.shape)
    b.rebase()
    P0, L0 = b.momentum()
    for _ in range(8):
        b.step(0.025)
    P, L = b.momentum()
    assert (~b.alive).sum() > 0, "nothing broke, so breaking was not tested"
    assert np.linalg.norm(P - P0) < 1e-9 * b.M          # kg m/s, against 1 m/s
    assert np.linalg.norm(L - L0) < 1e-9 * np.linalg.norm(L0)


def test_A_FREE_PIECE_FALLS_AT_g():
    b = B.Bonded()
    b.piece((0, 0, 5), 1.0, (0.05,) * 3)
    b.finish()
    z0 = b.x[0, 2]
    for _ in range(20):
        sub = b.step(0.025)
    # symplectic Euler drops an extra g*t*h/2 (here about a millimetre)
    h = 0.025 / sub
    want = z0 - 0.5 * 9.81 * b.t ** 2 - 0.5 * 9.81 * b.t * h
    assert b.x[0, 2] == pytest.approx(want, abs=1e-9)


def test_A_CANTILEVER_SAGS_AS_BEAM_THEORY_SAYS():
    """F L^3 / (3 E I) at real wood stiffness: the bending spring of a bond
    is a beam's, so a row of them is a beam."""
    b = _beam()
    F = 50.0
    # eased on over 0.2 s, so the beam (138 Hz, lightly damped) is not rung
    for k in range(40):
        for c in range(10):
            u = min((k * 10 + c) / 80.0, 1.0)
            b.fext[-1, 2] = -F * u * u * (3 - 2 * u)
            b.step(0.0025)
    L, s = 0.5, 0.05
    I = s ** 4 / 12
    E, G = B.WOOD_L[0], B.WOOD_L[0] / (2 * 1.35)
    want = F * L ** 3 / (3 * E * I) + F * L / (G * s * s)
    got = 1.0 - b.x[-1, 2]
    assert got == pytest.approx(want, rel=0.03), (got, want)


def test_A_CANTILEVER_BREAKS_AT_ITS_ROOT():
    """Bending moment is largest where the beam is held, so that is where
    the first bond goes, whatever the flaws do elsewhere."""
    b = _beam()
    for k in range(2000):
        b.fext[-1, 2] = -20.0 * 2 ** (k / 200)    # doubling every 0.5 s
        b.step(0.0025)
        if (~b.alive).any():
            break
    gone = np.flatnonzero(~b.alive)
    assert len(gone) and gone[0] == 0, gone
    # and at the load beam theory says, to within the time step of the ramp
    s = 0.05
    want = B.WOOD_L[2] * (s ** 4 / 12) / (s / 2) / (0.5 - s / 2)
    assert -b.fext[-1, 2] == pytest.approx(want, rel=0.05)


def test_A_ROPE_SWINGS_AT_THE_PENDULUM_PERIOD():
    b, ix = S.rope(angle=np.radians(20))
    b.finish(stiff=1.0, strength_scale=1.0)
    xs, ts = [], []
    for _ in range(int(4.2 / 0.025)):
        b.step(0.025)
        xs.append(b.x[ix["bob"], 0])
        ts.append(b.t)
    xs, ts = np.array(xs), np.array(ts)
    cross = [ts[k] - xs[k] * (ts[k + 1] - ts[k]) / (xs[k + 1] - xs[k])
             for k in range(len(xs) - 1) if xs[k] > 0 >= xs[k + 1]]
    T = np.diff(cross).mean()
    assert T == pytest.approx(S.rope_period(), rel=0.01), (T, S.rope_period())


def test_A_SHAKEN_TREE_HOLDS_AND_A_FELLED_ONE_GOES_OVER_ON_ITS_HINGE():
    """The tree scene's two events, with nothing else in the way: the shaker
    stroke must not snap the trunk (wave 9's first runs snapped it below the
    clamp, then measured a broken tree falling), and once felled the tree
    must turn on its hinge until the hinge lets go near one radian (the butt
    resting on the ground used to hold it standing)."""
    b, ix = S.tree()
    b.finish(stiff=1.0, strength_scale=1.0)
    S.root(b, ix)
    b.retime()
    b.pin(ix["clamp"], amp=(0.015, 0.0, 0.0), freq=12.0, on=0.3, off=1.1)
    wood = np.isin(b.btag, (1, 2, 3, 9))
    for k in range(48):                        # gravity on, then the stroke
        u = min(k * 0.025 / 0.3, 1.0)
        b.g = 9.81 * u * u * (3 - 2 * u)
        b.step(0.025)
    assert b.alive[wood].all(), np.flatnonzero(~b.alive & wood)
    b.alive[ix["grip"]] = False
    S.fell(b, ix)
    b.fext[ix["trunk"][7], 0] = 300.0
    top, butt = ix["trunk"][-1], ix["trunk"][0]
    lean_at_break = None
    for k in range(60):
        if k == 20:
            b.fext[ix["trunk"][7], 0] = 0.0
        b.step(0.025)
        if lean_at_break is None and not b.alive[ix["root"]]:
            d = b.x[top] - b.x[butt]
            lean_at_break = np.degrees(np.arccos(d[2] / np.linalg.norm(d)))
    # the hinge let go near a radian of lean, the way the tree was pushed,
    # and no wood broke before it did
    assert lean_at_break is not None
    assert 45.0 < lean_at_break < 70.0, lean_at_break
    first = np.nanmin(np.where(wood, b.tbreak, np.nan))
    assert first == b.tbreak[ix["root"]], first
    assert b.x[top, 0] > 2.0, b.x[top]
