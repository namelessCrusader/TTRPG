"""THE LAZY LAWS (wave 15, S4-3): exact speedups for a world with a warm body in it,
and the audit mode that checks each one against the full answer.

Small worlds only (a few hundred thousand voxels would be slow here): the cost
is measured by tools in the lane's report, the EXACTNESS is what is tested.
"""
import numpy as np
import pytest

from src.voxel import lazyaudit, scenes
from src.voxel import outside as O
from src.voxel.bodyworld import BodyWorld
from src.voxel.coldworld import ColdWorld
from src.voxel.lazykernels import all_settled, any_nz, nz_box, same4
from src.voxel.materials import IRON
from src.voxel.sim import AIR, STONE, WOOD, World


def _room(lazy=True, cls=BodyWorld, shape=(44, 36, 24), people=1, fire=False):
    nx, ny, nz = shape
    w = cls(nx, ny, nz, voxel_cm=5)
    w.lazy_exact = lazy
    w.fill(0, nx, 0, ny, 0, 1, STONE)
    w.fill(4, nx - 4, 4, ny - 4, 0, nz - 4, STONE)
    w.mat[5:nx - 5, 5:ny - 5, 1:nz - 5] = AIR
    w.smass[5:nx - 5, 5:ny - 5, 1:nz - 5] = 0.0
    for i in range(people):
        scenes._person(w, 10 + 8 * i, 10)
    if fire:
        w.fill(24, 28, 20, 24, 1, 4, WOOD)
        w.E[24:28, 20:24, 1:4] = (600.0 - 20.0) * w.heat_capacity()[24:28, 20:24, 1:4]
    w.policy = scenes._Wants()
    return w


def _state(w):
    out = {k: getattr(w, k).copy() for k in ("E", "mat", "smass", "fl", "fvol", "o2", "smoke", "p_add")}
    out["pcell"] = None if w.pcell is None else w.pcell.copy()
    out["vcell"] = None if w.vcell is None else [v.copy() for v in w.vcell]
    return out


def _differs(a, b):
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, list):
            if not all(p.tobytes() == q.tobytes() for p, q in zip(x, y)):
                return k
        elif x is None or y is None:
            if x is not y:
                return k
        elif x.tobytes() != y.tobytes():
            return k
    return None


def _step(w):
    w.step()
    for p in w.persons:
        p["events"].clear()


# ------------------------------------------------------------------ the kernels
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_nz_box_is_the_window_of_not_equal_zero(dtype):
    rng = np.random.default_rng(3)
    w = BodyWorld(20, 16, 12, voxel_cm=5)
    for trial in range(60):
        a = np.zeros((20, 16, 12), dtype)
        for _ in range(rng.integers(0, 5)):
            a[tuple(rng.integers(0, s) for s in a.shape)] = rng.choice([1.5, -2.0, np.nan, np.inf])
        a[rng.integers(0, 20), rng.integers(0, 16), rng.integers(0, 12)] = -0.0   # not a non-zero
        for pad in (0, 1, 3):
            assert w._nz_window(a, pad) == w._window(a != 0, pad)


def test_all_settled_agrees_with_the_box_kernel():
    rng = np.random.default_rng(5)
    for trial in range(30):
        shp = (9, 7, 6)
        C = rng.uniform(1.0, 500.0, shp).astype(np.float32)
        cls = rng.integers(0, 4, shp).astype(np.uint8)
        rest, eout = np.float32(rng.choice([0.0, -25.0, 10.0])), np.float32(rng.choice([0.0, -30.0]))
        tgt = np.where((cls & 1) != 0, eout, rest).astype(np.float32)
        E = (tgt * C).astype(np.float32)
        if trial % 2:                                          # one cell off its rest, by more or less than 5 mK
            idx = tuple(rng.integers(0, s) for s in shp)
            E[idx] += np.float32(rng.choice([0.001, 0.02]) * C[idx])
        lo, hi, nun = O._box_nb(E, C, cls, rest, eout, np.float32(O.SETTLE_K), np.zeros((1, 1, 1), bool), False)
        assert bool(all_settled(E, C, cls, rest, eout, np.float32(O.SETTLE_K))) == (nun == 0)


@pytest.mark.parametrize("dtype", (np.float32, np.float64, np.uint8, bool))
def test_any_nz_is_any(dtype):
    rng = np.random.default_rng(11)
    for n in (1, 7, 4096, 4097, 20000):
        a = np.zeros(n, dtype)
        assert bool(any_nz(a)) == bool(a.any())
        a[int(rng.integers(0, n))] = 1
        assert bool(any_nz(a)) == bool(a.any())
    if dtype in (np.float32, np.float64):
        z = np.zeros(9000, dtype)
        z[8999] = np.nan
        assert bool(any_nz(z)) and bool(z.any())
        z[8999] = -0.0
        assert not any_nz(z) and not z.any()


def test_same4_is_four_array_equals():
    rng = np.random.default_rng(9)
    for trial in range(40):
        n = int(rng.integers(1, 20000))
        a = [rng.integers(0, 5, n).astype(np.uint8), rng.normal(size=n).astype(np.float32),
             rng.integers(0, 5, n).astype(np.uint8), rng.normal(size=n).astype(np.float32)]
        b = [x.copy() for x in a]
        if trial % 3 == 1:
            j = int(rng.integers(0, 4))
            b[j][int(rng.integers(0, n))] += 1
        if trial % 3 == 2:
            a[1][int(rng.integers(0, n))] = np.nan
            b[1] = a[1].copy()                                    # NaN is not equal to itself, as in numpy
        want = all(np.array_equal(x, y) for x, y in zip(a, b))
        assert bool(same4(a[0], b[0], a[1], b[1], a[2], b[2], a[3], b[3])) == want


# ------------------------------------------------------------------ bit for bit, audited
def test_a_lazy_body_world_is_the_full_world_bit_for_bit_and_every_answer_is_audited(monkeypatch):
    monkeypatch.setenv("MARK1_AUDIT_LAZY", "1")
    lazyaudit.COUNTS.clear()
    lazy, full = _room(True, fire=True), _room(False, fire=True)
    for i in range(30):
        _step(lazy)
        _step(full)
        assert _differs(_state(lazy), _state(full)) is None, "tick %d" % i
        assert lazy._matter_gen == full._matter_gen and lazy.shed == full.shed
    # the audit really ran, on every kind of answer (a guard that never fires proves nothing)
    for name in ("unsettled", "matter_moved", "o2_asleep", "warm_cells"):
        assert lazyaudit.COUNTS.get(name, 0) > 0, name


def test_burning_is_the_same_cell_for_cell_in_a_lazy_world():
    lazy, full = _room(True, fire=True), _room(False, fire=True)
    for _ in range(8):
        _step(lazy)
        _step(full)
        assert lazy.burning().tobytes() == full.burning().tobytes()
    assert lazy.burning().any()


def test_a_lazy_quiet_body_world_with_two_people_is_the_full_world(monkeypatch):
    monkeypatch.setenv("MARK1_AUDIT_LAZY", "1")
    lazyaudit.COUNTS.clear()
    lazy, full = _room(True, people=2), _room(False, people=2)
    for i in range(40):
        _step(lazy)
        _step(full)
        assert _differs(_state(lazy), _state(full)) is None, "tick %d" % i
        # the count of times matter was found moved is state a golden check compares: no ask may move
        assert lazy._matter_gen == full._matter_gen and (lazy._cap is None) == (full._cap is None)
        assert (lazy._por is None) == (full._por is None) and hasattr(lazy, "skin") == hasattr(full, "skin")
    for name in ("cells_por", "cells_padd", "law_air", "o2_asleep", "warm_cells", "burn_asleep",
                 "quiet_boil", "quiet_melt", "skin_water", "hot", "busy", "matter_same", "burning"):
        assert lazyaudit.COUNTS.get(name, 0) > 0, name


def test_a_plain_world_takes_the_old_path(monkeypatch):
    monkeypatch.setenv("MARK1_AUDIT_LAZY", "1")
    lazyaudit.COUNTS.clear()
    assert World.lazy_exact is False
    w = World(20, 16, 12, voxel_cm=5)
    w.fill(0, 20, 0, 16, 0, 1, STONE)
    w.E[10, 8, 3] = 500.0
    for _ in range(5):
        w.step()
    assert lazyaudit.COUNTS == {}


# ------------------------------------------------------------------ the guards fire
def test_MUTATION_a_forgotten_matter_writer_is_caught_by_the_audit(monkeypatch):
    """A law that is listed as unable to write matter, and does: the memo goes
    stale and the audit raises on the next ask. (The failure the audit is for.)"""
    monkeypatch.setenv("MARK1_AUDIT_LAZY", "1")

    class Liar(BodyWorld):
        def _law_o2(self):                          # `_law_o2` is on the pure list
            self.heat_capacity()                    # (a scan: the memo is now verified)
            self.smass[8, 8, 1] += 1.0              # ...and then it writes matter
            super()._law_o2()

    w = Liar(44, 36, 24, voxel_cm=5)
    w.fill(0, 44, 0, 36, 0, 1, STONE)
    w.E[20, 20, 5] = 50.0                           # something warm, so the laws run
    w.lazy_exact = True
    with pytest.raises(lazyaudit.LazyMismatch):
        for _ in range(6):
            w.step()


def test_MUTATION_a_box_that_is_one_cell_short_is_caught(monkeypatch):
    monkeypatch.setenv("MARK1_AUDIT_LAZY", "1")
    w = _room(True)
    real = type(w)._nz_window

    def short(self, a, pad):
        box = real(self, a, pad)
        return None if box is None else tuple(slice(s.start + 1, s.stop) for s in box)

    monkeypatch.setattr(type(w), "_nz_window", short)
    with pytest.raises(lazyaudit.LazyMismatch):
        for _ in range(3):
            _step(w)


def test_MUTATION_a_summary_of_the_heat_that_says_quiet_over_a_melting_bar_is_caught(monkeypatch):
    """The proofs read one summary of the heat. A summary that forgets the bar melts it
    (the law is skipped) and the audit says so."""
    monkeypatch.setenv("MARK1_AUDIT_LAZY", "1")
    w = _room(True, people=0)
    w.fill(20, 23, 20, 23, 1, 4, IRON)
    w.E[20:23, 20:23, 1:4] = (1700.0 - 20.0) * w.heat_capacity()[20:23, 20:23, 1:4]
    real = type(w)._hot_make

    def blind(self):
        h = real(self)
        h.T = None
        h.burn_asleep = h.skin_quiet = True
        return h

    monkeypatch.setattr(type(w), "_hot_make", blind)
    with pytest.raises(lazyaudit.LazyMismatch):
        for _ in range(4):
            _step(w)


def test_a_world_with_another_law_of_the_same_name_gets_no_proof_for_it():
    """A proof is about the law of heat.py; a mixin that stands in front of it with
    its own `_law_crack` is always called."""
    calls = []

    class W(BodyWorld):
        def _law_crack(self):
            calls.append(self.tick)
            return super()._law_crack()

    w = W(30, 24, 16, voxel_cm=5)
    w.fill(0, 30, 0, 24, 0, 1, STONE)
    w.E[10, 10, 3] = 30.0
    for _ in range(4):
        w.step()
    assert len(calls) == 4


def test_the_audit_is_off_by_default(monkeypatch):
    monkeypatch.delenv("MARK1_AUDIT_LAZY", raising=False)
    assert not lazyaudit.enabled()
    assert lazyaudit.check("x", 1, lambda: 2) == 1          # a wrong lazy answer passes, unchecked


# ------------------------------------------------------------------ the memo itself
def test_the_matter_memo_answers_no_inside_a_tick_and_never_between_ticks(monkeypatch):
    w = _room(True)
    _step(w)
    # between ticks: every ask is a real scan, so a write is seen at once
    w.heat_capacity()
    w.smass[8, 8, 1] += 1.0
    gen = w._matter_gen
    w.heat_capacity()
    assert w._matter_gen == gen + 1
    # inside a tick the memo stands until a matter-writing law has run
    scans = []
    real = type(w)._matter_scan
    monkeypatch.setattr(type(w), "_matter_scan", lambda self: scans.append(1) or real(self))
    _step(w)
    n = len(scans)
    assert 0 < n < 12          # (it was 12 scans a tick before the memo; the count is the saving)


def test_the_porosity_mean_is_rebuilt_when_matter_moves():
    w = _room(True)
    _step(w)
    a = w._cells(w.porosity())
    assert w._cells(w.porosity()) is a                       # kept while matter has not moved
    w.mat[10, 10, 10] = STONE
    w.smass[10, 10, 10] = 5.0
    b = w._cells(w.porosity())
    assert b is not a
    assert np.array_equal(b, World._cells(w, w.porosity()))


# ------------------------------------------------------------------ cold
def _cold_room(lazy, climate):
    w = ColdWorld(40, 30, 20, voxel_cm=5)
    w.lazy_exact = lazy
    w.fill(0, 40, 0, 30, 0, 1, STONE)
    w.fill(10, 30, 8, 22, 1, 10, STONE)
    w.fill(12, 28, 10, 20, 1, 10, AIR)
    w.fill(10, 30, 8, 22, 10, 12, WOOD)
    w.set_climate(climate, soak=True)
    return w


def test_cold_quiet_scan_is_the_full_scan_and_sees_an_outside_write_on_the_next_tick(monkeypatch):
    from src.voxel import climate as K
    monkeypatch.setenv("MARK1_AUDIT_LAZY", "1")
    lazyaudit.COUNTS.clear()
    clim = K.Climate.reference(t_c=-5.0)
    lazy, full = _cold_room(True, clim), _cold_room(False, clim)
    for i in range(12):
        lazy.step()
        full.step()
        assert lazy.E.tobytes() == full.E.tobytes(), "tick %d" % i
    assert lazyaudit.COUNTS.get("cold_settled", 0) > 0
    # a write from OUTSIDE the laws, between two ticks, on one cell: seen on the very next tick
    for w in (lazy, full):
        w.E[20, 15, 5] += np.float32(1e-3) * w.heat_capacity()[20, 15, 5] * 50
    box = None
    lazy._tick_start()
    try:
        box = lazy._unsettled(0)[0]
    finally:
        lazy._tick_end()
    assert box is not None and box[0].start <= 20 < box[0].stop
    lazy.step()
    full.step()
    assert lazy.E.tobytes() == full.E.tobytes()


def test_MUTATION_a_quiet_scan_that_misses_a_cell_is_caught_by_the_audit(monkeypatch):
    from src.voxel import climate as K
    from src.voxel import outside_world as OW
    monkeypatch.setenv("MARK1_AUDIT_LAZY", "1")
    w = _cold_room(True, K.Climate.reference(t_c=-5.0))
    w.step()
    w.E[20, 15, 5] += np.float32(1e-3) * w.heat_capacity()[20, 15, 5] * 50      # one cell off its rest
    monkeypatch.setattr(OW.LK, "all_settled", lambda *a: True)                  # a scan that never looks
    with pytest.raises(lazyaudit.LazyMismatch):
        w.step()
