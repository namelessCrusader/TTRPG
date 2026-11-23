"""THE TWIN FRAMEWORK AND THE SHUT ROOM'S STAR (wave 15 batch 2, lane 2d; S4-1, owner SC5).

Three things:
  1. the SHUT ROOM is DETECTED (shutroom.room_of), never passed in, and its star law is zero-sum;
  2. the tick law and the night jump agree on a sealed room's heat (where it goes, the air's rise), and
     both mutations the spec names are caught;
  3. the shared harness (twinharness.compare) reports the drift of a coarse version against the full one:
     proved on the breath twin (EXACT) and on the star (APPROXIMATE, the budgets written in slowtwins.BUDGETS).
"""
import inspect

import numpy as np
import pytest

from src.voxel import shutroom as SR
from src.voxel import slownet as N
from src.voxel import slowtime as ST
from src.voxel import slowtwins as TW
from src.voxel import twinharness as H
from src.voxel.airsink_world import RegionSink
from src.voxel.bodyheat import BodyHeatMixin
from src.voxel.materials import AIR, AMBIENT, SIGMA, STONE, TICK_S
from src.voxel.scenes import _person
from src.voxel.scenes_slow import SlowWorld, scene_inn_body

K0 = 273.15


def _room_world(nx=24, ny=20, nz=24, gap=False):
    w = SlowWorld(nx, ny, nz, voxel_cm=5)
    w.fill(0, nx, 0, ny, 0, 1, STONE)
    w.fill(0, nx, 0, 1, 0, nz, STONE)
    w.fill(0, nx, ny - 1, ny, 0, nz, STONE)
    w.fill(0, 1, 0, ny, 0, nz, STONE)
    w.fill(nx - 1, nx, 0, ny, 0, nz, STONE)
    w.fill(0, nx, 0, ny, nz - 1, nz, STONE)
    if gap:
        w.fill(0, 1, 8, 12, 2, 10, AIR)           # a doorway in the wall
    w.exits = []
    w.step()
    return w


def _bw(**kw):
    from src.voxel.bodyworld import BodyWorld

    class W(BodyWorld, SlowWorld):
        pass
    nx, ny, nz = 24, 20, 24
    w = W(nx, ny, nz, voxel_cm=5)
    w.fill(0, nx, 0, ny, 0, nz, STONE)
    w.fill(1, nx - 1, 1, ny - 1, 1, nz - 1, AIR)
    if kw.get("gap"):
        w.fill(0, 1, 8, 12, 2, 10, AIR)
    w.exits = []
    w.step()
    return w


# ------------------------------------------------------------ detection ----
def test_the_shut_room_is_detected_and_never_passed_in():
    assert "closed" not in inspect.signature(RegionSink.__init__).parameters
    w = _bw()
    assert SR.room_of(w) is None                              # no sink: the open world's law, nothing detected
    w.airsink = RegionSink(w, (slice(1, 23), slice(1, 19), slice(1, 23)))
    room = SR.room_of(w)
    assert room is not None and room.sealed
    assert int(room.n_in[0, 5, 5]) == 1 and int(room.n_in[0, 0, 0]) == 0     # the wall cell sees one face of air
    w2 = _bw(gap=True)
    w2.airsink = RegionSink(w2, (slice(0, 23), slice(1, 19), slice(1, 23)))
    assert SR.room_of(w2) is None                             # a hole in the wall: gas leaves, the old law stays
    w3 = _bw()
    w3.fill(0, 1, 8, 12, 2, 10, AIR)                         # air on the lattice edge: past it is the open world
    w3.airsink = RegionSink(w3)
    assert SR.room_of(w3) is None


def test_a_world_that_does_not_have_the_law_keeps_the_old_one():
    w = _bw()
    w.airsink = RegionSink(w, (slice(1, 23), slice(1, 19), slice(1, 23)))
    assert SR.room_of(w) is not None
    w.shut_room_star = False
    assert SR.room_of(w) is None
    plain = _room_world()                                      # SlowWorld: no BodyHeatMixin, no law
    plain.airsink = RegionSink(plain)
    assert SR.room_of(plain) is None


# ---------------------------------------------------------- the law --------
def _synthetic(seed=3, n=40):
    rng = np.random.default_rng(seed)
    Ts = 20.0 + rng.uniform(-3, 9, n)
    e = np.full(n, 0.9)
    af = np.ones(n)
    n_in = rng.integers(1, 4, n)
    body = np.zeros(n, bool)
    body[:5] = True
    Ts[:5] = 34.0
    return Ts, e, af, n_in, body


def test_the_star_sums_to_zero_and_a_uniform_room_gives_exactly_zero():
    Ts, e, af, n_in, body = _synthetic()
    Csur = np.full(Ts.shape, 1e9)
    loss, tmr, resid = SR.star_loss(Ts, e, af, n_in, n_in, 0.0025, TICK_S, Csur, body)
    assert abs(loss.sum()) < 1e-9 and abs(resid) < 1e-9
    assert loss[:5].min() > 0 and AMBIENT < tmr < 34.0             # a warm body gives; T_mr is among the walls
    u = np.full(10, 22.0)
    z, t0, _ = SR.star_loss(u, np.full(10, 0.9), np.ones(10), np.full(10, 2), np.full(10, 2), 0.0025, TICK_S,
                            np.full(10, 1e9), np.zeros(10, bool))
    assert not z.any() and t0 == 22.0


def test_a_body_in_an_empty_shut_room_gives_the_walls_what_the_small_object_formula_says():
    """Incropera: a small convex object in a large enclosure, q = eps A sigma (T1^4 - T2^4). Walls all at 20 C,
    a body of known skin and area: the walls receive that, to 5 percent (here to rounding: the body is not in the
    pool, so T_mr is exactly the wall temperature)."""
    nw, nb = 400, 20
    Ts = np.r_[np.full(nb, 33.0), np.full(nw, 20.0)]
    e = np.full(nb + nw, 0.98)
    af = np.ones(nb + nw)
    n_in = np.r_[np.full(nb, 6), np.full(nw, 1)]
    body = np.r_[np.ones(nb, bool), np.zeros(nw, bool)]
    A = 0.0025
    loss, tmr, _ = SR.star_loss(Ts, e, af, n_in, n_in, A, 1.0, np.full(nb + nw, 1e9), body)
    q_formula = 0.98 * (6 * nb * A) * SIGMA * ((33.0 + K0) ** 4 - (20.0 + K0) ** 4)
    assert abs(loss[:nb].sum() / q_formula - 1.0) < 0.05
    assert abs(-loss[nb:].sum() / q_formula - 1.0) < 0.05           # the walls got it, nothing left the room
    assert abs(tmr - 20.0) < 1e-9


def test_faces_onto_gas_outside_the_room_keep_the_old_law():
    Ts = np.array([40.0])
    ones = np.ones(1)
    loss, tmr, _ = SR.star_loss(Ts, ones * 0.9, ones, np.array([0]), np.array([2]), 0.0025, TICK_S,
                                np.full(1, 1e9), np.zeros(1, bool))
    want = 0.9 * SIGMA * 0.0025 * 2 * TICK_S * ((40 + K0) ** 4 - (AMBIENT + K0) ** 4)
    assert tmr is None and abs(loss[0] / want - 1.0) < 1e-12


# ------------------------------------------------ the tick law in a room ----
def test_a_sleeper_in_a_shut_room_books_no_radiation_to_shed_and_the_books_close():
    w, ps = scene_inn_body(nx=24, ny=20, nz=32, wall=3, n_sleepers=1, settle=60)
    assert SR.room_of(w) is not None
    sink = w.airsink
    sink.flush()
    t0, s0 = w.total_energy() + sink.latent_J, float(w.shed)
    for _ in range(400):
        w.step()
    sink.flush()
    assert abs(w.shed - s0) < 1e-6                                   # nothing of the room's radiation leaves
    assert abs((w.total_energy() + sink.latent_J) - t0) < 1.0
    air = (w.mat == AIR)
    assert float(w.E[~air & ~w._star_bodies()].sum()) > 100.0        # the walls received it


def test_the_old_tick_law_is_one_attribute_away_and_does_leak():
    """MUTATION 2: the body's loss left to the fixed 20 C world (shut_room_star False) books to shed."""
    w, ps = scene_inn_body(nx=24, ny=20, nz=32, wall=3, n_sleepers=1, settle=60)
    w.shut_room_star = False
    s0 = float(w.shed)
    for _ in range(200):
        w.step()
    assert w.shed - s0 > 100.0


# ------------------------------------------------- tick against the jump ---
def _facts():
    B = TW.BUDGETS["body_heat"]

    def air_j(w):
        return float(w.E[(w.mat == AIR)].astype(np.float64).sum())

    def wall_j(w):
        return float(w.E[(w.mat != AIR) & ~w._star_bodies()].astype(np.float64).sum())
    return [H.Fact("air_J", air_j, "change", *B["air_J"]),
            H.Fact("wall_J", wall_j, "change", *B["wall_J"]),
            H.Fact("shed_J", lambda w: float(w.shed), "change", *B["shed_J"]),
            H.Fact("air_rise_K", lambda w: w.airsink.mean_temp(), "change", *B["air_rise_K"]),
            H.Fact("core_C", lambda w: w.persons[0]["heat"]["T_pool"], "level", *B["core_C"])]


def _build():
    return scene_inn_body(nx=24, ny=20, nz=32, wall=3, n_sleepers=1, settle=60)[0]


def _jump(w, n):
    rep = ST.advance(w, n, approx=True, sched=w.slow, max_fine_ticks=3000)
    assert rep.tier_used == "reduced" and rep.jumped > 0.3 * n


def _books(w):
    w.airsink.flush()
    return w.total_energy() + w.airsink.latent_J


def test_the_harness_reports_the_star_tick_against_jump_inside_the_twins_budget():
    rep = H.compare("shut-room star, 1 sleeper, 45 s", _build, int(45.0 / TICK_S), _jump, _facts(),
                    grade=H.APPROX, books=_books, books_tol=1.0)
    print("\n" + rep.table())
    rep.assert_ok()
    assert rep.worst is not None and rep.speedup > 1.5
    assert TW.grade_of(TW.TWINS["body_heat"]) == TW.APPROX and H.contract(TW.TWINS["body_heat"]) == []


def test_the_jump_runs_the_star_and_books_no_shed():
    w = _build()
    ok, info = N.advance_region(w, int(30.0 / TICK_S), w.slow, sources=[], skip=np.concatenate(
        [np.asarray(p["heat"]["cells"]) for p in w.persons]))
    assert ok and info["star"] and info["star_why"] == ""
    # the net's boundary book is the conduction to the ring only; the star made and lost no joule
    assert abs(info["boundary_J"]) < 1.0


def test_MUTATION_the_radiative_share_back_in_the_air_is_caught():
    """MUTATION 1: the body's radiation given to the AIR again (the old behaviour) in both paths: the where
    columns no longer agree, and the harness says so."""
    real = BodyHeatMixin._star_credit
    BodyHeatMixin._star_credit = lambda self, h, ent, J: False
    try:
        rep = H.compare("mutant", _build, int(45.0 / TICK_S), _jump, _facts(), grade=H.APPROX)
    finally:
        BodyHeatMixin._star_credit = real
    # tick: the ray correction goes to the air but the lattice's own radiation still goes to the walls through
    # the star; jump: ALL of the body's radiation goes to the air. They disagree on the walls.
    assert not rep.ok, rep.table()


def test_a_region_that_does_not_hold_the_whole_room_is_not_a_star():
    w = _build()
    box = tuple(slice(0, s // 2) for s in w.shape)
    res = N.thermal_net(w, box)
    assert res.rad_star is None and "wholly inside" in res.star_why


# ------------------------------------------------------------ the harness ---
class _Toy:
    def __init__(self):
        self.x, self.books = 0.0, 10.0
        self.tick = 0

    def step(self):
        self.x += 1.0
        self.tick += 1


def test_the_harness_catches_a_coarse_version_that_drifts_and_one_that_loses_a_book():
    facts = [H.Fact("x", lambda w: w.x, "level", rel=1e-9)]
    good = H.compare("toy", _Toy, 100, lambda w, n: setattr(w, "x", w.x + n), facts, grade=H.EXACT,
                     books=lambda w: w.books, state_diff=lambda a, b: [] if a.x == b.x else ["x"])
    assert good.ok and "OK" in good.table()
    bad = H.compare("toy", _Toy, 100, lambda w, n: setattr(w, "x", w.x + n * 1.01), facts, grade=H.EXACT)
    assert not bad.ok and bad.worst.name == "x" and "OUT" in bad.table()

    def leaky(w, n):
        w.x += n
        w.books -= 5.0
    lost = H.compare("toy", _Toy, 100, leaky, facts, books=lambda w: w.books, books_tol=1.0)
    assert not lost.ok and lost.books["coarse"] == -5.0
    ex = H.compare("toy", _Toy, 10, lambda w, n: setattr(w, "x", w.x + n), facts, grade=H.EXACT,
                   state_diff=lambda a, b: ["tick: %d != %d" % (a.tick, b.tick)])
    assert not ex.ok and ex.state_diff                           # an exact twin must leave the same state


def test_contract_names_what_a_twin_leaves_out():
    class Bare(TW.Twin):
        name = "bare"
        grade = TW.APPROX
    probs = H.contract(Bare())
    assert any("budget" in p for p in probs)
    Bare.budget = {"x": (0.1, 0.0)}
    assert H.contract(Bare()) == []
    assert any("missing" in p for p in H.contract(object()))


# -------------------------------------------- the breath twin, EXACT --------
def test_the_harness_proves_the_breath_twin_exact_against_real_ticks():
    from tests.test_slowtwins import _person_in, _room, _warm

    def build():
        w = _room()
        _person_in(w, awake=True, asleep=True, blood_o2=0.9)
        _warm(w, 40)
        return w

    def jump(w, n):
        p = w.persons[0]
        twins, why = TW.person_twins(w, p)
        assert twins is not None, why
        for _nm, tw, ctx in twins:
            assert H.contract(tw) == []
            if _nm == "breath":
                assert TW.grade_of(tw) == TW.EXACT
            tw.advance(w, p, ctx, n)
        w.tick += n

    from src.voxel import quiet as Q
    B = TW.TWINS["breath"].budget
    facts = [H.Fact(k, lambda w, k=k: w.persons[0][k], "level", *B[k]) for k in TW.TWINS["breath"].keys]
    rep = H.compare("breath twin, a sleeper, 380 ticks", build, 380, jump, facts, grade=H.EXACT,
                    state_diff=lambda a, b: Q.diff(Q.digest(a), Q.digest(b)))
    print("\n" + rep.table())
    rep.assert_ok()
    assert max(r.drift_rel for r in rep.rows) < 1e-9
