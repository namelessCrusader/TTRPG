"""Surface layers on (wave 13c, lane C4, T3): the gate, soot deposition against its hand
formulas, the batch pool step against the one-face step, the emissivity read-out, the
books, snapshots, and one world with the other capabilities."""
import copy

import numpy as np
import pytest

from src.voxel import layer_laws as LL
from src.voxel.layer_data import AMBIENT, GRAIN, SOOT, SOOT_TRUE, pack_key
from src.voxel.layers import Layers, cell_key
from src.voxel.layers_world import D_SOOT, DEP_EVERY, RHO_SOOT, LayersMixin
from src.voxel.materials import AIR, STONE, TICK_S, WOOD
from src.voxel.scenes_layers import sooty_room
from src.voxel.sim import World


def _box(n=3, voxel_cm=10, layers=True):
    """A stone box with an n^3 air hollow."""
    w = World(n + 2, n + 2, n + 2, voxel_cm=voxel_cm)
    w.open_sky = False
    w.fill(0, n + 2, 0, n + 2, 0, n + 2, STONE)
    w.fill(1, n + 1, 1, n + 1, 1, n + 1, AIR)
    w.smass[1:n + 1, 1:n + 1, 1:n + 1] = 0.0
    if layers:
        w.enable_layers()
    return w


# ---- the gate --------------------------------------------------------------------
def test_the_gate_is_on_by_default_and_a_world_with_it_off_is_the_old_world():
    assert World.surface_laws is True and LayersMixin.surface_laws is True
    w = World(8, 8, 8, voxel_cm=10)
    for _ in range(12):
        w.step()
    assert w.layers is None                     # nothing is made until smoke settles
    w.surface_laws = False
    w.fill(0, 8, 0, 8, 0, 1, STONE)
    w.smoke[3, 3, 1] = 1.0
    for _ in range(12):
        w.step()
    assert w.layers is None and w.gone.get("smoke") is None
    v = World(8, 8, 8, voxel_cm=10)
    v.fill(0, 8, 0, 8, 0, 1, STONE)
    v.smoke[3, 3, 1] = 1.0
    for _ in range(12):
        v.step()
    assert v.layers is not None and v.gone.get("smoke", 0.0) > 0.0


def test_layers_on_without_soot_deposition_is_the_old_world_bit_for_bit_is_the_old_world_bit_for_bit():
    """A real fire (smoke, heat) in a world with the gate on but soot off: nothing is
    deposited, the pool stays empty, and every field equals the plain world's."""
    a = sooty_room(nx=12, ny=12, nz=8, layers=False)
    b = sooty_room(nx=12, ny=12, nz=8, soot=False)
    a.surface_laws = False
    for _ in range(400):
        a.step()
        b.step()
    assert float(a.smoke.sum()) > 0.0
    for name in ("E", "mat", "smass", "fvol", "smoke", "o2"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert b.layers.n == 0 and b.gone.get("smoke") == a.gone.get("smoke")


def test_a_fire_with_soot_on_differs_from_the_old_world_only_by_the_settled_smoke():
    a = sooty_room(nx=12, ny=12, nz=8, layers=False)
    a.surface_laws = False
    b = sooty_room(nx=12, ny=12, nz=8)
    for _ in range(400):
        a.step()
        b.step()
    assert b.layers.n > 0
    assert float(b.smoke.sum()) < float(a.smoke.sum())


# ---- the law against its hand formula ---------------------------------------------
def test_one_cell_decays_by_the_sum_of_its_six_face_speeds():
    w = _box(1)
    S0 = 4.0
    w.smoke[1, 1, 1] = S0
    dx = 0.1 * w.scale
    dt = DEP_EVERY * TICK_S
    v_s = LL.stokes_velocity(D_SOOT, RHO_SOOT)
    v_b = LL.V_BROWN
    # one air cell: floor (+v_s), ceiling (floored at 0), four walls
    k = (v_b + v_s + max(0.0, v_b - v_s) + 4 * v_b) / dx          # 1/s: sum(v A)/V
    n = 20
    for _ in range(n):
        w._soot_pass()
    expect = S0 * (1.0 - k * dt) ** n
    assert float(w.smoke[1, 1, 1]) == pytest.approx(expect, rel=1e-5)
    # the smoke field is float32; the two books are float64 and close on each other
    removed = w.soot_on_faces() / SOOT_TRUE
    assert w.gone["smoke"] == pytest.approx(removed * (1 - SOOT_TRUE), rel=1e-9)
    assert removed == pytest.approx(S0 - float(w.smoke[1, 1, 1]), rel=1e-3)
    assert w.layers.ledger_closes()


def test_in_cool_air_the_floor_gets_most_a_wall_less_and_a_ceiling_none():
    w = _box(1)
    w.smoke[1, 1, 1] = 4.0
    w._soot_pass()
    L = w.layers
    sh = w.shape
    floor = L.mass(cell_key(1, 1, 0, 4, sh))      # the floor cell's up face
    wall = L.mass(cell_key(0, 1, 1, 0, sh))       # the -x wall cell's +x face
    ceil = L.mass(cell_key(1, 1, 2, 5, sh))       # the ceiling cell's down face
    v_s = LL.stokes_velocity(D_SOOT, RHO_SOOT)
    assert v_s > LL.V_BROWN                      # 0.5 um soot settles faster than it diffuses
    assert floor > wall > 0.0 and ceil == 0.0    # a ceiling needs the heat (next test)
    assert floor / wall == pytest.approx((LL.V_BROWN + v_s) / LL.V_BROWN, rel=1e-6)


def test_hot_gas_over_a_cold_ceiling_deposits_by_thermophoresis():
    w = _box(1)
    C = w.heat_capacity()
    w.E[1, 1, 1] = 300.0 * C[1, 1, 1]                         # 320 C air, cold walls
    w.smoke[1, 1, 1] = 4.0
    T = w.T()
    v_th = float(LL.thermophoretic_velocity(float(T[1, 1, 1]), float(T[1, 1, 2])))
    assert v_th > 1e-4                                         # the spec's 2.5e-4 scale
    s0 = float(w.smoke[1, 1, 1])
    w._soot_pass()
    dx = 0.1 * w.scale
    dt = DEP_EVERY * TICK_S
    conc = s0 * 1000.0 / w.vox_l
    v_s = LL.stokes_velocity(D_SOOT, RHO_SOOT)
    want = max(0.0, LL.V_BROWN + v_th - v_s) * conc * dx * dx * dt * SOOT_TRUE
    got = w.layers.mass(cell_key(1, 1, 2, 5, w.shape))
    assert got == pytest.approx(want, rel=1e-5)


def test_a_pass_never_takes_more_than_half_a_cells_smoke():
    w = _box(1)
    w.layer_every = 10 ** 6                                    # a huge dt: the cap must hold
    w.smoke[1, 1, 1] = 1.0
    w._soot_pass()
    assert float(w.smoke[1, 1, 1]) == pytest.approx(0.5, rel=1e-6)


def test_smoke_inside_a_moving_body_does_not_settle_on_it():
    w = _box(1)
    w.moving[1, 1, 0] = True
    w.smoke[1, 1, 1] = 1.0
    w._soot_pass()
    assert w.layers.mass(cell_key(1, 1, 0, 4, w.shape)) == 0.0


# ---- the batch step equals the one-face step ----------------------------------------
def test_batch_deposit_equals_the_one_face_deposit_new_and_merged():
    a, b = Layers(dx=0.05), Layers(dx=0.05)
    keys = np.array(sorted(cell_key(x, 2, 3, f, (9, 9, 9)) for x in range(4) for f in (0, 4, 5)),
                    np.int64)
    rng = np.random.default_rng(3)
    for rnd in range(3):                                       # round 0 is new, 1 and 2 merge
        g = rng.uniform(1e-5, 4e-3, len(keys))
        for k, gi in zip(keys, g):
            a.add(int(k), {(SOOT, GRAIN, 1): float(gi)}, T_in=AMBIENT, Ts=AMBIENT, source="smoke")
        b.deposit_soot_batch(keys, g)
    for k in keys:
        sa, sb = a._find(int(k)), b._find(int(k))
        assert sa >= 0 and sb >= 0
        for f in ("m", "cover", "por", "adh", "coh"):
            assert getattr(a, f)[sa] == pytest.approx(float(getattr(b, f)[sb]), rel=1e-6), f
    assert a.total(SOOT) == pytest.approx(b.total(SOOT), rel=1e-12)
    assert b.ledger_closes() and a.n == b.n


def test_batch_onto_a_non_soot_top_goes_through_add():
    from src.voxel.layer_data import WATER
    L = Layers(dx=0.05)
    k = cell_key(1, 1, 1, 4, (5, 5, 5))
    L.add(k, {(WATER, 0, 0): 0.01}, T_in=AMBIENT, Ts=AMBIENT)
    L.deposit_soot_batch(np.array([k], np.int64), np.array([1e-3]))
    assert L.total(SOOT) == pytest.approx(1e-3, rel=1e-12) and L.ledger_closes()


# ---- the emissivity read-out ----------------------------------------------------------
def test_a_sooty_face_radiates_toward_soot_in_proportion_to_cover():
    w = _box(1)
    sh = w.shape
    sl = (slice(None),) * 3
    eps0 = w._emiss_table()[w.mat]
    assert np.array_equal(w._eps_over(eps0, sl), eps0)         # no layer: untouched
    key = cell_key(1, 1, 2, 5, sh)
    w.layers.deposit_soot_batch(np.array([key], np.int64), np.array([0.2]))
    cov = float(w.layers.cover[w.layers._find(key)])
    assert cov > 0.99
    e = w._eps_over(eps0, sl)
    sub = float(eps0[1, 1, 2])
    # one open face on that host (the other five touch stone): the whole cover counts
    assert float(e[1, 1, 2]) == pytest.approx(sub + (0.95 - sub) * cov, rel=1e-4)
    assert float(e[1, 1, 0]) == float(eps0[1, 1, 0])           # a clean cell keeps its own
    assert np.array_equal(eps0, w._emiss_table()[w.mat])       # the table was not written


# ---- books, snapshots, composition ---------------------------------------------------
def test_the_books_name_the_soot_and_a_snapshot_round_trips():
    w = _box(1)
    w.smoke[1, 1, 1] = 4.0
    for _ in range(8):
        w._soot_pass()
    b = w.books()
    assert any(k.startswith(f"{SOOT}:") for k in b["layer_mass"]["led_in"])
    snap = w.snapshot()
    ref = copy.deepcopy(w.layers.m[:w.layers.hi])
    w._soot_pass()
    w.restore(snap)
    assert np.array_equal(w.layers.m[:w.layers.hi], ref)
    w._soot_pass()
    assert w.layers.ledger_closes()


def test_one_world_with_cold_body_heat_and_layers_runs_and_closes():
    from src.voxel import climate as K
    from src.voxel.bodyworld import BodyWorld
    n = 24
    w = BodyWorld(n, n, 16, voxel_cm=10)
    w.fill(0, n, 0, n, 0, 1, STONE)
    w.fill(4, 20, 4, 20, 1, 8, STONE)
    w.fill(5, 19, 5, 19, 1, 7, AIR)
    w.smass[5:19, 5:19, 1:7] = 0.0
    w.open_sky = False
    w.set_climate(K.Climate.reference(t_c=5.0), soak=True)
    w.enable_layers()
    w.smoke[10, 10, 5] = 3.0
    e0 = w.total_energy()
    for _ in range(80):
        w.step()
    assert w.soot_on_faces() > 0.0 and w.layers.ledger_closes()
    assert abs(w.total_energy() - e0) < 1e-4 * (abs(e0) + float(np.abs(w.E).sum()) + 1.0)


# ---- the proof scene (P2) ---------------------------------------------------------------
def test_P2_a_burning_room_blackens_its_ceiling_and_the_books_close():
    """T3 spec P2 on a 2 x 2 x 1 m room (the 5 x 5 x 2.5 m room is the same law, 40 times
    the cost): the soot is the smoke the sim made, taken out at the law's own speeds."""
    ticks = 3600
    w = sooty_room(nx=20, ny=20, nz=10)
    for _ in range(ticks):
        w.step()
    L = w.layers
    assert L.n > 0 and L.ledger_closes()
    soot = w.soot_on_faces()
    removed = soot / SOOT_TRUE
    # every unit the air gave up is on a face or in the gone-book, to rounding
    assert w.gone["smoke"] / (1 - SOOT_TRUE) == pytest.approx(removed, rel=1e-9)
    # the share of all the smoke ever made that settled lies inside the well-mixed band.
    # A source running for T seconds makes smoke of mean age T/2, so the share is about
    # k T / 2 with k = v_d A_s / V. Lowest speed: Brownian only (1e-5 m/s). Highest: the
    # hot-layer value of the spec (2.8e-4 m/s). The factor 4 under the low end is for a
    # source that grows, which makes the smoke younger than T/2.
    sh = w.shape
    dx = 0.1 * w.scale
    inner = np.array([w.shape[0] - 2, w.shape[1] - 2, w.shape[2] - 2], float) * dx
    V = float(np.prod(inner))
    A_s = 2.0 * (inner[0] * inner[1] + inner[0] * inner[2] + inner[1] * inner[2])
    T = ticks * TICK_S
    made = float(w.smoke.sum(dtype=np.float64)) + removed
    share = removed / made
    lo = 0.25 * LL.V_BROWN * A_s / V * T / 2
    hi = 2.8e-4 * A_s / V * T / 2
    assert lo < share < hi, (lo, share, hi)
    # ceiling blacker than floor
    keys = L.keys()
    z = (keys >> 3) % sh[2]
    face = keys & 7
    cov = L.cover[L.lookup(keys)]
    ceil = cov[(face == 5) & (z == sh[2] - 1)]
    floor = cov[(face == 4) & (z == 0)]
    assert ceil.size and ceil.max() > 0.0
    assert floor.size == 0 or ceil.mean() > floor.mean()
    # the face law reads it: a sooty ceiling cell radiates more than clean stone
    sl = (slice(None),) * 3
    e0 = w._emiss_table()[w.mat]
    e1 = w._eps_over(e0, sl)
    assert e1.max() > e0.max() - 1e-9 and (e1 != e0).any()
    # a cloth wipes a patch (soot adhesion 100 Pa, cohesion 50 Pa; a cloth shears at 5 kPa)
    k = int(keys[(face == 5)][np.argmax(L.m[L.lookup(keys[(face == 5)])])])
    before = L.mass(k)
    cloth = (1 << 58) | (7 << 3)
    got, _adj = L.wipe(k, 5000.0, cloth)
    assert got == pytest.approx(before, rel=1e-9) and L.mass(k) == 0.0
    assert L.mass(cloth, SOOT) == pytest.approx(before, rel=1e-9) and L.ledger_closes()
