"""pathlaw.py: PathingMixin mixed into a World subclass (T11 spec v2, T-W9,
T-W10, T-W11 (the stepping half) and T-W12).

The mixin is tested by mixing it in HERE, so phase B changes only the class
list. Tags: [I] exactness against the unmodified World, [I] closed forms for
the stair books, [C] code against its own table.
"""
import numpy as np
import pytest

from src.voxel import routing, scenes, stairs
from src.voxel.materials import AIR, STONE
from src.voxel.motion import FlatRoutesMixin
from src.voxel.pathlaw import PathingMixin
from src.voxel.sim import World


class PathWorld(World):
    """World itself: motion.MotionMixin has PathingMixin first since 13a (B7).
    The step range is ON in a plain World since C9 (0.30 m); these tests are
    about the range as a parameter, so this world starts at 0 and each test
    sets what it wants."""
    step_up_m = 0.0
    step_down_m = 0.0


class OldWorld(World):
    """The base18 planner: the flat fit and routes, with PathingMixin out of the
    way. The reference every 'bit-identical' test below compares against."""
    step_up_m = 0.0
    step_down_m = 0.0
    _fit_grid = FlatRoutesMixin._fit_grid
    _routes = FlatRoutesMixin._routes


class NoDepositWorld(PathWorld):
    """A world with no deposit() road (World has one since wave 13a, from LedgersMixin):
    a descent then keeps its heat on the person."""
    deposit = None


class DepositWorld(PathWorld):
    """The books a descent needs, in the smallest form: one deposit() that adds
    joules to cells and counts them, as the mechanisms mixin's does."""
    mech_in = 0.0

    def deposit(self, cells, J, actor=None):
        c = tuple(np.asarray(cells).T) if isinstance(cells, np.ndarray) else cells
        n = len(c[0])
        self.E[c] += np.float32(J / n)
        self.mech_in += J


def _person_cells(w, p):
    comp, sl = w._person_cells(p)
    return w._claim_cells(p, comp, sl)


# ── random rooms ────────────────────────────────────────────────────────────
def _room(cls, seed):
    r = np.random.default_rng(seed)
    w = cls(36, 28, 34, voxel_cm=5)
    w.fill(0, 36, 0, 28, 0, 1, STONE)
    w.fill(0, 36, 0, 1, 1, 33, STONE)
    w.fill(0, 36, 27, 28, 1, 33, STONE)
    w.fill(0, 1, 0, 28, 1, 33, STONE)
    w.fill(35, 36, 0, 28, 1, 33, STONE)
    for _ in range(4):
        x0, y0 = int(r.integers(14, 30)), int(r.integers(3, 24))
        if r.random() < .5:
            w.fill(x0, x0 + int(r.integers(2, 9)), y0, y0 + 1, 1, 33, STONE)
        else:
            w.fill(x0, x0 + 1, y0, y0 + int(r.integers(2, 9)), 1, 33, STONE)
    scenes._person(w, 8, 10)
    w.step()
    return w


def _decisions(w, seed, n):
    """n legs decisions: the whole menu, then the body walks a few cells down a
    row's route (flat `_walk`, the same in both worlds). Returns the menus."""
    r = np.random.default_rng(seed)
    p = w.persons[0]
    out = []
    for _ in range(n):
        cells = _person_cells(w, p)
        rows = w._menu(p, cells, None, "legs", {}, cap=10 ** 6)
        out.append(rows)
        go = [x for x in rows if x.get("route")]
        if not go:
            continue
        row = go[int(r.integers(0, len(go)))]
        ax = int(round(float(cells[:, 0].mean())))
        ay = int(round(float(cells[:, 1].mean())))
        for (px, py) in row["route"][:6]:
            sx, sy = int(np.sign(px - ax)), int(np.sign(py - ay))
            if (sx or sy) and w._walk(cells, sx, sy):
                cells = cells + np.array([sx, sy, 0], cells.dtype)
                w._carry_own(p, sx, sy)
                ax, ay = ax + sx, ay + sy
    return out


def test_with_step_range_0_menu_rows_and_routes_are_bit_identical_over_20_worlds():
    n_routes = 0
    for seed in range(20):
        a, b = _room(OldWorld, seed), _room(PathWorld, seed)
        assert b._step_range(b.persons[0]) == (0, 0)
        ma, mb = _decisions(a, seed, 40), _decisions(b, seed, 40)
        assert ma == mb, seed
        n_routes += sum(1 for rows in ma for x in rows if x.get("route"))
        assert (a.mat == b.mat).all() and (a.eid == b.eid).all()
    assert n_routes > 400                     # the menus were real


def test_with_fast_sweep_off_it_is_super_itself_and_the_same_rows():
    a, b = _room(OldWorld, 3), _room(PathWorld, 3)
    b.fast_sweep = False
    assert _decisions(a, 3, 12) == _decisions(b, 3, 12)


def test_with_a_range_but_no_ledge_the_menu_rows_are_also_identical():
    for seed in range(4):
        a, b = _room(OldWorld, seed), _room(PathWorld, seed)
        b.persons[0]["step_up_m"] = b.persons[0]["step_down_m"] = 0.30
        assert b._step_range(b.persons[0]) == (6, 6)
        ma, mb = _decisions(a, seed, 12), _decisions(b, seed, 12)
        assert ma == mb, seed


def test_the_range_is_read_from_the_person_then_the_world_then_BODY(monkeypatch):
    w = PathWorld(10, 10, 10, voxel_cm=5)
    p = {}
    assert w._step_range(p) == (0, 0)
    w.step_up_m, w.step_down_m = 0.30, 0.15
    assert w._step_range(p) == (6, 3)
    p["step_up_m"] = 0.10
    assert w._step_range(p) == (2, 3)
    w10 = PathWorld(10, 10, 10, voxel_cm=10)
    w10.step_up_m = 0.30
    assert w10._step_range({})[0] == 3                         # d = 10 cm


# ── a body that can step ────────────────────────────────────────────────────
def _kerb_room(height, cls=PathWorld, side="right"):
    """A floor with a step `height` cells high across x >= 20, a person at x = 14."""
    w = cls(48, 24, 40, voxel_cm=5)
    w.fill(0, 48, 0, 24, 0, 1, STONE)
    w.fill(20, 48, 0, 24, 1, 1 + height, STONE)
    scenes._person(w, 14, 12)
    w.step()
    p = w.persons[0]
    p["step_up_m"] = p["step_down_m"] = 0.30
    return w, p


def _feet_supported(w, cells):
    zlo = int(cells[:, 2].min())
    low = cells[cells[:, 2] == zlo]
    return bool((w.mat[low[:, 0], low[:, 1], zlo - 1] != AIR).any())


def test_a_15_cm_kerb_is_walked_up_and_down_with_the_soles_supported_each_step():
    w, p = _kerb_room(3)
    cells = _person_cells(w, p)
    z0 = int(cells[:, 2].min())
    ups = 0
    for _ in range(12):
        cells, dz = w.step_along(p, cells, 1, 0)
        assert dz is not None, "should walk onto the kerb"
        assert _feet_supported(w, cells)
        ups += dz
    assert ups == 3 and int(cells[:, 2].min()) == z0 + 3
    for _ in range(12):
        cells, dz = w.step_along(p, cells, -1, 0)
        assert dz is not None and _feet_supported(w, cells)
    assert int(cells[:, 2].min()) == z0


def test_a_40_cm_wall_is_refused_and_the_body_does_not_climb_its_own_shins():
    w, p = _kerb_room(8)
    cells = _person_cells(w, p)
    steps = 0
    while steps < 20:
        cells2, dz = w.step_along(p, cells, 1, 0)
        if dz is None:
            break
        cells = cells2
        steps += 1
    assert dz is None
    assert int(cells[:, 2].min()) == 1                           # still on the floor
    assert _feet_supported(w, cells)
    # and with R_up 6 the planner knows it too: no route over a 40 cm wall
    up = w._routes(cells, [(40, 12)], p)
    assert up[(40, 12)] is None or all(x < 20 for x, _y in up[(40, 12)])
    # the base walk is the base walk: _walk is not overridden
    assert PathingMixin._walk is World._walk if hasattr(PathingMixin, "_walk") else True
    assert "_walk" not in PathingMixin.__dict__ and "_walkable" not in PathingMixin.__dict__


def test_a_step_up_under_a_low_ceiling_is_refused_by_the_arriving_cells():
    w = PathWorld(48, 24, 40, voxel_cm=5)
    w.fill(0, 48, 0, 24, 0, 1, STONE)
    w.fill(20, 48, 0, 24, 1, 4, STONE)                          # a 15 cm kerb
    w.fill(18, 48, 0, 24, 32, 34, STONE)                        # a ceiling at z 32: the head hits at 3 up
    scenes._person(w, 14, 12)
    w.step()
    p = w.persons[0]
    p["step_up_m"] = p["step_down_m"] = 0.30
    cells = _person_cells(w, p)
    for _ in range(10):
        cells, dz = w.step_along(p, cells, 1, 0)
        if dz is None:
            break
    assert dz is None
    assert int(cells[:, 2].min()) == 1


def test_a_40_cm_drop_is_not_stepped_but_still_shows_its_landing_and_fall_energy():
    w = PathWorld(48, 24, 40, voxel_cm=5)
    w.fill(0, 48, 0, 24, 0, 1, STONE)
    w.fill(0, 20, 0, 24, 1, 9, STONE)                           # a 40 cm ledge, person on it
    scenes._person(w, 14, 12, z0=9)
    w.step()
    p = w.persons[0]
    p["step_up_m"] = p["step_down_m"] = 0.30
    cells = _person_cells(w, p)
    for _ in range(12):                                         # to the lip
        cells2, dz = w.step_along(p, cells, 1, 0)
        if dz is None:
            break
        cells = cells2
    assert dz is None                                           # not an edge the legs take
    z_before = int(cells[:, 2].min())
    # the planner does not offer the lower floor either
    r = w._routes(cells, [(30, 12)], p)
    assert r[(30, 12)] is None or all(x < 20 for x, _y in r[(30, 12)])
    # a body that is PUSHED off falls by the existing laws: a landing event
    for _ in range(3):
        if not w._walk(cells, 1, 0):
            break
        cells = cells + np.array([1, 0, 0], cells.dtype)
        w._carry_own(p, 1, 0)
    for _ in range(40):
        w.step()
    verbs = {e.verb for e in w.events.rows}
    assert "lands" in verbs, verbs
    cells = _person_cells(w, p)
    assert int(cells[:, 2].min()) < z_before


def test_codeonly_the_step_cost_table_prices_a_rise_by_the_time_it_takes():
    w = PathWorld(10, 10, 10, voxel_cm=5)
    F = w._step_ftab(6)
    assert F[0] == 8 and list(F) == stairs.step_table(5.0, 6)
    assert routing.edge_cost(70, 0, F) == 70 and routing.edge_cost(99, 0, F) == 99
    assert routing.edge_cost(70, 4, F) == 70 * F[4] // 8 > 70 * 10


# ── T-W11 (the stepping half): a 12-step flight, books closed ───────────────
def _flight_room(cls=DepositWorld):
    w = cls(90, 20, 90, voxel_cm=5)
    w.fill(0, 90, 0, 20, 0, 1, STONE)
    for i in range(12):
        w.fill(14 + 5 * i, 90, 0, 20, 1, 1 + 4 * (i + 1), STONE)
    scenes._person(w, 8, 10)
    w.step()
    p = w.persons[0]
    p["step_up_m"] = p["step_down_m"] = 0.30
    return w, p


def _walk_to(w, p, cells, tx, max_steps=400):
    for _ in range(max_steps):
        ax = int(round(float(cells[:, 0].mean())))
        if ax == tx:
            return cells
        cells, dz = w.step_along(p, cells, int(np.sign(tx - ax)), 0)
        assert dz is not None, "refused at x=%d" % ax
    raise AssertionError("did not arrive")


def test_a_12_step_flight_is_climbed_and_descended_with_the_books_closed():
    w, p = _flight_room()
    cells = _person_cells(w, p)
    m_kg = float(w.smass[tuple(cells.T)].sum()) / 1000.0
    d = 0.05
    z0 = int(cells[:, 2].min())
    mech0, e0 = w.mech_in, w.total_energy()
    ax0 = int(round(float(cells[:, 0].mean())))
    n_steps = 14 + 5 * 12 + 4 - ax0
    cells = _walk_to(w, p, cells, 14 + 5 * 12 + 4)               # the top
    rise = int(cells[:, 2].min()) - z0
    assert rise == 48
    b = p["_stairs"]
    # time: each riser step takes max(d/v, rise/vz); the 12 risers of 4 cells
    mass_kg = m_kg
    t_riser = stairs.step_time(4 * d, d, mass_kg)
    t_flat = stairs.step_time(0.0, d, mass_kg)
    n_flat = n_steps - 12
    assert b["time_s"] == pytest.approx(12 * t_riser + n_flat * t_flat, rel=1e-6)
    # a climb is NOT on the energy ledger: potential is off it
    assert w.mech_in == mech0 and w.total_energy() == pytest.approx(e0, rel=1e-9)
    # metabolic work: m g h / 0.22 plus the level cost over the run
    run = n_steps * d
    want = m_kg * 9.81 * (48 * d) / 0.22 + 2.5 * m_kg * run
    assert b["metabolic_J"] == pytest.approx(want, rel=1e-6)
    assert p["_work_J"] == pytest.approx(want, rel=1e-6)
    # and back down: delta total_energy == delta mech_in
    e1, m1 = w.total_energy(), w.mech_in
    w_before = p["_work_J"]
    cells = _walk_to(w, p, cells, 6)
    assert int(cells[:, 2].min()) == z0
    de, dm = w.total_energy() - e1, w.mech_in - m1
    assert dm == pytest.approx(m_kg * 9.81 * 48 * d, rel=1e-6)   # the potential lost
    assert de == pytest.approx(dm, rel=1e-4)
    assert p["_work_J"] > w_before


def test_a_descent_without_a_deposit_keeps_the_joules_on_the_person():
    w, p = _flight_room(NoDepositWorld)
    cells = _person_cells(w, p)
    cells = _walk_to(w, p, cells, 14 + 5 * 12 + 4)
    cells = _walk_to(w, p, cells, 6)
    m_kg = float(w.smass[tuple(cells.T)].sum()) / 1000.0
    assert p["_fall_heat_J"] == pytest.approx(m_kg * 9.81 * 48 * 0.05, rel=1e-6)


# ── T-W12: the belief ───────────────────────────────────────────────────────
def _loft_room():
    w = PathWorld(60, 24, 80, voxel_cm=5)
    w.fill(0, 60, 0, 24, 0, 1, STONE)
    for i in range(5):                                          # five risers of 2 cells, treads of 4
        w.fill(14 + 4 * i, 34, 4, 20, 1, 1 + 2 * (i + 1), STONE)
    w.fill(34, 52, 0, 24, 1, 11, STONE)                         # a landing (top at z 11)
    scenes._person(w, 8, 12)
    w.step()
    p = w.persons[0]
    p["step_up_m"] = p["step_down_m"] = 0.30
    return w, p


def test_a_body_that_has_not_seen_the_top_does_not_plan_to_it_until_it_looks():
    w, p = _loft_room()
    cells = _person_cells(w, p)
    nx, ny, _ = w.shape
    # it has looked at the room it stands in (x < 14), and nothing else
    seen = np.zeros((nx, ny), bool)
    seen[:14, :] = True
    p["known"][:] = True
    p["free"] = None                       # it has not looked yet
    p.pop("fz", None)                      # nor written what it could stand on
    w._see_levels(p, cells, reach=seen)
    assert p["fz"].shape == (2, nx, ny)
    assert p["free"][:14].all() and not p["free"][20:].any()
    goal = (42, 12)                                              # on the landing, unseen
    r = w._routes(cells, [goal], p)
    path = r[goal]
    assert path is None or path[-1][0] < 14, "no god-channel to the top"
    # now it looks at everything
    w._see_levels(p, cells, reach=np.ones((nx, ny), bool))
    r = w._routes(cells, [goal], p)
    assert r[goal] and r[goal][-1] == goal
    assert any(14 <= x < 34 for x, _y in r[goal])                 # up the flight
    # and the fit plane a reader sees (old shape: a bool plane)
    fit, start = w._fit_grid(cells, p)
    assert fit.dtype == bool and fit.shape == (nx, ny) and fit[goal]


def test_a_loft_above_the_floor_where_the_body_stands_is_a_separate_level():
    w = PathWorld(30, 20, 80, voxel_cm=5)
    w.fill(0, 30, 0, 20, 0, 1, STONE)
    w.fill(15, 30, 0, 20, 40, 41, STONE)                         # a loft slab over half the floor
    scenes._person(w, 6, 10)
    w.step()
    p = w.persons[0]
    cells = _person_cells(w, p)
    w._see_levels(p, cells, box=(0, 30, 0, 20))
    assert p["fz"][0, 22, 5] == 1 and p["fz"][1, 22, 5] == 41      # floor under, loft over
    assert p["fz"][0, 5, 5] == 1 and p["fz"][1, 5, 5] == -1
    assert p["free"][22, 5]


def test_see_levels_never_counts_the_bodys_own_flesh():
    w = PathWorld(30, 20, 60, voxel_cm=5)
    w.fill(0, 30, 0, 20, 0, 1, STONE)
    scenes._person(w, 10, 10)
    w.step()
    p = w.persons[0]
    cells = _person_cells(w, p)
    w._see_levels(p, cells, box=(0, 30, 0, 20))
    xs, ys = cells[:, 0], cells[:, 1]
    assert (p["fz"][0, xs, ys] == 1).all() and (p["fz"][1, xs, ys] == -1).all()
