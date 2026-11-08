"""mechlaw.py: MechanismsMixin mixed into a World subclass (T11 spec v2, tests
T-W1 .. T-W8, T-W11 and the proof-scene orderings of P1/P6/P10).

The mixin is tested by mixing it in HERE (no sim.py edit). Tags: [I] closed
forms, conservation and exact orders; [C] code against its own table; [U]
physical numbers whose source is unverified (the kick counts are a REPORT for
the user, not a gate: only their ORDER is asserted).
"""
import math

import numpy as np
import pytest

from src.voxel import mechanisms as M
from src.voxel import scenes
from src.voxel.materials import AIR, ASH, STONE, TICK_S, TIN, WATER, WOOD
from src.voxel.mechlaw import MechanismsMixin
from src.voxel.pathlaw import PathingMixin
from src.voxel.policy import TablePolicy
from src.voxel.sim import World

try:
    from src.voxel import locks as L
except ImportError:                                     # pragma: no cover
    L = None


class DoorWorld(World):        # MechanismsMixin is in World since the M3 merge
    pass


class _Stay(TablePolicy):
    """Index 0 of every menu is the null act: nobody in these rooms moves
    unless a test moves them."""
    def pick(self, situation, menu):
        return 0


class WalkDoorWorld(World):     # World has PathingMixin and MechanismsMixin since 13a
    pass


NX, NY, NZ = 50, 40, 50
WALL_Y = 25


def _room(cls=DoorWorld, voxel_cm=5, W=20, H=40, **door):
    """A stone wall along x at y = 25 with a door (hinge at (10, 25), leaf W
    cells wide) swinging into the room south of it. The person (asleep: the
    will layer does not move them) stands in the south room."""
    w = cls(NX, NY, NZ, voxel_cm=voxel_cm)
    w.fill(0, NX, 0, NY, 0, 1, STONE)
    w.fill(0, NX, WALL_Y, WALL_Y + 1, 1, 45, STONE)
    w.fill(9, 11 + W, WALL_Y, WALL_Y + 1, 1, 45, AIR)
    door.setdefault("heavy", voxel_cm > 5)
    mid = w.add_door((10, WALL_Y), W, H, swing=-1, along=(1, 0), corner=(-1, -1),
                     **door)
    w.policy = _Stay()
    scenes._person(w, 38, 6)
    w.step()
    return w, mid, w.persons[0]


def _leaf(w, rec):
    return {tuple(c) for c in w._cells_of(rec["leaf"]).tolist()}


def _run_until_still(w, rec, n=200):
    for i in range(n):
        w.step()
        if rec["swing"] is None and i > 1:
            return i
    raise AssertionError("the swing never ended")


def _run_until_open(w, mid, n=200):
    for i in range(n):
        w.step()
        if w.is_open(mid):
            return i
    raise AssertionError("the door never opened")


def _verbs(w, verb):
    return [e for e in w.events.rows if e.verb == verb]


# ── T-W1: add_door ──────────────────────────────────────────────────────────
def test_add_door_registers_leaf_and_frame_and_the_record_has_no_mass():
    w, mid, _p = _room()
    rec = w.mechs[mid]
    assert w.things[rec["leaf"]]["mech"] == mid == w.things[rec["frame"]]["mech"]
    assert rec["leaf"] != rec["frame"]
    # a plain JSON-safe dict: geometry and state, nothing a law computes
    import json
    json.dumps(rec)
    assert not {"mass", "mass_kg", "inertia", "I", "com"} & set(rec)
    assert not any("mass" in k.lower() for k in rec["joints"][0])
    leaf = np.array(sorted(_leaf(w, rec)))
    assert len(leaf) == 20 * 40
    assert leaf[:, 1].min() == leaf[:, 1].max() == WALL_Y        # one cell thick
    # no flag says LOCKED or BROKEN
    for bad in ("locked", "broken", "is_locked", "jammed_flag"):
        assert bad not in rec
    # the frame is wood cells round the opening
    assert w.mat[9, WALL_Y, 5] == WOOD and w.mat[30, WALL_Y, 5] == WOOD
    assert w.mat[15, WALL_Y, 41] == WOOD                          # the head


def test_the_fill_is_the_authored_frac_or_derived_from_real_clearances():
    w, mid, _p = _room(frac=0.97)
    assert w.mechs[mid]["frac"] == 0.97
    cell = (15, WALL_Y, 10)
    assert w.smass[cell] == pytest.approx(600.0 * 0.97 * w.vox_l, rel=1e-6)
    w2, mid2, _ = _room()
    f = w2.mechs[mid2]["frac"]
    Wm, Hm = 20 * 0.05, 40 * 0.05                                  # a 1.0 x 2.0 m leaf
    area = (2 * 3.0 * Hm + 3.0 * Wm + 12.0 * Wm) / 1000.0
    assert f == pytest.approx(1.0 - area / (Wm * Hm), rel=1e-12)
    # for the 0.9 x 2.03 m leaf of the spec: 0.986
    Wm, Hm = 0.9, 2.03
    assert 1.0 - (2 * 3.0 * Hm + 3.0 * Wm + 12.0 * Wm) / 1000.0 / (Wm * Hm) \
        == pytest.approx(0.986, abs=0.001)


def test_add_door_raises_at_10_cm_voxels_unless_heavy():
    with pytest.raises(ValueError, match="heavy"):
        DoorWorld(30, 30, 30, voxel_cm=10).add_door((5, 5), 9, 20)
    w = DoorWorld(30, 30, 30, voxel_cm=10)
    w.fill(0, 30, 0, 30, 0, 1, STONE)
    mid = w.add_door((5, 15), 9, 20, heavy=True)                   # accepted
    assert mid in w.mechs and w.mechs[mid]["voxel_cm"] == 10.0


def test_the_record_and_its_things_round_trip_snapshot_and_restore():
    w, mid, _p = _room(deadbolt=True, bar=True)
    snap = w.snapshot()
    import copy
    before = copy.deepcopy(w.mechs)
    things = copy.deepcopy(w.things)
    rec = w.mechs[mid]
    M.blow_path([w._mlink(rec, "latch.split")], 2000.0)            # change the state
    w.mechs[mid]["joints"][0]["theta"] = 1.0
    w.restore(snap)
    assert w.mechs == before and w.things == things
    assert "bar" in w.mechs[mid] and w.mechs[mid]["bar"] is not None


# ── the energy books: one flow ──────────────────────────────────────────────
def test_deposit_books_the_actor_or_mech_in_never_both():
    w, mid, p = _room()
    c = np.array([[15, 20, 5], [16, 20, 5]], np.int64)
    e0 = w.total_energy()
    assert w.deposit(c, 10.0) == "mech_in"
    assert (w.mech_in, w.work_out, p.get("_work_J")) == (10.0, 0.0, None)
    assert w.deposit(c, 6.0, actor=p) == "work_out"
    assert (w.mech_in, w.work_out, p["_work_J"]) == (10.0, 6.0, 6.0)
    assert w.total_energy() - e0 == pytest.approx(16.0, rel=1e-6)
    with pytest.raises(ValueError):
        M.deposit(w, c, -1.0)


# ── T-W2: a swing ───────────────────────────────────────────────────────────
def _push_open(w, mid, p, F=None):
    return w.operate(p, mid, "open", F=F)


def test_a_push_turns_the_leaf_the_quarter_turn_after_the_computed_time():
    w, mid, p = _room()
    rec = w.mechs[mid]
    cells0 = w._cells_of(rec["leaf"])
    I, mass = w._leaf_I(rec, cells0)
    r = _push_open(w, mid, p)
    F, arm = r["F"], r["arm"]
    tau_f = w._hinge_friction(rec, mass)
    t_closed = math.sqrt(2 * (math.pi / 2) * I / (F * arm - tau_f))
    t_num, om_num = M.integrate_swing(I, F * arm, tau_f, math.pi / 2, dt=1e-4)
    assert t_closed == pytest.approx(t_num, rel=1e-3)               # spec: 0.1 %
    n = _run_until_still(w, rec)
    ev = _verbs(w, "opens")
    assert len(ev) == 1 and ev[0].result == "open"
    t_ev = dict(ev[0].data)["t_s"]
    assert t_ev == pytest.approx(t_closed, rel=1e-6)                # exact stepping
    assert abs((n + 1) * TICK_S - t_closed) <= 2 * TICK_S
    # the cells afterwards are the exact rotation of the cells before
    ref = {tuple(c) for c in M.quarter_turn_float(cells0, -1, rec["pivot2"]).tolist()}
    assert _leaf(w, rec) == ref
    assert w.is_open(mid) and rec["joints"][0]["theta"] == pytest.approx(-math.pi / 2)


def test_a_swing_with_an_actor_books_work_out_and_a_spring_books_mech_in():
    w, mid, p = _room()
    rec = w.mechs[mid]
    e0, m0, w0 = w.total_energy(), w.mech_in, w.work_out
    _push_open(w, mid, p)
    _run_until_still(w, rec)
    de = w.total_energy() - e0
    assert de == pytest.approx(w.work_out - w0, rel=1e-4)
    assert w.mech_in == m0                                           # not both
    assert p["_work_J"] == pytest.approx(w.work_out - w0, rel=1e-12)
    # the closed form: a pusher's work is tau x the quarter turn
    r = rec["joints"][0]
    assert de == pytest.approx(p["_work_J"], rel=1e-4)
    # no actor: a spring let go
    w2, mid2, _ = _room(latch=False)
    rec2 = w2.mechs[mid2]
    rec2["pushes"].append({"pid": None, "F": 250.0, "arm": 0.8, "sign": -1,
                           "spring": True})
    w2._mech_active.add(mid2)
    e1, m1, wo1 = w2.total_energy(), w2.mech_in, w2.work_out
    _run_until_still(w2, rec2)
    assert w2.is_open(mid2)
    assert w2.total_energy() - e1 == pytest.approx(w2.mech_in - m1, rel=1e-4)
    assert w2.work_out == wo1


def test_a_swing_loses_no_mass_and_no_fluid_into_hot_air_or_over_a_thin_puddle():
    w, mid, p = _room()
    rec = w.mechs[mid]
    # hot air in the sector (the cells the leaf will enter), and a puddle on the
    # floor under the swing, thin enough to be passed (< 1 cell of fluid)
    entered = [(10, y, z) for y in range(WALL_Y - 19, WALL_Y) for z in (3, 20)]
    for c in entered:
        w.E[c] = 40.0
    for y in range(WALL_Y - 19, WALL_Y, 3):
        w.fl[10, y, 1] = WATER
        w.fvol[10, y, 1] = 0.5
    w.step()
    wood0 = w.total_mass(WOOD)
    water0 = w.total_fluid(WATER)
    # the EXCHANGE itself, exact: heat, fluid and mass move with their cells
    cells = w._cells_of(rec["leaf"])
    e_cells = float(w.E.astype(np.float64).sum())
    f_cells = float(w.fvol[w.fl == WATER].sum())
    assert w._turn_leaf(rec, cells, -1)
    assert float(w.E.astype(np.float64).sum()) == pytest.approx(e_cells, rel=1e-12)
    assert float(w.fvol[w.fl == WATER].sum()) == pytest.approx(f_cells, rel=1e-12)
    assert w.total_mass(WOOD) == wood0 and w.total_fluid(WATER) == water0
    # and the whole swing, run by the law, in the same world
    w2, mid2, p2 = _room()
    rec2 = w2.mechs[mid2]
    for c in entered:
        w2.E[c] = 40.0
    for y in range(WALL_Y - 19, WALL_Y, 3):
        w2.fl[10, y, 1] = WATER
        w2.fvol[10, y, 1] = 0.5
    wood0, water0 = w2.total_mass(WOOD), w2.total_fluid(WATER)
    e0, wo0 = w2.total_energy(), w2.work_out
    _push_open(w2, mid2, p2)
    _run_until_still(w2, rec2)
    assert w2.is_open(mid2)
    assert w2.total_mass(WOOD) == pytest.approx(wood0, rel=1e-12)
    assert w2.total_fluid(WATER) == pytest.approx(water0, rel=1e-6)
    # energy: the books close to the float32 rounding of the hot cells
    assert w2.total_energy() - e0 == pytest.approx(w2.work_out - wo0, abs=0.5)


def test_600_ticks_of_support_and_torque_on_the_open_leaf_leave_it_standing():
    w, mid, p = _room()
    rec = w.mechs[mid]
    _push_open(w, mid, p)
    _run_until_still(w, rec)
    open_cells = _leaf(w, rec)
    for _ in range(600):
        w.step()
    if _leaf(w, rec) != open_cells:                  # the risk named in 7.3
        pytest.fail("the torque/support law moved a free-standing open leaf: "
                    "phase B must add the leaf's joint to what _torque_solid "
                    "treats as anchored (torque-law finding)")
    assert w.things[rec["leaf"]]["dead"] is None


# ── T-W3: a body in the way ─────────────────────────────────────────────────
def _move_body(w, p, dx, dy, steps):
    for _ in range(steps):
        cells = w._person_cells_of(p)
        if not w._walk(cells, dx, dy):
            break
        w._carry_own(p, dx, dy)


def test_a_body_in_the_sector_blocks_the_swing_and_stepping_away_lets_it_finish():
    w, mid, p = _room()
    rec = w.mechs[mid]
    _move_body(w, p, -1, 0, 24)                      # into the swept sector
    cx = float(w._person_cells_of(p)[:, 0].mean())
    assert cx < 24
    w.operate(p, mid, "open")
    for _ in range(6):
        w.step()
    j = _verbs(w, "jams")
    assert j and j[0].result == "blocked"
    assert len(_verbs(w, "jams")) == 1                # said once, not every tick
    assert rec["swing"] is None and not w.is_open(mid)
    # the door does not push people out of the way: nothing moved
    assert len(_leaf(w, rec)) == 800
    _move_body(w, p, 1, 0, 24)                       # step away
    _run_until_still(w, rec)
    assert w.is_open(mid) and _verbs(w, "opens")


def test_a_body_stepping_in_mid_swing_aborts_it_with_the_energy_booked_as_a_blow():
    w, mid, p = _room()
    rec = w.mechs[mid]
    pusher = w.persons[0]
    # a second body to step in
    scenes._person(w, 44, 6)
    w.step()
    q = w.persons[1]
    e0, wo0 = w.total_energy(), w.work_out
    _push_open(w, mid, pusher)
    for _ in range(14):
        w.step()
    assert rec["swing"] is not None
    _move_body(w, q, 0, 1, 14)                       # into the swept part of the sector,
    _move_body(w, q, -1, 0, 20)                      # during the swing
    for _ in range(20):
        w.step()
        if rec["swing"] is None:
            break
    j = [e for e in _verbs(w, "jams") if e.result == "blocked"]
    assert j, "the swing should have aborted"
    assert not w.is_open(mid) and rec["joints"][0]["theta"] == 0.0
    assert len(_leaf(w, rec)) == 800
    ke = dict(j[0].data)["ke"]
    assert ke > 0.0                                   # it had kinetic energy
    # the books: work in = heat out (friction at the hinge, the blow at the body)
    assert w.total_energy() - e0 == pytest.approx(w.work_out - wo0, rel=1e-3)


# ── T-W4: the latch ─────────────────────────────────────────────────────────
def test_the_latch_holds_up_to_its_capacity_and_lets_go_above_it():
    w, mid, p = _room()
    rec = w.mechs[mid]
    # a plain push (no handle): the latch holds. 250 N at 0.8 m is far below it.
    w.operate(p, mid, "push", F=250.0, sign=-1)
    for _ in range(80):
        w.step()
    assert not w.is_open(mid) and rec["swing"] is None
    assert not any(mech_failed for mech_failed in
                   (M.link_failed(lk) for lk in rec["links"]))
    w.operate(p, mid, "release")
    assert rec["pushes"] == []
    # a ram: the load on the latch path beats the weakest link
    rec["pushes"].append({"pid": None, "F": 9000.0, "arm": 0.8, "sign": -1})
    w._mech_active.add(mid)
    _run_until_still(w, rec)
    assert w.is_open(mid)
    assert any(M.link_failed(lk) for lk in rec["links"] if lk["id"].startswith("latch."))
    br = _verbs(w, "breaks")
    assert br


def test_the_handle_act_retracts_the_bolt_and_the_door_opens_to_a_small_push():
    w, mid, p = _room()
    rec = w.mechs[mid]
    latch = rec["catches"][0]
    assert latch["x_mm"] >= latch["engage_mm"]
    r = w.operate(p, mid, "open")                    # handle, then push
    assert latch["x_mm"] == 0.0 and r["result"] == "pushing"
    _run_until_still(w, rec)
    assert w.is_open(mid)
    # the handle is let go: the bolt is out again (and clicks home on closing)
    assert latch["x_mm"] == latch["throw_mm"]
    # and the door can be closed again by a push the other way
    r = w.operate(p, mid, "open")
    assert r["result"] == "pushing"
    _run_until_still(w, rec)
    assert not w.is_open(mid) and _verbs(w, "closes")
    assert len(_leaf(w, rec)) == 800


def test_two_equal_and_opposite_pushes_leave_a_latch_free_door_shut():
    w, mid, p = _room(latch=False)
    rec = w.mechs[mid]
    w.operate(p, mid, "push", F=200.0, sign=-1)
    rec["pushes"].append({"pid": 99, "F": 200.0, "arm": 0.8, "sign": 1})
    for _ in range(60):
        w.step()
    assert not w.is_open(mid) and rec["swing"] is None
    # a third pusher tips the balance
    rec["pushes"].append({"pid": 98, "F": 150.0, "arm": 0.8, "sign": -1})
    _run_until_still(w, rec)
    assert w.is_open(mid)


# ── T-W5: kicks ─────────────────────────────────────────────────────────────
def _kick_until_open(w, mid, side="open", cap=400, **kw):
    rec = w.mechs[mid]
    out = []
    for k in range(1, cap + 1):
        r = w.blow(mid, m_s=5.6, v_s=6.0, side=side, **kw)
        out.append(r)
        if r["swing"]:
            return k, out
    return None, out


def test_the_failing_link_is_the_minimum_of_the_computed_capacities():
    w, mid, p = _room(flaw=False)
    rec = w.mechs[mid]
    cat = rec["catches"][0]
    links = [w._mlink(rec, i) for i in cat["path"]]
    idx, caps = M.weakest_link(links, flaw_amp=0.0)
    expect = links[idx]["id"]
    assert caps[idx] == min(caps)
    n, out = _kick_until_open(w, mid)
    assert n is not None
    assert out[-1]["failed"][0] == expect == "latch.split"
    # NOT circular: change the screw mode and the answer follows the capacities
    w2, mid2, _ = _room(flaw=False)
    rec2 = w2.mechs[mid2]
    strike = w2._mlink(rec2, "latch.strike")
    strike["mode"], strike["geom"] = "withdrawal", {"G": 0.45, "D": 0.00483,
                                                    "L": 0.003, "n": 2, "k": 3e5}
    links2 = [w2._mlink(rec2, i) for i in rec2["catches"][0]["path"]]
    idx2, caps2 = M.weakest_link(links2, flaw_amp=0.0)
    assert links2[idx2]["id"] == "latch.strike" != expect
    n2, out2 = _kick_until_open(w2, mid2)
    assert out2[-1]["failed"][0] == "latch.strike"


def test_the_keeper_wood_is_really_split_through_shatter_and_the_door_then_opens():
    w, mid, p = _room(flaw=False)
    rec = w.mechs[mid]
    keeper = tuple(rec["catches"][0]["keeper"])
    assert w.mat[keeper] == WOOD
    wood0 = w.total_mass(WOOD)
    n, out = _kick_until_open(w, mid)
    assert n is not None and out[-1]["swing"]
    assert w.mat[keeper] == AIR                         # the wood was split
    ring = [(keeper[0] + dx, keeper[1] + dy, keeper[2])
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    shards = [c for c in ring + [(keeper[0], keeper[1], keeper[2] - 1)]
              if w.rubble[c] and w.mat[c] == WOOD]
    assert shards, "splinters should lie where the keeper was"
    assert w.total_mass(WOOD) == pytest.approx(wood0, rel=1e-9)   # nothing minted
    assert w._break_log, "the existing shatter law reported the break"
    run = _run_until_still(w, rec)
    assert w.is_open(mid)
    # why(): the opens event is because of the break, which is because of a kick
    op = _verbs(w, "opens")[0]
    chain = [e.verb for e in w.why(op.id)]
    assert chain[-1] == "opens" and "breaks" in chain and "kicks" in chain
    assert chain.index("kicks") < chain.index("breaks") < chain.index("opens")
    # the noise: the kick went through the real _noise law
    assert any(s["db"] > 80.0 for s in w.sounds)


def test_the_same_kick_from_the_stop_side_leaves_the_latch_path_unharmed():
    w, mid, p = _room(flaw=False)
    rec = w.mechs[mid]
    for _ in range(60):
        r = w.blow(mid, m_s=5.6, v_s=6.0, side="stop")
        assert not r["swing"]
    assert all(lk["D"] == 0.0 and lk["life"] == 0.0 for lk in rec["links"]
               if lk["id"].startswith("latch."))
    stop = [w._mlink(rec, i) for i in rec["stop"]["links"]]
    assert max(lk["life"] for lk in stop) > 0.0          # the stop took it
    assert not w.is_open(mid)


def test_the_kick_act_finds_its_side_from_where_the_kicker_stands():
    w, mid, p = _room(flaw=False)
    rec = w.mechs[mid]
    # the person stands south of the wall = on the swing side = pushes into the stop
    r = w.kick(p, mid)
    assert r["side"] == "stop"
    assert w.work_out == pytest.approx(r["heat_J"] + r["KE_foot"] + r["KE_leaf_pending"]
                                       + (w.work_out - r["heat_J"] - r["KE_foot"]
                                          - r["KE_leaf_pending"]))
    # a kicker north of the wall pushes the way it opens
    scenes._person(w, 20, 32)
    w.step()
    q = w.persons[1]
    r = w.kick(q, mid)
    assert r["side"] == "open"
    # a kick's joules are the actor's: debited and credited to work_out (100.8 J)
    assert q["_work_J"] == pytest.approx(100.8, rel=1e-9)


def test_a_kick_closes_its_energy_the_actors_100_J_end_up_as_heat_or_pending_swing():
    w, mid, p = _room(flaw=False)
    scenes._person(w, 20, 32)
    w.step()
    q = w.persons[1]
    e0, wo0 = w.total_energy(), w.work_out
    r = w.kick(q, mid)
    assert r["side"] == "open"
    assert w.work_out - wo0 == pytest.approx(100.8, rel=1e-9)
    # held: everything is heat NOW (the leaf's rebound is dissipated)
    assert w.total_energy() - e0 == pytest.approx(100.8, rel=1e-3)
    # momentum: foot + leaf angular momentum about the axis (no failure)
    imp = r["impact"]
    assert imp["L_after"] == pytest.approx(imp["L_before"], rel=1e-9)


# THE KICKER. 100.8 J is the sim's typical kick (5.6 kg at 6 m/s, (u)). A much stronger kicker is
# real and has a source: Ramakrishnan, Wang, Shankar and Fien (2018), "A new method for the
# measurement and analysis of biomechanical energy delivered by kicking", Sports Engineering
# 21(1):53-62, doi 10.1007/s12283-017-0244-z: 52 volunteers did an open-stance front kick against
# a rig; two thirds averaged over 100 J, and the paper's probability analysis puts about one adult
# in a hundred above 215 J (the abstract; the full text was not read). `blow(joules=E)` keeps the
# speed at 6 m/s and scales the effective mass, the way `kick_energy` scales it with strength.
KICK_J_TYPICAL = 100.8
KICK_J_ONE_IN_100 = 215.0


def _kick_counts(joules, cap=400, stud=False):
    counts = {}
    rows = [("latch", {}), ("latch+deadbolt", {"deadbolt": True}), ("bar", {"bar": True})]
    if stud:                    # the strike screws reach the stud: 30 mm of wood behind them, not 12
        rows.insert(1, ("latch+stud", {"stud": True}))
    for name, kw in rows:
        w, mid, p = _room(flaw=False, **kw)
        n, out = _kick_until_open(w, mid, cap=cap, joules=joules)
        counts[name] = n if n is not None else math.inf
    return counts


def test_u_kicks_to_open_at_the_typical_kick_the_latch_opens_and_a_deadbolt_door_holds_REPORT():
    """Decision N6. With the NDS reduction term the strike screws are stronger than they were with
    the two ends of Rd swapped: 33 percent for the latch's #10 pair, 45 percent for the deadbolt's
    #8 pair. A latch + deadbolt door now holds a 100.8 J kick for the 400 kicks tried (it opened
    after 121 before the fix); the latch-only door went from 14 kicks to 8. The three-way order is
    asserted at the heavier kick below."""
    counts = _kick_counts(KICK_J_TYPICAL)
    print("kicks to open (100.8 J, flaw_amp 0):", counts)
    assert counts["latch"] <= 20                        # a few to ten kicks: 8
    assert counts["latch"] < counts["latch+deadbolt"]
    assert counts["bar"] == math.inf                    # the bar holds a 100.8 J kick


def test_u_kicks_to_open_latch_lt_deadbolt_lt_bar_at_the_one_in_a_hundred_kick_REPORT():
    """The same doors under the strongest 1-in-100 adult kick, 215 J (see KICK_J_ONE_IN_100): the
    latch door opens at once, a latch whose strike screws reach the stud takes tens of kicks, the
    deadbolt door hundreds, the bar holds. (A TV test of a strike plate reports 8 kicks with 3/4-inch
    screws and 22 with 3-inch ones, kicker unknown: the same order of ladder, an anecdote and NOT a
    calibration.)"""
    counts = _kick_counts(KICK_J_ONE_IN_100, stud=True)
    print("kicks to open (215 J, flaw_amp 0):", counts)
    assert counts["latch"] < counts["latch+stud"] < counts["latch+deadbolt"] < counts["bar"]
    assert counts["bar"] == math.inf                    # the bar holds a 215 J kick too


def _eight_doors_mean_kicks(joules=None, cap=300, **kw):
    # eight doors in one world: each gets its own flaws (hash of door id and link)
    w = DoorWorld(240, 40, 50, voxel_cm=5)
    w.fill(0, 240, 0, 40, 0, 1, STONE)
    w.fill(0, 240, WALL_Y, WALL_Y + 1, 1, 45, STONE)
    mids = []
    for i in range(8):
        x0 = 10 + 28 * i
        w.fill(x0 - 1, x0 + 21, WALL_Y, WALL_Y + 1, 1, 45, AIR)
        mids.append(w.add_door((x0, WALL_Y), 20, 40, swing=-1, **kw))
    ns = []
    for mid in mids:
        n, _ = _kick_until_open(w, mid, cap=cap, joules=joules)
        ns.append(n if n is not None else cap)
    return float(np.mean(ns)), ns


def test_u_8_doors_at_the_default_flaw_amp_the_means_keep_the_order():
    a, na = _eight_doors_mean_kicks()
    b, nb = _eight_doors_mean_kicks(deadbolt=True)
    print("8 doors, mean kicks:", a, na, "latch+deadbolt:", b, nb)
    assert a < b
    assert len(set(na)) > 1                              # the flaws differ per door


def test_u_8_doors_at_the_one_in_a_hundred_kick_the_deadbolt_doors_open_and_the_order_stays():
    a, na = _eight_doors_mean_kicks(KICK_J_ONE_IN_100)
    b, nb = _eight_doors_mean_kicks(KICK_J_ONE_IN_100, deadbolt=True)
    print("8 doors at 215 J, mean kicks:", a, na, "latch+deadbolt:", b, nb)
    assert a < b < 300                                   # under the cap: some deadbolt doors opened
    assert len(set(nb)) > 1                              # the flaws differ per door


# ── T-W6: the bar ───────────────────────────────────────────────────────────
def test_a_barred_door_holds_until_the_bar_is_taken_and_no_flag_says_so():
    w, mid, p = _room(flaw=False, bar=True)
    rec = w.mechs[mid]
    assert w.bar_seated(rec)
    # kicks fail the latch's link, the door stays shut while the bar is in place
    for _ in range(30):
        r = w.blow(mid, m_s=5.6, v_s=6.0, side="open")
        assert not r["swing"]
    w.step()
    assert not w.is_open(mid)
    assert "bar" in w.holds_open_side(rec)
    # a handle act is refused for the reason: the bar
    r = w.operate(p, mid, "open")
    assert r["result"] == "jammed" and r["why"] == "barred"
    # lift the bar (take the thing): the lattice says so, nothing else
    w.lift_bar(mid)
    assert not w.bar_seated(rec)
    r = w.operate(p, mid, "open")
    assert r["result"] == "pushing"
    _run_until_open(w, mid)
    assert w.is_open(mid)


def test_a_bar_gives_way_to_a_much_bigger_blow_and_its_wood_splits():
    w, mid, p = _room(flaw=False, bar=True)
    rec = w.mechs[mid]
    b = rec["bar"]
    wood0 = w.total_mass(WOOD)
    # a 100 J kick never moves it; a 3 kJ ram does
    for _ in range(5):
        r = w.blow(mid, m_s=5.6, v_s=6.0, side="open")
        assert "bar.beam" not in r["failed"]
    assert w.bar_seated(rec)
    r = w.blow(mid, m_s=40.0, v_s=12.0, side="open", verb="strikes")
    assert "bar.beam" in r["failed"] or any(M.link_failed(w._mlink(rec, i))
                                           for i in b["path"])
    assert not w.bar_seated(rec)                       # split in its middle
    assert w.total_mass(WOOD) == pytest.approx(wood0, rel=1e-6)   # float32 cells
    assert r["swing"] and r["result"] == "swing"       # nothing holds it any more
    assert w.holds_open_side(rec) == []
    # the splinters are particles in the solver for a while; the redraw waits
    # for the cells they hold, and the books close either way
    for _ in range(200):
        w.step()
        if w.is_open(mid) or rec["swing"] is None:
            break
    assert w.is_open(mid) or _verbs(w, "jams")
    assert rec["swing"] is None


# ── T-W7: matter-held catch, fire ───────────────────────────────────────────
def test_an_ice_block_latch_drops_when_the_ice_melts_with_no_melt_code_of_its_own():
    w, mid, p = _room(latch=False)
    rec = w.mechs[mid]
    block = [(45, 35, 1), (45, 35, 2)]      # resting on the floor, in the next room
    for c in block:
        w.fill(c[0], c[0] + 1, c[1], c[1] + 1, c[2], c[2] + 1, TIN)
    w.add_gravity_latch(mid, block, push_Nm=250.0)
    assert mid in w._mech_watch
    m0 = w.mech_in
    for _ in range(40):
        w.step()
    assert not w.is_open(mid), "the block holds the latch up"
    assert not _verbs(w, "unlocks")
    # heat it past its melting point, with the sim's own laws doing the rest
    for c in block:
        w.E[c] += 1.5e5
    got = False
    for _ in range(400):
        w.step()
        if _verbs(w, "unlocks"):
            got = True
            break
    assert got, "the tin should have melted and dropped the latch"
    assert all(w.mat[c] != TIN for c in block)
    _run_until_open(w, mid)
    assert w.is_open(mid)
    assert w.mech_in > m0                                # a spring let go: no actor


def test_a_latched_door_whose_jamb_burned_loses_its_keeper_link_through_the_identity_pass():
    w, mid, p = _room(flaw=False)
    rec = w.mechs[mid]
    keeper = tuple(rec["catches"][0]["keeper"])
    assert not any(M.link_failed(lk) for lk in rec["links"])
    r = w.operate(p, mid, "push", F=250.0, sign=-1)       # a gentle push: held
    for _ in range(40):
        w.step()
    assert not w.is_open(mid)
    w.mat[keeper] = ASH                                    # what fire leaves behind
    w.step()
    broken = {lk["id"] for lk in rec["links"] if M.link_failed(lk)}
    assert {"latch.keeper", "latch.strike", "latch.split"} <= broken
    assert "latch.bolt" not in broken                      # the bolt is on the leaf
    assert [e for e in _verbs(w, "breaks") if dict(e.data).get("link") == "latch.split"]
    _run_until_open(w, mid)
    assert w.is_open(mid), "with the keeper gone the same gentle push opens it"


# ── locks (T-W8, the world half) ────────────────────────────────────────────
@pytest.mark.skipif(L is None, reason="locks.py (A-T11a) not present")
def test_a_key_turns_the_deadbolt_and_a_wrong_key_does_not():
    lock = L.make_lock(11)
    w, mid, p = _room(deadbolt=True, lock=lock)
    rec = w.mechs[mid]
    dead = [c for c in rec["catches"] if c["kind"] == "deadbolt"][0]
    assert dead["x_mm"] == dead["throw_mm"]
    r = w.operate(p, mid, "open")
    assert r["result"] == "jammed" and r["why"] == "locked"
    wrong = L.key_for(L.make_lock(12))
    assert w.operate(p, mid, "key", lockkw={"key": wrong})["result"] in (
        "wrong_blade", "wrong_cuts")
    assert dead["x_mm"] == dead["throw_mm"]
    assert w.operate(p, mid, "key", lockkw={"key": L.key_for(lock)})["result"] == "turns"
    assert dead["x_mm"] == 0.0 and lock["thrown"] is False
    assert w.operate(p, mid, "open")["result"] == "pushing"
    _run_until_still(w, rec)
    assert w.is_open(mid)


@pytest.mark.skipif(L is None, reason="locks.py (A-T11a) not present")
def test_picking_makes_clicks_only_a_close_listener_hears():
    lock = L.make_lock(21)
    w, mid, p = _room(deadbolt=True, lock=lock)
    rec = w.mechs[mid]
    n0 = len(w.sounds)
    res = w.operate(p, mid, "pick", lockkw={"tension": 0.03, "skill": 0.9})
    assert res["opened"] and lock["thrown"] is False
    clicks = w.sounds[n0:]
    assert clicks and all(30.0 <= s["db"] <= 45.0 for s in clicks)
    at = clicks[0]["at"]
    # real hearing law: 0.5 m is audible, 10 m is not (the db at 1 m in the law)
    db1 = clicks[0]["db"]
    assert db1 - 20 * math.log10(10.0) < 35.0 <= db1 - 20 * math.log10(0.5)


# ── menu affordances, the _spend seam ───────────────────────────────────────
def test_mech_affords_gives_2_to_4_rows_per_door_in_reach_and_none_out_of_it():
    w, mid, p = _room()
    cells = w._person_cells_of(p)
    assert w.mech_affords(p) == []                       # the door is 19 cells away
    _move_body(w, p, -1, 1, 14)
    _move_body(w, p, 0, 1, 12)
    rows = w.mech_affords(p)
    verbs = [r["verb"] for r in rows]
    assert 2 <= len(rows) <= 4 and "operate" in verbs and "kick" in verbs
    assert all(r["mid"] == mid for r in rows)


def test_the_blow_seam_handler_takes_the_whole_blow_through_the_record():
    w, mid, p = _room(flaw=False)
    rec = w.mechs[mid]
    cell = tuple(int(v) for v in w._cells_of(rec["leaf"])[100])
    rem = w.mech_blow(rec["leaf"], np.array([cell]), 100.0, a_hit=0.0025,
                      direction="open", striker=None)
    assert rem == 0.0
    assert w.mech_in == pytest.approx(100.0, rel=1e-9)
    # a thing without a record gets its blow back untouched
    assert w.mech_blow(0, np.array([cell]), 100.0) == 100.0


# ── T-W11: stairs, with the books of the real mixin ─────────────────────────
def test_a_12_step_flight_descent_closes_total_energy_against_mech_in():
    w = WalkDoorWorld(90, 20, 90, voxel_cm=5)
    w.fill(0, 90, 0, 20, 0, 1, STONE)
    for i in range(12):
        w.fill(14 + 5 * i, 90, 0, 20, 1, 1 + 4 * (i + 1), STONE)
    scenes._person(w, 8, 10)
    w.step()
    p = w.persons[0]
    p["step_up_m"] = p["step_down_m"] = 0.30
    comp, sl = w._person_cells(p)
    cells = w._claim_cells(p, comp, sl)
    while int(round(float(cells[:, 0].mean()))) < 78:
        cells, dz = w.step_along(p, cells, 1, 0)
        assert dz is not None
    m_kg = float(w.smass[tuple(cells.T)].sum()) / 1000.0
    e1, mi1, wo1 = w.total_energy(), w.mech_in, w.work_out
    while int(round(float(cells[:, 0].mean()))) > 6:
        cells, dz = w.step_along(p, cells, -1, 0)
        assert dz is not None
    de, dm = w.total_energy() - e1, w.mech_in - mi1
    assert dm == pytest.approx(m_kg * 9.81 * 48 * 0.05, rel=1e-6)
    assert de == pytest.approx(dm, rel=1e-4)
    assert w.work_out == wo1                             # the potential is nobody's work


# ── polish 5.B.5: every heat deposit names at least one cell ───────────────
# mech.deposit() refuses joules with no cell to put them in. mechlaw calls it
# with block[:_HEAT_MAX_CELLS], _contact_cells, _hinge_cells, _stop_end_cells and
# `fc or c`. Each is non-empty by construction: _cells_of is None or has >= 1 cell
# (never an empty array); every door has 3 hinges; a blocker list is only used
# when len(block) > 0; the foot cells hold the lowest flesh cell. These tests run
# the calls on the SMALLEST blocks (a 1-cell leaf, a 1-cell blocker, a leaf of
# only rubble) and sweep random rubble masks, so a slice that could be empty fails.
class _Seen:
    """Wrap the world's heat routine and note every deposit's cell count."""
    def __init__(self, w):
        self.w, self.n = w, []
        self._orig = w._heat_cells
        w._heat_cells = self

    def __call__(self, cells, J):
        self.n.append((mech_n_cells(cells), J))
        self._orig(cells, J)


mech_n_cells = M._n_cells


def _tiny_open_door(W=3, H=3):
    """A small door (3 cells wide by default; 1 and 2 swing too, see
    test_a_leaf_of_one_or_two_cells_swings_open_and_closed)."""
    w, mid, p = _room(W=W, H=H, latch=False)
    return w, mid, w.mechs[mid], p


def _spring(rec, F=20.0, arm=0.09):
    rec["pushes"].append({"pid": None, "F": F, "arm": arm, "sign": -1, "spring": True})


def test_the_smallest_door_swings_open_and_every_heat_deposit_names_a_cell():
    w, mid, rec, _p = _tiny_open_door()
    seen = _Seen(w)
    _spring(rec)
    w._mech_active.add(mid)
    e0, m0 = w.total_energy(), w.mech_in
    _run_until_still(w, rec)
    assert w.is_open(mid) and _verbs(w, "opens")
    assert seen.n and all(n >= 1 for n, _J in seen.n)
    assert w.total_energy() - e0 == pytest.approx(w.mech_in - m0, rel=1e-5)
    assert sum(J for _n, J in seen.n) == pytest.approx(w.mech_in - m0, rel=1e-12)


def test_a_kick_let_go_against_a_one_cell_blocker_books_its_joules_in_that_cell():
    """_start_swing refused: the kicked leaf's energy goes as heat into
    block[:_HEAT_MAX_CELLS]. A 1-cell leaf with ONE stone in its sweep is refused
    by that one blocker (before the 13a fix its own latch jamb refused it)."""
    w, mid, p = _room(W=1, H=1, latch=False)
    rec = w.mechs[mid]
    w.fill(10, 11, WALL_Y - 1, WALL_Y, 1, 2, STONE)      # one stone in the sweep
    seen = _Seen(w)
    e0, m0, wo0 = w.total_energy(), w.mech_in, w.work_out
    assert w._start_swing(rec, -1, omega0=1.0, ke0=5.0, kick_actor=None) is False
    assert seen.n == [(1, 5.0)]
    assert w.mech_in - m0 == 5.0 and w.work_out == wo0
    assert w.total_energy() - e0 == pytest.approx(5.0, rel=1e-6)
    # with the kicker: the same cell, debited from them and credited to work_out
    rec["jammed"] = False
    assert w._start_swing(rec, -1, omega0=1.0, ke0=3.0, kick_actor=p) is False
    assert seen.n[-1] == (1, 3.0)
    assert (w.mech_in - m0, w.work_out - wo0) == (5.0, 3.0)
    assert p["_work_J"] == pytest.approx(3.0, rel=1e-12)
    assert w.total_energy() - e0 == pytest.approx(8.0, rel=1e-6)


def test_a_swing_stopped_by_one_cell_in_its_way_books_all_its_energy():
    """_finish_swing 'aborted': tgt = block[:_HEAT_MAX_CELLS] with ONE blocker cell."""
    w, mid, rec, _p = _tiny_open_door()
    cells = w._cells_of(rec["leaf"])
    sector = w._swing_sector(rec, cells, -1, math.pi / 2.0)
    leaf = {tuple(c) for c in cells.tolist()}
    far = [tuple(int(v) for v in c) for c in sector if tuple(int(v) for v in c) not in leaf]
    x, y, z = max(far, key=lambda c: (abs(c[0] - 10) + abs(c[1] - WALL_Y), c))
    seen = _Seen(w)
    _spring(rec)
    w._mech_active.add(mid)
    e0, m0 = w.total_energy(), w.mech_in
    w.step()
    assert rec["swing"] is not None                              # it is under way
    w.fill(x, x + 1, y, y + 1, z, z + 1, STONE)                  # one cell, now in the way
    _run_until_still(w, rec)
    assert [e for e in _verbs(w, "jams") if e.result == "blocked"]
    assert not w.is_open(mid)
    assert seen.n and all(n >= 1 for n, _J in seen.n)
    assert any(n == 1 for n, _J in seen.n)                       # the blocker's own cell
    assert w.total_energy() - e0 == pytest.approx(w.mech_in - m0, rel=1e-5)
    assert w.mech_in - m0 > 0.0


def test_a_stalled_swing_with_no_blocker_goes_to_the_hinge_cells():
    w, mid, rec, _p = _tiny_open_door()
    seen = _Seen(w)
    cells = w._cells_of(rec["leaf"])
    I, _m = w._leaf_I(rec, cells)
    om = 0.4
    ke0 = 0.5 * I * om * om
    e0, m0 = w.total_energy(), w.mech_in
    assert w._start_swing(rec, -1, omega0=om, ke0=ke0, kick_actor=None)
    _run_until_still(w, rec)
    assert [e for e in _verbs(w, "jams") if e.result == "stalled"]
    assert all(n >= 1 for n, _J in seen.n) and seen.n
    assert {n for n, _J in seen.n} == {len(rec["hinges"])}       # the 3 hinge cells
    assert w.mech_in - m0 == pytest.approx(ke0, rel=1e-9)
    assert w.total_energy() - e0 == pytest.approx(ke0, rel=1e-5)


def test_a_blow_on_a_leaf_that_is_all_rubble_heats_the_contact_cell_itself():
    """No leaf cell left (all rubble): _contact_cells falls back to the struck cell,
    _foot_cells to None, and every joule of the blow is still booked in a cell."""
    w, mid, rec, p = _tiny_open_door()
    leaf = w._cells_of(rec["leaf"])
    w.rubble[tuple(leaf.T)] = True
    assert w._cells_of(rec["leaf"]) is None
    cell = (11, WALL_Y, 3)
    assert w._contact_cells(rec, cell).tolist() == [list(cell)]
    seen = _Seen(w)
    e0, m0, wo0 = w.total_energy(), w.mech_in, w.work_out
    out = {}
    w._book_blow(rec, cell, 10.0, 2.0, 0.0, None, out, leaf_dissipated=3.0)
    assert out["heat_J"] == 5.0
    assert [(n, J) for n, J in seen.n] == [(1, 5.0), (3, 3.0), (1, 2.0)]
    assert w.mech_in - m0 == pytest.approx(10.0, rel=1e-12) and w.work_out == wo0
    assert w.total_energy() - e0 == pytest.approx(10.0, rel=1e-6)
    # with an actor: the foot's heat goes to the foot cells (the lowest flesh cell is in them)
    fc = w._foot_cells(p)
    assert fc is not None and len(fc) >= 1
    assert int(fc[:, 2].min()) == int(w._person_cells_of(p)[:, 2].min())
    w._book_blow(rec, cell, 10.0, 2.0, 0.0, p, {}, leaf_dissipated=3.0)
    assert seen.n[-1] == (len(fc), 2.0)
    assert w.work_out - wo0 == pytest.approx(10.0, rel=1e-12)
    # a stop at the end of a leaf with no cell falls back to the hinge cells
    assert len(w._stop_end_cells(rec, -1)) == len(rec["hinges"]) >= 1


@pytest.mark.parametrize("seed", range(12))
def test_no_cell_list_a_deposit_is_given_can_be_empty_over_random_rubble(seed):
    rng = np.random.default_rng(7300 + seed)
    W = int(rng.integers(3, 7))
    H = int(rng.integers(1, 4))
    w, mid, rec, _p = _tiny_open_door(W, H)
    leaf = w._cells_of(rec["leaf"])
    assert len(rec["hinges"]) == 3 and len(w._hinge_cells(rec)) == 3
    for frac in (0.0, 0.3, 0.7, 0.999, 1.0):
        w.rubble[tuple(leaf.T)] = rng.random(len(leaf)) < frac
        if frac == 1.0:
            w.rubble[tuple(leaf.T)] = True
        cells = w._cells_of(rec["leaf"])
        assert cells is None or (len(cells) >= 1 and cells.shape[1] == 3)
        cell = tuple(int(v) for v in leaf[int(rng.integers(len(leaf)))])
        assert len(w._contact_cells(rec, cell)) >= 1
        assert len(w._stop_end_cells(rec, -1)) >= 1
        assert len(w._stop_end_cells(rec, 1)) >= 1
        assert len(w._hinge_cells(rec)) >= 1
    w.rubble[tuple(leaf.T)] = False


# ── the heat book closes when a cell is named twice (verifier, polish-2) ─────
# A door one cell tall has three hinges at two heights, so _hinge_cells names a
# cell twice. `E[idx] += x` with a repeated index applies ONE of them and the
# joules were booked but never stored (3x1 door: 2.7e-4 short; H=1 hinge
# deposit stored 2/3 of its joules). _heat_cells now adds every share.
@pytest.mark.parametrize("seed", range(12))
def test_heat_cells_stores_every_joule_even_when_a_cell_is_named_many_times(seed):
    rng = np.random.default_rng(7400 + seed)
    w, mid, rec, _p = _tiny_open_door(int(rng.integers(3, 7)), int(rng.integers(1, 4)))
    n = int(rng.integers(1, 9))
    base = w._hinge_cells(rec)
    cells = base[rng.integers(0, len(base), size=n)]          # repeats on purpose
    J = float(rng.uniform(0.1, 50.0))
    e0 = w.total_energy()
    w.deposit(cells, J)
    assert w.total_energy() - e0 == pytest.approx(J, rel=1e-6)
    assert w.E[tuple(cells.T)].sum() >= 0.0
    # the tuple-of-arrays form takes the same road
    e1 = w.total_energy()
    w.deposit(tuple(cells.T), J)
    assert w.total_energy() - e1 == pytest.approx(J, rel=1e-6)


@pytest.mark.parametrize("H", [1, 2, 3, 4])
def test_a_deposit_into_the_hinge_cells_closes_at_every_door_height(H):
    w, mid, rec, _p = _tiny_open_door(3, H)
    hc = w._hinge_cells(rec)
    assert len(hc) == 3
    if H == 1:
        assert len({tuple(c) for c in hc.tolist()}) < 3        # the case that leaked
    e0, m0 = w.total_energy(), w.mech_in
    w.deposit(hc, 10.0)
    assert w.mech_in - m0 == 10.0
    assert w.total_energy() - e0 == pytest.approx(10.0, rel=1e-6)


@pytest.mark.parametrize("H", [1, 2])
def test_a_short_door_swung_open_closes_its_books_to_float32_rounding(H):
    w, mid, rec, _p = _tiny_open_door(3, H)
    _spring(rec)
    w._mech_active.add(mid)
    e0, m0 = w.total_energy(), w.mech_in
    _run_until_still(w, rec)
    assert w.is_open(mid)
    assert w.mech_in - m0 > 0.0
    assert (w.total_energy() - e0) / (w.mech_in - m0) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("seed", range(20))
def test_a_blow_by_an_actor_with_no_foot_cells_still_books_every_joule_in_a_cell(seed):
    """The foot-cells-None fallback with an actor (a body with no flesh): the foot's
    heat goes to the contact cells, and the books close for doors of H 1..3."""
    rng = np.random.default_rng(7500 + seed)
    w, mid, rec, p = _tiny_open_door(int(rng.integers(3, 7)), int(rng.integers(1, 4)))
    w._foot_cells = lambda actor: None
    seen = _Seen(w)
    e0, wo0 = w.total_energy(), w.work_out
    E0 = float(rng.uniform(1, 50))
    kf, kl, ld = (float(rng.uniform(0, f * E0)) for f in (0.5, 0.3, 0.1))
    cell = tuple(int(v) for v in w._cells_of(rec["leaf"])[int(rng.integers(0, 3))])
    w._book_blow(rec, cell, E0, kf, kl, p, {}, leaf_dissipated=ld)
    assert seen.n and all(n >= 1 for n, _J in seen.n)
    booked = E0 - kl                                          # kl waits as a pending swing
    assert w.work_out - wo0 == pytest.approx(booked, rel=1e-9)
    assert sum(J for _n, J in seen.n) == pytest.approx(booked, rel=1e-9)
    assert w.total_energy() - e0 == pytest.approx(booked, rel=1e-6)


# ── 13a (B7): a leaf of 1 or 2 cells swings ─────────────────────────────────
@pytest.mark.parametrize("W", [1, 2, 3, 5])
def test_a_leaf_of_one_or_two_cells_swings_open_and_closed(W):
    """The far corner of a W-cell leaf turned about its hinge corner reaches
    sqrt(W^2 + 1) cells, more than W: the 3 x 3 sub-sampled sweep rounded into
    the latch jamb and refused every swing of a 1 or 2 cell leaf. The frame is
    not a blocker while the leaf turns between closed and open."""
    w, mid, rec, _p = _tiny_open_door(W=W, H=3)
    _spring(rec)
    w._mech_active.add(mid)
    e0, m0 = w.total_energy(), w.mech_in
    _run_until_still(w, rec)
    assert w.is_open(mid) and _verbs(w, "opens") and not _verbs(w, "jams")
    assert w.total_energy() - e0 == pytest.approx(w.mech_in - m0, rel=1e-5)
    rec["pushes"].clear()
    rec["pushes"].append({"pid": None, "F": 20.0, "arm": 0.09, "sign": 1, "spring": True})
    w._mech_active.add(mid)
    _run_until_still(w, rec)
    assert not w.is_open(mid) and not _verbs(w, "jams")


def test_the_frame_still_blocks_a_leaf_turned_the_wrong_way():
    """Only the normal arc (closed to open and back) is exempt: a leaf turned
    into its hinge jamb still meets it."""
    import math
    w, mid, _p = _room(W=1, H=1, latch=False)
    rec = w.mechs[mid]
    cells = w._cells_of(rec["leaf"])
    sec = w._swing_sector(rec, cells, 1, math.pi / 2.0)
    assert len(w._sector_blockers(rec, cells, sec, normal=False))
    assert not len(w._sector_blockers(rec, cells, w._swing_sector(
        rec, cells, -1, math.pi / 2.0), normal=True))
