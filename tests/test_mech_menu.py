"""Wave 15, lane 2a (spec S3 item 1): DOORS AND MECHANISMS IN THE MENU.

A person next to a door has rows for it (open or close, turn a key, work the
lock, seat or lift a bar, put a shoulder to it, kick it); the rows are the same
whether the door gives or not; a refusal ('jams') is learned by the person as
its own memory; a closed door is a passable edge of the planner with an open
cost; a stuck door is a point where the person chooses again (force, another
way, wait, call out) from its own scores, never from a rule here.

The mutation each test catches is named in the test.
"""
import math
import time

import numpy as np
import pytest

from src.mind.choosers import Rules
from src.mind.loop import Mind, MindConfig
from src.voxel import mechlaw, mindlink, routing, scenes
from src.voxel.materials import AIR, STONE, TICK_S
from src.voxel.policy import ACTS, TablePolicy
from src.voxel.sim import World

try:
    from src.voxel import locks as L
except ImportError:                                     # pragma: no cover
    L = None

NX, NY, NZ = 50, 60, 50
WALL_Y = 25


class _Stay(TablePolicy):
    def pick(self, situation, menu):
        return 0


def _wall(w, openings):
    """The wall along x at y = 25, stone, with the given (x0, x1) openings left
    empty (carving with fill(AIR) leaves a stale mass in the cell, so the wall
    is built round them)."""
    x = 0
    for a, b in sorted(openings) + [(NX, NX)]:
        if a > x:
            w.fill(x, a, WALL_Y, WALL_Y + 1, 1, 45, STONE)
        x = max(x, b)


def _world(swing=1, W=20, hinge_x=10, extra=(), **door):
    """A stone wall along x at y = 25 with a door (hinge at (hinge_x, 25), W
    cells wide) that opens NORTH (swing +1, corner (-1, 1)); the south room is
    where people stand. `extra` are more (x0, x1) openings. Returns (world, mid)."""
    w = World(NX, NY, NZ, voxel_cm=5)
    w.fill(0, NX, 0, NY, 0, 1, STONE)
    _wall(w, [(hinge_x - 1, hinge_x + 1 + W)] + list(extra))
    mid = w.add_door((hinge_x, WALL_Y), W, 40, swing=swing, along=(1, 0),
                     corner=(-1, 1 if swing > 0 else -1), **door)
    w.policy = _Stay()
    return w, mid


def _person(w, x, y):
    scenes._person(w, x, y)
    w.step()
    return w.persons[-1]


def _rows(w, p, limb):
    return w._menu(p, w._person_cells_of(p), None, limb, {}, cap=10 ** 6)


def _run_until_still(w, rec, n=300):
    for i in range(n):
        w.step()
        if rec["swing"] is None and i > 1:
            return i
    raise AssertionError("the swing never ended")


# ── T1: rows near a door, none far ───────────────────────────────────────────
def test_t1_rows_for_a_door_in_reach_and_none_out_of_it():
    w, mid = _world()
    near = _person(w, 20, 20)                      # about 0.5 m from the wall
    far = _person(w, 20, 6)
    rows = [r["verb"] for r in _rows(w, near, "hands")]
    assert "operate" in rows and "shoulder" in rows
    assert "kick" in [r["verb"] for r in _rows(w, near, "legs")]
    assert not any(r["verb"] in ("operate", "shoulder") for r in _rows(w, far, "hands"))
    assert not any(r["verb"] == "kick" for r in _rows(w, far, "legs"))
    # MUTATION: with the reach test dropped (reach huge) the far body has rows
    old = mechlaw.REACH_M
    mechlaw.REACH_M = 100.0
    try:
        w2, _ = _world()
        far2 = _person(w2, 20, 6)
        assert any(r["verb"] == "operate" for r in w2.mech_affords(far2)), \
            "the far case must fail when the reach test is dropped"
    finally:
        mechlaw.REACH_M = old


def test_every_row_verb_is_an_act_of_the_limb_that_is_asked():
    w, mid = _world()
    p = _person(w, 20, 20)
    for limb in ("hands", "legs"):
        for r in _rows(w, p, limb):
            assert r["verb"] in ACTS and ACTS[r["verb"]]["limb"] == limb


def test_a_world_without_a_door_has_no_door_rows_and_no_gate():
    w = World(NX, NY, NZ, voxel_cm=5)
    w.fill(0, NX, 0, NY, 0, 1, STONE)
    p = _person(w, 20, 20)
    assert not w.mech_menu and not w.door_route
    assert not any(r["tag"] in ("operate", "kick", "shoulder", "go:door")
                   for limb in ("hands", "legs") for r in _rows(w, p, limb))


# ── T2: the ops map (the old fall-through to a push) ─────────────────────────
def test_t2_an_unknown_op_is_not_a_push():
    w, mid = _world()
    p = _person(w, 20, 20)
    r = w.operate(p, mid, "wiggle")
    assert r["result"] == "no such op" and w.mechs[mid]["pushes"] == []


def test_t2_bar_lifts_a_seated_bar_and_does_not_shove_the_door():
    w, mid = _world(bar=True, flaw=False)
    rec = w.mechs[mid]
    p = _person(w, 20, 20)
    assert w.bar_seated(rec)
    assert any(r["op"] == "bar" for r in w.mech_affords(p))
    r = w.operate(p, mid, "bar")
    assert r["result"] == "lifted" and not w.bar_seated(rec)
    assert rec["pushes"] == []                      # MUTATION: the old fall-through pushed
    w.step()
    assert not w.is_open(mid)
    assert w.operate(p, mid, "open")["result"] == "pushing"
    _run_until_still(w, rec)
    assert w.is_open(mid)


def test_t2_a_bar_in_the_hand_is_seated_and_the_door_is_held_again():
    w, mid = _world(bar=True, flaw=False)
    rec = w.mechs[mid]
    p = _person(w, 20, 20)
    w.operate(p, mid, "bar")                        # lift it
    assert not w.bar_seated(rec)
    bc = w._cells_of(rec["bar"]["eid"])
    c = tuple(int(v) for v in bc[0])
    p["held"] = {"cell": c, "mat": int(w.mat[c]), "label": "bar", "arm": None,
                 "eid": rec["bar"]["eid"]}
    assert any(r["op"] == "bar" and "seat" in r["key"] for r in w.mech_affords(p))
    assert w.operate(p, mid, "bar")["result"] == "seated"
    assert w.bar_seated(rec) and p["held"] is None
    assert w.operate(p, mid, "open")["why"] == "barred"


@pytest.mark.skipif(L is None, reason="locks.py not present")
def test_t2_pick_ops_reach_the_lock_work():
    lock = L.make_lock(21)
    w, mid = _world(deadbolt=True, lock=lock, flaw=False)
    p = _person(w, 20, 20)
    rec = w.mechs[mid]
    r = w.operate(p, mid, "pick:light", lockkw={"skill": 0.9})
    assert r["opened"] and lock["thrown"] is False
    assert rec["pushes"] == [] and not w.is_open(mid)   # MUTATION: it would have shoved the door
    assert w.operate(p, mid, "pick:wrong")["result"] == "no such op"


@pytest.mark.skipif(L is None, reason="locks.py not present")
def test_a_held_key_or_pick_is_a_row_only_when_held():
    lock = L.make_lock(12)
    w, mid = _world(deadbolt=True, lock=lock)
    p = _person(w, 20, 20)
    ops = {r["op"] for r in w.mech_affords(p)}
    assert "key" not in ops and "pick:light" not in ops
    p["held"] = {"cell": (20, 20, 2), "mat": 1, "label": "key", "arm": None, "eid": 999}
    w.things[999] = {"key": L.key_for(lock), "called": "a key"}
    assert "key" in {r["op"] for r in w.mech_affords(p)}
    w.things[999] = {"tool": L.make_pick(), "called": "picks"}
    assert {"pick:light", "pick:firm"} <= {r["op"] for r in w.mech_affords(p)}


# ── T3: no god-channel ───────────────────────────────────────────────────────
def test_t3_the_rows_do_not_say_whether_the_door_gives():
    out = []
    for kw in ({}, {"deadbolt": True}):
        w, mid = _world(**kw)
        p = _person(w, 20, 20)
        out.append(([(r["key"], r["tag"], r["op"]) for r in w.mech_affords(p)],
                    [r["key"] for r in _rows(w, p, "hands")],
                    [r["key"] for r in _rows(w, p, "legs")]))
    assert out[0] == out[1]                       # MUTATION: add "(locked)" to a key: fails
    for part in out[0]:
        text = " ".join(str(x) for x in part).lower()
        assert not any(word in text for word in ("locked", "barred", "stuck", "jammed"))


# ── T4: learn by trying ──────────────────────────────────────────────────────
def test_t4_a_refusal_is_the_persons_own_memory_and_is_not_undone_by_others():
    w, mid = _world(deadbolt=True)
    p = _person(w, 20, 20)
    q = _person(w, 28, 20)
    rec = w.mechs[mid]
    before = [r["key"] for r in w.mech_affords(p)]
    r = w.operate(p, mid, "open")
    assert r["result"] == "jammed" and r["why"] == "locked"
    ev = r["event"]
    assert ev.verb == "jams" and ev.actor == p["pid"] and ev.result == "locked"
    assert w.minds[p["pid"]].bel[("jam", ("thing", rec["leaf"]))][0] == "locked"
    after = {r["op"]: r["key"] for r in w.mech_affords(p)}
    assert "(it would not give when you tried)" in after["open"]
    assert after["kick"] == [k for k in before if k.startswith("kick")][0]
    assert not any("would not give" in r["key"] for r in w.mech_affords(q))
    # someone unlocks it: p's memory is wrong and stays (it is p's, not the door's)
    for c in rec["catches"]:
        if c["kind"] == "deadbolt":
            c["x_mm"] = 0.0
    assert w._door_memory(p["pid"], rec) == "locked"
    # p tries again, it gives, and the memory is replaced
    assert w.operate(p, mid, "open")["result"] == "pushing"
    _run_until_still(w, rec)
    assert w._door_memory(p["pid"], rec) == ""


def test_t4_a_body_with_no_mind_keeps_no_belief_and_does_not_crash():
    w, mid = _world(deadbolt=True)
    p = _person(w, 20, 20)
    p2 = dict(p, pid=99)
    assert w.operate(p2, mid, "open")["result"] == "jammed"
    assert w._door_memory(99, w.mechs[mid]) == ""


# ── T5: the actor on 'opens' ─────────────────────────────────────────────────
def test_t5_opens_and_closes_carry_the_pusher():
    w, mid = _world(flaw=False)
    p = _person(w, 20, 20)
    q = _person(w, 40, 20)
    rec = w.mechs[mid]
    w.operate(p, mid, "open")
    _run_until_still(w, rec)
    ev = [e for e in w.events.rows if e.verb == "opens"]
    assert len(ev) == 1 and ev[0].actor == p["pid"]
    assert w.operate(q, mid, "close")["result"] == "pushing"
    _run_until_still(w, rec)
    cl = [e for e in w.events.rows if e.verb == "closes"]
    assert len(cl) == 1 and cl[0].actor == q["pid"]


def test_t5_the_most_torque_is_the_actor_when_two_push():
    w, mid = _world(flaw=False)
    p = _person(w, 20, 20)
    q = _person(w, 25, 20)
    rec = w.mechs[mid]
    w.operate(p, mid, "open", F=100.0)
    w.operate(q, mid, "open", F=300.0)
    _run_until_still(w, rec)
    assert [e.actor for e in w.events.rows if e.verb == "opens"] == [q["pid"]]


# ── T6: force: a kick or a shoulder, booked once ─────────────────────────────
def test_t6_a_kick_is_the_bodys_own_kick_and_booked_once():
    w, mid = _world(flaw=False)
    p = _person(w, 20, 30)                          # north of the wall: the open side
    assert any(r["verb"] == "kick" for r in _rows(w, p, "legs"))
    e0, wo0, mi0 = w.total_energy(), w.work_out, w.mech_in
    m_k, v_k = w.kick_energy(p)
    e_k = 0.5 * m_k * v_k * v_k
    w.kick(p, mid)
    assert w.work_out - wo0 == pytest.approx(e_k, rel=1e-9)     # the actor's, debited
    assert w.mech_in == mi0                                     # MUTATION: also mech_in fails
    assert w.total_energy() - e0 == pytest.approx(e_k, rel=1e-3)


def test_t6_a_shoulder_is_a_forces_event_paid_by_the_body():
    w, mid = _world(flaw=False)
    p = _person(w, 20, 30)
    wo0, mi0 = w.work_out, w.mech_in
    r = w.shoulder(p, mid)
    assert r["event"].verb == "forces" and r["event"].actor == p["pid"]
    assert w.work_out - wo0 > 0.0 and w.mech_in == mi0


def test_t6_the_will_carries_out_a_shoulder_row_and_a_kick_row():
    w, mid = _world(flaw=False)
    p = _person(w, 20, 30)
    cells = w._person_cells_of(p)
    hrow = next(r for r in _rows(w, p, "hands") if r["verb"] == "shoulder")
    w._carry_out(p, {"hands": hrow}, None, cells)
    assert any(e.verb == "forces" and e.actor == p["pid"] for e in w.events.rows)
    lrow = next(r for r in _rows(w, p, "legs") if r["verb"] == "kick")
    w._carry_out(p, {"legs": lrow}, None, cells)
    assert any(e.verb == "kicks" and e.actor == p["pid"] for e in w.events.rows)


# ── T7: names ────────────────────────────────────────────────────────────────
def test_t7_two_doors_read_differently_and_a_third_gets_an_ordinal():
    w, mid = _world(hinge_x=3, W=10, extra=[(14, 26), (29, 39)])
    mid2 = w.add_door((15, WALL_Y), 10, 40, swing=1, along=(1, 0), corner=(-1, 1))
    p = _person(w, 14, 22)
    keys = [r["key"] for r in w.mech_affords(p) if r["op"] in ("open", "close")]
    assert len(keys) == 2 and keys[0] != keys[1]
    # a third door the same way from the person: told apart by order, never by id
    mid3 = w.add_door((30, WALL_Y), 8, 40, swing=1, along=(1, 0), corner=(-1, 1))
    nm = w.mech_names(p, [mid, mid2, mid3])
    assert len(set(nm.values())) == 3
    assert not any(str(m) == n.split()[-1] for m, n in nm.items())


# ── T8: order independence ───────────────────────────────────────────────────
def test_t8_the_rows_do_not_depend_on_the_order_the_mechanisms_were_made():
    w, mid = _world(hinge_x=3, W=10, extra=[(14, 26)])
    w.add_door((15, WALL_Y), 10, 40, swing=1, along=(1, 0), corner=(-1, 1))
    p = _person(w, 14, 22)
    a = [(r["key"], r["mid"], r["op"]) for r in w.mech_affords(p)]
    w.mechs = dict(reversed(list(w.mechs.items())))
    b = [(r["key"], r["mid"], r["op"]) for r in w.mech_affords(p)]
    assert a == b


# ── hinged things are not carried off ────────────────────────────────────────
def test_a_door_on_its_hinges_is_not_a_thing_to_take_up():
    w, mid = _world()
    p = _person(w, 20, 20)
    assert not any(r["tag"] in ("take", "take:dear") and "door" in r["key"]
                   for r in _rows(w, p, "hands"))


# ── the route edge ───────────────────────────────────────────────────────────
def test_routing_beyond_door_pays_the_opening():
    plain = [(0, 1), (0, 2), (0, 3), (0, 4)]                    # 4 axis steps = 280
    through = [(1, 0), (2, 0)]                                  # 2 axis steps = 140
    assert routing.path_units(plain, (0, 0)) == 280
    assert routing.path_units([(1, 1)], (0, 0)) == 99
    assert routing.beyond_door(None, through, (0, 0), 10 ** 6)  # no other way: whatever it costs
    assert routing.beyond_door(plain, through, (0, 0), 100)     # 140 + 100 < 280
    assert not routing.beyond_door(plain, through, (0, 0), 140)  # 280 is not < 280
    assert not routing.beyond_door(plain, None, (0, 0), 0)
    # 1.5 s at 1.4 m/s over 5 cm voxels is 42 voxels of walking: 2940 units
    assert routing.open_cost_units(1.5, 1.4, 0.05) == 2940
    assert routing.open_cost_units(float("inf"), 1.4, 0.05) > 10 ** 9
    assert routing.first_door(plain, {(0, 3): 7}) == (7, 2)
    assert routing.first_door(plain, {}) == (None, -1)


def test_routing_door_spots_are_cells_in_reach_on_each_side_of_the_wall():
    fit = np.ones((30, 30), bool)
    fit[:, 14:17] = False                                       # a wall with the leaf in it
    leaf = np.array([[x, 15] for x in range(10, 20)])
    spots = routing.door_spots(fit, leaf, 6, (0.0, 1.0), (19, 31))
    assert len(spots) == 2
    ys = sorted(y for _x, y in spots)
    assert ys[0] < 14 and ys[1] > 16
    assert spots == routing.door_spots(fit, leaf, 6, (0.0, 1.0), (19, 31))
    assert routing.door_spots(np.zeros((30, 30), bool), leaf, 6, (0.0, 1.0), (19, 31)) == ()


def _flee_world(lock=False, second_exit=False):
    """A person south of the wall, an exit north of the door. With
    `second_exit` the door is moved left and a wide plain doorway cut at the
    right (the doorway is 14 cells = 0.7 m)."""
    hx = 2 if second_exit else 10
    w, mid = _world(hinge_x=hx, deadbolt=lock, flaw=False,
                    extra=[(30, 44)] if second_exit else [])   # a plain doorway elsewhere
    w.exits = [(hx + 4, 37)] + ([(38, 37)] if second_exit else [])
    p = _person(w, hx + 10, 6)
    p["known"][:] = True
    return w, mid, p


def test_the_closed_door_is_a_route_edge_with_an_open_cost():
    w, mid, p = _flee_world()
    rows = _rows(w, p, "legs")
    door = [r for r in rows if r["tag"] == "go:door"]
    assert len(door) == 1
    d = door[0]
    assert "go:exit" in d["beyond_tags"]                       # the exit lies beyond it
    t_open = w._open_time_s(p, w.mechs[mid])
    assert d["cost_ticks"] > math.floor(t_open / TICK_S) + 10  # the walk, and the opening
    assert not any(r["tag"] == "go:exit" for r in rows)        # a wall still, to the walk itself
    w.door_route = False                                       # the gate off: the old menu
    assert not any(r["tag"] == "go:door" for r in _rows(w, p, "legs"))


def test_a_door_found_stuck_is_not_priced_as_a_way_through_for_that_person_only():
    w, mid, p = _flee_world(lock=True)
    assert [r for r in _rows(w, p, "legs") if r["tag"] == "go:door"][0]["beyond"] != ()
    w._learn_door(p["pid"], w.mechs[mid], "locked", None)
    d = [r for r in _rows(w, p, "legs") if r["tag"] == "go:door"]
    assert d and d[0]["beyond"] == ()                           # still a door to walk to
    q = _person(w, 30, 6)
    q["known"][:] = True
    assert [r for r in _rows(w, q, "legs") if r["tag"] == "go:door"][0]["beyond"] != ()


def test_a_plain_way_that_is_shorter_once_the_opening_is_paid_beats_the_door():
    w, mid, p = _flee_world(second_exit=True)
    near_door = _person(w, 8, 6)
    near_gap = _person(w, 37, 6)
    for q in (near_door, near_gap):
        q["known"][:] = True
    far_exit = lambda q: [t for r in _rows(w, q, "legs") if r["tag"] == "go:door"
                          for t in r["beyond_tags"]]
    assert "go:exit" in far_exit(near_door)        # the door is the short way out of here
    assert "go:exit" not in far_exit(near_gap)     # from here the open doorway is shorter


def test_an_open_door_is_not_a_door_row():
    w, mid, p = _flee_world()
    q = _person(w, 40, 20)
    w.operate(q, mid, "open")
    _run_until_still(w, w.mechs[mid])
    p["free"] = p["fz"] = None
    assert not any(r["tag"] == "go:door" for r in _rows(w, p, "legs"))


# ── the mind: open it, learn it, choose again ────────────────────────────────
def _mind_world(second_exit=False, lock=False, seed=0):
    w, mid, p = _flee_world(lock=lock, second_exit=second_exit)
    pol = mindlink.link(w, mind=Mind(MindConfig(seed=seed, menu="flat")))
    pol.mind.npc = Rules(seed, argmax=True)
    b = pol.backend
    b.fears = lambda q: True                                    # a fire somewhere: safety
    return w, mid, p, pol, b


def _free(p):
    p["free"] = p["fz"] = None                                  # the doorway is seen as it is now


def test_the_mind_opens_the_door_the_job_ends_done_and_it_leaves():
    w, mid, p, pol, b = _mind_world()
    for _ in range(400):
        _free(p)
        w.step()
        if p["safe"]:
            break
    jobs = [(e.verb, e.phase) for e in b.job_events]
    assert ("open", "start") in jobs and ("open", "done") in jobs
    ev = [e for e in w.events.rows if e.verb == "opens"]
    assert ev and ev[0].actor == p["pid"]
    assert ("flee_to", "start") in jobs
    assert p["safe"], "it never got out"


def test_t5_without_the_actor_the_job_does_not_end_done(monkeypatch):
    monkeypatch.setattr(mechlaw.MechanismsMixin, "_swing_actor",
                        lambda self, sw: None)                  # MUTATION: drop the actor
    w, mid, p, pol, b = _mind_world()
    for _ in range(200):
        _free(p)
        w.step()
    assert ("open", "done") not in [(e.verb, e.phase) for e in b.job_events]


def test_a_refusal_ends_the_job_failed_with_the_laws_word_and_the_mind_chooses_force():
    w, mid, p, pol, b = _mind_world(lock=True)
    for _ in range(260):
        _free(p)
        w.step()
    jobs = [(e.verb, e.phase, e.reason) for e in b.job_events]
    assert ("open", "failed", "locked") in jobs
    assert w._door_memory(p["pid"], w.mechs[mid]) == "locked"
    assert [e for e in w.events.rows if e.verb == "kicks" and e.actor == p["pid"]], \
        "the person did not force it"
    # it did not try the handle again: the memory is its reason
    assert sum(1 for v, ph, _r in jobs if v == "open" and ph == "start") == 1


def test_a_refusal_with_another_way_out_scores_the_other_way_above_force():
    """The replanning point: after the refusal the scores are the person's, and
    a way out that is open beats a kick (DOOR_REPLY); the handle is NO_REASON."""
    w, mid, p, pol, b = _mind_world(lock=True, second_exit=True)
    w.operate(p, mid, "open")                                   # it tries; 'jams'; it remembers
    p["known"][:] = True
    _free(p)
    opts = b.bind(str(p["pid"]))
    pri, _, _ = b.rule_scores(str(p["pid"]), opts)
    score = {o.verb: s for o, s in zip(opts, pri)}
    verbs = {o.verb for o in opts}
    assert {"kick_in", "wait", "flee_to"} <= verbs              # force, wait, another way
    assert score["open"] == mindlink.NO_REASON
    assert score["flee_to"] > score["kick_in"] > score["wait"]
    assert "shout_fire" in verbs                                # call out: the person's own mouth
    q = _person(w, 8, 6)                                        # one who never tried, by the door
    q["known"][:] = True
    opts2 = b.bind(str(q["pid"]))
    pri2, _, _ = b.rule_scores(str(q["pid"]), opts2)
    assert {o.verb: s for o, s in zip(opts2, pri2)}["open"] == 1.0


def test_a_calm_person_has_no_reason_to_kick_or_retry_a_door():
    w, mid, p, pol, b = _mind_world(lock=True)
    b.fears = lambda q: False
    w.operate(p, mid, "open")
    opts = b.bind(str(p["pid"]))
    pri, _, _ = b.rule_scores(str(p["pid"]), opts)
    s = {o.verb: v for o, v in zip(opts, pri)}
    assert s["kick_in"] == mindlink.NO_REASON and s["open"] == mindlink.NO_REASON
    assert s["wait"] == 0.2


class _Tripwire:
    name = "tripwire"
    remote = True

    def __init__(self):
        self.asked = []

    def submit(self, req):
        self.asked.append(req.id)
        raise AssertionError("reactive mode sent a request to the model")

    def poll(self, req_id, block=True):
        raise AssertionError("reactive mode polled the model")


def test_t9_reactive_mode_makes_no_model_call_with_a_door_in_the_scene():
    w, mid, p = _flee_world(lock=True)
    tw = _Tripwire()
    pol = mindlink.link(w, mind=Mind(MindConfig(seed=0, menu="flat"), model=tw))
    pol.mind.npc = Rules(0, argmax=True)
    pol.backend.fears = lambda q: True
    for _ in range(200):
        _free(p)
        w.step()
    assert tw.asked == []


# ── T10: the books stay closed with a person working a door ──────────────────
def test_t10_a_kick_run_books_the_actors_joules_once_over_a_run():
    w, mid, p, pol, b = _mind_world(lock=True)
    mi0 = w.mech_in
    for _ in range(300):
        _free(p)
        w.step()
    kicks = [e for e in w.events.rows if e.verb == "kicks" and e.actor == p["pid"]]
    assert kicks
    assert w.mech_in == pytest.approx(mi0, abs=1e-9)           # the actor's joules, never mech_in


def test_mech_affords_cost_with_five_doors():
    w, mid = _world(hinge_x=2, W=7, extra=[(x - 1, x + 8) for x in (11, 20, 29, 38)])
    for x in (11, 20, 29, 38):
        w.add_door((x, WALL_Y), 7, 40, swing=1, along=(1, 0), corner=(-1, 1))
    p = _person(w, 20, 22)
    w.mech_affords(p)
    t0 = time.perf_counter()
    for _ in range(50):
        w.mech_affords(p)
    ms = (time.perf_counter() - t0) / 50 * 1e3
    print("mech_affords, 5 doors: %.3f ms" % ms)
    assert ms < 5.0
