"""THE SCHEDULER: heap, processes, fold and unfold, the watch index, snapshots
(wave 13, T5, spec 3, 5 and 12 T15-T22b, proof scene P4).

A TEST-ONLY SlowWorld overrides `_tick_start` (super first, then `slow.pre`) so
no existing file is edited. The SHADOW host is a DictHost: a dict of node
values with a toy implicit-diffusion fine law.
"""
import copy
import heapq
import math
import time

import numpy as np
import pytest

from src.voxel import quiet as Q
from src.voxel import slowforms as F
from src.voxel import slowtime as ST
from src.voxel.materials import STONE, TICK_S
from src.voxel.scenes import _person
from src.voxel.sim import World

HOUR = 144000
DAY = 24 * HOUR


class SlowWorld(World):
    slow = None

    def _tick_start(self):
        super()._tick_start()
        if self.slow is not None:
            self.slow.pre(self)


def _room(nx=24, ny=20, nz=40, vc=5):
    w = SlowWorld(nx, ny, nz, voxel_cm=vc)
    w.fill(0, nx, 0, ny, 0, 1, STONE)
    w.exits = []
    return w


def _bar(name="bar", box=(10, 5, 1, 14, 9, 5), c0=None, D=1e-9, tick_s=1.0, Cs=0.1):
    x = np.linspace(0.0, 0.02, 101)
    prof = Cs + 0.9 * np.exp(-((x - 0.01) / 0.003) ** 2) * np.sin(np.pi * x / 0.02) ** 2
    prof[0] = prof[-1] = Cs
    return ST.DictHost(name, x, prof if c0 is None else c0, 0.02, Cs=Cs, D=D, tick_s=tick_s, box=box)


SLAB = {"L": 0.02, "D": 1e-9, "Cs": 0.1, "bc": "dd"}


def _shadow(sched, host, params=SLAB, now=0):
    sched.add_host(host.name, host)
    return sched.add("fick", "shadow", ("dict", host.name), params, None, now)


# ─────────────────────────────────────── T15 heap, ordering, cancel ──
def test_heap_pops_in_tick_then_creation_order_and_a_hundred_thousand_stay_sorted():
    s = ST.Scheduler()
    rng = np.random.default_rng(7)
    due = rng.integers(0, 10_000, 100_000)
    s.add("decay", "owner", ("world",), {"half_s": 3600.0}, (1.0,), 0)
    pid = next(iter(s.procs))
    for d in due:
        s._seq += 1
        heapq.heappush(s.heap, (int(d), s._seq, pid, 0, 0))
    out = [heapq.heappop(s.heap) for _ in range(len(s.heap))]
    assert out == sorted(out) and [o[0] for o in out] == sorted(int(d) for d in due)
    # equal ticks pop in creation order
    ties = [o for o in out if o[0] == out[0][0]]
    assert [t[1] for t in ties] == sorted(t[1] for t in ties)


def test_watchpoints_queue_in_order_and_fire_on_their_own_ticks():
    s = ST.Scheduler()
    s.add_drive("T", ("const", 25.0))
    p = {"b": 0.0437, "Tmin": -3.0, "Tmax": 48.0, "c": 0.3, "ymax": math.log(1e9),
         "T_kill": 60.0, "D_ref_s": 180.0, "T_ref_C": 121.1, "z": 10.0}
    marks = (ST.Watchpoint("log10", 6.0, "up", "spoiled", "note", True),
             ST.Watchpoint("log10", 4.0, "up", "turning", "note", False))
    pid = s.add("population", "owner", ("world",), p, (math.log(1e3), 0.0), 0,
                drives=(("T", "T"),), marks=marks)
    t4 = F.FORMS["population"].cross((math.log(1e3), 0.0), p, {"T": s.drives["T"]}, 0, "log10", 4.0, 1 << 30)
    t6 = F.FORMS["population"].cross((math.log(1e3), 0.0), p, {"T": s.drives["T"]}, 0, "log10", 6.0, 1 << 30)
    assert t4 < t6 and s.next_due() == t4
    w = _room()
    w.slow = s
    w.tick = t4 - 1
    w.step()                                           # pre() runs at tick t4 - 1: nothing
    assert not s.notes
    w.step()                                           # tick t4: the first mark fires
    assert [n[:3] for n in s.notes] == [(t4, pid, "crosses")]
    assert s.next_due() == t6
    w.tick = t6
    w.step()
    assert [n[0] for n in s.notes if n[2] == "crosses"] == [t4, t6]
    assert s.procs[pid].marks == ()                    # one-shot
    assert s.next_due() is None


def test_lazy_cancel_by_version_and_touch_requeues():
    s = ST.Scheduler()
    s.add_drive("T", ("const", -10.0))
    pf = {"mode": "ice", "K": 2.2, "R0": 0.0, "rho": 917.0, "L_f": 334e3, "Tf": 0.0}
    pid = s.add("front", "owner", ("world",), pf, (0.0,), 0, drives=(("T", "T"),),
                marks=(ST.Watchpoint("x", 0.1, "up", "skate", "note"),))
    d0 = s.next_due()
    assert d0 is not None and len(s.heap) == 1
    s.add_drive("warm", ("const", 5.0))
    s.touch(pid, {"drives": {"T": "warm"}}, 1000)      # above freezing: no growth, no crossing
    assert len(s.heap) == 1                            # the stale entry is still in the heap...
    assert s.next_due() is None and len(s.heap) == 0   # ...and is dropped when it surfaces
    s.touch(pid, {"drives": {"T": "T"}}, 2000)
    d1 = s.next_due()
    assert d1 is not None and d1 > d0                  # it lost 1000 ticks of growth
    s.remove(pid)
    assert s.next_due() is None and not s.procs


def test_a_hundred_runs_of_the_same_calls_give_the_same_reports():
    def run():
        w = _room()
        p = _person(w, 8, 8, z0=1)
        p["awake"], p["asleep"] = True, True
        s = ST.Scheduler()
        w.slow = s
        for _ in range(40):
            w.step()
        return ST.advance(w, 2500, sched=s)
    a, b = run(), run()
    assert a == b and a.key() == b.key() and a.jumped > 500


# ───────────────────────── T16 pre() with nothing to do writes nothing ──
def test_pre_with_no_process_writes_nothing_and_a_world_with_it_is_bit_identical():
    a, b = _room(), _room()
    b.slow = ST.Scheduler()
    for w in (a, b):
        _person(w, 8, 8, z0=1)
    for _ in range(200):
        a.step()
        b.step()
    assert Q.diff(Q.digest(a), Q.digest(b)) == [] and a.tick == b.tick == 200
    before = copy.deepcopy(b.slow.__dict__, {id(b): None})
    b.slow.pre(b)
    after = b.slow.__dict__
    assert before.keys() == after.keys()
    for k in before:
        if k not in ("history", "notes", "index"):
            assert before[k] == after[k] or k in ("clock", "drives", "hosts"), k
    assert not b.slow.notes and not b.slow.sinks and not b.slow.residual


# ──────────────────────────────── T17 fold and unfold conserve ──
def test_fold_and_unfold_conserve_the_hosts_mass_to_1e12_and_record_the_residual():
    s = ST.Scheduler()
    h = _bar()
    pid = _shadow(s, h)
    m0 = h.totals()[("mass", "c")]
    assert s.fold(pid, 0)
    p = s.procs[pid]
    assert p.mode == "folded" and 1 < len(p.state) <= 32
    assert abs(F.FORMS["fick"].totals(p.state, p.params, h.x) - m0) < 1e-12 * m0
    assert s.unfold(pid, 0, "test")
    assert p.mode == "live" and abs(h.totals()[("mass", "c")] - m0) < 1e-12 * m0
    assert s.total_residual() < 1e-12 * m0
    assert ("mass", "c") in s.residual


def test_a_thousand_fold_unfold_cycles_do_not_drift():
    s = ST.Scheduler()
    h = _bar()
    pid = _shadow(s, h)
    m0 = h.totals()[("mass", "c")]
    assert s.fold(pid, 0)
    s.unfold(pid, 0, "first")
    c1 = h.c.copy()                                    # the fitted profile (the fold smooths it once)
    for i in range(1000):
        assert s.fold(pid, 0)
        s.unfold(pid, 0, "cycle")
    assert abs(h.totals()[("mass", "c")] - m0) < 1e-11 * m0
    assert float(np.abs(h.c - c1).max()) < 1e-10       # idempotent: the same profile every cycle
    assert s.total_residual() < 1e-10 * m0


def test_fold_is_refused_for_a_profile_it_cannot_hold_and_retried_later():
    s = ST.Scheduler()
    x = np.linspace(0, 0.02, 101)
    step = np.where(np.abs(x - 0.01) < 0.004, 1.0, 0.1)
    step[0] = step[-1] = 0.1
    h = ST.DictHost("bar", x, step, 0.02, Cs=0.1, box=(10, 5, 1, 14, 9, 5))
    pid = _shadow(s, h)
    assert not s.fold(pid, 0) and s.procs[pid].mode == "live"
    assert any(n[2] == "fold refused" for n in s.notes)


# ───────────────────────────── T18 the continuity check (section 5) ──
def test_continuity_unfold_then_600_toy_ticks_matches_the_closed_form():
    D, L = 1e-9, 0.02
    h = _bar(D=D, tick_s=1.0)                          # one toy tick is 1 s (40 world ticks)
    s = ST.Scheduler()
    pid = _shadow(s, h)
    s.fold(pid, 0)
    t_unfold = 3 * HOUR
    s.unfold(pid, t_unfold, "continuity")
    c_at_unfold = h.c.copy()
    h.tick(600)                                        # the fine law, 600 toy ticks = 10 minutes
    # the closed form that never unfolded
    s2 = ST.Scheduler()
    h2 = _bar(D=D)
    pid2 = _shadow(s2, h2)
    s2.fold(pid2, 0)
    ref = s2.observe(pid2, "center", t_unfold + 600 * 40)
    assert abs(h.observe("center") - ref) < 2e-4 * abs(ref) + 1e-9      # Crank-Nicolson at 1 s
    # then fold again and compare to the closed form that never unfolded
    s.fold(pid, t_unfold + 600 * 40)
    assert abs(s.observe(pid, "center", t_unfold + 600 * 40) - ref) < 2e-4 * abs(ref) + 1e-9
    # and the unfolded profile at the hand-over was the closed form at that tick
    ref_c = s2.procs[pid2]
    got = F.slab_profile(F.FORMS["fick"].advance(ref_c.state, ref_c.params, {}, 0, t_unfold)[0],
                         h.x, L, "dd", 0.1)
    assert float(np.abs(got - c_at_unfold).max()) < 1e-9


# ─────────────────────────────────────── T19 the watch index ──
def _awake_person_at(w, x, y):
    p = _person(w, x, y, z0=1)
    return p


def test_a_folded_host_unfolds_when_a_person_comes_within_reach_not_before():
    w = _room(70, 20, 40)
    s = ST.Scheduler()
    w.slow = s
    pid = _shadow(s, _bar(box=(52, 5, 1, 56, 9, 5)))
    assert s.fold(pid, 0)
    p = _awake_person_at(w, 4, 7)                      # 10 cells away: out of reach
    for _ in range(3):
        w.step()
    assert s.procs[pid].mode == "folded"
    p["_eye"] = (40.0, 7.0, 20.0)                      # now a metre-ish from the box
    p["awake"] = True
    w.persons[0]["asleep"] = False
    w.step()
    assert s.procs[pid].mode == "live" and any(n[2] == "unfold" for n in s.notes)


def test_a_sleeping_or_unconscious_person_does_not_wake_a_folded_host():
    w = _room()
    s = ST.Scheduler()
    w.slow = s
    pid = _shadow(s, _bar(box=(10, 5, 1, 14, 9, 5)))
    s.fold(pid, 0)
    p = _awake_person_at(w, 11, 7)
    p["awake"], p["asleep"] = True, True
    for _ in range(30):
        w.step()
    assert s.procs[pid].mode == "folded"
    p["asleep"] = False
    w.step()
    assert s.procs[pid].mode == "live"


@pytest.mark.parametrize("what", ["hot", "smoke", "liquid", "moving", "pin"])
def test_each_awake_window_unfolds_a_host_it_covers(what):
    w = _room()
    s = ST.Scheduler()
    w.slow = s
    pid = _shadow(s, _bar(box=(10, 5, 1, 14, 9, 5)))
    for _ in range(3):
        w.step()
    s.fold(pid, w.tick)
    for _ in range(5):
        w.step()
    assert s.procs[pid].mode == "folded"
    if what == "hot":
        w.E[15, 7, 3] = 50.0                           # within the 2-cell margin of the box
    elif what == "smoke":
        w.smoke[16, 7, 3] = 0.5
    elif what == "liquid":
        w.pour(15, 7, 1, 1, 100.0)
    elif what == "moving":
        w.moving[15, 7, 3] = True
    else:
        s.pin(pid, w, w.tick)
    w.step()
    assert s.procs[pid].mode == "live", what
    assert (what != "pin") or pid in s.pins


def test_a_flame_lit_beside_a_folded_host_unfolds_it_before_any_law_writes():
    """pre() runs directly after _tick_start, before every law. The hot cell is
    already there when the tick begins, so the host is live before conduction runs."""
    w = _room()
    s = ST.Scheduler()
    w.slow = s
    host = _bar(box=(10, 5, 1, 14, 9, 5))
    pid = _shadow(s, host)
    for _ in range(3):
        w.step()
    s.fold(pid, w.tick)
    w.E[15, 7, 3] = 800.0
    seen = []
    real = w._law_conduct

    def spy(sl=None):
        seen.append(s.procs[pid].mode)
        return real(sl)
    w._law_conduct = spy
    w.step()
    assert seen and seen[0] == "live"


def test_a_live_host_folds_only_after_it_has_been_outside_every_window_for_200_ticks():
    w = _room()
    s = ST.Scheduler()
    w.slow = s
    pid = _shadow(s, _bar(box=(10, 5, 1, 14, 9, 5)))
    p = s.procs[pid]
    for _ in range(3):
        w.step()
    p.since = w.tick
    t0 = w.tick
    folded_at = None
    for i in range(400):
        w.step()
        if p.mode == "folded":
            folded_at = w.tick - t0
            break
    assert folded_at is not None and 200 <= folded_at <= 230
    w.E[12, 7, 3] = 1.0                                # something stirs: it unfolds and the clock restarts
    w.step()
    assert p.mode == "live" and p.since >= w.tick - 1


def test_watch_index_cost_is_min_blocks_or_processes_and_finds_the_same_hits():
    idx = ST._BlockIndex()
    rng = np.random.default_rng(5)
    boxes = {}
    for pid in range(400):
        x, y, z = (int(v) for v in rng.integers(0, 4000, 3))
        boxes[pid] = (x, y, z, x + 3, y + 3, z + 3)
        idx.add(pid, boxes[pid])
    small = (100, 100, 100, 102, 102, 102)
    big = (0, 0, 0, 4000, 4000, 4000)
    for q in (small, big, (10, 3000, 50, 2500, 3200, 900)):
        want = sorted(p for p, b in boxes.items() if ST.box_overlap(q, b))
        assert idx.query(q) == want
    # the big box iterates the 400 processes, not its 250^3 blocks
    t = time.perf_counter()
    idx.query(big)
    assert time.perf_counter() - t < 0.05
    idx.remove(5)
    assert 5 not in idx.query(big) and len(idx.boxes) == 399


# ───────────────────────────────── T20 deepcopy and snapshots ──
def _many(n):
    s = ST.Scheduler()
    for i in range(n):
        s.add("decay", "owner", ("cells", i, 0, 0, i + 1, 1, 1), {"half_s": 3600.0 + i % 7},
              (1.0 + i % 3,), i % 100)
    return s


@pytest.mark.parametrize("n,bound", [(1000, 0.1), (20000, 0.5)])
def test_deepcopy_of_a_scheduler_round_trips_in_time(n, bound):
    s = _many(n)
    t = time.perf_counter()
    c = copy.deepcopy(s)
    dt = time.perf_counter() - t
    assert dt < bound, dt                              # target <= 0.2 s per 10,000
    assert len(c.procs) == n and c.procs[7].state == s.procs[7].state
    assert c.procs[7] is not s.procs[7] and c.procs[7].params is s.procs[7].params   # interned, shared
    assert len({id(p.params) for p in s.procs.values()}) <= 7           # 7 distinct rows, interned
    c.procs[7].state = (99.0,)
    assert s.procs[7].state != (99.0,)
    assert c.observe(3, "x", 10_000) == s.observe(3, "x", 10_000)


def test_the_scheduler_is_plain_data_no_lambda_no_world_reference():
    s = ST.Scheduler()
    s.add_drive("T", ("sum", ("const", 4.0), ("sine", 0.0, 3.0, 86400.0, 0.0)))
    _shadow(s, _bar())
    s.add("decay", "owner", ("world",), {"half_s": 3600.0}, (1.0,), 0)

    def scan(o, seen, path="s"):
        if id(o) in seen:
            return
        seen.add(id(o))
        assert not callable(o) or isinstance(o, type), path
        assert not isinstance(o, World), path
        if isinstance(o, dict):
            for k, v in o.items():
                scan(v, seen, f"{path}[{k!r}]")
        elif isinstance(o, (list, tuple, set, frozenset)) or type(o).__name__ == "deque":
            for i, v in enumerate(o):
                scan(v, seen, f"{path}[{i}]")
        elif hasattr(o, "__dict__") or hasattr(type(o), "__slots__"):
            names = list(getattr(o, "__dict__", {}))
            for cls in type(o).__mro__:
                names += list(getattr(cls, "__slots__", ()))
            for k in names:
                if hasattr(o, k):
                    scan(getattr(o, k), seen, f"{path}.{k}")
    scan(s, set())


def test_a_world_snapshot_restores_the_scheduler_and_the_same_calls_give_the_same_bits():
    w = _room()
    p = _person(w, 8, 8, z0=1)
    p["awake"], p["asleep"] = True, True
    s = ST.Scheduler()
    w.slow = s
    s.add_drive("T", ("const", -10.0))
    s.add("front", "owner", ("world",), {"mode": "ice", "K": 2.2, "R0": 0.1, "rho": 917.0,
                                         "L_f": 334e3, "Tf": 0.0}, (0.0,), 0, drives=(("T", "T"),))
    for _ in range(40):
        w.step()
    snap = w.snapshot()
    rep1 = ST.advance(w, 2600, sched=w.slow)
    x1 = w.slow.observe(1, "x", w.tick)
    w.restore(snap)
    assert w.slow is not s and w.slow.procs[1].t == 0 and w.tick == 40
    rep2 = ST.advance(w, 2600, sched=w.slow)
    assert rep1 == rep2 and w.slow.observe(1, "x", w.tick) == x1


# ───────────────────────── T21 valid() failures, T22b owner touch ──
def test_a_shadow_is_handed_back_at_the_tick_valid_fails_never_later():
    w = _room()
    s = ST.Scheduler()
    w.slow = s
    fail = 6000
    s.add_drive("T", ("table", [(fail, 925.0), (10 * fail, 600.0)]))
    p = dict(SLAB, D0=2.3e-5, Q=148e3, A1_C=727.0)
    p.pop("D")
    host = _bar(box=(10, 5, 1, 14, 9, 5))
    s.add_host("bar", host)
    pid = s.add("fick", "shadow", ("dict", "bar"), p, None, 0, drives=(("T", "T"),))
    s.fold(pid, 0)
    due = s.next_due()
    assert due == fail
    w.tick = fail - 1
    w.step()
    assert s.procs[pid].mode == "folded"
    w.step()                                           # pre() at tick `fail`
    assert s.procs[pid].mode == "live"
    assert any(n[:3] == (fail, pid, "unfold") and n[3] == "invalid" for n in s.notes)
    assert s.procs[pid].t == fail


def test_an_owner_that_fails_valid_is_advanced_to_the_last_valid_tick_and_marked_invalid():
    s = ST.Scheduler()
    fail = 7200
    s.add_drive("T", ("table", [(fail, 925.0), (50 * fail, 600.0)]))
    p = {"geometry": "wells", "wa": 0.07, "wb": 0.06, "Cs": 1.0, "C0": 0.2, "Q_cal": 32000.0,
         "A1_C": 727.0}
    pid = s.add("fick", "owner", ("world",), p, (0.0,), 0, drives=(("T", "T"),))
    v_fail, tag = s.observe_tag(pid, ("depth", 0.4), 10 * fail)
    assert tag == "invalid"
    v_at = s.observe(pid, ("depth", 0.4), fail)
    assert v_fail == v_at                              # the answer stops at the last valid tick
    s.touch(pid, {}, 10 * fail)                        # advancing for real marks it
    assert s.procs[pid].mode == "invalid" and s.procs[pid].t == fail
    assert any(n[2] == "invalid" for n in s.notes)


def test_owner_touch_advances_to_now_with_the_old_input_then_switches_it_without_a_handback():
    s = ST.Scheduler()
    s.add_drive("cold", ("const", -10.0))
    s.add_drive("hand", ("const", 33.0))
    pf = {"mode": "ice", "K": 2.2, "R0": 0.05, "rho": 917.0, "L_f": 334e3, "Tf": 0.0}
    pid = s.add("front", "owner", ("world",), pf, (0.0,), 0, drives=(("T", "cold"),))
    x_day = s.observe(pid, "x", DAY)
    s.touch(pid, {"drives": {"T": "hand"}}, DAY)       # a hand on the ice
    p = s.procs[pid]
    assert p.t == DAY and abs(p.state[0] - x_day) < 1e-15 and p.mode == "folded"
    assert dict(p.drives)["T"] == "hand"
    assert s.observe(pid, "x", 3 * DAY) == p.state[0]  # no growth while a warm hand holds it
    s.touch(pid, {"drives": {"T": "cold"}}, 3 * DAY)
    assert s.observe(pid, "x", 4 * DAY) > p.state[0]
    s.touch(pid, {"params": {"R0": 0.5}}, 4 * DAY)     # snow on top: a thicker film, slower growth
    assert dict(s.procs[pid].params)["R0"] == 0.5


def test_a_drive_may_be_another_processs_observe_a_wall_feeds_the_front_inside_it():
    s = ST.Scheduler()
    s.add_drive("T_out", ("const", -12.0))
    wall = s.add("relax", "owner", ("world",), {"tau_s": 70 * 3600.0}, (20.0,), 0,
                 drives=(("target", "T_out"),))
    pf = {"mode": "ice", "K": 2.2, "R0": 0.05, "rho": 917.0, "L_f": 334e3, "Tf": 0.0}
    ice = s.add("front", "owner", ("world",), pf, (0.0,), 0,
                drives=(("T", ("proc", wall, "x")),))
    t = 20 * DAY
    x_wall_driven = s.observe(ice, "x", t)
    s.add_drive("fast", ("const", -12.0))
    ice2 = s.add("front", "owner", ("world",), pf, (0.0,), 0, drives=(("T", "fast"),))
    assert 0.0 < x_wall_driven < s.observe(ice2, "x", t)                 # the lagging wall freezes later


# ───────────────────────────── P4 look and touch (DictHost) ──
def test_look_is_free_a_touch_is_honest_and_the_shadow_folds_after_the_person_leaves():
    w = _room()
    s = ST.Scheduler()
    w.slow = s
    host = _bar(box=(14, 5, 1, 18, 9, 5))
    pid = _shadow(s, host)
    for _ in range(3):
        w.step()
    s.fold(pid, w.tick)
    snap_before = (s.procs[pid].state, s.procs[pid].t, s.procs[pid].ver)
    v_look = s.observe(pid, "center", w.tick + 5000)       # a look: nothing unfolds, nothing writes
    assert (s.procs[pid].state, s.procs[pid].t, s.procs[pid].ver) == snap_before
    assert s.procs[pid].mode == "folded" and v_look < 1.0
    # he leans a hand on it
    p = _person(w, 12, 7, z0=1)
    p["_eye"] = (13.0, 7.0, 20.0)
    t_touch = w.tick
    w.step()
    assert s.procs[pid].mode == "live"
    # the profile handed to the fine law is the closed form at that tick, to one node
    ref = F.FORMS["fick"].advance(snap_before[0], s.procs[pid].params, {}, snap_before[1], t_touch)[0]
    want = F.slab_profile(ref, host.x, 0.02, "dd", 0.1)
    assert float(np.abs(host.c - want).max()) < 0.01 * float(want.max())
    # he leaves; 200 ticks later it folds again
    p["asleep"] = True                                 # he lies down (a sleeper is not in anyone's way)
    n = 0
    while s.procs[pid].mode == "live" and n < 400:
        w.step()
        n += 1
    assert s.procs[pid].mode == "folded" and 190 <= n <= 260


def test_an_owner_wall_is_read_lazily_a_look_costs_nothing_per_tick():
    s = ST.Scheduler()
    s.add_drive("RH", ("const", 0.6))
    pc = {"mode": "carb", "K": 5.8e-12, "R0": 0.0, "c_rate": 1.0, "drive": "RH",
          "cap_mol_m3": 5400.0, "co2_avail": 1e12, "fn_table": ((0.0, 0.0), (1.0, 1.0))}
    pid = s.add("front", "owner", ("world",), pc, (0.0,), 0, drives=(("RH", "RH"),))
    t = 365 * DAY
    a = s.observe(pid, "x", t)
    assert abs(a - math.sqrt(2 * 5.8e-12 * 365 * 86400 * 0.6)) < 1e-9 * a + 1e-15
    assert s.procs[pid].t == 0                          # nothing was stored
    w = _room()
    w.slow = s
    t0 = time.perf_counter()
    for _ in range(50):
        w.step()
    per_tick_extra = (time.perf_counter() - t0) / 50
    assert per_tick_extra < 0.05                        # pre() with nothing due is one comparison


# ───────────────────────────── the clock, the books, the verbs ──
def test_clock_until_dawn_and_resolve_target():
    c = ST.Clock(epoch_s=0.0)
    assert abs(c.tod(0) - 0.0) < 1e-9 and c.parse("06:00") == 21600.0 == c.parse("dawn")
    assert c.ticks_until(0, 21600.0) == int(21600.0 / TICK_S)
    assert c.ticks_until(int(21600.0 / TICK_S), 21600.0) == int(86400.0 / TICK_S)   # strictly after now
    c2 = ST.Clock(epoch_s=22 * 3600.0)
    assert abs(c2.ticks_until(0, 21600.0) * TICK_S - 8 * 3600.0) < TICK_S
    w = _room()
    s = ST.Scheduler(ST.Clock(epoch_s=20 * 3600.0))
    t, who = s.resolve_target(w, until="06:00")
    assert abs(t * TICK_S - 10 * 3600.0) < TICK_S and who is None
    t, who = s.resolve_target(w, until=("woke", 3))
    assert who == 3 and t > 10 ** 8
    assert s.resolve_target(w, seconds=2.5)[0] == 100 and s.resolve_target(w, ticks=7)[0] == 7
    with pytest.raises(ValueError):
        s.resolve_target(w)


def test_sleep_needs_a_sleep_law_and_wait_and_until_clock_are_advance():
    w = _room()
    s = ST.Scheduler()
    w.slow = s
    with pytest.raises(NotImplementedError, match="sleep law"):
        ST.sleep(w, 0, until="06:00")
    r = ST.wait(w, 3.0, sched=s)
    assert r.t_to == 120 and r.stepped == 120
    r = ST.until_clock(w, "00:00:10", sched=s)
    assert r.halted_by == "target" and abs(w.tick * TICK_S - 10.0) < TICK_S


def test_fluxes_are_booked_into_the_sinks_and_the_worlds_books():
    w = _room()
    s = ST.Scheduler()
    w.slow = s
    s.add_drive("T", ("const", -10.0))
    pf = {"mode": "ice", "K": 2.2, "R0": 0.05, "rho": 917.0, "L_f": 334e3, "Tf": 0.0, "area": 3.0}
    pid = s.add("front", "owner", ("world",), pf, (0.0,), 0, drives=(("T", "T"),))
    shed0 = w.shed
    s.touch(pid, {}, 5 * DAY, w)
    x = s.procs[pid].state[0]
    lat = x * 917.0 * 334e3 * 3.0
    assert abs(s.sinks[("E", "latent", "air_top")] - lat) < 1e-3 * lat * 1e-6 + 1e-3
    assert abs(w.shed - shed0 - lat) < 1e-6 * lat                       # the latent heat went up to the air
    assert s.sinks[("vol", "ice", "host")] > 0


def test_crosses_is_a_note_until_the_verb_exists():
    from src.voxel import events
    s = ST.Scheduler()
    w = _room()
    w.slow = s
    s.add_drive("T", ("const", -10.0))
    pf = {"mode": "ice", "K": 2.2, "R0": 0.0, "rho": 917.0, "L_f": 334e3, "Tf": 0.0}
    s.add("front", "owner", ("world",), pf, (0.0,), 0, drives=(("T", "T"),),
          marks=(ST.Watchpoint("x", 0.001, "up", "skin", "note", True),))
    ids = w.events.next_id
    due = s.next_due()
    w.tick = due
    w.step()
    notes = [n for n in s.notes if n[2] == "crosses"]
    assert len(notes) == 1 and notes[0][3] == ("x", 0.001, "up", "skin")
    if "crosses" in events.VERBS:                                       # phase B appends the verb LAST
        assert events.VERBS.count("crosses") == 1
        assert any(e.verb == "crosses" for e in w.events.since(ids - 1))
    else:
        assert w.events.next_id == ids                                  # no event, no shifted id
