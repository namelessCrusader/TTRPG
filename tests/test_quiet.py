"""THE QUIET DETECTOR and the driver-level skip (wave 13, T5, spec 6, 7 and 12
T23-T30; the soundness fuzz is @pytest.mark.slow).

A world is quiet when the cheap gates are closed AND one real PERIOD of ticks
leaves a digest of the whole state unchanged. These tests build small rooms with
a TEST-ONLY SlowWorld (its `_tick_start` calls super first, then `slow.pre`),
put exactly one thing in each, and ask the detector to name it.
"""
import ast
import math
import os

import numpy as np
import pytest

from src.voxel import quiet as Q
from src.voxel import slowforms as F
from src.voxel import slowtime as ST
from src.voxel import slowtwins as TW
from src.voxel.materials import (ACID, AIR, ASH, CHAR, STONE, TICK_S, WATER, WOOD)
from src.voxel.policy import WILL
from src.voxel.scenes import _person
from src.voxel.sim import World

HERE = os.path.dirname(os.path.abspath(__file__))
VOX = os.path.join(os.path.dirname(HERE), "src", "voxel")

_FIELDS = ("mat", "smass", "fl", "fvol", "fpot", "E", "smoke", "o2", "skin",
           "edge", "rubble", "eid", "p_add")


class SlowWorld(World):
    slow = None

    def _tick_start(self):
        super()._tick_start()
        if self.slow is not None:
            self.slow.pre(self)


def _room(nx=24, ny=20, nz=40, vc=5, cls=SlowWorld):
    w = cls(nx, ny, nz, voxel_cm=vc)
    w.fill(0, nx, 0, ny, 0, 1, STONE)
    w.fill(0, nx, 0, 1, 0, nz, STONE)
    w.fill(0, nx, ny - 1, ny, 0, nz, STONE)
    w.fill(0, 1, 0, ny, 0, nz, STONE)
    w.fill(nx - 1, nx, 0, ny, 0, nz, STONE)
    w.fill(0, nx, 0, ny, nz - 1, nz, STONE)
    w.exits = []
    return w


def _out(w, cohb=0.3, at=(10, 10)):
    p = _person(w, at[0], at[1], z0=1)
    p["awake"], p["cohb"], p["blood_o2"] = False, cohb, 1.0
    return p


def _settle(w, n=40):
    for _ in range(3000):
        w.step()
        if not w.bodies:
            break
    for _ in range(n):
        w.step()


def _bits(w):
    out = {k: np.ascontiguousarray(getattr(w, k)).tobytes()
           for k in _FIELDS if getattr(w, k, None) is not None}
    out["shed"] = repr(w.shed)
    out["gen"] = w._matter_gen
    return out


def _gates(w, tier=0):
    return {b.gate for b in Q.quiet(w, tier).blockers}


# ───────────────────────────────────────────────── T23 each gate ──
def _lit(w):
    w.fill(5, 7, 5, 7, 1, 2, WOOD)
    C = w.heat_capacity()
    w.E[5:7, 5:7, 1:2] = (500.0 - 20.0) * C[5:7, 5:7, 1:2]


def _drop(w):
    w.drops.append([5.0, 5.0, 10.0, 0.0, 0.0, 0.0, 10.0, WATER])


def _warm_cell(w):
    w.E[5, 5, 2] = 100.0


def _smoke(w):
    w.smoke[5:8, 5:8, 3:5] = 0.5


def _mover(w):
    w.moving[3, 3, 3] = True


def _sound(w):
    w.sounds.append((0, 0, 0, 0.0, 0.0))


def _acid(w):
    w.pour(6, 6, 1, ACID, 500.0, pot=1.0)


def _thermo(w):
    w.thermostats.append((3, 3, 3, 50.0))


def _uneven_o2(w):
    w.o2[5, 5, 5] *= 0.5


def _pressure(w):
    w.p_add[5, 5, 5] = 1.0


@pytest.mark.parametrize("make,gate", [
    (_lit, "heat"), (_drop, "fluids"), (_warm_cell, "heat"), (_smoke, "smoke"),
    (_mover, "matter"), (_sound, "sounds"), (_acid, "fluids"), (_thermo, "heat"),
    (_uneven_o2, "air"), (_pressure, "air")])
def test_a_world_with_exactly_one_thing_is_not_quiet_and_names_it(make, gate):
    w = _room()
    for _ in range(3):
        w.step()
    assert Q.quiet(w).ok                              # the empty room, cheap gates only
    make(w)
    q = Q.quiet(w)
    assert not q.ok
    assert gate in {b.gate for b in q.blockers}, q.blockers
    text = Q.explain(w)
    assert gate in text and "at" in text


def test_the_flame_names_its_cell_and_the_candle_keeps_its_window_awake():
    w = _room()
    for _ in range(3):
        w.step()
    _lit(w)
    q = Q.quiet(w)
    flame = [b for b in q.blockers if b.reason.startswith("flame")]
    assert flame and isinstance(flame[0].place, tuple) and len(flame[0].place) == 3
    assert 5 <= flame[0].place[0] < 7 and 5 <= flame[0].place[1] < 7
    sch = ST.Scheduler()
    w.slow = sch
    rep = ST.advance(w, 1300, sched=sch)                # a world with a flame ticks, not jumps
    assert rep.jumped == 0 and rep.stepped == 1300
    assert any(g == "heat" for g, _p, _r in rep.blockers_seen)


def test_persons_gate_awake_pending_reflex_chosen_stamp_and_no_twin():
    w = _room()
    p = _out(w)
    _settle(w)
    assert Q.quiet(w).ok
    p["_reflex"] = {"flees": w.tick + 10}
    assert any("reflex" in b.reason for b in Q.quiet(w).blockers)
    del p["_reflex"]
    p["_chose"] = {"stand": w.tick + 5}
    assert any("chosen-act" in b.reason for b in Q.quiet(w).blockers)
    p["_chose"] = {"stand": w.tick - 5}                # stamps are never deleted: spent is fine
    assert Q.quiet(w).ok
    p["wounds"] = [{"bone": "head", "open": True}]
    assert any("no twin" in b.reason for b in Q.quiet(w).blockers)
    del p["wounds"]
    p["awake"] = True
    assert any(b.reason == "awake person" for b in Q.quiet(w).blockers)
    p["asleep"] = True                                  # a scene-set sleeper is allowed
    assert not any(b.reason == "awake person" for b in Q.quiet(w).blockers)
    p["alive"] = False                                  # and the dead stay out of it
    p["asleep"] = False
    assert Q.quiet(w).ok


def test_the_empty_room_is_quiet_only_after_the_probe_and_the_probe_steps_one_period():
    w = _room()
    for _ in range(3):
        w.step()
    t0 = w.tick
    q = Q.quiet(w, probe_=True)
    assert q.ok and w.tick == t0 + 1 + Q.PERIOD         # a warm-up tick, then one period
    assert Q.quiet(w, probe_=False).ok


def test_a_warm_cooling_room_is_quiet_at_tier_1_though_the_radiation_pass_keeps_a_stamp():
    """base19 (B0). heat._law_radiate (wave 12, FIRE) stamps `_rad_last` with the tick of every
    pass and banks `_rad_bank`. While E != 0 a pass runs every RAD_EVERY ticks, so the stamp changes
    over every period. If the probe counted it as state, no warm world could ever be quiet at tier 1:
    the cooled-hearth test of test_slownet.py ticked for ever. quiet.EXCLUDE lists both rows."""
    from src.voxel import heat
    if not hasattr(heat, "RAD_EVERY"):
        pytest.skip("this tree has no radiation pass (base18)")
    w = _room()
    w.E[5, 5, 2] = 5.0                                   # a little warmth, nothing near ignition
    for _ in range(2 * heat.RAD_EVERY + 1):
        w.step()
    assert hasattr(w, "_rad_last"), "a pass has run"
    q = Q.quiet(w, 1, probe_=True)                       # the probe steps one period and digests the state
    assert q.ok, [(b.gate, b.place, b.reason) for b in q.blockers]
    assert Q.quiet(w, 0).ok is False                     # tier 0 still refuses a world with heat in it


def test_skip_quiet_off_is_never_quiet():
    w = _room()
    for _ in range(3):
        w.step()
    w.skip_quiet = False
    q = Q.quiet(w)
    assert not q.ok and q.blockers[0].gate == "skip"


# ───────────────────────────────────────────────────── T24 PERIOD ──
def test_period_is_the_lcm_of_every_gate_and_is_600_today():
    got = {n: v for n, v, _w in Q.period_sources()}
    assert got["look_every"] == WILL["look_every"] == 4
    assert got["scan_every"] == 25 and got["decide_every"] == 10
    assert got["RAD_EVERY"] == 6 and got["GAS_GEO_TICKS"] == 20
    assert got["BODY_CLOCK"] == 40 and got["CRUMPLE"] == 8
    want = 1
    for v in (6, 8, 4, 25, 10, 20, 40):
        want = want * v // math.gcd(want, v)
    assert want == 600 and Q.PERIOD == 600 == Q.period()


def test_changing_look_every_widens_the_period():
    old = WILL["look_every"]
    try:
        WILL["look_every"] = 7
        assert Q.period() == 4200                       # lcm(7, 25, 10, 6, 20, 40, 8)
        WILL["look_every"] = 3
        assert Q.period() == 600                        # 3 divides 6
    finally:
        WILL["look_every"] = old
    assert Q.period() == 600


def _tick_gate_expressions():
    """[(file, line, source text, right operand)] of every `<..tick..> % N` and
    `<..tick..> // N` in src/voxel, found with the ast module so a literal cannot hide."""
    out = []
    for f in sorted(os.listdir(VOX)):
        if not f.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(VOX, f)).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mod, ast.FloorDiv)):
                names = {n.id for n in ast.walk(node.left) if isinstance(n, ast.Name)}
                attrs = {n.attr for n in ast.walk(node.left) if isinstance(n, ast.Attribute)}
                if "tick" in names or "tick" in attrs:
                    out.append((f, node.lineno, ast.unparse(node), node.right))
    return out


def _resolve(node, consts):
    """The integer an operand stands for, or None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.Name):
        return consts.get(node.id)
    if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and node.value.id == "WILL" and isinstance(node.slice, ast.Constant)):
        return int(WILL[node.slice.value])
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "max"):
        vals = [_resolve(a, consts) for a in node.args]
        return max(vals) if None not in vals else None
    return None


# The tree base18 had. A file added since (wave 12/13 lanes) whose tick gate has a
# period this scan cannot resolve is REPORTED (a warning naming the gate that
# wiring must add to quiet.DECLARED_PERIODS), not failed: it is another lane's.
BASE_FILES = {"air", "beliefs", "bodies", "demo", "events", "fields", "fluids", "hands",
              "harvest", "heat", "kernels", "label", "ledgers", "materials", "mindlink",
              "motion", "perceive", "persons", "physiology", "policy", "pose", "room",
              "scenes", "seam", "senses", "sim", "skeleton", "support", "things", "will"}

# a tick gate whose period is not a constant, with the reason it is covered
_UNRESOLVED_OK = {
    ("fluids.py", "n"): "an acid row's own slow clock; gate_fluids blocks every potent reactive fluid",
    ("motion.py", "len(far)"): "picks among a walker's detours by tick // decide_every; "
                               "only an awake, moving person walks, and awake persons are never quiet",
    # base19 (wave 12 MINDS, will.py _wander_in_room): the same pick, now inside the person's room
    ("will.py", "len(far)"): "picks the roam spot among the far places of the person's own room by "
                             "tick // decide_every; only an awake person wanders, and awake persons "
                             "are never quiet",
}
# not a gate: the clock arithmetic of the scheduler itself
NOT_GATES = {"slowtime.py"}


def test_every_tick_percent_and_tick_floor_div_in_the_laws_divides_the_period():
    found = _tick_gate_expressions()
    assert found, "the ast scan found nothing"
    seen = set()
    for f, line, text, right in found:
        consts = {}
        mod = ast.parse(open(os.path.join(VOX, f)).read())
        for n in mod.body:
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) \
                    and isinstance(n.value, ast.Constant) and isinstance(n.value.value, int):
                consts[n.targets[0].id] = n.value.value
        if f in NOT_GATES:
            continue
        v = _resolve(right, consts)
        if v is None:
            key = (f, getattr(right, "id", ast.unparse(right)))
            if f[:-3] not in BASE_FILES:
                declared = Q.DECLARED_GATES.get(key)
                if declared is not None:                 # B1: declared in quiet.DECLARED_PERIODS
                    assert declared in {n for n, _v, _w in Q.period_sources()}, \
                        f"{f}:{line} `{text}` names {declared}, which is not a declared period"
                    continue
                import warnings
                warnings.warn(f"{f}:{line} `{text}`: a period this scan cannot resolve; "
                              f"add it to quiet.DECLARED_PERIODS at wiring")
                continue
            assert key in _UNRESOLVED_OK, f"{f}:{line} `{text}` has a period nobody listed"
            continue
        seen.add(v)
        assert Q.PERIOD % v == 0, f"{f}:{line} `{text}`: {v} does not divide {Q.PERIOD}"
    assert 8 in seen                                    # the literal in _law_life's crumple test
    assert {4, 25, 10} <= seen                          # look_every, scan_every, decide_every


def test_every_guard_in_step_has_a_registered_quiet_gate():
    """If a lane adds a skip guard to World.step() it must add a gate line."""
    tree = ast.parse(open(os.path.join(VOX, "sim.py")).read())
    step = next(n for c in tree.body if isinstance(c, ast.ClassDef)
                for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "step")
    covered = {                                         # guard text -> the gate that mirrors it
        "self._busy(self.fvol) or self.drops": "fluids",
        "hot is not None": "heat",
        "risen is not None": "heat",
        # B1 (wave 13a): the three `self._busy(self.E)` gates ask the seam `_heat_busy()`, whose
        # default IS `self._busy(self.E)` (fields.py); gate G3 mirrors it
        "self._heat_busy()": "heat",
        "self._busy(self.fvol)": "fluids",
        "sooty is not None": "smoke",
        "self._heat_busy() or self._mixing()": "air",
        "self._pushed()": "air",
        "self.thermostats or self._heat_busy()": "heat",
        "self.o2 is None": "air",
        "self.thermostats": "heat",
        # base19 (wave 12 FIRE, sim.py:339): igniters held to a cell are heat; gate G3 blocks on them
        "getattr(self, 'igniters', None)": "heat",
    }
    # B1 (wave 13a): per-tick HOOKS, which skip nothing, so they need no gate. The scheduler
    # (T5) is not a law that can be quiet; it is what jumps over quiet.
    hooks = {"self.slow is not None"}
    names = {n for n, _fn in Q.QUIET_GATES} | {"persons", "matter"}
    for node in ast.walk(step):
        if isinstance(node, ast.If):
            text = ast.unparse(node.test)
            if text in hooks:
                continue
            assert text in covered, f"a new guard in step() with no quiet gate: `{text}`"
            assert covered[text] in names


# ───────────────────────────────── T26 jump equals ticks (tier 0) ──
def _empty(cls=SlowWorld):
    return _room(cls=cls)


def _sleeper_room():
    w = _room()
    p = _person(w, 10, 10, z0=1)
    p["awake"], p["asleep"], p["blood_o2"] = True, True, 0.95
    return w


def _dead_hearth_room():
    w = _room()
    w.fill(14, 18, 8, 12, 1, 4, STONE)                 # a hearth that went out long ago
    w.fill(15, 17, 9, 11, 4, 5, CHAR, frac=0.4)
    w.fill(15, 17, 9, 11, 5, 6, ASH, frac=0.3)
    return w


def _compare(a, b):
    assert a.tick == b.tick
    assert _bits(a) == _bits(b)
    assert Q.diff(Q.digest(a), Q.digest(b)) == []
    for pa, pb in zip(a.persons, b.persons):
        for k in TW.TWINS["breath"].keys:
            assert abs(pa[k] - pb[k]) <= 1e-9 * max(abs(pa[k]), abs(pb[k]), 1e-12), k


@pytest.mark.parametrize("build", [_empty, _sleeper_room, _dead_hearth_room])
def test_a_tier_0_jump_equals_ticking_on_small_rooms(build):
    N = 3200
    a, b = build(), build()
    sch = ST.Scheduler()
    b.slow = sch
    for _ in range(40):
        a.step()
        b.step()
    for _ in range(N):
        a.step()
    rep = ST.advance(b, N, sched=sch)
    assert rep.jumped > N // 2 and rep.halted_by == "target" and rep.tier_used == "exact"
    assert rep.t_to == a.tick == b.tick
    _compare(a, b)


def test_a_jump_to_the_target_also_leaves_the_dead_alone_and_books_no_energy():
    a, b = _sleeper_room(), _sleeper_room()
    sch = ST.Scheduler()
    b.slow = sch
    for _ in range(40):
        a.step()
        b.step()
    e0 = b.total_energy()
    ST.advance(b, 3000, sched=sch)
    assert b.total_energy() == e0                      # tier 0 moves no energy
    assert sch.total_residual() == 0.0
    assert b.persons[0]["blood_o2"] > a.persons[0]["blood_o2"] - 1e-9


@pytest.mark.slow
def test_a_furnished_room_with_two_sleepers_jumps_20000_ticks_exactly():
    def build():
        w = _room(36, 30, 40)
        w.fill(20, 24, 5, 8, 1, 6, WOOD, frac=0.7)
        w.fill(26, 30, 18, 24, 1, 3, STONE)
        # q is 17 voxels long when it keels over towards +x: the line y = 14
        # is clear of the wood (y 5-8) and the stone (y 18-24), so it lies flat
        # and still (at y = 20 it fell ONTO the stone and never came to rest).
        # It falls FIRST and the room settles, because the thud of a body
        # hitting the floor wakes a sleeper, who is then an awake person (and
        # rightly never quiet).
        q = _person(w, 8, 14, z0=1)
        q["awake"], q["cohb"], q["blood_o2"] = False, 0.31, 1.0
        _settle(w, 120)
        p = _person(w, 8, 8, z0=1)
        p["awake"], p["asleep"] = True, True
        return w
    a, b = build(), build()
    sch = ST.Scheduler()
    b.slow = sch
    for _ in range(200):
        a.step()
        b.step()
    for _ in range(20000):
        a.step()
    rep = ST.advance(b, 20000, sched=sch)
    assert rep.jumped > 15000
    _compare(a, b)


# ───────────────────────────────── T27 skipping off, T28 halting ──
def _small_sleeper():
    w = _room(14, 12, 36)
    p = _person(w, 6, 6, z0=1)
    p["awake"], p["asleep"], p["blood_o2"] = True, True, 0.95
    return w


def test_skip_quiet_off_advance_never_jumps_and_equals_a_step_loop():
    a, b = _small_sleeper(), _small_sleeper()
    for w in (a, b):
        w.skip_quiet = False
    sch = ST.Scheduler()
    b.slow = sch
    for _ in range(1500):
        a.step()
    rep = ST.advance(b, 1500, sched=sch)
    assert rep.jumped == 0 and rep.stepped == 1500 and rep.halted_by == "target"
    assert ("skip", "world", "skipping is off") in rep.blockers_seen
    assert _bits(a) == _bits(b)
    assert a.persons[0]["blood_o2"] == b.persons[0]["blood_o2"]


def test_with_no_scheduler_advance_is_a_loop_of_step_and_says_so():
    a, b = _sleeper_room(), _sleeper_room()
    for _ in range(500):
        a.step()
    rep = ST.advance(b, 500)
    assert rep.jumped == 0 and rep.stepped == 500 and _bits(a) == _bits(b)
    assert rep.blockers_seen == (("skip", "world", "no scheduler"),)
    capped = ST.advance(b, 5000, max_fine_ticks=100)
    assert capped.stepped == 100 and capped.halted_by == "blocked"


def test_advance_stops_two_ticks_before_a_wake_and_the_event_ids_match_a_ticking_world():
    def build():
        w = _room()
        p = _out(w, cohb=0.2006)                       # delivered passes wake_o2 after ~3300 ticks
        return w
    a, b = build(), build()
    sch = ST.Scheduler()
    b.slow = sch
    for _ in range(60):
        a.step()
        b.step()
    first = a.events.next_id
    for _ in range(4500):
        a.step()
    ea = [(e.id, e.tick, e.verb, e.actor) for e in a.events.since(first - 1)]
    assert any(v == "wakes" for _i, _t, v, _a in ea)
    first_b = b.events.next_id
    rep = ST.advance(b, until=("woke", 0), sched=sch)
    assert rep.halted_by == "woke" and rep.jumped > 1500
    eb = [(e.id, e.tick, e.verb, e.actor) for e in b.events.since(first_b - 1)]
    assert eb == [e for e in ea if e[0] <= eb[-1][0]]
    wake_tick = next(t for _i, t, v, _a in ea if v == "wakes")
    assert b.tick - 1 >= wake_tick and b.tick <= wake_tick + 8      # the tick laws played it


def test_a_due_event_ends_the_jump_two_ticks_before_it_and_pre_fires_it_on_its_tick():
    w = _room()
    _out(w, cohb=0.30)
    sch = ST.Scheduler()
    w.slow = sch
    sch.add_drive("T", ("const", -10.0))
    due = 5000
    p = {"mode": "ice", "K": 2.2, "R0": 0.0, "rho": 917.0, "L_f": 334e3, "Tf": 0.0}
    S = 10.0 * (due * TICK_S) / (917.0 * 334e3)
    level = math.sqrt(2 * 2.2 * S)
    pid = sch.add("front", "owner", ("world",), p, (0.0,), 0, drives=(("T", "T"),),
                  marks=(ST.Watchpoint("x", level, "up", "ice-skin", "note", True),))
    for _ in range(60):
        w.step()
    assert sch.next_due() is not None and abs(sch.next_due() - due) <= 1
    rep = ST.advance(w, 8000, sched=sch)
    notes = [n for n in sch.notes if n[2] == "crosses"]
    assert len(notes) == 1 and abs(notes[0][0] - due) <= 1
    assert rep.jumped > 2000 and rep.t_to == 60 + 8000 and rep.halted_by == "target"
    assert sch.observe(pid, "x", w.tick) > level - 1e-9                       # the owner is lazy: read at any tick


# ─────────────────────────────────────── T30 an unknown stamp ──
class StampWorld(SlowWorld):
    """A private tick stamp nobody listed in EXCLUDE."""

    def _tick_start(self):
        super()._tick_start()
        self._mystamp = self.tick


def test_an_unlisted_tick_stamp_makes_the_world_not_quiet_and_names_the_key():
    w = _room(cls=StampWorld)
    for _ in range(3):
        w.step()
    assert Q.quiet(w).ok                               # the cheap gates cannot see it
    q = Q.quiet(w, probe_=True)
    assert not q.ok
    assert any(b.gate == "probe" and b.place == "_mystamp" for b in q.blockers)
    assert "_mystamp" in q.changed


def test_an_unlisted_person_stamp_is_found_by_the_probe_too():
    class W(SlowWorld):
        def _tick_start(self):
            super()._tick_start()
            for p in self.persons:
                p["_last_poke"] = self.tick

    w = _room(cls=W)
    _out(w)
    _settle(w)
    q = Q.quiet(w, probe_=True)
    assert not q.ok and any("persons[0]._last_poke" == b.place for b in q.blockers)


def test_the_digest_sees_an_array_a_person_a_nested_object_and_ignores_the_exclude_rows():
    w = _room()
    _out(w)
    _settle(w)
    d0 = Q.digest(w)
    assert Q.digest(w) == d0 and "E" in d0 and "persons[0].known" in d0
    assert "tick" not in d0 and "persons[0].cohb" not in d0 and "_burning_now" not in d0
    w.E[3, 3, 3] = 1.0
    assert Q.diff(d0, Q.digest(w)) == ["E"]
    w.E[3, 3, 3] = 0.0
    w.persons[0]["name"] = "someone else"
    assert Q.diff(d0, Q.digest(w)) == ["persons[0].name"]
    w.persons[0]["name"] = "the person"
    w.persons[0]["cohb"] += 0.01                       # a twin key: the digest leaves it out
    w.tick += 7
    assert Q.diff(d0, Q.digest(w)) == []
    w.matter.mpm.t += 5.0                              # the solver clock is on the list
    assert Q.diff(d0, Q.digest(w)) == []


def test_horizon_lists_the_wake_the_reflexes_stamps_and_the_scheduler_due():
    w = _room()
    p = _out(w, cohb=0.2006)
    _settle(w)
    sch = ST.Scheduler()
    w.slow = sch
    p["_chose"] = {"stand": w.tick + 77}
    p["_reflex"] = {"flees": w.tick + 55}
    p["intent"] = {"wake_at": w.tick + 66}
    sch.add_drive("T", ("table", [(w.tick + 99, 1.0), (w.tick + 500, 2.0)]))
    q = Q.quiet(w)
    hz = dict((tag, t) for t, tag in Q.horizon(w, [(p, TW.person_twins(w, p)[0])], sch))
    assert hz["chose:stand"] == w.tick + 77 and hz["reflex:flees"] == w.tick + 55
    assert hz["wake_at:0"] == w.tick + 66 and hz["kink:T"] == w.tick + 99
    assert any(tag.startswith("wakes") for tag in hz)
    assert [t for t, _ in Q.horizon(w, [], sch)] == sorted(t for t, _ in Q.horizon(w, [], sch))


# ─────────────────────── T29 the soundness fuzz (slow, deterministic) ──
def _fuzz_world(seed):
    """A quiet-biased world: furniture, a door, a settled puddle, a burnt-out
    fire, a cooled hearth, and some of: a sleeper, an unconscious person, nobody."""
    rng = np.random.default_rng(seed)
    w = _room()
    for _ in range(int(rng.integers(1, 4))):
        x, y = int(rng.integers(3, 20)), int(rng.integers(3, 16))
        w.fill(x, x + int(rng.integers(1, 4)), y, y + int(rng.integers(1, 4)), 1,
               int(rng.integers(2, 6)), (WOOD, STONE)[int(rng.integers(0, 2))],
               frac=float(rng.choice([0.5, 1.0])))
    if rng.random() < 0.6:
        w.mat[0, 8:12, 1:14] = AIR                       # a doorway
        w.smass[0, 8:12, 1:14] = 0.0
    if rng.random() < 0.5:
        for x in range(4, 8):
            for y in range(4, 8):
                w.pour(x, y, 1, WATER, 800.0)
    if rng.random() < 0.5:
        w.fill(16, 20, 12, 16, 1, 2, CHAR, frac=0.3)     # a burnt-out fire
        w.fill(16, 20, 12, 16, 2, 3, ASH, frac=0.3)
    kind = int(rng.integers(0, 3))
    if kind == 1:
        p = _person(w, 10, 10, z0=1)
        p["awake"], p["asleep"], p["blood_o2"] = True, True, 0.9
    elif kind == 2:
        _out(w, cohb=float(0.25 + 0.1 * rng.random()), at=(11, 11))
    return w, kind


@pytest.mark.slow
def test_soundness_fuzz_finds_no_quiet_but_changed():
    """About 20 worlds x 3 probe positions of one PERIOD each. When quiet(w).ok
    after the probe, a spare copy is stepped 600 more ticks and the digest must
    not move. Incompleteness (a world that is quiet and is not said to be) is
    reported as a rate, never as a failure."""
    positives = trials = 0
    bad = []
    for seed in range(20):
        w, kind = _fuzz_world(seed)
        _settle(w, 200)
        for pos in range(3):
            trials += 1
            q = Q.quiet(w, probe_=True)
            if not q.ok:
                for _ in range(300):
                    w.step()
                continue
            positives += 1
            spare = _room()
            spare.restore(w.snapshot())
            d0 = Q.digest(spare)
            for _ in range(Q.PERIOD):
                spare.step()
            moved = Q.diff(d0, Q.digest(spare))
            if moved:
                bad.append((seed, pos, moved[:4]))
            for _ in range(Q.PERIOD):
                w.step()
    rate = positives / trials
    print(f"quiet positives: {positives} of {trials} probes ({rate:.0%})")
    assert not bad, f"quiet but changed: {bad}"
    assert positives >= 20, f"too few quiet worlds to prove anything ({positives}/{trials})"
