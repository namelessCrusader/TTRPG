"""Proof-scene builders for the slow-time topic (wave 13, T5, spec 12: P1, P2, P4).

Each `scene_*` returns a World (5 cm voxels) and the things a proof needs; each
`run_*` runs the proof and returns plain numbers. Nothing is switched on: they
use the scheduler (slowtime) and the breath twin (slowtwins) as they are.

    scene_p1   THE RESCUED MAN. One man, unconscious, COHb 0.30, blood O2 1.0,
               in clean air in a closed stone room. Without the scheduler he
               would be ticked for 187.2 sim-minutes (449,251 ticks) to see him
               wake; `run_p1` finds the tick by the twin (milliseconds).
               COHb 0.35 wakes at 258.4 minutes.
    scene_p2   THE INN NIGHT. A closed room, a cold hearth, a bed, four people
               set asleep by the scene (no law sets `asleep` yet, wave 15).
               `run_p2` jumps the night and says how many ticks it skipped.
    scene_p4   LOOK AND TOUCH. A bar of carbonating mortar kept by a SHADOW
               process on a DictHost. A look reads it and unfolds nothing; a
               person's hand on it unfolds it; 200 ticks after he leaves it
               folds again.

Every `scene_*` takes a `settle` count of ticks after the people are put in
(an unconscious man upright is laid down by the first tick divisible by 8, and
a body falling wakes sleepers by its thud, so the room must be still first).

A hot body in a hot room is NOT built here. The humidity functions now raise at
or above the boiling point of water at the total pressure, so a scene that feeds
a body from a world sink at 100 C or more must stay below it or catch the error.
"""
from .materials import AIR, ASH, CHAR, STONE, WOOD
from .scenes import _person
from .sim import World


class SlowWorld(World):
    """A World that hands each tick to `self.slow.pre(self)` (the scheduler),
    if it has one. `World` itself gets this seam from lane B1 (a class attribute
    `slow = None` and a call at the end of `_tick_start`); until then this
    subclass puts it there. When the base class already has the call, this adds
    nothing."""
    slow = None

    def _tick_start(self):
        super()._tick_start()
        if self.slow is not None and not _BASE_CALLS_SLOW:
            self.slow.pre(self)


def _calls_slow():
    import inspect
    try:
        return "slow.pre" in inspect.getsource(World._tick_start)
    except (OSError, TypeError):
        return False


_BASE_CALLS_SLOW = _calls_slow()


def _room(nx=24, ny=20, nz=40, vc=5):
    """A closed stone box: floor, four walls, ceiling, no exits."""
    w = SlowWorld(nx, ny, nz, voxel_cm=vc)
    w.fill(0, nx, 0, ny, 0, 1, STONE)
    w.fill(0, nx, 0, 1, 0, nz, STONE)
    w.fill(0, nx, ny - 1, ny, 0, nz, STONE)
    w.fill(0, 1, 0, ny, 0, nz, STONE)
    w.fill(nx - 1, nx, 0, ny, 0, nz, STONE)
    w.fill(0, nx, 0, ny, nz - 1, nz, STONE)
    w.exits = []
    return w


def _settle(w, n=40):
    """Step until nothing is toppling, then n more ticks."""
    for _ in range(3000):
        w.step()
        if not w.bodies:
            break
    for _ in range(n):
        w.step()


# ───────────────────────────────────────────────────────────── P1 ──
def scene_p1(cohb=0.30, settle=3):
    """-> (world, person). T5 spec P1: unconscious, perfusion 1, clean air."""
    w = _room()
    p = _person(w, 10, 10, z0=1)
    p["awake"], p["cohb"], p["blood_o2"] = False, cohb, 1.0
    _settle(w, settle)
    return w, p


def run_p1(cohb=0.30):
    """Minutes until he wakes, by the twin's horizon (no ticking). 187.2 for
    0.30 and 258.4 for 0.35; the exact tick counts are 449,251 and 620,049."""
    from . import slowtwins as TW
    from .materials import TICK_S
    w, p = scene_p1(cohb)
    twins, why = TW.person_twins(w, p)
    if twins is None:
        raise RuntimeError(why)
    tw, ctx = twins[0][1], twins[0][2]
    k, tag = tw.horizon(w, p, ctx)[0]
    return {"ticks": int(k), "minutes": k * TICK_S / 60.0, "event": tag}


# ───────────────────────────────────────────────────────────── P2 ──
def scene_p2(n_sleepers=4):
    """-> (world, [persons]). A 36 x 30 x 40 room: a dead hearth (stone, char,
    ash), a bed of wood, and sleepers lying about the floor. They are set
    asleep by the scene (`asleep` True, awake True, blood O2 0.95); a sleeper
    is not in anyone's way."""
    w = _room(36, 30, 40)
    w.fill(26, 30, 18, 24, 1, 3, STONE)                       # the hearth
    w.fill(27, 29, 19, 23, 3, 4, CHAR, frac=0.4)              # gone cold long ago
    w.fill(27, 29, 19, 23, 4, 5, ASH, frac=0.3)
    w.fill(20, 24, 5, 8, 1, 6, WOOD, frac=0.7)                # a bed frame
    spots = [(8, 8), (8, 14), (8, 20), (14, 14)][:n_sleepers]
    people = []
    for x, y in spots:
        p = _person(w, x, y, z0=1)
        p["awake"], p["asleep"], p["blood_o2"] = True, True, 0.95
        people.append(p)
    return w, people


def run_p2(ticks=20000, settle=120):
    """Jump `ticks` of the night and report. Returns the SkipReport's numbers."""
    from . import slowtime as ST
    w, _ = scene_p2()
    sch = ST.Scheduler()
    w.slow = sch
    for _ in range(settle):
        w.step()
    rep = ST.advance(w, ticks, sched=sch)
    return {"t_to": rep.t_to, "jumped": rep.jumped, "stepped": rep.stepped,
            "halted_by": rep.halted_by, "tier": rep.tier_used}


# ───────────────────────────────────────────────────────────── P4 ──
def scene_p4(box=(14, 5, 1, 18, 9, 5)):
    """-> (world, scheduler, host, pid). A mortar bar (a 1-D diffusion profile
    in a DictHost at `box`) under a SHADOW fick process. No person yet: the
    proof puts one beside it (`_person(w, 12, 7)` with `_eye` at 13, 7, 20)."""
    import numpy as np
    from . import slowtime as ST
    w = _room()
    s = ST.Scheduler()
    w.slow = s
    Cs = 0.1
    x = np.linspace(0.0, 0.02, 101)
    prof = Cs + 0.9 * np.exp(-((x - 0.01) / 0.003) ** 2) * np.sin(np.pi * x / 0.02) ** 2
    prof[0] = prof[-1] = Cs
    host = ST.DictHost("bar", x, prof, 0.02, Cs=Cs, D=1e-9, tick_s=1.0, box=box)
    s.add_host(host.name, host)
    pid = s.add("fick", "shadow", ("dict", host.name),
                {"L": 0.02, "D": 1e-9, "Cs": Cs, "bc": "dd"}, None, 0)
    return w, s, host, pid


def run_p4():
    """Look, then touch, then leave. -> (modes after look, touch, 200+ ticks)."""
    w, s, host, pid = scene_p4()
    for _ in range(3):
        w.step()
    s.fold(pid, w.tick)
    before = (s.procs[pid].state, s.procs[pid].t, s.procs[pid].ver)
    s.observe(pid, "center", w.tick + 5000)
    looked = (s.procs[pid].state, s.procs[pid].t, s.procs[pid].ver) == before
    mode_look = s.procs[pid].mode
    p = _person(w, 12, 7, z0=1)
    p["_eye"] = (13.0, 7.0, 20.0)
    w.step()
    mode_touch = s.procs[pid].mode
    p["asleep"] = True
    n = 0
    while s.procs[pid].mode == "live" and n < 400:
        w.step()
        n += 1
    return {"look_changed_nothing": looked, "mode_look": mode_look,
            "mode_touch": mode_touch, "ticks_to_fold": n, "mode_end": s.procs[pid].mode}


# ───────────────────────────────────────────────────────── P2 with bodies ──
def scene_inn_body(nx=60, ny=48, nz=40, n_sleepers=4, settle=300, sink=True, wall=8):
    """THE INN NIGHT WITH REAL BODIES (T5 tier 1, C10). A closed stone room of nx x ny x nz voxels inside,
    walls `wall` voxels thick (a thin wall at the lattice edge has nowhere to put heat, and the net cannot
    take a one-layer wall at 10 K, Biot), a dead hearth, `n_sleepers` asleep on the floor, each a heat engine
    in the lattice (BodyWorld), a scheduler and the sealed-room sink.
    -> (world, [persons]). The default room is 3 x 2.4 x 2 m inside (5 cm voxels), 14 m3 of air: four
    sleepers use 1 L of oxygen a minute between them."""
    from .airsink_world import RegionSink
    from .bodyworld import BodyWorld
    from . import slowtime as ST

    class _W(BodyWorld, SlowWorld):
        pass
    k = int(wall)
    X, Y, Z = nx + 2 * k, ny + 2 * k, nz + 2 * k
    w = _W(X, Y, Z, voxel_cm=5)
    w.fill(0, X, 0, Y, 0, Z, STONE)                                      # one block of masonry
    w.fill(k, k + nx, k, k + ny, k, k + nz, AIR)                         # the room is carved out of it
    w.exits = []
    w.fill(k + nx - 10, k + nx - 6, k + ny - 10, k + ny - 4, k, k + 2, STONE)           # the hearth
    w.fill(k + nx - 9, k + nx - 7, k + ny - 9, k + ny - 5, k + 2, k + 3, CHAR, frac=0.4)
    w.fill(k + nx - 9, k + nx - 7, k + ny - 9, k + ny - 5, k + 3, k + 4, ASH, frac=0.3)
    spots = [(8, 8), (8, 20), (8, 32), (20, 20)][:n_sleepers]
    people = []
    for x, y in spots:
        p = _person(w, k + x, k + y, z0=k)
        p["awake"], p["asleep"], p["blood_o2"] = True, True, 0.95
        people.append(p)
    w.slow = ST.Scheduler()
    if sink:
        w.airsink = RegionSink(w, tuple(slice(k, k + n) for n in (nx, ny, nz)))
    for _ in range(settle):
        w.step()
    return w, people
