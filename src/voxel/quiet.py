"""THE QUIET DETECTOR (wave 13, T5, spec section 6).

"Is every fine law provably asleep (tier 0) or owned by a jump (tier 1), and
until when?" Answered in two steps, and it READS the world, never writes it
(the probe steps the world, but those are the same ticks the world would run):

  1. CHEAP GATES, mirrored from `sim.World.step`: a few reductions that find the
     things that are obviously not quiet (a flame, a falling body, a moving
     cell, smoke, an awake person) and NAME them. Each lane that adds a skip
     gate to step() adds one line to QUIET_GATES.
  2. THE PROBE: stillness is PROVED, not counted. When every cheap gate is
     closed, one real PERIOD of ticks is run with a digest of the whole state
     before and after. A deterministic law on an identical state and an
     identical tick phase does the same thing next tick, so the world stays at
     this fixed point. The digest also says WHICH key changed, which becomes
     the blocker evidence. A private stamp nobody listed makes the digest
     differ, so the answer is "not quiet" (safe failure), never "quiet but
     changed".

PERIOD is the least common multiple of every `tick % n` / `tick // n` gate in
the laws, computed from the constants (8 in `_law_life`'s crumple test, WILL
look_every / scan_every / decide_every, and the optional RAD_EVERY,
GAS_GEO_TICKS and the body clock of T7). Today: lcm(6, 8, 4, 25, 10, 20, 40)
= 600. tests/test_quiet.py greps every such expression with the ast module so
a literal cannot hide.

THE EXCLUDE TABLE (the audited surface of the probe) lists what the digest
leaves out, with "why harmless" and what a jump must do about it.
"""
import hashlib
import math
import struct
from collections import namedtuple

import numpy as np

from . import slowtwins
from .materials import SMOKE_STILL, TICK_S, _REACTIVE

Blocker = namedtuple("Blocker", "gate place reason")


class Quiet:
    """The answer. `ok` is quiet now (after the probe when it was run); `until`
    the tick of the next thing that can happen (jump to until - 2 at most);
    `blockers` the reasons when not ok; `owned` the thermal regions a tier-1
    jump will own; `plan` the (person, twins) the jump will advance."""
    __slots__ = ("ok", "until", "blockers", "horizon", "owned", "tier", "plan",
                 "changed", "rates")

    def __init__(self, ok, until, blockers=(), horizon=(), owned=(), tier=0,
                 plan=(), changed=(), rates=None):
        self.ok, self.until = ok, until
        self.blockers, self.horizon, self.owned = tuple(blockers), tuple(horizon), tuple(owned)
        self.tier, self.plan = tier, tuple(plan)
        self.changed, self.rates = tuple(changed), rates or {}

    def __repr__(self):
        return f"Quiet(ok={self.ok}, until={self.until}, blockers={list(self.blockers)[:3]})"


# ───────────────────────────────────────────────────────────── PERIOD ──
# Gates that live in files other than the base tree or that a lane declares by
# name. A value found by getattr in the named module wins over the declared one
# (`attribute` may be dotted: "OutsideMixin.CLIMATE_EVERY" is read off a class).
# A declared value of 0 says "this time-dependence is NOT a tick gate": the
# row exists so that the scan of tests/test_quiet.py can see it was thought
# about, and the least common multiple leaves it out.
DECLARED_PERIODS = (
    # (name, module, attribute, declared value, where it is used)
    ("RAD_EVERY", "heat", "RAD_EVERY", 6, "heat._law_radiate runs every RAD_EVERY ticks"),
    ("GAS_GEO_TICKS", "gasflow", "GAS_GEO_TICKS", 20, "gasflow reads the pores every GAS_GEO_TICKS"),
    ("BODY_CLOCK", "bodyheat", "BODY_PERIOD", 40,
     "bodyheat._law_body_heat `(tick + pid) % period`: a body second, 40 ticks; the pid only "
     "shifts the phase, never the period"),
    ("CRUMPLE", None, None, 8, "physiology._law_life `self.tick % 8` (a literal)"),
    ("CLIMATE_EVERY", "outside_world", "OutsideMixin.CLIMATE_EVERY", 40,
     "outside_world._law_climate `tick % CLIMATE_EVERY`: T_out is read once a second. (Its other "
     "clocks, ENV_EVERY 8 and STATIC_EVERY 40, are 'at least this many ticks since the last "
     "look', stamped on a cache outside the world, not a `tick %` gate.)"),
    ("CLIMATE_DAY", "climate", None, 0,
     "climate.SimpleClock.year_fraction/hour_of_day: `% 1.0` and `% 24.0` of a FLOAT day "
     "coordinate. Not a tick gate. Time reaches the world as T_out in steps, through the "
     "CLIMATE_EVERY read; a driven outside keeps the heat gate (G3) closed, and the "
     "scheduler's drive kinks (sunrise, sunset) bound a jump."),
)

# The tick gates the AST scan of tests/test_quiet.py cannot turn into an
# integer (the period is a local name, a class attribute or a float modulus),
# each with the DECLARED_PERIODS row that covers it. The scan accepts exactly
# these and warns about any other; the key is (file, text of the right operand).
DECLARED_GATES = {
    ("bodyheat.py", "period"): "BODY_CLOCK",
    ("outside_world.py", "self.CLIMATE_EVERY"): "CLIMATE_EVERY",
    ("climate.py", "1.0"): "CLIMATE_DAY",
    ("climate.py", "24.0"): "CLIMATE_DAY",
}


def _declared_value(mod, attr):
    """The value of `attr` (dotted allowed) in the module `mod` of this
    package, or None if there is no such module or name."""
    import importlib
    try:
        got = importlib.import_module("." + mod, __package__)
        for part in attr.split("."):
            got = getattr(got, part)
        return got
    except (ImportError, AttributeError):
        return None


def period_sources():
    """[(name, value, where)] for every tick gate this tree has. A value of 0
    is a declared non-gate (see DECLARED_PERIODS)."""
    from .policy import WILL
    out = [("look_every", int(WILL["look_every"]), "perceive/will `tick % look_every`"),
           ("scan_every", int(WILL["scan_every"]), "will gaze sweep"),
           ("decide_every", int(WILL["decide_every"]), "will/motion `tick // decide_every`")]
    for name, mod, attr, val, where in DECLARED_PERIODS:
        got = _declared_value(mod, attr) if mod is not None and attr is not None else None
        out.append((name, int(got if got else val), where))
    return out


def period():
    """The least common multiple of every tick gate, recomputed from the
    constants each call (so a wider look_every widens it)."""
    p = 1
    for _n, v, _w in period_sources():
        if v > 0:
            p = p * v // math.gcd(p, v)
    return p


PERIOD = period()


# ──────────────────────────────────────────────────────── THE GATES ──
QUIET_GATES = []


def register_gate(name, fn):
    """fn(w, tier=0) -> list[Blocker]. One line per lane that adds a skip gate."""
    QUIET_GATES.append((name, fn))
    return fn


def _first(mask):
    hit = np.argwhere(mask)
    return tuple(int(v) for v in hit[0]) if len(hit) else None


def gate_matter(w, tier=0):
    """G1: bodies mid-topple, cells the particle solver moves, awake particles."""
    out = []
    if w.bodies:
        out.append(Blocker("matter", "bodies", f"{len(w.bodies)} body(ies) mid-topple"))
    if w.moving.any():
        out.append(Blocker("matter", _first(w.moving), "cells the particle solver moves"))
    m = getattr(w.matter, "mpm", None)
    if m is not None and len(m.x) and bool((m.nap < 0).any()):
        out.append(Blocker("matter", "mpm", "awake particles"))
    b = getattr(w.matter, "bonds", None)
    if b is not None and len(getattr(b, "pm", ())):
        out.append(Blocker("matter", "bonds", "bonded matter present"))
    return out


def gate_fluids(w, tier=0):
    """G2: drops in flight, and a reactive fluid with potency (an acid keeps its
    own slow clock, n ticks apart). A settled puddle is left to the probe."""
    out = []
    if w.drops:
        out.append(Blocker("fluids", "drops", f"{len(w.drops)} parcel(s) in flight"))
    for f in _REACTIVE:
        has = (w.fl == f) & (w.fvol > 1.0) & (w.fpot > 0.01)
        if has.any():
            out.append(Blocker("fluids", _first(has), "reactive fluid with potency"))
    return out


def gate_heat(w, tier=0):
    """G3: thermostats, igniters, a flame, and (tier 0) any deviation of E from
    zero. Under tier 1 a hot window is not a blocker: the net owns it, and
    slownet's refusals speak for it."""
    out = []
    if w.thermostats:
        out.append(Blocker("heat", tuple(w.thermostats[0][:3]), "a thermostat block pins a temperature"))
    ig = getattr(w, "igniters", None)
    if ig:
        out.append(Blocker("heat", "igniters", f"{len(ig)} igniter(s)"))
    # `burning()` goes through T() and heat_capacity(), which bump _matter_gen:
    # a detector that reads must not move a counter the laws compare. Nothing
    # can burn with E all zero (every cell is at 20 C), so it is not asked then.
    if bool(w.E.any()):
        burning = w.burning()
        if burning.any():
            out.append(Blocker("heat", _first(burning), f"flame: {int(burning.sum())} burning cell(s)"))
    # `_unsettled(pad)` is the world's own answer to "what is not at rest":
    # (box, reasons) (fields.py). With no outside it is the window of E != 0 and
    # no reasons; a world with an outside says WHY ('off_rest', 'interface',
    # 'sky_drive'), and each reason is a blocker. A box with no reason is the
    # old question: tier 0 asks that E be zero everywhere.
    uns = getattr(w, "_unsettled", None)
    box, reasons = uns(0) if callable(uns) else (None, [])
    for why in reasons:
        out.append(Blocker("heat", "world", str(why)))
    if not reasons and tier == 0 and (box is not None or not callable(uns)):
        nz = int(np.count_nonzero(w.E))
        if nz:
            out.append(Blocker("heat", _first(w.E != 0), f"E != 0 in {nz} cells"))
    return out


def gate_smoke(w, tier=0):
    """G4: smoke above SMOKE_STILL."""
    m = w.smoke > SMOKE_STILL
    if m.any():
        return [Blocker("smoke", _first(m), f"smoke in {int(m.sum())} cells")]
    return []


def gate_air(w, tier=0):
    """G5: oxygen not even, pressure or wind present. (Warm-air wind that is a
    function of E is THERMAL_DRIVEN: under tier 1 it is the net's.)"""
    out = []
    if w.o2 is None:
        return [Blocker("air", "world", "the first step has not been taken")]
    if w._mixing():
        sink = getattr(w, "airsink", None)
        # wave 13, C10: a sealed-room sink owns the room's oxygen under tier 1. The room's air is drawn down as
        # one body of gas; what blocks is a room whose own air is not even (a plume of smoke, a draught).
        if not (tier >= 1 and hasattr(sink, "mixing") and not sink.mixing()):
            out.append(Blocker("air", "o2", "oxygen is not even"))
    if tier >= 1:               # warm-air wind is a function of E: the net owns it (THERMAL_DRIVEN)
        return out
    if w.p_add.any():
        out.append(Blocker("air", _first(w.p_add != 0), "pressure injected"))
    if w.pcell is not None:
        if w.pcell.any() or any(bool(v.any()) for v in w.vcell):
            out.append(Blocker("air", "gas cells", "gas moving"))
    return out


def gate_sounds(w, tier=0):
    """G6: sounds live for two ticks."""
    if w.sounds:
        return [Blocker("sounds", "world", f"{len(w.sounds)} sound(s) alive")]
    return []


def gate_persons(w, tier=0, plan=None):
    """G8: every person is dead, safe, unconscious or asleep; its twin applies;
    nothing pending (a percept, a reflex due, a chosen-act stamp in the
    future). AWAKE PERSONS ARE NEVER QUIET: will appends to p["_trail"] every
    tick, sweeps the gaze, runs the look-around."""
    out = []
    for p in w.persons:
        if not p["alive"] or p["safe"]:
            continue
        who = p["pid"]
        if p["awake"] and not p.get("asleep"):
            out.append(Blocker("persons", who, "awake person"))
            continue
        if p.get("_react"):
            out.append(Blocker("persons", who, "a percept is pending"))
        rf = p.get("_reflex")
        if rf:
            out.append(Blocker("persons", who, f"reflex pending: {sorted(rf)}"))
        ch = p.get("_chose")
        if ch and any(t > w.tick for t in ch.values()):
            out.append(Blocker("persons", who, "a chosen-act stamp is still in the future"))
        twins, why = slowtwins.person_twins(w, p)
        if twins is None:
            out.append(Blocker("persons", who, f"no twin: {why}"))
        elif plan is not None:
            plan.append((p, twins))
    return out


register_gate("matter", gate_matter)
register_gate("fluids", gate_fluids)
register_gate("heat", gate_heat)
register_gate("smoke", gate_smoke)
register_gate("air", gate_air)
register_gate("sounds", gate_sounds)


def cheap_blockers(w, tier=0, plan=None):
    out = []
    for _name, fn in QUIET_GATES:
        out += fn(w, tier)
    out += gate_persons(w, tier, plan)
    return out


# ───────────────────────────────────────────────── THE DIGEST + EXCLUDE ──
class Exclude:
    """One audited line: what the digest leaves out, why that is harmless, and
    what a jump must do about it. kind: 'set' the jump sets it; 'twin' a twin
    advances it; 'clock' a float that grows at a steady rate (the jump adds the
    rate measured over the probe); 'countdown' decremented to a floor of 0;
    'ignore' recomputed on demand."""
    __slots__ = ("pattern", "kind", "why")

    def __init__(self, pattern, kind, why):
        self.pattern, self.kind, self.why = pattern, kind, why

    def matches(self, path):
        return path == self.pattern or (
            "*" in self.pattern and _glob(self.pattern, path))


def _glob(pattern, path):
    head, tail = pattern.split("*", 1)
    if not path.startswith(head):
        return False
    rest = path[len(head):]
    end = rest.find("]")
    return end >= 0 and rest[end + 1:] == tail.lstrip("]")


EXCLUDE = (
    Exclude("tick", "set", "the clock itself; the jump sets it"),
    Exclude("persons[*].blood_o2", "twin", "advanced by the breath twin (discrete factor)"),
    Exclude("persons[*].met", "twin", "advanced by the breath twin"),
    Exclude("persons[*].smoke", "twin", "advanced by the breath twin"),
    Exclude("persons[*].cohb", "twin", "advanced by the breath twin"),
    # wave 13, C10: the body-heat twin (slowbody.py) carries a body's compact model, its stores and its
    # ledgers; the sealed-room sink carries what the room was given. Both are advanced by the jump in chunks
    # and ticked by the body law otherwise, so they are not a fixed point.
    Exclude("persons[*].heat", "twin", "the body's compact model, advanced by the body_heat twin"),
    Exclude("persons[*].skin", "set", "max flesh temperature, written by _law_life each tick from the lattice; "
            "the first tick after a jump rewrites it"),
    Exclude("airsink", "twin", "the sealed-room sink's books; the jump applies a chunk's heat and gases"),
    Exclude("_bh_inject", "twin", "named injections of the body law (warm-up, food); the twin adds none"),
    Exclude("_bh_runtime", "ignore", "caches of the body law (rays, geometry), rebuilt on demand"),
    Exclude("persons[*]._claim_tick", "ignore",
            "the tick of the last body claim; claims are redone when it is stale"),
    Exclude("matter.mpm.t", "clock", "particle solver time; grows by dt each tick while particles sleep"),
    Exclude("matter.bonds.t", "clock", "bonded solver time; grows by dt each tick"),
    Exclude("_burning_now", "ignore",
            "(tick, lit, fire_at) stamped by _law_perceive and consumed by _law_will in the "
            "same tick; a fire makes the heat gate close, so only the stamp can differ"),
    Exclude("_torque_sleep", "countdown", "cooldown after a wedged landing; harmless at 0, a jump counts it down"),
    # base19 (B0): the torque law's memory of the last solid layout it checked (support.py _law_torque:
    # `if self._torque_solid is not None and (solid == self._torque_solid).all(): return`). It only decides
    # whether the check is repeated; a stable layout checks to the same answer. Several laws reset it to
    # None (pose, bodies, motion, hands), so after a run of ticks it holds the layout or None depending on
    # where in the last tick the last reset fell. A jump leaves it as it was; the next tick refills it.
    Exclude("_torque_solid", "ignore", "a cache of the last solid layout the torque law checked; a jump "
            "leaves it, the next tick refills it"),
    # base19 (B0): the radiation pass of heat._law_radiate (FIRE, wave 12) keeps a stamp and a bank. A pass runs
    # every RAD_EVERY ticks while the heat gate is open (E != 0), so over one period the stamp ALWAYS changes in a
    # warm world, and without these rows no warm world could ever be quiet at tier 1 (the cooled-hearth test of
    # test_slownet.py ticked for ever). Tier 0 has E = 0: no pass runs and the stamp does not move.
    Exclude("_rad_last", "set", "tick of the last radiation pass; a tier 1 jump sets it to the tick of the last "
            "pass it skipped (slowtime._jump), a tier 0 jump (E = 0, no pass runs) leaves it"),
    Exclude("_rad_bank", "ignore", "joules the flames radiated since the last pass; 0.0 unless a flame burns, "
            "and a flame is a blocker (gate G3)"),
)

# What a tier-1 jump owns: fields that are functions of E and recomputed by the
# first ticks after it. Excluded from the digest inside the net's region only.
THERMAL_DRIVEN = ("E", "skin", "p_add", "pcell", "vcell")
# and what a tier-1 jump books or stamps instead of leaving: left out of the
# digest wholesale at tier 1 (heat radiated into `shed`, the release-rate stamp,
# the gas particles warm air moves)
THERMAL_BOOKS = ("shed", "_hrr", "pcell", "vcell")

_SKIP_TYPES = (type(len), type(math), type(lambda: 0))


def _excluded(path, extra=()):
    for ex in EXCLUDE:
        if ex.matches(path):
            return True
    return path in extra


def _hash_obj(h, obj, path, seen, excl):
    if _excluded(path, excl):
        return
    if isinstance(obj, np.ndarray):
        h.update(b"A" + str(obj.dtype).encode() + str(obj.shape).encode())
        if obj.size == 0:
            return
        if obj.dtype.hasobject:
            h.update(repr(obj.tolist()).encode())
            return
        h.update(memoryview(np.ascontiguousarray(obj)).cast("B"))
        return
    if obj is None or isinstance(obj, (bool, int, str, bytes, np.integer, np.bool_)):
        h.update(b"S" + repr(obj).encode())
        return
    if isinstance(obj, (float, np.floating)):
        h.update(b"F" + struct.pack("<d", float(obj)))
        return
    if callable(obj) and not hasattr(obj, "__dict__"):
        return
    if isinstance(obj, type) or isinstance(obj, _SKIP_TYPES) or type(obj).__name__ in ("World", "module", "weakref"):
        return
    oid = id(obj)
    if oid in seen:
        h.update(b"C")
        return
    seen.add(oid)
    if isinstance(obj, dict):
        h.update(b"D%d" % len(obj))
        for k in obj:
            h.update(repr(k).encode())
            _hash_obj(h, obj[k], f"{path}.{k}", seen, excl)
    elif isinstance(obj, (list, tuple)) or type(obj).__name__ == "deque":
        h.update(b"L%d" % len(obj))
        for i, x in enumerate(obj):
            _hash_obj(h, x, f"{path}[{i}]", seen, excl)
    elif isinstance(obj, (set, frozenset)):
        h.update(b"T%d" % len(obj))
        for x in sorted(obj, key=repr):
            h.update(repr(x).encode())
    elif hasattr(obj, "__dict__") or hasattr(type(obj), "__slots__"):
        h.update(b"O" + type(obj).__name__.encode())
        names = list(getattr(obj, "__dict__", {}))
        for cls in type(obj).__mro__:
            names += [s for s in getattr(cls, "__slots__", ()) if s not in names]
        for k in names:
            if k.startswith("__"):
                continue
            h.update(k.encode())
            _hash_obj(h, getattr(obj, k, None), f"{path}.{k}", seen, excl)
    else:
        h.update(b"R" + repr(obj).encode())
    seen.discard(oid)


def _children(obj):
    if isinstance(obj, dict):
        return [(str(k), v) for k, v in obj.items()]
    if hasattr(obj, "__dict__") and type(obj).__name__ != "World":
        return list(vars(obj).items())
    return None


def digest(w, tier=0, region=None, extra_exclude=()):
    """{key: hex} of the whole snapshot state minus EXCLUDE. Keys are
    top-level attribute names, 'persons[i].key', and 'matter.<attr>' so a
    difference names its place. Under tier 1 the THERMAL_DRIVEN arrays are
    hashed with `region` (a tuple of slices) zeroed: they may change inside
    the net's box and nowhere else."""
    out = {}
    skip = set(getattr(w, "_NOT_STATE", ())) | {"slow"}      # the scheduler is not physics
    excl = set(extra_exclude)

    def put(path, obj):
        if _excluded(path, excl) or (tier >= 1 and path in THERMAL_BOOKS):
            return
        h = hashlib.blake2b(digest_size=8)
        if tier >= 1 and region is not None and (path in THERMAL_DRIVEN or (
                path == "o2" and hasattr(getattr(w, "airsink", None), "mixing"))) \
                and isinstance(obj, np.ndarray) and obj.shape[:3] == w.shape:
            obj = obj.copy()
            obj[region] = 0
        _hash_obj(h, obj, path, {id(w)}, excl)
        out[path] = h.hexdigest()
    for k, v in w.__dict__.items():
        if k in skip:
            continue
        if k == "persons":
            for i, p in enumerate(v):
                for pk, pv in p.items():
                    put(f"persons[{i}].{pk}", pv)
            put("persons.n", len(v))
        elif k == "matter":
            for mk, mv in vars(v).items():
                if mk == "w":
                    continue
                kids = _children(mv)
                if kids is not None and not isinstance(mv, np.ndarray):
                    for ck, cv in kids:
                        put(f"matter.{mk}.{ck}", cv)
                else:
                    put(f"matter.{mk}", mv)
        else:
            put(k, v)
    return out


def diff(d0, d1):
    """The keys that differ between two digests, sorted."""
    return sorted(k for k in set(d0) | set(d1) if d0.get(k) != d1.get(k))


def read_clocks(w):
    """The values of the 'clock' and 'countdown' EXCLUDE rows, by path."""
    out = {}
    m = getattr(w.matter, "mpm", None)
    if m is not None:
        out["matter.mpm.t"] = float(m.t)
    b = getattr(w.matter, "bonds", None)
    if b is not None and hasattr(b, "t"):
        out["matter.bonds.t"] = float(b.t)
    out["_torque_sleep"] = float(getattr(w, "_torque_sleep", 0))
    return out


# ───────────────────────────────────────────────────────── THE HORIZON ──
def horizon(w, plan=(), sched=None):
    """[(tick, tag)] of everything that can happen next, sorted. A jump goes to
    the first one minus 2 (F4: the margin lets the tick laws play the event)."""
    now = w.tick
    out = []
    s = sched if sched is not None else getattr(w, "slow", None)
    if s is not None:
        top = s.next_due()
        if top is not None:
            out.append((top, "due"))
        for t, tag in s.drive_kinks(now):
            out.append((t, tag))
    for p in w.persons:
        if not p["alive"] or p["safe"]:
            continue
        it = p.get("intent") or {}
        wk = it.get("wake_at")
        if wk is not None and wk > now:
            out.append((wk, f"wake_at:{p['pid']}"))
        for kind, due in (p.get("_reflex") or {}).items():
            if due > now:
                out.append((due, f"reflex:{kind}"))
        for tag, exp in (p.get("_chose") or {}).items():
            if exp > now:
                out.append((exp, f"chose:{tag}"))
    for p, twins in plan:
        for _name, tw, ctx in twins:
            for k, tag in tw.horizon(w, p, ctx):
                out.append((now + k, f"{tag}:{p['pid']}"))
    return sorted(out)


# ─────────────────────────────────────────────────────────── THE ASK ──
def inside_point(n):
    """A tick strictly inside a period of `n` ticks at which the probe looks
    as well as at the two ends: about the middle, moved up to the first tick
    that shares no factor with `n`, so that it falls at a different phase of
    EVERY tick gate (the middle of 600 is a whole number of 4, 6, 10, 20 and 25
    ticks and would look at those gates in the phase they started in). 0 when
    the period is too short to have an inside."""
    if n < 3:
        return 0
    m = n // 2
    while math.gcd(m, n) != 1:
        m += 1
    return m


def probe(w, tier=0, region=None, sched=None):
    """Run ONE REAL PERIOD with a digest before, INSIDE and after. Returns
    (ok, changed_keys, clock_rates). One warm-up tick runs first so lazily
    built caches exist before the first digest. Steps the world.

    The state at the end of a period equals the state at its start whenever the
    world sits at a fixed point, but also when it wanders off and comes back
    (a cycle of exactly the period, or something done at one phase and undone
    at another). The look INSIDE the period (`inside_point`) must agree with
    the start too, and it catches what leaves and returns between the two
    ends. It costs one more digest, not one more tick."""
    w.step()
    d0 = digest(w, tier, region)
    c0 = read_clocks(w)
    n = period()
    mid = inside_point(n)
    for _ in range(mid):
        w.step()
    dm = digest(w, tier, region) if mid else d0
    for _ in range(n - mid):
        w.step()
    d1 = digest(w, tier, region)
    c1 = read_clocks(w)
    changed = sorted(set(diff(d0, dm)) | set(diff(d0, d1)))
    rates = {k: (c1[k] - c0[k]) / n for k in c0}
    return (not changed), changed, rates


def quiet(w, tier=0, probe_=False, sched=None):
    """quiet(w, tier) -> Quiet. Without the probe it answers only 'the cheap
    gates are closed' (ok False when any is open, with the reasons). With the
    probe (probe_=True) it also steps one real period and proves the state is
    a fixed point."""
    if not w.skip_quiet:
        return Quiet(False, w.tick, [Blocker("skip", "world", "skipping is off")], tier=tier)
    plan = []
    bl = cheap_blockers(w, tier, plan)
    owned = ()
    if tier >= 1 and not bl:
        bl += slowbody.sink_blockers(w, plan)
    if tier >= 1 and not bl:
        try:
            from . import slownet
            srcs, _o2 = slownet.collect_sources(w, plan)
            res = slownet.check(w, sched if sched is not None else getattr(w, "slow", None),
                                sources=srcs, skip=slowbody.owned_cells(w))
            owned, bl2 = res
            bl += bl2
        except ImportError:
            pass
    if bl:
        return Quiet(False, w.tick, bl, tier=tier, owned=owned)
    hz = horizon(w, plan, sched)
    until = hz[0][0] if hz else None
    if not probe_:
        return Quiet(True, until, (), hz, owned, tier, plan)
    region = owned[0] if owned else None
    ok, changed, rates = probe(w, tier, region, sched)
    plan2 = []
    bl = cheap_blockers(w, tier, plan2)              # every gate closed at BOTH ends
    if not ok:
        bl += [Blocker("probe", k, "changed over one period") for k in changed]
    if bl:
        return Quiet(False, w.tick, bl, tier=tier, owned=owned, changed=changed, rates=rates)
    hz = horizon(w, plan2, sched)
    until = hz[0][0] if hz else None
    return Quiet(True, until, (), hz, owned, tier, plan2, rates=rates)


def explain(w, tier=0):
    """One line per blocker: 'flame at (12, 9, 5): ...'. Empty string if quiet."""
    q = quiet(w, tier)
    if q.ok:
        return ""
    return "\n".join(f"{b.gate} at {b.place}: {b.reason}" for b in q.blockers)


from . import slowbody  # noqa: E402,F401  (registers the body_heat twin; a world without body heat never sees it)
