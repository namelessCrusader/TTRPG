"""THE SLOW-TIME SCHEDULER (wave 13, T5, law L11, spec sections 3, 5, 7).

A heap of "next time something changes" and a record per slow process.

  OWNER   no tick law exists, so the closed form IS the law (a front, rust, a
          population, degree-days). Never "live": read lazily (`observe`) and
          fed new inputs when touched (`touch`).
  SHADOW  a closed-form twin of a tick law that exists (a box of cells as an
          rc_net; a record's node solver; the body-heat model). FOLDED while
          nothing disturbs it, LIVE (the tick law owns it) when something does,
          with an exact hand-over both ways.

A CLOSED FORM NEVER CAUSES A CONSEQUENCE. It moves a state to a point just
before a fine law's own threshold would trip, then hands back, and the fine law
does the rest. The only consequence of an OWNER form is a Watchpoint (a row)
that raises a "crosses" NOTE or a HANDBACK.

Everything stored is plain data (F10): slots, tuples, floats, interned params.
Forms, Drive kinds and Twins are found by NAME in registries and are never in
a snapshot. `Scheduler.__deepcopy__` copies 20,000 processes in tens of ms
because a Process is a bag of immutable fields.

THE HOOK. A World calls `self.slow.pre(self)` directly after `_tick_start()`.
Tests use a TEST-ONLY subclass whose `_tick_start` calls super first and then
`pre()`, so the order relative to the cause/disturbed/pending resets is the
same as the wiring's. With `w.slow = None` nothing here exists (F5).

DRIVER LEVEL ONLY: `advance`, `wait`, `until_clock` and `sleep` are for the
playtest harness, the DM loop and tests, never from inside a job (a job runs
inside `_law_will` inside `step()`, and `advance` would re-enter `step()`).
"""
import heapq
import math
from collections import deque

import numpy as np

from . import quiet as Q
from . import slowforms as F
from . import slowtwins as TW

TICK_S = F.TICK_S
FOLD_AFTER = 200                 # ticks (5 s) outside every watch set before a live SHADOW folds
RETRY = 200                      # ticks between fold attempts that were refused
BLOCK = 16                       # cells per block of the watch index
REACH_M = 1.0                    # a person's reach (u): the box a hand can touch
MARGIN = 2                       # cells: heat + buoyancy reach in ONE tick (pre sees last tick's windows)
HORIZON_MAX = 1 << 31            # ticks searched for a watchpoint crossing (~1.7 years)
PROBE_BACKOFF = 400              # ticks before a failed probe is tried again (10 s)
STEP_CHUNK = 40                  # real ticks per look while blocked


# ──────────────────────────────────────────────────────── plain data ──
class Watchpoint:
    """A row: the state key, the level, the direction, a tag and an action
    ('note' or 'handback'). Immutable by convention; shared, never copied."""
    __slots__ = ("key", "level", "dir", "tag", "action", "visible")

    def __init__(self, key, level, dir=None, tag="", action="note", visible=False):
        self.key, self.level, self.dir = key, level, dir
        self.tag, self.action, self.visible = tag, action, visible

    def __deepcopy__(self, memo):
        return self


class Process:
    """One slow thing. Plain data; every field is immutable (tuples, numbers,
    strings), so a copy is a field-by-field assignment."""
    __slots__ = ("id", "kind", "cls", "host", "params", "state", "t", "mode",
                 "drives", "marks", "ver", "since", "next_try", "tag")

    def clone(self):
        n = Process.__new__(Process)
        for f in Process.__slots__:
            setattr(n, f, getattr(self, f))
        return n

    def __deepcopy__(self, memo):
        return self.clone()


class Clock:
    """Wall-clock meaning for ticks: seconds since the epoch, the day and the
    year. 'until dawn' is time-of-day == dawn_s until the sun law replaces it."""
    __slots__ = ("epoch_s", "day_s", "year_s", "dawn_s")

    def __init__(self, epoch_s=0.0, day_s=86400.0, year_s=31557600.0, dawn_s=21600.0):
        self.epoch_s, self.day_s, self.year_s, self.dawn_s = epoch_s, day_s, year_s, dawn_s

    def seconds(self, tick):
        return self.epoch_s + tick * TICK_S

    def tod(self, tick):
        return self.seconds(tick) % self.day_s

    def ticks_until(self, tick, tod_s):
        """Ticks from `tick` to the NEXT time the clock reads tod_s (strictly after now)."""
        d = (tod_s - self.tod(tick)) % self.day_s
        if d <= 1e-9:
            d = self.day_s
        return int(math.ceil(d / TICK_S - 1e-9))

    def parse(self, text):
        if text == "dawn":
            return self.dawn_s
        hh, mm, *ss = (int(v) for v in str(text).split(":"))
        return hh * 3600.0 + mm * 60.0 + (ss[0] if ss else 0)


class SkipReport:
    """What an advance did. Only deterministic fields: equal worlds and equal
    calls give equal reports (E10); wall-clock cost goes to the perf log."""
    __slots__ = ("t_from", "t_to", "jumped", "stepped", "tier_used", "halted_by",
                 "blockers_seen", "residual", "jumps", "who")

    def __init__(self, t_from):
        self.t_from, self.t_to = t_from, t_from
        self.jumped = self.stepped = 0
        self.tier_used = "exact"
        self.halted_by = "target"
        self.blockers_seen = ()
        self.residual = 0.0
        self.jumps = ()
        self.who = None

    def key(self):
        return (self.t_from, self.t_to, self.jumped, self.stepped, self.tier_used,
                self.halted_by, self.blockers_seen, round(self.residual, 9), self.jumps)

    def __eq__(self, other):
        return isinstance(other, SkipReport) and self.key() == other.key()

    def __repr__(self):
        return (f"SkipReport({self.t_from}->{self.t_to}, jumped {self.jumped}, "
                f"stepped {self.stepped}, {self.tier_used}, halted by {self.halted_by})")

    def __deepcopy__(self, memo):
        return self


# ──────────────────────────────────────────────────────────── hosts ──
class Host:
    """The contract a SHADOW host meets (records, composition and layers meet
    it in phase B). `read()` the fine state as a dict, `write(coarse)` the
    state back, `totals()` the conserved amounts {key: float}, `box()` where it
    is (x0, y0, z0, x1, y1, z1 in cells) or None, `observe(key)` the live value."""

    def read(self):
        raise NotImplementedError

    def write(self, coarse):
        raise NotImplementedError

    def totals(self):
        raise NotImplementedError

    def box(self):
        return None

    def observe(self, key):
        raise NotImplementedError


class DictHost(Host):
    """A TEST DOUBLE host: a dict of node values with a toy implicit-diffusion
    fine law (Crank-Nicolson, vertex nodes, faces at Cs). Plain data."""

    def __init__(self, name, x, c, L, Cs=0.0, D=1e-9, tick_s=1.0, bc="dd",
                 box=None, species="c"):
        self.name = name
        self.x = np.asarray(x, float)
        self.c = np.asarray(c, float)
        self.L, self.Cs, self.D, self.tick_s, self.bc = L, Cs, D, tick_s, bc
        self.box_ = box
        self.species = species
        self.ticks_per = int(round(tick_s / TICK_S))      # world ticks per toy tick

    def read(self):
        return {"x": self.x.copy(), "c": self.c.copy()}

    def write(self, coarse):
        self.c = np.asarray(coarse, float).copy()

    def box(self):
        return self.box_

    def weights(self):
        return 0.5 * (np.r_[0.0, np.diff(self.x)] + np.r_[np.diff(self.x), 0.0])

    def totals(self):
        return {("mass", self.species): float(np.dot(self.weights(), self.c))}

    def observe(self, key):
        if key == "mass":
            return self.totals()[("mass", self.species)]
        if isinstance(key, tuple) and key[0] == "c":
            return float(np.interp(key[1], self.x, self.c))
        if key == "center":
            return float(np.interp(0.5 * self.L, self.x, self.c))
        raise KeyError(key)

    def tick(self, n=1):
        """n toy ticks of the fine law: Crank-Nicolson, vertex nodes, both
        faces held at Cs (bc 'dd' only). (I + r/2 T) u' = (I - r/2 T) u + r Cs
        (e_first + e_last), T = tridiag(-1, 2, -1), r = D dt / h^2."""
        N = len(self.x)
        m = N - 2
        h = float(self.x[1] - self.x[0])
        r = self.D * self.tick_s / (h * h)
        T = 2.0 * np.eye(m) - np.eye(m, k=1) - np.eye(m, k=-1)
        Minv = np.linalg.inv(np.eye(m) + 0.5 * r * T)
        P = Minv @ (np.eye(m) - 0.5 * r * T)
        bc = np.zeros(m)
        bc[0] += r * self.Cs
        bc[-1] += r * self.Cs
        bcv = Minv @ bc
        u = self.c[1:-1].copy()
        for _ in range(n):
            u = P @ u + bcv
        self.c = np.r_[self.Cs, u, self.Cs]


def box_overlap(a, b):
    return all(a[i] < b[i + 3] and b[i] < a[i + 3] for i in range(3))


def _grow(box, k):
    return (box[0] - k, box[1] - k, box[2] - k, box[3] + k, box[4] + k, box[5] + k)


def _box_of(sl, grow=0):
    """(x0, y0, z0, x1, y1, z1) from a slice triple (half-open), grown."""
    return _grow((sl[0].start or 0, sl[1].start or 0, sl[2].start or 0,
                  sl[0].stop, sl[1].stop, sl[2].stop), grow)


class _BlockIndex:
    """The watch index: a hash of blocks of BLOCK cells. Cost of an overlap
    query is O(min(blocks of the box, processes)), independent of world size."""

    def __init__(self):
        self.blocks = {}
        self.boxes = {}

    @staticmethod
    def _blocks(box):
        x0, y0, z0, x1, y1, z1 = box
        for i in range(x0 // BLOCK, (x1 - 1) // BLOCK + 1):
            for j in range(y0 // BLOCK, (y1 - 1) // BLOCK + 1):
                for k in range(z0 // BLOCK, (z1 - 1) // BLOCK + 1):
                    yield (i, j, k)

    def add(self, pid, box):
        self.boxes[pid] = box
        for b in self._blocks(box):
            self.blocks.setdefault(b, set()).add(pid)

    def remove(self, pid):
        box = self.boxes.pop(pid, None)
        if box is None:
            return
        for b in self._blocks(box):
            s = self.blocks.get(b)
            if s is not None:
                s.discard(pid)
                if not s:
                    del self.blocks[b]

    def query(self, box):
        nb = ((box[3] - 1) // BLOCK - box[0] // BLOCK + 1) * \
             ((box[4] - 1) // BLOCK - box[1] // BLOCK + 1) * \
             ((box[5] - 1) // BLOCK - box[2] // BLOCK + 1)
        if nb > len(self.boxes):
            return sorted(p for p, bx in self.boxes.items() if box_overlap(box, bx))
        out = set()
        for b in self._blocks(box):
            out |= self.blocks.get(b, set())
        return sorted(p for p in out if box_overlap(box, self.boxes[p]))

    def copy(self):
        n = _BlockIndex()
        n.boxes = dict(self.boxes)
        n.blocks = {k: set(v) for k, v in self.blocks.items()}
        return n


class ProcDrive:
    """A drive that is another process's observe() (3.4b): a wall's own
    temperature feeds a carbonation or tanning front inside it. Built on the
    fly from the scheduler, never stored."""
    smooth = True

    def __init__(self, sched, pid, key, slope=0.0):
        self.s, self.pid, self.key, self.slope = sched, pid, key, slope

    def value(self, t):
        return self.s.observe(self.pid, self.key, int(round(t)))

    def mean(self, a, b):
        return F.gl4(self.value, a, b) / max(b - a, 1)

    def kinks(self, a, b):
        return []

    def max_slope(self):
        return self.slope

    def level_ticks(self, a, b, level):
        return []

    def lo_hi(self):
        v = self.value(self.s.procs[self.pid].t)
        return v, v


# ──────────────────────────────────────────────────────── Scheduler ──
class Scheduler:
    """One per World, installed as `w.slow` (None = off)."""

    def __init__(self, clock=None):
        self.heap = []                      # (due, seq, pid, ver, mark_index)
        self.procs = {}
        self.drives = {}
        self.hosts = {}
        self.index = _BlockIndex()
        self.clock = clock if clock is not None else Clock()
        self.residual = {}
        self.sinks = {}
        self.history = deque(maxlen=64)
        self.notes = deque(maxlen=256)
        self.pins = set()
        self._seq = 0
        self._next_pid = 1
        self._intern = {}
        self._nshadow = 0
        self._nfolded = 0
        self._last_awake = 0

    # -- snapshots ---------------------------------------------------------
    def __deepcopy__(self, memo):
        import copy
        n = Scheduler.__new__(Scheduler)
        n.heap = list(self.heap)
        n.procs = {k: v.clone() for k, v in self.procs.items()}
        n.drives = copy.deepcopy(self.drives, memo)
        n.hosts = copy.deepcopy(self.hosts, memo)
        n.index = self.index.copy()
        n.clock = Clock(self.clock.epoch_s, self.clock.day_s, self.clock.year_s,
                        self.clock.dawn_s)
        n.residual = dict(self.residual)
        n.sinks = dict(self.sinks)
        n.history = deque(self.history, maxlen=64)
        n.notes = deque(self.notes, maxlen=256)
        n.pins = set(self.pins)
        n._seq, n._next_pid = self._seq, self._next_pid
        n._intern = dict(self._intern)
        n._nshadow, n._nfolded, n._last_awake = self._nshadow, self._nfolded, self._last_awake
        return n

    # -- registration -------------------------------------------------------
    def add_drive(self, name, drive):
        self.drives[name] = F.make_drive(drive)
        return name

    def add_host(self, name, host):
        self.hosts[name] = host
        return name

    def _intern_params(self, kind, params):
        items = tuple(sorted(params.items())) if isinstance(params, dict) else tuple(params)
        key = (kind, items)
        got = self._intern.get(key)
        if got is None:
            self._intern[key] = got = items
        return got

    def add(self, kind, cls, host, params, state=None, t=0, drives=(), marks=(),
            mode=None, pid=None, tag=""):
        """Register a process. `cls` 'owner' or 'shadow'. A SHADOW starts
        'folded' when a state is given and 'live' when not."""
        if kind not in F.FORMS:
            raise KeyError(f"no form named {kind!r}")
        p = Process()
        p.id = self._next_pid if pid is None else pid
        self._next_pid = max(self._next_pid, p.id) + 1
        p.kind, p.cls, p.host = kind, cls, tuple(host)
        p.params = self._intern_params(kind, params)
        p.state = tuple(float(v) for v in state) if state is not None else ()
        p.t = int(t)
        p.mode = "folded" if cls == "owner" else (mode or ("folded" if state else "live"))
        p.drives = tuple(drives)
        p.marks = tuple(marks)
        p.ver, p.since, p.next_try, p.tag = 0, int(t), 0, tag
        self.procs[p.id] = p
        if cls == "shadow":
            self._nshadow += 1
            if p.mode == "folded":
                self._nfolded += 1
            bx = self._host_box(p)
            if bx is not None:
                self.index.add(p.id, bx)
        self._requeue(p)
        return p.id

    def remove(self, pid):
        p = self.procs.pop(pid, None)
        if p is None:
            return
        p.ver += 1
        if p.cls == "shadow":
            self._nshadow -= 1
            if p.mode == "folded":
                self._nfolded -= 1
        self.index.remove(pid)
        self.pins.discard(pid)

    def pin(self, pid, w=None, now=None):
        self.pins.add(pid)
        p = self.procs.get(pid)
        if p is not None and p.cls == "shadow" and p.mode == "folded":
            self.unfold(pid, p.t if now is None else now, "pin", w)

    def unpin(self, pid):
        self.pins.discard(pid)

    # -- lookups --------------------------------------------------------------
    def _form(self, p):
        return F.FORMS[p.kind]

    def host_of(self, p):
        h = p.host
        if h and h[0] == "dict":
            return self.hosts[h[1]]
        if h and h[0] == "cells":
            return self.hosts.get(("cells",) + h[1:])
        return None

    def _host_box(self, p):
        h = self.host_of(p)
        return h.box() if h is not None else (tuple(p.host[1:7]) if p.host[:1] == ("cells",) else None)

    def box_of(self, p):
        return self.index.boxes.get(p.id)

    def _resolve(self, p):
        """{alias: Drive} for a process; ('proc', pid, key) targets are bound."""
        out = {}
        for alias, target in p.drives:
            if isinstance(target, str):
                out[alias] = self.drives[target]
            else:
                _, pid, key = target
                d = ProcDrive(self, pid, key)
                t0 = self.procs[pid].t
                d.slope = abs(d.value(t0 + 144000) - d.value(t0)) / 144000.0
                out[alias] = d
        return out

    def history_hosts(self):
        return [(p, self._form(p)) for p in self.procs.values()
                if self._form(p).needs_history or dict(p.params).get("needs_history")]

    # -- LOOK is free ------------------------------------------------------------
    def _state_at(self, p, now):
        """(state, valid_until_flag) at tick `now` WITHOUT writing anything."""
        if now <= p.t or p.mode == "live":
            return p.state, "ok"
        form = self._form(p)
        drv = self._resolve(p)
        ok, until, why = form.valid(p.state, p.params, drv, p.t, now)
        tag = "ok"
        if not ok:
            cap = p.t if until is None else max(int(until), p.t)
            tag = "invalid"
            if cap <= p.t:
                return p.state, tag
            now = min(now, cap)
        s, _ = form.advance(p.state, p.params, drv, p.t, now)
        return s, tag

    def observe(self, pid, key, now):
        """The value at tick `now`. Pure: nothing unfolds, nothing is written."""
        p = self.procs[pid]
        if p.mode == "live":
            h = self.host_of(p)
            return h.observe(key)
        s, _ = self._state_at(p, now)
        return self._form(p).observe(s, p.params, key)

    def observe_tag(self, pid, key, now):
        """(value, 'ok' | 'invalid' | 'live'): 'invalid' once valid() failed."""
        p = self.procs[pid]
        if p.mode == "live":
            return self.host_of(p).observe(key), "live"
        s, tag = self._state_at(p, now)
        if p.mode == "invalid":
            tag = "invalid"
        return self._form(p).observe(s, p.params, key), tag

    # -- the books ----------------------------------------------------------------
    def _book(self, w, fluxes, now=None):
        """Apply a flux dict to the sinks (F1). Energy out of the world goes to
        w.shed, mass to the air to w.gone[('vapor', species)]; everything is
        also kept in self.sinks so a test can audit the closure."""
        for key, v in fluxes.items():
            if v == 0.0:
                continue
            self.sinks[key] = self.sinks.get(key, 0.0) + v
            if w is None:
                continue
            what, sp, sink = key
            if what == "E" and sink in ("air", "air_top", "shed", "boundary"):
                w.shed += v
            elif what == "mass" and sink in ("air", "vapor"):
                w.gone[("vapor", sp)] = w.gone.get(("vapor", sp), 0.0) + v

    def _book_totals(self, key_before, key_after):
        for k, v in key_before.items():
            d = v - key_after.get(k, 0.0)
            self.residual[k] = self.residual.get(k, 0.0) + d

    def total_residual(self):
        return sum(abs(v) for v in self.residual.values())

    # -- advancing a process -----------------------------------------------------
    def _advance(self, p, now, w=None):
        """Move `p` to tick `now` by its closed form; apply fluxes. Respects
        valid(): an OWNER that fails is advanced to the last valid tick and
        marked 'invalid'; a SHADOW is advanced to the failure tick and handed
        back. Returns the tick reached."""
        if now <= p.t or p.mode in ("live", "invalid"):
            return p.t
        form = self._form(p)
        drv = self._resolve(p)
        ok, until, why = form.valid(p.state, p.params, drv, p.t, now)
        reach = now
        if not ok:
            reach = p.t if until is None else min(max(int(until), p.t), now)
        if reach > p.t:
            s, fl = form.advance(p.state, p.params, drv, p.t, reach)
            self._book(w, fl)
            p.state, p.t = s, reach
            p.ver += 1
        if not ok and reach < now:
            if p.cls == "owner":
                p.mode = "invalid"
                self.notes.append((reach, p.id, "invalid", why))
            else:
                self.notes.append((reach, p.id, "invalid", why))
        return reach

    def _requeue(self, p):
        """Solve each watchpoint's crossing and queue it. A mark that has no
        crossing before HORIZON_MAX is left out."""
        if p.mode in ("live", "invalid"):
            return
        form = self._form(p)
        drv = self._resolve(p)
        ok, until, _why = form.valid(p.state, p.params, drv, p.t, p.t + HORIZON_MAX)
        if not ok and until is not None and until > p.t:
            self._seq += 1                              # the assumptions fail at `until`:
            heapq.heappush(self.heap, (int(until), self._seq, p.id, p.ver, -1))
        for i, m in enumerate(p.marks):
            v0 = form.observe(p.state, p.params, m.key)
            if m.dir == "up" and v0 >= m.level or m.dir == "down" and v0 <= m.level:
                continue
            try:
                tc = form.cross(p.state, p.params, drv, p.t, m.key, m.level,
                                p.t + HORIZON_MAX)
            except F.OutOfRange:
                tc = None
            if tc is not None:
                self._seq += 1
                heapq.heappush(self.heap, (int(tc), self._seq, p.id, p.ver, i))

    def next_due(self):
        """The tick of the earliest live heap entry (lazy cancel of stale ones)."""
        while self.heap:
            due, _s, pid, ver, _i = self.heap[0]
            p = self.procs.get(pid)
            if p is None or p.ver != ver:
                heapq.heappop(self.heap)
                continue
            return due
        return None

    def drive_kinks(self, now):
        """[(tick, 'kink:<name>')]: the next table step of every drive."""
        out = []
        for name, d in self.drives.items():
            ks = d.kinks(now, now + HORIZON_MAX)
            if ks:
                out.append((ks[0], f"kink:{name}"))
        return out

    # -- the per-tick hook -------------------------------------------------------
    def pre(self, w):
        """Called at the start of the tick. (a) fire what is due; (b) if any
        SHADOW exists, unfold what the awake set now covers and fold what has
        been uncovered for FOLD_AFTER ticks."""
        now = w.tick
        if self.heap and self.heap[0][0] <= now:
            self._fire_due(w, now)
        if self._nshadow:
            if self._nfolded or now - self._last_awake >= 25:
                self._last_awake = now
                self._awake_pass(w, now)

    def _fire_due(self, w, now):
        while self.heap and self.heap[0][0] <= now:
            due, _s, pid, ver, i = heapq.heappop(self.heap)
            p = self.procs.get(pid)
            if p is None or p.ver != ver:
                continue
            if i < 0:                                   # valid() fails here (F11)
                if p.cls == "shadow" and p.mode == "folded":
                    self.unfold(pid, due, "invalid", w)
                else:
                    self._advance(p, due + 1, w)
                    p.mode = "invalid"
                    p.ver += 1
                continue
            mark = p.marks[i]
            self._advance(p, due, w)
            if p.mode == "invalid":
                continue
            form = self._form(p)
            val = form.observe(p.state, p.params, mark.key)
            self.notes.append((due, pid, "crosses", (mark.key, mark.level, mark.dir or "", mark.tag)))
            self._emit_cross(w, p, mark, due, val)
            p.marks = p.marks[:i] + p.marks[i + 1:]       # one-shot
            p.ver += 1
            if mark.action == "handback" and p.cls == "shadow" and p.mode == "folded":
                self.unfold(pid, due, "watch:" + mark.tag, w)
            else:
                self._requeue(p)

    def _emit_cross(self, w, p, mark, due, val):
        """Raise the typed 'crosses' event when the verb exists (it is appended
        to events.VERBS in phase B; until then the NOTE is on self.notes)."""
        try:
            from . import events
        except ImportError:
            return
        if "crosses" not in getattr(events, "VERBS", ()):
            return
        w.emit("crosses", target=("slow", p.id), visible=bool(mark.visible),
               data=(("quantity", mark.key), ("level", mark.level),
                     ("dir", mark.dir or ""), ("tag", mark.tag)),
               text="", site="slow:crosses")

    # -- the awake set (3.7) --------------------------------------------------------
    def awake_boxes(self, w):
        """[(why, box)] where a fine law not owned by a jump is working. Each is
        GROWN by the largest reach of one tick, because pre() sees the previous
        tick's windows. One box per window source (a single bounding box when
        there are no per-cluster windows: two far-apart hot spots unfold
        everything between them)."""
        out = []
        vox_m = 0.1 * w.scale
        reach = int(math.ceil(REACH_M / vox_m))
        for p in w.persons:                                          # W1
            if not p["alive"] or p["safe"] or not p["awake"] or p.get("asleep"):
                continue
            eye = p.get("_eye")
            if eye is None:
                ax, ay = p["anchor"]
                eye = (ax, ay, 0)
            ex, ey, ez = (int(round(v)) for v in eye)
            hgt = int(math.ceil(1.9 / vox_m))
            out.append(("person", (ex - reach, ey - reach, ez - hgt - reach,
                                   ex + reach + 1, ey + reach + 1, ez + reach + 1)))
        win = w._hot_window(pad=0) if hasattr(w, "_hot_window") else None   # W2
        if win is not None:
            out.append(("hot", _box_of(win, MARGIN)))
        if w.smoke is not None:
            sm = w._window(w.smoke > 1e-6, getattr(w, "_SMOKE_REACH", 4))
            if sm is not None:
                out.append(("smoke", _box_of(sm, 0)))
        liq = w._window(w.fvol > 0, MARGIN)
        if liq is not None:
            out.append(("liquid", _box_of(liq, 0)))
        mv = w._window(w.moving, MARGIN)                             # W3
        if mv is not None:
            out.append(("moving", _box_of(mv, 0)))
        for b in w.bodies:
            c = np.asarray(b["cells"]) + np.asarray(b.get("off", 0.0))
            if len(c):
                lo, hi = c.min(axis=0), c.max(axis=0)
                out.append(("body", (int(lo[0]) - 3, int(lo[1]) - 3, int(lo[2]) - 3,
                                     int(hi[0]) + 4, int(hi[1]) + 4, int(hi[2]) + 4)))
        for pid in self.pins:                                        # W4
            bx = self.index.boxes.get(pid)
            if bx is not None:
                out.append(("pin", bx))
        return out

    def covered(self, boxes):
        """Process ids whose host box overlaps any awake box."""
        hit = set()
        for _why, bx in boxes:
            hit.update(self.index.query(bx))
        return hit

    def _awake_pass(self, w, now):
        hit = self.covered(self.awake_boxes(w))
        for pid in sorted(hit):
            p = self.procs.get(pid)
            if p is None or p.cls != "shadow":
                continue
            if p.mode == "folded":
                self.unfold(pid, now, "awake", w)
            p.since = now
        for pid in sorted(self.procs):
            p = self.procs[pid]
            if (p.cls == "shadow" and p.mode == "live" and pid not in hit
                    and pid not in self.pins and now - p.since >= FOLD_AFTER
                    and now >= p.next_try):
                if not self.fold(pid, now, w):
                    p.next_try = now + RETRY

    def touch_box(self, box, now, w=None, why="touch"):
        """Unfold every folded SHADOW whose host overlaps `box`."""
        done = []
        for pid in self.index.query(box):
            p = self.procs[pid]
            if p.cls == "shadow" and p.mode == "folded":
                self.unfold(pid, now, why, w)
                done.append(pid)
            p.since = now
        return done

    # -- fold / unfold (5) ---------------------------------------------------------
    def fold(self, pid, now, w=None):
        """live -> folded. Refused (False) when the fit is out of tolerance."""
        p = self.procs[pid]
        if p.cls != "shadow" or p.mode != "live":
            return False
        form, host = self._form(p), self.host_of(p)
        if form.project is None:
            return False
        rd = host.read()
        try:
            state, info = form.project(rd["x"], rd["c"], p.params,
                                       mass_ref=list(host.totals().values())[0])
        except F.Refused as e:
            self.notes.append((now, pid, "fold refused", e.reason))
            return False
        before = host.totals()
        after = {k: form.totals(state, p.params, rd["x"]) for k in before}
        self._book_totals(before, after)
        p.state, p.t, p.mode = state, int(now), "folded"
        p.ver += 1
        p.since = now
        self._nfolded += 1
        self._requeue(p)
        return True

    def unfold(self, pid, now, why, w=None):
        """folded -> live: advance to now (or to the failure tick), apply the
        fluxes, write the profile into the host, and hand it to the tick laws."""
        p = self.procs[pid]
        if p.cls != "shadow" or p.mode != "folded":
            return False
        reach = self._advance(p, now, w)
        form, host = self._form(p), self.host_of(p)
        before = {k: form.totals(p.state, p.params, host.x) for k in host.totals()}
        host.write(form.inject(p.state, host.x, p.params))
        self._book_totals(before, host.totals())
        p.mode = "live"
        p.ver += 1
        p.since = now
        p.state = ()
        self._nfolded -= 1
        self.notes.append((now, pid, "unfold", why))
        return True

    # -- OWNER touch (F8) ------------------------------------------------------------
    def touch(self, pid, patch, now, w=None):
        """Advance an OWNER to `now` with its old inputs, then switch an input:
        patch = {'drives': {alias: target}, 'params': {k: v}}. Nothing is
        handed back. A folded SHADOW is unfolded instead."""
        p = self.procs[pid]
        if p.cls == "shadow":
            return self.unfold(pid, now, "touch", w)
        self._advance(p, now, w)
        if "drives" in patch:
            d = dict(p.drives)
            d.update(patch["drives"])
            p.drives = tuple(sorted(d.items(), key=lambda kv: kv[0]))
        if "params" in patch:
            q = dict(p.params)
            q.update(patch["params"])
            p.params = self._intern_params(p.kind, q)
        p.ver += 1
        self._requeue(p)
        return True

    # -- the skip-ahead driver -----------------------------------------------------------
    def resolve_target(self, w, ticks=None, seconds=None, until=None):
        if ticks is not None:
            return w.tick + int(ticks), None
        if seconds is not None:
            return w.tick + int(round(seconds / TICK_S)), None
        if isinstance(until, tuple) and until[0] == "woke":
            return w.tick + int(self.clock.year_s / TICK_S), until[1]
        if isinstance(until, str):
            return w.tick + self.clock.ticks_until(w.tick, self.clock.parse(until)), None
        if until is not None:
            return int(until), None
        raise ValueError("advance needs ticks, seconds or until")

    def advance(self, w, ticks=None, *, seconds=None, until=None, who=None,
                approx=False, max_fine_ticks=None):
        return advance(w, ticks, seconds=seconds, until=until, who=who, approx=approx,
                       max_fine_ticks=max_fine_ticks, sched=self)


# ─────────────────────────────────────────────────── the driver API ──
def _step(w, n, rep, stop_ev=None):
    for _ in range(n):
        w.step()
        rep.stepped += 1
        if stop_ev is not None and stop_ev(w):
            return True
    return False


def _woke(w, pid, first_id):
    for ev in w.events.since(first_id - 1):
        if ev.verb == "wakes" and (pid is None or ev.actor == pid):
            return True
    return False


def advance(w, ticks=None, *, seconds=None, until=None, who=None, approx=False,
            max_fine_ticks=None, sched=None):
    """Advance the world, jumping where the world is provably quiet, stepping
    where it is not. DRIVER LEVEL ONLY. approx=False (the default) means tier 0
    only, so golden and harvest can never get tier 1 by accident; the playtest
    passes approx=True.

    With w.slow None or w.skip_quiet False this is exactly a loop of step()
    (F5) and the report says why it did not jump; it never silently ticks an
    hour: max_fine_ticks caps the ticked fallback."""
    sched = sched if sched is not None else getattr(w, "slow", None)
    rep = SkipReport(w.tick)
    clock = sched.clock if sched is not None else Clock()
    helper = sched if sched is not None else Scheduler()
    target, wake_pid = helper.resolve_target(w, ticks, seconds, until)
    stop_on_wake = isinstance(until, tuple) and until[0] == "woke"
    first_ev = w.events.next_id
    tier = 1 if approx else 0
    seen = []
    jumps = []
    next_probe = w.tick
    cap = max_fine_ticks

    def done_by_wake():
        return stop_on_wake and _woke(w, wake_pid, first_ev)

    def finish(why):
        rep.t_to = w.tick
        rep.halted_by = why
        rep.blockers_seen = tuple(seen)
        rep.tier_used = "reduced" if tier == 1 and jumps and any(j[2] for j in jumps) else "exact"
        rep.jumps = tuple(jumps)
        rep.residual = sched.total_residual() if sched is not None else 0.0
        if sched is not None:
            sched.history.append(rep)
        return rep

    if sched is None or not w.skip_quiet:
        n = target - w.tick
        if cap is not None:
            n = min(n, cap)
        why = "skipping is off" if sched is not None else "no scheduler"
        seen.append(("skip", "world", why))
        if _step(w, n, rep, lambda ww: done_by_wake()):
            return finish("woke")
        return finish("target" if w.tick >= target else "blocked")

    while w.tick < target:
        if cap is not None and rep.stepped >= cap:
            return finish("blocked")
        left = target - w.tick
        stepped_now = False
        if w.tick >= next_probe and left > 2 * Q.PERIOD + 4:
            q = Q.quiet(w, tier, probe_=False, sched=sched)
            if q.ok:
                until_ = q.until if q.until is not None else target
                hop = min(until_ - 2, target)
                if hop - w.tick >= 2 * Q.PERIOD:
                    before = w.tick
                    q = Q.quiet(w, tier, probe_=True, sched=sched)
                    rep.stepped += w.tick - before
                    if q.ok:
                        until_ = q.until if q.until is not None else target
                        hop = min(until_ - 2, target)
                        n = hop - w.tick
                        if n >= 1:
                            ok, info = _jump(w, n, q, tier, sched)
                            if ok:
                                n = info.get("n", n)             # a coupled jump may end at a chunk edge
                                rep.jumped += n
                                jumps.append((before, n, tier >= 1 and bool(q.owned)))
                                if info.get("why"):
                                    seen.append(("thermal", None, info["why"]))
                            else:
                                seen.append(("thermal", None, str(info)))
                                next_probe = w.tick + PROBE_BACKOFF
                        if done_by_wake():
                            return finish("woke")
                        if w.tick < target and w.tick >= hop and until_ - 2 <= hop:
                            # now 2 ticks before the event: let the tick laws play it
                            if _step(w, min(target - w.tick, 4), rep,
                                     lambda ww: done_by_wake()):
                                return finish("woke")
                        stepped_now = True
                    else:
                        for b in q.blockers[:8]:
                            _remember(seen, b)
                        next_probe = w.tick + PROBE_BACKOFF
                        stepped_now = True
                else:
                    # the event is near: step to it and past it
                    k = max(1, min(left, (until_ + 3) - w.tick))
                    if _step(w, k, rep, lambda ww: done_by_wake()):
                        return finish("woke")
                    stepped_now = True
            else:
                for b in q.blockers[:8]:
                    _remember(seen, b)
        if not stepped_now:
            k = min(STEP_CHUNK, target - w.tick)
            if _step(w, k, rep, lambda ww: done_by_wake()):
                return finish("woke")
    return finish("target")


def _remember(seen, b):
    row = (b.gate, b.place, b.reason)
    if row not in seen:
        seen.append(row)


def _jump(w, n, q, tier, sched):
    """What a jump does, all of it (7.3): the net (tier 1), the twins, the
    EXCLUDE resets, the clock. Does nothing to owners and folded shadows (lazy).
    Returns (ok, info)."""
    from . import slownet
    now = w.tick
    plan = q.plan
    if tier >= 1 and any(getattr(tw, "coupled", False) and ctx.get("active")
                         for _p, twins in plan for _nm, tw, ctx in twins):
        # bodies carried by a twin (T7): the room and the bodies take turns, chunk by chunk (slowbody.py).
        # It moves w.tick itself and may end early at a chunk edge.
        from . import slowbody
        ok, info = slowbody.coupled_jump(w, n, q, sched)
        if not ok:
            return False, info["refusals"][0][2] if info.get("refusals") else "refused"
        n_done = info["n"]
        if n_done < 1:
            return False, info.get("why") or "the coupled jump could not start"
        slowbody.sync_clocks(w)
        _apply_resets(w, n_done, q.rates)
        if hasattr(w, "_rad_last") and bool(w.E.any()):
            from .heat import RAD_EVERY
            last = ((w.tick - 1) // RAD_EVERY) * RAD_EVERY
            if last >= now:
                w._rad_last = last
        return True, info
    if tier >= 1:
        sources, o2 = slownet.collect_sources(w, plan, n)
        if bool(w.E.any()) or sources:
            ok, info = slownet.advance_region(w, n, sched, sources=sources, o2_used_g=o2)
            if not ok:
                return False, info["refusals"][0][2] if info.get("refusals") else "refused"
    for p, twins in plan:
        for _name, tw, ctx in twins:
            sched._book(w, tw.advance(w, p, ctx, n))
    _apply_resets(w, n, q.rates)
    if tier >= 1 and hasattr(w, "_rad_last") and bool(w.E.any()):
        # base19 (B0): heat._law_radiate stamps the tick of its last pass and spends k = tick - stamp ticks of
        # flame at the next one. Ticking would have run a pass at every multiple of RAD_EVERY while E != 0, so
        # the stamp after the jump is the last such tick; left behind, the next pass would see k = n.
        from .heat import RAD_EVERY
        last = ((now + n - 1) // RAD_EVERY) * RAD_EVERY
        if last >= now:
            w._rad_last = last
    w.tick = now + n
    return True, {}


def _apply_resets(w, n, rates):
    """The EXCLUDE rows a jump must act on: clocks grow at the rate measured
    over the probe; countdowns count down to 0."""
    m = getattr(w.matter, "mpm", None)
    if m is not None and "matter.mpm.t" in rates:
        m.t += rates["matter.mpm.t"] * n
    b = getattr(w.matter, "bonds", None)
    if b is not None and "matter.bonds.t" in rates:
        b.t += rates["matter.bonds.t"] * n
    ts = getattr(w, "_torque_sleep", 0)
    if ts:
        w._torque_sleep = max(0, ts - n)


def wait(w, seconds, **kw):
    """Wait `seconds` of game time (driver level)."""
    return advance(w, seconds=seconds, **kw)


def until_clock(w, text="dawn", **kw):
    """Advance to the next time the clock reads `text` ('06:00' or 'dawn')."""
    return advance(w, until=text, **kw)


def sleep(w, who, until=None, **kw):
    """Lay someone down and sleep them: needs a sleep law (wave 15 body clocks)."""
    raise NotImplementedError("sleep needs a sleep law (wave 15); use advance() for a "
                              "scene-set sleeper")
