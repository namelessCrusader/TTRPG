"""RECORDS (T2, spec sections 2-4, 6.3, 10): the pools for things thinner than a cell.

A RECORD describes how matter and energy are SPREAD inside something thinner than a cell
(or how many small things there are). It rides on a HOST. This module is the books:
struct-of-arrays pools, one per class, a header on every row, the grouped census of what
the hosts hold, the audit, the exact row split, and the one ledger term `records_owned`.
It imports no World; every function takes plain arrays and the `Records` object.

TWO CLASSES OF MATTER (2.1)
  (a) SHAPE-ONLY (SECTION, SHEET, STRAND, CONDUIT): the host (thing cells, MPM particles,
      bonds, bodies in flight) owns every gram and joule. The record holds fractions and
      profiles and is RECONCILED TO the host once a tick. The ledgers never see it.
  (b) RECORD-OWNED (JOIN glue line, LOT, cup-scale VESSEL contents): the record owns the
      grams, joules and millilitres, and `records_owned` is the ONE ledger term that adds
      them (and the float64 remainders `rem` of every hand-over) to total_mass,
      total_fluid and total_energy. It returns exact zeros when no record and no
      remainder exists, so a world with no records steps bit for bit as it does today.

POOLS (3.1). One numpy array per field per class. `rid` is monotonic and never reused;
a freed ROW is reused (LIFO free list, deterministic). Everything in `Records` is plain
arrays, lists and dicts, so `copy.deepcopy` (the world snapshot) is exact. Caches (bind
matrices, footprints, materials) live OUTSIDE the state in a WeakKeyDictionary like
heat._GEO and are rebuilt lazily after a restore.

UNITS. Grams per species in `Y`/`g` (dense, the stand-in table of record_data.py; T1's
pool row ids replace them at wiring), joules above the 20 C reference, enthalpy `H` in
J/kg, millilitres per fluid slot. Energy of a node is (grams/1000) x H."""
import weakref

import numpy as np

from . import thinheat as th
from .record_data import (NF, NS, RHO, SP, SP_NAMES, lattice_material, mix_material, node_volumes,
                          row_material)

# record classes (class ids are append-only: they are stored)
SECTION, SHEET, STRAND, CONDUIT, LOT, VESSEL, JOIN, GROUP, MECHANISM = range(9)
CLASS_NAMES = ("SECTION", "SHEET", "STRAND", "CONDUIT", "LOT", "VESSEL", "JOIN", "GROUP",
               "MECHANISM")
SHAPE_ONLY = (SECTION, SHEET, STRAND, CONDUIT)          # class (a)
OWNING = (LOT, VESSEL, JOIN)                             # class (b)
THIN = (SECTION, SHEET)                                  # heat through the thickness

# host kinds
HOST_THING, HOST_REGION, HOST_PLOT, HOST_CELL, HOST_NONE = range(5)

NN = 13                                                  # nodes held per thin row (7 half, 13 full)
# the causes a typed change can carry; append-only (stored as an index)
CAUSES = ("make", "reconcile", "absorb", "consume", "split", "heat", "diffuse", "handover",
          "external", "fracture", "fuse", "spill")

_CACHE = weakref.WeakKeyDictionary()                     # outside the state, like heat._GEO


def cache_of(recs):
    """Per-Records scratch (bind matrices, footprints, materials). Never snapshotted."""
    c = _CACHE.get(recs)
    if c is None:
        c = _CACHE[recs] = {}
    return c


# ── schemas ──────────────────────────────────────────────────────────────────
# name -> (dtype, tail shape, kind, default). kind: "ext" extensive (split f, 1-f),
# "int" intensive (copied), "count" integer count (split by largest remainder).
def _f(tail=(), kind="int", default=0.0, dtype=np.float64):
    return (dtype, tail, kind, default)


_THIN = {
    "seg_len": _f(kind="ext"), "width": _f(), "area": _f(kind="ext"), "thick": _f(),
    "shape": _f(dtype=np.int8, default=0), "nn": _f(dtype=np.int8, default=7),
    "sym": _f(dtype=np.bool_, default=True),
    "H": _f((NN,)), "Y": _f((NN, NS), kind="ext"),
    "dmg": _f(), "loss": _f(), "edge_len": _f(kind="ext"), "stress": _f(),
    "port_h": _f((2,)), "port_T": _f((2,), default=20.0), "port_a": _f((2,), default=1.0),
    "port_rh": _f((2,), default=-1.0), "port_hm": _f((2,), default=1.0),
    "quiet": _f(dtype=np.bool_, default=False), "port_changed": _f(dtype=np.bool_, default=False),
    "host_scale": _f(),
    "crack_w": _f(), "crack_l": _f(),
}
_OWN = {"g": _f((NS,), kind="ext"), "E": _f(kind="ext"), "owns": _f(dtype=np.bool_, default=True)}
SCHEMA = {
    SECTION: _THIN,
    SHEET: _THIN,
    STRAND: {"diam": _f(), "length": _f(kind="ext"), "twist": _f(), "break_load": _f(),
             "wrap_idx": _f((8,), dtype=np.int32, default=-1),
             "wrap_kind": _f((8,), dtype=np.int16, default=-1),
             "g": _f((NS,), kind="ext"), "Hm": _f(), "quiet": _f(dtype=np.bool_, default=False)},
    CONDUIT: {"bore": _f(), "length": _f(kind="ext"), "z_a": _f(), "z_b": _f(),
              "primed": _f(dtype=np.bool_, default=False),
              "g": _f((NS,), kind="ext"), "Hm": _f(),
              "port_h": _f((2,)), "port_T": _f((2,), default=20.0)},
    LOT: {**_OWN, "n_items": _f(kind="count", dtype=np.int64, default=0),
          "hist": _f((16, 16), kind="count", dtype=np.int64, default=0),
          "T": _f(), "moisture": _f(), "ml": _f((NF,), kind="ext")},
    VESSEL: {**_OWN, "ml": _f((NF,), kind="ext"), "scale": _f(dtype=np.int8, default=0),
             "wall_rid": _f(dtype=np.int64, default=-1), "lip": _f(), "fill": _f(),
             "gas_g": _f(kind="ext"), "vent_area": _f()},
    JOIN: {**_OWN, "rid_a": _f(dtype=np.int64, default=-1), "rid_b": _f(dtype=np.int64, default=-1),
           "eid_a": _f(dtype=np.int64, default=-1), "eid_b": _f(dtype=np.int64, default=-1),
           "kind": _f(dtype=np.int16, default=0), "area": _f(kind="ext"), "alpha": _f(),
           "maturity": _f(), "moisture": _f(), "Ft": _f(), "Fs": _f(), "Mc": _f(),
           "fail_tick": _f(dtype=np.int32, default=-1), "fail_mode": _f(dtype=np.int16, default=-1)},
    GROUP: {"n_members": _f(dtype=np.int32, default=0), "payload": _f((4,), dtype=np.int64, default=0)},
    MECHANISM: {"n_parts": _f(dtype=np.int32, default=0), "stored_J": _f(),
                "payload": _f((4,), dtype=np.int64, default=0)},
}
_HEADER = {"rid": np.int64, "cls": np.int8, "host_kind": np.int8, "host_id": np.int64,
           "born": np.int32, "alive": np.bool_, "seed": np.uint32, "cause_tick": np.int32,
           "cause_verb": np.int16}


class Pool:
    """One class: header arrays and field arrays, `cap` rows, `used` high-water mark, a LIFO
    free list of dead rows. All state is arrays, lists and ints."""

    def __init__(self, cls, schema, cap=16):
        self.cls = cls
        self.schema = dict(schema)
        self.cap = 0
        self.used = 0
        self.free = []
        self.a = {}
        self._alloc(cap)

    def _alloc(self, cap):
        for name, dt in _HEADER.items():
            old = self.a.get(name)
            new = np.zeros(cap, dtype=dt)
            if old is not None:
                new[:self.cap] = old
            self.a[name] = new
        for name, (dt, tail, _, default) in self.schema.items():
            old = self.a.get(name)
            new = np.full((cap,) + tail, default, dtype=dt)
            if old is not None:
                new[:self.cap] = old
            self.a[name] = new
        self.cap = cap

    def take_row(self):
        if self.free:
            return self.free.pop()
        if self.used == self.cap:
            self._alloc(self.cap * 2)
        self.used += 1
        return self.used - 1

    def reset_row(self, row):
        for name, (dt, tail, _, default) in self.schema.items():
            self.a[name][row] = default
        for name in _HEADER:
            self.a[name][row] = 0

    def release(self, row):
        self.a["alive"][row] = False
        self.free.append(row)

    def alive_rows(self):
        return np.flatnonzero(self.a["alive"][:self.used])

    def count(self):
        return int(self.a["alive"][:self.used].sum())


class Records:
    """`w.records`. `n` is the live count: every hook starts `if not w.records.n: return`."""

    lattice_cp = False              # class default, so a pool saved before C5 still has it

    def __init__(self, extra=None, cap=16, seed=0):
        schema = {c: dict(s) for c, s in SCHEMA.items()}
        for c, fields in (extra or {}).items():
            schema[c].update(fields)
        self.pools = {c: Pool(c, schema[c], cap) for c in SCHEMA}
        self.n = 0
        self.next_rid = 1
        self.seed = int(seed) & 0xFFFFFFFF
        self.loc = {}               # rid -> (cls, row)
        self.rem = {}               # (host_id, field) -> float64 remainder, counted by records_owned
        self.meter = {}             # (host_id, cause) -> [|dE|, |dm|, dE, dm] absorbed this tick
        self.touched = {}           # host_id -> tick an external law changed it
        self.tick = 0
        self.causes = list(CAUSES)
        self.audit_log = []
        self.lattice_cp = False     # True: thin rows use the lattice's constant c (C5)

    # ── ticks and causes ─────────────────────────────────────────────────────
    def begin_tick(self, tick=None):
        self.tick = self.tick + 1 if tick is None else int(tick)
        self.meter = {}
        self.touched = {}

    def touch(self, host_id):
        """An external law (burn, melt, acid, a blow) changed this host this tick: a
        reconcile absorb on it is expected, not a fault."""
        self.touched[int(host_id)] = self.tick

    def cause_id(self, name):
        if name not in self.causes:
            self.causes.append(name)
        return self.causes.index(name)

    def note(self, rid, cause):
        cls, row = self.loc[rid]
        p = self.pools[cls]
        p.a["cause_tick"][row] = self.tick
        p.a["cause_verb"][row] = self.cause_id(cause)

    # ── create, kill, find ───────────────────────────────────────────────────
    def add(self, cls, host_kind=HOST_THING, host_id=0, **fields):
        p = self.pools[cls]
        row = p.take_row()
        p.reset_row(row)
        rid = self.next_rid
        self.next_rid += 1
        a = p.a
        a["rid"][row] = rid
        a["cls"][row] = cls
        a["host_kind"][row] = host_kind
        a["host_id"][row] = host_id
        a["born"][row] = self.tick
        a["alive"][row] = True
        a["seed"][row] = (rid * 2654435761 + self.seed) & 0xFFFFFFFF
        for k, v in fields.items():
            a[k][row] = v
        self.loc[rid] = (cls, row)
        self.n += 1
        self.note(rid, "make")
        return rid

    def kill(self, rid):
        cls, row = self.loc.pop(rid)
        self.pools[cls].release(row)
        self.n -= 1

    def row(self, rid):
        cls, row = self.loc[rid]
        return self.pools[cls], row

    def get(self, rid, name):
        p, r = self.row(rid)
        return p.a[name][r]

    def set(self, rid, name, value):
        p, r = self.row(rid)
        p.a[name][r] = value

    def rids(self, cls=None):
        """Live rids in a fixed order: class, then row."""
        out = []
        for c in ([cls] if cls is not None else sorted(self.pools)):
            p = self.pools[c]
            out += [int(x) for x in p.a["rid"][p.alive_rows()]]
        return out

    def hosts(self, classes=SHAPE_ONLY):
        """Sorted host ids that carry a live record of those classes."""
        ids = [self.pools[c].a["host_id"][self.pools[c].alive_rows()] for c in classes]
        return np.unique(np.concatenate(ids)) if ids else np.zeros(0, np.int64)

    # ── remainders (the float64 part of every float32 crossing) ──────────────
    def rem_add(self, host_id, field, value):
        k = (int(host_id), field)
        self.rem[k] = self.rem.get(k, 0.0) + float(value)

    def rem_take(self, host_id, field):
        return self.rem.pop((int(host_id), field), 0.0)

    # ── totals ───────────────────────────────────────────────────────────────
    def totals(self, cls):
        """(rows, g (n,NS) grams per species, E (n,) J) of the live rows of a class."""
        p = self.pools[cls]
        r = p.alive_rows()
        a = p.a
        if cls in THIN:
            Y = a["Y"][r]
            return r, Y.sum(axis=1), ((Y.sum(axis=2) / 1000.0) * a["H"][r]).sum(axis=1)
        if cls in (STRAND, CONDUIT):
            return r, a["g"][r], a["g"][r].sum(axis=1) / 1000.0 * a["Hm"][r]
        if cls in OWNING:
            return r, a["g"][r], a["E"][r]
        z = np.zeros((r.size, NS))
        return r, z, np.zeros(r.size)

    def row_totals(self, rid):
        """(g (NS,), E J) of one record."""
        cls, row = self.loc[rid]
        a = self.pools[cls].a
        if cls in THIN:
            Y = a["Y"][row]
            return Y.sum(axis=0), float((Y.sum(axis=1) / 1000.0 * a["H"][row]).sum())
        if cls in (STRAND, CONDUIT):
            return a["g"][row].copy(), float(a["g"][row].sum() / 1000.0 * a["Hm"][row])
        if cls in OWNING:
            return a["g"][row].copy(), float(a["E"][row])
        return np.zeros(NS), 0.0

    def mass_g(self, rid):
        return float(self.row_totals(rid)[0].sum())

    def energy_J(self, rid):
        return self.row_totals(rid)[1]

    # ── exact row split (F5, spec 6.1) ───────────────────────────────────────
    def split(self, rid, f, host_kind=None, host_id=None):
        """Cut one row at fraction f of its extensive content. The first share is f x whole,
        the SECOND is WHOLE MINUS FIRST for every extensive field (mass per node, energy,
        grams, length), so the sums close to one float64 ulp (1e-15 relative); counts split
        by largest remainder and stay integers; intensive fields (H, T, ports) are copied.
        Returns the new rid. The caller assigns the child to its host."""
        if not 0.0 < f < 1.0:
            raise ValueError("split fraction must be inside (0, 1)")
        cls, row = self.loc[rid]
        p = self.pools[cls]
        child = self.add(cls, int(p.a["host_kind"][row]) if host_kind is None else host_kind,
                         int(p.a["host_id"][row]) if host_id is None else host_id)
        crow = self.loc[child][1]
        for name, (dt, tail, kind, _) in p.schema.items():
            whole = p.a[name][row].copy()
            if kind == "ext":
                first = whole * f
                p.a[name][row] = first
                p.a[name][crow] = whole - first
            elif kind == "count":
                first = np.floor(whole * f + 0.5).astype(dt)
                p.a[name][row] = first
                p.a[name][crow] = whole - first
            else:
                p.a[name][crow] = whole
        if cls in THIN:
            refresh_geometry(self, rid)
            refresh_geometry(self, child)
        if "quiet" in p.schema:
            p.a["quiet"][row] = False
            p.a["quiet"][crow] = False
        self.note(rid, "split")
        self.note(child, "split")
        return child


# ── geometry kept true to the grams ──────────────────────────────────────────
def refresh_geometry(recs, rid):
    """Recompute `thick` from the node volumes (shrinkage on drying is geometry): thick =
    sum of node volumes / area. Also stamps `width` and `seg_len` consistency is the
    caller's. A row with no mass keeps thick 0."""
    p, r = recs.row(rid)
    if recs.loc[rid][0] not in THIN:
        return
    vol = float(node_volumes(p.a["Y"][r]).sum())
    area = float(p.a["area"][r])
    p.a["thick"][r] = vol / area if area > 0.0 else 0.0


REGRID_LOW = 0.25      # a node holding less than this share of its nominal volume triggers a regrid
REGRID_HIGH = 4.0


def regrid(recs, rid):
    """Conservative remap of the node profile onto the uniform grid the solver assumes.

    After a surface node has burnt away (or the thickness shrank unevenly) the stored
    nodes no longer sit at equal spacing. Each old node is a layer of width v_i / (A mult);
    the new nodes are the nominal volume shares of the total. Grams per species and
    JOULES are moved by overlap, so both sums are exact (to round-off); only the shape of
    the profile is smoothed, and only when a node has emptied or is far off its share."""
    p, r = recs.row(rid)
    n = int(p.a["nn"][r])
    sym = bool(p.a["sym"][r])
    Y = p.a["Y"][r, :n].copy()
    m = Y.sum(axis=1) / 1000.0
    E = m * p.a["H"][r, :n]
    v = node_volumes(Y)
    tot = float(v.sum())
    if tot <= 0.0 or float(m.sum()) <= 0.0:
        return False
    old = np.concatenate([[0.0], np.cumsum(v)]) / tot
    new = np.concatenate([[0.0], np.cumsum(th.node_fractions(n, sym))])
    # the stored order runs from the centre (sym) or from face A (full); the face is at the end
    ov = np.maximum(np.minimum(new[1:, None], old[None, 1:]) - np.maximum(new[:-1, None], old[None, :-1]), 0.0)
    W = ov / np.maximum(np.diff(old), 1e-300)[None, :]          # (new, old) share of each old layer
    Yn = W @ Y
    En = W @ E
    mn = Yn.sum(axis=1) / 1000.0
    p.a["Y"][r, :n] = Yn
    p.a["H"][r, :n] = np.where(mn > 0.0, En / np.maximum(mn, 1e-300), p.a["H"][r, :n])
    p.a["quiet"][r] = False
    refresh_geometry(recs, rid)
    return True


def maybe_regrid(recs, rid):
    """Regrid when any node is off its nominal share by more than REGRID_LOW / HIGH."""
    p, r = recs.row(rid)
    n = int(p.a["nn"][r])
    v = node_volumes(p.a["Y"][r, :n])
    tot = float(v.sum())
    if tot <= 0.0:
        return False
    ratio = (v / tot) / th.node_fractions(n, bool(p.a["sym"][r]))
    if ratio.min() < REGRID_LOW or ratio.max() > REGRID_HIGH:
        return regrid(recs, rid)
    return False


def _row_mat(recs, g):
    """Material of a row's grams: the lattice's constant c when the world asked for it."""
    g = np.asarray(g, dtype=np.float64)
    return lattice_material(g) if recs.lattice_cp else row_material(g)


def make_thin(recs, cls, comp, thick, width, length, T=20.0, nn=7, sym=True,
              host_kind=HOST_THING, host_id=0, ports=None, shape=0):
    """A SECTION or SHEET segment of `thick` x `width` x `length` m made of composition
    `comp` (species name -> mass fraction) at a uniform temperature T (C). The node
    grams follow the node volumes (end nodes half), H comes from the row's table.
    `ports` = dict(h=(h0,h1), T=(T0,T1), a=(a0,a1))."""
    w = np.zeros(NS)
    for k, v in comp.items():
        w[SP[k]] = v
    w = w / w.sum()
    area = width * length
    vol_total = area * thick
    dens = 1.0 / float(node_volumes(w * 1000.0))        # kg/m3 of the mix: 1 kg occupies this volume^-1
    mass_g = vol_total * dens * 1000.0
    frac = th.node_fractions(nn, sym)
    rid = recs.add(cls, host_kind, host_id)
    p, r = recs.row(rid)
    a = p.a
    a["seg_len"][r], a["width"][r], a["area"][r], a["thick"][r] = length, width, area, thick
    a["nn"][r], a["sym"][r], a["shape"][r] = nn, sym, shape
    a["Y"][r, :nn, :] = frac[:, None] * w[None, :] * mass_g
    mat = _row_mat(recs, w)
    a["H"][r, :nn] = float(mat.H_of_T(T))
    for k, key in (("h", "port_h"), ("T", "port_T"), ("a", "port_a")):
        if ports and k in ports:
            a[key][r] = ports[k]
    a["edge_len"][r] = length
    refresh_geometry(recs, rid)
    return rid


# ── the grouped census (3.10, resolution E10) ────────────────────────────────
class Census:
    """What every host holds, from ONE grouped pass: g (n_eid, NS) grams per species and
    E (n_eid,) joules (plus count (n_eid,) entries when asked)."""

    def __init__(self, n_eid, count=False, scratch=None):
        self.n_eid = n_eid
        self.g = np.zeros((n_eid, NS))
        self.E = np.zeros(n_eid)
        self.count = np.zeros(n_eid, dtype=np.int64) if count else None
        # scratch buffers reused tick to tick (keep the dict in cache_of(recs)): a fresh
        # 1.6 MB temporary per call costs a millisecond of page faults, more than the sum
        self.scratch = {} if scratch is None else scratch

    def _buf(self, name, n, dtype):
        b = self.scratch.get(name)
        if b is None or b.size < n or b.dtype != dtype:
            b = self.scratch[name] = np.empty(max(n, 1024), dtype=dtype)
        return b[:n]

    def add(self, eid, sp, mass_g, E_J=None):
        """eid, sp (species index; -1 skips), mass_g, E_J: parallel arrays of one source
        (lattice entries of cells with an eid, MPM particles, bonds, bodies in flight).
        bincount is the grouping; the arrays are given, never walked per record. The
        arrays are trusted to be in range (checked by min and max, 0.03 ms per 200k); a
        stray entry takes the masked slow path."""
        eid = np.asarray(eid)
        if eid.size == 0:
            return self
        sp = np.asarray(sp)
        if eid.min() < 0 or eid.max() >= self.n_eid or sp.min() < 0:
            ok = (eid >= 0) & (eid < self.n_eid) & (sp >= 0)
            eid, sp, mass_g = eid[ok], sp[ok], np.asarray(mass_g)[ok]
            E_J = None if E_J is None else np.asarray(E_J)[ok]
        n = self.n_eid
        k = eid.size
        e = self._buf("e", k, np.intp)
        key = self._buf("key", k, np.intp)
        w = self._buf("w", k, np.float64)
        np.copyto(e, eid, casting="unsafe")                    # one cast, shared by the bincounts
        np.multiply(e, NS, out=key)
        key += sp
        np.copyto(w, mass_g, casting="unsafe")
        self.g += np.bincount(key, weights=w, minlength=n * NS).reshape(n, NS)
        if E_J is not None:
            np.copyto(w, E_J, casting="unsafe")
            self.E += np.bincount(e, weights=w, minlength=n)
        if self.count is not None:
            self.count += np.bincount(e, minlength=n)
        return self


def census(n_eid, *sources, count=False, scratch=None):
    """Build a Census from sources (eid, sp, mass_g, E_J) tuples."""
    c = Census(n_eid, count, scratch)
    for s in sources:
        c.add(*s)
    return c


def lattice_energy_share(E_cell, smass, c_solid, C_cell):
    """A thing's share of a cell's energy: every part of a cell shares one temperature, so
    the capacity share is exact: E_cell x smass c_solid / C_cell (spec 3.10)."""
    return np.asarray(E_cell, dtype=np.float64) * smass * c_solid / C_cell


# ── the ledger term ──────────────────────────────────────────────────────────
class Owned:
    """What the records own: grams per species, ml per fluid, joules. Exact zeros when empty."""

    def __init__(self):
        self.g = np.zeros(NS)
        self.ml = np.zeros(NF)
        self.E = 0.0

    def mass_total(self):
        return float(self.g.sum())

    def species(self):
        return {SP_NAMES[i]: float(self.g[i]) for i in range(NS) if self.g[i] != 0.0}


def records_owned(recs):
    """The ONE ledger term (2.1): grams per species (and the float64 `rem` of every
    hand-over), ml per fluid, joules, of class (b) rows with `owns` set. total_mass,
    total_fluid and total_energy add it. While there is no record and no remainder it
    returns exact 0.0 / zero arrays and walks nothing."""
    out = Owned()
    if recs is None or (recs.n == 0 and not recs.rem):
        return out
    for cls in OWNING:
        p = recs.pools[cls]
        r = p.alive_rows()
        if r.size == 0:
            continue
        own = p.a["owns"][r]
        out.g += np.sum(p.a["g"][r][own], axis=0, dtype=np.float64)
        out.E += float(np.sum(p.a["E"][r][own], dtype=np.float64))
        if "ml" in p.schema:
            out.ml += np.sum(p.a["ml"][r][own], axis=0, dtype=np.float64)
    for (host, field), v in recs.rem.items():
        if field == "E":
            out.E += v
        elif isinstance(field, tuple) and field[0] == "m":
            out.g[field[1]] += v
        elif isinstance(field, tuple) and field[0] == "ml":
            out.ml[field[1]] += v
    return out


# ── absorb, consume, reconcile (4.3) ─────────────────────────────────────────
def _meter(recs, host_id, cause, dE, dm):
    k = (int(host_id), cause)
    m = recs.meter.setdefault(k, [0.0, 0.0, 0.0, 0.0])
    m[0] += abs(dE)
    m[1] += abs(dm)
    m[2] += dE
    m[3] += dm


def _node_capacity(recs, rid):
    p, r = recs.row(rid)
    n = int(p.a["nn"][r])
    Y = p.a["Y"][r, :n]
    H = p.a["H"][r, :n]
    m = Y.sum(axis=1) / 1000.0
    if not Y.sum() > 0.0:                       # burnt to nothing: no capacity, no material
        return m, H, np.zeros(n), n
    mat = _row_mat(recs, Y.sum(axis=0))
    return m, H, m * mat.c_app(H), n


def absorb(recs, rid, dE, cause="absorb"):
    """Spread dE joules into a record by heat capacity (the catch-all for energy a law
    changed on the host that the record did not book). METERED: the audit shows how much
    it had to fix per tick per cause. Conserves exactly: sum m dH = dE."""
    cls, row = recs.loc[rid]
    p = recs.pools[cls]
    host = int(p.a["host_id"][row])
    if cls in THIN:
        m, H, C, n = _node_capacity(recs, rid)
        tot = C.sum()
        if tot <= 0.0:
            return 0.0
        share = dE * C / tot
        p.a["H"][row, :n] = H + np.where(m > 0.0, share / np.maximum(m, 1e-300), 0.0)
        p.a["quiet"][row] = False
    elif cls in (STRAND, CONDUIT):
        mt = p.a["g"][row].sum() / 1000.0
        if mt <= 0.0:
            return 0.0
        p.a["Hm"][row] += dE / mt
    elif cls in OWNING:
        p.a["E"][row] += dE
    else:
        return 0.0
    _meter(recs, host, cause, dE, 0.0)
    recs.note(rid, cause)
    return dE


def consume(recs, rid, species, grams, where="surface"):
    """Take up to `grams` of a species out of a record: SURFACE first (outer nodes inward;
    burn, char, acid, rust, wear; also grows `loss`), BULK from the node with the highest
    fraction first (evaporation of bound water), never more than there is, never negative.
    The matter leaves at its node's enthalpy. Returns (removed_g, removed_J, unmet_g)."""
    sp = species if isinstance(species, (int, np.integer)) else SP[species]
    cls, row = recs.loc[rid]
    p = recs.pools[cls]
    want = max(float(grams), 0.0)
    got_g = got_J = 0.0
    if cls in THIN:
        n = int(p.a["nn"][row])
        Y = p.a["Y"][row]
        H = p.a["H"][row]
        if p.a["sym"][row]:
            order = list(range(n - 1, -1, -1))              # face first, centre last
        else:                                               # both faces alike, then inward
            order = []
            for k in range((n + 1) // 2):
                order.append(k)
                if n - 1 - k != k:
                    order.append(n - 1 - k)
        if where == "bulk":
            frac = Y[:n, sp] / np.maximum(Y[:n].sum(axis=1), 1e-300)
            order = [int(i) for i in np.argsort(-frac, kind="stable")]
        for i in order:
            if want <= 0.0:
                break
            take = min(want, float(Y[i, sp]))
            if take <= 0.0:
                continue
            Y[i, sp] -= take
            want -= take
            got_g += take
            got_J += take / 1000.0 * float(H[i])
        if got_g > 0.0:
            if where != "bulk":
                dv = got_g / 1000.0 / float(RHO[sp])
                p.a["loss"][row] += dv / max(float(p.a["area"][row]), 1e-300) / 2.0
            p.a["quiet"][row] = False
            refresh_geometry(recs, rid)
            maybe_regrid(recs, rid)
    elif cls in (STRAND, CONDUIT) or cls in OWNING:
        take = min(want, float(p.a["g"][row][sp]))
        if take > 0.0:
            mt = float(p.a["g"][row].sum())
            if cls in OWNING:
                eper = float(p.a["E"][row]) / mt if mt > 0.0 else 0.0
                p.a["E"][row] -= eper * take
                got_J = eper * take
            else:
                got_J = take / 1000.0 * float(p.a["Hm"][row])
            p.a["g"][row][sp] -= take
            want -= take
            got_g = take
    if got_g > 0.0:
        recs.note(rid, "consume")
    return got_g, got_J, want


def deposit(recs, rid, species, grams, H=None):
    """Add grams of a species to the SURFACE nodes of a thin record (a gain the host
    reported: rust, soot, dew). The new matter arrives at the surface node's enthalpy
    unless H is given. Returns the J it brought."""
    sp = species if isinstance(species, (int, np.integer)) else SP[species]
    cls, row = recs.loc[rid]
    p = recs.pools[cls]
    grams = max(float(grams), 0.0)
    if grams == 0.0:
        return 0.0
    if cls in THIN:
        n = int(p.a["nn"][row])
        i = n - 1
        mo = float(p.a["Y"][row, i].sum()) / 1000.0
        Hi = float(p.a["H"][row, i]) if H is None else float(H)
        mn = mo + grams / 1000.0
        p.a["H"][row, i] = (mo * float(p.a["H"][row, i]) + grams / 1000.0 * Hi) / mn
        p.a["Y"][row, i, sp] += grams
        p.a["quiet"][row] = False
        refresh_geometry(recs, rid)
        maybe_regrid(recs, rid)
        recs.note(rid, "reconcile")
        return grams / 1000.0 * Hi
    p.a["g"][row][sp] += grams
    recs.note(rid, "reconcile")
    return 0.0


def reconcile(recs, cen, cause="reconcile", mode="surface", classes=THIN):
    """Make the record agree with its host (class (a); the host is the truth), once a tick
    per host that carries records. Mass: a loss is shed SURFACE-first (or BULK), a gain goes
    to the surface nodes, split between the host's records in proportion to their mass.
    Energy: whatever differs after that is `absorb`ed by heat capacity and METERED. After
    this the audit holds by construction, so an unknown law cannot break the books; the
    meter says how much it had to fix. `cen` has g (n_eid,NS) and E (n_eid,)."""
    for cls in classes:
        p = recs.pools[cls]
        r = p.alive_rows()
        if r.size == 0:
            continue
        for host in np.unique(p.a["host_id"][r]):
            hid = int(host)
            if hid >= cen.n_eid:
                continue
            rids = [int(x) for x in p.a["rid"][r][p.a["host_id"][r] == host]]
            rows = [recs.loc[x][1] for x in rids]
            gs = np.array([p.a["Y"][x].sum(axis=0) if cls in THIN else p.a["g"][x] for x in rows])
            tot = gs.sum(axis=0)
            dm = cen.g[hid] - tot
            for s in range(NS):
                if dm[s] == 0.0:
                    continue
                if tot[s] > 0.0:
                    share = gs[:, s] / tot[s]
                else:
                    share = np.full(len(rids), 1.0 / len(rids))
                for rid, sh in zip(rids, share):
                    d = dm[s] * sh
                    if d < 0.0:
                        _, J, _ = consume(recs, rid, s, -d, mode)
                    else:
                        deposit(recs, rid, s, d)
                _meter(recs, hid, cause, 0.0, float(dm[s]))
            Cs = []
            for rid in rids:
                Cs.append(_row_capacity(recs, rid))
            Er = sum(recs.energy_J(rid) for rid in rids)
            dE = float(cen.E[hid]) - Er
            if dE != 0.0:
                tc = sum(Cs)
                for rid, c in zip(rids, Cs):
                    absorb(recs, rid, dE * (c / tc if tc > 0 else 1.0 / len(rids)), cause)
    return recs.meter


def _row_capacity(recs, rid):
    """J/K of a record (m c_app summed over its nodes)."""
    cls = recs.loc[rid][0]
    p, r = recs.row(rid)
    if cls in THIN:
        return float(_node_capacity(recs, rid)[2].sum())
    if cls in (STRAND, CONDUIT):
        g = p.a["g"][r]
        if g.sum() <= 0.0:
            return 0.0
        return float(g.sum()) / 1000.0 * float(_row_mat(recs, g).c_app(np.array(p.a["Hm"][r])))
    return 0.0


# ── heat and moisture ticks (5.1, 5.2) ───────────────────────────────────────
def heat_tick(recs, dt, classes=THIN, newton=th.NEWTON, material_of=None, dT_max=th.DT_MAX):
    """Advance every awake thin record's profile one tick. Returns the TYPED DELTAS the
    host must apply exactly ONCE: a list of (rid, host_id, Q_J) with Q the joules that
    left the record through its ports. `reconcile` is the metered catch-all; the audit
    flags an absorb on a host no external law touched (single-writer rule, 5.3).
    A record with no matter left (or an empty node) is skipped and put to sleep, never an
    error: `spent(recs)` lists them and the host kills them. Rows are grouped by node count and composition; the first tick after a port change
    takes 4 substeps; a record whose step could move less than the host's float32 energy
    resolution goes quiet and wakes on any change of port, absorb or consume."""
    deltas = []
    for cls in classes:
        p = recs.pools[cls]
        rows = np.flatnonzero(p.a["alive"][:p.used] & ~p.a["quiet"][:p.used])
        if rows.size == 0:
            continue
        a = p.a
        used = np.arange(a["Y"].shape[1])[None, :] < a["nn"][rows][:, None]
        empty = ~((a["Y"][rows].sum(axis=2) > 0.0) | ~used).all(axis=1)   # a node with no matter has no state
        if empty.any():                      # burnt to nothing (or a node still to be regridded):
            a["quiet"][rows[empty]] = True   # nothing to solve; asleep until a deposit/absorb wakes it
            rows = rows[~empty]
            if rows.size == 0:
                continue
        for nn in np.unique(a["nn"][rows]):
            for sym in (True, False):
                sel = rows[(a["nn"][rows] == nn) & (a["sym"][rows] == sym)]
                if sel.size == 0:
                    continue
                n = int(nn)
                Y = a["Y"][sel, :n]
                comp = Y.sum(axis=1)
                wts = np.round(comp / comp.sum(axis=1, keepdims=True), 3)
                uw, inv = np.unique(wts, axis=0, return_inverse=True)
                inv = inv.reshape(-1)
                for gi in range(int(inv.max()) + 1):
                    g = sel[inv == gi]
                    mat = (material_of or (lattice_material if recs.lattice_cp else mix_material))(uw[gi])
                    H = a["H"][g, :n]
                    m = a["Y"][g, :n].sum(axis=2) / 1000.0
                    area = a["area"][g]
                    thick = a["thick"][g]
                    gk = th.conduction_gk(area, thick, n, sym)
                    Gp = a["port_h"][g] * area[:, None] * a["port_a"][g] * (2.0 if sym else 1.0)
                    if sym:
                        Gp[:, 0] = 0.0                        # the centre is adiabatic
                    Tenv = a["port_T"][g]
                    nmin = np.where(a["port_changed"][g], th.FIRST_TICK_SUBSTEPS, 1)
                    Hn, Q, ns = th.step(H, m, gk, Gp, Tenv, mat, dt, newton=newton, nsub_min=nmin,
                                        dT_max=dT_max)
                    a["H"][g, :n] = Hn
                    a["port_changed"][g] = False
                    scale = np.where(a["host_scale"][g] > 0.0, a["host_scale"][g],
                                     np.abs(m * Hn).sum(axis=1) + (m * mat.c_app(Hn)).sum(axis=1))
                    a["quiet"][g] = th.quiet_mask(Hn, m, gk, Gp, Tenv, mat, dt, scale)
                    stash = cache_of(recs).setdefault("nsub", {})
                    for i, r in enumerate(g):
                        stash[int(a["rid"][r])] = int(ns[i])
                        q = float(Q[i, 0] + Q[i, 1])
                        a["cause_tick"][r] = recs.tick
                        a["cause_verb"][r] = recs.cause_id("heat")
                        deltas.append((int(a["rid"][r]), int(a["host_id"][r]), q))
    return deltas


def spent(recs, classes=THIN):
    """Record ids of live thin records with no matter left (burnt, eaten or boiled away).
    heat_tick skips them and does not kill them: the host decides what a spent record
    means (ash, a hole, nothing) and kills it. Pure read."""
    out = []
    for cls in classes:
        p = recs.pools[cls]
        r = p.alive_rows()
        if r.size:
            dead = ~(p.a["Y"][r].sum(axis=(1, 2)) > 0.0)
            out.extend(int(x) for x in p.a["rid"][r][dead])
    return out


def set_port(recs, rid, index, h=None, T=None, a_scale=None):
    """Change a port. A change wakes the record and makes its next tick 4 substeps."""
    p, r = recs.row(rid)
    changed = False
    for key, val in (("port_h", h), ("port_T", T), ("port_a", a_scale)):
        if val is not None and p.a[key][r, index] != val:
            p.a[key][r, index] = val
            changed = True
    if changed:
        p.a["port_changed"][r] = True
        p.a["quiet"][r] = False
    return changed


def species_tick(recs, dt, mobile="water", classes=THIN):
    """Move a mobile species (moisture) through the thickness of awake thin records with a
    mass port (`port_rh` >= 0): D(X, T) from the row's matrix, boundary X_eq from the
    sorption isotherm at the port's relative humidity. Returns typed deltas
    (rid, host_id, dm_g, dE_J): the water that left the record and the enthalpy it took.
    Thickness is recomputed from the node volumes afterwards (shrinkage is geometry)."""
    from .record_data import diffusivity, eq_moisture
    deltas = []
    ms = SP[mobile]
    for cls in classes:
        p = recs.pools[cls]
        a = p.a
        rows = np.flatnonzero(a["alive"][:p.used] & (a["port_rh"][:p.used] >= 0.0).any(axis=1))
        for r in rows:
            n = int(a["nn"][r])
            Yr = a["Y"][r, :n]
            dry = Yr.copy()
            dry[:, ms] = 0.0
            Md = dry.sum(axis=1) / 1000.0
            if (Md <= 0.0).any():
                continue
            mi = int(np.argmax(dry.sum(axis=0)))
            matrix = SP_NAMES[mi]
            vol = node_volumes(Yr)
            rho_dry = Md / np.maximum(vol, 1e-300)
            sym = bool(a["sym"][r])
            gk = th.conduction_gk(a["area"][r:r + 1], a["thick"][r:r + 1], n, sym)
            mult = 2.0 if sym else 1.0
            beta = a["port_hm"][r]
            rh = a["port_rh"][r]
            Gm = np.where(rh >= 0.0, beta * a["area"][r] * mult * np.array([rho_dry[0], rho_dry[-1]]), 0.0)
            if sym:
                Gm[0] = 0.0
            Xeq = np.where(rh >= 0.0, eq_moisture(rh, matrix), 0.0)
            Hn = a["H"][r:r + 1, :n]
            Tn = _row_mat(recs, Yr.sum(axis=0)).T_of_H(Hn)
            Wd = Yr[:, ms][None, :] / 1000.0
            Wn, Hnew, out, _ = th.diffuse_step(
                Wd, Md[None, :], rho_dry[None, :], gk, lambda X, T, _m=matrix: diffusivity(X, T, _m),
                Tn, Gm[None, :], Xeq[None, :], Hn, dt)
            e_before = float((Yr.sum(axis=1) / 1000.0 * a["H"][r, :n]).sum())
            a["Y"][r, :n, ms] = Wn[0] * 1000.0
            a["H"][r, :n] = Hnew[0]
            e_after = float((a["Y"][r, :n].sum(axis=1) / 1000.0 * a["H"][r, :n]).sum())
            dm_g = float(out.sum()) * 1000.0
            deltas.append((int(a["rid"][r]), int(a["host_id"][r]), dm_g, e_before - e_after))
            a["cause_tick"][r] = recs.tick
            a["cause_verb"][r] = recs.cause_id("diffuse")
            a["quiet"][r] = False
            refresh_geometry(recs, int(a["rid"][r]))
            maybe_regrid(recs, int(a["rid"][r]))
    return deltas


# ── the audit (3.10) ─────────────────────────────────────────────────────────
def tol_for(dtype, n_touched=1):
    """Audit tolerance by dtype: 1e-12 for record-only float64, 2^-23 x touched cells for
    a lattice-held float32 host."""
    return 1e-12 if np.dtype(dtype) == np.float64 else 2.0 ** -23 * max(1, int(n_touched))


def records_audit(recs, host=None, views=(), ledger=None, tol=1e-12, tol_host=None,
                  scale_E=None, sink=None):
    """Check every store against its truth and return the findings (list of dicts):
      - class (a) rows against `host` (a Census): grams per species and joules, relative,
        at `tol_host` (default `tol`; pass tol_for(float32, cells) for a lattice host);
      - class (b) against `ledger` = {"g": vector, "E": J} what the ledger believes;
      - corruption: NaN, inf, negative grams, non-positive thickness, a stale index;
      - the ABSORB METER: reconcile had to fix more than tolerance on a host no external
        law touched this tick (a double-applied or unbooked flux);
      - `views`: record_view headers of foreign stores (layers pool, bones and wounds,
        clothes, mechanisms); each header's own audit hook runs, so one audit covers all.
    `scale_E` (default: the host's |E| plus the record's) sets the energy tolerance.
    Findings are also appended to `sink` (w.id_audit) when given."""
    out = []
    th_ = tol if tol_host is None else tol_host
    for rid, (cls, row) in recs.loc.items():
        p = recs.pools[cls]
        if not p.a["alive"][row] or int(p.a["rid"][row]) != rid:
            out.append(dict(kind="index", rid=rid, detail="loc points at a dead or foreign row"))
    for cls in sorted(recs.pools):
        p = recs.pools[cls]
        r, g, E = recs.totals(cls)
        if r.size == 0:
            continue
        for bad, why in ((~np.isfinite(g).all(axis=1) | ~np.isfinite(E), "non-finite totals"),
                         ((g < 0.0).any(axis=1), "negative grams")):
            for i in np.flatnonzero(bad):
                out.append(dict(kind="corrupt", rid=int(p.a["rid"][r[i]]), cls=cls, detail=why))
        if cls in THIN:
            nomass = ~(g.sum(axis=1) > 0.0) & (p.a["thick"][r] == 0.0)
            for i in np.flatnonzero(nomass):                     # burnt to nothing: not a fault,
                out.append(dict(kind="spent", rid=int(p.a["rid"][r[i]]), cls=cls,   # a reminder to kill it
                                detail="no matter left; the host should kill this record"))
            for i in np.flatnonzero((~(p.a["thick"][r] > 0.0) | ~(p.a["area"][r] > 0.0)) & ~nomass):
                out.append(dict(kind="corrupt", rid=int(p.a["rid"][r[i]]), cls=cls,
                                detail="thickness or area not positive"))
            # the stored thickness must follow the node volumes
            vol = node_volumes(p.a["Y"][r]).sum(axis=1)
            want = vol / np.maximum(p.a["area"][r], 1e-300)
            for i in np.flatnonzero(np.abs(want - p.a["thick"][r]) > 1e-9 * np.maximum(want, 1e-300)):
                out.append(dict(kind="corrupt", rid=int(p.a["rid"][r[i]]), cls=cls,
                                detail="thickness does not follow the node volumes"))
    if host is not None:
        for cls in SHAPE_ONLY:
            p = recs.pools[cls]
            r, g, E = recs.totals(cls)
            if r.size == 0:
                continue
            hid = p.a["host_id"][r]
            for h in np.unique(hid):
                if int(h) >= host.n_eid:
                    out.append(dict(kind="host", rid=int(p.a["rid"][r[hid == h][0]]),
                                    detail="host id outside the census"))
                    continue
                sel = hid == h
                gs = g[sel].sum(axis=0)
                Es = float(E[sel].sum())
                for s in range(NS):
                    ref = float(host.g[h, s])
                    if abs(gs[s] - ref) > th_ * max(abs(ref), abs(gs[s]), 1e-300) and \
                            abs(gs[s] - ref) > 1e-300:
                        out.append(dict(kind="host", rid=int(p.a["rid"][r[sel][0]]), host=int(h),
                                        species=SP_NAMES[s], rec=float(gs[s]), host_g=ref,
                                        detail="grams differ from the host"))
                sc = (scale_E if scale_E is not None
                      else max(abs(float(host.E[h])), abs(Es), 1e-300))
                if abs(Es - float(host.E[h])) > th_ * sc:
                    out.append(dict(kind="host", rid=int(p.a["rid"][r[sel][0]]), host=int(h),
                                    rec=Es, host_E=float(host.E[h]), detail="energy differs from the host"))
    if ledger is not None:
        o = records_owned(recs)
        for s in range(NS):
            ref = float(ledger["g"][s])
            if abs(o.g[s] - ref) > tol * max(abs(ref), abs(o.g[s]), 1e-300):
                out.append(dict(kind="ledger", species=SP_NAMES[s], detail="owned grams differ from the ledger"))
        if abs(o.E - float(ledger["E"])) > tol * max(abs(o.E), abs(float(ledger["E"])), 1e-300):
            out.append(dict(kind="ledger", detail="owned joules differ from the ledger"))
    for (hid, cause), (aE, am, _, _) in sorted(recs.meter.items()):
        if hid in recs.touched or cause == "external":
            continue
        scale = scale_E
        if scale is None and host is not None and hid < host.n_eid:
            scale = abs(float(host.E[hid]))
        scale = max(scale or 0.0, 1e-300)
        mscale = float(host.g[hid].sum()) if host is not None and hid < host.n_eid else 1.0
        if aE > th_ * scale or am > th_ * max(mscale, 1e-300):
            out.append(dict(kind="absorb", host=hid, cause=cause, dE=aE, dm=am,
                            detail="reconcile absorbed a change no external law booked"))
    for v in views:
        out += list(v.audit())
    if sink is not None:
        sink.extend(out)
    return out
