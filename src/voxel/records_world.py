"""RecordsMixin: the records on the world (wave 13c, lane C5, T2).

A RECORD is a small exact description of something thinner than a cell (a blade, a
hide, a plate) or of how many small things there are, riding on a HOST thing. The pools
(`record.Records`), the thin heat solver, the hand-over and the joint library were built
and tested alone in phase A. This mixin is the glue to the live world.

WHAT IS ON (spec 10, slices):
  1 BOOKS.    `w.records` pools, the census of what each host holds, `reconcile` once a
              tick (the host is the truth for matter and heat), the ledger term
              `records_owned`, `records_audit_world()`, split with the host, death with
              the host, snapshot and restore (the pools are plain arrays).
  2a HEAT.    A declared SECTION or SHEET runs the thin enthalpy solve with ports on the
              host's own surroundings (air, or water on a neighbour face; h from `surfaceh`:
              convection, boiling curve, radiation). The result is a READOUT: surface and core temperature
              and the cooling rate. The lattice's own two-capacity cell law still moves
              the host's heat; the record's total follows the host through `reconcile`,
              and what reconcile had to move is metered (`records.meter`). 2b (the record
              hands its face flux to the host law) is NOT built: see the report.
  5 JOINTS.   `join_things` makes a JOIN (glue line, nail, lashing...) between two things.
              A composite keeps its parts and its joints; `composite_check` finds the
              weakest joint and lets it fail; the parts stay things with their own eids.
              Glue cures with time (`joinery.cure_step`), once a tick, while it is
              uncured.

THE GATE. A world with no record is the old world bit for bit: `records` is the class
default None, every hook starts `if self.records is None or not self.records.n`, and the
ledger term reads exact zero. Records are DECLARED by a scene or an act (decision 3 of the
spec), never inferred. `rec_laws` (class attribute, True) turns the heat readout off on a
world that wants only the books.

CHEAP ON PURPOSE: one census pass per tick over the cells of hosts that carry a record
(not the lattice); the heat solve is batched and quiet records are skipped; the port
environment is read from the six neighbours of the host's cells only when a record is
awake."""
import numpy as np

from . import joinery as JN
from . import record as R
from . import surfaceh as SH
from .materials import AIR, GLASS, IRON, TICK_S, WATER, WOOD, _CSOLID_ARR
from .record_data import JOIN_KINDS, NS, SP, SP_NAMES

# lattice material -> record species (stand-in table; T1's pool replaces it). -1 = not tracked
_SPMAP = np.full(64, -1, np.int64)
for _m, _n in ((IRON, "iron"), (WOOD, "wood"), (GLASS, "glass")):
    _SPMAP[_m] = SP[_n]
_KINDS = tuple(JOIN_KINDS)          # the stored kind index is the position in this tuple
_SUBSTRATE_OF = {WOOD: "wood", IRON: "steel", GLASS: "glass"}
T_K = 273.15
PORT_H_BAND = 0.05        # fraction of h; a smaller change keeps the port (and the 1-substep step)
PORT_T_BAND = 1.0         # K
PORT_EVERY = 4            # ticks between looks at the six neighbours (0.1 s)


class RecordsMixin:
    """Part of ThingsMixin's bases (so the hooks sit where a thing's identity is kept)."""

    rec_laws = True            # the heat readout (needs records); False = books only
    rec_read = None            # rid -> readout dict, made with the first awake record
    composites = None          # gid -> composite dict, made with the first join

    # ------------------------------------------------------------------ declare
    def _records_live(self):
        r = self.records
        return r is not None and r.n > 0

    def _host_state(self, eid):
        """(flat cells, grams per species, joules, mean T) of thing `eid` on the lattice."""
        at = self._thing_cells().get(int(eid))
        if at is None or not len(at):
            return None
        idx = np.unravel_index(at, self.shape)
        sp = _SPMAP[self.mat[idx]]
        g = self.smass[idx].astype(np.float64)
        keep = sp >= 0
        grams = np.bincount(sp[keep], weights=g[keep], minlength=NS)
        C = self.heat_capacity()[idx]
        Ecell = self.E[idx].astype(np.float64)
        E = float((Ecell * g * _CSOLID_ARR[self.mat[idx]] / C)[keep].sum())
        T = float(self.T_at(idx).mean())
        return at, grams, E, T

    def declare_section(self, eid, thick, width=None, length=None, sheet=False, nn=7):
        """Give thing `eid` a SECTION (bar, blade) or SHEET (hide, plate): a record of
        real thickness `thick` m. `width` and `length` default from the cells: the
        longest extent of the thing, and its mass over the other two. The grams and the
        temperature are the host's, so the record starts in agreement with it."""
        recs = self.ensure_records()
        recs.lattice_cp = True              # agree with the host's cells in joules
        st = self._host_state(eid)
        if st is None:
            raise ValueError("thing %r has no cells on the lattice" % (eid,))
        at, grams, E, T = st
        if grams.sum() <= 0.0:
            raise ValueError("thing %r holds no tracked matter" % (eid,))
        comp = {SP_NAMES[i]: float(grams[i]) for i in range(NS) if grams[i] > 0.0}
        dens = 1.0 / float(R.node_volumes(grams / grams.sum() * 1000.0))     # kg/m3 of the mix
        vol = grams.sum() / 1000.0 / dens
        xyz = np.stack(np.unravel_index(at, self.shape), -1)
        dx = 0.1 * self.scale
        ext = (xyz.max(axis=0) - xyz.min(axis=0) + 1) * dx
        L = float(length) if length else float(ext.max())
        Wd = float(width) if width else vol / (L * thick)
        rid = R.make_thin(recs, R.SHEET if sheet else R.SECTION, comp, thick, Wd, L,
                          T=T, nn=nn, host_kind=R.HOST_THING, host_id=int(eid))
        p, r = recs.row(rid)
        # make_thin sizes the record from thick x width x length; the host's grams are the
        # truth, so scale the nodes to them (mass exact; reconcile would do it next tick)
        have = float(p.a["Y"][r].sum())
        if have > 0.0:
            p.a["Y"][r] *= grams.sum() / have
        recs.touch(eid)
        return rid

    # ------------------------------------------------------------------ the books
    def records_owned(self):
        """Book T2: grams the records own (glue lines, lots, cups). Exact 0.0 with none."""
        if self.records is None:
            return 0.0
        return float(R.records_owned(self.records).mass_total())

    def records_owned_E(self):
        """Joules the records own; add to total_energy (ledgers.total_energy does)."""
        if self.records is None:
            return 0.0
        return float(R.records_owned(self.records).E)

    def records_audit_world(self, tol=1e-12):
        """records_audit against this world: every host's matter and joules, the
        absorb meter, and the corruption checks. [] when clean."""
        if self.records is None:
            return []
        cen = self._records_census(self.records)
        return R.records_audit(self.records, host=cen, tol=tol,
                               tol_host=R.tol_for(np.float32, 4))

    def _records_census(self, recs):
        hosts = recs.hosts(R.SHAPE_ONLY)
        n = int(self._next_eid) + 1
        cen = R.Census(n)
        for h in hosts:
            st = self._host_state(int(h))
            if st is None:                     # off the lattice: the record sleeps; its
                g, E = _host_totals(recs, int(h))      # totals stand for the host
                cen.g[int(h)], cen.E[int(h)] = g, E
            else:
                cen.g[int(h)], cen.E[int(h)] = st[1], st[2]
        return cen

    # ------------------------------------------------------------------ the tick
    def _law_records(self):
        """Once a tick, at the end of the identity pass (after the laws that change
        matter have run): reconcile to the host, port the awake thin records to their
        surroundings, advance them, read them out, cure the glue."""
        recs = self.records
        recs.begin_tick(self.tick)
        if self.rec_read is None:
            self.rec_read = {}
        if recs.hosts(R.SHAPE_ONLY).size:
            deltas = self._records_heat() if self.rec_laws else []     # the profile first...
            cen = self._records_census(recs)
            # the lattice's own law moved the host this tick: the metered catch-all
            # (reconcile) is expected to move the record, it is not a fault
            for h in recs.hosts(R.SHAPE_ONLY):
                recs.touch(int(h))
            R.reconcile(recs, cen, cause="reconcile")                   # ...then agree with the host
            st = self.rec_read.setdefault("_stat", {"absorb_J": 0.0, "ticks": 0})
            st["absorb_J"] += sum(v[0] for k, v in recs.meter.items() if k[1] == "reconcile")
            st["ticks"] += 1
            for rid in R.spent(recs):          # burnt to nothing: the host decides, a hole
                recs.kill(rid)
            self._records_read(deltas)
        if self.composites:
            self._records_cure()

    def _records_heat(self):
        recs = self.records
        for cls in R.THIN:
            p = recs.pools[cls]
            rows = np.flatnonzero(p.a["alive"][:p.used])
            for row in rows:
                rid = int(p.a["rid"][row])
                if (self.tick + rid) % PORT_EVERY:      # surroundings change in tenths of a second
                    continue
                env = self._port_env(int(p.a["host_id"][row]), rid, p, row)
                if env is not None:
                    h0, T0 = p.a["port_h"][row, 1], p.a["port_T"][row, 1]
                    # a port is re-set only when the surroundings moved past a band: a new
                    # port costs a 4-substep first tick, and h itself is a 20 percent law
                    if abs(env[0] - h0) > PORT_H_BAND * h0 or abs(env[1] - T0) > PORT_T_BAND:
                        R.set_port(recs, rid, 1, h=env[0], T=env[1])
                        if not p.a["sym"][row]:
                            R.set_port(recs, rid, 0, h=env[0], T=env[1])
        return R.heat_tick(recs, TICK_S)

    def _records_read(self, deltas):
        """The readout of every record that stepped: surface and core T, cooling rate."""
        recs = self.records
        nsub = R.cache_of(recs).get("nsub", {})
        for rid, hid, q in deltas:
            if rid not in recs.loc:
                continue
            p, r = recs.row(rid)
            n = int(p.a["nn"][r])
            Tn = self._record_T(p, r)
            prev = self.rec_read.get(rid)
            surf, core = float(Tn[n - 1]), float(Tn[0])
            self.rec_read[rid] = {
                "T_surf": surf, "T_core": core,
                "dTdt": 0.0 if prev is None else (prev["T_surf"] - surf) / TICK_S,
                "q_W": q / TICK_S, "nsub": nsub.get(rid, 1)}
        for rid in [k for k in self.rec_read if k != "_stat" and k not in recs.loc]:
            del self.rec_read[rid]

    def _record_T(self, p, r):
        n = int(p.a["nn"][r])
        mat = R._row_mat(self.records, p.a["Y"][r, :n].sum(axis=0))
        return mat.T_of_H(p.a["H"][r, :n])

    def _port_env(self, eid, rid, p, row):
        """(h W/m2K, T C) of what surrounds thing `eid`: the six neighbours of its cells
        that are not it. Water on a face wets the record (h from surfaceh: Churchill-Chu and the
        flat plate law, then the boiling curve above 100 C); air faces get convection plus
        radiation. Each medium has its own h, mixed by face count."""
        at = self._thing_cells().get(eid)
        if at is None or not len(at):
            return None
        nx, ny, nz = self.shape
        x, y, z = np.unravel_index(at, self.shape)
        wet_T, air_T = [], []
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            a, b, c = x + d[0], y + d[1], z + d[2]
            ok = (a >= 0) & (a < nx) & (b >= 0) & (b < ny) & (c >= 0) & (c < nz)
            a, b, c = a[ok], b[ok], c[ok]
            free = (self.eid[a, b, c] != eid) & ((self.mat[a, b, c] == AIR) | (self.fvol[a, b, c] > 0.0))
            a, b, c = a[free], b[free], c[free]
            if not len(a):
                continue
            Tn = self.T_at((a, b, c))
            wet = (self.fl[a, b, c] == WATER) & (self.fvol[a, b, c] > 0.02)
            wet_T.append(Tn[wet])
            air_T.append(Tn[~wet])
        wet_T = np.concatenate(wet_T) if wet_T else np.zeros(0)
        air_T = np.concatenate(air_T) if air_T else np.zeros(0)
        n = int(p.a["nn"][row])
        Ts = float(self._record_T(p, row)[n - 1])
        Lc = max(float(p.a["seg_len"][row]), 1e-3)
        faces = []                              # each medium its own h (S4-2): water faces get
        if len(wet_T):                          # the boiling curve, air faces convection + radiation
            faces.append((len(wet_T), float(wet_T.mean()), True))
        if len(air_T):
            faces.append((len(air_T), float(air_T.mean()), False))
        return SH.port_of(Ts, faces, Lc, int(self.mat[x[0], y[0], z[0]]))

    # ------------------------------------------------------------------ with the host
    def _records_split(self, e, born, masses):
        """Thing `e` came apart into the pieces `born` (new eids) with these masses (g;
        the keeper's first). Each record of `e` is cut at the pieces' mass shares: the
        shares and the energy add back exactly (R.split), the keeper keeps the rest."""
        recs = self.records
        for cls in sorted(recs.pools):
            p = recs.pools[cls]
            rids = [int(x) for x in p.a["rid"][p.alive_rows()][p.a["host_id"][p.alive_rows()] == e]]
            for rid in rids:
                left = float(sum(masses))
                for c, m in zip(born, masses[1:]):
                    if left <= 0.0 or m <= 0.0:
                        break
                    f = (left - m) / left               # the keeper's share of what is left
                    if 0.0 < f < 1.0:
                        recs.split(rid, f, host_id=int(c))
                    left -= m

    def _records_died(self, e):
        """Thing `e` is nowhere: its shape-only records have nothing to describe."""
        recs = self.records
        for cls in R.SHAPE_ONLY:
            p = recs.pools[cls]
            r = p.alive_rows()
            for rid in [int(x) for x in p.a["rid"][r][p.a["host_id"][r] == e]]:
                recs.kill(rid)
        if self.rec_read:
            for rid in [k for k in self.rec_read if k != "_stat" and k not in recs.loc]:
                del self.rec_read[rid]

    # ------------------------------------------------------------------ joints
    def join_things(self, a, b, kind, area, mass_g=0.0, substrate=None, name=None, **cap):
        """JOIN thing `a` to thing `b` with a joining material of `kind` (a JOIN_KINDS
        name) over `area` m2. The act brings `mass_g` grams of joining material (class b
        matter: the record owns it and `records_owned` counts it). A new JOIN has alpha 0
        for a glue that must cure and 1 for a nail or a lashing. Returns the joint's rid.
        The composite (a, b, and every thing already joined to them) keeps its parts."""
        recs = self.ensure_records()
        if kind not in JOIN_KINDS:
            raise KeyError(kind)
        j = JN.new_joint(kind, area, mass_g={"glue": mass_g} if mass_g else None, a=int(a), b=int(b))
        rid = recs.add(R.JOIN, R.HOST_THING, int(a), eid_a=int(a), eid_b=int(b),
                       kind=_KINDS.index(kind), area=float(area), alpha=j["alpha"],
                       moisture=1.0)
        p, r = recs.row(rid)
        if mass_g:
            p.a["g"][r, SP["glue"]] = float(mass_g)
            p.a["owns"][r] = True
        else:
            p.a["owns"][r] = False
        j["rid"] = rid
        j["substrate"] = substrate or self._substrate_of(a, b)
        j["cap"] = cap
        if self.composites is None:
            self.composites = {}
        gid = self._composite_of(a) or self._composite_of(b)
        if gid is None:
            gid = 1 + len(self.composites)
            self.composites[gid] = JN.new_group(gid, name or "composite", [int(a), int(b)], [j])
        else:
            g = self.composites[gid]
            for m in (int(a), int(b)):
                if m not in g["members"]:
                    g["members"].append(m)
            g["joints"].append(j)
            g["status"] = "whole"
        return rid

    def _substrate_of(self, a, b):
        out = []
        for e in (a, b):
            at = self._thing_cells().get(int(e))
            if at is None or not len(at):
                continue
            m = int(np.bincount(self.mat.ravel()[at]).argmax())
            out.append(_SUBSTRATE_OF.get(m, "wood"))
        return tuple(dict.fromkeys(out)) or "wood"

    def _composite_of(self, eid):
        for gid, g in (self.composites or {}).items():
            if int(eid) in g["members"]:
                return gid
        return None

    def composite_parts(self, gid):
        g = self.composites[gid]
        return list(g["members"])

    def joint_capacity(self, gid, rid):
        g = self.composites[gid]
        j = next(j for j in g["joints"] if j["rid"] == rid)
        a = j["alpha"]
        return JN.capacity(j["kind"], alpha=a, area=j["area"], substrate=j["substrate"], **j["cap"])

    def composite_check(self, gid, loads):
        """Apply `loads` ({joint rid: (Fn N, V N, M N m)}) to composite `gid`. The weakest
        joint (the largest utilisation) fails if it is past 1; the parts and the joint stay
        in the composite, which is 'apart' when its live joints no longer connect it.
        Returns (rid of the joint that failed or None, components)."""
        g = self.composites[gid]
        caps = {j["rid"]: self.joint_capacity(gid, j["rid"]) for j in g["joints"]
                if j.get("failed") is None}
        weak = JN.weakest_joint(g, loads, caps)
        if weak is None or weak[1] <= 1.0:
            return None, JN.components(g["members"], g["joints"])
        comps = JN.fail_joint(g, weak[0], tick=self.tick)
        self._props = (None, {})                    # what a mind read of the parts is stale
        recs = self.records
        p, r = recs.row(weak[0])
        p.a["fail_tick"][r], p.a["fail_mode"][r] = self.tick, 0
        p.a["alpha"][r] = 0.0
        return weak[0], comps

    def _records_cure(self):
        """Glue sets with time (once a tick, only while uncured): maturity and alpha."""
        recs = self.records
        for g in self.composites.values():
            for j in g["joints"]:
                if j.get("failed") is not None or j["alpha"] >= 1.0:
                    continue
                st = self._host_state(j["a"])
                T = (st[3] if st else 20.0) + T_K
                JN.cure_step(j, TICK_S, T=T, rh=1.0)
                p, r = recs.row(j["rid"])
                p.a["alpha"][r], p.a["maturity"][r], p.a["moisture"][r] = (
                    j["alpha"], j["te"], j["moisture"])

    def composite_props(self, eid):
        """What a mind may read of a thing in a composite: its parts and the load the
        weakest live joint holds (N, shear). None for a thing in none."""
        gid = self._composite_of(eid)
        if gid is None:
            return None
        g = self.composites[gid]
        live = [self.joint_capacity(gid, j["rid"]).Fs for j in g["joints"]
                if j.get("failed") is None]
        return {"composite": gid, "parts": tuple(g["members"]), "status": g["status"],
                "weakest_joint_N": min(live) if live else 0.0}


def _host_totals(recs, hid):
    """(grams per species, joules) the records on host `hid` hold now."""
    g, E = np.zeros(NS), 0.0
    for cls in R.SHAPE_ONLY:
        p = recs.pools[cls]
        r = p.alive_rows()
        for rid in p.a["rid"][r][p.a["host_id"][r] == hid]:
            gg, ee = recs.row_totals(int(rid))
            g, E = g + gg, E + ee
    return g, E
