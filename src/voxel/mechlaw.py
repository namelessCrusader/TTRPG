"""MechanismsMixin: doors, latches, bars and springs in the World (T11, 6).

A mixin written on World's public API only (`register`, `stamp`, `fill`,
`_lift`, `_put`, `_noise`, `_defer`, `emit`, `because`, `_shatter`), so phase B
changes the class list and nothing else:

    class World(MechanismsMixin, PathingMixin, ..., ThingsMixin, ...)

WHAT IT KEEPS (F1-F11 of the spec)

  * A door is cells. `add_door` stamps a frame thing and a leaf thing and
    builds a MECHANISM record (mechanisms.py) in `w.mechs`: geometry and
    state, no mass, nothing a law computes. Smoke, sound, heat and sight read
    the lattice, so a leaf that moves needs NO change in those laws.
  * "Locked" is a bolt crossing the gap into a keeper. "Barred" is a bar
    thing seated in two brackets. "Held by ice" is matter that must be there.
    Nothing is declared LOCKED or BROKEN: a lost capability comes from
    structure (the keeper split, the pin sheared, the jamb ash).
  * ENERGY, the ONE-FLOW rule. A joule an actor delivers to a mechanism is
    debited from that actor and credited to work_out; mech_in books only a
    release with no actor (a spring let go, a slam from stored swing, a burnt
    jamb). `deposit()` picks the book. Elastic, kinetic and potential energy of
    a mechanism are off the ledger and become heat inside the same event.
  * A SWING is an event with a duration: the leaf stays DRAWN CLOSED while it
    runs (every law that reads the lattice sees a shut door), then turns by an
    exact EXCHANGE (the leaf cells and the room air they enter swap, as `_walk`
    does), so air, heat and fluid are not lost. A body in the swept sector
    blocks it, or aborts it with the kinetic energy as a blow.
  * A blow reaches the links through the plate statics of a rigid leaf on
    springs, the two-body impact, and Miner damage; the failing link is the
    weakest of the computed capacities for THAT direction of push.

Declared simplifications: open and closed only (no ajar drawing, D1); the
leaf is one cell thick (the real 4.5 cm at 5 cm voxels, mass +11%; it raises
at 10 cm unless `heavy=True`); things hung on the leaf are ignored.
"""
import math

import numpy as np

from . import mechanisms as mech
from .events import hash01
from .materials import AIR, BODY, FLESH, TICK_S, WOOD
from . import exertion as _EX

try:                                      # the lock: phase A cross-module import
    from . import locks as _locks
except ImportError:                       # pragma: no cover - stand-in absent
    _locks = None

GRAV = mech.GRAVITY
KICK_M_KG = 5.6          # (u) effective mass of a kicking foot and shin
KICK_V_MPS = 6.0         # (u) its speed at contact: 100 J at strength 400 N
E_C_FOOT = 0.2           # (u) restitution of a shod foot on wood
K_CONTACT = 1.0e5        # N/m (u) a shod foot on wood
MU_FOOT = 0.6            # (u) a foot on a floor: a push is limited by mu m g
LEAK_CLEAR_MM = {"side": 3.0, "top": 3.0, "under": 12.0}   # a real leaf's fit
MAX_LEAF_CELL_M = 0.060  # a leaf one cell thick: 4.5 cm real; refuse beyond this
WAIT_TICKS = 80         # a redraw waits this long for the solver to let go of a cell
REACH_M = 0.7            # a mechanism within an arm's length of a body is in reach
# THE TENSION A PERSON CHOOSES when working a lock with picks (locks.py: the
# bind window is TAU_BIND 0.01 N m to tau_max 0.071 N m, `locks.window_Nm()`;
# a light turn is in the lower half of it, a firm one in the upper half).
# (u): the two named tensions; the window is the law's, the names are a choice.
_ORDINAL = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")
PICK_TENSION_NM = {"light": 0.03, "firm": 0.05}
# the heat of a stop or a slam is spread over the cells that took it
_HEAT_MAX_CELLS = 12

# Link stiffnesses (N/m) that make the latch path about 8e4 N/m in series,
# the order the spec's plate-statics check used (5e4 to 6.7e4), and the
# deadbolt path stiffer (it is a thrown 25 mm bolt in a longer-screwed plate).
# DATA (u): a reviewer may retune them; the tests assert orderings only.
_K_LATCH = {"bolt": 2.0e6, "keeper": 1.0e6, "strike": 3.0e5, "split": 4.0e5,
            "case": 6.0e5, "bore": 1.0e6}
_K_DEAD = {"shear": 4.0e6, "keeper": 3.0e6, "strike": 5.0e5, "split": 4.0e5}


def _tuplify(v):
    return tuple(int(x) for x in v)


def _cols(a):
    """The distinct (x, y) columns of a set of cells, as an (n, 2) int array
    (a leaf is 40 cells tall over 20 columns: the gap needs only the columns).
    Sorted by (x, y), and found on one integer key, which is much faster than
    a unique over rows."""
    a = np.asarray(a, np.int64)
    k = np.unique(a[:, 0] * 65536 + a[:, 1])
    return np.stack((k // 65536, k % 65536), -1)


def _gap_cells(a, b):
    """The smallest horizontal (Chebyshev) gap, in cells, between two sets of
    cells: the same number as a pairwise min over every cell of each, taken
    over their distinct columns only."""
    ca, cb = _cols(a), _cols(b)
    return int(np.abs(ca[:, None, :] - cb[None, :, :]).max(axis=2).min())


class MechanismsMixin:
    # the two energy books this mixin owns; instance values appear on first use
    mech_in = 0.0                 # joules booked for releases with no actor
    work_out = 0.0                # joules actors delivered to mechanisms
    _mech_phase = False           # True while the law runs (events are deferred)
    # GATE (wave 14b, lane G; ON for every world since 14bF, W15-Q8 (a)): True draws
    # every strike, throw, lift and haul from the body (exertion.py): muscle force,
    # segment mass, technique, training, fatigue, injury, and a per person spread
    # around the ordinary 115 J kick, paid from the body's books. False keeps the old
    # flat numbers (5.6 kg x 6 m/s = 100 J at strength 400 N), bit for bit.
    exertion_from_body = True

    # ── bookkeeping ─────────────────────────────────────────────────────────
    def _mech_tables(self):
        if "mechs" not in self.__dict__:
            self.mechs = {}
            self._mech_next = 1
            self._mech_active = set()
            self._mech_watch = set()
        return self.mechs

    def _ev(self, verb, **kw):
        """An event: deferred while a law runs (so ids are not consumed early),
        emitted at once from an act."""
        if self._mech_phase:
            return self._defer(verb, **kw)
        return self.emit(verb, **kw)

    def _d(self):
        return 0.1 * self.scale

    def _cells_of(self, eid):
        """The lattice cells of a thing as an (n, 3) int array (rubble shards
        excluded: a shard holds nothing), or None."""
        idx = self._thing_cells().get(int(eid))
        if idx is None or not len(idx):
            return None
        idx = idx[~self.rubble.ravel()[idx]]
        if not len(idx):
            return None
        return np.stack(np.unravel_index(idx, self.shape), -1).astype(np.int64)

    def _mass_kg(self, cells):
        c = np.asarray(cells, np.int64)
        return self.smass[c[:, 0], c[:, 1], c[:, 2]].astype(np.float64) / 1000.0

    def _flaw_of(self, rec, lid):
        if rec.get("flaw_amp", mech.FLAW_AMP) == 0.0:
            return 0.5
        return hash01("mech", rec["mid"], lid)

    def _flaws(self, rec):
        return {lk["id"]: self._flaw_of(rec, lk["id"]) for lk in self._mlinks(rec)}

    @staticmethod
    def _mlinks(rec):
        return rec["links"]

    @staticmethod
    def _mlink(rec, lid):
        for lk in rec["links"]:
            if lk["id"] == lid:
                return lk
        raise KeyError(lid)

    # ── building a door ─────────────────────────────────────────────────────
    def add_door(self, hinge_xy, width_cells, height_cells, swing=+1, mat=WOOD,
                 frac=None, leak_clear_mm=None, lock=None, latch=True, bar=None,
                 frame_mat=WOOD, flaw=True, heavy=False, z0=1, along=(1, 0),
                 corner=(-1, -1), deadbolt=False, name="door", stud=False):
        """Stamp a door and return its mechanism id.

        `hinge_xy` is the first cell of the leaf; the leaf runs `width_cells`
        along `along` and `height_cells` up from `z0`; the wall around it is the
        author's. The frame thing is the two jambs and the head (one cell each,
        in `frame_mat`) round the opening. `corner` picks the cell corner the
        hinge pin is on (the pivot, in doubled coordinates, is exact), `swing`
        (+1 or -1) the sense the leaf turns to open.

        `frac` is the authored fill of the leaf (the old fill()-door's 0.97,
        for a migrated door), or derived from the clearances round a real leaf
        (3 mm at the sides and head, 12 mm under: 0.986 for 0.9 x 2.03 m);
        `leak_clear_mm` overrides them. Smoke, sound and heat leak by the fill,
        as before. `latch` adds a latchbolt and its keeper; `deadbolt` a thrown
        bolt of its own (held by `lock` if one is given); `bar` seats a bar in
        two brackets. `stud` stiffens the jamb behind the keeper (t_e 30 mm
        instead of 12). `flaw` False sets the flaw spread to zero (exact orders).
        """
        d = self._d()
        if d > MAX_LEAF_CELL_M and not heavy:
            raise ValueError(
                "a one-cell leaf at voxel_cm %g is %.0f cm thick (a real one is "
                "4.5 cm); pass heavy=True to accept the mass" %
                (self.scale * 10.0, d * 100.0))
        if swing not in (1, -1):
            raise ValueError("swing is +1 or -1")
        tabs = self._mech_tables()
        x0, y0 = int(hinge_xy[0]), int(hinge_xy[1])
        ax, ay = int(along[0]), int(along[1])
        if abs(ax) + abs(ay) != 1:
            raise ValueError("along is a grid axis")
        W, H = int(width_cells), int(height_cells)
        mid = self._mech_next
        self._mech_next += 1
        clear = dict(LEAK_CLEAR_MM)
        if leak_clear_mm:
            clear.update(leak_clear_mm)
        Wm, Hm = W * d, H * d
        if frac is None:
            open_area = (2 * clear["side"] * Hm + clear["top"] * Wm
                         + clear["under"] * Wm) / 1000.0
            frac = 1.0 - open_area / (Wm * Hm)
        frame_e = self.register("%s frame" % name, "frame")
        leaf_e = self.register(name, "door")
        old = self._stamping
        # the frame: hinge jamb (j = -1), latch jamb (j = W), head over both
        self._stamping = frame_e
        zt = z0 + H
        for j in (-1, W):
            cx, cy = x0 + ax * j, y0 + ay * j
            self.fill(cx, cx + 1, cy, cy + 1, z0, zt, frame_mat)
        j0, j1 = (-1, W)
        hx0, hx1 = sorted((x0 + ax * j0, x0 + ax * j1))
        hy0, hy1 = sorted((y0 + ay * j0, y0 + ay * j1))
        self.fill(hx0, hx1 + 1, hy0, hy1 + 1, zt, zt + 1, frame_mat)
        # the leaf
        self._stamping = leaf_e
        lx0, lx1 = sorted((x0, x0 + ax * (W - 1)))
        ly0, ly1 = sorted((y0, y0 + ay * (W - 1)))
        self.fill(lx0, lx1 + 1, ly0, ly1 + 1, z0, zt, mat, frac=frac)
        self._stamping = old
        rec = mech.new_mechanism(mid, name, leaf=leaf_e, frame=frame_e)
        rec["clear_mm"] = clear
        rec["flaw_amp"] = mech.FLAW_AMP if flaw else 0.0
        pivot2 = (2 * x0 + int(corner[0]), 2 * y0 + int(corner[1]))
        rec.update(pivot2=list(pivot2), along=[ax, ay], s_open=int(swing),
                   z0=int(z0), W=W, H=H, frac=float(frac), pushes=[],
                   hinges=[], bar=None, voxel_cm=self.scale * 10.0)
        j = mech.new_joint("pin", "hinge", "leaf", "frame", axis2=pivot2,
                           sign=swing)
        rec["joints"].append(j)
        # which side the leaf goes to when it opens: the perpendicular component
        # of the displacement of its free end under the turn, as a unit vector
        end = np.array([[x0 + ax * (W - 1), y0 + ay * (W - 1), z0]])
        moved = mech.quarter_turn(end, swing, pivot2)[0] - end[0]
        perp = (-ay, ax)
        sg = 1.0 if (moved[0] * perp[0] + moved[1] * perp[1]) >= 0 else -1.0
        rec["swing_n"] = [sg * perp[0], sg * perp[1]]
        self._build_links(rec, x0, y0, ax, ay, W, H, z0, d, mat, frame_mat,
                          latch, deadbolt, lock, stud, frame_e, leaf_e)
        if bar:
            self._seat_bar(rec, x0, y0, ax, ay, W, H, z0, bar)
        tabs[mid] = rec
        self.things[leaf_e]["mech"] = mid
        self.things[frame_e]["mech"] = mid
        self._mech_watch_update(rec)
        # a world with a door gives its people the rows for it, and plans
        # through it (will.py `mech_menu`, `door_route`; ledger rows of the same names)
        self.mech_menu = True
        self.door_route = True
        return mid

    # geometry of the default load paths -----------------------------------
    def _build_links(self, rec, x0, y0, ax, ay, W, H, z0, d, mat, frame_mat,
                     latch, deadbolt, lock, stud, frame_e, leaf_e):
        L = mech.new_link
        wood = "wood"
        G, DS, LS = 0.45, 0.00483, 0.019                     # #10 screw in pine
        n_hinge = 3
        links = rec["links"]
        for i in range(n_hinge):
            zh = z0 + int(round((0.12 + 0.38 * i) * H))
            hx, hy = x0 - ax, y0 - ay                          # the hinge jamb cell
            hid = "h%d" % i
            pin = L(hid + ".pin", "hinge pin", "shear", "iron", geom={
                "d": 0.0095, "n": 1, "k": 2.0e6}, anchor=[leaf_e])
            lsc = L(hid + ".leaf", "hinge leaf screws", "lateral", wood, geom={
                "G": G, "D": DS, "l": LS, "n": 3, "k": 1.5e6}, anchor=[leaf_e])
            jsc = L(hid + ".jamb", "hinge jamb screws", "lateral", wood, geom={
                "G": G, "D": DS, "l": LS, "n": 3, "k": 1.5e6},
                anchor=[[hx, hy, zh, int(frame_mat)]])
            pull = L(hid + ".pull", "hinge screws pulled", "withdrawal", wood,
                     geom={"G": G, "D": DS, "L": LS, "n": 3, "k": 1.2e6},
                     anchor=[[hx, hy, zh, int(frame_mat)]])
            links += [pin, lsc, jsc, pull]
            rec["hinges"].append({"id": hid, "z": zh,
                                  "path": [pin["id"], lsc["id"], jsc["id"]],
                                  "pull": [pull["id"]]})
            rec["joints"][0]["links"].append(pin["id"])
        # the stop (rebate): the closing-side load path
        kx, ky = x0 + ax * W, y0 + ay * W                        # the latch jamb cell
        zk = z0 + int(round(0.47 * H))
        sb = L("stop.bear", "leaf edge on the stop", "bearing", wood, "across",
               geom={"A": 0.0045, "k": 4.0e6}, anchor=[leaf_e])
        sl = L("stop.lip", "stop lip shear-off", "bearing", wood, "across",
               geom={"cap_N": 12000.0, "k": 6.0e6},
               anchor=[[kx, ky, zk, int(frame_mat)]])
        ss = L("stop.split", "jamb split behind the stop", "split", wood, "across",
               geom={"l_s": 0.08, "t_e": 0.02, "k": 4.0e6},
               anchor=[[kx, ky, zk, int(frame_mat)]])
        links += [sb, sl, ss]
        rec["stop"] = {"links": [sb["id"], sl["id"], ss["id"]]}
        if latch:
            self._add_latch(rec, x0, y0, ax, ay, W, z0, zk, kx, ky, frame_mat,
                            stud, leaf_e, lock)
        if deadbolt:
            self._add_deadbolt(rec, x0, y0, ax, ay, W, H, z0, kx, ky, frame_mat,
                               lock, leaf_e)

    def _add_latch(self, rec, x0, y0, ax, ay, W, z0, zk, kx, ky, frame_mat,
                   stud, leaf_e, lock):
        L = mech.new_link
        wood = "wood"
        G, DS, LS = 0.45, 0.00483, 0.019
        te = 0.030 if stud else 0.012
        keeper = [[kx, ky, zk, int(frame_mat)]]
        links = [
            L("latch.bolt", "latch bolt", "shear", "iron", geom={
                "d": 0.012, "n": 1, "k": _K_LATCH["bolt"]}, anchor=[leaf_e]),
            L("latch.keeper", "keeper plate", "bearing", "iron", geom={
                "cap_N": 8000.0, "k": _K_LATCH["keeper"]}, anchor=keeper),
            L("latch.strike", "strike screws", "lateral", wood, geom={
                "G": G, "D": DS, "l": LS, "n": 2, "k": _K_LATCH["strike"]},
                anchor=keeper),
            L("latch.split", "jamb split behind the keeper", "split", wood,
              "across", geom={"l_s": 0.08, "t_e": te, "k": _K_LATCH["split"]},
              anchor=keeper),
            L("latch.case", "case screws", "lateral", wood, geom={
                "G": G, "D": DS, "l": LS, "n": 4, "k": _K_LATCH["case"]},
                anchor=[leaf_e]),
            L("latch.bore", "leaf edge bore", "split", wood, "across", geom={
                "l_s": 0.10, "t_e": 0.02, "k": _K_LATCH["bore"]}, anchor=[leaf_e]),
        ]
        rec["links"] += links
        sp = mech.new_spring("latch.spring", "coil", "steel",
                             {"d": 0.001, "D": 0.008, "n": 8, "G": 79e9})
        rec["springs"].append(sp)
        cat = mech.new_catch(
            "latch", "latchbolt", x_mm=15.0, throw_mm=15.0, engage_mm=10.0,
            spring=sp["id"], bevel_deg=45.0,
            at=[x0 + ax * (W - 1), y0 + ay * (W - 1), zk], side="open",
            path=[lk["id"] for lk in links],
            held_by=None if lock is None else lock.get("id", "lock"))
        cat["keeper"] = [kx, ky, zk]
        rec["catches"].append(cat)
        if lock is not None:
            lock.setdefault("thrown", True)
            rec["locks"].append(lock)

    def _add_deadbolt(self, rec, x0, y0, ax, ay, W, H, z0, kx, ky, frame_mat,
                      lock, leaf_e):
        L = mech.new_link
        wood = "wood"
        G, DS, LS = 0.45, 0.00483, 0.019
        zb = z0 + int(round(0.62 * H))
        keeper = [[kx, ky, zb, int(frame_mat)]]
        links = [
            L("dead.shear", "deadbolt", "shear", "iron", geom={
                "d": 0.019, "n": 1, "k": _K_DEAD["shear"]}, anchor=[leaf_e]),
            L("dead.keeper", "deadbolt keeper", "bearing", "iron", geom={
                "cap_N": 20000.0, "k": _K_DEAD["keeper"]}, anchor=keeper),
            L("dead.strike", "deadbolt strike screws (#8)", "lateral", wood, geom={
                "G": G, "D": 0.0042, "l": 0.075, "n": 2, "k": _K_DEAD["strike"]},
                anchor=keeper),
            L("dead.split", "jamb split behind the deadbolt", "split", wood,
              "across", geom={"l_s": 0.15, "t_e": 0.03, "k": _K_DEAD["split"]},
              anchor=keeper),
        ]
        rec["links"] += links
        cat = mech.new_catch(
            "dead", "deadbolt", x_mm=25.0, throw_mm=25.0, engage_mm=12.0,
            at=[x0 + ax * (W - 1), y0 + ay * (W - 1), zb], side="open",
            path=[lk["id"] for lk in links],
            held_by=None if lock is None else lock.get("id", "lock"))
        cat["keeper"] = [kx, ky, zb]
        rec["catches"].append(cat)
        if lock is not None and lock not in rec["locks"]:
            lock.setdefault("thrown", True)
            rec["locks"].append(lock)

    # the bar: lattice truth ----------------------------------------------
    def _seat_bar(self, rec, x0, y0, ax, ay, W, H, z0, bar):
        """Two cleats on the frame, on the side the leaf swings to, and a bar
        thing lying in the brackets above them. Seated means the bar thing's
        cells ARE the two bracket cells (an eid test), nothing else."""
        nx_, ny_ = (int(round(v)) for v in rec["swing_n"])
        zb = z0 + int(round(0.55 * H))
        cells = []
        for j in range(-1, W + 1):
            cells.append((x0 + ax * j + nx_, y0 + ay * j + ny_, zb))
        cleat = []
        for j in (-1, W):
            cleat.append((x0 + ax * j + nx_, y0 + ay * j + ny_, zb - 1))
        old = self._stamping
        self._stamping = rec["frame"]
        for c in cleat:
            self.fill(c[0], c[0] + 1, c[1], c[1] + 1, c[2], c[2] + 1, WOOD)
        be = self.register("bar", "bar")
        self._stamping = be
        for c in cells:
            self.fill(c[0], c[0] + 1, c[1], c[1] + 1, c[2], c[2] + 1, WOOD)
        self._stamping = old
        brackets = [list(cells[0]), list(cells[-1])]
        links = [
            mech.new_link("bar.beam", "bar bending", "bend", "wood", geom={
                "b": 0.038, "d": 0.089, "span": 1.2, "k": 6.0e5}, anchor=[be]),
            mech.new_link("bar.cleat", "cleat screws", "lateral", "wood", geom={
                "G": 0.45, "D": 0.00483, "l": 0.019, "n": 8, "k": 2.0e6},
                anchor=[[c[0], c[1], c[2], int(WOOD)] for c in cleat]),
            mech.new_link("bar.bearing", "leaf bearing on the bar", "bearing",
                          "wood", "across", geom={"A": 0.0045 * 4, "k": 4.0e6},
                          anchor=[rec["leaf"]]),
        ]
        rec["links"] += links
        rec["bar"] = {"eid": be, "brackets": brackets, "cleats": cleat,
                      "path": [lk["id"] for lk in links], "z": zb,
                      "n": len(cells), "home": [list(c) for c in cells]}
        self.things[be]["mech"] = rec["mid"]

    def bar_seated(self, rec):
        """True while the bar thing's cells lie in BOTH bracket cells and the
        bar is whole (a bar split in its middle holds nothing)."""
        b = rec.get("bar")
        if not b:
            return False
        if not all(int(self.eid[tuple(c)]) == b["eid"] and self.mat[tuple(c)] != AIR
                   and not self.rubble[tuple(c)] for c in b["brackets"]):
            return False
        cells = self._cells_of(b["eid"])
        return cells is not None and len(cells) >= b["n"]

    def lift_bar(self, mid):
        """Take the bar out of its brackets (the 'take' act, with nobody's hand
        drawn): the bar's cells move, matter and all, to the floor on the other
        side of the wall, out of the way of the swing. Returns the new cells."""
        rec = self.mechs[mid]
        b = rec["bar"]
        cells = self._cells_of(b["eid"])
        n = rec["swing_n"]
        dst = cells.copy()
        dst[:, 0] -= int(round(2 * n[0]))
        dst[:, 1] -= int(round(2 * n[1]))
        dst[:, 2] = rec["z0"] + (dst[:, 2] - cells[:, 2].min())
        src_set = {tuple(c) for c in cells.tolist()}
        dst_set = {tuple(c) for c in dst.tolist()}
        ent = sorted(dst_set - src_set)
        vac = sorted(src_set - dst_set)
        body = self._lift(tuple(cells.T))
        shoved = self._lift(tuple(np.array(ent, np.int64).T)) if ent else None
        self._put(tuple(dst.T), body)
        if shoved is not None:
            self._put(tuple(np.array(vac, np.int64).T), shoved)
        self._torque_solid = None
        return dst

    def seat_bar(self, mid):
        """Put the bar back in its brackets (the inverse of `lift_bar`): the
        bar's cells go to their home cells, matter and all, if those cells hold
        only air (the swing side of the wall must be clear: a leaf standing
        open there, or a body, refuses it). Returns True when it is seated."""
        rec = self.mechs[mid]
        b = rec["bar"]
        if not b or self.bar_seated(rec):
            return bool(b)
        cells = self._cells_of(b["eid"])
        home = np.array(b["home"], np.int64)
        if cells is None or len(cells) != len(home):
            return False
        order = np.lexsort((cells[:, 2], cells[:, 1], cells[:, 0]))
        src = cells[order]
        dst = home[np.lexsort((home[:, 2], home[:, 1], home[:, 0]))]
        src_set = {tuple(c) for c in src.tolist()}
        ent = [tuple(c) for c in dst.tolist() if tuple(c) not in src_set]
        for c in ent:
            if self.mat[c] != AIR or self.rubble[c] or self.moving[c]:
                return False
        body = self._lift(tuple(src.T))
        self._put(tuple(dst.T), body)
        self._torque_solid = None
        return True

    def add_gravity_latch(self, mid, blocked_by, push_Nm=None, z=None):
        """A catch held up by matter (an ice block, a wax plug): engaged while
        every cell in `blocked_by` holds solid. When it melts the latch drops,
        with no melt code of its own: the mechanism sits on a small WATCH list
        and re-reads those cells each tick. `push_Nm`, if given, is a spring
        pushing the leaf to open (a stored release: mech_in when it lets go)."""
        rec = self.mechs[mid]
        cat = mech.new_catch(
            "gravity", "sear", x_mm=15.0, throw_mm=15.0, engage_mm=10.0,
            at=None, side="open", path=[],
            blocked_by=[list(map(int, c)) for c in blocked_by])
        rec["catches"].append(cat)
        if push_Nm:
            rec["pushes"].append({"pid": None, "F": float(push_Nm), "arm": 1.0,
                                  "sign": rec["s_open"], "spring": True})
            self._mech_active.add(mid)
        self._mech_watch_update(rec)
        return cat

    def _mech_watch_update(self, rec):
        self._mech_tables()
        if any(c.get("blocked_by") for c in rec["catches"]):
            self._mech_watch.add(rec["mid"])
        else:
            self._mech_watch.discard(rec["mid"])

    # ── what holds ──────────────────────────────────────────────────────────
    def _held_by_matter(self, cat):
        """Is a matter-held catch still held up? Every blocking cell solid."""
        return all(self.mat[tuple(c)] != AIR and self.smass[tuple(c)] > 0.0
                   for c in cat["blocked_by"])

    def _catch_holds(self, rec, cat):
        """Does this catch hold the leaf shut right now? It must be thrown, its
        path unbroken (every link standing), and, if matter holds it, the
        matter there."""
        if cat.get("blocked_by") is not None:
            return self._held_by_matter(cat) and cat["x_mm"] >= cat["engage_mm"]
        if cat["x_mm"] < cat["engage_mm"]:
            return False
        return not any(mech.link_failed(self._mlink(rec, lid)) for lid in cat["path"])

    def holds_open_side(self, rec):
        """What stops the leaf swinging the way it opens: any holding catch or a
        seated bar. Returns a list of names (empty = nothing holds it)."""
        out = [c["id"] for c in rec["catches"] if self._catch_holds(rec, c)]
        if rec.get("bar") and self.bar_seated(rec):
            out.append("bar")
        return out

    def _stop_holds(self, rec):
        return not any(mech.link_failed(self._mlink(rec, lid))
                       for lid in rec["stop"]["links"])

    def is_open(self, mid):
        j = self.mechs[mid]["joints"][0]
        return abs(j.get("theta", 0.0)) > 1e-9

    # ── geometry for the laws ───────────────────────────────────────────────
    def _plan_dist_m(self, rec, cell):
        px2, py2 = rec["pivot2"]
        dx = (2 * cell[0] - px2) * 0.5 * self._d()
        dy = (2 * cell[1] - py2) * 0.5 * self._d()
        return math.hypot(dx, dy)

    def _leaf_I(self, rec, cells):
        m = self._mass_kg(cells)
        return mech.leaf_inertia(cells, m, rec["pivot2"], self._d()), float(m.sum())

    def _hinge_friction(self, rec, mass_kg):
        j = rec["joints"][0]
        return j["fric_Nm_per_N"] * mass_kg * GRAV + float(j.get("seize_Nm", 0.0))

    # ── plate statics ───────────────────────────────────────────────────────
    def _supports(self, rec, side):
        """[(x, y, k, kind, ref)] for the supports a push on `side` meets:
        the hinges always; the engaged catches on the opening side; the stop
        on the closing side. x is metres from the hinge axis along the leaf, y
        metres above the leaf bottom."""
        d = self._d()
        out = []
        for h in rec["hinges"]:
            links = [self._mlink(rec, i) for i in h["path"]]
            if any(mech.link_failed(lk) for lk in links):
                continue
            out.append((0.0, (h["z"] - rec["z0"] + 0.5) * d,
                        mech.path_stiffness(links), "hinge", h))
        if side == "open":
            for c in rec["catches"]:
                if not c["path"] or not self._catch_holds(rec, c):
                    continue
                links = [self._mlink(rec, i) for i in c["path"]]
                x = self._plan_dist_m(rec, c["at"])
                out.append((x, (c["at"][2] - rec["z0"] + 0.5) * d,
                            mech.path_stiffness(links), "catch", c))
            b = rec.get("bar")
            if b and self.bar_seated(rec):
                links = [self._mlink(rec, i) for i in b["path"]]
                out.append((rec["W"] * d * 0.5, (b["z"] - rec["z0"] + 0.5) * d,
                            mech.path_stiffness(links), "bar", b))
        else:
            if self._stop_holds(rec):
                links = [self._mlink(rec, i) for i in rec["stop"]["links"]]
                out.append((rec["W"] * d, None, mech.path_stiffness(links),
                            "stop", rec["stop"]))
        return out

    @staticmethod
    def _plate(sup, x0, y0):
        """Shares of a UNIT load at (x0, y0) on each support, and the stiffness
        of the plate at the contact (N/m)."""
        s = [(x, y, k) for x, y, k, _kind, _ref in sup]
        R = mech.plate_reactions(s, [(1.0, x0, y0)])
        A = np.zeros((3, 3))
        for x, y, k in s:
            v = np.array([1.0, x, y])
            A += k * np.outer(v, v)
        c = np.linalg.solve(A, np.array([1.0, x0, y0]))
        w0 = float(c @ np.array([1.0, x0, y0]))
        return R, 1.0 / w0

    def _side_of(self, rec, actor_xy):
        """'open' if a body at actor_xy is on the side the leaf swings AWAY
        from (it pushes the leaf the way it opens), else 'stop' (it is on the
        swing side and pushes the leaf into the stop)."""
        x0 = rec["pivot2"][0] / 2.0
        y0 = rec["pivot2"][1] / 2.0
        n = rec["swing_n"]
        dot = (actor_xy[0] - x0) * n[0] + (actor_xy[1] - y0) * n[1]
        return "stop" if dot > 0 else "open"

    # ── a blow on the leaf ──────────────────────────────────────────────────
    def _strength(self, p):
        """The most force this body puts on a load: lifts, hauls, drags, shoves and
        grips all ask this one question. Gate off: the sheet's or the stock 400 N.
        Gate on (`exertion_from_body`): the body's own muscle (exertion.force_N:
        strength, training, a per person spread, fatigue and injury)."""
        if self.exertion_from_body:
            return _EX.force_N(p)
        return p.get("strength_N", BODY["strength_N"])

    def kick_energy(self, p):
        """(m_s, v_s) of this body's kick. Gate off: 5.6 kg at 6 m/s (100 J)
        scaled by strength / 400 (u). Gate on (`exertion_from_body`): the
        person's own kick from exertion.strike (about 115 J for an ordinary
        adult, a real spread, a trained tail)."""
        if self.exertion_from_body:
            k = _EX.strike(p, "kick")
            return k["m_kg"], k["v_ms"]
        s = float(BODY.get("strength_N", 400.0)) / 400.0
        return KICK_M_KG * s, KICK_V_MPS

    def _pay_strike(self, p, J, tag, kind=None):
        """Pay a strike's joules from the actor's own books. EVERY strike is paid
        for, or it is weaker for it:
          - a body with a heat record pays from its stores at the efficiency of
            muscle (energetics.spend_J via note_work): the stores give J / eta, the
            muscle keeps the waste heat, and the work_out credit is handed back
            (release_work) because `deposit` credits the joules once when the
            blow lands. Each strike is its own bill (the tag carries the count of
            this body's strikes), so a second kick in the same tick is
            billed too. A refused bill pays for nothing (0.0), it never gives
            the strike for free.
          - a body with no heat record pays from its limb's reserve above
            critical power (exertion.pay_reserve), which falls with every
            strike and comes back with rest.
        Returns the share that was paid for: 1.0, or less when the stores or the
        reserve could not pay (the blow is then weaker)."""
        if J <= 0.0:
            return 1.0
        kind = kind or (tag if tag in _EX.KINDS else "shove")
        if p.get("heat") is None:
            return _EX.pay_reserve(p, kind, J, self.tick * TICK_S)
        book = p["heat"]["work"]
        book["strikes"] = n = book.get("strikes", 0) + 1       # one bill per strike
        got = self.note_work(p, "%s@%d:%d" % (tag, self.tick, n), J, "swing")
        if got is None:
            return 0.0
        sc = float(got.get("scale", 1.0))
        self.release_work(J * sc)
        return sc

    def kick(self, p, mid, cell=None, side=None):
        """The kick act: a blow of the body's kick energy on the leaf at `cell`
        (default: the leaf near the latch, at lock height)."""
        m_s, v_s = self.kick_energy(p)
        if self.exertion_from_body:
            sc = self._pay_strike(p, 0.5 * m_s * v_s * v_s, "kick")
            if sc <= 1e-9:          # nothing left to pay a kick with
                return {"result": "spent", "failed": [], "swing": False}
            m_s *= sc               # an unpaid part is not delivered (same speed)
        return self.blow(mid, cell=cell, m_s=m_s, v_s=v_s, actor=p, side=side,
                         verb="kicks")

    def shoulder(self, p, mid, cell=None, side=None):
        """The other way to force a door: the body's shove (exertion.strike
        "shove", about 120 J, a real spread) put into the leaf through the same
        blow law as a kick, paid for from the actor's own books the same way
        (`_pay_strike`). The event is 'forces'."""
        k = _EX.strike(p, "shove")
        m_s, v_s = k["m_kg"], k["v_ms"]
        sc = self._pay_strike(p, 0.5 * m_s * v_s * v_s, "shove")
        if sc <= 1e-9:
            return {"result": "spent", "failed": [], "swing": False}
        return self.blow(mid, cell=cell, m_s=m_s * sc, v_s=v_s, actor=p, side=side,
                         verb="forces")

    def _work_bar(self, p, rec):
        """Seat a bar the person holds, or lift a seated one out (the person
        must be within reach of the bar). The bar thing is the matter: nothing
        here says 'barred'; the door is held while the bar's cells are in the
        brackets (bar_seated)."""
        b = rec.get("bar")
        if not b:
            return {"result": "no bar"}
        cells = self._person_cells_of(p)
        bc = self._cells_of(b["eid"])
        if cells is None or bc is None:
            return {"result": "no bar"}
        d = self._d()
        if _gap_cells(bc, cells) * d > REACH_M:
            return {"result": "out of reach"}
        if self.bar_seated(rec):
            self.lift_bar(rec["mid"])
            ev = self.emit("unlocks", actor=p["pid"], target=("thing", rec["leaf"]),
                           result="bar lifted", text="")
            return {"result": "lifted", "event": ev}
        in_hand = bool(p.get("held")) and self._held_eid(p) == b["eid"]
        if not self.seat_bar(rec["mid"]):
            return {"result": "blocked"}
        if in_hand:
            p["held"] = None                  # the bar is in its brackets, not in the hand
        ev = self.emit("locks", actor=p["pid"], target=("thing", rec["leaf"]),
                       result="bar seated", text="")
        return {"result": "seated", "event": ev}

    def blow(self, mid, cell=None, m_s=KICK_M_KG, v_s=KICK_V_MPS, actor=None,
             side=None, e_c=E_C_FOOT, k_c=K_CONTACT, verb="kicks",
             joules=None):
        """One blow on a leaf: the two-body impact against the plate on its
        supports, Miner damage on every link in the path of that direction of
        push, and, if the leaf is no longer held, a swing with the omega the
        impact left it.

        `joules`, if given, sets m_s from v_s (the `_spend` seam hands over the
        energy of whatever struck). Returns a dict: F_peak, failed (link ids),
        side, swing (bool), result, energy books (heat_J, KE_foot, KE_leaf).
        The joules the striker brings are the actor's (debited, work_out) or,
        with none, mech_in."""
        rec = self.mechs[mid]
        d = self._d()
        if joules is not None:
            m_s = 2.0 * float(joules) / (v_s * v_s)
        cells = self._cells_of(rec["leaf"])
        if cells is None:
            return {"result": "no leaf"}
        if cell is None:
            W, ax, ay = rec["W"], rec["along"][0], rec["along"][1]
            x0 = int(round(rec["pivot2"][0] / 2.0 + 0.5))
            y0 = int(round(rec["pivot2"][1] / 2.0 + 0.5))
            cell = (x0 + ax * (W - 2), y0 + ay * (W - 2),
                    rec["z0"] + int(0.45 * rec["H"]))
            cell = self._nearest_leaf_cell(cells, cell)
        cell = tuple(int(v) for v in cell)
        if side is None:
            if actor is not None:
                ac = self._actor_center(actor)
                side = self._side_of(rec, ac)
            else:
                side = "open"
        I, mass = self._leaf_I(rec, cells)
        a = max(self._plan_dist_m(rec, cell), 0.5 * d)
        y_c = (cell[2] - rec["z0"] + 0.5) * d
        sup = self._supports(rec, side)
        if side == "stop":
            sup = [(x, (y if y is not None else y_c), k, kd, rf)
                   for x, y, k, kd, rf in sup]
        out = {"side": side, "cell": cell, "failed": [], "swing": False,
               "result": "held"}
        ev = self.emit(verb, actor=None if actor is None else actor["pid"],
                       target=("thing", rec["leaf"]), place=cell,
                       result=side, db=None,
                       text="", data=(("joules", round(0.5 * m_s * v_s * v_s, 3)),))
        out["event"] = ev
        # the sound: through the real law, on the leaf's material
        self._noise(cell, 0.5 * m_s * v_s * v_s, [WOOD], pid=None if actor is None
                    else actor["pid"], ev=ev.id)
        if not sup:
            out["result"] = "no support"
            self._book_blow(rec, cell, 0.5 * m_s * v_s * v_s, 0.0, 0.0, actor, out)
            return out
        try:
            shares, k_eff = self._plate(sup, a, y_c)
        except ValueError:
            out["result"] = "no support"
            self._book_blow(rec, cell, 0.5 * m_s * v_s * v_s, 0.0, 0.0, actor, out)
            return out
        out["shares"] = [float(s) for s in shares]
        imp = mech.two_body_impact(m_s, v_s, I, a, k_c, k_eff, e_c=e_c,
                                   leaf_held=True)
        F = imp["F_peak"]
        out["F_peak"] = F
        flaws = self._flaws(rec)
        failed_all = []
        for s, (_x, _y, _k, kind, ref) in zip(shares, sup):
            R = float(s) * F
            if kind == "hinge":
                ids = ref["pull"] if R < 0.0 else ref["path"]
            elif kind == "catch":
                ids = ref["path"]
            elif kind == "bar":
                ids = ref["path"]
            else:
                ids = ref["links"]
            links = [self._mlink(rec, i) for i in ids]
            failed_all += mech.blow_path(links, abs(R), flaws=flaws,
                                         flaw_amp=rec["flaw_amp"])
        out["failed"] = failed_all
        why = ev.id
        with self.because(ev.id):
            for lid in failed_all:
                bk = self._link_broken(rec, lid, "kick", F)
                if why == ev.id:
                    why = bk.id if hasattr(bk, "id") else why
        # does the leaf swing? not while anything on this side holds it
        released = False
        if side == "open":
            released = not self.holds_open_side(rec)
        else:
            released = not self._stop_holds(rec)
        E0 = 0.5 * m_s * v_s * v_s
        if released and failed_all:
            fl = failed_all[0]
            lk = self._mlink(rec, fl)
            capf = min(mech.link_capacity(lk, flaw01=flaws.get(fl, 0.5),
                                          flaw_amp=rec["flaw_amp"]), 0.999 * F)
            imp = mech.two_body_impact(m_s, v_s, I, a, k_c, k_eff, e_c=e_c,
                                       F_cap=capf, k_link=mech.link_stiffness(dict(lk, D=0.0)),
                                       ductility=mech.material_props(
                                           lk["mat"], lk["grain"])["ductility"],
                                       leaf_held=False)
            # THE FOLLOW-THROUGH. The link absorbed what it could and the foot
            # is still driving into a leaf that is now free: a second two-body
            # collision between them (the same restitution) hands the leaf its
            # swing. Angular momentum about the axis stays exact; the energy
            # the second contact loses is heat.
            J2 = imp["mu"] * (1.0 + e_c) * max(imp["v_rel_after"], 0.0)
            om = imp["omega"] + J2 * a / I
            v_f = imp["v_foot"] - J2 / m_s
            ke_l, ke_f = 0.5 * I * om * om, 0.5 * m_s * v_f * v_f
            imp.update(J2=J2, omega=om, v_foot=v_f, KE_leaf=ke_l,
                       KE_leaf_kept=ke_l, KE_foot=ke_f, heat_J=E0 - ke_l - ke_f,
                       L_after=m_s * a * v_f + I * om)
            out["impact"] = imp
            direction = rec["s_open"] if side == "open" else -rec["s_open"]
            started = self._start_swing(rec, direction, omega0=imp["omega"],
                                        actor=actor, why=why, ke0=imp["KE_leaf"],
                                        kick_actor=actor)
            out["swing"] = started
            out["result"] = "swing" if started else "blocked"
            self._book_blow(rec, cell, E0, imp["KE_foot"],
                            imp["KE_leaf"] if started else 0.0, actor, out,
                            leaf_dissipated=0.0 if started else imp["KE_leaf"])
        else:
            out["impact"] = imp
            out["result"] = "held" if not released else "cracked"
            self._book_blow(rec, cell, E0, imp["KE_foot"], 0.0, actor, out,
                            leaf_dissipated=imp["KE_leaf"])
        return out

    def _book_blow(self, rec, cell, E0, ke_foot, ke_leaf_pending, actor, out,
                   leaf_dissipated=0.0):
        """Close the striker's joules: contact deformation heat now; the foot's
        rebound and a held leaf's rebound dissipated at once; a leaf set
        swinging keeps its kinetic energy pending (off the ledger, in the
        swing) until the swing ends. Everything is the actor's debit (or
        mech_in with none)."""
        heat = E0 - ke_foot - ke_leaf_pending - leaf_dissipated
        c = self._contact_cells(rec, cell)
        if heat > 0.0:
            self.deposit(c, heat, actor=actor)
        if leaf_dissipated > 0.0:
            self.deposit(self._hinge_cells(rec), leaf_dissipated, actor=actor)
        if ke_foot > 0.0:
            fc = self._foot_cells(actor) if actor is not None else c
            self.deposit(fc if fc is not None else c, ke_foot, actor=actor)
        out["heat_J"] = heat
        out["KE_foot"] = ke_foot
        out["KE_leaf_pending"] = ke_leaf_pending

    def _nearest_leaf_cell(self, cells, cell):
        dd = np.abs(cells - np.array(cell)).sum(axis=1)
        return tuple(int(v) for v in cells[int(dd.argmin())])

    def _contact_cells(self, rec, cell):
        cells = self._cells_of(rec["leaf"])
        if cells is None:
            return np.array([cell], np.int64)
        dd = np.abs(cells - np.array(cell)).sum(axis=1)
        order = np.argsort(dd, kind="stable")[:_HEAT_MAX_CELLS]
        return cells[order]

    def _hinge_cells(self, rec):
        px2, py2 = rec["pivot2"]
        out = []
        for h in rec["hinges"]:
            c = (int(math.floor(px2 / 2.0)), int(math.floor(py2 / 2.0)), h["z"])
            out.append(c)
        return np.array(out, np.int64)

    def _stop_cells(self, rec):
        k = [c for c in rec["catches"] if c.get("keeper")]
        if k:
            return np.array([k[0]["keeper"]], np.int64)
        return self._hinge_cells(rec)

    def _actor_center(self, p):
        cells = self._person_cells_of(p)
        if cells is None:
            return (float(p["anchor"][0]), float(p["anchor"][1]))
        return (float(cells[:, 0].mean()), float(cells[:, 1].mean()))

    def _person_cells_of(self, p):
        """The body's flesh NOW (a fresh claim: an act is asked between ticks,
        when a body may have been moved since the tick's claim)."""
        self._claim_bodies()
        comp, sl = p.get("_claim_mask"), p.get("_claim_sl")
        if comp is None or not comp.any():
            return None
        return self._claim_cells(p, comp, sl)

    def _foot_cells(self, p):
        cells = self._person_cells_of(p)
        if cells is None:
            return None
        zlo = int(cells[:, 2].min())
        return cells[cells[:, 2] <= zlo + 1]

    def _link_broken(self, rec, lid, cause, F=None):
        """A link failed: tell the world, and if it is wood, SPLIT it for real.
        The keeper wood is shattered through `_shatter`, so splinters, noise
        and rubble come from the existing law."""
        lk = self._mlink(rec, lid)
        lk["D"] = 1.0
        bk = self._ev("breaks", target=("thing", rec["leaf"] if lk["anchor"]
                                   and isinstance(lk["anchor"][0], int)
                                   else rec["frame"]),
                 place=self._link_place(lk), result=lk["name"], why=cause,
                 text="", data=(("link", lid),))
        if lk["id"] == "bar.beam":
            # the bar splinters (a 38 x 89 mm pine beam beaten past its
            # bending strength): every cell between the brackets goes through
            # `_shatter`, the two ends stay in their cleats
            cells = self._cells_of(rec["bar"]["eid"])
            ends = {tuple(c) for c in rec["bar"]["brackets"]}
            if cells is not None:
                for x, y, z in cells.tolist():
                    if (x, y, z) not in ends and self.mat[x, y, z] != AIR:
                        self._shatter(int(x), int(y), int(z), over=2.0)
        if lk["mode"] == "split" and lk["anchor"] and not isinstance(lk["anchor"][0], int):
            x, y, z, m = lk["anchor"][0]
            if self.mat[x, y, z] == m:
                cap0 = max(mech.link_capacity(lk, flaw_amp=0.0), 1.0)
                self._shatter(int(x), int(y), int(z), over=max(1.0, (F or cap0) / cap0))
        return bk

    def _link_place(self, lk):
        a = lk["anchor"]
        if a and not isinstance(a[0], int):
            return tuple(int(v) for v in a[0][:3])
        return None

    # ── swings ──────────────────────────────────────────────────────────────
    def _swing_sector(self, rec, cells, direction, theta):
        """The cells a leaf turning `theta` radians in `direction` sweeps, as an
        (n, 3) array: the plan cells the leaf's slab passes over (its cells
        sub-sampled 3 x 3 and turned in 1 degree steps, so nothing between
        two steps is missed) over the leaf's height. Only what the leaf really
        crosses: no cells on the far side of the hinge."""
        px2, py2 = rec["pivot2"]
        px, py = px2 / 2.0, py2 / 2.0
        c = np.asarray(cells, np.int64)
        xy = np.unique(c[:, :2], axis=0).astype(float)
        sub = np.array([-1.0, 0.0, 1.0]) / 3.0
        sx, sy = np.meshgrid(sub, sub, indexing="ij")
        pts = (xy[:, None, :] + np.stack([sx.ravel(), sy.ravel()], 1)[None]) \
            .reshape(-1, 2)
        n = max(1, int(math.ceil(math.degrees(abs(theta)))))
        ang = np.linspace(0.0, abs(theta), n + 1) * direction
        cs, sn = np.cos(ang)[:, None], np.sin(ang)[:, None]
        dx, dy = pts[None, :, 0] - px, pts[None, :, 1] - py
        xs = np.rint(px + dx * cs - dy * sn).astype(np.int64).ravel()
        ys = np.rint(py + dx * sn + dy * cs).astype(np.int64).ravel()
        cols = np.unique(np.stack([xs, ys], 1), axis=0)
        nx, ny, nz = self.shape
        cols = cols[(cols[:, 0] >= 0) & (cols[:, 0] < nx)
                    & (cols[:, 1] >= 0) & (cols[:, 1] < ny)]
        zs = np.arange(int(c[:, 2].min()), int(c[:, 2].max()) + 1)
        zs = zs[(zs >= 0) & (zs < nz)]
        out = np.empty((len(cols) * len(zs), 3), np.int64)
        out[:, 0] = np.repeat(cols[:, 0], len(zs))
        out[:, 1] = np.repeat(cols[:, 1], len(zs))
        out[:, 2] = np.tile(zs, len(cols))
        return out

    def _sector_blockers(self, rec, cells, sector, normal=True):
        """The sector cells that are not the leaf and not clear air.

        `normal` is True while the leaf turns between its closed and its open
        position (the only way a hung door turns). Then the FRAME is not a
        blocker. A leaf of W cells turning about the corner of its hinge cell
        has a far corner at sqrt(W^2 + 1) cells, which is more than W: the
        voxel mosaic of a 1 or 2 cell leaf pokes into its own latch jamb by
        a rounding corner (a real door has a bevel and a clearance gap there).
        Before this, leaves of 1 or 2 cells could not swing. Turning the
        wrong way (into the frame, with no stop) still meets the frame."""
        if not len(sector):
            return sector
        idx = tuple(sector.T)
        leaf = self.eid[idx] == rec["leaf"]
        if normal:
            leaf = leaf | (self.eid[idx] == rec["frame"])
        # loose splinters (rubble) are pushed before the leaf, and particles the
        # solver is carrying through (`moving`) are in flight: neither blocks
        busy = (((self.mat[idx] != AIR) & ~self.rubble[idx]) \
                | (self.fvol[idx] > 1.0)) & ~self.moving[idx]
        return sector[busy & ~leaf]

    def _start_swing(self, rec, direction, omega0=0.0, actor=None, why=None,
                     ke0=0.0, kick_actor=None):
        """Begin a swing. Refused (and 'jams' emitted) when a body or object
        stands in the sector or the door is already at that limit."""
        cells = self._cells_of(rec["leaf"])
        j = rec["joints"][0]
        s_open = rec["s_open"]
        theta_now = j.get("theta", 0.0)
        target = theta_now + direction * math.pi / 2.0
        ok = abs(target) < 1e-9 or abs(target - s_open * math.pi / 2.0) < 1e-9 \
            or (abs(target + s_open * math.pi / 2.0) < 1e-9 and not self._stop_holds(rec))
        if cells is None or not ok:
            if not rec.get("jammed"):
                rec["jammed"] = True
                self._ev("jams", target=("thing", rec["leaf"]),
                         result="at its stop", cause=why, text="")
            return False
        sector = self._swing_sector(rec, cells, direction, math.pi / 2.0)
        normal = abs(target) < 1e-9 or abs(target - s_open * math.pi / 2.0) < 1e-9
        block = self._sector_blockers(rec, cells, sector, normal)
        if len(block):
            c = tuple(int(v) for v in block[0])
            if not rec.get("jammed"):
                rec["jammed"] = True
                if actor is not None:
                    self._learn_door(actor["pid"], rec, "blocked", None)
                self._ev("jams", actor=None if actor is None else actor["pid"],
                         target=("thing", rec["leaf"]), place=c, result="blocked",
                         cause=why, text="", data=(("by", int(self.eid[c])),))
            if ke0 > 0.0:                      # a kicked leaf's energy goes as heat
                self.deposit(block[:_HEAT_MAX_CELLS], ke0, actor=kick_actor)
            return False
        rec["jammed"] = False
        I, mass = self._leaf_I(rec, cells)
        rec["swing"] = {"t0": self.tick, "theta": 0.0, "omega": float(omega0),
                        "dir": int(direction), "why": why, "I": I, "mass": mass,
                        "tau_f": self._hinge_friction(rec, mass),
                        "ke0": float(ke0), "elapsed": 0.0, "work": {},
                        "kick_actor": None if kick_actor is None else kick_actor["pid"],
                        "pusher": self._top_pusher(rec),
                        "w_total": 0.0, "stalled": False, "normal": normal}
        j["omega"] = float(omega0)
        self._mech_tables()
        self._mech_active.add(rec["mid"])
        return True

    def _push_torque(self, rec, direction):
        """Net pushing torque along `direction` (+ helps the swing)."""
        return sum(p["sign"] * direction * p["F"] * p["arm"] for p in rec["pushes"])

    def _advance_swing(self, rec):
        """One tick of a swing: constant torque, so the motion is integrated in
        closed form (exact), the crossing time interpolated inside the tick.
        Returns 'done', 'aborted', 'stalled' or None (still going)."""
        sw = rec["swing"]
        if sw.get("waiting"):
            return "done"             # the swing is over; the redraw is waiting
        dirn, I, tau_f = sw["dir"], sw["I"], sw["tau_f"]
        tau = self._push_torque(rec, dirn)
        th, om = sw["theta"], sw["omega"]
        dt = TICK_S
        end = math.pi / 2.0
        t_used = 0.0
        # work done by the pushers over this tick, on the angle they turn
        th0 = th
        left = dt
        while left > 1e-15:
            if om > 1e-12:
                alpha = (tau - tau_f) / I
            elif tau > tau_f:
                alpha = (tau - tau_f) / I
                om = 0.0
            else:
                sw["stalled"] = True
                om = 0.0
                break
            t_stop = left
            if alpha < 0.0 and om > 0.0:
                t_stop = min(left, om / (-alpha))
            th_try = th + om * t_stop + 0.5 * alpha * t_stop * t_stop
            if th_try >= end:
                disc = om * om + 2.0 * alpha * (end - th)
                if abs(alpha) < 1e-15:
                    t_hit = (end - th) / om
                else:
                    t_hit = (-om + math.sqrt(max(disc, 0.0))) / alpha
                t_hit = min(max(t_hit, 0.0), t_stop)
                om_hit = om + alpha * t_hit
                t_used += t_hit
                sw["omega"], sw["theta"] = om_hit, end
                sw["elapsed"] += t_hit
                sw["t_hit"] = sw["elapsed"]
                self._swing_work(rec, th0, end)
                return "done"
            th = th_try
            om = om + alpha * t_stop
            left -= t_stop
            t_used += t_stop
            if alpha < 0.0 and om <= 1e-12:
                om = 0.0
                if not (tau > tau_f):
                    sw["stalled"] = True
                    break
        sw["theta"], sw["omega"] = th, om
        sw["elapsed"] += t_used
        rec["joints"][0]["omega"] = om
        self._swing_work(rec, th0, th)
        if sw["stalled"] and om <= 1e-12:
            return "stalled"
        # a body or object entered the sector swept so far
        cells = self._cells_of(rec["leaf"])
        if cells is not None:
            sector = self._swing_sector(rec, cells, dirn, th)
            block = self._sector_blockers(rec, cells, sector, sw.get("normal", True))
            if len(block):
                sw["blocker"] = tuple(int(v) for v in block[0])
                sw["blockers"] = block
                return "aborted"
        return None

    def _swing_work(self, rec, th0, th1):
        sw = rec["swing"]
        dth = th1 - th0
        for p in rec["pushes"]:
            key = p.get("pid")
            w = p["sign"] * sw["dir"] * p["F"] * p["arm"] * dth
            sw["work"][key] = sw["work"].get(key, 0.0) + w
        sw["w_total"] += sum(p["sign"] * sw["dir"] * p["F"] * p["arm"] * dth
                             for p in rec["pushes"])

    def _swing_books(self, rec, where_cells, ke_end_cells, ke_end):
        """Close the books of a swing that ends (at the stop, aborted, or
        stalled): all of its pending energy is heat NOW. Friction heat goes to
        the hinge cells, the kinetic energy at the end to `ke_end_cells`.
        Pushers who are people are debited (work_out); a spring or other
        no-actor release is mech_in (the ONE-FLOW rule)."""
        sw = rec["swing"]
        j = rec["joints"][0]
        th = sw["theta"]
        fric = sw["tau_f"] * th
        total = sw["ke0"] + sw["w_total"]           # everything put in
        ke_end = max(min(ke_end, max(total - fric, 0.0)), 0.0)
        fric = max(total - ke_end, 0.0)
        if fric > 0.0:
            self._book_split(rec, sw, self._hinge_cells(rec), fric, total)
        if ke_end > 0.0:
            self._book_split(rec, sw, ke_end_cells, ke_end, total)

    def _book_split(self, rec, sw, cells, J, total):
        """Book J joules of the swing's pending energy, apportioned among the
        sources in proportion to what each put in: the kicker's pending KE, and
        each pusher's work (a person, or None = a spring)."""
        if J <= 0.0 or total <= 0.0:
            return
        parts = []
        if sw["ke0"] > 0.0:
            parts.append((sw["kick_actor"], sw["ke0"]))
        for key, w in sw["work"].items():
            if w > 0.0:
                parts.append((key, w))
        pos = sum(w for _k, w in parts) or 1.0
        for key, w in parts:
            actor = None
            if key is not None:
                actor = next((q for q in self.persons if q["pid"] == key), None)
            self.deposit(cells, J * (w / pos), actor=actor)

    def _finish_swing(self, rec, outcome):
        """A swing ended: redraw the leaf (exact exchange), or leave it as it
        was, and book every joule."""
        sw = rec["swing"]
        j = rec["joints"][0]
        cells = self._cells_of(rec["leaf"])
        d = self._d()
        why = sw["why"]
        s_open = rec["s_open"]
        dirn = sw["dir"]
        ke_end = 0.5 * sw["I"] * sw["omega"] ** 2
        if outcome == "done":
            ok = self._turn_leaf(rec, cells, dirn)
            if ok == "wait":
                sw["waiting"] = sw.get("waiting", 0) + 1
                if sw["waiting"] <= WAIT_TICKS:
                    return                 # at the stop, redrawn when the solver lets go
                ok = False
            if not ok:
                outcome = "aborted"
                sw["blocker"] = None
        if outcome == "done":
            j["theta"] = float(j.get("theta", 0.0) + dirn * math.pi / 2.0)
            if abs(j["theta"]) < 1e-9:
                j["theta"] = 0.0
            # the stop takes the kinetic energy: a blow, a slam, a sound
            stop_c = self._stop_end_cells(rec, dirn)
            self._swing_books(rec, None, stop_c, ke_end)
            at = tuple(int(v) for v in stop_c[0])
            if ke_end > 0.0:
                self._noise(at, ke_end, [WOOD])
            opening = abs(j["theta"]) > 1e-9
            for c in rec["catches"]:
                if c.get("blocked_by") is None and c["kind"] == "latchbolt":
                    c["x_mm"] = c["throw_mm"]        # handle released: bolt out
            who = self._swing_actor(sw)
            if who is not None and self._door_memory(who, rec):
                self._learn_door(who, rec, "", None)       # it gave: not stuck now
            self._ev("opens" if opening else "closes", actor=who,
                     target=("thing", rec["leaf"]),
                     place=at, result="open" if opening else "shut", cause=why,
                     dur=sw["elapsed"], text="",
                     data=(("t_s", round(sw.get("t_hit", sw["elapsed"]), 6)),
                           ("omega", round(sw["omega"], 6))))
        else:
            block = sw.get("blockers")
            tgt = block[:_HEAT_MAX_CELLS] if block is not None and len(block) \
                else self._hinge_cells(rec)
            if outcome == "aborted" and block is not None and len(block):
                self._blow_obstacle(rec, tuple(int(v) for v in block[0]), ke_end, d)
            self._swing_books(rec, None, tgt, ke_end)
            who = self._swing_actor(sw)
            if who is not None:
                self._learn_door(who, rec, "blocked" if outcome == "aborted"
                                 else "stalled", None)
            self._ev("jams", actor=who,
                     target=("thing", rec["leaf"]),
                     place=sw.get("blocker"), result=("blocked" if outcome
                                                      == "aborted" else "stalled"),
                     cause=why, text="", data=(("ke", round(ke_end, 6)),))
        rec["swing"] = None
        j["omega"] = 0.0
        # the pushers let go of a finished swing (a spring is spent)
        rec["pushes"] = []
        rec["jammed"] = False
        self._mech_active.discard(rec["mid"])

    @staticmethod
    def _top_pusher(rec):
        """The pid of the person whose held push has the most torque on the leaf
        (F x arm), or None (a spring, a kick with nobody pushing)."""
        best, who = 0.0, None
        for q in rec["pushes"]:
            if q.get("pid") is not None and q["F"] * q["arm"] > best:
                best, who = q["F"] * q["arm"], q["pid"]
        return who

    def _swing_actor(self, sw):
        """Who did a swing: the pusher with the most torque when it started,
        else the one who kicked it. This is what lets a mind see that ITS push
        worked (the 'opens' event names the actor)."""
        who = sw.get("pusher")
        return who if who is not None else sw.get("kick_actor")

    def _learn_door(self, pid, rec, why, ev):
        """A person who tried a door and could not get it to give KNOWS that, as
        a belief of its own (beliefs.py `_believe`, how 'did'): the event that
        taught it, and the word the law gave ('locked', 'barred', 'held', 'blocked').
        It is never read for anyone else, and it stays until the person tries
        again or the belief store forgets it (the oldest key goes first at
        BEL_CAP), so it can be wrong once someone unlocks the door."""
        try:
            self._believe(pid, ("jam", ("thing", rec["leaf"])), why,
                          None if ev is None else ev.id, "did")
        except (KeyError, IndexError, TypeError, AttributeError):
            pass                          # a body with no mind keeps no belief

    def _door_memory(self, pid, rec):
        """The word a person remembers of the last time it tried this door and
        failed ('' if it has not, or has since got it open)."""
        try:
            b = self.minds[pid].bel.get(("jam", ("thing", rec["leaf"])))
        except (KeyError, IndexError, TypeError, AttributeError):
            return ""
        return "" if b is None else str(b[0])

    def _stop_end_cells(self, rec, dirn):
        cells = self._cells_of(rec["leaf"])
        if cells is None:
            return self._hinge_cells(rec)
        # the free end of the leaf: its cells farthest from the pivot
        px2, py2 = rec["pivot2"]
        dd = np.hypot(cells[:, 0] - px2 / 2.0, cells[:, 1] - py2 / 2.0)
        order = np.argsort(-dd, kind="stable")[:_HEAT_MAX_CELLS]
        return cells[order]

    def _blow_obstacle(self, rec, cell, joules, d):
        """The leaf's kinetic energy at the moment it meets a body in its way,
        as the existing blow law: a person's wound by the leaf's contact area."""
        if joules <= 0.0:
            return
        q = self._person_at(cell) if self.mat[cell] == FLESH else None
        if q is not None:
            self._blow(q, cell, joules, d * d)

    def _turn_leaf(self, rec, cells, sign):
        """Redraw the leaf a quarter turn on: the leaf cells and the entered
        air cells swap, CARRIED fields and all (nothing is lost). Refused if an
        entered cell is not air, holds a real puddle, is the solver's, or has a
        body in it."""
        dest, entered, vacated = mech.turn_exchange(cells, sign, rec["pivot2"])
        if len(entered):
            ei = tuple(entered.T)
            if self.moving[ei].any():
                return "wait"             # the solver holds a cell: next tick
            if ((self.mat[ei] != AIR) & ~self.rubble[ei]).any() \
                    or (self.fvol[ei] > 1.0).any():
                return False
        body = self._lift(tuple(cells.T))
        shoved = self._lift(tuple(entered.T)) if len(entered) else None
        self._put(tuple(dest.T), body)
        if shoved is not None:
            self._put(tuple(vacated.T), shoved)
        self._torque_solid = None
        self._slack_mat = None
        return True

    # ── the law ─────────────────────────────────────────────────────────────
    def _law_identity(self):
        """Mechanisms first (swings advance; the exchange changes mat; events
        go through _defer so ids are not consumed early), then the rest."""
        if "mechs" in self.__dict__ and self.mechs and (
                self._mech_active or self._mech_watch):
            self._law_mechs()
        super()._law_identity()

    def _law_mechs(self):
        self._mech_phase = True
        try:
            for mid in sorted(self._mech_active | self._mech_watch):
                rec = self.mechs.get(mid)
                if rec is None:
                    continue
                self._watch_catches(rec)
                if rec["swing"] is not None:
                    out = self._advance_swing(rec)
                    if out is not None:
                        self._finish_swing(rec, out)
                elif rec["pushes"]:
                    self._try_push(rec)
        finally:
            self._mech_phase = False

    def _watch_catches(self, rec):
        """A catch held by matter lets go when the matter does."""
        for c in rec["catches"]:
            if c.get("blocked_by") is not None and c["x_mm"] >= c["engage_mm"] \
                    and not self._held_by_matter(c):
                c["x_mm"] = 0.0
                self._ev("unlocks", target=("thing", rec["leaf"]),
                         place=tuple(c["blocked_by"][0]), result="dropped",
                         text="")

    def _try_push(self, rec):
        """Held pushes on a shut leaf: does the net torque beat what holds it?
        A holding catch is tried against the load the push puts on it (plate
        statics); above its capacity it lets go (a link breaks)."""
        j = rec["joints"][0]
        theta = j.get("theta", 0.0)
        s = rec["s_open"]
        tau_open = self._push_torque(rec, s)             # along the opening turn
        if abs(theta) < 1e-9:
            dirn = s if tau_open > 0.0 else (-s if tau_open < 0.0 else 0)
        else:
            dirn = -1 if theta > 0 else 1                 # an open leaf closes
            if self._push_torque(rec, dirn) <= 0.0:
                dirn = 0
        if dirn == 0:
            return
        cells = self._cells_of(rec["leaf"])
        if cells is None:
            return
        tau = self._push_torque(rec, dirn)
        I, mass = self._leaf_I(rec, cells)
        if abs(theta) < 1e-9:
            side = "open" if dirn == s else "stop"
            arm = max(max(p["arm"] for p in rec["pushes"]), 1e-3)
            F = tau / arm
            holders = self.holds_open_side(rec) if side == "open" \
                else ([] if not self._stop_holds(rec) else ["stop"])
            if holders:
                if not self._static_overload(rec, side, F, arm):
                    return
                # the load beat the weakest link: it let go
                top = self._top_pusher(rec)
                if top is not None:
                    self._ev("forces", actor=top, target=("thing", rec["leaf"]),
                             result=side, text="")
        if tau <= self._hinge_friction(rec, mass):
            return
        top = self._top_pusher(rec)
        who = None if top is None else next(
            (q for q in self.persons if q["pid"] == top), None)
        self._start_swing(rec, dirn, omega0=0.0, actor=who, why=None)

    def _static_overload(self, rec, side, F, arm):
        """A held push of F at lever arm `arm` on a leaf that is held: break
        the links whose static capacity is beaten. True if the leaf is free
        afterwards."""
        sup = self._supports(rec, side)
        if not sup:
            return True
        y_c = (rec["z0"] + 0.5 * rec["H"] - rec["z0"] + 0.5) * self._d()
        sup = [(x, (y if y is not None else y_c), k, kd, rf)
               for x, y, k, kd, rf in sup]
        try:
            shares, _ = self._plate(sup, arm, y_c)
        except ValueError:                 # held by matter alone: no force breaks it
            return False
        flaws = self._flaws(rec)
        broke = []
        for s, (_x, _y, _k, kind, ref) in zip(shares, sup):
            R = float(s) * F
            ids = (ref["pull"] if R < 0.0 else ref["path"]) if kind == "hinge" \
                else (ref["links"] if kind == "stop" else ref["path"])
            for lid in ids:
                lk = self._mlink(rec, lid)
                cap0 = mech.link_capacity(lk, flaw01=flaws.get(lid, 0.5),
                                          flaw_amp=rec["flaw_amp"])
                if mech.static_fails(abs(R), cap0, lk["D"]) \
                        and not mech.link_failed(lk):
                    broke.append((lid, abs(R)))
        for lid, R in broke:
            self._link_broken(rec, lid, "push", R)
        return not (self.holds_open_side(rec) if side == "open"
                    else not self._stop_holds(rec))

    # ── the identity pass: anchors ──────────────────────────────────────────
    def _identity(self):
        """Look again at every thing that moved or changed (the base law), then
        re-check the anchors of every mechanism whose leaf, frame or anchor
        things are among them: a link whose matter is no longer where it
        holds gets D = 1 and a 'breaks' event. Fire that burns a jamb to ash
        changes `mat` in place; the hygiene compare in `_law_identity` marks
        the thing dirty, so this is how 'a jamb burned to ash holds nothing'
        reaches the links."""
        todo = set(self._id_dirty) | {int(k) for k, _c in self._id_losses}
        why = {int(k): c for k, c in self._id_losses}
        super()._identity()
        if "mechs" not in self.__dict__ or not self.mechs or not todo:
            return
        for rec in self.mechs.values():
            if not ({rec["leaf"], rec["frame"]} & todo
                    or (rec.get("bar") and rec["bar"]["eid"] in todo)):
                continue
            cause = why.get(rec["frame"]) or why.get(rec["leaf"])
            self._anchor_upkeep(rec, cause)

    def _anchor_upkeep(self, rec, cause=None):
        for lk in rec["links"]:
            if mech.link_failed(lk):
                continue
            for a in lk["anchor"]:
                if isinstance(a, int):
                    t = self.things.get(a)
                    if t is None or t["dead"] is not None:
                        self._anchor_gone(rec, lk, cause)
                        break
                else:
                    x, y, z, m = (int(v) for v in a)
                    if self.mat[x, y, z] != m or self.rubble[x, y, z]:
                        self._anchor_gone(rec, lk, cause)
                        break

    def _anchor_gone(self, rec, lk, cause):
        lk["D"] = 1.0
        self._ev("breaks", target=("thing", rec["frame"]),
                 place=self._link_place(lk), result=lk["name"],
                 why=cause or "lost its matter", text="",
                 data=(("link", lk["id"]),))

    # ── acts ────────────────────────────────────────────────────────────────
    def push_limit(self, p, cells=None):
        """A push is limited by the foot: F <= mu_foot m g (68 kg at 0.6 gives
        400 N = BODY strength). `cells`: the body's flesh, if the caller has it."""
        cells = self._person_cells_of(p) if cells is None else cells
        kg = float(self._mass_kg(cells).sum()) if cells is not None else 68.0
        return min(float(BODY.get("strength_N", 400.0)), MU_FOOT * kg * GRAV)

    def operate(self, p, mid, op="open", F=None, arm=None, sign=None, lockkw=None):
        """The one door act. `op`:

          "open"    handle (retracts a latchbolt if nothing locks it), then push
                    to open (or, if the door is open, to close);
          "push"    push the leaf with force F at lever arm `arm` (m), in the
                    sense of `sign` (default: the way it opens);
          "release" let go of a held push;
          "key"     turn a key in the lock (`lockkw={"key": record, "T": N m}`);
          "pick"    work the lock with picks (`lockkw={"tension": N m, "skill":
                    0..1, "tool": record}`): a continuing act, probe by probe.

        A person who cannot (a locked door, a bar, ice) gets 'jams' with the
        reason as its result: the mind learns the state by trying (F7). Returns
        a dict {result, ...}. The push is a held force: the world law
        (`_law_mechs`) integrates the swing tick by tick."""
        rec = self.mechs[mid]
        pid = p["pid"]
        if op == "release":
            rec["pushes"] = [q for q in rec["pushes"] if q.get("pid") != pid]
            return {"result": "released"}
        if op in ("key", "pick"):
            return self._work_lock(p, rec, op, **(lockkw or {}))
        if op == "bar":
            return self._work_bar(p, rec)
        if op.startswith("pick:"):
            # the tension is a choice the person makes (locks.py: light or
            # firm); the skill is the person's own dial (`skill_pick`)
            tension = PICK_TENSION_NM.get(op[5:])
            if tension is None:
                return {"result": "no such op"}
            kw = dict(lockkw or {})
            kw.setdefault("tension", tension)
            return self._work_lock(p, rec, "pick", **kw)
        if op not in ("open", "close", "push"):
            return {"result": "no such op"}   # never a push by default (the old fall-through)
        if rec["swing"] is not None:
            return {"result": "busy"}
        j = rec["joints"][0]
        theta = j.get("theta", 0.0)
        s = rec["s_open"]
        d = self._d()
        if sign is None:
            sign = s if abs(theta) < 1e-9 else -s
        if op == "open" and abs(theta) < 1e-9:
            held = self._locked_by(rec)
            if held:
                ev = self.emit("jams", actor=pid, target=("thing", rec["leaf"]),
                               result=held, text="")
                self._learn_door(pid, rec, held, ev)
                return {"result": "jammed", "why": held, "event": ev}
            for c in rec["catches"]:
                if c["kind"] == "latchbolt" and c["x_mm"] > 0.0 \
                        and c.get("blocked_by") is None:
                    c["x_mm"] = 0.0                      # the handle draws the bolt
        lim = self.push_limit(p)
        F = lim if F is None else min(float(F), lim)
        if arm is None:
            arm = min(0.8, rec["W"] * d - 0.1)
        rec["pushes"] = [q for q in rec["pushes"] if q.get("pid") != pid]
        rec["pushes"].append({"pid": pid, "F": float(F), "arm": float(arm),
                              "sign": int(np.sign(sign))})
        self._mech_tables()
        self._mech_active.add(mid)
        return {"result": "pushing", "F": F, "arm": arm}

    def _work_lock(self, p, rec, op, key=None, T=1.0, tension=0.03, skill=None,
                   tool=None):
        """Turn a key, or pick, in the door's lock. Whether it opens is the
        lock's geometry, the tension and the hand (locks.py); nothing here
        rolls a die. A success throws or draws the bolts the lock holds."""
        if _locks is None or not rec["locks"]:
            return {"result": "no lock"}
        lock = rec["locks"][0]
        pid = p["pid"]
        at = self._stop_cells(rec)[0]
        at = (int(at[0]), int(at[1]), int(at[2]))
        if op == "key":
            r = _locks.try_key(lock, key, T)
            ev = self.emit("locks" if not lock.get("thrown", True) else "unlocks",
                           actor=pid, target=("thing", rec["leaf"]), place=at,
                           result=r["result"], text="")
            if r["result"] == "turns":
                self._lock_turn(rec, lock, ev)
            elif r["result"] == "snaps":
                self.emit("breaks", actor=pid, target=("thing", rec["leaf"]),
                          place=at, result="the key snaps", cause=ev.id, text="")
            return r
        sk = _locks.skill_pick(p) if skill is None else float(skill)
        res = _locks.pick_session(lock, tension, sk, pid=pid, tool=tool)
        ev = self.emit("picks", actor=pid, target=("thing", rec["leaf"]),
                       place=at, result="open" if res["opened"] else "closed",
                       text="", data=(("probes", res["probes"]),))
        for _ in range(min(res["probes"], 8)):
            _locks.emit_click(self, at, ev=ev.id, pid=pid)
        if res["opened"]:
            self._lock_turn(rec, lock, ev)
        return res

    def _lock_turn(self, rec, lock, ev):
        """The plug turned: the bolts the lock holds are thrown or drawn."""
        thrown = not lock.get("thrown", True)
        lock["thrown"] = thrown
        for c in rec["catches"]:
            if c.get("held_by") == lock["id"] and c["kind"] == "deadbolt":
                c["x_mm"] = c["throw_mm"] if thrown else 0.0
        self.emit("locks" if thrown else "unlocks", target=("thing", rec["leaf"]),
                  result="thrown" if thrown else "drawn", cause=ev.id, text="")

    def _locked_by(self, rec):
        """Why the handle will not open the door: a thrown deadbolt, a seated
        bar, a latch held by a lock, matter holding a catch. '' if nothing."""
        for c in rec["catches"]:
            if not self._catch_holds(rec, c):
                continue
            if c["kind"] == "deadbolt":
                return "locked"
            if c.get("blocked_by") is not None:
                return "held"
            if c["kind"] == "latchbolt" and c.get("held_by") is not None:
                lk = [l for l in rec["locks"] if l.get("id") == c["held_by"]]
                if lk and lk[0].get("thrown", True):
                    return "locked"
        if rec.get("bar") and self.bar_seated(rec):
            return "barred"
        return ""

    def mech_names(self, p, mids, at=None):
        """{mid: words} for the doors a body can see, as that body would say
        them: "the door to the east", from where it stands, by bearing
        (the same words `mindlink._doors` uses), never by id. Two that would be
        called alike are told apart by which is nearer ("the first door to the
        east"), never in metres. No word says what state a door is in."""
        ax, ay = self._actor_center(p) if at is None else at
        got = []
        for mid in mids:
            lc = self._cells_of(self.mechs[mid]["leaf"])
            if lc is None:
                continue
            cx, cy = float(lc[:, 0].mean()), float(lc[:, 1].mean())
            way = self._bearing(cx - ax, cy - ay)
            got.append((max(abs(cx - ax), abs(cy - ay)), mid,
                        "the door " + ("here" if way == "here" else "to the " + way)))
        count = {}
        for _, _, n in got:
            count[n] = count.get(n, 0) + 1
        out, nth = {}, {}
        for _dist, mid, n in sorted(got):
            if count[n] > 1:
                k = nth[n] = nth.get(n, 0) + 1
                n = n.replace("the door", "the %s door" % _ORDINAL[min(k, len(_ORDINAL)) - 1], 1)
            out[mid] = n
        return out

    def mech_near(self, p, reach_m=None, cells=None):
        """[(distance_m, mid, rec)], nearest first: the mechanisms whose leaf
        has a cell within `reach_m` of the body (cells, horizontally)."""
        reach_m = REACH_M if reach_m is None else reach_m
        tabs = self.__dict__.get("mechs")
        if not tabs:
            return []
        cells = self._person_cells_of(p) if cells is None else cells
        if cells is None:
            return []
        d = self._d()
        cb = _cols(cells)
        reach_c = reach_m / d
        lo, hi = cb.min(axis=0), cb.max(axis=0)
        near = []
        for mid, rec in tabs.items():
            # A LEAF TURNS ABOUT ITS PIVOT, so it is always within W cells of
            # it: a body farther than the reach from that box is out of reach
            # and the leaf's cells need not be read (exact, only a prefilter)
            px, py = rec["pivot2"][0] / 2.0, rec["pivot2"][1] / 2.0
            r = rec["W"] + 1
            if lo[0] - (px + r) > reach_c or (px - r) - hi[0] > reach_c \
                    or lo[1] - (py + r) > reach_c or (py - r) - hi[1] > reach_c:
                continue
            lc = self._cells_of(rec["leaf"])
            if lc is None:
                continue
            dd = int(np.abs(_cols(lc)[:, None, :] - cb[None, :, :]).max(axis=2).min()) * d
            if dd <= reach_m:
                near.append((float(dd), mid, rec))
        near.sort(key=lambda t: (t[0], t[1]))
        return near

    def mech_affords(self, p, cells=None):
        """Rows for the menu: per mechanism in reach, nearest first.

        What is offered is what the body can KNOW from where it stands: a door
        it can reach is something to open (or, standing open, to close), to put
        a shoulder to, to kick; a held key is something to turn; a held pick,
        something to work the lock with; a bar it holds, something to seat; a
        bar it can see in its brackets, something to lift. Never a word of
        'locked' or 'barred' or 'stuck': the same rows for a door that gives
        and one that does not (a person learns by trying, and what it learned
        is its own memory: `_door_memory`).

        Each row: key, tag, verb, mid, op, away (m). The hands' rows have
        verb 'operate' (op: open, close, key, bar, pick:light, pick:firm) or
        'shoulder'; the legs' row has verb 'kick'."""
        cells = self._person_cells_of(p) if cells is None else cells
        if cells is None:
            return []
        near = self.mech_near(p, cells=cells)
        if not near:
            return []
        names = self.mech_names(p, [m for _d, m, _r in near],
                                at=(float(cells[:, 0].mean()), float(cells[:, 1].mean())))
        held = None
        if p.get("held"):
            held = self.things.get(self._held_eid(p))
        he = self._held_eid(p) if p.get("held") else 0
        rows = []
        for dd, mid, rec in near:
            nm = names.get(mid)
            if nm is None:
                continue
            away = round(dd, 4)
            op = "close" if abs(rec["joints"][0].get("theta", 0.0)) > 1e-9 else "open"
            memo = " (it would not give when you tried)" \
                if self._door_memory(p["pid"], rec) else ""

            def row(key, tag, verb, op_):
                rows.append({"key": key, "tag": tag, "verb": verb, "mid": mid,
                             "op": op_, "away": away})
            row("%s %s%s" % (op, nm, memo), "operate", "operate", op)
            if held and "key" in held:
                row("turn the key in %s" % nm, "operate", "operate", "key")
            b = rec.get("bar")
            if b and he == b["eid"] and not self.bar_seated(rec):
                row("seat the bar across %s" % nm, "operate", "operate", "bar")
            elif b and he != b["eid"] and self.bar_seated(rec):
                bc = self._cells_of(b["eid"])
                if bc is not None and _gap_cells(bc, cells) * self._d() <= REACH_M:
                    row("lift the bar from %s" % nm, "operate", "operate", "bar")
            if held and "tool" in held:
                for tension in ("light", "firm"):
                    row("work the lock of %s with the picks, %s tension"
                        % (nm, tension), "operate", "operate", "pick:" + tension)
            row("put a shoulder to %s" % nm, "shoulder", "shoulder", "shoulder")
            row("kick %s" % nm, "kick", "kick", "kick")
        return rows

    # ── the _spend seam ─────────────────────────────────────────────────────
    def mech_blow(self, eid, cells, joules, a_hit=None, direction=None,
                  striker=None):
        """The handler a thing's record registers for the blow seam in
        bodies._spend: a struck cell whose thing has a record gets the blow
        first. It converts the blow through the record's own links and returns
        the joules it did NOT take (all of them are taken: the leaf absorbs a
        blow through its supports; the remainder is 0.0). No 'is a door' test:
        a thing is handled because its record says so."""
        mid = self.things.get(int(eid), {}).get("mech")
        if mid is None or mid not in self.mechs:
            return float(joules)
        cell = tuple(int(v) for v in np.asarray(cells).reshape(-1, 3)[0])
        side = None if direction is None else direction
        self.blow(mid, cell=cell, v_s=KICK_V_MPS, joules=joules,
                  actor=striker, side=side, verb="strikes")
        return 0.0
