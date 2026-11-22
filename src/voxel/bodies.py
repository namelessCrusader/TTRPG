"""Free rigid bodies: launch, ballistics, drag, a person going over (a pendulum), landing, and what a landing costs."""
import math

import numpy as np
from .materials import (AIR, AIR_DENS, BODY, DRAG_CD, FLESH, FRICTION, GRAVITY, MATNAME,
                        NOFLUID, TICK_S, _CSOLID_ARR, _TOUGH_ARR)
from .matdamage import shatter_threshold
from . import contact, matlaws, edgewear


_SIX = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

# The sideways speed (m/s) under which a man coming down is taken to have
# stopped: slower than a slow walk, and not worth carrying into his next step.
LAND_V = 0.3

# WHAT A CELL OF MATTER IS, AND SO WHAT GOES WITH IT WHEN IT MOVES (step E8).
# Every site that moves matter on the lattice used to keep its own list of
# fields, and each list missed something: a walk left `fallh` and `rubble`
# behind, a shove, a turn, a reach and a crouch left `rubble`, and a topple
# put back where it stood forgot its `fallh`. A shard shoved one cell became
# whole glass again. Now there is one list, and `_lift` and `_put` are the
# only way matter leaves a cell and arrives in one. Every field is zero when
# a cell is empty (AIR, NOFLUID, no edge, no eid), so lifting is zeroing.
CARRIED = ("mat", "smass", "E", "fl", "fvol", "fpot", "edge", "fallh",
           "rubble", "eid")
# the same fields as a free body keeps them, one array each, a row per cell
AS_BODY = ("mats", "masses", "Es", "fls", "fvols", "fpots", "edges",
           "fallhs", "rubbles", "eids")
_E, _FL, _FVOL, _FPOT, _FALLH = (CARRIED.index(f) for f in
                                 ("E", "fl", "fvol", "fpot", "fallh"))
assert AIR == 0 and NOFLUID == 0, "an empty cell is all zeros (`_lift`)"

# OPTIONAL CARRIED FIELDS (wave 13, plan section 1). Matter that wave 13 adds
# to a cell rides with it too: the mix of its solid (`scomp`) and of its fluid
# (`fcomp`, T1), its damage (`dmg`, T9), the overlay a body zone puts on a
# voxel (`bzone`, T7) and the ice grams in a cell that is not ice (`ice_g`,
# T6). Each is (field on the world, key on a free body).
#
# They are NOT in CARRIED, and that is the point. CARRIED and AS_BODY are
# pinned by tests (test_things.py, the base19 oracle) and every `_lift` result
# is read by position (the eid is LAST). An optional field is also a LAZY
# array: it is None until a lane that needs it makes it, and until then every
# line below is skipped, so a world that never makes one lifts, puts and
# hashes bodies exactly as it did. When the array exists but a body was lifted
# before it did (or is made by hand), the body has no such key and the cells it
# lands in get 0, as `_of_body` has always done for `fallhs` and `rubbles`.
CARRIED_OPT = (("scomp", "scomps"), ("fcomp", "fcomps"), ("dmg", "dmgs"),
               ("bzone", "bzones"), ("ice_g", "ice_gs"),
               ("wood_sp", "wood_sps"), ("wood_grain", "wood_grains"),
               ("wood_bound", "wood_bounds"))
# the body keys of those fields, in order
OPT_BODY = tuple(k for _, k in CARRIED_OPT)


class Lifted(tuple):
    """What `_lift` hands back: the CARRIED values as a plain tuple (len(CARRIED),
    CARRIED order, so every reader that indexes by position still works) and,
    in `.opt`, the optional fields whose arrays exist: {field: values}. `_put`
    and `_as_body` read `.opt`; a plain tuple or list has none."""

    def __new__(cls, vals=(), opt=None):
        self = super().__new__(cls, vals)
        self.opt = {} if opt is None else opt
        return self


# THE BLOW SEAM (wave 13; plan section 3.2, T11 6.3, T9 4.2a, T2 6.3).
# Every blow a moving thing lands is spent in `BodiesMixin._spend`. Before the
# cell law judges a struck cell, the blow goes through these stages, in this
# order, and each can only take joules from it:
#   1. mechanism    a struck thing whose record is a mechanism (a door, a
#                   latch, a bar) takes the blow first, through its own links
#                   (T11: MechanismsMixin.mech_blow)
#   2. record cut   a struck record (a blade, a rod: a SECTION) is cut if the
#                   blow can pay for the cut (T2: breaking.records_cut)
#   3. damage       the shatter threshold of each cell is lowered by the damage
#                   already there, thr x (1 - D) (T9: matdamage.shatter_threshold)
#   4. cell law     what is left is judged against the cell's toughness, as it
#                   always was
# A stage is a method of the world, found by name; a world whose class defines
# none (every world in 13a) has no stage, and the blow goes straight to the
# cell law. It is called per STRUCK THING (its eid) and per nothing else, as
#     stage(eid, cells, joules, a_hit, direction, striker) -> joules NOT taken
# `cells` is the (n, 3) array of that thing's struck cells, `joules` what the
# blow brings to all of them, `a_hit` the contact area (m2), `striker` the
# person who struck (their dict) or None. `direction` is None for now: the door
# handler works out which side it is struck from by where the striker stands
# (as its `kick` does) and takes "open" when nobody struck, and passing it a
# vector would be taken for a side name. A stage that wants the way the blow
# travelled gets it here, in this one call. A stage that does not own the thing
# returns `joules` unchanged, to the bit. Stage 3 is not a method: it reads the
# lazy `dmg` array, and does nothing while that is None.
BLOW_STAGES = (("mechanism", "mech_blow"), ("record_cut", "record_blow"))


def _pieces(cells):
    """How many 6-connected lumps a few hundred cells make (a plain flood:
    a body is never big enough to need the lattice labeller)."""
    left = set(map(tuple, np.asarray(cells, np.int64).reshape(-1, 3).tolist()))
    n = 0
    while left:
        n += 1
        stack = [left.pop()]
        while stack:
            x, y, z = stack.pop()
            for dx, dy, dz in _SIX:
                c = (x + dx, y + dy, z + dz)
                if c in left:
                    left.remove(c)
                    stack.append(c)
    return n


def _labels(cells):
    """Which 6-connected lump each of these cells is in: 0, 1, ... per row."""
    c = np.asarray(cells, np.int64).reshape(-1, 3)
    at = {tuple(v): i for i, v in enumerate(c.tolist())}
    lab = np.full(len(c), -1, np.int64)
    n = 0
    for i in range(len(c)):
        if lab[i] >= 0:
            continue
        lab[i] = n
        stack = [i]
        while stack:
            x, y, z = c[stack.pop()].tolist()
            for dx, dy, dz in _SIX:
                j = at.get((x + dx, y + dy, z + dz))
                if j is not None and lab[j] < 0:
                    lab[j] = n
                    stack.append(j)
        n += 1
    return lab


class BodiesMixin:
    # THE MATERIAL LAWS (T9, wave 13b): which of matlaws.ALL_LAWS are on. A CLASS
    # attribute, so a saved world still loads; `w.mat_laws = frozenset()` gives
    # a world the old behaviour. See matlaws.py. "damage" is a per-world
    # capability for now (matlaws.ALL_LAWS has it): its other two sites live in
    # support.py and five tests assume a landing adds no crack.
    mat_laws = frozenset(("friction", "members", "spall", "edge", "wound"))


    def _lift(self, idx):
        """Take the matter out of the cells `idx` (anything numpy can index
        with): a copy of every CARRIED field there, in CARRIED order, and the
        cells left empty. A thing whose cells these were has moved (or is
        about to), so the thing index hears of it HERE, once (`_things_moved`
        bumps `_thing_gen`): what is kept about where things are is out of
        date from the moment its matter leaves a cell.

        The optional fields (`CARRIED_OPT`) whose arrays exist come with it,
        in `.opt` of what is returned (a tuple of the CARRIED values still:
        see `Lifted`); the cells are left empty of those too."""
        vals = Lifted(getattr(self, f)[idx].copy() for f in CARRIED)
        for f in CARRIED:
            getattr(self, f)[idx] = 0
        for f, _ in CARRIED_OPT:
            a = getattr(self, f, None)
            if a is not None:
                vals.opt[f] = a[idx].copy()
                a[idx] = 0
        self._things_moved(vals[-1])
        return vals

    def _put(self, idx, vals, opt=None):
        """Put matter lifted by `_lift` (or made the same shape) into the
        cells `idx`, every CARRIED field. The `_lift` it came from already
        told the thing index; a site that puts down matter it did not lift
        in the same breath (a landing) tells it itself, once.

        The optional fields that exist go in too, from `opt` (or `vals.opt`):
        a field it does not name, because the matter was lifted before that
        array existed, is put as 0."""
        for f, v in zip(CARRIED, vals):
            getattr(self, f)[idx] = v
        if opt is None:
            opt = getattr(vals, "opt", None) or {}
        for f, _ in CARRIED_OPT:
            a = getattr(self, f, None)
            if a is not None:
                a[idx] = opt.get(f, 0)

    @staticmethod
    def _as_body(vals):
        """Lifted matter as a free body keeps it: {"mats": ..., ...}, and the
        optional keys ("scomps", ...) for the optional fields it was lifted
        with. Nothing is added while no optional array exists."""
        body = dict(zip(AS_BODY, vals))
        have = getattr(vals, "opt", None)
        if have:
            for f, k in CARRIED_OPT:
                if f in have:
                    body[k] = have[f]
        return body

    def _of_body(self, b, i):
        """Row(s) `i` of a free body, as `_put` takes them. A body made before
        it carried `fallhs` and `rubbles` (or made by hand) has none, and
        neither has one made before an optional field's array existed."""
        out = []
        for f, k in zip(CARRIED, AS_BODY):
            v = b.get(k)
            out.append(np.zeros_like(b["masses"][i], dtype=getattr(self, f).dtype)
                       if v is None else v[i])
        opt = {}
        for f, k in CARRIED_OPT:
            a = getattr(self, f, None)
            if a is not None:
                v = b.get(k)
                opt[f] = (np.zeros_like(b["masses"][i], dtype=a.dtype)
                          if v is None else v[i])
        return Lifted(out, opt)

    def _launch(self, cells, vel, owner=None, part=False):
        """PROMOTE a cluster to a body in free FLIGHT — the same promote/demote
        protocol as a topple, but travelling rather than turning.

        This is what having a velocity buys. A topple could only ever swing
        about a pivot it was already touching, so nothing in the sim could
        leave the ground: not a jump, not a thrown stone, not a swung axe. A
        flying body carries m/s, is pulled on by gravity, is pushed back on by
        the air it has to shove aside, and pays what it has left as energy
        when it arrives."""
        cells = np.asarray(cells, np.int64)
        grip = None
        if owner is not None and not part:
            cells = self._with_limbs(cells, owner)   # a swung arm comes too
            n0 = len(cells)
            cells, grip = self._with_held(cells, owner)   # and what he holds
        body = {"cells": cells.astype(np.float32),
                **self._as_body(self._lift(tuple(cells.T))),
                "fly": True, "vel": np.asarray(vel, np.float64).copy(),
                "off": np.zeros(3), "owner": owner,
                "axis": 0, "s": 1, "pivot": 0.0, "zb": 0.0,
                "theta": 0.0, "omega": 0.0, "phi0": 0.02, "L": 2.0}
        if grip is not None:
            body["grip"] = grip               # the row of the cell in his hand
        if owner is not None and not part and len(cells) > n0:
            body["held_from"] = n0            # rows from here on: what he holds
        self.bodies.append(body)
        if part:
            body["part"] = True               # a limb: PART of somebody
        elif owner is not None:
            # HE IS IN THE AIR NOW, AND THIS TICK KNOWS IT. The claim made
            # earlier in the tick still held the body he just left, so anything
            # later in the same tick that asked where he was got the cells he
            # stood in, which are air now: measured on fuzzer seed 16, a man
            # in flight "walked" 318 voxels of air at t0. Asked again, the
            # claim finds him aloft and says he has no body on the lattice.
            #
            # AND SO IS ANYONE WHOSE FLESH WENT WITH HIM. A lump that topples
            # can be two people (an unconscious man and the corpse against
            # him), and the owner names one; the eid on the flesh names both.
            fl = body["eids"][body["mats"] == FLESH]
            for q in self.persons:
                if q["name"] == owner or (q.get("eid") and (fl == q["eid"]).any()):
                    q["_claim_tick"], q["_claim_mask"] = None, None
        self._torque_solid = None
        self._slack_mat = None
        return body

    def _tip(self, p, cells, axis, s, pivot, zb, omega=0.0, lean=0.02):
        """A PERSON GOES OVER: promote the body to a free rigid body that turns
        about the edge of its base under real gravity, and land it by the same
        arrival as any toppled body (`_land_body`).

        The turn is a pendulum. About a horizontal line through the pivot (the
        edge of the contacts the weight is past, at the height of the lowest of
        them) the body has a mass m, a centre of mass `dxc` out past the edge
        and `dzc` above it, and a moment of inertia I (its voxels' m r^2 about
        that line). Gravity's torque turns it, so

            I alpha = m g (dxc cos(theta) + dzc sin(theta))

        with theta the lean from upright. Nothing here is typed in: a taller or
        heavier body, or one whose weight is higher, falls at its own pace.
        (`support._topple`, which tips THINGS, is left as it is.) `omega` is the
        turn it starts with in rad/s, from a push; `lean` the least lean it
        starts from. Returns the new body, or
        False if it cannot even begin to lean (it is wedged against something:
        it is put back where it stood, and let be for a while)."""
        name = p["name"]
        if p.get("_tip_wait", -1) > self.tick:
            return False
        cells = self._with_limbs(np.asarray(cells, np.float32), name)
        n0 = len(cells)
        cells, grip = self._with_held(cells, name)       # and what he holds
        idx = tuple(np.asarray(cells, np.int64).T)
        lifted = self._lift(idx)
        carried = self._as_body(lifted)
        kg = np.maximum(carried["masses"], 1e-6) / 1000.0
        vox_m = 0.1 * self.scale
        u = (cells[:, axis] - pivot) * s * vox_m          # out past the edge
        h = (cells[:, 2] - zb) * vox_m                     # up from the pivot
        m = float(kg.sum())
        dxc, dzc = float((kg * u).sum()) / m, float((kg * h).sum()) / m
        # nobody is balanced on a knife edge: the lean they start at is at
        # least a degree or so (`lean`, in radians), which also starts a body
        # whose centre sits exactly over the edge
        dxc = max(dxc, lean * max(dzc, vox_m))
        body = {"cells": cells.astype(np.float32), **carried,
                "axis": axis, "s": s, "pivot": float(pivot), "zb": float(zb),
                "theta": 0.0, "omega": 0.0, "w": float(omega),
                "pend": (m, float((kg * (u * u + h * h)).sum()), dxc, dzc),
                "phi0": 0.02, "L": 2.0, "owner": name}
        probe = np.round(self._body_pose(body, 0.1)).astype(np.int64)
        if self._in_the_way(probe) > max(2, len(cells) // 50):
            self._put(idx, lifted)                  # wedged: back, and wait
            p["_tip_wait"] = self.tick + 20
            return False
        if grip is not None:
            body["grip"] = grip
        if len(cells) > n0:
            body["held_from"] = n0
        self.bodies.append(body)
        fl = body["eids"][body["mats"] == FLESH]
        for q in self.persons:                      # he is in the air now
            if q["name"] == name or (q.get("eid") and (fl == q["eid"]).any()):
                q["_claim_tick"], q["_claim_mask"] = None, None
        self._torque_solid = None
        self._slack_mat = None
        self._defer("topples", actor=p["pid"],
                    place=tuple(float(v) for v in cells.mean(axis=0)), text="")
        return body

    def _pivot_mu(self, b):
        """Friction under the lowest cells of a toppling body: the mean over its
        columns of the floor's effective value (contact.floor_mu) with the dry
        value kept at FRICTION: the pivot's 0.40 is what the ledge and rescue
        scenes are calibrated on, and only a FILM (oil, water, mud) lowers it.
        FRICTION when the contact law is off, or nothing is under it."""
        if not self.contact_laws:
            return FRICTION
        c = np.round(b["cells"]).astype(np.int64)
        zlo = int(c[:, 2].min())
        low = c[c[:, 2] == zlo]
        mus = [contact.floor_mu(self, int(x), int(y), zlo, dry=FRICTION)
               for x, y, _z in low[:64]]
        mus = [m for m in mus if m is not None]
        return float(np.mean(mus)) if mus else FRICTION

    def _slips(self, b, th, w_s):
        """DOES THE FOOT LET GO. A man who is awake is not a plank on a hinge:
        his feet hold the edge only as long as friction can supply the push the
        turn asks of them. The reaction at the pivot is the centre of mass's
        own acceleration (the horizontal part, against FRICTION times the
        weight it bears), and when the first outruns the second the feet slide
        and he leaves the edge as a free body with the speed his centre of
        mass has, which is the hop `_law_footing` used to give him by a
        formula, now from the turn he has actually made. Past that point it is
        a flight (and a landing on his feet). A man who is out (unconscious or
        dead) has no feet to slide: he stays on the hinge to the floor.
        Returns True if the body became a flier."""
        q = next((r for r in self.persons if r["name"] == b.get("owner")), None)
        if q is None or not q["alive"] or not q["awake"]:
            return False
        m_, inertia, dxc, dzc = b["pend"]
        c, sn = float(np.cos(th)), float(np.sin(th))
        x_, z_ = dxc * c + dzc * sn, dzc * c - dxc * sn      # centre, from the pivot
        alpha = m_ * GRAVITY * x_ / inertia
        a_x = z_ * alpha - x_ * w_s * w_s
        a_z = -x_ * alpha - z_ * w_s * w_s
        # (T3, wave 13c C13) THE FOOT HOLDS AS WELL AS THE FLOOR LETS IT: the
        # friction under the pivot is read from what lies there (oil, water,
        # mud), not the 0.40 of a dry floor. With nothing on it, the same 0.40.
        if abs(a_x) <= self._pivot_mu(b) * max(GRAVITY + a_z, 0.0):
            return False
        vel = np.zeros(3)
        vel[int(b["axis"])] = float(b["s"]) * z_ * w_s
        vel[2] = -x_ * w_s
        # WHAT HE STANDS ON DOES NOT LET HIM SINK INTO IT. While the soles are
        # on something, the floor takes the downward part of the speed (the
        # feet slide along it); he drops only once he is off it.
        if len(self._contact(np.round(b["cells"]).astype(np.int64))):
            vel[2] = max(vel[2], 0.0)
        b["fly"], b["vel"], b["off"] = True, vel, np.zeros(3)
        del b["pend"]
        return True

    def _fly_pose(self, b, off=None):
        """Where a flying body's cells are, at its current offset."""
        off = b["off"] if off is None else off
        return b["cells"] + off

    def _drag_a(self, b):
        """Deceleration from the air, in m/s^2, along the way it is going.

        Frontal area is COUNTED, not declared: the cells that face the
        direction of travel are the ones doing the shoving. A body spread
        broadside to its own fall presents many of them and slows; the same
        matter end-on presents few and does not. That is a parachute, and also
        why a plank falls differently flat than edge-on."""
        v = float(np.linalg.norm(b["vel"]))
        if v < 1e-6:
            return np.zeros(3)
        vox_m = 0.1 * self.scale
        ax = int(np.argmax(np.abs(b["vel"])))            # the way it is going
        face = {tuple(int(c[i]) for i in range(3) if i != ax)
                for c in b["cells"]}                     # its shadow, that way
        area = len(face) * vox_m * vox_m
        # UNLESS IT IS NARROWER THAN A CELL, and says so. An arrow is 8 mm
        # across; drawn as one 5 cm voxel it would meet the air with thirty
        # times its real face and fall out of the sky in twenty metres. That
        # is sub-voxel shape, the same class of fact as an edge (Ruling 2 q1).
        if b.get("area_m2"):
            area = float(b["area_m2"])
        kg = max(float(b["masses"].sum()) / 1000.0, 1e-9)
        mag = 0.5 * AIR_DENS * DRAG_CD * area * v * v / kg
        return -mag * (b["vel"] / v)

    def _body_pose(self, b, theta):
        """Rotated float positions of a body's cells at lean angle theta."""
        c, ax, s = b["cells"], b["axis"], b["s"]
        d_lat = c[:, ax] - b["pivot"]
        d_z = c[:, 2] - b["zb"]
        ct, st = np.cos(theta), np.sin(theta)
        lat = b["pivot"] + d_lat * ct + s * d_z * st
        zz = b["zb"] - s * d_lat * st + d_z * ct
        out = c.copy()
        out[:, ax] = lat
        out[:, 2] = zz
        return out

    _PERCELL = ("cells",) + AS_BODY + OPT_BODY + ("segof",)

    def _release(self, b, thrown, at):
        """Let go of part of a swinging body, at the speed it was going.

        A throw is not a new kind of motion. The thing is already travelling —
        it has been going round on the end of an arm — and letting go only
        stops it being made to go round. So its speed is the speed it had,
        `omega` times how far out it was, and its direction is the tangent,
        which is where the hand was taking it anyway."""
        ax, sgn = int(b["axis"]), int(b["s"])
        vox_m = 0.1 * self.scale
        c = b["cells"][thrown]
        d_lat = float(c[:, ax].mean()) - b["pivot"]
        d_z = float(c[:, 2].mean()) - b["zb"]
        th = float(b["theta"])
        ct, st = float(np.cos(th)), float(np.sin(th))
        w_rad = float(b["omega"]) / max(TICK_S, 1e-9)
        vel = [0.0, 0.0, 0.0]
        vel[ax] = (-d_lat * st + sgn * d_z * ct) * w_rad * vox_m
        vel[2] = (-sgn * d_lat * ct - d_z * st) * w_rad * vox_m
        flier = {"cells": at[thrown].astype(np.float32),
                 "fly": True, "vel": np.asarray(vel, np.float64),
                 "off": np.zeros(3), "owner": None,
                 "axis": 0, "s": 1, "pivot": 0.0, "zb": 0.0,
                 "theta": 0.0, "omega": 0.0, "phi0": 0.02, "L": 2.0,
                 "born": self.tick}
        for k in AS_BODY + OPT_BODY:          # all that the cells carry
            if b.get(k) is not None:
                flier[k] = b[k][thrown].copy()
        for k in self._PERCELL:
            if k in b and b[k] is not None and len(b[k]) == len(thrown):
                b[k] = b[k][~thrown]
        # AND THE ARM IS LIGHTER NOW, so it comes round faster — which is what
        # anyone who has thrown something has felt.
        if len(b["cells"]):
            kg = b["masses"] / 1000.0
            r = (np.abs(b["cells"][:, ax] - b["pivot"])
                 + np.abs(b["cells"][:, 2] - b["zb"])) * vox_m
            b["I"] = max(float((kg * r * r).sum()), 1e-6)
        b["throw"] = False
        self.bodies.append(flier)
        for q in self.persons:                    # the hand is empty now
            if q["name"] == b.get("owner"):
                q["held"] = None
                # WHO THREW IT. Stamped here, where the person dict is in
                # hand (3.4's attribution rule), so a later `hits` on this
                # flier can name its thrower without a lookup by name.
                flier["by_pid"] = q["pid"]
                self.emit("releases", actor=q["pid"],
                          text=f"{q['name']} lets fly",
                          to=q, site="release:lets_fly")
                break
        return flier

    def _born_Es(self, cell, mat, grams):
        """The heat above 20 C a thing let go at `cell` carries: its own mass at the
        temperature of the cell it is born in (K above 20 C = E / C of that cell).
        It was always 0.0, an arrow at 20 C whatever the air: harmless while
        every world rested at 20 C, and in a cold world a stone arriving 32 K too
        warm (T8 4.3). In a cell at 20 C the answer is the same 0.0."""
        ix = tuple(int(round(float(v))) for v in cell)
        ok = all(0 <= i < n for i, n in zip(ix, self.shape))
        e = float(self.E[ix]) if ok else 0.0
        if e == 0.0:                       # a cell at 20 C: the old 0.0, and nothing asked of the caches
            return np.zeros(1, np.float32)
        th = e / float(self.heat_capacity()[ix])
        return np.array([th * grams * float(_CSOLID_ARR[mat])], np.float32)

    def loose(self, cell, vel, mat, grams, edge=0.0, area_m2=None):
        """Let a small thing go at a speed: an arrow off a string, a bolt off a
        trap, a stone out of a sling. It is one cell of matter in flight like
        anything thrown, and everything after this is the flight law's and
        the wound law's — this is only the moment of release.

        `edge` is the area its point or edge concentrates a blow into, and
        `area_m2` its real frontal area if it is narrower than a cell. Both
        are sub-voxel facts the object declares about itself."""
        c = np.asarray([cell], np.float32)
        b = {"cells": c, "fly": True, "vel": np.asarray(vel, np.float64),
             "off": np.zeros(3), "owner": None,
             "axis": 0, "s": 1, "pivot": 0.0, "zb": 0.0,
             "theta": 0.0, "omega": 0.0, "phi0": 0.02, "L": 2.0,
             "mats": np.array([mat], np.int16),
             "masses": np.array([grams], np.float32),
             "Es": self._born_Es(cell, mat, grams),   # heat above 20 C: the temperature of the air it is let go in
             "fls": np.zeros(1, np.int16), "fvols": np.zeros(1, np.float32),
             "fpots": np.zeros(1, np.float32),
             "edges": np.array([edge], np.float32),
             "fallhs": np.zeros(1, np.float32),
             "rubbles": np.zeros(1, bool),        # a whole bolt, no piece
             "eids": np.zeros(1, np.int32)}       # made here: no thing yet
        if area_m2:
            b["area_m2"] = float(area_m2)
        b["born"] = self.tick
        self.bodies.append(b)
        self._defer("looses", place=tuple(float(v) for v in cell), text="")
        return b

    def _law_bodies(self):
        """Advance every mid-topple body one tick; land the ones that arrive."""
        # THE MATERIAL LAWS THAT RUN ON A CLOCK, not on a blow (T9, C8a). They
        # run here, before support reads the layout, so a member the load check
        # breaks is a piece when `_law_support` looks at it this very tick.
        if self.mat_laws:
            if "members" in self.mat_laws:
                matlaws.members_veto(self)
            if "spall" in self.mat_laws:
                matlaws.spall_pass(self)
        if not self.bodies:
            return
        nx, ny, nz = self.shape
        vox_m = 0.1 * self.scale
        still = []
        for b in self.bodies:
            if b.get("fly"):
                b["vel"] += (np.array([0.0, 0.0, -GRAVITY])
                             + self._drag_a(b)) * TICK_S
                # SWEPT, NOT TELEPORTED. A tick is 25 ms, and an arrow at
                # 50 m/s covers two metres of it — forty cells. Checked only
                # where it ends up, it passed clean through a man without
                # touching him. So the move is walked a cell at a time and the
                # flight stops at the FIRST thing in the way; what it touched
                # there is what it struck.
                step = b["vel"] * TICK_S / vox_m
                n_sw = max(1, int(np.ceil(float(np.abs(step).max()))))
                off, struck, hit = b["off"], None, 0
                for k in range(1, n_sw + 1):
                    off_k = b["off"] + step * (k / n_sw)
                    pose = np.round(self._fly_pose(b, off_k)).astype(np.int64)
                    inb = ((pose[:, 0] >= 0) & (pose[:, 0] < nx)
                           & (pose[:, 1] >= 0) & (pose[:, 1] < ny)
                           & (pose[:, 2] >= 0) & (pose[:, 2] < nz))
                    solid_k = np.zeros(len(pose), bool)
                    solid_k[inb] = self.mat[tuple(pose[inb].T)] != AIR
                    hit = int((~inb).sum()) + int(solid_k.sum())
                    if hit:
                        struck = [tuple(map(int, c)) for c in pose[solid_k]]
                        if k > 1:                  # back to the last clear place
                            b["off"] = b["off"] + step * ((k - 1) / n_sw)
                        break
                    off = off_k
                # ANY contact is an arrival, for a flier. A topple tolerates a
                # few grazing cells because it is pivoting through its own
                # neighbourhood; a body in flight that tolerates them lands
                # driven INTO whatever it hit, and _land_body then shoves the
                # buried cells upward as debris — which comes out through the
                # top. Measured: a person landing from a jump arrived with 15
                # of their 342 voxels rearranged, and what a person has at the
                # top of them is their head. Stop at the last clear pose and
                # let the support law set them down the rest of the way.
                #
                # BUT A WALL IS NOT A FLOOR. What a wall takes is the sideways
                # speed; what stops a fall is something underneath. So a hit is
                # retried with the horizontal part removed: if straight down is
                # clear, the body is SCRAPING PAST — the wall keeps the
                # sideways speed and the fall goes on — and it arrives only
                # when that too is blocked. Known softness: the sideways energy
                # is absorbed by the wall unpaid; the honest version spends it
                # on the struck face, and matters once throwing does.
                if hit > 0:
                    landed = True
                    if abs(b["vel"][2]) > 1e-6 and \
                            (abs(b["vel"][0]) > 1e-6 or abs(b["vel"][1]) > 1e-6):
                        vv = np.array([0.0, 0.0, b["vel"][2]])
                        off_v = b["off"] + vv * TICK_S / vox_m
                        pv = np.round(self._fly_pose(b, off_v)).astype(np.int64)
                        inb_v = ((pv[:, 0] >= 0) & (pv[:, 0] < nx)
                                 & (pv[:, 1] >= 0) & (pv[:, 1] < ny)
                                 & (pv[:, 2] >= 0) & (pv[:, 2] < nz))
                        if not int((~inb_v).sum()) and \
                                not int((self.mat[tuple(pv[inb_v].T)] != AIR).sum()):
                            # THE WALL TAKES THE SIDEWAYS SPEED, AND PAYS FOR
                            # IT. This used to be the note "the sideways energy
                            # is absorbed by the wall unpaid; the honest version
                            # spends it on the struck face, and matters once
                            # throwing does". It mattered the first time an
                            # arrow was loosed: it met a man's forearm, the man
                            # counted as a wall, and 75 J of arrow became a
                            # gentle drop at his feet. Whatever stopped the
                            # sideways part of a thing's motion takes that
                            # part's energy, by the same law a swing uses.
                            if b.get("owner") is None and struck:
                                kg = float(b["masses"].sum()) / 1000.0
                                vh = b["vel"] - vv
                                self._spend(b, struck,
                                            0.5 * kg * float(vh @ vh))
                            b["vel"] = vv
                            b["off"] = off_v
                            still.append(b)
                            landed = False
                    if landed:
                        # WHAT IT HIT PAYS. A thrown or loosed thing with no
                        # owner spends what it carried on what stopped it —
                        # a man, a door, a wall — through the same wound and
                        # shatter laws a swing uses. A person landing from a
                        # jump keeps the landing law, which already shares
                        # the blow between feet and floor.
                        self._land_body(b, None, struck=(
                            struck if b.get("owner") is None else None))
                else:
                    b["off"] = off
                    still.append(b)
                continue
            if b.get("pend"):
                # A PERSON GOING OVER (`_tip`): a rigid pendulum under real g,
                # integrated in five steps a tick. If the weight comes back
                # over the base before the body has turned (a push that was
                # not enough) it simply stands again.
                m_, inertia, dxc, dzc = b["pend"]
                th, w_s = float(b["theta"]), float(b["w"])
                dt = TICK_S / 5.0
                for _ in range(5):
                    w_s += (m_ * GRAVITY * (dxc * np.cos(th) + dzc * np.sin(th))
                            / inertia) * dt
                    th += w_s * dt
                theta_next = min(th, np.pi / 2)
                if th < 0.0:
                    self._land_body(b, 0.0)
                    continue
                pose = np.round(self._body_pose(b, theta_next)).astype(np.int64)
                hit = self._in_the_way(pose)
                tol = max(2, len(b["cells"]) // 50)
                if hit > tol or theta_next >= np.pi / 2 - 1e-6:
                    self._land_body(b, b["theta"] if hit > tol else theta_next)
                elif self._slips(b, theta_next, w_s):
                    still.append(b)           # (now a flier: see `_slips`)
                else:
                    b["theta"], b["w"] = theta_next, w_s
                    b["omega"] = w_s * TICK_S
                    still.append(b)
                continue
            if b.get("Nm"):
                # DRIVEN by a muscle: angular acceleration is torque over the
                # limb's own inertia, so a loaded arm comes round slower and
                # nobody types how long a swing takes.
                w_s = b["omega"] / TICK_S                # rad/tick -> rad/s
                fade = max(0.0, 1.0 - w_s / max(BODY["arm_wmax"], 1e-9))
                w_0 = w_s
                w_s += (b["Nm"] * fade / b["I"]) * TICK_S
                if "paid_J" in b:
                    w_s = self._pay_swing(b, w_0, w_s)
                b["omega"] = w_s * TICK_S
                theta_next = min(b["theta"] + b["omega"], np.pi / 2)
                # LET GO, if that is what this swing was for. Before the block
                # check, because a thrown thing leaves the hand and what the
                # ARM then hits is the arm's business.
                if b.get("throw") and b.get("segof") is not None \
                        and theta_next >= BODY["throw_rad"] \
                        and (b["segof"] == -1).any():
                    b["theta"] = theta_next
                    rel_at = np.round(self._body_pose(b, theta_next))
                    self._release(b, b["segof"] == -1, rel_at)
                    if not len(b["cells"]):
                        self._book_swing(b, rel_at)     # the whole limb was the throw
                        continue                  # nothing of it left to swing
                pose = np.round(self._body_pose(b, theta_next)).astype(np.int64)
                inb = ((pose[:, 0] >= 0) & (pose[:, 0] < nx) & (pose[:, 1] >= 0)
                       & (pose[:, 1] < ny) & (pose[:, 2] >= 0) & (pose[:, 2] < nz))
                blocked = pose[inb][self.mat[tuple(pose[inb].T)] != AIR]
                if len(blocked) or int((~inb).sum()) or \
                        theta_next >= np.pi / 2 - 1e-6:
                    self._land_body(b, b["theta"], struck=blocked)
                else:
                    b["theta"] = theta_next
                    still.append(b)
                continue
            alpha = (1.0 / b["L"]) * max(float(np.sin(b["theta"] + b["phi0"])), 0.02)
            b["omega"] = min(b["omega"] + alpha, 0.35)   # and never faster than the
            if b["theta"] < 0.01:                        # wedge probe on the FIRST step
                b["omega"] = min(b["omega"], 0.06)
            theta_next = min(b["theta"] + b["omega"], np.pi / 2)
            pose = self._body_pose(b, theta_next)
            pi_ = np.round(pose).astype(np.int64)
            inb = ((pi_[:, 0] >= 0) & (pi_[:, 0] < nx) & (pi_[:, 1] >= 0)
                   & (pi_[:, 1] < ny) & (pi_[:, 2] >= 0) & (pi_[:, 2] < nz))
            hit = int((~inb).sum()) + int((self.mat[tuple(pi_[inb].T)] != AIR).sum())
            tol = max(2, len(b["cells"]) // 50)          # same tolerance as the wedge
            if hit > tol or theta_next >= np.pi / 2 - 1e-6:  # probe — a graze is not
                self._land_body(b, b["theta"] if hit > tol else theta_next)   # a landing
            else:
                b["theta"] = theta_next
                still.append(b)
        self.bodies = still

    def _left_world(self, b, i):
        """One voxel of a body went past the edge of the lattice, and is gone.

        Recorded rather than dropped, because "mass is conserved" has to stay a
        statement you can CHECK: everything is on the lattice, or in a body in
        flight, or on this tally. A world with edges loses things out of them;
        a world that cannot say how much it lost is just wrong."""
        g = self.gone
        m = int(b["mats"][i])
        g[m] = g.get(m, 0.0) + float(b["masses"][i])
        # AND IF IT WAS PART OF A THING, the thing lost it. Nothing on the
        # lattice shows that (the cell never landed), so it is written down
        # for the identity pass, which would otherwise never hear of it.
        e = int(b["eids"][i])
        if e:
            self._id_losses.append((e, "gone"))
        g["E"] = g.get("E", 0.0) + float(b["Es"][i])
        if b["fvols"][i] > 0:
            g[("fluid", int(b["fls"][i]))] = \
                g.get(("fluid", int(b["fls"][i])), 0.0) + float(b["fvols"][i])

    def _spend(self, b, struck, joules):
        """A moving thing's energy, spent on the cells that stopped it."""
        vox_m = 0.1 * self.scale
        c0 = tuple(int(v) for v in struck[0])   # swings hand over an array
        # A BLOW IS ITS STRIKER'S OWN NOISE, as a footstep is the walker's. A
        # man whose fist broke a pane heard the crash as news, turned to look,
        # and was asked to decide again mid-swing — and swung again.
        self._noise(c0, joules, [int(self.mat[c0])] + [int(m) for m in set(
            int(x) for x in b["mats"])], who=b.get("owner"),
            pid=self._body_pid(b))
        # WHO STRUCK. A limb swing is its owner's own arm, so the owner is
        # the actor. A thrown or loosed flier was somebody's throw, stamped
        # at release (`_release`) as `by_pid`. Neither is true of a body
        # falling whole: nobody SWUNG it, so the actor is None and the
        # faller is only the instrument, caused by their own `falls` (3.4's
        # attribution rule).
        owner_pid = next((r["pid"] for r in self.persons
                          if r["name"] == b.get("owner")), None)
        actor_pid, instrument, fall_cause = None, None, None
        if b.get("part"):
            actor_pid = owner_pid
        elif b.get("by_pid") is not None:
            actor_pid = b["by_pid"]
        elif owner_pid is not None:
            instrument = ("person", owner_pid)
            fall_cause = b.get("falls_ev")
        # spend it on the struck cells, each against its own toughness.
        # A small contact concentrates the same energy into a larger
        # stress. THE EDGE IS THE OBJECT'S OWN CLAIM to be smaller still:
        # an axe bit is sub-millimetre, which no 5 cm lattice can draw, so
        # a manufactured edge DECLARES the area it concentrates a blow
        # into — the same class of fact as its density. Which cell of the
        # swung thing made contact is below the lattice's resolution too,
        # so the sharpest edge the body carries is the one that strikes;
        # an axe is swung edge-first, which is a fact about how tools are
        # held, not a case about axes.
        ed = b["edges"][b["edges"] > 0] if "edges" in b else []
        a_hit = min(float(min(ed)) if len(ed) else vox_m ** 2, vox_m ** 2)
        per = joules / len(struck)
        # ONE BLOW PER PERSON STRUCK, not one per voxel of them. An edge
        # meets a body in one place; the lattice happens to report three
        # touching cells, and dividing the energy three ways would turn
        # one deep cut into three scratches.
        blows = {}
        for (sx, sy, sz) in struck:
            if self.mat[sx, sy, sz] == FLESH:
                q = self._person_at((int(sx), int(sy), int(sz)))
                if q is None:
                    # FLESH NOBODY OWNS: a dead man whose deed was cleared, or
                    # a piece no claim reached. There is no one for the blow to
                    # wound, and flesh deforms, it does not shatter, so its
                    # share is simply absorbed. Fuzzer seed 58 died here at t5,
                    # on a limb that struck a corpse.
                    continue
                k = id(q)
                if k not in blows:
                    blows[k] = [q, (int(sx), int(sy), int(sz)), 0.0]
                blows[k][2] += per
        eplan = edgewear.plan(self, b, struck)    # C8b: the edge as it is, before it strikes
        for q, cell, e in blows.values():
            # ONE CONTACT LAW (wave 14b, lane E; behind `body_contact`): a blunt body
            # that strikes a person (a fist or a swung tool, a thrown thing, a body
            # falling onto them) meets them as two bodies with masses, speeds, a
            # restitution and a friction, not as joules poured into flesh. An edge
            # (a blade) cuts: that is the old blow below.
            if self.body_contact and not len(ed) and self._contact_blow(
                    b, q, cell, e, joules, a_hit, actor_pid, instrument, fall_cause):
                continue
            # flesh DEFORMS, it does not fragment — TOUGH[FLESH] said so
            # in words while a big enough blow shattered feet into debris
            # anyway. What a blow does to a body is `_blow`'s business:
            # armour, then skin, then tissue, read off energy per area.
            hits_box = self._defer("hits", actor=actor_pid,
                                    target=("person", q["pid"]),
                                    instrument=instrument, place=cell,
                                    cause=fall_cause, db=float(e), text="")
            with self.because(hits_box):
                self._blow(q, cell, e, a_hit, tip=eplan and eplan[3])
        # THE BLOW SEAM (BLOW_STAGES): a struck thing with a mechanism or a
        # record gets the blow before its cells do. `left` is None, and every
        # cell is judged on `per` as before, unless a stage took joules.
        stages = self._blow_stages()
        left = self._through_stages(stages, b, struck, per, a_hit, actor_pid,
                                    fall_cause) if stages else None
        dmg = getattr(self, "dmg", None)     # None until T9 turns damage on
        for (sx, sy, sz) in struck:
            if self.mat[sx, sy, sz] == FLESH:
                continue
            e_cell = per if left is None else left.get(
                (int(sx), int(sy), int(sz)), per)
            thr = (_TOUGH_ARR[self.mat[sx, sy, sz]] * 1000.0 * a_hit
                   * (0.7 + 0.6 * self._flaw01(int(sx), int(sy), int(sz))))
            if dmg is not None:              # a cracked cell gives at thr x (1 - D)
                thr = shatter_threshold(thr, dmg[sx, sy, sz])
            if e_cell > thr:
                with self.because(fall_cause, "call") \
                        if fall_cause is not None else self.because(
                            self._cause_near((sx, sy, sz))):
                    self._shatter(int(sx), int(sy), int(sz),
                                  e_cell / max(thr, 1e-9))
            elif "damage" in self.mat_laws:
                # A BLOW THAT DID NOT BREAK IT STILL OPENED A CRACK (T9): the
                # next one meets thr x (1 - D).
                matlaws.impact_damage(self, (int(sx), int(sy), int(sz)),
                                      float(e_cell), a_hit)
        edgewear.after_blow(self, b, struck, joules, eplan)    # C8b: the edge that struck wears

    def _blow_stages(self):
        """The stage handlers this world has (BLOW_STAGES), in order: none
        for a world whose class defines no `mech_blow` or `record_blow`."""
        return tuple(h for h in (getattr(self, name, None)
                                 for _, name in BLOW_STAGES) if h is not None)

    def _through_stages(self, stages, b, struck, per, a_hit, actor_pid,
                        fall_cause):
        """Offer a blow to the stages, THING by thing (every struck cell that
        is not flesh and has an eid), each handed the joules its cells were
        to take. Returns None if no stage took any, else {cell: the joules
        still to be spent on that cell}: what is left of a thing's joules,
        shared equally by its struck cells. A cell of a thing nobody took
        from, and a cell with no thing, keeps `per` exactly (so a stage that
        owns nothing here changes nothing, bit for bit)."""
        things = {}
        for sx, sy, sz in struck:
            c = (int(sx), int(sy), int(sz))
            if self.mat[c] == FLESH:         # a person went first, in `_blow`
                continue
            e = int(self.eid[c])
            if e:
                things.setdefault(e, []).append(c)
        if not things:
            return None
        striker = None if actor_pid is None else next(
            (r for r in self.persons if r["pid"] == actor_pid), None)
        left = {}
        for eid, cells in things.items():
            have = per * len(cells)
            j = have
            at = np.array(cells, np.int64)
            # what a stage does to the thing (a door swings, a blade parts) is
            # caused as a shatter of the same cells would be
            with self.because(fall_cause, "call") if fall_cause is not None \
                    else self.because(self._cause_near(cells[0])):
                for stage in stages:
                    j = min(float(stage(eid, at, j, a_hit, None, striker)), j)
                    if j <= 0.0:
                        j = 0.0
                        break
            if j < have:
                share = j / len(cells)
                for c in cells:
                    left[c] = share
        return left or None

    def _add_to_deed(self, q, cells):
        """Write landed flesh into its owner's deed: his deed is now all of
        his flesh on the lattice, read by its eid.

        A man can come down in more than one landing: a swung arm, a body
        that toppled from the waist while his legs stood, two free bodies of
        his in one fall. Each landing used to write the deed as exactly what
        IT put down, so the last one wiped out the others and the rest of him
        belonged to nobody (roadmap 87); then each landing added what it put
        down, and a neighbour's arm he had been launched carrying came down
        as his (E6, fuzzer seed 108). The eid on the flesh says whose it is,
        so neither can happen: what he lands is his only if it is his, and a
        held tool that came down with him is never his. `cells` (what this
        landing put down) is kept for the callers; the eid already knows."""
        got = np.argwhere(self.body_by_eid(self._eid_of(q)))
        q["_own"] = frozenset(zip(got[:, 0], got[:, 1], got[:, 2]))

    def _in_the_way(self, pose):
        """How many cells of a landing pose are not air, or not in the world."""
        nx, ny, nz = self.shape
        inb = ((pose[:, 0] >= 0) & (pose[:, 0] < nx) & (pose[:, 1] >= 0)
               & (pose[:, 1] < ny) & (pose[:, 2] >= 0) & (pose[:, 2] < nz))
        return int((~inb).sum()) + int(
            (self.mat[tuple(pose[inb].T)] != AIR).sum())

    @staticmethod
    def _off_body(body, limb):
        """Would this limb, put here, be a piece of its own, off the body?"""
        return _pieces(np.vstack([body, limb])) > _pieces(body)

    def _touches(self, body, part):
        """Does any cell of `part` touch a cell of `body`, face to face?"""
        nx, ny, nz = self.shape
        if not len(body) or not len(part):
            return False
        kb = (body[:, 0] * ny + body[:, 1]) * nz + body[:, 2]
        nb = (part[None, :, :] + np.array(_SIX)[:, None, :]).reshape(-1, 3)
        nb = nb[((nb >= 0) & (nb < (nx, ny, nz))).all(axis=1)]
        return bool(np.isin((nb[:, 0] * ny + nb[:, 1]) * nz + nb[:, 2], kb).any())

    def _slide(self):
        """How far, in cells, a body coming down may be moved sideways to
        come down whole: about 30 cm, a slack body shoved off what it hit,
        and never less than 3 cells."""
        return max(3, int(round(0.3 / (0.1 * self.scale))))

    _MOVES = {}

    @staticmethod
    def _moves(r, up, down=0):
        """Every whole-cell move (dx, dy, dz) with |dx|, |dy| and |dz| no
        more than `r` sideways, `up` up and `down` down, nearest first; at
        the same distance a slide before a lift, and a lift before a drop.
        A whole-cell move moves every voxel the same way, so a body moved by
        one is in exactly the pieces it was in."""
        key = (int(r), int(up), int(down))
        m = BodiesMixin._MOVES.get(key)
        if m is None:
            g = np.mgrid[-key[0]:key[0] + 1, -key[0]:key[0] + 1,
                         -key[2]:key[1] + 1].reshape(3, -1).T
            o = np.lexsort((g[:, 1], g[:, 0], -g[:, 2], np.abs(g[:, 2]),
                            (g * g).sum(axis=1)))
            m = BodiesMixin._MOVES[key] = g[o]
        return m

    def _clear_move(self, pose, moves, ok=None):
        """The first of `moves` that puts every cell of `pose` in air inside
        the world, and that `ok` (if given) accepts; None if none does."""
        nx, ny, nz = self.shape
        pose = np.asarray(pose, np.int64)
        hi = np.array((nx, ny, nz))
        for a in range(0, len(moves), 256):
            mv = moves[a:a + 256]
            at = pose[None, :, :] + mv[:, None, :]
            inb = ((at >= 0) & (at < hi)).all(axis=2)
            c = np.minimum(np.maximum(at, 0), hi - 1)
            free = inb & (self.mat[c[..., 0], c[..., 1], c[..., 2]] == AIR)
            for k in np.flatnonzero(free.all(axis=1)):
                if ok is None or ok(mv[k]):
                    return mv[k]
        return None

    def _lay_down(self, b, theta):
        """Where a toppled body's cells go when it comes to rest at `theta`.

        A BODY COMES DOWN IN ONE PIECE. Turned and then rounded to the
        lattice, two of its voxels could round into one cell (the second
        piled up as debris) and two neighbours could round apart: a man who
        keeled over got up with a detached fifteen-voxel foot beside him, and
        the claim, which floods one lump, could not hold him together again.
        Measured on one standing body: 3 to 18 pieces at every lean between
        0.05 and 1.4 rad.

        So the turn is the skeleton's three shears (`_shear_steps`), which
        move every row onto a whole row and so can never put two voxels in
        one cell. They can still part a limb one voxel thin into a staircase
        (a leg leaning across the shoulder line is in pieces at every angle
        but nought and pi/2), so a pose is taken only if it is in no more
        pieces than the body was when it started to fall. Otherwise it lands
        upright (0) or flat (pi/2), the nearer first: a quarter turn about a
        whole-cell pivot is exact, and upright is where it came from.

        And a pose with nothing in the way beats one with a graze in it: a
        cell that lands in the floor or the wall is piled up its column as
        debris, or lost past the edge of the world, and a foot cell piled up
        is a foot cell somewhere else.

        FLAT IS FLAT ON THE FLOOR, not through it. Turned about his feet, a
        man whose shoulders are wider than his stance puts a shoulder below
        the floor at pi/2: that is exactly why the fall stopped short of pi/2.
        Measured on a body falling across its shoulder line: flat was 36 to
        54 cells into the floor every time, so upright always won and an
        unconscious man was stood up where he fell and keeled over again
        every 8 ticks, in an open room. So a pose that dips below the level
        it fell from is lifted, a whole cell at a time, until it rests on
        what it met. A whole-cell lift moves every voxel the same way, so it
        cannot break the body; and it never lifts him above that level."""
        ax, s = int(b["axis"]), int(b["s"])
        piv = np.zeros(3)
        piv[ax], piv[2] = float(b["pivot"]), float(b["zb"])
        flesh = b["mats"] == FLESH                # what must stay one lump
        if not flesh.any():
            flesh = np.ones(len(flesh), bool)
        # EACH BODY IN ONE PIECE, by its eid. A lump that toppled can be two
        # people, and two people lying in contact are one lump to a flood
        # that cannot tell them apart; the pose that keeps each of them whole
        # is the one to take. One body (the usual case) is exactly as before.
        who = b["eids"][flesh]
        groups = [flesh & (b["eids"] == e) for e in np.unique(who)] \
            if len(np.unique(who)) > 1 else [flesh]
        tol = max(2, len(b["cells"]) // 50)       # the topple's own graze

        def at(th):
            return self._apply_shears(self._shear_steps(piv, th, ax, s),
                                      b["cells"])

        def pieces(q):
            return sum(_pieces(q[g]) for g in groups)

        def rest(q):                           # flat on the floor, not in it
            for dz in range(int(round(piv[2])) - int(q[:, 2].min()) + 1):
                if self._in_the_way(q + (0, 0, dz)) == 0:
                    return q + (0, 0, dz)
            return q

        # DOWN THE WALL, FEET SLIDING OUT. A man who falls toward a wall
        # nearer than his height cannot lie flat where he fell: flat puts his
        # head in the wall. Upright was the only clear pose left, so he was
        # stood back up and collapsed again every 24 ticks, for ever (the
        # rescue scene: aloft when the rescuer left at t83, so not saved).
        # What a slack body does is slump down the wall while its feet slide
        # back out, so a pose that is still in the way is moved back from
        # where it fell, a whole cell at a time (which cannot break it), at
        # most his own height, and lifted onto what it meets.
        reach = int(np.ptp(b["cells"][:, 2])) + 1

        def slump(q):
            ahead = np.sign(float(q[:, ax].mean()) - piv[ax])
            if not ahead or self._in_the_way(q) == 0:
                return q
            back = np.zeros(3, np.int64)
            back[ax] = -int(ahead)
            for d in range(1, reach + 1):
                r = rest(q + back * d)
                if self._in_the_way(r) == 0:
                    return r
            return q

        start = pieces(at(0.0))
        th0 = 0.0 if theta is None else float(theta)
        tries = [th0] + sorted((0.0, np.pi / 2), key=lambda t: abs(t - th0))
        poses = [slump(rest(at(t))) if t else rest(at(t)) for t in tries]
        lump = [pieces(q) <= start for q in poses]
        way = [self._in_the_way(q) for q in poses]

        def take(t, q):
            # PROPPED ON WHAT STOPPED HIM. A topple ends short of flat only
            # when something is in the way, so a body that lands leaning, or
            # stood back up because there was no room to lie down even
            # sliding down the walls, is held up by the room. The landing
            # says so (`_propped`, physiology), or he is collapsed again
            # every 8 ticks for as long as he is out (fuzzer seed 9: a man 30
            # cells tall in a room 26 by 18; seed 48: a lean, then a topple
            # blocked before it began).
            b["propped"] = t < np.pi / 2 - 1e-6
            b["propped_way"] = (ax, s)    # (the way he was going over)
            return q
        for t, q, ok, n in zip(tries, poses, lump, way):
            if ok and n == 0:
                return take(t, q)
        # NOTHING CLEAR WHERE HE FELL, so he is moved until he fits: a whole
        # cell at a time, sideways or up, the nearest move first. A graze
        # used to be taken here, and a graze is a cell of him landing in the
        # floor or a wall, which is piled up its column as debris: a foot
        # cell piled up is a foot cell somewhere else, and a man in pieces
        # (fuzzer seeds 5 and 43: [369, 27] for 227 ticks, [376, 10, 10]).
        # Up as far as `rest` may lift him, and a little more, and sideways
        # as far as a slack body slides when it is shoved off what it hit
        # (`_slide`: seed 43 needed 20 cm, beside the man who had stepped
        # into the place he fell from).
        for t, q, ok in zip(tries, poses, lump):
            if ok:
                up = max(int(round(piv[2])) - int(q[:, 2].min()) + 1, 0) + 2
                mv = self._clear_move(q, self._moves(self._slide(), up))
                if mv is not None:
                    return take(t, q + mv)
        for cap in (tol, None):                # then a graze, then any
            for t, q, ok, n in zip(tries, poses, lump, way):
                if ok and (cap is None or n <= cap):
                    return take(t, q)          # in pieces is worse than piled
        return poses[0]

    def _pay_swing(self, b, w_0, w_1):
        """The muscle's work in one tick of a swing, paid from its owner's books as it
        is done (mechlaw._pay_strike: stores at the efficiency of muscle, or the limb's
        reserve). The kinetic energy the arm gained is 1/2 I (w1^2 - w0^2); a body that
        cannot pay all of it is weaker for it (the speed is the one the paid joules
        buy). The paid joules wait in b["paid_J"] and are deposited as heat when the
        swing ends (`_book_swing`). Returns the speed after the tick."""
        dW = 0.5 * b["I"] * (w_1 * w_1 - w_0 * w_0)
        if dW <= 0.0:
            return w_1
        who = next((q for q in self.persons if q["name"] == b.get("owner")), None)
        if who is None:
            return w_1
        sc = self._pay_strike(who, dW, "swing", kind=b.get("exkind", "punch"))
        if sc < 1.0:
            w_1 = math.sqrt(max(w_0 * w_0 + 2.0 * dW * sc / b["I"], 0.0))
        b["paid_J"] += dW * sc
        return w_1

    def _book_swing(self, b, at):
        """The swing is over (the arm landed, or the thing was let go): every joule
        its muscle was billed for becomes heat where it stopped, credited once to the
        person who did the work (the one-flow rule: the actor, not mech_in)."""
        J = float(b.get("paid_J", 0.0))
        b["paid_J"] = 0.0
        who = next((q for q in self.persons if q["name"] == b.get("owner")), None)
        if J <= 0.0 or who is None:
            return
        cells = np.round(np.asarray(at, dtype=np.float64)).astype(np.int64).reshape(-1, 3)
        nx, ny, nz = self.shape
        ok = ((cells >= 0) & (cells < (nx, ny, nz))).all(axis=1)
        cells = cells[ok]
        if not len(cells):                       # off the lattice: where the body is
            cells = np.round(np.asarray(b.get("body_at", (0, 0, 0)), np.float64)).astype(np.int64).reshape(-1, 3)
        self.deposit(cells, J, actor=who)

    def _land_body(self, b, theta, struck=None):
        """Demote: rasterize the body back onto the grid at its landing pose.
        Blocked cells pile upward as debris; the support law settles the rest.
        A landing at a barely-leant angle means the body is WEDGED — the torque
        law sleeps briefly so it retries occasionally, not every tick."""
        # WEDGED IS NOT A NUMBER NEAR ZERO, it is a topple that did not get
        # over. A real fall goes the whole way to pi/2 and lands flat; anything
        # that stops short of halfway has met something. The threshold used to
        # be 0.15, which caught only the ones that never started — a vessel on
        # a shelf edge stopped dead at 0.41 every time, was set down, was found
        # unbalanced again on the next tick, and toppled again, for ever. It
        # cost alchemist 862 saved frames instead of 130 and nothing else.
        if theta is not None and theta < 0.7:    # a flying body has no lean to
            self._torque_sleep = max(self._torque_sleep, 20)   # be wedged at
        if b.get("paid_J"):
            self._book_swing(b, b["cells"] if struck is None or not len(struck) else struck)
        if b.get("part"):
            # A LIMB IS HELD AT ITS JOINT. Rasterising it wherever the swing
            # stopped leaves the far end nowhere near the body, and a lump of
            # flesh that touches nothing is not an arm any more — measured, a
            # person went from one piece to three and left 18 voxels of hand on
            # the floor. Hanging IS the rest state of something pivoted at one
            # end, so the arm comes back to it once the blow is spent. The
            # swing is a transient; what it is attached to is not.
            pose = np.round(b["cells"]).astype(np.int64)
            # AND THE JOINT IS WHERE THE BODY IS NOW. A man can take a step
            # while his arm is out, and an arm put back where it left from
            # landed a step away from him: measured on fuzzer seed 134, a
            # 54-voxel arm lying on the floor beside its owner for the rest of
            # the run. So it may come back moved by as much as the rest of him
            # moved, read off his deed, which is his body without the arm.
            # That reading is only as good as the deed, so the move is taken
            # only when it is better: the arm on the body beats the arm off
            # it, and then fewer cells in the way; a tie stays where it was.
            q = next((r for r in self.persons if r["name"] == b.get("owner")
                      and r.get("_own")), None)
            bases = [pose]
            if q is not None and b.get("body_at") is not None:
                now = np.array([c for c in q["_own"] if self.mat[c] == FLESH],
                               np.int64).reshape(-1, 3)
                step = np.rint(now.mean(axis=0) - b["body_at"]).astype(
                    np.int64) if len(now) else np.zeros(3, np.int64)
                if step.any():
                    bases.append(pose + step)
                    pose = min((pose, pose + step), key=lambda c: (
                        self._off_body(now, c[b["mats"] == FLESH]),
                        self._in_the_way(c)))
            # AND IT COMES BACK WHOLE, ONTO HIM. Where it left from can be
            # taken by now (he turned, crouched or stepped into it, or
            # something fell there), and the cells in the way were piled up
            # their columns as debris. Or he has moved, and an arm that was
            # raised when it left (drawn as two strips that each meet only
            # his shoulder) came back beside him as two halves of an arm
            # (fuzzer seeds 2, 12, 19, 36: [342, 27, 27]). So it comes back
            # moved by a cell or two, the nearest move that is clear and has
            # every piece of it touching him; only when there is none is it
            # put down as before.
            fl = b["mats"] == FLESH
            if q is not None and fl.any():
                own = np.array([c for c in q["_own"] if self.mat[c] == FLESH],
                               np.int64).reshape(-1, 3)
                # ON HIM is every piece of it touching him: an arm drawn
                # raised can be two strips that each meet the shoulder and
                # not each other, and one of them touching is half an arm on.
                lab = _labels(np.round(b["cells"][fl]))

                def on(c):
                    return all(self._touches(own, c[fl][lab == i])
                               for i in range(int(lab.max()) + 1))
                if len(own) and not (self._in_the_way(pose) == 0 and on(pose)):
                    best = None
                    for base in bases:
                        mv = self._clear_move(base, self._moves(2, 2, 2),
                                              lambda m, base=base: on(base + m))
                        if mv is not None and (best is None or
                                               (mv * mv).sum() < (best * best).sum()):
                            best, pose = mv, base + mv
        elif b.get("fly") or not any(q["name"] == b.get("owner")
                                     for q in self.persons):
            # a thing, not a person: an owner that names no one here (a
            # topple pushed by "nobody") is a thing, and it turns as a thing,
            # past the edge of the world if that is where it goes
            pose = np.round(self._fly_pose(b) if b.get("fly")
                            else self._body_pose(b, theta)).astype(np.int64)
            # A MAN COMING DOWN FROM A JUMP COMES DOWN WHOLE. The flight
            # stops at its last clear pose, but the room can change under it
            # in a tick (somebody steps in, something falls there), and the
            # edge of the world is a wall to a man, not a door. The cells in
            # the way were piled up as debris or lost past the edge, which
            # puts him down in pieces. So he is moved, the nearest whole cell
            # move that is clear; only when there is none is he piled.
            if b.get("fly") and b.get("owner") is not None \
                    and any(q["name"] == b["owner"] for q in self.persons) \
                    and self._in_the_way(pose):
                mv = self._clear_move(pose, self._moves(self._slide(), 3))
                if mv is not None:
                    pose = pose + mv
        else:
            pose = self._lay_down(b, theta)          # somebody's body
        nx, ny, nz = self.shape
        put = []
        # IMPACT: the body's centre of mass FELL — that energy is paid out over
        # the cells that strike first, each judged by its own material's
        # toughness. A keeling vat of glass bursts on the shelf edge; a keeling
        # tree of wood just booms.
        vox_m = 0.1 * self.scale
        mtot = max(float(b["masses"].sum()), 1e-6)
        if b.get("Nm"):
            # A SWING ARRIVES WITH ITS ROTATIONAL ENERGY, and it arrives on
            # WHAT STOPPED IT — not on whatever happens to be underneath. A
            # falling thing is judged by the floor it meets; a swung thing is
            # judged by the thing it hit, which is the only difference between
            # leaning on an axe and swinging one.
            w_s = b["omega"] / max(TICK_S, 1e-9)          # rad/tick -> rad/s
            joules = 0.5 * b["I"] * w_s * w_s
        elif b.get("fly"):
            # IT ARRIVES WITH WHAT IT IS CARRYING. A flying body's blow is its
            # kinetic energy, not the height it happens to have lost — which is
            # the whole difference between a stone dropped on a plank and the
            # same stone thrown at it, and the difference between leaning on an
            # axe and swinging one.
            v = float(np.linalg.norm(b["vel"]))
            joules = 0.5 * (mtot / 1000.0) * v * v
        else:
            dh = float(((b["masses"] * b["cells"][:, 2]).sum()
                        - (b["masses"] * pose[:, 2].astype(np.float32)).sum()) / mtot) * vox_m
            joules = max(dh, 0.0) * 9.81 * (mtot / 1000.0)
        # A BLOW WOUNDS what it lands on and what lands. Flesh does not shatter
        # — TOUGH[FLESH] is a deforms-not-fragments number — but tissue takes
        # real damage far below that, and the same energy-per-contact-area
        # arithmetic that bursts glass is what breaks a leg. Anything beyond
        # the bruise threshold is absorbed by the BODY, not the cell.
        bruise = BODY["bruise_kJm2"] * 1000.0 * vox_m ** 2
        if struck is not None and len(struck):
            self._spend(b, struck, joules)
        if b.get("part"):
            # the blow has already been spent on what stopped it. An arm does
            # take the same impulse back — Newton's third — but paying the
            # whole energy TWICE, once to the post and once to the hand, is not
            # that law, it is double counting.
            joules = 0.0
        contact = pose[:, 2] <= pose[:, 2].min() + 1
        # A ROLL IS NOT ONE LANDING. It is several, each taking a share of the
        # fall on a different part of you — feet, then hip, then back, then
        # shoulder — and tissue damage is a THRESHOLD, so a blow divided into
        # parts that each fall under it does no harm at all while the energy is
        # unchanged. That is force being momentum divided by the time taken to
        # lose it, written in the arithmetic this model already has.
        #
        # How many parts is read off the body: how many times its own crouch
        # goes into its own length, which is how far a body rolling over itself
        # has to travel before it stops. Nothing typed in.
        shares = 1
        if b.get("owner") is not None and not b.get("part"):
            who = next((q for q in self.persons
                        if q["name"] == b["owner"]), None)
            if who is not None and who.get("rolling"):
                tall = (float(pose[:, 2].max() - pose[:, 2].min()) + 1.0) * vox_m
                shares = max(1, int(round(
                    tall / max(self._crouch_of(who), 1e-9))))
                who["rolling"] = False
                self.emit("rolls", actor=who["pid"], result="with_it",
                          text=f"{who['name']} rolls with it",
                          to=who, site="land:rolls_with_it")
        lands_box = None
        if joules > 0.0 and len(pose):
            lo = pose[np.argmin(pose[:, 2])]
            under = (int(lo[0]), int(lo[1]), max(int(lo[2]) - 1, 0))
            if all(0 <= under[i] < self.shape[i] for i in range(3)):
                self._noise(lo, joules, [int(m) for m in set(
                    int(x) for x in b["mats"][contact])]
                    + [int(self.mat[under])], who=b.get("owner"),
                    pid=self._body_pid(b))
            owner_pid2 = next((r["pid"] for r in self.persons
                               if r["name"] == b.get("owner")), None)
            # DEFERRED (`_defer`, not `emit`): a landing is common enough
            # that emitting it right away would, somewhere, land in the
            # same tick as a shout and shift the id `_say` stamps on its
            # own sound record for no reason of its own (see `_defer`).
            lands_box = self._defer("lands", actor=owner_pid2,
                                     place=tuple(int(v) for v in lo),
                                     db=float(joules), text="")
        e_per = joules / float(shares) / max(int(contact.sum()), 1)
        sore = 0.0                          # energy the faller's own flesh took
        order = np.argsort(pose[:, 2])      # kept: put[k] belongs to kept[k],
        kept = []                           # so a landing can be told apart
        for i in order:                     # into flesh and carried tool
            # THE WORLD HAS EDGES, AND WHAT GOES PAST THEM IS OUTSIDE IT.
            # These three lines used to CLAMP: a cell thrown past the wall was
            # set down on the wall instead, several of them into the same
            # column, and each `smass =` overwrote the last. Mass went missing,
            # quietly, in the one field every test in the suite leans on.
            #
            # Clamping was never right anyway — it teleports matter. A thing
            # that leaves is gone, and saying so keeps the books: it is on the
            # lattice, or in a body, or on this tally, and the three add up.
            dx, dy, dz = int(pose[i, 0]), int(pose[i, 1]), int(pose[i, 2])
            if not (0 <= dx < nx and 0 <= dy < ny and 0 <= dz < nz):
                self._left_world(b, i)
                continue
            while dz < nz - 1 and self.mat[dx, dy, dz] != AIR:
                dz += 1                                  # blocked: pile upward as debris
            if self.mat[dx, dy, dz] != AIR:
                self._left_world(b, i)       # the column is full to the ceiling
                continue                     # and there is nowhere left to put it
            put.append((dx, dy, dz))
            kept.append(i)
            # IT ARRIVES IN A CELL OF AIR, and what the air held stays: its
            # heat is added to, and water standing there stays unless the
            # cell brings its own. The fall is paid here, by this landing, so
            # nothing of it is left in `fallh` for the impact law to pay again.
            c = (dx, dy, dz)
            got = self._of_body(b, i)
            v = list(got)
            v[_E] = self.E[c] + v[_E]
            if not b["fvols"][i] > 0:
                v[_FL], v[_FVOL], v[_FPOT] = self.fl[c], self.fvol[c], self.fpot[c]
                if "fcomp" in got.opt:       # the water stays, and keeps its mix
                    got.opt["fcomp"] = self.fcomp[c]
            v[_FALLH] = 0.0
            self._put(c, v, got.opt)         # (the thing index: below)
            if contact[i]:
                if b["mats"][i] == FLESH:
                    if e_per > bruise:      # flesh deforms, never fragments:
                        sore += (e_per - bruise) * shares   # the overage, once
                    continue                                # per share of the
                                                            # blow, is the wound
                thr_i = (_TOUGH_ARR[b["mats"][i]] * 1000.0 * vox_m ** 2
                         * (0.7 + 0.6 * self._flaw01(dx, dy, dz)))
                if self.dmg is not None:     # T9: a cracked cell breaks at thr x (1 - D)
                    thr_i = shatter_threshold(thr_i, self.dmg[dx, dy, dz])
                if e_per > thr_i:
                    with self.because(lands_box) if lands_box is not None \
                            else self.because(self._cause_near((dx, dy, dz))):
                        self._shatter(dx, dy, dz, e_per / thr_i)
                elif "damage" in self.mat_laws:      # and a landing that did not break it cracked it
                    matlaws.impact_damage(self, (dx, dy, dz), float(e_per), vox_m ** 2)
        if b.get("owner") is not None and not b.get("part"):
            # A FALL HURTS. _land_body always paid 1/2 m v² against what a body
            # STRUCK and nothing against the body — a man dropped onto stone
            # landed whole and entirely unbothered, and "pulled off a cliff"
            # was a change of address. The energy his own contact cells took
            # is his, by the same third-law sharing as the struck side.
            for q in self.persons:
                if q["name"] == b["owner"]:
                    with self.because(lands_box) if lands_box is not None \
                            else self.because(None):
                        if sore > 0.0:
                            self._wound(q, sore)
                        self._injure_landing(q, joules, pose, contact,
                                             shares, lands_box)
                    if sore / BODY["hurt_J"] >= 0.02:
                        # SYNCHRONOUS, not deferred: the legacy text has to
                        # land in `q["events"]` exactly where this call is,
                        # byte for byte, so this one keeps its own id (no
                        # cause: the generic `lands_box` above is not
                        # resolved yet, and guessing at an id that does
                        # not exist is worse than naming none).
                        self.emit("lands", actor=q["pid"], result="hard",
                                  text=f"{q['name']} lands hard",
                                  to=q, site="land:lands_hard")
                    break
        self._torque_solid = None                        # lattice changed: recheck
        self._things_moved(b["eids"])
        if b.get("part") and b.get("seg") and b.get("owner") is not None:
            # EVERY CELL OF IT MAY HAVE LEFT THE WORLD — a limb swung off the
            # edge comes back as nothing, and `np.array([])` is not a list of
            # coordinates. Say the shape or the filter below has nothing to
            # filter.
            put_arr = np.array(put, np.int64).reshape(-1, 3)
            mats_put = b["mats"][np.array(kept, np.int64)] \
                if kept else b["mats"][:0]
            for q in self.persons:            # the limb came to rest somewhere
                if q["name"] == b["owner"] and q.get("segs") and q.get("_own"):
                    # AND IT IS HIS AGAIN. The claims made while it was in the
                    # air wrote his deed without it, and nothing said he was
                    # short when it came back, so the next claim gave it to
                    # whoever grew into it first. Before the corner is read:
                    # the arm is part of the body every offset is taken from.
                    self._add_to_deed(q, put_arr[mats_put == FLESH])
                    q["_claim_tick"] = None   # and this tick's claim lacks it:
                                              # a step later in the same tick
                                              # walked off and left it behind
                    origin = np.min(np.array(list(q["_own"])), axis=0)
                    # only the FLESH is the limb — a swung tool came along for
                    # the ride and is not part of anyone's arm
                    # EACH BONE BACK TO ITS OWN OFFSETS. Writing the whole
                    # arm under the arm's name leaves the bones holding the
                    # cells they were in before the swing — which are empty —
                    # and the next pose moves nothing and loses the limb.
                    seg_put = b["segof"][np.array(kept, np.int64)] \
                        if b.get("segof") is not None and kept \
                        else np.where(mats_put == FLESH, 0, -1)
                    for i, bone in enumerate(b.get("bones") or [b["seg"]]):
                        q["segs"][bone] = put_arr[seg_put == i] - origin
                    held = q.get("held")
                    if held and held.get("eid"):
                        # A NAMED THING IS HELD BY WHICH THING IT IS (3.3).
                        # The grip goes back to the matter it was on: the
                        # first cell of the thing in the swing is the grip's
                        # (`_held_cells` puts it first, and `_swing` keeps the
                        # order). Failing that, to any cell of the thing. The
                        # first cell that is not flesh, which is all an
                        # unnamed grip has, need not be the iron the hand
                        # closed on when the axe is iron and wood.
                        k = np.array(kept, np.int64)
                        eids_put = b["eids"][k] if kept else b["eids"][:0]
                        mine = np.flatnonzero(eids_put == held["eid"])
                        if b.get("segof") is not None and len(mine):
                            grip = np.flatnonzero((b["segof"] == -1)
                                                  & (b["eids"] == held["eid"]))
                            at = np.flatnonzero(k[mine] == grip[0]) \
                                if len(grip) else ()
                            if len(at):
                                mine = mine[at]
                        if len(mine):
                            c = tuple(map(int, put_arr[mine[0]]))
                            held["cell"], held["mat"] = c, int(self.mat[c])
                            held["label"] = MATNAME.get(held["mat"], "thing")
                    elif held and (mats_put != FLESH).any():
                        q["held"]["cell"] = tuple(map(
                            int, put_arr[mats_put != FLESH][0]))
                    break                     # say WHERE, or the next swing
                                              # reaches for cells that have gone
        put_all = np.array(put, np.int64).reshape(-1, 3)
        k_all = np.array(kept, np.int64)
        eids_all = b["eids"][k_all] if kept else b["eids"][:0]
        landed_fl = (b["mats"][k_all] == FLESH) if kept \
            else np.zeros(0, bool)

        def theirs(q):
            """The cells this landing put that are somebody else's flesh."""
            return landed_fl & (eids_all != 0) & (eids_all != q.get("eid", 0))
        if b.get("owner") is not None and not b.get("part"):   # hand the person
            for q in self.persons:                       # their own body, exactly
                if q["name"] == b["owner"]:              # where it came down
                    if not put:
                        # A LANDING CAN PLACE NOTHING, and it is not the end of
                        # him. The limb branch above already knows the shape of
                        # this — "a limb swung off the edge comes back as
                        # nothing" — and the branch for the man it belongs to
                        # went on to take the mean of those nothing cells. The
                        # mean of an empty list is NaN and `int(NaN)` raises:
                        # one world in 120 died here, at tick 1, on a body
                        # travelling fast enough that every cell of its landing
                        # pose was off the lattice.
                        #
                        # He keeps the anchor he had. The first version of this
                        # guard reasoned that an empty landing meant he had
                        # gone past the edge for good and marked him dead —
                        # measured, and wrong: the body was still in
                        # `self.bodies` with all 291 voxels, `gone` was empty,
                        # and forty ticks later it came down and put 41 of them
                        # back on the lattice. An empty landing is a landing
                        # that did not take, and the flight carries on.
                        #
                        # AND IT MUST NOT TOUCH THE DEED EITHER. Writing
                        # `_own = frozenset(put)` before this check wiped his
                        # ownership with an empty set, which is how a man who
                        # was halfway through landing ended up with no claim on
                        # the voxels he had already put on the floor — and the
                        # neighbour took them. Placing nothing changes nothing.
                        break
                    # THE DEED IS ADDED TO, NOT TAKEN WHOLESALE (roadmap 87).
                    # A person can own more than one free body, so this runs
                    # more than once for the same man, and each run used to
                    # replace the deed: a second body of his carrying no flesh
                    # wiped the good deed his real body had just set (harvest
                    # seed 34: 692 voxels against 25). It also deeded the tool
                    # he was holding. Now each landing adds its flesh.
                    #
                    # This fix was built once before and taken back out: with
                    # the deed right, two men who came down interleaved had no
                    # footing and were launched again every third tick, and
                    # the wrong deed had hidden that by giving them one lump.
                    # What changed since is the landing itself: it no longer
                    # shreds a body (`_lay_down`), and the claim no longer
                    # lets one man grow into another's missing cells.
                    self._add_to_deed(q, np.array(put, np.int64))
                    q["_claim_tick"] = None
                    q["_landed"] = self.tick   # legs have to gather again
                    # where he was left propped, if the walls hold him up:
                    # the corner of his flesh, which physiology reads back
                    own = np.array(list(q["_own"]), np.int64).reshape(-1, 3)
                    q["_propped"] = (tuple(int(v) for v in own.min(axis=0))
                                     if b.get("propped") and len(own) else None)
                    if b.get("propped") and b.get("propped_way"):
                        q["_propped_way"] = b["propped_way"]
                    else:
                        q.pop("_propped_way", None)
                    # where he came down: what this landing put, less any
                    # flesh of somebody else's that came down with him
                    # (and less what he held: a stone in his hand is not
                    # where HE is, `_with_held`)
                    his = ~theirs(q)
                    if b.get("held_from") is not None:
                        his &= k_all < b["held_from"]
                    at = put_all[his] if his.any() else put_all
                    q["anchor"] = (int(round(float(np.mean(at[:, 0])))),
                                   int(round(float(np.mean(at[:, 1])))))
                    # AND WHAT HE HELD CAME DOWN WITH HIM (`_with_held`): the
                    # grip goes where the cell in his hand went. If that one
                    # cell did not land (past the edge of the world) but the
                    # rest of the thing did, the hand is on the rest of it,
                    # found by its eid as the limb's landing above does: a
                    # grip is not lost because one cell of the thing was.
                    # Only a thing with none of it left in the world is let
                    # go (`_held_cells` finds it gone), and that is a fact,
                    # not a choice the hand made.
                    if b.get("grip") is not None and q.get("held"):
                        g = np.flatnonzero(k_all == b["grip"])
                        e = q["held"].get("eid") or 0
                        if len(g):
                            q["held"]["cell"] = tuple(map(int, put_all[g[0]]))
                        elif e and (eids_all == e).any():
                            c = tuple(map(int, put_all[eids_all == e][0]))
                            q["held"]["cell"], q["held"]["mat"] = \
                                c, int(self.mat[c])
                            q["held"]["label"] = MATNAME.get(
                                q["held"]["mat"], "thing")
                    # HE COMES DOWN WITH THE SPEED HE HAD. The footing law
                    # (`_law_footing`, the same tick) puts it into the turn he
                    # makes if the landing leaves his weight past what holds
                    # him up, as a push's speed (`push_person`); a landing
                    # that leaves him balanced spends it, as it always did.
                    if b.get("fly") and float(np.hypot(*b["vel"][:2])) > LAND_V:
                        q["_land_v"] = (float(b["vel"][0]), float(b["vel"][1]))
                    break
        if not b.get("part") and put:
            # AND ANYONE ELSE WHOSE FLESH CAME DOWN IN IT has landed too, where
            # his own flesh is: the owner of a body names one man, the eid on
            # the flesh names everyone in it (`_launch`).
            for q in self.persons:
                if q["name"] == b.get("owner") or not q.get("eid"):
                    continue
                his = landed_fl & (eids_all == q["eid"])
                if his.any():
                    q["_claim_tick"] = None
                    q["_landed"] = self.tick
                    q["anchor"] = (int(round(float(put_all[his, 0].mean()))),
                                   int(round(float(put_all[his, 1].mean()))))
        return put_all[landed_fl], eids_all[landed_fl]   # the flesh it put

    def _limbs_home(self, owner):
        """Bring every limb `owner` has out in a swing back onto him, now,
        and say which of his flesh that put back on the lattice ((n, 3)).

        A MAN WHO GOES OVER, OR JUMPS, DOES NOT KEEP SWINGING. The swung arm
        is a free body of its own, and it used to stay out while the rest of
        him left the lattice: it came back to where he had been, and he came
        down somewhere else, a man and an arm apart (fuzzer seeds 7, 27, 37:
        a jump in the tick of a swing, then [342, 54] for the rest of the
        run, carried as two pieces into every jump after). It is what the
        footing law already does to an arm held out in a reach: the arm
        comes down, all the way, at once. The swing's blow is not struck."""
        q = next((r for r in self.persons if r["name"] == owner), None)
        out = [np.zeros((0, 3), np.int64)]
        if q is None:
            return out[0]
        for b in [x for x in self.bodies
                  if x.get("part") and x.get("owner") == owner]:
            del self.bodies[next(i for i, x in enumerate(self.bodies)
                                 if x is b)]      # by identity: a body is a
                                                  # dict of arrays, not a value
            cells, eids = self._land_body(b, None)    # (no lean: a limb
            # has none to be wedged at, and the torque law is not put to sleep)
            if q.get("eid"):
                cells = cells[eids == q["eid"]]
            out.append(cells)
        return np.vstack(out)

    def _with_limbs(self, cells, owner):
        """`cells`, a body about to leave the lattice, and the flesh its swung
        limbs put back on it as it goes (`_limbs_home`). Each cell once: a
        claim made earlier in the tick still names the cells the arm swung
        out of, and the arm can come back into exactly those."""
        back = self._limbs_home(owner)
        if not len(back):
            return cells
        nx, ny, nz = self.shape
        c = np.asarray(cells)
        ci = np.rint(c).astype(np.int64)
        key = (ci[:, 0] * ny + ci[:, 1]) * nz + ci[:, 2]
        bk = (back[:, 0] * ny + back[:, 1]) * nz + back[:, 2]
        return np.vstack([c, back[~np.isin(bk, key)].astype(c.dtype)])

    def _with_held(self, cells, owner):
        """`cells`, a body about to leave the lattice, and the thing in its
        hand if the hand is what holds it up; and the row of the gripped cell
        in what is returned (None when nothing comes along).

        A HELD THING GOES WHERE THE HAND GOES. A man who jumped holding a
        stone left it in the air where his hand had been: it fell there, and
        his hand had let go of it without his ever deciding to (the grip
        named a cell the stone was no longer in). It leaves with him through
        the same `_lift` that takes him, and comes down with him through the
        same `_put` (`_land_body`), so every field it carries goes too.

        What the WORLD holds up stays, as it does for a reach (`_repose`): a
        post set in the ground, a block dragged along the floor. So does
        flesh: a hand on another body is not carrying it, and who goes with
        whom is the eid's business (`_launch`)."""
        q = next((r for r in self.persons if r["name"] == owner), None)
        held = (q or {}).get("held")
        if not held:
            return cells, None
        obj = self._held_cells(q)
        if obj is None or not len(obj):
            return cells, None
        c = np.asarray(cells)
        ci = np.rint(c).astype(np.int64)

        def grip(ri):
            """The row of the gripped cell in `ri`, or None. Asked of ALL
            that leaves, not only of what this adds: a gripped cell that
            was already among his cells leaves too, and the grip must go
            where it goes, or he lets go on landing without deciding to."""
            g = np.flatnonzero((ri == np.asarray(held["cell"], np.int64))
                               .all(axis=1))
            return int(g[0]) if len(g) else None
        if (self.mat[tuple(obj.T)] == FLESH).any() \
                or not self._in_reach(ci, obj) \
                or len(self._contact(obj, ignore=ci)):
            return cells, grip(ci)
        nx, ny, nz = self.shape
        key = lambda a: (a[:, 0] * ny + a[:, 1]) * nz + a[:, 2]
        new = obj[~np.isin(key(obj), key(ci))]
        if not len(new):
            return cells, grip(ci)
        ri = np.vstack([ci, new])
        return np.vstack([c, new.astype(c.dtype)]), grip(ri)

    def bodies_array(self):
        """Every mid-flight body voxel as (x, y, z, mat) for the renderer."""
        if not self.bodies:
            return np.zeros((0, 4), np.float32)
        parts = []
        for b in self.bodies:
            pose = self._fly_pose(b) if b.get("fly") \
                else self._body_pose(b, b["theta"])
            parts.append(np.column_stack([pose, b["mats"].astype(np.float32)]))
        return np.concatenate(parts).astype(np.float32)
