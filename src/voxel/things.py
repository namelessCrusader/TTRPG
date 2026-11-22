"""Things: which matter is one thing, what it is called, and what it is now.

Added by step S3 of docs/sim-mind-design.md so that the entity steps (E0 to
E4) can fill it in without another edit to sim.py. E0 put the first pieces
here: `_unmake`, and the every-tick hygiene in `_law_identity`. E1 added the
eid field and carries it wherever matter moves. E2 added the registry (a thing
can be made and named), the identity pass (a thing that comes apart becomes
its pieces, and a thing that is gone is dead, with a cause) and the names the
menus read. E3 made a hand hold a named thing by WHICH thing it is, made reach
join cells on the key of 3.3 (`_key`), and added the audit of what an author
says a thing is for. A world that registers nothing still steps exactly as it
did: every eid is 0, and every name is the text it always was (3.3).
"""
import numpy as np

from .label import label
from .records_world import RecordsMixin
from .materials import (ACID, AIR, AMBIENT, ASH, BLOOD, BODY, CHAR, FLESH,
                        FLUID, GRAVITY, MATNAME, MIRON, MLEAD, MTIN, OIL,
                        WATER, WEAK_ACID, _CSOLID_ARR, _FC_ARR, _FDENS_ARR,
                        _IGN_ARR)

# Identity verbs of 3.0 (ruling 4) that the closed table in events.py does not
# have yet. Spelled once, here, until it does.
SPLITS, ASHES = "splits", "ashes"

# What a thing that is nowhere died OF, by the verb that took its last matter
# (`_unmake`, `_left_world`, `_leave`). A thing that is gone and nothing said
# why is "lost": the census still records it, it only cannot say more.
_DIED = {"melts": "melted", "dissolves": "dissolved", "gone": "gone"}

# The temperature each solid catches at, for `burning_at`, is materials._IGN_ARR
# (inf for one that never does): the table `burning()` loops over, as one gather,
# NMAT_PAD long, so a cell of any id below it can be asked.

# What a fluid is called in `props(eid)["contains"]`.
_FLUIDNAME = {WATER: "water", OIL: "oil", ACID: "acid", WEAK_ACID: "vinegar",
              MTIN: "molten tin", MLEAD: "molten lead", MIRON: "molten iron",
              BLOOD: "blood"}

_ORDINAL = ("first", "second", "third", "fourth", "fifth", "sixth",
            "seventh", "eighth", "ninth", "tenth")

# What a word in `says["affords"]` needs the thing to BE, read off its own
# cells when it is made (`_made`). An author's word is a claim, not a fact: a
# "cut" on a thing with no edge goes to `id_audit` and never becomes a
# capability, and a word nothing here can check is not one either.
_NEEDS = {"cut": "sharp", "chop": "sharp", "slice": "sharp", "stab": "sharp",
          "pierce": "sharp", "burn": "flammable"}


def _with(called, word):
    """`called` with a state word after its first word: "the axe" and
    "burning" give "the burning axe", "a piece of the plank" and "charred"
    give "a charred piece of the plank"."""
    det, _, rest = called.partition(" ")
    return f"{det} {word} {rest}" if rest else f"{word} {called}"


class ThingsMixin(RecordsMixin):

    def _init_things(self):
        """Make the entity state of a new world. `__init__` calls it once,
        at its end, just before `_init_events`.

        `eid` is WHICH THING each cell is part of, beside `mat`: 0 is no
        thing. Every law that moves a cell's matter moves its eid with it.
        The rest is the registry and its bookkeeping (3.3):
          things      eid -> what that thing is (plain dicts, so snapshots
                      copy them; `register` says what is in one); an eid is
                      never used twice
          _thing_lut  eid -> "is a thing" (a person's eid is not). Long
                      enough to index by any eid in the field
          _next_eid   the next eid to give out
          _stamping   the eid `fill` writes (0: the matter is no thing)
          _thing_gen  bumped whenever a thing's cell moves or empties, so a
                      cache of where things are knows it is stale
          _id_dirty   the things whose cells moved, emptied or changed what
                      they are since the identity pass last looked
          _id_losses  (eid, cause) for matter a thing lost that nothing on
                      the lattice shows: it left the world (`_left_world`,
                      `_leave`) or was unmade (`_unmake`). The identity pass
                      reads it, so it can never miss a death, and empties it
          id_check    a test switch: the identity pass asserts the entity
                      layer's own invariants every tick (`_check_ids`)
          id_audit    what an author said a thing is that the thing is not
                      (`_made`), one dict a finding
          stamp_bodies  a switch, on: every person is registered when
                      added (`add_person`) and the scene stamps the body it
                      builds with that eid. A body IS the flesh with its eid
                      (`_claim_bodies`, E7). Off, a person is registered at
                      the first claim instead, and the claim adopts their
                      flesh then (`_adopt_flesh`), as it adopts any flesh
                      nobody stamped
          audit_bodies  a switch, off: when on, the claim writes to
                      `id_audit` each time it adopts flesh nobody stamped,
                      the one place a claim can still differ from the eid it
                      found (E6's audit, after E7)

        `_id_prev_mat` is the hygiene's memory: what `mat` was when
        `_law_identity` last ran. `reach_things_cap` and `_props` are E4's
        (below, where they are set)."""
        self.eid = np.zeros(self.shape, np.int32)
        self.things = {}
        self._thing_lut = np.zeros(1, bool)
        self._next_eid = 1
        self._stamping = 0
        self._thing_gen = 0
        self._thing_idx = None
        self._id_dirty = set()
        self._id_losses = []
        self.id_check = False
        self.id_audit = []
        self.stamp_bodies = True
        self.audit_bodies = False
        self._id_prev_mat = self.mat.copy()
        # How many things one look within reach may find (E4): the most
        # `_objects_within_reach` returns, or None for every one there is.
        # It was 2, always: a stop for speed, from when reach flooded once
        # per candidate cell, and it hid every thing past the second however
        # near it was. What attention costs is the menu's own cap
        # (`MENU_CAP`), which knows what else is on the menu; this does not.
        self.reach_things_cap = None
        # `props` keeps what it worked out until the tick or `_thing_gen`
        # moves: ((tick, gen), {eid: props})
        self._props = (None, {})

    # ── the registry (3.3) ──────────────────────────────────────────────────
    def register(self, name, kind="thing", parts=None, says=None, parent=0,
                 called=None, whole=None):
        """Make a new thing and return its eid. Nothing on the lattice is
        part of it until something is stamped with it (`stamp`, or `fill`
        while `_stamping` is the eid).

        `name` is the noun ("axe"). `parts` names what it is made of, part
        by part ({"head": IRON, "haft": WOOD}): a piece that breaks off and is
        one of them is called by it ("the haft of the axe"). `says` is what
        the author claims it is for, a claim `_made` audits. A
        "person" is registered too, but is not a thing: its eid stays out of
        `_thing_lut`, so nothing here splits, names or buries a body.

        `called` is what it is called before any state word ("the axe"), and
        `whole` what its pieces will be pieces OF; both default to "the" and
        the noun. The identity pass gives its children their own."""
        e = self._next_eid
        self._next_eid += 1
        if len(self._thing_lut) <= e:
            lut = np.zeros(max(2 * len(self._thing_lut), e + 1), bool)
            lut[:len(self._thing_lut)] = self._thing_lut
            self._thing_lut = lut
        self._thing_lut[e] = kind != "person"
        # A NAME THAT ALREADY HAS ITS ARTICLE keeps it: a thing an author
        # called "the oak" is "the oak", and its pieces "a piece of the oak",
        # not "a piece of the the oak". Not for a person: a person's
        # `called` is never read (`name` answers for things only), and it
        # is in every golden world that has someone in it ("the person").
        if not called:
            called = name if kind != "person" and name.lower().startswith(
                ("the ", "a ", "an ")) else "the " + name
        self.things[e] = {
            "name": name, "kind": kind, "parts": dict(parts or {}),
            "says": dict(says or {}), "called": called,
            "whole": whole or called, "parent": parent, "born": self.tick,
            "broken": False,     # it has lost a piece of itself
            "debris": 0,         # its one child that holds the loose bits
            "ashes": False,      # it has burned to nothing but ash
            "dead": None,        # what it died of ("melted", "gone", ...)
            "died": None}        # and when
        return e

    def stamp(self, eid, where):
        """Make the matter in `where` part of thing `eid`: a bool mask the
        shape of the world, or an (n, 3) array of cells. Air is no thing, and
        drawn particle matter (`moving`) is the solver's, so neither is
        stamped. Returns how many cells were."""
        w = np.asarray(where)
        if w.dtype == bool:
            idx = np.nonzero(w & (self.mat != AIR) & ~self.moving)
        else:
            c = w.reshape(-1, 3).astype(np.int64)
            c = c[(self.mat[tuple(c.T)] != AIR) & ~self.moving[tuple(c.T)]]
            idx = tuple(c.T)
        self._things_moved(self.eid[idx])
        self.eid[idx] = eid
        self._things_moved(self.eid[idx])
        if self._is_thing([eid])[0]:
            # what reach remembers as "the world, not a thing" was worked out
            # on the old keys, and a cell that has just become a thing may
            # not be the world any more (`_objects_within_reach`); and the
            # tip check joins cells on the key too (`_law_torque`), so the
            # lumps it last weighed may not be the lumps there are now
            self._bulk = None
            self._torque_solid = None
        return len(idx[0])

    def _is_thing(self, ids):
        """For each eid, whether it is a registered thing. An eid nobody
        registered is not one, and is not an error: a person's eid is never
        a thing, and matter can carry an eid stamped straight into `w.eid`
        (the solver's own tests do) before or without `register`."""
        ids = np.asarray(ids)
        lut = self._thing_lut
        out = np.zeros(ids.shape, bool)
        ok = ids < len(lut)
        out[ok] = lut[ids[ok]]
        return out

    def entity_at(self, x, y, z):
        """The thing the matter in this cell is part of, or 0: no thing, a
        person's flesh, or air."""
        e = int(self.eid[x, y, z])
        return e if e < len(self._thing_lut) and self._thing_lut[e] else 0

    def _made(self, e):
        """A thing an author has just finished making (`scenes.thing`): note
        what it weighed when it was made (`mass0`), and audit what the
        author said it is for. Each word of `says["affords"]` is checked
        against the thing's own cells (`_NEEDS`): one it lives up to is in
        `affords`, and one it does not ("cut" on a thing with no edge) goes
        to `id_audit` instead. A word nothing here can check is audited as
        unchecked. Either way an author's word alone is never a capability."""
        t = self.things[e]
        at = self._thing_cells().get(e)
        at = np.zeros(0, np.int64) if at is None else at
        t["mass0"] = float(self.smass.ravel()[at].sum())
        has = {"sharp": bool((self.edge.ravel()[at] > 0).any()),
               "flammable": bool(np.isfinite(
                   _IGN_ARR[self.mat.ravel()[at]]).any())}
        t["affords"] = []
        for word in t["says"].get("affords", ()):
            need = _NEEDS.get(word)
            if need is not None and has[need]:
                t["affords"].append(word)
                continue
            self.id_audit.append({
                "tick": self.tick, "eid": e, "says": word,
                "but": "unchecked" if need is None else "not " + need})

    def _body_eid(self, b):
        """The thing a body in flight is, read off its first cell (the cell
        its material word is read off too), or 0."""
        e = int(b["eids"][0]) if len(b["eids"]) else 0
        return e if e < len(self._thing_lut) and self._thing_lut[e] else 0

    def cells_of(self, eid):
        """Where thing `eid` is on the lattice, as an (n, 3) array of cells
        in flat order: read off the kept index (`_thing_cells`), so asking
        again costs nothing until something moves. Drawn particle matter is
        not counted (the particles are the thing there)."""
        at = self._thing_cells().get(int(eid))
        if at is None:
            return np.zeros((0, 3), np.int64)
        return np.stack(np.unravel_index(at, self.shape), -1)

    def _key(self, sl=None):
        """The join key of 3.3, over the box `sl` (the whole world when it is
        None): two touching solid cells are one thing exactly when their keys
        are equal. A thing's cell keys on its eid, so a named axe of iron and
        wood is one thing; every other cell (no thing, a person's flesh) keys
        on its material, as it always has, so an unnamed bar touching an
        unnamed iron post is still one lump with it.

        With no thing in the box the key IS `mat`, the same array: a world
        that names nothing joins cells exactly as it did."""
        mat = self.mat if sl is None else self.mat[sl]
        if not self._thing_lut.any():
            return mat
        e = self.eid if sl is None else self.eid[sl]
        named = self._is_thing(e)
        if not named.any():
            return mat
        return np.where(named, 256 + e.astype(np.int64), mat)

    def _things_moved(self, eids):
        """The cells of these eids just moved, emptied or changed hands. If
        any of them is a thing's, bump `_thing_gen`: whatever was kept about
        where things are is out of date. And the identity pass is to look at
        each of those things again. Every carry site calls this."""
        e = np.atleast_1d(np.asarray(eids))
        if e.any():
            e = e[self._is_thing(e)]
            if len(e):
                self._thing_gen += 1
                self._id_dirty.update(np.unique(e).tolist())

    def _thing_cells(self, fresh=False):
        """Where every thing is on the lattice: {eid: flat indices of its
        cells}, for each thing that has any.

        Kept until `_thing_gen` moves, so it is only right if every law that
        moves a thing's cell says so. `fresh=True` builds it again from the
        eid field and keeps nothing, which is what `_check_ids` compares the
        kept one with. Cells drawn from particles (`moving`) are not counted:
        the solver redraws them every tick, and the particles are the thing
        (`matter.entities()` says where they are)."""
        c = self._thing_idx
        if not fresh and c is not None and c[0] == self._thing_gen:
            return c[1]
        e = self.eid.ravel()
        at = np.flatnonzero(e)
        ids = e[at]
        keep = self._is_thing(ids) & ~self.moving.ravel()[at]
        at, ids = at[keep], ids[keep]
        o = np.argsort(ids, kind="stable")
        at, ids = at[o], ids[o]
        cut = np.flatnonzero(np.diff(ids)) + 1
        out = {int(i[0]): a for i, a in zip(np.split(ids, cut), np.split(at, cut))
               if len(i)}
        if not fresh:
            self._thing_idx = (self._thing_gen, out)
        return out

    def _check_ids(self):
        """`id_check`: what the entity layer promises, asserted. A cell of air
        is no thing; the kept index of where things are is the one the eid
        field gives now; every thing on the lattice is in the registry; and a
        dead thing is nowhere. The first fails when a law empties a cell and
        leaves its eid there; the second when a law moves a thing's cell and
        does not bump `_thing_gen`; the last two when the identity pass gets
        its own books wrong."""
        stray = (self.mat == AIR) & (self.eid != 0)
        if stray.any():
            c = tuple(int(v) for v in np.argwhere(stray)[0])
            raise AssertionError(
                f"t{self.tick}: {int(stray.sum())} cells of air still say which "
                f"thing they are, the first at {c} (eid {int(self.eid[c])})")
        kept, now = self._thing_cells(), self._thing_cells(fresh=True)
        if kept.keys() != now.keys() or any(
                not np.array_equal(kept[k], now[k]) for k in now):
            bad = sorted(k for k in set(kept) | set(now)
                         if k not in kept or k not in now
                         or not np.array_equal(kept[k], now[k]))
            raise AssertionError(
                f"t{self.tick}: the index of things is stale for eids {bad}: a "
                f"thing's cell moved and `_thing_gen` was not bumped")
        bad = sorted(k for k in now if self.things.get(k, {}).get("dead")
                     or k not in self.things)
        if bad:
            raise AssertionError(
                f"t{self.tick}: eids {bad} are on the lattice, and the registry "
                f"says each is dead or has never heard of it")

    # ── names (3.3) ─────────────────────────────────────────────────────────
    def name(self, eid, fallback, who=None):
        """What thing `eid` is called now, to a body: "the axe", "the burning
        axe", "the haft of the axe", "a piece of the plank".

        The FALLBACK, UNCHANGED, for anything that is not a registered thing
        (eid 0, a person's flesh). Every text site passes the words it always
        wrote, so a world that names nothing reads byte for byte as it did.

        One state word at most, the first that holds of: burning, ashes ("the
        ashes of the chair"), charred, broken ("the broken axe"; all of it in
        shards is "the pieces of the bottle").

        `who` is the body it is said to. When another thing would be called
        the same, the name says where it is from that body ("on your left",
        "at your feet", "in your hand"), and then which of them ("the second
        axe on your left"). Never in metres: the reader is a mind, not a
        surveyor."""
        e = int(eid)
        if e <= 0 or e >= len(self._thing_lut) or not self._thing_lut[e] \
                or e not in self.things:
            return fallback
        n = self._called(e)
        if who is None:
            return n
        twins = [k for k, t in self.things.items()
                 if k != e and t["dead"] is None and self._thing_lut[k]
                 and self._called(k) == n]
        if not twins:
            return n
        word = self._where_to(e, who)
        same = [k for k in twins if self._where_to(k, who) == word]
        if not same:
            return f"{n} {word}" if word else n
        # STILL TWO OF THEM THERE: count them off from the nearest
        order = sorted([e] + same, key=lambda k: (self._far(k, who), k))
        i = order.index(e)
        nth = _ORDINAL[i] if i < len(_ORDINAL) else f"{i + 1}th"
        n = _with(n, nth)
        return f"{n} {word}" if word else n

    def _called(self, e):
        """Thing `e`'s name with its state word, for no one in particular."""
        t = self.things[e]
        base = t["called"]
        idx = self._cells_now(e)
        rubble = False
        if len(idx):
            mat = self.mat.ravel()[idx]
            if self.burning_at(idx).any():
                return _with(base, "burning")
            if (mat == ASH).all():
                return "the ashes of " + t["whole"]
            if np.isin(mat, (CHAR, ASH)).any():
                return _with(base, "charred")
            rub = self.rubble.ravel()[idx]
            if rub.all() and t["kind"] != "debris":
                return "the pieces of " + t["whole"]
            rubble = bool(rub.any())
        if (t["broken"] or rubble) and t["kind"] != "debris":
            return _with(base, "broken")
        return base

    def _cells_now(self, e):
        """Flat indices of every cell thing `e` shows on the lattice now: its
        own cells, and the cells its particles and its bonded members were
        drawn into this tick."""
        own = self._thing_cells().get(e)
        mt = self.matter
        drawn = mt.drawn_at[mt.drawn_eid == e] if len(mt.drawn_at) else ()
        # ...and the cells its bonded matter was drawn into (src/bonds)
        bk = mt.bonds.drawn
        if len(bk.drawn_at):
            got = bk.drawn_at[bk.drawn_eid == e]
            if len(got):
                drawn = np.concatenate([np.asarray(drawn, np.int64), got])
        if own is None:
            return np.asarray(drawn, np.int64)
        return own if not len(drawn) else np.concatenate([own, drawn])

    def _where_to(self, e, who):
        """Where thing `e` is from body `who`, in a body's words, or None
        when it is nowhere to be seen (in flight, or all in the solver
        between two drawings)."""
        if self._held_eid(who) == e:
            return "in your hand"
        idx = self._cells_now(e)
        if not len(idx):
            return None
        return self._where_xyz(np.stack(np.unravel_index(idx, self.shape), -1),
                               who)

    def _where_xyz(self, xyz, who):
        """Where these cells (an (n, 3) array) are from body `who`, in the
        words `_where_to` uses for a thing. For matter with no name too: a
        lump of iron is on your left as much as a named bar is."""
        ax, ay = who["anchor"]
        dx = float(xyz[:, 0].mean()) - ax
        dy = float(xyz[:, 1].mean()) - ay
        if np.hypot(dx, dy) * 0.1 * self.scale <= 0.3:
            return "at your feet"             # right beside the body
        fx, fy = who.get("facing", (1.0, 0.0))
        ahead, side = dx * fx + dy * fy, fx * dy - fy * dx
        if abs(ahead) >= abs(side):
            return "in front of you" if ahead > 0 else "behind you"
        return "on your left" if side > 0 else "on your right"

    def _set_apart(self, called, rows, who, taken=()):
        """Words that tell lookalikes apart for body `who`: `rows` are
        (cells, how far) for things that would all be called `called`, and
        what comes back is one name each, in the same order, in the way
        `name` tells two registered lookalikes apart: where each is from the
        body, and then which of those there, nearest first. It is for menu
        rows, where matter with no name can look like a thing that has one
        ("the iron" and "the iron"), and a mind cannot pick one of two rows
        that say the same.

        `taken` are names other rows already say. None of what comes back
        is one of them, and no two are the same: a count that would land on
        a taken name goes on to the next ("the third iron at your feet",
        when a named first and second iron are there already)."""
        words = [self._where_xyz(np.asarray(c).reshape(-1, 3), who)
                 if len(c) else None for c, _ in rows]
        taken = set(taken)
        out = [None] * len(rows)
        for word in sorted(set(words), key=lambda v: min(
                rows[j][1] for j, u in enumerate(words) if u == v)):
            same = sorted((j for j, v in enumerate(words) if v == word),
                          key=lambda j: (rows[j][1], j))
            plain = f"{called} {word}" if word else called
            if len(same) == 1 and plain not in taken:
                out[same[0]] = plain
                taken.add(plain)
                continue
            k = 0
            for j in same:
                while True:
                    n = _with(called, _ORDINAL[k] if k < len(_ORDINAL)
                              else f"{k + 1}th")
                    n = f"{n} {word}" if word else n
                    k += 1
                    if n not in taken:
                        break
                out[j] = n
                taken.add(n)
        return out

    def _far(self, e, who):
        """How far thing `e` is from body `who`, for counting lookalikes off
        nearest first. Never shown: it only orders them."""
        idx = self._cells_now(e)
        if not len(idx):
            return float("inf")
        xyz = np.stack(np.unravel_index(idx, self.shape), -1)
        ax, ay = who["anchor"]
        return float(np.hypot(xyz[:, 0].mean() - ax, xyz[:, 1].mean() - ay))

    def _held_eid(self, p):
        """The thing body `p` is holding, or 0. Read off the grip's cell while
        the thing is on the lattice, and off the body that swings it while it
        is in the air with the arm."""
        held = (p or {}).get("held")
        if not held:
            return 0
        if held.get("eid"):
            return int(held["eid"])
        x, y, z = held["cell"]
        nx, ny, nz = self.shape
        if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz \
                and int(self.mat[x, y, z]) == held["mat"]:
            return self.entity_at(x, y, z)
        for b in self.bodies:
            if b.get("owner") == p["name"]:
                es = b["eids"][b["mats"] != FLESH]
                es = es[self._is_thing(es)]
                if len(es):
                    return int(es[0])
        return 0

    # ── properties (3.3, step B3 of E4) ─────────────────────────────────────
    def props(self, eid):
        """What thing `eid` is NOW, read off its own matter: a dict, or None
        for anything that is not a live registered thing.

          mass_g       grams of it: on the lattice, in the air, in the solver
          integrity    how much of what it weighed when it was made is still
                       it (`mass0`), 0 to 1; 1.0 when that was never noted
          sharp        its keenest edge, m² (the smallest edge above 0), or
                       0.0 for a thing with none
          hot_c        its hottest cell, °C (`T_at`); None with no cell on
                       the lattice to read (in the air, or all in the solver)
          burning      any cell of it alight (`burning_at`)
          charred      any of it char or ash
          ashes        all of it ash
          wet_ml       fluid soaked into its own cells
          contains     {fluid: ml} standing in its hollow
          can_hold_ml  how much its hollow holds when full (`_cavity` of the
                       particle seam, asked for every cell walled in, not only
                       the wet ones); 0.0 for a thing with no hollow
          broken       it has lost a piece of itself, or is in shards
          in_solver    some of it is particles now: named, and no hand's
          held_by      the names of the bodies holding it

        Kept until the tick or `_thing_gen` moves, so asking again in the
        same tick is free. Heat changes every tick, and that is what the tick
        in the key is for; within a tick, what the first asker saw is what
        everyone sees. Whether a body could lift it is a question about the
        body too (`liftable_by`)."""
        e = int(eid)
        if not (0 < e < len(self._thing_lut)) or not self._thing_lut[e] \
                or e not in self.things or self.things[e]["dead"] is not None:
            return None
        key = (self.tick, self._thing_gen)
        if self._props[0] != key:
            self._props = (key, {})
        got = self._props[1].get(e)
        if got is None:
            got = self._props[1][e] = self._props_of(e)
            if self.composites:                       # T2: a part of a composite says so
                cp = self.composite_props(e)
                if cp is not None:
                    got.update(cp)
        return dict(got, contains=dict(got["contains"]),
                    held_by=tuple(got["held_by"]))

    def _props_of(self, e):
        """`props`, worked out (no cache)."""
        t = self.things[e]
        own = self._thing_cells().get(e, np.zeros(0, np.int64))
        idx = self._cells_now(e)
        mpm = self.matter.points()
        mine = np.flatnonzero(mpm.eid == e) if len(mpm.x) \
            else np.zeros(0, np.int64)
        fly = [(b["eids"] == e) for b in self.bodies]
        grams = float(self.smass.ravel()[own].sum(dtype=np.float64)) \
            + 1000.0 * float(mpm.m[mine].sum()) \
            + sum(float(b["masses"][s].sum()) for b, s in zip(self.bodies, fly))
        edges = [self.edge.ravel()[idx]] + [b["edges"][s] for b, s
                                            in zip(self.bodies, fly)]
        edges = np.concatenate(edges) if edges else np.zeros(0, np.float32)
        edges = edges[edges > 0]
        mat = self.mat.ravel()[idx]
        hollow = self.matter._cavity(np.stack(np.unravel_index(
            idx, self.shape), -1), need_fluid=False) if len(idx) \
            else np.zeros(self.shape, bool)
        inside = {}
        wet = hollow & (self.fvol > 0.0)
        for f, ml in zip(self.fl[wet].tolist(), self.fvol[wet].tolist()):
            n = _FLUIDNAME.get(int(f), "fluid")
            inside[n] = inside.get(n, 0.0) + float(ml)
        rub = self.rubble.ravel()[idx]
        m0 = t.get("mass0") or 0.0
        return {
            "mass_g": grams,
            "integrity": min(grams / m0, 1.0) if m0 > 0 else 1.0,
            "sharp": float(edges.min()) if len(edges) else 0.0,
            "hot_c": float(self.T_at(idx).max()) if len(idx) else None,
            "burning": bool(len(idx) and self.burning_at(idx).any()),
            "charred": bool(np.isin(mat, (CHAR, ASH)).any()),
            "ashes": bool(t["ashes"] or (len(mat) and (mat == ASH).all())),
            "wet_ml": float(self.fvol.ravel()[idx].sum(dtype=np.float64)),
            "contains": inside,
            "can_hold_ml": float(hollow.sum()) * float(self.cap),
            "broken": bool(t["broken"] or rub.any()),
            "in_solver": bool(len(mine)),
            "held_by": tuple(p["name"] for p in self.persons
                             if self._held_eid(p) == e)}

    def liftable_by(self, eid, p):
        """Whether body `p` could lift thing `eid` off the ground: its weight
        against the strength of the body's better hand (`_take_up` asks the
        same of the hand it uses). False for anything that is not a thing."""
        got = self.props(eid)
        if got is None:
            return False
        mechs = self.__dict__.get("mechs")
        if mechs and any(int(eid) in (rec["leaf"], rec["frame"])
                         for rec in mechs.values()):
            return False                  # a hinge holds it: `operate` is the verb
        return got["mass_g"] / 1000.0 * GRAVITY \
            <= self._strength(p)

    def _adopt(self, x, y, z, label, by=None, within=None):
        """ADOPT-ON-TAKE (E4, D3). A hand has just closed on unnamed matter at
        (x, y, z): from now on it is a thing, the lump the hand took and only
        that, and the grip holds it by its eid. Returns the eid, or 0 when
        there is nothing here to adopt.

        Why. Unnamed matter is one lump with whatever of its own material it
        touches (3.3), so an iron bar put down against an iron post became
        part of the post, and a held stick laid on a wooden floor became the
        floor. A thing in a hand is a thing: what joins it from now on is
        what joins a named thing, its own cells and nothing else.

        Called "the iron" (by `label`, the word the take row used), so every
        line that named it says what it said before. Not for flesh, not for
        what the solver holds, and not for the world: a flood that runs to
        `_object_at`'s cap is not a thing a hand took.

        ONLY WHAT COULD BE LIFTED. A small world's floor, or a table, is
        under that cap, and a hand can close on it (a grip may close on what
        it cannot move: a constraint before a lift). Adopted, the floor was
        "the stone" from then on, a thing in the registry like a cup. So a
        lump is made a thing only if it is not the ground (no cell of it on
        the bottom layer, where support starts), it carries nothing (no
        other matter rests on top of it: a laid table is furniture, not a
        thing in a hand), and body `by` (or a body of the usual strength)
        could lift it. What fails stays the plain grip it always was.

        `within`, when given, is the cells (a set of (x, y, z)) the thing
        can be: a catch knows what came out of the air, and a wooden ball
        caught onto a wooden table is the ball, not the ball and the table
        it is joined to by being the same wood."""
        if int(self.mat[x, y, z]) in (AIR, FLESH) or self.moving[x, y, z] \
                or int(self.eid[x, y, z]):
            return 0
        lump = self._object_at(x, y, z)
        if not len(lump) or len(lump) >= 4000:
            return 0
        lump = np.asarray(lump, np.int64)
        if within is not None:
            lump = lump[[tuple(c) in within for c in lump.tolist()]]
            if not len(lump):
                return 0
        if (lump[:, 2] == 0).any():
            return 0                          # the ground
        if self._effort(lump)[0] > (self._strength(by) if by
                                    else BODY["strength_N"]):
            return 0                          # too heavy to lift
        # WHAT RESTS ON IT: the cell above each of its cells, when that is
        # matter and not its own. The taker's own flesh does not count (a
        # hand may be over what it takes), and neither does what the solver
        # holds (it is passing, not resting).
        up = lump[lump[:, 2] < self.shape[2] - 1] + np.array([0, 0, 1])
        mine = set(map(tuple, lump.tolist()))
        own = (by or {}).get("_own") or ()
        for c in map(tuple, up.tolist()):
            if c in mine or c in own or int(self.mat[c]) == AIR \
                    or self.moving[c]:
                continue
            return 0                          # it carries something
        e = self.register(label)
        self.stamp(e, lump)
        self._made(e)
        return e

    # ── per cell (3.3: props are computed on a thing's own cells) ───────────
    def T_at(self, idx):
        """`T()` at these cells only: a tuple of index arrays, as `np.nonzero`
        gives, or flat indices. The same sums in the same order, so the same
        numbers to the bit, without building the whole world's heat capacity
        to read a handful of cells."""
        if not isinstance(idx, tuple):
            idx = np.unravel_index(np.asarray(idx, np.int64), self.shape)
        mat, fl = self.mat[idx], self.fl[idx]
        if self.mixes_live:                      # T1: a mix is warmed by its own heat capacity
            cap = self._cap_of(mat, self.smass[idx], fl, self.fvol[idx],
                               self.scomp[idx], self.fcomp[idx])
            return AMBIENT + self.E[idx] / self._cap_plus(idx, cap)
        fmass = self.fvol[idx] * _FDENS_ARR[fl]
        cap = self.smass[idx] * _CSOLID_ARR[mat] + fmass * _FC_ARR[fl] \
            + self.c_airbase
        return AMBIENT + self.E[idx] / self._cap_plus(idx, cap)

    def _cap_plus(self, idx, cap):
        """`cap` (a cell list's own heat capacity) plus what a capability adds that is
        not in the four matter fields: the wet world's vapour, mist and ice (wetmix).
        The plain world adds nothing and hands `cap` back untouched."""
        return cap

    def burning_at(self, idx):
        """`burning()` at these cells only (the same index forms as `T_at`)."""
        if not isinstance(idx, tuple):
            idx = np.unravel_index(np.asarray(idx, np.int64), self.shape)
        T = self.T_at(idx)
        s = getattr(self, "skin", None)              # fuel burns at its SURFACE
        if s is not None and s.shape == self.shape:  # (heat._surfaces), as
            T = T + s[idx]                           # `burning()` reads it
        out = (self.fl[idx] == OIL) & (self.fvol[idx] > 0) & (T >= FLUID[OIL][3])
        return out | ((self.smass[idx] > 0) & (T >= _IGN_ARR[self.mat[idx]]))

    # ── unmaking, and the identity pass ─────────────────────────────────────
    def _unmake(self, mask, cause):
        """Empty the solid in the cells of `mask`: the matter there is gone.

        A cell that stops being solid stops being everything a solid cell
        was. Melt and acid used to set only `mat` and `smass`, and left `edge`
        and `rubble` behind in the air, so the next thing to arrive there was
        born sharp, or born a broken piece. Fluid, heat and smoke are not the
        solid's, so they stay where they are.

        `cause` is the verb that unmade it ("melts", "dissolves"). A thing
        that loses matter here has it written down (`_id_losses`), so that if
        this was the last of it the identity pass records that it died, and
        of what."""
        if self.mixes_live:
            self._vanish_mix(mask)
        rest = self._rest_pre(mask)             # cold (T8 4.3): None with no climate
        self.mat[mask] = AIR
        self.smass[mask] = 0.0
        if rest is not None:
            self._rest_post(mask, rest)
        self.edge[mask] = 0.0
        # `_rub_gen` is not bumped: rubble in a cell of air never reaches the
        # support field (`_relax_slack` clears it before it reads it), and
        # `mat` has changed here anyway, which already empties that cache.
        self.rubble[mask] = False
        e = self.eid[mask]
        if e.any():
            for k in np.unique(e[self._is_thing(e)]):
                self._id_losses.append((int(k), cause))
        self._things_moved(e)
        self.eid[mask] = 0

    def _rest_pre(self, mask):
        """THE SEAM for the matter rule (T8 4.3): called just before matter
        leaves the cells of `mask`; whatever it returns is handed to
        `_rest_post` just after. None (no climate) means nothing to do."""
        return None

    def _rest_post(self, mask, pre):
        """The other half of `_rest_pre`: the cells have changed their matter
        at a fixed E, so the REST part of E moves with the capacity and the
        cold of matter that leaves goes with it (outside.matter_change)."""

    def _vanish_mix(self, mask):
        """The solid mixes in `mask` leave the world (T1): their grams are
        booked to `gone` per species, so the species totals stay exact, and the
        cells go back to pure (scomp 0)."""
        s = self.scomp[mask]
        sel = s > 0
        if not sel.any():
            return
        T = self.mixes
        ent = T.entries(s[sel].astype(np.int64))
        T.book("gone", ent.species, ent.frac * self.smass[mask][sel].astype(np.float64)[ent.cell])
        self.scomp[mask] = 0

    def _law_identity(self):
        """Say what each thing IS after the laws that move and change matter.

        `step` calls it once a tick, before `_law_bleed`. First the HYGIENE:
        in every cell that became air since the last pass, it clears `edge`,
        `rubble` and `eid`. Those are facts about a solid, and a cell of air
        is not one. Every law that moves or removes matter should clear them
        itself, and most do; this catches the ones that do not, whatever they
        are, before anything can land on what they left. It compares `mat`
        with its copy from the last pass, so a quiet tick costs one compare
        and no writes. A thing whose cell changed what it is (wood burned to
        ash, glass to shards) is worth another look, and is marked for one.

        Then each thing that moved, changed or lost matter is looked at again
        (`_identity`): whether it came apart, and whether it is still
        anywhere at all. A world that names nothing has nothing to look at."""
        prev = self._id_prev_mat
        changed = np.flatnonzero(self.mat.ravel() != prev.ravel())
        if len(changed):
            e = self.eid.ravel()[changed]
            if e.any():
                e = e[self._is_thing(e)]
                self._id_dirty.update(np.unique(e).tolist())
            gone = changed[self.mat.ravel()[changed] == AIR]
            if len(gone):
                self.edge.ravel()[gone] = 0.0
                self.rubble.ravel()[gone] = False   # (no `_rub_gen` bump: see
                                                    # `_unmake`)
                self._things_moved(self.eid.ravel()[gone])
                self.eid.ravel()[gone] = 0
            np.copyto(prev, self.mat)
        if self._id_dirty or self._id_losses:
            self._identity()
        if self.id_check:
            self._check_ids()
        if self.records is not None and self.records.n:
            self._law_records()                  # T2: the records follow what was just settled

    def _identity(self):
        """Look again at every thing that moved, changed or lost matter.

        THE CENSUS. A thing is somewhere: on the lattice, in a body in the
        air, or in the solver's particles. One that is none of them is dead,
        of whatever took its last matter (`_id_losses`). That list is read
        here whether or not the lattice changed this tick, because a bar that
        goes over the edge of the world as it lands leaves nothing on the
        lattice to see: the loss is all the evidence there is.

        THE PIECES. A thing that is still somewhere may have come apart
        (`_pieces`). One in the air is left until it lands: a swing comes
        back down whole, and the pieces are counted where they come to rest.
        The loose bits of a thing (`kind` "debris") are a heap by nature, and
        are never split."""
        why = {}
        for k, cause in self._id_losses:
            why[int(k)] = cause
        self._id_losses = []
        todo, self._id_dirty = self._id_dirty | set(why), set()
        lut, things = self._thing_lut, self.things
        todo = sorted(k for k in todo if 0 < k < len(lut) and lut[k]
                      and k in things and things[k]["dead"] is None)
        if not todo:
            return
        where = self._thing_cells()
        mpm = self.matter.points()
        for e in todo:
            t = things[e]
            lat = where.get(e)
            mine = np.flatnonzero(mpm.eid == e) if len(mpm.x) \
                else np.zeros(0, np.int64)
            flying = any((b["eids"] == e).any() for b in self.bodies)
            if lat is None and not len(mine) and not flying:
                self._dies(e, why.get(e))
                continue
            if flying or t["kind"] == "debris":
                continue
            self._pieces(e, t, lat, mine)
            idx = self._cells_now(e)
            if len(idx) and not t["ashes"] and (self.mat.ravel()[idx] == ASH).all():
                t["ashes"] = True             # every cell of it burned to ash
                self.emit(ASHES, target=("thing", e), result="ashes")

    def _dies(self, e, cause):
        """Thing `e` is nowhere: write down that it died, when, and of what."""
        t = self.things[e]
        t["dead"], t["died"] = _DIED.get(cause, "lost"), self.tick
        if self.records is not None and self.records.n:
            self._records_died(e)
        self.emit(cause if cause in _DIED else "gone", target=("thing", e),
                  result=t["dead"])

    def _pieces(self, e, t, lat, mine):
        """Split thing `e` into the pieces it is now in, if it is in more than
        one. `lat` is its cells on the lattice (flat indices, or None), `mine`
        its particles in the solver.

        WHAT HOLDS A THING TOGETHER. Two of its cells that share a face are
        one piece, whatever each is made of: a named axe is iron and wood and
        still one axe. But ash is powder and a shard is a shard, so neither
        holds anything together. A plank burned through the middle is two
        pieces with ash between them, and a bottle on the floor in shards is
        no longer a bottle. In the solver, a piece is a bound cluster (`cid`):
        what one `promote` carried off and has not been torn apart since.

        WHERE THE LOOSE BITS GO. Ash or a shard touching a piece (through
        other loose bits, maybe) goes with the heaviest piece it touches. One
        that touches no piece at all goes to the thing's one DEBRIS child, a
        heap called "the pieces of the bottle", made the first time it is
        needed and given everything shed after. A thing that is nothing BUT
        loose bits has nothing left to come apart, and stays one thing.

        WHO KEEPS THE NAME. The heaviest piece keeps the eid, so a beam that
        loses a splinter is still the beam. Every other piece is a new thing,
        named from its parent: by the declared part it is ("the haft of the
        axe"), or "a piece of the plank". The parent is broken now.

        THE GRIP FOLLOWS THE PIECE IN THE HAND. A body holding the thing is
        holding whichever piece is at its fist, which need not be the one
        with the eid: a haft that snaps leaves the hand holding the haft."""
        units, loose_lat, loose_mpm = [], np.zeros(0, np.int64), mine[:0]
        if lat is not None:
            xyz = np.stack(np.unravel_index(lat, self.shape), -1)
            lo = xyz.min(axis=0)
            rel = tuple((xyz - lo).T)
            full = np.zeros(tuple(xyz.max(axis=0) - lo + 1), bool)
            full[rel] = True
            free = (self.mat.ravel()[lat] == ASH) | self.rubble.ravel()[lat]
            bind = full.copy()
            bind[tuple((xyz[free] - lo).T)] = False
            pb = label(bind, canon="min", fused=self.fused)[rel]
            g = self.smass.ravel()[lat].astype(np.float64)
            keys = np.unique(pb[pb >= 0])
            got = pb.copy()
            if free.any():
                # each lump of loose bits goes with the heaviest piece in it
                pf = label(full, canon="min", fused=self.fused)[rel]
                gk = {int(k): float(g[pb == k].sum()) for k in keys}
                owner = {}
                for k in sorted(gk, key=lambda k: (-gk[k], k)):
                    owner.setdefault(int(pf[pb == k][0]), k)
                for f in np.unique(pf[free]):
                    got[free & (pf == f)] = owner.get(int(f), -1)
            for k in keys:
                s = got == k
                units.append((float(g[s].sum()), 0, int(k), lat[s]))
            loose_lat = lat[got < 0]
        pts = self.matter.points() if len(mine) else None
        if len(mine):
            cid = pts.cid[mine]
            kg = pts.m[mine]
            for c in np.unique(cid[cid >= 0]):
                s = cid == c
                units.append((float(kg[s].sum()) * 1000.0, 1, int(c), mine[s]))
            loose_mpm = mine[cid < 0]
        if not units:
            return                            # nothing holds any of it together
        if len(units) == 1 and not len(loose_lat) and not len(loose_mpm):
            return                            # one piece: it is what it was
        units.sort(key=lambda u: (-u[0], u[1], u[2]))
        holders = [p for p in self.persons if self._held_eid(p) == e]
        hands = {id(p): self._in_hand(p, [u for u in units if u[1] == 0])
                 for p in holders}
        born = []
        for _g, where, _k, members in units[1:]:
            mats = set(self.mat.ravel()[members].tolist()) if where == 0 \
                else set(pts.vmat[members].tolist())
            part = self._part_of(t, mats)
            if part is not None:
                called = f"the {part} of {t['whole']}"
                c = self.register(t["name"], t["kind"], parent=e,
                                  called=called, whole=called)
            else:
                c = self.register(
                    t["name"], t["kind"], parent=e,
                    parts={k: v for k, v in t["parts"].items() if v in mats},
                    called=t["called"] if t["called"].startswith("a piece of ")
                    else "a piece of " + t["whole"], whole=t["whole"])
            self._give(c, where, members)
            self.things[c]["mass0"] = float(_g)   # what it weighed at birth
            born.append(c)
        if len(loose_lat) or len(loose_mpm):
            d = t["debris"]
            if not d or self.things[d]["dead"] is not None:
                d = self.register(t["name"], "debris", parent=e,
                                  called="the pieces of " + t["whole"],
                                  whole=t["whole"])
                t["debris"] = d
            self._give(d, 0, loose_lat)
            self._give(d, 1, loose_mpm)
            born.append(d)
        t["broken"] = True
        if self.records is not None and self.records.n:
            self._records_split(e, born[:len(units) - 1], [u[0] for u in units])
        for p in holders:
            self._grip_to(p, hands[id(p)])
        self.emit(SPLITS, target=("thing", e), result="split",
                  data=(("into", tuple(born)),))

    def _give(self, c, where, members):
        """Make `members` part of thing `c`: cells on the lattice (`where`
        0, flat indices) or particles in the solver (1). Drawn cells are
        redrawn from the particles at the next `rasterize`."""
        if not len(members):
            return
        if where == 0:
            self.eid.ravel()[members] = c
            self._thing_gen += 1
            self._bulk = None                 # new keys: see `stamp`
            self._torque_solid = None
        else:
            self.matter.set_eid(members, c)

    @staticmethod
    def _part_of(t, mats):
        """Which declared part of thing `t` a piece made of `mats` is, or
        None. What fire leaves of a part (char, ash) is still that part, and
        a part is only told apart by its material when no other part shares
        it."""
        m = set(mats) - {CHAR, ASH}
        mine = list(t["parts"].values())
        hit = [k for k, v in t["parts"].items() if m == {v} and mine.count(v) == 1]
        return hit[0] if len(hit) == 1 else None

    def _in_hand(self, p, units):
        """Which of these pieces on the lattice is at body `p`'s fist, and the
        cell of it nearest the fist: the piece with a cell nearest it. Without
        a fist to ask (no arm named, or none left), the piece the grip's cell
        is in, and no cell (the grip is on it already). None if neither."""
        if not units:
            return None
        arm = p["held"].get("arm")
        fist = None
        if arm:
            mine, sl = self._person_cells(p)
            if mine is not None and mine.any():
                own = np.argwhere(mine)
                own[:, 0] += sl[0].start or 0
                own[:, 1] += sl[1].start or 0
                fist = self._fist_of(p, arm, own)
        if fist is None:
            at = int(np.ravel_multi_index(p["held"]["cell"], self.shape))
            u = next((u for u in units if (u[3] == at).any()), None)
            return None if u is None else (u, None)
        f = np.asarray(fist, np.float64)
        best = None
        for u in units:
            d = ((np.stack(np.unravel_index(u[3], self.shape), -1) - f) ** 2).sum(axis=1)
            i = int(d.argmin())
            if best is None or d[i] < best[0]:
                best = (float(d[i]), u, int(u[3][i]))
        return best[1], best[2]

    def _grip_to(self, p, hand):
        """Point body `p`'s grip at the piece in its hand (`_in_hand`), if
        it is not already on it. A grip that holds a thing by its eid (E3)
        now holds the piece's: the haft that came away in the hand is a thing
        of its own, and the grip is good only while its cell is that thing."""
        if hand is None:
            return
        unit, at = hand
        held = p["held"]
        cell = tuple(int(v) for v in held["cell"])
        flat = int(np.ravel_multi_index(cell, self.shape))
        if at is not None and not (unit[3] == flat).any():
            cell = tuple(int(v) for v in np.unravel_index(at, self.shape))
            m = int(self.mat[cell])
            held["cell"], held["mat"] = cell, m
            held["label"] = MATNAME.get(m, "thing")
        e = self.entity_at(*cell)
        if held.get("eid") and e:
            held["eid"], held["name"] = e, self.name(e, "the " + held["label"])
