"""Which flesh is whose: adding people, body ownership by eid, status."""
import weakref

import numpy as np
from .events import Mind
from .materials import (FLESH)


# WHAT EACH CLAIM FOUND, kept beside the World and not in it (like senses.py's
# `_ROUND`), so a snapshot does not copy it and golden does not see it: per
# person, the claim mask `_own_claim` was given, its cells as np.argwhere
# gives them, the frozenset it made of them, and the claim's slice. A dozen laws a tick want a
# body's cells, and each ran np.argwhere over the whole lattice again to get
# them. An entry is used only while the person's `_claim_mask` IS the mask it
# was made from (the same object, which the entry holds, so it cannot be
# another under the same id), so it is never stale: a new claim makes a new
# mask, and nothing edits a claim mask in place.
_CLAIMED = weakref.WeakKeyDictionary()


class PersonsMixin:

    def _person_at(self, cell):
        """Whose flesh is this? Ownership, not adjacency: this tick's deeds,
        which are read off the flesh's eid (`_claim_bodies`).

        THIS TICK'S deeds. It read whatever was lying in `_own`, so the
        first thing to touch a person in a tick — an arrow, before anyone had
        asked where anybody was — found nobody there, and 75 J into a man's
        chest wounded no one. Every other body query refreshes the claims
        first; this one now does too."""
        if self.persons and any(q.get("_claim_tick") != self.tick
                                for q in self.persons):
            self._claim_bodies()
        for q in self.persons:
            if cell in (q.get("_own") or frozenset()):
                return q
        return None

    # ── physiology: living bodies ────────────────────────────────────────────
    def add_person(self, x, y, name="the person", knows_world=True):
        """Register a living body anchored near (x, y). The FLESH voxels are the
        body; this struct is only the slow chemistry riding on them. One person
        per neighborhood for now — the flesh search is anchor-local."""
        p = {"anchor": (int(x), int(y)), "pid": len(self.persons),
             "name": name, "blood_o2": 0.97,
             "smoke": 0.0, "burn": 0.0, "hurt": 0.0, "awake": True, "alive": True,
             "fleeing": False, "safe": False, "events": [],
             # WHAT THIS BODY HAS SEEN — not what the world contains. This is
             # the difference between a person and the sim: the sim holds the
             # register of exits, a person holds the ones they have laid eyes
             # on, and pathing runs on the latter.
             #
             # The default is True because the usual case is someone standing
             # in a place they live: they HAVE observed it, so knowing where
             # their own door is privileges nothing. Pass knows_world=False for
             # a stranger, and they must find the door by looking — which is
             # what makes wandering and looking around worth doing at all.
             "known": np.full(self.shape[:2], bool(knows_world)),
             # AND WHAT IT COULD WALK THROUGH when it looked. `known` says
             # where the eyes have been; this says what they FOUND there, and
             # it is what the route planner runs on. Built on the first look,
             # because a body's height decides which columns it fits down and
             # there is no flesh on the lattice yet at this point.
             "free": None, "knew": bool(knows_world)}
        if getattr(self, "stamp_bodies", False):
            # A BODY IS AN ENTITY TOO, but not a thing (3.3): registered as a
            # person, so nothing names, splits or buries it. Its flesh carries
            # this eid from the scene that builds it (`scenes._person`), or
            # from the first claim on flesh nobody stamped (`_claim_bodies`).
            p["eid"] = self.register(name, kind="person")
        self.persons.append(p)
        if hasattr(self, "minds"):           # unset only before _init_events
            self.minds.append(Mind())        # runs; every scene adds people
                                              # after __init__ finishes
        return p

    def _person_cells(self, p):
        """The person's OWN flesh — and OWN means owned, not merely connected.

        Bodies used to be identified by adjacency alone, so the moment two
        people touched they became one 3-voxel-wide, two-headed person: both
        resolved to the same 684 cells, both anchors converged, and anything
        that made them touch (carrying, dragging, a crowd in a doorway) broke
        them. Then they were claimed by a flood that stopped at a neighbour's
        flesh, and that flood was wrong wherever two bodies touched (roadmap
        82, 87). Now a body is the flesh that carries its eid
        (`body_by_eid`), read once a tick for everyone. Two people can stand
        shoulder to shoulder and stay two people because their flesh says
        whose it is."""
        if p.get("_claim_tick") != self.tick:
            self._claim_bodies()
        return p.get("_claim_mask"), p.get("_claim_sl")

    def _eid_of(self, p):
        """The person's eid, registered now if they have none yet. With
        `stamp_bodies` on (the default) `add_person` registers it and the
        scene stamps the body it builds. Off, the scene stamps nothing, and
        the first claim registers the person and adopts their flesh."""
        e = p.get("eid")
        if not e:
            e = p["eid"] = self.register(p["name"], kind="person")
        return e

    def body_by_eid(self, e, lab=None):
        """WHICH FLESH ON THE LATTICE IS THIS BODY: the flesh that carries
        eid `e`, as a mask the shape of the world. Flesh the particle solver
        holds (`moving`) is drawn there again every tick and is nobody's
        body. `lab` is `_claim_bodies`' own copy of the eid of every body
        cell (0 elsewhere), so a claim does not read the lattice once per
        person."""
        if lab is None:
            lab = np.where((self.mat == FLESH) & ~self.moving, self.eid, 0)
        return lab == e

    def body_of(self, p):
        """THE WHOLE BODY of person `p`, wherever its pieces are: (the cells
        of it on the lattice, an (n, 3) int array; the cells of it in the air,
        an (m, 3) float array at the poses the free bodies have now).

        A man in the air is a free body (`_launch`), and a landing is not all
        at once: for a few ticks some of him is on the floor and the rest is
        still coming down. Both halves carry his eid, so both are his. The
        claim gives a man in the air no body to move (`_claim_bodies`); this
        is where the rest of the sim can still find all of him. A limb in a
        swing is in the air too, and is in the second array."""
        e = p.get("eid")
        on = np.argwhere(self.body_by_eid(e)) if e else np.zeros((0, 3), np.int64)
        up = []
        for b in (self.bodies if e else ()):
            sel = (b["eids"] == e) & (b["mats"] == FLESH)
            if sel.any():
                pose = self._fly_pose(b) if b.get("fly") \
                    else self._body_pose(b, b["theta"])
                up.append(pose[sel])
        return on, (np.concatenate(up) if up else np.zeros((0, 3), np.float32))

    def _aloft(self):
        """The names of the people in the air: the owner of a free body that
        is not a limb, and anyone whose flesh rides in one. A body toppling
        as one lump can carry two people (an unconscious man and the corpse
        he lies against), and its owner names only one of them; the eid on
        the flesh names both."""
        up = {b["owner"] for b in self.bodies
              if b.get("owner") and not b.get("part")}
        up |= self.matter.aloft_owners()
        e = [b["eids"][b["mats"] == FLESH] for b in self.bodies
             if not b.get("part")]
        e = set(np.unique(np.concatenate(e)).tolist()) - {0} if e else ()
        if e:
            up |= {q["name"] for q in self.persons if q.get("eid") in e}
        return up

    def _claim_bodies(self):
        """Resolve every body once a tick: each is the flesh with its eid.

        This was a flood, and then a fair grow from what each body owned a
        moment ago, and every version of it was wrong somewhere two bodies
        touched: a neighbour's swung arm given to the man it hit, a corpse
        swallowed by the living man lying against it, a few cells handed
        across between two men standing together (roadmap 82 and 87, the
        E6 audit). The eid is stamped on a body when it is built and every
        law that moves matter moves it, so a cell's eid already says whose
        it is, and the claim only reads it (step E7).

        Nobody gets a body to move while any of them is in the air
        (`_aloft`): the free body owns the flight, and a man half landed
        stands on nothing yet. Their deed `_own` is then what they have on
        the lattice, so a blow on it still finds them (`_person_at`). The
        dead and the gone have no body and no deed. A body in pieces is
        still one body: the eid says whose a piece is, not whether it
        touches the rest.

        Flesh nobody stamped is adopted (`_adopt_flesh`), which is how a
        body built by hand before `add_person` comes to carry its eid."""
        for q in self.persons:
            self._eid_of(q)
        aloft = self._aloft()
        flesh = (self.mat == FLESH) & ~self.moving
        lab = np.where(flesh, self.eid, 0)
        live = [q for q in self.persons
                if not q["safe"] and q["alive"] and q["name"] not in aloft]
        if live and (flesh & (lab == 0)).any():
            self._adopt_flesh(live, flesh, lab)
        here = {id(q) for q in live}
        nx, ny, _nz = self.shape
        for q in self.persons:
            gone = q["safe"] or not q["alive"]
            mine = None if gone else self.body_by_eid(q["eid"], lab)
            if id(q) in here and mine.any():     # a whole-lattice mask, but the
                self._own_claim(q, mine,         # slice still starts at 0 so
                                (slice(0, nx), slice(0, ny), slice(None)))
                continue                         # every caller's offset maths
            q["_claim_mask"], q["_claim_sl"] = None, None
            q["_claim_tick"] = self.tick
            if id(q) in here or gone:
                q["_own"] = frozenset()          # no body on the lattice
            else:                                # in the air: what he has
                got = np.argwhere(mine)          # put down so far
                q["_own"] = frozenset(zip(got[:, 0], got[:, 1], got[:, 2]))

    def _adopt_flesh(self, live, flesh, lab):
        """FLESH NOBODY STAMPED IS ADOPTED by the body that reaches it first.
        A test that fills flesh by hand and then calls `add_person` has a
        body whose eid is on none of it, and so does a world with
        `stamp_bodies` off. Every live body grows into the unstamped flesh
        touching it, one voxel a round and all together; a cell two bodies
        reach in the same round goes to the one listed first, so the sim
        stays deterministic. A body with no flesh at all seeds on the
        unstamped flesh nearest its anchor, if any is within reach: two
        people built by hand in contact are one lump with nothing to divide
        it, and this splits it by who is nearer. Flesh someone already
        stamped is never taken. What each body grew into is stamped with
        its eid, here and in `lab`.

        Only the box around the unstamped flesh is grown, so a lump of flesh
        nobody touches costs a round of six shifts of that box a tick."""
        free = flesh & (lab == 0)
        got = np.argwhere(free)
        lo = np.maximum(got.min(axis=0) - 1, 0)
        hi = np.minimum(got.max(axis=0) + 2, self.shape)
        box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        n = len(live)
        FREE = np.int16(np.iinfo(np.int16).max)
        label = np.where(free[box], FREE, np.int16(-1)).astype(np.int16)
        sub = lab[box]
        has = []
        for i, q in enumerate(live):
            mine = sub == q["eid"]
            label[mine] = i
            has.append(bool(mine.any()) or bool((lab == q["eid"]).any()))
        r = max(int(1.8 / max(0.1 * self.scale, 1e-9)), 8)
        for i, q in enumerate(live):
            if has[i]:
                continue
            cand = np.argwhere(label == FREE)
            if not len(cand):
                break
            x0, y0 = q["anchor"]
            d = (np.abs(cand[:, 0] + lo[0] - x0)
                 + np.abs(cand[:, 1] + lo[1] - y0))
            if d.min() <= 2 * r:
                label[tuple(cand[int(d.argmin())])] = i
        grows = np.append(np.arange(n), FREE).astype(np.int16)
        while True:                              # every seed grows together
            free_b = label == FREE
            if not free_b.any():
                break
            best = np.full(label.shape, FREE, np.int16)
            lab_b = grows[np.minimum(np.where(label >= 0, label, FREE), n)]
            for ax in range(3):
                for d in (1, -1):
                    sh = np.full(label.shape, FREE, np.int16)
                    src = [slice(None)] * 3
                    dst = [slice(None)] * 3
                    src[ax] = slice(1, None) if d > 0 else slice(None, -1)
                    dst[ax] = slice(None, -1) if d > 0 else slice(1, None)
                    sh[tuple(dst)] = lab_b[tuple(src)]
                    np.minimum(best, sh, out=best)
            take = free_b & (best < FREE)
            if not take.any():
                break                            # the rest touches nobody
            label[take] = best[take]
        audit = getattr(self, "audit_bodies", False)
        for i, q in enumerate(live):
            mine = free[box] & (label == i)
            if not mine.any():
                continue
            cells = np.argwhere(mine) + lo
            self.stamp(q["eid"], cells)
            sub[mine] = q["eid"]
            if audit:
                # WHAT THE CLAIM ADOPTED (E6's audit, after E7): the only
                # place a claim can still differ from the eid it found
                self.id_audit.append({"body": q["name"], "eid": q["eid"],
                                      "tick": self.tick,
                                      "adopted": int(len(cells))})

    def _carry_own(self, q, dx, dy, dz=0):
        """KEEP THE DEED IN STEP WITH A BODY THAT MOVED THIS TICK.

        The deed `_own` is no longer what the claim grows from: a body is its
        eid (`_claim_bodies`), and the eid moves with the flesh. But `_own`
        is still read straight off the person dict, between one claim and
        the next, by the laws that ask whose a cell is (senses, support,
        perception, beliefs). So a mover that shifts a body without asking
        for a new claim (a step, a drag) shifts the deed with it, and those
        readers see where the body is now, as they always did. A thin view,
        kept for those readers; the next claim rewrites it from the eid."""
        own = q.get("_own")
        if own:
            q["_own"] = frozenset((c[0] + dx, c[1] + dy, c[2] + dz)
                                  for c in own)

    def _move_own(self, q, src, dst):
        """The same for a mover that is not a plain step: the cells in `src`
        became the cells in `dst` (two (k, 3) lists, row for row, called
        AFTER the move). A quarter turn, a limb posed and a crouch all move a
        body's own voxels. Only his flesh is deeded, read by its eid: a held
        tool that rode along in the same move is not part of anybody."""
        own = q.get("_own")
        if own:
            dst = np.asarray(dst, np.int64).reshape(-1, 3)
            at = tuple(dst.T)
            dst = dst[(self.mat[at] == FLESH)
                      & (self.eid[at] == q.get("eid", 0))]
            q["_own"] = ((own - set(map(tuple, np.asarray(src).tolist())))
                         | set(map(tuple, dst.tolist())))

    def _own_claim(self, q, mask, sl):
        """Record what this body turned out to be, and return its cells."""
        got = np.argwhere(mask)
        raw = got.copy() if (sl[0].start or sl[1].start) else got
        got[:, 0] += sl[0].start or 0            # a whole-lattice mask has no
        got[:, 1] += sl[1].start or 0            # offset to add back
        # the same tuples of np.int64, in the same order, as map(tuple, got):
        # so the same set, iterated in the same order, five times faster
        cells = frozenset(zip(got[:, 0], got[:, 1], got[:, 2]))
        q["_claim_mask"], q["_claim_sl"], q["_claim_tick"] = mask, sl, self.tick
        q["_own"] = cells
        _CLAIMED.setdefault(self, {})[id(q)] = (mask, raw, cells, sl)
        return cells

    def _claim_entry(self, p, comp):
        """`_CLAIMED`'s entry for this person's claim mask `comp`, or None."""
        ent = _CLAIMED.get(self, {}).get(id(p))
        return ent if ent is not None and ent[0] is comp else None

    def _claim_cells(self, p, comp, sl, shift=True):
        """`np.argwhere(comp)`, plus the claim's offset with `shift`: a body's
        own cells, from what its claim already found when it can be (a new
        array each call, so a caller may change it)."""
        ent = self._claim_entry(p, comp)
        c = ent[1].copy() if ent is not None else np.argwhere(comp)
        if shift:
            c[:, 0] += sl[0].start or 0
            c[:, 1] += sl[1].start or 0
        return c

    def person_status(self, p):
        state = ("DEAD" if not p["alive"] else
                 "unconscious" if not p["awake"] else "awake")
        return (f"{p['name']}: blood O2 {100 * p['blood_o2']:.0f}%, smoke load "
                f"{p['smoke']:.2f}, burns {100 * p['burn']:.0f}%, wounds "
                f"{100 * p['hurt']:.0f}% — {state}")
