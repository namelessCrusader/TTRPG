"""Grip, hold, carry, haul, shove, catch: what hands can reach and move."""
import numpy as np
from . import contact, matlaws
from .materials import (AIR, BODY, FLESH, FRICTION, GRAVITY, MATNAME)
from .label import flood, label


def _reach_distance(rel, cells, origin, shape, R):
    """How many cells each candidate is from the nearest cell of the body, as
    the largest step along any axis (a king's move): `rel` holds the
    candidates in the box, `cells` the body in the world, `origin` the corner
    of the box. Anything farther than R comes back as R + 1.

    The pair-by-pair sum (every candidate against every cell of the body) was
    most of the call in a room with furniture in reach. Here the body is
    grown one ring at a time, a ring being a step along each axis (three
    shifts of a boolean grid), and a candidate is as far as the ring it is
    first inside: the same integers, with the work in the size of the box."""
    inside = np.zeros(shape, bool)
    at = np.asarray(cells, np.int64) - np.array(origin)
    inside[at[:, 0], at[:, 1], at[:, 2]] = True
    dist = np.full(shape, R + 1, np.int64)
    dist[inside] = 0
    for k in range(1, R + 1):
        for ax in range(3):
            grown = inside.copy()
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[ax], hi[ax] = slice(None, -1), slice(1, None)
            grown[tuple(lo)] |= inside[tuple(hi)]
            grown[tuple(hi)] |= inside[tuple(lo)]
            inside = grown
        dist[inside & (dist > k)] = k
    return dist[rel[:, 0], rel[:, 1], rel[:, 2]]


class HandsMixin:

    def _resist(self, q, cells):
        """What it costs to move a body that would rather not be moved, in
        newtons.

        Friction is all a limp body gives you. A body that is AWAKE, alive and
        has something under its feet BRACES — it puts its own strength into the
        floor against you — and that is one number rather than a rule about
        rescues and a second rule about fights. Everything interesting falls
        out of the arithmetic: an unconscious man can be dragged and a standing
        one cannot, the same standing man CAN be moved by someone stronger, and
        he stops resisting the instant his feet have nothing to push against.
        Nobody wrote any of those three down.

        Before this, a grip was legal only on someone already unconscious. That
        was a case wearing a flag (Ruling 1): it made rescuing the only reason
        two people ever touched, and it made 'pull him off the ledge'
        unaskable — not hard, not physically refused, simply absent from the
        menu."""
        _lift, drag_N = self._effort(cells)
        if q["awake"] and q["alive"] and self._underfoot(cells):
            drag_N += q.get("strength_N", BODY["strength_N"])
        return drag_N

    def _within_reach(self, p, cells):
        """People this body could get a hand to: alive, and near.

        REACH decides whether a grip is possible; FORCE decides what the grip
        can then do, and it is asked again every tick in _haul. The force test
        used to live here, which quietly said a body knows before touching you
        whether you can be moved — and worse, it answered ONCE, so a brace
        that fails the moment a man's heel goes over an edge could never change
        the answer. Taking hold of someone stronger than you is a perfectly
        legal thing to do. It just does not move them."""
        out = []
        for q in self.persons:
            if q is p or not q["alive"] or q["safe"]:
                continue
            qc, qsl = self._person_cells(q)
            if qc is None or not qc.any():
                continue
            qcell = self._claim_cells(q, qc, qsl)
            if self._in_reach(cells, qcell):
                out.append(q)
        return out

    def _objects_within_reach(self, p, cells):
        """Loose THINGS a hand could close on: solid, not flesh, near, and
        light enough that this body could at least drag them. The last is a
        model of attention rather than a law — every wall in the world is
        within reach of somebody and on nobody's menu. A thing is what
        _object_at says it is: whatever is joined to what you grabbed.

        One row a thing: the cell a take closes on, its material and word,
        and `away`, how many cells it is from the body (what the menu's cap
        sorts by), in the order the box meets them, and at most
        `reach_things_cap` of them (None: all)."""
        vox_m = 0.1 * self.scale
        R = max(int(round(self._reach_of(p) / max(vox_m, 1e-9))), 1)
        nx, ny, nz = self.shape
        cells = np.asarray(cells)
        x0 = max(int(cells[:, 0].min()) - R, 0)
        x1 = min(int(cells[:, 0].max()) + R + 1, nx)
        y0 = max(int(cells[:, 1].min()) - R, 0)
        y1 = min(int(cells[:, 1].max()) + R + 1, ny)
        z0 = max(int(cells[:, 2].min()) - R, 0)
        z1 = min(int(cells[:, 2].max()) + R + 1, nz)
        box = self.mat[x0:x1, y0:y1, z0:z1]
        # NOT WHAT THE SOLVER HOLDS. A `moving` cell is particle matter drawn
        # onto the lattice for one tick; the particles are the thing, and the
        # cell is only where they were drawn. A hand that closed on it moved
        # the drawing, and the next tick the solver drew the particles again:
        # the same wood twice. So it is not a candidate, and it does not join
        # one thing to another either.
        still = ~self.moving[x0:x1, y0:y1, z0:z1]
        cand = np.argwhere((box != AIR) & (box != FLESH) & still)
        if not len(cand):
            return []
        cand += np.array([x0, y0, z0])
        # NO MARGIN. It was padded by another R so that a thing entirely
        # inside the padded box was provably whole — but that made the box
        # eight times bigger (115k cells against 27k) to answer a question the
        # flood fallback below answers anyway, once per component. Anything
        # touching the wall asks the real question; everything else is settled
        # by the labelling, which is the common case and the cheap one.
        if self._windows_on():
            d = _reach_distance(cand - np.array([x0, y0, z0]), cells, (x0, y0, z0),
                                box.shape, R)
        else:
            d = np.abs(cand[:, None, :] - cells[None, :, :]).max(axis=2).min(axis=1)
        cand, d = cand[d <= R], d[d <= R]
        # LABEL THE BOX ONCE, do not flood the world per candidate.
        #
        # This used to call `_object_at` for every candidate cell it had not
        # already seen, which is right when things are small and catastrophic
        # when they are not: `_object_at` stops at 4000 cells, and a stone
        # ledge is sixteen thousand, so the memo NEVER covered it and every
        # candidate standing on it paid another four-thousand-cell flood.
        # Profiled at 4576 floods in thirty ticks — 84% of the whole run, and
        # fifty million set operations. Lengthening a body's reach from 0.30 m
        # to 0.45 tripled the candidates and so tripled that.
        #
        # A thing is what is joined to what you grabbed, and "joined" only has
        # to be answered INSIDE THE BOX a hand can reach into. Labelling the
        # box is one pass over about thirty thousand cells with no Python loop
        # in it, and it answers for every candidate at once. Something that
        # runs out of the box is bigger than the box, which is the same thing
        # the flood cap was trying to say and says it without a magic number.
        #
        # SAME KEY (3.3), which is what `_object_at` means by "joined": the
        # same material, unless the cells are a named thing's, and then the
        # same thing. An unnamed table with iron legs is two things, because
        # joints do not exist yet; a named axe of iron and wood is one.
        # Labelling only on "solid and not flesh" merged an iron post into
        # the stone ledge it stands on, and the post then read as the world
        # and could not be picked up. Each piece is labelled by its smallest
        # flat index in the box.
        #
        # THE BOX DOES NOT WRAP (E4). It used to join each face of the box to
        # the opposite one, as the `np.roll` flood it replaced did, so two
        # iron bars at the two ends of a man's reach were one lump to it, and
        # only one of them was ever on his menu. A thing that runs out of the
        # box is asked about with a real flood below, so the wrap bought
        # nothing.
        solid = (box != AIR) & (box != FLESH) & still
        join = self._key((slice(x0, x1), slice(y0, y1), slice(z0, z1)))
        lab = label(solid, join, canon="min", wrap=False, fused=self.fused)
        # HOW FAR EACH PIECE IS, in cells: the nearest of its cells to the
        # nearest cell of the body (the same measure as `cand` above, and the
        # same unit as a place's "away")
        kc = lab[tuple((cand - np.array([x0, y0, z0])).T)]
        o = np.lexsort((d, kc))
        first = np.ones(len(o), bool)
        first[1:] = kc[o][1:] != kc[o][:-1]
        near = dict(zip(kc[o][first].tolist(), d[o][first].tolist()))
        self._matter_moved()                       # refresh the generation
        gen = getattr(self, "_matter_gen", 0)
        if self._bulk is None or self._bulk_gen != gen:
            self._bulk, self._bulk_gen = set(), gen   # the world may be a
        bulk = self._bulk                             # different shape now
        out, seen, flooded = [], set(), set()
        strength = p.get("strength_N", BODY["strength_N"])
        cap = self.reach_things_cap
        off = np.array([x0, y0, z0])
        for c in map(tuple, cand):
            key = int(lab[c[0] - x0, c[1] - y0, c[2] - z0])
            if key in seen:
                continue
            seen.add(key)
            where = np.argwhere(lab == key) + off
            # A THING THAT RUNS OUT OF THE BOX MIGHT STILL BE A THING. A post
            # is nineteen voxels tall and a ledge is thousands, and both leave
            # a box drawn round one man's reach — so leaving it cannot be the
            # test. For those, and only those, ask the real question with a
            # flood. It is at most a handful per call now rather than one per
            # candidate cell, because the memo keys on the COMPONENT and not on
            # which cells a capped flood happened to visit.
            if len(where) >= 4000:
                continue                  # already the world INSIDE the box —
                                          # no flood can make it less so
            if tuple(c) in bulk:
                continue                  # known world: something this cell is
                                          # part of was flooded before and ran
                                          # over the cap
            if (where[:, 0].min() == x0 or where[:, 0].max() == x1 - 1
                    or where[:, 1].min() == y0 or where[:, 1].max() == y1 - 1
                    or where[:, 2].min() == z0 or where[:, 2].max() == z1 - 1):
                # ONE THING THAT LEAVES THE BOX AND COMES BACK is two pieces
                # in it, and each floods to the same thing: one row
                if tuple(c) in flooded:
                    continue
                where = self._object_at(*c)
                if len(where) < 4000:
                    flooded.update(map(tuple, where.tolist()))
            if len(where) >= 4000:
                # AND REMEMBER IT. A floor SLICE inside one man's reach is only
                # a thousand cells, under the cap, so the box cannot tell it
                # from a crate — and the flood that settles it is the most
                # expensive thing left in the tick. It is also the same answer
                # every tick until somebody moves the floor, so it is kept
                # until somebody does: `_matter_moved` is the sim's own word
                # for that, and already earns its keep for two other caches.
                bulk.update(map(tuple, where))
                continue                  # the flood cap: that is the world,
                                          # not a thing to pick up
            _lift, drag_N = self._effort(where)
            if drag_N > strength:
                continue                  # could not even shift it: not a thing
                                          # worth a slot of attention
            # THE CELL AND ITS MATERIAL FROM THE SAME VOXEL. A named axe is
            # iron and wood, so the material of the cell the box happened to
            # meet first need not be the material of the cell the row names,
            # and a grip checks one against the other.
            cell = tuple(map(int, where[0]))
            m0 = int(self.mat[cell])
            out.append({"cell": cell, "mat": m0,
                        "label": MATNAME.get(m0, "thing"),
                        "away": int(near[key])})
            # THE STOP (`reach_things_cap`; none by default, see there)
            if cap is not None and len(out) >= cap:
                break
        return out

    def _held_cells(self, p):
        """Where the held THING is now, gripped cell first. Gone (burned,
        shattered, knocked away) means gone.

        A NAMED thing is held by WHICH thing it is (3.3): the grip is good
        while the gripped cell is still that thing's (`eid`), and the thing
        is every cell the registry says it has (`cells_of`). An unnamed one
        has no registry, so it is re-flooded from the gripped cell: identity
        is adjacency, checked against the material the hand closed on."""
        held = p.get("held")
        if not held:
            return None
        x, y, z = held["cell"]
        nx, ny, nz = self.shape
        inside = 0 <= x < nx and 0 <= y < ny and 0 <= z < nz
        e = held.get("eid") or (self._grip_learns(held, p) if inside else 0)
        if e:
            if inside and int(self.eid[x, y, z]) == e:
                if self.moving[x, y, z]:
                    return None               # the solver's now: no thing a
                                              # hand can move (`_object_at`)
                return self._grip_first(e, (x, y, z))
            for b in self.bodies:             # in the air with the arm: see
                if b.get("owner") == p["name"] and (b["eids"] == e).any():
                    return None               # below
            p["held"] = None
            return None
        if not inside or int(self.mat[x, y, z]) != held["mat"]:
            # IT MAY BE IN THE AIR — mid-swing, the thing is off the lattice
            # with the arm that swings it. A grip is not lost because it
            # moved (the same lesson _haul learned from fainting bodies).
            for b in self.bodies:
                if b.get("owner") == p["name"] and (b["mats"] != FLESH).any():
                    return None
            p["held"] = None
            return None
        obj = self._object_at(x, y, z)
        return obj.astype(np.int64) if len(obj) else None

    def _grip_learns(self, held, p=None):
        """A grip closed on a thing learns WHICH thing, the first time it is
        looked at: `eid`, and `name` for the words. The take row is written
        in will.py, and `_take_up` looks at the grip in the same act, before
        anything has moved. Returns the eid, or 0.

        A grip on UNNAMED matter makes it a thing first (`_adopt`, E4): what
        a hand has taken is one thing from then on, and is held by its eid
        like any other. Only a grip on what is no thing and cannot become
        one (the world, flesh, what the solver holds, what body `p` could
        not lift or what carries other matter: `_adopt`) stays the plain
        dict it always was."""
        x, y, z = held["cell"]
        if int(self.mat[x, y, z]) != held["mat"]:
            return 0
        e = self.entity_at(x, y, z) \
            or self._adopt(x, y, z, held.get("label", "thing"), by=p)
        if e:
            held["eid"] = e
            held["name"] = self.name(e, "the " + held.get("label", "thing"))
        return e

    def _grip_first(self, e, cell):
        """Thing `e`'s cells (`cells_of`), with the gripped cell first. The
        first cell is the grip's, for every law that moves a held thing and
        then moves the grip with it (`_take_up`, a reach, a step): the grip
        keeps pointing at the same piece of matter, and so at the same
        material."""
        at = self._thing_cells().get(e, np.zeros(0, np.int64))
        g = int(np.ravel_multi_index(cell, self.shape))
        at = np.concatenate(([g], at[at != g]))
        return np.stack(np.unravel_index(at, self.shape), -1).astype(np.int64)

    def _catch_person(self, p, b, arm):
        """Close a hand on someone who is falling.

        The same sum as catching a stone — stopping is force times time — and
        then the same GRIP that has always been able to hold a hanging man.
        Nothing was added for this: a caught body goes back on the lattice
        where it was caught, and `_grip_cells` seeds what a hand holds as
        supported, so he hangs there because a hand is holding him and for no
        other reason."""
        m = float(b["masses"].sum()) / 1000.0
        v = float(np.linalg.norm(b["vel"]))
        if m * v / max(BODY["catch_s"], 1e-9) > self._hand_strength(p, arm):
            return False
        who = b.get("owner")
        b["vel"] = np.zeros(3)
        self._land_body(b, None)
        self._out_of_flight(b)
        p["dragging"] = who
        return True

    def _out_of_flight(self, b):
        """Take body `b` out of `self.bodies`, found by IDENTITY.

        `b in self.bodies` and `self.bodies.remove(b)` compare by equality
        once the first entry is not `b`, and a body dict holds numpy arrays,
        so with two fliers in the air the comparison raised instead of
        answering. `_decide` already asks this way (the catch verbs in
        will.py); the two catches now remove the same way."""
        for i, x in enumerate(self.bodies):
            if x is b:
                del self.bodies[i]
                return

    def _catch(self, p, b, arm):
        """Close a hand on something in flight, if the hand can stop it.

        Stopping is force times time: a thing of mass m at speed v needs
        `m v / t` to be brought to rest in `t`, and a hand has only so much.
        That is the whole rule, and it is the same arithmetic as lifting — a
        catch that is too much for the arm simply does not happen, and the
        thing goes on its way past a hand that could not close on it."""
        m = float(b["masses"].sum()) / 1000.0
        v = float(np.linalg.norm(b["vel"]))
        if m * v / max(BODY["catch_s"], 1e-9) > self._hand_strength(p, arm):
            return False
        # SEEN IN TIME, counted from when HE saw it, not from when it was
        # thrown (`_in_time`, R3). Counted from the launch, a thing he had
        # watched coming for a second could not be caught in its first
        # 0.2 s, and one that came at his back unseen could be caught the
        # moment 0.2 s had passed. A thing he never saw is not caught.
        if not self._in_time(p, b):
            return False                      # seen too late to answer
        at = np.round(self._fly_pose(b)).astype(np.int64)
        e = self._body_eid(b)                 # read before it lands
        nx, ny, nz = self.shape
        # WHAT WAS THERE BEFORE IT LANDED, in the columns it lands in (a
        # blocked cell piles upward): the hand closes on what came out of
        # the air, and a cell that was full before is not that
        lo = np.clip(at.min(axis=0), 0, (nx - 1, ny - 1, nz - 1)) \
            if len(at) else np.zeros(3, np.int64)
        hi = np.clip(at.max(axis=0), 0, (nx - 1, ny - 1, nz - 1)) + 1 \
            if len(at) else np.ones(3, np.int64)
        box = (slice(int(lo[0]), int(hi[0])), slice(int(lo[1]), int(hi[1])),
               slice(int(lo[2]), nz))
        was = self.mat[box] != AIR
        b["vel"] = np.zeros(3)                # caught: it arrives with nothing
        self._land_body(b, None)
        self._out_of_flight(b)
        if e:
            # A NAMED THING IS CAUGHT BY WHICH THING IT IS: the hand closes
            # on a cell of it, wherever the landing put it, and the grip
            # knows its eid. Not on the first cell of its flight pose that is
            # not air, which is whatever happens to be there.
            got = [tuple(int(v) for v in c) for c in at
                   if (0 <= c[0] < nx and 0 <= c[1] < ny and 0 <= c[2] < nz)
                   and int(self.eid[tuple(c)]) == e]
            got += [tuple(int(v) for v in c) for c in self.cells_of(e)[:1]]
            if got:
                m = int(self.mat[got[0]])
                word = MATNAME.get(m, "thing")
                p["held"] = {"cell": got[0], "mat": m, "label": word,
                             "arm": arm, "eid": e,
                             "name": self.name(e, "the " + word)}
                return True
        # AN UNNAMED THING IS CAUGHT BY WHAT CAME OUT OF THE AIR. This took
        # the first cell of the flight pose that was not air once it landed,
        # and that could be the floor it landed on, or the table: a hand
        # that caught a stone was found holding the floor, and taking it
        # made the floor a thing (`_adopt`). A cell that was full before the
        # landing is not what landed, and neither is flesh. The pose's own
        # cells first, in its order (the cell it always named, when that is
        # the stone); then where a blocked cell piled up to.
        new = (self.mat[box] != AIR) & ~was & (self.mat[box] != FLESH)
        o = np.array([box[0].start, box[1].start, box[2].start])
        got = [tuple(int(v) for v in c) for c in at
               if (0 <= c[0] < nx and 0 <= c[1] < ny and 0 <= c[2] < nz)
               and new[tuple(c - o)]]
        got += [tuple(int(v) for v in c + o) for c in np.argwhere(new)[:1]]
        if got:
            m = int(self.mat[got[0]])
            word = MATNAME.get(m, "thing")
            p["held"] = {"cell": got[0], "mat": m, "label": word, "arm": arm}
            # AND IT IS A THING OF WHAT LANDED ONLY (`_adopt`). Left to the
            # first look, the flood ran on through matter of its own
            # material, and a wooden ball caught onto a wooden table was
            # adopted with the table, as one thing.
            landed = {tuple(int(v) for v in c + o) for c in np.argwhere(new)}
            e = self._adopt(*got[0], word, by=p, within=landed)
            if e:
                p["held"]["eid"] = e
                p["held"]["name"] = self.name(e, "the " + word)
        return True

    def _grip_holds(self, p, cluster):
        """Whether this body's grip can CARRY that load where it is: strength
        against weight, the holder's own feet against the floor — and a hand
        holds things UP, so a load starting at or above the holder's own crown
        is not hanging from anything. (Reach already ignores height, a stated
        softness: an ankle 1.5 m overhead can be GRIPPED, but not borne.)

        A FIST CAN PULL, NOT CLAMP. What hangs BELOW the hold is a pendulum
        and asks the wrist for nothing; a load whose weight rides ABOVE the
        hold is balanced on the fist, and the wrist pays weight times lever
        to keep it upright. An axe at the fist costs a few newton-metres and
        is held; a man gripped by the ankle is hundreds, and rotates out of
        the hand — the grip on the ankle survives, the CARRY does not."""
        if not p["alive"] or not p["awake"] or p["safe"]:
            return False
        mine, msl = self._person_cells(p)
        if mine is None or not mine.any():
            return False
        own = self._claim_cells(p, mine, msl)
        if not self._underfoot(own):
            return False
        cluster = np.asarray(cluster)
        if float(cluster[:, 2].min()) >= float(own[:, 2].max()):
            return False
        w_N, _ = self._effort(cluster)
        if w_N > p.get("strength_N", BODY["strength_N"]):
            return False
        # THE FIST IS AT THE END OF AN ARM, not wherever two bodies happen to
        # press together — measured against the nearest touching cell, a man
        # landed against his holder's chest read as gripped at the chest, and
        # hung there. The load is held at its cell nearest a fist.
        fists = self._fists(p, own)
        d2 = ((cluster[None, :, :].astype(np.float64)
               - np.asarray(fists)[:, None, :]) ** 2).sum(-1)
        touch_z = float(cluster[int(d2.min(0).argmin()), 2])
        kg = self.smass[tuple(cluster.T)]
        com_z = float((cluster[:, 2] * kg).sum() / max(float(kg.sum()), 1e-9))
        lever_m = max(0.0, com_z - touch_z) * 0.1 * self.scale
        return w_N * lever_m <= BODY["wrist_Nm"]

    def _grip_cells(self):
        """Every cell a HAND is holding up, for the support law. A grip is an
        edge in the support graph: what a footed body holds, within its
        strength, is supported THROUGH the body — the way anything in a hand
        is — not through the lattice, which is why a hanging man does not
        need his flesh to span like a girder."""
        if not any(p.get("held") or p.get("dragging") for p in self.persons):
            return frozenset()
        out = set()
        for p in self.persons:
            for cl in self._held_clusters(p):
                if self._grip_holds(p, cl):
                    out.update(map(tuple, np.asarray(cl)))
        return frozenset(out)

    def _borne_cells(self):
        """A BODY IS HELD TOGETHER BY ITS SKELETON, not by its flesh spanning.

        `_grip_cells` already says a hanging man does not need his flesh to
        span like a girder — but it said it only about what his HAND holds, and
        never about the man. So a body was a pile of meat as far as the support
        law was concerned: load travelled sideways through it one span per hop,
        and an arm held straight out was a cantilever of flesh.

        It was within a hair of breaking, and a wider arm found the hair. One
        more voxel of arm put the outer column one hop past what flesh spans,
        and the support law tore three cells off the end of a reaching hand and
        dropped them on the floor — mass conserved, man in two pieces. The
        flesh was not wrong and the span was not wrong: a limb is carried by
        BONE, and there is no bone in the lattice.

        So a body with something under its feet seeds its own cells as
        supported, exactly as the ground does and exactly as a grip does. It is
        the same edge in the same graph, and it is what a skeleton is. A body
        with nothing under its feet is not seeded — it has already left the
        lattice as a body by then (`_law_footing`), which is the other way a
        person stops needing to span."""
        out = set()
        for p in self.persons:
            comp, sl = self._person_cells(p)
            if comp is None or not comp.any():
                continue
            c = self._claim_cells(p, comp, sl)
            if self._underfoot(c):
                # the claim's own frozenset of these cells, when this is the
                # claim it made it from: the same tuples of np.int64, made once
                ent = self._claim_entry(p, comp)
                out.update(ent[2] if ent is not None and ent[3] is sl
                           else map(tuple, c))
        return frozenset(out)

    def _rub(self, p, own):
        """Dragging a thing over a floor HEATS BOTH OF THEM.

        Work is force times distance and friction is a force, so a thing hauled
        one voxel over the ground has turned `drag_N * voxel` joules of muscle
        into heat — there is nowhere else for it to have gone. Split evenly
        between the thing and what it is dragged over, because a rubbing pair
        is two surfaces and the third law does not care which one you were
        thinking about.

        And this needed NO NEW RULE. `E` is the field combustion already reads,
        so a thing dragged far enough over a rough floor gets hot, and a thing
        that gets hot enough catches. Nobody wrote "dragging can start a fire";
        it is what `mu N v` and a combustion law mean together.

        Known softness, and it is the body's: muscle is an unmodelled source of
        energy here. A man has `blood_o2` and no calorie budget, so the joules
        he puts into the floor come from nowhere the sim is counting."""
        vox_m = 0.1 * self.scale
        nz = self.shape[2]
        for cl in self._held_clusters(p):
            cl = np.asarray(cl, np.int64)
            if not len(cl):
                continue
            rub = self._contact(cl, ignore=own)
            if not len(rub):
                continue                      # carried, not dragged: no rubbing
            joules = float(self._effort(cl, kinetic=True)[1]) * vox_m
            if joules <= 0.0:
                continue
            # A BODY WITH A HEAT RECORD PAYS FOR WHAT IT DRAGS (T7, wave 13b): the
            # work comes out of its stores at 22 percent efficiency, the rest of what
            # it burned warms its muscle, and the work_out book it was credited with
            # is spent in the floor below. A body with no record is as it was.
            if p.get("heat") is not None:
                if self.note_work(p, "drag@%d" % self.tick, joules, "drag") is not None:
                    self.release_work(joules)
            half = joules * 0.5 / len(rub)
            self.E[tuple(rub.T)] += half
            under = rub.copy()
            under[:, 2] -= 1
            under = under[under[:, 2] >= 0]
            if len(under):
                self.E[tuple(under.T)] += joules * 0.5 / len(under)

    def _haul_cost(self, p, own):
        """How much of this body's strength its load is already using.

        Two costs, because they are two different things. A thing LIFTED is
        held up, and what it costs is its weight. A thing TRAILED along the
        floor is not held up at all, and what it costs is friction — the same
        `_effort` a shove asks about, so a heavy thing on a rough floor is dear
        and the same thing on ice is not."""
        cost = 0.0
        for cl in self._held_clusters(p):
            cl = np.asarray(cl, np.int64)
            if not len(cl):
                continue
            lift_N, drag_N = self._effort(cl)
            cost += drag_N if len(self._contact(cl, ignore=own)) else lift_N
        return cost

    def _held_clusters(self, p):
        """Everything in this body's hands, thing or person, as cell arrays.
        Whether the grip can CARRY any of it is _grip_holds' question."""
        out = []
        obj = self._held_cells(p)
        if obj is not None:
            out.append(obj)
        who = p.get("dragging")
        if who:
            q = next((r for r in self.persons if r["name"] == who), None)
            if q is not None and q["alive"] and not q["safe"]:
                comp, sl = self._person_cells(q)
                if comp is not None and comp.any():
                    qc = self._claim_cells(q, comp, sl)
                    out.append(qc)
        return out

    def _take_up(self, p, own):
        """Bring a held thing to the HAND, if the arm can lift it. A liftable
        thing hangs from the fist — where a swing will find it, and where the
        grip (not the floor) carries it. Too heavy stays where it lies: still
        held, a grip is a constraint before it is a lift."""
        obj = self._held_cells(p)
        if obj is None:
            return
        arm = (p.get("held") or {}).get("arm") \
            or self._hand(p, own, toward=obj.mean(axis=0))
        if arm is None:
            return
        if self.contact_laws and self.layers is not None:
            contact.grasp(self, p, obj, arm)    # the hand meets the thing (T3)
        w_N, _ = self._effort(obj)
        if w_N > self._hand_strength(p, arm):    # the off hand is weaker, and
            return                               # that is what dominance MEANS
        hand = self._fist_of(p, arm, own)
        if hand is None:
            return
        hand = np.array([int(round(c)) for c in hand], np.int64)
        top = int(obj[:, 2].max())
        # HANG IT BESIDE THE FIST, NOT THROUGH THE CHEST. Centred ON the hand, a
        # load any wider than a hand reaches back into the body of whoever is
        # holding it, and the move is refused for the perfectly good reason that
        # a man is already standing there — measured, a 38 kg block a 4000 N man
        # could obviously lift stayed on the floor being dragged instead.
        # Outward is away from the body's own middle, which is the side the arm
        # is on.
        cx, cy = float(own[:, 0].mean()), float(own[:, 1].mean())
        ax = 0 if abs(hand[0] - cx) >= abs(hand[1] - cy) else 1
        away = 1 if (hand[ax] - (cx if ax == 0 else cy)) >= 0 else -1
        beside = [int(hand[0]), int(hand[1])]
        beside[ax] += away * (int(np.ptp(obj[:, ax])) // 2 + 1)
        d = (beside[0] - int(round(float(obj[:, 0].mean()))),
             beside[1] - int(round(float(obj[:, 1].mean()))),
             int(hand[2]) - 1 - top)
        if d != (0, 0, 0) and self._shove(obj, *d):
            c = p["held"]["cell"]
            p["held"]["cell"] = (c[0] + d[0], c[1] + d[1], c[2] + d[2])

    def _haul_held(self, p):
        """Bring the held THING along, exactly as _haul brings a person: it
        trails toward the hauler, and an arm stretched past its reach is not
        holding anything. No brace, no resistance — things do not fight."""
        obj = self._held_cells(p)
        if obj is None:
            return
        mine, msl = self._person_cells(p)
        if mine is None or not mine.any():
            return
        hcell = self._claim_cells(p, mine, msl)
        if not self._in_reach(hcell, obj):
            what = self.name(
                self._held_eid(p),
                f"the {MATNAME.get(int(self.mat[tuple(obj[0])]), 'thing')}")
            p["held"] = None
            self.emit("releases", actor=p["pid"],
                      text=f"{p['name']} loses hold of {what} "
                           f"— an arm is only so long",
                      to=p, site="haul_held:loses_hold")
            return
        _lift, drag_N = self._effort(obj)
        if min(_lift, drag_N) > p.get("strength_N", BODY["strength_N"]):
            return                            # the grip holds; the pull fails
        dx = float(hcell[:, 0].mean()) - float(obj[:, 0].mean())
        dy = float(hcell[:, 1].mean()) - float(obj[:, 1].mean())
        if max(abs(dx), abs(dy)) <= 0.5 * (self._span_xy(hcell)
                                           + self._span_xy(obj)):
            return                            # already in hand
        order = ((int(np.sign(dx)), 0), (0, int(np.sign(dy)))) \
            if abs(dx) >= abs(dy) else ((0, int(np.sign(dy))), (int(np.sign(dx)), 0))
        for step in order:
            if step != (0, 0) and self._shove(obj, *step):
                c = p["held"]["cell"]
                p["held"]["cell"] = (c[0] + step[0], c[1] + step[1], c[2])
                return

    def _law_grips(self):
        """A HOLD IS RE-EARNED EVERY TICK, not only when its owner walks.

        This test lived inside `_haul`, which runs when a body takes a STEP —
        so a holder who stands still never asked it again. Measured on the
        rescue: a man on a ledge went on holding somebody lying on the ground
        twenty voxels below him for two hundred and sixty ticks, because he had
        no reason to walk anywhere and the only thing that would have checked
        was walking.

        Whether an arm still reaches is a question about now. Nothing about it
        is about stepping."""
        for p in self.persons:
            who = p.get("dragging")
            if not who:
                continue
            q = next((r for r in self.persons if r["name"] == who), None)
            if q is None or q["safe"] or not q["alive"]:
                p["dragging"] = None
                continue
            # THE ONE HE IS HOLDING MAY BE IN THE AIR — and that is not a
            # reason to stop checking, it is the whole reason to check. This
            # skipped the test outright while ANY flesh was mid-fall, for
            # EVERY holder in the world, and handed the case to `_haul`, which
            # only runs when somebody walks. So a man caught at the lip was
            # never re-examined for as long as he hung there: measured on this
            # scene, the rescuer went on holding him at 12 voxels, then 14,
            # then 18, and let go at 32 — a metre and a half — and only then
            # because the man had landed and become checkable again.
            #
            # A body off the lattice still has cells and still has somewhere to
            # be. Ask where they are.
            comp, sl = self._person_cells(q)
            mine, msl = self._person_cells(p)
            if mine is None or not mine.any():
                continue
            if comp is not None and comp.any():
                theirs = self._claim_cells(q, comp, sl)
            else:
                air = next((x for x in self.bodies
                            if x.get("owner") == q["name"] and not x.get("part")
                            and (x["mats"] == FLESH).any()), None)
                if air is None:
                    continue
                # A BODY OFF THE LATTICE IS ONE OF TWO THINGS and they are not
                # posed the same way. A flier carries an `off` and is placed by
                # it; a body mid-TOPPLE carries a `theta` and is swung about
                # its pivot. Asking a toppling body for its `off` is a KeyError
                # — found by the harvester on its seventieth random world,
                # which is the first time in this repo that a man has been
                # holding somebody who then tipped over rather than fell.
                theirs = np.round(self._fly_pose(air) if air.get("fly")
                                  else self._body_pose(air, air["theta"])
                                  ).astype(np.int64)
            hcell = self._claim_cells(p, mine, msl)
            reach_v = self._reach_of(p) / max(0.1 * self.scale, 1e-9)
            if not self._in_reach(hcell, theirs, reach_v):   # VOXELS, not metres
                p["dragging"] = None
                self.emit("releases", actor=p["pid"],
                          target=("person", q["pid"]),
                          text=f"{p['name']} loses hold of {who} "
                               f"— an arm is only so long",
                          to=p, site="law_grips:loses_hold")

    def _haul(self, p, sx, sy):
        """Bring along whoever this body took hold of. The dragged body is not
        a passenger with special rules — it is a thing being shoved, moved by
        the same _shove that shifts a crate, and it is left behind the moment
        it will not shift (wedged, or grown too heavy to matter). When both are
        at the door, both are out; that is the only thing rescuing anyone ever
        needed to mean.

        WHAT IS DRAGGED FOLLOWS. It used to be shoved along the hauler's own
        step vector, which is a grip only while the two happen to move alike:
        once the load snagged for a tick, the hauler walked on and the pair
        drifted apart, still joined — measured, a rescuer at x=29 towing a body
        at x=36, seven voxels AHEAD of them and pulling it further. A load
        trails the thing pulling it, so the step is toward the hauler, and an
        arm that has been stretched past its reach is not holding anything."""
        who = p.get("dragging")
        if not who:
            return
        q = next((r for r in self.persons if r["name"] == who), None)
        if q is None or q["safe"] or not q["alive"]:
            p["dragging"] = None
            return
        # THEY MAY BE IN THE AIR. An unconscious body is promoted to a free body
        # while it keels over, and for those ticks it is off the lattice or only
        # half back on it. Grabbing at it then fails — and letting go on that
        # failure ended every rescue at precisely the moment the person being
        # rescued finished falling over, which is when they need carrying most.
        # Wait for them to come down; a grip is not lost because someone moved.
        if any((b["mats"] == FLESH).any() for b in self.bodies):
            return
        comp, sl = self._person_cells(q)
        if comp is None or not comp.any():
            p["dragging"] = None
            return
        cells = self._claim_cells(q, comp, sl)
        mine, msl = self._person_cells(p)
        if mine is None or not mine.any():
            return
        hcell = self._claim_cells(p, mine, msl)
        if not self._in_reach(hcell, cells):
            p["dragging"] = None                  # stretched past an arm: gone
            self.emit("releases", actor=p["pid"], target=("person", q["pid"]),
                      text=f"{p['name']} loses hold of {who} "
                           f"— an arm is only so long",
                      to=p, site="haul:loses_hold")
            return
        # HE MAY BE OVER THE EDGE. With nothing underfoot, what an arm holds is
        # his WEIGHT and not the friction of a floor — the same _effort, its
        # other half. The support law now knows a grip carries load
        # (`_grip_cells`), so a man whose weight the arm can take HANGS from
        # it: no step is made, and "left holding him over the drop" is a state
        # of the world. Hauling him back up is not built. A load beyond the
        # arm — or one wholly above the holder's own crown, which no grip
        # bears — goes, exactly as before.
        if not self._underfoot(cells):
            if self._grip_holds(p, cells):
                if not p.get("_held_over"):
                    p["_held_over"] = True
                    self.emit("holds", actor=p["pid"],
                              target=("person", q["pid"]), result="over_drop",
                              text=f"{p['name']} is left "
                                   f"holding {who} over the drop",
                              to=p, site="haul:held_over")
                return
            p["dragging"] = None
            self.emit("releases", actor=p["pid"], target=("person", q["pid"]),
                      result="not_enough",
                      text=f"{who} goes over, and the grip "
                           f"is not enough to hold a hanging man",
                      to=p, site="haul:goes_over")
            return
        p["_held_over"] = False
        # AND HE MAY NOT WANT TO COME. Asked every tick, because the answer
        # changes: the same pull that a braced man shrugs off for twenty ticks
        # succeeds the moment his footing is gone or someone stronger takes over.
        if self._resist(q, cells) > p.get("strength_N", BODY["strength_N"]):
            return                                # the grip holds; the pull fails
        dx = float(hcell[:, 0].mean()) - float(cells[:, 0].mean())
        dy = float(hcell[:, 1].mean()) - float(cells[:, 1].mean())
        if max(abs(dx), abs(dy)) <= 0.5 * (self._span_xy(hcell)
                                           + self._span_xy(cells)):
            return                                # already at their heels
        order = ((int(np.sign(dx)), 0), (0, int(np.sign(dy)))) \
            if abs(dx) >= abs(dy) else ((0, int(np.sign(dy))), (int(np.sign(dx)), 0))
        for step in order:
            if step != (0, 0) and self._shove(cells, *step):
                self._carry_own(q, step[0], step[1])
                q["anchor"] = (int(round(float(cells[:, 0].mean()))) + step[0],
                               int(round(float(cells[:, 1].mean()))) + step[1])
                return
        p["dragging"] = None                      # it would not come; let go

    def _object_at(self, x, y, z, cap=4000):
        """The connected thing that voxel belongs to — the same key (3.3),
        flood filled. Unnamed matter is whatever is joined to whatever you
        grabbed, of the same material (which is why an unnamed table with
        iron legs is two things: joints do not exist yet); a named thing is
        its own cells, whatever each is made of.

        Drawn particle matter (`moving`) is never part of a thing here: it is
        the solver's, and it is not where it is drawn (`_objects_within_reach`
        says why)."""
        m0 = int(self.mat[x, y, z])
        if m0 == AIR or self.moving[x, y, z]:
            return np.zeros((0, 3), np.int32)
        # depth first, and the ORDER matters: the first cell is the one a
        # take row names (label.py, `flood`)
        if not self._thing_lut.any():
            # the key IS `mat` (`_key`), and costs nothing to have
            return flood(self.mat, x, y, z, cap, block=self.moving,
                         fused=self.fused)
        # A BOX ROUND THE CELL, NOT THE WORLD. Once anything is named the key
        # is an array to build, and building it for the whole house to flood
        # one stick was most of what a take cost. Flood a box round the
        # cell instead, and trust the answer only if no cell of it lies on a
        # face of the box that is not a face of the world: then no cell it
        # met had a neighbour outside, so the flood met every cell, in the
        # same order, that the whole world's would have. Otherwise the box
        # was too small: double it and ask again.
        nx, ny, nz = self.shape
        at, r = (int(x), int(y), int(z)), 8
        while True:
            lo = [max(c - r, 0) for c in at]
            hi = [min(c + r + 1, n) for c, n in zip(at, (nx, ny, nz))]
            sl = tuple(slice(a, b) for a, b in zip(lo, hi))
            got = flood(np.ascontiguousarray(self._key(sl)), at[0] - lo[0],
                        at[1] - lo[1], at[2] - lo[2], cap,
                        block=np.ascontiguousarray(self.moving[sl]),
                        fused=self.fused)
            short = False
            for a, n in enumerate((nx, ny, nz)):
                if (lo[a] > 0 and (got[:, a] == 0).any()) or \
                        (hi[a] < n and (got[:, a] == hi[a] - lo[a] - 1).any()):
                    short = True
                    break
            if not short:
                return (got + np.array(lo, np.int32)).astype(np.int32)
            r *= 2

    def _effort(self, cells, kinetic=False):
        """What it costs this body to move that thing, in newtons: the weight
        to LIFT it, the friction to SHOVE it along the floor.

        This is the whole of 'hands'. Push, pull, drag, lift and press are not
        five verbs with five rules — they are one force meeting either gravity
        or friction, and the arithmetic decides which are possible. A grown
        person comes out able to shove a loaded chest they could never pick up,
        and — the case that matters — able to DRAG someone unconscious but not
        to carry them, which is exactly how it goes."""
        if not len(cells):
            return 0.0, 0.0
        kg = float(self.smass[tuple(cells.T)].sum()) / 1000.0
        mu = FRICTION
        # T9 (wave 13b, C8a): the friction of the PAIR of surfaces and the film
        # between them. The force that STARTS a haul is mu_s m g; `kinetic=True`
        # (the heat of dragging, `_rub`) asks for the sliding value mu_k.
        if "friction" in self.mat_laws:
            m = matlaws.contact_mu(self, cells, kinetic)
            if m is not None:
                mu = m
        return kg * GRAVITY, kg * GRAVITY * mu

    def _joined_by_mechanism(self, cells):
        """True if any of these cells is a door leaf or frame (a thing joined
        by a mechanism joint, T11): it is operated, not shoved or lifted.
        Inert while no mechanism exists (`mechs` is absent or empty)."""
        mechs = getattr(self, "mechs", None)
        if not mechs:
            return False
        held = set()
        for rec in mechs.values():
            held.add(rec["leaf"])
            held.add(rec["frame"])
        eids = np.unique(self.eid[tuple(np.asarray(cells).T)])
        return any(int(e) in held for e in eids)

    def _shove(self, cells, dx, dy, dz=0):
        """Translate a thing by one voxel, carrying everything it holds. Refuses
        if any arriving cell is occupied by something that is not itself."""
        if not len(cells):
            return False
        if self.moving[tuple(np.asarray(cells).T)].any():
            return False                      # the solver's, not a hand's: see
                                              # `_objects_within_reach`
        if self._joined_by_mechanism(cells):
            return False                      # a hinge holds it: `operate` is the verb
        nx, ny, nz = self.shape
        tgt = cells + np.array([dx, dy, dz], np.int32)
        if (tgt < 0).any() or (tgt[:, 0] >= nx).any() \
                or (tgt[:, 1] >= ny).any() or (tgt[:, 2] >= nz).any():
            return False
        own = {tuple(c) for c in cells}
        for t in map(tuple, tgt):
            if t not in own and int(self.mat[t]) != AIR:
                return False
        self._put(tuple(tgt.T), self._lift(tuple(cells.T)))   # all of it
        self._torque_solid = None
        self._slack_mat = None
        return True
