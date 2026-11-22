"""Limb motion on the lattice: swing, turn, repose, pose and joint laws, crouch, lean."""
import hashlib
import math

import numpy as np
from .materials import (AIR, BODY, CHEST0, GRAVITY, TICK_S, _SIX)
from . import exertion as _EX
_SWEEP_RAD = 0.08              # how finely a limb's turn is swept for
                               # things in the way. One voxel at the end
                               # of a 9-voxel arm is about 0.11 rad, so
                               # this cannot step over a wall.


class PoseMixin:

    def _swing(self, p, name, toward=None, back=0.0, kind=None):
        """Put a LIMB in motion about its joint, driven by a muscle.

        This is the same rotation a toppling tree does — `_body_pose` has always
        turned a set of cells about an arbitrary pivot — with two things
        changed: the pivot is INSIDE the body, and the acceleration comes from a
        muscle instead of from gravity. That is the whole of what a shoulder is.

        It is not animation, and the distinction is sharp in a sim where
        everything reads the lattice. The arm's voxels leave their cells and
        arrive in new ones, so while it is coming round the arm really is
        somewhere else: it blocks what it now occupies, its mass sits where it
        now sits, and whatever stops it is paid the rotational energy it had.
        A pose that only changed the picture would need a picture to change,
        and there isn't one — the renderer reads `mat` like everything else."""
        segs, joints = p.get("segs"), p.get("joints")
        bones = [b for b in self._bones(p, name) if b in (segs or {})]
        if not segs or not bones:
            return None
        # WHICH WAY. The arm hangs from the shoulder, so turning it about the
        # sideways axis sweeps it forward; the sign is which forward.
        dx, dy = toward if toward else p.get("facing", (1.0, 0.0))
        axis = 0 if abs(dx) >= abs(dy) else 1
        s = -1 if (dx if axis == 0 else dy) >= 0 else 1
        comp, sl = self._person_cells(p)
        if comp is None or not comp.any():
            return None
        own = self._claim_cells(p, comp, sl)
        parts = self._limb_parts(p, name, own)
        if not parts or not sum(len(c) for c in parts):
            return None                       # the limb is not there any more
        limb = np.concatenate(parts)
        # WHICH BONE EACH VOXEL IS. A whole arm swings as ONE rigid thing —
        # the elbow is locked by the same muscles that drive the shoulder —
        # but it has to come to rest as an arm with an elbow in it, or the
        # next pose reaches for bones that are no longer where they are said
        # to be. -1 is a carried tool: along for the ride, part of nobody.
        segof = np.concatenate([np.full(len(c), i, np.int32)
                                for i, c in enumerate(parts)])
        # A HELD THING SWINGS WITH THE ARM — that is most of what holding a
        # tool is for. Its cells join the limb's rigid body, so its mass slows
        # the swing (Hill does the rest) and its edge is what arrives.
        obj = self._held_cells(p)
        if obj is not None and self._in_reach(limb, obj):
            limb = np.concatenate([limb, obj])
            segof = np.concatenate([segof, np.full(len(obj), -1, np.int32)])
        piv = self._pivot_of(p, name)
        if piv is None:
            return None
        jx, jy, jz = (piv + own.min(axis=0)).astype(np.int64)
        vox_m = 0.1 * self.scale
        kg = self.smass[tuple(limb.T)] / 1000.0
        r = (np.abs(limb[:, axis] - float(jx if axis == 0 else jy))
             + np.abs(limb[:, 2] - float(jz))) * vox_m
        inertia = float((kg * r * r).sum())
        if inertia <= 0.0:
            return None
        b = self._launch(limb, (0.0, 0.0, 0.0), owner=p["name"], part=True)
        b["fly"] = False                      # it turns, it does not travel
        b["axis"], b["s"] = axis, s
        b["pivot"] = float(jx if axis == 0 else jy)
        b["zb"] = float(jz)
        b["theta"], b["omega"] = 0.0, 0.0
        b["Nm"], b["I"] = BODY["arm_Nm"], inertia
        if self.exertion_from_body:
            # THE MUSCLE IS THIS PERSON'S (wave 14b): the stock adult's arm torque scaled
            # by their strike over the stock strike (strength, training, technique, a
            # spread of their own, fatigue and injury: exertion.strike). The work the
            # arm does is paid from their books as it is done (`_pay_swing`).
            b["exkind"] = kind or ("throw" if back else "punch")
            b["Nm"] = max(BODY["arm_Nm"] * _EX.torque_scale(p, b["exkind"]), 1e-6)
            b["paid_J"] = 0.0
        b["seg"] = name
        b["segof"], b["bones"] = segof, bones
        # WHERE THE REST OF HIM WAS, so the arm can come back to the body and
        # not to the place: a man may step while his arm is out (`_land_body`).
        arm = {tuple(c) for c in limb.tolist()}
        rest_of = np.array([c for c in own.tolist() if tuple(c) not in arm])
        if len(rest_of):
            b["body_at"] = rest_of.mean(axis=0)
        # AND IT MAY START BEHIND THE BODY. `back` is how far, and it is
        # honoured only if the arm can actually BE there — a man in a doorway
        # with a wall at his shoulder throws from where he stands, and throws
        # worse, which is right.
        if back:
            nx, ny, nz = self.shape
            mine = {tuple(c) for c in limb}
            for tryback in (back, back * 0.5):
                at = np.round(self._body_pose(dict(b, cells=limb.astype(
                    np.float64)), -tryback)).astype(np.int64)
                if ((at < 0).any() or (at[:, 0] >= nx).any()
                        or (at[:, 1] >= ny).any() or (at[:, 2] >= nz).any()):
                    continue
                if any(tuple(t) not in mine and int(self.mat[tuple(t)]) != AIR
                       for t in at):
                    continue
                b["theta"] = -float(tryback)
                break
        b["phi0"], b["L"] = 0.02, 2.0
        return b

    def _turn(self, p, quarters):
        """Turn a body on its own feet, in QUARTER turns.

        A quarter turn is the only rotation a lattice can do EXACTLY. It is a
        permutation of the cells — (dx, dy) becomes (-dy, dx) and back again —
        so nothing rounds together and nothing rounds apart, and the two checks
        `_repose` needs for every other angle are not needed for this one at
        all. Everything else in this file that turns has to argue with the
        grid; this does not.

        Which is why a body could not turn until now, and why it matters that
        it can. The humanoid is built with its shoulders along x and its
        `facing` rotates freely, so it FACES ALONG ITS OWN SHOULDER LINE half
        the time — and an arm turns in the plane the body faces, so half the
        time that plane holds the torso too. That is one cause behind a
        wind-up that sweeps an arm through its own chest, a thicker arm that
        collides with its own body when it reaches, and a pose space that
        depends on which way a body happens to be pointed.

        It can be REFUSED, and that is a feature: a body in a space too tight
        to turn in does not turn. Corridors are like that."""
        if self._swinging(p):
            return False                      # an arm is out: see `_swinging`
        comp, sl = self._person_cells(p)
        if comp is None or not comp.any():
            return False
        own = self._claim_cells(p, comp, sl)
        q = int(quarters) % 4
        if q == 0:
            return True
        cx = int(round(float(own[:, 0].mean())))
        cy = int(round(float(own[:, 1].mean())))

        def spin(cells):
            c = np.asarray(cells, np.int64).copy()
            dx, dy = c[:, 0] - cx, c[:, 1] - cy
            for _ in range(q):
                dx, dy = -dy, dx
            c[:, 0], c[:, 1] = cx + dx, cy + dy
            return c

        want = spin(own)
        nx, ny, nz = self.shape
        if ((want < 0).any() or (want[:, 0] >= nx).any()
                or (want[:, 1] >= ny).any() or (want[:, 2] >= nz).any()):
            return False
        mine = {tuple(c) for c in own}
        if any(tuple(t) not in mine and int(self.mat[tuple(t)]) != AIR
               for t in want):
            return False                      # no room to turn round in
        # THE AIR IT TURNS INTO MOVES TO WHERE IT TURNED OUT OF (wave 13b, C6; the
        # walking leak of T7, -1.21 J a turn for a 34 C body). The turn used to put the
        # body down on its new cells and leave the old cells as the lift left
        # them, so the energy (and the oxygen, smoke and fluid) of the air it
        # turned into was written over and was gone. Exactly what `_walk` does
        # with the cells it enters: lift that air, move the body, and put the
        # air in the cells the body has left. At E = 0 (every old scene) the
        # lifted air is empty and nothing differs.
        new = {tuple(int(v) for v in t) for t in want}
        entered = [tuple(int(v) for v in t) for t in want if tuple(int(v) for v in t) not in mine]
        vacated = [tuple(int(v) for v in c) for c in own if tuple(int(v) for v in c) not in new]
        body = self._lift(tuple(own.T))                           # all it carries
        if entered:
            ent = tuple(np.array(entered, np.int64).T)
            vac = tuple(np.array(vacated, np.int64).T)
            shoved = self._lift(ent)
        self._put(tuple(want.T), body)
        if entered:
            self._put(vac, shoved)
        # THE DEED TURNS WITH THE FLESH. The claim seeds each body with what
        # it owned a moment ago; left naming the cells he turned out of, it
        # seeded him short and a neighbour could grow into the rest of him.
        self._move_own(p, own, want)
        # AND EVERY PART OF IT TURNS WITH IT. Segments, joints and rest shapes
        # are offsets from the body's own corner, and that corner has moved —
        # so each is rotated about the same centre and re-based, or the body
        # keeps a map of a shape it no longer has.
        base = want.min(axis=0)
        for book in ("segs", "rest", "joints"):
            d = p.get(book)
            if not d:
                continue
            origin = own.min(axis=0)
            for k, v in list(d.items()):
                v = np.asarray(v)
                flat = v.ndim == 1
                a = spin((v.reshape(1, 3) if flat else v) + origin) - base
                d[k] = a[0] if flat else a
        fx, fy = p.get("facing", (1.0, 0.0))
        cx_, cy_ = p.get("chest", CHEST0)
        for _ in range(q):
            fx, fy = -fy, fx
            cx_, cy_ = -cy_, cx_
        p["facing"] = (float(fx), float(fy))
        p["chest"] = (float(cx_), float(cy_))
        p["_claim_tick"] = None               # it is a different shape now
        self._torque_solid = None
        self._slack_mat = None
        return True

    def _swinging(self, p):
        """Is a limb of this body out in a swing (a free body of its own)?

        A BODY WITH AN ARM OUT IN A SWING DOES NOT TURN ON ITS FEET. The arm
        is off the lattice, so the turn moved the rest of him and not it,
        and the arm came back to where it had left from: after a half turn,
        that is where his OTHER arm now was, and every cell of it was piled
        up its column onto him (fuzzer seeds 17, 111 and 134, 54 cells each:
        a stride's square-up in the tick of a swing). He turns once the arm
        is back."""
        return any(b.get("part") and b.get("owner") == p["name"]
                   for b in self.bodies)

    def _square_up(self, p):
        """Turn the body to face where it is about to walk, if there is room.

        `facing` was only ever a direction for the eyes and the swing; the
        flesh stayed as built, shoulders along x, chest to +y — so a man
        walking east walked along his own shoulder line, every render showed
        a crowd all facing one way, and which side of him an arrow met did not
        depend on which way he had turned. Quarter turns only (`_turn` is
        exact), and only a free body on its feet: one holding or held by
        something would leave it behind, and one in the air has no feet to
        turn on, nor one with an arm out in a swing (`_swinging`). A turn the
        room refuses is remembered until the facing changes, so a body in a
        corridor is not asked every tick."""
        fx, fy = p.get("facing", (1.0, 0.0))
        want = ((1.0 if fx > 0 else -1.0, 0.0) if abs(fx) >= abs(fy)
                else (0.0, 1.0 if fy > 0 else -1.0))
        if p.get("_squared_for") == want:
            return False
        p["_squared_for"] = want
        if p.get("held") or p.get("dragging") \
                or any(q.get("dragging") == p["name"] for q in self.persons) \
                or self._swinging(p):
            p["_squared_for"] = None          # ask again once it is free
            return False
        cx_, cy_ = p.get("chest", CHEST0)
        for q in range(4):
            if abs(cx_ - want[0]) < 1e-6 and abs(cy_ - want[1]) < 1e-6:
                break
            cx_, cy_ = -cy_, cx_
        if q == 0:
            return False
        face = p.get("facing", (1.0, 0.0))
        if self._turn(p, q):
            p["facing"] = face                # it turned TO its facing
            return True
        return False

    def _repose(self, p, name, theta, toward=None, dry=False, ctx=None):
        """Put one limb where its ANGLES say it is, on the lattice, now.

        With `dry`, ASK instead: every check runs and nothing moves. That is
        what lets a body find out what it could do without doing it — which the
        lattice makes worth having, because whether a pose is possible at all
        is a question with a surprising answer here.

        This is the difference between a pose and a picture. The limb is not
        lifted off the world, turned, and set back down — its voxels are moved
        from the cells they are in to the cells the angles ask for, so while the
        arm is out it really IS out: it blocks what it now occupies, its mass
        sits where it now sits, a hand at the end of it is somewhere new, and
        anything already in the way STOPS it.

        A LIMB IS A CHAIN OF BONES, each turning about its own joint, and the
        bones nearer the body carry the ones further out along with them. One
        angle per joint; a bare number means the joint nearest the body and the
        rest held straight. An arm with one bone is the same code with a chain
        of length one, which is why nothing above this had to learn about
        elbows.

        Answers WHY it could not, because the two reasons are not alike.
        "blocked" is the world saying no — something solid is there, and an arm
        that meets a post stops at the post. "undrawable" is the LATTICE saying
        no: a limb one voxel wide has only a handful of angles it can be drawn
        at, because at the others two of its voxels round into the same cell and
        a voxel of flesh would be destroyed by the act of moving. A limb is
        never allowed to cost the body mass, so those angles are skipped over —
        the arm holds its last drawable pose while the angle goes on, and
        catches up when the two agree again. At 5 cm that is not a workaround,
        it is what a thin thing turning on a coarse grid IS.

        Turned from the limb's REST shape rather than from wherever it happens
        to be, so angles do not accumulate the rounding of every angle before
        them — and, for a chain, so that the elbow can be turned about where it
        RESTS rather than about wherever the shoulder has just carried it."""
        # ASKED MANY TIMES OVER, OF ONE BODY THAT HAS NOT MOVED. A search for
        # somewhere to put an arm (`_reach_around`, `_bend_max`) asks this
        # dozens of times in a row, and only the angles change between the
        # questions. Everything else is read once (`_repose_ctx`) and handed
        # back in; it is good until a pose is actually taken.
        if ctx is None:
            ctx = self._repose_ctx(p, name, toward)
        if isinstance(ctx, str):
            return ctx                        # it cannot be asked at all
        segs, joints, rest = ctx["segs"], ctx["joints"], ctx["rest"]
        bones, moving = ctx["bones"], ctx["moving"]
        own, origin, cur_parts = ctx["own"], ctx["origin"], ctx["cur_parts"]
        # TWO ANGLES PER BONE, laid out flat: swing, spread, swing, spread.
        # A bare number is still the joint nearest the body swinging and
        # nothing else, which is what every caller meant before joints had a
        # second axis — so nothing above this had to learn about it either.
        th = self._angles(theta, 2 * len(bones))
        parts = self._repose_place(ctx, th)
        want = np.round(np.concatenate(parts)).astype(np.int64)
        # THE ROTATION COULD NOT DRAW IT — try the shears, which always can.
        # An arm that lost a voxel to rounding used to simply not move; now it
        # takes the bijective path instead and moves about a voxel wide of
        # true, which is a far better answer than standing still. Measured: a
        # bone held 4 of 31 angles under the rotation alone, 27 with this.
        #
        # ONLY EVER AN IMPROVEMENT, and that condition is the whole of it. The
        # shears put the limb somewhere slightly else, and somewhere slightly
        # else can be inside a wall — so the swap is taken only when the shear
        # pose is one the body can actually BE in. Without that, a search for
        # somewhere to put an arm traded "the lattice cannot draw this" for
        # "the post is in the way", learned nothing, and a man beside a post
        # stopped being able to reach at all.
        nx, ny, nz = self.shape
        flat = lambda a: (a[:, 0] * ny + a[:, 1]) * nz + a[:, 2]
        if len(np.unique(flat(want))) != len(want):
            alt_parts = self._repose_place(ctx, th, shear=True)
            alt = np.concatenate(alt_parts)
            ok = (len(np.unique(flat(alt))) == len(alt)
                  and (alt >= 0).all()
                  and (alt[:, 0] < nx).all() and (alt[:, 1] < ny).all()
                  and (alt[:, 2] < nz).all())
            if ok:                            # every cell its own or air
                ok = bool(((self.mat[tuple(alt.T)] == AIR)
                           | np.isin(flat(alt), ctx["cur_keys"])).all())
            if ok:
                parts, want = alt_parts, alt
        ride, moved_hold = ctx["ride"], None
        if ctx["ride_mode"] == "world":
            # WHAT THE WORLD IS HOLDING UP DOES NOT COME WITH YOU (see
            # `_repose_ctx`): the post stays and the ARM has to reach. If the
            # pose would put the fist further from it than an arm is long, it
            # is refused — a limit that is about the body rather than about
            # what happens to be under the thing it is holding.
            tip = moving.index(bones[-1])
            fist = np.round(parts[tip][0]).astype(np.int64)
            span = float(np.abs(ride - fist).sum(axis=1).min()) \
                * 0.1 * self.scale
            if span > self._reach_of(p):
                return "blocked"              # the arm does not stretch
        elif ctx["ride_mode"] == "hand":
            # A HELD THING HANGS FROM THE FIST — it does not ride CLAMPED to
            # it. So it follows where the hand goes and keeps its own attitude:
            # gravity has a say in which way up a carried thing is, and the
            # wrist is a joint whether or not this sim models one yet. Turned
            # rigidly about the shoulder instead, a 38 kg block hanging below a
            # man's hand was swung up over his head by the act of putting his
            # arm out, which is not what carrying is.
            #
            # By WHERE THE HAND IS, not by what the angle reads: the two part
            # company whenever the lattice cannot draw an angle, because the
            # flesh holds its last good pose while the angle runs on and then
            # catches up several angles at once. Measured with the delta in
            # angles: the arm went out eight voxels and the block moved one, so
            # the lever never formed and the man kept his feet.
            #
            # The bone's voxels keep their order from `rest`, so cell 0 of the
            # far bone is the SAME piece of flesh before and after — which is
            # what makes this an honest displacement and not a guess.
            tip = moving.index(bones[-1])     # the limb's OWN far bone, which
            shift = (np.round(parts[tip][0]).astype(np.int64)   # is not the last
                     - cur_parts[tip][0])                       # of the subtree
                                              # once a head and arms come along
            ride_to = ride + shift
            want = np.concatenate([want, ride_to])
            moved_hold = (tuple(ride_to[0]), tuple(ride[0]))
        cur, mine = ctx["cur_all"], ctx["mine"]
        if ((want < 0).any() or (want[:, 0] >= nx).any()
                or (want[:, 1] >= ny).any() or (want[:, 2] >= nz).any()):
            return "blocked"
        key = flat(want)
        if len(np.unique(key)) != len(key):
            return "undrawable"               # two voxels into one: mass lost
        if not self._still_joined(p, want[:len(want) - (len(ride) if
                                  moved_hold is not None else 0)], own, mine,
                                  rest=ctx["still"]):
            return "undrawable"               # ...and two voxels into none
        if ((self.mat[tuple(want.T)] != AIR)
                & ~np.isin(key, ctx["mine_keys"])).any():
            return "blocked"                  # something is in the way, and the
                                              # way is the world, not a picture
        # AND NOTHING WAS IN THE WAY ON THE ROAD THERE. Only the pose at the
        # END of a tick used to be checked, which is honest while a limb turns
        # a little at a time — and stops being honest the moment it does not.
        # The flesh waits at every angle the lattice cannot draw and then
        # catches up several angles at once, so a rail that happens to sit in
        # the undrawable part of the sweep was never touched by anything: the
        # arm was on one side of it, and then it was on the other.
        #
        # So the angles in between are swept. They do not have to be DRAWABLE
        # — two voxels rounding together is an artifact of drawing, not an
        # event in the world — they only have to be unoccupied. This is the
        # same rule a walking body already keeps: you may not arrive somewhere
        # by passing through something.
        was_th = ctx["was_th"]
        gap = max(abs(a - b) for a, b in zip(th, was_th)) if bones else 0.0
        if gap > _SWEEP_RAD:
            for f in np.arange(_SWEEP_RAD, gap, _SWEEP_RAD) / gap:
                mid = [w + (t - w) * float(f) for w, t in zip(was_th, th)]
                for pt_f in self._repose_place(ctx, mid):
                    pt = np.round(pt_f).astype(np.int64)
                    if ((pt < 0).any() or (pt[:, 0] >= nx).any()
                            or (pt[:, 1] >= ny).any() or (pt[:, 2] >= nz).any()):
                        return "blocked"
                    if ((self.mat[tuple(pt.T)] != AIR)
                            & ~np.isin(flat(pt), ctx["mine_keys"])).any():
                        return "blocked"      # it would have had to go through
        if dry:
            return "moved"                # asked, not done: every check above
                                          # has run and nothing has been touched
        # THE EDGE GOES WHERE THE IRON GOES, and so does everything else a
        # cell carries (`CARRIED`). `edge` was once missing from this site's
        # own list, so a held blade moved to new cells and left its edge
        # behind: the tool went blunt during a reach, and flesh that moved
        # into a cell kept whatever edge that cell held.
        self._put(tuple(want.T), self._lift(tuple(cur.T)))   # all it carries
        # THE DEED GOES WHERE THE LIMB WENT, as it does for a turn. Only the
        # limb's own rows: what the hand holds rode along, and it is not his
        # even when it is flesh (a hand on another man's arm).
        n_flesh = len(want) - (len(ride) if moved_hold is not None else 0)
        self._move_own(p, cur[:n_flesh], want[:n_flesh])
        # THE BODY'S CORNER CAN MOVE WHEN ONLY PART OF THE BODY DOES. Every
        # segment is kept as an offset from the min corner of the whole body,
        # and that corner is read fresh each time — so a lean, which carries a
        # torso and both arms forward and leaves the legs, can change which
        # voxel is the corner, and then EVERY offset on the body is out by one
        # and the arms are no longer where the body says they are. An arm
        # reaching never showed it: an arm goes forward, and the corner is
        # behind. Rebase, and nothing has to know.
        stay = np.array([c for c in own if tuple(c) not in mine], np.int64)
        allf = np.concatenate([stay.reshape(-1, 3), want[:n_flesh]])
        base = allf.min(axis=0)
        if not np.array_equal(base, origin):
            shift = origin - base
            for b in segs:
                segs[b] = np.asarray(segs[b]) + shift
            for b in rest:
                rest[b] = np.asarray(rest[b]) + shift
            for b in (joints or {}):
                joints[b] = np.asarray(joints[b]) + shift
        k = 0
        for b in moving:                       # each bone keeps its own offsets
            n = len(rest[b])
            segs[b] = want[k:k + n] - base
            k += n
        p.setdefault("drawn", {})[name] = list(th)    # where the flesh IS
        if max((abs(t) for t in th), default=0.0) > 1e-9:
            p.pop("_no_reach", None)          # it got somewhere: the question is
                                              # open again. Coming back to REST
                                              # is not getting somewhere — the
                                              # arm unwinding after a refused
                                              # reach ends in a successful move
                                              # to angle zero, and that cleared
                                              # the very memory it had just made
        if moved_hold is not None and p.get("held"):
            p["held"]["cell"] = moved_hold[0]      # the grip follows the thing
        # AND THE BODY IS NOT WHERE IT WAS. Identity is resolved once a tick
        # and kept; this moved flesh INSIDE that tick, so the kept answer is
        # now a picture of a body that has since bent. It only ever mattered
        # once a pose could move the body's own corner — everything read
        # afterwards was measured from a corner that had gone.
        p["_claim_tick"] = None
        self._torque_solid = None
        self._slack_mat = None
        return "moved"

    def _repose_ctx(self, p, name, toward=None):
        """What `_repose` reads off the body that does not depend on the
        angles it is asked about: the bones that move, where they are now,
        their joints and rest shapes, what the hand holds and how, and the
        sets the checks compare against. A string when the limb cannot be
        posed at all ("blocked"), which is what `_repose` then answers.

        Good for as long as nothing moves: a search hands it to every
        question it asks, and drops it the moment one is answered "moved"."""
        segs, joints = p.get("segs"), p.get("joints")
        bones = [b for b in self._bones(p, name) if b in (segs or {})]
        if not segs or not bones or any(b not in (joints or {}) for b in bones):
            return "blocked"
        # EVERYTHING BEYOND THE JOINT COMES TOO. The limb's own bones are the
        # ones with angles; the moving SET is those plus whatever hangs off
        # them, which for an arm is nothing and for a torso is a head and two
        # arms. A body with no parents declared has a moving set of exactly its
        # own bones, which is why nothing above this changed.
        moving = [b for b in self._subtree(p, bones[0]) if b in segs]
        for b in bones:
            if b not in moving:
                moving.append(b)
        paths = {b: self._path_from(p, bones[0], b) or [b] for b in moving}
        comp, sl = self._person_cells(p)
        if comp is None or not comp.any():
            return "blocked"
        own = self._claim_cells(p, comp, sl)
        origin = own.min(axis=0)
        rest = p.setdefault("rest", {})
        for b in moving:
            if b not in rest:
                rest[b] = np.asarray(segs[b]).copy()
        cur_parts = self._bone_parts(p, moving, own)
        if cur_parts is None or any(len(c) != len(rest[b])
                                    for c, b in zip(cur_parts, moving)):
            return "blocked"                  # a bone has lost voxels; its rest
                                              # shape no longer describes it
        cur = np.concatenate(cur_parts)
        dx, dy = toward if toward else p.get("facing", (1.0, 0.0))
        axis = 0 if abs(dx) >= abs(dy) else 1
        sgn = -1 if (dx if axis == 0 else dy) >= 0 else 1
        bend = set(p.get("bend") or ())    # which joints bend rather than turn
        pivot = {}
        for b in moving:
            pivot[b] = (np.asarray(joints.get(b, joints[bones[0]]))
                        + origin).astype(np.float64)
        # PER MOVING BONE: its path of joints, their pivots, the angles of the
        # joints on it that are not this limb's (they hold where they are),
        # which of them bend, and its rest cells where the body stands now.
        drawn = [(b, paths[b], [pivot[x] for x in paths[b]],
                  [self._bone_angle(p, x) for x in paths[b]],
                  [x in bend for x in paths[b]],
                  (np.asarray(rest[b]) + origin).astype(np.float64))
                 for b in moving]
        # WHAT THE HAND HOLDS TURNS WITH THE HAND. Not an extra rule — it rides
        # rigidly with the bone at the END of the chain, carried from where the
        # old angles put that bone to where the new ones do. Without it an arm
        # could not reach out while holding anything, because the thing it held
        # was standing in the way of its own arm.
        ride = self._held_cells(p)
        mode = None
        if ride is not None and self._in_reach(cur, ride) \
                and len(self._contact(ride, ignore=own)):
            # WHAT THE WORLD IS HOLDING UP DOES NOT COME WITH YOU. A post set
            # in the ground is not carried by the hand on it — the hand is
            # CONSTRAINED by the post. Moved along like a stone, leaning while
            # holding one tried to drag the post, the move was refused because
            # the post could not go there, and the lean stopped at 19.3 degrees
            # where a free spine bends to 45.5. The outcome was right for the
            # wrong reason, and the wrong reason is the sort that stops being
            # right the moment anything changes.
            #
            # So the post stays and the ARM has to reach (`_repose`).
            mode = "world"
        elif ride is not None and self._in_reach(cur, ride):
            mode = "hand"                     # it hangs from the fist
        cur_all = np.concatenate([cur, ride]) if mode == "hand" else cur
        nx, ny, nz = self.shape
        flat = lambda a: (a[:, 0] * ny + a[:, 1]) * nz + a[:, 2]
        mine = {tuple(c) for c in cur_all.tolist()}
        return {"segs": segs, "joints": joints, "rest": rest, "bones": bones,
                "moving": moving, "own": own, "origin": origin,
                "cur_parts": cur_parts, "cur_all": cur_all,
                "cur_keys": flat(cur), "mine": mine,
                "mine_keys": flat(cur_all),
                # the body the limb hangs from (`_still_joined`)
                "still": {tuple(c) for c in own.tolist()} - mine,
                "ride": ride, "ride_mode": mode,
                "axis": axis, "sgn": sgn, "drawn": drawn,
                "was_th": self._angles((p.get("drawn") or {}).get(name, 0.0),
                                       2 * len(bones)),
                "memo": {}}

    def _repose_place(self, ctx, ang, shear=False):
        """Every moving bone, at these angles: its own joint first, then
        every joint between it and the limb being turned.

        TWO WAYS TO DRAW THE SAME TURN, and the difference matters. The
        rotation puts the bone exactly where the angle says, and sometimes
        rounds two of its voxels into one cell, which is not a bone. The
        shears cannot do that — they are bijections — but they round three
        times over and put the bone about a voxel from where it aimed.

        So: the rotation first, because a reach that is one voxel off is a
        reach that misses. The shears are what a bone falls back on when
        the rotation has no drawable answer at all, which used to be the
        end of the matter and is now merely the harder path.

        A bone whose own joints are at the same angles as in an earlier
        question of the same search is where it was then (`memo`): an upper
        arm does not move when only the elbow is being tried."""
        held = {b: (float(ang[2 * k]), float(ang[2 * k + 1]))
                for k, b in enumerate(ctx["bones"])}
        sgn, axis, memo = ctx["sgn"], ctx["axis"], ctx["memo"]
        out = []
        for b, path, piv, still, bd, cells in ctx["drawn"]:
            pr = [held.get(x, a) for x, a in zip(path, still)]
            # the sign of a zero too: -0.0 == 0.0, and the two are kept apart
            key = (b, shear, tuple(pr),
                   tuple(math.copysign(1.0, v) for pair in pr for v in pair))
            got = memo.get(key)
            if got is None:
                got = memo[key] = (
                    self._turn_cells(piv, pr, len(path) - 1, sgn, axis, bd,
                                     cells)
                    if shear else
                    self._apply3(self._chain3(piv, pr, len(path) - 1, sgn,
                                              axis, bd), cells))
            out.append(got)
        return out

    def _hanging_cells(self, p, name, own=None):
        """EVERYTHING BEYOND THIS JOINT — bones and burden alike.

        What a shoulder holds is an arm AND whatever is in the hand, and for a
        long time only the first half was true. `_hold_torque` said so in its
        own docstring and then counted body segments only, so holding an anvil
        at arm's length cost a man exactly what holding nothing cost him. The
        lever was right and the mass on the end of it was missing.

        One answer, so the muscle's effort and the joint's inertia are read off
        the same set of voxels and cannot drift apart."""
        bones = self._bones(p, name)
        bones = [b for b in self._subtree(p, bones[0])
                 if b in (p.get("segs") or {})] if bones else []
        parts = self._bone_parts(p, bones, own) if bones else None
        cells = [c for c in (parts or []) if len(c)]
        # AND WHAT THE HAND HAS, if this is the arm that has it. An arm holding
        # something is a heavier, slower arm — which is why a loaded reach is
        # not the same act as an empty one.
        held = p.get("held") or {}
        if held.get("arm") == name:
            ride = self._held_cells(p)
            if ride is not None and len(ride):
                cells.append(np.asarray(ride))
        if not cells:
            return None
        return np.concatenate(cells)

    def _hold_torque(self, p, name, theta):
        """What the muscle must find to hold this limb at that angle: the limb's
        own weight times how far out its middle hangs from the joint. Zero
        hanging straight down, most of it held straight out — which is why an
        arm can be kept at your side all day and not at arm's length."""
        # EVERYTHING BEYOND THE JOINT hangs off it, not just the bones with
        # angles: what the hips have to hold up is a torso and a head and two
        # arms, and what a shoulder holds is an arm and whatever is in the hand.
        limb = self._hanging_cells(p, name)
        piv = self._pivot_of(p, name)
        if limb is None or piv is None:
            return 0.0, 0.0
        comp, sl = self._person_cells(p)
        if comp is None or not comp.any():
            return 0.0, 0.0                   # in the air since the tick began:
                                              # no body here to hold anything up
        own = self._claim_cells(p, comp, sl)
        # the joint nearest the body, and the WHOLE limb's weight hanging off
        # it — a bent elbow is cheaper to hold out precisely because it brings
        # the mass back in, and that falls out of the lever with no new rule
        jx, jy, _jz = (piv + own.min(axis=0)).astype(np.int64)
        kg = self.smass[tuple(limb.T)] / 1000.0
        m = float(kg.sum())
        if m <= 0.0:
            return 0.0, 0.0
        cx = float((limb[:, 0] * kg).sum()) / m
        cy = float((limb[:, 1] * kg).sum()) / m
        lever = max(abs(cx - jx), abs(cy - jy)) * 0.1 * self.scale
        return m * GRAVITY * lever, m

    def _swing_of(self, p, limb, own=None):
        """What it takes to turn this limb about its own joint, MEASURED.

        Three numbers, all read off the voxels that are actually there: the
        limb's mass, how far its centre of mass sits from the joint, and its
        moment of inertia about that joint. Nothing here is typed. A thick arm
        is harder to start and harder to stop than a thin one because it has
        more cells further out, and a man holding an anvil has an arm with an
        anvil's inertia — the same loop `_body_span` and `_legs_of` close, at
        the joint.

        Returns (I, M, d) in kg m², kg and m, or None if there is no such limb
        or no joint under it."""
        joints = p.get("joints") or {}
        bones = [b for b in self._bones(p, limb) if b in (p.get("segs") or {})]
        if not bones or bones[0] not in joints:
            return None
        if own is None:
            comp, sl = self._person_cells(p)
            if comp is None or not comp.any():
                return None
            own = self._claim_cells(p, comp, sl)
        origin = own.min(axis=0)
        piv = np.asarray(joints[bones[0]], np.float64) + origin
        cells = self._hanging_cells(p, limb, own)
        if cells is None or not len(cells):
            return None
        kg = self.smass[tuple(np.asarray(cells).T)] / 1000.0
        if not kg.sum():
            return None
        vox_m = 0.1 * self.scale
        r = (np.asarray(cells, np.float64) - piv) * vox_m
        M = float(kg.sum())
        com = (kg[:, None] * r).sum(axis=0) / M
        I = float((kg * (r * r).sum(axis=1)).sum())
        return (max(I, 1e-9), M, float(np.linalg.norm(com)))

    def _law_joints(self):
        """A JOINT WITH NOTHING HOLDING IT IS TURNED BY GRAVITY.

        This is the other half of `_law_pose`, and the half that makes a
        shoulder a joint rather than a dial. `_law_pose` says where a muscle
        puts a limb; nothing said what a limb does when no muscle is putting it
        anywhere. The answer was "stays exactly where it was", so a dead man
        held his arm out at shoulder height for ever, and being alive was the
        only thing keeping it up — which is a flag doing a force's job.

        Now the angle has a SPEED of its own. Gravity's moment about the joint
        is `M g d sin(theta)`, the limb's own inertia is what resists it, and
        the arm accelerates, swings past the bottom, comes back and settles.
        Nobody writes how long that takes: it falls out of the cells.

        The angle is the TRUTH and the flesh is a drawing of it. A 5 cm lattice
        cannot draw every angle — that is measured and it is not going away —
        so the joint keeps turning through the ones it cannot draw and the
        voxels catch up at the next one they can. That is what makes a body
        able to BE somewhere the grid cannot put it, which has been behind most
        of the trouble.

        A joint is only free if the limb under it is hanging. A leg with its
        foot on the floor is carrying the body and is not a pendulum, and that
        is a question about CONTACT, not about which limb it is."""
        damp = float(BODY["joint_damp"])
        for p in self.persons:
            if not p.get("segs") or not p.get("joints"):
                continue
            spin = p.setdefault("spin", {})
            pose = p.setdefault("pose", {})
            # NOTHING IS MOVING AND NOTHING IS OUT OF PLACE — the overwhelming
            # case, and it must cost nothing. A body standing with its arms
            # down has no joint to integrate.
            if not any(np.abs(np.atleast_1d(v)).max() > 1e-9
                       for v in list(pose.values()) + list(spin.values())):
                continue
            comp, sl = self._person_cells(p)
            if comp is None or not comp.any():
                continue
            own = self._claim_cells(p, comp, sl)
            for limb in self._limbs(p):
                if limb not in (p.get("chain") or {}):
                    continue
                n = 2 * len(self._bones(p, limb))
                at = np.array(self._angles(pose.get(limb, 0.0), n))
                w = float(spin.get(limb, 0.0))
                # A MUSCLE HAS IT. Not "he is still asking for it" — wanting
                # is not a force, and the first version of this law made that
                # mistake: a refused reach is POPPED as a note about what this
                # body will ask for next, and reading that pop as a release
                # dropped a live man's arm the instant he gave up on a wall.
                #
                # What holds a limb up is muscle tone, and muscle tone is
                # something a living waking body spends energy on every tick —
                # `_law_life` is already paying for it. So this is not a flag
                # standing in for a force; losing consciousness IS losing tone,
                # which is why an unconscious man's arm falls and a thinking
                # man's does not.
                if p["alive"] and p["awake"]:
                    spin[limb] = 0.0
                    continue
                if abs(at[0]) < 1e-9 and abs(w) < 1e-9:
                    continue                      # hanging, and staying hanging
                # STANDING ON IT IS NOT HANGING FROM IT. A limb touching
                # anything but its own body is carrying load somewhere.
                cells = self._limb_cells(p, limb, own)
                if cells is None or not len(cells):
                    continue
                if len(self._contact(np.asarray(cells), ignore=own)):
                    spin[limb] = 0.0
                    continue
                got = self._swing_of(p, limb, own)
                if got is None:
                    continue
                I, M, d = got
                if d < 1e-6:
                    continue
                alpha = -(M * 9.81 * d / I) * float(np.sin(at[0]))
                w += alpha * TICK_S
                # WHAT THE JOINT TAKES OUT OF THE SWING IS HEAT, not nothing.
                slow = w * max(0.0, 1.0 - damp * TICK_S)
                self.shed += 0.5 * I * (w * w - slow * slow)
                w = slow
                nxt = at.copy()
                nxt[0] = float(at[0] + w * TICK_S)
                if abs(nxt[0]) < 1e-3 and abs(w) < 1e-2:
                    nxt[0], w = 0.0, 0.0           # come to rest at the bottom
                spin[limb] = w
                pose[limb] = list(nxt)
                # The flesh follows if it CAN. If it cannot, the joint is still
                # at this angle and the voxels wait for one they can be drawn
                # at — the angle outrunning the flesh, on purpose.
                if self._repose(p, limb, list(nxt)) != "moved" and w:
                    pass

    def _law_pose(self):
        """Limbs go where they are asked, at the speed a muscle can manage, and
        stay there while the muscle can hold them.

        A shoulder is a joint with a motor and a limit, which is a solved thing
        (box3d, MIT) — what is taken here is the shape of it rather than the
        code: an angle, a speed cap, and a torque the motor either has or does
        not. The cap is the same `arm_wmax` Hill's relation already uses for a
        swing, so an arm reaches out at a shoulder's pace and not instantly.

        And an arm held out is HELD OUT, at a cost. If the weight hanging off
        the joint asks more of the muscle than it has, the limb sinks back until
        the lever is short enough — so a limb can be kept at your side for ever
        and at arm's length only while something is paying for it. That is what
        makes a reach a physical act and not a pose in a picture."""
        for p in self.persons:
            want = p.get("reach")
            pose = p.get("pose")
            if not want and not pose:
                continue
            if not p["alive"] or p["safe"]:
                p["reach"] = {}
                want = None
            pose = p.setdefault("pose", {})
            for name in sorted(set(list(pose) + list(want or {}))):
                # TWO ANGLES A BONE: swing and spread, flat. Sorted above
                # because the sim carries no die and a set of names has no
                # order.
                n = 2 * len(self._bones(p, name))
                at = np.array(self._angles(pose.get(name, 0.0), n))
                goal = np.array(self._angles((want or {}).get(name, 0.0), n))
                # AS FAR AS THIS JOINT HAS EVER BEEN ABLE TO GO, this way
                # round. Asking for more than the flesh can be drawn at is how
                # the ladder above got climbed in the first place.
                cap = (p.get("_span") or {}).get(
                    "bend:%s:%d" % (name, self._face_axis(p)))
                if cap is not None and np.abs(goal).max() > cap:
                    goal = np.clip(goal, -cap, cap)
                can = (p.get("torque_Nm") or {}).get(name)
                if can is None:
                    can = self._torque_of(
                        p, name, p.get("arm_Nm", BODY["arm_Nm"]))
                if np.abs(goal).max() > 0.0:
                    need, _m = self._hold_torque(p, name, goal)
                    if need > can:
                        goal = np.zeros(n)    # too heavy to hold out there
                d = goal - at
                if np.abs(d).max() < 1e-9:
                    # ARRIVED IN ANGLE, BUT NOT IN FLESH. The command has been
                    # given in full and the body is not there — so THIS is how
                    # far this joint goes on this lattice, and the body settles
                    # to where it actually is rather than holding a number it
                    # can never be at.
                    #
                    # The angle outrunning the flesh is deliberate and right:
                    # it is how a limb crosses the angles that cannot be drawn
                    # and catches up at the next one that can. What was missing
                    # is the end of that story. A lean has a long undrawable
                    # tail, so a body's angle read 45.8 degrees while it was
                    # bent 19.3, for ever, and anything that believed the angle
                    # was wrong about the body.
                    dr = np.array(self._angles(          # n is already the
                        (p.get("drawn") or {}).get(name, 0.0), n))   # pair count
                    if np.abs(dr - at).max() > 1e-9:
                        pose[name] = list(dr)
                        if want is not None and name in want:
                            want[name] = list(dr)
                        # AND IT REMEMBERS. Settling to where the flesh got to
                        # was only half the story: the next tick asked for the
                        # same angle again, climbed the same undrawable ladder,
                        # and snapped back — for ever. A man told to raise his
                        # arm while facing along x ran that loop 0.375, 0.75,
                        # 0.9, 0.0 without his arm EVER moving a voxel.
                        #
                        # So this is a fact learned about a joint, filed where
                        # `_bend_max` keeps them, and keyed by which way he was
                        # facing when he learned it — because that is what
                        # decides which angles can be drawn at all.
                        p.setdefault("_span", {})[
                            "bend:%s:%d" % (name, self._face_axis(p))] = \
                            float(np.abs(dr).max())
                    continue
                # HOW FAST THAT JOINT GOES. A property of the joint, like the
                # torque it can hold — a trunk is not a shoulder. But the cap
                # is a CEILING and not a speed: what a limb actually manages is
                # the torque it has over the inertia it must shift, and then
                # only as fast as it can still STOP in the angle it has left.
                #
                # That is the whole difference between reaching with an empty
                # hand and reaching with an anvil. Nobody writes down that a
                # loaded arm is slower; the anvil is in `_hanging_cells`, so it
                # is in the inertia, so it is in the time.
                w_cap = float((p.get("wmax") or {}).get(
                    name, BODY["arm_wmax"]))
                got = self._swing_of(p, name)
                if got is not None:
                    alpha = float(can) / got[0]
                    w_cap = min(w_cap,
                                float(np.sqrt(2.0 * alpha
                                              * max(np.abs(d).max(), 1e-9))))
                step = w_cap * TICK_S
                nxt = at + np.clip(d, -step, step)
                ctx = self._repose_ctx(p, name)
                how = self._repose(p, name, nxt, ctx=ctx)
                # UNDRAWABLE IS ALSO A REASON TO BEND. This tried going round
                # only when the WORLD was in the way, and left the other case —
                # the lattice cannot draw the arm straight at this angle — to
                # the "angle outruns the flesh" path, which waits for a later
                # angle that can be drawn. With a wall four cells off and a
                # two-voxel arm there is no such angle: measured, 183 of 210
                # poses were undrawable, 18 blocked, and 9 both drawable and
                # clear — and the arm never tried one of the 9 because it was
                # never told the refusal had come from the grid.
                #
                # Which way the bones get there is motor competence either way.
                if how in ("blocked", "undrawable"):
                    bent = self._reach_around(p, name, nxt, at, ctx=ctx)
                    if bent is not None:
                        pose[name] = list(bent)
                        continue
                    # the world said no, every way round. An arm that meets a
                    # post stops AT the post and does not go on wanting to be
                    # past it — and REMEMBERS, against the spot it was standing
                    # on and the way it was facing, so it does not ask the same
                    # question of the same wall every few ticks for ever.
                    (p.get("reach") or {}).pop(name, None)
                    p["_no_reach"] = (p["anchor"],
                                      tuple(p.get("facing", (1.0, 0.0))))
                    if not np.abs(at).max():
                        pose.pop(name, None)
                    continue
                if how == "moved":
                    # AND A POSE YOU CANNOT KEEP IS NOT A POSE YOU ADOPT. The
                    # same gate the muscle already has, with the other reason a
                    # body stops short: not "my back will not hold that" but
                    # "that puts me over my own toes". So a body leans as far
                    # as it can stand and no further, and going over is left to
                    # the things that really do take you over — a load, a shove,
                    # a floor that leaves. Balance measured on the FLESH; what
                    # a body carries is `_pulled_over`'s question, and the two
                    # are complementary rather than duplicated.
                    comp, sl = self._person_cells(p)
                    if comp is not None and comp.any():
                        c = self._claim_cells(p, comp, sl)
                        if self._underfoot(c) \
                                and self._overbalanced(p, c) is not None:
                            self._repose(p, name, at)      # think better of it
                            # AND HOLD THERE. The intention was "as far as I
                            # can", and this is how far — so it resolves to
                            # HERE rather than being abandoned. Dropped
                            # instead, the goal fell back to nothing and the
                            # body straightened up again the moment it reached
                            # the furthest it could stand, which is a strange
                            # thing to watch and a stranger thing to mean.
                            if want is not None and name in want:
                                want[name] = list(at)
                            continue
                pose[name] = list(nxt)        # the angle moves either way; the
                                              # flesh catches up at the next
                                              # angle the lattice can draw

    def _reach_key(self, p, name, th, at):
        """Everything `_reach_around` reads, as one key: the limb, what it was asked,
        the pose and what is held, the person's own cells, and the cells around
        the person out to 24 voxels (an arm is under 10 long). None if the person
        has no body to ask about."""
        comp, sl = self._person_cells(p)
        if comp is None or not comp.any():
            return None
        own = self._claim_cells(p, comp, sl)
        lo = np.maximum(own.min(0) - 24, 0)
        hi = np.minimum(own.max(0) + 25, self.shape)
        box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        h = hashlib.blake2b(digest_size=16)
        for a in (self.mat[box], self.eid[box], own):
            h.update(np.ascontiguousarray(a).tobytes())
        return (name, tuple(map(float, th)), None if at is None else tuple(map(float, at)),
                tuple(p["anchor"]), tuple(p.get("facing", ())),
                repr(p.get("pose")), repr(p.get("held")),
                repr(p.get("drawn")), repr(p.get("bend")), repr(p.get("joints")),  # the sweep's start
                h.digest())

    def _reach_around(self, p, name, th, at=None, ctx=None):
        """`_reach_search`, remembering a FAILURE. A refused arm asks again every
        tick with the same pose, the same hands and the same room, and the search
        (about 220 `_repose` tries, 60 ms with a heavy arm) is deterministic in
        those, so the same answer, None, is given without it. EXACT: only a failure
        is kept, one per limb, and any change in the key (the room, the pose, what
        is held) searches again. Found by wave 13b: friction made one hauling
        person refuse an arm for 60 ticks, 5.6 -> 36 ms per tick."""
        key = self._reach_key(p, name, th, at)
        if key is None:
            return self._reach_search(p, name, th, at, ctx)
        memo = p.setdefault("_reach_fail", {})
        if memo.get(name) == key:
            return None
        out = self._reach_search(p, name, th, at, ctx)
        if out is None:
            memo[name] = key
        else:
            memo.pop(name, None)
        return out

    def _reach_search(self, p, name, th, at=None, ctx=None):
        """An arm that meets something does not simply stop. It BENDS.

        With one degree of freedom there is exactly one path to a place, so a
        man standing beside the person he is holding could not put his arm out
        at all — that person was standing in the one path. Real arms go round,
        and going round is what a second joint IS.

        The CHOICE was "reach out"; which way the bones get there is motor
        competence, the same kind of thing as knowing the way round a table. So
        this belongs here and not on any menu. It is a sampled search over the
        elbow, coarse on purpose: at 5 cm a one-voxel forearm has only a
        handful of drawable angles anyway, and `_repose` is what says which —
        the first bend that really fits is the one the arm takes."""
        bones = self._bones(p, name)
        if len(bones) < 2:
            return None                       # one bone has nowhere to bend
        # LEAST BEND FIRST — a body that can straighten its arm does. And the
        # shoulder is allowed to stay where it IS as well as go where it was
        # asked: getting the hand somewhere matters more than the angle the
        # shoulder happens to be at, which is the difference between a joint
        # and a dial. Fine steps because drawability on a 5 cm lattice is
        # SPIKY — a coarse ladder walks straight past the angles that fit.
        heads = [float(th[0])]
        if at is not None and abs(float(at[0]) - float(th[0])) > 1e-9:
            heads.append(float(at[0]))
        # AND A SHORTER REACH, if the full one has nowhere to go. The shoulder
        # was the one joint this search never touched: it took the angle it was
        # asked for and hunted the elbow and the spread underneath it. So an
        # arm that could not get out AT THAT SHOULDER ANGLE gave up, while a
        # pose with the same elbow and a shallower shoulder sat there unfound —
        # measured with a wall four cells off, 9 poses were drawable and
        # unblocked and the search returned None.
        #
        # A man who cannot reach at full stretch does not stop reaching; he
        # reaches less far. Least reduction first, so he still goes as far as
        # he can.
        heads += [float(th[0]) * f for f in (0.75, 0.5, 0.25)]
        # AND THERE ARE TWO WAYS ROUND NOW. An elbow bends the arm in the
        # plane it is swinging in; a SPREAD takes the whole arm out of that
        # plane altogether, which is the way round a thing directly in front
        # of you — and is what a shoulder does when you reach past someone.
        # Least movement first, and the elbow before the shoulder, because
        # bending an arm is cheaper than swinging the whole of it sideways.
        rungs = [s * b for b in np.arange(0.15, 1.8, 0.15) for s in (1.0, -1.0)]
        n2 = 2 * len(bones)
        # ONE BODY, MANY QUESTIONS: nothing moves until an answer is "moved",
        # so what `_repose` reads off the body is read once for all of them.
        if ctx is None:
            ctx = self._repose_ctx(p, name)
        for head in heads:
            for bend in rungs:
                alt = ([head, 0.0, float(bend), 0.0] + [0.0] * n2)[:n2]
                if self._repose(p, name, alt, ctx=ctx) == "moved":
                    return alt
            for out in rungs:
                alt = ([head, float(out)] + [0.0] * n2)[:n2]
                if self._repose(p, name, alt, ctx=ctx) == "moved":
                    return alt
        return None

    def _bunch(self, want, mine, cap=4):
        """Two bones want one cell: the flesh between them has to go somewhere.

        A folded knee presses calf against thigh, and the flesh in the crease
        does not vanish — it BULGES. On a lattice that shows up as two voxels
        wanting the same cell, and both of the easy answers are lies. Refuse
        the pose and the body cannot bend its knee past about seventeen
        degrees, which is false. Drop the surplus voxel and flesh ceases to
        exist when a man squats, which is worse, and it breaks the mass books
        besides.

        So the surplus voxel is put in the nearest free cell instead. The leg
        thickens at the knee, which is where a real leg thickens, and not one
        gram goes missing. Measured on this body: a knee at 1.1 rad crowds
        five cells out of forty-two, and they land in the crease.

        None if there is nowhere within `cap` steps for the surplus to go —
        flesh pressed into a gap smaller than itself, which IS a refusal."""
        out = np.array(want, np.int64, copy=True)
        used, dup = set(), []
        for i, c in enumerate(map(tuple, out)):
            if c in used:
                dup.append(i)
            else:
                used.add(c)
        if not dup:
            return out
        nx, ny, nz = self.shape
        for i in dup:
            c = tuple(out[i])
            found, frontier, seen = None, [c], {c}
            for _ in range(cap):
                nxt = []
                for q in frontier:
                    for d in _SIX:
                        n = (q[0] + d[0], q[1] + d[1], q[2] + d[2])
                        if n in seen:
                            continue
                        seen.add(n)
                        if not (0 <= n[0] < nx and 0 <= n[1] < ny
                                and 0 <= n[2] < nz):
                            continue
                        if n in used:
                            nxt.append(n)      # taken, but flesh reaches past
                            continue
                        if n in mine or int(self.mat[n]) == AIR:
                            found = n
                            break
                        nxt.append(n)
                    if found:
                        break
                if found:
                    break
                frontier = nxt
            if found is None:
                return None
            used.add(found)
            out[i] = found
        return out

    def _crouch(self, p, theta, dry=False):
        """Sink through a crouch: the legs fold and the whole body comes down.

        A CROUCH IS NOT A LIMB POSE, and the lattice says so before anything
        else does. Bending a knee on its own swings the hip away from a torso
        that has not moved, so the leg comes off the body — measured, 1 of 289
        knee angles was drawable, and the one was standing up straight. You
        cannot bend a knee without the hips moving.

        Nor can you DRAW the fold at this size. A thigh tipped one way and a
        shin tipped the other want the same cells at the knee, and at five
        centimetres a knee crease is one cell wide — measured, the crease
        crowded five cells of forty-two at a right angle, and the same drift
        pushed the foot through the floor. The bend is finer than the grid.

        What IS visible at five centimetres is a leg going SHORTER and THICKER,
        and that is the whole observable of a crouch from outside. So the legs
        compress: every cell keeps its foot on the ground and comes down in
        proportion to how far up the leg it sits, the surplus bulges out where
        a folded leg bulges, and everything above the hips travels down by
        exactly what the legs lost. The hip and the waist move together by
        construction, so the body cannot come apart at the waist.

        HOW FAR is not a number typed here — `_crouch_of` reads it off this
        body's own legs, so a tall man sinks further than a child and nobody
        writes it down twice. `theta` asks for a knee angle and gets as much of
        it as the legs have to give.

        SYMMETRIC BY DEFAULT. `theta` may be a pair, one leg each, and then the
        body follows the HIGHER hip: standing on one bent leg and one straight
        one is a lunge, and a lunge is a thing a body can do. Two feet on the
        ground is a loop rather than a tree, and the higher hip is what
        resolves it — the straighter leg is the one carrying you."""
        segs, joints = p.get("segs"), p.get("joints")
        legs = [k for k in self._limbs(p) if "leg" in k]
        if not segs or len(legs) < 2:
            return "blocked"
        # SYMMETRIC BY DEFAULT — and this is where that lives. One number
        # spread over one limb means the joint nearest the body and the rest
        # straight, which is what an arm wants; one number spread over TWO
        # LEGS that way is a man bending one knee, which is a lunge, not a
        # crouch. Ask for a crouch with a number and both legs do it; ask with
        # a pair and they do as they are told.
        th = ([float(theta)] * len(legs) if np.isscalar(theta)
              else self._angles(theta, len(legs)))
        comp, sl = self._person_cells(p)
        if comp is None or not comp.any():
            return "blocked"
        own = self._claim_cells(p, comp, sl)
        origin = own.min(axis=0)
        rest = p.setdefault("rest", {})
        deep = self._crouch_of(p) / max(0.1 * self.scale, 1e-9)
        moving, parts, sunk = [], [], []
        for li, leg in enumerate(sorted(legs)):
            bones = [b for b in self._bones(p, leg) if b in segs]
            if not bones:
                return "blocked"
            for b in bones:
                rest.setdefault(b, np.asarray(segs[b]).copy())
            zs = np.concatenate([np.asarray(rest[b]) for b in bones])[:, 2]
            z0 = int(zs.min()) + int(origin[2])
            span = max(int(zs.max() - zs.min()) + 1, 1)
            # AS MUCH OF THE ANGLE AS THE LEGS HAVE. The geometry asks for
            # span*(1 - cos theta); the body says how deep it goes at all.
            d = int(min(round(span * (1.0 - float(np.cos(float(th[li]))))),
                        round(deep), span - 1))
            sunk.append(d)
            for b in bones:
                c = (np.asarray(rest[b]) + origin).astype(np.int64)
                c[:, 2] = z0 + np.rint(
                    (c[:, 2] - z0) * (span - d) / span).astype(np.int64)
                moving.append(b)
                parts.append(c)
        drop = int(min(sunk))      # the body follows the HIGHER hip
        upper = [b for b in segs if b not in moving]
        for b in upper:
            rest.setdefault(b, np.asarray(segs[b]).copy())
            c = (np.asarray(rest[b]) + origin).astype(np.int64)
            c[:, 2] -= drop
            moving.append(b)
            parts.append(c)
        cur_parts = self._bone_parts(p, moving, own)
        if cur_parts is None or any(len(c) != len(rest[b])
                                    for c, b in zip(cur_parts, moving)):
            return "blocked"
        cur = np.concatenate(cur_parts)
        want = np.round(np.concatenate(parts)).astype(np.int64)
        nx, ny, nz = self.shape
        if ((want < 0).any() or (want[:, 0] >= nx).any()
                or (want[:, 1] >= ny).any() or (want[:, 2] >= nz).any()):
            return "blocked"
        mine = {tuple(c) for c in cur}
        flat = (want[:, 0] * ny + want[:, 1]) * nz + want[:, 2]
        if len(np.unique(flat)) != len(flat):
            # THE KNEE CREASE. Not a drawing failure — flesh with nowhere to be
            want = self._bunch(want, mine)
            if want is None:
                return "blocked"
        if not self._still_joined(p, want, own, mine):
            return "undrawable"
        if any(tuple(t) not in mine and int(self.mat[tuple(t)]) != AIR
               for t in want):
            return "blocked"
        if dry:
            return "moved"
        # all of it moves, `edge` too, for the same reason as in `_repose`: a
        # leg that bends into a cell must not pick up an edge left there.
        self._put(tuple(want.T), self._lift(tuple(cur.T)))   # all it carries
        self._move_own(p, cur, want)          # the deed sinks with the body
        base = want.min(axis=0)
        if not np.array_equal(base, origin):
            shift = origin - base
            for b in segs:
                segs[b] = np.asarray(segs[b]) + shift
            for b in rest:
                rest[b] = np.asarray(rest[b]) + shift
            for b in (joints or {}):
                joints[b] = np.asarray(joints[b]) + shift
        k = 0
        for b in moving:
            n = len(rest[b])
            segs[b] = want[k:k + n] - base
            k += n
        p["crouched"] = list(th)
        p["_claim_tick"] = None
        self._torque_solid = None
        self._slack_mat = None
        return "moved"

    def _bend_max(self, p, limb, ceiling):
        """The furthest this body can bend that joint AND STILL BE DRAWN.

        `lean_max` was 0.8 rad — a real spine's stop — and this body cannot be
        drawn past about 0.34, so the typed number named an angle it could
        never be at and `pose` chased it for ever (item 49). The honest stop is
        whichever comes first: anatomy, or the lattice.

        Asked once per body and kept, because it is a fact about a shape and
        the shape does not change. Asked at all only because `_repose` can now
        be asked without being obeyed."""
        key = "bend:%s:%d" % (limb, self._face_axis(p))
        got = (p.get("_span") or {}).get(key)
        if got is not None:
            return got
        lo, hi = 0.0, float(ceiling)
        ctx = self._repose_ctx(p, limb)       # asked, never moved: read once
        for _ in range(7):                    # bisect: seven halvings of 0.8
            mid = 0.5 * (lo + hi)             # rad is finer than the lattice
            if self._repose(p, limb, mid, dry=True, ctx=ctx) == "moved":
                lo = mid
            else:
                hi = mid
        p.setdefault("_span", {})[key] = lo
        return lo
