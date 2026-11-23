"""Body geometry: bones, chains, joints, transforms, spans, fists, reach (reads, never moves)."""
import numpy as np
from .materials import (BODY, CHEST0, FLESH, _SIX)


class SkeletonMixin:

    def _limbs(self, p):
        """The body's named LIMBS — an arm counts ONCE, by the name of the
        whole arm, not once per bone. Everything outside this file asks for
        "right arm"; whether that is one bone or three is the body's own
        business, declared by whoever built it."""
        chain = p.get("chain") or {}
        bones = {b for bs in chain.values() for b in bs}
        out = list(chain)
        out += [k for k in (p.get("segs") or {}) if k not in bones
                and k not in chain]
        return out

    def _bones(self, p, name):
        """The bones of a limb, SHOULDER FIRST. A limb with no chain declared
        is one bone with its own name, which is why nothing had to change when
        arms grew elbows."""
        chain = (p.get("chain") or {}).get(name)
        return list(chain) if chain else [name]

    def _parents(self, p):
        """Which bone each bone hangs from.

        A limb's CHAIN already says most of this — the forearm hangs from the
        upper arm, that is what writing them in that order means — so it is
        read off there rather than declared a second time, which would be a
        second place to be wrong. `p["parent"]` says the rest: what the arms
        and the head hang from, which no chain covers because they are not in
        one another's chains."""
        par = dict(p.get("parent") or {})
        for bs in (p.get("chain") or {}).values():
            for a, b in zip(bs, bs[1:]):
                par.setdefault(b, a)
        return par

    def _kids(self, p):
        """Bone -> the bones hanging off it."""
        kids = {}
        for b, par in self._parents(p).items():
            kids.setdefault(par, []).append(b)
        return kids

    def _subtree(self, p, root):
        """`root` and everything hanging off it, nearest the body first.

        TURNING A JOINT CARRIES EVERYTHING BEYOND IT. That is what a joint is,
        and it is the whole reason a body needs a tree and not a list: leaning
        at the hips takes the head and both arms with it, and leaves the legs
        exactly where they are standing. A body with no parents declared is all
        roots, so this returns the one bone and nothing changes."""
        kids = self._kids(p)
        out, q = [], [root]
        while q:
            b = q.pop(0)
            if b in out:
                continue
            out.append(b)
            q.extend(sorted(kids.get(b, [])))   # sorted: the sim carries no die
        return out

    def _path_from(self, p, root, bone):
        """The bones from `root` out to `bone`, inclusive — the joints whose
        turns compose to say where that bone ends up."""
        par = self._parents(p)
        path, b = [], bone
        while True:
            path.append(b)
            if b == root:
                return path[::-1]
            b = par.get(b)
            if b is None or len(path) > 32:
                return None

    def _pivot_of(self, p, name):
        """The joint a whole limb turns about: for a chain, the one nearest the
        body. What a shoulder torque is measured at."""
        joints = p.get("joints") or {}
        for b in self._bones(p, name):
            if b in joints:
                return np.asarray(joints[b])
        return None

    @staticmethod
    def _axis_xform(piv, theta, ax, sgn, bend=False):
        """One joint's turn about ONE horizontal axis, as a 3x4 affine on
        (x, y, z).

        A JOINT HAS TWO AXES. A shoulder is not a hinge: an arm swings forward
        and back, and it also goes out to the SIDE, and the two are different
        rotations about perpendicular lines through the same point. With one
        axis an arm could never reach sideways, never across its own chest, and
        never overhead-and-out — and which single plane it DID have depended on
        which way the body happened to be facing.

        Written as a 3x4 rather than the 2x3 this replaces, because two
        rotations in different planes do not compose in two dimensions. The
        horizontal coordinate this turn does not touch passes through
        unchanged, so one axis alone is exactly the old transform with a row
        added — which is what makes it safe to swap underneath everything."""
        M = np.zeros((3, 4))
        M[0, 0] = M[1, 1] = M[2, 2] = 1.0
        other = 1 - ax
        M[other, other] = 1.0
        if bend:
            t = -float(np.tan(theta))
            M[ax, ax] = 1.0
            M[ax, 2] = sgn * t
            M[ax, 3] = -sgn * t * piv[2]
            M[2, 2] = 1.0
            return M
        c, sn = float(np.cos(theta)), float(np.sin(theta))
        M[ax, ax] = c
        M[ax, 2] = sgn * sn
        M[ax, 3] = piv[ax] - c * piv[ax] - sgn * sn * piv[2]
        M[2, ax] = -sgn * sn
        M[2, 2] = c
        M[2, 3] = piv[2] + sgn * sn * piv[ax] - c * piv[2]
        return M

    @staticmethod
    def _shear_steps(piv, theta, ax, sgn, bend=False):
        """One joint's turn written as SHEARS instead of as a rotation.

        A rotation on a lattice has to round, and rounding sends two voxels
        into one cell and no voxel into the next — a bone loses mass in one
        place and doubles it in another, which is why a turned limb kept
        coming back undrawable. Measured on this body's own bones: a torso
        could be turned through 4 of 31 angles, a thigh through 9.

        A shear does not have that problem. `x += round(k*z)` slides a whole
        row sideways onto a whole row, so it is a bijection of the lattice
        for ANY k — nothing is lost and nothing is doubled. Three shears
        make a rotation (Paeth's raster-rotation algorithm, 1986), so three
        bijections make a turn that cannot lose a voxel at any angle. The
        same bones then turn through 31 of 31.

        Returned as a list of steps rather than a matrix because the whole
        property lives in the rounding BETWEEN them. Multiply them together
        first and it is an ordinary rotation again, with ordinary holes.

        A step is (dst, src, k, off): dst += round(k * (src - off)). The
        pivot is a fixed point of every step, so it is a fixed point of the
        turn."""
        if bend:
            return [(ax, 2, -sgn * float(np.tan(theta)), float(piv[2]))]
        a = sgn * float(theta)
        k = float(np.tan(a / 2.0))
        s = -float(np.sin(a))
        return [(ax, 2, k, float(piv[2])),
                (2, ax, s, float(piv[ax])),
                (ax, 2, k, float(piv[2]))]

    @staticmethod
    def _apply_shears(steps, cells):
        """Move a set of voxels through a list of shears, rounding between."""
        c = np.rint(np.asarray(cells, np.float64)).astype(np.int64)
        for dst, src, k, off in steps:
            c[:, dst] += np.rint(k * (c[:, src] - off)).astype(np.int64)
        return c

    def _turn_cells(self, pivots, pairs, upto, sgn, axis, bends, cells):
        """WHERE A BONE'S VOXELS GO. The one answer both callers use.

        Shears to decide WHICH cells, because that cannot lose one; the exact
        rotation to decide WHERE, because that cannot drift. Three shears make
        a turn no voxel falls out of, but each shear rounds, and the roundings
        pile up — a fist came out about a voxel from where the arm had aimed
        it, which is nothing on a wall and everything when the thing you are
        reaching past is one voxel away. It read as a man unable to reach at
        all, and the flesh was fine; the arm was simply put down beside where
        it meant to be.

        So the shape comes from the shears and the PLACE comes from the
        rotation: shift the sheared cells bodily until their centre sits where
        the rotation put the centre. A whole-cell shift of the whole set is
        still a bijection, so nothing is lost by fixing it."""
        want = self._apply_shears(
            self._chain_shear(pivots, pairs, upto, sgn, axis, bends), cells)
        if len(want) == 0:
            # A BONE WITH NO VOXELS. A limb swung off the world's edge comes
            # back as nothing, and the landing (bodies.py) writes that bone
            # back empty. An empty set has no centre to match, and there is
            # nothing to place, so the empty set is the answer. The mean of
            # nothing gave NaN, a RuntimeWarning, and a garbage shift added
            # to no cells (fuzzer seed 58).
            return want
        aim = self._apply3(
            self._chain3(pivots, pairs, upto, sgn, axis, bends), cells)
        off = np.rint(aim.mean(axis=0) - want.mean(axis=0)).astype(np.int64)
        return want + off

    def _chain_shear(self, pivots, pairs, upto, sgn, axis, bends=None):
        """Where a bone's REST cells end up, as shears — the bijective twin
        of _chain3, and the one that actually draws the voxels.

        Same order: a bone's own joint turns first and every joint between it
        and the limb carries the result along, spread before swing within a
        joint. _chain3 still answers WHERE a limb reaches, which wants real
        arithmetic; this answers WHICH CELLS, which wants whole ones."""
        steps = []
        for j in range(upto, -1, -1):
            swing, spread = pairs[j]
            bend = bool(bends[j]) if bends else False
            if spread:
                steps += self._shear_steps(pivots[j], spread, 1 - axis,
                                           sgn, bend)
            steps += self._shear_steps(pivots[j], swing, axis, sgn, bend)
        return steps

    @staticmethod
    def _comp3(a, b):
        """Do b, then a — 3x4 affines."""
        m = np.empty((4, 4))                  # b with (0, 0, 0, 1) under it:
        m[:3] = b                             # the same matrix `np.vstack`
        m[3] = (0.0, 0.0, 0.0, 1.0)           # made, without its overhead
        return a @ m

    @staticmethod
    def _apply3(M, cells):
        """Move a set of voxels by a 3x4 affine."""
        c = np.asarray(cells, np.float64)
        return c @ M[:, :3].T + M[:, 3]

    def _chain3(self, pivots, pairs, upto, sgn, axis, bends=None):
        """Where a bone's REST cells end up, with TWO axes at every joint.

        Its own joint turns first, then every joint between it and the limb
        being moved. Within one joint the SPREAD goes first and the SWING on
        top of it — an arm is taken out to the side and then swung, which is
        the order a shoulder does it in and the order that keeps "swing"
        meaning what it meant when there was only one axis.

        Composed in the REST frame, which is what saves tracking where a
        shoulder has dragged an elbow to."""
        M = np.zeros((3, 4))
        M[0, 0] = M[1, 1] = M[2, 2] = 1.0
        for j in range(upto, -1, -1):
            swing, spread = pairs[j]
            bend = bool(bends[j]) if bends else False
            R = self._axis_xform(pivots[j], swing, axis, sgn, bend)
            if spread:
                R = self._comp3(R, self._axis_xform(pivots[j], spread,
                                                    1 - axis, sgn, bend))
            M = self._comp3(R, M)
        return M

    def _limb_parts(self, p, name, own=None):
        """One array of live cells PER BONE, shoulder first.

        Anything that has to know which bone a voxel belongs to asks here — a
        swung arm that comes to rest has to be able to say where its elbow went
        and not merely where its arm did."""
        segs = p.get("segs")
        bones = [b for b in self._bones(p, name) if b in (segs or {})]
        if not segs or not bones:
            return None
        return self._bone_parts(p, bones, own)

    def _bone_parts(self, p, bones, own=None):
        """The live cells of each NAMED bone, in the order asked for."""
        segs = p.get("segs") or {}
        if own is None:
            comp, sl = self._person_cells(p)
            if comp is None or not comp.any():
                return None
            own = self._claim_cells(p, comp, sl)
        base = np.asarray(own).min(axis=0)
        nx, ny, nz = self.shape
        out = []
        for b in bones:
            if b not in segs:
                return None
            c = (np.asarray(segs[b]) + base).astype(np.int64)
            ok = ((c[:, 0] >= 0) & (c[:, 0] < nx) & (c[:, 1] >= 0)
                  & (c[:, 1] < ny) & (c[:, 2] >= 0) & (c[:, 2] < nz))
            c = c[ok]
            if len(c):
                c = c[self.mat[tuple(c.T)] == FLESH]
            out.append(c)
        return out

    def _bone_angle(self, p, bone):
        """One bone's OWN joint angles — (swing, spread) — whichever limb owns
        it. A bone nobody poses, a head, is at rest and rides on what is under
        it."""
        for limb, bs in (p.get("chain") or {}).items():
            if bone in bs:
                th = self._angles((p.get("pose") or {}).get(limb, 0.0),
                                  2 * len(bs))
                k = bs.index(bone)
                return (th[2 * k], th[2 * k + 1])
        a = np.atleast_1d((p.get("pose") or {}).get(bone, 0.0))
        return (float(a[0]), float(a[1]) if len(a) > 1 else 0.0)

    def _limb_cells(self, p, name, own=None):
        """Where one named part of this body actually is, right now.

        Segments are kept as offsets from the body's own corner, so they ride
        along when it walks, is shoved, or is carried out. A limb that has been
        swung and come to rest somewhere new updates its own offsets on landing.
        A limb made of several bones answers with all of them.
        Returns None when there is no such limb, or nothing left of it."""
        parts = self._limb_parts(p, name, own)
        if not parts:
            return None
        limb = np.concatenate(parts)
        return limb if len(limb) else None

    def _still_joined(self, p, want, own, mine, rest=None):
        """Is the limb, DRAWN AT THIS ANGLE, still one piece of one body?

        The other half of "undrawable", and the half that was missing. A line
        of voxels turned to anything but a right angle rounds to a STAIRCASE,
        and a staircase is not joined — its steps touch at corners only. So the
        check that two voxels must not round into one has a twin: they must not
        round APART either. Measured before this: an arm bent 1.2 rad came out
        in two pieces, still all its own mass, no longer all one arm.

        The honest answer is the same one the lattice has always given — the
        angle is not drawable, the flesh holds its last good pose, and the arm
        catches up at the next angle that IS. It is not a smoothing hack; at
        5 cm a one-voxel limb genuinely has only a handful of poses."""
        if rest is None:                      # the body it hangs from (a
            rest = {tuple(c) for c in          # caller asking many times
                    np.asarray(own).tolist()} - mine   # passes it in)
        S = {tuple(c) for c in np.asarray(want).tolist()}
        seen = {c for c in S
                if any((c[0] + dx, c[1] + dy, c[2] + dz) in rest
                       for dx, dy, dz in _SIX)}
        if not rest:
            # NOTHING STAYED BEHIND. A crouch moves every bone there is, so
            # there is no still half for the moved half to hang from, and
            # asking to be attached to it said a sinking body had come apart.
            # When the whole body goes, joined means joined TO ITSELF.
            seen = {next(iter(S))} if S else set()
        elif not seen:
            return False                      # it is not attached at all
        stack = list(seen)
        while stack:
            c = stack.pop()
            for dx, dy, dz in _SIX:
                n = (c[0] + dx, c[1] + dy, c[2] + dz)
                if n in S and n not in seen:
                    seen.add(n)
                    stack.append(n)
        return len(seen) == len(S)

    @staticmethod
    def _angles(theta, n):
        """One angle per bone. A single number turns the joint NEAREST THE BODY
        and leaves the rest straight — which is exactly what every caller meant
        back when an arm was one bone, so nothing had to be rewritten when it
        grew an elbow."""
        th = [float(theta)] if np.isscalar(theta) else [float(t) for t in theta]
        return (th + [0.0] * n)[:n]

    def _bone_at(self, p, cell):
        """Which named bone of this body a cell belongs to, or the nearest."""
        segs = p.get("segs") or {}
        if not segs:
            return None
        names = sorted(segs)
        parts = self._bone_parts(p, names)
        if parts is None:
            return None
        c = np.asarray(cell, np.int64)
        best, bd = None, None
        for name, cs in zip(names, parts):
            if not len(cs):
                continue
            d = int(np.abs(cs - c).sum(axis=1).min())
            if bd is None or d < bd:
                best, bd = name, d
        return best

    def _thickness(self, p, bone):
        """How thick this bone's flesh is across its narrow way, in metres."""
        parts = self._bone_parts(p, [bone])
        if not parts or not len(parts[0]):
            return 0.1 * self.scale
        c = parts[0]
        span = c.max(axis=0) - c.min(axis=0) + 1
        return float(np.sort(span)[0]) * 0.1 * self.scale

    def _bone_frame(self, p, bone):
        """One bone's own frame, in lattice cells (wave 15, S2 A): where a cut
        on it is, ALONG it and AROUND it, in a way that stays true when the
        limb is swung.

        Returns a dict: o (centroid), axis (unit, proximal to distal: away
        from the trunk), f (the bone's anterior, the body's rest facing carried
        through the swing that took the rest axis to the present one), e2
        (the medial direction, 90 degrees around from the anterior), t0 and t1
        (the bone's extent along the axis, cells), trunk (the trunk's centroid)
        and z_heart (the heart's height in cells: three quarters up the
        trunk). None when the body has no such bone.

        Measured from the bone's own cells (the first principal axis); the
        swing is the minimal rotation from the rest cells' axis (`p["rest"]`)
        to the present one, which is the whole of what a swing and a spread do
        to a limb that does not twist. A bone of one cell has the vertical for
        its axis."""
        if not p.get("segs") or bone not in p["segs"]:
            return None
        parts = self._bone_parts(p, [bone, "torso"]) if "torso" in p["segs"] \
            else self._bone_parts(p, [bone])
        if not parts or not len(parts[0]):
            return None
        cells = parts[0].astype(np.float64)
        trunk = parts[1].astype(np.float64) if len(parts) > 1 and len(parts[1]) else None
        c0 = cells.mean(axis=0)
        ct = trunk.mean(axis=0) if trunk is not None else c0 + np.array([0.0, 0.0, 1.0])
        down = np.array([0.0, 0.0, -1.0])

        def axis_of(c, cen, ctr):
            d = c - cen
            if len(c) > 1:
                u, sv, vt = np.linalg.svd(d, full_matrices=False)
                a = vt[0] if sv[0] > 1e-9 else down.copy()
            else:
                a = down.copy()
            away = float(a @ (cen - ctr))
            if away < -1e-9 or (abs(away) <= 1e-9 and float(a @ down) < 0.0):
                a = -a
            return a / max(float(np.linalg.norm(a)), 1e-12)

        a1 = axis_of(cells, c0, ct)
        rest = p.get("rest") or {}
        a0, f0, med0 = a1, None, None
        fx, fy = p.get("chest", CHEST0)        # the way the chest faces (not the way it walks)
        f0 = np.array([float(fx), float(fy), 0.0])
        if bone in rest and len(rest[bone]) > 1:
            base = np.asarray(self._claim_cells(p, *self._person_cells(p))).min(axis=0)
            r = np.asarray(rest[bone], np.float64) + base
            rt = (np.asarray(rest["torso"], np.float64) + base) if "torso" in rest else None
            rc = r.mean(axis=0)
            # `rest` holds only the limbs that were posed: the trunk that stayed is the trunk now
            rct = rt.mean(axis=0) if rt is not None else ct
            a0 = axis_of(r, rc, rct)
            med0 = rct - rc
        # the swing that took the rest axis a0 to the present axis a1
        v = np.cross(a0, a1)
        cdot = float(a0 @ a1)
        if cdot > -0.999999:
            vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
            R = np.eye(3) + vx + vx @ vx / (1.0 + cdot)
        else:                                  # a half turn: about any axis square to a0
            k = np.cross(a0, [1.0, 0.0, 0.0])
            if np.linalg.norm(k) < 1e-6:
                k = np.cross(a0, [0.0, 1.0, 0.0])
            k = k / np.linalg.norm(k)
            R = 2.0 * np.outer(k, k) - np.eye(3)
        f = R @ f0
        f = f - (f @ a1) * a1
        if np.linalg.norm(f) < 1e-9:
            f = np.cross(a1, [1.0, 0.0, 0.0]) if abs(a1[0]) < 0.9 else np.cross(a1, [0.0, 1.0, 0.0])
        f = f / np.linalg.norm(f)
        e2 = np.cross(a1, f)
        # which way is MEDIAL at rest: toward the trunk, square to the axis
        if med0 is None:
            med0 = ct - c0
        fr = R.T @ f                            # the anterior at rest, in the rest frame
        m = med0 - (med0 @ a0) * a0
        if np.linalg.norm(m) > 1e-9 and float(np.cross(a0, fr) @ m) < 0.0:
            e2 = -e2
        t = (cells - c0) @ a1
        zs = trunk[:, 2] if trunk is not None else cells[:, 2]
        z_heart = float(zs.min()) + 0.75 * float(zs.max() - zs.min() + 1.0)
        return {"o": c0, "axis": a1, "f": f, "e2": e2, "t0": float(t.min()) - 0.5,
                "t1": float(t.max()) + 0.5, "trunk": ct, "z_heart": z_heart}

    @staticmethod
    def _frame_az_along(fr, cell):
        """(az degrees in the bone's own frame, along 0-1 from the proximal end)
        of a lattice cell, read in `_bone_frame`'s frame. A cell on the bone's
        axis (within a third of a cell) has no side of its own: az 0."""
        v = np.asarray(cell, np.float64) - fr["o"]
        t = float(v @ fr["axis"])
        along = (t - fr["t0"]) / max(fr["t1"] - fr["t0"], 1e-9)
        perp = v - t * fr["axis"]
        if float(np.linalg.norm(perp)) < 0.34:
            return 0.0, min(max(along, 0.0), 1.0)
        az = float(np.degrees(np.arctan2(perp @ fr["e2"], perp @ fr["f"]))) % 360.0
        return az, min(max(along, 0.0), 1.0)

    def _isch_dead(self, p, bones):
        """The share of these bones' tissue that a tourniquet has killed (0-1),
        the mean over the bones (S2 A, Q2): the limb law reads it, so a dead
        forearm cannot grip and a dead leg cannot bear weight. 0.0 for a body
        that never wore a cuff."""
        isch = p.get("isch")
        if not isch or not bones:
            return 0.0
        return sum(isch.get(b, {}).get("dead", 0.0) for b in bones) / len(bones)

    def _cut_section(self, p, bones):
        """How much of these bones' muscle section is CUT, in m^2.

        A cut is a plane through the limb, depth times length, and muscle on
        the far side of it pulls on nothing. That is the whole of why a cut
        arm is a weak arm, and it lands on the same section-times-stress the
        strength of every limb is already made of."""
        return sum(w["depth_m"] * w["length_m"]
                   for w in (p.get("wounds") or ())
                   if w.get("open") and w.get("bone") in bones)

    def _in_reach(self, mine, theirs, reach_v=None):
        # `reach_v` is in VOXELS, not metres — the one that got passed metres
        # made every grip in the sim let go on the tick after it was made
        """Is that thing close enough to get a hand to?

        Measured between the two SURFACES, not the two centres. A body is five
        voxels across, so a centre-to-centre limit is a different limit for a
        child and for a cart, and the arm is the same arm."""
        # BETWEEN THE BOXES, IN THREE DIMENSIONS. This measured across the
        # FLOOR and ignored height — a stated softness, on the grounds that an
        # ankle 1.5 m overhead is about an arm away and it only goes wrong at
        # five metres. It went wrong: a man standing on a ledge held someone
        # lying on the ground twenty voxels below him, for two hundred ticks,
        # because the vertical gap was not in the sum at all.
        #
        # It compounded with the other half. `half` allowed for both things'
        # WIDTH, and a body lying down is its own length wide — so a prone man
        # was reachable from much further than a standing one. True of his
        # hand, and not true of the rest of him.
        mine = np.asarray(mine)
        theirs = np.asarray(theirs)
        gap = 0.0
        for ax in (0, 1, 2):
            a0, a1 = float(mine[:, ax].min()), float(mine[:, ax].max())
            b0, b1 = float(theirs[:, ax].min()), float(theirs[:, ax].max())
            gap = max(gap, b0 - a1, a0 - b1)   # 0 if the boxes overlap on this
        return gap <= reach_v if reach_v is not None else \
            gap <= BODY["reach_m"] / max(0.1 * self.scale, 1e-9)

    @staticmethod
    def _span_xy(cells):
        """How wide that thing is on the floor, in voxels."""
        return float(max(np.ptp(cells[:, 0]), np.ptp(cells[:, 1])))

    def _body_span(self, p, what):
        """A LENGTH THIS BODY ACTUALLY HAS, in metres, measured off its own
        voxels — not a number typed into `BODY` and hoped to match.

        This is the loop worth having: the sim already knows where every voxel
        of every limb is, so anything shaped like "how far can an arm reach" or
        "how wide is a pair of shoulders" has a ground truth sitting right
        there. Reading it instead of declaring it means a child, a giant and a
        one-armed man each get the right answer without anybody writing three
        rows, and it means the number cannot drift out of step with the body it
        describes — which `BODY["reach_m"]` silently had, at 0.30 m against an
        arm that is 0.45 m long.

        The constants stay as the FALLBACK, for a body with no segments
        declared. They are what a person is like when nobody has said."""
        got = (p.get("_span") or {}).get(what)
        if got is not None:
            return got
        vox_m = 0.1 * self.scale
        out = None
        arms = [a for a in self._limbs(p) if "arm" in a]
        if what == "reach" and arms:
            piv = self._pivot_of(p, arms[0])
            parts = self._limb_parts(p, arms[0])
            if piv is not None and parts:
                cells = np.concatenate([c for c in parts if len(c)]) \
                    if any(len(c) for c in parts) else None
                if cells is not None and len(cells):
                    comp, sl = self._person_cells(p)
                    if comp is not None and comp.any():
                        own = self._claim_cells(p, comp, sl)
                        j = np.asarray(piv) + own.min(axis=0)
                        out = float(np.abs(cells - j).sum(axis=1).max()) * vox_m
        elif what == "leg":
            legs = [k for k in self._limbs(p) if "leg" in k]
            cells = [c for k in legs
                     for c in (self._limb_parts(p, k) or []) if len(c)]
            if cells:
                c = np.concatenate(cells)
                out = float(c[:, 2].max() - c[:, 2].min() + 1) * vox_m
        elif what == "shoulders" and len(arms) > 1:
            mids = []
            for a in arms:
                parts = self._limb_parts(p, a)
                if parts and len(parts[0]):
                    mids.append(parts[0].mean(axis=0))
            if len(mids) > 1:
                out = float(np.abs(mids[0] - mids[1]).max()) * vox_m
        if out is None or out <= 0.0:
            out = {"reach": BODY["reach_m"],
                   "shoulders": BODY["off_hand_m"] * 2,
                   "leg": BODY["crouch_m"] / BODY["crouch_frac"]}[what]
        p.setdefault("_span", {})[what] = out
        return out

    def _reach_of(self, p):
        """How far THIS body's arm goes."""
        return self._body_span(p, "reach")

    def _legs_of(self, p):
        """What THIS body's legs can push with, in newtons: the cross-section
        of the legs it actually has, times a muscle stress."""
        vox_m = 0.1 * self.scale
        legs = [k for k in self._limbs(p) if "leg" in k]
        cells = [c for k in legs
                 for c in (self._limb_parts(p, k) or []) if len(c)]
        if not cells:
            return BODY["legs_N"]
        c = np.concatenate(cells)
        tall = max(float(c[:, 2].max() - c[:, 2].min() + 1), 1.0)
        area = (len(c) / tall) * vox_m * vox_m     # mean section, in m^2
        leg_bones = [b for k in legs for b in self._bones(p, k)]
        area = max(area - self._cut_section(p, leg_bones), 0.0)
        dead = self._isch_dead(p, leg_bones)       # a dead leg bears no weight (S2 A, Q2)
        if dead > 0.0:
            area *= 1.0 - dead
        return area * BODY["muscle_Pa"]

    def _torque_of(self, p, limb, fallback):
        """What THIS body's joint can hold, in newton-metres.

        Section times stress gives the force; times the insertion lever gives
        the torque. Everything in it is the body's own except the stress and
        the lever fraction, and both of those are named guesses in the ledger
        rather than numbers hiding in a law."""
        vox_m = 0.1 * self.scale
        # THE LIMB'S OWN BONES, not everything beyond them. What a joint must
        # HOLD is its whole subtree — that is `_hold_torque`, and it is right —
        # but what its muscle can PRODUCE is the section of the limb the muscle
        # is IN. Summed over the subtree, a waist borrowed the arms hanging at
        # its sides and came out at 570 N.m instead of 400.
        bones = [b for b in self._bones(p, limb) if b in (p.get("segs") or {})]
        parts = self._bone_parts(p, bones) if bones else None
        cells = [c for c in (parts or []) if len(c)]
        if not cells:
            return fallback
        c = np.concatenate(cells)
        tall = max(float(c[:, 2].max() - c[:, 2].min() + 1), 1.0)
        area = (len(c) / tall) * vox_m * vox_m
        area = max(area - self._cut_section(p, bones), 0.0)
        dead = self._isch_dead(p, bones)           # dead muscle makes no torque (S2 A, Q2)
        if dead > 0.0:
            area *= 1.0 - dead
        return area * BODY["muscle_Pa"] * (tall * vox_m) * BODY["lever_frac"]

    @staticmethod
    def _face_axis(p):
        """Which horizontal axis this body's joints turn in. A shoulder swings
        in the plane the body FACES, so which angles can be drawn at all
        depends on which way it is standing — measured, a right arm has 30 of
        30 angles facing one way and 1 of 30 facing the other. Anything that
        remembers what a joint can do has to remember which way it was
        facing when it found out."""
        dx, dy = p.get("facing", (1.0, 0.0))
        return 0 if abs(dx) >= abs(dy) else 1

    def _crouch_of(self, p):
        """How far THIS body can sink before it pushes off.

        A jump is the legs' force over the distance they have to apply it, and
        that distance is a LEG — so a child's jump and a tall man's differ
        without anybody writing two rows. `crouch_frac` is what is left over as
        a guess: how much of a leg's length a crouch actually uses, which is
        not something the lattice knows."""
        return self._body_span(p, "leg") * BODY["crouch_frac"]

    def _hand(self, p, own, toward=None, free=False):
        """WHICH HAND does this.

        The dominant one, unless the other is plainly the one for the job: a
        thing on your left is nearer your left hand, and past enough difference
        that beats being right-handed. So a body uses its good hand most of the
        time and its other hand when the world asks for it — never because a
        random number said so, and the sim carries no die anyway.

        An arm that is not there, or already holding something, is not a
        candidate: an indisposed hand is not a choice, it is an absence."""
        arms = [a for a in self._limbs(p) if "arm" in a]
        if not arms:
            return None
        dom = str(p.get("handed", "right")) + " arm"
        # HALF THIS BODY'S OWN SHOULDER WIDTH, read off its own shoulders.
        # The cost of the off hand is that it is weaker; the benefit of the
        # near hand is not reaching across yourself; those trade at about the
        # distance between your shoulders, so a broad man switches hands later
        # than a narrow one and nobody writes that down twice.
        bias = 0.5 * self._body_span(p, "shoulders") / max(0.1 * self.scale, 1e-9)
        busy = (p.get("held") or {}).get("arm") if free else None
        best, score = None, None
        for a in sorted(arms):                # sorted: deterministic ties
            if a == busy:
                continue
            f = self._fist_of(p, a, own)
            if f is None:
                continue                      # nothing left of that arm
            sc = 0.0 if a == dom else bias
            if toward is not None:
                sc += float(np.linalg.norm(np.asarray(toward, np.float64)
                                           - np.asarray(f, np.float64)))
            if score is None or sc < score:
                best, score = a, sc
        return best

    def _hand_strength(self, p, arm):
        """What that hand can put out. The other one is weaker, which is most
        of what having a dominant hand MEANS."""
        full = self._strength(p)
        full = full if arm == str(p.get("handed", "right")) + " arm" \
            else full * BODY["off_hand"]
        dead = self._isch_dead(p, self._bones(p, arm)[-1:])   # a dead forearm cannot grip (S2 A, Q2)
        return full * (1.0 - dead) if dead > 0.0 else full

    def _fist_of(self, p, name, own):
        """The HAND at the end of one arm: the voxel of the last bone furthest
        from the joint the whole limb hangs from.

        Said that way because it is true in every pose. It used to be "the mean
        x and y of the arm, at its lowest z", which is exactly right for an arm
        hanging straight down and exactly wrong for one held out — a horizontal
        arm's mean is its ELBOW. Measured: a man holding 38 kg at arm's length
        had the load's weight applied half way up his forearm, which halved the
        lever and let him keep his feet under a load that should plainly have
        taken him off them. Arms only hung down when that was written."""
        parts = self._limb_parts(p, name, own)
        piv = self._pivot_of(p, name)
        if not parts or piv is None or not len(parts[-1]):
            return None
        far = parts[-1]
        j = np.asarray(piv) + np.asarray(own).min(axis=0)
        d = np.abs(far.astype(np.int64) - j.astype(np.int64)).sum(axis=1)
        return [float(c) for c in far[int(np.argmax(d))]]

    def _fists(self, p, own):
        """Where this body's hands are. A body with no segments declared falls
        back to its own middle height."""
        fists = []
        for k in self._limbs(p):              # per ARM, not per bone: an elbow
            if "arm" not in k:                # is not a second hand
                continue
            f = self._fist_of(p, k, own)
            if f is not None:
                fists.append(f)
        if not fists:
            zs = own[:, 2]
            fists = [[float(own[:, 0].mean()), float(own[:, 1].mean()),
                      float(zs.min()) + 0.55 * float(zs.max() - zs.min())]]
        return fists
