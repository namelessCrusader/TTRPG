"""Lattice mechanics: support, falling columns, impacts, shattering, cracking, toppling."""
import math
import numpy as np
from .materials import (AIR, AIR_DENS, DRAG_CD, FALL_SUBSTEPS, GRAVITY, NOFLUID, STONE, TICK_S, TSHOCK, _FDENS_ARR, _SPAN_ARR, _TOUGH_ARR, _TSHOCK_ARR)
from .kernels import HAVE_NUMBA, _relax_nb
from .label import label
from .window import _piece_bounds, _relax_list, compact_labels
from .bodies import CARRIED


class SupportMixin:

    def _law_support(self):
        """Gravity + STRENGTH for solids. Support starts at the ground; a voxel sitting
        directly on a supported voxel INHERITS ITS CARRIER'S slack (clamped to its own
        span — ash on a stone lintel is still just ash); support that travels SIDEWAYS
        spends 1 span per hop. When the span runs out, the overhang BREAKS and falls —
        so an ash slab crumbles where the wood that held it burned away. Slack only ever
        DECAYS along a path: a plank resting on a shelf cannot lend the shelf reach it
        never had (the old full-restore-on-carry rule made any two-layer slab an
        infinite girder — one table leg carried 89% of the tabletop). SPAN is physical
        length: measured in 10 cm reference voxels, converted by the world's scale.
        Falling voxels carry their heat and clinging fluid, landing on solids or pools."""
        steps, moved_tot, airborne = None, None, None
        for _ in range(FALL_SUBSTEPS):
            r = self._support_once(steps)
            if r is None:
                break
            steps, moved, airborne = r
            moved_tot = moved if moved_tot is None else (moved_tot | moved)
            if not steps.any():
                break
        return self._cash_impacts(moved_tot, airborne)

    def _support_once(self, steps):
        """One sweep of the support law. Returns the columns' remaining fall
        budget, what moved, and what is STILL IN THE AIR — or None when
        nothing is falling any more.

        A column with several voxels of fall owed this tick comes back through
        here for each of them: matter may not skip over what it might have
        landed on, so the support question is asked again between every cell."""
        solid = self.mat != AIR
        hang = solid[:, :, 1:] > solid[:, :, :-1]            # any solid with air below?
        if not hang.any():
            return None                          # last tick's landings still pay
        # The slack field is a relaxation run to fixpoint over the WHOLE grid, and
        # it was 45% of all sim time. But it is a pure function of the material
        # layout: if not one voxel changed material since last tick, last tick's
        # answer is still exactly right. Comparing the layout costs ~0.2 ms
        # against ~134 ms to recompute, and it is correct by construction rather
        # than by remembering to invalidate — in a standing room nothing moves,
        # so this is skipped almost every tick.
        grip = (self._grip_cells() | self._borne_cells()) \
            if self.persons else frozenset()
        if self._slack_mat is not None \
                and np.array_equal(self._slack_mat, self.mat) \
                and grip == self._slack_grip \
                and self._rub_gen == getattr(self, "_slack_rub", None):
            slack = self._slack
        else:
            slack = self._relax_slack(solid, grip)
            self._slack_mat, self._slack = self.mat.copy(), slack
            self._slack_grip = grip
            self._slack_rub = self._rub_gen
        falling = solid & (slack < 0)
        if not falling.any():
            return None
        if steps is None:                        # once a tick: how fast is each
            if self.free_fall_to_seam and self._free_fall_to_seam(solid, falling):
                return None                      # the lattice has less to do now
            steps = self._fall_speed(falling)    # column actually going?
        go = falling & (steps > 0)[:, :, None]
        if not go.any():
            return np.zeros_like(steps), np.zeros(self.shape, bool), falling
        moved = self._settle(solid, go)
        return np.maximum(steps - 1, 0), moved, falling

    def _fall_speed(self, falling):
        """How fast each falling column is going, in whole voxels this tick.

        THE AIR PUSHES BACK. A falling column shoves air out of its way, and
        what that costs is drag — half rho C_d A v-squared — against a weight
        of m g. The two balance at a terminal speed, and because the AREA is
        one voxel face however deep the column is, the same matter spread thin
        falls slower than the same matter balled up. That is the whole of a
        parachute, and there is nothing about parachutes in it: a canopy is a
        large area with very little mass behind it, which is a SHAPE, and the
        sim has always known the shape.

        Before this everything unsupported dropped exactly one voxel a tick,
        so a feather and an anvil fell alike and nothing ever accelerated."""
        vox_m = 0.1 * self.scale
        col = falling.any(axis=2)
        kg = (self.smass * falling).sum(axis=2) / 1000.0
        area = vox_m * vox_m
        drag = (0.5 * AIR_DENS * DRAG_CD * area
                * self.vfall * self.vfall / np.maximum(kg, 1e-9))
        self.vfall = np.where(
            col, np.maximum(self.vfall + (GRAVITY - drag) * TICK_S, 0.0), 0.0)
        self.fdrop = np.where(col, self.fdrop + self.vfall * TICK_S / vox_m, 0.0)
        steps = np.minimum(np.floor(self.fdrop), FALL_SUBSTEPS).astype(np.int32)
        self.fdrop -= steps
        return steps

    def _relax_slack(self, solid, grip=frozenset()):
        """Support reach, spread from the ground until it stops changing.

        A GRIP IS AN EDGE IN THIS GRAPH. Support relaxes from the ground up
        through material, and a hand is not material — so nothing on the
        lattice could ever hold a hanging man, and everything a hand carried
        had to rest on something. Cells a footed body holds within its
        strength (`_grip_cells`) seed as supported, exactly like the ground
        does: the load hangs from the body, and the body stands on its feet."""
        # STRENGTH FADES WITH THE MASS THAT IS LEFT. Span used to be read
        # straight off the material, so a trunk voxel eaten to a tenth of
        # itself carried like sound timber right up to the tick it became ash.
        # Reach is a property of the beam that is actually there: full span
        # while at least half the material remains, then falling away with
        # what is left. A burning tree now comes down onto its own fire,
        # a fire-thinned lintel drops its load, and scene `frac` fills (a
        # stick at 0.6) keep their strength — which is why the knee is at a
        # half and not at one.
        # A CELL WITH NOTHING IN IT IS NOT A BROKEN PIECE OF ANYTHING. Debris
        # that has fallen away, burned, or been cleared leaves the flag behind,
        # and the next thing to occupy that cell would inherit "is a fragment".
        # Support runs every tick, so an emptied cell is always caught while it
        # is still empty.
        self.rubble &= self.mat != AIR
        # (Span is worked out for the solid cells alone: it is read nowhere
        # else, and an empty cell reaches nothing. Each cell's span is made
        # of its own mass and material, so the cells listed give the bits the
        # whole lattice would, at a tenth of the price in a big world.)
        if self._windows_on():
            at = np.flatnonzero(solid)
            m = self.mat.reshape(-1)[at]
            sc = self.scomp.reshape(-1)[at] if self.mixes_live else None    # T1: a mix packs
            packed = np.clip(self.smass.reshape(-1)[at] / np.maximum(      # against its own density
                self._solid_rho(m, sc) * self.vox_l, 1e-9), 0.0, 1.0)
            span = np.zeros(self.shape, np.int16)
            span.reshape(-1)[at] = np.round(
                _SPAN_ARR[m] / self.scale
                * np.clip(packed / 0.5, 0.0, 1.0)).astype(np.int16)
        else:
            packed = np.clip(self.smass / np.maximum(
                self._solid_rho(self.mat, self.scomp if self.mixes_live else None)
                * self.vox_l, 1e-9), 0.0, 1.0)
            span = np.round(_SPAN_ARR[self.mat] / self.scale
                            * np.clip(packed / 0.5, 0.0, 1.0)).astype(np.int16)
        slack = np.full(self.shape, -1, np.int16)
        slack[:, :, 0][solid[:, :, 0]] = span[:, :, 0][solid[:, :, 0]]
        if grip:
            gi = tuple(np.asarray(list(grip), np.int64).T)
            slack[gi] = np.where(solid[gi],
                                 np.maximum(slack[gi], span[gi]), slack[gi])
        # THE SAME FIXPOINT, COMPILED (`_relax_nb`): a queue of the cells whose
        # inputs changed in place of whole-lattice rounds. It reaches the same
        # field (kernels.py says why); the loop below is the plain path and
        # the definition, and `w.fused = False` runs it.
        if self.fused and HAVE_NUMBA:
            if self._windows_on():
                return _relax_list(at, solid, span, self.rubble, slack)
            return _relax_nb(solid, span, self.rubble, slack)
        while True:
            cand = np.full(self.shape, -1, np.int16)         # carried from directly below:
            cand[:, :, 1:] = np.where(                       # inherit the carrier's slack
                solid[:, :, 1:] & (slack[:, :, :-1] >= 0), slack[:, :, :-1], -1)
            nb = np.full(self.shape, -1, np.int16)           # sideways/hanging: span - 1
            nb[1:, :, :] = np.maximum(nb[1:, :, :], slack[:-1, :, :] - 1)
            nb[:-1, :, :] = np.maximum(nb[:-1, :, :], slack[1:, :, :] - 1)
            nb[:, 1:, :] = np.maximum(nb[:, 1:, :], slack[:, :-1, :] - 1)
            nb[:, :-1, :] = np.maximum(nb[:, :-1, :], slack[:, 1:, :] - 1)
            nb[:, :, 1:] = np.maximum(nb[:, :, 1:], slack[:, :, :-1] - 1)
            # NOT FROM ABOVE. Resting against the underside of something is not
            # being held up by it — that takes a bond in TENSION, and the only
            # thing in this sim that has one is a hand (`_grip_cells` seeds
            # those). Measured: a glass block touching only a shelf's underside
            # hung there for ever, and a bottle knocked off a shelf punched
            # through it and stopped halfway, wedged in its own hole.
            # nb[:, :, :-1] = np.maximum(nb[:, :, :-1], slack[:, :, 1:] - 1)
            # A PIECE RESTS ON WHAT IS UNDER IT AND HOLDS NOTHING. `nb` is
            # support arriving sideways, which is what a continuous member
            # does; rubble is not a member, so it gets `cand` — carried from
            # directly below — and nothing else.
            reach = np.where(self.rubble, cand, np.maximum(cand, nb))
            new = np.where(solid, reach, -1).astype(np.int16)
            new = np.minimum(new, np.where(self.rubble, 0, span))
            new = np.maximum(new, slack)         # own material can carry (ash caps at 1)
            if (new == slack).all():
                break
            slack = new
        return slack

    def _free_fall_to_seam(self, solid, falling):
        """A WHOLE BODY IN THE AIR IS THE SOLVER'S, not the lattice's.

        `_topple` has handed tipping clusters to the seam since the seam
        existed. This is the other half, and without it half the scenes that
        look like they exercise the solver do not touch it: a dropped vat
        fell by this law, one cell at a time, straight down, and so could not
        turn as it fell, could not land on a corner, and could not arrive
        anywhere but a cell centre — the voxel free-fall this phase exists to
        replace. `drop_test` ran end to end without creating a single
        particle.

        What counts as a free body: a connected cluster every cell of which
        is unsupported AND which touches no supported matter. The second half
        matters — an overhang is unsupported at its tip while its root is
        held, and that is a thing that breaks off, which the span law already
        decides. Nothing here is holding any part of a free body up.

        It leaves with the speed its fall has already earned, so a cluster
        that has been dropping for three ticks does not start again at rest.
        """
        if not falling.any() or not falling.sum() >= 2:
            return 0
        lab = np.where(falling,
                       np.arange(falling.size, dtype=np.int32).reshape(self.shape), -1)
        while True:                                  # ANY-material flood: a glass
            new = lab.copy()                         # vat on a wood shelf is held
            for axis in range(3):                    # by it, whatever it is made of
                a = [slice(None)] * 3; b = [slice(None)] * 3
                a[axis], b[axis] = slice(None, -1), slice(1, None)
                a, b = tuple(a), tuple(b)
                both = falling[a] & falling[b]
                new[a] = np.where(both, np.maximum(new[a], lab[b]), new[a])
                new[b] = np.where(both, np.maximum(new[b], lab[a]), new[b])
            if (new == lab).all():
                break
            lab = new
        _, comp = np.unique(lab[falling], return_inverse=True)
        if not comp.size:
            return 0
        n = int(comp.max()) + 1
        L = np.full(self.shape, -1, np.int32)
        L[falling] = comp.astype(np.int32)
        anchored = np.zeros(n, bool)
        held = solid & ~falling
        for axis in range(3):
            a = [slice(None)] * 3; b = [slice(None)] * 3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            m1 = falling[a] & held[b]
            if m1.any():
                anchored[L[a][m1]] = True
            m2 = falling[b] & held[a]
            if m2.any():
                anchored[L[b][m2]] = True
        took = 0
        for li in np.nonzero(~anchored)[0]:
            cells = np.argwhere(L == li)
            if not 2 <= len(cells) <= 20000:
                continue
            if self.persons:                         # a person in the air is a
                want = set(map(tuple, cells.tolist()))   # body, and bodies keep the
                mine = False                         # old path this phase
                for q in self.persons:
                    self._person_cells(q)
                    if (q.get("_own") or frozenset()) & want:
                        mine = True
                        break
                if mine:
                    continue
            h = float(self.fallh[tuple(cells.T)].mean()) * 0.1 * self.scale
            vz = -math.sqrt(2.0 * 9.81 * h) if h > 0.0 else 0.0
            self._things_moved(self.eid[tuple(cells.T)])   # the eid goes too
            self.matter.promote(cells, vel=(0.0, 0.0, vz / (0.1 * self.scale)))
            took += 1
        return took

    def _settle(self, solid, falling):
        """Everything the support law does once it knows what is unsupported."""
        fdens = self._fdens_grid().astype(np.float32) * 1000.0   # fluid, g/L (a mix: its own row's)
        sdens = self.smass / self.vox_l                                  # this solid, g/L
        arrived = np.zeros(self.shape[:2], bool)             # which columns moved this tick
        moved = np.zeros(self.shape, bool)
        for z in range(1, self.shape[2]):                    # bottom-up: columns stay whole
            pool = self.fvol[:, :, z - 1] >= 0.5 * self.cap
            # a solid DENSER than the pool below SINKS into it and the displaced
            # fluid rides up into the vacated cell (the swap moves it) — stone
            # drops through a pond, wood is lighter than water and rests afloat
            sinks = sdens[:, :, z] > fdens[:, :, z - 1]
            m = falling[:, :, z] & (self.mat[:, :, z - 1] == AIR) & (~pool | sinks)
            arrived |= m
            if not m.any():
                continue
            # EVERYTHING A CELL CARRIES RIDES WITH THE MATTER (`CARRIED`).
            # `rubble` was once left out: a shard fell one voxel, arrived as
            # ordinary glass, and the span it was never supposed to have caught
            # it again — debris dropped exactly once and then hung there.
            xs, ys = np.nonzero(m)
            hi, lo = (xs, ys, z), (xs, ys, z - 1)
            fall, air = self._lift(hi), self._lift(lo)
            self._put(lo, fall)                              # the cell and the
            self._put(hi, air)                               # air swap
            # the drop height RIDES with the voxel and grows — except through
            # liquid, where drag bleeds it away fast (a pond is a cushion, and
            # it also forgives the speed the fall brought INTO the water)
            self._rub_gen += 1
            h = fall[CARRIED.index("fallh")]
            self.fallh[lo] = np.where(pool[m], h * 0.4, h + 1.0)
            self.fallh[hi] = 0.0
            moved[:, :, z - 1] |= m
            # SPLASH: the displaced fluid (now sitting where the solid was) is
            # shoved outward + thrown as droplets where neighbors have room
            for x, y in np.argwhere(m & pool):
                self._splash(int(x), int(y), z)
        # a LIGHTER solid that just STOPPED on a pool still SLAPS it — a log
        # hitting a pond splashes even though it will float. Only columns that
        # actually MOVED this tick slap (a resting float never re-splashes).
        sdens2 = self.smass / self.vox_l
        landed = (self.mat[:, :, 1:] != AIR) & (self.mat[:, :, :-1] == AIR) \
            & (self.fvol[:, :, :-1] >= 0.5 * self.cap) \
            & (sdens2[:, :, 1:] <= (self._fdens_grid()[:, :, :-1] if self.mixes_live
                                    else _FDENS_ARR[self.fl[:, :, :-1]]) * 1000.0)
        for x, y, zl in np.argwhere(landed):
            if arrived[x, y] and zl + 1 < self.shape[2]:
                self._splash(int(x), int(y), int(zl), int(zl) + 1)
        return moved            # the caller cashes impacts once, after the last
                                # sweep of the tick — a thing part-way through a
                                # multi-voxel fall has not landed on anything

    def _cash_impacts(self, moved, airborne=None):
        """IMPACT: a voxel that fell and now CANNOT move has landed — its
        stored drop is cashed in as m·g·h against the material's toughness.
        Brittle stuff (glass, char) shatters; wood just thuds; a fall cushioned
        by a pond arrives with no height to cash."""
        self._just_shattered = set()
        rest = (self.mat != AIR) & (self.fallh > 0.5)
        if moved is not None:
            rest &= ~moved
        # STILL FALLING IS NOT LANDED. "Did not move this tick" used to be a
        # safe reading of "has come to rest", because everything unsupported
        # moved a voxel every single tick. Now that a thing can be going slower
        # than a voxel a tick, a light body in open air stops for a tick
        # between steps — and cashing its drop then shattered it in MID-AIR:
        # measured, a glass block bursting at z=8 on the way down to a pond it
        # never reached. What matters is whether the floor is under it.
        if airborne is not None:
            rest &= ~airborne
        if not rest.any():
            return
        vox_m = 0.1 * self.scale
        cells = np.argwhere(rest)
        e = (self.smass[rest] / 1000.0) * 9.81 * self.fallh[rest] * vox_m
        lands_box = None
        if len(e):
            k = int(np.argmax(e))
            x, y, z = (int(v) for v in cells[k])
            below = int(self.mat[x, y, z - 1]) if z > 0 else STONE
            self._noise((x, y, z), float(e.sum()),
                        [int(self.mat[x, y, z]), below])
            # ONE "lands" event for the worst impact, not one per cell cashed —
            # a dropped block is one landing, however many voxels came down
            # with it. Breaks it causes are linked to this, not to each other.
            # DEFERRED (`_defer`): this fires on nearly every ordinary
            # settle, so emitting it right away risks landing in the same
            # tick as a shout and shifting the id `_say` stamps on its own
            # sound record (`_defer`'s docstring has the full account).
            lands_box = self._defer("lands", place=(x, y, z),
                                     db=float(e.sum()), text="")
        thr = _TOUGH_ARR[self.mat[rest]] * 1000.0 * vox_m * vox_m
        flaw = np.array([0.7 + 0.6 * self._flaw01(*c) for c in cells], np.float32)
        thr = thr * flaw                                 # every piece has its own
        for (x, y, z), broke, ov, joules in zip(cells, e > thr,
                                                e / np.maximum(thr, 1e-9), e):
            if broke:                                    # worst flaw — no two break
                with self.because(lands_box if lands_box is not None
                                   else self._cause_near((x, y, z)), "near"):
                    self._shatter(int(x), int(y), int(z), float(ov))   # alike
            # AND WHAT IT LANDED ON TAKES THE SAME BLOW. Newton's third law: the
            # impulse is shared, so the floor is tested against its OWN toughness
            # with the same energy. Before this a falling anvil could only ever
            # hurt itself — it went through a glass table without marking it,
            # which is the wrong way round. A struck cell that shatters drops
            # whatever it was holding, so collapses cascade on their own.
            zb = int(z) - 1
            if zb < 0 or self.mat[int(x), int(y), zb] == AIR:
                continue
            below = int(self.mat[int(x), int(y), zb])
            tb = (_TOUGH_ARR[below] * 1000.0 * vox_m * vox_m
                  * (0.7 + 0.6 * self._flaw01(int(x), int(y), zb)))
            if joules > tb:
                with self.because(lands_box if lands_box is not None
                                   else self._cause_near((x, y, zb)), "near"):
                    self._shatter(int(x), int(y), zb,
                                  float(joules / max(tb, 1e-9)))
        self.fallh[rest] = 0.0

    @staticmethod
    def _flaw01(x, y, z):
        """Deterministic per-cell 'flaw' in [0, 1) — real brittle failure is
        governed by the worst microscopic flaw in the piece (Weibull's
        statistics), so neither the breaking point nor the fragments should
        come out uniform. No RNG: the same drop always breaks the same way."""
        return ((x * 73 + y * 151 + z * 211) % 17) / 16.0

    def _shatter(self, x, y, z, over=1.0):
        """Brittle failure: the voxel's cohesion is gone. The cell empties and
        its mass scatters as UNEVEN fragments of the same material — heavier
        shards near the break, slighter ones farther out, and the more the
        impact exceeded the material's toughness (`over`), the farther the
        spray reaches. Any fluid it held SPILLS where the cell stood. Boxed in
        on every side, the rubble stays wedged — a crushed cell deep inside a
        wall is still a wall of rubble."""
        m0, e0 = int(self.mat[x, y, z]), int(self.eid[x, y, z])
        mass, E0 = float(self.smass[x, y, z]), float(self.E[x, y, z])
        f, fv, fp = int(self.fl[x, y, z]), float(self.fvol[x, y, z]), float(self.fpot[x, y, z])
        # THE MIXES THE CELL HOLDS (T1) ride out with it: a shard of a bronze
        # bar is bronze, and a spilled brine is still brine. Shatter changes no
        # species, so it is not a gated legacy law: it carries (D6 asks for a
        # gate only where a law transmutes). The shatter THRESHOLD gate
        # (toughness x (1 - damage)) is T9's, wave 13b.
        s0 = int(self.scomp[x, y, z]) if self.mixes_live else 0
        fc0 = self._frow((x, y, z)) if self.mixes_live else 0
        nx, ny, nz = self.shape
        ring1 = ((1, 0), (-1, 0), (0, 1), (0, -1))
        ring2 = ((2, 0), (-2, 0), (0, 2), (0, -2), (1, 1), (1, -1), (-1, 1), (-1, -1))
        offsets = ring1 + (ring2 if over >= 2.0 else ())
        targets = []                                     # (cell, weight): heavy
        if z > 0 and self.mat[x, y, z - 1] == AIR:       # shards drop, light ones
            targets.append(((x, y, z - 1), 1.6))         # fly — never evenly
        # ...but never INTO a hole that is being made in the same instant. When
        # a row of cells breaks together under one blow, each was scattering
        # fragments into its neighbours' just-emptied cells and refilling them,
        # so a shattered plate stayed a plate and an anvil sat on the wreckage
        # of the table it had just smashed. Fragments fall out of a break; they
        # do not queue up to plug it.
        fresh = self._just_shattered
        for dx, dy in offsets:
            tx, ty = x + dx, y + dy
            d = abs(dx) + abs(dy)
            if (tx, ty, z) in fresh:
                continue
            if 0 <= tx < nx and 0 <= ty < ny and self.mat[tx, ty, z] == AIR:
                targets.append(((tx, ty, z),
                                (0.4 + 1.2 * self._flaw01(tx, ty, z)) / (0.5 + d)))
        if not targets:
            return
        self._just_shattered.add((x, y, z))
        # BREAKS ARE REPORTED, not raised here — V2 (4.4) holds them until
        # `_tick_end` so a tick's breaks come out in one order (by cell)
        # no matter which mask or loop found them first. The cause is
        # whatever `because()` block called in, if any; a cell shattered
        # with none (a wall nobody was blamed for) just carries None.
        self._break_log.append((x, y, z, m0, None, None,
                                 self._cause, self._link))
        self._rub_gen += 1
        # THE CELL IS EMPTIED, all of what it carried (`_lift`): nothing is
        # there to be a piece of, rubble is blunt (an edge is made, and
        # breaking unmakes it), and the shards keep the eid (below).
        self._lift((x, y, z))
        if s0:
            self.scomp[x, y, z] = 0              # the mix went with the shards, not into `recast`
        wsum = sum(wt for _t, wt in targets)
        for (tx, ty, tz), wt in targets:
            self.mat[tx, ty, tz] = m0
            if s0:
                self.scomp[tx, ty, tz] = s0
            self.smass[tx, ty, tz] += mass * wt / wsum
            self.E[tx, ty, tz] += E0 * wt / wsum
            # AND IT IS A PIECE NOW, not a member. This said "the voxel's
            # cohesion is gone" in its own first line and then handed the
            # fragments back as ordinary glass — which spans eight reference
            # voxels, sixteen cells at 5 cm. So a bottle knocked off a shelf
            # shattered correctly, conserved its 48750 g exactly, and then hung
            # in the air beside the shelf: measured, 47 of 234 shards with
            # nothing at all beneath them. Broken glass does not cantilever.
            self.rubble[tx, ty, tz] = True
            # AND IT IS BLUNT. The target was air, but air can still hold an
            # edge that the last thing there left behind; a shard must not
            # land on it and become a blade.
            self.edge[tx, ty, tz] = 0.0
            # BUT IT IS STILL A PIECE OF THE SAME THING. A shard is the bottle
            # broken, not new glass from nowhere; which pieces it is now is
            # the identity pass's question, and it needs the eid to ask it.
            self.eid[tx, ty, tz] = e0
        if fv > 0 and f != NOFLUID:              # the held fluid pours out where
            self.fl[x, y, z] = f                 # the cell used to be — the flow
            self.fvol[x, y, z] = fv              # law takes it from there
            self.fpot[x, y, z] = fp
            if self.mixes_live:
                self.fcomp[x, y, z] = fc0

    def _law_crack(self):
        """THERMAL SHOCK — a brittle solid spanning a steep temperature
        difference cracks: the hot side expands, the cold side refuses.
        TSHOCK is data (the ΔT one face can carry); the conduction field
        already knows every face's ΔT, so this law only reads it. A cold
        vat licked by flame bursts; a gently warmed one is fine."""
        if not (np.isin(self.mat, list(TSHOCK)).any()):
            return
        T = self.T()
        hits = set()
        for axis in range(3):
            a = [slice(None)] * 3; b = [slice(None)] * 3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            dT = np.abs(T[a] - T[b])
            for s_, d_ in ((a, b), (b, a)):          # the brittle cell is at d_
                m = dT > _TSHOCK_ARR[self.mat[d_]]
                if not m.any():
                    continue
                off = [0, 0, 0]
                off[axis] = 1 if d_ is b else 0
                for x, y, z in np.argwhere(m):
                    hits.add((int(x) + off[0], int(y) + off[1], int(z) + off[2]))
        for x, y, z in hits:
            if self.mat[x, y, z] != AIR:
                # no decision caused this -- heat did -- so the only cause
                # on offer is whatever else was disturbed near here this
                # tick (a torch set against the vat, say).
                with self.because(self._cause_near((x, y, z)), "near"):
                    self._shatter(x, y, z)

    def _topple(self, cells, axis, s, pivot, zb, owner=None):
        """PROMOTE a tipping cluster to a free rigid body (the sandbox lineage's
        promote/demote protocol): its voxels leave the grid and the body leans
        about the pivot line with real pendulum acceleration — slow at first,
        crashing at the end — until it lands and re-rasterizes."""
        # THROUGH THE SEAM, when it is nobody's. There is no pivot, axis or
        # angle to hand over any more: gravity and the contact surface decide
        # how it falls. A cluster that is someone's BODY — a passed-out man
        # keeling over — keeps the old path for this phase: people stay on the
        # lattice until `arms` and `crouch` migrate (roadmap 93, decision 4).
        # Measured without this: the unconscious became particles and the
        # mind lost its body, so nobody ever came round.
        if owner is None:
            cells = np.asarray(cells, np.int64)
            self._things_moved(self.eid[tuple(cells.T)])   # the eid goes too
            return self.matter.promote(cells)

        cells = self._with_limbs(np.asarray(cells, np.float32),   # a swung
                                 owner)                            # arm too
        idx = tuple(cells.astype(np.int64).T)
        lifted = self._lift(idx)                         # lift the body off the
        carried = self._as_body(lifted)                  # grid, all it carries
        masses = carried["masses"]                       # (held moisture too)
        m = np.maximum(masses, 1e-6)
        d_lat = cells[:, axis] - pivot
        d_z = cells[:, 2] - zb
        dxc = float((m * d_lat).sum() / m.sum()) * s     # COM lever, toward the overhang
        dzc = float((m * d_z).sum() / m.sum())
        body = {
            "cells": cells, **carried,
            "axis": axis, "s": s, "pivot": float(pivot), "zb": float(zb),
            "theta": 0.0, "omega": 0.0,
            "phi0": max(float(np.arctan2(max(dxc, 0.1), max(dzc, 0.5))), 0.02),
            "L": max(float(np.hypot(dxc, dzc)), 2.0),
            # WHOSE BODY THIS IS, if it is anybody's. The sim is the thing that
            # picked this flesh up and it is the thing that puts it down, so it
            # KNOWS where the person went: making identity re-derive that from
            # adjacency afterwards is throwing away an answer we already hold,
            # and re-deriving it is what handed a fainting person their
            # rescuer's body.
            "owner": owner}
        # WEDGED check: if the body cannot even BEGIN to lean (its first sliver
        # of rotation already collides), it is stuck against something — put it
        # back and leave it until the world changes around it. Without this, a
        # blocked vat tip-landed-retipped every tick, forever.
        probe = np.round(self._body_pose(body, 0.1)).astype(np.int64)
        nx, ny, nz = self.shape
        inb = ((probe[:, 0] >= 0) & (probe[:, 0] < nx) & (probe[:, 1] >= 0)
               & (probe[:, 1] < ny) & (probe[:, 2] >= 0) & (probe[:, 2] < nz))
        wedged = int((~inb).sum()) + int((self.mat[tuple(probe[inb].T)] != AIR).sum())
        if wedged > max(2, len(cells) // 50):
            # restore, ALL of it (the drop height too, which this once left
            # at nought); the snapshot stays valid, so no re-check until the
            # lattice actually changes
            self._put(idx, lifted)
            return
        self.bodies.append(body)
        self._torque_solid = None
        pid = next((r["pid"] for r in self.persons if r["name"] == owner),
                    None) if owner else None
        place = tuple(float(v) for v in cells.mean(axis=0)) if len(cells) else None
        self._defer("topples", actor=pid, place=place, text="")

    def _key_at(self, idx):
        """`_key()` at the cells whose flat indices are `idx`, without the key
        of the whole world: the same values for the same cells."""
        mat = self.mat.reshape(-1)[idx]
        if not self._thing_lut.any():
            return mat
        e = self.eid.reshape(-1)[idx]
        named = self._is_thing(e)
        if not named.any():
            return mat
        return np.where(named, 256 + e.astype(np.int64), mat)

    def _law_torque(self):
        """TIPPING. An object is a connected cluster of same-material solid voxels
        (a wood tree on a stone floor is its own body, RESTING on the floor).
        Two checks per rooted body, and one verdict — where the weight hangs
        (center of mass) must sit over what carries it:
          1. BASE: COM vs the support footprint on the ground. Fails -> the whole
             body topples 90 degrees about the footprint edge, toward the overhang.
          2. WAIST: wherever the body NARROWS sharply (a section under 40% of the
             plane above it — an axe notch, table legs under a top), everything
             above stands on that section; its COM must sit over the section's
             bbox. Fails -> the part above tears off and topples about the waist
             edge. This is why a tree with a thin hinge left FALLS TOWARD THE
             NOTCH, and a table that lost its east legs tips east.
        Runs only when the solid lattice changed, or what is one thing did
        (`stamp`). v1: instant quarter-turn, no arc; same-material contact
        merges unnamed bodies (wood-on-wood is one body), and a named thing
        is its own body whatever touches it (`_key`)."""
        if self._torque_sleep > 0:
            self._torque_sleep -= 1
            return
        solid = self.mat != AIR
        if self._torque_solid is not None and bool((solid == self._torque_solid).all()):
            return
        self._torque_solid = solid.copy()
        # SAME-MATERIAL PIECES, each labelled by its largest flat index. This
        # was a flood here, one voxel a round over the whole world, and it was
        # half of every tick in a house where anyone moved: 1,895 ms against
        # 2 for one pass of `label` (label.py), which gives the same array.
        #
        # ON THE KEY (3.3, E4): a named thing is one lump whatever it is made
        # of, so an axe leaning on its wooden haft tips as one axe, head and
        # all, and not as a haft with an iron lump left hanging in the air.
        # With nothing named the key IS `mat`, the very array, and every
        # label is what it was.
        #
        # THE LAW WORKS ON THE MATTER, NOT ON THE LATTICE (`windows`). Every
        # number below is a sum over the solid cells, so the cells are listed
        # once (`idx`, in flat order, which is the order a boolean mask would
        # give) and nothing after this is the size of the world. In a yard of
        # 1.6 million voxels about 33 thousand are solid, and this law was
        # 25 ms a tick whenever anyone walked, because the labels, a pair of
        # `np.indices` and a dozen masks were all lattice-sized. With
        # `windows` off the labels come from the whole-lattice labeller, as
        # they always did; everything from here on is the same code, so the
        # two give the same bits (tests/test_scale.py).
        ny, nz = self.shape[1], self.shape[2]
        idx = np.flatnonzero(solid)
        if not len(idx):
            return
        cx = idx // (ny * nz)
        cy = (idx // nz) % ny
        cz = idx % nz
        cw = self.smass.reshape(-1)[idx]
        windowed = self._windowed()
        if windowed:
            comp, below, n = compact_labels(idx, self._key_at(idx), self.shape)
        else:
            lab = label(solid, self._key(), canon="max", fused=self.fused)
            _, comp = np.unique(lab[solid], return_inverse=True)
            n = int(comp.max()) + 1
            Lf = np.full(solid.size, -1, np.int32)
            Lf[idx] = comp.astype(np.int32)
            below = np.where(cz > 0, Lf[idx - 1], -1)    # the piece it stands on
        # per-(cluster, z) stats in one bincount family
        ids = comp * nz + cz
        cnt = np.bincount(ids, minlength=n * nz).reshape(n, nz)
        msz = np.bincount(ids, weights=cw, minlength=n * nz).reshape(n, nz)
        mxz = np.bincount(ids, weights=cw * cx, minlength=n * nz).reshape(n, nz)
        myz = np.bincount(ids, weights=cw * cy, minlength=n * nz).reshape(n, nz)
        mass = msz.sum(axis=1)
        com_x = mxz.sum(axis=1) / np.maximum(mass, 1e-9)
        com_y = myz.sum(axis=1) / np.maximum(mass, 1e-9)
        if windowed:
            (bx0, bx1, by0, by1, fx0, fx1, fy0, fy1, fz, sup_lo, sup_hi,
             has_ground) = _piece_bounds(comp, cx, cy, cz, below, n, nz)
        else:
            bx0 = np.full((n, nz), 10 ** 6); bx1 = np.full((n, nz), -(10 ** 6))
            by0 = np.full((n, nz), 10 ** 6); by1 = np.full((n, nz), -(10 ** 6))
            np.minimum.at(bx0, (comp, cz), cx); np.maximum.at(bx1, (comp, cz), cx)
            np.minimum.at(by0, (comp, cz), cy); np.maximum.at(by1, (comp, cz), cy)
            # what each voxel STANDS on: the ground, or another piece
            contact = (cz == 0) | ((below >= 0) & (below != comp))
            cL = comp[contact]
            fx0 = np.full(n, 10 ** 6); fx1 = np.full(n, -(10 ** 6))
            fy0 = np.full(n, 10 ** 6); fy1 = np.full(n, -(10 ** 6))
            fz = np.full(n, 10 ** 6)
            np.minimum.at(fx0, cL, cx[contact]); np.maximum.at(fx1, cL, cx[contact])
            np.minimum.at(fy0, cL, cy[contact]); np.maximum.at(fy1, cL, cy[contact])
            np.minimum.at(fz, cL, cz[contact])
            # RIDERS: a cluster of another material whose entire support stands
            # on ONE body (a leaf canopy around a trunk, a stick on a table)
            # belongs to that body when it tips — it rides along instead of
            # hovering behind.
            sup_lo = np.full(n, 10 ** 9); sup_hi = np.full(n, -1)
            cnz = contact & (cz > 0)
            np.minimum.at(sup_lo, comp[cnz], below[cnz])
            np.maximum.at(sup_hi, comp[cnz], below[cnz])
            has_ground = np.zeros(n, bool)
            cg = contact & (cz == 0)
            if cg.any():
                has_ground[np.unique(comp[cg])] = True
        rooted = fx1 >= 0

        def rider_mask(li):
            rs = np.nonzero(~has_ground & (sup_lo == li) & (sup_hi == li))[0]
            rs = rs[rs != li]
            return rs if len(rs) else None

        def body_cells(li, rs, above=None):
            """The cells of piece `li` and of its riders `rs` (those above
            layer `above`, when given), as an (n, 3) array in flat order."""
            m = comp == li
            if rs is not None:
                m |= np.isin(comp, rs)
            if above is not None:
                m &= cz > above
            at = np.flatnonzero(m)
            return np.stack((cx[at], cy[at], cz[at]), axis=1)

        def whose(cells):
            """Which PEOPLE this cluster is made of.

            Flesh touching flesh is one cluster to the labeller, so a cluster
            can be nobody, one person, or two people in contact."""
            if not self.persons or not len(cells):
                return []
            want = set(map(tuple, np.asarray(cells, np.int64).tolist()))
            out = []
            for q in self.persons:
                self._person_cells(q)             # make sure the claim is today's
                if (q.get("_own") or frozenset()) & want:
                    out.append(q)
            return out

        def outside(comx, comy, x0, x1, y0, y1):
            ox = comx - x1 if comx > x1 + 0.5 else (comx - x0 if comx < x0 - 0.5 else 0.0)
            oy = comy - y1 if comy > y1 + 0.5 else (comy - y0 if comy < y0 - 0.5 else 0.0)
            return ox, oy

        tipped = 0
        for li in np.nonzero(rooted)[0]:
            if tipped >= 4:
                break
            # 1. BASE check: whole-body COM over the ground footprint
            ox, oy = outside(com_x[li], com_y[li], fx0[li], fx1[li], fy0[li], fy1[li])
            if ox != 0 or oy != 0:
                cells = body_cells(li, rider_mask(li))
                mine = whose(cells)
                if any(q["alive"] and q["awake"] for q in mine):
                    continue                     # A PERSON ON THEIR FEET HOLDS
                                                 # THEMSELVES UP. This law asks
                                                 # where a rigid lump's weight
                                                 # hangs, which is the right
                                                 # question for a wall and the
                                                 # wrong one for someone stood
                                                 # at a lip: it tipped them over
                                                 # for leaning. Balance is what
                                                 # the living do that the dead
                                                 # do not, and the sim already
                                                 # has the other half — a body
                                                 # that goes slack is toppled by
                                                 # _collapse, one whose footing
                                                 # is gone falls by _law_footing
                if 2 <= len(cells) <= 20000:
                    ow = mine[0]["name"] if mine else None
                    if abs(ox) >= abs(oy):
                        self._topple(cells, 0, 1 if ox > 0 else -1,
                                     int(fx1[li] if ox > 0 else fx0[li]),
                                     int(fz[li]), owner=ow)
                    else:
                        self._topple(cells, 1, 1 if oy > 0 else -1,
                                     int(fy1[li] if oy > 0 else fy0[li]),
                                     int(fz[li]), owner=ow)
                    tipped += 1
                continue
            # 2. WAIST check: sharp narrowings carry everything above them
            c = cnt[li]
            for lz in range(int(fz[li]), nz - 1):
                if c[lz] == 0 or c[lz + 1] == 0 or c[lz] >= 0.4 * c[lz + 1]:
                    continue
                am = msz[li, lz + 1:].sum()              # the body above the waist
                if am <= 0:
                    continue
                acx = mxz[li, lz + 1:].sum() / am
                acy = myz[li, lz + 1:].sum() / am
                ox, oy = outside(acx, acy, bx0[li, lz], bx1[li, lz],
                                 by0[li, lz], by1[li, lz])
                if ox == 0 and oy == 0:
                    continue
                cells = body_cells(li, rider_mask(li), lz)
                mine = whose(cells)
                if any(q["alive"] and q["awake"] for q in mine):
                    break                        # nor does a standing body tear
                                                 # in half at the waist: a person
                                                 # narrows at the neck, the waist
                                                 # and both knees, and every one
                                                 # of those reads as a notch
                if 2 <= len(cells) <= 20000:
                    ow = mine[0]["name"] if mine else None
                    if abs(ox) >= abs(oy):
                        self._topple(cells, 0, 1 if ox > 0 else -1,
                                     int(bx1[li, lz] if ox > 0 else bx0[li, lz]),
                                     lz + 1, owner=ow)
                    else:
                        self._topple(cells, 1, 1 if oy > 0 else -1,
                                     int(by1[li, lz] if oy > 0 else by0[li, lz]),
                                     lz + 1, owner=ow)
                    tipped += 1
                break                                    # one waist verdict per body

    def _collapse(self, p, comp, sl):
        """An unconscious body is a slack object: promote it as a free rigid
        body keeling toward the openest side — the same fall as a chopped tree."""
        xs0, ys0 = sl[0].start, sl[1].start
        cells = np.argwhere(comp).astype(np.int64)       # this person's body only
        cells[:, 0] += xs0
        cells[:, 1] += ys0
        z_lo = int(cells[:, 2].min())
        zt = z_lo + int(0.6 * (int(cells[:, 2].max()) - z_lo))    # chest height
        best, pick = -1, None
        for axis, s in ((0, 1), (0, -1), (1, 1), (1, -1)):
            edge = int(cells[:, axis].max() if s > 0 else cells[:, axis].min())
            mid = int(round(float(cells[:, 1 - axis].mean())))
            free = 0
            for k in range(1, int(cells[:, 2].max()) - z_lo + 2):
                q = [0, 0]
                q[axis], q[1 - axis] = edge + s * k, mid
                if (0 <= q[0] < self.shape[0] and 0 <= q[1] < self.shape[1]
                        and self.mat[q[0], q[1], zt] == AIR):
                    free += 1
                else:
                    break
            if free > best:
                best, pick = free, (axis, s)
        axis, s = pick
        base = cells[cells[:, 2] <= z_lo + 1]
        pivot = int(base[:, axis].max() if s > 0 else base[:, axis].min())
        self._topple(cells.astype(np.float32), axis, s, float(pivot), float(z_lo),
                     owner=p["name"])
