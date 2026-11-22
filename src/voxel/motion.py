"""Where legs can take a body: places, routes, walking, leaping, footing, balance."""
import heapq
import numpy as np
from .materials import (AIR, BODY, FLESH, GRAVITY, PUSH_THROUGH, SOUND, TICK_S, _DENS_ARR)
from .policy import (WILL, _DIRS)
from . import contact, stairs
from .pathlaw import PathingMixin
from .reflex import ReflexMixin
from .bodycontact import BodyContactMixin, ankle_cop_m

# How full of liquid a cell can be and still be waded through: half a cell,
# the same "flooded" line `physiology.BREATH` draws for a face in water.
WADE_CELLS = 0.5

# How many ticks of floor position are kept for a body being hauled, to know
# the speed it is hauled at (a quarter of a second at 0.025 s a tick).
HAUL_TICKS = 10

# The eight steps a route may take, and what each costs: 70 along a grid
# axis, 99 on a diagonal (sqrt(2) * 70 = 98.99). Grid steps first, so that of
# two routes of the same length the sweep keeps the one found first, always.
_STEPS8 = ((1, 0, 70), (-1, 0, 70), (0, 1, 70), (0, -1, 70),
           (1, 1, 99), (1, -1, 99), (-1, 1, 99), (-1, -1, 99))


class FlatRoutesMixin:
    """The flat (one level) fit grid and route sweep: the base that
    pathlaw.PathingMixin calls with the step range at 0. Moved out of
    MotionMixin unchanged, so the mixin can sit in front of it (T11, 13a)."""

    def _fit_grid(self, cells, p=None):
        """Columns this body could stand in, WHOLE — and knows about.

        A column is passable by how much is actually IN it, not by whether it is
        pure air: a body shoves through a hedge or a curtain and does not shove
        through a wall. Same continuous read as sight and sound — before this, a
        single leaf voxel was as impassable as masonry. PUSH_THROUGH is a
        stand-in for a force the body has not got yet; the force law replaces
        it with strength against what holds the material."""
        # ONE rule for where a body may stand, not two. This had its own copy
        # of the fill test, so teaching _walkable that a column needs a floor
        # under it taught the PLACES but not the PLANNER — and the planner is
        # the half that decides where the legs actually go.
        # THE FLOOR AS THIS BODY LAST SAW IT, not as it is. A mind with no
        # belief yet (or no mind at all — a bare route query) reads the world,
        # which is the honest default for a question nobody is asking.
        free = p.get("free") if p is not None else None
        walk_ok = self._walkable(cells) if free is None else free.copy()
        if p is not None:
            walk_ok &= p["known"]     # you cannot plan a route through rooms
                                      # you have never seen
        # A BODY IS NOT A POINT. The route is walked by the body's CENTRE, but
        # the body is several voxels across, so a centre column is only usable
        # if the whole footprint fits there. Without this the planner happily
        # routes a centre into a column two voxels off a wall, the legs refuse
        # (correctly — the shoulder is in the masonry), and the walker jams
        # against the wall shuffling sideways forever. Measured on the
        # glasshouse: a body stuck 2 voxels short of its door for 500 ticks.
        cx = int(round(float(cells[:, 0].mean())))
        cy = int(round(float(cells[:, 1].mean())))
        for c in cells:                       # a body does not block ITSELF, and
            walk_ok[int(c[0]), int(c[1])] = True   # this has to be true BEFORE
                                              # the footprint test: its own flesh
                                              # is denser than PUSH_THROUGH, so
                                              # eroding first left a body unable
                                              # to stand where it was standing
        fit = walk_ok.copy()
        nx, ny = walk_ok.shape
        for dx, dy in {(int(c[0]) - cx, int(c[1]) - cy) for c in cells}:
            if dx == 0 and dy == 0:
                continue
            sh = np.zeros_like(walk_ok)
            xs_lo, xs_hi = max(0, -dx), min(nx, nx - dx)
            ys_lo, ys_hi = max(0, -dy), min(ny, ny - dy)
            if xs_lo < xs_hi and ys_lo < ys_hi:
                sh[xs_lo:xs_hi, ys_lo:ys_hi] = \
                    walk_ok[xs_lo + dx:xs_hi + dx, ys_lo + dy:ys_hi + dy]
            fit &= sh
        fit[cx, cy] = True                    # wherever it is now, it fits
        return fit, (cx, cy)

    def _routes(self, cells, targets, p=None, fit=None):
        """Every target's route, from ONE sweep.

        A menu prices a dozen places at once, and asking the planner a dozen
        separate questions re-walks the same room a dozen times. One sweep
        records where every column was reached FROM, and each route is read
        back out of it — cheaper than the three separate searches this replaced,
        for four times the places."""
        if fit is None:
            fit, start = self._fit_grid(cells, p)
        else:
            start = (int(round(float(cells[:, 0].mean()))),
                     int(round(float(cells[:, 1].mean()))))
        # EIGHT WAYS, PRICED BY LENGTH. The sweep was breadth-first over the
        # four grid steps, so every route was a staircase and a body crossing
        # a room at 45 degrees walked 1.41 times as far as it had to (a real
        # walker cuts the corner). A diagonal step is one voxel along each
        # axis at once, and it costs what it is long: 99 against 70 for a
        # grid step, which is sqrt(2) to 1 part in 10,000, in whole numbers so
        # equal routes tie exactly. It may not CUT a corner: both grid cells
        # beside it must fit too, so no route squeezes between two walls that
        # touch only at an edge, and what can be reached at all is exactly
        # what the four-way sweep reached.
        nx, ny = fit.shape
        ok = fit.ravel().tolist()
        s0 = start[0] * ny + start[1]
        cost = {s0: 0}
        prev = {s0: None}
        heap = [(0, s0)]
        while heap:
            d, cur = heapq.heappop(heap)
            if d > cost[cur]:
                continue
            cx, cy = divmod(cur, ny)
            for dx, dy, c in _STEPS8:
                x, y = cx + dx, cy + dy
                if not (0 <= x < nx and 0 <= y < ny):
                    continue
                nxt = x * ny + y
                if not ok[nxt]:
                    continue
                if dx and dy and not (ok[x * ny + cy] and ok[cx * ny + y]):
                    continue
                nd = d + c
                if nd < cost.get(nxt, nd + 1):
                    cost[nxt] = nd
                    prev[nxt] = cur
                    heapq.heappush(heap, (nd, nxt))
        cand = None
        out = {}
        for t in targets:
            goal = (int(t[0]), int(t[1]))
            if not fit[goal]:                 # a doorway in an outer wall has
                if cand is None:              # no room for a body's centre —
                    cand = np.argwhere(fit)   # aim at the nearest place that does
                if not len(cand):
                    out[(int(t[0]), int(t[1]))] = None
                    continue
                d = np.abs(cand[:, 0] - goal[0]) + np.abs(cand[:, 1] - goal[1])
                near = cand[int(d.argmin())]
                goal = (int(near[0]), int(near[1]))
            cur = goal[0] * ny + goal[1]
            if cur not in prev:
                out[(int(t[0]), int(t[1]))] = None                     # no way through
                continue
            path = []
            while cur is not None:
                path.append(divmod(cur, ny))
                cur = prev[cur]
            out[(int(t[0]), int(t[1]))] = path[::-1][1:]
        return out


class MotionMixin(PathingMixin, FlatRoutesMixin, ReflexMixin, BodyContactMixin):
    # PathingMixin is FIRST: with the step range at 0 (BODY has no step_up_m
    # key) it gives the old fit and the same paths, tie for tie (T11, 13a).

    def _walkable(self, cells):
        """Columns this body could STAND in: clear enough to occupy, and with
        something under them to stand on.

        The floor test is not a detail. Without it a column of open air reads
        as walkable, so the route planner would happily march a body straight
        out over a chasm and the legs would carry it — walking on nothing,
        because nothing ever asked what was underneath. It also makes a gap a
        real obstacle, which is what gives jumping across one a point.

        FLAT, and that is a known limit rather than an oversight: every column
        is tested at the body's own height and no other, so a kerb is a wall and
        a stair is a wall. Making it height-aware is item 5, and it was TRIED —
        see "What a step up cost" in the roadmap for why it came back out and
        what it has to wait for."""
        zlo, zhi = int(cells[:, 2].min()), int(cells[:, 2].max())
        col = (slice(None), slice(None), slice(zlo, zhi + 1))
        packed = self.smass[col] / np.maximum(
            _DENS_ARR[self.mat[col]] * self.vox_l, 1e-9)
        clear = (packed < PUSH_THROUGH).all(axis=2)
        if zlo <= 0:
            return clear
        return clear & (self.mat[:, :, zlo - 1] != AIR)

    def _bearing(self, dx, dy):
        """Which way that is, in words. A label, so a menu row reads like
        something a person would say to themselves."""
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return "here"
        return _DIRS[int(round(float(np.arctan2(dy, dx)) / (np.pi / 4))) % 8]

    def _reach_along(self, fit, ax_, ay_, dx, dy, reach):
        """The furthest this body could get going THAT WAY without turning.

        This is what makes "straight on" and "to the left" real destinations
        rather than steps. A body that has not seen anywhere worth naming can
        still have an intention about a direction, and the legs still do the
        walking — the difference between choosing a way to go and being asked
        to place your feet."""
        n = float(np.hypot(dx, dy))
        if n < 1e-9:
            return None
        dx, dy = dx / n, dy / n
        nx, ny = fit.shape
        last = None
        for k in range(1, int(reach) + 1):
            x, y = int(round(ax_ + dx * k)), int(round(ay_ + dy * k))
            if not (0 <= x < nx and 0 <= y < ny) or not fit[x, y]:
                break
            last = (x, y)
        if last is None or max(abs(last[0] - ax_), abs(last[1] - ay_)) < 2.0:
            return None                   # nowhere to get to that way
        return last

    def _places(self, p, cells, fit, seek=()):
        """Every spot this body could NAME, drawn from its OWN map.

        A place is where an intention points. They are named, and the naming
        matters: "explore" told a body to go and look at something the SIM
        picked, which is a decision taken away from whoever is choosing. Now
        the unseen edges are listed one per direction, so going to look at the
        dark doorway north is a different choice from going to look at the
        unlit end of the hall, and the body makes it.

        An exit it has never laid eyes on is not a destination — it is not even
        an option — so a stranger in a burning house must find the door the
        hard way and a resident (add_person(knows_world=True)) runs straight
        for it. That difference used to be impossible to express: everyone read
        the world's register.

        `seek` lists the spots a linked mind means to strike from (3.5, step
        M10; `intent["seek"]`, which the will layer hands on): each one
        `{"who", "at", "band"}`, someone, where they stand and how far from
        them, centre to centre in cells, to stand. Each is a `go:seek` row
        that carries whom it is for (`_seek_spot`)."""
        ax_ = float(cells[:, 0].mean()); ay_ = float(cells[:, 1].mean())
        vox_m = 0.1 * self.scale
        out = []

        # A PLACE YOU ARE ALREADY AT IS NOT SOMEWHERE TO GO. Arrival is judged
        # from the body's NEAREST CELL and these were judged from its CENTRE,
        # and the two are a whole body apart: `_reach_along` will name a spot
        # 2 cells from the centre, arrival triggers at 2.4 from the edge, so a
        # body three cells across arrived at "straight on" the instant it chose
        # it — cleared the goal, was offered the same clipped spot next
        # decision, and took it again. Measured over 40 worlds: 41 of 127 leg
        # options with a target, 32%, were places the body was already stood
        # at, and one witness spent 180 ticks in a corner walking into a wall
        # two cells in front of it.
        #
        # That is a third of a seven-slot attention budget spent on options
        # that do nothing, and worse than nothing for a policy learning from
        # this: choosing one is indistinguishable from choosing to stand still,
        # so the same situation teaches both answers. Judged here by the same
        # arithmetic that judges arrival, so the two cannot drift apart again.
        arrive = WILL["arrive_m"] / max(vox_m, 1e-9)

        baulked = p.get("_baulked") or {}

        def add(tag, label, xy):
            xy = (int(xy[0]), int(xy[1]))
            if float(np.abs(cells[:, :2] - np.array(xy)).max(axis=1).min()) <= arrive:
                return                        # already there
            if self.tick - baulked.get(xy, -10 ** 9) < WILL["baulk_for"]:
                return                        # tried that, could not get there
            d = max(abs(xy[0] - ax_), abs(xy[1] - ay_))
            out.append({"tag": tag, "label": label, "xy": xy, "away": d})

        for e in self.exits:                       # doors it has SEEN
            if p["known"][int(e[0]), int(e[1])]:
                add("go:exit", "the door " + self._bearing(e[0] - ax_, e[1] - ay_),
                    (e[0], e[1]))
        # THE UNSEEN EDGES: somewhere known that touches somewhere unknown.
        # Walking to one is how the map grows, and it is why a body that has
        # been dropped into a dark building does not simply stand in the dark.
        unseen = ~p["known"]
        edge = np.zeros_like(fit)
        edge[1:, :] |= unseen[:-1, :]; edge[:-1, :] |= unseen[1:, :]
        edge[:, 1:] |= unseen[:, :-1]; edge[:, :-1] |= unseen[:, 1:]
        front = np.argwhere(fit & edge & p["known"])
        if len(front):
            d = np.abs(front[:, 0] - ax_) + np.abs(front[:, 1] - ay_)
            near = max(8.0, WILL["arrive_m"] / max(vox_m, 1e-9) * 3)
            keep = front[d > near] if (d > near).any() else front
            kd = np.abs(keep[:, 0] - ax_) + np.abs(keep[:, 1] - ay_)
            seen_dirs = {}
            for i in np.argsort(kd):               # nearest first, one per way
                c = keep[i]
                w = self._bearing(c[0] - ax_, c[1] - ay_)
                if w not in seen_dirs:
                    seen_dirs[w] = c
            for w, c in list(seen_dirs.items())[:4]:
                add("go:frontier", "the unseen ground " + w, c)
        # STRAIGHT ON, AND TO EITHER SIDE — relative to the face, because that
        # is how a person says it. Offered only as far as the body could
        # actually get, so a wall in front removes "straight on" rather than
        # offering a walk into it.
        fx, fy = p.get("facing", (1.0, 0.0))
        for name, (dx, dy) in (("straight on", (fx, fy)),
                               ("to the left", (-fy, fx)),
                               ("to the right", (fy, -fx)),
                               ("back the way it came", (-fx, -fy))):
            tgt = self._reach_along(fit, ax_, ay_, dx, dy,
                                    WILL["step_m"] / max(vox_m, 1e-9))
            if tgt is not None:
                add("go:step", name, tgt)
        # SOMEWHERE IT KNOWS, a way off. Deterministic: the sim carries no die,
        # so which way a mind wanders comes from the tick and the name.
        spots = np.argwhere(fit & p["known"])
        if len(spots) > 1:
            d = np.abs(spots[:, 0] - ax_) + np.abs(spots[:, 1] - ay_)
            far = spots[d > max(6.0, np.percentile(d, 70))]
            if len(far):
                i = (self.tick // max(WILL["decide_every"], 1)
                     + len(p["name"])) % len(far)
                add("go:roam", "the floor "
                    + self._bearing(far[i][0] - ax_, far[i][1] - ay_), far[i])
        # A SPOT TO STRIKE FROM, for a mind that has someone to strike: a
        # spot for the body's middle (`mid`), so "already there" is its own
        for sk in seek:
            xy = self._seek_spot(cells, fit, p, sk)
            if xy is not None and self.tick - baulked.get(xy, -10 ** 9) >= WILL["baulk_for"]:
                out.append({"tag": "go:seek", "label": "where " + sk["who"] + " is in reach",
                            "xy": xy, "who": sk["who"], "mid": True,
                            "away": max(abs(xy[0] - ax_), abs(xy[1] - ay_))})
        seen = {}                                  # two doors due east are two
        for o in out:                              # rows, so say which is which
            seen.setdefault(o["label"], []).append(o)
        for label, rows in seen.items():
            if len(rows) > 1:
                for r in rows:
                    r["label"] = f"{label} ({r['away'] * vox_m:.0f} m)"
        return out

    def _seek_spot(self, cells, fit, p, sk):
        """WHERE THIS BODY'S MIDDLE SHOULD STAND to have `sk["who"]`, who
        stands at `sk["at"]`, between `sk["band"]` cells away, centre to
        centre, and IN LINE with them along x or y: a swing comes round in
        the plane of one axis (`_swing`), at the arm's own place beside the
        body, so a man stood off that line is swung past -- the band was
        measured in line. And ON ITS OWN SIDE of them, along the axis it
        mostly comes from: it walks up to a man, it does not plan a way
        round him (and walking up is the stance the band was measured in:
        a free body turns to face its way, `_square_up`). Of those spots,
        the one nearest to it that it could stand in (`fit`) and has seen,
        and not the one it stands on (its middle within the cell of it:
        where the legs stop stepping toward a spot, `_stride`). None when
        there is none.

        Its MIDDLE, where every other goal is judged from the body's nearest
        cell (`_law_will`): a goal near a person is a spot to stand at, and
        "within arrive_m of my nearest cell" is a whole body short of it.
        The row says so (`mid`)."""
        ax_, ay_ = float(cells[:, 0].mean()), float(cells[:, 1].mean())
        tx, ty = int(sk["at"][0]), int(sk["at"][1])
        lo, hi = sk["band"]
        nx, ny = fit.shape
        vx, vy = ax_ - tx, ay_ - ty
        sx, sy = (1 if vx >= 0 else -1, 0) if abs(vx) >= abs(vy) else (0, 1 if vy >= 0 else -1)
        best = None
        for k in range(int(np.ceil(lo - 1e-6)), int(np.floor(hi + 1e-6)) + 1):
            for x, y in ((tx + sx * k, ty + sy * k),):
                if not (0 <= x < nx and 0 <= y < ny) or not fit[x, y] \
                        or not p["known"][x, y] \
                        or max(abs(x - ax_), abs(y - ay_)) < 1.0:
                    continue
                d = (float(np.hypot(x - ax_, y - ay_)), x, y)
                if best is None or d < best:
                    best = d
        return None if best is None else (best[1], best[2])

    def _underfoot(self, cells):
        """Is there anything under this body's feet to push against?

        The FEET, deliberately: this answers whether the legs have something to
        drive into, which is what deciding to jump and deciding to brace both
        need. Whether a body is HELD UP is a different question with a different
        answer — see _contact — and running them together made a man teetering
        on a lip both unable to jump and unable to fall."""
        zlo = int(cells[:, 2].min())
        if zlo <= 0:
            return True
        c = np.asarray(cells).astype(np.int64)    # int() of each, as it was
        low = c[c[:, 2] == zlo]                   # the soles
        # A SOLE PAST THE EDGE OF THE WORLD HAS NOTHING UNDER IT. The soles
        # are read all at once now, so one sole off the lattice raised an
        # IndexError (the old loop stopped at the first sole that stood on
        # something, and never reached it), and one at x = -1 read the far
        # side of the world. Only the soles inside it are asked.
        nx, ny, nz = self.shape
        inside = ((low[:, 0] >= 0) & (low[:, 0] < nx)
                  & (low[:, 1] >= 0) & (low[:, 1] < ny) & (zlo - 1 < nz))
        low = low[inside]
        return bool((self.mat[low[:, 0], low[:, 1], zlo - 1] != AIR).any())

    def _contact(self, cells, ignore=None):
        """Which cells of this body are actually resting on something else.

        Not the bottom layer — ANY cell with something solid beneath it that is
        not part of this same body. A man at the very lip has his feet over air
        and a forearm still over the stone, and that forearm is genuinely
        touching the world. Standing on your own foot proves nothing.

        And neither does standing on what you are CARRYING, which is what
        `ignore` is for. A man holding a block has his arm directly above it, so
        the block came back as part of his own footprint — which made his base
        as wide as his reach and said he could never be overbalanced by anything
        he could hold. What you carry is not what carries you."""
        cells = np.asarray(cells, np.int64)
        on_floor = cells[:, 2] <= 0                   # the floor of the world
        z1 = np.maximum(cells[:, 2] - 1, 0)
        under = self.mat[cells[:, 0], cells[:, 1], z1] != AIR
        ny, nz = self.shape[1], self.shape[2]
        key = (cells[:, 0] * ny + cells[:, 1]) * nz + cells[:, 2]
        mine = np.isin(key - 1, key)
        if ignore is not None and len(ignore):
            ig = np.asarray(ignore, np.int64)
            mine = mine | np.isin(key - 1,
                                  (ig[:, 0] * ny + ig[:, 1]) * nz + ig[:, 2])
        return cells[on_floor | (under & ~mine)]

    def _leap_speed(self, cells, p=None):
        """The fastest this body can leave the ground, in m/s. The legs put
        legs_N into the floor over a crouch; what is left after holding the
        body's own weight up becomes speed. A heavier body gets less out of the
        same legs, for free, because it is the same arithmetic."""
        kg = float(self.smass[tuple(np.asarray(cells).T)].sum()) / 1000.0
        if kg <= 0.0:
            return 0.0
        net = (self._legs_of(p) if p is not None else BODY["legs_N"]) / kg \
            - GRAVITY
        if net <= 0.0:
            return 0.0                        # too heavy to lift itself
        return float(np.sqrt(2.0 * net * (self._crouch_of(p) if p is not None
                                          else BODY["crouch_m"])))

    def _leap_targets(self, p, cells, fit, vmax):
        """Where this body could LAND if it jumped, and how far that is.

        A jump straight up is a jump that goes nowhere, and going nowhere was
        all the legs could do: the whole point of having a velocity is that it
        has a direction. These are the furthest spots it could stand on in each
        way it could face, within the range its own legs allow — and crucially
        they are not filtered by whether it could WALK there, which is what
        makes leaping a gap different from stepping over one.

        Range at the best angle is v-squared over g. Nothing here picks a
        height or a distance; the legs and the mass do."""
        if vmax <= 0.0:
            return []
        vox_m = 0.1 * self.scale
        reach = int((vmax * vmax / GRAVITY) / max(vox_m, 1e-9))
        if reach < 2:
            return []
        ax_ = float(cells[:, 0].mean()); ay_ = float(cells[:, 1].mean())
        nx, ny = fit.shape
        fx, fy = p.get("facing", (1.0, 0.0))
        out = []
        for name, (dx, dy) in (("straight on", (fx, fy)),
                               ("to the left", (-fy, fx)),
                               ("to the right", (fy, -fx)),
                               ("back the way it came", (-fx, -fy))):
            n = float(np.hypot(dx, dy))
            if n < 1e-9:
                continue
            ux, uy = dx / n, dy / n
            best = None
            for k in range(2, reach + 1):
                x, y = int(round(ax_ + ux * k)), int(round(ay_ + uy * k))
                if not (0 <= x < nx and 0 <= y < ny):
                    break
                if fit[x, y]:
                    best = (x, y, float(k))       # the furthest good landing
            if best is not None:
                out.append({"label": name, "xy": (best[0], best[1]),
                            "away": best[2]})
        return out

    def _leap(self, p, cells, goal=None):
        """Push off — up, or up and ALONG.

        Aimed at somewhere, the body leaves at the angle that carries furthest
        and at exactly the speed that distance needs (v-squared = g times the
        range), so it lands where it meant to instead of hurling itself as hard
        as it can. A place further than the legs can reach is not offered."""
        vmax = self._leap_speed(cells, p)
        if vmax <= 0.0:
            return
        if p is not None:
            self._chose(p, "leap", 2)         # a chosen act: no reflex answers it
        if goal is None:
            self._launch(cells, (0.0, 0.0, vmax), owner=p["name"])
            return
        ax_ = float(cells[:, 0].mean()); ay_ = float(cells[:, 1].mean())
        dx, dy = goal[0] - ax_, goal[1] - ay_
        n = float(np.hypot(dx, dy))
        if n < 1e-9:
            self._launch(cells, (0.0, 0.0, vmax), owner=p["name"])
            return
        d = n * 0.1 * self.scale
        v = min(float(np.sqrt(GRAVITY * d)), vmax)      # just far enough
        half = v / float(np.sqrt(2.0))                  # 45 degrees: best range
        self._launch(cells, (half * dx / n, half * dy / n, half),
                     owner=p["name"])

    def _gather_ticks(self, p):
        """How long this body's legs take to gather after a landing.

        A LANDING COSTS SOMETHING. Nothing said so, and at a quarter-second
        reaction that showed: a body landed and pushed off again INSIDE ONE
        TICK, so its flesh was never on the lattice at a tick boundary and a
        leaper was permanently airborne.

        The time is the body's own: a landing is absorbed by sinking through a
        crouch, and how long that takes under gravity is `sqrt(2 h / g)`. A
        taller body has a deeper crouch and takes longer to gather, which is
        the same square-cube family as the jump itself and wants no constant of
        its own."""
        h = max(self._crouch_of(p), 1e-6)
        return max(1, int(round(float(np.sqrt(2.0 * h / GRAVITY)) / TICK_S)))

    def _firm(self, cells, ignore=None):
        """The cells of this body that rest on something that can BEAR them:
        `_contact`, less the ones whose support is only a pushed-through
        curtain of leaves or a hedge (lighter than PUSH_THROUGH), or matter
        the particle solver is drawing for one tick.

        Said as a property of the matter, not of the place: a foot on a leaf
        is a foot on nothing. A body that stands ON such a thing still counts
        as standing (`_underfoot`, the legs' question), but it does not widen
        the base it balances on."""
        c = self._contact(cells, ignore)
        if not len(c):
            return c
        z1 = np.maximum(c[:, 2] - 1, 0)
        under = (c[:, 0], c[:, 1], z1)
        packed = self.smass[under] / np.maximum(
            _DENS_ARR[self.mat[under]] * self.vox_l, 1e-9)
        return c[(c[:, 2] <= 0) | ((packed >= PUSH_THROUGH)
                                   & ~self.moving[under])]

    @staticmethod
    def _hull_of(pts):
        """The convex hull of a few points in the floor plane (Andrew's
        monotone chain), counter-clockwise, as a list of (x, y). One point or
        two give back themselves."""
        pts = sorted(set(map(tuple, pts)))
        if len(pts) <= 2:
            return pts

        def half(seq):
            out = []
            for q in seq:
                while len(out) >= 2 and (
                        (out[-1][0] - out[-2][0]) * (q[1] - out[-2][1])
                        - (out[-1][1] - out[-2][1]) * (q[0] - out[-2][0])) <= 0:
                    out.pop()
                out.append(q)
            return out
        lo, hi = half(pts), half(pts[::-1])
        return lo[:-1] + hi[:-1]

    @staticmethod
    def _past_hull(hull, x, y):
        """How far (x, y) is outside the hull, as the vector from the nearest
        point of the hull to it; (0, 0) on or inside. For a rectangle this is
        exactly the overshoot on each axis, as the bounding box always gave."""
        n = len(hull)
        if n == 1:
            return x - hull[0][0], y - hull[0][1]
        if n >= 3:
            inside = True
            for i in range(n):
                ax, ay = hull[i]
                bx, by = hull[(i + 1) % n]
                if (bx - ax) * (y - ay) - (by - ay) * (x - ax) < 0:
                    inside = False
                    break
            if inside:
                return 0.0, 0.0
        best = None
        for i in range(n if n >= 3 else 1):
            ax, ay = hull[i]
            bx, by = hull[(i + 1) % n]
            dx, dy = bx - ax, by - ay
            t = ((x - ax) * dx + (y - ay) * dy) / max(dx * dx + dy * dy, 1e-12)
            t = min(max(t, 0.0), 1.0)
            vx, vy = x - (ax + t * dx), y - (ay + t * dy)
            if best is None or vx * vx + vy * vy < best[0]:
                best = (vx * vx + vy * vy, vx, vy)
        return best[1], best[2]

    def _footing(self, p, own):
        """THE FOOTING LAW: what a standing body's weight rests on.

        Weight goes only on contacts that can bear it (`_firm`), and what a
        body carries is not what carries it (a held thing hangs from him).
        A hand on a held thing that is itself resting on the world is a
        contact too, at the fist: a grip is an edge in the support graph.
        Returns the firm contact cells (n, 3) and the grips (fist cells)."""
        held = []
        if p.get("held") or p.get("dragging"):
            held = [np.asarray(cl, np.int64) for cl in self._held_clusters(p)]
        # (a body asked about a PLACE it is not in yet stands on nothing it
        # is made of now: `_step_target` names the cells it would leave)
        gone = p.get("_ghost_from")
        ig = held + ([gone] if gone is not None else [])
        foot = self._firm(own, ignore=np.concatenate(ig) if ig else None)
        grips = []
        if held:
            fists = np.asarray(self._fists(p, own), np.float64)
            # A HOLD TAKEN BY REFLEX IS REACHED FOR: the hand is on the hold
            # (an arm's length is the limit it was taken at), where a hold
            # chosen in the ordinary way has the hand wherever the arm is.
            reached = bool((p.get("held") or {}).get("grab"))
            for i, cl in enumerate(held):
                if not len(cl) or not len(self._contact(cl)):
                    continue                  # it hangs from HIM, not he from it
                d2 = ((cl[None, :, :].astype(np.float64)
                       - fists[:, None, :]) ** 2).sum(-1)
                if reached and i == 0:
                    grips.append([int(c) for c in cl[int(d2.min(0).argmin())]])
                else:
                    grips.append([int(round(c))
                                  for c in fists[int(d2.min(1).argmin())]])
        return foot, grips

    def _com_of(self, own, kg=None):
        """The centre of mass of these cells (x, y, z), in cells, and the
        mass in kg. `kg` (a mass in grams for each row) is for a body asked
        about a place it is not in yet, where the lattice holds none of it."""
        if kg is None:
            kg = self.smass[tuple(own.T)]
        m = float(kg.sum())
        if m <= 0.0:
            return None, 0.0
        return ((own * kg[:, None]).sum(axis=0) / m), m / 1000.0

    def _xcom(self, p, com, z_foot):
        """The EXTRAPOLATED centre of mass (Hof 2005): x + v / w0, w0 = sqrt(g
        / h) for an inverted pendulum of height h. At rest it is the centre of
        mass itself; a body that has been pushed (`push_person`) carries the
        speed the push gave it, and that moves where it will have to put its
        feet. Returns (x, y) in cells."""
        v = p.get("_v")
        if not v:
            return float(com[0]), float(com[1])
        vox_m = 0.1 * self.scale
        h = max((float(com[2]) - z_foot + 0.5) * vox_m, 0.25)
        k = 1.0 / (np.sqrt(GRAVITY / h) * vox_m)
        return float(com[0]) + v[0] * k, float(com[1]) + v[1] * k

    def push_person(self, p, vx, vy):
        """A shove, a gust or a blow gives this standing person a speed over
        their own feet (m/s, floor plane). They are not moved: they are LEANING
        on the speed (`_xcom`), and it dies away at the rate an upright body
        steadies itself, w0 a second. If the speed carries the extrapolated
        centre past the base, the balance reflex has to step."""
        p["_v"] = (float(vx), float(vy))

    def _overbalanced(self, p, own, raw=False):
        """Is this body's OWN weight past its OWN base — and which way.

        The plainest statement of balance there is. The base is the convex
        hull of the contacts that bear weight (`_footing`: the feet that are
        over something, and a hand on a hold), and the weight is the
        extrapolated centre of mass (`_xcom`; at rest, the centre of mass).
        Returns None while the centre is over the base, else how far past it
        is, as (dx, dy) in cells. There is no allowance: this IS his weight,
        and it is already out there. A foot over nothing is simply not in the
        base, so a body with one foot over a gap is overbalanced until the
        reflex has put it back on firm ground (`_balance_reflex`).

        The foot span is the tolerance and needs no other — a contact patch
        three voxels long is a 15 cm foot, which is about what an ankle buys
        you. Measured on the shear: the centre of mass crosses the toes at
        0.3 rad, which is 17 degrees, which is roughly as far as anyone leans
        forward without moving their hips back or taking a step.

        (Asked of a copy of the person that holds nothing and has no speed,
        it asks about a PLACE: where the body would stand after a step.)

        WAVE 14b (gate `body_contact`): a body that is MOVING (it was pushed)
        has the three Hof mechanisms besides the step (bodycontact.py): the
        ankle's torque limits how much of the foot the centre of pressure can
        use (a tired leg shrinks the base), and what is left past the base can
        be brought back by the counter-rotating trunk and arms or a hand on
        something that holds. `raw=True` asks without them. Off: as before."""
        # ONLY A BODY THAT IS STANDING UP CAN BE TAKEN OFF ITS FEET. Said as
        # geometry rather than as a flag: a standing body is TALLER than it is
        # long, and a body lying down is not. That is the whole difference, and
        # it is the difference between toppling and having already toppled.
        #
        # Measured without it: a man lying unconscious across a floor rests on
        # the nine voxels of him that touch it, with the centre of his mass
        # fourteen voxels away along his own length — which reads exactly like
        # a man leaning too far, and the sim threw him across the room. What
        # should happen to the parts of him that are over nothing is that they
        # SAG, and that is the support law's question, not this one.
        tall = float(own[:, 2].max() - own[:, 2].min())
        if tall <= max(float(own[:, 0].max() - own[:, 0].min()),
                       float(own[:, 1].max() - own[:, 1].min())):
            return None
        # AND A HAND ON SOMETHING THE WORLD IS HOLDING UP IS A FOOT. This is
        # the same sum `_pulled_over` does, from the other end: a held thing
        # NOT resting on anything hangs from you and drags you over, and a
        # held thing that IS resting on something is holding itself up — so
        # it can hold you too. A man with a hand on the rail leans out over
        # the lip; a man with nothing to hold leans as far as his own toes
        # allow and stops.
        #
        # The base reaches the HAND, which is conservative: really you can
        # pull on a rail and go well past it, and that wants the arm's
        # tension rather than a wider footprint (item 47).
        foot, grips = self._footing(p, own)
        gone = p.get("_ghost_from")
        if grips:
            g = np.array(grips, np.int64)
            foot = np.vstack([foot, g]) if len(foot) else g
        if not len(foot):
            return None
        com, kg = self._com_of(own, p.get("_ghost_kg"))
        if com is None:
            return None
        hull = self._hull_of(foot[:, :2].tolist())
        dyn = bool(self.body_contact and p.get("_v") and not raw)
        if dyn:
            hull = self._ankle_hull(p, own, hull, kg)
        x, y = self._xcom(p, com, float(foot[:, 2].min()))
        out = list(self._past_hull(hull, x, y))
        if out == [0.0, 0.0]:
            return None
        if dyn:
            axis = 0 if abs(out[0]) >= abs(out[1]) else 1
            ok, how = self._dynamic_recovery(
                p, own, com, kg, float(np.hypot(out[0], out[1])),
                (axis, 1 if out[axis] > 0 else -1), float(foot[:, 2].min()))
            if ok:
                p["_cr"] = (self.tick, how)
                return None
        # NOBODY FALLS INTO WHAT IS HOLDING THEM UP. A man slumped against a
        # wall has his weight past his toes and the wall under his shoulder —
        # that is leaning, and it is stable. Asked only about the floor, the
        # sim tipped him into the wall, the wall stopped him at once, he was
        # set back down against it and tipped again: an unconscious man
        # propped on the edge of the world "overbalanced" 3,925 times in ten
        # simulated minutes. If the way he would go over is blocked by solid
        # matter that is not his own, he is resting on it.
        nx, ny, nz = self.shape
        sx = int(np.sign(out[0])) if abs(out[0]) >= abs(out[1]) else 0
        sy = int(np.sign(out[1])) if sx == 0 else 0
        mine = set(map(tuple, own.tolist()))
        if gone is not None:
            mine |= set(map(tuple, np.asarray(gone).tolist()))
        for c in own:
            x, y, z = int(c[0]) + sx, int(c[1]) + sy, int(c[2])
            if (x, y, z) in mine:
                continue
            if not (0 <= x < nx and 0 <= y < ny):
                return None                      # the edge of the world holds him
            if self.mat[x, y, z] != AIR and z > 0 \
                    and z > int(own[:, 2].min()):   # beside him, not under him
                return None
        return tuple(out)

    def _ankle_hull(self, p, own, hull, kg):
        """The base a MOVING body's ankle can use: the hull, shrunk about its
        middle to what the ankle's torque can reach (bodycontact.ankle_cop_m)
        when that is less than half the foot. A healthy leg uses all of it
        (the same hull); a tired or broken one less."""
        if len(hull) < 2:
            return hull
        st = self._state_of(p, "kick")
        vox_m = 0.1 * self.scale
        xs = [q[0] for q in hull]
        ys = [q[1] for q in hull]
        half = 0.5 * max(max(xs) - min(xs), max(ys) - min(ys)) * vox_m
        if half <= 0.0:
            return hull
        f = min(1.0, ankle_cop_m(kg, st) / half)
        if f >= 1.0:
            return hull
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        return [(cx + (q[0] - cx) * f, cy + (q[1] - cy) * f) for q in hull]

    def _pulled_over(self, p, own):
        """Which way what this body is holding drags it off its own feet — or
        None, if the pair of them still balance.

        Newton's third, asked with arithmetic already here. A hand carries its
        load's weight THROUGH the body into the floor, so the weight that has
        to sit over this body's feet is its own PLUS whatever hangs from its
        fists. A man dangling from your arm hangs beyond your toes, and what
        takes you over the lip after him is not a sideways tug — a load
        hanging still pulls straight DOWN — but that the two of you together
        no longer balance on the ground you are stood on.

        Only what the grip actually HOLDS UP counts, which is why this needs no
        threshold of its own: a load too heavy to lift, or one lying on the
        floor, is resting on the world and pulls nobody anywhere. It is also
        why a stick costs nothing and a man costs everything — the same sum,
        and the mass decides."""
        # AND ONLY WHAT IS ACTUALLY HANGING. A grip carries what the world is
        # not already carrying: a crate still stood on the floor weighs on its
        # own footprint, not on the hand resting against it. Counted anyway, a
        # man was tipped over by a block he had merely taken hold of, one tick
        # after he touched it and while it was still sat on the rock.
        loads = [np.asarray(cl) for cl in self._held_clusters(p)
                 if self._grip_holds(p, cl) and not len(self._contact(cl))]
        if not loads:
            return None
        foot = self._contact(own, ignore=np.concatenate(loads))
        if not len(foot):
            return None
        kg = self.smass[tuple(own.T)]
        m_s = max(float(kg.sum()), 1e-9)
        sx = float((own[:, 0] * kg).sum()) / m_s
        sy = float((own[:, 1] * kg).sum()) / m_s
        # THE WEIGHT COMES IN AT THE FIST, not at the load's own middle. A
        # thing hanging still pulls straight DOWN along the arm holding it, so
        # where its own mass happens to sit says nothing about the holder's
        # balance — that is the WRIST's question, and _grip_holds asks it.
        # Summed at the load's centre instead, a body trailing on the floor
        # behind a hauler read as hanging metres out in front of his toes.
        fists = np.asarray(self._fists(p, own), np.float64)
        m_l, lx, ly = 0.0, 0.0, 0.0
        for cl in loads:
            m_l_here = float(self.smass[tuple(cl.T)].sum())
            if m_l_here <= 0.0:
                continue
            d2 = ((cl[None, :, :].astype(np.float64)
                   - fists[:, None, :]) ** 2).sum(-1)
            hand = fists[int(d2.min(1).argmin())]     # the hand it hangs from
            m_l += m_l_here
            lx += hand[0] * m_l_here
            ly += hand[1] * m_l_here
        if m_l <= 0.0:
            return None
        lx, ly = lx / m_l, ly / m_l
        # AND THE HOLDER CAN LEAN. A person shifts their own weight back
        # against what they carry, and how far back is bounded by how big they
        # are — their own half-width, read off the body, not a number typed in.
        # Without it the sum says nobody may hold anything at arm's length,
        # which is plainly false; with it, whether a load takes you over is
        # its weight and how far out it hangs against YOUR weight and YOUR
        # build, which is the real answer and is different for every pair.
        lean = 0.5 * self._span_xy(own)
        out = [0.0, 0.0]
        for ax, (lo, hi, l_, s_) in enumerate(
                ((int(foot[:, 0].min()), int(foot[:, 0].max()), lx, sx),
                 (int(foot[:, 1].min()), int(foot[:, 1].max()), ly, sy))):
            if l_ > hi:
                arm, back, sign = l_ - hi, (hi - s_) + lean, 1.0
            elif l_ < lo:
                arm, back, sign = lo - l_, (s_ - lo) + lean, -1.0
            else:
                continue                      # the load is over the feet
            if m_l * arm > m_s * max(back, 0.0):
                out[ax] = sign * arm
        return None if out == [0.0, 0.0] else tuple(out)

    def _leave(self, p, cells):
        """Out means OUT. A body that reached the door walks through it and off
        the lattice, the way gas leaves at an open sky — its mass is accounted
        as having left the world, not destroyed inside it.

        This is not tidiness. Bodies have real depth now, so one person standing
        in a doorway they had already escaped through was a wall to everyone
        behind them: measured, a second body stuck three voxels short of the
        exit for 800 ticks because the first was still stood in it."""
        idx = tuple(np.asarray(cells).T)
        self.left_mass += float(self.smass[idx].sum())
        # AND IT GOES ON THE TALLY, BY MATERIAL. `left_mass` is one untyped
        # number that nothing reads; `self.gone` is what `total_mass`,
        # `total_energy` and `total_lost` actually count, and walking out of a
        # door never touched it. So every escape quietly took a body's worth of
        # FLESH out of the conservation check and its body heat out of the
        # energy one — which is the exact failure `total_mass` warns about in
        # its own docstring, a check that is right for one material and wrong
        # for the rest. A man leaving by the door is the same event as a voxel
        # going past the edge, so it is recorded the same way.
        g = self.gone
        mats = self.mat[idx]
        kg = self.smass[idx]
        for m in np.unique(mats):
            g[int(m)] = g.get(int(m), 0.0) + float(kg[mats == m].sum())
        g["E"] = g.get("E", 0.0) + float(self.E[idx].sum())
        vols, fls = self.fvol[idx], self.fl[idx]
        wet = vols > 0
        for f in np.unique(fls[wet]) if wet.any() else ():
            key = ("fluid", int(f))
            g[key] = g.get(key, 0.0) + float(vols[wet & (fls == f)].sum())
        # and any thing that went with it is gone from the world, which the
        # identity pass hears about here or not at all (`_left_world`)
        eids = self.eid[idx]
        for e in np.unique(eids[eids != 0]):
            self._id_losses.append((int(e), "gone"))
        self._lift(idx)                       # and the cells are empty
        p["_own"] = frozenset()
        p["_claim_mask"], p["_claim_tick"] = None, None
        self._slack_mat = None
        self._torque_solid = None

    def _plan_path(self, cells, ex, p=None):
        """Shortest route from where this body is to one goal.
        Deterministic, 8-connected, replanned when the world changes underfoot."""
        return self._routes(cells, [ex], p)[(int(ex[0]), int(ex[1]))]

    def _gait(self, p):
        """How this body is moving its legs: "run", "walk" or "creep".

        It RUNS when it is fleeing: the will layer sets `fleeing` when a walk
        is begun by a percept (a fire seen, a shout of "Fire!", smoke, a
        runner, a fall) or by a linked mind that says it is running from
        something. A noise only turns the head (`ORIENTING` asks the eyes
        alone), so nobody breaks into a run to look at a crackle. And what
        others see as "someone bolting past" is now a body that really is
        going faster. It CREEPS when it is moving quietly on purpose.
        Otherwise it walks."""
        if p.get("fleeing"):
            return "run"
        return "creep" if p.get("sneak") else "walk"

    def _pace(self, p, haul=0.0):
        """Ticks this body takes to go one voxel along a grid axis, now.

        The speed is set in metres a second (`WILL` walk_mps and run_mps; a
        creep is a walk slowed by `SOUND["sneak_slow"]`) and turned into ticks
        here, with the size of THIS world's voxel: so the same body crosses
        the same room in the same time at 5 cm and at 10 cm. A walk at 5 cm
        is 1.43 ticks a voxel, and a run is 0.8 — more than a voxel a tick.

        WHAT YOU DRAG, YOU PAY FOR. A man hauling an unconscious body walked at
        exactly the pace of a man carrying nothing — the force arithmetic said
        the haul was legal and then charged him nothing for it, which made
        rescuing someone free and dragging a crate the same act as strolling.

        A body has only so much to put out. What the load takes (`haul`, in
        newtons), the legs do not get, and pace goes with what is left:
        hauling nothing is the pace it always was, and at the very limit of
        what a man can shift he barely moves — which is what being at your
        limit IS. The floor of 5% is not a fudge for the model, it is a fence
        against dividing by nothing."""
        gait = self._gait(p)
        mps = WILL["run_mps"] if gait == "run" else WILL["walk_mps"]
        if gait == "creep":
            mps /= SOUND["sneak_slow"]
        full = self._strength(p)
        return (0.1 * self.scale / (mps * TICK_S)
                * full / max(full - haul, full * 0.05))

    def _stride(self, p, cells, ex, ax_, ay_):
        """One tick of walking toward `ex`: every voxel the pace has paid for,
        each one planned, squared up to, stepped and heard.

        `cells` is the body's flesh and (`ax_`, `ay_`) its centre, as the will
        layer found them this tick."""
        haul = self._haul_cost(p, cells)
        pace = self._pace(p, haul)
        p["_pace"] = pace
        # STARTS ONE SHORT, so that the very first tick completes a full
        # stride and the cadence of a body carrying nothing is exactly the
        # cadence it always had — 0, 3, 6, 9 and not 0, 2, 5, 8. That is
        # the property that made it safe to replace the modulo at all, and
        # being one out shifted every unladen body by a voxel.
        p["_stride"] = p.get("_stride", pace - 1.0) + 1.0
        # AS MANY STEPS AS THE TICK HAS PAID FOR. One step a tick at most
        # capped every body at a voxel a tick, which is 2 m/s at 5 cm: under
        # the speed people change from a walk to a run, so nobody could ever
        # run. Each voxel stepped is paid for out of the stride, and what is
        # left over is kept for the next tick.
        moved = False
        stepping = any(self._step_range(p))
        # how far a stride that rises or drops may reach: a foot and a half
        # (25 cm), at least two cells
        lunge = max(2, int(round(0.25 / (0.1 * self.scale))))
        while p["_stride"] >= pace:
            p["_stride"] -= pace
            # ONLY ON A FLOOR. A second step in the same tick is not a step
            # off the edge the first one reached: the body that has nothing
            # under it is the footing law's now, not the legs'.
            if moved and not self._underfoot(cells):
                return
            # walking is PLANNED, not greedy: legs that jam forever on the
            # first baffle aren't legs (greedy + wall-slide both livelocked,
            # measured). A shortest route over stand-able columns is the
            # body's motor competence — knowing the way around the table is
            # not thinking, any more than balance is.
            # WHAT IS CARRIED WALKS WITH THE BODY. A thing hanging from the
            # fist is one kinematic unit with the hand that holds it — walked
            # as separate matter it is a wall in front of its own carrier, and
            # the carrier jams on it forever (and the planner jams first, so
            # the carried cells count as the body's own there too). A held
            # thing still ON the floor (too heavy to lift) is not carried; it
            # TRAILS, like a dragged person does.
            obj = self._held_cells(p) if p.get("held") else None
            carried = obj is not None and not self._underfoot(obj)
            walk_cells = np.concatenate([cells, obj]) if carried else cells
            if (not p.get("_path")) or self.tick - p.get("_planned", -99) > 30:
                p["_path"] = self._plan_path(walk_cells, ex, p)
                p["_planned"] = self.tick
            path = p.get("_path") or []
            while path and max(abs(path[0][0] - ax_), abs(path[0][1] - ay_)) < 1.0:
                path.pop(0)
            if not path:
                return
            sx = int(np.sign(path[0][0] - ax_))
            sy = int(np.sign(path[0][1] - ay_))
            # YOU TURN TO WALK. A body squares up to where its next step goes
            # before it takes it, and the turn is that stride. Only when it
            # walks: a body standing still keeps whatever way the scene built
            # it, which is also how a scene says which way someone faces.
            ahead = (sx, 0) if (abs(path[0][0] - ax_) >= abs(path[0][1] - ay_)
                                and sx) or not sy else (0, sy)
            if ahead != (0, 0) and not carried:
                was = p.get("facing")
                p["facing"] = (float(ahead[0]), float(ahead[1]))
                if self._square_up(p):
                    return
                p["facing"] = was if was is not None else p["facing"]
            steps = ((sx, 0), (0, sy)) if abs(path[0][0] - ax_) >= \
                abs(path[0][1] - ay_) else ((0, sy), (sx, 0))
            # A DIAGONAL ON THE ROUTE IS ONE STEP, the length of a voxel's
            # diagonal, and it is paid for as that: sqrt(2) of a stride. Taken
            # as two grid steps, a body on a 45 degree route covered the ground
            # at 0.71 of its pace. Which cell is next is read from the column
            # the route starts at (the rounded centre), not from the centre
            # itself, which a lopsided body puts off the grid.
            gx = path[0][0] - int(round(ax_))
            gy = path[0][1] - int(round(ay_))
            if gx and gy:
                steps = ((int(np.sign(gx)), int(np.sign(gy))),) + steps
            for step in steps:
                if step == (0, 0):
                    continue
                # THE LEGS HAVE A STEP RANGE (T11, wave 13c): a body that can
                # step up or down a few centimetres takes its step by
                # `_step_to`, which tries a flat step first and then the
                # smallest rise that works. With the range at 0 it is the old
                # flat `_walk`, exactly.
                kk = 1
                if stepping:
                    got = self._step_to_k(walk_cells, step[0], step[1], p,
                                          lunge)
                    done = got is not None
                    dz, kk = got if done else (0, 1)
                else:
                    dz = 0
                    done = self._walk(walk_cells, *step)
                if done:
                    # a step with a rise may reach several cells (see
                    # `_step_to_k`): everything below moves by the whole of it
                    sk = (step[0] * kk, step[1] * kk)
                    # A STEP IS A SMALL LANDING, and it makes the noise one
                    # does: the body coming down a centimetre, through flesh,
                    # onto whatever the floor is. As hard as the body is going:
                    # the heel strikes at the walking speed squared.
                    zlo = int(walk_cells[:, 2].min()) + dz
                    fx_, fy_ = (int(round(ax_)) + sk[0],
                                int(round(ay_)) + sk[1])
                    stepped = self.emit("steps", actor=p["pid"],
                                        place=(float(fx_), float(fy_),
                                               float(zlo)),
                                        site="will:steps")
                    walk = 0.1 * self.scale / (WILL["walk_mps"] * TICK_S)
                    if zlo > 0 and 0 <= fx_ < self.shape[0] \
                            and 0 <= fy_ < self.shape[1]:
                        kg = float(self.smass[tuple(walk_cells.T)].sum()) / 1000.0
                        quick = walk / max(pace, 1e-9)
                        self._noise((fx_, fy_, zlo), kg * GRAVITY
                                    * SOUND["step_drop_m"] * quick * quick,
                                    [FLESH, int(self.mat[fx_, fy_, zlo - 1])],
                                    who=p["name"], ev=stepped.id)
                    if self.contact_laws and zlo > 0:
                        self._foot_contact(p, fx_, fy_, zlo, step,
                                           walk / max(pace, 1e-9), walk_cells,
                                           stepped)
                    self._carry_own(p, sk[0], sk[1], dz)
                    if step[0] and step[1]:
                        p["facing"] = (float(ahead[0]), float(ahead[1]))
                        p["_stride"] -= (np.sqrt(2.0) - 1.0) * pace * kk
                    else:
                        p["facing"] = (float(step[0]), float(step[1]))
                    if kk > 1:                 # the extra cells are paid for
                        p["_stride"] -= (kk - 1) * pace
                    if dz:
                        # A RISE COSTS TIME AND WORK: the move takes
                        # max(d / v_walk, h / vz_max) seconds against a flat
                        # cell's d / v_walk, and the extra is taken out of the
                        # stride; the work and the heat of a descent are booked.
                        t_dz = self._step_books(
                            p, walk_cells + np.array([sk[0], sk[1], dz],
                                                     dtype=walk_cells.dtype),
                            sk[0], sk[1], dz)
                        t_flat = (0.1 * self.scale * float(np.hypot(*sk))
                                  / stairs.V_WALK)
                        p["_stride"] -= max(t_dz / t_flat - 1.0, 0.0) * pace
                    if carried:
                        c = p["held"]["cell"]
                        p["held"]["cell"] = (c[0] + sk[0],
                                             c[1] + sk[1], c[2] + dz)
                    self._haul(p, *sk)
                    if not carried:
                        self._haul_held(p)
                    if haul > 0.0:
                        self._rub(p, cells)    # friction becomes heat, and heat
                                               # is what combustion reads
                    cells = cells + np.array([sk[0], sk[1], dz],
                                             dtype=cells.dtype)
                    ax_, ay_ = ax_ + sk[0], ay_ + sk[1]
                    moved = True
                    break
            else:
                p["_path"] = None                    # blocked mid-route: replan
                # AND LEARN IT. A body that remembers a way as clear will plan
                # the same way again the moment it replans, and jam against the
                # thing in it forever. Walking into something is a percept: you
                # find out the corridor is shut by shutting your shin in it.
                free = p.get("free")
                if free is not None:
                    for st in ((sx, 0), (0, sy)):
                        if st == (0, 0):
                            continue
                        bx = int(round(ax_)) + st[0]
                        by = int(round(ay_)) + st[1]
                        if 0 <= bx < free.shape[0] and 0 <= by < free.shape[1]:
                            free[bx, by] = False
                return

    # CONTACT EVENTS AND THE SLIP LAW (T3, wave 13c C13). The gate is a plain
    # attribute of the world: False gives the old world bit for bit. It composes
    # with every other capability (it only reads the layer pool and the lattice).
    contact_laws = True

    def _foot_contact(self, p, fx, fy, zlo, step, speed_ratio, walk_cells, ev):
        """A foot has come down at (fx, fy) on the floor under cell z = zlo: the
        footfall contact (layers move between sole and floor) and the slip law
        (the friction the step needs against what the floor gives)."""
        if not (0 <= fx < self.shape[0] and 0 <= fy < self.shape[1]):
            return
        kg = float(self.smass[tuple(walk_cells.T)].sum()) / 1000.0
        contact.footfall(self, p, fx, fy, zlo, step, kg,
                         cause=ev.id if ev is not None else None)
        mu = contact.floor_mu(self, fx, fy, zlo)
        if mu is None:
            return
        v = contact.slip_speed(mu, speed_ratio)
        if v > 0.0:
            n = float(np.hypot(*step)) or 1.0
            self.push_person(p, -step[0] / n * v, -step[1] / n * v)
            p["_slips"] = p.get("_slips", 0) + 1
            self.emit("touches", actor=p["pid"], result="slip",
                      place=(float(fx), float(fy), float(zlo)),
                      cause=ev.id if ev is not None else None,
                      data=(("mu", round(float(mu), 4)), ("v_mps", round(v, 4))),
                      site="contact:slip")

    def _walk(self, cells, sx, sy):
        """Translate a body one voxel sideways, SHOVING ASIDE what is light
        enough to shove.

        A step needs its arriving cells to be sparse enough to push through
        (the same fill read the route planner uses — legs and plans must agree,
        or the planner routes through a hedge the legs then refuse). Anything
        packed, and any real puddle, still stops the step.

        What gets shoved has to GO somewhere: the branches the body enters are
        moved into the cells the body leaves, so the hedge closes behind you and
        no leaf is quietly deleted. A rigid step vacates exactly as many cells as
        it enters, so the exchange always balances."""
        tgt = cells.copy()
        tgt[:, 0] += sx
        tgt[:, 1] += sy
        nx, ny, nz = self.shape
        if (tgt[:, 0] < 0).any() or (tgt[:, 0] >= nx).any() \
                or (tgt[:, 1] < 0).any() or (tgt[:, 1] >= ny).any():
            return False
        own = set(map(tuple, cells))
        new = set(map(tuple, tgt))
        entered = [c for c in map(tuple, tgt) if c not in own]
        # NOR THROUGH WHAT THE SOLVER HOLDS. A `moving` cell is particle
        # matter drawn for one tick; shoved into the cells behind, it would
        # stay there as lattice matter while the solver drew the same
        # particles again (`_objects_within_reach` says more). The walker's
        # own cells are checked too, for a carried thing.
        if self.moving[tuple(cells.T)].any() \
                or any(self.moving[t] for t in entered):
            return False
        # A FILM OF WATER IS NOT A WALL. A step refused any cell with more than
        # a millilitre in it, so a strip 2 mm deep (5 ml a cell) stopped a
        # walker dead for five hundred ticks, and a bucket thrown at a fire
        # could trap the thrower. Liquid is shoved aside like a hedge: it goes
        # into the cells the body leaves (below), so none is lost. Only water
        # past half a cell deep (the "flooded" line the breath law uses) is a
        # pool a walker cannot wade; drag against it comes later.
        for t in entered:
            packed = self.smass[t] / max(_DENS_ARR[self.mat[t]] * self.vox_l, 1e-9)
            if packed >= PUSH_THROUGH or self.fvol[t] > WADE_CELLS * self.cap:
                return False
        vacated = [c for c in map(tuple, cells) if c not in new]
        # ALL OF IT MOVES (`CARRIED`): a step once left the drop height and
        # the broken-piece flag behind, so a shard shoved aside became whole
        # glass where it landed.
        ent = tuple(np.array(entered, np.int64).reshape(-1, 3).T)
        vac = tuple(np.array(vacated, np.int64).reshape(-1, 3).T)
        body = self._lift(tuple(cells.T))                 # the walker's, and
        shoved = self._lift(ent)                          # what it shoves
        self._put(tuple(tgt.T), body)                     # the body arrives
        self._put(vac, shoved)                            # the hedge closes behind
        self._torque_solid = None
        return True

    def _law_footing(self):
        """A PERSON IS A RIGID THING, so a person goes over as ONE thing.

        Support is relaxed column by column, which is exactly right for a wall
        and quite wrong for a man. Walked off his own ledge, he came down as
        five separate showers of flesh — every gram still there, the person
        gone — and hooked by a forearm on the lip he came apart the other way,
        the arm staying at z31 while his legs ran down to the floor like sand.
        A jumper never did either, and the reason is that a leap PROMOTES the
        body to a free rigid object first. That promotion was the piece missing
        everywhere else.

        The question is the FEET: a body whose lowest voxels have nothing under
        them is no longer standing on anything. Deliberately not the torque
        law's fuller test of weight-over-footprint, which reads a body already
        lying in a heap as hopelessly overbalanced — most of a slumped body
        rests on the rest of itself — and launched it into the ground it was
        already on, over and over.

        Going over an edge is a rotation about that edge, so a body leaves it
        moving OUTWARD as well as down, and by the time it has turned far
        enough to clear the lip its weight has fallen through about the
        overhang: that is where the sideways speed comes from, and it is the
        only reason the thing still touching is not struck at once. Without it,
        a man tipped off a lip with a forearm over the stone dropped one voxel,
        hit the very rock his arm had been on, and was set back down in the
        same place — three times a second, for ever.

        Nothing here is about ledges. A shove, a haul, a floor that burns out
        from underneath all arrive at the same test, and all pay the same
        1/2 m v² on landing.

        AND A BODY IS NOT ONLY ITS OWN WEIGHT. What it holds up rides on the
        same two feet, so a man still firmly stood on rock can be taken over
        by the one hanging from his arm — see _pulled_over. That is the other
        half of a grip being an edge in the support graph: the load hangs from
        the holder, and the holder answers for it."""
        if not self.persons:
            return
        aloft = {b.get("owner") for b in self.bodies} | self.matter.aloft_owners()
        holders = {q["dragging"]: q for q in self.persons if q.get("dragging")}
        for p in self.persons:
            landed_v = p.pop("_land_v", None)    # (set by a landing this tick)
            if p["name"] in aloft and p.get("_reflex"):
                # IN THE AIR THE FOOT IS OVER NOTHING NO LONGER: what the
                # reflex was counting down to is over. A count that went on
                # through a leap or a fall would answer the moment he lands,
                # with no delay at all; he gets a fresh one.
                self._reflex_done(p, "balance")
                p.pop("_bal_plan", None)
            if not p["alive"] or p["safe"] or p["name"] in aloft:
                continue
            comp, sl = self._person_cells(p)
            if comp is None or not comp.any():
                continue
            cells = self._claim_cells(p, comp, sl)
            over, why = None, "loses their footing"
            # A BODY THAT IS BEING HAULED KEEPS THE SPEED IT IS HAULED AT: the
            # floor-plane position of the last few ticks is kept (for hauled
            # bodies only) so that, if the haul takes it over an edge, it goes
            # over with that speed and not from rest (`_tip_over`).
            if p["name"] in holders:
                hist = p.setdefault("_xy_hist", [])
                hist.append((self.tick, float(cells[:, 0].mean()),
                             float(cells[:, 1].mean())))
                del hist[:-HAUL_TICKS]
            elif "_xy_hist" in p:
                del p["_xy_hist"]
            standing = self._underfoot(cells)
            if standing:
                # STOOD ON SOMETHING — but a body is not only its own weight.
                # What it holds up rides on the same two feet, and a man
                # dangling from an arm hangs well beyond the toes.
                over = self._pulled_over(p, cells)
                why = "is pulled off their feet by what they are holding"
                if over is None:
                    # AND A BODY CAN OVERBALANCE ALL BY ITSELF. Balance was
                    # only ever asked about what a man was CARRYING, so a man
                    # carrying nothing could not fall over however he stood —
                    # which nobody noticed while a body was one rigid block,
                    # because a block that stands up straight has its weight
                    # over its feet by construction. Give it a waist and the
                    # hole opens: measured, a man bent 46 degrees with his head
                    # fourteen voxels past his toes stood there indefinitely.
                    over = self._overbalanced(p, cells)
                    if over is not None and landed_v is not None:
                        # He came down with this speed, onto a place that
                        # does not hold his weight: it goes into the turn
                        # he makes (a landing that leaves him balanced
                        # spends it, as it always did).
                        p["_v"] = landed_v
                        over = self._overbalanced(p, cells) or over
                    if over is None:
                        cr = p.get("_cr")
                        if cr is not None and cr[0] == self.tick and \
                                self.body_contact and not p.get("_cr_ev"):
                            # THE BODY KEPT ITS FEET WITHOUT A STEP (Hof's
                            # counter-rotation, a hand on a wall): written
                            # down as the reflex's own act
                            p["_cr_ev"] = True
                            self._reflex_fire(
                                p, "balance", "stands",
                                f"{p['name']} flings out their arms and keeps their feet"
                                if len(cr[1]) == 1 else
                                f"{p['name']} puts a hand on {cr[1][-1][8:]} and keeps their feet",
                                result="steadied", how="+".join(cr[1]))
                        if p.get("_reflex") or p.get("_v"):
                            self._steadied(p, cells)
                        continue
                    why = "overbalances"
                    # LEANING ON WHAT STOPPED HIS LAST FALL IS NOT GOING OVER
                    # AGAIN. A fall that the room cut short (a wall, a table)
                    # leaves him propped (`_land_body`), and nothing about his
                    # weight has changed until he moves: tipping him again
                    # every few ticks was a man in the air for ever.
                    #
                    # ONLY THE WAY IT STOPPED HIM. What stopped the fall is
                    # on that side; his weight going over the OTHER way (a
                    # man stood back up on another man's shoulder, past the
                    # shoulder's far edge) is held by nothing, and he goes.
                    way = p.get("_propped_way")
                    if p.get("_propped") == tuple(
                            int(v) for v in cells.min(axis=0)) and (
                            way is None or over[way[0]] * way[1] > 0.0):
                        continue
                    # THE REFLEX FIRST: a body that is over its base does not
                    # simply go over. Unload the foot that is over nothing,
                    # step to firm ground, take hold of a hold, else fall
                    # (`_balance_reflex`).
                    if self._balance_reflex(p, cells, over):
                        continue
                # AND THE ARM COMES DOWN — all the way, at once. You cannot
                # keep a thing at arm's length while it is taking you over: the
                # reach is the first thing to go, which shortens the lever,
                # which is why a man goes over ONCE instead of being thrown off
                # his feet again on every tick he spends back on them.
                #
                # ALL THE WAY, and not a tick at a time. Holding a limb out is
                # work a standing body does, and a body going over has stopped
                # doing it — the arm falls, it does not unwind at a shoulder's
                # pace. Dropping only the INTENTION left `_law_pose` to walk the
                # angle home, and since the flesh waits at every angle the
                # lattice cannot draw, the man landed with the weight still out
                # and went over a second time, and a third.
                p["reach"] = {}
                came_down = False
                for limb in list(p.get("pose") or {}):
                    if self._repose(p, limb, 0.0) == "moved":
                        (p["pose"]).pop(limb, None)
                        (p.get("drawn") or {}).pop(limb, None)
                        came_down = True
                if came_down:
                    # AND HE IS NOT WHERE HE WAS A MOMENT AGO. `cells` was
                    # read before the arm came down, so it named the cells
                    # the arm had just left (air now) and not the ones it
                    # went to: he went over and his arm stayed on the floor,
                    # a man and an arm apart until the arm was swung again
                    # (fuzzer seed 6: 44 cells of him left behind at t40,
                    # [342, 18] at t43). Read him again.
                    comp, sl = self._person_cells(p)
                    if comp is None or not comp.any():
                        continue
                    cells = self._claim_cells(p, comp, sl)
            else:
                h = holders.get(p["name"])
                if h is not None and self._grip_holds(h, cells):
                    continue                  # he HANGS from the grip: a hand
                                              # that can lift him can hold him
                # ...but is anything still touching? Only what BEARS him
                # counts, and what he carries is not what carries him: a
                # hand over a block he holds is not a ledge (fuzzer seed 119
                # was set down on the thing in his hand every third tick).
                foot, _grips = self._footing(p, cells)
                if len(foot):
                    kg = self.smass[tuple(cells.T)]
                    tot = max(float(kg.sum()), 1e-9)
                    over = (float((cells[:, 0] * kg).sum()) / tot
                            - float(foot[:, 0].mean()),
                            float((cells[:, 1] * kg).sum()) / tot
                            - float(foot[:, 1].mean()))
            body = None
            if over is not None and standing:
                # GOING OVER IS A TURN ABOUT THE EDGE OF THE BASE, under real
                # g (`_tip`), not a hop: the weight swings out over what
                # holds it up and the body arrives where a rigid thing of its
                # inertia would.
                body = self._tip_over(p, cells, over)
                if body is False:
                    continue                  # wedged against the world
            if body is None:
                # NOT STANDING ON ANYTHING (his soles are over air; only a
                # forearm is on the lip): he cannot turn about the lip as a
                # rigid body, because the rest of him hangs against the face
                # of it. He goes off it moving outward as well as down, the
                # speed a rotation about that edge would have given him by
                # the time he cleared it (v^2 = 2 g overhang): without it he
                # dropped a voxel onto the rock his arm had been on and was
                # set back down in the same place, three times a second.
                vel = [0.0, 0.0, 0.0]
                if over is not None:
                    ox, oy = over
                    axis, o = (0, ox) if abs(ox) >= abs(oy) else (1, oy)
                    vel[axis] = float(np.sign(o) * np.sqrt(
                        2.0 * GRAVITY * abs(o) * 0.1 * self.scale))
                body = self._launch(cells, vel, owner=p["name"])
            falls_ev = self.emit("falls", actor=p["pid"], why=why,
                                  text=f"{p['name']} {why}",
                                  to=p, site="footing:falls")
            # SO A BLOW THIS BODY LANDS CAN BE BLAMED ON THE FALL, not on the
            # faller -- the attribution rule for a whole-body fall (3.4):
            # `hits` gets actor None, the faller as instrument, and this as
            # cause.
            body["falls_ev"] = falls_ev.id

    # ── the balance reflex (the contract is in reflex.py) ───────────────────
    def _steadied(self, p, own):
        """The body is over its base again (or never was): the reflex that was
        counting down stops, a plan made for it is dropped, and the speed a
        push gave it dies away at the rate an upright body steadies itself
        (w0 = sqrt(g / h) a second, h the height of its centre of mass)."""
        self._reflex_done(p, "balance")
        p.pop("_bal_plan", None)
        v = p.get("_v")
        if v:
            com, _m = self._com_of(own)
            h = max((float(com[2]) - float(own[:, 2].min()) + 0.5)
                    * 0.1 * self.scale, 0.25) if com is not None else 0.7
            k = float(np.exp(-np.sqrt(GRAVITY / h) * TICK_S))
            v0 = v
            v = (v[0] * k, v[1] * k)
            if abs(v[0]) + abs(v[1]) < 0.01:
                self._book_foot_loss(p, own, v0, (0.0, 0.0))
                p.pop("_v", None)
                p.pop("_cr_ev", None)
                p.pop("_v_src", None)
            else:
                self._book_foot_loss(p, own, v0, v)
                p["_v"] = v

    def _balance_reflex(self, p, own, over):
        """THE BALANCE REFLEX. This body's weight is past its base (`over`).
        It does not go over yet: the nervous system answers first, after the
        person's own delay, with the first of these that can work.

        1. UNLOAD AND STEP. A foot that is over nothing carries no load (the
           base is only the firm contacts, `_footing`), so the weight is
           already off it. The body steps, a few cells, to the nearest place
           where it stands firm (`_balance_plan`), against the push if it was
           pushed.
        2. GRAB. Failing a step, take hold of a firm hold within arm's reach
           that is resting on the world and heavy enough not to come with
           you; a hand on it widens the base to the hand.
        3. FALL. Failing both, or when the reflex is off (the person is out,
           or chose the fall: `_reflex_off`), the body goes over.

        Returns True while the body is being held up (waiting out its delay,
        or steadied), False when it is to go over NOW. A body that nothing can
        save does not wait for a delay that would buy it nothing: it falls on
        the tick it is found over."""
        gate = self._reflex_gate(p, "balance")
        if gate == "off":
            return False
        plan = p.get("_bal_plan")
        if plan is None:
            plan = self._balance_plan(p, own, over)
            if plan is None:
                self._reflex_done(p, "balance")
                return False
            p["_bal_plan"] = plan
        if gate == "wait":
            return True
        # the delay has passed: do it
        p.pop("_bal_plan", None)
        self._reflex_done(p, "balance")
        plan = self._balance_plan(p, own, over)    # the world may have moved
        if plan is None:
            return False
        if plan[0] == "grab":
            return self._balance_grab(p, own, plan[1])
        if not self._step_grips(p, own, plan[1], over):
            # THE FOOT WOULD SLIDE OUT (wave 14b H): the floor under the step
            # gives less friction than the step needs, so it buys nothing. A
            # hold within reach can still be taken; else the body goes over.
            hold = self._hold_target(p, own)
            if hold is not None:
                return self._balance_grab(p, own, hold)
            self._reflex_fire(p, "balance", "touches",
                              f"{p['name']}'s foot slides out from under them",
                              result="slip")
            return False
        return self._balance_step(p, own, plan[1])

    def _step_grips(self, p, own, shift, over):
        """Does the floor under this recovery step give the friction it needs
        (`contact.step_holds`)? The pendulum is the body's centre of mass over
        the soles; `over` is how far (in cells) its extrapolated centre is past
        the base, the lean the step must undo. Gated by `contact_laws` like
        the slip law: off, every step grips, as before."""
        if not self.contact_laws:
            return True
        zlo = int(own[:, 2].min())
        sole = own[own[:, 2] == zlo]
        dx, dy = int(shift[0]), int(shift[1])
        vox_m = 0.1 * self.scale
        mus = [contact.floor_mu(self, int(c[0]) + dx, int(c[1]) + dy, zlo)
               for c in sole[:: max(len(sole) // 6, 1)]]
        mus = [m for m in mus if m is not None]
        if not mus:
            return True
        com, _kg = self._com_of(own)
        cz = float(com[2]) if com is not None else float(own[:, 2].mean())
        h = (cz - zlo + 0.5) * vox_m
        # the foot is placed where it can undo the lean AND the step it takes:
        # the lean is the part past the base, and a step is never shorter
        past = float(np.hypot(over[0], over[1])) * vox_m
        return contact.step_holds(min(mus), past, h)

    def _balance_plan(self, p, own, over):
        """What the reflex would do for this body now: ("step", (dx, dy)),
        ("grab", hold row) or None."""
        # (a man somebody has hold of cannot step away from them: whether the
        # grip holds him is the hold's question, `_haul`'s, not the leg's)
        if not (p.get("held") or p.get("dragging")
                or any(q.get("dragging") == p["name"] for q in self.persons)):
            sh = self._step_target(p, own, over)
            if sh is not None:
                return "step", sh
        hold = self._hold_target(p, own)
        if hold is not None:
            return "grab", hold
        return None

    def _step_target(self, p, own, over):
        """The nearest step (dx, dy), in cells and no longer than this
        person's own reach, after which the body stands firm: it fits there,
        and `_overbalanced` has nothing to say of it AT REST (the step stops
        the speed a push gave). A pushed body must step at least as far as
        its extrapolated centre went past the base, in the way it went. Ties
        go to the step that leads away from the drop."""
        vox_m = 0.1 * self.scale
        R = max(int(self.reflex_reach_m(p, "balance") / vox_m), 2)
        ghost = dict(p, held=None, dragging=None, _v=None, _ghost_from=own,
                     _ghost_kg=self.smass[tuple(own.T)].copy())
        ox, oy = over
        n_o = float(np.hypot(ox, oy))
        pushed = bool(p.get("_v"))
        need = int(np.ceil(n_o)) if pushed else 0
        zlo = int(own[:, 2].min())
        nx, ny, nz = self.shape
        g = np.mgrid[-R:R + 1, -R:R + 1].reshape(2, -1).T
        d2 = (g * g).sum(axis=1)
        g = g[d2 <= R * R]
        d2 = d2[d2 <= R * R]
        # nearest first; at the same distance, away from the drop
        away = -(g[:, 0] * ox + g[:, 1] * oy) / max(n_o, 1e-9) if not pushed \
            else (g[:, 0] * ox + g[:, 1] * oy) / max(n_o, 1e-9)
        order = np.lexsort((-away, d2))
        mine = set(map(tuple, own.tolist()))
        # the soles: the lowest layer. A step is worth looking at only if
        # every sole ends up over something.
        sole = own[own[:, 2] == zlo][:, :2]
        if zlo > 0:
            for k in order:
                dx, dy = int(g[k, 0]), int(g[k, 1])
                if dx == 0 and dy == 0:
                    continue
                if pushed and dx * ox + dy * oy < need * n_o - 1e-9:
                    continue
                tx, ty = sole[:, 0] + dx, sole[:, 1] + dy
                if ((tx < 0) | (tx >= nx) | (ty < 0) | (ty >= ny)).any():
                    continue
                if (self.mat[tx, ty, zlo - 1] == AIR).any():
                    continue                    # a foot over nothing
                tgt = own + np.array([dx, dy, 0])
                if not self._fits(tgt, mine):
                    continue
                if self._overbalanced(ghost, tgt) is not None:
                    continue
                return dx, dy
        return None

    def _fits(self, tgt, mine):
        """Could a body stand in these cells (a body that is `mine` now): in
        the world, and every cell either its own or sparse enough to shove
        through?"""
        nx, ny, nz = self.shape
        if ((tgt[:, 0] < 0) | (tgt[:, 0] >= nx) | (tgt[:, 1] < 0)
                | (tgt[:, 1] >= ny) | (tgt[:, 2] < 0) | (tgt[:, 2] >= nz)).any():
            return False
        at = (tgt[:, 0], tgt[:, 1], tgt[:, 2])
        packed = self.smass[at] / np.maximum(
            _DENS_ARR[self.mat[at]] * self.vox_l, 1e-9)
        bad = packed >= PUSH_THROUGH
        if bad.any() and not all(tuple(c) in mine for c in tgt[bad].tolist()):
            return False
        return not self.moving[at].any()

    def _balance_step(self, p, own, shift):
        """Take the step the plan chose: the body goes there a cell at a time,
        as a walk does (so what is in the way is shoved or refuses), and the
        speed a push gave it is spent. The step is an event the mind
        perceives. Returns True if he is on his feet, False if the step was
        refused at its first cell and he must go over."""
        cells = np.asarray(own)
        dx, dy = shift
        moved = 0
        for ax, k in ((0, dx), (1, dy)):
            st = [0, 0]
            st[ax] = int(np.sign(k))
            for _ in range(abs(int(k))):
                if not self._walk(cells, *st):
                    break
                self._carry_own(p, *st)
                cells = cells + np.array([st[0], st[1], 0], dtype=cells.dtype)
                moved += 1
        v_was = p.pop("_v", None)
        if v_was and self.body_contact:
            self._book_foot_loss(p, cells, v_was, (0.0, 0.0))    # the step spends it
        p.pop("_cr_ev", None)
        p.pop("_v_src", None)
        if not moved:
            return False
        p["_claim_tick"] = None
        self._reflex_fire(p, "balance", "stands",
                          f"{p['name']} catches their balance with a step",
                          result="steadied", step=moved,
                          place=tuple(float(c) for c in cells.mean(axis=0)))
        return True

    def _hold_target(self, p, own):
        """A firm hold within arm's reach: the nearest thing that is resting
        on the world, heavy enough that the person cannot pull it over (half
        their own weight), at a height a hand can close on, and that a hand
        on it would make the body stand. Returns the thing's row (and the
        arm that would take it) or None."""
        if p.get("held") or p.get("dragging"):
            return None
        zlo, zhi = int(own[:, 2].min()), int(own[:, 2].max())
        com, w_me = self._com_of(own)
        foot = self._firm(own)
        if com is None or not len(foot):
            return None
        for row in sorted(self._objects_within_reach(p, own),
                          key=lambda r: r["away"])[:4]:
            cl = self._object_at(*row["cell"])
            if not len(cl) or cl[:, 2].max() < zlo + 0.2 * (zhi - zlo):
                continue
            if not len(self._contact(cl, ignore=own)):
                continue                      # not resting on the world
            if float(self.smass[tuple(cl.T)].sum()) / 1000.0 < 0.5 * w_me:
                continue
            arm = self._hand(p, own, toward=row["cell"], free=True)
            if arm is None:
                continue
            # would a hand on it put the weight over the base? (asked of the
            # hull, so that asking takes nothing: a grip is only made if it
            # is used.) The hand is on the hold, at its nearest cell.
            fists = np.asarray(self._fists(p, own), np.float64)
            d2 = ((cl[None, :, :].astype(np.float64)
                   - fists[:, None, :]) ** 2).sum(-1)
            g = cl[int(d2.min(0).argmin())]
            hull = self._hull_of(foot[:, :2].tolist() + [[int(g[0]), int(g[1])]])
            x, y = self._xcom(p, com, float(foot[:, 2].min()))
            if self._past_hull(hull, x, y) == (0.0, 0.0):
                return dict(row, arm=arm)
        return None

    def _balance_grab(self, p, own, row):
        """Close a hand on the hold the plan chose."""
        cell = row["cell"]
        if int(self.mat[cell]) != row["mat"]:
            return False
        e = self.entity_at(*cell)
        p["held"] = {"cell": cell, "mat": row["mat"], "label": row["label"],
                     "arm": row["arm"], "grab": True}
        if self._overbalanced(p, own) is not None:
            p["held"] = None                 # it would not hold him after all
            return False
        self._reflex_fire(p, "balance", "holds",
                          f"{p['name']} grabs "
                          + self.name(e, f"the {row['label']}")
                          + " to keep their feet", result="grabbed",
                          target=("thing", e) if e else None)
        return True

    def _tip_over(self, p, cells, over):
        """The body is going over: turn it about the edge of its base. The
        edge is the side of the firm contacts the weight is past, and the
        pivot is the lowest of the contacts there; a speed from a push (the
        part along the way he goes) is the turn it starts with. Returns the
        new free body, None if there is nothing to turn about (let him go
        instead), or False if he is wedged against the world."""
        foot, _g = self._footing(p, cells)
        if not len(foot):
            return None
        axis = 0 if abs(over[0]) >= abs(over[1]) else 1
        s = 1 if over[axis] > 0 else -1
        edge = foot[:, axis].max() if s > 0 else foot[:, axis].min()
        zb = float(foot[foot[:, axis] == edge][:, 2].min())
        v = p.pop("_v", None) or (0.0, 0.0)
        if self.body_contact and p.pop("_v_src", None) is not None:
            com0, kg0 = self._com_of(cells)          # the speed goes into the turn
            self._cbook()["to_fall_J"] += 0.5 * kg0 * (v[0] ** 2 + v[1] ** 2)
        p.pop("_cr_ev", None)
        hist = p.get("_xy_hist")
        if hist and len(hist) >= 2 and hist[-1][0] > hist[0][0]:
            # the speed he was hauled at, over the ticks kept: his weight
            # arrives at the edge with it, as a push's would
            dt = (hist[-1][0] - hist[0][0]) * TICK_S
            vm = 0.1 * self.scale
            v = (v[0] + (hist[-1][1] - hist[0][1]) * vm / dt,
                 v[1] + (hist[-1][2] - hist[0][2]) * vm / dt)
        w0 = 0.0
        if v[axis] * s > 0.0:
            com, _m = self._com_of(cells)
            h = max((float(com[2]) - zb + 0.5) * 0.1 * self.scale, 0.25)
            w0 = v[axis] * s / h
        return self._tip(p, cells, axis, s, float(edge), zb, w0)
