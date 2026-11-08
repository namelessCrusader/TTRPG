"""Perception: what reached each body's senses this tick, as records.

Step V3 of docs/sim-mind-design.md (3.4). Every tick `_law_perceive` writes,
into each person's Mind, a row (`Rec`) for each thing that reached them --
heard, saw, felt, did, looked -- and each row names the event that caused
it. The decision models will read only these rows, so a row says only what
the senses could tell: a sound heard through a wall has no actor, a thing
seen is placed where it was seen.

V4 adds beliefs (beliefs.py, the one writer of `Mind.bel`): each row, as
it is written, is handed to `_learn`, and the "looked" pass also places the
named things the eyes fall on.

V5 adds the reader: `render(rec, viewer)` says a row in words, from the
viewer's side ("someone hits Berel, to the north", "a crash to the
north-east"), and beliefs.py `view` hands rows and beliefs to a mind.

ALL OF IT IS WRITE-ONLY for now. Today's percept chain in `_law_will`
decides exactly as it did; it only learns, here, which event each of its
percepts came from (`noticed`). Nothing that decides reads a row until a
later step says so (3.4, step M7). And nothing here calls anything that
caches or claims: one `burning()` under the gate `_law_will` always used,
`_law_hear` as it was, and plain reads of the lattice.
"""
import weakref

import numpy as np

from .beliefs import BeliefsMixin
from .events import Rec, _Pending, _rid
from .materials import (MATNAME, SEE_LIT_SMOKE, SEE_TAU, SOUND, _DENS_ARR,
                        _TRANSMIT_ARR)
from .policy import LINES, WILL

# `_owners`' last answer per world, beside it and not in it (`_owners`).
_OWNERS = weakref.WeakKeyDictionary()

# FIRES ARE KEPT AS BOXES, by place (design prototype fire_proto.py). A lit
# cell within FIRE_GROW cells of a fire's box is part of that fire and grows
# its box; lit cells that no fire claims, closer than FIRE_SEED to each
# other, are one new fire ("ignites"); a fire with nothing lit in its box for
# FIRE_OUT ticks has gone out ("burns_out"). The wait is what stops a fire
# that flickers between ticks from going out and lighting again every tick.
FIRE_GROW, FIRE_SEED, FIRE_OUT = 2, 3, 10
# A STANDING SIGHT (a fire, still burning) is written again only after it has
# been out of sight this long: the time the attention law will wait before a
# percept counts as news again (`attend_gap`, 3.5).
LOOK_AGAIN = 40
GOT_CAP = 256                      # rows of `got` before old ones are dropped
# What, felt on the body, readies it for the next thing (`alert_s`): a body
# that has been struck is a body in a fight.
ALERTS = frozenset({"hits", "wounds"})
# Events too ordinary to be written as seen or done. A footstep is HEARD
# (that is how a sleeper next door is woken), but seeing someone walk is the
# `goes` they chose and the place `looked` finds them at, not a row a step.
QUIET = frozenset({"steps"})
# Events that happen in a head, not in the room: no sense perceives them, not
# even the body whose decision it was (its acts are what it did). `decides`
# is the cause log's own record (V5).
INNER = frozenset({"decides"})

# ── how a row reads (`render`, V5) ─────────────────────────────────────────
# What a body says it saw or did, verb by verb: {who} is the doer, {t} what
# it was done to. Written in the third person; a row of the viewer's own
# ("you ...") is turned into the second (`_you`). A verb not here reads as
# itself ("someone rolls").
SAYS_AS = {
    "swings": "{who} swings an arm", "hits": "{who} hits {t}",
    "wounds": "{who} wounds {t}", "takes": "{who} takes up {t}",
    "holds": "{who} takes hold of {t}", "catches": "{who} catches {t}",
    "releases": "{who} lets go", "reaches": "{who} reaches out",
    "pulls_in": "{who} lowers an arm", "jumps": "{who} jumps",
    "goes": "{who} sets off", "stays": "{who} stays put",
    "looks": "{who} looks round", "leans": "{who} leans out",
    "stands": "{who} straightens up", "says": "{who} shouts",
    "falls": "{who} falls", "lands": "{who} lands", "rolls": "{who} tucks to roll",
    "faints": "{who} faints", "dies": "{who} dies", "wakes": "{who} wakes",
    "arrives": "{who} gets out", "rescues": "{who} hauls {t} clear",
    "retries": "{who} tries another way", "gives_up": "{who} gives up",
    "topples": "{who} topples", "looses": "{who} looses {t}",
    "steps": "{who} steps", "gone": "{who} is gone",
}
# The same, for a result that changes what was done.
SAYS_AS_RESULT = {("swings", "thrown"): "{who} throws something",
                  ("catches", "missed"): "{who} grabs at {t} and misses"}
# Things that happen to matter, with no doer: what it happened to is the
# subject ("the glass breaks").
SAYS_AS_WORLD = {
    "breaks": "{t} breaks", "melts": "{t} melts", "sets": "{t} sets",
    "dissolves": "{t} dissolves", "ignites": "a fire starts",
    "catches_fire": "{t} catches fire", "burns_out": "a fire goes out",
    "touches": "{t} knocks against something", "strikes": "{t} strikes something",
    "cracks": "{t} cracks", "parts": "{t} comes apart",
    "splits": "{t} comes apart", "ashes": "{t} burns to ash",
}
# What a sound is, heard and not seen: the event it is the noise of names
# only its kind. A shout whose words were made out is said with its words.
HEARD_AS = {"says": "a shout", "steps": "footsteps", "hits": "a blow",
            "lands": "a thud", "breaks": "a crash", "strikes": "a knock"}
# The verbs whose third-person form is not the base form plus s or es.
_BASE = {"goes": "go", "dies": "die", "tries": "try", "is": "are",
         "has": "have"}


def _you(phrase):
    """A third-person phrase whose doer is the reader: "you take up the
    axe" for "{who} takes up the axe"."""
    verb, _, rest = phrase.partition(" ")
    base = _BASE.get(verb)
    if base is None:
        base = verb[:-2] if verb.endswith(("ches", "shes", "sses", "xes")) \
            else verb[:-1] if verb.endswith("s") else verb
    return ("you " + base + (" " + rest if rest else "")).rstrip()


class PerceiveMixin(BeliefsMixin):

    def _law_perceive(self):
        """What reached each body's senses this tick. `step` calls it in the
        place where it called `_law_hear`, just before `_law_will`.

        (a) The fires: `burning()` once, under exactly the gate `_law_will`
            used for it (someone alive and awake), handed on to `_law_will`
            as `_burning_now`. A wider gate would call it on ticks it was
            never called on, and `burning()` -> `T()` bumps `_matter_gen`.
        (b) `_law_hear`, unchanged but for the `heard` rows it now writes.
        (c, d) What each body saw, felt and did: every event since last
            time, put to each pair of eyes (`_witness`).
        (e) Every `look_every` ticks, when `_law_will` sweeps the eyes
            (`_look_around`),
            who, which fires and which named things those eyes fall on
            (`_look`)."""
        lit = None
        if self.persons and any(p["alive"] and p["awake"]
                                for p in self.persons):
            lit = self.burning()
        fire_at = np.argwhere(lit) if lit is not None and lit.any() else None
        if lit is not None:
            self._fire_book(fire_at)
        self._burning_now = (self.tick, lit, fire_at)
        self._law_hear()
        self._witness()
        if self.persons and self.tick % WILL["look_every"] == 0:
            self._look(fire_at)

    # ── (a) fires ────────────────────────────────────────────────────────────
    def _fire_book(self, cells):
        """Bring the fires up to date with this tick's lit cells.

        Every lit cell ends up inside some fire's box, so a lit cell can
        always say which fire it is (`_fire_of`). A fire IS its `ignites`
        event: its id is the fire's name, `("fire", ev_id)`, and the same id
        is the cause of its `burns_out` and of every `catches_fire` it
        starts. All three are DEFERRED to the tick's end, like every event
        raised before `_law_will` (`_defer` says why), so for the rest of
        the tick a new fire's id is a box."""
        tick, fires = self.tick, self._fires
        if cells is None:
            cells = np.zeros((0, 3), np.int64)
        claimed = np.zeros(len(cells), bool)
        # THE LIT CELLS ARE IN ORDER (`np.argwhere`: by x, then y, then z), so
        # the ones a fire's box can reach are a slab found by bisection, and
        # a house's worth of lit cells is not read once per fire.
        banded = self._windows_on() and len(cells) > 64
        for f in fires:                              # oldest fire first
            if claimed.all():
                break
            lo, hi = np.array(f[0]), np.array(f[1])
            a, b = 0, len(cells)
            if banded:
                a = int(np.searchsorted(cells[:, 0], lo[0] - FIRE_GROW, "left"))
                b = int(np.searchsorted(cells[:, 0], hi[0] + FIRE_GROW, "right"))
            near = cells[a:b]
            inb = ((near >= lo - FIRE_GROW) & (near <= hi + FIRE_GROW)
                   ).all(axis=1) & ~claimed[a:b]
            if inb.any():
                f[0] = tuple(int(v) for v in np.minimum(lo, near[inb].min(axis=0)))
                f[1] = tuple(int(v) for v in np.maximum(hi, near[inb].max(axis=0)))
                f[2] = tick
                claimed[a:b] |= inb
        rest = cells[~claimed]                       # argwhere order: the same
        while len(rest):                             # every run
            lo = hi = rest[0]
            while True:
                inb = ((rest >= lo - FIRE_SEED) & (rest <= hi + FIRE_SEED)
                       ).all(axis=1)
                nlo, nhi = rest[inb].min(axis=0), rest[inb].max(axis=0)
                if (nlo == lo).all() and (nhi == hi).all():
                    break
                lo, hi = nlo, nhi
            lo = tuple(int(v) for v in lo)
            hi = tuple(int(v) for v in hi)
            box = self._defer("ignites", place=tuple(int(v) for v in rest[0]),
                              data=(("hi", hi), ("lo", lo)), text="")
            self._boxed = True
            fires.append([lo, hi, tick, box])
            rest = rest[~inb]
        # A THING THAT IS BURNING caught fire from the fire it is in. Only a
        # registered thing: unnamed matter is just where a fire is.
        if len(cells):
            e = self.eid[tuple(cells.T)]
            for u in np.unique(e[self._is_thing(e)]):
                u = int(u)
                if u not in self._alight:
                    at = tuple(int(v) for v in cells[e == u][0])
                    self._defer("catches_fire", target=("thing", u), place=at,
                                cause=self._fire_of(at), link="carried",
                                text="")
                self._alight[u] = tick
        for u in [u for u, t in self._alight.items() if tick - t >= FIRE_OUT]:
            del self._alight[u]
        keep = []
        for f in fires:
            if tick - f[2] >= FIRE_OUT:
                self._defer("burns_out", target=("fire", _rid(f[3])),
                            place=tuple((a + b) // 2 for a, b in zip(f[0], f[1])),
                            cause=f[3], link="carried", text="")
            else:
                keep.append(f)
        self._fires = keep

    def _fire_of(self, cell):
        """The fire (its `ignites` id, or the box it is still in) whose box
        holds `cell`, or None."""
        for lo, hi, _t, ev in self._fires:
            if all(lo[i] <= cell[i] <= hi[i] for i in range(3)):
                return ev
        return None

    def _fire_near(self, at):
        """The fire whose box is nearest the point `at`, or None if there is
        no fire. What smoke at a mouth or heat on a skin is blamed on: the
        smoke came from SOME fire, and the nearest is the best guess there
        is (a "near" cause, 3.4)."""
        best, bd = None, None
        for lo, hi, _t, ev in self._fires:
            d = max(max(lo[i] - at[i], at[i] - hi[i], 0.0) for i in range(3))
            if bd is None or d < bd:
                best, bd = ev, d
        return best

    # ── the records ──────────────────────────────────────────────────────────
    @staticmethod
    def _senses(p):
        """Which senses this body has right now (3.4, "who perceives"): None
        for the dead, the safe and the unconscious; "heard" for a sleeper;
        "all" for anyone awake."""
        if not p["alive"] or p["safe"] or not p["awake"]:
            return None
        return "heard" if p.get("asleep") else "all"

    def _note(self, pid, how, ev, verb=None, actor_seen=None, target_seen=None,
              place=None, db=None, data=(), key=None):
        """Write one row into a Mind, unless the same thing (`key`; by
        default the event and the sense) was already written and has been
        perceived again within LOOK_AGAIN ticks. Returns the row, or None."""
        m = self.minds[pid]
        key = (ev, how) if key is None else key
        last = m.got.get(key)
        m.got[key] = self.tick
        if last is not None and self.tick - last < LOOK_AGAIN:
            return None
        if len(m.got) > GOT_CAP:       # a row that old would be news again
            for k in [k for k, t in m.got.items()      # anyway: dropping it
                      if self.tick - t >= LOOK_AGAIN]:  # changes nothing
                del m.got[k]
        rec = Rec(tick=self.tick, how=how, ev=ev, verb=verb,
                  actor_seen=actor_seen, target_seen=target_seen, place=place,
                  db=db, data=tuple(data))
        m.perceived.append(rec)
        if isinstance(ev, _Pending):
            self._boxed = True
        return rec

    def _owners(self):
        """Whose body each cell is part of, as (sorted flat cell indices,
        the pid of each), from each person's `_own` -- the claim
        `_claim_bodies` made this tick, read and never remade. `rays` uses
        it to let NEITHER END'S OWN BODY be a wall: the looker's head is
        not in the way of its own eyes, and the person looked at is not in
        the way of being seen. A ray aimed at a head would otherwise stop
        on that head (the rule `_law_hear` and `_look_around` already keep)."""
        # KEPT WHILE THE DEEDS ARE THE SAME OBJECTS. The answer is a function
        # of each person's `_own`, by its place in `persons`, and nothing else
        # (a cell in two deeds goes to the first, by the stable sort, however
        # either set iterates). A claim or a mover makes a NEW frozenset, so
        # holding the old ones and asking `is` cannot be fooled; `rays` is
        # called many times a tick and the deeds change at most once in it.
        owns = tuple(q.get("_own") for q in self.persons)
        kept = _OWNERS.get(self)
        if kept is not None and len(kept[0]) == len(owns) \
                and all(a is b for a, b in zip(kept[0], owns)):
            return kept[1]
        out = self._owners_of()
        _OWNERS[self] = (owns, out)
        return out

    def _owners_of(self):
        """`_owners`, made from the deeds."""
        keys, who = [], []
        for i, q in enumerate(self.persons):
            own = q.get("_own")
            if own:
                c = np.fromiter((v for cell in own for v in cell), np.int64,
                                count=3 * len(own)).reshape(-1, 3)
                keys.append(np.ravel_multi_index(c.T, self.shape))
                who.append(np.full(len(own), i, np.int64))
        if not keys:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        k, p = np.concatenate(keys), np.concatenate(who)
        o = np.argsort(k, kind="stable")
        return k[o], p[o]

    # ── (b) hearing ──────────────────────────────────────────────────────────
    def _src(self, snd):
        """The event a sound is the noise of: the `says` a shout carries
        (`ev_id`), or what `_noise` was told or will be (`src`)."""
        return snd.get("ev_id", snd.get("src"))

    def _heard(self, p, snd, lvl):
        """A `heard` row for one audible pair: one line beside the `_level`
        call in `_law_hear`, which is unchanged otherwise.

        Audible means at least `notice_db`, the level a noise in a quiet
        room is noticed at. A sleeper hears (3.4); the unconscious do not.
        The row never says WHO: `actor_seen` is set only for a `says` whose
        words were made out by a listener who already knows that voice --
        and a voice is learned by seeing and hearing the same `says`
        (`_witness`). Made out is `_understood`, the rule `_hears` keeps
        too: loud enough to be words (HEAR_DB) and not drowned by the room
        (`w.ambient_db`), so a row never says a line came through that the
        hearing law says was masked."""
        if lvl < SOUND["notice_db"] or not p["awake"]:
            return
        src = self._src(snd)
        pid = p["pid"]
        ev = self.events.get(src) if isinstance(src, int) else None
        verb = ev.verb if ev is not None else None
        who, data = None, []
        said = verb == "says" and self._understood(lvl)
        if said:
            data.append(("line", ev.result))
            if self._knows_voice(pid, ev.actor):
                who = ev.actor
        if p.get("asleep"):
            data.append(("asleep", True))
        # a sound lasts two ticks; it is the same sound on both
        rec = self._note(pid, "heard", src, verb=verb, actor_seen=who,
                         place=tuple(snd["at"]), db=float(lvl),
                         data=sorted(data),
                         key=("heard", snd["tick"], tuple(snd["at"])))
        if rec is not None and said:
            self._learn(pid, "heard", ev, who, rec.place)
            self._told(pid, ev, rec.place)

    # ── (c, d) seeing, feeling, doing ────────────────────────────────────────
    def _witness(self):
        """Put every event since last time to everyone who could perceive it.

        - DID: the actor's own act (not to the sleeping or unconscious).
        - FELT: an event that targets a person, other than its actor.
        - SAW: an awake body whose eyes are turned toward where it happened,
          within `see_m`, with a clear line to it (`rays`, one batched call
          for every pair). Where it happened is its `place`, or else where
          its actor stands, or else where the person it targets stands --
          as of the start of this tick (3.4's risk list).

        A `saw` row names the actor only if the actor was seen too: an act
        with no place of its own is seen AT its actor, and for one with a
        place (a stone landing) a second ray, in the same batch, asks
        whether the thrower was in view as well. Seeing an act at its actor
        also places them (`("at", pid)`). Many rows of one happening (a
        pane's shatter is a `breaks` a cell) make one row a viewer."""
        new = self.events.since(self._seen_upto - 1)
        self._seen_upto = self.events.next_id
        persons = self.persons
        if not new or not persons:
            return
        n = len(persons)
        sense = [self._senses(p) for p in persons]
        vox_m = 0.1 * self.scale
        see = WILL["see_m"] / max(vox_m, 1e-9)
        pairs = []
        for ev in new:
            if ev.verb in INNER:
                continue
            a = ev.actor if ev.actor is not None and 0 <= ev.actor < n else None
            res = (("result", ev.result),) if ev.result is not None else ()
            if a is not None and sense[a] == "all" and ev.verb not in QUIET:
                self._note(a, "did", ev.id, verb=ev.verb, actor_seen=a,
                           target_seen=ev.target, place=ev.place, data=res)
                self._learn(a, "did", ev, a, ev.place)
            t = ev.target
            if t is not None and t[0] == "person" and t[1] != a \
                    and 0 <= t[1] < n and sense[t[1]] == "all":
                self._note(t[1], "felt", ev.id, verb=ev.verb,
                           place=ev.place, data=res)
                if ev.verb in ALERTS:          # struck: ready for the next
                    persons[t[1]]["_alert"] = self.tick
            if not ev.visible or ev.verb in QUIET:
                continue
            # where it happened, and whose body it happened at (theirs is
            # not in the way of seeing it): a placed event is at the person
            # it targets, if any, else its actor; an event with no place is
            # where its actor stands, else where its target does
            hit = t[1] if t is not None and t[0] == "person" \
                and 0 <= t[1] < n else None
            where, who = ev.place, (hit if hit is not None else a)
            if where is None:
                who = a if a is not None else hit
                if who is not None:
                    where = persons[who].get("_eye")
            if where is None:
                continue
            # and, when that is not where its actor stands, the actor too:
            # a stone seen landing does not show who threw it
            other = persons[a].get("_eye") if a is not None and who != a \
                else None
            for v, p in enumerate(persons):
                eye = p.get("_eye")
                if v == a or sense[v] != "all" or eye is None:
                    continue
                if float(np.linalg.norm(np.subtract(where, eye))) > see:
                    continue
                gaze = p.get("gaze", p.get("facing", (1.0, 0.0)))
                if self._in_cone(gaze, eye, where):
                    # SEEN AT THE EDGE OF THE VIEW, it is seen to happen, but
                    # not who did it or to whom (`_in_focus`): the actor is
                    # told only from the middle of the view, at the place or
                    # (a second ray) where the actor stands
                    pairs.append((v, ev, eye, where, res,
                                  -1 if who is None else who,
                                  other if other is not None and float(
                                      np.linalg.norm(np.subtract(other, eye)))
                                  <= see and self._in_focus(gaze, eye, other)
                                  else None,
                                  self._in_focus(gaze, eye, where)))
        if not pairs:
            return
        todo = [(q[2], q[3], (q[0], q[5])) for q in pairs] \
            + [(q[2], q[6], (q[0], q[1].actor)) for q in pairs
               if q[6] is not None]
        ok = self.rays([r[0] for r in todo], [r[1] for r in todo],
                       skip=[r[2] for r in todo])
        also = iter(ok[len(pairs):])
        done = set()
        for (v, ev, eye, where, res, who, other, focus), seen in zip(pairs, ok):
            by = ev.actor if (who == ev.actor and focus) or (
                other is not None and next(also)) else None
            if not seen or (v, ev.verb, ev.actor, ev.cause) in done:
                continue
            done.add((v, ev.verb, ev.actor, ev.cause))
            at = tuple(int(round(c)) for c in where)
            tgt = ev.target
            if not focus and tgt is not None and tgt[0] == "person":
                tgt = None                         # someone: not who
            if self.misread:                       # off by default
                by, tgt = self._misread(v, ev, by, tgt, at)
            self._note(v, "saw", ev.id, verb=ev.verb, actor_seen=by,
                       target_seen=tgt, place=at,
                       data=() if ev.verb == "says" else res)  # words are
                                                               # heard, not seen
            self._learn(v, "saw", ev, by, at, tgt)

    # ── (e) looking ──────────────────────────────────────────────────────────
    def _look(self, fire_at):
        """Who and which fires the eyes fall on, every 4 ticks: the cadence
        `_look_around` sweeps them at (`_law_will`).

        A person in the middle of the view (`_in_focus`: only there is a
        person told apart) is believed where they stand (`("at", pid)`, how
        = "looked"): the only other way an `at` belief is made is `saw`. A
        fire anywhere in view is a `looked` row naming the fire, written
        again only once it has been out of sight for LOOK_AGAIN ticks. For
        each pair of eyes it tries the 25 lit cells nearest it, out to
        `see_m`. A fire lit this tick has no id yet, so it is looked at
        next time.

        A NAMED THING in view is believed where it is (`("at", ("thing",
        eid))`, at its middle): of the THINGS_SEEN named things nearest the
        eye, each in the middle of the view and in sight (V4). The eye looks at the cell
        of it nearest the eye. A thing not looked at again stays believed
        where it was last seen, wherever it has gone since. A world that
        names nothing has nothing to look at here."""
        persons = self.persons
        vox_m = 0.1 * self.scale
        see = WILL["see_m"] / max(vox_m, 1e-9)
        things = self._things_now()
        pairs = []
        for v, p in enumerate(persons):
            eye = p.get("_eye")
            if self._senses(p) != "all" or eye is None:
                continue
            gaze = p.get("gaze", p.get("facing", (1.0, 0.0)))
            for u, q in enumerate(persons):
                qe = q.get("_eye")
                if u == v or not q["alive"] or q["safe"] or qe is None:
                    continue
                if float(np.linalg.norm(np.subtract(qe, eye))) > see:
                    continue
                if self._in_focus(gaze, eye, qe):    # WHO: the middle of
                    pairs.append((v, ("person", u), eye, qe, u, None))  # view
            if fire_at is not None:
                # only the band of x the eye can see across (the lit cells
                # come in x order), grown by a cell to be sure of its edges;
                # the cells it keeps are in the order they had
                lit = fire_at
                if self._windows_on() and len(fire_at) > 64:
                    lit = fire_at[
                        int(np.searchsorted(fire_at[:, 0], eye[0] - see - 1, "left")):
                        int(np.searchsorted(fire_at[:, 0], eye[0] + see + 1, "right"))]
                d = np.abs(lit[:, :2] - np.array(eye[:2])).max(axis=1)
                close = lit[d <= see]
                for c in close[np.argsort(d[d <= see], kind="stable")][:25]:
                    if self._in_cone(gaze, eye, c):
                        pairs.append((v, ("fire", tuple(int(x) for x in c)),
                                      eye, c.astype(np.float32), -1, None))
            if things:
                for e, c, mid in self._things_to_look_at(eye, see, things):
                    if self._in_focus(gaze, eye, c):     # WHICH thing
                        pairs.append((v, ("thing", e), eye,
                                      c.astype(np.float32), -1, mid))
        if not pairs:
            return
        ok = self.rays([q[2] for q in pairs], [q[3] for q in pairs],
                       skip=[(q[0], q[4]) for q in pairs],
                       lit=[q[1][0] == "fire" for q in pairs])
        lit = set()
        for (v, what, eye, at, _u, mid), seen in zip(pairs, ok):
            if not seen:
                continue
            if what[0] == "person":
                self._looked_at(v, what, tuple(int(round(c)) for c in at))
                continue
            if what[0] == "thing":
                self._looked_at(v, what, mid)
                continue
            f = self._fire_of(what[1])
            if f is None or isinstance(f, _Pending):
                continue
            self._note(v, "looked", f, target_seen=("fire", f), place=what[1])
            if (v, f) not in lit:                  # its nearest lit cell
                lit.add((v, f))
                self._looked_at(v, ("fire", f), what[1], f)

    # ── the reader (V5) ──────────────────────────────────────────────────────
    def render(self, rec, viewer):
        """One percept row in words, from the side of the body it reached.

        `viewer` is that body (a pid or its person dict). A row says only
        what its senses could tell, and so does this: a person is named only
        when the row names them (`actor_seen`, `target_seen`); otherwise it
        is "someone", or "something" for what was felt. A sound says only
        what kind of sound it was, and the words when they were made out.
        Where it was is a direction from the viewer's eyes. `Ev.text`, the
        legacy line, is never read; the event itself is read only for the
        words of a shout that was understood (a speaker's sheet can word a
        line its own way)."""
        v = self.persons[viewer] if isinstance(viewer, int) else viewer
        me = v["pid"]
        where = self._where_words(rec.place, v)
        data = dict(rec.data)
        if rec.how == "heard":
            line = data.get("line")
            if rec.verb == "says" and line is not None:
                ev = self.events.get(_rid(rec.ev)) if rec.ev is not None \
                    else None
                q = self.persons[ev.actor] if ev is not None \
                    and ev.actor is not None \
                    and 0 <= ev.actor < len(self.persons) else {}
                words = q.get("lines", LINES).get(line, LINES.get(line, line))
                who = self._who_words(rec.actor_seen, me, "someone")
                out = f'{who} shouts "{words}"'
            else:
                out = HEARD_AS.get(rec.verb, "a noise")
            return f"{out} {where}" if where else out
        if rec.how == "looked":
            what = self._what_words(rec.target_seen, me)
            return f"{what} {where}" if where else what
        t = rec.target_seen
        if rec.how == "felt":                   # felt on the body: no doer
            who, t = "something", ("person", me)
        else:
            who = self._who_words(rec.actor_seen, me, "someone")
        form = SAYS_AS_RESULT.get((rec.verb, data.get("result")))
        if form is None and rec.verb in SAYS_AS_WORLD:
            form = SAYS_AS_WORLD[rec.verb]
            if rec.actor_seen is not None and rec.verb in ("breaks", "cracks"):
                form = "{who} " + rec.verb + " {t}"   # someone broke it
        if form is None:
            form = SAYS_AS.get(rec.verb, "{who} " + (rec.verb or "does something")
                               .replace("_", " "))
        out = form.format(who=who, t=self._what_words(t, me))
        if form.startswith("{who} ") and who == "you":
            out = _you(out[4:])
        if rec.how == "saw" and where:
            out += f", {where}"
        return out

    def _who_words(self, pid, me, nobody):
        """The doer of a row: "you", a person's name, or `nobody`."""
        if pid is None or not 0 <= pid < len(self.persons):
            return nobody
        return "you" if pid == me else self.persons[pid]["name"]

    def _what_words(self, target, me):
        """What a row says something was done to, in words. A person only
        by a name the row holds; a thing by its name (`name`); a fire as a
        fire; matter with no name as its material."""
        if target is None:
            return "something"
        kind = target[0]
        if kind == "person":
            return self._who_words(target[1], me, "someone")
        if kind == "thing":
            return self.name(target[1], "something")
        if kind == "fire":
            return "a fire"
        if kind == "mat":
            return "the " + MATNAME.get(int(target[1]), "stuff")
        return "something"

    def _where_words(self, place, v):
        """Where `place` (cells) is from body `v`'s eyes: "close by" within a
        metre, else "to the north-east" (`_bearing`), or "" when either end
        is not known."""
        if place is None:
            return ""
        eye = v.get("_eye")
        if eye is None:
            a = v.get("anchor")
            if a is None:
                return ""
            eye = (a[0], a[1])
        dx, dy = float(place[0]) - float(eye[0]), float(place[1]) - float(eye[1])
        if max(abs(dx), abs(dy)) * 0.1 * self.scale < 1.0:
            return "close by"
        return "to the " + self._bearing(dx, dy)

    # ── sight, batched ───────────────────────────────────────────────────────
    def rays(self, A, B, skip=None, lit=None):
        """`_sees(A[i], B[i])` for every i, in one pass over the lattice.

        The same samples as `_sees` -- the `np.linspace` points between the
        two ends, rounded to cells, the eye's own cell and its neighbours
        skipped, then clipped to the world -- and the same optical depth:
        each cell dims the view by how full it is and how opaque its
        material, plus the smoke in it, and the far end is seen if the
        total is under SEE_TAU. Only the order the depths are added in
        differs (one sum per ray here, numpy's pairwise sum there), which
        moves a depth by about a part in ten million.

        `skip[i]`, when given, is a pair of pids (-1 for none) whose own
        flesh does not dim ray i (`_owners`): the looker's, and that of the
        person it looks at. Without it this is `_sees` exactly. With it,
        THE RAY STARTS IN THE HEAD: the cells next to the eye are not
        skipped as `_sees` skips them (its "eyelashes"), only flesh is, so
        a wall a face is pressed against is a wall (`_sees(own=)`).

        `lit[i]` True: the far end GIVES light (a flame), so the smoke on ray i
        dims it by SEE_LIT_SMOKE as much (materials.SEE_LIT_SMOKE). Solids do not."""
        A = np.asarray(A, np.float32).reshape(-1, 3)
        B = np.asarray(B, np.float32).reshape(-1, 3)
        m = len(A)
        if not m:
            return np.ones(0, bool)
        n = np.ceil(np.abs(B - A).max(axis=1)).astype(np.int64) + 1
        k = np.where(n > 2, n - 2, 0)              # points between the ends
        tot = int(k.sum())
        if not tot:
            return np.ones(m, bool)
        ray = np.repeat(np.arange(m), k)
        j = np.arange(tot) - np.repeat(np.cumsum(k) - k, k) + 1
        ts = j * (1.0 / (np.repeat(n, k) - 1))      # == np.linspace's points
        a = A[ray]
        pts = np.round(a + ts[:, None] * (B - A)[ray]).astype(np.int64)
        if skip is None:
            near = np.abs(pts - a).max(axis=1) > 1.0   # skip own eyelashes
            pts, ray = pts[near], ray[near]
        for ax in range(3):
            np.clip(pts[:, ax], 0, self.shape[ax] - 1, out=pts[:, ax])
        idx = tuple(pts.T)
        vox_m = 0.1 * self.scale
        mat = self.mat[idx]
        packed = np.clip(self.smass[idx] / np.maximum(
            _DENS_ARR[mat] * self.vox_l, 1e-9), 0.0, 0.999)
        through = (1.0 - packed) + packed * _TRANSMIT_ARR[mat]
        opaque = -np.log(np.maximum(through, 1e-9))
        if skip is not None:
            keys, pid = self._owners()
            if len(keys):
                flat = np.ravel_multi_index(idx, self.shape)
                at = np.minimum(np.searchsorted(keys, flat), len(keys) - 1)
                own = np.where(keys[at] == flat, pid[at], -2)
                sk = np.asarray(skip, np.int64).reshape(-1, 2)[ray]
                opaque = np.where((own == sk[:, 0]) | (own == sk[:, 1]),
                                  0.0, opaque)
        smk = np.bincount(ray, self._smoke_ext() * (self.smoke[idx] / self.vox_l)
                          * vox_m, m)
        if lit is not None:
            smk = np.where(np.broadcast_to(np.asarray(lit, bool), (m,)), smk * SEE_LIT_SMOKE, smk)
        tau = np.bincount(ray, opaque, m) + smk
        return (np.bincount(ray, minlength=m) == 0) | (tau < SEE_TAU)

    # ── what the percept chain of `_law_will` came from ─────────────────────
    def _last_act(self, pid, verb, result=None, back=512):
        """The id of the newest `verb` this person did (with this `result`),
        among the last `back` events, or None."""
        for ev in reversed(self.events.rows[-back:]):
            if ev.actor == pid and ev.verb == verb \
                    and (result is None or ev.result == result):
                return ev.id
        return None
