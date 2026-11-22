"""Beliefs: what each person holds to be so, and the event that made them.

Step V4 of docs/sim-mind-design.md (3.4). Perception (perceive.py) writes
what reached each body's senses; this file turns those percepts into
BELIEFS, `Mind.bel[key] = (value, ev_id, how, tick)`, and is the one place
that writes them. A belief keeps the event it came from, so a wrong one can
always be traced back to the truth in the log.

The keys (3.4):
  ("at", pid)              where someone is        saw, looked
  ("at", ("thing", eid))   where a thing is        looked
  ("holds", pid)           what someone holds      saw, did
  ("state", pid)           fainted, dead, ...      saw, did
  ("hurt_by", victim)      who hurt them           saw
  ("alight", fire)         where a fire is         saw, looked, told
  ("said", pid)            their last words        heard, saw + heard
  ("voice", pid)           I know their voice      saw + heard

Three rules hold for every key:
  - At most BEL_CAP keys. The one changed longest ago goes first, and a
    `voice` is never forgotten.
  - What a body saw for itself is replaced by anything it perceives later.
  - A `told` belief never replaces a FRESHER first-hand one: what you were
    told about a fire as of tick 40 does not undo the fire you saw go out at
    tick 60 (the lab rule, lab/world2/world.py:616-621).

Two hooks make WRONG BELIEFS THAT HAVE A REASON, and both are off by default
(`w.misread`, `w.claims`):
  - A MISREAD: a sighting names the wrong person or thing. The row and the
    belief keep the true event, so the log still says what really happened.
  - A CLAIM: a line of speech that says something is so ("Fire!"). The
    hearer believes it unless an insight roll sees through a claim the
    speaker had no grounds for. The `says` event records whether the claim
    was true and whether it was grounded in the speaker's own senses.

Every draw is `hash01` of what is being drawn about: the sim keeps no RNG
(2.4), so a snapshot replays its misreads exactly.

V5 adds the reader a mind is handed, `view(w, p)`: the body's percept rows
and its beliefs as plain dicts shaped like the mind interface's `Percept`
and `Belief` (src/mind/api.py), written in words from the body's side.
"""
import numpy as np

from .events import _rid, hash01
from .materials import BODY, GRAVITY
from .policy import LINES, MEANS

BEL_CAP = 64                       # beliefs a Mind keeps (3.4)
THINGS_SEEN = 8                    # named things each pair of eyes tries
# What a body perceives for itself. A `told` belief is the only other kind.
FIRSTHAND = frozenset({"saw", "looked", "felt", "did", "heard"})

# ── what a perceived event makes a body believe, verb by verb ──────────────
# What someone has in hand after it. A take names no target today, so the
# belief is only "something" (True) until it does.
HOLDS = {"takes": True, "holds": True, "catches": True, "releases": None}
# The state of the actor after it.
STATE = {"faints": "fainted", "dies": "dead", "wakes": "awake",
         "falls": "falling", "stands": "standing", "jumps": "jumping"}
# Who hurt whom: the actor of these, when the senses told who it was.
HURTS = frozenset({"hits", "wounds"})

# ── the two hooks, off by default ──────────────────────────────────────────
# MISREAD: for each verb, the chance that a sighting of it names the wrong
# actor (or the wrong thing, for a verb aimed at a thing). Used when
# `w.misread` is True; `w.misread` may also be a dict like this one.
MISREAD = {"hits": 0.15, "swings": 0.15, "takes": 0.15, "catches": 0.15,
           "releases": 0.15, "says": 0.10, "goes": 0.10}
# CLAIMS: what a line of speech claims, by the line's key in LINES. "fire"
# claims a fire. Used when `w.claims` is True.
CLAIMS = {"fire": "alight"}

# ── what a mind is handed (`view`, V5) ─────────────────────────────────────
# The interrupt class of a percept (verbs.yaml `interrupts_known`), by how it
# came and what it was. A percept with one has salience 1, else 0.
HARM = frozenset({"hits", "wounds", "strikes"})
VIOLENCE = frozenset({"hits", "wounds", "swings"})
FIRE = frozenset({"ignites", "catches_fire"})
# The mind interface's channels (src/mind/api.py CHANNELS). Looking round is
# seeing.
CHANNEL = {"heard": "heard", "saw": "saw", "felt": "felt", "did": "did",
           "looked": "saw", "told": "told"}
# How a belief came, in the mind interface's words: what a body did itself
# is its own.
HOW = {"did": "own", "felt": "saw"}
# A state belief, said of someone.
STATE_WORDS = {"fainted": "has fainted", "dead": "is dead", "awake": "is awake",
               "falling": "is falling", "standing": "is standing",
               "jumping": "is jumping"}
# Beliefs a scene never states: knowing a voice is how words are placed, not
# a fact about the world.
UNSTATED = frozenset({"voice"})


class BeliefsMixin:

    # ── the one writer ──────────────────────────────────────────────────────
    def _believe(self, pid, key, value, ev, how, asof=None):
        """Set one belief, `(value, ev_id, how, tick)`, and say whether it
        was set.

        A `told` belief is about the world as of `asof` (the tick the teller
        knew it, by default now). It is refused when the belief already
        held at that key is first-hand and at least that fresh. Anything
        else replaces what was there. The newest goes last; past BEL_CAP
        keys the oldest goes first, except a `voice`, which is never
        forgotten (3.4)."""
        bel = self.minds[pid].bel
        if how == "told":
            old = bel.get(key)
            asof = self.tick if asof is None else asof
            if old is not None and old[2] in FIRSTHAND and old[3] >= asof:
                return False
        bel.pop(key, None)
        bel[key] = (value, ev, how, self.tick)
        if len(bel) > BEL_CAP:
            for k in bel:
                if k[0] != "voice":
                    del bel[k]
                    break
        return True

    def _knows_voice(self, pid, who):
        """Whether person `pid` knows the voice of person `who`."""
        return ("voice", who) in self.minds[pid].bel

    # ── what each kind of percept teaches ──────────────────────────────────
    def _learn(self, pid, how, ev, who=None, place=None, target=None):
        """What one percept of event `ev` (an `Ev`) makes person `pid`
        believe. `who` is the person the senses said did it (`actor_seen`,
        maybe a misread), `target` what they said it was done to (the
        event's own target when None), `place` where it seemed to be
        (cells).

        `saw` and `did` teach the verb's own rule (HOLDS, STATE, HURTS, a
        fire lit or gone out). A thing seen done with no place of its own
        was seen at its doer, which places them. A `says` both seen and
        heard teaches the voice, and the words. `heard` words of a known
        voice say who said them; a heard sound alone teaches nothing."""
        verb, eid = ev.verb, ev.id
        # A VOICE IS LEARNED AS WHOEVER THE EYES SAID SPOKE. With misreads
        # on, the eyes can put A's words in B's mouth: then this mind learns
        # B's voice and B's words, never A's, which it was not shown.
        if how == "heard":
            if verb != "says":
                return
            m = self.minds[pid]
            m.got[(eid, "heard")] = self.tick
            if (eid, "saw") in m.got:
                seen = next((r.actor_seen for r in reversed(m.perceived)
                             if r.ev == eid and r.how == "saw"), None)
                if seen is not None:
                    self._believe(pid, ("voice", seen), True, eid, "heard")
                    who = seen
            if who is not None:
                self._believe(pid, ("said", who), ev.result, eid, "heard")
            return
        if how == "saw" and verb == "says" and who is not None:
            if (eid, "heard") in self.minds[pid].got:
                self._believe(pid, ("voice", who), True, eid, "saw")
                self._believe(pid, ("said", who), ev.result, eid, "saw")
        if how not in ("saw", "did"):
            return
        if how == "did":
            who = ev.actor
        if how == "saw" and ev.place is None and who is not None:
            self._believe(pid, ("at", who), place, eid, how)
        t = ev.target if target is None else target
        if who is not None:
            if verb in HOLDS and (verb != "catches" or ev.result == "caught"):
                v = HOLDS[verb]
                self._believe(pid, ("holds", who),
                              t if v is True and t is not None else v, eid, how)
            if verb in STATE:
                self._believe(pid, ("state", who), STATE[verb], eid, how)
            if verb in HURTS and t is not None and t[0] == "person":
                self._believe(pid, ("hurt_by", t[1]), who, eid, how)
        if verb == "ignites":
            self._believe(pid, ("alight", ("fire", eid)), place, eid, how)
        elif verb == "burns_out" and t is not None:
            self._believe(pid, ("alight", t), None, eid, how)
        elif verb == "catches_fire" and t is not None:
            self._believe(pid, ("alight", t), place, eid, how)

    def _looked_at(self, pid, what, place, ev=None):
        """Eyes fell on `what` (a ("person", pid), ("thing", eid) or
        ("fire", ev_id)) at `place`: the `looked` pass (perceive.py)."""
        if what[0] == "fire":
            self._believe(pid, ("alight", what), place, ev, "looked")
        else:
            self._believe(pid, ("at", what[1] if what[0] == "person" else what),
                          place, ev, "looked")

    # ── sightings of things ────────────────────────────────────────────────
    def _things_now(self):
        """Every registered, living thing on the lattice (or drawn from the
        solver this tick), as {eid: (n, 3) cells}. Empty, and nearly free,
        in a world that names nothing."""
        if not self.things or not self._thing_lut.any():
            return {}
        out = {}
        for e, t in self.things.items():
            if t["dead"] is not None or not self._thing_lut[e]:
                continue
            idx = self._cells_now(e)
            if len(idx):
                out[e] = np.stack(np.unravel_index(
                    np.asarray(idx, np.int64), self.shape), -1)
        return out

    def _things_to_look_at(self, eye, see, things):
        """The THINGS_SEEN named things nearest the eye within `see` cells,
        nearest first, each as (eid, the cell of it nearest the eye, where
        it is: its middle, rounded). The eye looks at the near cell, so the
        thing's own far side is not in the way of seeing it."""
        near = []
        e0 = np.asarray(eye, np.float64)
        for e, xyz in things.items():
            d = np.linalg.norm(xyz - e0, axis=1)
            i = int(np.argmin(d))
            if d[i] <= see:
                near.append((float(d[i]), e, xyz[i],
                             tuple(int(round(float(c)))
                                   for c in xyz.mean(axis=0))))
        near.sort(key=lambda r: (r[0], r[1]))
        return [(e, c, mid) for _d, e, c, mid in near[:THINGS_SEEN]]

    # ── what is within reach ───────────────────────────────────────────────
    def within_reach(self, p, prop=None):
        """The named things a hand of body `p` could get to now, nearest
        first, that have `prop` (any, when None). Within reach is the test
        hands use (`_in_reach`, the arm this body has); the props are read
        off the thing's own cells (`_has`)."""
        own = p.get("_own")
        if not own:
            return []
        mine = np.array(sorted(own), np.int64)
        R = self._reach_of(p) / max(0.1 * self.scale, 1e-9)
        idx = self._thing_cells()
        mid = mine.mean(axis=0)
        out = []
        for e in sorted(idx):
            if self.things.get(e, {}).get("dead") is not None:
                continue
            xyz = np.stack(np.unravel_index(idx[e], self.shape), -1)
            if self._in_reach(mine, xyz, R) and (
                    prop is None or self._has(e, prop, p, xyz)):
                out.append((float(np.abs(xyz - mid).max(axis=1).min()), e))
        return [e for _d, e in sorted(out)]

    def has_within_reach(self, p, prop):
        """Whether a named thing within reach of `p` is `prop`: "sharp",
        "holds_liquid", "burning" or "liftable"."""
        return bool(self.within_reach(p, prop))

    def _has(self, e, prop, p, xyz):
        """Whether thing `e` (its cells `xyz`) is `prop`, for body `p`.

        Read here from the cells, as `props()` (3.3) will: sharp is an edge
        on any cell, burning is `burning_at` (no cache, no `_matter_gen`),
        holds_liquid is a hollow it walls in (`_cavity`, what a vessel can
        hold), liftable is a weight this body's arms can lift."""
        idx = tuple(xyz.T)
        if prop == "sharp":
            return bool((self.edge[idx] > 0).any())
        if prop == "burning":
            return bool(self.burning_at(idx).any())
        if prop == "holds_liquid":
            return bool(self.matter._cavity(xyz, need_fluid=False).any())
        if prop == "liftable":
            kg = float(self.smass[idx].sum()) / 1000.0
            return kg * GRAVITY <= self._strength(p)
        raise KeyError(f"no such property: {prop!r}")

    # ── MISREAD ────────────────────────────────────────────────────────────
    def _misread(self, v, ev, who, target, place):
        """A sighting of `ev` by person `v`, maybe misread: returns the
        (actor, target) the eyes report. Off unless `w.misread`.

        The draw is keyed on what was seen -- the world's seed, the viewer,
        the tick, the verb, the true actor, the place -- so the same sight
        is misread the same way every time it is replayed. A wrong actor is
        another person (not the viewer, not the doer); a wrong thing is
        another named thing. With no one else to mistake them for, the
        sight is right. Nor is the wrong actor ever the one the act was done
        to: "A hits B" may be seen as "C hits B", never as "B hits B"."""
        rates = MISREAD if self.misread is True else self.misread
        rate = rates.get(ev.verb, 0.0) if rates else 0.0
        if rate <= 0.0 or hash01(self.seed, v, self.tick, ev.verb, ev.actor,
                                 place) >= rate:
            return who, target
        if who is not None:
            hit = target[1] if target is not None \
                and target[0] == "person" else None
            others = [q["pid"] for q in self.persons
                      if q["pid"] not in (v, who, hit) and q["alive"]]
            if others:
                i = int(hash01(self.seed, v, self.tick, ev.id, "who")
                        * len(others))
                return others[i], target
        if target is not None and target[0] == "thing":
            others = [e for e, t in sorted(self.things.items())
                      if e != target[1] and t["dead"] is None
                      and self._thing_lut[e]]
            if others:
                i = int(hash01(self.seed, v, self.tick, ev.id, "thing")
                        * len(others))
                return who, ("thing", others[i])
        return who, target

    # ── CLAIMS ─────────────────────────────────────────────────────────────
    def _told(self, pid, ev, place):
        """Person `pid` made out the words of `ev`, a `says`. If its line is
        a claim and claims are on, believe it -- unless the claim had no
        grounds and an insight roll (`p["insight"]`, 0 to 1) sees through
        it. Returns True if it was believed.

        What the claim says was settled when the words were spoken
        (`_claim`, from `emit`), so every hearer is told the same thing. A
        line spoken while claims were off claims nothing, even if they are
        on by the time it is heard. `place` is not read: the claim already
        holds where the voice came from."""
        got = self._claimed.get(ev.id) if self.claims else None
        if got is None:
            return False
        key, value, asof, _true, grounded = got
        p = self.persons[pid]
        if not grounded and hash01(self.seed, pid, ev.id, "insight") \
                < p.get("insight", 0.0):
            return False
        return self._believe(pid, key, value, ev.id, "told", asof=asof)

    def _claim(self, ev_id, actor, line):
        """What a `says` of `line` by person `actor` claims, as the
        `(("grounded", g), ("true", t))` pairs `emit` adds to its data; ()
        when the line claims nothing (not in CLAIMS). `emit` calls this as
        the words are spoken (claims on), and the whole claim -- (key,
        value, asof, true, grounded) -- is kept in `_claimed` under the id
        the event is about to get, for `_told`.

        GROUNDED: the speaker had a first-hand belief of that kind ("a
        fire, there", from their own eyes). Then the claim passes on that
        belief -- its key, its value, and the tick it was true as of. A
        speaker with no such belief claims only that there is one, where
        the voice comes from (the speaker's head, where a hearer places the
        shout), as of now.
        TRUE: what was claimed is so as it is said (the fire it names, or
        any fire, is still on the books).

        Settled once, at the moment of speaking, and written into the
        event as it is made: the row never changes after it is in the log,
        so an `event_sink` copy and a hearer a tick later read the same."""
        kind = CLAIMS.get(line)
        if kind is None:
            return ()
        q = self.persons[actor] if actor is not None \
            and 0 <= actor < len(self.persons) else None
        bel = self.minds[actor].bel if q is not None else {}
        mine = [(b[3], k, b) for k, b in bel.items()
                if k[0] == kind and b[2] in FIRSTHAND and b[0] is not None]
        if mine:
            _t, key, b = max(mine, key=lambda r: (r[0], repr(r[1])))
            value, asof, grounded = b[0], b[3], True
            true = any(f[3] == key[1][1] for f in self._fires) \
                if key[1][0] == "fire" else bool(self._fires)
        else:
            eye = q.get("_eye") if q is not None else None
            key, value, asof, grounded = (kind, None), \
                (tuple(eye) if eye is not None else None), self.tick, False
            true = bool(self._fires)
        self._claimed = {k: v for k, v in self._claimed.items()
                         if self.events.get(k) is not None}
        self._claimed[ev_id] = (key, value, asof, bool(true), grounded)
        return (("grounded", grounded), ("true", bool(true)))


# ── the reader a mind is handed (V5) ───────────────────────────────────────
class View(dict):
    """What one body's mind holds now, as plain dicts (`view`).

    `view["percepts"]`: every row the body has perceived, oldest first, each
    shaped like `Percept` (t, channel, text, source, verb, actor, target,
    phase, kind, salience). `view["beliefs"]`: every belief, the one set
    longest ago first, each shaped like `Belief` (kind, about, value, text,
    source, how, t). A person, thing or place is a `Ref`-shaped dict (kind,
    id, name). It is a dict, so `json.dumps(view)` works as it is."""

    def recent(self, k=5):
        """The last `k` percepts, oldest first, one per source (a source
        perceived again counts once, at its last place: `mind.core.perceive`)."""
        got = {}
        for x in self["percepts"]:
            got.pop(x["source"], None)
            got[x["source"]] = x
        return list(got.values())[-k:] if k > 0 else []

    def beliefs_for(self, menu, k=2):
        """The at most `k` beliefs that bear on `menu`, the most relevant
        first (the rule of `mind.core.relevant`).

        `menu` is a limb's menu (option dicts with a "key", maybe a "who"),
        a dict of them by limb, or a list of option texts. A belief bears
        on it when a person or thing it names is named by an option. First
        those; then the beliefs that place nothing. A belief that places a
        thing no option names is never stated, nor one about the body
        itself (the scene says that), nor a voice. Fresher first."""
        if isinstance(menu, dict):
            menu = [o for opts in menu.values() for o in opts]
        texts, names = [], set()
        for o in menu:
            if isinstance(o, dict):
                texts.append(str(o.get("key", "")))
                if o.get("who"):
                    names.add(o["who"])
            else:
                texts.append(str(o))
        me = str(self["pid"])

        def named(ref):
            return isinstance(ref, dict) and ref["kind"] in ("person", "thing") \
                and (ref["name"] in names
                     or any(ref["name"] in t for t in texts))

        ranked = []
        for i, b in enumerate(self["beliefs"]):
            about, value = b["about"], b["value"]
            if b["kind"] in UNSTATED or (about["kind"] == "person"
                                         and about["id"] == me):
                continue
            locating = b["kind"] == "holds" or (b["kind"] == "at"
                                               and about["kind"] == "thing")
            if locating:
                thing = about if b["kind"] == "at" else value
                if not named(thing):
                    continue
                tier = 0
            else:
                tier = 0 if named(about) or named(value) else 1
            ranked.append((tier, -b["t"], i, b))
        return [b for *_, b in sorted(ranked, key=lambda r: r[:3])[:k]]


def view(w, p):
    """What body `p` (a person dict or pid) of world `w` has perceived and
    believes, in its own words: a `View`. Read-only: it changes nothing in
    the world, so a policy may be shown it (`observe`) every decision."""
    p = w.persons[p] if isinstance(p, int) else p
    me = p["pid"]
    m = w.minds[me]
    percepts = [_percept(w, r, p) for r in m.perceived]
    beliefs = [_belief(w, k, b, p) for k, b in m.bel.items()]
    return View(who=p["name"], pid=me, t=w.tick, percepts=percepts,
                beliefs=beliefs)


def _ref(w, what, me):
    """A Ref-shaped dict for a person pid, a target tuple, or None."""
    if what is None:
        return None
    if isinstance(what, int):
        what = ("person", what)
    kind = what[0]
    if kind == "person":
        q = w.persons[what[1]] if 0 <= what[1] < len(w.persons) else None
        return {"kind": "person", "id": str(what[1]),
                "name": q["name"] if q is not None else "someone"}
    if kind == "thing":
        return {"kind": "thing", "id": str(what[1]),
                "name": w.name(what[1], "something")}
    if kind == "fire":
        return {"kind": "fire", "id": str(_rid(what[1])), "name": "a fire"}
    if kind == "mat":
        cell = ",".join(str(int(c)) for c in what[2])
        return {"kind": "thing", "id": f"mat:{cell}",
                "name": w._what_words(what, me)}
    return {"kind": str(kind), "id": str(what[1:]), "name": "something"}


def _place(w, at, p):
    """A Ref-shaped dict for a place (cells), named from body `p`."""
    if at is None:
        return None
    return {"kind": "place", "id": ",".join(str(int(round(float(c))))
                                            for c in at),
            "name": w._where_words(at, p) or "somewhere"}


def _source(ev, how, tick):
    ev = _rid(ev)
    return str(ev) if ev is not None else f"{how}@{tick}"


def _percept(w, r, p):
    """One percept row (`Rec`) as a `Percept`-shaped dict."""
    me = p["pid"]
    data = dict(r.data)
    kind = ""
    if r.how == "felt" and r.verb in HARM:
        kind = "harm_to_me"
    elif r.how == "saw" and r.verb in VIOLENCE:
        kind = "violence_seen"
    elif r.how == "heard" and r.verb == "says" \
            and MEANS.get(data.get("line")) == "hears_alarm":
        kind = "alarm_heard"
    elif r.how in ("saw", "looked") and (
            r.verb in FIRE or (r.target_seen or ("",))[0] == "fire"):
        kind = "fire_seen"
    return {"t": r.tick, "channel": CHANNEL.get(r.how, "saw"),
            "text": w.render(r, p), "source": _source(r.ev, r.how, r.tick),
            "verb": r.verb or "", "actor": _ref(w, r.actor_seen, me),
            "target": _ref(w, r.target_seen, me) if r.how != "felt"
            else _ref(w, me, me),
            "phase": "done", "kind": kind, "salience": 1.0 if kind else 0.0}


def _belief(w, key, b, p):
    """One belief, `key -> (value, ev, how, tick)`, as a `Belief`-shaped
    dict."""
    me = p["pid"]
    value, ev, how, t = b
    kind, it = key[0], key[1]
    about = _ref(w, it, me) if it is not None else \
        {"kind": "fire", "id": "?", "name": "a fire"}
    an = "you" if it == me else about["name"]
    if kind == "at":
        val = _place(w, value, p)
        text = f"{an} {'are' if an == 'you' else 'is'} " \
            f"{val['name'] if val else 'somewhere'}"
    elif kind == "holds":
        val = _ref(w, value, me) if isinstance(value, tuple) else (
            "something" if value is True else "nothing")
        text = f"{an} {'hold' if an == 'you' else 'holds'} " \
            f"{val['name'] if isinstance(val, dict) else val}"
    elif kind == "state":
        val = str(value)
        text = f"{an} {STATE_WORDS.get(val, 'is ' + val)}"
    elif kind == "hurt_by":
        val = _ref(w, value, me)
        text = f"{'you' if value == me else val['name']} hurt {an}"
    elif kind == "alight":
        val = _place(w, value, p) if value is not None else "out"
        if value is None:
            text = f"{an} has gone out" if about["kind"] == "fire" \
                else f"{an} is no longer burning"
        elif about["kind"] == "fire":
            text = f"a fire is burning {val['name']}"
        else:
            text = f"{an} is burning"
    elif kind == "said":
        q = w.persons[it] if isinstance(it, int) \
            and 0 <= it < len(w.persons) else {}
        val = str(q.get("lines", LINES).get(value, LINES.get(value, value)))
        text = f'{an} said "{val}"'
    elif kind == "voice":
        val = "known"
        text = f"you know {an}'s voice"
    else:
        val, text = str(value), f"{an}: {value}"
    return {"kind": kind, "about": about, "value": val, "text": text,
            "source": _source(ev, how, t), "how": HOW.get(how, how), "t": t}
