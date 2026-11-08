"""The voxel mind link: the mind interface (src/mind) choosing for bodies in
the voxel world (docs/sim-mind-design.md 3.5, step M8).

The mind works with TASKS ("take the iron", "strike the defender"); the
voxel world works with per-limb menus, one decision a reaction time. This
file joins them, as src/rooms joins the inn, and keeps control inverted: the
world steps, and at each decision of a body the will layer calls the
policy's hooks (will.py `_decide`).

    VoxelBackend   the world as a Backend of the mind interface (api.py):
                   bind, rule_scores, self_state, percepts, beliefs, start,
                   and the job's life: its step tests and its interrupts
    MindPolicy     the policy a world is given (`w.policy`). Its hooks:
                   prepare   every change of state a decision needs, in order:
                             the running job's step test, an interrupt by
                             news, a question to the mind when the job is
                             over, and the step compiled into `p["intent"]`
                   limbs     the limbs the step needs, and no other: at most
                             one, so a woken decide never stops a walk
                   pick      PURE: it finds the row of the step BY PAYLOAD
                   on_restore  forgets what a restore undid

THE LINK NEEDS THE "onset" ATTENTION LAW (3.5). Under "latch" the first
percept commits a body and it stops deciding until its goal ends; a mind
must be asked again when its job ends, and when news comes.

A JOB is a catalog verb with its roles filled (api.Job). Its program on this
backend (verbs.yaml, `backends: voxel`) is a list of STEPS: the limb to ask,
the menu row to match, and the step test that ends the step. A step asks its
limb once a decide period until the act is carried out. The step test reads
TYPED EVENTS, the body's own since the step was first asked (3.0, ruling 5),
never a flag: "a `takes` event by me". A step whose act has not happened in
PERIODS decide periods fails the job, and the mind is asked again.

MATCHED BY PAYLOAD, NEVER BY LABEL. The words for a place change as a body
moves ("the door to the east", then "the door to the north-east"), so a
step finds its row by what the row is about: the goal cell of a walk, the
cell or the eid of a thing to take, the tag of a swing (will.py `_wants`).
`p["intent"]["want"]` keeps that row on the menu when the cap cuts it.

FALLING DECIDES GO TO THE REFLEX TABLE. A body in the air is asked how to
meet the ground (`_decide_falling`), with no prepare and no job; the reflex
table answers it. Then the second pick of `recall_check` changes nothing,
because `pick` is pure.

A STRIKE IS MADE FROM A STAND-OFF (step M10). The catalog says a strike's
target is in reach of a band the backend measures (`reach: strike_band`).
Here the band is where a swing lands a real blow (`STAND_OFF_M`), and it
depends on how the striker stands. A person out of it is still offered:
the job then starts with an APPROACH step, the legs walking to the spot the
band says (`intent["seek"]`, a `go:seek` row of the legs' menu, motion.py
`_places`), until the target is in the band (`in_reach`).

NEWS COMES WHEN IT HAPPENS, AND IS ANSWERED A REACTION LATER. A blow felt
wakes the body on the tick it lands (`intent["felt"]`, will.py `_attend`),
so the job stops and the mind is asked then; a job started for news the
will layer did not wait on (a row of what the body perceived) asks its
first limb only a reaction time after the news (`not_before`). A harmful
job that starts in view of others is a percept to them: a start percept
(`sees_attack`), as the inn has start records.

A REACTION IN STAGES IS ONE DECISION (will.py `_react`). A surprised body
first flinches (only the eyes are asked: the table answers), and a body
making sense of an alarm is asked again when its pre-movement is over (the
"go"): the link answers the percept once, at the stage in between, and the
legs do not leave before the "go".

THE MOTIVE IS A CHARACTER SHEET (3.5, "Motive in voxel"). Voxel bodies have
no traits; `p["sheet"] = {"harm": {name: weight}, "traits": {...}}` drives
the rule scores (`rule_scores`). Traits are not read yet: how they bias
strike against flee is question Q5, so the score of an average character
is the character's own.

NO GATE HIDES AN OPTION (step M12). `bind` filters only by fact: the doors
it has seen and has a route to, the things in reach, whom it can strike from
here or walk to strike. Whether the body wants any of it is the rule
scores': an act whose drives it does not have now scores NO_REASON, and a
calm man walks out, rarely, by choice. A mind with `MindConfig(menu="two")`
asks a scope first, then an act in it (src/mind/scope.py).

The rows a linked mind's body writes to `w.traces` say `by: "mind"`. They are
the steps of a job, matched by payload, not choices made over that menu, so
a harvest skips them (harvest.py). The mind's own decisions are `Mind.rows`,
and `record.Recorder` keeps their menus for the lab (`MindPolicy(record=True)`).
"""
from __future__ import annotations

import math

import numpy as np

from src.mind import core
from src.mind.api import Belief, Job, JobEvent, Percept, Ref, SelfState
from src.mind.loop import Mind, MindConfig
from src.mind.record import Recorder

from .beliefs import HARM, view, _percept
from .events import _rid
from .materials import CHEST0, FLESH, TICK_S
from .policy import LIMBS, LINES, ORIENTING, WILL, TablePolicy, sheet

NAME = "voxel"                  # the key of this backend in the verb catalog
# A STEP FAILS when its act has not happened this many decide periods after
# its limb was first asked (3.5).
PERIODS = 3
# THE STAND-OFF BAND: how far apart two bodies stand, centre to centre in
# metres, when a swing started now lands at least half the energy of the
# best blow that swing can land. A swing builds speed through its arc, so
# nearer is a blow with nothing behind it (too close to cut), and further
# is no blow at all. tests/test_mindlink.py calibrates it: it moves where
# the defender of tests/test_wounds.py `_duel` stands, and reads the joules
# of the first blow. Measured at 5 cm, with the duel's iron:
#
#     side, held     0.55 m 15.9 J   0.60 21.3   0.65 27.0   0.70 38.6   0.75 none
#     square, held   0.25 m 17.4 J   0.30 20.7   ...         0.45 38.2   0.50 none
#     side, bare     0.45 m 12.9 J   0.50 19.1   0.55 25.0   0.60 30.4   0.65 none
#     square, bare   0.25 m 12.5 J   0.30 20.9   ...         0.45 30.5   0.50 none
#
# It depends on the STANCE (`_stance`): "square" is the chest along the
# line to the other body, "side" is the shoulders along it with the
# swinging arm's shoulder the nearer one. The arm then comes round in front
# of the chest and reaches 25-30 cm further. With the FAR arm the swing goes
# through the swinger's own body, so that stance has no band. It depends on
# WHAT THE HAND HOLDS (the second key): from the side, a thing in the fist
# reaches 10 cm further than the fist. And the other body's stance moves it
# by a cell: a defender squared up to the swing is met 5 cm nearer from the
# side. A band is where the blow holds for both ways the other can stand.
STAND_OFF_M = {("side", True): (0.60, 0.65), ("square", True): (0.30, 0.45),
               ("side", False): (0.50, 0.55), ("square", False): (0.30, 0.45)}
# SQUARE is a chest within 45 degrees of the line to the other body.
SQUARE_COS = math.sqrt(0.5)
# The rule scores of the verbs whose score is a number (3.5, the table of
# "Motive in voxel"). A drop is not in the table: nothing makes a body want
# to let go of what it holds, so it scores nothing.
SCORE = {"walk_to": 0.1, "wait": 0.2, "drop": 0.0}
# AN ACT WITH NO REASON (step M12). Every option is offered, whatever the
# body wants: a way out, a warning, a walk over to strike someone (the
# gates of M8 and M10 that hid them are gone: the engine does not choose for
# the character). What the character wants is in its rule scores. An act
# that serves a drive (catalog `serves`) scores from that drive; when the
# body has none of the drives it serves now, it scores NO_REASON: a calm man
# can still walk out of the door, by choice, rarely. Against a wait (0.2) at
# the rules' T = 0.2 one such option is drawn e^((-1.0 - 0.2) / 0.2) = 1/403
# as often as the wait. A walk out of the door here is a way out: it serves
# safety, as a flight does. (Offered with their old scores, a walk 0.1 or a
# strike 0 against a wait 0.2 was close to a coin: the duel's defender
# walked out, or over to punch the attacker, at about a third of his
# decisions, M8 and M10.)
NO_REASON = -1.0
LEAVE = frozenset({"walk_to", "flee_to"})       # every walk the link offers is to a way out
# The event that says a step's act was carried out, by the tag (or verb) of
# the row the step matches: the swing was swung, the thing taken.
ACT = {"take": "takes", "swing": "swings", "let_go": "releases",
       "say:fire": "says", "go": "goes", "go:seek": "goes"}
# The limbs each stage of a reaction asks (will.py `_decide`): a flinch the
# eyes, the "go" after a pre-movement the legs and hands. While a body still
# makes sense of an alarm ("premove") its legs stay where they are.
PHASE_LIMBS = {"flinch": frozenset({"eyes"}), "go": frozenset({"legs", "hands"}),
               "premove": frozenset(LIMBS) - {"legs"}}
# The INTERRUPT CLASS of a percept of the will layer (catalog
# `interrupts_known`). A percept not named here is news of the class "any".
# An ORIENTING percept (a noise) is never news to a job: it turns the head.
CLASS = {"sees_fire": "fire_seen", "scorched": "fire_seen",
         "hears_alarm": "alarm_heard"}
# The word a reason gives for news found in what a body perceived: being hit
# is being "struck" (the voxel test of the drive safety says so too).
WORD = {"harm_to_me": "struck", "violence_seen": "sees_violence",
        "alarm_heard": "hears_alarm", "fire_seen": "sees_fire"}
# Which news stops a job first, when two come in one decide.
FIRST = ("harm_to_me", "fire_seen", "alarm_heard", "violence_seen", "any")


def catalog():
    """The verb catalog, read once (the one the mind reads)."""
    return core.catalog()


def _refof(d):
    """A Ref-shaped dict of `beliefs.view` as a Ref, or None."""
    return None if d is None else Ref(d["kind"], d["id"], d["name"])


class VoxelBackend:
    """The voxel world as a Backend of the mind interface. `who` is a
    person's pid, as a string. The job a person runs is `p["job"]`, so a
    snapshot copies it with the person (it is plain data, api.Job)."""
    name = NAME

    def __init__(self, w, cat=None):
        self.w = w
        self.cat = cat or catalog()
        self.job_events = []    # each step in the life of each job
        # THE START RECORDS: a harmful job as it starts, and who saw it start
        # (`_seers`): {"t", "id", "who", "target", "verb", "text", "seen"}
        self.starts = []
        self._bound = {}        # who -> (tick, options, {key: payload}) of its last bind
        self._view = None       # (tick, pid, beliefs.View) of the last view
        test = self.cat.drive_test("safety", NAME) or {}
        self._safety = frozenset(test.get("percept_any", ()))
        caps = {c: self.cat.cap_test(c, NAME) for c in ("sharp", "heavy")}
        self._sharp = (caps["sharp"]["edge_gt_m2"], caps["sharp"]["edge_max_m2"])
        self._heavy_g = 1000.0 * caps["heavy"]["mass_min_kg"]
        # what a weapon is for: the drives the verb it is a weapon of serves
        self._armed = self.cat.verbs["strike"].serves

    # ------------------------------------------------------------ people
    def person(self, who):
        return self.w.persons[int(who)]

    def _cells(self, p):
        """The body's cells on the lattice, or None while it is in the air."""
        comp, sl = self.w._person_cells(p)
        if comp is None:
            return None
        cells = np.argwhere(comp)
        cells[:, 0] += sl[0].start
        cells[:, 1] += sl[1].start
        return cells

    def needs_decision(self):
        """(who, reason) for each body that can decide and runs no job. A
        world in motion asks at its own time (MindPolicy.prepare); this is
        the Backend protocol's view of the same fact."""
        return [(str(p["pid"]), self._reason(p.get("job")))
                for p in self.w.persons
                if p["alive"] and p["awake"] and not p["safe"]
                and (p.get("job") is None or p["job"].status != "running")]

    @staticmethod
    def _reason(job):
        """Why a body whose job is over chooses now."""
        if job is None:
            return "idle"
        if job.status == "done":
            return "done"
        return f"failed:{job.reason}" if job.status == "failed" else job.reason

    # ------------------------------------------------------------ what a body knows
    def _seen(self, p):
        """`beliefs.view` of `p`, built once a tick."""
        got = self._view
        if got is None or got[0] != self.w.tick or got[1] != p["pid"]:
            got = self._view = (self.w.tick, p["pid"], view(self.w, p))
        return got[2]

    def fears(self, p):
        """The percepts of the drive safety (its voxel test, `percept_any`)
        that this body has now: what it noticed in the last decide period,
        a blow it felt and still holds in mind ("struck"), and a fire it
        believes is burning."""
        w, m = self.w, self.w.minds[p["pid"]]
        got = set()
        if m.noticed is not None and w.tick - m.noticed[0] <= WILL["decide_every"]:
            got.add(m.noticed[1])
        if any(r.how == "felt" and r.verb in HARM for r in m.perceived):
            got.add("struck")
        if any(k[0] == "alight" and b[0] is not None for k, b in m.bel.items()):
            got.add("sees_fire")
        return got & self._safety

    def _harm(self, p):
        """{name: weight} of the sheet, for the people here, alive and in
        the world."""
        harm = (p.get("sheet") or {}).get("harm") or {}
        here = {q["name"] for q in self.w.persons
                if q is not p and q["alive"] and not q["safe"]}
        return {n: float(v) for n, v in harm.items() if n in here and v > 0}

    def self_state(self, who):
        w, p = self.w, self.person(who)
        holds = (p.get("dragging"),) if p.get("dragging") else ()
        holds += tuple(h[0].name for h in self._holds(p))
        afraid, harm = self.fears(p), self._harm(p)
        feels = [word for word, on in (
            ("hurt", p["hurt"] > 0.0), ("short of breath", p["smoke"] > 0.05),
            ("bleeding", any(wd.get("open") for wd in p.get("wounds") or ())),
            ("afraid", bool(afraid))) if on]
        if harm:
            feels.append("angry at " + " and ".join(sorted(harm)))
        drives = tuple(d for d, on in (("safety", bool(afraid)), ("harm", bool(harm)))
                       if on)
        return SelfState(p["name"], holds=holds, feels=tuple(feels),
                         drives=tuple(d for d in self.cat.drives if d in drives))

    def percepts(self, who):
        """What the body perceived, in its own words (`beliefs.view`)."""
        out = []
        for x in self._seen(self.person(who))["percepts"]:
            out.append(Percept(x["t"], x["channel"], x["text"], x["source"], x["verb"],
                               _refof(x["actor"]), _refof(x["target"]), x["phase"],
                               x["kind"], x["salience"]))
        # and the starts it saw, in the order things happened
        out += [self._start_percept(s, int(who)) for s in self.starts if int(who) in s["seen"]]
        return tuple(sorted(out, key=lambda x: x.t))

    def _start_percept(self, s, me):
        """A start record as the body `me` saw it: "attacker raised a hand
        against you". A harmful act aimed at me is harm coming (harm_to_me),
        any other one is violence seen (catalog `interrupts_known`)."""
        w = self.w
        a, t = w.persons[s["who"]], w.persons[s["target"]] if s["target"] is not None else None
        names = {r: ("you" if r == "target" and s["target"] == me else n)
                 for r, n in s["roles"].items()}
        text = self.cat.verbs[s["verb"]].told_text(a["name"], names, start=True)
        return Percept(s["t"], "saw", text, s["id"], s["verb"],
                       Ref("person", str(a["pid"]), a["name"]),
                       None if t is None else Ref("person", str(t["pid"]), t["name"]),
                       "start", "harm_to_me" if s["target"] == me else "violence_seen", 1.0)

    def beliefs(self, who):
        """What the body believes (`beliefs.view`: where people and things
        are, who hurt whom, where a fire burns), and three beliefs of the
        link: each way out it has seen, and each place it tried and could
        not get to."""
        w, p = self.w, self.person(who)
        out = []
        for b in self._seen(p)["beliefs"]:
            v = b["value"]
            out.append(Belief(b["kind"], _refof(b["about"]),
                              _refof(v) if isinstance(v, dict) else str(v),
                              b["text"], b["source"], b["how"], b["t"]))
        for xy, name in self._doors(p):
            out.append(Belief("way_out", Ref("place", f"{xy[0]},{xy[1]}", name), "known",
                              f"{name} is a way out", "sight", "looked", 0))
        for xy, t in sorted((p.get("_baulked") or {}).items(), key=lambda kv: kv[1]):
            if w.tick - t >= WILL["baulk_for"]:
                continue                      # forgotten: the menu offers it again
            words = w._where_words(xy, p) or "close by"
            out.append(Belief("unreachable", Ref("place", f"{xy[0]},{xy[1]}", f"the spot {words}"),
                              "baulked", f"you could not get to the spot {words}", "own", "own",
                              int(t)))
        return tuple(out)

    def _doors(self, p):
        """((xy, name), ...): the exits this body has seen, named from where
        it stands, "the door to the east". Two that would be called the same
        are told apart by which is nearer, never in metres."""
        w = self.w
        ax, ay = p["anchor"]
        got = []
        for e in w.exits:
            xy = (int(e[0]), int(e[1]))
            if p["known"][xy]:
                way = w._bearing(xy[0] - ax, xy[1] - ay)
                got.append((max(abs(xy[0] - ax), abs(xy[1] - ay)), xy,
                            "the door " + ("here" if way == "here" else "to the " + way)))
        out, seen = [], {}
        for _, _, n in got:
            seen[n] = seen.get(n, 0) + 1
        nth = {}
        for d, xy, n in sorted(got):
            if seen[n] > 1:
                k = nth[n] = nth.get(n, 0) + 1
                n = n.replace("the door", f"the {_ORDINAL[min(k, len(_ORDINAL)) - 1]} door", 1)
            out.append((xy, n))
        return out

    # ------------------------------------------------------------ options
    def _rows(self, p, cells, limb, seek=None):
        """Every row of one limb's menu, uncut (the will layer's own
        legality sweep). Read-only: the count the sweep leaves behind is
        put back, and so is the intent a `seek` list stands in for (the
        spots to strike from that the legs are asked about, `_places`; and
        only those: not the one a running job asks for)."""
        w = self.w
        had = w.__dict__.get("_offered")
        intent = p.get("intent")
        if seek is not None:
            p["intent"] = dict(intent or {}, seek=seek)
        try:
            rows = w._menu(p, cells, None, limb, {}, cap=10 ** 6)
        finally:
            if seek is not None:
                if intent is None:
                    del p["intent"]
                else:
                    p["intent"] = intent
        if had is not None:
            w._offered = had
        return rows

    def _thing(self, cells=None, body=None):
        """(sharp, heavy) of a thing: its cells on the lattice, or the
        part of a flying body that is not flesh (a held thing mid-swing)."""
        w = self.w
        if cells is not None and len(cells):
            idx = tuple(np.asarray(cells).T)
            edges, grams = w.edge[idx], float(w.smass[idx].sum(dtype=np.float64))
        elif body is not None:
            s = body["mats"] != FLESH
            edges, grams = body["edges"][s], float(body["masses"][s].sum())
        else:
            return False, False
        lo, hi = self._sharp
        return bool(((edges > lo) & (edges <= hi)).any()), grams >= self._heavy_g

    def _holds(self, p):
        """[(Ref, sharp, heavy)] of every thing `p` has in its hands. A body
        has two hands, and the link counts what they hold as a list, so
        nothing here says "the one thing it holds". (The will layer gives a
        body one thing in a hand today, `p["held"]`, and a person it drags,
        which is not a thing; `self_state` lists both.)"""
        w = self.w
        if not p.get("held"):
            return []
        at = w._held_cells(p)
        if not p.get("held"):
            return []                         # (or it was, and it is gone)
        e = w._held_eid(p)
        name = w.name(e, "the " + p["held"]["label"], p)
        body = None
        if at is None:                        # in the air with the arm
            mine = [b for b in w.bodies if b.get("owner") == p["name"]]
            body = next((b for b in mine if b.get("part")), mine[0] if mine else None)
        x, y, z = p["held"]["cell"]
        tid = f"e{e}" if e else f"c{x},{y},{z}"
        return [(Ref("thing", tid, name),) + self._thing(at, body)]

    def _held_by_others(self, p):
        """The things other bodies hold: their eids, and their gripped cells."""
        w, eids, cells = self.w, set(), set()
        for q in w.persons:
            if q is p or not q.get("held"):
                continue
            e = w._held_eid(q)
            if e:
                eids.add(e)
            cells.add(tuple(int(c) for c in q["held"]["cell"]))
        return eids, cells

    # ------------------------------------------------------------ stand-off
    def _stance(self, p, cells, u):
        """How `p` stands to a swing toward the unit xy `u`: "square" (the
        chest along the line), "side" (the shoulders along it, the swinging
        arm's the nearer), or None (the swinging arm is the far one: the
        swing would go through its own body). The arm is the one the hands'
        menu would swing (will.py `_menu`): the one holding something, else
        a free hand."""
        w = self.w
        cx, cy = p.get("chest", CHEST0)
        if abs(cx * u[0] + cy * u[1]) >= SQUARE_COS:
            return "square"
        arm = (p.get("held") or {}).get("arm") or w._hand(p, cells, free=True) \
            or w._hand(p, cells)
        piv = w._pivot_of(p, arm) if arm is not None else None
        if piv is None:
            return None
        lo = cells.min(axis=0)
        sx = float(piv[0] + lo[0]) - float(cells[:, 0].mean())
        sy = float(piv[1] + lo[1]) - float(cells[:, 1].mean())
        return "side" if sx * u[0] + sy * u[1] > 0.0 else None

    def _reach(self, p, cells, q):
        """(the band `p` strikes `q` from now, or None; how far apart they
        stand, centre to centre, in metres; the unit xy from `p` to `q`)."""
        vox_m = 0.1 * self.w.scale
        dx, dy = q["anchor"][0] - p["anchor"][0], q["anchor"][1] - p["anchor"][1]
        n = math.hypot(dx, dy)
        if n < 1e-9:
            return None, 0.0, (1.0, 0.0)
        u = (dx / n, dy / n)
        stance = self._stance(p, cells, u)
        band = None if stance is None else STAND_OFF_M[stance, bool(p.get("held"))]
        return band, n * vox_m, u

    def _seek(self, p, cells, q):
        """The spot to strike `q` from, as the legs are asked for it
        (`intent["seek"]`, motion.py `_places`): who, where they stand, and
        the band to stand in from them, in cells -- the band of the stance
        the body has now. A body that walks free turns to face its way
        (`_square_up`), so it may arrive in another stance: it is asked
        again from there (`compile`). With its far arm toward them it has no
        band: free, it walks to the side band, and turns as it goes; holding
        something, it cannot turn, and has no spot to strike from (None)."""
        band = self._reach(p, cells, q)[0]
        if band is None:
            if p.get("held") or p.get("dragging"):
                return None
            band = STAND_OFF_M["side", False]
        vox_m = 0.1 * self.w.scale
        return {"who": q["name"], "pid": q["pid"],
                "at": [int(q["anchor"][0]), int(q["anchor"][1])],
                "band": [band[0] / vox_m, band[1] / vox_m]}

    def _in_band(self, p, q, cells=None):
        """Is `q` where `p` would strike it from, now: within its band, and
        in line along x or y? A swing comes round in the plane of one axis
        (pose.py `_swing`), beside the body, so a man a cell or two off that
        line is swung past (motion.py `_seek_spot`): the band was measured
        in line."""
        cells = self._cells(p) if cells is None else cells
        if cells is None:
            return False
        dx, dy = abs(q["anchor"][0] - p["anchor"][0]), abs(q["anchor"][1] - p["anchor"][1])
        if min(dx, dy) > 1:
            return False
        band, d, _ = self._reach(p, cells, q)
        return band is not None and band[0] - 1e-9 <= d <= band[1] + 1e-9

    def bind(self, who):
        """Every option of the catalog's voxel verbs for `who` now: a walk
        or a flight to each exit it has seen and has a route to, a take of
        each thing in reach that nobody holds (keyed by eid, or by cell for
        a thing with no name) while it has a free hand, a drop of each thing
        it holds, a strike at each person in the stand-off band, or with a
        route to a spot in it (with what it holds, when that is sharp or
        heavy), a warning shouted when it has the words, and waiting. Only
        facts filter: what it wants is its rule scores' business."""
        w, p = self.w, self.person(who)
        opts, pay = [], {}

        def add(verb, roles, payload, duration=None):
            o = self.cat.option(NAME, verb, roles, duration=duration)
            if o.key not in pay:
                opts.append(o)
                pay[o.key] = payload

        cells = self._cells(p)
        if cells is not None:
            vox_m = 0.1 * w.scale
            # EVERY WAY OUT IT HAS SEEN, as a walk and as a flight. Here a
            # walk to a door is not a walk across a room: the body goes out
            # (`safe`) and is gone from the scene. Whether it wants to is
            # its rule score (NO_REASON)
            doors = dict(self._doors(p))
            # WHOM IT COULD STRIKE, and from where: in the band now, or a
            # walk away from it (`_seek`), each walk asked of the legs' menu
            # in the same sweep as the doors
            near, seeks = {}, []
            for q in w.persons:
                if q is p or not q["alive"] or q["safe"]:
                    continue
                if self._in_band(p, q, cells):
                    near[q["pid"]] = None
                    continue
                s = self._seek(p, cells, q)
                if s is not None:
                    seeks.append(s)
            legs = self._rows(p, cells, "legs", seek=seeks) if doors or seeks else ()
            whose = {s["who"]: s["pid"] for s in seeks}
            for r in legs:
                if r["verb"] != "go":
                    continue
                if r["tag"] == "go:seek":
                    near.setdefault(whose[r["who"]], len(r["route"]))
                    continue
                xy = tuple(int(c) for c in r.get("goal") or ())
                if xy not in doors:
                    continue
                place = Ref("place", f"{xy[0]},{xy[1]}", doors[xy])
                for verb, mps in (("walk_to", WILL["walk_mps"]), ("flee_to", WILL["run_mps"])):
                    ticks = max(1, int(math.ceil(len(r["route"]) * vox_m / (mps * TICK_S))))
                    add(verb, {"place": place}, {"goal": xy, "danger": float(r.get("danger", 0.0))},
                        duration=ticks)
            holds = self._holds(p)
            # A FREE HAND TAKES (the catalog's `free_hand`), whatever the
            # other one holds; which things it is offered is the hands'
            # menu's business
            if w._hand(p, cells, free=True) is not None:
                eids, gripped = self._held_by_others(p)
                for r in self._rows(p, cells, "hands"):
                    if r["tag"] != "take":
                        continue
                    cell = tuple(int(c) for c in r["what"]["cell"])
                    e = w.entity_at(*cell)
                    at = w._object_at(*cell)
                    if (e and e in eids) or (gripped and gripped & set(map(tuple, at.tolist()))):
                        continue              # in someone's hand: not a thing to take
                    tid = f"e{e}" if e else f"c{cell[0]},{cell[1]},{cell[2]}"
                    sharp, heavy = self._thing(at)
                    add("take", {"target": Ref("thing", tid, r["key"][len("take hold of "):])},
                        {"cell": cell, "eid": e, "weapon": sharp or heavy})
            for h in holds:
                add("drop", {"target": h[0]}, {})
            # what it strikes with: a sharp thing before a heavy one
            weapon = max((h for h in holds if h[1] or h[2]), key=lambda h: (h[1], h[2]),
                         default=None)
            dur = self.cat.verbs["strike"].duration(NAME)
            for q in w.persons:
                if q["pid"] not in near:
                    continue
                roles = {"target": Ref("person", str(q["pid"]), q["name"])}
                if weapon is not None:
                    roles["instrument"] = weapon[0]
                route = near[q["pid"]]
                walk = 0 if route is None else int(math.ceil(
                    route * vox_m / (WILL["walk_mps"] * TICK_S)))
                add("strike", roles, {"pid": q["pid"], "approach": route is not None},
                    duration=dur + walk)
            # and A WARNING, when it has the words. A body that hears one
            # runs (`hears_alarm`); it serves safety, so with nothing to
            # fear it scores NO_REASON
            if (p.get("lines", LINES) or {}).get("fire"):
                add("shout_fire", {}, {})
        add("wait", {}, {})
        self._bound[who] = (w.tick, tuple(opts), pay)
        return tuple(opts)

    def _payload(self, who, option):
        when, _, pay = self._bound.get(who, (None, (), {}))
        if when != self.w.tick or option.key not in pay:
            raise ValueError(f"{option.text!r} is not an option of {who} now (bind again)")
        return pay[option.key]

    def rule_scores(self, who, options):
        """(prior, prior_avg, habit) of each option, from the sheet (3.5):

            strike X   harm[X] x (1.5 armed, else 0.5)
            take       0.8 x the drive the thing serves (a weapon serves
                       what a strike serves: harm and safety)
            flee       safety x (1 - the danger of the route)
            shout      safety, when others can hear
            walk 0.1, wait 0.2, drop 0

        safety is 1 while the body has a percept of the drive (`fears`),
        else 0. An act whose drives the body does not have now scores
        NO_REASON: a strike at X with no harm meant to X and nothing to
        fear, and a walk out, a flight or a warning with nothing to fear.
        Traits are not read (Q5), so prior_avg is prior. Habit is 0."""
        w, p = self.w, self.person(who)
        harm = (p.get("sheet") or {}).get("harm") or {}
        safety = 1.0 if self.fears(p) else 0.0
        drive = {"harm": max((float(v) for v in harm.values()), default=0.0), "safety": safety}
        heard = None
        out = []
        for o in options:
            v, pay = o.verb, self._payload(who, o)
            if v == "strike":
                want = float(harm.get(o.role("target").name, 0.0))
                s = want * (1.5 if o.role("instrument") else 0.5) if want > 0 or safety else NO_REASON
            elif v in LEAVE and not safety:
                s = NO_REASON
            elif v == "shout_fire" and not safety:
                s = NO_REASON
            elif v == "take":
                s = 0.8 * max((drive.get(d, 0.0) for d in self._armed), default=0.0) \
                    if pay["weapon"] else 0.0
            elif v == "flee_to":
                s = safety * (1.0 - pay["danger"])
            elif v == "shout_fire":
                if heard is None:
                    eye = p.get("_eye", (*p["anchor"], 0.0))
                    heard = any(q is not p and q["alive"] and q["awake"]
                                and w._hears(eye, q.get("_eye", (*q["anchor"], eye[2])))
                                for q in w.persons)
                s = safety if heard else 0.0
            else:
                s = SCORE[v]
            out.append(s)
        prior = tuple(out)
        return prior, prior, (0.0,) * len(prior)

    # ------------------------------------------------------------ a job
    def start(self, who, option, why, menu=None, probs=None, probs_avg=None):
        """Start `option` as `who`'s job. It is checked again against the
        world as it is now: an option of a bind made in another tick is
        bound again. The program of the verb becomes the job's steps; a
        strike at someone out of the band starts with the walk to it."""
        w, p = self.w, self.person(who)
        if self._bound.get(who, (None,))[0] != w.tick:
            self.bind(who)
        pay = self._payload(who, option)
        if menu is not None and option not in menu.options:
            raise ValueError(f"{option.text!r} is not on the menu its chooser saw")
        dur = option.duration
        steps = [self._step(p, option, pay, st, dur) for st in self.cat.program(option.verb, NAME)]
        if pay.get("approach"):
            steps.insert(0, self._approach(w.persons[pay["pid"]]))
        state = {"steps": steps, "i": 0, "checked": w.tick, "starts": len(self.starts),
                 "known": [self._noticed(p)] if self._noticed(p) else []}
        # AN ANSWER TAKES A REACTION TIME (rows P12-P16), counted from the
        # news. The will layer waits it out for its own percepts (`_react`);
        # news it did not wait on -- a blow felt, a start seen -- stopped the
        # last job on the tick it came, so the limbs wait here
        last = p.get("job")
        news_at = None if last is None or last.status != "stopped" or last.ended != w.tick \
            else last.state.get("news_at")
        if news_at is not None:
            due = news_at + w._ticks(sheet(p, "ready_s" if w._primed(p) else "surprise_s"))
            # (the answer to news it was already answering is not put off:
            # a second blow does not restart the reaction to the first)
            was = last.state.get("not_before")
            if was is not None and was > w.tick:
                due = min(due, was)
            if due > w.tick:
                state["not_before"] = due
        job = Job(why.decision, who, option, w.tick, why, state=state)
        p["job"] = job
        verb = self.cat.verbs[option.verb]
        said = ""
        if "loud" in verb.flags or "violent" in verb.flags:
            said = verb.told_text(p["name"], {r: ref.name for r, ref in option.roles}, start=True)
        self.job_events.append(JobEvent(w.tick, who, job.id, option.verb, "start", said, why.reason))
        # A HARMFUL ACT IS SEEN TO START: a raised arm, a step in. Whoever
        # has the actor in view now holds it as a start percept
        # (`sees_attack`), as the inn keeps start records
        if "harmful" in verb.flags:
            t = option.role("target")
            seen = frozenset(self._seers(p))
            self.starts.append({"t": w.tick, "id": f"start:{job.id}", "who": p["pid"],
                                "target": int(t.id) if t is not None and t.kind == "person"
                                else None, "verb": option.verb,
                                "roles": {r: ref.name for r, ref in option.roles},
                                "seen": seen})
            # and it is news to them now, not when their own jobs next look
            for q in w.persons:
                if q["pid"] in seen and q.get("intent") is not None:
                    q["intent"]["wake_at"] = min(q["intent"].get("wake_at", w.tick), w.tick)
        return job

    def _seers(self, p):
        """The pids of the bodies that have `p` in view now: awake, within
        sight (`see_m`), `p`'s head in their field of view, and a clear line
        from head to head (the will layer's own test of who sees whom,
        `_runners`)."""
        w = self.w
        eye = p.get("_eye")
        if eye is None:
            return ()
        far = WILL["see_m"] / max(0.1 * w.scale, 1e-9)
        cand = [q for q in w.persons
                if q is not p and q["alive"] and q["awake"] and not q["safe"]
                and not q.get("asleep") and q.get("_eye") is not None
                and math.hypot(q["_eye"][0] - eye[0], q["_eye"][1] - eye[1]) <= far
                and w._in_cone(q.get("gaze") or q.get("facing", (1.0, 0.0)), q["_eye"], eye)]
        if not cand:
            return ()
        ok = w.rays([q["_eye"] for q in cand], [eye] * len(cand),
                    skip=[(q["pid"], p["pid"]) for q in cand])
        return tuple(q["pid"] for q, seen in zip(cand, ok) if seen)

    def _approach(self, q):
        """The step that walks to the stand-off band of `q` (`in_reach`):
        the legs asked for the spot to strike from (`go:seek`), matched by
        whom it is about."""
        return {"limb": "legs", "until": "in_reach", "want": {"tag": "go:seek", "who": q["name"]},
                "take": None, "act": ACT["go:seek"], "aim": q["pid"], "flee": False,
                "dur": None, "ends": None, "asked": None, "last": None, "from": None,
                "acted": False}

    def _step(self, p, option, pay, st, dur):
        """One step of a program, as plain data the job keeps."""
        m, limb = st.get("match"), st["limb"]
        if m is not None and m not in ("goal", "cell") and not isinstance(m, dict):
            raise ValueError(f"voxel steps match no {m!r} yet")
        if st["until"] not in ("arrived", "took", "swung", "said", "wake"):
            raise ValueError(f"voxel has no step test {st['until']!r} yet")
        want = take = None
        if m == "goal":
            want = {"verb": "go", "goal": pay["goal"]}
        elif m == "cell":
            take = pay["eid"] or pay["cell"]
            want = {"tag": "take", "eid": pay["eid"]} if pay["eid"] else \
                {"tag": "take", "cell": pay["cell"]}
        elif m is not None:
            want = m["tag"]
        aim = option.role(st["aim"]) if st.get("aim") else None
        return {"limb": limb, "until": st["until"], "want": want, "take": take,
                "act": None if limb is None else ACT[want if isinstance(want, str)
                                                     else want.get("tag", want.get("verb"))],
                "aim": None if aim is None else int(aim.id), "flee": bool(st.get("flee")),
                "dur": dur, "ends": None if limb is not None else self.w.tick + dur,
                "asked": None, "last": None, "from": None, "acted": False}

    def _noticed(self, p):
        """The will layer's percept of this tick and where it came from, as
        a key a job can remember, or None."""
        n = self.w.minds[p["pid"]].noticed
        if n is None or n[0] != self.w.tick or n[1] in ORIENTING:
            return None
        return [n[1], str(_rid(n[2]))]

    def _end(self, p, job, status, reason=""):
        w = self.w
        job.status, job.ended, job.reason = status, w.tick, reason
        text = ""
        if status == "done":
            text = self.cat.verbs[job.option.verb].told_text(
                p["name"], {r: ref.name for r, ref in job.option.roles})
        self.job_events.append(JobEvent(w.tick, job.who, job.id, job.option.verb, status, text,
                                        reason))

    def advance(self, p, percept, phase=None):
        """The running job's step test, then news that stops it. Returns why
        the body must choose now, or None while the job runs on."""
        job = p.get("job")
        if job is None:
            return "idle"
        if job.status == "running":
            self._test(p, job)
        if job.status == "running":
            news = self._news(p, job, percept, phase)
            if news is not None:
                self._end(p, job, "stopped", f"interrupt:{news}")
        return None if job.status == "running" else self._reason(job)

    def _test(self, p, job):
        """Run the step tests of the running job, from its current step on."""
        w, st = self.w, job.state
        while job.status == "running":
            step = st["steps"][st["i"]]
            done, failed = self._done(p, step)
            if failed:
                self._end(p, job, "failed", failed)
                return
            if not done:
                if step["limb"] is not None and not step["acted"] and step["asked"] is not None \
                        and w.tick - step["asked"] >= PERIODS * WILL["decide_every"]:
                    self._end(p, job, "failed", f"no {step['until']} after asking")
                return
            st["i"] += 1
            if st["i"] >= len(st["steps"]):
                self._end(p, job, "done")

    def _done(self, p, step):
        """(done, why it failed or ""): the step test, read off the body's own
        events since its limb was first asked."""
        w, me = self.w, p["pid"]
        until = step["until"]
        if until == "in_reach":
            # IN REACH is a fact about two bodies, not an act: done the
            # moment the other is in the band and the legs are still,
            # however that came about
            q = w.persons[step["aim"]]
            if not q["alive"] or q["safe"]:
                return False, "gone"
            if p.get("goal") is None and self._in_band(p, q):
                return True, ""
            if step["from"] is None:
                return False, ""
            mine = [ev for ev in w.events.since(step["from"]) if ev.actor == me]
            step["acted"] = step["acted"] or any(ev.verb == step["act"] for ev in mine)
            return False, "gave up" if any(ev.verb == "gives_up" for ev in mine) else ""
        if step["limb"] is not None:
            if step["from"] is None:
                return False, ""
            mine = [ev for ev in w.events.since(step["from"]) if ev.actor == me]
            if not step["acted"]:
                step["acted"] = any(ev.verb == step["act"] for ev in mine)
            if not step["acted"]:
                return False, ""
        else:
            mine = ()
        if until == "wake":
            return w.tick >= step["ends"], ""
        if until == "took":
            return any(ev.verb == "takes" for ev in mine), ""
        if until == "swung":
            return any(ev.verb == "swings" and ev.result in ("swung", "thrown")
                       for ev in mine), ""
        if until == "said":
            return any(ev.verb == "says" for ev in mine), ""
        # arrived: at a door the body is out (`arrives`); at another place the
        # goal is put down with no event, so the walk is over when the goal is
        # gone and the legs did not give up on it
        if any(ev.verb == "gives_up" for ev in mine):
            return False, "gave up"
        return (any(ev.verb == "arrives" for ev in mine) or p.get("goal") is None), ""

    def _news(self, p, job, percept, phase=None):
        """The word of the first news that stops the job, or None. News is
        what the job has not met: a salient percept row (a blow felt, a fire
        or violence seen, an alarm heard) perceived since the job was last
        looked at, a harmful start seen since then (`sees_attack`), and a
        percept of the will layer whose source the job did not start with.
        A noise is not news to a job (ORIENTING), nor the percept of the
        "go" after a pre-movement: it was answered when the pre-movement
        began. A verb stops only for the classes its `interrupt_on` lists
        ("any": every one), and not at all once it is committed
        (`commit_after`). The tick of news found in a row is kept
        (`news_at`): the next job answers it a reaction time after it."""
        w, st, me = self.w, job.state, p["pid"]
        verb = self.cat.verbs[job.option.verb]
        on = set(verb.interrupt_on)
        if verb.commit_after is not None and isinstance(job.option.duration, int) \
                and w.tick - job.started >= verb.commit_after * job.option.duration:
            on = set()
        found = {}
        for r in w.minds[me].perceived:
            if r.tick <= st["checked"]:
                continue
            kind = _percept(w, r, p)["kind"]
            if kind:
                found.setdefault(kind, (WORD[kind], r.tick))
        # by index, not by tick: a start made later in this same tick, by a
        # body the will layer asked after this one, is still to come
        for s in self.starts[st.get("starts", 0):]:
            if me in s["seen"]:
                found.setdefault("harm_to_me" if s["target"] == me else "violence_seen",
                                 ("sees_attack", s["t"]))
        st["checked"], st["starts"] = w.tick, len(self.starts)
        key = self._noticed(p)
        if key is not None and key not in st["known"]:
            st["known"].append(key)
            if phase != "go":
                found.setdefault(CLASS.get(percept, "any"), (percept, None))
        for cls in FIRST:
            if cls in found and (cls in on or "any" in on):
                word, st["news_at"] = found[cls]
                return word
        return None

    def compile(self, p, percept, phase=None):
        """The job's current step, as the will layer reads it: `p["intent"]`
        (the row to keep, the thing to take, where to aim, the spot to
        strike from, whether it runs from something, when to be woken, and
        that a blow wakes it) and the limbs to ask now, at most one. A step
        asks its limb once a decide period until its act is done. A decide
        about a noise asks only the eyes, a "go" only the legs and hands,
        and a pre-movement never the legs: a step whose limb is not asked
        waits for the next decide. An approach asks the legs again when
        they stop short of the band, and stops them in it."""
        w, tick = self.w, self.w.tick
        intent, ask = {}, set()
        allowed = PHASE_LIMBS.get(phase, frozenset(LIMBS))
        if percept in ORIENTING:
            allowed = frozenset({"eyes"})
        job = p.get("job")
        if job is not None and job.status == "running":
            st = job.state["steps"][job.state["i"]]
            # A BLOW IS NEWS ON THE TICK IT LANDS (will.py `_attend`)
            intent["felt"] = True
            if any(s["flee"] for s in job.state["steps"]):
                intent["flee"] = True
            if st["limb"] is None:
                intent["wake_at"] = st["ends"]
            else:
                want = st["want"]
                due = st["last"] is None or tick - st["last"] >= WILL["decide_every"]
                if st["until"] == "in_reach":
                    q = w.persons[st["aim"]]
                    there = self._in_band(p, q)
                    if there:
                        want = "stay"                 # in it: the legs stop
                    else:
                        s = self._seek(p, self._cells(p), q)
                        if s is not None:
                            intent["seek"] = [s]
                    # asked first, and again only when the legs are still
                    # short of it, or still going in it
                    due = due and (st["asked"] is None or (p.get("goal") is None) != there)
                else:
                    due = due and not st["acted"]
                intent["want"] = {st["limb"]: want}
                if st["take"] is not None:
                    intent["take"] = st["take"]
                if st["aim"] is not None:
                    q = w.persons[st["aim"]]
                    dx, dy = q["anchor"][0] - p["anchor"][0], q["anchor"][1] - p["anchor"][1]
                    n = math.hypot(dx, dy)
                    if n > 0:
                        intent["aim"] = (dx / n, dy / n)
                wait = job.state.get("not_before")
                if due and wait is not None and tick < wait:
                    intent["wake_at"] = wait          # still taking it in
                elif due and st["limb"] not in allowed:
                    if not (phase == "premove" and st["limb"] == "legs"):
                        intent["wake_at"] = tick + 1  # (the "go" asks the legs)
                elif due:
                    ask.add(st["limb"])
                    st["last"] = tick
                    if st["asked"] is None or st["until"] == "in_reach":
                        st["asked"], st["from"] = tick, w.events.next_id - 1
                        st["acted"] = False
                        if st["until"] == "wake":
                            st["ends"] = tick + st["dur"]
                # A WALKING BODY DECIDES ONLY ON NEWS, and the step test is
                # not news: so the job is looked at again a decide period
                # on, whatever the body is doing
                intent.setdefault("wake_at", tick + WILL["decide_every"])
        p["intent"] = intent
        return ask


_ORDINAL = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")


class MindPolicy:
    """A world's policy that is the mind interface (see the top of this
    file). `mind` is a src.mind loop.Mind (reactive mode with the Rules
    chooser by default: 0 model calls). With `record`, the backend is a
    record.Recorder, and `backend.cases(mind.rows)` gives the lab's case
    rows."""
    name = "mind"

    def __init__(self, mind=None, config=None, record=False):
        self.mind = mind if mind is not None else Mind(config or MindConfig())
        self.cat = self.mind.cat
        self.table = TablePolicy()
        self.record = record
        self.backend = None
        self._ctx = None        # (world, tick, pid, name, limbs, want, phase) of the last prepare

    def link(self, w):
        """The backend of world `w`: made once, and again for another world."""
        b = self.backend
        if b is None or b.w is not w:
            b = VoxelBackend(w, self.cat)
            self.backend = b = Recorder(b) if self.record else b
        return b

    # ------------------------------------------------------------ the hooks
    def prepare(self, w, p, percept, phase=None):
        """Every change of state of one decision of body `p` (will.py
        `_decide`): the job's step test, an interrupt by news, a question to
        the mind when the job is over, and the step compiled into
        `p["intent"]`. After it, `limbs` and `pick` only read.

        `phase` is where the decision stands in answering its percept
        (will.py `_react`). A FLINCH changes nothing: the eyes go to the
        thing, as the table says, and the percept is answered at the next
        stage. The "go" after a pre-movement is not news again (`_news`)."""
        if w.attention != "onset":
            raise ValueError("the mind link needs the attention law 'onset' (3.5)")
        b = self.link(w)
        if phase == "flinch":
            self._ctx = (w, w.tick, p["pid"], p["name"], PHASE_LIMBS["flinch"], {}, phase)
            return
        self.mind.at(w.tick)
        reason = b.advance(p, percept, phase)
        if reason is not None:
            self.mind.decide(b, str(p["pid"]), reason)
        ask = b.compile(p, percept, phase)
        self._ctx = (w, w.tick, p["pid"], p["name"], frozenset(ask),
                     dict(p["intent"].get("want") or {}), phase)

    def limbs(self, w, p, percept):
        """The limbs the step needs now: at most one, often none (a
        flinch: the eyes)."""
        c = self._ctx
        if c is None or c[1] != w.tick or c[2] != p["pid"]:
            return set()
        return set(c[4])

    def pick(self, sit, menu):
        """PURE. A falling body's question, and a flinch, go to the reflex
        table. Else the row of the step's limb that matches the step BY
        PAYLOAD (`World._wants`), or the null act when there is none."""
        if sit.get("falling") or sit.get("phase") == "flinch":
            return self.table.pick(sit, menu)
        c = self._ctx
        if c is None or c[1] != sit["tick"] or c[3] != sit["who"]:
            return 0
        want = c[5].get(sit["limb"])
        if want is None:
            return 0
        return next((i for i, o in enumerate(menu) if c[0]._wants(o, want)), 0)

    def on_restore(self, w):
        """A restore takes the world back to a tick that has not happened
        yet: what was asked at or after it is forgotten (the rows, the
        request ids, the job events), so the run from there asks the same
        questions with the same ids and draws the same answers. The jobs
        are the persons' own (`p["job"]`) and come back with them."""
        self._ctx = None
        b = self.backend
        if b is None or b.w is not w:
            return
        inner = b.backend if self.record else b
        m = self.mind
        gone = [r for r in m.rows if r["t"] >= w.tick]
        m.rows = [r for r in m.rows if r["t"] < w.tick]
        m._ids.difference_update(r["id"] for r in gone)
        m.remote = {t: n for t, n in m.remote.items() if t < w.tick}
        m.jobs = {str(p["pid"]): p["job"] for p in w.persons if p.get("job") is not None}
        inner.job_events[:] = [e for e in inner.job_events if e.t < w.tick]
        inner.starts[:] = [s for s in inner.starts if s["t"] < w.tick]
        inner._bound.clear()
        inner._view = None
        if self.record:
            for r in gone:
                b.menus.pop(r["id"], None)


def link(w, config=None, mind=None, record=False):
    """Give world `w` a mind: its policy becomes a MindPolicy, and its
    attention law "onset", which the link needs (3.5). Returns the policy."""
    w.attention = "onset"
    w.policy = pol = MindPolicy(mind=mind, config=config, record=record)
    pol.link(w)
    return pol
