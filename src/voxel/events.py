"""Typed events: what happened, who did it, and what caused it.

Step V0 of docs/sim-mind-design.md (3.4). It adds the record (`Ev`), the
bounded log that holds it (`EventLog`), the closed verb table, a Mind per
person, and the three small functions every later events step builds on:
`emit`, `because` and `why`. Nothing in the sim calls any of this yet -- the
35 legacy sites are converted one by one, from V1 on -- so the world steps
exactly as it did before (golden SAME; the new names show as NEW:).

V5 makes it a CAUSE LOG that can be read: each decision is a `decides`
event (will.py), every trace row carries `row["mind"]` (`_mind_row`: the
decision's id, its cause chain and what the body had just perceived, as
plain JSON), and `why_json` hands a chain out as plain JSON too.
"""
import hashlib
from collections import deque
from dataclasses import dataclass, field, replace

# ── the closed verb table (3.4) ─────────────────────────────────────────────
# Every verb a typed event can carry. Closed, because a chooser or a belief
# key that matched on an open string would silently stop matching the day a
# site's wording drifted. New verbs are added here, never spelled ad hoc at
# a call site.
VERBS = (
    "decides", "steps", "lands", "falls", "swings", "hits", "wounds",
    "breaks", "topples", "looses", "melts", "sets", "dissolves",
    "ignites", "catches_fire", "burns_out",
    "says", "heard", "saw", "looked",
    # V1 (will.py, senses.py, physiology.py): the will layer's own acts, and
    # the body's slow chemistry. "wakes" covers both a sound waking a
    # sleeper and the air coming good enough to come round -- the two
    # legacy sites say different things but mean the same fact.
    "gone", "faints", "wakes", "dies",
    "holds", "takes", "reaches", "pulls_in", "catches", "releases",
    "jumps", "goes", "stays", "looks", "leans", "stands",
    "retries", "gives_up", "arrives", "rescues", "rolls",
    # E2 (things.py): the identity pass. A thing that comes apart "splits";
    # one whose last wood is ash "ashes".
    "splits", "ashes",
    # V3 (`_after_raster`): what the particle matter did, one verb for each
    # kind `matter.events()` reports (touch, struck, crack, part). They are
    # kept apart from "lands" and "breaks", which are the lattice's own
    # facts, and from "splits", which only the identity pass decides.
    "touches", "strikes", "cracks", "parts",
    # WAVE 13 (plumbing, part 13a). Fifteen verbs, appended in ONE change in
    # the order the plan fixes (plan section 4): T2, T3, T5, T7, T11. The
    # order is part of the contract (a verb's place in this tuple is its
    # VERB_ID), so a later wave appends AFTER these and never reorders. No law of
    # the old sim emits them; the new modules do, once a lane switches them on.
    #   T2 records: a cup runs over, two pieces are glued or tied, two are
    #       fused, a joined group comes apart.
    "spills", "joins", "fuses", "comes_apart",
    #   T3 layers: none (the layer topic names no verb).
    #   T5 scheduler: a watched quantity crossed its level during a skip.
    "crosses",
    #   T7 body heat: the thermoregulatory reflexes (on at 10 percent of the
    #       most they can give, off at 5 percent: one event per onset).
    "shivers", "sweats",
    #   T11 mechanisms: a door or lid and what is done to it.
    "opens", "closes", "locks", "unlocks", "picks", "kicks", "jams", "forces",
)
VERB_ID = {v: i for i, v in enumerate(VERBS)}


def plain(v):
    """`v` as plain JSON values: a tuple becomes a list, a numpy number or
    array a Python one, a `_Pending` box the id in it (None while it has
    none), a dict key a string. So `json.loads(json.dumps(plain(v)))` gives
    back exactly `plain(v)`. Anything else is written as its `str`."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, (int, float)):
        return v if type(v) in (int, float) else (
            int(v) if isinstance(v, int) else float(v))
    if isinstance(v, _Pending):
        return v.id
    if hasattr(v, "tolist"):                  # numpy, without importing it
        return plain(v.tolist())
    if isinstance(v, (tuple, list)):
        return [plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    return str(v)


def hash01(*parts):
    """A float in [0, 1), the same for the same `parts`, forever.

    splitmix64 over a digest of the parts' repr, not Python's `hash()`: that
    one is salted per process for strings (and per run, unless
    `PYTHONHASHSEED` is pinned), so a misread draw keyed on it would not
    replay in another process. (Until V4 this did call `hash()`, against
    what this note said; nothing drew from it yet.) A digest has no such
    seed, so the sim keeps no RNG of its own (2.4) and a misread is exactly
    as reproducible as the tick that drew it."""
    x = int.from_bytes(hashlib.blake2b(repr(parts).encode(),
                                       digest_size=8).digest(), "little")
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    x = x ^ (x >> 31)
    return (x & 0xFFFFFFFFFFFFFFFF) / 2.0**64


# ── the record (3.4) ─────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Ev:
    """One typed event. Immutable, so a reference to it (carried on a body, a
    particle batch, or a Mind's `*_ev` slot) is never stale underneath its
    holder."""
    id: int
    tick: int
    verb: str
    actor: int | None              # a pid, or None for the world
    target: tuple | None           # ("person", pid) | ("thing", eid) |
                                    # ("fire", ev_id) | ("mat", m, cell)
    instrument: tuple | None
    phase: str | None               # start, done or stopped
    result: str | None
    why: str | None
    place: tuple | None             # (x, y, z)
    origin: tuple | None             # the float point a sound starts from
    db: float | None
    visible: bool
    dur: float | None
    cause: int | None               # the ev_id this one is because of
    link: str | None                # call, carried or near
    text: str                       # the legacy line; never shown to a model
    data: tuple = ()                # extras as sorted (k, v) pairs

    def __deepcopy__(self, memo):
        # Frozen and never mutated, so sharing the instance across a snapshot
        # copy is exact and free: 4,096 records copy in under a millisecond
        # (3.4) because none of them is actually copied.
        return self

    def to_json(self):
        """Every field but `text`, as plain JSON values (`plain`)."""
        return plain({
            "id": self.id, "tick": self.tick, "verb": self.verb,
            "actor": self.actor, "target": self.target,
            "instrument": self.instrument, "phase": self.phase,
            "result": self.result, "why": self.why, "place": self.place,
            "origin": self.origin, "db": self.db, "visible": self.visible,
            "dur": self.dur, "cause": self.cause, "link": self.link,
            "data": self.data,
        })


class EventLog:
    """A ring of at most `cap` events, addressed by id. Ids never repeat
    within a world's life (`restore` bumps `_epoch` so they never repeat
    across one either); old rows fall off the front as new ones are
    appended."""
    __slots__ = ("rows", "first", "cap")

    def __init__(self, cap=4096):
        self.rows = []
        self.first = 0
        self.cap = cap

    @property
    def next_id(self):
        return self.first + len(self.rows)

    def append(self, ev):
        self.rows.append(ev)
        if len(self.rows) > self.cap:              # amortised trim: the ring
            cut = len(self.rows) - self.cap // 2    # drops half at once, not
            del self.rows[:cut]                     # one row every append
            self.first += cut
        return ev

    def get(self, ev_id):
        """The event with this id, or None once it has fallen off the ring."""
        i = ev_id - self.first
        return self.rows[i] if 0 <= i < len(self.rows) else None

    def since(self, ev_id):
        """Every row with an id strictly greater than `ev_id`, oldest first.
        `ev_id` before the ring's start is clamped, so a job started long ago
        still gets everything the ring still holds."""
        i = max(ev_id - self.first + 1, 0)
        return self.rows[i:]


# ── what reached one mind (3.4, V3) ─────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Rec:
    """One thing that reached one person's senses: a row of `Mind.perceived`.

    It says only what the senses could tell. `ev` is the event that caused
    it, so a later step can ask why, but the fields a reader may show are
    the rest, and they are written from the perceiver's side: `actor_seen`
    is None for a sound unless the voice was known and the words made out,
    and `place` is where it seemed to be, not where the cause was."""
    tick: int
    how: str                        # heard | saw | felt | did | looked
    ev: int | None                  # the source event's id
    verb: str | None                # what the senses could tell it was
    actor_seen: int | None          # a pid, only when the senses told who
    target_seen: tuple | None
    place: tuple | None             # where it seemed to be (cells)
    db: float | None                # how loud at the ear (heard only)
    data: tuple = ()                # extras as sorted (k, v) pairs

    def __deepcopy__(self, memo):
        # Frozen, like Ev: a snapshot shares the row instead of copying it.
        return self


# ── a mind per person (3.4, 3.5) ────────────────────────────────────────────
@dataclass(slots=True)
class Mind:
    """What one person's mind holds, outside the person dict (3.0, ruling 3).
    A world's `minds` list is one of these per `pid`, in `pid` order.

    V0 declared the shape. V3 (perceive.py) fills `perceived` with `Rec`
    rows, `got` with what each row was about (so the same thing is not
    written twice), `bel` (written only by beliefs.py, from V4), `harm` from the
    blows and the bad air the body took, the `*_ev` slots from the acts
    that set them, and `noticed` from the percept chain of `_law_will`.
    `attn`/`news` are the attention law's (3.5, step M7)."""
    perceived: deque = field(default_factory=lambda: deque(maxlen=32))
    got: dict = field(default_factory=dict)          # what a row was about -> tick
    bel: dict = field(default_factory=dict)         # capped at 64 keys (3.4)
    harm: dict = field(default_factory=dict)         # wounds|burns|air -> ev_id
    said_ev: int | None = None
    go_ev: int | None = None
    hold_ev: int | None = None
    lean_ev: int | None = None
    attn: dict = field(default_factory=dict)         # ev_id seen -> tick
    news: int | None = None                          # the latest unattended
    noticed: tuple | None = None                     # (tick, percept, ev_id)


class _Pending:
    """A box for an event id that does not exist yet (`_defer`, V2). `.id`
    is None until `_tick_end` resolves it, then the real id, for good --
    so a holder that keeps the box itself, not the id, always reads the
    right thing whether it looks before or after that tick ends."""
    __slots__ = ("id",)

    def __init__(self):
        self.id = None


def _rid(v):
    """The id in `v` if it is a box, else `v` itself."""
    return v.id if isinstance(v, _Pending) else v


# The verb each kind of `matter.events()` record becomes (`_after_raster`).
_MATTER_VERB = {"touch": "touches", "struck": "strikes", "crack": "cracks",
                "part": "parts"}
# The events a sound made by `_noise` can be the noise of (`_unbox`). Wave 13
# adds what makes a noise at a door: a kick, a jam, a forcing, and a leaf that
# opens by slamming (T11 5.1). "picks" is NOT here: the click of a lock being
# picked is passed to `_noise` as its own `ev`, so it never needs matching.
_SOUNDING = frozenset({"hits", "lands", "breaks",
                       "kicks", "jams", "forces", "opens"})


class EventsMixin:

    def _init_events(self):
        """Make the event state of a new world. `__init__` calls it once,
        after `_init_things`.

        The log, one Mind per person already registered, the current cause,
        the sim's own seed (used only by `hash01` draws, never by an RNG:
        2.4) and the per-site golden-check marker (`_cover`, off by
        default). V2 adds `_disturbed` (a place -> ev_id map, for "near"
        causes: 3.4), `_break_log` (this tick's shatters, held back so
        `_tick_end` can report them in one deterministic order) and
        `_pending` (every OTHER new V2 event, held back for the same
        reason `_defer` explains).

        V3 (perceive.py) adds the fires being kept track of (`_fires`, and
        `_alight`, the things burning), the fire map handed from
        `_law_perceive` to `_law_will` (`_burning_now`), the first event the
        eyes have not yet been shown (`_seen_upto`), and the bookkeeping of
        `_unbox`: the sounds still waiting to learn their event
        (`_sound_boxes`), whether any box went out this tick (`_boxed`) and
        the ring rows emitted with a cause still in a box (`_boxed_rows`).

        V4 (beliefs.py) adds the two switches for wrong beliefs that have a
        reason, both off: `misread` (True, or a {verb: chance} table) and
        `claims`; and `_claimed`, what each `says` claimed, settled as it was
        spoken (`emit`).

        V5 adds `_mind_rows`: this tick's `row["mind"]` dicts, each with
        the percept rows it was written from, finished at the tick's end
        (`_finish_minds`) once every box in them has its id."""
        self.events = EventLog()
        self.minds = [Mind() for _ in self.persons]
        self.seed = 0
        self._cover = None
        self._cause = None
        self._link = None
        self._epoch = 0
        self._disturbed = {}
        self._break_log = []
        self._pending = []
        self._fires = []
        self._alight = {}
        self._burning_now = None
        self._seen_upto = 0
        self._sound_boxes = []
        self._boxed = False
        self._boxed_rows = []
        self.misread = False
        self.claims = False
        self._claimed = {}
        self._mind_rows = []

    def _tick_start(self):
        """Open a tick. `step` calls it first, before any law runs.

        Resets the current cause, so a tick that emits no event under
        `because(...)` never attributes its events to last tick's cause.
        Also clears `_disturbed`, `_break_log` and `_pending`: a "near"
        cause only reaches back into the SAME tick's disturbances (3.4
        calls it a heuristic), and a tick's deferred events are reported
        once, at its own end."""
        self._cause = None
        self._disturbed = {}
        self._break_log = []
        self._pending = []

    def _defer(self, verb, cause=None, link=None, **kw):
        """Queue an event for `_tick_end`, instead of emitting it now.

        Every event V2 adds (`hits`, `lands`, `wounds`, `topples`...) is
        raised from inside a PHYSICAL law, and every physical law runs
        before `_law_will` in a tick (2.2). `_say` (senses.py) stamps the
        id of its OWN `says` event onto the shout's sound record -- and
        ids come from one counter, shared by the whole world. Emitting a
        new kind of event straight away, the moment the law that causes it
        runs, would consume an id earlier in the tick than it used to be
        consumed, and a shout later in the SAME tick would end up stamped
        with a different number for no reason of its own. Holding every
        new V2 event back to `_tick_end` -- after everything else this
        tick was going to do with the ring -- leaves that alone.

        Returns a `_Pending` box: `.id` is None until the tick ends, then
        the real event id, forever. A caller that needs to hand this id to
        a LATER tick's `cause=` (a fall, landed on much later) can keep the
        box itself and read `.id` whenever it likes -- by then it is
        always resolved. `cause` may itself be a `_Pending` box from an
        earlier `_defer` in this same tick; it is read as `.id` only once
        this tick's pending events are flushed, by which time every box
        made before it in the SAME tick already has one (deferred events
        flush in the order they were queued, and a cause is always queued
        before its effect)."""
        if cause is None:
            cause = self._cause
        if link is None:
            link = self._link
        box = _Pending()
        self._pending.append((box, verb, cause, link, kw))
        return box

    def _mark_disturbed(self, place, ev_id):
        """Record that something happened at `place` this tick, as `ev_id`.
        `_cause_near` reads this back for a "near" link (3.4) when no direct
        cause is in hand -- a cell that breaks because a wall three cells
        away just took a blow, say, rather than from any cause of its own."""
        if place is not None:
            self._disturbed[tuple(int(v) for v in place)] = ev_id

    def _cause_near(self, place, radius=1):
        """The most recent disturbance within `radius` cells of `place`, this
        tick, or None. Used only as a last resort, when nothing calling in
        has a direct cause to hand over -- see `because`'s "near" link."""
        if place is None or not self._disturbed:
            return None
        x, y, z = (int(v) for v in place)
        best = None
        for (dx, dy, dz), ev_id in self._disturbed.items():
            if (abs(dx - x) <= radius and abs(dy - y) <= radius
                    and abs(dz - z) <= radius):
                if best is None or ev_id > best:
                    best = ev_id
        return best

    def _after_raster(self):
        """Read what the particle matter did this tick, as typed events.
        `step` calls it just after `matter.rasterize`.

        `matter.events()` is the one producer and this is the one consumer
        (3.0, ruling 7). Each record becomes one event of its own verb:
        touch -> touches, struck -> strikes, crack -> cracks,
        part -> parts. The piece a record is about is a thing only when its
        eid is not 0, so an unnamed lump is named by place alone. DEFERRED,
        like every other physical event (`_defer` says why): this runs well
        before `_law_will`, and a shout later this tick must get the id it
        always got. What the records do NOT do yet is make a sound or hurt
        anyone: that waits for `matter.seam_sounds` and `seam_blows`, both
        off (3.4's risk list)."""
        for r in self.matter.events():
            verb = _MATTER_VERB.get(r["kind"])
            if verb is None:
                continue
            piece = ("thing", r["eid"]) if r["eid"] else None
            target, instrument = piece, None
            if r["kind"] == "touch":
                target, instrument = (("thing", r["other"][0])
                                      if r["other"][0] else None), piece
            elif r["kind"] == "struck":
                hit = [k[0] for k in r["against"] if k[0]]
                target, instrument = (("thing", hit[0]) if hit else None), piece
            data = tuple(sorted((k, v) for k, v in r.items()
                                if k not in ("kind", "tick", "eid", "at",
                                             "cell")))
            self._defer(verb, target=target, instrument=instrument,
                        place=r["cell"], origin=r["at"], data=data, text="")

    def _tick_end(self):
        """Close a tick. `step` calls it last, before the tick count goes up.

        First flushes every event `_defer` queued, in the order queued (a
        cause is always queued before its effect, so by the time an
        effect's `_Pending` cause is read, it is resolved). Then flushes
        this tick's breaks, sorted by cell so the order is the same every
        run regardless of which mask or loop happened to find them first
        (4.4, V2: "flushed sorted at tick end") -- AFTER the pending
        events, so a break caused by one of them (a landing that shatters
        what it lands on) sees a resolved id too. The fires that went out
        are among the pending events (`_law_perceive` defers them). Then
        `_unbox` swaps every box handed out this tick for its real id. Last,
        `_finish_minds` writes the cause chains of this tick's decisions,
        now that every id in them is real."""
        for box, verb, cause, link, kw in self._pending:
            if isinstance(cause, _Pending):
                cause = cause.id
            box.id = self.emit(verb, cause=cause, link=link, **kw).id
        self._pending = []
        for x, y, z, mat, actor, instrument, cause, link in sorted(
                self._break_log, key=lambda row: row[:3]):
            if isinstance(cause, _Pending):
                cause = cause.id
            self.emit("breaks", actor=actor, target=("mat", mat, (x, y, z)),
                      instrument=instrument, place=(x, y, z), cause=cause,
                      link=link, text="")
        self._break_log = []
        self._unbox()
        self._finish_minds()

    def _unbox(self):
        """Put the real event id wherever this tick left a box.

        Perception runs before `_law_will`, so it meets events that do not
        have an id yet: a crash this tick (its `lands` waits for the flush
        above), a fire that lit this tick (its `ignites` too), a wound
        (`_blow`). It hands their boxes on -- into a sound record, a percept
        row, `p["noise"]`, a Mind's harm or `noticed` slot, the cause of a
        `wakes` or `faints` -- and here, once the flush has run, each box
        is swapped for the number in it. After this nothing a tick leaves
        behind holds a box, so a snapshot copies plain ints: not a Mind's
        attention keys or `news` either (M7's onset law stores the source
        it attended to, box and all).

        First the sounds that were made without being told their event
        (`_noise` with no `ev`: a crash in `_spend`, a landing in
        `_land_body` or `_cash_impacts`). Each is the noise of the first
        `hits`, `lands` or `breaks` of the same tick within 2 cells of it --
        those sites make the sound and queue the event a line apart. One
        that matches nothing keeps None."""
        # The attention law (`_attend`, onset) keys what it attended to by
        # source, and a source seen the tick it was raised is a box. Swap in
        # the id, keeping the order and the newer tick when the id is
        # already a key (the same event seen under both names). A box that
        # never got an id (a sound that matched nothing) is dropped: nothing
        # can be news "again" under it. Done before the `_boxed` gate,
        # because a box can reach a percept by a road that never set it (a
        # `_defer` box handed to `_noise` as its `ev`), and it costs a look
        # at a few small dicts.
        for m in self.minds:
            if isinstance(m.news, _Pending):
                m.news = m.news.id
            if m.attn and any(isinstance(k, _Pending) for k in m.attn):
                attn = {}
                for k, t in m.attn.items():
                    k = _rid(k)
                    if k is not None:
                        attn[k] = max(attn.get(k, t), t)
                m.attn = attn
        if not self._boxed:
            return
        self._boxed = False
        if self._sound_boxes:
            rows = self.events.rows
            for box, tick, at in self._sound_boxes:
                if box.id is not None:
                    continue
                c = [int(round(v)) for v in at]
                i = len(rows) - 1
                while i >= 0 and rows[i].tick > tick:
                    i -= 1
                while i >= 0 and rows[i].tick == tick:
                    i -= 1
                for ev in rows[i + 1:]:
                    if ev.tick != tick:
                        break
                    if ev.verb in _SOUNDING and ev.place is not None and max(
                            abs(int(ev.place[k]) - c[k]) for k in range(3)) <= 2:
                        box.id = ev.id
                        break
            self._sound_boxes = []
        for snd in self.sounds:
            if isinstance(snd.get("src"), _Pending):
                snd["src"] = snd["src"].id
        for f in self._fires:
            f[3] = _rid(f[3])
        for m in self.minds:
            for i in range(len(m.perceived)):
                r = m.perceived[i]
                if isinstance(r.ev, _Pending):
                    ev = self.events.get(r.ev.id) if r.ev.id is not None else None
                    m.perceived[i] = replace(
                        r, ev=r.ev.id,
                        verb=r.verb if r.verb or ev is None else ev.verb)
            for k, v in m.harm.items():
                m.harm[k] = _rid(v)
            if m.noticed is not None and isinstance(m.noticed[2], _Pending):
                m.noticed = (m.noticed[0], m.noticed[1], m.noticed[2].id)
        for p in self.persons:
            nz = p.get("noise")
            if nz is not None and isinstance(nz.get("ev_id"), _Pending):
                nz["ev_id"] = nz["ev_id"].id
        for ev_id in self._boxed_rows:
            ev = self.events.get(ev_id)
            if ev is not None and isinstance(ev.cause, _Pending):
                self.events.rows[ev_id - self.events.first] = replace(
                    ev, cause=ev.cause.id)
        self._boxed_rows = []

    # ── the API every later events step calls (3.4) ─────────────────────────
    def emit(self, verb, actor=None, target=None, instrument=None,
             phase=None, result=None, why=None, place=None, origin=None,
             db=None, visible=True, dur=None, cause=None, link=None,
             text="", data=(), to=None, site=None):
        """Record a typed event and return it.

        `verb` must be one of the closed `VERBS`. `cause` defaults to the
        current `because(...)` cause, so a call made inside one does not have
        to repeat it. When `to` is given, the legacy text is also appended to
        that person dict's `p["events"]` list, at the same position the old
        site wrote it -- the one legacy side effect this has, so that a step
        converting a site byte for byte still produces the old text.

        `site` names the call site for the coverage check (4.4, V1): when
        `w._cover` is a dict, converting a site bumps its count there. It is
        never state and it costs nothing when `_cover` is off (the default,
        None).

        `cause` may be a `_Pending` box (V3: a sleeper woken by a crash whose
        `lands` is still queued, a faint from a wound `_blow` just dealt).
        Its id is used if it has one; if not, the row keeps the box and
        `_unbox` puts the id in at the end of the tick."""
        if cause is None:
            cause = self._cause
        if link is None:
            link = self._link
        if isinstance(cause, _Pending):
            if cause.id is not None:
                cause = cause.id
            else:
                self._boxed_rows.append(self.events.next_id)
                self._boxed = True
        if verb == "says" and self.claims:
            # what the words claim is settled as they are spoken (V4,
            # beliefs.py `_claim`), so the row is final the moment it exists
            data = tuple(sorted(tuple(data) + self._claim(
                self.events.next_id, actor, result)))
        ev = Ev(id=self.events.next_id, tick=self.tick, verb=verb,
                actor=actor, target=target, instrument=instrument,
                phase=phase, result=result, why=why, place=place,
                origin=origin, db=db, visible=visible, dur=dur, cause=cause,
                link=link, text=text, data=tuple(data))
        self.events.append(ev)
        if to is not None and text:
            to["events"].append(f"t{self.tick}: {text}")
        if site is not None and self._cover is not None:
            self._cover[site] = self._cover.get(site, 0) + 1
        return ev

    def because(self, ev_id, link="call"):
        """A context manager: events emitted inside it are caused by `ev_id`,
        with this `link` kind, unless they name their own `cause`. Restores
        the cause that was current before, even if the block raises."""
        return _Because(self, ev_id, link)

    def why(self, ev_id):
        """Walk the cause chain from `ev_id` back to its root, oldest first.
        Stops at an id the ring no longer holds, or at a cause of None.
        `ev_id` may be a box (`_defer`); a cause still in a box, read before
        its tick has ended, ends the chain there for now."""
        chain = []
        seen = set()
        ev_id = _rid(ev_id)
        while ev_id is not None and ev_id not in seen:
            seen.add(ev_id)
            ev = self.events.get(ev_id)
            if ev is None:
                break
            chain.append(ev)
            ev_id = _rid(ev.cause)
        chain.reverse()
        return chain

    def why_json(self, ev_id):
        """`why(ev_id)` as plain JSON: one dict per event (`Ev.to_json`),
        oldest first, each with its `cause` and its `link`. This is what a
        reader outside the sim gets (`row["mind"]`, the lab), so it never
        holds `Ev.text`."""
        return [ev.to_json() for ev in self.why(ev_id)]

    # ── the cause log, as a trace row reads it (V5) ─────────────────────────
    MIND_RECENT = 5                 # percept rows a `row["mind"]` keeps

    def _mind_row(self, p, dec):
        """`row["mind"]` for a decision of body `p`, logged as event `dec`.

        Plain JSON (`plain`): the `decides` id; its `cause` (the event the
        percept came from) and the whole chain back from the decision
        (`why_json`), so a trace file alone says why; and the last
        MIND_RECENT rows the body had perceived, as their fields (who,
        what, where, which event), never as text. Some ids are still boxes
        while the tick runs, so the row is written again at the tick's end
        (`_finish_minds`); a decision made outside `step` keeps what was
        known when it was made."""
        recs = list(self.minds[p["pid"]].perceived)[-self.MIND_RECENT:]
        row = {}
        self._mind_fill(row, dec, recs)
        self._mind_rows.append((row, dec, recs))
        return row

    def _mind_fill(self, row, dec, recs):
        ev = self.events.get(dec) if dec is not None else None
        row["decides"] = dec
        row["cause"] = plain(ev.cause) if ev is not None else None
        row["why"] = self.why_json(dec)
        row["recent"] = [plain({"tick": r.tick, "how": r.how, "ev": r.ev,
                                "verb": r.verb, "actor_seen": r.actor_seen,
                                "target_seen": r.target_seen,
                                "place": r.place, "db": r.db,
                                "data": r.data}) for r in recs]

    def _finish_minds(self):
        """Write this tick's `row["mind"]` dicts again, now that every box
        in their chains has its id (`_tick_end`, after `_unbox`)."""
        if self._mind_rows:
            for row, dec, recs in self._mind_rows:
                self._mind_fill(row, dec, recs)
            self._mind_rows = []


class _Because:
    """`with w.because(ev_id, link):` -- see `EventsMixin.because`. Also sets
    the default `link` an `emit()` inside the block gets when it names none
    of its own, so a chain of synchronous callees need not repeat "call"."""
    __slots__ = ("w", "ev_id", "link", "_prev_cause", "_prev_link")

    def __init__(self, w, ev_id, link):
        self.w, self.ev_id, self.link = w, ev_id, link

    def __enter__(self):
        self._prev_cause, self._prev_link = self.w._cause, self.w._link
        self.w._cause, self.w._link = self.ev_id, self.link
        return self

    def __exit__(self, exc_type, exc, tb):
        self.w._cause, self.w._link = self._prev_cause, self._prev_link
        return False
