"""The rooms backend: the inn (world.py) behind the mind interface
(src/mind/api.py, Backend). docs/sim-mind-design.md 3.5, step M3.

THE TURN. The inn runs in turns. Inside a turn the people act one by one, in
an order that is drawn for that turn. A person whose act has ended chooses at
its place in that order, and what it does can stop the acts of the people
after it. So a turn stops at each choice, and the mind answers there:

    b.step()                            start a turn; it runs to the first choice
    while b.needs_decision():           at most one person at a time
        (who, reason), = b.needs_decision()
        options = b.bind(who)
        prior, prior_avg, habit = b.rule_scores(who, options)
        ...                             the mind makes a menu; a chooser picks
        b.start(who, option, why)       the act starts; the turn runs on to the next choice
    # needs_decision() is empty: the turn is over

`night()` runs this loop in COMPAT MODE: each choice offers every option, in
the inn's order, and a lab chooser draws with the world's own rng (rules.py).
Then the events are the lab's events, byte for byte (tests/test_rooms.py
compares them with the inn goldens).

WHAT THE BACKEND KEEPS, apart from the world:
    world.log     the events, in the lab's format and with the lab's verbs, so
                  lab/judge/signals.py reads them as it reads the lab's nights
    decisions     one row for each job that starts: the Why of its decision.
                  These rows are the decisions JSONL; the event rows stay the
                  lab's.
    starts        the START RECORDS: an act of a loud or violent verb as it
                  starts ("Fenn raised a hand against Tom"), and who saw or
                  heard it. They are a separate stream, and they never change
                  the world: they never call _learn, never stop an act and
                  never draw from the rng. The mind reads them as percepts.
    jobs          the job of each person (the last one it started)
    job_events    each step in the life of each job: start, done, failed, stopped

OPTIONS. An option of the inn is a row of World.options. `bind` gives each
row as an Option of the verb catalog (src/mind/data/verbs.yaml): the catalog
verb, its roles and its words. The words come from the catalog, and `bind`
checks that they are the inn's words. An Option of this backend has no
`features`: the rule scores come from the inn's own rows, which the backend
keeps until the job starts. So an Option is fixed by its verb and its roles,
and the backend makes each one only once.
"""
from __future__ import annotations

import functools
import json
from dataclasses import fields

from src.mind import catalog as _catalog
from src.mind.api import Belief, DecisionRequest, Job, JobEvent, Percept, Ref, SelfState, Why

from . import rules
from .world import HARMFUL, ROOMS, VIOLENT, World

NAME = "rooms"                  # the key of this backend in the verb catalog
QUESTION = "What does {who} do next?"
PERCEPTS = 8                    # how many percepts `percepts` gives: the most recent ones
NOT_BOUND = (None, (), [], [], {})
WHY = tuple(f.name for f in fields(Why))     # the fields of a decision row that come from its Why

# What a person in the next room hears of a loud act. A sound does not say
# who made it (docs/sim-mind-design.md 2.4).
HEARD = {"strike": "sounds of a fight in {room}", "shove": "sounds of a fight in {room}",
         "threaten": "angry voices in {room}", "accuse": "angry voices in {room}",
         "shout for help": "a shout for help in {room}", "play": "music in {room}",
         "flee to": "running feet in {room}"}
# the first word of a note (World.note), as others tell it
PAST = {"notice": "noticed", "come": "came"}


@functools.lru_cache(maxsize=1)
def catalog():
    """The verb catalog, read once."""
    return _catalog.load()


def verb_id(action, text):
    """The catalog verb of an inn option or event. Two inn verbs run two
    catalog verbs each, told apart by their words: "steal X from Q" and "take
    X from Q" (Q lies on the floor), "drink X" and "drink from X".
    tests/test_mind_catalog.py holds the catalog's words to the inn's."""
    if action == "steal":
        return "take_from" if text.startswith("take ") else "steal"
    if action == "drink":
        return "drink_from" if text.startswith("drink from ") else "drink"
    return action.replace(" ", "_")


def jsonl(rows):
    """Rows as JSON lines, as lab/world2/run.py writes the events of a night."""
    return "".join(json.dumps(r) + "\n" for r in rows)


class RoomsBackend:
    """The inn as a Backend of the mind interface. `who` is a person's name."""
    name = NAME

    def __init__(self, seed=0, turns=300, bandit=None, cat=None):
        self.seed = seed
        self.world = World(seed=seed, bandit=bandit, turns=turns)
        self.world.on_end = self._ended
        self.cat = cat or catalog()
        self.decisions = []
        self.starts = []
        self.jobs = {}
        self.job_events = []
        self._free = {}         # who -> why it chooses: idle | done | failed:... | interrupt:...
        self._bound = {}        # who -> (when, options, rows, keys, {key: index}) of its last bind
        self._scores = {}       # who -> (when, keys, (prior, prior_avg, habit)) of its last rule_scores
        self._options = {}      # (inn verb, text) -> (Option, its key)
        self._refs = {}         # (kind, id) -> Ref
        self._seen = {}         # who -> the start records it saw or heard, in order
        self._percepts = {}     # (who, source) -> Percept
        self._beliefs = {}      # (kind, thing, value, source) -> Belief
        self._drives = [(d, t) for d in self.cat.drives if (t := self.cat.drive_test(d, NAME))]
        self._native = frozenset(self.cat.declared(NAME)["verbs"])     # the inn's verbs

    # ------------------------------------------------------------ the turn
    def step(self):
        """Start the next turn. It runs up to the first person who must
        choose; `needs_decision` names that person."""
        self.world.step()

    def needs_decision(self):
        p = self.world.deciding
        return [] if p is None else [(p.name, self._free.get(p.name, "idle"))]

    def over(self):
        """Every person is dead or down: the lab's run stops here."""
        return all(p.dead or p.down for p in self.world.people.values())

    def _now(self):
        # the world changes only at an event, or when the turn moves on
        w = self.world
        return w.t, w.at, w.n

    # ------------------------------------------------------------ what a person knows
    def self_state(self, who):
        p, w = self.world.people[who], self.world
        feels = [word for word, on in (("hungry", p.hunger > 0.6), ("thirsty", p.thirst > 0.6),
                                       ("afraid", p.fear > 0.4), ("drunk", p.drunk > 0.4),
                                       ("hurt", p.hp < 7)) if on]
        mad = [n for n, a in p.anger.items() if a > 0.4]
        if mad:
            feels.append("angry at " + " and ".join(mad))
        return SelfState(p.name, p.public, p.room, tuple(w.held(p)), tuple(feels),
                         tuple(d for d, test in self._drives if self._above(p, test)))

    @staticmethod
    def _above(p, test):
        v = max(p.anger.values(), default=0.0) if test["field"] == "anger_max" else getattr(p, test["field"])
        return v > test["above"]

    def percepts(self, who, last=PERCEPTS):
        """The `last` most recent things `who` perceived, oldest first: the
        events of others that it saw or heard, and the start records."""
        p, log = self.world.people[who], self.world.log
        # an event id is "e" and the event's index in the log; a start record
        # comes after the events that were in the log when it was made
        got = sorted([(int(i[1:]), 1, i) for i in p.perceived]
                     + [(s["after"], 0, s["id"]) for s in self._seen.get(who, ())])[-last:]
        out = []
        for n, is_event, src in got:
            pc = self._percepts.get((who, src))
            if pc is None:
                if is_event:
                    pc = self._percept(p, log[n], "done", src in p.heard)
                else:
                    s = self.starts[int(src[1:])]
                    pc = self._percept(p, s, "start", who in s["heard"])
                self._percepts[(who, src)] = pc
            out.append(pc)
        return tuple(out)

    def _percept(self, p, e, phase, heard):
        """One event (or start record) as p perceived it."""
        a = e["action"]
        if heard:
            text, actor, target = HEARD[a].format(room=e["room"]), None, None
        else:
            text = e["text"] if phase == "start" else self._told(e)
            actor, target = self._ref("person", e["actor"]), self._target(e["target"])
        kind = ("harm_to_me" if e["target"] == p.name and a in HARMFUL else
                "violence_seen" if a in VIOLENT else "alarm_heard" if a == "shout for help" else "")
        # what stops an act in the inn (World._perceive): these, and a theft seen
        salience = 1.0 if kind or (a == "steal" and not heard) else 0.0
        verb = e["verb"] if phase == "start" else (verb_id(a, e["text"]) if a in self._native else a)
        return Percept(e["t"], "heard" if heard else "saw", text, e["id"], verb, actor, target, phase,
                       kind, salience)

    def _told(self, e):
        """An event as others tell it: "Fenn struck Tom with the dagger (missed)"."""
        a = e["action"]
        if a in self._native:
            thing = e["parts"].get("thing") or e["parts"].get("with")
            roles = self._roles(a, e["target"], thing, e.get("facts", {}).get("who") if a == "tell" else None)
            told = self.cat.verbs[verb_id(a, e["text"])].told_text(
                e["actor"], {r: ref.name for r, ref in roles.items()})
        else:                                   # a note: "notice that the lute is gone"
            first, _, rest = e["text"].partition(" ")
            told = f"{e['actor']} {PAST.get(first, first)} {rest}".rstrip()
        return told + (f" ({e['result']})" if e.get("result") else "")

    def beliefs(self, who):
        """Everything `who` believes: where each thing is, and who took what."""
        out = []
        for (k, thing), (value, src) in self.world.people[who].bel.items():
            b = self._beliefs.get((k, thing, value, src))
            if b is None:
                b = self._beliefs[(k, thing, value, src)] = self._belief(k, thing, value, src)
            out.append(b)
        return tuple(out)

    def _belief(self, k, thing, value, src):
        w = self.world
        if src in ("start", "sight"):
            how, t = "looked", 0
        elif src == "own":
            how, t = "own", 0
        else:
            e = w.log[int(src[1:])]
            how, t = ("told" if e["action"] == "tell" else "saw"), e["t"]
        about = self._ref("thing", thing)
        if k == "took":
            return Belief("took", about, self._ref("person", value), f"{value} took {thing}", src, how, t)
        if value in w.people:
            val, text = self._ref("person", value), f"{value} has {thing}"
        elif value in ROOMS:
            val, text = self._ref("place", value), f"{thing} is in {value}"
        else:
            val, text = value, f"{thing} is {value}"            # "gone"
        return Belief("at", about, val, text, src, how, t)

    # ------------------------------------------------------------ options
    def bind(self, who):
        """Every option the inn permits `who` now, in the inn's order. The inn
        has no blocked options: what it lists, the person can do."""
        rows = self.world.options(self.world.people[who])
        made = [self._option(r) for r in rows]
        opts, keys = tuple(o for o, _ in made), [k for _, k in made]
        self._bound[who] = (self._now(), opts, rows, keys, {k: i for i, k in enumerate(keys)})
        return opts

    def rule_scores(self, who, options):
        """(prior, prior_avg, habit) for each option (rules.py)."""
        w, p = self.world, self.world.people[who]
        rows, keys = self._rows(who, options)
        scores = (tuple(rules._utility(w, p, r, p.traits) for r in rows),
                  tuple(rules._utility(w, p, r, rules.MID) for r in rows),
                  tuple(rules.habit(r) for r in rows))
        self._scores[who] = (self._now(), keys, scores)
        return scores

    def _rows(self, who, options):
        """The inn's rows of options from the last bind, while it holds, and
        the keys of the options."""
        when, opts, rows, keys, index = self._bound.get(who, NOT_BOUND)
        if when != self._now():
            raise ValueError(f"the options of {who} are not from a bind of this moment")
        if options is opts:                     # the bind itself: no key to look up
            return rows, keys
        ks = [o.key for o in options]
        try:
            return [rows[index[k]] for k in ks], ks
        except KeyError as e:
            raise ValueError(f"{e.args[0]!r} is not an option of {who} now") from None

    def _option(self, row):
        """The Option of an inn row, and its key. Each is made once."""
        got = self._options.get((row["verb"], row["text"]))
        if got is None:
            vid = verb_id(row["verb"], row["text"])
            o = self.cat.option(NAME, vid, self._roles(row["verb"], row["target"], row["obj"],
                                                       row["f"].get("said")))
            if o.text != row["text"]:
                raise ValueError(f"the catalog says {o.text!r} where the inn says {row['text']!r}")
            got = self._options[(row["verb"], row["text"])] = (o, o.key)
        return got

    def _roles(self, action, target, thing, culprit):
        """The catalog roles of an inn option or event: its target (a room is
        a place), its thing (the instrument of a blow) and, for "tell", the
        person said to have taken the thing."""
        roles = {}
        if target in ROOMS:
            roles["place"] = self._ref("place", target)
        elif target is not None:
            roles["target"] = self._ref("person", target)
        if thing is not None:
            role = "instrument" if action == "strike" else ("thing" if "target" in roles else "target")
            roles[role] = self._ref("thing", thing)
        if culprit is not None:
            roles["culprit"] = self._ref("person", culprit)
        return roles

    def _ref(self, kind, name):
        r = self._refs.get((kind, name))
        if r is None:
            r = self._refs[(kind, name)] = Ref(kind, name, name)
        return r

    def _target(self, target):
        if target in ROOMS:
            return self._ref("place", target)
        return self._ref("person", target) if target in self.world.people else None

    # ------------------------------------------------------------ starting a job
    def start(self, who, option, why, menu=None, probs=None, probs_avg=None):
        """`who`, the person who must choose now, starts `option`. The option
        is checked again against the world as it is now. Then the turn runs
        on, up to the next person who must choose (or to its end).

        The log keeps what the person chose from. `menu` is the Menu that the
        chooser saw. Without it, that is every option of the bind, in the
        inn's order (compat mode). `probs` and `probs_avg` are the draw of
        this character and of an average character over those options; each
        one defaults to the rules' (rules.py)."""
        w = self.world
        p = w.deciding
        if p is None or p.name != who:
            raise ValueError(f"{who} is not the person who must choose now")
        if self._bound.get(who, NOT_BOUND)[0] != self._now():
            self.bind(who)
        when, opts, rows, keys, index = self._bound[who]
        k = option.key
        if k not in index:
            raise ValueError(f"{option.text!r} is not an option of {who} now")
        offered, okeys = (opts, keys) if menu is None else (menu.options, [o.key for o in menu.options])
        if k not in okeys or (menu is not None and not index.keys() >= set(okeys)):
            raise ValueError(f"the menu of {who} lacks the option picked, or has options that"
                             " the inn does not permit now")
        if probs is None or probs_avg is None:
            if menu is not None:
                prior, avg, hab = menu.prior, menu.prior_avg, menu.habit
            else:
                got = self._scores.get(who)
                prior, avg, hab = (got[2] if got and got[0] == when and got[1] == okeys
                                   else self.rule_scores(who, offered))
            if probs is None:
                probs = rules.softmax([a + h for a, h in zip(prior, hab)], rules.T_RULES)
            if probs_avg is None:
                probs_avg = rules.softmax([a + h for a, h in zip(avg, hab)], rules.T_RULES)
        row = rows[index[k]]
        choice = w.choice(p, [o.text for o in offered], okeys.index(k), probs, probs_avg)
        job = self.jobs[who] = Job(why.decision, who, option, w.t, why)
        d = {"t": w.t, "who": who, "verb": option.verb, "text": option.text, "options": len(offered)}
        d.update((f, getattr(why, f)) for f in WHY)
        self.decisions.append(d)
        verb = self.cat.verbs[option.verb]
        loud = "loud" in verb.flags
        said = ""
        if loud or "violent" in verb.flags:
            said = verb.told_text(who, {r: ref.name for r, ref in option.roles}, start=True)
            self._start_record(p, option, row, job, said, loud)
        self.job_events.append(JobEvent(w.t, who, job.id, option.verb, "start", said, why.reason))
        w.start(p, {**row, "job": job.id}, choice)
        return job

    def _start_record(self, p, option, row, job, text, loud):
        """Who sees or hears an act start: the rule of World._perceive with no
        draw. A theft, the one act seen only by chance, is not loud and not
        violent, so it has no start record."""
        w = self.world
        saw, heard = [], []
        for q in w.people.values():
            if q is p or q.dead:
                continue
            if q.room == p.room and not q.down:
                saw.append(q.name)
            elif loud and p.room in ROOMS[q.room]:
                heard.append(q.name)
        s = {"id": f"s{len(self.starts)}", "t": w.t, "actor": p.name, "action": row["verb"],
             "verb": option.verb, "target": row["target"], "text": text, "room": p.room,
             "job": job.id, "after": w.n, "saw": saw, "heard": heard}
        self.starts.append(s)
        for name in saw + heard:
            self._seen.setdefault(name, []).append(s)

    def _ended(self, p, d, ev, status, reason):
        """World.on_end: the act of a job ended."""
        job = self.jobs.get(p.name)
        if job is None or job.id != d.get("job"):
            return
        w = self.world
        job.status, job.ended, job.reason = status, w.t, reason
        text = self._told(ev) if status == "done" else ""
        self.job_events.append(JobEvent(w.t, p.name, job.id, job.option.verb, status, text, reason))
        self._free[p.name] = ("done" if status == "done" else f"failed:{reason}" if status == "failed"
                              else reason if reason.startswith("interrupt:") else f"interrupt:{reason}")


def night(seed=0, turns=300, chooser="rules", bandit=None, until_down=True):
    """One night of the inn in COMPAT MODE, run through the Backend protocol:
    each choice offers every option in the inn's order, and the lab chooser
    `chooser` ("rules" or "random") draws with the world's own rng. The
    events are the lab's (World(seed, bandit, turns), then World.step(policy)
    for each turn). `until_down` stops the night when every person is dead or
    down, as lab/world2/run.py and golden.py do; lab/world2/test_world.py
    runs every turn. Returns the backend.

    There is no scene text in compat mode, so the Why of each decision names
    as its percepts the events that the person perceived since its last
    choice (the lab's "seen")."""
    b = RoomsBackend(seed, turns, bandit)
    pick = rules.LAB_CHOOSERS[chooser](b.world.rng)
    for _ in range(turns):
        b.step()
        while b.needs_decision():
            (who, reason), = b.needs_decision()
            opts = b.bind(who)
            # the lab's Random reads no score, and spends no time on one
            prior, avg, hab = b.rule_scores(who, opts) if pick.scored else ((0.0,) * len(opts),) * 3
            req = DecisionRequest(f"{seed}:{b.world.t}:{who}", who, reason, "", QUESTION.format(who=who),
                                  tuple(o.text for o in opts), tuple(o.group for o in opts), prior, avg, hab)
            dec, pa = pick.draw(req)            # submit and poll, and the draw of the average character
            pc, i = dec.probs, dec.pick
            why = Why(req.id, pick.name, reason, tuple(b.world.people[who].pending), (), pc[i],
                      pc[i] - max((x for j, x in enumerate(pc) if j != i), default=0.0))
            b.start(who, opts[i], why, probs=pc, probs_avg=pa)
        if until_down and b.over():
            break
    return b
