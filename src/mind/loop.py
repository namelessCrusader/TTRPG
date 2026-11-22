"""The loop of the mind: one decision from start to end, and the turns of a
backend that runs in turns (docs/sim-mind-design.md 2.3 and 3.5, step M5).

ONE DECISION (Mind.decide), for a person who must choose now:
1. The backend binds the catalog verbs (`bind`), and gives three rule
   scores for each option (`rule_scores`).
2. core.menu makes the menu: MENU_MAX options or fewer, in canonical order,
   with the null act. A player's menu is never cut. core.scene and
   describe write the scene text.
3. The chooser of that person gets a DecisionRequest, and picks.
4. The backend starts the pick as a job (`start`), and checks it again
   against the world as it is now. The job keeps the Why of its decision.
5. A decision row goes to `Mind.rows`: the request, the menu, the cut
   options, the pick, the chooser and the Why.

WHO PICKS (MindConfig.mode):
    reactive   the default. Every NPC has the rules (`npc`). The model is
               never asked.
    featured   the model picks for the `featured` people, and for a person
               that the player addressed, at that person's next decision.
               Every other NPC has the rules.
    all        the model picks for every NPC. For offline evaluation only.
The player (`player`) always picks for itself, from the full menu. The
model is used as the design says: Combined(sense=model), behind
Gate(gate) when `gate` is not None. The rules of the gate draw with the
rules' temperature, whatever `npc` is.

THE REMOTE COUNTER. The design's budget (2.4, and the memory note
latency-budget): reactive mode makes 0 model calls on every tick, interrupt
ticks too. Featured mode has at most 1 outstanding request for each
featured character. `Mind.remote` counts the requests that go to the model in
each tick, with an entry for every tick, also when it is 0. A second request
for a person while one is outstanding is an error.

THE MENU (MindConfig.menu, step M12):
    flat       the default. One question: the menu of core.menu, pre-ranked
               to `cap` options or fewer.
    two        small questions (scope.py): first a SCOPE (level 1, at most
               `scopes` entries: "deal with Oren", "go somewhere or leave",
               "wait", "something else"), then an ACT inside it (level 2, at
               most `acts` options). Nothing is cut, and every legal option
               can be picked: the scopes hold every one of them. A menu too
               big for two questions (more than `scopes` x `acts`) gives
               "something else" a page of the scopes it folded: a third
               question, asked only when "something else" is picked. A level
               with one entry is not asked. The request ids are the
               decision's id with ":scope" (":scope2", ... for the pages that
               "something else" opens) and ":act" after it. The backend's log
               keeps the questions as one draw over the full menu: P(scope) x
               P(option | scope), the scopes not asked drawn by the rules.
A player always picks from the full flat menu.

THE BATCH (MindConfig.batch, wave 14b lane F; None keeps the old pre-rank).
With a number B, the flat menu of an NPC is not cut. A menu of B options or
fewer is asked as it always was. A longer one is a TOURNAMENT (tourney.py):
canonical order, batches of B or fewer, a pick in each, the winners picked
again. The same options in any order give the same pick. Rules cost no model
call; a model chooser gets about n/(B-1) small questions, each counted by
the remote counter, so reactive mode stays at 0.

WHAT A BACKEND GIVES. The Backend protocol (api.py), and two more things:
- `start` takes the keywords `menu`, `probs` and `probs_avg`: the Menu that
  the chooser saw, and the draws of this character and of an average one
  over its options. The backend's log keeps them (the inn's events keep
  them as `options`, `p_char` and `p_avg`). `probs_avg` is None when the
  chooser has no average; then the backend uses the rules' average.
- `run` drives a backend that runs in TURNS (src/rooms): `step()` starts a
  turn and runs it up to the first person who must choose, each `start`
  runs it on, and `over()` says that the run is over.
A person's Ref.id is its `who` (the backend's id of the person), so the
mind can see whom the player addressed.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

from . import core, describe, scope
from .tourney import Tourney
from .api import Decision, DecisionRequest, Menu, Why
from .choosers import Combined, Gate, Player, Random, Rules, one_hot

MODES = ("reactive", "featured", "all")
MENUS = ("flat", "two")                         # one question, or scope then act (scope.py)
NPC = {"rules": Rules, "random": Random}        # the choosers an NPC can have, by name
WHY = tuple(f.name for f in fields(Why))


@dataclass(frozen=True)
class MindConfig:
    """Who picks, and how (see the top of this file)."""
    mode: str = "reactive"                  # one of MODES
    featured: tuple[str, ...] = ()          # featured mode: who the model picks for
    player: str | None = None               # who the player is, or None
    npc: str = "rules"                      # the chooser of every other NPC, a key of NPC
    gate: float | None = 0.7                # the rules pick when their top probability is this or more
    seed: int = 0                           # the world seed: a chooser that draws is seeded from it
    setting: str = ""                       # the first sentence of every scene text
    cap: int = core.MENU_MAX                # the most options of an NPC menu
    menu: str = "flat"                      # one of MENUS
    scopes: int = scope.SCOPE_MAX           # two: the most entries of level 1
    acts: int = scope.ACT_MAX               # two: the most options of level 2
    batch: int | None = None                # flat: None keeps the pre-rank cut; a number MERGES
                                            # the full menu in batches of this size (tourney.py)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode {self.mode!r} is not one of {MODES}")
        if self.menu not in MENUS:
            raise ValueError(f"menu {self.menu!r} is not one of {MENUS}")
        if self.npc not in NPC:
            raise ValueError(f"npc chooser {self.npc!r} is not one of {tuple(NPC)}")
        if self.batch is not None and self.batch < 2:
            raise ValueError("a batch has room for two options at least")
        object.__setattr__(self, "featured", tuple(self.featured))


class _Metered:
    """The model, as the mind uses it: it counts each request that goes to
    the model in the mind's tick, and refuses a second request for a person
    while one is outstanding."""

    def __init__(self, model, mind):
        self.model, self.mind = model, mind
        self.name = model.name
        self.remote = getattr(model, "remote", False)
        self.waiting = {}                   # who -> the id of its outstanding request

    def submit(self, req):
        if req.who in self.waiting:
            raise RuntimeError(f"a second request for {req.who} while {self.waiting[req.who]} is outstanding")
        m = self.mind
        m.remote[m.tick] = m.remote.get(m.tick, 0) + 1
        self.waiting[req.who] = req.id
        self.model.submit(req)

    def poll(self, req_id, block=True):
        d = self.model.poll(req_id, block)
        if d is not None:
            self.waiting = {w: r for w, r in self.waiting.items() if r != req_id}
        return d


class Mind:
    """The mind of every person of one backend. `model` is the model chooser
    (featured and all modes need one). `ask` is the player's (choosers.Player).

    It keeps:
        tick       the tick (or turn) that `at` set
        remote     tick -> the number of requests that went to the model
        rows       one decision row for each job started: plain JSON
        jobs       who -> its last Job
        addressed  the people the player addressed, who have not chosen since
    """

    def __init__(self, config=None, model=None, ask=None, cat=None):
        self.config = c = config or MindConfig()
        self.cat = cat or core.catalog()
        self.tick = 0
        self.remote = {}
        self.rows = []
        self.jobs = {}
        self.addressed = set()
        self._ids = set()
        self.npc = NPC[c.npc](c.seed)
        self.model = None
        if model is not None:
            combined = Combined(_Metered(model, self), seed=c.seed)
            self.model = combined if c.gate is None else Gate(c.gate, Rules(c.seed), combined)
        elif c.mode != "reactive":
            raise ValueError(f"{c.mode} mode needs a model chooser")
        if (c.player is None) != (ask is None):
            raise ValueError("a player needs `ask`, and `ask` needs a player")
        self.player = None if ask is None else Player(ask)

    def at(self, tick):
        """The tick (or turn) of the decisions that follow."""
        self.tick = tick
        self.remote.setdefault(tick, 0)

    def chooser(self, who):
        """The chooser of `who` now."""
        c = self.config
        if who == c.player:
            return self.player
        if c.mode == "all" or (c.mode == "featured" and (who in c.featured or who in self.addressed)):
            return self.model
        return self.npc

    def decide(self, backend, who, reason):
        """`who` chooses now, and the backend starts the pick. Returns the Job."""
        c = self.config
        chooser = self.chooser(who)
        player = chooser is self.player
        if c.menu == "two" and not player:
            return self._two(backend, who, reason, chooser)
        opts = backend.bind(who)
        prior, avg, hab = backend.rule_scores(who, opts)
        # BATCHED THEN MERGED (tourney.py): nothing is cut, a long menu is picked in
        # small batches and the winners are picked again. A menu of `batch` or
        # fewer is asked as it always was.
        m = core.menu(opts, prior, avg, hab, cat=self.cat, player=player or c.batch is not None,
                      cap=c.cap)
        if c.batch is not None and not player and len(m.options) > c.batch:
            chooser = Tourney(chooser, c.batch)
        state = backend.self_state(who)
        sc = core.scene(state, backend.percepts(who), backend.beliefs(who), m, cat=self.cat,
                        backend=backend.name)
        req = DecisionRequest(self._id(who), who, reason, describe.describe(sc, c.setting),
                              describe.question(state), tuple(o.text for o in m.options),
                              tuple(o.group for o in m.options), m.prior, m.prior_avg, m.habit)
        chain = self._chain(who, reason, m) if player else ()
        if chain:                           # the next option of the player's chain: nobody is asked
            dec = Decision(req.id, m.options.index(chain[0]), "chain")
            then, pa = chain[1:], None
        else:
            dec = self._ask(chooser, req)
            then = tuple(m.options[j] for j in dec.then)
            pa = chooser.average(req)
        n, i = len(req.options), dec.pick
        pc = one_hot(n, i) if dec.probs is None else tuple(dec.probs)
        if len(pc) != n or (pa is not None and len(pa) != n):
            raise ValueError(f"{dec.chooser}: one probability for each option of request {req.id}")
        psrc, bsrc = sc.sources
        why = Why(req.id, dec.chooser, reason, psrc, bsrc, pc[i],
                  pc[i] - max((x for j, x in enumerate(pc) if j != i), default=0.0), dec.fallback)
        option = m.options[i]
        job = backend.start(who, option, why, menu=m, probs=pc, probs_avg=pa)
        job.then = then
        self.jobs[who] = job
        self.addressed.discard(who)
        if player:
            target = option.role("target")
            if target is not None and target.kind == "person" and target.id != who:
                self.addressed.add(target.id)
        self.rows.append({
            "id": req.id, "t": self.tick, "who": who, "reason": reason, "chooser": dec.chooser,
            "spelling": describe.SPELLING, "situation": req.situation, "question": req.question,
            "options": list(req.options), "groups": list(req.groups), "cut": [o.text for o in m.cut],
            "blocked": [[o.text, list(o.missing)] for o in m.blocked], "pick": i, "text": option.text,
            "verb": option.verb, "then": [o.text for o in then], "probs": [round(x, 6) for x in pc],
            "job": job.id, "why": {f: getattr(why, f) for f in WHY}})
        return job

    def _two(self, backend, who, reason, chooser):
        """`who` chooses in small questions (scope.py): a scope, then an act
        in it (and a page of the scopes "something else" folded, when it
        holds too many for one act question). The backend starts the act.
        Returns the Job."""
        c, cat = self.config, self.cat
        opts = backend.bind(who)
        prior, avg, hab = backend.rule_scores(who, opts)
        full = core.menu(opts, prior, avg, hab, cat=cat, player=True)
        state = backend.self_state(who)
        sc = core.scene(state, backend.percepts(who), backend.beliefs(who), full, cat=cat,
                        backend=backend.name)
        job = self.jobs.get(who)
        last = job.option.key if job is not None and job.status in ("stopped", "failed") else None
        lv = scope.levels(full, cat, state.drives, sc.percepts, last, c.scopes, c.acts)
        rid = self._id(who)
        situation, question = describe.describe(sc, c.setting), describe.question(state)
        page, here, pages = lv, situation, []
        while True:                         # the pages: level 1, then what "something else" opens
            k = len(pages)
            r = DecisionRequest(f"{rid}:scope{k + 1 if k else ''}", who, reason, here, question,
                                page.texts, page.groups, page.prior, page.prior_avg, page.habit)
            d, p, a, asked = self._level(chooser, r)
            pages.append((page, r, d, p, a, asked))
            nxt = page.acts[d.pick]
            if not isinstance(nxt, scope.Level):
                break
            page, here = nxt, f"{situation} {scope.means(state.name, page.scopes[d.pick])}"
        m, picked = nxt, page.scopes[d.pick]
        r2 = DecisionRequest(f"{rid}:act", who, reason, f"{situation} {scope.means(state.name, picked)}",
                             question, tuple(o.text for o in m.options), tuple(o.group for o in m.options),
                             m.prior, m.prior_avg, m.habit)
        d2, p2, a2, asked2 = self._level(chooser, r2)
        option = m.options[d2.pick]
        decs = [(x[2], x[5]) for x in pages] + [(d2, asked2)]
        # (a level not asked picks nothing)
        who_picked = next((d.chooser for d, ok in reversed(decs) if ok), pages[0][2].chooser)

        # one draw over every option of the full menu, for the backend's log
        def rules(members, avg=False):
            z = full.prior_avg if avg else full.prior
            return core.softmax([z[i] + full.habit[i] for i in members], core.T_RULES)

        pc = scope.joint(lv, [(x[0], x[2].pick, x[3]) for x in pages], p2, rules)
        pa = scope.joint(lv, [(x[0], x[2].pick, x[4]) for x in pages], a2, lambda ms: rules(ms, True))
        i = full.options.index(option)
        psrc, bsrc = sc.sources
        why = Why(rid, who_picked, reason, psrc, bsrc, pc[i],
                  pc[i] - max((x for k, x in enumerate(pc) if k != i), default=0.0),
                  any(d.fallback for d, _ in decs))
        job = backend.start(who, option, why, menu=full, probs=pc, probs_avg=pa)
        job.then = ()
        self.jobs[who] = job
        self.addressed.discard(who)

        def said(page, r, d, p, a, asked):
            return {"id": r.id, "situation": r.situation, "options": list(r.options),
                    "kinds": [s.kind for s in page.scopes], "pick": d.pick, "chooser": d.chooser,
                    "probs": [round(x, 6) for x in p], "prior": list(page.prior),
                    "prior_avg": list(page.prior_avg),
                    "members": [[full.options[k].text for k in s.members] for s in page.scopes],
                    "folded": [s.text for s in page.folded], "asked": asked}

        self.rows.append({
            "id": rid, "t": self.tick, "who": who, "reason": reason, "chooser": who_picked,
            "spelling": describe.SPELLING, "situation": r2.situation, "question": question,
            "options": list(r2.options), "groups": list(r2.groups), "cut": [o.text for o in m.cut],
            "blocked": [[o.text, list(o.missing)] for o in full.blocked], "pick": d2.pick,
            "text": option.text, "verb": option.verb, "then": [],
            "probs": [round(x, 6) for x in p2], "job": job.id,
            "why": {f: getattr(why, f) for f in WHY},
            "menu": "two", "asked": [x[5] for x in pages] + [asked2],
            "full": [o.text for o in full.options], "full_groups": [o.group for o in full.options],
            "joint": round(pc[i], 6),
            "act": {"id": r2.id, "prior": list(m.prior), "prior_avg": list(m.prior_avg),
                    "habit": list(m.habit), "members": [o.text for o in m.full], "scope": picked.text},
            "scope": said(*pages[0]),
            "else": [said(*x) for x in pages[1:]]})
        return job

    def _level(self, chooser, req):
        """(Decision, its probabilities, the average character's, whether it
        was asked) of one level. A level with one entry is not asked."""
        n = len(req.options)
        if n == 1:
            return Decision(req.id, 0, chooser.name, (1.0,)), (1.0,), (1.0,), False
        d = self._ask(chooser, req)
        pc = one_hot(n, d.pick) if d.probs is None else tuple(d.probs)
        pa = chooser.average(req)
        if pa is None:                      # the rules' average, as a backend uses it
            pa = Rules().average(req)
        if len(pc) != n or len(pa) != n:
            raise ValueError(f"{d.chooser}: one probability for each option of request {req.id}")
        return d, pc, tuple(pa), True

    def _id(self, who):
        """The id of a new request: the world seed, the tick and who. A second
        request of one person in one tick gets a number after it."""
        rid = base = f"{self.config.seed}:{self.tick}:{who}"
        k = 1
        while rid in self._ids:
            k += 1
            rid = f"{base}:{k}"
        self._ids.add(rid)
        return rid

    def _chain(self, who, reason, m):
        """The rest of the player's chain, when its next option runs now: the
        last job was done, and that option is legal now. Else nothing: the
        player is asked again."""
        job = self.jobs.get(who)
        if job is None or not job.then or reason != "done" or job.then[0] not in m.options:
            return ()
        return job.then

    @staticmethod
    def _ask(chooser, req):
        chooser.submit(req)
        d = chooser.poll(req.id, block=True)
        if d is None or d.request != req.id:
            raise RuntimeError(f"{chooser.name} gave no decision for request {req.id}")
        if not 0 <= d.pick < len(req.options) or not all(0 <= j < len(req.options) for j in d.then):
            raise ValueError(f"{d.chooser} picked an option that request {req.id} does not have")
        return d


def run(backend, mind, turns, until_over=True):
    """Run `turns` turns of a backend that runs in turns. The tick of the
    mind is the number of the turn, from 1. With `until_over`, the run stops
    when `backend.over()` says so, as the lab's runs stop when every person
    is dead or down. Returns the mind."""
    for t in range(1, turns + 1):
        mind.at(t)
        backend.step()
        while True:
            need = backend.needs_decision()
            if not need:
                break
            mind.decide(backend, *need[0])
        if until_over and backend.over():
            break
    return mind
