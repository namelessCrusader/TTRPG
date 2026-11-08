"""Record and replay: a decision model chooses for a run without running in
the run's process (docs/sim-mind-design.md 3.5, the model crossing, step M6).

The model (GLiNER2.5-Decide) runs in the lab, in the lab's own venv. The lab
and src never import each other: they exchange JSONL files. The default
crossing has three parts:

1. RECORD. A Recorder stands between the mind and a backend, and keeps the
   menu of each decision. `Recorder.cases` gives each decision of the run as
   a lab case row: the scene text, the question and the options that the
   chooser saw, with the full menu and the cut on record. `write` writes
   the rows.
2. SCORE. lab/methods/score_requests.py reads the case rows in the lab venv,
   one process at a time. It gives the model the situation, the question
   and the options of each row, nothing more, and writes one decision row
   for each.
3. REPLAY. `Replay` is the model of a run that applies those decisions. The
   mind gets it as its model chooser (loop.Mind(model=...)), so Combined and
   the gate use it as they would use a live model: Replay gives the model's
   scores, and Combined adds the traits. Replay finds a decision by the
   request id, and only uses it when it answers the same question (the same
   `digest`).

A CASE ROW. It has every field that lab/methods/run.py reads (id,
situation, question, options, factor, value and good), so the lab's runner
can read a file of them too: with --perms it measures how much the order of
the options moves the pick.
    id          the id of the request
    kind        "mind"
    factor      "reason", and `value` is the kind of reason: idle, done,
                failed or interrupt
    good        []: a request has no gold answer
    situation   the scene text that the chooser read
    question    "What does Fenn do next?"
    options     the option texts of the menu, in canonical order
    groups      the group of each option
    full_options  each text of the full menu (every legal option before
                the pre-rank cut) once, in canonical order. A player's
                menu is never cut: there it is `options`.
    cut         the option texts that the pre-rank cut
    prior, prior_avg, habit   the rule numbers of each option. The model
                never gets them. They let the lab compare the model with
                the rules, and do what Combined does.
    who, t, reason, spelling, backend   who decided, at which tick (or
                turn), why it was asked, the spelling of the text
                (describe.SPELLING) and the backend
    chooser, pick   the chooser that picked in the recorded run, and the
                index of its pick
    digest      the question in 16 hex digits (`digest`)

TWO LEVELS (MindConfig.menu "two", scope.py). A decision made in small
questions gives a case row for each question asked, and one more, each with
`level`:
    "scope"     a page of scopes (level 1, `page` 1; or the page that
                "something else" opened, `page` 2 or more): the scope texts
                as `options`; `members` gives the texts of the full menu
                that each scope holds, and `full_options` the full menu. So
                the lab can measure RECALL: does the scope the model picks
                hold the option it picks from the full menu?
    "act"       level 2: the options of the scope picked (all of them,
                nothing is cut), the scene text with the scope it means to
                deal with, and `scope`, the text of that scope
    "full"      the full menu, uncut, as one question that the run never
                asked (chooser and pick are null): the model's full-menu
                pick, for recall
A level with one entry was not asked, and has no row; the "full" row is
always there. Their ids are the decision's id with ":scope" (":scope2", ...),
":act" and ":full" after it.

EVERY QUESTION OF A DECISION (`tree`, wave 11). The rows above are the
questions the RUN asked: the act question of the scope the run's chooser
picked. To follow another chooser down its own path (the model, scoring
offline), the lab needs the questions the run did not ask too. `tree` gives
every question one Level can ask: each page (level 1 and each page that
"something else" opens, asked or not) and the act question of each entry,
in the words the loop would use. Each row says where it is:
    level "scope"  page k (1, 2, ...); `kinds`, and `members` (the full-menu
                   texts of each entry); `asked` False when it has one entry
    level "act"    page k, entry j: the options of entry j of page k;
                   `asked` False when it has one option
    level "full"   the full menu as one question, as in `cases_two`
The ids are the decision's id with ":scope", ":scope2", ..., ":act<k>.<j>"
and ":full" after it, each behind `tag` but the "full" one: two layouts of
one decision (tag "v1|", "v2|") share their full-menu question.

A DECISION ROW (lab/methods/score_requests.py writes it).
    request     the id of the request
    digest      the digest of the case row, copied
    pick        the index of the model's best option
    scores      the model's score of each option, or null when the scorer
                gives none. Combined needs scores.
    chooser     the name of the scorer
The lab can add more fields (the time of the forward pass): Replay does not
read them.

DETERMINISTIC AND REALTIME (design 3.5). In deterministic mode (the
default), a request with no decision for its question is an error
(choosers.Missing), and the run stops. In realtime mode the rules' argmax
answers it, and the decision says that it is a fallback (the Why of the
job keeps that). Inside Combined, which hides the rule scores from the
model, that fallback has no view of its own: every option gets the same
score, and the traits and habit pick among them.

A RUN THAT GOES ANOTHER WAY. The decisions answer the questions of the run
that was recorded. When the model picks an option that the recorded run did
not pick, the run goes another way after that pick, and later requests ask
other questions. A request id can then come back with another question: its
old decision does not answer it. So a run that the model drives (a closed
loop) is made in rounds:
1. replay in realtime mode, and record the requests that Replay missed
   (`Replay.missed`; `Recorder.cases(rows, ids=...)`);
2. score them, and add their rows to the decisions file;
3. replay again.
Each round answers at least the first request that the last round missed.
A round with no fallback is the run of the model, and it replays exactly in
deterministic mode. A decisions file can hold two rows for one request id,
one for each question; each (id, digest) pair occurs once.

A Recorder or a Replay is for one run.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

from . import choosers, scope
from .api import Decision

KIND = "mind"                       # the `kind` of a case row
REASON = "reason"                   # the `factor` of a case row


def digest(situation, question, options):
    """The question of one decision, in 16 hex digits. Two requests with the
    same scene text, question and options (in the same order) have the same
    digest, in every process."""
    data = json.dumps([situation, question, list(options)], ensure_ascii=False)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def case(row, menu, backend=""):
    """One decision as a lab case row (see the top of this file). `row` is
    the decision row of the mind (loop.Mind.rows), and `menu` the Menu that
    the chooser saw."""
    if [o.text for o in menu.options] != list(row["options"]):
        raise ValueError(f"request {row['id']}: the menu is not the menu of its row")
    return {
        "id": row["id"], "kind": KIND, "factor": REASON, "value": row["reason"].split(":")[0],
        "good": [], "situation": row["situation"], "question": row["question"],
        "options": list(row["options"]), "groups": list(row["groups"]),
        "full_options": list(dict.fromkeys(o.text for o in menu.full)), "cut": list(row["cut"]),
        "prior": list(menu.prior), "prior_avg": list(menu.prior_avg), "habit": list(menu.habit),
        "who": row["who"], "t": row["t"], "reason": row["reason"], "spelling": row["spelling"],
        "backend": backend, "chooser": row["chooser"], "pick": row["pick"],
        "digest": digest(row["situation"], row["question"], row["options"]),
    }


def cases_two(row, backend=""):
    """The case rows of one decision made in small questions (see the top
    of this file). `row` is its decision row: it holds every level."""
    act = row["act"]
    common = {"kind": KIND, "factor": REASON, "value": row["reason"].split(":")[0], "good": [],
              "question": row["question"], "who": row["who"], "t": row["t"], "reason": row["reason"],
              "spelling": row["spelling"], "backend": backend}
    out = []
    for k, sc in enumerate([row["scope"]] + row["else"]):
        if sc["asked"]:
            opts = sc["options"]
            out.append({"id": sc["id"], **common, "level": "scope", "page": k + 1,
                        "situation": sc["situation"], "options": list(opts),
                        "groups": ["scope"] * len(opts), "full_options": list(row["full"]), "cut": [],
                        "members": sc["members"], "kinds": sc["kinds"], "prior": sc["prior"],
                        "prior_avg": sc["prior_avg"], "habit": [0.0] * len(opts),
                        "chooser": sc["chooser"], "pick": sc["pick"],
                        "digest": digest(sc["situation"], row["question"], opts)})
    if row["asked"][-1]:
        out.append({"id": act["id"], **common, "level": "act", "situation": row["situation"],
                    "options": list(row["options"]), "groups": list(row["groups"]),
                    "full_options": list(act["members"]), "cut": list(row["cut"]),
                    "scope": act["scope"], "prior": act["prior"],
                    "prior_avg": act["prior_avg"], "habit": act["habit"], "chooser": row["chooser"],
                    "pick": row["pick"],
                    "digest": digest(row["situation"], row["question"], row["options"])})
    full = list(row["full"])
    situation = row["scope"]["situation"]
    out.append({"id": f"{row['id']}:full", **common, "level": "full", "situation": situation,
                "options": full, "groups": list(row["full_groups"]), "full_options": full, "cut": [],
                "chooser": None, "pick": None, "digest": digest(situation, row["question"], full)})
    return out


def tree(rid, lv, situation, question, name, tag="", **more):
    """Every question that one decision on Level `lv` can ask (the top of
    this file): `situation` is its scene text, `question` its question,
    `name` the name of who decides; `more` goes into every row."""
    full = [o.text for o in lv.full.options]
    common = {"kind": KIND, "good": [], "question": question, **more}
    out, page, here, k = [], lv, situation, 1
    while True:
        sid = f"{tag}{rid}:scope{k if k > 1 else ''}"
        opts = list(page.texts)
        out.append({"id": sid, **common, "level": "scope", "page": k, "situation": here,
                    "options": opts, "kinds": [s.kind for s in page.scopes],
                    "members": [[full[i] for i in s.members] for s in page.scopes],
                    "asked": len(opts) > 1, "digest": digest(here, question, opts)})
        nxt = None
        for j, (s, a) in enumerate(zip(page.scopes, page.acts)):
            if isinstance(a, scope.Level):
                nxt = s, a
                continue
            at = f"{situation} {scope.means(name, s)}"
            acts = [o.text for o in a.options]
            out.append({"id": f"{tag}{rid}:act{k}.{j}", **common, "level": "act", "page": k, "entry": j,
                        "situation": at, "options": acts, "scope": s.text, "asked": len(acts) > 1,
                        "digest": digest(at, question, acts)})
        if nxt is None:
            break
        here, page, k = f"{situation} {scope.means(name, nxt[0])}", nxt[1], k + 1
    out.append({"id": f"{rid}:full", **common, "level": "full", "situation": situation, "options": full,
                "asked": True, "digest": digest(situation, question, full)})
    return out


class Recorder:
    """A backend as the mind sees it, with the menu of each decision kept.

    The mind gives `start` the Menu that the chooser saw (loop.py). The
    Recorder keeps it under the id of the request, then hands everything on
    to the backend. Every other attribute is the backend's own. Thus

        rec = Recorder(RoomsBackend(seed))
        mind = loop.run(rec, Mind(config), turns)
        write(path, rec.cases(mind.rows))
    """

    def __init__(self, backend):
        self.backend = backend
        self.menus = {}                 # request id -> the Menu its chooser saw

    def __getattr__(self, name):
        if name == "backend":           # not set yet: no endless lookup
            raise AttributeError(name)
        return getattr(self.backend, name)

    def start(self, who, option, why, **kw):
        if kw.get("menu") is None:
            raise ValueError(f"request {why.decision}: the mind gave no menu to record")
        self.menus[why.decision] = kw["menu"]
        return self.backend.start(who, option, why, **kw)

    def cases(self, rows, ids=None):
        """The case rows of the decision rows `rows`, in their order. With
        `ids`, only the rows of those request ids: for a decision made in
        two questions, its case rows whose ids are in `ids` (a Replay misses
        a level by its own id)."""
        out = []
        for r in rows:
            if r.get("menu") == "two":
                out += [c for c in cases_two(r, self.backend.name) if ids is None or c["id"] in ids]
            elif ids is None or r["id"] in ids:
                out.append(case(r, self.menus[r["id"]], self.backend.name))
        return out


class Replay(choosers.Replay):
    """The decisions of a decisions file, by request id and question.

    `rows` are decision rows (see the top of this file; `read` reads a
    file). `mode` is "deterministic" (a request with no decision for its
    question raises Missing) or "realtime" (the rules' argmax answers it, as
    a fallback). `missed` lists the ids of the requests that had no decision,
    in order: record them for the next round."""
    MODES = ("deterministic", "realtime")

    def __init__(self, rows, mode="deterministic"):
        if mode not in self.MODES:
            raise ValueError(f"replay mode {mode!r} is not one of {self.MODES}")
        self.mode = mode
        self.missed = []
        self.decisions = {}             # (request id, digest) -> Decision
        for r in rows:
            k = (r["request"], r["digest"])
            if k in self.decisions:
                raise ValueError(f"two decisions for request {k[0]} with the question {k[1]}")
            s = r["scores"]
            if s is not None and not 0 <= r["pick"] < len(s):
                raise ValueError(f"request {k[0]}: the pick {r['pick']} has no score")
            self.decisions[k] = Decision(k[0], r["pick"], r["chooser"],
                                         None if s is None else tuple(float(x) for x in s))
        # the stub's picks stay empty: `decide` reads `decisions`
        super().__init__({}, choosers.Rules(argmax=True) if mode == "realtime" else None)

    def decide(self, req):
        got = self.decisions.get((req.id, digest(req.situation, req.question, req.options)))
        if got is not None:
            return got
        self.missed.append(req.id)
        if self.fallback is None:
            other = any(rid == req.id for rid, _ in self.decisions)
            raise choosers.Missing(f"no recorded decision for request {req.id}"
                                   + (": the recorded one answers another question" if other else ""))
        self.fallback.submit(req)
        return dataclasses.replace(self.fallback.poll(req.id), fallback=True)


# ---------------------------------------------------------------- the files
def write(path, rows):
    """Write rows as JSON lines, the whole file at once: a run that stops
    leaves no half file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(path)


def read(path):
    """The rows of a JSON lines file."""
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
