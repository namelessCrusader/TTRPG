"""Record and replay (docs/sim-mind-design.md 3.5, the model crossing, step M6).

The tests do four things:
- RECORD: a night of the inn, through the Recorder, gives lab case rows: the
  fields that the lab's runner reads, the menu and the full menu, the cut,
  and the rule numbers that the run used.
- RECORD, SCORE, REPLAY: the requests of a night are scored by
  lab/methods/score_requests.py with its fake scorer, in another process,
  and the night replayed with those decisions is the same night, byte for
  byte.
- A MISSING DECISION: deterministic mode stops the run; realtime mode lets
  the rules answer, says so, and lists what it missed. Rounds of record,
  score and replay end in the run of the model itself.
- THE CROSSING: the scorer reads only the texts of a request, and needs no
  torch. The lab and src never import each other.

This file never imports the lab. It runs the lab's scorer in a separate
process, and it reads the lab's runner as text.
"""
import ast
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from src.mind import core
from src.mind.api import Decision
from src.mind.choosers import Missing
from src.mind.loop import Mind, MindConfig, run
from src.mind.record import Recorder, Replay, digest, read, write
from src.rooms import RoomsBackend, jsonl
from src.rooms.__main__ import SETTING

ROOT = Path(__file__).resolve().parents[1]
SCORER = "lab.methods.score_requests"
CAT = core.catalog()


def fake_scores(options):
    """The fake scorer of lab/methods/score_requests.py, written out again
    here: the same options get the same scores, and they sum to 1."""
    s = [(int(hashlib.sha256(o.encode()).hexdigest()[:8], 16) % 1000 + 1) / 1001 for o in options]
    total = sum(s)
    return [x / total for x in s]


class Sense:
    """A live model for the tests: it scores the options as the fake scorer
    does, and counts the requests it gets."""
    name, remote = "fake", True

    def __init__(self):
        self.seen, self._done = [], {}

    def submit(self, r):
        self.seen.append(r)
        p = fake_scores(r.options)
        self._done[r.id] = Decision(r.id, max(range(len(p)), key=p.__getitem__), self.name, tuple(p))

    def poll(self, req_id, block=True):
        return self._done.pop(req_id, None)

    def average(self, r):
        return None


def night(seed, turns, model=None, **cfg):
    """One night of the inn through a Recorder: (recorder, mind)."""
    rec = Recorder(RoomsBackend(seed, turns))
    mind = Mind(MindConfig(seed=seed, setting=SETTING, **cfg), model=model)
    run(rec, mind, turns)
    return rec, mind


def decision(c, scores=None, pick=0):
    """A decision row for the case row `c`, as the lab writes one."""
    return {"request": c["id"], "digest": c["digest"], "pick": pick,
            "scores": fake_scores(c["options"]) if scores is None else scores, "chooser": "fake"}


def score(requests, out):
    """lab/methods/score_requests.py with the fake scorer, in another process."""
    got = subprocess.run([sys.executable, "-m", SCORER, "fake", str(requests), str(out)], cwd=ROOT,
                         capture_output=True, text=True, timeout=120, env={**os.environ, "LAB_DUTY": "1"})
    assert got.returncode == 0, got.stderr[-2000:]
    return got.stdout


@pytest.fixture(scope="module")
def recorded():
    """A night in reactive mode (the rules for every NPC), recorded."""
    rec, mind = night(0, 300)
    return rec, mind, rec.cases(mind.rows)


# ---------------------------------------------------------------- record
def test_a_night_RECORDS_LAB_CASE_ROWS(recorded):
    rec, mind, cases = recorded
    assert [c["id"] for c in cases] == [r["id"] for r in mind.rows] and len(cases) > 300
    # the fields that the lab's runner reads from a case
    runner = (ROOT / "lab" / "methods" / "run.py").read_text()
    wants = set(re.findall(r'\bc\["(\w+)"\]', runner))
    assert {"id", "situation", "question", "options", "factor", "value", "good"} <= wants
    reasons = set()
    for c, r in zip(cases, mind.rows):
        assert wants <= set(c) and (c["kind"], c["factor"], c["good"]) == ("mind", "reason", [])
        reasons.add(c["value"])
        assert c["situation"] == r["situation"] and c["question"] == r["question"]
        assert c["options"] == r["options"] and c["pick"] == r["pick"] and c["chooser"] == "rules"
        assert c["backend"] == "rooms" and c["spelling"] == r["spelling"] == "mark1-mind-v2"
        assert c["digest"] == digest(c["situation"], c["question"], c["options"])
        assert not re.search(r"\d", c["situation"])
        # the full menu: each text once, in canonical order; the menu and the
        # cut make it up
        full = c["full_options"]
        assert len(set(full)) == len(full) and set(full) == set(c["options"]) | set(c["cut"])
        assert [x for x in full if x in c["options"]] == c["options"]
        place = {}
        for o in rec.menus[c["id"]].full:
            place.setdefault(o.text, core.canonical(CAT, o))
        assert [place[x] for x in full] == sorted(place[x] for x in full)
        assert len(c["options"]) <= core.MENU_MAX
        # the rule numbers are the ones the rules drew with
        n = len(c["options"])
        assert len(c["prior"]) == len(c["prior_avg"]) == len(c["habit"]) == len(c["groups"]) == n
        z = [a + h for a, h in zip(c["prior"], c["habit"])]
        e = [math.exp((x - max(z)) / core.T_RULES) for x in z]
        assert r["probs"] == [round(x / sum(e), 6) for x in e]
    assert reasons == {"idle", "done", "failed", "interrupt"}
    assert sum(bool(c["cut"]) for c in cases) > 100            # the pre-rank was at work
    # the digest is the question: the text, the question and the options in their order
    d = digest("Fenn is here.", "What does Fenn do next?", ["wait", "eat"])
    assert len(d) == 16 and d not in (digest("Fenn is out.", "What does Fenn do next?", ["wait", "eat"]),
                                      digest("Fenn is here.", "What does Tom do next?", ["wait", "eat"]),
                                      digest("Fenn is here.", "What does Fenn do next?", ["eat", "wait"]))
    # only the rows asked for
    some = {cases[3]["id"], cases[7]["id"]}
    assert [c["id"] for c in rec.cases(mind.rows, ids=some)] == [cases[3]["id"], cases[7]["id"]]
    # what the Recorder hands on is the backend's own
    assert rec.name == "rooms" and rec.world is rec.backend.world


# ---------------------------------------------------------------- record, score, replay
def test_RECORD_SCORE_REPLAY_gives_the_SAME_NIGHT(tmp_path):
    sense = Sense()
    rec, mind = night(0, 300, sense, mode="all")
    assert len(sense.seen) > 100 and {r["chooser"] for r in mind.rows} == {"rules", "combined"}
    requests, decisions = tmp_path / "requests.jsonl", tmp_path / "decisions.jsonl"
    write(requests, rec.cases(mind.rows))
    out = score(requests, decisions)
    rows = read(decisions)
    assert len(rows) == len(mind.rows) and f"fake: {len(rows)} to do" in out
    assert {r["request"] for r in rows} == {r["id"] for r in mind.rows}
    # a second time: every request is done, and the file is not changed
    before = decisions.read_bytes()
    assert "fake: 0 to do" in score(requests, decisions) and decisions.read_bytes() == before
    # a request id that comes back with another question (a later round) is
    # scored again, and its first decision stays
    moved = dict(read(requests)[5], situation="Fenn is in the cellar.")
    moved["digest"] = digest(moved["situation"], moved["question"], moved["options"])
    write(tmp_path / "round2.jsonl", [moved])
    assert "fake: 1 to do" in score(tmp_path / "round2.jsonl", decisions)
    rows = read(decisions)
    assert [r["digest"] for r in rows if r["request"] == moved["id"]] == [read(requests)[5]["digest"],
                                                                          moved["digest"]]
    # the replay: the same night, byte for byte
    replay = Replay(rows)
    again, mind2 = night(0, 300, replay, mode="all")
    assert jsonl(again.world.log) == jsonl(rec.world.log)
    assert jsonl(mind2.rows) == jsonl(mind.rows)
    assert mind2.remote == mind.remote and replay.missed == []
    write(tmp_path / "again.jsonl", again.cases(mind2.rows))
    assert (tmp_path / "again.jsonl").read_bytes() == requests.read_bytes()


# ---------------------------------------------------------------- a missing decision
def test_a_MISSING_DECISION_stops_a_deterministic_replay():
    sense = Sense()
    rec, mind = night(1, 120, sense, mode="featured", featured=("Oren",))
    asked = {r.id for r in sense.seen}
    cases = [c for c in rec.cases(mind.rows) if c["id"] in asked]
    rows = [decision(c) for c in cases]
    gone = cases[len(cases) // 2]["id"]
    with pytest.raises(Missing, match=re.escape(f"request {gone}") + "$"):
        night(1, 120, Replay([d for d in rows if d["request"] != gone]), mode="featured", featured=("Oren",))
    # a decision for another question, with the same id, is no decision
    stale = [dict(d, digest="0" * 16) if d["request"] == gone else d for d in rows]
    with pytest.raises(Missing, match="another question"):
        night(1, 120, Replay(stale), mode="featured", featured=("Oren",))
    # realtime mode: the rules answer, and the Why says it was a fallback
    rp = Replay(stale, mode="realtime")
    rec2, mind2 = night(1, 120, rp, mode="featured", featured=("Oren",))
    assert rp.missed and rp.missed[0] == gone
    fell = {r["id"] for r in mind2.rows if r["why"]["fallback"]}
    assert fell == set(rp.missed)
    assert all(r["chooser"] == "combined" for r in mind2.rows if r["id"] in fell)
    # what a file of decisions must not hold
    with pytest.raises(ValueError, match="two decisions"):
        Replay(rows + rows[:1])
    with pytest.raises(ValueError, match="has no score"):
        Replay([dict(rows[0], pick=99)])
    with pytest.raises(ValueError, match="mode"):
        Replay(rows, mode="live")


def test_ROUNDS_of_record_score_replay_end_in_the_MODEL_S_OWN_NIGHT():
    """A run that the model drives, made offline: replay in realtime mode,
    score what it missed, and again, until nothing is missed. The night that
    ends it is the night of the live model, byte for byte."""
    cfg = dict(mode="featured", featured=("Oren",))
    live, mind = night(0, 100, Sense(), **cfg)
    rows, rounds, first = [], 0, -1
    while True:
        rounds += 1
        assert rounds <= 60, "the rounds do not end"
        rp = Replay(rows, mode="realtime")
        rec, got = night(0, 100, rp, **cfg)
        if not rp.missed:
            break
        ids = [r["id"] for r in got.rows]
        assert ids.index(rp.missed[0]) > first        # each round answers at least the first it missed
        first = ids.index(rp.missed[0])
        rows += [decision(c) for c in rec.cases(got.rows, ids=set(rp.missed))]
    assert rounds > 2
    assert jsonl(rec.world.log) == jsonl(live.world.log) and jsonl(got.rows) == jsonl(mind.rows)
    # and the decisions of the last round replay it in deterministic mode
    again, mind3 = night(0, 100, Replay(rows), **cfg)
    assert jsonl(mind3.rows) == jsonl(mind.rows)


# ---------------------------------------------------------------- the crossing
PROBE = """
import json, sys
import lab.methods.score_requests as S
seen = []

class Probe(S.Fake):
    def rank(self, *args):
        seen.append(args)
        return super().rank(*args)

S.make = lambda name: Probe()
S.main(["fake", sys.argv[1], sys.argv[2]])
print(json.dumps({"seen": seen, "torch": "torch" in sys.modules,
                  "src": sorted(m for m in sys.modules if m == "src" or m.startswith("src."))}))
"""


def test_the_SCORER_reads_only_the_TEXTS_and_needs_no_torch(recorded, tmp_path):
    cases = recorded[2][:20]
    write(tmp_path / "req.jsonl", cases)
    got = subprocess.run([sys.executable, "-c", PROBE, str(tmp_path / "req.jsonl"), str(tmp_path / "dec.jsonl")],
                         cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert got.returncode == 0, got.stderr[-2000:]
    out = json.loads(got.stdout.splitlines()[-1])
    assert out["seen"] == [[c["situation"], c["question"], c["options"]] for c in cases]
    assert out["torch"] is False and out["src"] == []
    rows = read(tmp_path / "dec.jsonl")
    for c, d in zip(cases, rows):
        assert (d["request"], d["digest"], d["chooser"]) == (c["id"], c["digest"], "fake")
        assert d["scores"] == fake_scores(c["options"]) and d["pick"] == d["scores"].index(max(d["scores"]))
    # a request row that lacks a field is refused, with its line
    bad = tmp_path / "bad.jsonl"
    write(bad, [{k: v for k, v in cases[0].items() if k != "digest"}])
    got = subprocess.run([sys.executable, "-m", SCORER, "fake", str(bad), str(tmp_path / "x.jsonl")],
                         cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert got.returncode != 0 and "bad.jsonl:1" in got.stderr and "lacks digest" in got.stderr


def _imports(path, package):
    """The modules that the file at `path` imports, by their full names; a
    relative import is read from `package`, the package of the file."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package.split(".")[:len(package.split(".")) + 1 - node.level] if node.level else []
            names.add(".".join(base + ([node.module] if node.module else [])))
    return names


def _under(names, top):
    return {n for n in names if n == top or n.startswith(top + ".")}


def test_the_LAB_and_SRC_never_import_each_other():
    assert not _under(_imports(ROOT / "tests" / "test_mind_record.py", "tests"), "lab")
    assert not _under(_imports(ROOT / "lab" / "methods" / "score_requests.py", "lab.methods"), "src")
    # src/mind imports only src/mind: no backend, and no lab
    got = _imports(ROOT / "src" / "mind" / "record.py", "src.mind")
    assert _under(got, "src") == {"src.mind", "src.mind.api"} and not _under(got, "lab")
