"""The loop of the mind, its choosers, and the command line of the inn
(docs/sim-mind-design.md 3.5, step M5).

The tests do four things:
- THE CHOOSERS, one by one, on requests made by hand: the rules' draw and
  its seed, the floor, the gate, the combined chooser (the lab's bias mode,
  behind the guard), the replay stub and the player.
- NIGHTS OF THE INN through the loop, as `python -m src.rooms run` runs them:
  every job started was an option of its menu; two runs give the same bytes,
  in another process too; every decision row has its Why; the nights have
  harm and help; compat mode gives the lab's nights.
- WHO PICKS: reactive mode never asks the model, on any tick; featured mode
  asks it only for the featured people and for a person the player
  addressed, with at most one request outstanding for each.
- THE PLAYER: the full menu, chains, the play loop, and a session over many
  calls.

The lab is never imported. The HARM and HELP verbs of lab/judge/signals.py
are read from the text of that file.
"""
import ast
import collections
import dataclasses
import hashlib
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.mind import core
from src.mind.api import Decision, DecisionRequest, Job, JobEvent, Option, Ref, Why
from src.mind.choosers import GUARD, Combined, Gate, Missing, Player, Random, Replay, Rules
from src.mind.loop import MindConfig, Mind, run, WHY
from src.rooms import RoomsBackend, jsonl
from src.rooms.__main__ import SETTING, Stop, main, news, play

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
T = core.T_RULES


def softmax(z, t=1.0):
    """Written out again here, so that the tests do not check the code with itself."""
    m = max(z)
    e = [math.exp((x - m) / t) for x in z]
    return [x / sum(e) for x in e]


def req(prior, prior_avg=None, habit=None, rid="r1", who="Fenn"):
    n = len(prior)
    return DecisionRequest(rid, who, "idle", "Fenn is in the main room.", "What does Fenn do next?",
                           tuple(f"option {chr(97 + i)}" for i in range(n)), ("wait",) * n, tuple(prior),
                           tuple(prior if prior_avg is None else prior_avg),
                           tuple(habit or [0.0] * n))


def decide(chooser, r):
    chooser.submit(r)
    return chooser.poll(r.id)


def _score(text):
    """A score in (0, 1) from the words of an option: the same in every process."""
    return (int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % 1000 + 1) / 1001


class Sense:
    """A model for the tests: it scores each option by its words, and keeps
    every request it gets."""
    name, remote = "fake", True

    def __init__(self, scores=None):
        self.scores, self.seen, self._done = scores or {}, [], {}

    @property
    def calls(self):
        return len(self.seen)

    def submit(self, r):
        self.seen.append(r)
        s = self.scores.get(r.options) or tuple(_score(t) for t in r.options)
        self._done[r.id] = Decision(r.id, max(range(len(s)), key=s.__getitem__), self.name, s)

    def poll(self, req_id, block=True):
        return self._done.pop(req_id, None)

    def average(self, r):
        return None


class Slow(Sense):
    """A model that never answers."""

    def poll(self, req_id, block=True):
        return None


# ---------------------------------------------------------------- the choosers
def test_RULES_draw_from_the_rules_and_replay_exactly():
    r = req([1.0, 0.5, 0.0, -1.0], [0.0, 0.5, 1.0, -1.0], [0.0, 0.2, 0.0, 0.0])
    c = Rules(seed=7)
    assert c.probs(r) == pytest.approx(softmax([1.0, 0.7, 0.0, -1.0], T))
    assert c.average(r) == pytest.approx(softmax([0.0, 0.7, 1.0, -1.0], T))
    d = decide(c, r)
    assert (d.request, d.chooser, d.probs) == ("r1", "rules", c.probs(r))
    assert c.poll("r1") is None                     # an answer is given out once
    c.submit(r)
    with pytest.raises(ValueError, match="submitted already"):
        c.submit(r)
    # the draw is fixed by (seed, who, request id): not by the requests before it
    ids = [f"r{k}" for k in range(600)]
    picks = [decide(Rules(seed=7), dataclasses.replace(r, id=i)).pick for i in ids]
    again = Rules(seed=7)
    decide(again, dataclasses.replace(r, id="another"))
    assert [decide(again, dataclasses.replace(r, id=i)).pick for i in reversed(ids)] == picks[::-1]
    assert [decide(Rules(seed=8), dataclasses.replace(r, id=i)).pick for i in ids] != picks
    assert [decide(Rules(seed=7), dataclasses.replace(r, id=i, who="Tom")).pick for i in ids] != picks
    # and the draws follow the rules' probabilities
    n = collections.Counter(picks)
    for i, p in enumerate(c.probs(r)):
        assert abs(n[i] / len(picks) - p) < 0.05, (i, n, c.probs(r))
    assert decide(Rules(argmax=True), r).pick == 0


def test_RANDOM_is_the_floor():
    r = req([5.0, 0.0, 0.0, -5.0])
    c = Random(seed=1)
    assert c.probs(r) == c.average(r) == (0.25,) * 4
    picks = collections.Counter(decide(c, dataclasses.replace(r, id=f"r{k}")).pick for k in range(400))
    assert set(picks) == {0, 1, 2, 3} and min(picks.values()) > 60


def test_the_GATE_lets_the_rules_answer_when_they_are_sure():
    sense = Sense()
    g = Gate(0.7, Rules(seed=1), Combined(sense, seed=1))
    sure, unsure = req([3.0, 0.0, 0.0]), req([0.1, 0.0, 0.05], rid="r2")
    assert max(Rules().probs(sure)) >= 0.7 > max(Rules().probs(unsure))
    d = decide(g, sure)
    assert d.chooser == "rules" and sense.calls == 0 and g.average(sure) == Rules().average(sure)
    d = decide(g, unsure)
    assert d.chooser == "combined" and sense.calls == 1 and g.poll("r2") is None
    assert g.average(unsure) is not None
    for bad in (0, 1.5):
        with pytest.raises(ValueError, match="probability"):
            Gate(bad, Rules(), sense)


def _lab_bias(scores, r, tau=0.5, beta=1.0):
    """The lab's bias mode (lab/world2/policy.py, Model.probs and _z),
    written out: (p_char, p_avg). utility() holds habit, so the bias, which
    is utility(traits) - utility(MID), is prior - prior_avg."""
    z = [math.log(max(x, 1e-9)) for x in scores]
    mu = sum(z) / len(z)
    sd = math.sqrt(sum((x - mu) ** 2 for x in z) / len(z))
    z = [(x - mu) / sd for x in z]
    hab = [h / T for h in r.habit]
    bias = [((a + h) - (b + h)) / T for a, b, h in zip(r.prior, r.prior_avg, r.habit)]
    return (softmax([x / tau + beta * bb + h for x, bb, h in zip(z, bias, hab)]),
            softmax([x / tau + h for x, h in zip(z, hab)]), z)


def test_COMBINED_is_the_lab_s_bias_mode_behind_the_guard():
    r = req([0.2, 1.5, -0.5, 0.9], [0.4, 0.2, -0.5, 0.1], [0.0, -0.6, 0.0, 0.3])
    scores = (0.5, 0.45, 0.1, 0.05)
    sense = Sense({r.options: scores})
    pc, pa, z = _lab_bias(scores, r)
    # with the guard off: the lab's bias mode
    c = Combined(sense, seed=3, guard=0)
    d = decide(c, r)
    assert d.chooser == "combined" and d.probs == pytest.approx(pc) and c.average(r) == pytest.approx(pa)
    # the model never sees the rule scores
    seen = sense.seen[-1]
    assert seen.prior == seen.prior_avg == seen.habit == (0.0,) * 4 and seen.options == r.options
    # the guard: the bias moves only the options that the model finds
    # plausible, with GUARD of the probability of its best option or more
    view = softmax([x / 0.5 for x in z])
    near = [v >= GUARD * max(view) for v in view]
    assert near == [True, True, False, False]       # the third and fourth are not plausible
    soft = decide(Combined(sense, seed=3), dataclasses.replace(r, id="r2")).probs
    logit = [x / 0.5 + (h / T) + ((a - b) / T if ok else 0.0)
             for x, a, b, h, ok in zip(z, r.prior, r.prior_avg, r.habit, near)]
    assert soft == pytest.approx(softmax(logit))
    assert soft[3] < pc[3]                          # the traits want the fourth; it does not make sense
    hard = decide(Combined(sense, seed=3, hard=True), dataclasses.replace(r, id="r3")).probs
    assert hard[2] == hard[3] == 0.0 and hard[:2] == pytest.approx(softmax(logit[:2]))
    # one option: nothing to ask
    one = decide(Combined(sense), req([0.0], rid="r4"))
    assert (one.pick, one.probs, sense.calls) == (0, (1.0,), 3)
    # a model that gives only a pick cannot be combined
    with pytest.raises(ValueError, match="no score"):
        decide(Combined(Replay({"r5": 0})), req([0.0, 1.0], rid="r5"))


def test_REPLAY_gives_back_the_picks_and_refuses_a_missing_id():
    model = Decision("r2", 1, "remote", (0.1, 0.9, 0.0))
    rp = Replay({"r1": 2, "r2": model})
    d = decide(rp, req([0.0, 0.0, 0.0]))
    assert (d.pick, d.chooser, d.probs, d.fallback) == (2, "replay", None, False)
    assert decide(rp, req([0.0, 0.0, 0.0], rid="r2")) == model
    with pytest.raises(Missing, match="r3"):
        rp.submit(req([0.0, 0.0, 0.0], rid="r3"))
    with pytest.raises(ValueError, match="not an option"):
        Replay({"r1": 5}).submit(req([0.0, 0.0]))
    d = decide(Replay({}, fallback=Rules(seed=2)), req([0.0, 1.0, 0.0], rid="r3"))
    assert d.fallback and d.chooser == "rules"


def test_a_PLAYER_picks_and_can_chain():
    d = decide(Player(lambda r: [2, 0, 1]), req([0.0, 0.0, 0.0]))
    assert (d.pick, d.then, d.probs, d.chooser) == (2, (0, 1), (0.0, 0.0, 1.0), "player")
    assert decide(Player(lambda r: 1), req([0.0, 0.0])).pick == 1
    for bad in (9, [], [0, -1], "1"):
        with pytest.raises(ValueError, match="number of an option"):
            Player(lambda r, bad=bad: bad).submit(req([0.0, 0.0]))


def test_the_CONFIG_refuses_what_it_cannot_do():
    with pytest.raises(ValueError, match="mode"):
        MindConfig(mode="dream")
    with pytest.raises(ValueError, match="npc chooser"):
        MindConfig(npc="model")
    for mode in ("featured", "all"):
        with pytest.raises(ValueError, match="needs a model"):
            Mind(MindConfig(mode=mode))
    with pytest.raises(ValueError, match="player"):
        Mind(MindConfig(player="Fenn"))
    with pytest.raises(ValueError, match="player"):
        Mind(ask=lambda r: 0)


# ---------------------------------------------------------------- nights through the command line
@pytest.fixture(scope="module")
def six(tmp_path_factory):
    """`python -m src.rooms run --seeds 6 --turns 300 --chooser rules`, in
    this process: the folder of its files."""
    out = tmp_path_factory.mktemp("rules6")
    main(["run", "--seeds", "6", "--turns", "300", "--chooser", "rules", "--out", str(out)])
    return out


def _rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_SIX_NIGHTS_start_only_jobs_of_their_menus(six):
    events = decisions = 0
    for seed in range(6):
        evs, rows = _rows(six / f"seed{seed}.events.jsonl"), _rows(six / f"seed{seed}.decisions.jsonl")
        events, decisions = events + len(evs), decisions + len(rows)
        made = collections.Counter()
        for r in rows:
            assert len(r["options"]) <= core.MENU_MAX and len(set(r["options"])) == len(r["options"])
            assert "wait" in r["groups"] and r["options"][r["pick"]] == r["text"] and r["chooser"] == "rules"
            assert not set(r["cut"]) & set(r["options"])
            made[(r["who"], tuple(r["options"]), r["pick"])] += 1
        # each choice the log keeps is a job started from its menu: the
        # texts on the menu, and the pick
        kept = collections.Counter((e["actor"], tuple(e["options"]), e["chosen"])
                                   for e in evs if "options" in e)
        assert kept and not kept - made, kept - made
        assert all(e["options"][e["chosen"]] == e["text"] for e in evs if "options" in e)
    assert events > 2000 and decisions > 2000, (events, decisions)
    # the pre-rank was at work: some menus were cut
    assert any(r["cut"] for r in _rows(six / "seed0.decisions.jsonl"))


def test_two_runs_give_the_SAME_BYTES_in_any_process(six, tmp_path):
    again = tmp_path / "again"
    main(["run", "--seeds", "6", "--turns", "300", "--out", str(again)])
    for f in sorted(six.iterdir()):
        assert (again / f.name).read_bytes() == f.read_bytes(), f.name
    # the command itself, in another process with another hash seed
    other = tmp_path / "other"
    env = {**os.environ, "PYTHONHASHSEED": "4242"}
    got = subprocess.run([sys.executable, "-m", "src.rooms", "run", "--seed", "4", "--seeds", "2",
                          "--out", str(other)], cwd=ROOT, env=env, capture_output=True, text=True,
                         timeout=300)
    assert got.returncode == 0, got.stderr[-2000:]
    assert "seed 5:" in got.stdout and "0 model calls" in got.stdout
    for f in ("seed4.events.jsonl", "seed4.decisions.jsonl", "seed5.events.jsonl", "seed5.decisions.jsonl"):
        assert (other / f).read_bytes() == (six / f).read_bytes(), f


def test_every_JOB_ROW_has_its_WHY(six):
    reasons = collections.Counter()
    for seed in range(6):
        for r in _rows(six / f"seed{seed}.decisions.jsonl"):
            why = r["why"]
            assert set(why) == set(WHY)
            assert (why["decision"], why["chooser"], why["reason"]) == (r["id"], r["chooser"], r["reason"])
            assert r["job"] == r["id"] and 0 < why["score"] <= 1 and why["fallback"] is False
            assert all(isinstance(x, str) for x in why["percepts"] + why["beliefs"])
            reasons[r["reason"].split(":")[0]] += 1
    assert set(reasons) == {"idle", "done", "failed", "interrupt"}, reasons


def _signals():
    """HARM and HELP of lab/judge/signals.py, read from its text."""
    sets = {}
    for node in ast.parse((ROOT / "lab" / "judge" / "signals.py").read_text()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in ("HARM", "HELP"):
            sets[node.targets[0].id] = ast.literal_eval(node.value)
    return sets["HARM"], sets["HELP"]


def _run_numbers(events):
    """run_numbers of lab/judge/signals.py, written out: the share of the
    finished events that harm, and that help."""
    harm, help_ = _signals()
    done = [e for e in events if e.get("done", True)]
    n = max(len(done), 1)
    return sum(e["action"] in harm for e in done) / n, sum(e["action"] in help_ for e in done) / n


def test_the_nights_have_HARM_and_HELP(six):
    for seed in range(6):
        harm, help_ = _run_numbers(_rows(six / f"seed{seed}.events.jsonl"))
        assert harm > 0 and help_ > 0, (seed, harm, help_)


def test_COMPAT_mode_gives_the_lab_s_nights(tmp_path):
    main(["run", "--compat", "--seeds", "3", "--out", str(tmp_path)])
    for seed in range(3):
        golden = (DATA / f"inn_rules_seed{seed}.jsonl").read_text().partition("\n")[2]
        assert (tmp_path / f"seed{seed}.events.jsonl").read_text() == golden, seed
        rows = _rows(tmp_path / f"seed{seed}.decisions.jsonl")
        assert rows and all(r["chooser"] == "rules" and r["decision"] for r in rows)


# ---------------------------------------------------------------- who picks
def _night(seed, turns=300, model=None, ask=None, **cfg):
    b = RoomsBackend(seed, turns)
    mind = Mind(MindConfig(seed=seed, setting=SETTING, **cfg), model=model, ask=ask)
    run(b, mind, turns)
    return b, mind


def test_REACTIVE_mode_never_asks_the_model_on_any_tick():
    sense = Sense()
    for seed in range(3):
        b, mind = _night(seed, model=sense)
        assert sorted(mind.remote) == list(range(1, b.world.t + 1))     # an entry for every tick
        assert set(mind.remote.values()) == {0}
        stopped = {r["t"] for r in mind.rows if r["reason"].startswith("interrupt:")}
        assert stopped and all(mind.remote[t] == 0 for t in stopped)    # interrupt ticks too
        # the model is not there, for all the night can tell
        assert mind.rows == _night(seed)[1].rows
    assert sense.calls == 0


def test_FEATURED_mode_asks_the_model_for_the_featured_only():
    sense = Sense()
    b, mind = _night(0, model=sense, mode="featured", featured=("Oren",))
    by = collections.defaultdict(collections.Counter)
    for r in mind.rows:
        by[r["who"]][r["chooser"]] += 1
    assert set(by["Oren"]) == {"rules", "combined"}                 # the gate let the rules answer some
    assert all(set(c) == {"rules"} for who, c in by.items() if who != "Oren")
    assert sum(mind.remote.values()) == sense.calls == by["Oren"]["combined"]
    assert max(mind.remote.values()) == 1
    assert {r.who for r in sense.seen} == {"Oren"}
    assert all(r.prior == r.prior_avg == r.habit == (0.0,) * len(r.options) for r in sense.seen)
    # all mode: the model for every NPC
    b, mind = _night(1, model=Sense(), mode="all", gate=None)
    assert {r["chooser"] for r in mind.rows} == {"combined"}


def test_ONE_OUTSTANDING_request_for_each_person():
    b = RoomsBackend(0)
    mind = Mind(MindConfig(mode="all", gate=None), model=Slow())
    mind.at(1)
    b.step()
    (who, reason), = b.needs_decision()
    with pytest.raises(RuntimeError, match="gave no decision"):
        mind.decide(b, who, reason)
    with pytest.raises(RuntimeError, match="second request"):
        mind.decide(b, who, reason)
    assert mind.remote[1] == 1


# ---------------------------------------------------------------- the player
def test_a_PLAYER_gets_the_full_menu_and_chains():
    asked = []

    def ask(r):
        asked.append(r)
        w = r.options.index("wait")
        return [w, w]                       # wait, and when that is done, wait again

    b, mind = _night(0, 150, ask=ask, player="Fenn")
    rows = [r for r in mind.rows if r["who"] == "Fenn"]
    assert all(r["cut"] == [] for r in rows) and max(len(r["options"]) for r in rows) > core.MENU_MAX
    assert all(len(r["options"]) <= core.MENU_MAX for r in mind.rows if r["who"] != "Fenn")
    assert [r["chooser"] for r in rows].count("player") == len(asked)
    chained = [i for i, r in enumerate(rows) if r["chooser"] == "chain"]
    assert chained
    for i in chained:                       # a chain runs on only when the job before it was done
        assert rows[i]["reason"] == "done" and rows[i - 1]["then"] == ["wait"] and rows[i]["then"] == []
    for i, r in enumerate(rows[:-1]):       # else the player is asked again
        if r["then"] and rows[i + 1]["reason"] != "done":
            assert rows[i + 1]["chooser"] == "player"


def test_the_one_the_player_ADDRESSED_gets_the_model_next():
    praised = []

    def ask(r):
        social = [i for i, t in enumerate(r.options) if t.startswith("praise ")]
        i = social[0] if social else r.options.index("wait")
        if social:
            praised.append(r.options[i][len("praise "):])
        return i

    sense = Sense()
    b, mind = _night(2, 200, model=sense, ask=ask, player="Fenn", mode="featured", gate=None)
    waiting, answered = set(), 0
    for r in mind.rows:
        if r["who"] == "Fenn":
            if r["text"].startswith("praise "):
                waiting.add(r["text"][len("praise "):])
            continue
        assert r["chooser"] == ("combined" if r["who"] in waiting else "rules"), r
        answered += r["who"] in waiting
        waiting.discard(r["who"])
    assert praised and answered and sense.calls == answered


def _walk(text):
    """The number of the first walk in the last menu that `text` shows."""
    menu = text[text.rindex("What does "):]
    return int(menu[:menu.index("  walk to ")].rsplit("\n", 1)[1])


class _Lines:
    """A player's input. "W" in a line is the number of the first walk on
    the menu that `out` shows now: the menu is in canonical order, so its
    numbers come from the catalog, not from the test."""

    def __init__(self, out, lines):
        self.out, self.lines = out, list(lines)

    def readline(self):
        if not self.lines:
            return ""
        line = self.lines.pop(0)
        return line.replace("W", str(_walk(self.out.getvalue()))) + "\n" if "W" in line else line + "\n"


def test_PLAY_reads_the_choices_and_stops(monkeypatch, capsys):
    out = io.StringIO()
    b, mind = play(0, 300, "Fenn", "rules", _Lines(out, ["x", "99", "W", "W W", "q"]), out)
    text = out.getvalue()
    first, second = [r for r in mind.rows if r["who"] == "Fenn"][:2]
    assert (first["chooser"], first["then"]) == ("player", [])
    assert first["text"].startswith("walk to ") and second["text"].startswith("walk to ")
    assert (second["chooser"], second["then"]) == ("player", [second["text"]])     # "W W": a chain
    assert text.count("--- turn") == 3 and text.count("What does Fenn do next?") == 3
    assert text.count("type the number of an option") == 2 and text.rstrip().endswith("stopped.")
    # the chain's second walk cannot start where the first one ended: the player is told
    assert text.count("the chain stops") == 1
    assert f"(Fenn cannot {second['then'][0]} now: the chain stops)" in text
    assert "\n  move\n" in text and "\n  wait\n" in text and "It is night at the village inn." in text
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    main(["run", "--play", "Tom", "--turns", "20"])
    assert "What does Tom do next?" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="not a person of the inn"):
        main(["run", "--play", "Nobody"])


def test_NEWS_says_when_a_CHAIN_STOPS():
    def walk(room):
        return Option("walk_to", {"place": Ref("place", room, room)}, f"walk to {room}", "move", 1)

    why = Why("j1", "player", "idle")
    job = Job("j1", "Fenn", walk("the kitchen"), 1, why, then=(walk("the cellar"),))
    b = SimpleNamespace(job_events=[JobEvent(1, "Fenn", "j1", "walk_to", "start"),
                                    JobEvent(3, "Tom", "j0", "walk_to", "done", "Tom walked to the yard"),
                                    JobEvent(4, "Fenn", "j1", "walk_to", "done", "Fenn walked to the kitchen")])
    mind = SimpleNamespace(rows=[{"job": "j1", "text": "walk to the kitchen"}], jobs={"Fenn": job})
    job.status = "done"
    assert news(b, mind, "Fenn", 0, asked=True) == [
        "  (Fenn walked to the kitchen)", "  (Fenn cannot walk to the cellar now: the chain stops)"]
    assert news(b, mind, "Fenn", 0) == ["  (Fenn walked to the kitchen)"]       # not asked: no word of it
    b.job_events[-1] = JobEvent(2, "Fenn", "j1", "walk_to", "stopped", reason="interrupt:struck")
    job.status = "stopped"
    assert news(b, mind, "Fenn", 0, asked=True) == [
        "  (Fenn was stopped and did not walk to the kitchen)", "  (the chain stops)"]
    job.then = ()
    assert news(b, mind, "Fenn", 0, asked=True) == ["  (Fenn was stopped and did not walk to the kitchen)"]


def test_a_SESSION_plays_one_choice_for_each_call(tmp_path, capsys):
    d = ["--dir", str(tmp_path), "--tag", "t1"]
    main(["session", "start", "--as", "Fenn", "--seed", "1", *d])
    first = capsys.readouterr().out
    assert "you are Fenn" in first and "What does Fenn do next?" in first
    main(["session", "show", *d])
    assert capsys.readouterr().out == first.partition("\n")[2]          # the same night, run again
    w1 = _walk(first)
    main(["session", "pick", str(w1), *d])
    w2 = _walk(capsys.readouterr().out)
    main(["session", "pick", str(w2), str(w2), *d])
    out = capsys.readouterr().out
    assert out.count(">>> ") == 1 and "; then " in out
    assert out.count("the chain stops") == 1           # the chain's second walk cannot start
    state = json.loads((tmp_path / "session-t1.json").read_text())
    assert state["picks"] == [[w1], [w2, w2]] and len(state["asked"]) == 2
    with pytest.raises(SystemExit, match="number of an option"):
        main(["session", "pick", "999", *d])
    # the session is the night that the same picks give in one run
    picks = iter([[w1], [w2, w2]])

    def ask(r):
        try:
            return next(picks)
        except StopIteration:
            raise Stop(r) from None

    b = RoomsBackend(1)
    mind = Mind(MindConfig(player="Fenn", seed=1, setting=SETTING), ask=ask)
    with pytest.raises(Stop):
        run(b, mind, 300)
    # a session that no longer asks the same questions says so
    bad = {**state, "asked": ["0" * 16] + state["asked"][1:]}
    (tmp_path / "session-t2.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="no longer replays"):
        main(["session", "show", "--dir", str(tmp_path), "--tag", "t2"])
    main(["session", "end", *d])
    assert "ended after 2 choices" in capsys.readouterr().out
    assert (tmp_path / "session-t1.events.jsonl").read_text() == jsonl(b.world.log)
    assert _rows(tmp_path / "session-t1.decisions.jsonl") == json.loads(json.dumps(mind.rows))
    assert not (tmp_path / "session-t1.json").exists()
    with pytest.raises(SystemExit, match="no session"):
        main(["session", "show", *d])
