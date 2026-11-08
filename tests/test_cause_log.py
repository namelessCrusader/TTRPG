"""New in V5 (docs/sim-mind-design.md, 3.4 and 4.4): the cause log and its
readers.

Every decision is now an event, `decides`, caused by the event its percept
came from, and every act of the decision is caused by the decision. So
`w.why` walks from an act back through the decision to what the body
perceived, and on to whatever made THAT: a man runs because he heard a
shout, the shout was a decision, that decision was a fire seen. Each trace
row carries the chain as plain JSON (`row["mind"]`), which is what a reader
outside the sim -- the lab's multi-hop questions -- gets. Two readers say
it in words from a body's side: `render` a percept row, `beliefs.view` all
a body holds. What must hold: the rows still dump as JSON, the chain is
whole, a reader never names a person the senses did not name, and none of
it changes what anyone does.
"""
import contextlib
import io
import json

import numpy as np
import pytest

from src.voxel import harvest, scenes
from src.voxel.beliefs import view
from src.voxel.events import Rec, plain
from src.voxel.policy import TablePolicy
from src.voxel.scenes import _person
from src.voxel.sim import AIR, OIL, STONE, WOOD, World


class _Kept(World):
    """A world that keeps every percept row it writes, as (pid, row): a
    Mind keeps only its last 32."""

    def _note(self, pid, *a, **k):
        rec = super()._note(pid, *a, **k)
        if rec is not None:
            self.__dict__.setdefault("kept", []).append((pid, rec))
        return rec


class _NoLog(World):
    """A world whose decisions are not logged: no `decides` event, so its
    acts have no cause. What it does must be what a logging world does."""

    def _decides(self, p, percept, src, program, site):
        return None


def _rows(w):
    """The trace rows without the new key, as text."""
    return [json.dumps({k: v for k, v in r.items() if k != "mind"}
                       | {"situation": {k: v for k, v in r["situation"].items()
                                        if k != "reflexes"}},
                       sort_keys=True, default=repr) for r in w.traces]


def _same_run(a, b):
    """Two worlds stepped alike did the same things: the same decisions,
    words, legacy lines, events (by what they were, not their ids), bodies
    and matter."""
    assert _rows(a) == _rows(b)
    assert a.speech == b.speech
    assert [p["events"] for p in a.persons] == [p["events"] for p in b.persons]
    assert [(p["anchor"], p["alive"], p["safe"], p.get("goal"))
            for p in a.persons] \
        == [(p["anchor"], p["alive"], p["safe"], p.get("goal"))
            for p in b.persons]

    def evs(w):
        return [(e.tick, e.verb, e.actor, e.result, e.place)
                for e in w.events.rows if e.verb != "decides"]
    assert evs(a) == evs(b)
    for f in ("mat", "smass", "smoke", "E", "fvol"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f


# ── an alarm passed on: the chain the lab will ask about ────────────────────

def _alarm_room(cls=World):
    """Aldan faces a burning crib; Berel stands behind a wall he cannot see
    the crib past, in earshot. Aldan sees the fire, makes for the door and
    shouts; Berel hears the shout and makes for the door too. Both by the
    reflex table (TablePolicy)."""
    w = cls(48, 40, 40, voxel_cm=5)
    w.open_sky = False
    # THE CHAIN, NOT THE SMOKE. `_run_alarm` lights the crib by putting 2500 J a
    # tick into one cell (100 kW, fifty torches), which flash-boils 23 ml of oil
    # into 21 g of smoke in half a second. Under the soot law (wave 13c, C4b)
    # smoke that thick hides even a flame (depth 8; 3 counting the flame's own
    # light, test_smoke_sight). This test is about how the links of the chain are
    # made, so it keeps sight clear of smoke; smoke and sight have their own tests.
    w.smoke_opacity = False
    w.fill(0, 48, 0, 40, 0, 40, STONE)
    w.mat[1:47, 1:39, 1:39] = AIR
    w.smass[1:47, 1:39, 1:39] = 0.0
    w.fill(20, 47, 20, 22, 1, 39, STONE)        # the wall Berel is behind
    w.fill(30, 36, 9, 15, 1, 3, WOOD, frac=0.8)  # the crib
    w.pour(31, 10, 3, OIL, 100.0)
    w.exits = [(2, 12)]
    a = _person(w, 14, 12)
    a["name"], a["facing"] = "Aldan", (1.0, 0.0)
    b = _person(w, 30, 30)
    b["name"], b["facing"] = "Berel", (1.0, 0.0)
    for p in (a, b):                     # quick to answer and no pre-movement:
        p["surprise_s"] = p["premove_s"] = 0.25   # one decision each (R3)
    return w, a, b


def _run_alarm(cls=_Kept, ticks=60):
    w, a, b = _alarm_room(cls)
    assert isinstance(w.policy, TablePolicy)
    for t in range(ticks):
        if t < 20:
            w.E[31, 10, 3] += 2500.0
        w.step()
    return w, a, b


@pytest.fixture(scope="module")
def alarm():
    w, a, b = _run_alarm()
    yield w, a, b
    del w, a, b


def test_WHY_the_hearer_RUNS_walks_back_to_the_FIRE(alarm):
    """Berel's `goes` <- his `decides` (hears_alarm) <- Aldan's `says` <-
    Aldan's `decides` (sees_fire) <- the fire's `ignites`. Each link says
    how it was made: an act is CALLED by its decision; a decision is CARRIED
    to by its percept's source."""
    w, a, b = alarm
    goes = [e for e in w.events.rows if e.verb == "goes" and e.actor == b["pid"]
            and e.why == "hears_alarm"]
    assert goes, "Berel heard the alarm and went"
    chain = w.why(goes[0].id)
    assert [e.verb for e in chain] == \
        ["ignites", "decides", "says", "decides", "goes"]
    assert [e.why for e in chain if e.verb == "decides"] == \
        ["sees_fire", "hears_alarm"]
    assert [e.actor for e in chain[1:]] == \
        [a["pid"], a["pid"], b["pid"], b["pid"]]
    assert [e.link for e in chain[1:]] == ["carried", "call", "carried", "call"]
    # and the trace row of Berel's decision carries the same chain, plain
    row = next(r for r in w.traces if r["who"] == "Berel"
               and r["percept"] == "hears_alarm")
    m = row["mind"]
    assert m["decides"] == chain[3].id and m["cause"] == chain[2].id
    assert [e["verb"] for e in m["why"]] == \
        ["ignites", "decides", "says", "decides"]
    assert json.loads(json.dumps(m)) == m
    assert all("text" not in e for e in m["why"]), "Ev.text never leaves the sim"


def test_every_ACT_of_a_decision_is_CAUSED_by_it(alarm):
    w, _a, _b = alarm
    decs = {e.id for e in w.events.rows if e.verb == "decides"}
    assert decs
    for e in w.events.rows:
        if e.verb in ("goes", "says", "stays", "looks"):
            assert e.cause in decs and e.link == "call", e


def test_a_DECISION_is_NOT_PERCEIVED(alarm):
    """A decision happens in a head: nobody sees, hears or feels it, and
    the body itself remembers its acts, not the deciding."""
    w, _a, _b = alarm
    decs = {e.id for e in w.events.rows if e.verb == "decides"}
    for m in w.minds:
        assert not any(r.ev in decs for r in m.perceived)


def test_TABLEPOLICY_does_THE_SAME_with_the_log_or_without(alarm):
    w, _a, _b = alarm
    n, _a2, _b2 = _run_alarm(_NoLog)
    assert not any(e.verb == "decides" for e in n.events.rows)
    _same_run(w, n)


def test_ROAM_does_THE_SAME_with_the_log_or_without():
    """The harvest's world and chooser (Roam), with and without the log.
    Seed 7: two people, both awake, deciding."""
    out = []
    for cls in (World, _NoLog):
        w, people, _lip = harvest.world(7)
        w.__class__ = cls
        w.seed = 7
        w.policy = harvest.Roam(7)
        for _ in range(60):
            w.step()
        out.append(w)
    assert any(e.verb == "decides" for e in out[0].events.rows)
    _same_run(*out)


# ── the readers ──────────────────────────────────────────────────────────────

def test_a_VIEW_is_PLAIN_and_SHAPED_for_the_mind(alarm):
    """`beliefs.view`: Percept- and Belief-shaped dicts (src/mind/api.py),
    that dump as JSON as they are; at most 5 recent percepts, one per
    source; at most 2 beliefs for a menu, none of them about the body
    itself or a voice."""
    w, a, b = alarm
    for p in (a, b):
        v = view(w, p)
        assert json.loads(json.dumps(v)) == v
        for x in v["percepts"]:
            assert set(x) == {"t", "channel", "text", "source", "verb", "actor",
                              "target", "phase", "kind", "salience"}
            assert x["channel"] in ("saw", "heard", "felt", "told", "did",
                                    "noticed")
        for x in v["beliefs"]:
            assert set(x) == {"kind", "about", "value", "text", "source", "how",
                              "t"}
        rec = v.recent()
        assert len(rec) <= 5 and len({x["source"] for x in rec}) == len(rec)
        menu = [{"key": "go(the door west)"}] + [
            {"key": f"take hold of {q['name']}", "who": q["name"]}
            for q in (a, b) if q is not p]
        got = v.beliefs_for(menu)
        assert len(got) <= 2
        assert all(x["kind"] != "voice" and x["about"] != {
            "kind": "person", "id": str(p["pid"]), "name": p["name"]}
            for x in got)
    # Berel heard the shout as words, from no one he could name: he has
    # never seen Aldan speak, so he does not know the voice
    heard = [r for pid, r in w.kept if pid == b["pid"] and r.how == "heard"
             and r.verb == "says"]
    assert heard and heard[0].actor_seen is None
    assert w.render(heard[0], b).startswith('someone shouts "Fire!')


def test_RENDER_never_NAMES_a_person_the_SENSES_did_not():
    """A row says who only when its senses could tell (`actor_seen`,
    `target_seen`): a blow heard through a wall is "a blow", a fall seen
    without its doer is "someone"; the viewer is "you"."""
    w = World(40, 40, 30, voxel_cm=5)
    w.fill(0, 40, 0, 40, 0, 1, STONE)
    w.exits = []
    ps = [_person(w, x, 20) for x in (8, 20, 32)]
    for p, n in zip(ps, ("Aldan", "Berel", "Cadoc")):
        p["name"] = n
    w.step()
    names = [p["name"] for p in ps]
    rows = []
    for how in ("saw", "heard", "felt", "did", "looked"):
        for verb in ("hits", "wounds", "swings", "says", "takes", "goes",
                     "breaks", "lands", "falls", "ignites", None):
            for actor in (None, 0, 2):
                for target in (None, ("person", 2), ("person", 1),
                               ("mat", int(STONE), (5, 5, 0)), ("fire", 0)):
                    data = (("line", "fire"),) if verb == "says" else ()
                    rows.append(Rec(tick=0, how=how, ev=None, verb=verb,
                                    actor_seen=actor, target_seen=target,
                                    place=(30, 30, 10), db=50.0, data=data))
    for r in rows:
        for v in range(3):
            text = w.render(r, v)
            told = {r.actor_seen} | ({r.target_seen[1]} if r.target_seen
                                     and r.target_seen[0] == "person"
                                     else set())
            if r.how == "felt":
                told = {v}
            for i, n in enumerate(names):
                if n in text:
                    assert i in told and i != v, (text, r, v)
    assert w.render(Rec(0, "heard", None, "hits", None, None, (30, 30, 10),
                        50.0), 0) == "a blow to the north-east"
    assert w.render(Rec(0, "saw", None, "hits", None, ("person", 2),
                        (30, 30, 10), None), 1).startswith("someone hits Cadoc")
    assert w.render(Rec(0, "did", None, "takes", 0, None, None, None), 0) \
        == "you take up something"


# ── fire_alarm, cut short ────────────────────────────────────────────────────

def test_FIRE_ALARM_rows_DUMP_as_JSON(monkeypatch, tmp_path):
    """`fire_alarm` cut to 120 ticks, frames not written: every trace row
    has its `mind`, plain JSON that comes back as it went in, and the
    scene's own traces.json (which dumps every row) still writes."""
    got = []

    class Kept(World):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            got.append(self)

    monkeypatch.setattr(scenes, "World", Kept)
    monkeypatch.setattr(scenes, "_save", lambda *a, **k: None)
    with contextlib.redirect_stdout(io.StringIO()):
        scenes.fire_alarm(str(tmp_path), ticks=120, every=10 ** 9)
    w = got[0]
    assert w.traces
    for r in w.traces:
        json.dumps({k: v for k, v in r.items() if k != "situation"})
        m = r["mind"]
        assert json.loads(json.dumps(m)) == m
        assert m["why"][-1]["id"] == m["decides"]
        assert m["why"][-1]["verb"] == "decides"
        assert len(m["recent"]) <= 5
    with open(tmp_path / "traces.json") as f:
        assert all("mind" in r for r in json.load(f))
    assert plain(w.events.rows[-1].to_json()) == w.events.rows[-1].to_json()
    got.clear()
    del w


def test_a_POLICY_that_OBSERVES_is_shown_the_VIEW_before_it_PICKS():
    """`policy.observe(view)`, asked only of a policy that has it, once a
    decision, before any limb is put the question."""
    seen = []

    class Watcher(TablePolicy):
        name = "watcher"

        def observe(self, v):
            seen.append(("observe", v["who"], json.dumps(v)))

        def pick(self, sit, menu):
            seen.append(("pick", sit["who"]))
            return super().pick(sit, menu)

    w = World(30, 30, 30, voxel_cm=5)
    w.fill(0, 30, 0, 30, 0, 1, STONE)
    w.exits = []
    p = _person(w, 15, 15)
    p["name"] = "Aldan"
    w.policy = Watcher()
    w.step()
    assert w.traces and seen and seen[0][:2] == ("observe", "Aldan")
    assert sum(1 for s in seen if s[0] == "observe") == len(w.traces)
    assert all(s[0] == "pick" for s in seen[1:]), seen
