"""New in M8 (docs/sim-mind-design.md 3.5 and 4.4): the voxel mind link.
M10 adds the approach: the stand-off band per stance, the walk to it, the
start percept of a harmful job, a blow that is news on the tick it is felt,
and a staged reaction answered once. M12 takes away the gates that hid the
door, the warning and the walk over to strike from a body with no reason
(they score NO_REASON now), and adds the two-level menu: a scope, then an
act.

`src/voxel/mindlink.py` puts the mind interface (src/mind) in charge of the
bodies of a voxel world: the mind chooses a TASK from a menu of catalog verbs
(take the iron, strike the defender, wait), and the link runs it as a job,
one limb at a time, through the will layer's own per-limb menus.

Most of these run the `_duel` geometry of tests/test_wounds.py, the scene of
the design's Verify list: an attacker at x=17 with an axe-edged iron at his
feet (22-24, edge 2e-4), a defender at x=31, 150 ticks, attention "onset".
The attacker has a character sheet that wants the defender hurt; the
defender has none. Nobody is told to hit anybody: the mind picks tasks, the
rule scores come from the sheet, and whether a swing opens a wound is the
world's business.
"""
import collections
import functools
import json
import math
import pickle
import re
from pathlib import Path

import pytest

from src.mind import core, scope
from src.mind.choosers import Rules
from src.mind.api import DecisionRequest
from src.mind.core import catalog
from src.mind.loop import Mind, MindConfig
from src.voxel import harvest, mindlink
from src.voxel.mindlink import STAND_OFF_M, MindPolicy
from src.voxel.policy import REFLEXES, WILL, TablePolicy, sheet
from src.voxel.scenes import _person
from src.voxel.events import Rec
from src.voxel.sim import AIR, IRON, STONE, World

SRC = Path(__file__).resolve().parent.parent / "src" / "voxel"
AXE = 2e-4                                    # the edge of the duel's iron, m²


class _Tripwire:
    """A model chooser that must never be asked: reactive mode makes no
    model call at all (design 2.4, the memory note latency-budget)."""
    name = "tripwire"
    remote = True

    def __init__(self):
        self.asked = []

    def submit(self, req):
        self.asked.append(req.id)
        raise AssertionError(f"reactive mode sent request {req.id} to the model")

    def poll(self, req_id, block=True):
        raise AssertionError("reactive mode polled the model")


def _duel(seed=0, argmax=True, model=None, record=False, iron=True, menu="flat"):
    """The `_duel` world of tests/test_wounds.py, linked to a mind (with
    no iron at the attacker's feet, if not `iron`), whose menu is `menu`
    (flat, or two: a scope, then an act)."""
    w = World(46, 20, 40, voxel_cm=5)
    # Lane C9 (stairs, step range 30 cm by default): a body now steps up onto the
    # 5 cm iron slab, so the punch spot exists and the attacker may punch bare-handed
    # (a fist opens no skin) before ever taking the axe: 15 of 20 seeds wound, base21 19.
    # The duel is about the iron being an obstacle, so it keeps the old flat legs.
    w.step_up_m = w.step_down_m = 0.0
    w.fill(0, 46, 0, 20, 0, 1, STONE)
    w.exits = [(44, 10)]
    a = _person(w, 17, 10)
    a["name"] = "attacker"
    b = _person(w, 31, 10)
    b["name"] = "defender"
    if iron:
        w.fill(22, 24, 10, 11, 1, 2, IRON, edge=AXE)
    a["sheet"] = {"harm": {"defender": 0.8}}
    pol = mindlink.link(w, mind=Mind(MindConfig(seed=seed, menu=menu), model=model), record=record)
    if argmax:
        pol.mind.npc = Rules(seed, argmax=True)
    return w, a, b, pol


def _be(pol):
    """The VoxelBackend of a policy, inside its Recorder if it has one."""
    return pol.backend.backend if pol.record else pol.backend


def _wounded(b):
    return any(wd["open"] for wd in b.get("wounds") or ())


def _step(w, b, ticks, until_wound=False):
    """Step `ticks` ticks. Returns the persons on each tick, and the tick of
    the first blow the defender took (None: none)."""
    persons, hit, seen = [], None, w.events.next_id - 1
    for _ in range(ticks):
        w.step()
        persons.append(len(w.persons))
        if hit is None:
            hit = next((ev.tick for ev in w.events.since(seen)
                        if ev.verb == "hits" and ev.target == ("person", b["pid"])), None)
        seen = w.events.next_id - 1
        if until_wound and (_wounded(b) or b["safe"]):
            break                             # wounded, or out of the door for good
    return persons, hit


def _summary(w, a, b, pol, persons, hit):
    """What a run did, as plain data: the world itself is let go."""
    return {"rows": [dict(r) for r in pol.mind.rows],
            "jobs": [(e.t, e.who, e.verb, e.phase, e.reason) for e in pol.backend.job_events],
            "traces": [(r["tick"], r["who"], r["by"], sorted(r["menus"]), r["pick"])
                       for r in w.traces],
            "persons": persons, "open": _wounded(b), "hurt": b["hurt"],
            "first_hit": hit, "remote": dict(pol.mind.remote),
            "cases": pol.backend.cases(pol.mind.rows) if pol.record else None}


@functools.lru_cache(maxsize=1)
def _argmax_run():
    """The verify scene with the argmax Rules chooser, run once for every
    test that reads it, with a model that must never be asked."""
    trip = _Tripwire()
    w, a, b, pol = _duel(model=trip, record=True)
    out = _summary(w, a, b, pol, *_step(w, b, 150))
    out["asked"] = list(trip.asked)
    out["model"] = pol.mind.model is not None
    return out


def _rows(who=None, run=None):
    rows = (run or _argmax_run())["rows"]
    return [r for r in rows if who is None or r["who"] == who]


# ── the Verify list of M8 ───────────────────────────────────────────────────

def test_the_ARGMAX_rules_chooser_OPENS_A_WOUND():
    s = _argmax_run()
    assert s["open"], (s["hurt"], [(r["t"], r["who"], r["text"]) for r in s["rows"]])
    verbs = [r["verb"] for r in _rows("0")]
    assert verbs[:2] == ["take", "strike"], verbs


def test_18_of_20_SAMPLED_request_seeds_open_a_wound():
    """Twenty runs of the duel, one a world seed. The Rules chooser draws
    every pick, the attacker's and the defender's, with a generator seeded
    from (world seed, who, request id), so each seed is a different run of
    draws. A run stops at the first open wound, or when the defender is out
    of the door for good. Measured: 20 of seeds 0-19, and 59 of seeds 0-59
    (M8, before the stand-off bands of M10: 19 and 56; its misses began
    with a punch at t0, and the spot to punch from is where the iron lies;
    the one miss left is a defender who took the iron himself).
    (Before a way out and a warning needed a reason to fear, 11 of 20: a
    defender who wanted nothing walked out of the door, or the other's
    shout of fire at nothing sent him running.) M12 took those gates and
    M10's away; a reasonless act scores NO_REASON instead: 19 of 20 and 58
    of 60, flat and two-level alike (one miss each is a calm defender who
    walked out at his first decision, by a draw of about 1 in 280; the other
    is the defender who took the iron)."""
    opened = []
    for seed in range(20):
        w, a, b, pol = _duel(seed=seed, argmax=False)
        _step(w, b, 150, until_wound=True)
        opened.append(_wounded(b))
        del w, a, b, pol
    assert sum(opened) >= 18, opened


def test_the_DEFENDERS_NEXT_REQUEST_after_the_blow_says_INTERRUPT_STRUCK():
    s = _argmax_run()
    assert s["first_hit"] is not None, "the swing found him"
    after = [r for r in _rows("1") if r["t"] > s["first_hit"]]
    assert after and after[0]["reason"] == "interrupt:struck", \
        [(r["t"], r["reason"]) for r in _rows("1")]
    stopped = [j for j in s["jobs"] if j[1] == "1" and j[3] == "stopped"]
    assert stopped and stopped[0][4] == "interrupt:struck"


def test_at_most_ONE_LIMB_MENU_a_decide_and_TWO_REQUESTS_a_tick():
    s = _argmax_run()
    assert s["traces"], "the bodies decided"
    assert max(len(menus) for _, _, _, menus, _ in s["traces"]) <= 1, s["traces"]
    per_tick = collections.Counter(r["t"] for r in s["rows"])
    assert per_tick and max(per_tick.values()) <= 2, per_tick


def test_TWO_PERSONS_on_every_tick():
    """Bug 87 can hit a two-person scene with falls; this one has none, and
    nobody is ever added or lost."""
    assert _argmax_run()["persons"] == [2] * 150


def test_a_DETERMINISTIC_REPLAY():
    """The same seed asks the same questions and gets the same answers:
    a second run, and a run restored to tick 40 and stepped on again."""
    s = _argmax_run()
    w, a, b, pol = _duel(record=True)
    persons, hit = _step(w, b, 40)
    snap = w.snapshot()
    more, hit2 = _step(w, b, 110)
    again = _summary(w, a, b, pol, persons + more, hit if hit is not None else hit2)
    assert again["rows"] == s["rows"] and again["first_hit"] == s["first_hit"]
    assert again["jobs"] == s["jobs"] and again["traces"] == s["traces"]
    assert again["cases"] == s["cases"]
    assert again["hurt"] == s["hurt"]
    w.restore(snap)
    assert all(r["t"] < 40 for r in pol.mind.rows)
    _step(w, b, 110)
    replay = _summary(w, a, b, pol, [], None)
    key = lambda r: (r["t"], r["who"], r["reason"], r["text"], r["options"], r["pick"])
    assert [key(r) for r in replay["rows"]] == [key(r) for r in s["rows"]]
    assert [r["id"] for r in replay["rows"]] == [r["id"] for r in s["rows"]]
    assert replay["jobs"] == s["jobs"]
    assert replay["open"] == s["open"] and replay["hurt"] == pytest.approx(s["hurt"])


def test_ONLY_CATALOG_VOXEL_VERBS_run():
    voxel = set(catalog().runnable("voxel"))
    s = _argmax_run()
    assert {r["verb"] for r in s["rows"]} <= voxel
    assert {j[2] for j in s["jobs"]} <= voxel
    offered = {o.split(" ")[0] for r in s["rows"] for o in r["options"] + r["cut"]}
    texts = {catalog().verbs[v].text.split(" ")[0] for v in voxel}
    assert offered <= texts, offered - texts


def test_REACTIVE_mode_makes_ZERO_REMOTE_SUBMITS():
    s = _argmax_run()
    assert s["model"], "a model chooser was there to be asked"
    assert s["asked"] == []
    assert s["remote"] and set(s["remote"].values()) == {0}, s["remote"]
    assert {r["chooser"] for r in s["rows"]} == {"rules"}


# ── what the link is made of (the What of M8) ───────────────────────────────

def test_TRACE_ROWS_say_BY_MIND_and_the_HARVEST_skips_them():
    s = _argmax_run()
    assert {by for _, _, by, _, _ in s["traces"]} == {"mind"}
    assert harvest.LINKED == MindPolicy.name

    class _W:
        def trace_outcomes(self):
            row = {"options": ["a", "b"], "spelling": "x"}
            return [{"by": "mind", "spelled": [dict(row, limb="hands")]},
                    {"by": "roam", "spelled": [dict(row, limb="legs"),
                                               {"options": ["only"], "limb": "eyes"}]}]
    got = harvest.rows_of(_W(), 7)
    assert [r["limb"] for r in got] == ["legs"] and got[0]["seed"] == 7


def test_the_MENUS_AND_THE_RECORD_of_a_decision():
    """Every option a chooser saw is a catalog verb with its roles filled;
    the scene text holds no number; the recorded case rows are the lab's."""
    s = _argmax_run()
    first = _rows("0")[0]
    assert "take the iron" in first["options"] and "wait" in first["options"]
    # (and no punch: the spot to punch him from, in line on this side, is
    # where the iron lies -- M10's stand-off, `_seek_spot`)
    assert "strike defender" not in first["options"]
    assert not re.search(r"\d", first["situation"])
    assert "Attacker feels angry at defender" in first["situation"]
    assert len(s["cases"]) == len(s["rows"])
    assert all(c["backend"] == "voxel" and c["options"] == r["options"]
               for c, r in zip(s["cases"], s["rows"]))
    json.dumps(s["rows"]), json.dumps(s["cases"])


def test_with_the_STEP_RANGE_ON_a_5_cm_slab_is_no_stand_off():
    """Lane C9 (legs climb 30 cm): the iron slab is a step, not a wall, so the
    spot to punch from (on the slab) exists and 'strike defender' is offered,
    scored as an unarmed blow."""
    w, a, b, pol = _duel()
    del w.step_up_m, w.step_down_m            # the class default (30 cm), as C9 set it
    be = _be(pol)
    opts = be.bind("0")
    got = dict(zip((o.text for o in opts), be.rule_scores("0", opts)[0]))
    assert got["strike defender"] == pytest.approx(0.5 * 0.8)


def test_RULE_SCORES_come_from_the_SHEET():
    w, a, b, pol = _duel()
    be = _be(pol)                             # before anyone has chosen
    opts = be.bind("0")
    prior, avg, habit = be.rule_scores("0", opts)
    got = {o.text: s for o, s in zip(opts, prior)}
    assert got["take the iron"] == pytest.approx(0.8 * 0.8)    # a weapon: 0.8 x harm
    assert got["wait"] == 0.2
    assert avg == prior and set(habit) == {0.0}
    w, a, b, pol = _duel(iron=False)          # nothing between them: a punch
    be = _be(pol)
    opts = be.bind("0")
    got = dict(zip((o.text for o in opts), be.rule_scores("0", opts)[0]))
    assert got["strike defender"] == pytest.approx(0.5 * 0.8)  # unarmed
    theirs = dict(zip((o.text for o in be.bind("1")), be.rule_scores("1", be.bind("1"))[0]))
    # the defender's sheet wants nobody hurt, and he fears nothing: a walk
    # over to strike is offered (M12: no gate hides it), and it is an act
    # with no reason (`NO_REASON`)
    assert theirs["strike attacker"] == mindlink.NO_REASON, theirs
    w.minds[b["pid"]].perceived.append(Rec(tick=w.tick, how="felt", ev=None, verb="hits",
                                           actor_seen=None, target_seen=None, place=None,
                                           db=None))
    opts = be.bind("1")                       # afraid now: a strike serves safety
    theirs = dict(zip((o.text for o in opts), be.rule_scores("1", opts)[0]))
    assert theirs["strike attacker"] == 0.0, "the defender's sheet wants nobody hurt"


def test_a_WAY_OUT_and_a_WARNING_are_OFFERED_with_no_fear_and_score_NO_REASON():
    """M12 took the gate away: with nothing to fear, a walk or a flight out
    of the door and a shout of fire are on the menu (the door is known, and
    the words), and they score NO_REASON. A blow felt is a percept of
    safety, and then they score as the table does: walk 0.1, flee safety x
    (1 - route danger), shout safety while someone can hear."""
    w, a, b, pol = _duel()
    be = _be(pol)
    assert any(x.kind == "way_out" for x in be.beliefs("0")), "the door is known"
    opts = be.bind("0")
    calm = dict(zip((o.text for o in opts), be.rule_scores("0", opts)[0]))
    for text in ("walk to the door to the east", "flee to the door to the east", "shout a warning"):
        assert calm[text] == mindlink.NO_REASON, calm
    w.minds[a["pid"]].perceived.append(Rec(tick=w.tick, how="felt", ev=None, verb="hits",
                                           actor_seen=None, target_seen=None, place=None,
                                           db=None))
    assert be.fears(a) == {"struck"}
    opts = be.bind("0")
    got = dict(zip((o.text for o in opts), be.rule_scores("0", opts)[0]))
    assert got["walk to the door to the east"] == 0.1
    assert got["flee to the door to the east"] == pytest.approx(1.0 - be._payload(
        "0", next(o for o in opts if o.verb == "flee_to"))["danger"])
    assert got["shout a warning"] == 1.0, "the defender is awake and near"
    opts = be.bind("1")
    theirs = {o.verb: s for o, s in zip(opts, be.rule_scores("1", opts)[0])}
    assert {theirs[v] for v in ("walk_to", "flee_to", "shout_fire")} == {mindlink.NO_REASON}, \
        "the defender felt nothing"


def test_BIND_keys_things_by_EID_or_CELL_and_leaves_out_HELD_things():
    w = World(40, 40, 30, voxel_cm=5)
    w.fill(0, 40, 0, 40, 0, 1, STONE)
    w.exits = []
    p = _person(w, 20, 20)
    q = _person(w, 20, 30)
    e = w.register("bar")
    w._stamping = e
    w.fill(22, 23, 20, 21, 1, 2, IRON)
    w._stamping = 0
    w.fill(18, 19, 20, 21, 1, 2, IRON)             # a lump with no name
    w.fill(20, 21, 22, 23, 1, 2, IRON)             # what q will hold
    be = _be(mindlink.link(w))
    takes = {o.role("target").id for o in be.bind("0") if o.verb == "take"}
    assert takes == {f"e{e}", "c18,20,1", "c20,22,1"}, takes
    q["held"] = {"cell": (20, 22, 1), "mat": int(IRON), "label": "iron", "arm": None}
    takes = {o.role("target").id for o in be.bind("0") if o.verb == "take"}
    assert takes == {f"e{e}", "c18,20,1"}, "the one in q's hand is not offered"
    p["held"] = {"cell": (18, 20, 1), "mat": int(IRON), "label": "iron", "arm": None}
    opts = be.bind("0")
    assert not any(o.verb == "take" for o in opts), "no free hand"
    assert [o.text for o in opts if o.verb == "drop"] == ["drop the iron"]


def test_PERCEPTS_and_BELIEFS_come_from_the_VIEW():
    w, a, b, pol = _duel()
    _step(w, b, 30)
    be = _be(pol)
    felt = [x for x in be.percepts("1") if x.channel == "felt"]
    assert felt and felt[0].kind == "harm_to_me" and felt[0].actor is None, \
        "felt on the body: no doer"
    beliefs = be.beliefs("1")
    assert any(x.kind == "way_out" and x.about.id == "44,10" for x in beliefs)
    b["_baulked"] = {(40, 12): w.tick}
    nope = [x for x in be.beliefs("1") if x.kind == "unreachable"]
    assert len(nope) == 1 and nope[0].about.id == "40,12"
    assert "struck" in be.fears(b) and "safety" in be.self_state("1").drives


def test_PICK_is_PURE_and_matches_BY_PAYLOAD_never_by_label():
    w, a, b, pol = _duel()
    w.step()
    goal = (40, 10)
    pol._ctx = (w, w.tick, 0, "attacker", frozenset({"legs"}),
                {"legs": {"verb": "go", "goal": goal}})
    menu = [{"key": "stay", "tag": "stay", "verb": "stay"},
            {"key": "go(the door east)", "tag": "go:exit", "verb": "go", "goal": (44, 10)},
            {"key": "leap(over there)", "tag": "leap", "verb": "jump", "goal": goal},
            {"key": "go(somewhere else entirely)", "tag": "go:roam", "verb": "go",
             "goal": goal}]
    sit = {"tick": w.tick, "who": "attacker", "limb": "legs", "percept": None}
    before = json.dumps(menu, sort_keys=True), dict(sit)
    assert pol.pick(sit, menu) == 3 == pol.pick(sit, menu)
    assert (json.dumps(menu, sort_keys=True), sit) == before
    assert pol.pick(dict(sit, limb="hands"), menu) == 0, "a limb the step does not need"
    assert pol.pick(dict(sit, who="defender"), menu) == 0
    assert pol.limbs(w, a, None) == {"legs"} and pol.limbs(w, b, None) == set()


def test_FALLING_decides_go_to_the_REFLEX_TABLE():
    pol = MindPolicy()
    menu = [{"key": "stay as you are", "tag": "stay", "verb": "stay"},
            {"key": "roll on landing", "tag": "roll", "verb": "roll"}]
    roll = {"falling": {"legs": "roll"}}
    for reflexes in (REFLEXES, roll):
        sit = {"tick": 3, "who": "x", "limb": "legs", "percept": "falling",
               "falling": True, "reflexes": reflexes}
        assert pol.pick(sit, menu) == TablePolicy().pick(sit, menu)
    assert pol.pick(dict(sit), menu) == 1


def test_a_STEP_FAILS_after_THREE_DECIDE_PERIODS_with_no_event():
    """The attacker sets out to take the iron, and the iron is gone before
    a hand can close on it: no `takes` ever comes, the take row is never
    there to match, and the job fails three decide periods after the hands
    were first asked. The mind is asked again, and says why."""
    w, a, b, pol = _duel()
    be = _be(pol)
    real = be.compile

    def compile_then_vanish(p, percept, *stage):
        ask = real(p, percept, *stage)
        if p is a and "hands" in ask and p["job"].option.verb == "take":
            w.fill(22, 24, 10, 11, 1, 2, AIR)        # gone before the hand gets there
        return ask
    be.compile = compile_then_vanish
    _step(w, b, 50)
    rows = _rows("0", {"rows": pol.mind.rows})
    assert rows[0]["verb"] == "take"
    failed = [r for r in rows if r["reason"].startswith("failed:")]
    assert failed and failed[0]["reason"] == "failed:no took after asking", \
        [(r["t"], r["reason"]) for r in rows]
    asked = rows[0]["t"]
    assert failed[0]["t"] - asked == 3 * WILL["decide_every"]


def test_a_LINKED_world_SNAPSHOTS_its_JOBS_and_they_PICKLE():
    w, a, b, pol = _duel()
    _step(w, b, 12)
    snap = w.snapshot()
    job = snap["persons"][a["pid"]]["job"]
    assert job == a["job"] and job is not a["job"]
    assert pickle.loads(pickle.dumps(job)) == a["job"]
    assert "policy" not in snap
    json.dumps(a["intent"])


def test_the_LINK_needs_ONSET_and_src_never_imports_LAB():
    w, a, b, pol = _duel()
    w.attention = "latch"
    with pytest.raises(ValueError, match="onset"):
        w.step()
    text = (SRC / "mindlink.py").read_text()
    assert not re.search(r"^\s*(from|import)\s+lab\b", text, re.M)


# ── M10: the approach ───────────────────────────────────────────────────────

class _Swinger:
    """The `_Duel` policy of tests/test_wounds.py: the attacker takes what
    is at his feet (or not, `bare`) and swings; the other stands."""
    name = "swinger"

    def __init__(self, bare):
        self.bare = bare

    def pick(self, sit, menu):
        want = "stay"
        if sit["who"] == "attacker" and sit["limb"] == "hands":
            want = "swing" if sit["holding"] or self.bare else "take"
        elif sit["limb"] == "hands":
            want = "keep"
        return next((i for i, o in enumerate(menu) if o["tag"] == want), 0)


def _first_blow(cells, square, they_square, bare):
    """The joules of the first blow the attacker's swing lands on the
    defender, standing `cells` voxels from him centre to centre in the
    `_duel` geometry, or None when it lands nothing. `square`: the attacker
    turned to face the defender (chest along the line); `they_square`: the
    defender turned to face him."""
    w = World(46, 20, 40, voxel_cm=5)
    w.fill(0, 46, 0, 20, 0, 1, STONE)
    w.exits = []
    a = _person(w, 17, 10)
    a["name"] = "attacker"
    b = _person(w, 17 + cells, 10)
    b["name"] = "defender"
    for p, way, turn in ((a, 1.0, square), (b, -1.0, they_square)):
        if turn:
            p["facing"] = (way, 0.0)
            assert w._square_up(p), "room to turn"
    w.fill(22, 24, 10, 11, 1, 2, IRON, edge=AXE)
    w.policy = _Swinger(bare)
    seen = w.events.next_id - 1
    try:
        for _ in range(25):
            w.step()
            for ev in w.events.since(seen):
                if ev.verb == "hits" and ev.target == ("person", b["pid"]):
                    return ev.db
            seen = w.events.next_id - 1
        return None
    finally:
        del w, a, b


@pytest.mark.parametrize("bare", [False, True], ids=["held", "bare"])
def test_the_STAND_OFF_BAND_is_where_a_swing_lands_half_its_best_blow(bare):
    """The calibration of `STAND_OFF_M` (design M10): the defender of
    tests/test_wounds.py `_duel` is moved through each band and a cell past
    either end of it, the attacker side-on as built (his swinging arm the
    near one) and squared up to him, and the defender either way too. A
    swing builds speed through its arc: inside the band every first blow
    carries at least half the best blow of its curve, whichever way the
    defender stands; a cell nearer, the blow has too little behind it for
    one of them, and a cell further it falls short or lands nothing.
    Measured: side, held 21.3-27.0 J inside of 38.6 at best; square 20.7-
    38.2 of 38.2. Squared up, the band sits 25-30 cm nearer."""
    vox = 0.05
    bands = {}
    for stance, square in (("side", False), ("square", True)):
        lo, hi = STAND_OFF_M[stance, not bare]
        k0, k1 = int(round(lo / vox)), int(round(hi / vox))
        for they in (False, True):
            curve = {k: _first_blow(k, square, they, bare) for k in range(k0 - 1, k1 + 2)}
            best = max(j for j in curve.values() if j)
            inside = {k: curve[k] for k in range(k0, k1 + 1)}
            assert all(j is not None and j >= 0.5 * best for j in inside.values()), \
                (stance, they, curve)
            bands[stance, they] = curve
        for edge in (k0 - 1, k1 + 1):
            assert any(bands[stance, they][edge] is None
                       or bands[stance, they][edge] < 0.5 * max(
                           j for j in bands[stance, they].values() if j)
                       for they in (False, True)), (stance, edge, bands)
    assert STAND_OFF_M["square", not bare][1] < STAND_OFF_M["side", not bare][0], \
        "squared up, the swing meets him nearer"


def test_a_STANCE_is_read_off_the_body():
    """Square: the chest along the line. Side: the shoulders along it, the
    swinging arm's the near one. The far arm would swing through its own
    body: no stance to strike from."""
    w = World(46, 20, 40, voxel_cm=5)
    w.fill(0, 46, 0, 20, 0, 1, STONE)
    w.exits = []
    p = _person(w, 23, 10)
    be = _be(mindlink.link(w))
    cells = be._cells(p)
    arm = w._hand(p, cells)
    sx = w._pivot_of(p, arm)[0] + cells[:, 0].min() - cells[:, 0].mean()
    near = (1.0, 0.0) if sx > 0 else (-1.0, 0.0)
    assert be._stance(p, cells, near) == "side"
    assert be._stance(p, cells, (-near[0], 0.0)) is None
    assert be._stance(p, cells, (0.0, 1.0)) == be._stance(p, cells, (0.0, -1.0)) == "square"


def _far(seed=0, argmax=True, gap=31, facing=None, record=False):
    """The duel stretched: the defender 1.55 m off (31 cells, centre to
    centre), the iron at the attacker's feet as in `_duel`, a door behind
    the defender."""
    w = World(70, 20, 40, voxel_cm=5)
    w.fill(0, 70, 0, 20, 0, 1, STONE)
    w.exits = [(68, 10)]
    a = _person(w, 10, 10)
    a["name"] = "attacker"
    b = _person(w, 10 + gap, 10)
    b["name"] = "defender"
    if facing is not None:
        b["facing"] = facing
    w.fill(15, 17, 10, 11, 1, 2, IRON, edge=AXE)
    a["sheet"] = {"harm": {"defender": 0.8}}
    pol = mindlink.link(w, mind=Mind(MindConfig(seed=seed)), record=record)
    if argmax:
        pol.mind.npc = Rules(seed, argmax=True)
    return w, a, b, pol


def _approach_run(w, a, b, pol, ticks=150):
    """Step until the first open wound; on every tick, the persons, where
    the attacker stands, and anyone in the air as a whole body."""
    out = {"persons": [], "ax": [], "aloft": [], "hit": None, "hit_d": None}
    seen = w.events.next_id - 1
    for _ in range(ticks):
        w.step()
        out["persons"].append(len(w.persons))
        out["ax"].append(a["anchor"])
        out["aloft"] += [(w.tick, b_.get("owner")) for b_ in w.bodies
                         if b_.get("owner") and not b_.get("part")]
        for ev in w.events.since(seen):
            if ev.verb == "hits" and ev.target == ("person", b["pid"]) and out["hit"] is None:
                out["hit"] = ev.tick
                out["hit_d"] = math.dist(a["anchor"], b["anchor"]) * 0.1 * w.scale
        seen = w.events.next_id - 1
        if _wounded(b):
            break
    out["falls"] = [ev.tick for ev in w.events.since(0) if ev.verb == "falls"]
    out["rows"] = [dict(r) for r in pol.mind.rows]
    out["jobs"] = [(e.t, e.who, e.verb, e.phase, e.reason) for e in pol.backend.job_events]
    out["open"], out["hurt"] = _wounded(b), b["hurt"]
    return out


@functools.lru_cache(maxsize=1)
def _far_run():
    w, a, b, pol = _far()
    start = math.dist(a["anchor"], b["anchor"]) * 0.1 * w.scale
    out = _approach_run(w, a, b, pol)
    out["start"] = start
    out["goes"] = [ev.tick for ev in w.events.since(0) if ev.verb == "goes"
                   and ev.actor == a["pid"]]
    del w, a, b, pol
    return out


def test_the_ATTACKER_APPROACHES_TAKES_THE_WEAPON_and_OPENS_A_WOUND_from_1_5_m():
    """The Verify list of M10. Nobody walks him there: the mind picks a
    strike at a man out of reach, and the job walks the legs to the band
    first (`in_reach`), the iron in his hand."""
    s = _far_run()
    assert s["start"] >= 1.5, s["start"]
    verbs = [r["verb"] for r in s["rows"] if r["who"] == "0"]
    assert verbs[:2] == ["take", "strike"], verbs
    assert s["goes"], "he walked to him"
    walked = abs(s["ax"][-1][0] - s["ax"][0][0]) * 0.05
    assert walked >= 0.75, walked
    assert s["open"], (s["hurt"], s["jobs"])
    lo, hi = STAND_OFF_M["side", True]
    assert lo - 1e-9 <= s["hit_d"] <= hi + 1e-9, \
        f"the blow was struck from the band: {s['hit_d']:.2f} m"


def test_NO_FALL_next_to_the_attacker_and_TWO_PERSONS_on_every_tick():
    """An approach that ran him into the other man, or a swing from too
    near, would put a body in the air (bug 87 waits for two-person falls)."""
    s = _far_run()
    assert s["falls"] == [] and s["aloft"] == [], (s["falls"], s["aloft"])
    assert set(s["persons"]) == {2}


def test_the_APPROACH_REPLAYS_exactly_and_from_a_restore():
    s = _far_run()
    w, a, b, pol = _far()
    for _ in range(20):                       # mid-approach
        w.step()
    snap = w.snapshot()
    again = _approach_run(w, a, b, pol)
    assert again["rows"] == s["rows"] and again["jobs"] == s["jobs"]
    assert again["hit"] == s["hit"] and again["hurt"] == s["hurt"]
    w.restore(snap)
    a, b = w.persons                          # the restored bodies
    replay = _approach_run(w, a, b, pol)
    key = lambda r: (r["t"], r["who"], r["reason"], r["text"], r["options"], r["pick"])
    assert [key(r) for r in replay["rows"]] == [key(r) for r in s["rows"]]
    assert replay["jobs"] == s["jobs"] and replay["hit"] == s["hit"]


def test_a_HARMFUL_START_in_view_is_a_SEES_ATTACK_percept():
    """The defender faces him: he sees the strike start ("attacker raised
    a hand against you"), and it stops his wait then, not when the wait
    runs out. Turned away, he sees nothing start."""
    w, a, b, pol = _far(facing=(-1.0, 0.0))
    for _ in range(14):
        w.step()
    be = _be(pol)
    starts = [x for x in be.percepts("1") if x.phase == "start"]
    assert starts and starts[0].kind == "harm_to_me" and starts[0].verb == "strike", starts
    assert starts[0].text == "attacker raised a hand against you"
    assert starts[0].actor.name == "attacker"
    t0 = starts[0].t
    stopped = [j for j in be.job_events if j.who == "1" and j.phase == "stopped"]
    assert stopped and stopped[0].reason == "interrupt:sees_attack" and stopped[0].t <= t0 + 1, \
        be.job_events
    w2, a2, b2, pol2 = _far(facing=(1.0, 0.0))
    for _ in range(14):
        w2.step()
    assert not [x for x in _be(pol2).percepts("1") if x.phase == "start"]


def test_a_FELT_BLOW_is_news_on_the_tick_it_is_felt_and_ANSWERED_A_REACTION_LATER():
    """A blow lands, and the defender's next question to the mind comes on
    the tick he feels it (the tick after the `hits`: what a body feels is
    written as the tick begins), not at his next decide period. His legs
    answer it a ready reaction later (he is in a fight: rows P12-P16)."""
    s = _argmax_run()
    asked = next(r for r in _rows("1") if r["t"] > s["first_hit"])
    assert asked["reason"] == "interrupt:struck" and asked["t"] == s["first_hit"] + 1, \
        (s["first_hit"], [(r["t"], r["reason"]) for r in _rows("1")])
    w, a, b, pol = _duel()
    react = w._ticks(sheet(b, "ready_s"))
    persons, hit = _step(w, b, asked["t"] + react + 12)
    goes = [ev.tick for ev in w.events.since(0) if ev.verb == "goes" and ev.actor == b["pid"]]
    assert goes and goes[0] >= asked["t"] + react, (asked["t"], react, goes)


def test_a_STAGED_REACTION_is_ANSWERED_ONCE():
    """R3's stages (will.py `_react`): a flinch asks only the eyes and the
    table turns them to the thing; a pre-movement is the one question to
    the mind, and its legs stay; the "go" asks the legs, and no question."""
    w, a, b, pol = _duel()
    for _ in range(2):
        w.step()
    m = w.minds[b["pid"]]
    n = len(pol.mind.rows)
    pol.prepare(w, b, "sees_thrown", phase="flinch")
    assert pol.limbs(w, b, "sees_thrown") == {"eyes"} and len(pol.mind.rows) == n
    menu = [{"key": "go on looking about", "tag": "about", "verb": "about"},
            {"key": "look toward what is coming", "tag": "look:cue", "verb": "look"}]
    sit = {"tick": w.tick, "who": b["name"], "limb": "eyes", "percept": "sees_thrown",
           "phase": "flinch", "reflexes": REFLEXES}
    assert pol.pick(sit, menu) == 1
    m.noticed = (w.tick, "sees_fire", 7)
    pol.prepare(w, b, "sees_fire", phase="premove")
    asked = pol.mind.rows[n:]
    assert [r["reason"] for r in asked] == ["interrupt:sees_fire"], asked
    assert "legs" not in pol.limbs(w, b, "sees_fire")
    job = b["job"]
    w.tick += 40                              # the pre-movement is over
    m.noticed = (w.tick, "sees_fire", 7)
    pol.prepare(w, b, "sees_fire", phase="go")
    assert len(pol.mind.rows) == n + 1 and b["job"] is job, "not asked again"
    if job.state["steps"][0]["limb"] == "legs":
        assert pol.limbs(w, b, "sees_fire") == {"legs"}


def test_what_a_body_HOLDS_is_a_LIST():
    """Everything in its hands, and a person it drags: each held thing can
    be dropped, the sharpest is what it strikes with, and a take is the
    hands' menu's (a free hand, and a menu that offers one)."""
    w, a, b, pol = _duel()
    be = _be(pol)
    a["held"] = {"cell": (22, 10, 1), "mat": int(IRON), "label": "iron", "arm": "right arm"}
    a["dragging"] = "defender"
    assert be.self_state("0").holds == ("defender", "the iron")
    a["dragging"] = None
    b["anchor"] = (a["anchor"][0] + 13, a["anchor"][1])    # 0.65 m: in the band
    opts = be.bind("0")
    assert [o.text for o in opts if o.verb == "drop"] == ["drop the iron"]
    assert not any(o.verb == "take" for o in opts), "the hands' menu offers none while it holds"
    strike = next(o for o in opts if o.verb == "strike")
    assert strike.role("instrument").name == "the iron"


# ── M12: the two-level menu, and no gate ────────────────────────────────────

def test_a_CALM_body_is_OFFERED_a_way_out_and_takes_it_RARELY():
    """The defender at the start of the duel fears nothing and wants nobody
    hurt. Nothing hides the door from him (M12): a walk and a flight out
    are offered, at NO_REASON against a wait at 0.2. With a take at 0 (a
    thing within his reach, wanted for nothing), a strike and a shout at
    NO_REASON, the rules leave in 2 e^-6 / (1 + e^-1 + 4 e^-6) = 0.36% of
    his decisions. The two-level menu gives the scope "go somewhere or
    leave" the same probability, and the rules' draws over 4,000 request
    ids take it about that often."""
    w, a, b, pol = _duel()
    be = _be(pol)
    opts = be.bind("1")
    prior, avg, hab = be.rule_scores("1", opts)
    full = core.menu(opts, prior, avg, hab, cat=catalog(), player=True)
    p = core.softmax([x + h for x, h in zip(full.prior, full.habit)], core.T_RULES)
    leave = sum(q for o, q in zip(full.options, p) if o.verb in mindlink.LEAVE)
    score = {o.verb: x for o, x in zip(full.options, full.prior)}
    assert score == {"walk_to": mindlink.NO_REASON, "flee_to": mindlink.NO_REASON,
                     "strike": mindlink.NO_REASON, "shout_fire": mindlink.NO_REASON,
                     "take": 0.0, "wait": 0.2}, score
    assert len(full.options) == 6
    assert leave == pytest.approx(2 * math.exp(-6) / (1 + math.exp(-1) + 4 * math.exp(-6)), rel=1e-9)
    lv = scope.levels(full, catalog(), be.self_state("1").drives)
    p1 = core.softmax(list(lv.prior), core.T_RULES)
    assert p1[lv.texts.index("go somewhere or leave")] == pytest.approx(leave, rel=1e-9)
    rules, n = Rules(0), 4000
    texts = tuple(o.text for o in full.options)
    took = 0
    for k in range(n):
        r = DecisionRequest(f"0:{k}:1", "1", "idle", "", "", texts, ("x",) * len(texts),
                            full.prior, full.prior_avg, full.habit)
        took += full.options[rules.decide(r).pick].verb in mindlink.LEAVE
    assert 5 <= took <= 45, (took, leave * n)


def test_the_DUEL_in_TWO_QUESTIONS_18_of_20_SAMPLED_seeds_open_a_wound():
    """The sampled-seeds test of M8, with the two-level menu: a scope, then
    an act, both drawn by the rules. No gate: the defender is offered the
    door and a walk over to strike, and does neither unless he draws it."""
    opened = []
    for seed in range(20):
        w, a, b, pol = _duel(seed=seed, argmax=False, menu="two")
        _step(w, b, 150, until_wound=True)
        opened.append(_wounded(b))
        assert all(r["menu"] == "two" and sum(r["asked"]) <= 2 for r in pol.mind.rows)
        del w, a, b, pol
    assert sum(opened) >= 18, opened


@functools.lru_cache(maxsize=1)
def _two_run():
    """The verify scene in two questions, argmax, a model that must never
    be asked, recorded."""
    trip = _Tripwire()
    w, a, b, pol = _duel(model=trip, record=True, menu="two")
    out = _summary(w, a, b, pol, *_step(w, b, 150))
    out["asked"] = list(trip.asked)
    return out


def test_TWO_QUESTIONS_open_a_wound_ask_no_model_and_RECORD_both_levels():
    s = _two_run()
    assert s["open"], (s["hurt"], [(r["t"], r["who"], r["text"]) for r in s["rows"]])
    first = _rows("0", s)
    assert [r["verb"] for r in first[:2]] == ["take", "strike"]
    assert first[0]["scope"]["options"][first[0]["scope"]["pick"]] == "do something with the iron"
    assert s["asked"] == [] and set(s["remote"].values()) == {0}
    assert all(sum(r["asked"]) <= 2 for r in s["rows"])
    levels = collections.Counter(c["level"] for c in s["cases"])
    assert levels["full"] == len(s["rows"]) and levels["scope"] and levels["act"]


def test_TWO_QUESTIONS_REPLAY_exactly_and_from_a_restore():
    s = _two_run()
    w, a, b, pol = _duel(record=True, menu="two")
    persons, hit = _step(w, b, 40)
    snap = w.snapshot()
    more, hit2 = _step(w, b, 110)
    again = _summary(w, a, b, pol, persons + more, hit if hit is not None else hit2)
    assert again["rows"] == s["rows"] and again["jobs"] == s["jobs"] and again["cases"] == s["cases"]
    w.restore(snap)
    _step(w, b, 110)
    key = lambda r: (r["t"], r["who"], r["reason"], r["text"], r["options"], r["pick"], r["scope"])
    assert [key(r) for r in pol.mind.rows] == [key(r) for r in s["rows"]]
