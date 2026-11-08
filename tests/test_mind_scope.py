"""The two-level menu: scope, then act (step M12, src/mind/scope.py).

The tests do three things:
- THE LEVELS, on menus made by hand: the scopes hold every legal option
  once; level 1 has at most 6 entries, level 2 at most 8, and NOTHING IS
  CUT: every option is reachable (a big scope is split by its verbs, the
  scopes that do not fit alone are packed in named bins, and a menu too big
  for two questions gets "something else", which opens a page); the scopes
  alone are the salient ones; no scope is at zero, and the rules' draw of a
  scope then an act is their flat draw.
- NIGHTS OF THE INN in small questions: every option reachable in every
  decision; two requests a decision unless "something else" opens a page;
  the job is an option of its scope; the log keeps one draw over the full
  menu; the same night twice; reactive mode never asks the model; the model
  answers the same small questions.
- THE RECORD: case rows for every level and the full menu, so a later run
  can measure the model's recall (does its scope hold its full-menu pick?),
  and a record, score and replay round that gives the same night; and every
  question a decision can ask (record.tree), in the loop's own words.
- THE NAMES (wave 11): short and unambiguous, and they give way, one rung
  at a time, only to save a third question.
"""
import collections
import math
import re

import pytest

from src.mind import core, record, scope
from src.mind.api import Decision, Option, Percept, Ref, SelfState
from src.mind.choosers import Rules
from src.mind.loop import Mind, MindConfig, run
from src.rooms import RoomsBackend
from src.rooms.__main__ import SETTING

CAT = core.catalog()
T = core.T_RULES


def P(name):
    return Ref("person", name, name)


def Th(name):
    return Ref("thing", name, name)


def Pl(name):
    return Ref("place", name, name)


def opt(verb, **roles):
    return CAT.option("rooms", verb, roles)


def softmax(z, t=1.0):
    """Written out again here, so that the tests do not check the code with itself."""
    m = max(z)
    e = [math.exp((x - m) / t) for x in z]
    return [x / sum(e) for x in e]


def _crowd():
    """A crowded room: 47 legal options, of 5 people, 10 things, 3 places."""
    people = ["Oren", "Tom", "Sir Ada", "Marta", "the bandit"]
    things = ["the stool", "the dice", "the mug of ale", "the iron pot", "the lantern",
              "the bucket of water", "the kitchen knife", "the loaf of bread", "the bowl of stew", "the ladle"]
    out = [opt("walk_to", place=Pl(r)) for r in ("the kitchen", "the yard", "the cellar")]
    out += [opt("take", target=Th(n)) for n in things]
    for v in ("praise", "strike", "insult", "threaten", "shove", "search"):
        out += [opt(v, target=P(n)) for n in people]
    return out + [opt("shout_for_help"), opt("wait")]


def _full(options, scores=None):
    s = scores or [0.0] * len(options)
    return core.menu(options, s, s, [0.0] * len(options), cat=CAT, player=True)


# ---------------------------------------------------------------- the levels
def test_the_SCOPES_hold_every_legal_option_ONCE():
    full = _full(_crowd())
    every = scope.partition(full, CAT)
    got = sorted(i for s in every for i in s.members)
    assert got == list(range(len(full.options)))
    kinds = collections.Counter(s.kind for s in every)
    assert kinds == {"deal": 5, "talk": 5, "thing": 10, "go": 1, "call": 1, "wait": 1}
    texts = {s.text for s in every}
    assert {"deal with Oren", "talk with Oren", "do something with the ladle", "go somewhere or leave",
            "call out", "wait"} <= texts and len(texts) == len(every)


def _reached(lv, path=()):
    """Every (option, the entries picked on the way to it) of the pages of
    `lv`, walked by hand. Checks the sizes of every page and level 2."""
    assert len(lv.scopes) <= scope.SCOPE_MAX
    out = []
    for j, (s, a) in enumerate(zip(lv.scopes, lv.acts)):
        if isinstance(a, scope.Level):
            assert s.kind == scope.ELSE
            out += _reached(a, path + (j,))
        else:
            assert len(a.options) <= scope.ACT_MAX and a.cut == ()
            assert list(a.options) == [lv.full.options[i] for i in s.members]      # canonical, all
            out += [(o, path + (j,)) for o in a.options]
    return out


def test_LEVEL_ONE_has_six_entries_and_EVERY_option_is_REACHABLE_in_TWO_questions():
    """45 legal options: level 1 has 6 entries, each of 8 options or fewer,
    and every option is in exactly one of them. The scopes that do not fit
    alone are packed in bins, named by what is in them."""
    full = _full(_crowd())
    lv = scope.levels(full, CAT)
    assert len(full.options) == 45 and len(lv.scopes) == scope.SCOPE_MAX and lv.depth == 2
    got = _reached(lv)
    assert sorted((o for o, _ in got), key=full.options.index) == list(full.options)
    assert all(len(p) == 1 for _, p in got) and lv.folded == ()
    held = sorted(i for s in lv.scopes for i in s.members)
    assert held == list(range(len(full.options)))
    # canonical order (the first option of each entry), never by score
    firsts = [s.members[0] for s in lv.scopes]
    assert firsts == sorted(firsts)
    # a bin says what is in it
    for s in lv.scopes:
        if s.kind == scope.BIN:
            words = [full.options[i].text for i in s.members]
            assert "; or " in s.text and len(words) <= scope.ACT_MAX
    assert any("go somewhere or leave" in t for t in lv.texts)
    # and a short menu is its scopes, no bin, no "something else"
    small = _full([opt("strike", target=P("Oren")), opt("walk_to", place=Pl("the yard")), opt("wait")])
    lv = scope.levels(small, CAT)
    assert lv.texts == ("deal with Oren", "go somewhere or leave", "wait") and lv.folded == ()


def _bigger():
    """51 legal options: more than two small questions can hold."""
    return _crowd() + [opt(v, target=P(n)) for v in ("praise", "strike") for n in ("Wil", "Ned", "Jon")]


def test_a_menu_TOO_BIG_for_two_questions_has_SOMETHING_ELSE_and_NOTHING_is_HIDDEN():
    """6 x 8 is the most that two small questions hold. A bigger menu gets
    "something else", which opens a page of what it folded: a third
    question, asked only when "something else" is picked."""
    full = _full(_bigger())
    lv = scope.levels(full, CAT)
    assert len(full.options) == 51 and lv.depth == 3
    assert lv.scopes[-1].kind == scope.ELSE and lv.scopes[-1].text == "something else"
    assert "wait" in lv.texts
    got = _reached(lv)
    assert sorted((o for o, _ in got), key=full.options.index) == list(full.options)
    deep = [o for o, p in got if len(p) == 2]
    assert deep and all(p[0] == len(lv.scopes) - 1 for o, p in got if len(p) == 2)
    page = lv.acts[-1]
    assert sorted(i for s in page.scopes for i in s.members) == sorted(lv.scopes[-1].members)
    assert page.folded == () and tuple(lv.folded) == page.scopes
    assert scope.means("Fenn", lv.scopes[-1]) == "Fenn means to do something else."


def test_the_SCOPES_ALONE_are_the_SALIENT_ones():
    """When not every scope fits alone, the ones alone are the job it was
    doing, then whom the news is about, then what serves a drive it has;
    never the rule scores. The rest are packed."""
    people = ["Oren", "Tom", "Sir Ada", "Marta", "Wil", "Ned", "the bandit"]
    opts = [opt(v, target=P(n)) for n in people for v in ("strike", "shove", "search")] + [opt("wait")]
    full = _full(opts, [5.0 if o.role("target") and o.role("target").name == "Ned" else 0.0 for o in opts])
    hit = Percept(3, "saw", "the bandit struck you", "e9", "strike", P("the bandit"), P("Fenn"),
                  kind="harm_to_me", salience=1.0)
    last = next(o for o in full.options if o.text == "search Wil").key
    lv = scope.levels(full, CAT, drives=("safety",), percepts=(hit,), last=last)
    # 7 people of 3 (Wil's search is "keep"), wait: room for 3 alone and 2 bins
    # (Marta is first in canonical order of those that serve safety)
    assert lv.texts == ("deal with Marta", "deal with Ned, Oren or Wil", "deal with Sir Ada or Tom",
                        "deal with the bandit", "keep at it (search Wil)", "wait"), lv.texts
    assert all(len(s.members) <= scope.ACT_MAX for s in lv.scopes) and lv.depth == 2
    # the rules love Ned, and still he is in a bin
    assert not any(t == "deal with Ned" for t in lv.texts)
    # an interrupted wait is still "wait"
    wait = next(o for o in full.options if o.verb == "wait").key
    assert "wait" in scope.levels(full, CAT, last=wait).texts
    # a drive: six people to praise, and a shout for help that serves safety
    people = ["Oren", "Tom", "Sir Ada", "Marta", "Wil", "Ned"]
    full = _full([opt("praise", target=P(n)) for n in people] + [opt("shout_for_help"), opt("wait")])
    calm = scope.levels(full, CAT)
    afraid = scope.levels(full, CAT, drives=("safety",))
    assert "call out" not in calm.texts and "call out" in afraid.texts
    assert any(t.endswith("; or call out") for t in calm.texts)       # in a bin, not hidden


def _parts(text):
    """The parts of a bin's name: "a; b; or c" -> [a, b, c]."""
    return text.replace("; or ", "; ").split("; ")


def _room(people, things):
    """A room of `people` (six acts on each) and `things` (take), a place and wait."""
    opts = [opt(v, target=P(n)) for n in people for v in ("strike", "shove", "search", "praise", "insult",
                                                          "threaten")]
    return _full(opts + [opt("take", target=Th(t)) for t in things]
                 + [opt("walk_to", place=Pl("the yard")), opt("wait")])


def test_BIN_NAMES_are_SHORT_and_UNAMBIGUOUS_and_GIVE_WAY_only_to_save_a_QUESTION():
    """M12's verifier: a name of 192 characters; "keep at it (...)" read as
    one of Sir Ada's verbs inside "deal with Sir Ada: steal, keep at it
    (...), or call out"; "wait" inside a bin. Now the parts of a name are
    cut by ";", keep and wait stay alone, and a name is at most NAME_MAX
    characters when a cap is set (the user chose none, 2026-10-16); these give way (LADDER) only when the page would otherwise
    need "something else"."""
    people = ["Oren", "Tom", "Sir Ada", "Marta", "Wil"]
    things = ["the stool", "the dice", "the mug of ale", "the lantern", "the iron pot", "the ladle",
              "the bucket of water", "the kitchen knife", "the loaf of bread", "the bowl of stew"]
    starts = ("deal with", "talk with", "do something with", "get or use", "go somewhere", "call out", "wait",
              "keep at it")
    # 30 options and a job to keep at: the strict rung fits
    full = _room(people[:4], things[:4])
    last = next(o for o in full.options if o.text == "search Tom").key
    lv = scope.levels(full, CAT, last=last)
    assert lv.depth == 2 and lv.relaxed == 0 and len(lv.scopes) == scope.SCOPE_MAX
    assert "keep at it (search Tom)" in lv.texts and "wait" in lv.texts
    assert any("; or " in t for t in lv.texts)
    for s in lv.scopes:
        assert scope.NAME_MAX is None or len(s.text) <= scope.NAME_MAX, s.text
        parts = _parts(s.text)
        # each part is a scope's own name; wait and keep are never a part of a bin
        assert all(p.startswith(starts) for p in parts), s.text
        assert len(parts) == 1 or not any(p == "wait" or p.startswith("keep at it") for p in parts), s.text
    # 42 options: keep and wait alone leave 4 x 8 = 32 places for 41, so the
    # strict rung needs "something else"; wait joins a bin instead
    full = _room(people, things)
    strict = scope.levels(full, CAT, ladder=scope.LADDER[:1])
    loose = scope.levels(full, CAT)
    assert len(full.options) == 42
    assert strict.depth == 3 and strict.scopes[-1].kind == scope.ELSE and "wait" in strict.texts
    assert loose.depth == 2 and loose.relaxed == 1 and not loose.folded
    assert any(len(_parts(t)) > 1 and "wait" in _parts(t) for t in loose.texts), loose.texts
    assert scope.NAME_MAX is None or all(len(t) <= scope.NAME_MAX for t in loose.texts)
    # the crowd and a job: only the last rung saves the question; keep goes
    # into a bin, as a part of its own after ";", not one of a person's verbs
    full = _full(_crowd())
    last = next(o for o in full.options if o.text == "search Tom").key
    crowd = scope.levels(full, CAT, last=last)
    assert crowd.depth == 2 and crowd.relaxed == len(scope.LADDER) - 1
    assert any(p.startswith("keep at it") for t in crowd.texts if len(_parts(t)) > 1 for p in _parts(t))
    assert scope.levels(full, CAT, last=last, ladder=scope.LADDER[:1]).depth == 3
    for lv in (loose, strict, crowd):   # nothing is hidden on any rung
        got = _reached(lv)
        assert sorted((o for o, _ in got), key=lv.full.options.index) == list(lv.full.options)


def test_a_BIG_SCOPE_is_SPLIT_by_its_VERBS_and_NOTHING_is_CUT():
    people = ["Oren", "Tom", "Marta"]
    things = ["the purse", "the lute", "the dagger", "the ale"]
    opts = [opt("tell", target=P("Sir Ada"), culprit=P(c), thing=Th(t)) for c in people for t in things]
    opts += [opt("praise", target=P("Sir Ada")), opt("insult", target=P("Sir Ada")), opt("wait")]
    scores = [3.0 - 0.1 * i if o.verb == "tell" else -2.0 for i, o in enumerate(opts)]
    full = _full(opts, scores)
    lv = scope.levels(full, CAT)
    parts = [s for s in lv.scopes if s.kind == "talk"]
    assert [s.text for s in parts] == ["talk with Sir Ada: praise or tell (1 of 2)",
                                       "talk with Sir Ada: tell (2 of 2) or insult"], lv.texts
    assert sorted(i for s in parts for i in s.members) == [i for i, o in enumerate(full.options)
                                                           if o.verb != "wait"]
    for s, m in zip(lv.scopes, lv.acts):
        assert len(m.options) <= scope.ACT_MAX and m.cut == ()
        assert list(m.options) == [full.options[i] for i in s.members]
        assert m.prior == tuple(full.prior[i] for i in s.members)
    # a part holds whole verbs while they fit; a verb too big is cut in even pieces
    split = scope.split(scope.partition(full, CAT)[0], full, 8)
    assert [len(s.members) for s in split] == [7, 7]


def test_NO_SCOPE_is_at_ZERO_and_the_RULES_two_draws_are_their_FLAT_draw():
    """A scope's rule score is the soft maximum of its options': the rules'
    probability of a scope is the sum of theirs, so scope then act is the
    flat draw over the same options when nothing is cut."""
    opts = [opt("strike", target=P("Oren")), opt("shove", target=P("Oren")),
            opt("praise", target=P("Oren")), opt("take", target=Th("the axe")),
            opt("walk_to", place=Pl("the yard")), opt("walk_to", place=Pl("the cellar")), opt("wait")]
    prior = [1.1, -0.4, -2.0, 0.6, -3.0, -2.5, 0.2]
    avg = [0.3, 0.1, -1.0, 0.6, -1.0, -2.0, 0.2]
    habit = [0.0, -0.8, 0.0, 0.0, -1.0, 0.0, 0.0]
    full = core.menu(opts, prior, avg, habit, cat=CAT, player=True)
    lv = scope.levels(full, CAT)
    assert all(not m.cut for m in lv.acts)
    flat = softmax([a + h for a, h in zip(full.prior, full.habit)], T)
    flat_avg = softmax([a + h for a, h in zip(full.prior_avg, full.habit)], T)
    p1 = softmax([a + h for a, h in zip(lv.prior, lv.habit)], T)
    a1 = softmax(list(lv.prior_avg), T)
    assert min(p1) > 0 and min(a1) > 0
    for k, (s, m) in enumerate(zip(lv.scopes, lv.acts)):
        p2 = softmax([a + h for a, h in zip(m.prior, m.habit)], T)
        for o, q in zip(m.options, p2):
            i = full.options.index(o)
            assert p1[k] * q == pytest.approx(flat[i], rel=1e-12)
        assert a1[k] == pytest.approx(sum(flat_avg[i] for i in s.members), rel=1e-12)
    # the walk to the yard is the calm man's way out: rare, never zero
    go = lv.texts.index("go somewhere or leave")
    assert 0 < p1[go] < 0.001


def test_a_LEVEL_is_REFUSED_what_it_cannot_hold():
    full = _full(_crowd())
    with pytest.raises(ValueError, match="level 1"):
        scope.levels(full, CAT, scope_max=2)
    with pytest.raises(ValueError, match="level 2"):
        scope.levels(full, CAT, act_max=0)
    with pytest.raises(ValueError, match="menu"):
        MindConfig(menu="three")


# ---------------------------------------------------------------- nights in two questions
def _night(seed, turns=120, model=None, record_=False, **cfg):
    b = RoomsBackend(seed, turns)
    b = record.Recorder(b) if record_ else b
    mind = Mind(MindConfig(seed=seed, setting=SETTING, menu="two", **cfg), model=model)
    run(b, mind, turns)
    return b, mind


class Sense:
    """A model for the tests: it scores each option by its words, and keeps
    every request it gets."""
    name, remote = "fake", True

    def __init__(self):
        self.seen, self._done = [], {}

    def submit(self, r):
        self.seen.append(r)
        s = tuple((sum(map(ord, t)) % 97 + 1) / 98 for t in r.options)
        self._done[r.id] = Decision(r.id, max(range(len(s)), key=s.__getitem__), self.name, s)

    def poll(self, req_id, block=True):
        return self._done.pop(req_id, None)

    def average(self, r):
        return None


@pytest.fixture(scope="module")
def nights():
    return [_night(seed) for seed in range(2)]


def test_TWO_QUESTIONS_unless_the_menu_is_too_big_and_the_job_is_an_option_of_its_scope(nights):
    n = 0
    for b, mind in nights:
        log = {e["id"]: e for e in b.world.log if e.get("options")}
        for r in mind.rows:
            n += 1
            sc = r["scope"]
            assert r["menu"] == "two" and len(sc["options"]) <= scope.SCOPE_MAX
            assert len(r["options"]) <= scope.ACT_MAX and r["options"][r["pick"]] == r["text"]
            assert r["text"] in sc["members"][sc["pick"]] and r["text"] in r["full"]
            assert sorted(t for m in sc["members"] for t in m) == sorted(r["full"])
            assert r["cut"] == [] and r["act"]["members"] == r["options"]
            # a third question only on a page that "something else" opened,
            # and only on a menu too big for two
            assert len(r["asked"]) == 2 + len(r["else"]) and r["chooser"] == "rules"
            if len(r["full"]) <= scope.SCOPE_MAX * scope.ACT_MAX:
                assert r["else"] == []
            for prev, page in zip([sc] + r["else"], r["else"]):
                assert prev["kinds"][prev["pick"]] == scope.ELSE
                assert sorted(t for m in page["members"] for t in m) == sorted(prev["members"][prev["pick"]])
            assert 0 < r["joint"] <= 1 and r["why"]["score"] == pytest.approx(r["joint"], abs=1e-6)
            assert r["situation"].startswith(sc["situation"]) and " means to " in r["situation"]
        # the log keeps one draw over every option offered at level 2
        for e in log.values():
            assert abs(sum(e["p_char"]) - 1) < 1e-3 and abs(sum(e["p_avg"]) - 1) < 1e-3
            assert e["options"][e["chosen"]] == e["text"]
    assert n > 200
    rows = [r for _, m in nights for r in m.rows]
    assert any(r["else"] for r in rows) and any(sum(r["asked"]) == 2 for r in rows)
    kinds = collections.Counter(r["scope"]["kinds"][r["scope"]["pick"]] for r in rows)
    assert {"wait", "go"} <= set(kinds) and len(kinds) >= 4, kinds


def test_on_a_NIGHT_every_legal_option_is_REACHABLE_in_every_decision(monkeypatch):
    """The verify of M12 found 20% of the options of an inn night out of
    reach of the cut level 2. Nothing is cut now: walk every page of every
    decision of a night."""
    seen, levels = [], scope.levels

    def keep(*a, **k):
        lv = levels(*a, **k)
        seen.append(lv)
        return lv

    monkeypatch.setattr(scope, "levels", keep)
    _night(3, 300)
    deep = 0
    for lv in seen:
        got = _reached(lv)
        assert sorted((o for o, _ in got), key=lv.full.options.index) == list(lv.full.options)
        deep += lv.depth > 2
        assert lv.depth == 2 or len(lv.full.options) > scope.SCOPE_MAX * scope.ACT_MAX
    assert len(seen) > 300 and deep < 0.2 * len(seen), (len(seen), deep)


def test_the_SAME_NIGHT_twice_and_REACTIVE_mode_never_asks_the_model(nights):
    sense = Sense()
    b, mind = _night(0, model=sense)
    assert sense.seen == [] and set(mind.remote.values()) == {0}
    assert mind.rows == nights[0][1].rows
    assert b.world.log == nights[0][0].world.log


def test_the_MODEL_answers_the_same_TWO_SMALL_QUESTIONS():
    sense = Sense()
    b, mind = _night(1, 60, model=sense, mode="all", gate=None)
    assert {r["chooser"] for r in mind.rows} == {"combined"}
    per = collections.Counter(r.id.rsplit(":", 1)[0] for r in sense.seen)
    pages = {r["id"]: len(r["else"]) for r in mind.rows}
    assert per and all(n <= 2 + pages[k] for k, n in per.items())
    assert max(mind.remote.values()) <= (2 + max(pages.values())) * len({r["who"] for r in mind.rows})
    for r in sense.seen:
        level = r.id.rsplit(":", 1)[1]
        assert level == "act" or re.fullmatch(r"scope\d*", level)
        assert len(r.options) <= (scope.ACT_MAX if level == "act" else scope.SCOPE_MAX)
        assert r.prior == r.prior_avg == r.habit == (0.0,) * len(r.options)   # texts only


def test_the_RECORD_has_BOTH_LEVELS_and_a_RECORD_SCORE_REPLAY_gives_the_same_night():
    sense = Sense()
    b, mind = _night(2, 60, model=sense, record_=True, mode="all", gate=None)
    cases = b.cases(mind.rows)
    levels = collections.Counter(c["level"] for c in cases)
    assert levels["full"] == len(mind.rows) and levels["scope"] and levels["act"]
    for c in cases:
        assert c["digest"] == record.digest(c["situation"], c["question"], c["options"])
        if c["level"] == "scope":
            # recall: each scope's members are full-menu options. On page 1
            # they are all of them (some behind "something else"); a later
            # page holds only what "something else" stood for. With no cap
            # on names (2026-10-16) this night opens such a page; with the
            # 80 cap it happened not to.
            flat = sorted(t for m in c["members"] for t in m)
            if c["page"] == 1:
                assert flat == sorted(c["full_options"])
            else:
                assert flat and set(flat) < set(c["full_options"])
    asked = {r.id for r in sense.seen}
    assert asked == {c["id"] for c in cases if c["level"] != "full"}
    # score the asked rows with the model, and replay them
    rows = []
    for c in cases:
        if c["level"] == "full":
            continue
        s = Sense()
        s.submit(_req(c))
        d = s.poll(c["id"])
        rows.append({"request": c["id"], "digest": c["digest"], "pick": d.pick,
                     "scores": list(d.probs), "chooser": "fake"})
    again = Mind(MindConfig(seed=2, setting=SETTING, menu="two", mode="all", gate=None),
                 model=record.Replay(rows))
    b2 = RoomsBackend(2, 60)
    run(b2, again, 60)
    assert again.rows == mind.rows and b2.world.log == b.world.log


def test_the_TREE_has_EVERY_QUESTION_in_the_LOOP_S_OWN_WORDS(monkeypatch):
    """record.tree gives every question one decision can ask: each page
    (asked or not) and the act question of each entry, so the lab can
    follow a model down its own path. The questions the run asked are among
    them, byte for byte (the same digest)."""
    seen, levels = [], scope.levels

    def keep(*a, **k):
        lv = levels(*a, **k)
        seen.append(lv)
        return lv

    monkeypatch.setattr(scope, "levels", keep)
    b, mind = _night(4, 300, record_=True)
    assert len(seen) == len(mind.rows)
    n_deep = 0
    for r, lv in zip(mind.rows, seen):
        name = r["question"][len("What does "):-len(" do next?")]
        rows = record.tree(r["id"], lv, r["scope"]["situation"], r["question"], name, tag="x|", who=r["who"])
        by = {c["id"]: c for c in rows}
        assert len(by) == len(rows) and all(c["who"] == r["who"] for c in rows)
        for c in rows:
            assert c["digest"] == record.digest(c["situation"], c["question"], c["options"])
        full = by[f"{r['id']}:full"]
        assert full["options"] == r["full"] and full["level"] == "full"
        pages = [c for c in rows if c["level"] == "scope"]
        assert [c["page"] for c in pages] == list(range(1, len(pages) + 1)) and len(pages) == lv.depth - 1
        n_deep += len(pages) > 1
        for c in pages:
            assert c["id"] == f"x|{r['id']}:scope{c['page'] if c['page'] > 1 else ''}"
            # the act question of each entry holds that entry's members, in order
            for j, (k, m) in enumerate(zip(c["kinds"], c["members"])):
                if k == scope.ELSE:
                    nxt = by[f"x|{r['id']}:scope{c['page'] + 1}"]
                    assert sorted(t for x in nxt["members"] for t in x) == sorted(m)
                    assert nxt["situation"] == f"{r['scope']['situation']} {name} means to do something else."
                else:
                    a = by[f"x|{r['id']}:act{c['page']}.{j}"]
                    assert a["options"] == m and a["asked"] == (len(m) > 1)
                    assert a["situation"] == f"{r['scope']['situation']} {name} means to {c['options'][j]}."
        assert sorted(t for m in pages[0]["members"] for t in m) == sorted(r["full"])
        # the run's own questions are in the tree, byte for byte
        tree = {c["digest"] for c in rows}
        assert {c["digest"] for c in record.cases_two(r, "rooms")} <= tree
    assert n_deep > 0


def _req(c):
    from src.mind.api import DecisionRequest
    n = len(c["options"])
    return DecisionRequest(c["id"], c["who"], c["reason"], c["situation"], c["question"],
                           tuple(c["options"]), tuple(c["groups"]), (0.0,) * n, (0.0,) * n, (0.0,) * n)


def test_a_PLAYER_keeps_the_FLAT_full_menu():
    asked = []

    def ask(r):
        asked.append(r)
        return r.options.index("wait")

    b = RoomsBackend(0, 40)
    mind = Mind(MindConfig(seed=0, setting=SETTING, menu="two", player="Fenn"), ask=ask)
    run(b, mind, 40)
    mine = [r for r in mind.rows if r["who"] == "Fenn"]
    assert mine and all("scope" not in r and r["cut"] == [] for r in mine)
    assert all("scope" in r for r in mind.rows if r["who"] != "Fenn")
    assert asked and all(":" not in r.id.split(":", 2)[2] for r in asked)


def test_SCOPE_is_pure_and_STANDARD_LIBRARY_only():
    import ast
    from pathlib import Path
    import sys
    src = Path(scope.__file__)
    tree = ast.parse(src.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            assert node.module.split(".")[0] in sys.stdlib_module_names, node.module
        elif isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] in sys.stdlib_module_names for a in node.names)
    full = _full(_bigger())
    assert scope.levels(full, CAT) == scope.levels(full, CAT)
    st = SelfState("Fenn")
    assert scope.means(st.name, scope.levels(full, CAT).scopes[-1]) == "Fenn means to do something else."
    assert isinstance(full.options[0], Option)
