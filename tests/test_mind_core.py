"""The neutral functions of the mind (docs/sim-mind-design.md 3.5, step M4):
the menu, the facts of a scene, and the scene text.

The tests do three things:
- THE RULES, one by one, on small menus and scenes made by hand: the
  pre-rank, canonical order, the null act, blocked options, the people a
  scene names, the beliefs it states, what it says is missing, the words.
- SIX INN NIGHTS (src/rooms) run through the menu and the scene text, with
  the numbers that the design names: menus of 16 or fewer, each with the null
  act and no text two times; the rules' favourite first in few menus; less
  than 1% of the rules' probability cut; needs said; no list of things, no
  digits, at most 2 beliefs and 5 percepts.
- THE SAME BYTES: the same input gives the same text and the same menus, in
  this process and in two other processes with other hash seeds.
"""
import collections
import dataclasses
import functools
import hashlib
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest

from src.mind import core, describe as D
from src.mind.api import Belief, DecisionRequest, Percept, Ref, SelfState, Why
from src.rooms import RoomsBackend, rules

ROOT = Path(__file__).resolve().parents[1]
CAT = core.catalog()
SETTING = "It is night at the village inn"


def P(name):
    return Ref("person", name, name)


def T(name):
    return Ref("thing", name, name)


def Pl(name):
    return Ref("place", name, name)


def opt(verb, missing=(), **roles):
    return CAT.option("rooms", verb, roles, missing=missing)


def at(thing, v, src="sight", how="looked", t=0):
    """An `at` belief, with the words the inn writes for it."""
    text = (f"{v.name} has {thing}" if isinstance(v, Ref) and v.kind == "person" else
            f"{thing} is in {v.name}" if isinstance(v, Ref) else f"{thing} is {v}")
    return Belief("at", T(thing), v, text, src, how, t)


def took(thing, who, src="e1", how="saw", t=1):
    return Belief("took", T(thing), P(who), f"{who} took {thing}", src, how, t)


def menu_of(options, scores=None, **kw):
    s = scores or [0.0] * len(options)
    return core.menu(options, s, s, [0.0] * len(options), cat=CAT, **kw)


FENN = SelfState("Fenn", "a traveller", "the main room", ("the dagger",), ("hungry", "angry at Oren"),
                 ("hunger", "harm"))


# ---------------------------------------------------------------- the menu
def _crowd():
    """A crowded room: 41 legal options of 9 verbs."""
    people = ["Oren", "Tom", "Sir Ada", "Marta", "the bandit"]
    things = ["the stool", "the dice", "the mug of ale", "the iron pot", "the lantern", "the bucket of water",
              "the kitchen knife", "the loaf of bread", "the bowl of stew", "the ladle"]
    out = [opt("walk_to", place=Pl(r)) for r in ("the kitchen", "the yard", "the cellar")]
    out += [opt("take", target=T(n)) for n in things]
    for v in ("praise", "strike", "insult", "threaten", "shove", "search"):
        out += [opt(v, target=P(n)) for n in people]
    return out + [opt("wait")]


def test_a_SHORT_menu_is_not_cut_and_comes_in_CANONICAL_order():
    os_ = [opt("wait"), opt("strike", target=P("Oren")), opt("praise", target=P("Oren")),
           opt("take", target=T("the stool")), opt("walk_to", place=Pl("the yard")),
           opt("strike", target=P("Oren"), instrument=T("the dagger"))]
    m = menu_of(os_, [0.0, 2.0, -1.0, 0.5, -0.5, 3.0])
    # group (body, care, fight, handle, move, social, wait), then verb, then text; never by score
    assert [o.text for o in m.options] == ["strike Oren", "strike Oren with the dagger", "take the stool",
                                           "walk to the yard", "praise Oren", "wait"]
    assert m.cut == () and m.blocked == () and m.full == m.options
    assert m.prior == (2.0, 3.0, 0.5, -0.5, -1.0, 0.0)          # the numbers follow their options
    assert [g for g, _ in m.groups] == ["fight", "handle", "move", "social", "wait"]


def test_the_PRE_RANK_keeps_16_with_the_null_act_and_each_verb():
    os_ = _crowd()
    verbs = list(dict.fromkeys(o.verb for o in os_))
    # one option the rules want, the null act far down
    s = [-2.0] * len(os_)
    s[os_.index(opt("take", target=T("the dice")))] = 3.0
    s[-1] = -5.0
    m = menu_of(os_, s)
    assert len(m.options) == core.MENU_MAX and len(m.full) == len(os_)
    assert {o.verb for o in m.options} == set(verbs)                # the null act, and each verb
    assert "take the dice" in {o.text for o in m.options}
    assert {o.key for o in m.options + m.cut} == {o.key for o in os_} and not set(m.options) & set(m.cut)
    assert [core.canonical(CAT, o) for o in m.options] == sorted(core.canonical(CAT, o) for o in m.options)
    # flat scores: the first FIRST options in canonical order hold too little of
    # the mass to stop step 2; then the null act, then other verbs while there is room
    flat = menu_of(os_)
    first = sorted(os_, key=lambda o: core.canonical(CAT, o))[:core.FIRST]
    assert len(flat.options) == core.MENU_MAX and set(first) < set(flat.options)
    assert [o.verb for o in flat.options if o not in first] == ["take", "walk_to", "praise", "wait"]


def test_the_PRE_RANK_keeps_the_rules_mass_first():
    """Step 2 keeps the best options until they hold MASS of the rules'
    probability. Eleven options with equal high scores, and the rest low:
    all eleven are kept, before the null act and the verbs."""
    os_ = _crowd()
    s = [-3.0] * len(os_)
    top = [o for o in os_ if o.verb == "take"][:10] + [opt("praise", target=P("Tom"))]
    for o in top:
        s[os_.index(o)] = 1.0
    m = menu_of(os_, s)
    assert set(top) <= set(m.options)
    # with prior_avg, the option an AVERAGE character wants is kept too
    avg = [-3.0] * len(os_)
    avg[os_.index(opt("search", target=P("Marta")))] = 4.0
    m2 = core.menu(os_, s, avg, [0.0] * len(os_), cat=CAT)
    assert "search Marta" in {o.text for o in m2.options}
    # habit counts once: a strong negative habit pushes an option out
    hab = [0.0] * len(os_)
    hab[os_.index(top[0])] = -9.0
    assert top[0] in menu_of(os_, s).options
    assert top[0] not in core.menu(os_, s, s, hab, cat=CAT).options


def test_HABIT_counts_once():
    """The score is max(prior, prior_avg) + habit. Seventeen takes and the
    null act: the two worst are cut. With habit counted once they are the
    two at -0.1. Counted twice, "take the thing a" (1 - 0.6 - 0.6) would be
    cut; not counted, "take the thing b" (-0.5) would be."""
    things = [f"the thing {chr(97 + k)}" for k in range(17)]
    os_ = [opt("take", target=T(n)) for n in things] + [opt("wait")]
    s, hab = [0.0] * len(os_), [0.0] * len(os_)
    s[0], hab[0] = 1.0, -0.6            # 0.4
    s[1], hab[1] = -0.5, 0.6            # 0.1
    s[2] = s[3] = -0.1
    m = core.menu(os_, s, s, hab, cat=CAT)
    assert set(m.cut) == {os_[2], os_[3]}
    assert m.habit[m.options.index(os_[0])] == -0.6


def test_the_menu_does_not_depend_on_the_ORDER_of_the_options():
    os_ = _crowd()
    rng = random.Random(7)
    s = [round(rng.uniform(-2, 2), 1) for _ in os_]          # many ties
    want = menu_of(os_, s)
    for k in range(5):
        ix = list(range(len(os_)))
        random.Random(k).shuffle(ix)
        assert menu_of([os_[i] for i in ix], [s[i] for i in ix]) == want


def test_a_BLOCKED_option_is_never_offered_and_never_cut():
    axe = opt("take", missing=("a free hand",), target=T("the axe"))
    m = menu_of([opt("wait"), axe, opt("praise", target=P("Oren"))])
    assert m.blocked == (axe,) and axe not in m.options + m.cut + m.full
    assert [o.text for o in m.options] == ["praise Oren", "wait"]


def test_a_PLAYER_menu_is_never_cut():
    os_ = _crowd()
    m = menu_of(os_, player=True)
    assert len(m.options) == len(os_) and m.cut == ()


def test_the_NULL_ACT_is_always_there_or_the_menu_is_refused():
    with pytest.raises(ValueError, match="null act"):
        menu_of([opt("praise", target=P("Oren"))])
    os_ = _crowd()
    s = [1.0] * len(os_)
    s[-1] = -50.0
    assert "wait" in {o.verb for o in menu_of(os_, s).options}


def test_two_options_with_the_SAME_TEXT_are_one_choice():
    a, b = Ref("thing", "e1", "the iron"), Ref("thing", "e2", "the iron")
    one, two = opt("take", target=a), opt("take", target=b)
    assert one.text == two.text and one.key != two.key
    m = menu_of([one, two, opt("wait")], [0.0, 1.0, 0.0])
    assert two in m.options and one in m.cut and one not in m.options
    assert menu_of([opt("wait"), two, one], [0.0, 1.0, 0.0]) == m


def test_a_menu_REFUSES_what_it_cannot_order():
    w = opt("wait")
    with pytest.raises(ValueError, match="one number"):
        core.menu([w], [0.0], [0.0, 1.0], [0.0], cat=CAT)
    with pytest.raises(ValueError, match="same key"):
        menu_of([w, w])
    stray = dataclasses.replace(w, verb="dance", text="dance")
    with pytest.raises(ValueError, match="not in the catalog"):
        menu_of([w, stray])


# ---------------------------------------------------------------- perceive
def test_PERCEIVE_gives_the_last_five_oldest_first():
    ps = [Percept(t, "saw", f"Oren waited {chr(97 + t)}", f"e{t}", "wait", P("Oren"))
          for t in (6, 1, 2, 3, 4, 5, 7)]
    got = core.perceive(ps)
    assert [x.source for x in got] == ["e3", "e4", "e5", "e6", "e7"]
    again = core.perceive(ps + [dataclasses.replace(ps[0], t=9)])          # e6 again, later
    assert [x.source for x in again] == ["e3", "e4", "e5", "e7", "e6"]
    assert core.perceive(ps, k=0) == ()


# ---------------------------------------------------------------- the facts of a scene
def test_the_scene_names_the_PEOPLE_one_can_act_on_and_what_they_hold():
    os_ = [opt("wait"), opt("praise", target=P("Marta")),
           opt("tell", target=P("Marta"), culprit=P("Oren"), thing=T("the purse of silver"))]
    bels = (at("the purse of silver", P("Oren")), at("the ladle", P("Marta")),
            Belief("state", P("Marta"), "lying on the floor", "lying on the floor", "sight", "looked"))
    m = menu_of(os_)
    who, seen = core.people(m, bels)
    # Oren is only named in what is said: he can be away, so he is not "here"
    assert who == (core.Present(P("Marta"), ("the ladle",), ("lying on the floor",)),)
    assert seen == bels[1:]
    # where Oren keeps the purse bears on the "tell" option: the scene states it
    assert core.relevant(FENN, m, bels, seen) == (bels[0],)
    sc = core.scene(FENN, (), bels, m, cat=CAT)
    assert "Here: Marta (lying on the floor, holding the ladle)." in D.describe(sc)


def test_a_thing_that_LIES_HERE_is_never_named():
    """No "Things here": a sentence that names a useless thing makes the
    model pick it (design 1.1)."""
    os_ = [opt("wait"), opt("walk_to", place=Pl("the yard")), opt("take", target=T("the stool"))]
    bels = (at("the stool", Pl("the main room")), at("the table", Pl("the main room")),
            at("the axe", Pl("the yard")), at("the dagger", P("Fenn")))
    m = menu_of(os_)
    assert core.relevant(FENN, m, bels) == ()
    text = D.describe(core.scene(FENN, (), bels, m, cat=CAT))
    assert not re.search(r"stool|table|axe|Things", text), text
    assert text.count("the dagger") == 1 and "Fenn holds the dagger." in text


def test_at_most_TWO_beliefs_the_relevant_and_fresh_first():
    os_ = [opt("wait"), opt("accuse", target=P("Oren"), thing=T("the lute"))]
    bels = (took("the dice", "Tom", t=9), took("the lute", "Oren", t=2), took("the axe", "the bandit", t=5),
            at("the lute", "gone", src="own", how="own", t=0))
    m = menu_of(os_)
    # the two beliefs about the lute bear on the menu (accuse), the fresher first
    assert core.relevant(FENN, m, bels) == (bels[1], bels[3])
    # then the others, the fresher first
    assert core.relevant(FENN, m, bels, k=5) == (bels[1], bels[3], bels[0], bels[2])
    with pytest.raises(ValueError, match="at most 2 beliefs"):
        core.Scene(FENN, (), (), (), bels[:3])
    with pytest.raises(ValueError, match="at most 5 percepts"):
        core.Scene(FENN, (), (), tuple(Percept(t, "saw", "x", f"e{t}") for t in range(6)), ())



def test_its_OWN_DEED_is_not_a_belief_unless_someone_TOLD_it():
    """ "Fenn believes Fenn took the lute" is noise: Fenn did it. Told by
    another, it says what others think, so it stays."""
    os_ = [opt("wait"), opt("accuse", target=P("Oren"), thing=T("the lute"))]
    mine, told = took("the lute", "Fenn", t=9), took("the lute", "Fenn", src="e7", how="told", t=7)
    other = took("the lute", "Oren", t=2)
    m = menu_of(os_)
    assert core.relevant(FENN, m, (mine, other)) == (other,)
    assert core.relevant(FENN, m, (mine, told, other)) == (told, other)

def test_a_NEED_says_what_is_missing():
    w = opt("wait")
    hungry = dataclasses.replace(FENN, drives=("hunger",))
    # nothing here serves it: the words of what is missing, never what is there
    (n,) = core.needs(hungry, menu_of([w]), CAT)
    assert (n.drive, n.says, n.blocked) == ("hunger", ("something to eat",), None)
    assert "Fenn needs something to eat." in D.describe(core.scene(hungry, (), (), menu_of([w]), cat=CAT))
    # a legal way to meet it: no need
    assert core.needs(hungry, menu_of([w, opt("eat", target=T("the loaf of bread"))]), CAT) == ()
    # a blocked way: what it lacks, all of it
    eat = opt("eat", missing=("a free hand",), target=T("the loaf of bread"))
    (n,) = core.needs(hungry, menu_of([w, eat]), CAT)
    assert n.blocked == eat and n.says == ("a free hand",)
    text = D.describe(core.scene(hungry, (), (), menu_of([w, eat]), cat=CAT))
    assert "Fenn needs a free hand to eat the loaf of bread." in text
    # two verbs with one hint give the words once
    thirsty = dataclasses.replace(FENN, drives=("thirst",))
    assert core.needs(thirsty, menu_of([w]), CAT)[0].says == ("something to drink",)
    # no sentence for a drive whose verbs give no words, or that no verb of the backend serves
    assert core.needs(dataclasses.replace(FENN, drives=("safety",)), menu_of([w]), CAT) == ()
    assert core.needs(hungry, menu_of([w]), CAT, backend="voxel") == ()
    with pytest.raises(ValueError, match="not in the catalog"):
        core.needs(dataclasses.replace(FENN, drives=("boredom",)), menu_of([w]), CAT)


# ---------------------------------------------------------------- the words
def test_DESCRIBE_writes_the_v2_spelling():
    """The bytes of the v2 spelling. A change of words is a new SPELLING
    (v1 said "Tom holds nothing."; v2 says what a character holds only when
    it holds something)."""
    assert D.SPELLING == "mark1-mind-v2"
    os_ = [opt("wait"), opt("walk_to", place=Pl("the kitchen")), opt("praise", target=P("Oren")),
           opt("strike", target=P("Oren")), opt("strike", target=P("Oren"), instrument=T("the dagger"))]
    bels = (at("the purse of silver", P("Oren")), at("the stool", Pl("the main room")),
            at("the dagger", P("Fenn")), at("the axe", Pl("the yard")),
            took("the lute", "the bandit", t=2), took("the axe", "Tom", src="e9", how="told", t=8))
    ps = (Percept(3, "saw", "Oren walked to the main room", "e3", "walk_to", P("Oren")),
          Percept(5, "heard", "sounds of a fight in the kitchen", "e5", "strike"))
    sc = core.scene(FENN, ps, bels, menu_of(os_), cat=CAT, backend="rooms")
    assert D.describe(sc, SETTING) == (
        "It is night at the village inn. Fenn is a traveller, in the main room. "
        "Here: Oren (holding the purse of silver). Fenn holds the dagger. "
        "Fenn feels hungry and angry at Oren. Fenn needs something to eat. "
        "Recently: Oren walked to the main room. Fenn heard sounds of a fight in the kitchen. "
        "Fenn was told that Tom took the axe. Fenn believes the bandit took the lute.")
    assert sc.sources == (("e3", "e5"), ("e9", "e1", "sight"))
    assert D.question(FENN) == "What does Fenn do next?"
    bare = SelfState("Tom")
    assert D.describe(core.scene(bare, (), (), menu_of([opt("wait")]), cat=CAT)) == ""
    assert "holds" not in D.describe(core.scene(dataclasses.replace(FENN, holds=()), ps, bels,
                                                menu_of(os_), cat=CAT, backend="rooms"), SETTING)


def test_a_HEARD_percept_names_no_one_and_a_DIGIT_is_refused():
    w = menu_of([opt("wait")])
    heard = Percept(4, "heard", "a shout for help in the yard", "s2", "shout_for_help", phase="start")
    text = D.describe(core.scene(FENN, (heard,), (), w, cat=CAT))
    assert "Recently: Fenn heard a shout for help in the yard." in text
    with pytest.raises(ValueError, match="no numbers"):
        D.describe(core.scene(FENN, (dataclasses.replace(heard, text="a shout 3 m away"),), (), w, cat=CAT))


# ---------------------------------------------------------------- six inn nights
def _night(seed, turns=300, stats=None):
    """One night of the inn: each choice goes through core.menu, core.scene
    and describe, and the lab's rules draw over the menu with an rng of
    their own. Returns the backend and one row per decision."""
    b = RoomsBackend(seed, turns)
    pick = rules.LabRules(random.Random(1000 + seed))
    rows, n = [], 0
    for _ in range(turns):
        b.step()
        while b.needs_decision():
            (who, reason), = b.needs_decision()
            opts = b.bind(who)
            prior, avg, hab = b.rule_scores(who, opts)
            m = core.menu(opts, prior, avg, hab, cat=b.cat)
            sc = core.scene(b.self_state(who), b.percepts(who), b.beliefs(who), m, cat=b.cat, backend=b.name)
            text = D.describe(sc, SETTING)
            n += 1
            req = DecisionRequest(f"{seed}:{n}", who, reason, text, D.question(sc.state),
                                  tuple(o.text for o in m.options), tuple(o.group for o in m.options),
                                  m.prior, m.prior_avg, m.habit)
            dec, pa = pick.draw(req)
            i, pc = dec.pick, dec.probs
            psrc, bsrc = sc.sources
            # the rules' probability, over every legal option, that the cut removed
            u = [a + h for a, h in zip(prior, hab)]
            pf = dict(zip((o.key for o in opts), core.softmax(u, core.T_RULES)))
            room = b.world.people[who].room
            rows.append(dict(who=who, menu=m, scene=sc, text=text,
                             cut=1 - sum(pf[o.key] for o in m.options),
                             loose=tuple(b.world.loose(room))))
            b.start(who, m.options[i], Why(req.id, pick.name, reason, psrc, bsrc, pc[i]),
                    menu=m, probs=pc, probs_avg=pa)
        if b.over():
            break
    return b, rows


@functools.lru_cache(maxsize=1)
def _six_nights():
    return [_night(seed) for seed in range(6)]


def test_SIX_INN_NIGHTS_give_short_canonical_menus():
    rows = [r for _, rs in _six_nights() for r in rs]
    n = len(rows)
    assert n > 1500, n
    cut = sorted(r["cut"] for r in rows)
    first = 0
    for r in rows:
        m = r["menu"]
        texts = [o.text for o in m.options]
        assert len(texts) <= core.MENU_MAX and len(set(texts)) == len(texts)
        assert CAT.null_act.id in {o.verb for o in m.options}
        places = [core.canonical(CAT, o) for o in m.options]
        assert places == sorted(places)
        u = [a + h for a, h in zip(m.prior, m.habit)]
        first += max(range(len(u)), key=u.__getitem__) == 0
    # the inn's menus are long (median 27): the pre-rank is at work in most decisions
    assert sum(len(r["menu"].full) > core.MENU_MAX for r in rows) / n > 0.5
    # Canonical order puts the rules' favourite first far less often than
    # score order does (100%): 8.7% here, and the design measured about 8% (the catalog's
    # groups are sorted by name). With `move` first it was about 28%.
    assert first / n < 0.12, first / n
    # the cut costs the rules less than 1% of their probability, on the mean
    assert sum(cut) / n < 0.01, (sum(cut) / n, cut[int(0.99 * n)], cut[-1])
    # every pick was an option of its menu: start() refuses any other
    for b, rs in _six_nights():
        assert len(b.decisions) == len(rs)
        assert all(d["options"] <= core.MENU_MAX for d in b.decisions)


def test_SIX_INN_NIGHTS_give_clean_scene_texts():
    rows = [r for _, rs in _six_nights() for r in rs]
    needs = collections.Counter()
    for r in rows:
        sc, text, m = r["scene"], r["text"], r["menu"]
        assert "Things here" not in text and not re.search(r"\d", text), text
        assert len(sc.beliefs) <= core.BELIEFS and len(sc.percepts) <= core.PERCEPTS
        # a need is a drive above its threshold that no legal option serves
        for nd in sc.needs:
            needs[nd.drive] += 1
            assert nd.drive in sc.state.drives
            assert not {o.verb for o in m.full} & {v.id for v in CAT.verbs.values() if nd.drive in v.serves}
            assert f"{sc.state.name} needs ".casefold() in text.casefold()     # "The bandit needs"
        # a thing that lies in the room is named only by a percept, a stated
        # belief, or as what someone holds
        said = " ".join([x.text for x in sc.percepts] + [b.text for b in sc.beliefs]
                        + [h for p in sc.people for h in p.holds])
        for thing in r["loose"]:
            assert thing not in text or thing in said, (thing, text)
    assert needs["hunger"] > 0 and needs["thirst"] > 0, needs
    assert sum(1 for r in rows if r["scene"].needs) / len(rows) > 0.1
    assert max(len(r["scene"].beliefs) for r in rows) == core.BELIEFS      # the cap is reached, and holds
    assert max(len(r["scene"].percepts) for r in rows) == core.PERCEPTS


# ---------------------------------------------------------------- the same bytes
def _digest(seeds=(0, 1), turns=150):
    """sha256 of every scene text and every menu of a few short nights."""
    h = hashlib.sha256()
    for seed in seeds:
        for r in _night(seed, turns)[1]:
            m = r["menu"]
            h.update(r["text"].encode())
            for part in (m.options, m.cut, m.blocked):
                h.update("|".join(o.key for o in part).encode() + b"\n")
            h.update(repr((m.prior, m.prior_avg, m.habit)).encode())
    return h.hexdigest()


def test_the_SAME_INPUT_gives_the_SAME_BYTES_in_every_process():
    want = _digest()
    assert _digest() == want
    code = "import tests.test_mind_core as t; print(t._digest())"
    for seed in ("0", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        got = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                             text=True, timeout=300)
        assert got.returncode == 0, got.stderr[-2000:]
        assert got.stdout.strip() == want, seed
