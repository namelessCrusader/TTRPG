"""New in V4 (docs/sim-mind-design.md, 3.4 and 4.4): beliefs.

Perception writes what reached each body's senses; beliefs.py turns those
rows into what each person holds to be so, `Mind.bel[key] = (value, ev_id,
how, tick)`, and is the only thing that writes them. What must hold: a
belief comes only from what the senses could tell (a sound behind a wall
names no one), it keeps the event it came from, a thing not looked at stays
believed where it was last seen, what you were told never undoes a fresher
thing you saw, and the two kinds of WRONG belief -- a misread, a claim --
are off unless switched on, replay exactly, and leave the truth in the log.
"""
import os
import subprocess
import sys

import numpy as np

from src.voxel.beliefs import BEL_CAP
from src.voxel.events import hash01
from src.voxel.sim import AIR, IRON, LEAD, LEAF, STONE, WOOD, World
from src.voxel.scenes import _person, _Wants, thing

STAY = {"legs": "stay", "waist": "stand"}


def _bel(w, p, key):
    return w.minds[p["pid"]].bel.get(key)


def _room(nx=40, ny=40, nz=30):
    w = World(nx, ny, nz, voxel_cm=5)
    w.fill(0, nx, 0, ny, 0, 1, STONE)
    w.exits = []
    return w


def _look(p, w, fx, fy):
    """Hold the eyes on one direction: a chosen look wins over the sweep
    (`_law_will`), as long as it is fresh."""
    p["look"] = (fx, fy, w.tick)


# ── a thrown hit, three ways ────────────────────────────────────────────────

def test_a_THROWN_HIT_is_FELT_by_the_victim_SEEN_by_a_witness_HEARD_behind_a_screen():
    """One man throws an iron stone at another (test_voxel `_pitch`). The
    victim feels it; a witness across the hall, who has both men in view,
    sees it and believes the THROWER hurt him (`hurt_by`); a man behind a
    hedge beside them hears the thud and nothing more -- the row names no
    one and he believes nothing about who hurt whom.

    The thud of iron on flesh is quiet (61 dB at a metre: flesh does not
    ring) and the lightest screen that blinds, a hedge, takes 26 dB of it,
    so he hears it at 35.6 dB, just over the 35 dB a quiet room's noise is
    noticed at."""
    w = World(100, 50, 50, voxel_cm=5)
    w.open_sky = False
    w.fill(0, 100, 0, 50, 0, 50, STONE)
    w.mat[1:99, 1:49, 1:46] = AIR
    w.smass[1:99, 1:49, 1:46] = 0.0
    w.fill(5, 38, 22, 23, 1, 46, LEAF, frac=0.87)    # a hedge: blind, not deaf
                                                     # (long enough that the
                                                     # wider view, R3, does not
                                                     # see round its end)
    w.exits = []
    a = _person(w, 10, 17)
    a["name"], a["facing"], a["strength_N"] = "thrower", (1.0, 0.0), 3000.0
    b = _person(w, 40, 17)
    b["name"], b["facing"], b["strength_N"] = "victim", (-1.0, 0.0), 3000.0
    from tests.test_exertion import stock_arm
    stock_arm(a), stock_arm(b)        # a load knob of 3000 N, the stock adult's arm
    c = _person(w, 25, 4)
    c["name"], c["facing"] = "witness", (0.0, 1.0)
    d = _person(w, 30, 28)
    d["name"], d["facing"] = "listener", (0.0, -1.0)
    w.fill(14, 15, 16, 18, 1, 3, IRON)
    want = {"thrower": {"hands": "take hold of the iron", **STAY},
            "victim": {"hands": "hands free", **STAY},
            "witness": STAY, "listener": STAY}
    w.policy = _Wants(each=want)
    for t in range(100):
        if t == 30:
            want["thrower"]["hands"] = "throw the iron"
        _look(c, w, 0.0, 1.0)
        _look(d, w, 0.0, -1.0)
        w.step()
    hits = [e for e in w.events.rows if e.verb == "hits"
            and e.target == ("person", b["pid"])]
    assert len(hits) == 1 and hits[0].actor == a["pid"], hits
    hit = hits[0].id

    def rows(p):
        return [r for r in w.minds[p["pid"]].perceived if r.ev == hit]

    assert [r.how for r in rows(b) if r.how == "felt"], "the victim felt it"
    seen = [r for r in rows(c) if r.how == "saw"]
    assert seen and seen[0].actor_seen == a["pid"], rows(c)
    assert _bel(w, c, ("hurt_by", b["pid"]))[:3] == (a["pid"], hit, "saw")
    heard = [r for r in rows(d) if r.how == "heard"]
    assert heard and all(r.actor_seen is None for r in heard), rows(d)
    assert not [r for r in rows(d) if r.how == "saw"], "the hedge hides it"
    assert _bel(w, d, ("hurt_by", b["pid"])) is None
    assert ("at", a["pid"]) not in w.minds[d["pid"]].bel


# ── told against saw ────────────────────────────────────────────────────────

def test_TOLD_never_overrides_a_FRESHER_SAW():
    """The rule itself (`_believe`): a told belief as of a tick no later
    than what the body saw for itself is refused; a told belief fresher
    than it is taken; anything seen afterwards replaces a told one."""
    w = _room()
    p = _person(w, 20, 20)
    k = ("at", 7)
    w.tick = 10
    assert w._believe(p["pid"], k, (1, 1, 1), 3, "saw")
    w.tick = 20
    assert not w._believe(p["pid"], k, (2, 2, 2), 4, "told", asof=9)
    assert not w._believe(p["pid"], k, (2, 2, 2), 4, "told", asof=10)
    assert _bel(w, p, k) == ((1, 1, 1), 3, "saw", 10)
    assert w._believe(p["pid"], k, (2, 2, 2), 4, "told", asof=11)
    assert _bel(w, p, k) == ((2, 2, 2), 4, "told", 20)
    assert w._believe(p["pid"], k, (3, 3, 3), 5, "told", asof=12), \
        "told over told is just the newer word"
    w.tick = 21
    assert w._believe(p["pid"], k, (4, 4, 4), 6, "looked")
    assert _bel(w, p, k)[2] == "looked"


def _fire_room():
    """A crib alight; a witness 0.6 m off facing it, a second man beside
    him also facing it, and a third facing the other way."""
    w = World(100, 60, 40, voxel_cm=5)
    w.fill(0, 100, 0, 60, 0, 1, STONE)
    w.fill(26, 32, 26, 32, 1, 6, WOOD, frac=0.7)
    w.E[31, 29, 5] = 9.0e5
    w.exits = []
    wit = _person(w, 44, 29)
    wit["name"], wit["facing"] = "witness", (-1.0, 0.0)
    two = _person(w, 44, 37)
    two["name"], two["facing"] = "beside", (-1.0, 0.0)
    away = _person(w, 52, 29)
    away["name"], away["facing"] = "away", (1.0, 0.0)
    w.policy = _Wants(**STAY)
    w.claims = True
    return w, wit, two, away


def _fires_believed(w, p):
    return {k: b for k, b in w.minds[p["pid"]].bel.items()
            if k[0] == "alight" and b[0] is not None}


def _step_looking(w, people, n):
    for _ in range(n):
        for p, (fx, fy) in people:
            _look(p, w, fx, fy)
        w.step()


def test_an_HONEST_WITNESS_is_GROUNDED_and_the_one_who_SAW_keeps_his_own_word():
    """The witness sees the fire and shouts it. His `says` is a claim that
    is TRUE and GROUNDED in his own eyes. The man facing away believes him:
    the same fire, as of when the witness saw it, `told`. The man beside
    him saw it himself at least as freshly, and keeps what he saw."""
    w, wit, two, away = _fire_room()
    look = [(wit, (-1.0, 0.0)), (two, (-1.0, 0.0)), (away, (1.0, 0.0))]
    for _ in range(60):
        _step_looking(w, look, 1)
        if _fires_believed(w, wit) and _fires_believed(w, two):
            break
    mine = _fires_believed(w, wit)
    assert mine and all(b[2] in ("saw", "looked") for b in mine.values()), mine
    assert not _fires_believed(w, away)
    w._say(wit, "fire")
    said = w.events.rows[-1]
    _step_looking(w, look, 2)
    ev = w.events.get(said.id)
    assert ev.verb == "says" and dict(ev.data) == {"grounded": True,
                                                   "true": True}, ev.data
    key = next(iter(mine))
    got = _bel(w, away, key)
    assert got is not None and got[1:3] == (said.id, "told"), got
    assert got[0] == w.minds[wit["pid"]].bel[key][0], "where HE saw it"
    kept = _bel(w, two, key)
    assert kept is not None and kept[2] in ("saw", "looked"), kept


def test_a_SCRIPTED_LIAR_is_UNGROUNDED_and_UNTRUE_and_INSIGHT_sees_through_him():
    """No fire anywhere; a man shouts "Fire!" all the same. The claim is
    neither grounded nor true -- the log says so -- and is believed anyway
    by a listener with no insight (a vague fire, where the voice came
    from), but not by one whose insight sees through every ungrounded
    claim. With claims off (the default) the shout teaches nobody a fire
    and the event is left as it was."""
    for on in (True, False):
        w = _room()
        liar = _person(w, 10, 20)
        liar["name"] = "liar"
        dupe = _person(w, 22, 20)
        dupe["name"] = "dupe"
        sharp = _person(w, 14, 28)
        sharp["name"], sharp["insight"] = "sharp", 1.0
        w.policy = _Wants(**STAY)
        assert w.claims is False and w.misread is False, "both off by default"
        w.claims = on
        w.step()
        w._say(liar, "fire")
        said = w.events.rows[-1]
        if on:     # settled as it is said, before anyone has heard it
            assert dict(said.data) == {"grounded": False, "true": False}
        w.step()
        w.step()
        ev = w.events.get(said.id)
        assert ev is said, "the row in the log is never rewritten"
        heard = [r for r in w.minds[dupe["pid"]].perceived
                 if r.ev == said.id and r.how == "heard"]
        assert heard and dict(heard[0].data).get("line") == "fire"
        assert [r for r in w.minds[sharp["pid"]].perceived
                if r.ev == said.id and r.how == "heard"]
        if not on:
            assert ev.data == () and not _fires_believed(w, dupe)
            continue
        assert dict(ev.data) == {"grounded": False, "true": False}
        got = _bel(w, dupe, ("alight", None))
        assert got is not None and got[1:3] == (said.id, "told"), got
        assert _bel(w, sharp, ("alight", None)) is None


# ── things ──────────────────────────────────────────────────────────────────

def test_a_THING_MOVED_while_you_LOOKED_AWAY_is_still_believed_at_its_OLD_PLACE():
    """A named box in front of a man is believed where it is. He looks the
    other way; the box is shoved half a metre; he still believes it is
    where he saw it, for as long as he does not look. When he looks back
    he sees where it is now."""
    w = _room(60, 60, 30)
    p = _person(w, 20, 30)
    p["name"], p["facing"] = "p", (1.0, 0.0)
    with thing(w, "box") as e:
        w.fill(30, 33, 29, 32, 1, 3, WOOD)
    w.policy = _Wants(**STAY)
    key = ("at", ("thing", e))
    _step_looking(w, [(p, (1.0, 0.0))], 12)
    old = _bel(w, p, key)
    assert old is not None and old[2] == "looked", w.minds[p["pid"]].bel
    mid0 = tuple(int(round(float(c))) for c in w.cells_of(e).mean(axis=0))
    assert old[0] == mid0
    _step_looking(w, [(p, (-1.0, 0.0))], 8)
    assert w._shove(w.cells_of(e), 0, 10)
    mid1 = tuple(int(round(float(c))) for c in w.cells_of(e).mean(axis=0))
    assert mid1 != mid0
    _step_looking(w, [(p, (-1.0, 0.0))], 16)
    assert _bel(w, p, key)[0] == mid0, "not looked at, so where it was"
    _step_looking(w, [(p, (1.0, 0.0))], 12)
    assert _bel(w, p, key)[0] == mid1, "looked at again, where it is"


def test_WHAT_IS_WITHIN_REACH_says_SHARP_HOLDS_LIQUID_BURNING_LIFTABLE():
    """`has_within_reach(p, prop)`, read off each thing's own cells: an
    edged iron head is sharp, a hollow wooden cup can hold a liquid, a hot
    plank is burning, a small block can be lifted and a big lead one
    cannot. Out of reach, nothing counts."""
    def one(make, near=True):
        w = _room(70, 40, 30)
        p = _person(w, 20, 20)
        w.policy = _Wants(**STAY)
        w.step()
        x = 24 if near else 60
        with thing(w, "it") as e:
            make(w, x)
        return w, p, e

    w, p, _e = one(lambda w, x: w.fill(x, x + 1, 19, 21, 1, 2, IRON, edge=2e-4))
    assert w.has_within_reach(p, "sharp")
    assert not w.has_within_reach(p, "holds_liquid")
    w, p, _e = one(lambda w, x: w.fill(x, x + 1, 19, 21, 1, 2, IRON, edge=2e-4),
                   near=False)
    assert w.within_reach(p) == [] and not w.has_within_reach(p, "sharp")

    def cup(w, x):
        w.fill(x, x + 3, 19, 22, 1, 3, WOOD)
        w.mat[x + 1, 20, 2] = AIR
        w.smass[x + 1, 20, 2] = 0.0
    w, p, _e = one(cup)
    assert w.has_within_reach(p, "holds_liquid")
    assert not w.has_within_reach(p, "sharp")

    def plank(w, x):
        w.fill(x, x + 1, 18, 22, 1, 2, WOOD)
    w, p, e = one(plank)
    assert not w.has_within_reach(p, "burning")
    w.E[tuple(w.cells_of(e).T)] = 9.0e5
    assert w.has_within_reach(p, "burning")
    assert w.has_within_reach(p, "liftable")
    w, p, _e = one(lambda w, x: w.fill(x, x + 6, 16, 24, 1, 6, LEAD))
    assert not w.has_within_reach(p, "liftable")


# ── the cap ─────────────────────────────────────────────────────────────────

def test_the_KEY_COUNT_stays_within_64_over_2000_TICKS():
    """Three people stand in a yard where named stones keep coming and
    going: every 20 ticks the eight stones there are gone ("dissolves") and
    eight new ones lie somewhere else, for 2,000 ticks. Each stone looked
    at is a belief of its own, so far more than BEL_CAP keys come and go.
    Each mind holds at most BEL_CAP of them at every tick, the cap really
    bites, and a voice is never the one forgotten."""
    w = _room(50, 50, 30)
    people = [_person(w, x, y) for x, y in ((15, 15), (35, 18), (25, 35))]
    for i, q in enumerate(people):
        q["name"] = f"p{i}"
    w.policy = _Wants(**STAY)
    w.step()
    for q in people:                                  # a voice each knows
        w._believe(q["pid"], ("voice", 99), True, None, "heard")
    rng = np.random.default_rng(5)
    seen = [set() for _ in people]
    most = 0
    for t in range(2000):
        if t % 20 == 0:
            w._unmake(np.isin(w.eid, [e for e, d in w.things.items()
                                      if d["kind"] != "person"]), "dissolves")
            made = 0
            while made < 8:
                x, y = (int(v) for v in rng.integers(3, 47, 2))
                if (w.mat[x, y, 1:3] == AIR).all() and all(
                        max(abs(x - q["anchor"][0]), abs(y - q["anchor"][1]))
                        > 4 for q in people):
                    with thing(w, "stone"):
                        w.fill(x, x + 1, y, y + 1, 1, 2, STONE)
                    made += 1
        w.step()
        for i, q in enumerate(people):
            bel = w.minds[q["pid"]].bel
            assert len(bel) <= BEL_CAP, (w.tick, len(bel))
            assert ("voice", 99) in bel, (w.tick, q["name"])
            seen[i] |= set(bel)
            most = max(most, len(bel))
    assert most == BEL_CAP, most
    assert min(len(s) for s in seen) > 2 * BEL_CAP, [len(s) for s in seen]


# ── misreads ────────────────────────────────────────────────────────────────

def _misread_room(seed):
    w = _room()
    v = _person(w, 20, 8)
    v["name"], v["facing"] = "viewer", (0.0, 1.0)
    a = _person(w, 12, 30)
    b = _person(w, 28, 30)
    c = _person(w, 20, 34)
    a["name"], b["name"], c["name"] = "a", "b", "c"
    w.policy = _Wants(**STAY)
    w.seed = seed
    w.misread = {"hits": 0.5}
    return w, v, a, b


def _blows(w, v, a, b, n):
    """`a` hits `b` every tick for `n` ticks, in the viewer's sight; what
    the viewer says he saw of each blow."""
    out = []
    for _ in range(n):
        ev = w.emit("hits", actor=a["pid"], target=("person", b["pid"]))
        _look(v, w, 0.0, 1.0)
        w.step()
        out += [(r.tick, r.ev, r.actor_seen, r.target_seen)
                for r in w.minds[v["pid"]].perceived
                if r.ev == ev.id and r.how == "saw"]
    return out


def test_the_SAME_SEED_gives_the_SAME_MISREADS_across_a_SNAPSHOT():
    """A viewer watches `a` strike `b` again and again, with misreads on.
    Some blows he puts on someone else; the row and the belief still name
    the true event, whose actor is `a`. Restored from a snapshot, the same
    ticks are misread the same way; another seed misreads others."""
    w, v, a, b = _misread_room(seed=11)
    _blows(w, v, a, b, 6)
    snap = w.snapshot()
    first = _blows(w, v, a, b, 30)
    assert len(first) >= 25, first
    wrong = [r for r in first if r[2] != a["pid"]]
    assert wrong and len(wrong) < len(first), first
    for _t, ev_id, who, _tg in wrong:
        assert w.events.get(ev_id).actor == a["pid"], "the log keeps the truth"
        assert who not in (v["pid"], a["pid"], b["pid"]), \
            "never the viewer, the doer, or the one the blow was done to"
    hb = _bel(w, v, ("hurt_by", b["pid"]))
    assert hb is not None and w.events.get(hb[1]).actor == a["pid"]
    assert hb[0] == first[-1][2], "the belief is what he last saw"
    w.restore(snap)
    v, a, b = (w.persons[q["pid"]] for q in (v, a, b))   # the restored ones
    again = _blows(w, v, a, b, 30)
    assert [r[2] for r in again] == [r[2] for r in first]
    w2, v2, a2, b2 = _misread_room(seed=12)
    _blows(w2, v2, a2, b2, 6)
    other = _blows(w2, v2, a2, b2, 30)
    assert [r[2] for r in other] != [r[2] for r in first]
    w3, v3, a3, b3 = _misread_room(seed=11)
    w3.misread = False
    assert all(r[2] == a3["pid"] for r in _blows(w3, v3, a3, b3, 20))


def test_HASH01_is_the_same_in_ANOTHER_PROCESS():
    """A misread draw must replay in a process with another string-hash
    salt, or a harvest could not be replayed anywhere else."""
    parts = (7, 3, 120, "hits", 2, (20, 30, 31))
    code = ("from src.voxel.events import hash01; "
            f"print(repr(hash01(*{parts!r})))")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outs = set()
    for salt in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=salt)
        outs.add(subprocess.run([sys.executable, "-c", code], cwd=root,
                                env=env, capture_output=True, text=True,
                                check=True).stdout.strip())
    assert outs == {repr(hash01(*parts))}
    assert 0.0 <= hash01(*parts) < 1.0


def test_a_MISREAD_SPEAKER_is_the_one_whose_VOICE_and_WORDS_are_learned():
    """With every sighting of a `says` misread, the viewer's eyes put a's
    shout in someone else's mouth. The voice and the words he learns are
    that someone's -- the one his eyes named -- never a's, which he was not
    shown; the log still says a spoke. The same holds when the words reach
    him after the sight of them (`_learn`'s heard branch)."""
    w, v, a, b = _misread_room(seed=11)
    w.misread = {"says": 1.0}
    _look(v, w, 0.0, 1.0)
    w.step()
    w._say(a, "fire")
    said = w.events.rows[-1]
    for _ in range(2):
        _look(v, w, 0.0, 1.0)
        w.step()
    m = w.minds[v["pid"]]
    saw = [r for r in m.perceived if r.ev == said.id and r.how == "saw"]
    assert saw and [r for r in m.perceived
                    if r.ev == said.id and r.how == "heard"], m.perceived
    who = saw[0].actor_seen
    assert who is not None and who not in (v["pid"], a["pid"]), who
    assert ("voice", a["pid"]) not in m.bel and ("said", a["pid"]) not in m.bel
    assert m.bel[("voice", who)][1] == said.id
    assert m.bel[("said", who)][:2] == ("fire", said.id)
    assert w.events.get(said.id).actor == a["pid"], "the log keeps the truth"
    # the other order: seen first, heard after
    w2, v2, a2, b2 = _misread_room(seed=11)
    ev = w2.emit("says", actor=a2["pid"], result="fire")
    c2 = next(q["pid"] for q in w2.persons
              if q["pid"] not in (v2["pid"], a2["pid"], b2["pid"]))
    w2._note(v2["pid"], "saw", ev.id, verb="says", actor_seen=c2,
             place=(12, 30, 5))
    w2._learn(v2["pid"], "saw", ev, c2, (12, 30, 5))
    assert not [k for k in w2.minds[v2["pid"]].bel
                if k[0] in ("voice", "said")], "words not heard teach nothing"
    w2._learn(v2["pid"], "heard", ev, None, (12, 30, 5))
    bel = w2.minds[v2["pid"]].bel
    assert ("voice", c2) in bel and bel[("said", c2)][0] == "fire"
    assert ("voice", a2["pid"]) not in bel and ("said", a2["pid"]) not in bel


def test_a_LINE_SPOKEN_while_CLAIMS_were_OFF_claims_nothing_later():
    """A claim is settled as it is spoken. A shout of "Fire!" made while
    claims are off is left as it was, and switching claims on before it is
    heard does not make it one: nobody is told a fire."""
    w = _room()
    liar = _person(w, 10, 20)
    dupe = _person(w, 22, 20)
    w.policy = _Wants(**STAY)
    w.step()
    w._say(liar, "fire")
    said = w.events.rows[-1]
    w.claims = True
    w.step()
    w.step()
    assert [r for r in w.minds[dupe["pid"]].perceived
            if r.ev == said.id and r.how == "heard"]
    assert w.events.get(said.id).data == ()
    assert not _fires_believed(w, dupe)
    assert _bel(w, dupe, ("alight", None)) is None


def test_a_LINE_DROWNED_by_the_ROOM_is_HEARD_but_NOT_MADE_OUT():
    """A shout half a metre away in a deafening room: loud enough to be
    noticed, but the room's noise (`w.ambient_db`) drowns the words, as
    `_hears` says. The heard row carries no line, and the listener learns
    no words and is told nothing. In a quiet room the same shout is made
    out."""
    for ambient, made_out in ((None, True), (130.0, False)):
        w = _room()
        liar = _person(w, 10, 20)
        dupe = _person(w, 20, 20)
        w.policy = _Wants(**STAY)
        w.claims = True
        if ambient is not None:
            w.ambient_db = ambient
        w.step()
        w._say(liar, "fire")
        said = w.events.rows[-1]
        w.step()
        w.step()
        assert w._hears(liar["_eye"], dupe["_eye"]) is made_out
        heard = [r for r in w.minds[dupe["pid"]].perceived
                 if r.ev == said.id and r.how == "heard"]
        assert heard, "loud enough to be noticed either way"
        assert ("line" in dict(heard[0].data)) is made_out, heard[0]
        assert (_bel(w, dupe, ("alight", None)) is not None) is made_out
