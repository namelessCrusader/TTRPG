"""The rooms backend (docs/sim-mind-design.md 3.5, step M3): the lab inn,
forked into src/rooms, behind the mind interface.

The tests do four things:
- COMPAT. Run through the Backend protocol, with the lab's chooser drawing
  from the world's own rng, the fork writes the lab's events byte for byte.
  The lab's nights come from the inn goldens (lab/world2/golden.py, step M2),
  and their hashes are pinned here.
- THE LAB'S OWN CHECKS (lab/world2/test_world.py:20-61), on the fork.
- START RECORDS are perceived, and they change nothing.
- THE PROTOCOL: what the backend gives the mind, and what `start` checks.

The lab is never imported: src and lab do not import each other, and the
goldens carry the lab's nights into this file.
"""
import ast
import functools
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

import pytest

from src.mind.api import Backend, Menu, Why
from src.rooms import RoomsBackend, World, jsonl, night
from src.rooms import rules, world as inn
from src.rooms.backend import PERCEPTS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
# the inn goldens: sha256 of each file, and its number of events
GOLDEN = {0: ("cb5180f09d62c5d873fc710b280d424b221695939bcf86fce2bb4afeef3043c6", 438),
          1: ("e781efb523b40e3d798c94cdc44e7284395024d5068c706b2b750c8eea1b514c", 425),
          2: ("9a329cd9b149d3ee226b044883f561986774634b94a9c8ec8db346482504c30d", 449)}


def _golden(seed):
    """(header, the text of the events) of one golden file."""
    head, _, body = (DATA / f"inn_rules_seed{seed}.jsonl").read_text().partition("\n")
    return json.loads(head), body


def _same_events(b, seed):
    """The events of backend `b` are the golden's, byte for byte. When they
    are not, the test fails at once and names the first event that differs.
    Do not write `assert text == golden`: pytest's own diff of two texts of
    600 kB takes many minutes, and the lock stays held all that time."""
    got, want = jsonl(b.world.log), _golden(seed)[1]
    if got == want:
        return
    a, g = got.splitlines(), want.splitlines()
    i = next((i for i, (x, y) in enumerate(zip(a, g)) if x != y), min(len(a), len(g)))
    pytest.fail(f"seed {seed}: the events differ from event {i} on ({len(a)} events here, {len(g)} in"
                f" the golden)\n  here:   {a[i][:400] if i < len(a) else '(no event)'}"
                f"\n  golden: {g[i][:400] if i < len(g) else '(no event)'}", pytrace=False)


@functools.lru_cache(maxsize=None)
def _night(seed, turns=300, chooser="rules", bandit=None, until_down=True):
    """A night in compat mode, run once for every test that reads it. A test
    reads it and never changes it."""
    return night(seed, turns, chooser, bandit, until_down)


def _lab_run(chooser, seed, turns):
    """lab/world2/test_world.py `_run`: a bandit comes, and every turn runs."""
    return _night(seed, turns, chooser, True, False).world


# ---------------------------------------------------------------- compat
def test_the_GOLDENS_are_the_pinned_lab_nights():
    for seed, (sha, n) in GOLDEN.items():
        f = DATA / f"inn_rules_seed{seed}.jsonl"
        assert hashlib.sha256(f.read_bytes()).hexdigest() == sha, f.name
        head, body = _golden(seed)
        assert (head["seed"], head["turns"], head["events"], body.count("\n")) == (seed, 300, n, n)
        # the fork is the lab file that made the goldens; else sync it (world.py, header)
        assert head["sha1"] == {"lab/world2/world.py": inn.LAB_SHA1, "lab/world2/policy.py": rules.LAB_SHA1}


@pytest.mark.parametrize("seed", sorted(GOLDEN))
def test_COMPAT_mode_writes_the_lab_s_events_byte_for_byte(seed):
    b = _night(seed)
    _same_events(b, seed)
    # one decision row for each choice; the last ones can still be running
    choices = [e for e in b.world.log if e.get("options")]
    assert len(b.decisions) >= len(choices) == len(set(e["id"] for e in choices))
    assert len(b.decisions) == sum(1 for e in b.job_events if e.phase == "start")


# ---------------------------------------------------------------- the lab's checks
def test_a_THOUSAND_random_turns():
    """lab test_thousand_random_turns: no errors, every chosen action was on
    the list the world gave, and every belief comes from something the
    character saw, heard or was told."""
    w = _lab_run("random", 1, 1000)
    choices = [e for e in w.log if e.get("options")]
    assert len(choices) > 500
    for e in choices:
        assert e["options"][e["chosen"]] == e["text"], e
        assert abs(sum(e["p_char"]) - 1) < 1e-3 and abs(sum(e["p_avg"]) - 1) < 1e-3
    ids = {e["id"] for e in w.log}
    for p in w.people.values():
        for key, (value, src) in p.bel.items():
            # a belief comes from the start, from a look around, from noticing
            # one's own loss, or from an event the character perceived
            assert src in ("start", "sight", "own") or (src in ids and src in p.perceived) \
                or any(e["id"] == src and e["actor"] == p.name for e in w.log), (p.name, key, value, src)


def test_SEEN_is_only_what_was_perceived():
    w = _lab_run("rules", 2, 300)
    for e in w.log:
        p = w.people[e["actor"]]
        for sid in e.get("seen", []):
            assert sid in p.perceived, (e["id"], sid)


def test_the_CHEST_is_decided_once():
    w = _lab_run("random", 3, 1000)
    found = {e.get("found") for e in w.log if e["action"] == "open" and e["done"]}
    assert len(found) <= 1, found


def test_a_LIE_blames_another_for_what_the_teller_holds():
    """A lie is judged by what the teller believes, not by the world: it
    blames someone else, for a thing the teller itself holds."""
    n = 0
    for seed in range(6):
        w = _lab_run("rules", seed, 300)
        for e in w.log:
            f = e.get("facts", {})
            if f.get("lie"):
                n += 1
                assert f["who"] != e["actor"] and f["who"] != e["target"], e
    assert n > 0


# ---------------------------------------------------------------- start records
def _minds(w):
    """Every person, as bytes: equal bytes are equal persons."""
    return pickle.dumps({n: vars(p) for n, p in w.people.items()})


def test_START_RECORDS_never_learn_and_never_draw(monkeypatch):
    """A start record is perceived, and it changes nothing: while it is made,
    _learn is never called, and the rng and every person are the same after
    it as before it. The night with its records is still the lab's night."""
    making, learned, same = [False], [], []
    learn, record = World._learn, RoomsBackend._start_record

    def spy_learn(self, q, ev, actor):
        learned.append(making[0])
        return learn(self, q, ev, actor)

    def spy_record(self, *a):
        w = self.world
        before = (w.rng.getstate(), _minds(w), len(w.log))
        making[0] = True
        try:
            record(self, *a)
        finally:
            making[0] = False
        same.append(before == (w.rng.getstate(), _minds(w), len(w.log)))

    monkeypatch.setattr(World, "_learn", spy_learn)
    monkeypatch.setattr(RoomsBackend, "_start_record", spy_record)
    b = night(2)
    assert learned and True not in learned
    assert len(same) == len(b.starts) > 50 and all(same)
    _same_events(b, 2)
    # a record for each act of a loud or violent verb as it starts, and no other
    for s in b.starts:
        assert {"loud", "violent"} & set(b.cat.verbs[s["verb"]].flags), s
        assert s["action"] in inn.LOUD | inn.VIOLENT and s["actor"] not in s["saw"] + s["heard"]
    assert sum(1 for e in b.job_events if e.phase == "start" and e.text) == len(b.starts)
    assert any(s["saw"] for s in b.starts) and any(s["heard"] for s in b.starts)


# ---------------------------------------------------------------- the protocol
def _first_choice(seed=0):
    b = RoomsBackend(seed)
    while not b.needs_decision():
        b.step()
    return b, b.needs_decision()[0][0]


def test_the_backend_is_a_BACKEND_of_the_mind():
    b, who = _first_choice()
    assert isinstance(b, Backend) and b.name == "rooms"
    assert b.needs_decision() == [(who, "idle")]
    w, p = b.world, b.world.people[who]
    opts = b.bind(who)
    rows = w.options(p)
    assert [o.text for o in opts] == [r["text"] for r in rows]          # the inn's words, in its order
    assert len({o.key for o in opts}) == len(opts)
    runs = set(b.cat.runnable("rooms"))
    for o, r in zip(opts, rows):
        assert o.verb in runs and o.ok and not o.missing and not o.features
        assert o.duration == inn.DUR[r["verb"]] and b.cat.program(o.verb, "rooms")["verb"] == r["verb"]
    assert "wait" in {o.verb for o in opts}
    prior, avg, hab = b.rule_scores(who, opts)
    assert [a + h for a, h in zip(prior, hab)] == [rules.utility(w, p, r, p.traits) for r in rows]
    assert [a + h for a, h in zip(avg, hab)] == [rules.utility(w, p, r, rules.MID) for r in rows]
    # a part of the bind is scored the same, by the options' keys
    part = opts[::3]
    assert b.rule_scores(who, part) == tuple(x[::3] for x in (prior, avg, hab))


def _why(who, reason="idle"):
    return Why(f"test:{who}", "test", reason)


def test_START_checks_the_option_again():
    b, who = _first_choice()
    opts = b.bind(who)
    other = next(n for n in b.world.people if n != who)
    with pytest.raises(ValueError, match="must choose now"):
        b.start(other, opts[0], _why(other))
    stray = b.cat.option("rooms", "take", {"target": b._ref("thing", "the axe")})   # the axe is in the yard
    assert stray.text not in {o.text for o in opts}
    with pytest.raises(ValueError, match="not an option"):
        b.start(who, stray, _why(who))
    with pytest.raises(ValueError, match="not an option"):
        b.rule_scores(who, (stray,))
    with pytest.raises(RuntimeError, match="must choose"):
        b.step()
    wait = next(o for o in opts if o.verb == "wait")
    job = b.start(who, wait, _why(who))
    assert job.who == who and job.option is wait and job.started == b.world.t
    # the world moved on: the old bind no longer holds
    with pytest.raises(ValueError, match="not from a bind"):
        b.rule_scores(who, opts)


def test_a_MENU_is_what_the_log_keeps():
    """With a Menu, the log keeps the menu's texts, and the draws over it."""
    b, who = _first_choice(1)
    opts = b.bind(who)
    prior, avg, hab = b.rule_scores(who, opts)
    keep = [i for i, o in enumerate(opts) if o.verb in ("wait", "walk_to")]
    menu = Menu(tuple(opts[i] for i in keep), (), opts, tuple(o for i, o in enumerate(opts) if i not in keep),
                tuple(prior[i] for i in keep), tuple(avg[i] for i in keep), tuple(hab[i] for i in keep))
    wait = next(o for o in menu.options if o.verb == "wait")
    b.start(who, wait, _why(who), menu=menu)
    p = b.world.people[who]
    choice = p.doing["choice"] if p.doing else next(e for e in reversed(b.world.log) if e["actor"] == who)
    assert choice["options"] == [o.text for o in menu.options]
    assert choice["chosen"] == list(menu.options).index(wait)
    pc = rules.softmax([prior[i] + hab[i] for i in keep], rules.T_RULES)
    assert choice["p_char"] == [round(x, 5) for x in pc]
    # a menu with an option the inn does not permit now is refused
    b2, who2 = _first_choice(1)
    opts2 = b2.bind(who2)
    stray = b2.cat.option("rooms", "take", {"target": b2._ref("thing", "the axe")})
    bad = Menu((opts2[0], stray), (), (opts2[0], stray), (), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    with pytest.raises(ValueError, match="does not permit"):
        b2.start(who2, opts2[0], _why(who2), menu=bad)


def test_a_JOB_ends_as_its_act_ends():
    b = _night(0)
    ends = [e for e in b.job_events if e.phase != "start"]
    assert {e.phase for e in ends} <= {"done", "failed", "stopped"}
    # each end the log keeps is one event of a choice; an act dropped when its
    # person goes down has no event
    logged = [e for e in ends if e.reason != "down"]
    assert len(logged) == sum(1 for e in b.world.log if e.get("options"))
    running = [j for j in b.jobs.values() if j.status == "running"]
    assert len(b.decisions) == len(ends) + len(running)
    for e in ends:
        assert (e.text.startswith(e.who) if e.phase == "done" else e.text == ""), e
    # the reason of a decision is how the person's last job ended
    ended, last = {e.job: e for e in ends}, {}
    for d in b.decisions:
        assert d["reason"] == last.get(d["who"], "idle"), d
        e = ended.get(d["decision"])
        if e is not None:
            last[d["who"]] = ("done" if e.phase == "done" else f"failed:{e.reason}" if e.phase == "failed"
                              else e.reason if e.reason.startswith("interrupt:") else f"interrupt:{e.reason}")
    # every row holds the Why of its decision, and is JSON
    rows = [json.loads(x) for x in jsonl(b.decisions).splitlines()]
    assert len(rows) == len(b.decisions) and all(r["chooser"] == "rules" for r in rows)
    assert {"t", "who", "verb", "text", "options", "decision", "chooser", "reason", "percepts",
            "beliefs", "score", "margin", "fallback"} == set(rows[0])


def test_PERCEPTS_and_BELIEFS_come_from_what_was_perceived():
    for b in (_night(0), _night(1, 1000, "random", True, False)):
        w = b.world
        names = set(w.people)
        for who, p in w.people.items():
            seen = {s["id"] for s in b.starts if who in s["saw"] + s["heard"]}
            pcs = b.percepts(who)
            assert len(pcs) <= PERCEPTS
            assert [x.t for x in pcs] == sorted(x.t for x in pcs)
            for x in pcs:
                assert x.source in p.perceived or x.source in seen, (who, x)
                assert (x.phase == "start") == x.source.startswith("s")
                if x.channel == "heard":
                    # a sound does not say who made it
                    assert x.actor is None and x.target is None and not any(n in x.text for n in names), x
                else:
                    assert x.actor.name in x.text, x
            ids = {e["id"] for e in w.log}
            for bel in b.beliefs(who):
                src = bel.source
                assert src in ("start", "sight", "own") or (src in ids and src in p.perceived) \
                    or any(e["id"] == src and e["actor"] == who for e in w.log), (who, bel)
                assert bel.kind in ("at", "took") and bel.about.name in bel.text
                assert (bel.how == "told") == (src in ids and w.log[int(src[1:])]["action"] == "tell")
            s = b.self_state(who)
            assert (s.name, s.place, s.holds) == (who, p.room, tuple(w.held(p)))
            assert ("hunger" in s.drives) == (p.hunger > 0.6) and ("safety" in s.drives) == (p.fear > 0.4)


# ---------------------------------------------------------------- imports
def test_src_rooms_IMPORTS_the_mind_and_no_lab_no_voxel():
    """src/rooms reads src/mind and the standard library. lab, src/voxel,
    src/mpm and src/core are never imported, and src/mind never imports
    src/rooms."""
    def mods(f):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                yield from (a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                yield node.module

    for f in sorted((ROOT / "src" / "rooms").glob("*.py")):
        for m in mods(f):
            assert m.split(".")[0] in sys.stdlib_module_names or m == "src.mind" \
                or m.startswith("src.mind."), (f.name, m)
        # the design's text check (step M1): `grep -rn "import lab\|from lab" src/`
        # finds nothing, in a comment either
        assert not re.search(r"import lab|from lab", f.read_text(encoding="utf-8")), f.name
    for f in sorted((ROOT / "src" / "mind").glob("*.py")):
        assert not any(m.startswith("src.rooms") for m in mods(f)), f.name
