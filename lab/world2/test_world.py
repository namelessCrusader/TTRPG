"""The plan's check for the lab world (Workstream E): 1,000 turns with random
choices give no errors, every chosen action was on the list the world gave,
and every belief comes from something the character saw, heard or was told.

The learned-mind world changes (docs/learned-mind-plan.md, section 6) have a
test each below the line: with every flag off the inn is byte for byte what
it was; then drives, the lifetime, gossip and reputation, trade and refusal,
watching, checking lies, and the scripted partners.

    lab/.venv/bin/python -m lab.world2.test_world
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from lab.world2 import partners as P
from lab.world2.lifetime import NAMES, Cast, lifetime
from lab.world2.policy import Random, Rules, utility
from lab.world2.world import DRIVES, FLAGS, NATURE, World


def _run(policy, seed, turns):
    w = World(seed=seed, turns=turns, bandit=True)
    for _ in range(turns):
        w.step(policy)
    return w


def test_thousand_random_turns():
    w = _run(Random(), 1, 1000)
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


def test_seen_is_only_what_was_perceived():
    w = _run(Rules(), 2, 300)
    for e in w.log:
        p = w.people[e["actor"]]
        for sid in e.get("seen", []):
            assert sid in p.perceived, (e["id"], sid)


def test_the_chest_is_decided_once():
    w = _run(Random(), 3, 1000)
    found = {e.get("found") for e in w.log if e["action"] == "open" and e["done"]}
    assert len(found) <= 1, found


def test_a_lie_blames_another_for_what_the_teller_holds():
    """A lie is judged by what the teller believes, not by the world: it
    blames someone else, for a thing the teller itself holds."""
    n = 0
    for seed in range(6):
        w = _run(Rules(), seed, 300)
        for e in w.log:
            f = e.get("facts", {})
            if f.get("lie"):
                n += 1
                assert f["who"] != e["actor"] and f["who"] != e["target"], e
    assert n > 0


# ------------------------------------------------------------------ the learned-mind world changes
BASELINE = Path(__file__).resolve().parents[1] / "results" / "mindL" / "world_flags_off.json"


def _night(policy, seed, turns=300, bandit=None, **kw):
    w = World(seed=seed, turns=turns, bandit=bandit, **kw)
    for _ in range(turns):
        w.step(policy)
        if all(p.dead or p.down for p in w.people.values()):
            break
    return w


def _act(w, name, text):
    """Make `name` do the listed option `text` now (it must be on the menu)."""
    p = w.people[name]
    opts = w.options(p)
    o = next((o for o in opts if o["text"] == text), None)
    assert o is not None, (text, [x["text"] for x in opts])
    p.doing = {**o, "left": 0, "choice": {}}
    w._finish(p)
    return next(e for e in reversed(w.log) if e["actor"] == name and e["text"] == text)


def _texts(w, name):
    return [o["text"] for o in w.options(w.people[name])]


def _quiet(w):
    """No bandit, everyone in the main room, nothing under way."""
    w.bandit_at = None
    for p in w.people.values():
        p.room, p.doing, p.pending = "the main room", None, []


def test_flags_off_is_the_inn_as_before():
    """Recorded before any change (2026-10-18): rules and random, seeds 0-5,
    the bandit by chance and forced. Every flag off gives the same events."""
    want = json.loads(BASELINE.read_text())
    for key, (digest, n) in want.items():
        name, seed, bandit = key.split("/")
        pol = Rules() if name == "rules" else Random()
        w = _night(pol, int(seed[4:]), bandit=None if bandit == "banditNone" else True)
        text = "\n".join(json.dumps(e) for e in w.log)
        assert (hashlib.sha256(text.encode()).hexdigest(), len(w.log)) == (digest, n), key


def test_all_flags_random_turns_are_valid():
    """1,000 random turns with every flag on: the chosen act is on the menu,
    beliefs come from what was perceived, the needs stay in 0..1."""
    w = World(seed=4, turns=1000, bandit=True, flags="all")
    pol = Random()
    for _ in range(1000):
        w.step(pol)
        for p in w.people.values():
            for d, x in w.drives(p).items():
                assert 0 <= x <= 1, (p.name, d, x)
    choices = [e for e in w.log if e.get("options")]
    assert len(choices) > 500
    for e in choices:
        assert e["options"][e["chosen"]] == e["text"], e
    acts = {e["action"] for e in w.log if e["done"]}
    assert {"sit down", "sleep", "look around", "gossip", "ask", "offer"} <= acts, acts
    ids = {e["id"] for e in w.log}
    for p in w.people.values():
        for key, (value, src) in p.bel.items():
            assert src in ("start", "sight", "own") or src in ids, (p.name, key, src)


def test_drives_seven_with_nature():
    w = World(seed=0, flags=("drives",))
    p = w.people["Tom"]
    assert tuple(w.drives(p)) == DRIVES and len(DRIVES) == 7
    for kind, nat in NATURE.items():
        assert tuple(nat) == DRIVES, kind
    assert NATURE["peasant"]["wealth"][0] == 0 and NATURE["merchant"]["wealth"][0] > 1   # only some care
    # every need at or under its set point: no error
    for a in ("hunger", "thirst", "fear", "fatigue", "boredom", "lonely", "slight"):
        setattr(p, a, 0.0)
    assert w.drive_error(p) == 0.0
    p.fatigue = 0.9
    assert abs(w.drive_error(p) - NATURE["peasant"]["rest"][0] * (0.9 - NATURE["peasant"]["rest"][1])) < 1e-9


def test_rest_rises_and_sitting_and_sleep_lower_it():
    w = World(seed=0, flags=("drives",))
    _quiet(w)
    tom = w.people["Tom"]
    f0 = tom.fatigue
    for _ in range(50):
        w._drift(tom)
    assert tom.fatigue > f0 + 0.09                       # fatigue rises through the night
    f1 = tom.fatigue
    _act(w, "Tom", "sit down")
    assert tom.fatigue < f1 - 0.1
    tom.fatigue = 0.8
    _act(w, "Tom", "sleep")
    assert tom.fatigue < 0.3
    assert "sit down" not in _texts(World(seed=0), "Tom")   # flag off: no such act


def test_a_sleeper_sees_nothing_and_a_noise_wakes_it():
    w = World(seed=0, flags=("drives",))
    _quiet(w)
    tom = w.people["Tom"]
    o = next(o for o in w.options(tom) if o["verb"] == "sleep")
    tom.doing = {**o, "left": 12, "choice": {}}
    assert w.asleep(tom)
    ev = _act(w, "Oren", "praise Sir Ada")               # quiet: not seen, not heard
    assert ev["id"] not in tom.perceived and w.asleep(tom)
    assert any(o["text"].startswith("steal the copper coin from Tom") for o in w.options(w.people["Fenn"]))
    ev = _act(w, "Fenn", "strike Oren")                  # loud: heard, and it wakes him
    assert ev["id"] in tom.perceived and not w.asleep(tom)
    assert any(e["actor"] == "Tom" and e["action"] == "sleep" and not e["done"] for e in w.log)


def test_curiosity_falls_with_the_new():
    w = World(seed=0, flags=("drives",))
    _quiet(w)
    fenn = w.people["Fenn"]
    fenn.boredom = 0.9
    _act(w, "Fenn", "look around")
    b1 = fenn.boredom
    assert b1 < 0.9
    _act(w, "Fenn", "look around")                       # nothing new the second time
    assert fenn.boredom > b1 - 0.03
    _act(w, "Fenn", "walk to the kitchen")
    b2 = fenn.boredom
    assert b2 < b1 - 0.1                                 # a room first seen tonight
    fenn.room = "the cellar"
    fenn.boredom = 0.9
    _act(w, "Fenn", "open the old chest")
    assert fenn.boredom <= 0.5                           # the first to open the chest
    w2 = World(seed=0, flags=("drives",))
    q = w2.people["Tom"]
    q.boredom = 0.5
    for _ in range(20):
        w2._drift(q)
    assert q.boredom > 0.55                              # nothing new: it rises


def test_company_and_standing_move_with_how_others_treat_me():
    w = World(seed=0, flags=("drives",))
    _quiet(w)
    tom = w.people["Tom"]
    s0, l0 = tom.slight, tom.lonely
    _act(w, "Oren", "insult Tom")
    assert tom.slight > s0 + 0.15 and tom.lonely > l0
    s1, l1 = tom.slight, tom.lonely
    _act(w, "Marta", "praise Tom")
    assert tom.slight < s1 and tom.lonely < l1
    # alone, company is wanted more and more
    tom.room = "the yard"
    l2 = tom.lonely
    for _ in range(20):
        w._drift(tom)
    assert tom.lonely > l2 + 0.05


def test_gossip_spreads_a_deed_and_moves_liking():
    w = World(seed=0, flags=("gossip",))
    _quiet(w)
    w.people["Sir Ada"].room = "the yard"                # Ada does not see it
    _act(w, "Fenn", "insult Tom")
    marta = w.people["Marta"]
    assert any(d["actor"] == "Fenn" and d["action"] == "insult" for d in marta.deeds)
    marta.room = w.people["Sir Ada"].room = "the yard"
    ada = w.people["Sir Ada"]
    ada.aff["Marta"] = 0.5
    before = ada.aff["Fenn"]
    _act(w, "Marta", "tell Sir Ada that Fenn insulted Tom")
    assert ada.aff["Fenn"] < before and ada.rep["Fenn"] < 0
    assert "tell Sir Ada that Fenn insulted Tom" not in _texts(w, "Marta")   # told once
    # a distrusted teller is not believed
    w2 = World(seed=0, flags=("gossip",))
    _quiet(w2)
    _act(w2, "Fenn", "insult Tom")
    ada2 = w2.people["Sir Ada"]
    ada2.aff["Marta"] = -0.8
    b2 = ada2.aff["Fenn"]
    _act(w2, "Marta", "tell Sir Ada that Fenn insulted Tom")
    assert ada2.aff["Fenn"] == b2


def test_trade_the_cook_serves_or_refuses():
    w = World(seed=0, flags=("trade",))
    _quiet(w)
    assert w.things["the loaf of bread"]["at"] == "Marta"
    _act(w, "Tom", "ask Marta for the loaf of bread")
    give = next(o for o in w.options(w.people["Marta"]) if o["text"] == "give the loaf of bread to Tom")
    assert give["f"]["asked"] == 1.0
    assert "refuse Tom" in _texts(w, "Marta")
    # a liked asker is served by the rules; a distrusted one is refused
    marta = w.people["Marta"]
    marta.aff["Tom"] = 0.5
    opts = w.options(marta)
    u = {o["text"]: utility(w, marta, o, marta.base) for o in opts}
    assert u["give the loaf of bread to Tom"] > u["refuse Tom"]
    marta.aff["Tom"], marta.liars["Tom"] = -0.6, 1
    opts = w.options(marta)
    u = {o["text"]: utility(w, marta, o, marta.base) for o in opts}
    assert u["refuse Tom"] > u["give the loaf of bread to Tom"]
    before = w.people["Tom"].aff["Marta"]
    _act(w, "Marta", "refuse Tom")
    assert w.people["Tom"].aff["Marta"] < before and "Tom" not in marta.requests
    # an offer accepted swaps the things and their owners
    _act(w, "Oren", "offer Sir Ada the lute for the sword")
    _act(w, "Sir Ada", "accept Oren's the lute for the sword")
    assert w.things["the sword"]["owner"] == "Oren" and w.things["the lute"]["owner"] == "Sir Ada"
    # a request lapses after 30 turns
    _act(w, "Fenn", "ask Marta for the bowl of stew")
    w.t += 31
    w._drift(marta)
    assert "Fenn" not in marta.requests


def test_watch_sees_a_distrusted_thief_more_often():
    def seen(flags, trials=400):
        n = 0
        for i in range(trials):
            w = World(seed=i, flags=flags)
            _quiet(w)
            w.people["Tom"].liars["Fenn"] = 1                 # Tom distrusts Fenn
            w.rng = random.Random(i)
            ev = {"id": "x", "actor": "Fenn", "room": "the main room", "action": "steal", "target": "Oren",
                  "parts": {"thing": "the lute"}, "done": True}
            w._perceive(ev, sneaky=True)
            n += "x" in w.people["Tom"].perceived
        return n / trials
    off, on = seen(()), seen(("watch",))
    assert on > off + 0.15, (off, on)


def test_check_finds_lies_out():
    # 1. seen holding it openly
    w = World(seed=0, flags=("check",))
    _quiet(w)
    w.things["the lute"].update(at="Fenn", hidden=False)
    w.people["Marta"].aff["Fenn"] = 0.3                    # she believes him
    lie = _act(w, "Fenn", "tell Marta that Tom took the lute") if "tell Marta that Tom took the lute" in \
        _texts(w, "Fenn") else None
    assert lie is not None and w.people["Marta"].bel[("took", "the lute")][0] == "Tom"
    w._look(w.people["Marta"], "sight")
    assert w.people["Marta"].bel[("took", "the lute")][0] == "Fenn" and w.people["Marta"].liars["Fenn"] == 1
    # the same without the flag: not found out by a look
    w0 = World(seed=0)
    _quiet(w0)
    w0.things["the lute"].update(at="Fenn", hidden=False)
    w0.people["Marta"].aff["Fenn"] = 0.3
    _act(w0, "Fenn", "tell Marta that Tom took the lute")
    w0._look(w0.people["Marta"], "sight")
    assert w0.people["Marta"].bel[("took", "the lute")][0] == "Tom"
    # 2. the named culprit searched, and it is not on him: the tale is doubted
    w = World(seed=0, flags=("check",))
    _quiet(w)
    w.things["the lute"].update(at="Fenn", hidden=True)
    w.people["Marta"].aff["Fenn"] = 0.3
    _act(w, "Fenn", "tell Marta that Tom took the lute")
    a0 = w.people["Marta"].aff["Fenn"]
    _act(w, "Marta", "search Tom")
    assert ("took", "the lute") not in w.people["Marta"].bel and w.people["Marta"].aff["Fenn"] < a0
    assert any(e["action"] == "doubt" for e in w.log)
    # 3. over many nights, more of the lies told are found out
    def found(flags):
        lies = out = 0
        for seed in range(12):
            w = _night(Rules(), seed, flags=flags)
            ids = {e["id"] for e in w.log if e["done"] and e.get("facts", {}).get("lie")}
            lies += len(ids)
            out += len({e["facts"]["contradicts"] for e in w.log if e.get("facts", {}).get("contradicts") in ids})
        return out / lies
    assert found(("check",)) > 1.5 * found(()), (found(()), found(("check",)))


def test_lifetime_keeps_the_cast_and_the_memory_of_dying():
    calls = []

    class Spy(Rules):
        def night_start(self, w, n):
            calls.append(("start", n))

        def night_end(self, w, n):
            calls.append(("end", n))

        def on_death(self, w, p, ev):
            calls.append(("death", p.name))

    traits = []

    def look(w, n):
        traits.append({p: dict(w.people[p].base) for p in NAMES})
        w.people["Tom"].aff["Fenn"] = -0.77            # a liking set tonight is there tomorrow
    carry, stats = lifetime(Spy(), nights=3, seed=5, turns=120, on_night=look)
    assert traits[0] == traits[1] == traits[2]           # the same cast every night
    assert [c for c in calls if c[0] != "death"] == [("start", 0), ("end", 0), ("start", 1), ("end", 1),
                                                     ("start", 2), ("end", 2)], calls
    assert carry["Tom"]["aff"]["Fenn"] == -0.77
    assert len(stats) == 3 and all(s["turns"] <= 120 for s in stats)
    # a real death: hooked, remembered, and the dead are back the next night
    w = World(seed=0, flags="all")
    _quiet(w)
    tom = w.people["Tom"]
    tom.hp = -1.0
    tom.dead = tom.down = True
    nxt = World(seed=1, flags="all", carry=w.carry_out(), night=1)
    assert not nxt.people["Tom"].dead and nxt.people["Tom"].hp == 10.0
    assert nxt.people["Tom"].base == tom.base


def test_lifetime_death_hook_fires():
    died = []

    class Killer(Rules):
        def on_death(self, w, p, ev):
            died.append((p.name, ev and ev["actor"]))
    carry, stats = None, []
    for seed in range(40):
        carry, stats = lifetime(Killer(), nights=2, seed=seed, turns=300)
        if died:
            break
    assert died, "no death in 40 two-night lifetimes"
    cast_deaths = [(n, by) for n, by in died if n in NAMES]
    for name, by in cast_deaths:                        # the memory of dying, carried
        assert any(m["what"] == "died" and m["by"] == by for m in carry[name]["memory"])


def _setup_partner(policy_by):
    w = World(seed=0, flags="all")
    _quiet(w)
    pol = Cast(Rules(), policy_by)
    pol.night_start(w, 0)
    return w, pol


def _pick(w, pol, name):
    p = w.people[name]
    opts = w.options(p)
    pc, _ = pol.probs(w, p, opts)
    return opts[max(range(len(opts)), key=pc.__getitem__)], pc


def test_partner_kind_pays_back_a_gift():
    w, pol = _setup_partner({"Oren": P.Kind(p=1.0)})
    _act(w, "Tom", "give the copper coin to Oren")
    o, pc = _pick(w, pol, "Oren")
    assert o["verb"] == "give" and o["target"] == "Tom" and o["obj"] == "the purse of silver" and max(pc) == 1
    w, pol = _setup_partner({"Oren": P.Kind(p=0.0)})        # p = 0: never
    _act(w, "Tom", "give the copper coin to Oren")
    o, _ = _pick(w, pol, "Oren")
    assert not (o["verb"] == "give" and o["target"] == "Tom")


def test_partner_hard_hits_back_when_insulted():
    w, pol = _setup_partner({"Sir Ada": P.Hard(p=1.0)})
    _act(w, "Tom", "insult Sir Ada")
    o, _ = _pick(w, pol, "Sir Ada")
    assert o["text"] == "strike Tom with the sword"
    w, pol = _setup_partner({"Sir Ada": P.Hard(p=1.0)})
    _act(w, "Tom", "praise Sir Ada")                     # nothing else is answered
    o, _ = _pick(w, pol, "Sir Ada")
    assert o["verb"] in P.BACKGROUND


def test_partner_deaf_never_touches_anyone():
    seen = []

    def grab(wn, n):
        seen.extend(e for e in wn.log if e["actor"] == "Oren" and e.get("options"))
    lifetime(Cast(Rules(), {"Oren": P.Deaf()}), nights=3, seed=1, turns=300, on_night=grab)
    assert seen and all(e["action"] in P.BACKGROUND for e in seen), {e["action"] for e in seen}


def test_partner_switch_changes_rule_at_set_nights():
    sw = P.make_partner("P-switch:kind>hard@2,4")
    w = World(seed=0, flags="all")
    got = []
    for n in range(6):
        sw.night_start(w, n)
        got.append(sw.now.name)
    assert got == ["P-kind", "P-kind", "P-hard", "P-hard", "P-kind", "P-kind"], got
    assert "until night 2" in sw.rule


def _hide_change(w, p, rng):
    """Change, at random, facts p has not perceived: the others' anger at p,
    where the things outside p's room are, what anyone outside p's room
    holds, what is in the old chest."""
    import copy
    w = copy.deepcopy(w)
    p = w.people[p.name]
    for q in w.people.values():
        if q is not p:
            q.anger[p.name] = rng.random()
    far = [r for r in ("the main room", "the kitchen", "the cellar", "the yard") if r != p.room]
    out_of_sight = {q.name for q in w.people.values() if q.room != p.room}
    for n, th in w.things.items():
        if "fixed" in th["props"]:
            continue
        if th["at"] in far or (th["at"] in out_of_sight):
            th["at"] = rng.choice(far)
    w.chest = rng.choice([None, "some old silver coins", "nothing but rags"])
    return w, p


def _menu(w, p):
    return [(o["text"], sorted((k, v) for k, v in o["f"].items() if k != "deed")) for o in w.options(p)]


def test_known_menu_reads_no_hidden_fact():
    """With `known` on, a chooser's menu and every number on it stay the
    same when facts it has not perceived change (the verifier's leak: others'
    private anger, unseen rooms' contents, the chest). Without the flag the
    same change does move the menu (so the test can fail)."""
    rng = random.Random(7)
    moved_without = 0
    for seed in range(4):
        for flags in ("all", ("drives", "gossip", "trade", "watch", "check")):
            w = World(seed=seed, turns=300, bandit=True, flags=flags)
            pol = Rules()
            for t in range(240):
                w.step(pol)
                if t % 20 != 5:
                    continue
                for p in list(w.people.values()):
                    if not p.acts():
                        continue
                    before = _menu(w, p)
                    w2, p2 = _hide_change(w, p, rng)
                    after = _menu(w2, p2)
                    if flags == "all":
                        assert before == after, (seed, t, p.name, [x for x in after if x not in before][:3],
                                                 [x for x in before if x not in after][:3])
                        assert w.danger(p) == w2.danger(p2)
                    else:
                        moved_without += before != after
    assert moved_without > 0


def test_known_a_held_thing_taken_unseen_is_noticed_at_the_next_choice():
    """Under `known` p notices the loss of ANY thing it believed it held (not
    only its own), so the wealth need never drops unnoticed; handing a thing
    over itself is not a loss it 'notices'."""
    w = World(seed=0, flags="all")
    _quiet(w)
    fenn, oren = w.people["Fenn"], w.people["Oren"]
    w.things["the lantern"].update(at="Fenn", owner=None)        # not his own
    w._look(fenn, "sight")
    rich = w.wealth(fenn)
    w.things["the lantern"]["at"] = "Oren"                       # taken, unseen
    assert w.wealth(fenn) == rich                               # not felt before it is noticed
    w._check_own(fenn)
    assert w.wealth(fenn) == rich - 1
    assert any(e["text"] == "notice that the lantern is gone" and e["actor"] == "Fenn" for e in w.log)
    w2 = World(seed=0, flags="all")
    _quiet(w2)
    w2.things["the lantern"].update(at="Fenn", owner=None)
    w2._look(w2.people["Fenn"], "sight")
    _act(w2, "Fenn", "give the lantern to Oren")
    w2._check_own(w2.people["Fenn"])
    assert not any(e["action"] == "notice" for e in w2.log)


def test_known_danger_is_the_hostility_perceived():
    """Under `known`, a hidden grudge adds no danger and no flee option; an
    insult then a blow seen add it; a gift lowers it."""
    w = World(seed=0, flags="all")
    _quiet(w)
    tom, fenn = w.people["Tom"], w.people["Fenn"]
    fenn.anger["Tom"] = 1.0
    assert w.danger(tom) == 0 and not any(t.startswith("flee") for t in _texts(w, "Tom"))
    _act(w, "Fenn", "threaten Tom")
    _act(w, "Fenn", "shove Tom")
    d = w.danger(tom)
    assert d > 0.2 and any(t.startswith("flee") for t in _texts(w, "Tom")), d
    h = w.hostility(tom, "Fenn")
    w.things["the dagger"]["at"] = "Fenn"
    _act(w, "Fenn", "give the dagger to Tom")
    assert w.hostility(tom, "Fenn") < h


# ------------------------------------------------------------------ the fixes (2026-10-29): reprisal and wear
class _Spam:
    """Fenn does `verb` to Tom whenever the menu lets it (the learner's exploit);
    everyone else is the rules policy."""

    def __init__(self, verb, who="Fenn", target="Tom"):
        self.verb, self.who, self.target, self.rules = verb, who, target, Rules()

    def probs(self, w, p, opts):
        if p.name == self.who:
            for i, o in enumerate(opts):
                if o["verb"] == self.verb and o["target"] == self.target:
                    return [float(j == i) for j in range(len(opts))], [float(j == i) for j in range(len(opts))]
        return self.rules.probs(w, p, opts)


def _spam_night(flags, policy, seed, turns=200):
    from lab.world2.world import DUR
    w = World(seed=seed, turns=turns, bandit=False, flags=flags)
    for _ in range(turns):
        w.step(policy)
    tom = [e for e in w.log if e["actor"] == "Tom"]
    return {"finished": sum(1 for e in tom if e["done"] and DUR.get(e["action"], 0) >= 3),
            "harm": sum(1 for e in w.log if e["actor"] == "Fenn" and e["done"] and e["action"] in ("shove", "insult")),
            "err": w.drive_error(w.people["Fenn"]),
            "liked": sum(w.people[n].aff["Fenn"] for n in ("Tom", "Sir Ada", "Oren", "Marta")) / 4}


def test_reprisal_closes_the_interrupt_loophole():
    """A learner that shoves or insults one person every chance used to lock
    that person for the whole night (it finished nothing). With `reprisal`
    the target finishes what it does, the spammer's harm acts are cut to a
    few, and the spammer ends worse off than the same character left calm."""
    import statistics as S
    for verb in ("shove", "insult"):
        res = {}
        for label, flags in (("off", "all"), ("on", "all2")):
            for who, mk in (("spam", lambda: _Spam(verb)), ("calm", Rules)):
                runs = [_spam_night(flags, mk(), s) for s in range(1, 7)]
                res[label, who] = {k: S.mean(r[k] for r in runs) for k in runs[0]}
        off_s, off_c, on_s, on_c = res["off", "spam"], res["off", "calm"], res["on", "spam"], res["on", "calm"]
        assert off_s["finished"] < 0.2 * off_c["finished"], (verb, off_s, off_c)       # the loophole existed
        assert on_s["finished"] > 0.6 * on_c["finished"], (verb, on_s, on_c)           # and is closed
        assert on_s["harm"] < 0.3 * off_s["harm"], (verb, on_s, off_s)                 # it no longer pays to spam
        assert on_s["err"] > on_c["err"] + 0.1, (verb, on_s, on_c)                     # the spammer is worse off
        assert on_s["liked"] < on_c["liked"] - 0.1, (verb, on_s, on_c)                 # and thought the less of


def test_reprisal_one_rule_for_every_harmful_act():
    """Every harmful act aimed at a person takes at least HARM_MIN turns, and
    a witness's liking of the doer falls (target and friends aside)."""
    from lab.world2.world import HARM_MIN, HARMFUL, WITNESS_COST
    seen = set()
    w = World(seed=3, turns=300, bandit=True, flags="all2")
    for _ in range(300):
        for p in w.people.values():
            if p.acts():
                for o in w.options(p):
                    if o["verb"] in HARMFUL and o["target"] in w.people:
                        seen.add(o["verb"])
                        assert o["dur"] >= HARM_MIN, o
        w.step(Random())
    assert {"strike", "shove", "threaten", "insult", "search"} <= seen, seen
    w = World(seed=0, flags=("reprisal",))
    _quiet(w)
    for q in w.people.values():
        q.aff["Fenn"] = 0.0
    _act(w, "Fenn", "shove Tom")
    assert abs(w.people["Oren"].aff["Fenn"] + WITNESS_COST) < 1e-9
    assert abs(w.people["Marta"].aff["Fenn"] + WITNESS_COST) < 1e-9


def test_reprisal_an_interrupted_act_is_taken_up_where_it_stopped():
    w = World(seed=0, flags=("reprisal",))
    _quiet(w)
    tom = w.people["Tom"]
    tom.hunger = 0.9
    tom.base["bravery"] = 0.0
    w.things["the loaf of bread"].update(at="Tom")
    eat = next(o for o in w.options(tom) if o["text"] == "eat the loaf of bread")
    tom.doing = {**eat, "left": eat["dur"] - 3, "choice": {}}          # 3 of its 4 turns done
    w.interrupt(tom, "a shove")
    assert tom.doing is None and tom.partial[("eat", None, "the loaf of bread")][0] == 3

    class Eat:
        def probs(self, w_, p, opts):
            pc = [float(o["text"] == "eat the loaf of bread") for o in opts]
            return pc, pc
    w._choose(tom, Eat())
    assert tom.doing is None or tom.doing["left"] == 1                  # one turn was left (1 = done at once)
    assert w.things["the loaf of bread"]["at"] is None                  # it ate
    # too late: the progress is gone, the act takes its whole time
    w.things["the loaf of bread"].update(at="Tom")
    tom.doing = {**eat, "left": 1, "choice": {}}
    w.interrupt(tom, "a shove")
    w.t += 30
    w._choose(tom, Eat())
    assert tom.doing["left"] == eat["dur"]


def test_wear_the_same_words_mean_less_each_time():
    from lab.world2.world import WEAR, WEAR_SPAN
    w = World(seed=0, flags=("drives", "wear"))
    _quiet(w)
    tom = w.people["Tom"]
    drops, anger = [], []
    for i in range(3):
        a0, g0 = tom.aff["Oren"], tom.anger.get("Oren", 0)
        _act(w, "Oren", "insult Tom")
        drops.append(a0 - tom.aff["Oren"])
        anger.append(tom.anger["Oren"] - g0)
        w.t += 1
    assert drops[0] > 0 and abs(drops[1] / drops[0] - WEAR) < 0.05 and abs(drops[2] / drops[0] - WEAR ** 2) < 0.05, drops
    assert anger[1] < anger[0] and anger[2] < anger[1], anger
    # the same words from SOMEONE ELSE, or the same ones after the span, count in full
    a0 = tom.aff["Marta"]
    _act(w, "Marta", "insult Tom")
    assert abs((a0 - tom.aff["Marta"]) - drops[0]) < 0.05
    w.t += WEAR_SPAN + 1
    a0 = tom.aff["Oren"]
    _act(w, "Oren", "insult Tom")
    assert abs((a0 - tom.aff["Oren"]) - drops[0]) < 0.05
    # praise wears too, and standing follows (the slight)
    w = World(seed=0, flags=("drives", "wear"))
    _quiet(w)
    tom = w.people["Tom"]
    gains, slights = [], []
    for i in range(3):
        a0, s0 = tom.aff["Oren"], tom.slight
        tom.aff["Oren"] = 0.0
        a0 = 0.0
        _act(w, "Oren", "praise Tom")
        gains.append(tom.aff["Oren"] - a0)
        slights.append(s0 - tom.slight)
        w.t += 1
    assert gains[1] < 0.6 * gains[0] and gains[2] < 0.6 * gains[1], gains
    # with the flag off the same words always mean the same
    w = World(seed=0, flags=("drives",))
    _quiet(w)
    tom = w.people["Tom"]
    d = []
    for i in range(3):
        a0 = tom.aff["Oren"]
        _act(w, "Oren", "insult Tom")
        d.append(a0 - tom.aff["Oren"])
        w.t += 1
    assert abs(d[0] - d[1]) < 1e-9 and abs(d[1] - d[2]) < 1e-9, d


def test_the_new_flags_off_leave_all_as_it_was():
    from lab.world2.world import BASE_FLAGS
    assert set(World(flags="all").flags) == set(BASE_FLAGS)
    assert set(World(flags="all2").flags) == set(FLAGS) and {"reprisal", "wear"} <= set(FLAGS)


def test_unknown_flag_is_refused():
    try:
        World(flags=("telepathy",))
    except AssertionError:
        return
    raise AssertionError("an unknown flag was accepted")


if __name__ == "__main__":
    n = 0
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            n += 1
    print(f"{n} passed")
