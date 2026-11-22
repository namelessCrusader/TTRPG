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
from lab.world2 import world as W
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


# ---------------------------------------------------------------- the inn fix (2026-11-08): meals, partner, news, calm
def _fix(flags, seed=0, **kw):
    w = World(seed=seed, flags=flags, **kw)
    _quiet(w)
    return w


def test_fix_flags_are_new_and_all2_has_none_of_them():
    from lab.world2.world import ALL3, NEW_FLAGS
    assert not set(NEW_FLAGS) & set(World(flags="all2").flags)
    assert set(World(flags="all3").flags) == set(FLAGS) | set(NEW_FLAGS) == set(ALL3)
    try:
        World(flags=("meals",))                      # meals needs the cook's trade
    except AssertionError:
        pass
    else:
        raise AssertionError("meals without trade was accepted")


def test_meals_a_portion_feeds_a_little_and_the_pot_stays():
    from lab.world2.world import PORTION, STOCK
    w = _fix(("trade", "meals"))
    marta = w.people["Marta"]
    assert w.portions == STOCK and "the bowl of stew" not in w.things
    assert w.things["the pot of stew"]["at"] == "Marta"
    marta.hunger = 0.8
    _act(w, "Marta", "eat a portion of the pot of stew")
    assert abs(marta.hunger - (0.8 - PORTION)) < 1e-9
    assert w.portions["the pot of stew"] == STOCK["the pot of stew"] - 1 and w.things["the pot of stew"]["at"] == "Marta"
    marta.hunger = 0.05                                  # a full stomach holds no more
    assert not any(t.startswith("eat") for t in _texts(w, "Marta"))
    # the whole supply can feed five people through a night, with effort (and not with none)
    assert sum(STOCK.values()) * PORTION >= 5 * (0.003 * 300 + 0.3 - 0.3)
    # the supply is a dial
    assert World(flags=("trade", "meals"), supply=0.5).portions["the pot of stew"] == 6
    assert World(flags=("trade", "meals"), supply=0.0).portions["the loaf of bread"] == 1   # never none at all


def test_meals_serving_costs_a_fed_giver_a_portion_and_feeds_a_hungry_one():
    from lab.world2.world import PORTION
    w = _fix(("trade", "meals"))
    marta, tom = w.people["Marta"], w.people["Tom"]
    marta.hunger, tom.hunger, tom.aff["Marta"] = 0.05, 0.9, 0.0
    n0 = w.portions["the pot of stew"]
    _act(w, "Marta", "serve Tom a portion of the pot of stew")
    assert abs(tom.hunger - (0.9 - PORTION)) < 1e-9 and marta.hunger == 0.05      # the giver did not lose any need
    assert w.portions["the pot of stew"] == n0 - 1 and tom.aff["Marta"] >= 0.3
    # a whole pot is not handed over: the portions are shared
    assert not any(t.startswith("give the pot") for t in _texts(w, "Marta"))
    # the one asked answers by serving, and the request is cleared
    _act(w, "Tom", "ask Marta for a portion of the pot of stew")
    opts = {o["text"]: o for o in w.options(marta)}
    assert opts["serve Tom a portion of the pot of stew"]["f"]["asked"] == 1.0
    _act(w, "Marta", "serve Tom a portion of the pot of stew")
    assert "Tom" not in marta.requests
    # the rules score serves the hungry asker and not a stranger it distrusts
    marta.aff["Tom"], marta.liars.pop("Tom", None) and None
    w.people["Tom"].hunger = 0.9
    _act(w, "Tom", "ask Marta for a portion of the pot of stew")
    u = {o["text"]: utility(w, marta, o, marta.base) for o in w.options(marta)}
    assert u["serve Tom a portion of the pot of stew"] > u["refuse Tom"] and u["serve Tom a portion of the pot of stew"] > 0


def test_meals_a_thing_buys_a_meal_and_work_earns_a_ration():
    from lab.world2.world import PORTION, STOCK
    w = _fix(("trade", "meals", "drives"))
    tom, marta = w.people["Tom"], w.people["Marta"]
    tom.hunger = 0.9
    assert "pay Marta the copper coin for a meal from the pot of stew" in _texts(w, "Tom")
    assert not any("purse of silver" in t and t.startswith("pay") for t in _texts(w, "Oren"))   # no change is given
    _act(w, "Tom", "pay Marta the copper coin for a meal from the pot of stew")
    assert abs(tom.hunger - (0.9 - 2 * PORTION)) < 1e-9
    assert w.things["the copper coin"]["at"] == "Marta" and w.things["the copper coin"]["owner"] == "Marta"
    f0, m0 = tom.fatigue, marta.fatigue
    _act(w, "Tom", "work for Marta for a share of the pot of stew")
    assert abs(tom.hunger - (0.9 - 3 * PORTION)) < 1e-9
    assert tom.fatigue > f0 and marta.fatigue < m0
    assert w.portions["the pot of stew"] == STOCK["the pot of stew"] - 3
    # an empty pot is gone and the counter offers nothing
    w.portions["the pot of stew"] = 1
    _act(w, "Tom", "work for Marta for a share of the pot of stew")
    assert w.things["the pot of stew"]["at"] is None
    assert not any("pot of stew" in t and not t.startswith("tell") for t in _texts(w, "Tom"))


def test_meals_flags_off_the_cook_still_holds_one_bowl():
    w = _fix(("trade",))
    assert w.portions == {} and "the bowl of stew" in w.things and "eat the bowl of stew" in _texts(w, "Marta")


def test_partner_one_rule_for_everyone_trust_from_history():
    from lab.world2.policy import flag_terms
    w = _fix(("partner", "trade"))
    tom = w.people["Tom"]
    tom.aff.update({"Sir Ada": 0.8, "Fenn": -0.5, "Oren": 0.0})
    tom.liars["Oren"] = 1
    tom.memory.append({"what": "robbed", "who": "Fenn", "night": 0, "t": 1})
    assert w.trust(tom, "Sir Ada") == 0.8 and w.trust(tom, "Oren") == -0.5
    assert w.trust(tom, "Fenn") < -0.5                    # liking, and the robbery it remembers
    opts = {o["text"]: o for o in w.options(tom)}
    assert abs(opts["praise Sir Ada"]["f"]["trust"] - 0.8) < 1e-9
    kind = flag_terms(w, tom, opts["praise Sir Ada"])
    harm = flag_terms(w, tom, opts["insult Sir Ada"])
    assert kind > 0 and harm < 0                           # kind to the trusted, restraint toward the trusted
    assert flag_terms(w, tom, opts["praise Fenn"]) < 0 and flag_terms(w, tom, opts["insult Fenn"]) == 0.0
    # the same terms for any character with the same history, and no trait in them
    ada = w.people["Sir Ada"]
    ada.aff.update({"Tom": 0.8})
    aopts = {o["text"]: o for o in w.options(ada)}
    assert abs(flag_terms(w, ada, aopts["praise Tom"]) - kind) < 1e-9
    assert utility(w, tom, opts["praise Sir Ada"], tom.base) - utility(w, tom, opts["praise Sir Ada"], tom.base) == 0
    w0 = _fix(("trade",))
    assert "trust" not in {k for o in w0.options(w0.people["Tom"]) for k in o["f"]}


def test_news_a_rare_good_deed_is_worth_telling_and_is_told():
    w = _fix(("gossip", "news", "reprisal"))
    marta = w.people["Marta"]
    deeds = [{"key": (0, f"e{i}"), "actor": "Fenn", "action": "insult", "target": "Tom"} for i in range(8)]
    good = {"key": (0, "e9"), "actor": "Fenn", "action": "give", "target": "Tom"}
    marta.deeds = [good] + deeds                          # the good deed is the OLDEST
    marta.aff["Fenn"] = -0.6                              # and it goes against her opinion of the doer
    assert w._salience(marta, good) > 4 * w._salience(marta, deeds[0])
    opts = {o["text"]: o for o in w.options(marta)}
    t = "tell Sir Ada that Fenn gave a gift to Tom"
    assert t in opts, list(opts)                          # top 3 by salience, not the 3 newest
    bad = [o for o in opts.values() if o["verb"] == "gossip" and o["f"]["bad"]]
    u_good = utility(w, marta, opts[t], marta.base)
    assert u_good > max(utility(w, marta, o, marta.base) for o in bad)
    # flag off: only the three newest
    w0 = _fix(("gossip",))
    w0.people["Marta"].deeds = [dict(good)] + [dict(d) for d in deeds]
    assert t not in _texts(w0, "Marta")
    # deeds are forgotten from the commoner kind, so the rare one is kept
    for i in range(20):
        w._witness(marta, "Fenn", "insult", "Tom", f"x{i}")
    assert any(d["action"] == "give" for d in marta.deeds) and len(marta.deeds) == 12


def test_news_a_witness_warms_to_one_who_helps_as_it_cools_to_one_who_harms():
    w = _fix(("drives", "reprisal", "news"))
    ada, tom = w.people["Sir Ada"], w.people["Tom"]
    ada.aff["Marta"], ada.slight, w.people["Marta"].lonely = 0.0, 0.5, 0.5
    _act(w, "Marta", "give the ladle to Tom")
    assert abs(ada.aff["Marta"] - 0.04) < 1e-9           # WITNESS_COST, the mirror of reprisal
    assert w.people["Marta"].lonely < 0.5
    w0 = _fix(("drives", "reprisal"))
    w0.people["Sir Ada"].aff["Marta"] = 0.0
    _act(w0, "Marta", "give the ladle to Tom")
    assert w0.people["Sir Ada"].aff["Marta"] == 0.0


def test_calm_what_one_said_weighs_all_night_and_anger_is_spent_by_acting():
    from lab.world2.world import DISCHARGE
    w = _fix(("calm", "gossip", "drives"))
    fenn, ada = w.people["Fenn"], w.people["Sir Ada"]
    ada.anger["Fenn"] = 0.8
    ada.bel[("took", "the dagger")] = ("Fenn", "e0")
    _act(w, "Sir Ada", "insult Fenn")
    assert abs(ada.anger["Fenn"] - 0.8 * (1 - DISCHARGE)) < 1e-6
    w.t += 100                                            # well past the old 40-turn habit window
    o = next(o for o in w.options(ada) if o["text"] == "insult Fenn")
    assert 0.3 < o["f"]["repeat"] < 1.0 and o["f"]["repeat_kind"] > o["f"]["repeat"]   # remembered, faded
    w0 = _fix(("gossip", "drives"))
    w0.people["Sir Ada"].anger["Fenn"] = 0.8
    _act(w0, "Sir Ada", "insult Fenn")
    w0.t += 100
    o0 = next(o for o in w0.options(w0.people["Sir Ada"]) if o["text"] == "insult Fenn")
    assert o0["f"]["repeat"] == 0.0 and w0.people["Sir Ada"].anger["Fenn"] == 0.8      # flag off: as before


def test_calm_a_witness_charges_the_answer_to_a_wrong_less():
    from lab.world2.world import PROVOKED_SHARE, WITNESS_COST
    def cost(flags, first):
        w = _fix(flags)
        for p in w.people.values():
            p.aff = {k: 0.0 for k in p.aff}
        if first:
            _act(w, "Fenn", "strike Tom")
        before = w.people["Sir Ada"].aff["Tom"]
        _act(w, "Tom", "strike Fenn")
        return before - w.people["Sir Ada"].aff["Tom"], w
    c_wrong, w = cost(("reprisal", "calm"), first=False)         # Tom strikes first
    c_answer, _ = cost(("reprisal", "calm"), first=True)         # Tom answers Fenn's blow
    c_old, _ = cost(("reprisal",), first=True)
    assert c_old > 0 and abs(c_answer - c_old * PROVOKED_SHARE) < 1e-9 and c_wrong >= c_old - 1e-9


def test_calm_the_rules_read_wounds_and_loneliness_and_drop_a_moot_charge():
    from lab.world2.policy import flag_terms
    w = _fix(("calm", "drives", "known", "gossip"))
    tom = w.people["Tom"]
    o = next(o for o in w.options(tom) if o["text"] == "threaten Fenn")
    tom.hp = 10.0
    well = flag_terms(w, tom, o)
    tom.hp = 4.0
    assert flag_terms(w, tom, o) < well - 1.0                      # a hurt one is held back from blows and threats
    p = next(o for o in w.options(tom) if o["text"] == "praise Fenn")
    tom.lonely = 0.1
    a = flag_terms(w, tom, p)
    tom.lonely = 0.9
    assert flag_terms(w, tom, p) > a + 0.5
    # a charge about a thing the accuser no longer thinks the culprit holds is moot
    tom.bel[("took", "the purse of silver")] = ("Fenn", "e0")
    tom.bel[("where", "the purse of silver")] = ("Fenn", "e0")
    acc = next(o for o in w.options(tom) if o["text"] == "accuse Fenn of taking the purse of silver")
    assert acc["f"]["moot"] == 0.0
    tom.bel[("where", "the purse of silver")] = ("Oren", "e1")
    acc2 = next(o for o in w.options(tom) if o["text"] == "accuse Fenn of taking the purse of silver")
    assert acc2["f"]["moot"] == 1.0 and utility(w, tom, acc2, tom.base) < utility(w, tom, acc, tom.base) - 1.0


def test_fix_all3_random_turns_are_valid():
    w = World(seed=6, turns=1000, bandit=True, flags="all3")
    pol = Random()
    for _ in range(1000):
        w.step(pol)
        for p in w.people.values():
            for d, x in w.drives(p).items():
                assert 0 <= x <= 1, (p.name, d, x)
    choices = [e for e in w.log if e.get("options")]
    for e in choices:
        assert e["options"][e["chosen"]] == e["text"], e
    acts = {e["action"] for e in w.log if e["done"]}
    assert {"serve", "work", "buy"} & acts, acts
    # and 200 turns of rules with everything on
    w = World(seed=7, turns=200, bandit=True, flags="all3")
    for _ in range(200):
        w.step(Rules())


def test_fix_all3_the_food_reaches_people_who_are_not_the_cook():
    """Rules characters over a few nights: more of the food leaves the cook's own mouth than before."""
    def fed(flags):
        got = {}
        for night in range(4):
            w = World(seed=night, bandit=False, flags=flags)
            for _ in range(300):
                w.step(Rules())
            for e in w.log:
                if e["done"] and e["action"] in ("serve", "buy", "work", "eat") and e["actor"] != "the bandit":
                    who = e["target"] if e["action"] == "serve" else e["actor"]
                    got[who] = got.get(who, 0) + 1
        return got
    old, new = fed("all2"), fed("all3")
    others_old = sum(v for k, v in old.items() if k != "Marta")
    others_new = sum(v for k, v in new.items() if k != "Marta")
    assert others_new >= 5 * max(1, others_old), (old, new)


# ---------------------------------------------------------------- luck (2026-11-20)
def test_luck_flag_is_new_and_needs_meals():
    from lab.world2.world import ALL3, ALL4
    assert "luck" not in set(World(flags="all3").flags) and set(World(flags="all4").flags) == set(ALL3) | {"luck"} == set(ALL4)
    w = World(flags="all3")
    assert w.catch == {} and not any("catch" in n for n in w.things) and w.portions == {"the pot of stew": 12, "the loaf of bread": 8}
    try:
        World(flags=("trade", "luck"))
    except AssertionError:
        pass
    else:
        raise AssertionError("luck without meals was accepted")


def test_luck_zero_is_an_even_split_and_the_dial_sets_the_empty_nights():
    from lab.world2.world import LUCK_SHARE, STOCK
    w = World(seed=3, flags="all4", luck=0.0)
    cooks = {"Marta"}
    assert set(w.catch) == {"Tom", "Sir Ada", "Oren", "Fenn"} and len(set(w.catch.values())) == 1
    total = sum(w.catch.values()) + sum(v for n, v in w.portions.items() if "catch" not in n)
    assert abs(total - sum(STOCK.values())) <= 2                 # the same expected food as the all3 inn, split
    assert abs(w.portions["the pot of stew"] - 12 * (1 - LUCK_SHARE)) <= 1
    empty = 0
    got = []
    for seed in range(400):
        c = World(seed=seed, flags="all4", luck=0.5).catch
        got += list(c.values())
        empty += sum(1 for k in c.values() if k == 0)
    assert 0.45 < empty / len(got) < 0.62                         # ~ v (a little more: small gammas round to 0)
    assert abs(sum(got) / len(got) - 2.5) < 0.25     # the mean is the even share (0.5 * 20 / 4), whatever v is


def test_luck_is_independent_between_people_and_the_same_for_every_chooser():
    a = World(seed=11, flags="all4", luck=0.5).catch
    b = World(seed=11, flags="all4", luck=0.5).catch
    assert a == b                                               # a function of the night seed alone
    # drawn from a generator of its own: the world's own draws (the cast, the bandit, the first likings) are unchanged
    x, y = World(seed=11, flags="all4", luck=0.1), World(seed=11, flags="all4", luck=0.9)
    assert [p.hunger for p in x.people.values()] == [p.hunger for p in y.people.values()]
    assert x.people["Tom"].aff == y.people["Tom"].aff and x.bandit_at == y.bandit_at
    # not all equal, and not copies of one draw
    seen = {tuple(World(seed=s, flags="all4", luck=0.5).catch.values()) for s in range(30)}
    assert len(seen) > 20
    pairs = [(World(seed=s, flags="all4", luck=0.5).catch["Tom"] == 0, World(seed=s, flags="all4", luck=0.5).catch["Fenn"] == 0)
             for s in range(300)]
    pt, pf = sum(p[0] for p in pairs) / 300, sum(p[1] for p in pairs) / 300
    both = sum(1 for p in pairs if p[0] and p[1]) / 300
    assert abs(both - pt * pf) < 0.06


def test_luck_a_catch_is_eaten_and_shared_by_the_portion_and_is_not_handed_over():
    from lab.world2.world import PORTION
    w = None
    for seed in range(50):                                      # a night on which Tom has a catch
        w = _fix(("trade", "meals", "luck"), seed=seed, luck=0.3)
        if w.catch.get("Tom", 0) >= 2:
            break
    k = w.catch["Tom"]
    tom, oren = w.people["Tom"], w.people["Oren"]
    assert w.portions["Tom's catch"] == k and w.things["Tom's catch"]["at"] == "Tom"
    tom.hunger, oren.hunger, oren.room = 0.9, 0.9, tom.room
    assert "eat a portion of Tom's catch" in _texts(w, "Tom")
    assert not any(t.startswith("give Tom's catch") for t in _texts(w, "Tom"))
    _act(w, "Tom", "serve Oren a portion of Tom's catch")
    assert abs(oren.hunger - (0.9 - PORTION)) < 1e-9 and w.portions["Tom's catch"] == k - 1
    _act(w, "Tom", "eat a portion of Tom's catch")
    assert abs(tom.hunger - (0.9 - PORTION)) < 1e-9
    # someone with nothing to eat may buy a meal from a lucky person, or work for it
    opts = _texts(w, "Oren")
    assert "work for Tom for a share of Tom's catch" in opts
    # an empty catch is gone
    w.portions["Tom's catch"] = 1
    _act(w, "Tom", "eat a portion of Tom's catch")
    assert w.things["Tom's catch"]["at"] is None


def test_luck_all4_random_turns_are_valid_and_the_catch_does_not_outlast_the_night():
    w = World(seed=4, turns=600, bandit=True, flags="all4", luck=0.4)
    pol = Random()
    for _ in range(600):
        w.step(pol)
        for p in w.people.values():
            for d, x in w.drives(p).items():
                assert 0 <= x <= 1, (p.name, d, x)
    assert all(v >= 0 for v in w.portions.values())
    nxt = World(seed=5, flags="all4", luck=0.4, carry=w.carry_out(["Tom", "Sir Ada", "Oren", "Fenn", "Marta"]))
    assert all(nxt.things[f"{n}'s catch"]["owner"] == n for n, k in nxt.catch.items() if k > 0)   # a new night, new catches


# ---------------------------------------------------------------- the three needs only another person can meet (2026-11-20)
CAST5 = ["Tom", "Sir Ada", "Oren", "Fenn", "Marta"]


def test_need_flags_are_new_and_all5_has_all_three():
    from lab.world2.world import ALL4, ALL5, NEED_FLAGS
    assert not set(NEED_FLAGS) & set(World(flags="all4").flags)
    assert set(World(flags="all5").flags) == set(ALL4) | set(NEED_FLAGS) == set(ALL5)
    for f in NEED_FLAGS:
        World(flags=tuple(ALL4) + (f,))                          # each one alone is a valid world
    try:
        World(flags=tuple(set(ALL4) - {"luck"}) + ("labour",))   # labour splits the catch: it needs luck
    except AssertionError:
        pass
    else:
        raise AssertionError("labour without luck was accepted")


def test_tend_ailments_come_from_their_own_generator_at_the_dial_rate():
    a = World(seed=7, flags="all5", ill=1.0)
    assert all(p.ail >= 0.25 for p in a.people.values())                 # a dial of 1: everyone starts ill
    b, c = World(seed=7, flags="all5", ill=0.0), World(seed=7, flags="all5", ill=1.0)
    assert all(p.ail == 0 for p in b.people.values())
    # the world's own draws are the same whatever the dial
    assert [p.hunger for p in b.people.values()] == [p.hunger for p in c.people.values()] and b.bandit_at == c.bandit_at
    assert [x.ail for x in World(seed=9, flags="all5").people.values()] == [x.ail for x in World(seed=9, flags="all5").people.values()]
    n = sum(1 for sd in range(400) for p in World(seed=sd, flags="all5").people.values() if p.ail > 0)
    assert abs(n / 2000 - 0.10) < 0.025                                  # AIL_P, by default
    off = World(seed=7, flags="all4")
    assert all(p.ail == 0 for p in off.people.values())                  # without the flag no one is ill


def test_tend_the_owner_cannot_treat_it_and_another_person_can_at_a_cost():
    from lab.world2.world import AIL_GROW, TEND_RELIEF, TEND_TIRE
    w = _fix("all5", seed=2, ill=0.0)
    tom, ada = w.people["Tom"], w.people["Sir Ada"]
    tom.ail, ada.ail = 0.5, 0.0
    assert w.drives(tom)["safety"] >= 0.5                               # it is a need
    assert not any(t.startswith("tend Tom") for t in _texts(w, "Tom"))   # no one tends itself
    assert "tend Tom" in _texts(w, "Sir Ada") and "tend Sir Ada" not in _texts(w, "Tom")
    assert "ask Sir Ada to tend you" in _texts(w, "Tom")
    w._drift(tom)
    assert abs(tom.ail - (0.5 + AIL_GROW)) < 1e-9                       # left alone it grows
    before, fat = tom.ail, ada.fatigue
    o = next(o for o in w.options(ada) if o["text"] == "tend Tom")
    assert o["dur"] == 6                                                # the helper's time
    _act(w, "Sir Ada", "tend Tom")
    assert abs(tom.ail - (before - TEND_RELIEF)) < 1e-9
    assert abs(ada.fatigue - (fat + TEND_TIRE)) < 1e-9                  # and effort
    assert tom.aff["Sir Ada"] > 0.3
    tom.ail = 0.1
    assert "tend Tom" not in _texts(w, "Sir Ada")                       # a slight ailment is not worth a visit


def test_tend_an_ask_for_care_is_answered_by_a_kind_partner_and_the_same_rule_applies_to_everyone():
    w = _fix("all5", seed=3, ill=0.0)
    w.people["Fenn"].ail = 0.6
    pk = P.Kind(1.0, seed=1)
    ev = _act(w, "Fenn", "ask Sir Ada to tend you")
    assert w.people["Sir Ada"].requests["Fenn"]["thing"] == "care"
    opts = w.options(w.people["Sir Ada"])
    pc, _ = pk.probs(w, w.people["Sir Ada"], opts)
    assert opts[pc.index(1.0)]["text"] == "tend Fenn"
    # one rule: the same facts give the same score to two different people (only their own traits differ)
    w2 = _fix("all5", seed=4, ill=0.0)
    w2.people["Tom"].ail = 0.6
    t = {"bravery": .5, "greed": .5, "pride": .5, "kindness": .5, "temper": .5}
    for name in ("Sir Ada", "Oren"):
        for q in w2.people.values():
            q.fear, q.anger = 0.0, {}
        o = next(o for o in w2.options(w2.people[name]) if o["text"] == "tend Tom")
        assert o["f"]["need"] == 0.6 and utility(w2, w2.people[name], o, t) > utility(w2, w2.people[name], {**o, "f": {**o["f"], "need": 0.2}}, t)


def _fighters(n, seed, others_away=True):
    """n people in the yard with a raid in it, all chosen to fend off; the rest wait far away. -> (world, who fights)."""
    from lab.world2.world import RAID_HP
    w = World(seed=seed, flags="all5", raid=0.0, ill=0.0)
    _quiet(w)
    fighters = CAST5[:n]
    for name, p in w.people.items():
        p.room = "the yard" if name in fighters else "the cellar"
        p.hp = 10.0
    w.threat = {"name": "the wolf", "room": "the yard", "hp": RAID_HP, "t0": 0}

    class Script:
        name = "script"

        def probs(self, w_, p, opts):
            i = next((k for k, o in enumerate(opts) if o["verb"] == "fend off"), None)
            if i is None:
                i = next(k for k, o in enumerate(opts) if o["verb"] == "wait")
            pc = [0.0] * len(opts)
            pc[i] = 1.0
            return pc, pc
    return w, fighters, Script()


def test_allies_one_person_loses_and_two_win():
    res = {}
    for n in (1, 2, 3):
        won = lost_hp = down = 0
        for seed in range(30):
            w, fighters, pol = _fighters(n, seed)
            for _ in range(150):
                w.step(pol)
            won += w.threat is None
            lost_hp += sum(10 - w.people[f].hp for f in fighters) / n
            down += sum(w.people[f].down or w.people[f].dead for f in fighters)
        res[n] = (won / 30, lost_hp / 30, down / 30)
    assert res[1][0] <= 0.15, res                       # it beats one person
    assert res[2][0] >= 0.85 and res[3][0] >= 0.95, res  # and loses to two or more acting together
    assert res[2][1] > 0.5                              # joining costs the joiner risk: it is hit
    assert res[1][1] > res[3][1]                        # the more there are, the less each loses (pooled risk)
    assert res[1][2] > res[3][2]


def test_allies_raid_comes_from_its_own_generator_and_no_trait_decides_who_fights():
    a, b = World(seed=5, flags="all5", raid=1.0), World(seed=5, flags="all5", raid=1.0)
    assert a._raid_at == b._raid_at is not None and a._raid_kind == b._raid_kind
    x, y = World(seed=5, flags="all5", raid=0.0), World(seed=5, flags="all5", raid=1.0)
    assert x._raid_at is None and x.bandit_at == y.bandit_at and [p.hunger for p in x.people.values()] == [p.hunger for p in y.people.values()]
    n = sum(World(seed=s, flags="all5").threat is None and World(seed=s, flags="all5")._raid_at is not None for s in range(400))
    assert abs(n / 400 - 0.15) < 0.05                   # RAID_P
    assert World(seed=5, flags="all4")._raid_at is None
    # the same facts, the same score: two people with the same dials fend off the same
    w, fighters, _ = _fighters(2, 1)
    t = {"bravery": .6, "greed": .5, "pride": .5, "kindness": .5, "temper": .5}
    for name in fighters:
        w.people[name].fear = 0.0
    o1 = next(o for o in w.options(w.people["Tom"]) if o["verb"] == "fend off")
    o2 = next(o for o in w.options(w.people["Sir Ada"]) if o["verb"] == "fend off")
    assert o1["f"] == o2["f"] and utility(w, w.people["Tom"], o1, t) == utility(w, w.people["Sir Ada"], o2, t)
    # allies already in it pull more, a lone newcomer less
    lone = {**o1, "f": {**o1["f"], "allies": 0.0}}
    joined = {**o1, "f": {**o1["f"], "allies": 2.0}}
    assert utility(w, w.people["Tom"], joined, t) > utility(w, w.people["Tom"], lone, t)


def test_labour_catch_is_split_by_the_dial_and_a_repeated_kind_relieves_less():
    from lab.world2.world import ALL4, LABOUR_REPEAT, PORTION
    flags = tuple(ALL4) + ("labour",)
    w = None
    for seed in range(60):
        w = World(seed=seed, flags=flags, luck=0.3)
        if w.catch.get("Sir Ada", 0) >= 3:
            break
    ada = w.people["Sir Ada"]
    game, greens = w.portions.get("Sir Ada's game", 0), w.portions.get("Sir Ada's greens", 0)
    assert game + greens == w.catch["Sir Ada"] and not any(n.endswith("'s catch") for n in w.things)
    assert game > greens                                                   # Sir Ada's bravery is high: she brings game
    tot = {"game": 0, "greens": 0}
    for sd in range(200):
        x = World(seed=sd, flags=flags, luck=0.3)
        for pp in x.people.values():
            if pp.kind == "peasant":
                tot["game"] += x.portions.get(f"{pp.name}'s game", 0)
                tot["greens"] += x.portions.get(f"{pp.name}'s greens", 0)
    assert tot["greens"] > tot["game"]                                     # Tom's bravery is low: he brings greens
    # sensory-specific satiety
    w2 = _fix(flags, seed=1, luck=0.0)
    for p in w2.people.values():
        p.hunger = 0.9
    t, o = w2.people["Tom"], w2.people["Oren"]
    w2.portions["the pot of stew"], w2.kind_of = 12, w2.kind_of
    w2._feed(t, 1, "the pot of stew")
    assert abs(t.hunger - (0.9 - PORTION)) < 1e-9 and t.last_kind == "staple"            # the first is full
    w2._feed(t, 1, "the pot of stew")
    assert abs(t.hunger - (0.9 - PORTION - PORTION * LABOUR_REPEAT)) < 1e-9              # the same kind again: 60 percent
    # the menu says what a ration is worth to this person: the kind it just ate is worth less, another kind full
    t.hunger = 0.9
    w2.things["the pot of stew"]["at"] = "Tom"
    eat = next(o_ for o_ in w2.options(t) if o_["verb"] == "eat" and o_["obj"] == "the pot of stew")
    assert abs(eat["f"]["need"] - 0.9 * LABOUR_REPEAT) < 1e-9
    t.last_kind = "game"
    eat = next(o_ for o_ in w2.options(t) if o_["verb"] == "eat" and o_["obj"] == "the pot of stew")
    assert abs(eat["f"]["need"] - 0.9) < 1e-9
    t.last_kind = "staple"
    assert w2._eff(t, "the pot of stew") == LABOUR_REPEAT and w2._eff(o, "the pot of stew") == 1.0


def test_all5_random_turns_are_valid():
    w = World(seed=6, turns=900, bandit=True, flags="all5", luck=0.4, ill=0.5, raid=1.0)
    pol = Random()
    for _ in range(900):
        w.step(pol)
        for p in w.people.values():
            for d, x in w.drives(p).items():
                assert 0 <= x <= 1, (p.name, d, x)
    assert any(e["action"] == "raid arrives" for e in w.log)
    assert any(e["action"] == "tend" and e["done"] for e in w.log) or any(e["action"] == "fend off" for e in w.log)
    for e in w.log:
        if e.get("options"):
            assert e["options"][e["chosen"]] == e["text"]
    nxt = World(seed=7, flags="all5", carry=w.carry_out(CAST5))
    assert all(p.ail >= 0 for p in nxt.people.values()) and nxt.threat is None


# ---------------------------------------------------------------- lumpy, stock, choosy (2026-11-20, the reciprocity-conditions answer)
def test_lumpy_stock_choosy_are_new_and_all6_has_them_and_each_needs_its_base():
    from lab.world2.world import ALL5, LUMPY_FLAGS
    assert not set(LUMPY_FLAGS) & set(World(flags="all5").flags)
    assert set(World(flags="all6").flags) == set(ALL5) | set(LUMPY_FLAGS)
    for f in LUMPY_FLAGS:
        World(flags=tuple(ALL5) + (f,))                          # each one alone is a valid world
    for drop, add in (("luck", "lumpy"), ("meals", "stock"), ("partner", "choosy")):
        try:
            World(flags=tuple(x for x in ALL5 if x not in (drop, "labour")) + (add,))       # (labour needs luck too)
        except AssertionError:
            continue
        raise AssertionError(f"{add} without {drop} was accepted")


def test_lumpy_packages_keep_the_expected_catch_and_come_from_their_own_generator():
    from lab.world2.world import ALL5, LUMP_P
    fl = tuple(ALL5) + ("lumpy",)
    sizes, per = [], []
    for sd in range(300):
        w = World(seed=sd, flags=fl, luck=0.3)
        per += list(w.catch.values())
        sizes += [k for k in w.catch.values() if k]
        assert set(w.perish) == {n for n in w.things if n in w.portions and n not in ("the pot of stew", "the loaf of bread")}
    assert abs(sum(per) / len(per) - 2.5) < 0.35                  # the same expected catch as the smooth draw
    assert abs(len(sizes) / len(per) - LUMP_P) < 0.05             # hits are LUMP_P of the days
    assert all(7 <= k <= 9 for k in sizes)                       # a package of about 8 rations, more than one can eat
    a, b = World(seed=11, flags=fl, luck=0.3), World(seed=11, flags=fl, luck=0.9)
    assert a.catch == b.catch and a.perish == b.perish           # the draw is a function of the night seed only
    assert [p.hunger for p in a.people.values()] == [p.hunger for p in b.people.values()] and a.bandit_at == b.bandit_at
    assert a.catch != World(seed=11, flags=tuple(ALL5), luck=0.3).catch or sum(a.catch.values()) == 0


def test_lumpy_a_package_comes_in_at_its_turn_and_spoils_but_the_larder_keeps():
    from lab.world2.world import ALL4, SPOIL_HALF
    fl = tuple(ALL4) + ("lumpy",)
    for sd in range(100):
        w = World(seed=sd, flags=fl, luck=0.3)
        if any(w.catch.values()):
            break
    who = next(n for n, k in w.catch.items() if k)
    item = next(n for n in w.perish if n.startswith(who))
    t0, k = w.perish[item], w.portions[item]
    assert w.things[item]["at"] is None and item not in w.held(w.people[who])        # not in the house yet
    w.bandit_at = None
    for p in w.people.values():
        p.doing = None
    while w.t < t0 - 1:
        w.t += 1
        w._spoil()
    assert w.things[item]["at"] is None
    w.t += 1
    w._spoil()
    assert w.things[item]["at"] == who and w.portions[item] == k                     # it came in whole
    stew = w.portions["the pot of stew"]
    for _ in range(SPOIL_HALF):
        w.t += 1
        w._spoil()
    left = w.portions[item]
    assert k * 0.35 <= left <= k * 0.65, (k, left)                                  # about half after one half-life
    assert w.portions["the pot of stew"] == stew                                    # what keeps does not spoil
    assert abs(w.portions[item] / w.stock[item] - 1.0) < 1e-9                       # `left` (a share of the pot) is unchanged by rot


def test_lumpy_giving_a_package_costs_the_holder_little_and_the_rule_is_the_same_for_all():
    from lab.world2.world import ALL5
    from lab.world2.policy import MID, utility
    fl = tuple(ALL5) + ("lumpy",)
    for sd in range(100):
        w = _fix(fl, seed=sd, luck=0.3)
        if any(w.catch.values()):
            break
    who = next(n for n, k in w.catch.items() if k)
    item = next(n for n in w.perish if n.startswith(who))
    w.things[item]["at"] = who
    me, other = w.people[who], w.people["Marta"]
    me.hunger, other.hunger = 0.1, 0.9
    o = next(o for o in w.options(me) if o["verb"] == "serve" and o["target"] == "Marta" and o["obj"] == item)
    assert o["f"]["spare"] > 0.5                                    # most of it will rot before the holder can eat it
    # the staples keep: no spare, and the same fact is read for every person by the same rule
    stew = next(x for x in w.options(w.people["Marta"]) if x["verb"] == "serve")
    assert stew["f"]["spare"] == 0.0
    lo = dict(o, f=dict(o["f"], spare=0.0))
    for t in (MID, me.traits, other.traits):
        assert utility(w, me, o, t) >= utility(w, me, lo, t)         # spare only lowers the cost of giving
    flat = _fix(tuple(ALL5), seed=sd, luck=0.3)                     # without the flag the feature is not there
    assert all("spare" not in x["f"] for x in flat.options(flat.people[who]))


def test_stock_the_counter_carries_over_and_only_work_and_the_cook_refill_it():
    from lab.world2.world import ALL5, COOK_MAKES, STOCK_CAP, WORK_YIELD
    fl = tuple(ALL5) + ("stock",)
    w = _fix(fl, seed=1, luck=0.3)
    nominal = dict(w.portions)
    assert nominal["the pot of stew"] == 6 and nominal["the loaf of bread"] == 4
    for n in COOK_MAKES:
        w.portions[n] = 0
    nxt = World(seed=2, flags=fl, luck=0.3, carry=w.carry_out(CAST5))
    assert {n: nxt.portions[n] for n in COOK_MAKES} == COOK_MAKES      # an empty counter is only what the cook made
    for n in COOK_MAKES:
        w.portions[n] = 99
    nxt = World(seed=2, flags=fl, luck=0.3, carry=w.carry_out(CAST5))
    assert all(nxt.portions[n] == int(STOCK_CAP * nxt.stock[n]) for n in COOK_MAKES)      # capped
    plain = World(seed=2, flags=tuple(ALL5), luck=0.3, carry=World(seed=1, flags=tuple(ALL5), luck=0.3).carry_out(CAST5))
    assert plain.portions["the pot of stew"] == 6                       # without the flag the house is full again every night
    # work puts WORK_YIELD back after the worker takes a ration
    w = _fix(fl, seed=1, luck=0.3)
    tom, marta = w.people["Tom"], w.people["Marta"]
    tom.room = marta.room = "the main room"
    tom.hunger = 0.8
    w.portions["the pot of stew"] = 3
    _act(w, "Tom", "work for Marta for a share of the pot of stew")
    assert w.portions["the pot of stew"] == 3 - 1 + WORK_YIELD
    # a bad run empties the counter: nothing to buy, and the option is not on the menu
    w.portions["the pot of stew"] = 0
    w.things["the pot of stew"]["at"] = None
    assert not any("the pot of stew" in t for t in _texts(w, "Tom") if t.startswith(("work", "pay")))


def test_choosy_the_partner_rule_governs_tending_and_defending_with_the_serve_weight():
    from lab.world2.world import ALL5, CHOOSY_W
    from lab.world2.policy import MID, utility
    gaps = {}
    for fl in (tuple(ALL5) + ("choosy",), tuple(ALL5)):
        w = _fix(fl, seed=3, ill=0.0)
        for q in ("Oren", "Fenn"):
            w.people[q].ail = 0.5
            w.people[q].room = w.people["Sir Ada"].room = w.people["Tom"].room = "the main room"
        for actor in ("Sir Ada", "Tom"):
            a = w.people[actor]
            a.aff["Oren"], a.aff["Fenn"] = 0.0, -0.8                 # the same history with each of them
            opts = {o["target"]: o for o in w.options(a) if o["verb"] == "tend"}
            gaps[(len(fl), actor)] = utility(w, a, opts["Oren"], MID) - utility(w, a, opts["Fenn"], MID)
    on = {k[1]: v for k, v in gaps.items() if k[0] == len(ALL5) + 1}
    off = {k[1]: v for k, v in gaps.items() if k[0] == len(ALL5)}
    for actor in on:
        assert abs((on[actor] - off[actor]) - CHOOSY_W * 0.8) < 1e-9, (on, off)    # one extra weight on distrust, nothing else
    assert abs(on["Sir Ada"] - on["Tom"]) < 1e-9 and abs(off["Sir Ada"] - off["Tom"]) < 1e-9    # the same for every person (traits in MID)
    # defend too
    w = _fix(tuple(ALL5) + ("choosy",), seed=4)
    from lab.world2.policy import flag_terms
    a = w.people["Sir Ada"]
    a.aff["Oren"], a.aff["Fenn"] = 0.0, -0.8
    fake = lambda v, q: {"verb": v, "target": q, "f": {"trust": 0.0}, "obj": None}
    for v in ("defend", "bandage", "help up", "tend"):
        assert abs(flag_terms(w, a, fake(v, "Oren")) - flag_terms(w, a, fake(v, "Fenn")) - CHOOSY_W * 0.8) < 1e-9
    assert flag_terms(w, a, fake("praise", "Oren")) == flag_terms(w, a, fake("praise", "Fenn"))     # not a help act


def test_tending_and_defending_cannot_be_bought_at_the_counter():
    """S2.3: what is relation-specific is not for sale. The counter sells portions of food, nothing else."""
    w = World(seed=9, turns=900, bandit=True, flags="all6", luck=0.4, ill=0.5, raid=1.0)
    pol = Random()
    seen = set()
    for _ in range(900):
        w.step(pol)
        for p in w.people.values():
            if p.dead:
                continue
            for o in w.options(p):
                seen.add(o["verb"])
                if o["verb"] in ("buy", "work"):
                    assert o["obj"] in w.portions                    # only food is bought or worked for
                if o["verb"] in ("tend", "defend", "fend off", "bandage", "help up"):
                    assert "pay" not in o["f"] and o["obj"] in (None, "the wolf", "a raider")      # no price on it
    assert {"buy", "work"} <= seen
    assert not any(e["action"] in ("buy", "work") and e["obj"] not in w.portions for e in w.log if e.get("obj"))
    assert not any(th["props"] & {"care", "defence"} for th in w.things.values())


def test_all6_random_turns_are_valid_across_nights():
    carry = None
    for night in range(4):
        w = World(seed=20 + night, turns=600, bandit=True, flags="all6", luck=0.4, ill=0.5, raid=1.0, carry=carry, night=night)
        pol = Random()
        for _ in range(600):
            w.step(pol)
            for p in w.people.values():
                for d, x in w.drives(p).items():
                    assert 0 <= x <= 1, (p.name, d, x)
        assert all(v >= 0 for v in w.portions.values())
        for e in w.log:
            if e.get("options"):
                assert e["options"][e["chosen"]] == e["text"]
        carry = w.carry_out(CAST5)
    assert "house" in carry["Marta"]

# ---------------------------------------------------------------- signs and P-need (2026-11-21, SV2 and SV4)
def _signs_world(seed=3, flags="all6s"):
    from lab.world2.world import ALL6, SIGNS_FLAGS
    return World(seed=seed, turns=300, flags=tuple(ALL6) + SIGNS_FLAGS if flags == "all6s" else flags)


def _readings(w, obs, tgt, key, n=300):
    out = []
    for t in range(n):
        w.t = t
        out.append(w.perceived_need(obs, tgt)[key])
    return out


def test_signs_stoic_pain_is_under_read():
    w = _signs_world()
    tom, fenn = w.people["Tom"], w.people["Fenn"]
    tom.room = fenn.room
    tom.hp, tom.ail = 4.0, 0.0                              # true hurt 0.6
    tom.express = 0.25
    stoic = sum(_readings(w, "Fenn", "Tom", "hurt")) / 300
    tom.express = 1.0
    open_ = sum(_readings(w, "Fenn", "Tom", "hurt")) / 300
    assert stoic < 0.3 < 0.5 < open_ and abs(open_ - 0.6) < 0.03 and abs(stoic - 0.15) < 0.03, (stoic, open_)


def test_signs_acuity_sets_the_noise():
    import statistics
    w = _signs_world()
    tom, fenn = w.people["Tom"], w.people["Fenn"]
    tom.room = fenn.room
    tom.hp, tom.express = 5.0, 1.0
    fenn.acuity = 1.6
    sharp = statistics.pstdev(_readings(w, "Fenn", "Tom", "hurt"))
    fenn.acuity = 0.4
    dull = statistics.pstdev(_readings(w, "Fenn", "Tom", "hurt"))
    assert sharp < dull / 2 and 0 < sharp, (sharp, dull)
    w.t = 5
    assert w.perceived_need("Fenn", "Tom") == w.perceived_need("Fenn", "Tom")      # a reading repeats


def test_signs_need_the_same_room_and_an_awake_observer():
    w = _signs_world()
    w.people["Tom"].room, w.people["Marta"].room = "the main room", "the kitchen"
    assert w.perceived_need("Marta", "Tom") is None
    assert w.perceived_need("Tom", "Tom") is None
    w.people["Marta"].room = "the main room"
    assert w.perceived_need("Marta", "Tom") is not None
    w.people["Tom"].dead = True
    assert w.perceived_need("Marta", "Tom") is None


def test_signs_flag_off_changes_nothing():
    from lab.world2.world import ALL6, SIGNS_FLAGS
    off = World(seed=7, turns=200, bandit=True, flags="all6")
    on = World(seed=7, turns=200, bandit=True, flags=tuple(ALL6) + SIGNS_FLAGS)
    assert off.perceived_need("Tom", "Fenn") is None and not hasattr(off.people["Tom"], "express")
    for w in (off, on):
        pol = Random()
        for _ in range(200):
            w.step(pol)
    dig = lambda w: hashlib.sha256(json.dumps([[e["t"], e["actor"], e["action"], e["target"], e["done"]] for e in w.log]).encode()).hexdigest()
    assert dig(off) == dig(on)                               # the dials and readings take nothing from the world's generator


def test_signs_dials_stay_with_the_person_across_nights():
    from lab.world2.world import ALL6, SIGNS_FLAGS
    fl = tuple(ALL6) + SIGNS_FLAGS
    w0 = World(seed=11, turns=20, flags=fl, night=0)
    c = w0.carry_out(NAMES)
    w1 = World(seed=999, turns=20, flags=fl, carry=c, night=1)
    assert all((w0.people[n].express, w0.people[n].acuity) == (w1.people[n].express, w1.people[n].acuity) for n in NAMES)
    assert len({round(w0.people[n].express, 6) for n in NAMES}) > 1                # drawn per person


def _need_setup(**state):
    w = _signs_world()
    tom, fenn = w.people["Tom"], w.people["Fenn"]
    tom.room = fenn.room
    for p in (tom, fenn):
        p.express, p.acuity = 1.0, 1.6
    tom.hunger, tom.thirst, tom.fear, tom.ail, tom.hp = 0.2, 0.2, 0.0, 0.0, 10.0
    for k, v in state.items():
        setattr(tom, k, v)
    w.things["the loaf of bread"].update(at="Fenn", owner="Fenn")
    pn = P.Need(1.0, seed=1)
    pn.night_start(w, 0)
    pn.todo.append(("pay", "Tom", None))
    opts = w.options(fenn)
    return w, pn, opts, fenn


def test_pneed_pays_the_perceived_need():
    for state, verbs in ((dict(hunger=0.95), {"serve", "give"}), (dict(ail=0.6), {"tend"}), (dict(hp=3.0), {"bandage", "tend", "help up"}),
                         (dict(), {"praise"})):
        w, pn, opts, fenn = _need_setup(**state)
        i = pn.answer(w, fenn, opts)
        assert i is not None and opts[i]["verb"] in verbs and opts[i]["target"] == "Tom", (state, opts[i]["text"] if i is not None else None)
        assert not pn.todo
    w, pn, opts, fenn = _need_setup(hunger=0.95)
    assert opts[pn.answer(w, fenn, opts)] is not None


def test_pneed_misreads_a_stoic_and_pays_the_wrong_thing():
    w, pn, opts, fenn = _need_setup(hunger=0.95)
    w.people["Tom"].express = 0.2                            # the hunger shows as 0.19: under the threshold
    assert opts[pn.answer(w, fenn, opts)]["verb"] == "praise"


def test_pneed_without_signs_pays_company_and_answers_asks():
    w = World(seed=3, turns=300, flags="all6")
    fenn = w.people["Fenn"]
    w.people["Tom"].room, w.people["Tom"].hunger = fenn.room, 0.95
    pn = P.Need(1.0, seed=1)
    pn.night_start(w, 0)
    pn.todo.append(("pay", "Tom", None))
    assert pn.answer(w, fenn, w.options(fenn)) is not None
    assert P.make_partner("P-need:0.5").p == 0.5


def test_pneed_hears_the_care_acts_and_never_starts():
    w, pn, opts, fenn = _need_setup()
    pn.todo.clear()
    for a in ("give", "serve", "tend", "bandage", "help up", "defend", "praise"):
        pn.hear(w, fenn, {"action": a, "actor": "Tom", "parts": {}})
    assert [t[0] for t in pn.todo] == ["pay"] * 7
    pn.todo.clear()
    pn.hear(w, fenn, {"action": "insult", "actor": "Tom", "parts": {}})
    assert not pn.todo
    pc, _ = pn.probs(w, fenn, opts)
    assert abs(sum(pc) - 1) < 1e-9 and all(opts[i]["verb"] in P.BACKGROUND for i, x in enumerate(pc) if x > 0)



# ------------------------------------------------------------------ hard: the hardships from the source rates (2026-11-21)
HARD = tuple(sorted(set(W.ALL6) | set(W.SIGNS_FLAGS) | set(W.HARD_FLAGS)))


def _hard_chain(nights, seed0=0):
    """Carry the episodes of the cast through many nights without playing them: (person-nights laid up, person-nights, episode lengths)."""
    w = World(seed=0, flags=HARD)
    carry, lay, lens, cur = None, 0, [], {}
    for n in range(nights):
        w.laid = {}
        w._episodes(seed0 + n, carry)
        for name in w.people:
            if name in w.laid:
                lay += 1
                cur[name] = cur.get(name, 0) + 1
            elif name in cur:
                lens.append(cur.pop(name))
        carry = {name: {"ill": (max(0, w.laid.get(name, 0) - 1), w.people[name].ail)} for name in w.people}
    return lay, nights * len(w.people), lens


def test_hard_episode_length_law_hits_the_source_shares():
    pm = W.hard_duration_pmf()
    assert abs(sum(pm) - 1) < 1e-6
    mean = sum((k + 1) * q for k, q in enumerate(pm))
    assert abs(mean - 8.7) < 0.01, mean                                   # Gurven 2012: mean 8.7 days
    assert abs(sum(pm[3:]) - 0.91) < 0.003, sum(pm[3:])                   # 91 percent over 3 days
    assert abs(sum(pm[29:]) - 0.13) < 0.003, sum(pm[29:])                 # 13 percent a month or more (Sugiyama 2004)


def test_hard_prevalence_comes_out_at_ten_percent_and_episodes_follow_the_law():
    lay, tot, lens = _hard_chain(30000)
    assert abs(lay / tot - 0.10) < 0.012, lay / tot                       # prevalence ~ 10 percent of days
    assert 0.011 < W.hard_onset() < 0.014
    assert abs(sum(lens) / len(lens) - 8.7) < 1.0, sum(lens) / len(lens)  # mean length of the episodes that began and ended
    assert abs(sum(1 for x in lens if x > 3) / len(lens) - 0.91) < 0.03
    assert abs(sum(1 for x in lens if x >= 30) / len(lens) - 0.13) < 0.03


def test_hard_catch_is_as_lumpy_as_the_field():
    cs = []
    for s in range(1200):
        w = World(seed=s, flags=HARD)
        cs += [k for n, k in w.catch.items() if n not in w.laid]
    m = sum(cs) / len(cs)
    cv = (sum((c - m) ** 2 for c in cs) / len(cs)) ** 0.5 / m
    empty = sum(1 for c in cs if c == 0) / len(cs)
    assert 0.45 <= empty <= 0.50, empty                                   # Tsimane 45 percent, Ache about half
    assert 1.1 <= cv <= 1.6, cv                                           # person-day CV 1.1 (Ache) to 1.6 (Tsimane households)
    assert abs(m - 2.5) < 0.2, m                                          # the expected catch is unchanged


def test_hard_raids_are_rare_from_the_violent_death_rates():
    assert abs(W.HARD_RAID - 0.01 / (365 * 0.011)) < 1e-12 and W.HARD_RAID < 0.003
    n = sum(World(seed=s, flags=HARD, raid=1.0)._raid_at is not None for s in range(1500))
    assert n < 30, n                                                      # the dial given (1.0) is ignored: about 0.25 percent of nights


def test_hard_laid_up_cannot_work_move_or_catch_but_can_ask_and_be_fed():
    w = World(seed=3, flags=HARD)
    tom, fenn = w.people["Tom"], w.people["Fenn"]
    tom.room = fenn.room
    w.laid["Tom"] = 5
    tom.ail = 0.5
    verbs = {o["verb"] for o in w.options(tom)}
    assert not verbs & (set(W.EXERT) | {"work", "buy"}), verbs
    texts = [o["text"] for o in w.options(tom)]
    assert any(t.startswith("ask Fenn") for t in texts) and "sleep" in texts
    assert any(o["text"] == "tend Tom" for o in w.options(fenn))          # the others can tend it
    assert {"walk to", "tend"} & {o["verb"] for o in w.options(fenn)}      # and the healthy still can move and tend
    for s in range(300):
        w = World(seed=s, flags=HARD)
        assert all(w.catch.get(n, 0) == 0 for n in w.laid)                 # a laid-up person brings no catch


def test_hard_an_episode_goes_on_across_nights_with_its_level_and_ends():
    w = World(seed=5, flags=HARD)
    w.laid = {"Tom": 3}
    w.people["Tom"].ail = 0.7
    c = w.carry_out()
    assert c["Tom"]["ill"] == (2, 0.7) and c["Fenn"]["ill"][0] == 0
    w2 = World(seed=6, flags=HARD, carry=c)
    assert w2.laid["Tom"] == 2 and w2.people["Tom"].ail == 0.7
    w3 = World(seed=7, flags=HARD, carry={**w2.carry_out(), "Tom": {**w2.carry_out()["Tom"], "ill": (0, 0.9)}})
    assert "Tom" not in w3.laid or w3.people["Tom"].ail != 0.9              # at 0 nights left it is over (a new onset draws a fresh level)


def test_hard_flag_off_changes_nothing():
    a = World(seed=4, flags=W.ALL6, turns=60)
    b = World(seed=4, flags=W.ALL6, turns=60)
    assert a.laid == {} and not a.hard
    for _ in range(60):
        a.step(Rules())
        b.step(Rules())
    assert [e["text"] for e in a.log] == [e["text"] for e in b.log]
    assert "ill" not in a.carry_out()["Tom"]
    assert any(p.ail > 0 for s in range(30) for p in World(seed=s, flags=W.ALL6).people.values())   # the old one-night ailment is still there


# ------------------------------------------------------------------ lethal: a need held at its extreme kills (2026-11-21, owner D1 (c))

class _Idle:
    """Chooses 'wait' every time (a person who cannot or does not reach water or food)."""
    name = "idle"

    def probs(self, w, p, opts):
        pc = [1.0 if o["verb"] == "wait" else 0.0 for o in opts]
        return pc, pc


def _idle_days(w, who, key, days, cap=None):
    """Hold `who` at the extreme of need `key` (the person never drinks or eats); step until it dies. Returns the days it took (or None)."""
    p = w.people[who]
    for t in range(int(days * W.LETHAL_TURNS)):
        setattr(p, key, 1.0)
        setattr(p, "hunger" if key == "thirst" else "thirst", 0.0)      # only this need is at its extreme
        w.step(_Idle())
        if p.dead:
            return (t + 1) / W.LETHAL_TURNS
    return None


def test_lethal_thirst_and_hunger_kill_at_the_source_rates():
    for key, total in (("thirst", 5.0), ("hunger", 45.0)):
        w = World(seed=3, flags=("lethal",), turns=10 ** 6)
        d = _idle_days(w, "Tom", key, total + 2)
        assert d is not None and abs(d - W.LETHAL_RESERVE[key]) < 0.02, (key, d)
        assert w.people["Tom"].cause == key
        # the time from a usual level to the cap plus the reserve is the source figure
        assert abs(W.LETHAL_RESERVE[key] + (1 - W.LETHAL_START) / (W.LETHAL_DRIFT[key] * W.LETHAL_TURNS) - total) < 1e-9
    assert 3 <= W.LETHAL_DAYS["thirst"] <= 7 and 30 <= W.LETHAL_DAYS["hunger"] <= 60


def test_lethal_off_nothing_dies_and_nothing_carries():
    w = World(seed=3, flags=(), turns=10 ** 6)
    assert _idle_days(w, "Tom", "thirst", 8) is None
    assert "body" not in w.carry_out()["Tom"] and "gone" not in w.carry_out()["Tom"]


def test_lethal_the_reserve_comes_back_away_from_the_extreme_and_ill_is_faster():
    w = World(seed=3, flags=("lethal",), turns=10 ** 6)
    p = w.people["Tom"]
    for _ in range(int(2 * W.LETHAL_TURNS)):          # two days at the extreme
        p.thirst = 1.0
        w.step(_Idle())
    used = W.LETHAL_RESERVE["thirst"] - p.reserve["thirst"]
    assert abs(used - 2.0) < 0.02
    for _ in range(int(1 * W.LETHAL_TURNS)):          # a day with water in reach: a day back
        p.thirst = 0.2
        w.step(_Idle())
    assert abs((W.LETHAL_RESERVE["thirst"] - p.reserve["thirst"]) - 1.0) < 0.03
    ill = World(seed=3, flags=("lethal",), turns=10 ** 6)
    ill.people["Tom"].ail = 1.0
    d = _idle_days(ill, "Tom", "thirst", 6)
    assert abs(d - W.LETHAL_RESERVE["thirst"] / 2.0) < 0.1, d      # a full ailment doubles the rate (u)


def test_lethal_the_body_carries_and_a_laid_up_person_nobody_feeds_dies():
    FL = tuple(sorted(W.ALL6)) + ("signs", "hard", "lethal", "perma")
    w = World(seed=5, flags=FL, turns=300)
    w.people["Tom"].thirst = 1.0
    w.people["Tom"].reserve["thirst"] = 1.5
    c = w.carry_out()
    assert c["Tom"]["body"]["reserve"]["thirst"] == 1.5 and c["Tom"]["body"]["thirst"] == 1.0
    w2 = World(seed=6, flags=FL, carry=c)
    assert w2.people["Tom"].thirst == 1.0 and w2.people["Tom"].reserve["thirst"] == 1.5 and not w2.people["Tom"].dead


def test_perma_a_death_is_final_and_without_it_the_dead_come_back():
    for perma in (True, False):
        FL = ("lethal", "perma") if perma else ("lethal",)
        w = World(seed=5, flags=FL, turns=300)
        w.people["Tom"].dead = w.people["Tom"].down = True
        w.people["Tom"].thirst, w.people["Tom"].reserve["thirst"] = 1.0, -0.1
        c = w.carry_out()
        w2 = World(seed=6, flags=FL, carry=c)
        assert w2.people["Tom"].dead == perma and ("Tom" in w2.gone) == perma
        if perma:
            assert w2.carry_out()["Tom"]["gone"]
            for _ in range(5):
                w2.step(Rules())
            assert w2.people["Tom"].dead and w2.people["Tom"].doing is None
        else:
            assert w2.people["Tom"].reserve["thirst"] == W.LETHAL_RESERVE["thirst"]      # back afresh


def test_lethal_off_the_old_inn_is_unchanged():
    a = World(seed=4, flags=W.ALL6, turns=60)
    assert not a.lethal and a.gone == set()
    for _ in range(60):
        a.step(Rules())
    assert "body" not in a.carry_out()["Tom"]


# ------------------------------------------------------------------ lethal2: bring any consumable, the raid law, one self-care (2026-11-22)
L2 = tuple(sorted(set(W.ALL6) | set(W.SIGNS_FLAGS) | set(W.HARD_FLAGS) | set(W.LETHAL_FLAGS) | set(W.LETHAL2_FLAGS)))


def _laid_tom(flags, room="the main room"):
    w = World(seed=3, flags=flags, turns=300)
    tom, fenn = w.people["Tom"], w.people["Fenn"]
    tom.room = fenn.room = room
    w.laid["Tom"] = 5
    tom.ail, tom.thirst = 0.5, 1.0
    for q in w.people.values():
        q.doing = None
    return w, tom, fenn


def test_bring_a_drink_goes_to_the_laid_up_as_one_act_with_the_trip_paid():
    w, tom, fenn = _laid_tom(L2)
    br = [o for o in w.options(fenn) if o["verb"] == "serve" and "drink" in o["text"]]
    well = [o for o in br if o["obj"] == "the well"]
    assert well and well[0]["target"] == "Tom" and well[0]["dur"] == W.DUR["serve"] + 2 * 1 * W.DUR["walk to"]   # the well is one room away: there and back
    bucket = [o for o in br if o["obj"] == "the bucket of water"]
    assert bucket and bucket[0]["dur"] == W.DUR["serve"] + 2 * 1 * W.DUR["walk to"]                            # the kitchen is one room away
    assert any(o["obj"] == "the barrel of ale" and o["dur"] == W.DUR["serve"] + 2 * 2 * W.DUR["walk to"] for o in br)   # the cellar is two
    ok, _ = w._apply(fenn, {**well[0], "left": 0})
    assert ok and abs(tom.thirst - 0.5) < 1e-9
    tom.thirst = 1.0
    assert not [o for o in w.options(tom) if o["verb"] == "serve"]       # the laid up cannot carry from the well (but can ask)
    assert any(o["text"] == "ask Fenn for a drink from the well" and o["f"]["need"] == 1.0 for o in w.options(tom))
    w2, tom2, fenn2 = _laid_tom(L2, "the kitchen")                       # the same laid-up person carries what is in the room
    assert any(o["verb"] == "serve" and o["obj"] == "the bucket of water" for o in w2.options(tom2)) is False   # not to itself
    w2.laid["Fenn"] = 3
    assert any(o["verb"] == "serve" and o["obj"] == "the bucket of water" and o["dur"] == W.DUR["serve"] for o in w2.options(fenn2))
    assert not any(o["obj"] == "the well" for o in w2.options(fenn2) if o["verb"] == "serve")


def test_bring_serves_only_a_person_who_cannot_fetch_and_is_thirsty():
    w, tom, fenn = _laid_tom(L2)
    tom.thirst = 0.05
    assert not [o for o in w.options(fenn) if "drink" in o["text"]]       # not thirsty
    tom.thirst = 1.0
    del w.laid["Tom"]
    assert not [o for o in w.options(fenn) if "drink" in o["text"]]       # can fetch it itself


def test_bring_off_nothing_is_added_and_a_meal_still_serves():
    off = tuple(f for f in L2 if f != "bring")
    w, tom, fenn = _laid_tom(off)
    assert not [o for o in w.options(fenn) if "drink" in o["text"] and o["verb"] in ("serve", "ask")]
    assert not [o for o in w.options(tom) if "drink" in o["text"] and o["verb"] == "ask" and o["obj"] in ("the well", "the barrel of ale")]
    w, tom, fenn = _laid_tom(L2)
    pot = next(n for n in w.portions if n in W.COOK_MAKES)
    w.things[pot].update(at="Fenn", owner="Fenn")
    tom.hunger = 0.9
    assert any(o["verb"] == "serve" and o["obj"] == pot for o in w.options(fenn))      # the meal path is the same act


def test_raidlaw_the_fallen_are_not_struck_again_and_the_raid_goes_when_none_stand():
    for law in (True, False):
        fl = L2 if law else tuple(f for f in L2 if f != "raidlaw")
        w = World(seed=3, flags=fl, turns=300)
        w._raid_arrives()
        for q in w.people.values():
            q.room = "the yard"
        w.people["Tom"].hp, w.people["Tom"].down = 2.0, True
        for _ in range(30):
            if w.threat is None:
                break
            w._raid_round()
        hits = [e for e in w.log if e["action"] == "raid hits"]
        fallen = [e for e in hits if e["target"] == "Tom" and "knocked" not in e["result"]]
        assert (not fallen) if law else fallen, (law, len(fallen))
        w2 = World(seed=3, flags=fl, turns=300)
        w2._raid_arrives()
        for q in w2.people.values():
            q.room, q.down = "the yard", True
        w2._raid_round()
        assert (w2.threat is None) == law                                 # nobody left to fight: they go (law) / they stay (before)
    assert abs(W.RAID_KILL - 0.055) < 1e-12


def test_raidlaw_deaths_per_raid_follow_the_source_per_raid():
    kills, n = 0, 3000
    for s in range(n):
        w = World(seed=s, flags=L2, turns=300)
        w._raid_arrives()
        kills += bool(w.threat["mortal"])
    assert abs(kills / n - 0.055) < 0.02, kills / n                      # 0.055 a raid, whatever the blows
    w = World(seed=1, flags=L2, turns=300)
    w._raid_arrives()
    w.threat["mortal"] = True
    for q in w.people.values():
        q.room, q.hp = "the yard", 10.0
    w._raid_round()
    assert w.log[-1]["result"].startswith("killed") and not w.threat["mortal"]  # the first blow kills, once


def test_selfcare_is_one_shared_rule_not_a_lethal_patch():
    from lab.world2.policy import self_care
    for fl in (L2, tuple(f for f in L2 if f != "lethal")):                # no lethal needed: the rule is the flag
        w, tom, fenn = _laid_tom(fl, "the kitchen")
        fenn.hunger = 0.9
        opts = w.options(fenn)
        pc = P.background(w, fenn, opts)
        care = [i for i, o in enumerate(opts) if self_care(o)]
        assert care and all(pc[i] > 0 for i in care), "a hungry partner asks, buys or works for its own food"
        assert not any(pc[i] > 0 for i, o in enumerate(opts) if o["verb"] in ("strike", "insult", "give", "praise"))   # and starts nothing else
    off = tuple(f for f in L2 if f != "selfcare")
    w, tom, fenn = _laid_tom(off, "the kitchen")
    fenn.hunger = 0.9
    opts = w.options(fenn)
    pc = P.background(w, fenn, opts)
    assert not any(pc[i] > 0 for i, o in enumerate(opts) if o["verb"] in ("ask", "buy", "work", "offer"))      # flag off: as before


if __name__ == "__main__":
    n = 0
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            n += 1
    print(f"{n} passed")
