"""Tests for the step 0a tournament (lab/mindL/tournament.py). Plain asserts, as in test_mind.py.

    lab/.venv/bin/python -m lab.mindL.test_tournament

The archetypes must pick acts only from their stance and share one self-care; the measured lifetime must be the
lifetime of lab/world2/lifetime.py; the four worlds must switch the flags they say they switch.
"""
from __future__ import annotations

import hashlib
import json
import time

import lab.mindL.tournament as T
from lab.world2.lifetime import NAMES, Cast, lifetime
from lab.world2.policy import Rules
from lab.world2.world import FLAGS, World

PERSON_VERBS = {"strike", "shove", "threaten", "insult", "praise", "search", "steal", "bandage", "help up", "defend",
                "accuse", "tell", "give", "gossip", "ask", "offer", "accept", "refuse"}
SELF_CARE = {"eat", "drink", "sit down", "sleep", "look around", "wait", "drop", "hide", "play", "shout for help"}


def scene(seed=3, flags=FLAGS, night=0, rec=None):
    """Everyone in the main room, no bandit, nothing under way."""
    w = T.MWorld(seed=seed, bandit=False, flags=flags, night=night, rec=rec)
    for p in w.people.values():
        p.room, p.doing, p.pending = "the main room", None, []
    w.bandit_at = None
    w.t = 10
    return w


def act(w, name, text):
    """Make `name` do the listed option `text` now."""
    p = w.people[name]
    o = next((o for o in w.options(p) if o["text"] == text), None)
    assert o is not None, (text, [x["text"] for x in w.options(p)])
    p.doing = {**o, "left": 0, "choice": {"options": [o["text"]], "chosen": 0, "p_char": [1.0], "p_avg": [1.0],
                                           "seen": []}}                        # a choice, as the log marks one
    w._finish(p)
    return next(e for e in reversed(w.log) if e["actor"] == name and e["text"] == text)


def stance_texts(arch, w, opts, pc):
    """The person-directed acts (outside the shared self-care) that get probability."""
    return [o["text"] for x, o in zip(pc, opts) if x > 0 and arch.is_social(w, o)]


def is_food_request(w, e):
    thing = e["parts"].get("thing")
    return e["action"] in ("ask", "offer") and thing in w.things and bool(w.things[thing]["props"] & {"food", "ale", "water"})


# ------------------------------------------------------------------ the worlds
def test_the_four_worlds_switch_the_flags_they_say():
    assert T.WORLDS["all2"] == frozenset(FLAGS)
    assert frozenset(FLAGS) - T.WORLDS["noR"] == {"reprisal"}
    assert frozenset(FLAGS) - T.WORLDS["noG"] == {"gossip", "watch", "check"}
    assert frozenset(FLAGS) - T.WORLDS["none"] == {"reprisal", "gossip", "watch", "check"}
    for name, fl in T.WORLDS.items():
        assert World(flags=tuple(sorted(fl))).flags == fl, name


# ------------------------------------------------------------------ the measured lifetime is the lifetime
def test_fast_rules_chooses_exactly_as_rules():
    def digest(pol):
        w = World(seed=5, bandit=True, flags="all2")
        for _ in range(300):
            w.step(pol)
        ev = [{k: v for k, v in e.items() if k != "p_avg"} for e in w.log]
        return hashlib.sha256(json.dumps(ev, sort_keys=True).encode()).hexdigest(), len(w.log)
    assert digest(Rules()) == digest(T.FastRules())


def test_the_recorder_changes_nothing_in_the_lifetime():
    old = T.WIN0
    T.WIN0 = 1                                              # record from the second night on
    try:
        for world in ("all2", "none"):
            spec = dict(kind="A", world=world, chooser="rules", seed=2, nights=4, focal="Oren", id="x")
            stats, rec = T.lifetime_measured(spec)
            flags = tuple(sorted(T.WORLDS[world]))
            _, ref = lifetime(Cast(T.FastRules(), {}), 4, 2, flags=flags, fade=T.FADE)
            for a, b in zip(stats, ref):
                assert a["drive_error"] == b["drive_error"] and a["deaths"] == b["deaths"] and a["acts"] == b["acts"]
            assert rec.agg, "nothing recorded"
    finally:
        T.WIN0 = old


# ------------------------------------------------------------------ the archetypes
def test_each_archetype_takes_only_acts_of_its_stance():
    for name, cls in T.ARCHETYPES.items():
        for seed in (1, 2):
            pol = cls(seed=seed)
            focal = NAMES[seed % 5]
            cast = Cast(T.FastRules(), {focal: pol})
            w = World(seed=seed, bandit=True, flags="all2")
            for _ in range(300):
                w.step(cast)
            mine = [e for e in w.log if e["actor"] == focal and "options" in e and e["target"] in w.people]
            for e in mine:
                v = e["action"]
                if name == "hermit":
                    raise AssertionError(f"the strict hermit made a person-directed act: {e['text']}")
                assert v in cls.allowed or is_food_request(w, e), (name, e["text"])
            if name in ("hermit_ask", "feeder"):
                assert all(is_food_request(w, e) for e in mine), name
            if name == "hoarder":
                assert all(e["action"] in ("refuse",) or is_food_request(w, e) for e in mine)
            if name == "always-harsh":
                assert {e["action"] for e in mine if not is_food_request(w, e)} <= {"insult", "threaten", "shove", "refuse"}


def _scenes():
    """Menus that exercise every stance: a quiet room, an attack in view, the focal struck, someone down, a request."""
    out = []
    w = scene()
    out.append(("quiet", w, "Tom"))
    w = scene()
    act(w, "Fenn", "strike Oren")                              # an attack in view
    out.append(("attack", w, "Tom"))
    w = scene()
    act(w, "Fenn", "insult Tom")                               # the focal is insulted
    out.append(("insulted", w, "Tom"))
    w = scene()
    act(w, "Fenn", "praise Tom")
    out.append(("praised", w, "Tom"))
    w = scene()
    w.people["Oren"].down, w.people["Oren"].hp = True, 2.0     # someone on the floor
    out.append(("down", w, "Sir Ada"))
    w = scene()
    act(w, "Fenn", "ask Marta for the loaf of bread")          # a request to the cook
    w.people["Marta"].hunger = 0.1
    out.append(("asked", w, "Marta"))
    w = scene()
    w.people["Tom"].hunger = 0.8
    out.append(("hungry", w, "Tom"))
    return out


def test_self_care_is_the_same_for_every_archetype_and_stances_are_closed():
    for label, w, focal in _scenes():
        p = w.people[focal]
        opts = w.options(p)
        rules, _ = T.FastRules().probs(w, p, opts)
        for name, cls in T.ARCHETYPES.items():
            arch = cls(seed=1)
            pc, pa = arch.probs(w, p, opts)
            assert abs(sum(pc) - 1.0) < 1e-9 and min(pc) >= 0.0 and pc == pa, (label, name)
            care = [i for i, o in enumerate(opts) if o["verb"] in SELF_CARE and not arch.is_social(w, o)]
            assert len(care) >= 2
            for i in care[1:]:                                 # the same ratios as rules among the self-care acts
                assert abs(pc[i] * rules[care[0]] - pc[care[0]] * rules[i]) < 1e-9, (label, name, opts[i]["text"])
            for i, o in enumerate(opts):                       # and no person-directed act outside the stance
                if arch.is_social(w, o) and pc[i] > 0:
                    assert o["verb"] in cls.allowed, (label, name, o["text"])
                if name == "hermit" and o["target"] in w.people:
                    assert pc[i] == 0.0, (label, o["text"])


def test_the_strict_hermit_does_not_even_ask_for_food_and_leaves_a_crowded_room():
    w = scene()
    p = w.people["Tom"]
    p.hunger = 0.9
    opts = w.options(p)
    pc, _ = T.Hermit().probs(w, p, opts)
    assert any(o["verb"] == "ask" and o["f"]["need"] > 0 for o in opts)
    assert sum(x for x, o in zip(pc, opts) if o["verb"] in ("ask", "offer")) == 0.0
    walk = sum(x for x, o in zip(pc, opts) if o["verb"] == "walk to" and o["f"]["goal"]["company"] == 0)
    assert walk >= T.R_AVOID                                   # the yard is empty: it leaves with that chance at least
    pc2, _ = T.HermitAsk().probs(w, p, opts)
    assert sum(x for x, o in zip(pc2, opts) if o["verb"] == "ask") > 0


def test_tit_for_tat_answers_with_the_same_act_and_opens_kind():
    w = scene()
    tom = w.people["Tom"]
    tft = T.TitForTat(seed=1)
    opts = w.options(tom)
    pc, _ = tft.probs(w, tom, opts)                            # nobody has done anything: open with praise
    moved = set(stance_texts(tft, w, opts, pc))
    assert moved == {"praise Sir Ada", "praise Oren", "praise Fenn", "praise Marta"}, moved
    assert abs(sum(x for x, o in zip(pc, opts) if o["verb"] == "praise") - T.R_SPONT) < 0.05
    act(w, "Fenn", "insult Tom")
    opts = w.options(tom)
    pc, _ = tft.probs(w, tom, opts)
    hit = stance_texts(tft, w, opts, pc)
    assert hit == ["insult Fenn"], hit
    assert abs(sum(x for x, o in zip(pc, opts) if o["verb"] == "insult") - T.R_REACT) < 1e-9
    act(w, "Tom", "insult Fenn")                               # answered: nothing is owed now
    act(w, "Oren", "praise Tom")
    opts = w.options(tom)
    pc, _ = tft.probs(w, tom, opts)
    assert stance_texts(tft, w, opts, pc) == ["praise Oren"]
    act(w, "Tom", "praise Oren")                               # answered too
    act(w, "Marta", "strike Tom")                              # strike with strike
    opts = w.options(tom)
    pc, _ = tft.probs(w, tom, opts)
    assert stance_texts(tft, w, opts, pc) == ["strike Marta"]


def test_the_extractor_threatens_and_then_takes():
    w = scene()
    fenn = w.people["Fenn"]
    ex = T.Extractor(seed=1)
    opts = w.options(fenn)
    pc, _ = ex.probs(w, fenn, opts)
    acts = {o.split()[0] for o in stance_texts(ex, w, opts, pc)}
    assert acts == {"threaten"}, acts                          # nothing to take from an awake one yet
    act(w, "Fenn", "threaten Oren")
    w.t += 5
    opts = w.options(fenn)
    pc, _ = ex.probs(w, fenn, opts)
    taken = {t for t in stance_texts(ex, w, opts, pc) if t.startswith("steal")}
    assert taken == {"steal the purse of silver from Oren", "steal the lute from Oren"}, taken
    assert not any(x > 0 and opts[i]["verb"] == "steal" and opts[i]["target"] != "Oren" for i, x in enumerate(pc))
    w.t += T.THREAT_SPAN + 5                                   # the threat is old: threaten again, do not take
    opts = w.options(fenn)
    pc, _ = ex.probs(w, fenn, opts)
    assert {t.split()[0] for t in stance_texts(ex, w, opts, pc)} == {"threaten"}


def test_the_protector_defends_and_stops_the_attacker_only_when_there_is_an_attack():
    w = scene()
    tom = w.people["Tom"]
    prot = T.Protector(seed=1)
    opts = w.options(tom)
    pc, _ = prot.probs(w, tom, opts)
    assert stance_texts(prot, w, opts, pc) == []                                   # peace: no social act
    act(w, "Fenn", "strike Oren")
    opts = w.options(tom)
    pc, _ = prot.probs(w, tom, opts)
    got = set(stance_texts(prot, w, opts, pc))
    assert got == {"defend Oren", "threaten Fenn", "strike Fenn"}, got


def test_the_hoarder_gathers_and_never_gives():
    w = scene()
    w.people["Tom"].room = "the cellar"
    tom = w.people["Tom"]
    hd = T.Hoarder(seed=1)
    opts = w.options(tom)
    pc, _ = hd.probs(w, tom, opts)
    assert sum(x for x, o in zip(pc, opts) if o["verb"] == "give") == 0.0
    assert sum(x for x, o in zip(pc, opts) if o["verb"] == "take" and o["f"]["gain"] > 0) > 0   # the lantern


def test_oracle_never_provokes_the_hard_partner_and_works_the_kind_one():
    w = scene()
    ada = w.people["Sir Ada"]
    ada.lonely = 0.9
    ada.aff["Fenn"] = 0.0
    orc = T.Oracle({"Fenn": "kind", "Oren": "hard"})
    opts = w.options(ada)
    pc, _ = orc.probs(w, ada, opts)
    assert abs(sum(pc) - 1) < 1e-9
    for x, o in zip(pc, opts):
        if o["target"] == "Oren" and o["verb"] in T.PROVOKE:
            assert x == 0.0, o["text"]
    asks = sum(x for x, o in zip(pc, opts) if o["verb"] == "ask" and o["target"] == "Fenn")
    assert asks > 0.05
    blind, _ = T.FastRules().probs(w, ada, opts)
    assert any(x > 0 for x, o in zip(blind, opts) if o["target"] == "Oren" and o["verb"] in T.PROVOKE)
    # the avoid-only oracle leaves everything else as rules has it
    av, _ = T.Oracle({"Fenn": "kind", "Oren": "hard"}, kind=False).probs(w, ada, opts)
    for a, b, o in zip(av, blind, opts):
        if not (o["target"] == "Oren" and o["verb"] in T.PROVOKE):
            assert a >= b - 1e-12


# ------------------------------------------------------------------ the measures
def test_b_and_c_are_the_error_change_around_the_act():
    rec = T.Recorder("Marta")
    w = scene(night=T.WIN0, rec=rec)
    tom, oren = w.people["Tom"], w.people["Oren"]
    tom.lonely = 0.9
    e0, o0 = w.drive_error(tom), w.drive_error(oren)
    act(w, "Oren", "praise Tom")
    r = rec.acts[-1]
    assert r["verb"] == "praise" and r["a"] == "Oren" and r["q"] == "Tom" and r["done"]
    assert abs(r["b"] - (e0 - w.drive_error(tom))) < 1e-12 and r["b"] > 0.05, r
    assert abs(r["c"] - (w.drive_error(oren) - o0)) < 1e-12
    assert r["present"] == 3 and r["seen"] == 3                # Ada, Fenn and Marta saw it
    # a gift of the purse costs the merchant his wealth
    o1 = w.drive_error(oren)
    act(w, "Oren", "give the purse of silver to Tom")
    r = rec.acts[-1]
    assert r["c"] > 0.3 and abs(r["c"] - (w.drive_error(oren) - o1)) < 1e-12, r
    # a loaf given to someone hungry could be eaten: the use is counted on both sides
    w.people["Marta"].room = "the main room"
    w.people["Tom"].hunger = 0.9
    act(w, "Marta", "give the loaf of bread to Tom")
    r = rec.acts[-1]
    assert r["bpot"] > 0.3 and r["cpot"] >= 0.0, r


def test_a_harmful_act_is_seen_and_the_witnesses_think_less_of_the_actor():
    rec = T.Recorder("Marta")
    w = scene(night=T.WIN0, rec=rec)
    act(w, "Fenn", "insult Tom")
    r = rec.acts[-1]
    assert r["seen"] == 3 and r["daff"] < -0.1, r              # reprisal: each witness's liking falls by WITNESS_COST
    w2 = scene(night=T.WIN0, rec=T.Recorder("Marta"), flags=tuple(sorted(T.WORLDS["noR"])))
    act(w2, "Fenn", "insult Tom")
    assert abs(w2.rec.acts[-1]["daff"]) < abs(r["daff"])


def test_a_witness_who_likes_the_victim_thinks_better_of_the_helper():
    rec = T.Recorder("Marta")
    w = scene(night=T.WIN0, rec=rec)
    w.people["Tom"].hp = 5.0
    for name in ("Fenn", "Marta"):
        w.people[name].aff["Tom"] = -0.5                       # the others do not like Tom
    w.people["Sir Ada"].aff["Tom"] = 0.5                       # Ada likes Tom, and watches Oren bandage him
    w.people["Sir Ada"].aff["Oren"] = 0.0
    act(w, "Oren", "bandage Tom")
    r = rec.acts[-1]
    assert r["verb"] == "bandage" and r["seen"] == 3 and r["b"] > 0, r
    assert abs(r["daff"] - 0.1) < 1e-9, r                      # only Ada's liking of Oren moved (+0.1)
    rec.end_night(w, T.WIN0, 0.5)                              # the end-of-night pass marks the reply fields
    assert rec.pending and rec.pending[-1]["rk_d"] is False


def test_a_run_writes_one_reproducible_file(tmp=None):
    import tempfile
    from pathlib import Path
    old, old_runs = T.WIN0, T.RUNS
    T.WIN0 = 1
    try:
        with tempfile.TemporaryDirectory() as d:
            T.RUNS = Path(d)
            spec = dict(kind="O", world="all2", chooser="oracle_both", seed=1, nights=3, focal="Sir Ada", id="t1")
            T.run(spec)
            a = json.loads((Path(d) / "t1.json").read_text())
            (Path(d) / "t1.json").unlink()
            T.run(spec)
            b = json.loads((Path(d) / "t1.json").read_text())
            a.pop("cpu_s"), b.pop("cpu_s")
            assert a == b and len(a["err"]) == 3 and a["agg"]
    finally:
        T.WIN0, T.RUNS = old, old_runs


if __name__ == "__main__":
    n = 0
    t0 = time.perf_counter()
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            n += 1
    print(f"{n} passed in {time.perf_counter() - t0:.1f} s")
