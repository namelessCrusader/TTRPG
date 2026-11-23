"""The types of the mind interface and its verb catalog (docs/sim-mind-design.md
3.5, step M1).

The catalog is data. Most errors in data are typos that nothing notices until
a verb is never offered, or is offered with the wrong words. So these tests
do three things: they hold the shipped file to every rule of the loader, they
show that each rule catches the error it is for, and they hold the file to the
inn that the rooms backend will fork (lab/world2/world.py), word for word.

The lab file is READ here, never imported: lab and src do not import each
other.
"""
import ast
import copy
import dataclasses
import pickle
import re
import sys
from pathlib import Path

import pytest
import yaml

from src.mind import api, catalog
from src.mind.api import (Backend, Belief, Chooser, Decision, DecisionRequest, Job, JobEvent, Menu,
                          Option, Percept, Ref, SelfState, Why)
from src.mind.catalog import Catalog, CatalogError, load

ROOT = Path(__file__).resolve().parents[1]
INN = ROOT / "lab" / "world2" / "world.py"


def _raw():
    with open(catalog.PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _verb(raw, vid):
    return next(v for v in raw["verbs"] if v["id"] == vid)


# ---------------------------------------------------------------- the file
def test_the_SHIPPED_catalog_passes_every_check():
    c = load()
    assert c.null_act.id == "wait"
    assert c.backends == ("rooms", "voxel")
    # (wave 15, S3 item 1: the door and mechanism verbs run on voxel only,
    # except `open`, which the inn also has)
    assert set(c.runnable("voxel")) == {"walk_to", "flee_to", "take", "drop", "strike",
                                        "shout_fire", "wait", "open", "close_door",
                                        "use_key", "pick_lock", "bar_door",
                                        "unbar_door", "kick_in"}
    # a verb with no program anywhere is in the catalog for its words, and is never offered
    assert "cut" not in c.runnable("rooms") + c.runnable("voxel")


def _set(path, value):
    """A change to the raw catalog: put `value` at `path` (a key, or a verb id
    first). `value` DEL removes the key."""
    def change(raw):
        node = raw
        keys = list(path)
        if keys[0] == "verb":
            node = _verb(raw, keys[1])
            keys = keys[2:]
        for k in keys[:-1]:
            node = node[k]
        if value is DEL:
            del node[keys[-1]]
        else:
            node[keys[-1]] = value
    return change


DEL = object()


def _add_null_key(raw):
    v = _verb(raw, "wait")
    del v["null_act"]
    v.update(yaml.safe_load("null: true"))      # YAML reads the key `null` as None


def _dup_take(raw):
    raw["verbs"].append(copy.deepcopy(_verb(raw, "take")))


def _drop_search(raw):
    raw["verbs"].remove(_verb(raw, "search"))


def _no_edge_floor(raw):
    del raw["capabilities"]["sharp"]["voxel"]["edge_gt_m2"]


def _no_hit_test(raw):
    raw["capabilities"]["sharp"]["voxel"] = None
    raw["capabilities"]["heavy"]["voxel"] = None


RULES = [
    # the checks of check_catalog_v2.py
    ("a template names a role that is not declared", _set(("verb", "take", "text"), "take {thing}"),
     "uses ['thing'], which are not roles"),
    ("a verb that runs has no told report", _set(("verb", "take", "told"), DEL), "no told report"),
    ("a verb that runs has no duration there", _set(("verb", "walk_to", "duration", "rooms"), DEL),
     "has no rooms duration"),
    ("an unknown capability", _set(("verb", "take", "needs"), ["free_hands"]),
     "unknown capability 'free_hands'"),
    ("a capability with no words", _set(("capabilities", "sharp", "says"), DEL), "has no words"),
    ("a need with no test on a backend that runs the verb",
     _set(("capabilities", "free_hand", "voxel"), None), "needs 'free_hand', which has no voxel test"),
    ("an unknown drive", _set(("verb", "eat", "serves"), ["hungry"]), "unknown drive 'hungry'"),
    ("an unknown flag", _set(("verb", "strike", "flags"), ["loud", "noisy"]), "unknown flag 'noisy'"),
    ("a rooms verb the inn does not have", _set(("verb", "take", "backends", "rooms"), {"verb": "grab"}),
     "the program is not {verb: <one of the rooms verbs>}"),
    ("no such limb", _set(("verb", "walk_to", "backends", "voxel", 0, "limb"), "feet"),
     "no such limb 'feet'"),
    ("an unknown step test", _set(("verb", "take", "backends", "voxel", 0, "until"), "grabbed"),
     "unknown step test 'grabbed'"),
    ("an unknown field", _set(("verb", "take", "when"), "always"), "unknown field 'when'"),
    ("an edge test that admits a blunt edge", _no_edge_floor, "admits edge=0 (blunt)"),
    # the null act
    ("a key written null", _add_null_key, "a key named null reads as None in YAML"),
    ("no null act", _set(("verb", "wait", "null_act"), False), "exactly one verb is the null act"),
    ("a null act with roles", _set(("verb", "wait", "roles"), {"place": {"kind": "place"}}),
     "the null act has no roles"),
    ("a null act that a backend cannot run", _set(("verb", "wait", "backends", "voxel"), None),
     "the null act runs on every backend"),
    # the words
    ("a text that leaves out a role", _set(("verb", "give", "text"), "give {thing}"),
     "text uses every role that is not optional"),
    ("an optional role with no text_with", _set(("verb", "strike", "text_with"), DEL),
     "an optional role needs text_with"),
    ("a report with no actor", _set(("verb", "take", "told"), "{target} was taken"),
     "does not name the actor"),
    ("a loud verb with no told_start", _set(("verb", "strike", "told_start"), DEL),
     "a loud or violent verb has a told_start"),
    ("a brace that does not close", _set(("verb", "take", "text"), "take {target"),
     "not a plain template"),
    ("a template that reads an attribute", _set(("verb", "take", "text"), "take {target.name}"),
     "not a plain template"),
    # names and references
    ("an id two times", _dup_take, "the id occurs more than once"),
    ("an unknown group", _set(("verb", "take", "group"), "grab"), "group 'grab' is not in groups"),
    ("an unknown role kind", _set(("verb", "take", "roles", "target", "kind"), "object"),
     "kind 'object' is not one of"),
    ("held_by names no role", _set(("verb", "drop", "roles", "target", "held_by"), "owner"),
     "held_by 'owner' is not self or another role"),
    ("a need that is not of the actor", _set(("verb", "take", "needs"), ["sharp"]),
     "not a capability of the actor"),
    ("a role cap that is of the actor", _set(("verb", "eat", "roles", "target", "cap"), "free_hand"),
     "is a capability of the actor (self), not of a role"),
    ("a role cap with no test where the verb runs", _set(("capabilities", "food", "rooms"), None),
     "role 'target' needs 'food', which has no rooms test"),
    ("a cap_any with no test where the verb runs", _no_hit_test,
     "no capability of role 'instrument' cap_any has a voxel test"),
    ("an unknown interrupt class", _set(("verb", "take", "interrupt_on"), ["harm"]),
     "unknown interrupt class 'harm'"),
    ("an aim that is not a role", _set(("verb", "strike", "backends", "voxel", 0, "aim"), "victim"),
     "aim 'victim' is not a role"),
    ("a limb step that matches no row", _set(("verb", "take", "backends", "voxel", 0, "match"), DEL),
     "a step that asks a limb matches a row"),
    ("a match by label", _set(("verb", "take", "backends", "voxel", 0, "match"), "label"),
     "match 'label' is not one of"),
    ("an unknown backend", _set(("verb", "take", "backends", "mud"), {"verb": "take"}),
     "unknown backend 'mud'"),
    ("a duration of zero", _set(("verb", "take", "duration", "rooms"), 0),
     "not a positive integer or route"),
    ("a commit_after that is not a fraction", _set(("verb", "strike", "commit_after"), 2),
     "commit_after is not a fraction"),
    ("a verb of the inn that no verb runs", _drop_search, "no verb runs its verb 'search'"),
    ("an unknown top-level key", _set(("extra",), 1), "unknown top-level key 'extra'"),
    ("limbs with no step tests", _set(("backends", "voxel", "step_tests"), DEL),
     "limbs and step_tests go together"),
    # the shape of the file
    ("a top-level part that is missing", _set(("drives",), DEL), "no top-level key 'drives'"),
    ("verbs that are not a list", _set(("verbs",), {"wait": {}}), "verbs is not a list of mappings"),
    ("a capability that is not a mapping", _set(("capabilities", "sharp"), "sharp"),
     "capability 'sharp' is not a mapping"),
    ("a backend with no list of its verbs", _set(("backends", "rooms", "verbs"), None),
     "backend 'rooms' verbs is not a list of names"),
]


@pytest.mark.parametrize("what,change,says", RULES, ids=[r[0] for r in RULES])
def test_each_RULE_catches_the_error_it_is_for(what, change, says):
    raw = _raw()
    change(raw)
    with pytest.raises(CatalogError) as e:
        Catalog(raw)
    assert any(says in p for p in e.value.problems), e.value.problems


def test_the_loader_reports_EVERY_problem_at_once():
    """An author corrects the file in one pass, not one error per run."""
    raw = _raw()
    _verb(raw, "take")["needs"] = ["free_hands"]
    _verb(raw, "strike")["flags"] = ["loud", "noisy"]
    raw["capabilities"]["sharp"]["says"] = ""
    with pytest.raises(CatalogError) as e:
        Catalog(raw)
    got = " / ".join(e.value.problems)
    for says in ("unknown capability 'free_hands'", "unknown flag 'noisy'", "'sharp' has no words"):
        assert says in got, got


def test_a_broken_SHAPE_stops_the_checks_early():
    """When a top-level part is missing, the verbs cannot be read against
    it. Then the loader reports the shape only, not a problem for each verb."""
    raw = _raw()
    del raw["capabilities"]
    _verb(raw, "strike")["flags"] = ["loud", "noisy"]
    with pytest.raises(CatalogError) as e:
        Catalog(raw)
    assert e.value.problems == ["no top-level key 'capabilities'"]


def _places(node, at=()):
    """Every place in the parsed file, as the path of keys to it."""
    items = node.items() if isinstance(node, dict) else enumerate(node) if isinstance(node, list) else ()
    for k, v in items:
        yield at + (k,)
        yield from _places(v, at + (k,))


def test_a_WRONG_VALUE_anywhere_gives_a_CatalogError_never_a_crash():
    """A typo must give the author the list of problems, not a TypeError from
    inside the loader. Each place of the file gets each wrong value in turn:
    no value, a number, a word, a list and a mapping (a list and a mapping
    cannot be a key or a member of a set). Of the verbs, only some are
    changed (to keep the test fast); between them they have every field, role
    test and step key that a verb can have."""
    raw = _raw()
    some = {"strike", "steal", "take_from", "flee_to", "eat", "wait"}
    crashed = []
    for at in _places(raw):
        if at[0] == "verbs" and len(at) > 1 and raw["verbs"][at[1]]["id"] not in some:
            continue
        for wrong in (None, 7, "x", ["x"], {"x": 1}):
            bad = copy.deepcopy(raw)
            node = bad
            for k in at[:-1]:
                node = node[k]
            node[at[-1]] = wrong
            try:
                Catalog(bad)
            except CatalogError:
                pass
            except Exception as e:      # any other error is the defect
                crashed.append((at, wrong, repr(e)))
    assert crashed == [], crashed[:5]


# ---------------------------------------------------------------- the inn
def _inn():
    """DUR, the flag sets, and the skeleton of every option text of the inn,
    read from the source (not imported)."""
    tree = ast.parse(INN.read_text(encoding="utf-8"))
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("DUR", "LOUD", "HARMFUL", "VIOLENT", "PHYSICAL"):
                consts[name] = ast.literal_eval(node.value)
    words = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "add"
                and len(node.args) >= 2 and isinstance(node.args[0], ast.Constant)):
            words.setdefault(node.args[0].value, set()).add(_skeleton(node.args[1]))
    return consts, words


def _skeleton(node):
    if isinstance(node, ast.IfExp):
        # A conditional text is a lab-flag variant ("eat a portion of X" under meals);
        # the else branch is the plain wording the rooms backend shares.
        return _skeleton(node.orelse)
    if isinstance(node, ast.Constant):
        return node.value
    return "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in node.values)


def _native(c, vid):
    return c.program(vid, "rooms")["verb"]


def test_the_catalog_runs_EVERY_verb_of_the_inn():
    (consts, _), c = _inn(), load()
    assert set(c.declared("rooms")["verbs"]) == set(consts["DUR"])
    assert {_native(c, k) for k in c.runnable("rooms")} == set(consts["DUR"])


def test_a_rooms_DURATION_is_the_inn_s():
    (consts, _), c = _inn(), load()
    for k in c.runnable("rooms"):
        assert c.verbs[k].duration("rooms") == consts["DUR"][_native(c, k)], k


def test_the_FLAGS_are_the_inn_s():
    """loud, harmful, violent and physical: the inn's sets, verb for verb. Two
    catalog verbs that run one inn verb (steal, take_from) carry the same flags."""
    (consts, _), c = _inn(), load()
    by_native = {}
    for k in c.runnable("rooms"):
        by_native.setdefault(_native(c, k), set()).add(c.verbs[k].flags)
    assert all(len(f) == 1 for f in by_native.values()), by_native
    for flag, inn in (("loud", "LOUD"), ("harmful", "HARMFUL"), ("violent", "VIOLENT"),
                      ("physical", "PHYSICAL")):
        mine = {n for n, f in by_native.items() if flag in next(iter(f))}
        assert mine == consts[inn], flag


def test_the_option_WORDS_are_the_inn_s():
    """Every option text the inn writes has the same words as a catalog
    template for that verb, and no template has words the inn does not write.
    So the rooms backend can take its texts from the catalog.
    Only the inn's own verbs (the DUR literal) count. The verbs the learned-mind
    lab adds behind its flags (DUR.update: sit down, sleep, ask, gossip, ...)
    are not the rooms backend's, as in test_the_catalog_runs_EVERY_verb_of_the_inn."""
    (consts, words), c = _inn(), load()
    words = {verb: w for verb, w in words.items() if verb in consts["DUR"]}
    mine = {}
    for k in c.runnable("rooms"):
        v = c.verbs[k]
        for tpl in filter(None, (v.text, v.text_with)):
            mine.setdefault(_native(c, k), set()).add(re.sub(r"\{\w+\}", "{}", tpl))
    assert mine == words


# ---------------------------------------------------------------- reading the catalog
def test_every_TEMPLATE_fills_from_its_roles():
    c = load()
    for v in c.verbs.values():
        need = {r.name: f"<{r.name}>" for r in v.roles if not r.optional}
        full = {r.name: f"<{r.name}>" for r in v.roles}
        for text in (v.option_text(need), v.option_text(full)):
            assert "{" not in text and all(n in text for n in need.values()), (v.id, text)
        if v.told:
            for names in (need, full):
                told = v.told_text("<actor>", names)
                assert "{" not in told and "<actor>" in told, (v.id, told)
        if v.told_start:
            assert "<actor>" in v.told_text("<actor>", need, start=True)
    strike = c.verbs["strike"]
    assert strike.option_text({"target": "Tom"}) == "strike Tom"
    assert strike.option_text({"target": "Tom", "instrument": "the axe"}) == "strike Tom with the axe"
    assert strike.told_text("Fenn", {"target": "Tom", "instrument": "the axe"}) == "Fenn struck Tom with the axe"
    with pytest.raises(ValueError, match="no name for role 'place'"):
        c.verbs["walk_to"].option_text({})


def test_CANONICAL_order_is_the_group_then_the_file():
    """Never the score: the order of `groups`, then the order of the file."""
    c, raw = load(), _raw()
    ranked = sorted(c.verbs, key=c.order)
    assert sorted(c.order(k) for k in c.verbs) == list(range(len(c.verbs)))
    gi = [c.groups.index(c.verbs[k].group) for k in ranked]
    assert gi == sorted(gi)
    in_file = [v["id"] for v in raw["verbs"]]
    for g in c.groups:
        mine = [k for k in ranked if c.verbs[k].group == g]
        assert mine == [k for k in in_file if k in mine], g


def test_a_PROGRAM_or_a_TEST_is_a_copy():
    """A job keeps its own steps and marks them (asked_at). That must never
    write into the catalog, which every job shares."""
    c = load()
    steps = c.program("take", "voxel")
    steps[0]["asked_at"] = 5
    assert "asked_at" not in c.program("take", "voxel")[0]
    t = c.cap_test("sharp", "voxel")
    t["edge_gt_m2"] = -1.0
    assert c.cap_test("sharp", "voxel")["edge_gt_m2"] == 0.0
    with pytest.raises(TypeError):
        c.verbs["take"] = c.verbs["drop"]


def test_an_OPTION_from_the_catalog_has_the_catalog_s_words():
    c = load()
    tom, axe = Ref("person", "Tom", "Tom"), Ref("thing", "the axe", "the axe")
    o = c.option("rooms", "strike", {"target": tom, "instrument": axe})
    assert (o.text, o.group, o.duration, o.ok) == ("strike Tom with the axe", "fight", 2, True)
    assert o.role("instrument") == axe
    walk = c.option("voxel", "walk_to", {"place": Ref("place", "c3", "the door")}, duration=31)
    assert walk.duration == 31
    with pytest.raises(ValueError, match="duration of the route"):
        c.option("voxel", "walk_to", {"place": Ref("place", "c3", "the door")})
    with pytest.raises(ValueError, match="takes a person"):
        c.option("rooms", "strike", {"target": axe})
    with pytest.raises(ValueError, match="is not filled"):
        c.option("rooms", "give", {"target": tom})
    with pytest.raises(ValueError, match="does not run on voxel"):
        c.option("voxel", "eat", {"target": Ref("thing", "b", "the bread")})
    blocked = c.option("rooms", "take", {"target": axe}, missing=("a free hand",))
    assert not blocked.ok and blocked.missing == ("a free hand",)


# ---------------------------------------------------------------- the types
TOM = Ref("person", "p1", "Tom")


def _samples():
    """One instance of each frozen type, built with dicts and lists where a
    caller would give them."""
    o = Option("take", {"target": Ref("thing", "e7", "the axe")}, "take the axe", "handle", 2,
               features={"gain": 0.4, "theft": 0.0})
    w = Option("wait", {}, "wait", "wait", 3)
    return {
        Ref: TOM,
        Percept: Percept(4, "saw", "Tom struck Fenn.", "e12", "strike", TOM, Ref("person", "p2", "Fenn")),
        Belief: Belief("took", Ref("thing", "e3", "the purse"), TOM, "Oren believes Tom took the purse.", "e9"),
        SelfState: SelfState("Fenn", "a traveller", "the yard", ["the dagger"], ["hungry"], ["hunger"]),
        Option: o,
        Menu: Menu([o, w], [], [w, o], [], [0.1, 0.0], [0.0, 0.0], [0.0, 0.0]),
        Why: Why("r1", "rules", "idle", ["e12"], [], 0.6, 0.2),
        JobEvent: JobEvent(5, "p1", "r1", "take", "start", "Tom reached for the axe"),
        DecisionRequest: DecisionRequest("r1", "p1", "idle", "It is night.", "What does Tom do next?",
                                         ["take the axe", "wait"], ["handle", "wait"],
                                         [0.1, 0.0], [0.0, 0.0], [0.0, 0.0]),
        Decision: Decision("r1", 0, "rules", [0.6, 0.4]),
    }


def _plain(x):
    """True when x holds no dict, list or set at any depth."""
    if isinstance(x, (dict, list, set)):
        return False
    if isinstance(x, tuple):
        return all(_plain(y) for y in x)
    if dataclasses.is_dataclass(x):
        return all(_plain(getattr(x, f.name)) for f in dataclasses.fields(x))
    return True


def test_FROZEN_types_hash_and_hold_no_dict():
    samples = _samples()
    frozen = {t for t in vars(api).values() if isinstance(t, type) and dataclasses.is_dataclass(t)
              and t.__dataclass_params__.frozen}
    assert frozen == set(samples)                     # a new frozen type gets a sample here
    for t, x in samples.items():
        assert _plain(x), t
        assert hash(x) == hash(copy.deepcopy(x)) and x == copy.deepcopy(x), t
        assert pickle.loads(pickle.dumps(x)) == x, t
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(x, dataclasses.fields(x)[0].name, None)
    for v in load().verbs.values():                    # the catalog's own types too
        assert _plain(v) and hash(v) == hash(copy.deepcopy(v)), v.id


def test_an_OPTION_key_does_not_depend_on_the_order_of_roles():
    a, b = Ref("thing", "e7", "the purse"), Ref("person", "p2", "Oren")
    one = Option("steal", {"thing": a, "target": b}, "steal the purse from Oren", "handle", 3)
    two = Option("steal", (("thing", a), ("target", b)), "steal the purse from Oren", "handle", 3)
    assert one == two and hash(one) == hash(two)
    assert one.key == "steal|target=p2|thing=e7"


def test_a_BLOCKED_option_says_what_is_missing_and_only_it_does():
    with pytest.raises(ValueError):
        Option("take", {}, "take", "handle", 2, ok=False)
    with pytest.raises(ValueError):
        Option("take", {}, "take", "handle", 2, missing=("a free hand",))
    with pytest.raises(ValueError):
        Option("take", {"target": "the axe"}, "take the axe", "handle", 2)   # a role is a Ref


def test_a_MENU_keeps_its_rules():
    s = _samples()
    o, w = s[Menu].options
    assert s[Menu].groups == (("handle", (0,)), ("wait", (1,)))
    lone = Option("take", {"target": Ref("thing", "e8", "the axe")}, "take the axe", "handle", 2)
    blocked = Option("drop", {}, "drop", "handle", 1, ok=False, missing=("a free hand",))
    bad = [
        dict(options=[o, lone, w], full=[o, lone, w], prior=[0, 0, 0]),   # the same text two times
        dict(prior=[0.1]),                                                  # one score for two options
        dict(options=[o, blocked], full=[o, blocked]),                      # a blocked option offered
        dict(blocked=[w]),                                                  # an offered option as blocked
        dict(full=[o]),                                                     # options and cut are not full
        dict(options=[], full=[], prior=[], prior_avg=[], habit=[]),        # nothing to choose
    ]
    for change in bad:
        args = dict(options=[o, w], blocked=[], full=[o, w], cut=[], prior=[0.1, 0.0],
                    prior_avg=[0.0, 0.0], habit=[0.0, 0.0])
        args.update(change)
        n = len(args["options"])
        for k in ("prior_avg", "habit"):
            if "prior" in change and k not in change:
                args[k] = [0.0] * n
        with pytest.raises(ValueError):
            Menu(**args)


def test_a_REQUEST_holds_texts_and_scores_and_no_ref_or_tick():
    """A chooser gets this and nothing more (design 2.4)."""
    names = {f.name for f in dataclasses.fields(DecisionRequest)}
    assert not names & {"t", "tick", "roles", "refs", "payload"}
    req = _samples()[DecisionRequest]

    def walk(x):
        if isinstance(x, tuple):
            return all(walk(y) for y in x)
        return isinstance(x, (str, int, float))
    assert all(walk(getattr(req, n)) for n in names)
    with pytest.raises(ValueError):
        dataclasses.replace(req, options=("wait", "wait"))
    with pytest.raises(ValueError):
        dataclasses.replace(req, prior=(0.1,))


def test_a_JOB_is_plain_data_that_copies_and_pickles():
    s = _samples()
    job = Job("r1", "p1", s[Option], 5, s[Why], state={"step": 0, "asked_at": None})
    for twin in (copy.deepcopy(job), pickle.loads(pickle.dumps(job))):
        assert twin == job and twin is not job
    twin = copy.deepcopy(job)
    twin.state["step"] = 1
    assert job.state["step"] == 0
    with pytest.raises(ValueError):
        Job("r1", "p1", s[Option], 5, s[Why], status="paused")


def test_a_PERCEPT_and_a_JOB_EVENT_refuse_unknown_words():
    with pytest.raises(ValueError):
        Percept(1, "smelled", "smoke", "e1")
    with pytest.raises(ValueError):
        Percept(1, "saw", "smoke", "e1", phase="over")
    with pytest.raises(ValueError):
        JobEvent(1, "p1", "r1", "take", "paused")


def test_the_PROTOCOLS_name_what_a_chooser_and_a_backend_do():
    class Rules:
        name = "rules"

        def submit(self, req):
            self.last = Decision(req.id, 0, self.name)

        def poll(self, req_id, block=True):
            return self.last

    class Mute:
        name = "mute"

        def submit(self, req):
            pass

    assert isinstance(Rules(), Chooser) and not isinstance(Mute(), Chooser)
    assert not isinstance(Rules(), Backend)


# ---------------------------------------------------------------- imports
def _imports(path):
    """(module, at the top level of the file) for every absolute import."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top = set(map(id, tree.body))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(a.name, id(node) in top) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            out.append((node.module, id(node) in top))
    return out


def test_src_mind_IMPORTS_the_standard_library_only():
    """No backend, no lab, no numpy. PyYAML only inside load(), so that
    importing the mind needs nothing outside the standard library."""
    files = sorted((ROOT / "src" / "mind").glob("*.py"))
    assert {f.name for f in files} >= {"__init__.py", "api.py", "catalog.py"}
    for f in files:
        for mod, at_top in _imports(f):
            root = mod.split(".")[0]
            if root == "yaml":
                assert f.name == "catalog.py" and not at_top, f
            else:
                assert root in sys.stdlib_module_names, (f.name, mod)


def test_nothing_in_src_IMPORTS_lab():
    """lab and src never import each other (design 2.1). The same test as
    `grep -rn "import lab\\|from lab" src/`, without its false hits."""
    pat = re.compile(r"^\s*(?:import|from)\s+lab(?:\.|\s|$)", re.M)
    hits = [str(f.relative_to(ROOT)) for f in (ROOT / "src").rglob("*.py")
            if pat.search(f.read_text(encoding="utf-8", errors="replace"))]
    assert hits == []
