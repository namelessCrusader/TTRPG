"""New in M7 (docs/sim-mind-design.md, 3.5 and 4.4): the attention law and
the policy hooks.

`World.attention` is a law for every body, like the others. "latch" is the
law there has always been, and the default: a percept commits a body and it
stops sensing until the emergency is over. "onset" senses every tick and
decides only on NEWS -- a percept whose source this mind has not attended to
for `attend_gap` ticks -- or when it has nowhere to be, or when a linked
mind asked to be woken. A percept that goes on is news once.

The hooks are what a linked mind (M8) needs from the will layer: `prepare`
and `limbs` before the limb loop, `on_restore` after a restore, and
`p["intent"]` (aim, flee, want, take, wake_at) read by the acts and menus.
"""
import pickle
import re
from pathlib import Path

import numpy as np

from src.voxel.events import _Pending
from src.voxel.sim import IRON, STONE, WILL, WOOD, TablePolicy, World
from src.voxel.scenes import _person

SRC = Path(__file__).resolve().parent.parent / "src" / "voxel"


def _smoky_room(known):
    """The review's smoky room: a crib of wood alight at one end, a witness
    at the other, a door in the east wall. `known` is whether the witness
    knows the room (and so the door) before it starts."""
    w = World(34, 26, 32, voxel_cm=5)
    w.open_sky = False
    w.fill(0, 34, 0, 26, 0, 1, STONE)
    w.fill(0, 34, 0, 26, 31, 32, STONE)
    for (x0, x1, y0, y1) in ((0, 1, 0, 26), (33, 34, 0, 26),
                             (0, 34, 0, 1), (0, 34, 25, 26)):
        w.fill(x0, x1, y0, y1, 0, 32, STONE)
    w.fill(5, 13, 10, 16, 1, 3, WOOD, frac=0.6)
    w.exits = [(32, 13)]
    p = _person(w, 22, 13, z0=1)
    p["name"] = "the witness"
    if not known:
        p["known"][:] = False
    return w, p


def _run_smoky(w, p, ticks):
    for t in range(ticks):
        if t < 60:
            w.E[8, 13, 2] += 2500.0           # keep the crib lit
        p["smoke"] = max(p["smoke"], 0.2)     # and the witness coughing
        w.step()
        p["events"].clear()
    return [r for r in w.traces if r["who"] == p["name"]]


def _standing(x=20, y=20):
    w = World(40, 40, 30, voxel_cm=5)
    w.fill(0, 40, 0, 40, 0, 1, STONE)
    w.exits = []
    p = _person(w, x, y)
    comp, sl = w._person_cells(p)
    cells = np.argwhere(comp)
    cells[:, 0] += sl[0].start
    cells[:, 1] += sl[1].start
    ax_, ay_ = float(cells[:, 0].mean()), float(cells[:, 1].mean())
    eye = (ax_, ay_, float(cells[:, 2].max()))
    return w, p, cells, ax_, ay_, eye


class _Linked(TablePolicy):
    """A stub of a linked mind: the table's picks, plus every hook, each
    one writing down when it was called and with what."""
    name = "linked-stub"

    def __init__(self, ask=None, want=None):
        self.ask, self.want, self.calls = ask, want or {}, []

    def prepare(self, w, p, percept):
        self.calls.append(("prepare", w.tick, p["name"], percept))

    def limbs(self, w, p, percept):
        self.calls.append(("limbs", w.tick, p["name"], percept))
        return self.ask

    def pick(self, sit, menu):
        self.calls.append(("pick", sit["tick"], sit["who"], sit["limb"]))
        tag = self.want.get(sit["limb"])
        if tag is not None:
            hit = [i for i, o in enumerate(menu) if o["tag"] == tag]
            return hit[0] if hit else 0
        return super().pick(sit, menu)

    def on_restore(self, w):
        self.calls.append(("on_restore", w.tick))


# ── the law ─────────────────────────────────────────────────────────────────

def test_the_DEFAULT_is_latch_and_it_never_touches_attention_state():
    w, p = _smoky_room(known=True)
    assert w.attention == "latch"
    rows = _run_smoky(w, p, 80)                # a surprised reaction: 1-1.5 s
    assert rows, "the witness chokes, so it decides"
    m = w.minds[p["pid"]]
    assert m.attn == {} and m.news is None, \
        "latch must leave the attention law's state alone (golden is exact)"
    assert all("onset" not in r["situation"] for r in rows), \
        "sit['onset'] exists only under onset"
    assert "intent" not in p and "job" not in p


def test_ONSET_in_the_smoky_room_decides_at_most_twice_and_gets_out():
    """Measured in the design (3.5): latch with no exit known decides once
    and never gets out -- the percept latches the body where it stands.
    Onset decides again when it has nowhere to be, and by then it has
    looked about and found the door."""
    w, p = _smoky_room(known=False)
    w.attention = "onset"
    rows = _run_smoky(w, p, 120)
    assert len(rows) <= 2, [(r["tick"], r["percept"]) for r in rows]
    assert p["safe"], "the witness must get out under onset"
    assert sum(1 for r in rows if r["situation"].get("onset")) == 1, \
        "the choking goes on all along: it is news once"


def test_a_PERCEPT_THAT_GOES_ON_is_news_once_and_again_after_the_gap():
    w, p, *_ = _standing()
    w.attention = "onset"
    p["goal"] = (30, 20)                      # busy: only news decides
    p["_decided"] = w.tick
    go, news = w._attend(p, "sees_fire", 5)
    assert go and news == 5
    w.tick += 1
    p["_decided"] = w.tick
    assert w._attend(p, "sees_fire", 5) == (False, None)
    w.tick += WILL["attend_gap"] + 1          # out of sight for the whole gap
    p["_decided"] = w.tick
    assert w._attend(p, "sees_fire", 5)[1] == 5
    w.tick += 1
    p["_decided"] = w.tick
    assert w._attend(p, "sees_fire", 6)[1] == 6, "another fire is news"


def test_a_SOURCE_with_no_id_yet_is_not_news_again_when_its_id_comes():
    """An event raised in the tick it is perceived has only a box for an id
    (`_Pending`) until the tick ends. Seen again the tick after, under its
    real id, it is the same event."""
    w, p, *_ = _standing()
    w.attention = "onset"
    p["goal"] = (30, 20)
    box = _Pending()
    p["_decided"] = w.tick
    assert w._attend(p, "hears_noise", box)[0]
    box.id = 41                               # what `_unbox` does at tick end
    w.tick += 1
    p["_decided"] = w.tick
    assert w._attend(p, "hears_noise", 41) == (False, None)
    m = w.minds[p["pid"]]
    assert not any(isinstance(k, _Pending) for k in m.attn) \
        and not isinstance(m.news, _Pending)


def test_a_CAUSE_FOUND_LATER_is_not_news():
    w, p, *_ = _standing()
    w.attention = "onset"
    p["goal"] = (30, 20)
    p["_decided"] = w.tick
    assert w._attend(p, "chokes", None)[1] == "chokes"
    w.tick += 1
    p["_decided"] = w.tick
    assert w._attend(p, "chokes", 9) == (False, None), \
        "the same smoke, now blamed on a fire, is not news"


def test_only_the_FIRST_cause_found_later_is_not_news():
    """A percept whose source flickers -- none, a fire, none, a second fire
    -- was blamed on the second fire after a sourceless tick, and that was
    taken for the first cause found again: the second fire was never news."""
    w, p, *_ = _standing()
    w.attention = "onset"
    p["goal"] = (30, 20)
    got = []
    for src in (None, 9, None, 10, 10):
        p["_decided"] = w.tick
        got.append(w._attend(p, "chokes", src))
        w.tick += 1
    assert got == [(True, "chokes"), (False, None), (False, None),
                   (True, 10), (False, None)], got


def test_a_LONG_RUN_then_a_short_break_is_ONE_decision_about_one_onset():
    """The percept's own name was kept only on its sourceless ticks, so a
    long run blamed on one fire outlived it: after a short break the smoke
    came back with no source as news, and the first source found for it
    the tick after was news as well. One onset, one decision -- after a
    short break and after one longer than `attend_gap` alike."""
    for gap in (2, WILL["attend_gap"] + 5):
        w, p, *_ = _standing()
        w.attention = "onset"
        p["goal"] = (30, 20)
        got = []
        for src in [None] + [9] * 45 + ["-"] * gap + [None, 10]:
            p["_decided"] = w.tick
            if src != "-":
                got.append(w._attend(p, "chokes", src))
            w.tick += 1
        assert sum(d for d, _ in got[-2:]) == 1, (gap, got[-2:])


def test_WAKE_AT_decides_a_busy_body_once():
    w, p, *_ = _standing()
    w.attention = "onset"
    p["goal"] = (30, 20)
    p["_decided"] = w.tick
    p["intent"] = {"wake_at": w.tick + 3}
    assert not w._attend(p, None, None)[0]
    w.tick += 3
    assert w._attend(p, None, None)[0]
    assert "wake_at" not in p["intent"], "a rung alarm is spent"
    w.tick += 1
    assert not w._attend(p, None, None)[0]


def test_the_TABLE_says_its_mouth_row_only_on_news():
    t = TablePolicy()
    menu = [{"key": "say nothing", "tag": "quiet", "verb": "quiet"},
            {"key": "say(fire)", "tag": "say:fire", "verb": "say"}]
    sit = {"percept": "sees_fire", "limb": "mouth",
           "reflexes": {"sees_fire": {"legs": "go:exit", "mouth": "say:fire"}}}
    assert t.pick(sit, menu) == 1, "latch: no onset key, the row as always"
    assert t.pick(dict(sit, onset=True), menu) == 1
    assert t.pick(dict(sit, onset=False), menu) == 0
    legs = [{"key": "stay", "tag": "stay", "verb": "stay"},
            {"key": "go(door)", "tag": "go:exit", "verb": "go"}]
    assert t.pick(dict(sit, limb="legs", onset=False), legs) == 1, \
        "the legs still run on a percept that is not news"


# ── the hooks ───────────────────────────────────────────────────────────────

def test_PREPARE_and_LIMBS_run_once_before_any_pick():
    w, p, cells, ax_, ay_, eye = _standing()
    w.policy = pol = _Linked(ask=None)
    w._decide(p, None, cells, ax_, ay_, eye, 0.1 * w.scale)
    kinds = [c[0] for c in pol.calls]
    assert kinds[:2] == ["prepare", "limbs"] and kinds.count("prepare") == 1 \
        and kinds.count("limbs") == 1 and set(kinds[2:]) == {"pick"}
    assert pol.calls[0][2:] == (p["name"], None)


def test_LIMBS_asks_only_the_parts_named_and_a_walk_goes_on():
    w, p, cells, ax_, ay_, eye = _standing()
    p["goal"] = (30, 20)
    w.policy = _Linked(ask={"mouth"})
    w._decide(p, None, cells, ax_, ay_, eye, 0.1 * w.scale)
    assert set(w.traces[-1]["menus"]) == {"mouth"}
    assert p["goal"] == (30, 20), "legs not asked: their null act never ran"


def test_a_DECIDE_THAT_ASKS_NO_LIMB_still_counts_as_deciding():
    w, p, cells, ax_, ay_, eye = _standing()
    w.tick = 7
    w.policy = _Linked(ask=set())
    n = len(w.traces)
    w._decide(p, None, cells, ax_, ay_, eye, 0.1 * w.scale)
    assert len(w.traces) == n and p["_decided"] == 7


def test_ON_RESTORE_is_called_after_a_restore():
    w, p, *_ = _standing()
    w.policy = pol = _Linked()
    snap = w.snapshot()
    w.step()
    w.restore(snap)
    assert pol.calls[-1] == ("on_restore", 0)
    assert w.policy is pol, "the mind is kept across a restore"


def test_INTENT_AIM_points_the_swing():
    w, p, cells, ax_, ay_, eye = _standing()
    got = {}
    w._swing = lambda p_, arm, toward=None, back=0.0: got.update(toward=toward)
    w.policy = _Linked(want={"hands": "swing"})
    p["facing"] = (1.0, 0.0)
    p["intent"] = {"aim": (0.0, 1.0)}
    w._decide(p, None, cells, ax_, ay_, eye, 0.1 * w.scale)
    assert got["toward"] == (0.0, 1.0)
    p["intent"] = {}
    w._decide(p, None, cells, ax_, ay_, eye, 0.1 * w.scale)
    assert got["toward"] == (1.0, 0.0), "no aim: straight ahead, as always"


def test_INTENT_FLEE_makes_a_walk_a_flight():
    w, p, cells, ax_, ay_, eye = _standing()
    w.policy = _Linked(want={"legs": "go:step"})
    w._decide(p, None, cells, ax_, ay_, eye, 0.1 * w.scale)
    assert p["goal"] is not None and p["fleeing"] is False
    p["intent"] = {"flee": True}
    w._decide(p, None, cells, ax_, ay_, eye, 0.1 * w.scale)
    assert p["goal"] is not None and p["fleeing"] is True


def test_INTENT_WANT_keeps_its_row_under_the_cap():
    w, p, cells, *_ = _standing()
    full = w._menu(p, cells, None, "legs", {}, cap=10 ** 6)
    cut = w._menu(p, cells, None, "legs", {}, cap=3)
    keys = {o["key"] for o in cut}
    lost = next(o for o in full if o["key"] not in keys and o.get("goal"))
    p["intent"] = {"want": {"legs": {"goal": lost["goal"]}}}
    kept = w._menu(p, cells, None, "legs", {}, cap=3)
    assert lost["key"] in {o["key"] for o in kept} and len(kept) == 3
    assert kept[0]["tag"] == "stay", "the null act never falls off"


def test_INTENT_TAKE_offers_the_thing_meant_and_only_a_real_one():
    w, p, cells, *_ = _standing()
    for k, (x, y) in enumerate(((22, 20), (18, 20), (20, 22), (20, 18))):
        w.fill(x, x + 1, y, y + 1, 1, 2, IRON if k % 2 == 0 else WOOD)
    w.fill(35, 36, 35, 36, 1, 2, IRON)        # out of reach
    rows = lambda: [o for o in w._menu(p, cells, None, "hands", {}, cap=10 ** 6)
                    if o["tag"] == "take"]
    for want in ((22, 20, 1), (20, 22, 1)):
        p["intent"] = {"take": want}
        got = [tuple(o["what"]["cell"]) for o in rows()]
        assert want in got and len(got) == len(set(got)), got
    p["intent"] = {"take": (35, 35, 1)}
    assert (35, 35, 1) not in [tuple(o["what"]["cell"]) for o in rows()]
    p["intent"] = {"take": (20, 20, 3)}       # the body's own flesh
    assert all(o["what"]["cell"] != (20, 20, 3) for o in rows())


def test_TAKE_ROWS_each_say_something_else_and_the_cap_keeps_the_nearest():
    """Four unnamed lumps of iron in reach were four rows that each read
    "take hold of the iron", with no distance on them, so the cap kept them
    in the order the reach box met them. Each now says where it is (and
    which of those there), and carries how far it is."""
    w, p, cells, *_ = _standing()
    for x, y in ((22, 20), (17, 20), (20, 23), (20, 16)):
        w.fill(x, x + 1, y, y + 1, 1, 2, IRON)
    reach = {o["cell"]: o["away"] for o in w._objects_within_reach(p, cells)}
    full = w._menu(p, cells, None, "hands", {}, cap=10 ** 6)
    takes = [o for o in full if o["tag"] == "take"]
    keys = [o["key"] for o in takes]
    assert len(takes) == 4 and len(set(keys)) == 4, keys
    assert all("iron" in k for k in keys)
    assert all(o["away"] == reach[o["what"]["cell"]] for o in takes)
    cut = w._menu(p, cells, None, "hands", {}, cap=len(full) - 1)
    lost = [o for o in takes if o["key"] not in {c["key"] for c in cut}]
    assert len(lost) == 1 and lost[0]["away"] == max(reach.values()), \
        "the furthest falls off, not the nearest"


def test_ADOPTED_lookalikes_and_UNNAMED_lumps_are_ALL_offered():
    """Two irons a hand had taken and put down (adopted: "the first iron at
    your feet", "the second") beside two unnamed lumps of iron. The lumps
    were set apart into the same words, and a last guard kept only the
    nearer row of each text: the two irons left the menu, and so did the
    one a linked mind meant to take. Every thing is a row, every row says
    something else, and a named thing's row says its name."""
    w, p, cells, *_ = _standing()
    for x, y in ((22, 20), (19, 20), (20, 23), (17, 20)):
        w.fill(x, x + 1, y, y + 1, 1, 2, IRON)
    a, b = w._adopt(19, 20, 1, "iron"), w._adopt(17, 20, 1, "iron")
    assert a and b
    reach = w._objects_within_reach(p, cells)
    for want in (None, b):
        p["intent"] = {} if want is None else {"take": want}
        takes = [o for o in w._menu(p, cells, None, "hands", {}, cap=10 ** 6)
                 if o["tag"] == "take"]
        keys = [o["key"] for o in takes]
        got = {w.entity_at(*o["what"]["cell"]) for o in takes}
        assert len(takes) == len(reach) == 4 and len(set(keys)) == 4, keys
        assert {a, b} <= got, keys
        for o in takes:
            e = w.entity_at(*o["what"]["cell"])
            if e:
                assert o["key"] == "take hold of " + w.name(e, None, p), keys


def test_a_NAMED_thing_keeps_its_NAME_beside_an_unnamed_lookalike():
    """A registered "iron" and a lump of unnamed iron both read "take hold
    of the iron". The thing keeps the words every line about it says; the
    lump is the one set apart."""
    w, p, cells, *_ = _standing()
    e = w.register("iron")
    w._stamping = e
    w.fill(22, 23, 20, 21, 1, 2, IRON)
    w._stamping = 0
    w.fill(17, 18, 20, 21, 1, 2, IRON)
    takes = [o for o in w._menu(p, cells, None, "hands", {}, cap=10 ** 6)
             if o["tag"] == "take"]
    keys = {w.entity_at(*o["what"]["cell"]): o["key"] for o in takes}
    assert keys[e] == "take hold of the iron" == "take hold of " \
        + w.name(e, None, p), keys
    assert keys[0] != keys[e] and "iron" in keys[0], keys


def test_INTENT_TAKE_of_a_named_thing_IN_FLIGHT_is_not_offered():
    """By eid, as by a cell: a thing whose cells are drawn particle matter
    (`moving`) is the solver's, and no take row is offered for it, even
    while the kept index of where things are has not caught up."""
    w, p, cells, *_ = _standing()
    e = w.register("bar")
    w._stamping = e
    w.fill(22, 23, 20, 21, 1, 2, IRON)
    w._stamping = 0
    assert w._wanted_thing(p, cells, e) is not None
    assert len(w.cells_of(e)) == 1            # the index is kept from here
    w.moving[22, 20, 1] = True
    assert w._wanted_thing(p, cells, e) is None
    p["intent"] = {"take": e}
    assert not any(o["tag"] == "take" for o in
                   w._menu(p, cells, None, "hands", {}, cap=10 ** 6))


# ── hygiene and snapshots ───────────────────────────────────────────────────

def test_INTENT_and_JOB_were_not_already_person_keys():
    """`p["intent"]` and `p["job"]` belong to a linked mind (3.5). Nothing
    else in the voxel sim may read or write them under those names: the
    will layer reads `intent` and nothing writes `job` but the link."""
    pat = re.compile(r"""(?:\[|\.get\(|setdefault\(|pop\()\s*["'](intent|job)["']""")
    users = {}
    for f in sorted(SRC.glob("*.py")):
        for m in pat.finditer(f.read_text()):
            users.setdefault(m.group(1), set()).add(f.name)
    # quiet.py (wave 13, T5) is a second, read-only reader: horizon() reads
    # p["intent"]["wake_at"] so that a night skip stops before a sleeper wakes.
    assert users.get("intent", set()) <= {"will.py", "mindlink.py", "quiet.py"}, users
    assert users.get("job", set()) <= {"mindlink.py"}, users


def test_a_SNAPSHOT_of_a_linked_world_copies_the_intent_and_the_job_pickles():
    w, p = _smoky_room(known=True)
    w.attention = "onset"
    w.policy = pol = _Linked()
    p["intent"] = {"aim": (1.0, 0.0), "want": {"legs": {"goal": (32, 13)}},
                   "wake_at": 5}
    p["job"] = {"verb": "flee_to", "roles": {"goal": (32, 13)},
                "steps": [{"limb": "legs", "match": ("goal", (32, 13))}],
                "i": 0, "status": "running", "started": 0}
    _run_smoky(w, p, 3)
    snap = w.snapshot()
    q = snap["persons"][p["pid"]]
    assert q["job"] == p["job"] and q["job"] is not p["job"]
    assert q["intent"] == p["intent"] and q["intent"] is not p["intent"]
    assert "policy" not in snap
    assert pickle.loads(pickle.dumps(q["job"])) == p["job"]
    attn = dict(w.minds[p["pid"]].attn)
    _run_smoky(w, p, 3)
    w.restore(snap)
    assert w.persons[p["pid"]]["job"] == q["job"]
    assert w.minds[p["pid"]].attn == attn
    assert pol.calls[-1][0] == "on_restore"
