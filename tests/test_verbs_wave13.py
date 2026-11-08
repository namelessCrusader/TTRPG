"""events.VERBS, appended once for wave 13 (part 13a, lane B2).

The verb table is closed (a chooser or a belief key that matched on an open
string would silently stop matching), and `emit` does not check a verb against
it. A verb's place in the tuple is its VERB_ID, so the 47 verbs that existed
stay where they are, and the wave's fifteen follow in the order the plan fixes
(plan section 4): T2, T3, T5, T7, T11, in one change. T3 names none.

Tags: [I] identity and closed forms."""
import ast
import os

import pytest

from src.voxel import events
from src.voxel.materials import STONE
from src.voxel.sim import World

HERE = os.path.dirname(os.path.abspath(__file__))
VOX = os.path.join(os.path.dirname(HERE), "src", "voxel")

# base19's table, verbatim
BASE19 = (
    "decides", "steps", "lands", "falls", "swings", "hits", "wounds",
    "breaks", "topples", "looses", "melts", "sets", "dissolves",
    "ignites", "catches_fire", "burns_out",
    "says", "heard", "saw", "looked",
    "gone", "faints", "wakes", "dies",
    "holds", "takes", "reaches", "pulls_in", "catches", "releases",
    "jumps", "goes", "stays", "looks", "leans", "stands",
    "retries", "gives_up", "arrives", "rescues", "rolls",
    "splits", "ashes",
    "touches", "strikes", "cracks", "parts",
)
# the wave 13 tail, in the plan's order
T2 = ("spills", "joins", "fuses", "comes_apart")
T3 = ()
T5 = ("crosses",)
T7 = ("shivers", "sweats")
T11 = ("opens", "closes", "locks", "unlocks", "picks", "kicks", "jams", "forces")


def test_the_old_forty_seven_verbs_have_not_moved():
    assert len(BASE19) == 47
    assert events.VERBS[:47] == BASE19
    assert [events.VERB_ID[v] for v in BASE19] == list(range(47)), "a verb's id is its place"


def test_the_fifteen_new_verbs_follow_in_the_plans_order_T2_T3_T5_T7_T11():
    assert events.VERBS[47:] == T2 + T3 + T5 + T7 + T11
    assert len(events.VERBS) == 62
    assert events.VERBS[47:51] == T2 and events.VERBS[51] == "crosses"
    assert events.VERBS[52:54] == T7 and events.VERBS[54:] == T11


def test_the_table_has_no_repeat_and_the_id_map_covers_it():
    assert len(set(events.VERBS)) == len(events.VERBS)
    assert events.VERB_ID == {v: i for i, v in enumerate(events.VERBS)}
    assert all(isinstance(v, str) and v.islower() for v in events.VERBS)


def test_what_makes_a_noise_at_a_door_is_a_sounding_verb_and_a_pick_is_not():
    assert events._SOUNDING == frozenset({"hits", "lands", "breaks", "kicks", "jams", "forces", "opens"})
    assert "picks" not in events._SOUNDING, "its click is passed to `_noise` as its own event"
    assert events._SOUNDING <= set(events.VERBS)


@pytest.mark.parametrize("verb,sounding", [("kicks", True), ("forces", True), ("opens", True),
                                           ("jams", True), ("picks", False), ("closes", False)])
def test_a_noise_made_with_no_event_is_matched_to_a_sounding_event_at_its_place(verb, sounding):
    """`_noise` called without `ev` gets a box; `_unbox` fills it from the first sounding
    event of the same tick within 2 cells. A kick at a door is that event now."""
    w = World(12, 12, 8)
    snd = w._noise((5, 5, 2), 50.0, [STONE])
    box = snd["src"]
    assert box.id is None
    ev = w.emit(verb, place=(5, 5, 2), text="")
    w._unbox()
    assert box.id == (ev.id if sounding else None)


def _verbs_named_in(src):
    """Every string literal given as the verb of an `emit`, `_defer` or `_ev` call (the first
    argument, or either branch of a conditional there) or as a `verb=` keyword or default."""
    out = set()

    def consts(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            yield n.value
        elif isinstance(n, ast.IfExp):
            yield from consts(n.body)
            yield from consts(n.orelse)

    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name in ("emit", "_defer", "_ev") and n.args:
                out.update(consts(n.args[0]))
            for kw in n.keywords:
                if kw.arg == "verb":
                    out.update(consts(kw.value))
        elif isinstance(n, ast.FunctionDef):
            args = n.args
            pos = args.args[len(args.args) - len(args.defaults):]
            for a, d in list(zip(pos, args.defaults)) + [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d]:
                if a.arg == "verb":
                    out.update(consts(d))
    return out


def test_every_verb_a_module_of_src_voxel_names_is_in_the_closed_table():
    """`emit` does not validate. A typo in a new lane's verb would be an event no chooser
    matches, and nothing would say so."""
    missing = {}
    for f in sorted(os.listdir(VOX)):
        if f.endswith(".py"):
            for v in _verbs_named_in(open(os.path.join(VOX, f)).read()):
                if v not in events.VERBS:
                    missing.setdefault(v, []).append(f)
    assert not missing, missing


def test_the_scan_sees_a_conditional_a_keyword_and_a_default():
    src = ('def kick(self, verb="kicks"):\n'
           '    self.emit("opens" if x else "closes")\n'
           '    self._defer("jams")\n'
           '    self.blow(verb="strikes")\n')
    assert _verbs_named_in(src) == {"kicks", "opens", "closes", "jams", "strikes"}
    assert _verbs_named_in('def f():\n    self.emit("no_such_verb")\n') == {"no_such_verb"}


def test_a_new_verb_can_be_emitted_and_read_back_as_json():
    w = World(8, 8, 8)
    for v in T2 + T5 + T7 + T11:
        ev = w.emit(v, place=(1, 1, 1), text="")
        assert ev.verb == v and ev.to_json()["verb"] == v
    assert len(w.events.rows) == 15
