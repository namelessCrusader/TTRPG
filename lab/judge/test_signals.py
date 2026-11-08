"""Checks for the scene measures on hand-made event logs.

    lab/.venv/bin/python -m lab.judge.test_signals
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from lab.judge.signals import PATTERNS, analyse, chains, match, report, surprise, surprising, variety

P = {p["name"]: p for p in PATTERNS}


def ev(i, t, actor, action, target=None, kind="peasant", **kw):
    e = {"id": f"e{i}", "t": t, "actor": actor, "kind": kind, "action": action, "target": target,
         "target_kind": kw.pop("target_kind", None), "traits": kw.pop("traits", {"bravery": 0.5}),
         "options": ["a", "b"], "chosen": 0, "p_char": [0.5, 0.5], "p_avg": [0.5, 0.5],
         "seen": [], "valid": True}
    e.update(kw)
    return e


def test_revenge_needs_the_same_two():
    log = [ev(0, 1, "bram", "strike", "mara"),
           ev(1, 5, "mara", "strike", "tom"),          # not revenge: a different target
           ev(2, 9, "mara", "shove", "bram")]          # revenge
    ms = match(log, P["revenge"])
    assert [m["bind"] for m in ms] == [{"?x": "bram", "?a": "mara"}], ms


def test_betrayal_order_matters():
    assert match([ev(0, 1, "a", "strike", "b"), ev(1, 2, "a", "give", "b")], P["betrayal"]) == []
    assert len(match([ev(0, 1, "a", "give", "b"), ev(1, 2, "a", "strike", "b")], P["betrayal"])) == 1


def test_rescue_needs_a_third_and_is_soon():
    harm = ev(0, 1, "bandit", "threaten", "child", kind="bandit")
    assert match([harm, ev(1, 3, "child", "strike", "bandit")], P["rescue"]) == []      # not a third
    assert match([harm, ev(1, 40, "mara", "strike", "bandit")], P["rescue"]) == []      # too late
    ms = match([harm, ev(1, 4, "mara", "strike", "bandit")], P["rescue"])
    assert ms and ms[0]["bind"]["?h"] == "mara"
    ms = match([harm, ev(1, 6, "tom", "help up", "child")], P["rescue"])
    assert ms and ms[0]["bind"]["?h"] == "tom"


def test_lie_found_out_links_the_lie():
    lie = ev(0, 1, "fenn", "tell", "mara", facts={"said": "the cellar is empty", "true": False, "lie": True})
    other = ev(1, 2, "mara", "notice", facts={"learns": "x", "contradicts": "e9"})
    found = ev(2, 8, "mara", "notice", facts={"learns": "the cellar is empty", "contradicts": "e0"})
    assert match([lie, other], P["lie found out"]) == []
    assert len(match([lie, other, found], P["lie found out"])) == 1


def test_coward_acts_bravely():
    brave = ev(0, 1, "tom", "strike", "wolf", target_kind="wolf", traits={"bravery": 0.1})
    usual = ev(1, 1, "sir ada", "strike", "wolf", kind="knight", target_kind="wolf", traits={"bravery": 0.9})
    assert [m["bind"]["?a"] for m in match([brave, usual], P["coward acts bravely"])] == ["tom"]


def test_surprise_splits_in_and_out_of_character():
    a = ev(0, 1, "tom", "strike", "wolf", p_avg=[0.05, 0.95], p_char=[0.4, 0.6])   # the character explains it
    b = ev(1, 1, "tom", "strike", "wolf", p_avg=[0.05, 0.95], p_char=[0.02, 0.98])  # it does not
    assert surprise(a)["unexpected"] > 2.9 and surprise(a)["fit"] > 0
    assert surprise(b)["fit"] < 0
    inc, off = surprising([a, b, ev(2, 1, "tom", "wait", p_avg=[0.001, 0.999], p_char=[0.004, 0.996])])
    assert [e["id"] for _, _, e in inc] == ["e0"]            # plausible for tom, and twice the average
    assert {e["id"] for _, _, e in off} == {"e1", "e2"}      # e2: 4x likelier, but a 1-in-250 draw
    assert surprise({"id": "x"}) is None


def test_chains_follow_what_was_seen():
    log = [ev(0, 1, "a", "shove", "b"),
           ev(1, 2, "b", "strike", "a", seen=["e0"]),
           ev(2, 3, "c", "defend", "a", seen=["e1"]),
           ev(3, 4, "c", "run to", seen=["e2"]),         # its own event: not a reaction
           ev(4, 5, "d", "sing", seen=["e1"])]           # saw it, but is not aimed at anyone in it
    c = chains(log)
    assert c["longest"] == 3 and c["actors"] == ["a", "b", "c"], c
    assert c["reactive"] == 0.4


def test_variety_and_report():
    calm = [ev(i, i, "a", "give", "b") for i in range(6)]
    wild = [ev(i, i, "a", "strike", "b") for i in range(6)]
    v = variety({"calm": calm, "wild": wild})
    assert v["coverage"] == 2 / 25 and abs(v["run_distance"] - 1.0) < 1e-9
    assert variety({"x": calm, "y": list(calm)})["run_distance"] == 0.0
    with tempfile.TemporaryDirectory() as d:
        for name, log in (("s1", calm), ("s2", wild + [ev(9, 9, "b", "strike", "a")])):
            Path(d, name).mkdir()
            Path(d, name, "events.jsonl").write_text("\n".join(json.dumps(e) for e in log))
        text = report(d)
        assert "revenge | 50%" in text, text
        assert analyse({"x": wild})["pattern_freq"] == {}


if __name__ == "__main__":
    n = 0
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            n += 1
    print(f"{n} passed")
