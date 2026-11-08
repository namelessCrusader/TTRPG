"""Seam v2 (step E5): named things reach the menu, numbers never do.

v2 changed two things and nothing else. A thing's number (`eid`, `eids`) is
hidden like the tick and the name of the body, and the header says v2. The
rest of the text is the v1 bytes, so a dataset can tell the two spellings
apart by the header alone. And a harvest never mixes the two in one file.
"""
import hashlib
import json

import pytest

from src.voxel import harvest, scenes, seam
from src.voxel.policy import LIMBS


# sha256 of `seam.rules(LIMBS)` under mark1-menu-v1, without its header line.
# Captured from the v1 tree before the bump. If this moves, the words of the
# cached half moved, and that is a new SPELLING, not an edit.
V1_RULES_BODY = "9fb5a80b284275b57a80048558efb95df660dca7af82c89672b51cb522151727"


def _sit(**more):
    return dict({"tick": 7, "who": "smith", "percept": "nothing",
                 "reflexes": {}, "smoke": 0.0, "blood_o2": 1.0, "burn": 0.0,
                 "hurt": 0.0, "holding": "the axe", "knows_a_way_out": True,
                 "seen_of_the_world": 0.5, "ground_it_trusts": 0.5,
                 "others_in_earshot": 0}, **more)


def test_v2_is_v1_with_a_new_HEADER():
    """Only the first line of the cached half moved."""
    assert seam.SPELLING == "mark1-menu-v2"
    head, body = seam.rules(list(LIMBS)).split("\n", 1)
    assert head == "[mark1-menu-v2]"
    assert hashlib.sha256(body.encode()).hexdigest() == V1_RULES_BODY, \
        "the rules text moved below the header: that needs a new SPELLING"


def test_a_THINGS_NUMBER_is_never_spelled():
    """`eid` and `eids` are hidden like `tick` and `who`: a situation that
    carries them spells exactly as one that does not."""
    assert {"eid", "eids"} <= set(seam.HIDDEN)
    menu = [{"key": "keep hold of the axe"}, {"key": "swing the axe"}]
    plain = seam.situation(_sit(), "hands", {}, menu)
    numbered = seam.situation(_sit(eid=41, eids=[41, 42]), "hands", {}, menu)
    assert plain == numbered
    assert "eid" not in numbered and "41" not in numbered


def test_a_HARVEST_FILE_holds_ONE_spelling(tmp_path):
    """Appending to a file of another spelling is refused before any world
    runs, and a file that holds two spellings is refused on reading."""
    old = tmp_path / "v1.jsonl"
    row = {"spelling": "mark1-menu-v1", "text": "t\n", "limb": "legs",
           "options": ["stay", "jump"], "pick": 0, "outcome": None,
           "key": "t\n", "seed": 0}
    old.write_text(json.dumps(row) + "\n")
    with pytest.raises(harvest.MixedSpelling):
        harvest.to_file(str(old), [0], workers=1, append=True)
    assert harvest.read(str(old)) == [row], "and the file is left as it was"

    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(json.dumps(row) + "\n"
                     + json.dumps(dict(row, spelling=seam.SPELLING)) + "\n")
    with pytest.raises(harvest.MixedSpelling):
        harvest.read(str(mixed))


def test_a_NAMED_SCENE_harvests_NOUNS():
    """The workshop's axe, bottle and plank are made with `thing()`, so its
    rows say "take hold of the axe" where a harvest world says "the iron",
    and no row carries a number for a thing."""
    rows = harvest.run(3, ticks=60, build=scenes.workshop_world)
    assert rows, "a smith among things decides"
    keys = {k for r in rows for k in r["options"]}
    for noun in ("the axe", "the bottle", "the plank"):
        assert f"take hold of {noun}" in keys, (noun, sorted(keys))
    assert "swing the axe" in keys, sorted(keys)
    for r in rows:
        assert r["spelling"] == seam.SPELLING
        assert "eid" not in r["text"]
        for k in r["options"]:
            assert not any(c.isdigit() for c in k), \
                f"an option names a thing by a number: {k!r}"
        held = [l for l in r["text"].split("\n") if l.startswith("holding: ")]
        assert held and not any(c.isdigit() for c in held[0]), held
