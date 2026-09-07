"""Small-model harness ports (SMALL_MODEL_HARNESS.md): validity-filtered menus,
input canonicalization, quoted-speech fast path, margin-gated selection.
"""

from src.core import engine
from src.core.composer import MockPicker
from src.core.seed import vault


# DESIGN (Jericho): an option that would change NOTHING is not an option.
# Sparking oil that's already igniting-hot is noise; the menu drops it.
def test_menu_drops_options_that_change_nothing():
    world, player = vault(seed=0)
    player.pos = (3, 0)
    tgt = (3, 1, 0)
    labels = [o.label for o in engine.affordance_menu(world, player)]
    assert any("ignite" in l for l in labels), "cold oil: ignite is real"
    world.cell(tgt).heat = 150.0                        # already past ignition — spark adds nothing
    labels = [o.label for o in engine.affordance_menu(world, player)]
    assert not any("ignite S" == l or ("ignite" in l and "S" in l.split()) for l in labels), \
        f"sparking already-igniting oil changes nothing: {labels}"


# DESIGN: the same dry-run guards free text — a no-op composes into a diagnostic
# refusal instead of a wasted turn.
def test_freetext_noop_refuses_with_reason():
    world, player = vault(seed=0)
    player.pos = (3, 0)
    world.cell((3, 1, 0)).heat = 150.0
    res = engine.free_text(world, player, "light the oil", MockPicker())
    line = " ".join(res["narrative"]).lower()
    assert "nothing" in line or "already" in line, f"say WHY it's pointless: {res['narrative']}"


# DESIGN (thadunge2/Clover): canonicalize before classifying — "I throw the
# knife at Sly" and "throw knife at sly" are the same utterance.
def test_first_person_input_is_canonicalized():
    from tests.test_interactions import _arm
    world, player = vault(seed=0)
    sly = world.entities["thief"]
    player.pos, sly.pos = (2, 2), (2, 0)
    player.props["finesse"] = 2.0
    _arm(world, player)
    engine.free_text(world, player, "I throw my knife at Sly!!", MockPicker())
    assert sly.props["hp"] < 10.0, "first-person phrasing must not confuse the composer"


# DESIGN: quoted input IS dialogue — no classification call needed. With one
# NPC in earshot, "..." goes straight to them.
def test_quoted_input_is_dialogue_by_construction():
    from src.core.state import Cell
    world, player = vault(seed=0)
    pip = world.entities["apprentice"]
    player.pos, pip.pos = (1, 1), (1, 2)
    for y in range(4):                                   # wall the east half off
        world.cells[(3, y)] = Cell(material="stone")
    world.entities["thief"].pos = (5, 0)
    world.entities["guard"].pos = (5, 2)
    res = engine.free_text(world, player, '"good evening, friend"', MockPicker())
    joined = " ".join(res["narrative"])
    assert "You:" in joined and "Pip" in joined, f"quoted speech reaches the one listener: {res['narrative']}"

    crowded, player2 = vault(seed=0)                     # everyone in earshot → ask
    player2.pos = (1, 1)
    res2 = engine.free_text(crowded, player2, '"good evening"', MockPicker())
    assert "whom" in " ".join(res2["narrative"]).lower(), "a crowded room needs an addressee"


# DESIGN (margin gate): when the scorer can't tell two victims apart, it must
# ASK, not coin-flip. Confidence is a number; use it.
def test_narrow_margin_on_violence_asks_back():
    class Waffler(MockPicker):
        def pick_scored(self, scene, utterance, question, options):
            return 0, [0.101, 0.100] + [0.0] * (len(options) - 2)

    from tests.test_interactions import _arm
    world, player = vault(seed=0)
    world.entities["thief"].pos = (2, 1)
    world.entities["apprentice"].pos = (3, 2)
    player.pos = (2, 2)
    _arm(world, player)
    res = engine.free_text(world, player, "throw the knife at them", Waffler())
    line = " ".join(res["narrative"]).lower()
    assert "?" in line, f"a coin-flip stabbing is not adjudication — ask: {res['narrative']}"
    assert all(e.props.get("hp", 10) == 10 for e in world.entities.values() if "person" in e.tags)