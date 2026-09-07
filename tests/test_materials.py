"""Item 2b — the MATERIAL TEMPLATE carries properties to everything made of it (DF's
USE_MATERIAL_TEMPLATE). `corrodible` lives on wood/metal in MATERIALS, not on objects:
an entity nobody hand-tagged corrodes because of what it is MADE of. And fire's aftermath
is load-bearing: charred things are brittle (the D7 ledger's `charred` debt, paid)."""
from src.core import engine, reactions
from src.core.seed import vault
from src.core.state import Entity, mat_tags


def test_material_template_supplies_corrodible():
    assert "corrodible" in mat_tags("wood") and "corrodible" in mat_tags("metal")
    assert "corrodible" not in mat_tags("stone") and "corrodible" not in mat_tags("flesh")

    w, _ = vault(seed=12)
    crate = Entity("crate", "plain crate", (3, 3, 0), material="wood",
                   tags={"item", "container"})          # NO corrodible hand-tag anywhere
    w.entities["crate"] = crate
    w.cell(crate.pos).fluids["acid"] = 200.0
    hp0 = crate.props.setdefault("hp", 4.0)
    reactions.tick(w)
    assert crate.props["hp"] < hp0, "acid eats an untagged wooden crate — the template is the tag"


def test_seed_containers_corrode_without_hand_tags():
    w, _ = vault(seed=12)
    cask = w.entities["cask_water"]
    assert "corrodible" not in cask.tags, "the hand-tag is gone from the seed"
    w.cell(cask.pos).fluids["acid"] = 200.0
    hp0 = cask.props["hp"]
    reactions.tick(w)
    assert cask.props["hp"] < hp0, "behaviour preserved: the material carries it now"


def test_charred_is_brittle_under_smash():
    w, p = vault(seed=12)
    cask = w.entities["cask_water"]
    cask.pos = (1, 0, 0)                                 # adjacent to the player
    smash = next(o for o in engine.affordance_menu(w, p) if "smash the cask" in o.label)
    plain = smash.effects[0].data["delta"]
    cask.tags.add("charred")
    smash = next(o for o in engine.affordance_menu(w, p) if "smash the cask" in o.label)
    brittle = smash.effects[0].data["delta"]
    assert brittle < plain, "fire's aftermath is load-bearing: charred smashes harder"
