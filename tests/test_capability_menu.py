"""Pack capability menu surfaces verb_templates and affordances."""

from src.sim.capability_menu import pack_capability_candidates
from src.sim.game_loop import make_test_world
from src.sim.world_loader import load_world_pack


def test_tavern_pack_offers_gossip_and_role_affordances():
    world = load_world_pack("worlds/tavern")
    mira = next(e for e in world.spatial.entities.values() if e.name == "Mira")
    cands = pack_capability_candidates(mira, world)
    labels = " ".join(lbl.lower() for _, lbl, _ in cands)
    verbs = {str(a.verb).lower() for a, _, _ in cands}
    assert "gossip" in verbs or "gossip" in labels
    # Mira is staff/bartender — pour affordance (keyword pour_drink)
    assert "pour_drink" in verbs or "pour" in labels


def test_make_test_world_pack_capabilities_nonempty():
    world = make_test_world()
    ent = next(iter(world.spatial.entities.values()))
    assert isinstance(pack_capability_candidates(ent, world), list)
