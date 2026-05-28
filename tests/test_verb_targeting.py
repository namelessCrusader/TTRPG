"""Tests for dynamic verb target classification from verb_templates."""

from src.sim.verb_targeting import (
    action_missing_required_target,
    template_is_self_only,
    template_requires_entity_target,
    verb_requires_entity_target,
)
from src.sim.schemas import SemanticAction
from src.sim.world_loader import load_world_pack


def test_barter_requires_entity_target():
    world = load_world_pack("worlds/default")
    assert verb_requires_entity_target(world, "barter")
    tmpl = world.config.verb_templates["barter"]
    assert template_requires_entity_target(tmpl)


def test_disguise_is_self_only_not_entity_target():
    world = load_world_pack("worlds/default")
    tmpl = world.config.verb_templates["disguise"]
    assert template_is_self_only(tmpl)
    assert not verb_requires_entity_target(world, "disguise")


def test_action_missing_required_target_detects_barter_without_target():
    world = load_world_pack("worlds/default")
    action = SemanticAction(verb="barter", actor="ent_a", target=None)
    assert action_missing_required_target(action, world)


def test_tavern_pack_capability_barter_has_target():
    from src.sim.capability_menu import pack_capability_candidates

    world = load_world_pack("worlds/tavern")
    mira = next(e for e in world.spatial.entities.values() if e.name == "Mira")
    cands = pack_capability_candidates(mira, world)
    barter = [a for a, lbl, _ in cands if str(a.verb).lower() == "barter"]
    if barter:
        assert barter[0].target is not None
