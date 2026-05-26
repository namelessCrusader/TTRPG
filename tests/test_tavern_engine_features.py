"""Tavern pack: Linna spells, object menu, per-verb narration."""

from src.sim.capability_menu import pack_capability_candidates
from src.sim.compiler import compile_action
from src.sim.narrator import render_event
from src.sim.schemas import (
    EntityId,
    Event,
    IntentBlock,
    ObjectId,
    SemanticAction,
)
from src.sim.world_loader import load_world_pack


def _entity(world, name: str):
    return next(e for e in world.spatial.entities.values() if e.name == name)


def test_linna_has_cast_spell_menu():
    world = load_world_pack("worlds/tavern")
    assert world.config.spell_vocab is not None
    linna = _entity(world, "Linna")
    assert "hearth_glow" in linna.inscribed_spells
    verbs = {a.verb for a, _, _ in pack_capability_candidates(linna, world)}
    assert "cast" in verbs


def test_mira_sees_object_affordances_when_near_bar():
    world = load_world_pack("worlds/tavern")
    mira = _entity(world, "Mira")
    labels = " ".join(lbl.lower() for _, lbl, _ in pack_capability_candidates(mira, world))
    assert "pour" in labels or "pour_drink" in labels
    # Object affordances depend on visibility; at least one fixture verb may appear.
    assert any(
        kw in labels
        for kw in ("light", "extinguish", "ale cask", "hearth", "mug")
    )


def test_gossip_narrative_from_pack_template():
    world = load_world_pack("worlds/tavern")
    mira = _entity(world, "Mira")
    tomas = _entity(world, "Tomas")
    action = SemanticAction(
        verb="gossip",
        actor=mira.entity_id,
        target=str(tomas.entity_id),
        intent=IntentBlock(manner="purposeful", desired_outcome=["gossip"]),
        raw_input="[test]",
    )
    result = compile_action(action, world)
    assert result.valid
    event = Event(tick=1, action=action, transitions=result.concrete_transitions)
    line = render_event(event, world)
    assert "rumour" in line.lower() or "story" in line.lower()


def test_pour_drink_on_mug_uses_object_narrative():
    world = load_world_pack("worlds/tavern")
    mira = _entity(world, "Mira")
    mug_id = None
    for oid, obj in world.spatial.objects.items():
        if obj.name == "ale mug":
            mug_id = oid
            break
    assert mug_id is not None
    action = SemanticAction(
        verb="pour_drink",
        actor=mira.entity_id,
        target=str(mug_id),
        intent=IntentBlock(manner="purposeful", desired_outcome=["pour_drink"]),
        raw_input="[test]",
    )
    result = compile_action(action, world)
    assert result.valid
    event = Event(tick=1, action=action, transitions=result.concrete_transitions)
    line = render_event(event, world)
    assert "ale" in line.lower() and "mug" in line.lower()
