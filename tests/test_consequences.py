"""Sticky world: actions leave transitions + interaction traces."""

from src.sim.compiler import apply_transitions, compile_action
from src.sim.consequences import has_world_stick
from src.sim.schemas import IntentBlock, SemanticAction
from src.sim.world_loader import load_world_pack


def test_toast_has_sticky_effects_and_trace():
    world = load_world_pack("worlds/tavern")
    mira = next(e for e in world.spatial.entities.values() if e.name == "Mira")
    tomas = next(e for e in world.spatial.entities.values() if e.name == "Tomas")
    action = SemanticAction(
        verb="toast",
        actor=mira.entity_id,
        target=str(tomas.entity_id),
        intent=IntentBlock(manner="purposeful", desired_outcome=["toast"]),
        raw_input="[test]",
    )
    result = compile_action(action, world)
    assert result.valid
    assert has_world_stick(result.concrete_transitions)
    apply_transitions(world, result.concrete_transitions)
    assert tomas.meta.get("interaction_traces")


def test_hollow_unknown_verb_gets_fallback_edge():
    world = load_world_pack("worlds/tavern")
    mira = next(e for e in world.spatial.entities.values() if e.name == "Mira")
    tomas = next(e for e in world.spatial.entities.values() if e.name == "Tomas")
    action = SemanticAction(
        verb="banter",
        actor=mira.entity_id,
        target=str(tomas.entity_id),
        raw_input="[test]",
    )
    result = compile_action(action, world)
    assert result.valid
    assert has_world_stick(result.concrete_transitions)
