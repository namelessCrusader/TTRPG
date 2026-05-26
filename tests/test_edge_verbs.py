"""First-class edge_add / edge_update verbs for LM-driven relationship changes."""

from pathlib import Path

from src.sim.compiler import compile_action
from src.sim.schemas import ActionType, EntityId, IntentBlock, SemanticAction, TransitionKind
from src.sim.world_loader import load_world_pack


def _tomas_id(world):
    return next(
        eid for eid, e in world.spatial.entities.items() if "Tomas" in e.name
    )


def _aldric_id(world):
    return next(
        eid for eid, e in world.spatial.entities.items() if "Aldric" in e.name
    )


def test_edge_add_creates_relational_edge():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")
    actor = _aldric_id(world)
    target = _tomas_id(world)
    action = SemanticAction(
        verb=ActionType.EDGE_ADD,
        actor=actor,
        target=target,
        intent=IntentBlock(manner="distrusts", desired_outcome=["weight:0.7"]),
    )
    result = compile_action(action, world)
    assert result.valid
    kinds = [t.kind for t in result.concrete_transitions]
    assert TransitionKind.EDGE_CREATED in kinds
    edge = next(t for t in result.concrete_transitions if t.kind == TransitionKind.EDGE_CREATED)
    assert edge.payload["source"] == actor
    assert edge.payload["target"] == target
    assert edge.payload["edge_kind"] == "distrusts"


def test_edge_update_applies_delta():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")
    actor = _aldric_id(world)
    target = _tomas_id(world)
    action = SemanticAction(
        verb="edge_update",
        actor=actor,
        target=target,
        intent=IntentBlock(manner="wary_of", desired_outcome=["delta:0.2"]),
    )
    result = compile_action(action, world)
    assert result.valid
    upd = next(t for t in result.concrete_transitions if t.kind == TransitionKind.EDGE_UPDATED)
    assert upd.payload["delta"] == 0.2
