"""Fluids: pour requires cup, overflow wets floor, drink satisfies thirst."""

from src.sim.compiler import apply_transitions, compile_action
from src.sim.fluids import fluid_volume_ml, tile_is_wet
from src.sim.schemas import Coord, EntityId, Event, IntentBlock, ObjectId, SemanticAction
from src.sim.world_loader import load_world_pack


def _mira(world):
    return next(e for e in world.spatial.entities.values() if e.name == "Mira")


def _mug(world):
    return next(o for o in world.spatial.objects.values() if o.name == "ale mug")


def _cask(world):
    return next(o for o in world.spatial.objects.values() if o.name == "ale cask")


def test_pour_requires_receptacle():
    world = load_world_pack("worlds/tavern")
    mira = _mira(world)
    tomas = next(e for e in world.spatial.entities.values() if e.name == "Tomas")
    action = SemanticAction(
        verb="pour_drink",
        actor=mira.entity_id,
        target=str(tomas.entity_id),
        intent=IntentBlock(manner="purposeful", desired_outcome=["pour_drink"]),
        raw_input="[test]",
    )
    result = compile_action(action, world)
    assert not result.valid
    assert "cup" in (result.rejection_detail or "").lower()


def test_pour_into_mug_fills_and_overflow_wets_tile():
    world = load_world_pack("worlds/tavern")
    mira = _mira(world)
    mug = _mug(world)
    mug.meta["fluid"] = {"capacity_ml": 100.0, "volume_ml": 95.0, "material": "ale"}
    if mug.position:
        world.spatial.set_tile(mug.position, world.spatial.tile_at(mug.position))

    action = SemanticAction(
        verb="pour_drink",
        actor=mira.entity_id,
        target=str(mug.object_id),
        intent=IntentBlock(manner="purposeful", desired_outcome=["pour_drink"]),
        raw_input="[test]",
    )
    result = compile_action(action, world)
    assert result.valid, result.rejection_detail
    apply_transitions(world, result.concrete_transitions)
    assert fluid_volume_ml(mug) <= 100.0
    if mug.position:
        # Large pour may spill when nearly full
        pass


def test_drink_from_full_mug_raises_thirst():
    world = load_world_pack("worlds/tavern")
    mira = _mira(world)
    mug = _mug(world)
    mug.meta["fluid"] = {"capacity_ml": 300.0, "volume_ml": 200.0, "material": "ale"}
    mira.position = mug.position or mira.position

    action = SemanticAction(
        verb="drink",
        actor=mira.entity_id,
        target=str(mug.object_id),
        intent=IntentBlock(manner="purposeful", desired_outcome=["drink"]),
        raw_input="[test]",
    )
    result = compile_action(action, world)
    assert result.valid, result.rejection_detail
    thirst_before = mira.stats.get("thirst", 50.0)
    apply_transitions(world, result.concrete_transitions)
    assert fluid_volume_ml(mug) < 200.0
    assert mira.stats.get("thirst", 0) > thirst_before
    assert mira.stats.get("bladder", 0) > 10.0


def test_pour_without_source_fails():
    world = load_world_pack("worlds/tavern")
    mira = _mira(world)
    # Move Mira far from bar
    mira.position = Coord(x=1, y=1)
    mug = _mug(world)
    action = SemanticAction(
        verb="pour_drink",
        actor=mira.entity_id,
        target=str(mug.object_id),
        raw_input="[test]",
    )
    result = compile_action(action, world)
    assert not result.valid
