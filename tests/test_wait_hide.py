"""Tests for WAIT and HIDE kernel compilers."""

from __future__ import annotations

from src.sim.compiler import compile_action
from src.sim.schemas import (
    ActionType,
    AlertnessLevel,
    Coord,
    EntityKind,
    EntityState,
    Region,
    SemanticAction,
    TransitionKind,
    WorldState,
    new_entity_id,
)
from src.sim.spatial import SpatialGrid, entities_visible_from


def _minimal_world() -> WorldState:
    grid = SpatialGrid(width=8, height=8)
    thief_id = new_entity_id()
    guard_id = new_entity_id()
    grid.entities[thief_id] = EntityState(
        entity_id=thief_id,
        name="Thief",
        kind=EntityKind.NPC,
        position=Coord(x=3, y=3),
        alertness=AlertnessLevel.HIGH,
        attributes={"stealth": 80, "perception": 40},
    )
    grid.entities[guard_id] = EntityState(
        entity_id=guard_id,
        name="Guard",
        kind=EntityKind.NPC,
        position=Coord(x=5, y=3),
        facing="west",
        attributes={"stealth": 30, "perception": 30},
    )
    return WorldState(
        name="test",
        regions={"region_main": Region(region_id="region_main", name="main", grid=grid)},
        rng_seed=42,
    )


def test_wait_recovers_fatigue_and_calms_alertness():
    world = _minimal_world()
    thief_id = next(iter(world.spatial.entities))
    action = SemanticAction(verb=ActionType.WAIT, actor=thief_id)
    result = compile_action(action, world)
    assert result.valid
    kinds = {t.kind for t in result.concrete_transitions}
    assert TransitionKind.NEED_CHANGED in kinds
    assert TransitionKind.ENTITY_ALERTNESS_CHANGED in kinds
    fatigue = next(
        t for t in result.concrete_transitions
        if t.kind == TransitionKind.NEED_CHANGED
        and t.payload.get("need") == "fatigue"
    )
    assert float(fatigue.payload["delta"]) > 0


def test_hide_sets_hidden_when_not_detected():
    world = _minimal_world()
    thief_id = next(
        eid for eid, e in world.spatial.entities.items() if e.name == "Thief"
    )
    guard = next(e for e in world.spatial.entities.values() if e.name == "Guard")
    guard.position = Coord(x=7, y=7)

    action = SemanticAction(verb=ActionType.HIDE, actor=thief_id)
    result = compile_action(action, world)
    assert result.valid
    hidden_t = next(
        t for t in result.concrete_transitions
        if t.kind == TransitionKind.ENTITY_PROPERTY_CHANGED
        and t.payload.get("prop") == "hidden"
    )
    assert hidden_t.payload["new_value"] is True


def test_hidden_entity_filtered_from_projection():
    world = _minimal_world()
    thief_id = next(
        eid for eid, e in world.spatial.entities.items() if e.name == "Thief"
    )
    guard_id = next(
        eid for eid, e in world.spatial.entities.items() if e.name == "Guard"
    )
    thief = world.spatial.entities[thief_id]
    thief.meta["hidden"] = True
    thief.meta["hidden_ticks"] = 0

    visible = entities_visible_from(world.spatial, guard_id)
    ids = {e.entity_id for e in visible}
    assert thief_id not in ids
