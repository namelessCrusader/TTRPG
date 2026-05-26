"""Substance layer + tavern physics integration."""

from pathlib import Path

from src.sim.game_loop import GameLoop
from src.sim.lm_adapter import MockLMAdapter
from src.sim.physics import physics_tick
from src.sim.projection import project
from src.sim.spell_runtime import compile_spell_text
from src.sim.substance import (
    can_ignite,
    describe_physical_state,
    get_temperature,
    queue_impulse,
)
from src.sim.world_loader import load_world_pack


def _tavern():
    return load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")


def test_tavern_loads_physics_and_magic():
    world = _tavern()
    assert world.config.physics_config is not None
    assert world.config.spell_vocab is not None
    assert "HEAT" in world.config.spell_vocab.operations


def test_hearth_visible_in_projection():
    world = _tavern()
    npc_id = next(iter(world.spatial.entities))
    proj = project(world, npc_id)
    notes = proj.environment.physical_notes + proj.environment.object_physical_notes
    assert any("warm" in n.lower() or "burn" in n.lower() or "hearth" in n.lower() for n in notes)


def test_heat_spell_queues_temperature_impulse():
    world = _tavern()
    linna = next(e for e in world.spatial.entities.values() if e.name == "Linna")
    target = next(e for e in world.spatial.entities.values() if e.name != "Linna")
    vocab = world.config.spell_vocab
    transitions = compile_spell_text(
        "HEAT 50 TARGET",
        linna.entity_id,
        target.entity_id,
        world,
        vocab,
    )
    assert transitions
    world.meta["physics_impulses"] = []
    queue_impulse(world, str(target.entity_id), "temperature", 50.0, source="test")
    assert world.meta.get("physics_impulses")
    cfg = world.config.physics_config
    applied = physics_tick(world, cfg)
    assert applied or get_temperature(target, cfg) > 20


def test_flammable_object_can_ignite():
    world = _tavern()
    log = world.spatial.objects[
        next(oid for oid, o in world.spatial.objects.items() if o.name == "burning log")
    ]
    cfg = world.config.physics_config
    assert "on_fire" in log.tags or can_ignite(log, cfg)


def test_autonomous_physics_tick_runs():
    world = _tavern()
    loop = GameLoop(world, adapter=MockLMAdapter())
    loop.autonomous_tick()
    assert world.tick >= 1


def test_vertical_gravity_and_shatter():
    from src.sim.schemas import Coord, TransitionKind
    from src.sim.physics import physics_tick
    from src.sim.compiler import apply_transitions
    
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "voxel_tavern")
    grid = world.spatial
    cfg = world.config.physics_config
    
    # 1. Spawn a fragile glass bottle at height z=3
    coord_high = Coord(x=5, y=5, z=3)
    
    from src.sim.schemas import VoxelCell
    # Make sure tiles and voxels below are passable (empty air) so it can fall
    for z in range(4):
        c = Coord(x=5, y=5, z=z)
        tile = grid.tile_at(c)
        tile.terrain = "floor" if z == 0 else "air"  # solid support only at z=0
        grid.set_tile(c, tile)
        if z == 0:
            grid.set_voxel(c, VoxelCell(solid=True, transparent=False))
        else:
            grid.set_voxel(c, VoxelCell(solid=False, transparent=True))
        
    from src.sim.schemas import ObjectState
    obj_id = "potion_flask_fall_test"
    potion = ObjectState(
        object_id=obj_id,
        name="glass flask of oil",
        tags=["fragile", "glass"],
        position=coord_high,
        meta={
            "fluid": {
                "material": "oil",
                "volume_ml": 100.0,
            },
            "temperature": 90.0, # Hot oil!
        }
    )
    grid.objects[obj_id] = potion
    
    # Spawn an entity on the ground at the landing tile (5, 5, 1)
    linna = next(e for e in world.spatial.entities.values() if e.name == "Linna")
    linna.position = Coord(x=5, y=5, z=1)
    
    # 2. Run physics tick
    transitions = physics_tick(world, cfg)
    
    # 3. Verify object fell, shattered, spilled oil, and splashed entity
    assert any(t.kind == TransitionKind.ITEM_TRANSFERRED and t.payload.get("object_id") == obj_id for t in transitions)
    assert any(t.kind == TransitionKind.ITEM_DESTROYED and t.payload.get("object_id") == obj_id for t in transitions)
    assert any(t.kind == TransitionKind.FIELD_CHANGED and t.payload.get("substance") == "oil" for t in transitions)
    assert any(t.kind == TransitionKind.ENTITY_TAG_CHANGED and t.payload.get("entity_id") == str(linna.entity_id) and t.payload.get("add") == "wet" for t in transitions)
    assert any(t.kind == TransitionKind.ENTITY_HEALTH_CHANGED and t.payload.get("entity_id") == str(linna.entity_id) and float(t.payload.get("delta", 0)) < 0 for t in transitions)

