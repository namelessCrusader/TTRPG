"""Tests for Tier 1 + Tier 3 production features."""

from __future__ import annotations

from src.sim.combat_tactics import (
    cover_protection,
    initiative_rank,
    range_damage_multiplier,
    range_zone,
    refresh_combat_initiative,
)
from src.sim.compiler import compile_action, apply_transitions
from src.sim.game_loop import make_test_world
from src.sim.npc_goal_actions import goal_driven_candidates
from src.sim.npc_policy import ReactivePolicy
from src.sim.schemas import (
    ActionType,
    AlertnessLevel,
    EntityId,
    EntityKind,
    IntentBlock,
    ObjectId,
    ObjectState,
    SemanticAction,
    Transition,
    TransitionKind,
    WorldFact,
    WorldFactScope,
    coord_key,
)
from src.sim.economy import get_market_price, init_market, record_trade
from src.sim.world_adjudicator import register_world_fact


def test_generic_bribe_compiles_gold_transfer():
    world = make_test_world()
    player_id = next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.PLAYER
    )
    npc_id = next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.NPC
    )
    player = world.spatial.entities[player_id]
    player.stats["gold"] = 50.0

    action = SemanticAction(
        verb="bribe",
        actor=player_id,
        target=str(npc_id),
        raw_input="bribe the guard with 10 gold",
    )
    result = compile_action(action, world)
    assert result.valid
    kinds = {t.kind for t in result.concrete_transitions}
    assert TransitionKind.ENTITY_STAT_CHANGED in kinds


def test_generic_sabotage_marks_and_damages_object():
    world = make_test_world()
    player_id = next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.PLAYER
    )
    oid = ObjectId("obj_test_cask")
    world.spatial.objects[oid] = ObjectState(
        object_id=oid,
        name="cask",
        tags=["container"],
        durability=80,
        max_durability=100,
        position=world.spatial.entities[player_id].position,
    )
    action = SemanticAction(
        verb="sabotage",
        actor=player_id,
        target=str(oid),
        raw_input="sabotage the cask",
    )
    result = compile_action(action, world)
    apply_transitions(world, result.concrete_transitions)
    assert world.spatial.objects[oid].durability < 80


def test_item_durability_destroy_at_zero():
    world = make_test_world()
    player_id = next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.PLAYER
    )
    oid = ObjectId("obj_fragile")
    world.spatial.entities[player_id].inventory.append(oid)
    world.spatial.objects[oid] = ObjectState(
        object_id=oid,
        name="fragile blade",
        tags=["weapon"],
        durability=2,
        max_durability=100,
    )
    apply_transitions(world, [
        Transition(
            kind=TransitionKind.ITEM_DURABILITY_CHANGED,
            payload={"object_id": str(oid), "delta": -5, "cause": "test"},
        ),
    ])
    assert oid not in world.spatial.objects


def test_combat_cover_reduces_protection_bonus():
    world = make_test_world()
    grid = world.spatial
    player = next(e for e in world.spatial.entities.values() if e.kind == EntityKind.PLAYER)
    npc = next(e for e in world.spatial.entities.values() if e.kind == EntityKind.NPC)
    tile = grid.tile_at(npc.position)
    tile.tags = list(tile.tags) + ["cover"]
    grid.set_tile(npc.position, tile)
    bonus = cover_protection(grid, player.position, npc.position)
    assert bonus >= 2


def test_combat_initiative_orders_combatants():
    world = make_test_world()
    for ent in world.spatial.entities.values():
        ent.alertness = AlertnessLevel.COMBAT
        ent.meta["combat_target"] = "someone"
    refresh_combat_initiative(world)
    order = world.meta.get("combat_initiative", [])
    assert len(order) >= 2
    assert initiative_rank(order[0], world) == 0


def test_range_zones_scale_damage():
    assert range_zone(1) == "melee"
    assert range_damage_multiplier(1, "unarmed melee") == 1.0
    assert range_damage_multiplier(8, "ranged weapon") < range_damage_multiplier(4, "ranged weapon")
    assert range_damage_multiplier(2, "unarmed melee") == 0.0


def test_goal_driven_move_toward_mark():
    world = make_test_world()
    npc = next(e for e in world.spatial.entities.values() if e.kind == EntityKind.NPC)
    tile = world.spatial.tile_at(npc.position)
    tile.marks = ["blood on the floor"]
    world.spatial.set_tile(npc.position, tile)
    npc.goals = ["Inspect the fresh mark here: blood on the floor"]
    cands = goal_driven_candidates(npc, world)
    assert any(c[0].verb == ActionType.EXAMINE for c in cands)


def test_economy_trade_moves_price():
    world = make_test_world()
    init_market(world)
    before = get_market_price(world, "ale")
    record_trade(world, "ale", quantity=20.0)
    after = get_market_price(world, "ale")
    assert after != before


def test_compile_bribe_via_full_compiler():
    world = make_test_world()
    player_id = next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.PLAYER
    )
    npc_id = next(
        eid for eid, e in world.spatial.entities.items()
        if e.kind == EntityKind.NPC
    )
    world.spatial.entities[player_id].stats["gold"] = 100.0
    action = SemanticAction(
        verb="bribe",
        actor=player_id,
        target=str(npc_id),
        raw_input="bribe with 15 gold",
    )
    result = compile_action(action, world)
    assert result.valid
