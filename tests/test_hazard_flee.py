"""NPC hazard flee — smoke/fire avoidance without hearth false positives."""

from pathlib import Path

from src.sim.compiler import apply_transitions, compile_action
from src.sim.fields import MEDIA_TILE_AIR, build_field_transition
from src.sim.game_loop import GameLoop
from src.sim.hazard_avoidance import (
    assess_hazards,
    entity_walk_coord,
    find_hazard_aware_path,
    pick_flee_coord,
    _hazard_distance,
)
from src.sim.lm_adapter import MockLMAdapter
from src.sim.npc_policy import ReactivePolicy
from src.sim.npc_reflection import run_npc_reflection
from src.sim.scenario_executor import apply_scenario_entry_effects
from src.sim.schemas import ActionType, Coord, IntentBlock, SemanticAction, TransitionKind
from src.sim.world_loader import load_world_pack


def _flood_smoke(world, amount: float = 45.0) -> None:
    grid = world.spatial
    flood_trs = []
    for y in range(grid.height):
        for x in range(grid.width):
            c = Coord(x=x, y=y, z=1)
            if not grid.is_in_bounds(c) or not grid.is_passable(c):
                continue
            flood_trs.append(build_field_transition(
                MEDIA_TILE_AIR, "smoke", amount,
                coord=c, operation="set",
                cause="test_flood", tick=world.tick,
                remaining_ticks=30,
            ))
    apply_transitions(world, flood_trs)
    region = world.regions[world.active_region_id]
    atm = region.grid.region_env.setdefault("fields", {}).setdefault("atmosphere", {})
    atm["smoke"] = {"amount": amount, "since_tick": world.tick}


def _nearest_exit(world, origin: Coord) -> Coord | None:
    from src.sim.hazard_avoidance import _exit_coords

    exits = _exit_coords(world.spatial, origin)
    if not exits:
        return None
    return min(exits, key=lambda e: origin.manhattan(e))


def test_controlled_hearth_not_dangerous_fire():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "tavern")
    mira = next(e for e in world.spatial.entities.values() if e.name == "Mira")
    run_npc_reflection(mira, world, MockLMAdapter())

    haz = assess_hazards(mira, world)
    assert not haz.should_flee

    policy = ReactivePolicy()
    verbs = [str(c[0].verb).lower().split(".")[-1] for c in policy.generate_candidates(mira, world)]
    assert "flee" not in verbs


def test_oil_fire_triggers_flee():
    import yaml
    pack_path = Path(__file__).resolve().parents[1] / "worlds" / "voxel_tavern"
    world = load_world_pack(pack_path)
    with open(pack_path / "scenarios" / "kitchen_fire_panic.yaml", encoding="utf-8") as sf:
        world.config.extra["scenario"] = yaml.safe_load(sf) or {}
    linna = next(e for e in world.spatial.entities.values() if e.name == "Linna")

    world.tick = 4
    apply_scenario_entry_effects(world)
    world.tick = 5
    from src.sim.scenario_executor import apply_conditional_scenario_beats
    apply_conditional_scenario_beats(world)

    coord = next(o.position for o in world.spatial.objects.values() if o.name == "lamp oil flask")
    assert coord is not None
    linna.position = Coord(x=coord.x + 1, y=coord.y, z=1)

    haz = assess_hazards(linna, world)
    assert haz.should_flee

    action = ReactivePolicy().decide(linna, world)
    assert action.verb == ActionType.FLEE

    result = compile_action(action, world)
    assert result.valid
    dest = result.concrete_transitions[0].payload["to"]
    assert dest["x"] != linna.position.x or dest["y"] != linna.position.y


def test_flee_steps_away_from_fire():
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "voxel_tavern")
    ent = next(e for e in world.spatial.entities.values() if e.name == "Linna")
    fire = Coord(x=10, y=10, z=ent.position.z)
    tile = world.spatial.tile_at(fire)
    tile.tags = list(tile.tags) + ["on_fire"]
    world.spatial.set_tile(fire, tile)

    ent.position = Coord(x=11, y=10, z=ent.position.z)
    dest = pick_flee_coord(ent, world)
    assert dest is not None
    assert _hazard_distance(dest, fire) >= _hazard_distance(ent.position, fire)


def test_smoke_panic_flee_in_autonomous_run():
    import yaml
    pack_path = Path(__file__).resolve().parents[1] / "worlds" / "voxel_tavern"
    world = load_world_pack(pack_path)
    with open(pack_path / "scenarios" / "kitchen_fire_panic.yaml", encoding="utf-8") as sf:
        world.config.extra["scenario"] = yaml.safe_load(sf) or {}
    loop = GameLoop(world, adapter=MockLMAdapter(), npcs_act_each_turn=True)
    flee_count = 0
    for _ in range(14):
        loop.autonomous_tick()
        for ev in world.event_log[-6:]:
            if str(ev.action.verb).lower().endswith("flee"):
                flee_count += 1
    assert flee_count >= 1, "expected at least one flee after oil fire escalation"


def test_can_path_through_smoke_to_exit():
    """Smoke adds cost but never blocks — NPCs can route toward doors."""
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "voxel_tavern")
    ent = next(e for e in world.spatial.entities.values() if e.name == "Linna")
    _flood_smoke(world, amount=50.0)

    exit_coord = _nearest_exit(world, ent.position)
    assert exit_coord is not None, "tavern pack should have door/exit tiles"

    path = find_hazard_aware_path(
        world.spatial, ent.position, exit_coord, world, max_steps=512,
    )
    assert path is not None, "hazard-aware path should reach exit through smoke"
    assert path[-1] == exit_coord


def test_move_through_fire_applies_burn():
    """Stepping onto a burning tile hurts but is allowed."""
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "voxel_tavern")
    ent = next(e for e in world.spatial.entities.values() if e.name == "Linna")
    fire = Coord(x=ent.position.x + 1, y=ent.position.y, z=ent.position.z)
    tile = world.spatial.tile_at(fire)
    tile.tags = list(tile.tags) + ["on_fire"]
    world.spatial.set_tile(fire, tile)

    move = SemanticAction(
        verb=ActionType.MOVE,
        actor=ent.entity_id,
        target=fire,
        intent=IntentBlock(rationale="dash through flames"),
    )
    result = compile_action(move, world)
    assert result.valid

    health_trs = [
        t for t in result.concrete_transitions
        if t.kind == TransitionKind.ENTITY_HEALTH_CHANGED
    ]
    assert health_trs, "walking through fire should apply burn damage"
    assert health_trs[0].payload.get("cause") == "walked_through_flames"


def test_flee_paths_toward_exit_through_smoke():
    """During smoke panic, flee picks a step along the exit route, not away."""
    world = load_world_pack(Path(__file__).resolve().parents[1] / "worlds" / "voxel_tavern")
    ent = next(e for e in world.spatial.entities.values() if e.name == "Linna")
    _flood_smoke(world, amount=50.0)

    haz = assess_hazards(ent, world)
    assert haz.should_flee
    assert haz.exit_coords

    exit_coord = min(haz.exit_coords, key=lambda e: ent.position.manhattan(e))
    walk = entity_walk_coord(ent, world.spatial)
    path = find_hazard_aware_path(
        world.spatial, walk, exit_coord, world, max_steps=512,
    )
    assert path

    dest = pick_flee_coord(ent, world)
    assert dest is not None
    assert dest == path[0], "flee should follow hazard-aware path toward exit"
    assert dest.manhattan(exit_coord) < walk.manhattan(exit_coord)
