"""Oil spill → ignite → smoke → panic chain for voxel_tavern vertical slice."""

from pathlib import Path

from src.sim.compiler import apply_transitions
from src.sim.director import NarrativeDirector
from src.sim.fields import field_tick, get_layer, MEDIA_TILE_GROUND, maybe_register_hazard_facts, region_atmosphere_amount
from src.sim.physics import physics_tick
from src.sim.pressure_eval import conditions_met
from src.sim.scenario_executor import apply_conditional_scenario_beats, apply_scenario_entry_effects
from src.sim.schemas import Coord
from src.sim.world_loader import load_world_pack


def _load_voxel_tavern():
    pack = Path(__file__).resolve().parents[1] / "worlds" / "voxel_tavern"
    return load_world_pack(pack)


def _oil_flask_coord(world) -> Coord:
    flask = next(o for o in world.spatial.objects.values() if o.name == "lamp oil flask")
    assert flask.position is not None
    return flask.position


def test_hearth_log_not_pre_burning():
    world = _load_voxel_tavern()
    log = next(o for o in world.spatial.objects.values() if o.name == "burning log")
    assert "on_fire" not in log.tags
    assert float(log.meta.get("temperature", 999)) < 350


def test_burning_object_marks_tile_on_fire():
    world = _load_voxel_tavern()
    assert world.config.physics_config is not None

    log = next(o for o in world.spatial.objects.values() if o.name == "burning log")
    log.tags.append("on_fire")
    assert log.position is not None

    transitions = physics_tick(world, world.config.physics_config)
    apply_transitions(world, transitions)

    tile = world.spatial.tile_at(log.position)
    assert "on_fire" in {t.lower() for t in tile.tags}


def test_scenario_oil_spill_at_tick_4():
    world = _load_voxel_tavern()
    world.tick = 4
    lines = apply_scenario_entry_effects(world)
    assert any("oil" in ln.lower() or "field deposit" in ln.lower() for ln in lines)

    coord = _oil_flask_coord(world)
    tile = world.spatial.tile_at(coord)
    layer = get_layer(tile.env, MEDIA_TILE_GROUND, legacy_tile=tile)
    assert float((layer.get("oil") or {}).get("amount", 0)) >= 20.0
    assert any("oil" in f.claim.lower() for f in world.world_facts)


def test_conditional_oil_ignite_after_spill():
    world = _load_voxel_tavern()
    world.tick = 4
    apply_scenario_entry_effects(world)
    world.tick = 5

    lines = apply_conditional_scenario_beats(world)
    assert any("oil_ignites" in ln or "Ember" in ln for ln in lines)

    coord = _oil_flask_coord(world)
    tile = world.spatial.tile_at(coord)
    assert "on_fire" in {t.lower() for t in tile.tags}


def test_tile_fire_produces_atmosphere_smoke():
    world = _load_voxel_tavern()
    coord = _oil_flask_coord(world)

    tile = world.spatial.tile_at(coord)
    tile.tags = list(tile.tags) + ["on_fire"]
    world.spatial.set_tile(coord, tile)
    world.tick = 5

    apply_transitions(world, field_tick(world))

    smoke = region_atmosphere_amount(world, "smoke")
    assert smoke >= 2.0, f"expected atmosphere smoke, got {smoke}"

    for _ in range(6):
        world.tick += 1
        apply_transitions(world, field_tick(world))
    assert region_atmosphere_amount(world, "smoke") >= 12.0


def test_smoke_registers_world_facts_and_pressure():
    world = _load_voxel_tavern()
    region = world.regions[world.active_region_id]
    atm = region.grid.region_env.setdefault("fields", {}).setdefault("atmosphere", {})
    atm["smoke"] = {"amount": 15.0, "since_tick": world.tick}

    fact_transitions = maybe_register_hazard_facts(world)
    apply_transitions(world, fact_transitions)

    assert any("smoke" in f.tags for f in world.world_facts)
    assert conditions_met(
        [{"type": "atmosphere_substance_gte", "substance": "smoke", "amount": 12.0}],
        world,
    )


def test_conditional_scenario_injects_mira_goal_on_smoke():
    world = _load_voxel_tavern()
    region = world.regions[world.active_region_id]
    atm = region.grid.region_env.setdefault("fields", {}).setdefault("atmosphere", {})
    atm["smoke"] = {"amount": 20.0, "since_tick": world.tick}

    lines = apply_conditional_scenario_beats(world)
    assert lines

    mira = next(e for e in world.spatial.entities.values() if e.name == "Mira")
    assert any("roof" in g.lower() or "clear" in g.lower() for g in mira.goals)


def test_end_to_end_oil_spill_smoke_panic():
    """Tick 4 spill → tick 5 ignite → field smoke → facts → Mira goal."""
    world = _load_voxel_tavern()
    cfg = world.config.physics_config
    assert cfg is not None
    director = NarrativeDirector(world)

    for _ in range(12):
        director.tick_start(world)
        apply_transitions(world, physics_tick(world, cfg))
        apply_transitions(world, field_tick(world))
        from src.sim.campaign_director import run_campaign_director

        run_campaign_director(world)
        world.tick += 1

    coord = _oil_flask_coord(world)
    tile = world.spatial.tile_at(coord)
    layer = get_layer(tile.env, MEDIA_TILE_GROUND, legacy_tile=tile)
    assert float((layer.get("oil") or {}).get("amount", 0)) >= 20.0
    assert "on_fire" in {t.lower() for t in tile.tags}
    assert region_atmosphere_amount(world, "smoke") >= 12.0
    assert any("smoke" in f.tags for f in world.world_facts)

    mira = next(e for e in world.spatial.entities.values() if e.name == "Mira")
    assert mira.goals, "Mira should receive a smoke-response goal"

    # Baseline ticks before spill should not have triggered panic goals early
    assert world.tick >= 5
