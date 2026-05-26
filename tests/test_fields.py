"""Generic field layer — emitters, media, substance deposits."""

from pathlib import Path

from src.sim.compiler import apply_transitions
from src.sim.fields import field_tick, load_substance_catalog
from src.sim.schemas import Coord
from src.sim.world_loader import load_world_pack


def test_substance_catalog_loads():
    cat = load_substance_catalog("worlds/default")
    assert cat.get("water") is not None
    assert cat.get("influenza") is not None
    assert len(cat.emitters) >= 1


def test_tile_reactions_fire_produces_smoke():
    pack = Path(__file__).resolve().parents[1] / "worlds" / "tavern"
    world = load_world_pack(pack)
    coord = Coord(x=2, y=2)
    tile = world.spatial.tile_at(coord)
    tile.tags = list(tile.tags) + ["on_fire"]
    world.spatial.set_tile(coord, tile)
    world.tick = 4
    apply_transitions(world, field_tick(world))
    tile = world.spatial.tile_at(coord)
    air_layer = (tile.env.get("fields") or {}).get("air") or {}
    assert "smoke" in air_layer


def test_wipe_clears_field_layer():
    from src.sim.compiler import compile_action
    from src.sim.schemas import SemanticAction, Transition, TransitionKind
    pack = Path(__file__).resolve().parents[1] / "worlds" / "tavern"
    world = load_world_pack(pack)
    actor = next(iter(world.spatial.entities.values()))
    coord = actor.position
    # Deposit blood on actor's tile via FIELD_CHANGED
    apply_transitions(world, [Transition(
        kind=TransitionKind.FIELD_CHANGED,
        payload={"medium": "tile.ground", "substance": "blood", "operation": "add",
                 "amount": 50.0, "x": coord.x, "y": coord.y, "cause": "test", "tick": 0},
    )])
    tile = world.spatial.tile_at(coord)
    assert "blood" in (tile.env.get("fields") or {}).get("ground", {})
    # Wipe using the interactions kernel
    from src.sim.schemas import IntentBlock
    action = SemanticAction(
        action_id="test_wipe", actor=actor.entity_id,
        verb="wipe", intent=IntentBlock(), target=None,
    )
    result = compile_action(action, world)
    assert result.valid, result.rejection_reason
    apply_transitions(world, result.concrete_transitions)
    tile = world.spatial.tile_at(coord)
    ground = (tile.env.get("fields") or {}).get("ground", {})
    assert "blood" not in ground


def test_rain_emitter_via_field_tick():
    pack = Path(__file__).resolve().parents[1] / "worlds" / "tavern"
    world = load_world_pack(pack)
    world.meta["weather"] = "rain"
    coord = Coord(x=2, y=2)
    tile = world.spatial.tile_at(coord)
    tile.tags = list(tile.tags) + ["mud"]
    world.spatial.set_tile(coord, tile)
    world.tick = 2
    apply_transitions(world, field_tick(world))
    tile = world.spatial.tile_at(coord)
    layer = (tile.env.get("fields") or {}).get("ground") or {}
    assert "rainwater" in layer or tile.env.get("wet")


# ─────────────────────────────────────────────────────────────────────────
# Cross-pack integration: the canonical field layer and the legacy mirror
# must stay in sync regardless of which writer set up the state.  These
# tests pin Phase 1/2 of the fields-storage migration.
# ─────────────────────────────────────────────────────────────────────────


def test_apply_wet_tile_writes_canonical_field_layer():
    """``fluids.apply_wet_tile`` is a direct call (e.g. from
    ``FLUID_CHANGED`` apply / overflow pour) — after Phase 1 it must
    populate ``tile.env["fields"]["ground"][material]`` so downstream
    consumers (pressure_eval, npc_planner, fields.field_amount) see the
    spill immediately.
    """
    from src.sim.fluids import apply_wet_tile

    pack = Path(__file__).resolve().parents[1] / "worlds" / "tavern"
    world = load_world_pack(pack)
    coord = Coord(x=2, y=2)
    apply_wet_tile(world.spatial, coord, "ale", 200.0)
    tile = world.spatial.tile_at(coord)

    ground = (tile.env.get("fields") or {}).get("ground") or {}
    assert "ale" in ground, f"canonical ground layer must contain ale; got {ground}"
    entry = ground["ale"]
    assert entry["meta"]["volume_ml"] == 200.0
    # Legacy mirror must still be written so back-compat code keeps working.
    assert (tile.env.get("fluid_spill") or {}).get("material") == "ale"
    assert tile.env.get("wet") is True
    assert "wet" in tile.tags


def test_field_amount_and_tile_smoke_read_both_storages():
    """The public ``field_amount`` and ``tile_smoke`` helpers should
    succeed against both the canonical and the legacy storage so
    Phase 4 (mirror deletion) can remove the legacy keys without
    breaking gameplay code that uses these helpers."""
    from src.sim.fields import (
        MEDIA_TILE_AIR,
        MEDIA_TILE_GROUND,
        field_amount,
        tile_smoke,
    )

    pack = Path(__file__).resolve().parents[1] / "worlds" / "tavern"
    world = load_world_pack(pack)
    coord = Coord(x=3, y=3)
    tile = world.spatial.tile_at(coord)
    tile.env["fields"] = {
        "ground": {"oil": {"amount": 45.0, "meta": {"volume_ml": 120.0}}},
        "air": {"smoke": {"amount": 30.0, "meta": {"remaining_ticks": 8}}},
    }
    world.spatial.set_tile(coord, tile)

    assert field_amount(world, coord, MEDIA_TILE_GROUND, "oil") == 45.0
    assert field_amount(world, coord, MEDIA_TILE_AIR, "smoke") == 30.0

    # Region atmosphere is independent — tile_smoke should take the max.
    region = world.regions[world.active_region_id]
    region.grid.region_env.setdefault("fields", {})["atmosphere"] = {
        "smoke": {"amount": 55.0, "since_tick": world.tick}
    }
    assert tile_smoke(world, coord) == 55.0

    # And it must also handle the case where the tile has nothing but
    # the region atmosphere has smoke (the typical reactive_only hazard
    # synthesis path).
    blank = Coord(x=5, y=5)
    assert field_amount(world, blank, MEDIA_TILE_AIR, "smoke") == 0.0
    assert tile_smoke(world, blank) == 55.0


def test_tile_dry_clears_canonical_layer_and_legacy_mirror():
    """``FLUID_CHANGED`` with ``target_kind=tile_dry`` should clear BOTH
    the canonical ground layer and the legacy ``fluid_spill`` / ``wet``
    mirror so the tile reads as clean from every code path."""
    from src.sim.fluids import apply_wet_tile, tile_is_wet
    from src.sim.schemas import Transition, TransitionKind

    pack = Path(__file__).resolve().parents[1] / "worlds" / "tavern"
    world = load_world_pack(pack)
    coord = Coord(x=4, y=2)
    apply_wet_tile(world.spatial, coord, "water", 75.0)
    assert tile_is_wet(world.spatial, coord)

    apply_transitions(world, [Transition(
        kind=TransitionKind.FLUID_CHANGED,
        payload={
            "target_kind": "tile_dry",
            "x": coord.x, "y": coord.y,
            "cause": "test_dry",
        },
    )])

    tile = world.spatial.tile_at(coord)
    assert not tile_is_wet(world.spatial, coord)
    ground = (tile.env.get("fields") or {}).get("ground") or {}
    assert "water" not in ground, f"water should be cleared from canonical layer; got {ground}"
    assert not tile.env.get("fluid_spill"), "legacy fluid_spill must be cleared"
    assert "wet" not in tile.tags

