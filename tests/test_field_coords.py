"""Field layer coordinate keys (x,y,z payloads and legacy x,y read)."""

from __future__ import annotations

from src.sim.compiler import apply_transitions
from src.sim.field_coords import (
    coord_key,
    field_tile_coord,
    parse_coord_key,
)
from src.sim.fields import build_field_transition, deposit_at
from src.sim.schemas import Coord
from src.sim.world_loader import load_world_pack
from pathlib import Path


def test_parse_legacy_and_3d_keys():
    assert parse_coord_key("4,10") == Coord(x=4, y=10, z=0)
    assert parse_coord_key("4,10,2") == Coord(x=4, y=10, z=2)
    assert coord_key(Coord(x=1, y=2, z=3)) == "1,2,3"


def test_build_field_transition_includes_z():
    t = build_field_transition(
        "tile.ground", "blood", 40.0,
        coord=Coord(x=5, y=6, z=2),
        tick=1,
    )
    assert t.payload["x"] == 5
    assert t.payload["y"] == 6
    assert t.payload["z"] == 2


def test_field_apply_roundtrip_at_z():
    pack = Path(__file__).resolve().parents[1] / "worlds" / "tavern"
    world = load_world_pack(pack)
    coord = Coord(x=3, y=4, z=0)
    apply_transitions(world, [
        deposit_at(coord, "tile.ground", "blood", 50.0, cause="test", tick=0),
    ])
    tile = world.spatial.tile_at(field_tile_coord(coord))
    assert "blood" in (tile.env.get("fields") or {}).get("ground", {})


def test_contamination_spread_uses_3d_pending_keys():
    from src.sim.contamination import contamination_tick

    pack = Path(__file__).resolve().parents[1] / "worlds" / "tavern"
    world = load_world_pack(pack)
    c = Coord(x=5, y=5, z=0)
    from src.sim.contamination import build_deposit_air

    apply_transitions(world, [
        build_deposit_air(c, "cough_droplets", 80.0, remaining_ticks=20, cause="test", tick=0),
    ])
    world.tick = 2
    transitions = contamination_tick(world)
    spread = [
        t for t in transitions
        if t.payload.get("operation") == "spread"
    ]
    if spread:
        assert "z" in spread[0].payload


def test_voxel_world_field_deposit_at_entity_z():
    pack = Path(__file__).resolve().parents[1] / "worlds" / "voxel_tavern"
    if not pack.is_dir():
        return
    world = load_world_pack(pack)
    ent = next(iter(world.spatial.entities.values()))
    assert ent.position.z >= 1
    apply_transitions(world, [
        deposit_at(ent.position, "tile.ground", "blood", 30.0, cause="test", tick=0),
    ])
    tile = world.spatial.tile_at(field_tile_coord(ent.position))
    assert "blood" in (tile.env.get("fields") or {}).get("ground", {})
